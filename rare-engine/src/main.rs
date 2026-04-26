use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_core::quantized::{gguf_file, QMatMul};
use clap::Parser;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

// ── Qwen2.5-1.5B Architecture Constants ─────────────────────────────────────
const N_LAYERS: usize = 28;
const N_HEADS: usize = 12;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const ROPE_THETA: f64 = 1_000_000.0;
const IM_END_TOKEN: u32 = 151_645;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long, default_value = "/tokenizer.json")]
    tokenizer: PathBuf,

    #[arg(long, default_value = "2+2=")]
    prompt: String,

    #[arg(long, default_value_t = 2)]
    max_tokens: usize,
}

// ── Funções Auxiliares Matemáticas ──────────────────────────────────────────
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let norm = (x.sqr()?.mean_keepdim(2)? + eps)?.sqrt()?;
    let x = x.broadcast_div(&norm)?;
    let x = x.to_dtype(dtype)?;
    Ok(x.broadcast_mul(weight)?)
}

fn softmax(t: &Tensor, dim: usize) -> Result<Tensor> {
    let max = t.max_keepdim(dim)?;
    let exp = t.broadcast_sub(&max)?.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    Ok(exp.broadcast_div(&sum)?)
}

fn silu(t: &Tensor) -> Result<Tensor> {
    let neg = t.affine(-1.0, 0.0)?;
    let exp = neg.exp()?;
    let den = (exp + 1.0)?;
    Ok(t.broadcast_div(&den)?)
}

// ── RoPE Rotation ────────────────────────────────────────────────────────────
fn apply_rope(q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let dtype = q.dtype();
    let dim = HEAD_DIM;
    let seq_len = q.dim(2)?;

    let freqs: Vec<f32> = (0..dim / 2)
        .map(|i| (ROPE_THETA as f32).powf(-(2.0 * i as f32) / dim as f32))
        .collect();

    let mut cos = Vec::with_capacity(seq_len * (dim / 2));
    let mut sin = Vec::with_capacity(seq_len * (dim / 2));
    
    for pos in start_pos..start_pos + seq_len {
        for &freq in &freqs {
            let theta = (pos as f32) * freq;
            cos.push(theta.cos());
            sin.push(theta.sin());
        }
    }

    let rotate = |t: &Tensor| -> Result<Tensor> {
        let t32 = t.to_dtype(DType::F32)?;
        let (_, _, _, d) = t32.dims4()?;
        let half = d / 2;
        let x1 = t32.narrow(3, 0, half)?;
        let x2 = t32.narrow(3, half, half)?;
        
        let c = Tensor::from_vec(cos.clone(), (1, 1, seq_len, half), device)?;
        let s_t = Tensor::from_vec(sin.clone(), (1, 1, seq_len, half), device)?;
        
        let r1 = x1.broadcast_mul(&c)?.broadcast_sub(&x2.broadcast_mul(&s_t)?)?;
        let r2 = x2.broadcast_mul(&c)?.broadcast_add(&x1.broadcast_mul(&s_t)?)?;
        
        let rotated = Tensor::cat(&[&r1, &r2], 3)?;
        Ok(rotated.to_dtype(dtype)?)
    };

    Ok((rotate(q)?, rotate(k)?))
}

// ── Greedy Sampler ───────────────────────────────────────────────────────────
fn greedy(logits: &Tensor) -> Result<u32> {
    let v = logits.to_vec1::<f32>()?;
    let (idx, _) = v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    Ok(idx as u32)
}

// ── Forward Pass ─────────────────────────────────────────────────────────────
fn forward(
    tokens: &[u32],
    weights: &HashMap<String, Arc<candle_core::quantized::QTensor>>,
    device: &Device,
    kv_cache: &mut Vec<(Tensor, Tensor)>,
) -> Result<Tensor> {
    let t = |name: &str| -> Result<Arc<candle_core::quantized::QTensor>> {
        weights.get(name).cloned().context(format!("Tensor {} not found", name))
    };

    let embd = t("token_embd.weight")?.dequantize(device)?;
    let seq = tokens.len();
    let ids: Vec<u32> = tokens.to_vec();
    let token_tensor = Tensor::from_vec(ids, (seq,), device)?;
    let mut x = embd.index_select(&token_tensor, 0)?.unsqueeze(0)?;

    for layer in 0..N_LAYERS {
        let ln_weight_name = format!("blk.{}.attn_norm.weight", layer);
        let h_w = t(&ln_weight_name)?.dequantize(device)?;
        let h = rms_norm(&x, &h_w, 1e-6)?;

        let q_name = format!("blk.{}.attn_q.weight", layer);
        let w_q = QMatMul::from_arc(t(&q_name)?)?;
        let q = w_q.forward(&h)?;

        let k_name = format!("blk.{}.attn_k.weight", layer);
        let w_k = QMatMul::from_arc(t(&k_name)?)?;
        let k = w_k.forward(&h)?;

        let v_name = format!("blk.{}.attn_v.weight", layer);
        let w_v = QMatMul::from_arc(t(&v_name)?)?;
        let v = w_v.forward(&h)?;

        let reshape_heads = |tensor: &Tensor, n: usize| -> Result<Tensor> {
            let (b, s, _) = tensor.dims3()?;
            Ok(tensor.reshape((b, s, n, HEAD_DIM))?.transpose(1, 2)?)
        };
        
        let q = reshape_heads(&q, N_HEADS)?;
        let k = reshape_heads(&k, N_KV_HEADS)?;
        let v = reshape_heads(&v, N_KV_HEADS)?;

        let pos = kv_cache.get(layer).map(|(k_c, _)| k_c.dim(2).unwrap_or(0)).unwrap_or(0);
        let (q, k) = apply_rope(&q, &k, pos)?;

        let (k, v) = if let Some((kp, vp)) = kv_cache.get(layer) {
            (Tensor::cat(&[kp, &k], 2)?, Tensor::cat(&[vp, &v], 2)?)
        } else {
            (k, v)
        };
        
        if layer < kv_cache.len() {
            kv_cache[layer] = (k.clone(), v.clone());
        } else {
            kv_cache.push((k.clone(), v.clone()));
        }

        let repeat = N_HEADS / N_KV_HEADS;
        let repeat_kv = |t: &Tensor| -> Result<Tensor> {
            let (_, _, s, d) = t.dims4()?;
            Ok(t.unsqueeze(2)?
             .broadcast_as((1, N_KV_HEADS, repeat, s, d))?
             .reshape((1, N_HEADS, s, d))?)
        };
        
        let k = repeat_kv(&k)?;
        let v = repeat_kv(&v)?;

        let scale = (HEAD_DIM as f64).sqrt();
        let mut attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        
        // 🚀 OTIMIZAÇÃO/CORREÇÃO: Causal Mask para prompts multi-token
        let seq_len = q.dim(2)?;
        if seq_len > 1 {
            let mask: Vec<f32> = (0..seq_len).flat_map(|i| {
                (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
            }).collect();
            let mask_t = Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?;
            attn = attn.broadcast_add(&mask_t)?;
        }

        let attn = softmax(&attn, 3)?;
        let attn_out = attn.matmul(&v)?.transpose(1, 2)?.flatten_from(2)?;

        let o_name = format!("blk.{}.attn_output.weight", layer);
        let w_o = QMatMul::from_arc(t(&o_name)?)?;
        x = x.broadcast_add(&w_o.forward(&attn_out)?)?;

        let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer);
        let h_ffn_w = t(&ffn_norm_name)?.dequantize(device)?;
        let h_ffn = rms_norm(&x, &h_ffn_w, 1e-6)?;

        let gate_name = format!("blk.{}.ffn_gate.weight", layer);
        let w_gate = QMatMul::from_arc(t(&gate_name)?)?;
        let gate = w_gate.forward(&h_ffn)?;

        let up_name = format!("blk.{}.ffn_up.weight", layer);
        let w_up = QMatMul::from_arc(t(&up_name)?)?;
        let up = w_up.forward(&h_ffn)?;
        
        let act = silu(&gate)?.broadcast_mul(&up)?;

        let down_name = format!("blk.{}.ffn_down.weight", layer);
        let w_d = QMatMul::from_arc(t(&down_name)?)?;
        x = x.broadcast_add(&w_d.forward(&act)?)?;
    }

    let norm_w = t("output_norm.weight")?.dequantize(device)?;
    let x = rms_norm(&x, &norm_w, 1e-6)?;

    let seq_len = x.dim(1)?;
    let last = x.i((0, seq_len - 1, ..))?;
    
    let w_lm_head = QMatMul::from_arc(t("output.weight")?)?;
    let logits = w_lm_head.forward(&last.unsqueeze(0)?)?;
    Ok(logits.i((0, ..))?)
}

// ── Main ─────────────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    let args = Args::parse();
    println!("🧠 [RARE-Engine v0.4.0] Forward Pass com Cache RAM, Causal Mask e Prompt Template");

    let device = Device::Cpu;

    let meta = std::fs::metadata(&args.model)?;
    println!("📦 Modelo: {} MB", meta.len() / 1024 / 1024);

    println!("📝 Carregando tokenizer...");
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("{}", e))
        .context("Tokenizer não encontrado")?;

    // Formato de prompt opcional (se o usuário não passar as tags, mantemos puro para evitar prompts gigantes no QEMU)
    let formatted_prompt = args.prompt.clone();

    let encoded = tokenizer.encode(formatted_prompt.clone(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let mut tokens: Vec<u32> = encoded.get_ids().to_vec();
    println!("📝 Prompt formatado ({} tokens)", tokens.len());

    println!("⏳ Lendo metadados GGUF...");
    let mut file = std::fs::File::open(&args.model)?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    // Carregar todos os pesos em RAM uma única vez
    println!("⏳ Carregando pesos para a RAM (evitar I/O repetido)...");
    let mut weights: HashMap<String, Arc<candle_core::quantized::QTensor>> = HashMap::new();
    for name in content.tensor_infos.keys() {
        let qtensor = content.tensor(&mut file, name, &device)?;
        weights.insert(name.clone(), Arc::new(qtensor));
    }
    println!("🧩 Tensores carregados: {}", weights.len());

    println!("\n🤖 Gerando {} token(s):", args.max_tokens);
    let mut kv_cache: Vec<(Tensor, Tensor)> = Vec::new();
    let gen_start = Instant::now();

    for step in 0..args.max_tokens {
        let t0 = Instant::now();

        let ctx = if step == 0 { tokens.as_slice() } else { &tokens[tokens.len()-1..] };
        let logits = forward(ctx, &weights, &device, &mut kv_cache)
            .context("Forward pass falhou")?;

        let next_token = greedy(&logits)?;
        let ms = t0.elapsed().as_millis();

        let decoded = tokenizer.decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        print!("{}", decoded);
        std::io::stdout().flush()?;

        eprintln!(" [tok={} | {}ms | ~{:.1} tok/s]", next_token, ms, 1000.0 / ms as f64);

        tokens.push(next_token);
        if next_token == IM_END_TOKEN { break; }
    }

    let total_ms = gen_start.elapsed().as_millis();
    println!("\n\n✅ Engine concluída em {}ms.", total_ms);

    // Evitar Kernel Panic do Alpine quando PID 1 finaliza com sucesso
    std::process::exit(0);
}
