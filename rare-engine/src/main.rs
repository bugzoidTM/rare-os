use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

// ── Qwen2.5-1.5B Architecture Constants ─────────────────────────────────────
const N_LAYERS: usize = 28;
const N_HEADS: usize = 12;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const HIDDEN_DIM: usize = 1536;
const FFN_DIM: usize = 8960;
const ROPE_THETA: f64 = 1_000_000.0;
const IM_END_TOKEN: u32 = 151_645;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    /// Tokenizer JSON file (default alongside model)
    #[arg(long, default_value = "/tokenizer.json")]
    tokenizer: PathBuf,

    /// Prompt (smoke-test: "2+2=")
    #[arg(long, default_value = "2+2=")]
    prompt: String,

    /// Tokens to generate (keep at 2 for CI smoke-test)
    #[arg(long, default_value_t = 2)]
    max_tokens: usize,
}

// ── RMS Normalisation ────────────────────────────────────────────────────────
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let norm = (x.sqr()?.mean_keepdim(candle_core::D::Minus1)? + eps)?.sqrt()?;
    let x = x.broadcast_div(&norm)?;
    let x = x.to_dtype(dtype)?;
    Ok(x.broadcast_mul(weight)?)
}

// ── RoPE Rotation ────────────────────────────────────────────────────────────
fn apply_rope(q: &Tensor, k: &Tensor, pos: usize) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let dtype = q.dtype();
    let dim = HEAD_DIM;

    // Build cos/sin for this position
    let freqs: Vec<f32> = (0..dim / 2)
        .map(|i| {
            let theta = (ROPE_THETA as f32).powf(-(2.0 * i as f32) / dim as f32);
            (pos as f32) * theta
        })
        .collect();
    let cos: Vec<f32> = freqs.iter().map(|f| f.cos()).collect();
    let sin: Vec<f32> = freqs.iter().map(|f| f.sin()).collect();

    let rotate = |t: &Tensor| -> Result<Tensor> {
        // t shape: [1, n_heads, seq, head_dim]
        let t32 = t.to_dtype(DType::F32)?;
        let (b, h, s, d) = t32.dims4()?;
        let half = d / 2;
        let x1 = t32.narrow(3, 0, half)?;
        let x2 = t32.narrow(3, half, half)?;
        let c = Tensor::from_vec(cos.clone(), (1, 1, 1, half), device)?;
        let s_t = Tensor::from_vec(sin.clone(), (1, 1, 1, half), device)?;
        let rotated = Tensor::cat(&[(&x1 * &c)? - (&x2 * &s_t)?,
                                    (&x2 * &c)? + (&x1 * &s_t)?], 3)?;
        rotated.to_dtype(dtype)
    };

    Ok((rotate(q)?, rotate(k)?))
}

// ── Greedy Sampler ───────────────────────────────────────────────────────────
fn greedy(logits: &Tensor) -> Result<u32> {
    // logits: [vocab]
    let v = logits.to_vec1::<f32>()?;
    let (idx, _) = v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    Ok(idx as u32)
}

// ── Forward Pass ─────────────────────────────────────────────────────────────
fn forward(
    tokens: &[u32],
    reader: &mut std::fs::File,
    content: &gguf_file::Content,
    device: &Device,
    kv_cache: &mut Vec<(Tensor, Tensor)>,
) -> Result<Tensor> {
    let t = |name: &str| content.tensor(reader, name, device);

    // Embedding lookup
    let embd = t("token_embd.weight")?
        .dequantize(device)?;                          // [vocab, hidden]
    let seq = tokens.len();
    let ids: Vec<u32> = tokens.to_vec();
    let token_tensor = Tensor::from_vec(ids, (seq,), device)?;
    let mut x = embd.index_select(&token_tensor, 0)?  // [seq, hidden]
        .unsqueeze(0)?;                                // [1, seq, hidden]

    // Transformer layers
    for layer in 0..N_LAYERS {
        let ln = |suffix: &str| -> Result<Tensor> {
            let w = t(&format!("blk.{}.{}", layer, suffix))?
                .dequantize(device)?;
            rms_norm(&x, &w, 1e-6)
        };

        // Attention pre-norm
        let h = ln("attn_norm.weight")?;

        // QKV projections (dequantize quantized weights)
        let proj = |name: &str, x: &Tensor| -> Result<Tensor> {
            let w = t(&format!("blk.{}.{}", layer, name))?
                .dequantize(device)?
                .t()?;                       // [hidden, out_dim]
            x.broadcast_matmul(&w)           // [1, seq, out_dim]
        };

        let q = proj("attn_q.weight", &h)?;  // [1, seq, n_heads*head_dim]
        let k = proj("attn_k.weight", &h)?;  // [1, seq, n_kv*head_dim]
        let v = proj("attn_v.weight", &h)?;

        // Reshape to [1, n_heads, seq, head_dim]
        let reshape_heads = |t: &Tensor, n: usize| -> Result<Tensor> {
            let (b, s, _) = t.dims3()?;
            t.reshape((b, s, n, HEAD_DIM))?.transpose(1, 2)
        };
        let q = reshape_heads(&q, N_HEADS)?;
        let k = reshape_heads(&k, N_KV_HEADS)?;
        let v = reshape_heads(&v, N_KV_HEADS)?;

        // RoPE
        let pos = kv_cache.get(layer).map(|(k, _)| k.dim(2).unwrap_or(0)).unwrap_or(0);
        let (q, k) = apply_rope(&q, &k, pos)?;

        // KV cache concatenation
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

        // Grouped Query Attention: repeat KV heads
        let repeat = N_HEADS / N_KV_HEADS;
        let k = k.repeat((1, repeat, 1, 1))?;
        let v = v.repeat((1, repeat, 1, 1))?;

        // Scaled dot-product attention
        let scale = (HEAD_DIM as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn = candle_core::ops::softmax(&attn, candle_core::D::Minus1)?;
        let attn_out = attn.matmul(&v)?  // [1, n_heads, seq, head_dim]
            .transpose(1, 2)?            // [1, seq, n_heads, head_dim]
            .flatten_from(2)?;           // [1, seq, hidden]

        // Output projection + residual
        let w_o = t(&format!("blk.{}.attn_output.weight", layer))?
            .dequantize(device)?
            .t()?;
        x = (x + attn_out.broadcast_matmul(&w_o)?)?;

        // FFN with SwiGLU
        let h_ffn = ln("ffn_norm.weight")?;
        let gate = proj("ffn_gate.weight", &h_ffn)?;
        let up   = proj("ffn_up.weight",   &h_ffn)?;
        // SwiGLU: silu(gate) * up
        let silu = (gate.clone() / ((-gate)?.exp()? + 1.0)?)?;
        let act  = (silu * up)?;
        let w_d  = t(&format!("blk.{}.ffn_down.weight", layer))?
            .dequantize(device)?
            .t()?;
        x = (x + act.broadcast_matmul(&w_d)?)?;
    }

    // Final norm + LM head
    let norm_w = t("output_norm.weight")?.dequantize(device)?;
    let x = rms_norm(&x, &norm_w, 1e-6)?;

    // Take last token: [1, hidden]
    let last = x.i((0, x.dim(1)? - 1, ..))?;
    let lm_head = t("output.weight")?.dequantize(device)?.t()?;
    let logits = last.unsqueeze(0)?.broadcast_matmul(&lm_head)?; // [1, vocab]
    logits.i((0, ..))  // [vocab]
}

// ── Main ─────────────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    let args = Args::parse();
    println!("🧠 [RARE-Engine v0.3.0] Forward Pass Mínimo");

    let device = Device::Cpu;

    // Validate model file
    let meta = std::fs::metadata(&args.model)?;
    println!("📦 Modelo: {} MB", meta.len() / 1024 / 1024);

    // Load tokenizer
    println!("📝 Carregando tokenizer...");
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("{}", e))
        .context("Tokenizer não encontrado")?;

    // Encode prompt
    let encoded = tokenizer.encode(args.prompt.clone(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let mut tokens: Vec<u32> = encoded.get_ids().to_vec();
    println!("📝 Prompt '{}' → {} tokens: {:?}", args.prompt, tokens.len(), tokens);

    // Load GGUF metadata
    println!("⏳ Lendo metadados GGUF...");
    let mut file = std::fs::File::open(&args.model)?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    println!("🧩 Tensores: {}", content.tensor_infos.len());

    // Generation loop
    println!("\n🤖 Gerando {} token(s):\n", args.max_tokens);
    let mut kv_cache: Vec<(Tensor, Tensor)> = Vec::new();
    let gen_start = Instant::now();

    for step in 0..args.max_tokens {
        let t0 = Instant::now();

        // Forward: on first step send all tokens; subsequent steps send only last
        let ctx = if step == 0 { tokens.as_slice() } else { &tokens[tokens.len()-1..] };
        let logits = forward(ctx, &mut file, &content, &device, &mut kv_cache)
            .context("Forward pass falhou")?;

        let next_token = greedy(&logits)?;
        let ms = t0.elapsed().as_millis();

        // Decode single token
        let decoded = tokenizer.decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        print!("{}", decoded);
        std::io::stdout().flush()?;

        eprintln!(" [tok={} | {}ms | ~{:.1} tok/s]",
            next_token, ms, 1000.0 / ms as f64);

        tokens.push(next_token);
        if next_token == IM_END_TOKEN { break; }
    }

    let total_ms = gen_start.elapsed().as_millis();
    println!("\n\n✅ Concluído em {}ms | ~{:.2} tok/s médio",
        total_ms, args.max_tokens as f64 * 1000.0 / total_ms as f64);

    // Latency summary for documentation
    eprintln!("\n─── Latência Medida (QEMU/KVM) ───────────────────────");
    eprintln!("  Total: {}ms para {} tokens", total_ms, args.max_tokens);
    eprintln!("  Nota: QEMU ~3-8x mais lento que hardware real (sem KVM)");
    eprintln!("  Estimativa hardware real: {:.1}-{:.1} tok/s",
        args.max_tokens as f64 * 1000.0 / total_ms as f64 * 3.0,
        args.max_tokens as f64 * 1000.0 / total_ms as f64 * 8.0);

    Ok(())
}
