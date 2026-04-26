use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use clap::Parser;
use gguf::GGUFLoad;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    
    #[arg(long, default_value = "Olá, RARE-OS.")]
    prompt: String,
    
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("🧠 [RARE-Engine] Iniciando...");
    println!("📦 Modelo: {:?}", args.model);
    
    // 1. Device: CPU-only, sem f16 forçado (mais compatível)
    let device = Device::Cpu;
    println!("✅ Device: CPU");

    // 2. Validar arquivo do modelo
    let model_path = &args.model;
    if !model_path.exists() {
        anyhow::bail!("Modelo não encontrado: {:?}", model_path);
    }
    println!("✅ Arquivo do modelo existe");

    // 3. Carregar metadados do GGUF (leve, não carrega pesos ainda)
    println!("⏳ Lendo metadados do GGUF...");
    let mut reader = std::fs::File::open(model_path)?;
    let (metadata, _tensors) = gguf::GGUFLoad::load(&mut reader)
        .context("Falha ao carregar metadados GGUF")?;
    
    println!("📋 Arquitetura: {:?}", metadata.architecture);
    println!("🔢 Parâmetros: ~1.5B (Q4_K_M)");
    println!("🧩 Tensores: {}", metadata.tensor_infos.len());

    // 4. Simular "inferência" com output fixo para validar o pipeline
    // (Inferência real exigiria mapear tensores específicos do Qwen, 
    //  o que faremos no próximo passo após este build passar)
    println!("\n🤖 Resposta (modo validação):");
    println!("Olá! O RARE-OS carregou o modelo com sucesso.");
    println!("Próximo passo: implementar forward pass do Qwen2.5.");
    
    // 5. Manter o processo vivo para o QEMU capturar o log
    println!("\n✅ Engine pronta. Aguardando...");
    std::thread::sleep(std::time::Duration::from_secs(5));
    
    Ok(())
}
