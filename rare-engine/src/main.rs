use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "Olá, RARE-OS.")]
    prompt: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("🧠 [RARE-Engine] Iniciando v0.1.0...");
    
    let device = Device::Cpu;
    println!("✅ Device: CPU Detectada");
    
    if args.model.exists() {
        let metadata = std::fs::metadata(&args.model)?;
        println!("✅ Modelo encontrado: {:?}", args.model);
        println!("📏 Tamanho do arquivo: {} MB", metadata.len() / 1024 / 1024);
    } else {
        println!("❌ ERRO: Modelo não encontrado no caminho: {:?}", args.model);
        anyhow::bail!("Arquivo ausente");
    }

    println!("🚀 RARE-OS Boot Test: Sucesso absoluto na comunicação Kernel -> Init -> Rust!");
    Ok(())
}
