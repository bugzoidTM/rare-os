# Rare OS

Rare OS é um sistema operacional mínimo baseado em Linux, focado em integrar um ambiente customizado com um runtime Rust estático para executar o modelo Qwen2.5-1.5B.

## Estrutura de Diretórios

- `kernel/` - Código-fonte e `.config` do Linux
- `initramfs/` - Sistema mínimo (BusyBox + libs + init)
- `bootloader/` - Configurações do GRUB2 e módulos
- `scripts/` - Scripts para build, empacotamento, flash e testes
- `rare-engine/` - Binário Rust estático (Qwen2.5-1.5B runtime)
- `iso_root/` - Estrutura raiz para a geração da ISO/USB bootável

## Checklist de Build (Abordagem SO Dedicado / Imutável)

- [x] Obter o binário do Kernel Linux pré-compilado (`kernel/bzImage` - ex: Alpine LTS)
- [x] Obter o binário do BusyBox estático (`initramfs/busybox`)
- [x] Criar o script de inicialização principal (`initramfs/init`)
- [ ] Compilar a `rare-engine` (Rust target: `x86_64-unknown-linux-musl`) e adicionar ao initramfs
- [ ] Empacotar a estrutura do `initramfs/` em um arquivo `.cpio.gz`
- [ ] Configurar o GRUB2 (`bootloader/grub.cfg`)
- [ ] Gerar a ISO bootável (`grub-mkrescue` via container ou pipeline)
- [ ] Testar a ISO no QEMU

## Dependências do Sistema

Para compilar todo o projeto, as seguintes dependências devem ser instaladas no seu ambiente de build (Debian/Ubuntu):

```bash
sudo apt update
sudo apt install build-essential flex bison libssl-dev libelf-dev cpio grub2 qemu-system-x86
```

## Roadmap e Logs

### Fase 1: Kernel e Boot
- Definição do sistema de compilação e infraestrutura mínima.
- (A fazer)

### Fase 2: Initramfs e Userland
- Compilação estática do BusyBox e script de `/init`.
- (A fazer)

### Fase 3: Rare-Engine (Rust AI Runtime)
- Implementar e compilar a integração do modelo Qwen2.5-1.5B através da engine Rust.
- (A fazer)

### Log de Progresso
*Espaço para adicionar logs importantes de compilação ou de decisões de arquitetura e bugs encontrados durante o desenvolvimento.*
