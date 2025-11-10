import time

import torch

print("==== Teste PyTorch + CUDA ====")

# Verifica disponibilidade
print(f"PyTorch versão: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("⚠️  CUDA não está disponível. Verifique drivers e instalação do PyTorch.")
    exit(1)

# Mostra informações da GPU
device_name = torch.cuda.get_device_name(0)
print(f"GPU detectada: {device_name}")
print(f"Versão CUDA usada pelo PyTorch: {torch.version.cuda}")
print(f"Versão cuDNN: {torch.backends.cudnn.version()}")

# Teste de alocação e operação
device = torch.device("cuda")
a = torch.randn((5000, 5000), device=device)
b = torch.randn((5000, 5000), device=device)

torch.cuda.synchronize()
start = time.time()
c = torch.matmul(a, b)  # operação pesada na GPU
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"✅ Multiplicação de matrizes concluída com sucesso em {elapsed:.3f} s")
print(f"Resultado: média={c.mean().item():.5f}, desvio padrão={c.std().item():.5f}")

# Teste de transferência CPU ↔ GPU
a_cpu = a.cpu()
a_gpu = a_cpu.to(device)
print("✅ Transferência CPU <-> GPU ok")

print("Tudo parece estar funcionando corretamente! 🚀")
