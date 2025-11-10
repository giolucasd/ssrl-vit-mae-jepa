import time

import torch
import torch.nn as nn
import torch.optim as optim

print("==== Benchmark PyTorch + CUDA ====")

# --- Configurações básicas ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch versão: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("⚠️  CUDA não disponível. Abortando teste.")
    exit(1)

print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
print(f"Versão CUDA usada: {torch.version.cuda}")
print(f"Versão cuDNN: {torch.backends.cudnn.version()}")


# --- Rede neural simples ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 10),
        )

    def forward(self, x):
        return self.net(x)


# Instancia modelo e envia pra GPU
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Dados sintéticos ---
batch_size = 64
inputs = torch.randn(batch_size, 3, 64, 64, device=device)
targets = torch.randint(0, 10, (batch_size,), device=device)

# --- Benchmark ---
n_warmup = 5
n_iters = 20

print("\nRodando benchmark...")
torch.cuda.synchronize()

# Aquecimento
for _ in range(n_warmup):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()

# Benchmark principal
start = time.time()
for _ in range(n_iters):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"✅ Treinamento concluído em {elapsed:.3f} s para {n_iters} iterações")
print(f"Tempo médio por iteração: {elapsed / n_iters * 1000:.2f} ms")

# --- Teste de inferência pura ---
model.eval()
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(50):
        _ = model(inputs)
torch.cuda.synchronize()
inference_time = time.time() - start

print(f"✅ Inferência (50 forwards) em {inference_time:.3f} s")
print(f"Tempo médio por forward: {inference_time / 50 * 1000:.2f} ms")

print(f"\nMemória GPU alocada: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
print(f"Memória GPU reservada: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

print("\nTudo parece ok! 🚀")
