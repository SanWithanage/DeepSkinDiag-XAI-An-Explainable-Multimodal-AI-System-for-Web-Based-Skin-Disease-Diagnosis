import torch, os
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_default_device(device)
print("Default device set to:", device)
# quick tensor test
x = torch.randn(1000, 1000)
y = x @ x.T
print("Compute OK on:", y.device)
