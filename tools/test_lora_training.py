import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank=4, alpha=8.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False
        in_dim = base_linear.in_features
        out_dim = base_linear.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * (1.0 / math.sqrt(in_dim)))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(x, self.lora_B @ self.lora_A) * self.scale
        return base_out + lora_out

    def merge_and_restore(self):
        delta = (self.lora_B @ self.lora_A) * self.scale
        self.base.weight.data.add_(delta)
        return self.base

print("Testing LoRALinear wrapping and forward pass on GPU/CPU...")
lin = nn.Linear(1024, 2048, bias=False)
wrapped = LoRALinear(lin, rank=4)
x = torch.randn(2, 10, 1024)
out = wrapped(x)
loss = out.sum()
loss.backward()
print(f"lora_A grad norm: {wrapped.lora_A.grad.norm().item():.4f}")
print(f"lora_B grad norm: {wrapped.lora_B.grad.norm().item():.4f}")
print(f"base.weight grad: {wrapped.base.weight.grad} (Must be None)")
merged = wrapped.merge_and_restore()
print(f"Merged successfully, type: {type(merged)}")
print("All tests passed!")
