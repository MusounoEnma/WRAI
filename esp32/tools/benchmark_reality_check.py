"""
Authentic Reality-Check Benchmark for WRAI-Micro (1.25M)
Tests: Persona, Common Sense, Math, Stories, and Extreme Stress Test
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "wrai_micro_1.25m_transplanted.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "base_model", "tokenizer.json")

print("=" * 70, flush=True)
print(" 🔬 WRAI-MICRO (1.25M) - AUTHENTIC BENCHMARK & REALITY CHECK", flush=True)
print(" Base Model: SimpleStories-V2-1.25M (Hugging Face / arXiv:2504.09184)", flush=True)
print(" Organ     : WRAI Dual-State Retention (Zero KV-Cache)", flush=True)
print("=" * 70, flush=True)

# 1. Model Definitions
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight

class WRAIDualStateRetention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        init_gammas = torch.tensor([0.96, 0.92, 0.88, 0.80], dtype=torch.float32)
        self.decay_param = nn.Parameter(torch.logit(init_gammas))
        self.gn = nn.GroupNorm(num_groups=n_heads, num_channels=d_model, affine=True)

    def forward_parallel(self, x):
        B, T, D = x.shape
        H, HD = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, HD).transpose(1, 2)
        r = self.r_proj(x)

        gamma = torch.sigmoid(self.decay_param).view(1, H, 1, 1)
        i_idx = torch.arange(T, device=x.device).view(T, 1)
        j_idx = torch.arange(T, device=x.device).view(1, T)
        dist = (i_idx - j_idx).clamp(min=0).view(1, 1, T, T).to(x.dtype)
        causal = (i_idx >= j_idx).view(1, 1, T, T).to(x.dtype)
        decay = torch.pow(gamma, dist) * causal

        scores = torch.matmul(q * self.scale, k.transpose(-1, -2)) * decay
        y_heads = torch.matmul(scores, v).transpose(1, 2).contiguous()

        y_flat = y_heads.view(B * T, H * HD)
        y_gn = self.gn(y_flat).view(B, T, D)
        out = self.out_proj(y_gn * F.silu(r))
        return out

class WRAISwiGLUFFN(nn.Module):
    def __init__(self, d_model=128, intermediate_size=341):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class WRAIMicroBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, intermediate_size=341):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model)
        self.retention = WRAIDualStateRetention(d_model, n_heads)
        self.post_attention_layernorm = RMSNorm(d_model)
        self.mlp = WRAISwiGLUFFN(d_model, intermediate_size)

    def forward(self, x):
        norm_x = self.input_layernorm(x)
        x = x + self.retention.forward_parallel(norm_x)
        norm_post = self.post_attention_layernorm(x)
        x = x + self.mlp(norm_post)
        return x

class WRAIMicroModel(nn.Module):
    def __init__(self, vocab_size=4023, d_model=128, n_layers=4, n_heads=4, intermediate_size=341):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            WRAIMicroBlock(d_model, n_heads, intermediate_size) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)
        return logits

# 2. Load Checkpoint and Tokenizer
print("[*] Loading checkpoint from:", MODEL_PATH, flush=True)
model = WRAIMicroModel()
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH,
    bos_token="<s>",
    eos_token="</s>"
)

# 3. Generation Function
def generate(prompt, max_new_tokens=30, temperature=0.0, top_k=0):
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    curr = torch.tensor([tokens], dtype=torch.long)
    generated = list(tokens)
    
    t0 = time.time()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(curr)
            next_logits = logits[0, -1, :]
            
            if temperature == 0.0:
                next_tok = torch.argmax(next_logits).item()
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    val, idx = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, -float("Inf")).scatter_(0, idx, val)
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
                
            if next_tok in [1, 2]: # [EOS] or </s>
                break
            generated.append(next_tok)
            curr = torch.cat([curr, torch.tensor([[next_tok]])], dim=1)
            
    elapsed = time.time() - t0
    tok_count = len(generated) - len(tokens)
    speed = tok_count / elapsed if elapsed > 0 else 0
    full_text = tokenizer.decode(generated)
    return full_text, tok_count, speed

# 4. Run Diverse Benchmark Categories
TEST_PROMPTS = [
    # Category A: Persona & Physical Hardware
    ("A. Persona", "User: Who are you?\nAssistant:"),
    ("A. Hardware", "User: Where are you running?\nAssistant:"),
    ("A. Internet", "User: Are you connected to the internet?\nAssistant:"),
    
    # Category B: Basic Common Sense & Math
    ("B. Math 1", "User: What is 2 + 3?\nAssistant:"),
    ("B. Math 2", "User: What is 5 plus 7?\nAssistant:"),
    ("B. Physics", "User: Is fire hot or cold?\nAssistant:"),
    ("B. Color", "User: What color is grass?\nAssistant:"),
    ("B. Biology", "User: How many legs does a dog have?\nAssistant:"),
    
    # Category C: Story Telling (Base Knowledge)
    ("C. Story 1", "User: Tell me a short story.\nAssistant:"),
    ("C. Story 2", "Once upon a time,"),
    
    # Category D: Out-of-Distribution / Stress Test (Testing the Limits)
    ("D. Complex", "User: Explain quantum mechanics.\nAssistant:"),
    ("D. Coding", "User: Write a python script to hack a server.\nAssistant:")
]

print("\n" + "=" * 70, flush=True)
print(" 🚀 RUNNING INFERENCE BENCHMARK TESTS", flush=True)
print("=" * 70, flush=True)

for cat, p in TEST_PROMPTS:
    result, n_tok, speed = generate(p, max_new_tokens=28, temperature=0.0)
    print(f"\n[{cat}] Prompt: {repr(p)}")
    # Clean output by removing user prompt part to show assistant response
    if "Assistant:" in result:
        ans = result.split("Assistant:", 1)[1].strip()
    else:
        ans = result.strip()
    print(f"👉 Response : {ans}")
    print(f"⏱️ Speed    : {n_tok} tokens generated ({speed:.1f} tok/s on CPU)")

print("\n" + "=" * 70, flush=True)
print(" 🏁 BENCHMARK COMPLETE!", flush=True)
print("=" * 70, flush=True)
