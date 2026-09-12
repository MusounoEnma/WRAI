import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
print("Loading Qwen...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
qwen = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True, local_files_only=True)
qwen.eval()

prompt = "<|im_start|>user\nhalo apa kabar?<|im_end|>\n<|im_start|>assistant\n"
inp = tok(prompt, return_tensors="pt")
inp_ids = inp["input_ids"]

with torch.no_grad():
    out_qwen = qwen(inp_ids)
    qwen_logits = out_qwen.logits

next_token_qwen = torch.argmax(qwen_logits[:, -1, :], dim=-1).item()
print(f"Qwen next token: {next_token_qwen} ('{tok.decode([next_token_qwen])}')")

# Now check what WRAI-X forward_parallel gives for layer 0 vs Qwen layer 0
print("\n--- Inspecting Layer 0 Attention vs RetNet ---")
qwen_l0 = qwen.model.layers[0]
x = qwen.model.embed_tokens(inp_ids)
x_norm = qwen_l0.input_layernorm(x)

# Qwen Attention
B, T, D = x.shape
H, HD = 16, 128
q = qwen_l0.self_attn.q_proj(x_norm).view(B, T, H, HD).transpose(1, 2)
k = qwen_l0.self_attn.k_proj(x_norm).view(B, T, 8, HD).repeat_interleave(2, dim=2).transpose(1, 2)
v = qwen_l0.self_attn.v_proj(x_norm).view(B, T, 8, HD).repeat_interleave(2, dim=2).transpose(1, 2)

# RoPE from Qwen
pos_ids = torch.arange(T).view(1, T)
cos, sin = qwen.model.rotary_emb(v, position_ids=pos_ids)
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

scale = 1.0 / math.sqrt(HD)
attn_weights = torch.matmul(q_rope, k_rope.transpose(-1, -2)) * scale
causal_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
attn_weights = attn_weights + causal_mask
attn_probs = F.softmax(attn_weights, dim=-1)
o_attn = torch.matmul(attn_probs, v)
o_attn_flat = o_attn.transpose(1, 2).contiguous().view(B, T, H * HD)
o_qwen = qwen_l0.self_attn.o_proj(o_attn_flat)

print(f"Qwen attn_probs sum per row: {attn_probs[0, 0, -1].sum().item():.4f}")
print(f"Qwen o_qwen norm: {o_qwen.norm().item():.4f}, std: {o_qwen.std().item():.4f}, mean: {o_qwen.mean().item():.4f}")

# Now WRAI-X Retention without Softmax:
# Notice: In Retention, decay_m = pow(gamma, dist)
# What is scores_m in WRAI-X?
gamma = 0.95
i_idx = torch.arange(T).view(T, 1)
j_idx = torch.arange(T).view(1, T)
dist = (i_idx - j_idx).clamp(min=0).view(1, 1, T, T)
causal = (i_idx >= j_idx).view(1, 1, T, T)
decay_m = torch.pow(gamma, dist) * causal

ret_weights = torch.matmul(q_rope, k_rope.transpose(-1, -2)) * scale * decay_m
o_ret = torch.matmul(ret_weights, v)

# Without Head-RMSNorm:
print(f"RetNet raw o_ret norm: {o_ret.norm().item():.4f}, std: {o_ret.std().item():.4f}")

# With Head-RMSNorm:
o_ret_norm = o_ret * torch.rsqrt(o_ret.pow(2).mean(-1, keepdim=True) + 1e-6)
print(f"RetNet with Head-RMSNorm norm: {o_ret_norm.norm().item():.4f}, std: {o_ret_norm.std().item():.4f}")

# And what is w_out * 0.38?
o_wrai = (qwen_l0.self_attn.o_proj(o_ret_norm.transpose(1, 2).contiguous().view(B, T, H * HD))) * 0.38
print(f"WRAI o_wrai norm: {o_wrai.norm().item():.4f}, std: {o_wrai.std().item():.4f}")

# Compare cosine similarity between o_qwen and o_wrai:
cos_sim = F.cosine_similarity(o_qwen.view(-1), o_wrai.view(-1), dim=0)
print(f"Cosine similarity between Qwen Attention and WRAI Retention output: {cos_sim.item():.4f}")
