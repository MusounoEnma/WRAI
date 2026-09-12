import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32, device_map="cpu", local_files_only=True)
qwen.eval()

prompt = "<|im_start|>user\nhalo apa kabar?<|im_end|>\n<|im_start|>assistant\n"
inp = tok(prompt, return_tensors="pt")
inp_ids = inp["input_ids"]

x = qwen.model.embed_tokens(inp_ids)
B, T, D = x.shape
H, HD = 16, 128
pos_ids = torch.arange(T).view(1, T)

for l in range(28):
    layer = qwen.model.layers[l]
    x_norm = layer.input_layernorm(x)
    
    q = layer.self_attn.q_proj(x_norm).view(B, T, H, HD).transpose(1, 2)
    k = layer.self_attn.k_proj(x_norm).view(B, T, 8, HD).repeat_interleave(2, dim=2).transpose(1, 2)
    v = layer.self_attn.v_proj(x_norm).view(B, T, 8, HD).repeat_interleave(2, dim=2).transpose(1, 2)
    
    scale_head = 1.0 / math.sqrt(HD)
    
    # ELU+1 Linear Attention Kernel (exactly like wrai_x_engine.c)
    phi_q = F.elu(q * scale_head) + 1.0
    phi_k = F.elu(k * scale_head) + 1.0
    
    gamma = 0.98
    i_idx = torch.arange(T).view(T, 1)
    j_idx = torch.arange(T).view(1, T)
    dist = (i_idx - j_idx).clamp(min=0).view(1, 1, T, T)
    causal = (i_idx >= j_idx).view(1, 1, T, T)
    decay = torch.pow(gamma, dist) * causal
    
    scores = torch.matmul(phi_q, phi_k.transpose(-1, -2)) * decay
    denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
    o_lin = torch.matmul(scores, v) / denom
    
    o_attn = layer.self_attn.o_proj(o_lin.transpose(1, 2).contiguous().view(B, T, H * HD))
    x = x + o_attn
    
    x_ffn_norm = layer.post_attention_layernorm(x)
    x = x + layer.mlp(x_ffn_norm)

x_final = qwen.model.norm(x)
logits = qwen.lm_head(x_final)

top5 = torch.topk(logits[0, -1, :], 5)
print("Top 5 tokens with C-Engine ELU+1 Normalized Kernel:")
for idx, prob in zip(top5.indices, top5.values):
    token_str = tok.decode([idx.item()]).encode("ascii", errors="replace").decode("ascii")
    print(f"  - ID {idx.item():6d} ('{token_str}') : logit {prob.item():.2f}")
