import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.8B", local_files_only=True)
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.8B", torch_dtype=torch.float32, device_map="cpu", local_files_only=True)
qwen.eval()

prompt = "<|im_start|>user\nhalo apa kabar?<|im_end|>\n<|im_start|>assistant\n"
inp = tok(prompt, return_tensors="pt")
inp_ids = inp["input_ids"]

with torch.no_grad():
    qwen_out = qwen(inp_ids, output_hidden_states=True)
    qwen_hidden = qwen_out.hidden_states # 29 states (embed + 28 layers)

print("Layer-by-layer Cosine Similarity of hidden state x between True Qwen and Linear Attention:")

B, T = inp_ids.shape
H, HD = 16, 128
scale_head = 1.0 / math.sqrt(HD)
pos_ids = torch.arange(T).view(1, T)

x = qwen.model.embed_tokens(inp_ids)

for l in range(28):
    layer = qwen.model.layers[l]
    x_norm = layer.input_layernorm(x)
    
    q = layer.self_attn.q_proj(x_norm).view(B, T, H, HD).transpose(1, 2)
    k = layer.self_attn.k_proj(x_norm).view(B, T, 8, HD).repeat_interleave(2, dim=2).transpose(1, 2)
    v = layer.self_attn.v_proj(x_norm).view(B, T, 8, HD).repeat_interleave(2, dim=2).transpose(1, 2)
    
    cos, sin = qwen.model.rotary_emb(v, position_ids=pos_ids)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    
    # 1. Compare True Attention output vs Linear Attention output for this layer
    # True Attention
    attn_weights = torch.matmul(q_rope, k_rope.transpose(-1, -2)) * scale_head
    causal_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
    true_attn_probs = F.softmax(attn_weights + causal_mask, dim=-1)
    true_o = torch.matmul(true_attn_probs, v)
    true_attn_out = layer.self_attn.o_proj(true_o.transpose(1, 2).contiguous().view(B, T, H * HD))
    
    # Linear Attention (ELU+1)
    phi_q = F.elu(q_rope * scale_head) + 1.0
    phi_k = F.elu(k_rope * scale_head) + 1.0
    scores = torch.matmul(phi_q, phi_k.transpose(-1, -2))
    causal = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    scores = scores * causal
    denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
    lin_o = torch.matmul(scores, v) / denom
    lin_attn_out = layer.self_attn.o_proj(lin_o.transpose(1, 2).contiguous().view(B, T, H * HD))
    
    sim_attn = F.cosine_similarity(true_attn_out.view(-1), lin_attn_out.view(-1), dim=0).item()
    
    x = x + lin_attn_out
    x_ffn_norm = layer.post_attention_layernorm(x)
    x = x + layer.mlp(x_ffn_norm)
    
    true_x_layer = qwen_hidden[l+1]
    sim_x = F.cosine_similarity(true_x_layer.view(-1), x.view(-1), dim=0).item()
    
    if l in [0, 1, 2, 3, 4, 9, 14, 19, 27]:
        print(f"Layer {l:2d}: Attn CosSim = {sim_attn:.4f} | Hidden State CosSim = {sim_x:.4f} | x_norm = {x.norm().item():.1f} (True: {true_x_layer.norm().item():.1f})")
