import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32, device_map="cpu", local_files_only=True)
qwen.eval()

prompt = "<|im_start|>user\nhalo apa kabar?<|im_end|>\n<|im_start|>assistant\n"
p_ids = tok.encode(prompt, add_special_tokens=False)

# Let's test autoregressive recurrent generation using the Normalized Linear Kernel:
B = 1
H = 16
HD = 128
scale_head = 1.0 / math.sqrt(HD)

# Initialize recurrent states for 28 layers:
# (state_m, state_zm)
states_m = [torch.zeros(B, H, HD, HD) for _ in range(28)]
states_zm = [torch.zeros(B, H, HD) for _ in range(28)]

gamma = 0.98

print("Feeding prompt tokens...")
for pos, tid in enumerate(p_ids):
    t_tensor = torch.tensor([[tid]])
    x = qwen.model.embed_tokens(t_tensor).squeeze(0) # (1, 1024)
    pos_ids = torch.tensor([[pos]])
    
    for l in range(28):
        layer = qwen.model.layers[l]
        x_norm = layer.input_layernorm(x)
        
        q = layer.self_attn.q_proj(x_norm).view(1, H, HD)
        k = layer.self_attn.k_proj(x_norm).view(1, 8, HD).repeat_interleave(2, dim=1)
        v = layer.self_attn.v_proj(x_norm).view(1, 8, HD).repeat_interleave(2, dim=1)
        
        cos, sin = qwen.model.rotary_emb(v, position_ids=pos_ids)
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        q_rope, k_rope = apply_rotary_pos_emb(q.unsqueeze(2), k.unsqueeze(2), cos, sin)
        q_rope = q_rope.squeeze(2)
        k_rope = k_rope.squeeze(2)
        
        phi_q = F.elu(q_rope * scale_head) + 1.0
        phi_k = F.elu(k_rope * scale_head) + 1.0
        
        # Update recurrent state
        states_m[l] = states_m[l] * gamma + torch.einsum('bhr,bhc->bhrc', phi_k, v)
        states_zm[l] = states_zm[l] * gamma + phi_k
        
        num = torch.einsum('bhr,bhrc->bhc', phi_q, states_m[l])
        denom = torch.einsum('bhr,bhr->bh', phi_q, states_zm[l]).unsqueeze(-1).clamp(min=1e-5)
        o_head_m = num / denom
        
        o_m = layer.self_attn.o_proj(o_head_m.reshape(1, H * HD))
        x = x + o_m
        
        x_ffn_norm = layer.post_attention_layernorm(x)
        x = x + layer.mlp(x_ffn_norm)

x_final = qwen.model.norm(x)
logits = qwen.lm_head(x_final)

top5 = torch.topk(logits[0, :], 5)
print("\nNext token candidates after prompt:")
for idx, prob in zip(top5.indices, top5.values):
    token_str = tok.decode([idx.item()]).encode("ascii", errors="replace").decode("ascii")
    print(f"  - ID {idx.item():6d} ('{token_str}') : logit {prob.item():.2f}")
