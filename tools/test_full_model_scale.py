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

# Let's test what happens if we replace Attention in Qwen with Retention:
# For each layer, we test scale = 0.38 vs scale = 0.116 vs scale = 0.10
for test_scale in [0.38, 0.116, 0.08]:
    # Clone Qwen
    import copy
    print(f"\n=================== TESTING SCALE: {test_scale} ===================")
    # We can test by running forward pass where self_attn output is replaced
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
        
        cos, sin = qwen.model.rotary_emb(v, position_ids=pos_ids)
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Retention
        gamma = 0.95
        i_idx = torch.arange(T).view(T, 1)
        j_idx = torch.arange(T).view(1, T)
        dist = (i_idx - j_idx).clamp(min=0).view(1, 1, T, T)
        causal = (i_idx >= j_idx).view(1, 1, T, T)
        decay = torch.pow(gamma, dist) * causal
        
        scale_head = 1.0 / math.sqrt(HD)
        scores = torch.matmul(q_rope, k_rope.transpose(-1, -2)) * scale_head * decay
        o_ret = torch.matmul(scores, v)
        o_ret_norm = o_ret * torch.rsqrt(o_ret.pow(2).mean(-1, keepdim=True) + 1e-6)
        
        o_attn = layer.self_attn.o_proj(o_ret_norm.transpose(1, 2).contiguous().view(B, T, H * HD)) * test_scale
        x = x + o_attn
        
        # FFN
        x_ffn_norm = layer.post_attention_layernorm(x)
        x = x + layer.mlp(x_ffn_norm)
    
    x_final = qwen.model.norm(x)
    logits = qwen.lm_head(x_final)
    
    top5 = torch.topk(logits[0, -1, :], 5)
    print(f"Top 5 tokens for scale {test_scale}:")
    for idx, prob in zip(top5.indices, top5.values):
        print(f"  - ID {idx.item():6d} ('{tok.decode([idx.item()])}') : logit {prob.item():.2f}")
