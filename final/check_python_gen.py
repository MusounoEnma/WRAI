import sys
import torch
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "final")
from test_wrai_x_06b_english import WRAIX06BModel, sample_token

print("=" * 70)
print("[*] Memuat Tokenizer Qwen...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

print("[*] Menginisialisasi Model WRAI-X 0.6B...")
model = WRAIX06BModel(vocab_size=151936, num_layers=28, hidden_dim=1024, ffn_dim=3072)

ckpt_path = "models x/wrai_x_06b_transplanted.pt"
print(f"[*] Memuat bobot dari {ckpt_path}...")
sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=False)
model.eval()

query = "halo"
prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
p_ids = tok.encode(prompt, add_special_tokens=False)

print(f"Prompt Tokens: {len(p_ids)} tokens")
print(f"Decoded Prompt: {tok.decode(p_ids)}")

states = None
with torch.no_grad():
    for tid in p_ids:
        logits, states = model.forward_step(torch.tensor([tid]), states)
    
    print("\nGeneration (Python PyTorch):")
    gen_tokens = []
    stop_ids = {tok.eos_token_id, 151643, 151645}
    for step in range(50):
        # argmax first to see top prediction
        top_k = torch.topk(logits[0], 5)
        if step == 0:
            print(f"Top 5 Next Tokens at step 0:")
            for val, idx in zip(top_k.values, top_k.indices):
                print(f"  Token {idx.item()}: '{tok.decode([idx.item()])}' (logit={val.item():.2f})")
                
        next_tok = sample_token(logits, gen_tokens, temperature=0.1, top_p=0.9, top_k=40, rep_penalty=1.1)
        if next_tok in stop_ids:
            break
        gen_tokens.append(next_tok)
        print(tok.decode([next_tok]), end="", flush=True)
        logits, states = model.forward_step(torch.tensor([next_tok]), states)
        
    print("\n" + "=" * 70)
