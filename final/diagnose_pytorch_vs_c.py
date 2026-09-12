import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "final")
from test_wrai_x_08b_english import WRAIX06BModel, sample_token

print("=" * 80)
print(" 🔬 PYTORCH CHECKPOINT DIAGNOSTIC: TESTING GENERATION ON CPU")
print("=" * 80)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.8B", local_files_only=True)
print("[OK] Tokenizer loaded offline from cache.")

model = WRAIX06BModel(vocab_size=151936, num_layers=28, hidden_dim=1024, ffn_dim=3072)
ckpt_path = "models x/wrai_x_08b_transplanted.pt"
print(f"[*] Loading weights from {ckpt_path}...")
sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

# Weight tying check
if "output_proj.weight" not in sd and "embed.weight" in sd:
    sd["output_proj.weight"] = sd["embed.weight"]
for l in range(28):
    for w_name in ["w_qr", "w_kr", "w_vr", "w_out_r"]:
        base_name = w_name.replace("r", "") if w_name != "w_qr" else "w_q"
        if w_name == "w_out_r": base_name = "w_out"
        k_r = f"layers.{l}.{w_name}.weight"
        k_b = f"layers.{l}.{base_name}.weight"
        if k_r not in sd and k_b in sd:
            sd[k_r] = sd[k_b]

model.load_state_dict(sd, strict=False)
model.eval()
print("[OK] PyTorch model loaded 100%.\n")

def test_prompt(query):
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    p_ids = tok.encode(prompt, add_special_tokens=False)
    print("=" * 80)
    print(f"PROMPT: '{query}' ({len(p_ids)} tokens)")
    print(f"Tokens: {p_ids}")
    
    states = None
    with torch.no_grad():
        for tid in p_ids:
            logits, states = model.forward_step(torch.tensor([tid]), states)
            
        print("\nTop 5 tokens predicted right after prompt:")
        top5 = torch.topk(logits[0], 5)
        for i in range(5):
            idx = top5.indices[i].item()
            val = top5.values[i].item()
            txt = tok.decode([idx])
            print(f"  #{i+1}: ID {idx:<6} ('{txt}') -> Logit: {val:.3f}")
            
        print("\nGreedy / Top-P Generation (first 40 tokens):")
        print("Model: ", end="", flush=True)
        gen = []
        for step in range(40):
            # Deterministic greedy (temperature=0.0)
            next_tok = torch.argmax(logits, dim=-1).item()
            if next_tok in [tok.eos_token_id, 151643, 151645]:
                print(f" [STOP: {next_tok}]")
                break
            gen.append(next_tok)
            word = tok.decode([next_tok])
            print(word, end="", flush=True)
            logits, states = model.forward_step(torch.tensor([next_tok]), states)
        print("\n")

test_prompt("halo apa kabar?")
test_prompt("Siapa kamu?")
