from transformers import AutoTokenizer, AutoConfig

model_name = "Qwen/Qwen3-0.8B"
print(f"[*] Validating Tokenizer & Config for: {model_name}...")
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

print(f"[OK] Vocab size        : {len(tok)}")
print(f"[OK] Hidden Size       : {cfg.hidden_size}")
print(f"[OK] Intermediate Size : {cfg.intermediate_size}")
print(f"[OK] Num Layers        : {cfg.num_hidden_layers}")
print(f"[OK] Num Heads (Q/KV)  : {cfg.num_attention_heads} / {cfg.num_key_value_heads}")
print(f"[OK] Head Dim          : {cfg.head_dim}")

im_start = tok.encode("<|im_start|>", add_special_tokens=False)
im_end = tok.encode("<|im_end|>", add_special_tokens=False)
print(f"[OK] Special tokens    : im_start={im_start}, im_end={im_end}")

test_text = "<|im_start|>user\nhalo apa kabar?<|im_end|>\n<|im_start|>assistant\n"
encoded = tok.encode(test_text, add_special_tokens=False)
print(f"[OK] Encoded test chat : {len(encoded)} tokens -> {encoded}")
decoded = tok.decode(encoded)
print(f"[OK] Round-trip decode : {repr(decoded)}")
