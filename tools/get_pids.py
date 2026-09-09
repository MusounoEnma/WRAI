import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
with open("models/pruned_vocab_map_v14_4.json", "r", encoding="utf-8") as f:
    vmap = json.load(f)

t2p = {int(k): int(v) for k, v in vmap["teacher_id_to_pruned_id"].items()}

prompts = [
    "<ID> Halo, jelaskan apa fungsi dari AI",
    "<EN> Artificial Intelligence is defined as",
    "<PY> def calculate_area(radius):"
]

for p in prompts:
    tids = tok(p)["input_ids"]
    pids = [t2p.get(t, 0) for t in tids]
    print(f"PROMPT: {p}")
    print(f"PIDS: {pids}")
