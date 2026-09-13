#!/usr/bin/env python3
"""
=============================================================================
 WRAI v14.7 — STANDALONE VOCABULARY MAP GENERATOR & DRIVE SYNC
=============================================================================
 Scans identical multi-domain corpus (Alpaca ID, Alpaca EN,
 OPUS Translation, Hermes Function Calling, WRAI Swarm Persona) and
 generates vocab_map_v14_7_transplant.json with 100% precision,
 directly saving to Google Drive!
=============================================================================
"""

import os
import sys
import time
import json
import random
import shutil
from collections import Counter
import torch

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    os.system("pip install -q datasets transformers huggingface_hub")
    from datasets import load_dataset
    from transformers import AutoTokenizer

TEACHER_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ROUTE_TAGS = ["<ID>", "<EN>"]
VOCAB_CAP = 32000
MAX_CHARS_RESPONSE = 800
SHUFFLE_BUFFER_SIZE = 2500

DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_Models_Transplant"
DRIVE_SAVE_DIR_OLD = "/content/drive/MyDrive/WRAI_Models"
OUTPUT_DIR = "models_v14_7_transplant"
VOCAB_MAP_FILENAME = "vocab_map_v14_7_transplant.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

drive_active = False
try:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT_PATH, force_remount=False)
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    drive_active = True
    print(f"\n[OK] GOOGLE DRIVE MOUNTED! Vocab will save to: {DRIVE_SAVE_DIR}\n", flush=True)
except Exception as e:
    print(f"\n[INFO] Local environment: {e}\n", flush=True)

# -----------------------------------------------------------------------------
# Multi-Domain Corpus Streamers (Identical seed=42)
# -----------------------------------------------------------------------------

def build_wrai_persona_dataset():
    qa_list = [
        ("<ID>", "Halo", "Halo! Saya WRAI Master Router. Ada yang bisa saya bantu atau arahkan ke spesialis AI hari ini?"),
        ("<ID>", "Halo apa kabar?", "Halo! Kabar saya sangat baik dan siap mengorkestrasi tugas Anda. Ada yang ingin Anda diskusikan?"),
        ("<ID>", "Hai", "Hai! Selamat datang di ekosistem WRAI. Apa yang ingin Anda tanyakan atau delegasikan hari ini?"),
        ("<ID>", "Siapa kamu?", "Saya adalah WRAI (Wavelet Recurrent AI), model router sentral dan orkestrator koloni AI edge yang ringan, cepat, dan efisien."),
        ("<ID>", "Siapa namamu?", "Nama saya WRAI Master Router. Saya bertugas mengelola percakapan dan mendelegasikan tugas khusus ke koloni model spesialis WRAI."),
        ("<ID>", "Jelaskan siapa kamu dan bagaimana ekosistem WRAI bekerja.", "Saya adalah WRAI Master Router. Saya memimpin koloni model edge AI spesialis."),
        ("<EN>", "Buatkan script python untuk web scraping data harga", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_coder\", \"task\": \"Buatkan script python untuk web scraping data harga\"}}\n</tool_call>"),
        ("<EN>", "Analisis potensi celah SQL Injection pada query database ini", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_security\", \"task\": \"Analisis potensi celah SQL Injection dan remedi parameterization\"}}\n</tool_call>"),
        ("<EN>", "Cari berita teknologi AI dan model terbaru hari ini di internet.", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_browser\", \"query\": \"berita teknologi AI dan model terbaru hari ini\"}}\n</tool_call>"),
        ("<EN>", "Hitung turunan pertama dari f(x) = x^3 * e^(2x)", "<tool_call>\n{\"name\": \"route_to_agent\", \"arguments\": {\"agent\": \"wrai_math\", \"expression\": \"derivative(x^3 * exp(2*x), x)\"}}\n</tool_call>"),
        ("<EN>", "Cari berita celah zero-day terbaru di internet lalu analisis dan buatkan script patch perbaikannya di Python.", "<tool_call>\n{\"name\": \"route_pipeline\", \"arguments\": {\"steps\": [{\"step\": 1, \"agent\": \"wrai_browser\", \"task\": \"Cari berita celah zero-day terbaru\"}, {\"step\": 2, \"agent\": \"wrai_security\", \"task\": \"Analisis akar penyebab exploit dan vektor serangan\"}, {\"step\": 3, \"agent\": \"wrai_coder\", \"task\": \"Tulis script Python patch perbaikan dan verifikasi\"}]}}\n</tool_call>")
    ]
    return qa_list

def stream_wrai_persona(n=5000):
    qa_list = build_wrai_persona_dataset()
    count = 0
    while count < n:
        random.shuffle(qa_list)
        for tag, q, a in qa_list:
            yield (tag, q, a + "<|im_end|>")
            count += 1
            if count >= n:
                return

def _iter_sharegpt_pairs(convo):
    pairs = []
    turns = convo.get("conversations", [])
    last_human = None
    for turn in turns:
        role = turn.get("from", "") or turn.get("role", "")
        text = turn.get("value", "") or turn.get("content", "")
        if role in ("human", "user"):
            last_human = text
        elif role in ("gpt", "assistant") and last_human is not None:
            pairs.append((last_human, text))
            last_human = None
    return pairs

def stream_alpaca_id(n=40000):
    ds = load_dataset("FreedomIntelligence/alpaca-gpt4-indonesian", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
    count = 0
    for row in ds:
        for q, a in _iter_sharegpt_pairs(row):
            q, a = q.strip(), a.strip()
            if len(a) <= MAX_CHARS_RESPONSE and len(q) > 0 and len(a) > 0:
                yield ("<ID>", q, a + "<|im_end|>")
                count += 1
                if count >= n:
                    return

def stream_alpaca_en(n=40000):
    ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
    count = 0
    for row in ds:
        if "conversations" in row:
            for q, a in _iter_sharegpt_pairs(row):
                q, a = q.strip(), a.strip()
                if len(a) <= MAX_CHARS_RESPONSE and len(q) > 0 and len(a) > 0:
                    yield ("<EN>", q, a + "<|im_end|>")
                    count += 1
                    if count >= n:
                        return
        elif "instruction" in row:
            q = row["instruction"].strip()
            extra = row.get("input", "")
            if extra:
                q = f"{q}\n{extra.strip()}"
            a = row["output"].strip()
            if len(a) <= MAX_CHARS_RESPONSE and len(q) > 0 and len(a) > 0:
                yield ("<EN>", q, a + "<|im_end|>")
                count += 1
                if count >= n:
                    return

def stream_opus_translation(n=20000):
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-id", split="train", streaming=True)
    except Exception:
        try:
            ds = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-ind", split="train", streaming=True)
        except Exception:
            return
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
    count = 0
    for row in ds:
        trans = row.get("translation", {})
        en = trans.get("en", "") or trans.get("eng", "")
        id_txt = trans.get("id", "") or trans.get("ind", "")
        en, id_txt = en.strip(), id_txt.strip()
        if not en or not id_txt or len(en) > MAX_CHARS_RESPONSE or len(id_txt) > MAX_CHARS_RESPONSE:
            continue
        if random.random() < 0.5:
            yield ("<ID>", f"Terjemahkan ke bahasa Indonesia:\n\"{en}\"", id_txt + "<|im_end|>")
        else:
            yield ("<EN>", f"Translate to English:\n\"{id_txt}\"", en + "<|im_end|>")
        count += 1
        if count >= n:
            return

def stream_hermes_tool(n=11500):
    ds = load_dataset("NousResearch/hermes-function-calling-v1",
                       "func_calling_singleturn", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
    count = 0
    for convo in ds:
        for q, a in _iter_sharegpt_pairs(convo):
            q, a = q.strip(), a.strip()
            if "<tool_call>" in a and len(a) <= MAX_CHARS_RESPONSE:
                yield ("<EN>", q, a + "<|im_end|>")
                count += 1
                if count >= n:
                    return

def round_robin(*generators):
    iters = [iter(g) for g in generators]
    while iters:
        for it in list(iters):
            try:
                yield next(it)
            except StopIteration:
                iters.remove(it)

def make_combined_stream():
    return round_robin(
        stream_wrai_persona(5000),
        stream_alpaca_id(40000),
        stream_alpaca_en(40000),
        stream_opus_translation(20000),
        stream_hermes_tool(11500)
    )

# -----------------------------------------------------------------------------
# Main Generator
# -----------------------------------------------------------------------------

def main():
    print("=================================================================")
    print("   WRAI v14.7 STANDALONE VOCABULARY MAP GENERATOR & DRIVE SYNC   ")
    print("=================================================================")

    # 1. Check vocab size from .pt checkpoint if present in Drive
    target_vocab_size = None
    check_ckpts = [
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_7_transplant_best.pt"),
        os.path.join(DRIVE_SAVE_DIR, "wrai_v14_7_transplant_latest.pt"),
        os.path.join(DRIVE_SAVE_DIR_OLD, "wrai_v14_7_best.pt"),
        os.path.join(DRIVE_SAVE_DIR_OLD, "wrai_v14_7_latest.pt"),
    ]
    for cp in check_ckpts:
        if os.path.exists(cp):
            try:
                ckpt = torch.load(cp, map_location="cpu")
                state = ckpt.get("model_state", ckpt)
                if "embed.weight" in state:
                    target_vocab_size = state["embed.weight"].size(0)
                    print(f"[FOUND CHECKPOINT] Detected exact target vocab size from {cp}: {target_vocab_size}")
                    break
            except Exception:
                pass

    print(f"[*] Loading Tokenizer Base: {TEACHER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, trust_remote_code=True)

    print("[*] Scanning multi-domain dataset to build vocabulary map...", flush=True)
    t0 = time.perf_counter()
    counter = Counter()
    n_seen = 0
    num_tags = len(ROUTE_TAGS) # 2 (<ID>, <EN>)
    if target_vocab_size:
        vocab_target_v = target_vocab_size - num_tags - 2 # 31991 - 2 - 2 = 31987
    else:
        vocab_target_v = VOCAB_CAP - num_tags - 2

    print(f"  --> Target Unique Vocab V: {vocab_target_v} (Total Vocab with Pad/Unk/Tags: {vocab_target_v + num_tags + 2})")

    for tag, q, a in make_combined_stream():
        text = f"{q} {a}"
        tok_ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(tok_ids)
        n_seen += 1
        if len(counter) >= vocab_target_v and n_seen >= 10000:
            break

    most_common = counter.most_common(vocab_target_v)
    teacher_to_pruned = {tok_id: rank + 2 for rank, (tok_id, _) in enumerate(most_common)}
    pruned_to_teacher = {v: k for k, v in teacher_to_pruned.items()}
    V = len(most_common)

    tag_to_id = {tag: V + 2 + i for i, tag in enumerate(ROUTE_TAGS)}
    total_vocab_size = V + 2 + len(ROUTE_TAGS)
    print(f"[OK] Vocab scanned from {n_seen} samples in {(time.perf_counter()-t0):.1f}s | V={V}, total={total_vocab_size}")

    save_obj = {
        "teacher_id_to_pruned_id": {str(k): v for k, v in teacher_to_pruned.items()},
        "pruned_id_to_teacher_id": {str(k): v for k, v in pruned_to_teacher.items()},
        "tag_to_id": tag_to_id,
        "pad_id": 0, "unk_id": 1,
        "total_vocab_size": total_vocab_size,
        "V": V,
    }

    # Save to local and all Drive candidate folders
    save_locations = [
        os.path.join(OUTPUT_DIR, VOCAB_MAP_FILENAME),
        VOCAB_MAP_FILENAME
    ]
    if drive_active:
        save_locations.append(os.path.join(DRIVE_SAVE_DIR, VOCAB_MAP_FILENAME))
        save_locations.append(os.path.join(DRIVE_SAVE_DIR_OLD, VOCAB_MAP_FILENAME))

    for loc in save_locations:
        try:
            os.makedirs(os.path.dirname(loc) if os.path.dirname(loc) else ".", exist_ok=True)
            with open(loc, "w", encoding="utf-8") as f:
                json.dump(save_obj, f, ensure_ascii=False, indent=2)
            print(f"[OK SAVED] -> {loc}")
        except Exception as e:
            print(f"[WARN] Failed to write {loc}: {e}")

    print("\n=================================================================")
    print(f"   🎉 VOCABULARY MAP SUCCESSFULLY CREATED (Total Vocab: {total_vocab_size})   ")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
