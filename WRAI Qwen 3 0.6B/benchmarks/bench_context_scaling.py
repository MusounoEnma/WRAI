#!/usr/bin/env python3
"""Context Scaling and Break-Even Memory Audit.
Evaluates Qwen Native vs WRAI across prompt lengths [27, 128, 512, 1024] tokens.
Measures Working Set, Private Bytes, Peak Working Set, Prefill time, Decode speed,
and exact Attention/KV-Cache payload to find the empirical memory break-even point.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import threading
import time
from pathlib import Path
import sys

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from wrai_standalone_runner import load_full_standalone_wrai_model


class ProcessMemoryTracker:
    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.proc = psutil.Process()
        self.peak_wset = 0
        self.peak_private = 0
        self._stop = threading.Event()
        self._thread = None
        self.sample()

    def sample(self):
        try:
            m = self.proc.memory_info()
            ws = getattr(m, "wset", m.rss)
            pv = getattr(m, "private", m.vms)
            pw = getattr(m, "peak_wset", ws)
            if ws > self.peak_wset: self.peak_wset = ws
            if pw > self.peak_wset: self.peak_wset = pw
            if pv > self.peak_private: self.peak_private = pv
            return ws, pv, self.peak_wset
        except Exception:
            return 0, 0, self.peak_wset

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        return self.sample()

    def current(self):
        m = self.proc.memory_info()
        ws = getattr(m, "wset", m.rss)
        pv = getattr(m, "private", m.vms)
        pw = getattr(m, "peak_wset", self.peak_wset)
        return ws, pv, pw


def build_synthetic_prompt(tokenizer, target_tokens: int) -> str:
    base = (
        "Artificial intelligence and neural network systems utilize multi-layer attention "
        "and feedforward projections to model natural language hierarchies across long context spans. "
        "Each layer computes key-value projections and recurrent state updates to maintain temporal coherence. "
    )
    tokens = tokenizer(base, add_special_tokens=False)["input_ids"]
    repeat_count = (target_tokens // len(tokens)) + 2
    full_text = base * repeat_count
    encoded = tokenizer(full_text, add_special_tokens=False)["input_ids"][:target_tokens - 14]
    trimmed_text = tokenizer.decode(encoded)
    prompt = (
        "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        f"<|im_start|>user\n{trimmed_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def run_scaling_benchmark():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-lengths", nargs="+", type=int, default=[27, 128, 512, 1024])
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default=r"C:\Users\Administrator\Desktop\WRAI_v2_CPU_AUDIT\context_scaling_comparison.csv")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    tok_dir = Path("standalone_models/tokenizer_and_config").resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    device = torch.device("cpu")
    dtype = torch.float32

    rows = []

    # -------------------------------------------------------------
    # 1. EVALUATE WRAI v2.0
    # -------------------------------------------------------------
    print("=" * 70)
    print(" [BENCH] EVALUATING WRAI v2.0 STANDALONE ACROSS CONTEXT LENGTHS")
    print("=" * 70)
    tracker_w = ProcessMemoryTracker(0.005)
    tracker_w.start()

    pt_path = Path("standalone_models/wrai_qwen3_0_6b_micro_int8_standalone.pt").resolve()
    print(f"[*] Loading WRAI Model ({pt_path.name})...")
    model_w, _, wrai_layers = load_full_standalone_wrai_model(pt_path, tok_dir, device, dtype, backend_override="jit_float")
    ws_load_w, pv_load_w, peak_load_w = tracker_w.current()
    print(f"[+] WRAI Loaded | Working Set: {ws_load_w / (1024**2):.1f} MiB | Peak: {peak_load_w / (1024**2):.1f} MiB")

    core_w = model_w.model
    poison_cache = DynamicCache()
    def _forbidden_kv_write(*_a, **_kw):
        raise RuntimeError("FATAL KV_CACHE_POISON_TRAP TRIGGERED!")
    poison_cache.update = _forbidden_kv_write

    for target_len in args.target_lengths:
        prompt_text = build_synthetic_prompt(tokenizer, target_len)
        inp_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cpu")
        actual_ctx = int(inp_ids.shape[1])
        print(f"\n---> [WRAI] Context: {actual_ctx} tokens | Target: {target_len} tokens")

        t0 = time.perf_counter()
        out_pre = core_w(input_ids=inp_ids, past_key_values=poison_cache, use_cache=True, return_dict=True)
        prefill_s = time.perf_counter() - t0
        ws_post_pre, pv_post_pre, peak_post_pre = tracker_w.sample()

        next_id = int(torch.argmax(model_w.lm_head(out_pre.last_hidden_state[:, -1:, :])[:, -1, :], dim=-1).item())
        generated = [next_id]
        next_input = torch.tensor([[next_id]], dtype=torch.long, device="cpu")
        pos_ids = torch.tensor([[actual_ctx]], dtype=torch.long, device="cpu")

        t1 = time.perf_counter()
        for si in range(1, args.decode_tokens):
            next_input[0, 0] = next_id
            pos_ids[0, 0] = actual_ctx + si - 1
            out = core_w(input_ids=next_input, position_ids=pos_ids, past_key_values=poison_cache, use_cache=True, return_dict=True)
            next_id = int(torch.argmax(model_w.lm_head(out.last_hidden_state[:, -1, :]), dim=-1).item())
            generated.append(next_id)
        decode_s = time.perf_counter() - t1
        ws_post_dec, pv_post_dec, peak_post_dec = tracker_w.sample()

        decode_steps = len(generated) - 1
        decode_tps = decode_steps / max(decode_s, 1e-9)
        actual_state_mib = sum(l.get_actual_state_bytes() for l in wrai_layers) / (1024.0**2)

        row = {
            "model": "WRAI_v2.0",
            "target_context": target_len,
            "actual_prompt_tokens": actual_ctx,
            "generated_tokens": len(generated),
            "prefill_s": round(prefill_s, 3),
            "decode_s": round(decode_s, 3),
            "decode_tps": round(decode_tps, 3),
            "post_load_wset_mib": round(ws_load_w / (1024**2), 1),
            "post_prefill_wset_mib": round(ws_post_pre / (1024**2), 1),
            "post_decode_wset_mib": round(ws_post_dec / (1024**2), 1),
            "peak_wset_mib": round(tracker_w.peak_wset / (1024**2), 1),
            "private_bytes_mib": round(pv_post_dec / (1024**2), 1),
            "attention_memory_payload_mib": round(actual_state_mib, 2),
            "memory_type": "Constant State O(1)",
        }
        rows.append(row)
        print(f"     Prefill: {prefill_s:.2f}s | Decode: {decode_tps:.2f} tok/s | Post-Dec Working Set: {row['post_decode_wset_mib']} MiB | WRAI State: {actual_state_mib:.2f} MiB")
        gc.collect()

    tracker_w.stop()
    del model_w, core_w
    gc.collect()

    # -------------------------------------------------------------
    # 2. EVALUATE QWEN NATIVE
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [BENCH] EVALUATING QWEN3-0.6B NATIVE ACROSS CONTEXT LENGTHS")
    print("=" * 70)
    tracker_q = ProcessMemoryTracker(0.005)
    tracker_q.start()

    print("[*] Loading Native Qwen Model...")
    model_q = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=False,
    )
    model_q.to("cpu").eval()
    model_q.requires_grad_(False)
    core_q = model_q.model
    ws_load_q, pv_load_q, peak_load_q = tracker_q.current()
    print(f"[+] Qwen Native Loaded | Working Set: {ws_load_q / (1024**2):.1f} MiB | Peak: {peak_load_q / (1024**2):.1f} MiB")

    for target_len in args.target_lengths:
        prompt_text = build_synthetic_prompt(tokenizer, target_len)
        inp_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cpu")
        actual_ctx = int(inp_ids.shape[1])
        print(f"\n---> [Qwen Native] Context: {actual_ctx} tokens | Target: {target_len} tokens")

        t0 = time.perf_counter()
        out_pre = core_q(input_ids=inp_ids, use_cache=True, return_dict=True)
        prefill_s = time.perf_counter() - t0
        cache = out_pre.past_key_values
        ws_post_pre, pv_post_pre, peak_post_pre = tracker_q.sample()

        next_id = int(torch.argmax(model_q.lm_head(out_pre.last_hidden_state[:, -1:, :])[:, -1, :], dim=-1).item())
        generated = [next_id]
        next_input = torch.tensor([[next_id]], dtype=torch.long, device="cpu")

        t1 = time.perf_counter()
        for si in range(1, args.decode_tokens):
            next_input[0, 0] = next_id
            pos_ids = torch.tensor([[actual_ctx + si - 1]], dtype=torch.long, device="cpu")
            out = core_q(input_ids=next_input, position_ids=pos_ids, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            next_id = int(torch.argmax(model_q.lm_head(out.last_hidden_state[:, -1, :]), dim=-1).item())
            generated.append(next_id)
        decode_s = time.perf_counter() - t1
        ws_post_dec, pv_post_dec, peak_post_dec = tracker_q.sample()

        decode_steps = len(generated) - 1
        decode_tps = decode_steps / max(decode_s, 1e-9)

        # 28 layers * 2 (K+V) * 8 heads * 128 dim * 4 bytes/float = 229,376 bytes/token ≈ 224 KiB/token
        total_tokens = actual_ctx + len(generated)
        kv_cache_mib = (total_tokens * 229376) / (1024**2)

        row = {
            "model": "Qwen3-0.6B_Native",
            "target_context": target_len,
            "actual_prompt_tokens": actual_ctx,
            "generated_tokens": len(generated),
            "prefill_s": round(prefill_s, 3),
            "decode_s": round(decode_s, 3),
            "decode_tps": round(decode_tps, 3),
            "post_load_wset_mib": round(ws_load_q / (1024**2), 1),
            "post_prefill_wset_mib": round(ws_post_pre / (1024**2), 1),
            "post_decode_wset_mib": round(ws_post_dec / (1024**2), 1),
            "peak_wset_mib": round(tracker_q.peak_wset / (1024**2), 1),
            "private_bytes_mib": round(pv_post_dec / (1024**2), 1),
            "attention_memory_payload_mib": round(kv_cache_mib, 2),
            "memory_type": "Linear Growing O(N)",
        }
        rows.append(row)
        print(f"     Prefill: {prefill_s:.2f}s | Decode: {decode_tps:.2f} tok/s | Post-Dec Working Set: {row['post_decode_wset_mib']} MiB | Qwen KV-Cache: {kv_cache_mib:.2f} MiB")
        gc.collect()

    tracker_q.stop()

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[DONE] Context Scaling Comparison saved to {out_file.resolve()}")


if __name__ == "__main__":
    run_scaling_benchmark()
