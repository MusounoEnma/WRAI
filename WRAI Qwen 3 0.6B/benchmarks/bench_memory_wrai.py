#!/usr/bin/env python3
"""Memory footprint audit for WRAI v2.0 Standalone on CPU.
Tracks Working Set, Private Bytes, Peak Working Set, Prefill Peak,
Decode Footprint, and Actual Tensor State Size across all lifecycle phases.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import threading
import time
import sys
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import psutil
import torch
from transformers import DynamicCache

from wrai_standalone_runner import load_full_standalone_wrai_model


class ProcessMemoryTracker:
    def __init__(self, sample_interval_s: float = 0.005):
        self.interval = sample_interval_s
        self.proc = psutil.Process()
        self.peak_wset = 0
        self.peak_private = 0
        self._stop_event = threading.Event()
        self._thread = None
        self.sample()

    def sample(self):
        try:
            m = self.proc.memory_info()
            ws = getattr(m, "wset", m.rss)
            pv = getattr(m, "private", m.vms)
            pw = getattr(m, "peak_wset", ws)
            if ws > self.peak_wset:
                self.peak_wset = ws
            if pw > self.peak_wset:
                self.peak_wset = pw
            if pv > self.peak_private:
                self.peak_private = pv
            return ws, pv, self.peak_wset
        except Exception:
            return 0, 0, self.peak_wset

    def start_background(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop_event.is_set():
            self.sample()
            time.sleep(self.interval)

    def stop_background(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        return self.sample()

    def get_current(self):
        m = self.proc.memory_info()
        ws = getattr(m, "wset", m.rss)
        pv = getattr(m, "private", m.vms)
        pw = getattr(m, "peak_wset", self.peak_wset)
        return ws, pv, pw


def run_benchmark():
    parser = argparse.ArgumentParser(description="WRAI Memory Audit")
    parser.add_argument("--model", default="standalone_models/wrai_qwen3_0_6b_micro_int8_standalone.pt")
    parser.add_argument("--backend", default="jit_float")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--user-text", default="Say hello in one short sentence.")
    parser.add_argument("--out", default=r"C:\Users\Administrator\Desktop\WRAI_v2_CPU_AUDIT\wrai_memory_footprint_t2.csv")
    args = parser.parse_args()

    # Track memory right from process start
    mem_tracker = ProcessMemoryTracker(0.005)
    mem_tracker.start_background()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    bundle_dir = Path(__file__).resolve().parent
    pt_path = bundle_dir / args.model
    tok_dir = bundle_dir / "standalone_models" / "tokenizer_and_config"
    device = torch.device("cpu")
    dtype = torch.float32

    print(f"[MEM-INIT] Working Set before load: {mem_tracker.sample()[0] / (1024**2):.2f} MiB")
    print(f"[LOAD] Loading WRAI Model: {pt_path.name} on CPU ({dtype})...")
    t_load_0 = time.perf_counter()
    model, tokenizer, wrai_layers = load_full_standalone_wrai_model(
        pt_path, tok_dir, device, dtype, backend_override=args.backend
    )
    t_load = time.perf_counter() - t_load_0
    ws_load, pv_load, peak_load = mem_tracker.get_current()
    print(f"[LOAD-DONE] Took {t_load:.2f}s | Post-Load Working Set: {ws_load / (1024**2):.2f} MiB | Peak: {peak_load / (1024**2):.2f} MiB")

    # Calculate actual state size in tensor bytes
    actual_state_bytes = sum(l.get_actual_state_bytes() for l in wrai_layers)
    actual_state_mib = actual_state_bytes / (1024.0**2)
    print(f"[WRAI-STATE] Actual Tensor State Size across all 28 layers: {actual_state_mib:.2f} MiB")

    prompt = (
        "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        f"<|im_start|>user\n{args.user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inp_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cpu")
    ctx_len = int(inp_ids.shape[1])
    core = model.model

    poison_cache = DynamicCache()
    def _forbidden_kv_write(*_a, **_kw):
        raise RuntimeError("FATAL KV_CACHE_POISON_TRAP TRIGGERED!")
    poison_cache.update = _forbidden_kv_write

    # Warmup
    for w in range(args.warmup_runs):
        print(f"[WARMUP {w+1}/{args.warmup_runs}] Running...")
        out_w = core(input_ids=inp_ids, past_key_values=poison_cache, use_cache=True, return_dict=True)
        next_id_w = int(torch.argmax(model.lm_head(out_w.last_hidden_state[:, -1, :]), dim=-1).item())
        tok_w = torch.tensor([[next_id_w]], dtype=torch.long, device="cpu")
        pos_w = torch.tensor([[ctx_len]], dtype=torch.long, device="cpu")
        for si in range(1, args.max_new_tokens):
            tok_w[0, 0] = next_id_w
            pos_w[0, 0] = ctx_len + si - 1
            out_w = core(input_ids=tok_w, position_ids=pos_w, past_key_values=poison_cache, use_cache=True, return_dict=True)
            next_id_w = int(torch.argmax(model.lm_head(out_w.last_hidden_state[:, -1, :]), dim=-1).item())

    gc.collect()

    results = []
    for r in range(args.runs):
        print(f"\n[RUN {r+1}/{args.runs}] Beginning controlled memory audit run...")
        ws_pre_run, pv_pre_run, _ = mem_tracker.get_current()
        peak_before_prefill = mem_tracker.peak_wset

        # 1. Prefill Phase
        t0 = time.perf_counter()
        out_pre = core(input_ids=inp_ids, past_key_values=poison_cache, use_cache=True, return_dict=True)
        prefill_s = time.perf_counter() - t0
        ws_post_prefill, pv_post_prefill, peak_after_prefill = mem_tracker.sample()
        prefill_peak_delta_mib = max(0, (peak_after_prefill - peak_before_prefill)) / (1024**2)

        next_id = int(torch.argmax(model.lm_head(out_pre.last_hidden_state[:, -1, :]), dim=-1).item())
        generated = [next_id]
        next_input = torch.tensor([[next_id]], dtype=torch.long, device="cpu")
        pos_ids = torch.tensor([[ctx_len]], dtype=torch.long, device="cpu")

        # 2. Decode Phase
        peak_before_decode = mem_tracker.peak_wset
        t1 = time.perf_counter()
        for step_i in range(1, args.max_new_tokens):
            next_input[0, 0] = next_id
            pos_ids[0, 0] = ctx_len + step_i - 1
            out = core(input_ids=next_input, position_ids=pos_ids, past_key_values=poison_cache, use_cache=True, return_dict=True)
            next_id = int(torch.argmax(model.lm_head(out.last_hidden_state[:, -1, :]), dim=-1).item())
            generated.append(next_id)
        decode_s = time.perf_counter() - t1
        ws_post_decode, pv_post_decode, peak_after_decode = mem_tracker.sample()
        decode_peak_delta_mib = max(0, (peak_after_decode - peak_before_decode)) / (1024**2)

        decode_steps = len(generated) - 1
        decode_tps = decode_steps / max(decode_s, 1e-9)
        actual_state_mib = sum(l.get_actual_state_bytes() for l in wrai_layers) / (1024.0**2)

        row = {
            "run_id": r + 1,
            "model": "WRAI_v2.0",
            "threads": args.threads,
            "prompt_tokens": ctx_len,
            "generated_tokens": len(generated),
            "prefill_s": round(prefill_s, 3),
            "decode_s": round(decode_s, 3),
            "decode_tps": round(decode_tps, 3),
            "post_load_wset_mib": round(ws_load / (1024**2), 2),
            "post_load_private_mib": round(pv_load / (1024**2), 2),
            "post_prefill_wset_mib": round(ws_post_prefill / (1024**2), 2),
            "post_prefill_private_mib": round(pv_post_prefill / (1024**2), 2),
            "post_decode_wset_mib": round(ws_post_decode / (1024**2), 2),
            "post_decode_private_mib": round(pv_post_decode / (1024**2), 2),
            "peak_wset_mib": round(mem_tracker.peak_wset / (1024**2), 2),
            "peak_private_mib": round(mem_tracker.peak_private / (1024**2), 2),
            "prefill_peak_delta_mib": round(prefill_peak_delta_mib, 2),
            "decode_peak_delta_mib": round(decode_peak_delta_mib, 2),
            "actual_state_mib": round(actual_state_mib, 2),
            "kv_cache_type": "Zero-KV (Recurrent State)",
        }
        results.append(row)
        print(
            f"[RUN {r+1} RESULT] Decode: {decode_tps:.2f} tok/s | "
            f"Prefill: {prefill_s:.2f}s | "
            f"Working Set: {row['post_decode_wset_mib']} MiB | "
            f"Private: {row['post_decode_private_mib']} MiB | "
            f"Peak Working Set: {row['peak_wset_mib']} MiB | "
            f"Actual State: {row['actual_state_mib']} MiB"
        )
        gc.collect()

    mem_tracker.stop_background()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n[SAVED] WRAI Memory Metrics saved to {out_path.resolve()}")


if __name__ == "__main__":
    run_benchmark()
