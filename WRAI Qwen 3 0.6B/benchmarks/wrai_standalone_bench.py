#!/usr/bin/env python3
"""CPU-only WRAI standalone benchmark for controlled comparison with Qwen native.

Uses Hugging Face Transformers + PyTorch skeleton, FP32/JIT Float weights,
identical ChatML prompt framing, greedy argmax, and a fixed decode length.
The fixed-length benchmark intentionally does not stop on EOS so both models
can be measured over exactly the same number of decode steps.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path
import sys

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import torch
from transformers import DynamicCache

from wrai_standalone_runner import load_full_standalone_wrai_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WRAI CPU Benchmark Runner")
    p.add_argument("--model", default="standalone_models/wrai_qwen3_0_6b_micro_int8_standalone.pt",
                   help="Path to WRAI standalone .pt model")
    p.add_argument("--tokenizer", default="standalone_models/tokenizer_and_config",
                   help="Tokenizer directory shared with WRAI")
    p.add_argument("--backend", default="jit_float", choices=["jit_float", "pytorch_int8", "cpp_avx1"])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--user-text", default="Say hello in one short sentence.")
    p.add_argument("--out", default="wrai_cpu_metrics.csv")
    return p.parse_args()


@torch.no_grad()
def run_once_wrai(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    inp_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cpu")
    ctx_len = int(inp_ids.shape[1])
    core = model.model

    poison_cache = DynamicCache()
    def _forbidden_kv_write(*_a, **_kw):
        raise RuntimeError("FATAL KV_CACHE_POISON_TRAP TRIGGERED!")
    poison_cache.update = _forbidden_kv_write

    t0 = time.perf_counter()
    out = core(input_ids=inp_ids, past_key_values=poison_cache, use_cache=True, return_dict=True)
    prefill_s = time.perf_counter() - t0

    next_id = int(torch.argmax(model.lm_head(out.last_hidden_state[:, -1, :]), dim=-1).item())
    generated = [next_id]
    next_input = torch.tensor([[next_id]], dtype=torch.long, device="cpu")

    # Fixed decode count mirroring native benchmark loop. Deliberately no EOS early-stopping.
    t1 = time.perf_counter()
    for step_i in range(1, max_new_tokens):
        next_input[0, 0] = next_id
        pos_ids = torch.tensor([[ctx_len + step_i - 1]], dtype=torch.long, device="cpu")
        out = core(
            input_ids=next_input,
            position_ids=pos_ids,
            past_key_values=poison_cache,
            use_cache=True,
            return_dict=True,
        )
        next_id = int(torch.argmax(model.lm_head(out.last_hidden_state[:, -1, :]), dim=-1).item())
        generated.append(next_id)
    decode_s = time.perf_counter() - t1

    decode_steps = max(0, len(generated) - 1)
    return {
        "prompt_tokens": ctx_len,
        "generated_tokens": len(generated),
        "decode_steps": decode_steps,
        "prefill_seconds": prefill_s,
        "decode_seconds": decode_s,
        "prefill_tokens_per_second": ctx_len / max(prefill_s, 1e-9),
        "decode_tokens_per_second": decode_steps / max(decode_s, 1e-9),
        "total_tokens_per_second": len(generated) / max(prefill_s + decode_s, 1e-9),
        "text_preview": tokenizer.decode(generated[:32], skip_special_tokens=False),
    }


def main() -> None:
    args = parse_args()
    if args.threads < 1 or args.max_new_tokens < 2 or args.runs < 1:
        raise SystemExit("threads >= 1, max-new-tokens >= 2, runs >= 1 required")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    bundle_dir = Path(__file__).resolve().parent
    pt_path = bundle_dir / args.model
    tok_dir = bundle_dir / args.tokenizer
    device = torch.device("cpu")
    dtype = torch.float32

    print(f"[ENV] torch={torch.__version__}")
    print(f"[ENV] device=cpu dtype=float32 backend={args.backend} threads={args.threads}")
    print(f"[MODEL] {pt_path.name}")
    print(f"[TOKENIZER] {tok_dir}")
    print(f"[LOAD] Loading WRAI Standalone Model on CPU...")

    model, tokenizer, wrai_layers = load_full_standalone_wrai_model(
        pt_path, tok_dir, device, dtype, backend_override=args.backend
    )

    prompt = (
        "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        f"<|im_start|>user\n{args.user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    print(f"[PROMPT] {args.user_text!r}")
    print(f"[BENCH] fixed_new_tokens={args.max_new_tokens}; no EOS early-stop")

    for i in range(args.warmup_runs):
        print(f"[WARMUP {i + 1}/{args.warmup_runs}]")
        run_once_wrai(model, tokenizer, prompt, args.max_new_tokens)

    results = []
    for i in range(args.runs):
        row = run_once_wrai(model, tokenizer, prompt, args.max_new_tokens)
        row["run_id"] = i + 1
        row["threads"] = args.threads
        row["dtype"] = "float32"
        row["backend"] = args.backend
        row["model"] = pt_path.name
        results.append(row)
        print(
            f"[RUN {i + 1}/{args.runs}] prompt={row['prompt_tokens']} tok | "
            f"prefill={row['prefill_seconds']:.3f}s | "
            f"decode={row['decode_tokens_per_second']:.3f} tok/s | "
            f"total={row['total_tokens_per_second']:.3f} tok/s | "
            f"generated={row['generated_tokens']} tok"
        )
        print(f"[PREVIEW] {row['text_preview']!r}")
        gc.collect()

    out_path = Path(args.out)
    fieldnames = [
        "run_id", "threads", "dtype", "backend", "model", "prompt_tokens",
        "generated_tokens", "decode_steps", "prefill_seconds", "decode_seconds",
        "prefill_tokens_per_second", "decode_tokens_per_second",
        "total_tokens_per_second", "text_preview",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"[SAVED] {out_path.resolve()}")


if __name__ == "__main__":
    main()
