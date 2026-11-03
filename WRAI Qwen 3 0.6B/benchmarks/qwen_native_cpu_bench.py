#!/usr/bin/env python3
"""CPU-only Qwen3 native benchmark for controlled comparison with WRAI.

Uses Hugging Face Transformers + PyTorch, FP32 weights, eager attention,
identical ChatML prompt framing, greedy argmax, and a fixed decode length.
The fixed-length benchmark intentionally does not stop on EOS so both models
can be measured over exactly the same number of decode steps.
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B",
                   help="HF model ID or local path to the original Qwen checkpoint")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer directory shared with WRAI; defaults to --model")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--user-text", default="Say hello in one short sentence.")
    p.add_argument("--out", default="qwen_native_cpu_metrics.csv")
    return p.parse_args()


def local_only_if_path_exists(ref: str) -> bool:
    return Path(ref).expanduser().exists()


@torch.inference_mode()
def run_once(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    input_ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    )["input_ids"].to("cpu")
    ctx_len = int(input_ids.shape[1])
    core = model.model

    t0 = time.perf_counter()
    out = core(input_ids=input_ids, use_cache=True, return_dict=True)
    prefill_s = time.perf_counter() - t0
    cache = out.past_key_values

    next_id = int(torch.argmax(model.lm_head(out.last_hidden_state[:, -1, :]), dim=-1).item())
    generated = [next_id]
    next_input = torch.tensor([[next_id]], dtype=torch.long, device="cpu")

    # Fixed decode count, mirroring WRAI's step loop. EOS is deliberately not
    # used as a stopping condition in this performance-only benchmark.
    t1 = time.perf_counter()
    for step_i in range(1, max_new_tokens):
        next_input[0, 0] = next_id
        pos_ids = torch.tensor([[ctx_len + step_i - 1]], dtype=torch.long, device="cpu")
        out = core(
            input_ids=next_input,
            position_ids=pos_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
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

    # Set thread limits before loading/running the model. Set corresponding
    # environment variables in PowerShell before launching Python as well.
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    tokenizer_ref = args.tokenizer or args.model
    print(f"[ENV] torch={torch.__version__}")
    print(f"[ENV] device=cpu dtype=float32 attention=eager threads={args.threads}")
    print(f"[MODEL] {args.model}")
    print(f"[TOKENIZER] {tokenizer_ref}")
    print("[LOAD] Loading native Qwen checkpoint. First run may download weights.")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        local_files_only=local_only_if_path_exists(tokenizer_ref),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=local_only_if_path_exists(args.model),
    )
    model.to("cpu")
    model.eval()
    model.requires_grad_(False)

    prompt = (
        "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        f"<|im_start|>user\n{args.user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    print(f"[PROMPT] {args.user_text!r}")
    print(f"[BENCH] fixed_new_tokens={args.max_new_tokens}; no EOS early-stop")

    for i in range(args.warmup_runs):
        print(f"[WARMUP {i + 1}/{args.warmup_runs}]")
        run_once(model, tokenizer, prompt, args.max_new_tokens)

    results = []
    for i in range(args.runs):
        row = run_once(model, tokenizer, prompt, args.max_new_tokens)
        row["run_id"] = i + 1
        row["threads"] = args.threads
        row["dtype"] = "float32"
        row["attention"] = "eager"
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
        "run_id", "threads", "dtype", "attention", "prompt_tokens",
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
    print("[NOTE] Compare decode tok/s only with WRAI measured using the same fixed-step loop,")
    print("       prompt tokenization, dtype, PyTorch version and thread count.")


if __name__ == "__main__":
    main()
