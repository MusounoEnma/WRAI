#!/usr/bin/env python3
"""
=============================================================================
  🔬 QWEN/QWEN3-0.8B NATIVE COLAB DIAGNOSTIC & BEHAVIOR INSPECTOR
=============================================================================
 Purpose:
  Run the ORIGINAL un-transplanted Qwen3-0.8B model on Google Colab to uncover:
   1. The exact Native Chat Template & Special Tokens (<think>, </think>, etc.)
   2. How Native Qwen handles Thinking Chain-of-Thought (<think>...</think>)
   3. Native Indonesian vs English language separation & reasoning quality
   4. How Native Qwen solves the exact benchmark questions (Math, Code, Science)
   5. How to control/suppress thinking mode (e.g. system prompt or prefill)
=============================================================================
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.8B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inspect_tokenizer_and_templates(tok):
    print("\n" + "=" * 75)
    print(" 1. TOKENIZER & CHAT TEMPLATE INSPECTION")
    print("=" * 75)
    print(f"[*] Vocab Size: {tok.vocab_size:,}")
    print(f"[*] EOS Token: {tok.eos_token} (ID: {tok.eos_token_id})")
    print(f"[*] Pad Token: {tok.pad_token} (ID: {tok.pad_token_id})")
    
    # Check Special Tokens
    special_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>"]
    print("\n[*] Special Tokens Check:")
    for st in special_tokens:
        tid = tok.convert_tokens_to_ids(st)
        in_vocab = st in tok.get_vocab()
        print(f"    - {st:<15}: ID = {tid} | In Vocab: {in_vocab}")

    # Inspect Chat Template
    print("\n[*] Native Chat Template (First 400 chars):")
    if tok.chat_template:
        print(tok.chat_template[:400] + ("..." if len(tok.chat_template) > 400 else ""))
    else:
        print("    [None defined - Uses standard ChatML]")

    # Example rendered ChatML
    test_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    try:
        rendered = tok.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
        print("\n[*] Rendered Prompt with apply_chat_template(add_generation_prompt=True):")
        print("-" * 50)
        print(repr(rendered))
        print("-" * 50)
    except Exception as e:
        print(f"[!] apply_chat_template error: {e}")

def run_native_generation(model, tok, prompt_or_msgs, max_new_tokens=256, temperature=0.1):
    if isinstance(prompt_or_msgs, list):
        try:
            prompt = tok.apply_chat_template(prompt_or_msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = ""
            for m in prompt_or_msgs:
                prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
    else:
        prompt = prompt_or_msgs

    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.90 if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id
        )
    gen_tokens = outputs[0][input_len:]
    gen_text = tok.decode(gen_tokens, skip_special_tokens=False)
    elapsed = time.time() - t0
    tok_sec = len(gen_tokens) / max(elapsed, 0.001)

    return gen_text, elapsed, tok_sec

def run_comparative_diagnostics(model, tok):
    print("\n" + "=" * 75)
    print(" 2. COMPARATIVE BENCHMARK DIAGNOSTICS (ORIGINAL QWEN 0.8B)")
    print("=" * 75)

    test_cases = [
        # (Category, Description, System Prompt, User Query)
        ("ENGLISH SCIENCE", "Photosynthesis test", 
         "You are a helpful assistant.", 
         "What is photosynthesis?"),

        ("INDONESIAN SCIENCE", "Uji Fotosintesis Bahasa Indonesia", 
         "Kamu adalah asisten cerdas yang berbahasa Indonesia.", 
         "Apa itu fotosintesis dan mengapa penting bagi bumi?"),

        ("ENGLISH CS", "Compiler vs Interpreter", 
         "You are a helpful assistant.", 
         "Explain the difference between a compiler and an interpreter."),

        ("INDONESIAN CS", "Compiler vs Interpreter Indonesia", 
         "Kamu adalah asisten cerdas yang berbahasa Indonesia.", 
         "Jelaskan perbedaan mendasar antara compiler dan interpreter."),

        ("ENGLISH MATH REASONING", "Chain of thought Math (12 x 15)", 
         "You are a helpful assistant.", 
         "What is 12 multiplied by 15? Explain the steps."),

        ("INDONESIAN MATH REASONING", "Penalaran Matematika (12 x 15)", 
         "Kamu adalah asisten cerdas yang berbahasa Indonesia.", 
         "Berapa 12 dikali 15? Jelaskan langkah perhitungannya."),

        ("ENGLISH MATH ALGEBRA", "Solve equation 2x + 6 = 14", 
         "You are a helpful assistant.", 
         "Solve the equation: 2x + 6 = 14. What is the value of x?"),

        ("INDONESIAN MATH ALGEBRA", "Aljabar 2x + 6 = 14", 
         "Kamu adalah asisten cerdas yang berbahasa Indonesia.", 
         "Selesaikan persamaan: 2x + 6 = 14. Berapa nilai x?"),

        ("PROGRAMMING C", "Hello World C", 
         "You are an expert programmer.", 
         "Write a simple C program to print Hello World."),

        ("PROGRAMMING PYTHON", "Palindrome Python", 
         "You are an expert programmer.", 
         "Write a Python function to check if a word is a palindrome."),

        ("LANGUAGE SWITCH TEST", "English Query with Indonesian System", 
         "Jawablah selalu dalam bahasa Indonesia yang baik dan benar.", 
         "What causes day and night on Earth?"),

        ("THINKING CONTROL TEST", "Direct Answer without <think>", 
         "Answer directly and concisely without using any <think> tags.", 
         "If today is Wednesday, what day will it be in 10 days?")
    ]

    for idx, (cat, desc, sys_prompt, user_q) in enumerate(test_cases, 1):
        print(f"\n[{idx:02d}/{len(test_cases)}] [{cat}] - {desc}")
        print(f"System > {sys_prompt}")
        print(f"User   > {user_q}")
        
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_q}
        ]

        text, elapsed, speed = run_native_generation(model, tok, msgs, max_new_tokens=220, temperature=0.1)
        
        # Analyze structure: Did it use <think>?
        has_think = "<think>" in text
        has_end_think = "</think>" in text
        
        print("Qwen3  >")
        print("-" * 50)
        print(text.strip())
        print("-" * 50)
        print(f"Stats  : {elapsed:.2f}s ({speed:.1f} tok/s) | Uses <think>: {has_think} | Closes </think>: {has_end_think}")

def start_interactive_session(model, tok):
    print("\n" + "=" * 75)
    print(" 💬 INTERACTIVE CHAT WITH NATIVE QWEN 0.8B")
    print(" (Test any prompt in English, Indonesian, or code. Type 'exit' to quit)")
    print("=" * 75)

    sys_prompt = "You are a helpful assistant."
    history = [{"role": "system", "content": sys_prompt}]

    while True:
        try:
            q = input("\nYou > ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break
            if q.startswith("/system "):
                sys_prompt = q[8:].strip()
                history = [{"role": "system", "content": sys_prompt}]
                print(f"[*] System prompt updated to: {sys_prompt}")
                continue
            if q.lower() == "/clear":
                history = [{"role": "system", "content": sys_prompt}]
                print("[*] Conversation history cleared.")
                continue

            history.append({"role": "user", "content": q})
            print("Qwen3 > ", end="", flush=True)

            text, elapsed, speed = run_native_generation(model, tok, history, max_new_tokens=256, temperature=0.1)
            print(text.strip())
            print(f"[{elapsed:.2f}s, {speed:.1f} tok/s]")
            history.append({"role": "assistant", "content": text})

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

def main():
    print("=" * 75)
    print("   🔬 QWEN 3 (0.8B) NATIVE TRANSFORMER BENCHMARK & DIAGNOSTIC")
    print("=" * 75)
    print(f"[*] Device: {DEVICE}")

    print(f"[*] Loading native {MODEL_NAME} from HuggingFace...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("[OK] Native Qwen Model Loaded Successfully into GPU Memory!")

    # 1. Inspect Tokenizer & Templates
    inspect_tokenizer_and_templates(tok)

    # 2. Run Comparative Diagnostics
    run_comparative_diagnostics(model, tok)

    # 3. Interactive prompt
    try:
        choice = input("\nWould you like to start interactive chat with native Qwen? [y/N]: ").strip().lower()
        if choice in ["y", "yes"]:
            start_interactive_session(model, tok)
    except (EOFError, KeyboardInterrupt):
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3 0.8B Native Diagnostic")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode directly")
    args, _ = parser.parse_known_args()

    if args.interactive:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        start_interactive_session(model, tok)
    else:
        main()
