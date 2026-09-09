#!/usr/bin/env python3
"""
=================================================================================
  WRAI v11 GEMMA 4 LOCAL CPU INFERENCE TESTER
=================================================================================
Menjalankan inferensi lokal WRAI v11 (Gemma 4 Transmuted 300M) di CPU Laptop.
Menggunakan cache biner .npz hasil ekspor dari Google Colab.
"""

import json
import math
import os
import sys
import time
import numpy as np

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

class WRAIv11CPUInferenceEngine:
    def __init__(self, npz_path, meta_path):
        print(f"[*] Loading WRAI v11 Gemma 4 Model from NPZ Cache: {npz_path}...")
        t0 = time.perf_counter()
        
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
            
        self.vocab_size = self.meta["vocab_size"]
        self.hidden_dim = self.meta["hidden_dim"]
        self.num_layers = self.meta["num_layers"]
        
        self.word_to_id = self.meta["vocabulary"]
        self.id_to_word = {int(i): w for i, w in self.meta["id_to_word"].items()}
        
        self.weights = np.load(npz_path)
        self.W_emb = self.weights["W_emb"]
        self.W_out = self.weights["W_out"]
        self.b_out = self.weights["b_out"]
        
        self.sc1_real = self.weights["sc1_real"]
        self.sc1_imag = self.weights["sc1_imag"]
        self.sc2_real = self.weights["sc2_real"]
        self.sc2_imag = self.weights["sc2_imag"]
        
        self.gru_weights = {}
        for l in range(self.num_layers):
            self.gru_weights[f"W_ih_l{l}"] = self.weights[f"W_ih_l{l}"]
            self.gru_weights[f"b_ih_l{l}"] = self.weights[f"b_ih_l{l}"]
            self.gru_weights[f"W_hh_l{l}"] = self.weights[f"W_hh_l{l}"]
            self.gru_weights[f"b_hh_l{l}"] = self.weights[f"b_hh_l{l}"]
            
        t1 = time.perf_counter()
        print(f"[OK] WRAI v11 Model Loaded in {(t1-t0):.2f}s! (Hidden: {self.hidden_dim}, Layers: {self.num_layers})")

    def _spectral_conv(self, x, real, imag):
        x_fft = np.fft.rfft(x, axis=-1)
        w_complex = real + 1j * imag
        out_fft = x_fft * w_complex
        out = np.fft.irfft(out_fft, n=self.hidden_dim, axis=-1)
        return np.tanh(out)

    def _encode_bpe(self, text):
        """Greedy Subword BPE Encoder matching model vocabulary."""
        text_clean = text.strip()
        tokens = []
        words = text_clean.split()
        for w in words:
            # Try full word with BPE end of word suffix
            w_w = w + "</w>"
            if w_w in self.word_to_id:
                tokens.append(self.word_to_id[w_w])
            elif w in self.word_to_id:
                tokens.append(self.word_to_id[w])
            else:
                # Subword decomposition
                i = 0
                while i < len(w):
                    matched = False
                    for j in range(len(w), i, -1):
                        sub = w[i:j]
                        if j == len(w):
                            sub_w = sub + "</w>"
                            if sub_w in self.word_to_id:
                                tokens.append(self.word_to_id[sub_w])
                                i = j
                                matched = True
                                break
                        if sub in self.word_to_id:
                            tokens.append(self.word_to_id[sub])
                            i = j
                            matched = True
                            break
                    if not matched:
                        tokens.append(self.word_to_id.get("<UNK>", 1))
                        i += 1
        return tokens if tokens else [1]

    def generate(self, prompt, max_new_tokens=40, temperature=0.7, top_k=40):
        tokens = self._encode_bpe(prompt)
            
        print(f"\n[PROMPT] {prompt}")
        print(f"[TOKENS] {tokens}")
        print(f"[GENERATING] ...", end="", flush=True)
        
        generated = list(tokens)
        h = np.zeros((self.num_layers, self.hidden_dim), dtype=np.float32)
        
        for _ in range(max_new_tokens):
            curr_id = generated[-1]
            emb = self.W_emb[curr_id]
            feat1 = self._spectral_conv(emb, self.sc1_real, self.sc1_imag)
            
            curr_in = feat1
            for l in range(self.num_layers):
                W_ih = self.gru_weights[f"W_ih_l{l}"]
                b_ih = self.gru_weights[f"b_ih_l{l}"]
                W_hh = self.gru_weights[f"W_hh_l{l}"]
                b_hh = self.gru_weights[f"b_hh_l{l}"]
                
                gate_x = np.dot(curr_in, W_ih) + b_ih
                gate_h = np.dot(h[l], W_hh) + b_hh
                
                r_x, z_x, n_x = np.split(gate_x, 3)
                r_h, z_h, n_h = np.split(gate_h, 3)
                
                r = 1.0 / (1.0 + np.exp(-np.clip(r_x + r_h, -15, 15)))
                z = 1.0 / (1.0 + np.exp(-np.clip(z_x + z_h, -15, 15)))
                n = np.tanh(n_x + r * n_h)
                
                h[l] = (1.0 - z) * n + z * h[l]
                curr_in = h[l]
                
            feat2 = self._spectral_conv(curr_in, self.sc2_real, self.sc2_imag)
            logits = np.dot(feat2, self.W_out) + self.b_out
            
            # Top-K Sampling
            top_k_indices = np.argsort(logits)[-top_k:]
            top_k_logits = logits[top_k_indices] / temperature
            top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
            top_k_probs /= np.sum(top_k_probs)
            
            next_id = np.random.choice(top_k_indices, p=top_k_probs)
            generated.append(next_id)
            
            word = self.id_to_word.get(next_id, f"<token_{next_id}>")
            clean_word = word.replace("</w>", " ").replace(" ", " ")
            print(f" {clean_word}", end="", flush=True)
            if word in ["<EOS>", "<eos>"]:
                break
                
        print("\n")
        words = [self.id_to_word.get(i, f"").replace("</w>", " ") for i in generated]
        full_text = "".join(words)
        return full_text

def main():
    npz_path = "models/wrai_v11_gemma4_best.npz" if os.path.exists("models/wrai_v11_gemma4_best.npz") else "wrai_v11_gemma4_best.npz"
    meta_path = "models/wrai_v11_gemma4_best_meta.json" if os.path.exists("models/wrai_v11_gemma4_best_meta.json") else "wrai_v11_gemma4_best_meta.json"
    
    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        print(f"[NOTE] File {npz_path} belum ditemukan di folder lokal.")
        print(f"Silakan download 2 file ini dari Google Drive (/MyDrive/WRAI_Models/):")
        print(f" 1. {npz_path}")
        print(f" 2. {meta_path}")
        print(f"Lalu masukkan ke folder: c:\\porto\\11MYPORTO\\WRAI\\models\\")
        return
        
    engine = WRAIv11CPUInferenceEngine(npz_path, meta_path)
    prompts = [
        "<ID> Apa itu kecerdasan buatan",
        "<ID> Sebutkan makanan khas Indonesia",
        "<PY> def hitung_diskon(harga, persen):",
        "<MATH> Berapa hasil 50 dikali 4",
    ]
    for p in prompts:
        engine.generate(p, max_new_tokens=35)

if __name__ == "__main__":
    main()
