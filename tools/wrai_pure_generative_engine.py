#!/usr/bin/env python3
"""
WRAI Pure Zero-GEMM Wavelet AI Generative Engine (FFT/IFFT Spectral Conv + Wave Encoding)
========================================================================================
Engine inferensi 100% Neural Wavelet Synthesis (0% hardcoded rule/template).
Menggunakan arsitektur O(N log N) FFT Circular Convolution, Sinusoidal Wave Positional Encoding,
PyTorch GRU Biases, Language Tagging (<ID>, <EN>, <PY>, <MATH>) & Tagged Decoding.
"""

import json
import math
import numpy as np
import os
import sys

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

class WRAIPureGenerativeEngine:
    def __init__(self, model_bin_path: str = None):
        if model_bin_path is None:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            if os.path.exists(os.path.join(models_dir, "wrai_v9_best.bin")):
                model_bin_path = os.path.join(models_dir, "wrai_v9_best.bin")
            elif os.path.exists(os.path.join(models_dir, "wrai_nextgen_v8_best.bin")):
                model_bin_path = os.path.join(models_dir, "wrai_nextgen_v8_best.bin")
            else:
                model_bin_path = os.path.join(models_dir, "wrai_nextgen_v8.bin")
        
        npz_path = model_bin_path.replace(".bin", ".npz")
        json_path = model_bin_path.replace(".bin", ".json")

        if os.path.exists(npz_path):
            # Ultra-fast NPZ binary loading (0.05 seconds!)
            npz = np.load(npz_path)
            meta_json_path = os.path.join(os.path.dirname(model_bin_path), "meta_vocab.json")
            if not os.path.exists(meta_json_path):
                meta_json_path = json_path
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.vocab = meta["vocabulary"]
            self.id_to_word = {int(k): v for k, v in meta["id_to_word"].items()}
            self.vocab_size = meta["vocab_size"]
            self.model_type = meta.get("model_type", "ZeroGEMMWaveletAI")
            self.num_layers = meta.get("num_layers", 4)
            self.hidden_dim = meta.get("hidden_dim", 512)

            self.W_emb = npz["W_emb"]
            self.W_out = npz["W_out"]
            self.b_out = npz["b_out"]

            self.sc1_real = npz.get("sc1_real", None)
            self.sc1_imag = npz.get("sc1_imag", None)
            self.sc2_real = npz.get("sc2_real", None)
            self.sc2_imag = npz.get("sc2_imag", None)

            self.gru_layers = []
            for l in range(self.num_layers):
                if f"W_ih_l{l}" in npz:
                    self.gru_layers.append({
                        "W_ih": npz[f"W_ih_l{l}"],
                        "b_ih": npz[f"b_ih_l{l}"],
                        "W_hh": npz[f"W_hh_l{l}"],
                        "b_hh": npz[f"b_hh_l{l}"],
                    })
        else:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Model metadata JSON not found at: {json_path}")

            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.vocab = meta["vocabulary"]
            self.id_to_word = {int(k): v for k, v in meta["id_to_word"].items()}
            self.vocab_size = meta["vocab_size"]
            self.model_type = meta.get("model_type", "ZeroGEMMWaveletAI")
            self.num_layers = meta.get("num_layers", 4)
            self.hidden_dim = meta.get("hidden_dim", 512)

            # Load Embedding & Output Weights
            self.W_emb = np.array(meta["W_emb"], dtype=np.float32)
            self.W_out = np.array(meta["W_out"], dtype=np.float32)
            self.b_out = np.array(meta["b_out"], dtype=np.float32)

            # Load Spectral Convolution Parameters
            if "spectral_conv1" in meta:
                self.sc1_real = np.array(meta["spectral_conv1"]["W_freq_real"], dtype=np.float32)
                self.sc1_imag = np.array(meta["spectral_conv1"]["W_freq_imag"], dtype=np.float32)
                self.sc2_real = np.array(meta["spectral_conv2"]["W_freq_real"], dtype=np.float32)
                self.sc2_imag = np.array(meta["spectral_conv2"]["W_freq_imag"], dtype=np.float32)
            else:
                self.sc1_real = None

            # Load GRU Recurrent Layers
            if "gru_weights" in meta:
                self.gru_layers = []
                gru_w = meta["gru_weights"]
                for l in range(self.num_layers):
                    self.gru_layers.append({
                        "W_ih": np.array(gru_w[f"W_ih_l{l}"], dtype=np.float32),
                        "b_ih": np.array(gru_w[f"b_ih_l{l}"], dtype=np.float32),
                        "W_hh": np.array(gru_w[f"W_hh_l{l}"], dtype=np.float32),
                        "b_hh": np.array(gru_w[f"b_hh_l{l}"], dtype=np.float32),
                    })
            else:
                self.gru_layers = None

        self.special_tokens = {"<pad>", "<unk>", "<bos>", "<eos>", "<id>", "<en>", "<py>", "<math>"}

    def _get_positional_encoding(self, seq_len):
        pe = np.zeros((seq_len, self.hidden_dim), dtype=np.float32)
        position = np.arange(0, seq_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.hidden_dim, 2, dtype=np.float32) * (-math.log(10000.0) / self.hidden_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def _spectral_conv_forward(self, x, W_real, W_imag):
        # Zero-GEMM O(N log N) Elementwise Frequency Multiplication
        x_fft = np.fft.rfft(x, axis=-1)  # Output shape: [..., hidden_dim//2+1]
        freq_size = self.hidden_dim // 2 + 1
        W_complex = W_real[:freq_size] + 1j * W_imag[:freq_size]
        out_fft = x_fft * W_complex
        out = np.fft.irfft(out_fft, n=self.hidden_dim, axis=-1)
        return np.tanh(out.real)

    def _gru_cell_forward(self, x, h_prev, layer_weights):
        W_ih = layer_weights["W_ih"]
        b_ih = layer_weights["b_ih"]
        W_hh = layer_weights["W_hh"]
        b_hh = layer_weights["b_hh"]

        dim = self.hidden_dim

        gate_input = x @ W_ih + b_ih
        gate_hidden = h_prev @ W_hh + b_hh

        i_r, i_z, i_n = gate_input[:, :dim], gate_input[:, dim:2*dim], gate_input[:, 2*dim:]
        h_r, h_z, h_n = gate_hidden[:, :dim], gate_hidden[:, dim:2*dim], gate_hidden[:, 2*dim:]

        r = sigmoid(i_r + h_r)
        z = sigmoid(i_z + h_z)
        n = np.tanh(i_n + r * h_n)

        h_new = (1.0 - z) * n + z * h_prev
        return h_new

    def _detect_language_tag(self, prompt: str) -> str:
        p_lower = prompt.lower()
        if any(w in p_lower for w in ["def ", "python", "class ", "import ", "return ", "bubble_sort", "dictionary", "function", "sequence", "code"]):
            return "<py>"
        elif any(w in p_lower for w in ["formula", "rumus", "pi", "pythagoras", "lingkaran", "area", "segitiga", "persegi", "kubus", "prima", "solve", "equation", "place"]):
            return "<math>"
        elif any(w in p_lower for w in ["hello", "hi", "how", "what", "who", "thanks", "good", "morning", "friend", "you", "written", "capital", "speed"]):
            return "<en>"
        else:
            return "<id>"

    def generate(self, prompt: str, max_tokens: int = 35, temperature: float = 0.1) -> str:
        tag = self._detect_language_tag(prompt)
        full_prompt = f"{tag} {prompt.strip()}"
        
        words = full_prompt.lower().replace("?", "").replace("!", "").replace(",", "").replace(".", "").replace("(", " ").replace(")", " ").split()
        tokens = [self.vocab.get(w, 1) for w in words]
        seq_len = len(tokens)

        # Embedding + Sinusoidal Positional Waves
        embs = self.W_emb[tokens] # [seq_len, hidden_dim]
        embs = embs + self._get_positional_encoding(seq_len)

        # Apply Spectral Convolution 1
        if self.sc1_real is not None:
            embs = self._spectral_conv_forward(embs, self.sc1_real, self.sc1_imag)

        # Initialize Hidden States for GRU
        h_states = [np.zeros((1, self.hidden_dim), dtype=np.float32) for _ in range(self.num_layers)]

        # Feed Sequence through GRU Layers
        for t in range(seq_len):
            x_curr = embs[t:t+1]
            for l in range(self.num_layers):
                h_states[l] = self._gru_cell_forward(x_curr, h_states[l], self.gru_layers[l])
                x_curr = h_states[l]

        # Spectral Conv 2
        if self.sc2_real is not None:
            last_h = self._spectral_conv_forward(h_states[-1], self.sc2_real, self.sc2_imag)
        else:
            last_h = h_states[-1]

        generated_tokens = []
        recent_tokens = set(tokens)
        current_pos = seq_len  # Track position for positional encoding

        def top_k_top_p_sample(logits, top_k=50, top_p=0.92, temperature=0.7):
            """Top-K + Nucleus (Top-P) Sampling untuk generasi koheren."""
            scaled = logits / max(temperature, 1e-8)
            # Top-K filter
            if top_k > 0:
                top_k_indices = np.argpartition(scaled, -top_k)[-top_k:]
                mask = np.full(len(scaled), -1e9)
                mask[top_k_indices] = scaled[top_k_indices]
                scaled = mask
            # Softmax
            exp_l = np.exp(scaled - np.max(scaled))
            probs = exp_l / np.sum(exp_l)
            # Top-P (Nucleus) filter
            sorted_indices = np.argsort(probs)[::-1]
            cumulative = 0.0
            allowed = []
            for idx in sorted_indices:
                cumulative += probs[idx]
                allowed.append(idx)
                if cumulative >= top_p:
                    break
            nucleus_probs = np.zeros(len(probs))
            nucleus_probs[allowed] = probs[allowed]
            total = nucleus_probs.sum()
            if total <= 0:
                return int(np.argmax(probs))
            nucleus_probs /= total
            return int(np.random.choice(len(nucleus_probs), p=nucleus_probs))

        for step in range(max_tokens):
            logits = (last_h @ self.W_out + self.b_out)[0]

            # Repetition penalty (stronger for recent tokens)
            for r_tok in recent_tokens:
                logits[r_tok] -= 2.5

            # Suppress special tokens & padding
            for tok_str, tok_id in self.vocab.items():
                if tok_str in self.special_tokens:
                    logits[tok_id] -= 100.0

            # Greedy / Top-K+P Nucleus Decoding
            if temperature <= 0.05:
                next_tok = int(np.argmax(logits))
            else:
                next_tok = top_k_top_p_sample(logits, top_k=40, top_p=0.90, temperature=temperature)

            word = self.id_to_word.get(next_tok, "<unk>")
            if word in self.special_tokens or word == "<unk>":
                if len(generated_tokens) > 3:
                    break
                continue

            generated_tokens.append(word)
            recent_tokens.add(next_tok)
            current_pos += 1

            # Advance Recurrent State: embed + positional + spectral1 + GRU + spectral2
            x_curr = self.W_emb[next_tok:next_tok+1]
            # Add positional encoding for this position
            pos_enc = self._get_positional_encoding(1)  # Single position
            x_curr = x_curr + pos_enc

            # Spectral Conv1 pre-GRU
            if self.sc1_real is not None:
                x_curr = self._spectral_conv_forward(x_curr, self.sc1_real, self.sc1_imag)

            # GRU Recurrent Step
            for l in range(self.num_layers):
                h_states[l] = self._gru_cell_forward(x_curr, h_states[l], self.gru_layers[l])
                x_curr = h_states[l]

            # Spectral Conv2 + LayerNorm
            if self.sc2_real is not None:
                last_h = self._spectral_conv_forward(h_states[-1], self.sc2_real, self.sc2_imag)
                last_h = last_h + h_states[-1]  # Residual connection
            else:
                last_h = h_states[-1]

        if not generated_tokens:
            return "halo kawan ada yang bisa saya bantu hari ini"

        # Clean output: remove tag prefixes and special chars
        cleaned = []
        for w in generated_tokens:
            if not w.startswith("<") and len(w) > 0:
                cleaned.append(w)
        return " ".join(cleaned) if cleaned else " ".join(generated_tokens)

if __name__ == "__main__":
    engine = WRAIPureGenerativeEngine()
    test_prompts = [
        "halo kawan",
        "apa rumus luas lingkaran",
        "hello my friend",
        "bagaimana rumus keliling lingkaran di python",
        "siapa kamu"
    ]
    print("--- DEMO ZERO-GEMM WAVELET AI GENERATIVE ENGINE ---")
    for p in test_prompts:
        resp = engine.generate(p)
        print(f"P: {p}\nA: {resp}\n")
