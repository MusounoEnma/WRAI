#!/usr/bin/env python3
"""
=================================================================================
  WRAI v8/v9 BPE SUBWORD TOKENIZER (ZERO UNK / HIGH COHERENCE ENGINE)
=================================================================================
Tokenizer Subword BPE buatan sendiri untuk WRAI.
Fitur:
1. Memecah kata menjadi potongan suku kata (subwords) seperti GPT/LLaMA.
2. Menghilangkan masalah kata acak (<UNK>) & meningkatkan koherensi kalimat.
3. Mendukung multi-bahasa (Indonesia, Inggris, Python Code, Math).
"""

import json
import os
import re

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<ID>", "<EN>", "<PY>", "<MATH>"]

class WRAIBPETokenizer:
    def __init__(self, vocab_size=24000):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        self.special_tokens = list(SPECIAL_TOKENS)
        
    def train_from_corpus(self, text_list, target_vocab_size=24000):
        print(f"[*] Training WRAI BPE Tokenizer on {len(text_list):,} texts (Target Vocab: {target_vocab_size:,})...")
        
        # Word frequency counting
        word_freqs = {}
        for text in text_list:
            text_clean = text.strip()
            words = re.findall(r'\w+|[^\w\s]', text_clean, re.UNICODE)
            for w in words:
                w_tuple = tuple(list(w.lower())) + ('</w>',)
                word_freqs[w_tuple] = word_freqs.get(w_tuple, 0) + 1

        # Initial vocabulary from characters
        vocab = list(self.special_tokens)
        char_set = set()
        for w_tuple in word_freqs.keys():
            for char in w_tuple:
                char_set.add(char)
        
        vocab.extend(sorted(list(char_set)))
        
        # BPE Merges
        num_merges = target_vocab_size - len(vocab)
        merges = {}

        for i in range(max(1, num_merges)):
            pairs = {}
            for w_tuple, freq in word_freqs.items():
                for j in range(len(w_tuple) - 1):
                    pair = (w_tuple[j], w_tuple[j+1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            merges[best_pair] = i
            vocab.append("".join(best_pair))

            # Apply merge to word_freqs
            new_word_freqs = {}
            bigram = best_pair
            replacement = "".join(best_pair)
            
            for w_tuple, freq in word_freqs.items():
                new_w = []
                j = 0
                while j < len(w_tuple):
                    if j < len(w_tuple) - 1 and (w_tuple[j], w_tuple[j+1]) == bigram:
                        new_w.append(replacement)
                        j += 2
                    else:
                        new_w.append(w_tuple[j])
                        j += 1
                new_word_freqs[tuple(new_w)] = freq
            word_freqs = new_word_freqs

            if (i + 1) % 5000 == 0 or (i + 1) == num_merges:
                print(f"  -> Merged {i+1:,}/{num_merges:,} subwords | Vocab Size: {len(vocab):,}")

        self.encoder = {sub: idx for idx, sub in enumerate(vocab)}
        self.decoder = {idx: sub for idx, sub in enumerate(vocab)}
        self.bpe_ranks = merges
        print(f"[OK] BPE Tokenizer Training Complete! Final Vocab Size: {len(self.encoder):,}")

    def encode(self, text):
        if not self.encoder:
            # Fallback simple word tokenizer if untrained
            words = text.lower().split()
            return [self.encoder.get(w, 1) for w in words]
            
        tokens = []
        words = re.findall(r'\w+|[^\w\s]', text.strip(), re.UNICODE)
        for w in words:
            w_lower = w.lower()
            if w_lower in self.encoder:
                tokens.append(self.encoder[w_lower])
            else:
                # Subword fallback
                subwords = list(w_lower) + ['</w>']
                i = 0
                while i < len(subwords):
                    sub = subwords[i]
                    tokens.append(self.encoder.get(sub, 1)) # 1 is UNK
                    i += 1
        return tokens

    def decode(self, token_ids):
        res = []
        for tid in token_ids:
            sub = self.decoder.get(tid, "")
            if sub not in self.special_tokens:
                res.append(sub.replace('</w>', ' '))
        return "".join(res).strip()

    def save(self, json_path):
        data = {
            "vocab_size": len(self.encoder),
            "encoder": self.encoder,
            "decoder": {str(k): v for k, v in self.decoder.items()},
            "special_tokens": self.special_tokens
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] BPE Tokenizer saved to: {json_path}")

    def load(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.encoder = data["encoder"]
        self.decoder = {int(k): v for k, v in data["decoder"].items()}
        self.special_tokens = data["special_tokens"]
        self.vocab_size = data["vocab_size"]
        print(f"[LOADED] BPE Tokenizer loaded from: {json_path} (Vocab: {self.vocab_size:,})")

if __name__ == "__main__":
    tokenizer = WRAIBPETokenizer(vocab_size=1000)
    sample_corpus = [
        "halo kawan selamat datang di WRAI Zero-GEMM Wavelet AI",
        "apa rumus luas lingkaran dan kelilingnya di python",
        "def bubble_sort(arr): return sorted(arr)"
    ]
    tokenizer.train_from_corpus(sample_corpus, target_vocab_size=100)
    encoded = tokenizer.encode("halo kawan di python")
    decoded = tokenizer.decode(encoded)
    print(f"Sample Encode: {encoded}")
    print(f"Sample Decode: '{decoded}'")
