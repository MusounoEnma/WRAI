#!/usr/bin/env python3
"""
WRAI Subword / BPE Tokenizer
Handles Indonesian vocabulary prefixes, suffixes, and root words with fixed-point IDs.
"""

import json
import os
import re

class WRAISubwordTokenizer:
    def __init__(self, vocab_list=None):
        if vocab_list is None:
            self.vocab = [
                "<PAD>", "<UNK>", "<BOS>", "<EOS>",
                "halo", " hai", " salam", " kawan", " senang", " bertemu", " dengan", " anda",
                " saya", " adalah", " wrai", " asisten", " ai", " cerdas", " berbasis", " gelombang",
                " spektral", " fixed-point", " zero-gemm", " siap", " membantu", " anda", " hari", " ini",
                " kabar", " baik", " sehat", " selalu", " ya", " terima", " kasih", " sama-sama",
                " matematika", " aljabar", " geometri", " kalkulus", " sains", " fisika", " kimia",
                " beroperasi", " efisien", " tanpa", " perkalian", " matriks", " ram", " sram", "16", "kb",
                "bagaimana", "bisa", "apa", "siapa", "mengapa", "dimana", "tentu", "jelas", "mari",
                "belajar", "mengerjakan", "soal", "persamaan", "sampai", "jumpa", "kembali"
            ]
        else:
            self.vocab = vocab_list

        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_word = {i: w for w, i in enumerate(self.vocab)}

    def encode(self, text: str):
        """Simple greedy subword matching tokenizer."""
        text_clean = text.lower().strip()
        tokens = []
        words = text_clean.split()
        for i, w in enumerate(words):
            word_str = (" " if i > 0 else "") + w
            if word_str in self.word_to_id:
                tokens.append(self.word_to_id[word_str])
            elif w in self.word_to_id:
                tokens.append(self.word_to_id[w])
            else:
                # Subword split fallback
                sub_matched = False
                for sub in self.vocab:
                    if len(sub.strip()) > 2 and sub.strip() in w:
                        tokens.append(self.word_to_id[sub])
                        sub_matched = True
                        break
                if not sub_matched:
                    tokens.append(self.word_to_id.get("<UNK>", 1))
        return tokens

    def decode(self, token_ids):
        """Decode token IDs back to human-readable string."""
        words = []
        for tid in token_ids:
            w = self.id_to_word.get(tid, "")
            if w not in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"):
                words.append(w)
        result = "".join(words).strip()
        return result

if __name__ == "__main__":
    tok = WRAISubwordTokenizer()
    encoded = tok.encode("halo kawan senang bertemu")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tok.decode(encoded)}")
