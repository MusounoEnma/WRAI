#!/usr/bin/env python3
"""
WRAI Subword BPE Decoder
Converts GPT-2/LLaMA/Tiktoken ByteBPE raw tokens containing 'Ġ' into clean spaces and human-readable text.
Automatically adds space separators for plain word token sequences.
"""

import re
import sys

class WRAIBPEDecoder:
    def __init__(self):
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def _bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def decode_token(self, token_str: str) -> str:
        """Converts a single BPE subword token into clean text string."""
        if not token_str:
            return ""
        
        # Replace BPE space and newline characters
        cleaned = token_str.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\n")

        # Decode byte-level unicode characters
        byte_list = bytearray()
        for char in cleaned:
            if char in self.byte_decoder:
                byte_list.append(self.byte_decoder[char])
            else:
                try:
                    byte_list.extend(char.encode('utf-8'))
                except Exception:
                    pass

        try:
            return byte_list.decode('utf-8', errors='replace')
        except Exception:
            return cleaned

    def decode_tokens_list(self, token_list) -> str:
        """Decodes a list of subword tokens into a clean human sentence."""
        if not token_list:
            return ""
        
        has_bpe_markers = any("Ġ" in str(t) for t in token_list)

        if has_bpe_markers:
            raw_joined = "".join([self.decode_token(str(t)) for t in token_list])
        else:
            raw_joined = " ".join([self.decode_token(str(t)) for t in token_list])
        
        # Clean up multi-spaces and leading/trailing whitespace
        clean_text = re.sub(r'\s+', ' ', raw_joined).strip()
        
        # Capitalize first letter
        if clean_text:
            clean_text = clean_text[0].upper() + clean_text[1:]
        
        return clean_text

if __name__ == "__main__":
    decoder = WRAIBPEDecoder()
    raw_samples = ["halo", "kawan", "selamat", "datang"]
    print("Decoded text:", decoder.decode_tokens_list(raw_samples))
