#!/usr/bin/env python3
"""
Debug Forward Pass script to inspect exact neural logits.
"""

import numpy as np
from wrai_pure_generative_engine import WRAINextGenEngineV8, softmax

engine = WRAINextGenEngineV8()

word = "kawan"
token_id = engine.vocab[word]
print(f"Token ID for '{word}': {token_id}")

h = engine.W_emb[token_id]
logits = h @ engine.W_out + engine.b_out
probs = softmax(logits)

top5_ids = np.argsort(probs)[-5:][::-1]
print(f"Top 5 predicted next tokens after '{word}':")
for tid in top5_ids:
    w = engine.id_to_word.get(tid, "")
    print(f"  -> ID {tid:3d} ('{w}'): Prob = {probs[tid]*100.0:.2f}%")
