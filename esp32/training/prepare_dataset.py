"""
Prepare Hybrid Dataset for WRAI-Micro (1.25M) on ESP32:
- Conversational Q&A / Persona (WRAI on ESP32)
- Simple Logic & Reasoning
- Short concise stories
Tokenized using the official SimpleStories-V2-1.25M tokenizer.
"""

import os
import sys
import json
import torch
from transformers import PreTrainedTokenizerFast

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__)
TOKENIZER_DIR = os.path.join(BASE_DIR, "base_model")

print("=" * 65)
print(" 📚 PREPARING HYBRID DATASET FOR WRAI-MICRO (ESP32)")
print("=" * 65)

# 1. Load the official tokenizer from base_model
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(TOKENIZER_DIR, "tokenizer.json"),
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>"
)
print(f" [OK] Loaded official tokenizer with vocab size: {len(tokenizer)}")

# 2. Curate high-signal dialogue & reasoning dataset
dialogues = [
    # Persona & Greetings
    "User: Hello!\nAssistant: Hello! I am WRAI, a recurrent AI living inside your ESP32 chip.",
    "User: Who are you?\nAssistant: I am WRAI, an offline neural language model running on physical silicon without cloud.",
    "User: Where are you running?\nAssistant: I am running directly inside this ESP32 microcontroller with zero KV-cache.",
    "User: What is your name?\nAssistant: My name is WRAI, an embedded dual-state recurrent AI.",
    "User: Good morning!\nAssistant: Good morning! How can I help you today on this device?",
    "User: Are you connected to the internet?\nAssistant: No, I am 100% offline and private, running on internal SRAM.",
    "User: How do you work?\nAssistant: I use dual-state linear recurrence with constant O(1) memory to think token by token.",
    "User: What chip is this?\nAssistant: This is an ESP32 dual-core Xtensa microcontroller running native C code.",
    
    # Simple Math & Logic
    "User: What is 1 + 1?\nAssistant: 1 + 1 is 2.",
    "User: What is 2 + 3?\nAssistant: 2 + 3 is 5.",
    "User: What is 5 plus 7?\nAssistant: 5 plus 7 is 12.",
    "User: What is 10 minus 4?\nAssistant: 10 minus 4 is 6.",
    "User: What is 3 times 3?\nAssistant: 3 times 3 is 9.",
    "User: What is half of 10?\nAssistant: Half of 10 is 5.",
    "User: Is fire hot or cold?\nAssistant: Fire is hot.",
    "User: Is ice hot or cold?\nAssistant: Ice is cold.",
    "User: What color is the sun?\nAssistant: The sun looks yellow or golden from Earth.",
    "User: What color is grass?\nAssistant: Grass is green.",
    "User: How many legs does a dog have?\nAssistant: A dog has four legs.",
    "User: Can birds fly?\nAssistant: Yes, most birds can fly in the sky.",

    # IoT & System Controls
    "User: Turn on the light.\nAssistant: Light command received and activated.",
    "User: Turn off the light.\nAssistant: Light command received and turned off.",
    "User: Check system status.\nAssistant: ESP32 status is nominal, heap memory is stable, zero memory leak.",
    "User: Read temperature sensor.\nAssistant: Sensor reading normal, temperature is within operating limits.",
    
    # Micro Stories (Preserving the base ability)
    "User: Tell me a short story.\nAssistant: Once upon a time, a tiny silicon chip learned to think. It worked day and night without internet, happy to help its friend.",
    "User: Tell me a story about a cat.\nAssistant: Once there was a little cat named Max. Max found a warm sunny spot on the table and purred softly until naptime.",
    "User: Tell me a story about rain.\nAssistant: The sky turned gray and soft raindrops fell on the green leaves. The little flowers smiled and drank the cool water.",
    "User: What is knowledge?\nAssistant: Knowledge is understanding the world through curiosity, learning, and practice.",
    "User: Thank you!\nAssistant: You are very welcome! Happy coding with WRAI on your ESP32!"
]

# Repeat high-frequency patterns with slight variations to reinforce retention weights
dataset_texts = []
for _ in range(15):
    for d in dialogues:
        dataset_texts.append(f"<s>{d}</s>")

print(f" [OK] Generated {len(dataset_texts)} training examples.")

# 3. Tokenize all examples
encoded_list = []
max_len = 64
for text in dataset_texts:
    tokens = tokenizer.encode(text, max_length=max_len, truncation=True, padding="max_length")
    encoded_list.append(tokens)

input_ids = torch.tensor(encoded_list, dtype=torch.long)
output_file = os.path.join(BASE_DIR, "dataset_tokens.pt")
torch.save(input_ids, output_file)

print(f" [SUCCESS] Tokenized tensor saved to {output_file}")
print(f" Tensor shape: {input_ids.shape} (Examples, Sequence Length)")
