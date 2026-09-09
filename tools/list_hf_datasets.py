#!/usr/bin/env python3
"""
WRAI Hugging Face Open-Source Dataset Directory & Inspector Tool
Lists and inspects large open-source datasets on Hugging Face Hub ready for WRAI synthesis.
"""

import json
import urllib.request

HF_DATASET_CATALOG = [
    {
        "name": "databricks/databricks-dolly-15k",
        "category": "Instruction & General Knowledge",
        "num_items": "15,000 Items",
        "description": "Dataset instruksi manusia buatan Databricks meliputi Q&A, ekstraksi informasi, dan penalaran.",
        "hf_url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k"
    },
    {
        "name": "tatsu-lab/alpaca",
        "category": "Complex Instruction & Science",
        "num_items": "52,000 Items",
        "description": "Dataset instruksi Stanford Alpaca untuk pengujian pengetahuan dan penalaran kompleks.",
        "hf_url": "https://huggingface.co/datasets/tatsu-lab/alpaca"
    },
    {
        "name": "gsm8k",
        "category": "Math & Chain-of-Thought Logic",
        "num_items": "8,500 Items",
        "description": "Soal matematika berantai (Grade School Math) dengan langkah-langkah penalaran CoT.",
        "hf_url": "https://huggingface.co/datasets/gsm8k"
    },
    {
        "name": "allenai/sciq",
        "category": "Science & Physics/Biochemistry",
        "num_items": "13,679 Items",
        "description": "Dataset soal sains (Fisika, Biologi, Kimia) dari Allen Institute dengan konteks penjelasan.",
        "hf_url": "https://huggingface.co/datasets/allenai/sciq"
    },
    {
        "name": "cais/mmlu",
        "category": "Multi-Domain Academic Exam",
        "num_items": "15,908 Items",
        "description": "Soal ujian akademik 57 mata pelajaran (Fisika Kuantum, Kedokteran, Komputer, Hukum).",
        "hf_url": "https://huggingface.co/datasets/cais/mmlu"
    },
    {
        "name": "medmcqa",
        "category": "Medical & Biological Science",
        "num_items": "194,000+ Items",
        "description": "Dataset ujian medis profesional dengan penjelasan ilmiah anatomi, farmakologi, & imunologi.",
        "hf_url": "https://huggingface.co/datasets/medmcqa"
    },
    {
        "name": "Open-Orca/OpenOrca",
        "category": "Ultra-Large Reasoning Knowledge",
        "num_items": "3,500,000+ Items",
        "description": "Dataset penalaran dan pengetahuan umum skala besar dari riset Microsoft Orca.",
        "hf_url": "https://huggingface.co/datasets/Open-Orca/OpenOrca"
    },
    {
        "name": "OpenAssistant/oasst1",
        "category": "Multilingual Conversation & Chat",
        "num_items": "161,443 Messages",
        "description": "Dataset percakapan multiturn dalam berbagai bahasa manusia.",
        "hf_url": "https://huggingface.co/datasets/OpenAssistant/oasst1"
    }
]

def main():
    print("=================================================================")
    print("  KATALOG DATASET OPEN-SOURCE HUGGING FACE PILIHAN UNTUK WRAI    ")
    print("=================================================================\n")

    for idx, ds in enumerate(HF_DATASET_CATALOG, start=1):
        print(f"[{idx}] {ds['name']}")
        print(f"    * Kategori  : {ds['category']}")
        print(f"    * Skala Data: {ds['num_items']}")
        print(f"    * Deskripsi : {ds['description']}")
        print(f"    * HF URL    : {ds['hf_url']}")
        print("-" * 65)

    print("\n[INFO] Semua dataset di atas dapat diimpor langsung ke WRAI via tools/import_dataset.py!")

if __name__ == "__main__":
    main()
