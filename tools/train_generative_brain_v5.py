#!/usr/bin/env python3
"""
WRAI Generative Corpus Training Tool & Model Builder v5 (TRUE AI)
Trains Intent Classification Centroids + Chain of Thought Spectral Cascade.
- Categorizes training corpus into Intent Clusters (greeting, math, tech, general).
- Computes Q15 Spectral Centroid per Intent Category from real corpus statistics.
- Computes Bigram Transition Probabilities P(w_{k+1} | w_k) PER INTENT.
- Outputs production model: models/wrai_brain_v5.json
"""

import collections
import json
import math
import os
import struct
import sys
import time

LUT_SIZE = 512
FFT_SIZE = 512
SPECTRAL_BINS = 256

def float_to_q15(v: float) -> int:
    val = round(v * 32768.0)
    return max(-32768, min(32767, val))

def fnv1a(s: str) -> int:
    h = 2166136261
    for char in s.encode('utf-8'):
        h ^= char
        h = (h * 16777619) & 0xFFFFFFFF
    return h

SIN_LUT = [float_to_q15(math.sin(2.0 * math.pi * i / LUT_SIZE)) for i in range(LUT_SIZE)]

# =================================================================
# INTENT-CATEGORIZED INDONESIAN TRAINING CORPUS
# Each entry: (intent_category, sentence)
# =================================================================
CATEGORIZED_CORPUS = [
    # INTENT: greeting (salam & sapaan)
    ("greeting", "halo kawan selamat datang senang bertemu dengan anda"),
    ("greeting", "salam hangat kawan apa kabar hari ini"),
    ("greeting", "hai kawan saya senang bisa menyapa anda hari ini"),
    ("greeting", "halo selamat pagi semoga hari anda menyenangkan"),
    ("greeting", "salam kenal kawan perkenalkan saya wrai asisten anda"),
    ("greeting", "hai salam hangat senang sekali bisa berdiskusi bersama anda"),
    ("greeting", "halo kawan ada yang bisa saya bantu hari ini"),
    ("greeting", "selamat siang kawan semoga aktivitas anda berjalan lancar"),
    ("greeting", "hai senang bertemu anda kawan mari kita mengobrol"),
    ("greeting", "halo apa kabar kawan saya harap anda sehat selalu"),
    ("greeting", "salam kawan terima kasih sudah menyapa saya hari ini"),
    ("greeting", "hai kawan selamat datang kembali mari kita lanjutkan percakapan"),
    ("greeting", "halo kawan senang sekali bisa membantu anda hari ini"),
    ("greeting", "salam hangat dari saya untuk anda kawan semoga baik baik saja"),
    ("greeting", "hai kawan bagaimana kabar anda hari ini semoga sehat selalu"),

    # INTENT: farewell (perpisahan)
    ("farewell", "sampai jumpa kawan semoga hari anda menyenangkan"),
    ("farewell", "terima kasih kawan sampai bertemu lagi nanti"),
    ("farewell", "selamat tinggal kawan senang bisa berdiskusi dengan anda"),
    ("farewell", "sampai nanti kawan jaga kesehatan selalu"),
    ("farewell", "terima kasih sudah mengobrol kawan sampai jumpa lagi"),
    ("farewell", "baik kawan saya akan selalu siap membantu kapan saja"),

    # INTENT: identity (pertanyaan tentang identitas WRAI)
    ("identity", "saya adalah wrai asisten kecerdasan buatan berbasis gelombang spektral"),
    ("identity", "nama saya wrai wavelet resonance artificial intelligence"),
    ("identity", "saya wrai sistem kecerdasan buatan yang mengolah bahasa dengan sinyal gelombang"),
    ("identity", "saya adalah asisten ai yang dirancang khusus tanpa perkalian matriks berat"),
    ("identity", "wrai adalah arsitektur ai spektral non transformer yang sangat efisien"),
    ("identity", "saya diciptakan menggunakan metode wavelet resonance tanpa transformer"),
    ("identity", "saya bukan transformer saya adalah ai berbasis gelombang spektral fixed point"),
    ("identity", "wrai singkatan dari wavelet resonance artificial intelligence"),

    # INTENT: math (matematika & logika)
    ("math", "saya siap membantu anda mengerjakan soal matematika dan logika"),
    ("math", "mari kita kerjakan persamaan matematika ini bersama sama kawan"),
    ("math", "rumus luas lingkaran adalah pi dikali jari jari kuadrat"),
    ("math", "teorema pythagoras menyatakan kuadrat sisi miring sama dengan jumlah kuadrat kedua sisi"),
    ("math", "untuk menghitung persamaan linier kita perlu mencari nilai variabel yang memenuhi"),
    ("math", "matematika adalah bahasa alam semesta yang mengungkap pola tersembunyi"),
    ("math", "aljabar membantu kita menyelesaikan persamaan dengan variabel yang belum diketahui"),
    ("math", "geometri mempelajari bentuk ukuran dan posisi objek dalam ruang"),
    ("math", "kalkulus adalah cabang matematika yang mempelajari perubahan dan laju"),
    ("math", "statistika membantu kita menganalisis data dan membuat kesimpulan dari informasi"),
    ("math", "trigonometri mempelajari hubungan antara sisi dan sudut segitiga"),
    ("math", "bilangan prima hanya bisa dibagi satu dan dirinya sendiri"),

    # INTENT: tech (teknologi & hardware WRAI)
    ("tech", "wrai beroperasi murni zero gemm tanpa perkalian matriks floating point"),
    ("tech", "sistem ini menggunakan aritmatika fixed point q15 integer enam belas bit"),
    ("tech", "target hardware utama adalah mikrokontroler esp32 stm32 dan cortex m"),
    ("tech", "model disimpan dalam format biner dan distream via double ring buffer dma"),
    ("tech", "alokasi ram sram tetap statis pada enam belas kilobyte tanpa kebocoran heap"),
    ("tech", "pencarian spektral menggunakan fft radix dua lima ratus dua belas titik"),
    ("tech", "kompleksitas waktu inferensi wrai adalah konstan tanpa bergantung ukuran model"),
    ("tech", "wrai tidak membutuhkan gpu atau tpu untuk berjalan cukup cpu biasa"),
    ("tech", "arsitektur ini dirancang untuk perangkat edge computing dan iot"),

    # INTENT: science (sains umum)
    ("science", "sains adalah proses sistematis untuk memahami alam semesta melalui pengamatan dan eksperimen"),
    ("science", "fisika mempelajari hukum dasar alam semesta dari partikel subatom hingga galaksi"),
    ("science", "kimia mempelajari komposisi struktur dan perubahan materi"),
    ("science", "biologi mempelajari makhluk hidup dan proses kehidupan"),
    ("science", "astronomi mempelajari benda langit dan fenomena luar angkasa"),
    ("science", "ekologi mempelajari hubungan antara organisme dan lingkungannya"),

    # INTENT: general (percakapan umum)
    ("general", "tentu saya bisa membantu anda dengan pertanyaan tersebut kawan"),
    ("general", "pertanyaan yang menarik mari kita bahas bersama sama"),
    ("general", "saya akan berusaha menjawab sebaik mungkin berdasarkan pengetahuan saya"),
    ("general", "mari kita diskusikan topik ini lebih mendalam kawan"),
    ("general", "baik kawan saya akan menjelaskan dengan bahasa yang mudah dipahami"),
    ("general", "itu pertanyaan bagus kawan izinkan saya menjelaskan"),
    ("general", "saya memahami pertanyaan anda kawan berikut penjelasan saya"),
    ("general", "dengan senang hati saya akan membantu menjawab pertanyaan anda"),
]

def compute_sentence_spectral(sentence_words):
    """Compute 256-bin spectral signature for a sentence."""
    r = [0] * FFT_SIZE
    num_words = max(1, len(sentence_words))
    scale = 0
    temp = num_words
    while temp > 1:
        scale += 1
        temp >>= 1

    for n in range(FFT_SIZE):
        acc = 0
        for w in sentence_words:
            h = fnv1a(w)
            freq = (h % (SPECTRAL_BINS - 2)) + 1
            phase = (h >> 8) % LUT_SIZE
            lut_idx = (freq * n + phase) & (LUT_SIZE - 1)
            acc += SIN_LUT[lut_idx]
        r[n] = acc >> scale
    
    # Simple magnitude approximation for first SPECTRAL_BINS
    mags = [abs(r[m]) for m in range(SPECTRAL_BINS)]
    return mags

def main():
    print("=================================================================")
    print("  WRAI GENERATIVE CORPUS TRAINING V5 (INTENT + CHAIN OF THOUGHT)")
    print("=================================================================\n")

    # 1. Extract all intent categories
    intent_categories = sorted(set(cat for cat, _ in CATEGORIZED_CORPUS))
    print(f"[*] Intent Categories: {intent_categories}")
    print(f"[*] Total Training Sentences: {len(CATEGORIZED_CORPUS)}")

    # 2. Compute Intent Centroids (average spectral signature per category)
    intent_centroids = {}
    intent_sentences = {cat: [] for cat in intent_categories}
    
    for cat, sentence in CATEGORIZED_CORPUS:
        words = sentence.lower().split()
        intent_sentences[cat].append(words)

    for cat in intent_categories:
        centroid = [0] * SPECTRAL_BINS
        count = len(intent_sentences[cat])
        for words in intent_sentences[cat]:
            mags = compute_sentence_spectral(words)
            for m in range(SPECTRAL_BINS):
                centroid[m] += mags[m]
        # Average
        for m in range(SPECTRAL_BINS):
            centroid[m] = centroid[m] // max(1, count)
        intent_centroids[cat] = centroid
        print(f"  [CENTROID] {cat}: {count} sentences, peak energy = {max(centroid)}")

    # 3. Build per-intent vocabulary & bigram transitions
    global_word_counts = collections.Counter()
    intent_bigrams = {cat: collections.defaultdict(collections.Counter) for cat in intent_categories}
    intent_word_sets = {cat: set() for cat in intent_categories}

    for cat, sentence in CATEGORIZED_CORPUS:
        words = sentence.lower().split()
        for w in words:
            global_word_counts[w] += 1
            intent_word_sets[cat].add(w)
        for i in range(len(words) - 1):
            intent_bigrams[cat][words[i]][words[i+1]] += 1

    # Global vocabulary (top 2048)
    all_words = [w for w, _ in global_word_counts.most_common(2048)]
    word_to_id = {w: idx + 1 for idx, w in enumerate(all_words)}
    vocab_size = len(all_words)
    print(f"\n[*] Global Vocabulary Size: {vocab_size}")

    # 4. Compute co-occurrence based spectral embeddings
    co_occurrences = collections.defaultdict(collections.Counter)
    for cat, sentence in CATEGORIZED_CORPUS:
        words = sentence.lower().split()
        for i, w in enumerate(words):
            window = words[max(0, i-3) : min(len(words), i+4)]
            for w_co in window:
                if w != w_co:
                    co_occurrences[w][w_co] += 1

    vocab_entries = []
    for word in all_words:
        w_id = word_to_id[word]
        coeffs = [0] * SPECTRAL_BINS
        total_co = sum(co_occurrences[word].values())
        if total_co > 0:
            for co_word, count in co_occurrences[word].items():
                if co_word in word_to_id:
                    co_id = word_to_id[co_word]
                    bin_idx = (co_id * 13) % SPECTRAL_BINS
                    energy = float_to_q15(min(1.0, count / total_co))
                    coeffs[bin_idx] = min(32767, coeffs[bin_idx] + energy)
        else:
            bin_idx = (w_id * 31) % SPECTRAL_BINS
            coeffs[bin_idx] = float_to_q15(0.5)
        
        vocab_entries.append({
            "id": w_id,
            "word": word,
            "coeffs": coeffs
        })

    # 5. Build per-intent bigram transition table
    all_bigram_transitions = {}
    for cat in intent_categories:
        transitions = []
        for from_word, next_words in intent_bigrams[cat].items():
            if from_word not in word_to_id:
                continue
            total_next = sum(next_words.values())
            for to_word, count in next_words.items():
                if to_word not in word_to_id:
                    continue
                prob = count / total_next
                transitions.append({
                    "from_id": word_to_id[from_word],
                    "to_id": word_to_id[to_word],
                    "prob_q15": float_to_q15(prob)
                })
        all_bigram_transitions[cat] = transitions
        print(f"  [BIGRAM] {cat}: {len(transitions)} transition pairs")

    # 6. Build intent keyword boosters (words strongly associated with each intent)
    intent_keyword_boost = {}
    for cat in intent_categories:
        boosted_words = {}
        for word in intent_word_sets[cat]:
            if word in word_to_id:
                # Compute TF-IDF-like boost: how unique is this word to this intent?
                total_intents_with_word = sum(1 for c in intent_categories if word in intent_word_sets[c])
                idf = math.log(len(intent_categories) / max(1, total_intents_with_word))
                boost = float_to_q15(min(1.0, idf * 0.5))
                if boost > 0:
                    boosted_words[word_to_id[word]] = boost
        intent_keyword_boost[cat] = boosted_words

    # 7. Chain of Thought Template Structures per Intent
    cot_templates = {
        "greeting": [
            "Mengidentifikasi sapaan dari pengguna",
            "Menyusun respons salam hangat yang sesuai konteks",
            "Menambahkan elemen personal berdasarkan konteks percakapan"
        ],
        "farewell": [
            "Mendeteksi sinyal perpisahan dari pengguna",
            "Menyusun ucapan selamat tinggal yang ramah",
            "Menyampaikan harapan baik untuk pengguna"
        ],
        "identity": [
            "Mengidentifikasi pertanyaan tentang identitas WRAI",
            "Mengakses informasi arsitektur dan kemampuan WRAI",
            "Menyusun penjelasan identitas yang informatif"
        ],
        "math": [
            "Mengidentifikasi topik matematika yang ditanyakan",
            "Menganalisis konsep matematika yang relevan",
            "Menyusun penjelasan langkah-demi-langkah yang logis"
        ],
        "tech": [
            "Mengidentifikasi topik teknologi yang ditanyakan",
            "Mengakses basis pengetahuan teknis WRAI",
            "Menyusun penjelasan teknis yang mudah dipahami"
        ],
        "science": [
            "Mengidentifikasi cabang sains yang ditanyakan",
            "Menganalisis konsep ilmiah yang relevan",
            "Menyusun penjelasan saintifik yang akurat"
        ],
        "general": [
            "Menganalisis pertanyaan umum pengguna",
            "Mencari konteks dan informasi yang relevan",
            "Menyusun jawaban yang komprehensif dan mudah dipahami"
        ]
    }

    # 8. Save complete v5 model
    output_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "wrai_brain_v5.json")

    model_data = {
        "version": "5.0",
        "vocabulary": [{"id": e["id"], "word": e["word"], "coeffs": e["coeffs"]} for e in vocab_entries],
        "intent_centroids": {cat: centroid for cat, centroid in intent_centroids.items()},
        "intent_bigrams": all_bigram_transitions,
        "intent_keyword_boost": {cat: {str(k): v for k, v in boost.items()} for cat, boost in intent_keyword_boost.items()},
        "cot_templates": cot_templates
    }

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False)

    model_size_kb = os.path.getsize(model_path) / 1024.0
    print(f"\n[OK] Model v5 (Intent + CoT) Created Successfully!")
    print(f"  -> Model path        : {os.path.abspath(model_path)} ({model_size_kb:.1f} KB)")
    print(f"  -> Intent Categories : {len(intent_categories)}")
    print(f"  -> Vocab Size        : {vocab_size}")
    print(f"  -> CoT Templates     : {sum(len(v) for v in cot_templates.values())} reasoning steps")

if __name__ == "__main__":
    main()
