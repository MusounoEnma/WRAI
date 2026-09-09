"""
=============================================================================
   🔬 WRAI v16 (1.7B) FORENSIC DEEP-INSPECTION TOOL
=============================================================================
 Alat diagnostik mandiri (self-contained) untuk memeriksa:
 1. Status Checkpoint & Metadata
 2. Kesehatan Bobot Tensor (NaN, Inf, Zero, Norm, Distribusi)
 3. Status Gate Wavelet Spectral & Faktor Peluruhan Retention (Gamma)
 4. Forensik Logits & Probabilitas Top-10 Next-Token
=============================================================================
"""

import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

DRIVE_SAVE_DIR = "/content/drive/MyDrive/WRAI_v16_1.7B_Models_Transplant"
OUTPUT_DIR = "models_v16_1.7b_transplant"
QWEN3_MODEL_NAME = "Qwen/Qwen3-1.7B"
FALLBACK_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

HIDDEN_DIM = 2048
FFN_INTERMEDIATE_DIM = 6144
NUM_LAYERS = 28
NUM_HEADS = 16
HEAD_DIM = 128
WAVELET_LEVELS = 4
MAX_SEQ_LEN = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Standalone Model Definition
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        in_dtype = x.dtype
        x_f32 = x.float()
        norm = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f32 * norm).to(in_dtype) * self.weight

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class HaarDWT1D(nn.Module):
    def __init__(self, hidden_dim, levels=4):
        super().__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.sqrt2 = math.sqrt(2.0)
        self.detail_gains = nn.Parameter(torch.ones(levels))
        self.approx_gain = nn.Parameter(torch.tensor(1.0))
        self.gate_weight = nn.Parameter(torch.zeros(hidden_dim))
        self.gate_bias = nn.Parameter(torch.full((hidden_dim,), -4.0))

    def forward(self, x):
        B, T, D = x.shape
        if T % 2 != 0:
            pad = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, pad], dim=1)
            padded = True
        else:
            x_pad = x
            padded = False

        curr_approx = x_pad
        for lvl in range(min(self.levels, int(math.log2(max(2, curr_approx.size(1)))))):
            cur_len = curr_approx.size(1)
            if cur_len % 2 != 0:
                break
            even = curr_approx[:, 0::2, :]
            odd = curr_approx[:, 1::2, :]
            approx = (even + odd) / self.sqrt2
            detail = (even - odd) / self.sqrt2
            curr_approx = approx * self.approx_gain + detail * self.detail_gains[lvl]

        if curr_approx.size(1) != x_pad.size(1):
            curr_approx = F.interpolate(curr_approx.transpose(1, 2), size=x_pad.size(1), mode="linear", align_corners=False).transpose(1, 2)

        if padded:
            curr_approx = curr_approx[:, :T, :]

        gate = torch.sigmoid(self.gate_weight * x + self.gate_bias)
        return (1.0 - gate) * x + gate * curr_approx

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadRetentionLayer(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=16, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        self.w_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.w_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        init_gammas = 1.0 - torch.pow(2.0, -5.0 - torch.arange(num_heads).float())
        self.decay_logit = nn.Parameter(torch.logit(init_gammas))
        self.group_norm = nn.GroupNorm(num_heads, num_heads * head_dim)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.w_q(x).view(B, T, H, D)
        k = self.w_k(x).view(B, T, H, D)
        v = self.w_v(x).view(B, T, H, D)
        gamma = torch.sigmoid(self.decay_logit).view(1, H, 1, 1)

        # Parallel Causal Retention
        qh = q.permute(0, 2, 1, 3)
        kh = k.permute(0, 2, 1, 3)
        vh = v.permute(0, 2, 1, 3)

        i_idx = torch.arange(T, device=x.device).view(T, 1)
        j_idx = torch.arange(T, device=x.device).view(1, T)
        dist = i_idx - j_idx
        causal_mask = (dist >= 0).to(x.dtype)
        decay_matrix = torch.pow(gamma, dist.clamp(min=0).view(1, 1, T, T)) * causal_mask.view(1, 1, T, T)

        attn = torch.matmul(qh, kh.transpose(-1, -2))
        attn = attn * decay_matrix
        out = torch.matmul(attn, vh).permute(0, 2, 1, 3).reshape(B, T, H * D)
        out_normed = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        return self.w_out(out_normed), None

class WRAI17BLayer(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, ffn_dim=FFN_INTERMEDIATE_DIM,
                 num_heads=NUM_HEADS, head_dim=HEAD_DIM):
        super().__init__()
        self.rms_ret = RMSNorm(hidden_dim)
        self.retention = MultiHeadRetentionLayer(hidden_dim, num_heads, head_dim)
        self.rms_ffn = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward(self, x, s=None):
        res = x
        out_ret, next_s = self.retention(self.rms_ret(x), s)
        x = res + out_ret
        res = x
        x = res + self.ffn(self.rms_ffn(x))
        return x, next_s

class WRAI17BModel(nn.Module):
    def __init__(self, vocab_size=151936, hidden_dim=HIDDEN_DIM, ffn_dim=FFN_INTERMEDIATE_DIM,
                 num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_seq_len)
        self.spectral1 = HaarDWT1D(hidden_dim, levels=WAVELET_LEVELS)
        self.spectral2 = HaarDWT1D(hidden_dim, levels=WAVELET_LEVELS)
        self.layers = nn.ModuleList([WRAI17BLayer(hidden_dim, ffn_dim) for _ in range(num_layers)])
        self.ln_final = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight

    def forward(self, input_ids, hidden_states=None):
        x = self.embed(input_ids)
        x = self.spectral1(x)
        next_hidden_states = []
        for i, layer in enumerate(self.layers):
            h_i = hidden_states[i] if hidden_states is not None else None
            x, next_h = layer(x, h_i)
            next_hidden_states.append(next_h)
            if i == (len(self.layers) // 2) - 1:
                x = self.spectral2(x)
        x = self.ln_final(x)
        return self.output_proj(x), next_hidden_states

# -----------------------------------------------------------------------------
# Forensic Main Procedure
# -----------------------------------------------------------------------------

def print_header(title):
    print("\n" + "="*70)
    print(f"  🔍 {title}")
    print("="*70)

def main():
    print_header("WRAI v16 FORENSIC INSPECTION & DIAGNOSTIC SUITE")

    # 1. Mount Google Drive if in Colab
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("[OK] Google Drive mounted successfully.")
    except Exception:
        pass

    # 2. Locate Checkpoints
    candidates = [
        os.path.join(DRIVE_SAVE_DIR, "wrai_v16_1.7b_best.pt"),
        os.path.join(DRIVE_SAVE_DIR, "wrai_v16_1.7b_latest.pt"),
        os.path.join(DRIVE_SAVE_DIR, "wrai_v16_1.7b_transplant_initial.pt"),
        os.path.join(OUTPUT_DIR, "wrai_v16_1.7b_best.pt"),
        os.path.join(OUTPUT_DIR, "wrai_v16_1.7b_latest.pt"),
        os.path.join(OUTPUT_DIR, "wrai_v16_1.7b_transplant_initial.pt"),
    ]

    found_files = [c for c in candidates if os.path.exists(c)]
    print(f"\n[*] Daftar File Checkpoint yang Ditemukan:")
    for f in found_files:
        sz = os.path.getsize(f) / (1024**2)
        print(f"   -> {f} ({sz:.1f} MB)")

    if not found_files:
        print("[!] ERROR: Tidak ada berkas checkpoint yang ditemukan! Periksa direktori Drive.")
        return

    active_ckpt_path = found_files[0]
    print(f"\n[*] Melakukan Forensik pada Checkpoint Utama: {active_ckpt_path}")
    ckpt = torch.load(active_ckpt_path, map_location="cpu")

    print(f"   Epoch          : {ckpt.get('epoch', 'N/A')}")
    print(f"   Step           : {ckpt.get('step', 'N/A')}")
    print(f"   Optimizer Step : {ckpt.get('optimizer_step', 'N/A')}")
    print(f"   Reported Loss  : {ckpt.get('loss', 'N/A')}")
    print(f"   Arch Config    : {ckpt.get('arch', {})}")

    state_dict = ckpt.get("model_state", {})
    print(f"   Total Weight Tensors : {len(state_dict)}")

    # -------------------------------------------------------------------------
    # 3. Weight Sanity & Numerical Integrity Checks
    # -------------------------------------------------------------------------
    print_header("1. PEMERIKSAAN INTEGRITAS DAN KESEHATAN BOBOT")

    nan_keys = []
    inf_keys = []
    zero_keys = []
    summary_table = []

    for name, tensor in state_dict.items():
        t = tensor.float()
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        l2_norm = torch.norm(t).item()
        mean = t.mean().item()
        std = t.std().item()

        if has_nan: nan_keys.append(name)
        if has_inf: inf_keys.append(name)
        if l2_norm == 0.0 and "gate_weight" not in name: zero_keys.append(name)

        if any(k in name for k in ["embed", "ln_final", "layers.0.rms_ret", "layers.0.retention.w_q", "layers.0.retention.decay_logit", "layers.0.ffn.w_gate", "spectral1.gate_bias", "spectral1.approx_gain"]):
            summary_table.append((name, list(t.shape), mean, std, l2_norm))

    if nan_keys:
        print(f"❌ [KRITIS] DITEMUKAN NaN PADA TENSOR: {nan_keys}")
    else:
        print("✅ [SEHAT] 0% NaN: Seluruh tensor bebas dari nilai NaN (Tidak ada gradien rusak).")

    if inf_keys:
        print(f"❌ [KRITIS] DITEMUKAN Infinity PADA TENSOR: {inf_keys}")
    else:
        print("✅ [SEHAT] 0% Inf: Tidak ada gradient explosion.")

    if zero_keys:
        print(f"⚠️ [PERINGATAN] Tensor Bernilai 0 Mutlak: {zero_keys}")
    else:
        print("✅ [SEHAT] Bobot neuron aktif dan hidup.")

    print("\n   [Statistik Lapisan Sampel]:")
    print(f"   {'Nama Tensor':<35} | {'Shape':<18} | {'Mean':<10} | {'Std':<10} | {'L2 Norm':<10}")
    print("   " + "-"*92)
    for n, sh, m, s, l2 in summary_table:
        print(f"   {n:<35} | {str(sh):<18} | {m:<10.5f} | {s:<10.5f} | {l2:<10.2f}")

    # Inspect Retention Decay Logit (Gammas)
    if "layers.0.retention.decay_logit" in state_dict:
        raw_gammas = torch.sigmoid(state_dict["layers.0.retention.decay_logit"].float())
        print(f"\n   [Retention Layer 0 Decay Factors (\u03b3)]: min={raw_gammas.min():.4f}, max={raw_gammas.max():.4f}")
        print(f"   Gammas per head: {[round(g.item(), 4) for g in raw_gammas]}")

    # Inspect Wavelet Spectral Gate
    if "spectral1.gate_bias" in state_dict:
        gb = state_dict["spectral1.gate_bias"].float().mean().item()
        gw = state_dict["spectral1.gate_weight"].float().mean().item()
        approx = state_dict["spectral1.approx_gain"].float().item()
        print(f"   [Wavelet Spectral 1]: gate_bias={gb:.4f} (gate={torch.sigmoid(torch.tensor(gb)):.4f}), approx_gain={approx:.4f}")

    # -------------------------------------------------------------------------
    # 4. Live Logits Distribution & Top-10 Next-Token Forensics
    # -------------------------------------------------------------------------
    print_header("2. FORENSIK DISTRIBUSI LOGITS & PREDIKSI TOP-10 TOKEN")

    print("[*] Membersihkan CPU RAM sebelum inferensi logits...")
    del ckpt
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[*] Memuat Tokenizer...")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    try:
        tok = AutoTokenizer.from_pretrained(QWEN3_MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    MODEL_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    vocab_size = state_dict["embed.weight"].size(0) if "embed.weight" in state_dict else 151936
    print(f"[*] Mengalokasikan model WRAI v16 langsung di GPU VRAM (Vocab: {vocab_size:,}, Dtype: {MODEL_DTYPE})...")
    with torch.device(DEVICE):
        model = WRAI17BModel(vocab_size=vocab_size).to(dtype=MODEL_DTYPE)

    # Muat bobot langsung ke GPU
    with torch.no_grad():
        for k, v in state_dict.items():
            if k in model.state_dict():
                dst = model.state_dict()[k]
                if dst.shape == v.shape:
                    dst.copy_(v.to(DEVICE, dtype=MODEL_DTYPE))
                elif k == "embed.weight":
                    min_v = min(dst.size(0), v.size(0))
                    dst[:min_v].copy_(v[:min_v].to(DEVICE, dtype=MODEL_DTYPE))
    
    # Hapus state_dict dari CPU RAM untuk menghemat 3.5 GB
    del state_dict
    gc.collect()

    # Ensure weight-tying
    model.output_proj.weight = model.embed.weight
    model.eval()

    test_prompts = [
        "Halo, siapa kamu?",
        "Jelaskan siapa kamu dan bagaimana ekosistem WRAI bekerja.",
        "Write a Python function to reverse a string:\ndef reverse_string(s):",
        "The capital city of France is",
        "Mengapa langit terlihat berwarna biru pada siang hari?",
    ]

    for p in test_prompts:
        print(f"\n" + "-"*70)
        print(f"📝 [PROMPT]: \"{p}\"")
        chatml_p = f"<|im_start|>user\n{p.strip()}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tok.encode(chatml_p, add_special_tokens=False)
        inp_t = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            logits, _ = model(inp_t)
            last_logits = logits[0, -1, :].float()

        # Logits health metrics
        l_mean = last_logits.mean().item()
        l_std = last_logits.std().item()
        l_max = last_logits.max().item()
        l_min = last_logits.min().item()
        probs = F.softmax(last_logits, dim=-1)

        print(f"   Logits Stats: Mean={l_mean:.2f} | Std={l_std:.2f} | Max={l_max:.2f} | Min={l_min:.2f}")

        # Top-10 Predicted Tokens
        top10_probs, top10_ids = torch.topk(probs, 10)
        print(f"   Top-10 Predicted Next Tokens:")
        for rank, (pr, tid) in enumerate(zip(top10_probs, top10_ids)):
            token_str = tok.decode([tid.item()])
            repr_str = repr(token_str)
            print(f"     #{rank+1:<2} | TokenID: {tid.item():<7} | Prob: {pr.item()*100:6.2f}% | Text: {repr_str:<18}")

    print_header("3. STATUS REKAYASA & KESIMPULAN")
    print("Skrip forensik selesai dijalankan.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
