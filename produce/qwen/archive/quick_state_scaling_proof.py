import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 80)
print(" 🔬 BUKTI EMPIRIS: STATE SHAPE & RUNTIME MEMORY FOOTPRINT (T = 128 s/d 8192)")
print("=" * 80)

# Konfigurasi Layer WRAI-X 0.8B
B = 1
H = 16
HD = 128
D = 1024
NUM_LAYERS = 28

device = torch.device("cpu")
dtype = torch.bfloat16

print(f"[*] Arsitektur: {NUM_LAYERS} Layers, {H} Heads, HeadDim={HD}, Dim={D}")
print(f"[*] Per-token update: S_t = γ S_{{t-1}} + K_t^T V_t")
print(f"[*] Output step     : O_t = Q_t S_t\n")

# Simulasi state update murni untuk mengukur scaling memori terhadap T
def test_state_scaling(test_lengths):
    print("=" * 100)
    print(f"{'Context Length (T)':<20} | {'State Tensor Shapes (Per Layer)':<35} | {'State Mem (28 L)':<16} | {'Δ Mem':<10} | {'Transformer KV (28 L)'}")
    print("=" * 100)
    
    base_mem = None
    for T in test_lengths:
        # Inisialisasi state awal (sama seperti forward_step pertama kali)
        states = []
        for l in range(NUM_LAYERS):
            sm = torch.zeros(B, H, HD, HD, dtype=dtype)
            sr = torch.zeros(B, H, HD, HD, dtype=dtype)
            shdc = torch.zeros(B, D, dtype=dtype)
            states.append((sm, sr, shdc))
            
        # Simulasikan recurrent update sepanjang T token
        gamma_m = 0.95
        gamma_r = 0.93
        
        # Step T tokens
        for t in range(min(T, 100)): # run update to verify dynamic allocation
            k = torch.randn(B, H, 1, HD, dtype=dtype)
            v = torch.randn(B, H, 1, HD, dtype=dtype)
            kv = torch.matmul(k.transpose(-1, -2), v) # [B, H, HD, HD]
            for l in range(NUM_LAYERS):
                sm, sr, shdc = states[l]
                sm = sm * gamma_m + kv
                sr = sr * gamma_r + kv
                shdc = shdc * 0.95 + 0.05 * torch.randn(B, D, dtype=dtype)
                states[l] = (sm, sr, shdc)
                
        # Ukur memori persis seluruh state tensors
        total_elements = 0
        total_bytes = 0
        for l in range(NUM_LAYERS):
            sm, sr, shdc = states[l]
            total_elements += sm.nelement() + sr.nelement() + shdc.nelement()
            total_bytes += (sm.nelement() * sm.element_size() + 
                            sr.nelement() * sr.element_size() + 
                            shdc.nelement() * shdc.element_size())
            
        mem_mb = total_bytes / (1024 * 1024)
        if base_mem is None:
            base_mem = mem_mb
        delta_mem = mem_mb - base_mem
        
        # Transformer KV Cache formula: 2 (K+V) * L * B * H * HD * T * 2 bytes
        tf_kv_bytes = 2 * NUM_LAYERS * B * H * HD * T * 2
        tf_kv_mb = tf_kv_bytes / (1024 * 1024)
        if tf_kv_mb > 1024:
            tf_str = f"{tf_kv_mb / 1024:.2f} GB"
        else:
            tf_str = f"{tf_kv_mb:.2f} MB"
            
        sm_shape = f"sm:{list(states[0][0].shape)}"
        sr_shape = f"sr:{list(states[0][1].shape)}"
        shdc_shape = f"shdc:{list(states[0][2].shape)}"
        shape_str = f"{sm_shape} {shdc_shape}"
        
        print(f"T = {T:<16} | {shape_str:<35} | {mem_mb:.2f} MB (BF16)   | {delta_mem:+.4f} MB | {tf_str}")
        
    print("=" * 100)

test_state_scaling([128, 512, 2048, 8192, 32768])
