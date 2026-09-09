# SPESIFIKASI FORMULASI MATEMATIKA WRAI (FASE 1)

## 1. Domain Aritmatika Integer Fixed-Point (Q15 & Q31)

Untuk menghindari penggunaan perkalian floating-point ($FP32/FP16$) pada hardware berdaya rendah, WRAI merepresentasikan amplitudo sinyal, koefisien spektral, dan tabel trigonometri dalam format **Q15 Fixed-Point** (`int16_t`) dan **Q31 Fixed-Point** (`int32_t`).

### 1.1 Kuantisasi Q15
Format `q15_t` merepresentasikan nilai desimal kontinu $x \in [-1.0, 1.0 - 2^{-15}]$ dalam bentuk integer berbertanda 16-bit:
$$X_{Q15} = \text{round}(x \cdot 32768)$$

- **Perkalian Fixed-Point Q15:**
  $$Z_{Q15} = (X_{Q15} \times Y_{Q15}) \gg 15$$
  Pada arsitektur 32-bit (ARM Cortex-M, ESP32, AVX1), operasi ini diakselerasi dengan instruksi Hardware MAC (`SMULBB`, `SMLABB`).

- **Perkalian Fixed-Point Q31:**
  $$Z_{Q31} = (X_{Q31} \times Y_{Q31}) \gg 31$$

---

## 2. Token-to-Wave Encoding & Superposisi

Setiap token $T_k$ dari kosakata terdefinisi dipetakan secara terdeterministik ke besaran gelombang dasar:
1. **Frekuensi Sub-Harmonik ($\omega_k$):** $\omega_k = \frac{2\pi \cdot f_k}{F_s} \in [0, \pi]$
2. **Sudut Fase Dasar ($\phi_k$):** $\phi_k \in [0, 2\pi)$
3. **Amplitudo Resonansi ($A_k$):** $A_k \in [0, 32767]$ (Format Q15)

### 2.1 Gelombang Superposisi Kontinu Diskrit
Untuk urutan konteks token $\{T_1, T_2, \dots, T_K\}$, sinyal superposisi komposit $S[n]$ pada sampel diskrit $n \in [0, N-1]$ dihitung sebagai:
$$S[n] = \sum_{k=1}^{K} \left( (A_k \times \text{LUT}_{\sin}[(\omega_k \cdot n + \phi_k) \pmod{2\pi}]) \gg 15 \right)$$

Untuk mencegah akumulasi overflow pada penjumlahan $K$ gelombang, hasil superposisi discale menggunakan bitwise right-shift bergantung pada $K$:
$$S_{\text{scaled}}[n] = S[n] \gg \lceil \log_2 K \rceil$$

---

## 3. Resonansi Harmonis & Peak Detection (Decoding Non-GEMM)

Proses klasifikasi/decoding semantik dalam WRAI **TIDAK** menggunakan perkalian matriks $Y = W \cdot X$. Sebagai gantinya, konteks dianalisis melalui resonansi spektral spektrum FFT Fixed-Point.

### 3.1 Vektor Spektral Magnitudo ($M[m]$)
Diberikan hasil transform FFT $X[m] = R[m] + j \cdot I[m]$ untuk bin frekuensi $m \in [0, N/2 - 1]$:
$$M[m] = \text{approx\_magnitude}(R[m], I[m])$$
Di mana magnitudo diestimasi tanpa akar kuadrat floating-point menggunakan **Alpha Max + Beta Min Algorithm**:
$$M[m] \approx \max(|R[m]|, |I[m]|) + \frac{3}{8} \cdot \min(|R[m]|, |I[m]|)$$

### 3.2 Skor Resonansi Harmonis ($R_{\text{score}}$)
Kesesuaian antara sinyal input spektral $M[m]$ dengan profil pola target $P[m]$ dihitung melalui perkalian titik spektral terkuantisasi (*Spectral Dot Product*):
$$R_{\text{score}} = \sum_{m=0}^{N/2-1} (M[m] \times P[m]) \gg 15$$

Puncak resonansi tertinggi ($\max R_{\text{score}}$) menentukan token/respons yang dihasilkan oleh sistem.
