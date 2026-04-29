# Experiment 6: Label + Order + Distance (Pre-Softmax)

## Objective
Test whether applying **Distance Decay (α) PRE-softmax** improves the combination of **Label (Legendre polynomials)** and **Order (Δx displacement)** for positional structure.

## Research Question
**Does applying pre-softmax distance decay help or hurt the synergy between Label and Order components compared to no distance (Exp 5)?**

## Mathematical Formulation

### Core Equation
```
X'_i = X_i + T_i + P_i + O_i

where:
  X_i  = value embedding (semantic)
  T_i  = temporal embedding
  P_i  = Legendre position label (legendre_embedding.py)
  O_i  = delta_x from embed.py — order in positional space
  w_ij = 1/(1+|i-j|^a) — baked into attn.py, PRE-softmax
         score_ij = w_ij * (Q_i · K_j)/sqrt(d), then softmax
```

### Components

**1. LABEL (P_i):**
- Legendre Polynomials: Provides distinctiveness (each position unique)

**2. ORDER (O_i):**
- Signed Displacement: `Δx[i] = x_i - x_{i-1}`, computed from `value_emb` ONLY. Used as the `V` matrix in attention.

**3. DISTANCE (D - Pre-Softmax):**
- Distance Decay: `α(i,j) = 1 / (1 + |i-j|^a)`
- Applied to attention scores **before** the softmax operation, acting as a sharper suppression mechanism that competes during normalization.

---

## Implementation

### Components Used

- `legendre_embedding.py` - Label component (P_i)
- `embed.py` - Combines Label (P_i) and computes Order (Δx for V matrix)
- `attn.py` - Applies pre-softmax distance decay

### Forward Pass
```python
# 1. Embeddings (embed.py)
value_emb = self.value_embedding(x)
delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1) # Order (V)
temporal_emb = self.temporal_embedding(x_mark)
legendre_pos = self.legendre_embedding(x) # Label

combined_emb = value_emb + temporal_emb + legendre_pos # Used for Q and K

# 2. Attention (attn.py)
scores = torch.einsum("blhe,bshe->bhls", queries, keys)
alpha = 1.0 / (1.0 + dist_matrix ** self.decay_a)
scores = scores * alpha # Pre-softmax distance decay
A = torch.softmax(scale * scores, dim=-1)
V_out = torch.einsum("bhls,bshd->blhd", A, values) # values = delta_x
```

---

## Key Differences from Other Experiments

| Aspect | Exp 5 (L+O) | Exp 2 (LOD Post) | Exp 1 (D Pre) | **Exp 6 (LOD Pre)** |
|--------|-------------|------------------|---------------|---------------------|
| **Label** | ✅ | ✅ | ❌ | ✅ |
| **Order** | ✅ | ✅ | ❌ | ✅ |
| **Distance** | ❌ | ✅ (Post) | ✅ (Pre) | ✅ (Pre) |
| **Temporal** | ✅ | ✅ | ✅ | ✅ |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ETTh1 |
| Model | Informer |
| Attention | Full |
| Sequence Length | 96 |
| Prediction Length | 48, 96, 192 |
| d_model | 512 |
| α (decay) | 1.0 (Phase 2) |

---

## File Structure
```
experiments/exp6_lod_pre/
├── models/
│   ├── legendre_embedding.py   - Label component
│   ├── embed.py                - Computes Q/K (L) and V (O)
│   ├── attn.py                 - Pre-softmax distance decay
│   └── ... (vanilla files)
├── README-E6-Pre.md            - This file
├── exp6_lod_pre_phase1.sh      - Alpha tuning script
└── exp6_lod_pre_phase2.sh      - Seed evaluation script
```

---

## Hypotheses

### **Hypothesis A: Pre-Softmax Synergy**
`LOD (Pre) < L+O (Exp 5)`
Distance decay before softmax successfully filters distant noise without dampening the L+O positional signal.

### **Hypothesis B: Pre-Softmax Interference**
`LOD (Pre) > L+O (Exp 5)`
Distance decay before softmax distorts the score distribution, breaking the synergy of Label and Order.

---

## Results

**Phase 1 (Alpha Tuning):**
`α = 1.0` (linear decay) was the best performer at both pred=96 and pred=192.

**Phase 2 (Seed Stability - α=1.0):**

| Pred Length | Seed 2021 | Seed 2022 | Seed 2023 | **Average MSE** |
|-------------|-----------|-----------|-----------|-----------------|
| 48          | 0.8638    | 0.7529    | 0.8664    | **0.8277**      |
| 96          | 0.7908    | 0.9980    | 0.9743    | **0.9210**      |
| 192         | 1.0472    | 0.8828    | 0.9093    | **0.9464**      |

---

## Complete Comparison Table

| Experiment | Components | Decay Type | MSE (96) ↓ | MSE (192) ↓ |
|------------|-----------|------------|------------|-------------|
| Exp 5      | L+O       | None       | **0.8519** | 0.9788      |
| **Exp 6**  | **L+O+D** | **Pre**    | 0.9210     | **0.9464**  |
| Exp 2      | L+O+D     | Post       | 0.8242     | 0.9002      |
| Exp 1      | D         | Pre        | 0.8683     | 0.8463      |

---

## Analysis

### Key Findings

1. **Mixed Results vs L+O (Exp 5):**
   - At `pred=96`: Exp6-Pre (0.9210) is **worse** than Exp5 (0.8519). Distance decay hurts the synergy.
   - At `pred=192`: Exp6-Pre (0.9464) is **better** than Exp5 (0.9788). Distance decay helps filter long-range noise.

2. **Pre-Softmax vs Post-Softmax (Exp 6 vs Exp 2):**
   - Exp2 (LOD Post) achieved `0.8242` (96) and `0.9002` (192).
   - Exp6 (LOD Pre) achieved `0.9210` (96) and `0.9464` (192).
   - **Conclusion:** Post-softmax decay preserves the L+O signal much better than Pre-softmax decay. Applying distance penalties before softmax distorts the probability distribution.

### Why Did Pre-Softmax Underperform?
- **Normalization Competition:** Applying decay before softmax means the distance penalty alters the relative logits. After softmax normalization, the distribution can become overly spiky or flat, destroying the careful positional balance created by the Legendre and Order components.
- **Post-softmax** (Exp 2) simply re-scales the already normalized probabilities, acting as a soft gate without breaking the fundamental attention distribution.

### Theoretical Implications
The experiment confirms that **when combining complex positional encodings (Label + Order), soft re-weighting (post-softmax) is structurally safer than logit-level interference (pre-softmax).**
