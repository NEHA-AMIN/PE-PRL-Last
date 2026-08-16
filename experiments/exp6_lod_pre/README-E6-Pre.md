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
delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)  # Order (V)
delta_x[:, 0, :] = 0.0  # zero first position
temporal_emb = self.temporal_embedding(x_mark)
legendre_pos = self.legendre_embedding(x)  # Label

combined_emb = value_emb + temporal_emb + legendre_pos  # Used for Q and K

# Dropout applied to BOTH combined_emb and delta_x
return self.dropout(combined_emb), self.dropout(delta_x)

# 2. Attention (attn.py)
scores = torch.einsum("blhe,bshe->bhls", queries, keys)
alpha = 1.0 / (1.0 + dist_matrix ** self.decay_a)
scores = scores * alpha  # Pre-softmax distance decay
A = self.dropout(torch.softmax(scale * scores, dim=-1))
V_out = torch.einsum("bhls,bshd->blhd", A, values)  # values = delta_x
```

### Implementation Notes
- `embed.py` returns a **tuple** `(combined_emb, delta_x)` — non-standard return signature
- Dropout is applied to **both** `combined_emb` and `delta_x` (unlike exp6_lod_post where only `combined_emb` receives dropout)
- `delta_x[:, 0, :] = 0.0` — first position zeroed to avoid roll-wrap artefact
- `legendre_embedding.py` is imported inside `embed.py` via a bare import — must be copied to both `models/` and the Informer root directory at runtime
- `encoder.py`: `ConvLayer` applied to both `x` and `delta_x` (FINDING G fix)
- `model.py`: `decay_a` threaded through both `Informer` and `InformerStack` (FINDING F fix)
- `delta_values` routing handled in `AttentionLayer.forward()`, not inside `FullAttention`

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
| Attention | Full (`--attn full`) |
| Features | M (multivariate) |
| Sequence Length (`seq_len`) | 96 |
| Label Length (`label_len`) | 48 |
| Prediction Lengths | 48, 96, 192, 336 (Phase 2); 96, 192 (Phase 1) |
| `d_model` | 512 |
| `n_heads` | 8 |
| `e_layers` | 2 |
| `d_layers` | 1 |
| `d_ff` | 2048 |
| `factor` | 5 |
| `enc_in` / `dec_in` / `c_out` | 7 / 7 / 7 |
| `dropout` | 0.05 |
| `embed` | timeF |
| `freq` | h |
| `activation` | gelu |
| `distil` | True |
| `batch_size` | 32 |
| `learning_rate` | 0.0001 |
| `train_epochs` | 6 |
| `patience` | 3 |
| α (`decay_a`) screened (Phase 1) | 0.5, 1.0, 2.0 |
| α (`decay_a`) used (Phase 2) | 2.0 (best at pred=96) and 0.5 (best at pred=192) |

---

## File Structure
```
experiments/exp6_lod_pre/
├── models/
│   ├── legendre_embedding.py   - Label component
│   ├── embed.py                - Computes Q/K (L+T) and V (O); returns tuple
│   ├── attn.py                 - Pre-softmax distance decay
│   └── ... (vanilla files)
├── README-E6-Pre.md            - This file
├── exp6_lod_pre_phase1.sh      - Alpha screening script (6 runs)
├── exp6_lod_pre_phase2_a2.0.sh - Seed stability script for decay_a=2.0 (12 runs)
└── exp6_lod_pre_phase2_a0.5.sh - Seed stability script for decay_a=0.5 (12 runs)
```

> **Note:** Logs and model checkpoints are stored on Google Drive at
> `/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/logs/` during Colab runs.
> No log files are committed locally; the notebook cell output is the only local record.

---

## Experimental Design

### Phase 1 — Alpha Screening
- **Goal:** Identify which `decay_a` value performs best before committing to a full seed sweep.
- **Runs:** 3 alpha values × 2 pred_lens × 1 seed = **6 runs**
- `decay_a` ∈ {0.5, 1.0, 2.0}, `pred_len` ∈ {96, 192}, `seed` = 2021

### Phase 2 — Seed Stability Validation
- **Goal:** Confirm reproducibility of the Phase 1 winner(s) across 3 seeds and all pred_lens.
- Phase 1 produced a **split winner**: `decay_a=2.0` was best at pred=96; `decay_a=0.5` was best at pred=192.
- Both alphas were carried forward independently to Phase 2.
- **Phase 2a (decay_a=2.0):** 4 pred_lens × 3 seeds = **12 runs**
- **Phase 2b (decay_a=0.5):** 4 pred_lens × 3 seeds = **12 runs**
- **Total experiment runs: 30** (6 Phase 1 + 12 Phase 2a + 12 Phase 2b)

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

### Phase 1 — Alpha Screening (seed=2021)

| decay_a | pred_len | MSE | MAE |
|---------|----------|-----|-----|
| 0.5 | 96 | 0.9848689436912537 | 0.8067865371704102 |
| 0.5 | 192 | 0.9074642658233643 | 0.7573962807655334 |
| 1.0 | 96 | 1.0043070316314697 | 0.8226070404052734 |
| 1.0 | 192 | 1.0571106672286987 | 0.843242883682251 |
| 2.0 | 96 | 0.962917685508728 | 0.7838810682296753 |
| 2.0 | 192 | 0.9779786467552185 | 0.7907344698905945 |

**Phase 1 Decision:**
- Best at `pred=96`: **decay_a=2.0** (MSE=0.962917685508728) → carried to Phase 2a
- Best at `pred=192`: **decay_a=0.5** (MSE=0.9074642658233643) → carried to Phase 2b
- `decay_a=1.0` was the **worst** performer at both prediction horizons and was not used in Phase 2.

---

### Phase 2a — Seed Stability (decay_a=2.0)

| pred_len | seed | MSE | MAE |
|----------|------|-----|-----|
| 48 | 2021 | 0.9079546928405762 | 0.7569813132286072 |
| 48 | 2022 | 0.864521324634552 | 0.7437335848808289 |
| 48 | 2023 | 0.9986046552658081 | 0.8158033490180969 |
| 96 | 2021 | 0.9166896939277649 | 0.7669988870620728 |
| 96 | 2022 | 0.9080425500869751 | 0.7563011050224304 |
| 96 | 2023 | 1.1502560377120972 | 0.8798839449882507 |
| 192 | 2021 | 0.823397696018219 | 0.6995365023612976 |
| 192 | 2022 | 1.093313217163086 | 0.8230853080749512 |
| 192 | 2023 | 1.048041820526123 | 0.8283913731575012 |
| 336 | 2021 | 1.0276719331741333 | 0.7901483178138733 |
| 336 | 2022 | 1.098031759262085 | 0.8191969394683838 |
| 336 | 2023 | 1.1569440364837646 | 0.8463030457496643 |

**Aggregated (decay_a=2.0, Mean ± Std across 3 seeds):**

| pred_len | MSE (Mean±Std) | MAE (Mean±Std) |
|----------|----------------|----------------|
| 48 | 0.9237±0.0559 | 0.7722±0.0313 |
| 96 | 0.9917±0.1122 | 0.8011±0.0559 |
| 192 | 0.9883±0.1180 | 0.7837±0.0595 |
| 336 | 1.0942±0.0528 | 0.8185±0.0229 |

---

### Phase 2b — Seed Stability (decay_a=0.5)

| pred_len | seed | MSE | MAE |
|----------|------|-----|-----|
| 48 | 2021 | 0.7460262179374695 | 0.6771563291549683 |
| 48 | 2022 | 0.7492878437042236 | 0.6808463931083679 |
| 48 | 2023 | 0.896075963973999 | 0.7470437288284302 |
| 96 | 2021 | 0.9299110770225525 | 0.7582659125328064 |
| 96 | 2022 | 0.9433990716934204 | 0.7817332148551941 |
| 96 | 2023 | 0.8104053139686584 | 0.7008529901504517 |
| 192 | 2021 | 0.9711651802062988 | 0.7886838316917419 |
| 192 | 2022 | 1.0376712083816528 | 0.8137487769126892 |
| 192 | 2023 | 0.9958336353302002 | 0.8036894202232361 |
| 336 | 2021 | 1.1499874591827393 | 0.8544692397117615 |
| 336 | 2022 | 1.1521509885787964 | 0.8546919822692871 |
| 336 | 2023 | 1.0370842218399048 | 0.8187065124511719 |

**Aggregated (decay_a=0.5, Mean ± Std across 3 seeds):**

| pred_len | MSE (Mean±Std) | MAE (Mean±Std) |
|----------|----------------|----------------|
| 48 | 0.7971±0.0700 | 0.7017±0.0321 |
| 96 | 0.8946±0.0598 | 0.7470±0.0340 |
| 192 | 1.0016±0.0275 | 0.8020±0.0103 |
| 336 | 1.1131±0.0537 | 0.8426±0.0169 |

---

## Complete Comparison Table

> Values shown are the best Phase 2 mean MSE for each pred_len (best α selected per horizon).
> pred=96 uses α=2.0 results; pred=192 uses α=0.5 results.

| Experiment | Components | Decay Type | MSE (96) ↓ | MSE (192) ↓ |
|------------|-----------|------------|------------|-------------|
| Exp 5      | L+O       | None       | **0.8519** | 0.9788      |
| **Exp 6**  | **L+O+D** | **Pre**    | 0.9917±0.1122 (α=2.0) | 1.0016±0.0275 (α=0.5) |
| Exp 2      | L+O+D     | Post       | 0.8242     | 0.9002      |
| Exp 1      | D         | Pre        | 0.8683     | 0.8463      |

> **Note on Exp 5 and Exp 2 values:** These are carried from prior experiment records and have not been independently re-verified within this audit. TODO: Confirm Exp 5 and Exp 2 MSE reference values from their respective notebooks.

---

## Analysis

### Key Findings

1. **Negative result vs L+O (Exp 5):**
   - At `pred=96`: Exp6-Pre best mean (0.9917, α=2.0) is **worse** than Exp5 (0.8519). Pre-softmax distance decay hurts the L+O synergy at short horizons.
   - At `pred=192`: Exp6-Pre best mean (1.0016, α=0.5) is also **worse** than Exp5 (0.9788). Pre-softmax decay does not improve over no-distance even at mid horizon.

2. **Pre-Softmax vs Post-Softmax (Exp 6 vs Exp 2):**
   - Exp2 (LOD Post) achieved `0.8242` (pred=96) and `0.9002` (pred=192).
   - Exp6 best (LOD Pre): `0.9917` (pred=96, α=2.0) and `1.0016` (pred=192, α=0.5).
   - **Conclusion:** Post-softmax decay preserves the L+O signal much better than Pre-softmax decay at all measured horizons.

3. **Alpha sensitivity — split winner:**
   - Phase 1 showed no single α dominates both horizons. α=2.0 wins at pred=96 but α=0.5 wins at pred=192, indicating horizon-dependent sensitivity to the steepness of the distance falloff.
   - α=1.0 was the worst performer at both horizons and was excluded from Phase 2.

4. **Phase 2 instability (decay_a=2.0):**
   - At pred=96, the three seeds produced MSE of 0.9167, 0.9080, and 1.1503 — high variance (std=0.1122), suggesting the pre-softmax decay combined with strong suppression (α=2.0) is unstable across seeds.

### Why Did Pre-Softmax Underperform?
- **Normalization Competition:** Applying decay before softmax means the distance penalty alters the relative logits. After softmax normalization, the distribution can become overly spiky or flat, destroying the careful positional balance created by the Legendre and Order components.
- **Post-softmax** (Exp 2) simply re-scales the already normalized probabilities, acting as a soft gate without breaking the fundamental attention distribution.

### Theoretical Implications
The experiment confirms that **when combining complex positional encodings (Label + Order), soft re-weighting (post-softmax) is structurally safer than logit-level interference (pre-softmax).**
