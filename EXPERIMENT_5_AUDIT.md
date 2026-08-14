# EXPERIMENT_5_AUDIT.md — Exp5b
## Label + Order (Clean Delta, Q/K vs V Split)
**Audit Status:** COMPLETE  
**Evidence Source:** `exp5b_label_order_clean_delta_MV/models/`, `mse_mae_scores_sorted.txt`, `TECHNICAL_OBSERVATIONS.md`

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Exp 5b — Label + Order with Clean Delta (Semantic Separation) |
| Folder | `experiments/exp5b_label_order_clean_delta_MV/` |
| Notebook | `exp5b_label_order_clean_delta_MV.ipynb` |
| Shell script | `e5b_lab_ord_clean_delta_mv_ph1.sh` |
| Theory doc | `theory.md` |
| Result directory | **NONE** — no `results/exp5b*/` folder |
| Central results | `mse_mae_scores_sorted.txt` under "Exp5b" — Phase 1 only |

---

## 2. Objective

**Architectural innovation:** Separate the positional signal (Label, via Q/K) from the ordering signal (consecutive delta, via V). The hypothesis is that decoupling "what to attend to" (Q·K similarity using label-enriched embeddings) from "what to aggregate" (V using temporal change signal) improves forecasting.

---

## 3. Mathematical Formulation (Derived from Code)

**Source:** [`experiments/exp5b_label_order_clean_delta_MV/models/embed.py:138-159`](experiments/exp5b_label_order_clean_delta_MV/models/embed.py:138)

```python
value_emb = self.value_embedding(x)                          # x_i
delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)  # Δx_i
delta_x[:, 0, :] = 0.0                                       # zero boundary
temporal_emb = self.temporal_embedding(x_mark)                # T_i
legendre_pos = self.legendre_embedding(x)                     # p_i
combined_emb = value_emb + temporal_emb + legendre_pos       # x_i + T_i + p_i
return dropout(combined_emb), delta_x                        # TUPLE
```

**Formulas:**
```
Q_i, K_i ← projection(combined_emb_i)    combined_emb_i = x_i + T_i + p_i
V_i ← projection(delta_x_i)              delta_x_i = x_i − x_{i-1}; delta_x_0 = 0

Attention: A = softmax(scale · QKᵀ)
Output_i = Σ_j A_ij · V_j = Σ_j A_ij · (x_j − x_{j-1})
```

**Key semantic property:** The output aggregates attention-weighted first-differences of token embeddings. Position 0 contributes nothing to any output because `V_0 = 0`.

---

## 4. Algorithm Flow

```
embed.py returns (combined_emb, delta_x)
         │                │
         ▼                ▼
    Q, K projections   V projection
    (combined_emb)     (delta_x)
         │                │
         └────────────────┘
                  ↓
    A = softmax(scale · QKᵀ)
                  ↓
    Output = A · V  [= weighted sum of temporal deltas]
                  ↓
encoder.py: delta_x also downsampled through ConvLayer (distillation)
```

---

## 5. Code Walkthrough

### 5.1 Encoder ConvLayer Downsampling (encoder.py:82-86)
```python
x = conv_layer(x)
if delta_x is not None:
    delta_x = conv_layer(delta_x)
```
**Critical:** delta_x is downsampled by the same ConvLayer as the combined embedding. This maintains shape consistency when distillation layers halve sequence length. **This is correctly implemented in Exp5b.**

### 5.2 AttentionLayer.forward() signature (attn.py:202)
```python
def forward(self, queries, keys, values, attn_mask, delta_queries=None, delta_keys=None, delta_values=None):
```
- `delta_queries` and `delta_keys` are accepted but NOT used — queries and keys always use `combined_emb`
- Only `delta_values` is used for V projection
- The signature is over-engineered; delta_queries/keys are dead parameters

### 5.3 EncoderLayer passes delta_x to ALL three positions (encoder.py:48-53)
```python
new_x, attn = self.attention(x, x, x, attn_mask,
    delta_queries=delta_x, delta_keys=delta_x, delta_values=delta_x)
```
Passing delta_x as delta_queries and delta_keys has no effect (they're ignored in attn.py), but the signature allows future modification.

### 5.4 Decoder (DataEmbeddingDecoder in embed.py:162-198)
The decoder uses a separate `DataEmbeddingDecoder` class that returns only `combined_emb` (no delta). This is correct: the decoder attends into encoder outputs and predicts future values; V in decoder cross-attention comes from encoder, not decoder delta.

---

## 6. Configuration Audit

| Parameter | TECHNICAL_OBSERVATIONS.md | mse_mae_scores_sorted.txt | Consistent? |
|-----------|--------------------------|--------------------------|-------------|
| delta source | value_emb only | — | ✅ confirmed from code |
| zero boundary | position 0 = 0 | — | ✅ confirmed from code |
| combined_emb | x_i + T_i + p_i | — | ✅ confirmed from code |

---

## 7. Result Verification

### Phase 1 only (from `mse_mae_scores_sorted.txt`):
| α (irrelevant) | pred_len | seed | MSE | MAE |
|----------------|---------|------|-----|-----|
| 1.0 | 96 | 2021 | 0.9004 | 0.7599 |
| 1.0 | 192 | 2021 | 0.9511 | 0.7929 |

**Phase 2:** Explicitly noted as "NOT FOUND in notebook outputs."
**No results directory** exists in `results/` for Exp5b.

**TECHNICAL_OBSERVATIONS.md table:** All Exp5b results listed as "TBD" — document was never updated after runs.

---

## 8. Consistency Checks

### 8.1 Architecture
- ✅ Q/K from combined, V from delta_x — confirmed from code
- ✅ Zero boundary at position 0 — confirmed
- ⚠️ delta_queries/delta_keys passed but not used — dead parameters

### 8.2 Results
- 🔴 Phase 2 not completed; TECHNICAL_OBSERVATIONS.md performance table shows all TBD
- 🔴 No results directory — cannot verify from logs

---

## 9. Inconsistency Report

### Critical Issues
1. **Experiment incomplete**: Phase 2 not run; TECHNICAL_OBSERVATIONS.md table never filled in.

### Moderate Issues
2. **Dead parameters**: `delta_queries` and `delta_keys` in AttentionLayer accepted but never used.
3. **TECHNICAL_OBSERVATIONS.md reference incorrect**: File references path `experiments/exp5b_label_order_clean_delta/models/embed.py:142-144` but the actual folder is `exp5b_label_order_clean_delta_MV`.

---

## 10. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation confidence | 9/10 | Architecture is well-designed; delta pipeline correct |
| Documentation confidence | 3/10 | TECH_OBS table is TBD; Phase 2 absent |
| Result confidence | 2/10 | Only Phase 1 exists; no log files in results/ |
