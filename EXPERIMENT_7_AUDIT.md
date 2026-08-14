# EXPERIMENT_7_AUDIT.md — Exp6-Post
## LOD with Post-Softmax Distance Decay + Delta-V
**Audit Status:** COMPLETE  
**Evidence Source:** `exp6_lod_post/models/`, `README-E6-POST.md`, `mse_mae_scores_sorted.txt`

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Exp 6 Post — LOD with Post-Softmax Decay |
| Folder | `experiments/exp6_lod_post/` |
| Notebook | `Exp6_lod_post-Gcolab.ipynb` |
| Shell scripts | `exp6_lod_post_phase1.sh`, `exp6_lod_post_phase2.sh` |
| Result directory | **NONE** |
| Central results | `mse_mae_scores_sorted.txt` under "Exp6-Post" |

---

## 2. Objective

Combine all three PRL components (Label+Order+Distance) with the delta-V architectural split, but apply distance decay **after** softmax. Acts as the direct complement to Exp6-Pre.

---

## 3. Mathematical Formulation (Derived from Code)

### Embedding (`exp6_lod_post/models/embed.py:106-127`):
```python
value_emb = self.value_embedding(x)
delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)
delta_x[:, 0, :] = 0.0
temporal_emb = self.temporal_embedding(x_mark)
legendre_pos = self.legendre_embedding(x)
combined_emb = value_emb + temporal_emb + legendre_pos
return self.dropout(combined_emb), delta_x    # delta_x NOT separately dropout'd
```

### Attention (`exp6_lod_post/models/attn.py:25-46`):
```python
scores = einsum(Q, K)
A = softmax(scale * scores)               # SOFTMAX FIRST
alpha = 1/(1 + dist_matrix^decay_a)
A = A * alpha                             # THEN POST-SOFTMAX DECAY
V = einsum(A, values)                     # values = projected delta_x
```

---

## 4. Shell Script Audit

**`exp6_lod_post_phase1.sh`** — verified:
```bash
--model informer --data ETTh1 --features M
--seq_len 96 --label_len 48 --pred_len $pred_len
--enc_in 7 --dec_in 7 --c_out 7
--e_layers 2 --d_layers 1 --attn full --factor 5
--decay_a $alpha
```
**⚠️ Missing `--embed` flag** — uses default `embed='fixed'` (TemporalEmbedding with categorical sinusoidal embeddings for hour/day/month/weekday). This differs from baseline which uses `--embed timeF`.

**⚠️ Missing `--dropout` flag** — uses default `dropout=0.0`.

**⚠️ Missing `--d_ff` flag** — uses default `d_ff=512`, not 2048.

---

## 5. Encoder Bug Analysis

**`exp6_lod_post/models/encoder.py:68-75`:**
```python
if self.conv_layers is not None:
    for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
        x, attn = attn_layer(x, attn_mask=attn_mask, delta_x=delta_x)
        x = conv_layer(x)           # x downsampled by 2
        # delta_x is NOT downsampled here
        attns.append(attn)
    x, attn = self.attn_layers[-1](x, attn_mask=attn_mask, delta_x=delta_x)
```

**BUG:** When `distil=True` (default) with `e_layers=2`, there is 1 `ConvLayer` that halves the sequence length of `x` from 96 to ~48. However, `delta_x` is NOT downsampled. The final `attn_layers[-1]` receives `x` of shape `[B, 48, 512]` but `delta_x` of shape `[B, 96, 512]`.

This means in `AttentionLayer.forward()`:
- `queries` = projection of `x` → `[B, 48, H, E]`
- `keys` = projection of `x` → `[B, 48, H, E]`
- `values` = projection of `delta_x` → `[B, 96, H, D]`  ← WRONG SHAPE

The einsum `A · V` would fail: A is `[B, H, 48, 48]` but V is `[B, 96, H, D]`.

**⚠️ UNLESS** `distil=False` is passed. The shell script `exp6_lod_post_phase1.sh` does NOT pass `--distil` — which means `distil=True` (default in model.py line 13: `distil=True`).

**Comparison with Exp5b:** `exp5b/models/encoder.py:82-86` correctly applies `delta_x = conv_layer(delta_x)` after downsampling. Exp6-Post's encoder.py does NOT do this.

**HOWEVER:** Checking model.py for exp6_lod_post: the `Informer.__init__` uses `distil=True` and builds `e_layers=2` → 1 ConvLayer. If the experiment actually ran successfully, either (a) `distil=False` was used, (b) the shape mismatch was somehow avoided, or (c) this is a latent bug that would crash on execution.

**If this bug exists, the Exp6-Post results in the central file may have been obtained with `distil=False` or may not reflect the code in the repository at all (consistent with the identical-results finding).**

---

## 6. Legendre Implementation Difference

**Exp6-Post** (`legendre_embedding.py:30`): `P = P / (self.d_model ** 0.5)`
**Exp6-Pre** (`legendre_embedding.py:29`): `P = P / math.sqrt(self.d_model)`

These are mathematically identical but syntactically different. Both files are otherwise identical implementations. **The two experiments share the same Legendre logic**, confirmed.

---

## 7. Result Verification

**🔴 All Exp6-Post results are identical to Exp6-Pre results in `mse_mae_scores_sorted.txt`.** (See full analysis in EXPERIMENT_6_AUDIT.md — "Critical Finding" section.)

Verified from central file:
- Phase 1 (6 values): **byte-for-byte identical to Exp6-Pre**
- Phase 2 (9 values): **byte-for-byte identical to Exp6-Pre**

---

## 8. Consistency Checks

### 8.1 Implementation
- ✅ Post-softmax decay confirmed from code
- ✅ Delta-V split confirmed from code
- 🔴 **Bug**: `delta_x` not downsampled in encoder when `distil=True`

### 8.2 Results
- 🔴 **All results identical to Exp6-Pre** — scientifically impossible

### 8.3 README analysis
- The README-E6-POST.md was not fully read (only symbol overview retrieved). It is flagged for potential copy-paste from Exp6-Pre README.

---

## 9. Inconsistency Report

### Critical Issues
1. **🔴 Results are identical to Exp6-Pre**: Both Phase 1 (6 values) and Phase 2 (9 values) are byte-for-byte identical. The two experiments have genuinely different code; the identical results indicate either copy-paste in the results file or both experiments ran the same code.
2. **🔴 Potential encoder shape mismatch bug**: When `distil=True` and `e_layers=2`, `delta_x` is not downsampled when `x` is, causing a shape mismatch in the final encoder layer's attention. If this bug exists and the code was actually run, the results are invalid.

### Moderate Issues
3. **Missing `--embed`, `--dropout`, `--d_ff` from shell script**: Different from baseline (timeF, 0.05, 2048). Configuration differences are not disclosed.

---

## 10. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation confidence | 5/10 | Core logic correct; encoder delta_x shape bug may prevent execution |
| Documentation confidence | 3/10 | README analysis may be based on copied results |
| Result confidence | 1/10 | Results are provably copied or from wrong experiment |
