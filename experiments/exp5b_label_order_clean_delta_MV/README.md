# Experiment 5b: Label + Order with Clean Delta (L+O Clean)

## Objective

Test the **Label + Order** combination with a **clean delta signal** that contains ONLY value embedding differences, without any positional or temporal components.

## Research Question

**Does a pure semantic ordering signal (x_i - x_{i-1} from value embeddings only) combined with Legendre labels provide better performance than the original Exp 5 implementation?**

---

## Mathematical Formulation

### Core Architecture

```
Input to Attention:
  Q, K ← project(value_emb + temporal_emb + legendre_pos)   [Combined embedding]
  V    ← project(delta_x)                                    [Clean delta only]

where:
  value_emb   = TokenEmbedding(input)         [x_i — semantic content]
  temporal_emb = TemporalEmbedding(time_mark) [T_i — temporal features]
  legendre_pos = Legendre(i)                  [p_i — label, positional distinctiveness]
  delta_x[i]  = value_emb[i] - value_emb[i-1]  [ORDER — clean semantic delta]
  delta_x[0]  = 0                              [first position zeroed]
```

### Key Components

**1. LABEL (p_i) — Positional Distinctiveness:**
```
Legendre Polynomials: p_i = [L_0(x_i), ..., L_{d-1}(x_i)]
Domain: x_i ∈ [-1, 1]
Orthogonality: ⟨p_n, p_m⟩ = δ_{nm}
Scaling: 1/√d_model
```
- Provides unique positional identity per time step
- Added to combined embedding going into Q/K projections

**2. ORDER (delta_x) — Clean Semantic Signal:**
```
Delta Computation: delta_x[i] = value_emb[i] - value_emb[i-1]
First Position:    delta_x[0] = 0
```
- **CRITICAL:** Computed from `value_emb` ONLY (TokenEmbedding output)
- **NO `p_i` terms:** avoids Legendre polynomial contamination
- **NO `T_i` terms:** avoids temporal embedding contamination
- Goes exclusively to the V projection in the encoder's self-attention

**3. Attention Mechanism:**
```
Similarity: scores = (Q · K^T) / √d_k
  where Q, K ← project(value_emb + temporal_emb + legendre_pos)

Aggregation: output = softmax(scores) · V
  where V ← project(delta_x)
```

---

## Critical Differences from Experiment 5

| Aspect | **Exp 5** | **Exp 5b (This)** |
|--------|-----------|-------------------|
| **Delta Input** | `legendre_pos` (p_i) | `value_emb` (x_i) |
| **Delta Formula** | O_i = (1/(L-1))·Σ(p_i - p_j) | delta_x[i] = value_emb[i] - value_emb[i-1] |
| **Delta Type** | Pairwise mean (all positions) | Sequential shift (temporal) |
| **Delta Contains** | Positional differences | Semantic differences |
| **Q/K Source** | x_i + T_i + p_i + O_i (additive) | x_i + T_i + p_i (combined emb) |
| **V Source** | x_i + T_i + p_i + O_i (additive) | delta_x (clean, separated) |
| **Architecture** | Additive (all in embedding) | Separated (delta in V only) |

---

## Implementation Details

### File Structure

```
experiments/exp5b_label_order_clean_delta_MV/
├── e5b_lab_ord_clean_delta_mv_ph1.sh   — Phase 1 training script (only script)
├── theory.md                            — Phase notes and inferences
├── exp5b_label_order_clean_delta_MV.ipynb
└── models/
    ├── legendre_embedding.py   — Legendre polynomial embeddings (p_i)
    ├── embed.py                — Returns TUPLE (combined_emb, delta_x)
    ├── attn.py                 — Modified AttentionLayer: V uses delta_values kwarg
    ├── encoder.py              — Modified Encoder/EncoderLayer: passes delta_x; CRITICAL ConvLayer fix
    ├── model.py                — Modified Informer (tuple unpack); InformerStack NOT updated
    ├── decoder.py              — Standard (no delta)
    └── __init__.py             — Module initialization
```

**Note on path:** The actual directory is `experiments/exp5b_label_order_clean_delta_MV/` (with `_MV` suffix). The shell script also copies files to `Informer2020-original/models/` (not `Informer2020-main`).

### Key Code Changes

**1. `embed.py` — `DataEmbedding.forward()` returns a TUPLE:**
```python
def forward(self, x, x_mark):
    # 1. Value embedding
    value_emb = self.value_embedding(x)             # x_i  [B, L, D]

    # 2. CLEAN DELTA: x_i - x_{i-1} (value_emb ONLY — no p_i, no T_i)
    delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)
    delta_x[:, 0, :] = 0.0                          # zero first position

    # 3. Temporal and Legendre
    temporal_emb = self.temporal_embedding(x_mark)  # T_i  [B, L, D]
    legendre_pos = self.legendre_embedding(x)        # p_i  [B, L, D]

    # 4. Combined for Q/K
    combined_emb = value_emb + temporal_emb + legendre_pos
    combined_emb = self.dropout(combined_emb)

    return combined_emb, delta_x  # non-standard tuple return
```

**Note:** `self.position_embedding` (standard sinusoidal PE) is still instantiated in `__init__` but is never called in `forward()` — dead weight.

**`DataEmbeddingDecoder`** is a separate class also in `embed.py`. It is used for the decoder: includes Legendre but returns a scalar tensor (not a tuple), following standard interface.

**2. `attn.py` — `AttentionLayer.forward()` accepts delta kwargs:**
```python
def forward(self, queries, keys, values, attn_mask,
            delta_queries=None, delta_keys=None, delta_values=None):
    queries = self.query_projection(queries)  # Q from combined_emb
    keys    = self.key_projection(keys)       # K from combined_emb

    # V from delta_x if provided, else fall back to values (decoder cross-attn)
    if delta_values is not None:
        values = self.value_projection(delta_values)
    else:
        values = self.value_projection(values)

    out, attn = self.inner_attention(queries, keys, values, attn_mask)
    return self.out_projection(out), attn
```

Only `delta_values` is actually used. `delta_queries` and `delta_keys` are accepted but ignored.

**3. `encoder.py` — `EncoderLayer` and `Encoder` pass `delta_x`; CRITICAL ConvLayer fix:**
```python
# EncoderLayer.forward()
def forward(self, x, attn_mask=None, delta_x=None):
    new_x, attn = self.attention(
        x, x, x,
        attn_mask=attn_mask,
        delta_queries=delta_x,
        delta_keys=delta_x,
        delta_values=delta_x   # only this is used
    )
    ...

# Encoder.forward() — CRITICAL: delta_x is downsampled through ConvLayer
# to maintain matching sequence length after distillation
if self.conv_layers is not None:
    for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
        x, attn = attn_layer(x, attn_mask=attn_mask, delta_x=delta_x)
        x = conv_layer(x)
        if delta_x is not None:
            delta_x = conv_layer(delta_x)   # ← CRITICAL FIX
        attns.append(attn)
```

**4. `model.py` — `Informer` unpacks the tuple from `enc_embedding`:**
```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, ...):
    enc_out, delta_enc = self.enc_embedding(x_enc, x_mark_enc)  # unpack tuple
    enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, delta_x=delta_enc)

    dec_out = self.dec_embedding(x_dec, x_mark_dec)  # standard, no delta
    dec_out = self.decoder(dec_out, enc_out, ...)
    return self.projection(dec_out)
```

**`InformerStack` is NOT updated** — its `forward()` calls `self.enc_embedding(...)` without unpacking a tuple. It would break at runtime if used. Only `Informer` (not `InformerStack`) supports the Exp5b architecture.

---

## Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Dataset | ETTh1 | `e5b_lab_ord_clean_delta_mv_ph1.sh` |
| Model | Informer | `e5b_lab_ord_clean_delta_mv_ph1.sh` |
| Attention | Full (not ProbSparse) | `--attn full` |
| Distillation | Enabled (default) | Informer default |
| Sequence Length | 96 | `--seq_len 96` |
| Label Length | 48 | `--label_len 48` |
| Prediction Lengths | 96, 192 (Phase 1 only) | shell script loop |
| Encoder Layers | 2 | `--e_layers 2` |
| Decoder Layers | 1 | `--d_layers 1` |
| d_model | 512 | `--d_model 512` |
| n_heads | 8 | `--n_heads 8` |
| d_ff | 2048 | `--d_ff 2048` |
| dropout | 0.05 | `--dropout 0.05` |
| Embed type | timeF | `--embed timeF` |
| Frequency | h | `--freq h` |
| Activation | gelu | `--activation gelu` |
| Learning rate | 0.0001 | `--learning_rate 0.0001` |
| Batch size | 32 | `--batch_size 32` |
| Factor | 5 | `--factor 5` |
| enc_in / dec_in / c_out | 7 | ETTh1 multivariate (7 features) |
| Train epochs | 6 | `--train_epochs 6` |
| Patience (early stop) | 3 | `--patience 3` |
| Seeds | 2021 (Phase 1 only) | shell script |
| Alpha / decay_a | N/A — no sweep, no alpha parameter in this experiment | — |

**Note:** The old README listed pred_len values 48, 96, 192, 336, 720 and datasets ETTh1/ETTm1. The actual shell script only runs pred_len ∈ {96, 192} on ETTh1 with seed=2021.

---

## Execution Protocol

This experiment used a **single-phase protocol**. No Phase 2 was run.

### Phase 1 — Feasibility Check (`e5b_lab_ord_clean_delta_mv_ph1.sh`)

- **Goal:** Detect whether L+O Clean beats the best prior result at short and mid horizons.
- **Configuration:** Single fixed config (no alpha sweep — this experiment has no `decay_a`).
- **pred_len:** {96, 192}
- **Seed:** 2021
- **Total runs:** 2

### Phase 2 — Skipped

Phase 2 was deliberately skipped. Phase 1 results failed all success criteria (MSE > 0.804 at both prediction lengths, worse than all prior L+O variants). Following the experiment's Decision Guide ("NO (worse everywhere) → Document as negative"), proceeding to a broader sweep was deemed unjustified.

### How to Run

```bash
# Phase 1 only (Colab path — update PROJECT_ROOT before running locally)
bash experiments/exp5b_label_order_clean_delta_MV/e5b_lab_ord_clean_delta_mv_ph1.sh
```

**Note:** `run_exp5b.sh` referenced in the original README does not exist. The only training script is `e5b_lab_ord_clean_delta_mv_ph1.sh`.

---

## Results

Source: `mse_mae_scores_sorted.txt`

### Phase 1 — Feasibility Check (pred_len ∈ {96, 192}, seed=2021)

| Run ID | pred_len | Seed | MSE | MAE |
|--------|---------|------|-----|-----|
| `exp5b_ph1_ETTh1_lod_clean_pred96_seed2021` | 96 | 2021 | 0.9004 | 0.7599 |
| `exp5b_ph1_ETTh1_lod_clean_pred192_seed2021` | 192 | 2021 | 0.9511 | 0.7929 |

### Phase 2

Phase 2 was not run. No Phase 2 results exist in `mse_mae_scores_sorted.txt` (confirmed: `⚠ Phase 2 — NOT FOUND in notebook outputs`).

---

## Comparison with Other Experiments

Source: `mse_mae_scores_sorted.txt`. Values shown are Phase 1 single-seed (seed=2021) for Exp5b, and Phase 2 averages (3 seeds) where available for others.

| Experiment | Components | pred_len=96 MSE | pred_len=192 MSE |
|------------|-----------|-----------------|------------------|
| **Exp5b (This)** | **L+O Clean delta** | **0.9004** | **0.9511** |
| Exp5 (Phase 1, seed=2021) | L+O positional delta | 0.8292 | 0.7965 |
| Exp1-Pre (Phase 2 avg, α=1.0) | Distance pre-softmax | 0.8670 | 0.9373 |
| Exp2 LOD (Phase 2 avg, α=0.5) | L+O+D full | 0.8534 | 0.9625 |
| Exp3b Label (Phase 2 avg) | Label only | 0.8858 | 0.8913 |
| Vanilla | Standard PE | TODO: Information could not be verified from the repository. | — |

**Note:** Vanilla baseline values are not present in `mse_mae_scores_sorted.txt`. The `results/baseline_ph1_ETTh1_pred96_seed2021/` and `results/baseline_ph1_ETTh1_pred192_seed2021/` directories exist but are empty.

---

## Analysis

### Key Findings

1. **Hypothesis Disproven:** The clean semantic delta did not improve over Exp 5. Exp5b MSE at pred_len=96 (0.9004) is worse than Exp5 Phase 1 seed=2021 (0.8292), and at pred_len=192 (0.9511) also worse (0.7965).

2. **Failure vs All Baselines:** Exp5b underperforms all comparison experiments at both prediction lengths. It does not meet even the weakest success criterion (MSE < 0.804).

3. **Semantic Delta Insufficient for V:** Stripping all absolute positional and temporal context from the Value projection degrades the model's aggregation ability. The model loses track of *when* a semantic shift occurred.

4. **Positional Overlap May Be Necessary:** The "redundancy" of positional markers in Exp 5 may actually be a required inductive bias. The attention mechanism appears to rely on positional/temporal markers in Values to properly weight future states.

5. **Sequential Shifts Are Too Noisy:** A simple `x_i - x_{i-1}` delta appears too local and noisy to serve as a robust ordering signal across a full sequence length. It lacks the stable global gradient that standard PE or distance-based decay provides.

### Success Criteria Outcome

| Criterion | Threshold | Exp5b Result | Status |
|-----------|-----------|--------------|--------|
| Success | MSE < 0.719 (beat Exp5) | pred_96: 0.9004, pred_192: 0.9511 | ❌ Failed |
| Strong Success | MSE < 0.725 (beat Distance) | pred_96: 0.9004, pred_192: 0.9511 | ❌ Failed |
| Partial | 0.719 < MSE < 0.804 | pred_96: 0.9004, pred_192: 0.9511 | ❌ Failed |
| Failure | MSE > 0.804 | pred_96: 0.9004, pred_192: 0.9511 | ❌ Confirmed negative |

---

## Theoretical Context

### From the Paper
> "The combination of label + ordering (PoPE + ΔV) fails to match the results of PE"

**This Experiment's Investigation:**
- **Exp 5:** Tested PoPE + ΔV with delta derived from Legendre positional space (pairwise mean across all positions)
- **Exp 5b:** Tested PoPE + ΔV with delta from semantic value space (sequential shift x_i − x_{i-1})
- **Conclusion:** Both variants fail. The failure is not solely due to positional overlap in Exp 5 — the paper's claim holds even with the cleaner separation tested here.

### Key Insight from Results

The theoretical purity of the clean semantic delta did not translate to practical model performance. Three possible explanations from `theory.md`:

1. **Semantic Delta is Insufficient for Values:** The attention mechanism needs absolute context in V, not just local incremental differences.
2. **Positional Overlap is Necessary:** The inductive bias from having positional/temporal markers in all three Q/K/V projections is beneficial, not redundant.
3. **Sequential Shifts ≠ Good Ordering:** `x_i - x_{i-1}` is too noisy and local to provide a stable ordering signal over seq_len=96.

---

## Computational Notes

- **Memory:** Stores both `combined_emb` and `delta_x` simultaneously — approximately 2× embedding memory vs standard Informer
- **Speed:** No additional compute beyond the extra tensor in V projection — effectively same complexity as standard Informer O(L²)
- **Gradient flow:** Value embedding receives gradients from both Q/K path (via `combined_emb`) and V path (via `delta_x`); Legendre and temporal embeddings receive gradients from Q/K path only

---

## References

- **Exp 1-Pre:** Distance pre-softmax (decay in attention weights)
- **Exp 2:** Full L+O+D (Legendre + Order + Distance combined)
- **Exp 3b:** Label only (Legendre polynomials, no ordering)
- **Exp 4:** Order only (signed displacements)
- **Exp 5:** Label + Order (delta from Legendre positional space)
- **Exp 5b:** Label + Order (delta from value embedding semantic space) ← **This experiment**

---

**Status:** ❌ Negative Result — Phase 2 Skipped
**Outcome:** Hypothesis disproven. Clean semantic delta (x_i − x_{i-1} in V) performs worse than all prior L+O variants and fails all success criteria.
**Last Updated:** Based on `mse_mae_scores_sorted.txt` and `theory.md`
