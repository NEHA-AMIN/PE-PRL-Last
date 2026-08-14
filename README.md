# Experiment 1 — Pre-Softmax Distance Decay (Exp1-Pre)

## Objective
Test the effect of a simple index-based distance decay on Informer's attention, applied **before** softmax, with no standard positional encoding.

## Hypothesis
Distance-based positional bias (using only index distance |i-j|) can provide sufficient positional information for time-series forecasting without explicit positional embeddings.

## Modifications

### 1. Attention Mechanism (`models/attn.py`)
**Location**: `FullAttention.forward()` ([`attn.py:19-46`](models/attn.py:19))

```python
def forward(self, queries, keys, values, attn_mask):
    B, L, H, E = queries.shape
    _, S, _, D = values.shape
    scale = self.scale or 1./sqrt(E)

    scores = torch.einsum("blhe,bshe->bhls", queries, keys)

    # === EXPERIMENT 1: DISTANCE DECAY ONLY ===
    q_idx = torch.arange(L).unsqueeze(1).to(queries.device)
    k_idx = torch.arange(S).unsqueeze(0).to(queries.device)
    dist_matrix = torch.abs(q_idx - k_idx).float()

    alpha = 1.0 / (1.0 + dist_matrix ** self.decay_a)

    # Apply distance decay to attention scores BEFORE softmax
    scores = scores * alpha.unsqueeze(0).unsqueeze(0)
    # === END EXPERIMENT 1 ===

    if self.mask_flag:
        if attn_mask is None:
            attn_mask = TriangularCausalMask(B, L, device=queries.device)
        scores.masked_fill_(attn_mask.mask, -np.inf)

    A = self.dropout(torch.softmax(scale * scores, dim=-1))
    V = torch.einsum("bhls,bshd->blhd", A, values)
    ...
```

**Key points**:
- Uses **index distance only**: `|i - j|`, no directionality.
- **Multiplicative bias**: `α(i,j) = 1 / (1 + |i-j|^a)` applied to attention **scores**, before softmax.
- **`decay_a` (`a`) is a runtime hyperparameter**, not hardcoded — it's threaded through `FullAttention.__init__(..., decay_a=1.0)` ([`attn.py:11`](models/attn.py:11)) and passed all the way from `Informer.__init__` / `InformerStack.__init__` ([`model.py:16`](models/model.py:16), [`model.py:89`](models/model.py:89)) down to every encoder/decoder `AttentionLayer`. Both `Informer` and `InformerStack` wire it correctly (unlike some later Exp6 variants where `InformerStack` drops it).
- Set via CLI as `--decay_a <value>`; default is `1.0`.

### 2. Embedding Layer (`models/embed.py`)
**Location**: `DataEmbedding.forward()` ([`embed.py:106-109`](models/embed.py:106))

```python
def forward(self, x, x_mark):
    x = self.value_embedding(x) + self.temporal_embedding(x_mark)
    return self.dropout(x)
```

**Key points**:
- ✅ Keeps **value embedding** (`TokenEmbedding`, Conv1D)
- ✅ Keeps **temporal embedding** (calendar features)
- ❌ Removes **sinusoidal positional embedding** entirely — `self.position_embedding` is never called in `forward()`, even though it's still instantiated in `__init__` ([`embed.py:101`](models/embed.py:101)) as dead weight.

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ETTh1 |
| Model | Informer (`--model informer`) |
| Attention | Full (not ProbSparse) |
| Distil | Default `True` (not disabled by either phase) |
| Sequence Length | 96 |
| Label Length | 48 |
| Prediction Length | 48, 96, 192, 336 (swept across phases — see below) |
| Encoder Layers | 2 |
| Decoder Layers | 1 |
| Decay Parameter (`decay_a`) | Swept in Phase 1 (`0.5`, `1.0`, `2.0`); fixed at `1.0` for Phase 2 |
| Seeds | `2021` only in Phase 1; `2021, 2022, 2023` in Phase 2 |

## How to Run

The Colab notebook (`exp1_phase1_pre.ipynb`) invokes two shell scripts:

```bash
bash experiments/exp1_distance_pre_softmax_decay/exp1_pre_distance.sh       # Phase 1
bash experiments/exp1_distance_pre_softmax_decay/exp1_pre_dist_phase2.sh    # Phase 2
```

> **Note:** neither `.sh` file is currently checked into this folder (only `README-E1.md`, the notebook, and `models/` exist on disk) — the notebook must have run them from a copy that was never committed back to this folder. If you need to rerun, reconstruct them from the notebook's cell history using `python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len <L> --decay_a <a> --des <run_id>`, following the same pattern as the other experiments' phase scripts in this repo.

## Results

Source: `exp1_phase1_pre.ipynb`, markdown cells 0, 8, 11, 12.

### Phase 1 — Alpha Selection (pred_len ∈ {96, 192}, seed=2021, 6 runs)

| Alpha | pred_len=96 MSE | pred_len=192 MSE | Verdict |
|-------|------------------|-------------------|---------|
| 0.5 | 0.9024 | 0.8948 | ❌ Worst at both |
| 1.0 | **0.8683** | **0.8463** | ✅ Best at both |
| 2.0 | 0.8784 | 0.9573 | ❌ Degrades at 192 |

**Decision:** fix `decay_a = 1.0`, proceed to Phase 2.

### Phase 2 — Full Validation (decay_a=1.0, 3 seeds, 15 runs)

| PredLen | Seed 2021 | Seed 2022 | Seed 2023 | Avg MSE | Avg MAE | Stability |
|---------|-----------|-----------|-----------|---------|---------|-----------|
| 48  | 0.7680 | 0.7742 | 0.8518 | **0.7980** | **0.6851** | ✅ Stable |
| 96  | 0.8796 | 0.8706 | 0.8509 | **0.8670** | **0.7178** | ✅ Stable |
| 192 | 0.9376 | 0.9298 | 0.9446 | **0.9373** | **0.7486** | ✅ Stable |
| 336 | 0.8764 | 1.1616 | 1.0725 | **1.0368** | **0.7892** | ⚠️ Unstable |

## Analysis

**Works well at short-to-mid horizons (48–192).** Results are consistent across all 3 seeds with low variance — the decay mechanism genuinely helps the model focus on locally relevant tokens for shorter forecasts.

**Breaks down at pred_len=336.** Seed variance explodes (0.876 → 1.162 → 1.072, a spread of ~0.29). This is a real instability, not noise: pre-softmax decay penalizes attention scores before the softmax competition resolves them, which appears to suppress the model's ability to attend to distant-but-relevant tokens once the forecast horizon gets long. The seed=2021 dip at pred_len=336 (0.876, lower than pred_len=192) is a lucky initialization, not a genuine improvement — the high cross-seed variance confirms this.

**Overall conclusion:** Pre-softmax distance decay with `decay_a=1.0` is a **partially effective** modification — consistent gains at pred_len 48–192, unreliable at pred_len=336.

**Open question for Exp1-Post:** does moving the decay to *after* softmax (scaling already-computed attention weights instead of the raw scores) fix the long-horizon instability, since it would no longer interfere with the softmax competition itself?