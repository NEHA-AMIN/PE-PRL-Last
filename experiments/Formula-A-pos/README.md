# Formula-A-pos — Ordering in Positional Space

**Folder:** `experiments/Formula-A-pos/`  
**Class:** `DataEmbedding_ordering_pos`  
**pe_mode flag:** `--pe_mode ordering_pos`

---

## Formula

```
X'_i = X_i + T_i + P_i + O_i^pos
```

where:

```
delta_i^leg  = 0                if i = 0
               P_i - P_{i-1}   if i >= 1

p_bar^leg    = (1/N) * sum_i ||P_i||_2   (scalar, fixed per sequence length)

O_i^pos      = delta_i^leg / (p_bar^leg + 1e-8)
```

`P_i` is the Legendre positional embedding vector at position i, pre-computed
as a non-trainable buffer (`LegendrePositionEmbedding`, scaled by `1/sqrt(d_model)`).

---

## Components

| Component | Included |
|---|---|
| Value embedding (TokenEmbedding) | ✅ |
| Temporal embedding | ✅ |
| Legendre embedding P_i (label) | ✅ |
| Ordering signal O_i^pos (positional space) | ✅ |
| Sinusoidal PositionalEmbedding (Zhou) | ❌ removed |

---

## Properties

- **Content-independent:** token identity never enters O_i^pos; P_i depends only on index i
- **Scale-invariant:** multiplying all P_i by scalar α leaves O_i unchanged
- **Translation-invariant:** adding constant c to all P_i leaves delta unchanged
- **Locally order-sensitive:** permuting positions changes O_i
- **delta_p[:, 0, :] == 0** exactly (zero-pad boundary condition)
- **No new learnable parameters** — LegendrePositionEmbedding is a fixed buffer;
  only TokenEmbedding and temporal embedding have trainable weights

---

## Tensor shapes (forward pass)

```
val         [B, L, D]   X_i  = TokenEmbedding(x)
temp        [B, L, D]   T_i  = temporal_embedding(x_mark)
leg         [B, L, D]   P_i  = LegendrePositionEmbedding(x)  (buffer, no grad)
delta_p     [B, L, D]   delta_i^leg; delta_p[:, 0, :] = 0
p_bar       [B, 1, 1]   mean of ||P_i||_2 over sequence dim
ordering    [B, L, D]   O_i^pos = delta_p / (p_bar + 1e-8)
output      [B, L, D]   dropout(val + temp + leg + ordering)
```

---

## Implementation Notes

- `embed.py` imports `LegendrePositionEmbedding` inside `__init__` via `sys.path.insert(0, os.path.dirname(__file__))` — the import is scoped to the `models/` directory at runtime
- `legendre_embedding.py` must be copied to `models/legendre_embedding.py`; unlike some other experiments in this repo, copying to the Informer root directory is **not** required here
- `model.py` selects `DataEmbedding_ordering_pos` when `pe_mode == 'ordering_pos'`; raises `ValueError` for unknown modes
- `attn.py`, `encoder.py`, `decoder.py` are Zhou's originals — unchanged

---

## Files

```
models/__init__.py            empty (unchanged)
models/attn.py                Zhou's original (unchanged)
models/encoder.py             Zhou's original (unchanged)
models/decoder.py             Zhou's original (unchanged)
models/legendre_embedding.py  LegendrePositionEmbedding (from exp3_label_only)
models/embed.py               DataEmbedding (unchanged) + DataEmbedding_ordering_pos (new)
models/model.py               Informer with pe_mode conditional branch
```

**Critical:** `embed.py` imports `LegendrePositionEmbedding` from `legendre_embedding.py`
at `__init__` time. The shell script copies both files. Without `legendre_embedding.py`
the import fails at runtime.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ETTh1 |
| Model | Informer |
| Attention | full (`--attn full`) |
| Features | M (multivariate) |
| `seq_len` | 96 |
| `label_len` | 48 |
| Prediction lengths (Phase 1) | 96, 192 |
| Prediction lengths (Phase 2) | 48, 96, 192, 336 |
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
| `pe_mode` | ordering_pos |

---

## Experimental Design

### Phase 1 — Baseline Screening
- **Goal:** Establish whether consecutive-delta ordering in positional (Legendre) space can match or exceed the Exp1-Pre reference at short and mid horizon.
- **Runs:** 2 (seed=2021, pred_len ∈ {96, 192})

### Phase 2 — Seed Stability Validation
- **Goal:** Multi-seed robustness check across seeds {2021, 2022, 2023} and extended horizons.
- **Runs:** 12 (3 seeds × 4 pred_lens ∈ {48, 96, 192, 336})

**Total experiment runs: 14** (2 Phase 1 + 12 Phase 2) — all completed successfully.

> **Note:** `main_informer.py` does not accept a `--seed` flag. Each run uses a
> distinct `--des` identifier so results are logged separately per seed.
> Logs are stored on Google Drive at
> `/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/logs/Formula-A-pos-phase{1,2}/`.
> The notebook cell output is the only local record of results.

---

## Running

```bash
# Phase 1
bash experiments/Formula-A-pos/formula-A-pos-ph1.sh

# Phase 2
bash experiments/Formula-A-pos/Formula-A-pos-ph2.sh
```

The scripts copy all 7 model files (including `legendre_embedding.py`) into
`Informer2020-original/models/` and call `main_informer.py` with `--pe_mode ordering_pos`.

---

## Sanity test

```python
import torch
from models.embed import DataEmbedding_ordering_pos

B, L, D, c_in = 2, 96, 512, 7
x      = torch.randn(B, L, c_in)
x_mark = torch.randn(B, L, 4)

pos = DataEmbedding_ordering_pos(c_in, D)
out = pos(x, x_mark)

assert out.shape == (B, L, D)
print("Shape check passed:", out.shape)
```

---

## Results

### Phase 1 — Baseline Screening (seed=2021)

| pred_len | MSE | MAE |
|----------|-----|-----|
| 96 | 0.8909913301467896 | 0.7516491413116455 |
| 192 | 0.8714473843574524 | 0.74016934633255 |

**Reference (Exp1-Pre, α=1.0, seed=2021):**
- pred=96 → MSE=0.8683
- pred=192 → MSE=0.8463

**Phase 1 Outcome:** Formula-A-pos is marginally worse than Exp1-Pre at both pred=96 (0.8910 vs 0.8683) and pred=192 (0.8714 vs 0.8463). Phase 2 was run to check seed stability and extended horizons.

---

### Phase 2 — Seed Stability (seeds: 2021, 2022, 2023)

| seed | pred_len | MSE | MAE |
|------|----------|-----|-----|
| 2021 | 48 | 0.8837067484855652 | 0.742670476436615 |
| 2021 | 96 | 0.8045598864555359 | 0.7026992440223694 |
| 2021 | 192 | 0.9127898812294006 | 0.7710499167442322 |
| 2021 | 336 | 1.0232146978378296 | 0.8233616948127747 |
| 2022 | 48 | 0.8641096353530884 | 0.7387957572937012 |
| 2022 | 96 | 0.8698857426643372 | 0.7494027614593506 |
| 2022 | 192 | 0.8873394131660461 | 0.7371362447738647 |
| 2022 | 336 | 1.200138807296753 | 0.8784056305885315 |
| 2023 | 48 | 0.8290919661521912 | 0.7213220596313477 |
| 2023 | 96 | 0.7605789303779602 | 0.6719040870666504 |
| 2023 | 192 | 0.9974824786186218 | 0.7885332703590393 |
| 2023 | 336 | 1.1193406581878662 | 0.8653407692909241 |

---

## Analysis

### Key Findings

1. **Phase 1 vs reference (Exp1-Pre, α=1.0, seed=2021):**
   - pred=96: Formula-A-pos (0.8910) is slightly **worse** than Exp1-Pre (0.8683).
   - pred=192: Formula-A-pos (0.8714) is slightly **worse** than Exp1-Pre (0.8463).
   - The positional-space ordering signal does not improve over distance-only pre-softmax at single-seed screening.

2. **Phase 2 seed variability:**
   - At pred=96, seeds produce MSE of 0.8046 (seed=2021), 0.8699 (seed=2022), 0.7606 (seed=2023) — a range of ~0.11, suggesting moderate seed sensitivity.
   - At pred=192, results span 0.8873–0.9975, indicating more variability at longer horizons.
   - At pred=336, results span 1.0232–1.2001, showing the largest variance across seeds.

3. **Best observed results:**
   - pred=48: seed=2023 (MSE=0.8291)
   - pred=96: seed=2023 (MSE=0.7606) — the best single result of the entire experiment
   - pred=192: seed=2022 (MSE=0.8873)
   - pred=336: seed=2021 (MSE=1.0232)

4. **Contrast with Formula-B-pos:**
   - Formula-A uses a consecutive delta: `delta_i = P_i - P_{i-1}` (local positional change)
   - Formula-B uses global mean deviation: `delta_i = mu_p - P_i`
   - TODO: Information could not be verified from the repository — Formula-B-pos results are not available in this experiment's files for direct comparison.
