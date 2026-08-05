# Experiment 5 Revised — Ordering in Positional Space

**Folder:** `experiments/exp5_ordering_new_pos_space/`  
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

## Running

```bash
bash experiments/exp5_ordering_new_pos_space/exp5_ordering_new_pos_space_ph1.sh
```

The script copies all 7 model files (including `legendre_embedding.py`) into
`Informer2020-original/models/` and calls `main_informer.py` with `--pe_mode ordering_pos`.

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
