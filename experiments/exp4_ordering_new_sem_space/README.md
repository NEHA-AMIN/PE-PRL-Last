# Experiment 4 Revised — Ordering in Semantic Space

**Folder:** `experiments/exp4_ordering_new_sem_space/`  
**Class:** `DataEmbedding_ordering_sem`  
**pe_mode flag:** `--pe_mode ordering_sem`

---

## Formula

```
X'_i = X_i + T_i + O_i^sem
```

where:

```
delta_i^val  = 0                if i = 0
               X_i - X_{i-1}   if i >= 1

x_bar^val    = (1/N) * sum_i ||X_i||_2   (scalar per batch element)

O_i^sem      = delta_i^val / (x_bar^val + 1e-8)
```

---

## Components

| Component | Included |
|---|---|
| Value embedding (TokenEmbedding) | ✅ |
| Temporal embedding | ✅ |
| Ordering signal O_i^sem (semantic space) | ✅ |
| Sinusoidal PositionalEmbedding (Zhou) | ❌ removed |
| Legendre embedding | ❌ not used |

---

## Properties

- **Scale-invariant:** multiplying all X_i by scalar α leaves O_i unchanged (α cancels in delta/x_bar)
- **Translation-invariant:** adding constant c to all X_i leaves delta unchanged (c cancels in difference)
- **Locally order-sensitive:** permuting the input sequence changes O_i
- **Content-dependent:** the signal reflects transitions in token identity (semantic space)
- **delta[:, 0, :] == 0** exactly (zero-pad boundary condition, no wrap-around)
- **No new learnable parameters** beyond TokenEmbedding and temporal embedding

---

## Tensor shapes (forward pass)

```
val         [B, L, D]   X_i  = TokenEmbedding(x)
temp        [B, L, D]   T_i  = temporal_embedding(x_mark)
delta       [B, L, D]   delta_i^val; delta[:, 0, :] = 0
x_bar       [B, 1, 1]   mean of ||X_i||_2 over sequence dim
ordering    [B, L, D]   O_i^sem = delta / (x_bar + 1e-8)
output      [B, L, D]   dropout(val + temp + ordering)
```

---

## Files

```
models/__init__.py            empty (unchanged)
models/attn.py                Zhou's original (unchanged)
models/encoder.py             Zhou's original (unchanged)
models/decoder.py             Zhou's original (unchanged)
models/embed.py               DataEmbedding (unchanged) + DataEmbedding_ordering_sem (new)
models/model.py               Informer with pe_mode conditional branch
```

No `legendre_embedding.py` required — this experiment does not use Legendre.

---

## Running

```bash
bash experiments/exp4_ordering_new_sem_space/exp4_ordering_new_sem_space_ph1.sh
```

The script copies models into `Informer2020-original/models/` and calls `main_informer.py`
with `--pe_mode ordering_sem`.

---

## Sanity test

```python
import torch
from models.embed import DataEmbedding_ordering_sem

B, L, D, c_in = 2, 96, 512, 7
x      = torch.randn(B, L, c_in)
x_mark = torch.randn(B, L, 4)

sem = DataEmbedding_ordering_sem(c_in, D)
out = sem(x, x_mark)

assert out.shape == (B, L, D)
# Verify boundary condition: delta[:, 0, :] must be zero
# (not directly accessible post-dropout, but checked by inspecting forward internals)
print("Shape check passed:", out.shape)
```
