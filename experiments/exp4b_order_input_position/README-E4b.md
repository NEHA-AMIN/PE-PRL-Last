# Experiment 4b: Order Only — Applied to Positional (Legendre) Space

## Status
**Appendix Experiment** — requested by mentor explicitly for appendix inclusion.

> *"Order only that we have and all other experiments I want you to do it in input as well,
> I see no reason to do it but I wanna have it in my appendix"*
> — Mentor meeting notes

---

## Motivation

Exp4 (Order Only) applies the ordering operator to **value embeddings** (semantic space):
```
O_i = (1/N-1) · Σ_{j≠i} (X_i - X_j)   [X = value embeddings]
```

The mentor wanted to know: what if the same ordering operator is applied to **positional (Legendre) embeddings** instead? This tests whether the space in which the ordering operator acts — semantic vs. geometric/positional — fundamentally changes what the model learns.

---

## Mathematical Formulation

### Legendre Positional Vectors
Each position `i` is mapped to `x_i ∈ [-1, 1]`:
```
x_i = 2i / (seq_len - 1) - 1
```

Legendre vectors are generated via the recurrence relation:
```
L_0(x) = 1
L_1(x) = x
L_n(x) = ((2n-1)·x·L_{n-1}(x) - (n-1)·L_{n-2}(x)) / n

P_i = [L_0(x_i), L_1(x_i), ..., L_{d-1}(x_i)] / sqrt(d_model)   ∈ R^d
```

These are orthogonal in infinite dimensions: `<P_i, P_j> = 0` for `i ≠ j`.
In finite `d_model=512`, minimum correlation `≈ c/sqrt(d)` is accepted (as discussed by mentor).

### Ordering Signal in Positional Space
```
Signed positional displacement:
    Δp_ij = P_i - P_j   ∈ R^d

Ordering operator (uniform weighted, positional space):
    O_i = (1/(N-1)) · Σ_{j≠i} Δp_ij
        = (1/(N-1)) · Σ_{j≠i} (P_i - P_j)
        = P_i - (1/(N-1)) · Σ_{j≠i} P_j        [algebraic simplification]
```

### Final Input Embedding
```
X'_i = X_i + T_i + O_i(P)

where:
    X_i    = value_embedding(x)           [B, L, D]  semantic content
    T_i    = temporal_embedding(x_mark)   [B, L, D]  hour/day/month/weekday
    O_i(P) = ordering_operator(P)         [1, L, D]  geometric ordering signal
```

---

## Key Distinctions from Other Experiments

| Aspect | exp4 (O on value) | **exp4b (O on position)** | exp3 (Label) | exp5b (L+O) |
|--------|-------------------|---------------------------|--------------|-------------|
| **Ordering input** | X (value emb) | **P (Legendre)** | — | P (Legendre) |
| **Label added directly** | ❌ | ❌ | ✅ P_i added | ✅ P_i to Q/K |
| **Delta in V matrix** | ❌ | ❌ | ❌ | ✅ Δx (clean) |
| **Temporal** | ✅ | ✅ | ❌ | ✅ |
| **Distance decay** | ❌ | ❌ | ❌ | ❌ |
| **Weighting** | Uniform | **Uniform** | — | Uniform |
| **Space** | Semantic | **Geometric** | Geometric | Both |

### Algebraic relationship between exp4b and exp3
Note that:
```
O_i(P) = P_i - mean_{j≠i}(P_j)
```
So exp4b adds a **mean-centered version of the Legendre label** rather than the raw label.
This is strictly different from exp3 which adds `P_i` directly.
If all `P_j` are roughly symmetric (which Legendre vectors are), `mean(P_j) ≈ 0`,
making `O_i(P) ≈ P_i`. This is exactly why the mentor said "I see no reason to do it"
— the result may collapse toward exp3 behavior. But confirming this empirically is
precisely the point of the appendix experiment.

---

## Implementation

### New file: `ordering_operator_positional.py`

```python
class LegendreEmbedding(nn.Module):
    """Generates Legendre orthogonal positional vectors P_i."""
    def forward(self, seq_len, device) -> Tensor:  # [1, L, d_model]
        ...

class OrderingOperatorPositional(nn.Module):
    """
    O_i = (1/N-1) · Σ_{j≠i} (P_i - P_j)   in positional/Legendre space.
    """
    def forward(self, seq_len, device) -> Tensor:  # [1, L, d_model]
        P = self.legendre(seq_len, device)   # [1, L, D]
        P_i = P.unsqueeze(2)                 # [1, L, 1, D]
        P_j = P.unsqueeze(1)                 # [1, 1, L, D]
        delta_p = P_i - P_j                  # [1, L, L, D]
        # mask diagonal, uniform aggregate
        O = delta_p.sum(dim=2) / (L - 1)    # [1, L, D]
        return O
```

### Modified: `embed.py`

```python
def forward(self, x, x_mark):
    value_emb   = self.value_embedding(x)            # X_i
    temporal_emb = self.temporal_embedding(x_mark)   # T_i
    ordering_pos = self.ordering_operator_pos(        # O_i(P)
        seq_len=value_emb.shape[1],
        device=value_emb.device
    )
    x_out = value_emb + temporal_emb + ordering_pos  # X'_i = X_i + T_i + O_i
    return self.dropout(x_out)
```

**Unchanged files** (identical to vanilla Informer):
- `attn.py` — standard FullAttention, no distance modification
- `encoder.py` — no delta_values passed
- `decoder.py` — unchanged
- `model.py` — unchanged

---

## Computational Notes

### Complexity
- Legendre matrix computation: `O(L × d_model)` — done once per forward pass
- Pairwise positional displacement: `O(L² × d_model)` — same as exp4
- For `L=96`, `d_model=512`: ~4.7M FLOPs per batch element (identical to exp4)

### Key Implementation Detail
The ordering signal `O_i(P)` has shape `[1, L, d_model]` — it is **batch-independent**
because Legendre vectors depend only on position index, not on input data `x`.
This is broadcasted over batch dimension `B` in `embed.py`.

This is different from exp4, where `O_i(X)` has shape `[B, L, d_model]` because
it depends on the actual token values.

---

## File Structure

```
experiments/exp4b_order_input_position/
├── models/
│   ├── ordering_operator_positional.py   [NEW] — O_i(P) computation
│   ├── embed.py                          [MODIFIED] — uses O_i(P) not O_i(X)
│   ├── attn.py                           [VANILLA] — unchanged
│   ├── encoder.py                        [VANILLA] — unchanged
│   ├── decoder.py                        [VANILLA] — unchanged
│   └── model.py                          [VANILLA] — unchanged
├── README-E4b.md                         — this file
└── run_exp4b.sh                          — training script
```

---

## How to Run

```bash
bash experiments/exp4b_order_input_position/run_exp4b.sh
```

Runs: 2 datasets × 3 seeds × 5 prediction lengths = **30 total runs**

---

## Expected Behavior and Hypothesis

### Mentor's Intuition
The mentor explicitly said "I see no reason to do it" — implying he expects exp4b
to perform similarly to exp3 (Label Only), because:

```
O_i(P) = P_i - mean_{j≠i}(P_j)
```

If Legendre vectors are approximately mean-zero (which they are for large `seq_len`),
then `O_i(P) ≈ P_i`, and exp4b degenerates toward exp3 behavior.

### Predicted Outcome
```
exp4b MSE ≈ exp3 MSE (≈ 1.124)   [if Legendre mean ≈ 0]
```
OR slightly better if mean-centering provides marginal benefit over raw labels.

### What This Confirms
If exp4b ≈ exp3: confirms that the ordering operator in positional space
adds no new information beyond what the label itself provides — validating
the mentor's intuition and strengthening the paper's argument that the ordering
operator must be applied in value/semantic space (exp4) or combined with
clean delta-V (exp5b) to be meaningful.

---

## Comparison Table (to be filled after results)

| Experiment | Space | Components | Temporal | MSE ↓ | MAE ↓ |
|------------|-------|-----------|----------|-------|-------|
| Vanilla    | —     | Standard PE | ✅ | 0.519 | 0.513 |
| exp3 (L)   | Positional | Label only | ❌ | 1.124 | 0.855 |
| exp4 (O)   | Semantic | Order on X | ✅ | 0.835 | 0.720 |
| **exp4b (O-pos)** | **Positional** | **Order on P** | ✅ | TBD | TBD |
| exp5b (L+O)| Both | Label+CleanDelta | ✅ | 0.719 | 0.635 |

---

## Appendix Narrative

This experiment answers: *Does it matter whether the ordering operator acts on
semantic content (value embeddings) or geometric structure (positional embeddings)?*

The answer informs our understanding of what the ordering primitive actually captures
and whether its benefit is tied to the content-based displacement signal or
can be reproduced through positional geometry alone.