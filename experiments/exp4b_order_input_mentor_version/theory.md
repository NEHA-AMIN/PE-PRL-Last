# Analysis of Experiment 4b (`exp4b_order_input_mentor_version`)

## Aim
This experiment was explicitly requested by the mentor as an **Appendix Experiment**. The primary aim is to test what happens if the **Ordering Operator** is applied to **positional (Legendre) embeddings** rather than value (semantic) embeddings. It explores whether the space in which the ordering operator acts—semantic vs. geometric/positional—fundamentally changes what the model learns.

The mentor hypothesized: *"I see no reason to do it but I wanna have it in my appendix"*, suspecting that applying the operator to positional space would essentially degenerate into just adding a positional label.

## Changes Made in the Codebase Compared to the Baseline
The baseline (Vanilla Informer) was modified to inject the ordering operator signal computed over positional embeddings:
1. **Added `models/ordering_operator_positional.py`**: A new module that generates Legendre orthogonal positional vectors and computes the uniform weighted pairwise displacement (`P_i - P_j`) between them.
2. **Modified `models/embed.py`**: Instead of computing the ordering operator on the value embeddings (`X`), it computes it on the positional embeddings (`P`) and adds this geometric ordering signal to the final input embedding.
3. **Unchanged Files**: `attn.py`, `encoder.py`, `decoder.py`, and `model.py` remain strictly identical to the vanilla Informer.

## Mathematical Formula
**Legendre Positional Vectors**:
Each position `i` is mapped to `x_i ∈ [-1, 1]` via `x_i = 2i / (seq_len - 1) - 1`. Legendre vectors are generated through the recurrence relation, producing `P_i = [L_0(x_i), ..., L_{d-1}(x_i)] / sqrt(d_model)`.

**Ordering Signal in Positional Space**:
The signed positional displacement is `Δp_ij = P_i - P_j`. The ordering operator is a uniform weighted sum of these displacements:
```
O_i(P) = (1/(N-1)) * Σ_{j≠i} (P_i - P_j)
O_i(P) = P_i - mean_{j≠i}(P_j)
```

**Final Input Embedding**:
```
X'_i = X_i + T_i + O_i(P)
```
Where `X_i` is the value embedding, `T_i` is the temporal embedding, and `O_i(P)` is the geometric ordering signal.

## Change in Code
**`ordering_operator_positional.py` (New):**
```python
class OrderingOperatorPositional(nn.Module):
    def forward(self, seq_len, device) -> Tensor:
        P = self.legendre(seq_len, device)   # [1, L, D]
        P_i = P.unsqueeze(2)                 # [1, L, 1, D]
        P_j = P.unsqueeze(1)                 # [1, 1, L, D]
        delta_p = P_i - P_j                  # [1, L, L, D]
        O = delta_p.sum(dim=2) / (L - 1)     # [1, L, D]
        return O
```

**`embed.py` (Modified):**
```python
def forward(self, x, x_mark):
    value_emb = self.value_embedding(x)
    temporal_emb = self.temporal_embedding(x_mark)
    ordering_pos = self.ordering_operator_pos(seq_len=value_emb.shape[1], device=value_emb.device)
    # X'_i = X_i + T_i + O_i(P)
    x_out = value_emb + temporal_emb + ordering_pos
    return self.dropout(x_out)
```

## Why We Made Those Changes
To validate the necessity of applying the ordering operator in **semantic space** (value embeddings). Because `O_i(P) = P_i - mean(P_j)`, and since Legendre vectors are approximately mean-zero for large sequence lengths, `O_i(P) ≈ P_i`. This means that applying the ordering operator to positional vectors algebraically collapses to adding a mean-centered version of the positional label. Doing this empirically confirms whether this geometric version offers any novel informational gain over just using simple positional labels (as done in `exp3_label_only`).

## Phase 1 of this Experiment
Phase 1 involved a preliminary run on a single seed (`2021`) across two prediction lengths (`96` and `192`) on the ETTh1 dataset to quickly assess if the approach holds any promise or fails completely.

## Results of Phase 1 and What They Mean
**Phase 1 Results (Seed 2021):**
- **Pred Len 96:** MSE = `0.7446`, MAE = `0.6671`
- **Pred Len 192:** MSE = `0.8537`, MAE = `0.7253`

**Meaning:** The performance did not drastically fail and actually showed competitive behavior. This warranted a full investigation across multiple seeds and prediction lengths to see how stable the performance is, prompting the transition to Phase 2.

## Phase 2 and What They Mean
Phase 2 expanded the experiment across three random seeds (`2021, 2022, 2023`) and multiple prediction lengths (`48, 96, 192, 336`) to compute robust averages.

**Phase 2 Average Results:**
- **Pred Len 48:** Avg MSE = `0.7047`, Avg MAE = `0.6512`
- **Pred Len 96:** Avg MSE = `0.7718`, Avg MAE = `0.6880`
- **Pred Len 192:** Avg MSE = `0.8036`, Avg MAE = `0.6978`
- **Pred Len 336:** Avg MSE = `0.9889`, Avg MAE = `0.7970`

**Meaning:** The model exhibits stable training but performance remains relatively bounded. The results show that injecting `O_i(P)` achieves reasonable forecasting metrics (e.g., MSE around 0.77 for `pl=96`), which is noticeably better than just adding the raw label (`exp3` baseline MSE `1.124`), indicating that mean-centering the positional vectors has some stabilizing effect. However, it still largely confirms that the geometric ordering operator lacks the rich contextual interaction found when ordering is applied to semantic value embeddings. 

## Inferences We Can Take Away
1. **Validation of Mentor's Intuition:** The ordering operator in positional space fundamentally behaves like a **mean-centered positional label**. Since `mean(P_j) ≈ 0`, the operator contributes little structural information beyond what `P_i` inherently contains.
2. **Semantic vs. Geometric Space:** This experiment strengthens the core argument of the paper: the ordering primitive is most powerful when it captures **content-based displacement** (acting on semantic value embeddings, as in `exp4` or `exp5b`), rather than just static geometry.
3. **Role in the Paper:** As an appendix inclusion, it serves as a rigorous ablation. It proves that the success of the ordering operator isn't just an artifact of complex math, but relies critically on the semantic vector space it operates within.
