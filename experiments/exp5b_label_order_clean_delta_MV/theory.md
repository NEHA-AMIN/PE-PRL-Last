# Experiment 5b Analysis: Label + Order with Clean Delta (L+O Clean)

## Aim (What the experiment tries to achieve)
The objective of Experiment 5b is to test the **Label + Order** combination using a **clean delta signal** for ordering. Unlike previous experiments where the order (delta) was derived from positional or temporal embeddings, this experiment computes the delta *purely* from the value embeddings (semantic space) without any positional or temporal components. The goal is to determine whether separating positional distinctiveness (Label) and semantic sequential shifts (Order) reduces representation redundancy and improves forecasting performance compared to the original Exp 5 implementation.

## Changes Made in the Codebase Compared to the Baseline
The core difference between Exp 5 and Exp 5b lies in how the delta is computed and injected into the Attention mechanism:

| Aspect | **Exp 5** | **Exp 5b (This Experiment)** |
|--------|-----------|-------------------|
| **Delta Input** | `legendre_pos` ($p_i$) | `value_emb` ($x_i$) |
| **Delta Formula** | $O_i = \frac{1}{L-1}\sum_{j \neq i}(p_i - p_j)$ | $\Delta x_i = x_i - x_{i-1}$ |
| **Delta Type** | Pairwise mean (all positions) | Sequential shift (temporal) |
| **Delta Contains** | Positional differences | Semantic differences |
| **Architecture** | Additive ($Q, K, V \leftarrow x_i + T_i + p_i + O_i$) | Separated ($Q, K \leftarrow x_i + T_i + p_i$; $V \leftarrow \Delta x$) |

## Mathematical Formula
The attention input is now split into two complementary spaces:

**1. Queries (Q) and Keys (K) (Label & Context):**
Computed from the combined embeddings:
$$Q, K \leftarrow \text{project}(x_i + T_i + p_i)$$
Where:
- $x_i = \text{TokenEmbedding}(\text{input})$
- $T_i = \text{TemporalEmbedding}(\text{time\_mark})$
- $p_i = \text{Legendre}(i)$ (Label - positional distinctiveness)

**2. Values (V) (Order):**
Computed purely from the semantic sequential shift:
$$V \leftarrow \text{project}(\Delta x)$$
Where the clean delta is:
$$\Delta x_i = x_i - x_{i-1} \quad \text{with} \quad \Delta x_0 = 0$$

## Change in Code
To achieve this split architecture, several key codebase modifications were implemented:
- **`embed.py`**: Modified `DataEmbedding` to compute the clean delta using `value_emb - torch.roll(value_emb, shifts=1, dims=1)`. It now returns a tuple: `(combined_emb, delta_x)`.
- **`attn.py`**: The `AttentionLayer` was updated to accept `delta_values`. While `queries` and `keys` are projected from `combined_emb`, `values` are projected directly from `delta_x`.
- **`encoder.py`**: The `EncoderLayer` forwards `delta_x` explicitly into the `delta_values` argument of the inner attention.
- **`model.py`**: The `Informer` model handles the dual return from the encoder embedding and passes `delta_enc` through the encoder layers. The decoder remains unchanged.

## Why We Made Those Changes
In Exp 5, the delta was computed from Legendre polynomials. Because it was an average of pairwise differences across *all* positions in the sequence, it resulted in an ordering signal that was fundamentally **positional**. Since the Label ($p_i$) was also positional, combining them in the same embedding space likely caused redundancy and feature collision. 

By computing the delta from raw values ($x_i - x_{i-1}$) and pushing it exclusively to the Values ($V$) projection, we create a strictly **semantic** ordering signal. This orthogonalizes the Label (positional) and Order (semantic) components, hypothesizing that this cleaner separation will yield better multi-horizon predictions.

## Phase 1 of this Experiment
Phase 1 was designed as a validation check to detect whether the "L+O Clean" architecture could beat the best previous reference results (Exp 5 positional delta or Exp 1 distance baseline) at short and mid horizons.
- **Configuration:** Single fixed configuration (no alpha sweep).
- **Prediction Lengths:** 96 and 192.
- **Seed:** 2021.
- **Success Criteria:** To proceed to Phase 2, Exp 5b needed to beat Exp 5 (MSE ~0.719) at both prediction lengths, or at least show mixed success against Exp 1 (MSE ~0.725).

## Results of Phase 1 and What They Mean
Based on the execution logs in Phase 1, the results were extracted as follows:

| RUN_ID | MSE | MAE |
|--------|-----|-----|
| `exp5b_ph1_ETTh1_lod_clean_pred96_seed2021` | **0.9004** | **0.7599** |
| `exp5b_ph1_ETTh1_lod_clean_pred192_seed2021` | **0.9511** | **0.7929** |

**What they mean:** 
The results are substantially **worse** than the original Exp 5 (MSE ~0.719) and the Exp 1 distance baseline (MSE ~0.725). In fact, an MSE of > 0.804 means this formulation underperforms compared to even the worst prior Label+Order variants (like Exp 2). The theoretical purity of the clean semantic delta did not translate to practical model performance.

## Phase 2 and What They Mean
Because the Phase 1 results completely failed the success criteria (they were worse everywhere), **Phase 2 was skipped**. Following the experiment's internal Decision Guide ("NO (worse everywhere) → Document as negative"), proceeding to a broader sweep (Phase 2) across multiple seeds, prediction lengths, and alphas was deemed unjustified. The hypothesis was cleanly disproven in Phase 1.

## Inferences We Can Take Away From This Experiment
1. **Semantic Delta is Insufficient for Values:** While separating positional queries/keys from semantic values mathematically reduces redundancy, stripping the Value projection ($V$) of all absolute positional and temporal context severely degrades the Transformer's aggregation ability. The model loses track of *when* a semantic shift occurred.
2. **Positional Overlap Might Be Necessary:** The "redundancy" observed in Exp 5 might actually be a required inductive bias. The attention mechanism seemingly relies on having positional/temporal markers embedded in the Values to properly weight and aggregate future states.
3. **Sequential Shifts $\neq$ Good Ordering:** A simple $x_i - x_{i-1}$ delta might be too noisy or too local to serve as a robust ordering signal across a long sequence. It fails to provide the stable, global gradient that a standard positional encoding or a distance-based decay provides.
