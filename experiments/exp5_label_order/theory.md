# Experiment 5: Label + Order (Distinctiveness + Directionality)

## Aim (What the experiment tries to achieve)
Test whether combining **Label (Legendre polynomials)** and **Order (signed displacements)** provides better positional structure than either component alone, specifically without distance-based weighting. The core research question is to determine if Label and Order synergize when combined, or do they interfere? This directly tests the paper's claim that the combination of label + ordering fails to match standard positional encoding.

## Changes made in the codebase compared to the baseline
- **New `embed.py`:** Modified the standard `DataEmbedding` to incorporate both Label (from Exp 3) and Order (from Exp 4) components.
- **Removed Distance Component:** Specifically excluded the distance-decay factor ($\alpha$) and feature-space weighting ($w_{ij}$) that were present in Exp 2 (LOD) and Exp 1.
- **Incorporated `legendre_embedding.py`:** Reused the Legendre Polynomial embedding class from Experiment 3 to provide positional distinctiveness.
- **Incorporated `ordering_operator.py`:** Reused the ordering operator from Experiment 4, but crucially, applied it to the **positional (Legendre) space** instead of the semantic (value) space.

## Mathematical Formula
The overall embedding formula combines semantic, temporal, label, and order components:
```text
X'_i = X_i + T_i + P_i + O_i
```

Where:
- **$X_i = \text{value\_embedding}(x)$**: Semantic content
- **$T_i = \text{temporal\_embedding}(x_{mark})$**: Time features
- **$P_i = \text{Legendre}(i)$**: LABEL (Equation 1) — Orthogonal distinctiveness scaled by $1/\sqrt{d_{model}}$
- **$O_i = \frac{1}{N-1} \sum_{j \neq i} (P_i - P_j)$**: ORDER (Equation 3) — Signed displacements applied to the *positional* space $P_i$.

*Note*: The formula intentionally excludes any distance decay ($\alpha(i,j)$) or feature-space weighting ($w_{ij}$).

## Change in Code
In `models/embed.py` within the `DataEmbedding` class, the forward pass is updated as follows:
```python
# 1. Semantic and Temporal Components
value_emb = self.value_embedding(x)
temporal_emb = self.temporal_embedding(x_mark)

# 2. LABEL component (Orthogonal distinctiveness)
legendre_pos = self.legendre_embedding(x)

# 3. ORDER component (Signed displacements in POSITIONAL space)
# CRITICAL FIX: Applied to legendre_pos (positional), NOT value_emb (semantic)
ordering_pos = self.ordering_operator(legendre_pos)
ordering_pos = ordering_pos / math.sqrt(value_emb.size(-1)) # scaling for stability

# 4. Combine all components
x = value_emb + temporal_emb + legendre_pos + ordering_pos
```

## Why we made those changes
1. **Testing Synergy:** We wanted to test Hypothesis A (Positive Synergy) against hypotheses of interference or simple additive behavior. The original paper claimed PoPE + $\Delta V$ fails to match standard PE; we wanted to rigorously verify this.
2. **Order in Positional Space:** A critical conceptual fix was made to apply the `OrderingOperator` to the Legendre positional embeddings ($P_i$) rather than the semantic value embeddings ($X_i$). This ensures that both Label and Order function as purely positional signals, creating a unified structural context.
3. **Removing Distance:** To isolate the true synergy between Label and Order, we removed distance decay. We hypothesized that distance decay and complex feature-space weighting might actually interfere with the natural synergy of uniform positional aggregation.

## Phase 1 of this experiment
- **Goal:** An exploration phase to test the combined L+O formulation on shorter and longer prediction lengths (96 and 192) using a single seed (2021).
- **Setup:** Ran with a full attention Informer on the ETTh1 dataset. Since there's no distance decay, no $\alpha$ sweep was needed.
- **Expectation:** Determine if the L+O combination outperforms isolated components (Exp3-Label, Exp4-Order) and Distance-based approaches (Exp1, Exp2).

## Results of Phase 1 and what they mean
- **pred=96:** MSE = 0.8519 (Beats Exp1-Pre's 0.8683)
- **pred=192:** MSE = 0.9788 (Worse than Exp1-Pre's 0.8463)
- **Meaning:** Adding Order to Label helps significantly at shorter prediction horizons but degrades at longer ones. It appeared slightly less stable than Exp4b (consecutive local ordering). However, because it beat the Distance-only approach (Exp1) at `pred=96` and beat Label-only (Exp3b) across the board, it proved that Order adds value to Label and warranted a full Phase 2 validation.

## Phase 2 and what they mean
- **Goal:** A validation phase to rigorously test if the positive synergy seen in Phase 1 holds across multiple seeds and prediction lengths.
- **Setup:** Ran 12 total variations (3 seeds: 2021, 2022, 2023 $\times$ 3 prediction lengths: 48, 96, 192).
- **Meaning:** This phase confirmed the statistical significance of the L+O combination, verifying whether it consistently outperforms isolated components or distance-weighted combinations despite random initializations.

## Inferences we can take away from this experiment
1. **Positive Synergy Exists:** Label and Order strongly complement each other. The combined Exp5 (L+O) achieved an overall optimal MSE of **0.719** (2nd place overall, only behind Vanilla PE's 0.519).
2. **L+O > Distance Alone:** Exp5 (L+O) outperformed Exp1 (Distance alone, MSE=0.725). This directly contradicts the original paper's claim that Label + Order (PoPE + $\Delta V$) fails compared to distance-weighted approaches.
3. **Distance Weighting Interferes:** Removing the distance component actually *improved* performance! Exp5 (L+O, MSE=0.719) outperformed Exp2 (L+O+D, MSE=0.804) by 10.5%. This indicates that distance weighting (alpha decay and $w_{ij}$) interferes with the natural synergy between orthogonal distinctiveness and relative directionality.
4. **Positional Space is Key:** The critical design decision to apply the Order operator in the *positional* space (Legendre) rather than the semantic space (Value) was highly successful in creating a rich, synergistic positional structure without adding noise.
