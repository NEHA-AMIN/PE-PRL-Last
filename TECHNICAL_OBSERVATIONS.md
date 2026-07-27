# Technical Observations & Design Choices

**Project:** Position-Relative Learning (PRL) Ablation Study  
**Dataset:** ETTh1, ETTm1  
**Model:** Informer with Modified Position Encodings  
**Last Updated:** 2026-03-27

---

## Table of Contents

1. [Delta Boundary Condition Analysis](#1-delta-boundary-condition-analysis)
2. [Implementation Details](#2-implementation-details)
3. [Design Choices & Limitations](#3-design-choices--limitations)
4. [Performance Observations](#4-performance-observations)
5. [Architectural Insights](#5-architectural-insights)
6. [Future Improvements](#6-future-improvements)
7. [Cross-Experiment Consistency](#7-cross-experiment-consistency)

---

## 1. Delta Boundary Condition Analysis

### Issue: Zero-Boundary Treatment for Position 0

**Discovered:** 2026-03-27  
**Experiments Affected:** Exp 5b (Label + Order Clean Delta), and all delta-based experiments

#### Implementation

**Location:** `experiments/exp5b_label_order_clean_delta/models/embed.py:142-144`

```python
# Compute delta: x_i - x_{i-1}
delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)
# Zero out first position (no previous value)
delta_x[:, 0, :] = 0.0
```

**Result:** Position 0 carries a literal zero vector `[0, 0, ..., 0]` into the Value matrix.

#### Impact on Attention Mechanism

**In Attention Computation (`attn.py:66`):**
```python
V = torch.einsum("bhls,bshd->blhd", A, values)
```

For any query position `i` attending to position 0:
- Attention weight: `A[i, 0]` (can be non-zero)
- Value vector: `values[0] = 0` (zero vector)
- **Contribution: `A[i, 0] × 0 = 0`** (wasted attention weight)

**Implications:**
1. **Attention Waste:** Any attention weight assigned to position 0 contributes nothing to output
2. **Information Loss:** First token's semantic content is completely ignored in value aggregation
3. **Asymmetry:** Position 0 is fundamentally different from all other positions
4. **Effective Sequence Length:** Model effectively operates on L-1 positions for value aggregation

#### Alternative Approaches Considered

| Approach | Implementation | Pros | Cons | Decision |
|----------|---------------|------|------|----------|
| **Zero (Current)** | `delta_x[:, 0, :] = 0.0` | Simple, no artifacts | Wastes attention, loses info | ✅ **Keep for now** |
| **Identity** | `delta_x[:, 0, :] = value_emb[:, 0, :]` | First token contributes, represents "initial state" | Breaks delta semantics | Consider for future |
| **Forward Difference** | `delta_x[:, 0, :] = value_emb[:, 1, :] - value_emb[:, 0, :]` | Maintains difference semantics | Different direction (forward vs backward) | Consider for future |
| **Learnable Init** | `delta_x[:, 0, :] = nn.Parameter(...)` | Model learns optimal boundary | Adds parameters, complexity | Consider for future |

#### Rationale for Current Choice

**Decision: Keep zero-boundary treatment as-is**

**Reasons:**
1. **Experimental Consistency:** Zero-boundary is used across ALL delta-based experiments (Exp 4, 5, 5b, 6)
2. **Fair Comparison:** Changing it now would confound comparisons between experiments
3. **Simplicity:** Avoids introducing new variables mid-experiment
4. **Documentation:** Can be addressed in paper's limitations/appendix section

**Quote from Discussion:**
> "The zero boundary is consistent across all your experiments that use delta — so it won't confound your comparisons. Changing it now risks introducing a new variable mid-experiment."

#### Future Work

**Potential Improvements:**
- Test identity boundary: `delta_x[:, 0, :] = value_emb[:, 0, :]`
- Compare performance with/without boundary treatment
- Analyze attention weight distribution to position 0
- Consider learnable boundary initialization

**Paper Section:** Include in "Limitations and Future Work" or "Implementation Details" appendix

---

## 2. Implementation Details

### 2.1 Delta Computation Method

**Experiment:** Exp 5b (Label + Order Clean Delta)

**Method:** Sequential backward difference using `torch.roll`
```python
delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)
```

**Equivalent to:**
```python
for i in range(1, L):
    delta_x[i] = value_emb[i] - value_emb[i-1]
delta_x[0] = 0  # Boundary condition
```

**Efficiency:** O(1) memory overhead, vectorized operation

### 2.2 Embedding Architecture

**Experiment:** Exp 5b

**Forward Pass Order:**
1. `value_emb = TokenEmbedding(x)` → x_i
2. `delta_x = x_i - x_{i-1}` → Δx_i (computed from raw x)
3. `temporal_emb = TemporalEmbedding(x_mark)` → T_i
4. `legendre_pos = LegendreEmbedding(x)` → p_i
5. `combined = x_i + T_i + p_i` → x̃_i (for Q/K)
6. Return: `(combined, delta_x)`

**Critical:** Delta is computed BEFORE adding positional/temporal components.

### 2.3 Attention Layer Modifications

**Q/K Projections:** Use `combined_emb = x_i + T_i + p_i`
```python
queries = self.query_projection(queries)  # From combined
keys = self.key_projection(keys)          # From combined
```

**V Projection:** Uses `delta_x = x_i - x_{i-1}`
```python
if delta_values is not None:
    values = self.value_projection(delta_values)  # From delta
```

**Separation:** Similarity computation (Q·K) uses full context, value aggregation uses temporal signal.

### 2.4 Distance Decay Parameter (Experiment 1)

**Discovered:** 2026-03-27
**Experiments Affected:** Exp 1 (Distance Decay), Exp 1-Post (Distance Post-Softmax)

#### Distance Decay Formula

**Formula:** αᵢⱼ = 1 / (1 + |i - j|^a)

Where:
- `i, j` are query and key position indices
- `|i - j|` is the absolute distance between positions
- `a` is the decay exponent parameter (`decay_a`)

#### Implementation Locations

**Pre-Softmax (exp1_distance_decay):**
- **Location:** `experiments/exp1_distance_decay/models/attn.py:24-44`
- **Application:** `scores = scores * alpha` (line 35)
- **Operation:** softmax(scale · (QKᵀ ⊙ α))

**Post-Softmax (exp1_distance_post_softmax):**
- **Location:** `experiments/exp1_distance_post_softmax/models/attn.py:24-45`
- **Application:** `A = A * alpha` (line 43)
- **Operation:** softmax(scale · QKᵀ) ⊙ α

#### Parameter Configuration

**Command-Line Configurable:** ✅ Yes

The `decay_a` parameter is passed via command-line argument and flows through the architecture:

1. **Run Script:** `--decay_a 1.0` (default value)
   - `experiments/exp1_distance_decay/run_exp1.sh:60`
   - `experiments/exp1_distance_post_softmax/run_exp1_post.sh:61`

2. **Model Constructor:** `model.py:16`
   ```python
   def __init__(self, ..., decay_a=1.0):
   ```

3. **Attention Layer:** `model.py:31, 50, 52`
   ```python
   AttentionLayer(Attn(..., decay_a=decay_a), ...)
   ```

4. **Attention Module:** `attn.py:11, 17`
   ```python
   def __init__(self, ..., decay_a=1.0):
       self.decay_a = decay_a
   ```

**No Separate Folders Needed:** Different alpha values can be tested by simply changing the `--decay_a` argument in the run script.

#### Application Method

**Multiplicative (NOT Additive):**
- Unlike ALiBi which adds a bias term to attention scores
- These experiments multiply by a decay factor
- Pre-softmax: modifies raw attention scores before normalization
- Post-softmax: modifies attention probabilities after normalization

#### Testing Different Alpha Values

To test different decay rates, modify the run script:

```bash
--decay_a 0.5   # Slower decay (wider attention range)
--decay_a 1.0   # Default (linear decay)
--decay_a 2.0   # Faster decay (more local attention)
```

**Interpretation:**
- `a = 0.5`: Distance penalty grows slowly → model attends to distant positions
- `a = 1.0`: Linear distance penalty → balanced local/global attention
- `a = 2.0`: Distance penalty grows quickly → model focuses on nearby positions

---

## 3. Design Choices & Limitations

### 3.1 Zero-Boundary Limitation

**Status:** Known limitation, documented above  
**Impact:** Moderate - affects first position only  
**Mitigation:** Consistent across experiments for fair comparison  
**Future:** Consider alternative boundary conditions

### 3.2 Memory Overhead

**Issue:** Storing both `combined_emb` and `delta_x`

**Memory Usage:**
- Combined embedding: `[B, L, D]`
- Delta embedding: `[B, L, D]`
- **Total overhead:** ~2x for embeddings

**Impact:** Manageable for typical batch sizes (32) and sequence lengths (96-720)

**Mitigation:** None needed currently, but could optimize if memory becomes constraint

### 3.3 Gradient Flow

**Observation:** Value embedding receives gradients from two paths:

1. **Q/K Path:** `combined_emb → value_emb`
2. **V Path:** `delta_x → value_emb`

**Implication:** Value embedding is trained more heavily than positional/temporal components

**Status:** Intentional design - value embedding should be most important

---

## 4. Performance Observations

### 4.1 Experiment Results

**To be filled after training runs complete**

| Experiment | Pred Len | MSE | MAE | Notes |
|------------|----------|-----|-----|-------|
| Exp 5b | 48 | TBD | TBD | |
| Exp 5b | 96 | TBD | TBD | |
| Exp 5b | 192 | TBD | TBD | |
| Exp 5b | 336 | TBD | TBD | |
| Exp 5b | 720 | TBD | TBD | |

### 4.2 Training Observations

**To be filled during training:**
- Convergence speed
- Loss curves
- Attention pattern analysis
- Gradient statistics

---

## 5. Architectural Insights

### 5.1 Signal Separation

**Key Insight:** Separating positional (Label) and semantic (Order) signals

**Exp 5 Approach:**
- Delta from positional space: `O_i = (1/(L-1)) · Σ(p_i - p_j)`
- Both Label and Order in positional space → potential redundancy

**Exp 5b Approach:**
- Delta from semantic space: `Δx_i = x_i - x_{i-1}`
- Label (positional) and Order (semantic) are orthogonal → complementary

**Hypothesis:** Orthogonal signals should improve learning efficiency

### 5.2 Attention Mechanism Design

**Standard Attention:**
```
Q, K, V all from same embedding
```

**Exp 5b Attention:**
```
Q, K from combined (x + T + p)  → Similarity in full context
V from delta (x_i - x_{i-1})    → Aggregation of temporal signal
```

**Rationale:** Decouple "what to attend to" from "what to aggregate"

---

## 6. Future Improvements

### 6.1 Boundary Condition Experiments

**Priority:** Medium  
**Effort:** Low

**Tasks:**
- [ ] Implement identity boundary: `delta_x[:, 0, :] = value_emb[:, 0, :]`
- [ ] Run ablation study comparing boundary conditions
- [ ] Analyze attention weight distribution to position 0
- [ ] Document performance impact

### 6.2 Memory Optimization

**Priority:** Low (if needed)  
**Effort:** Medium

**Ideas:**
- Compute delta on-the-fly in attention layer
- Use gradient checkpointing for embeddings
- Reduce precision for delta storage

### 6.3 Alternative Delta Formulations

**Priority:** Low  
**Effort:** Medium

**Ideas:**
- Multi-step deltas: `x_i - x_{i-k}` for k > 1
- Weighted deltas: `Σ w_k(x_i - x_{i-k})`
- Learnable delta operator

### 6.4 Attention Pattern Analysis

**Priority:** High  
**Effort:** Low

**Tasks:**
- [ ] Visualize attention weights for position 0
- [ ] Compare attention patterns: Exp 5 vs Exp 5b
- [ ] Analyze which positions attend to delta vs combined
- [ ] Document findings

---

## 7. Cross-Experiment Consistency

### 7.1 Delta-Based Experiments

**Experiments using delta/ordering signals:**

| Experiment | Delta Source | Delta Formula | Boundary | Consistent? |
|------------|--------------|---------------|----------|-------------|
| Exp 4 | Ordering operator | `O_i = Σ(x_i - x_j)` | Zero | ✅ |
| Exp 5 | Legendre positions | `O_i = Σ(p_i - p_j)` | N/A (mean) | ✅ |
| Exp 5b | Value embeddings | `Δx_i = x_i - x_{i-1}` | Zero | ✅ |
| Exp 6 (LOD) | TBD | TBD | TBD | TBD |

**Consistency Check:** All experiments using sequential delta use zero-boundary ✅

### 7.2 Embedding Architecture

**Common components across experiments:**
- Value embedding: `TokenEmbedding` (1D convolution)
- Temporal embedding: `TemporalEmbedding` or `TimeFeatureEmbedding`
- Positional embedding: Varies by experiment

**Variations:**
- Exp 1: Distance decay operator
- Exp 3: Legendre polynomials only
- Exp 4: Ordering operator only
- Exp 5: Legendre + ordering (positional delta)
- Exp 5b: Legendre + ordering (semantic delta)

---

## Usage Guidelines

### Adding New Observations

**When to add:**
- During code reviews
- After training runs
- During result analysis
- In architecture discussions
- When discovering bugs or limitations

**How to add:**
1. Choose appropriate section (or create new one)
2. Include date discovered
3. Provide code references with line numbers
4. Explain impact and implications
5. Document decision rationale
6. Add to relevant comparison tables

**Format:**
```markdown
### X.Y Observation Title

**Discovered:** YYYY-MM-DD  
**Experiments Affected:** List experiments

**Description:** Clear explanation

**Code Reference:** `path/to/file.py:line_numbers`

**Impact:** How it affects results/comparisons

**Decision:** What was decided and why
```

### Maintaining This Document

- Update after each major experiment
- Review before writing paper sections
- Use for limitations/appendix content
- Reference in code comments
- Keep cross-references updated

---

## References

- Main project: `/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1`
- Experiment READMEs: `experiments/*/README.md`
- Training logs: `results/*/training_log.txt`
- Code: `experiments/*/models/`

---

**Maintained by:** Research Team  
**Purpose:** Document implementation details, design choices, and observations for paper writing and future work  
**Status:** Living document - update as discoveries are made


# Positional Space vs Semantic Space — A Complete Explanation

## 1. Starting From Scratch: What Problem Does a Transformer Have?

A transformer processes a sequence of tokens as a **set**, not a sequence. The self-attention operation:

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

is **permutation-equivariant** — if you shuffle the input tokens, the output shuffles identically. It contains no built-in notion of "token 3 came before token 7." The model is blind to order.

To fix this, every token's representation must be augmented with *some* signal that encodes its position. The question is: **what kind of information should that signal encode, and where does it come from?**

This question is exactly the split between **positional space** and **semantic space** in this codebase.

---

## 2. What Is Semantic Space?

**Semantic space** is the vector space spanned by $X_i$ — the **content** of the data at timestep $i$.

In this codebase, $X_i$ is produced by `TokenEmbedding`:

```python
# Informer2020-original/models/embed.py — Lines 30–38
self.tokenConv = nn.Conv1d(
    in_channels=c_in,      # 7 raw ETTh1 features
    out_channels=d_model,  # 512-dimensional output
    kernel_size=3, padding=padding, padding_mode='circular'
)

def forward(self, x):
    x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
    return x   # [B, L, 512]
```

The 7 raw input features at each timestep — oil temperature (OT), load, transformer readings — are projected from 7 dimensions into 512 dimensions using a 1D convolution with kernel size 3. This captures **local interactions between neighbouring raw features** to produce a rich 512-dimensional vector that represents the *meaning* or *content* of that timestep.

The **key property of semantic space:** it is **input-dependent**. Every different batch of ETTh1 data produces different $X_i$ vectors. The 3am reading on a January morning has a completely different $X_i$ than the 3pm reading on a summer afternoon — because the actual physical measurements differ. Semantic space answers the question: **"What is happening at this timestep?"**

Formally, semantic space is:

$$
X_{\mathrm{sem}}
=
\left\{
X_i\in\mathbb{R}^{512}
\mid
X_i=f_\theta(\mathrm{raw\_data}[i-1:i+2])
\right\}
$$

where $f_\theta$ is the Conv1D with learnable parameters $\theta$. It is **batch-dependent, data-dependent, and changes during training** as $\theta$ updates through backpropagation.

---

## 3. What Is Positional Space?

**Positional space** is the vector space spanned by $P_i$ — the **location identity** of a timestep, independent of what data is observed there.

In this codebase, $P_i$ is produced by `LegendrePositionEmbedding`:

```python
# experiments/exp2_full_paper/models/legendre_embedding.py — Lines 52–70
positions = 2.0 * np.arange(seq_len) / (seq_len - 1) - 1.0  # [-1, 1]

for k in range(d_model):          # k = 0, 1, 2, ..., 511
    L_k = legendre(k)             # k-th Legendre polynomial
    P[:, k] = L_k(positions)      # evaluate at all positions

P = torch.FloatTensor(P) / math.sqrt(d_model)   # scale

self.register_buffer('legendre_emb', P)         # NON-TRAINABLE, fixed forever
```

Position 0 always maps to $\hat{i}=-1$. Position 47 (the middle of a 96-step window) always maps to $\hat{i}=0$. Position 95 always maps to $\hat{i}=+1$. These never change — **regardless of what data sits at those positions, regardless of the training epoch, regardless of the batch.**

The result is a fixed matrix

$$
P\in\mathbb{R}^{96\times512}
$$

where:

- Row $i$ is the "address" of the $i$-th slot in the sequence
- Each row is the evaluation of 512 Legendre polynomials at a single normalized position
- No two rows are the same (each position has a unique fingerprint)

The **key property of positional space:** it is **input-independent**. $P_i$ is purely a function of the index $i$, nothing else. Positional space answers the question: **"Where in the sequence is this slot?"**

Formally:

$$
X_{\mathrm{pos}}
=
\left\{
P_i\in\mathbb{R}^{512}
\mid
P_{i,k}
=
L_k\!\left(\frac{2i}{N-1}-1\right)\!/\sqrt{d}
\right\}
$$

It is **batch-independent, data-independent, and never changes after construction.**

---

## 4. The Fundamental Difference: A Concrete Example

Take ETTh1. Suppose at position $i=20$ in batch 1, the oil temperature is **35°C** (a hot summer reading), and at position $i=20$ in batch 2, the oil temperature is **8°C** (a cold winter reading).

| | Semantic $X_{20}$ | Positional $P_{20}$ |
|---|---|---|
| **Batch 1 (35°C)** | `[0.83, -0.21, 0.44, ...]` (512-dim, depends on data) | $[L_0(-0.58),\ L_1(-0.58),\ \ldots]/\sqrt{512}$ |
| **Batch 2 (8°C)** | `[-0.52, 0.61, -0.19, ...]` (completely different) | $[L_0(-0.58),\ L_1(-0.58),\ \ldots]/\sqrt{512}$ |
| **Same?** | ❌ **No** — reflects the data | ✅ **Yes** — reflects the index |

$X_{20}$ is different in every batch because the content changes. $P_{20}$ is identical in every batch because the position doesn't change. This is the entire distinction.

The same logic applies to the sinusoidal PE in the baseline at `Informer2020-original/models/embed.py:113`:

```python
x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
# ─── semantic ───────────  ─── positional ───────────  ─── temporal (also positional) ───
```

`position_embedding(x)` uses `x` only to read `x.size(1)` (the sequence length) — it never looks at the actual values of `x`. It's positional space by design.

## 7. Why Separation Matters for the Model

The transformer's embedding layer must answer **two distinct questions** for every token:

1. **What are you?** (semantic content → helps attention know which values are relevant)
2. **Where are you?** (positional address → helps attention know the ordering)

The final embedding at `exp2_full_paper/models/embed.py:159` is:

```python
x = value_emb + temporal_emb + legendre_pos + distance_pos
#     ── WHAT ──   ── WHEN ────   ── WHERE ── + ── WHERE (relative) ──
```

When the ordering operator is applied in **positional space** ($P_i$), the resulting ordering signal $O_i$ is **purely positional**. It tells the attention mechanism **how far from the sequence center this position lies** (in the Legendre polynomial space). The signal is therefore orthogonal to the semantic (**WHAT**) representation.

By contrast, if the ordering operator is applied in **semantic space** ($X_i$), the resulting $O_i$ becomes a **mixed WHAT + WHERE signal**. Instead of encoding only position, it represents something like:

> "How different is this token's meaning from the average meaning?"

This entangles semantic content with positional information, which introduces several scientific problems.

### 1. Gradient Interference

During backpropagation, the parameters of the `TokenEmbedding` layer receive gradients from multiple paths simultaneously:

- the direct semantic embedding path ($X_i$),
- the ordering path ($O_i$, computed from $X_i$),
- and the task loss.

Because these objectives are coupled, the gradients can conflict with one another, making optimization more difficult.

### 2. Non-Stationarity

When ordering is computed from semantic embeddings, the ordering signal changes throughout training because $X_i$ changes whenever the embedding parameters $\theta$ are updated.

A positional embedding ($P_i$), however, is **stationary**—it is fixed from the first epoch onward. The model always observes the same positional codes, providing a stable optimization target and more consistent gradients.

### 3. Content Confounding

In Experiment 4 (semantic ordering), consider a region of the ETTh1 dataset where all measurements are nearly identical (a flat segment).

In that case,

$$
X_i - X_j \approx 0
$$

for most neighboring positions, which implies

$$
O_i \approx 0.
$$

The ordering signal effectively disappears for the entire segment, even though the sequence positions themselves remain perfectly well-defined.

Thus, the model temporarily loses its positional information simply because the input values happen to be similar—a behavior that should never occur in a true positional encoding.