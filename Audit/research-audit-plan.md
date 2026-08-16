# Research Implementation Audit Plan
## Ordering-Aware Transformer Experiments — Forensic Codebase Audit

### Top-Level Overview

This plan documents the complete, evidence-grounded audit of the repository at
`/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1`.

The research goal is:

> Replace traditional positional information with **ordering information** derived
> directly from the input sequence. Focus is **ORDER ONLY**.

The intended design calls for exactly **4 experiments**: two formulas × two
injection spaces.

| Axis | Option A | Option B |
|------|----------|----------|
| Formula | **Formula 1 — Delta Ordering** `Δ(x_i) = x_i − x_{i−1}` | **Formula 2 — Normalized Delta Ordering** `Δ(x_i) / x̄` |
| Space | **Positional Space** — replaces `p_i` | **Semantic (Value) Space** — modifies `x_i` |

Expected experiments:

| # | Formula | Space |
|---|---------|-------|
| E1 | Delta | Positional |
| E2 | Delta | Semantic |
| E3 | Normalized Delta | Positional |
| E4 | Normalized Delta | Semantic |

---

## STEP 1 — Full Experiment Inventory

The repository contains the following ordering-related experiment folders:

| Folder | Notes |
|--------|-------|
| `Archived_experiments/exp4_order_only/` | Archived — pairwise displacement O(L²D) in **semantic** space |
| `experiments/exp4a_order_only_mentor_version/` | Delta in **attention value matrix** (semantic space, consecutive) |
| `experiments/exp4b_order_input_mentor_version/` | Delta on **Legendre vectors in attention V** (positional space, consecutive) |
| `experiments/exp4b_order_input_position/` | Pairwise displacement operator on **Legendre/positional** embeddings |
| `experiments/exp4_ordering_new_sem_space/` | **Normalized** delta in **semantic** space → injected at embedding layer |
| `experiments/exp5_ordering_new_pos_space/` | **Normalized** delta in **positional** (Legendre) space → injected at embedding layer |
| `experiments/exp5_label_order/` | Label (Legendre) + pairwise ordering in positional space (hybrid) |
| `experiments/exp5b_label_order_clean_delta_MV/` | Label (Legendre) in Q/K + consecutive delta in V (hybrid) |
| `experiments/exp5_label_order/` | Legendre label + pairwise ordering signal (hybrid) |
| `experiments/exp6_lod_pre/` | Full LOD (Label+Order+Distance) pre-softmax (hybrid) |
| `experiments/exp6_lod_post/` | Full LOD (Label+Order+Distance) post-softmax (hybrid) |
| `experiments/exp3_label_only/` | Legendre label only, no ordering |
| `experiments/exp2_full_paper/` | Full LOD with distance decay (hybrid) |

---

## STEP 2 — Complete Per-Experiment Classification

### A. `experiments/exp4_ordering_new_sem_space/` ← **PRIMARY CANDIDATE for E4 (Normalized Delta, Semantic)**

**Files:**
- `models/embed.py` — class `DataEmbedding_ordering_sem`
- `models/model.py` — `pe_mode='ordering_sem'` selects this class
- `exp4_ordering_new_sem_space_ph1.sh` — `--pe_mode ordering_sem`

**What signal is computed** ([`embed.py:162-171`](experiments/exp4_ordering_new_sem_space/models/embed.py:162)):
```python
delta = torch.zeros_like(val)
delta[:, 1:, :] = val[:, 1:, :] - val[:, :-1, :]   # X_i - X_{i-1}

x_bar = val.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)  # (1/N)·sum_i ||X_i||_2

ordering = delta / (x_bar + 1e-8)        # normalized delta
```

**Injection formula:**
```
X'_i = X_i + T_i + O_i^sem
```
Sinusoidal PE excluded. Legendre excluded.

**Injection point:** Embedding layer (pre-attention).

**Modifies:**
- Semantic/value embeddings: ✅ YES (ordering signal derived from `X_i = TokenEmbedding(x)`)
- Positional embeddings: ❌ NO (sinusoidal PE not used)

**Classification: B — Normalized Delta Ordering, Semantic Space**

**Match to intended design:** The formula implements `Δ(x_i) / x̄` where `x̄` is defined as
the **mean of L2 norms** `(1/N)·Σ_i ||X_i||₂`, which is mathematically equivalent to dividing
by the mean embedding magnitude. This is a valid implementation of "normalized by sequence mean."

**Research impact of formula deviation:** The design says `x̄ = (x₁+x₂+…+xₙ)/n`. The code
uses `(1/N)·Σ_i ||X_i||₂` (mean of L2 norms of embedding vectors, a **scalar**) rather than
the vector mean. This is a **deliberate design choice** — it makes the normalization scalar
and scale-invariant, not a vector mean. The results from this formula are not identical to
dividing by the vector mean.

---

### B. `experiments/exp5_ordering_new_pos_space/` ← **PRIMARY CANDIDATE for E3 (Normalized Delta, Positional)**

**Files:**
- `models/embed.py` — class `DataEmbedding_ordering_pos`
- `models/legendre_embedding.py` — `LegendrePositionEmbedding` (Legendre polynomial buffer)
- `models/model.py` — `pe_mode='ordering_pos'`
- `exp5_ordering_new_pos_space_ph1.sh` — `--pe_mode ordering_pos`

**What signal is computed** ([`embed.py:172-181`](experiments/exp5_ordering_new_pos_space/models/embed.py:172)):
```python
delta_p = torch.zeros_like(leg_d)
delta_p[:, 1:, :] = leg_d[:, 1:, :] - leg_d[:, :-1, :]  # P_i - P_{i-1}

p_bar = leg_d.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)  # (1/N)·sum_i ||P_i||_2

ordering = delta_p / (p_bar + 1e-8)      # normalized positional delta
```

**Injection formula:**
```
X'_i = X_i + T_i + P_i + O_i^pos
```
Sinusoidal PE excluded. Legendre label **also included** alongside the ordering signal.

**CRITICAL NOTE:** This experiment **also adds `P_i` (Legendre label) directly** in addition to
the normalized delta ordering signal. This means it is:
- `X_i` — semantic embedding
- `T_i` — temporal embedding
- `P_i` — Legendre label (raw positional vector)
- `O_i^pos = (P_i - P_{i-1}) / p̄` — normalized positional delta

**It is NOT a pure ordering experiment.** It also contains the Label component. This matches
the classification **G — Label + Ordering** (in positional space with normalization).

**Injection point:** Embedding layer (pre-attention).

**Modifies:**
- Semantic/value embeddings: ❌ NO
- Positional embeddings: ✅ YES (ordering derived from Legendre P_i)

**Classification: G — Normalized Delta Ordering + Label (Legendre), Positional Space**

---

### C. `experiments/exp4a_order_only_mentor_version/` ← **CANDIDATE for E2 (Delta Ordering, Semantic Space, Value Matrix)**

**Files:**
- `models/embed.py` — `DataEmbedding` (standard, no ordering at embedding layer)
- `models/attn.py` — `FullAttention.forward()` modified

**What signal is computed** ([`attn.py:35-39`](experiments/exp4a_order_only_mentor_version/models/attn.py:35)):
```python
delta_values = torch.zeros_like(values)
delta_values[:, 1:, :, :] = values[:, 1:, :, :] - values[:, :-1, :, :]
V = torch.einsum("bhls,bshd->blhd", A, delta_values)
```

**Injection formula (reconstructed):**
```
V_i = Σ_j A_ij · Δv_j
where Δv_j = V_j − V_{j−1}  (consecutive delta in value matrix post-projection)
```

The `values` tensor at this point has already been projected by `value_projection` (a linear
layer). The delta is computed on projected values, not on `X_i` directly.

**Injection point:** Inside `FullAttention.forward()` — the value tensor is replaced by its
consecutive differences **after** the `value_projection` linear layer.

**Modifies:**
- Semantic/value embeddings: ✅ YES (the projected value representations are differenced)
- Positional embeddings: ❌ NO (embedding layer is unchanged standard)

**Classification: A — Delta Ordering, Semantic (Projected Value) Space**

**Formula deviation note:** The design says `Δ(x_i) = x_i − x_{i−1}` replacing `p_i`. In
this implementation, the delta is applied to `V_j = W_V · x_j`, not to `x_j` directly. This
is: `ΔV_j = W_V · x_j − W_V · x_{j−1} = W_V · Δx_j`. The linear projection distributes over
the subtraction, so this is mathematically equivalent to projecting the delta. The final V
vector seen by the transformer is `Σ_j A_ij · W_V · Δx_j`, not the standard `Σ_j A_ij · W_V · x_j`.

---

### D. `experiments/exp4b_order_input_position/` ← **CANDIDATE for E1 (Delta Ordering, Positional Space)**

**Files:**
- `models/embed.py` — class `DataEmbedding` using `OrderingOperatorPositional`
- `models/ordering_operator_positional.py` — `OrderingOperatorPositional` + `LegendreEmbedding`

**What signal is computed** ([`ordering_operator_positional.py:127-141`](experiments/exp4b_order_input_position/models/ordering_operator_positional.py:127)):
```python
P = self.legendre(seq_len, device)      # [1, L, D]  Legendre vectors
P_i = P.unsqueeze(2)                   # [1, L, 1, D]
P_j = P.unsqueeze(1)                   # [1, 1, L, D]
delta_p = P_i - P_j                    # [1, L, L, D]  ALL pairwise differences
# Mask diagonal, then sum
O = delta_p.sum(dim=2) / (L - 1)      # (1/(N-1)) · Σ_{j≠i} (P_i − P_j)
```

**Injection formula:**
```
X'_i = X_i + T_i + O_i
where O_i = (1/(N-1)) · Σ_{j≠i} (P_i − P_j)
```

**CRITICAL:** This is **not** consecutive delta `P_i − P_{i−1}`. It is a **pairwise
all-to-all displacement sum**. The formula `(1/(N-1)) · Σ_{j≠i} (P_i − P_j)` is algebraically
equivalent to `P_i − P̄` (position i's Legendre vector minus the mean Legendre vector). It does
NOT implement `Δ(p_i) = p_i − p_{i−1}`.

**Injection point:** Embedding layer (pre-attention).

**Modifies:**
- Semantic/value embeddings: ❌ NO
- Positional embeddings: ✅ YES (ordering derived from Legendre space)

**Classification: Not A or B — this is pairwise centroid deviation, not consecutive delta.**

---

### E. `experiments/exp4b_order_input_mentor_version/` ← Delta Ordering in Legendre Positional Space (Attention Value)

**Files:**
- `models/embed.py` — returns `(x_out, P)` where `P` = Legendre vectors
- `models/attn.py` — computes `Δp_i = P_i − P_{i−1}` in value matrix

**What signal is computed** ([`attn.py:36-39`](experiments/exp4b_order_input_mentor_version/models/attn.py:36)):
```python
delta_p = torch.zeros_like(p_emb)
delta_p[:, 1:, :, :] = p_emb[:, 1:, :, :] - p_emb[:, :-1, :, :]
V = torch.einsum("bhls,bshd->blhd", A, delta_p)
```

**Injection formula:**
```
V = A · ΔP  where ΔP_i = P_i − P_{i−1}  (Legendre consecutive delta, post-projection)
Input: X'_i = X_i + T_i  (no positional embedding at input)
```

**Classification: A — Delta Ordering, Positional Space** (consecutive delta on Legendre vectors, injected into attention value matrix)

---

### F. `Archived_experiments/exp4_order_only/` — Archived, All-Pairs Semantic Ordering

**Classification: Not A or B** — pairwise displacement `(1/(N-1)) · Σ_{j≠i} (X_i − X_j)` in semantic space. This is NOT consecutive delta. Archived, not active.

---

### G. Non-Pure Ordering Experiments (to be ignored per user's requirement)

| Folder | Why Excluded |
|--------|-------------|
| `exp5_label_order/` | Legendre label + pairwise ordering (Label + Ordering hybrid) |
| `exp5b_label_order_clean_delta_MV/` | Legendre label in Q/K + delta in V (Label + Ordering hybrid) |
| `exp3_label_only/` | Legendre label only — no ordering signal at all |
| `exp6_lod_pre/` | Full LOD (Label + Order + Distance) — three-way hybrid |
| `exp6_lod_post/` | Full LOD (Label + Order + Distance) — three-way hybrid |
| `exp2_full_paper/` | Full LOD with α(i,j) distance decay — three-way hybrid |
| `exp1_distance_post_softmax/` | Distance post-softmax — not ordering |
| `exp1_distance_pre_softmax_decay/` | Distance pre-softmax — not ordering |

---

## STEP 3 — Formula Classification (A–H)

| Experiment | Classification | Reason |
|------------|---------------|--------|
| `exp4_ordering_new_sem_space` | **B — Normalized Delta Ordering** | `delta / x_bar` where `x_bar = mean(‖X_i‖₂)` |
| `exp5_ordering_new_pos_space` | **G — Label + Normalized Delta Ordering** | Includes both `P_i` label and `(P_i−P_{i−1})/p̄` |
| `exp4a_order_only_mentor_version` | **A — Delta Ordering** | `Δv_j = V_j − V_{j−1}` in attention V (semantic projected) |
| `exp4b_order_input_position` | **Not A/B** | Pairwise centroid deviation on Legendre: not consecutive delta |
| `exp4b_order_input_mentor_version` | **A — Delta Ordering** | `Δp_j = P_j − P_{j−1}` in attention V (positional) |
| `Archived/exp4_order_only` | **Not A/B** | Pairwise displacement sum in semantic space |
| All exp5/exp6 variants | **E/G/other** | Label or Distance hybrid |

---

## STEP 4 — Implementation Space Classification

| Experiment | Positional Space | Semantic Space | Notes |
|------------|-----------------|----------------|-------|
| `exp4_ordering_new_sem_space` | ❌ | ✅ | Delta computed from `TokenEmbedding(x)` output |
| `exp5_ordering_new_pos_space` | ✅ | ❌ | Delta computed from Legendre vectors `P_i` |
| `exp4a_order_only_mentor_version` | ❌ | ✅ | Delta applied inside attention on projected values |
| `exp4b_order_input_position` | ✅ | ❌ | Pairwise operator on Legendre `P_i` |
| `exp4b_order_input_mentor_version` | ✅ | ❌ | Consecutive delta on Legendre `P_i` in attention V |

---

## STEP 5 — Gap Table: Do the Four Intended Experiments Exist?

| Expected Experiment | Exists | Files | Confidence | Notes |
|---------------------|--------|-------|------------|-------|
| **E1: Delta Ordering, Positional Space** | ⚠️ PARTIAL | `exp4b_order_input_mentor_version/` | Medium | Implements `Δp_i = P_i − P_{i−1}` but injected inside attention V, not at embedding layer as `p_i` replacement |
| **E2: Delta Ordering, Semantic Space** | ⚠️ PARTIAL | `exp4a_order_only_mentor_version/` | Medium | Implements `Δv_j = V_j − V_{j−1}` but applied to **projected** values after `value_projection`, not to `x_i` directly |
| **E3: Normalized Delta Ordering, Positional Space** | ⚠️ PARTIAL | `exp5_ordering_new_pos_space/` | Low | Formula is correct `(P_i−P_{i−1})/p̄` but the experiment **also adds `P_i` (Legendre label)**, making it a hybrid |
| **E4: Normalized Delta Ordering, Semantic Space** | ✅ YES | `exp4_ordering_new_sem_space/` | High | Implements `(X_i−X_{i−1})/x̄` cleanly in embedding layer, pure ordering, no label or distance |

---

## STEP 6 — Formula Deviation Analysis

### E1 — Delta Ordering, Positional Space

**Expected formula:**
```
X'_i = X_i + Δ(p_i)
where Δ(p_i) = P_i − P_{i−1}, Δ(p_1) = P_1 − 0
```

**Actual code** ([`exp4b_order_input_mentor_version/models/attn.py:35-39`](experiments/exp4b_order_input_mentor_version/models/attn.py:35)):
```python
delta_p = torch.zeros_like(p_emb)
delta_p[:, 1:, :, :] = p_emb[:, 1:, :, :] - p_emb[:, :-1, :, :]
V = torch.einsum("bhls,bshd->blhd", A, delta_p)
```

**Difference:**
1. The delta is injected into **attention value matrix**, not added to the embedding.
2. `p_emb` is the Legendre vector **after** `value_projection` (a learned linear layer).
   The delta is `W_V·P_i − W_V·P_{i-1} = W_V·ΔP_i`. This is a projected delta, not a raw delta.
3. The final output is `Σ_j A_ij · W_V · ΔP_j`, not `x_i + ΔP_i`.

**Research impact:** The transformer processes attention-weighted sums of consecutive
positional deltas as its "value" output rather than seeing the delta added to the token
embedding. These are structurally different: the embedding-layer approach injects the signal
before all attention layers; the value-matrix approach injects it only into the attention
output and the signal is **attention-weighted**, not position-additive. They test different
hypotheses. Neither is the same as directly replacing `p_i` with `Δp_i` in the standard
embedding formula `x_i + p_i`.

---

### E2 — Delta Ordering, Semantic Space

**Expected formula:**
```
X'_i = x_i + Δ(x_i)
where Δ(x_i) = x_i − x_{i−1}, Δ(x_1) = x_1 − 0
```

**Actual code** ([`exp4a_order_only_mentor_version/models/attn.py:35-39`](experiments/exp4a_order_only_mentor_version/models/attn.py:35)):
```python
delta_values = torch.zeros_like(values)
delta_values[:, 1:, :, :] = values[:, 1:, :, :] - values[:, :-1, :, :]
V = torch.einsum("bhls,bshd->blhd", A, delta_values)
```

**Difference:**
- `values` is the output of `value_projection(TokenEmbedding(x))` — i.e., `W_V·X_i`.
- The consecutive delta is computed on projected embeddings: `W_V·X_i − W_V·X_{i-1}`.
- The output replaces the standard `Σ_j A_ij·W_V·x_j` with `Σ_j A_ij·W_V·Δx_j`.

**Research impact:** The formula `x_i + Δ(x_i)` would mean the embedding is the token's
own value plus its first difference — in this implementation, the "value" seen by each
query is the **attention-weighted sum of first differences**, which is an information-theoretic
derivative of the original approach. Mathematically, the output converges to
`Σ_j A_ij·Δx_j` rather than `x_i + Δ(x_i)`. This is different from the design specification.

---

### E3 — Normalized Delta Ordering, Positional Space

**Expected formula:**
```
X'_i = X_i + Δ(p_i)/p̄
where Δ(p_i) = P_i − P_{i−1}, p̄ = (1/N)·Σ_i P_i (vector mean) OR scalar
```

**Actual code** ([`exp5_ordering_new_pos_space/models/embed.py:163-183`](experiments/exp5_ordering_new_pos_space/models/embed.py:163)):
```python
leg_d = leg.detach()
delta_p[:, 1:, :] = leg_d[:, 1:, :] - leg_d[:, :-1, :]     # P_i − P_{i−1}
p_bar = leg_d.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)  # scalar
ordering = delta_p / (p_bar + 1e-8)    # O_i^pos

return self.dropout(val + temp + leg + ordering)  # ← ALSO ADDS leg (P_i)
```

**Difference 1 — Extra label term:** The experiment adds `P_i` (the raw Legendre label)
directly: `val + temp + leg + ordering`. This is `X_i + T_i + P_i + O_i^pos`.
A pure ordering experiment would be `X_i + T_i + O_i^pos` only.

**Difference 2 — Normalization:** `p̄ = mean(‖P_i‖₂)` (scalar) vs. any vector mean.

**Research impact of the extra label term:** This is **not a pure ordering experiment**.
The model receives both the raw orthogonal Legendre position label and the normalized
consecutive positional delta. The two signals are not independent — the delta is derived
from the label. Ablation cannot isolate ordering-only because the label is always present.

---

### E4 — Normalized Delta Ordering, Semantic Space

**Expected formula:**
```
X'_i = x_i + Δ(x_i)/x̄
where Δ(x_i) = x_i − x_{i−1}, x̄ = (1/N)·Σ_i x_i
```

**Actual code** ([`exp4_ordering_new_sem_space/models/embed.py:162-171`](experiments/exp4_ordering_new_sem_space/models/embed.py:162)):
```python
delta[:, 1:, :] = val[:, 1:, :] - val[:, :-1, :]
x_bar = val.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)   # scalar
ordering = delta / (x_bar + 1e-8)
return self.dropout(val + temp + ordering)
```

**Difference:** `x̄ = (1/N)·Σ_i ‖X_i‖₂` (scalar, mean of L2 norms) vs. the conceptual
`(1/N)·Σ_i x_i` (vector mean). This is a scale-invariant scalar normalization.

**Research impact:** Dividing by the scalar mean norm gives a dimensionless ordering signal
where the magnitude of the delta is relative to the typical magnitude of the embeddings.
This is a valid and reasonable approximation. The formula is **mathematically interpretable**
and the research claim (normalized ordering in semantic space) is faithfully represented.
No other components (label, distance) contaminate this experiment.

---

## STEP 7 — Duplicate and Alias Detection

### Near-duplicates (same operator, different injection point):

**`exp4b_order_input_position/` vs `exp4b_order_input_mentor_version/`:**

Both use consecutive Legendre delta in positional space, but:
- `exp4b_order_input_position/` uses **pairwise all-to-all** displacement `(1/(N-1))·Σ_{j≠i}(P_i−P_j)` added at embedding layer
- `exp4b_order_input_mentor_version/` uses **consecutive delta** `P_i − P_{i−1}` applied to attention V

These are **different mathematics**, not true duplicates.

**`Archived_experiments/exp4_order_only/` vs `exp4b_order_input_position/`:**

Both use all-pairs displacement formula `(1/(N-1))·Σ_{j≠i}`, but:
- Archived uses **semantic** space (on `value_emb`)
- `exp4b_order_input_position/` uses **positional** (Legendre) space

**`exp4a_order_only_mentor_version/` vs `exp4_ordering_new_sem_space/`:**

Both are Delta Ordering in Semantic Space, but:
- `exp4a` injects delta inside attention V (post-projection, consecutive, not normalized)
- `exp4_ordering_new_sem_space` injects normalized delta at embedding layer (pre-attention, normalized)

These test the same conceptual hypothesis (semantic ordering) but with different:
1. Injection point (embedding vs. attention value)
2. Normalization (none vs. divided by scalar mean norm)

---

## STEP 8 — Dependency Graph

### `exp4_ordering_new_sem_space` (Normalized Delta, Semantic — E4)

```
--pe_mode ordering_sem (exp4_ordering_new_sem_space_ph1.sh)
    ↓
model.py: Informer.__init__() → pe_mode == 'ordering_sem'
    ↓
embed.py: DataEmbedding_ordering_sem.__init__()
    ↓
embed.py: DataEmbedding_ordering_sem.forward(x, x_mark)
  val = TokenEmbedding(x)                   [B,L,D]
  delta[:, 1:] = val[:, 1:] - val[:, :-1]  [B,L,D]
  x_bar = mean(‖val‖₂) per batch           [B,1,1]
  ordering = delta / (x_bar + 1e-8)        [B,L,D]
  out = val + temporal_emb + ordering       [B,L,D]
    ↓
encoder.py/decoder.py: standard Informer Encoder+Decoder
    ↓
attn.py: FullAttention (UNCHANGED — standard scaled dot-product)
    ↓
main_informer.py: training loop, MSE/MAE evaluation
```

### `exp5_ordering_new_pos_space` (Normalized Delta, Positional + Label — NOT pure E3)

```
--pe_mode ordering_pos (exp5_ordering_new_pos_space_ph1.sh)
    ↓
model.py: Informer.__init__() → pe_mode == 'ordering_pos'
    ↓
embed.py: DataEmbedding_ordering_pos.__init__()
  LegendrePositionEmbedding (from legendre_embedding.py, scipy-based, buffer)
    ↓
embed.py: DataEmbedding_ordering_pos.forward(x, x_mark)
  val = TokenEmbedding(x)                           [B,L,D]
  leg = LegendrePositionEmbedding(x)                [B,L,D]
  delta_p[:, 1:] = leg[:, 1:] - leg[:, :-1]        [B,L,D]
  p_bar = mean(‖leg‖₂) per batch                   [B,1,1]
  ordering = delta_p / (p_bar + 1e-8)               [B,L,D]
  out = val + temporal_emb + leg + ordering          [B,L,D]  ← INCLUDES leg (Label)
    ↓
attn.py: FullAttention (UNCHANGED — standard scaled dot-product)
    ↓
main_informer.py: training loop, MSE/MAE evaluation
```

### `exp4a_order_only_mentor_version` (Delta Ordering, Semantic, Attention V — E2 partial)

```
(no --pe_mode flag visible — uses standard DataEmbedding)
    ↓
embed.py: standard DataEmbedding.forward(x, x_mark)
  out = TokenEmbedding(x) + temporal_emb             [B,L,D]  (NO positional PE)
    ↓
encoder.py: standard Informer Encoder
    ↓
attn.py: FullAttention.forward() MODIFIED
  values = value_projection(enc_out)                  [B,S,H,D]
  delta_values[:, 1:] = values[:, 1:] - values[:, :-1]
  V = einsum(A, delta_values)                         [B,L,H,D]
    ↓
main_informer.py: training loop
```

### `exp4b_order_input_position` (Pairwise Displacement on Legendre — NOT consecutive delta)

```
embed.py: DataEmbedding.__init__() imports OrderingOperatorPositional
    ↓
embed.py: DataEmbedding.forward(x, x_mark)
  val = TokenEmbedding(x)
  ordering_pos = OrderingOperatorPositional(seq_len, device)  [1,L,D]
    which computes: (1/(L-1)) · Σ_{j≠i} (P_i − P_j)
  out = val + temporal_emb + ordering_pos
    ↓
attn.py: FullAttention (UNCHANGED)
```

---

## STEP 9 — Final Conclusions

### Q1. Did you actually implement the four experiments you intended?

**NO.** Only one of the four intended experiments is implemented cleanly:
- **E4 (Normalized Delta, Semantic)** — implemented cleanly in `exp4_ordering_new_sem_space/`
- **E1, E2, E3** — exist only in partial or structurally different forms (see below)

### Q2. What experiments did you actually implement?

| What was implemented | Folder | Label |
|---------------------|--------|-------|
| Normalized Delta Ordering in Semantic Space | `exp4_ordering_new_sem_space/` | E4 ✅ |
| Normalized Delta Ordering in Positional Space + Legendre label | `exp5_ordering_new_pos_space/` | E3 + Label (hybrid) |
| Consecutive Delta in Semantic Space (in attention V) | `exp4a_order_only_mentor_version/` | E2 partial |
| Consecutive Delta on Legendre (in attention V) | `exp4b_order_input_mentor_version/` | E1 partial |
| Pairwise displacement on Legendre at embedding layer | `exp4b_order_input_position/` | Neither E1–E4 |
| Pairwise displacement in semantic space (ARCHIVED) | `Archived_experiments/exp4_order_only/` | Neither E1–E4 |

### Q3. Which experiments are pure ordering experiments?

**Strictly pure (no label, no distance):**
- `exp4_ordering_new_sem_space/` — pure normalized delta ordering in semantic space ✅
- `exp4a_order_only_mentor_version/` — pure delta ordering (attention V, semantic) ✅
- `exp4b_order_input_mentor_version/` — pure delta ordering (attention V, Legendre) ✅
- `exp4b_order_input_position/` — pure ordering signal (but pairwise, not consecutive delta) ✅

### Q4. Which experiments accidentally combine ordering with labels or distance?

- `exp5_ordering_new_pos_space/` — adds Legendre label `P_i` **and** normalized positional delta
- `exp5_label_order/` — Label (Legendre) + pairwise ordering
- `exp5b_label_order_clean_delta_MV/` — Label in Q/K + delta in V
- `exp6_lod_pre/`, `exp6_lod_post/`, `exp2_full_paper/` — full LOD (Label+Order+Distance)

### Q5. Which implementation is closest to the mathematical design?

The **research design** calls for:
- `X'_i = X_i + Δ(x_i)` (positional space) or `X'_i = x_i + Δ(x_i)` (semantic space)
- Injection at the embedding layer (replacing `p_i`)
- Consecutive delta formula

**Closest match:** `exp4_ordering_new_sem_space/` — clean normalized delta in semantic space
at embedding layer. The only deviation is normalization by scalar `mean(‖X_i‖₂)` rather
than vector mean, which is a deliberate and defensible design choice.

**Second closest:** A clean E1 (positional space, consecutive delta, embedding layer) does NOT
exist in the repository. `exp5_ordering_new_pos_space/` is the closest but it also contains
the Legendre label. A pure E3/E1 would require removing the `+ leg` term from the output.

### Q6. Which experiments should be kept for the paper?

**Keep (pure ordering, defensible):**
- `exp4_ordering_new_sem_space/` — E4, Normalized Delta, Semantic ✅
- `exp4a_order_only_mentor_version/` — E2 variant, Delta, Semantic (in V) — needs caveat about injection point
- `exp4b_order_input_mentor_version/` — E1 variant, Delta, Positional Legendre (in V) — needs caveat

**Consider fixing before keeping:**
- `exp5_ordering_new_pos_space/` — E3, but needs `+ leg` removed to become pure ordering

### Q7. Which experiments should be ignored as not pure ordering?

Ignore entirely (label + ordering hybrids, or label-only, or distance-involved):
- `exp5_label_order/`
- `exp5b_label_order_clean_delta_MV/`
- `exp3_label_only/`
- `exp6_lod_pre/`, `exp6_lod_post/`
- `exp2_full_paper/`
- `exp1_distance_post_softmax/`, `exp1_distance_pre_softmax_decay/`
- `Archived_experiments/exp4_order_only/` (archived, pairwise, not consecutive)

---

## STEP 10 — Recommended Actions (Implementation Sub-Tasks)

Based on the audit, **three gaps need to be resolved** to achieve the intended 4-experiment design:

### Sub-Task 1 — Create Pure E3: Normalized Delta Ordering, Positional Space
**Intent:** `exp5_ordering_new_pos_space/` implements the right formula but is contaminated by the Legendre label `P_i`. A one-line fix (removing `+ leg` from the output) would produce a pure E3.

**Expected Outcome:** A clean experiment with formula `X'_i = X_i + T_i + O_i^pos` where `O_i^pos = (P_i − P_{i−1}) / p̄`.

**Todo:**
1. Copy `exp5_ordering_new_pos_space/` to a new folder `exp5_ordering_new_pos_space_pure/`
2. In `embed.py`, change [`line 183`](experiments/exp5_ordering_new_pos_space/models/embed.py:183): `return self.dropout(val + temp + leg + ordering)` → `return self.dropout(val + temp + ordering)`
3. Update the shell script `--des` identifier and header comments
4. Verify: the `legendre_embedding.py` file must still be copied (it is used internally to compute `P_i` for the delta, even though `P_i` itself is no longer added to the output)

**Relevant context:**
- [`experiments/exp5_ordering_new_pos_space/models/embed.py:183`](experiments/exp5_ordering_new_pos_space/models/embed.py:183)

**Status:** [ ] pending

---

### Sub-Task 2 — Create Pure E1: Delta Ordering, Positional Space (Embedding Layer)
**Intent:** No experiment currently injects `Δ(p_i) = P_i − P_{i−1}` directly at the **embedding layer** (as an additive term replacing `p_i`). The `exp4b_order_input_mentor_version/` injects it inside attention V (post-projection). A pure E1 would add the consecutive positional delta at the embedding layer.

**Expected Outcome:** An experiment with formula `X'_i = X_i + T_i + ΔP_i` where `ΔP_i = P_i − P_{i−1}`, added at the embedding layer, not inside attention V.

**Todo:**
1. Create `experiments/exp_delta_pos_emb/`
2. In `embed.py`, build a `DataEmbedding_delta_pos` class that:
   - Computes Legendre vectors `P` via `LegendrePositionEmbedding`
   - Computes `delta_p[:, 1:] = P[:, 1:] - P[:, :-1]` with `delta_p[:, 0] = 0`
   - Returns `val + temp + delta_p` (no raw `P_i`, no normalization)
3. Add `pe_mode='delta_pos'` branch in `model.py`
4. Write shell script with `--pe_mode delta_pos`

**Relevant context:**
- [`experiments/exp5_ordering_new_pos_space/models/legendre_embedding.py`](experiments/exp5_ordering_new_pos_space/models/legendre_embedding.py)

**Status:** [ ] pending

---

### Sub-Task 3 — Create Pure E2: Delta Ordering, Semantic Space (Embedding Layer)
**Intent:** `exp4a_order_only_mentor_version/` implements delta in attention V (post-projection). A pure E2 would instead inject `x_i + Δ(x_i)` at the embedding layer, where `Δ(x_i) = X_i − X_{i−1}` (no normalization). This is the un-normalized version of E4.

**Expected Outcome:** An experiment with formula `X'_i = X_i + T_i + ΔX_i` where `ΔX_i = X_i − X_{i−1}`, at embedding layer, no normalization.

**Todo:**
1. Create `experiments/exp_delta_sem_emb/`
2. In `embed.py`, build a `DataEmbedding_delta_sem` class that:
   - Computes `delta[:, 1:] = val[:, 1:] - val[:, :-1]` with `delta[:, 0] = 0`
   - Returns `val + temp + delta` (no normalization, no Legendre)
3. Add `pe_mode='delta_sem'` branch in `model.py`
4. Write shell script with `--pe_mode delta_sem`

**Relevant context:**
- [`experiments/exp4_ordering_new_sem_space/models/embed.py`](experiments/exp4_ordering_new_sem_space/models/embed.py) — this is the normalized version; remove normalization step

**Status:** [ ] pending

---

## Open Questions for User Confirmation Before Implementation

1. **E3 fix strategy:** Should `exp5_ordering_new_pos_space/` be modified in-place, or
   should a new clean copy be created (preserving the hybrid version for reference)?

2. **Normalization in E1/E2:** The design says "Delta Ordering" (un-normalized) and
   "Normalized Delta Ordering" (normalized). Should E1 and E2 be strictly un-normalized
   (no `x̄` divisor), or is the normalization also acceptable for the un-normalized variants?

3. **Injection point for E1/E2:** The design says ordering "replaces `p_i`" (embedding layer
   addition). The existing `exp4a` and `exp4b_input_mentor` inject inside attention V.
   Should new embedding-layer versions be created, or are the attention-V variants acceptable
   as-is with a notation caveat in the paper?

4. **`exp4b_order_input_position/`:** This uses all-pairs Legendre displacement rather than
   consecutive delta. Should it be kept as an additional ablation or discarded?
