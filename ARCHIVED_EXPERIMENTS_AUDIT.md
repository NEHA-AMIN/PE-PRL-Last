# ARCHIVED EXPERIMENTS AUDIT

**Location:** `Archived_experiments/`  
**Audit Date:** 2025  
**Purpose:** Document the archived experiment variants — their implementations, why they were superseded, and what design choices they represent.

---

## Overview

Five archived sub-folders exist:

| Folder | Description |
|--------|-------------|
| `exp4_order_only/` | Original "Ordering Only" experiment — pairwise all-to-all semantic displacement |
| `exp4_ordering_archive/` | Archive of `DataEmbedding_ordering_sem` with normalised consecutive delta |
| `exp4a_order_only_mentor_version/` | Mentor-revised version: delta moved to attention values, NOT embedding |
| `exp4b_order_input_mentor_version/` | Mentor-revised Exp4b: Legendre delta used in attention value matrix |
| `exp4b_order_input_position/` | Exp4b with pairwise ordering in Legendre positional space |

All are marked archived — they exist as historical reference and appendix material only.

---

## exp4_order_only — Pairwise All-to-All Ordering in Semantic Space

**Status:** Archived — replaced by Formula-A-sem

### Implementation (verified from [`embed.py`](Archived_experiments/exp4_order_only/models/embed.py))

Uses `OrderingOperator` from `ordering_operator.py`:
```
O_i = (1/N-1) · Σ_{j≠i} (X_i − X_j)
```

This is the **all-to-all pairwise displacement** operator. It computes a separate `[B, L, L, D]` tensor before masking the diagonal and summing. **O(L²·D) memory complexity** — much more expensive than the consecutive-delta approach.

**Critical algebraic observation verified from code:**
```python
delta_x = X_i - X_j  # [B, L, L, D]
# sum over j ≠ i:
O_i = delta_x.sum(dim=2) / (L - 1)
    = (Σ_j X_i - Σ_j X_j) / (L-1)
    = X_i - (1/(L-1)) Σ_j X_j
```
This is equivalent to `X_i - μ_adjusted` (where μ_adjusted uses N-1 denominator). Approximately mean-centring. **Reduces to a normalised version of Formula-B-sem** for large L.

**Formula:** `X'_i = X_i + T_i + O_i` (sinusoidal PE removed, ordering replaces it)

**Reason archived:** O(L²) cost; algebraically equivalent to Formula-B-sem at large L; consecutive delta was simpler, cheaper, and more interpretable.

---

## exp4_ordering_archive — Normalised Consecutive Delta in Semantic Space

**Status:** Archived — an intermediate development step

### Implementation (verified from [`embed.py`](Archived_experiments/exp4_ordering_archive/models/embed.py))

Uses `DataEmbedding_ordering_sem`:
```
delta_i = 0              if i = 0
           X_i - X_{i-1} if i >= 1
x_bar = (1/N) Σ_i ||X_i||_2   [B, 1, 1]
O_i^sem = delta_i / (x_bar + 1e-8)
```

**Formula:** `X'_i = X_i + T_i + O_i^sem`

This is identical to Formula-A-sem EXCEPT:
- Formula-A-sem boundary: `delta[:, 0, :] = val[:, 0, :]` (Δ = X_0)
- This archive: `delta[:, 0, :] = 0` (zero boundary)

**Critical difference:** The boundary condition at i=0 differs. `exp4_ordering_archive` uses zero-pad (Δ_0 = 0), while `formula-A-sem` uses X_0 as the boundary. Both are documented as valid choices.

This archive also normalises by x_bar (mean L2 norm), while the final `formula-A-sem` does NOT normalise. So this archive is closer to Formula-B-sem conceptually (normalised, zero boundary) applied with a consecutive delta.

---

## exp4a_order_only_mentor_version — Delta in Attention Values

**Status:** Archived — mentor-directed design where delta moves from embedding to attention

### Implementation (verified from [`embed.py`](Archived_experiments/exp4a_order_only_mentor_version/models/embed.py))

The embedding forward:
```python
x = self.value_embedding(x) + self.temporal_embedding(x_mark)
# NO ordering term added here
```

The ordering signal (consecutive delta) is expected to be injected in `attn.py` by replacing the attention value matrix. The embedding returns a plain tensor — ordering is handled by the attention mechanism.

**This is the architectural predecessor of Exp5b and Exp6.** The design principle — separate content (embedding) from ordering (values) — evolved into the clean delta-V split in Exp5b.

---

## exp4b_order_input_mentor_version — Legendre Delta in Attention Value Matrix

**Status:** Archived — mentor-directed variant

### Implementation (verified from [`embed.py`](Archived_experiments/exp4b_order_input_mentor_version/models/embed.py))

```python
x_out = value_emb + temporal_emb  # [B, L, d_model]
return self.dropout(x_out), P     # returns TUPLE: (clean_embedding, Legendre_vectors)
```

The embedding returns a **tuple** `(x_out, P)` where P are the Legendre vectors. The attention mechanism is expected to compute `Δp_i = P_i − P_{i−1}` and use these as the value matrix.

**This is the direct predecessor of Exp5b (clean delta-V).** Exp5b uses the same tuple-return pattern but computes `delta_x` (semantic consecutive delta) rather than Legendre delta as the value override.

**Key note:** `ordering_operator_positional.py` uses `LegendreEmbedding` (not `LegendrePositionEmbedding`) — a different class. This suggests an earlier, different Legendre implementation before the shared `LegendrePositionEmbedding` buffer was standardised.

---

## exp4b_order_input_position — Pairwise Ordering from Legendre Positional Embeddings

**Status:** Archived — tested positional-space ordering with pairwise all-to-all formula

### Implementation (verified from [`embed.py`](Archived_experiments/exp4b_order_input_position/models/embed.py))

```python
ordering_pos = self.ordering_operator_pos(seq_len=..., device=...)  # [1, L, d_model]
x_out = value_emb + temporal_emb + ordering_pos
return self.dropout(x_out)
```

The `OrderingOperatorPositional` computes:
```
O_i = (1/(N-1)) Σ_{j≠i} (P_i − P_j)
```
where P_i are Legendre positional vectors. Returns shape `[1, L, d_model]` (broadcasts over batch).

**Algebraically equivalent to the pairwise version of Formula-A-pos** with zero normalisation. Archived in favour of the cleaner consecutive-delta normalised version in Formula-A-pos.

---

## Archived Experiment Lineage

```
exp4_order_only (pairwise, semantic, O(L²))
    ↓ simplified to consecutive delta
exp4_ordering_archive (consecutive, semantic, normalised, zero boundary)
    ↓ mentor revised — move ordering from embedding to attention values
exp4a_order_only_mentor_version (delta-in-attn-V architecture established)
    ↓ evolved into Exp5b (full clean-delta-V with Legendre label in Q/K)
exp4b_order_input_position (pairwise positional) → Formula-A-pos (consecutive positional)
exp4b_order_input_mentor_version (Legendre delta in value matrix) → ancestor of Exp6
```

---

## Consistency Findings

### exp4_ordering_archive ↔ formula-A-sem
- Both implement consecutive delta in semantic space
- **Difference 1:** exp4_ordering_archive normalises by x_bar; formula-A-sem does not
- **Difference 2:** boundary condition differs (0 vs X_0)
- These are distinct experiments, not duplicates

### exp4b_order_input_mentor_version embed tuple
- Returns `(x_out, P)` — tuple pattern
- `attn.py` must unpack this tuple; if the archived attn.py doesn't do so, it will fail
- This is the same architectural vulnerability found in Exp5b/Exp6 (archived encoders may have mismatched attn.py)

### exp4_order_only OrderingOperator
- Verified to implement O(L²) all-to-all pairwise displacement, not consecutive delta
- Algebraically reduces to approximate mean-centring for large L

---

## Confidence Summary for Archived Experiments

| Folder | Implementation Confidence | Notes |
|--------|--------------------------|-------|
| exp4_order_only | 8/10 | Clear pairwise formula; O(L²) cost; algebraically = approx. mean-centring |
| exp4_ordering_archive | 8/10 | Normalised consecutive delta; precursor to Formula-A-sem |
| exp4a_order_only_mentor_version | 7/10 | Embedding correct; attn.py changes not audited |
| exp4b_order_input_mentor_version | 7/10 | Tuple-return embed; Legendre delta in V-matrix; attn.py not audited |
| exp4b_order_input_position | 7/10 | Pairwise Legendre ordering; attn.py not audited |
