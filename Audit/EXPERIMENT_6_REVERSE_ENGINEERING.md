# EXPERIMENT_6_REVERSE_ENGINEERING.md
## Forensic Reconstruction of Original Author Intent
### Evidence source: implementation only. No notebooks. No scripts. README used only as lowest-priority corroboration.

---

## Preamble: Evidence Status

| Artifact | Status | Trust Level |
|----------|--------|-------------|
| `exp6_lod_post/models/*.py` (6 files) | ✅ Present | PRIMARY — implementation is ground truth |
| `exp6_lod_pre/models/*.py` (6 files) | ✅ Present | PRIMARY — implementation is ground truth |
| `exp6_lod_post/README-E6-POST.md` | ✅ Present | CORROBORATING — used only where it does not contradict code |
| `exp6_lod_pre/README-E6-Pre.md` | ✅ Present | CORROBORATING |
| `exp6_lod_post/Exp6_lod_post-Gcolab.ipynb` | 🗑 DELETED | UNAVAILABLE |
| `exp6_lod_post/exp6_lod_post_phase1.sh` | 🗑 DELETED | UNAVAILABLE |
| `exp6_lod_post/exp6_lod_post_phase2.sh` | 🗑 DELETED | UNAVAILABLE |
| `exp6_lod_pre/exp6_lod_pre_phase1.sh` | ❌ Never existed in directory | UNAVAILABLE |
| `exp6_lod_pre/exp6_lod_pre_phase2.sh` | ❌ Never existed in directory | UNAVAILABLE |
| `mse_mae_scores_sorted.txt` Exp6 entries | ⚠️ Identical to each other — unreliable | NOT USED |

**Key rule applied throughout:** Where the README describes code that differs from the actual file, the file wins. One such divergence is documented in Phase 3.

---

## Phase 1: Complete Dependency Map

### All Repository References to Experiment 6

#### 1. Experiment Folders
```
experiments/exp6_lod_post/
    models/__init__.py          — empty, registration stub
    models/model.py             — Informer + InformerStack, LOD variant
    models/embed.py             — DataEmbedding returning (combined_emb, delta_x)
    models/attn.py              — FullAttention with POST-softmax decay
    models/encoder.py           — EncoderLayer + Encoder + EncoderStack, delta_x aware
    models/decoder.py           — DecoderLayer + Decoder, delta_x aware
    models/legendre_embedding.py — LegendrePositionEmbedding (pure PyTorch recurrence)
    README-E6-POST.md           — surviving documentation

experiments/exp6_lod_pre/
    models/__init__.py          — empty, registration stub
    models/model.py             — Informer + InformerStack, LOD variant
    models/embed.py             — DataEmbedding returning (combined_emb, delta_x)
    models/attn.py              — FullAttention with PRE-softmax decay
    models/encoder.py           — EncoderLayer + Encoder + EncoderStack, delta_x aware
    models/decoder.py           — DecoderLayer + Decoder, delta_x aware
    models/legendre_embedding.py — LegendrePositionEmbedding (pure PyTorch recurrence)
    README-E6-Pre.md            — surviving documentation
```

#### 2. Result References (in mse_mae_scores_sorted.txt)
```
Lines 493–538:  EXPERIMENT: Exp6-Pre  — 15 result values (Phase 1 + Phase 2)
Lines 541–584:  EXPERIMENT: Exp6-Post — 15 result values (identical to Exp6-Pre)
```

#### 3. Cross-references in other audit documents
```
EXPERIMENT_AUDIT.md            — rows 33–34, 54–55, 72–73, 87, 94, 101
EXPERIMENT_6_AUDIT.md          — full prior audit of exp6_lod_pre
EXPERIMENT_7_AUDIT.md          — full prior audit of exp6_lod_post
INCONSISTENCIES.md             — IC-001, IC-009, IC-011, IC-022
DEPENDENCY_MAP.md              — Exp6-Pre and Exp6-Post sections
```

#### 4. Classes and Functions Referenced
```
LegendrePositionEmbedding      — legendre_embedding.py (both variants)
DataEmbedding                  — embed.py (both variants)
FullAttention                  — attn.py (both variants)
ProbAttention                  — attn.py (both variants, unmodified from baseline)
AttentionLayer                 — attn.py (both variants)
EncoderLayer                   — encoder.py (both variants)
Encoder                        — encoder.py (both variants)
EncoderStack                   — encoder.py (both variants)
DecoderLayer                   — decoder.py (both variants)
Decoder                        — decoder.py (both variants)
Informer                       — model.py (both variants)
InformerStack                  — model.py (both variants)
```

#### 5. Key Variables and Parameters
```
decay_a          — distance decay exponent, flows from model.py → attn.py
delta_x          — consecutive embedding difference, flows embed → encoder/decoder → attn
combined_emb     — x_i + T_i + p_i, flows embed → encoder/decoder → Q/K projections
delta_enc        — delta_x from encoder embedding
delta_dec        — delta_x from decoder embedding
delta_values     — keyword argument in AttentionLayer.forward() for V override
delta_slice      — sliced delta_x in EncoderStack for multi-resolution stack
legendre_pos     — Legendre polynomial evaluation at each position
alpha            — computed distance decay matrix [L, S]
dist_matrix      — |i-j| absolute index distance matrix
```

---

## Phase 2: Complete Call Hierarchy

### Variant A: exp6_lod_pre (Pre-Softmax Distance Decay)

```
main_informer.py
  └── Exp_Informer.train()
        └── model = Informer(
                enc_in=7, dec_in=7, c_out=7,
                seq_len=96, label_len=48, out_len=pred_len,
                d_model=512, n_heads=8, e_layers=2, d_layers=1,
                decay_a=1.0,   ← threaded throughout
                attn='full',
                distil=True    ← default, causes delta_x bug
            )
            │
            ├── DataEmbedding(enc_in=7, d_model=512, ...)  → self.enc_embedding
            │     ├── TokenEmbedding(7, 512)               → self.value_embedding
            │     ├── TemporalEmbedding(512, 'fixed', 'h') → self.temporal_embedding
            │     └── LegendrePositionEmbedding(512)        → self.legendre_embedding
            │
            ├── DataEmbedding(dec_in=7, d_model=512, ...)  → self.dec_embedding
            │     (identical to enc_embedding)
            │
            ├── Encoder(
            │     attn_layers=[
            │       EncoderLayer(
            │         AttentionLayer(
            │           FullAttention(mask_flag=False, decay_a=1.0),  ← decay_a present
            │           d_model=512, n_heads=8
            │         ),
            │         d_model=512, d_ff=512
            │       ),
            │       EncoderLayer(...)  ← same, second layer
            │     ],
            │     conv_layers=[ConvLayer(512)]  ← distillation (halves seq_len)
            │   )
            │
            └── Decoder(
                  layers=[
                    DecoderLayer(
                      self_attention=AttentionLayer(
                        FullAttention(mask_flag=True, decay_a=1.0),   ← decay_a present
                        d_model=512, n_heads=8
                      ),
                      cross_attention=AttentionLayer(
                        FullAttention(mask_flag=False, decay_a=1.0),  ← decay_a present
                        d_model=512, n_heads=8
                      )
                    )
                  ]
                )

Informer.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
  │
  ├── enc_out, delta_enc = self.enc_embedding(x_enc, x_mark_enc)
  │     │
  │     └── DataEmbedding.forward(x, x_mark):
  │           value_emb  = TokenEmbedding(x)                          [B,96,512]
  │           delta_x    = value_emb - roll(value_emb, 1, dim=1)      [B,96,512]
  │           delta_x[:,0,:] = 0.0
  │           temporal_emb = TemporalEmbedding(x_mark)                [B,96,512]
  │           legendre_pos = LegendrePositionEmbedding(x)             [B,96,512]
  │           combined_emb = value_emb + temporal_emb + legendre_pos  [B,96,512]
  │           return dropout(combined_emb), dropout(delta_x)   ← BOTH dropout'd
  │
  ├── enc_out, attns = self.encoder(enc_out, delta_x=delta_enc)
  │     │
  │     └── Encoder.forward(x=combined_emb, delta_x=delta_enc):
  │           [conv_layers present because distil=True]
  │           ├── x, attn = EncoderLayer(x, delta_x=delta_x)
  │           │     └── AttentionLayer.forward(x, x, x, delta_values=delta_x):
  │           │           Q = query_projection(x)     [B,96,H,E]  ← from combined_emb
  │           │           K = key_projection(x)       [B,96,H,E]  ← from combined_emb
  │           │           V = value_projection(delta_x) [B,96,H,D] ← from delta_x
  │           │           └── FullAttention.forward(Q, K, V, mask):
  │           │                 scores = einsum(Q,K)              [B,H,96,96]
  │           │                 dist_matrix = |i-j|               [96,96]
  │           │                 alpha = 1/(1+dist_matrix^decay_a)  [96,96]
  │           │                 scores = scores * alpha  ← PRE-SOFTMAX DECAY
  │           │                 scores.masked_fill_(-inf)
  │           │                 A = dropout(softmax(scale * scores))
  │           │                 V_out = einsum(A, V)
  │           │
  │           ├── x = ConvLayer(x)    → x now [B,~48,512]
  │           │   [delta_x NOT downsampled — stays [B,96,512]]  ← SHAPE BUG
  │           │
  │           └── x, attn = EncoderLayer[-1](x, delta_x=delta_x)
  │                 [x=[B,48,512], delta_x=[B,96,512] — MISMATCH]
  │
  ├── dec_out, delta_dec = self.dec_embedding(x_dec, x_mark_dec)
  │     (same forward() as enc_embedding)
  │
  └── dec_out = self.decoder(dec_out, enc_out, delta_x=delta_dec)
        └── DecoderLayer.forward(x, cross, delta_x=delta_dec):
              self_attention(x, x, x, delta_values=delta_dec)  ← delta in self-attn
              cross_attention(x, cross, cross, delta_values=None)  ← NO delta in cross
```

---

### Variant B: exp6_lod_post (Post-Softmax Distance Decay)

```
[Identical call hierarchy to Variant A, EXCEPT inside FullAttention.forward():]

FullAttention.forward(Q, K, V, mask):      ← V is already projected delta_x
  scores = einsum(Q,K)                     [B,H,L,S]
  scores.masked_fill_(-inf)                ← mask applied BEFORE softmax
  A = dropout(softmax(scale * scores))     ← SOFTMAX FIRST
  dist_matrix = |i-j|                      [L,S]
  alpha = 1/(1+dist_matrix^decay_a)        [L,S]
  A = A * alpha                            ← POST-SOFTMAX DECAY
  V_out = einsum(A, V)

[Everything else in the call hierarchy is byte-for-byte identical to Variant A]
```

**One additional difference in embed.py.forward():**
```
Variant A (exp6_lod_pre):
    return self.dropout(combined_emb), self.dropout(delta_x)   ← dropout on BOTH

Variant B (exp6_lod_post):
    return self.dropout(combined_emb), delta_x                  ← dropout on combined ONLY
```

---

## Phase 3: File-by-File Comparison

| File | Variant A (exp6_lod_pre) | Variant B (exp6_lod_post) | Difference |
|------|--------------------------|---------------------------|-----------|
| `model.py` (Informer) | `decay_a` in encoder, decoder self-attn, decoder cross-attn | Identical | ✅ No difference |
| `model.py` (InformerStack) | Broken: does NOT unpack tuple from embedding | Broken: does NOT unpack tuple from embedding | ✅ Both broken identically |
| `embed.py` (DataEmbedding.__init__) | `TokenEmbedding + TemporalEmbedding + LegendrePositionEmbedding` | Identical | ✅ No difference |
| `embed.py` (DataEmbedding.forward) | `return self.dropout(combined_emb), self.dropout(delta_x)` | `return self.dropout(combined_emb), delta_x` | **⚡ DIFFERENCE 1: dropout(delta_x) vs bare delta_x** |
| `attn.py` (FullAttention.forward) | `scores *= alpha` → then `softmax` | `softmax` → then `A *= alpha` | **⚡ DIFFERENCE 2 (primary): pre-softmax vs post-softmax decay** |
| `attn.py` (all other methods) | Identical to Variant B | Identical to Variant A | ✅ No difference |
| `encoder.py` (EncoderLayer.forward) | Identical | Identical | ✅ No difference |
| `encoder.py` (Encoder.forward) | Does NOT downsample delta_x | Does NOT downsample delta_x | ✅ Both have same potential shape bug |
| `encoder.py` (EncoderStack.forward) | Slices `delta_x[:, -inp_len:, :]` | Identical | ✅ No difference |
| `decoder.py` (DecoderLayer.forward) | `delta_values=delta_x` in self-attn, `delta_values=None` in cross-attn | Identical | ✅ No difference |
| `legendre_embedding.py` | Uses `math.sqrt(self.d_model)` | Uses `(self.d_model ** 0.5)` | ⚠️ Syntactically different, mathematically identical |

### README Describes Different embed.py Code

**What README-E6-POST.md lines 237–241 documents:**
```python
# README version:
delta_x = torch.zeros_like(x)           # raw input x
delta_x[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
delta_x = self.value_embedding(delta_x)  # then embed the delta
```

**What the actual embed.py code does (line 112):**
```python
# Actual code:
delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)  # delta of embedded x
delta_x[:, 0, :] = 0.0
```

**These are different operations.** The README describes computing the difference of raw inputs and then embedding, which would give `Conv1D(x_i - x_{i-1})`. The code computes the difference of embeddings, which gives `Conv1D(x_i) - Conv1D(x_{i-1})`. Because `Conv1D` is linear, these are algebraically equivalent for the body of the sequence — BUT with circular padding, the boundary behavior differs. More importantly, the README's version never went through dropout (it was applied at the end), while the actual code's delta is applied to already-computed `value_emb`. **The README is describing an earlier or alternate design. Trust the code.**

---

## Phase 4: First Point of Divergence

```
main_informer.py
   ↓ [shared: identical]
Informer.__init__()
   ↓ [shared: identical]
DataEmbedding.__init__()
   ↓ [shared: identical]
DataEmbedding.forward()
   ├── value_emb = TokenEmbedding(x)               [shared]
   ├── delta_x = value_emb - roll(value_emb, 1)    [shared]
   ├── delta_x[:,0,:] = 0                          [shared]
   ├── temporal_emb = TemporalEmbedding(x_mark)    [shared]
   ├── legendre_pos = LegendrePositionEmbedding(x) [shared]
   ├── combined_emb = val + temp + leg             [shared]
   │
   └── DIVERGENCE POINT 1 (embed.py, return statement):
       Pre:  return dropout(combined_emb), dropout(delta_x)
       Post: return dropout(combined_emb), delta_x
       ▲
       File: embed.py
       Function: DataEmbedding.forward()
       Line: 128 (Pre) vs 127 (Post)
       Operation: whether dropout is applied to the delta_x stream

Then:
AttentionLayer.forward()
   ├── Q = query_projection(combined_emb)           [shared]
   ├── K = key_projection(combined_emb)             [shared]
   ├── V = value_projection(delta_x)               [shared — but delta_x may be dropped differently]
   │
   └── inner_attention → FullAttention.forward()
         scores = einsum(Q, K)                       [shared]
         │
         └── DIVERGENCE POINT 2 (attn.py — THE PRIMARY DIVERGENCE):
             Pre:
               alpha = 1/(1+|i-j|^a)
               scores = scores * alpha          ← applies decay to RAW LOGITS
               scores.masked_fill_(-inf)
               A = softmax(scale * scores)      ← softmax sees biased scores

             Post:
               scores.masked_fill_(-inf)
               A = softmax(scale * scores)      ← softmax sees CLEAN scores
               alpha = 1/(1+|i-j|^a)
               A = A * alpha                    ← applies decay to PROBABILITIES
             ▲
             File: attn.py
             Class: FullAttention
             Method: forward()
             Line: 37 (Pre: scores *= alpha) vs 46 (Post: A *= alpha)
             Operation: multiplication of distance decay matrix with logits vs probabilities
```

**Summary:**
- **Divergence Point 1 (minor):** `embed.py` line 128/127 — dropout application to `delta_x`
- **Divergence Point 2 (primary, defining):** `attn.py` lines 37 vs 46 — pre-softmax vs post-softmax multiplication of distance decay `α(i,j)` with the attention tensor

Everything else — embedding construction, encoder architecture, decoder architecture, Legendre formula, delta formula, all variable names, all layer configurations — is **byte-for-byte identical**.

---

## Phase 5: Mathematical Intent of Every Unique Operation

---

### Operation 1: Legendre Polynomial Embedding
**File:** `legendre_embedding.py:15-27`

#### Formula
```
positions_i = 2·i/(L−1) − 1,    i ∈ {0, 1, ..., L−1}

P_{i,0} = 1
P_{i,1} = positions_i
P_{i,n} = ((2n−1)·positions_i·P_{i,n−1} − (n−1)·P_{i,n−2}) / n

p_i = P_i / √d_model     [scaling]
```

#### Interpretation
Positional distinctiveness encoding. Each integer sequence position `i` is mapped to a unique 512-dimensional fingerprint derived from orthogonal polynomial evaluations. Position 0 always maps to the vector `[1, -1, 1, -1, ...]` (odd/even Legendre values at -1). Position 95 maps to `[1, +1, +1, +1, ...]` (Legendre values at +1). No two positions share the same vector.

#### Motivation
The author wanted every token's position to be **uniquely identifiable** without information shared between positions, using polynomials that are mathematically guaranteed orthogonal (`⟨L_n, L_m⟩ = 0` for n≠m). This replaces sinusoidal PE while preserving the non-trainable, deterministic, position-indexed nature.

#### Expected Effect
- Provides stable, fixed "address labels" for each position
- Q and K projections encode both semantic content (from `value_emb`) and positional address (from `legendre_pos`) in a single vector
- The attention similarity score `Q_i · K_j` now incorporates both "what is this token" and "where is this token" information

---

### Operation 2: Clean Consecutive Delta
**File:** `embed.py:112-113`

#### Formula
```
Δx_i = x_i − x_{i−1},    i ∈ {1, ..., L−1}
Δx_0 = 0                 [zero-boundary condition]
```
where `x_i` = `TokenEmbedding(raw_input)[i]` — the 512-dim embedding of the i-th raw timestep.

**Implementation:** `value_emb - torch.roll(value_emb, shifts=1, dims=1)` then zeroing index 0.

#### Interpretation
A first-order finite difference of the token embedding sequence. It captures the **direction and magnitude of change** in the feature space from one timestep to the next. This is the temporal derivative of the embedding sequence.

#### Motivation
The author separated the signal used for "deciding which tokens to attend to" (Q·K, using combined_emb = semantic + temporal + positional) from the signal used for "what to aggregate" (V, using Δx = direction of change). The underlying belief: attention weights should be governed by **where and what** tokens are (combined_emb), but the value being aggregated should carry **how the sequence is moving** (Δx).

This is a deliberate architectural decoupling, treating the attention mechanism as a selective filter that chooses which timesteps' *changes* to aggregate, rather than which timesteps' *states* to aggregate.

#### Expected Effect
- Output at each query position becomes: `Σ_j A_{ij} · W_V · Δx_j` — a weighted sum of embedding differences
- The model learns to attend to tokens whose changes are most informative for forecasting the target
- Position 0 contributes nothing to any output (zero vector in V)

---

### Operation 3: Q/K from combined_emb, V from delta_x
**File:** `attn.py:162-174`

#### Formula
```
Q_i = W_Q · (x_i + T_i + p_i)
K_j = W_K · (x_j + T_j + p_j)
V_j = W_V · (x_j − x_{j−1})         [when delta_values is not None]
```

#### Interpretation
A **role separation** within the attention mechanism. Q and K carry full context (what is here, when is it, where is it). V carries only temporal change information (what changed).

The similarity function `Q_i · K_j / √d` measures "how relevant is position j (in all its context) to position i." The aggregated value is "the weighted sum of how much the signal changed at those relevant positions."

#### Motivation
The author appears to have been testing whether decoupling the *selection criterion* (Q·K) from the *aggregated content* (V) improves forecasting. In standard attention, the same representation is used for both — tokens that are "similar" to the query are also the ones whose full embedding is aggregated. Here, similarity is measured in full-context space while aggregation happens in temporal-derivative space.

---

### Operation 4 (PRIMARY DIVERGENCE): Pre-Softmax Distance Decay
**File:** `exp6_lod_pre/models/attn.py:28-37`

#### Formula
```
α(i,j) = 1 / (1 + |i−j|^a)

s'_{ij} = (Q_i · K_j / √d) · α(i,j)     [biased logits]

A_{ij} = softmax({s'_{ij}}_j)             [probability over biased logits]
       = exp(s'_{ij}) / Σ_k exp(s'_{ik})
       = exp((Q_i·K_j/√d)·α(i,j)) / Σ_k exp((Q_i·K_k/√d)·α(i,k))
```

#### Interpretation
The distance decay is applied **inside the exponential** (via multiplicative bias on the pre-softmax logit). Because softmax preserves relative ordering, the effect is: distant tokens have their scores shrunk by `α < 1` before normalization, so they contribute proportionally less to the probability mass. The result is a valid probability distribution, but one that has been tilted toward local attention by discounting distant logits during competition.

**Key property:** The softmax normalization absorbs the decay into the probability computation. Nearby tokens win more probability mass because their logits were shrunk less.

#### Motivation
The author wanted the distance decay to act as a **hard inductive bias during probability assignment**. By biasing the competition before normalization, nearby tokens get a structural advantage in the softmax competition. Distant tokens don't just get smaller weights — they are less competitive during the normalization process itself.

The mental model behind this: "Position should influence who wins the attention competition, not just how much the winner contributes."

#### Expected Effect
- Produces a valid probability distribution (sums to 1)
- Creates sharper, more local attention patterns
- Nearby tokens win a greater share of the total probability mass
- Very distant tokens are aggressively suppressed (their logits are small before normalization)
- The learning objective (backprop) sees gradients through the biased logit — distance decay actively shapes gradient flow

---

### Operation 5 (PRIMARY DIVERGENCE): Post-Softmax Distance Decay
**File:** `exp6_lod_post/models/attn.py:34-46`

#### Formula
```
A_{ij} = softmax({Q_i·K_j/√d}_j)         [unbiased probability distribution]
       = exp(Q_i·K_j/√d) / Σ_k exp(Q_i·K_k/√d)

A'_{ij} = A_{ij} · α(i,j)                 [rescaled — no longer sums to 1]

V_out_i = Σ_j A'_{ij} · V_j
        = Σ_j A_{ij} · α(i,j) · V_j       [weighted by both attention and distance]
```

#### Interpretation
The softmax produces a clean probability distribution over the attention scores without any positional influence. Then the distance decay is applied as a **multiplicative mask** on top of the already-normalized weights. The result is NOT a probability distribution — the weights no longer sum to 1. Instead, the output is a proximity-gated version of the standard attention output.

**Key property:** The softmax computation itself is unaffected by distance. Distance only influences which portion of the (already learned) attention weights actually contributes to the output. This is mathematically equivalent to elementwise-multiplying the output by a position-dependent scale.

#### Motivation
The author wanted to test whether distance decay could be applied **after** the model has learned its semantic attention patterns, so that distance acts as a *refinement* rather than a *constraint*. The intuition: "Let the model learn which tokens matter semantically, then suppress the distant ones in the output."

This is also safer numerically and architecturally — the learned attention distribution is preserved intact, and distance decay merely gates it. The model can learn to compensate for the decay in the V projection.

#### Expected Effect
- Produces a non-normalized output (total weight < 1, decreasing with distance)
- The model must learn that distant tokens contribute proportionally less to the output regardless of attention score
- Gradients through backprop do not flow through distance-influenced softmax (clean separation)
- Semantically, this means: "even if the model strongly attends to a distant token, that attention is discounted by the proximity factor"

---

### Operation 6: Dropout on delta_x (Variant A only)
**File:** `exp6_lod_pre/models/embed.py:128`

#### Formula
```
Pre:  return dropout(combined_emb), dropout(delta_x)
Post: return dropout(combined_emb), delta_x
```

#### Interpretation
In Variant A (Pre), `dropout` is applied to the delta stream before it is used as V. In Variant B (Post), the delta stream is passed raw. This means:
- In Pre: some components of the V matrix are randomly zeroed during training
- In Post: the V matrix receives the full delta signal every training step

#### Motivation (reconstructed)
This appears to be an **accidental asymmetry** rather than a deliberate design choice. Evidence: (a) the README for Exp6-Post (the more detailed surviving document) does not mention this difference anywhere — not in the "Key Differences" table, not in the implementation details; (b) the comment on the Pre version's return line says "Apply dropout to both" which reads as a design intention, but the Post version simply lacks the second `self.dropout()` call without explanation.

**Confidence: LOW** — could be deliberate regularization in Pre or could be a copy-paste oversight in Post.

---

## Phase 6: Reconstructed Original Hypothesis

### Variant A: exp6_lod_pre

**"I probably implemented this because I believed that..."**

**Hypothesis 1: Distance should constrain the attention competition itself, not just its output.**
The author believed that proximity is an inductive bias that should shape *which* tokens win attention, not just *how much* their winning contributes. By applying decay before softmax, distant tokens become fundamentally less competitive during normalization. The thought was: "time series forecasting has locality — nearby timesteps should be intrinsically more relevant, so I should encode this before the probability distribution is formed."

*Evidence:*
- Code comment at `attn.py:27`: `"# Step 2: Apply distance decay BEFORE softmax (LOD PRE)"`
- README-E6-Pre.md: "Distance Decay: `α(i,j) = 1 / (1 + |i-j|^a)` — Applied to attention scores **before** the softmax operation, acting as a sharper suppression mechanism that competes during normalization."
- The placement of `scores = scores * alpha` before `scores.masked_fill_` confirms the author understood this order changes the character of suppression.

**Hypothesis 2: Pre-softmax placement creates a stronger locality inductive bias that may help with longer prediction horizons.**
The author expected pre-softmax to be a "harder" locality constraint — one where proximity truly gates semantic relevance. For longer forecasting horizons (192, 336), where local patterns may be more predictive, this harder constraint might outperform the weaker post-softmax version.

*Evidence:*
- README-E6-Pre.md "When to Use Pre-Softmax: Strong positional bias is needed; Positional information should dominate"
- The alpha sweep tested (0.5, 1.0, 2.0) — three different "hardness" levels — directly tests this hypothesis

**Hypothesis 3: The LOD combination with pre-softmax reverses the degradation seen in earlier experiments.**
Earlier experiments showed that adding the full LOD at the embedding level (Exp2) actually hurt performance vs L+O alone (Exp5). The author hypothesized that the degradation was specifically due to *how* Distance was injected in Exp2 (at the embedding level), not the Distance component itself. By moving Distance to the attention mechanism (pre-softmax), the author expected to "rescue" the LOD combination and find better synergy between L, O, and D.

*Evidence:*
- The full experimental lineage: Exp1→Exp2→Exp5→Exp5b→Exp6 shows a systematic exploration — Exp6 is clearly positioned as "what if we do LOD but with the delta-V split from Exp5b + distance in attention from Exp1?"
- README-E6-Pre.md comparison table explicitly frames Exp6-Pre as completing the LOD combination that Exp5 left unfinished (Exp5 = L+O only; Exp6-Pre = L+O+D-pre)

---

### Variant B: exp6_lod_post

**"I probably implemented this because I believed that..."**

**Hypothesis 1: The learned semantic attention distribution should be protected from positional distortion.**
The author believed that the Q·K dot product — operating over combined embeddings that already include Legendre position labels — contains rich information about token relationships. Corrupting this with positional decay *before* normalization (as in Variant A) distorts the competition unfairly. Better to let the model freely learn attention patterns, then apply distance as a *modulator* on the outcome.

*Evidence:*
- Code comment at `attn.py:33`: `"# Step 2: Apply softmax FIRST (LOD POST)"`
- README-E6-POST.md hypothesis: "Applying distance decay after softmax may preserve the learned attention distribution better than pre-softmax decay, potentially leading to improved performance by: 1. Maintaining normalized attention weights; 2. Allowing the model to learn semantic relationships first; 3. Applying positional bias as a refinement step."
- README-E6-POST.md theoretical context: "Post-softmax...preserves attention distribution shape" vs "Pre-softmax...distorted before norm"

**Hypothesis 2: Post-softmax locality is a softer, more conservative constraint that may interact better with the Legendre label signal.**
Because Legendre polynomials are already in the Q and K embeddings, the attention scores already contain positional information. Applying pre-softmax decay would then "double-count" positional proximity — once via the Legendre similarity in Q·K and once via the explicit α(i,j) decay. Post-softmax avoids this double-counting by separating the two mechanisms: Q·K handles semantic+positional similarity; α(i,j) handles only the gating.

*Evidence:*
- README-E6-POST.md: "Advantage: Clear separation of semantic vs positional attention"
- The fact that the author explicitly noted the Label component (Legendre) is in Q/K while the Distance is post-softmax suggests awareness that both are positional signals — the author intentionally chose to apply them at different stages
- The separation of Q/K (from combined_emb, which includes Legendre) vs V (from delta_x) is itself a signal that the author was thinking carefully about *which information goes where* in the attention pipeline

**Hypothesis 3: Post-softmax should produce more stable training because the gradient path through the softmax is cleaner.**
With pre-softmax decay, the gradient flows backward through softmax through biased logits. With post-softmax decay, the gradient through softmax sees only clean logit-scaled scores. The author may have expected that post-softmax would converge more reliably.

*Evidence:*
- README-E6-POST.md explicitly notes under "Disadvantages" that "Gradient Flow: May affect backpropagation differently" for post-softmax
- The description of post-softmax as a "refinement step" implies the author expected it to be a gentler modification with more stable optimization properties

---

## Phase 7: Philosophical Comparison

| Question | Variant A: Pre-Softmax | Variant B: Post-Softmax |
|----------|----------------------|-------------------------|
| **What problem is being solved?** | How to make the attention *competition* locality-aware | How to make the attention *output* locality-aware without disrupting the competition |
| **What signal is being introduced?** | Locality bias into the probability assignment mechanism | Locality gating of already-assigned attention weights |
| **What assumption is being made?** | Nearby tokens should be intrinsically *more competitive* in the attention softmax | The softmax should freely determine which tokens are relevant; proximity should only moderate the contribution after relevance is determined |
| **Which inductive bias is being added?** | Hard positional prior (locality wins the competition) | Soft positional gate (locality moderates the output) |
| **Which relationships are being emphasized?** | Local semantic relationships (nearby tokens dominate attention weights) | Semantic relationships at all distances, with local relationships receiving larger output weights |
| **Which information is being suppressed?** | Distant-token score information (suppressed during normalization) | Distant-token attention weight contribution (suppressed after normalization, but distant tokens still win probability mass if semantically relevant) |
| **Where does distance "live" mathematically?** | Inside the exponent (via biased logit) | Outside the softmax (as a multiplicative mask on the output distribution) |
| **Is the attention output a probability distribution?** | Yes — softmax over biased logits produces valid probabilities | No — post-softmax multiplication breaks the sum-to-1 property |
| **What was the author testing?** | Whether locality as a competitive constraint improves LOD | Whether locality as a refinement filter improves LOD |
| **Philosophical name** | "Locality-constrained competition" | "Semantics-first, locality-second" |

---

## Phase 8: Human-Readable Explanation

### What Were You Trying to Do?

You had been building a new kind of positional encoding for the Informer transformer, testing it on a time-series forecasting task. You had three components you were experimenting with:

1. **Label (L):** A fixed mathematical fingerprint (Legendre polynomials) for each position in the sequence. Like giving each timestep a unique name tag that never changes.

2. **Order (O):** Instead of encoding what the value *is*, encode how much it *changed* from the previous step. You put this in the "value" part of the attention (the V matrix), so the model aggregates changes rather than absolute values.

3. **Distance (D):** A locality bias — the idea that nearby timesteps should be weighted more heavily than distant ones. The formula is `1 / (1 + |i−j|^a)` where `|i−j|` is how far apart two positions are.

In Experiment 6, you wanted to combine all three of these (L+O+D) — the full "LOD" stack. You had already tried combining them before (Experiment 2), but that earlier attempt put everything into the embedding layer. This time you were trying a more sophisticated split:
- L goes into the Q/K embeddings (position fingerprints help determine *which* tokens to attend to)
- O goes into the V matrix (temporal changes determine *what* to aggregate)
- D goes into the attention mechanism itself

### Why Did You Create Two Variants Instead of One?

There is a fundamental question about **where** to apply the distance decay within the attention formula. Attention has two stages:

**Stage 1:** Score every (query, key) pair: `score = Q·K / √d`  
**Stage 2:** Normalize the scores to probabilities: `A = softmax(scores)`

You can apply the decay either **between** Stage 1 and Stage 2 (Pre-Softmax), or **after** Stage 2 (Post-Softmax). These produce genuinely different behaviors, and you wanted to know which one is better.

### What Is the Conceptual Difference?

Think of it like a job interview process:

**Pre-Softmax (Variant A):** Before the candidates (tokens) compete for the job, you handicap the ones who live far away. Nearby candidates get a bonus. Only then do you pick the winner (softmax). The winner is determined by a combination of their qualifications AND how close they live.

**Post-Softmax (Variant B):** The candidates compete fairly based purely on qualifications (softmax). Then, after you've ranked them by merit, you apply a "commuting discount" — you reduce how much weight you give to distant candidates, even if they won the competition. The winner can still win, but their contribution is scaled down by how far they are.

### What Were You Trying to Prove?

You were testing a specific hypothesis: **does the placement of the distance decay (before or after the probability normalization) matter for forecasting performance?**

You already knew from earlier experiments:
- Distance alone (Exp1) helped some
- Label + Order together (Exp5) was better than distance alone
- But when you added distance to Label+Order (Exp2), it actually hurt

So in Experiment 6, you redesigned the LOD architecture using the better "split V" approach from Exp5b (where V carries temporal changes, not full embeddings), and then tested whether pre- or post-softmax distance decay would finally make the full LOD combination work.

### What Was the Architecture Change from Exp5b?

Experiment 5b had the same Q/K vs V split — combined embedding for Q/K, delta (change signal) for V. Experiment 6 added one more thing: distance decay applied to attention. In other words: Exp6 = Exp5b + D. The two variants test *where* D goes.

---

## Phase 9: Uncertainties and Confidence Scores

---

### FINDING A: The single defining architectural difference is pre- vs post-softmax decay in `attn.py`

```
Confidence: HIGH

Evidence:
- attn.py line 37 (Pre): scores = scores * alpha
- attn.py line 46 (Post): A = A * alpha
- Code comments in both files explicitly label the operation: "(LOD PRE)" and "(LOD POST)"
- README-E6-POST.md comparison table confirms this as the intended distinction
- All other files (embed.py Delta aside, encoder.py, decoder.py, model.py) are structurally identical
```

---

### FINDING B: The dropout asymmetry on delta_x is unintentional

```
Confidence: MEDIUM

Evidence:
- Pre returns: dropout(combined_emb), dropout(delta_x)
- Post returns: dropout(combined_emb), delta_x
- The README-E6-POST.md does not mention this difference anywhere in any table or section
- The Pre version's comment says "Apply dropout to both" which reads as intentional for Pre
- The Post version silently omits the second dropout with no comment explaining why
- This asymmetry was not included in any "Key Differences" table in either README

Reason for uncertainty:
- The Pre version may have been written first; the Post version may have dropped the dropout by oversight
- Alternatively, the Post version may have intentionally omitted delta dropout as a deliberate design choice
  to give the gradient path through V a cleaner signal
- No comment, no documentation, no test validates which interpretation is correct
```

---

### FINDING C: Both variants implement the same "clean delta" architecture from Exp5b

```
Confidence: HIGH

Evidence:
- embed.py in both variants: delta_x = value_emb - roll(value_emb, 1)
- attn.py in both variants: AttentionLayer.forward() accepts delta_values kwarg
- encoder.py in both variants: passes delta_x=delta_x to EncoderLayer
- decoder.py in both variants: passes delta_values=delta_x to self_attention, None to cross_attention
- The architecture is identical to exp5b_label_order_clean_delta_MV except for the distance decay addition
```

---

### FINDING D: The experiments were conceived as a pair to answer a specific question

```
Confidence: HIGH

Evidence:
- README-E6-POST.md explicitly frames Exp6-Post as testing: "Does post-softmax outperform pre-softmax?"
- README-E6-POST.md "Primary Questions" section lists this as Question 1
- The README contains a "Comparison with Exp6 Pre" section dedicated to the Pre/Post distinction
- Both experiments have identical non-attention code, confirming controlled experiment design
- The naming convention "lod_pre" vs "lod_post" directly encodes the comparison intent
```

---

### FINDING E: The delta computation differs from what the README describes

```
Confidence: HIGH (the implementation is correct; the README is outdated)

Evidence:
- README-E6-POST.md lines 237–241 describes delta as raw-input difference then re-embedded
- Actual embed.py code computes delta on already-embedded value_emb using torch.roll
- These are different operations at different stages of the computation graph
- The actual implementation (torch.roll on value_emb) is consistent with Exp5b and Exp6-Pre
- The README appears to document an earlier design iteration that was superseded

Implication for intent reconstruction:
- The ACTUAL delta (on embedded values) is what the author intended to use for training
- The README description of delta should be IGNORED for intent reconstruction
```

---

### FINDING F: The InformerStack in both model.py variants is broken for the LOD architecture

```
Confidence: HIGH

Evidence:
- exp6_lod_pre/models/model.py:154-158 — InformerStack.forward() calls
  enc_out = self.enc_embedding(x_enc, x_mark_enc)  [returns tuple but not unpacked]
  enc_out, attns = self.encoder(enc_out, ...)       [tuple being passed as single arg]
- exp6_lod_post/models/model.py:154-158 — Identical broken code

The primary Informer.forward() (lines 73-81) correctly unpacks:
  enc_out, delta_enc = self.enc_embedding(...)

Implication for intent:
- The author likely only used the Informer class (not InformerStack) for training
- The InformerStack class was not the intended execution path
- This breakage is consistent with InformerStack being a legacy/unused variant
```

---

### FINDING G: The encoder does not downsample delta_x when distil=True

```
Confidence: HIGH (this is a bug in both variants)

Evidence:
- encoder.py Encoder.forward() lines 71-73 (both variants):
  x, attn = attn_layer(x, delta_x=delta_x)
  x = conv_layer(x)                      ← x halved in length
  [delta_x is NOT processed by conv_layer — stays at original length]
- Exp5b correctly handles this: exp5b encoder.py lines 82-86 applies conv_layer to delta_x
- With e_layers=2, distil=True: there is 1 ConvLayer — after it, x=[B,48,512] but delta_x=[B,96,512]
- The final attention layer would receive mismatched shapes

Implication for intent:
- This bug means the experiments as implemented CANNOT run with default distil=True settings
- The author either used --distil False, or the bug was not discovered, or training was not completed
- This is consistent with the absence of any results directory for either Exp6 variant
```

---

## Remaining Ambiguities

| Ambiguity | Evidence Available | Resolution Path |
|-----------|-------------------|-----------------|
| Were any training runs actually completed with this exact code? | No results directories exist; central results file has identical values for both variants | Re-run both experiments independently |
| Is the dropout-on-delta asymmetry intentional? | No documentation; no test; no comment | Ask the author; or treat as identical by removing dropout from Pre OR adding it to Post |
| Which `--distil` setting was intended? | No shell scripts survive for either variant | Implied `distil=False` from the delta_x shape bug; but not confirmed |
| Was the README's delta formula (raw input difference → embed) an earlier design considered and discarded? | The README and code disagree; code is consistent with Exp5b; README appears older | The code is authoritative; the README documents a prior design |
| Were 90 runs (2 datasets × 3 decay × 3 seeds × 5 pred_len) actually planned for these experiments? | README-E6-POST.md states 90 runs; but only ETTh1 results appear in central file | Unknown; may have been aspirational scope in README |
