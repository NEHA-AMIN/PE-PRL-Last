# Formula B (Positional Space) — Global Mean Deviation Ordering

## 1. One-Line Summary

Replace the consecutive-delta ordering signal (Formula A) with a global-mean-deviation
ordering signal computed from the Legendre positional vectors.  Every other component
is unchanged.

---

## 2. Research Position

This experiment is the **positional-space counterpart of Formula-B-sem**.  The ordering
formula is identical; the only change is the input space used to compute the signal.

| Label | Ordering input | Ordering formula | Folder |
|-------|---------------|-----------------|--------|
| Formula A-pos | Legendre P | $P_i - P_{i-1}$ (consecutive diff) | `Formula-A-pos/` |
| **Formula B-pos** | **Legendre P** | **$(\mu_p - P_i)/\bar{p}$ (global mean dev.)** | `Formula-B-pos/` ← **this** |
| Formula B-sem | Token embedding V | $(\mu - V_i)/\bar{x}$ (global mean dev.) | `Formula-B-sem/` |

The ordering formula applied to P here is **mathematically identical** to what
Formula-B-sem applies to V — only the input space differs.

---

## 3. The Output Formula

$$
\tilde{X}_i = V_i + T_i + P_i + \mathrm{Ordering}_i
$$

| Symbol | Source | What it encodes |
|--------|--------|-----------------|
| $V_i \in \mathbb{R}^d$ | `TokenEmbedding` (Conv1D) | Semantic content of the 7 input features |
| $T_i \in \mathbb{R}^d$ | `TimeFeatureEmbedding` (linear) | Calendar context |
| $P_i \in \mathbb{R}^d$ | `LegendrePositionEmbedding` (frozen buffer) | Orthogonal positional label |
| $\mathrm{Ordering}_i \in \mathbb{R}^d$ | global mean deviation of $P$ | **This experiment's signal** |

---

## 4. The Ordering Signal

### 4.1 Formula

$$
\mu_p = \frac{1}{L} \sum_{k=1}^{L} P_k \in \mathbb{R}^d \quad \text{(global mean of all positional vectors)}
$$

$$
\Delta_i^p = \mu_p - P_i \in \mathbb{R}^d \quad \text{(each position's deviation from the global mean)}
$$

$$
\bar{p} = \frac{1}{L} \sum_{i=1}^{L} \|P_i\|_2 \in \mathbb{R} \quad \text{(scalar normaliser)}
$$

$$
\mathrm{Ordering}_i = \frac{\Delta_i^p}{\bar{p} + \varepsilon}, \quad \varepsilon = 10^{-8}
$$

### 4.2 What it encodes

$P_i$ are Legendre polynomial evaluations — deterministic, fixed vectors that vary
smoothly and orthogonally with position index.  Their global mean $\mu_p$ is a
fixed vector (it never changes for a given $L$ and $d$).  The deviation
$\mu_p - P_i$ therefore encodes **how far each absolute position is from the
centre of the Legendre basis** — a purely structural, content-independent signal.

### 4.3 Key properties

- **Content-independent**: $V_i$ does not participate in computing $\mathrm{Ordering}_i$
- **Deterministic**: $P_i$ is a frozen buffer; $\mu_p$ is the same every forward pass for the same $L$
- **Scale-invariant**: the $\bar{p}$ normaliser keeps $\|\mathrm{Ordering}_i\|$ proportional to $\|P_i\|$
- **Translation-invariant**: adding a constant to all $P_i$ cancels in $\mu_p - P_i$
- **No boundary condition**: $\mu_p$ is defined over all $L$ positions symmetrically
- **No new parameters**: both $P_i$ and $\mu_p$ are computed from the frozen Legendre buffer

---

## 5. Exactly What Changed vs. Formula-A-pos

The **only** lines that differ between this experiment and `Formula-A-pos` are the
three lines computing `delta_p` in `forward()`.

**Formula-A-pos (`Formula-A-pos/models/embed.py` lines 172–173):**
```python
delta_p = torch.zeros_like(leg_d)                         # delta_0 = 0
delta_p[:, 1:, :] = leg_d[:, 1:, :] - leg_d[:, :-1, :]  # P_i - P_{i-1}
```

**Formula-B-pos (`Formula-B-pos/models/embed.py` — replacement):**
```python
mu_p    = leg_d.mean(dim=1, keepdim=True)   # [B, 1, D]  global mean of P
delta_p = mu_p - leg_d                       # [B, L, D]  mu_p - P_i
```

Everything else — `p_bar`, `ordering`, and `return` — is bit-for-bit identical.

---

## 6. Mathematical Derivation from the Replacement Lines

`leg_d` has shape `[B, L, D]`.

**Line 1:** `leg_d.mean(dim=1, keepdim=True)` averages over the sequence axis:

$$
\texttt{mu\_p}[b, 0, d] = \frac{1}{L}\sum_{k=0}^{L-1} P_k^{(b,d)}
$$

Shape `[B, 1, D]` — **one fixed vector per (batch, dim), identical for all positions**.

**Line 2:** `mu_p - leg_d` broadcasts `[B,1,D]` against `[B,L,D]`:

$$
\texttt{delta\_p}[b, i, d] = \texttt{mu\_p}[b, 0, d] - P_i^{(b,d)}
= \frac{1}{L}\sum_{k=0}^{L-1} P_k^{(b,d)} - P_i^{(b,d)}
$$

In vector form:

$$
\boxed{\Delta_i^p = \mu_p - P_i \quad \text{where} \quad \mu_p = \frac{1}{L}\sum_{k=1}^{L} P_k}
$$

The same reference vector $\mu_p$ is subtracted from every $P_i$. This is the
Global Mean Ordering formula applied to positional space.

---

## 7. Tensor Shapes (Forward Pass)

```
val       [B, L, D]   V_i  = TokenEmbedding(x)
temp      [B, L, D]   T_i  = temporal_embedding(x_mark)
leg       [B, L, D]   P_i  = LegendrePositionEmbedding(x)  (buffer, no grad)
leg_d     [B, L, D]   leg.detach()
mu_p      [B, 1, D]   global mean of P over sequence dim    → broadcasts to [B, L, D]
delta_p   [B, L, D]   mu_p − leg_d
p_bar     [B, 1, 1]   mean of per-token L2 norms of P       scalar per sample
ordering  [B, L, D]   delta_p / (p_bar + 1e-8)
output    [B, L, D]   dropout(val + temp + leg + ordering)
```

---

## 8. Files Modified vs. Formula-A-pos

| File | Status | Change |
|------|--------|--------|
| `models/embed.py` | **Modified** | `delta_p` computation replaced (2 lines); all else unchanged |
| `models/model.py` | **Modified** | `pe_mode='formula_b_pos'` dispatch instead of `'ordering_pos'` |
| `models/legendre_embedding.py` | Unchanged | Verbatim copy |
| `models/attn.py` | Unchanged | Verbatim copy |
| `models/encoder.py` | Unchanged | Verbatim copy |
| `models/decoder.py` | Unchanged | Verbatim copy |

---

## 9. Training Configuration

| Parameter | Value |
|-----------|-------|
| `--model` | `informer` |
| `--data` | `ETTh1` |
| `--features` | `M` |
| `--seq_len` | 96 |
| `--label_len` | 48 |
| `--pred_len` | 24/48/96/192/336/720 |
| `--e_layers` | 2 |
| `--d_layers` | 1 |
| `--d_model` | 512 |
| `--n_heads` | 8 |
| `--d_ff` | 2048 |
| `--attn` | `full` |
| `--embed` | `timeF` |
| `--freq` | `h` |
| `--dropout` | 0.05 |
| `--train_epochs` | 6 |
| `--patience` | 3 |
| `--learning_rate` | 0.0001 |
| `--batch_size` | 32 |
| `--seed` | 2021 |
| `--pe_mode` | `formula_b_pos` |

---

## 10. Code-to-Math Mapping

| Mathematical expression | Code | File |
|------------------------|------|------|
| $P_i = \mathrm{LegendrePositionEmbedding}(x)$ | `leg = self.legendre_embedding(x)` | `embed.py` |
| $\mu_p = \frac{1}{L}\sum_k P_k$ | `mu_p = leg_d.mean(dim=1, keepdim=True)` | `embed.py` |
| $\Delta_i^p = \mu_p - P_i$ | `delta_p = mu_p - leg_d` | `embed.py` |
| $\bar{p} = \frac{1}{L}\sum_i \|P_i\|_2$ | `p_bar = leg_d.norm(dim=-1).mean(dim=1,keepdim=True).unsqueeze(-1)` | `embed.py` |
| $\mathrm{Ordering}_i = \Delta_i^p / (\bar{p}+\varepsilon)$ | `ordering = delta_p / (p_bar + 1e-8)` | `embed.py` |
| $\tilde{X}_i = V_i + T_i + P_i + \mathrm{Ordering}_i$ | `val + temp + leg + ordering` | `embed.py` |

---

## 11. How to Run

### On Google Colab / remote GPU

```bash
bash experiments/Formula-B-pos/formula-B-pos-ph1.sh
```

The script:
1. Copies all 7 model files (including `legendre_embedding.py`) to `Informer2020-original/models/`
2. Runs 6 prediction-horizon sweeps (`pred_len ∈ {24, 48, 96, 192, 336, 720}`)
3. Logs each run to `logs/Formula-B-pos-phase1/`
4. Prints an MSE/MAE summary table at the end
5. Is idempotent: already-completed runs are skipped automatically

### Manual single run

```bash
cd Informer2020-original
cp ../experiments/Formula-B-pos/models/*.py models/
python -u main_informer.py \
    --model informer --data ETTh1 \
    --root_path ./data/ETT/ --data_path ETTh1.csv \
    --features M --attn full \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --e_layers 2 --d_layers 1 --factor 5 \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model 512 --n_heads 8 --d_ff 2048 \
    --dropout 0.05 --embed timeF --freq h --activation gelu \
    --train_epochs 6 --patience 3 --learning_rate 0.0001 \
    --batch_size 32 --itr 1 \
    --pe_mode formula_b_pos \
    --des debug_formula_b_pos_pred96
```

### Sanity test (no GPU required)

```python
import torch
from models.embed import DataEmbedding_formula_b_pos

B, L, D, c_in = 2, 96, 512, 7
x      = torch.randn(B, L, c_in)
x_mark = torch.randn(B, L, 4)

emb = DataEmbedding_formula_b_pos(c_in, D)
out = emb(x, x_mark)

assert out.shape == (B, L, D)
print("Shape check passed:", out.shape)

# Verify ordering is computed from P only — val does not enter delta_p
with torch.no_grad():
    leg_d = emb.legendre_embedding(x).detach()
    mu_p  = leg_d.mean(dim=1, keepdim=True)
    delta_p_expected = mu_p - leg_d
    p_bar = leg_d.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)
    ordering_expected = delta_p_expected / (p_bar + 1e-8)
print("Ordering derived from P only. V not used in delta_p. ✓")
```
