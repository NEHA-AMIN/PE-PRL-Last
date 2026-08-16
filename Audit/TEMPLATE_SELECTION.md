# TEMPLATE_SELECTION.md
# Evidence-Based Selection of Phase 1 Script Template
# ============================================================

## 1. Candidate Templates

All Phase 1 scripts were evaluated as potential templates for Exp6-Pre and Exp6-Post.

| Candidate | Architecture Match | Legendre Required | delta_x Tuple | decay_a Present | Score |
|-----------|-------------------|-------------------|---------------|-----------------|-------|
| Baseline | No (prob attn, no mods) | No | No | No | 0/5 |
| Exp5 Phase 1 | Partial (L+O, no clean delta) | Yes | Partial | No | 2/5 |
| **Exp5b Phase 1** | **Full (L+O + clean delta-V)** | **Yes** | **Yes** | **No** | **4/5** |
| Formula-A-pos Phase 1 | Partial (ordering_pos pe_mode) | Yes | No | No | 2/5 |
| Formula-B-pos Phase 1 | Partial (ordering_pos pe_mode) | Yes | No | No | 2/5 |

---

## 2. Selection: Exp5b Phase 1

**File**: `experiments/exp5b_label_order_clean_delta_MV/e5b_lab_ord_clean_delta_mv_ph1.sh`

### Justification

#### Architecture similarity
Exp5b is the closest predecessor to Exp6 in the experiment lineage:

| Feature | Exp5b | Exp6-Pre | Exp6-Post |
|---------|-------|----------|-----------|
| Label embedding | Yes | Yes | Yes |
| Order (Legendre) | Yes | Yes | Yes |
| Clean delta-V | Yes | Yes | Yes |
| Distance weighting pre-softmax | No | **Yes** | No |
| Distance weighting post-softmax | No | No | **Yes** |
| `decay_a` parameter | No | **Yes (=1.0)** | **Yes (=1.0)** |
| `legendre_embedding.py` required | Yes | Yes | Yes |
| `embed.py` returns tuple | Yes | Yes | Yes |

Exp6 = Exp5b + distance weighting. The delta_x tuple return, the legendre copy requirement, and the clean delta-V architecture are identical.

#### Structural similarity
Exp5b Phase 1 script structure:
- 7-file copy (including `legendre_embedding.py`)
- No alpha sweep
- Single seed (2021)
- pred_lens: {96, 192}
- Resume logic: `STATUS: COMPLETED`
- `python -u` (not `python3`)
- `git checkout ./models/` restore

All of these are required for Exp6. The only addition is `--decay_a 1.0`.

#### No `sed` patching needed
Unlike Exp1-post scripts which use `sed` to patch `decay_a` into `attn.py`, Exp6 Phase 1 uses `decay_a=1.0` which is the default in both `attn.py` and `model.py`. FINDING H fix ensures `--decay_a 1.0` is forwarded through `exp_informer.py` correctly using keyword arguments.

---

## 3. Modifications from Template

| Change | Reason |
|--------|--------|
| Script header updated | Exp6 architecture description |
| `EXP_DIR` path updated | Points to `exp6_lod_pre` / `exp6_lod_post` |
| `LOG_DIR` updated | Distinct log directory per experiment |
| `RUN_ID` format updated | `exp6pre_ph1_...` / `exp6post_ph1_...` |
| Add `--decay_a 1.0` to python invocation | Exp6 requires `decay_a`; Phase 1 uses α=1.0 |
| Run count comment updated | Still 2 runs |
| Decision guide updated | Reference benchmarks updated for Exp6 context |
| Architecture description updated | Explains pre vs post softmax placement |

Everything else is **identical** to the Exp5b template.

---

## 4. What Is NOT Modified

- All 22 fixed hyperparameters (verified to be identical)
- File copy pattern (same 7 files, same copy commands)
- Resume logic (same `STATUS: COMPLETED` pattern)
- Python invocation structure
- `git checkout ./models/` restore
- Summary table parsing logic

---

*Selection is evidence-based. Every claim traceable to source code reading.*
