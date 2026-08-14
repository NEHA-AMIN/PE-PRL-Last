# INCONSISTENCIES.md
## Complete Inconsistency Report — All Experiments
### Evidence-Based: Every claim traced to source code or result files

---

## SEVERITY CLASSIFICATION

- 🔴 **CRITICAL** — Invalidates experiment results or makes code unrunnable
- 🟡 **MODERATE** — Degrades comparison validity or documentation accuracy
- 🟢 **MINOR** — Typographical, cosmetic, or non-impacting

---

## I. RESULT INTEGRITY ISSUES

### 🔴 IC-001 — Exp6-Pre and Exp6-Post Have Identical Results
**Files:** `mse_mae_scores_sorted.txt` (lines ~499–584)  
**Evidence:** All Phase 1 (6 runs) and Phase 2 (9 runs) values are byte-for-byte identical between "Exp6-Pre" and "Exp6-Post" sections. Example:
```
Phase 1, α=1.0, pred=96: BOTH show MSE=0.8694, MAE=0.7332
Phase 2, pred=96, seed=2021: BOTH show MSE=0.7908, MAE=0.6579
```
**Impact:** Either (a) results were manually copy-pasted between the two entries, (b) both experiments accidentally executed the same model code, or (c) there was a recording error. In any case, the results table cannot be used to compare Exp6-Pre vs Exp6-Post. **The central finding distinguishing these two experiments is unverifiable.**  
**Resolution needed:** Re-run both experiments from scratch with independent logging.

---

### 🔴 IC-002 — README.md Summary Table Values Cannot Be Verified
**File:** `README.md` (lines 171–181)  
**Evidence:** The README results table claims:
- Baseline MSE=0.519 — **no log file exists in results/**
- Exp5 (L+O) MSE=0.719 — central results file shows 0.8519 (Phase 1) and 0.8655 avg (Phase 2)
- Exp1 (D pre) MSE=0.725 — central results file shows 0.8683 (Phase 1 best), 0.8796 (Phase 2 seed=2021)
- Exp2 (LOD) MSE=0.804 — central results file shows 0.8242 (Phase 1)
- Exp4 (O) MSE=0.835 — Phase 1 shows 0.8220; close
- Exp1-Post MSE=0.9072 — Phase 1 shows 0.8992 (α=1.0); Phase 2 seed=2021 shows 0.9206
- Exp3 (L) MSE=1.124 — central results file shows 1.0989 (Phase 1)

**Impact:** None of the key results in the project summary table can be traced to a specific logged run. The ordering claim "Exp5 > Exp1-D > Exp2" may not reflect the actual logged results.  
**Note:** It is possible these values come from earlier runs not preserved in the current results/ directory structure.

---

### 🔴 IC-003 — Baseline Results Directories Are Empty
**Files:** `results/baseline_ph1_ETTh1_pred96_seed2021/`, `results/baseline_ph1_ETTh1_pred192_seed2021/`  
**Evidence:** Both directories exist but contain zero files. No `training_log.txt`.  
**Impact:** Baseline MSE/MAE cannot be verified from any artifact in the repository.

---

### 🔴 IC-004 — Baseline Absent from Central Results File
**File:** `mse_mae_scores_sorted.txt`  
**Evidence:** The file documents all other experiments (Exp1-Pre through Exp6-Post) but has no "Baseline" entry.  
**Impact:** Cannot compare baseline against other experiments using the canonical results source.

---

### 🔴 IC-005 — README Ranking May Be Fabricated
**File:** `README.md` (lines 171–181)  
**Evidence:** Given IC-002 above — Exp5 is ranked 🥈 best ablation with MSE=0.719 but logged results show 0.8655 average. If the README numbers do not come from logged runs, the stated ranking "Exp5 > Exp1-D" cannot be substantiated.  
**Impact:** The primary scientific conclusion of the project ("Label + Order is the best ablation") is built on unverified numbers.

---

## II. METHODOLOGY ISSUES

### 🔴 IC-006 — Baseline Uses Different Attention Type Than All Ablations
**Files:** `experiments/Baseline/baseline_phase1.sh` (line 55: `--attn prob`), all ablation shell scripts (`--attn full`)  
**Evidence:** Baseline runs with **ProbSparse attention** (`ProbAttention`). Every ablation experiment runs with **FullAttention** (`--attn full`). These are architecturally different attention mechanisms.  
**Impact:** MSE differences between baseline and ablations are confounded by attention type, not just positional encoding. A controlled comparison would require ablations to also be evaluated with `--attn prob`, or baseline to be re-run with `--attn full`.

---

### 🔴 IC-007 — Baseline Uses Different Temporal Embedding Than Ablations
**Files:** `experiments/Baseline/baseline_phase1.sh` (line 74: `--embed timeF`), ablation shell scripts (no `--embed` flag → default `'fixed'`)  
**Evidence:** Baseline uses `TimeFeatureEmbedding` (linear layer on 4 continuous time features). Ablations use `TemporalEmbedding` (fixed sinusoidal embeddings on categorical hour/day/month/weekday indices).  
**Impact:** Temporal embedding architecture difference confounds the comparison.

---

### 🔴 IC-008 — Baseline Uses Different d_ff Than Ablations
**Files:** `experiments/Baseline/baseline_phase1.sh` (line 68: `--d_ff 2048`), ablation model.py defaults (`d_ff=512`)  
**Evidence:** Baseline uses `d_ff=2048` (4x d_model). Ablations use default `d_ff=512` (1x d_model, unusual).  
**Impact:** Baseline has 4x the feedforward capacity. This biases the baseline's advantage.

---

### 🟡 IC-009 — Exp6-Post Encoder May Have Shape Bug with distil=True
**File:** `experiments/exp6_lod_post/models/encoder.py` (lines 68-75)  
**Evidence:** When `distil=True` (default, `e_layers=2`), `ConvLayer` downsamples `x` from length 96 to ~48 but `delta_x` is NOT downsampled. The final `attn_layers[-1]` would receive mismatched `x` and `delta_x` shapes.  
**Comparison:** `exp5b/models/encoder.py:82-86` correctly downsamples `delta_x = conv_layer(delta_x)`.  
**Impact:** If run with default settings, Exp6-Post would crash. Either `distil=False` was used (undocumented) or the code was never successfully executed with this encoder.  
**Note:** Exp6-Pre has the same encoder code and the same potential bug.

---

## III. DOCUMENTATION ERRORS

### 🔴 IC-010 — Multiple READMEs Have Wrong pred_len in Configuration Table
**Files:** `README-E1.md`, `README-E2.md` (and likely others not fully read)  
**Evidence:** Both README files state `Prediction Length: 24` in their configuration table. Actual runs use pred_len 48, 96, 192, 336.  
**Impact:** Anyone trying to reproduce results from the README would use wrong prediction length.  
**Note:** This error is present in at least 2 experiments; likely copied from a common template.

---

### 🟡 IC-011 — Exp6-Pre Shell Scripts Referenced but Missing
**File:** `experiments/exp6_lod_pre/README-E6-Pre.md` (line 99: references `exp6_lod_pre_phase1.sh`)  
**Evidence:** Directory listing of `experiments/exp6_lod_pre/` shows no shell scripts.  
**Impact:** Cannot verify exact execution parameters for Exp6-Pre from shell scripts.

---

### 🟡 IC-012 — TECHNICAL_OBSERVATIONS.md Path References Are Stale
**File:** `TECHNICAL_OBSERVATIONS.md`  
**Evidence:** File contains references to `/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1` (line 440) which is a local machine path, not the repository path.  
**Impact:** Code references are unresolvable by other collaborators.

---

### 🟡 IC-013 — README-E2 Results Table Values Differ from Central Results
**File:** `experiments/exp2_full_paper/README-E2.md` (line 171: MSE=0.8036)  
**Evidence:** README-E2.md reports MSE=0.8036, MAE=0.7102 (from an early single run). Central results file shows Phase 1 as 0.8242/0.7279.  
**Impact:** Outdated results in experiment-level README.

---

### 🟡 IC-014 — TECHNICAL_OBSERVATIONS.md References Wrong Folder Name
**File:** `TECHNICAL_OBSERVATIONS.md` (line 33)  
**Evidence:** References `experiments/exp5b_label_order_clean_delta/models/embed.py:142-144`. Actual folder is `exp5b_label_order_clean_delta_MV`.  
**Impact:** Code reference resolves to nonexistent path.

---

### 🟡 IC-015 — run_phase1_baseline.sh Uses Wrong Directory Name
**File:** `run_phase1_baseline.sh`  
**Evidence:** Script references `Informer2020` directory but actual directory is `Informer2020-original`. Script would fail immediately.  
**Impact:** The master baseline script cannot be run as written.

---

### 🟡 IC-016 — experiments/Baseline/baseline_phase1.sh Uses Wrong Directory Name
**File:** `experiments/Baseline/baseline_phase1.sh` (line 9)  
**Evidence:** `INFORMER_DIR="$PROJECT_ROOT/Informer2020"` — same error as IC-015.  
**Impact:** Script fails at `cd "$INFORMER_DIR"`.

---

## IV. IMPLEMENTATION CONCERNS

### 🟡 IC-017 — Exp2's "Order" and "Distance" Are Not Independently Ablatable
**File:** `experiments/exp2_full_paper/models/distance_operator.py`  
**Evidence:** The `DistancePositionOperator` fuses both index-decay (α) and feature-weighting (w_ij) in a single operation. Setting α=1 (no index decay) or w_ij=1 (no feature weighting) would require code changes, not just parameter changes.  
**Impact:** Exp2 cannot be used to separately attribute performance to O vs D components.

---

### 🟡 IC-018 — Exp5's "Label" and "Order" Signals Are Not Independent
**File:** `experiments/exp5_label_order/models/ordering_operator.py`  
**Evidence:** `O_i = (1/(L-1))·Σ_{j≠i}(P_i−P_j)` algebraically equals `P_i − P̄`. So `X'_i = X_i + T_i + P_i + (P_i−P̄)/√d`. Label and Order are derived from the same Legendre vectors; they are not orthogonal signals.  
**Impact:** The "Label + Order" ablation is actually "Label + (Label minus mean)", not two distinct positional signals.

---

### 🟢 IC-019 — Dead Code: PositionalEmbedding/TemporalEmbedding Instantiated but Unused
**Files:** Multiple experiment `embed.py` files (Exp1-Pre, Exp3)  
**Evidence:** `self.position_embedding = PositionalEmbedding(...)` and `self.temporal_embedding = TemporalEmbedding(...)` instantiated in `__init__` but never called in `forward()`.  
**Impact:** Wasted memory allocation; no functional impact.

---

### 🟢 IC-020 — Dead Parameters: delta_queries/delta_keys in Exp5b AttentionLayer
**File:** `experiments/exp5b_label_order_clean_delta_MV/models/attn.py:202`  
**Evidence:** `delta_queries` and `delta_keys` parameters accepted but never used; only `delta_values` is used.  
**Impact:** Dead parameters; no functional impact.

---

### 🟢 IC-021 — Typographical: pe.require_grad vs pe.requires_grad
**File:** `Informer2020-original/models/embed.py:11`, multiple experiment embed.py files  
**Evidence:** `pe.require_grad = False` — missing trailing `s`. Should be `pe.requires_grad = False`.  
**Impact:** Functionally harmless because `register_buffer` prevents gradient tracking regardless.

---

### 🟡 IC-022 — Exp6 Distance Decay Active in Decoder Cross-Attention
**Files:** `exp6_lod_post/models/model.py:54-55`, `exp6_lod_pre/models/model.py:53-55`  
**Evidence:** Cross-attention in decoder (`FullAttention(False, ..., decay_a=decay_a)`) receives `decay_a`. Decoder cross-attention keys are encoder outputs (length ~seq_len) and queries are decoder steps (length label_len+pred_len). Index distance `|i-j|` has no semantic meaning across these two different sequences.  
**Impact:** Distance decay in cross-attention may introduce spurious biases; not documented.

---

## V. RESULTS COMPLETENESS

### 🟡 IC-023 — Multiple Experiments Missing Phase 2 Results
| Experiment | Phase 2 Status |
|------------|---------------|
| Exp3 (Label Only) | NOT FOUND in notebook outputs (stated in central file) |
| Exp5b (Clean Delta) | NOT FOUND in notebook outputs (stated in central file) |
| Exp4-SemSpace | NOT FOUND in notebook outputs (stated in central file) |
| Exp4a | NOT FOUND in notebook outputs (stated in central file) |
| Exp6-Pre | No results directory |
| Exp6-Post | No results directory |

**Impact:** Cannot assess stability (cross-seed variance) for multiple experiments.

---

## VI. SUMMARY TABLE

| ID | Severity | Category | Short Description |
|----|----------|----------|-------------------|
| IC-001 | 🔴 Critical | Results | Exp6-Pre and Exp6-Post have identical results |
| IC-002 | 🔴 Critical | Results | README summary values not in any log file |
| IC-003 | 🔴 Critical | Results | Baseline result directories empty |
| IC-004 | 🔴 Critical | Results | Baseline missing from central results file |
| IC-005 | 🔴 Critical | Results | README ranking unverifiable |
| IC-006 | 🔴 Critical | Methodology | Baseline uses ProbSparse; ablations use FullAttention |
| IC-007 | 🔴 Critical | Methodology | Baseline uses timeF embed; ablations use fixed |
| IC-008 | 🔴 Critical | Methodology | Baseline uses d_ff=2048; ablations use d_ff=512 |
| IC-009 | 🟡 Moderate | Implementation | Exp6-Post encoder may crash with distil=True |
| IC-010 | 🔴 Critical | Documentation | README config tables show wrong pred_len=24 |
| IC-011 | 🟡 Moderate | Documentation | Exp6-Pre shell scripts missing |
| IC-012 | 🟡 Moderate | Documentation | Stale local machine paths in docs |
| IC-013 | 🟡 Moderate | Documentation | README-E2 results outdated |
| IC-014 | 🟡 Moderate | Documentation | TECH_OBS references wrong folder name |
| IC-015 | 🟡 Moderate | Scripts | run_phase1_baseline.sh references wrong dir |
| IC-016 | 🟡 Moderate | Scripts | Baseline phase1.sh references wrong dir |
| IC-017 | 🟡 Moderate | Implementation | Exp2 O and D are fused, not ablatable |
| IC-018 | 🟡 Moderate | Implementation | Exp5 L and O are not independent signals |
| IC-019 | 🟢 Minor | Implementation | Dead code: unused embeddings instantiated |
| IC-020 | 🟢 Minor | Implementation | Dead parameters in attn.py |
| IC-021 | 🟢 Minor | Implementation | require_grad typo (harmless) |
| IC-022 | 🟡 Moderate | Implementation | Cross-attention decay semantically undefined |
| IC-023 | 🟡 Moderate | Results | Multiple experiments missing Phase 2 |

**Total: 8 Critical, 11 Moderate, 4 Minor**
