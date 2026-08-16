# SCRIPT_INVENTORY.md
# Complete Inventory of All Shell Scripts in Repository
# Generated: forensic analysis — every file verified by reading
# ============================================================

## 1. Scope

All `.sh` files in the repository were discovered via recursive directory scan.
Each entry below records: path, phase, alpha sweep, seeds, pred_lens, run count.

---

## 2. Complete Script Inventory Table

| # | File Path | Experiment | Phase | Alpha Values | Seeds | Pred Lens | Run Count |
|---|-----------|------------|-------|-------------|-------|-----------|-----------|
| 1 | `experiments/Baseline/baseline_phase1.sh` | Baseline | 1 | N/A | 2021 | 96, 192 | 2 |
| 2 | `experiments/Baseline/baseline_phase2.sh` | Baseline | 2 | N/A | 2021,2022,2023 | 48,96,192,336,720 | 15 |
| 3 | `experiments/exp1_distance_pre_softmax/exp1_pre_phase1.sh` | Exp1-Pre | 1 | N/A (default=1.0) | 2021 | 96, 192 | 2 |
| 4 | `experiments/exp1_distance_pre_softmax/exp1_pre_phase2.sh` | Exp1-Pre | 2 | 0.5,1.0,2.0 | 2021,2022,2023 | 48,96,192,336 | 36 |
| 5 | ~~`experiments/exp1_distance_post_softmax/exp1_post_phase1.sh`~~ **MISSING — never created** | Exp1-Post | 1 | 0.5,1.0,2.0 | 2021 | 96, 192 | **6 (gap)** |
| 6 | `experiments/exp1_distance_post_softmax/exp1_post_phase2_alpha0.5.sh` | Exp1-Post | 2 | 0.5 (sed-patched) | 2021,2022,2023 | 48,96,192,336 | 12 |
| 7 | `experiments/exp1_distance_post_softmax/exp1_remaining_ball.sh` | Exp1-Post | 2 | (varies) | (varies) | (varies) | varies |
| 8 | `experiments/exp1_distance_post_softmax/exp1_alpha_0.5.sh` | Exp1-Post | 2 | 0.5 (sed-patched) | 2021,2022,2023 | 48,96,192,336 | 12 |
| 9 | `experiments/exp5_label_order/exp5_label_order_ph1.sh` | Exp5 | 1 | N/A | 2021 | 96, 192 | 2 |
| 10 | `experiments/exp5b_label_order_clean_delta_MV/e5b_lab_ord_clean_delta_mv_ph1.sh` | Exp5b | 1 | N/A | 2021 | 96, 192 | 2 |
| 11 | `experiments/Formula-A-pos/formula-A-pos-ph1.sh` | Formula-A-pos | 1 | N/A | 2021 | 96, 192 | 2 |
| 12 | `experiments/Formula-B-pos/formula-B-pos-ph1.sh` | Formula-B-pos | 1 | N/A | 2021 | 96, 192 | 2 |
| 13 | `experiments/Formula-A-sem/formula-A-sem-ph1.sh` | Formula-A-sem | 1 | N/A | 2021 | 96, 192 | 2 |
| 14 | `experiments/Formula-B-sem/formula-B-sem-ph1.sh` | Formula-B-sem | 1 | N/A | 2021 | 96, 192 | 2 |
| 15 | `experiments/exp6_lod_pre/exp6_lod_pre_phase1.sh` | Exp6-Pre | 1 | **0.5, 1.0, 2.0** | 2021 | 96, 192 | **6** |
| 16 | `experiments/exp6_lod_post/exp6_lod_post_phase1.sh` | Exp6-Post | 1 | **0.5, 1.0, 2.0** | 2021 | 96, 192 | **6** |

**Gap identified:**

| Gap | File | Status |
|-----|------|--------|
| Exp1-Post Phase 1 script was never created | `exp1_distance_post_softmax/exp1_post_phase1.sh` | **MISSING** — jumped directly to Phase 2 (alpha=0.5 only). Phase 1 screening was skipped. |

---

## 3. Phase Classification

### Phase 1 Scripts (exploration)
- Single seed: 2021
- pred_lens: {96, 192}
- Goal: determine if experiment beats reference benchmarks
- Run count: 2 (no alpha sweep) or 6 (3 alpha values × 2 pred_lens, for decay_a experiments)

### Phase 2 Scripts (validation)
- Multiple seeds: 2021, 2022, 2023
- pred_lens: {48, 96, 192, 336} (sometimes +720)
- Goal: statistical stability confirmation
- Run count: 12–36+

---

## 4. Alpha / `decay_a` Handling Across Scripts

| Script | Has decay_a | Values | Approach |
|--------|-------------|--------|----------|
| Baseline | No | — | — |
| Exp1-Pre Phase 1 | No | default=1.0 in attn.py | hardcoded default |
| Exp1-Pre Phase 2 | Yes | 0.5, 1.0, 2.0 | `sed` patch on `attn.py` |
| **Exp1-Post Phase 1** | **MISSING** | — | **script never created** |
| Exp1-Post Phase 2 | Yes | 0.5 only | `sed` patch on `attn.py` |
| Exp5 Phase 1 | No | — | — |
| Exp5b Phase 1 | No | — | — |
| Formula-A-pos Phase 1 | No | — | — |
| Formula-B-pos Phase 1 | No | — | — |
| Formula-A-sem Phase 1 | No | — | — |
| Formula-B-sem Phase 1 | No | — | — |
| **Exp6-Pre Phase 1** | Yes | **0.5, 1.0, 2.0** | `--decay_a` CLI flag (FINDING H fix) |
| **Exp6-Post Phase 1** | Yes | **0.5, 1.0, 2.0** | `--decay_a` CLI flag (FINDING H fix) |

**Note:** The `--decay_a` CLI flag existed in `main_informer.py` (line 41) but was never forwarded to the model constructor in `exp_informer.py`. FINDING H fix passes it as a keyword argument `decay_a=getattr(self.args, 'decay_a', 1.0)` in `_build_model()`, making the CLI flag functional for all experiments.

---

## 5. `legendre_embedding.py` Copy Requirement

| Script | Copies legendre_embedding.py | Evidence |
|--------|------------------------------|---------|
| Baseline | No | embed.py is vanilla DataEmbedding |
| Exp1-Pre/Post | No | embed.py is original, no legendre import |
| Exp5 Phase 1 | Yes | cp explicitly in script |
| Exp5b Phase 1 | Yes | cp explicitly in script, comment explains why |
| Formula-A/B-pos/sem Phase 1 | Yes | cp explicitly in script |
| **Exp6-Pre Phase 1** | Yes | embed.py imports LegendrePositionEmbedding |
| **Exp6-Post Phase 1** | Yes | embed.py imports LegendrePositionEmbedding |

---

## 6. Resume Logic Variants

| Variant | Pattern | Scripts Using It |
|---------|---------|-----------------|
| A | `grep -q "STATUS: COMPLETED" "$RUN_LOG"` | Exp5, Exp5b, Formula-A/B-pos/sem |
| B | `grep -q "^mse:" "$RUN_LOG"` | Baseline, Exp1-post |

Variant A is preferred (more explicit). Exp6 Phase 1 scripts will use Variant A.

---

*All data derived from direct reading of source files. No assumptions made.*
