#!/bin/bash
# =============================================================================
# Experiment 6 LOD Post-Softmax: Phase 1 — Alpha Screening Run
#
# Architecture summary:
#   Q, K ← project(value_emb + temporal_emb + legendre_pos)   [combined]
#   V    ← project(delta_x)   where delta_x[i] = x_i - x_{i-1}
#
#   Distance weighting (POST-softmax):
#     dist_matrix[i,j] = |i - j|
#     alpha[i,j]       = 1 / (1 + dist_matrix[i,j] ^ decay_a)
#     weights          = softmax(scores)               ← softmax FIRST
#     weights          = weights * alpha               ← applied AFTER softmax
#     output           = weights · V
#
# Key facts from code inspection:
#   - embed.py returns a TUPLE (combined_emb, delta_x) — non-standard
#   - embed.py applies dropout to combined_emb but NOT to delta_x (differs from Pre)
#   - legendre_embedding.py is imported inside embed.py → MUST be copied
#   - encoder.py: ConvLayer applied to both x and delta_x (FINDING G fix)
#   - model.py: InformerStack fully corrected (FINDING F fix)
#   - decay_a forwarded through exp_informer.py (FINDING H fix)
#
# Difference vs Exp6-Pre:
#   Pre:  scores = scores * alpha → softmax(scores) → weights  [stronger suppression]
#   Post: softmax(scores) → weights → weights = weights * alpha [softer, linear]
#
# Phase 1 goal:
#   Screen all three alpha values to identify which decay_a performs best
#   and compare pre- vs post-softmax placement across horizons.
#
# Runs: 3 alpha × 2 pred_lens × 1 seed = 6 runs
#   decay_a   : 0.5, 1.0, 2.0
#   pred_lens : 96, 192
#   seed      : 2021
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp6_lod_post"
LOG_DIR="$PROJECT_ROOT/logs/exp6_lod_post_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo " Experiment 6 LOD Post-Softmax — Phase 1: Alpha Screening"
echo " Date: $(date)"
echo "------------------------------------------------------------"
echo " Architecture:"
echo "   Q, K  ← combined_emb  (value + temporal + legendre)"
echo "   V     ← delta_x       (x_i - x_{i-1}, no dropout)"
echo "   weights = softmax(Q·K^T / sqrt(d))   [softmax first]"
echo "   alpha   = 1/(1 + |i-j|^decay_a)       [post-softmax]"
echo "   output  = (weights * alpha) · V"
echo "------------------------------------------------------------"
echo " Alpha sweep:   decay_a in {0.5, 1.0, 2.0}"
echo " Pred lens:     96, 192"
echo " Seed:          2021 (Phase 1 only)"
echo " Total runs:    6  (3 alpha × 2 pred_len × 1 seed)"
echo "------------------------------------------------------------"
echo " Bug fixes applied before run:"
echo "   FINDING F: InformerStack decay_a threading + forward() tuple fix"
echo "   FINDING G: delta_x downsampling through ConvLayer"
echo "   FINDING H: exp_informer.py now forwards --decay_a to model"
echo "------------------------------------------------------------"
echo " Reference results (Exp1-Pre, alpha=1.0, seed=2021):"
echo "   pred_len=96  → MSE ~0.8683  [update with actuals from logs]"
echo "   pred_len=192 → MSE ~0.8463  [update with actuals from logs]"
echo " Reference results (Exp5b, L+O clean delta, seed=2021):"
echo "   pred_len=96  → MSE TBD      [check exp5b_label_order_clean_delta_phase1 logs]"
echo "   pred_len=192 → MSE TBD      [check exp5b_label_order_clean_delta_phase1 logs]"
echo " Reference results (Exp6-Pre, LOD pre-softmax, seed=2021):"
echo "   pred_len=96  → MSE TBD      [check exp6_lod_pre_phase1 logs]"
echo "   pred_len=192 → MSE TBD      [check exp6_lod_pre_phase1 logs]"
echo "============================================================"
} | tee "$MASTER_LOG"

# --- FILE COPY ---
echo "" | tee -a "$MASTER_LOG"
echo "Copying model files to $INFORMER_DIR/models/ ..." | tee -a "$MASTER_LOG"

cp "$EXP_DIR/models/__init__.py"           "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"               "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"              "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"            "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"            "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"              "$INFORMER_DIR/models/model.py"
# CRITICAL: embed.py does `from legendre_embedding import LegendrePositionEmbedding`
# at import time. Without this file the module load fails at runtime.
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"

echo "File copy complete (7 files including legendre_embedding.py)." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

# --- LOOPS ---
# Phase 1: sweep decay_a in {0.5, 1.0, 2.0}, seed=2021 only, pred_len in {96, 192}

SEED=2021

for DECAY_A in 0.5 1.0 2.0; do
    for pred_len in 96 192; do

        TOTAL=$((TOTAL + 1))
        RUN_ID="exp6post_ph1_ETTh1_lod_post_a${DECAY_A}_pred${pred_len}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN $TOTAL/6: $RUN_ID" | tee -a "$MASTER_LOG"
        echo "pred_len=$pred_len | seed=$SEED | decay_a=$DECAY_A" | tee -a "$MASTER_LOG"
        echo "Start: $(date)" | tee -a "$MASTER_LOG"

        # RESUME LOGIC — skip if already completed (safe for Colab disconnects)
        if grep -q "STATUS: COMPLETED" "$RUN_LOG" 2>/dev/null; then
            echo "SKIPPING — already completed. Remove log to re-run." | tee -a "$MASTER_LOG"
            SKIPPED=$((SKIPPED + 1))
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

        # PYTHON RUN
        cd "$INFORMER_DIR" || { echo "ERROR: cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

        if python -u main_informer.py \
            --model informer \
            --data ETTh1 \
            --root_path ./data/ETT/ \
            --data_path ETTh1.csv \
            --features M \
            --attn full \
            --seq_len 96 \
            --label_len 48 \
            --pred_len "$pred_len" \
            --e_layers 2 \
            --d_layers 1 \
            --factor 5 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --d_model 512 \
            --n_heads 8 \
            --d_ff 2048 \
            --dropout 0.05 \
            --embed timeF \
            --freq h \
            --activation gelu \
            --train_epochs 6 \
            --patience 3 \
            --learning_rate 0.0001 \
            --batch_size 32 \
            --itr 1 \
            --decay_a "$DECAY_A" \
            --des "$RUN_ID" \
            2>&1 | tee "$RUN_LOG"; then

            COMPLETED=$((COMPLETED + 1))
            echo "STATUS: COMPLETED" | tee -a "$RUN_LOG"
            echo "STATUS: COMPLETED" | tee -a "$MASTER_LOG"
        else
            FAILED=$((FAILED + 1))
            echo "STATUS: FAILED — check $RUN_LOG" | tee -a "$MASTER_LOG"
        fi

        echo "End: $(date)" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"

    done
done

# --- RESTORE ORIGINAL MODELS ---
cd "$INFORMER_DIR" || exit 1
git checkout ./models/

# --- SUMMARY TABLE ---
echo "============================================================" | tee -a "$MASTER_LOG"
echo " PHASE 1 SUMMARY — Experiment 6 LOD Post-Softmax"           | tee -a "$MASTER_LOG"
echo " Total: $TOTAL | Completed: $COMPLETED | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo " Results by alpha (parsed from logs):"                        | tee -a "$MASTER_LOG"
echo ""                                                             | tee -a "$MASTER_LOG"

printf " %-66s %10s %10s\n" "RUN_ID" "MSE" "MAE" | tee -a "$MASTER_LOG"
printf " %-66s %10s %10s\n" "------------------------------------------------------------------" "----------" "----------" | tee -a "$MASTER_LOG"

for DECAY_A in 0.5 1.0 2.0; do
    for pred_len in 96 192; do
        RUN_ID="exp6post_ph1_ETTh1_lod_post_a${DECAY_A}_pred${pred_len}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"

        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -i "mse:" "$RUN_LOG" | tail -1 | grep -oP 'mse:\s*\K[0-9]+\.[0-9]+')
            MAE=$(grep -i "mae:" "$RUN_LOG" | tail -1 | grep -oP 'mae:\s*\K[0-9]+\.[0-9]+')
            MSE=${MSE:-"N/A"}
            MAE=${MAE:-"N/A"}
        else
            MSE="NO LOG"
            MAE="NO LOG"
        fi

        printf " %-66s %10s %10s\n" "$RUN_ID" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
    done
    echo "" | tee -a "$MASTER_LOG"
done

echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo " Reference (Exp1-Pre, alpha=1.0, seed=2021):"               | tee -a "$MASTER_LOG"
echo "   pred_96  → MSE ~0.8683  [update with actuals]"           | tee -a "$MASTER_LOG"
echo "   pred_192 → MSE ~0.8463  [update with actuals]"           | tee -a "$MASTER_LOG"
echo " Reference (Exp5b, L+O clean delta, seed=2021):"            | tee -a "$MASTER_LOG"
echo "   pred_96  → MSE TBD      [check exp5b logs]"              | tee -a "$MASTER_LOG"
echo "   pred_192 → MSE TBD      [check exp5b logs]"              | tee -a "$MASTER_LOG"
echo " Reference (Exp6-Pre, LOD pre-softmax, seed=2021):"         | tee -a "$MASTER_LOG"
echo "   pred_96  → MSE TBD      [check exp6_lod_pre_phase1 logs]" | tee -a "$MASTER_LOG"
echo "   pred_192 → MSE TBD      [check exp6_lod_pre_phase1 logs]" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"
echo " DECISION GUIDE:"                                            | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"
echo " Step 1 — Which alpha wins for Post-Softmax?"               | tee -a "$MASTER_LOG"
echo "   Compare MSE across decay_a in {0.5, 1.0, 2.0}."         | tee -a "$MASTER_LOG"
echo "   Smaller decay_a (0.5) → gentler re-weighting after softmax" | tee -a "$MASTER_LOG"
echo "   Larger decay_a (2.0)  → steeper re-weighting after softmax" | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"
echo " Step 2 — Does best Post alpha beat Exp5b (no distance)?"   | tee -a "$MASTER_LOG"
echo "   YES → Post-softmax distance adds value. Proceed to Phase 2." | tee -a "$MASTER_LOG"
echo "   NO  → Document as negative."                             | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"
echo " Step 3 — Pre vs Post comparison (same alpha):"             | tee -a "$MASTER_LOG"
echo "   Compare Exp6-Pre and Exp6-Post at each decay_a."         | tee -a "$MASTER_LOG"
echo "   Pre applies alpha before softmax → stronger suppression" | tee -a "$MASTER_LOG"
echo "   Post applies alpha after softmax → softer, linear effect" | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"
echo " Step 4 — Does best Post alpha beat Exp1-Pre (distance only)?" | tee -a "$MASTER_LOG"
echo "   pred_96  benchmark: MSE ~0.8683"                         | tee -a "$MASTER_LOG"
echo "   pred_192 benchmark: MSE ~0.8463"                         | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
