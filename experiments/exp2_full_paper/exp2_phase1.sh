#!/bin/bash
# =============================================================================
# Phase 1: Experiment 2 Full LOD on ETTh1 — Alpha Exploration
#
# Experiment name  : exp2_full_paper
# Mechanism        : L + O + D (full LOD formulation)
#                    X'_i = value_emb + temporal_emb + Legendre_pos + distance_pos
#                    distance_pos = distance_operator(legendre_pos) * 1/sqrt(d_model)
#                    α(i,j) = 1 / (1 + |i-j|^a)   [controlled by decay_a in embed.py]
#
# Phase 1 goal
# ------------
#   Explore three alpha values {0.5, 1.0, 2.0} at pred_len ∈ {96, 192}
#   with a single seed (2021) to identify which alpha is most promising
#   before committing to full multi-seed runs in Phase 2.
#
#   Alpha values: 0.5, 1.0, 2.0
#   Dataset:      ETTh1 (multivariate, 7 features)
#   Seed:         2021 (single — exploration only)
#   Pred lengths: 96, 192 (trend detection only)
#   Total runs:   6  (3 alpha × 1 seed × 2 pred_len)
#
# Decision rule after Phase 1
# ---------------------------
#   Same alpha wins at BOTH pred_lens → pattern stable → Phase 2
#     (best alpha only, all pred_lens {48, 96, 192, 336}, 3 seeds)
#   Different alpha wins at each → pattern unstable → run full grid
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp2_full_paper"
LOG_DIR="$PROJECT_ROOT/logs/exp2_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo "Exp2-LOD  |  PHASE 1 — ALPHA EXPLORATION"
echo "Components:   L + O + D (Full LOD formulation)"
echo "Alpha values: 0.5, 1.0, 2.0"
echo "Dataset:      ETTh1"
echo "Seed:         2021 (single — exploration only)"
echo "Pred lengths: 96, 192 (trend detection only)"
echo "Total runs:   6 (3 alpha × 1 seed × 2 pred)"
echo "Start time:   $(date)"
echo "============================================================"
} | tee "$MASTER_LOG"

# --- FILE COPY ---
cd "$INFORMER_DIR" || { echo "ERROR: Cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }
cp -r "$EXP_DIR/models/"* ./models/
echo "Files copied." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

SEED=2021

for ALPHA in 0.5 1.0 2.0; do

    # Patch decay_a in embed.py for the current alpha
    sed -i.bak -e "s/decay_a=[0-9.]*/decay_a=${ALPHA}/g" ./models/embed.py
    rm -f ./models/embed.py.bak
    echo "Patched embed.py: decay_a=${ALPHA}" | tee -a "$MASTER_LOG"

    for PRED_LEN in 96 192; do

        TOTAL=$((TOTAL + 1))
        RUN_ID="exp2_ph1_ETTh1_alpha${ALPHA}_pred${PRED_LEN}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "" | tee -a "$MASTER_LOG"
        echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN $TOTAL/6: $RUN_ID"                                        | tee -a "$MASTER_LOG"
        echo "pred_len=$PRED_LEN | seed=$SEED | alpha=$ALPHA"               | tee -a "$MASTER_LOG"
        echo "Start: $(date)"                                                | tee -a "$MASTER_LOG"
        echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"

        # Resume logic — check for MSE output
        if grep -q "^mse:" "$RUN_LOG" 2>/dev/null; then
            echo "SKIPPING — already completed with valid results" | tee -a "$MASTER_LOG"
            SKIPPED=$((SKIPPED + 1))
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

        if python3 -u main_informer.py \
            --model informer \
            --data ETTh1 \
            --root_path ./data/ETT/ \
            --data_path ETTh1.csv \
            --features M \
            --attn full \
            --seq_len 96 \
            --label_len 48 \
            --pred_len "$PRED_LEN" \
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
            --des "$RUN_ID" \
            2>&1 | tee "$RUN_LOG"; then

            if grep -q "^mse:" "$RUN_LOG"; then
                COMPLETED=$((COMPLETED + 1))
                echo "STATUS: COMPLETED" | tee -a "$RUN_LOG"
                echo "STATUS: COMPLETED" | tee -a "$MASTER_LOG"
            else
                FAILED=$((FAILED + 1))
                echo "STATUS: FAILED — exited 0 but no MSE found — check $RUN_LOG" | tee -a "$MASTER_LOG"
            fi
        else
            FAILED=$((FAILED + 1))
            echo "STATUS: FAILED — check $RUN_LOG" | tee -a "$MASTER_LOG"
        fi

        echo "End: $(date)" | tee -a "$MASTER_LOG"

    done
done

# Restore original model files
git checkout ./models/

# =============================================================================
# SUMMARY
# =============================================================================
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "PHASE 1 COMPLETE — PATTERN SUMMARY"                          | tee -a "$MASTER_LOG"
echo "Total: $TOTAL | Done: $COMPLETED | Skipped: $SKIPPED | Failed: $FAILED" | tee -a "$MASTER_LOG"
echo "End time: $(date)"                                           | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"

echo "MSE results extracted from logs:"                            | tee -a "$MASTER_LOG"
echo "Alpha  | PredLen | MSE      | MAE"                           | tee -a "$MASTER_LOG"
echo "-------|---------|----------|----------"                     | tee -a "$MASTER_LOG"

for ALPHA in 0.5 1.0 2.0; do
    for PRED_LEN in 96 192; do
        RUN_ID="exp2_ph1_ETTh1_alpha${ALPHA}_pred${PRED_LEN}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"

        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep "mse:" "$RUN_LOG" | tail -1 | sed -E 's/.*mse:([0-9]+\.[0-9]+).*/\1/')
            MAE=$(grep "mae:" "$RUN_LOG" | tail -1 | sed -E 's/.*mae:([0-9]+\.[0-9]+).*/\1/')
            MSE=${MSE:-"N/A"}
            MAE=${MAE:-"N/A"}
        else
            MSE="NO LOG"; MAE="NO LOG"
        fi

        printf "%-6s | %-7s | %-8s | %s\n" "$ALPHA" "$PRED_LEN" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
    done
done

echo "" | tee -a "$MASTER_LOG"
echo "NEXT STEP:"                                                                              | tee -a "$MASTER_LOG"
echo "  1. Check if same alpha wins at both pred_lens 96+192"                                 | tee -a "$MASTER_LOG"
echo "  2. If YES  → pattern stable → run Phase 2 (best alpha only, pred_lens 48/96/192/336, 3 seeds)" | tee -a "$MASTER_LOG"
echo "  3. If NO   → pattern unstable → must run full grid (3 alpha × 3 seeds × 4 pred_len)" | tee -a "$MASTER_LOG"
echo "  Phase 2 script: experiments/exp2_full_paper/exp2_phase2_alpha0.5.sh"                  | tee -a "$MASTER_LOG"
echo "  (update alpha in that script to match the winner found here)"                         | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
