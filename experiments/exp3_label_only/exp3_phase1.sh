#!/bin/bash
# =============================================================================
# Phase 1: Experiment 3 — Label Only (Pure Legendre Embedding)
#
# Experiment name  : exp3_label_only
# Mechanism        : Label only (Legendre polynomial position embedding)
#                    X'_i = X_i + P_i
#                    where P_i = Legendre(i)  [pure orthogonal labeling]
#                    NO temporal embedding, NO distance decay, NO alpha
#
# Phase 1 goal
# ------------
#   Run 2 pred_lens {96, 192} with a single seed (2021) to gauge whether
#   Legendre-only embedding provides any useful positional signal before
#   committing to full multi-seed / multi-horizon runs in Phase 2.
#
#   Dataset:      ETTh1 (multivariate, 7 features)
#   Seed:         2021 (single — exploration only)
#   Pred lengths: 96, 192 (short + mid horizon)
#   Total runs:   2  (1 config × 1 seed × 2 pred_len)
#
# Decision rule after Phase 1
# ---------------------------
#   Exp3 wins at BOTH pred_lens vs best previous experiment
#     → Proceed to Phase 2 (all seeds, all pred_lens)
#   Exp3 wins at ONE pred_len (mixed result)
#     → Still proceed to Phase 2, note instability
#   Exp3 loses at BOTH pred_lens clearly
#     → Document as negative result, skip Phase 2
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp3_label_only"
LOG_DIR="$PROJECT_ROOT/logs/exp3_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo "Experiment 3 — Phase 1: Label Only (Pure Legendre Embedding)"
echo "============================================================"
echo "What is being tested:"
echo "  Embedding : value + Legendre position only"
echo "  Formula   : X'_i = X_i + P_i"
echo "  No temporal embedding, no distance decay, no alpha"
echo "Fixed config  : scaling=True (hardcoded in embed.py)"
echo "Seed          : 2021 (Phase 1 single seed)"
echo "Pred lengths  : 96, 192 (Phase 1 short + mid horizon)"
echo "Total runs    : 2"
echo "Decision after: compare MSE vs best previous experiment"
echo "Start: $(date)"
echo "============================================================"
} | tee "$MASTER_LOG"

# --- FILE COPY ---
echo "" | tee -a "$MASTER_LOG"
echo "Copying experiment files..." | tee -a "$MASTER_LOG"
cd "$INFORMER_DIR" || { echo "ERROR: Cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

cp "$EXP_DIR/models/__init__.py"            ./models/__init__.py
cp "$EXP_DIR/models/attn.py"               ./models/attn.py
cp "$EXP_DIR/models/embed.py"              ./models/embed.py
cp "$EXP_DIR/models/encoder.py"            ./models/encoder.py
cp "$EXP_DIR/models/decoder.py"            ./models/decoder.py
cp "$EXP_DIR/models/model.py"              ./models/model.py
cp "$EXP_DIR/models/legendre_embedding.py" ./models/legendre_embedding.py

echo "Files copied successfully." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

SEED=2021
CONFIG="legendre_only"

for PRED_LEN in 96 192; do

    TOTAL=$((TOTAL + 1))
    RUN_ID="exp3_ph1_ETTh1_legendre_pred${PRED_LEN}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    mkdir -p "$RESULTS_DIR/${RUN_ID}"

    echo "" | tee -a "$MASTER_LOG"
    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "RUN $TOTAL/2: $RUN_ID"                                        | tee -a "$MASTER_LOG"
    echo "pred_len=$PRED_LEN | seed=$SEED | config=$CONFIG"             | tee -a "$MASTER_LOG"
    echo "Start: $(date)"                                                | tee -a "$MASTER_LOG"

    # Resume logic — skip if already completed
    if grep -q "STATUS: COMPLETED" "$RUN_LOG" 2>/dev/null; then
        echo "SKIPPING — already completed." | tee -a "$MASTER_LOG"
        SKIPPED=$((SKIPPED + 1))
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    if python3 -u main_informer.py \
        --model         informer \
        --data          ETTh1 \
        --root_path     ./data/ETT/ \
        --data_path     ETTh1.csv \
        --features      M \
        --target        OT \
        --freq          h \
        --checkpoints   ./checkpoints/ \
        --seq_len       96 \
        --label_len     48 \
        --pred_len      "$PRED_LEN" \
        --enc_in        7 \
        --dec_in        7 \
        --c_out         7 \
        --d_model       512 \
        --n_heads       8 \
        --e_layers      2 \
        --d_layers      1 \
        --s_layers      "3,2,1" \
        --d_ff          2048 \
        --factor        5 \
        --padding       0 \
        --distil \
        --dropout       0.05 \
        --attn          full \
        --embed         timeF \
        --activation    gelu \
        --mix \
        --itr           1 \
        --train_epochs  6 \
        --batch_size    32 \
        --patience      3 \
        --learning_rate 0.0001 \
        --des           "$RUN_ID" \
        --loss          mse \
        --lradj         type1 \
        --num_workers   0 \
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

# Restore original model files after experiment
git checkout ./models/

# =============================================================================
# SUMMARY
# =============================================================================
echo ""                                                                         | tee -a "$MASTER_LOG"
echo "============================================================"             | tee -a "$MASTER_LOG"
echo "PHASE 1 RESULTS SUMMARY — Experiment 3 (Label Only)"                    | tee -a "$MASTER_LOG"
echo "============================================================"             | tee -a "$MASTER_LOG"
echo "Run counters: TOTAL=$TOTAL | COMPLETED=$COMPLETED | FAILED=$FAILED | SKIPPED=$SKIPPED" | tee -a "$MASTER_LOG"
echo ""                                                                         | tee -a "$MASTER_LOG"
echo "Per-run MSE / MAE:"                                                       | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------"             | tee -a "$MASTER_LOG"

for PRED_LEN in 96 192; do
    RUN_ID="exp3_ph1_ETTh1_legendre_pred${PRED_LEN}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"

    if [ -f "$RUN_LOG" ]; then
        MSE=$(grep "^mse:" "$RUN_LOG" | tail -1 | sed -E 's/mse:([0-9]+\.[0-9]+).*/\1/')
        MAE=$(grep "^mse:" "$RUN_LOG" | tail -1 | sed -E 's/.*mae:([0-9]+\.[0-9]+).*/\1/')
        STATUS=$(grep "STATUS:" "$RUN_LOG" | tail -1 | sed 's/STATUS: //')
        MSE=${MSE:-"N/A"}
        MAE=${MAE:-"N/A"}
        STATUS=${STATUS:-"NO STATUS"}
    else
        MSE="NO LOG"; MAE="NO LOG"; STATUS="NO LOG"
    fi

    printf "pred_len=%-3s | MSE=%-26s | MAE=%-26s | %s\n" \
        "$PRED_LEN" "$MSE" "$MAE" "$STATUS" | tee -a "$MASTER_LOG"
done

echo ""                                                                         | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------"             | tee -a "$MASTER_LOG"
echo "REFERENCE — Best previous experiment (fill in before deciding):"         | tee -a "$MASTER_LOG"
echo "  Exp1-Pre alpha=1.0 | pred_len=96  | MSE=??? | MAE=???"                | tee -a "$MASTER_LOG"
echo "  Exp1-Pre alpha=1.0 | pred_len=192 | MSE=??? | MAE=???"                | tee -a "$MASTER_LOG"
echo "  Exp2-LOD           | pred_len=96  | MSE=??? | MAE=???"                | tee -a "$MASTER_LOG"
echo "  Exp2-LOD           | pred_len=192 | MSE=??? | MAE=???"                | tee -a "$MASTER_LOG"
echo ""                                                                         | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------"             | tee -a "$MASTER_LOG"
echo "DECISION GUIDE:"                                                          | tee -a "$MASTER_LOG"
echo "  Exp3 wins at BOTH pred_lens vs best previous"                          | tee -a "$MASTER_LOG"
echo "    → Proceed to Phase 2 (all seeds, all pred_lens)"                    | tee -a "$MASTER_LOG"
echo "  Exp3 wins at ONE pred_len (mixed result)"                              | tee -a "$MASTER_LOG"
echo "    → Still proceed to Phase 2, note instability"                        | tee -a "$MASTER_LOG"
echo "  Exp3 loses at BOTH pred_lens clearly"                                  | tee -a "$MASTER_LOG"
echo "    → Document as negative result, skip Phase 2"                         | tee -a "$MASTER_LOG"
echo "============================================================"             | tee -a "$MASTER_LOG"
echo "End: $(date)"                                                             | tee -a "$MASTER_LOG"
