#!/bin/bash
# =============================================================================
# Experiment 4: Phase 1 EXPLORATION — Ordering Only
# Formula:  X'_i = X_i + T_i + O_i
# O_i = (1/N-1) · Σ_{j≠i} (X_i - X_j)  — pure directional signal
# No distance decay, no Legendre, no alpha
# Runs:     2 (pred_len 96, 192 × seed 2021)
# =============================================================================

PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp4_order_only"
LOG_DIR="$PROJECT_ROOT/logs/exp4_order_only_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

echo "============================================================" | tee "$MASTER_LOG"
echo "Experiment 4 — Phase 1: Ordering Only"                       | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "Formula   : X'_i = X_i + T_i + O_i"                         | tee -a "$MASTER_LOG"
echo "Value     : TokenEmbedding (semantic content)"                | tee -a "$MASTER_LOG"
echo "Temporal  : TimeFeatureEmbedding"                             | tee -a "$MASTER_LOG"
echo "Order     : O_i = (1/N-1) * sum_j(X_i - X_j)"               | tee -a "$MASTER_LOG"
echo "  Signed displacements — pure directional signal"            | tee -a "$MASTER_LOG"
echo "  NO index-based decay alpha(i,j)"                           | tee -a "$MASTER_LOG"
echo "  NO feature-space weighting w_ij"                           | tee -a "$MASTER_LOG"
echo "  NO Legendre labels"                                         | tee -a "$MASTER_LOG"
echo ""                                                             | tee -a "$MASTER_LOG"
echo "Seed      : 2021 | Pred lengths: 96, 192 | Total runs: 2"    | tee -a "$MASTER_LOG"
echo "Start: $(date)"                                               | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "Copying experiment files..." | tee -a "$MASTER_LOG"
cp "$EXP_DIR/models/__init__.py"          "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"              "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"             "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"           "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"           "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"             "$INFORMER_DIR/models/model.py"
cp "$EXP_DIR/models/ordering_operator.py" "$INFORMER_DIR/models/ordering_operator.py"
echo "Files copied successfully." | tee -a "$MASTER_LOG"

cd "$INFORMER_DIR" || { echo "ERROR: Cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

TOTAL=0; COMPLETED=0; FAILED=0; SKIPPED=0
SEED=2021

for PRED_LEN in 96 192; do
    TOTAL=$((TOTAL + 1))
    RUN_ID="exp4_ph1_ETTh1_ordering_pred${PRED_LEN}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    mkdir -p "$RESULTS_DIR/${RUN_ID}"

    echo "" | tee -a "$MASTER_LOG"
    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "RUN $TOTAL/2: $RUN_ID"                                        | tee -a "$MASTER_LOG"
    echo "pred_len=$PRED_LEN | seed=$SEED | config=ordering_only"       | tee -a "$MASTER_LOG"
    echo "Start: $(date)"                                                | tee -a "$MASTER_LOG"

    # Resume logic — check for actual MSE output
    if grep -q "^mse:" "$RUN_LOG" 2>/dev/null; then
        echo "SKIPPING — already completed with valid results" | tee -a "$MASTER_LOG"
        SKIPPED=$((SKIPPED + 1))
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    if python -u main_informer.py \
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

# =============================================================================
# SUMMARY
# =============================================================================
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "PHASE 1 RESULTS — Experiment 4 (Ordering Only)"             | tee -a "$MASTER_LOG"
echo "TOTAL=$TOTAL | COMPLETED=$COMPLETED | FAILED=$FAILED | SKIPPED=$SKIPPED" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "PredLen | MSE      | MAE      | Status"                      | tee -a "$MASTER_LOG"
echo "--------|----------|----------|-------"                       | tee -a "$MASTER_LOG"

for PRED_LEN in 96 192; do
    RUN_ID="exp4_ph1_ETTh1_ordering_pred${PRED_LEN}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    if [ -f "$RUN_LOG" ]; then
        MSE=$(grep -oP 'mse:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
        MAE=$(grep -oP 'mae:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
        STATUS=$(grep -oP 'STATUS: \K\w+' "$RUN_LOG" | tail -1)
        printf "%-7s | %-8s | %-8s | %s\n" "$PRED_LEN" "$MSE" "$MAE" "$STATUS" | tee -a "$MASTER_LOG"
    else
        echo "$PRED_LEN  | LOG NOT FOUND" | tee -a "$MASTER_LOG"
    fi
done

echo "" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo "REFERENCE — Previous experiments (seed=2021):"               | tee -a "$MASTER_LOG"
echo "Experiment          | pred=96 MSE | pred=192 MSE"            | tee -a "$MASTER_LOG"
echo "--------------------|-------------|-------------"             | tee -a "$MASTER_LOG"
echo "Exp1-Pre a=1.0      | 0.8683      | 0.8463  ← best so far"  | tee -a "$MASTER_LOG"
echo "Exp2-LOD            | 0.8242      | 0.9002"                  | tee -a "$MASTER_LOG"
echo "Exp3b-LabelTemporal | 0.9265      | 0.8888"                  | tee -a "$MASTER_LOG"
echo "Exp3-LabelOnly      | 1.0989      | 1.4844  ← no temporal"   | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "DECISION GUIDE:"                                              | tee -a "$MASTER_LOG"
echo "  Exp4 wins at BOTH vs Exp1-Pre → ordering signal helps → Phase 2"      | tee -a "$MASTER_LOG"
echo "  Exp4 between Exp3b and Exp1-Pre → partial signal → Phase 2 to confirm" | tee -a "$MASTER_LOG"
echo "  Exp4 worse than Exp3b at both → ordering adds noise → stop"            | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "End: $(date)" | tee -a "$MASTER_LOG"