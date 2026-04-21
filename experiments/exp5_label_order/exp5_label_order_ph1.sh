#!/bin/bash
# =============================================================================
# Experiment 5: Phase 1 EXPLORATION — Label + Order (L + O)
#
# What this tests:
#   Formula: X'_i = X_i + T_i + P_i + O_i
#
#   X_i  = value_embedding(x)           [Semantic content]
#   T_i  = temporal_embedding(x_mark)   [Time features]
#   P_i  = Legendre(i)                  [Label - orthogonal distinctiveness]
#   O_i  = OrderingOperator(P_i)        [Order - signed displacements in POSITIONAL space]
#
# CRITICAL DISTINCTION from Exp2-LOD:
#   Exp2-LOD: O applied to value_emb (semantic space) + distance decay (w_ij)
#   Exp5:     O applied to legendre_pos (positional space) + NO distance decay
#
# CRITICAL DISTINCTION from Exp4-OrderOnly:
#   Exp4: O applied to value_emb (semantic) — no Label component
#   Exp5: O applied to legendre_pos (positional) — WITH Label component
#
# Components:
#   Label (L): Legendre polynomial embeddings — orthogonal position labels
#   Order (O): Uniform signed displacements in Legendre space
#              O_i = (1/L-1) * Σ_{j≠i} (P_i - P_j)
#   NO Distance decay (alpha=1, uniform weighting)
#   YES Temporal embedding
#
# Extra files needed: legendre_embedding.py AND ordering_operator.py
# WARNING: ordering_operator is O(L²) — expect ~50-100s per epoch
# No alpha sweep — fixed config
# Runs: 2 (pred_len 96, 192 × seed 2021)
# =============================================================================

PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp5_label_order"
LOG_DIR="$PROJECT_ROOT/logs/exp5_label_order_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

echo "============================================================" | tee "$MASTER_LOG"
echo "Experiment 5 — Phase 1: Label + Order (L + O)"              | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "Formula   : X'_i = X_i + T_i + P_i + O_i"                  | tee -a "$MASTER_LOG"
echo "  X_i = value embedding (semantic)"                          | tee -a "$MASTER_LOG"
echo "  T_i = temporal embedding"                                  | tee -a "$MASTER_LOG"
echo "  P_i = Legendre position label"                             | tee -a "$MASTER_LOG"
echo "  O_i = OrderingOperator(P_i) — order in POSITIONAL space"  | tee -a "$MASTER_LOG"
echo "        O_i = (1/L-1) * Σ_{j≠i} (P_i - P_j)"               | tee -a "$MASTER_LOG"
echo ""                                                             | tee -a "$MASTER_LOG"
echo "Key distinctions:"                                            | tee -a "$MASTER_LOG"
echo "  vs Exp2-LOD : No distance decay, O in positional not semantic" | tee -a "$MASTER_LOG"
echo "  vs Exp4-Only: Has Label (P_i), O in positional not semantic"   | tee -a "$MASTER_LOG"
echo "  vs Exp3b    : Has Order (O_i) on top of Label"            | tee -a "$MASTER_LOG"
echo ""                                                             | tee -a "$MASTER_LOG"
echo "WARNING: O(L²) operator — expect ~50-100s per epoch"        | tee -a "$MASTER_LOG"
echo "Seed      : 2021 | Pred lengths: 96, 192 | Total runs: 2"    | tee -a "$MASTER_LOG"
echo "Start: $(date)"                                               | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "Copying experiment files..." | tee -a "$MASTER_LOG"
cp "$EXP_DIR/models/__init__.py"           "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"               "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"              "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"            "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"            "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"              "$INFORMER_DIR/models/model.py"
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"
cp "$EXP_DIR/models/ordering_operator.py"  "$INFORMER_DIR/models/ordering_operator.py"
echo "Files copied successfully." | tee -a "$MASTER_LOG"

cd "$INFORMER_DIR" || { echo "ERROR: Cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

TOTAL=0; COMPLETED=0; FAILED=0; SKIPPED=0
SEED=2021

for PRED_LEN in 96 192; do
    TOTAL=$((TOTAL + 1))
    RUN_ID="exp5_ph1_ETTh1_label_order_pred${PRED_LEN}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    mkdir -p "$RESULTS_DIR/${RUN_ID}"

    echo "" | tee -a "$MASTER_LOG"
    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "RUN $TOTAL/2: $RUN_ID"                                        | tee -a "$MASTER_LOG"
    echo "pred_len=$PRED_LEN | seed=$SEED | config=label+order_positional" | tee -a "$MASTER_LOG"
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
echo "PHASE 1 RESULTS — Experiment 5 (Label + Order)"             | tee -a "$MASTER_LOG"
echo "TOTAL=$TOTAL | COMPLETED=$COMPLETED | FAILED=$FAILED | SKIPPED=$SKIPPED" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "PredLen | MSE      | MAE      | Status"                      | tee -a "$MASTER_LOG"
echo "--------|----------|----------|-------"                       | tee -a "$MASTER_LOG"

for PRED_LEN in 96 192; do
    RUN_ID="exp5_ph1_ETTh1_label_order_pred${PRED_LEN}_seed${SEED}"
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
echo "Experiment                    | pred=96 MSE | pred=192 MSE"  | tee -a "$MASTER_LOG"
echo "------------------------------|-------------|-------------"   | tee -a "$MASTER_LOG"
echo "Exp1-Pre a=1.0                | 0.8683      | 0.8463  ← best overall" | tee -a "$MASTER_LOG"
echo "Exp2-LOD (L+O+D full)         | 0.8242      | 0.9002"        | tee -a "$MASTER_LOG"
echo "Exp4b-OrderPositional (consec)| 0.8539      | 0.8541"        | tee -a "$MASTER_LOG"
echo "Exp3b-LabelTemporal (L only)  | 0.9265      | 0.8888"        | tee -a "$MASTER_LOG"
echo "Exp4-OrderEmbed (O semantic)  | 0.8220      | 0.9997"        | tee -a "$MASTER_LOG"
echo "Exp4a-OrderMentor (consec Δx) | 0.8613      | 1.0153"        | tee -a "$MASTER_LOG"
echo "Exp3-LabelOnly (no temporal)  | 1.0989      | 1.4844"        | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "DECISION GUIDE:"                                              | tee -a "$MASTER_LOG"
echo "  Exp5 wins at BOTH vs Exp1-Pre → L+O positional is best → Phase 2"    | tee -a "$MASTER_LOG"
echo "  Exp5 wins at BOTH vs Exp3b   → Order adds value to Label → Phase 2"  | tee -a "$MASTER_LOG"
echo "  Exp5 worse than Exp3b at both → Order hurts Label → document, stop"  | tee -a "$MASTER_LOG"
echo "  Exp5 mixed results           → Phase 2 to confirm"                   | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "End: $(date)" | tee -a "$MASTER_LOG"