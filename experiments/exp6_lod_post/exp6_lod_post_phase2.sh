#!/bin/bash

# exp6_lod_post_phase2.sh
# Test across all prediction lengths and seeds using optimal decay_a=1.0

PRED_LENS=(48 96 192)
SEEDS=(2021 2022 2023)
ALPHA=1.0
ALPHA_STR="1p0"

for seed in "${SEEDS[@]}"; do
  for pred_len in "${PRED_LENS[@]}"; do
    echo "Running Phase 2: decay_a=${ALPHA}, pred_len=${pred_len}, seed=${seed}"
    
    python -u main_informer.py \
      --model informer --data ETTh1 --features M \
      --seq_len 96 --label_len 48 --pred_len $pred_len \
      --enc_in 7 --dec_in 7 --c_out 7 \
      --e_layers 2 --d_layers 1 --attn full --factor 5 \
      --des "exp6_lod_post_ph2_ETTh1_a${ALPHA_STR}_pred${pred_len}_seed${seed}" \
      --itr 1 --train_epochs 6 \
      --decay_a $ALPHA
  done
done
