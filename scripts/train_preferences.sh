#!/usr/bin/env bash
set -euo pipefail

# Train KTO and IPO adapters with TRL. Both scripts abort if the saved adapter
# is still zero (i.e. if training did nothing).
#   conda activate mlx
#   bash scripts/train_preferences.sh

# Defaults match the verified 3B run on a 24GB M4 (MPS + bfloat16).
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
EPOCHS="${EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-128}"

python scripts/kto_train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --epochs "$EPOCHS" \
  --max-length "$MAX_LENGTH" \
  --output adapters/kto-3b-lora

python scripts/ipo_train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --epochs "$EPOCHS" \
  --max-length "$MAX_LENGTH" \
  --output adapters/ipo-3b-lora
