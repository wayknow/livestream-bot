#!/usr/bin/env bash
set -euo pipefail

# Compare the untouched Qwen3.5-4B baseline with the sft-v4 LoRA adapter on
# the same benchmark, prompt version, seed and generation parameters.
#
#   conda activate mlx
#   SEED=0 SAMPLES=30 bash scripts/eval_qwen35_pair.sh

SEED="${SEED:-0}"
SAMPLES="${SAMPLES:-30}"
MAX_TOKENS="${MAX_TOKENS:-300}"
MODEL="${MODEL:-$HOME/work/finetuning/models/Qwen3.5-4B}"
ADAPTER="${ADAPTER:-adapters/qwen3.5-4b-sft-v4}"

python scripts/eval_pipeline.py \
  --model "$MODEL" \
  --name "qwen3.5-4b-base-v3-s${SEED}" \
  --samples "$SAMPLES" --max-tokens "$MAX_TOKENS" --seed "$SEED" --save

python scripts/eval_pipeline.py \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --name "qwen3.5-4b-sft-v3-s${SEED}" \
  --samples "$SAMPLES" --max-tokens "$MAX_TOKENS" --seed "$SEED" --save

python scripts/eval_pipeline.py --compare $(ls -t eval_results/eval_v3_qwen3.5-4b-{base,sft}-v3-s${SEED}_*.json | head -2)
