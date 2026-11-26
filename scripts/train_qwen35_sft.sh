#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-4B LoRA SFT. Batch 1 is the verified setting for a 24GB M4
# (batch 2 runs out of Metal memory on this model).
#
#   conda activate mlx
#   bash scripts/train_qwen35_sft.sh

python -m mlx_lm lora -c config_qwen35_sft.yaml
