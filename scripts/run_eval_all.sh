#!/usr/bin/env bash
set -euo pipefail

# Evaluate every model on the same 30 prompts, same prompt version and same
# generation settings. Run from the repository root:
#   conda activate mlx
#   SAMPLES=30 bash scripts/run_eval_all.sh

SAMPLES="${SAMPLES:-30}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-300}"

run_eval() {
  local name="$1"
  local model="$2"
  local adapter="${3:-}"
  local cmd=(python scripts/eval_pipeline.py --model "$model" --name "$name"
    --samples "$SAMPLES" --temp "$TEMP" --top-p "$TOP_P"
    --max-tokens "$MAX_TOKENS" --save)
  if [[ -n "$adapter" ]]; then
    if [[ ! -d "$adapter" ]]; then
      echo "SKIP $name: adapter not found: $adapter"
      return
    fi
    cmd+=(--adapter "$adapter")
  fi
  echo "=== $name ==="
  "${cmd[@]}"
}

run_eval "base-3b" "Qwen/Qwen2.5-3B-Instruct"
run_eval "base-7b" "Qwen/Qwen2.5-7B-Instruct"
run_eval "sft-7b" "Qwen/Qwen2.5-7B-Instruct" "adapters/sft-7b"
run_eval "dpo-3b" "fused_model/dpo-3b"
run_eval "kto-3b" "fused_model/kto-3b"
run_eval "ipo-3b" "fused_model/ipo-3b"
