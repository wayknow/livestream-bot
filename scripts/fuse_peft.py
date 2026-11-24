#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a full HF model for MLX inference.

TRL saves PEFT-format adapters (adapter_model.safetensors with lora_A/lora_B).
mlx-lm expects its own adapter layout, so evaluation/deployment uses a merged
full model instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into a full model")
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    manifest = {
        "base": args.base,
        "adapter": args.adapter,
        "output": str(output),
        "dtype": args.dtype,
        "num_parameters": int(model.num_parameters()),
    }
    (output / "fuse_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
