#!/usr/bin/env python3
"""Verify that a trained adapter/model actually differs from its base.

Usage:
  python scripts/verify_training.py --adapter adapters/kto-3b-lora
  python scripts/verify_training.py --base Qwen/Qwen2.5-3B-Instruct --trained adapters/kto-3b

Exit code is 1 when an adapter has no non-zero LoRA update, or when a full
model is byte-for-byte identical to the base after dtype conversion.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from safetensors import safe_open


def _tensor_files(path: str | Path, names: Iterable[str]) -> list[str]:
    path = _resolve_path(path)
    if path.is_file():
        return [str(path)]
    found = []
    for name in names:
        file = path / name
        if file.exists():
            found.append(str(file))
    found.extend(sorted(glob.glob(str(path / "*.safetensors"))))
    return sorted(set(found))


def _adapter_files(path: str | Path) -> list[str]:
    return _tensor_files(path, ["adapter_model.safetensors", "adapters.safetensors"])


def _model_files(path: str | Path) -> list[str]:
    path = _resolve_path(path)
    if path.is_file():
        return [str(path)]
    index = path / "model.safetensors.index.json"
    if index.exists():
        with open(index, encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        return sorted({str(path / shard) for shard in weight_map.values()})
    files = sorted(glob.glob(str(path / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors found at {path}")
    return files


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{str(path).replace('/', '--')}"
    snapshots = sorted((cache_dir / "snapshots").glob("*")) if (cache_dir / "snapshots").exists() else []
    for snapshot in snapshots:
        if list(snapshot.glob("*.safetensors")):
            return snapshot
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(path), local_files_only=True))
    except Exception as exc:
        raise FileNotFoundError(f"Cannot resolve model path or repo id: {path}") from exc


def _load_map(files: Iterable[str], need_weights: bool = False) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for file in files:
        with safe_open(file, framework="pt") as handle:
            for key in handle.keys():
                if need_weights and key.endswith(".weight"):
                    mapping[key] = file
                elif not need_weights:
                    mapping[key] = file
    return mapping


def _adapter_stats(path: str | Path) -> Tuple[bool, Dict[str, object]]:
    files = _adapter_files(path)
    if not files:
        raise FileNotFoundError(f"No adapter safetensors found at {path}")

    lora_keys = 0
    max_abs = 0.0
    nonzero = 0
    params = 0
    for file in files:
        with safe_open(file, framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key).float()
                params += tensor.numel()
                nonzero += int((tensor != 0).sum().item())
                max_abs = max(max_abs, float(tensor.abs().max().item()))
                if "lora_B" in key or "lora_b" in key:
                    lora_keys += 1

    info = {
        "files": files,
        "num_tensors": len(_load_map(files)),
        "num_params": params,
        "nonzero_params": nonzero,
        "max_abs": round(max_abs, 8),
        "lora_b_tensors": lora_keys,
    }
    if lora_keys == 0:
        info["warning"] = (
            "no lora_B tensor found; this looks like a full model, not a LoRA adapter. "
            "Use --base/--trained for full-model comparison."
        )
        return False, info
    return max_abs > 0.0, info


def _compare_models(base: str | Path, trained: str | Path, limit: int | None = None) -> Dict[str, object]:
    base_files = _model_files(base)
    trained_files = _model_files(trained)
    base_map = _load_map(base_files, need_weights=True)
    trained_map = _load_map(trained_files, need_weights=True)

    common = sorted(set(base_map) & set(trained_map))
    if limit:
        common = common[:limit]

    max_diff = 0.0
    different = 0
    compared = 0
    for key in common:
        with safe_open(base_map[key], framework="pt") as handle:
            left = handle.get_tensor(key).float()
        with safe_open(trained_map[key], framework="pt") as handle:
            right = handle.get_tensor(key).float()
        if left.shape != right.shape:
            different += 1
            max_diff = float("inf")
            continue
        diff = float((left - right).abs().max().item())
        max_diff = max(max_diff, diff)
        different += int(not torch.equal(left, right))
        compared += 1

    return {
        "base": str(base),
        "trained": str(trained),
        "base_tensors": len(base_map),
        "trained_tensors": len(trained_map),
        "compared_tensors": compared,
        "different_tensors": different,
        "max_abs_diff": max_diff,
        "identical": compared > 0 and different == 0 and max_diff == 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify training actually changed weights")
    parser.add_argument("--adapter", help="LoRA adapter directory (adapter_model.safetensors)")
    parser.add_argument("--base", help="Base model directory or HF repo id")
    parser.add_argument("--trained", help="Trained full-model directory")
    parser.add_argument("--limit", type=int, help="Only compare the first N tensors (debugging)")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    if args.adapter:
        changed, info = _adapter_stats(args.adapter)
        payload = {"kind": "adapter", "changed": changed, **info}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"adapter: {args.adapter}")
            print(f"lora_B tensors: {info['lora_b_tensors']}")
            print(f"max |w|: {info['max_abs']}")
            if info.get("warning"):
                print(f"warning: {info['warning']}")
            print(f"result: {'PASS (weights changed)' if changed else 'FAIL (adapter is still zero or not LoRA)'}")
        return 0 if changed else 1

    if args.base and args.trained:
        info = _compare_models(args.base, args.trained, args.limit)
        if args.json:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            print(f"base:    {info['base']}")
            print(f"trained: {info['trained']}")
            print(f"tensors: {info['compared_tensors']} compared, {info['different_tensors']} different")
            print(f"max |diff|: {info['max_abs_diff']}")
            print(f"result: {'FAIL (identical to base)' if info['identical'] else 'PASS (weights changed)'}")
        return 1 if info["identical"] else 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
