#!/usr/bin/env python3
"""KTO training with TRL's real KTOTrainer.

The previous implementation computed log-probabilities under ``torch.no_grad()``
and wrapped the resulting float in a new tensor, so the optimizer never updated
the model. This version uses TRL's KTOTrainer with LoRA, which requires a
reference policy by construction and back-propagates through the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_training import _adapter_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KTO preference training (TRL)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default=str(ROOT / "data" / "dpo.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "adapters" / "kto-3b-lora"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--limit", type=int, help="Limit number of preference pairs (debug/smoke)")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def load_kto_dataset(path: str, limit: int | None = None) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if limit and len(rows) >= limit * 2:
                break
            rows.append({"prompt": item["prompt"], "completion": item["chosen"], "label": True})
            rows.append({"prompt": item["prompt"], "completion": item["rejected"], "label": False})
    if not rows:
        raise ValueError(f"No KTO rows loaded from {path}")
    desirable = sum(1 for row in rows if row["label"])
    undesirable = len(rows) - desirable
    print(f"KTO rows: {len(rows)} (desirable={desirable}, undesirable={undesirable})")
    return Dataset.from_list(rows)


def resolve_device(requested: str) -> tuple[bool, str]:
    if requested == "cpu":
        return True, "cpu"
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS requested but not available")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return False, "mps"
    return (not mps_available), "mps" if mps_available else "cpu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.batch_size * args.grad_accum <= 1:
        raise ValueError("KTO requires effective batch size > 1 (batch_size * grad_accum)")
    use_cpu, resolved_device = resolve_device(args.device)
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print("=" * 60)
    print("KTO training (TRL KTOTrainer)")
    print(f"model={args.model}")
    print(f"device={resolved_device} dtype={args.dtype} use_cpu={use_cpu}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    dataset = load_kto_dataset(args.data, args.limit)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = KTOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type="kto",
        desirable_weight=1.0,
        undesirable_weight=1.0,
        max_length=args.max_length,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        bf16=args.dtype == "bfloat16",
        fp16=False,
        use_cpu=use_cpu,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=42,
    )

    trainer = KTOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    adapter_file = Path(args.output) / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"TRL did not save an adapter at {adapter_file}")

    changed, stats = _adapter_stats(args.output)
    manifest = {
        "method": "kto",
        "model": args.model,
        "output": args.output,
        "device": resolved_device,
        "dtype": args.dtype,
        "rows": len(dataset),
        "global_step": trainer.state.global_step,
        "training_loss": getattr(train_result, "training_loss", None),
        "adapter_sha256": sha256(adapter_file),
        "adapter_changed": changed,
        "adapter_stats": stats,
        "hyperparameters": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "beta": args.beta,
        },
    }
    manifest_path = Path(args.output) / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"adapter: {adapter_file}")
    print(f"lora_B tensors: {stats['lora_b_tensors']}, max |w|: {stats['max_abs']}")
    print(f"manifest: {manifest_path}")
    if not args.no_verify and not changed:
        raise RuntimeError("Training did not change the adapter; refusing to report success")
    print("PASS: KTO adapter weights changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
