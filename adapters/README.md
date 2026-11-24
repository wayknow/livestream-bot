# Adapter audit

`kto-3b/` and `ipo-3b/` are **invalid** artifacts. Their `model.safetensors`
files are byte-for-byte identical to the Qwen2.5-3B-Instruct base weights after
upcasting to float32; the old training scripts detached gradients and never
updated the model. Do not use them for evaluation or as evidence of preference
training.

Real adapters produced by the repaired scripts live in:

- `kto-3b-lora/` (TRL `KTOTrainer`, LoRA)
- `ipo-3b-lora/` (TRL `DPOTrainer` with `loss_type=["ipo"]`, LoRA)

Verify any adapter with:

```bash
python scripts/verify_training.py --adapter adapters/kto-3b-lora
```

`kto-3b/` and `ipo-3b/` can be deleted to reclaim about 24 GB.
