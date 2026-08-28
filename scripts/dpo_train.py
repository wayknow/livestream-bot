#!/usr/bin/env python3
"""
DPO 偏好优化训练脚本
使用 TRL 在 Mac MPS 上训练
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer
from datasets import load_dataset
import os

# 配置
model_name = "Qwen/Qwen2.5-3B-Instruct"
output_dir = "./adapters/dpo-3b"
data_file = "./data/dpo.jsonl"

print("=" * 50)
print("DPO 偏好优化训练")
print("=" * 50)

# 1. 检查设备
# MPS 在大模型 DPO 训练时可能有内存问题，使用 CPU 更稳定
device = "cpu"
print(f"ℹ️ 使用 CPU 训练（MPS 对 DPO 不稳定）")

# 2. 加载模型和 tokenizer
print(f"\n[1/4] 加载模型：{model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 确保有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Mac 用 float32
    device_map=device,
    trust_remote_code=True
)
print(f"  模型参数：{model.num_parameters()/1e9:.2f}B")

# 3. 配置 LoRA（不在此处应用，让 DPOTrainer 自己应用）
print(f"\n[2/4] 配置 LoRA")
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
print(f"  LoRA 配置完成（r=8, alpha=16）")

# 4. 加载数据
print(f"\n[3/4] 加载数据：{data_file}")
dataset = load_dataset("json", data_files=data_file, split="train")
print(f"  数据量：{len(dataset)} 条")
print(f"  样例：{dataset[0]['prompt'][:50]}...")

# 5. 训练参数
print(f"\n[4/4] 开始训练")
training_args = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,      # 24GB 用 batch=1
    gradient_accumulation_steps=4,      # 等效 batch_size=4
    num_train_epochs=3,                 # 3 轮
    learning_rate=5e-6,                 # DPO 学习率要低
    beta=0.1,                           # DPO 温度参数
    max_length=512,
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    fp16=False,                         # Mac 不支持 fp16
    bf16=False,
    remove_unused_columns=False,
    report_to="none",                   # 不上报到 wandb
    dataloader_num_workers=0,           # Mac 上用 0 更稳定
    seed=42,
)

# 6. 创建 DPO Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 用 peft 时自动创建
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,  # TRL 1.12+ 用 processing_class
    peft_config=peft_config,
)

# 7. 开始训练
print("\n🚀 开始 DPO 训练...")
train_result = trainer.train()

# 8. 保存模型
print(f"\n💾 保存模型到：{output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# 9. 打印结果
print("\n" + "=" * 50)
print("训练完成！")
print(f"训练损失：{train_result.training_loss:.4f}")
print(f"训练时长：{train_result.metrics['train_runtime']:.1f} 秒")
print(f"保存路径：{output_dir}")
print("=" * 50)
