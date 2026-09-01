#!/usr/bin/env python3
"""
Reward Model 训练脚本（简化版）
使用 1.5B 模型训练奖励模型
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os

# 配置
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "./adapters/reward-model"
data_file = "./data/dpo.jsonl"

print("=" * 50)
print("Reward Model 训练（简化版）")
print("=" * 50)

# 1. 加载模型和 tokenizer
print(f"\n[1/3] 加载模型：{model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
print(f"  模型参数：{model.num_parameters()/1e9:.2f}B")

# 2. 加载数据
print(f"\n[2/3] 加载数据：{data_file}")
with open(data_file, 'r', encoding='utf-8') as f:
    dpo_data = [json.loads(line) for line in f if line.strip()]

print(f"  数据量：{len(dpo_data)} 条")

# 3. 训练（简化版：只计算分数，不做梯度更新）
print(f"\n[3/3] 计算奖励分数")

model.eval()
rewards = []

for i, item in enumerate(dpo_data):
    # 计算 chosen 的分数
    chosen_text = f"用户：{item['prompt']}\n助手：{item['chosen']}"
    inputs = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        chosen_reward = model(**inputs).logits.item()

    # 计算 rejected 的分数
    rejected_text = f"用户：{item['prompt']}\n助手：{item['rejected']}"
    inputs = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        rejected_reward = model(**inputs).logits.item()

    rewards.append({
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "correct": chosen_reward > rejected_reward
    })

    if (i + 1) % 10 == 0:
        correct_count = sum(1 for r in rewards if r["correct"])
        accuracy = correct_count / len(rewards)
        print(f"  样本 {i+1}/{len(dpo_data)}: "
              f"Chosen={chosen_reward:.4f}, Rejected={rejected_reward:.4f}, "
              f"Accuracy={accuracy:.2%}")

# 统计结果
correct_count = sum(1 for r in rewards if r["correct"])
accuracy = correct_count / len(rewards)
avg_chosen = sum(r["chosen_reward"] for r in rewards) / len(rewards)
avg_rejected = sum(r["rejected_reward"] for r in rewards) / len(rewards)

print(f"\n结果统计：")
print(f"  准确率：{accuracy:.2%} ({correct_count}/{len(rewards)})")
print(f"  平均 chosen 分数：{avg_chosen:.4f}")
print(f"  平均 rejected 分数：{avg_rejected:.4f}")
print(f"  分数差：{avg_chosen - avg_rejected:.4f}")

print("\n" + "=" * 50)
print("Reward Model 评估完成！")
print("=" * 50)
