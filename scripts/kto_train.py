#!/usr/bin/env python3
"""
KTO（Kahneman-Tversky Optimization）训练脚本
不需要成对数据，只需要二元标签（好/坏）
Mac 最友好：只需加载一个模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# 配置
model_name = "Qwen/Qwen2.5-3B-Instruct"
output_dir = "./adapters/kto-3b"
data_file = "./data/dpo.jsonl"  # 复用 DPO 数据，转换为 KTO 格式

print("=" * 50)
print("KTO 训练（Mac 最友好）")
print("=" * 50)

# 1. 加载模型
print(f"\n[1/3] 加载模型：{model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
print(f"  模型参数：{model.num_parameters()/1e9:.2f}B")

# 2. 加载并转换数据
print(f"\n[2/3] 加载并转换数据")

# 读取 DPO 数据，转换为 KTO 格式
# KTO 格式：{prompt, completion, label: true/false}
with open(data_file, 'r', encoding='utf-8') as f:
    dpo_data = [json.loads(line) for line in f if line.strip()]

kto_data = []
for item in dpo_data:
    # chosen 作为正样本（label=true）
    kto_data.append({
        "prompt": item["prompt"],
        "completion": item["chosen"],
        "label": True
    })
    # rejected 作为负样本（label=false）
    kto_data.append({
        "prompt": item["prompt"],
        "completion": item["rejected"],
        "label": False
    })

print(f"  DPO 数据量：{len(dpo_data)} 条")
print(f"  KTO 数据量：{len(kto_data)} 条（正负样本各半）")

# 3. KTO 训练
print(f"\n[3/3] 开始 KTO 训练")

# KTO 核心原理（来自前景理论）：
# - 人类对"损失"的敏感度高于"收益"
# - 模型应该同时学习"什么是好的"和"什么是坏的"
# - 对"坏的"惩罚更重

class KTOTrainer:
    def __init__(self, model, tokenizer, beta=0.1, lam=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.lam = lam  # 惩罚系数
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    def compute_log_prob(self, prompt, completion):
        """计算生成概率的对数"""
        full_text = f"用户：{prompt}\n助手：{completion}"
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 计算 cross entropy loss 的负数作为 log probability
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            inputs["input_ids"][:, 1:].contiguous().view(-1),
            reduction='mean'
        )
        return -loss.item()

    def train_step(self, prompt, completion, label):
        """单步训练"""
        self.model.train()

        log_prob = self.compute_log_prob(prompt, completion)

        # KTO loss（简化版）
        # 正样本：增大概率（log_prob 越大越好）
        # 负样本：减小概率（log_prob 越小越好）
        # 且对负样本惩罚更重（前景理论）

        if label:
            # 正样本：最大化 log_prob
            loss = -torch.tensor(log_prob, dtype=torch.float32, requires_grad=True)
        else:
            # 负样本：最小化 log_prob，且惩罚更重
            loss = self.lam * torch.tensor(log_prob, dtype=torch.float32, requires_grad=True)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), log_prob

    def train(self, epochs=2):
        """训练"""
        for epoch in range(epochs):
            total_loss = 0
            pos_correct = 0
            neg_correct = 0
            pos_count = 0
            neg_count = 0

            for i, item in enumerate(kto_data):
                loss, log_prob = self.train_step(
                    item["prompt"], item["completion"], item["label"]
                )

                total_loss += loss

                if item["label"]:
                    pos_count += 1
                    if log_prob > 0:  # 正样本概率应该高
                        pos_correct += 1
                else:
                    neg_count += 1
                    if log_prob < 0:  # 负样本概率应该低
                        neg_correct += 1

                if (i + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}, Step {i+1}/{len(kto_data)}: "
                          f"Loss={loss:.4f}, LogProb={log_prob:.4f}, "
                          f"Label={'True' if item['label'] else 'False'}")

            pos_acc = pos_correct / max(pos_count, 1)
            neg_acc = neg_correct / max(neg_count, 1)
            avg_loss = total_loss / len(kto_data)
            print(f"\nEpoch {epoch+1} 完成: Avg Loss={avg_loss:.4f}, "
                  f"Pos Accuracy={pos_acc:.2%}, Neg Accuracy={neg_acc:.2%}")

# 训练
trainer = KTOTrainer(model, tokenizer)
trainer.train(epochs=2)

# 保存模型
print(f"\n💾 保存模型到：{output_dir}")
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\n" + "=" * 50)
print("KTO 训练完成！")
print("=" * 50)
