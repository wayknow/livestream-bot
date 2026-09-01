#!/usr/bin/env python3
"""
PPO 训练脚本（简化版 Demo）
展示 RLHF 的基本流程，不做真正的梯度更新
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# 配置
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
data_file = "./data/benchmark_prompts.jsonl"

print("=" * 50)
print("PPO 训练（简化版 Demo）")
print("=" * 50)

# 1. 加载模型
print(f"\n[1/3] 加载模型：{model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
model.eval()
print(f"  模型参数：{model.num_parameters()/1e9:.2f}B")

# 2. 加载测试数据
print(f"\n[2/3] 加载测试数据：{data_file}")
with open(data_file, 'r', encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f if line.strip()]

# 取前 3 条测试
test_data = test_data[:3]
print(f"  测试数据量：{len(test_data)} 条")

# 3. 生成并评估
print(f"\n[3/3] 生成并评估")

SYSTEM_PROMPT = """你是一位抖音直播带货话术专家。你的风格特征：
①短句为主，每句不超过15字
②每3句话插入一次互动指令
③先讲痛点场景，再给解决方案，最后价格逼单
④情绪递进：平静→兴奋→急促"""

results = []

for step, item in enumerate(test_data):
    prompt = item["prompt"]
    full_prompt = f"{SYSTEM_PROMPT}\n\n用户：{prompt}\n助手："

    # 生成回答
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 计算奖励（简单规则：包含互动词加分，句子短加分）
    reward = 0
    interactive_words = ["扣1", "对不对", "有没有", "是不是", "听我说"]
    for word in interactive_words:
        if word in response:
            reward += 1

    # 句子长度奖励
    sentences = [s for s in response.split("！") if s.strip()]
    avg_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
    if avg_len < 15:
        reward += 2

    results.append({
        "prompt": prompt,
        "response": response,
        "reward": reward
    })

    print(f"\n  样本 {step+1}/{len(test_data)}:")
    print(f"    Prompt: {prompt[:60]}...")
    print(f"    Response: {response[:150]}...")
    print(f"    Reward: {reward}")

# 统计结果
print(f"\n{'='*50}")
print("结果统计")
print(f"{'='*50}")
avg_reward = sum(r["reward"] for r in results) / len(results)
print(f"  平均奖励：{avg_reward:.2f}")
print(f"  最高奖励：{max(r['reward'] for r in results)}")
print(f"  最低奖励：{min(r['reward'] for r in results)}")

print(f"\n{'='*50}")
print("RLHF Demo 完成！")
print(f"{'='*50}")
print("\n说明：这是一个简化的 RLHF demo，展示了基本流程。")
print("真正的 RLHF 需要：")
print("1. 训练 Reward Model（需要 GPU 和更多数据）")
print("2. PPO 训练（需要同时加载 4 个模型）")
print("3. 更多的计算资源和调参")
