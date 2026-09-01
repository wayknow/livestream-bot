#!/bin/bash
# RLHF 完整流程脚本

echo "=========================================="
echo "RLHF 完整流程（1.5B 模型 Demo）"
echo "=========================================="

# 激活环境
conda activate mlx

# 进入项目目录
cd /Users/xiaoxiao/work/finetuning/livestream-bot

echo ""
echo "步骤 1/3：训练 Reward Model"
echo "------------------------------------------"
python scripts/reward_model_train.py

echo ""
echo "步骤 2/3：PPO 训练"
echo "------------------------------------------"
python scripts/ppo_train.py

echo ""
echo "步骤 3/3：测试生成效果"
echo "------------------------------------------"
python -m mlx_lm generate \
    --model ./adapters/ppo-1.5b \
    --prompt '产品：充电宝；卖点：20000mAh、快充、轻便；生成1分钟话术' \
    --max-tokens 200 \
    --temp 0.7

echo ""
echo "=========================================="
echo "RLHF 流程完成！"
echo "=========================================="
