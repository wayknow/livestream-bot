# 直播话术助手（Livestream Script Assistant）

基于 QLoRA + DPO 微调的直播带货话术生成器。输入产品信息，输出真实主播风格的带货文案。

**硬件要求**：24GB Mac Mini（M1/M2/M4 Pro）  
**模型**：Qwen2.5-7B-Instruct（SFT）+ Qwen2.5-3B-Instruct（DPO）

## Features

- **QLoRA SFT 微调**：用 mlx-lm 在本地微调 7B 模型，学习直播话术风格
- **DPO 偏好优化**：用 TRL 训练模型"拒绝平庸，选择激情"
- **本地部署**：融合 LoRA 权重后可直接本地推理，无需云端 GPU
- **完整闭环**：数据构造 → SFT → DPO → 部署，全流程可复现

## 目录结构

```
livestream-bot/
├── README.md           # 本文件
├── PRODUCT.md          # 产品描述与数据格式
├── STATUS.md           # 开发状态与进度
├── PROJECT_LOG.md      # 决策日志
├── data/               # 训练数据
│   ├── train.jsonl     # SFT 训练集
│   ├── valid.jsonl     # SFT 验证集
│   └── dpo.jsonl       # DPO 偏好对
├── adapters/           # LoRA 适配器权重
│   ├── sft-7b/         # SFT 训练后的 LoRA
│   └── dpo-3b/         # DPO 训练后的 LoRA
└── fused_model/        # 融合后的完整模型
    └── livestream-7b-sft/
```

## 快速开始

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate mlx

# 进入项目目录
cd /Users/xiaoxiao/work/finetuning/livestream-bot
```

### 2. Week 1：数据构造

```bash
# 运行数据构造脚本（待创建）
python scripts/generate_data.py
```

### 3. Week 2：SFT 微调

```bash
python -m mlx_lm.lora \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train \
    --data ./data \
    --batch-size 2 \
    --lora-layers 16 \
    --r 16 \
    --lora-alpha 32 \
    --iters 600 \
    --learning-rate 1e-5 \
    --adapter-path ./adapters/sft-7b
```

### 4. Week 3：DPO 优化

```bash
python scripts/dpo_train.py
```

### 5. Week 4：部署测试

```bash
python -m mlx_lm.generate \
    --model ./fused_model/livestream-7b-sft \
    --prompt "产品：面膜；卖点：玻尿酸、急救补水；生成话术" \
    --max-tokens 300
```

## 文档

- [产品描述与数据格式](PRODUCT.md)
- [开发状态与进度](STATUS.md)
- [项目决策日志](PROJECT_LOG.md)

## Tech Stack

- **训练框架**：mlx-lm（Apple Silicon 原生加速）
- **DPO 框架**：TRL + PyTorch MPS
- **基座模型**：Qwen2.5-7B-Instruct / Qwen2.5-3B-Instruct
- **硬件**：24GB unified memory，Metal GPU 加速
