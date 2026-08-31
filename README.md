# 直播话术助手（Livestream Script Assistant）

基于 QLoRA + DPO 微调的直播带货话术生成器。输入产品信息，输出真实主播风格的带货文案。

**硬件要求**：24GB Mac Mini（M1/M2/M4 Pro）  
**模型**：Qwen2.5-7B-Instruct（SFT）+ Qwen2.5-3B-Instruct（DPO）

## Features

- **QLoRA SFT 微调**：用 mlx-lm 在本地微调 7B 模型，学习直播话术风格
- **DPO 偏好优化**：用 TRL 训练模型"拒绝平庸，选择激情"
- **评估体系**：8 个硬指标 + 固定 Benchmark，量化评估话术质量
- **本地部署**：融合 LoRA 权重后可直接本地推理，无需云端 GPU

## 目录结构

```
livestream-bot/
├── README.md               # 本文件
├── PRODUCT.md              # 产品描述与数据格式
├── STATUS.md               # 开发状态与进度
├── PROJECT_LOG.md          # 决策日志
├── REPORT.md               # 练手报告
├── config.yaml             # 训练配置
├── data/
│   ├── train.jsonl         # SFT 训练集（170 条）
│   ├── valid.jsonl         # SFT 验证集（30 条）
│   ├── dpo.jsonl           # DPO 偏好对（100 条）
│   └── benchmark_prompts.jsonl  # 固定评估测试集（30 条）
├── scripts/
│   ├── generate_data.py    # 数据生成脚本
│   ├── dpo_train.py        # DPO 训练脚本
│   └── eval_pipeline.py    # 评估脚本
├── adapters/
│   ├── sft-7b/             # SFT LoRA 权重
│   └── dpo-3b/             # DPO LoRA 权重
├── fused_model/
│   └── livestream-7b-sft/  # 融合后的完整模型
└── eval_results/           # 评估报告（版本化存储）
```

## 快速开始

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate mlx

# 进入项目目录
cd /Users/xiaoxiao/work/finetuning/livestream-bot
```

### 2. 生成话术

```bash
python -m mlx_lm generate \
    --model /Users/xiaoxiao/work/finetuning/livestream-bot/fused_model/livestream-7b-sft \
    --prompt "产品：充电宝；卖点：20000mAh、快充、轻便；生成1分钟话术" \
    --max-tokens 300 \
    --temp 0.7 \
    --top-p 0.9
```

### 3. 评估模型

```bash
# 评估基座模型
python scripts/eval_pipeline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --name "基座模型" \
    --save

# 评估 SFT 模型
python scripts/eval_pipeline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter ./adapters/sft-7b \
    --name "SFT模型" \
    --save

# 对比多个模型
python scripts/eval_pipeline.py \
    --compare eval_results/eval_基座模型_*.json eval_results/eval_SFT模型_*.json
```

## 文档

- [产品描述与数据格式](PRODUCT.md)
- [开发状态与进度](STATUS.md)
- [项目决策日志](PROJECT_LOG.md)
- [练手报告](REPORT.md)

## Tech Stack

- **训练框架**：mlx-lm（Apple Silicon 原生加速）
- **DPO 框架**：TRL + PyTorch MPS
- **评估体系**：规则引擎 + LLM 裁判
- **基座模型**：Qwen2.5-7B-Instruct / Qwen2.5-3B-Instruct
- **硬件**：24GB unified memory，Metal GPU 加速
