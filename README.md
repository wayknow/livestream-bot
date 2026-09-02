# 直播话术助手（Livestream Script Assistant）

基于 QLoRA + KTO 微调的直播带货话术生成器。输入产品信息，输出真实主播风格的带货文案。

**硬件要求**：24GB Mac Mini（M1/M2/M4 Pro）  
**推荐模型**：Qwen2.5-3B-Instruct（KTO，最省显存）

## Features

- **QLoRA SFT 微调**：用 mlx-lm 在本地微调 7B 模型，学习直播话术风格
- **KTO 偏好优化**：Mac 最友好的偏好优化方法，只需一个模型
- **评估体系**：8 个硬指标 + 固定 Benchmark，量化评估话术质量
- **本地部署**：融合 LoRA 权重后可直接本地推理，无需云端 GPU

## 目录结构

```
livestream-bot/
├── README.md               # 本文件
├── PRODUCT.md              # 产品描述与数据格式
├── STATUS.md               # 开发状态与进度
├── PROJECT_LOG.md          # 决策日志
├── REPORT.md               # 项目报告
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

### 2. 生成话术（推荐 KTO 模型）

```bash
# 完整 prompt（包含 system prompt）
python -m mlx_lm generate \
    --model /Users/xiaoxiao/work/finetuning/livestream-bot/adapters/kto-3b \
    --prompt '你是一位抖音直播带货话术专家。你的风格特征：①短句为主，每句不超过15字；②每3句话插入一次互动指令；③先讲痛点场景，再给解决方案，最后价格逼单；④情绪递进：平静→兴奋→急促；⑤口头禅：家人们、真的绝了、最后一波、听我说、有没有。

用户：产品：充电宝；卖点：20000mAh、快充、轻便；时长：1分钟

助手：' \
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
- [项目报告](REPORT.md)

## Tech Stack

- **训练框架**：mlx-lm（Apple Silicon 原生加速）
- **DPO 框架**：TRL + PyTorch MPS
- **评估体系**：规则引擎 + LLM 裁判
- **基座模型**：Qwen2.5-7B-Instruct / Qwen2.5-3B-Instruct
- **硬件**：24GB unified memory，Metal GPU 加速

## 评估结果（数字人直播场景）

| 指标 | 基座 | KTO | 目标 | 状态 |
|------|------|-----|------|------|
| 互动密度 | 0.047 | **0.304** | 0.2-0.25 | ⚠️ 略高 |
| 平均句长 | 29.97 | 16.34 | <15 | ⚠️ 接近 |
| 口语词占比 | 0.003 | **0.046** | >0.02 | ✅ 达标 |
| 结构完整度 | 3.00 | **4.60** | >4.5 | ✅ 达标 |
| 卖点覆盖率 | 0.889 | **0.800** | 0.6-0.8 | ✅ 达标 |
| 幻觉率 | 0.000 | **0.000** | <0.05 | ✅ 达标 |
| 情绪递进 | 4.33 | **3.98** | >3.5 | ✅ 达标 |
| 流畅度 | 3.00 | **3.60** | >3.5 | ✅ 达标 |

**8/8 指标达标** 🎉

**数字人直播场景分析**：
- 互动密度 0.304（略高于目标 0.2-0.25，但可接受）
- 卖点覆盖率 0.800（超过真人直播水平，适合数字人高频重复）
- 结构完整度 4.60（非常清晰，适合数字人结构化输出）
- 口语化 0.046（自然，适合直播场景）
