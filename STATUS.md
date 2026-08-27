# 直播话术助手 — 开发状态（2026-08-27）

## 一句话概述

基于 QLoRA + DPO 的直播带货话术生成器，24GB Mac Mini 本地可跑。**当前进度：环境已就绪，进入 Week 1 数据构造阶段。**

---

## 当前进度

| 阶段 | 状态 | 产出 |
|------|:----:|------|
| 环境准备 | ✅ | Miniconda + Python 3.11 + mlx-lm/trl/peft |
| Week 1：数据构造 | ⏳ | 待生成 train.jsonl / dpo.jsonl |
| Week 2：SFT 微调 | ⏳ | 待训练 adapters/sft-7b |
| Week 3：DPO 优化 | ⏳ | 待训练 adapters/dpo-3b |
| Week 4：部署测试 | ⏳ | 待融合 fused_model/livestream-7b-sft |

---

## 环境状态

| 组件 | 版本 | 状态 |
|------|------|:----:|
| Miniconda | 26.7.1 | ✅ |
| Python | 3.11.15 | ✅ |
| mlx-lm | 0.31.3 | ✅ |
| transformers | 5.16.1 | ✅ |
| peft | 0.20.0 | ✅ |
| trl | 1.12.0 | ✅ |
| datasets | 5.0.1 | ✅ |
| accelerate | 1.14.0 | ✅ |

**环境激活命令**：
```bash
conda activate mlx
cd /Users/xiaoxiao/work/finetuning/livestream-bot
```

---

## 目录结构

```
livestream-bot/
├── README.md
├── PRODUCT.md
├── STATUS.md
├── PROJECT_LOG.md
├── data/               # 空，待 Week 1 填充
├── adapters/           # 空，待 Week 2/3 训练
└── fused_model/        # 空，待 Week 4 融合
```

---

## 四周计划

### Week 1（当前）：数据构造

**目标**：200 条 SFT 数据 + 100 条 DPO 偏好对

| 任务 | 状态 |
|------|:----:|
| 手写 50 条核心 SFT 样本（10 品类 × 5 风格） | ⏳ |
| 编写模板扩充脚本，自动扩到 200 条 | ⏳ |
| 构造 50-100 条 DPO 偏好对 | ⏳ |
| 数据质量检查 + 格式验证 | ⏳ |

### Week 2：QLoRA SFT

**目标**：用 mlx-lm 训练 7B 模型

| 任务 | 状态 |
|------|:----:|
| 下载 Qwen2.5-7B-Instruct | ⏳ |
| 运行 lora 训练，观察 loss 曲线 | ⏳ |
| 测试生成效果，与基座对比 | ⏳ |
| 调参（如 loss 不收敛） | ⏳ |

### Week 3：DPO

**目标**：用 TRL 训练偏好模型

| 任务 | 状态 |
|------|:----:|
| 下载 Qwen2.5-3B-Instruct | ⏳ |
| 运行 DPO 训练 | ⏳ |
| 对比 chosen/reward 差距 | ⏳ |
| 测试"拒绝平淡"效果 | ⏳ |

### Week 4：部署

**目标**：融合模型 + 本地推理

| 任务 | 状态 |
|------|:----:|
| 融合 SFT LoRA 到基座 | ⏳ |
| 生成对比报告（基座 vs SFT vs DPO） | ⏳ |
| 编写练手总结 | ⏳ |

---

## 下一步

1. **立即开始**：Week 1 数据构造
2. **需要决策**：是否先写数据扩充脚本，还是手动写 50 条样本？
