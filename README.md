# 直播话术助手（Livestream Script Assistant）

基于 QLoRA SFT + 偏好优化的直播带货话术生成实验。输入产品信息，输出主播风格的带货文案。

**硬件要求**：24GB Mac Mini（Apple Silicon）<br>
**正式评估**：Qwen2.5 六个模型 + Qwen3.5-4B/9B，同 benchmark、同生成参数<br>
**当前最好**：`base-7b` 5/9；Qwen3.5-4B/9B 均为 4/9，偏好优化 3B 均为 4/9

> **审计说明**：旧版“KTO 8/8 指标达标”的结论已撤销。旧 KTO/IPO 权重与基座逐元素一致，旧评估数字由多次报告拼接而成。2026-11-24 的 180-token v2 结果也因全部输出被截断而作废；当前结论只使用 2026-11-25 的 300-token 报告。详见 [AUDIT.md](AUDIT.md) 与 [V2_RESULTS.md](eval_results/V2_RESULTS.md)。

## 当前状态

| 环节 | 状态 | 说明 |
|---|---|---|
| 数据 | ✅ | 170 条 SFT train、30 条 valid、100 条 DPO 偏好对 |
| SFT 7B LoRA | ⚠️ | 训练真实发生；v2 4/9，未优于 base-7b 的 5/9 |
| 融合模型 | ✅ | `fused_model/livestream-7b-sft` |
| DPO 3B LoRA | ✅ | 已训练并融合；v2 4/9，幻觉率 0.036 为全组最低 |
| KTO 3B | ✅ | 已用 TRL 重训并融合；v2 4/9 |
| IPO 3B | ✅ | 已用 TRL 重训并融合；v2 4/9 |
| Qwen3.5-4B | ✅ | 本地 bf16；v2 4/9，互动 0.392 达标 |
| Qwen3.5-9B | ✅ | 本地 bf16；v2 4/9，互动 0.483 达标但幻觉 0.281 |
| 统一评估 v2 | ✅ | 8 个模型；base-7b 5/9 最高，其余 3B/Qwen3.5 为 3–4/9 |

## 目录结构

```text
livestream-bot/
├── README.md
├── AUDIT.md                 # 审计与修复记录（先读这个）
├── PRODUCT.md
├── STATUS.md
├── PROJECT_LOG.md
├── PREFERENCE_METHODS.md
├── REPORT.md
├── config.yaml
├── data/
│   ├── train.jsonl          # 170 条 SFT 训练集
│   ├── valid.jsonl          # 30 条 SFT 验证集
│   ├── dpo.jsonl            # 100 条偏好对
│   └── benchmark_prompts.jsonl
├── scripts/
│   ├── generate_data.py
│   ├── dpo_train.py
│   ├── kto_train.py         # TRL KTOTrainer + 权重校验
│   ├── ipo_train.py         # TRL DPOTrainer(loss_type=ipo) + 权重校验
│   ├── verify_training.py   # 检测 adapter/模型是否真的变化
│   ├── fuse_peft.py         # PEFT LoRA → 完整模型，供 mlx-lm 推理
│   ├── eval_pipeline.py     # 评估协议 v2
│   ├── test_eval_metrics.py
│   ├── run_eval_all.sh
│   └── train_preferences.sh
├── adapters/
│   ├── README.md            # 旧 KTO/IPO 产物为何无效
│   ├── sft-7b/
│   ├── dpo-3b/
│   ├── kto-3b-lora/
│   └── ipo-3b-lora/
├── fused_model/
│   ├── livestream-7b-sft/
│   ├── dpo-3b/
│   ├── kto-3b/
│   └── ipo-3b/
└── eval_results/            # v1 仅历史；V2_RESULTS.md 是当前结论
```

## 快速开始

```bash
conda activate mlx
cd /Users/xiaoxiao/work/finetuning/livestream-bot
```

### 生成话术（示例：SFT 7B，v2 4/9）

```bash
python -m mlx_lm generate \
  --model ./fused_model/livestream-7b-sft \
  --system-prompt '你是一位抖音直播带货话术专家。你的风格特征：①短句为主；②每3句话插入一次互动指令；③先讲痛点场景，再给解决方案，最后价格逼单；④情绪递进：平静→兴奋→急促；⑤口头禅：家人们、真的绝了、最后一波。' \
  --prompt '产品：充电宝；卖点：20000mAh、快充、轻便；时长：1分钟' \
  --max-tokens 300 \
  --temp 0.7 \
  --top-p 0.9
```

### 重新训练偏好优化（可选）

```bash
# KTO + IPO，默认 Qwen2.5-3B-Instruct + LoRA（24GB M4 已验证配置）
DEVICE=mps DTYPE=bfloat16 EPOCHS=1 MAX_LENGTH=128 bash scripts/train_preferences.sh

# 训练后必须通过权重校验；仍为零会直接报错
python scripts/verify_training.py --adapter adapters/kto-3b-lora
python scripts/verify_training.py --adapter adapters/ipo-3b-lora
```

### 统一评估

```bash
# 指标回归测试（不需要 Metal）
python scripts/test_eval_metrics.py

# 同一 prompt、同一 30 条 benchmark、同一采样参数
# QWEN35_ROOT 下存在 Qwen3.5-4B / Qwen3.5-9B 时会一并评估
QWEN35_ROOT=/Users/xiaoxiao/work/finetuning/models SAMPLES=30 bash scripts/run_eval_all.sh

# 只对比同 metric/prompt/benchmark/生成参数的报告；不一致会打印警告
python scripts/eval_pipeline.py --compare eval_results/eval_v2_*.json
```

## 评估协议 v2

| 指标 | 定义 | 目标 |
|---|---|---|
| 互动句占比 | 含互动词的句子数 / 总句数 | 0.30–0.60 |
| 平均句长 | 总字数 / 总句数 | 15–25 字 |
| 口语词占比 | 口语词出现次数 / 总字数 | ≥0.02 |
| 结构完整度 | 痛点/方案/价格/限时/行动五维命中数 | ≥4 |
| 卖点覆盖率 | 命中的 prompt 卖点数 / 总卖点数 | ≥0.80 |
| 幻觉率 | 输出中未在 prompt 出现的数字 / 输出数字总数 | ≤0.10 |
| 情绪递进 | 平静→兴奋→急促的真实递进程度 | ≥3.5 |
| 口头禅重复度 | 最高频口头禅次数 / 总句数 | ≤0.30 |
| 流畅度 | 默认规则评分；传 `--judge-model` 才用 LLM | ≥3.5 |

注意：当前 system prompt 写的是“每句不超过 15 字”，而平均句长目标是 15–25 字，两者本身冲突。句长结果只能用于相对比较，目标值需要下一轮重新定义。

每份 v2 报告都会记录 `metric_version`、`prompt_version`、`git_commit`、`git_dirty`、`evaluator_sha256`、`benchmark_sha256`、模型路径、adapter 路径和生成参数。不要跨版本或跨生成参数拼数字。

## 当前 v2 结果（30 条，max_tokens=300，Qwen chat template）

| 模型 | 目标命中 | 关键问题 |
|---|---:|---|
| base-3b | 3/9 | 互动 0.749、结构 2.767、卖点 0.700、幻觉 0.117 |
| kto-3b | 4/9 | 卖点 0.828，但结构 3.267、情绪 2.860、幻觉 0.150 |
| ipo-3b | 4/9 | 卖点 0.839，但结构 3.067、情绪 3.167、幻觉 0.116 |
| dpo-3b | 4/9 | 幻觉 0.036 最好，但结构 2.600、情绪 2.570、卖点 0.767 |
| base-7b | **5/9** | 最高命中；幻觉 0.284、结构 3.167、情绪 2.900 |
| sft-7b | 4/9 | 互动 0.354 达标，但句长 7.18、结构 2.233、幻觉 0.320 |
| qwen3.5-4b | 4/9 | 互动 0.392 达标；卖点 0.733、幻觉 0.152、结构 3.200 |
| qwen3.5-9b | 4/9 | 互动 0.483 最好；但卖点 0.728、幻觉 0.281 |

详细表格和 caveat 见 [eval_results/V2_RESULTS.md](eval_results/V2_RESULTS.md)。结论：**没有任何模型接近 8/9；base-7b 仍是当前最强基线，Qwen3.5-4B/9B 没有解决结构、卖点、幻觉和情绪问题。旧 8/8 不可复现。**

## 文档

- [审计与修复记录](AUDIT.md)
- [产品描述与数据格式](PRODUCT.md)
- [开发状态](STATUS.md)
- [项目决策日志](PROJECT_LOG.md)
- [偏好优化方法](PREFERENCE_METHODS.md)
- [项目报告](REPORT.md)

## Tech Stack

- 训练：mlx-lm（SFT）、TRL + PyTorch（DPO/KTO/IPO）
- 基座：Qwen2.5-7B-Instruct、Qwen2.5-3B-Instruct
- 评估：规则引擎 + 可选本地 LLM 流畅度裁判
- 硬件：Mac Mini M4，24GB unified memory
