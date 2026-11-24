# 直播话术助手（Livestream Script Assistant）

基于 QLoRA SFT + 偏好优化的直播带货话术生成实验。输入产品信息，输出主播风格的带货文案。

**硬件要求**：24GB Mac Mini（Apple Silicon）<br>
**已验证模型**：Qwen2.5-7B-Instruct + SFT LoRA<br>
**偏好优化**：DPO/KTO/IPO 均已训练；3B 已完成 v2 评估（KTO 3/9、IPO 5/9、base 4/9）

> **2026-11-24 审计说明**：旧版“KTO 8/8 指标达标”的结论已撤销。旧 KTO/IPO 权重与基座逐元素一致，旧评估数字由多次报告拼接而成。完整证据和修复记录见 [AUDIT.md](AUDIT.md)。

## 当前状态

| 环节 | 状态 | 说明 |
|---|---|---|
| 数据 | ✅ | 170 条 SFT train、30 条 valid、100 条 DPO 偏好对 |
| SFT 7B LoRA | ✅ | `adapters/sft-7b` 有非零 LoRA 权重 |
| 融合模型 | ✅ | `fused_model/livestream-7b-sft` |
| DPO 3B LoRA | ⚠️ | 已训练并融合到 `fused_model/dpo-3b`，尚未跑 v2 评估 |
| KTO 3B | ✅ | 已用 TRL 重训并融合；v2 30 条结果 3/9 达标 |
| IPO 3B | ✅ | 已用 TRL 重训并融合；v2 30 条结果 5/9 达标 |
| 统一评估 v2 | ✅ | base-3b 4/9、kto-3b 3/9、ipo-3b 5/9；详见 `eval_results/V2_RESULTS.md` |

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

### 生成话术（当前可用：SFT 7B）

```bash
python -m mlx_lm generate \
  --model ./fused_model/livestream-7b-sft \
  --prompt '你是一位抖音直播带货话术专家。你的风格特征：①短句为主；②每3句话插入一次互动指令；③先讲痛点场景，再给解决方案，最后价格逼单；④情绪递进：平静→兴奋→急促；⑤口头禅：家人们、真的绝了、最后一波。

用户：产品：充电宝；卖点：20000mAh、快充、轻便；时长：1分钟

助手：' \
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
SAMPLES=30 bash scripts/run_eval_all.sh

# 只对比 v2 报告；混入 v1 会打印警告
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

每份 v2 报告都会记录 `metric_version`、`prompt_version`、`git_commit`、`git_dirty`、`evaluator_sha256`、`benchmark_sha256`、模型路径和 adapter 路径。不要跨版本拼数字。

## 当前 v2 结果（30 条，max_tokens=180）

| 模型 | 目标命中 | 关键问题 |
|---|---:|---|
| base-3b | 4/9 | 互动过多、句长偏长、结构/卖点/情绪不达标 |
| kto-3b | 3/9 | 幻觉率 0.167，卖点覆盖 0.722，未优于基座 |
| ipo-3b | 5/9 | 结构 3.6、情绪 2.973，仍不达标 |

详细表格和 caveat 见 [eval_results/V2_RESULTS.md](eval_results/V2_RESULTS.md)。结论：**真实重训后的 KTO 没有优于 3B 基座；旧 8/8 不可复现。**

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
