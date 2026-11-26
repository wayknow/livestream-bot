# 直播话术助手（Livestream Script Assistant）

基于 QLoRA SFT + 偏好优化的直播带货话术生成实验。输入产品信息，输出主播风格的带货文案。

**硬件要求**：24GB Mac Mini（Apple Silicon）<br>
**正式评估**：Qwen2.5 六个模型 + Qwen3.5-4B/9B；Qwen3.5-4B SFT-v4 已用 v3 prompt 重新生成<br>
**当前最好**：`Qwen3.5-4B + SFT-v4` **8/8**（v3，seed 0/1）；自动指标达标，但仍需人工听感评审

> **审计说明**：旧版“KTO 8/8 指标达标”的结论已撤销。旧 KTO/IPO 权重与基座逐元素一致，旧评估数字由多次报告拼接而成。当前结论只来自同一 benchmark、同一 v3 prompt、同一生成参数的报告。SFT-v4 的详细结果见 [V3_SFT_V4_RESULTS.md](eval_results/V3_SFT_V4_RESULTS.md)，历史审计见 [AUDIT.md](AUDIT.md)。

## 当前状态

| 环节 | 状态 | 说明 |
|---|---|---|
| 数据 | ✅ | 旧数据 170/30/100；新增 SFT-v4 600 train + 120 valid，产品级切分 |
| SFT 7B LoRA | ⚠️ | 训练真实发生；v2 4/9，未优于 base-7b 的 5/9 |
| 融合模型 | ✅ | `fused_model/livestream-7b-sft` |
| DPO 3B LoRA | ✅ | 已训练并融合；v2 4/9，幻觉率 0.036 为全组最低 |
| KTO 3B | ✅ | 已用 TRL 重训并融合；v2 4/9 |
| IPO 3B | ✅ | 已用 TRL 重训并融合；v2 4/9 |
| Qwen3.5-4B | ✅ | v3 基线 6/8；SFT-v4 8/8（幻觉 0.011/0.000，两个 seed） |
| Qwen3.5-9B | ✅ | v3 基线 6/8；结构 0.657、幻觉 0.611 仍失败 |
| 统一评估 v3 | ✅ | 30 条同协议生成 + 95% CI + 单条达标率 + 输出多样性诊断 |

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
├── config_qwen35_sft.yaml   # Qwen3.5-4B SFT-v4 LoRA 配置
├── data/
│   ├── train.jsonl          # 170 条 SFT 训练集
│   ├── valid.jsonl          # 30 条 SFT 验证集
│   ├── dpo.jsonl            # 100 条偏好对
│   ├── gold_scripts.jsonl   # 10 条人工 gold 话术
│   ├── sft_v4/              # 600 train + 120 valid，事实一致、产品级切分
│   └── benchmark_prompts.jsonl
├── scripts/
│   ├── generate_data.py
│   ├── dpo_train.py
│   ├── kto_train.py         # TRL KTOTrainer + 权重校验
│   ├── ipo_train.py         # TRL DPOTrainer(loss_type=ipo) + 权重校验
│   ├── verify_training.py   # 检测 adapter/模型是否真的变化
│   ├── fuse_peft.py         # PEFT LoRA → 完整模型，供 mlx-lm 推理
│   ├── eval_pipeline.py     # 评估协议 v3
│   ├── test_eval_metrics.py # 30 条指标回归测试
│   ├── build_sft_v4.py      # 构造 SFT-v4
│   ├── train_qwen35_sft.sh
│   ├── eval_qwen35_pair.sh  # base / SFT 同协议对比
│   ├── report_eval.py       # CI、单条达标率、多样性报告
│   ├── rescore_v3.py        # 旧 v2 输出离线重算
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
└── eval_results/            # v1/v2 仅历史；V3_SFT_V4_RESULTS.md 是当前结论
```

## 快速开始

```bash
conda activate mlx
cd /Users/xiaoxiao/work/finetuning/livestream-bot
```

### 生成话术（Qwen3.5-4B + SFT-v4）

```bash
python -m mlx_lm generate \
  --model /Users/xiaoxiao/work/finetuning/models/Qwen3.5-4B \
  --adapter-path ./adapters/qwen3.5-4b-sft-v4 \
  --system-prompt '你是一位抖音直播带货话术专家。短句为主，每句不超过15字；互动句占比30%–50%；结构为痛点场景→解决方案→价格说明→行动指令；价格、数量、库存、时间、优惠等具体信息只能来自用户输入，用户没给就省略或说价格以直播间为准。' \
  --prompt '产品：充电宝；卖点：20000mAh、快充、轻便；时长：1分钟；价格：99元' \
  --max-tokens 300 \
  --temp 0.7 \
  --top-p 0.9
```

注意：生成时必须使用 `livestream-v3` system prompt（见 `scripts/eval_pipeline.py`），否则报告不可比。

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

# 4B base 与 SFT-v4，同 prompt、seed、30 条 benchmark、300 token
SEED=0 SAMPLES=30 MAX_TOKENS=300 bash scripts/eval_qwen35_pair.sh

# 输出 CI、单条达标率、句级唯一率和两两相似度
python scripts/report_eval.py \
  eval_results/eval_v3_qwen3.5-4b-base-v3-s0_*.json \
  eval_results/eval_v3_qwen3.5-4b-sft-v3-s0_*.json

# 对已有 v2 输出做 v3 离线重算（不重新生成）
python scripts/rescore_v3.py
```

## 评估协议 v3

| 指标 | 定义 | 目标 |
|---|---|---|
| 互动句占比 | 含互动词的句子数 / 总句数 | 0.30–0.60 |
| 口语词占比 | 口语词出现次数 / 总字数 | ≥0.02 |
| 结构覆盖 | 只要求 prompt 实际提供的模块：基础痛点/方案/行动，有价格才要求价格锚定，有库存/限时优惠才要求限时福利 | ≥0.75 |
| 卖点覆盖率 | 中文数字/单位/常见同义表达归一化后的卖点命中数 | ≥0.80 |
| 幻觉率 | 无依据或与事实表冲突的 claim / 唯一 claim 数 | ≤0.10 |
| 情绪递进 | 平静→兴奋→急促；按单条通过率统计 | ≥3.5 且 ≥70% 样本通过 |
| 口头禅重复度 | 最高频口头禅次数 / 总句数 | ≤0.30 |
| 流畅度 | 默认规则评分；传 `--judge-model` 才用 LLM | ≥3.5 |

平均句长仍会记录，但只作为诊断，不计入目标或通过数。价格、库存等变量在风格指标中替换为 `<PRICE>` / `<COUNT>` typed slot；幻觉校验始终使用原始输出和 benchmark 的 `facts` 字段，未提供的事实用 `null` 表示。

每份 v3 报告都会记录 `metric_version`、`prompt_version`、`git_commit`、`git_dirty`、`evaluator_sha256`、`benchmark_sha256`、事实表、模型路径、adapter 路径和生成参数。不要跨版本或跨生成参数拼数字。

## 当前 v3 结果（2026-11-26，30 条）

| 模型 | 目标命中 | 结构 | 幻觉率 | 说明 |
|---|---:|---:|---:|---|
| Qwen3.5-4B base | 6/8 | 0.588 ❌ | 0.539 ❌ | 会编价格/库存 |
| **Qwen3.5-4B SFT-v4** | **8/8** | **0.987 ✅** | **0.011 ✅** | seed 1 为 0.000；模板化仍重 |
| Qwen3.5-9B base | 6/8 | 0.657 ❌ | 0.611 ❌ | 更大模型未解决事实一致性 |
| Qwen2.5 六个模型 | 4/8 | — | — | 结构/幻觉仍失败 |

详细数字、CI、checkpoint 选择和 caveat 见 [eval_results/V3_SFT_V4_RESULTS.md](eval_results/V3_SFT_V4_RESULTS.md)。结论：**SFT-v4 让 4B 的自动指标从 6/8 到 8/8，但输出模板化严重（句级唯一率 46.3%、两两相似度 0.582），不能仅凭 8/8 宣称生产可用。**

## 文档

- [审计与修复记录](AUDIT.md)
- [产品描述与数据格式](PRODUCT.md)
- [开发状态](STATUS.md)
- [项目决策日志](PROJECT_LOG.md)
- [偏好优化方法](PREFERENCE_METHODS.md)
- [项目报告](REPORT.md)

## Tech Stack

- 训练：mlx-lm（SFT）、TRL + PyTorch（DPO/KTO/IPO）
- 基座：Qwen2.5-7B/3B-Instruct、Qwen3.5-4B/9B
- 评估：规则引擎 + 可选本地 LLM 流畅度裁判
- 硬件：Mac Mini M4，24GB unified memory
