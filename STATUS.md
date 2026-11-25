# 直播话术助手 — 开发状态

**最后更新**：2026-11-25<br>
**当前结论**：训练链路可信；六个模型已完成同一协议的 300-token v2 评估；base-7b 5/9 为当前最好，SFT 与偏好优化均没有跑赢对应基座。

---

## 一句话状态

项目已经从“宣称 KTO 8/8 达标”回退到可验证状态：**六个模型均已按同一 benchmark、同一生成参数完成 v2 评估；最高为 base-7b 的 5/9，没有任何模型接近 8/9；旧 8/8 与 2026-11-24 的 180-token 结果全部作废。**

---

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 环境准备 | ✅ | Python 3.11、mlx-lm、trl、peft |
| 数据构造 | ✅ | 170 train + 30 valid + 100 DPO 偏好对 |
| SFT 7B | ⚠️ | 训练真实发生；v2 4/9，低于 base-7b 的 5/9 |
| DPO 3B | ✅ | 已融合；v2 4/9，幻觉率 0.036 最低 |
| KTO 3B | ✅ | 已用 TRL 重训并融合；v2 4/9 |
| IPO 3B | ✅ | 已用 TRL 重训并融合；v2 4/9 |
| 评估 v1 | ❌ | 跨报告拼数、指标被放宽，已作废 |
| 评估 v2 | ✅ | 6 个模型完成 300-token 正式评估；旧 180-token 报告已作废 |

---

## 已验证事实

- `adapters/sft-7b` 的 LoRA 权重非零，SFT 训练确实发生。
- `adapters/dpo-3b` 的 LoRA 权重非零，DPO 训练确实发生。
- 旧 `adapters/kto-3b`（无 `-lora` 后缀）与 `Qwen/Qwen2.5-3B-Instruct` 逐元素完全一致。
- 旧 `adapters/ipo-3b`（无 `-lora` 后缀）与 `Qwen/Qwen2.5-3B-Instruct` 逐元素完全一致。
- 重训的 `adapters/kto-3b-lora`、`adapters/ipo-3b-lora` 均有非零 LoRA 权重并通过 `verify_training.py`。
- README 旧的“DPO”列等于 `eval_SFT优化版v7_20260831_163445.json`。
- README 旧 KTO 列混用了 `eval_KTO模型30条_20260902_173400.json` 与 `eval_KTO模型重新定义_20260903_163535.json`。

---

## 本次修复

| 文件 | 修复 |
|---|---|
| `scripts/kto_train.py` | 改用 TRL `KTOTrainer` + LoRA，训练后校验权重 |
| `scripts/ipo_train.py` | 改用 TRL `DPOTrainer(loss_type="ipo")` + LoRA，训练后校验 |
| `scripts/verify_training.py` | 新增：检测 adapter 是否仍为零 / 模型是否与基座一致 |
| `scripts/fuse_peft.py` | 新增：把 TRL/PEFT LoRA 融合为 mlx-lm 可加载的完整模型 |
| `scripts/eval_pipeline.py` | 评估 v2：统一 prompt、修复互动/幻觉/情绪/结构指标、记录来源 |
| `scripts/test_eval_metrics.py` | 8 条指标与报告来源回归测试 |
| `scripts/run_eval_all.sh` | 新增：同一协议评估全部模型 |
| 文档 | README/STATUS/PRODUCT/REPORT/PROJECT_LOG 撤销旧结论 |

冒烟测试结果：用 Qwen2.5-1.5B 跑 1 步 KTO/IPO，两个脚本都保存了非零 LoRA 并通过 `verify_training.py`。

3B 重训结果（2026-11-24，MPS + bf16 + max_length=128）：

| 方法 | 数据 | 耗时 | 关键训练指标 | 产物 |
|---|---:|---:|---|---|
| KTO | 200 条二元样本 | 216.6s | loss 0.4948，reward margin 0.0756 | `adapters/kto-3b-lora` |
| IPO | 100 条偏好对 | 144.9s | loss 24.99，reward margin 0.0094 | `adapters/ipo-3b-lora` |

注意：训练指标不等于生成质量；必须以统一评估 v2 为准。

---

## 正式 v2 评估结果（2026-11-25，30 条，max_tokens=300）

| 模型 | 目标命中 | 关键问题 |
|---|---:|---|
| base-3b | 3/9 | 互动 0.749、结构 2.767、卖点 0.700、幻觉 0.117 |
| kto-3b | 4/9 | 卖点 0.828，但结构 3.267、情绪 2.860、幻觉 0.150 |
| ipo-3b | 4/9 | 卖点 0.839，但结构 3.067、情绪 3.167、幻觉 0.116 |
| dpo-3b | 4/9 | 幻觉 0.036 最好，但结构 2.600、情绪 2.570、卖点 0.767 |
| base-7b | **5/9** | 当前最高；幻觉 0.284、结构 3.167、情绪 2.900 |
| sft-7b | 4/9 | 互动 0.354 达标，但句长 7.18、结构 2.233、幻觉 0.320 |

**结论**：base-7b 是当前最强基线；3B 偏好优化相对 base-3b 只多命中 1 项，不能证明方法有效。所有模型都未达到 8/9；结构完整度和情绪递进是共同短板。

评测口径提示：2026-11-24 的 180-token 报告因 30/30 输出顶到上限、21/90 出现角色标签泄漏而作废。当前正式报告改用 Qwen chat template，输出已无角色泄漏。

---

## 下一步

1. 已完成 3B KTO/IPO 重训和 v2 评估（2026-11-24）。如要复现训练：

```bash
DEVICE=mps DTYPE=bfloat16 EPOCHS=1 MAX_LENGTH=128 bash scripts/train_preferences.sh
```

2. 重新定义平均句长目标：当前 system prompt 要求“每句不超过 15 字”，指标目标却是 15–25 字，二者冲突。

3. 用 Qwen3-8B/4B（关闭 thinking）跑同 benchmark 基线；当前 2.5 系列所有模型都未达到 6/9。

4. 加人工听感/人工评审；自动指标不能完全代表直播效果。

---

## 环境

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| mlx-lm | 0.31.3 |
| transformers | 5.16.1 |
| trl | 1.12.0 |
| peft | 0.20.0 |
| 硬件 | Mac Mini M4, 24GB unified memory |
