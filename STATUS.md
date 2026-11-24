# 直播话术助手 — 开发状态

**最后更新**：2026-11-24<br>
**当前结论**：SFT/DPO/KTO/IPO 训练链路可信；旧 KTO/IPO 适配器已作废；base-3b/KTO/IPO 已完成统一评估 v2。

---

## 一句话状态

项目已经从“宣称 KTO 8/8 达标”回退到可验证状态：**SFT 7B 可用，DPO/KTO/IPO 3B 都已真实训练；base-3b/KTO/IPO 已按同一 benchmark 完成 v2 评估，7B/DPO 待补；旧评估结论全部撤销。**

---

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 环境准备 | ✅ | Python 3.11、mlx-lm、trl、peft |
| 数据构造 | ✅ | 170 train + 30 valid + 100 DPO 偏好对 |
| SFT 7B | ✅ | `adapters/sft-7b` 非零 LoRA，`fused_model/livestream-7b-sft` 存在 |
| DPO 3B | ⚠️ | adapter 非零，已融合到 `fused_model/dpo-3b`，未跑 v2 评估 |
| KTO 3B | ✅ | 已用 TRL 重训并融合；v2 30 条 3/9 达标 |
| IPO 3B | ✅ | 已用 TRL 重训并融合；v2 30 条 5/9 达标 |
| 评估 v1 | ❌ | 跨报告拼数、指标被放宽，已作废 |
| 评估 v2 | ✅ | base-3b 4/9、kto-3b 3/9、ipo-3b 5/9；7B/DPO 待补 |

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
| `scripts/test_eval_metrics.py` | 新增 5 条指标回归测试 |
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

## v2 评估结果（30 条，max_tokens=180）

| 指标 | 目标 | base-3b | kto-3b | ipo-3b |
|---|---|---|---:|---:|
| 互动句占比 | 0.30–0.60 | 0.796 ❌ | 0.810 ❌ | 0.780 ❌ |
| 平均句长 | 15–25 | 26.233 ❌ | 27.323 ❌ | 22.787 ✅ |
| 口语词占比 | ≥0.02 | 0.026 ✅ | 0.030 ✅ | 0.027 ✅ |
| 结构完整度 | ≥4 | 3.233 ❌ | 3.400 ❌ | 3.600 ❌ |
| 卖点覆盖率 | ≥0.80 | 0.789 ❌ | 0.722 ❌ | 0.794 ❌ |
| 幻觉率 | ≤0.10 | 0.086 ✅ | 0.167 ❌ | 0.078 ✅ |
| 情绪递进 | ≥3.5 | 3.377 ❌ | 3.187 ❌ | 2.973 ❌ |
| 口头禅重复度 | ≤0.30 | 0.300 ✅ | 0.285 ✅ | 0.209 ✅ |
| 流畅度 | ≥3.5 | 4.267 ✅ | 4.367 ✅ | 4.333 ✅ |
| **达标数** | 9 | **4/9** | **3/9** | **5/9** |

**结论**：真实重训后的 KTO 没有优于 3B 基座；IPO 略高但仍未达标；旧 8/8 不可复现。

---

## 下一步

1. 已完成 3B KTO/IPO 重训和 v2 评估（2026-11-24）。如要复现训练：

```bash
DEVICE=mps DTYPE=bfloat16 EPOCHS=1 MAX_LENGTH=128 bash scripts/train_preferences.sh
```

2. 补跑 base-7b / sft-7b / dpo-3b 的 v2 评估：

```bash
SAMPLES=30 bash scripts/run_eval_all.sh
```

3. 用 `max_tokens=300` 复核 3B 结果，排除 180 token 截断影响。

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
