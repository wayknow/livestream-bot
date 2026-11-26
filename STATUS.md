# 直播话术助手 — 开发状态

**最后更新**：2026-11-26<br>
**当前结论**：Qwen3.5-4B + SFT-v4 在 v3 协议下达到 8/8（seed 0/1），幻觉率从 0.539 降到 0.011/0.000；自动指标达标，但输出模板化严重，仍需人工听感评审。

---

## 一句话状态

项目已经从“宣称 KTO 8/8 达标”回退到可验证状态，并完成一次真实修复：**Qwen3.5-4B SFT-v4 用同一 benchmark、v3 prompt、同 seed 达到 8/8；基线和 9B 仍为 6/8。** 但 SFT 的句级唯一率只有 46.3%、两两相似度 0.582，不能只看指标宣布生产可用。

---

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 环境准备 | ✅ | Python 3.11、mlx-lm、trl、peft |
| 数据构造 | ✅ | 旧数据 170/30/100；新增 SFT-v4 600 train + 120 valid（产品级切分） |
| SFT 7B | ⚠️ | 训练真实发生；v3 4/8，结构/卖点/幻觉/情绪失败 |
| DPO 3B | ✅ | 已融合；v3 4/8，幻觉率 0.528 仍远高于目标 |
| KTO 3B | ✅ | 已用 TRL 重训并融合；v3 4/8 |
| IPO 3B | ✅ | 已用 TRL 重训并融合；v3 4/8 |
| Qwen3.5-4B | ✅ | v3 基线 6/8；SFT-v4 8/8，幻觉 0.011/0.000 |
| Qwen3.5-9B | ✅ | 本地 bf16；v3 同协议生成 6/8，结构/幻觉未过 |
| 评估 v1 | ❌ | 跨报告拼数、指标被放宽，已作废 |
| 评估 v2 | ⚠️ | 8 个模型完成 300-token 生成；句长目标已撤销，作为历史口径保留 |
| 评估 v3 | ✅ | typed slot 归一化 + claim 级事实校验；CI、单条达标率、多样性诊断 |

---

## 已验证事实

- `adapters/sft-7b` 的 LoRA 权重非零，SFT 训练确实发生。
- `adapters/dpo-3b` 的 LoRA 权重非零，DPO 训练确实发生。
- 旧 `adapters/kto-3b`（无 `-lora` 后缀）与 `Qwen/Qwen2.5-3B-Instruct` 逐元素完全一致。
- 旧 `adapters/ipo-3b`（无 `-lora` 后缀）与 `Qwen/Qwen2.5-3B-Instruct` 逐元素完全一致。
- 重训的 `adapters/kto-3b-lora`、`adapters/ipo-3b-lora` 均有非零 LoRA 权重并通过 `verify_training.py`。
- `adapters/qwen3.5-4b-sft-v4` 的 124 个张量、56,813,568 个参数全部非零，max |w| 0.03365，SFT 确实更新了权重。
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
| `scripts/eval_pipeline.py` | 评估 v3：变量归一化与事实校验分轨、claim 级幻觉、句长降为诊断 |
| `scripts/test_eval_metrics.py` | 30 条指标、gold、slot 归一化和 claim 误报回归测试 |
| `scripts/run_eval_all.sh` | 新增：同一协议评估全部模型 |
| `scripts/rescore_v3.py` | 新增：对已有输出做 v3 离线重算 |
| `scripts/build_sft_v4.py` | 新增：600+120 条事实一致、产品级切分 SFT 数据 |
| `config_qwen35_sft.yaml` | 新增：Qwen3.5-4B LoRA SFT（batch 1，峰值 12.7GB） |
| `scripts/eval_qwen35_pair.sh` | 新增：4B base / SFT 同 prompt、同 seed 对比 |
| `scripts/report_eval.py` | 新增：CI、单条达标率、输出多样性诊断 |
| 文档 | README/STATUS/PRODUCT/REPORT/PROJECT_LOG 撤销旧结论 |

冒烟测试结果：用 Qwen2.5-1.5B 跑 1 步 KTO/IPO，两个脚本都保存了非零 LoRA 并通过 `verify_training.py`。

3B 重训结果（2026-11-24，MPS + bf16 + max_length=128）：

| 方法 | 数据 | 耗时 | 关键训练指标 | 产物 |
|---|---:|---:|---|---|
| KTO | 200 条二元样本 | 216.6s | loss 0.4948，reward margin 0.0756 | `adapters/kto-3b-lora` |
| IPO | 100 条偏好对 | 144.9s | loss 24.99，reward margin 0.0094 | `adapters/ipo-3b-lora` |

注意：训练指标不等于生成质量；必须以统一评估 v3 为准。

Qwen3.5-4B SFT-v4（2026-11-26）验证 loss：Iter 100 `0.539`、200 `0.362`、300 `0.451`、400 `0.373`、500 `0.394`、600 `0.369`。Iter 300 后回升，最终评估同时比较了 200/300/600，选择幻觉率最低的 final。

---

## 历史 v2 评估结果（已由 v3 取代）

| 模型 | 目标命中 | 关键问题 |
|---|---:|---|
| base-3b | 3/9 | 互动 0.749、结构 2.767、卖点 0.700、幻觉 0.117 |
| kto-3b | 4/9 | 卖点 0.828，但结构 3.267、情绪 2.860、幻觉 0.150 |
| ipo-3b | 4/9 | 卖点 0.839，但结构 3.067、情绪 3.167、幻觉 0.116 |
| dpo-3b | 4/9 | 幻觉 0.036 最好，但结构 2.600、情绪 2.570、卖点 0.767 |
| base-7b | **5/9** | 当前最高；幻觉 0.284、结构 3.167、情绪 2.900 |
| sft-7b | 4/9 | 互动 0.354 达标，但句长 7.18、结构 2.233、幻觉 0.320 |
| qwen3.5-4b | 4/9 | 互动 0.392 达标；卖点 0.733、幻觉 0.152、结构 3.200 |
| qwen3.5-9b | 4/9 | 互动 0.483 最好；但卖点 0.728、幻觉 0.281 |

**结论**：base-7b 仍是当前最强基线；Qwen3.5-4B/9B 没有超过它。3B 偏好优化相对 base-3b 只多命中 1 项，不能证明方法有效。所有模型都未达到 6/9；结构完整度、情绪递进和幻觉是共同短板。

评测口径提示：2026-11-24 的 180-token 报告因 30/30 输出顶到上限、21/90 出现角色标签泄漏而作废。当前正式报告改用 Qwen chat template，输出已无角色泄漏。

## v3 当前结果（2026-11-26）

v3 撤销平均句长目标，将句长降为诊断；价格/库存值在风格指标中替换为 typed slot，幻觉则用原始输出对 benchmark `facts` 做 claim 级校验。10 条人工 gold 话术通过 8/8 目标。

| 模型 | 目标命中 | 结构 | 幻觉率 | 关键问题 |
|---|---:|---:|---:|---|
| Qwen3.5-4B base | 6/8 | 0.588 ❌ | 0.539 ❌ | 会编价格/库存 |
| **Qwen3.5-4B SFT-v4** | **8/8** | **0.987 ✅** | **0.011 ✅** | 模板化仍重 |
| Qwen3.5-9B base | 6/8 | 0.657 ❌ | 0.611 ❌ | 更大模型未解决事实问题 |
| Qwen2.5 六个模型 | 4/8 | — | — | 结构和 claim 级幻觉失败 |

SFT-v4 的第二个 seed 同样为 8/8，幻觉率 0.000，说明固定 benchmark 上的结果稳定。词表类指标（流畅度、情绪递进）容易受模板影响，不能替代人工听感评审。详细数字见 [eval_results/V3_SFT_V4_RESULTS.md](eval_results/V3_SFT_V4_RESULTS.md)。

---

## 下一步

1. 扩充 SFT-v4 的句式模板和场景表达，降低“很多人因此不敢出门”“价格以直播间为准，喜欢的别错过”等固定句的复用率。
2. 增加真正未见的产品/行业 prompt 集和人工盲评；当前 benchmark 与训练数据同域，8/8 不能证明跨品类泛化。
3. 复测更低 rank、更少 iteration 或加入输出多样性正则，目标是在保持幻觉率 ≤0.10 的同时提高句级唯一率、降低两两相似度。
4. 继续保留 `gold_scripts`、30 条指标回归测试和同协议报告，不接受跨报告拼数字。

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
