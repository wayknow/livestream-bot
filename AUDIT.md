# 项目审计与修复记录

**审计日期**：2026-11-24  
**审计对象**：`livestream-bot` 提交历史、训练产物、评估报告与文档  
**结论**：SFT/DPO 训练链路基本真实；旧 KTO/IPO 训练没有生效；旧“8/8 指标达标”由跨评估报告挑数拼接而成，不能作为项目结论。

---

## 1. 严重问题：KTO/IPO 权重没有变化

对 `adapters/kto-3b/model.safetensors`、`adapters/ipo-3b/model.safetensors` 与 `Qwen/Qwen2.5-3B-Instruct` 做了逐张量比对：

- 三个模型都是 434 个张量、3,085,938,688 个参数。
- 抽样张量 `torch.equal=True`，`max_abs_diff=0`。
- 全量 `sum`/`sumsq` 完全一致。
- KTO/IPO 文件约 12.34GB，是基座 bf16 权重上转为 fp32 后保存的结果。

根因在旧训练脚本：

- `scripts/kto_train.py` 旧实现：`torch.no_grad()` → `-loss.item()` → 重新包一个 `torch.tensor(..., requires_grad=True)`，梯度没有连到模型。
- `scripts/ipo_train.py` 旧实现：同样使用 `torch.no_grad().item()` 和独立 `reward_diff` 张量，优化器不会更新模型。
- 即使梯度连通，旧 KTO 也不是标准 KTO：没有 reference policy、KL 项和前景理论损失。

因此旧 `adapters/kto-3b`、`adapters/ipo-3b` 应视为无效产物；旧的 KTO 评估实际上是“3B 基座 + system prompt”的评估。

修复：

- `scripts/kto_train.py` 改为 TRL `KTOTrainer` + LoRA。
- `scripts/ipo_train.py` 改为 TRL `DPOTrainer(loss_type=["ipo"])` + LoRA。
- 两个脚本训练后自动调用 `scripts/verify_training.py`，LoRA 权重仍为零时直接失败。
- 已用 Qwen2.5-1.5B 跑通 1 步冒烟测试，两个脚本都能保存非零 LoRA 并通过校验。
- 已用 Qwen2.5-3B + MPS 完整重训：KTO 216.6s（loss 0.4948，reward margin 从 -0.0156 转正到 0.0756），IPO 144.9s（loss 24.99，reward margin 0.0094）；两个 adapter 均通过非零校验。

---

## 2. 严重问题：KTO 指标跨报告拼接

README/STATUS 中的 KTO 列不是同一次评估的结果：

| 指标 | 文档值 | `eval_KTO模型30条_20260902_173400.json` | `eval_KTO模型重新定义_20260903_163535.json` |
|---|---:|---:|---:|
| 互动密度 | 0.431 | **0.431** | 0.405 |
| 平均句长 | 19.41 | **19.41** | 16.853 |
| 口语词占比 | 0.035 | **0.035** | 0.031 |
| 结构完整度 | 4.567 | **4.567** | 4.467 |
| 卖点覆盖率 | 0.845 | 0.789 | **0.845** |
| 幻觉率 | 0.000 | 0.066 | **0.000** |
| 情绪递进 | 3.993 | **3.993** | 3.933 |
| 流畅度 | 4.933 | **4.933** | 4.933 |

- 提交 `751eebf` 已开始混用 `30条`、`逼单修复` 等报告。
- 提交 `16471d7` 只把卖点/幻觉/流畅度替换为新报告里的更好数值，其余继续保留旧报告。
- 按文档自己的 `0.3–0.4` 目标，`0.431` 本来就不达标，但表格仍标 ✅ 并宣称 8/8。

修复：

- 评估协议升级为 `metric_version=2.0`。
- 每次报告写入 `metric_version`、`prompt_version`、`git_commit`、`benchmark_sha256`、模型与 adapter 路径。
- 对比模式检测到旧版本报告时明确警告“不能作为结论”。

---

## 3. 严重问题：README 的 DPO 列实际是 SFT v7

README 中的 DPO 数字：

`0.181 / 7.28 / 0.022 / 4.40 / 0.833 / 0.000 / 3.70 / 5.00`

与 `eval_results/eval_SFT优化版v7_20260831_163445.json` 完全一致。

- `git log --all -- eval_results` 中从未出现 DPO 评估报告。
- 提交 `5c93e9d` 把这一列命名为 DPO；当时 STATUS 里还写着 `DPO (SFT v7)`。
- 最新提交 `66c3b5c` 基于这组 SFT 数字写了“DPO 长度偏差”解释，属于错误归因。
- DPO adapter 本身确实有非零 LoRA 权重，但从未在 30 条 benchmark 上评估过。

修复：

- 不再把 SFT v7 当作 DPO。
- 统一评估脚本支持 `--adapter adapters/dpo-3b`，需要重新跑 DPO 才能得到真实 DPO 数字。

---

## 4. 评估口径被反复放宽

历史提交中的目标变化：

| 指标 | 旧目标 | 新目标 | 提交 |
|---|---|---|---|
| 互动密度 | `>0.4` | `>0.2` | `c6aacb8` |
| 口语词占比 | `>0.15` | `>0.02` | `c6aacb8` |
| 平均句长 | `<15` | `15-20` | `98551c6` |
| 卖点覆盖率 | `>0.85` | `>0.8` | `2852e77` |
| 卖点覆盖率 | `0.6-0.8` | `>0.8` | `0c4f055` |

其他口径问题：

- 旧幻觉率只检测负数或 `>10000` 的数字，价格、数量、日期全部白名单化；`0.000` 几乎必然出现。
- 旧“LLM 裁判”实际没有传 `model_path`，永远走规则函数。
- 旧互动密度统计“命中的不同互动词种类数 / 句数”，不是出现次数，也不是句子覆盖率，可能大于 1。
- 基座评估 3 条且没有 system prompt，SFT 5 条，KTO 30 条且有 system prompt，条件不可比。

修复：

- 互动密度改为“包含互动词的句子数 / 总句数”，范围 `[0,1]`。
- 幻觉率改为“输出中未在 prompt 出现的数字 / 输出数字总数”，只豁免倒计时和互动数字。
- 流畅度默认标记为 `rule`；只有传 `--judge-model` 才调用 LLM 裁判。
- 结构完整度改用更具体的短语模式，不再用单个“3/2/1”“这个”“今天”凑分。
- 情绪递进要求真实递增，平坦文本不再拿高分。

---

## 5. 提交与仓库卫生

- 69 个提交，只有 43 个不同树状态；18 对提交内容完全相同，3 个 merge，22 个只改 `.md`。
- 最终 6 个评估 JSON 在审计时仍是 untracked，文档却引用了这些文件。
- `data/train.jsonl`（170 条）、`valid.jsonl`（30 条）、`dpo.jsonl`（100 条）此前被 gitignore，克隆后无法复现。
- `REPORT.md` 仍停留在 2026-08-28 的 DPO 阶段。

修复：

- `.gitignore` 允许跟踪三份数据文件和 `adapters/README.md`。
- 新增 `eval_results/README.md` 说明旧报告作废。
- 新增 `scripts/run_eval_all.sh`，用同一 prompt、同一样本量评估所有模型。

---

## 6. 当前可信状态

| 项目 | 状态 |
|---|---|
| 数据生成 | 可信，`train=170 / valid=30 / dpo=100` |
| SFT 7B LoRA | 可信，adapter 有非零 LoRA 权重 |
| DPO 3B LoRA | 已训练并完成 300-token v2 评估：4/9 |
| KTO 3B | 旧产物无效；已用 TRL 重训并完成 300-token v2 评估：4/9 |
| IPO 3B | 旧产物无效；已用 TRL 重训并完成 300-token v2 评估：4/9 |
| base-3b | 已完成 300-token v2 评估：3/9 |
| base-7b | 已完成 300-token v2 评估：5/9 |
| SFT 7B | 已完成 300-token v2 评估：4/9 |
| Qwen3.5-4B | 已完成 300-token v2 评估（thinking 关闭）：4/9 |
| Qwen3.5-9B | 已完成 300-token v2 评估（thinking 关闭）：4/9 |
| 旧 `8/8` 结论 | 撤销 |

**v2 结论**：base-7b 5/9 是当前最强基线；KTO/IPO/DPO-3b 均为 4/9，只比 base-3b 多命中 1 项，不能证明偏好优化有效。SFT-7b 4/9 低于 base-7b；Qwen3.5-4B/9B 也都是 4/9，9B 互动风格最好但幻觉率 0.281。所有模型都未达到 6/9，结构完整度、情绪递进和幻觉是共同短板。旧 8/8 无法复现。详细数字见 `eval_results/V2_RESULTS.md`。

2026-11-24 的 180-token 报告已作废：30/30 base 输出达到上限，21/90 条输出出现“用户/助手”角色标签泄漏。正式报告改为 Qwen chat template，`max_tokens=300`，输出已无角色泄漏。

评测边界：30 条 benchmark 与 train/valid/dpo 无完全相同的 prompt，但有 9/30 个产品也出现在训练数据中（卖点与句式不同）。当前结果属于同域测试，不能当作完全未见产品的泛化结论。

---

## 7. 复现方式

```bash
conda activate mlx

# 1. 验证旧产物（应失败）
python scripts/verify_training.py --base Qwen/Qwen2.5-3B-Instruct --trained adapters/kto-3b

# 2. 重新训练 KTO / IPO（3B LoRA，可选；2026-11-24 已重训）
DEVICE=mps DTYPE=bfloat16 EPOCHS=1 MAX_LENGTH=128 bash scripts/train_preferences.sh

# 3. 统一评估所有模型（同 prompt、同 30 条样本）
SAMPLES=30 bash scripts/run_eval_all.sh

# 4. 只跑指标回归测试
python scripts/test_eval_metrics.py
```

不要再使用“KTO 最适合数字人直播”的结论：正式 v2 中 KTO 与 IPO/DPO 都是 4/9，且都低于 base-7b 的 5/9。
