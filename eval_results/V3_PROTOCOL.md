# 评估协议 v3

**指标版本**：`3.0`<br>
**生成 prompt 版本**：`livestream-v3`（新生成）；旧 v2 输出离线重算时保留 `livestream-v2`<br>
**Benchmark**：`data/benchmark_prompts.jsonl`，30 条，产品级冻结

---

## 1. 双轨评估

v3 把“风格”和“事实”分开，避免一个全局通配符把编造的价格/库存洗掉。

- **风格轨**：结构、卖点、情绪等使用 `normalize_slots()`，价格/库存/倒计时值替换为 `<PRICE>` / `<COUNT>`，中文数字和常见单位先归一化。
- **事实轨**：`hallucination_rate` 始终使用原始输出和 benchmark `facts`，不做通配。缺失字段为 `null`，输出中的具体数字或泛化逼单 claim 会被判为 unsupported / contradicted。

```text
风格：平时299，今天只要159，最后20台
     → 平时<PRICE>，今天只要<PRICE>，最后<COUNT>台
事实：原始值 299 / 159 / 20 分别对 facts 表校验
```

## 2. Claim 级幻觉

提取四类 claim：

1. 价格：`元/块/块钱`，或 `原价/到手价/只要/仅需/价格` 后紧跟数字。
2. 库存：`最后/仅剩/只剩/还剩/库存/限量` + `单/件/支/个/套/盒/瓶`。
3. 规格/优惠：`mAh/小时/分钟/%/SPF/UPF/买N送N` 等。
4. 定性 claim：`绝对/唯一/保证/认证/专利/平替/大牌同款` 等。

每条 claim 标记为：

| 状态 | 含义 | 是否计入幻觉 |
|---|---|---|
| `supported` | facts 表或 prompt 中有依据 | 否 |
| `unsupported` | 事实未提供，模型自行补充 | 是 |
| `contradicted` | 与 facts 表冲突（如 `最后10单` 写成 `最后50单`） | 是 |

`价格以直播间为准` 不产生数字 claim，因此不会因为“没有价格”被判幻觉。

## 3. 结构覆盖

结构不再固定要求五个模块全部由 prompt 提供。只要求当前输入实际支持的模块：

- **基础必需**：痛点场景、解决方案、行动指令。
- **价格锚定**：prompt 提供价格/原价，或提供价格类优惠（优惠、半价、买N送N、立减）时才要求。
- **限时福利**：prompt 提供库存，或提供限时类优惠（限时、倒计时、最后、秒杀）时才要求。

`structure_coverage = 命中的必需模块数 / 必需模块数`，目标 `>= 0.75`。`structure_completeness = coverage * 5` 仅作为 0–5 的可读诊断值。

## 4. 目标

| 指标 | 目标 |
|---|---|
| 互动句占比 | 0.30–0.60 |
| 口语词占比 | ≥0.02 |
| 结构覆盖（必需模块） | ≥0.75 |
| 卖点覆盖率 | ≥0.80 |
| 幻觉率 | ≤0.10 |
| 情绪递进 | 单条 ≥3.5 的样本占比 ≥70% |
| 口头禅重复度 | ≤0.30 |
| 流畅度 | ≥3.5 |

平均句长不再计入目标，只作为诊断。原因是 system prompt 要求短句，旧的“平均 15–25 字”目标与之冲突。

## 5. 统计

- 每个数值指标报告 mean、std、min、max 和 95% percentile bootstrap CI。
- 每个目标同时报告单条样本 pass rate 及其 95% bootstrap CI。
- 情绪递进按单条 pass rate 判定，不再用 30 条均值直接判失败。
- 同一对比必须使用相同 `metric_version`、`prompt_version`、`benchmark_sha256` 和生成参数（含 seed）。

## 6. 回归测试

`scripts/test_eval_metrics.py` 覆盖：

- 价格/库存通配不影响事实校验；
- prompt 未给价格时，`只要99` 失败，`价格以直播间为准` 通过；
- prompt 给 `最后10单` 时，`最后50单` 为 contradiction；
- `扣1`、`选一个` 不被当成价格/数量 claim；
- `价格29元1盒` 中 `1盒` 不被误判为价格；
- 小数价格 `39.9` 不被归一化成 `399`；
- 中文数字/单位/卖点改写可以通过；
- 结构只要求 prompt 实际提供的模块；
- `最后45条`、`最后18张`、`最后40袋`、`最后20台`、`最后70包` 等库存单位能正确解析；
- gold 话术 8/8 通过。

`scripts/report_eval.py` 额外输出每条指标的 95% bootstrap CI、单条达标率、单条样本 8/8 数量、句级唯一率、两两相似度和平均句长诊断。多样性与句长都只用于判断模板化/过拟合，不计入目标。

## 7. 复现

```bash
python scripts/test_eval_metrics.py
python scripts/rescore_v3.py                 # 旧 v2 输出离线重算
SEED=0 SAMPLES=30 bash scripts/eval_qwen35_pair.sh
python scripts/report_eval.py eval_results/eval_v3_qwen3.5-4b-base-v3-s0_*.json eval_results/eval_v3_qwen3.5-4b-sft-v3-s0_*.json
```
