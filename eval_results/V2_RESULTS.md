# Protocol v2 results (2026-11-25)

> **Superseded for target conclusions by protocol v3.** The v2 average-sentence-length target was withdrawn, selling-point coverage used exact string matching, and hallucination only checked numbers. Use this file for v2 generation provenance; current conclusions are in [`V3_RESCORE.md`](V3_RESCORE.md).

## Scope

- Benchmark: `data/benchmark_prompts.jsonl` (30 prompts)
- Metric version: `2.0`; prompt version: `livestream-v2`
- Input format: Qwen chat template (`system` + `user`, generation prompt)
- Generation: `temp=0.7`, `top_p=0.9`, `max_tokens=300`
- Models: `base-3b`, `kto-3b`, `ipo-3b`, `dpo-3b`, `base-7b`, `sft-7b`, `qwen3.5-4b`, `qwen3.5-9b`
- All eight reports use the same benchmark hash, prompt version and generation parameters.
- Qwen3.5 models use the same 300-token protocol with thinking disabled.

## Results

| 指标 | 目标 | base-3b | KTO-3b | IPO-3b | DPO-3b | base-7b | SFT-7b |
|---|---|---:|---:|---:|---:|---:|---:|
| 互动句占比 | 0.30–0.60 | 0.749 ❌ | 0.750 ❌ | 0.788 ❌ | 0.755 ❌ | 0.691 ❌ | 0.354 ✅ |
| 平均句长 | 15–25 | 12.703 ❌ | 14.550 ❌ | 14.893 ❌ | 12.510 ❌ | 15.170 ✅ | 7.180 ❌ |
| 口语词占比 | ≥0.02 | 0.032 ✅ | 0.032 ✅ | 0.031 ✅ | 0.032 ✅ | 0.050 ✅ | 0.021 ✅ |
| 结构完整度 | ≥4 | 2.767 ❌ | 3.267 ❌ | 3.067 ❌ | 2.600 ❌ | 3.167 ❌ | 2.233 ❌ |
| 卖点覆盖率 | ≥0.80 | 0.700 ❌ | 0.828 ✅ | 0.839 ✅ | 0.767 ❌ | 0.844 ✅ | 0.756 ❌ |
| 幻觉率 | ≤0.10 | 0.117 ❌ | 0.150 ❌ | 0.116 ❌ | 0.036 ✅ | 0.284 ❌ | 0.320 ❌ |
| 情绪递进 | ≥3.5 | 3.113 ❌ | 2.860 ❌ | 3.167 ❌ | 2.570 ❌ | 2.900 ❌ | 2.537 ❌ |
| 口头禅重复度 | ≤0.30 | 0.271 ✅ | 0.178 ✅ | 0.243 ✅ | 0.214 ✅ | 0.149 ✅ | 0.031 ✅ |
| 流畅度 | ≥3.5 | 4.933 ✅ | 4.900 ✅ | 4.967 ✅ | 4.933 ✅ | 4.967 ✅ | 5.000 ✅ |
| **达标数** | 9 | **3/9** | **4/9** | **4/9** | **4/9** | **5/9** | **4/9** |

### Qwen3.5 baselines

| 指标 | 目标 | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---:|---:|
| 互动句占比 | 0.30–0.60 | 0.392 ✅ | 0.483 ✅ |
| 平均句长 | 15–25 | 13.753 ❌ | 13.910 ❌ |
| 口语词占比 | ≥0.02 | 0.045 ✅ | 0.051 ✅ |
| 结构完整度 | ≥4 | 3.200 ❌ | 3.300 ❌ |
| 卖点覆盖率 | ≥0.80 | 0.733 ❌ | 0.728 ❌ |
| 幻觉率 | ≤0.10 | 0.152 ❌ | 0.281 ❌ |
| 情绪递进 | ≥3.5 | 3.440 ❌ | 3.370 ❌ |
| 口头禅重复度 | ≤0.30 | 0.164 ✅ | 0.189 ✅ |
| 流畅度 | ≥3.5 | 5.000 ✅ | 5.000 ✅ |
| **达标数** | 9 | **4/9** | **4/9** |

## Reading

- `base-7b` 是当前最强基线（5/9），但仍未达到 6/9。
- Qwen3.5-4B 和 Qwen3.5-9B 都是 4/9，没有超过 base-7b。9B 的互动风格最好（0.483 达标），但卖点覆盖更低（0.728）、幻觉率更高（0.281）。
- Qwen3.5-4B 的单条冒烟样本曾达到 8/9，但 30 条完整 benchmark 只有 4/9，说明单条结果不能代表整体。
- KTO/IPO/DPO-3b 都是 4/9，仅比 base-3b 多命中 1 项；样本只有 30 条，不能据此证明偏好优化稳定有效。
- DPO-3b 的幻觉率最低（0.036），但结构完整度（2.600）、情绪递进（2.570）和卖点覆盖（0.767）都差。
- SFT-7b 只有 4/9，低于 base-7b；平均句长 7.180、结构 2.233、幻觉率 0.320，当前 SFT 数据和训练目标需要重做。
- 所有模型都未通过结构完整度和情绪递进；除 SFT-7b 外都过度互动。
- 没有模型达到 8/9 或 9/9；旧“8/8”结论不可复现。

## Caveats

- 2026-11-24 的 180-token 报告已作废：30/30 base 输出达到上限，21/90 条出现“用户/助手”角色标签泄漏。当前报告改用 Qwen chat template，正式报告中角色泄漏为 0。
- 30 条 benchmark 与 train/valid/dpo 没有完全相同的 prompt，但有 9/30 个产品也出现在训练数据中（卖点和句式不同）；这是同域测试，不是完全未见产品的泛化测试。
- 平均句长目标与 system prompt 冲突：prompt 要求“每句不超过 15 字”，指标目标却是 15–25 字。该指标目前只能做相对比较，目标值需要重新定义。
- 这是自动指标结果，仍需要人工听感/人工评审补充。

## Reports

- `eval_v2_base-3b_20261125_141655.json` — 3/9
- `eval_v2_kto-3b_20261125_142209.json` — 4/9
- `eval_v2_ipo-3b_20261125_143140.json` — 4/9
- `eval_v2_dpo-3b_20261125_143524.json` — 4/9
- `eval_v2_base-7b_20261125_144233.json` — 5/9
- `eval_v2_sft-7b_20261125_145245.json` — 4/9
- `eval_v2_qwen3.5-4b_20261125_153905.json` — 4/9
- `eval_v2_qwen3.5-9b_20261125_162618.json` — 4/9
