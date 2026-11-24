# Protocol v2 results (2026-11-24)

## Scope

- Benchmark: `data/benchmark_prompts.jsonl` (30 prompts)
- Metric version: `2.0`; prompt version: `livestream-v2`
- Generation: `temp=0.7`, `top_p=0.9`, `max_tokens=180`
- Compared models: `base-3b`, `kto-3b` (fused from TRL LoRA), `ipo-3b` (fused from TRL LoRA)

All three reports use the same prompt, benchmark hash and metric version.

## Results

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

## Reading

- 真实重训后的 KTO **没有**优于 3B 基座：目标命中 3/9 vs 4/9，幻觉率和卖点覆盖更差。
- IPO 命中 5/9，略高于基座，但结构、卖点、情绪仍未达标。
- 三个 3B 模型都严重过度互动（0.78–0.81，目标上限 0.60）且句长偏长。
- 没有模型达到 8/9 或 9/9；旧“8/8”结论不可复现。

## Caveats

- `max_tokens=180` 是为了控制评测时间，可能截断长输出；正式最终结果建议用 `max_tokens=300` 重跑。
- 30 条 benchmark 与 train/valid/dpo 没有完全相同的 prompt，但有 9/30 个产品也出现在训练数据中（卖点和句式不同）；因此这是同域测试，不是完全未见产品的泛化测试。
- 尚未评估 `base-7b`、`sft-7b`、`dpo-3b`。
- 这是自动指标结果，仍需要人工听感/人工评审补充。

## Reports

- `eval_v2_base-3b_20261124_164739.json`
- `eval_v2_kto-3b_20261124_165856.json`
- `eval_v2_ipo-3b_20261124_170453.json`
