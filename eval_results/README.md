# Evaluation reports

## Current conclusion (2026-11-26)

The authoritative result is now [`V3_SFT_V4_RESULTS.md`](V3_SFT_V4_RESULTS.md):

- `Qwen3.5-4B + SFT-v4` passes **8/8** v3 targets on 30 prompts with seed 0 and seed 1.
- Hallucination rate falls from `0.539` (base) to `0.011` / `0.000`; structure coverage rises from `0.588` to `0.987` / `0.976`.
- Qwen3.5-9B base is `6/8` under the same v3 prompt (`structure=0.657`, `hallucination=0.611`).
- The result is not production-ready: SFT sentence uniqueness is only 46.3% and mean pairwise similarity is 0.582, versus 94.3% and 0.244 for the base model.

Use these four reports as the current comparison set:

- `eval_v3_qwen3.5-4b-base-v3-s0_20261126_160315.json` — 6/8
- `eval_v3_qwen3.5-4b-sft-v3-s0_20261126_160739.json` — 8/8
- `eval_v3_qwen3.5-4b-sft-v3-s1_20261126_162147.json` — 8/8
- `eval_v3_qwen3.5-9b-base-v3-s0_20261126_163503.json` — 6/8

After the stock-unit fix (`最后45条/18张/40袋/20台/70包`), all current reports were re-scored from their saved outputs with `scripts/rescore_saved_reports.py`. Their original `timestamp` and generation parameters are preserved, and `rescored_at` / `source_evaluator_sha256` record the re-score. No model was regenerated.

## v1 reports (invalid for conclusions)

All `eval_*.json` files dated before 2026-11-24 use the old evaluator
(`metric_version` absent). They are kept only for history. Known problems:

- KTO numbers were cherry-picked across different runs.
- The README "DPO" column came from an SFT v7 report, not a DPO model.
- Hallucination only counted negative or `>10000` numbers, so `0.000` was
  almost guaranteed.
- The "LLM judge" fell back to a rule-based scorer.
- Base/SFT/KTO were evaluated with different sample counts and prompts.

Do not use these files to support an 8/8 claim.

## Superseded v2 runs (2026-11-24)

The three `20261124_*` v2 reports used a manual prompt string and
`max_tokens=180`. All 30 base outputs reached the token limit, and 21/90
outputs contained fake `用户/助手` turns. They are retained for history only and
must not be compared with the 2026-11-25 reports.

## v2 generation reports (2026-11-25)

Formal reports use the Qwen chat template, `max_tokens=300`, the same 30-prompt
benchmark and the same sampling parameters. Each report records:

- `metric_version: "2.0"` and `prompt_version: "livestream-v2"`
- `system_prompt` and `generation_params`
- `git_commit`, `git_dirty` and `evaluator_sha256`
- `benchmark_sha256`
- `model_path`, `adapter_path`, `judge_model`
- per-sample `hallucinated_numbers`
- `targets` and `target_checks`

Only compare reports with the same `metric_version`, `prompt_version`,
`benchmark_sha256` and `generation_params`. `scripts/eval_pipeline.py --compare`
prints a warning when any of these differ.

## Current reports

- `eval_v2_base-3b_20261125_141655.json` — 3/9
- `eval_v2_kto-3b_20261125_142209.json` — 4/9
- `eval_v2_ipo-3b_20261125_143140.json` — 4/9
- `eval_v2_dpo-3b_20261125_143524.json` — 4/9
- `eval_v2_base-7b_20261125_144233.json` — 5/9
- `eval_v2_sft-7b_20261125_145245.json` — 4/9
- `eval_v2_qwen3.5-4b_20261125_153905.json` — 4/9
- `eval_v2_qwen3.5-9b_20261125_162618.json` — 4/9

These reports remain the source of the saved model outputs, but their v2 targets included the retired average-sentence-length goal and a numeric-only hallucination check. Use them for generation provenance, not for current target counts.

Summary and caveats: [`V2_RESULTS.md`](V2_RESULTS.md).

## Current v3 reports (offline re-score)

`scripts/rescore_v3.py` re-scores the saved v2 outputs with metric protocol v3. No model was regenerated.

- `metric_version: "3.0"`; source `prompt_version: "livestream-v2"`
- Average sentence length is diagnostic only and is not in `targets` or `target_checks`.
- Style metrics use typed slot normalisation for price/stock values.
- Hallucination uses raw output plus the benchmark `facts` table; missing values are `null` and are never wildcarded for grounding.
- Ten hand-written gold responses in `data/gold_scripts.jsonl` pass 8/8 targets.

Reports:

- `eval_v3_rescore_base-3b_20261126_152359.json` — 4/8
- `eval_v3_rescore_kto-3b_20261126_152359.json` — 4/8
- `eval_v3_rescore_ipo-3b_20261126_152359.json` — 4/8
- `eval_v3_rescore_dpo-3b_20261126_152359.json` — 4/8
- `eval_v3_rescore_base-7b_20261126_152359.json` — 4/8
- `eval_v3_rescore_sft-7b_20261126_152359.json` — 4/8
- `eval_v3_rescore_qwen3.5-4b_20261126_152359.json` — 6/8
- `eval_v3_rescore_qwen3.5-9b_20261126_152359.json` — 6/8

This is the latest offline re-score: after fixing decimal prices, discourse numbers, conditional structure and claim matching, Qwen3.5-4B and 9B are both 6/8. These files still use `prompt_version=livestream-v2` and must not be mixed with the new v3 generation reports.

## SFT-v4 checkpoint reports

- `eval_v3_qwen3.5-4b-sft200-v3-s0_20261126_161232.json` — 8/8
- `eval_v3_qwen3.5-4b-sft300-v3-s0_20261126_161657.json` — 8/8
- final `eval_v3_qwen3.5-4b-sft-v3-s0_20261126_160739.json` — 8/8, selected for lower hallucination and pairwise similarity

Summary: [`V3_RESCORE.md`](V3_RESCORE.md).
