# Evaluation reports

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

## Current v2 reports (2026-11-25)

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

Summary and caveats: [`V2_RESULTS.md`](V2_RESULTS.md).
