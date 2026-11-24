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

## v2 reports

New reports are written as `eval_v2_<model>_<timestamp>.json` and include:

- `metric_version: "2.0"`
- `prompt_version`
- `system_prompt`
- `git_commit`
- `git_dirty` and `evaluator_sha256` (new reports; the 2026-11-24 reports predate this field)
- `benchmark_sha256`
- `model_path`, `adapter_path`, `judge_model`
- per-sample `hallucinated_numbers`
- `targets` and `target_checks`

Only compare reports with the same `metric_version`, `prompt_version` and
`benchmark_sha256`. Use `scripts/eval_pipeline.py --compare ...`; it prints a
warning when legacy reports are mixed in.

## Current v2 reports

- `eval_v2_base-3b_20261124_164739.json` — 4/9 targets
- `eval_v2_kto-3b_20261124_165856.json` — 3/9 targets
- `eval_v2_ipo-3b_20261124_170453.json` — 5/9 targets

Summary and caveats: [`V2_RESULTS.md`](V2_RESULTS.md).
