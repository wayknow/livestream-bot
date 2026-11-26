# Protocol v3 offline re-score (2026-11-26)

## Scope

This is an **offline re-score** of the eight v2 generations. No model was re-generated. The prompt strings are unchanged (`prompt_version=livestream-v2`); the benchmark carries an explicit `facts` object with `null` for missing values. This supersedes the earlier 2026-11-25 re-score after fixes to decimal prices, discourse-number false positives, required-structure selection and claim matching.

- Metric version: `3.0`
- Benchmark: `data/benchmark_prompts.jsonl` (30 prompts)
- Source reports: `eval_v2_*_20261125_*.json`
- Generation parameters: unchanged from v2 (`temp=0.7`, `top_p=0.9`, `max_tokens=300`)
- Gold set: `data/gold_scripts.jsonl` (10 hand-written responses; 8/8 targets pass)

The old v2 sentence-length target is retired. `avg_sentence_len` is still recorded as a diagnostic, but it is not included in `targets` or `target_checks`.

## Target definitions

| Metric | Definition | Target |
|---|---|---|
| Interaction ratio | Sentences with an interaction cue / total sentences | 0.30–0.60 |
| Spoken ratio | Spoken-word occurrences / characters | ≥0.02 |
| Structure | Only components actually supported by the prompt: pain / solution / CTA, plus price when provided and urgency when stock or a limited offer is provided | ≥0.75 |
| Selling-point coverage | Canonically matched prompt selling points | ≥0.80 |
| Hallucination | Unsupported or contradicted claims / unique claims | ≤0.10 |
| Emotion progression | Calm → excited → urgent | Per-sample ≥3.5 for ≥70% samples |
| Catchphrase repetition | Most frequent catchphrase count / sentences | ≤0.30 |
| Fluency | Rule-based fluency score | ≥3.5 |

Style metrics use `normalize_slots()` to ignore price/stock values and normalise Chinese numerals and units. Grounding always uses the raw output and the fact table; values are **not** wildcarded for hallucination.

## Results

| Metric | base-3b | KTO-3b | IPO-3b | DPO-3b | base-7b | SFT-7b | Qwen3.5-4B | Qwen3.5-9B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Interaction ratio | 0.749 ❌ | 0.750 ❌ | 0.788 ❌ | 0.755 ❌ | 0.691 ❌ | 0.354 ✅ | 0.454 ✅ | 0.527 ✅ |
| Spoken ratio | 0.032 ✅ | 0.032 ✅ | 0.031 ✅ | 0.032 ✅ | 0.050 ✅ | 0.021 ✅ | 0.045 ✅ | 0.051 ✅ |
| Structure | 0.537 ❌ | 0.632 ❌ | 0.622 ❌ | 0.470 ❌ | 0.583 ❌ | 0.386 ❌ | 0.657 ❌ | 0.619 ❌ |
| Selling-point coverage | 0.811 ✅ | 0.883 ✅ | 0.889 ✅ | 0.872 ✅ | 0.900 ✅ | 0.756 ❌ | 0.833 ✅ | 0.856 ✅ |
| Hallucination | 0.675 ❌ | 0.673 ❌ | 0.658 ❌ | 0.587 ❌ | 0.660 ❌ | 0.761 ❌ | 0.604 ❌ | 0.651 ❌ |
| Emotion progression (mean) | 2.917 | 2.677 | 2.717 | 2.447 | 2.600 | 2.517 | 3.410 | 3.337 |
| Emotion pass rate | 47% ❌ | 33% ❌ | 37% ❌ | 23% ❌ | 37% ❌ | 10% ❌ | 70% ✅ | 77% ✅ |
| Catchphrase repetition | 0.271 ✅ | 0.178 ✅ | 0.243 ✅ | 0.214 ✅ | 0.149 ✅ | 0.031 ✅ | 0.164 ✅ | 0.189 ✅ |
| Fluency | 4.933 ✅ | 4.900 ✅ | 4.967 ✅ | 4.933 ✅ | 4.967 ✅ | 5.000 ✅ | 5.000 ✅ | 5.000 ✅ |
| **Targets passed** | **4/8** | **4/8** | **4/8** | **4/8** | **4/8** | **4/8** | **6/8** | **6/8** |
| Claim support rate | 25.9% | 26.1% | 27.5% | 28.0% | 27.3% | 17.2% | 36.2% | 31.6% |
| Avg sentence length (diagnostic) | 12.703 | 14.550 | 14.893 | 12.510 | 15.170 | 7.180 | 13.753 | 13.910 |

## Reading

- `Qwen3.5-4B` and `Qwen3.5-9B` are tied at 6/8. Both fail structure and factual grounding; 9B has the slightly better interaction ratio and 4B the better hallucination rate.
- All eight models still fail structure and factual grounding. The best structure coverage is 0.657; the best hallucination rate is 0.587.
- Claim support is low: only 2–9 of 30 samples per model contain no unsupported or contradicted claim. The main failures are invented price and stock claims (`最后一波`, `500单`, `一杯奶茶钱`, `99元`).
- Canonical selling-point matching fixes the previous exact-string false negatives. Selling-point coverage now passes for 7/8 models; SFT-7B remains below target at 0.756.
- Qwen3.5-4B passes the emotion target at exactly 21/30 samples (70%); 9B reaches 23/30 (77%). The mean alone would incorrectly fail 9B, so the per-sample pass-rate rule matters.
- SFT-7B remains a clear regression: structure 2.233, selling points 0.756, hallucination 0.744, emotion pass rate 13%.

## Caveats

- This is a re-score of saved v2 outputs, not a new generation run. Do not call it a new model benchmark without regenerating under v3.
- The fact table makes missing price/stock explicit. A model that says `价格以直播间为准` is not penalised; a model that invents `99元` or `最后50单` is.
- The high hallucination rates are partly the result of stricter claim detection: generic scarcity phrases are now claims too. This is intentional for production livestream scripts, but it means v3 scores are not directly comparable to v2 scores.
- The 30-prompt benchmark is still a same-domain test. Product-level generalisation and human listening tests remain required.

## Reports

- `eval_v3_rescore_base-3b_20261126_152359.json` — 4/8
- `eval_v3_rescore_kto-3b_20261126_152359.json` — 4/8
- `eval_v3_rescore_ipo-3b_20261126_152359.json` — 4/8
- `eval_v3_rescore_dpo-3b_20261126_152359.json` — 4/8
- `eval_v3_rescore_base-7b_20261126_152359.json` — 4/8
- `eval_v3_rescore_sft-7b_20261126_152359.json` — 4/8
- `eval_v3_rescore_qwen3.5-4b_20261126_152359.json` — 6/8
- `eval_v3_rescore_qwen3.5-9b_20261126_152359.json` — 6/8
