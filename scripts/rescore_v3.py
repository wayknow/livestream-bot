#!/usr/bin/env python3
"""Offline re-score existing v2 generations with metric protocol v3."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pipeline import (  # noqa: E402
    BENCHMARK_PATH,
    METRIC_VERSION,
    PROMPT_VERSION,
    TARGETS,
    LiveStreamEvaluator,
    _aggregate_target_passes,
    _evaluator_sha256,
    _git_commit,
    _git_dirty,
    _public_facts,
    _sha256_file,
    parse_prompt_facts,
)

REPORTS = [
    ("base-3b", "eval_results/eval_v2_base-3b_20261125_141655.json"),
    ("kto-3b", "eval_results/eval_v2_kto-3b_20261125_142209.json"),
    ("ipo-3b", "eval_results/eval_v2_ipo-3b_20261125_143140.json"),
    ("dpo-3b", "eval_results/eval_v2_dpo-3b_20261125_143524.json"),
    ("base-7b", "eval_results/eval_v2_base-7b_20261125_144233.json"),
    ("sft-7b", "eval_results/eval_v2_sft-7b_20261125_145245.json"),
    ("qwen3.5-4b", "eval_results/eval_v2_qwen3.5-4b_20261125_153905.json"),
    ("qwen3.5-9b", "eval_results/eval_v2_qwen3.5-9b_20261125_162618.json"),
]


def rescore_report(evaluator: LiveStreamEvaluator, model_name: str, source_path: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    benchmark = {item["prompt"]: item for item in evaluator.benchmark}
    results = []
    for sample in source["sample_results"]:
        item = benchmark[sample["prompt"]]
        facts = parse_prompt_facts(sample["prompt"], item.get("facts"))
        metrics = evaluator.evaluate_single(sample["output"], sample["prompt"], facts=facts)
        results.append(
            {
                "benchmark_id": sample["benchmark_id"],
                "category": sample["category"],
                "scene": sample["scene"],
                "prompt": sample["prompt"],
                "facts": _public_facts(facts),
                "output": sample["output"],
                "metrics": metrics,
            }
        )

    aggregated = evaluator._aggregate_metrics(results)
    target_checks = {
        key: _aggregate_target_passes(key, aggregated[key])
        for key in TARGETS
        if key in aggregated and "mean" in aggregated[key]
    }
    return {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "metric_version": METRIC_VERSION,
        "prompt_version": source.get("prompt_version", PROMPT_VERSION),
        "system_prompt": source.get("system_prompt"),
        "model_path": source.get("model_path"),
        "adapter_path": source.get("adapter_path"),
        "judge_model": source.get("judge_model"),
        "generation_params": source.get("generation_params"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "evaluator_sha256": _evaluator_sha256(),
        "benchmark_size": len(results),
        "benchmark_sha256": _sha256_file(BENCHMARK_PATH),
        "targets": TARGETS,
        "target_checks": target_checks,
        "sample_results": results,
        "aggregated": aggregated,
        "source_report": str(source_path),
        "source_metric_version": source.get("metric_version"),
        "source_benchmark_sha256": source.get("benchmark_sha256"),
        "rescore_note": "Offline re-score of saved v2 generations with v3 metrics; no model generation was rerun.",
    }


def main() -> int:
    evaluator = LiveStreamEvaluator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports = []
    for model_name, relative_path in REPORTS:
        source_path = ROOT / relative_path
        report = rescore_report(evaluator, model_name, source_path)
        evaluator.save_report(report, filename=f"eval_v3_rescore_{model_name}_{timestamp}.json")
        reports.append(report)

    print(f"\n{'=' * 60}")
    print("v3 离线重算对比（未重新生成）")
    print(f"{'=' * 60}")
    evaluator.print_comparison(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
