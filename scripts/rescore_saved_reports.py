#!/usr/bin/env python3
"""Re-score saved generations with the current evaluator, without rerunning models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pipeline import (  # noqa: E402
    BENCHMARK_PATH,
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


def rescore(path: Path, evaluator: LiveStreamEvaluator) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for sample in source["sample_results"]:
        facts = parse_prompt_facts(sample["prompt"], sample.get("facts"))
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
        **source,
        "timestamp": source.get("timestamp"),
        "rescored_at": datetime.now().isoformat(),
        "source_git_commit": source.get("git_commit"),
        "source_evaluator_sha256": source.get("evaluator_sha256"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "evaluator_sha256": _evaluator_sha256(),
        "benchmark_sha256": _sha256_file(BENCHMARK_PATH),
        "targets": TARGETS,
        "target_checks": target_checks,
        "sample_results": results,
        "aggregated": aggregated,
        "rescore_note": "Re-scored from saved outputs with the current evaluator; no model generation was rerun.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Saved evaluation JSON reports")
    args = parser.parse_args()
    evaluator = LiveStreamEvaluator()
    for raw_path in args.reports:
        path = Path(raw_path)
        report = rescore(path, evaluator)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        passed = sum(report["target_checks"].values())
        print(f"{path}: {passed}/{len(report['target_checks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
