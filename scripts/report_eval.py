#!/usr/bin/env python3
"""Print a compact markdown summary from one or more v3 evaluation reports."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List


def load_report(path: Path) -> Dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)




def fmt_metric(entry: Dict) -> str:
    mean = entry.get("mean")
    if mean is None:
        return "N/A"
    low = entry.get("ci95_low")
    high = entry.get("ci95_high")
    value = f"{mean:.3f}"
    if low is not None and high is not None:
        value += f" [{low:.3f}, {high:.3f}]"
    return value


def passed(report: Dict, key: str) -> bool:
    return bool(report.get("target_checks", {}).get(key))


def _sample_passes(report: Dict, metrics: Dict, key: str) -> bool:
    spec = report["targets"][key]
    value = metrics.get(key)
    if value is None:
        return False
    lower = spec.get("lower")
    upper = spec.get("upper")
    return (lower is None or value >= lower) and (upper is None or value <= upper)


def diversity_stats(report: Dict) -> Dict[str, object]:
    outputs = [sample["output"] for sample in report.get("sample_results", [])]
    sentences = [
        sentence.strip()
        for output in outputs
        for sentence in re.split(r"[。！？!?\n]+", output)
        if sentence.strip()
    ]
    similarities = [
        SequenceMatcher(None, outputs[left], outputs[right]).ratio()
        for left in range(len(outputs))
        for right in range(left + 1, len(outputs))
    ]
    per_sample_passes = sum(
        all(_sample_passes(report, sample["metrics"], key) for key in report["targets"])
        for sample in report.get("sample_results", [])
    )
    return {
        "sentence_unique": len(set(sentences)),
        "sentence_total": len(sentences),
        "sentence_unique_ratio": len(set(sentences)) / max(len(sentences), 1),
        "pairwise_similarity": sum(similarities) / max(len(similarities), 1),
        "per_sample_passes": per_sample_passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    reports = [load_report(Path(path)) for path in args.reports]
    if not reports:
        return 1

    keys = list(reports[0]["targets"].keys())
    header = "| 指标 | " + " | ".join(report["model_name"] for report in reports) + " | 目标 |"
    sep = "|---|" + "---:|" * len(reports) + "---|"
    print(header)
    print(sep)
    for key in keys:
        spec = reports[0]["targets"][key]
        cells = []
        for report in reports:
            entry = report["aggregated"].get(key, {})
            if entry.get("mean") is None:
                cells.append("N/A")
                continue
            mark = "✅" if passed(report, key) else "❌"
            extra = f" ({entry.get('pass_rate', 0):.0%})" if "pass_rate" in entry else ""
            cells.append(f"{fmt_metric(entry)} {mark}{extra}")
        target = ""
        if "lower" in spec and "upper" in spec:
            target = f"{spec['lower']}–{spec['upper']}"
        elif "lower" in spec:
            target = f">={spec['lower']}"
        elif "upper" in spec:
            target = f"<={spec['upper']}"
        if "min_pass_rate" in spec:
            target += f"; 单条达标率>={spec['min_pass_rate']:.0%}"
        print(f"| {spec['label']} | " + " | ".join(cells) + f" | {target} |")

    print("| **目标命中** | " + " | ".join(
        f"**{sum(report['target_checks'].values())}/{len(keys)}**" for report in reports
    ) + " | |")

    diagnostics = [diversity_stats(report) for report in reports]
    print("| 单条 8/8 | " + " | ".join(
        f"{stats['per_sample_passes']}/{len(report.get('sample_results', []))}"
        for report, stats in zip(reports, diagnostics)
    ) + " | 诊断 |")
    print("| 句级唯一率 | " + " | ".join(
        f"{stats['sentence_unique']}/{stats['sentence_total']} "
        f"({stats['sentence_unique_ratio']:.1%})"
        for stats in diagnostics
    ) + " | 诊断 |")
    print("| 两两相似度 | " + " | ".join(
        f"{stats['pairwise_similarity']:.3f}" for stats in diagnostics
    ) + " | 诊断 |")
    print("| 平均句长 | " + " | ".join(
        f"{report['aggregated'].get('avg_sentence_len', {}).get('mean', 0):.2f}"
        for report in reports
    ) + " | 诊断 |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
