#!/usr/bin/env python3
"""Livestream script evaluation pipeline, protocol v2.

Changes from the historical v1 evaluator:
- one frozen system prompt for every model and every sample;
- interaction density is the fraction of sentences containing an interactive
  cue, bounded to [0, 1];
- hallucination detection flags any number that is not grounded in the prompt;
- fluency is explicitly reported as rule-based unless --judge-model is given;
- reports carry metric_version, prompt_version, git commit and model paths so
  numbers from different runs cannot be mixed silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_prompts.jsonl"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"

METRIC_VERSION = "2.0"
PROMPT_VERSION = "livestream-v2"

SYSTEM_PROMPT = """你是一位抖音直播带货话术专家。你的风格特征：
①短句为主，每句不超过15字
②每3句话插入一次互动指令（"扣1"、"对不对"、"听我说"）
③先讲痛点场景，再给解决方案，最后价格逼单
④情绪递进：平静→兴奋→急促
⑤口头禅：家人们、真的绝了、最后一波、听我说、有没有
⑥价格、数量、时间等具体信息只能来自用户输入；用户没给就不要编造。"""

INTERACTIVE_WORDS = [
    "扣1", "扣个1", "有没有", "想要的", "打想要", "对不对", "是不是",
    "让我看看", "姐妹们", "家人们", "听我说", "看过来",
]

SPOKEN_WORDS = [
    "了", "呢", "吧", "啊", "啦", "嘛", "我跟你说", "真的", "绝了",
    "好用到哭", "太香了", "真的绝", "绝绝子", "yyds", "nice",
]

STRUCTURE_PATTERNS = {
    "痛点场景": r"有没有发现|是不是|最怕|烦恼|困扰|痛点|崩溃|尴尬|难看|不舒服|问题",
    "解决方案": r"解决|帮你|这款|这个产品|采用|配方|设计|技术|材质|成分|可以",
    "价格锚定": r"原价|平时|专柜|日常价|直播间|到手价|只要|仅需|优惠|划算|性价比",
    "限时福利": r"最后|限量|库存|卖完|补货|倒计时|前\d+单|仅剩",
    "行动指令": r"下单|拍|抢|冲|链接|购物车|赶紧|立刻|马上",
}

EMOTION_WORDS = {
    "平静": ["来", "今天", "这个", "大家好", "产品", "卖点"],
    "兴奋": ["绝了", "真的", "直接", "太香了", "好用到哭", "绝绝子", "yyds"],
    "急促": ["最后", "快", "上链接", "冲", "抢", "赶紧", "倒计时", "321", "3、2、1"],
}

INTERACTION_SAFE_NUMBERS = {1, 2, 3, 6, 8, 9, 520, 666, 888, 1314}

TARGETS = {
    "interactive_sentence_ratio": {"label": "互动句占比", "lower": 0.30, "upper": 0.60},
    "avg_sentence_len": {"label": "平均句长(字)", "lower": 15.0, "upper": 25.0},
    "spoken_ratio": {"label": "口语词占比", "lower": 0.02},
    "structure_completeness": {"label": "结构完整度(0-5)", "lower": 4.0},
    "selling_point_coverage": {"label": "卖点覆盖率", "lower": 0.80},
    "hallucination_rate": {"label": "幻觉率", "upper": 0.10},
    "emotion_progression": {"label": "情绪递进(0-5)", "lower": 3.5},
    "catchphrase_repeat": {"label": "口头禅重复度", "upper": 0.30},
    "fluency_score": {"label": "流畅度(0-5)", "lower": 3.5},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return True


def _evaluator_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _clean_generation(text: str) -> str:
    """Keep only the assistant turn if the model starts a fake next turn."""
    for marker in ("\n用户：", "\n助手：", "\n<|im_start|>", "<|im_end|>"):
        text = text.split(marker, 1)[0]
    return text.strip()


def _target_passes(key: str, value: float) -> bool:
    spec = TARGETS.get(key)
    if spec is None:
        return True
    if "lower" in spec and value < spec["lower"]:
        return False
    if "upper" in spec and value > spec["upper"]:
        return False
    return True


class LiveStreamEvaluator:
    """直播话术评估器"""

    def __init__(self, benchmark_path: str | Path | None = None):
        self.benchmark_path = Path(benchmark_path) if benchmark_path else BENCHMARK_PATH
        self.benchmark = self._load_benchmark()
        EVAL_RESULTS_DIR.mkdir(exist_ok=True)

    def _load_benchmark(self) -> List[Dict]:
        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark 文件不存在: {self.benchmark_path}")
        with open(self.benchmark_path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"[。！？!?\n]+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def hard_metrics(self, text: str, prompt: str) -> Dict:
        sentences = self._split_sentences(text)
        total_chars = len(text)
        total_sentences = max(len(sentences), 1)

        interactive_sentences = sum(
            1 for sentence in sentences if any(word in sentence for word in INTERACTIVE_WORDS)
        )
        interactive_ratio = round(interactive_sentences / total_sentences, 3)

        avg_sentence_len = round(total_chars / total_sentences, 1)
        spoken_count = sum(text.count(word) for word in SPOKEN_WORDS)
        spoken_ratio = round(spoken_count / max(total_chars, 1), 4)

        structure_score = sum(
            1 for pattern in STRUCTURE_PATTERNS.values() if re.search(pattern, text)
        )

        selling_points: List[str] = []
        if "卖点：" in prompt:
            part = prompt.split("卖点：", 1)[1].split("；", 1)[0]
            selling_points = [item.strip() for item in part.replace("、", ",").split(",") if item.strip()]
        normalized_text = _normalize(text)
        covered = sum(1 for point in selling_points if _normalize(point) in normalized_text)
        selling_coverage = round(covered / max(len(selling_points), 1), 3)

        output_numbers = sorted(set(re.findall(r"\d+(?:\.\d+)?", text)))
        prompt_numbers = set(re.findall(r"\d+(?:\.\d+)?", prompt))
        hallucinated: List[str] = []
        for number in output_numbers:
            if number in prompt_numbers:
                continue
            try:
                value = float(number)
            except ValueError:
                continue
            if value <= 3 or int(value) in INTERACTION_SAFE_NUMBERS:
                continue
            hallucinated.append(number)
        hallucination_rate = round(len(hallucinated) / max(len(output_numbers), 1), 3)

        return {
            "interactive_sentence_ratio": interactive_ratio,
            "avg_sentence_len": avg_sentence_len,
            "spoken_ratio": spoken_ratio,
            "structure_completeness": structure_score,
            "selling_point_coverage": selling_coverage,
            "hallucination_rate": hallucination_rate,
            "hallucinated_numbers": hallucinated,
            "emotion_progression": self._emotion_progression_score(text),
            "catchphrase_repeat": self._catchphrase_repeat(text, total_sentences),
            "sentence_count": total_sentences,
            "char_count": total_chars,
        }

    @staticmethod
    def _catchphrase_repeat(text: str, total_sentences: int) -> float:
        catchphrases = ["家人们", "我跟你说", "真的绝了", "绝绝子", "好用到哭"]
        max_repeat = max((text.count(phrase) for phrase in catchphrases), default=0)
        return round(max_repeat / max(total_sentences, 1), 3)

    @staticmethod
    def _emotion_progression_score(text: str) -> float:
        sentences = LiveStreamEvaluator._split_sentences(text)
        if len(sentences) < 3:
            return 2.0

        sequence = []
        for sentence in sentences:
            if any(word in sentence for word in EMOTION_WORDS["急促"]):
                sequence.append(2)
            elif any(word in sentence for word in EMOTION_WORDS["兴奋"]):
                sequence.append(1)
            elif any(word in sentence for word in EMOTION_WORDS["平静"]):
                sequence.append(0)
            else:
                sequence.append(-1)

        valid = [value for value in sequence if value >= 0]
        if len(valid) < 3:
            return 2.0

        increases = sum(1 for left, right in zip(valid, valid[1:]) if right > left)
        decreases = sum(1 for left, right in zip(valid, valid[1:]) if right < left)
        score = 2.0 + 1.2 * increases - 0.8 * decreases
        if valid[0] == 0:
            score += 0.2
        if valid[-1] == 2:
            score += 0.3
        return round(min(5.0, max(0.0, score)), 1)

    def soft_metrics_llm(self, text: str, model_path: str | None = None) -> Dict:
        """Use a local MLX model as judge when requested; otherwise use rules."""
        if model_path:
            judge_prompt = f"""你是一个直播话术评估专家。请分析以下话术的流畅度。

只输出 JSON，不要其他内容：
{{"fluency_score": 0-5的整数, "fluency_reason": "一句话说明"}}

【话术】{text}
"""
            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "mlx_lm", "generate",
                        "--model", model_path,
                        "--prompt", judge_prompt,
                        "--max-tokens", "100",
                        "--temp", "0.1",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                match = re.search(r"\{[^}]+\}", result.stdout)
                if match:
                    payload = json.loads(match.group())
                    payload["fluency_method"] = "llm"
                    return payload
            except Exception:
                pass

        return self._estimate_fluency(text)

    @staticmethod
    def _estimate_fluency(text: str) -> Dict:
        sentences = LiveStreamEvaluator._split_sentences(text)
        score = 5
        long_sentences = [sentence for sentence in sentences if len(sentence) > 30]
        long_ratio = len(long_sentences) / max(len(sentences), 1)

        if long_ratio > 0.4:
            score -= 1
        if any(len(sentence) > 50 for sentence in sentences):
            score -= 1
        if any(re.search(r"[\u4e00-\u9fff]{25,}", sentence) for sentence in sentences):
            score -= 1
        if len(sentences) < 3:
            score -= 1

        return {
            "fluency_score": max(score, 1),
            "fluency_reason": f"长句数: {len(long_sentences)}/{len(sentences)} ({long_ratio:.0%})",
            "fluency_method": "rule",
        }

    def evaluate_single(self, text: str, prompt: str, judge_model: str | None = None) -> Dict:
        return {**self.hard_metrics(text, prompt), **self.soft_metrics_llm(text, judge_model)}

    def evaluate_model(
        self,
        generate_func,
        model_name: str = "model",
        num_samples: int | None = None,
        model_path: str | None = None,
        adapter_path: str | None = None,
        judge_model: str | None = None,
        generation_params: Dict | None = None,
    ) -> Dict:
        benchmark = self.benchmark[:num_samples] if num_samples else self.benchmark
        print(f"\n{'=' * 60}")
        print(f"开始评估: {model_name}")
        print(f"测试集大小: {len(benchmark)} 条 | metric={METRIC_VERSION} prompt={PROMPT_VERSION}")
        print(f"{'=' * 60}\n")

        results = []
        for index, item in enumerate(benchmark, start=1):
            prompt = item["prompt"]
            print(f"[{index}/{len(benchmark)}] {item['id']} - {item['category']}/{item['scene']}")
            try:
                output = generate_func(prompt)
            except Exception as exc:
                print(f"  ❌ 生成失败: {exc}")
                output = ""

            metrics = self.evaluate_single(output, prompt, judge_model)
            results.append({
                "benchmark_id": item["id"],
                "category": item["category"],
                "scene": item["scene"],
                "prompt": prompt,
                "output": output,
                "metrics": metrics,
            })
            print(
                f"  互动句占比: {metrics['interactive_sentence_ratio']:.2f} | "
                f"句长: {metrics['avg_sentence_len']:.1f} | "
                f"幻觉率: {metrics['hallucination_rate']:.2f} | "
                f"结构: {metrics['structure_completeness']}/5"
            )

        aggregated = self._aggregate_metrics(results)
        target_checks = {
            key: _target_passes(key, aggregated[key]["mean"])
            for key in TARGETS
            if key in aggregated and "mean" in aggregated[key]
        }
        return {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metric_version": METRIC_VERSION,
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "model_path": model_path,
            "adapter_path": adapter_path,
            "judge_model": judge_model,
            "generation_params": generation_params,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "evaluator_sha256": _evaluator_sha256(),
            "benchmark_size": len(benchmark),
            "benchmark_sha256": _sha256_file(self.benchmark_path),
            "targets": TARGETS,
            "target_checks": target_checks,
            "sample_results": results,
            "aggregated": aggregated,
        }

    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        all_metrics = [result["metrics"] for result in results]
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [metrics[key] for metrics in all_metrics]
            if isinstance(values[0], (int, float)):
                aggregated[key] = {
                    "mean": round(float(np.mean(values)), 3),
                    "std": round(float(np.std(values)), 3),
                    "min": round(float(np.min(values)), 3),
                    "max": round(float(np.max(values)), 3),
                }
            elif isinstance(values[0], str):
                mode = Counter(values).most_common(1)[0][0]
                aggregated[key] = {"mode": mode, "unique_count": len(set(values))}
        return aggregated

    def save_report(self, report: Dict, filename: str | None = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = report["model_name"].replace("/", "_")
            filename = f"eval_v2_{model_name}_{timestamp}.json"
        filepath = EVAL_RESULTS_DIR / filename
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\n✅ 报告已保存: {filepath}")
        return str(filepath)

    def print_comparison(self, reports: List[Dict]) -> str:
        if len(reports) < 2:
            print("需要至少 2 个报告才能对比")
            return ""

        versions = {report.get("metric_version", "v1") for report in reports}
        if versions != {METRIC_VERSION}:
            print(
                f"⚠️ 报告指标版本为 {sorted(versions)}，不是当前 {METRIC_VERSION}；"
                "以下对比仅供排查，不能作为结论。"
            )

        prompt_versions = {report.get("prompt_version", "unknown") for report in reports}
        if len(prompt_versions) > 1:
            print(f"⚠️ 报告 prompt 版本不一致: {sorted(prompt_versions)}；不能作为结论。")

        benchmark_hashes = {report.get("benchmark_sha256", "unknown") for report in reports}
        if len(benchmark_hashes) > 1:
            print("⚠️ 报告 benchmark 哈希不一致；不能作为结论。")

        generation_params = {
            json.dumps(report.get("generation_params") or {}, sort_keys=True)
            for report in reports
        }
        if len(generation_params) > 1:
            print("⚠️ 报告生成参数不一致（如 max_tokens/temp/top_p）；不能作为结论。")

        header = f"{'指标':<20} |"
        for report in reports:
            header += f" {report['model_name'][:15]:^15} |"
        header += f" {'目标':^12} |"
        rows = [header, "-" * len(header)]

        for key, spec in TARGETS.items():
            row = f"{spec['label']:<20} |"
            for report in reports:
                aggregated = report.get("aggregated", {})
                value = aggregated.get(key, {}).get("mean")
                if value is None and key == "interactive_sentence_ratio":
                    value = aggregated.get("interactive_density", {}).get("mean")
                row += f" {value if value is not None else 'N/A':^15} |"
            target = f">={spec['lower']}" if "lower" in spec else f"<={spec['upper']}"
            if "lower" in spec and "upper" in spec:
                target = f"{spec['lower']}-{spec['upper']}"
            row += f" {target:^12} |"
            rows.append(row)

        table = "\n".join(rows)
        print(table)
        return table


def create_mlx_generator(
    model_path: str,
    adapter_path: str | None = None,
    temp: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 300,
):
    """Load an MLX model once and return a prompt -> text function."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    sampler = make_sampler(temp=temp, top_p=top_p)

    def generate_text(prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        output = generate(
            model,
            tokenizer,
            prompt=full_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return _clean_generation(output)

    return generate_text


def main() -> int:
    parser = argparse.ArgumentParser(description="直播话术评估体系 v2")
    parser.add_argument("--model", help="基座模型路径")
    parser.add_argument("--adapter", help="LoRA adapter 路径")
    parser.add_argument("--name", default="model", help="模型名称")
    parser.add_argument("--samples", type=int, help="评估样本数（默认全部 30 条）")
    parser.add_argument("--save", action="store_true", help="保存报告到文件")
    parser.add_argument("--compare", nargs="+", help="对比多个报告文件")
    parser.add_argument("--judge-model", help="可选：本地模型路径，用于 LLM 流畅度裁判")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    evaluator = LiveStreamEvaluator()
    if args.compare:
        reports = []
        for filepath in args.compare:
            with open(filepath, encoding="utf-8") as handle:
                reports.append(json.load(handle))
        evaluator.print_comparison(reports)
        return 0

    if not args.model:
        parser.error("--model is required unless --compare is used")

    generate_func = create_mlx_generator(
        args.model,
        adapter_path=args.adapter,
        temp=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    report = evaluator.evaluate_model(
        generate_func,
        model_name=args.name,
        num_samples=args.samples,
        model_path=args.model,
        adapter_path=args.adapter,
        judge_model=args.judge_model,
        generation_params={"max_tokens": args.max_tokens, "temp": args.temp, "top_p": args.top_p},
    )

    print(f"\n{'=' * 60}")
    print("评估完成")
    print(f"{'=' * 60}")
    for key, spec in TARGETS.items():
        aggregated = report["aggregated"].get(key)
        if not aggregated or "mean" not in aggregated:
            continue
        status = "PASS" if report["target_checks"].get(key) else "FAIL"
        print(f"{spec['label']}: {aggregated['mean']:.3f} [{status}]")
    passed = sum(report["target_checks"].values())
    print(f"targets: {passed}/{len(report['target_checks'])}")

    if args.save:
        evaluator.save_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
