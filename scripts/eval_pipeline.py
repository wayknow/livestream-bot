#!/usr/bin/env python3
"""
直播话术评估体系 - 完整可运行版本
包含：8个硬指标（规则引擎）+ 2个软指标（LLM裁判）
"""

import json
import re
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import subprocess

# ========== 配置 ==========
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_prompts.jsonl"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"

# 固定词库（永不更改）
INTERACTIVE_WORDS = [
    "扣1", "扣个1", "有没有", "想要的", "打想要", "对不对", "是不是",
    "让我看看", "姐妹们", "家人们", "听我说", "来", "看过来"
]

SPOKEN_WORDS = [
    "了", "呢", "吧", "啊", "啦", "嘛", "我跟你说", "真的", "绝了",
    "好用到哭", "太香了", "真的绝", "绝绝子", "yyds", "nice"
]

STRUCTURE_KEYWORDS = {
    "痛点场景": [
        "有没有发现", "是不是", "困扰", "烦恼", "痛点", "最怕",
        "崩溃", "烦", "焦虑", "担心", "问题"
    ],
    "解决方案": [
        "解决", "帮你", "只要", "采用", "有效", "能", "可以",
        "这个", "今天", "推荐"
    ],
    "价格锚定": [
        "原价", "平时", "专柜", "今天", "直播间", "只要",
        "便宜", "划算", "性价比", "买", "送"
    ],
    "限时福利": [
        "前", "单", "限量", "最后一波", "倒计时", "最后",
        "库存", "告急", "卖完", "补不到"
    ],
    "行动指令": [
        "3", "2", "1", "上链接", "去拍", "赶紧", "快",
        "冲", "抢", "下单"
    ]
}

# 情绪词库
EMOTION_WORDS = {
    "平静": ["来", "今天", "这个", "大家好", "产品", "卖点"],
    "兴奋": ["绝了", "真的", "直接", "太香了", "好用到哭", "绝绝子", "yyds"],
    "急促": ["最后", "快", "3", "2", "1", "上链接", "冲", "抢", "赶紧", "倒计时"]
}

# 常见数字（用于幻觉检测）
COMMON_NUMBERS = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "100", "200", "300", "500", "1000", "2000", "3000", "5000",
    "10000", "20000", "30", "60", "90", "120", "180", "360"
}

# 时间相关的数字模式（不计入幻觉）
TIME_PATTERNS = [
    r'\d+秒', r'\d+分钟', r'\d+小时', r'\d+天', r'\d+周', r'\d+月',
    r'第\d+', r'前\d+', r'后\d+'
]


class LiveStreamEvaluator:
    """直播话术评估器"""

    def __init__(self, benchmark_path: str = None):
        """初始化评估器"""
        self.benchmark_path = benchmark_path or BENCHMARK_PATH
        self.benchmark = self._load_benchmark()
        EVAL_RESULTS_DIR.mkdir(exist_ok=True)

    def _load_benchmark(self) -> List[Dict]:
        """加载固定 benchmark 测试集"""
        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark 文件不存在: {self.benchmark_path}")

        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # ========== 硬指标（规则引擎，100% 稳定） ==========

    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        sentences = re.split(r'[。！？\n]', text)
        return [s.strip() for s in sentences if s.strip()]

    def hard_metrics(self, text: str, prompt: str) -> Dict:
        """计算所有硬指标"""
        sentences = self._split_sentences(text)
        total_chars = len(text)
        total_sentences = max(len(sentences), 1)

        # 1. 互动指令密度
        inter_count = sum(1 for w in INTERACTIVE_WORDS if w in text)
        interactive_density = round(inter_count / total_sentences, 3)

        # 2. 平均句长
        avg_sentence_len = round(total_chars / total_sentences, 1)

        # 3. 口语词占比
        spoken_count = sum(text.count(w) for w in SPOKEN_WORDS)
        spoken_ratio = round(spoken_count / max(total_chars, 1), 4)

        # 4. 结构完整度（0-5）
        structure_score = 0
        matched_structure = []
        for dim, keywords in STRUCTURE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                structure_score += 1
                matched_structure.append(dim)

        # 5. 卖点覆盖率
        # 从 prompt 提取卖点（假设卖点在"卖点："之后）
        prompt_selling_points = []
        if "卖点：" in prompt:
            sp_part = prompt.split("卖点：")[1].split("；")[0].split("；")[0]
            prompt_selling_points = [s.strip() for s in sp_part.replace("、", ",").split(",") if s.strip()]

        covered_points = sum(1 for sp in prompt_selling_points if sp in text)
        selling_point_coverage = round(covered_points / max(len(prompt_selling_points), 1), 3)

        # 6. 幻觉率（排除时间相关的数字）
        nums_in_output = set(re.findall(r'\d+', text))
        # 从 prompt 提取所有数字
        nums_in_prompt = set(re.findall(r'\d+', prompt))
        # 排除时间相关的数字（如"15秒"、"30分钟"等）
        time_related_nums = set()
        for pattern in TIME_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                num = re.findall(r'\d+', match)
                if num:
                    time_related_nums.update(num)
        # 检查不在 prompt 中且不是常见数字且不是时间相关数字的
        hallucinated = nums_in_output - nums_in_prompt - COMMON_NUMBERS - time_related_nums
        hallucination_rate = round(len(hallucinated) / max(len(nums_in_output), 1), 3)

        # 7. 情绪词分布（递进检测）
        emotion_score = self._emotion_progression_score(text)

        # 8. 口头禅重复度
        catchphrases = ["家人们", "我跟你说", "真的绝了", "绝绝子", "好用到哭"]
        repeat_counts = [text.count(cp) for cp in catchphrases]
        max_repeat = max(repeat_counts) if repeat_counts else 0
        catchphrase_repeat = round(max_repeat / total_sentences, 3)

        return {
            "interactive_density": interactive_density,
            "avg_sentence_len": avg_sentence_len,
            "spoken_ratio": spoken_ratio,
            "structure_completeness": structure_score,
            "selling_point_coverage": selling_point_coverage,
            "hallucination_rate": hallucination_rate,
            "emotion_progression": emotion_score,
            "catchphrase_repeat": catchphrase_repeat,
            "sentence_count": total_sentences,
            "char_count": total_chars
        }

    def _emotion_progression_score(self, text: str) -> float:
        """检测情绪递进（0-5分）"""
        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            return 2.0  # 句子太少，无法判断

        # 给每个句子打情绪标签
        emotion_seq = []
        for sent in sentences:
            has_calm = any(w in sent for w in EMOTION_WORDS["平静"])
            has_excited = any(w in sent for w in EMOTION_WORDS["兴奋"])
            has_urgent = any(w in sent for w in EMOTION_WORDS["急促"])

            if has_urgent:
                emotion_seq.append(2)  # 急促
            elif has_excited:
                emotion_seq.append(1)  # 兴奋
            elif has_calm:
                emotion_seq.append(0)  # 平静
            else:
                emotion_seq.append(-1)  # 无情绪

        # 过滤掉无情绪的句子
        valid_seq = [e for e in emotion_seq if e >= 0]
        if len(valid_seq) < 3:
            return 2.0

        # 计算递进得分
        score = 0
        for i in range(1, len(valid_seq)):
            if valid_seq[i] >= valid_seq[i-1]:
                score += 1  # 情绪递进或持平

        # 归一化到 0-5
        progression_ratio = score / max(len(valid_seq) - 1, 1)
        return round(progression_ratio * 5, 1)

    # ========== 软指标（LLM 裁判，结构化 prompt） ==========

    def soft_metrics_llm(self, text: str, model_path: str = None) -> Dict:
        """
        用本地模型做 LLM 裁判
        如果无模型，返回模拟值
        """
        # 构造裁判 prompt
        judge_prompt = f"""你是一个直播话术评估专家。请分析以下话术的流畅度。

评估标准：
1. 句子是否通顺，无语法错误
2. 是否口语化，朗读是否流畅
3. 有无生僻词或拗口表达

只输出 JSON，不要其他内容：
{{
  "fluency_score": 0-5的整数,
  "fluency_reason": "一句话说明"
}}

【话术】{text}
"""
        # 尝试用本地模型生成
        if model_path:
            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "mlx_lm", "generate",
                        "--model", model_path,
                        "--prompt", judge_prompt,
                        "--max-tokens", "100",
                        "--temp", "0.1"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                output = result.stdout
                # 提取 JSON
                json_match = re.search(r'\{[^}]+\}', output)
                if json_match:
                    return json.loads(json_match.group())
            except Exception as e:
                pass

        # 如果无法调用模型，用规则估算流畅度
        return self._estimate_fluency(text)

    def _estimate_fluency(self, text: str) -> Dict:
        """用规则估算流畅度（作为 LLM 裁判的 fallback）"""
        sentences = self._split_sentences(text)
        long_sentences = [s for s in sentences if len(s) > 20]

        # 计算流畅度（0-5）
        score = 5
        if len(long_sentences) > len(sentences) * 0.3:
            score -= 1  # 长句过多
        if any(re.search(r'[一-鿿]{15,}', s) for s in sentences):
            score -= 1  # 有超过15字无标点的片段
        if len(sentences) < 3:
            score -= 1  # 句子太少

        return {
            "fluency_score": max(score, 1),
            "fluency_reason": f"长句数: {len(long_sentences)}/{len(sentences)}"
        }

    # ========== 评估单个样本 ==========

    def evaluate_single(self, text: str, prompt: str, model_path: str = None) -> Dict:
        """评估单个样本"""
        hard = self.hard_metrics(text, prompt)
        soft = self.soft_metrics_llm(text, model_path)

        return {**hard, **soft}

    # ========== 评估整个模型 ==========

    def evaluate_model(
        self,
        generate_func,
        model_name: str = "model",
        num_samples: int = None
    ) -> Dict:
        """
        评估整个模型

        Args:
            generate_func: 生成函数，输入 prompt，输出 text
            model_name: 模型名称
            num_samples: 评估样本数（None=全部）
        """
        results = []
        benchmark = self.benchmark[:num_samples] if num_samples else self.benchmark

        print(f"\n{'='*60}")
        print(f"开始评估: {model_name}")
        print(f"测试集大小: {len(benchmark)} 条")
        print(f"{'='*60}\n")

        for i, item in enumerate(benchmark):
            prompt = item["prompt"]
            print(f"[{i+1}/{len(benchmark)}] {item['id']} - {item['category']}/{item['scene']}")

            # 生成输出
            try:
                output = generate_func(prompt)
            except Exception as e:
                print(f"  ❌ 生成失败: {e}")
                output = ""

            # 评估
            metrics = self.evaluate_single(output, prompt)

            results.append({
                "benchmark_id": item["id"],
                "category": item["category"],
                "scene": item["scene"],
                "prompt": prompt,
                "output": output,
                "metrics": metrics
            })

            print(f"  互动密度: {metrics['interactive_density']:.2f} | "
                  f"句长: {metrics['avg_sentence_len']:.1f} | "
                  f"口语占比: {metrics['spoken_ratio']:.3f} | "
                  f"结构: {metrics['structure_completeness']}/5")

        # 聚合所有样本
        aggregated = self._aggregate_metrics(results)

        return {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "benchmark_size": len(benchmark),
            "sample_results": results,
            "aggregated": aggregated
        }

    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """聚合所有样本的平均指标"""
        all_metrics = [r["metrics"] for r in results]
        all_keys = all_metrics[0].keys()

        aggregated = {}
        for key in all_keys:
            values = [m[key] for m in all_metrics]
            # 只对数值类型计算统计量
            if isinstance(values[0], (int, float)):
                aggregated[key] = {
                    "mean": round(float(np.mean(values)), 3),
                    "std": round(float(np.std(values)), 3),
                    "min": round(float(np.min(values)), 3),
                    "max": round(float(np.max(values)), 3)
                }
            else:
                # 字符串类型只记录众数
                from collections import Counter
                most_common = Counter(values).most_common(1)[0][0]
                aggregated[key] = {
                    "mode": most_common,
                    "unique_count": len(set(values))
                }

        return aggregated

    # ========== 生成评估报告 ==========

    def save_report(self, report: Dict, filename: str = None) -> str:
        """保存评估报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = report["model_name"].replace("/", "_")
            filename = f"eval_{model_name}_{timestamp}.json"

        filepath = EVAL_RESULTS_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 报告已保存: {filepath}")
        return str(filepath)

    def print_comparison(self, reports: List[Dict]) -> str:
        """打印多模型对比报告"""
        if len(reports) < 2:
            print("需要至少 2 个报告才能对比")
            return ""

        # 定义指标显示名称和目标值
        metric_config = {
            "interactive_density": {"name": "互动密度(句)", "target": ">0.4", "higher_better": True},
            "avg_sentence_len": {"name": "平均句长(字)", "target": "<15", "higher_better": False},
            "spoken_ratio": {"name": "口语词占比", "target": ">0.15", "higher_better": True},
            "structure_completeness": {"name": "结构完整度(0-5)", "target": ">4.0", "higher_better": True},
            "selling_point_coverage": {"name": "卖点覆盖率", "target": ">0.85", "higher_better": True},
            "hallucination_rate": {"name": "幻觉率", "target": "<0.05", "higher_better": False},
            "emotion_progression": {"name": "情绪递进得分", "target": ">3.5", "higher_better": True},
            "fluency_score": {"name": "流畅度得分", "target": ">3.5", "higher_better": True}
        }

        # 构建表格
        header = f"{'指标':<20} |"
        for report in reports:
            name = report["model_name"][:15]
            header += f" {name:^15} |"
        header += f" {'目标值':^10} |"
        header += "\n" + "-" * len(header)

        rows = [header]
        for key, config in metric_config.items():
            row = f"{config['name']:<20} |"
            values = []
            for report in reports:
                val = report["aggregated"][key]["mean"]
                values.append(val)
                row += f" {val:^15.3f} |"

            # 添加变化箭头
            if len(values) >= 2:
                delta = values[-1] - values[-2]
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            else:
                arrow = ""

            row += f" {config['target']:^10} {arrow}"
            rows.append(row)

        table = "\n".join(rows)

        # 打印
        print(f"\n{'='*80}")
        print("模型迭代评估报告")
        print(f"Benchmark: {reports[0]['benchmark_size']}条固定测试集")
        print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*80}\n")
        print(table)

        # 关键发现
        print(f"\n{'='*80}")
        print("关键发现:")
        print(f"{'='*80}")

        base = reports[0]["aggregated"]
        latest = reports[-1]["aggregated"]

        improvements = []
        degradations = []

        for key, config in metric_config.items():
            base_val = base[key]["mean"]
            latest_val = latest[key]["mean"]
            delta = latest_val - base_val

            if config["higher_better"] and delta > 0.1:
                improvements.append((config["name"], delta))
            elif not config["higher_better"] and delta < -0.01:
                improvements.append((config["name"], -delta))
            elif config["higher_better"] and delta < -0.1:
                degradations.append((config["name"], delta))
            elif not config["higher_better"] and delta > 0.01:
                degradations.append((config["name"], delta))

        for name, delta in improvements:
            print(f"✅ {name}: 提升 {delta:.3f}")
        for name, delta in degradations:
            print(f"⚠️ {name}: 下降 {abs(delta):.3f}")

        return table


# ========== 使用示例 ==========

def create_mlx_generator(model_path: str, adapter_path: str = None, temp: float = 0.7):
    """创建 MLX 生成函数"""
    def generate(prompt: str) -> str:
        cmd = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", model_path,
            "--prompt", prompt,
            "--max-tokens", "300",
            "--temp", str(temp)
        ]
        if adapter_path:
            cmd.extend(["--adapter-path", adapter_path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # 提取生成内容（去掉元数据）
        output = result.stdout
        # 找到 ========== 之间的内容
        match = re.search(r'==========\n(.*?)\n==========', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return output.strip()

    return generate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="直播话术评估体系")
    parser.add_argument("--model", help="基座模型路径")
    parser.add_argument("--adapter", help="LoRA adapter 路径")
    parser.add_argument("--name", default="model", help="模型名称")
    parser.add_argument("--samples", type=int, help="评估样本数（默认全部30条）")
    parser.add_argument("--save", action="store_true", help="保存报告到文件")
    parser.add_argument("--compare", nargs="+", help="对比多个报告文件")

    args = parser.parse_args()

    evaluator = LiveStreamEvaluator()

    if args.compare:
        # 对比模式
        reports = []
        for filepath in args.compare:
            with open(filepath) as f:
                reports.append(json.load(f))
        evaluator.print_comparison(reports)
    else:
        # 评估模式
        generate_func = create_mlx_generator(args.model, args.adapter)

        report = evaluator.evaluate_model(
            generate_func,
            model_name=args.name,
            num_samples=args.samples
        )

        # 打印结果
        print(f"\n{'='*60}")
        print("评估完成!")
        print(f"{'='*60}")
        for key, val in report["aggregated"].items():
            if key not in ["sentence_count", "char_count"]:
                if "mean" in val:
                    print(f"{key}: {val['mean']:.3f} (±{val.get('std', 0):.3f})")
                else:
                    print(f"{key}: {val.get('mode', 'N/A')}")

        if args.save:
            evaluator.save_report(report)
