#!/usr/bin/env python3
"""Small regression tests for eval_pipeline v2 metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pipeline import LiveStreamEvaluator  # noqa: E402


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = LiveStreamEvaluator()

    def test_interactive_ratio_is_bounded(self):
        text = "家人们！你们有没有发现？对不对？这款只要79，赶紧下单！"
        metrics = self.evaluator.hard_metrics(text, "产品：防晒霜；卖点：SPF50+")
        self.assertGreaterEqual(metrics["interactive_sentence_ratio"], 0)
        self.assertLessEqual(metrics["interactive_sentence_ratio"], 1)

    def test_prompt_grounded_numbers_are_not_hallucinations(self):
        prompt = "产品：充电宝；卖点：20000mAh、快充；时长：1分钟"
        text = "家人们，20000mAh，1分钟充满，最后3单，赶紧下单！"
        metrics = self.evaluator.hard_metrics(text, prompt)
        self.assertEqual(metrics["hallucinated_numbers"], [])
        self.assertEqual(metrics["hallucination_rate"], 0.0)

    def test_invented_price_is_flagged(self):
        prompt = "产品：面霜；卖点：玻色因、7天淡纹；时长：1分钟"
        text = "家人们，原价299，今天只要79，最后10单，赶紧下单！"
        metrics = self.evaluator.hard_metrics(text, prompt)
        self.assertIn("299", metrics["hallucinated_numbers"])
        self.assertIn("79", metrics["hallucinated_numbers"])
        self.assertIn("10", metrics["hallucinated_numbers"])
        self.assertGreater(metrics["hallucination_rate"], 0.0)

    def test_emotion_progression_rewards_real_progression(self):
        flat = "这个产品很好。这个产品不错。这个产品可以。"
        progression = "今天给大家介绍这款面霜。真的绝了，太好用了！最后10单，赶紧抢！"
        flat_score = self.evaluator.hard_metrics(flat, "产品：面霜")["emotion_progression"]
        progression_score = self.evaluator.hard_metrics(progression, "产品：面霜")["emotion_progression"]
        self.assertGreater(progression_score, flat_score)

    def test_structure_requires_specific_cues(self):
        weak = "今天这个产品可以。"
        strong = "你们是不是最怕卡粉？这款采用新配方能解决。原价299，直播间只要79。最后10单，赶紧下单！"
        weak_score = self.evaluator.hard_metrics(weak, "产品：粉底液")["structure_completeness"]
        strong_score = self.evaluator.hard_metrics(strong, "产品：粉底液")["structure_completeness"]
        self.assertGreater(strong_score, weak_score)

    def test_report_records_provenance(self):
        report = self.evaluator.evaluate_model(
            lambda prompt: "家人们，这个真的绝了！最后3单，赶紧下单！",
            model_name="test",
            num_samples=1,
            generation_params={"max_tokens": 32},
        )
        self.assertEqual(report["metric_version"], "2.0")
        self.assertEqual(report["prompt_version"], "livestream-v2")
        self.assertIn("git_commit", report)
        self.assertIn("git_dirty", report)
        self.assertEqual(len(report["evaluator_sha256"]), 64)
        self.assertEqual(report["benchmark_size"], 1)


if __name__ == "__main__":
    unittest.main()
