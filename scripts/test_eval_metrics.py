#!/usr/bin/env python3
"""Regression tests for eval_pipeline v3 metrics."""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pipeline import (  # noqa: E402
    TARGETS,
    LiveStreamEvaluator,
    _aggregate_target_passes,
    _chat_template_kwargs,
    _clean_generation,
    extract_claims,
    normalize_slots,
    parse_prompt_facts,
    required_structure_components,
    verify_claims,
)


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
        text = "家人们，20000mAh，1分钟，扣1！"
        metrics = self.evaluator.hard_metrics(text, prompt)
        self.assertEqual(metrics["hallucinated_numbers"], [])
        self.assertEqual(metrics["hallucination_rate"], 0.0)

    def test_invented_price_is_flagged(self):
        prompt = "产品：面霜；卖点：玻色因、7天淡纹；时长：1分钟"
        text = "家人们，原价299，今天只要79，最后10单，赶紧下单！"
        metrics = self.evaluator.hard_metrics(text, prompt)
        self.assertEqual(set(metrics["hallucinated_numbers"]), {"10", "79", "299"})
        self.assertGreater(metrics["hallucination_rate"], 0.0)

    def test_unprovided_stock_is_flagged(self):
        prompt = "产品：口红；卖点：显白；时长：1分钟"
        metrics = self.evaluator.hard_metrics("最后3单，扣1！", prompt)
        self.assertIn("3", metrics["hallucinated_numbers"])

    def test_declared_stock_is_supported(self):
        for unit, count in {"单": 10, "条": 45, "张": 18, "袋": 40, "台": 20, "包": 70}.items():
            with self.subTest(unit=unit):
                prompt = f"产品：面膜；卖点：玻尿酸；时长：30秒；最后{count}{unit}"
                metrics = self.evaluator.hard_metrics(
                    f"最后{count}{unit}，卖完就没了！",
                    prompt,
                )
                self.assertEqual(metrics["hallucinated_numbers"], [])
                self.assertEqual(metrics["hallucination_rate"], 0.0)

    def test_stock_mismatch_is_contradicted(self):
        prompt = "产品：面膜；卖点：玻尿酸；时长：30秒；最后10单"
        metrics = self.evaluator.hard_metrics("最后50单，扣1！", prompt)
        self.assertEqual(metrics["contradicted_claim_count"], 1)
        self.assertIn("50", metrics["hallucinated_numbers"])

    def test_price_mismatch_is_contradicted(self):
        prompt = "产品：面膜；卖点：补水；时长：1分钟；价格：99元"
        metrics = self.evaluator.hard_metrics("今天只要199，赶紧下单！", prompt)
        self.assertEqual(metrics["contradicted_claim_count"], 1)

    def test_decimal_price_keeps_its_decimal_point(self):
        prompt = "产品：黑芝麻丸；卖点：0蔗糖；时长：1分钟；价格：39.9元；原价：79元"
        metrics = self.evaluator.hard_metrics(
            "平时79元，直播间只要39.9元，赶紧下单！",
            prompt,
        )
        self.assertEqual(metrics["hallucinated_numbers"], [])
        self.assertEqual(metrics["hallucination_rate"], 0.0)

    def test_missing_price_abstention_passes(self):
        prompt = "产品：面膜；卖点：补水；时长：1分钟"
        metrics = self.evaluator.hard_metrics("价格以直播间为准，赶紧下单！", prompt)
        self.assertEqual(metrics["hallucination_rate"], 0.0)

    def test_global_slot_normalization_does_not_hide_grounding(self):
        prompt = "产品：口红；卖点：显白；时长：1分钟"
        facts = parse_prompt_facts(prompt)
        text = "最后50支，只要99，赶紧下单！"
        style = normalize_slots(text, facts)
        self.assertIn("<PRICE>", style)
        self.assertIn("<COUNT>", style)
        claims = verify_claims(text, facts)
        self.assertEqual(claims["unsupported_claim_count"], 2)
        self.assertEqual(set(claims["hallucinated_numbers"]), {"50", "99"})

    def test_selling_point_numeric_and_unit_equivalence(self):
        prompt = "产品：防晒霜；卖点：SPF50+、3秒成膜、不油腻；时长：2分钟"
        text = "这款防晒霜三秒成膜，清爽得像没涂，SPF50+！"
        metrics = self.evaluator.hard_metrics(text, prompt)
        self.assertEqual(metrics["selling_point_coverage"], 1.0)

    def test_selling_point_bag_paraphrase(self):
        prompt = "产品：零食大礼包；卖点：10袋装、网红爆款、独立包装；时长：1分钟"
        text = "十袋独立装，网红爆款，独立包装，想要的扣1！"
        metrics = self.evaluator.hard_metrics(text, prompt)
        self.assertEqual(metrics["selling_point_coverage"], 1.0)

    def test_structure_score_is_value_agnostic(self):
        prompt = "产品：口红；卖点：显白；时长：1分钟"
        left = self.evaluator.hard_metrics("最后50支，只要99，赶紧下单！", prompt)
        right = self.evaluator.hard_metrics("最后10支，只要79，赶紧下单！", prompt)
        self.assertEqual(left["structure_completeness"], right["structure_completeness"])

    def test_structure_only_requires_facts_that_were_provided(self):
        no_facts = parse_prompt_facts("产品：口红；卖点：显白；时长：1分钟")
        self.assertEqual(required_structure_components(no_facts), ["痛点场景", "解决方案", "行动指令"])
        metrics = self.evaluator.hard_metrics(
            "有没有姐妹最怕显黑？这款口红能显白。想要的扣1，赶紧下单！",
            "产品：口红；卖点：显白；时长：1分钟",
        )
        self.assertEqual(metrics["structure_coverage"], 1.0)
        self.assertEqual(metrics["structure_missing"], [])

    def test_structure_requires_urgency_when_stock_is_given(self):
        prompt = "产品：面膜；卖点：补水；时长：30秒；最后10单"
        facts = parse_prompt_facts(prompt)
        self.assertIn("限时福利", required_structure_components(facts))
        metrics = self.evaluator.hard_metrics(
            "有没有姐妹最怕脸干？这款面膜能补水。想要的扣1，赶紧下单！",
            prompt,
        )
        self.assertEqual(metrics["structure_coverage"], 0.75)
        self.assertEqual(metrics["structure_missing"], ["限时福利"])

    def test_price_label_does_not_swallow_nearby_package_count(self):
        facts = parse_prompt_facts("产品：面膜；卖点：补水；时长：1分钟")
        claims = extract_claims("价格29元1盒，最后10单，赶紧下单！", facts)
        by_surface = {(claim["value"], claim["unit"], claim["kind"]) for claim in claims}
        self.assertIn(("29", "元", "price"), by_surface)
        self.assertIn(("1", "盒", "spec"), by_surface)
        self.assertIn(("10", "单", "stock"), by_surface)

    def test_interaction_number_is_not_a_price_claim(self):
        facts = parse_prompt_facts("产品：口红；卖点：显白；时长：1分钟")
        claims = verify_claims("想要的扣1，价格你自己定，赶紧下单！", facts)
        self.assertEqual(claims["unsupported_claim_count"], 0)

    def test_discourse_number_is_not_a_claim(self):
        facts = parse_prompt_facts("产品：口红；卖点：显白；时长：1分钟")
        claims = verify_claims("姐妹们选一个喜欢的颜色，赶紧下单！", facts)
        self.assertEqual(claims["unsupported_claim_count"], 0)

    def test_usage_duration_is_not_a_product_claim(self):
        facts = parse_prompt_facts("产品：面霜；卖点：持妆12小时；时长：1分钟")
        claims = verify_claims("出门一天也不怕，续航很稳！", facts)
        self.assertEqual(claims["unsupported_claim_count"], 0)

    def test_battery_duration_is_checked_near_spec_words(self):
        facts = parse_prompt_facts("产品：充电宝；卖点：20000mAh；时长：1分钟")
        claims = verify_claims("续航30小时，赶紧下单！", facts)
        self.assertIn("30", claims["hallucinated_numbers"])

    def test_buy_gift_unit_variants_match_prompt(self):
        facts = parse_prompt_facts("产品：洗衣液；卖点：亮白；时长：1分钟；买2送1")
        claims = verify_claims("今天买2瓶直接送1瓶，赶紧下单！", facts)
        self.assertEqual(claims["unsupported_claim_count"], 0)

    def test_vague_amount_phrase_is_not_a_claim(self):
        facts = parse_prompt_facts("产品：口红；卖点：显白；时长：1分钟")
        for text in [
            "气色直接提升1个度，赶紧下单！",
            "一罐顶三餐，赶紧下单！",
            "十袋一大包，囤着超划算！",
        ]:
            with self.subTest(text=text):
                claims = verify_claims(text, facts)
                self.assertEqual(claims["unsupported_claim_count"], 0)

    def test_emotion_target_uses_sample_pass_rate(self):
        self.assertTrue(
            _aggregate_target_passes(
                "emotion_progression",
                {"mean": 3.0, "pass_rate": 0.7},
            )
        )
        self.assertFalse(
            _aggregate_target_passes(
                "emotion_progression",
                {"mean": 4.0, "pass_rate": 0.69},
            )
        )

    def test_sentence_length_is_diagnostic_not_target(self):
        report = self.evaluator.evaluate_model(
            lambda prompt: "家人们，这个真的绝了！最后3单，赶紧下单！",
            model_name="test",
            num_samples=1,
            generation_params={"max_tokens": 32},
        )
        self.assertIn("avg_sentence_len", report["aggregated"])
        self.assertNotIn("avg_sentence_len", report["target_checks"])
        self.assertEqual(len(report["target_checks"]), len(TARGETS))

    def test_report_records_provenance(self):
        report = self.evaluator.evaluate_model(
            lambda prompt: "家人们，这个真的绝了！最后3单，赶紧下单！",
            model_name="test",
            num_samples=1,
            generation_params={"max_tokens": 32},
        )
        self.assertEqual(report["metric_version"], "3.0")
        self.assertEqual(report["prompt_version"], "livestream-v3")
        self.assertIn("git_commit", report)
        self.assertIn("git_dirty", report)
        self.assertEqual(len(report["evaluator_sha256"]), 64)
        self.assertEqual(report["benchmark_size"], 1)

    def test_gold_scripts_pass_all_targets(self):
        path = Path(__file__).resolve().parent.parent / "data" / "gold_scripts.jsonl"
        gold = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        responses = {item["prompt"]: item["output"] for item in gold}
        with redirect_stdout(StringIO()):
            report = self.evaluator.evaluate_model(
                lambda prompt: responses[prompt],
                model_name="gold",
                num_samples=len(gold),
                generation_params={"gold": True},
            )
        self.assertEqual(sum(report["target_checks"].values()), len(TARGETS))

    def test_generation_is_cut_at_fake_role_turn(self):
        text = "家人们，这款真的绝了！\n用户：下一个产品\n助手：好的"
        self.assertEqual(_clean_generation(text), "家人们，这款真的绝了！")

    def test_thinking_is_disabled_only_when_supported(self):
        class Tokenizer:
            chat_template = "{% if enable_thinking %}...{% endif %}"

        class LegacyTokenizer:
            chat_template = "plain template"

        self.assertEqual(_chat_template_kwargs(Tokenizer()), {"enable_thinking": False})
        self.assertEqual(_chat_template_kwargs(LegacyTokenizer()), {})

    def test_reasoning_prefix_is_removed(self):
        text = "反复思考...\n</think>\n家人们，这款真的绝了！"
        self.assertEqual(_clean_generation(text), "家人们，这款真的绝了！")

    def test_compare_warns_when_generation_params_differ(self):
        base = {
            "model_name": "a",
            "metric_version": "3.0",
            "prompt_version": "livestream-v2",
            "benchmark_sha256": "same",
            "aggregated": {},
            "generation_params": {"max_tokens": 180},
        }
        other = {**base, "model_name": "b", "generation_params": {"max_tokens": 300}}
        output = StringIO()
        with redirect_stdout(output):
            self.evaluator.print_comparison([base, other])
        self.assertIn("生成参数不一致", output.getvalue())


if __name__ == "__main__":
    unittest.main()
