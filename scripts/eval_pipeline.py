#!/usr/bin/env python3
"""Livestream script evaluation pipeline, protocol v3.

v3 keeps the v2 generation contract and adds fact/style separation:
- one frozen system prompt for every model and every sample;
- interaction density is the fraction of sentences containing an interactive
  cue, bounded to [0, 1];
- style metrics normalise price/stock values, while hallucination checks always
  use the raw output and the benchmark fact table;
- hallucination is claim-level: unsupported and contradicted claims are counted
  separately, with qualitative claims included;
- structure coverage only requires modules the prompt actually supplies;
- reports include bootstrap confidence intervals and per-sample pass rates;
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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_prompts.jsonl"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"

METRIC_VERSION = "3.0"
PROMPT_VERSION = "livestream-v3"

SYSTEM_PROMPT = """你是一位抖音直播带货话术专家。你的风格特征：
①短句为主，每句不超过15字
②互动句占比控制在30%–50%，不要每句都喊"扣1"
③结构：痛点场景→解决方案→价格说明→行动指令；有库存或限时信息时才做紧迫感
④情绪递进：平静→兴奋→急促
⑤口头禅：家人们、真的绝了、听我说、有没有
⑥价格、数量、库存、时间、优惠等具体信息只能来自用户输入；用户没给就省略，或说"价格以直播间为准"。"""

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
    "限时福利": r"最后|限量|库存|卖完|补货|倒计时|前(?:\d+|<COUNT>)单|仅剩",
    "行动指令": r"下单|拍|抢|冲|链接|购物车|赶紧|立刻|马上",
}

STRUCTURE_BASE_COMPONENTS = ["痛点场景", "解决方案", "行动指令"]


def required_structure_components(facts: Dict) -> List[str]:
    """Only require price/urgency modules when the prompt actually provides them."""
    required = list(STRUCTURE_BASE_COMPONENTS)
    promotion = [str(item) for item in (facts.get("promotion") or [])]
    has_price = facts.get("price") is not None or facts.get("original_price") is not None
    has_price_offer = any(re.search(r"买\d+送\d+|半价|优惠|折扣|立减|减\d+", item) for item in promotion)
    if has_price or has_price_offer:
        required.append("价格锚定")
    if facts.get("stock") is not None or any(URGENT_PROMOTION_RE.search(item) for item in promotion):
        required.append("限时福利")
    return required

EMOTION_WORDS = {
    "平静": ["今天", "这个", "大家好", "产品", "卖点"],
    "兴奋": ["绝了", "真的", "直接", "太香了", "好用到哭", "绝绝子", "yyds"],
    "急促": ["最后", "快点", "快冲", "上链接", "冲啊", "抢", "赶紧", "马上", "倒计时", "321", "3、2、1", "手慢无", "抢完"],
}

INTERACTION_SAFE_NUMBERS = {1, 2, 3, 6, 8, 9, 321, 520, 666, 888, 1314}

PRICE_CONTEXT = re.compile(r"原价|日常价|专柜|直播价|到手价|价格|售价|只要|仅需|元|块|¥", re.I)
STOCK_CONTEXT = re.compile(r"最后|仅剩|只剩|库存|限量|补货|卖完|抢|拍", re.I)
STOCK_UNIT_PATTERN = r"单|件|支|个|套|盒|瓶|条|张|台|包|双|只|根|罐|杯|份|箱|提|片|粒|颗|袋"
STOCK_UNITS = set(STOCK_UNIT_PATTERN.split("|"))
TIME_UNITS = {"s", "min", "h", "d"}
PROMOTION_CONTEXT = re.compile(r"限时|倒计时|半价|优惠|买\d+送\d+", re.I)
PRICE_LABEL_RE = re.compile(r"(?:原价|日常价|专柜价|直播价|到手价|价格|售价|只要|仅需)\s*$")
STOCK_LABEL_RE = re.compile(r"(?:最后|仅剩|只剩|库存|限量|剩余)\s*$")
URGENT_PROMOTION_RE = re.compile(r"限时|倒计时|最后|库存|限量|秒杀|抢完|手慢无")
DISCOURSE_NUMBER_RE = re.compile(r"(?:选|挑|换|任选|其中|随便)\s*(?P<num>[一二两三四五六七八九\d])\s*(?:个|款|种|件)")
DISCOURSE_AMOUNT_RE = re.compile(r"(?:提升|提高|增加|降低|好|高|低|快|慢|大|小)\s*(?P<num>[一二两三四五六七八九\d])\s*(?:个|点|档|度)")
TIME_SPEC_CONTEXT_RE = re.compile(r"续航|待机|保温|保冷|持妆|充电|运行|工作|使用")
TIME_URGENCY_CONTEXT_RE = re.compile(r"最后|倒计时|限时|秒杀|抢|手慢无")
BUY_GIFT_CONTEXT_RE = re.compile(r"买[^，。！？!?]{0,8}送")
STOCK_FACT_CONTEXT_RE = re.compile(r"最后|仅剩|只剩|还剩|库存|限量|补货|剩余|买|送|拍")

NUM_UNIT_RE = re.compile(
    r"(?<![\d.])(?P<num>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>块钱|元|块|毫安时|毫安|mah|小时|h|分钟|min|秒钟|秒|s|天|d|抽|种|%|{STOCK_UNIT_PATTERN}|克|g|毫升|ml|升|l|瓦|w)?",
    re.I,
)
SPEC_PREFIX_RE = re.compile(r"(?P<prefix>spf|upf)(?P<num>\d+)\+?", re.I)

QUALITATIVE_CLAIM_PATTERNS = {
    "brand_equivalence": re.compile(r"赫莲娜同款|大牌同款|平替|大牌同源"),
    "certification": re.compile(r"医用级|械字号|药监|认证|专利|无菌"),
    "additive": re.compile(r"零添加|无添加"),
    "absolute": re.compile(r"全网最低|销量第一|行业第一|唯一|绝对|百分百"),
    "guarantee": re.compile(r"保证|根治|永久|无效退款"),
}

GENERIC_STOCK_CLAIMS = re.compile(r"库存(?:不多|告急|紧张|有限)|最后(?:一波|一批)|卖完(?:就)?(?:没了|补不到)", re.I)
VAGUE_PRICE_CLAIMS = re.compile(r"两位数|一杯奶茶钱|白菜价|骨折价", re.I)

CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
CN_NUMBER_RE = re.compile(
    r"(?<!几)[零〇一二两三四五六七八九十百千万亿]+"
    rf"(?=(?:块钱|元|块|毫安时|毫安|mah|小时|h|分钟|min|秒钟|秒|s|天|d|抽|种|%|{STOCK_UNIT_PATTERN}|克|g|毫升|ml|升|l|瓦|w))",
    re.I,
)

STYLE_ALIASES = [
    (r"秒回血|快速充电|充电速度飞快|充电快", "快充"),
    (r"独立小包装|独立包装", "独立包装"),
    (r"小包装", "包装"),
    (r"独立装", "装"),
    (r"颜值爆表|颜值拉满", "颜值高"),
    (r"不黏腻|不黏|清爽", "不油腻"),
    (r"无线充电不耽误|无线充", "无线充电兼容"),
    (r"主动降噪", "降噪"),
    (r"超强|很强", "强"),
]

TARGETS = {
    "interactive_sentence_ratio": {"label": "互动句占比", "lower": 0.30, "upper": 0.60},
    "spoken_ratio": {"label": "口语词占比", "lower": 0.02},
    "structure_coverage": {"label": "结构覆盖(必需模块)", "lower": 0.75},
    "selling_point_coverage": {"label": "卖点覆盖率", "lower": 0.80},
    "hallucination_rate": {"label": "幻觉率", "upper": 0.10},
    "emotion_progression": {
        "label": "情绪递进(0-5)",
        "lower": 3.5,
        "min_pass_rate": 0.70,
    },
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
    if "</think>" in text:
        text = text.split("</think>", 1)[1].lstrip()
    for marker in ("\n用户：", "\n助手：", "\n<|im_start|>", "<|im_end|>"):
        text = text.split(marker, 1)[0]
    return text.strip()


def _chinese_to_arabic(value: str) -> int:
    total = 0
    section = 0
    number = 0
    for char in value:
        if char in CN_DIGITS:
            number = CN_DIGITS[char]
            continue
        unit = CN_UNITS[char]
        if unit == 10000:
            section += number
            total += section * unit
            section = 0
            number = 0
        else:
            section += (number or 1) * unit
            number = 0
    return total + section + number


def _canonicalize_text(text: str) -> str:
    """Normalize numeric spelling and common unit/phrase variants."""
    text = text.lower()
    text = CN_NUMBER_RE.sub(lambda match: str(_chinese_to_arabic(match.group(0))), text)
    for pattern, replacement in STYLE_ALIASES:
        text = re.sub(pattern, replacement, text)
    unit_patterns = [
        (r"(\d+(?:\.\d+)?)\s*(?:毫安时|毫安|mah)", r"\1mah"),
        (r"(\d+(?:\.\d+)?)\s*(?:毫升|ml)", r"\1ml"),
        (r"(\d+(?:\.\d+)?)\s*(?:小时|h)", r"\1h"),
        (r"(\d+(?:\.\d+)?)\s*(?:分钟|min)", r"\1min"),
        (r"(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s)", r"\1s"),
        (r"(\d+(?:\.\d+)?)\s*(?:块钱|元|块)", r"\1元"),
        (r"(\d+(?:\.\d+)?)\s*(?:天|d)", r"\1d"),
        (r"(\d+(?:\.\d+)?)\s*(?:袋装|袋)", r"\1袋"),
        (r"(\d+(?:\.\d+)?)\s*(?:抽装|抽)", r"\1抽"),
        (r"(\d+(?:\.\d+)?)\s*(?:盒装|盒)", r"\1盒"),
        (r"(\d+(?:\.\d+)?)\s*(?:升|l)", r"\1l"),
        (r"(\d+(?:\.\d+)?)\s*(?:瓦|w)", r"\1w"),
    ]
    for pattern, replacement in unit_patterns:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff%+.]+", "", text)


def _match_selling_point(point: str, text: str) -> bool:
    point_text = _canonicalize_text(point)
    body = _canonicalize_text(text)
    if not point_text:
        return True
    if point_text in body:
        return True
    point_numbers = re.findall(r"\d+(?:\.\d+)?", point_text)
    if point_numbers and any(number not in body for number in point_numbers):
        return False
    if len(point_text) < 2 or not body:
        return False

    best = 0.0
    max_window = min(len(body), len(point_text) + 6)
    min_window = max(1, len(point_text) - 2)
    for window in range(min_window, max_window + 1):
        for start in range(0, len(body) - window + 1):
            ratio = SequenceMatcher(None, point_text, body[start:start + window]).ratio()
            if ratio > best:
                best = ratio
            if best >= 0.78:
                return True
    return False


def _extract_numeric_pairs(text: str) -> set[Tuple[str, str]]:
    canonical = _canonicalize_text(text)
    pairs = set()
    for match in NUM_UNIT_RE.finditer(canonical):
        pairs.add((match.group("num"), (match.group("unit") or "").lower()))
    for match in SPEC_PREFIX_RE.finditer(canonical):
        pairs.add((match.group("num"), match.group("prefix").lower()))
    return pairs


def parse_prompt_facts(prompt: str, structured: Optional[Dict] = None) -> Dict:
    """Parse product facts, falling back to structured benchmark fields when present."""
    facts: Dict = {
        "product": None,
        "selling_points": [],
        "duration": None,
        "audience": None,
        "price": None,
        "original_price": None,
        "stock": None,
        "stock_unit": None,
        "promotion": [],
    }
    for segment in re.split(r"[；;]", prompt):
        if "：" not in segment:
            continue
        key, value = segment.split("：", 1)
        key = key.strip()
        value = value.strip()
        if key == "产品":
            facts["product"] = value
        elif key == "卖点":
            facts["selling_points"] = [
                item.strip() for item in value.replace("、", ",").split(",") if item.strip()
            ]
        elif key in {"时长", "时长要求"}:
            facts["duration"] = value
        elif key in {"目标人群", "人群"}:
            facts["audience"] = value

    price_re = re.compile(
        r"(?P<label>原价|日常价|专柜价|直播价|到手价|价格|售价|只要|仅需)"
        r"\s*[：:]?\s*(?P<num>\d+(?:\.\d+)?)\s*(?:元|块|块钱)?"
    )
    for match in price_re.finditer(prompt):
        value = float(match.group("num"))
        if match.group("label") in {"原价", "日常价"}:
            facts["original_price"] = value
        else:
            facts["price"] = value

    stock_match = re.search(
        rf"(?:最后|仅剩|只剩|库存|限量|补货)\s*(?P<num>\d+)\s*(?P<unit>{STOCK_UNIT_PATTERN})",
        prompt,
    )
    if stock_match:
        facts["stock"] = int(stock_match.group("num"))
        facts["stock_unit"] = stock_match.group("unit")

    facts["promotion"] = re.findall(
        rf"限时[^；;，,]*|买\d+送\d+|半价|优惠|最后\d+(?:{STOCK_UNIT_PATTERN})",
        prompt,
    )
    if structured:
        for key in list(facts):
            if key in structured:
                facts[key] = structured[key]

    if facts["price"] is not None:
        facts["price"] = float(facts["price"])
    if facts["original_price"] is not None:
        facts["original_price"] = float(facts["original_price"])
    if facts["stock"] is not None:
        facts["stock"] = int(facts["stock"])
    if isinstance(facts["selling_points"], str):
        facts["selling_points"] = [
            item.strip() for item in facts["selling_points"].replace("、", ",").split(",") if item.strip()
        ]
    if not facts["selling_points"]:
        facts["selling_points"] = []
    if facts["promotion"] is None:
        facts["promotion"] = []
    if isinstance(facts["promotion"], str):
        facts["promotion"] = [facts["promotion"]]

    fact_text = "；".join(
        [
            prompt,
            "、".join(str(item) for item in facts["selling_points"]),
            str(facts["duration"] or ""),
            "、".join(str(item) for item in facts["promotion"]),
        ]
    )
    facts["_canonical_prompt"] = _canonicalize_text(fact_text)
    facts["_number_pairs"] = _extract_numeric_pairs(fact_text)
    if facts["price"] is not None:
        facts["_number_pairs"].add((f"{facts['price']:g}", "元"))
    if facts["original_price"] is not None:
        facts["_number_pairs"].add((f"{facts['original_price']:g}", "元"))
    if facts["stock"] is not None:
        facts["_number_pairs"].add((str(facts["stock"]), facts["stock_unit"] or "单"))
    facts["_numbers"] = {number for number, _ in facts["_number_pairs"]}
    return facts


def _public_facts(facts: Dict) -> Dict:
    return {key: value for key, value in facts.items() if not key.startswith("_")}


def normalize_slots(text: str, facts: Optional[Dict] = None) -> str:
    """Replace price/stock values with typed slots for style-only scoring."""
    protected: List[str] = []

    def protect_interaction(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    style = re.sub(
        r"(?:扣|刷|打|听懂的|想要的|有没有)[^，。！？!?]{0,6}?\d+|3、2、1|321|666|888",
        protect_interaction,
        text,
    )
    style = re.sub(
        r"(?P<label>原价|日常价|专柜价|直播价|到手价|价格|售价|只要|仅需)"
        r"\s*[：:]?\s*\d+(?:\.\d+)?\s*(?:元|块|块钱)?",
        r"\g<label><PRICE>",
        style,
    )
    style = re.sub(r"\d+(?:\.\d+)?\s*(?:元|块|块钱)", "<PRICE>", style)
    style = re.sub(
        rf"(?P<lead>最后|仅剩|只剩|库存|限量|补货)\s*\d+(?:\.\d+)?\s*(?P<unit>{STOCK_UNIT_PATTERN})",
        r"\g<lead><COUNT>\g<unit>",
        style,
    )
    style = re.sub(
        r"(?P<lead>倒计时|限时)\s*\d+(?:\.\d+)?\s*(?P<unit>秒|分钟|小时|s|min|h)",
        r"\g<lead><COUNT>\g<unit>",
        style,
    )
    for index, original in enumerate(protected):
        style = style.replace(f"\x00{index}\x00", original)
    return style


def _is_interaction_number(text: str, start: int, end: int, number: str, unit: str) -> bool:
    prefix = text[max(0, start - 8):start]
    context = text[max(0, start - 8):min(len(text), end + 8)]

    # "扣1" / "刷个1" are interaction cues even when a price word is nearby.
    if re.search(r"(?:扣|刷|打|听懂|想要|公屏|评论区)[^，。！？!?]{0,4}$", prefix):
        return True
    if unit in STOCK_UNITS and STOCK_CONTEXT.search(context):
        return False
    if unit == "元" or PRICE_LABEL_RE.search(prefix) or PRICE_CONTEXT.search(context):
        return False
    if unit in {"个", "款", "种", "件"} and re.search(
        r"(?:选|挑|换|任选|其中|随便|每)\s*$", prefix
    ):
        return True
    return number in {str(value) for value in INTERACTION_SAFE_NUMBERS} and not unit


def _claim_kind(context: str, unit: str, prefix: str = "") -> str:
    if unit == "元" or PRICE_LABEL_RE.search(prefix):
        return "price"
    if unit in STOCK_UNITS and (
        STOCK_LABEL_RE.search(prefix)
        or re.search(r"(?:最后|仅剩|只剩|还剩|库存|限量|补货|剩余)\s*$", prefix)
    ):
        return "stock"
    if unit in TIME_UNITS | {"%"} and PROMOTION_CONTEXT.search(context):
        return "promotion"
    return "spec"


def extract_claims(text: str, facts: Optional[Dict] = None) -> List[Dict]:
    """Extract unique numeric, price, stock and qualitative claims from raw text."""
    facts = facts or {}
    canonical = _canonicalize_text(text)
    claims: List[Dict] = []
    seen = set()
    discourse_spans = {
        (match.start("num"), match.end("num"))
        for pattern in (DISCOURSE_NUMBER_RE, DISCOURSE_AMOUNT_RE)
        for match in pattern.finditer(canonical)
    }

    for match in NUM_UNIT_RE.finditer(canonical):
        number = match.group("num")
        unit = (match.group("unit") or "").lower()
        if (match.start("num"), match.end("num")) in discourse_spans:
            continue
        if _is_interaction_number(canonical, match.start(), match.end(), number, unit):
            continue
        context = canonical[max(0, match.start() - 10):min(len(canonical), match.end() + 10)]
        prefix = canonical[max(0, match.start() - 8):match.start()]
        if unit in STOCK_UNITS and not STOCK_FACT_CONTEXT_RE.search(prefix + context):
            continue
        if unit in TIME_UNITS:
            local = canonical[max(0, match.start() - 6):match.start()] + canonical[match.end():min(len(canonical), match.end() + 3)]
            if TIME_URGENCY_CONTEXT_RE.search(local):
                kind = "promotion"
            elif TIME_SPEC_CONTEXT_RE.search(local):
                kind = "spec"
            else:
                # "出门一天", "跑两天" are usage scenarios, not factual specs.
                continue
        else:
            kind = _claim_kind(context, unit, prefix)
        key = (kind, number, unit)
        if key in seen:
            continue
        seen.add(key)
        claims.append({"kind": kind, "value": number, "unit": unit, "surface": match.group(0), "context": context})

    for match in SPEC_PREFIX_RE.finditer(canonical):
        number = match.group("num")
        unit = match.group("prefix").lower()
        key = ("spec", number, unit)
        if key in seen:
            continue
        seen.add(key)
        claims.append({"kind": "spec", "value": number, "unit": unit, "surface": match.group(0), "context": match.group(0)})

    for match in VAGUE_PRICE_CLAIMS.finditer(text):
        key = ("price", "vague", "")
        if key not in seen:
            seen.add(key)
            claims.append({"kind": "price", "value": "vague", "unit": "", "surface": match.group(0), "context": match.group(0)})

    for match in GENERIC_STOCK_CLAIMS.finditer(text):
        key = ("stock", "generic", "")
        if key not in seen:
            seen.add(key)
            claims.append({"kind": "stock", "value": "generic", "unit": "", "surface": match.group(0), "context": match.group(0)})

    for kind, pattern in QUALITATIVE_CLAIM_PATTERNS.items():
        for match in pattern.finditer(canonical):
            key = (kind, match.group(0), "")
            if key in seen:
                continue
            seen.add(key)
            claims.append({"kind": kind, "value": match.group(0), "unit": "", "surface": match.group(0), "context": canonical})
    return claims


def _same_number(left: str, right: float) -> bool:
    try:
        return abs(float(left) - float(right)) < 0.001
    except (TypeError, ValueError):
        return False


def verify_claims(text: str, facts: Dict) -> Dict:
    """Compare claims against the prompt fact table without wildcarding values."""
    claims = extract_claims(text, facts)
    supported: List[Dict] = []
    unsupported: List[Dict] = []
    contradicted: List[Dict] = []
    pair_index = facts.get("_number_pairs", set())
    number_index = facts.get("_numbers", set())

    for claim in claims:
        kind = claim["kind"]
        value = claim["value"]
        status = "unsupported"
        reason = "no_fact"
        if kind == "price":
            prices = [price for price in (facts.get("price"), facts.get("original_price")) if price is not None]
            if not prices:
                reason = "price_not_provided"
            elif value == "vague":
                reason = "vague_price"
            elif any(_same_number(value, price) for price in prices):
                status, reason = "supported", "price_matches"
            else:
                status, reason = "contradicted", "price_mismatch"
        elif kind == "stock":
            stock = facts.get("stock")
            if stock is None:
                reason = "stock_not_provided"
            elif value == "generic":
                status, reason = "supported", "stock_is_available"
            elif int(float(value)) == int(stock):
                status, reason = "supported", "stock_matches"
            else:
                status, reason = "contradicted", "stock_mismatch"
        elif kind in {"spec", "promotion"}:
            unit = claim["unit"]
            if (value, unit) in pair_index or (not unit and value in number_index):
                status, reason = "supported", "fact_matches"
            elif value in number_index and BUY_GIFT_CONTEXT_RE.search(claim.get("context", "")):
                status, reason = "supported", "promotion_numbers_match"
            elif unit and any(existing_unit == unit and not _same_number(existing_value, value) for existing_value, existing_unit in pair_index):
                status, reason = "contradicted", "fact_mismatch"
        elif kind in QUALITATIVE_CLAIM_PATTERNS:
            if value in facts.get("_canonical_prompt", ""):
                status, reason = "supported", "claim_in_prompt"
            else:
                reason = "claim_not_provided"

        checked = {**claim, "status": status, "reason": reason}
        if status == "supported":
            supported.append(checked)
        elif status == "contradicted":
            contradicted.append(checked)
        else:
            unsupported.append(checked)

    total = len(claims)
    bad = len(unsupported) + len(contradicted)
    hallucinated_numbers = sorted(
        {
            claim["value"]
            for claim in unsupported + contradicted
            if claim["value"] not in {"vague", "generic"} and re.fullmatch(r"\d+(?:\.\d+)?", claim["value"])
        },
        key=float,
    )
    return {
        "hallucination_rate": round(bad / max(total, 1), 3),
        "claim_count": total,
        "claim_support_rate": round(len(supported) / max(total, 1), 3),
        "unsupported_claim_count": len(unsupported),
        "contradicted_claim_count": len(contradicted),
        "hallucinated_numbers": hallucinated_numbers,
        "unsupported_claims": unsupported,
        "contradicted_claims": contradicted,
    }


def _chat_template_kwargs(tokenizer) -> Dict:
    """Disable thinking for Qwen3.5-style templates; no-op for Qwen2.5."""
    template = getattr(tokenizer, "chat_template", "") or ""
    return {"enable_thinking": False} if "enable_thinking" in template else {}


def _generation_signature(report: Dict) -> str:
    params = dict(report.get("generation_params") or {})
    params.setdefault("enable_thinking", False)
    return json.dumps(params, sort_keys=True)


def _target_passes(key: str, value: float) -> bool:
    spec = TARGETS.get(key)
    if spec is None:
        return True
    if "lower" in spec and value < spec["lower"]:
        return False
    if "upper" in spec and value > spec["upper"]:
        return False
    return True


def _bootstrap_ci(values: List[float], statistic: str, threshold: float | None = None) -> Tuple[float, float]:
    """Percentile bootstrap CI for a mean or a per-sample pass rate."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0, 0.0
    if array.size == 1:
        point = float(array[0] if statistic == "mean" else array[0] >= (threshold or 0.0))
        return point, point
    rng = np.random.default_rng(42)
    indices = rng.integers(0, array.size, size=(1000, array.size))
    samples = array[indices]
    if statistic == "mean":
        estimates = samples.mean(axis=1)
    else:
        estimates = (samples >= (threshold if threshold is not None else 0.0)).mean(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return round(float(low), 3), round(float(high), 3)


def _aggregate_target_passes(key: str, aggregated: Dict) -> bool:
    spec = TARGETS[key]
    if "min_pass_rate" in spec:
        return aggregated.get("pass_rate", 0.0) >= spec["min_pass_rate"]
    return _target_passes(key, aggregated["mean"])


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

    def hard_metrics(self, text: str, prompt: str, facts: Optional[Dict] = None) -> Dict:
        facts = facts or parse_prompt_facts(prompt)
        sentences = self._split_sentences(text)
        total_chars = len(text)
        total_sentences = max(len(sentences), 1)

        interactive_sentences = sum(
            1 for sentence in sentences if any(word in _normalize(sentence) for word in INTERACTIVE_WORDS)
        )
        interactive_ratio = round(interactive_sentences / total_sentences, 3)

        avg_sentence_len = round(total_chars / total_sentences, 1)
        spoken_count = sum(text.count(word) for word in SPOKEN_WORDS)
        spoken_ratio = round(spoken_count / max(total_chars, 1), 4)

        structure_text = normalize_slots(text, facts)
        structure_matches = {
            name: re.search(pattern, structure_text)
            for name, pattern in STRUCTURE_PATTERNS.items()
        }
        required_components = required_structure_components(facts)
        required_hits = [name for name in required_components if structure_matches.get(name)]
        structure_missing = [name for name in required_components if not structure_matches.get(name)]
        structure_coverage = round(len(required_hits) / max(len(required_components), 1), 3)
        structure_score = round(5.0 * structure_coverage, 1)
        found_components = [
            (name, match.start())
            for name, match in structure_matches.items()
            if match is not None
        ]
        if len(found_components) > 1:
            ordered_pairs = sum(
                1 for left, right in zip(found_components, found_components[1:]) if left[1] < right[1]
            )
            structure_order = round(ordered_pairs / (len(found_components) - 1), 3)
        else:
            structure_order = 0.0

        selling_points = facts.get("selling_points") or []
        covered = sum(1 for point in selling_points if _match_selling_point(point, text))
        selling_coverage = round(covered / max(len(selling_points), 1), 3)

        claim_metrics = verify_claims(text, facts)

        return {
            "interactive_sentence_ratio": interactive_ratio,
            "avg_sentence_len": avg_sentence_len,
            "spoken_ratio": spoken_ratio,
            "structure_coverage": structure_coverage,
            "structure_completeness": structure_score,
            "structure_required": required_components,
            "structure_required_hits": required_hits,
            "structure_missing": structure_missing,
            "structure_order_score": structure_order,
            "selling_point_coverage": selling_coverage,
            **claim_metrics,
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

    def evaluate_single(
        self,
        text: str,
        prompt: str,
        judge_model: str | None = None,
        facts: Optional[Dict] = None,
    ) -> Dict:
        facts = facts or parse_prompt_facts(prompt)
        return {
            **self.hard_metrics(text, prompt, facts),
            **self.soft_metrics_llm(text, judge_model),
        }

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
            facts = parse_prompt_facts(prompt, item.get("facts"))
            print(f"[{index}/{len(benchmark)}] {item['id']} - {item['category']}/{item['scene']}")
            try:
                output = generate_func(prompt)
            except Exception as exc:
                print(f"  ❌ 生成失败: {exc}")
                output = ""

            metrics = self.evaluate_single(output, prompt, judge_model, facts)
            results.append({
                "benchmark_id": item["id"],
                "category": item["category"],
                "scene": item["scene"],
                "prompt": prompt,
                "facts": _public_facts(facts),
                "output": output,
                "metrics": metrics,
            })
            print(
                f"  互动句占比: {metrics['interactive_sentence_ratio']:.2f} | "
                f"句长: {metrics['avg_sentence_len']:.1f} | "
                f"幻觉率: {metrics['hallucination_rate']:.2f} | "
                f"结构覆盖: {metrics['structure_coverage']:.0%} "
                f"({len(metrics['structure_required_hits'])}/{len(metrics['structure_required'])})"
            )

        aggregated = self._aggregate_metrics(results)
        target_checks = {
            key: _aggregate_target_passes(key, aggregated[key])
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
                mean_low, mean_high = _bootstrap_ci(values, "mean")
                entry = {
                    "mean": round(float(np.mean(values)), 3),
                    "std": round(float(np.std(values)), 3),
                    "min": round(float(np.min(values)), 3),
                    "max": round(float(np.max(values)), 3),
                    "ci95_low": mean_low,
                    "ci95_high": mean_high,
                }
                if key in TARGETS:
                    passes = [1.0 if _target_passes(key, value) else 0.0 for value in values]
                    pass_low, pass_high = _bootstrap_ci(passes, "mean")
                    entry["pass_rate"] = round(float(np.mean(passes)), 3)
                    entry["pass_rate_ci95_low"] = pass_low
                    entry["pass_rate_ci95_high"] = pass_high
                aggregated[key] = entry
            elif isinstance(values[0], str):
                mode = Counter(values).most_common(1)[0][0]
                aggregated[key] = {"mode": mode, "unique_count": len(set(values))}
        return aggregated

    def save_report(self, report: Dict, filename: str | None = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = report["model_name"].replace("/", "_")
            filename = f"eval_v3_{model_name}_{timestamp}.json"
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
            _generation_signature(report)
            for report in reports
        }
        if len(generation_params) > 1:
            print("⚠️ 报告生成参数不一致（如 max_tokens/temp/top_p）；不能作为结论。")

        header = f"{'指标':<20} |"
        for report in reports:
            header += f" {report['model_name'][:15]:^15} |"
        header += f" {'目标':^22} |"
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
            if "min_pass_rate" in spec:
                target = f"{spec['lower']}+ 且 {spec['min_pass_rate']:.0%} 样本"
            row += f" {target:^22} |"
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
    seed: int | None = None,
):
    """Load an MLX model once and return a prompt -> text function."""
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    if seed is not None:
        mx.random.seed(seed)
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
            **_chat_template_kwargs(tokenizer),
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
    parser = argparse.ArgumentParser(description="直播话术评估体系 v3")
    parser.add_argument("--model", help="基座模型路径")
    parser.add_argument("--adapter", help="LoRA adapter 路径")
    parser.add_argument("--name", default="model", help="模型名称")
    parser.add_argument("--samples", type=int, help="评估样本数（默认全部 30 条）")
    parser.add_argument("--save", action="store_true", help="保存报告到文件")
    parser.add_argument("--compare", nargs="+", help="对比多个报告文件")
    parser.add_argument("--judge-model", help="可选：本地模型路径，用于 LLM 流畅度裁判")
    parser.add_argument("--benchmark", help="可选：自定义 benchmark JSONL（默认 data/benchmark_prompts.jsonl）")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluator = LiveStreamEvaluator(args.benchmark) if args.benchmark else LiveStreamEvaluator()
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
        seed=args.seed,
    )
    report = evaluator.evaluate_model(
        generate_func,
        model_name=args.name,
        num_samples=args.samples,
        model_path=args.model,
        adapter_path=args.adapter,
        judge_model=args.judge_model,
        generation_params={
            "max_tokens": args.max_tokens,
            "temp": args.temp,
            "top_p": args.top_p,
            "seed": args.seed,
            "enable_thinking": False,
        },
    )

    print(f"\n{'=' * 60}")
    print("评估完成")
    print(f"{'=' * 60}")
    for key, spec in TARGETS.items():
        aggregated = report["aggregated"].get(key)
        if not aggregated or "mean" not in aggregated:
            continue
        status = "PASS" if report["target_checks"].get(key) else "FAIL"
        extra = ""
        if "min_pass_rate" in spec:
            extra = f" | 单条达标率={aggregated.get('pass_rate', 0):.0%}"
        print(f"{spec['label']}: {aggregated['mean']:.3f}{extra} [{status}]")
    diagnostics = report["aggregated"].get("avg_sentence_len", {})
    if diagnostics:
        print(f"诊断（不计入目标）: 平均句长={diagnostics['mean']:.3f} 字")
    passed = sum(report["target_checks"].values())
    print(f"targets: {passed}/{len(report['target_checks'])}")

    if args.save:
        evaluator.save_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
