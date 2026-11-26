#!/usr/bin/env python3
"""Build a fact-grounded, product-disjoint SFT dataset for livestream scripts.

Every reference is checked against the same v3 evaluator used for model
evaluation. A sample is written only when all eight targets pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pipeline import (  # noqa: E402
    SYSTEM_PROMPT,
    TARGETS,
    LiveStreamEvaluator,
    _target_passes,
    parse_prompt_facts,
)


PRODUCTS = [
    {"name": "氨基酸洗面奶", "category": "美妆", "audience": "敏感肌", "pains": ["洗完脸紧绷", "换季泛红", "毛孔粗大"], "selling_points": ["氨基酸配方", "温和清洁", "不紧绷"], "benefits": ["洗完脸很舒服", "皮肤状态稳定多了"], "price": 69, "original_price": 129, "stock": 50, "stock_unit": "支", "promotion": None},
    {"name": "烟酰胺精华", "category": "美妆", "audience": "熬夜党", "pains": ["脸色暗沉", "痘印明显", "上妆发灰"], "selling_points": ["5%烟酰胺", "提亮肤色", "清爽好吸收"], "benefits": ["气色透亮多了", "素颜也敢出门"], "price": 99, "original_price": 199, "stock": None, "stock_unit": None, "promotion": "买2送1"},
    {"name": "视黄醇眼霜", "category": "美妆", "audience": "熬夜党", "pains": ["眼纹明显", "黑眼圈重", "眼周干燥"], "selling_points": ["视黄醇配方", "淡化细纹", "滋润不腻"], "benefits": ["眼周滋润多了", "看起来精神很多"], "price": 129, "original_price": 259, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "卸妆膏", "category": "美妆", "audience": "化妆党", "pains": ["卸不干净", "闷痘", "糊眼睛"], "selling_points": ["一秒乳化", "深层清洁", "不糊眼"], "benefits": ["卸完很清爽", "毛孔舒服多了"], "price": 79, "original_price": 159, "stock": 30, "stock_unit": "盒", "promotion": None},
    {"name": "遮瑕膏", "category": "美妆", "audience": "痘印党", "pains": ["痘印遮不住", "卡纹", "妆感厚重"], "selling_points": ["高遮瑕", "轻薄服帖", "不卡纹"], "benefits": ["妆面干净多了", "看着自然很多"], "price": 59, "original_price": 119, "stock": None, "stock_unit": None, "promotion": "限时优惠"},
    {"name": "身体乳", "category": "美妆", "audience": "干皮姐妹", "pains": ["皮肤起皮", "秋冬干痒", "涂完黏腻"], "selling_points": ["烟酰胺", "长效保湿", "清爽不黏"], "benefits": ["皮肤滑多了", "睡衣也不粘"], "price": 49, "original_price": 99, "stock": 100, "stock_unit": "瓶", "promotion": None},

    {"name": "无线鼠标", "category": "数码", "audience": "办公党", "pains": ["有线太乱", "点击声太吵", "用久了手酸"], "selling_points": ["静音按键", "2.4G连接", "续航6个月"], "benefits": ["桌面清爽多了", "办公安静很多"], "price": 89, "original_price": 169, "stock": 80, "stock_unit": "个", "promotion": None},
    {"name": "桌面风扇", "category": "数码", "audience": "学生党", "pains": ["夏天闷热", "风扇太吵", "占地方"], "selling_points": ["3档风速", "静音运行", "USB供电"], "benefits": ["桌面凉快多了", "学习也不吵"], "price": 39, "original_price": 79, "stock": None, "stock_unit": None, "promotion": "限时半价"},
    {"name": "便携音箱", "category": "数码", "audience": "户外党", "pains": ["声音太小", "续航太短", "进水损坏"], "selling_points": ["360度环绕", "12小时续航", "防水设计"], "benefits": ["户外氛围拉满", "续航很持久"], "price": 159, "original_price": 299, "stock": 20, "stock_unit": "台", "promotion": None},
    {"name": "数据线", "category": "数码", "audience": "通勤党", "pains": ["线材容易断", "充电太慢", "长度不够"], "selling_points": ["1.5米加长", "快充协议", "耐弯折"], "benefits": ["车里用着方便", "充电快多了"], "price": 29, "original_price": 59, "stock": None, "stock_unit": None, "promotion": "买2送1"},
    {"name": "移动硬盘", "category": "数码", "audience": "上班族", "pains": ["文件太多", "传输太慢", "携带不便"], "selling_points": ["1TB容量", "高速传输", "小巧便携"], "benefits": ["资料放心多了", "出门带着不累"], "price": 399, "original_price": 599, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "平板支架", "category": "数码", "audience": "追剧党", "pains": ["低头脖子疼", "支架不稳", "角度不合适"], "selling_points": ["多角度调节", "铝合金材质", "防滑设计"], "benefits": ["脖子舒服多了", "看剧更轻松"], "price": 59, "original_price": 119, "stock": 60, "stock_unit": "个", "promotion": None},

    {"name": "黑芝麻丸", "category": "食品", "audience": "熬夜党", "pains": ["头发没光泽", "下午容易饿", "零食不健康"], "selling_points": ["0蔗糖", "九蒸九晒", "独立包装"], "benefits": ["吃起来很方便", "口感香而不腻"], "price": 39.9, "original_price": 79, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "牛肉干", "category": "食品", "audience": "追剧党", "pains": ["嘴巴寂寞", "零食不顶饱", "肉质太柴"], "selling_points": ["高蛋白", "独立小包", "五香口味"], "benefits": ["越嚼越香", "追剧很解馋"], "price": 49.9, "original_price": 89, "stock": 40, "stock_unit": "袋", "promotion": None},
    {"name": "坚果燕麦", "category": "食品", "audience": "健身党", "pains": ["早餐来不及", "饱腹感差", "糖分太高"], "selling_points": ["7种坚果", "低温烘焙", "无添加"], "benefits": ["早餐很方便", "吃完很顶饱"], "price": 59, "original_price": 109, "stock": None, "stock_unit": None, "promotion": "限时优惠"},
    {"name": "冻干咖啡", "category": "食品", "audience": "打工人", "pains": ["早上没精神", "咖啡太苦", "冲泡麻烦"], "selling_points": ["0添加", "3秒速溶", "冷热双泡"], "benefits": ["早上省时间", "口感很顺滑"], "price": 69, "original_price": 129, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "自热米饭", "category": "食品", "audience": "加班党", "pains": ["加班没饭吃", "外卖太慢", "口味单一"], "selling_points": ["15分钟即食", "大份量", "多口味"], "benefits": ["加班也能吃热饭", "分量很实在"], "price": 29.9, "original_price": 49.9, "stock": 100, "stock_unit": "盒", "promotion": None},
    {"name": "低脂鸡胸肉", "category": "食品", "audience": "健身党", "pains": ["减脂不知道吃什么", "鸡胸肉太柴", "做法太麻烦"], "selling_points": ["高蛋白", "低脂", "开袋即食"], "benefits": ["吃起来很方便", "口感一点不柴"], "price": 39.9, "original_price": 69.9, "stock": None, "stock_unit": None, "promotion": "买2送1"},

    {"name": "垃圾袋", "category": "家居", "audience": "懒人党", "pains": ["袋子容易破", "倒垃圾脏手", "味道大"], "selling_points": ["加厚材质", "抽绳设计", "50只装"], "benefits": ["提起来不脏手", "用着很结实"], "price": 19.9, "original_price": 39.9, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "厨房纸巾", "category": "家居", "audience": "做饭党", "pains": ["油污难擦", "纸巾掉屑", "吸水差"], "selling_points": ["3层加厚", "吸水强", "不掉屑"], "benefits": ["厨房干净多了", "擦油很省心"], "price": 29.9, "original_price": 59.9, "stock": 70, "stock_unit": "包", "promotion": None},
    {"name": "衣架", "category": "家居", "audience": "租房党", "pains": ["衣服滑落", "肩膀鼓包", "衣柜太乱"], "selling_points": ["防滑设计", "无痕肩线", "10个装"], "benefits": ["衣柜整齐多了", "衣服不再鼓包"], "price": 25.9, "original_price": 49.9, "stock": None, "stock_unit": None, "promotion": "限时优惠"},
    {"name": "拖把", "category": "家居", "audience": "宝妈", "pains": ["拖地费腰", "手洗太脏", "水渍太多"], "selling_points": ["免手洗", "干湿两用", "360度旋转"], "benefits": ["拖地轻松多了", "地面干净很多"], "price": 79, "original_price": 159, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "香薰蜡烛", "category": "家居", "audience": "助眠党", "pains": ["房间有异味", "睡前焦虑", "灯光太刺眼"], "selling_points": ["天然大豆蜡", "燃烧20小时", "助眠香型"], "benefits": ["房间味道舒服多了", "睡前放松很多"], "price": 49, "original_price": 99, "stock": 25, "stock_unit": "盒", "promotion": None},
    {"name": "加湿器", "category": "家居", "audience": "空调房党", "pains": ["空气太干", "嗓子不舒服", "夜里口渴"], "selling_points": ["静音运行", "大雾量", "缺水断电"], "benefits": ["空气湿润多了", "睡觉舒服很多"], "price": 99, "original_price": 199, "stock": None, "stock_unit": None, "promotion": "限时半价"},

    {"name": "打底衫", "category": "服饰", "audience": "通勤党", "pains": ["冬天不够暖", "衣服显胖", "领口松垮"], "selling_points": ["德绒发热", "修身版型", "不勾丝"], "benefits": ["穿起来很暖和", "显瘦效果不错"], "price": 59, "original_price": 119, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "羽绒服", "category": "服饰", "audience": "北方姐妹", "pains": ["冬天不够暖", "穿起来臃肿", "钻绒"], "selling_points": ["90%白鸭绒", "防风面料", "轻便保暖"], "benefits": ["穿着轻还暖和", "版型不显胖"], "price": 399, "original_price": 799, "stock": 15, "stock_unit": "件", "promotion": None},
    {"name": "袜子", "category": "服饰", "audience": "通勤党", "pains": ["袜子闷脚", "容易破洞", "洗几次变形"], "selling_points": ["纯棉材质", "防臭设计", "5双装"], "benefits": ["穿起来不闷", "洗了也不变形"], "price": 29.9, "original_price": 59.9, "stock": None, "stock_unit": None, "promotion": "买2送1"},
    {"name": "帽子", "category": "服饰", "audience": "户外党", "pains": ["太阳太晒", "帽子显脸大", "头围不合适"], "selling_points": ["显脸小", "防晒设计", "可调节"], "benefits": ["戴上很显精神", "出门不怕晒"], "price": 39, "original_price": 79, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "瑜伽裤", "category": "服饰", "audience": "健身党", "pains": ["运动裤掉裆", "出汗黏腿", "显小肚子"], "selling_points": ["高腰设计", "提臀版型", "速干面料"], "benefits": ["运动很自在", "显身材不勒"], "price": 69, "original_price": 139, "stock": 45, "stock_unit": "条", "promotion": None},
    {"name": "衬衫", "category": "服饰", "audience": "上班族", "pains": ["衬衫易皱", "版型显壮", "不好搭配"], "selling_points": ["免烫面料", "垂感版型", "百搭设计"], "benefits": ["早上省时间", "穿着很显气质"], "price": 89, "original_price": 179, "stock": None, "stock_unit": None, "promotion": "限时优惠"},

    {"name": "宝宝面霜", "category": "母婴", "audience": "宝妈", "pains": ["宝宝脸干", "换季泛红", "皮肤受刺激"], "selling_points": ["无香精", "长效保湿", "舒缓配方"], "benefits": ["宝宝用着放心", "小脸润多了"], "price": 49, "original_price": 99, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "奶瓶", "category": "母婴", "audience": "新手妈妈", "pains": ["宝宝胀气", "奶瓶难洗", "容易摔坏"], "selling_points": ["PPSU材质", "防胀气设计", "宽口径"], "benefits": ["清洗很方便", "宝宝喝得顺"], "price": 89, "original_price": 169, "stock": 35, "stock_unit": "个", "promotion": None},
    {"name": "婴儿浴巾", "category": "母婴", "audience": "宝妈", "pains": ["浴巾不透气", "吸水差", "越洗越硬"], "selling_points": ["6层纱布", "柔软亲肤", "吸水快"], "benefits": ["宝宝裹着舒服", "洗了也不硬"], "price": 59, "original_price": 119, "stock": None, "stock_unit": None, "promotion": "买2送1"},
    {"name": "儿童牙刷", "category": "母婴", "audience": "宝妈", "pains": ["宝宝不爱刷牙", "刷毛太硬", "手柄难握"], "selling_points": ["软毛设计", "防滑手柄", "2支装"], "benefits": ["宝宝愿意刷了", "用着很顺手"], "price": 25.9, "original_price": 49.9, "stock": None, "stock_unit": None, "promotion": None},
    {"name": "爬行垫", "category": "母婴", "audience": "宝妈", "pains": ["地面太凉", "垫子打滑", "收纳占地方"], "selling_points": ["加厚设计", "防滑底面", "可折叠"], "benefits": ["宝宝爬着舒服", "收起来不占地"], "price": 129, "original_price": 259, "stock": 18, "stock_unit": "张", "promotion": None},
    {"name": "学步鞋", "category": "母婴", "audience": "宝妈", "pains": ["宝宝走路不稳", "鞋底太硬", "闷脚"], "selling_points": ["软底设计", "防滑鞋底", "透气网面"], "benefits": ["宝宝走得稳多了", "穿着不闷脚"], "price": 79, "original_price": 159, "stock": None, "stock_unit": None, "promotion": "限时优惠"},
]

PROMOTIONS = ["限时优惠", "限时半价", "买2送1"]
DURATIONS = ["30秒", "1分钟", "2分钟"]


def build_prompt(product: Dict, selling_points: List[str], duration: str, audience: Optional[str], variant: str) -> Tuple[str, Dict]:
    parts = [f"产品：{product['name']}", f"卖点：{'、'.join(selling_points)}", f"时长：{duration}"]
    if audience:
        parts.append(f"目标人群：{audience}")
    facts: Dict = {"product": product["name"], "selling_points": list(selling_points), "duration": duration, "audience": audience, "price": None, "original_price": None, "stock": None, "stock_unit": None, "promotion": []}
    if variant in {"price", "price_stock", "price_promo"}:
        facts["price"] = product["price"]
        parts.append(f"价格：{product['price']:g}元")
        if product.get("original_price"):
            facts["original_price"] = product["original_price"]
            parts.append(f"原价：{product['original_price']:g}元")
    if variant in {"stock", "price_stock"} and product.get("stock"):
        facts["stock"] = product["stock"]
        facts["stock_unit"] = product["stock_unit"]
        parts.append(f"最后{product['stock']}{product['stock_unit']}")
    if variant in {"promo", "price_promo"}:
        promotion = product.get("promotion") or "限时优惠"
        facts["promotion"] = [promotion]
        parts.append(promotion)
    return "；".join(parts), facts


def build_reference(product: Dict, selling_points: List[str], facts: Dict, rng: random.Random) -> str:
    audience = facts.get("audience") or "姐妹们"
    pain = rng.choice(product["pains"])
    benefit = rng.choice(product["benefits"])

    opener = rng.choice([
        f"今天简单聊聊{product['name']}。",
        f"今天说说这款{product['name']}。",
        f"今天给大家带来{product['name']}。",
    ])
    pain_line = rng.choice([
        f"有没有{audience}最怕{pain}？",
        f"{audience}是不是也遇到过{pain}？",
        f"有没有发现{pain}真的很烦？",
    ])
    pain_extra = rng.choice([
        "每次遇到都特别尴尬。",
        "这种感觉太折磨人了。",
        "很多人因此不敢出门。",
    ])
    lead_point = selling_points[0]
    rest_points = selling_points[1:]
    solution = f"这款{product['name']}主打{lead_point}。"
    solution_extra = f"还有{'、'.join(rest_points)}，用着很省心。" if rest_points else "用起来真的很省心。"
    benefit_line = rng.choice([
        f"{benefit}，真的绝了。",
        f"{benefit}，真的太香了。",
        f"{benefit}，效果真的绝了。",
    ])
    interaction_mid = rng.choice([
        "听我说，姐妹们看过来。",
        "家人们，先别划走。",
        "想要的在公屏打想要。",
    ])
    interaction_check = rng.choice([
        "是不是很心动呢？",
        "对不对？",
        "有没有同感？",
    ])

    price = facts.get("price")
    original_price = facts.get("original_price")
    promotion = (facts.get("promotion") or [None])[0]
    if price is not None:
        if original_price:
            price_line = f"平时{original_price:g}元，直播间只要{price:g}元，太香了！"
        else:
            price_line = f"直播间只要{price:g}元，太香了！"
    elif promotion:
        price_line = f"直播间{promotion}，太香了！"
    else:
        price_line = "价格以直播间为准，喜欢的别错过。"

    stock = facts.get("stock")
    urgent_promo = bool(promotion and any(word in promotion for word in ["限时", "倒计时", "最后", "半价"]))
    if stock:
        urgency = f"最后{stock}{facts.get('stock_unit') or '单'}，卖完就没了！"
    elif urgent_promo:
        urgency = "手慢无，赶紧下单吧！"
    else:
        urgency = "喜欢就赶紧下单吧！"

    interaction_cta = rng.choice([
        "想要的扣1，让我看看。",
        "想要的打想要，我给你们留。",
        "听懂的扣1。",
    ])
    final_cta = rng.choice([
        "点击下方链接，赶紧下单！",
        "购物车已经上好，马上去拍！",
        "赶紧拍下，别犹豫！",
    ])
    return "".join([opener, pain_line, pain_extra, solution, solution_extra, benefit_line, interaction_mid, interaction_check, price_line, urgency, interaction_cta, final_cta])


def evaluate_reference(evaluator: LiveStreamEvaluator, prompt: str, output: str, facts: Dict) -> Tuple[bool, List[str]]:
    metrics = evaluator.evaluate_single(output, prompt, facts=facts)
    failures = [key for key in TARGETS if not _target_passes(key, metrics[key])]
    return not failures, failures


def build_split(products: List[Dict], count: int, rng: random.Random, evaluator: LiveStreamEvaluator) -> Tuple[List[Dict], Dict]:
    per_product = max(1, count // len(products))
    samples: List[Dict] = []
    seen_prompts = set()
    failures: Counter = Counter()
    attempts = 0
    max_attempts = count * 40

    while len(samples) < count and attempts < max_attempts:
        for product in products:
            if len(samples) >= count:
                break
            made = 0
            local_attempts = 0
            while made < per_product and local_attempts < 80:
                attempts += 1
                local_attempts += 1
                k = rng.choice([2, 3, min(4, len(product["selling_points"]))])
                k = max(2, min(k, len(product["selling_points"])))
                selling_points = rng.sample(product["selling_points"], k)
                duration = rng.choice(DURATIONS)
                audience = product.get("audience") if rng.random() < 0.65 else None
                variant = rng.choices(
                    ["none", "price", "stock", "promo", "price_stock", "price_promo"],
                    weights=[40, 20, 12, 10, 9, 9],
                    k=1,
                )[0]
                if variant in {"stock", "price_stock"} and not product.get("stock"):
                    variant = "price" if variant == "price_stock" else "none"
                if variant in {"promo", "price_promo"} and not product.get("promotion"):
                    variant = "price" if variant == "price_promo" else "none"
                prompt, facts = build_prompt(product, selling_points, duration, audience, variant)
                if prompt in seen_prompts:
                    continue
                output = build_reference(product, selling_points, facts, rng)
                passed, failed = evaluate_reference(evaluator, prompt, output, parse_prompt_facts(prompt, facts))
                if not passed:
                    failures.update(failed)
                    continue
                seen_prompts.add(prompt)
                samples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": output},
                    ],
                    "meta": {"product": product["name"], "category": product["category"], "variant": variant},
                })
                made += 1
        if len(samples) >= count:
            break

    stats = {
        "requested": count,
        "written": len(samples),
        "prompt_unique": len(seen_prompts),
        "prompt_unique_ratio": round(len(seen_prompts) / max(len(samples), 1), 3),
        "attempts": attempts,
        "reference_metric_passes": len(samples),
        "reference_metric_pass_ratio": round(len(samples) / max(len(samples) + sum(failures.values()), 1), 3),
        "failure_counts": dict(failures),
    }
    return samples, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/sft_v4")
    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--valid-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    evaluator = LiveStreamEvaluator()
    shuffled = list(PRODUCTS)
    rng.shuffle(shuffled)
    valid_products = shuffled[:6]
    train_products = shuffled[6:]
    if len(train_products) * 2 < args.train_size // 20:
        raise SystemExit("not enough products for a product-disjoint split")

    train_samples, train_stats = build_split(train_products, args.train_size, rng, evaluator)
    valid_samples, valid_stats = build_split(valid_products, args.valid_size, rng, evaluator)
    if len(train_samples) < args.train_size * 0.9 or len(valid_samples) < args.valid_size * 0.9:
        raise SystemExit(f"dataset generation fell short: train={len(train_samples)} valid={len(valid_samples)}")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("train.jsonl", train_samples), ("valid.jsonl", valid_samples)]:
        with open(out_dir / filename, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    benchmark_products = set()
    with open(ROOT / "data" / "benchmark_prompts.jsonl", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            benchmark_products.add(item["facts"]["product"])

    metadata = {
        "version": "sft-v4",
        "seed": args.seed,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "system_prompt": SYSTEM_PROMPT,
        "train_products": [product["name"] for product in train_products],
        "valid_products": [product["name"] for product in valid_products],
        "benchmark_product_overlap": sorted(
            {product["name"] for product in PRODUCTS} & benchmark_products
        ),
        "train": train_stats,
        "valid": valid_stats,
        "split": "product-disjoint; train and valid share no product",
        "validation_gate": "every reference passes all v3 targets",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
