#!/usr/bin/env python3
"""
直播话术数据生成器
生成 200 条 SFT 训练数据 + 100 条 DPO 偏好对
"""

import json
import random
import os

# 固定随机种子，保证可复现
random.seed(42)

# ========== 1. 基础数据定义 ==========

# System prompt
SYSTEM_PROMPT = """你是一位抖音直播带货话术专家。你的风格特征：
①短句为主，每句不超过15字
②每3句话插入一次互动指令（"扣1"、"对不对"、"听我说"）
③先讲痛点场景，再给解决方案，最后价格逼单
④情绪递进：平静→兴奋→急促
⑤口头禅：家人们、真的绝了、最后一波、听我说、有没有
⑥数字要具体：价格、数量、时间都要明确说出"""

# 10 个品类
CATEGORIES = [
    {
        "name": "防晒霜",
        "pain_points": ["夏天脸反光", "妆花成二维码", "晒黑两个度", "皮肤刺痛脱皮"],
        "solutions": ["3秒成膜", "SPF50+防护", "清爽不油腻", "养肤不闷痘"],
        "prices": {"原价": 199, "直播价": 79, "赠品": "送同款小样"},
        "scenes": ["通勤路上", "海边度假", "户外运动", "学车暴晒"]
    },
    {
        "name": "抗皱面霜",
        "pain_points": ["眼角纹能夹死蚊子", "法令纹深得能存水", "脸垮得像沙皮", "拍照显老十岁"],
        "solutions": ["玻色因成分", "赫莲娜同款", "7天淡纹", "紧致提拉"],
        "prices": {"原价": 399, "直播价": 168, "赠品": "送精华小样"},
        "scenes": ["熬夜加班", "换季干燥", "约会救急", "拍照上镜"]
    },
    {
        "name": "蓝牙耳机",
        "pain_points": ["地铁听不清", "健身房总掉", "通话对方听不清", "电量焦虑"],
        "solutions": ["主动降噪", "狂甩不掉", "通话清晰", "续航30小时"],
        "prices": {"原价": 299, "直播价": 99, "赠品": "送保护套"},
        "scenes": ["通勤地铁", "健身房", "视频会议", "深夜追剧"]
    },
    {
        "name": "面膜",
        "pain_points": ["皮肤干到起皮", "上妆卡粉", "脸色暗沉", "毛孔粗大"],
        "solutions": ["玻尿酸补水", "急救修复", "提亮肤色", "收缩毛孔"],
        "prices": {"原价": 129, "直播价": 49, "赠品": "买2送1"},
        "scenes": ["熬夜急救", "约会前", "换季维稳", "日常保养"]
    },
    {
        "name": "零食大礼包",
        "pain_points": ["嘴巴寂寞", "追剧没零食", "办公室下午茶", "孩子总要吃零食"],
        "solutions": ["10袋装超值", "网红爆款集合", "独立包装", "好吃不贵"],
        "prices": {"原价": 89, "直播价": 39.9, "赠品": "再送2袋"},
        "scenes": ["追剧", "办公室", "出游", "宅家"]
    },
    {
        "name": "洗衣液",
        "pain_points": ["衣服发黄", "汗味洗不掉", "宝宝衣服不敢乱洗", "泡沫难漂净"],
        "solutions": ["亮白增艳", "99%除菌", "母婴适用", "易漂洗"],
        "prices": {"原价": 79, "直播价": 39.9, "赠品": "送衣架"},
        "scenes": ["日常洗衣", "宝宝衣物", "运动服", "内衣"]
    },
    {
        "name": "电动牙刷",
        "pain_points": ["手动刷不干净", "牙龈出血", "口臭尴尬", "牙齿泛黄"],
        "solutions": ["声波震动", "5种模式", "30天续航", "IPX7防水"],
        "prices": {"原价": 249, "直播价": 99, "赠品": "送4个刷头"},
        "scenes": ["早晚洗漱", "出差旅行", "送礼", "家庭装"]
    },
    {
        "name": "保温杯",
        "pain_points": ["水凉得快", "杯子漏水", "内胆生锈", "太重不想带"],
        "solutions": ["12小时保温", "防漏设计", "316不锈钢", "轻量便携"],
        "prices": {"原价": 159, "直播价": 69, "赠品": "送杯刷"},
        "scenes": ["办公室", "户外运动", "出差", "送长辈"]
    },
    {
        "name": "收纳盒",
        "pain_points": ["桌面乱成狗窝", "找东西翻半天", "衣柜塞不下", "化妆品东倒西歪"],
        "solutions": ["多层分类", "透明可视", "可折叠", "颜值超高"],
        "prices": {"原价": 99, "直播价": 39.9, "赠品": "送标签贴"},
        "scenes": ["桌面收纳", "衣柜整理", "化妆品收纳", "文具收纳"]
    },
    {
        "name": "运动水壶",
        "pain_points": ["塑料杯有异味", "运动时漏水", "容量太小", "不保温"],
        "solutions": ["Tritan材质", "密封防漏", "1L大容量", "保温保冷"],
        "prices": {"原价": 129, "直播价": 59, "赠品": "送背带"},
        "scenes": ["健身", "户外徒步", "骑行", "日常通勤"]
    }
]

# 5 种直播风格
STYLES = [
    {
        "name": "激情逼单型",
        "特征": "语速快、情绪高、逼单感强",
        "patterns": [
            "来！{action}给我{signal}！",
            "听我说！{pain}！对不对？",
            "今天！就今天！{price}！",
            "最后一波！{urgency}！",
            "3、2、1——上链接！"
        ],
        "互动词": ["扣1", "扣2", "扣个666", "给我刷个火箭", "疯狂扣1"],
        "口头禅": ["真的绝了", "我跟你说", "家人们", "最后一波", "听我说"]
    },
    {
        "name": "温柔种草型",
        "特征": "语速慢、亲切、像朋友聊天",
        "patterns": [
            "姐妹们，你们有没有{pain}？",
            "我跟你们说，这个{product}真的好用到哭",
            "{solution}，用完你会回来谢我",
            "现在{price}，真的太划算了",
            "你们一定要试试，不试后悔"
        ],
        "互动词": ["有同感的扣1", "有没有", "是不是", "对不对", "同款扣1"],
        "口头禅": ["姐妹们", "真的好用", "安利给你们", "用完你会回来谢我", "太香了"]
    },
    {
        "name": "专业科普型",
        "特征": "讲成分、讲原理、有说服力",
        "patterns": [
            "我给大家科普一下，{product}为什么好",
            "你们知道{pain}的原因吗？",
            "关键在于{solution}",
            "这个成分{benefit}",
            "所以{conclusion}，现在{price}"
        ],
        "互动词": ["听懂的扣1", "学到了吗", "涨知识的扣1", "明白的扣1", "懂了的扣1"],
        "口头禅": ["划重点", "敲黑板", "记住了", "知识点", "干货来了"]
    },
    {
        "name": "故事型",
        "特征": "先讲故事，再引出产品，代入感强",
        "patterns": [
            "我之前{pain}，真的崩溃",
            "直到我遇到了{product}",
            "{solution}，用了一周我就惊了",
            "现在{price}，我自己都囤了{number}个",
            "你们一定要试试，我自己就是活广告"
        ],
        "互动词": ["有同款经历的扣1", "有没有同感", "跟我一样的扣1", "也是这样的扣1", "同款扣1"],
        "口头禅": ["我跟你说", "真的", "不骗你们", "亲测有效", "自己都在用"]
    },
    {
        "name": "对比型",
        "特征": "对比其他产品，突出优势",
        "patterns": [
            "你们用过{competitor}吗？{pain}对不对",
            "今天这个{product}完全不一样",
            "{solution}，吊打市面上90%的产品",
            "别人卖{high_price}，我们只要{low_price}",
            "这个性价比，不买真的亏"
        ],
        "互动词": ["用过的扣1", "有没有同感", "对比过的扣1", "是不是真的", "对不对"],
        "口头禅": ["对比一下", "吊打", "性价比之王", "真的不一样", "秒杀"]
    }
]

# 互动指令模板
INTERACTION_TEMPLATES = [
    "来！{target}给我{signal}！",
    "{target}在哪里？给我{signal}！",
    "有没有{target}？{signal}让我看看！",
    "{target}的姐妹{signal}！",
    "所有{target}，{signal}！"
]

TARGETS = ["油皮姐妹", "干皮姐妹", "学生党", "宝妈", "打工人", "熬夜党", "健身党", "颜值党"]

SIGNALS = ["扣1", "扣个1", "扣个666", "疯狂扣1", "刷个爱心", "给我刷个赞"]

# 逼单话术
URGENCY_PHRASES = [
    "库存只剩最后{number}单了",
    "{number}单卖完就没了",
    "倒计时{number}秒",
    "卖完真的补不到货",
    "下次开播不知道什么时候了"
]


def generate_interaction(style, target=None, signal=None):
    """生成互动指令"""
    if target is None:
        target = random.choice(TARGETS)
    if signal is None:
        signal = random.choice(SIGNALS)
    template = random.choice(INTERACTION_TEMPLATES)
    return template.format(target=target, signal=signal)


def generate_urgency(number=None):
    """生成逼单话术"""
    if number is None:
        number = random.choice([10, 20, 30, 50, 100])
    template = random.choice(URGENCY_PHRASES)
    return template.format(number=number)


def generate_sft_sample(category, style, duration="1分钟"):
    """生成一条 SFT 样本"""
    # 随机选择痛点和解决方案
    pain = random.choice(category["pain_points"])
    solution = random.choice(category["solutions"])
    scene = random.choice(category["scenes"])
    price = category["prices"]["直播价"]
    original_price = category["prices"]["原价"]
    gift = category["prices"]["赠品"]

    # 生成目标人群
    target = random.choice(TARGETS)

    # 根据风格生成话术
    if style["name"] == "激情逼单型":
        script = f"来！{target}给我扣1！让我看看有多少人{pain}的？对不对？{scene}的时候最怕这个了！我跟你说，今天这个{category['name']}真的绝了——{solution}！平时专柜{original_price}，今天直播间，家人们，前50单直接{price}！{gift}！最后一波库存了！来，3、2、1——上链接！"

    elif style["name"] == "温柔种草型":
        script = f"姐妹们，你们有没有{pain}的烦恼？我之前也是，{scene}的时候特别明显。后来我发现了这个{category['name']}，真的好用到哭。{solution}，用完你会回来谢我。现在{price}，比专柜便宜一半还多，{gift}。你们一定要试试，不试真的后悔。"

    elif style["name"] == "专业科普型":
        script = f"家人们，我给大家科普一下，为什么你{pain}？关键在于{scene}的时候没有用对产品。今天这个{category['name']}，{solution}，能有效解决这个问题。所以如果你想要好效果，现在{price}就能拿下，{gift}。划重点，这个真的值得囤。"

    elif style["name"] == "故事型":
        script = f"我跟你们说，我之前{pain}，真的崩溃。每次{scene}都特别烦。直到我遇到了这个{category['name']}，{solution}，用了一周我就惊了。现在我自己都囤了3个。今天{price}，{gift}。你们一定要试试，我自己就是活广告，不骗你们。"

    elif style["name"] == "对比型":
        script = f"你们用过其他{category['name']}吗？{pain}对不对？今天这个完全不一样，{solution}，吊打市面上大部分产品。别人卖{original_price}，我们只要{price}，{gift}。这个性价比，不买真的亏，有同感的扣1！"

    # 插入互动指令（每3句左右插入一次）
    sentences = script.split("。")
    final_sentences = []
    for i, sent in enumerate(sentences):
        if sent.strip():
            final_sentences.append(sent.strip())
            # 每2-3句插入互动
            if i > 0 and i % 2 == 0 and i < len(sentences) - 1:
                final_sentences.append(generate_interaction(style, target))

    script = "！".join(final_sentences) + "！"

    # 构建完整的对话
    user_prompt = f"产品：{category['name']}；卖点：{solution}；时长：{duration}；目标人群：{target}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": script}
        ]
    }


def generate_dpo_sample(category, style):
    """生成一条 DPO 偏好对"""
    pain = random.choice(category["pain_points"])
    solution = random.choice(category["solutions"])
    scene = random.choice(category["scenes"])
    price = category["prices"]["直播价"]
    original_price = category["prices"]["原价"]
    target = random.choice(TARGETS)

    prompt = f"产品：{category['name']}；卖点：{solution}；生成1分钟直播话术"

    # chosen: 高能量直播风格
    if style["name"] == "激情逼单型":
        chosen = f"来！{target}给我扣1！{pain}的姐妹有没有？对不对？今天这个{category['name']}真的绝了——{solution}！平时{original_price}，今天直播间{price}！最后一波！来，3、2、1！"
    elif style["name"] == "温柔种草型":
        chosen = f"姐妹们，{pain}是不是特别烦？我懂，{scene}的时候真的崩溃。今天这个{category['name']}，{solution}，用完你会回来谢我。现在{price}，真的太划算了，你们一定要试试。"
    elif style["name"] == "专业科普型":
        chosen = f"划重点！{pain}的原因是什么？{scene}没用对产品！今天这个{category['name']}，{solution}，能有效解决。现在{price}就能拿下，知识点来了，记住了！"
    elif style["name"] == "故事型":
        chosen = f"我跟你们说，我之前{pain}，真的崩溃。直到遇到这个{category['name']}，{solution}，用了一周我就惊了。今天{price}，我自己都囤了，你们一定要试试，不骗你们。"
    else:
        chosen = f"你们用过其他{category['name']}吗？{pain}对不对？今天这个{solution}，吊打市面产品！别人{original_price}，我们{price}！性价比之王，有同感的扣1！"

    # rejected: 平淡说明书风格
    rejected = f"这款{category['name']}含有{solution}成分，能够有效改善{pain}问题。经过测试，在{scene}场景下效果显著。现在购买享受{price}优惠价格（原价{original_price}），欢迎下单购买。"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


def generate_augmented_sample(base_sample, variations=3):
    """对一条样本进行扩充（同义替换）"""
    augmented = [base_sample]

    # 定义替换词
    replacements = {
        "真的绝了": ["太绝了", "绝绝子", "太神了", "好用到哭"],
        "家人们": ["姐妹们", "宝子们", "宝贝们", "亲们"],
        "最后一波": ["最后机会", "卖完就没了", "库存告急", "最后XX单"],
        "扣1": ["扣个1", "扣111", "疯狂扣1", "刷个1"],
        "我跟你说": ["我跟你们说", "听我说", "划重点", "敲黑板"],
        "有没有": ["有同感的", "是不是", "对不对", "同款的"]
    }

    for _ in range(variations - 1):
        new_sample = json.loads(json.dumps(base_sample))  # 深拷贝
        content = new_sample["messages"][2]["content"]

        # 随机替换一些词
        for old_word, new_words in replacements.items():
            if old_word in content and random.random() > 0.5:
                content = content.replace(old_word, random.choice(new_words), 1)

        new_sample["messages"][2]["content"] = content
        augmented.append(new_sample)

    return augmented


def split_data(data, train_ratio=0.8):
    """分割数据为训练集和验证集"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("直播话术数据生成器")
    print("=" * 50)

    # 1. 生成 50 条核心 SFT 样本（10 品类 × 5 风格）
    print("\n[1/4] 生成 50 条核心 SFT 样本...")
    sft_samples = []
    for cat in CATEGORIES:
        for style in STYLES:
            sample = generate_sft_sample(cat, style)
            sft_samples.append(sample)
    print(f"  生成 {len(sft_samples)} 条核心样本")

    # 2. 扩充到 200 条
    print("\n[2/4] 扩充到 200 条...")
    augmented_samples = []
    for sample in sft_samples:
        augmented = generate_augmented_sample(sample, variations=4)
        augmented_samples.extend(augmented)

    # 如果超过 200 条，随机采样
    if len(augmented_samples) > 200:
        augmented_samples = random.sample(augmented_samples, 200)
    print(f"  扩充后共 {len(augmented_samples)} 条")

    # 3. 分割训练集和验证集
    print("\n[3/4] 分割训练集和验证集...")
    train_data, valid_data = split_data(augmented_samples, train_ratio=0.85)
    print(f"  训练集：{len(train_data)} 条")
    print(f"  验证集：{len(valid_data)} 条")

    # 4. 保存 SFT 数据
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for sample in valid_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  已保存：{train_path}")
    print(f"  已保存：{valid_path}")

    # 5. 生成 DPO 偏好对
    print("\n[4/4] 生成 DPO 偏好对...")
    dpo_samples = []
    for cat in random.sample(CATEGORIES, 7):  # 随机选 7 个品类
        for style in random.sample(STYLES, 3):  # 每个品类 3 种风格
            sample = generate_dpo_sample(cat, style)
            dpo_samples.append(sample)

    # 扩充到 100 条
    while len(dpo_samples) < 100:
        cat = random.choice(CATEGORIES)
        style = random.choice(STYLES)
        sample = generate_dpo_sample(cat, style)
        dpo_samples.append(sample)

    dpo_samples = dpo_samples[:100]

    dpo_path = os.path.join(output_dir, "dpo.jsonl")
    with open(dpo_path, "w", encoding="utf-8") as f:
        for sample in dpo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  生成 {len(dpo_samples)} 条 DPO 偏好对")
    print(f"  已保存：{dpo_path}")

    # 6. 打印样例
    print("\n" + "=" * 50)
    print("样例数据")
    print("=" * 50)

    print("\n【SFT 样例】")
    sample = augmented_samples[0]
    print(f"User: {sample['messages'][1]['content']}")
    print(f"Assistant: {sample['messages'][2]['content'][:200]}...")

    print("\n【DPO 样例】")
    sample = dpo_samples[0]
    print(f"Prompt: {sample['prompt']}")
    print(f"Chosen: {sample['chosen'][:150]}...")
    print(f"Rejected: {sample['rejected'][:150]}...")

    print("\n" + "=" * 50)
    print("数据生成完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
