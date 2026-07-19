"""古诗词库 + 错别字生成 + 逐字比对。

- POEMS：内置若干首唐诗，每首含标题/作者/正确诗句(逐行)。
- 也支持从外部传入任意诗词文本(load_poem 的 lines 参数)。
- make_typo：在正确诗句里把某一个字替换成形近/音近错字，返回错字位置与正误字。
- compare_lines：把 OCR 识别到的逐行文字与正确诗句逐字比对，定位错字。

形近/音近混淆表 CONFUSABLES 是"制造错字"用的候选来源；比对本身不依赖它，
比对只看"识别字 != 正确字"。这样 detect 节点的 OCR 是真在识别，不是查表作弊。
"""
import random

# 内置诗库：lines 为不含标点的整句(逐句一行)，便于逐字比对与网格渲染。
POEMS = [
    {"title": "静夜思", "author": "李白",
     "lines": ["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡"]},
    {"title": "春晓", "author": "孟浩然",
     "lines": ["春眠不觉晓", "处处闻啼鸟", "夜来风雨声", "花落知多少"]},
    {"title": "登鹳雀楼", "author": "王之涣",
     "lines": ["白日依山尽", "黄河入海流", "欲穷千里目", "更上一层楼"]},
    {"title": "相思", "author": "王维",
     "lines": ["红豆生南国", "春来发几枝", "愿君多采撷", "此物最相思"]},
    {"title": "江雪", "author": "柳宗元",
     "lines": ["千山鸟飞绝", "万径人踪灭", "孤舟蓑笠翁", "独钓寒江雪"]},
]

# 形近 / 音近混淆字：制造"看起来/听起来像但写错"的错别字。值为候选列表。
CONFUSABLES = {
    "明": ["朋", "萌"], "月": ["夕", "肖"], "光": ["先", "洸"], "床": ["庆"],
    "疑": ["凝"], "霜": ["箱", "孀"], "举": ["兴"], "望": ["忘", "王"],
    "低": ["底", "抵"], "思": ["恩", "丝"], "故": ["古", "固"], "乡": ["夕"],
    "眠": ["珉"], "觉": ["党"], "晓": ["浇", "绕"], "闻": ["间", "问"],
    "啼": ["蹄", "缔"], "鸟": ["岛", "乌"], "风": ["凤"], "雨": ["两"],
    "花": ["华"], "落": ["洛"], "白": ["百"], "日": ["曰"], "依": ["衣"],
    "尽": ["近"], "河": ["何"], "海": ["每"], "流": ["留"], "穷": ["究"],
    "里": ["理"], "目": ["日"], "层": ["曾"], "楼": ["搂"], "红": ["江"],
    "豆": ["斗"], "南": ["难"], "国": ["困"], "来": ["未"], "枝": ["技"],
    "愿": ["原"], "君": ["群"], "采": ["彩"], "物": ["务"], "相": ["想"],
    "千": ["干"], "鸟飞": ["岛"], "绝": ["色"], "径": ["经"], "踪": ["宗"],
    "灭": ["减"], "孤": ["狐"], "舟": ["丹"], "翁": ["嗡"], "钓": ["钩"],
    "寒": ["塞"], "雪": ["雷"], "处": ["外"], "夜": ["液"],
}


def load_poem(index=None, lines=None, title="自定义", author=""):
    """取一首诗。lines 非空则用外部传入的任意诗词；否则按 index 取内置(默认随机)。"""
    if lines:
        clean = [str(s).strip() for s in lines if str(s).strip()]
        return {"title": title, "author": author, "lines": clean}
    if index is None:
        index = random.randrange(len(POEMS))
    return POEMS[index % len(POEMS)]


def make_typo(poem, seed=None):
    """在诗句中把一个字替换成形近/音近错字。

    返回 dict：
      wrong_lines: 含错字的逐行文本
      line, col:   错字所在行/列(0基)
      correct:     正确字
      wrong:       错字
    找不到可替换字时 line=col=-1(极少见)，wrong_lines == 正确诗句。
    """
    rng = random.Random(seed)
    lines = poem["lines"]
    # 收集所有"在混淆表里、且有单字候选"的位置
    cands = []
    for li, line in enumerate(lines):
        for ci, ch in enumerate(line):
            opts = [w for w in CONFUSABLES.get(ch, []) if len(w) == 1]
            if opts:
                cands.append((li, ci, ch, opts))
    wrong_lines = list(lines)
    if not cands:
        return {"wrong_lines": wrong_lines, "line": -1, "col": -1,
                "correct": "", "wrong": ""}
    li, ci, ch, opts = rng.choice(cands)
    wrong_ch = rng.choice(opts)
    row = list(wrong_lines[li])
    row[ci] = wrong_ch
    wrong_lines[li] = "".join(row)
    return {"wrong_lines": wrong_lines, "line": li, "col": ci,
            "correct": ch, "wrong": wrong_ch}


def compare_lines(ocr_lines, correct_lines):
    """逐行逐字比对，返回错字位置列表 [{line, col, correct, seen}, ...]。

    ocr_lines / correct_lines 均为逐行字符串。只在两者该行字数一致时逐字比，
    字数不一致的行跳过(OCR 分行/漏字时不误报)。
    """
    errors = []
    n = min(len(ocr_lines), len(correct_lines))
    for li in range(n):
        a, b = ocr_lines[li], correct_lines[li]
        if len(a) != len(b):
            continue
        for ci, (ca, cb) in enumerate(zip(a, b)):
            if ca != cb:
                errors.append({"line": li, "col": ci, "correct": cb, "seen": ca})
    return errors
