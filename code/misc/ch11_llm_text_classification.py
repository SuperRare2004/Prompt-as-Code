"""第 11 章案例：用大语言模型做交通投诉文本分类与信息抽取，并在金标准上评价。

输入：一个 CSV 文件，至少包含 text（原文，已脱敏）与 label（人工标注类别）两列；
     可选 location（人工标注的地点，名录中的标准名称，未提及则为空）。
     公开练习数据可用 NHTSA 车辆安全投诉数据库（https://www.nhtsa.gov/nhtsa-datasets-and-apis）：
     以投诉描述为 text、以官方归类的故障部件为 label，字段说明见其随数据发布的字段定义文件。
用法：python ch11_llm_text_classification.py gold.csv --gazetteer stations.txt --repeats 3

LLM 调用使用 Anthropic Python SDK（pip install anthropic pydantic），需要配置 API 凭据。
换用其他模型服务时，只需替换 classify_with_llm()；评价部分与模型无关。
"""
import argparse
import json
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

PERIODS = ["早高峰", "平峰", "晚高峰", "夜间", "未提及"]


def build_prompt(guide, examples, text):
    return f"""你是交通投诉文本的分类助手。请阅读下面的文本，按照分类说明完成分类，并抽取关键信息。

【分类说明】
{guide}

【示例】
{examples}

【抽取规则】
1. location：只有当文本中明确出现地点名称、且能唯一对应名录中的标准名称时才填写；否则填 null，不得推测。
2. time_period：只能从 {PERIODS} 中选择，不得生成具体时刻。
3. category：从给定类别中选择 1 个主类别；确实涉及两类时，可在 secondary 中填写第二类。
4. evidence：逐字摘录原文中支持分类的句子，不得改写。

【文本】
{text}"""


def classify_with_llm(prompt, categories):
    import anthropic
    from pydantic import BaseModel

    class Result(BaseModel):
        category: Literal[tuple(categories)]            # 取值限定在给定类别内
        secondary: Optional[str]
        location: Optional[str]
        time_period: Literal[tuple(PERIODS)]
        evidence: str

    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model="claude-opus-5",
        max_tokens=1024,
        output_config={"effort": "low"},                 # 简单分类任务使用较低 effort
        messages=[{"role": "user", "content": prompt}],
        output_format=Result,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        return None
    return resp.parsed_output.model_dump()


def tfidf_baseline(train, test):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)   # 字符 n-gram，适用于中文
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(train.text), train.label)
    return clf.predict(vec.transform(test.text))


def evaluate(gold, outputs, gazetteer=None):
    pred = [o["category"] if o else "__failed__" for o in outputs]
    print("格式合规率：", np.mean([o is not None for o in outputs]).round(3))
    print("准确率：", round(accuracy_score(gold.label, pred), 3), " 宏平均 F1：", round(f1_score(gold.label, pred, average="macro"), 3))
    print(classification_report(gold.label, pred, zero_division=0))
    ev_ok = [o is not None and o["evidence"] in t for o, t in zip(outputs, gold.text)]
    print("evidence 为原文子串的比例：", np.mean(ev_ok).round(3))
    if gazetteer is not None:
        locs = [o["location"] if o else None for o in outputs]
        in_gaz = [l is None or l in gazetteer for l in locs]
        print("地点在名录中（或为 null）的比例：", np.mean(in_gaz).round(3))
        if "location" in gold:
            no_loc = gold.location.isna().to_numpy()
            fabricated = [l is not None for l, n in zip(locs, no_loc) if n]
            print("原文无地点却填写了地点（编造率）：", np.mean(fabricated).round(3) if fabricated else "n/a")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--train", help="用于 TF-IDF 基线的人工归类样本；缺省时从 gold 中划分")
    ap.add_argument("--guide", default="guide.txt", help="分类说明文本文件")
    ap.add_argument("--examples", default="examples.txt", help="少样本示例文本文件")
    ap.add_argument("--gazetteer", help="地点名录，每行一个标准名称")
    ap.add_argument("--repeats", type=int, default=1, help="重复运行次数，用于检查稳定性")
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args()

    gold = pd.read_csv(a.gold)
    train = pd.read_csv(a.train) if a.train else None
    if train is None:
        train, gold = train_test_split(gold, test_size=0.3, random_state=0, stratify=gold.label)
    print("== TF-IDF + 逻辑回归基线")
    base = tfidf_baseline(train, gold)
    print("准确率：", round(accuracy_score(gold.label, base), 3), " 宏平均 F1：", round(f1_score(gold.label, base, average="macro"), 3))
    if a.baseline_only:
        raise SystemExit
    guide, examples = open(a.guide).read(), open(a.examples).read()
    gaz = set(open(a.gazetteer).read().split("\n")) if a.gazetteer else None
    cats = sorted(train.label.unique())
    runs = [[classify_with_llm(build_prompt(guide, examples, t), cats) for t in gold.text] for _ in range(a.repeats)]
    print("== 大语言模型（第 1 次运行）")
    evaluate(gold, runs[0], gaz)
    if a.repeats > 1:
        cats_by_run = np.array([[o["category"] if o else None for o in r] for r in runs])
        unstable = (cats_by_run != cats_by_run[0]).any(axis=0)
        print(f"重复 {a.repeats} 次运行中类别发生变化的样本比例：{unstable.mean():.3f}")
        gold.assign(unstable=unstable).to_csv("llm_unstable_samples.csv", index=False)
    with open("llm_outputs.jsonl", "w") as f:
        for o in runs[0]:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
