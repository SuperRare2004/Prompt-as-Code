"""第 11 章：在 NHTSA 投诉评价集（160 条，8 类各 20 条）上评价大语言模型分类结果与 TF-IDF 基线。

LLM 结果文件 misc/nhtsa_eval/nhtsa_llm_outputs.jsonl：每行 {"odi", "category", "evidence"}。
本书报告的结果由 Claude 在写作会话中按 nhtsa_eval/nhtsa_guide.txt 的说明逐条盲分类得到（未查看标签）；
读者可用 ch11_llm_text_classification.py 通过 API 重新生成该文件。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EVAL = Path(__file__).resolve().parent / "nhtsa_eval"          # 书中使用的评价集与 LLM 输出（随代码一起发布）
gold = pd.read_csv(EVAL / "nhtsa_gold.csv", dtype={"odi": str})
train = pd.read_csv(RAW / "nhtsa_complaints.csv", dtype={"odi": str})
train = train[~train.odi.isin(gold.odi)]                        # 训练集 = 其余全部投诉
out = pd.read_json(EVAL / "nhtsa_llm_outputs.jsonl", lines=True, dtype={"odi": str})
d = gold.merge(out, on="odi", how="left")
labels = sorted(gold.label.unique())

print(f"评价集 {len(gold)} 条；训练集 {len(train):,} 条")
print("\n== 大语言模型（少样本说明 + 边界规则，盲分类）")
print(f"准确率={accuracy_score(d.label, d.category):.3f} 宏平均F1={f1_score(d.label, d.category, average='macro'):.3f}")
print(classification_report(d.label, d.category, digits=3, zero_division=0))
ev_ok = [e in t for e, t in zip(d.evidence, d.text)]
print(f"evidence 为原文子串的比例：{np.mean(ev_ok):.3f}")
cm = pd.DataFrame(confusion_matrix(d.label, d.category, labels=labels), index=labels, columns=labels)
print("\n混淆矩阵（行=NHTSA 部件字段，列=LLM）：\n", cm.to_string())
print("\n不一致样本：")
for _, r in d[d.label != d.category].iterrows():
    print(f"  {r.odi} 标签={r.label:<18s} LLM={r.category:<18s} | {r.text[:110]}")

print("\n== TF-IDF + 逻辑回归基线")
for n in (500, 2000, len(train)):
    tr = train.sample(n, random_state=0) if n < len(train) else train
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(vec.fit_transform(tr.text), tr.label)
    p = clf.predict(vec.transform(gold.text))
    print(f"训练样本 {n:>6,}：准确率={accuracy_score(gold.label, p):.3f} 宏平均F1={f1_score(gold.label, p, average='macro'):.3f}")

# 参照标签本身是否一致？看两类边界情形在训练集中的标签分布
print("\n== 参照标签的一致性（训练集）")
t = train.text.str.lower()
for name, pat in [("提到 stall/stalled 的投诉", r"\bstall"), ("提到自动紧急制动/幻影刹车的投诉", r"automatic emergency brak|phantom brak|automatic brak")]:
    sub = train[t.str.contains(pat, regex=True)]
    print(f"{name}（{len(sub)} 条）：", (sub.label.value_counts(normalize=True).round(3)).head(4).to_dict())
