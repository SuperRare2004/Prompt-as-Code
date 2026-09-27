"""把 NHTSA 投诉划分为评价集（每类 20 条，分层随机抽样）与训练集（其余），
并单独输出不含标签的评价集文本，供大语言模型盲分类。"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
df = pd.read_csv(RAW / "nhtsa_complaints.csv", dtype={"odi": str})
gold = df.groupby("label", group_keys=False).sample(20, random_state=0).sample(frac=1, random_state=1)
train = df[~df.odi.isin(gold.odi)]
gold.to_csv(RAW / "nhtsa_gold.csv", index=False)
train.to_csv(RAW / "nhtsa_train.csv", index=False)
gold[["odi", "text"]].reset_index(drop=True).to_json(RAW / "nhtsa_gold_texts.jsonl", orient="records", lines=True, force_ascii=False)
print(len(gold), len(train))
