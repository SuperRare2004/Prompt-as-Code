"""第 1 章 / 第 17 章：AI 的默认建模 vs 经过字段审查的建模（案例 B）。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score
from sklearn.model_selection import train_test_split
from common import load_collisions, load_involvement, time_split, PRE_CAT, POST_OR_LABEL

c = load_collisions()
print("记录数：", len(c), "年份：", c.collision_year.min(), "-", c.collision_year.max())
print("KSI 占比（全体）：", round(c.ksi.mean(), 4))
print("KSI 占比按年份：", c.groupby("collision_year").ksi.mean().round(4).to_dict())
print("KSI 占比按报告系统（collision_injury_based: 1=基于伤情的系统如 CRASH/COPA, 0=警员自行判断）：",
      c.groupby("collision_injury_based").ksi.mean().round(4).to_dict(),
      c.collision_injury_based.value_counts().to_dict())
print("按警员是否到场：", c.groupby("did_police_officer_attend_scene_of_accident").ksi.mean().round(3).to_dict())

# ---------- 1) AI 默认做法：所有数值列都当特征，随机划分，只看 Accuracy / AUC ----------
num = c.select_dtypes("number").drop(columns=["ksi", "collision_severity", "collision_year",
                                                "location_easting_osgr", "location_northing_osgr",
                                                "longitude", "latitude"], errors="ignore")
X, y = num, c["ksi"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
m = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
p = m.predict_proba(Xte)[:, 1]
print("\n[默认做法] 特征数", X.shape[1], " Accuracy=%.3f ROC-AUC=%.3f PR-AUC=%.3f Recall@0.5=%.3f" % (
    accuracy_score(yte, p > 0.5), roc_auc_score(yte, p), average_precision_score(yte, p), recall_score(yte, p > 0.5)))
print("全部预测为非 KSI 的 Accuracy=%.3f" % (1 - yte.mean()))
from sklearn.inspection import permutation_importance
imp = permutation_importance(m, Xte.sample(20000, random_state=0), yte.loc[Xte.sample(20000, random_state=0).index],
                             scoring="average_precision", n_repeats=3, random_state=0)
print("默认模型置换重要性 Top8：\n", pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(8).round(4).to_string())

# ---------- 2) 审查后：只用事故前条件 + 涉事方；时间划分；类别特征按类别处理 ----------
c = c.join(load_involvement(), on="collision_index")
INV = ["inv_pedestrian", "inv_motorcycle", "inv_pedal_cycle", "inv_hgv"]
feats = PRE_CAT + INV
d = c[feats + ["ksi", "collision_year"]].copy()
for f in PRE_CAT:
    d[f] = d[f].astype(str).astype("category")
tr, va, te = time_split(d)
m2 = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0).fit(tr[feats], tr["ksi"])
p2 = m2.predict_proba(te[feats])[:, 1]
print("\n[审查后] 训练 2021-2023，测试 2025：ROC-AUC=%.3f PR-AUC=%.3f（随机猜测 PR-AUC≈%.3f）" % (
    roc_auc_score(te.ksi, p2), average_precision_score(te.ksi, p2), te.ksi.mean()))
