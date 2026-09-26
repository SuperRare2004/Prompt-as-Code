"""第 17 章综合案例二：逐项复现并修正 AI 在事故分析中的典型错误（STATS19）。

每一节对应书中的一类错误，打印“错误做法”与“修正做法”的对比。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from common import RAW, load_collisions, load_involvement, time_split, PRE_CAT

c = load_collisions()
inv = load_involvement()
c = c.join(inv, on="collision_index")
INV = list(inv.columns)


def auc_pr(y, p):
    return f"ROC-AUC={roc_auc_score(y, p):.3f} PR-AUC={average_precision_score(y, p):.3f}"


def as_cat(d, cols):
    d = d.copy()
    for f in cols:
        d[f] = d[f].astype(str).astype("category")
    return d


tr, va, te = time_split(c)

# E1 标签派生字段 ----------------------------------------------------------
print("== E1 标签派生字段")
for f in ["collision_adjusted_severity_serious", "collision_adjusted_severity_slight", "enhanced_severity_collision"]:
    print(f"  {f}: 与 KSI 的 ROC-AUC（单字段）=", round(roc_auc_score(c.ksi, c[f].fillna(-1)), 3))

# E2 事后字段 --------------------------------------------------------------
print("== E2 事后字段（警员是否到场、伤亡人数）")
base = PRE_CAT + INV
post = base + ["did_police_officer_attend_scene_of_accident", "number_of_casualties"]
for name, feats in [("事前字段", base), ("事前 + 事后字段", post)]:
    d_tr, d_te = as_cat(tr, PRE_CAT), as_cat(te, PRE_CAT)
    m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0).fit(d_tr[feats], d_tr.ksi)
    print(f"  {name:<12s}", auc_pr(d_te.ksi, m.predict_proba(d_te[feats])[:, 1]))

# E3 编码当数值 ------------------------------------------------------------
print("== E3 把类别编码当作连续数值（逻辑回归）")
codes = ["light_conditions", "weather_conditions", "road_type", "junction_detail", "urban_or_rural_area", "speed_limit"]
Xn_tr, Xn_te = tr[codes + INV].clip(lower=-1), te[codes + INV].clip(lower=-1)
lr_num = LogisticRegression(max_iter=2000).fit(Xn_tr, tr.ksi)
coef = dict(zip(codes, lr_num.coef_[0][:len(codes)]))
print("  数值编码：", auc_pr(te.ksi, lr_num.predict_proba(Xn_te)[:, 1]),
      " light_conditions 系数=%.3f（意为“编码每 +1”，无业务含义）" % coef["light_conditions"])
Xo_tr = pd.get_dummies(tr[codes].astype(str), dtype=float).join(tr[INV])
Xo_te = pd.get_dummies(te[codes].astype(str), dtype=float).join(te[INV]).reindex(columns=Xo_tr.columns, fill_value=0)
lr_oh = LogisticRegression(max_iter=3000).fit(Xo_tr, tr.ksi)
print("  独热编码：", auc_pr(te.ksi, lr_oh.predict_proba(Xo_te)[:, 1]))

# E4 报告系统变化导致的标签漂移 ----------------------------------------------
print("== E4 标签口径随报告系统变化")
share = c.groupby("collision_year").collision_injury_based.mean().round(3)
print("  基于伤情的报告系统占比（按年）：", share.to_dict())
print("  KSI 占比 按 年份 x 报告系统：\n", c.pivot_table(index="collision_year", columns="collision_injury_based", values="ksi").round(3))
print("  DfT 调整后严重程度（均值）按年份：", c.groupby("collision_year").collision_adjusted_severity_serious.mean().round(3).to_dict())
d_tr, d_te = as_cat(tr, PRE_CAT), as_cat(te, PRE_CAT)
m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0).fit(d_tr[base], d_tr.ksi)
p = m.predict_proba(d_te[base])[:, 1]
print("  2025 测试集：平均预测 KSI 概率=%.3f，实际=%.3f" % (p.mean(), d_te.ksi.mean()))
for k, g in d_te.assign(p=p).groupby("collision_injury_based"):
    print(f"    injury_based={k}: 预测 {g.p.mean():.3f} vs 实际 {g.ksi.mean():.3f} (n={len(g):,})")

# E5 车辆级记录的随机划分：同一事故出现在训练集和测试集 ---------------------
print("== E5 车辆级数据：随机划分 vs 按事故分组划分")
v = pd.read_csv(RAW / "stats19_vehicle_5y.csv", usecols=["collision_index", "vehicle_type", "age_of_driver",
                                                         "sex_of_driver", "vehicle_manoeuvre", "engine_capacity_cc"], low_memory=False)
vv = v.merge(c[["collision_index", "ksi", "longitude", "latitude", "hour_band", "speed_limit", "light_conditions"]], on="collision_index")
vv = vv[vv.collision_index.isin(c[c.collision_year <= 2023].collision_index)].sample(400_000, random_state=0)
F5 = ["vehicle_type", "age_of_driver", "sex_of_driver", "vehicle_manoeuvre", "engine_capacity_cc",
      "longitude", "latitude", "speed_limit", "light_conditions"]
print(f"  车辆级记录中，每起事故平均 {vv.groupby('collision_index').size().mean():.2f} 条")
Xtr_, Xte_, ytr_, yte_ = train_test_split(vv[F5], vv.ksi, test_size=0.2, random_state=0)
m = HistGradientBoostingClassifier(max_iter=500, max_leaf_nodes=255, random_state=0).fit(Xtr_, ytr_)
print("  随机划分：", auc_pr(yte_, m.predict_proba(Xte_)[:, 1]))
gi, gj = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(vv, groups=vv.collision_index))
m = HistGradientBoostingClassifier(max_iter=500, max_leaf_nodes=255, random_state=0).fit(vv.iloc[gi][F5], vv.iloc[gi].ksi)
print("  按事故分组：", auc_pr(vv.iloc[gj].ksi, m.predict_proba(vv.iloc[gj][F5])[:, 1]))

# E6 数量 vs 比例：只看 KSI 数量排序 -----------------------------------------
print("== E6 KSI 数量与 KSI 比例的排序不同（且两者都未考虑暴露量）")
t = c.groupby(["urban_or_rural_area", "speed_limit"]).ksi.agg(["sum", "mean", "size"])
t = t[t["size"] > 2000].sort_values("sum", ascending=False)
print(t.rename(columns={"sum": "KSI数", "mean": "KSI比例", "size": "事故数"}).round(3).head(8).to_string())
