"""第 8 章案例：逻辑回归与随机森林识别 KSI 事故条件（案例 B，STATS19）。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from common import load_collisions, load_involvement, time_split

c = load_collisions().join(load_involvement(), on="collision_index")
# 为便于解释，把编码合并为有业务含义的类别（参照类别写在第一个）
c["light"] = c.light_conditions.map({1: "daylight", 4: "dark_lit", 5: "dark_unlit", 6: "dark_no_lighting"}).fillna("light_unknown")
c["weather"] = c.weather_conditions.map({1: "fine", 2: "rain", 5: "rain", 3: "snow", 6: "snow", 7: "fog"}).fillna("weather_other")
c["area"] = c.urban_or_rural_area.map({1: "urban", 2: "rural"}).fillna("area_unknown")
c["road"] = c.road_type.map({6: "single_cw", 3: "dual_cw", 1: "roundabout", 2: "one_way", 7: "slip_road"}).fillna("road_other")
c["speed"] = c.speed_limit.where(c.speed_limit.isin([20, 30, 40, 50, 60, 70]))
c["junction"] = np.where(c.junction_detail == 0, "no_junction", "junction")
c["night"] = c.hour_band.isin(["0-5", "20-23"]).astype(int)
c = c.dropna(subset=["speed"])
CAT = {"light": "daylight", "weather": "fine", "area": "urban", "road": "single_cw", "junction": "no_junction"}
INV = ["inv_pedestrian", "inv_motorcycle", "inv_pedal_cycle", "inv_hgv"]


def design(d):
    X = pd.get_dummies(d[list(CAT)], dtype=float)
    X = X.drop(columns=[f"{k}_{v}" for k, v in CAT.items()])        # 去掉参照类别
    X["speed_per10"] = d["speed"] / 10
    X[INV + ["night"]] = d[INV + ["night"]].astype(float)
    return X


tr, va, te = time_split(c)
Xtr, Xte = design(tr), design(te).reindex(columns=design(tr).columns, fill_value=0)
ytr, yte = tr.ksi, te.ksi
print(f"训练集 {len(tr):,}（KSI {ytr.mean():.3f}），测试集 {len(te):,}（KSI {yte.mean():.3f}）")


def summarize(name, p, thr=0.5):
    print(f"{name:<22s} Acc={accuracy_score(yte, p > thr):.3f} Prec={precision_score(yte, p > thr):.3f} "
          f"Rec={recall_score(yte, p > thr):.3f} ROC-AUC={roc_auc_score(yte, p):.3f} PR-AUC={average_precision_score(yte, p):.3f} "
          f"mean_p={p.mean():.3f}")


lr0 = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
lr1 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
summarize("logit (unweighted)", lr0.predict_proba(Xte)[:, 1])
summarize("logit (balanced)", lr1.predict_proba(Xte)[:, 1])

# 优势比与 95% 置信区间（statsmodels，不加权模型）
res = sm.Logit(ytr.to_numpy(), sm.add_constant(Xtr)).fit(disp=0)
orr = pd.DataFrame({"OR": np.exp(res.params), "lo": np.exp(res.conf_int()[0]), "hi": np.exp(res.conf_int()[1])}).round(2)
print("\n优势比（参照：白天/晴/城市/单幅路/非交叉口）\n", orr.drop("const").to_string())

# 雨天“悖论”：分组看
print("\n雨天与晴天的 KSI 占比：", c[c.weather.isin(["fine", "rain"])].groupby("weather").ksi.mean().round(3).to_dict())
print(c[c.weather.isin(["fine", "rain"])].groupby(["area", "weather"]).ksi.mean().unstack().round(3))
print("雨天事故中城市道路占比 vs 晴天：", c[c.weather == "rain"].area.eq("urban").mean().round(3),
      c[c.weather == "fine"].area.eq("urban").mean().round(3))

# 随机森林
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=0).fit(Xtr, ytr)
prf = rf.predict_proba(Xte)[:, 1]
summarize("random forest", prf)
sub = Xte.sample(30000, random_state=0)
imp = permutation_importance(rf, sub, yte.loc[sub.index], scoring="average_precision", n_repeats=5, random_state=0, n_jobs=-1)
print("\n随机森林置换重要性 Top10：\n", pd.Series(imp.importances_mean, index=Xte.columns).sort_values(ascending=False).head(10).round(4).to_string())

# 交互：限速 x 光照 的 KSI 占比（原始数据）
print("\n限速 x 光照 KSI 占比：\n", c[c.light.isin(["daylight", "dark_no_lighting"])].pivot_table(index="speed", columns="light", values="ksi").round(3))

# 从 AUC 到决策：前 20% 覆盖多少 KSI；对照规则“农村 + 夜间”
te = te.assign(score=prf)
# 保存测试集得分，供绘制 PR 曲线（code/figures/make_figures.py）
from pathlib import Path
te.assign(p_logit=lr0.predict_proba(Xte)[:, 1])[["ksi", "score", "p_logit", "area", "night"]].to_parquet(Path(__file__).resolve().parents[1] / "data" / "B_ch08_test_scores.parquet")
top = te.score >= te.score.quantile(0.8)
rule = (te.area == "rural") & (te.night == 1)
print(f"\n模型前 20% 覆盖 KSI：{te.ksi[top].sum() / te.ksi.sum():.3f}")
print(f"规则“农村+夜间”选中 {rule.mean():.3f} 的事故，覆盖 KSI：{te.ksi[rule].sum() / te.ksi.sum():.3f}")
top_same = te.score >= te.score.quantile(1 - rule.mean())
print(f"模型选中同样比例（{rule.mean():.3f}）时覆盖 KSI：{te.ksi[top_same].sum() / te.ksi.sum():.3f}")

# 雨天“悖论”的进一步分解：涉事方构成是否不同？
w = c[c.weather.isin(["fine", "rain"])]
print("\n按天气的涉事方构成：\n", w.groupby("weather")[INV].mean().round(3))
print("\n农村道路：按天气 x 是否涉及摩托车 的 KSI 占比：\n",
      w[w.area == "rural"].pivot_table(index="inv_motorcycle", columns="weather", values="ksi").round(3))
