"""第 7 章案例：基准模型、岭回归与模型复杂度诊断（案例 A，METR-LA）。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from common import load_speed, to_long, split, report, mae, H

df = to_long(load_speed())
g = df.groupby("sensor")["speed"]              # 按检测器分组后再平移，避免跨检测器错位
df["target"] = g.shift(-H)
LAGS = [f"lag{k}" for k in range(1, 6)]
for k in range(1, 6):
    df[f"lag{k}"] = g.shift(k)
tod = df.timestamp.dt.hour * 12 + df.timestamp.dt.minute // 5
df["tod"] = tod
df["tod_sin"], df["tod_cos"] = np.sin(2 * np.pi * tod / 288), np.cos(2 * np.pi * tod / 288)
df["weekend"] = (df.timestamp.dt.dayofweek >= 5).astype(int)

tr, va, te = split(df)
for name, d in [("train", tr), ("valid", va), ("test", te)]:
    print(f"{name}: {d.timestamp.min()} -> {d.timestamp.max()}, rows={len(d):,}")

# ---------- 基准 ----------
profile = tr.groupby(["sensor", "weekend", "tod"])["target"].mean().rename("hist")  # 只用训练集


def add_baselines(d):
    d = d.join(profile, on=["sensor", "weekend", "tod"])
    d["persist"] = d["speed"]
    d["mix"] = d[["persist", "hist"]].mean(axis=1, skipna=False)
    return d


va, te = add_baselines(va), add_baselines(te)
va_eval = va.dropna(subset=["target", "speed"])
print("\n[验证集] 基准模型（mph）")
print(report(va_eval, ["persist", "hist", "mix"]).to_string(index=False))

# ---------- 岭回归 ----------
FEATS = ["speed"] + LAGS + ["tod_sin", "tod_cos", "weekend"]
trm = tr.dropna(subset=FEATS + ["target"])
vam = va_eval.dropna(subset=FEATS)
print("\n[岭回归] alpha 选择（验证集 MAE）")
best = None
for a in [0.01, 0.1, 1, 10, 100]:
    m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(trm[FEATS], trm["target"])
    tr_mae = mae(trm["target"].to_numpy(), m.predict(trm[FEATS]))
    va_mae = mae(vam["target"].to_numpy(), m.predict(vam[FEATS]))
    print(f"alpha={a:<6} train MAE={tr_mae:.3f}  valid MAE={va_mae:.3f}")
    if best is None or va_mae < best[1]:
        best = (a, va_mae, m)
ridge = best[2]
coef = pd.Series(ridge[-1].coef_, index=FEATS).sort_values(key=np.abs, ascending=False)
print("标准化系数：\n", coef.round(3).to_string())
vam = vam.assign(ridge=ridge.predict(vam[FEATS]))
print("\n[验证集] 基准 vs 岭回归")
print(report(vam, ["persist", "hist", "mix", "ridge"]).to_string(index=False))

# ---------- 复杂度诊断：回归树深度 ----------
sub = trm.sample(400_000, random_state=0)
print("\n[回归树] 深度 vs 训练/验证 MAE")
for d in [2, 4, 6, 8, 10, 12, 15, 20, 25]:
    t = DecisionTreeRegressor(max_depth=d, min_samples_leaf=1, random_state=0).fit(sub[FEATS], sub["target"])
    print(f"depth={d:<3} train={mae(sub['target'].to_numpy(), t.predict(sub[FEATS])):.3f}  "
          f"valid={mae(vam['target'].to_numpy(), t.predict(vam[FEATS])):.3f}")

# ---------- 测试集：只评估一次 ----------
tem = te.dropna(subset=FEATS + ["target"])
tem = tem.assign(ridge=ridge.predict(tem[FEATS]))
print(f"\n[测试集] 最终模型 alpha={best[0]}")
print(report(tem, ["persist", "hist", "mix", "ridge"]).to_string(index=False))
