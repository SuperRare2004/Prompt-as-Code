"""前言开篇案例：随机划分 + 居中滑动窗口造成的数据泄露。

复现“一天完成的预测模型”：先按 AI 的默认做法（随机 8:2 划分、居中滑动平均），
再按正确做法（时间顺序划分、只用过去数据），比较结果。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from common import load_speed, to_long, H

wide = load_speed()
df = to_long(wide)
g = df.groupby("sensor")["speed"]
df["target"] = g.shift(-H)
for k in range(1, 6):
    df[f"lag{k}"] = g.shift(k)
df["hour"] = df["timestamp"].dt.hour
df["dow"] = df["timestamp"].dt.dayofweek
# AI 的默认写法：前后 30 分钟居中滑动平均（包含未来 15 分钟！）
df["roll_center"] = g.transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())
# 正确写法：只使用过去 30 分钟
df["roll_past"] = g.transform(lambda s: s.rolling(7, min_periods=1).mean())
df = df.dropna(subset=["target", "speed"])

base = ["speed", "lag1", "lag2", "lag3", "lag4", "lag5", "hour", "dow"]


def fit_eval(tr, te, feats, name):
    m = HistGradientBoostingRegressor(max_iter=200, random_state=0)
    m.fit(tr[feats], tr["target"])
    p = m.predict(te[feats])
    print(f"{name:<32s} R2={r2_score(te['target'], p):.3f}  "
          f"MAE={mean_absolute_error(te['target'], p):.2f} mph")


# 1) AI 默认方案：随机划分 + 居中窗口
tr, te = train_test_split(df, test_size=0.2, random_state=0)
fit_eval(tr, te, base + ["roll_center"], "random split + centered window")
# 1b) 只保留随机划分这一处问题
fit_eval(tr, te, base + ["roll_past"], "random split + past window")
# 2) 只修正划分方式
cut = df["timestamp"].quantile(0.8)
tr, te = df[df.timestamp < cut], df[df.timestamp >= cut]
fit_eval(tr, te, base + ["roll_center"], "time split + centered window")
# 3) 两处都修正
fit_eval(tr, te, base + ["roll_past"], "time split + past window")
# 4) 持续性基准：15 分钟后 = 现在
print(f"{'persistence baseline (time split)':<32s} "
      f"R2={r2_score(te['target'], te['speed']):.3f}  "
      f"MAE={mean_absolute_error(te['target'], te['speed']):.2f} mph")
print("test period:", te.timestamp.min(), "->", te.timestamp.max())

# 5) 如果把 0 当作真实速度（未按数据说明处理缺失）
raw = to_long(load_speed(zero_as_missing=False))
print("zero readings:", f"{(raw.speed == 0).mean():.1%} of all records;",
      "timestamps where ALL sensors read 0:",
      int((load_speed(zero_as_missing=False) == 0).all(axis=1).sum()))
g0 = raw.groupby("sensor")["speed"]
raw["target"] = g0.shift(-H)
for k in range(1, 6):
    raw[f"lag{k}"] = g0.shift(k)
raw["hour"], raw["dow"] = raw.timestamp.dt.hour, raw.timestamp.dt.dayofweek
raw["roll_past"] = g0.transform(lambda s: s.rolling(7, min_periods=1).mean())
raw = raw.dropna(subset=["target"]).copy()
tr0, te0 = raw[raw.timestamp < cut], raw[raw.timestamp >= cut]
m = HistGradientBoostingRegressor(max_iter=200, random_state=0)
m.fit(tr0[base + ["roll_past"]], tr0["target"])
p0 = m.predict(te0[base + ["roll_past"]])
print(f"{'zeros kept as speed (all rows)':<32s} R2={r2_score(te0['target'], p0):.3f}  "
      f"MAE={mean_absolute_error(te0['target'], p0):.2f} mph")
ok = (te0["target"] > 0).to_numpy() & (te0["speed"] > 0).to_numpy()
print(f"{'  same model, valid rows only':<32s} "
      f"MAE={mean_absolute_error(te0['target'][ok], p0[ok]):.2f} mph")

# 6) AI 的完整初始做法：0 当作真实车速 + 随机划分 + 居中窗口
raw["roll_center"] = g0.transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())
tr1, te1 = train_test_split(raw, test_size=0.2, random_state=0)
m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(tr1[base + ["roll_center"]], tr1["target"])
p1 = m.predict(te1[base + ["roll_center"]])
print(f"{'AI initial: zeros kept + random + centered':<32s} R2={r2_score(te1['target'], p1):.3f}  "
      f"MAE={mean_absolute_error(te1['target'], p1):.2f} mph")
