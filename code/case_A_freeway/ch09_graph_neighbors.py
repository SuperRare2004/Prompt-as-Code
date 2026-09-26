"""第 9 章综合案例：图结构怎么定义？直线距离邻居 vs 路网有向邻居（案例 A）。

DCRNN 公开的 distances_la_2012.csv 给出检测器之间沿路网的行驶距离（有方向，单位米）。
本脚本比较三种输入：只用本检测器历史；加入“直线距离最近的 k 个检测器”；
加入“沿路网下游最近的 k 个检测器”。模型统一使用梯度提升树。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from common import RAW, load_speed, to_long, split, report, onset_mask, H

K = 3
wide = load_speed()
sensors = list(wide.columns)
loc = pd.read_csv(RAW / "graph_sensor_locations.csv", dtype={"sensor_id": str}).set_index("sensor_id").loc[sensors]
dist = pd.read_csv(RAW / "distances_la_2012.csv", dtype={"from": str, "to": str})
dist = dist[dist["from"].isin(sensors) & dist["to"].isin(sensors) & (dist["from"] != dist["to"])]

# 1) 直线距离（haversine）最近邻：AI 常见的默认构图方式
lat, lon = np.radians(loc.latitude.to_numpy()), np.radians(loc.longitude.to_numpy())
dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
hav = 2 * 6371000 * np.arcsin(np.sqrt(np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2))
np.fill_diagonal(hav, np.inf)
eu_nb = {s: [sensors[j] for j in np.argsort(hav[i])[:K]] for i, s in enumerate(sensors)}

# 2) 路网有向最近邻：从 i 出发沿路网可达的下游检测器
road_nb = (dist.sort_values("cost").groupby("from")["to"].apply(lambda s: list(s[:K])).to_dict())
road_cost = dist.set_index(["from", "to"])["cost"]

# 诊断：直线最近邻中，有多少在 5 km 内沿路网双向都不可达（多为对向车道或相交的另一条高速）
unreach = 0
for s, nbs in eu_nb.items():
    for n in nbs:
        c1, c2 = road_cost.get((s, n), np.inf), road_cost.get((n, s), np.inf)
        unreach += min(c1, c2) > 5000
print(f"直线最近邻对数={K*len(sensors)}，其中路网 5km 内不可达的={unreach} ({unreach/(K*len(sensors)):.0%})")
print(f"有路网下游邻居的检测器数={len(road_nb)}/{len(sensors)}")

# ---------- 构造特征 ----------
cur = wide
lag1 = wide.shift(1)
df = to_long(wide)
g = df.groupby("sensor")["speed"]
df["target"] = g.shift(-H)
for k in range(1, 6):
    df[f"lag{k}"] = g.shift(k)
tod = df.timestamp.dt.hour * 12 + df.timestamp.dt.minute // 5
df["tod_sin"], df["tod_cos"] = np.sin(2 * np.pi * tod / 288), np.cos(2 * np.pi * tod / 288)
df["weekend"] = (df.timestamp.dt.dayofweek >= 5).astype(int)
OWN = ["speed"] + [f"lag{k}" for k in range(1, 6)] + ["tod_sin", "tod_cos", "weekend"]


def neighbor_feats(nb, tag):
    cols = []
    for r in range(K):
        m_cur = {s: (nbs[r] if r < len(nbs) else None) for s, nbs in nb.items()}
        for name, tab in [("cur", cur), ("lag1", lag1)]:
            c = f"{tag}{r}_{name}"
            long = tab.stack(future_stack=True).rename("v").reset_index()
            long.columns = ["timestamp", "nbsensor", "v"]
            key = pd.DataFrame({"sensor": list(m_cur), "nbsensor": list(m_cur.values())}).dropna()
            tmp = df[["timestamp", "sensor"]].merge(key, on="sensor", how="left").merge(long, on=["timestamp", "nbsensor"], how="left")
            df[c] = tmp["v"].to_numpy()
            cols.append(c)
    return cols


EU = neighbor_feats(eu_nb, "eu")
RD = neighbor_feats(road_nb, "rd")

tr, va, te = split(df)
tr = tr.dropna(subset=["target", "speed"]).sample(1_500_000, random_state=0)
te = te.dropna(subset=["target", "speed"])
print(f"测试集拥堵形成样本数={int(onset_mask(te.speed.to_numpy(), te.target.to_numpy()).sum())}")
for name, feats in [("own_only", OWN), ("own+euclid_nb", OWN + EU), ("own+road_downstream_nb", OWN + RD)]:
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0).fit(tr[feats], tr["target"])
    te[name] = m.predict(te[feats])
te["persist"] = te["speed"]
print(report(te, ["persist", "own_only", "own+euclid_nb", "own+road_downstream_nb"]).to_string(index=False))

# 保存每个检测器的测试 MAE，供绘制空间误差图（code/figures/make_figures.py）
from pathlib import Path
out = Path(__file__).resolve().parents[1] / "outputs"
err = te.assign(ae=(te["own+road_downstream_nb"] - te["target"]).abs(), ae_own=(te["own_only"] - te["target"]).abs())
per = err.groupby("sensor")[["ae", "ae_own"]].mean().join(loc[["latitude", "longitude"]])
per.rename(columns={"ae": "mae_road_nb", "ae_own": "mae_own"}).to_csv(out / "A_ch09_sensor_mae.csv")
