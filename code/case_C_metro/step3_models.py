"""案例 C 第三步：基准与模型。加权 MAE：高峰（7-9、17-19 时）或大站的低估误差权重为 2。"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
TOP = ["Times Sq-42 St", "Grand Central-42 St", "Fulton St"]
BASE = ["station", "hour", "dow", "month", "holiday", "days_to_holiday", "lag_7d", "lag_14d",
        "lag_21d", "lag_28d", "mean_4w", "trend_7_28", "game", "hours_from_start"]


def weighted_mae(y, p, peak, top):
    err = p - y
    w = np.where((err < 0) & (peak | top), 2.0, 1.0)
    m = ~np.isnan(y) & ~np.isnan(p)
    return float(np.sum(w[m] * np.abs(err[m])) / np.sum(w[m]))


def splits(df):
    df = df[df.date >= "2022-03-01"]
    tr = df[df.date < "2024-01-01"]
    va = df[(df.date >= "2024-01-01") & (df.date < "2024-04-01")]
    te = df[(df.date >= "2024-04-01") & (df.date < "2024-07-01")]
    return tr.dropna(subset=["entries"]), va.dropna(subset=["entries"]), te.dropna(subset=["entries"])


def day_type(d):
    lab = np.select([d.game == 1, (d.holiday == 1) | (d.days_to_holiday <= 1), d.dow >= 5],
                    [1, 2, 3], 0)                     # 先用整数，避免字符串被截断
    return pd.Series(lab, index=d.index).map({0: "regular_weekday", 1: "game_day", 2: "holiday±1", 3: "weekend"})


def fit(tr, feats, loss="absolute_error", q=None, seed=0):
    kw = dict(loss=loss, max_iter=600, learning_rate=0.05, categorical_features="from_dtype", random_state=seed)
    if q is not None:
        kw["quantile"] = q
    return HistGradientBoostingRegressor(**kw).fit(tr[feats], tr.entries)


def evaluate(te, cols):
    peak = te.hour.isin([7, 8, 9, 17, 18, 19]).to_numpy()
    top = te.station.isin(TOP).to_numpy()
    y = te.entries.to_numpy()
    out = {}
    for c in cols:
        p = te[c].to_numpy()
        row = {"wMAE": weighted_mae(y, p, peak, top)}
        for t in ["regular_weekday", "weekend", "holiday±1", "game_day"]:
            m = (te.dtype == t).to_numpy()
            row[t] = weighted_mae(y[m], p[m], peak[m], top[m])
        out[c] = row
    print("各组样本数：", te.dtype.value_counts().to_dict())
    return pd.DataFrame(out).T.round(1)


if __name__ == "__main__":
    df = pd.read_parquet(RAW.parent / "mta_features.parquet")
    tr, va, te = splits(df)
    print(f"train {tr.date.min().date()}~{tr.date.max().date()} n={len(tr):,}; valid n={len(va):,}; test {te.date.min().date()}~{te.date.max().date()} n={len(te):,}")
    res = {}
    for name, d in [("valid", va.copy()), ("test", te.copy())]:
        d["dtype"] = day_type(d)
        d["B1_last_week"], d["B2_mean_4w"] = d.lag_7d, d.mean_4w
        m_obs = fit(tr, BASE + ["obs_precip", "obs_tmax"])
        m_fc = fit(tr, BASE + ["fcst_precip", "fcst_tmax"])
        m_asym = fit(tr, BASE + ["fcst_precip", "fcst_tmax"], loss="quantile", q=2 / 3)
        d["GBDT_obs_weather"] = m_obs.predict(d[BASE + ["obs_precip", "obs_tmax"]])
        d["GBDT_fcst_weather"] = m_fc.predict(d[BASE + ["fcst_precip", "fcst_tmax"]])
        d["GBDT_fcst_q0.67"] = m_asym.predict(d[BASE + ["fcst_precip", "fcst_tmax"]])
        print(f"\n[{name}] 加权 MAE（人次/小时）")
        print(evaluate(d, ["B1_last_week", "B2_mean_4w", "GBDT_obs_weather", "GBDT_fcst_weather", "GBDT_fcst_q0.67"]).to_string())
        res[name] = d
    te = res["test"]
    # 预测区间：10% 与 90% 分位数
    F = BASE + ["fcst_precip", "fcst_tmax"]
    lo, hi = fit(tr, F, "quantile", 0.1).predict(te[F]), fit(tr, F, "quantile", 0.9).predict(te[F])
    cover = ((te.entries >= lo) & (te.entries <= hi))
    print(f"\n80% 预测区间实际覆盖率：{cover.mean():.3f}；按日期类型：", cover.groupby(te.dtype).mean().round(3).to_dict())
    te.assign(lo=lo, hi=hi).to_parquet(RAW.parent / "mta_test_pred.parquet")
