"""案例 A 公共函数：数据读取、划分、评价。

数据：METR-LA（Li et al., 2018），洛杉矶县高速公路 207 个线圈检测器，
2012-03-01 至 2012-06-27，5 分钟平均速度（mph）。原始文件中 0 表示缺失。
"""
from pathlib import Path
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
H = 3                      # 预测步长：3 x 5 min = 15 min
TRAIN_END = "2012-05-15"   # 训练集：3/1 - 5/14
VALID_END = "2012-06-01"   # 验证集：5/15 - 5/31；测试集：6/1 - 6/27


def load_speed(zero_as_missing=True):
    """返回宽表：行为时间，列为检测器编号。"""
    df = pd.read_csv(RAW / "METR-LA.csv", index_col=0, parse_dates=True)
    df.columns = df.columns.astype(str)
    if zero_as_missing:
        df = df.replace(0.0, np.nan)   # 数据说明：0 为缺失，不是停滞
    return df


def to_long(wide):
    long = wide.stack(future_stack=True).rename("speed").reset_index()
    long.columns = ["timestamp", "sensor", "speed"]
    return long.sort_values(["sensor", "timestamp"]).reset_index(drop=True)


def split(df, col="timestamp"):
    tr = df[df[col] < TRAIN_END]
    va = df[(df[col] >= TRAIN_END) & (df[col] < VALID_END)]
    te = df[df[col] >= VALID_END]
    return tr.copy(), va.copy(), te.copy()


def period(ts):
    """洛杉矶高速公路的高峰时段：早 6-9 点，晚 15-19 点（工作日）。"""
    h = ts.dt.hour
    wd = ts.dt.dayofweek < 5
    return np.select([wd & h.between(6, 8), wd & h.between(15, 18)],
                     ["am_peak", "pm_peak"], "other")


def onset_mask(cur, target):
    """拥堵形成阶段：当前速度 >= 50 mph，15 分钟后 < 35 mph。"""
    return (cur >= 50) & (target < 35)


def mae(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    return float(np.mean(np.abs(y[m] - p[m])))


def rmse(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def report(df, pred_cols, target="target"):
    """总体、分时段、拥堵形成阶段的 MAE 表（单位 mph）。"""
    rows = []
    per = period(df["timestamp"])
    on = onset_mask(df["speed"].to_numpy(), df[target].to_numpy())
    for c in pred_cols:
        y, p = df[target].to_numpy(), df[c].to_numpy()
        rows.append({
            "method": c, "MAE": mae(y, p), "RMSE": rmse(y, p),
            "AM_peak": mae(y[per == "am_peak"], p[per == "am_peak"]),
            "PM_peak": mae(y[per == "pm_peak"], p[per == "pm_peak"]),
            "other": mae(y[per == "other"], p[per == "other"]),
            "onset": mae(y[on], p[on]),
        })
    return pd.DataFrame(rows).round(2)
