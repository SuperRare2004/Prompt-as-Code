"""案例 C 第二步：特征工程。预测时刻为 D-1 日 16:00，预测 D 日 5:00-23:00 各小时进站量。

信息可得性约束：历史客流只使用 D-2 日及以前的完整日，以及 D-7k（k=1..4）同一小时；
日历、已公布的 MLB 赛程可用；天气提供“实测”和“历史预报”两个版本，用于比较。
"""
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
STADIUM = {"161 St-Yankee Stadium": "yankee", "Mets-Willets Point": "mets"}
HOURS = range(5, 24)


def add_weekly_lags(df, lags=(7, 14, 21, 28)):
    """按日期平移后合并，而不是 shift(k)：即使某些行不存在也不会错位。"""
    base = df[["station", "date", "hour", "entries"]]
    for k in lags:
        s = base.copy()
        s["date"] = s["date"] + pd.Timedelta(days=k)
        df = df.merge(s.rename(columns={"entries": f"lag_{k}d"}), on=["station", "date", "hour"], how="left")
    cols = [f"lag_{k}d" for k in lags]
    df["mean_4w"] = df[cols].mean(axis=1, skipna=True)
    df.loc[df[cols].notna().sum(axis=1) < 2, "mean_4w"] = np.nan        # 有效值少于 2 个则置缺失
    return df


def add_recent_trend(df):
    """最近 7 个完整日（D-8..D-2）同小时均值 / 更早 28 天（D-36..D-9）同小时均值。"""
    wide = df.pivot_table(index=["station", "hour"], columns="date", values="entries")
    r7 = wide.T.rolling(7, min_periods=4).mean().shift(2)             # 截至 D-2
    r28 = wide.T.rolling(28, min_periods=14).mean().shift(9)          # 截至 D-9
    ratio = (r7 / r28).T.stack(future_stack=True).rename("trend_7_28").reset_index()
    return df.merge(ratio, on=["station", "hour", "date"], how="left")


def build():
    h = pd.read_parquet(RAW.parent / "mta_clean.parquet")
    h["date"], h["hour"] = h.ts.dt.normalize(), h.ts.dt.hour
    df = h[h.hour.isin(HOURS)][["station", "date", "hour", "entries"]].copy()
    df = add_weekly_lags(df)
    df = add_recent_trend(df)
    # 日历
    hol = USFederalHolidayCalendar().holidays("2022-01-01", "2025-01-31")
    df["dow"], df["month"] = df.date.dt.dayofweek, df.date.dt.month
    df["holiday"] = df.date.isin(hol).astype(int)
    d2h = np.array([(hol - d).days for d in df.date.drop_duplicates()])
    near = pd.Series([int(np.min(np.abs(x))) for x in d2h], index=df.date.drop_duplicates())
    df["days_to_holiday"] = df.date.map(near).clip(upper=7)
    # MLB 主场比赛（只对球场车站有效）
    g = pd.read_csv(RAW / "mlb_home_games.csv")
    g["start_local"] = pd.to_datetime(g.game_utc, utc=True).dt.tz_convert("America/New_York")
    g["date"] = pd.to_datetime(g.official_date)
    g["start_hour"] = g.start_local.dt.hour
    g = g.groupby(["venue", "date"]).start_hour.min().reset_index()
    df["venue"] = df.station.map(STADIUM)
    df = df.merge(g, on=["venue", "date"], how="left")
    df["game"] = df.start_hour.notna().astype(int)
    df["hours_from_start"] = (df.hour - df.start_hour).where(df.game == 1, -99)
    # 天气：实测与预报
    w = pd.read_csv(RAW / "nyc_weather_daily.csv", parse_dates=["date"])
    df = df.merge(w, on="date", how="left")
    df["station"] = df.station.astype("category")
    return df.drop(columns=["venue", "start_hour"])


if __name__ == "__main__":
    df = build()
    df.to_parquet(RAW.parent / "mta_features.parquet")
    print(df.shape)
    print(df[df.station == "161 St-Yankee Stadium"].groupby("game").entries.mean().round(0))
    print(df.isna().mean().round(3)[lambda s: s > 0].to_string())
