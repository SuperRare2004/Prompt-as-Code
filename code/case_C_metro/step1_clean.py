"""案例 C 第一步：数据描述与清洗（纽约 MTA 分小时进站客流）。

先描述、后决策：本脚本打印支付方式构成、缺失小时、异常低值等现象，
清洗规则（见书中表格）由人确定后在 clean() 中实现。
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_raw():
    d = pd.read_csv(RAW / "mta_hourly_selected.csv", parse_dates=["transit_timestamp"])
    return d.rename(columns={"transit_timestamp": "ts"})


def describe(d):
    m = d.assign(month=d.ts.dt.to_period("M")).pivot_table(index="month", columns="payment_method",
                                                            values="ridership", aggfunc="sum")
    m["omny_share"] = m["omny"] / m.sum(axis=1)
    print("按月进站量（9 座车站合计，千人次）与 OMNY 占比：")
    print((m[["metrocard", "omny"]] / 1000).round(0).join(m.omny_share.round(3)).iloc[::3].to_string())
    mc = m["metrocard"]
    print(f"只看 MetroCard：2022-03 {mc.iloc[1]/1e3:.0f}k -> 2024-11 {mc.iloc[-2]/1e3:.0f}k "
          f"({mc.iloc[-2]/mc.iloc[1]-1:+.1%})；合计：{m[['metrocard','omny']].sum(axis=1).iloc[1]/1e3:.0f}k -> "
          f"{m[['metrocard','omny']].sum(axis=1).iloc[-2]/1e3:.0f}k")


def clean(d):
    """规则 C1：两种支付方式相加得到进站量；C2：补全完整的小时索引，缺失小时显式为 NaN（不填 0）。"""
    h = d.groupby(["station", "ts"])["ridership"].sum().rename("entries").reset_index()
    full = pd.MultiIndex.from_product([h.station.unique(),
                                       pd.date_range(h.ts.min().floor("D"), h.ts.max().ceil("D") - pd.Timedelta(hours=1), freq="h")],
                                      names=["station", "ts"])
    h = h.set_index(["station", "ts"]).reindex(full).reset_index()
    return h


if __name__ == "__main__":
    d = load_raw()
    print(d.groupby("station").ts.agg(["min", "max", "count"]))
    describe(d)
    h = clean(d)
    miss = h.entries.isna()
    print(f"\n补全小时索引后缺失 {miss.sum():,} / {len(h):,} 个车站-小时（{miss.mean():.2%}）")
    print("缺失最多的日期：\n", h[miss].ts.dt.date.value_counts().head(8).to_string())
    daily = h.groupby(["station", h.ts.dt.date]).entries.sum(min_count=20)
    med = daily.groupby("station").transform("median")
    low = daily[daily < 0.3 * med]
    print(f"\n日进站量低于该站中位数 30% 的车站-日：{len(low)}")
    print(low.reset_index().groupby("ts").size().sort_values(ascending=False).head(10).to_string())
    h.to_parquet(RAW.parent / "mta_clean.parquet")
