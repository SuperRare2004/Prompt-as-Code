"""下载案例 C 数据：纽约 MTA 地铁分小时客流（data.ny.gov，数据集 wujg-7c2s，
“MTA Subway Hourly Ridership: 2020-2024”）。

为控制数据量，只下载若干代表性车站，并在服务器端按“车站 x 小时 x 支付方式”聚合。
ridership 为进站刷卡/扫码次数（MetroCard 或 OMNY 非接触支付），不含出站。
"""
import time
from pathlib import Path
import pandas as pd
import requests

URL = "https://data.ny.gov/resource/wujg-7c2s.json"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "mta_hourly_selected.csv"
STATIONS = {
    "611": "Times Sq-42 St", "610": "Grand Central-42 St", "628": "Fulton St",
    "617": "Atlantic Av-Barclays Ctr", "604": "161 St-Yankee Stadium",
    "448": "Mets-Willets Point", "447": "Flushing-Main St", "616": "Jackson Hts-Roosevelt Av",
    "397": "86 St (4,5,6)",
}


def fetch_month(sid, start, end):
    params = {
        "$select": "transit_timestamp,payment_method,sum(ridership) as ridership",
        "$where": f"station_complex_id='{sid}' AND transit_timestamp between '{start}' and '{end}'",
        "$group": "transit_timestamp,payment_method",
        "$limit": 50000,
    }
    for attempt in range(5):
        try:
            r = requests.get(URL, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {sid} {start}")


def fetch_station(sid, first="2022-02-01", last="2024-12-31"):
    """按月分段查询（服务器端聚合大时间范围时很慢），再拼接。"""
    rows = []
    for m in pd.date_range(first, last, freq="MS"):
        end = (m + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%dT23:59:59")
        rows += fetch_month(sid, m.strftime("%Y-%m-%dT00:00:00"), end)
    df = pd.DataFrame(rows)
    df["station_complex_id"] = sid
    return df


if __name__ == "__main__":
    parts = []
    for sid, name in STATIONS.items():
        t = time.time()
        d = fetch_station(sid)
        print(f"{sid} {name}: {len(d):,} rows ({time.time() - t:.0f}s)", flush=True)
        parts.append(d)
    df = pd.concat(parts)
    df["ridership"] = df["ridership"].astype(float)
    df["station"] = df["station_complex_id"].map(STATIONS)
    df.to_csv(OUT, index=False)
    print("saved", OUT, len(df))
