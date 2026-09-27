"""第 11 章：从 NHTSA 车辆安全投诉接口下载真实投诉文本，构建带部件标签的公开练习数据。

接口：https://api.nhtsa.gov/complaints/complaintsByVehicle?make=..&model=..&modelYear=..
只保留“只涉及一个部件类别”的投诉，并限定在若干常见部件类别中；
输出 data/raw/nhtsa_complaints.csv（text, label, odi）。
"""
import time
from pathlib import Path
import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
VEHICLES = [("toyota", "camry"), ("toyota", "rav4"), ("honda", "civic"), ("honda", "cr-v"), ("ford", "f-150"),
            ("ford", "escape"), ("chevrolet", "silverado"), ("chevrolet", "equinox"), ("nissan", "rogue"),
            ("nissan", "altima"), ("hyundai", "elantra"), ("kia", "sorento"), ("jeep", "grand cherokee"),
            ("tesla", "model 3"), ("subaru", "outback"), ("ram", "1500")]
YEARS = [2018, 2019, 2020, 2021]
CLASSES = ["AIR BAGS", "ELECTRICAL SYSTEM", "ENGINE", "SERVICE BRAKES", "STEERING", "POWER TRAIN",
           "STRUCTURE", "SEAT BELTS"]


def fetch(make, model, year):
    for attempt in range(4):
        try:
            r = requests.get("https://api.nhtsa.gov/complaints/complaintsByVehicle",
                             params={"make": make, "model": model, "modelYear": year}, timeout=60)
            return r.json().get("results", [])
        except (requests.RequestException, ValueError):
            time.sleep(3 * (attempt + 1))
    return []


if __name__ == "__main__":
    rows = []
    for make, model in VEHICLES:
        for y in YEARS:
            for c in fetch(make, model, y):
                rows.append({"odi": c["odiNumber"], "make": make, "model": model, "year": y,
                             "components": c.get("components", ""), "text": c.get("summary", "")})
    df = pd.DataFrame(rows).drop_duplicates("odi")
    single = df[~df.components.str.contains(",") & df.components.isin(CLASSES) & (df.text.str.len() > 80)]
    single = single.rename(columns={"components": "label"})
    single.to_csv(RAW / "nhtsa_complaints.csv", index=False)
    print(f"下载 {len(df):,} 条投诉，其中单一部件且属于 {len(CLASSES)} 类的有 {len(single):,} 条")
    print(single.label.value_counts().to_string())
