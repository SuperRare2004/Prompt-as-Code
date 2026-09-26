"""下载案例 C 的外部信息：
1) MLB 官方赛程 API：扬基队（Yankee Stadium，teamId=147）与大都会队（Citi Field，teamId=121）主场比赛；
2) Open-Meteo：纽约中央公园附近的逐日实测降水/气温（ERA5 再分析）与历史预报存档。
"""
from pathlib import Path
import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def mlb_home_games(team_id, venue, start="2022-01-01", end="2024-12-31"):
    r = requests.get("https://statsapi.mlb.com/api/v1/schedule",
                     params={"sportId": 1, "teamId": team_id, "startDate": start, "endDate": end}, timeout=120)
    rows = []
    for d in r.json()["dates"]:
        for g in d["games"]:
            if g["teams"]["home"]["team"]["id"] == team_id and g["status"]["abstractGameState"] == "Final" \
                    and g["gameType"] in ("R", "F", "D", "L", "W"):
                rows.append({"venue": venue, "official_date": g["officialDate"], "game_utc": g["gameDate"]})
    return pd.DataFrame(rows)


def weather(api, prefix):
    url = {"obs": "https://archive-api.open-meteo.com/v1/archive",
           "fcst": "https://historical-forecast-api.open-meteo.com/v1/forecast"}[api]
    r = requests.get(url, params={"latitude": 40.78, "longitude": -73.97, "start_date": "2022-01-01",
                                  "end_date": "2024-12-31", "daily": "precipitation_sum,temperature_2m_max",
                                  "timezone": "America/New_York"}, timeout=120)
    d = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date", "precipitation_sum": f"{prefix}_precip",
                                                         "temperature_2m_max": f"{prefix}_tmax"})
    return d


if __name__ == "__main__":
    # 注意：该 API 对跨多个赛季的日期范围只返回一个赛季，因此按赛季分别请求
    games = pd.concat([mlb_home_games(t, v, f"{y}-01-01", f"{y}-12-31")
                       for t, v in [(147, "yankee"), (121, "mets")] for y in (2022, 2023, 2024)])
    print(games.assign(year=games.official_date.str[:4]).groupby(["venue", "year"]).size())
    games.to_csv(RAW / "mlb_home_games.csv", index=False)
    print(games.groupby("venue").size())
    w = weather("obs", "obs").merge(weather("fcst", "fcst"), on="date")
    w.to_csv(RAW / "nyc_weather_daily.csv", index=False)
    print(w.describe().round(2))
