"""案例 B 公共函数：英国 STATS19 道路交通伤亡事故数据（DfT，最近五年）。

标签：collision_severity 1=死亡(fatal) 2=重伤(serious) 3=轻伤(slight)；
本书使用 KSI（killed or seriously injured）= 1 或 2 作为“严重事故”。
字段编码见 DfT 发布的 STATS19 数据指南（Road Safety Open Data Guide）。
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# 事故发生前即可知道的道路与环境条件（类别变量，编码值没有大小含义）
PRE_CAT = ["road_type", "speed_limit", "light_conditions", "weather_conditions",
           "road_surface_conditions", "urban_or_rural_area", "junction_detail",
           "junction_control", "first_road_class", "special_conditions_at_site",
           "carriageway_hazards", "pedestrian_crossing", "hour_band", "day_of_week",
           "trunk_road_flag"]
# 事故发生后才产生、或由标签派生的字段：不得作为特征
POST_OR_LABEL = ["did_police_officer_attend_scene_of_accident", "number_of_casualties",
                 "enhanced_severity_collision", "collision_injury_based",
                 "collision_adjusted_severity_serious", "collision_adjusted_severity_slight"]


def load_collisions():
    c = pd.read_csv(RAW / "stats19_collision_5y.csv", low_memory=False)
    c["ksi"] = (c["collision_severity"] <= 2).astype(int)
    hour = pd.to_numeric(c["time"].str[:2], errors="coerce")
    c["hour_band"] = pd.cut(hour, [-1, 5, 9, 15, 19, 23], labels=["0-5", "6-9", "10-15", "16-19", "20-23"]).astype(str)
    return c


def load_involvement():
    """由车辆表与伤亡表汇总“涉事方”信息：是否涉及行人、摩托车、自行车、重型货车。"""
    v = pd.read_csv(RAW / "stats19_vehicle_5y.csv", usecols=["collision_index", "vehicle_type"], low_memory=False)
    k = pd.read_csv(RAW / "stats19_casualty_5y.csv", usecols=["collision_index", "casualty_class"], low_memory=False)
    inv = pd.DataFrame({
        "inv_motorcycle": v.vehicle_type.isin([2, 3, 4, 5, 23, 97]).groupby(v.collision_index).any(),
        "inv_pedal_cycle": (v.vehicle_type == 1).groupby(v.collision_index).any(),
        "inv_hgv": (v.vehicle_type == 21).groupby(v.collision_index).any(),
    })
    inv["inv_pedestrian"] = (k.casualty_class == 3).groupby(k.collision_index).any()
    return inv.fillna(False).astype(int)


def time_split(c):
    tr = c[c.collision_year <= 2023]
    va = c[c.collision_year == 2024]
    te = c[c.collision_year == 2025]
    return tr, va, te
