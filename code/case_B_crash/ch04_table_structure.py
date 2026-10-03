"""第 4 章：STATS19 三张表的规模与一对多关系（用于观测单位示意图）。"""
from common import RAW
import pandas as pd

c = pd.read_csv(RAW / "stats19_collision_5y.csv", usecols=["collision_index", "collision_year"], low_memory=False)
v = pd.read_csv(RAW / "stats19_vehicle_5y.csv", usecols=["collision_index"], low_memory=False)
k = pd.read_csv(RAW / "stats19_casualty_5y.csv", usecols=["collision_index"], low_memory=False)
print("年份：", c.collision_year.min(), "-", c.collision_year.max())
print("事故表行数：", len(c))
print("车辆表行数：", len(v), " 平均每起事故车辆数：", round(len(v) / len(c), 2),
      " 单起事故最多车辆数：", v.collision_index.value_counts().max())
print("伤亡人员表行数：", len(k), " 平均每起事故伤亡人数：", round(len(k) / len(c), 2),
      " 单起事故最多伤亡人数：", k.collision_index.value_counts().max())
