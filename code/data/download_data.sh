#!/usr/bin/env bash
# 下载本书配套案例使用的公开数据。所有数据均来自官方或论文作者公开发布的来源。
set -e
cd "$(dirname "$0")"; mkdir -p raw; cd raw

# 案例 A：METR-LA 洛杉矶高速公路线圈检测器速度数据（Li et al., 2018, DCRNN）
# 207 个检测器，2012-03-01 至 2012-06-27，5 分钟粒度，单位 mph；0 表示缺失。
curl -L -o METR-LA.csv "https://zenodo.org/record/5146275/files/METR-LA.csv?download=1"
BASE=https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph
curl -L -o graph_sensor_locations.csv $BASE/graph_sensor_locations.csv
curl -L -o distances_la_2012.csv      $BASE/distances_la_2012.csv
curl -L -o graph_sensor_ids.txt       $BASE/graph_sensor_ids.txt

# 案例 B：英国交通部 STATS19 道路交通伤亡事故数据（最近五年）
DFT=https://data.dft.gov.uk/road-accidents-safety-data
curl -L -o stats19_collision_5y.csv $DFT/dft-road-casualty-statistics-collision-last-5-years.csv
curl -L -o stats19_vehicle_5y.csv   $DFT/dft-road-casualty-statistics-vehicle-last-5-years.csv
curl -L -o stats19_casualty_5y.csv  $DFT/dft-road-casualty-statistics-casualty-last-5-years.csv

# 案例 C：纽约大都会运输署（MTA）地铁分小时客流，经 Socrata API 按车站聚合下载
cd ../../case_C_metro && python fetch_mta.py && python fetch_context.py

# 第 11 章：NHTSA 车辆安全投诉（公开接口）
cd ../misc && python fetch_nhtsa.py

# 第 9 章：RDD2022 的四个国家子集（从 figshare 上 13 GB 的总压缩包中按成员分段读取，约 1.4 GB）
cd ../misc && python fetch_rdd2022.py
