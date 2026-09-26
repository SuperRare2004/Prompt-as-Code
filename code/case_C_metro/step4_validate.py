"""案例 C 第四步：验证（量纲与数量级、极端情景、分组、天气实测 vs 预报）。"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
te = pd.read_parquet(RAW.parent / "mta_test_pred.parquet")
df = pd.read_parquet(RAW.parent / "mta_features.parquet")
p = "GBDT_fcst_weather"

# 1) 数量级：预测超过该站该小时训练期最大值 1.5 倍，或为负
hist_max = df[df.date < "2024-01-01"].groupby(["station", "hour"], observed=True).entries.max().rename("hist_max")
te = te.join(hist_max, on=["station", "hour"])
print("预测 > 1.5 x 历史最大：", int((te[p] > 1.5 * te.hist_max).sum()), " 预测 < 0：", int((te[p] < 0).sum()))

# 2) 球场车站：比赛日按“距开赛小时数”的误差
g = te[te.game == 1]
print("\n球场车站比赛日：按距开赛小时数的平均误差（预测-实际）与实际均值")
print(g.groupby("hours_from_start").apply(lambda x: pd.Series({
    "n": len(x), "actual": x.entries.mean(), "bias": (x[p] - x.entries).mean(),
    "bias_B2": (x.mean_4w - x.entries).mean()}), include_groups=False).round(0).loc[-3:5].to_string())

# 3) 天气：实测与预报差异大的日子
w = te.drop_duplicates("date")[["date", "obs_precip", "fcst_precip"]]
big = w[(w.obs_precip - w.fcst_precip).abs() > 10].date
sub = te[te.date.isin(big)]
mae = lambda a, b: float(np.mean(np.abs(a - b)))
print(f"\n实测与预报降水相差 >10mm 的日期数：{len(big)}；这些日期上 MAE：实测天气模型 "
      f"{mae(sub.GBDT_obs_weather, sub.entries):.1f}，预报天气模型 {mae(sub[p], sub.entries):.1f}")

# 4) 按车站的误差（相对误差）
st = te.groupby("station", observed=True).apply(lambda x: pd.Series({
    "mean_entries": x.entries.mean(), "MAE": mae(x[p], x.entries), "MAE_B2": mae(x.mean_4w.fillna(x.entries.mean()), x.entries)}),
    include_groups=False)
st["rel_MAE"] = st.MAE / st.mean_entries
print("\n按车站：\n", st.round(2).sort_values("rel_MAE", ascending=False).to_string())

# 5) 预测区间覆盖率按车站
te["covered"] = (te.entries >= te.lo) & (te.entries <= te.hi)
print("\n80% 区间覆盖率按车站：", te.groupby("station", observed=True).covered.mean().round(3).to_dict())
