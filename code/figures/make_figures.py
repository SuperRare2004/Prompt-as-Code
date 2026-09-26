"""根据配套代码的真实运行结果绘制书中插图，输出到仓库根目录的 figures/chapterXX/。
需先运行 run_all.sh（或相应脚本）生成 outputs/ 与 data/ 下的中间结果。"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT.parent / "figures"
plt.rcParams.update({"font.family": ["Heiti TC", "Arial Unicode MS", "sans-serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def depth_curve():
    txt = (ROOT / "outputs" / "A_ch07_baselines_ridge.txt").read_text()
    rows = re.findall(r"depth=(\d+)\s+train=([\d.]+)\s+valid=([\d.]+)", txt)
    d, tr, va = (np.array([float(r[i]) for r in rows]) for i in range(3))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(d, tr, "o-", label="训练集 MAE", color="#4C72B0")
    ax.plot(d, va, "s-", label="验证集 MAE", color="#DD8452")
    ax.axhline(3.26, ls="--", lw=1, color="gray"); ax.text(24.5, 3.30, "岭回归 3.26", ha="right", fontsize=8, color="gray")
    ax.axhline(3.38, ls=":", lw=1, color="gray"); ax.text(24.5, 3.42, "持续性基准 3.38", ha="right", fontsize=8, color="gray")
    ax.set_xlabel("回归树最大深度"); ax.set_ylabel("MAE（mph）"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "chapter07" / "depth_curve.pdf"); plt.close(fig)


def sensor_map():
    m = pd.read_csv(ROOT / "outputs" / "A_ch09_sensor_mae.csv")
    m["gain"] = m.mae_own - m.mae_road_nb
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    sc = axes[0].scatter(m.longitude, m.latitude, c=m.mae_road_nb, cmap="viridis_r", s=18)
    fig.colorbar(sc, ax=axes[0], label="测试集 MAE（mph）")
    axes[0].set_title("加入路网下游邻居后的误差")
    v = np.nanmax(np.abs(m.gain))
    sc2 = axes[1].scatter(m.longitude, m.latitude, c=m.gain, cmap="RdBu", vmin=-v, vmax=v, s=18)
    fig.colorbar(sc2, ax=axes[1], label="误差降低量（mph）")
    axes[1].set_title("相对“仅本检测器历史”的改进")
    for ax in axes:
        ax.set_xlabel("经度"); ax.set_aspect(1 / np.cos(np.radians(34.1)))
    axes[0].set_ylabel("纬度")
    fig.tight_layout(); fig.savefig(FIG / "chapter09" / "sensor_mae_map.pdf"); plt.close(fig)


def payment_trend():
    d = pd.read_csv(ROOT / "data" / "raw" / "mta_hourly_selected.csv", parse_dates=["transit_timestamp"])
    m = d.assign(month=d.transit_timestamp.dt.to_period("M").dt.to_timestamp()).pivot_table(
        index="month", columns="payment_method", values="ridership", aggfunc="sum") / 1e6
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(m.index, m.metrocard, label="仅 MetroCard", color="#C44E52")
    ax.plot(m.index, m.omny, label="仅 OMNY", color="#55A868")
    ax.plot(m.index, m.metrocard + m.omny, label="两者合计", color="#4C72B0", lw=2)
    ax.set_ylabel("月进站量（百万人次）"); ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "chapter16" / "payment_trend.pdf"); plt.close(fig)


def game_profile():
    te = pd.read_parquet(ROOT / "data" / "mta_test_pred.parquet")
    g = te[te.game == 1].groupby("hours_from_start")[["entries", "GBDT_fcst_weather", "mean_4w"]].mean().loc[-3:5]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(g.index, g.entries, "o-", label="实际", color="black")
    ax.plot(g.index, g.GBDT_fcst_weather, "s--", label="GBDT 预测", color="#4C72B0")
    ax.plot(g.index, g.mean_4w, "^:", label="近四周同时段均值（B2）", color="#DD8452")
    ax.set_xlabel("距开赛的小时数"); ax.set_ylabel("平均进站量（人次/小时）"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "chapter16" / "game_profile.pdf"); plt.close(fig)


def reporting_system():
    c = pd.read_csv(ROOT / "data" / "raw" / "stats19_collision_5y.csv", usecols=["collision_year", "collision_severity", "collision_injury_based"])
    c["ksi"] = c.collision_severity <= 2
    t = c.pivot_table(index="collision_year", columns="collision_injury_based", values="ksi")
    share = c.groupby("collision_year").collision_injury_based.mean()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(t.index, t[1], "o-", label="基于伤情的报告系统", color="#C44E52")
    ax.plot(t.index, t[0], "s-", label="警员现场判断", color="#4C72B0")
    ax.plot(t.index, c.groupby("collision_year").ksi.mean(), "^--", label="全体", color="black")
    ax.set_ylabel("KSI 比例"); ax.set_xticks(t.index); ax.set_xlabel("年份")
    ax2 = ax.twinx(); ax2.bar(share.index, share, alpha=0.15, color="gray", width=0.5)
    ax2.set_ylabel("基于伤情系统的事故占比（柱）"); ax2.set_ylim(0, 1); ax2.spines["top"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "chapter17" / "reporting_system.pdf"); plt.close(fig)


def pr_curves():
    from sklearn.metrics import precision_recall_curve
    d = pd.read_parquet(ROOT / "data" / "B_ch08_test_scores.parquet")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for col, lab, colr in [("score", "随机森林", "#4C72B0"), ("p_logit", "逻辑回归", "#DD8452")]:
        p, r, _ = precision_recall_curve(d.ksi, d[col]); ax.plot(r, p, label=lab, color=colr)
    rule = (d.area == "rural") & (d.night == 1)
    ax.scatter([d.ksi[rule].sum() / d.ksi.sum()], [d.ksi[rule].mean()], marker="*", s=120, color="black", label="规则：农村 + 夜间", zorder=3)
    ax.axhline(d.ksi.mean(), ls=":", color="gray"); ax.text(0.5, d.ksi.mean() - 0.05, f"随机水平 {d.ksi.mean():.3f}", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("Recall（召回率）"); ax.set_ylabel("Precision（精确率）"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "chapter08" / "pr_curves.pdf"); plt.close(fig)


if __name__ == "__main__":
    for d in ("chapter07", "chapter08", "chapter09", "chapter16", "chapter17"):
        (FIG / d).mkdir(parents=True, exist_ok=True)
    for f in (depth_curve, sensor_map, payment_trend, game_profile, reporting_system, pr_curves):
        f(); print("ok", f.__name__)
