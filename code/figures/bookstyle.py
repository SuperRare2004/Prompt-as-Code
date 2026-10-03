"""第 7—11 章新增统计图的样式（concept_figures.py 中以 @newfig 注册的函数使用）：配色、字体、坐标轴与网格。

分类色按固定顺序使用（不循环），前三个颜色（蓝、橙、青绿）在散点图中也能被色觉障碍读者区分；
黄、粉、青绿与白底对比度不足 3:1，使用时必须配直接标注。文字一律用墨色，不用数据色。
顺序色只用蓝色单色阶；发散色为蓝—灰—红。
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 分类色（固定顺序）
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
# 中性色与文字
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
NEUTRAL = "#a8a7a2"        # 基准、参照线、“其他”
GRIDC = "#e6e5e1"          # 网格线：比底色深一级的灰
AXISC = "#bdbcb7"          # 坐标轴线
# 顺序色阶（蓝）与发散色（蓝—灰—红）
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("book_seq", ["#f4f8fd"] + BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("book_div", ["#184f95", "#6da7ec", "#f0efec", "#ef8a7f", "#b8302f"])


RC = {
        "font.family": ["Heiti TC", "Arial Unicode MS", "sans-serif"], "font.size": 9.5,
        "axes.titlesize": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXISC, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "text.color": INK,
        "axes.grid": False, "grid.color": GRIDC, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 5, "patch.linewidth": 0,
        "pdf.fonttype": 42, "mathtext.fontset": "cm",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
}


def apply():
    plt.rcParams.update(RC)


def grid(ax, axis="y"):
    """只画一个方向的浅色实线网格，置于数据之下。"""
    ax.grid(True, axis=axis, color=GRIDC, linewidth=0.6)
    ax.set_axisbelow(True)


def bars(ax, x, h, width=0.6, color=BLUE, horizontal=False, **kw):
    """柱子之间留白色细缝，不加描边。"""
    f = ax.barh if horizontal else ax.bar
    return f(x, h, width if not horizontal else width, color=color, edgecolor="white", linewidth=1.0, **kw)
