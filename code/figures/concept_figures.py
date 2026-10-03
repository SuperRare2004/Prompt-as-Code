"""绘制书中的概念示意图（第 7、8 章等）。能用真实数据的图尽量使用本书配套数据：
ROC 示意使用 STATS19 模型得分，手肘图、GMM、PCA 使用 METR-LA。其余为示意性图形。"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Circle, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT.parent / "figures"
plt.rcParams.update({"font.family": ["Heiti TC", "Arial Unicode MS", "sans-serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "mathtext.fontset": "cm"})
BLUE, ORANGE, GREEN, RED, GRAY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C"
FIGS = {}


def fig(chapter, name):
    def deco(f):
        FIGS[name] = (chapter, f)
        return f
    return deco


def save(f, chapter, name):
    (FIG / chapter).mkdir(parents=True, exist_ok=True)
    f.tight_layout(); f.savefig(FIG / chapter / f"{name}.pdf"); plt.close(f)


# ---------------- 第 7 章 ----------------
@fig("chapter07", "gradient_descent")
def gradient_descent():
    A = np.array([[1.0, 0.0], [0.0, 6.0]])
    L = lambda t: 0.5 * t @ A @ t
    x, y = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-2.2, 2.2, 200))
    Z = 0.5 * (A[0, 0] * x ** 2 + A[1, 1] * y ** 2)
    f, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, lr, title in [(axes[0], 0.08, r"学习率 $\alpha=0.08$：稳定但步数较多"), (axes[1], 0.31, r"学习率 $\alpha=0.31$：来回振荡")]:
        ax.contour(x, y, Z, levels=12, colors=GRAY, linewidths=0.6)
        t = np.array([-3.6, 1.8]); path = [t.copy()]
        for _ in range(25):
            t = t - lr * (A @ t); path.append(t.copy())
        path = np.array(path)
        for a, b in zip(path[:-1], path[1:]):
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
        ax.plot(0, 0, "*", color=RED, ms=12); ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\theta_1$")
    axes[0].set_ylabel(r"$\theta_2$")
    return f


@fig("chapter07", "confusion_matrix")
def confusion_matrix_fig():
    f, ax = plt.subplots(figsize=(4.8, 3.6))
    cells = [("TP\n真正例", "预测为 1，真实为 1", GREEN), ("FN\n假负例（漏报）", "预测为 0，真实为 1", RED),
             ("FP\n假正例（误报）", "预测为 1，真实为 0", ORANGE), ("TN\n真负例", "预测为 0，真实为 0", BLUE)]
    for k, (t, sub, c) in enumerate(cells):
        i, j = divmod(k, 2)
        ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=c, alpha=0.18, edgecolor="black"))
        ax.text(j + 0.5, 1.58 - i, t, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(j + 0.5, 1.2 - i, sub, ha="center", va="center", fontsize=8, color="#333333")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["预测为 1", "预测为 0"])
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["真实为 1", "真实为 0"]); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.xaxis.tick_top()
    return f


@fig("chapter07", "roc_construction")
def roc_construction():
    from sklearn.metrics import roc_curve
    d = pd.read_parquet(ROOT / "data" / "B_ch08_test_scores.parquet")
    y, s = d.ksi.to_numpy(), d.score.to_numpy()
    f, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    bins = np.linspace(0, 1, 41)
    axes[0].hist(s[y == 0], bins, density=True, alpha=0.5, color=BLUE, label="非 KSI 事故")
    axes[0].hist(s[y == 1], bins, density=True, alpha=0.5, color=RED, label="KSI 事故")
    ths = [0.35, 0.5, 0.65]; cols = [GREEN, "black", ORANGE]
    for th, c in zip(ths, cols):
        axes[0].axvline(th, color=c, ls="--", lw=1)
        axes[0].text(th, axes[0].get_ylim()[1] * 0.95, f"阈值 {th}", color=c, fontsize=8, ha="center")
    axes[0].set_xlabel("模型输出的风险得分"); axes[0].set_ylabel("密度"); axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fpr, tpr, _ = roc_curve(y, s)
    axes[1].plot(fpr, tpr, color=BLUE, label="ROC 曲线"); axes[1].plot([0, 1], [0, 1], ":", color=GRAY)
    for th, c in zip(ths, cols):
        p = s >= th
        axes[1].plot((p & (y == 0)).sum() / (y == 0).sum(), (p & (y == 1)).sum() / (y == 1).sum(), "o", color=c, ms=7)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR（召回率）"); axes[1].set_aspect("equal")
    axes[1].set_title("每个阈值对应 ROC 空间中的一个点", fontsize=9)
    return f


@fig("chapter07", "fit_regimes")
def fit_regimes():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 1, 18)); y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, x.size)
    xx = np.linspace(0, 1, 300)
    f, axes = plt.subplots(1, 3, figsize=(9.5, 2.9), sharey=True)
    for ax, deg, t in zip(axes, [1, 4, 15], ["欠拟合（1 次多项式）", "合适拟合（4 次多项式）", "过拟合（15 次多项式）"]):
        c = np.polyfit(x, y, deg)
        ax.plot(xx, np.sin(2 * np.pi * xx), ":", color=GRAY, label="真实规律")
        ax.plot(xx, np.clip(np.polyval(c, xx), -2, 2), color=BLUE, label="模型")
        ax.scatter(x, y, s=14, color="black", zorder=3); ax.set_title(t, fontsize=10); ax.set_ylim(-2, 2)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8)
    return f


# ---------------- 第 8 章 ----------------
@fig("chapter08", "ridge_lasso_geometry")
def ridge_lasso_geometry():
    f, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, kind in zip(axes, ["Ridge：$L_2$ 约束（圆）", "Lasso：$L_1$ 约束（菱形）"]):
        c = np.array([2.2, 1.3])
        for s in [0.5, 1.0, 1.5, 2.0, 2.4]:
            ax.add_patch(Ellipse(c, 2.2 * s, 1.1 * s, angle=-25, fill=False, color=GRAY, lw=0.8))
        ax.plot(*c, "k.", ms=6); ax.text(c[0] + 0.1, c[1] + 0.1, r"$\hat\beta_{OLS}$", fontsize=10)
        if "Ridge" in kind:
            ax.add_patch(Circle((0, 0), 1.0, color=BLUE, alpha=0.25)); ax.plot(0.93, 0.36, "o", color=RED)
        else:
            ax.add_patch(Polygon([[1, 0], [0, 1], [-1, 0], [0, -1]], color=GREEN, alpha=0.25)); ax.plot(1.0, 0.0, "o", color=RED)
        ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
        ax.set_xlim(-1.6, 4.2); ax.set_ylim(-1.6, 3.4); ax.set_aspect("equal"); ax.set_title(kind, fontsize=10)
        ax.set_xlabel(r"$\beta_1$"); ax.set_ylabel(r"$\beta_2$"); ax.set_xticks([]); ax.set_yticks([])
    return f


@fig("chapter08", "logistic")
def logistic():
    h = np.array([0.5, .75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
    p = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    f, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    z = np.linspace(-8, 8, 200); axes[0].plot(z, 1 / (1 + np.exp(-z)), color=BLUE)
    axes[0].axhline(0.5, ls=":", color=GRAY); axes[0].set_xlabel(r"线性部分 $z=w_0+\mathbf{w}^{T}\mathbf{x}$"); axes[0].set_ylabel(r"$\sigma(z)$")
    axes[0].set_title("Logistic 函数把任意实数压缩到 (0, 1)", fontsize=10)
    xx = np.linspace(0, 6, 200); axes[1].plot(xx, 1 / (1 + np.exp(-(-4.0777 + 1.5046 * xx))), color=BLUE, label="拟合的通过概率")
    axes[1].scatter(h, p, color="black", s=16, zorder=3, label="学生（0=未通过，1=通过）")
    axes[1].axhline(0.5, ls=":", color=GRAY); axes[1].set_xlabel("复习时间（小时）"); axes[1].set_ylabel("通过概率")
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    return f


@fig("chapter08", "softmax_regions")
def softmax_regions():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    C = np.array([[0, 0], [3, 0.5], [1.4, 2.8]])
    X = np.vstack([rng.normal(c, 0.75, (60, 2)) for c in C]); y = np.repeat([0, 1, 2], 60)
    m = LogisticRegression(max_iter=1000).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-2.5, 5.5, 300), np.linspace(-2.5, 5, 300))
    P = m.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    f, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cols = [BLUE, ORANGE, GREEN]
    from matplotlib.colors import ListedColormap
    axes[0].contourf(xx, yy, P.argmax(1).reshape(xx.shape), levels=[-.5, .5, 1.5, 2.5], cmap=ListedColormap(cols), alpha=0.2)
    for k in range(3): axes[0].scatter(*X[y == k].T, s=8, color=cols[k], label=f"类别 {k+1}")
    axes[0].contour(xx, yy, P.max(1).reshape(xx.shape), levels=[0.5], colors=GRAY, linestyles=":")
    axes[0].set_title("概率最大的类别形成决策区域（虚线：最大概率 = 0.5）", fontsize=9); axes[0].legend(frameon=False, fontsize=8)
    pt = np.array([[1.6, 1.2]]); axes[0].plot(*pt[0], "k*", ms=12)
    axes[1].bar(["类别 1", "类别 2", "类别 3"], m.predict_proba(pt)[0], color=cols)
    axes[1].set_ylim(0, 1); axes[1].set_title("星号样本的 Softmax 输出：一组类别概率", fontsize=9)
    return f


def _svm_data(sep):
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal([-sep, -sep], 0.8, (30, 2)), rng.normal([sep, sep], 0.8, (30, 2))]); y = np.repeat([-1, 1], 30)
    return X, y


def _svm_plot(ax, X, y, C, title):
    from sklearn.svm import SVC
    m = SVC(kernel="linear", C=C).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 200), np.linspace(-4.5, 4.5, 200))
    Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["gray", "black", "gray"], linestyles=["--", "-", "--"])
    ax.scatter(*X[y == -1].T, color=BLUE, s=14); ax.scatter(*X[y == 1].T, color=ORANGE, s=14)
    sv = m.support_vectors_; ax.scatter(*sv.T, s=90, facecolors="none", edgecolors="black", lw=1)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    return m


@fig("chapter08", "svm_margin")
def svm_margin():
    f, ax = plt.subplots(figsize=(4.4, 4))
    X, y = _svm_data(1.8); _svm_plot(ax, X, y, 1e6, "硬间隔：实线为分类超平面，虚线为 $\\pm1$ 边界，圈出者为支持向量")
    ax.title.set_fontsize(8)
    return f


@fig("chapter08", "svm_soft")
def svm_soft():
    f, axes = plt.subplots(1, 2, figsize=(8.4, 4))
    X, y = _svm_data(1.0)
    _svm_plot(axes[0], X, y, 100, "$C=100$：间隔较窄，尽量少犯错")
    _svm_plot(axes[1], X, y, 0.05, "$C=0.05$：间隔较宽，容忍更多违例")
    return f


@fig("chapter08", "tree_partition")
def tree_partition():
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (220, 2)); y = ((X[:, 0] > 0.55) & (X[:, 1] > 0.35) | (X[:, 1] > 0.8)).astype(int)
    y = np.where(rng.random(220) < 0.08, 1 - y, y)
    f, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, d, t in zip(axes, [3, None], ["深度为 3：少数几次轴向切分", "不限深度：切分出噪声点"]):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y)
        xx, yy = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
        ax.contourf(xx, yy, m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), levels=[-.5, .5, 1.5], colors=[BLUE, ORANGE], alpha=0.2)
        ax.scatter(*X[y == 0].T, s=8, color=BLUE); ax.scatter(*X[y == 1].T, s=8, color=ORANGE)
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    return f


@fig("chapter08", "tree_pruning")
def tree_pruning():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(Xtr, ytr)
    al = path.ccp_alphas[:-1]
    tr, te, leaves = [], [], []
    for a in al:
        m = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(Xtr, ytr)
        tr.append(m.score(Xtr, ytr)); te.append(m.score(Xte, yte)); leaves.append(m.get_n_leaves())
    f, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(leaves, tr, "o-", color=BLUE, ms=3, label="训练集准确率"); ax.plot(leaves, te, "s-", color=ORANGE, ms=3, label="测试集准确率")
    ax.set_xlabel("剪枝后叶节点数（由代价复杂度参数 $\\alpha$ 控制）"); ax.set_ylabel("准确率"); ax.legend(frameon=False)
    ax.set_title("乳腺肿瘤数据：后剪枝路径", fontsize=9)
    return f


@fig("chapter08", "kmeans_steps")
def kmeans_steps():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(c, 0.55, (40, 2)) for c in [(0, 0), (3, 1), (1, 3)]])
    mu = np.array([[0.2, 2.5], [0.6, 2.9], [2.8, -0.3]])
    cols = np.array([BLUE, ORANGE, GREEN])
    f, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharex=True, sharey=True)
    titles = ["初始化质心", "第 1 轮：分配", "第 1 轮：更新质心", "收敛后"]
    for k, ax in enumerate(axes):
        lab = np.argmin(((X[:, None] - mu[None]) ** 2).sum(-1), 1)
        if k == 0:
            ax.scatter(*X.T, s=8, color=GRAY)
        else:
            ax.scatter(*X.T, s=8, color=cols[lab])
        if k == 2:
            new = np.array([X[lab == j].mean(0) for j in range(3)])
            for a, b in zip(mu, new): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.2))
            mu = new
        if k == 3:
            for _ in range(20):
                lab = np.argmin(((X[:, None] - mu[None]) ** 2).sum(-1), 1); mu = np.array([X[lab == j].mean(0) for j in range(3)])
            ax.scatter(*X.T, s=8, color=cols[lab])
        ax.scatter(*mu.T, marker="X", s=110, color=cols, edgecolors="black", zorder=3)
        ax.set_title(titles[k], fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    return f


def _speed_profiles():
    sys.path.insert(0, str(ROOT / "case_A_freeway"))
    from common import load_speed
    w = load_speed()
    w2 = w[w.index.dayofweek < 5].between_time("06:00", "19:59")
    return w, w2.groupby(w2.index.floor("15min").time).mean().T.dropna()


@fig("chapter08", "elbow")
def elbow():
    from sklearn.cluster import KMeans
    _, P = _speed_profiles()
    ks = range(1, 9); sse = [KMeans(k, n_init=10, random_state=0).fit(P).inertia_ / 1e4 for k in ks]
    f, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(ks, sse, "o-", color=BLUE); ax.set_xlabel("簇数 $K$"); ax.set_ylabel("簇内平方误差 Inertia（$\\times 10^4$）")
    ax.set_title("METR-LA 检测器日变化曲线的手肘图", fontsize=9)
    return f


@fig("chapter08", "gmm_responsibility")
def gmm_responsibility():
    from scipy.stats import norm
    x = np.linspace(-1, 9, 400)
    p1, p2 = 0.6 * norm.pdf(x, 2.5, 1.0), 0.4 * norm.pdf(x, 5.5, 1.2)
    f, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(x, p1, color=BLUE, label=r"$\pi_1\mathcal{N}(x|\mu_1,\sigma_1^2)$"); axes[0].plot(x, p2, color=ORANGE, label=r"$\pi_2\mathcal{N}(x|\mu_2,\sigma_2^2)$")
    axes[0].plot(x, p1 + p2, "k--", lw=1, label="混合密度 $p(x)$"); axes[0].axvline(4.2, color=GRAY, ls=":")
    axes[0].legend(frameon=False, fontsize=8); axes[0].set_xlabel("$x$"); axes[0].set_title("先验 × 似然 = 各分量的联合密度", fontsize=9)
    g1 = p1 / (p1 + p2)
    axes[1].plot(x, g1, color=BLUE, label=r"$\gamma_1(x)=P(z=1|x)$"); axes[1].plot(x, 1 - g1, color=ORANGE, label=r"$\gamma_2(x)=P(z=2|x)$")
    axes[1].axvline(4.2, color=GRAY, ls=":"); axes[1].legend(frameon=False, fontsize=8); axes[1].set_xlabel("$x$")
    axes[1].set_title("贝叶斯定理得到的后验“责任”", fontsize=9)
    return f


@fig("chapter08", "gmm_speed")
def gmm_speed():
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm
    w, _ = _speed_profiles()
    s = w.stack().dropna().to_numpy()
    s = np.random.default_rng(0).choice(s, 200_000, replace=False)
    g = GaussianMixture(4, random_state=0).fit(s[:, None])
    x = np.linspace(0, 80, 400)
    f, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(s, bins=80, density=True, color=GRAY, alpha=0.35, label="实测速度")
    tot = np.zeros_like(x)
    for k in np.argsort(g.means_.ravel()):
        comp = g.weights_[k] * norm.pdf(x, g.means_[k, 0], np.sqrt(g.covariances_[k].ravel()[0])); tot += comp
        ax.plot(x, comp, "--", color=RED, lw=1)
    ax.plot(x, tot, color=BLUE, lw=2, label="4 分量混合")
    ax.set_xlabel("5 分钟平均速度（mph）"); ax.set_ylabel("密度"); ax.legend(frameon=False)
    ax.set_title("METR-LA 全部检测器速度分布的高斯混合拟合", fontsize=9)
    return f


@fig("chapter08", "pca")
def pca_fig():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    _, P = _speed_profiles()
    Z = PCA(2).fit_transform(P - P.mean()); lab = KMeans(3, n_init=10, random_state=0).fit_predict(P)
    rng = np.random.default_rng(0); X = rng.multivariate_normal([0, 0], [[3, 2.2], [2.2, 2.2]], 200)
    pc = PCA(2).fit(X)
    f, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].scatter(*X.T, s=6, color=GRAY)
    for v, l, c in zip(pc.components_, np.sqrt(pc.explained_variance_), [RED, BLUE]):
        axes[0].annotate("", xy=v * 2 * l, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    axes[0].text(*(pc.components_[0] * 2.2 * np.sqrt(pc.explained_variance_[0])), "PC1", color=RED)
    axes[0].text(*(pc.components_[1] * 2.6 * np.sqrt(pc.explained_variance_[1])), "PC2", color=BLUE)
    axes[0].set_aspect("equal"); axes[0].set_title("寻找变化最大的方向并换坐标系", fontsize=9); axes[0].set_xticks([]); axes[0].set_yticks([])
    for k, c in zip(range(3), [BLUE, ORANGE, GREEN]):
        axes[1].scatter(*Z[lab == k].T, s=10, color=c)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].set_title("METR-LA 检测器日变化曲线投影（颜色为 K-均值簇）", fontsize=9)
    return f


# ---------------- 第 9 章 ----------------
@fig("chapter09", "xor")
def xor():
    f, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for ax, (name, y, line) in zip(axes, [("AND", [0, 0, 0, 1], (1.5, 1.5)), ("OR", [0, 1, 1, 1], (0.5, 0.5)), ("XOR", [0, 1, 1, 0], None)]):
        for (a, b), t in zip(pts, y):
            ax.scatter(a, b, s=120, marker="o" if t else "x", color=ORANGE if t else BLUE, lw=2)
        if line:
            xx = np.linspace(-0.3, 1.3, 10); ax.plot(xx, line[0] - xx, color="black", lw=1)
        else:
            ax.text(0.5, 0.5, "?", fontsize=22, ha="center", va="center", color=RED)
        ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.3); ax.set_aspect("equal"); ax.set_title(name, fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    return f


@fig("chapter09", "gradient_vanishing")
def gradient_vanishing():
    k = np.arange(0, 41)
    f, ax = plt.subplots(figsize=(5.6, 3.3))
    for lam, c in [(0.8, BLUE), (0.95, GREEN), (1.0, GRAY), (1.1, RED)]:
        ax.semilogy(k, lam ** k, color=c, label=f"$\\lambda={lam}$")
    ax.set_xlabel("回传的时间步数 $k$"); ax.set_ylabel("梯度相对大小（对数坐标）")
    ax.legend(frameon=False, fontsize=8); ax.set_title("连乘 $k$ 次后的梯度：$\\lambda<1$ 时消失，$\\lambda>1$ 时爆炸", fontsize=9)
    return f


@fig("chapter09", "image_pixels")
def image_pixels():
    from sklearn.datasets import load_digits
    img = load_digits().images[0]
    f, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    axes[0].imshow(img, cmap="gray_r"); axes[0].set_title("8×8 灰度图像（手写数字“0”）", fontsize=9)
    axes[1].imshow(img, cmap="gray_r", alpha=0.25)
    for i in range(8):
        for j in range(8):
            axes[1].text(j, i, int(img[i, j]), ha="center", va="center", fontsize=8)
    axes[1].set_title("计算机看到的：像素值矩阵（0—16）", fontsize=9)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    return f


@fig("chapter09", "iou")
def iou():
    f, ax = plt.subplots(figsize=(4.6, 3.4))
    gt, pr = (1, 1, 4, 3), (2.2, 1.8, 4, 3)
    ax.add_patch(Rectangle(gt[:2], gt[2], gt[3], fill=False, ec=GREEN, lw=2, label="真实框 $B_{gt}$"))
    ax.add_patch(Rectangle(pr[:2], pr[2], pr[3], fill=False, ec=RED, lw=2, ls="--", label="预测框 $B_p$"))
    ix0, iy0 = max(gt[0], pr[0]), max(gt[1], pr[1]); ix1, iy1 = min(gt[0] + gt[2], pr[0] + pr[2]), min(gt[1] + gt[3], pr[1] + pr[3])
    ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, color=BLUE, alpha=0.3, label="交集"))
    inter = (ix1 - ix0) * (iy1 - iy0); union = gt[2] * gt[3] + pr[2] * pr[3] - inter
    ax.text(3.6, 0.3, f"IoU = 交集 / 并集 = {inter:.2f} / {union:.2f} = {inter / union:.2f}", ha="center", fontsize=9)
    ax.set_xlim(0, 7.2); ax.set_ylim(0, 5.5); ax.set_aspect("equal"); ax.axis("off"); ax.legend(frameon=False, fontsize=8, loc="upper right")
    return f


# ---------------- 第 10 章 ----------------
def _grid43(gamma=1.0, r=-0.04):
    """经典 4x3 网格世界：(1,1) 为墙，(3,0) 为 +1，(3,1) 为 -1；行动成功概率 0.8，左右偏离各 0.1。坐标 (列, 行)，行 0 在上。"""
    W, H, wall, term = 4, 3, (1, 1), {(3, 0): 1.0, (3, 1): -1.0}
    moves = {"↑": (0, -1), "↓": (0, 1), "←": (-1, 0), "→": (1, 0)}
    perp = {"↑": "←→", "↓": "←→", "←": "↑↓", "→": "↑↓"}
    S = [(c, rr) for c in range(W) for rr in range(H) if (c, rr) != wall]
    def step(s, a):
        n = (s[0] + moves[a][0], s[1] + moves[a][1])
        return s if (n == wall or not (0 <= n[0] < W and 0 <= n[1] < H)) else n
    V = {s: 0.0 for s in S}
    for _ in range(500):
        V2 = {}
        for s in S:
            if s in term: V2[s] = term[s]; continue
            V2[s] = r + gamma * max(0.8 * V[step(s, a)] + sum(0.1 * V[step(s, b)] for b in perp[a]) for a in moves)
        V = V2
    pol = {s: max(moves, key=lambda a: 0.8 * V[step(s, a)] + sum(0.1 * V[step(s, b)] for b in perp[a])) for s in S if s not in term}
    return V, pol, wall, term


def _draw_grid(ax, text_of, color_of=None):
    V, pol, wall, term = _grid43()
    for c in range(4):
        for rr in range(3):
            fc = "#555555" if (c, rr) == wall else (color_of((c, rr)) if color_of else "white")
            ax.add_patch(Rectangle((c, 2 - rr), 1, 1, facecolor=fc, edgecolor="black"))
            if (c, rr) != wall:
                ax.text(c + 0.5, 2 - rr + 0.5, text_of((c, rr)), ha="center", va="center", fontsize=12)
    ax.set_xlim(0, 4); ax.set_ylim(0, 3); ax.set_aspect("equal"); ax.axis("off")


@fig("chapter10", "grid_policy")
def grid_policy():
    V, pol, wall, term = _grid43()
    f, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), gridspec_kw={"width_ratios": [1.4, 1]})
    _draw_grid(axes[0], lambda s: ("+1" if term.get(s, 0) > 0 else "−1") if s in term else pol[s],
               lambda s: ("#d9f0d3" if term[s] > 0 else "#f4cccc") if s in term else "white")
    axes[0].set_title("确定性策略 $\\pi(s)=a$：每个状态一个行动", fontsize=9)
    axes[1].bar(["↑", "↓", "←", "→"], [0.7, 0.1, 0.1, 0.1], color=[BLUE, GRAY, GRAY, GRAY])
    axes[1].set_ylim(0, 1); axes[1].set_title("随机性策略 $\\pi(a|s)$：某一状态下的行动分布", fontsize=9)
    return f


@fig("chapter10", "grid_values")
def grid_values():
    V, pol, wall, term = _grid43()
    import matplotlib.cm as cm
    norm = matplotlib.colors.Normalize(-1, 1)
    f, ax = plt.subplots(figsize=(4.6, 3.2))
    _draw_grid(ax, lambda s: f"{V[s]:+.3f}" if s not in term else ("+1" if term[s] > 0 else "−1"),
               lambda s: cm.RdYlGn(norm(V[s])))
    ax.set_title("最优状态价值 $V^*(s)$（每步奖励 −0.04，$\\gamma=1$）", fontsize=9)
    return f


# ---------------- 前言、第 1 章（由运行日志中的真实结果绘制） ----------------
OUT = ROOT / "outputs"


def _parse_log(name, pattern):
    import re
    return re.findall(pattern, (OUT / name).read_text(encoding="utf-8"))


@fig("chapter00", "preface_leakage")
def preface_leakage():
    rows = dict((k.strip(), (float(r), float(m))) for k, r, m in _parse_log(
        "A_preface_leakage.txt", r"^(.+?)\s+R2=([\d.]+)\s+MAE=([\d.]+) mph$".replace("^", "(?m)^")))
    items = [("AI 初始做法\n（0 当作车速 + 随机划分 + 居中窗口）", "AI initial: zeros kept + random + centered", RED),
             ("时间划分 + 居中窗口", "time split + centered window", RED),
             ("随机划分 + 过去窗口", "random split + past window", ORANGE),
             ("修正后的做法\n（时间划分 + 过去窗口）", "time split + past window", BLUE),
             ("持续性基准\n（15 分钟后 = 现在）", "persistence baseline (time split)", GRAY)]
    f, ax = plt.subplots(figsize=(7.6, 3.4))
    for i, (lab, key, c) in enumerate(items):
        r2, mae = rows[key]
        ax.barh(i, mae, color=c, height=0.62)
        ax.text(mae + 0.05, i, f"MAE {mae:.2f}  ($R^2$ = {r2:.3f})", va="center", fontsize=9)
    ax.set_yticks(range(len(items)), [x[0] for x in items], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, 4.6); ax.set_xlabel("测试集 MAE（mph），越小越好")
    ax.axvline(rows["persistence baseline (time split)"][1], color=GRAY, ls="--", lw=1)
    ax.set_title("红色：使用了预测时刻不可得的信息；蓝色：修正后；灰色：朴素基准", fontsize=9)
    return f


@fig("chapter01", "default_vs_audit")
def default_vs_audit():
    txt = (OUT / "B_ch01_default_vs_audit.txt").read_text(encoding="utf-8")
    import re
    d_auc, d_pr = map(float, re.search(r"\[默认做法\].*?ROC-AUC=([\d.]+) PR-AUC=([\d.]+)", txt).groups())
    a_auc, a_pr, rnd = map(float, re.search(r"\[审查后\].*?ROC-AUC=([\d.]+) PR-AUC=([\d.]+).*?≈([\d.]+)", txt).groups())
    pol = eval(re.search(r"按警员是否到场： (\{.*\})", txt).group(1))
    f, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]; x = np.arange(2); w = 0.36
    for k, (vals, lab, c) in enumerate([((d_auc, d_pr), "一键建模（全部字段 + 随机划分）", RED),
                                         ((a_auc, a_pr), "审查后（事前字段 + 时间划分）", BLUE)]):
        b = ax.bar(x + (k - 0.5) * w, vals, w, color=c, label=lab)
        ax.bar_label(b, fmt="%.3f", fontsize=8.5, padding=2)
    ax.axhline(rnd, color=GRAY, ls="--", lw=1)
    ax.text(0.5, rnd - 0.02, f"随机猜测\nPR-AUC\n≈{rnd:.3f}", fontsize=7.5, color="#555555", ha="center", va="top",
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_xticks(x, ["ROC-AUC", "PR-AUC"]); ax.set_ylim(0, 1.42); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(fontsize=8, loc="upper center", frameon=False, ncol=1)
    ax.set_title("(a) 同一份数据上的两种做法", fontsize=9.5)
    ax = axes[1]
    labs = {1: "警员到场", 2: "警员未到场", 3: "当事人\n自行报告"}
    ks = [k for k in (1, 2, 3) if k in pol]
    b = ax.bar(range(len(ks)), [pol[k] * 100 for k in ks], color=[ORANGE, GRAY, GRAY], width=0.6)
    ax.bar_label(b, fmt="%.1f%%", fontsize=8.5, padding=2)
    ax.set_xticks(range(len(ks)), [labs[k] for k in ks], fontsize=8.5)
    ax.set_ylabel("死亡或重伤（KSI）事故占比（%）"); ax.set_ylim(0, 36)
    ax.set_title("(b) 事后字段与标签的关联", fontsize=9.5)
    return f



# ---------------- 第 2 章 ----------------
@fig("chapter02", "prompt_levels")
def prompt_levels():
    import re
    txt = (OUT / "A_preface_leakage.txt").read_text(encoding="utf-8")
    get = lambda key: tuple(map(float, re.search(re.escape(key) + r"\s+R2=([\d.]+)\s+MAE=([\d.]+)", txt).groups()))
    rows = [("只说目标", get("AI initial: zeros kept + random + centered"), RED),
            ("+ 数据说明\n（0 表示缺失）", get("random split + centered window"), RED),
            ("+ 约束与验收标准\n（时间划分、只用过去数据）", get("time split + past window"), BLUE)]
    base = get("persistence baseline (time split)")
    f, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for ax, j, lab in [(axes[0], 0, "$R^2$（越大越好）"), (axes[1], 1, "MAE，mph（越小越好）")]:
        vals = [r[1][j] for r in rows]
        b = ax.bar(range(3), vals, color=[r[2] for r in rows], width=0.6)
        ax.bar_label(b, fmt="%.3f" if j == 0 else "%.2f", fontsize=8.5, padding=2)
        ax.axhline(base[j], color=GRAY, ls="--", lw=1, label=f"持续性基准（{base[j]:.3f}）" if j == 0 else f"持续性基准（{base[j]:.2f}）")
        ax.legend(fontsize=8, frameon=False, loc="upper right" if j == 0 else "upper left")
        ax.set_xticks(range(3), [r[0] for r in rows], fontsize=8)
        ax.set_title(lab, fontsize=9.5)
        ax.set_ylim(0, (1.25 if j == 0 else 4.5))
    return f


# ---------------- 第 3 章 ----------------
def _metr():
    sys.path.insert(0, str(ROOT / "case_A_freeway"))
    from common import load_speed
    return load_speed(zero_as_missing=True)


@fig("chapter03", "weekly_profile")
def weekly_profile():
    df = _metr()
    net = df.mean(axis=1)                       # 每个时刻全网平均车速（跳过缺失）
    d = net.to_frame("v")
    d["tod"] = d.index.hour + d.index.minute / 60
    d["date"] = d.index.normalize()
    holidays = [pd.Timestamp("2012-05-28")]     # 阵亡将士纪念日
    d["kind"] = np.where((d.index.dayofweek >= 5) | d["date"].isin(holidays), "周末与节假日", "工作日")
    f, ax = plt.subplots(figsize=(7.4, 3.2))
    for kind, c in [("工作日", BLUE), ("周末与节假日", ORANGE)]:
        g = d[d.kind == kind].groupby("tod").v
        q = g.quantile([0.1, 0.5, 0.9]).unstack()
        ax.fill_between(q.index, q[0.1], q[0.9], color=c, alpha=0.18, lw=0)
        ax.plot(q.index, q[0.5], color=c, lw=1.8, label=f"{kind}（中位数与 10%–90% 区间，{d[d.kind == kind].date.nunique()} 天）")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("一天中的时刻"); ax.set_ylabel("全网平均车速（mph）")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    return f


# ---------------- 第 4 章 ----------------
def _metr_raw():
    sys.path.insert(0, str(ROOT / "case_A_freeway"))
    from common import load_speed
    return load_speed(zero_as_missing=False)


@fig("chapter04", "sensor_week")
def sensor_week():
    raw = _metr_raw()
    allz = (raw == 0).all(axis=1)
    sid, a, b = "717472", "2012-03-26", "2012-04-01 23:55"
    x = raw.loc[a:b, sid]; az = allz.loc[a:b]
    f, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.plot(x.index, x.where(x > 0), color=BLUE, lw=0.9, label="有效读数")
    own = x[(x == 0) & ~az]
    ax.scatter(own.index, own.values, s=8, color=RED, zorder=3, label=f"该检测器单独为 0（{len(own)} 个时刻）")
    run = az.astype(int).diff().fillna(az.iloc[0]).ne(0).cumsum()
    first = True
    for _, g in az[az].groupby(run[az]):
        ax.axvspan(g.index[0], g.index[-1] + pd.Timedelta("5min"), color=GRAY, alpha=0.3, lw=0,
                   label=f"全部 207 个检测器同时为 0（{int(az.sum())} 个时刻）" if first else None)
        first = False
    ax.set_ylim(-3, 75); ax.set_ylabel("车速（mph）")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.1))
    ax.set_title(f"检测器 {sid}，2012 年 3 月 26 日至 4 月 1 日", fontsize=9)
    print("sensor_week own zeros", len(own), "network-zero ts", int(az.sum()))
    return f


@fig("chapter04", "aggregation")
def aggregation():
    w = _metr()
    sid, day = "717472", "2012-03-28"
    x = w.loc[day, sid]
    h = x.resample("1h").mean()
    f, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.plot(x.index, x, color=BLUE, lw=1.0, label="5 分钟")
    ax.step(h.index, h, where="post", color=ORANGE, lw=1.6, label="1 小时平均")
    ax.axhline(x.mean(), color=GRAY, ls="--", lw=1.2, label=f"全天平均（{x.mean():.1f} mph）")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("车速（mph）"); ax.set_ylim(0, 75)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title(f"检测器 {sid}，2012 年 3 月 28 日（星期三）", fontsize=9)
    print("aggregation min5", round(x.min(), 1), "minhour", round(h.min(), 1), "daymean", round(x.mean(), 1))
    return f


@fig("chapter04", "label_thresholds")
def label_thresholds():
    w = _metr()
    v = w.stack()
    ref = w.quantile(0.85)                       # 各检测器的“畅通车速”：85% 分位数
    rel = (w.div(ref, axis=1) < 0.6).where(w.notna()).stack()
    items = [("车速 < 25 mph", (v < 25).mean()), ("车速 < 35 mph", (v < 35).mean()),
             ("车速 < 45 mph", (v < 45).mean()), ("低于该检测器\n畅通车速的 60%", rel.mean())]
    f, ax = plt.subplots(figsize=(6.4, 2.9))
    b = ax.bar(range(4), [100 * x[1] for x in items], color=[BLUE, BLUE, BLUE, ORANGE], width=0.58)
    ax.bar_label(b, fmt="%.1f%%", fontsize=8.5, padding=2)
    ax.set_xticks(range(4), [x[0] for x in items], fontsize=8.5)
    ax.set_ylabel("被标记为“拥堵”的记录占比（%）"); ax.set_ylim(0, max(100 * x[1] for x in items) * 1.25)
    print("label_thresholds", [(k.replace(chr(10), ""), round(100 * x, 1)) for k, x in items])
    return f


@fig("chapter04", "speed_heatmap")
def speed_heatmap():
    w = _metr()
    net = w.mean(axis=1)
    t = net.groupby([net.index.dayofweek, net.index.hour]).mean().unstack()
    f, ax = plt.subplots(figsize=(7.6, 2.6))
    im = ax.imshow(t.values, aspect="auto", cmap="RdYlGn", vmin=t.values.min(), vmax=t.values.max())
    ax.set_yticks(range(7), ["周一", "周二", "周三", "周四", "周五", "周六", "周日"], fontsize=8.5)
    ax.set_xticks(range(0, 24, 2), [f"{h}" for h in range(0, 24, 2)], fontsize=8.5)
    ax.set_xlabel("小时")
    c = f.colorbar(im, ax=ax, pad=0.015); c.set_label("全网平均车速（mph）", fontsize=8.5)
    return f


# ---------------- 第 5 章 ----------------
@fig("chapter05", "missing_matrix")
def missing_matrix():
    raw = _metr_raw()
    z = (raw == 0)
    daily = z.groupby(raw.index.normalize()).mean().T          # 检测器 x 日期
    order = daily.mean(axis=1).sort_values().index
    daily = daily.loc[order]
    f, ax = plt.subplots(figsize=(8.2, 3.4))
    im = ax.imshow(daily.values * 100, aspect="auto", cmap="Greys", vmin=0, vmax=100, interpolation="nearest")
    ticks = [i for i, d in enumerate(daily.columns) if d.day == 1 or d.day == 15]
    ax.set_xticks(ticks, [daily.columns[i].strftime("%m-%d") for i in ticks], fontsize=8)
    ax.set_yticks([]); ax.set_ylabel("207 个检测器\n（按总体缺失率排序）", fontsize=8.5)
    c = f.colorbar(im, ax=ax, pad=0.015); c.set_label("当日读数为 0 的比例（%）", fontsize=8.5)
    full = (daily.mean(axis=0) > 0.1).sum()
    print("missing_matrix days with >10% network zeros", int(full), "sensors >5% overall", int((z.mean() > 0.05).sum()))
    return f


@fig("chapter05", "gap_lengths")
def gap_lengths():
    raw = _metr_raw()
    allz = (raw == 0).all(axis=1)
    lens_own, lens_net = [], []
    for sid in raw.columns:
        zz = (raw[sid] == 0).values
        own = zz & ~allz.values
        for arr, out in [(own, lens_own)]:
            r = np.diff(np.concatenate([[0], arr.astype(int), [0]]))
            st, en = np.where(r == 1)[0], np.where(r == -1)[0]
            out.extend(en - st)
    r = np.diff(np.concatenate([[0], allz.values.astype(int), [0]]))
    lens_net = np.where(r == -1)[0] - np.where(r == 1)[0]
    lens_own = np.array(lens_own)
    bins = [1, 2, 4, 7, 13, 37, 289, 10**6]
    labs = ["1 步\n(5 分钟)", "2–3 步", "4–6 步", "7–12 步\n(≤1 小时)", "13–36 步\n(≤3 小时)", "37–288 步\n(≤1 天)", "> 1 天"]
    h_own = np.histogram(lens_own, bins=bins)[0]
    h_net = np.histogram(lens_net, bins=bins)[0]
    f, ax = plt.subplots(figsize=(7.6, 3.0))
    x = np.arange(len(labs)); w = 0.4
    b1 = ax.bar(x - w / 2, h_own, w, color=BLUE, label=f"单个检测器的缺失段（共 {len(lens_own):,} 段）")
    b2 = ax.bar(x + w / 2, h_net, w, color=GRAY, label=f"全网同时缺失的时段（共 {len(lens_net)} 段）")
    ax.bar_label(b1, fontsize=7.5, padding=1); ax.bar_label(b2, fontsize=7.5, padding=1)
    ax.set_yscale("log"); ax.set_ylim(0.8, h_own.max() * 4)
    ax.set_xticks(x, labs, fontsize=8); ax.set_ylabel("缺失段数（对数刻度）")
    ax.legend(fontsize=8, frameon=False)
    print("gap_lengths own", len(lens_own), "share<=3 steps", round((lens_own <= 3).mean() * 100, 1),
          "share>12", round((lens_own > 12).mean() * 100, 1), "net segments", len(lens_net), list(h_own), list(h_net))
    return f


@fig("chapter05", "imputation")
def imputation():
    w = _metr()
    sid = "717472"
    x = w[sid]
    day = "2012-03-28"
    a, b = pd.Timestamp(f"{day} 08:00"), pd.Timestamp(f"{day} 08:55")      # 人为挖去一小时（早高峰拥堵段）
    truth = x.loc[a:b]
    masked = x.copy(); masked.loc[a:b] = np.nan
    lin = masked.interpolate(limit_area="inside").loc[a:b]
    ffill = masked.ffill().loc[a:b]
    hist = x[(x.index.dayofweek == 2) & (x.index < "2012-03-28")]
    hist = hist.groupby(hist.index.time).mean()
    hm = pd.Series([hist.get(t.time(), np.nan) for t in truth.index], index=truth.index)
    show = x.loc[f"{day} 06:00":f"{day} 11:00"]
    f, ax = plt.subplots(figsize=(7.6, 3.1))
    ax.axvspan(a, b + pd.Timedelta("5min"), color=GRAY, alpha=0.15, lw=0)
    ax.plot(show.index, show, color="black", lw=1.2, label="真实读数（灰色区间被人为挖去）")
    res = {}
    for s_, lab, c, ls in [(lin, "线性插值", BLUE, "-"), (ffill, "前向填充", ORANGE, "--"), (hm, "此前各周三同一时刻的均值", GREEN, "-.")]:
        mae = (s_ - truth).abs().mean(); res[lab] = round(mae, 1)
        ax.plot(s_.index, s_, color=c, lw=1.8, ls=ls, label=f"{lab}（MAE {mae:.1f} mph）")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("车速（mph）"); ax.set_ylim(0, 80)
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    ax.set_title(f"检测器 {sid}，2012 年 3 月 28 日早高峰", fontsize=9)
    print("imputation MAE", res)
    return f


@fig("chapter05", "hourly_box")
def hourly_box():
    w = _metr()
    sid = "717472"
    x = w[sid].dropna()
    x = x[x.index.dayofweek < 5]
    q1, q3 = x.quantile([0.25, 0.75])
    lo = q1 - 1.5 * (q3 - q1)                      # 箱线图规则，按全部数据统一计算
    data = [x[x.index.hour == h].values for h in range(24)]
    f, ax = plt.subplots(figsize=(7.8, 3.1))
    ax.boxplot(data, positions=range(24), widths=0.6, showfliers=True, flierprops=dict(markersize=1.5, alpha=0.4),
               medianprops=dict(color=ORANGE), patch_artist=True, boxprops=dict(facecolor="#dfe6f1", edgecolor=BLUE))
    ax.axhline(lo, color=RED, ls="--", lw=1.2, label=f"全局阈值：$Q_1 - 1.5\\,\\mathrm{{IQR}}$（{lo:.1f} mph）")
    flagged = x[x < lo]
    peak = flagged.index.hour.isin(list(range(6, 10)) + list(range(15, 20))).mean()
    ax.set_xticks(range(0, 24, 2), [str(h) for h in range(0, 24, 2)])
    ax.set_xlabel("小时（工作日）"); ax.set_ylabel("车速（mph）")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title(f"检测器 {sid}：被全局阈值判为“异常”的 {len(flagged)} 条记录中，{peak * 100:.0f}% 位于早晚高峰", fontsize=9)
    print("hourly_box lo", round(lo, 1), "flagged", len(flagged), "peak share", round(peak * 100, 1))
    return f


# ---------------- 第 6 章 ----------------
@fig("chapter06", "acf")
def acf():
    w = _metr()
    lags = np.arange(0, 2 * 288 + 1)
    vals = []
    for sid in w.columns:
        x = w[sid]
        x = x - x.mean()
        v0 = (x * x).mean()
        vals.append([(x * x.shift(k)).mean() / v0 for k in lags])
    r = np.nanmedian(np.array(vals), axis=0)
    f, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.plot(lags * 5 / 60, r, color=BLUE, lw=1.4)
    ax.axhline(0, color=GRAY, lw=0.8)
    for k, lab in [(1, "5 分钟"), (12, "1 小时"), (288, "1 天")]:
        ax.plot(k * 5 / 60, r[k], "o", color=ORANGE, ms=4)
        ax.annotate(f"{lab}：{r[k]:.2f}", (k * 5 / 60, r[k]), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("滞后（小时）"); ax.set_ylabel("自相关系数（207 个检测器的中位数）")
    ax.set_xticks(range(0, 49, 6)); ax.set_xlim(0, 48)
    print("acf", {k: round(float(r[k]), 3) for k in (1, 3, 12, 144, 288)})
    return f


@fig("chapter06", "cyclic_encoding")
def cyclic_encoding():
    h = np.arange(24)
    f, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.scatter(h, np.zeros(24), c=h, cmap="twilight", s=30, zorder=3)
    for k in (0, 6, 12, 18, 23):
        ax.annotate(f"{k}", (k, 0), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate("", xy=(23, -0.25), xytext=(0, -0.25), arrowprops=dict(arrowstyle="<->", color=RED))
    ax.text(11.5, -0.42, "数值上 0 与 23 相距最远", ha="center", fontsize=8.5, color=RED)
    ax.set_ylim(-0.6, 0.5); ax.set_yticks([]); ax.set_xlabel("小时 $h$ 作为普通数值")
    ax.spines["left"].set_visible(False)
    ax = axes[1]
    th = 2 * np.pi * h / 24
    ax.scatter(np.sin(th), np.cos(th), c=h, cmap="twilight", s=30, zorder=3)
    for k in (0, 6, 12, 18, 23):
        ax.annotate(f"{k}", (np.sin(th[k]), np.cos(th[k])), xytext=(7 * np.sin(th[k]), 7 * np.cos(th[k])),
                    textcoords="offset points", ha="center", va="center", fontsize=8)
    ax.plot([np.sin(th[23]), np.sin(th[0])], [np.cos(th[23]), np.cos(th[0])], color=GREEN, lw=2)
    ax.set_aspect("equal"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel(r"$\sin(2\pi h/24)$"); ax.set_ylabel(r"$\cos(2\pi h/24)$")
    ax.set_title("编码后 0 与 23 相邻", fontsize=8.5, color=GREEN)
    return f


@fig("chapter06", "rf_importance")
def rf_importance():
    import re
    txt = (OUT / "B_ch08_logit_forest.txt").read_text(encoding="utf-8")
    block = txt.split("随机森林置换重要性 Top10：")[1].split("\n\n")[0]
    rows = re.findall(r"^\s*(\S+)\s+([\d.]+)$", block, flags=re.M)
    names = {"inv_motorcycle": "涉及摩托车", "inv_pedestrian": "涉及行人", "speed_per10": "限速（每 10 mph）",
             "area_rural": "农村道路", "inv_pedal_cycle": "涉及自行车", "night": "夜间",
             "road_roundabout": "环岛", "road_dual_cw": "双幅路", "junction_junction": "交叉口",
             "light_dark_lit": "黑暗但有路灯"}
    rows = rows[::-1]
    f, ax = plt.subplots(figsize=(6.6, 3.2))
    vals = [float(v) for _, v in rows]
    ax.barh(range(len(rows)), vals, color=[ORANGE if k.startswith("inv_") else BLUE for k, _ in rows], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [names.get(k, k) for k, _ in rows], fontsize=8.5)
    ax.set_xlabel("置换重要性（打乱该特征后 PR-AUC 的下降）")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_title("橙色：由车辆表与伤亡人员表汇总的“涉事方”特征", fontsize=8.5)
    return f


@fig("chapter06", "interaction")
def interaction():
    txt = (OUT / "B_ch08_logit_forest.txt").read_text(encoding="utf-8")
    block = txt.split("限速 x 光照 KSI 占比：")[1].strip().splitlines()
    rows = []
    for l in block[2:]:
        parts = l.split()
        if len(parts) != 3:
            break
        rows.append(parts)
    sp = [float(r[0]) for r in rows]; dark = [float(r[1]) for r in rows]; day = [float(r[2]) for r in rows]
    f, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(sp, [100 * d for d in day], "o-", color=BLUE, label="白天")
    ax.plot(sp, [100 * d for d in dark], "s-", color=ORANGE, label="黑暗且无路灯")
    for x_, a_, b_ in zip(sp, day, dark):
        ax.annotate(f"{100 * (b_ - a_):+.1f}", (x_, 100 * b_), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=ORANGE)
    ax.set_xlabel("限速（mph）"); ax.set_ylabel("KSI 事故占比（%）"); ax.set_xticks(sp)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_ylim(15, 43)
    ax.set_title("橙色数字：同一限速下，黑暗无路灯比白天高出的百分点", fontsize=8.5, pad=10)
    return f


# ---------------- 第 11 章 ----------------
@fig("chapter11", "nhtsa_eval")
def nhtsa_eval():
    import re
    txt = (OUT / "ch11_nhtsa_eval.txt").read_text(encoding="utf-8")
    llm_acc = float(re.search(r"准确率=([\d.]+) 宏平均F1", txt).group(1))
    base = [(int(n.replace(",", "")), float(a)) for n, a in re.findall(r"训练样本\s+([\d,]+)：准确率=([\d.]+)", txt)]
    blk = txt.split("混淆矩阵（行=NHTSA 部件字段，列=LLM）：")[1].split("\n\n")[0].strip().splitlines()
    classes = ["AIR BAGS", "ELECTRICAL SYSTEM", "ENGINE", "POWER TRAIN", "SEAT BELTS", "SERVICE BRAKES", "STEERING", "STRUCTURE"]
    M = np.array([[int(v) for v in l.split()[-8:]] for l in blk[1:]])     # 每行最后 8 个数为计数
    zh = ["安全气囊", "电气系统", "发动机", "动力传动", "安全带", "制动系统", "转向", "车身结构"]
    f, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    xs = [b[0] for b in base]
    ax.plot(xs, [b[1] for b in base], "o-", color=BLUE, label="TF-IDF + 逻辑回归")
    for x_, y_ in base:
        ax.annotate(f"{y_:.3f}", (x_, y_), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(llm_acc, color=ORANGE, ls="--", lw=1.4, label=f"大语言模型（无训练样本）：{llm_acc:.3f}")
    ax.set_xscale("log"); ax.set_xticks(xs, [f"{x:,}" for x in xs])
    ax.set_xlabel("基线的训练样本数（对数刻度）"); ax.set_ylabel("评价集准确率"); ax.set_ylim(0.7, 0.96)
    ax.legend(fontsize=7.8, frameon=False, loc="lower right")
    ax.set_title("(a) 与低成本基线比较", fontsize=9)
    ax = axes[1]
    ax.imshow(M, cmap="Blues", vmin=0, vmax=20)
    for i in range(8):
        for j in range(8):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 12 else "black")
    ax.set_xticks(range(8), zh, rotation=45, ha="right", fontsize=7.5); ax.set_yticks(range(8), zh, fontsize=7.5)
    ax.set_xlabel("大语言模型的分类", fontsize=8.5); ax.set_ylabel("参照标签（NHTSA 部件字段）", fontsize=8.5)
    ax.set_title("(b) 大语言模型的混淆矩阵（160 条）", fontsize=9)
    return f


# ---------------- 第 13 章 ----------------
def _report_rows(name):
    """从案例 A 的运行日志中读取最后一张 report() 输出的 MAE 表。"""
    lines = (OUT / name).read_text(encoding="utf-8").splitlines()
    tables, cur = [], None
    for l in lines:
        parts = l.split()
        if parts[:1] == ["method"]:
            cur = {}; tables.append(cur); continue
        if cur is not None and len(parts) >= 7:
            try:
                vals = list(map(float, parts[-6:]))
            except ValueError:
                cur = None; continue
            cur[" ".join(parts[:-6])] = dict(zip(["MAE", "RMSE", "AM_peak", "PM_peak", "other", "onset"], vals))
        else:
            cur = None
    return tables[-1]


@fig("chapter13", "method_compare")
def method_compare():
    a = _report_rows("A_ch07_baselines_ridge.txt")
    b = _report_rows("A_ch09_mlp_lstm.txt")
    g = _report_rows("A_ch09_graph_neighbors.txt")
    methods = [("持续性基准", a["persist"], GRAY), ("历史同时段均值", a["hist"], "#B0B0B0"), ("岭回归", a["ridge"], BLUE),
               ("前馈网络", b["mlp_s0"], GREEN), ("LSTM", b["lstm"], ORANGE), ("前馈网络 + 路网下游邻居", g["own+road_downstream_nb"], RED)]
    groups = [("MAE", "总体"), ("AM_peak", "早高峰"), ("PM_peak", "晚高峰"), ("other", "其他时段")]
    f, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]; w = 0.13
    for k, (lab, d, c) in enumerate(methods):
        ax.bar(np.arange(4) + (k - 2.5) * w, [d[g_] for g_, _ in groups], w, color=c, label=lab)
    ax.set_xticks(range(4), [x[1] for x in groups]); ax.set_ylabel("测试集 MAE（mph）")
    ax.set_ylim(0, 9.5); ax.legend(fontsize=7.2, frameon=False, ncol=2, loc="upper left")
    ax.set_title("(a) 总体与各时段", fontsize=9)
    ax = axes[1]
    vals = [d["onset"] for _, d, _ in methods]
    bb = ax.bar(range(len(methods)), vals, color=[c for *_, c in methods], width=0.65)
    ax.bar_label(bb, fmt="%.1f", fontsize=7.5, padding=1)
    ax.set_xticks([]); ax.set_ylim(0, 40)
    ax.set_title("(b) 拥堵形成阶段", fontsize=9)
    return f


def _persist_test():
    sys.path.insert(0, str(ROOT / "case_A_freeway"))
    from common import load_speed, to_long, split, H, onset_mask
    df = to_long(load_speed())
    g = df.groupby("sensor")["speed"]
    df["target"] = g.shift(-H)
    for k in range(1, 6):
        df[f"lag{k}"] = g.shift(k)
    _, _, te = split(df)
    te = te.dropna(subset=["speed", "target"] + [f"lag{k}" for k in range(1, 6)])
    te["onset"] = onset_mask(te["speed"].to_numpy(), te["target"].to_numpy())
    return te


@fig("chapter13", "persist_scatter")
def persist_scatter():
    te = _persist_test()
    err = (te["target"] - te["speed"]).abs()
    f, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    hb = ax.hexbin(te["speed"], te["target"], gridsize=45, bins="log", cmap="Blues", mincnt=1, extent=(0, 75, 0, 75))
    ax.plot([0, 75], [0, 75], color=GRAY, lw=1)
    ax.add_patch(Rectangle((50, 0), 25, 35, fill=False, ec=RED, lw=1.5, ls="--"))
    ax.text(51, 2, "拥堵形成\n阶段", color=RED, fontsize=7.5)
    ax.set_xlabel("当前车速 = 持续性基准的预测（mph）", fontsize=8.5); ax.set_ylabel("15 分钟后的实际车速（mph）", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_title(f"(a) 预测与实际（测试集 {len(te):,} 个样本）", fontsize=9)
    ax = axes[1]
    te = te.assign(err=err, hour=te.timestamp.dt.hour, wd=te.timestamp.dt.dayofweek < 5)
    for flag, lab, c in [(True, "工作日", BLUE), (False, "周末", ORANGE)]:
        h = te[te.wd == flag].groupby("hour").err.mean()
        ax.plot(h.index, h.values, "o-", ms=3, color=c, label=lab)
    ax.axhline(err.mean(), color=GRAY, ls="--", lw=1, label=f"总体 MAE {err.mean():.2f}")
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("小时"); ax.set_ylabel("MAE（mph）")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(b) 误差的日内分布", fontsize=9)
    print("persist_scatter n", len(te), "mae", round(err.mean(), 3), "onset", int(te.onset.sum()))
    return f


# ---------------- 第 16 章 ----------------
@fig("chapter16", "station_profiles")
def station_profiles():
    d = pd.read_parquet(ROOT / "data" / "mta_clean.parquet")
    d = d[d.ts.dt.dayofweek < 5]
    d = d.assign(hour=d.ts.dt.hour)
    sts = [("Grand Central-42 St", "Grand Central-42 St（通勤枢纽）", BLUE),
           ("Flushing-Main St", "Flushing-Main St（居住区）", GREEN),
           ("161 St-Yankee Stadium", "161 St-Yankee Stadium（球场）", ORANGE)]
    f, ax = plt.subplots(figsize=(7.4, 3.0))
    for st, lab, c in sts:
        x = d[d.station == st].groupby("hour").entries.mean()
        ax.plot(x.index, 100 * x / x.sum(), "o-", ms=3, color=c, label=lab)
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("小时（工作日）")
    ax.set_ylabel("占全天进站量的比例（%）")
    ax.legend(fontsize=8, frameon=False)
    return f


@fig("chapter16", "coverage")
def coverage():
    import re
    t3 = (OUT / "C_step3_models.txt").read_text(encoding="utf-8")
    t4 = (OUT / "C_step4_validate.txt").read_text(encoding="utf-8")
    overall = float(re.search(r"80% 预测区间实际覆盖率：([\d.]+)", t3).group(1))
    by_type = eval(re.search(r"80% 预测区间实际覆盖率：[\d.]+；按日期类型： (\{.*\})", t3).group(1))
    by_st = eval(re.search(r"80% 区间覆盖率按车站： (\{.*\})", t4).group(1))
    rel = {}
    blk = t4.split("按车站：")[1].split("\n\n")[0].strip().splitlines()[2:]
    for l in blk:
        parts = l.split()
        rel[" ".join(parts[:-4])] = float(parts[-1])
    f, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    names = {"regular_weekday": "常规工作日", "weekend": "周末", "holiday±1": "节假日前后", "game_day": "比赛日"}
    ks = ["regular_weekday", "weekend", "holiday±1", "game_day"]
    b = ax.bar(range(4), [100 * by_type[k] for k in ks], color=[BLUE, BLUE, ORANGE, ORANGE], width=0.6)
    ax.bar_label(b, fmt="%.1f", fontsize=8, padding=1)
    ax.axhline(80, color=RED, ls="--", lw=1.2, label="名义覆盖率 80%")
    ax.axhline(100 * overall, color=GRAY, ls=":", lw=1.2, label=f"总体 {100 * overall:.1f}%")
    ax.set_xticks(range(4), [names[k] for k in ks], fontsize=8); ax.set_ylim(50, 92)
    ax.set_ylabel("实际覆盖率（%）"); ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("(a) 80% 预测区间的实际覆盖率", fontsize=9)
    ax = axes[1]
    order = sorted(rel, key=rel.get)
    ax.barh(range(len(order)), [100 * rel[k] for k in order], color=[ORANGE if ("Stadium" in k or "Willets" in k) else BLUE for k in order], height=0.6)
    for i, k in enumerate(order):
        ax.text(100 * rel[k] + 0.6, i, f"{100 * rel[k]:.0f}%  （覆盖率 {100 * by_st[k]:.1f}%）", va="center", fontsize=7.5)
    ax.set_yticks(range(len(order)), order, fontsize=7.5); ax.set_xlim(0, 80)
    ax.set_xlabel("相对误差：MAE / 平均进站量（%）")
    ax.set_title("(b) 各车站的相对误差（橙色为球场车站）", fontsize=9)
    return f


# ---------------- 第 17 章 ----------------
@fig("chapter17", "error_pairs")
def error_pairs():
    import re
    t = (OUT / "B_ch17_error_audit.txt").read_text(encoding="utf-8")
    t1 = (OUT / "B_ch01_default_vs_audit.txt").read_text(encoding="utf-8")
    d_auc = float(re.search(r"\[默认做法\].*?ROC-AUC=([\d.]+)", t1).group(1))
    pre = float(re.search(r"事前字段\s+ROC-AUC=([\d.]+)", t).group(1))
    post = float(re.search(r"事前 \+ 事后字段\s+ROC-AUC=([\d.]+)", t).group(1))
    num = float(re.search(r"数值编码： ROC-AUC=([\d.]+)", t).group(1))
    oh = float(re.search(r"独热编码： ROC-AUC=([\d.]+)", t).group(1))
    rs = float(re.search(r"随机划分： ROC-AUC=([\d.]+)", t).group(1))
    gs = float(re.search(r"按事故分组： ROC-AUC=([\d.]+)", t).group(1))
    cal = re.findall(r"injury_based=(\d): 预测 ([\d.]+) vs 实际 ([\d.]+)", t)
    pairs = [("错误一：标签派生字段入模", d_auc, pre), ("错误二：事后字段入模", post, pre),
             ("错误三：车辆级记录随机划分", rs, gs), ("错误六：类别编码当作连续数值", num, oh)]
    f, axes = plt.subplots(1, 2, figsize=(8.8, 3.1), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for i, (lab, wv, cv) in enumerate(pairs[::-1]):
        ax.plot([cv, wv], [i, i], color=GRAY, lw=1.5, zorder=1)
        ax.scatter([wv], [i], color=RED, s=36, zorder=2, label="错误做法" if i == 0 else None)
        ax.scatter([cv], [i], color=BLUE, s=36, zorder=2, label="修正后" if i == 0 else None)
        ax.text(max(wv, cv) + 0.008, i, f"{wv:.3f} → {cv:.3f}", va="center", fontsize=7.8)
    ax.set_yticks(range(4), [p[0] for p in pairs[::-1]], fontsize=8)
    ax.set_xlim(0.6, 1.12); ax.set_xlabel("ROC-AUC")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("(a) 错误对指标的影响：方向并不总是“变高”", fontsize=9)
    ax = axes[1]
    labs = {"0": "警员现场判断", "1": "基于伤情的系统"}
    x = np.arange(len(cal)); w = 0.35
    b1 = ax.bar(x - w / 2, [100 * float(c[1]) for c in cal], w, color=GRAY, label="模型平均预测")
    b2 = ax.bar(x + w / 2, [100 * float(c[2]) for c in cal], w, color=ORANGE, label="实际 KSI 占比")
    ax.bar_label(b1, fmt="%.1f", fontsize=7.5, padding=1); ax.bar_label(b2, fmt="%.1f", fontsize=7.5, padding=1)
    ax.set_xticks(x, [labs[c[0]] for c in cal], fontsize=8); ax.set_ylim(0, 34)
    ax.set_ylabel("%"); ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("(b) 2025 年测试集：按报告系统的校准", fontsize=9)
    return f

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        ch, f = FIGS[n]; save(f(), ch, n); print("ok", n)
