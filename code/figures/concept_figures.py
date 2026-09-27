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


if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        ch, f = FIGS[n]; save(f(), ch, n); print("ok", n)
