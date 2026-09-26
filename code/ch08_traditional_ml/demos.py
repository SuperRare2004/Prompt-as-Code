"""第 8 章各小节的演示代码（对应书中“实现”段落）。

回归：scikit-learn 自带的 diabetes 数据集（442 名患者，目标为一年后的疾病进展指标）。
分类：scikit-learn 自带的 breast_cancer 数据集（569 个样本，良性/恶性）。
无监督：METR-LA 各检测器的工作日平均速度日变化曲线（需先运行 data/download_data.sh）。
用法：python demos.py            # 运行全部
      python demos.py linear ridge   # 只运行指定小节
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
DEMOS = {}


def demo(f):
    DEMOS[f.__name__] = f
    return f


def diabetes_split(cols=None):
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X[cols] if cols else X
    return train_test_split(X, y, test_size=0.25, random_state=0)


@demo
def linear():
    from sklearn.linear_model import LinearRegression
    Xtr, Xte, ytr, yte = diabetes_split(["bmi"])
    m = LinearRegression().fit(Xtr, ytr)
    print(f"[线性回归, BMI] 斜率={m.coef_[0]:.1f} 截距={m.intercept_:.1f} 测试 R2={r2_score(yte, m.predict(Xte)):.3f} MAE={mean_absolute_error(yte, m.predict(Xte)):.1f}")


@demo
def poly():
    from sklearn.linear_model import LinearRegression
    Xtr, Xte, ytr, yte = diabetes_split(["bmi"])
    for d in (1, 2, 3, 8):
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(Xtr, ytr)
        print(f"[多项式回归, BMI] 阶数={d} 训练 R2={r2_score(ytr, m.predict(Xtr)):.3f} 测试 R2={r2_score(yte, m.predict(Xte)):.3f}")


@demo
def ridge():
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    Xtr, Xte, ytr, yte = diabetes_split()
    for name, m in [("OLS", LinearRegression()), ("Ridge(a=1)", Ridge(1.0)), ("Lasso(a=0.5)", Lasso(0.5)),
                    ("ElasticNet(a=0.01,l1=0.5)", ElasticNet(alpha=0.01, l1_ratio=0.5))]:
        m = make_pipeline(StandardScaler(), m).fit(Xtr, ytr)
        coef = m[-1].coef_
        print(f"[{name:<26s}] 测试 R2={r2_score(yte, m.predict(Xte)):.3f} 非零系数={int(np.sum(np.abs(coef) > 1e-6))}/10")


@demo
def stepwise():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    Xtr, Xte, ytr, yte = diabetes_split()
    chosen, remaining, best_prev = [], list(Xtr.columns), -np.inf
    while remaining:
        scores = {f: cross_val_score(LinearRegression(), Xtr[chosen + [f]], ytr, cv=5, scoring="r2").mean() for f in remaining}
        f, s = max(scores.items(), key=lambda kv: kv[1])
        if s <= best_prev + 0.002:           # 改进小于 0.002 即停止
            break
        chosen.append(f); remaining.remove(f); best_prev = s
        print(f"[向前逐步回归] 加入 {f:<4s} 交叉验证 R2={s:.3f}")
    m = LinearRegression().fit(Xtr[chosen], ytr)
    print(f"[向前逐步回归] 最终特征={chosen} 测试 R2={r2_score(yte, m.predict(Xte[chosen])):.3f}")


def cancer_split():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


@demo
def logistic():
    from sklearn.linear_model import LogisticRegression
    Xtr, Xte, ytr, yte = cancer_split()
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    print(f"[逻辑回归] 测试 Accuracy={accuracy_score(yte, p > 0.5):.3f} ROC-AUC={roc_auc_score(yte, p):.3f}")


@demo
def svm():
    from sklearn.svm import SVC
    Xtr, Xte, ytr, yte = cancer_split()
    for k in ("linear", "rbf"):
        for scale in (False, True):
            m = make_pipeline(StandardScaler(), SVC(kernel=k)) if scale else SVC(kernel=k)
            m.fit(Xtr, ytr)
            print(f"[SVM kernel={k:<6s} 标准化={scale!s:<5s}] 测试 Accuracy={accuracy_score(yte, m.predict(Xte)):.3f} 支持向量数={int(np.sum(m[-1].n_support_ if scale else m.n_support_))}")


@demo
def tree():
    from sklearn.tree import DecisionTreeClassifier, export_text
    Xtr, Xte, ytr, yte = cancer_split()
    for d in (2, 4, None):
        t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        print(f"[决策树 max_depth={d}] 叶节点={t.get_n_leaves()} 训练 Acc={t.score(Xtr, ytr):.3f} 测试 Acc={t.score(Xte, yte):.3f}")
    print(export_text(DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xtr, ytr), feature_names=list(Xtr.columns)))


@demo
def forest():
    from sklearn.ensemble import RandomForestClassifier
    Xtr, Xte, ytr, yte = cancer_split()
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    print(f"[随机森林] 测试 Accuracy={rf.score(Xte, yte):.3f} ROC-AUC={roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]):.3f}")
    imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False).head(5)
    print("[随机森林] 不纯度重要性 Top5：", imp.round(3).to_dict())


def speed_profiles():
    """每个检测器工作日 6:00-20:00 的平均速度曲线（每 15 分钟一个点）。"""
    sys.path.insert(0, "../case_A_freeway")
    from common import load_speed
    w = load_speed()
    w = w[w.index.dayofweek < 5].between_time("06:00", "19:59")
    prof = w.groupby(w.index.floor("15min").time).mean().T       # 行：检测器；列：时段
    return prof.dropna()


@demo
def kmeans():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    P = speed_profiles()
    for k in range(2, 8):
        km = KMeans(k, n_init=10, random_state=0).fit(P)
        print(f"[K-means k={k}] SSE={km.inertia_:,.0f} 轮廓系数={silhouette_score(P, km.labels_):.3f}")
    km = KMeans(3, n_init=10, random_state=0).fit(P)
    for c in range(3):
        prof = P[km.labels_ == c].mean()
        print(f"[K-means k=3] 簇{c}: 检测器数={int((km.labels_ == c).sum())} 早8点均速={prof.iloc[8]:.1f} 晚17点均速={prof.iloc[44]:.1f} 全天最低={prof.min():.1f} mph")


@demo
def gmm():
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    P = speed_profiles()
    Z = PCA(5, random_state=0).fit_transform(P)
    for k in range(1, 6):
        g = GaussianMixture(k, covariance_type="full", random_state=0).fit(Z)
        print(f"[GMM k={k}] BIC={g.bic(Z):.1f}")
    g = GaussianMixture(3, random_state=0).fit(Z)
    prob = g.predict_proba(Z).max(axis=1)
    print(f"[GMM k=3] 最大后验概率 < 0.8 的检测器（归属不确定）：{int((prob < 0.8).sum())}/{len(prob)}")


@demo
def pca():
    from sklearn.decomposition import PCA
    P = speed_profiles()
    p = PCA().fit(P - P.mean())
    cum = np.cumsum(p.explained_variance_ratio_)
    print(f"[PCA] 曲线维度={P.shape[1]}；前 1/2/3 个主成分累计解释方差={cum[0]:.3f}/{cum[1]:.3f}/{cum[2]:.3f}")
    pc1 = pd.Series(p.components_[0], index=[str(t) for t in P.columns])
    print("[PCA] 第一主成分载荷绝对值最大的时段：", pc1.abs().sort_values(ascending=False).head(4).index.tolist())


if __name__ == "__main__":
    for name in (sys.argv[1:] or DEMOS):
        DEMOS[name]()
