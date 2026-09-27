"""第 9 章案例：RDD2022 路面病害图像的跨地区泛化检验（CPU 可运行的轻量版）。

做法：每个国家随机抽取 N 张训练集图像，用 ImageNet 预训练的 ResNet-18 提取图像特征（不微调），
对四类病害（D00 纵向裂缝、D10 横向裂缝、D20 龟裂、D40 坑槽）分别训练“图像中是否存在该病害”的
逻辑回归分类器，比较：
  (a) 国内评估：在同一国家内随机 70/30 划分；
  (b) 留一国家：用其他国家的全部抽样图像训练，在该国家同一批 30% 测试图像上评估。
另外检查国内随机划分中测试图像与训练图像的近似重复程度（特征余弦相似度）。
数据：RDD2022（Arya et al., 2024），需先下载并解压各国家压缩包到 RDD_ROOT。
"""
import os, sys, warnings, xml.etree.ElementTree as ET
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = Path(os.environ.get("RDD_ROOT", Path(__file__).resolve().parents[1] / "data" / "raw" / "rdd2022"))
COUNTRIES = ["Czech", "India", "United_States", "China_MotorBike"]
CLASSES = ["D00", "D10", "D20", "D40"]
N = int(os.environ.get("RDD_N", 800))
OUT = Path(__file__).resolve().parents[1] / "outputs"


def labels(xml):
    names = {o.find("name").text for o in ET.parse(xml).getroot().findall("object")}
    return [int(c in names) for c in CLASSES]


def sample(country, rng):
    imgs = sorted((ROOT / country / "train" / "images").glob("*.jpg"))
    pick = rng.choice(len(imgs), min(N, len(imgs)), replace=False)
    rows = []
    for i in sorted(pick):
        p = imgs[i]; y = labels(ROOT / country / "train" / "annotations" / "xmls" / (p.stem + ".xml"))
        rows.append([country, str(p), *y])
    return pd.DataFrame(rows, columns=["country", "path", *CLASSES])


def features(paths):
    from torchvision.models import resnet18, ResNet18_Weights
    w = ResNet18_Weights.IMAGENET1K_V1; net = resnet18(weights=w); net.fc = torch.nn.Identity(); net.eval()
    tf = w.transforms(); feats = []
    with torch.no_grad():
        for k in range(0, len(paths), 64):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[k:k + 64]])
            feats.append(net(x).numpy())
    return np.vstack(feats)


if __name__ == "__main__":
    torch.set_num_threads(8); np.seterr(all="ignore")
    rng = np.random.default_rng(0)
    df = pd.concat([sample(c, rng) for c in COUNTRIES], ignore_index=True)
    cache = OUT.parent / "data" / "rdd_features.npy"
    if cache.exists() and len(np.load(cache)) == len(df):
        F = np.load(cache)
    else:
        F = features(df.path.tolist()); np.save(cache, F)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    df["test"] = False
    for c in COUNTRIES:
        idx = df.index[df.country == c]; df.loc[rng.choice(idx, int(0.3 * len(idx)), replace=False), "test"] = True
    print("各国家抽样图像数与病害出现比例：")
    print(df.groupby("country")[CLASSES].agg(["mean"]).round(3).droplevel(1, axis=1).assign(n=df.groupby("country").size()).to_string())
    rows = []
    for c in COUNTRIES:
        te = (df.country == c) & df.test
        for setting, tr in [("国内评估", (df.country == c) & ~df.test), ("留一国家", df.country != c)]:
            aps = []
            for k in CLASSES:
                y_tr, y_te = df.loc[tr, k].to_numpy(), df.loc[te, k].to_numpy()
                if y_te.sum() < 5 or y_tr.sum() < 5: aps.append(np.nan); continue
                m = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(F[tr], y_tr)
                aps.append(average_precision_score(y_te, m.predict_proba(F[te])[:, 1]))
            rows.append({"国家": c, "设置": setting, **dict(zip(CLASSES, np.round(aps, 3))), "平均AP": round(np.nanmean(aps), 3)})
        # 近似重复：国内随机划分下，测试图像与训练图像的最大余弦相似度
        S = F[te] @ F[(df.country == c) & ~df.test].T
        So = F[te] @ F[df.country != c].T
        rows[-1]["测试图与同国训练图最大相似度中位数"] = round(float(np.median(S.max(1))), 3)
        rows[-1]["测试图与他国训练图最大相似度中位数"] = round(float(np.median(So.max(1))), 3)
    res = pd.DataFrame(rows)
    print("\n平均精确率（AP）：")
    print(res.to_string(index=False))
    print("\n先验比例（随机猜测的 AP）已在上表“病害出现比例”中给出。")
