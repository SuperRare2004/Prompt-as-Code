"""第 9 章案例：采用公开路面病害检测模型之前的评估框架（RDD2022）。

数据：RDD2022（Arya et al., 2022），https://doi.org/10.6084/m9.figshare.21431547
     解压后按国家分目录：<root>/<Country>/train/images/*.jpg 与 annotations/xmls/*.xml（Pascal VOC）。
本脚本实现：(1) VOC 标注转 YOLO 格式；(2) “留一国家”划分；(3) 训练与分病害类型评价的命令组织。
检测模型使用 ultralytics（pip install ultralytics）。数据体积较大，书中未报告数值，请读者自行运行。
按路线分组划分需要拍摄路线信息；RDD2022 未提供时，可用文件名前缀或拍摄时间近似分组（需人工核对）。
"""
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["D00", "D10", "D20", "D40"]   # 纵向裂缝、横向裂缝、龟裂、坑槽


def voc_to_yolo(xml_path, out_txt):
    root = ET.parse(xml_path).getroot()
    w, h = float(root.find("size/width").text), float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASSES:
            continue                                     # 其他类别不参与本评估
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"])
        lines.append(f"{CLASSES.index(name)} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    out_txt.write_text("\n".join(lines))


def prepare(root, out):
    """转换标注，并返回 {国家: [图像路径]}。"""
    by_country = {}
    for c in sorted(p.name for p in Path(root).iterdir() if p.is_dir()):
        imgs = sorted((Path(root) / c / "train" / "images").glob("*.jpg"))
        lab_dir = Path(out) / "labels" / c
        lab_dir.mkdir(parents=True, exist_ok=True)
        kept = []
        for im in imgs:
            xml = Path(root) / c / "train" / "annotations" / "xmls" / (im.stem + ".xml")
            if xml.exists():
                voc_to_yolo(xml, lab_dir / (im.stem + ".txt"))
                kept.append(im)
        by_country[c] = kept
        print(c, len(kept))
    return by_country


def leave_one_country_out(by_country, holdout, out, val_frac=0.1, seed=0):
    """用其他国家训练与验证，在 holdout 国家上测试；返回 ultralytics 数据配置文件路径。"""
    random.seed(seed)
    train, val = [], []
    for c, imgs in by_country.items():
        if c == holdout:
            continue
        imgs = imgs[:]
        random.shuffle(imgs)          # 注意：连续帧近似重复，严格做法应按路线/前缀分组后再划分
        k = int(len(imgs) * val_frac)
        val += imgs[:k]
        train += imgs[k:]
    d = Path(out) / f"loco_{holdout}"
    d.mkdir(parents=True, exist_ok=True)
    for name, lst in [("train", train), ("val", val), ("test", by_country[holdout])]:
        (d / f"{name}.txt").write_text("\n".join(map(str, lst)))
    cfg = d / "data.yaml"
    cfg.write_text(f"train: {d/'train.txt'}\nval: {d/'val.txt'}\ntest: {d/'test.txt'}\nnames: {CLASSES}\n")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="RDD2022 解压目录")
    ap.add_argument("--out", default="rdd_work")
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    by_country = prepare(a.root, a.out)
    from ultralytics import YOLO
    for holdout in by_country:
        cfg = leave_one_country_out(by_country, holdout, a.out)
        model = YOLO("yolov8s.pt")
        model.train(data=str(cfg), epochs=a.epochs, imgsz=640, project=a.out, name=f"loco_{holdout}")
        m = model.val(data=str(cfg), split="test")
        # 分病害类型报告：重点关注 D40（坑槽）的召回率
        for i, c in enumerate(CLASSES):
            print(f"[holdout={holdout}] {c}: precision={m.box.p[i]:.3f} recall={m.box.r[i]:.3f} mAP50={m.box.ap50[i]:.3f}")
