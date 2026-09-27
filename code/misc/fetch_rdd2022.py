"""下载 RDD2022 的部分国家子集。总压缩包约 13 GB，但其中各国家子包为“存储”方式，
可用 HTTP 分段请求（remotezip）只读取需要的子包。pip install remotezip"""
import shutil, zipfile
from pathlib import Path
from remotezip import RemoteZip

URL = "https://ndownloader.figshare.com/files/38030910"     # RDD2022_released_through_CRDDC2022.zip
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "rdd2022"
OUT.mkdir(parents=True, exist_ok=True)
for c in ["Czech", "China_MotorBike", "United_States", "India"]:
    z = OUT / f"{c}.zip"
    with RemoteZip(URL) as rz, rz.open(f"RDD2022/{c}.zip") as src, open(z, "wb") as dst:
        shutil.copyfileobj(src, dst, 16 << 20)
    zipfile.ZipFile(z).extractall(OUT)
    print("done", c)
