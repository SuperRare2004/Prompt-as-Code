# 交接文档：《提示即代码：面向真实问题的 AI 协作方法》

更新时间：2026-09-27。最近一次提交为 `bef6b06 Update: Examples`，此后的修改尚未提交（见第 8 节）。

## 1. 当前状态

- 全书 18 章 + 前言 + 附录 A 已成稿，`out/prompt_as_code.pdf` 共 288 页。用 XeLaTeX 编译无报错，也没有未定义的引用。
- 三个贯穿案例全部基于真实公开数据。书中报告的每一个数字都由 `code/` 中的脚本实际运行得到，`code/run_all.sh` 可以端到端复现（最近一次于 2026-09-27 完整跑通）。
- 全书已没有图占位：数据图由 Python 生成，结构图为 TikZ 源文件，均为原创。
- 剩余待办见第 7 节，主要是作者信息、商标许可和通读。

## 2. 仓库结构

```
prompt_as_code.tex          主文件（XeLaTeX）
chapters/00–18_*.tex        前言与各章；19_appendix_code_data.tex 为附录 A
pre_settings/               宏包与宏定义（\figholder、aioutput 环境、下划线可断行等）
bib_Reference/reference.bib 参考文献
figures/chapterXX/          插图：*.pdf 由脚本生成；chapter09–11/tikz/*.tex 为 TikZ 源文件
code/                       配套代码（说明见 code/README.md）
  case_A_freeway/           案例 A：METR-LA（前言、第 7、9、10 章）
  case_B_crash/             案例 B：STATS19（第 1、8、17 章）
  case_C_metro/             案例 C：NYC MTA（第 15、16 章）
  ch08_traditional_ml/      第 8 章各小节演示
  ch10_rl/                  第 10 章各小节演示
  misc/                     第 9 章 RDD2022、第 11 章 NHTSA / LLM
  figures/                  make_figures.py（案例结果图）与 concept_figures.py（概念图）
  data/download_data.sh     下载全部原始数据到 data/raw/（已被 git 忽略，约 1.9 GB）
  outputs/                  各脚本的运行日志（纳入版本管理，是书中数字的依据）
design/出版前待办清单.md    出版前检查清单
```

## 3. 编译与复现

**编译书稿**

```bash
latexmk -xelatex -shell-escape -outdir=out -jobname=prompt_as_code prompt_as_code.tex
```

**复现全部数字与插图**

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash data/download_data.sh   # METR-LA、STATS19、MTA（按月分段请求，约 10 分钟）、MLB、天气、NHTSA、RDD2022 四个子集
bash run_all.sh              # 输出写入 outputs/，插图写入 ../figures/
```

**环境注意事项**

以下是本机（macOS，Darwin 27）上遇到的问题。

- `scipy 1.15.3` 的二进制在本系统上加载失败（dlopen 报错），降级到 `scipy==1.13.1` 后正常。
- `lightgbm` 需要 `libomp`，本机没有，所以代码改用 sklearn 的 `HistGradientBoosting*`，不依赖 lightgbm。
- 写作时使用的虚拟环境在临时目录中，已不存在，接手时请按上面的步骤新建。
- Python 3.10；主要依赖为 numpy、pandas、scikit-learn、statsmodels、torch、torchvision、gymnasium、requests、remotezip、pyarrow。

## 4. 三个贯穿案例与关键结果

| 案例 | 数据 | 关键真实结果（出处） |
| --- | --- | --- |
| A 高速公路短时预测 | METR-LA，207 个检测器，2012-03 至 06，5 分钟车速（mph），**0 表示缺失** | AI 默认做法（0 当作车速 + 随机划分 + 居中窗口）得到 R² 0.964；修正后为 0.826，只比持续性基准好约 9%（前言）。LSTM 总体最好（MAE 2.93），但在拥堵形成阶段最差（33.4，第 9 章）。路网下游邻居使拥堵形成阶段误差降低约 9%，直线距离邻居几乎无效（第 9 章）。 |
| B 事故严重程度 | STATS19 2021—2025，513,801 起事故 | 全字段 + 随机划分得到 AUC 1.000，原因是标签派生字段入模（第 1、17 章）。审查字段后 AUC 为 0.653、PR-AUC 为 0.412。基于伤情报告系统的事故占比从 2021 年的 49.6% 升到 2025 年的 86.5%，导致 KSI 比例“虚升”（第 17 章）。 |
| C 地铁客流 | MTA 分小时进站量，9 座车站，2022-02 至 2024-12；另有 MLB 赛程与 Open-Meteo 天气 | 只看 MetroCard，客流“下降” 35%；实际加上 OMNY 后是上升（第 15、16 章）。GBDT 的加权 MAE 为 227.7，四周均值基准为 270.2。比赛散场时段仍低估约 40%，80% 预测区间的实际覆盖率只有 74%（第 16 章）。 |

**独立案例**

| 章节 | 数据 | 关键结果 |
| --- | --- | --- |
| 第 9 章 | RDD2022（捷克、印度、美国、中国摩托车四个子集） | 留一国家评估下 AP 全部下降，例如印度从 0.406 降到 0.174 |
| 第 10 章 | 简化交叉口仿真 | 只奖励匝道排队的 RL 策略会“饿死”其他进口 |
| 第 11 章 | NHTSA 投诉，160 条评价集 | LLM 准确率 0.925；TF-IDF 基线用 10,411 条样本训练，准确率 0.875 |

**第 11 章的特别说明**：由于本机没有 API 凭据，LLM 分类由 Claude 在写作会话中按 `code/misc/nhtsa_eval/nhtsa_guide.txt` 逐条盲分类完成，书中已如实说明。评价集和输出保存在 `code/misc/nhtsa_eval/`。

## 5. 写作约定（作者明确提出的要求）

1. **案例必须来自真实数据或可核实的真实事件**，数字必须由代码实际计算得到，不得编造。算不出来的内容，只写方法，不写数值。
2. **学术语体**，面向大学生和科研人员。不使用拟人化角色（如“小林”）或“组会”一类的口语场景。
3. 每个案例都要写“完整代码见 `code/...`”，代码按“案例 / 章节”组织。
4. 书中的“项目情境”（谁提出需求、项目组的决定）属于教学设计，要写明不代表相关机构真实开展过该项目。数据、数据现象和计算结果必须真实。
5. 插图可以先用 `\figholder{图题}{标签}{说明}` 占位（宏定义在 `pre_settings/macros.tex`），但不要从网上直接拿图（版权问题）。

## 6. 已知的坑（改代码或改数据时要注意）

**数据**

- **METR-LA**：0 是缺失，不是停车；有 2,148 个时刻全部检测器都为 0。先 `dropna` 再按行切窗口，会有 4.6% 的窗口跨越缺失。
- **STATS19**：`collision_adjusted_severity_*` 和 `enhanced_severity_collision` 由标签派生，“警员是否到场”“伤亡人数”是事后字段，都不能作为特征。“最近五年”文件每年滚动，重新下载后数字会略有变化（书中对应 2026 年 9 月下载的版本）。
- **MTA 接口**：跨长时间段的服务器端聚合很慢，必须按“车站 × 月”分段请求。标签是 MetroCard 与 OMNY 之和。夏令时日期会缺一小时。
- **MLB 赛程接口**：跨多个赛季的日期范围只会返回一个赛季，必须逐赛季请求。
- **NHTSA**：数据库持续新增投诉，重新下载后抽样会变，所以书中的评价集固定保存在仓库里。
- **RDD2022**：总包 13 GB，但其中各国家子包是“存储”方式，可以用 `remotezip` 分段下载（`misc/fetch_rdd2022.py`）。Norway 子集有 10.6 GB，没有使用。

**LaTeX / TikZ**

- 在 `\foreach` 的列表里不能出现 `/`（例如 “CNN/ViT” 会报错）。
- TikZ 样式不要命名为 `out` 或 `id`，会和内置键冲突。
- `pre_settings/macros.tex` 把 `\_` 重定义为可断行，用于让长代码路径换行。
- 图题中含 `\texttt` 路径时，请用 `\caption[短题]{长题}`，否则插图目录会溢出版心。

## 7. 待办（详见 `design/出版前待办清单.md`）

- [ ] 作者信息，以及前言末尾的落款（目前是“编者 `\today`”）。
- [ ] 第 1 章各 AI 产品的标志图属于商标，需要确认使用许可，或改为文字列表。
- [ ] Open-Meteo 的免费接口限非商业用途。书中只引用由它得到的统计结果，但商业出版前建议与出版社确认；如有需要，天气实测可改用 NOAA 的公有领域数据。
- [ ] （可选）第 11 章改用 API 批量运行，并报告多次运行的稳定性（`code/misc/ch11_llm_text_classification.py`）。
- [ ] （可选）第 9 章在 GPU 上训练完整的检测模型（`code/misc/ch09_road_damage_eval.py`）。
- [ ] 全书通读：术语统一、中英文空格与标点风格，以及源稿中部分公式段落的排版微调。

## 8. Git 状态与建议

- 上次提交之后，新增和修改了第 9—11 章的插图与案例、附录、参考文献、`code/misc/` 下的 NHTSA 与 RDD 脚本、`code/figures/concept_figures.py`，以及若干插图 PDF。这些都尚未提交，请审阅后再提交。
- `.gitignore` 已忽略 `code/data/raw/`、`code/data/*.parquet`、`code/data/*.npy` 和 `__pycache__/`。`code/outputs/` 需要纳入版本管理。
