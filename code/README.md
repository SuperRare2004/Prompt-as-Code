# 《提示即代码》配套代码

本目录收录书中各案例的完整代码。这些代码是按照书中的 Prompt 与约束，由 AI（Claude）生成、经过人工审查并在真实数据上实际运行的版本；书中报告的数值均来自这些脚本的运行结果（运行日志见 `outputs/`）。

## 数据来源（全部为公开数据）

| 案例 | 数据 | 来源 |
| --- | --- | --- |
| A 高速公路短时预测 | METR-LA：洛杉矶县高速公路 207 个线圈检测器，2012-03-01 至 2012-06-27，5 分钟平均车速（mph），0 表示缺失；检测器坐标与路网行驶距离 | Li et al. (2018) DCRNN；Zenodo record 5146275；github.com/liyaguang/DCRNN |
| B 事故严重程度 | STATS19 英国道路交通伤亡事故数据，2021–2025 年事故表、车辆表、伤亡人员表 | 英国交通部 data.dft.gov.uk |
| C 地铁进站客流 | MTA Subway Hourly Ridership 2020–2024（选取 9 座车站，按车站 × 小时 × 支付方式聚合）；MLB 主场赛程；Open-Meteo 天气实测与历史预报 | data.ny.gov（wujg-7c2s）；statsapi.mlb.com；open-meteo.com |
| 第 9 章路面病害 | RDD2022 多国路面病害图像 | Arya et al. (2022)，figshare 21431547 |
| 第 11 章文本分类 | NHTSA 车辆安全投诉（16 款车型 2018–2021 年款）；书中评价集与模型输出保存在 `misc/nhtsa_eval/` | api.nhtsa.gov |

## 运行步骤

```bash
pip install -r requirements.txt
bash data/download_data.sh      # 下载数据到 data/raw/（约 350 MB；MTA 数据按月分段请求，约需 10 分钟）
bash run_all.sh                 # 依次运行案例 A、B、C，输出写入 outputs/
```

在普通笔记本电脑（CPU）上，全部脚本约需 15–25 分钟，其中 LSTM 训练最慢。

## 文件与章节对应

| 文件 | 章节 | 内容 |
| --- | --- | --- |
| `case_A_freeway/common.py` | — | 数据读取（0 → 缺失）、时间划分、分时段与“拥堵形成阶段”评价 |
| `case_A_freeway/preface_leakage.py` | 前言、第 13 章 | 随机划分、居中滑动窗口、0 值处理对结果的影响 |
| `case_A_freeway/ch07_baselines_ridge.py` | 第 7 章 | 持续性与历史均值基准、岭回归、回归树深度诊断 |
| `case_A_freeway/ch09_mlp_lstm.py` | 第 9 章 | MLP 多随机种子比较；LSTM；“先 dropna 再切窗口”错误的规模 |
| `case_A_freeway/ch09_graph_neighbors.py` | 第 9 章 | 直线距离邻居 vs 路网有向下游邻居 |
| `case_A_freeway/ch10_signal_rl.py` | 第 10、14 章 | 简化交叉口仿真器与表格型 Q-learning：奖励与约束设计的影响 |
| `case_B_crash/common.py` | — | STATS19 读取、KSI 标签、涉事方汇总、时间划分 |
| `case_B_crash/ch01_ch17_default_vs_audit.py` | 第 1、17 章 | 默认做法（全字段 + 随机划分）与字段审查后的对比 |
| `case_B_crash/ch08_logit_forest.py` | 第 8 章 | 逻辑回归（加权/不加权、优势比）、随机森林、雨天分组、覆盖率 |
| `case_B_crash/ch17_error_audit.py` | 第 17 章 | 十类错误中可量化部分的逐项复现 |
| `case_C_metro/fetch_mta.py`、`fetch_context.py` | 第 16 章 | 下载 MTA 客流、MLB 赛程与天气 |
| `case_C_metro/step1_clean.py` … `step4_validate.py` | 第 15、16 章 | 描述与清洗、特征、基准与模型、验证 |
| `ch08_traditional_ml/demos.py` | 第 8 章 | 各模型小节的演示：diabetes、乳腺肿瘤数据；METR-LA 检测器日变化曲线的 K-均值、GMM、PCA |
| `ch10_rl/demos.py` | 第 10 章 | 价值迭代、迷宫 Q-learning、悬崖行走中的 Q-learning 与 SARSA、CartPole 上的 DQN 与 REINFORCE |
| `figures/concept_figures.py` | 第 7—10 章 | 概念示意图（其中 ROC、手肘图、GMM、PCA、网格世界价值等使用真实数据或真实计算） |
| `figures/make_figures.py` | 第 7、8、9、16、17 章 | 根据运行结果绘制书中插图，输出到仓库的 `figures/` |
| `misc/ch09_rdd_transfer.py` | 第 9 章 | RDD2022 四个子集的轻量跨地区检验（ResNet-18 特征 + 逻辑回归，CPU 可运行） |
| `misc/ch09_road_damage_eval.py` | 第 9 章 | RDD2022 留一国家评估框架（未在书中报告数值） |
| `misc/fetch_nhtsa.py`、`misc/ch11_nhtsa_eval.py` | 第 11 章 | 下载 NHTSA 投诉；评价 LLM 输出与 TF-IDF 基线 |
| `misc/ch11_llm_text_classification.py` | 第 11 章 | LLM 分类与抽取、TF-IDF 基线、证据/名录/稳定性检查 |

## 说明

- 简化仿真器（`ch10_signal_rl.py`）仅用于演示 MDP 设计的影响，工程研究应使用 SUMO 等经过标定的微观仿真平台。
- `ch11_llm_text_classification.py` 默认通过 Anthropic Python SDK 调用模型；换用其他服务时只需替换 `classify_with_llm()`，评价逻辑不变。
- 公开数据会更新（例如 STATS19 的“最近五年”文件会随年度发布滚动），重新下载后个别数值可能与书中略有差异；书中数值对应 2026 年 9 月下载的数据版本。
- 这些代码的意义不在于“标准答案”，而在于展示：在明确的 Prompt 与约束下 AI 会生成什么样的代码，以及人需要在哪些地方审查和修改。书中各案例的“本案例中人的判断”一节列出了这些审查点。
