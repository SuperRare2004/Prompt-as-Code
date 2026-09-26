#!/usr/bin/env bash
# 依次运行全部可复现案例，并把输出保存到 outputs/。需先运行 data/download_data.sh。
set -e
cd "$(dirname "$0")"; mkdir -p outputs
( cd case_A_freeway
  python preface_leakage.py        > ../outputs/A_preface_leakage.txt
  python ch07_baselines_ridge.py   > ../outputs/A_ch07_baselines_ridge.txt
  python ch09_mlp_lstm.py          > ../outputs/A_ch09_mlp_lstm.txt
  python ch09_graph_neighbors.py   > ../outputs/A_ch09_graph_neighbors.txt
  python ch10_signal_rl.py         > ../outputs/A_ch10_signal_rl.txt )
( cd case_B_crash
  python ch01_ch17_default_vs_audit.py > ../outputs/B_ch01_default_vs_audit.txt
  python ch08_logit_forest.py          > ../outputs/B_ch08_logit_forest.txt
  python ch17_error_audit.py           > ../outputs/B_ch17_error_audit.txt )
( cd case_C_metro
  python step1_clean.py     > ../outputs/C_step1_clean.txt
  python step2_features.py  > ../outputs/C_step2_features.txt
  python step3_models.py    > ../outputs/C_step3_models.txt
  python step4_validate.py  > ../outputs/C_step4_validate.txt )
( cd ch08_traditional_ml && python demos.py > ../outputs/ch08_demos.txt )
( cd ch10_rl && python demos.py > ../outputs/ch10_demos.txt )
python figures/make_figures.py          # 根据运行结果绘制书中插图
echo "done: see outputs/ and ../figures/"
