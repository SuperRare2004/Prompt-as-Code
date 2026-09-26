"""第 9 章案例：前馈网络与 LSTM（案例 A，METR-LA）。

1) MLP 与岭回归使用相同特征，比较多个随机种子下的差异；
2) LSTM 读取过去 12 个时刻（1 小时）的速度序列；
3) 检查一个常见的窗口构造错误：先 dropna 删除缺失行再按行切窗口，
   会把缺失前后相隔很久的记录拼成“连续”窗口。
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load_speed, to_long, split, report, H, TRAIN_END, VALID_END

torch.set_num_threads(8)
wide = load_speed()
L = 12

# ---------- 窗口构造错误的规模：先删缺失行再切窗口 ----------
bad, total = 0, 0
for s in wide.columns:
    ts = wide[s].dropna().index            # 错误做法：删除缺失行
    gap = (ts[L - 1 + H:] - ts[:-(L - 1 + H)]) != pd.Timedelta(minutes=5 * (L - 1 + H))
    bad += int(gap.sum()); total += len(gap)
print(f"先 dropna 再按行切窗口：{bad:,}/{total:,} 个窗口（{bad/total:.1%}）实际跨越了缺失时段")

# ---------- 表格特征：岭回归 vs MLP ----------
df = to_long(wide)
g = df.groupby("sensor")["speed"]
df["target"] = g.shift(-H)
for k in range(1, 6):
    df[f"lag{k}"] = g.shift(k)
tod = df.timestamp.dt.hour * 12 + df.timestamp.dt.minute // 5
df["tod_sin"], df["tod_cos"] = np.sin(2 * np.pi * tod / 288), np.cos(2 * np.pi * tod / 288)
df["weekend"] = (df.timestamp.dt.dayofweek >= 5).astype(int)
FEATS = ["speed"] + [f"lag{k}" for k in range(1, 6)] + ["tod_sin", "tod_cos", "weekend"]
tr, va, te = split(df)
tr = tr.dropna(subset=FEATS + ["target"]).sample(1_000_000, random_state=0)
te = te.dropna(subset=FEATS + ["target"])
te["persist"] = te["speed"]
te["ridge"] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(tr[FEATS], tr.target).predict(te[FEATS])
for seed in range(3):
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True, n_iter_no_change=5,
                                                       max_iter=50, batch_size=2048, random_state=seed))
    te[f"mlp_s{seed}"] = mlp.fit(tr[FEATS], tr.target).predict(te[FEATS])
print(report(te, ["persist", "ridge", "mlp_s0", "mlp_s1", "mlp_s2"]).to_string(index=False))

# ---------- LSTM：在完整时间索引上构造窗口 ----------
arr = wide.to_numpy(dtype=np.float32)               # (T, N)，缺失为 NaN，时间索引完整
t_idx = wide.index
mu, sd = np.nanmean(arr[t_idx < TRAIN_END]), np.nanstd(arr[t_idx < TRAIN_END])  # 只用训练集
z = (arr - mu) / sd
tod_all = ((t_idx.hour * 12 + t_idx.minute // 5) / 288.0).to_numpy()


def windows(t_mask, max_n=None, seed=0):
    ts = np.where(t_mask)[0]
    ts = ts[(ts >= L - 1) & (ts + H < len(t_idx))]
    T, N = np.meshgrid(ts, np.arange(arr.shape[1]), indexing="ij")
    T, N = T.ravel(), N.ravel()
    y = arr[T + H, N]
    ok = ~np.isnan(y) & ~np.isnan(arr[T, N])
    T, N, y = T[ok], N[ok], y[ok]
    if max_n and len(T) > max_n:
        sel = np.random.default_rng(seed).choice(len(T), max_n, replace=False)
        T, N, y = T[sel], N[sel], y[sel]
    seq = np.stack([z[T - L + 1 + i, N] for i in range(L)], 1)          # (n, L)
    miss = np.isnan(seq).astype(np.float32)
    seq = np.nan_to_num(seq)                                             # 标准化后 0 = 训练集均值
    tf = np.stack([np.sin(2 * np.pi * tod_all[T - L + 1 + i]) for i in range(L)], 1)
    tc = np.stack([np.cos(2 * np.pi * tod_all[T - L + 1 + i]) for i in range(L)], 1)
    X = np.stack([seq, miss, tf, tc], -1).astype(np.float32)             # (n, L, 4)
    return torch.tensor(X), torch.tensor((y - mu) / sd), T, N


class LSTMReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


Xtr, ytr, _, _ = windows(t_idx < TRAIN_END, max_n=800_000)
Xva, yva, _, _ = windows((t_idx >= TRAIN_END) & (t_idx < VALID_END), max_n=200_000, seed=1)
Xte, yte, Tte, Nte = windows(t_idx >= VALID_END)
torch.manual_seed(0)
model = LSTMReg()
opt = torch.optim.Adam(model.parameters(), 1e-3)
best, best_state, patience = np.inf, None, 0
for epoch in range(15):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 2048):
        b = perm[i:i + 2048]
        opt.zero_grad()
        loss = nn.functional.l1_loss(model(Xtr[b]), ytr[b])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        v = float(torch.mean(torch.abs(model(Xva) - yva))) * sd
    print(f"epoch {epoch}: valid MAE={v:.3f} mph")
    if v < best - 1e-3:
        best, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 3:
            break
model.load_state_dict(best_state)
with torch.no_grad():
    pred = torch.cat([model(Xte[i:i + 65536]) for i in range(0, len(Xte), 65536)]).numpy() * sd + mu
lt = pd.DataFrame({"timestamp": t_idx[Tte], "sensor": wide.columns[Nte], "lstm": pred})
te = te.merge(lt, on=["timestamp", "sensor"], how="left")
print(report(te.dropna(subset=["lstm"]), ["persist", "ridge", "mlp_s0", "lstm"]).to_string(index=False))
