"""第 10 章各小节的演示代码：价值迭代、Q-learning、SARSA、DQN、REINFORCE。

用法：python demos.py                   # 运行全部（DQN/REINFORCE 在 CPU 上约需数分钟）
      python demos.py gridworld maze    # 只运行指定小节
CartPole 环境来自 gymnasium（pip install gymnasium），是 OpenAI Gym 的后续维护版本。
"""
import sys
import numpy as np

DEMOS = {}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]           # 上、下、左、右
NAMES = "↑↓←→"


def demo(f):
    DEMOS[f.__name__] = f
    return f


@demo
def gridworld(gamma=0.9, theta=1e-6):
    """4x4 确定性网格：目标 (0,3) 奖励 +1，陷阱 (1,3) 奖励 -1，其余每步 -0.04。"""
    H, W, goal, trap = 4, 4, (0, 3), (1, 3)
    V = np.zeros((H, W))
    for it in range(1000):
        delta = 0
        for r in range(H):
            for c in range(W):
                if (r, c) in (goal, trap):
                    continue
                vals = []
                for dr, dc in ACTIONS:
                    nr, nc = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
                    rew = 1 if (nr, nc) == goal else -1 if (nr, nc) == trap else -0.04
                    vals.append(rew + gamma * (0 if (nr, nc) in (goal, trap) else V[nr, nc]))
                delta = max(delta, abs(max(vals) - V[r, c])); V[r, c] = max(vals)
        if delta < theta:
            break
    print(f"[价值迭代] 迭代 {it + 1} 次收敛；V(3,0)={V[3, 0]:.3f}")
    pol = []
    for r in range(H):
        row = ""
        for c in range(W):
            if (r, c) == goal: row += "G"; continue
            if (r, c) == trap: row += "X"; continue
            q = []
            for dr, dc in ACTIONS:
                nr, nc = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
                rew = 1 if (nr, nc) == goal else -1 if (nr, nc) == trap else -0.04
                q.append(rew + gamma * (0 if (nr, nc) in (goal, trap) else V[nr, nc]))
            row += NAMES[int(np.argmax(q))]
        pol.append(row)
    print("[价值迭代] 最优策略：\n  " + "\n  ".join(pol))


MAZE = ["S.W.", "..W.", "....", "W..G"]


def maze_step(pos, a):
    r, c = pos[0] + ACTIONS[a][0], pos[1] + ACTIONS[a][1]
    if not (0 <= r < 4 and 0 <= c < 4):
        return pos, -1, False
    if MAZE[r][c] == "W":
        return pos, -100, False                        # 撞墙：惩罚并留在原位
    if MAZE[r][c] == "G":
        return (r, c), 100, True
    return (r, c), -1, False


def run_td(step, start, n_states_shape, algo="q", episodes=500, alpha=0.1, gamma=0.95, eps=0.1, seed=0, max_steps=200):
    rng = np.random.default_rng(seed)
    Q = np.zeros(n_states_shape + (4,))
    returns = []
    pick = lambda s: int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s]))
    for _ in range(episodes):
        s, a, G = start, None, 0
        a = pick(s)
        for _ in range(max_steps):
            s2, r, done = step(s, a)
            a2 = pick(s2)
            target = r + (0 if done else gamma * (Q[s2].max() if algo == "q" else Q[s2][a2]))
            Q[s][a] += alpha * (target - Q[s][a])
            s, a, G = s2, a2, G + r
            if done:
                break
        returns.append(G)
    return Q, np.array(returns)


def greedy_path(Q, step, start, max_steps=50):
    s, path = start, [start]
    for _ in range(max_steps):
        s, _, done = step(s, int(np.argmax(Q[s])))
        path.append(s)
        if done:
            break
    return path


@demo
def maze():
    Q, ret = run_td(maze_step, (0, 0), (4, 4), "q")
    path = greedy_path(Q, maze_step, (0, 0))
    print(f"[Q-learning 迷宫] 前 50 回合平均回报={ret[:50].mean():.1f}，最后 50 回合={ret[-50:].mean():.1f}")
    print(f"[Q-learning 迷宫] 贪心路径（{len(path) - 1} 步）：{path}")


def cliff_step(pos, a):
    """4x12 悬崖行走：起点 (3,0)，终点 (3,11)，底行中间为悬崖（-100 并回到起点）。"""
    r, c = min(max(pos[0] + ACTIONS[a][0], 0), 3), min(max(pos[1] + ACTIONS[a][1], 0), 11)
    if r == 3 and 1 <= c <= 10:
        return (3, 0), -100, False
    return (r, c), -1, (r, c) == (3, 11)


@demo
def sarsa_vs_q():
    for algo in ("q", "sarsa"):
        rs = [run_td(cliff_step, (3, 0), (4, 12), algo, episodes=500, seed=s)[1][-100:].mean() for s in range(10)]
        Q, _ = run_td(cliff_step, (3, 0), (4, 12), algo, episodes=500, seed=0)
        path = greedy_path(Q, cliff_step, (3, 0))
        rows = sorted({p[0] for p in path})
        print(f"[{algo.upper():<5s}] 训练中（ε=0.1）最后 100 回合平均回报={np.mean(rs):.1f}（10 个种子）；贪心路径 {len(path) - 1} 步，经过的行={rows}")


@demo
def dqn(episodes=300, seed=0):
    import gymnasium as gym
    import torch
    from torch import nn
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    tgt = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    buf, lens, steps = [], [], 0                          # 经验回放缓冲区
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep); done, L = False, 0
        eps = max(0.05, 1 - ep / 150)
        while not done:
            with torch.no_grad():
                a = int(rng.integers(2)) if rng.random() < eps else int(net(torch.tensor(s)).argmax())
            s2, r, term, trunc, _ = env.step(a); done = term or trunc
            buf.append((s, a, r, s2, float(term))); buf = buf[-20000:]
            s, L, steps = s2, L + 1, steps + 1
            if len(buf) >= 1000:
                idx = rng.integers(len(buf), size=64)
                S, A, R, S2, D = (torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*[buf[i] for i in idx]))
                q = net(S).gather(1, A.long()[:, None]).squeeze()
                with torch.no_grad():
                    y = R + 0.99 * (1 - D) * tgt(S2).max(1).values        # 目标网络
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward(); opt.step()
            if steps % 500 == 0:
                tgt.load_state_dict(net.state_dict())
        lens.append(L)
    print(f"[DQN CartPole] 回合长度：前 50 回合均值={np.mean(lens[:50]):.1f}，最后 50 回合均值={np.mean(lens[-50:]):.1f}，最大={max(lens)}（上限 500）")
    return lens


@demo
def reinforce(episodes=600, seed=0):
    import gymnasium as gym
    import torch
    from torch import nn
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    pi = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.Adam(pi.parameters(), 1e-2)
    lens = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep); done, logps, rews = False, [], []
        while not done:
            dist = torch.distributions.Categorical(logits=pi(torch.tensor(s)))
            a = dist.sample(); logps.append(dist.log_prob(a))
            s, r, term, trunc, _ = env.step(int(a)); done = term or trunc; rews.append(r)
        G, rets = 0, []
        for r in reversed(rews):
            G = r + 0.99 * G; rets.insert(0, G)
        rets = torch.tensor(rets); rets = (rets - rets.mean()) / (rets.std() + 1e-8)   # 标准化以降低方差
        loss = -(torch.stack(logps) * rets).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        lens.append(len(rews))
    lens = np.array(lens)
    print(f"[REINFORCE CartPole] 回合长度：前 50 回合均值={lens[:50].mean():.1f}，最后 50 回合均值={lens[-50:].mean():.1f}，"
          f"最后 100 回合的标准差={lens[-100:].std():.1f}")
    return lens


if __name__ == "__main__":
    for name in (sys.argv[1:] or DEMOS):
        DEMOS[name]()
