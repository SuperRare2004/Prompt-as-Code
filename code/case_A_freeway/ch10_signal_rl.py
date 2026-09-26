"""第 10 章案例：单交叉口信号控制的强化学习（教学用简化仿真器 + 表格型 Q-learning）。

交叉口：北进口为高速公路下匝道（需求高，排队过长会回溢到主线），东西为主干路，南为次要道路。
两个相位：南北放行 / 东西放行。仿真步长 1 秒，每 5 秒决策一次。
本仿真器为点排队模型，用于演示 MDP 设计的影响；工程研究应使用 SUMO 等微观仿真平台。
"""
import collections
import numpy as np

RATES = {"N": 0.22, "S": 0.05, "E": 0.14, "W": 0.14}   # 到达率（辆/秒），示意性早高峰
SAT = 0.5                                                 # 绿灯饱和放行率（辆/秒/进口）
SPILL = 30                                                # 下匝道排队超过 30 辆（约 180 m）即回溢
PHASE_APPROACH = {0: ("N", "S"), 1: ("E", "W")}


class Intersection:
    def __init__(self, safe, demand=1.0, dropout=0.0, seed=0):
        self.safe, self.demand, self.dropout = safe, demand, dropout
        self.rng = np.random.default_rng(seed)
        self.q = {a: collections.deque() for a in RATES}
        self.t, self.phase, self.green, self.trans = 0, 0, 0, 0
        self.delays, self.max_wait, self.spill_s, self.violations = [], 0, 0, 0

    def obs(self):
        """观测：各进口排队（分箱）、当前相位、绿灯时长（分箱）。dropout 模拟检测器失效。"""
        qs = []
        for a in "NSEW":
            n = len(self.q[a])
            if self.rng.random() < self.dropout:
                n = 0                                     # 检测器失效时读数为 0
            qs.append(min(n // 5, 7))
        return tuple(qs) + (self.phase, min(self.green // 10, 6))

    def step(self, switch):
        """执行 5 秒。switch=True 表示切换到另一相位。"""
        if switch:
            if self.safe and self.green < 15:
                switch = False                            # 安全版：最小绿灯 15 秒
            elif not self.safe and self.green < 15:
                self.violations += 1                      # 默认版：记录违反最小绿灯的切换
        if self.safe and self.green >= 60:
            switch = True                                 # 安全版：最大绿灯 60 秒
        if switch:
            self.phase, self.green = 1 - self.phase, 0
            self.trans = 5 if self.safe else 0            # 安全版：3 秒黄灯 + 2 秒全红
        for _ in range(5):
            self.t += 1
            for a, r in RATES.items():
                if self.rng.random() < r * self.demand:
                    self.q[a].append(self.t)
            if self.trans > 0:
                self.trans -= 1
            else:
                self.green += 1
                for a in PHASE_APPROACH[self.phase]:
                    if self.q[a] and self.rng.random() < SAT:
                        self.delays.append(self.t - self.q[a].popleft())
            self.spill_s += len(self.q["N"]) > SPILL
            self.max_wait = max([self.max_wait] + [self.t - d[0] for d in self.q.values() if d])

    def waits(self):
        return {a: (self.t - d[0] if d else 0) for a, d in self.q.items()}


def reward(env, kind):
    if kind == "ramp_only":                                # AI 的默认奖励：只看下匝道排队
        return -len(env.q["N"])
    total = sum(len(d) for d in env.q.values())            # 人定义的奖励：总排队 + 回溢与长等待惩罚
    pen = 20 * (len(env.q["N"]) > SPILL) + 10 * sum(w > 120 for w in env.waits().values())
    return -total - pen


def train(kind, safe, episodes=300, horizon=3600, seed=0):
    rng = np.random.default_rng(seed)
    Q = collections.defaultdict(lambda: np.zeros(2))
    eps, alpha, gamma = 1.0, 0.1, 0.95
    for ep in range(episodes):
        env = Intersection(safe, demand=rng.uniform(0.8, 1.2), seed=seed * 1000 + ep)
        s = env.obs()
        for _ in range(horizon // 5):
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[s]))
            env.step(bool(a))
            s2, r = env.obs(), reward(env, kind)
            Q[s][a] += alpha * (r + gamma * Q[s2].max() - Q[s][a])
            s = s2
        eps = max(0.05, eps * 0.985)
    return Q


def run(policy, safe, demand=1.0, dropout=0.0, seed=12345, horizon=3600):
    env = Intersection(safe, demand, dropout, seed)
    for k in range(horizon // 5):
        env.step(policy(env, k))
    return {"avg_delay_s": np.mean(env.delays) if env.delays else np.nan, "spill_min": env.spill_s / 60,
            "max_wait_s": env.max_wait, "left_in_queue": sum(len(d) for d in env.q.values()),
            "min_green_violations": env.violations}


def fixed_time(g_ns=40, g_ew=30):
    def pol(env, k):
        return env.green >= (g_ns if env.phase == 0 else g_ew)
    return pol


def actuated(env, k, gap_queue=1):  # gap_queue 以排队分箱（每 5 辆一箱）计
    """感应控制：当前相位仍有排队则延长（受最大绿灯约束），无排队且另一相位有车则切换。"""
    o = dict(zip("NSEW", env.obs()[:4]))                    # 与 RL 使用同样的（可能失效的）检测器读数
    cur = sum(o[a] for a in PHASE_APPROACH[env.phase])
    other = sum(o[a] for a in PHASE_APPROACH[1 - env.phase])
    return env.green >= 15 and (cur < gap_queue or env.green >= 55) and other > 0


def greedy(Q):
    return lambda env, k: bool(np.argmax(Q[env.obs()]))


if __name__ == "__main__":
    Q_default = train("ramp_only", safe=False)
    Q_human = train("human", safe=True)
    rows = {
        "fixed_time": (fixed_time(), True), "actuated": (actuated, True),
        "RL_default(ramp-only reward, no constraints)": (greedy(Q_default), False),
        "RL_human(total queue + penalties, constrained)": (greedy(Q_human), True),
    }
    for scen, kw in [("test demand x1.0", {}), ("test demand x1.3", {"demand": 1.3}),
                     ("x1.0 + 10% detector dropout", {"dropout": 0.1})]:
        print(f"\n== {scen}")
        for name, (pol, safe) in rows.items():
            res = [run(pol, safe, seed=s, **kw) for s in range(20)]
            m = {k: np.mean([r[k] for r in res]) for k in res[0]}
            print(f"{name:<48s} " + "  ".join(f"{k}={v:7.1f}" for k, v in m.items()))
