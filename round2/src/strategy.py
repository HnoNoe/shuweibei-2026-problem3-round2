"""第 6 步：两阶段随机调度引擎 + 基线梯度。

设计文档：round2/docs/strategy_design.md（先读那个）。
策略对外的协议与 sampler.run_trial 配套：reset(trial_cfg) + decide(snapshot)。

引擎结构：
- TwoStagePolicy：主策略，L1~L5 五个杠杆开关（CONFIGS 预设消融配置）。
- 基线 B1 EDD / B2 WSPT-mean / B3 WSPT-q90：BaselinePolicy，规则做对了的版本
  （预算/H 合法性预检，但无随机守卫/无取舍/无前瞻/无序列/无更新器）。

合规红线：decide 只许用 snapshot 里已有的字段 + 第 5 步时长模型；
b 恒为 B_MIN（0.2，b_dominance.md 钉死）；永不调用 review（no_review.md 钉死）。
"""

import math

import numpy as np

from common import (B_MIN, CHANGEOVER, FEE_DIST_PER_UNIT, FEE_FIXED,
                    FEE_RESERVE_PER_S, MAX_ACTIONS, zone_dist)
from duration_model3 import DurationModel, MixedLikelihoodUpdater

# ---------------- 消融配置（第 7 步矩阵，见设计文档 §4）----------------
CONFIGS = {
    "C0_bare":   dict(l1=False, l2=False, l3=False, l4=False, l5=False),
    "C1_atc":    dict(l1=True,  l2=False, l3=False, l4=False, l5=False),
    "C2_guard":  dict(l1=True,  l2=True,  l3=False, l4=False, l5=False),
    "C3_budget": dict(l1=True,  l2=False, l3=True,  l4=False, l5=False),
    "C4_roll":   dict(l1=True,  l2=False, l3=False, l4=True,  l5=False),
    "C5_online": dict(l1=True,  l2=True,  l3=True,  l4=True,  l5=False),
    "C6_full":   dict(l1=True,  l2=True,  l3=True,  l4=True,  l5=True),
}

HYPER = dict(kappa=2.0,          # ATC 折中参数（文献标准位 2~3）
             roll_K=20,          # rollout 情景数
             roll_m=3,           # rollout 候选数
             ls_G=10,            # L5 离线城市数
             ls_max_steps=60)    # L5 局部搜索步数上限

EPS = 1e-9
# rollout 里模拟违规的罚（远低于一次违规真实代价：trial 直接终止）
VIOLATE_PENALTY = 1.0


# =====================================================================
# 公共零件
# =====================================================================

def fee_of(D, b=B_MIN):
    """单作业费用（common.py 公式的副本，供规划用）。"""
    return FEE_FIXED + FEE_RESERVE_PER_S * b + FEE_DIST_PER_UNIT * D


class FastSim:
    """内部快模拟器：逐字节复刻 Trial.execute/wait 的数值语义，无副作用。

    只用于 L4 rollout 与 L5 序列评价。任何语义改动必须与 simulator.py 同步
    （test_strategy.py 的退化与回放断言盯这件事）。
    """

    def __init__(self, snap, durations, alpha, beta, H, B):
        self.jobs = snap["jobs"]
        self.dur = durations            # {jid: d}
        self.alpha, self.beta = alpha, beta
        self.H, self.B = H, B
        self.t = snap["t"]
        self.zone = snap["zone"]
        self.cost = snap["cost"]
        self.completed = set(snap["completed"])
        self.executed = set(snap["executed"])
        self.comp_t = dict(snap["completion_time"])
        self.violated = False

    def _p(self, jid):
        return self.alpha * self.dur[jid] + self.beta

    def exec_ok(self, jid, b=B_MIN):
        """试算 execute 的全部合法性 + 状态推进。违规返回 False 且标记。"""
        if jid in self.executed or jid not in self.jobs:
            return self._viol()
        if not (math.isfinite(b) and b >= B_MIN - EPS):
            return self._viol()
        D = zone_dist(self.zone, self.jobs[jid]["zone"])
        s = self.t + D + CHANGEOVER
        fee = fee_of(D, b)
        if self.cost + fee > self.B + EPS:
            return self._viol()
        if s + b > self.H + EPS:
            return self._viol()
        p = self._p(jid)
        C = s + p
        self.cost += fee
        self.executed.add(jid)
        if C <= self.H + EPS:
            self.completed.add(jid)
            self.comp_t[jid] = C
            nxt = s + max(p, b)
        else:
            nxt = self.H
        self.t = min(nxt, self.H)
        self.zone = self.jobs[jid]["zone"]
        return True

    def _viol(self):
        self.violated = True
        return False

    def wait_to(self, t_target):
        self.t = min(t_target, self.H)

    def weighted_on_time(self):
        tot_w = sum(i["weight"] for i in self.jobs.values()) or 1.0
        wt = sum(i["weight"] for jid, i in self.jobs.items()
                 if jid in self.completed
                 and self.comp_t[jid] <= i["due_s"] + EPS)
        return wt / tot_w


def draw_scenarios(model_params, jobs, unexec, K, rng):
    """CRN 情景矩阵：scen[k][jid] = d 抽样。同序号 k 跨候选/跨调用复用。"""
    scen = []
    for _ in range(K):
        row = {}
        for jid in unexec:
            mu, sigma = model_params[jobs[jid]["family"]]
            row[jid] = float(np.exp(rng.normal(mu, sigma)))
        scen.append(row)
    return scen


def greedy_complete(sim, scorer):
    """余程贪心补完（rollout 的基策略）：每步在合法候选里按 scorer 最高执行。"""
    guard = MAX_ACTIONS * 2
    while guard > 0:
        guard -= 1
        cands = []
        for jid, info in sim.jobs.items():
            if jid in sim.executed:
                continue
            if sim.t + EPS < info["release_s"]:
                continue
            if any(p not in sim.completed for p in info["predecessors"]):
                continue
            D = zone_dist(sim.zone, info["zone"])
            if sim.cost + fee_of(D) > sim.B + EPS:
                continue
            if sim.t + D + CHANGEOVER + B_MIN > sim.H + EPS:
                continue
            cands.append(jid)
        if not cands:
            future = [i["release_s"] for j, i in sim.jobs.items()
                      if j not in sim.executed]
            if not future:
                break
            sim.wait_to(min(future))
            if sim.t >= sim.H - EPS:
                break
            continue
        best = max(cands, key=lambda j: scorer(sim, j))
        if not sim.exec_ok(best):
            break
    return sim


# =====================================================================
# 主策略
# =====================================================================

class TwoStagePolicy:
    """两阶段策略引擎。config 从 CONFIGS 取，hyper 从 HYPER 取（可覆盖）。"""

    def __init__(self, config=None, hyper=None, model=None):
        self.cfg = dict(CONFIGS["C6_full"] if config is None else
                        (CONFIGS[config] if isinstance(config, str) else config))
        self.h = dict(HYPER, **(hyper or {}))
        self.model = model or DurationModel.from_gofit()

    # ---- 离线层 ----
    def reset(self, trial):
        self.trial = dict(trial)
        self.alpha, self.beta = trial["alpha"], trial["beta"]
        self.H, self.B = trial["H"], trial["B"]
        self.updater = MixedLikelihoodUpdater(self.model.params)
        self._exec_moments = {}       # jid -> 我方发起 execute 时的 (t, zone)
        self._decision_idx = 0
        self._jobs_cache = {}
        self._seq = None
        self._fed = set()
        # 注意：trial_cfg 不含 job 明细，离线序列延迟到首个 snapshot 再规划

    def _mean_params(self):
        return self.updater.params if self.updater.n_obs else self.model.params

    def _d_hat(self, family, q=None):
        mu, sigma = self._mean_params()[family]
        if q is None:
            return math.exp(mu + 0.5 * sigma ** 2)      # 均值
        return math.exp(mu + sigma * _z(q))             # 分位数

    def _est(self, jid, info, q=None):
        """点估计钩子。q 给定时取 family 分位数；q=None 时默认中位数——
        依据：消融 B3−B2 显著为负（高估时长有害），对数正态右偏下
        e^μ（中位数）比 e^{μ+σ²/2}（均值）稳健。子类可覆盖（如 GBM）。"""
        if q is not None:
            return self._d_hat(info["family"], q=q)
        mu, _ = self._mean_params()[info["family"]]
        return math.exp(mu)

    def _offline_sequence(self):
        """期望时长确定性扫描得初解，G 情景局部搜索改进（L5）。

        纪律：初解与全部邻居必须在同一批情景上评价——跨情景集比较
        会把抽样噪声当改进信号（2026-11-23 自检 6 抓到的真 bug）。
        """
        snap0 = self._toy_snapshot()
        init = self._scan_sequence(snap0, mean_est=True)
        if not init:
            return init
        rng = np.random.default_rng(self._seed("l5"))
        jobs = snap0["jobs"]
        params = self.model.params
        scen = [{j: float(np.exp(rng.normal(*params[info["family"]])))
                 for j, info in jobs.items()} for _ in range(self.h["ls_G"])]
        best, best_ev = init, self._sequence_ev(init, snap0, scen)
        for _ in range(self.h["ls_max_steps"]):
            neigh = self._neighbors(best)
            improved = False
            for cand in neigh:
                ev = self._sequence_ev(cand, snap0, scen)
                if ev > best_ev + 1e-12:
                    best, best_ev = cand, ev
                    improved = True
                    break
            if not improved:
                break
        return best

    def _toy_snapshot(self):
        """reset 时刻（t=0, Z0, cost=0）的合成 snapshot，离线规划用。"""
        jobs = {j: dict(info) for j, info in self._trial_jobs().items()}
        return dict(t=0.0, zone="Z0", cost=0.0, H=self.H, B=self.B,
                    alpha=self.alpha, beta=self.beta, jobs=jobs,
                    executed=set(), completed=set(), completion_time={})

    def _trial_jobs(self):
        # trial_cfg 里没有 jobs 明细；decide 首次 snapshot 才拿到。
        # 离线层需要 job 表：从 sampler 层传入（见 reset 的调用方），
        # 这里延迟到首个 snapshot 再补做离线规划。
        return getattr(self, "_jobs_cache", {})

    def _seed(self, tag):
        tb = self.trial
        return abs(hash((tag, tb["case"], tb["seed"], tb["alpha"], tb["beta"]))) % (2 ** 32)

    def _scan_sequence(self, snap, mean_est=True):
        """确定性期望扫描：重复选当前合法候选中打分最高者，得一个序列。"""
        sim = FastSim(snap, {}, self.alpha, self.beta, self.H, self.B)
        dur = {j: self.alpha * self._est(j, snap["jobs"][j]) + self.beta
               for j in snap["jobs"]}
        sim.dur = dur
        seq = []
        guard = MAX_ACTIONS * 2
        while guard > 0:
            guard -= 1
            cands = [j for j, info in sim.jobs.items()
                     if j not in sim.executed
                     and sim.t + EPS >= info["release_s"]
                     and all(p in sim.completed for p in info["predecessors"])
                     and sim.cost + fee_of(zone_dist(sim.zone, info["zone"])) <= sim.B + EPS
                     and sim.t + zone_dist(sim.zone, info["zone"]) + CHANGEOVER + B_MIN <= sim.H + EPS]
            if not cands:
                future = [i["release_s"] for j, i in sim.jobs.items()
                          if j not in sim.executed]
                if not future:
                    break
                sim.wait_to(min(future))
                if sim.t >= sim.H - EPS:
                    break
                continue
            best = max(cands, key=lambda j: self._score(sim, j, snap["jobs"]))
            if not sim.exec_ok(best):
                break
            seq.append(best)
        return seq

    def _neighbors(self, seq):
        """相邻交换 + 前插。"""
        out = []
        for i in range(len(seq) - 1):
            s2 = seq[:]
            s2[i], s2[i + 1] = s2[i + 1], s2[i]
            out.append(s2)
        for i in range(1, len(seq)):
            for j in range(0, i - 1):
                s2 = seq[:]
                x = s2.pop(i)
                s2.insert(j, x)
                out.append(s2)
        return out

    def _sequence_ev(self, seq, snap, scen=None, rng=None):
        """固定序列在 G 情景下的期望加权准时率（first-improvement 评价）。

        scen 给定则用之（搜索内必须同一批情景）；否则用 seed='l5ev' 的
        固定流生成——两次不带 scen 的调用评价的是同一批情景，可比。
        """
        if scen is None:
            if rng is None:
                rng = np.random.default_rng(self._seed("l5ev"))
            jobs_ = snap["jobs"]
            params = self.model.params
            scen = [{j: float(np.exp(rng.normal(*params[i["family"]])))
                     for j, i in jobs_.items()} for _ in range(self.h["ls_G"])]
        jobs = snap["jobs"]
        vals = []
        for durs in scen:
            sim = FastSim(snap, durs, self.alpha, self.beta, self.H, self.B)
            ok = True
            for jid in seq:
                if jid in sim.executed:
                    continue
                info = jobs[jid]
                # 跟随固定序列：未释放则等到释放
                if sim.t + EPS < info["release_s"]:
                    sim.wait_to(info["release_s"])
                if sim.t >= sim.H - EPS:
                    break
                if all(p in sim.completed for p in info["predecessors"]) \
                        and sim.exec_ok(jid):
                    continue
                # 前序未完成或被预算/H 卡死：跳过该作业（recourse 语义）
                if sim.violated:
                    ok = False
                    break
            if not ok:
                vals.append(0.0)
                continue
            vals.append(sim.weighted_on_time())
        return sum(vals) / len(vals)

    # ---- 在线层 ----
    def _score(self, sim, jid, jobs):
        """L1 ATC（含 L2 软罚项）或退化 WSPT。sim 提供 t/zone/H。"""
        info = jobs[jid]
        D = zone_dist(sim.zone, info["zone"])
        s = sim.t + D + CHANGEOVER
        p_hat = sim.dur.get(jid)
        if p_hat is None:
            p_hat = self.alpha * self._est(jid, info) + self.beta
        base = info["weight"] / max(p_hat, EPS)
        rest = [j for j in jobs if j not in sim.executed]
        p_bar = (sum(self.alpha * self._est(j, jobs[j]) + self.beta
                     for j in rest) / len(rest)) if rest else p_hat
        if not self.cfg["l1"]:
            score = base
        else:
            slack = info["due_s"] - (s + p_hat)
            score = base * math.exp(-max(slack, 0.0) /
                                    (self.h["kappa"] * max(p_bar, EPS)))
        if self.cfg["l2"]:
            # 软罚项（A1 改进，2026-11-23）：越过 q95 线的程度以指数衰减进
            # 打分——决策与 rollout 模拟共用同一 scorer，天然一致；
            # 替代旧版硬过滤放松链（其 C2−C1 = -0.0047 显著为负）。
            mu, sig = self._mean_params()[info["family"]]
            p95 = self.alpha * math.exp(mu + sig * _z95()) + self.beta
            over = s + p95 - sim.H
            if over > 0:
                score *= math.exp(-over / (self.h["kappa"] * max(p_bar, EPS)))
        return score

    def _seq_score(self, sim, jid, jobs):
        """L5 模式打分：先验序列位置为主键（越靠前分越高），ATC 只做并列打破。"""
        if self._seq and jid in self._seq:
            return -self._seq.index(jid) + 1e-6 * self._score(sim, jid, jobs)
        return -1e6 + 1e-6 * self._score(sim, jid, jobs)   # 不在序列内排最后

    def _apply_feedback(self, snap):
        """第 5 步通道：从 snapshot 反推完成/删失反馈喂更新器。"""
        jobs = snap["jobs"]
        for jid in snap["executed"]:
            if jid in self._fed or jid not in self._exec_moments:
                continue  # 已喂过 / 没记录到动作时刻（不该发生），保守跳过
            self._fed.add(jid)
            t0, zone0 = self._exec_moments[jid]
            D = zone_dist(zone0, jobs[jid]["zone"])
            if jid in snap["completed"]:
                d_true = snap["completion_time"][jid] - (t0 + D + CHANGEOVER)
                self.updater.add_completed(jobs[jid]["family"], max(d_true, 1e-6))
            else:
                thr = self.H - (t0 + D + CHANGEOVER)
                if thr > 0:
                    self.updater.add_censored(jobs[jid]["family"], thr)
        self.updater.fit()

    def decide(self, snap):
        jobs = snap["jobs"]
        if not self._trial_jobs():
            self._jobs_cache = jobs
            if self.cfg["l5"] and self._seq is None:
                self._seq = self._offline_sequence()
        if self.h.get("feedback", True):
            self._apply_feedback(snap)
        self._decision_idx += 1

        alpha, beta = snap["alpha"], snap["beta"]
        t = snap["t"]

        # 候选集：已释放 ∧ 前序完 ∧ 未执行
        cands = []
        for jid, info in jobs.items():
            if jid in snap["executed"]:
                continue
            if t + EPS < info["release_s"]:
                continue
            if any(p not in snap["completed"] for p in info["predecessors"]):
                continue
            cands.append(jid)

        # L3：预算取舍 —— 剩余预算装不下所有剩余作业时按价值密度筛选
        if self.cfg["l3"]:
            cands = self._budget_filter(snap, cands)

        # 打分（L5 开=序列位置，否则 ATC/WSPT）
        scorer = self._seq_score if (self.cfg["l5"] and self._seq) else self._score
        sim_now = FastSim(snap, {}, alpha, beta, snap["H"], snap["B"])
        sim_now.dur = {j: alpha * self._est(j, jobs[j]) + beta
                       for j in cands}
        feasible = []          # 过 预算+H 预检的
        for jid in cands:
            info = jobs[jid]
            D = zone_dist(snap["zone"], info["zone"])
            s = t + D + CHANGEOVER
            if snap["cost"] + fee_of(D) > snap["B"] + EPS:
                continue
            if s + B_MIN > snap["H"] + EPS:
                continue
            feasible.append(jid)
        if not feasible:
            return self._wait_or_stop(snap)

        # L2 已改为打分内软罚项（见 _score），此处不再做候选硬过滤

        # L4 rollout 前瞻
        if self.cfg["l4"] and feasible:
            ranked = sorted(feasible, key=lambda j: scorer(sim_now, j, jobs),
                            reverse=True)
            top = ranked[:self.h["roll_m"]]
            rng = np.random.default_rng(self._seed(f"roll{self._decision_idx}"))
            params = self._mean_params()
            unexec = [j for j in jobs if j not in snap["executed"]]
            scen = draw_scenarios(params, jobs, unexec, self.h["roll_K"], rng)
            best_j, best_q = None, -float("inf")
            def gs(sim_, j_):
                return scorer(sim_, j_, sim_.jobs)
            for jid in top:
                tot = 0.0
                for k in range(self.h["roll_K"]):
                    sim = FastSim(snap, scen[k], alpha, beta, snap["H"], snap["B"])
                    if not sim.exec_ok(jid):
                        tot -= VIOLATE_PENALTY
                        continue
                    greedy_complete(sim, gs)
                    if sim.violated:
                        tot -= VIOLATE_PENALTY
                        continue
                    tot += sim.weighted_on_time()
                q_val = tot / self.h["roll_K"]
                if q_val > best_q + 1e-12:
                    best_j, best_q = jid, q_val
            if best_j is not None:
                chosen = best_j
            else:
                chosen = max(feasible, key=lambda j: scorer(sim_now, j, jobs))
        else:
            chosen = max(feasible, key=lambda j: scorer(sim_now, j, jobs))

        self._exec_moments[chosen] = (t, snap["zone"])
        return ("execute", chosen, B_MIN)

    def _budget_filter(self, snap, cands):
        """价值密度：剩余预算装不下全部剩余作业时，只保留密度最高且累计费用装得下的。"""
        info = snap["jobs"]
        items = []
        for jid in cands:
            D = zone_dist(snap["zone"], info[jid]["zone"])
            items.append((info[jid]["weight"] / max(fee_of(D), EPS), jid,
                          fee_of(D)))
        items.sort(reverse=True)
        keep, budget = [], snap["B"] - snap["cost"]
        for _, jid, fee in items:
            if fee <= budget + EPS:
                keep.append(jid)
                budget -= fee
        return keep

    def _wait_or_stop(self, snap):
        jobs = snap["jobs"]
        pending = [j for j in jobs if j not in snap["executed"]]
        if not pending:
            return None  # stop
        # 等下一个可能解锁的时刻：未释放作业的最小 release，
        # 或前序未完成的作业的前序预计完成时刻（取 release 下界，保守）
        future = [jobs[j]["release_s"] for j in pending
                  if snap["t"] + EPS < jobs[j]["release_s"]]
        if not future:
            return None
        w = min(future) - snap["t"]
        if w <= EPS or snap["t"] + w > snap["H"] + EPS:
            return None
        return ("wait", w)


def _z(q):
    from scipy.stats import norm
    return norm.ppf(q)


_Z95 = None  # 惰性缓存：打分热路径不许每次调 scipy


def _z95():
    global _Z95
    if _Z95 is None:
        _Z95 = _z(0.95)
    return _Z95


# =====================================================================
# 基线（规则做对了的版本：合法性预检有，随机层全无）
# =====================================================================

class BaselinePolicy:
    """B1 EDD / B2 WSPT-mean / B3 WSPT-q90。无更新器、无守卫、无前瞻、无序列。"""

    def __init__(self, rule="wspt", est="mean", model=None):
        assert rule in ("edd", "wspt")
        assert est in ("mean", "q90", "median")
        self.rule, self.est = rule, est
        self.model = model or DurationModel.from_gofit()

    def reset(self, trial):
        self.trial = dict(trial)
        self.alpha, self.beta = trial["alpha"], trial["beta"]

    def _d_hat(self, family):
        mu, sigma = self.model.params[family]
        if self.est == "mean":
            return math.exp(mu + 0.5 * sigma ** 2)
        if self.est == "median":
            return math.exp(mu)
        return math.exp(mu + sigma * _z(0.90))

    def decide(self, snap):
        jobs = snap["jobs"]
        t = snap["t"]
        cands = []
        for jid, info in jobs.items():
            if jid in snap["executed"]:
                continue
            if t + EPS < info["release_s"]:
                continue
            if any(p not in snap["completed"] for p in info["predecessors"]):
                continue
            D = zone_dist(snap["zone"], info["zone"])
            if snap["cost"] + fee_of(D) > snap["B"] + EPS:
                continue
            if t + D + CHANGEOVER + B_MIN > snap["H"] + EPS:
                continue
            cands.append(jid)
        if not cands:
            pending = [j for j in jobs if j not in snap["executed"]]
            future = [jobs[j]["release_s"] for j in pending
                      if t + EPS < jobs[j]["release_s"]]
            if not future:
                return None
            w = min(future) - t
            if w <= EPS or t + w > snap["H"] + EPS:
                return None
            return ("wait", w)
        if self.rule == "edd":
            chosen = min(cands, key=lambda j: (jobs[j]["due_s"], j))
        else:
            chosen = max(cands,
                         key=lambda j: jobs[j]["weight"] /
                         max(self.alpha * self._d_hat(jobs[j]["family"]) + self.beta, EPS))
        return ("execute", chosen, B_MIN)


def baseline(name, model=None):
    """B1_EDD / B2_WSPT_MEAN / B2b_WSPT_MEDIAN / B3_WSPT_Q90 工厂。"""
    table = {"B1_EDD": ("edd", "mean"),
             "B2_WSPT_MEAN": ("wspt", "mean"),
             "B2b_WSPT_MEDIAN": ("wspt", "median"),
             "B3_WSPT_Q90": ("wspt", "q90")}
    rule, est = table[name]
    return BaselinePolicy(rule, est, model=model)
