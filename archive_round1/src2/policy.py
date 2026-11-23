"""调度策略（重做版核心）。

目标（题目允许自选并论证）：最大化期望加权准时率（weighted_on_time_rate），
同时报告全部公共指标与权衡。

策略构成（全部决策只用公开信息）：
1. 预留决策：b=0.2（最小预留）。解析依据：C=s+p 与 b 无关、设备占用 s+max(p,b)、
   费用随 b 递增，且 p>=0.9>0.2 恒成立 ⇒ 小预留弱占优于大预留（单测试验证）。
2. 审查决策：不买。依据：前缀特征对 d 无预测力（分组 CV R²<0），
   审查的期望信息价值≈0，而成本=1 单位 + 占用设备 1s。
3. 排序/取舍：
   - ATC-travel 指数  I_j = (w_j/peff_j)·exp(−max(S_j,0)/(κ·p̄))
     peff_j = E[p_j] + D_j + 0.5（有效加工含移动与换型）
   - 截断防护：仅当 s_j + Q_q(p_j) ≤ H 才执行（q 为可调分位）
   - 预算内取舍：剩余预算不足以覆盖全部剩余作业时，按价值密度舍尾
   - Rollout 加强版：对 ATC 前 k 候选做一步展开 + SAA 情景评估选最优首作业
4. 等待/停止：无可行作业且有未来释放则等待到最近释放；否则停止。
"""
import math
import numpy as np
from common import (ZONES, zone_dist, FEE_FIXED, FEE_RESERVE_PER_S,
                    FEE_DIST_PER_UNIT, CHANGEOVER, B_MIN)

B = B_MIN  # 固定最小预留


# ---------- 公共工具 ----------
def fee_of(D, b=B):
    return FEE_FIXED + FEE_RESERVE_PER_S * b + FEE_DIST_PER_UNIT * D


class BasePolicy:
    def reset(self, trial_info):
        self.info = trial_info
        self.deferred = set()      # 被预算规则放弃的作业
        # 固定种子：情景采样不依赖 trial 种子（赛题规定种子不是策略输入）
        self.rng = np.random.default_rng(20261121)

    def decide(self, snap):
        raise NotImplementedError


# ---------- 规则策略 ----------
class DispatchPolicy(BasePolicy):
    """ATC-travel + 截断防护 + 预算取舍。"""

    def __init__(self, model, kappa=2.0, guard_q=0.90, selection=True,
                 reserve_q=None):
        self.model = model
        self.kappa = kappa
        self.guard_q = guard_q
        self.selection = selection
        self.reserve_q = reserve_q   # None=最小预留0.2；否则 b=Q_q(p)

    def _reserve_b(self, snap, jid):
        if self.reserve_q is None:
            return B
        q = self.reserve_q
        job = snap["jobs"][jid]
        b = self.model.p_quantile(job["family"], snap["alpha"], snap["beta"], q)
        D = zone_dist(snap["zone"], job["zone"])
        max_b = snap["H"] - (snap["t"] + D + CHANGEOVER)
        return float(min(max(b, B), max(max_b, B)))

    # 候选过滤：释放、前序、预算、s+b<=H、截断防护、未被放弃
    def _candidates(self, snap, exclude_guard=False):
        out = []
        alpha, beta = snap["alpha"], snap["beta"]
        for jid, job in snap["jobs"].items():
            if jid in snap["executed"] or jid in self.deferred:
                continue
            if snap["t"] < job["release_s"] - 1e-9:
                continue
            if any(p not in snap["completed"] for p in job["predecessors"]):
                continue
            D = zone_dist(snap["zone"], job["zone"])
            s = snap["t"] + D + CHANGEOVER
            b = B if exclude_guard else self._reserve_b(snap, jid)
            fee = fee_of(D, b)
            if snap["cost"] + fee > snap["B"] + 1e-9:
                continue
            if s + b > snap["H"] + 1e-9:
                continue
            if not exclude_guard:
                q = self.guard_q
                p_q = self.model.p_quantile(job["family"], alpha, beta, q)
                if s + p_q > snap["H"] + 1e-9:
                    continue
            out.append((jid, D, s, fee, b))
        return out

    def _index(self, snap, jid, D, s):
        job = snap["jobs"][jid]
        p_mean = self.model.p_mean(job["family"], snap["alpha"], snap["beta"])
        peff = p_mean + D + CHANGEOVER
        slack = job["due_s"] - (s + p_mean)
        return (job["weight"] / max(peff, 0.3)) * math.exp(
            -max(slack, 0.0) / max(self.kappa * peff, 0.3))

    def _apply_selection(self, snap, cands):
        """预算不足覆盖全部剩余作业时，按价值密度舍去尾部的作业。"""
        if not self.selection:
            return cands
        remaining = [j for j in snap["jobs"] if j not in snap["executed"]
                     and j not in self.deferred]
        # 最低费用估计（当前位置、b=0.2）
        min_fees = {}
        for j in remaining:
            D = zone_dist(snap["zone"], snap["jobs"][j]["zone"])
            min_fees[j] = fee_of(D)
        total_min = sum(min_fees.values())
        if snap["cost"] + total_min <= snap["B"] + 1e-9:
            return cands          # 预算够，无需取舍
        # 按 权重/最低费用 升序舍尾（有后继的作业加权提升）
        def density(j):
            unlock = sum(1 for x in snap["jobs"].values()
                         if j in x["predecessors"])
            return (snap["jobs"][j]["weight"] * (1 + 0.5 * unlock)
                    ) / min_fees[j]
        order = sorted(remaining, key=density)
        keep = set(remaining)
        c = snap["cost"]
        for j in order:
            need = sum(min_fees[x] for x in keep)
            if c + need <= snap["B"] + 1e-9:
                break
            keep.discard(j)
            self.deferred.add(j)
        return [c0 for c0 in cands if c0[0] in keep]

    def decide(self, snap):
        cands = self._apply_selection(snap, self._candidates(snap))
        if cands:
            scored = [(self._index(snap, jid, D, s), jid, b)
                      for jid, D, s, fee, b in cands]
            scored.sort(reverse=True)
            return ("execute", scored[0][1], scored[0][2])
        # 无候选：尝试无防护候选（截断风险作业）——仅当什么都不做也只能干等
        loose = [c for c in self._candidates(snap, exclude_guard=True)
                 if c[0] not in self.deferred]
        if loose:
            scored = [(self._index(snap, jid, D, s), jid, b)
                      for jid, D, s, fee, b in loose]
            scored.sort(reverse=True)
            return ("execute", scored[0][1], scored[0][2])
        # 等待到最近释放
        future = [j for j, job in snap["jobs"].items()
                  if j not in snap["executed"] and j not in self.deferred]
        if future:
            nxt = min(snap["jobs"][j]["release_s"] for j in future)
            if snap["t"] < nxt <= snap["H"]:
                return ("wait", nxt - snap["t"])
        return ("stop",)


class BaselineWSPT(BasePolicy):
    """复现基线策略 A：score = w/(E[p]+D+0.5)，无防护无取舍，b=0.2。"""

    def __init__(self, model):
        self.model = model

    def decide(self, snap):
        cands = []
        for jid, job in snap["jobs"].items():
            if jid in snap["executed"]:
                continue
            if snap["t"] < job["release_s"] - 1e-9:
                continue
            if any(p not in snap["completed"] for p in job["predecessors"]):
                continue
            D = zone_dist(snap["zone"], job["zone"])
            s = snap["t"] + D + CHANGEOVER
            if snap["cost"] + fee_of(D) > snap["B"] + 1e-9:
                continue
            if s + B > snap["H"] + 1e-9:
                continue
            p_mean = self.model.p_mean(job["family"], snap["alpha"],
                                       snap["beta"])
            score = job["weight"] / max(p_mean + D + CHANGEOVER, 0.3)
            cands.append((score, jid))
        if cands:
            cands.sort(reverse=True)
            return ("execute", cands[0][1], B)
        future = [j for j in snap["jobs"] if j not in snap["executed"]]
        if future:
            nxt = min(snap["jobs"][j]["release_s"] for j in future)
            if snap["t"] < nxt <= snap["H"]:
                return ("wait", nxt - snap["t"])
        return ("stop",)


# ---------- 序列评估器（SAA 用）----------
def eval_sequence(plan, snap, d_scen, alpha, beta):
    """在单一时长情景下评估一个固定作业序列的加权准时率。

    按序贪心执行：轮到的作业若未释放则等待；预算不足或 s+b>H 则跳过。
    返回加权准时权重和（未除 W，由调用方除）。
    """
    t, zone, cost = snap["t"], snap["zone"], snap["cost"]
    H = snap["H"]
    jobs = snap["jobs"]
    done = set(snap["completed"])
    gain = 0.0
    for jid in plan:
        if jid in done:
            continue
        job = jobs[jid]
        if any(p not in done for p in job["predecessors"]):
            continue
        if t < job["release_s"]:
            t = job["release_s"]            # 等到释放
        D = zone_dist(zone, job["zone"])
        s = t + D + CHANGEOVER
        fee = fee_of(D)
        if cost + fee > snap["B"] + 1e-9:
            continue
        if s + B > H + 1e-9:
            continue
        p = alpha * d_scen[jid] + beta
        C = s + p
        cost += fee
        done.add(jid)
        if C <= H + 1e-9:
            if C <= job["due_s"] + 1e-9:
                gain += job["weight"]
            t = s + max(p, B)
        else:
            t = H
            break
        zone = job["zone"]
    return gain


class RolloutPolicy(DispatchPolicy):
    """ATC 候选集前 k 名 → 用中位时长贪心补全序列 → M 个 SAA 情景评估
    期望加权准时权重 → 选最优首作业。其余同 DispatchPolicy。"""

    def __init__(self, model, kappa=2.0, guard_q=0.90, selection=True,
                 top_k=3, n_scen=24):
        super().__init__(model, kappa, guard_q, selection)
        self.top_k = top_k
        self.n_scen = n_scen

    def _complete_greedy(self, snap, first, est_p):
        """从 first 出发按 ATC 指数（中位时长）补全序列。"""
        plan = [first]
        t, zone = snap["t"], snap["zone"]
        done = set(snap["completed"]) | {first}
        jobs = snap["jobs"]
        alpha, beta = snap["alpha"], snap["beta"]
        job_first = jobs[first]
        D = zone_dist(zone, job_first["zone"])
        t = t + D + CHANGEOVER + max(est_p[first], B)
        zone = job_first["zone"]
        while len(plan) < len(jobs):
            best = None
            for jid, job in jobs.items():
                if jid in done:
                    continue
                if any(p not in done for p in job["predecessors"]):
                    continue
                D = zone_dist(zone, job["zone"])
                s = max(t, job["release_s"]) + D + CHANGEOVER
                if s + B > snap["H"] + 1e-9:
                    continue
                peff = est_p[jid] + D + CHANGEOVER
                slack = job["due_s"] - (s + est_p[jid])
                idx = (job["weight"] / max(peff, 0.3)) * math.exp(
                    -max(slack, 0.0) / max(self.kappa * peff, 0.3))
                if best is None or idx > best[0]:
                    best = (idx, jid)
            if best is None:
                break
            jid = best[1]
            plan.append(jid)
            done.add(jid)
            job = jobs[jid]
            D = zone_dist(zone, job["zone"])
            t = max(t, job["release_s"]) + D + CHANGEOVER + max(est_p[jid], B)
            zone = job["zone"]
        return plan

    def decide(self, snap):
        cands = self._apply_selection(snap, self._candidates(snap))
        if not cands:
            return super().decide(snap)
        cands.sort(key=lambda c: self._index(snap, c[0], c[1], c[2]),
                   reverse=True)
        cands = cands[: self.top_k]
        alpha, beta = snap["alpha"], snap["beta"]
        jobs = snap["jobs"]
        est_p = {j: self.model.p_median(f["family"], alpha, beta)
                 for j, f in jobs.items()}
        W = sum(f["weight"] for f in jobs.values())
        best = None
        for jid, D, s, fee, b in cands:
            plan = self._complete_greedy(snap, jid, est_p)
            gains = []
            for _ in range(self.n_scen):
                d_scen = {j: float(self.model.sample_d(f["family"], 1, self.rng)[0])
                          for j, f in jobs.items() if j not in snap["executed"]}
                # 已执行作业不再产生收益，但 d_scen 只需覆盖未执行集合
                d_scen.update({j: 0.0 for j in snap["executed"]})
                gains.append(eval_sequence(plan, snap, d_scen, alpha, beta))
            mean_gain = float(np.mean(gains))
            if best is None or mean_gain > best[0]:
                best = (mean_gain, jid, b)
        return ("execute", best[1], best[2])
