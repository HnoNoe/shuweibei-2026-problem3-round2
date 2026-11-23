"""第 7 步 · 之二："买到顶"变体（review_max）——零审查结论的反证出口。

设计（step7_plan.md §2 / no_review.md §5）：给反方一切好处——
C5 在线全家引擎 + 买满审查（每作业 2 次、每案用足 K 配额）+ 数据里最好的
前缀模型（Q2 网格胜者 GBM，train split 拟合）用满已解锁特征。

若它仍赢不了 C5 ⇒ "审查在这道题里没有信息"从我们的选择升格为数据判决；
若它赢了 ⇒ 按约改写信息层结论（此分支逻辑在 analyze_ablation.py 里检查并明示）。
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from common import (B_MIN, CHANGEOVER, PREFIXES, PROBLEM, REVIEW_COST, ROOT,
                    zone_dist)
from strategy import CONFIGS, EPS, FastSim, TwoStagePolicy, fee_of

MODELS = {}


def _train_gbms():
    """train split：三窗 GBM（p02/p06/p10 前缀特征 -> duration_s）。"""
    global MODELS
    if MODELS:
        return MODELS
    feats = pd.read_csv(ROOT / "prefix_features.csv")
    df = feats[feats["split"] == "train"]     # 文件自带 duration_s（真值列，仅训练用）
    MODELS = {}
    for lvl, pre in [(PREFIXES[0], "p02"), (PREFIXES[1], "p06"),
                     (PREFIXES[2], "p10")]:
        cols = [c for c in df.columns if c.startswith(pre + "_")]
        sub = df[cols + ["duration_s"]].dropna()      # Q2 同款清洗：丢含缺口的行
        g = GradientBoostingRegressor(random_state=7)
        # 裸 numpy 训练（不带特征名），否则 predict 每次都刷
        # "X does not have valid feature names" 警告，批量仿真输出爆炸。
        g.fit(sub[cols].to_numpy(), sub["duration_s"].to_numpy())
        # 训练列带 "p10_" 前缀；快照特征键已剥前缀（sampler.load_features），
        # 存映射避免预测时 all(c in f) 恒 False 的静默失效（2026-11-23 修复）。
        # 快照特征可能含 NaN（源表缺口），训练时丢行、预测时中位数填补。
        MODELS[lvl] = (g, cols, {c: c[len(pre) + 1:] for c in cols},
                       sub[cols].median())
    return MODELS


class ReviewMaxPolicy(TwoStagePolicy):
    """C5 引擎 + 审查买满 + GBM 前缀估计。"""

    def __init__(self, hyper=None, model=None):
        super().__init__(CONFIGS["C5_online"], hyper=hyper, model=model)
        _train_gbms()
        self._snap = None
        self._gbm_cache = {}

    # ---- GBM 估计替换点估计（守卫分位数仍走 family 分布）----
    def _est(self, jid, info, q=None):
        if q is not None:
            return self._d_hat(info["family"], q=q)
        pred = self._gbm_d(jid)
        if pred is not None:
            return min(max(pred, 0.05), 300.0)
        return self._d_hat(info["family"])

    def _gbm_d(self, jid):
        if self._snap is None:
            return None
        per = self._snap.get("prefix_feat", {}).get(jid, {})
        for lvl in sorted(MODELS, reverse=True):      # 用最高已解锁档
            if lvl in per and per[lvl]:
                g, cols, mapping, med = MODELS[lvl]
                f = per[lvl]
                if all(mapping[c] in f for c in cols):
                    # (jid, lvl) 缓存：特征值静态，解锁档只增，重复 predict
                    # 是 rollout 热路径（单局 8.5 万次调用）的 10 倍加速，语义不变。
                    key = (jid, lvl)
                    if key not in self._gbm_cache:
                        x = [f[mapping[c]] if pd.notna(f[mapping[c]])
                             else med[c] for c in cols]
                        self._gbm_cache[key] = float(g.predict([x])[0])
                    return self._gbm_cache[key]
        return None

    # ---- 买审查触发 ----
    def _review_trigger(self, snap):
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
            return None
        sim_now = self._sim_now(snap, cands)
        top = max(cands, key=lambda j: self._score(sim_now, j, jobs))
        if snap["review_count"][top] < 2 and snap["reviews"] < snap["K"] \
                and snap["cost"] + REVIEW_COST <= snap["B"] + EPS \
                and t + 1.0 <= snap["H"] + EPS:
            return ("review", top)
        return None

    def _sim_now(self, snap, cands):
        sim = FastSim(snap, {}, snap["alpha"], snap["beta"], snap["H"], snap["B"])
        sim.dur = {j: snap["alpha"] * self._est(j, snap["jobs"][j]) + snap["beta"]
                   for j in cands}
        return sim

    def decide(self, snap):
        self._snap = snap
        act = self._review_trigger(snap)
        if act is not None:
            return act
        return super().decide(snap)


if __name__ == "__main__":
    # 冒烟：3 局，验证 GBM 装载 + 审查触发 + 协议合规
    from common import CASES, VAL_SEEDS
    from duration_model3 import DurationModel
    from sampler import SourceSampler, run_trial

    sampler = SourceSampler(feat_csv=str(ROOT / "prefix_features.csv"))
    pol = ReviewMaxPolicy(model=DurationModel.from_gofit())
    for case in ("S1", "S3"):
        tr = run_trial(case, VAL_SEEDS[0], 1.0, 0.5, sampler, pol, "validation")
        m = tr.metrics()
        print(case, "reviews=", m["observations"], "cost=", m["cost"],
              "ot=", round(m["weighted_on_time_rate"], 4),
              "invalid=", m["invalid_actions"])
    print("REVIEW_MAX SMOKE OK")
