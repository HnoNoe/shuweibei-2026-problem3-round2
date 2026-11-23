
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from common import MAPPINGS, RESULTS, CASES, ROOT
from duration_model3 import DurationModel
import saa_gap

PRIMARY_METRIC = "weighted_on_time_rate"
PAIRS = [
    ("B2-B1 规则升级", "B2_WSPT_MEAN", "B1_EDD"),
    ("B2b-B2 中位数vs均值点估计", "B2b_WSPT_MEDIAN", "B2_WSPT_MEAN"),
    ("B3-B2 稳健点估计", "B3_WSPT_Q90", "B2_WSPT_MEAN"),
    ("C1-B2 ATC折中", "C1_atc", "B2_WSPT_MEAN"),
    ("C2-C1 截断防护", "C2_guard", "C1_atc"),
    ("C3-C1 预算取舍", "C3_budget", "C1_atc"),
    ("C4-C1 rollout前瞻", "C4_roll", "C1_atc"),
    ("C6-C5 离线搜索", "C6_full", "C5_online"),
    ("C5-B3 在线全家vs最强朴素", "C5_online", "B3_WSPT_Q90"),
    ("主对比 C6-B3", "C6_full", "B3_WSPT_Q90"),
    ("C6-B2 全栈vs标准解", "C6_full", "B2_WSPT_MEAN"),
    ("RM-C5 买到顶反证", "RM_buytop", "C5_online"),
]

def paired(df, a, b, metric=PRIMARY_METRIC):
    da = df[df.arm == a].set_index(["alpha", "beta", "case", "seed"])[metric]
    db = df[df.arm == b].set_index(["alpha", "beta", "case", "seed"])[metric]
    common = da.index.intersection(db.index)
    d = (da.loc[common] - db.loc[common]).dropna()
    n = len(d)
    mean = float(d.mean())
    if n > 1 and float(d.std(ddof=1)) > 0:
        ci = float(stats.t.interval(0.95, n - 1, loc=mean,
                                    scale=d.std(ddof=1) / np.sqrt(n))[1] - mean)
    else:
        ci = 0.0
    return dict(n=n, mean_diff=mean, ci95_halfwidth=abs(ci),
                median_diff=float(d.median()))

def _run_saa():
    model = DurationModel.from_gofit()
    frozen_h = json.load(open(RESULTS / "q7_tuning.json",
                              encoding="utf-8"))["frozen"]
    from strategy import TwoStagePolicy
    saa_rows = {}
    for case in ("S1", "S2", "S3"):
        jobs = saa_gap.public_jobs(case)
        pol = TwoStagePolicy("C6_full", hyper=frozen_h, model=model)
        pol.reset(dict(case=case, seed=0, alpha=1.0, beta=0.5,
                       H=CASES[case]["H"], B=CASES[case]["B"],
                       K=CASES[case]["K"]))
        pol._jobs_cache = jobs
        cand = pol._offline_sequence()
        M, N = (20, 30) if case == "S1" else (10, 20)
        print(f"SAA {case}: candidate={cand}, M={M} N={N}", flush=True)
        t1 = time.time()
        r = saa_gap.saa_gap(case, candidate_seq=cand, M=M, N=N, n_eval=10000,
                            model=model)
        saa_rows[case] = {k: v for k, v in r.items() if k not in
                          ("batch_seqs", "candidate_seq")}
        saa_rows[case]["wall_s"] = time.time() - t1
        print(f"  -> LB {r['lb']:.4f}±{r['lb_ci95']:.4f} UB {r['ub']:.4f} "
              f"gap_rel {r['gap_rel']:.4f} exact={r['exact_lower_bound']} "
              f"{saa_rows[case]['wall_s']:.0f}s", flush=True)
    return saa_rows

def main():
    t0 = time.time()
    df = pd.read_csv(RESULTS / "q7_ablation.csv")
    df.columns = [c.strip("\ufeff") for c in df.columns]

    bad_protocol = int((df["valid_protocol"] != 1).sum())
    assert bad_protocol == 0, f"协议不合规 {bad_protocol} 局，返工不解释"
    assert len(df) == 12 * 450, f"行数 {len(df)} != 5400"

    pairs = []
    for label, a, b in PAIRS:
        r = paired(df, a, b)
        r.update(label=label, arm_a=a, arm_b=b)
        pairs.append(r)
        print(f"{label}: diff {r['mean_diff']:+.4f} "
              f"±{r['ci95_halfwidth']:.4f} (n={r['n']})", flush=True)

    primary = [p for p in pairs if p["label"].startswith("主对比")][0]
    assert primary["ci95_halfwidth"] < 0.03, \
        f"主对比 CI 宽 {primary['ci95_halfwidth']:.4f} ≥ 0.03，功效门槛未达"

    breakdown = {}
    for view in ("case", "mapping"):
        rows = {}
        for key, sub in (df.groupby("case") if view == "case"
                         else df.groupby(["alpha", "beta"])):
            try:
                rows[str(key)] = paired(sub, "C6_full", "B3_WSPT_Q90")
            except Exception:
                pass
        breakdown[view] = rows

    second = {}
    for arm in ("C6_full", "B3_WSPT_Q90", "RM_buytop", "C5_online"):
        sub = df[df.arm == arm]
        second[arm] = dict(
            cost_mean=float(sub["cost"].mean()),
            completion_mean=float(sub["weighted_completion_rate"].mean()),
            reviews_mean=float(sub["observations"].mean()),
            censored_mean=float(sub["censored_units"].mean()))

    rm_pair = [p for p in pairs if p["label"].startswith("RM-C5")][0]
    rm_verdict = ("RM 未显著优于 C5（CI 含 0 或为负）→ 零审查结论封死"
                  if rm_pair["mean_diff"] - rm_pair["ci95_halfwidth"] <= 0
                  else "RM 显著优于 C5 → 按 no_review.md §5 重写信息层结论！")
    print("RM verdict:", rm_verdict, flush=True)

    summary = dict(
        primary_metric=PRIMARY_METRIC,
        n_rows=int(len(df)),
        valid_protocol_violations=bad_protocol,
        frozen=json.load(open(RESULTS / "q7_tuning.json", encoding="utf-8"))["frozen"],
        primary_pair=dict(arms=["C6_full", "B3_WSPT_Q90"], **primary),
        pairs=pairs, breakdown=breakdown, secondary=second,
        rm_verdict=rm_verdict)
    with open(RESULTS / "q7_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    skip_saa = "--skip-saa" in sys.argv
    saa_path = RESULTS / "q7_saa_gap.json"
    if skip_saa and saa_path.exists():
        saa_rows = json.load(open(saa_path, encoding="utf-8"))
        print("SAA: 复用已有 q7_saa_gap.json（--skip-saa）", flush=True)
    else:
        saa_rows = _run_saa()
    with open(saa_path, "w", encoding="utf-8") as f:
        json.dump(saa_rows, f, ensure_ascii=False, indent=2)

    md = ["# 第 7 步消融报告（validation 集，450 局/臂，CRN 配对）\n",
          f"主终点：{PRIMARY_METRIC}；主对比 C6_full vs B3_WSPT_Q90；"
          f"超参冻结 {summary['frozen']}（train 内部种子选定）\n",
          "## 归因链（配对差分，± 为 95% CI 半宽）\n",
          "| 对比 | 差值 | 95% CI | 中位差 | n |", "|---|---|---|---|---|"]
    for p in pairs:
        md.append(f"| {p['label']} | {p['mean_diff']:+.4f} | "
                  f"±{p['ci95_halfwidth']:.4f} | {p['median_diff']:+.4f} "
                  f"| {p['n']} |")
    md += ["\n## 主对比分案例分解\n", "| 案例 | 差值 | 95% CI |", "|---|---|---|"]
    for k, v in breakdown["case"].items():
        md.append(f"| {k} | {v['mean_diff']:+.4f} | ±{v['ci95_halfwidth']:.4f} |")
    md += ["\n## 次级指标均值\n", "| 臂 | 费用 | 完成率 | 审查数 | 越界数 |",
           "|---|---|---|---|---|"]
    for arm, v in second.items():
        md.append(f"| {arm} | {v['cost_mean']:.2f} | {v['completion_mean']:.4f} "
                  f"| {v['reviews_mean']:.2f} | {v['censored_mean']:.2f} |")
    md += ["\n## 买到顶反证判定\n", rm_verdict, "\n## SAA 最优性间隙（名义映射）\n",
           "| 案例 | LB | ±95%CI | UB | 相对间隙 | 下界性质 |", "|---|---|---|---|---|---|"]
    for case, v in saa_rows.items():
        md.append(f"| {case} | {v['lb']:.4f} | ±{v['lb_ci95']:.4f} "
                  f"| {v['ub']:.4f} | {v['gap_rel']*100:.2f}% | "
                  f"{'精确枚举' if v['exact_lower_bound'] else '局部搜索(caveat)'} |")
    md.append("\n口径：SAA 度量固定先验序列策略类，在线 recourse 弱占优于该类，"
              "间隙对本策略是保守的。")
    with open(ROOT / "round2/docs/ablation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"ALL DONE in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
