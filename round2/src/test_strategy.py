
import json
import time

from common import CASES, RESULTS, VAL_SEEDS
from duration_model3 import DurationModel
from sampler import SourceSampler, run_trial
from strategy import CONFIGS, TwoStagePolicy, baseline
import saa_gap

checks = []

def rec(name, passed, detail):
    checks.append(dict(name=name, passed=bool(passed), detail=detail))
    print(("PASS " if passed else "FAIL ") + name + " | " + detail)

SAMPLER = SourceSampler(feat_csv=None)
MODEL = DurationModel.from_gofit()
GRID = [(c, VAL_SEEDS[i]) for c in ("S1", "S2", "S3") for i in (0, 9, 21)]

bad = []
t0 = time.time()
for case, seed in GRID:
    pol = TwoStagePolicy("C6_full", model=MODEL)
    tr = run_trial(case, seed, 1.0, 0.5, SAMPLER, pol, "validation")
    m = tr.metrics()
    if m["invalid_actions"] != 0:
        bad.append((case, seed, m["invalid_actions"]))
dt = time.time() - t0
rec("1 合法性(C6 9局)", not bad,
    f"invalid≠0: {bad if bad else '无'}; 9局共 {dt:.1f}s")

mismatch = []
for case, seed in GRID[:4]:
    pol1 = TwoStagePolicy("C6_full", model=MODEL)
    tr1 = run_trial(case, seed, 1.0, 0.5, SAMPLER, pol1, "validation")
    pol2 = TwoStagePolicy("C6_full", model=MODEL)
    tr2 = run_trial(case, seed, 1.0, 0.5, SAMPLER, pol2, "validation")
    if tr1.log != tr2.log:
        mismatch.append((case, seed))
rec("2 回放一致(CRN地基)", not mismatch,
    f"log 不一致: {mismatch if mismatch else '无'}（4 局逐字节比对）")

snap = dict(t=98.0, zone="Z0", cost=0.0, H=100.0, B=32.0,
            alpha=1.0, beta=0.5,
            jobs={"J1": dict(family="P", release_s=0.0, due_s=200.0,
                             weight=1.0, zone="Z0", predecessors=[])},
            executed=set(), completed=set(), completion_time={})
pol = TwoStagePolicy("C2_guard", model=MODEL)
pol.reset(dict(case="S1", seed=1, alpha=1.0, beta=0.5,
               H=100.0, B=32.0, K=4))
act = pol.decide(snap)
ok3 = act is None or (act[0] == "execute" and act[2] >= 0.2 - 1e-9) \
      or act[0] == "wait"
rec("3 放松链不空", ok3, f"近 H 合成快照返回: {act}")

diff = []
for case, seed in GRID[:3]:
    p_c0 = TwoStagePolicy("C0_bare", hyper={"feedback": False}, model=MODEL)
    tr_a = run_trial(case, seed, 1.0, 0.5, SAMPLER, p_c0, "validation")
    p_b2 = baseline("B2b_WSPT_MEDIAN", model=MODEL)
    tr_b = run_trial(case, seed, 1.0, 0.5, SAMPLER, p_b2, "validation")
    la = [(e["time"], e["action"], e.get("job"), e.get("reserve"))
          for e in tr_a.log]
    lb = [(e["time"], e["action"], e.get("job"), e.get("reserve"))
          for e in tr_b.log]
    if la != lb:
        diff.append((case, seed))
rec("4 引擎退化≡WSPT中位", not diff,
    f"动作序列不一致: {diff if diff else '无'}（3 局逐动作比对；引擎默认估计=中位数）")

jobs = saa_gap.public_jobs("S2")
pol5 = TwoStagePolicy("C6_full", model=MODEL)
pol5.reset(dict(case="S2", seed=7, alpha=1.0, beta=0.5,
                H=CASES["S2"]["H"], B=CASES["S2"]["B"], K=6))
pol5._jobs_cache = jobs
s1_ = pol5._offline_sequence()
pol5b = TwoStagePolicy("C6_full", model=MODEL)
pol5b.reset(dict(case="S2", seed=7, alpha=1.0, beta=0.5,
                 H=CASES["S2"]["H"], B=CASES["S2"]["B"], K=6))
pol5b._jobs_cache = jobs
s2_ = pol5b._offline_sequence()
rec("5 离线序列确定性", s1_ == s2_,
    f"双构建{'一致' if s1_ == s2_ else '不一致'}; 序列长 {len(s1_)}")

pol6 = TwoStagePolicy("C6_full", model=MODEL)
pol6.reset(dict(case="S2", seed=7, alpha=1.0, beta=0.5,
                H=CASES["S2"]["H"], B=CASES["S2"]["B"], K=6))
pol6._jobs_cache = jobs
init_seq = pol6._scan_sequence(pol6._toy_snapshot())
ev_init = pol6._sequence_ev(init_seq, pol6._toy_snapshot())
best_seq = pol6._offline_sequence()
ev_best = pol6._sequence_ev(best_seq, pol6._toy_snapshot())
rec("6 L5单调不劣", ev_best >= ev_init - 1e-9,
    f"初解 {ev_init:.4f} -> 搜索后 {ev_best:.4f}（10 情景均值）")

t0 = time.time()
r7 = saa_gap.saa_gap("S1", M=5, N=10, n_eval=2000, model=MODEL)
ok7 = r7["gap"] >= -1e-9 and r7["gap_rel"] < 0.3
rec("7 SAA间隙冒烟(S1)", ok7,
    f"LB {r7['lb']:.4f}±{r7['lb_ci95']:.4f}(精确={r7['exact_lower_bound']}), "
    f"UB {r7['ub']:.4f}, gap_rel {r7['gap_rel']:.4f}, {time.time()-t0:.1f}s")

out = dict(step="q6 两阶段策略引擎",
           configs={k: v for k, v in CONFIGS.items()},
           checks=checks, all_passed=all(c["passed"] for c in checks))
with open(RESULTS / "q6_strategy_checks.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("ALL PASS" if out["all_passed"] else "SOME FAILED")
