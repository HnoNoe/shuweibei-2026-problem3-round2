"""仿真器协议单元测试（重建版）。

覆盖：距离与费用手算、b 占优性、释放/前序规则、截断、overrun/idle、
审查全部规则、等待规则、违规语义（钱不退+trial终止）、预算拦截、
硬性检查不用 p、真值不泄漏、分母含未完成作业、
snapshot 隔离性（改拷贝不影响裁判内部状态）、审查不要求前序完成。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import Job, Trial, zone_dist

EPS = 1e-9


def J(jid, fam="P", rel=0.0, due=50.0, w=1.0, zone="Z0", preds=(), d=5.0,
      ref="U0"):
    return Job(jid, fam, rel, due, w, zone, list(preds), d, ref)


def test_dist():
    assert abs(zone_dist("Z0", "Z4") - 2.0) < EPS
    assert abs(zone_dist("Z0", "Z3") - 2 ** 0.5) < EPS
    assert abs(zone_dist("Z2", "Z3") - 1.0) < EPS


def test_execute_math():
    tr = Trial([J("J1", zone="Z4", d=5.0)], 100, 32, 4, 1.0, 0.5, 1, "S1")
    ok, _ = tr.execute("J1", 0.2)
    assert ok
    # s = 0 + 2 + 0.5 = 2.5; fee = 2 + 0.02 + 0.4 = 2.42
    assert abs(tr.log[0]["service_start"] - 2.5) < EPS
    assert abs(tr.log[0]["fee"] - 2.42) < EPS
    # p = 5.5; C = 8.0; next avail = s + max(p, b) = 8.0
    assert abs(tr.t - 8.0) < EPS and tr.zone == "Z4"
    assert tr.metrics()["weighted_on_time_rate"] == 1.0


def test_b_dominance():
    """b 不影响完成时刻：b=0.2 与 b=5 的 C 相同，费用更低、idle 为 0。"""
    def run(b, d=5.0):
        tr = Trial([J("J1", d=d)], 100, 32, 4, 1.0, 0.5, 1, "S1")
        tr.execute("J1", b)
        return tr
    m_small, m_big = run(0.2).metrics(), run(5.0, d=2.0).metrics()
    assert m_small["weighted_on_time_rate"] == m_big["weighted_on_time_rate"]
    assert m_small["cost"] < m_big["cost"]
    assert m_small["reserved_idle_s"] == 0.0 and m_big["reserved_idle_s"] > 0


def test_release_and_preds():
    tr = Trial([J("J1", rel=10.0), J("J2", preds=["J1"])], 100, 32, 4,
               1.0, 0.5, 1, "S1")
    assert not tr.execute("J1", 1.0)[0] and tr.violations == 1  # 未释放
    tr = Trial([J("J1", rel=10.0), J("J2", preds=["J1"])], 100, 32, 4,
               1.0, 0.5, 1, "S1")
    assert not tr.execute("J2", 1.0)[0]                      # 前序未完成
    tr = Trial([J("J1"), J("J2", preds=["J1"], rel=0.0)], 100, 32, 4,
               1.0, 0.5, 1, "S1")
    assert not tr.execute("J2", 1.0)[0]                      # 前序未完成(未执行)


def test_censoring():
    tr = Trial([J("J1", d=40.0)], 10, 32, 4, 1.0, 0.5, 1, "S1")
    ok, _ = tr.execute("J1", 0.2)
    assert ok and tr.terminated
    m = tr.metrics()
    assert m["censored_units"] == 1 and m["weighted_completion_rate"] == 0.0
    # C=H=10 计入延误: w*max(10-due,0)
    assert abs(m["weighted_tardiness_s"] - max(10 - 50.0, 0)) < EPS


def test_overrun_and_idle():
    tr = Trial([J("J1", d=2.0, zone="Z0")], 100, 32, 4, 1.0, 0.5, 1, "S1")
    tr.execute("J1", 5.0)   # p=2.5 < b -> idle 2.5
    assert abs(tr.metrics()["reserved_idle_s"] - 2.5) < EPS
    tr = Trial([J("J1", d=2.0, zone="Z0")], 100, 32, 4, 1.0, 0.5, 1, "S1")
    tr.execute("J1", 1.0)   # p=2.5 > b -> overrun 1.5
    assert abs(tr.metrics()["observed_overrun_s"] - 1.5) < EPS


def test_overrun_capped_at_H():
    # p 很大被截断：overrun 只计到 H-s
    tr = Trial([J("J1", d=40.0)], 10, 32, 4, 1.0, 0.5, 1, "S1")
    tr.execute("J1", 0.2)   # s=0.5, H-s=9.5, p=40.5 -> overrun 9.3
    assert abs(tr.metrics()["observed_overrun_s"] - 9.3) < EPS


def test_review_rules():
    tr = Trial([J("J1"), J("J2")], 100, 32, 4, 1.0, 0.5, 1, "S1")
    assert tr.review("J1")[0]
    assert tr.review("J1")[0]
    assert not tr.review("J1")[0]          # 每 job 最多 2 次
    tr = Trial([J("J1")], 100, 32, 1, 1.0, 0.5, 1, "S1")   # K=1
    assert tr.review("J1")[0]
    assert not tr.review("J1")[0]          # 案例级上限 K
    tr = Trial([J("J1")], 100, 1.0, 4, 1.0, 0.5, 1, "S1")  # B=1
    assert tr.review("J1")[0]
    assert not tr.review("J1")[0]          # 预算不足（cost+1>B）
    tr = Trial([J("J1", rel=5.0)], 100, 32, 4, 1.0, 0.5, 1, "S1")
    assert not tr.review("J1")[0]          # 未释放不可审查
    tr = Trial([J("J1")], 1.0, 32, 4, 1.0, 0.5, 1, "S1")
    assert tr.wait(0.5)[0]
    assert not tr.review("J1")[0]          # 返回时刻 t+1 > H


def test_review_needs_no_predecessor():
    """原文第 2 节：审查不要求前序完成（与执行不同）。"""
    tr = Trial([J("J1"), J("J2", preds=["J1"])], 100, 32, 4,
               1.0, 0.5, 1, "S1")
    assert tr.review("J2")[0]              # J1 未完成也能审查 J2


def test_review_occupies_and_prefix_ladder():
    pf = {0.2: {"x": 1.0}, 0.6: {"x": 2.0}, 1.0: {"x": 3.0}}
    j = Job("J1", "P", 0, 50, 1, "Z0", [], 5.0, "U0", prefix_feat=pf)
    tr = Trial([j], 100, 32, 4, 1.0, 0.5, 1, "S1")
    tr.review("J1")
    assert abs(tr.t - 1.0) < EPS and tr.prefix_level["J1"] == 0.6
    snap = tr.snapshot()
    assert 0.6 in snap["prefix_feat"]["J1"] and 1.0 not in snap["prefix_feat"]["J1"]


def test_wait_rules():
    def fresh():
        return Trial([J("J1")], 100, 32, 4, 1.0, 0.5, 1, "S1")
    assert not fresh().wait(0)[0]
    assert not fresh().wait(-1)[0]
    assert not fresh().wait(float("nan"))[0]
    assert not fresh().wait(200)[0]          # 跨越 H
    tr = fresh()
    assert tr.wait(3.5)[0] and abs(tr.t - 3.5) < EPS


def test_violation_terminates_and_keeps_cost():
    tr = Trial([J("J1", d=5.0), J("J2", rel=30.0)], 100, 32, 4, 1.0, 0.5,
               1, "S1")
    tr.execute("J1", 0.2)
    c0 = tr.cost
    assert not tr.execute("J1", 0.2)[0]    # 重复作业 -> 违规终止
    m = tr.metrics()
    assert m["valid_protocol"] == 0 and m["invalid_actions"] == 1
    assert tr.cost == c0                   # 已消耗资源保留


def test_budget_blocks_execution():
    tr = Trial([J("J1", zone="Z4")], 100, 2.0, 4, 1.0, 0.5, 1, "S1")
    assert not tr.execute("J1", 0.2)[0]    # fee 2.42 > B=2


def test_sbH_check_uses_b_not_p():
    # s+b <= H 是硬性检查（不得用未释放的 p）
    tr = Trial([J("J1", d=500.0, zone="Z0")], 10, 32, 4, 1.0, 0.5, 1, "S1")
    assert tr.execute("J1", 0.2)[0]        # s+0.2=0.7<=10 接受，然后被截断
    assert tr.metrics()["censored_units"] == 1


def test_no_d_leakage():
    tr = Trial([J("J1")], 100, 32, 4, 1.0, 0.5, 1, "S1")
    snap = tr.snapshot()
    assert "d" not in snap["jobs"]["J1"]
    assert "duration" not in str(snap["jobs"]["J1"])


def test_snapshot_isolation():
    """策略拿到的是拷贝：改它不能污染裁判内部状态（防作弊纵深）。"""
    tr = Trial([J("J1")], 100, 32, 4, 1.0, 0.5, 1, "S1")
    snap = tr.snapshot()
    snap["t"] = 999.0
    snap["jobs"]["J1"]["due_s"] = -1.0
    snap["completed"].add("J1")
    snap2 = tr.snapshot()
    assert snap2["t"] == 0.0
    assert snap2["jobs"]["J1"]["due_s"] == 50.0
    assert "J1" not in snap2["completed"]


def test_metrics_denominator_all_jobs():
    tr = Trial([J("J1", w=3.0, due=100.0), J("J2", w=1.0, due=100.0,
                                              rel=200.0)], 100, 32, 4,
               1.0, 0.5, 1, "S1")
    tr.execute("J1", 0.2)
    m = tr.metrics()
    assert abs(m["weighted_completion_rate"] - 0.75) < EPS  # 3/4，分母含未释放


def test_stop_allowed_anytime():
    tr = Trial([J("J1")], 100, 32, 4, 1.0, 0.5, 1, "S1")
    assert tr.stop()[0]
    assert tr.terminated and tr.stopped
    m = tr.metrics()
    assert m["elapsed_s"] == 0.0 and m["valid_protocol"] == 1


if __name__ == "__main__":
    tests = [f for f in list(globals().values())
             if callable(f) and f.__name__.startswith("test_")]
    for f in tests:
        f()
        print(f.__name__, "OK")
    print(f"ALL {len(tests)} TESTS PASSED")
