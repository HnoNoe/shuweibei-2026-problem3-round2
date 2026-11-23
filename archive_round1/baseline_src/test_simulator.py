"""仿真器协议单元测试。"""
import sys
sys.path.insert(0, "src")
from simulator import Job, Trial, zone_dist

def make_jobs():
    return [
        Job("J1", "P", 0, 35, 3, "Z4", [], 5.0, "U1"),
        Job("J2", "A", 0, 45, 1, "Z3", [], 3.0, "U2"),
        Job("J3", "L", 20, 55, 2, "Z2", [], 4.0, "U3"),
    ]

def test_zone_dist():
    assert abs(zone_dist("Z0", "Z4") - 2.0) < 1e-9
    assert abs(zone_dist("Z0", "Z3") - 2**0.5) < 1e-9
    print("test_zone_dist OK")

def test_basic_execution():
    tr = Trial(make_jobs(), H=100, B=32, K=4, alpha=1, beta=0.5, seed=1, case="S1")
    # 执行 J1: D=2, s=0+2+0.5=2.5, fee=2+0.1b+0.4
    b = 6.0
    assert tr.do_execute("J1", b).ok
    p1 = 1*5+0.5
    C1 = 2.5 + p1
    assert abs(tr.completion_time["J1"] - C1) < 1e-9
    assert abs(tr.cost - (2+0.1*b+0.4)) < 1e-9
    assert tr.zone == "Z4"
    print("test_basic_execution OK  C1=%.2f cost=%.2f" % (C1, tr.cost))

def test_invalid_predecessor():
    jobs = make_jobs() + [Job("J4","P",0,50,2,"Z1",["J2"],3.0,"U4")]
    tr = Trial(jobs, H=100, B=32, K=4, alpha=1, beta=0.5, seed=1, case="S1")
    r = tr.do_execute("J4", 4.0)
    assert not r.ok and "predecessor" in r.reason
    assert tr.violations == 1 and tr.terminated
    print("test_invalid_predecessor OK")

def test_budget_violation():
    tr = Trial(make_jobs(), H=100, B=0.5, K=4, alpha=1, beta=0.5, seed=1, case="S1")
    r = tr.do_execute("J1", 6.0)
    assert not r.ok and "budget" in r.reason
    assert tr.valid_protocol_metric() if hasattr(tr,'valid_protocol_metric') else True
    m = tr.metrics()
    assert m["valid_protocol"] == 0 and m["invalid_actions"] == 1
    print("test_budget_violation OK")

def test_censoring():
    jobs = [Job("J1","P",0,35,3,"Z1",[],40.0,"U1")]  # p=40.5
    tr = Trial(jobs, H=10, B=32, K=4, alpha=1, beta=0.5, seed=1, case="S1")
    r = tr.do_execute("J1", 5.0)
    assert r.ok
    m = tr.metrics()
    assert m["censored_units"] == 1 and m["jobs_completed"] == 0
    assert abs(m["weighted_completion_rate"]) < 1e-9
    print("test_censoring OK  t=%.2f"%tr.t)

def test_review():
    tr = Trial(make_jobs(), H=100, B=32, K=4, alpha=1, beta=0.5, seed=1, case="S1")
    r = tr.do_review("J1")
    assert r.ok and tr.prefix_level["J1"] == 0.6 and tr.cost == 1 and tr.reviews == 1
    assert abs(tr.t - 1.0) < 1e-9
    tr.do_review("J1")
    assert tr.prefix_level["J1"] == 1.0 and tr.cost == 2
    r = tr.do_review("J1")
    assert not r.ok  # 每job最多2次
    print("test_review OK")

def test_overrun_idle():
    # p=2.5, b=5 -> idle=2.5; b=1 -> overrun
    j1 = [Job("J1","P",0,35,3,"Z0",[],2.0,"U1")]
    tr = Trial(j1, H=100, B=32, K=4, alpha=1, beta=0.5, seed=1, case="S1")
    tr.do_execute("J1", 5.0)
    m = tr.metrics()
    assert abs(m["reserved_idle_s"] - 2.5) < 1e-9, m
    j2 = [Job("J1","P",0,35,3,"Z0",[],2.0,"U1")]
    tr2 = Trial(j2, H=100, B=32, K=4, alpha=1, beta=0.5, seed=1, case="S1")
    tr2.do_execute("J1", 1.0)
    m2 = tr2.metrics()
    assert abs(m2["observed_overrun_s"] - 1.5) < 1e-9, m2
    print("test_overrun_idle OK")

if __name__ == "__main__":
    test_zone_dist()
    test_basic_execution()
    test_invalid_predecessor()
    test_budget_violation()
    test_censoring()
    test_review()
    test_overrun_idle()
    print("ALL TESTS PASSED")
