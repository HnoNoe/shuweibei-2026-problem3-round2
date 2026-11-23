import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_records import rep_point
from p1_pipeline import vel, holm, kw_test
from p2_pipeline import (acc6, cv_pred, hold_pred, holm_vec, kalman_pred,
                         kalman_setup, ridge_feats, ridge_fit, vel6)

EPS = 1e-9


def synth_hands(n=20, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, (n - 1) / 30.0, n)
    cL = np.column_stack([0.5 * t, 0.1 * t ** 2, 0.02 * t])
    cR = cL + np.array([0.30, 0.0, 0.0])
    L = np.stack([cL + rng.normal(0, 0.001, cL.shape) for _ in range(4)], axis=1)
    R = np.stack([cR + rng.normal(0, 0.001, cR.shape) for _ in range(4)], axis=1)
    return t, L, R


def test_rep_point_geometry():
    t, L, R = synth_hands()
    cL, cR, sL, sR, d = rep_point(L, R)
    assert cL.shape == (len(t), 3) and cR.shape == (len(t), 3)
    assert np.allclose(cL.mean(axis=0), L.mean(axis=(0, 1)), atol=1e-9)
    assert np.all(sL < 0.01) and np.all(sR < 0.01)
    assert np.allclose(d, 0.30, atol=0.01)
    assert np.all(d >= 0.0)


def test_vel_constant_motion():
    n = 30
    t = np.arange(n) / 30.0
    c = np.column_stack([0.6 * t, np.zeros(n), np.zeros(n)])
    v = vel(c, t)
    assert np.allclose(v[2:-2], 0.6, atol=1e-6)
    v6 = vel6(c, t)
    assert v6.shape == (n, 3)
    assert np.allclose(v6[2:-2, 0], 0.6, atol=1e-6)
    a6 = acc6(c, t)
    assert a6.shape == (n, 3)
    assert np.allclose(a6[3:-3], 0.0, atol=1e-4)


def test_baseline_preds():
    c0 = np.ones(6)
    v0 = np.full(6, 0.5)
    a0 = np.zeros(6)
    dt = 2.0
    assert np.array_equal(hold_pred(c0, v0, a0, dt), c0)
    assert np.allclose(cv_pred(c0, v0, a0, dt), c0 + 1.0)
    assert np.allclose(cv_pred(c0, v0, a0, 0.0), c0)


def test_kalman_init_and_step():
    c0 = np.arange(6, dtype=float)
    v0 = np.full(6, 0.2)
    st_ = kalman_setup()
    assert st_["x"] is None and st_["P"] is None
    out = kalman_pred(c0, v0, np.zeros(6), 1.0, st_)
    assert np.allclose(out, c0 + 0.2, atol=1e-9)
    assert st_["x"].shape == (12,) and st_["P"].shape == (12, 12)
    assert np.allclose(st_["P"], st_["P"].T)
    assert np.all(np.linalg.eigvalsh(st_["P"]) > 0.0)


def test_ridge_feats_layout():
    f = ridge_feats(np.ones(6), np.full(6, 0.3), np.zeros(6), 1.5, "L", 1, 0.4)
    assert f.dtype == np.float64
    assert len(f) == 1 + 6 + 6 + 3 + 3 + 1 + 6
    assert f[0] == 1.5 and f[-6] == 1.0 and f[-1] == 1.0
    assert f[14] == 1.0 and f[16] == 0.0 and f[17] == 1.0


def test_ridge_fit_recovers_linear_map():
    rng = np.random.default_rng(1)
    W = np.array([0.1, -0.2, 0.3, 0.0, 0.05, -0.1])
    samples = []
    for _ in range(2000):
        x = rng.normal(0, 1, 6)
        y = x + 0.5 * W + rng.normal(0, 0.01, 6)
        samples.append({"X": x, "Y": y})
    m = ridge_fit(samples)
    x_new = rng.normal(0, 1, 6)
    pred = m.predict(x_new.reshape(1, -1))[0]
    assert np.allclose(pred, x_new + 0.5 * W, atol=0.05)
    train_err = np.mean([np.abs(m.predict(s["X"].reshape(1, -1))[0] - s["Y"])
                         for s in samples[:50]])
    assert train_err < 0.05


def test_holm_bounds_and_monotonicity():
    p = np.array([0.001, 0.01, 0.04, 0.2])
    adj = holm(p)
    assert np.all(adj >= p - EPS)
    assert np.all(adj <= 1.0 + EPS)
    assert np.all(np.diff(adj) >= -EPS)
    assert adj[np.argmin(p)] <= p.min() * len(p) + EPS
    adj2 = holm_vec(p)
    assert np.allclose(adj, adj2)
    single = holm([0.03])
    assert abs(single[0] - 0.03) < EPS


def test_holm_orders_by_value_not_position():
    p = np.array([0.5, 0.001, 0.05])
    adj = holm(p)
    assert adj[1] < adj[2] < adj[0]
    assert np.allclose(np.sort(adj), holm(np.sort(p)))


def test_kw_test_detects_shift():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "g": ["a"] * 40 + ["b"] * 40,
        "x": np.concatenate([rng.normal(0, 1, 40), rng.normal(3, 1, 40)]),
    })
    r = kw_test(df, "x", "g")
    assert r is not None and r["p"] < 1e-6 and r["k"] == 2 and r["n"] == 80
    same = pd.DataFrame({"g": ["a"] * 40 + ["b"] * 40,
                         "x": rng.normal(0, 1, 80)})
    r2 = kw_test(same, "x", "g")
    assert r2["p"] > 0.001


def test_kw_test_needs_two_groups():
    df = pd.DataFrame({"g": ["a"] * 10, "x": np.arange(10.0)})
    assert kw_test(df, "x", "g") is None
    small = pd.DataFrame({"g": ["a", "a", "b"], "x": [1.0, 2.0, 3.0]})
    assert kw_test(small, "x", "g") is None


if __name__ == "__main__":
    tests = [f for f in list(globals().values())
             if callable(f) and f.__name__.startswith("test_")]
    for f in tests:
        f()
        print(f.__name__, "OK")
    print(f"ALL {len(tests)} TESTS PASSED")
