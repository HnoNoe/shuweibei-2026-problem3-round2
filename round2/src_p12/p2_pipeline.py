import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from load_records import index_tables, open_record, rep_point

ROOT = Path(r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c")
OUT = ROOT / "round2" / "results_p12"
OUT.mkdir(parents=True, exist_ok=True)

DELTAS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
N_T0 = 15
SEED = 20261123
REVIEW_DELAY = 1.0


def eligible_intervals():
    ss, rs = index_tables()
    ss = ss[ss["eligible"] == 1].reset_index(drop=True)
    env = rs.set_index("episode_uuid")["environment_id"]
    split = rs.set_index("episode_uuid")["split"]
    ss["environment_id"] = ss["episode_uuid"].map(env)
    ss["split"] = ss["episode_uuid"].map(split)
    return ss


def track(uuid, start, end):
    t, L, R, bad, reason, segs = open_record(uuid)
    m = (t >= start) & (t <= end)
    tt = t[m]
    cL, cR, sL, sR, d = rep_point(L[m], R[m])
    c = np.concatenate([cL, cR], axis=1)
    return tt, c


def vel6(c, t):
    return np.gradient(c, t, axis=0)


def acc6(c, t):
    return np.gradient(np.gradient(c, t, axis=0), t, axis=0)


def hold_pred(c0, v0, a0, dt):
    return c0


def cv_pred(c0, v0, a0, dt):
    return c0 + v0 * dt


def kalman_setup():
    return {"x": None, "P": None}


def kalman_pred(c0, v0, a0, dt, st_):
    if st_["x"] is None:
        st_["x"] = np.concatenate([c0, v0]).astype(float)
        st_["P"] = np.eye(12) * 0.01
    F = np.eye(12)
    F[:6, 6:] = np.eye(6) * dt
    H = np.eye(12)[:6]
    Q = np.eye(12) * 1e-3
    Q[6:, 6:] *= 0.1
    R = np.eye(6) * 1e-4
    st_["x"] = F @ st_["x"]
    st_["P"] = F @ st_["P"] @ F.T + Q
    return st_["x"][:6]


def ridge_fit(samples):
    from sklearn.linear_model import Ridge
    X = np.stack([s["X"] for s in samples])
    Y = np.stack([s["Y"] for s in samples])
    m = Ridge(alpha=10.0)
    m.fit(X, Y)
    return m


def ridge_feats(c0, v0, a0, dt, fam, ds_code, seg_frac):
    f = [dt]
    f += list(v0)
    f += list(a0)
    f += [float(fam == "A"), float(fam == "L"), float(fam == "P")]
    f += [float(ds_code == 0), float(ds_code == 1), float(ds_code == 2)]
    f += [seg_frac]
    f += list(c0)
    return np.asarray(f, dtype=np.float64)


def masking_experiment():
    ss, rs = index_tables()
    rng = np.random.default_rng(SEED)
    train_samples, rows = [], []
    for _, r in rs.iterrows():
        try:
            t, L, R, bad, reason, segs = open_record(r["episode_uuid"])
        except Exception:
            continue
        if len(t) < 45 or (t[-1] - t[0]) < 3.0:
            continue
        cL, cR, sL, sR, d = rep_point(L, R)
        c = np.concatenate([cL, cR], axis=1)
        v = vel6(c, t)
        a = acc6(c, t)
        dur = t[-1] - t[0]
        fam = None
        sub = ss[ss["episode_uuid"] == r["episode_uuid"]]
        if len(sub):
            fam = sub["family_id"].mode().iloc[0]
        fam = fam or "A"
        ds_code = {"EgoDemo": 0, "EgoPro": 1, "EgoStandard": 2}[r["dataset"]]
        for _ in range(20):
            dt_choice = DELTAS[rng.integers(len(DELTAS))]
            lo = t[0] + 0.8
            hi = t[-1] - dt_choice - 0.1
            if hi <= lo:
                continue
            t0 = rng.uniform(lo, hi)
            i0 = int(np.searchsorted(t, t0))
            i1 = min(int(np.searchsorted(t, t0 + dt_choice)), len(t) - 1)
            if i1 <= i0 or i0 < 2:
                continue
            c0, v0, a0 = c[i0], v[i0], a[i0]
            cT = c[i1]
            seg_frac = (t0 - t[0]) / max(dur, 1e-9)
            rec = {
                "episode_uuid": r["episode_uuid"],
                "environment_id": r["environment_id"],
                "split": r["split"], "family_id": fam, "dataset": r["dataset"],
                "delta": dt_choice, "t0": round(float(t0), 3),
                "err_hold": float(np.linalg.norm(hold_pred(c0, v0, a0, dt_choice) - cT)),
                "err_cv": float(np.linalg.norm(cv_pred(c0, v0, a0, dt_choice) - cT)),
            }
            X = ridge_feats(c0, v0, a0, dt_choice, fam, ds_code, seg_frac)
            rec["X"], rec["Y"] = X, cT - c0
            (train_samples if r["split"] == "train" else None)
            rows.append(rec)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("X", "Y")} for r in rows])
    from sklearn.preprocessing import StandardScaler
    Xall = np.stack([r["X"] for r in rows])
    Yall = np.stack([r["Y"] for r in rows])
    tr = df["split"].to_numpy() == "train"
    sc = StandardScaler().fit(Xall[tr])
    Xs = sc.transform(Xall)
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=10.0)
    m.fit(Xs[tr], Yall[tr])
    Pall = m.predict(Xs)
    for k in range(len(rows)):
        df.loc[k, "err_ridge"] = float(np.linalg.norm(Pall[k] - Yall[k]))
    resid = Yall - Pall
    sig = {}
    for d in DELTAS:
        msk = (df["delta"] == d).to_numpy()
        if msk.sum() > 5:
            sig[float(d)] = float(np.linalg.norm(resid[msk], axis=1).std())
    df.to_csv(OUT / "masking_predictions.csv", index=False)
    import pickle
    with open(OUT / "ridge_model.pkl", "wb") as f:
        pickle.dump({"model": m, "sigma": sig, "scaler": sc}, f)
    json.dump({str(k): v for k, v in sig.items()}, open(OUT / "masking_sigma.json", "w"), indent=2)
    return df


def error_curves(df):
    out = []
    for d in DELTAS:
        sub = df[df["delta"] == d]
        for meth in ["hold", "cv", "ridge"]:
            e = sub[f"err_{meth}"].to_numpy()
            lo, hi = st.bootstrap((e,), np.mean, confidence_level=0.95, n_resamples=2000,
                                  method="percentile").confidence_interval
            out.append({"delta": float(d), "method": meth, "mae": float(e.mean()),
                        "rmse": float(np.sqrt((e ** 2).mean())),
                        "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(e))})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "error_by_delta.csv", index=False)
    return res


def reobservation_sims():
    ss = eligible_intervals()
    rng = np.random.default_rng(SEED + 1)
    with open(OUT / "ridge_model.pkl", "rb") as f:
        pack = pickle.load(f)
    m = pack["model"]
    sc = pack["scaler"]
    sig = {float(k): v for k, v in pack["sigma"].items()}

    def predict(c0, v0, a0, dt_since, fam, ds_code, frac):
        X = ridge_feats(c0, v0, a0, dt_since, fam, ds_code, frac)
        return m.predict(sc.transform(X[None, :]))[0] + c0

    def sigma_of(dt):
        ks = sorted(sig)
        for k in ks:
            if dt <= k:
                return sig[k]
        return sig[ks[-1]]

    def sim_strategy(tt, c, strat, param, delay=REVIEW_DELAY):
        v = vel6(c, tt)
        a = acc6(c, tt)
        dur = tt[-1] - tt[0]
        n = len(tt)
        errs = []
        n_req = 0
        last_obs = 0
        i = 0
        next_fixed = param if strat == "fixed" else np.inf
        pending_ready = -1.0
        while i < n:
            t = tt[i]
            if strat == "fixed":
                if t >= next_fixed and t < tt[-1] - 0.1:
                    n_req += 1
                    pending_ready = t + delay
                    next_fixed = t + param
                    last_obs = i
            elif strat in ("thresh", "oracle"):
                dt_since = t - tt[last_obs]
                cond = sigma_of(dt_since) > param if strat == "thresh" else \
                    (np.linalg.norm(cv_pred(c[last_obs], v[last_obs], a[last_obs], dt_since) - c[i]) > param)
                if cond and t >= tt[last_obs] + delay and t < tt[-1] - 0.1:
                    n_req += 1
                    last_obs = i
                    pending_ready = -1.0
            if pending_ready > 0 and t >= pending_ready:
                last_obs = i
                pending_ready = -1.0
            dt_since = t - tt[last_obs]
            if last_obs == i:
                pred = c[i]
            else:
                pred = predict(c[last_obs], v[last_obs], a[last_obs], dt_since, "A", 0, 0.5)
            errs.append(float(np.linalg.norm(pred - c[i])))
            i += 1
        return {"mean_err": float(np.mean(errs)), "p95_err": float(np.percentile(errs, 95)),
                "max_err": float(np.max(errs)), "n_req": int(n_req),
                "req_per_min": float(n_req / max(dur / 60, 1e-9))}

    grid = {
        "fixed": [0.5, 1.0, 2.0, 4.0],
        "thresh": [0.05, 0.10, 0.20, 0.40],
        "oracle": [0.10, 0.20, 0.40],
    }
    rows = []
    for _, r in ss.iterrows():
        try:
            tt, c = track(r["episode_uuid"], r["start_s"], r["end_s"])
        except Exception:
            continue
        if len(tt) < 60:
            continue
        row = {"unit_id": r["unit_id"], "split": r["split"], "family_id": r["family_id"]}
        v = vel6(c, tt)
        a = acc6(c, tt)
        errs_hold = [0.0]
        last = 0
        for i in range(1, len(tt)):
            dt_since = tt[i] - tt[last]
            pred = predict(c[last], v[last], a[last], dt_since, r["family_id"],
                           {"EgoDemo": 0, "EgoPro": 1, "EgoStandard": 2}[r["dataset"]],
                           min((tt[i] - tt[0]) / max(tt[-1] - tt[0], 1e-9), 1))
            errs_hold.append(float(np.linalg.norm(pred - c[i])))
        row["no_request_mean"] = float(np.mean(errs_hold))
        row["no_request_p95"] = float(np.percentile(errs_hold, 95))
        for strat, params in grid.items():
            for p in params:
                res = sim_strategy(tt, c, strat, p)
                for k2, v2 in res.items():
                    row[f"{strat}_{p}_{k2}"] = v2
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "reobservation_sims.csv", index=False)
    return df


def stats_p12():
    df = pd.read_csv(OUT / "masking_predictions.csv")
    by_iv = df.groupby(["episode_uuid", "delta"])[["err_hold", "err_cv", "err_ridge"]].mean().reset_index()
    comp = [("ridge", "hold"), ("ridge", "cv"), ("cv", "hold")]
    rows = []
    for a, b in comp:
        d = (by_iv[f"err_{a}"] - by_iv[f"err_{b}"]).to_numpy()
        d = d[~np.isnan(d)]
        w = st.wilcoxon(d)
        dz = float(d.mean() / d.std(ddof=1))
        rows.append({"comparison": f"{a} vs {b}", "n_pairs": int(len(d)),
                     "mean_diff": float(d.mean()), "wilcoxon_p": float(w.pvalue),
                     "cohen_dz": dz})
    st_df = pd.DataFrame(rows)
    st_df["p_holm"] = holm_vec(st_df["wilcoxon_p"].to_numpy())
    st_df.to_csv(OUT / "stats_p2_predictors.csv", index=False)
    return st_df


def holm_vec(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    r = p[o] * (m - np.arange(m))
    adj = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


if __name__ == "__main__":
    df = masking_experiment()
    print("masking:", df.shape)
    cur = error_curves(df)
    print(cur.round(4).to_string())
    sims = reobservation_sims()
    print("sims:", sims.shape)
    stt = stats_p12()
    print(stt.round(6).to_string())
