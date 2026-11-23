import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from load_records import index_tables, open_record, rep_point

ROOT = Path(r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c")
OUT = ROOT / "round2" / "results_p12"
OUT.mkdir(parents=True, exist_ok=True)
FPS = 30.0


def vel(c, t):
    v = np.gradient(c, t, axis=0)
    return np.linalg.norm(v, axis=1)


def audit():
    ss, rs = index_tables()
    rows = []
    for _, r in rs.iterrows():
        t, L, R, bad, reason, segs = open_record(r.episode_uuid)
        dt = np.diff(t)
        cL, cR, sL, sR, d = rep_point(L, R)
        sub = ss[ss["episode_uuid"] == r["episode_uuid"]]
        fams = ",".join(sorted(sub["family_id"].dropna().unique())) if len(sub) else ""
        rows.append({
            "episode_uuid": r["episode_uuid"],
            "dataset": r["dataset"],
            "environment_id": r["environment_id"],
            "split": r["split"],
            "n_frames": len(t),
            "duration_s": round(float(t[-1] - t[0]), 3),
            "fps_median": round(float(1.0 / np.median(dt)), 2) if len(dt) else np.nan,
            "fps_jitter_std_s": round(float(np.std(dt)), 5),
            "gap_frames_over_2dt": int((dt > 2.5 / FPS).sum()),
            "bad_ratio": round(float(bad.mean()), 5),
            "n_segments": len(segs),
            "seg_coverage": round(float(sum(b - a for a, b, _ in segs) / max(t[-1] - t[0], 1e-9)), 4),
            "families": fams,
            "mean_speed": round(float(vel(cL, t).mean()), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "audit_records.csv", index=False)

    iv = ss.copy()
    iv["retained"] = iv["eligible"] == 1
    iv_summary = iv.groupby("dataset").agg(
        intervals=("unit_id", "count"),
        retained=("retained", "sum"),
        mean_dur=("duration_s", "mean"),
    ).round(3)
    iv_summary.to_csv(OUT / "audit_intervals.csv")

    dup = rs["episode_uuid"].duplicated().sum()
    env_overlap = rs.groupby("environment_id")["episode_uuid"].count()
    scene_op = rs.groupby(["scene_id", "operator_id"]).size()
    meta = {
        "n_records": int(len(rs)),
        "duplicate_uuids": int(dup),
        "n_environments": int(rs["environment_id"].nunique()),
        "n_scenes": int(rs["scene_id"].nunique()),
        "n_operators": int(rs["operator_id"].nunique()),
        "records_per_environment_max": int(env_overlap.max()),
        "split_counts": rs["split"].value_counts().to_dict(),
        "scene_operator_pairs": int(len(scene_op)),
        "total_frames": int(df["n_frames"].sum()),
        "total_duration_s": round(float(df["duration_s"].sum()), 1),
    }
    json.dump(meta, open(OUT / "audit_meta.json", "w"), indent=2)
    return df, meta


def interval_frame_features():
    ss, rs = index_tables()
    rec_env = rs.set_index("episode_uuid")[["environment_id", "split", "dataset", "scene_id", "operator_id"]]
    rows = []
    for _, r in ss.iterrows():
        uuid = r["episode_uuid"]
        if not (ROOT / "round2" / "cache_p12" / f"{uuid}.npz").exists():
            continue
        t, L, R, bad, reason, segs = open_record(uuid)
        m = (t >= r["start_s"]) & (t <= r["end_s"])
        if m.sum() < 5:
            continue
        tt = t[m]
        cL, cR, sL, sR, d = rep_point(L[m], R[m])
        vL = vel(cL, tt)
        vR = vel(cR, tt)
        path = np.sum(np.linalg.norm(np.diff(cL, axis=0), axis=1))
        disp = np.linalg.norm(cL[-1] - cL[0])
        acc = np.gradient(vL, tt)
        seg_idx = int(r["source_segment_index"])
        stage_pos = seg_idx / max(len(segs) - 1, 1)
        bad_seg = bad[m]
        rows.append({
            "unit_id": r["unit_id"], "dataset": r["dataset"], "family_id": r["family_id"],
            "environment_id": r["environment_id"], "split": r["split"],
            "task": r["task"], "seg_idx": seg_idx, "stage_pos": round(stage_pos, 3),
            "duration_s": r["duration_s"],
            "path_length": round(float(path), 4),
            "displacement": round(float(disp), 4),
            "speed_mean": round(float(vL.mean()), 4),
            "speed_p90": round(float(np.percentile(vL, 90)), 4),
            "speed_max": round(float(vL.max()), 4),
            "speed_std": round(float(vL.std()), 4),
            "acc_mean_abs": round(float(np.abs(acc).mean()), 4),
            "spread_mean": round(float(sL.mean()), 4),
            "spread_std": round(float(sL.std()), 4),
            "hands_dist_mean": round(float(d.mean()), 4),
            "bad_ratio": round(float(bad_seg.mean()), 5),
            "speedR_mean": round(float(vR.mean()), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "interval_features.csv", index=False)
    return df


def kw_test(df, col, group):
    groups = [g[col].dropna().to_numpy() for _, g in df.groupby(group)]
    groups = [g for g in groups if len(g) > 3]
    if len(groups) < 2:
        return None
    H, p = st.kruskal(*groups)
    n = sum(len(g) for g in groups)
    eps2 = (H - len(groups) + 1) / (n - len(groups))
    return {"factor": group, "metric": col, "k": len(groups), "n": n,
            "H": round(float(H), 3), "p": float(p),
            "eps2": round(float(max(eps2, 0.0)), 4)}


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * (m - np.arange(m))
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def difference_analysis(df):
    feats = ["speed_mean", "speed_p90", "speed_max", "speed_std", "acc_mean_abs",
             "path_length", "displacement", "spread_mean", "spread_std",
             "hands_dist_mean", "bad_ratio"]
    rows = []
    for f in feats:
        for g in ["family_id", "dataset", "split"]:
            r = kw_test(df, f, g)
            if r:
                rows.append(r)
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p"].to_numpy())
    res.to_csv(OUT / "difference_kw.csv", index=False)

    post = []
    for f in ["speed_mean", "bad_ratio", "spread_mean"]:
        for g in ["family_id", "dataset"]:
            levels = sorted(df[g].dropna().unique())
            ps = []
            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    a = df[df[g] == levels[i]][f].dropna()
                    b = df[df[g] == levels[j]][f].dropna()
                    if len(a) > 3 and len(b) > 3:
                        u, p = st.mannwhitneyu(a, b, alternative="two-sided")
                        ps.append((f, g, levels[i], levels[j], p))
            if ps:
                ph = holm([x[4] for x in ps])
                for (f0, g0, l0, l1, _), q in zip(ps, ph):
                    post.append({"metric": f0, "group": g0, "a": l0, "b": l1, "p_holm": float(q)})
    pd.DataFrame(post).to_csv(OUT / "difference_posthoc.csv", index=False)
    return res


def icc_environment(df, col):
    g = df.groupby("environment_id")[col]
    means, nobs, allv = [], [], []
    for _, v in g:
        v = v.dropna().to_numpy()
        if len(v) > 1:
            means.append(v.mean())
            nobs.append(len(v))
            allv.append(v)
    if len(means) < 2:
        return np.nan
    grand = np.concatenate(allv).mean()
    msb = sum(n * (m - grand) ** 2 for n, m in zip(nobs, means)) / (len(means) - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in allv) / (sum(nobs) - len(means))
    if msb + msw == 0:
        return np.nan
    return float((msb - msw) / (msb + (len(nobs) / (len(nobs) - 1) - 1) * msw + msw))


def build_frame_dataset():
    ss, rs = index_tables()
    rec = rs.set_index("episode_uuid")
    chunks = []
    for uuid in rec.index:
        t, L, R, bad, reason, segs = open_record(uuid)
        if len(t) < 10:
            continue
        cL, cR, sL, sR, d = rep_point(L, R)
        vL = vel(cL, t)
        vR = vel(cR, t)
        dt = np.diff(t, prepend=t[0])
        spread = 0.5 * (sL + sR)
        r = rec.loc[uuid]
        seg_id = np.zeros(len(t), dtype=int)
        for k, (a, b, _) in enumerate(segs):
            seg_id[(t >= a) & (t < b)] = k + 1
        n_seg = max(len(segs), 1)
        for hand, c, v, sp in (("L", cL, vL, sL), ("R", cR, vR, sR)):
            chunks.append(pd.DataFrame({
                "episode_uuid": uuid,
                "environment_id": r["environment_id"],
                "split": r["split"],
                "dataset": r["dataset"],
                "scene_id": r["scene_id"],
                "hand": hand,
                "t": t,
                "v": v,
                "spread": sp,
                "dist": d,
                "x": c[:, 0], "y": c[:, 1], "z": c[:, 2],
                "dt": dt,
                "seg_id": seg_id,
                "seg_frac": seg_id / n_seg,
                "bad": bad.astype(int),
            }))
    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(OUT / "frames.parquet") if _has_pa() else df.to_pickle(OUT / "frames.pkl")
    return df


def _has_pa():
    try:
        import pyarrow
        return True
    except ImportError:
        return False


def reliability_model():
    try:
        df = pd.read_parquet(OUT / "frames.parquet")
    except Exception:
        df = pd.read_pickle(OUT / "frames.pkl")
    df = df.dropna(subset=["v", "spread", "dist"])
    envs = df[["environment_id", "split"]].drop_duplicates()
    tr_envs = envs[envs["split"] == "train"]["environment_id"].tolist()
    va_envs = envs[envs["split"] == "validation"]["environment_id"].tolist()
    te_envs = envs[envs["split"] == "test"]["environment_id"].tolist()

    feats = ["v", "spread", "dist", "dt", "seg_frac", "x", "y", "z"]
    Xall = df[feats + ["dataset", "hand", "seg_id"]].copy()
    Xall["dataset"] = Xall["dataset"].astype("category").cat.codes
    Xall["hand"] = (Xall["hand"] == "R").astype(int)
    X = Xall.to_numpy(np.float32)
    y = df["bad"].to_numpy(int)
    g = df["environment_id"].to_numpy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.model_selection import GroupKFold

    def metrics(yt, pp):
        out = {}
        out["auroc"] = roc_auc_score(yt, pp)
        out["auprc"] = average_precision_score(yt, pp)
        out["brier"] = brier_score_loss(yt, pp)
        base = yt.mean()
        out["ece"] = _ece(yt, pp)
        out["auprc_base"] = base
        return out

    def _ece(yt, pp, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        e = 0.0
        for i in range(bins):
            m = (pp >= edges[i]) & (pp < edges[i + 1] + (i == bins - 1) * 1e-9)
            if m.sum() > 0:
                e += m.mean() * abs(yt[m].mean() - pp[m].mean())
        return float(e)

    results = []
    calib = []
    Xtr, ytr = X[np.isin(g, tr_envs)], y[np.isin(g, tr_envs)]
    gtr = g[np.isin(g, tr_envs)]

    for name, make in [
        ("logistic_l1", lambda: LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                                   max_iter=2000, class_weight="balanced")),
        ("gbm", lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                                       max_leaf_nodes=31, l2_regularization=1.0,
                                                       class_weight="balanced",
                                                       random_state=0)),
    ]:
        gkf = GroupKFold(n_splits=5)
        for k, (a, b) in enumerate(gkf.split(Xtr, ytr, gtr)):
            m = make()
            m.fit(Xtr[a], ytr[a])
            pp = m.predict_proba(Xtr[b])[:, 1]
            mt = metrics(ytr[b], pp)
            results.append({"model": name, "fold": k, **mt})
            calib.append(pd.DataFrame({"model": name, "fold": k, "y": ytr[b], "p": pp}))
        m = make()
        m.fit(Xtr, ytr)
        for split_name, env_set in [("validation", va_envs), ("test", te_envs)]:
            msk = np.isin(g, env_set)
            if msk.sum() == 0:
                continue
            pp = m.predict_proba(X[msk])[:, 1]
            mt = metrics(y[msk], pp)
            results.append({"model": name, "fold": split_name, **mt})
            calib.append(pd.DataFrame({"model": name, "fold": split_name, "y": y[msk], "p": pp}))

    base_rate = float(ytr.mean())
    for split_name, env_set in [("train", tr_envs), ("validation", va_envs), ("test", te_envs)]:
        msk = np.isin(g, env_set)
        if msk.sum():
            results.append({"model": "constant_baseline", "fold": split_name,
                            "auroc": 0.5, "auprc": float(y[msk].mean()),
                            "brier": float(brier_score_loss(y[msk], np.full(msk.sum(), base_rate))),
                            "ece": np.nan, "auprc_base": float(y[msk].mean())})

    pd.DataFrame(results).to_csv(OUT / "reliability_metrics.csv", index=False)
    pd.concat(calib, ignore_index=True).to_csv(OUT / "reliability_calib.csv", index=False)

    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=0)
    m.fit(Xtr, ytr)
    imp = permutation_importance_sorted(m, X[np.isin(g, te_envs)], y[np.isin(g, te_envs)], feats)
    json.dump(imp, open(OUT / "reliability_importance.json", "w"), indent=2)
    return pd.DataFrame(results)


def permutation_importance_sorted(model, X, y, names, n=8, seed=0):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for j, nm in enumerate(names):
        drops = []
        for _ in range(n):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[nm] = round(float(np.mean(drops)), 5)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    df, meta = audit()
    print("audit:", meta)
    iv = interval_frame_features()
    print("interval features:", iv.shape)
    res = difference_analysis(iv)
    print(res.sort_values("p").head(8).to_string())
    iccs = {c: round(icc_environment(iv, c), 3) for c in ["speed_mean", "bad_ratio", "spread_mean"]}
    json.dump(iccs, open(OUT / "icc_environment.json", "w"), indent=2)
    print("ICC:", iccs)
    build_frame_dataset()
    print("frame dataset done")
    reliability_model()
    print("reliability done")
