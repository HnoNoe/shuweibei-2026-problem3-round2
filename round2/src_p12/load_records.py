import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EGO_DEMO = Path(r"C:/Users/Camevian/Downloads/C_Problem_Data/Restocking_Problem_and_Data_EN/Restocking_Problem_and_Data_EN/Raw_Data/EgoDemo/EgoDemo")
EGO_TAB = Path(r"C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37/data/raw")
ROOT = Path(r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c")
CACHE = ROOT / "round2" / "cache_p12"
CACHE.mkdir(parents=True, exist_ok=True)

FPS = 30.0


def load_egodemo(sample):
    base = EGO_DEMO / sample
    pq = base / "data" / "chunk-000" / "file-000.parquet"
    import duckdb
    df = duckdb.sql(f'select * from "{pq}"').df()
    ts = np.asarray(df["timestamp"], dtype=np.float64)
    t = ts - ts[0]
    if t[-1] > 1e6:
        t = t / 1e9
    def hand(col):
        s = df[col]
        out = np.stack([np.stack(list(v)) for v in s]).astype(np.float32)
        return out
    L = hand("observation.state.hand_left_world")
    R = hand("observation.state.hand_right_world")
    n = len(df)
    is_bad = np.zeros(n, dtype=bool)
    reason = np.array(["ok"] * n, dtype=object)
    bf = json.load(open(base / "bad_frame_ratio.json", encoding="utf-8"))
    for d in bf.get("hand_bad_frame_details", []):
        f = int(d["frame"])
        if 0 <= f < n:
            is_bad[f] = True
            lr = d.get("left_bad_reason") or d.get("right_bad_reason") or "bad"
            reason[f] = lr
    ann = json.load(open(base / "annotation.json", encoding="utf-8"))
    segs = [(float(s["start_time"]), float(s["end_time"]), s.get("description", ""))
            for s in ann.get("subtask_annotations", [])]
    return t, L, R, is_bad, reason, segs


def _csv(uuid, dataset, name):
    return pd.read_csv(EGO_TAB / dataset / uuid / "tables" / name)


def load_egox(uuid, dataset):
    pl = _csv(uuid, dataset, "pose_left_hand.csv")
    pr = _csv(uuid, dataset, "pose_right_hand.csv")
    t = np.asarray(pl["_offset_from_first_log_s"], dtype=np.float64)
    def hand(df):
        cols = {}
        for i in range(21):
            for axis in "xyz":
                cols[(i, axis)] = np.asarray(df[f"transforms[{i}].pos.{axis}"], dtype=np.float32)
        out = np.stack([np.stack([cols[(i, a)] for a in "xyz"], axis=-1) for i in range(21)], axis=1)
        return out
    L = hand(pl)
    R = hand(pr)
    n = len(t)
    is_bad = np.zeros(n, dtype=bool)
    reason = np.array(["ok"] * n, dtype=object)
    try:
        bf = _csv(uuid, dataset, "annotation_bad_frame_pose_hand.csv")
        bad = np.asarray(bf["is_bad"], dtype=bool)
        m = min(n, len(bad))
        is_bad[:m] = bad[:m]
        pt = bf.get("problem_type.length")
        if pt is not None:
            pt = np.asarray(pt.astype(str))
            reason[:m] = np.where(bad[:m], np.where(pt[:m] == "0", "occlusion", pt[:m]), "ok")
        else:
            reason[:m] = np.where(bad[:m], "bad", "ok")
    except FileNotFoundError:
        pass
    segs = []
    try:
        sg = _csv(uuid, dataset, "annotation_semantic_segments.csv")
        for _, r in sg.iterrows():
            segs.append((float(r["segment.start_time"]), float(r["segment.end_time"]),
                         str(r.get("segment.subtask_description", ""))))
    except FileNotFoundError:
        pass
    return t, L, R, is_bad, reason, segs


def index_tables():
    ss = pd.read_csv(ROOT / "problem" / "source_segment_index.csv")
    rs = pd.read_csv(ROOT / "problem" / "record_split.csv")
    return ss, rs


def load_all(force=False):
    ss, rs = index_tables()
    out = []
    for _, r in rs.iterrows():
        uuid = r["episode_uuid"]
        dataset = r["dataset"]
        cache = CACHE / f"{uuid}.npz"
        if cache.exists() and not force:
            out.append(uuid)
            continue
        try:
            sub = ss[ss["episode_uuid"] == uuid]
            if dataset == "EgoDemo":
                sample = sub["sample"].iloc[0]
                t, L, R, bad, reason, segs = load_egodemo(sample)
            else:
                sample = sub["sample"].iloc[0]
                t, L, R, bad, reason, segs = load_egox(sample, dataset)
            np.savez_compressed(cache, t=t, L=L, R=R, is_bad=bad,
                                reason=reason.astype(str),
                                segs_json=json.dumps(segs))
            out.append(uuid)
            print("OK", dataset, uuid[:8], len(t))
        except Exception as e:
            print("FAIL", dataset, uuid[:8], repr(e)[:120])
    return out


def open_record(uuid):
    z = np.load(CACHE / f"{uuid}.npz", allow_pickle=True)
    segs = [(float(a), float(b), str(c)) for a, b, c in json.loads(str(z["segs_json"]))]
    return z["t"], z["L"], z["R"], z["is_bad"], z["reason"], segs


def rep_point(L, R):
    cL = L.mean(axis=1)
    cR = R.mean(axis=1)
    spreadL = np.linalg.norm(L - cL[:, None, :], axis=2).mean(axis=1)
    spreadR = np.linalg.norm(R - cR[:, None, :], axis=2).mean(axis=1)
    d = np.linalg.norm(cL - cR, axis=1)
    return cL, cR, spreadL, spreadR, d


if __name__ == "__main__":
    load_all(force="--force" in sys.argv)
