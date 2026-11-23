"""EgoPro / EgoStandard 数据集特征提取（CSV 表格格式）。

与 EgoDemo(parquet) 的异构性：
- 时间戳列：_offset_from_first_log_s
- 手部21关键点：pose_left_hand.csv / pose_right_hand.csv，
  每点为 transforms[i].pos.{x,y,z} 与 transforms[i].quat.{w,x,y,z}
- 头部：pose_head.csv（1点）；头相机：pose_headcam.csv（1点）
- 语义区间：annotation_semantic_segments.csv 的 segment.start_time / end_time
- 质量标志：annotation_bad_frame_pose_hand.csv 的 is_bad

源区间通过 (dataset, sample=session_uuid, start_s, end_s) 与 source_segment_index 对齐。
"""
import csv, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from config import CONSTRAINTS, DATA, PREFIXES
from extract_features import describe

IDX = CONSTRAINTS / "source_segment_index.csv"
RAW = DATA / "raw"

def read_pose_csv(path, n_points):
    df = pd.read_csv(path)
    t = df["_offset_from_first_log_s"].to_numpy(float)
    pos_cols = []
    for i in range(n_points):
        pos_cols += [f"transforms[{i}].pos.x", f"transforms[{i}].pos.y", f"transforms[{i}].pos.z"]
    pos = df[pos_cols].to_numpy(float).reshape(len(df), n_points, 3)
    quat_cols = []
    for i in range(n_points):
        quat_cols += [f"transforms[{i}].quat.w", f"transforms[{i}].quat.x",
                      f"transforms[{i}].quat.y", f"transforms[{i}].quat.z"]
    quat = df[quat_cols].to_numpy(float).reshape(len(df), n_points, 4)
    bad = None
    return t, pos, quat

def load_session(dataset, session):
    base = RAW / dataset / session
    tabs = base / "tables"
    if not (tabs / "pose_left_hand.csv").exists():
        return None
    t, L, qL = read_pose_csv(tabs / "pose_left_hand.csv", 21)
    _, R, qR = read_pose_csv(tabs / "pose_right_hand.csv", 21)
    _, H, qH = read_pose_csv(tabs / "pose_head.csv", 1)
    H = H[:, 0, :]; qH = qH[:, 0, :]
    cam = None
    if (tabs / "pose_headcam.csv").exists():
        _, C, qC = read_pose_csv(tabs / "pose_headcam.csv", 1)
        cam = C[:, 0, :]
    # 质量标志
    bad = None
    if (tabs / "annotation_bad_frame_pose_hand.csv").exists():
        bd = pd.read_csv(tabs / "annotation_bad_frame_pose_hand.csv")
        bad = bd["is_bad"].astype(str).str.lower().eq("true").to_numpy()
    return dict(t=t, L=L, R=R, qL=qL, qR=qR, H=H, qH=qH, cam=cam, bad=bad)

def features_from_window(epi, lo, hi):
    t = epi["t"]
    m = (t >= lo) & (t <= hi)
    idxs = np.where(m)[0]
    f = {}
    n = len(idxs)
    f["n_frames"] = n
    f["span_s"] = float(t[idxs[-1]] - t[idxs[0]]) if n >= 2 else 0.0
    if n < 2:
        return f
    L = epi["L"][idxs]; R = epi["R"][idxs]
    dt = np.diff(t[idxs]); dt = np.where(dt <= 0, 1/30.0, dt)
    vL = np.linalg.norm(np.diff(L, axis=0), axis=2) / dt[:, None]
    vR = np.linalg.norm(np.diff(R, axis=0), axis=2) / dt[:, None]
    f.update(describe(vL.mean(axis=1), "vL"))
    f.update(describe(vR.mean(axis=1), "vR"))
    f.update(describe(np.concatenate([vL.ravel(), vR.ravel()]), "vall"))
    wL = L[:, 0, :]; wR = R[:, 0, :]
    swL = np.linalg.norm(np.diff(wL, axis=0), axis=1) / dt
    swR = np.linalg.norm(np.diff(wR, axis=0), axis=1) / dt
    f.update(describe(swL, "wristL")); f.update(describe(swR, "wristR"))
    if len(swL) >= 2:
        aL = np.abs(np.diff(swL)) / dt[1:]; aR = np.abs(np.diff(swR)) / dt[1:]
        f.update(describe(np.concatenate([aL, aR]), "accel"))
    sep = np.linalg.norm(wL - wR, axis=1)
    f.update(describe(sep, "handsep")); f["handsep_range"] = float(np.ptp(sep))
    tip = [4, 8, 12, 16, 20]
    spreadL = np.linalg.norm(L[:, tip, :] - L[:, [0], :], axis=2).mean(axis=1)
    spreadR = np.linalg.norm(R[:, tip, :] - R[:, [0], :], axis=2).mean(axis=1)
    f.update(describe(spreadL, "spreadL")); f.update(describe(spreadR, "spreadR"))
    h = epi["H"][idxs]
    sh = np.linalg.norm(np.diff(h, axis=0), axis=1) / dt
    f.update(describe(sh, "head"))
    if epi["cam"] is not None:
        cam = epi["cam"][idxs]
        sv = np.linalg.norm(np.diff(cam, axis=0), axis=1) / dt
        f.update(describe(sv, "camL")); f.update(describe(sv, "camR"))
    for q, key in [(epi["qL"], "qL"), (epi["qR"], "qR")]:
        q = q[idxs]
        dq = np.linalg.norm(np.diff(q, axis=0), axis=2) / dt[:, None]
        f.update(describe(dq.mean(axis=1), key))
    # 质量标志坏帧比例
    if epi["bad"] is not None:
        f["bad_ratio"] = float(np.mean(epi["bad"][idxs]))
    return f

def main(only_dataset=None):
    rows = list(csv.DictReader(open(IDX, encoding="utf-8-sig")))
    out = []
    cache = {}
    for r in rows:
        if r["eligible"] != "1":
            continue
        ds = r["dataset"]
        if ds not in ("EgoPro", "EgoStandard"):
            continue
        if only_dataset and ds != only_dataset:
            continue
        session = r["sample"]
        key = (ds, session)
        if key not in cache:
            cache[key] = load_session(ds, session)
        epi = cache[key]
        rec = dict(unit_id=r["unit_id"], dataset=ds, sample=session,
                   environment_id=r["environment_id"], split=r["split"],
                   family_id=r["family_id"],
                   source_segment_index=int(r["source_segment_index"]),
                   start_s=float(r["start_s"]), end_s=float(r["end_s"]),
                   duration_s=float(r["duration_s"]), available=epi is not None)
        if epi is not None:
            lo = float(r["start_s"])
            for w in PREFIXES:
                feats = features_from_window(epi, lo, lo + w)
                rec.update({f"p{int(w*10):02d}_{k}": v for k, v in feats.items()})
        out.append(rec)
    df = pd.DataFrame(out)
    name = only_dataset or "EgoPro_EgoStandard"
    dest = DATA / f"prefix_features_{name}.csv"
    df.to_csv(dest, index=False)
    print("saved", dest, "rows", len(df), "available", int(df["available"].sum()))
    print("by split:", df["split"].value_counts().to_dict())

if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else None
    main(a)
