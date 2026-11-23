"""特征提取：把每个源区间的 0.2/0.6/1.0 秒数值前缀转成监督学习特征。

合规要点（题目第2节）：
- 只用前缀窗口内 [start_s, start_s+w] 的已存在样本，不外推、不插值、不填充。
- 特征中不含源标识、区间端点、总时长、语义描述和后续样本。
- 数据划分按 environment_id 整体分组（train/validation/test），由 source_index 的 split 列给定。
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from config import CONSTRAINTS, RAW, DATA, PREFIXES

IDX = CONSTRAINTS / "source_segment_index.csv"

# 三个数据集的原始根目录：EgoDemo 已解压在 RAW；另两个在 data/raw/<name>
DATASET_DIR = {
    "EgoDemo": RAW,
    "EgoPro": DATA / "raw" / "EgoPro",
    "EgoStandard": DATA / "raw" / "EgoStandard",
}

NUM_COLS = [
    "observation.state.hand_left_world",
    "observation.state.hand_right_world",
    "observation.state.hand_left_world_rotation",
    "observation.state.hand_right_world_rotation",
    "observation.state.head_world",
    "observation.state.head_world_rotation",
    "observation.state.head_left_camera_position",
    "observation.state.head_left_camera_rotation",
    "observation.state.head_right_camera_position",
    "observation.state.head_right_camera_rotation",
]

def load_episode(sample: str, dataset: str):
    """返回该 episode 的 dict: timestamp(1D) 与各数值通道 ndarray。"""
    root = DATASET_DIR[dataset]
    base = root / sample
    # 兼容两种布局: <root>/<sample>/data/chunk-000/file-000.parquet  或 <root>/data/...
    cands = [
        base / "data" / "chunk-000" / "file-000.parquet",
        root / "data" / "chunk-000" / f"{sample}.parquet",
        base / "file-000.parquet",
    ]
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        # 递归找该 sample 下第一个 parquet
        hits = list(base.rglob("*.parquet")) if base.exists() else []
        if hits:
            p = sorted(hits)[0]
    if p is None:
        return None
    t = pq.read_table(p)
    ts = np.asarray(t["timestamp"].to_numpy(), dtype=float).reshape(-1)
    out = {"timestamp": ts, "n": len(ts)}
    for c in NUM_COLS:
        arr = np.asarray(t[c].to_pylist(), dtype=float)  # (n, k, d)
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
        out[c] = arr
    return out

def describe(x, prefix):
    """对一维数组给出稳健统计量。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
                f"{prefix}_max": 0.0, f"{prefix}_p90": 0.0}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
    }

def motion_features(epi, lo, hi):
    """用 [lo,hi] 秒窗口内的帧计算运动学特征。"""
    ts = epi["timestamp"]
    mask = (ts >= lo) & (ts <= hi)
    idxs = np.where(mask)[0]
    f = {}
    n = len(idxs)
    f["n_frames"] = n
    f["span_s"] = float(ts[idxs[-1]] - ts[idxs[0]]) if n >= 2 else 0.0
    if n < 2:
        return f, n

    L = epi["observation.state.hand_left_world"][idxs]   # (n,21,3)
    R = epi["observation.state.hand_right_world"][idxs]
    # 逐帧位移速度 (用实际时间差)
    dt = np.diff(ts[idxs])
    dt = np.where(dt <= 0, 1/30.0, dt)
    vL = np.linalg.norm(np.diff(L, axis=0), axis=2) / dt[:, None]
    vR = np.linalg.norm(np.diff(R, axis=0), axis=2) / dt[:, None]
    f.update(describe(vL.mean(axis=1), "vL"))
    f.update(describe(vR.mean(axis=1), "vR"))
    f.update(describe(np.concatenate([vL.ravel(), vR.ravel()]), "vall"))

    # 手腕(第0点)速度与加速度
    wL = L[:, 0, :]; wR = R[:, 0, :]
    swL = np.linalg.norm(np.diff(wL, axis=0), axis=1) / dt
    swR = np.linalg.norm(np.diff(wR, axis=0), axis=1) / dt
    f.update(describe(swL, "wristL"))
    f.update(describe(swR, "wristR"))
    if len(swL) >= 2:
        aL = np.abs(np.diff(swL)) / dt[1:]
        aR = np.abs(np.diff(swR)) / dt[1:]
        f.update(describe(np.concatenate([aL, aR]), "accel"))

    # 双手间距与变化
    dist = np.linalg.norm(wL - wR, axis=1)
    f.update(describe(dist, "handsep"))
    f["handsep_range"] = float(np.ptp(dist))

    # 手部姿态张开的程度：指尖到手腕平均距离
    tip = [4, 8, 12, 16, 20]
    spreadL = np.linalg.norm(L[:, tip, :] - L[:, [0], :], axis=2).mean(axis=1)
    spreadR = np.linalg.norm(R[:, tip, :] - R[:, [0], :], axis=2).mean(axis=1)
    f.update(describe(spreadL, "spreadL"))
    f.update(describe(spreadR, "spreadR"))

    # 头部/相机运动
    h = epi["observation.state.head_world"][idxs]
    if h.ndim == 2:
        sh = np.linalg.norm(np.diff(h, axis=0), axis=1) / dt
        f.update(describe(sh, "head"))
    for c, key in [("observation.state.head_left_camera_position", "camL"),
                   ("observation.state.head_right_camera_position", "camR")]:
        cam = epi[c][idxs]
        if cam.ndim == 2:
            sv = np.linalg.norm(np.diff(cam, axis=0), axis=1) / dt
            f.update(describe(sv, key))

    # 四元数角速度近似（相邻帧四元数差）
    for c, key in [("observation.state.hand_left_world_rotation", "qL"),
                   ("observation.state.hand_right_world_rotation", "qR")]:
        q = epi[c][idxs]  # (n,21,4)
        dq = np.linalg.norm(np.diff(q, axis=0), axis=2) / dt[:, None]
        f.update(describe(dq.mean(axis=1), key))
    return f, n

def main(only_dataset=None):
    rows = list(csv.DictReader(open(IDX, encoding="utf-8-sig")))
    if only_dataset:
        rows = [r for r in rows if r["dataset"] == only_dataset]
    out = []
    cache = {}
    for r in rows:
        if r["eligible"] != "1":
            continue
        ds = r["dataset"]; sample = r["sample"]
        cache_key = (ds, sample)
        if cache_key not in cache:
            cache[cache_key] = load_episode(sample, ds)
        epi = cache[cache_key]
        rec = {
            "unit_id": r["unit_id"], "dataset": ds, "sample": sample,
            "environment_id": r["environment_id"],
            "split": r["split"], "family_id": r["family_id"],
            "source_segment_index": int(r["source_segment_index"]),
            "start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
            "duration_s": float(r["duration_s"]),
            "available": epi is not None,
        }
        if epi is not None:
            ts = epi["timestamp"]
            for w in PREFIXES:
                lo = float(r["start_s"])
                hi = lo + w
                feats, n = motion_features(epi, lo, hi)
                feats = {f"p{int(w*10):02d}_{k}": v for k, v in feats.items()}
                rec.update(feats)
        out.append(rec)
    import pandas as pd
    df = pd.DataFrame(out)
    dest = DATA / ("prefix_features_%s.csv" % only_dataset if only_dataset else "prefix_features.csv")
    df.to_csv(dest, index=False)
    print("saved", dest, "rows", len(df))
    print("available:", df["available"].sum(), "/", len(df))
    print("by dataset:", df["dataset"].value_counts().to_dict())
    print("by split:", df["split"].value_counts().to_dict())
    if not df.empty:
        print("feature cols:", len([c for c in df.columns if c.startswith("p")]))

if __name__ == "__main__":
    main()
