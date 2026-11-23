
import sys

import pandas as pd

from common import RESULTS

def main():
    main_csv = RESULTS / "q7_ablation.csv"
    fix_csv = RESULTS / "q7_ablation_rm_fixed.csv"
    df = pd.read_csv(main_csv)
    df.columns = [c.strip("\ufeff") for c in df.columns]
    fix = pd.read_csv(fix_csv)
    fix.columns = [c.strip("\ufeff") for c in fix.columns]

    assert len(fix) == 450, f"修复版 RM 行数 {len(fix)} != 450"
    assert (fix["valid_protocol"] == 1).all(), "修复版 RM 存在协议违规局"

    out = df[df.arm != "RM_buytop"]
    out = pd.concat([out, fix[out.columns]], ignore_index=True)
    assert len(out) == 5400, f"合并后行数 {len(out)} != 5400"
    out.to_csv(main_csv, index=False, encoding="utf-8-sig")
    print(f"MERGED: RM 行替换完成，总行数 {len(out)} -> {main_csv}")

if __name__ == "__main__":
    sys.exit(main())
