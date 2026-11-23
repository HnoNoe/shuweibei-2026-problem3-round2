"""阶段1：用外层密码把两个加密的内层 zip 落盘（各只读一次大加密流）。"""
import zipfile, time
from pathlib import Path

OUTER = r"C:/Users/Camevian/Downloads/C_Problem_Data/Restocking_Problem_and_Data_EN.zip"
PWD = "SwbMcM2612&2qP7!rTz@5bN8#vG3$dF"
STAGE = Path(r"C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37/data/stage")
STAGE.mkdir(parents=True, exist_ok=True)

zf = zipfile.ZipFile(OUTER)
zf.setpassword(PWD.encode())
for key in ["EgoPro", "EgoStandard"]:
    member = f"Restocking_Problem_and_Data_EN/Raw_Data/{key}.zip"
    out = STAGE / f"{key}.zip"
    if out.exists() and out.stat().st_size > 1_000_000_000:
        print(key, "already staged", out.stat().st_size)
        continue
    t0 = time.time()
    print("extracting", key, "from outer...")
    with zf.open(member) as src, open(out, "wb") as dst:
        total = 0
        while True:
            buf = src.read(8 * 1024 * 1024)
            if not buf:
                break
            dst.write(buf)
            total += len(buf)
            if total // (200 * 1024 * 1024) > (total - len(buf)) // (200 * 1024 * 1024):
                print(f"  {total/1e9:.2f} GB  {time.time()-t0:.0f}s")
    print(key, "done", out.stat().st_size, f"{time.time()-t0:.0f}s")
print("STAGE_DONE")
