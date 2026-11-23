import zipfile, io, os
from pathlib import Path

OUTER = r"C:/Users/Camevian/Downloads/C_Problem_Data/Restocking_Problem_and_Data_EN.zip"
PWD = "SwbMcM2612&2qP7!rTz@5bN8#vG3$dF"
TARGET = Path(r"C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37/data/raw")
TARGET.mkdir(parents=True, exist_ok=True)

zf = zipfile.ZipFile(OUTER)
zf.setpassword(PWD.encode())

for key in ["EgoPro", "EgoStandard"]:
    member = f"Restocking_Problem_and_Data_EN/Raw_Data/{key}.zip"
    print("读取外层条目:", key)
    data = zf.read(member)          # 用外层密码解密
    inner = zipfile.ZipFile(io.BytesIO(data))
    # 内层再尝试同一密码
    try:
        inner.setpassword(PWD.encode())
        probe = [n for n in inner.namelist() if not n.endswith("/")][0]
        inner.open(probe).read(16)
        print("  内层密码可用, 条目数", len(inner.namelist()))
    except Exception as e:
        print("  内层无需密码或密码不同:", repr(e)[:100])
    # 只解压数值 parquet 与 json，跳过 mp4
    keep = [n for n in inner.namelist()
            if not n.endswith("/") and (n.endswith(".parquet") or n.endswith(".json"))]
    print("  需解压数值/json文件数:", len(keep))
    dest = TARGET / key
    dest.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(keep):
        inner.setpassword(PWD.encode())
        content = inner.read(n)
        out = dest / Path(n).name
        # 保留相对子目录以匹配 sample->目录映射
        rel = Path(n)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(keep)}")
    print("  完成", key, "->", dest)

print("ALL_DONE")
