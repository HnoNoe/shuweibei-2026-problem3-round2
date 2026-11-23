# -*- coding: utf-8 -*-
"""构建论文：按顺序执行四个内容脚本（共享命名空间）。"""
import os
here = os.path.dirname(os.path.abspath(__file__))
for part in ("paper_part1.py", "paper_part2.py", "paper_part3.py",
             "paper_part4.py"):
    exec(compile(open(os.path.join(here, part), encoding="utf-8").read(),
                 part, "exec"))
