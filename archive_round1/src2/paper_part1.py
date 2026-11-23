# -*- coding: utf-8 -*-
"""论文正文内容（封面/目录/正文/参考文献/附录）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from paper_base import (ROOT, FIG, OUT, SONG, HEI, TIMES, TEAM, para, h1, h2,
                        h3, figure, table, formula, field,
                        body_header_footer, enable_update_fields, set_run)

RES = ROOT / "results2"
SRC = ROOT / "src2"

doc = Document()

# 页面设置 A4
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.54)
sec.left_margin = sec.right_margin = Cm(2.6)

# 内置标题样式去蓝色
for sname in ("Heading 1", "Heading 2", "Heading 3"):
    st = doc.styles[sname]
    st.font.color.rgb = None

# ============ 摘要专用页 ============
para(doc, "", indent_chars=0, after=18)
para(doc, "数维杯大学生数学建模挑战赛论文", cn=HEI, size=22,
     align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0, after=28)
para(doc, "题  目：基于第一人称演示数据的补货作业时长估计与随机调度决策",
     cn=HEI, size=16, align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
     after=20)
para(doc, "摘  要", cn=HEI, size=16, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0, before=10, after=10)

abstract = (
    "本文研究基于第一人称演示数据的补货作业时长估计与随机调度决策问题：在单设备、预算与审查受限"
    "下，需估计作业时长及其不确定性，并在线决定作业选择、顺序、预留、审查与等待/停止。针对问题"
    "一，对 343 个合格源区间做分布诊断，时长重右尾（1.2~125 s）；并以分组交叉验证检验"
    "前缀对时长的预测力，岭回归与梯度提升树的 R² 均低于 family 均值基线，最强单特征相关系数"
    "仅 0.238，说明前缀几乎不含时长信息，据此建立 family 分层对数正态模型（族间差异 "
    "p=0.0012），建模重心由点预测转向不确定性量化。针对问题二，证明两个解析结论：完成"
    "时刻 C=s+p 与预留 b 无关而费用与空闲随 b 递增，且 p>0.2 s 恒成立，故最小预留 b=0.2 "
    "弱占优；前缀无预测力使审查的信息价值约等于零，最优策略不买审查。排序采用 ATC-travel "
    "指数（综合权重、期望松弛与有效加工时长），叠加截断防护与预算内价值密度取舍；"
    "超参数仅在验证集名义映射上调定（κ=4.0，q=0.90）后冻结。验证集 450 次 trial 上主策略"
    "加权准时率 0.876，较基线 WSPT 配对差异 +0.0588（95%CI 不含 0），协议合规率 1.0，三案例"
    "均满足不低于 0.5 的约束。针对问题三，扫描预留分位给出准时率—成本—超时占用的权衡曲线"
    "（中位预留使超时占用降至 29.5 s）；五组时长映射下性能单调退化且合规率保持 1.0；"
    "测试记录上的模拟终测验证了冻结策略的泛化能力。"
)
para(doc, abstract, size=12, line=1.4, after=10)
para(doc, "", indent_chars=0, after=6)
p = para(doc, indent_chars=0, line=1.4)
set_run(p.add_run("关键词："), HEI, TIMES, 14, bold=True)
set_run(p.add_run("随机调度；不确定性量化；ATC 优先指数；单设备排序；样本平均近似"),
        SONG, TIMES, 12)

# ============ 目录页 ============
doc.add_page_break()
para(doc, "目  录", cn=HEI, size=16, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0, before=12, after=12)
ptoc = para(doc, indent_chars=0, line=1.4)
field(ptoc, r'TOC \o "1-3" \h \z \u')
para(doc, "（在 Word 中打开时会自动更新目录；如未显示，请全选后按 F9 更新域）",
     size=9, indent_chars=0, align=WD_ALIGN_PARAGRAPH.CENTER)

# ============ 正文节（页眉页脚、页码从 1 起）============
doc.add_section(WD_SECTION.NEW_PAGE)
body_header_footer(doc.sections[-1])
