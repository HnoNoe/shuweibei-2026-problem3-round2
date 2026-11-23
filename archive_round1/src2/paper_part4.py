# -*- coding: utf-8 -*-
"""第六章、参考文献、附录与保存。"""

# ================= 六、模型评价及优化 =================
h1(doc, "六、模型评价及优化")

h2(doc, "6.1 误差分析")
para(doc,
     "问题一（时长模型）的误差主要来自分布近似的设定误差。族内 log d 仍呈右偏（偏度 0.72~1.28），"
     "严格单峰的对数正态会低估右尾，表现为中位数覆盖率偏低（验证 0.43、测试 0.37，低于名义 0.50），"
     "但策略实际使用的 Q90 分位覆盖率在验证与测试上分别为 0.886 与 0.968，接近名义水平且偏保守"
     "（高估时长），对截断防护是安全的偏差方向。σ_f 的下限 0.35 与 L 族仅 10 个训练样本是该层"
     "模型最大的不确定性来源。问题二（策略）的误差以 trial 均值的抽样波动为主：名义映射下每案例 "
     "30 次 trial，准时率的 95% 置信区间半宽约 ±0.02~0.04；主策略相对基线的配对差异置信区间"
     " [+0.041,+0.076] 不含 0，提升是统计显著的。验证集到测试集的准时率差距（如 S1 的 "
     "0.963→0.900）来自环境间时长构成差异，属数据分层内噪声而非策略失效。问题三（权衡与敏感性）"
     "的各映射结论基于 90 次 trial/映射，单调趋势在基线与主策略上一致出现，可信度高。")

h2(doc, "6.2 模型的优点")
para(doc,
     "其一，结论有解析支撑：最小预留弱占优（命题 1）与审查零价值（5.1.2 的分组交叉验证）都是可"
     "证明、可复现的论断，策略由此被压缩到排序单一维度，极大地降低了过拟合空间。其二，策略设计"
     "克制而有效：ATC-travel 指数仅引入 κ 一个实质超参数，配合两个有明确机理的风险机制，在验证"
     "集上取得 +5.9 个百分点的显著提升且对参数邻域不敏感。其三，评估严格：全部结论基于共享源分配"
     "的配对比较与 bootstrap 置信区间，450 次 trial 协议合规率 1.0，并通过五组映射与测试环境"
     "双重稳健性检验。其四，过程可审计：每次执行的源区间引用、时间、费用与占用均写入日志，满足"
     "赛题对合规核查的要求。")

h2(doc, "6.3 模型的缺点")
para(doc,
     "其一，排序规则是指数族的，不包含对前序解锁结构、释放时刻耦合的全局优化，SAA rollout 对照"
     "虽未带来增益，但也说明全局随机优化空间尚未被充分挖掘。其二，时长模型为静态分层先验，未利用"
     "trial 内已完成作业反馈更新评估（协议允许用执行反馈更新当前评估，本文保守地未使用）。"
     "其三，截断防护分位 q 与预算取舍密度系数在验证集上调定，虽做了邻域敏感性检查，仍存在对验证"
     "环境的轻度依赖。其四，对数正态尾部设定偏薄，极端长时长区间的风险被 Q90 防护部分掩盖但未"
     "被显式建模。")

h2(doc, "6.4 模型的推广")
para(doc,
     "模型可沿三个方向推广。第一，信息价值通式化：把“前缀→预测力”检验推广为任意信息购买动作的"
     "期望价值上界，可在保留审查机制的变体题目中自动决定买不买、买几次。第二，把族分层先验升级为"
     "贝叶斯在线更新：每个 trial 内利用已执行作业的真实时长逐步更新族参数（符合协议的反馈更新"
     "条款），可望缓解静态先验对环境漂移的迟钝。第三，排序层面可用两阶段随机规划或强化学习替代"
     "指数规则：以冻结的 ATC-travel 为基准策略，用策略梯度在其邻域内搜索，兼顾可解释性与最优性。"
     "此外，命题 1 的占优性论证与协议状态机实现均可直接迁移到其他具有“预留不释放、按时计费”"
     "特征的调度场景（如算力预留、会议室预订）。")

# ================= 参考文献 =================
doc.add_page_break()
h1(doc, "参考文献")
refs = [
    "[1] Pinedo M L. Scheduling: Theory, Algorithms, and Systems[M]. 5th ed. "
    "New York: Springer, 2016: 35-96.",
    "[2] Vepsalainen A P J, Morton T E. Priority rules for job shops with "
    "weighted tardiness costs[J]. Management Science, 1987, 33(8): 1035-1047.",
    "[3] Birge J R, Louveaux F. Introduction to Stochastic Programming[M]. "
    "2nd ed. New York: Springer, 2011: 1-60.",
    "[4] Bertsekas D P. Rollout, Policy Iteration, and Distributed "
    "Reinforcement Learning[M]. Belmont: Athena Scientific, 2020: 1-45.",
    "[5] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine "
    "learning in Python[J]. Journal of Machine Learning Research, 2011, "
    "12: 2825-2830.",
    "[6] 姜启源, 谢金星, 叶俊. 数学模型（第五版）[M]. 北京: 高等教育出版社, "
    "2018: 1-30.",
]
for r in refs:
    p = para(doc, r, size=10.5, line=1.3, indent_chars=0, after=3)
    p.paragraph_format.left_indent = Pt(21)
    p.paragraph_format.first_line_indent = Pt(-21)

# ================= 附录 =================
doc.add_page_break()
h1(doc, "附  录")
para(doc, "本附录列出支撑论文的全部核心源程序（Python 3，依赖 numpy/pandas/scipy/scikit-learn/"
          "matplotlib）。完整工程（含全部结果 CSV、过程日志样例与运行说明）随支撑材料一并提交。"
          "代码结构：common.py（常量与数据装载）、simulator.py（协议状态机）、sampler.py"
          "（源区间采样）、duration.py（时长模型）、policy.py（调度策略）、analyze_duration.py"
          "（可预测性研究）、tune_eval.py（调参与评估）、tradeoff.py（b 权衡扫描）、"
          "make_figures.py（绘图）。复现命令：python test_simulator2.py；"
          "python tune_eval.py 1/2/3；python tradeoff.py。", size=10.5,
     line=1.3, indent_chars=0)


def code_block(fname, max_lines=None):
    p = para(doc, f"【{fname}】", cn=HEI, size=10.5, indent_chars=0,
             line=1.3, before=6, after=2)
    text = (SRC / fname).read_text(encoding="utf-8")
    lines = text.splitlines()
    if max_lines:
        lines = lines[:max_lines] + ["……（后续代码见支撑材料）"]
    for ln in lines:
        p = para(doc, ln if ln else " ", en="Consolas", cn=SONG, size=8,
                 line=1.0, indent_chars=0)
        p.paragraph_format.space_after = Pt(0)


code_block("duration.py")
code_block("policy.py", max_lines=120)
code_block("simulator.py", max_lines=80)

enable_update_fields(doc)
doc.save(str(OUT))
print("saved:", OUT)
