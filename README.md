# 数维杯 2026 Problem 3 —— 项目目录

**题意一句话**：一台设备、时限与预算内完成 36 个补货作业；每个作业的真实时长每次 trial
从真人录像库随机抽取、事先未知；决定选排顺序、预留时长、是否购买信息审查、等待或停止。
目标（已定）：最大化期望加权准时率。

## 目录结构

```
shuwei_c/
├── README.md                本文件
├── problem/                 赛题原文与全部输入数据（只读）
│   ├── Problem_3_Requirements.txt / .pdf
│   ├── tasks.csv            36 个作业（release/due/weight/zone/前序）
│   ├── source_segment_index.csv   349 段真人录像标注（343 段 eligible）
│   ├── record_split.csv     环境划分（train/validation/test 整组切分）
│   ├── job_families.csv     P/A/L 三个抽样族
│   ├── service_zones.csv    区域坐标
│   └── common_conditions.json   全部数值条件 + 30 个验证种子
├── prefix_features.csv      前缀运动特征（0.2/0.6/1.0s 三档窗口）
├── literature/              文献检索结果（8 个主题 CSV）
├── round2/                  ★ 本轮重建（当前工作区）
│   ├── src/                 代码：common(常量) simulator(裁判) sampler(CRN采样) + 单测
│   ├── results/             计算结果
│   ├── figures/             图
│   └── docs/                诊断汇总、排版规范
└── archive_round1/          上一轮全部产出（留档可查，不再改动）
```

## 已定技术方案（审题五板）

1. 目标：最大化期望加权准时率
2. 时长模型：family 分层对数正态（矩估计 σ≥0.35）+ 生存分析对照（利用截断样本）
3. 排序：ATC 打底 + 截断防护 + 预算取舍 + Rollout/SAA 前瞻 + 局部搜索；基线 WSPT
4. 信息层：数据驱动证伪——分组 CV 测前缀预测力，R²≈0 则零审查
5. 预留 b=0.2：解析弱占优 + 分位数预留对照实验

## 执行进度

- [x] 第 1 步 审题定方案（含文献调研 40+ 篇）
- [x] 第 2 步 裁判：常量 + 仿真器 + 18 条单测
- [x] 第 3 步 采样器：CRN 确定性验证
- [x] 第 4 步 数据诊断：分布 + 生存对照 + 前缀预测（`round2/docs/diagnosis_summary.md`）
- [x] 第 5 步 时长模型 + b 解析 + 零审查论证（6 项自检全过，见 `round2/results/q5_model_checks.json`、`round2/docs/b_dominance.md`、`round2/docs/no_review.md`）
- [x] 第 6 步 策略全家桶（两阶段引擎 L1~L5 + B1/B2/B3 基线 + SAA 间隙证书，7 项自检全过，见 `round2/docs/strategy_design.md`、`round2/results/q6_strategy_checks.json`）
- [x] 第 7 步 调参→冻结→450 局配对对比（主对比 C6−B3 = **+0.0759±0.0091**；买到顶反证 RM−C5 = **−0.0394±0.0072** 零审查封死；S1 SAA 精确间隙 ≈0，见 `round2/docs/ablation_report.md`）
- [x] 第 8 步前半：审计升级 A1（L2 硬过滤→软罚项）+ A2（均值→中位数点估计）+ 重调参（κ=3.0、K=40）+ 5400 局重跑（主对比改报 C5_online：C5−B3 = **+0.0786±0.0093**）
- [x] 第 8 步 bug 修复（2026-11-23）：① `run_test.py` 臂映射错误——"C5_online" 实际跑成了 RM，test 集真 C5 缺失，已修；② `review_max.py` GBM 特征键前缀不匹配——反证臂的信息通道从未生效，已修（裸 numpy 训练 + 剥前缀映射 + NaN 中位数填补）
- [ ] 第 8 步重跑中：test 四臂 + validation RM 臂（GBM 生效后 RM 单局变慢，预计 ~1h）；`results/q8_test*.csv/json` 当前为修复前旧值，**以重跑后的更新为准**

## 关键数字速查（论文以此为准，全部脚本可复现）

- 时长：ANOVA p=0.0012；对数正态 AIC 全胜（A 229.5 / L 32.6 / P 287.9 vs 威布尔 557.8/72.2/751.4）；参数 A(1.272,0.662) L(1.844,1.013) P(1.588,0.760)
- 截断：朴素 μ 偏差 −0.593/−0.609（n=60/124），EM +0.005/−0.008
- 前缀：R² 全负（最优 GBM −0.481，基线 −0.599）⇒ 零审查
- 消融（validation 450 局/臂，配对）：B2−B1 +0.0358±0.0161；B2b−B2 +0.0129±0.0052；C5−B3 **+0.0786±0.0093**；C6−C5 −0.0060±0.0065；RM−C5（修复前旧值，待重跑）
- SAA（名义映射）：S1 LB 0.9351±0.0084 精确枚举、UB 0.9173；S2/S3 局部搜索下界带 caveat
- b 扫描：ot/completion 全 b 不变，费用 25.43→27.44 单调升，b=0.2 弱占优
- 冻结超参：κ=3.0、roll_K=40（`results/q7_tuning.json`）

## 复现

```bash
# 单测（协议/CRN/策略 7 项）
python3 round2/src/test_strategy.py
# 消融主表（含 SAA，约 40 分钟；已有结果时 analyze 用 --skip-saa 复用）
python3 round2/src/run_ablation.py && python3 round2/src/analyze_ablation.py
# test 终验（4 臂 × 450 局，约 1 小时）
python3 round2/src/run_test.py
# 图
python3 round2/src/make_figures2.py
```

计算只用 CPU（sklearn/scipy/pandas），无需 GPU。

## 仓库说明

`archive_round1/` 为上一轮基线产出（留档对照）：朴素 WSPT 点估计 + 确定性调度流水线，
本仓库 round2/ 的全部改进均以它为对照基线。
