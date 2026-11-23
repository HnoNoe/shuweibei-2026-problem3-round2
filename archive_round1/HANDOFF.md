# 工作区交接文档（HANDOFF）

> 交接时间：2026-11-21 17:33
> 交接原因：用户决定换一个 AI 继续写论文。本文件把已完成的工作、数据、代码、结论和**用户的不满点**一次性讲清楚，接手方请从头到尾读一遍再动手。

---

## 0. 最重要：用户当前的不满（第一优先级，务必先读）

用户看了当前论文后的原话：

> **"你这个格式一眼不就是markdown吗？那个一看就是瞎写的。"**

含义拆解（接手方不要误解）：
1. **格式问题**：当前论文是用 `Markdown → HTML → DOCX` 自动流水线生成的，版式有很重的"模板/markdown 转换"痕迹（字体、段落、表格、图题排版生硬，像机器套壳），**一眼就能看出不是人正常排的数模论文**。用户对此明确不满。
2. **"那个一看就是瞎写的"**：指当前这版给人的观感是"敷衍、瞎写"，**不是指模型结果算错了**。模型与数据本身用户没有质疑。
3. **用户诉求**：要一份**看起来像真正参赛论文**的成品——排版专业、像人写的、符合数维杯官方观感。这大概率是接手方接下来的主任务。

**建议接手方向（供参考，不必照搬）**：
- 别再走 `build_html.py` 那套 markdown→HTML→DOCX 流水线，它就是"一眼 markdown"的根源。
- 优先用**原生 Word 编辑能力**（本地 editor_sdk / editor 工具，或 python-docx 精调）直接排版一份 docx；或按官方 Word 模板（`output/refs/` 里已解出的规范模板）逐项套用样式。
- 参考真实优秀论文的**观感**（见第 6 节）：标题层级、图表样式、摘要框、页眉页脚、段首缩进、公式编号等，逐项对齐。
- 用户如果能提供"你觉得好的样本"，以样本为准。

---

## 1. 项目是什么

- **赛事/题目**：数维杯 C 题（Problem 3）——**基于第一人称演示数据的补货作业时长估计与随机调度决策**（英文原题 `Problem_3_Requirements.pdf`）。
- **目标**：产出一篇完整数模建模论文 + 可复现代码 + 结果表。
- **工作区根目录**：`C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37`

---

## 2. 数据（路径、结构、密码）

### 2.1 原始数据（只读，勿改动）
- 主压缩包：`C:/Users/Camevian/Downloads/C_Problem_Data/Restocking_Problem_and_Data_EN.zip`（5.8 GB）
- 解压根目录：`C:/Users/Camevian/Downloads/C_Problem_Data/Restocking_Problem_and_Data_EN/Restocking_Problem_and_Data_EN/`
  - `Restocking_Tasks_and_Constraints/`：`tasks.csv`、`source_segment_index.csv`、`common_conditions.json` 及各案例约束 CSV（**赛题约束来源**）
  - `Restocking_Tasks_and_Constraints/Problem_3_Requirements.pdf`：**题目要求原文**
  - `Raw_Data/EgoDemo/EgoDemo/E001..E031/`：EgoDemo 原始数值 parquet（已在下载目录就地解压，可直接读）
- 内层子包 `EgoPro.zip` / `EgoStandard.zip` 是**加密**的，密码见 `src/unzip_data.py` 里的 `PWD`（用户提供）。用 `unzip -P` 解（Python zipfile 纯实现解密极慢，约 30MB/min；系统 unzip 快几十倍）。

### 2.2 已处理数据（在工作区内，可直接用）
- `data/raw/EgoPro/`、`data/raw/EgoStandard/`：只解了 `tables/*.csv` 和 `session_metadata.csv`（跳过视频）
- `data/prefix_features_EgoDemo.csv`、`_EgoPro.csv`、`_EgoStandard.csv`：分数据集特征
- `data/prefix_features.csv`：**合并全量特征**（343 个合格区间 × 3 窗口，177 列）

### 2.3 数据结构要点
- 三个数据集异构：EgoDemo = parquet 逐帧（30fps）；EgoPro / EgoStandard = CSV tables。
- 数值通道：双手各 21 关键点位置/四元数、头部位姿、双目头相机位姿、时间戳、手部质量标志。**视频不是策略输入**。
- 时间对齐：按 `source_segment_index.csv` 的 `start_s`/`end_s` 对齐动作片段。
- 源区间 349 个，剔除 6 个时长 <1.2s 的，剩 **343 个合格区间**（train 246 / validation 35 / test 62，按环境整体分组）。

---

## 3. 已完成的建模工作与核心结论（勿重复造轮子）

### 3.1 三条核心结论
1. **前缀预测不了时长**：0.2/0.6/1.0s 前缀运动特征对总时长 R² 为负（约 −0.58 ~ −1.13）、相关系数仅 r≈0.30。因此建模重心是**不确定性量化 + 鲁棒调度**，不是点预测。
2. **预留时长占优推导（最关键的洞察）**：完成时刻 `C = s + p` 与预留 b 无关；设备下次可用 `s + max(p, b)`。当 `b ≤ p` 时设备占用恒为 `s+p`，故取**最小预留 b=0.2 占优**（省费用、零 reserved_idle、不损完成时间）；`b>p` 只增加 idle 和费用。用户/评审若质疑 overrun 高，用"目标自选"解释。
3. **WSPT 排序主导性能**：`score = w / (α·d̂ + β + D + 0.5)`。24 组参数搜索后选定两套策略：
   - **策略 A 准时导向**：min 预留 + WSPT + 不审查（默认，`scheduler.default_params()`）
   - **策略 B 风险规避**：median 预留 + WSPT + 不审查

### 3.2 关键结果数字（论文里所有数字须与之一致）
- 验证集（5 映射 × 3 案例 × 30 种子 = 450 trials/策略），名义映射 (α=1.0, β=0.5)：
  - A on-time：S1 **0.940** / S2 **0.751** / S3 **0.741**；完成率 **0.959**；协议合规率 **1.000**
  - B on-time：S1 0.892 / S2 0.633 / S3 0.647
  - overrun：A **77.96s** vs B **14.43s**（多目标权衡）
- 独立 test split 复现（冻结策略）：A 0.883 / 0.671 / 0.707，B 0.833 / 0.636 / 0.630，全部 >0.5 约束。
- 1800 次试验（验证 900 + 测试 900）`valid_protocol = 1.0`。
- 完整数字来源：`output/q3-20261121/sources/materials_digest.md`（论文事实源，所有数字可追溯到 results/ CSV）。

---

## 4. 代码与结果清单

### 4.1 `src/`（16 个 .py）
| 文件 | 作用 |
|---|---|
| `config.py` | 路径常量、题目常量（映射、种子、案例参数）、特征列 |
| `extract_features.py` / `extract_features_csv.py` | EgoDemo(parquet) / EgoPro+EgoStandard(CSV) 前缀特征提取 |
| `duration_model.py` | 对数正态分层先验 + 前缀强度弱贝叶斯更新，输出中位数与 q10–q90 |
| `simulator.py` | 严格按题目规则的 trial 状态机（7 项单测全过，见 test_simulator.py） |
| `scheduler.py` | 滚动时域调度策略，`default_params()` 是主策略 A |
| `runner.py` | 源区间采样（`random.Random(seed).choice`）、作业实例化、驱动仿真 |
| `evaluate.py` | 验证/测试评估（跑 450 trials），命令行 `--feat data/prefix_features.csv --split validation` |
| `tune.py` / `tune_v2.py` | 参数搜索（v2 是最终版，搜 reserve_mode × order_rule × review_mode） |
| `compare.py` | 多策略对比 + test split 模拟 |
| `make_figures.py` | 生成 4 张论文配图 |
| `build_html.py` | ⚠️ **用户不满的那套 markdown→HTML→DOCX 流水线的生成器**，接手方大概率要弃用 |
| `unzip_data.py` / `stage_inner_zip.py` | 数据解压（含密码；内层 zip 落盘用） |
| `test_simulator.py` | 仿真器单测 |

### 4.2 `results/`（CSV）
`comparison_validation.csv`、`comparison_test.csv`（各 900 行 trial 级）、`trial_results_validation.csv`（450 行全指标）、`summary_validation.csv`、`param_search.csv`、`param_search_v2.csv`、`param_search_v2.csv`。

### 4.3 `figures/`（4 张配图）
`fig1_ontime_by_case.png`、`fig2_ontime_by_mapping.png`、`fig3_tradeoff.png`、`fig4_duration_diagnosis.png`（图 4 是"前缀强度 vs 时长"弱相关诊断，建模路线依据，建议保留）。

### 4.4 复现命令
```bash
cd C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37
.venv/Scripts/python.exe src/test_simulator.py            # 单测
.venv/Scripts/python.exe src/evaluate.py --feat data/prefix_features.csv --split validation
.venv/Scripts/python.exe src/compare.py                    # 策略对比 + test
.venv/Scripts/python.exe src/make_figures.py              # 重画配图
```
运行环境：`.venv/`（Windows），已装 pandas 3.1.1 / pyarrow / scikit-learn 1.9.1 / scipy / matplotlib / pypdf 7.1.1。

---

## 5. 论文现状（当前成品 + 中间产物）

- **当前成品**（用户不满的那版）：工作区根目录 `数维杯C题_建模论文.docx`（249 KB；封面+目录+4图+4表+参考文献；HTML 门禁 100 分、DOCX 预检通过——但用户仍认为"一眼 markdown、瞎写的"，说明**技术合规 ≠ 观感合格**）。
- 中间产物在 `output/q3-20261121/`：
  - `sources/materials_digest.md`：论文事实源
  - `stage1/final_draft.md`：当前稿 Markdown 正文
  - `stage1/final_draft_v1_backup.md`：更早版本备份
  - `stage2/formatted-数维杯C题建模论文.html`：流水线生成的 HTML
  - `stage3/数维杯C题_补货作业时长估计与随机调度建模论文.docx`：流水线生成的 docx（根目录那份是它的拷贝）
- ⚠️ 当前稿摘要按"针对问题1/2/3"三段写（528 字），新增了"问题分析"章、图表按章编号、页眉 Team#/Page——这些**结构改动是对的**，问题在**版式观感**，接手方不要把这些内容改动又退回去了。

---

## 6. 比赛规范与优秀论文参考（重排论文时要对齐）

### 6.1 规范文件（已解出）
- `output/refs/2025年第十届数维杯大学生数学建模挑战赛论文Word模板及格式规范.doc`（已用 Word COM 转成 `output/refs/format_spec.txt`，可直接读）
- 核心规范摘要：
  - **摘要 300–600 字、控制一页、总分总**，必须写出具体方法、模型、结果；段首可用"针对问题1/2/3…"
  - 结构：问题重述 → 问题分析 → 模型假设 → 符号说明 → 模型建立与求解 → 模型评价及优化 → 参考文献 → 附录
  - 正文宋体小四，标题三号黑体，1.5 倍行距；页码页脚居中
  - 图题五号宋体、居中于图**下方**，按章编号（图3-2）；表题五号、居中于表**上方**（表5-4）
  - 参考文献三号黑体标题、5号宋体、重新起页；附录重新起页
  - 页眉格式参考：`Team # 2025000000001   Page 1 of 9`
  - 支撑材料 ≤20MB，压缩包含所有源程序，不得含身份信息
  - **最终提交 PDF**

### 6.2 优秀论文（参考观感与设计思路，非同题）
- 路径：`C:/Users/Camevian/Downloads/2025年第十届数维杯春季赛优秀论文及赛题/`
  - `A题/`、`B题/`、`C题/`：各 9–11 篇获奖论文 PDF
  - **注意：这些是 2025 第十届春季赛 C 题「清明时节雨纷纷，何处踏青不误春？」（气象/物候题），与本项目补货调度题不是同一道题**——只能参考写作范式与排版观感，模型内容必须用本项目自己的。
- 已提取正文到 `output/refs/papers/`（`pypdf` 提取；⚠️ 自带 pdf-cli 对这些大 PDF 会堆损坏崩溃，用 pypdf）。
- 优秀论文可借鉴点：摘要"针对问题1/2/3"逐条写方法+模型+数字结果；每个问题独立闭环（分析→建模→求解→结论）；模型评价分优点/缺点/推广且每条具体化带数字；突出亮点。

### 6.3 队号
当前论文页眉填的是占位 `2026112100001`，**提交前须换真实队号**。

---

## 7. 工作记录 / 历史
- 本项目当日日志：`.workbuddy/memory/2026-11-21.md`（含数据集、建模结论、交付物、技术备忘全过程）。
- 完整对话历史（如需追溯）：`C:/Users/Camevian/.workbuddy/projects/c-Users-Camevian-WorkBuddy-2026-11-21-08-56-37/b8a23e9a-7d8a-4082-a76a-ceaee8b54a47.jsonl`

---

## 8. 给接手方的行动建议（按优先级）
1. **先解决用户不满的格式问题**：弃用 `build_html.py` 流水线，改用原生 Word/editor 工具或官方模板重排 `数维杯C题_建模论文.docx`，对齐优秀论文的观感（这是用户换 AI 的直接原因）。
2. 内容结构已基本正确（问题一/二/三、问题分析章、按章编号图表、误差分析/优缺点/推广），**在此基础上改版式，不要重写模型结论**。
3. 所有数字以第 3.2 节和 `materials_digest.md` 为准，改版时逐一核对。
4. 若需要重跑结果，用第 4.4 节命令；结果应稳定复现。
5. 最终如需 PDF，用规范要求的提交格式转换；队号换真实值。
