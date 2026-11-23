# 建模结果材料摘要（数维杯 C 题 / Problem 3）

> 本文件是论文写作的事实来源，所有数字均可追溯到 `results/` 下的 CSV。
> 原始题目：Problem 3 Requirements（Restocking Problem and Data EN）。

## 一、题目与数据

- 数据包：`Restocking_Problem_and_Data_EN.zip`，含三个异构子数据集 EgoDemo（parquet 逐帧）、EgoPro / EgoStandard（CSV 表格），共 349 个文本标注区间，其中 **343 个合格**（时长 ≥ 1.2 s），6 个因过短被排除。
- 数据划分（按环境整体分组）：train 246、validation 35、test 62。
- 采样族 P/A/L 是抽样分层，非机器人技能或商品类别。
- 三案例：S1（8 作业，H=100，B=32，K=4）、S2（12，100，34，6）、S3（16，125，40，8）。
- 时长映射 p = αd + β：名义 (1.0,0.5)，敏感性 (0.75,0.5)、(1.5,0.5)、(1.0,0.0)、(1.0,1.0)，共 5 种。
- 验证种子 30 个（2026111700–2026111729）；每案例 30 trials，5 映射共 450 trials。
- 信息机制：每作业免费前 0.2 s 数值前缀；最多 2 次付费审查（成本 1，占用 1 s，延迟 1 s）扩展到 0.6 / 1.0 s。
- 数值通道：双手各 21 关键点位置/四元数、头部位姿、双目头相机位姿、时间戳、手部质量标志（30 fps）；视频不是策略输入。

## 二、关键建模诊断（论文核心论证点）

1. **前缀时长预测力弱**：用 HistGradientBoosting、随机森林在 343 区间上做按环境分组交叉验证，0.2/0.6/1.0 s 前缀特征的 R² 均为负（−0.58 ~ −1.13），与时长相关系数仅 r≈0.30；族先验基线 MAE≈3.5 s。说明短前缀无法精确反推总时长。
2. **因此建模重心从"点预测"转向"不确定性量化 + 鲁棒调度"**：采用 family 分层的对数正态先验 + 前缀强度弱贝叶斯更新，输出中位数与 q10–q90 区间；q10–q90 覆盖率 train 83.3%、validation 88.6%。
3. **预留时长的关键推导**：完成时刻 C = s + p 与预留 b 无关；设备下次可用 s + max(p,b)。当 b ≤ p 时占用恒为 s+p。因此 b 取最小值 0.2 可使费用（2+0.1b+0.2D）与 reserved_idle 最小，且不损失完成时间——这是以完成率/准时率为目标时的占优策略。b>p 只增加空闲与费用，仅在压制 observed_overrun 时有意义。

## 三、策略与参数搜索

- 策略 = 滚动时域启发式，三个旋钮：预留模式 reserve_mode ∈ {min, median, q75}、排序规则 order_rule ∈ {wspt, edd, slack, combined}、审查模式 review_mode ∈ {none, selective}。
- 搜索 24 组配置 × 90 trials（validation，名义映射），其中 10 组满足"每案例 on-time ≥ 0.5"。
- 选定两套策略（validation 上冻结）：
  - **策略 A 准时导向**：min 预留 + WSPT（权重/加工时间比最大优先）+ 不审查。
  - **策略 B 风险规避**：median 预留 + WSPT + 不审查。
- WSPT 排序得分 = w / (预估加工时间 + 旅行时间)，其中预估加工时间 = α·d_median + β。

## 四、验证结果（validation，450 trials/策略，含 5 映射）

名义映射 (1.0,0.5) 各案例均值（源 `comparison_validation.csv`）：

| 策略 | S1 准时率 | S2 | S3 | 完成率(三案例) | 超期量 | 预留空闲 | 成本 |
|---|---|---|---|---|---|---|---|
| A 准时导向 | 0.940 | 0.751 | 0.741 | 1.000/0.957/0.921 | 77.96 | 0.00 | 25.68 |
| B 风险规避 | 0.892 | 0.633 | 0.647 | 1.000/0.868/0.819 | 14.43 | 17.11 | 29.96 |

- 全局（A，名义）：加权完成率 0.959±0.047，准时率 0.811±0.116，加权拖期 118.68±93.94，成本 25.675±6.146，审查 0，超期 77.96±19.54，空闲 0，截断单元 0.478±0.500，协议合规率 1.000。
- 约束检验（A，名义，各 30 trials）：S1 0.940（95%CI 0.917–0.958）、S2 0.751（0.722–0.781）、S3 0.741（0.716–0.768），**三案例均通过 ≥0.5 约束**。
- 五映射平均准时率：A 0.801，B 0.721（源 `comparison_validation.csv` 全映射 groupby）。

## 五、模拟最终测试（test split + 30 种子，冻结策略）

名义映射各案例准时率（源 `comparison_test.csv`）：

| 策略 | S1 | S2 | S3 | 全映射平均 |
|---|---|---|---|---|
| A 准时导向 | 0.883 | 0.671 | 0.707 | 0.741 |
| B 风险规避 | 0.833 | 0.636 | 0.630 | 0.682 |

两套策略在独立 test split 上仍全部满足 ≥0.5 约束，性能与 validation 一致，未出现过拟合。

## 六、合规性与可复现

- 全部 1800 trials（验证 900 + 测试 900）`valid_protocol = 1.0`，无不可行动作。
- 时长模型仅用 train 拟合；超参与策略在 validation 选定；test 前冻结。
- 策略不访问源标识、区间端点、总时长真值、语义描述、种子或后续样本；不使用前缀匹配检索。
- 源采样严格按 `random.Random(seed).choice`，按 job_id 排序、有放回、保持源文件行序、每案例重置生成器。
- 代码文件：`src/config.py`（配置）、`extract_features*.py`（特征）、`duration_model.py`（时长估计）、`simulator.py`（协议仿真器，7 项单测通过）、`scheduler.py`（策略）、`runner.py`（采样与驱动）、`evaluate.py`（验证）、`tune_v2.py`（调参）、`compare.py`（对比）、`test_simulator.py`（单测）。
- 结果文件：`results/trial_results_validation.csv`（450 行 trial 级全指标）、`summary_validation.csv`、`comparison_validation.csv`、`comparison_test.csv`、`param_search_v2.csv`。

## 七、论文配图（本地绝对路径）

- `C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37/figures/fig1_ontime_by_case.png`：验证集/测试集各案例准时率对比（含 0.5 约束线）。
- `C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37/figures/fig2_ontime_by_mapping.png`：五种映射下两策略准时率。
- `C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37/figures/fig3_tradeoff.png`：完成率/超期量/预留空闲的多指标权衡。
- `C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37/figures/fig4_duration_diagnosis.png`：时长分布与前缀强度弱相关诊断。
