# 参考文献清单（截至第 6 步策略研究，按用途分组）

来源：8 个主题的学术检索（结果存 `literature/*.csv`）。
★ = 建议写进论文参考文献的核心引用；其余为背景调研支撑。

## 一、调度优先规则（第 6 步策略的骨架）

1. ★ Vepsalainen A P J, Morton T E. Priority rules for job shops with weighted tardiness costs[J]. Management Science, 1987.（被引 888 —— ATC 规则的原始出处，本文排序方法的招牌引用）
2. ★ Ow P S, Morton T E. The single machine early/tardy problem[J]. Management Science, 1989.（被引 451 —— ATC 在单机 E/T 问题的后继）
3. ★ Kanet J J, Li X. A weighted modified due date rule for sequencing to minimize weighted tardiness[J]. Journal of Scheduling, 2004.（被引 117 —— MDD 规则，对照类）
4. Sun X, Noble J S, Klein C M. Single-machine scheduling with sequence dependent setup to minimize total weighted squared tardiness[J]. IIE Transactions, 1999.（被引 99 —— ATCS，序相关准备时间版）
5. Xi Y, Jang J. Minimizing total weighted tardiness on a single machine with sequence-dependent setup and future ready time[J]. Int. J. Advanced Manufacturing Technology, 2013.（被引 21 —— 带 ready time 的 ATC 扩展，与我们 release 约束对应）
6. Martins A S M, Valente J M S. A computational evaluation of new and existing dispatching rules for the single machine total weighted tardiness problem[J]. ...in Operational Research, 2026.（被引 0 —— 最新规则评测）

## 二、随机单机调度（问题的学名定位）

7. ★ Soroush H M, Fredendall L D. The stochastic single machine scheduling problem with earliness and tardiness costs[J]. European Journal of Operational Research, 1994.（被引 73 —— 随机工期调度经典）
8. ★ Adamu M O, Adewumi A O. A survey of single machine scheduling to minimize weighted number of tardy jobs[J]. Journal of Industrial and Management Optimization, 2014.（被引 83 —— 综述，引言用）
9. Jang W. Dynamic scheduling of stochastic jobs on a single machine[J]. European Journal of Operational Research, 2002.（被引 56 —— 动态调度）
10. Gutjahr W J, Hellmayr A, Pflug G C. Optimal stochastic single-machine-tardiness scheduling by stochastic branch-and-bound[J]. EJOR, 1999.（被引 48 —— SAA/随机分支定界）
11. Cai X, Wang L, Zhou X. Single-machine scheduling to stochastically minimize maximum lateness[J]. Journal of Scheduling, 2007.（被引 28）
12. Ertem M, Ozcelik F, Saraç T. Single machine scheduling problem with stochastic sequence-dependent setup times[J]. Int. J. Production Research, 2019.（被引 34 —— 随机序相关 setup，对应我们的移动+换型）
13. Montalvão E H R, Pessoa A A. Single machine total weighted tardiness problem with budgeted uncertainty set on the processing times[C]. SBPO, 2016.（预算不确定集，备选方案）

## 三、时长分布拟合（第 4 步 Q1 + 第 5 步模型）

14. ★ Strum D P, May J H, Vargas L G. Modeling the uncertainty of surgical procedure times: comparison of log-normal and normal models[J]. Anesthesiology, 2000.（被引 392 —— 作业时长对数正态的经典证据）
15. ★ May J H, Strum D P, Vargas L G. Fitting the lognormal distribution to surgical procedure times[J]. Decision Sciences, 2000.（被引 182）
16. Van der Linden W J. A lognormal model for response times on test items[J]. Journal of Educational and Behavioral Statistics, 2006.（被引 587）
17. Rummel B. Beyond Average: Weibull Analysis of Task Completion Times[J]. Journal of Usability Studies, 2017.（被引 14 —— 对照分布 Weibull）

## 四、截断与生存分析（第 4 步截断对照实验）

18. ★ Fleming T R, Harrington D P. Counting Processes and Survival Analysis[M]. Wiley, 2013.（被引 4409 —— 删失数据方法经典）
19. ★ Wang P, Li Y, Reddy C K. Machine learning for survival analysis: A survey[J]. ACM Computing Surveys, 2019.（被引 1113）
20. Gijbels I. Censored data[J]. Wiley Interdisciplinary Reviews: Computational Statistics, 2010.（被引 97）
21. Lin D Y, Sun W, Ying Z. Nonparametric estimation of the gap time distribution for serial events with censored data[J]. Biometrika, 1999.（被引 276）
22. Turkson A J, Ayiah-Mensah F, et al. Handling censoring and censored data in survival analysis: a standalone systematic literature review[J]. Int. J. ... 2021.（被引 189）

## 五、信息价值与前缀预测（第 4 步 Q2 → 零审查决策）

23. ★ Xing Z, Pei J, Philip S Y. Early prediction on time series: A nearest neighbor approach[C]. IJCAI, 2009.（被引 190 —— 前缀预测的学名：早期时间序列预测）
24. ★ Gupta A, Gupta H P, Biswas B, et al. Approaches and applications of early classification of time series: A review[J]. IEEE Trans. ..., 2020.（被引 202 —— 综述）
25. Xing Z, Pei J, Yu P S. Early classification on time series[J]. Knowledge and Information Systems, 2012.（被引 210）
26. Wu R, Der A, Keogh E J. When is early classification of time series meaningful?[J]. IEEE TKDE, 2021.（被引 32 —— "什么时候值得早预测"，与我们"审查值不值"同构）
27. Bölöni L, Turgut D. Value of information based scheduling of cloud computing resources[J]. Future Generation Computer Systems, 2017.（被引 49 —— VoI 调度框架）

## 六、仿真对比方法：共同随机数与配对设计（第 3 步 + 第 7 步验证）

28. ★ Nelson B L, Matejcik F J. Using common random numbers for indifference-zone selection and multiple comparisons in simulation[J]. Management Science, 1995.（被引 204 —— CRN 核心引用）
29. ★ Goldsman D, Nelson B L. Ranking, selection and multiple comparisons in computer simulation[C]. Winter Simulation Conference, 1994.（被引 122）
30. Nelson B L, Goldsman D. Comparisons with a standard in simulation experiments[J]. Management Science, 2001.（被引 125）
31. Swisher J R, Jacobson S H, Yücesan E. Discrete-event simulation optimization using ranking, selection, and multiple comparison procedures: A survey[C]. ACM TOMACS, 2003.（被引 210 —— 综述）
32. Chick S E, Inoue K. New procedures to select the best simulated system using common random numbers[J]. Management Science, 2001.（被引 181）
33. Yang W N, Nelson B L. Using common random numbers and control variates in multiple-comparison procedures[J]. Operations Research, 1991.（被引 105）
34. Ioannou P G, Martinez J C. Comparison of construction alternatives using matched simulation experiments[J]. Journal of Construction Engineering and Management, 1996.（被引 97 —— 配对仿真应用）
35. Sharma U. When Does Pairing Seeds Reduce Variance? Evidence from a Multi-Agent Economic Simulation[J]. arXiv:2512.24145, 2025.（配对种子何时减方差）

## 七、滚动重优化（第 6 步 Rollout 组件的方法定位）

36. ★ Yao S, Wang P, Liu X, et al. Rolling optimization of mobile energy storage fleets for resilient service restoration[J]. IEEE Trans. ..., 2019.（被引 384 —— 滚动重优化近年应用代表作）
37. Shen X, Lv J, Du S C, Matta A. Dynamic capacity allocation for integrated online and offline outpatient services with stochastic multiple revisits[J]. IEEE Trans. ..., 2026.（被引 3 —— 滚动+蒙特卡洛）
38. Özel B, Sezen B, Balcıoğlu Y S. Spontaneous volunteer task assignment in the acute phase of disaster response: A rolling-horizon MIP approach[J]. Sustainability, 2026.（被引 0）

## 未采用但调研过的方向（备查，不写进论文）

- 强化学习调度：Szwarcfiter C, Herer Y T, Shtub A. Balancing project schedule, cost, and value under uncertainty: A reinforcement learning approach[J]. Algorithms, 2023.（被引 31 —— 被否决，小样本+冻结纪律下过拟合风险）
- 分布鲁棒：Zhang S, Shen Z, Chen S, Zhou J. A rolling-horizon stochastic optimization framework ... arXiv:2604.06548, 2026.
- 信息价值（通信领域）：Ayan O, Vilgelm M, Klügel M, Hirche S. Age-of-information vs. value-of-information scheduling..., 2019.（被引 217）
- MDP 探索：Sledge I J, Emigh M S, Príncipe J C. Guided policy exploration for MDPs using an uncertainty-based VoI criterion[J]. IEEE Trans., 2018.（被引 25）

## 八、策略前瞻：Rollout 与 SAA（第 6 步 L4/L5 的方法锚）

39. ★Bertsekas D P, Castañón D A. Rollout algorithms for stochastic scheduling problems[J]. Journal of Heuristics, 1999.（被引 496 —— L4 rollout 的原始锚：基策略 + 单步前瞻，报告近最优表现；性能上界由基策略质量决定，所以 L1~L3 必须先打牢）
40. ★Kleywegt A J, Shapiro A, Homem-de-Mello T. The sample average approximation method for stochastic discrete optimization[J]. SIAM Journal on Optimization, 2002.（被引 2989 —— SAA 总纲：期望目标用样本均值逼近，给收敛率与复杂度）
41. ★Verweij B, Ahmed S, Kleywegt A J, et al. The sample average approximation method applied to stochastic routing problems: a computational study[J]. Computational Optimization and Applications, 2003.（被引 716 —— SAA 最优性间隙估计的操作手册：候选解 + 多批独立样本估 gap；用于给主策略发"距最优多远"的证书）
42. ★Elaswad A, Carrasco R A, Sakr N. Decision-focused learning for operating room scheduling under uncertain surgery durations with two-stage stochastic optimization[C]. Springer, 2026.（被引 0 —— 与本题几乎同构：手术时长拟合 Lognormal、两阶段"先定序列 + 现场反应"；是我们"离线先验序列 + 在线修正"框架的最近文献镜像）
43. Roohnavazfar M, Manerba D, Fotio Tiotsop L, et al. Stochastic single machine scheduling problem as a multi-stage dynamic random decision process[J]. Computational Management Science, 2021.（被引 11 —— 随机单机调度的多阶段 DP 表述 + 两阶段 SP 求解对照）
44. Riviere L, Artigues C, Fargier H. Two-stage stochastic/robust scheduling based on permutable operation groups[J]. Annals of Operations Research, 2024.（被引 7 —— 两阶段调度中"可置换组"的中间结构：介于纯先验序列与完全重排之间，备查）

## 九、优先规则的随机化扩展与元启发（第 6 步 L1/L5 备查）

45. Xi Y, Jang J. Minimizing total weighted tardiness on a single machine with sequence-dependent setup and future ready time[J]. Int. J. Advanced Manufacturing Technology, 2013.（被引 21 —— ATC 族规则在"未来释放时间"场景的扩展，与我们 release_s 约束直接对应）
46. Anghinolfi D, Paolucci M. A new discrete particle swarm optimization approach for the single-machine total weighted tardiness scheduling problem with sequence-dependent setup times[J]. EJOR, 2009.（被引 196 —— 元启发对照组；本方案 L5 只做局部搜索，PSO 备查不采用）
47. Ozcelik F, Ertem M, Saraç T. A stochastic approach for the single-machine scheduling problem to minimize total expected cost with client-dependent tardiness costs[J]. Engineering Optimization, 2022.（被引 12 —— 随机 setup 下单机期望成本最小化，目标函数与我们加权准时率同族）

## 十、整车对标：解决同类问题的应用论文（按我们的四层组织）

### 数据侧：第一人称视频 → 作业时长
48. ★Grauman K, et al. Ego-exo4d: Understanding skilled human activity from first- and third-person perspectives[J]. 2024.（被引 897 —— 第一人称作业视频理解的标准基准；我们的"前缀特征"在文献里的学名是 egocentric action anticipation/early recognition）
49. Chavan V, Imgrund Y, Dao T L, Bai S, et al. Indego: A dataset of industrial scenarios and collaborative work for egocentric assistants[J]. 2026.（被引 15 —— 工业协作场景的第一人称数据集，与本题"仓库补货录像"场景同族）

### 分布侧：作业时长建模与拟合优度
50. ★Trietsch D, Mazmanyan L, Gevorgyan L, et al. Modeling activity times by the Parkinson distribution with a lognormal core: Theory and validation[J]. 2012.（被引 156 —— 作业时长=对数正态核+右尾膨胀的真实数据建模与验证；我们 Q1 的"对数正态 AIC 全胜"与本文结论互证）
51. Hvingelby C. Effects of processing time distributions and rescheduling policies on efficiency[J]. 2023.（被引 0 —— 时长分布形状对重调度策略效率的影响实测）

### 决策侧：随机调度在真实系统的解法
52. ★Guda H, Dawande M, et al. Optimal policy for a stochastic scheduling problem with applications to surgical scheduling[J]. 2016.（被引 48 —— 医疗随机调度的最优策略推导+真实应用；与我们"分布感知调度"路线同宗）
53. Muter I, Öncan T. Order batching and picker scheduling in warehouse order picking[J]. 2022.（被引 53 —— 仓库拣选调度应用文的当代代表）
54. Löffler M, Boysen N, et al. Picker routing in AGV-assisted order picking systems[J]. 2022.（被引 138 —— 拣选路径规划高被引综述性应用）
55. Daoud H, Won D. Real-time order picking planning framework for warehouses and distribution centres[J]. 2022.（被引 30 —— 实时性要求下的拣选计划框架，对应我们的在线 recourse 层）
56. Zambrano-Rey G M, González-Neira E M, et al. Minimizing the expected maximum lateness for a job shop subject to stochastic machine breakdowns[J]. 2024.（被引 9 —— 期望型目标+随机环境的作业车间应用）

### 信息侧：审查动作 = "调度中的测试/观测"
57. ★Levi R, Magnanti T, Shaposhnik Y. Scheduling with testing of heterogeneous jobs[J]. 2024.（被引 9 —— "花资源测试作业以获得更准时长"的学术标准表述；我们的 review 动作在该文献族里的学名是 scheduling with testing，零审查结论与它互补：该文给"何时该测"，我们用数据证明"此题测了也白测"）
58. Monahan G E. A survey of partially observable Markov decision processes[J]. 1982.（被引 1586 —— POMDP 综述；"买不买观测"决策的理论祖庭）
59. Memarzadeh M, Pozzi M. Integrated inspection scheduling and maintenance planning for infrastructure systems[J]. 2016.（被引 119 —— 检查调度一体化的高被引应用）
