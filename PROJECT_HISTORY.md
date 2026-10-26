# Uni-Poly 项目实验历史与当前基线

更新时间：2026-10-26。本文件保留项目级的当前基线、正式实验结果、路线结论和证据入口；逐作业日志、tmux 窗口、事故快照与完整配置继续保留在 results 和 logs 目录。指标展示保留三位小数，选模、差值与门槛判断使用未舍入值。

状态说明：已写入本文件只表示项目记录已更新，不自动等于独立审查通过。result JSON 仍为 PENDING 的周期继续标为待审查；这些结论可以用于后续规划，但不能改写为已验收模型。

## 1. 当前模型基线

用户于 2026-10-26 指定 UA3D-SPATIAL-20261025-01/r4 的 FULL/BASE 臂为当前模型基线。该决定用于后续新实验的 reference 和默认性能门槛，不改写 r4 原记录中的 FINAL=C_CONST。

- 基线 ID：UA3D-SPATIAL-20261025-01/r4/FULL/BASE
- 正式七任务 macro R²：0.795244745287872
- 展示：0.795
- 统计：paper5_outer，七任务各五折，seed 42+fold，任务等权 macro
- 原始证据：results/ua3d_spatial_20261025/r4/final_review/results.json

BASE 是原生 TMOL3M 架构：INT2D 键路径与原生 3D MaskedGalformer3D，Morgan FP 作为 2048 维输入但不再作为预测目标；不额外加入非键空间距离、角度残差或 P10K 表示。BASE 不是 P10K checkpoint 与 TMOL3M/FP 的简单拼接，也不能把 UA2D 的 PTSCALE/P10K 或历史 R10 融合结果当成 BASE。

| 任务 | BASE R² | SOTA 参考 R² | BASE−参考 |
|---|---:|---:|---:|
| Eea | 0.924 | 0.934 | −0.010 |
| Egb | 0.917 | 0.923 | −0.006 |
| Egc | 0.913 | 0.911 | +0.002 |
| Ei | 0.806 | 0.839 | −0.033 |
| EPS | 0.771 | 0.787 | −0.016 |
| Nc | 0.821 | 0.856 | −0.035 |
| Xc | 0.415 | 0.423 | −0.008 |

论文参考来自 SOTA_RESULT.md，是 Periodic-TDL 在不同任务上的最佳配置包络，不是一个可运行单模型。BASE 只在 Egc 上数值高于对应参考，macro 低于参考包络约 0.015；数据、split 和协议等价性未证明，因此只作数值对照。

历史 R10 的 macro 为 0.8035089342046258，高于 BASE 0.0082641889167538。它确实相对 UA2D PTSCALE/P10K 达到过原定的 +0.010 门槛，但相对新 BASE 未达到默认 +0.010，且模型组成和数据协议不同于 BASE，因此不改设当前基线，旧冻结结论继续保留。

## 2. 2026-10-25 至 10-26 最新正式实验

### UA3M-PaperFuse：三模态整体比较

科学问题：在同一输入缓存、同一七任务五折和正式 TEST 口径下，UNI3M、TMOL3M 与 B_A020 的整体性能是否达到 +0.010 的预设门槛。

| 臂 | 正式 macro R² | 相对 B_A020 |
|---|---:|---:|
| B_A020 | 0.785 | 0.000 |
| UNI3M | 0.778 | −0.007 |
| TMOL3M | 0.790 | +0.005 |

validation 选择 B_A020，FINAL=B_A020。TMOL3M 的 TEST 数值更高，但按未舍入值仅 +0.004838913368283682，未达门槛；不能在看到 TEST 后改选，也不能把 +0.005 写成性能通过。最终 CPU exit 0，独立审查仍为 PENDING。

证据：results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/final_review/results.json。

### TMOL3M 下游 3D 路径配对消融

科学问题：保持同一三模态预训练起点、35 套原 head 初态、固定 train-only scaler 和七任务五折预算，仅关闭下游 3D 路径，是否使 FULL 产生至少 +0.010 的变化。

| 臂 | 正式 macro R² |
|---|---:|
| FULL | 0.788 |
| NO3D_FT | 0.784 |

FULL−NO3D_FT=+0.004578983000486758，未达门槛。Nc 五折均改善且均值为 +0.025，但这不能替代七任务 macro 门槛。该实验只评价整个下游 3D 路径，不能单独归因于几何信息、3D 预训练或 PT/FT 交互。独立审查仍为 PENDING。

证据：results/tmol3m_3d_ablation_20261025/r1/repairs/json_import_repair1/final_review/results.json。

### UA3D-SPATIAL/r4：空间距离与角度残差

科学问题：在 BASE 上加入有限 Trimer 的非键空间残差，是否同时改善完整模型和固定的 3D 可读性探针，并达到整体性能门槛。

| 臂 | FULL 正式 macro R² | probe macro R² |
|---|---:|---:|
| BASE | 0.795 | 0.671 |
| C_CONST | 0.792 | 0.682 |
| S_RAD | 0.790 | 0.693 |
| S_RADANG | 0.792 | 0.700 |

S_RADANG−BASE：FULL 为 −0.0036057119529214754，probe 为 +0.029943616592139066。空间表征经探针读取后明显更可辨，但完整模型没有迁移收益。r4 的 validation 选择仍记录为 FINAL=C_CONST；用户本次规定的当前基线是 BASE，两者不能混写。

原最终 CPU 为 exit 1，原因属于测试/报告缺陷；隔离修复后 14 passed、controller exit 0。原始失败日志和修复记录均保留，独立审查仍为 PENDING。

证据：results/ua3d_spatial_20261025/r4/final_review/results.json；results/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1/final_review/results.json。

### UA3D-STAGE/r2：PT 与 FT 阶段拆分

科学问题：把空间 3D 干预分别放在预训练和微调阶段，判断收益来自 PT、FT 还是二者组合，并用统一冻结后正式 TEST 检验。

| 单元 | 组成 | 正式 macro R² |
|---|---|---:|
| E00 | BASE，PT/FT 均关闭空间分支 | 0.795 |
| E10 | PT_ONLY | 0.790 |
| E01 | FT_ONLY | 0.797 |
| E11 | S_RADANG | 0.792 |
| REF | C_CONST | 0.792 |

validation 提名 C2=PT_ONLY，但 C2−REF=−0.0011331146039976758，因此 NEXT_FINAL=REF。空间 PT 对完整模型为负向；FT 空间分支有小幅正的描述性效应，但整体仍不足以通过门槛。FT_ONLY 是 TEST 最高臂，不能据此做事后改选。描述性交互约为 −0.000431，不应解释为显著效应。

原最终 CPU 为 9 passed、1 failed、exit 1；非法阶段 reservation 入口修复后 10 passed、exit 0。原失败记录继续保留，独立审查仍为 PENDING。

证据：results/ua3d_stage_20261026/r2/final_review/results.json；results/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1/stage_acceptance.json。

### UA3D-FROZEN-TRANSFER/r1：冻结 3D 表征迁移

科学问题：在共同 BASE 非 3D 背景中冻结 S_RADANG 3D 表征包，只训练下游，是否能把空间信息稳定迁移到完整模型。

| 单元 | 正式 macro R² |
|---|---:|
| F_BASE | 0.787 |
| F_G3D | 0.785 |
| BASE | 0.795 |
| REF=C_CONST | 0.792 |

F_G3D−F_BASE=−0.0024166099482554326；F_G3D−BASE=−0.01024043997257651；F_G3D−REF=−0.006599918921751935。冻结迁移未带来整体收益，NEXT_RECOMMEND=REF。完成 70 套 FULL、4,900 epochs、112,000 updates、70 次正式 TEST 和 11,750 行预测；最终 CPU 8 passed、controller exit 0，独立审查仍为 PENDING。

证据：results/ua3d_frozen_transfer_20261026/r1/final_review/results.json；results/ua3d_frozen_transfer_20261026/r1/final_review/closeout.md。

## 3. 近期路线结论

- 空间 3D 探针改善不等于完整模型改善。r4 证明有限的表示可读性收益存在，stage 和 frozen transfer 都没有把它迁移为整体性能。
- 当前应把 BASE 作为所有新比较的 reference。任何新空间编码、阶段策略、重复链构象或 FP 融合都应报告相对 BASE 的未舍入 Δmacro R²。
- 默认性能门槛维持 Δmacro R² ≥ 0.010。低于门槛的正差值只记为机制或描述性信号，不能晋级，也不能按任务或折拼接。
- validation 用于选模，冻结后只执行正式 TEST。看到 TEST 后改选、改权重、换 alpha、拼接任务胜者或扩大候选都属于违反协议。
- 共享 validation/test 已参与多轮路线开发，不构成独立盲测；单次 PT、单个 checkpoint、probe、attention 或 loss 都不替代正式预测评估。
- Base 的原生 3D 路径已经存在。后续研究应检验新增 3D 表示对完整模型的增量，而不是把“是否使用任意 3D”与“是否新增空间表征”混为同一问题。

## 4. UA2D 历史实验摘要

下表只保留已执行周期的最终候选或代表性结果。详细逐任务、逐折指标、预算和失败现场在对应 results 根保留。

| 周期或臂 | 受控变化或问题 | 结果 | 结论 |
|---|---|---|---|
| FP/V1 | 预训练 FP 目标的 V1 路线 | TEST 0.787 | 未达门槛 |
| FP/V2 | 第二个 FP 预训练方案 | TEST 0.785 | 未达门槛 |
| FP/B_FP | FP 预训练对照臂 | TEST 0.791 | 完整正式 TEST |
| CLEAN/G_ALL1 | clean-FP 固定候选 | TEST 0.778 | 最终固定候选，未晋级 |
| CLEAN/QL_DYN | 匹配内部对照 | TEST 0.790 | 仅数值诊断，不能追认为最终候选 |
| ATTN/V1_ATTN64 | 改变 attention 尺度 | TEST 0.782 | 未晋级 |
| DEMOL/SELF、NEIGHBOR | 只做开发筛选 | validation 0.820、0.819；TEST=0 | 正式性能不可比较 |
| ET_SINGLE/repair1 | 单分支 ET 开发 | validation 0.815；TEST=0 | 正式性能不可比较 |
| HOP_EQ | 固定等权四 hop | TEST 0.788 | 未达门槛 |
| HOP_PH | 空间 PH 条件路由 | TEST 0.788 | 与 EQ 差值 <0.001，未达门槛 |
| FROZEN3D | 冻结原生 3D encoder | validation 相对 HOP_PH +0.002；TEST=0 | 开发门槛未通过 |
| OPT/A_MIX246 | 冻结后最佳正式候选 | TEST 0.791 | 未达 +0.010 |
| TRANSFER/R_MEAN | 多表示迁移数值最佳臂 | TEST 0.791 | 数值候选，最终冻结协议以原周期为准 |
| WAVG/EMA60 | 最终冻结的权重平均候选 | TEST 0.777 | 未晋级 |
| PTSCALE/P5K | 5K 预训练 | TEST 0.788 | 未达门槛 |
| PTSCALE/P10K | 10K 预训练 | TEST 0.793 | 当时主参考，后由 R10/新 BASE 区分 |
| PTSCALE/P20K | 20K 预训练 | TEST 0.785 | 训练更长没有自动改善 |
| FPDECAY/FP_DECAY | FP 损失衰减 | TEST 0.790 | 严格差值未达门槛 |
| 3DSHIFT/P_G10K | 周期几何位移 | TEST 0.783 | 未晋级 |
| FPM/CONCAT_FP、GATE_FP | 10K 后直接融合 Morgan FP | TEST 0.757、0.758 | 明显下降 |
| PH/CONCAT_PH、GATE_PH | 有限 Trimer PH 描述符 | TEST 0.782、0.781 | 未晋级；不是无限周期 PH |
| R10 | P10K 与 r10-style KRR 全局冻结融合，w=0.75 | TEST 0.804；相对 P10K +0.01027562964426787 | 当时固定候选达到门槛；不是新 BASE |

DEMOL、ET_SINGLE repair1 和 FROZEN3D 没有正式 TEST，不能以 validation 数值填补七任务正式对照。FP 的跨表示正结果、CLEAN 的 QL_DYN 内部匹配差值、以及 R10 的早期正向结果均保留其原协议含义，不扩张为其他机制或当前基线的普遍结论。

## 5. 早期路线压缩记录

- MCL-PH 20260921/r13-PAPER5 完成五任务五折五臂；PT 历史基准和 M_XATTN 的 macro5 均为 0.837，没有稳定增益。旧运行随后发现角度监督和 PH 系数域适用边界，旧分数保留为历史观测，不能当作修复后合同已通过。
- MCL-PH-NEXT r4—r6 依次推进直接来源、预训练和七任务正式复评，并继续验证 Xc 目标尺度与 LR；r6 候选被 validation 拒绝且没有新增正式 TEST，整体没有证实稳定改善，后续关闭。
- GLT-OPTIMIZATION 20261011/r1—r9 的 U1/UA_2D 等候选在约 0.765—0.789 区间，UA_2D 为 0.788；当时未达到相对 F_CE 的 +0.010 开发门槛，相关路线未晋级。
- 2026-10-18 F0 的 Morgan 条件化融合只完成 105 个开发 validation 单元，NULL、SHIFT、FILM 分别为 0.814、0.815、0.815，正式 TEST 未运行，因此七任务正式性能不可比较。
- 旧 SOTA 主线只发生取消和部分消费，没有完成正式 TEST；不能用其他周期、smoke 或训练中间结果补齐。

## 6. 结果定义和证据边界

- 正式性能只看冻结后的五折正式 TEST 预测。validation 用于选模和停止开发，不用于正式晋级。
- 每个任务的均值是五折 R² 算术平均；macro 是七任务等权平均。单次 seed 规则 seed=42+fold；不是五次独立 seed。
- 差值使用未舍入值计算，展示三位小数。舍入后同为 0.795 或 0.792 的候选，应以 JSON 中的完整值排序和判定门槛。
- 合并非键空间残差、阶段拆分、冻结迁移和整体 pretrain 对换都包含多个因素；臂间差值不能自动解释成某个几何、PH、PT 或 interaction 的单因素因果效应。
- 论文结果是七任务最佳模型包络。项目 Egc、Xc 的样本量与论文不全相同，预训练与评估协议等价性也未证明；正值只表示该数值较高。

当前七任务论文参考：Eea 0.934、Egb 0.923、Egc 0.911、Ei 0.839、EPS 0.787、Nc 0.856、Xc 0.423。EAT、Eib、Tg 不纳入当前七任务对照。

## 7. 证据入口

| 内容 | 原始结果 | 补充证据 |
|---|---|---|
| 当前 BASE 与 r4 空间实验 | results/ua3d_spatial_20261025/r4/final_review/results.json | results/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1/final_review/results.json |
| PT/FT 阶段拆分 | results/ua3d_stage_20261026/r2/final_review/results.json | results/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1/stage_acceptance.json |
| 冻结 3D 迁移 | results/ua3d_frozen_transfer_20261026/r1/final_review/results.json | results/ua3d_frozen_transfer_20261026/r1/final_review/closeout.md |
| UA3M-PaperFuse | results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/final_review/results.json | 同目录 report.md、budget.json、execution_manifest |
| TMOL3M 下游 3D 消融 | results/tmol3m_3d_ablation_20261025/r1/repairs/json_import_repair1/final_review/results.json | 同目录 report.md、budget.json、execution_manifest |
| 2026-10-18 至 10-23 UA2D 周期 | results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json 等各周期 final_review | logs 下同名周期根保留真实 exit 与作业日志 |
| MCL-PH 与 GLT 早期周期 | results/mcl_ph_20260921/、results/mcl_ph_r4_20261002/、results/glt_optimization_20261011/ | 对应 logs 根与 CURRENT_PLAN 历史交接 |
| 论文数值参考 | SOTA_RESULT.md | 数据、split 与统计限制见本文件第 6 节 |

本次仅压缩项目历史并把长期规则写入 AGENTS.md；未删除结果、checkpoint、失败现场或日志，未运行训练、模型预测、GPU 或测试，未读取 TODO.md，未做哈希验证或远程同步。
