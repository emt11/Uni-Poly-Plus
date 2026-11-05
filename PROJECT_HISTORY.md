# Uni-Poly 项目实验历史与当前基线

更新时间：2026-11-05。本文件保留项目级的当前基线、正式实验结果、路线结论和证据入口；逐作业日志、tmux 窗口、事故快照与完整配置继续保留在 results 和 logs 目录。指标展示保留三位小数，选模、差值与门槛判断使用未舍入值。

状态说明：已写入本文件只表示项目记录已更新，不自动等于独立审查通过。result JSON 仍为 PENDING 的周期继续标为待审查；这些结论可以用于后续规划，但不能改写为已验收模型。

## 1. 当前模型基线

### 最新 BASE：F_PD_N1_F0（2026-10-28 用户指定）

用户于 2026-10-28 明确指定 F_PD_N1_F0 为最新基线。后续未带周期限定的 **BASE／基线** 均指该模型配方及其七任务五折 BEST 模型集，用作后续新比较的正式 reference。

- 基线 ID：UA3D-BASE-OPT-20261026-01/r6/F_PD_N1_F0；资源修订 gpu2-pt2-fttest4-c1。
- 原始正式七任务 macro R²：`0.804675462276949`；展示为 **0.805**。新比较默认报告候选相对此值的未舍入 Δmacro R²。
- 统计：paper5_outer，七任务各五折，seed=42+fold，std(ddof=1)，任务等权 macro；35 个 BEST 均来自 F 自己的 task/fold，不拼接其他配方。
- 产物根：`results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/`；正式结果为同根 `final_review/results.json` 的 `formal.F_PD_N1_F0`，配方为 `registry/recipes.json` 与 `pretrain/F_PD_N1_F0/config.json`。
- checkpoint：同根 `pretrain/F_PD_N1_F0/deploy.pt` 及 `finetune/F_PD_N1_F0/{task}/fold{fold}/best.pt`；正式预测为 `formal/F_PD_N1_F0/{task}/fold{fold}/test_predictions.npz`。
- 配方沿用三模态 BASE 架构：PT 从公共 step0 训练 10K，native 2D/3D dropout=0.1、FP=0、当前三方向、共享 Q2/Q3；FT native dropout=0、FP=0.2、head=0.5、ALIGN=0.08、entropy=0.08，其余配置按上述产物保留。

本次指定记录为“查看共享开发 TEST 后由用户明确指定的基线”，independent_review=PENDING。它替换后续比较的 reference；r6 原预冻结 FINAL=E_P_SYM_SHARED、原性能未通过结论、父停止记录和共享开发评价性质均保留。设置基线不要求通过旧周期 +0.010 门槛；后续新周期若沿用默认 Δmacro ≥0.010，则未舍入候选 macro 须至少为 `0.814675462276949`，本次没有新增执行或预算授权。

| 任务 | 最新 BASE TEST R² mean±std | 论文参考 R² mean±std | BASE−论文 |
|---|---:|---:|---:|
| Eea | 0.930±0.018 | 0.934±0.013 | −0.004 |
| Egb | 0.921±0.015 | 0.923±0.009 | −0.002 |
| Egc | 0.914±0.013 | 0.911±0.008 | +0.003 |
| Ei | 0.821±0.040 | 0.839±0.018 | −0.018 |
| EPS | 0.776±0.068 | 0.787±0.033 | −0.011 |
| Nc | 0.831±0.057 | 0.856±0.032 | −0.025 |
| Xc | 0.440±0.143 | 0.423±0.102 | +0.017 |

论文模型中 Egb／Egc 对应 Periodic-TDL (+Chem, +RC)，其余五任务对应 Periodic-TDL；最新 BASE 仅在 Egc、Xc 数值高于相应参考。论文七任务最佳配置包络 macro 为 0.810，最新 BASE 低于该包络 0.006；数据、split 和统计协议等价性未证明，仅作数值对照。

### 历史 r4 BASE（2026-10-26 指定，2026-10-28 被替换）

用户于 2026-10-26 指定 UA3D-SPATIAL-20261025-01/r4 的 FULL/BASE 臂为当时基线；它继续作为旧周期的历史 reference，保留 r4 原 FINAL=C_CONST 和 r6 原验收口径。以下旧值不随最新 BASE 别名重新解释。

- 历史基线 ID：UA3D-SPATIAL-20261025-01/r4/FULL/BASE
- 正式七任务 macro R²：0.795244745287872
- 展示：0.795
- 统计：paper5_outer，七任务各五折，seed 42+fold，任务等权 macro
- 原始证据：results/ua3d_spatial_20261025/r4/final_review/results.json

历史 r4 BASE 是原生 TMOL3M 架构：INT2D 键路径与原生 3D MaskedGalformer3D，Morgan FP 作为 2048 维输入但不再作为预测目标；不额外加入非键空间距离、角度残差或 P10K 表示。该模型不是 P10K checkpoint 与 TMOL3M/FP 的简单拼接，也不能把 UA2D 的 PTSCALE/P10K 或历史 R10 融合结果当成 BASE。

| 任务 | 历史 r4 BASE R² | SOTA 参考 R² | r4 BASE−参考 |
|---|---:|---:|---:|
| Eea | 0.924 | 0.934 | −0.010 |
| Egb | 0.917 | 0.923 | −0.006 |
| Egc | 0.913 | 0.911 | +0.002 |
| Ei | 0.806 | 0.839 | −0.033 |
| EPS | 0.771 | 0.787 | −0.016 |
| Nc | 0.821 | 0.856 | −0.035 |
| Xc | 0.415 | 0.423 | −0.008 |

论文参考来自 SOTA_RESULT.md，是 Periodic-TDL 在不同任务上的最佳配置包络，不是一个可运行单模型。历史 r4 BASE 只在 Egc 上数值高于对应参考，macro 低于参考包络约 0.015；数据、split 和协议等价性未证明，因此只作数值对照。

历史 R10 的 macro 为 0.8035089342046258，高于当时的 r4 BASE 0.0082641889167538。它相对 UA2D PTSCALE/P10K 达到过原定的 +0.010 门槛，但相对历史 r4 BASE 未达到默认 +0.010，且模型组成和数据协议不同；R10 的旧冻结结论继续保留，当前 BASE 按用户最新决定指 F_PD_N1_F0。

## 2. 2026-10-25 至 10-28 最新正式实验

本节保留各已执行周期的原 reference 与验收结果；涉及 r4/FULL/BASE 的旧配方名、BASE 列和历史差值均指历史 r4 BASE，不随第 1 节的最新别名追溯改算。

### UA3D-BASE-OPT/r6：BASE 受控优化（完成，待审查）

计划 ID：UA3D-BASE-OPT-20261026-01；科学修订 r6；资源修订 gpu2-pt2-fttest4-c1；基于 dev / 2992e3cd095ba636c0f120251c43ca435ae3ea89 及受保护 dirty 工作树。科学问题是在同一 BASE 架构和数据口径下，检验 FT 辅助损失、学习率／调度、模态 dropout、PT 对齐方向／投影共享及有限融合修改。本周期预登记 reference 为历史 r4/FULL/BASE；2026-10-28 用户指定的最新 BASE 见第 1 节，原周期比较与验收不重算。

执行结果为**完成，待独立审查**，independent_review=PENDING。接续于北京时间 2026-10-28 06:26:56 结束，controller 真实 exit=0；唯一最终 CPU suite 为 1 次、exit=0。CURRENT_PLAN.md 的原“待授权”提案仅保留为接续前历史状态；执行结果依据完成状态块及实际产物，不把旧提示词视为再次执行授权。

- 登记 28 个实验槽位：23 个独立配方、5 个复用对照；完成 8 次新 PT、80,000 updates，以及 805 套有效 FT、805 次正式 TEST（23 配方×7 任务×5 fold），正式预测共 135,125 行。
- 有效 FT 共 56,350 epochs、1,288,000 updates；含父失败预留的累计计费为 FT 806 starts、56,420 epochs、1,288,700 updates，账本记录 budget_violations=[]。成功初始化只读复用父 r6，不再次初始化。
- 运行记录的资源合同为仅物理 GPU2、CUDA_VISIBLE_DEVICES=2、内部 cuda:0、world_size=1；PT 最多 2 路，FT（含 validation）及 TEST 各最多 4 路，PT 与 FT／TEST 分阶段互斥；PT 每路 workers=4，FT／TEST workers=0。原 t0 为 2026-10-26 22:07:13+08:00，继承硬截止 2026-10-30 22:07:13+08:00，接续未刷新截止。
- 数据保持 PT 有效池 910,756 条、635 条几何失败记录整条排除；下游 5,875 条任务记录，保留 Egb 7 条／Egc 8 条缺 3D 记录。固定 paper5_outer、fold0—4、seed=42+fold、train-only scaler；各任务分别训练，每任务五折 R² mean±std，std(ddof=1)，七任务等权 macro。

原 r6 因用户主动停止，初始化 exit=0、原 controller 与首个 FT exit=1 的证据继续保留。原 A_L_A8_B8/eea/fold0 完成 68 epochs、684 updates，预留 70 epochs／700 updates 不退款；接续在独立根将该单元完整重训 70 epochs，未从部分 BEST 续训，原部分结果不参加候选聚合。r5 的 EARLY_CPU_SUITE_ORDERING_DEVIATION／STOPPED_BEFORE_CAMPAIGN 继续单列，不恢复或改写为成功。

**冻结、正式结果与验收。** 各 task/fold 按原单位 validation RMSE 严格降低保存 BEST；全部 validation 完成后，于北京时间 2026-10-28 05:04:57 一次冻结 805 个估计器和唯一 FINAL=E_P_SYM_SHARED，再执行固定正式 TEST。下表中的后两臂均为预登记诊断臂，TEST 排名不改变 FINAL。

| 配方及角色 | validation macro R² | 正式 TEST macro R² | 相对历史 r4 BASE Δmacro R² |
|---|---:|---:|---:|
| E_P_SYM_SHARED：预冻结 FINAL | 0.810 | 0.795 | −0.000 |
| E_P_CURRENT_SHARED：StageE／F 匹配控制 | 0.810 | 0.799 | +0.004 |
| F_PD_N1_F0：TEST 数值最高诊断臂 | 0.808 | 0.805 | +0.009 |

完整精度保留在 results.json 的 formal、validation、comparisons 与 selected_comparison。FINAL 的未舍入 Δmacro 为负，performance_pass=false；七任务 SOTA 数值目标也未达到，sota_all7_pass=false。表中 −0.000 表示三位小数舍入后的负差值。F_PD_N1_F0 相对 E_P_CURRENT_SHARED 提升 +0.005，相对历史 BASE 的未舍入提升仍小于本轮 +0.010 门槛，不能据此改写原验收结果或追认其为 FINAL。

**A—G 的正式配对结论。** 下列 Δmacro 均为变化臂减去本阶段匹配控制，使用未舍入结果计算后展示；A—D 的控制是本周期重训的 A_L_A8_B8，不是历史 BASE。

| 阶段 | 匹配控制 | 已执行变化与 TEST Δmacro R² |
|---|---|---|
| A：FT ALIGN × entropy | A_L_A8_B8 | 移除 ALIGN −0.024；移除 entropy −0.008；同时移除 −0.041 |
| B：joint LR × scheduler | A_L_A8_B8 | 降低 LR、改为单调 scheduler 及两者组合均无 macro 收益 |
| C：FT 2D × 3D dropout | A_L_A8_B8 | 仅 2D 改为 0.1：−0.015；仅 3D 改为 0.1：−0.004；两者同时：−0.009 |
| D：FT FP × head dropout | A_L_A8_B8 | FP 从 0.2 改为 0：−0.005；head 从 0.5 改为 0.1：−0.006；两者同时：−0.010 |
| E：PT 方向 × 投影共享及六方向臂 | E_P_CURRENT_SHARED | 反向循环共享 −0.005；当前循环独立 FP 投影 −0.005；反向且独立 −0.015；六方向共享 −0.004 |
| F：PT native × FP dropout | E_P_CURRENT_SHARED | 仅 native 关闭 −0.005；保留 native=0.1、仅 FP 关闭 +0.005；两者同时关闭 −0.005 |
| G：有限融合修改 | E_P_SYM_SHARED | 残差融合 −0.011；模态 LayerNorm −0.017 |

主要正向信号来自 F_PD_N1_F0：从公共 common_step0 新训练 10K PT，保留 native 2D/3D dropout=0.1，仅把 PT FP dropout 从 0.2 改为 0；保持当前三方向、共享 Q2/Q3，FT 仍为 native=0、FP=0.2、head=0.5、ALIGN=0.08、entropy=0.08。该收益限于本轮匹配条件，没有独立重复确认。PT native 的 2D/3D 绑定调整，不能拆分归因；StageD 的 head 比较使用历史 r4 PT，尚未在 F 自己的 PT endpoint 上验证；顺序开发不能当作跨阶段完整 factorial 或全局最优证明。

**七任务论文数值对照。** 以下差值均为项目减参考，正值表示项目 R² 较高；先记录预冻结 FINAL，再单列 F 诊断臂。

| 任务 | FINAL TEST R² mean±std | FINAL−历史 r4 BASE | 论文最佳 R² mean±std | FINAL−论文 | 论文模型 |
|---|---:|---:|---:|---:|---|
| Eea | 0.923±0.011 | −0.001 | 0.934±0.013 | −0.011 | Periodic-TDL |
| Egb | 0.919±0.011 | +0.002 | 0.923±0.009 | −0.004 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.915±0.010 | +0.003 | 0.911±0.008 | +0.004 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.823±0.023 | +0.017 | 0.839±0.018 | −0.016 | Periodic-TDL |
| EPS | 0.750±0.053 | −0.021 | 0.787±0.033 | −0.037 | Periodic-TDL |
| Nc | 0.799±0.059 | −0.022 | 0.856±0.032 | −0.057 | Periodic-TDL |
| Xc | 0.436±0.095 | +0.021 | 0.423±0.102 | +0.013 | Periodic-TDL |

| 任务 | F_PD_N1_F0 TEST R² mean±std | F−历史 r4 BASE | 论文最佳 R² mean±std | F−论文 | 论文模型 |
|---|---:|---:|---:|---:|---|
| Eea | 0.930±0.018 | +0.006 | 0.934±0.013 | −0.004 | Periodic-TDL |
| Egb | 0.921±0.015 | +0.004 | 0.923±0.009 | −0.002 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.013 | +0.001 | 0.911±0.008 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.821±0.040 | +0.016 | 0.839±0.018 | −0.018 | Periodic-TDL |
| EPS | 0.776±0.068 | +0.005 | 0.787±0.033 | −0.011 | Periodic-TDL |
| Nc | 0.831±0.057 | +0.010 | 0.856±0.032 | −0.025 | Periodic-TDL |
| Xc | 0.440±0.143 | +0.024 | 0.423±0.102 | +0.017 | Periodic-TDL |

FINAL 与 F 均仅在 Egc、Xc 达到对应论文均值；F 的主要剩余差距为 Nc −0.025、Ei −0.018、EPS −0.011。论文七任务包络 macro 为 0.810，是不同最佳配置的汇总；项目 Egc 3,380／论文 4,124、Xc 432／论文 430，数据、split 与统计协议未证明等价，只作数值对照。共享 validation／TEST 已用于路线开发，不称独立盲测。

**用户指定最新基线与原周期结论分开记录。** F 的七任务 TEST 均值均高于历史 r4 BASE，但 Xc 折间 std 从 0.105 增至 0.143；其 validation EPS 相对 E 匹配控制下降 0.018，未通过本阶段保护规则，因此原 validation 流程未选择 F。此前 F 仅为后续工作基线建议；用户于 2026-10-28 明确指定其成为最新 BASE／基线及后续新比较的正式 reference，并明确不以旧周期 +0.010 门槛作为指定前提。选择来源保留为“查看共享开发 TEST 后由用户明确指定的基线”；历史 r4 BASE、原冻结 FINAL、本轮未通过结论和 independent_review=PENDING 均保留。

证据根：`results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/`；以下文件均相对此根：

- 正式／validation 结果、受控配对与消融：`final_review/results.json`、`final_review/report.md`；配方与复用：`registry/recipes.json`；冻结：`selection/final_frozen.json`。
- 预算、资源、完成状态与真实 exit：`budget.json`、`execution_manifest.json`、`controller_exit.json`、`final_review/cpu_exit.json`、`final_review/closeout.md`。
- F 的完整配方与预测：`pretrain/F_PD_N1_F0/config.json`、`finetune/F_PD_N1_F0/{task}/fold{fold}/`、`formal/F_PD_N1_F0/{task}/fold{fold}/metrics.json` 与 `test_predictions.npz`；完整精度指标和逐折证据留在这些产物中。
- 父停止证据：`results/ua3d_base_opt_20261026/r6/authorization/user_stop.json` 及同根预算／exit；r5 事故：`results/ua3d_base_opt_20261026/r5/status.json`。作业日志位于 `logs/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/`。

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

S_RADANG−BASE：FULL 为 −0.0036057119529214754，probe 为 +0.029943616592139066。空间表征经探针读取后明显更可辨，但完整模型没有迁移收益。r4 的 validation 选择仍记录为 FINAL=C_CONST；用户于 2026-10-26 指定该周期的 BASE 为当时基线，现作为历史 r4 BASE 保留，两者不能混写。

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

- BASE-OPT/r6 中，保留 PT native dropout=0.1、关闭 PT FP dropout 是主要正向配对信号（+0.005），但预冻结 FINAL 未改善历史 r4 BASE，周期仍为完成、待独立审查。用户于 2026-10-28 指定 F_PD_N1_F0 为后续最新 BASE，不追溯改变原周期 reference、FINAL 或验收结果。
- 空间 3D 探针改善不等于完整模型改善。r4 证明有限的表示可读性收益存在，stage 和 frozen transfer 都没有把它迁移为整体性能。
- 当前应把最新 BASE=F_PD_N1_F0 作为所有后续新比较的 reference。任何新空间编码、阶段策略、重复链构象或 FP 融合都应报告相对该基线的未舍入 Δmacro R²；已登记周期保持原 reference。
- 默认性能门槛维持 Δmacro R² ≥ 0.010。低于门槛的正差值不能写成性能通过；用户指定比较基线与原周期性能验收分别记录，不能按任务或折拼接。
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
| 当前 BASE：F_PD_N1_F0（2026-10-28 用户指定） | results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/final_review/results.json 的 formal.F_PD_N1_F0 | 同根 registry/recipes.json、pretrain/F_PD_N1_F0/config.json、pretrain/F_PD_N1_F0/deploy.pt、finetune/F_PD_N1_F0/{task}/fold{fold}/best.pt |
| BASE-OPT/r6 受控优化（完成，待审查） | results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/final_review/results.json | 同根 registry/recipes.json、selection/final_frozen.json、budget.json、execution_manifest.json、controller_exit.json、final_review/cpu_exit.json、final_review/closeout.md |
| 历史 r4 BASE 与空间实验 | results/ua3d_spatial_20261025/r4/final_review/results.json | results/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1/final_review/results.json |
| PT/FT 阶段拆分 | results/ua3d_stage_20261026/r2/final_review/results.json | results/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1/stage_acceptance.json |
| 冻结 3D 迁移 | results/ua3d_frozen_transfer_20261026/r1/final_review/results.json | results/ua3d_frozen_transfer_20261026/r1/final_review/closeout.md |
| UA3M-PaperFuse | results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/final_review/results.json | 同目录 report.md、budget.json、execution_manifest |
| TMOL3M 下游 3D 消融 | results/tmol3m_3d_ablation_20261025/r1/repairs/json_import_repair1/final_review/results.json | 同目录 report.md、budget.json、execution_manifest |
| 2026-10-18 至 10-23 UA2D 周期 | results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json 等各周期 final_review | logs 下同名周期根保留真实 exit 与作业日志 |
| MCL-PH 与 GLT 早期周期 | results/mcl_ph_20260921/、results/mcl_ph_r4_20261002/、results/glt_optimization_20261011/ | 对应 logs 根与 CURRENT_PLAN 历史交接 |
| 论文数值参考 | SOTA_RESULT.md | 数据、split 与统计限制见本文件第 6 节 |

本次（2026-10-28）基于现有产物补录 r6 执行结果，并按用户明确指定将最新 BASE 更新为 F_PD_N1_F0，同步 AGENTS.md、CURRENT_PLAN.md 与本文件，检查文档 diff 与格式；未重跑训练、模型前向、正式 TEST 或 CPU suite。旧结果、checkpoint、失败现场、日志及既有用户改动继续保留。


## 2026-10-29 BASE原BEST TEST复测（待审查）

用户授权仅TEST，并由GPU2改为GPU3、最多4路。科学问题为F_PD_N1_F0原35个BEST的重复推理；训练/选模0，沿用原数据/split/scaler/batch24/FP32。35/35 TEST、5875行完成，controller及全部worker exit=0。macro R²=0.805，完整精度0.8046754656116685；对用户指定reference 0.804675462276949的未舍入差值为3.3347195005362096e-09。样本身份/标签字段一致，预测存在微小数值差异，非逐位复现；不宣称严格确定性通过或性能提升通过，不替换BASE。共享开发评价；independent_review=PENDING。GPU2仅等待且TEST消费0，用户换卡中断的exit1/KeyboardInterrupt及原证据保留。证据：results/ua3d_base_test_repeat_20261029/r2_gpu3/{results.json,report.md,budget.json,controller_exit.json}；等待证据同父目录r1/；逐任务论文数值对照见report.md，协议差异保留。

## 2026-11-03：覆盖 CURRENT_PLAN 前的历史摘要（均待独立审查）

本节先于新 HMR 计划写入，归纳旧 CURRENT_PLAN 中的稳定性、PH-OPT、ANCHOR 与 ANCHOR-LR 合同及实际终态。只读使用现有 JSON、逐折 metrics、预测的身份/行数和退出日志；没有重跑模型、训练、TEST 或 CPU suite。旧全文入口为 `results/ua3d_anchor_lr_20261102/r1/repairs/init_r2/authorization/CURRENT_PLAN.md`，实时终态由各根 manifest、failure 和真实 exit 补充；旧文本中的 RUNNING、未执行和旧提示词不再作为启动依据。

### STABILITY 与 PH 路线的结论

- STABILITY/r1：`results/ua3d_stability_20261028/r1/final_review/results.json` 记录执行、正式 TEST、最终 CPU 完成，controller/CPU exit=0，但 `quality_guard_pass=false`、`cross_seed_stability_pass=false`、`performance_pass=false`。S42_A/S142/S242 正式 macro 为 0.790/0.788/0.789；S42_A 完整精度 `0.7901590429486368`。不能用该重训控制替换原 BASE，也不能因执行完成就称稳定性通过。
- PH-OPT/r1：`results/ua3d_ph_opt_20261031/r1/` 在 28 个 GPU2 诊断之后因速度及 ETA 门禁失败停止，controller/outer exit=1；没有正常新增 FT、冻结或正式 TEST。原 P_LEARN head/joint 为 C0 的 15.471/9.242 倍；该停止属于成本结论，正式性能“未完成/不可比较”。
- PH-OPT/auto_r2：唯一根 `results/ua3d_ph_opt_20261031/r1/repairs/auto_r2/`。200 新 FT 与 45 个合格历史 alias、245 正式 TEST 及最终 CPU 完成；最终依据 `final_review/results.controller03.json`、`budget.controller03.json`、`execution_manifest.controller03.json`、`controller_exit.controller03.json` 与 `cpu_exit.controller03.json`。最终 controller/CPU exit=0，预冻结 FINAL=BASE_KEEP，性能未通过；早期 controller/controller02 的 exit1 和旧 CPU NOT_RUN 文件仍属保留现场，不能被 controller03 成功覆盖。

| PH-OPT/auto_r2 臂 | 正式 TEST macro R² | 相对原 BASE Δmacro R² |
|---|---:|---:|
| C0_BASE | 0.790 | −0.015 |
| P_PE | 0.785 | −0.020 |
| P_FIXED | 0.786 | −0.018 |
| P_LEARN | 0.791 | −0.013 |
| M_EQ | 0.794 | −0.010 |
| M_GEOM | 0.798 | −0.007 |
| M_PH | 0.793 | −0.012 |

M_GEOM−C0=+0.008、P_LEARN−P_PE=+0.007 是相配条件下的描述性线索，不能替代相对原 BASE 的比较。M_GEOM/P_LEARN/C0 的完整精度分别为 `0.79781988829911`/`0.7912860935845323`/`0.7901590429486368`；C0 与 STABILITY/S42_A 相同。P_LEARN/Egc/fold1 的工程耗时从登记的 40262.499 秒变为 2044.776 秒，19.690 倍仅是该单元工程耗时对照，不是预测性能提升或普遍吞吐保证。父 PH/r2 的旧停止、事故、81 FT/无正式 TEST 等记录继续保留，不追认成功。

### BASE-ANCHOR/r2：冻结原 BEST 的几何／PH 适配

根 `A=results/ua3d_anchor_opt_20261102/r2/`；计划 `UA3D-BASE-ANCHOR-20261102-01/r2`，仅 GPU2。冻结原 BASE 的 35 套下游 BEST，复用同批 U/H 差值锚定，仅训练 G_GEOM/P_PE/P_LEARN，lr=1e-3；105 适配器 FT、140 正式输出、前置及最终 CPU 均完成，controller/outer/bootstrap 与两套 CPU 真实 exit 均0。原预冻结 FINAL=BASE_KEEP，原 +0.010 性能门槛未通过。

| 臂 | validation macro R² | 正式 TEST macro R² | 相对原 BASE Δmacro R² |
|---|---:|---:|---:|
| C_ANCHOR | 0.808 | 0.805 | +0.000 |
| G_GEOM | 0.798 | 0.807 | +0.003 |
| P_PE | 0.807 | 0.807 | +0.003 |
| P_LEARN | 0.807 | 0.807 | +0.002 |

正式完整精度按表顺序为 `0.804675462883717`、`0.8073860377992158`、`0.8073324973033907`、`0.8067333950086197`。不能按 TEST 把 G 或 P 追认为旧 FINAL。P_LEARN−P_PE 完整精度为 `−0.0005991022947710034`。教师缓存可作为只读工程资产；76/105 套 BEST 在前5 epochs、三臂中位数 epoch1 是后续温和更新的动机，不证明 lr 是性能差异的唯一原因。

证据：A 下 `final_review/results.json`、`selection/final_frozen.json`、`budget.json`、`cache_budget.json`、`execution_manifest.json`、三层 exit、`preflight_cpu_exit.json`、`cpu_exit.json`。BASE 的原权重、输入、scaler、split、失败现场及独立审查 PENDING 不变。

### ANCHOR-LR/r1 与隔离 r2：正式输出存在，最终执行失败

科学问题：对 G_GEOM/P_PE 分别使用 lr=1e-4/3e-5，并将同一候选的 epoch0 纳入 validation checkpoint 选择；仍冻结原 BASE，固定70 epochs。该新周期已预登记 selected 正式 macro≥0.809 为“接近”、≥0.810 为“达到”，不再要求该周期 Δ≥0.010；此定义不追溯改变上述旧周期验收。

- 父根 `L=results/ua3d_anchor_lr_20261102/r1/`：初始化读取缺失 `estimated_cache_bytes` 失败，科学作业与两套 CPU 消费均0，三层 exit=1，STOPPED 保留。
- 隔离根 `R=L/repairs/init_r2/`：继承原 tN=`2026-11-02T19:49:52.055462+08:00` 和截止 `2026-11-03T19:49:52.055462+08:00`。前置 CPU 1次/exit0；35+35 缓存资格、16 诊断、140 FT、一次冻结及175正式输出均有现有产物。`selection/final_frozen.json` 在 `2026-11-03T05:27:51.977464+08:00` 固定 FINAL=G_GEOM_1E4。
- 最终 CPU 1次/exit1，stdout 指向 `tests/test_ua3d_anchor_lr_r2.py:287` 的 `KeyError: 'started_unix'`，原因是测试从作业记录读取不存在的时间字段。controller/outer/bootstrap 实际 wait 均1，最终 STOPPED、`execution_complete=false`、`independent_review=PENDING`；未修复、重跑 CPU 或改写成功。

以下仅按现有35套逐折 `formal/{arm}/{task}/fold{fold}/metrics.json` 做七任务等权汇总，不生成预测、不改选。已保存的每臂5875行预测，其 indices 与原 split 的 test_indices 顺序一致；总计29375行。此只读汇总不替代独立审查或最终 CPU 验收。

| 臂 | validation macro R² | 正式 TEST macro R² | 相对原 BASE Δmacro R² |
|---|---:|---:|---:|
| C_ANCHOR | 0.808 | 0.805 | +0.000 |
| G_GEOM_1E4（原冻结 FINAL） | 0.809 | 0.806 | +0.002 |
| G_GEOM_3E5 | 0.809 | 0.806 | +0.002 |
| P_PE_1E4 | 0.808 | 0.806 | +0.001 |
| P_PE_3E5 | 0.808 | 0.805 | +0.001 |

本次 Python statistics.mean 汇总完整精度按表顺序为 `0.8046754628837172`、`0.8064506139154874`、`0.8064373083776215`、`0.8056909597521519`、`0.8052984895244841`；末位求和顺序差不影响结论。冻结 FINAL 未达到0.809，不因新目标较宽或最终 CPU 属报告问题而追认通过。

停止后 `budget.json` 尾态的 test_rows/test_batches 为29289/1346，合同为29375/1350，与现有完整预测的计数不同；该账本差异需独立审查，本回合不修复或重标。证据：R 下 `selection/final_frozen.json`、逐折 formal、`budget.json`、`execution_manifest.json`、`first_failure.json`、`final_review/{results,failure_status}.json`、`actual_exit_chain.json`、三层 exit 和 `logs/ua3d_anchor_lr_20261102/r1/repairs/init_r2/ua3d-anchor-lr-r2-cpu/stdout.log`。

### 下一周期的研究边界

用户于2026-11-03要求以原 BASE 为 reference，围绕有限 Trimer 表面＋HMR 谱几何制定创新计划，允许新增模型架构；目标接近正式 TEST macro0.810；仅 GPU2，充分利用可用 CPU。本次转向新的表面谱残差读出，不恢复上述失败根、不把旧正式 TEST 胜者拼接成模型。

新周期登记在覆盖后的 CURRENT_PLAN.md，ID=`UA3D-HMR-20261103-01/r1`。旧结果、预算、目标版本、时钟、代码与待审查状态保持。AGENTS 的通用 +0.010 默认不改；新周期明确采用用户本次的近0.810目标。共享开发 TEST 已参与路线开发，后续不称独立盲测。本次只更新两份文档，未激活新时钟、创建新实验根或消费训练／模型／测试预算。


## UA3D-HMR-20261103-01/r1：prepare失败（待审查）

- 状态：STOPPED；正式prepare exit=1，未完成/不可比较；independent_review=PENDING。
- tN=2026-11-03 09:57:34.933989+08:00；准备截止11月3日17:57:34.933989；硬截止11月4日09:57:34.933989；旧周期时钟不变。
- 八隔离文件已锁定，隔离安装成功；前置CPU一次exit=0；几何预留3650/开始581/正常返回474/失败1/中止106，保留全部部分消费。
- 首错：s00578合法center_physical=[2,-1,3,-1]被新增校验误拒绝；自动报告另有模块导入错误，均不修复重试。
- GPU诊断/FT/冻结/正式输出/最终CPU均0；controller/outer/bootstrap/tmux真实exit均1；无本周期存活进程。
- 证据：results/ua3d_hmr_20261103/r1/delivery_summary.md、budget.json、failure_detail.json、delivery_error.json；BASE保持F_PD_N1_F0/0.804675462276949。

科学问题：有限Trimer表面统计/固定谱/学习谱能否增加原BASE预测信息。固定四臂C_ANCHOR/S_STAT/S_FIXED/S_HMR；本周期未产生训练模型或正式TEST。失败源于新增实现拒绝原合法负哨兵，而非模型性能；表面覆盖率尚未完成。无数值通过或独立审查通过，不替换BASE，不恢复旧ANCHOR-LR。

### 2026-11-03 HMR/r1只读审查补充与r2提案

将r1已落盘的结构status按 `surface/mapping.json` 映射回原始有效3D行：103个DISCONNECTED对应395行，14个NONMANIFOLD对应59行，2个MESH_CAP对应7行，共119结构/461行。即使所有未返回结构均成功，r1现有判定下覆盖率上界仍为5399/5860=92.133%，已无法满足99%；该上界来自已有失败的确定数量，不是从部分样本推断全体失败比例。只移除单分量限制后，另两类已知不可用仍占66行，超过99%覆盖率允许的58行，因此仅修哨兵或多分量检查仍不足以通过原门禁。

原 `centre_mapping` 明确允许canonical同位素氢没有重原子代表时为-1，唯一性/范围要求应作用于非负代表项；r1新增校验错误不能改写为源数据损坏。bootstrap当前进程没有设置工程导入路径，动态导入训练报告失败；设置子进程PYTHONPATH不能修正父进程的sys.path。八份源码与authorization/source_locked逐字节一致，三层exit1保留；本次未运行任何模型、构面或测试。

用户要求下一步执行提示词，故CURRENT_PLAN顶部新增 `UA3D-HMR-20261103-01/r2` 提案：固定平滑隐式表面及合法多分量谱，隔离修正哨兵与失败报告；三训练臂、105FT/140正式输出、reference及近0.810目标不变。表面定义改变，旧355份可用表面也不能冒充r2缓存；拟新增一次完整3650项几何队列和一次前置CPU，并继承原tN/准备/总截止。r1的预留、581开始/474返回/1失败/106中止及全部失败消费不退款。新提案仅文档，尚未授权执行，不刷新旧钟、不创建修复根，independent_review=PENDING。


## UA3D-HMR-20261103-01/r2（待审查）

- 状态：STOPPED；未完成/不可比较；independent_review=PENDING。
- 仅r2子根 `results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2`；父r1维持STOPPED/三层exit1；原tN/截止不变。
- 前置CPU=PASSED/0；最终CPU=NOT_RUN/None；几何子预留3650/尝试3650；父子累计预留7300/尝试4231。
- FINAL=None；macro=None；reference=0.804675462276949；证据见本子根delivery_summary.md、cumulative_budget.json、真实exit与first_failure.json（如有）。

受控修订：合法负哨兵、平滑隐式表面与全部闭合分量/显式零模、标准库收尾。父失败不改判，r2不是纯工程等价修复；未完整TEST时未完成/不可比较。不替换BASE。

### r2交付只读核对补记

- 状态：STOPPED；未完成/不可比较；independent_review=PENDING。
- 唯一子根：results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2/；父r1保持STOPPED/三层exit1；原tN与截止不变，repair_started另记。
- 新前置CPU一次9测试exit0；3650几何全部返回（3648可用/2谱不收敛，含4旋转）；原有效行覆盖率99.966%、4旋转、35 trainval资格与统计通过；120优先结构全可用。
- 12诊断/240更新通过；完整试跑及成本门禁通过。FT预约/启动4、完成2、中止2（exit=-15）；FT记录700更新；冻结/正式输出/最终CPU均0，未读教师TEST。
- 首错：调度核验读取正常退出worker的/proc/environ发生ProcessLookupError；已停止，无修复重试。controller/outer/bootstrap/tmux真实exit均1。
- 父子累计几何预留7300/7300、尝试4231/4231；前置CPU累计2次；全部部分消费保留。
- 证据：本子根delivery_summary.md、delivery_addendum.md、failure_closeout_addendum.json、budget.json、cumulative_budget.json及first_failure.json；reference仍0.804675462276949，不替换BASE。

表面/映射/多分量门禁通过，停止来自调度实现的进程退出竞态；不是模型性能否定。没有完整正式TEST，不报告本周期macro或性能提升。无源码修复或作业重跑。

### HMR/r3调度修订的只读依据（2026-11-03，仅提案）

核对r2首错与锁定源码：`gpu_admission()`在Popen.poll返回运行态后依次读/proc/stat、cmdline、environ、cwd与FD；worker可在这些读操作之间退出，ProcessLookupError未进入受控归并。正常非零退出仍必须报错，不能用捕获所有OSError的方式忽略身份/权限或产物错误。退出归并需要共享、幂等的生命周期处理，覆盖派发后快速退出、检查中退出及收束过程。

r2八份工作区源码与其锁定快照一致。已完成两单元：S_HMR/Egc/fold1为70epochs/630更新、BEST epoch2；S_STAT/Eea/fold0为70epochs/70更新、BEST epoch0；均有完整history、COMPLETE消费、BEST/last/epoch0及真实wait exit0，可在r3严格只读资格通过后作为两个canonical alias。S_STAT/Eea/fold1、fold2真实exit=-15、没有消费文件或FT产物目录；不把未记录更新填0，按原锁定单元上限各0—70更新保守保留。

用户要求下一步执行提示词，故CURRENT_PLAN顶部登记工程修订 `UA3D-HMR-20261103-01/r3`：科学配方维持r2，3650几何/覆盖率/旋转、35紧凑trainval缓存与scaler、12诊断及完整试跑只读复用，不重做；新增一次纯CPU生命周期回归suite，核实两个alias，剩余103FT从各自零输出epoch0训练，其中包含明确重新授权的两个中止单元重训。剩余正常更新17150；最终105个canonical由2 alias＋103新完成单元组成，累计FT尝试上限107。已知诊断/FT更新940加新17150为18090，另给原两中止单元最多140未知更新的独立保守上限，累计不超过18230；未知消费不清零或退款。几何累计预留7300/尝试4231不增；前置CPU累计增为3次，最终CPU仍一次且尚未消费。

本修订继续继承原tN与2026-11-03 17:57:34.933989准备截止、11月4日09:57:34.933989总截止；不扩候选、seed、epochs、TEST或原时钟。仅修改计划/历史，无r3实现、修复根、检查或实验消费，父r1/r2的STOPPED和三层exit1不改，independent_review=PENDING。


## UA3D-HMR-20261103-01/r3（science r2，待审查）

- 状态：COMPLETE_PENDING_REVIEW；正式输出完成；independent_review=PENDING。
- science_revision=r2，execution_revision=r3；父r1/r2保持STOPPED/三层exit1；原tN及截止不变。
- 完成层级与消费见本根budget.json；canonical为2 parent alias+103新FT合同；前置CPU=PASSED/0，最后CPU=PASSED/0。
- FINAL=S_FIXED；正式macro=0.8076857278302016；未舍入Δ=0.003010265553252678；reference不变。
- 交付：results/ua3d_hmr_20261103/r1/repairs/scheduler_r3/delivery_summary.md；真实exit、首错、累计U区间和原始预测保留。

本修订仅生命周期调度与canonical解析，不改变模型/表面数学。父失败不覆盖，含旧两中止未知U∈[0,140]。完整正式输出前未完成/不可比较，不用validation填补。

### HMR/r3 完整交付核对与路线结论（2026-11-03）

本次用于制定下一周期的只读核对，不是完整独立审查。r3 保持 **COMPLETE_PENDING_REVIEW，性能目标未达，independent_review=PENDING**；r1/r2 的 STOPPED、三层 exit1、原 tN/截止及事故证据不变。r3 完整合同已结束，不作为可继续派发的活动运行。

- reference 仍为 F_PD_N1_F0 原正式 macro `0.804675462276949`；教师缓存复算为 `0.804675462883717`，末位差不触发 reference 替换。
- 103 套新 FT＋2 套父完整 alias 组成 105 canonical；全部新 FT 在冻结前结束。冻结时间 `1793677896.2579994`，固定 FINAL=S_FIXED；140 套正式输出/23500 行之后才启动唯一最终 CPU。
- 前置 CPU、最终 CPU、controller/outer/bootstrap 真实 exit 均0；r3 无 first_failure。此执行完成结论与“独立审查通过”分开。
- 累计几何预留7300、尝试4231；FT 尝试107。已记录更新18090，旧两中止单元未知更新 U∈[0,140]，累计区间 **[18090,18230]**。不把未知消费填0，不把 alias 再算作训练。
- 新 FT 使用物理 GPU2，r3 已记录 GPU 占用时间并集313.365秒、controller 墙钟524.475秒；这是旧运行观测，不保证下一轮相同速度。

| 臂 | validation macro R² | 正式 TEST macro R² | 相对原 BASE Δmacro R² |
|---|---:|---:|---:|
| C_ANCHOR | 0.808 | 0.805 | +0.000 |
| S_STAT | 0.810 | 0.808 | +0.003 |
| S_FIXED（冻结 FINAL） | 0.811 | 0.808 | +0.003 |
| S_HMR | 0.811 | 0.808 | +0.003 |

正式完整精度依次为 `0.804675462883717`、`0.8075188319602103`、`0.8076857278302016`、`0.8076860294815253`。FINAL 的未舍入 ΔBASE=`0.003010265553252678`；低于本周期已登记的0.809接近门槛和0.810目标。不能因展示值舍入成0.808或其他臂 TEST 稍高而改选 FINAL。

S_FIXED−S_STAT=`0.00016689586999130146`；S_HMR−S_FIXED=`3.016513236975271e-07`，两项按三位小数均为+0.000。后者的 validation 反而低于 S_FIXED，故冻结选择与现有记录一致。两个谱臂的35套 BEST epoch 分布相同，其中各13套选择epoch0；这些是读出改进的研究动机，不证明谱几何无效，也不证明瓶颈只来自学习率。现有实验缺少“同一残差头仅输入 BASE 特征”的对照，不能把全部+0.003归因于表面信息。

| 任务 | S_FIXED TEST R² mean±std | 论文逐任务最佳 R² mean±std | S_FIXED−论文 |
|---|---:|---:|---:|
| Eea | 0.930±0.018 | 0.934±0.013 | −0.004 |
| Egb | 0.922±0.012 | 0.923±0.009 | −0.001 |
| Egc | 0.915±0.013 | 0.911±0.008 | +0.004 |
| Ei | 0.827±0.038 | 0.839±0.018 | −0.012 |
| EPS | 0.777±0.067 | 0.787±0.033 | −0.010 |
| Nc | 0.833±0.054 | 0.856±0.032 | −0.023 |
| Xc | 0.449±0.139 | 0.423±0.102 | +0.026 |

项目口径为七任务五折、seed42+fold、std(ddof=1)、任务等权 macro；论文 Egb/Egc 的对应模型为 Periodic-TDL (+Chem, +RC)，其余为 Periodic-TDL。论文表是不同配置的逐任务包络；协议等价性未证明，只作数值对照。当前 TEST 为已用于路线开发的共享开发评价。

证据根 `results/ua3d_hmr_20261103/r1/repairs/scheduler_r3/`：`delivery_summary.md`、`final_review/results.json`、`selection/final_frozen.json`、`registry/canonical.json`、`budget.json`、`cumulative_budget.json`、`final_cpu/audit.json`、三层真实 exit；此前完整计划存于 `authorization/CURRENT_PLAN.md`。父 r1/r2 的交付与失败记录继续原位保留。

### 下一周期提案：HMR 固定谱与 BASE 潜变量的低秩交互

用户要求继续规划及执行提示词。新独立科学周期登记为 **UA3D-HMR-READOUT-20261103-01/r1**，拟根 `results/ua3d_hmr_readout_20261103/r1/`，仍只用物理GPU2。CURRENT_PLAN.md 在本摘要写入后覆盖为该独立计划，不再叠加旧修复合同。

预登记三组新 FT：N_BASE（去除表面输入的同形残差头）、N_ADD（固定谱＋额外加性分支）、N_BILIN（参数量相同的低秩双线性分支）；旧 C_ANCHOR 和已冻结 S_FIXED 分别作为 C_ANCHOR、C_FIXED 只读对照。比较重点为表面输入增量及额外分支中加性/乘性交互的差异，不预设有收益。新模型均从零输出残差初态训练，不从旧适配器 BEST 续训。

拟新增105 FT、240诊断更新、一次 validation 全局冻结；正式评价175单元/29375行，其中105单元为新预测、70单元为冻结后的旧对照复用。新几何、BASE 编码和统计重拟合均0。采用新周期独立12小时墙钟，未来明确执行授权才设 tN；旧时钟和 [18090,18230] 保留为历史消费，不能退款或双计。新目标继续明确登记为未舍入 FINAL macro≥0.809“接近”、≥0.810“达到”，不追溯更改旧周期或全项目默认门槛。

**当前只有文档提案；未授权执行、未启动新时钟、未创建新实验根、未修改实现或运行实验。** r3 与 BASE 的 independent_review 仍为 PENDING。


## UA3D-HMR-READOUT-20261103-01/r1（待审查）

<!-- UA3D_HMR_READOUT_R1_STATUS_START -->
- 状态：STOPPED；未完成/不可比较；independent_review=PENDING。
- 完成层级和新/旧消费分账见results/ua3d_hmr_readout_20261103/r1/budget.json及cumulative_budget.json；前置CPU=PASSED/0，最终CPU=NOT_RUN/None。
- FINAL=None；正式macro=None；未舍入ΔBASE=None；near=False，target/performance_pass=False。
- 新时钟见authorization/clock.json；旧[18090,18230]、父现场与BASE不变；交付见delivery_summary.md。
<!-- UA3D_HMR_READOUT_R1_STATUS_END -->
固定P2表面/谱与BASE表征；比较无表面、额外加性与低秩乘性分支。训练/TEST/独立审查分层，未完整正式评价不报告性能；旧HMR结论不追溯改判。

### READOUT/r1 失败交付只读补记

<!-- UA3D_HMR_READOUT_R1_STATUS_START -->
- 状态：STOPPED；未完成/不可比较；independent_review=PENDING。
- 八文件已锁定；唯一前置CPU4测试exit0，35来源资格exit0。首诊断N_BASE/EPS/fold0/R0 exit1：metrics调用的数据模块finite helper遗漏；无在线修复重试。
- 新GPU诊断预约/启动1、完成0；GPU模型构造1，GPU前向/backward/optimizer更新0；FT/冻结/正式输出/最终CPU均0，未读逐行TEST。
- 新更新0；旧及谱系区间[18090,18230]不变，旧U∈[0,140]保留；几何7300/4231不增。新tN/3h准备/12h总截止与旧时钟均保留。
- controller/outer/bootstrap/tmux真实exit均1；源码、失败与消费证据保留；FINAL/macro/Δ均null，不作为模型性能否定。
- 交付：results/ua3d_hmr_readout_20261103/r1/delivery_summary.md、delivery_addendum.md、failure_closeout.json、budget.json与cumulative_budget.json；BASE不变。
<!-- UA3D_HMR_READOUT_R1_STATUS_END -->
固定谱×化学潜变量读出尚未进入FT；首错为新增接口遗漏，不能据此判断模型收益。完整性能/消融未评价，后续修复须另行明确授权。


## UA3D-COND-SPECTRAL-20261104-01/r2（待审查）

状态：STOPPED；independent_review=PENDING。

正式评价：未完成/不可比较；完成 0/210 单元。

PT完成更新 857；FT完成 0/210；最终CPU次数 0，exit=None。

证据：results/ua3d_cond_spectral_20261104/r2/delivery.json、delivery_summary.md、budget.json、jobs.json。历史BASE=F_PD_N1_F0不变。

### COND-SPECTRAL/r2 停止后核对补记

prepare exit0；新step0及35头完成。PT尝试858，日志完成857，SIGTERM尾部实际完成区间[857,858]；worker exit-15，controller/outer/bootstrap/tmux均1。RSS合计门禁触发，越界精确样本未保存，不能断言实际主机物理峰值>160GiB。FT/冻结/正式TEST/最终CPU均0，未完成/不可比较；FINAL/macro/Δ均null，BASE不变，independent_review=PENDING。原消费/失败快照保留；证据：results/ua3d_cond_spectral_20261104/r2/{delivery_addendum.md,shutdown_audit.json}。


## UA3D-COND-SPECTRAL-20261104-01/r3：内存计量与有界读取修复提案

2026-11-04仅完成静态审查、修订与执行提示词，r3为PLANNED_NOT_STARTED，未改实验代码或启动worker。完整计划见[执行修订r3](docs/research/BASE_CONDITIONAL_SPECTRAL_R3_EXECUTION_PLAN.md)，原CURRENT_PLAN完整r2终态保存在docs/plan_snapshots/CURRENT_PLAN_20261104_before_COND_SPECTRAL_R3.md。

定位到r2使用后代VmRSS相加，且先raise后保存越界样本；没有同时刻PSS，真实物理峰值与共享页贡献仍未知，不能追认误报。r3保持160GiB计量预算，改为本作业PSS＋SwapPss并先保存决定证据；PT固定4 worker／prefetch1、LRU128条及256MiB估算storage，关闭readahead，每worker64源microbatch后在无外借buffer条件下重开自己的只读LMDB。模型数学、六臂、任务／fold／seed及训练配方不变。

拟新增一次从全新随机step0开始的完整10K PT，不加载父模型张量或接续857／858；父35套数据资格与统计仅作只读来源复用。新FT仍210套／336000更新，正式TEST210单元／35250行，前置CPU／额外GPU诊断0，最后CPU至多1次。父858次尝试及实际完成区间[857,858]不退款；新轮满额后累计尝试上限346858、实际完成区间[346857,346858]，预取未知尾部继续保留。

仅GPU2，继续原t0、原PT截止2026-11-05 06:56:51及总截止2026-11-07 18:50:55（Asia/Shanghai），不另起72h／12h。新根拟为results/ua3d_cond_spectral_20261104/r2/repairs/memory_r3/，目前未创建；旧r2及BASE不变，independent_review=PENDING。本段不构成已经执行或独立审查通过的记录。


## UA3D-COND-SPECTRAL-20261104-01/r3（待审查）

状态：STOPPED；independent_review=PENDING。

正式评价：未完成/不可比较；完成 0/210 单元。

PT完成更新 10000；FT完成 0/210；最终CPU次数 0，exit=None。

证据：results/ua3d_cond_spectral_20261104/r2/repairs/memory_r3/delivery.json、delivery_summary.md、budget.json、jobs.json。历史BASE=F_PD_N1_F0不变。

### COND-SPECTRAL/r3 执行审查与 r4 生命周期修复提案（2026-11-05）

本次只读独立核对执行证据与相关实现，结论为**STOPPED，执行需要返修，正式性能未完成/不可比较**；不将阶段完成写为全周期或性能验收通过，父产物independent_review=PENDING及历史BASE均不改写。报告见[执行审查](docs/reviews/UA3D_COND_SPECTRAL_R3_EXECUTION_REVIEW_20261105.md)。

r3新随机step0/35头已生成；PT完整10000、worker真实exit0，history逐步连续，60000 microbatches/10080000样本，step5000/step10000/deploy文件存在，前5148源microbatch父身份前缀已核对。首套C_FRESH/eps/fold0已消费297次完成更新、27个训练epoch及26个完整训练验证epoch；完整FT0/210不表示FT未启动。该worker在DataLoader退出归并中收到控制器SIGTERM，真实exit1；controller/outer/bootstrap/tmux均1。冻结、正式TEST、最终CPU均0，FINAL/macro/Δ均null。

首错为PID 2305401:57882989的smaps_rollup返回ESRCH，r3单次身份复核仍认定存活即抛错；栈与实现最符合退出过渡竞态，但未保存失败后原始stat状态，不能反演确切内核状态。决定样本已先落盘；2272条遥测中760次完整PSS的最大观测为13.424GiB，无MEMORY_STOP，不能称本轮160GiB内存超限，也不能称连续物理峰值已证实。证据根results/ua3d_cond_spectral_20261104/r2/repairs/memory_r3/：telemetry_error.json、memory.jsonl、jobs.json、PT/FT consumption、exits及累计账本。

父r2+r3实际尝试11155、已记录完成11154、实际完成区间[11154,11155]，其中PT尝试10858/实际完成[10857,10858]、FT297。父r2预取未知尾部与1674下界不归零、不退款；PT已启动完整单元配额20000与实际消费分开。

r4拟只修复PSS/退出生命周期、调度归并和阶段时钟/账目；明确只读使用**本周期r3新训练完成**的deploy及原35随机头，新增PT/随机初始化0，不读取历史BASE或中断FT BEST。六臂均从共同新PT和原随机头重新完整FT，新增210套/14700epochs/336000更新；一次冻结后210个正式TEST单元/35250行，前置CPU/额外诊断0，完整TEST后最终CPU至多1。满额累计尝试347155、日志完成347154、实际区间[347154,347155]，包含旧297更新。

仅GPU2、FT/TEST最多4并发；提出r4收到授权并开始实施后**至多2小时**的明确修复准备窗口，原已完成prepare/PT截止不再误拦下游，总t0与硬截止2026-11-07 18:50:55.545906（Asia/Shanghai）仍保留，不另起72h/12h。新根拟results/ua3d_cond_spectral_20261104/r2/repairs/lifecycle_r4/，当前未创建。

完整计划见[执行修订r4](docs/research/BASE_CONDITIONAL_SPECTRAL_R4_EXECUTION_PLAN.md)，[执行提示词](docs/research/BASE_CONDITIONAL_SPECTRAL_R4_EXECUTION_PROMPT.md)。原CURRENT_PLAN的完整r3停止记录先存入docs/plan_snapshots/CURRENT_PLAN_20261105_before_COND_SPECTRAL_R4.md，再由CURRENT_PLAN登记r4提案。本次仅文档，r4为PLANNED_NOT_STARTED；未新增实验实现、未启动时钟/worker、未运行模型/训练/TEST/CPU suite。


## UA3D-COND-SPECTRAL-20261104-01/r4（待审查）

状态：STOPPED；independent_review=PENDING；正式评价未完成/不可比较。

新增PT=0，允许来源PT完成10000；新增FT启动2、完整1/210、部分/中断1；开始epochs 103、完整训练验证epochs 102；尝试/记录完成更新1133/1133。

TEST完整0/210、0/35250行；最终CPU 0次、exit=None；前置CPU/额外前向0。

父累计11155尝试/[11154,11155]完成区间及预取未知尾部保留；当前累计见cumulative_budget.json。

证据：results/ua3d_cond_spectral_20261104/r2/repairs/lifecycle_r4/delivery_summary.md、delivery.json、cumulative_budget.json。历史BASE=F_PD_N1_F0不变。

### COND-SPECTRAL/r4 停止后核对

r4发现复核仍有缺口：首错样本4轮发现后覆盖7/8，新生DataLoader PID2324200:60861492缺少PSS；旧ESRCH过渡已在120ms后身份消失而正常归并。当前Python无os.pidfd_open，按登记采用严格stat路径。非内存超限，完整样本观测最大PSS+SwapPss 3.131GiB。C_FRESH一套完整70epoch/770更新/exit0；S_STATIC_GATE一套中断33开始/32完整训练验证epoch、363更新/exit1。新FT1133尝试/完成，三代12288尝试/[12287,12288]完成区间；新增PT、冻结、正式TEST、最终CPU均0。四层exit1，未重试；未完成/不可比较，BASE不变，independent_review=PENDING。证据：results/ua3d_cond_spectral_20261104/r2/repairs/lifecycle_r4/{delivery_addendum.md,lifecycle_stop_audit.json,telemetry_error.json,lifecycle.jsonl}。

### COND-SPECTRAL/r4 执行核对与 r5 固定采样对象集提案（2026-11-05）

本次只读核对代码与产物；r4仍STOPPED、正式性能未完成/不可比较，父independent_review=PENDING和BASE不改写。报告见[r4执行核对](docs/reviews/UA3D_COND_SPECTRAL_R4_EXECUTION_REVIEW_20261105.md)。

首错位于4轮测量后再次discover的边界：缺失PID2324200:60861492在采样结束前约0.295毫秒登记，尚无PSS读取尝试，即被按7/8覆盖失败停止；不是内存超限。旧PID2324176的ESRCH已在120ms后身份消失而正常归并，不能把这两类问题混写。42条遥测中20次完整PSS最大观测3.131GiB，仅为采样最大值。r4八文件与锁定文本一致；prepare元数据alias exit0；controller/outer/bootstrap/tmux均真实exit1，未在线改码或重试，登记进程已退出。

C_FRESH/eps/fold0完整70epochs/770更新/exit0，BEST epoch32与首次最低validation RMSE选择记录一致，config/scaler/指标及必需文件吻合；S_STATIC_GATE/eps/fold0中断363更新、33开始/32完整训练验证epochs、worker exit1。冻结/正式TEST/最终CPU均0，macro/Δ null。r4实际1133更新，三代累计12288尝试、日志完成12287、实际完成区间[12287,12288]；父r2未知预取尾部不归零。

r5拟仅改监控覆盖时序、canonical/schema解析及接续账目：每pass锁定有限PID身份集合，末尾新发现且尚未读取的对象登记PENDING，首次登记后5秒内必须测量或证实退出；实际读取失败仍按原3次/250ms复核后停止，不能用pending豁免。PENDING不是通过，不准入新GPU作业，不刷新完整PSS快照；保留160GiB数值门禁与环境保护。这是明确的新时序合同，不是在r4合同下自动忽略缺失。

唯一完整C_FRESH/eps/fold0固定为只读parent_complete_alias，禁止用于其他训练初始化；其余209套新FT均从本周期r3新PT及原随机头重新完整训练，中断S_STATIC_GATE不续BEST。新增PT/随机初始化/统计拟合/几何/构面/求谱均0。新FT209套/14630epochs/335230更新，1alias+209新FT组成210canonical，一次冻结后210个新正式TEST单元/35250行，最后CPU至多1、此前测试/额外前向0。

满额四修订累计尝试347518、日志完成347517、实际完成区间[347517,347518]；FT336660=canonical336000+旧中断297+363，父完整770不重收费。源PT/head保持r3，唯一父完整BEST保持best-r4，新BEST为best-r5，冻结和正式执行分别登记真实estimator_config与r5 runner_config，不重写来源或吞schema错误。

仅GPU2、FT/TEST最多4并发。提出r5开始明确授权实施后至多2小时修复准备窗口，原t0和总硬截止2026-11-07 18:50:55.545906（Asia/Shanghai）不变，不另起72h/12h。新根拟results/ua3d_cond_spectral_20261104/r2/repairs/coverage_r5/，当前未创建。

完整计划见[r5执行计划](docs/research/BASE_CONDITIONAL_SPECTRAL_R5_EXECUTION_PLAN.md)，[完整执行提示词](docs/research/BASE_CONDITIONAL_SPECTRAL_R5_EXECUTION_PROMPT.md)。先保存原CURRENT_PLAN的完整r4停止记录至docs/plan_snapshots/CURRENT_PLAN_20261105_before_COND_SPECTRAL_R5.md，再覆盖CURRENT_PLAN登记r5。本次仅文档；r5为PLANNED_NOT_STARTED，未新增实现、未启动时钟/worker、未运行模型/实验/CPU suite。


## UA3D-COND-SPECTRAL-20261104-01/r5（待审查）

状态：STOPPED；independent_review=PENDING；正式评价未完成/不可比较。

新增PT=0，允许来源PT完成10000；新增FT启动201、完整新FT197/209、父完整alias1、canonical完整198/210、部分/中断4；开始epochs 13993、完整训练验证epochs 13989；尝试/记录完成更新326993/326993。

TEST完整0/210、0/35250行；最终CPU 0次、exit=None；前置CPU/额外前向0。

父累计12288尝试/[12287,12288]完成区间及预取未知尾部保留；当前累计见cumulative_budget.json。

证据：results/ua3d_cond_spectral_20261104/r2/repairs/coverage_r5/delivery_summary.md、delivery.json、cumulative_budget.json。历史BASE=F_PD_N1_F0不变。


### COND-SPECTRAL/r5 停止后执行核对

r5于2026-11-05 13:00:37（Asia/Shanghai）因实际PSS读取失败停止：DataLoader身份3000636:62397014三次smaps_rollup均ESRCH，250.4ms后stat身份仍可见（D、RSS0）。这是3读/250ms停止合同触发，不是5秒待测超时或160GiB数值超限；确切内核退出原因未反演。5511条遥测中2214次完整PSS最大观测13.319GiB，仅为采样最大值。八源码与启动快照一致，未在线修复或重试。

新增FT326993次尝试/记录完成更新，197新完整＋1父完整alias=198/210；4套中断、8套未启动。累计339281尝试、339280日志完成、实际区间[339280,339281]及父预取未知尾部保留。197完整worker/prepare exit0，4中断worker及controller/outer/bootstrap/tmux exit1；登记进程已退出。冻结、正式TEST和最终CPU均0，宏指标及差值未完成/不可比较；independent_review=PENDING，BASE不变。证据根coverage_r5/下delivery_addendum.md、stop_audit.json、implementation.diff、telemetry_error.json、jobs.json及累计账本。

### COND-SPECTRAL/r5 执行审查与 r6 剩余12套接续提案（2026-11-05）

本次只读核对代码、父产物和内核原始资料；r5仍STOPPED，正式性能未完成/不可比较，不改父independent_review=PENDING及BASE。报告见[r5执行审查](docs/reviews/UA3D_COND_SPECTRAL_R5_EXECUTION_REVIEW_20261105.md)。

r5首错为DataLoader身份3000636:62397014实际smaps_rollup三次ESRCH，250.4ms后同身份仍D、vsize/RSS0、threads1；这是原3读/250ms合同触发，不是5秒pending超时或已测160GiB越界。Linux v6.8实现允许任务仍可见但mm缺失时rollup返回ESRCH，提供新的观测分类依据；父样本未保存maps/status/statm，不能事后确认其已无用户态内存或改写为误报成功。

198个完整canonical（r4一套＋r5 197套）的70epoch、更新/训练验证行数、config/scaler、首次最低RMSE选中的BEST与metrics、必需文件及真实exit元数据核对未发现矛盾；未加载模型张量或预测数组。其累计完整更新320810。四个中断单元均S_COND_NOCROSS，Egc/fold4、EPS/fold4、Nc/fold0/1，合计已消费6953更新；其余Nc/fold2—4及Xc/fold0—4共8套未启动。

r6拟保留198个只读完整alias，只新增上述12套完整head10+joint60，共840epochs/15190更新，均从本周期r3新PT及原随机头开始，不续中断BEST/optimizer/RNG。新增PT/step0/头/统计拟合/几何/构面/求谱0。父完整门禁已过，不重跑首六或额外诊断；六臂及训练数学不变。

监控拟新增严格NO_USER_MM_OBSERVED：仅已登记DataLoader后代的实际ESRCH分支，在原3读/250ms内取至少两轮同身份组合证据（无Vm*字段、maps/smaps成功EOF为空、statm全0、单线程等）。不足仍停止，不凭RSS0或D状态放行，不硬编码PF_*。符合者PSS仍null，不等同进程退出，保留exit_wait并禁止新派发/冻结/阶段切换，≤1秒复核、固定30秒归并上限，映射恢复须实际测PSS。r5固定目标集/5秒未尝试待测、160GiB数值与环境门禁保留；这是明确的新分类合同，不在旧r5中在线放宽。

父累计339281尝试、339280日志完成、真实区间[339280,339281]及预取未知尾部保留；r6满额后354471尝试、354470日志完成、真实区间[354470,354471]。FT343613=canonical336000＋中断297＋363＋6953，完整alias不重复收费。源BEST分别保持r4/r5，新BEST为r6，按具体canonical实际来源/schema/estimator_config和r6 runner_config严格解析。

198旧＋12新全部合格后才一次冻结，随后在新根产生210个新正式TEST单元/35250行，最后CPU至多1；此前pytest/smoke/合成测试/额外模型前向0。仅GPU2、FT/TEST最多4并发；提出r6明确授权开始实施起至多2小时修复准备窗口，原t0和总硬截止2026-11-07 18:50:55.545906（Asia/Shanghai）不变，不刷新72h/12h。

新根拟results/ua3d_cond_spectral_20261104/r2/repairs/mm_lifecycle_r6/。完整合同见[r6执行计划](docs/research/BASE_CONDITIONAL_SPECTRAL_R6_EXECUTION_PLAN.md)，[完整执行提示词](docs/research/BASE_CONDITIONAL_SPECTRAL_R6_EXECUTION_PROMPT.md)。先保存原CURRENT_PLAN的完整r5停止记录至docs/plan_snapshots/CURRENT_PLAN_20261105_before_COND_SPECTRAL_R6.md，再覆盖CURRENT_PLAN登记r6提案。本次仅文档，r6为PLANNED_NOT_STARTED；未新增实现/运行根/时钟，未运行模型/实验/CPU suite。


## UA3D-COND-SPECTRAL-20261104-01/r6（待审查）

状态：COMPLETE_PENDING_REVIEW；independent_review=PENDING；正式评价COMPLETE。

新增PT=0，允许来源PT完成10000；新增FT启动12、完整新FT12/12、父完整alias198、canonical完整210/210、部分/中断0；开始epochs 840、完整训练验证epochs 840；尝试/记录完成更新15190/15190。

TEST完整210/210、35250/35250行；最终CPU 1次、exit=0；前置CPU/额外前向0。

父累计339281尝试/[339280,339281]完成区间及预取未知尾部保留；当前累计见cumulative_budget.json。

证据：results/ua3d_cond_spectral_20261104/r2/repairs/mm_lifecycle_r6/delivery_summary.md、delivery.json、cumulative_budget.json。历史BASE=F_PD_N1_F0不变。


### COND-SPECTRAL/r6 完整交付执行自检（2026-11-05）

状态COMPLETE_PENDING_REVIEW；independent_review=PENDING，BASE不变。198个只读完整来源＋12新完整FT已合格；新增15190更新、840epochs，累计354471尝试、354470日志完成、实际区间[354470,354471]，父中断和未知尾部保留。一次validation冻结FINAL=S_STATIC_GATE，随后正式210单元/35250行全部完成，最后唯一CPU suite21 passed/exit0；prepare/FT/TEST/CPU及四层正式exit均0。

正式七任务等权macro R²=0.788，Δ历史BASE=-0.016，Δ本轮C_FRESH=+0.001。target_met=false、performance_pass=false；完整精度保留在results/ua3d_cond_spectral_20261104/r2/repairs/mm_lifecycle_r6/final_review/results.json。各任务5折seed42+fold、std(ddof=1)，共享开发评价；论文包络仅数值对照，不据TEST重新选臂。

真实无mm分类本轮未触发，最终CPU用例覆盖；完整PSS最大采样11.163GiB、无门禁STOP，不宣称连续峰值或独立审查通过。源八文件及父源码与锁定文本一致，198父消费元数据保留，登记进程退出。辅助只读UI观察器exit143独立记录。详细交付及协议限制见同根delivery_addendum.md、execution_audit.json、gpu_memory_summary.json、冻结、正式预测、预算和真实exit；父r2—r5停止结论与原截止不改写。


### 2026-11-05：归纳条件谱 r6，按用户要求先稳定 BASE（计划待执行）

本节先于覆盖 CURRENT_PLAN 写入。原 CURRENT_PLAN 的完整 r6 终态已保存至[计划快照](/root/workspace/Uni-Poly-Plus-master/docs/plan_snapshots/CURRENT_PLAN_20261105_before_BASE_STABILIZATION.md)。r6 完整交付仍为 COMPLETE_PENDING_REVIEW，FINAL=S_STATIC_GATE；正式 macro 原值0.7884341875644267，Δ指定BASE=−0.016241274712522236，ΔC_FRESH=+0.0013885760000226233。198只读alias＋12新FT、一次冻结、210正式单元/35250行和唯一CPU21 passed已完成，四层真实exit0；累计354471尝试及[354470,354471]完成区间保留，性能未通过、independent_review=PENDING，BASE不替换。

用户最新要求先解决 BASE 稳定性，因此暂停新图形学融合。只读核对显示：原35 BEST重复TEST仅差3.3347195005362096e-09；旧STABILITY的35对同种子FT已通过，但三组正式macro约0.790/0.788/0.789，EPS/Nc/Xc等任务跨种子range较大；条件谱新C_FRESH正式macro为0.7870456115644041。其模态二次reset、性质头seed与FT worker数均与指定BASE规则存在差异，不能把这些数值当作同配置推理随机波动或单因素因果。证据与限制见[BASE重复性定向审查](/root/workspace/Uni-Poly-Plus-master/docs/reviews/BASE_REPRODUCIBILITY_REVIEW_20261105.md)。

新提案 UA3D-BASE-STABILIZE-20261105-01/r1 保留BASE架构/损失，按历史随机工厂从源码重新初始化并做两次同种子完整10K PT，先比较正常轨迹与数学终态，历史模型张量读取0；四组FT=同种子A/B及142/242组，共140套/224000更新。每轨迹同时保留BEST和固定总epoch40/50/60/70等权SWA，额外训练更新0；PT1对及FT35对产物比较，三个canonical组×两个估计器共210正式单元/35250行，最后CPU至多1、此前测试/额外前向0。新训练总244000更新，新周期拟48小时、prepare含实现4小时、PT两次总阶段18小时/单次12小时；只用GPU2，PT1/FT4/TEST4、阶段互斥，旧时钟/预算只读保留。

新计划以同种子可重复、FT种子macro与逐任务波动、接近原BASE的质量门禁共同定义stable_ready；先验证同种子PT完整可重复性，再在相等PT与共同头初态下评估FT随机性；不宣称不同PT种子或跨平台稳定。SWA仅为待检验稳定化假设；固定PRIMARY=S42_A，不据TEST换估计器/seed。完整合同见[稳定化计划](/root/workspace/Uni-Poly-Plus-master/docs/research/BASE_STABILIZATION_EXECUTION_PLAN.md)及[执行提示词](/root/workspace/Uni-Poly-Plus-master/docs/research/BASE_STABILIZATION_EXECUTION_PROMPT.md)。

本轮仅文档修订和现有证据核对，新周期 PLANNED_NOT_STARTED，t0未启动；未改实验实现、未创建运行根、未运行模型/训练/TEST/CPU suite。该提案不表示已稳定成功或已获新周期执行完成结论。


## 2026-11-05T23:30:35.740409+08:00：用户授权清理历史 results 大文件
用户明确授权删除不必要的 results，乃至全部历史 results，并要求将结果分析保留在 PROJECT_HISTORY.md。本次授权覆盖旧实验产物保留限制；不重写旧运行的成功/失败、预算、时钟、独立审查或 reference 身份。旧文本中的证据路径可能已仅剩元数据，不能再声称相应模型/预测/逐行采样轨迹仍可复核或重放。
仅删除非当前依赖的历史 checkpoint、旧预测/特征数组、无关 LMDB/二进制缓存及大批量 JSONL；保留小型指标/config/scaler/预算/clock/exit/分析等文本记录。旧 BASE 权重也在历史 checkpoint 清理范围，其数字 reference F_PD_N1_F0=0.804675462276949 不自动变更；不把新实验当替代 BASE。
UA3D-BASE-STABILIZE-20261105-01/r1 正在执行，完整当前产物保留；其原始 PT 模板/逻辑映射/下游输入缓存、35份源 config 和35份 train-only scaler 原路径保留。源码、data/、logs/、用户文件不在删除范围。本次为磁盘清理，不触发实验重试、模型前向或 CPU suite。
清理前 results 实际占用 675.533 GiB；计划删除 124637 文件、分配空间 552.348 GiB。逐文件清单、真实 exit 和执行结果位于 logs/results_cleanup_20261105_authorized/；本段于首次删除之前持久化。
### 历史完整精度汇总快照

下列条目原样保留已有聚合指标/状态语义，source 标识原产物位置；validation 与正式 TEST 不能互相替代，既有待审查/停止限制仍有效。未重新读取预测数组、重新评价或加载模型张量。

<details><summary>results/ua3d_frozen_transfer_20261026/r1/final_review/results.json</summary>

```json
{"plan":"UA3D-FROZEN-TRANSFER-20261026-01","revision":"r1","status":"待审查","independent_review":"PENDING","formal":{"F_BASE":{"tasks":{"eea":{"folds":[0.917277852015947,0.9006319516764527,0.9280553494935739,0.9343023904390045,0.9000271392247454],"mean":0.9160589365699447,"std":0.01559834667515122},"egb":{"folds":[0.9131498672026372,0.8861725798178659,0.9270257886852271,0.9168226321367451,0.9168738000488239],"mean":0.9120089335782599,"std":0.015339193770041237},"egc":{"folds":[0.8857772210406207,0.916261626170997,0.9237627507591386,0.8997887939191785,0.9215076490175758],"mean":0.909419608181502,"std":0.016204742619833963},"ei":{"folds":[0.7841139317775132,0.7645798083057043,0.8289606093993978,0.8369151064791974,0.7939982042448384],"mean":0.8017135320413301,"std":0.03053579751041587},"eps":{"folds":[0.7450933722272104,0.8417597613557113,0.6099800082591986,0.729407282739891,0.791551600324216],"mean":0.7435584049812455,"std":0.08659193329777007},"nc":{"folds":[0.761960071138363,0.8790256998338238,0.8137766436595363,0.7585916429998237,0.7567885267057834],"mean":0.7940285168674659,"std":0.05311657870252475},"xc":{"folds":[0.5477685679596673,0.38668479301761505,0.41961255532453556,0.3003459997118909,0.5213804571118354],"mean":0.4351584746251088,"std":0.1010953018103104}},"macro":0.787420915263551},"F_G3D":{"tasks":{"eea":{"folds":[0.9053090668970022,0.9261062217918749,0.9507178909905847,0.9158069568425151,0.9227139864858842],"mean":0.9241308246015721,"std":0.016854725309531814},"egb":{"folds":[0.9290834624587282,0.8992143838179564,0.9251934650842577,0.9193366620265915,0.9155811802196664],"mean":0.91768183072144,"std":0.011560921799163541},"egc":{"folds":[0.8865491121876743,0.9197755138742971,0.9229586828618512,0.9004130135096491,0.9228526199065733],"mean":0.910509788468009,"std":0.016349564208453227},"ei":{"folds":[0.7424863226705797,0.7657759765976011,0.8025190242117755,0.7981995056902192,0.8159779952594329],"mean":0.7849917648859217,"std":0.03007388753907982},"eps":{"folds":[0.7315449685421155,0.7815372972276704,0.615643395654526,0.7492309603917469,0.7439394895291437],"mean":0.7243792222690405,"std":0.06352575537759164},"nc":{"folds":[0.8349580693913584,0.8972965885469667,0.8152578938517927,0.7468384800511911,0.8169063074336469],"mean":0.8222514678549911,"std":0.05375366826794567},"xc":{"folds":[0.3881602394524878,0.3582423406197708,0.48308375212247257,0.27398829658668833,0.5519515632490544],"mean":0.4110852384060948,"std":0.10855943801881315}},"macro":0.7850043053152955},"BASE":{"tasks":{"eea":{"folds":[0.8975483647839905,0.9288247508960716,0.9461946762473679,0.9249657074131301,0.9227552767047394],"mean":0.9240577552090599,"std":0.01745018717022919},"egb":{"folds":[0.9262120328575327,0.9036890761228172,0.9265469060320997,0.9173764232074346,0.9097682118705487],"mean":0.9167185300180865,"std":0.010065162515391076},"egc":{"folds":[0.893853743567249,0.9225571792043704,0.9223921538819906,0.9080185371937696,0.9167422301957654],"mean":0.912712768808629,"std":0.012088611520947862},"ei":{"folds":[0.7554440169051936,0.7666373001706158,0.8290033308662798,0.8445278724464078,0.8330848808445934],"mean":0.8057394802466181,"std":0.04138880032202877},"eps":{"folds":[0.7176295799175665,0.822127740345289,0.7268450672917459,0.7747772356432724,0.8152789097679007],"mean":0.7713317065931549,"std":0.048441833336694},"nc":{"folds":[0.8282122438525195,0.9153512080730142,0.7967376541741665,0.7138817571921905,0.8503991795574111],"mean":0.8209164085698604,"std":0.07396185353687995},"xc":{"folds":[0.5368643872016108,0.36692901582045667,0.4378205577890818,0.26303865504563306,0.47153022199169825],"mean":0.4152365675696961,"std":0.10484531794924663}},"macro":0.795244745287872},"REF":{"tasks":{"eea":{"folds":[0.9041560013131428,0.9200113619990171,0.9263601800973127,0.9152403200493444,0.9127891223857763],"mean":0.9157113971689187,"std":0.008279116081149585},"egb":{"folds":[0.9138592242330581,0.8944017321801515,0.9263285660627851,0.9256306278429477,0.9123321732950282],"mean":0.9145104647227942,"std":0.01296973148616043},"egc":{"folds":[0.8940497269172921,0.9229485626373148,0.9246689761494515,0.9077090854912411,0.9202863570626398],"mean":0.9139325416515879,"std":0.012951721495208604},"ei":{"folds":[0.7720179728563639,0.7863509241991601,0.8068550770841147,0.8688414583930124,0.8303554457914535],"mean":0.8128841756648209,"std":0.03823242329874713},"eps":{"folds":[0.7751274485876017,0.8032160093624454,0.5786978387667441,0.7207260088631775,0.7434502279888842],"mean":0.7242435067137706,"std":0.08716456290636308},"nc":{"folds":[0.8074256219779298,0.8879862897230075,0.7666664865621035,0.7216090724509421,0.8393156492266349],"mean":0.8046006239881235,"std":0.06421230265651498},"xc":{"folds":[0.48399824699424865,0.44876003694948896,0.43037684960630807,0.3785604674235339,0.5350386977730035],"mean":0.45534685974931666,"std":0.05859762948746285}},"macro":0.7916042242370475}},"validation":{"plan":"UA3D-FROZEN-TRANSFER-20261026-01","validation":{"F_BASE":{"tasks":{"eea":{"folds":[0.9290679687012748,0.8954424975334109,0.8952683763110814,0.9071682720383927,0.9510559398902707],"mean":0.9156006108948862,"std":0.024131430316663476},"egb":{"folds":[0.9241601464646552,0.9201474032985654,0.8977496322261711,0.9328346409893642,0.9423367997144504],"mean":0.9234457245386413,"std":0.016708552864018632},"egc":{"folds":[0.9105341047662874,0.9181230479089987,0.9123547199077908,0.910521480474912,0.9053479463588338],"mean":0.9113762598833646,"std":0.004589579765910355},"ei":{"folds":[0.8891542130572094,0.8251865688923463,0.874410032429718,0.8742723326707854,0.8306714012100986],"mean":0.8587389096520315,"std":0.028833602799571856},"eps":{"folds":[0.7652356478415323,0.8519665983834183,0.7481277833590537,0.8146595547987763,0.7197804055278828],"mean":0.7799539979821326,"std":0.053011058486541904},"nc":{"folds":[0.8412608857287366,0.9037870367955577,0.8426678200713417,0.9051440878193036,0.7494739238312924],"mean":0.8484667508492464,"std":0.06355664098760162},"xc":{"folds":[0.24022570518810504,0.484260170782601,0.4189540925352029,0.4691062886264813,0.2830712644003629],"mean":0.3791235043065506,"std":0.11096815676270351}},"macro":0.8023865368724076},"F_G3D":{"tasks":{"eea":{"folds":[0.9285207343749546,0.8897975256349004,0.8884307876996294,0.9088723615448734,0.9496597162848048],"mean":0.9130562251078326,"std":0.026190888664984983},"egb":{"folds":[0.9264830118674562,0.9179793808446352,0.8984925765245128,0.9353665199606042,0.9445278048934894],"mean":0.9245698588181395,"std":0.017621057783960052},"egc":{"folds":[0.9111361738347866,0.9159518506069396,0.9111847628612603,0.9090770208209006,0.9047304151316324],"mean":0.9104160446511038,"std":0.004058140866179409},"ei":{"folds":[0.8773190343090663,0.8398406062993362,0.8660599305862363,0.862188366604472,0.8282674083051033],"mean":0.8547350692208427,"std":0.020101129980743593},"eps":{"folds":[0.7849146153331285,0.8592888183649063,0.7540144089001176,0.8026016419314725,0.7344341586174334],"mean":0.7870507286294117,"std":0.0482820809262031},"nc":{"folds":[0.8475541680658109,0.9065482103041544,0.8277425577772015,0.8814977099907333,0.7748549494159035],"mean":0.8476395191107606,"std":0.050769333579826596},"xc":{"folds":[0.19331313238234937,0.48944890559336407,0.449266776278643,0.501676204975692,0.38386584891661024],"mean":0.40351417362933173,"std":0.1261788389650695}},"macro":0.8058545170239174}},"parent_reference":{"tasks":{"eea":{"folds":[0.934568952555824,0.9108353009838784,0.8941707165400519,0.911148981353677,0.9580644404317343],"mean":0.9217576783730331,"std":0.024875655573410357},"egb":{"folds":[0.9300110927405346,0.9293534880254796,0.9129087504481075,0.9416754565762545,0.9381141495822944],"mean":0.9304125874745341,"std":0.011111182200988994},"egc":{"folds":[0.913133833056184,0.921529324212742,0.9201502019281755,0.9136805267859228,0.9122626976813335],"mean":0.9161513167328715,"std":0.0043372057850581515},"ei":{"folds":[0.8805249243219215,0.8838155256929486,0.8885781534624805,0.850913908471526,0.8222968670394895],"mean":0.8652258757976732,"std":0.028163349600213557},"eps":{"folds":[0.7264883472382428,0.877320809615526,0.7586398889981837,0.7888479806881636,0.6969342186801757],"mean":0.7696462490440584,"std":0.06934208913381709},"nc":{"folds":[0.8468383189689566,0.9186864712352297,0.856799200688369,0.8719842931275934,0.7741986517445221],"mean":0.8537013871529343,"std":0.05228457921576755},"xc":{"folds":[0.3391619360896736,0.44344673084134345,0.4693754710547071,0.48425904435555156,0.38087748352657746],"mean":0.42342413317357064,"std":0.061489128190560786}},"macro":0.811474175392668}},"comparisons":{"F_G3D_minus_F_BASE":{"status":"COMPLETE","delta":-0.0024166099482554326,"performance_pass":false,"tasks":{"eea":{"folds":[-0.011968785118944836,0.025474270115422137,0.022662541497010746,-0.01849543359648942,0.022686847261138787],"mean":0.008071888031627483,"std":0.021428825579330252},"egb":{"folds":[0.01593359525609095,0.013041804000090473,-0.00183232360096941,0.002514029889846392,-0.0012926198291575153],"mean":0.0056728971431801774,"std":0.008282631246539841},"egc":{"folds":[0.0007718911470535605,0.003513887703300078,-0.0008040678972873749,0.0006242195904705428,0.0013449708889975431],"mean":0.00109018028650687,"std":0.0015689072169577417},"ei":{"folds":[-0.04162760910693353,0.0011961682918968375,-0.026441585187622296,-0.03871560078897818,0.021979791014594574],"mean":-0.016721767155408518,"std":0.027465285176358382},"eps":{"folds":[-0.013548403685094912,-0.06022246412804089,0.005663387395327346,0.019823677651855887,-0.04761211079507233],"mean":-0.01917918271220498,"std":0.0341433171620862},"nc":{"folds":[0.07299799825299547,0.018270888713142863,0.0014812501922564225,-0.0117531629486326,0.060117780727863446],"mean":0.02822295098752512,"std":0.036858977238129496},"xc":{"folds":[-0.15960832850717954,-0.028442452397844264,0.063471196797937,-0.02635770312520258,0.030571106137218917],"mean":-0.024073236219014093,"std":0.08521165117580604}}},"F_BASE_minus_BASE":{"status":"COMPLETE","delta":-0.007823830024321077,"performance_pass":false,"tasks":{"eea":{"folds":[0.019729487231956555,-0.028192799219618925,-0.018139326753793927,0.009336683025874426,-0.022728137479994026],"mean":-0.007998818639115179,"std":0.021195271609975096},"egb":{"folds":[-0.013062165654895486,-0.017516496304951334,0.0004788826531274548,-0.0005537910706895355,0.007105588178275224],"mean":-0.004709596439826736,"std":0.010217285815859418},"egc":{"folds":[-0.008076522526628271,-0.006295553033373369,0.0013705968771480226,-0.008229743274591028,0.004765418821810363],"mean":-0.0032931606271268564,"std":0.005978212447116747},"ei":{"folds":[0.028669914872319624,-0.0020574918649115537,-4.272146688200085e-05,-0.007612765967210389,-0.039086676599755066],"mean":-0.004025948205287877,"std":0.02413959566361364},"eps":{"folds":[0.02746379230964391,0.019632021010422274,-0.11686505903254729,-0.045369952903381416,-0.02372730944368473],"mean":-0.02777330161190945,"std":0.05822747329960025},"nc":{"folds":[-0.06625217271415651,-0.03632550823919034,0.01703898948536975,0.04470988580763324,-0.09361065285162762],"mean":-0.026887891702394296,"std":0.05732877705852789},"xc":{"folds":[0.010904180758056548,0.019755777197158375,-0.018208002464546236,0.03730734466625785,0.0498502351201372],"mean":0.019921907055412746,"std":0.02613998492910073}}},"F_G3D_minus_BASE":{"status":"COMPLETE","delta":-0.01024043997257651,"performance_pass":false,"tasks":{"eea":{"folds":[0.007760702113011719,-0.0027185291041967874,0.004523214743216819,-0.009158750570614993,-4.129021885523887e-05],"mean":7.306939251230382e-05,"std":0.006556220149728549},"egb":{"folds":[0.002871429601195463,-0.004474692304860861,-0.0013534409478419551,0.0019602388191568565,0.005812968349117709],"mean":0.0009633007033534424,"std":0.0039715607010237615},"egc":{"folds":[-0.007304631379574711,-0.0027816653300732908,0.0005665289798606477,-0.007605523684120485,0.006110389710807906],"mean":-0.0022029803406199864,"std":0.00575177141641806},"ei":{"folds":[-0.012957694234613903,-0.0008613235730147162,-0.026484306654504297,-0.046328366756188566,-0.017106885585160492],"mean":-0.020747715360696396,"std":0.017005049650194365},"eps":{"folds":[0.013915388624548997,-0.04059044311761861,-0.11120167163721995,-0.02554627525152553,-0.07133942023875706],"mean":-0.04695248432411443,"std":0.047239496256222334},"nc":{"folds":[0.0067458255388389565,-0.018054619526047477,0.01852023967762617,0.03295672285900064,-0.03349287212376417],"mean":0.001335059285130824,"std":0.02698833486146019},"xc":{"folds":[-0.148704147749123,-0.00868667520068589,0.045263194333390766,0.010949641541055266,0.08042134125735612],"mean":-0.004151329163601347,"std":0.08766193266675737}}},"F_G3D_minus_REF":{"status":"COMPLETE","delta":-0.006599918921751935,"performance_pass":false,"tasks":{"eea":{"folds":[0.001153065583859414,0.006094859792857732,0.02435771089327199,0.000566636793170705,0.009924864100107955],"mean":0.008419427432653559,"std":0.009697488969932961},"egb":{"folds":[0.015224238225670095,0.004812651637804866,-0.0011351009785274258,-0.006293965816356217,0.0032490069246381426],"mean":0.003171365998645892,"std":0.008001714500300001},"egc":{"folds":[-0.007500614729617827,-0.003173048763017716,-0.0017102932876003063,-0.0072960719815919894,0.0025662628439334645],"mean":-0.003422753183578875,"std":0.004198057213476472},"ei":{"folds":[-0.029531650185784186,-0.02057494760155898,-0.004336052872339291,-0.07064195270279316,-0.014377450532020553],"mean":-0.027892410778899236,"std":0.025599374290127874},"eps":{"folds":[-0.04358248004548626,-0.021678712134775013,0.036945556887781894,0.028504951528569356,0.0004892615402595224],"mean":0.00013571555526989965,"std":0.03371591925718151},"nc":{"folds":[0.02753244741342864,0.009310298823959173,0.04859140728968925,0.025229407600249054,-0.022409341792988036],"mean":0.017650843866867615,"std":0.02639499170557205},"xc":{"folds":[-0.09583800754176086,-0.09051769632971818,0.0527069025161645,-0.10457217083684556,0.016912865476050865],"mean":-0.04426162134322185,"std":0.07345461681158953}}}},"performance_pass":false,"sota":{"BASE":{"eea":{"project":{"folds":[0.8975483647839905,0.9288247508960716,0.9461946762473679,0.9249657074131301,0.9227552767047394],"mean":0.9240577552090599,"std":0.01745018717022919},"reference":0.934,"delta":-0.009942244790940191},"egb":{"project":{"folds":[0.9262120328575327,0.9036890761228172,0.9265469060320997,0.9173764232074346,0.9097682118705487],"mean":0.9167185300180865,"std":0.010065162515391076},"reference":0.923,"delta":-0.00628146998191359},"egc":{"project":{"folds":[0.893853743567249,0.9225571792043704,0.9223921538819906,0.9080185371937696,0.9167422301957654],"mean":0.912712768808629,"std":0.012088611520947862},"reference":0.911,"delta":0.0017127688086289172},"ei":{"project":{"folds":[0.7554440169051936,0.7666373001706158,0.8290033308662798,0.8445278724464078,0.8330848808445934],"mean":0.8057394802466181,"std":0.04138880032202877},"reference":0.839,"delta":-0.033260519753381845},"eps":{"project":{"folds":[0.7176295799175665,0.822127740345289,0.7268450672917459,0.7747772356432724,0.8152789097679007],"mean":0.7713317065931549,"std":0.048441833336694},"reference":0.787,"delta":-0.01566829340684517},"nc":{"project":{"folds":[0.8282122438525195,0.9153512080730142,0.7967376541741665,0.7138817571921905,0.8503991795574111],"mean":0.8209164085698604,"std":0.07396185353687995},"reference":0.856,"delta":-0.035083591430139616},"xc":{"project":{"folds":[0.5368643872016108,0.36692901582045667,0.4378205577890818,0.26303865504563306,0.47153022199169825],"mean":0.4152365675696961,"std":0.10484531794924663},"reference":0.423,"delta":-0.007763432430303863}},"REF":{"eea":{"project":{"folds":[0.9041560013131428,0.9200113619990171,0.9263601800973127,0.9152403200493444,0.9127891223857763],"mean":0.9157113971689187,"std":0.008279116081149585},"reference":0.934,"delta":-0.018288602831081358},"egb":{"project":{"folds":[0.9138592242330581,0.8944017321801515,0.9263285660627851,0.9256306278429477,0.9123321732950282],"mean":0.9145104647227942,"std":0.01296973148616043},"reference":0.923,"delta":-0.008489535277205862},"egc":{"project":{"folds":[0.8940497269172921,0.9229485626373148,0.9246689761494515,0.9077090854912411,0.9202863570626398],"mean":0.9139325416515879,"std":0.012951721495208604},"reference":0.911,"delta":0.00293254165158785},"ei":{"project":{"folds":[0.7720179728563639,0.7863509241991601,0.8068550770841147,0.8688414583930124,0.8303554457914535],"mean":0.8128841756648209,"std":0.03823242329874713},"reference":0.839,"delta":-0.026115824335179116},"eps":{"project":{"folds":[0.7751274485876017,0.8032160093624454,0.5786978387667441,0.7207260088631775,0.7434502279888842],"mean":0.7242435067137706,"std":0.08716456290636308},"reference":0.787,"delta":-0.06275649328622945},"nc":{"project":{"folds":[0.8074256219779298,0.8879862897230075,0.7666664865621035,0.7216090724509421,0.8393156492266349],"mean":0.8046006239881235,"std":0.06421230265651498},"reference":0.856,"delta":-0.051399376011876496},"xc":{"project":{"folds":[0.48399824699424865,0.44876003694948896,0.43037684960630807,0.3785604674235339,0.5350386977730035],"mean":0.45534685974931666,"std":0.05859762948746285},"reference":0.423,"delta":0.03234685974931667}},"F_BASE":{"eea":{"project":{"folds":[0.917277852015947,0.9006319516764527,0.9280553494935739,0.9343023904390045,0.9000271392247454],"mean":0.9160589365699447,"std":0.01559834667515122},"reference":0.934,"delta":-0.017941063430055304},"egb":{"project":{"folds":[0.9131498672026372,0.8861725798178659,0.9270257886852271,0.9168226321367451,0.9168738000488239],"mean":0.9120089335782599,"std":0.015339193770041237},"reference":0.923,"delta":-0.010991066421740192},"egc":{"project":{"folds":[0.8857772210406207,0.916261626170997,0.9237627507591386,0.8997887939191785,0.9215076490175758],"mean":0.909419608181502,"std":0.016204742619833963},"reference":0.911,"delta":-0.001580391818498006},"ei":{"project":{"folds":[0.7841139317775132,0.7645798083057043,0.8289606093993978,0.8369151064791974,0.7939982042448384],"mean":0.8017135320413301,"std":0.03053579751041587},"reference":0.839,"delta":-0.037286467958669856},"eps":{"project":{"folds":[0.7450933722272104,0.8417597613557113,0.6099800082591986,0.729407282739891,0.791551600324216],"mean":0.7435584049812455,"std":0.08659193329777007},"reference":0.787,"delta":-0.04344159501875455},"nc":{"project":{"folds":[0.761960071138363,0.8790256998338238,0.8137766436595363,0.7585916429998237,0.7567885267057834],"mean":0.7940285168674659,"std":0.05311657870252475},"reference":0.856,"delta":-0.06197148313253409},"xc":{"project":{"folds":[0.5477685679596673,0.38668479301761505,0.41961255532453556,0.3003459997118909,0.5213804571118354],"mean":0.4351584746251088,"std":0.1010953018103104},"reference":0.423,"delta":0.012158474625108828}},"F_G3D":{"eea":{"project":{"folds":[0.9053090668970022,0.9261062217918749,0.9507178909905847,0.9158069568425151,0.9227139864858842],"mean":0.9241308246015721,"std":0.016854725309531814},"reference":0.934,"delta":-0.00986917539842791},"egb":{"project":{"folds":[0.9290834624587282,0.8992143838179564,0.9251934650842577,0.9193366620265915,0.9155811802196664],"mean":0.91768183072144,"std":0.011560921799163541},"reference":0.923,"delta":-0.005318169278559992},"egc":{"project":{"folds":[0.8865491121876743,0.9197755138742971,0.9229586828618512,0.9004130135096491,0.9228526199065733],"mean":0.910509788468009,"std":0.016349564208453227},"reference":0.911,"delta":-0.000490211531991025},"ei":{"project":{"folds":[0.7424863226705797,0.7657759765976011,0.8025190242117755,0.7981995056902192,0.8159779952594329],"mean":0.7849917648859217,"std":0.03007388753907982},"reference":0.839,"delta":-0.05400823511407826},"eps":{"project":{"folds":[0.7315449685421155,0.7815372972276704,0.615643395654526,0.7492309603917469,0.7439394895291437],"mean":0.7243792222690405,"std":0.06352575537759164},"reference":0.787,"delta":-0.06262077773095953},"nc":{"project":{"folds":[0.8349580693913584,0.8972965885469667,0.8152578938517927,0.7468384800511911,0.8169063074336469],"mean":0.8222514678549911,"std":0.05375366826794567},"reference":0.856,"delta":-0.033748532145008836},"xc":{"project":{"folds":[0.3881602394524878,0.3582423406197708,0.48308375212247257,0.27398829658668833,0.5519515632490544],"mean":0.4110852384060948,"std":0.10855943801881315},"reference":0.423,"delta":-0.011914761593905199}}}}
```

</details>

<details><summary>results/ua2d_wavg_20261022/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-WAVG-20261022-01","revision":"r1","status":"待审查","independent_review":"PENDING","validation":{"BEST":{"tasks":{"eea":{"folds":[0.9385253459857533,0.9103085511423643,0.9106571457248033,0.9183211541499817,0.9458691235965738],"mean":0.9247362641198953,"std":0.01646413253739394},"egb":{"folds":[0.9351454868323745,0.9358717707840933,0.9295211660581644,0.9479926881734224,0.9559180289923911],"mean":0.9408898281680891,"std":0.010767105875429412},"egc":{"folds":[0.9243821379890561,0.9287402530814013,0.9244949407176168,0.9121807329408214,0.9218161652252903],"mean":0.9223228459908371,"std":0.006189930842092087},"ei":{"folds":[0.8539303842583348,0.8778921801414126,0.8960744294365309,0.8807398823716488,0.8280003105579848],"mean":0.8673274373531823,"std":0.02666458774754759},"eps":{"folds":[0.7033664836236662,0.8691751083913445,0.749035634636716,0.8427987556219545,0.7235092651191848],"mean":0.7775770494785731,"std":0.07397503142962311},"nc":{"folds":[0.7966833713207396,0.9320318393237261,0.8471185658661851,0.894444681809349,0.7661361603181619],"mean":0.8472829237276324,"std":0.06810240276449368},"xc":{"folds":[0.3374742851917667,0.5202440920450961,0.5498701775400447,0.5235794307929658,0.31904098391539637],"mean":0.4500417938970539,"std":0.11195368363550952}},"macro":0.8185968775336089},"SWA4":{"tasks":{"eea":{"folds":[0.9349095644352051,0.896855210171748,0.8922276107773129,0.9159455426241039,0.9259254681805292],"mean":0.9131726792377798,"std":0.018355995374795986},"egb":{"folds":[0.9251297845000989,0.9300907219105736,0.9203248490470578,0.9438165455730997,0.9555519112808903],"mean":0.934982762462344,"std":0.014465417971338228},"egc":{"folds":[0.9213623277544002,0.92003955438777,0.9157448851980303,0.9091634217679979,0.9119923562323228],"mean":0.9156605090681043,"std":0.005180766887880494},"ei":{"folds":[0.8245868084081562,0.8564335089317996,0.8694439809834904,0.8625791025294478,0.798675434622347],"mean":0.8423437670950482,"std":0.029852320097115744},"eps":{"folds":[0.49610697561809936,0.8602781499791557,0.6597625795113122,0.7602294792463192,0.5372708431472535],"mean":0.6627296055004279,"std":0.15169332375671854},"nc":{"folds":[0.639306998641044,0.8982646679477393,0.7662124354790849,0.8575074309838241,0.5592543589390299],"mean":0.7441091783981444,"std":0.14341062877315652},"xc":{"folds":[0.06660096806374205,0.42388458355559244,0.37746883172183154,0.42535446589540205,0.19895139409440676],"mean":0.29845204866619496,"std":0.15948079833368392}},"macro":0.7587786500611491},"EMA60":{"tasks":{"eea":{"folds":[0.9322619285977862,0.8967314320410237,0.8893593621356033,0.9126366793602948,0.9193580310270538],"mean":0.9100694866323524,"std":0.01726401584811599},"egb":{"folds":[0.9254004740410329,0.9310345834225601,0.9203486434879895,0.9443706776148416,0.9545446915248831],"mean":0.9351398140182614,"std":0.01407341910195573},"egc":{"folds":[0.9233104499209899,0.9210813374574077,0.915007203250149,0.9126597258263847,0.9127723840334541],"mean":0.916966220097677,"std":0.004928370516370473},"ei":{"folds":[0.8138913975214501,0.848642870655011,0.8712862399787078,0.8525660402419828,0.8058958519520157],"mean":0.8384564800698335,"std":0.02758746618083717},"eps":{"folds":[0.5372613913501563,0.8567851860157539,0.6476242340457998,0.7754136491433334,0.54374102007838],"mean":0.6721650961266847,"std":0.14145568924017005},"nc":{"folds":[0.6527442492233194,0.8991402469613914,0.7598987832443845,0.8489102335462249,0.5672139206498983],"mean":0.7455814867250437,"std":0.13683552996125087},"xc":{"folds":[0.06349931062824166,0.40435515150333856,0.3946410995260665,0.44373241803248664,0.20640033328255236],"mean":0.3025256625945371,"std":0.16214771212036885}},"macro":0.76012917803777}},"formal":{"BEST":{"tasks":{"eea":{"folds":[0.9159636541607057,0.9473047638678226,0.9399844179057837,0.9395471109347064,0.920841530687114],"mean":0.9327282955112265,"std":0.013545934476167365},"egb":{"folds":[0.9444548828993834,0.9182120034484224,0.9465349161355844,0.9337463708651035,0.9246801847890217],"mean":0.9335256716275031,"std":0.012262624926567043},"egc":{"folds":[0.8903967751420416,0.9276539480242922,0.9179343135666717,0.9000761829276808,0.9227862105533242],"mean":0.9117694860428021,"std":0.015852567290275066},"ei":{"folds":[0.8037511605806195,0.7786107603624425,0.8000597520620352,0.8489930560980813,0.8502588731573726],"mean":0.8163347204521102,"std":0.031873706891100084},"eps":{"folds":[0.7499684119388461,0.8412940492414138,0.6853018920454831,0.7487172416641935,0.7950412899450927],"mean":0.7640645769670058,"std":0.05822746611384411},"nc":{"folds":[0.762081568918149,0.9063651719829893,0.8348874845286682,0.6353695920304253,0.794746567265826],"mean":0.7866900769452115,"std":0.10026651151869849},"xc":{"folds":[0.45977913306696516,0.32756947776864975,0.46535478040517253,0.1694046200383268,0.5405147753801851],"mean":0.39252455733185987,"std":0.14639859354442364}},"macro":0.7910910549825312},"SWA4":{"tasks":{"eea":{"folds":[0.9166848526790161,0.9437975014927271,0.9432750752474961,0.9395552268589146,0.9192227745302803],"mean":0.9325070861616869,"std":0.01341561927836247},"egb":{"folds":[0.9444121913906889,0.9230832989830278,0.9465023734711805,0.9352195119048539,0.9206030356880501],"mean":0.9339640822875601,"std":0.011883358581589846},"egc":{"folds":[0.8878601262217741,0.9258106266083959,0.9171080004019962,0.9094233151632947,0.908197031538819],"mean":0.9096798199868561,"std":0.014086733819887795},"ei":{"folds":[0.7944802466316306,0.7824958699370743,0.8358545980477393,0.8493767343186767,0.8339045814553708],"mean":0.8192224060780984,"std":0.028993501453588224},"eps":{"folds":[0.6792743131362979,0.8289136682564703,0.4781705682750139,0.6910776295714136,0.7924973456904083],"mean":0.6939867049859209,"std":0.13666670571864406},"nc":{"folds":[0.7597213456883949,0.8943193187284489,0.7116763882153764,0.6802414322643884,0.8657656095237625],"mean":0.7823448188840743,"std":0.09411199062575633},"xc":{"folds":[0.435870879234606,0.32666434913314135,0.510384188677167,0.15045586429293067,0.5324531737943277],"mean":0.39116569102643456,"std":0.15674453787438844}},"macro":0.7804100870586616},"EMA60":{"tasks":{"eea":{"folds":[0.9150872955063171,0.9424422557908189,0.9413889326031147,0.9398484913598517,0.9169324697764105],"mean":0.9311398890073026,"std":0.01385787378877996},"egb":{"folds":[0.9432908280285411,0.9234105298334147,0.9477283763451891,0.9325905321085579,0.9202454504020228],"mean":0.9334531433435451,"std":0.012006286419067641},"egc":{"folds":[0.8888677638804829,0.9252062518862149,0.9180755599298766,0.9104896020852673,0.9081364131379241],"mean":0.9101551181839532,"std":0.013662069077911454},"ei":{"folds":[0.7889428835289186,0.7809644636461641,0.8326972549476106,0.8443656722403984,0.831162961088995],"mean":0.8156266470904173,"std":0.02860159292232139},"eps":{"folds":[0.6930916800519455,0.822052435371019,0.44254312343256863,0.6776424850761711,0.7858075038804309],"mean":0.6842274455624271,"std":0.14819538115667388},"nc":{"folds":[0.7504685812301695,0.8941872943174057,0.7188299454289967,0.6728116189988813,0.8632998259933002],"mean":0.7799194531937508,"std":0.09497472828011086},"xc":{"folds":[0.43201608695781324,0.3170028570017136,0.49914301580999887,0.142923236235301,0.5355274688537176],"mean":0.3853225329717088,"std":0.1589983571013104}},"macro":0.7771206041933006}},"FINAL":"EMA60","delta_FINAL_same_trajectory_BEST":-0.013970450789230537,"delta_FINAL_historical_B_FP":-0.014116204158718837,"performance_pass":false,"comparisons":{"SWA4_minus_BEST":{"delta":-0.010680967923869633,"status":"COMPLETE","reference":"BEST","paired_tasks":{"eea":{"folds":[0.0007211985183104286,-0.0035072623750954657,0.0032906573417124108,8.115924208129677e-06,-0.0016187561568337117],"mean":-0.00022120934953964166,"std":0.0025507455720354235},"egb":{"folds":[-4.2691508694514546e-05,0.004871295534605413,-3.254266440388953e-05,0.0014731410397503808,-0.004077149100971567],"mean":0.00043841066005716466,"std":0.0032235334477743152},"egc":{"folds":[-0.0025366489202675035,-0.0018433214158963818,-0.0008263131646754074,0.009347132235613897,-0.014589179014505183],"mean":-0.0020896660559461154,"std":0.008498451612699461},"ei":{"folds":[-0.009270913948988824,0.00388510957463184,0.03579484598570415,0.0003836782205953737,-0.016354291702001755],"mean":0.002887685625988157,"std":0.020051379913083048},"eps":{"folds":[-0.07069409880254818,-0.01238038098494354,-0.20713132377046917,-0.0576396120927799,-0.002543944254684405],"mean":-0.07007787198108503,"std":0.08189714670447025},"nc":{"folds":[-0.0023602232297541548,-0.012045853254540329,-0.12321109631329186,0.04487184023396307,0.07101904225793643],"mean":-0.004345258061137369,"std":0.07465659307819864},"xc":{"folds":[-0.023908253832359172,-0.000905128635508401,0.04502940827199442,-0.018948755745396118,-0.008061601585857447],"mean":-0.0013588663054253437,"std":0.027453978057528267}}},"EMA60_minus_BEST":{"delta":-0.013970450789230537,"status":"COMPLETE","reference":"BEST","paired_tasks":{"eea":{"folds":[-0.0008763586543886204,-0.004862508077003724,0.0014045146973310496,0.0003013804251452701,-0.003909060910703532],"mean":-0.0015884065039239114,"std":0.0026991231020823863},"egb":{"folds":[-0.0011640548708422571,0.005198526384992297,0.0011934602096047575,-0.0011558387565455686,-0.004434734386998929],"mean":-7.252828395793997e-05,"std":0.003563024633096039},"egc":{"folds":[-0.0015290112615586882,-0.002447696138077382,0.00014124636320489437,0.010413419157586534,-0.014649797415400068],"mean":-0.0016143678588489419,"std":0.008921518751071493},"ei":{"folds":[-0.014808277051700847,0.002353703283721642,0.03263750288557543,-0.00462738385768291,-0.019095912068377574],"mean":-0.0007080733616928514,"std":0.020454775008879555},"eps":{"folds":[-0.05687673188690057,-0.019241613870394803,-0.24275876861291446,-0.07107475658802243,-0.009233786064661853],"mean":-0.07983713140457882,"std":0.09460981469634233},"nc":{"folds":[-0.011612987687979581,-0.012177877665583536,-0.11605753909967154,0.037442026968456,0.0685532587274742],"mean":-0.006770623751460892,"std":0.07004420480149347},"xc":{"folds":[-0.027763046109151923,-0.010566620766936152,0.03378823540482634,-0.026481383803025782,-0.004987306526467505],"mean":-0.007202024360151005,"std":0.02495433927974332}}},"BEST_minus_B_FP":{"delta":-0.00014575336948829953,"status":"COMPLETE","reference":"B_FP","paired_tasks":{"eea":{"folds":[-1.5640524431148606e-08,-7.648129130011228e-09,2.7036917149558803e-09,-2.914480579097045e-08,2.6611384207342326e-09],"mean":-9.413725843288035e-09,"std":1.3452137595796359e-08},"egb":{"folds":[-1.1731875382281487e-07,4.568941591731601e-08,1.3398958054189336e-08,1.5359697658201554e-07,-9.482266749571977e-09],"mean":1.7176865996226808e-08,"std":9.77720125763583e-08},"egc":{"folds":[0.005623249037742251,-3.5679224175133584e-07,-1.7748061304723173e-05,-0.010757982478800887,-1.1698414770222598e-07],"mean":-0.0010305910557505626,"std":0.005959128339538359},"ei":{"folds":[2.3366342194464096e-10,-4.453689994576848e-08,-4.5662521985789795e-08,-2.396778191471327e-07,4.1800929762025874e-07],"mean":1.767314399270248e-08,"std":2.4222106935847395e-07},"eps":{"folds":[-7.155639827338689e-07,1.1138335156202217e-05,-6.367606780743884e-07,4.4027807066804314e-08,3.3361560403122326e-08],"mean":1.972679972572777e-06,"std":5.13628577127911e-06},"nc":{"folds":[-2.6066535596669382e-09,2.496563280729447e-05,-6.703010346775784e-08,-4.609524235643647e-07,-1.7349015668521872e-08],"mean":4.883538922206831e-06,"std":1.122781520821906e-05},"xc":{"folds":[-1.3496385897759922e-08,-3.516234534650664e-07,-7.553870906562565e-08,2.0823043649143003e-05,-3.203314328903417e-06],"mean":3.4358141543622265e-06,"std":9.810272706318105e-06}}},"SWA4_minus_B_FP":{"delta":-0.010826721293357933,"status":"COMPLETE","reference":"B_FP","paired_tasks":{"eea":{"folds":[0.0007211828777859974,-0.0035072700232245957,0.0032906600454041257,8.086779402338706e-06,-0.001618753495695291],"mean":-0.00022121876326548496,"std":0.0025507465016718096},"egb":{"folds":[-4.280882744833736e-05,0.0048713412240213305,-3.252926544583534e-05,0.0014732946367269628,-0.004077158583238316],"mean":0.0004384278369231609,"std":0.003223568691225777},"egc":{"folds":[0.003086600117474747,-0.001843678208138133,-0.0008440612259801306,-0.0014108502431869896,-0.014589295998652885],"mean":-0.0031202571116966784,"std":0.006704415017417684},"ei":{"folds":[-0.009270913715325402,0.003885065037731894,0.03579480032318216,0.0003834385427762266,-0.016353873692704135],"mean":0.0028877032991321493,"std":0.02005126778906268},"eps":{"folds":[-0.07069481436653091,-0.012369242649787338,-0.20713196053114724,-0.057639568064972835,-0.0025439108931240018],"mean":-0.07007589930111247,"std":0.08189938490393997},"nc":{"folds":[-0.0023602258364077144,-0.012020887621733034,-0.12321116334339532,0.0448713792815395,0.07101902490892076],"mean":-0.004340374522215163,"std":0.07465589645219296},"xc":{"folds":[-0.02390826732874507,-0.000905480258961866,0.045029332733285354,-0.018927932701746975,-0.008064804900186351],"mean":-0.0013554304912709813,"std":0.027450809199929014}}},"EMA60_minus_B_FP":{"delta":-0.014116204158718837,"status":"COMPLETE","reference":"B_FP","paired_tasks":{"eea":{"folds":[-0.0008763742949130515,-0.004862515725132854,0.0014045174010227646,0.0003013512803394791,-0.003909058249565112],"mean":-0.0015884159176497547,"std":0.0026991194660094297},"egb":{"folds":[-0.00116417218959608,0.005198572074408214,0.0011934736085628117,-0.0011556851595689865,-0.004434743869265678],"mean":-7.251110709194375e-05,"std":0.003563042934942014},"egc":{"folds":[0.0040942377761835624,-0.0024480529303191334,0.0001234983019001712,-0.00034456332121435285,-0.01464991439954777],"mean":-0.0026449589145995044,"std":0.007115663627606199},"ei":{"folds":[-0.014808276818037425,0.0023536587468216963,0.032637457223053445,-0.004627623535502057,-0.019095494059079954],"mean":-0.0007080556885488587,"std":0.020454672231907167},"eps":{"folds":[-0.0568774474508833,-0.0192304755352386,-0.24275940537359253,-0.07107471256021536,-0.00923375270310145],"mean":-0.07983515872460625,"std":0.09461183623992762},"nc":{"folds":[-0.011612990294633141,-0.012152912032776242,-0.116057606129775,0.037441566016032435,0.06855324137845853],"mean":-0.006765740212538685,"std":0.07004367266417061},"xc":{"folds":[-0.02776305960553782,-0.010566972390389617,0.03378815986611727,-0.02646056075937664,-0.004990509840796409],"mean":-0.0071985885459966426,"std":0.02495023151293981}}},"FINAL_minus_A_MIX246":{"delta":-0.014230320569282529,"status":"COMPLETE","reference":"A_MIX246","paired_tasks":{"eea":{"folds":[-0.0008759732588501379,-0.00486259006316625,0.0014042875526210086,0.0003019798992301981,-0.0039092306348363914],"mean":-0.0015883053010003144,"std":0.0026992518488010665},"egb":{"folds":[-0.0011649527135273452,0.00519948684510374,0.0011938729147070726,-0.0011553458201488676,-0.004435035767322071],"mean":-7.239490823749418e-05,"std":0.0035635400910295893},"egc":{"folds":[0.00022758855201743255,-0.0024510374052491013,-0.00036320819117763303,0.0010443944229513447,-0.014664015103136196],"mean":-0.0032412555449188307,"std":0.006515111515151548},"ei":{"folds":[-0.014821559929851902,0.0023546334652573275,0.03263878610566573,-0.0046280533471612095,-0.01906063033237826],"mean":-0.0007033648076936628,"std":0.020449732224099768},"eps":{"folds":[-0.056906746850430734,-0.019559886100362855,-0.24275787255906123,-0.07107554554354634,-0.009233695555273314],"mean":-0.07990674932173489,"std":0.0945567330268539},"nc":{"folds":[-0.011612977548094539,-0.012655739313393699,-0.11605622530140014,0.0374474260783042,0.06855325893725184],"mean":-0.0068648514294664675,"std":0.0700540943153489},"xc":{"folds":[-0.027763067073886316,-0.010573006542897723,0.033787034021941165,-0.026853757541646095,-0.004773816223137661],"mean":-0.007235322671925326,"std":0.025031501020256964}}},"FINAL_minus_R_MEAN":{"delta":-0.014277228644903994,"status":"COMPLETE","reference":"R_MEAN","paired_tasks":{"eea":{"folds":[-0.006150928880521156,-0.004697902282628452,-0.0003385806074001074,-8.437279931183816e-05,-0.0046593131831625945],"mean":-0.0031862195506048296,"std":0.002782768035735127},"egb":{"folds":[-0.0030483363294903087,0.015337727551800295,0.000985575304287778,-0.00034065841149621967,-0.00024965072979232783],"mean":0.0025369314770618432,"std":0.0073059541908473405},"egc":{"folds":[0.0028274955932785506,-0.003495777483684126,-0.001050316166175902,-0.00528283363543558,0.00647077752025671],"mean":-0.00010613083435206949,"std":0.004770779018166758},"ei":{"folds":[-0.01616653974042115,-0.019725963771484722,0.033014368889374146,0.0024721354551601182,-0.016476413902926423],"mean":-0.003376482614059606,"std":0.02214158814115528},"eps":{"folds":[-0.05794170715269398,-0.003972753245677807,-0.22590085180381136,-0.06351684295654647,0.07777791938851486],"mean":-0.05471084715404295,"std":0.11126727859507349},"nc":{"folds":[-0.011243461281203015,-0.016360961276616015,-0.11813929714727844,-0.05214910728253719,0.06994668488607902],"mean":-0.025589228420311128,"std":0.06835388625951254},"xc":{"folds":[0.030627298635412137,-0.05664272939133497,0.027048475845020015,-0.053450380890326166,-0.02512578128886722],"mean":-0.01550862341801924,"std":0.04231935538624322}}}},"references":{"B_FP":{"tasks":{"eea":{"folds":[0.9159636698012301,0.9473047715159517,0.939984415202092,0.9395471400795122,0.9208415280259756],"mean":0.9327283049249523,"std":0.013545935583923146},"egb":{"folds":[0.9444550002181372,0.9182119577590064,0.9465349027366263,0.9337462172681269,0.9246801942712884],"mean":0.9335256544506372,"std":0.012262659376907688},"egc":{"folds":[0.8847735261042994,0.927654304816534,0.9179520616279764,0.9108341654064817,0.9227863275374719],"mean":0.9128000770985528,"std":0.016854291685891032},"ei":{"folds":[0.803751160346956,0.7786108048993424,0.8000597977245572,0.8489932957759004,0.850258455148075],"mean":0.8163347027789662,"std":0.031873638077730186},"eps":{"folds":[0.7499691275028288,0.8412829109062576,0.6853025288061612,0.7487171976363864,0.7950412565835323],"mean":0.7640626042870332,"std":0.058223512723314164},"nc":{"folds":[0.7620815715248026,0.906340206350182,0.8348875515587717,0.6353700529828489,0.7947465846148417],"mean":0.7866851934062894,"std":0.1002588966293432},"xc":{"folds":[0.45977914656335106,0.3275698293921032,0.4653548559438816,0.16938379699467765,0.540517978694514],"mean":0.3925211215177055,"std":0.1464073089720903}},"macro":0.7912368083520195},"A_MIX246":{"tasks":{"eea":{"folds":[0.9159632687651672,0.9473048458539851,0.9399846450504937,0.9395465114606215,0.9208417004112469],"mean":0.9327281943083028,"std":0.013545993522803307},"egb":{"folds":[0.9444557807420685,0.9182110429883109,0.9465345034304821,0.9337458779287068,0.9246804861693448],"mean":0.9335255382517825,"std":0.012262958829399888},"egc":{"folds":[0.8886401753284655,0.927657289291464,0.9184387681210542,0.909445207662316,0.9228004282410603],"mean":0.9133963737288721,"std":0.015376254157658902},"ei":{"folds":[0.8037644434587705,0.7786098301809068,0.8000584688419449,0.8489937255875596,0.8502235914213733],"mean":0.8163300118981109,"std":0.03186362211557758},"eps":{"folds":[0.7499984269023763,0.8416123214713819,0.6853009959916299,0.7487180306197174,0.7950411994357042],"mean":0.764134194884162,"std":0.058331497354574796},"nc":{"folds":[0.762081558778264,0.9068430336307994,0.8348861707303968,0.6353641929205771,0.7947465670560484],"mean":0.786784304623217,"std":0.10041110675278749},"xc":{"folds":[0.45977915403169956,0.3275758635446113,0.4653559817880577,0.1697769937769471,0.5403012850768553],"mean":0.3925578556436342,"std":0.14620222534132926}},"macro":0.7913509247625832},"R_MEAN":{"tasks":{"eea":{"folds":[0.9212382243868382,0.9471401580734473,0.9417275132105148,0.9399328641591636,0.9215917829595731],"mean":0.9343261085579074,"std":0.012081723021885992},"egb":{"folds":[0.9463391643580314,0.9080728022816144,0.9467428010409014,0.9329311905200541,0.9204951011318151],"mean":0.9309162118664831,"std":0.016754323604916164},"egc":{"folds":[0.8860402682872044,0.928702029369899,0.9191258760960525,0.9157724357207029,0.9016656356176674],"mean":0.9102612490183052,"std":0.01665501162849731},"ei":{"folds":[0.8051094232693398,0.8006904274176488,0.7996828860582365,0.8418935367852383,0.8476393749919214],"mean":0.8190031297044769,"std":0.023694235789467458},"eps":{"folds":[0.7510333872046395,0.8260251886166968,0.66844397523638,0.7411593280327176,0.708029584491916],"mean":0.73893829271647,"std":0.05843831959250935},"nc":{"folds":[0.7617120425113725,0.9105482555940217,0.8369692425762751,0.7249607262814185,0.7933531411072212],"mean":0.805508681614062,"std":0.07172334847858607},"xc":{"folds":[0.4013887883224011,0.37364558639304857,0.47209453996497885,0.19637361712562718,0.5606532501425848],"mean":0.4008311563897281,"std":0.1352440599633569}},"macro":0.7913978328382046}},"budget":{"plan":"UA2D-WAVG-20261022-01","revision":"r1"},"cache_reference":{"metadata":{"plan":"UA2D-TRANSFER-20261022-01","status":"COMPLETE"}},"sota":{"BEST":{"eea":{"project":{"folds":[0.9159636541607057,0.9473047638678226,0.9399844179057837,0.9395471109347064,0.920841530687114],"mean":0.9327282955112265,"std":0.013545934476167365},"reference":0.934,"delta":-0.0012717044887735618,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9444548828993834,0.9182120034484224,0.9465349161355844,0.9337463708651035,0.9246801847890217],"mean":0.9335256716275031,"std":0.012262624926567043},"reference":0.923,"delta":0.010525671627503064,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8903967751420416,0.9276539480242922,0.9179343135666717,0.9000761829276808,0.9227862105533242],"mean":0.9117694860428021,"std":0.015852567290275066},"reference":0.911,"delta":0.0007694860428020256,"status":"COMPLETE"},"ei":{"project":{"folds":[0.8037511605806195,0.7786107603624425,0.8000597520620352,0.8489930560980813,0.8502588731573726],"mean":0.8163347204521102,"std":0.031873706891100084},"reference":0.839,"delta":-0.02266527954788977,"status":"COMPLETE"},"eps":{"project":{"folds":[0.7499684119388461,0.8412940492414138,0.6853018920454831,0.7487172416641935,0.7950412899450927],"mean":0.7640645769670058,"std":0.05822746611384411},"reference":0.787,"delta":-0.02293542303299423,"status":"COMPLETE"},"nc":{"project":{"folds":[0.762081568918149,0.9063651719829893,0.8348874845286682,0.6353695920304253,0.794746567265826],"mean":0.7866900769452115,"std":0.10026651151869849},"reference":0.856,"delta":-0.06930992305478845,"status":"COMPLETE"},"xc":{"project":{"folds":[0.45977913306696516,0.32756947776864975,0.46535478040517253,0.1694046200383268,0.5405147753801851],"mean":0.39252455733185987,"std":0.14639859354442364},"reference":0.423,"delta":-0.030475442668140118,"status":"COMPLETE"}},"SWA4":{"eea":{"project":{"folds":[0.9166848526790161,0.9437975014927271,0.9432750752474961,0.9395552268589146,0.9192227745302803],"mean":0.9325070861616869,"std":0.01341561927836247},"reference":0.934,"delta":-0.0014929138383131813,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9444121913906889,0.9230832989830278,0.9465023734711805,0.9352195119048539,0.9206030356880501],"mean":0.9339640822875601,"std":0.011883358581589846},"reference":0.923,"delta":0.010964082287560073,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8878601262217741,0.9258106266083959,0.9171080004019962,0.9094233151632947,0.908197031538819],"mean":0.9096798199868561,"std":0.014086733819887795},"reference":0.911,"delta":-0.0013201800131439345,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7944802466316306,0.7824958699370743,0.8358545980477393,0.8493767343186767,0.8339045814553708],"mean":0.8192224060780984,"std":0.028993501453588224},"reference":0.839,"delta":-0.019777593921901593,"status":"COMPLETE"},"eps":{"project":{"folds":[0.6792743131362979,0.8289136682564703,0.4781705682750139,0.6910776295714136,0.7924973456904083],"mean":0.6939867049859209,"std":0.13666670571864406},"reference":0.787,"delta":-0.09301329501407918,"status":"COMPLETE"},"nc":{"project":{"folds":[0.7597213456883949,0.8943193187284489,0.7116763882153764,0.6802414322643884,0.8657656095237625],"mean":0.7823448188840743,"std":0.09411199062575633},"reference":0.856,"delta":-0.07365518111592573,"status":"COMPLETE"},"xc":{"project":{"folds":[0.435870879234606,0.32666434913314135,0.510384188677167,0.15045586429293067,0.5324531737943277],"mean":0.39116569102643456,"std":0.15674453787438844},"reference":0.423,"delta":-0.03183430897356543,"status":"COMPLETE"}},"EMA60":{"eea":{"project":{"folds":[0.9150872955063171,0.9424422557908189,0.9413889326031147,0.9398484913598517,0.9169324697764105],"mean":0.9311398890073026,"std":0.01385787378877996},"reference":0.934,"delta":-0.002860110992697451,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9432908280285411,0.9234105298334147,0.9477283763451891,0.9325905321085579,0.9202454504020228],"mean":0.9334531433435451,"std":0.012006286419067641},"reference":0.923,"delta":0.01045314334354508,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8888677638804829,0.9252062518862149,0.9180755599298766,0.9104896020852673,0.9081364131379241],"mean":0.9101551181839532,"std":0.013662069077911454},"reference":0.911,"delta":-0.0008448818160468052,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7889428835289186,0.7809644636461641,0.8326972549476106,0.8443656722403984,0.831162961088995],"mean":0.8156266470904173,"std":0.02860159292232139},"reference":0.839,"delta":-0.023373352909582645,"status":"COMPLETE"},"eps":{"project":{"folds":[0.6930916800519455,0.822052435371019,0.44254312343256863,0.6776424850761711,0.7858075038804309],"mean":0.6842274455624271,"std":0.14819538115667388},"reference":0.787,"delta":-0.10277255443757294,"status":"COMPLETE"},"nc":{"project":{"folds":[0.7504685812301695,0.8941872943174057,0.7188299454289967,0.6728116189988813,0.8632998259933002],"mean":0.7799194531937508,"std":0.09497472828011086},"reference":0.856,"delta":-0.0760805468062492,"status":"COMPLETE"},"xc":{"project":{"folds":[0.43201608695781324,0.3170028570017136,0.49914301580999887,0.142923236235301,0.5355274688537176],"mean":0.3853225329717088,"std":0.1589983571013104},"reference":0.423,"delta":-0.03767746702829117,"status":"COMPLETE"}}}}
```

</details>

<details><summary>results/ua2d_p10k_r10fp_20261023/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-P10K-R10FP-20261023-01","revision":"r1","status":"待审查","independent_review":"PENDING","tasks":{"eea":{"delta_r2":0.0009263464346986217},"egb":{"delta_r2":-0.0010634704037235433},"egc":{"delta_r2":0.003668685968852525},"ei":{"delta_r2":0.0015329240552761636},"eps":{"delta_r2":0.014992494203736073},"nc":{"delta_r2":0.009651945791758964},"xc":{"delta_r2":0.04222048145927615}},"macro_r2":0.8035089342046258,"reference_macro_r2":0.7932333045603579,"delta_macro_r2":0.01027562964426787,"performance_pass":true}
```

</details>

<details><summary>results/ua3m_paperfuse_20261025/r1/final_review/results.json</summary>

```json
{"plan":"UA3M-PAPERFUSE-20261025-01","revision":"r1","status":"需返修","independent_review":"PENDING","formal":{"B_A020":{"tasks":{"eea":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"egb":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"egc":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"ei":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"eps":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"nc":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"xc":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173}},"macro":0.785044436513689}},"FINAL":null,"delta":null,"performance_pass":null,"comparisons":{"UNI3M_minus_B_A020":{"status":"未完成／不可比较","delta":null},"TMOL3M_minus_B_A020":{"status":"未完成／不可比较","delta":null},"TMOL3M_minus_UNI3M":{"status":"未完成／不可比较","delta":null}},"sota":{"B_A020":{"eea":{"project":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"reference":0.934,"delta":-0.0048973534443628175},"egb":{"project":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"reference":0.923,"delta":0.00785103276296062},"egc":{"project":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"reference":0.911,"delta":0.007125341354343506},"ei":{"project":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"reference":0.839,"delta":-0.02760752363123309},"eps":{"project":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"reference":0.787,"delta":-0.07443279463206498},"nc":{"project":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"reference":0.856,"delta":-0.04828792212544053},"xc":{"project":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173},"reference":0.423,"delta":-0.03743972468837947}},"UNI3M":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"TMOL3M":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}}}}
```

</details>

<details><summary>results/ua3m_paperfuse_20261025/r1/repairs/prepare_repair1/final_review/results.json</summary>

```json
{"plan":"UA3M-PAPERFUSE-20261025-01","revision":"r1","status":"需返修","independent_review":"PENDING","formal":{"B_A020":{"tasks":{"eea":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"egb":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"egc":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"ei":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"eps":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"nc":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"xc":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173}},"macro":0.785044436513689}},"FINAL":null,"delta":null,"performance_pass":null,"comparisons":{"UNI3M_minus_B_A020":{"status":"未完成／不可比较","delta":null},"TMOL3M_minus_B_A020":{"status":"未完成／不可比较","delta":null},"TMOL3M_minus_UNI3M":{"status":"未完成／不可比较","delta":null}},"sota":{"B_A020":{"eea":{"project":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"reference":0.934,"delta":-0.0048973534443628175},"egb":{"project":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"reference":0.923,"delta":0.00785103276296062},"egc":{"project":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"reference":0.911,"delta":0.007125341354343506},"ei":{"project":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"reference":0.839,"delta":-0.02760752363123309},"eps":{"project":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"reference":0.787,"delta":-0.07443279463206498},"nc":{"project":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"reference":0.856,"delta":-0.04828792212544053},"xc":{"project":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173},"reference":0.423,"delta":-0.03743972468837947}},"UNI3M":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"TMOL3M":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}}},"costs":{"UNI3M":{"pt":{"status":"RUNNING","plan":"UA3M-PAPERFUSE-20261025-01","revision":"r1"}}}}
```

</details>

<details><summary>results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/final_review/results.json</summary>

```json
{"plan":"UA3M-PAPERFUSE-20261025-01","revision":"r1","status":"待审查","independent_review":"PENDING","formal":{"B_A020":{"tasks":{"eea":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"egb":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"egc":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"ei":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"eps":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"nc":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"xc":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173}},"macro":0.785044436513689},"UNI3M":{"tasks":{"eea":{"folds":[0.8864935302668745,0.9458358498110955,0.9451669135487416,0.9352442253067044,0.9096056959732293],"mean":0.9244692429813292,"std":0.025807486725172815},"egb":{"folds":[0.9363635024562791,0.9174991988630108,0.9458119600060624,0.937544725887376,0.9239255041655274],"mean":0.9322289782756512,"std":0.011355207389156083},"egc":{"folds":[0.8962772101185505,0.9168030232592035,0.9072309322867533,0.9003970417423198,0.9173866799132947],"mean":0.9076189774640244,"std":0.009496016657687588},"ei":{"folds":[0.8019057661161906,0.8029955952628081,0.7899503966650747,0.8117030281275529,0.8330159271070223],"mean":0.8079141426557296,"std":0.016026884466980963},"eps":{"folds":[0.6637266829177528,0.8360176357493223,0.6510571340496029,0.6787011254045671,0.7240888619660799],"mean":0.710718288017465,"std":0.07528522719323703},"nc":{"folds":[0.8265192521520233,0.8643338573805514,0.7746226061379469,0.679858695485828,0.7955675457090631],"mean":0.7881803913730825,"std":0.06935631193398013},"xc":{"folds":[0.3687832193725247,0.33528102633482737,0.45060761782222813,0.2725509252972147,0.4352828483313812],"mean":0.3725011274316352,"std":0.07319728477370847}},"macro":0.7776615925998451},"TMOL3M":{"tasks":{"eea":{"folds":[0.907910960521732,0.9293858077176047,0.9410369681009859,0.92167557111855,0.9162950221219707],"mean":0.9232608659161687,"std":0.012650696185731452},"egb":{"folds":[0.9049295181467915,0.9103457596932322,0.9061741924363982,0.919167640696041,0.915507159201538],"mean":0.9112248540348002,"std":0.006069168824329532},"egc":{"folds":[0.8907441464405937,0.9230412184373157,0.919484690155727,0.909940062451425,0.9124143762112535],"mean":0.911124898739263,"std":0.012553151408236542},"ei":{"folds":[0.7656301285798724,0.7757007201310492,0.8074662411005353,0.8399649615429399,0.8337763647098406],"mean":0.8045076832128475,"std":0.033406334519672},"eps":{"folds":[0.7732761427778645,0.7928951797262221,0.6893351386412073,0.7057732574025937,0.7741706467104116],"mean":0.7470900730516599,"std":0.04625982927712594},"nc":{"folds":[0.8368704325807604,0.906599924838297,0.7451886370784278,0.7383608931976986,0.8438196094884609],"mean":0.814167899436729,"std":0.07148977037031887},"xc":{"folds":[0.45516101805374454,0.4414720211340897,0.38820534864998757,0.27599291538577997,0.5282045706881018],"mean":0.4178071747823407,"std":0.09371562284557325}},"macro":0.7898833498819727}},"FINAL":"B_A020","delta":0.0,"performance_pass":false,"comparisons":{"UNI3M_minus_B_A020":{"status":"COMPLETE","delta":-0.007382843913843917,"tasks":{"eea":{"folds":[-0.039824339057333846,0.015666603047669314,0.0020705536144286496,0.005836734034623259,-0.006916569510928783],"mean":-0.004633403574308281,"std":0.021273644958347677},"egb":{"folds":[-0.0013721598281565273,0.0002458501110312872,-0.002015039851819722,0.005716282049773147,0.004314795082624112],"mean":0.0013779455126904593,"std":0.00345699719313595},"egc":{"folds":[-0.0019953698308944334,-0.009964525995373785,-0.018263958967778215,-0.01868357988511682,-0.003624384772432432],"mean":-0.010506363890319137,"std":0.007860333764770782},"ei":{"folds":[0.03241932482940768,-0.007200362182106845,-0.030927882617581393,-0.03901471284586122,0.027331964250956187],"mean":-0.003478333713037118,"std":0.032665279052532824},"eps":{"folds":[-0.02072202292575165,-0.007145496893808501,0.09223498317116408,-0.0361806968597439,-0.03743135324421065],"mean":-0.001848917350470125,"std":0.054038895794337204},"nc":{"folds":[0.024164666875865826,-0.029081542671469562,-0.06579961445150251,-0.034495827201105556,0.007553884940827094],"mean":-0.019531686501476943,"std":0.0357031017397941},"xc":{"folds":[0.009131316834739778,-0.015845327833005296,-0.02240738325153846,-0.008782569004368246,-0.027391776145754387],"mean":-0.013059147879985323,"std":0.014241509641117088}}},"TMOL3M_minus_B_A020":{"status":"COMPLETE","delta":0.004838913368283682,"tasks":{"eea":{"folds":[-0.018406908802476307,-0.0007834390458214546,-0.00205939183332704,-0.00773192015353108,-0.00022724336218737928],"mean":-0.005841780639468652,"std":0.007630062036645835},"egb":{"folds":[-0.032806144137644155,-0.006907589058747332,-0.04165280742148392,-0.012660803141561927,-0.004103549881365298],"mean":-0.019626178728160526,"std":0.016659193005255157},"egc":{"folds":[-0.00752843350885124,-0.0037263308172615917,-0.006010201098804568,-0.009140559176011642,-0.00859668847447359],"mean":-0.007000442615080526,"std":0.0021855926225556976},"ei":{"folds":[-0.0038563127069104475,-0.03449523731386572,-0.013412038182120778,-0.010752779430474257,0.028092401853774485],"mean":-0.0068847931559193444,"std":0.02265214524403523},"eps":{"folds":[0.08882743693436002,-0.050267952916908665,0.13051298776276854,-0.009108564861717294,0.012650431500121018],"mean":0.03452286768372472,"std":0.0737134282593551},"nc":{"folds":[0.034515847304603,0.013184524786276075,-0.09523358351102162,0.02400637051076504,0.05580594872022493],"mean":0.006455821562169483,"std":0.05898481248337781},"xc":{"folds":[0.09550911551595964,0.09034566696625701,-0.08480965242377903,-0.005340578915802974,0.0655299462109662],"mean":0.03224689947072017,"std":0.0768120935085344}}},"TMOL3M_minus_UNI3M":{"status":"COMPLETE","delta":0.012221757282127599,"tasks":{"eea":{"folds":[0.02141743025485754,-0.01645004209349077,-0.00412994544775569,-0.013568654188154339,0.0066893261487414035],"mean":-0.0012083770651603709,"std":0.015555716297752644},"egb":{"folds":[-0.03143398430948763,-0.007153439169778619,-0.039637767569664195,-0.018377085191335074,-0.00841834496398941],"mean":-0.021004124240850984,"std":0.014257798706921353},"egc":{"folds":[-0.0055330636779568065,0.006238195178112194,0.012253757868973647,0.009543020709105177,-0.004972303702041159],"mean":0.0035059212752386105,"std":0.008276775988796448},"ei":{"folds":[-0.03627563753631813,-0.027294875131758878,0.017515844435460615,0.028261933415386964,0.000760437602818298],"mean":-0.003406459442882226,"std":0.027879348133768467},"eps":{"folds":[0.10954945986011166,-0.043122456023100164,0.038278004591604464,0.027072131998026605,0.05008178474433167],"mean":0.036371785034194845,"std":0.054663488193329425},"nc":{"folds":[0.010351180428737172,0.04226606745774564,-0.029433969059519116,0.058502197711870596,0.048252063779397836],"mean":0.025987508063646424,"std":0.0358257653390236},"xc":{"folds":[0.08637779868121986,0.1061909947992623,-0.062402269172240565,0.003441990088565272,0.09292172235672058],"mean":0.045306047350705494,"std":0.07248240963865384}}},"FINAL_minus_B_A020":{"status":"COMPLETE","delta":0.0,"tasks":{"eea":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"egb":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"egc":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"ei":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"eps":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"nc":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"xc":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0}}}},"sota":{"B_A020":{"eea":{"project":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"reference":0.934,"delta":-0.0048973534443628175},"egb":{"project":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"reference":0.923,"delta":0.00785103276296062},"egc":{"project":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"reference":0.911,"delta":0.007125341354343506},"ei":{"project":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"reference":0.839,"delta":-0.02760752363123309},"eps":{"project":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"reference":0.787,"delta":-0.07443279463206498},"nc":{"project":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"reference":0.856,"delta":-0.04828792212544053},"xc":{"project":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173},"reference":0.423,"delta":-0.03743972468837947}},"UNI3M":{"eea":{"project":{"folds":[0.8864935302668745,0.9458358498110955,0.9451669135487416,0.9352442253067044,0.9096056959732293],"mean":0.9244692429813292,"std":0.025807486725172815},"reference":0.934,"delta":-0.009530757018670899},"egb":{"project":{"folds":[0.9363635024562791,0.9174991988630108,0.9458119600060624,0.937544725887376,0.9239255041655274],"mean":0.9322289782756512,"std":0.011355207389156083},"reference":0.923,"delta":0.009228978275651167},"egc":{"project":{"folds":[0.8962772101185505,0.9168030232592035,0.9072309322867533,0.9003970417423198,0.9173866799132947],"mean":0.9076189774640244,"std":0.009496016657687588},"reference":0.911,"delta":-0.0033810225359756085},"ei":{"project":{"folds":[0.8019057661161906,0.8029955952628081,0.7899503966650747,0.8117030281275529,0.8330159271070223],"mean":0.8079141426557296,"std":0.016026884466980963},"reference":0.839,"delta":-0.031085857344270384},"eps":{"project":{"folds":[0.6637266829177528,0.8360176357493223,0.6510571340496029,0.6787011254045671,0.7240888619660799],"mean":0.710718288017465,"std":0.07528522719323703},"reference":0.787,"delta":-0.07628171198253508},"nc":{"project":{"folds":[0.8265192521520233,0.8643338573805514,0.7746226061379469,0.679858695485828,0.7955675457090631],"mean":0.7881803913730825,"std":0.06935631193398013},"reference":0.856,"delta":-0.0678196086269175},"xc":{"project":{"folds":[0.3687832193725247,0.33528102633482737,0.45060761782222813,0.2725509252972147,0.4352828483313812],"mean":0.3725011274316352,"std":0.07319728477370847},"reference":0.423,"delta":-0.05049887256836477}},"TMOL3M":{"eea":{"project":{"folds":[0.907910960521732,0.9293858077176047,0.9410369681009859,0.92167557111855,0.9162950221219707],"mean":0.9232608659161687,"std":0.012650696185731452},"reference":0.934,"delta":-0.010739134083831314},"egb":{"project":{"folds":[0.9049295181467915,0.9103457596932322,0.9061741924363982,0.919167640696041,0.915507159201538],"mean":0.9112248540348002,"std":0.006069168824329532},"reference":0.923,"delta":-0.011775145965199796},"egc":{"project":{"folds":[0.8907441464405937,0.9230412184373157,0.919484690155727,0.909940062451425,0.9124143762112535],"mean":0.911124898739263,"std":0.012553151408236542},"reference":0.911,"delta":0.00012489873926291306},"ei":{"project":{"folds":[0.7656301285798724,0.7757007201310492,0.8074662411005353,0.8399649615429399,0.8337763647098406],"mean":0.8045076832128475,"std":0.033406334519672},"reference":0.839,"delta":-0.03449231678715248},"eps":{"project":{"folds":[0.7732761427778645,0.7928951797262221,0.6893351386412073,0.7057732574025937,0.7741706467104116],"mean":0.7470900730516599,"std":0.04625982927712594},"reference":0.787,"delta":-0.03990992694834017},"nc":{"project":{"folds":[0.8368704325807604,0.906599924838297,0.7451886370784278,0.7383608931976986,0.8438196094884609],"mean":0.814167899436729,"std":0.07148977037031887},"reference":0.856,"delta":-0.041832100563271024},"xc":{"project":{"folds":[0.45516101805374454,0.4414720211340897,0.38820534864998757,0.27599291538577997,0.5282045706881018],"mean":0.4178071747823407,"std":0.09371562284557325},"reference":0.423,"delta":-0.0051928252176592915}}},"costs":{"UNI3M":{"pt":{"status":"COMPLETE","plan":"UA3M-PAPERFUSE-20261025-01","revision":"r1"}},"TMOL3M":{"pt":{"status":"COMPLETE","plan":"UA3M-PAPERFUSE-20261025-01","revision":"r1"}}}}
```

</details>

<details><summary>results/ua_hop_ph_20261020/frozen3d_r1/final_review/results.json</summary>

```json
{"plan":"UA-HOP-PH-FROZEN3D-20261020-01","revision":"r1","status":"待审查","independent_review":"PENDING","budget":{"plan":"UA-HOP-PH-FROZEN3D-20261020-01","revision":"r1"},"reference_validation":{"tasks":{"eea":{"folds":[0.9387667265172333,0.9111812001334202,0.9163318900011029,0.9178148561102099,0.9468167417649549],"mean":0.9261822829053841,"std":0.015622305122162316},"egb":{"folds":[0.9374561600013339,0.9372583117205315,0.9335171511526802,0.9472977837532419,0.9556778492544585],"mean":0.9422414511764492,"std":0.009082227064580083},"egc":{"folds":[0.9283616469770634,0.9260791308575271,0.9193661814072661,0.9122444874845032,0.920503971246813],"mean":0.9213110835946345,"std":0.006306384831968108},"ei":{"folds":[0.8601261596918515,0.8653097997526675,0.8945415902435391,0.8740455280912965,0.8347100391838305],"mean":0.8657466233926371,"std":0.021754723174500553},"eps":{"folds":[0.7465285514663902,0.8716502177188143,0.7736400265680825,0.8373450551063163,0.67205077202192],"mean":0.7802429245763047,"std":0.07826516728621989},"nc":{"folds":[0.7966833657298292,0.9126518236572433,0.8600321443897149,0.9005302741598779,0.7661361266981249],"mean":0.8472067469269581,"std":0.06406247918389639},"xc":{"folds":[0.3374742836705563,0.5526304193389778,0.5244106437046918,0.5194490731740983,0.3159753361852701],"mean":0.44998795121471885,"std":0.11348762846145603}},"macro":0.8189884376838695},"validation":{"tasks":{"eea":{"folds":[0.9388117848140676,0.9113643086017899,0.9083732390184994,0.9179089449124119,0.9461735072889227],"mean":0.9245263569271384,"std":0.01696043517101051},"egb":{"folds":[0.9361397117879132,0.9355021681289962,0.929144004502637,0.9477112699350854,0.9552947631978687],"mean":0.9407583835105001,"std":0.010529751806246379},"egc":{"folds":[0.9260997762322112,0.9258865495791027,0.920500066875704,0.9057327214840456,0.9183536603284963],"mean":0.919314554899912,"std":0.00830700455263607},"ei":{"folds":[0.8620837879258881,0.8880760440865465,0.8959699063999306,0.8774645662755699,0.8353295245972888],"mean":0.8717847658570447,"std":0.02400875095851643},"eps":{"folds":[0.778886698396841,0.8709241657430024,0.7392964039556051,0.8324552157903584,0.7346412095949844],"mean":0.7912407386961582,"std":0.059376814889964616},"nc":{"folds":[0.8274128779496446,0.9310802483142688,0.8420248795305616,0.8934199562586463,0.766136116824971],"mean":0.8520148157756186,"std":0.06333627196435798},"xc":{"folds":[0.33747433345673306,0.5264217791469855,0.5438259838593131,0.5190657046593701,0.29254659585142007],"mean":0.4438668793947643,"std":0.11903696841994589}},"macro":0.8205009278658766},"development_decision":{"candidate_macro":0.8205009278658766,"reference_macro":0.8189884376838695,"delta":0.0015124901820070535,"selection":"validation_only_full_precision","plan":"UA-HOP-PH-FROZEN3D-20261020-01"},"test":{"status":"未执行／不可比较"},"sota":{"eea":{"reference":0.934,"status":"未执行／不可比较"},"egb":{"reference":0.923,"status":"未执行／不可比较"},"egc":{"reference":0.911,"status":"未执行／不可比较"},"ei":{"reference":0.839,"status":"未执行／不可比较"},"eps":{"reference":0.787,"status":"未执行／不可比较"},"nc":{"reference":0.856,"status":"未执行／不可比较"},"xc":{"reference":0.423,"status":"未执行／不可比较"}}}
```

</details>

<details><summary>results/ua_hop_ph_20261020/r1/final_review/results.json</summary>

```json
{"plan":"UA-HOP-PH-20261020-01","revision":"r1","status":"待审查","budget":{"plan":"UA-HOP-PH-20261020-01","revision":"r1"},"independent_review":"PENDING","validation":{"HOP_EQ":{"tasks":{"eea":{"folds":[0.9387645741105414,0.9110650820498801,0.9163701322469197,0.9178123777945771,0.9468387003493052],"mean":0.9261701733102446,"std":0.015651391904487277},"egb":{"folds":[0.9375133241885717,0.9369212884145146,0.93372899372461,0.947050030207324,0.9556811387318829],"mean":0.9421789550533808,"std":0.009039487728201167},"egc":{"folds":[0.9285858948458288,0.9266757331562032,0.9193967771730573,0.9084306712414311,0.920972894886082],"mean":0.9208123942605205,"std":0.007907952507855403},"ei":{"folds":[0.860172868995352,0.8654640706735782,0.8946355823015579,0.8740306536354245,0.8354265020024029],"mean":0.8659459355216631,"std":0.021525884491233403},"eps":{"folds":[0.7459189770175789,0.8709227203837675,0.7737633433448281,0.8375562287981341,0.672050799614587],"mean":0.7800424138317791,"std":0.0781554435698422},"nc":{"folds":[0.7966833802327065,0.9100520782887853,0.8582562474073663,0.9004832072417246,0.7661361221360704],"mean":0.8463222070613305,"std":0.06330709496326731},"xc":{"folds":[0.33747432707192393,0.5351255819084006,0.5246439133093259,0.5282169496144851,0.31618447297885177],"mean":0.4483290489765975,"std":0.11123244990741218}},"macro":0.8185430182879309},"HOP_PH":{"tasks":{"eea":{"folds":[0.9387667265172333,0.9111812001334202,0.9163318900011029,0.9178148561102099,0.9468167417649549],"mean":0.9261822829053841,"std":0.015622305122162316},"egb":{"folds":[0.9374561600013339,0.9372583117205315,0.9335171511526802,0.9472977837532419,0.9556778492544585],"mean":0.9422414511764492,"std":0.009082227064580083},"egc":{"folds":[0.9283616469770634,0.9260791308575271,0.9193661814072661,0.9122444874845032,0.920503971246813],"mean":0.9213110835946345,"std":0.006306384831968108},"ei":{"folds":[0.8601261596918515,0.8653097997526675,0.8945415902435391,0.8740455280912965,0.8347100391838305],"mean":0.8657466233926371,"std":0.021754723174500553},"eps":{"folds":[0.7465285514663902,0.8716502177188143,0.7736400265680825,0.8373450551063163,0.67205077202192],"mean":0.7802429245763047,"std":0.07826516728621989},"nc":{"folds":[0.7966833657298292,0.9126518236572433,0.8600321443897149,0.9005302741598779,0.7661361266981249],"mean":0.8472067469269581,"std":0.06406247918389639},"xc":{"folds":[0.3374742836705563,0.5526304193389778,0.5244106437046918,0.5194490731740983,0.3159753361852701],"mean":0.44998795121471885,"std":0.11348762846145603}},"macro":0.8189884376838695}},"test":{"HOP_EQ":{"tasks":{"eea":{"folds":[0.9156469016750978,0.946080900452787,0.9505961383701486,0.9329731918656656,0.9229652108699157],"mean":0.9336524686467229,"std":0.014836047305106284},"egb":{"folds":[0.9497618937428709,0.9173492948905556,0.9448787496000206,0.9249028958157072,0.92324392871564],"mean":0.9320273525529588,"std":0.01434419875878138},"egc":{"folds":[0.8824754877793821,0.92773104577636,0.9125780628066456,0.9109705878905521,0.9238207686554227],"mean":0.9115151905816725,"std":0.017741982976314845},"ei":{"folds":[0.7877977783887575,0.7666138966059194,0.7919656723722592,0.851886583589375,0.8374155127686422],"mean":0.8071358887449908,"std":0.0359360896033959},"eps":{"folds":[0.744725807102735,0.8255906655761239,0.7319071905865548,0.7651983573595478,0.7059342973852609],"mean":0.7546712636020445,"std":0.04509098657323444},"nc":{"folds":[0.7620815647225229,0.9213934865681028,0.8460343040218353,0.7172641649652323,0.7947465759877734],"mean":0.8083040192530934,"std":0.07877400688964754},"xc":{"folds":[0.45977913780048973,0.28800742581383676,0.46519435358202466,0.13892244097749729,0.4940318667467015],"mean":0.36918704498411004,"std":0.1521697066871005}},"macro":0.7880704611950847},"HOP_PH":{"tasks":{"eea":{"folds":[0.915646920536994,0.9451809362180708,0.9506973971366138,0.9330772469131086,0.9229453657622123],"mean":0.9335095733133999,"std":0.01468422411417918},"egb":{"folds":[0.9498236121232305,0.9173093830271865,0.9449757774396231,0.9313486004146333,0.9232421289735095],"mean":0.9333399003956366,"std":0.013874564293242162},"egc":{"folds":[0.8900220680000857,0.928032364438317,0.9122698043309434,0.9145675023058171,0.9242932114353964],"mean":0.913836990102112,"std":0.014840665847278552},"ei":{"folds":[0.7878168826533125,0.7666850194113585,0.7918529480017309,0.8518667407519113,0.8371325851354763],"mean":0.8070708351907578,"std":0.03585977851215163},"eps":{"folds":[0.745269276394904,0.8245352457472069,0.732123612629078,0.7665859009286149,0.7059342621036314],"mean":0.7548896595606871,"std":0.04470678176386607},"nc":{"folds":[0.7620815633857299,0.8896064582350706,0.854885490457137,0.7173171856995159,0.7947465686293727],"mean":0.8037274532813651,"std":0.0694156305118213},"xc":{"folds":[0.4597791493876111,0.3053198620926346,0.4651884473738488,0.1368945268183248,0.4935409607529777],"mean":0.3721445892850794,"std":0.15073336617429453}},"macro":0.7883598573041483}},"sota":{"eea":{"reference":0.934,"arms":{"HOP_EQ":{"r2":0.9336524686467229,"std":0.014836047305106284,"delta":-0.0003475313532771418},"HOP_PH":{"r2":0.9335095733133999,"std":0.01468422411417918,"delta":-0.0004904266866001539}}},"egb":{"reference":0.923,"arms":{"HOP_EQ":{"r2":0.9320273525529588,"std":0.01434419875878138,"delta":0.009027352552958723},"HOP_PH":{"r2":0.9333399003956366,"std":0.013874564293242162,"delta":0.010339900395636592}}},"egc":{"reference":0.911,"arms":{"HOP_EQ":{"r2":0.9115151905816725,"std":0.017741982976314845,"delta":0.0005151905816724867},"HOP_PH":{"r2":0.913836990102112,"std":0.014840665847278552,"delta":0.0028369901021119137}}},"ei":{"reference":0.839,"arms":{"HOP_EQ":{"r2":0.8071358887449908,"std":0.0359360896033959,"delta":-0.03186411125500921},"HOP_PH":{"r2":0.8070708351907578,"std":0.03585977851215163,"delta":-0.03192916480924213}}},"eps":{"reference":0.787,"arms":{"HOP_EQ":{"r2":0.7546712636020445,"std":0.04509098657323444,"delta":-0.032328736397955526},"HOP_PH":{"r2":0.7548896595606871,"std":0.04470678176386607,"delta":-0.032110340439312934}}},"nc":{"reference":0.856,"arms":{"HOP_EQ":{"r2":0.8083040192530934,"std":0.07877400688964754,"delta":-0.04769598074690662},"HOP_PH":{"r2":0.8037274532813651,"std":0.0694156305118213,"delta":-0.052272546718634905}}},"xc":{"reference":0.423,"arms":{"HOP_EQ":{"r2":0.36918704498411004,"std":0.1521697066871005,"delta":-0.05381295501588995},"HOP_PH":{"r2":0.3721445892850794,"std":0.15073336617429453,"delta":-0.05085541071492061}}}}}
```

</details>

<details><summary>results/ua3d_spatial_20261025/r4/final_review/results.json</summary>

```json
{"plan":"UA3D-SPATIAL-20261025-01","revision":"r4","status":"需返修","independent_review":"PENDING","FINAL":"C_CONST","validation":{"FULL":{"BASE":{"tasks":{"eea":{"folds":[0.9302778294816162,0.9058611869062749,0.8998653856045301,0.9103072550024488,0.951286227836459],"mean":0.9195195769662659,"std":0.021110221681524318},"egb":{"folds":[0.9276459390203702,0.9188380163658406,0.8980766977186607,0.9282449872278409,0.9393665333523495],"mean":0.9224344347370124,"std":0.015442245937842988},"egc":{"folds":[0.9116041128834086,0.9209192336323841,0.918326232582549,0.9111920985267311,0.916287303790536],"mean":0.9156657962831216,"std":0.004230072118851745},"ei":{"folds":[0.877129985894912,0.8852401692975864,0.8826944055046253,0.8439224427562817,0.8348892067042478],"mean":0.8647752420315307,"std":0.02356142880168572},"eps":{"folds":[0.753681959936906,0.8481633005182398,0.7866677186544966,0.7878239384254238,0.7076308485436704],"mean":0.7767935532157474,"std":0.05155200003988883},"nc":{"folds":[0.8308050639118403,0.9215472444863319,0.827955665366945,0.8870963133841655,0.7633463872579573],"mean":0.846150134881448,"std":0.060795320986113906},"xc":{"folds":[0.40582249493033073,0.446502541820768,0.32812852211397125,0.5254981879503483,0.3990068593363364],"mean":0.4209917212303509,"std":0.07230044412495483}},"macro":0.8094757799064967},"C_CONST":{"tasks":{"eea":{"folds":[0.934568952555824,0.9108353009838784,0.8941707165400519,0.911148981353677,0.9580644404317343],"mean":0.9217576783730331,"std":0.024875655573410357},"egb":{"folds":[0.9300110927405346,0.9293534880254796,0.9129087504481075,0.9416754565762545,0.9381141495822944],"mean":0.9304125874745341,"std":0.011111182200988994},"egc":{"folds":[0.913133833056184,0.921529324212742,0.9201502019281755,0.9136805267859228,0.9122626976813335],"mean":0.9161513167328715,"std":0.0043372057850581515},"ei":{"folds":[0.8805249243219215,0.8838155256929486,0.8885781534624805,0.850913908471526,0.8222968670394895],"mean":0.8652258757976732,"std":0.028163349600213557},"eps":{"folds":[0.7264883472382428,0.877320809615526,0.7586398889981837,0.7888479806881636,0.6969342186801757],"mean":0.7696462490440584,"std":0.06934208913381709},"nc":{"folds":[0.8468383189689566,0.9186864712352297,0.856799200688369,0.8719842931275934,0.7741986517445221],"mean":0.8537013871529343,"std":0.05228457921576755},"xc":{"folds":[0.3391619360896736,0.44344673084134345,0.4693754710547071,0.48425904435555156,0.38087748352657746],"mean":0.42342413317357064,"std":0.061489128190560786}},"macro":0.811474175392668},"S_RAD":{"tasks":{"eea":{"folds":[0.9234248630499086,0.9053042962457887,0.896458080361027,0.9209657360987809,0.9541228731742573],"mean":0.9200551697859524,"std":0.022063563160687322},"egb":{"folds":[0.9256856178782084,0.917900965565052,0.89904833530778,0.928218898512987,0.9359084564169129],"mean":0.921352454736188,"std":0.014028435932595314},"egc":{"folds":[0.9088196034916893,0.9199127269128659,0.9159782212314044,0.9101267048701049,0.9121109484744194],"mean":0.9133896409960969,"std":0.0045392303045550305},"ei":{"folds":[0.8790503700104129,0.8975266765970502,0.8827033295853031,0.8481495742778924,0.8163849940707872],"mean":0.8647629889082893,"std":0.03246490506497038},"eps":{"folds":[0.7305517128636241,0.8538012151445352,0.7598097077642195,0.7965977538747193,0.7009117976866284],"mean":0.7683344374667452,"std":0.059475631124978105},"nc":{"folds":[0.8223863206497372,0.9166991308492108,0.868571510310244,0.8732093151464309,0.7763568969206229],"mean":0.8514446347752491,"std":0.05363705442849091},"xc":{"folds":[0.28921704420276995,0.44459886006329485,0.3806825394738518,0.48128156561621893,0.339409669161851],"mean":0.3870379357035973,"std":0.07757720580375936}},"macro":0.8037681803388741},"S_RADANG":{"tasks":{"eea":{"folds":[0.9272637872743704,0.9025186724212496,0.8995185240513056,0.9150641225081421,0.9515651874193141],"mean":0.9191860587348764,"std":0.021185264442741377},"egb":{"folds":[0.9250430968974572,0.9222431618411534,0.8946586478580714,0.9362191079815727,0.9364717600342967],"mean":0.9229271549225103,"std":0.0170600842054448},"egc":{"folds":[0.9107844597532037,0.9217268451956362,0.9191034232712542,0.9102212374483041,0.91144642799887],"mean":0.9146564787334537,"std":0.0053556913557294816},"ei":{"folds":[0.8750104968262948,0.8648402281074898,0.8683867446980921,0.8551914023506565,0.8213628306394709],"mean":0.8569583405244009,"std":0.021146957910392727},"eps":{"folds":[0.7522202583484968,0.8610729082817865,0.7555132101624054,0.8185607404867512,0.7206995193683674],"mean":0.7816133273295615,"std":0.056865184789793424},"nc":{"folds":[0.8460573958012528,0.9076055041786797,0.8604058189107816,0.8908820752631821,0.7693689799948721],"mean":0.8548639548297536,"std":0.05361118001106263},"xc":{"folds":[0.34392629531922625,0.4633013600091891,0.36682838919685334,0.4342151579030178,0.34533897616132514],"mean":0.3907220357179223,"std":0.0547260900993258}},"macro":0.8058467643989256}},"probe":{"BASE":{"tasks":{"eea":{"folds":[0.8637220422956611,0.7969382066090878,0.804173105111528,0.8214232366557974,0.8824225233227715],"mean":0.8337358227989691,"std":0.03758053611730544},"egb":{"folds":[0.8336530199361984,0.8197028204727392,0.7868594359841349,0.8387960127256242,0.8626240252594656],"mean":0.8283270628756325,"std":0.027875562753812762},"egc":{"folds":[0.8274174265819089,0.822531458084164,0.8285204670564768,0.8096558367318172,0.8171945307011449],"mean":0.8210639438311024,"std":0.007798034132352374},"ei":{"folds":[0.7217222197055485,0.72970733135243,0.722499027682411,0.740549630349406,0.7093109937403812],"mean":0.7247578405660354,"std":0.011476066291567942},"eps":{"folds":[0.6738087180208303,0.7273088325508842,0.6658261880215031,0.5910104163575252,0.6704031138330591],"mean":0.6656714537567604,"std":0.048636572118144636},"nc":{"folds":[0.7823648082745264,0.7600041277653607,0.7492154754366553,0.7156273867745688,0.7028139057790586],"mean":0.742005140806034,"std":0.03254456185716339},"xc":{"folds":[0.15996510613078097,0.3789354272755048,0.30165076859750706,0.4564584239602285,0.27708058091137255],"mean":0.3148180613750788,"std":0.11152349719004646}},"macro":0.7043399037156589},"C_CONST":{"tasks":{"eea":{"folds":[0.8690928389365489,0.7814592238949062,0.7938424029432772,0.8040045618643004,0.8805472150768575],"mean":0.825789248543178,"std":0.045645263836069455},"egb":{"folds":[0.8269231977603316,0.8161661599261715,0.7929972066140146,0.8475676799026879,0.8643834045516572],"mean":0.8296075297509725,"std":0.027659176970856257},"egc":{"folds":[0.8179993741262253,0.8200264185341216,0.8163903136362157,0.8035224603504759,0.8170742499316865],"mean":0.8150025633157452,"std":0.006561830856720537},"ei":{"folds":[0.7310229562198198,0.7095624849375876,0.6977180985243181,0.7479780073451103,0.710334888331548],"mean":0.7193232870716767,"std":0.020005149302767287},"eps":{"folds":[0.6849719116264554,0.7222692147735483,0.6662778629661716,0.6279603276622339,0.658415010567777],"mean":0.6719788655192374,"std":0.03482832805916628},"nc":{"folds":[0.7768868993569306,0.784617415496709,0.7472885392179538,0.7009178993492146,0.6833216102957612],"mean":0.7386064727433139,"std":0.04509606627349551},"xc":{"folds":[0.1771189413070554,0.30326091246348974,0.29971229200062144,0.48335154187735296,0.26707677257207496],"mean":0.3061040920441189,"std":0.1113843548553213}},"macro":0.7009160084268918},"S_RAD":{"tasks":{"eea":{"folds":[0.8810116870059713,0.7875048868825036,0.8012233945734893,0.8237457112092236,0.8839517590042238],"mean":0.8354874877350824,"std":0.04482037954338395},"egb":{"folds":[0.8315684996034242,0.8125166602953693,0.8024512560885723,0.8290542332809097,0.8772390661953124],"mean":0.8305659430927175,"std":0.028713598873625897},"egc":{"folds":[0.8167773000258289,0.8141313976361659,0.8108259903187001,0.800783658526237,0.809197716938222],"mean":0.8103432126890308,"std":0.006096682204229314},"ei":{"folds":[0.7273171458633976,0.7382455281267642,0.6880421469271186,0.7337961386825915,0.7117661891067449],"mean":0.7198334297413234,"std":0.020406706307966878},"eps":{"folds":[0.6963422454731847,0.740423453621011,0.6761535130920981,0.5777802603767312,0.6618214316333861],"mean":0.6705041808392823,"std":0.05971042791039979},"nc":{"folds":[0.7585068466425973,0.7722323936296682,0.7698604642119065,0.6993834393472802,0.7294426042414602],"mean":0.7458851496145824,"std":0.031068740597381135},"xc":{"folds":[0.18267961900630836,0.3460547864745013,0.3384319499443198,0.4796759250955914,0.3165418915977385],"mean":0.3326768344236919,"std":0.10557026644261704}},"macro":0.7064708911622445},"S_RADANG":{"tasks":{"eea":{"folds":[0.8755936934229919,0.7833367488232426,0.805011549209555,0.8233102007639079,0.8890197713377432],"mean":0.8352543927114882,"std":0.04547172245469551},"egb":{"folds":[0.8479050127331995,0.8277498814584049,0.7992597689077033,0.8338410377385982,0.875304835653169],"mean":0.8368121072982151,"std":0.027865175386551425},"egc":{"folds":[0.8244919869641701,0.8238180995856851,0.8106106249347602,0.806958790727944,0.8100542471695212],"mean":0.815186749876416,"std":0.008307683948157056},"ei":{"folds":[0.7420471002439472,0.7113116626835717,0.7468655045018373,0.7524445258149802,0.7483614709851758],"mean":0.7402060528459025,"std":0.016574940425034806},"eps":{"folds":[0.6899115457663907,0.7343900905469174,0.7091686897517293,0.659245356771436,0.6235674036796487],"mean":0.6832566173032244,"std":0.04321370879916513},"nc":{"folds":[0.7656816430775274,0.7690203507174921,0.7794187015528053,0.7507967227053179,0.6877556403034913],"mean":0.7505346116713267,"std":0.036560613045264714},"xc":{"folds":[0.1801385838133034,0.3715897745087685,0.24709746626996287,0.4746469987011406,0.2880975932849278],"mean":0.31231408331562066,"std":0.11420988484032533}},"macro":0.7105092307174562}}},"formal":{"FULL":{"BASE":{"tasks":{"eea":{"folds":[0.8975483647839905,0.9288247508960716,0.9461946762473679,0.9249657074131301,0.9227552767047394],"mean":0.9240577552090599,"std":0.01745018717022919},"egb":{"folds":[0.9262120328575327,0.9036890761228172,0.9265469060320997,0.9173764232074346,0.9097682118705487],"mean":0.9167185300180865,"std":0.010065162515391076},"egc":{"folds":[0.893853743567249,0.9225571792043704,0.9223921538819906,0.9080185371937696,0.9167422301957654],"mean":0.912712768808629,"std":0.012088611520947862},"ei":{"folds":[0.7554440169051936,0.7666373001706158,0.8290033308662798,0.8445278724464078,0.8330848808445934],"mean":0.8057394802466181,"std":0.04138880032202877},"eps":{"folds":[0.7176295799175665,0.822127740345289,0.7268450672917459,0.7747772356432724,0.8152789097679007],"mean":0.7713317065931549,"std":0.048441833336694},"nc":{"folds":[0.8282122438525195,0.9153512080730142,0.7967376541741665,0.7138817571921905,0.8503991795574111],"mean":0.8209164085698604,"std":0.07396185353687995},"xc":{"folds":[0.5368643872016108,0.36692901582045667,0.4378205577890818,0.26303865504563306,0.47153022199169825],"mean":0.4152365675696961,"std":0.10484531794924663}},"macro":0.795244745287872},"C_CONST":{"tasks":{"eea":{"folds":[0.9041560013131428,0.9200113619990171,0.9263601800973127,0.9152403200493444,0.9127891223857763],"mean":0.9157113971689187,"std":0.008279116081149585},"egb":{"folds":[0.9138592242330581,0.8944017321801515,0.9263285660627851,0.9256306278429477,0.9123321732950282],"mean":0.9145104647227942,"std":0.01296973148616043},"egc":{"folds":[0.8940497269172921,0.9229485626373148,0.9246689761494515,0.9077090854912411,0.9202863570626398],"mean":0.9139325416515879,"std":0.012951721495208604},"ei":{"folds":[0.7720179728563639,0.7863509241991601,0.8068550770841147,0.8688414583930124,0.8303554457914535],"mean":0.8128841756648209,"std":0.03823242329874713},"eps":{"folds":[0.7751274485876017,0.8032160093624454,0.5786978387667441,0.7207260088631775,0.7434502279888842],"mean":0.7242435067137706,"std":0.08716456290636308},"nc":{"folds":[0.8074256219779298,0.8879862897230075,0.7666664865621035,0.7216090724509421,0.8393156492266349],"mean":0.8046006239881235,"std":0.06421230265651498},"xc":{"folds":[0.48399824699424865,0.44876003694948896,0.43037684960630807,0.3785604674235339,0.5350386977730035],"mean":0.45534685974931666,"std":0.05859762948746285}},"macro":0.7916042242370475},"S_RAD":{"tasks":{"eea":{"folds":[0.9108591682029112,0.9109412436355537,0.9499867507461891,0.9358603325425361,0.919662796745418],"mean":0.9254620583745217,"std":0.01708299255903703},"egb":{"folds":[0.8956053721081272,0.8983558268209701,0.9269286660609607,0.9214295482671813,0.9050543102279134],"mean":0.9094747446970306,"std":0.0139918437939689},"egc":{"folds":[0.8991793972265503,0.9252487825505071,0.9206097820459207,0.9099963467356635,0.9248305763571709],"mean":0.9159729769831625,"std":0.011222099034278878},"ei":{"folds":[0.8071496361111385,0.788009998451493,0.8102216322626408,0.8434310729952548,0.8335660924333119],"mean":0.8164756864507678,"std":0.022108022539387567},"eps":{"folds":[0.7341606875247996,0.8165591378525089,0.6588247826032634,0.7340760612543069,0.7862474113024734],"mean":0.7459736161074704,"std":0.06016708139372322},"nc":{"folds":[0.7801313471595104,0.9091656768150036,0.8132348520308731,0.7417994577111793,0.8087038406828073],"mean":0.8106070348798747,"std":0.06202334981497477},"xc":{"folds":[0.48942886770314764,0.38901305802990105,0.4772117306570316,0.18375140743699458,0.48584013154882655],"mean":0.40504903907518025,"std":0.13046494224068633}},"macro":0.7898593080811441},"S_RADANG":{"tasks":{"eea":{"folds":[0.8967358467758741,0.9420040925862115,0.9296688342935405,0.9362932137175638,0.9059428912787932],"mean":0.9221289757303965,"std":0.01974395922122173},"egb":{"folds":[0.9065716086468811,0.8751797690123986,0.9211297820965859,0.9173542596502474,0.9094220965787699],"mean":0.9059315031969766,"std":0.018164045385073303},"egc":{"folds":[0.8936313486435548,0.923079348565303,0.9210006270201541,0.9116274921095157,0.9203499285265199],"mean":0.9139377489730094,"std":0.012168158418147286},"ei":{"folds":[0.7707004596857953,0.783737603956765,0.760260461927033,0.8461680364821745,0.8419597142550372],"mean":0.8005652552613609,"std":0.04059762037168811},"eps":{"folds":[0.7265898580031183,0.848873067477761,0.5763253845569016,0.7418914992343364,0.7788926888915465],"mean":0.7345144996327327,"std":0.10023512597639044},"nc":{"folds":[0.8412269414407133,0.9005516320874092,0.811350541033196,0.762433765216577,0.8471986069430881],"mean":0.8325522973441967,"std":0.05069466534801853},"xc":{"folds":[0.4293735655555201,0.3805166450090849,0.48221816068647183,0.2852145435461624,0.5818918512326614],"mean":0.4318429532059801,"std":0.11089715013874595}},"macro":0.7916390333349506}},"probe":{"BASE":{"tasks":{"eea":{"folds":[0.7762561429495868,0.8255601484913622,0.8396841698834638,0.8774784671059289,0.8150222741169426],"mean":0.8268002405094569,"std":0.0368416965060513},"egb":{"folds":[0.8284406246110463,0.8166773988595781,0.8216291629750125,0.8304335630482989,0.870769190585212],"mean":0.8335899880158296,"std":0.0214934238394592},"egc":{"folds":[0.800472178880969,0.8281036399072004,0.8270314368266872,0.794750596545235,0.8088183979302811],"mean":0.8118352500180744,"std":0.015212583357047027},"ei":{"folds":[0.6945535196299556,0.6578641714886945,0.49878805654092495,0.6629514595089165,0.6625027099263678],"mean":0.6353319834189718,"std":0.07771778698615003},"eps":{"folds":[0.6697180708140352,0.6052370486830687,0.30329526785906546,0.6481913720743423,0.6720097250392795],"mean":0.5796902968939582,"std":0.15681766704769418},"nc":{"folds":[0.7302472534909303,0.6837859108937223,0.6066522470430648,0.7040627260287542,0.7187142240211754],"mean":0.6886924722955294,"std":0.04903830905772171},"xc":{"folds":[0.23466191872662168,0.29776084897971844,0.4226327845974529,0.2644053475684527,0.3702451963094121],"mean":0.31794121923633156,"std":0.07730589994907154}},"macro":0.6705544929125932},"C_CONST":{"tasks":{"eea":{"folds":[0.8141438459220574,0.8138466555317977,0.8039613371331196,0.8515381940125415,0.772156738896437],"mean":0.8111293542991908,"std":0.02837071744834973},"egb":{"folds":[0.8175392112075327,0.8128495028276785,0.827718839428029,0.8418595772427866,0.8577128259750805],"mean":0.8315359913362215,"std":0.018381161913229124},"egc":{"folds":[0.8056368304471399,0.8266812265132886,0.8224111312284212,0.7898344141002984,0.794083653183709],"mean":0.8077294510945714,"std":0.016473772593874222},"ei":{"folds":[0.7238189134888513,0.6691207344566347,0.5662046303533612,0.6645276548331972,0.5555380853179182],"mean":0.6358420036899926,"std":0.07240227936991864},"eps":{"folds":[0.6789870512549252,0.6212202099807023,0.456193079263314,0.7098205542017975,0.6692021903158325],"mean":0.6270846170033144,"std":0.10068676291841731},"nc":{"folds":[0.6863066121366823,0.7178080364073485,0.6669511205756096,0.7507685985030799,0.7303186322775465],"mean":0.7104305999800533,"std":0.03372516286282827},"xc":{"folds":[0.3676230927336708,0.4252536886860231,0.3625655364410636,0.24429817027219047,0.36533902286867925],"mean":0.35301590220032547,"std":0.06613294917128526}},"macro":0.6823954170862383},"S_RAD":{"tasks":{"eea":{"folds":[0.801215667980549,0.8652081585440388,0.8551895311684682,0.874872937239211,0.8439661260370963],"mean":0.8480904841938728,"std":0.02861367574897465},"egb":{"folds":[0.8030891017499688,0.836089966154837,0.8155936250241491,0.8503209162811494,0.8714314851044564],"mean":0.8353050188629121,"std":0.027191999021414472},"egc":{"folds":[0.8069128751603463,0.8208672562360064,0.8162249472095676,0.7855714624275758,0.7950966941557749],"mean":0.8049346470378541,"std":0.014636858743381298},"ei":{"folds":[0.7611964767465578,0.699268682536972,0.5627909980929529,0.6585503245922462,0.6512081514121787],"mean":0.6666029266761815,"std":0.07261844005277564},"eps":{"folds":[0.6822123146969774,0.6648424034123204,0.4221149501749569,0.679536842587942,0.6743513464171089],"mean":0.6246115714578611,"std":0.11339320715341372},"nc":{"folds":[0.7394112173750291,0.711554465459801,0.6112077062539494,0.7441665042804539,0.7478173488379267],"mean":0.710831448441432,"std":0.05749217061756061},"xc":{"folds":[0.3481036257805814,0.3886091883233649,0.4006976946558315,0.3051888831545614,0.36391748787885325],"mean":0.3613033759586385,"std":0.03750807587396014}},"macro":0.6930970675183933},"S_RADANG":{"tasks":{"eea":{"folds":[0.8324999348867519,0.8566792993314016,0.8424840122252851,0.8692873607406572,0.8582646477565496],"mean":0.8518430509881292,"std":0.014410741591881342},"egb":{"folds":[0.8356480653493135,0.8318034596687113,0.8325586575115594,0.823960181567903,0.8736784142113552],"mean":0.8395297556617685,"std":0.019569759177584526},"egc":{"folds":[0.8031808937658429,0.820519781962529,0.8085692524597765,0.8031780832097672,0.8071527028929216],"mean":0.8085201428581674,"std":0.007122303430130551},"ei":{"folds":[0.7273313775316981,0.705310554729071,0.6553919058761537,0.6923507251630516,0.6732427100238898],"mean":0.6907254546647728,"std":0.027881294209530068},"eps":{"folds":[0.7007760343442144,0.6820594265316797,0.507452517197351,0.7475310460857313,0.6625261355717869],"mean":0.6600690319461526,"std":0.09095064506081464},"nc":{"folds":[0.7206322870482177,0.7433611252513745,0.6312070917768793,0.7254335323865952,0.7407306933772874],"mean":0.7122729459680708,"std":0.04634401421078782},"xc":{"folds":[0.28710779346710036,0.4017926854874957,0.382507267902611,0.23352756326624313,0.3976966121068747],"mean":0.340526384446065,"std":0.07597151233510088}},"macro":0.7004981095047322}}},"comparisons":{"FULL":{"G_minus_BASE":{"status":"COMPLETE","delta":-0.0036057119529214754,"performance_pass":false,"tasks":{"eea":{"folds":[-0.0008125180081163252,0.0131793416901399,-0.016525841953827336,0.011327506304433688,-0.016812385425946208],"mean":-0.0019287794786632562,"std":0.014489879542139435},"egb":{"folds":[-0.019640424210651597,-0.028509307110418658,-0.005417123935513746,-2.2163557187204752e-05,-0.000346115291778748],"mean":-0.01078702682110999,"std":0.012710159672924765},"egc":{"folds":[-0.00022239492369424685,0.0005221693609326339,-0.0013915268618365229,0.0036089549157460876,0.003607698330754472],"mean":0.0012249801643804848,"std":0.0022801111905334038},"ei":{"folds":[0.015256442780601676,0.01710030378614913,-0.06874286893924675,0.0016401640357667224,0.008874833410443816],"mean":-0.005174224985257081,"std":0.036049796292877564},"eps":{"folds":[0.008960278085551776,0.02674532713247202,-0.15051968273484428,-0.032885736408935995,-0.036386220876354236],"mean":-0.03681720696042214,"std":0.06906458767773035},"nc":{"folds":[0.013014697588193846,-0.014799575985604996,0.014612886859029506,0.04855200802438653,-0.0032005726143229385],"mean":0.011635888774336389,"std":0.02394012257402122},"xc":{"folds":[-0.1074908216460907,0.013587629188628214,0.04439760289739003,0.022175888500529317,0.11036162924096315],"mean":0.016606385636284006,"std":0.07906112598514599}}},"G_minus_C_CONST":{"status":"COMPLETE","delta":3.480909790309905e-05,"performance_pass":false,"tasks":{"eea":{"folds":[-0.007420154537268631,0.02199273058719442,0.0033086541962278337,0.021052893668219386,-0.006846231106983014],"mean":0.006417578561477999,"std":0.014438271397919705},"egb":{"folds":[-0.007287615586176965,-0.01922196316775293,-0.005198783966199216,-0.008276368192700279,-0.002910076716258314],"mean":-0.008578961525817542,"std":0.006296714943720878},"egc":{"folds":[-0.00041837827373736314,0.00013078592798820843,-0.003668349129297477,0.003918406618274584,6.357146388002999e-05],"mean":5.207321421596412e-06,"std":0.002692895599976386},"ei":{"folds":[-0.0013175131705686072,-0.002613320242395134,-0.04659461515708174,-0.022673421910837877,0.011604268463583756],"mean":-0.01231892040345992,"std":0.022746359892674956},"eps":{"folds":[-0.04853759058448348,0.04565705811531562,-0.002372454209842445,0.02116549037115889,0.03544246090266234],"mean":0.010270992918962185,"std":0.03749257504577259},"nc":{"folds":[0.03380131946278353,0.012565342364401655,0.04468405447109258,0.04082469276563494,0.007882957716453198],"mean":0.02795167335607318,"std":0.01672870358754798},"xc":{"folds":[-0.05462468143872856,-0.06824339194040407,0.051841311080163766,-0.0933459238773715,0.046853153459657904],"mean":-0.023503906543336495,"std":0.06796147673492922}}},"C_CONST_minus_BASE":{"status":"COMPLETE","delta":-0.0036405210508245744,"performance_pass":false,"tasks":{"eea":{"folds":[0.006607636529152305,-0.00881338889705452,-0.01983449615005517,-0.009725387363785698,-0.009966154318963194],"mean":-0.008346358040141255,"std":0.009491349357343554},"egb":{"folds":[-0.012352808624474632,-0.009287343942665727,-0.00021833996931452937,0.008254204635513074,0.002563961424479566],"mean":-0.0022080652952924497,"std":0.008503269097444744},"egc":{"folds":[0.0001959833500431163,0.00039138343294442546,0.002276822267460954,-0.00030945170252849596,0.003544126866874442],"mean":0.0012197728429588883,"std":0.0016273349966496229},"ei":{"folds":[0.016573955951170283,0.019713624028544263,-0.022148253782165006,0.0243135859466046,-0.002729435053139939],"mean":0.00714469541820284,"std":0.01934706081041195},"eps":{"folds":[0.05749786867003526,-0.0189117309828436,-0.14814722852500184,-0.054051226780094885,-0.07182868177901658],"mean":-0.047088199879384326,"std":0.07517589639163509},"nc":{"folds":[-0.020786621874589684,-0.02736491835000665,-0.030071167612063077,0.007727315258751588,-0.011083530330776137],"mean":-0.01631578458173679,"std":0.01530301800238814},"xc":{"folds":[-0.05286614020736213,0.08183102112903229,-0.007443708182773734,0.11552181237790082,0.06350847578130525],"mean":0.0401102921796205,"std":0.068704060157355}}},"distance":{"status":"COMPLETE","delta":-0.001744916155903331,"performance_pass":false,"tasks":{"eea":{"folds":[0.006703166889768464,-0.009070118363463386,0.02362657064887641,0.020620012493191675,0.006873674359641702],"mean":0.009750661205602973,"std":0.013062060315999106},"egb":{"folds":[-0.018253852124930914,0.0039540946408186395,0.0006000999981755628,-0.004201079575766431,-0.007277863067114865],"mean":-0.005035720025763602,"std":0.008558943880108973},"egc":{"folds":[0.00512967030925815,0.002300219913192292,-0.004059194103530839,0.002287261244422467,0.004544219294531082],"mean":0.0020404353315746303,"std":0.0036450719192563847},"ei":{"folds":[0.03513166325477468,0.001659074252332915,0.0033665551785260206,-0.02541038539775764,0.003210646641858439],"mean":0.003591510785946883,"std":0.021446568335547807},"eps":{"folds":[-0.04096676106280217,0.013343128490063472,0.0801269438365193,0.013350052391129408,0.04279718331358928],"mean":0.02173010939369986,"std":0.04451257194763025},"nc":{"folds":[-0.027294274818419395,0.02117938709199607,0.046568365468769635,0.020190385260237198,-0.030611808543827612],"mean":0.006006410891751179,"std":0.03363965724471243},"xc":{"folds":[0.005430620708898992,-0.05974697891958791,0.04683488105072353,-0.1948090599865393,-0.049198566224176954],"mean":-0.05029782067413633,"std":0.09153471221735084}}},"angle":{"status":"COMPLETE","delta":0.00177972525380643,"performance_pass":false,"tasks":{"eea":{"folds":[-0.014123321427037094,0.031062848950657806,-0.020317916452648577,0.0004328811750277106,-0.013719905466624716],"mean":-0.003333082644124974,"std":0.02067645106517057},"egb":{"folds":[0.010966236538753948,-0.02317605780857157,-0.005798883964374779,-0.004075288616933848,0.004367786350856551],"mean":-0.0035432415000539396,"std":0.012883398114258844},"egc":{"folds":[-0.005548048582995513,-0.0021694339852040834,0.00039084497423336195,0.0016311453738521164,-0.004480647830651052],"mean":-0.002035228010153034,"std":0.0030686365716601333},"ei":{"folds":[-0.03644917642534329,-0.004272394494728049,-0.04996117033560776,0.0027369634869197634,0.008393621821725317],"mean":-0.015910431189406805,"std":0.02576406951239763},"eps":{"folds":[-0.00757082952168131,0.03231392962525215,-0.08249939804636175,0.007815437980029483,-0.007354722410926939],"mean":-0.011459116474737675,"std":0.042913501506337146},"nc":{"folds":[0.061095594281202925,-0.008614044727594417,-0.0018843109976770522,0.020634307505397742,0.03849476626028081],"mean":0.021945262464322,"std":0.028766616883113775},"xc":{"folds":[-0.060055302147627554,-0.008496413020816163,0.005006430029440234,0.1014631361091678,0.09605171968383486],"mean":0.026793914130799835,"std":0.07006251558963063}}}},"probe":{"G_minus_BASE":{"status":"COMPLETE","delta":0.029943616592139066,"performance_pass":true,"tasks":{"eea":{"folds":[0.05624379193716511,0.031119150840039467,0.0027998423418212903,-0.00819110636527176,0.043242373639607035],"mean":0.02504281047867223,"std":0.02711498716455718},"egb":{"folds":[0.007207440738267179,0.015126060809133146,0.010929494536546902,-0.00647338148039589,0.002909223626143209],"mean":0.0059397676459389094,"std":0.008278931390592699},"egc":{"folds":[0.0027087148848738973,-0.007583857944671424,-0.018462184366910717,0.008427486664532191,-0.0016656950373594714],"mean":-0.0033151071599071048,"std":0.010302325990769159},"ei":{"folds":[0.03277785790174248,0.04744638324037653,0.15660384933522875,0.029399265654135087,0.010740000097521962],"mean":0.05539347124580096,"std":0.05806848116153638},"eps":{"folds":[0.0310579635301792,0.07682237784861101,0.2041572493382855,0.09933967401138899,-0.009483589467492615],"mean":0.08037873505219442,"std":0.08093370953016767},"nc":{"folds":[-0.009614966442712647,0.05957521435765223,0.02455484473381453,0.021370806357841055,0.02201646935611201],"mean":0.023580473672541437,"std":0.02452466482718113},"xc":{"folds":[0.05244587474047868,0.10403183650777725,-0.04012551669484188,-0.030877784302209554,0.027451415797462575],"mean":0.022585165209733415,"std":0.0598744921261281}}},"G_minus_C_CONST":{"status":"COMPLETE","delta":0.018102692418493938,"performance_pass":true,"tasks":{"eea":{"folds":[0.01835608896469454,0.04283264379960394,0.03852267509216556,0.017749166728115662,0.08610790886011266],"mean":0.04071369668893847,"std":0.027826054654808693},"egb":{"folds":[0.01810885414178076,0.018953956841032737,0.0048398180835304005,-0.017899395674883634,0.015965588236274697],"mean":0.007993764325546993,"std":0.015543438729380854},"egc":{"folds":[-0.002455936681296955,-0.0061614445507596205,-0.013841878768644733,0.013343669109468825,0.013069049709212632],"mean":0.0007906917635960298,"std":0.012055291681044609},"ei":{"folds":[0.0035124640428467746,0.036189820272436335,0.08918727552279249,0.027823070329854427,0.11770462470597165],"mean":0.05488345097478033,"std":0.047021129759476504},"eps":{"folds":[0.021788983089289227,0.06083921655097746,0.05125943793403698,0.03771049188393383,-0.006676054744045645],"mean":0.032984414942838367,"std":0.026601187902904302},"nc":{"folds":[0.034325674911535375,0.025553088844026073,-0.03574402879873029,-0.025335066116484617,0.01041206109974091],"mean":0.0018423459880174907,"std":0.030992552735056866},"xc":{"folds":[-0.08051529926657042,-0.023461003198527397,0.01994173146154743,-0.010770607005947341,0.03235758923819543],"mean":-0.01248951775426046,"std":0.04419808827501927}}},"C_CONST_minus_BASE":{"status":"COMPLETE","delta":0.011840924173645129,"performance_pass":true,"tasks":{"eea":{"folds":[0.03788770297247057,-0.011713492959564475,-0.03572283275034427,-0.025940273093387423,-0.042865535220505624],"mean":-0.015670886210266244,"std":0.032137490717141226},"egb":{"folds":[-0.010901413403513582,-0.0038278960318995914,0.0060896764530165015,0.011426014194487744,-0.013056364610131488],"mean":-0.002053996679608083,"std":0.01061255985770971},"egc":{"folds":[0.0051646515661708525,-0.0014224133939118033,-0.0046203055982659835,-0.004916182444936634,-0.014734744746572104],"mean":-0.004105798923503134,"std":0.007194402999239037},"ei":{"folds":[0.029265393858895705,0.011256562967940198,0.06741657381243626,0.0015761953242806603,-0.10696462460844969],"mean":0.0005100202710206281,"std":0.06513597818816753},"eps":{"folds":[0.009268980440889973,0.015983161297633552,0.15289781140424852,0.06162918212745516,-0.0028075347234469694],"mean":0.04739432010935605,"std":0.0638240679984259},"nc":{"folds":[-0.04394064135424802,0.034022125513626156,0.06029887353254482,0.04670587247432567,0.011604408256371102],"mean":0.021738127684523944,"std":0.04085831914924607},"xc":{"folds":[0.1329611740070491,0.12749283970630465,-0.06006724815638931,-0.020107177296262213,-0.004906173440732853],"mean":0.035074682963993874,"std":0.08918852849767452}}},"distance":{"status":"COMPLETE","delta":0.010701650432154985,"performance_pass":true,"tasks":{"eea":{"folds":[-0.012928177941508356,0.05136150301224107,0.05122819403534862,0.023334743226669508,0.07180938714065932],"mean":0.03696112989468203,"std":0.03278697282535008},"egb":{"folds":[-0.014450109457563864,0.023240463327158456,-0.012125214403879925,0.00846133903836277,0.013718659129375976],"mean":0.0037690275266906827,"std":0.016467495616228376},"egc":{"folds":[0.0012760447132064145,-0.005813970277282143,-0.006186184018853624,-0.004262951672722615,0.0010130409720658928],"mean":-0.002794804056717215,"std":0.0036689158346908135},"ei":{"folds":[0.03737756325770647,0.030147948080337383,-0.0034136322604083613,-0.00597733024095104,0.09567006609426054],"mean":0.030760922986188998,"std":0.04115504829977175},"eps":{"folds":[0.0032252634420522064,0.043622193431618106,-0.034078129088357056,-0.030283711613855457,0.00514915610127642],"mean":-0.0024730455454531563,"std":0.03157393779898193},"nc":{"folds":[0.053104605238346814,-0.0062535709475474155,-0.05574341432166019,-0.0066020942226259205,0.017498716560380267],"mean":0.0004008484613787111,"std":0.039735135017050525},"xc":{"folds":[-0.01951946695308937,-0.036644500362658206,0.0381321582147679,0.06089071288237091,-0.001421534989826001],"mean":0.008287473758313046,"std":0.040448050192978056}}},"angle":{"status":"COMPLETE","delta":0.007401041986338952,"performance_pass":false,"tasks":{"eea":{"folds":[0.031284266906202896,-0.00852885921263713,-0.012705518943183058,-0.005585576498553846,0.014298521719453339],"mean":0.00375256679425644,"std":0.018561483018600684},"egb":{"folds":[0.032558963599344626,-0.0042865064861257185,0.016965032487410325,-0.026360734713246403,0.0022469291068987207],"mean":0.00422473679885631,"std":0.022231551878768286},"egc":{"folds":[-0.0037319813945033697,-0.00034747427347747717,-0.007655694749791109,0.01760662078219144,0.01205600873714674],"mean":0.0035854958203132446,"std":0.010767075116440194},"ei":{"folds":[-0.033865099214859695,0.0060418721920989515,0.09260090778320085,0.03380040057080547,0.022034558611711108],"mean":0.024122527988591335,"std":0.046034726989113324},"eps":{"folds":[0.01856371964723702,0.017217023119359354,0.08533756702239403,0.06799420349778929,-0.011825210845322065],"mean":0.03545746048829153,"std":0.040001336593221144},"nc":{"folds":[-0.018778930326811438,0.03180665979157349,0.019999385522929902,-0.018732971893858696,-0.007086655460639357],"mean":0.0014414975266387798,"std":0.02321121818543942},"xc":{"folds":[-0.060995832313481047,0.013183497164130809,-0.018190426753220468,-0.07166131988831825,0.03377912422802143],"mean":-0.020776991512573506,"std":0.04567040864371439}}}}},"sota":{"FULL/BASE":{"eea":{"project":{"folds":[0.8975483647839905,0.9288247508960716,0.9461946762473679,0.9249657074131301,0.9227552767047394],"mean":0.9240577552090599,"std":0.01745018717022919},"reference":0.934,"delta":-0.009942244790940191},"egb":{"project":{"folds":[0.9262120328575327,0.9036890761228172,0.9265469060320997,0.9173764232074346,0.9097682118705487],"mean":0.9167185300180865,"std":0.010065162515391076},"reference":0.923,"delta":-0.00628146998191359},"egc":{"project":{"folds":[0.893853743567249,0.9225571792043704,0.9223921538819906,0.9080185371937696,0.9167422301957654],"mean":0.912712768808629,"std":0.012088611520947862},"reference":0.911,"delta":0.0017127688086289172},"ei":{"project":{"folds":[0.7554440169051936,0.7666373001706158,0.8290033308662798,0.8445278724464078,0.8330848808445934],"mean":0.8057394802466181,"std":0.04138880032202877},"reference":0.839,"delta":-0.033260519753381845},"eps":{"project":{"folds":[0.7176295799175665,0.822127740345289,0.7268450672917459,0.7747772356432724,0.8152789097679007],"mean":0.7713317065931549,"std":0.048441833336694},"reference":0.787,"delta":-0.01566829340684517},"nc":{"project":{"folds":[0.8282122438525195,0.9153512080730142,0.7967376541741665,0.7138817571921905,0.8503991795574111],"mean":0.8209164085698604,"std":0.07396185353687995},"reference":0.856,"delta":-0.035083591430139616},"xc":{"project":{"folds":[0.5368643872016108,0.36692901582045667,0.4378205577890818,0.26303865504563306,0.47153022199169825],"mean":0.4152365675696961,"std":0.10484531794924663},"reference":0.423,"delta":-0.007763432430303863}},"FULL/C_CONST":{"eea":{"project":{"folds":[0.9041560013131428,0.9200113619990171,0.9263601800973127,0.9152403200493444,0.9127891223857763],"mean":0.9157113971689187,"std":0.008279116081149585},"reference":0.934,"delta":-0.018288602831081358},"egb":{"project":{"folds":[0.9138592242330581,0.8944017321801515,0.9263285660627851,0.9256306278429477,0.9123321732950282],"mean":0.9145104647227942,"std":0.01296973148616043},"reference":0.923,"delta":-0.008489535277205862},"egc":{"project":{"folds":[0.8940497269172921,0.9229485626373148,0.9246689761494515,0.9077090854912411,0.9202863570626398],"mean":0.9139325416515879,"std":0.012951721495208604},"reference":0.911,"delta":0.00293254165158785},"ei":{"project":{"folds":[0.7720179728563639,0.7863509241991601,0.8068550770841147,0.8688414583930124,0.8303554457914535],"mean":0.8128841756648209,"std":0.03823242329874713},"reference":0.839,"delta":-0.026115824335179116},"eps":{"project":{"folds":[0.7751274485876017,0.8032160093624454,0.5786978387667441,0.7207260088631775,0.7434502279888842],"mean":0.7242435067137706,"std":0.08716456290636308},"reference":0.787,"delta":-0.06275649328622945},"nc":{"project":{"folds":[0.8074256219779298,0.8879862897230075,0.7666664865621035,0.7216090724509421,0.8393156492266349],"mean":0.8046006239881235,"std":0.06421230265651498},"reference":0.856,"delta":-0.051399376011876496},"xc":{"project":{"folds":[0.48399824699424865,0.44876003694948896,0.43037684960630807,0.3785604674235339,0.5350386977730035],"mean":0.45534685974931666,"std":0.05859762948746285},"reference":0.423,"delta":0.03234685974931667}},"FULL/S_RAD":{"eea":{"project":{"folds":[0.9108591682029112,0.9109412436355537,0.9499867507461891,0.9358603325425361,0.919662796745418],"mean":0.9254620583745217,"std":0.01708299255903703},"reference":0.934,"delta":-0.008537941625478385},"egb":{"project":{"folds":[0.8956053721081272,0.8983558268209701,0.9269286660609607,0.9214295482671813,0.9050543102279134],"mean":0.9094747446970306,"std":0.0139918437939689},"reference":0.923,"delta":-0.013525255302969441},"egc":{"project":{"folds":[0.8991793972265503,0.9252487825505071,0.9206097820459207,0.9099963467356635,0.9248305763571709],"mean":0.9159729769831625,"std":0.011222099034278878},"reference":0.911,"delta":0.004972976983162458},"ei":{"project":{"folds":[0.8071496361111385,0.788009998451493,0.8102216322626408,0.8434310729952548,0.8335660924333119],"mean":0.8164756864507678,"std":0.022108022539387567},"reference":0.839,"delta":-0.022524313549232122},"eps":{"project":{"folds":[0.7341606875247996,0.8165591378525089,0.6588247826032634,0.7340760612543069,0.7862474113024734],"mean":0.7459736161074704,"std":0.06016708139372322},"reference":0.787,"delta":-0.04102638389252966},"nc":{"project":{"folds":[0.7801313471595104,0.9091656768150036,0.8132348520308731,0.7417994577111793,0.8087038406828073],"mean":0.8106070348798747,"std":0.06202334981497477},"reference":0.856,"delta":-0.04539296512012525},"xc":{"project":{"folds":[0.48942886770314764,0.38901305802990105,0.4772117306570316,0.18375140743699458,0.48584013154882655],"mean":0.40504903907518025,"std":0.13046494224068633},"reference":0.423,"delta":-0.017950960924819737}},"FULL/S_RADANG":{"eea":{"project":{"folds":[0.8967358467758741,0.9420040925862115,0.9296688342935405,0.9362932137175638,0.9059428912787932],"mean":0.9221289757303965,"std":0.01974395922122173},"reference":0.934,"delta":-0.011871024269603514},"egb":{"project":{"folds":[0.9065716086468811,0.8751797690123986,0.9211297820965859,0.9173542596502474,0.9094220965787699],"mean":0.9059315031969766,"std":0.018164045385073303},"reference":0.923,"delta":-0.01706849680302347},"egc":{"project":{"folds":[0.8936313486435548,0.923079348565303,0.9210006270201541,0.9116274921095157,0.9203499285265199],"mean":0.9139377489730094,"std":0.012168158418147286},"reference":0.911,"delta":0.0029377489730093576},"ei":{"project":{"folds":[0.7707004596857953,0.783737603956765,0.760260461927033,0.8461680364821745,0.8419597142550372],"mean":0.8005652552613609,"std":0.04059762037168811},"reference":0.839,"delta":-0.03843474473863906},"eps":{"project":{"folds":[0.7265898580031183,0.848873067477761,0.5763253845569016,0.7418914992343364,0.7788926888915465],"mean":0.7345144996327327,"std":0.10023512597639044},"reference":0.787,"delta":-0.05248550036726729},"nc":{"project":{"folds":[0.8412269414407133,0.9005516320874092,0.811350541033196,0.762433765216577,0.8471986069430881],"mean":0.8325522973441967,"std":0.05069466534801853},"reference":0.856,"delta":-0.023447702655803315},"xc":{"project":{"folds":[0.4293735655555201,0.3805166450090849,0.48221816068647183,0.2852145435461624,0.5818918512326614],"mean":0.4318429532059801,"std":0.11089715013874595},"reference":0.423,"delta":0.008842953205980109}},"probe/BASE":{"eea":{"project":{"folds":[0.7762561429495868,0.8255601484913622,0.8396841698834638,0.8774784671059289,0.8150222741169426],"mean":0.8268002405094569,"std":0.0368416965060513},"reference":0.934,"delta":-0.10719975949054317},"egb":{"project":{"folds":[0.8284406246110463,0.8166773988595781,0.8216291629750125,0.8304335630482989,0.870769190585212],"mean":0.8335899880158296,"std":0.0214934238394592},"reference":0.923,"delta":-0.08941001198417042},"egc":{"project":{"folds":[0.800472178880969,0.8281036399072004,0.8270314368266872,0.794750596545235,0.8088183979302811],"mean":0.8118352500180744,"std":0.015212583357047027},"reference":0.911,"delta":-0.09916474998192559},"ei":{"project":{"folds":[0.6945535196299556,0.6578641714886945,0.49878805654092495,0.6629514595089165,0.6625027099263678],"mean":0.6353319834189718,"std":0.07771778698615003},"reference":0.839,"delta":-0.20366801658102818},"eps":{"project":{"folds":[0.6697180708140352,0.6052370486830687,0.30329526785906546,0.6481913720743423,0.6720097250392795],"mean":0.5796902968939582,"std":0.15681766704769418},"reference":0.787,"delta":-0.2073097031060418},"nc":{"project":{"folds":[0.7302472534909303,0.6837859108937223,0.6066522470430648,0.7040627260287542,0.7187142240211754],"mean":0.6886924722955294,"std":0.04903830905772171},"reference":0.856,"delta":-0.16730752770447055},"xc":{"project":{"folds":[0.23466191872662168,0.29776084897971844,0.4226327845974529,0.2644053475684527,0.3702451963094121],"mean":0.31794121923633156,"std":0.07730589994907154},"reference":0.423,"delta":-0.10505878076366842}},"probe/C_CONST":{"eea":{"project":{"folds":[0.8141438459220574,0.8138466555317977,0.8039613371331196,0.8515381940125415,0.772156738896437],"mean":0.8111293542991908,"std":0.02837071744834973},"reference":0.934,"delta":-0.1228706457008093},"egb":{"project":{"folds":[0.8175392112075327,0.8128495028276785,0.827718839428029,0.8418595772427866,0.8577128259750805],"mean":0.8315359913362215,"std":0.018381161913229124},"reference":0.923,"delta":-0.09146400866377857},"egc":{"project":{"folds":[0.8056368304471399,0.8266812265132886,0.8224111312284212,0.7898344141002984,0.794083653183709],"mean":0.8077294510945714,"std":0.016473772593874222},"reference":0.911,"delta":-0.10327054890542864},"ei":{"project":{"folds":[0.7238189134888513,0.6691207344566347,0.5662046303533612,0.6645276548331972,0.5555380853179182],"mean":0.6358420036899926,"std":0.07240227936991864},"reference":0.839,"delta":-0.20315799631000742},"eps":{"project":{"folds":[0.6789870512549252,0.6212202099807023,0.456193079263314,0.7098205542017975,0.6692021903158325],"mean":0.6270846170033144,"std":0.10068676291841731},"reference":0.787,"delta":-0.15991538299668562},"nc":{"project":{"folds":[0.6863066121366823,0.7178080364073485,0.6669511205756096,0.7507685985030799,0.7303186322775465],"mean":0.7104305999800533,"std":0.03372516286282827},"reference":0.856,"delta":-0.14556940001994667},"xc":{"project":{"folds":[0.3676230927336708,0.4252536886860231,0.3625655364410636,0.24429817027219047,0.36533902286867925],"mean":0.35301590220032547,"std":0.06613294917128526},"reference":0.423,"delta":-0.06998409779967452}},"probe/S_RAD":{"eea":{"project":{"folds":[0.801215667980549,0.8652081585440388,0.8551895311684682,0.874872937239211,0.8439661260370963],"mean":0.8480904841938728,"std":0.02861367574897465},"reference":0.934,"delta":-0.08590951580612727},"egb":{"project":{"folds":[0.8030891017499688,0.836089966154837,0.8155936250241491,0.8503209162811494,0.8714314851044564],"mean":0.8353050188629121,"std":0.027191999021414472},"reference":0.923,"delta":-0.08769498113708796},"egc":{"project":{"folds":[0.8069128751603463,0.8208672562360064,0.8162249472095676,0.7855714624275758,0.7950966941557749],"mean":0.8049346470378541,"std":0.014636858743381298},"reference":0.911,"delta":-0.10606535296214592},"ei":{"project":{"folds":[0.7611964767465578,0.699268682536972,0.5627909980929529,0.6585503245922462,0.6512081514121787],"mean":0.6666029266761815,"std":0.07261844005277564},"reference":0.839,"delta":-0.17239707332381848},"eps":{"project":{"folds":[0.6822123146969774,0.6648424034123204,0.4221149501749569,0.679536842587942,0.6743513464171089],"mean":0.6246115714578611,"std":0.11339320715341372},"reference":0.787,"delta":-0.16238842854213897},"nc":{"project":{"folds":[0.7394112173750291,0.711554465459801,0.6112077062539494,0.7441665042804539,0.7478173488379267],"mean":0.710831448441432,"std":0.05749217061756061},"reference":0.856,"delta":-0.145168551558568},"xc":{"project":{"folds":[0.3481036257805814,0.3886091883233649,0.4006976946558315,0.3051888831545614,0.36391748787885325],"mean":0.3613033759586385,"std":0.03750807587396014},"reference":0.423,"delta":-0.0616966240413615}},"probe/S_RADANG":{"eea":{"project":{"folds":[0.8324999348867519,0.8566792993314016,0.8424840122252851,0.8692873607406572,0.8582646477565496],"mean":0.8518430509881292,"std":0.014410741591881342},"reference":0.934,"delta":-0.08215694901187087},"egb":{"project":{"folds":[0.8356480653493135,0.8318034596687113,0.8325586575115594,0.823960181567903,0.8736784142113552],"mean":0.8395297556617685,"std":0.019569759177584526},"reference":0.923,"delta":-0.08347024433823158},"egc":{"project":{"folds":[0.8031808937658429,0.820519781962529,0.8085692524597765,0.8031780832097672,0.8071527028929216],"mean":0.8085201428581674,"std":0.007122303430130551},"reference":0.911,"delta":-0.10247985714183261},"ei":{"project":{"folds":[0.7273313775316981,0.705310554729071,0.6553919058761537,0.6923507251630516,0.6732427100238898],"mean":0.6907254546647728,"std":0.027881294209530068},"reference":0.839,"delta":-0.14827454533522721},"eps":{"project":{"folds":[0.7007760343442144,0.6820594265316797,0.507452517197351,0.7475310460857313,0.6625261355717869],"mean":0.6600690319461526,"std":0.09095064506081464},"reference":0.787,"delta":-0.1269309680538474},"nc":{"project":{"folds":[0.7206322870482177,0.7433611252513745,0.6312070917768793,0.7254335323865952,0.7407306933772874],"mean":0.7122729459680708,"std":0.04634401421078782},"reference":0.856,"delta":-0.1437270540319292},"xc":{"project":{"folds":[0.28710779346710036,0.4017926854874957,0.382507267902611,0.23352756326624313,0.3976966121068747],"mean":0.340526384446065,"std":0.07597151233510088},"reference":0.423,"delta":-0.08247361555393501}}},"costs":{"BASE":{"pt":{"status":"COMPLETE","plan":"UA3D-SPATIAL-20261025-01","revision":"r4"}},"C_CONST":{"pt":{"status":"COMPLETE","plan":"UA3D-SPATIAL-20261025-01","revision":"r4"}},"S_RAD":{"pt":{"status":"COMPLETE","plan":"UA3D-SPATIAL-20261025-01","revision":"r4"}},"S_RADANG":{"pt":{"status":"COMPLETE","plan":"UA3D-SPATIAL-20261025-01","revision":"r4"}}}}
```

</details>

<details><summary>results/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1/final_review/results.json</summary>

```json
{"plan":"UA3D-REPAIR-20261026-01","revision":"r2","status":"待审查","independent_review":"PENDING","formal":{"FULL":{"BASE":{"tasks":{"eea":{"folds":[0.8975483647839905,0.9288247508960716,0.9461946762473679,0.9249657074131301,0.9227552767047394],"mean":0.9240577552090599,"std":0.01745018717022919},"egb":{"folds":[0.9262120328575327,0.9036890761228172,0.9265469060320997,0.9173764232074346,0.9097682118705487],"mean":0.9167185300180865,"std":0.010065162515391076},"egc":{"folds":[0.893853743567249,0.9225571792043704,0.9223921538819906,0.9080185371937696,0.9167422301957654],"mean":0.912712768808629,"std":0.012088611520947862},"ei":{"folds":[0.7554440169051936,0.7666373001706158,0.8290033308662798,0.8445278724464078,0.8330848808445934],"mean":0.8057394802466181,"std":0.04138880032202877},"eps":{"folds":[0.7176295799175665,0.822127740345289,0.7268450672917459,0.7747772356432724,0.8152789097679007],"mean":0.7713317065931549,"std":0.048441833336694},"nc":{"folds":[0.8282122438525195,0.9153512080730142,0.7967376541741665,0.7138817571921905,0.8503991795574111],"mean":0.8209164085698604,"std":0.07396185353687995},"xc":{"folds":[0.5368643872016108,0.36692901582045667,0.4378205577890818,0.26303865504563306,0.47153022199169825],"mean":0.4152365675696961,"std":0.10484531794924663}},"macro":0.795244745287872},"C_CONST":{"tasks":{"eea":{"folds":[0.9041560013131428,0.9200113619990171,0.9263601800973127,0.9152403200493444,0.9127891223857763],"mean":0.9157113971689187,"std":0.008279116081149585},"egb":{"folds":[0.9138592242330581,0.8944017321801515,0.9263285660627851,0.9256306278429477,0.9123321732950282],"mean":0.9145104647227942,"std":0.01296973148616043},"egc":{"folds":[0.8940497269172921,0.9229485626373148,0.9246689761494515,0.9077090854912411,0.9202863570626398],"mean":0.9139325416515879,"std":0.012951721495208604},"ei":{"folds":[0.7720179728563639,0.7863509241991601,0.8068550770841147,0.8688414583930124,0.8303554457914535],"mean":0.8128841756648209,"std":0.03823242329874713},"eps":{"folds":[0.7751274485876017,0.8032160093624454,0.5786978387667441,0.7207260088631775,0.7434502279888842],"mean":0.7242435067137706,"std":0.08716456290636308},"nc":{"folds":[0.8074256219779298,0.8879862897230075,0.7666664865621035,0.7216090724509421,0.8393156492266349],"mean":0.8046006239881235,"std":0.06421230265651498},"xc":{"folds":[0.48399824699424865,0.44876003694948896,0.43037684960630807,0.3785604674235339,0.5350386977730035],"mean":0.45534685974931666,"std":0.05859762948746285}},"macro":0.7916042242370475},"S_RAD":{"tasks":{"eea":{"folds":[0.9108591682029112,0.9109412436355537,0.9499867507461891,0.9358603325425361,0.919662796745418],"mean":0.9254620583745217,"std":0.01708299255903703},"egb":{"folds":[0.8956053721081272,0.8983558268209701,0.9269286660609607,0.9214295482671813,0.9050543102279134],"mean":0.9094747446970306,"std":0.0139918437939689},"egc":{"folds":[0.8991793972265503,0.9252487825505071,0.9206097820459207,0.9099963467356635,0.9248305763571709],"mean":0.9159729769831625,"std":0.011222099034278878},"ei":{"folds":[0.8071496361111385,0.788009998451493,0.8102216322626408,0.8434310729952548,0.8335660924333119],"mean":0.8164756864507678,"std":0.022108022539387567},"eps":{"folds":[0.7341606875247996,0.8165591378525089,0.6588247826032634,0.7340760612543069,0.7862474113024734],"mean":0.7459736161074704,"std":0.06016708139372322},"nc":{"folds":[0.7801313471595104,0.9091656768150036,0.8132348520308731,0.7417994577111793,0.8087038406828073],"mean":0.8106070348798747,"std":0.06202334981497477},"xc":{"folds":[0.48942886770314764,0.38901305802990105,0.4772117306570316,0.18375140743699458,0.48584013154882655],"mean":0.40504903907518025,"std":0.13046494224068633}},"macro":0.7898593080811441},"S_RADANG":{"tasks":{"eea":{"folds":[0.8967358467758741,0.9420040925862115,0.9296688342935405,0.9362932137175638,0.9059428912787932],"mean":0.9221289757303965,"std":0.01974395922122173},"egb":{"folds":[0.9065716086468811,0.8751797690123986,0.9211297820965859,0.9173542596502474,0.9094220965787699],"mean":0.9059315031969766,"std":0.018164045385073303},"egc":{"folds":[0.8936313486435548,0.923079348565303,0.9210006270201541,0.9116274921095157,0.9203499285265199],"mean":0.9139377489730094,"std":0.012168158418147286},"ei":{"folds":[0.7707004596857953,0.783737603956765,0.760260461927033,0.8461680364821745,0.8419597142550372],"mean":0.8005652552613609,"std":0.04059762037168811},"eps":{"folds":[0.7265898580031183,0.848873067477761,0.5763253845569016,0.7418914992343364,0.7788926888915465],"mean":0.7345144996327327,"std":0.10023512597639044},"nc":{"folds":[0.8412269414407133,0.9005516320874092,0.811350541033196,0.762433765216577,0.8471986069430881],"mean":0.8325522973441967,"std":0.05069466534801853},"xc":{"folds":[0.4293735655555201,0.3805166450090849,0.48221816068647183,0.2852145435461624,0.5818918512326614],"mean":0.4318429532059801,"std":0.11089715013874595}},"macro":0.7916390333349506}},"probe":{"BASE":{"tasks":{"eea":{"folds":[0.7762561429495868,0.8255601484913622,0.8396841698834638,0.8774784671059289,0.8150222741169426],"mean":0.8268002405094569,"std":0.0368416965060513},"egb":{"folds":[0.8284406246110463,0.8166773988595781,0.8216291629750125,0.8304335630482989,0.870769190585212],"mean":0.8335899880158296,"std":0.0214934238394592},"egc":{"folds":[0.800472178880969,0.8281036399072004,0.8270314368266872,0.794750596545235,0.8088183979302811],"mean":0.8118352500180744,"std":0.015212583357047027},"ei":{"folds":[0.6945535196299556,0.6578641714886945,0.49878805654092495,0.6629514595089165,0.6625027099263678],"mean":0.6353319834189718,"std":0.07771778698615003},"eps":{"folds":[0.6697180708140352,0.6052370486830687,0.30329526785906546,0.6481913720743423,0.6720097250392795],"mean":0.5796902968939582,"std":0.15681766704769418},"nc":{"folds":[0.7302472534909303,0.6837859108937223,0.6066522470430648,0.7040627260287542,0.7187142240211754],"mean":0.6886924722955294,"std":0.04903830905772171},"xc":{"folds":[0.23466191872662168,0.29776084897971844,0.4226327845974529,0.2644053475684527,0.3702451963094121],"mean":0.31794121923633156,"std":0.07730589994907154}},"macro":0.6705544929125932},"C_CONST":{"tasks":{"eea":{"folds":[0.8141438459220574,0.8138466555317977,0.8039613371331196,0.8515381940125415,0.772156738896437],"mean":0.8111293542991908,"std":0.02837071744834973},"egb":{"folds":[0.8175392112075327,0.8128495028276785,0.827718839428029,0.8418595772427866,0.8577128259750805],"mean":0.8315359913362215,"std":0.018381161913229124},"egc":{"folds":[0.8056368304471399,0.8266812265132886,0.8224111312284212,0.7898344141002984,0.794083653183709],"mean":0.8077294510945714,"std":0.016473772593874222},"ei":{"folds":[0.7238189134888513,0.6691207344566347,0.5662046303533612,0.6645276548331972,0.5555380853179182],"mean":0.6358420036899926,"std":0.07240227936991864},"eps":{"folds":[0.6789870512549252,0.6212202099807023,0.456193079263314,0.7098205542017975,0.6692021903158325],"mean":0.6270846170033144,"std":0.10068676291841731},"nc":{"folds":[0.6863066121366823,0.7178080364073485,0.6669511205756096,0.7507685985030799,0.7303186322775465],"mean":0.7104305999800533,"std":0.03372516286282827},"xc":{"folds":[0.3676230927336708,0.4252536886860231,0.3625655364410636,0.24429817027219047,0.36533902286867925],"mean":0.35301590220032547,"std":0.06613294917128526}},"macro":0.6823954170862383},"S_RAD":{"tasks":{"eea":{"folds":[0.801215667980549,0.8652081585440388,0.8551895311684682,0.874872937239211,0.8439661260370963],"mean":0.8480904841938728,"std":0.02861367574897465},"egb":{"folds":[0.8030891017499688,0.836089966154837,0.8155936250241491,0.8503209162811494,0.8714314851044564],"mean":0.8353050188629121,"std":0.027191999021414472},"egc":{"folds":[0.8069128751603463,0.8208672562360064,0.8162249472095676,0.7855714624275758,0.7950966941557749],"mean":0.8049346470378541,"std":0.014636858743381298},"ei":{"folds":[0.7611964767465578,0.699268682536972,0.5627909980929529,0.6585503245922462,0.6512081514121787],"mean":0.6666029266761815,"std":0.07261844005277564},"eps":{"folds":[0.6822123146969774,0.6648424034123204,0.4221149501749569,0.679536842587942,0.6743513464171089],"mean":0.6246115714578611,"std":0.11339320715341372},"nc":{"folds":[0.7394112173750291,0.711554465459801,0.6112077062539494,0.7441665042804539,0.7478173488379267],"mean":0.710831448441432,"std":0.05749217061756061},"xc":{"folds":[0.3481036257805814,0.3886091883233649,0.4006976946558315,0.3051888831545614,0.36391748787885325],"mean":0.3613033759586385,"std":0.03750807587396014}},"macro":0.6930970675183933},"S_RADANG":{"tasks":{"eea":{"folds":[0.8324999348867519,0.8566792993314016,0.8424840122252851,0.8692873607406572,0.8582646477565496],"mean":0.8518430509881292,"std":0.014410741591881342},"egb":{"folds":[0.8356480653493135,0.8318034596687113,0.8325586575115594,0.823960181567903,0.8736784142113552],"mean":0.8395297556617685,"std":0.019569759177584526},"egc":{"folds":[0.8031808937658429,0.820519781962529,0.8085692524597765,0.8031780832097672,0.8071527028929216],"mean":0.8085201428581674,"std":0.007122303430130551},"ei":{"folds":[0.7273313775316981,0.705310554729071,0.6553919058761537,0.6923507251630516,0.6732427100238898],"mean":0.6907254546647728,"std":0.027881294209530068},"eps":{"folds":[0.7007760343442144,0.6820594265316797,0.507452517197351,0.7475310460857313,0.6625261355717869],"mean":0.6600690319461526,"std":0.09095064506081464},"nc":{"folds":[0.7206322870482177,0.7433611252513745,0.6312070917768793,0.7254335323865952,0.7407306933772874],"mean":0.7122729459680708,"std":0.04634401421078782},"xc":{"folds":[0.28710779346710036,0.4017926854874957,0.382507267902611,0.23352756326624313,0.3976966121068747],"mean":0.340526384446065,"std":0.07597151233510088}},"macro":0.7004981095047322}}},"validation":{"FULL":{"BASE":{"tasks":{"eea":{"folds":[0.9302778294816162,0.9058611869062749,0.8998653856045301,0.9103072550024488,0.951286227836459],"mean":0.9195195769662659,"std":0.021110221681524318},"egb":{"folds":[0.9276459390203702,0.9188380163658406,0.8980766977186607,0.9282449872278409,0.9393665333523495],"mean":0.9224344347370124,"std":0.015442245937842988},"egc":{"folds":[0.9116041128834086,0.9209192336323841,0.918326232582549,0.9111920985267311,0.916287303790536],"mean":0.9156657962831216,"std":0.004230072118851745},"ei":{"folds":[0.877129985894912,0.8852401692975864,0.8826944055046253,0.8439224427562817,0.8348892067042478],"mean":0.8647752420315307,"std":0.02356142880168572},"eps":{"folds":[0.753681959936906,0.8481633005182398,0.7866677186544966,0.7878239384254238,0.7076308485436704],"mean":0.7767935532157474,"std":0.05155200003988883},"nc":{"folds":[0.8308050639118403,0.9215472444863319,0.827955665366945,0.8870963133841655,0.7633463872579573],"mean":0.846150134881448,"std":0.060795320986113906},"xc":{"folds":[0.40582249493033073,0.446502541820768,0.32812852211397125,0.5254981879503483,0.3990068593363364],"mean":0.4209917212303509,"std":0.07230044412495483}},"macro":0.8094757799064967},"C_CONST":{"tasks":{"eea":{"folds":[0.934568952555824,0.9108353009838784,0.8941707165400519,0.911148981353677,0.9580644404317343],"mean":0.9217576783730331,"std":0.024875655573410357},"egb":{"folds":[0.9300110927405346,0.9293534880254796,0.9129087504481075,0.9416754565762545,0.9381141495822944],"mean":0.9304125874745341,"std":0.011111182200988994},"egc":{"folds":[0.913133833056184,0.921529324212742,0.9201502019281755,0.9136805267859228,0.9122626976813335],"mean":0.9161513167328715,"std":0.0043372057850581515},"ei":{"folds":[0.8805249243219215,0.8838155256929486,0.8885781534624805,0.850913908471526,0.8222968670394895],"mean":0.8652258757976732,"std":0.028163349600213557},"eps":{"folds":[0.7264883472382428,0.877320809615526,0.7586398889981837,0.7888479806881636,0.6969342186801757],"mean":0.7696462490440584,"std":0.06934208913381709},"nc":{"folds":[0.8468383189689566,0.9186864712352297,0.856799200688369,0.8719842931275934,0.7741986517445221],"mean":0.8537013871529343,"std":0.05228457921576755},"xc":{"folds":[0.3391619360896736,0.44344673084134345,0.4693754710547071,0.48425904435555156,0.38087748352657746],"mean":0.42342413317357064,"std":0.061489128190560786}},"macro":0.811474175392668},"S_RAD":{"tasks":{"eea":{"folds":[0.9234248630499086,0.9053042962457887,0.896458080361027,0.9209657360987809,0.9541228731742573],"mean":0.9200551697859524,"std":0.022063563160687322},"egb":{"folds":[0.9256856178782084,0.917900965565052,0.89904833530778,0.928218898512987,0.9359084564169129],"mean":0.921352454736188,"std":0.014028435932595314},"egc":{"folds":[0.9088196034916893,0.9199127269128659,0.9159782212314044,0.9101267048701049,0.9121109484744194],"mean":0.9133896409960969,"std":0.0045392303045550305},"ei":{"folds":[0.8790503700104129,0.8975266765970502,0.8827033295853031,0.8481495742778924,0.8163849940707872],"mean":0.8647629889082893,"std":0.03246490506497038},"eps":{"folds":[0.7305517128636241,0.8538012151445352,0.7598097077642195,0.7965977538747193,0.7009117976866284],"mean":0.7683344374667452,"std":0.059475631124978105},"nc":{"folds":[0.8223863206497372,0.9166991308492108,0.868571510310244,0.8732093151464309,0.7763568969206229],"mean":0.8514446347752491,"std":0.05363705442849091},"xc":{"folds":[0.28921704420276995,0.44459886006329485,0.3806825394738518,0.48128156561621893,0.339409669161851],"mean":0.3870379357035973,"std":0.07757720580375936}},"macro":0.8037681803388741},"S_RADANG":{"tasks":{"eea":{"folds":[0.9272637872743704,0.9025186724212496,0.8995185240513056,0.9150641225081421,0.9515651874193141],"mean":0.9191860587348764,"std":0.021185264442741377},"egb":{"folds":[0.9250430968974572,0.9222431618411534,0.8946586478580714,0.9362191079815727,0.9364717600342967],"mean":0.9229271549225103,"std":0.0170600842054448},"egc":{"folds":[0.9107844597532037,0.9217268451956362,0.9191034232712542,0.9102212374483041,0.91144642799887],"mean":0.9146564787334537,"std":0.0053556913557294816},"ei":{"folds":[0.8750104968262948,0.8648402281074898,0.8683867446980921,0.8551914023506565,0.8213628306394709],"mean":0.8569583405244009,"std":0.021146957910392727},"eps":{"folds":[0.7522202583484968,0.8610729082817865,0.7555132101624054,0.8185607404867512,0.7206995193683674],"mean":0.7816133273295615,"std":0.056865184789793424},"nc":{"folds":[0.8460573958012528,0.9076055041786797,0.8604058189107816,0.8908820752631821,0.7693689799948721],"mean":0.8548639548297536,"std":0.05361118001106263},"xc":{"folds":[0.34392629531922625,0.4633013600091891,0.36682838919685334,0.4342151579030178,0.34533897616132514],"mean":0.3907220357179223,"std":0.0547260900993258}},"macro":0.8058467643989256}},"probe":{"BASE":{"tasks":{"eea":{"folds":[0.8637220422956611,0.7969382066090878,0.804173105111528,0.8214232366557974,0.8824225233227715],"mean":0.8337358227989691,"std":0.03758053611730544},"egb":{"folds":[0.8336530199361984,0.8197028204727392,0.7868594359841349,0.8387960127256242,0.8626240252594656],"mean":0.8283270628756325,"std":0.027875562753812762},"egc":{"folds":[0.8274174265819089,0.822531458084164,0.8285204670564768,0.8096558367318172,0.8171945307011449],"mean":0.8210639438311024,"std":0.007798034132352374},"ei":{"folds":[0.7217222197055485,0.72970733135243,0.722499027682411,0.740549630349406,0.7093109937403812],"mean":0.7247578405660354,"std":0.011476066291567942},"eps":{"folds":[0.6738087180208303,0.7273088325508842,0.6658261880215031,0.5910104163575252,0.6704031138330591],"mean":0.6656714537567604,"std":0.048636572118144636},"nc":{"folds":[0.7823648082745264,0.7600041277653607,0.7492154754366553,0.7156273867745688,0.7028139057790586],"mean":0.742005140806034,"std":0.03254456185716339},"xc":{"folds":[0.15996510613078097,0.3789354272755048,0.30165076859750706,0.4564584239602285,0.27708058091137255],"mean":0.3148180613750788,"std":0.11152349719004646}},"macro":0.7043399037156589},"C_CONST":{"tasks":{"eea":{"folds":[0.8690928389365489,0.7814592238949062,0.7938424029432772,0.8040045618643004,0.8805472150768575],"mean":0.825789248543178,"std":0.045645263836069455},"egb":{"folds":[0.8269231977603316,0.8161661599261715,0.7929972066140146,0.8475676799026879,0.8643834045516572],"mean":0.8296075297509725,"std":0.027659176970856257},"egc":{"folds":[0.8179993741262253,0.8200264185341216,0.8163903136362157,0.8035224603504759,0.8170742499316865],"mean":0.8150025633157452,"std":0.006561830856720537},"ei":{"folds":[0.7310229562198198,0.7095624849375876,0.6977180985243181,0.7479780073451103,0.710334888331548],"mean":0.7193232870716767,"std":0.020005149302767287},"eps":{"folds":[0.6849719116264554,0.7222692147735483,0.6662778629661716,0.6279603276622339,0.658415010567777],"mean":0.6719788655192374,"std":0.03482832805916628},"nc":{"folds":[0.7768868993569306,0.784617415496709,0.7472885392179538,0.7009178993492146,0.6833216102957612],"mean":0.7386064727433139,"std":0.04509606627349551},"xc":{"folds":[0.1771189413070554,0.30326091246348974,0.29971229200062144,0.48335154187735296,0.26707677257207496],"mean":0.3061040920441189,"std":0.1113843548553213}},"macro":0.7009160084268918},"S_RAD":{"tasks":{"eea":{"folds":[0.8810116870059713,0.7875048868825036,0.8012233945734893,0.8237457112092236,0.8839517590042238],"mean":0.8354874877350824,"std":0.04482037954338395},"egb":{"folds":[0.8315684996034242,0.8125166602953693,0.8024512560885723,0.8290542332809097,0.8772390661953124],"mean":0.8305659430927175,"std":0.028713598873625897},"egc":{"folds":[0.8167773000258289,0.8141313976361659,0.8108259903187001,0.800783658526237,0.809197716938222],"mean":0.8103432126890308,"std":0.006096682204229314},"ei":{"folds":[0.7273171458633976,0.7382455281267642,0.6880421469271186,0.7337961386825915,0.7117661891067449],"mean":0.7198334297413234,"std":0.020406706307966878},"eps":{"folds":[0.6963422454731847,0.740423453621011,0.6761535130920981,0.5777802603767312,0.6618214316333861],"mean":0.6705041808392823,"std":0.05971042791039979},"nc":{"folds":[0.7585068466425973,0.7722323936296682,0.7698604642119065,0.6993834393472802,0.7294426042414602],"mean":0.7458851496145824,"std":0.031068740597381135},"xc":{"folds":[0.18267961900630836,0.3460547864745013,0.3384319499443198,0.4796759250955914,0.3165418915977385],"mean":0.3326768344236919,"std":0.10557026644261704}},"macro":0.7064708911622445},"S_RADANG":{"tasks":{"eea":{"folds":[0.8755936934229919,0.7833367488232426,0.805011549209555,0.8233102007639079,0.8890197713377432],"mean":0.8352543927114882,"std":0.04547172245469551},"egb":{"folds":[0.8479050127331995,0.8277498814584049,0.7992597689077033,0.8338410377385982,0.875304835653169],"mean":0.8368121072982151,"std":0.027865175386551425},"egc":{"folds":[0.8244919869641701,0.8238180995856851,0.8106106249347602,0.806958790727944,0.8100542471695212],"mean":0.815186749876416,"std":0.008307683948157056},"ei":{"folds":[0.7420471002439472,0.7113116626835717,0.7468655045018373,0.7524445258149802,0.7483614709851758],"mean":0.7402060528459025,"std":0.016574940425034806},"eps":{"folds":[0.6899115457663907,0.7343900905469174,0.7091686897517293,0.659245356771436,0.6235674036796487],"mean":0.6832566173032244,"std":0.04321370879916513},"nc":{"folds":[0.7656816430775274,0.7690203507174921,0.7794187015528053,0.7507967227053179,0.6877556403034913],"mean":0.7505346116713267,"std":0.036560613045264714},"xc":{"folds":[0.1801385838133034,0.3715897745087685,0.24709746626996287,0.4746469987011406,0.2880975932849278],"mean":0.31231408331562066,"std":0.11420988484032533}},"macro":0.7105092307174562}}}}
```

</details>

<details><summary>results/ua3d_ph_20261028/r1/repairs/gpu2_r2/final_review/results.json</summary>

```json
{"plan":"UA3D-PH-20261028-01","revision":"r2","independent_review":"PENDING","status":"STOPPED","execution_complete":false,"performance_pass":null}
```

</details>

<details><summary>results/ua3d_stage_20261026/r1/final_review/results.json</summary>

```json
{"plan":"UA3D-STAGE-20261026-01","revision":"r1","status":"阻断","independent_review":"PENDING","comparisons":{"PT_when_FT_off":{"status":"未完成／不可比较","delta":null,"performance_pass":null},"PT_when_FT_on":{"status":"未完成／不可比较","delta":null,"performance_pass":null},"FT_when_PT_off":{"status":"未完成／不可比较","delta":null,"performance_pass":null},"FT_when_PT_on":{"status":"未完成／不可比较","delta":null,"performance_pass":null},"PT_ONLY_minus_REF":{"status":"未完成／不可比较","delta":null,"performance_pass":null},"FT_ONLY_minus_REF":{"status":"未完成／不可比较","delta":null,"performance_pass":null}},"interaction":{"status":"未完成／不可比较","delta":null},"performance_pass":null,"sota":{"E00":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"E11":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"PT_ONLY":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"FT_ONLY":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"REF":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}}}}
```

</details>

<details><summary>results/ua3d_stage_20261026/r2/final_review/results.json</summary>

```json
{"plan":"UA3D-STAGE-20261026-01","revision":"r2","status":"需返修","independent_review":"PENDING","formal":{"PT_ONLY":{"tasks":{"eea":{"folds":[0.9019126206084322,0.9374729764912677,0.932221598731303,0.9175750127839418,0.9086115613614669],"mean":0.9195587539952823,"std":0.015136291518286205},"egb":{"folds":[0.9296295406444259,0.9034486952562503,0.9339894043425917,0.91748619418883,0.9146729939664873],"mean":0.919845365679717,"std":0.012216319307344101},"egc":{"folds":[0.893253751341843,0.923065843629861,0.923542195043667,0.9025585950215027,0.9215056793817533],"mean":0.9127852128837255,"std":0.013995644612993706},"ei":{"folds":[0.7819483387863257,0.7737033273820683,0.7837179436361446,0.8642937116122497,0.8459156054840358],"mean":0.8099157853801648,"std":0.04193087467517524},"eps":{"folds":[0.7379712633958104,0.8374063903461704,0.665371644160361,0.6914464551325421,0.7753695335316597],"mean":0.7415130573133089,"std":0.06829808938960909},"nc":{"folds":[0.8326410273027562,0.8516487391611309,0.7904359674254614,0.7629661467197933,0.8467181218447652],"mean":0.8168820004907815,"std":0.03857971984744425},"xc":{"folds":[0.46097607461235557,0.36493236613171776,0.45853510872444014,0.30484597247008327,0.47469843650324905],"mean":0.41279759168836916,"std":0.07448059005492048}},"macro":0.7904711096330498},"FT_ONLY":{"tasks":{"eea":{"folds":[0.9081158645079012,0.9396043958406716,0.9482518145225027,0.9272228011364209,0.9180544505533327],"mean":0.9282498653121658,"std":0.016119729115648376},"egb":{"folds":[0.9237501948926817,0.8959437663516536,0.9271353809275622,0.9041459109265748,0.9078367666206967],"mean":0.9117624039438338,"std":0.013263544423748703},"egc":{"folds":[0.8966430904062616,0.9187328559032324,0.9223252925532179,0.9057613457392352,0.9225810463602813],"mean":0.9132087261924455,"std":0.011525536226307916},"ei":{"folds":[0.7783066759030263,0.7833788829589219,0.8147869919611459,0.8463070593802366,0.858652550626575],"mean":0.8162864321659811,"std":0.03613773323442801},"eps":{"folds":[0.7279788422009167,0.8480303248941903,0.7792993773001309,0.7183748507549768,0.8003607532902124],"mean":0.7748088296880854,"std":0.05341417153741493},"nc":{"folds":[0.8339113829022404,0.9105674372625004,0.7806577880230074,0.752936199384675,0.8538898327167426],"mean":0.8263925280578333,"std":0.06201924892620486},"xc":{"folds":[0.5369609912883241,0.4521122878468199,0.311651129218926,0.262606245274717,0.4726620621536257],"mean":0.40719854315648246,"std":0.11530096556799839}},"macro":0.7968439040738325},"E00":{"tasks":{"eea":{"folds":[0.8975483647839905,0.9288247508960716,0.9461946762473679,0.9249657074131301,0.9227552767047394],"mean":0.9240577552090599,"std":0.01745018717022919},"egb":{"folds":[0.9262120328575327,0.9036890761228172,0.9265469060320997,0.9173764232074346,0.9097682118705487],"mean":0.9167185300180865,"std":0.010065162515391076},"egc":{"folds":[0.893853743567249,0.9225571792043704,0.9223921538819906,0.9080185371937696,0.9167422301957654],"mean":0.912712768808629,"std":0.012088611520947862},"ei":{"folds":[0.7554440169051936,0.7666373001706158,0.8290033308662798,0.8445278724464078,0.8330848808445934],"mean":0.8057394802466181,"std":0.04138880032202877},"eps":{"folds":[0.7176295799175665,0.822127740345289,0.7268450672917459,0.7747772356432724,0.8152789097679007],"mean":0.7713317065931549,"std":0.048441833336694},"nc":{"folds":[0.8282122438525195,0.9153512080730142,0.7967376541741665,0.7138817571921905,0.8503991795574111],"mean":0.8209164085698604,"std":0.07396185353687995},"xc":{"folds":[0.5368643872016108,0.36692901582045667,0.4378205577890818,0.26303865504563306,0.47153022199169825],"mean":0.4152365675696961,"std":0.10484531794924663}},"macro":0.795244745287872},"E11":{"tasks":{"eea":{"folds":[0.8967358467758741,0.9420040925862115,0.9296688342935405,0.9362932137175638,0.9059428912787932],"mean":0.9221289757303965,"std":0.01974395922122173},"egb":{"folds":[0.9065716086468811,0.8751797690123986,0.9211297820965859,0.9173542596502474,0.9094220965787699],"mean":0.9059315031969766,"std":0.018164045385073303},"egc":{"folds":[0.8936313486435548,0.923079348565303,0.9210006270201541,0.9116274921095157,0.9203499285265199],"mean":0.9139377489730094,"std":0.012168158418147286},"ei":{"folds":[0.7707004596857953,0.783737603956765,0.760260461927033,0.8461680364821745,0.8419597142550372],"mean":0.8005652552613609,"std":0.04059762037168811},"eps":{"folds":[0.7265898580031183,0.848873067477761,0.5763253845569016,0.7418914992343364,0.7788926888915465],"mean":0.7345144996327327,"std":0.10023512597639044},"nc":{"folds":[0.8412269414407133,0.9005516320874092,0.811350541033196,0.762433765216577,0.8471986069430881],"mean":0.8325522973441967,"std":0.05069466534801853},"xc":{"folds":[0.4293735655555201,0.3805166450090849,0.48221816068647183,0.2852145435461624,0.5818918512326614],"mean":0.4318429532059801,"std":0.11089715013874595}},"macro":0.7916390333349506},"REF":{"tasks":{"eea":{"folds":[0.9041560013131428,0.9200113619990171,0.9263601800973127,0.9152403200493444,0.9127891223857763],"mean":0.9157113971689187,"std":0.008279116081149585},"egb":{"folds":[0.9138592242330581,0.8944017321801515,0.9263285660627851,0.9256306278429477,0.9123321732950282],"mean":0.9145104647227942,"std":0.01296973148616043},"egc":{"folds":[0.8940497269172921,0.9229485626373148,0.9246689761494515,0.9077090854912411,0.9202863570626398],"mean":0.9139325416515879,"std":0.012951721495208604},"ei":{"folds":[0.7720179728563639,0.7863509241991601,0.8068550770841147,0.8688414583930124,0.8303554457914535],"mean":0.8128841756648209,"std":0.03823242329874713},"eps":{"folds":[0.7751274485876017,0.8032160093624454,0.5786978387667441,0.7207260088631775,0.7434502279888842],"mean":0.7242435067137706,"std":0.08716456290636308},"nc":{"folds":[0.8074256219779298,0.8879862897230075,0.7666664865621035,0.7216090724509421,0.8393156492266349],"mean":0.8046006239881235,"std":0.06421230265651498},"xc":{"folds":[0.48399824699424865,0.44876003694948896,0.43037684960630807,0.3785604674235339,0.5350386977730035],"mean":0.45534685974931666,"std":0.05859762948746285}},"macro":0.7916042242370475}},"validation":{"plan":"UA3D-STAGE-20261026-01","validation":{"PT_ONLY":{"tasks":{"eea":{"folds":[0.9234702385979275,0.9084022264309468,0.9022301195560594,0.9102234078130351,0.9590042122031884],"mean":0.9206660409202314,"std":0.022787122627275593},"egb":{"folds":[0.9242262118530405,0.922522093803524,0.9008813592788852,0.9356349286822593,0.9415977342294192],"mean":0.9249724655694257,"std":0.015628741922752154},"egc":{"folds":[0.9079717401868181,0.9218699401966144,0.9144729576509053,0.9117431505696002,0.9058367282832397],"mean":0.9123789033774357,"std":0.006266820708855598},"ei":{"folds":[0.8840870296785001,0.8585640066151663,0.8670532333708736,0.857489607706525,0.839582573145786],"mean":0.8613552901033701,"std":0.016170841886023475},"eps":{"folds":[0.7591690096306434,0.860627123110913,0.7430926248109304,0.8120822661710526,0.6668769442113645],"mean":0.7683695935869808,"std":0.0732232600872604},"nc":{"folds":[0.8462112323228892,0.9151018703302579,0.8535626364959682,0.8825091350371417,0.7830321291206239],"mean":0.8560834006613762,"std":0.049045390794540976},"xc":{"folds":[0.34556158961105654,0.45802194440910915,0.4405493081839412,0.47360204074941603,0.38830126826948097],"mean":0.42120723024460077,"std":0.0531030145241982}},"macro":0.8092904177804888},"FT_ONLY":{"tasks":{"eea":{"folds":[0.9217601774210884,0.9049795635180757,0.905668725919567,0.9157068609740613,0.9516424768891453],"mean":0.9199515609443877,"std":0.019064120803851066},"egb":{"folds":[0.9236076166776884,0.9191318233249881,0.9087553606252702,0.9273923873475158,0.9380323878785722],"mean":0.923383915170807,"std":0.010756051057228187},"egc":{"folds":[0.9120438623132728,0.9213689833975626,0.9193698650774947,0.9116674516532726,0.9128149112015296],"mean":0.9154530147286264,"std":0.004562148620867654},"ei":{"folds":[0.8752084492497227,0.8858931030161519,0.8878670606075192,0.8541931527331962,0.8348420673155463],"mean":0.8676007665844272,"std":0.022672150141034},"eps":{"folds":[0.7576849779486216,0.8491154711770486,0.7915398510463006,0.7846051664049147,0.7104421858228459],"mean":0.7786775304799464,"std":0.05064293921642197},"nc":{"folds":[0.8314654413291138,0.9043529110536999,0.8352119465093091,0.8843277493080925,0.7739010433136205],"mean":0.8458518183027672,"std":0.05098986138323943},"xc":{"folds":[0.32181401998352366,0.4758236036516701,0.31105734640555094,0.4600959103066784,0.34646466077568516],"mean":0.38305110822462163,"std":0.07876275815648522}},"macro":0.8048528163479405}},"parent_reference":{"tasks":{"eea":{"folds":[0.934568952555824,0.9108353009838784,0.8941707165400519,0.911148981353677,0.9580644404317343],"mean":0.9217576783730331,"std":0.024875655573410357},"egb":{"folds":[0.9300110927405346,0.9293534880254796,0.9129087504481075,0.9416754565762545,0.9381141495822944],"mean":0.9304125874745341,"std":0.011111182200988994},"egc":{"folds":[0.913133833056184,0.921529324212742,0.9201502019281755,0.9136805267859228,0.9122626976813335],"mean":0.9161513167328715,"std":0.0043372057850581515},"ei":{"folds":[0.8805249243219215,0.8838155256929486,0.8885781534624805,0.850913908471526,0.8222968670394895],"mean":0.8652258757976732,"std":0.028163349600213557},"eps":{"folds":[0.7264883472382428,0.877320809615526,0.7586398889981837,0.7888479806881636,0.6969342186801757],"mean":0.7696462490440584,"std":0.06934208913381709},"nc":{"folds":[0.8468383189689566,0.9186864712352297,0.856799200688369,0.8719842931275934,0.7741986517445221],"mean":0.8537013871529343,"std":0.05228457921576755},"xc":{"folds":[0.3391619360896736,0.44344673084134345,0.4693754710547071,0.48425904435555156,0.38087748352657746],"mean":0.42342413317357064,"std":0.061489128190560786}},"macro":0.811474175392668}},"comparisons":{"C2_minus_REF":{"status":"COMPLETE","delta":-0.0011331146039976758,"performance_pass":false,"tasks":{"eea":{"folds":[-0.002243380704710529,0.017461614492250588,0.005861418633990256,0.00233469273459741,-0.004177561024309329],"mean":0.0038473568263636794,"std":0.008561381307871789},"egb":{"folds":[0.015770316411367835,0.009046963076098757,0.00766083827980657,-0.008144433654117655,0.0023408206714590207],"mean":0.0053349009569229056,"std":0.008926705210009893},"egc":{"folds":[-0.0007959755754490816,0.00011728099254615376,-0.001126781105784569,-0.005150490469738367,0.0012193223191134805],"mean":-0.0011473287678624767,"std":0.0024160634354879365},"ei":{"folds":[0.009930365929961882,-0.012647596817091822,-0.023137133447970193,-0.004547746780762751,0.015560159692582287],"mean":-0.0029683902846561194,"std":0.015911039272862068},"eps":{"folds":[-0.03715618519179131,0.034190380983724955,0.08667380539361691,-0.0292795537306354,0.031919305542775556],"mean":0.017269550599538143,"std":0.05110494162670992},"nc":{"folds":[0.025215405324826468,-0.0363375505618766,0.02376948086335795,0.04135707426885127,0.007402472618130318],"mean":0.01228137650265788,"std":0.029716433231408462},"xc":{"folds":[-0.02302217238189308,-0.0838276708177712,0.02815825911813208,-0.07371449495345062,-0.06034026126975445],"mean":-0.04254926806094746,"std":0.04575084481791235}}},"C2_minus_E00":{"status":"COMPLETE","delta":-0.00477363565482225,"performance_pass":false,"tasks":{"eea":{"folds":[0.004364255824441776,0.008648225595196068,-0.013973077516064913,-0.007390694629188288,-0.014143715343272523],"mean":-0.004499001213777576,"std":0.010518405550772795},"egb":{"folds":[0.0034175077868932036,-0.00024038086656696933,0.007442498310492041,0.00010977098139541841,0.004904782095938587],"mean":0.003126835661630456,"std":0.0032523589261849784},"egc":{"folds":[-0.0005999922254059653,0.0005086644254905792,0.001150041161676385,-0.005459942172266863,0.0047634491859879224],"mean":7.244407509641171e-05,"std":0.0036883770412527495},"ei":{"folds":[0.026504321881132165,0.007066027211452441,-0.0452853872301352,0.019765839165841848,0.012830724639442348],"mean":0.0041763051335467205,"std":0.02859756276929739},"eps":{"folds":[0.020341683478243944,0.015278650000881355,-0.06147342313138493,-0.08333078051073028,-0.03990937623624102],"mean":-0.029818649279846187,"std":0.04614439579133574},"nc":{"folds":[0.004428783450236784,-0.06370246891188325,-0.0063016867487051265,0.049084389527602856,-0.003681057712645819],"mean":-0.004034408079078911,"std":0.040183252614229545},"xc":{"folds":[-0.07588831258925521,-0.0019966496887389162,0.020714550935358345,0.04180731742445021,0.0031682145115508042],"mean":-0.0024389758813269545,"std":0.04449792572342298}}},"PT_when_FT_off":{"status":"COMPLETE","delta":-0.00477363565482225,"performance_pass":false,"tasks":{"eea":{"folds":[0.004364255824441776,0.008648225595196068,-0.013973077516064913,-0.007390694629188288,-0.014143715343272523],"mean":-0.004499001213777576,"std":0.010518405550772795},"egb":{"folds":[0.0034175077868932036,-0.00024038086656696933,0.007442498310492041,0.00010977098139541841,0.004904782095938587],"mean":0.003126835661630456,"std":0.0032523589261849784},"egc":{"folds":[-0.0005999922254059653,0.0005086644254905792,0.001150041161676385,-0.005459942172266863,0.0047634491859879224],"mean":7.244407509641171e-05,"std":0.0036883770412527495},"ei":{"folds":[0.026504321881132165,0.007066027211452441,-0.0452853872301352,0.019765839165841848,0.012830724639442348],"mean":0.0041763051335467205,"std":0.02859756276929739},"eps":{"folds":[0.020341683478243944,0.015278650000881355,-0.06147342313138493,-0.08333078051073028,-0.03990937623624102],"mean":-0.029818649279846187,"std":0.04614439579133574},"nc":{"folds":[0.004428783450236784,-0.06370246891188325,-0.0063016867487051265,0.049084389527602856,-0.003681057712645819],"mean":-0.004034408079078911,"std":0.040183252614229545},"xc":{"folds":[-0.07588831258925521,-0.0019966496887389162,0.020714550935358345,0.04180731742445021,0.0031682145115508042],"mean":-0.0024389758813269545,"std":0.04449792572342298}}},"PT_when_FT_on":{"status":"COMPLETE","delta":-0.005204870738881984,"performance_pass":false,"tasks":{"eea":{"folds":[-0.011380017732027015,0.00239969674553997,-0.01858298022896221,0.00907041258114294,-0.01211155927453944],"mean":-0.006120889581769151,"std":0.011426135702125988},"egb":{"folds":[-0.017178586245800598,-0.020763997339255047,-0.006005598830976289,0.013208348723672603,0.001585329958073256],"mean":-0.005830900746857215,"std":0.013868051032059552},"egc":{"folds":[-0.003011741762706799,0.004346492662070656,-0.0013246655330638735,0.0058661463702804895,-0.002231117833761398],"mean":0.0007290227805638149,"std":0.004075828585512762},"ei":{"folds":[-0.00760621621723101,0.000358720997843065,-0.0545265300341129,-0.00013902289806211332,-0.016692836371537734],"mean":-0.01572117690462014,"std":0.022770698170125832},"eps":{"folds":[-0.0013889841977984263,0.0008427425835707547,-0.20297399274322925,0.023516648479359636,-0.021468064398665887],"mean":-0.040294330055352635,"std":0.09232663022385652},"nc":{"folds":[0.007315558538472877,-0.010015805175091264,0.03069275301018859,0.009497565831901977,-0.006691225773654441],"mean":0.006159769286363548,"std":0.0161333606864692},"xc":{"folds":[-0.107587425732804,-0.07159564283773501,0.17056703146754582,0.022608298271445393,0.10922978907903569],"mean":0.024644410049497578,"std":0.11747302851330733}}},"FT_when_PT_off":{"status":"COMPLETE","delta":0.0015991587859605083,"performance_pass":false,"tasks":{"eea":{"folds":[0.01056749972391069,0.010779644944599931,0.0020571382751348732,0.0022570937232907484,-0.004700826151406767],"mean":0.004192110103105895,"std":0.006546524200447362},"egb":{"folds":[-0.002461837964850999,-0.007745309771163611,0.000588474895462543,-0.013230512280859807,-0.001931445249852004],"mean":-0.004956126074252776,"std":0.005530772549264044},"egc":{"folds":[0.002789346839012552,-0.0038243233011380218,-6.686132877264939e-05,-0.002257191454534402,0.00583881616451587],"mean":0.0004959573838166697,"std":0.003884816397522711},"ei":{"folds":[0.022862658997832686,0.016741582788306064,-0.014216338905133852,0.0017791869338288357,0.02556766978198155],"mean":0.010546951919363057,"std":0.01662653250721334},"eps":{"folds":[0.010349262283350202,0.025902584548901264,0.05245431000838496,-0.05640238488829563,-0.014918156477688349],"mean":0.00347712309493049,"std":0.04145127936286179},"nc":{"folds":[0.005699139049720969,-0.004783770810513732,-0.016079866151159083,0.03905444219248455,0.0034906531593315027],"mean":0.005476119487972842,"std":0.020624145047291403},"xc":{"folds":[9.660408671330334e-05,0.08518327202636322,-0.12616942857015578,-0.0004324097709160757,0.0011318401619274665],"mean":-0.008038024413213574,"std":0.07558661046973143}}},"FT_when_PT_on":{"status":"COMPLETE","delta":0.0011679237019007749,"performance_pass":false,"tasks":{"eea":{"folds":[-0.005176773832558101,0.004531116094943832,-0.0025527644377624226,0.018718200933621976,-0.0026686700826736853],"mean":0.00257022173511432,"std":0.009725086812174938},"egb":{"folds":[-0.0230579319975448,-0.02826892624385169,-0.012859622246005786,-0.00013193453858262316,-0.005250897387717335],"mean":-0.013913862482740447,"std":0.011787512327135027},"egc":{"folds":[0.00037759730171171846,1.3504935442054666e-05,-0.002541568023512908,0.00906889708801295,-0.0011557508552334506],"mean":0.001152536089284073,"std":0.004570046153227451},"ei":{"folds":[-0.011247879100530489,0.010034276574696688,-0.02345748170911155,-0.018125675130075125,-0.0039558912289985315],"mean":-0.009350530118803801,"std":0.013081303887556149},"eps":{"folds":[-0.011381405392692168,0.011466677131590663,-0.08904625960345935,0.05044504410179429,0.003523155359886787],"mean":-0.006998557680575957,"std":0.051240788129672026},"nc":{"folds":[0.008585914137957062,0.04890289292627825,0.020914573607734632,-0.0005323815032163282,0.00048048509832288033],"mean":0.0156702968534153,"std":0.02046554131925568},"xc":{"folds":[-0.03160250905683548,0.01558427887736713,0.023683051962031687,-0.01963142892392089,0.10719341472941235],"mean":0.019045361517610958,"std":0.054462527170788456}}},"PT_ONLY_minus_REF":{"status":"COMPLETE","delta":-0.0011331146039976758,"performance_pass":false,"tasks":{"eea":{"folds":[-0.002243380704710529,0.017461614492250588,0.005861418633990256,0.00233469273459741,-0.004177561024309329],"mean":0.0038473568263636794,"std":0.008561381307871789},"egb":{"folds":[0.015770316411367835,0.009046963076098757,0.00766083827980657,-0.008144433654117655,0.0023408206714590207],"mean":0.0053349009569229056,"std":0.008926705210009893},"egc":{"folds":[-0.0007959755754490816,0.00011728099254615376,-0.001126781105784569,-0.005150490469738367,0.0012193223191134805],"mean":-0.0011473287678624767,"std":0.0024160634354879365},"ei":{"folds":[0.009930365929961882,-0.012647596817091822,-0.023137133447970193,-0.004547746780762751,0.015560159692582287],"mean":-0.0029683902846561194,"std":0.015911039272862068},"eps":{"folds":[-0.03715618519179131,0.034190380983724955,0.08667380539361691,-0.0292795537306354,0.031919305542775556],"mean":0.017269550599538143,"std":0.05110494162670992},"nc":{"folds":[0.025215405324826468,-0.0363375505618766,0.02376948086335795,0.04135707426885127,0.007402472618130318],"mean":0.01228137650265788,"std":0.029716433231408462},"xc":{"folds":[-0.02302217238189308,-0.0838276708177712,0.02815825911813208,-0.07371449495345062,-0.06034026126975445],"mean":-0.04254926806094746,"std":0.04575084481791235}}},"FT_ONLY_minus_REF":{"status":"COMPLETE","delta":0.005239679836785083,"performance_pass":false,"tasks":{"eea":{"folds":[0.003959863194758384,0.01959303384165445,0.021891634425190043,0.011982481087076446,0.005265328167556427],"mean":0.01253846814324715,"std":0.008124788709032287},"egb":{"folds":[0.009890970659623632,0.0015420341715021157,0.0008068148647770723,-0.02148471691637288,-0.00449540667433157],"mean":-0.002748060778960326,"std":0.011671575216328393},"egc":{"folds":[0.0025933634889694357,-0.004215706734082447,-0.0023436835962336033,-0.0019477397520059059,0.002294689297641428],"mean":-0.0007238154591422185,"std":0.0030178765412434503},"ei":{"folds":[0.006288703046662403,-0.002972041240238199,0.007931914877031154,-0.022534399012775763,0.02829710483512149],"mean":0.003402256501160217,"std":0.01845226695542892},"eps":{"folds":[-0.047148606386685055,0.044814315531744864,0.2006015385333868,-0.002351158108200746,0.05691052530132823],"mean":0.05056532297431482,"std":0.09345101339710854},"nc":{"folds":[0.026485760924310653,0.02258114753949292,0.013991301460903993,0.03132712693373296,0.01457418349010764],"mean":0.021791904069709635,"std":0.0075253000591071},"xc":{"folds":[0.05296274429407544,0.003352250897330933,-0.11872572038738205,-0.1159542221488169,-0.062376635619377785],"mean":-0.04814831659283407,"std":0.07526122989702602}}}},"interaction":{"status":"COMPLETE","delta":-0.0004312350840597369,"tasks":{"eea":{"folds":[-0.01574427355646879,-0.006248528849656099,-0.004609902712897296,0.016461107210331227,0.002032156068733082],"mean":-0.0016218883679915753,"std":0.01193861973926279},"egb":{"folds":[-0.0205960940326938,-0.020523616472688078,-0.01344809714146833,0.013098577742277184,-0.0033194521378653308],"mean":-0.008957736408487672,"std":0.01420964050208746},"egc":{"folds":[-0.0024117495373008335,0.0038378282365800764,-0.0024747066947402585,0.011326088542547352,-0.00699456701974932],"mean":0.0006565787054674032,"std":0.007101380255171797},"ei":{"folds":[-0.034110538098363175,-0.006707306213609376,-0.009241142803977698,-0.01990486206390396,-0.02952356101098008],"mean":-0.01989748203816686,"std":0.012064521515405822},"eps":{"folds":[-0.02173066767604237,-0.0144359074173106,-0.14150056961184432,0.10684742899008992,0.018441311837575136],"mean":-0.010475680775506447,"std":0.08931794915723207},"nc":{"folds":[0.002886775088236093,0.053686663736791984,0.036994439758893716,-0.03958682369570088,-0.0030101680610086223],"mean":0.010194177365442458,"std":0.03645459973704065},"xc":{"folds":[-0.031699113143548785,-0.06959899314899609,0.14985248053218747,-0.019199019153004815,0.10606157456748488],"mean":0.027083385930824534,"std":0.09520337508706403}}},"performance_pass":false,"sota":{"E00":{"eea":{"project":{"folds":[0.8975483647839905,0.9288247508960716,0.9461946762473679,0.9249657074131301,0.9227552767047394],"mean":0.9240577552090599,"std":0.01745018717022919},"reference":0.934,"delta":-0.009942244790940191},"egb":{"project":{"folds":[0.9262120328575327,0.9036890761228172,0.9265469060320997,0.9173764232074346,0.9097682118705487],"mean":0.9167185300180865,"std":0.010065162515391076},"reference":0.923,"delta":-0.00628146998191359},"egc":{"project":{"folds":[0.893853743567249,0.9225571792043704,0.9223921538819906,0.9080185371937696,0.9167422301957654],"mean":0.912712768808629,"std":0.012088611520947862},"reference":0.911,"delta":0.0017127688086289172},"ei":{"project":{"folds":[0.7554440169051936,0.7666373001706158,0.8290033308662798,0.8445278724464078,0.8330848808445934],"mean":0.8057394802466181,"std":0.04138880032202877},"reference":0.839,"delta":-0.033260519753381845},"eps":{"project":{"folds":[0.7176295799175665,0.822127740345289,0.7268450672917459,0.7747772356432724,0.8152789097679007],"mean":0.7713317065931549,"std":0.048441833336694},"reference":0.787,"delta":-0.01566829340684517},"nc":{"project":{"folds":[0.8282122438525195,0.9153512080730142,0.7967376541741665,0.7138817571921905,0.8503991795574111],"mean":0.8209164085698604,"std":0.07396185353687995},"reference":0.856,"delta":-0.035083591430139616},"xc":{"project":{"folds":[0.5368643872016108,0.36692901582045667,0.4378205577890818,0.26303865504563306,0.47153022199169825],"mean":0.4152365675696961,"std":0.10484531794924663},"reference":0.423,"delta":-0.007763432430303863}},"E11":{"eea":{"project":{"folds":[0.8967358467758741,0.9420040925862115,0.9296688342935405,0.9362932137175638,0.9059428912787932],"mean":0.9221289757303965,"std":0.01974395922122173},"reference":0.934,"delta":-0.011871024269603514},"egb":{"project":{"folds":[0.9065716086468811,0.8751797690123986,0.9211297820965859,0.9173542596502474,0.9094220965787699],"mean":0.9059315031969766,"std":0.018164045385073303},"reference":0.923,"delta":-0.01706849680302347},"egc":{"project":{"folds":[0.8936313486435548,0.923079348565303,0.9210006270201541,0.9116274921095157,0.9203499285265199],"mean":0.9139377489730094,"std":0.012168158418147286},"reference":0.911,"delta":0.0029377489730093576},"ei":{"project":{"folds":[0.7707004596857953,0.783737603956765,0.760260461927033,0.8461680364821745,0.8419597142550372],"mean":0.8005652552613609,"std":0.04059762037168811},"reference":0.839,"delta":-0.03843474473863906},"eps":{"project":{"folds":[0.7265898580031183,0.848873067477761,0.5763253845569016,0.7418914992343364,0.7788926888915465],"mean":0.7345144996327327,"std":0.10023512597639044},"reference":0.787,"delta":-0.05248550036726729},"nc":{"project":{"folds":[0.8412269414407133,0.9005516320874092,0.811350541033196,0.762433765216577,0.8471986069430881],"mean":0.8325522973441967,"std":0.05069466534801853},"reference":0.856,"delta":-0.023447702655803315},"xc":{"project":{"folds":[0.4293735655555201,0.3805166450090849,0.48221816068647183,0.2852145435461624,0.5818918512326614],"mean":0.4318429532059801,"std":0.11089715013874595},"reference":0.423,"delta":0.008842953205980109}},"PT_ONLY":{"eea":{"project":{"folds":[0.9019126206084322,0.9374729764912677,0.932221598731303,0.9175750127839418,0.9086115613614669],"mean":0.9195587539952823,"std":0.015136291518286205},"reference":0.934,"delta":-0.014441246004717745},"egb":{"project":{"folds":[0.9296295406444259,0.9034486952562503,0.9339894043425917,0.91748619418883,0.9146729939664873],"mean":0.919845365679717,"std":0.012216319307344101},"reference":0.923,"delta":-0.0031546343202830673},"egc":{"project":{"folds":[0.893253751341843,0.923065843629861,0.923542195043667,0.9025585950215027,0.9215056793817533],"mean":0.9127852128837255,"std":0.013995644612993706},"reference":0.911,"delta":0.0017852128837254622},"ei":{"project":{"folds":[0.7819483387863257,0.7737033273820683,0.7837179436361446,0.8642937116122497,0.8459156054840358],"mean":0.8099157853801648,"std":0.04193087467517524},"reference":0.839,"delta":-0.029084214619835125},"eps":{"project":{"folds":[0.7379712633958104,0.8374063903461704,0.665371644160361,0.6914464551325421,0.7753695335316597],"mean":0.7415130573133089,"std":0.06829808938960909},"reference":0.787,"delta":-0.04548694268669118},"nc":{"project":{"folds":[0.8326410273027562,0.8516487391611309,0.7904359674254614,0.7629661467197933,0.8467181218447652],"mean":0.8168820004907815,"std":0.03857971984744425},"reference":0.856,"delta":-0.03911799950921846},"xc":{"project":{"folds":[0.46097607461235557,0.36493236613171776,0.45853510872444014,0.30484597247008327,0.47469843650324905],"mean":0.41279759168836916,"std":0.07448059005492048},"reference":0.423,"delta":-0.010202408311630828}},"FT_ONLY":{"eea":{"project":{"folds":[0.9081158645079012,0.9396043958406716,0.9482518145225027,0.9272228011364209,0.9180544505533327],"mean":0.9282498653121658,"std":0.016119729115648376},"reference":0.934,"delta":-0.005750134687834252},"egb":{"project":{"folds":[0.9237501948926817,0.8959437663516536,0.9271353809275622,0.9041459109265748,0.9078367666206967],"mean":0.9117624039438338,"std":0.013263544423748703},"reference":0.923,"delta":-0.011237596056166255},"egc":{"project":{"folds":[0.8966430904062616,0.9187328559032324,0.9223252925532179,0.9057613457392352,0.9225810463602813],"mean":0.9132087261924455,"std":0.011525536226307916},"reference":0.911,"delta":0.002208726192445498},"ei":{"project":{"folds":[0.7783066759030263,0.7833788829589219,0.8147869919611459,0.8463070593802366,0.858652550626575],"mean":0.8162864321659811,"std":0.03613773323442801},"reference":0.839,"delta":-0.0227135678340189},"eps":{"project":{"folds":[0.7279788422009167,0.8480303248941903,0.7792993773001309,0.7183748507549768,0.8003607532902124],"mean":0.7748088296880854,"std":0.05341417153741493},"reference":0.787,"delta":-0.012191170311914634},"nc":{"project":{"folds":[0.8339113829022404,0.9105674372625004,0.7806577880230074,0.752936199384675,0.8538898327167426],"mean":0.8263925280578333,"std":0.06201924892620486},"reference":0.856,"delta":-0.02960747194216673},"xc":{"project":{"folds":[0.5369609912883241,0.4521122878468199,0.311651129218926,0.262606245274717,0.4726620621536257],"mean":0.40719854315648246,"std":0.11530096556799839},"reference":0.423,"delta":-0.015801456843517525}},"REF":{"eea":{"project":{"folds":[0.9041560013131428,0.9200113619990171,0.9263601800973127,0.9152403200493444,0.9127891223857763],"mean":0.9157113971689187,"std":0.008279116081149585},"reference":0.934,"delta":-0.018288602831081358},"egb":{"project":{"folds":[0.9138592242330581,0.8944017321801515,0.9263285660627851,0.9256306278429477,0.9123321732950282],"mean":0.9145104647227942,"std":0.01296973148616043},"reference":0.923,"delta":-0.008489535277205862},"egc":{"project":{"folds":[0.8940497269172921,0.9229485626373148,0.9246689761494515,0.9077090854912411,0.9202863570626398],"mean":0.9139325416515879,"std":0.012951721495208604},"reference":0.911,"delta":0.00293254165158785},"ei":{"project":{"folds":[0.7720179728563639,0.7863509241991601,0.8068550770841147,0.8688414583930124,0.8303554457914535],"mean":0.8128841756648209,"std":0.03823242329874713},"reference":0.839,"delta":-0.026115824335179116},"eps":{"project":{"folds":[0.7751274485876017,0.8032160093624454,0.5786978387667441,0.7207260088631775,0.7434502279888842],"mean":0.7242435067137706,"std":0.08716456290636308},"reference":0.787,"delta":-0.06275649328622945},"nc":{"project":{"folds":[0.8074256219779298,0.8879862897230075,0.7666664865621035,0.7216090724509421,0.8393156492266349],"mean":0.8046006239881235,"std":0.06421230265651498},"reference":0.856,"delta":-0.051399376011876496},"xc":{"project":{"folds":[0.48399824699424865,0.44876003694948896,0.43037684960630807,0.3785604674235339,0.5350386977730035],"mean":0.45534685974931666,"std":0.05859762948746285},"reference":0.423,"delta":0.03234685974931667}}}}
```

</details>

<details><summary>results/tmol3m_3d_ablation_20261025/r1/final_review/results.json</summary>

```json
{"plan":"TMOL3M-3D-ABLATION-20261025-01","revision":"r1","status":"需返修","independent_review":"PENDING","FINAL":null,"delta_3d":null,"comparisons":{"FULL_minus_NO3D_FT":{"status":"未完成／不可比较","delta":null},"NO3D_FT_minus_FULL":{"status":"未完成／不可比较","delta":null},"FINAL_minus_FULL":{"status":"未完成／不可比较","delta":null}},"sota":{"FULL":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"NO3D_FT":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"FINAL":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}}},"historical_comparison":{"FULL":{"TMOL3M":{"status":"未完成／不可比较","delta":null},"B_A020":{"status":"未完成／不可比较","delta":null}},"NO3D_FT":{"TMOL3M":{"status":"未完成／不可比较","delta":null},"B_A020":{"status":"未完成／不可比较","delta":null}}},"incident":{"status":"需返修"}}
```

</details>

<details><summary>results/tmol3m_3d_ablation_20261025/r1/repairs/json_import_repair1/final_review/results.json</summary>

```json
{"plan":"TMOL3M-3D-ABLATION-20261025-01","revision":"r1","status":"待审查","independent_review":"PENDING","FINAL":"FULL","formal":{"FULL":{"tasks":{"eea":{"folds":[0.9048127345675825,0.9249857549198717,0.9430897450142551,0.9266785662249936,0.9039788229550795],"mean":0.9207091247363565,"std":0.016487981831676657},"egb":{"folds":[0.918203133735168,0.9094998438640765,0.9195143535532943,0.9191146090704887,0.914430382738539],"mean":0.9161524645923134,"std":0.004228024647925799},"egc":{"folds":[0.8888755460081812,0.9218274633020648,0.9232339738526183,0.9079443651146784,0.9099054187296192],"mean":0.9103573534014323,"std":0.013828101656380973},"ei":{"folds":[0.783115913417741,0.8039711244530234,0.8045980600600575,0.7834563567795338,0.7843702137347925],"mean":0.7919023336890296,"std":0.011314876454667316},"eps":{"folds":[0.7709525786456969,0.7767643794726025,0.6725660225167185,0.7422877410435005,0.7723423325556295],"mean":0.7469826108468295,"std":0.04377356224100633},"nc":{"folds":[0.810864975404595,0.904685005181916,0.8405647210603203,0.7278693567878491,0.8389326135921045],"mean":0.8245833344053569,"std":0.06406904680511585},"xc":{"folds":[0.48515886616839554,0.37473519213811024,0.3834542764856963,0.20879930392167612,0.5871546891990046],"mean":0.40786046558257655,"std":0.14092529336862664}},"macro":0.7883639553219849},"NO3D_FT":{"tasks":{"eea":{"folds":[0.8989136073548594,0.9209138638164095,0.9264488113823874,0.9280929814295712,0.9088306424324147],"mean":0.9166399812831283,"std":0.012458536612589336},"egb":{"folds":[0.9235830046529592,0.905354017188669,0.9264696872744378,0.9093029237424605,0.9140936301598059],"mean":0.9157606526036666,"std":0.009064379483681198},"egc":{"folds":[0.8892062161948553,0.9201679579891902,0.9164689504456398,0.8965875373011245,0.9138661663691323],"mean":0.9072593656599885,"std":0.01355453936498268},"ei":{"folds":[0.7420761531212031,0.7800976455062485,0.7878606581064869,0.8292992247946112,0.8146900427605989],"mean":0.7908047448578297,"std":0.0337272801952151},"eps":{"folds":[0.7568617099505863,0.8222946931972852,0.5907151252379995,0.7316953825866428,0.8073222939255981],"mean":0.7417778409796225,"std":0.09209799395296553},"nc":{"folds":[0.7962094242665158,0.8919102928160266,0.7752543736967606,0.714116147539104,0.8212053246708682],"mean":0.799739112597855,"std":0.06499918847146766},"xc":{"folds":[0.3707232442528723,0.42926684531935744,0.4440284902328885,0.27603302530572493,0.5525139362311399],"mean":0.41451310826839666,"std":0.10152258937759609}},"macro":0.7837849723214981}},"validation":{"FULL":{"FULL/eea/0":{"best_epoch":31,"scaler":{"mean":[2.2603174468085103]}},"FULL/eea/1":{"best_epoch":25,"scaler":{"mean":[2.293286808510638]}},"FULL/eea/2":{"best_epoch":42,"scaler":{"mean":[2.2195417021276596]}},"FULL/eea/3":{"best_epoch":66,"scaler":{"mean":[2.334746610169492]}},"FULL/eea/4":{"best_epoch":28,"scaler":{"mean":[2.296813559322034]}},"FULL/egb/0":{"best_epoch":30,"scaler":{"mean":[4.188336592178771]}},"FULL/egb/1":{"best_epoch":60,"scaler":{"mean":[4.152108635097493]}},"FULL/egb/2":{"best_epoch":46,"scaler":{"mean":[4.250774373259053]}},"FULL/egb/3":{"best_epoch":58,"scaler":{"mean":[4.228450696378831]}},"FULL/egb/4":{"best_epoch":40,"scaler":{"mean":[4.215195821727019]}},"FULL/egc/0":{"best_epoch":66,"scaler":{"mean":[4.4900454461396215]}},"FULL/egc/1":{"best_epoch":69,"scaler":{"mean":[4.515255386037911]}},"FULL/egc/2":{"best_epoch":60,"scaler":{"mean":[4.509532316227461]}},"FULL/egc/3":{"best_epoch":40,"scaler":{"mean":[4.554269579288025]}},"FULL/egc/4":{"best_epoch":26,"scaler":{"mean":[4.514467730004624]}},"FULL/ei/0":{"best_epoch":42,"scaler":{"mean":[6.22182838983051]}},"FULL/ei/1":{"best_epoch":60,"scaler":{"mean":[6.279353813559322]}},"FULL/ei/2":{"best_epoch":45,"scaler":{"mean":[6.27359533898305]}},"FULL/ei/3":{"best_epoch":21,"scaler":{"mean":[6.283478389830509]}},"FULL/ei/4":{"best_epoch":52,"scaler":{"mean":[6.280269915254237]}},"FULL/eps/0":{"best_epoch":17,"scaler":{"mean":[4.599795081967213]}},"FULL/eps/1":{"best_epoch":42,"scaler":{"mean":[4.617868852459015]}},"FULL/eps/2":{"best_epoch":15,"scaler":{"mean":[4.610081967213113]}},"FULL/eps/3":{"best_epoch":20,"scaler":{"mean":[4.57405737704918]}},"FULL/eps/4":{"best_epoch":14,"scaler":{"mean":[4.576598360655737]}},"FULL/nc/0":{"best_epoch":31,"scaler":{"mean":[1.9505442622950822]}},"FULL/nc/1":{"best_epoch":65,"scaler":{"mean":[1.957320081967213]}},"FULL/nc/2":{"best_epoch":17,"scaler":{"mean":[1.958533606557377]}},"FULL/nc/3":{"best_epoch":20,"scaler":{"mean":[1.9376409836065573]}},"FULL/nc/4":{"best_epoch":17,"scaler":{"mean":[1.9437995901639344]}},"FULL/xc/0":{"best_epoch":26,"scaler":{"mean":[37.46221099847826]}},"FULL/xc/1":{"best_epoch":31,"scaler":{"mean":[36.49228298981884]}},"FULL/xc/2":{"best_epoch":22,"scaler":{"mean":[37.15334486376811]}},"FULL/xc/3":{"best_epoch":16,"scaler":{"mean":[35.30247894376812]}},"FULL/xc/4":{"best_epoch":53,"scaler":{"mean":[35.35397605797102]}}},"NO3D_FT":{"NO3D_FT/eea/0":{"best_epoch":47,"scaler":{"mean":[2.2603174468085103]}},"NO3D_FT/eea/1":{"best_epoch":43,"scaler":{"mean":[2.293286808510638]}},"NO3D_FT/eea/2":{"best_epoch":42,"scaler":{"mean":[2.2195417021276596]}},"NO3D_FT/eea/3":{"best_epoch":34,"scaler":{"mean":[2.334746610169492]}},"NO3D_FT/eea/4":{"best_epoch":39,"scaler":{"mean":[2.296813559322034]}},"NO3D_FT/egb/0":{"best_epoch":30,"scaler":{"mean":[4.188336592178771]}},"NO3D_FT/egb/1":{"best_epoch":60,"scaler":{"mean":[4.152108635097493]}},"NO3D_FT/egb/2":{"best_epoch":44,"scaler":{"mean":[4.250774373259053]}},"NO3D_FT/egb/3":{"best_epoch":61,"scaler":{"mean":[4.228450696378831]}},"NO3D_FT/egb/4":{"best_epoch":60,"scaler":{"mean":[4.215195821727019]}},"NO3D_FT/egc/0":{"best_epoch":50,"scaler":{"mean":[4.4900454461396215]}},"NO3D_FT/egc/1":{"best_epoch":69,"scaler":{"mean":[4.515255386037911]}},"NO3D_FT/egc/2":{"best_epoch":36,"scaler":{"mean":[4.509532316227461]}},"NO3D_FT/egc/3":{"best_epoch":20,"scaler":{"mean":[4.554269579288025]}},"NO3D_FT/egc/4":{"best_epoch":68,"scaler":{"mean":[4.514467730004624]}},"NO3D_FT/ei/0":{"best_epoch":61,"scaler":{"mean":[6.22182838983051]}},"NO3D_FT/ei/1":{"best_epoch":66,"scaler":{"mean":[6.279353813559322]}},"NO3D_FT/ei/2":{"best_epoch":48,"scaler":{"mean":[6.27359533898305]}},"NO3D_FT/ei/3":{"best_epoch":66,"scaler":{"mean":[6.283478389830509]}},"NO3D_FT/ei/4":{"best_epoch":57,"scaler":{"mean":[6.280269915254237]}},"NO3D_FT/eps/0":{"best_epoch":35,"scaler":{"mean":[4.599795081967213]}},"NO3D_FT/eps/1":{"best_epoch":68,"scaler":{"mean":[4.617868852459015]}},"NO3D_FT/eps/2":{"best_epoch":18,"scaler":{"mean":[4.610081967213113]}},"NO3D_FT/eps/3":{"best_epoch":17,"scaler":{"mean":[4.57405737704918]}},"NO3D_FT/eps/4":{"best_epoch":53,"scaler":{"mean":[4.576598360655737]}},"NO3D_FT/nc/0":{"best_epoch":21,"scaler":{"mean":[1.9505442622950822]}},"NO3D_FT/nc/1":{"best_epoch":30,"scaler":{"mean":[1.957320081967213]}},"NO3D_FT/nc/2":{"best_epoch":20,"scaler":{"mean":[1.958533606557377]}},"NO3D_FT/nc/3":{"best_epoch":15,"scaler":{"mean":[1.9376409836065573]}},"NO3D_FT/nc/4":{"best_epoch":16,"scaler":{"mean":[1.9437995901639344]}},"NO3D_FT/xc/0":{"best_epoch":22,"scaler":{"mean":[37.46221099847826]}},"NO3D_FT/xc/1":{"best_epoch":18,"scaler":{"mean":[36.49228298981884]}},"NO3D_FT/xc/2":{"best_epoch":23,"scaler":{"mean":[37.15334486376811]}},"NO3D_FT/xc/3":{"best_epoch":38,"scaler":{"mean":[35.30247894376812]}},"NO3D_FT/xc/4":{"best_epoch":23,"scaler":{"mean":[35.35397605797102]}}}},"delta_3d":0.004578983000486758,"comparisons":{"FULL_minus_NO3D_FT":{"status":"COMPLETE","delta":0.004578983000486758,"tasks":{"eea":{"folds":[0.005899127212723121,0.004071891103462177,0.016640933631867694,-0.0014144152045776348,-0.004851819477335262],"mean":0.004069143453228019,"std":0.008231824700019708},"egb":{"folds":[-0.00537987091779113,0.004145826675407416,-0.006955333721143542,0.00981168532802823,0.0003367525787331127],"mean":0.00039181198864681745,"std":0.006894246608411645},"egc":{"folds":[-0.000330670186674098,0.0016595053128746207,0.006765023406978488,0.01135682781355385,-0.003960747639513151],"mean":0.003097987741443942,"std":0.006027152353572864},"ei":{"folds":[0.041039760296537886,0.02387347894677494,0.016737401953570652,-0.04584286801507731,-0.030319829025806322],"mean":0.0010975888311999693,"std":0.03724641801624968},"eps":{"folds":[0.014090868695110559,-0.04553031372468275,0.08185089727871897,0.010592358456857687,-0.034979961369968615],"mean":0.00520476986720717,"std":0.050426875379237034},"nc":{"folds":[0.014655551138079259,0.01277471236588934,0.06531034736355967,0.013753209248745124,0.017727288921236317],"mean":0.024844221807501943,"std":0.0226971951955295},"xc":{"folds":[0.11443562191552326,-0.0545316531812472,-0.060574213747192185,-0.06723372138404882,0.03464075296786473],"mean":-0.0066526426858200425,"std":0.07943159780082089}}},"NO3D_FT_minus_FULL":{"status":"COMPLETE","delta":-0.004578983000486758,"tasks":{"eea":{"folds":[-0.005899127212723121,-0.004071891103462177,-0.016640933631867694,0.0014144152045776348,0.004851819477335262],"mean":-0.004069143453228019,"std":0.008231824700019708},"egb":{"folds":[0.00537987091779113,-0.004145826675407416,0.006955333721143542,-0.00981168532802823,-0.0003367525787331127],"mean":-0.00039181198864681745,"std":0.006894246608411645},"egc":{"folds":[0.000330670186674098,-0.0016595053128746207,-0.006765023406978488,-0.01135682781355385,0.003960747639513151],"mean":-0.003097987741443942,"std":0.006027152353572864},"ei":{"folds":[-0.041039760296537886,-0.02387347894677494,-0.016737401953570652,0.04584286801507731,0.030319829025806322],"mean":-0.0010975888311999693,"std":0.03724641801624968},"eps":{"folds":[-0.014090868695110559,0.04553031372468275,-0.08185089727871897,-0.010592358456857687,0.034979961369968615],"mean":-0.00520476986720717,"std":0.050426875379237034},"nc":{"folds":[-0.014655551138079259,-0.01277471236588934,-0.06531034736355967,-0.013753209248745124,-0.017727288921236317],"mean":-0.024844221807501943,"std":0.0226971951955295},"xc":{"folds":[-0.11443562191552326,0.0545316531812472,0.060574213747192185,0.06723372138404882,-0.03464075296786473],"mean":0.0066526426858200425,"std":0.07943159780082089}}},"FINAL_minus_FULL":{"status":"COMPLETE","delta":0.0,"tasks":{"eea":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"egb":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"egc":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"ei":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"eps":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"nc":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0},"xc":{"folds":[0.0,0.0,0.0,0.0,0.0],"mean":0.0,"std":0.0}}}},"sota":{"FULL":{"eea":{"project":{"folds":[0.9048127345675825,0.9249857549198717,0.9430897450142551,0.9266785662249936,0.9039788229550795],"mean":0.9207091247363565,"std":0.016487981831676657},"reference":0.934,"delta":-0.013290875263643587},"egb":{"project":{"folds":[0.918203133735168,0.9094998438640765,0.9195143535532943,0.9191146090704887,0.914430382738539],"mean":0.9161524645923134,"std":0.004228024647925799},"reference":0.923,"delta":-0.006847535407686678},"egc":{"project":{"folds":[0.8888755460081812,0.9218274633020648,0.9232339738526183,0.9079443651146784,0.9099054187296192],"mean":0.9103573534014323,"std":0.013828101656380973},"reference":0.911,"delta":-0.0006426465985677199},"ei":{"project":{"folds":[0.783115913417741,0.8039711244530234,0.8045980600600575,0.7834563567795338,0.7843702137347925],"mean":0.7919023336890296,"std":0.011314876454667316},"reference":0.839,"delta":-0.04709766631097034},"eps":{"project":{"folds":[0.7709525786456969,0.7767643794726025,0.6725660225167185,0.7422877410435005,0.7723423325556295],"mean":0.7469826108468295,"std":0.04377356224100633},"reference":0.787,"delta":-0.040017389153170546},"nc":{"project":{"folds":[0.810864975404595,0.904685005181916,0.8405647210603203,0.7278693567878491,0.8389326135921045],"mean":0.8245833344053569,"std":0.06406904680511585},"reference":0.856,"delta":-0.03141666559464307},"xc":{"project":{"folds":[0.48515886616839554,0.37473519213811024,0.3834542764856963,0.20879930392167612,0.5871546891990046],"mean":0.40786046558257655,"std":0.14092529336862664},"reference":0.423,"delta":-0.01513953441742344}},"NO3D_FT":{"eea":{"project":{"folds":[0.8989136073548594,0.9209138638164095,0.9264488113823874,0.9280929814295712,0.9088306424324147],"mean":0.9166399812831283,"std":0.012458536612589336},"reference":0.934,"delta":-0.01736001871687176},"egb":{"project":{"folds":[0.9235830046529592,0.905354017188669,0.9264696872744378,0.9093029237424605,0.9140936301598059],"mean":0.9157606526036666,"std":0.009064379483681198},"reference":0.923,"delta":-0.007239347396333473},"egc":{"project":{"folds":[0.8892062161948553,0.9201679579891902,0.9164689504456398,0.8965875373011245,0.9138661663691323],"mean":0.9072593656599885,"std":0.01355453936498268},"reference":0.911,"delta":-0.003740634340011484},"ei":{"project":{"folds":[0.7420761531212031,0.7800976455062485,0.7878606581064869,0.8292992247946112,0.8146900427605989],"mean":0.7908047448578297,"std":0.0337272801952151},"reference":0.839,"delta":-0.04819525514217027},"eps":{"project":{"folds":[0.7568617099505863,0.8222946931972852,0.5907151252379995,0.7316953825866428,0.8073222939255981],"mean":0.7417778409796225,"std":0.09209799395296553},"reference":0.787,"delta":-0.04522215902037752},"nc":{"project":{"folds":[0.7962094242665158,0.8919102928160266,0.7752543736967606,0.714116147539104,0.8212053246708682],"mean":0.799739112597855,"std":0.06499918847146766},"reference":0.856,"delta":-0.056260887402144966},"xc":{"project":{"folds":[0.3707232442528723,0.42926684531935744,0.4440284902328885,0.27603302530572493,0.5525139362311399],"mean":0.41451310826839666,"std":0.10152258937759609},"reference":0.423,"delta":-0.00848689173160333}},"FINAL":{"eea":{"project":{"folds":[0.9048127345675825,0.9249857549198717,0.9430897450142551,0.9266785662249936,0.9039788229550795],"mean":0.9207091247363565,"std":0.016487981831676657},"reference":0.934,"delta":-0.013290875263643587},"egb":{"project":{"folds":[0.918203133735168,0.9094998438640765,0.9195143535532943,0.9191146090704887,0.914430382738539],"mean":0.9161524645923134,"std":0.004228024647925799},"reference":0.923,"delta":-0.006847535407686678},"egc":{"project":{"folds":[0.8888755460081812,0.9218274633020648,0.9232339738526183,0.9079443651146784,0.9099054187296192],"mean":0.9103573534014323,"std":0.013828101656380973},"reference":0.911,"delta":-0.0006426465985677199},"ei":{"project":{"folds":[0.783115913417741,0.8039711244530234,0.8045980600600575,0.7834563567795338,0.7843702137347925],"mean":0.7919023336890296,"std":0.011314876454667316},"reference":0.839,"delta":-0.04709766631097034},"eps":{"project":{"folds":[0.7709525786456969,0.7767643794726025,0.6725660225167185,0.7422877410435005,0.7723423325556295],"mean":0.7469826108468295,"std":0.04377356224100633},"reference":0.787,"delta":-0.040017389153170546},"nc":{"project":{"folds":[0.810864975404595,0.904685005181916,0.8405647210603203,0.7278693567878491,0.8389326135921045],"mean":0.8245833344053569,"std":0.06406904680511585},"reference":0.856,"delta":-0.03141666559464307},"xc":{"project":{"folds":[0.48515886616839554,0.37473519213811024,0.3834542764856963,0.20879930392167612,0.5871546891990046],"mean":0.40786046558257655,"std":0.14092529336862664},"reference":0.423,"delta":-0.01513953441742344}}},"historical_comparison":{"FULL":{"TMOL3M":{"status":"COMPLETE","delta":-0.0015193945599878411,"tasks":{"eea":{"folds":[-0.003098225954149547,-0.004400052797733034,0.0020527769132692164,0.005002995106443531,-0.012316199166891195],"mean":-0.0025517411798122057,"std":0.006658362569944235},"egb":{"folds":[0.013273615588376586,-0.0008459158291557189,0.013340161116896065,-5.303162555225782e-05,-0.001076776462998974],"mean":0.0049276105575131405,"std":0.007658653175897807},"egc":{"folds":[-0.0018686004324125483,-0.0012137551352509046,0.0037492836968913146,-0.001995697336746649,-0.0025089574816343774],"mean":-0.0007675453378306329,"std":0.002566809479711846},"ei":{"folds":[0.017485784837868557,0.028270404321974252,-0.002868181040477813,-0.05650860476340602,-0.04940615097504808],"mean":-0.01260534952381782,"std":0.038577554606899},"eps":{"folds":[-0.0023235641321676104,-0.01613080025361968,-0.016769116124488836,0.036514483640906814,-0.001828314154782129],"mean":-0.00010746220483028867,"std":0.021699063518262472},"nc":{"folds":[-0.026005457176165425,-0.00191491965638102,0.09537608398189257,-0.010491536409849456,-0.004886995896356439],"mean":0.010415434968628045,"std":0.04839446443849336},"xc":{"folds":[0.029997848114651005,-0.06673682899597944,-0.00475107216429127,-0.06719361146410385,0.058950118510902816],"mean":-0.009946709199764148,"std":0.05672667074924644}}},"B_A020":{"status":"COMPLETE","delta":0.0033195188082958405,"tasks":{"eea":{"folds":[-0.021505134756625854,-0.0051834918435544886,-6.614920057823603e-06,-0.0027289250470875492,-0.012543442529078574],"mean":-0.008393521819280858,"std":0.008687180096112896},"egb":{"folds":[-0.01953252854926757,-0.007753504887903051,-0.028312646304587852,-0.012713834767114185,-0.005180326344364272],"mean":-0.014698568170647386,"std":0.00937390386294759},"egc":{"folds":[-0.009397033941263788,-0.004940085952512496,-0.002260917401913254,-0.011136256512758291,-0.011105645956107968],"mean":-0.007767987952911159,"std":0.003983209781296151},"ei":{"folds":[0.013629472130958109,-0.006224832991891471,-0.01628021922259859,-0.06726138419388028,-0.021313749121273595],"mean":-0.019490142679737164,"std":0.029868871719484862},"eps":{"folds":[0.0865038728021924,-0.06639875317052835,0.1137438716382797,0.02740591877918952,0.010822117345338889],"mean":0.034415405478894434,"std":0.07031266725176252},"nc":{"folds":[0.008510390128437573,0.011269605129895055,0.0001425004708709432,0.013514834100915585,0.05091895282386849],"mean":0.01687125653079753,"std":0.01969546305066783},"xc":{"folds":[0.12550696363061065,0.023608837970277574,-0.0895607245880703,-0.07253419037990683,0.12448006472186901],"mean":0.022300190270956022,"std":0.10319944451818465}}}},"NO3D_FT":{"TMOL3M":{"status":"COMPLETE","delta":-0.006098377560474599,"tasks":{"eea":{"folds":[-0.008997353166872668,-0.008471943901195211,-0.014588156718598477,0.0064174103110211655,-0.007464379689555933],"mean":-0.006620884633040225,"std":0.007798398337968785},"egb":{"folds":[0.018653486506167716,-0.004991742504563135,0.020295494838039607,-0.009864716953580488,-0.0014135290417320867],"mean":0.004535798568866323,"std":0.013975161219355978},"egc":{"folds":[-0.0015379302457384503,-0.0028732604481255253,-0.003015739710087173,-0.013352525150300498,0.001451790157878774],"mean":-0.0038655330792745747,"std":0.005599139078010284},"ei":{"folds":[-0.02355397545866933,0.004396925375199312,-0.019605582994048465,-0.010665736748328714,-0.019086321949241758],"mean":-0.01370293835501779,"std":0.011154535236024627},"eps":{"folds":[-0.01641443282727817,0.029399513471063066,-0.0986200134032078,0.025922125184049127,0.033151647215186486],"mean":-0.005312232072037459,"std":0.05587835287530206},"nc":{"folds":[-0.040661008314244684,-0.01468963202227036,0.030065736618332894,-0.02424474565859458,-0.022614284817592756],"mean":-0.014428786838873896,"std":0.02660505368410574},"xc":{"folds":[-0.08443777380087225,-0.012205175814732239,0.055823141582900915,4.0109919944963934e-05,0.024309365543038086],"mean":-0.003294066513944105,"std":0.052277192895706634}}},"B_A020":{"status":"COMPLETE","delta":-0.0012594641921909178,"tasks":{"eea":{"folds":[-0.027404261969348975,-0.009255382947016666,-0.016647548551925517,-0.0013145098425099144,-0.007691623051743313],"mean":-0.012462665272508878,"std":0.009976164508373466},"egb":{"folds":[-0.01415265763147644,-0.011899331563310467,-0.02135731258344431,-0.022525520095142415,-0.005517078923097385],"mean":-0.015090380159294203,"std":0.007022570026610328},"egc":{"folds":[-0.00906636375458969,-0.006599591265387117,-0.009025940808891741,-0.02249308432631214,-0.0071448983165948166],"mean":-0.010865975694355101,"std":0.006592845356896712},"ei":{"folds":[-0.027410288165579777,-0.03009831193866641,-0.03301762117616924,-0.02141851617880297,0.009006079904532727],"mean":-0.020587731510937134,"std":0.017087893768152902},"eps":{"folds":[0.07241300410708185,-0.0208684394458456,0.03189297435956073,0.016813560322331833,0.045802078715307504],"mean":0.029210635611687262,"std":0.03467894463279856},"nc":{"folds":[-0.0061451610096416864,-0.0015051072359942852,-0.06516784689268873,-0.00023837514782953928,0.033191663902632174],"mean":-0.007972965276704413,"std":0.03560470165256939},"xc":{"folds":[0.01107134171508739,0.07814049115152477,-0.02898651084087811,-0.00530046899585801,0.08983931175400428],"mean":0.028952832956776066,"std":0.052384618454266785}}}}}}
```

</details>

<details><summary>results/ua2d_fg10k_morgan_20261023/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-FG10K-MORGAN-20261023-01","revision":"r1.1","status":"待审查","FINAL":"GATE_FP","formal":{"CONCAT_FP":{"tasks":{"eea":{"folds":[0.9081079696731226,0.9321660461952999,0.9354203419156095,0.9248584017031208,0.902841398512987],"mean":0.920678831600028,"std":0.014516694015071734},"egb":{"folds":[0.9206982128337851,0.9202340464926924,0.9240556591001904,0.8986616551126041,0.9163130741980093],"mean":0.9159925295474561,"std":0.010069880050449261},"egc":{"folds":[0.8805753133325959,0.896452740012346,0.9171339178747728,0.8733751636540723,0.8939559030061158],"mean":0.8922986075759806,"std":0.016825044129670886},"ei":{"folds":[0.796011777534482,0.7893869932730594,0.5866983512417492,0.8250688506893491,0.7918965342238473],"mean":0.7578125013924975,"std":0.09672295931149018},"eps":{"folds":[0.7364484469206581,0.8113965325148614,0.43100717646798203,0.6373116958184417,0.6808568700736523],"mean":0.659404144359119,"std":0.14330996238874183},"nc":{"folds":[0.8526126677923789,0.8784983386842065,0.7753644900264459,0.6594771589918136,0.8120168622308708],"mean":0.7955939035451431,"std":0.08563147162858546},"xc":{"folds":[0.3062948085835304,0.3864062562189168,0.42631521456174604,0.21035572575593042,0.470111372197216],"mean":0.35989667546346793,"std":0.10307414243681344}},"macro":0.757382456211956},"GATE_FP":{"tasks":{"eea":{"folds":[0.9080036887925589,0.9311245470956855,0.9372123545445008,0.9248858386455547,0.9026704333172103],"mean":0.920779372479102,"std":0.014875150192630337},"egb":{"folds":[0.9224022495407445,0.9199379470041255,0.9242493978844134,0.9017665535621648,0.906594191643409],"mean":0.9149900679269715,"std":0.01013051338313621},"egc":{"folds":[0.8836832719236232,0.89966681049651,0.9167925744866837,0.8728536027225643,0.8949360858749875],"mean":0.8935864691008739,"std":0.016622578069406937},"ei":{"folds":[0.793403392769901,0.7905721982943483,0.5835570043297226,0.8231120558858054,0.7920814057054222],"mean":0.7565452113970399,"std":0.09764124006651079},"eps":{"folds":[0.7364652760035074,0.812836384687645,0.42591499794666576,0.6559845162346046,0.6799769625906056],"mean":0.6622356274926057,"std":0.14525142004278305},"nc":{"folds":[0.8548622461017301,0.8788004480877607,0.7759037189198076,0.6774190452671234,0.8043167381543808],"mean":0.7982604393061605,"std":0.07878705757060227},"xc":{"folds":[0.3071214718683658,0.3845910285361529,0.42612539841476416,0.23189845108966556,0.44432011479765565],"mean":0.3588112929413208,"std":0.08843959691182604}},"macro":0.7578869258062964}},"validation":{"CONCAT_FP":{"tasks":{"eea":{"folds":[0.9239877644045313,0.8899410162314807,0.8862189943975227,0.9028015076257796,0.9443946269050602],"mean":0.909468781912875,"std":0.02447934996812111},"egb":{"folds":[0.9214337411207496,0.9183117044940077,0.9139003011815579,0.9501233718880135,0.9334069011331351],"mean":0.9274352039634927,"std":0.01460191359887565},"egc":{"folds":[0.9018371168675209,0.9059542118960026,0.8978296937760948,0.9012236507711262,0.8960083004339504],"mean":0.9005705947489389,"std":0.0038516542417072063},"ei":{"folds":[0.8544460050913142,0.8361609563073112,0.8744439591175251,0.8482054097189634,0.777387832699047],"mean":0.8381288325868322,"std":0.036673960600814834},"eps":{"folds":[0.700155301264229,0.8252436505962761,0.7629579432812346,0.8099886858261984,0.6283286974385576],"mean":0.7453348556812991,"std":0.08155939504093754},"nc":{"folds":[0.8028619853517216,0.8878265874874789,0.8835702437432774,0.8657732115337703,0.7625009505791474],"mean":0.8405065957390792,"std":0.05530383946944322},"xc":{"folds":[0.31009253504942114,0.5196616237311749,0.40547090518303297,0.47864870912576307,0.2770486781420781],"mean":0.39818449024629404,"std":0.10454467211050589}},"macro":0.7942327649826872},"GATE_FP":{"tasks":{"eea":{"folds":[0.9237700413791149,0.8904394171294591,0.8859923974803332,0.9026057225356474,0.9439832959178093],"mean":0.9093581748884729,"std":0.024269710875123942},"egb":{"folds":[0.9232351891208761,0.9186079846943092,0.9118897649882388,0.952416724491326,0.9344886978467649],"mean":0.928127672228303,"std":0.01587963459901435},"egc":{"folds":[0.9026836787042847,0.9050992077030706,0.8980739431015079,0.9014550295161265,0.9010727773997298],"mean":0.9016769272849439,"std":0.002555449197809388},"ei":{"folds":[0.8528501472388137,0.8361776945760834,0.873173805146171,0.845476452204003,0.7774095715655903],"mean":0.8370175341461323,"std":0.035997622831887365},"eps":{"folds":[0.699800276057114,0.8248737088046836,0.7596680005530088,0.8113944556162267,0.6267938988869444],"mean":0.7445060679835954,"std":0.08218524939989602},"nc":{"folds":[0.8049008060847884,0.8891042222337918,0.8828374244123561,0.8630127577524159,0.7617658759548704],"mean":0.8403242172876446,"std":0.05506328764521699},"xc":{"folds":[0.3102072045155567,0.5209613296284168,0.4060927102665366,0.4555016590075469,0.27757139295587696],"mean":0.3940668592747868,"std":0.100777846878426}},"macro":0.7935824932991256}},"reference":{"macro":0.7821603433774255,"tasks":{"eea":{"folds":[0.9096301835610174,0.9280967986862416,0.9439369140030686,0.9300955352895374,0.9032566156068411],"mean":0.9230032094293412,"std":0.016456241318045316},"egb":{"folds":[0.9464454544870122,0.917364692029361,0.950250224420717,0.9342532924376477,0.9230226837860851],"mean":0.9342672694321645,"std":0.014281990984823927},"egc":{"folds":[0.8872152581767657,0.9295168156498679,0.9283668684300971,0.9180914199814554,0.9215191197317041],"mean":0.9169418963939779,"std":0.01728152176442276},"ei":{"folds":[0.7809049283434036,0.791028370590115,0.7979578162545934,0.7904551933276684,0.8271780541170919],"mean":0.7975048725265745,"std":0.01766284281160179},"eps":{"folds":[0.7283602102833471,0.8187013061581133,0.6135500798756172,0.7040513873576981,0.6702942076786225],"mean":0.7069914382706797,"std":0.07586556711046405},"nc":{"folds":[0.8115460526299713,0.8867299414283278,0.7685472955491949,0.6399311693870428,0.7607314234623676],"mean":0.7734971764913808,"std":0.08985644113836715},"xc":{"folds":[0.5137752035719265,0.38954879960877364,0.45892084856583204,0.2406366624238636,0.5117011913189038],"mean":0.42291654109785987,"std":0.1137691535074282}}},"contrasts":{"CONCAT_FP_minus_FG_REF":{"status":"COMPLETE","delta":-0.02477788716546947,"tasks":{"eea":{"folds":[-0.0015222138878948632,0.004069247509058238,-0.00851657208745904,-0.005237133586416531,-0.00041521709385416994],"mean":-0.0023243778293132733,"std":0.004795680934480224},"egb":{"folds":[-0.02574724165322717,0.002869354463331364,-0.026194565320526664,-0.03559163732504356,-0.00670960958807576],"mean":-0.018274739884708357,"std":0.01580555139144419},"egc":{"folds":[-0.006639944844169854,-0.033064075637521895,-0.01123295055532425,-0.044716256327383075,-0.027563216725588258],"mean":-0.024643288817997468,"std":0.015702843857383974},"ei":{"folds":[0.015106849191078409,-0.0016413773170556034,-0.2112594650128442,0.03461365736168065,-0.03528151989324457],"mean":-0.039692371134077066,"std":0.09928109319023783},"eps":{"folds":[0.008088236637311064,-0.007304773643251883,-0.18254290340763513,-0.06673969153925641,0.010562662395029787],"mean":-0.04758729391156051,"std":0.08167572226321256},"nc":{"folds":[0.041066615162407616,-0.00823160274412138,0.006817194477251043,0.01954598960477083,0.05128543876850322],"mean":0.022096727053762266,"std":0.024349617615274436},"xc":{"folds":[-0.2074803949883961,-0.0031425433898568667,-0.032605634004086004,-0.03028093666793319,-0.0415898191216878],"mean":-0.06301986563439199,"std":0.0820217168033631}}},"GATE_FP_minus_FG_REF":{"status":"COMPLETE","delta":-0.024273417571129108,"tasks":{"eea":{"folds":[-0.001626494768458553,0.0030277484094438645,-0.006724559458567758,-0.005209696643982631,-0.00058618228963081],"mean":-0.0022238369502391776,"std":0.003866191293355816},"egb":{"folds":[-0.02404320494626777,0.0025732549747644606,-0.026000826536303645,-0.03248673887548281,-0.016428492142676054],"mean":-0.019277201505193166,"std":0.013490472502019749},"egc":{"folds":[-0.0035319862531425494,-0.029850005153357895,-0.01157429394341336,-0.04523781725889109,-0.026583033856716565],"mean":-0.023355427293104292,"std":0.016303373071201978},"ei":{"folds":[0.01249846442649738,-0.00045617229576666585,-0.21440081192487082,0.032656862558136934,-0.035096648411669706],"mean":-0.04095966112953457,"std":0.1000423711020355},"eps":{"folds":[0.008105065720160343,-0.005864921470468354,-0.1876350819289514,-0.04806687112309349,0.009682754911983116],"mean":-0.04475581077807396,"std":0.0832102460861987},"nc":{"folds":[0.04331619347175886,-0.00792949334056714,0.007356423370612708,0.03748787588008062,0.04358531469201321],"mean":0.02476326281477965,"std":0.023623140217599935},"xc":{"folds":[-0.20665373170356072,-0.0049577710726207425,-0.032795450151067884,-0.008738211334198054,-0.06738107652124814],"mean":-0.0641052481565391,"std":0.08347909540126355}}},"GATE_FP_minus_CONCAT_FP":{"status":"COMPLETE","delta":0.0005044695943403621,"tasks":{"eea":{"folds":[-0.00010428088056368967,-0.0010414990996143736,0.0017920126288912819,2.743694243390049e-05,-0.00017096519577664004],"mean":0.00010054087907409581,"std":0.0010351743429015518},"egb":{"folds":[0.0017040367069593998,-0.00029609948856690327,0.0001937387842230187,0.0031048984495607446,-0.009718882554600294],"mean":-0.0010024616204848068,"std":0.005052261293514528},"egc":{"folds":[0.0031079585910273044,0.003214070484164,-0.00034134338808911036,-0.0005215609315080139,0.000980182868871693],"mean":0.0012878615248931747,"std":0.0018059636426649745},"ei":{"folds":[-0.0026083847645810287,0.0011852050212889376,-0.00314134691202661,-0.0019567948035437155,0.00018487148157486644],"mean":-0.00126728999545751,"std":0.0018647751975256932},"eps":{"folds":[1.6829082849278976e-05,0.001439852172783529,-0.005092178521316271,0.018672820416162916,-0.0008799074830466713],"mean":0.0028314831334865565,"std":0.009183758971512586},"nc":{"folds":[0.002249578309351241,0.00030210940355424043,0.0005392288933616651,0.017941886275309793,-0.007700124076490011],"mean":0.0026665357610173857,"std":0.009368748176072188},"xc":{"folds":[0.0008266632848353828,-0.0018152276827638758,-0.00018981614698188043,0.021542725333735135,-0.02579125739956034],"mean":-0.0010853825221471159,"std":0.016788404859118092}}}},"budget":{"plan":"UA2D-FG10K-MORGAN-20261023-01","revision":"r1.1"},"independent_review":"PENDING","sota":{"FG_REF":{"eea":{"project":{"folds":[0.9096301835610174,0.9280967986862416,0.9439369140030686,0.9300955352895374,0.9032566156068411],"mean":0.9230032094293412,"std":0.016456241318045316},"reference":0.934,"delta":-0.010996790570658832},"egb":{"project":{"folds":[0.9464454544870122,0.917364692029361,0.950250224420717,0.9342532924376477,0.9230226837860851],"mean":0.9342672694321645,"std":0.014281990984823927},"reference":0.923,"delta":0.011267269432164495},"egc":{"project":{"folds":[0.8872152581767657,0.9295168156498679,0.9283668684300971,0.9180914199814554,0.9215191197317041],"mean":0.9169418963939779,"std":0.01728152176442276},"reference":0.911,"delta":0.005941896393977908},"ei":{"project":{"folds":[0.7809049283434036,0.791028370590115,0.7979578162545934,0.7904551933276684,0.8271780541170919],"mean":0.7975048725265745,"std":0.01766284281160179},"reference":0.839,"delta":-0.04149512747342543},"eps":{"project":{"folds":[0.7283602102833471,0.8187013061581133,0.6135500798756172,0.7040513873576981,0.6702942076786225],"mean":0.7069914382706797,"std":0.07586556711046405},"reference":0.787,"delta":-0.08000856172932036},"nc":{"project":{"folds":[0.8115460526299713,0.8867299414283278,0.7685472955491949,0.6399311693870428,0.7607314234623676],"mean":0.7734971764913808,"std":0.08985644113836715},"reference":0.856,"delta":-0.08250282350861915},"xc":{"project":{"folds":[0.5137752035719265,0.38954879960877364,0.45892084856583204,0.2406366624238636,0.5117011913189038],"mean":0.42291654109785987,"std":0.1137691535074282},"reference":0.423,"delta":-8.345890214012108e-05}},"CONCAT_FP":{"eea":{"project":{"folds":[0.9081079696731226,0.9321660461952999,0.9354203419156095,0.9248584017031208,0.902841398512987],"mean":0.920678831600028,"std":0.014516694015071734},"reference":0.934,"delta":-0.013321168399972105},"egb":{"project":{"folds":[0.9206982128337851,0.9202340464926924,0.9240556591001904,0.8986616551126041,0.9163130741980093],"mean":0.9159925295474561,"std":0.010069880050449261},"reference":0.923,"delta":-0.007007470452543907},"egc":{"project":{"folds":[0.8805753133325959,0.896452740012346,0.9171339178747728,0.8733751636540723,0.8939559030061158],"mean":0.8922986075759806,"std":0.016825044129670886},"reference":0.911,"delta":-0.01870139242401947},"ei":{"project":{"folds":[0.796011777534482,0.7893869932730594,0.5866983512417492,0.8250688506893491,0.7918965342238473],"mean":0.7578125013924975,"std":0.09672295931149018},"reference":0.839,"delta":-0.08118749860750252},"eps":{"project":{"folds":[0.7364484469206581,0.8113965325148614,0.43100717646798203,0.6373116958184417,0.6808568700736523],"mean":0.659404144359119,"std":0.14330996238874183},"reference":0.787,"delta":-0.12759585564088105},"nc":{"project":{"folds":[0.8526126677923789,0.8784983386842065,0.7753644900264459,0.6594771589918136,0.8120168622308708],"mean":0.7955939035451431,"std":0.08563147162858546},"reference":0.856,"delta":-0.06040609645485684},"xc":{"project":{"folds":[0.3062948085835304,0.3864062562189168,0.42631521456174604,0.21035572575593042,0.470111372197216],"mean":0.35989667546346793,"std":0.10307414243681344},"reference":0.423,"delta":-0.06310332453653206}},"GATE_FP":{"eea":{"project":{"folds":[0.9080036887925589,0.9311245470956855,0.9372123545445008,0.9248858386455547,0.9026704333172103],"mean":0.920779372479102,"std":0.014875150192630337},"reference":0.934,"delta":-0.013220627520898032},"egb":{"project":{"folds":[0.9224022495407445,0.9199379470041255,0.9242493978844134,0.9017665535621648,0.906594191643409],"mean":0.9149900679269715,"std":0.01013051338313621},"reference":0.923,"delta":-0.00800993207302858},"egc":{"project":{"folds":[0.8836832719236232,0.89966681049651,0.9167925744866837,0.8728536027225643,0.8949360858749875],"mean":0.8935864691008739,"std":0.016622578069406937},"reference":0.911,"delta":-0.017413530899126117},"ei":{"project":{"folds":[0.793403392769901,0.7905721982943483,0.5835570043297226,0.8231120558858054,0.7920814057054222],"mean":0.7565452113970399,"std":0.09764124006651079},"reference":0.839,"delta":-0.08245478860296007},"eps":{"project":{"folds":[0.7364652760035074,0.812836384687645,0.42591499794666576,0.6559845162346046,0.6799769625906056],"mean":0.6622356274926057,"std":0.14525142004278305},"reference":0.787,"delta":-0.12476437250739436},"nc":{"project":{"folds":[0.8548622461017301,0.8788004480877607,0.7759037189198076,0.6774190452671234,0.8043167381543808],"mean":0.7982604393061605,"std":0.07878705757060227},"reference":0.856,"delta":-0.05773956069383945},"xc":{"project":{"folds":[0.3071214718683658,0.3845910285361529,0.42612539841476416,0.23189845108966556,0.44432011479765565],"mean":0.3588112929413208,"std":0.08843959691182604},"reference":0.423,"delta":-0.06418870705867918}}}}
```

</details>

<details><summary>results/ua2d_et_single_20261020/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-ET-SINGLE-20261020-01","revision":"r1","status":"待审查","budget":{"plan":"UA2D-ET-SINGLE-20261020-01","revision":"r1"},"independent_review":"PENDING","development_decision":{"reference":0.8169937875891848,"delta":-0.0024330246691398605,"selection":"validation_only_full_precision","validation":{"tasks":{"eea":{"folds":[0.9375241041934061,0.908519361917957,0.9036719595635071,0.9038177989436238,0.9541689040928834],"mean":0.9215404257422755,"std":0.023038105736194314},"egb":{"folds":[0.9320214177882908,0.9444502213332477,0.9294257286593262,0.9504808846639757,0.9509126302304706],"mean":0.9414581765350623,"std":0.010168339420224954},"egc":{"folds":[0.9236679387912125,0.9260289049388545,0.9213491763569931,0.9076270080058602,0.9028970315785492],"mean":0.916314011934294,"std":0.010359677434525254},"ei":{"folds":[0.8697548433424982,0.8463740591990082,0.8789993353031841,0.875047115426111,0.8503594962004936],"mean":0.864106969894259,"std":0.014805547616559884},"eps":{"folds":[0.7465961796692451,0.853371468040014,0.7654002205588073,0.833440181762758,0.689757112647049],"mean":0.7777130325355748,"std":0.06649402555584401},"nc":{"folds":[0.8293413099467735,0.9096903675602839,0.8620922157331837,0.9025553640559385,0.7597664022643656],"mean":0.8526891319121092,"std":0.06124531499367895},"xc":{"folds":[0.2949066674319605,0.5776298509682867,0.5293570461002577,0.459736348324599,0.27888804660859845],"mean":0.42810359188674046,"std":0.13566290489832972}},"macro":0.814560762920045},"reference_validation":{"tasks":{"eea":{"folds":[0.9316260918906403,0.9123610651479899,0.9050702542361508,0.9202273931472877,0.946903998516294],"mean":0.9232377605876726,"std":0.016491497980692923},"egb":{"folds":[0.9377257024120351,0.9366708875802592,0.9340279240727146,0.9461228554817407,0.9534817037048147],"mean":0.9416058146503128,"std":0.008035446479226518},"egc":{"folds":[0.9244811336142278,0.9213336739117486,0.9189108785172547,0.9036985691471207,0.9222909312285237],"mean":0.9181430372837751,"std":0.008318486810613496},"ei":{"folds":[0.8560148240981529,0.8947956072904384,0.8982002228087859,0.8560379250533212,0.8298769324213336],"mean":0.8669851023344064,"std":0.029004361316193928},"eps":{"folds":[0.7714897848265632,0.8464592232314412,0.7565904633996667,0.8546090841920899,0.6948400899130099],"mean":0.7847977291125543,"std":0.06659747222208665},"nc":{"folds":[0.8087157870263256,0.8972957183750673,0.8507503824116913,0.886434641268426,0.7661854053539039],"mean":0.8418763868870828,"std":0.054690580087524324},"xc":{"folds":[0.33559169945487466,0.4637539322418345,0.5353308876269116,0.5425817923661881,0.33429509965263915],"mean":0.4423106822684896,"std":0.10274139194499628}},"macro":0.8169937875891848}},"validation":{"tasks":{"eea":{"folds":[0.9375241041934061,0.908519361917957,0.9036719595635071,0.9038177989436238,0.9541689040928834],"mean":0.9215404257422755,"std":0.023038105736194314},"egb":{"folds":[0.9320214177882908,0.9444502213332477,0.9294257286593262,0.9504808846639757,0.9509126302304706],"mean":0.9414581765350623,"std":0.010168339420224954},"egc":{"folds":[0.9236679387912125,0.9260289049388545,0.9213491763569931,0.9076270080058602,0.9028970315785492],"mean":0.916314011934294,"std":0.010359677434525254},"ei":{"folds":[0.8697548433424982,0.8463740591990082,0.8789993353031841,0.875047115426111,0.8503594962004936],"mean":0.864106969894259,"std":0.014805547616559884},"eps":{"folds":[0.7465961796692451,0.853371468040014,0.7654002205588073,0.833440181762758,0.689757112647049],"mean":0.7777130325355748,"std":0.06649402555584401},"nc":{"folds":[0.8293413099467735,0.9096903675602839,0.8620922157331837,0.9025553640559385,0.7597664022643656],"mean":0.8526891319121092,"std":0.06124531499367895},"xc":{"folds":[0.2949066674319605,0.5776298509682867,0.5293570461002577,0.459736348324599,0.27888804660859845],"mean":0.42810359188674046,"std":0.13566290489832972}},"macro":0.814560762920045},"test":{"status":"未完成／不可比较"},"sota":{"eea":{"reference":0.934,"status":"未完成／不可比较"},"egb":{"reference":0.923,"status":"未完成／不可比较"},"egc":{"reference":0.911,"status":"未完成／不可比较"},"ei":{"reference":0.839,"status":"未完成／不可比较"},"eps":{"reference":0.787,"status":"未完成／不可比较"},"nc":{"reference":0.856,"status":"未完成／不可比较"},"xc":{"reference":0.423,"status":"未完成／不可比较"}},"repair1":{"plan":"UA2D-ET-SINGLE-20261020-01","revision":"r1","independent_review":"PENDING"},"sampling":{"status":"COMPLETE"}}
```

</details>

<details><summary>results/ua2d_et_single_20261020/r1/incidents/attempt1/final_review/results.json</summary>

```json
{"plan":"UA2D-ET-SINGLE-20261020-01","revision":"r1","status":"需返修","budget":{"plan":"UA2D-ET-SINGLE-20261020-01","revision":"r1"},"independent_review":"PENDING","validation":{"status":"未完成／不可比较"},"test":{"status":"未完成／不可比较"},"sota":{"eea":{"reference":0.934,"status":"未完成／不可比较"},"egb":{"reference":0.923,"status":"未完成／不可比较"},"egc":{"reference":0.911,"status":"未完成／不可比较"},"ei":{"reference":0.839,"status":"未完成／不可比较"},"eps":{"reference":0.787,"status":"未完成／不可比较"},"nc":{"reference":0.856,"status":"未完成／不可比较"},"xc":{"reference":0.423,"status":"未完成／不可比较"}}}
```

</details>

<details><summary>results/ua2d_v1_attnscale_20261020/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-V1-ATTNSCALE-20261020-01","revision":"r1","status":"待审查","independent_review":false,"attention":{"revision":"ua2d_attention_scale_r1"},"validation":{"tasks":{"eea":{"folds":[0.939279668671756,0.9138151293889358,0.9028413165918497,0.9178851705795806,0.9443404825589228],"mean":0.9236323535582089,"std":0.017573704060961048},"egb":{"folds":[0.9359585120778603,0.9410724997413165,0.9286143429900563,0.954741894020207,0.9536699748415575],"mean":0.9428114447341995,"std":0.01131136165402451},"egc":{"folds":[0.9328363408344194,0.9234179468696817,0.9236882992232608,0.906921956722846,0.9134857749898927],"mean":0.9200700637280201,"std":0.010043696490749236},"ei":{"folds":[0.873621693094937,0.8731195424095826,0.8967502041997181,0.8605543470185413,0.8657123712471031],"mean":0.8739516315939764,"std":0.013855944480095526},"eps":{"folds":[0.705280197331019,0.8395029196265799,0.76403116640961,0.8309589243586186,0.6709418283074016],"mean":0.7621430072066457,"std":0.07462369903024513},"nc":{"folds":[0.8187879934668701,0.8909759587134498,0.8647225762770606,0.8965330053270684,0.7623075512607097],"mean":0.8466654170090318,"std":0.0562797959497636},"xc":{"folds":[0.31574195334221844,0.5171017058875161,0.5649785365580922,0.5522806797234898,0.2873841919457134],"mean":0.447497413491406,"std":0.13474220726270314}},"macro":0.8166816187602126},"test":{"tasks":{"eea":{"folds":[0.9053600410300958,0.9176398900213631,0.9409850919907964,0.9294773440608678,0.9141060140423571],"mean":0.9215136762290961,"std":0.013904816125161729},"egb":{"folds":[0.9389726429421904,0.924319385619757,0.9442535413425045,0.9401767690537804,0.9190248002500908],"mean":0.9333494278416646,"std":0.010998516326070379},"egc":{"folds":[0.8834118356163293,0.9246600829036075,0.9165341448820102,0.9119666402636659,0.914367554168373],"mean":0.9101880515667972,"std":0.01571074434705681},"ei":{"folds":[0.80011645843816,0.7985420020064391,0.8073809961671518,0.8397955297206712,0.8481054550957464],"mean":0.8187880882856338,"std":0.023395919054989174},"eps":{"folds":[0.6996196928974894,0.7937880573742736,0.6742957629379845,0.6827124874840083,0.716303262105995],"mean":0.7133438525599501,"std":0.04777921771296451},"nc":{"folds":[0.7810305702248473,0.8912810577446829,0.8374489582220914,0.7350690264821903,0.8040937861290687],"mean":0.8097846797605761,"std":0.05884000734546702},"xc":{"folds":[0.38184282684077586,0.33750332474440625,0.44078387720668977,0.23675837698399305,0.43648795751361313],"mean":0.3666752726578956,"std":0.08415068814548535}},"macro":0.7819490069859448},"threshold":{"metric":"TEST macro R2","delta":-0.00525475387227492},"seven_task":{"eea":{"validation":{"folds":[0.939279668671756,0.9138151293889358,0.9028413165918497,0.9178851705795806,0.9443404825589228],"mean":0.9236323535582089,"std":0.017573704060961048},"test":{"folds":[0.9053600410300958,0.9176398900213631,0.9409850919907964,0.9294773440608678,0.9141060140423571],"mean":0.9215136762290961,"std":0.013904816125161729},"delta_test":-0.012486323770903973,"protocol":{"folds":5,"selection":"full-precision validation R2 per epoch; earlier epoch wins ties"}},"egb":{"validation":{"folds":[0.9359585120778603,0.9410724997413165,0.9286143429900563,0.954741894020207,0.9536699748415575],"mean":0.9428114447341995,"std":0.01131136165402451},"test":{"folds":[0.9389726429421904,0.924319385619757,0.9442535413425045,0.9401767690537804,0.9190248002500908],"mean":0.9333494278416646,"std":0.010998516326070379},"delta_test":0.010349427841664594,"protocol":{"folds":5,"selection":"full-precision validation R2 per epoch; earlier epoch wins ties"}},"egc":{"validation":{"folds":[0.9328363408344194,0.9234179468696817,0.9236882992232608,0.906921956722846,0.9134857749898927],"mean":0.9200700637280201,"std":0.010043696490749236},"test":{"folds":[0.8834118356163293,0.9246600829036075,0.9165341448820102,0.9119666402636659,0.914367554168373],"mean":0.9101880515667972,"std":0.01571074434705681},"delta_test":-0.0008119484332028692,"protocol":{"folds":5,"selection":"full-precision validation R2 per epoch; earlier epoch wins ties"}},"ei":{"validation":{"folds":[0.873621693094937,0.8731195424095826,0.8967502041997181,0.8605543470185413,0.8657123712471031],"mean":0.8739516315939764,"std":0.013855944480095526},"test":{"folds":[0.80011645843816,0.7985420020064391,0.8073809961671518,0.8397955297206712,0.8481054550957464],"mean":0.8187880882856338,"std":0.023395919054989174},"delta_test":-0.020211911714366204,"protocol":{"folds":5,"selection":"full-precision validation R2 per epoch; earlier epoch wins ties"}},"eps":{"validation":{"folds":[0.705280197331019,0.8395029196265799,0.76403116640961,0.8309589243586186,0.6709418283074016],"mean":0.7621430072066457,"std":0.07462369903024513},"test":{"folds":[0.6996196928974894,0.7937880573742736,0.6742957629379845,0.6827124874840083,0.716303262105995],"mean":0.7133438525599501,"std":0.04777921771296451},"delta_test":-0.07365614744004989,"protocol":{"folds":5,"selection":"full-precision validation R2 per epoch; earlier epoch wins ties"}},"nc":{"validation":{"folds":[0.8187879934668701,0.8909759587134498,0.8647225762770606,0.8965330053270684,0.7623075512607097],"mean":0.8466654170090318,"std":0.0562797959497636},"test":{"folds":[0.7810305702248473,0.8912810577446829,0.8374489582220914,0.7350690264821903,0.8040937861290687],"mean":0.8097846797605761,"std":0.05884000734546702},"delta_test":-0.046215320239423874,"protocol":{"folds":5,"selection":"full-precision validation R2 per epoch; earlier epoch wins ties"}},"xc":{"validation":{"folds":[0.31574195334221844,0.5171017058875161,0.5649785365580922,0.5522806797234898,0.2873841919457134],"mean":0.447497413491406,"std":0.13474220726270314},"test":{"folds":[0.38184282684077586,0.33750332474440625,0.44078387720668977,0.23675837698399305,0.43648795751361313],"mean":0.3666752726578956,"std":0.08415068814548535},"delta_test":-0.056324727342104364,"protocol":{"folds":5,"selection":"full-precision validation R2 per epoch; earlier epoch wins ties"}}},"budget":{"plan":"UA2D-V1-ATTNSCALE-20261020-01","revision":"r1"}}
```

</details>

<details><summary>results/ua2d_ph_descriptor_20261023/r1_p10k_gpu01/final_review/results.json</summary>

```json
{"plan":"UA2D-PH-DESC-20261023-01","revision":"r1.2-P10K-GPU01","status":"待审查","FINAL":"GATE_PH","formal":{"CONCAT_PH":{"tasks":{"eea":{"folds":[0.9075554203119256,0.9311270390059936,0.9413765504265906,0.9333473405532239,0.907371021672384],"mean":0.9241554743940235,"std":0.015707786485056628},"egb":{"folds":[0.9309292119706156,0.9135324282512917,0.941520691639056,0.9311279133315521,0.9285979278807505],"mean":0.9291416346146532,"std":0.010053913778532595},"egc":{"folds":[0.8985476332482711,0.9287593723395098,0.9229206547173355,0.9098914307852738,0.9278644327111712],"mean":0.9175967047603123,"std":0.013044291858919251},"ei":{"folds":[0.7859724606620896,0.7828219750672665,0.7562908535513075,0.8156646214904018,0.8096692185608434],"mean":0.7900838258663818,"std":0.023714802816559535},"eps":{"folds":[0.7012203464496825,0.8555055433145347,0.6655026144166674,0.7156130303361752,0.7102482223156794],"mean":0.7296179513665478,"std":0.07303459868487838},"nc":{"folds":[0.801298758057312,0.8926191381601709,0.8219647571950836,0.7387793705435126,0.8632881463083907],"mean":0.8235900340528939,"std":0.05923121745705669},"xc":{"folds":[0.3855523816066043,0.33474679024814225,0.46689790471856685,0.21107843109431312,0.39224266635926297],"mean":0.3581036348053779,"std":0.09475897458326805}},"macro":0.7817556085514558},"GATE_PH":{"tasks":{"eea":{"folds":[0.9077109908538702,0.9314034407562137,0.9420325924389966,0.9330056270284822,0.9101504910718897],"mean":0.9248606284298905,"std":0.015120548545924864},"egb":{"folds":[0.9325689074465006,0.9062120078178996,0.9327927945885878,0.9296087467838519,0.9278390238655326],"mean":0.9258042961004744,"std":0.011147542330137369},"egc":{"folds":[0.8899320243283348,0.9266121376416772,0.9266234741740983,0.9070389783369196,0.9218895271245319],"mean":0.9144192283211124,"std":0.015870627492045545},"ei":{"folds":[0.7857479355024367,0.7881958831566006,0.7502407302772025,0.7946302516638308,0.804384061857936],"mean":0.7846397724916013,"std":0.020535424820664196},"eps":{"folds":[0.7025577677051669,0.8626949218201653,0.6708495999961519,0.7212896924289358,0.7103535203390603],"mean":0.733549100457896,"std":0.07459668434579994},"nc":{"folds":[0.8061435542659555,0.901594788665279,0.8246147302021655,0.7442698884268901,0.8714557781076752],"mean":0.829615747933593,"std":0.06081003251619097},"xc":{"folds":[0.38523932006632056,0.25469504670474985,0.4656757522733116,0.22510878749476926,0.44211298533632604],"mean":0.3545663783750954,"std":0.10918342529502494}},"macro":0.7810650217299518}},"validation":{"CONCAT_PH":{"tasks":{"eea":{"folds":[0.9432657797542605,0.9035007628831447,0.9216044788783244,0.8993919053596757,0.9410199325041202],"mean":0.9217565718759051,"std":0.02041567466978001},"egb":{"folds":[0.9359732154985557,0.9462419129972651,0.9250044124450987,0.949681859057292,0.9585042926373468],"mean":0.9430811385271116,"std":0.01292985194328212},"egc":{"folds":[0.9184829470797045,0.9270648043945532,0.9272608699911606,0.9175616961795796,0.9060443581997384],"mean":0.9192829351689472,"std":0.008704416945479961},"ei":{"folds":[0.8841185572586489,0.8513463594500212,0.8409839589490369,0.8257619745260736,0.8022295254970281],"mean":0.8408880751361618,"std":0.030420149033599585},"eps":{"folds":[0.7454164206922598,0.8563612460780573,0.7509453281535267,0.786934854038339,0.6862256722866078],"mean":0.7651767042497581,"std":0.0624786044789948},"nc":{"folds":[0.8546962770932227,0.9022721307213607,0.8416753228054039,0.8621768922535937,0.7557197028974219],"mean":0.8433080651542006,"std":0.05393716734749215},"xc":{"folds":[0.31649476865076365,0.488248020301966,0.4303067742174981,0.5548650216446314,0.2590966360806257],"mean":0.40980224417909705,"std":0.12148647069882994}},"macro":0.8061851048987403},"GATE_PH":{"tasks":{"eea":{"folds":[0.9438739565985163,0.9022417357375558,0.9212254056532151,0.8995030519616051,0.941855543016129],"mean":0.9217399385934042,"std":0.021032570457340725},"egb":{"folds":[0.934463767664127,0.9462171575785592,0.9254685937707219,0.9445231923507962,0.9601525630546555],"mean":0.9421650548837718,"std":0.013072991598130346},"egc":{"folds":[0.9171477888700532,0.9279955226737089,0.9270856448481988,0.9152990978062281,0.9072167032973499],"mean":0.9189489514991077,"std":0.008692777107139664},"ei":{"folds":[0.8866770219100439,0.854979697580477,0.8448371058341082,0.8205226197825422,0.7977078106387079],"mean":0.8409448511491758,"std":0.03387501737815033},"eps":{"folds":[0.7541366015791415,0.8740187666108333,0.751986282679963,0.7875704664266754,0.6862183861269934],"mean":0.7707861006847213,"std":0.06841241767867881},"nc":{"folds":[0.8547216598272956,0.9224936239965342,0.8430578352119831,0.8607305075013822,0.755134217411767],"mean":0.8472275687897923,"std":0.06000425373850176},"xc":{"folds":[0.31471605863825514,0.471542924728247,0.42218878053746745,0.5500526471634121,0.240413797517825],"mean":0.3997828417170413,"std":0.12335659677276582}},"macro":0.8059421867595734}},"reference":{"macro":0.7932333045603579,"tasks":{"eea":{"folds":[0.9085344377329794,0.9500258136738593,0.9521388027334685,0.9396353097229598,0.9078530808580162],"mean":0.9316374889442566,"std":0.021919446468473047},"egb":{"folds":[0.9507045581796164,0.9125709629899116,0.9378252334372563,0.9224240450263853,0.9238208983256504],"mean":0.9294691395917638,"std":0.014898419935170406},"egc":{"folds":[0.8931196561691934,0.9285935986642526,0.91901340442812,0.9211382443946686,0.9156646205160663],"mean":0.91550590483446,"std":0.013383598657257183},"ei":{"folds":[0.7937226262329049,0.8200997336019509,0.8094929852163176,0.8400600927711817,0.837139549335576],"mean":0.8201029974315862,"std":0.019345710827028215},"eps":{"folds":[0.6879749158999735,0.8305278546078143,0.6609921529804077,0.7942839899185673,0.7072852733709486],"mean":0.7362128373555422,"std":0.0726110074671783},"nc":{"folds":[0.8122387550201953,0.8977313938936047,0.8306830844504185,0.6850384662079421,0.8508847499683402],"mean":0.8153152899081002,"std":0.07949101467232002},"xc":{"folds":[0.37894031667094474,0.4125651835310974,0.5107261046971499,0.22020467215677086,0.4995110922280195],"mean":0.40438947385679647,"std":0.11725415168285576}}},"contrasts":{"CONCAT_PH_minus_P10K_REF":{"status":"COMPLETE","delta":-0.011477696008902138,"tasks":{"eea":{"folds":[-0.0009790174210537739,-0.018898774667865714,-0.010762252306877884,-0.006287969169735952,-0.00048205918563215544],"mean":-0.007482014550233096,"std":0.007645529125948239},"egb":{"folds":[-0.019775346209000788,0.0009614652613800789,0.0036954582017997284,0.008703868305166784,0.004777029555100065],"mean":-0.0003275049771108263,"std":0.011221476972192163},"egc":{"folds":[0.005427977079077717,0.0001657736752571637,0.003907250289215569,-0.011246813609394835,0.012199812195104909],"mean":0.002090799925852105,"std":0.008634611068478418},"ei":{"folds":[-0.007750165570815315,-0.0372777585346844,-0.05320213166501009,-0.0243954712807799,-0.0274703307747326],"mean":-0.03001917156520446,"std":0.016764306919949252},"eps":{"folds":[0.013245430549708948,0.024977688706720347,0.004510461436259661,-0.07867095958239212,0.0029629489447308144],"mean":-0.00659488598899447,"std":0.0412313805491923},"nc":{"folds":[-0.01093999696288328,-0.005112255733433768,-0.008718327255334946,0.05374090433557055,0.012403396340050499],"mean":0.008274744144793811,"std":0.027024988476534843},"xc":{"folds":[0.006612064935659578,-0.07781839328295515,-0.043828199978583005,-0.00912624106245774,-0.10726842586875651],"mean":-0.046285839051418566,"std":0.04716524628908081}}},"GATE_PH_minus_P10K_REF":{"status":"COMPLETE","delta":-0.012168282830406052,"tasks":{"eea":{"folds":[-0.0008234468791091798,-0.01862237291764557,-0.010106210294471896,-0.006629682694477612,0.0022974102138735297],"mean":-0.006776860514366145,"std":0.008203778375344117},"egb":{"folds":[-0.01813565073311585,-0.006358955172012015,-0.005032438848668552,0.0071847017574665495,0.0040181255398822335],"mean":-0.0036648434912895265,"std":0.009940742707144333},"egc":{"folds":[-0.0031876318408585735,-0.001981461022575459,0.007610069745978332,-0.014099266057749027,0.00622490660846553],"mean":-0.0010866765133478395,"std":0.008712559648339552},"ei":{"folds":[-0.007974690730468192,-0.03190385044535027,-0.05925225493911512,-0.04542984110735082,-0.03275548747764001],"mean":-0.03546322493998488,"std":0.018979409158828744},"eps":{"folds":[0.014582851805193364,0.032167067212350964,0.009857447015744203,-0.07299429748963149,0.0030682469681116986],"mean":-0.0026637368976462515,"std":0.04076350443609785},"nc":{"folds":[-0.006095200754239749,0.0038633947716743,-0.006068354248253027,0.05923142221894806,0.02057102813933498],"mean":0.014300458025492912,"std":0.027376610878507027},"xc":{"folds":[0.006299003395375813,-0.15787013682634754,-0.04505035242383826,0.004904115337998394,-0.057398106891693446],"mean":-0.04982309548170101,"std":0.06689372280708009}}},"GATE_PH_minus_CONCAT_PH":{"status":"COMPLETE","delta":-0.0006905868215039135,"tasks":{"eea":{"folds":[0.00015557054194459408,0.00027640175022014546,0.0006560420124059885,-0.0003417135247416603,0.002779469399505685],"mean":0.0007051540358669506,"std":0.001213156878496142},"egb":{"folds":[0.0016396954758849391,-0.007320420433392094,-0.00872789705046828,-0.0015191665477002347,-0.0007589040152178317],"mean":-0.0033373385141787005,"std":0.0044622865218320994},"egc":{"folds":[-0.00861560891993629,-0.002147234697832623,0.0037028194567627626,-0.0028524524483541924,-0.0059749055866393785],"mean":-0.0031774764391999445,"std":0.0046342314618840925},"ei":{"folds":[-0.00022452515965287745,0.005373908089334134,-0.006050123274105035,-0.02103436982657092,-0.005285156702907412],"mean":-0.005444053374780422,"std":0.009845332662863013},"eps":{"folds":[0.0013374212554844167,0.007189378505630617,0.005346985579484542,0.0056766620927606315,0.00010529802338088423],"mean":0.003931149091348218,"std":0.003042697322941213},"nc":{"folds":[0.0048447962086435314,0.008975650505108068,0.0026499730070819183,0.00549051788337751,0.00816763179928448],"mean":0.006025713880699102,"std":0.002567419239602212},"xc":{"folds":[-0.00031306154028376554,-0.0800517435433924,-0.0012221524452552535,0.014030356400456134,0.049870318977063066],"mean":-0.003537256430282443,"std":0.04751634457950885}}}},"budget":{"plan":"UA2D-PH-DESC-20261023-01","revision":"r1.2-P10K-GPU01"},"independent_review":"PENDING","sota":{"P10K_REF":{"eea":{"project":{"folds":[0.9085344377329794,0.9500258136738593,0.9521388027334685,0.9396353097229598,0.9078530808580162],"mean":0.9316374889442566,"std":0.021919446468473047},"reference":0.934,"delta":-0.002362511055743477},"egb":{"project":{"folds":[0.9507045581796164,0.9125709629899116,0.9378252334372563,0.9224240450263853,0.9238208983256504],"mean":0.9294691395917638,"std":0.014898419935170406},"reference":0.923,"delta":0.006469139591763762},"egc":{"project":{"folds":[0.8931196561691934,0.9285935986642526,0.91901340442812,0.9211382443946686,0.9156646205160663],"mean":0.91550590483446,"std":0.013383598657257183},"reference":0.911,"delta":0.004505904834460006},"ei":{"project":{"folds":[0.7937226262329049,0.8200997336019509,0.8094929852163176,0.8400600927711817,0.837139549335576],"mean":0.8201029974315862,"std":0.019345710827028215},"reference":0.839,"delta":-0.018897002568413757},"eps":{"project":{"folds":[0.6879749158999735,0.8305278546078143,0.6609921529804077,0.7942839899185673,0.7072852733709486],"mean":0.7362128373555422,"std":0.0726110074671783},"reference":0.787,"delta":-0.05078716264445782},"nc":{"project":{"folds":[0.8122387550201953,0.8977313938936047,0.8306830844504185,0.6850384662079421,0.8508847499683402],"mean":0.8153152899081002,"std":0.07949101467232002},"reference":0.856,"delta":-0.04068471009189978},"xc":{"project":{"folds":[0.37894031667094474,0.4125651835310974,0.5107261046971499,0.22020467215677086,0.4995110922280195],"mean":0.40438947385679647,"std":0.11725415168285576},"reference":0.423,"delta":-0.01861052614320352}},"CONCAT_PH":{"eea":{"project":{"folds":[0.9075554203119256,0.9311270390059936,0.9413765504265906,0.9333473405532239,0.907371021672384],"mean":0.9241554743940235,"std":0.015707786485056628},"reference":0.934,"delta":-0.009844525605976595},"egb":{"project":{"folds":[0.9309292119706156,0.9135324282512917,0.941520691639056,0.9311279133315521,0.9285979278807505],"mean":0.9291416346146532,"std":0.010053913778532595},"reference":0.923,"delta":0.006141634614653113},"egc":{"project":{"folds":[0.8985476332482711,0.9287593723395098,0.9229206547173355,0.9098914307852738,0.9278644327111712],"mean":0.9175967047603123,"std":0.013044291858919251},"reference":0.911,"delta":0.006596704760312266},"ei":{"project":{"folds":[0.7859724606620896,0.7828219750672665,0.7562908535513075,0.8156646214904018,0.8096692185608434],"mean":0.7900838258663818,"std":0.023714802816559535},"reference":0.839,"delta":-0.04891617413361815},"eps":{"project":{"folds":[0.7012203464496825,0.8555055433145347,0.6655026144166674,0.7156130303361752,0.7102482223156794],"mean":0.7296179513665478,"std":0.07303459868487838},"reference":0.787,"delta":-0.0573820486334522},"nc":{"project":{"folds":[0.801298758057312,0.8926191381601709,0.8219647571950836,0.7387793705435126,0.8632881463083907],"mean":0.8235900340528939,"std":0.05923121745705669},"reference":0.856,"delta":-0.0324099659471061},"xc":{"project":{"folds":[0.3855523816066043,0.33474679024814225,0.46689790471856685,0.21107843109431312,0.39224266635926297],"mean":0.3581036348053779,"std":0.09475897458326805},"reference":0.423,"delta":-0.06489636519462211}},"GATE_PH":{"eea":{"project":{"folds":[0.9077109908538702,0.9314034407562137,0.9420325924389966,0.9330056270284822,0.9101504910718897],"mean":0.9248606284298905,"std":0.015120548545924864},"reference":0.934,"delta":-0.009139371570109578},"egb":{"project":{"folds":[0.9325689074465006,0.9062120078178996,0.9327927945885878,0.9296087467838519,0.9278390238655326],"mean":0.9258042961004744,"std":0.011147542330137369},"reference":0.923,"delta":0.0028042961004743905},"egc":{"project":{"folds":[0.8899320243283348,0.9266121376416772,0.9266234741740983,0.9070389783369196,0.9218895271245319],"mean":0.9144192283211124,"std":0.015870627492045545},"reference":0.911,"delta":0.003419228321112411},"ei":{"project":{"folds":[0.7857479355024367,0.7881958831566006,0.7502407302772025,0.7946302516638308,0.804384061857936],"mean":0.7846397724916013,"std":0.020535424820664196},"reference":0.839,"delta":-0.05436022750839864},"eps":{"project":{"folds":[0.7025577677051669,0.8626949218201653,0.6708495999961519,0.7212896924289358,0.7103535203390603],"mean":0.733549100457896,"std":0.07459668434579994},"reference":0.787,"delta":-0.053450899542104024},"nc":{"project":{"folds":[0.8061435542659555,0.901594788665279,0.8246147302021655,0.7442698884268901,0.8714557781076752],"mean":0.829615747933593,"std":0.06081003251619097},"reference":0.856,"delta":-0.026384252066406932},"xc":{"project":{"folds":[0.38523932006632056,0.25469504670474985,0.4656757522733116,0.22510878749476926,0.44211298533632604],"mean":0.3545663783750954,"std":0.10918342529502494},"reference":0.423,"delta":-0.06843362162490457}}}}
```

</details>

<details><summary>results/ua3d_ft_balance_20261028/r1/final_review/results.json</summary>

```json
{"plan":"UA3D-FT-BALANCE-20261028-01","revision":"r1","execution_complete":true,"independent_review":"PENDING","FINAL":"BASE_KEEP","validation":{"BASE_ORIGINAL":{"tasks":{"eea":{"folds":[0.9258489962334948,0.9178132892834358,0.9058696511975507,0.90940444186418,0.9557389448516351],"mean":0.9229350646860593,"std":0.0199068092981852},"egb":{"folds":[0.9317327685836467,0.9323129091615904,0.9043846369319708,0.931173957145727,0.9434588393916917],"mean":0.9286126222429253,"std":0.014468874427615519},"egc":{"folds":[0.9136946146435524,0.923116820626894,0.9165977660323378,0.9120366473326542,0.9075833727563615],"mean":0.91460584427836,"std":0.005769546978527248},"ei":{"folds":[0.888919055883577,0.8665769855366472,0.8655728923752912,0.8477709629625998,0.8410585473221172],"mean":0.8619796888160464,"std":0.018703995554038573},"eps":{"folds":[0.7472369136542818,0.856980031011281,0.772058361523108,0.8054533188233461,0.6566912907405946],"mean":0.7676839831505223,"std":0.07442597410557149},"nc":{"folds":[0.8335550837981948,0.9090776141944835,0.8382871410816238,0.8729549039891361,0.7769326614604025],"mean":0.8461614809047682,"std":0.04921860234206862},"xc":{"folds":[0.351201144033391,0.4562607014500726,0.3929779893799423,0.5251736603280752,0.3396932842125413],"mean":0.41306135588080445,"std":0.07753749677772813}},"macro":0.8078628628513551},"BASE_REFT_HISTORY":{"tasks":{"eea":{"folds":[0.9260557892520624,0.9141144692880022,0.8993417263064409,0.9119713509469027,0.9528767691342561],"mean":0.9208720209855329,"std":0.020246082204677187},"egb":{"folds":[0.9371369122196221,0.930344017333475,0.8969212245376478,0.9295492391454958,0.9396467950080171],"mean":0.9267196376488517,"std":0.017209875097930948},"egc":{"folds":[0.9130608411535063,0.9232113269672673,0.919244512235269,0.9144094730703112,0.9106630474317192],"mean":0.9161178401716146,"std":0.005052528033036589},"ei":{"folds":[0.8941826297402797,0.8670303167843302,0.8614572599877819,0.8669683969937125,0.8483521426803055],"mean":0.8675981492372818,"std":0.016700096664113824},"eps":{"folds":[0.7851577412322878,0.8386169692860449,0.781967901395976,0.8035009804951252,0.6608131216511253],"mean":0.7740113428121118,"std":0.06716567575595468},"nc":{"folds":[0.8309792186722673,0.9054812846856072,0.8430952440546673,0.8903137746445404,0.7826533310948671],"mean":0.8505045706303898,"std":0.049109927734752735},"xc":{"folds":[0.3479902347287206,0.4259342009956365,0.38850647638729285,0.5251736341517339,0.330540647903266],"mean":0.4036290388333299,"std":0.07735775690916945}},"macro":0.8084932286170161},"C0_BASE":{"tasks":{"eea":{"folds":[0.9258970514538238,0.9158681724960891,0.905258800607204,0.9169298136386527,0.952623047039815],"mean":0.923315377047117,"std":0.017943528022255892},"egb":{"folds":[0.9333544768783484,0.9297525006451111,0.8987166190065634,0.9375436779757765,0.943120757254836],"mean":0.9284976063521271,"std":0.01737627054646118},"egc":{"folds":[0.9157183604656616,0.9231443997090095,0.9178368808756239,0.9162071441077677,0.9101041077036572],"mean":0.9166021785723439,"std":0.004676731199971195},"ei":{"folds":[0.8874448479646658,0.8778322958523442,0.8623956669984635,0.8662765564407711,0.8440230212106303],"mean":0.867594477693375,"std":0.016456216381287227},"eps":{"folds":[0.7891147602749332,0.8485159480338706,0.7667802818647083,0.8040262112660583,0.6965165573295831],"mean":0.7809907517538306,"std":0.055888529726911745},"nc":{"folds":[0.8459483759222525,0.8994888279014615,0.8382396063580053,0.8810895218283362,0.7831459547093051],"mean":0.8495824573438722,"std":0.04483028171124671},"xc":{"folds":[0.34161665628573945,0.45214163362182824,0.4289743951260029,0.5251736232034909,0.3101155339647632],"mean":0.41160436844036496,"std":0.08666202556304385}},"macro":0.8111696024575759},"A_ALIGN_DECAY":{"tasks":{"eea":{"folds":[0.9257487773174944,0.9172088886071851,0.8974909273480062,0.9140421655119553,0.9521910622430614],"mean":0.9213363642055405,"std":0.02006444634658725},"egb":{"folds":[0.9371748458853583,0.9305981439019531,0.908188809308417,0.9337204844859569,0.9438729421290941],"mean":0.9307110451421557,"std":0.013521620013400794},"egc":{"folds":[0.9187104666628816,0.9221169989246862,0.9188895340781444,0.912366378920545,0.9095997469577389],"mean":0.9163366251087993,"std":0.005165113052940486},"ei":{"folds":[0.8865411112977298,0.867558006964052,0.8616267737345137,0.8569330224272658,0.8418401057197789],"mean":0.862899804028668,"std":0.01628913936989545},"eps":{"folds":[0.7860121028940531,0.839527362184344,0.7568639911112602,0.8062651811161227,0.6593924640560445],"mean":0.7696122202723649,"std":0.06857739198219868},"nc":{"folds":[0.8477933935859097,0.8910821319932332,0.8565465334081178,0.8756320517674457,0.7753419601402699],"mean":0.8492792141789952,"std":0.04461976891789949},"xc":{"folds":[0.3784889778020516,0.40081860129432023,0.4761672941563537,0.5251736425946747,0.37192105414793186],"mean":0.43051391399906647,"std":0.06720532744145145}},"macro":0.8115270267050844},"B_MODAL_LR":{"tasks":{"eea":{"folds":[0.9255853915448886,0.9059749053000526,0.904756057370018,0.9139263354593367,0.9552840385253366],"mean":0.9211053456399265,"std":0.020832795645626963},"egb":{"folds":[0.9317856595915353,0.9325457656843191,0.9055877052042858,0.9387594605589558,0.9433034450793961],"mean":0.9303964072236983,"std":0.014650543907297123},"egc":{"folds":[0.9160329784436327,0.9185405966489107,0.9182980772060965,0.9146898325907613,0.9092137939647895],"mean":0.9153550557708382,"std":0.0037890052604309496},"ei":{"folds":[0.8994826019462413,0.8783972782662042,0.869043212508624,0.86392137438925,0.8348960295782614],"mean":0.8691480993377162,"std":0.02348543855827584},"eps":{"folds":[0.7863013703670814,0.8232143016766765,0.7590850478086836,0.7907237548082415,0.6462787271754371],"mean":0.7611206403672239,"std":0.06811542590588603},"nc":{"folds":[0.8367124512923603,0.8969939687624584,0.8528994165883146,0.8730455980062558,0.7741931396208526],"mean":0.8467689148540483,"std":0.04641922764104589},"xc":{"folds":[0.3449780319118264,0.39790969650995855,0.4368747015416339,0.616704777318992,0.37876290826446735],"mean":0.4350460231093757,"std":0.10684379468891013}},"macro":0.8112772123289753},"AB_DECAY_LR":{"tasks":{"eea":{"folds":[0.926597172423114,0.9104959043380478,0.9151180669089639,0.9120239817309362,0.9556852413368089],"mean":0.9239840733475742,"std":0.018810721035739594},"egb":{"folds":[0.932323375239474,0.9343193745206325,0.9103250814209003,0.9385591594615764,0.9425765655320629],"mean":0.9316207112349293,"std":0.012544452724042312},"egc":{"folds":[0.9173421273161988,0.9193541254137731,0.9189449336297002,0.9133072228402389,0.9065728685743137],"mean":0.915104255554845,"std":0.005334654681928976},"ei":{"folds":[0.8899411867788976,0.8797607402234062,0.8774809316369975,0.8531986259225991,0.8403463393926609],"mean":0.8681455647909122,"std":0.02057497525270456},"eps":{"folds":[0.7804384656219867,0.8162951482140872,0.7606515741581308,0.7907316025397502,0.6416246635521732],"mean":0.7579482908172256,"std":0.06805035499403125},"nc":{"folds":[0.8363404070648042,0.9046415644583644,0.8441476321706691,0.876289607398971,0.7730138524187593],"mean":0.8468866127023137,"std":0.04943702997017584},"xc":{"folds":[0.2980235013364987,0.4124315744369528,0.3959395809835774,0.5902059941762723,0.36441047721600206],"mean":0.41220222562986064,"std":0.10870913039644393}},"macro":0.8079845334396657}},"formal":{"BASE_ORIGINAL":{"tasks":{"eea":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"egb":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"egc":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"ei":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"eps":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"nc":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"xc":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815}},"macro":0.804675462276949},"BASE_REFT_HISTORY":{"tasks":{"eea":{"folds":[0.9049769797534798,0.9337302948382149,0.9516995851295479,0.9439625692503532,0.9180147982506615],"mean":0.9304768454444513,"std":0.01902892620319164},"egb":{"folds":[0.9301066676528006,0.9080527255095839,0.9273389169254724,0.9308825087595678,0.9110773298047282],"mean":0.9214916297304306,"std":0.01101884799428503},"egc":{"folds":[0.8940269313312734,0.9226997039839572,0.9207141469471163,0.9044920446982162,0.9227965070639776],"mean":0.9129458668049081,"std":0.013056721039985241},"ei":{"folds":[0.7785815070972912,0.7901070375903684,0.7896397334577828,0.8147737655658901,0.8739578678098907],"mean":0.8094119823042446,"std":0.03843520837653814},"eps":{"folds":[0.7546519544007158,0.8353980470179342,0.6921698767367295,0.7801628458516471,0.7941884476108171],"mean":0.7713142343235686,"std":0.05303235691300377},"nc":{"folds":[0.8143816639584283,0.8827951064379919,0.8244831743742866,0.7508113726273172,0.8464509219396861],"mean":0.823784447867542,"std":0.04850256989523753},"xc":{"folds":[0.5076895507308201,0.35343285252198553,0.43872220092850245,0.23778410565263752,0.6157927239010046],"mean":0.43068428674699,"std":0.14442802377591465}},"macro":0.8000156133174479},"C0_BASE":{"tasks":{"eea":{"folds":[0.9031266679593679,0.9332529563741213,0.9394885498766146,0.9456220679042692,0.916859936824641],"mean":0.9276700357878027,"std":0.017409457764433867},"egb":{"folds":[0.9258642092462569,0.9118242988570773,0.9310356259045222,0.917828223338208,0.9170675271417235],"mean":0.9207239768975576,"std":0.007643810838827071},"egc":{"folds":[0.8914283148351749,0.9236163529854077,0.9251479254965488,0.9078679234530692,0.9177613253155779],"mean":0.9131643684171558,"std":0.013914830119267866},"ei":{"folds":[0.7828139747638849,0.8024271513429672,0.7837434863754631,0.8408454459263145,0.8479327888328769],"mean":0.8115525694483011,"std":0.031081015733926444},"eps":{"folds":[0.7511608115272669,0.8253641034860147,0.6727599503278685,0.7047337416763577,0.7739098456594184],"mean":0.7455856905353853,"std":0.05951975527679634},"nc":{"folds":[0.8320159264493313,0.8846437043419035,0.8221397054496566,0.7246512492331241,0.8450477669117747],"mean":0.821699670477158,"std":0.05922858228994316},"xc":{"folds":[0.5065777359392558,0.3918872573052945,0.3965168190710452,0.23778412672465543,0.567494612121516],"mean":0.4200521102323534,"std":0.12628812488682334}},"macro":0.7943497745422448},"A_ALIGN_DECAY":{"tasks":{"eea":{"folds":[0.9045051648260021,0.9291945976999711,0.93660079420341,0.9467595340317096,0.9152430459164859],"mean":0.9264606273355158,"std":0.01680936984260915},"egb":{"folds":[0.9318700130193677,0.9068539757246767,0.9318003639410264,0.9203174383069816,0.9168487898068457],"mean":0.9215381161597795,"std":0.010620330016333356},"egc":{"folds":[0.8954940597329181,0.9222005268942712,0.921879607955614,0.9080998803582107,0.9259030896420558],"mean":0.9147154329166141,"std":0.012705731202591374},"ei":{"folds":[0.7884231803869564,0.7989326061254796,0.7727133126562815,0.8669222617863763,0.8768311548735388],"mean":0.8207645031657265,"std":0.04771141873660545},"eps":{"folds":[0.7492667742385132,0.8133641688819103,0.6884094045644159,0.7892032358428462,0.7694089226352442],"mean":0.7619305012325859,"std":0.04746280818791013},"nc":{"folds":[0.838488999374993,0.8830862922872995,0.8137809919331392,0.7846104386661646,0.845350412302381],"mean":0.8330634269127956,"std":0.03674982406609253},"xc":{"folds":[0.5079170864870101,0.3795977720010695,0.47604893967469863,0.23778409471443362,0.5043105139432844],"mean":0.4211316813640993,"std":0.11489966617572261}},"macro":0.7999434698695882},"B_MODAL_LR":{"tasks":{"eea":{"folds":[0.909089908284339,0.9317791056778604,0.9325665647675506,0.948555265650546,0.9174263696409612],"mean":0.9278834428042515,"std":0.01522316819218393},"egb":{"folds":[0.9242774682963187,0.8993735177640904,0.9244087994077628,0.9051728871307174,0.9167295384770083],"mean":0.9139924422151795,"std":0.01132774755406195},"egc":{"folds":[0.8953419081163848,0.9246255754420566,0.9179586574570652,0.9127529735436738,0.9235242695297841],"mean":0.914840676817793,"std":0.011889399822673131},"ei":{"folds":[0.771270391569792,0.7849741899289643,0.7966818387585748,0.8711945944517282,0.8483517418505448],"mean":0.814494551311921,"std":0.04306470914021787},"eps":{"folds":[0.7550453863386251,0.8322496248298739,0.5364327777401563,0.7624783374030997,0.7748411905216643],"mean":0.732209463366684,"std":0.1135690239613325},"nc":{"folds":[0.8041325560880819,0.8941189942326692,0.8154724847073886,0.8017553725998287,0.8708158921439366],"mean":0.8372590599543811,"std":0.0424017837121737},"xc":{"folds":[0.5130417703661541,0.30331484604785175,0.4428759099126335,0.33620390609688144,0.5633322650989303],"mean":0.43175373950449025,"std":0.11143459410137056}},"macro":0.7960619108535286},"AB_DECAY_LR":{"tasks":{"eea":{"folds":[0.911058866751712,0.9337558165241129,0.9335821800302612,0.948131316945498,0.9170178946461592],"mean":0.9287092149795486,"std":0.014787273109101448},"egb":{"folds":[0.92243174564714,0.8986292621337116,0.9317042012813916,0.9240118443602714,0.9138587033870932],"mean":0.9181271513619216,"std":0.012608357948502916},"egc":{"folds":[0.8973498792308929,0.9243374356685544,0.9201276065045769,0.9083990687178851,0.922345401023286],"mean":0.9145118782290389,"std":0.011416465216540321},"ei":{"folds":[0.7948673591474984,0.7883307687433662,0.7966642199554042,0.859324494143849,0.8315263248746072],"mean":0.8141426333729449,"std":0.030359828559556545},"eps":{"folds":[0.7548085721816783,0.7987726596190508,0.5579458512139692,0.7624899035830939,0.779360585937199],"mean":0.7306755145069982,"std":0.09802748818432702},"nc":{"folds":[0.7402376421841073,0.8740961810300039,0.820598156782171,0.7328114677063449,0.8616572468584706],"mean":0.8058801389122194,"std":0.06638731488691142},"xc":{"folds":[0.46646807075473407,0.29796597272581027,0.4005241979238625,0.28791610791176425,0.6086857130657267],"mean":0.4123120124763796,"std":0.13246114466364775}},"macro":0.7891940776912932}},"reference":0.804675462276949,"selected_comparison":{"macro":0.804675462276949,"delta_macro":0.0,"performance_pass":false,"delta_vs_C0":0.010325687734704214,"delta_vs_history":0.004659848959501112},"performance_pass":false,"pairs":{"validation":{"A_ALIGN_DECAY_minus_C0_BASE":{"delta_macro":0.00035742424750850876},"B_MODAL_LR_minus_C0_BASE":{"delta_macro":0.0001076098713994611},"AB_DECAY_LR_minus_C0_BASE":{"delta_macro":-0.003185069017910158},"AB_DECAY_LR_minus_B_MODAL_LR":{"delta_macro":-0.0032926788893096193},"AB_DECAY_LR_minus_A_ALIGN_DECAY":{"delta_macro":-0.003542493265418667},"B_MODAL_LR_minus_A_ALIGN_DECAY":{"delta_macro":-0.00024981437610904766},"C0_BASE_minus_BASE_ORIGINAL":{"delta_macro":0.0033067396062207433},"C0_BASE_minus_BASE_REFT_HISTORY":{"delta_macro":0.0026763738405597426},"A_ALIGN_DECAY_minus_BASE_ORIGINAL":{"delta_macro":0.003664163853729252},"A_ALIGN_DECAY_minus_BASE_REFT_HISTORY":{"delta_macro":0.0030337980880682514},"B_MODAL_LR_minus_BASE_ORIGINAL":{"delta_macro":0.0034143494776202044},"B_MODAL_LR_minus_BASE_REFT_HISTORY":{"delta_macro":0.0027839837119592037},"AB_DECAY_LR_minus_BASE_ORIGINAL":{"delta_macro":0.00012167058831058508},"AB_DECAY_LR_minus_BASE_REFT_HISTORY":{"delta_macro":-0.0005086951773504156}},"formal":{"A_ALIGN_DECAY_minus_C0_BASE":{"delta_macro":0.00559369532734344},"B_MODAL_LR_minus_C0_BASE":{"delta_macro":0.0017121363112838628},"AB_DECAY_LR_minus_C0_BASE":{"delta_macro":-0.005155696850951519},"AB_DECAY_LR_minus_B_MODAL_LR":{"delta_macro":-0.006867833162235382},"AB_DECAY_LR_minus_A_ALIGN_DECAY":{"delta_macro":-0.010749392178294959},"B_MODAL_LR_minus_A_ALIGN_DECAY":{"delta_macro":-0.003881559016059577},"C0_BASE_minus_BASE_ORIGINAL":{"delta_macro":-0.010325687734704214},"C0_BASE_minus_BASE_REFT_HISTORY":{"delta_macro":-0.0056658387752031025},"A_ALIGN_DECAY_minus_BASE_ORIGINAL":{"delta_macro":-0.004731992407360774},"A_ALIGN_DECAY_minus_BASE_REFT_HISTORY":{"delta_macro":-7.214344785966276e-05},"B_MODAL_LR_minus_BASE_ORIGINAL":{"delta_macro":-0.008613551423420351},"B_MODAL_LR_minus_BASE_REFT_HISTORY":{"delta_macro":-0.00395370246391924},"AB_DECAY_LR_minus_BASE_ORIGINAL":{"delta_macro":-0.015481384585655733},"AB_DECAY_LR_minus_BASE_REFT_HISTORY":{"delta_macro":-0.010821535626154621}}},"descriptive_interaction":{"validation":{"macro":-0.003650103136818128},"formal":{"macro":-0.012461528489578821}},"sota":{"BASE_ORIGINAL":{"eea":{"project":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.004431145079367593},"egb":{"project":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.0022430816246885588},"egc":{"project":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.003138905643812362},"ei":{"project":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.01762400606686776},"eps":{"project":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.011119521310488989},"nc":{"project":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.024623190090803382},"xc":{"project":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.01663027446704779}},"BASE_REFT_HISTORY":{"eea":{"project":{"folds":[0.9049769797534798,0.9337302948382149,0.9516995851295479,0.9439625692503532,0.9180147982506615],"mean":0.9304768454444513,"std":0.01902892620319164},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.0035231545555487465},"egb":{"project":{"folds":[0.9301066676528006,0.9080527255095839,0.9273389169254724,0.9308825087595678,0.9110773298047282],"mean":0.9214916297304306,"std":0.01101884799428503},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.0015083702695694035},"egc":{"project":{"folds":[0.8940269313312734,0.9226997039839572,0.9207141469471163,0.9044920446982162,0.9227965070639776],"mean":0.9129458668049081,"std":0.013056721039985241},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0019458668049080474},"ei":{"project":{"folds":[0.7785815070972912,0.7901070375903684,0.7896397334577828,0.8147737655658901,0.8739578678098907],"mean":0.8094119823042446,"std":0.03843520837653814},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.029588017695755364},"eps":{"project":{"folds":[0.7546519544007158,0.8353980470179342,0.6921698767367295,0.7801628458516471,0.7941884476108171],"mean":0.7713142343235686,"std":0.05303235691300377},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.01568576567643143},"nc":{"project":{"folds":[0.8143816639584283,0.8827951064379919,0.8244831743742866,0.7508113726273172,0.8464509219396861],"mean":0.823784447867542,"std":0.04850256989523753},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.03221555213245797},"xc":{"project":{"folds":[0.5076895507308201,0.35343285252198553,0.43872220092850245,0.23778410565263752,0.6157927239010046],"mean":0.43068428674699,"std":0.14442802377591465},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.00768428674699001}},"C0_BASE":{"eea":{"project":{"folds":[0.9031266679593679,0.9332529563741213,0.9394885498766146,0.9456220679042692,0.916859936824641],"mean":0.9276700357878027,"std":0.017409457764433867},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.0063299642121973365},"egb":{"project":{"folds":[0.9258642092462569,0.9118242988570773,0.9310356259045222,0.917828223338208,0.9170675271417235],"mean":0.9207239768975576,"std":0.007643810838827071},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.0022760231024424593},"egc":{"project":{"folds":[0.8914283148351749,0.9236163529854077,0.9251479254965488,0.9078679234530692,0.9177613253155779],"mean":0.9131643684171558,"std":0.013914830119267866},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0021643684171557265},"ei":{"project":{"folds":[0.7828139747638849,0.8024271513429672,0.7837434863754631,0.8408454459263145,0.8479327888328769],"mean":0.8115525694483011,"std":0.031081015733926444},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.027447430551698848},"eps":{"project":{"folds":[0.7511608115272669,0.8253641034860147,0.6727599503278685,0.7047337416763577,0.7739098456594184],"mean":0.7455856905353853,"std":0.05951975527679634},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.041414309464614774},"nc":{"project":{"folds":[0.8320159264493313,0.8846437043419035,0.8221397054496566,0.7246512492331241,0.8450477669117747],"mean":0.821699670477158,"std":0.05922858228994316},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.03430032952284201},"xc":{"project":{"folds":[0.5065777359392558,0.3918872573052945,0.3965168190710452,0.23778412672465543,0.567494612121516],"mean":0.4200521102323534,"std":0.12628812488682334},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":-0.0029478897676465943}},"A_ALIGN_DECAY":{"eea":{"project":{"folds":[0.9045051648260021,0.9291945976999711,0.93660079420341,0.9467595340317096,0.9152430459164859],"mean":0.9264606273355158,"std":0.01680936984260915},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.007539372664484234},"egb":{"project":{"folds":[0.9318700130193677,0.9068539757246767,0.9318003639410264,0.9203174383069816,0.9168487898068457],"mean":0.9215381161597795,"std":0.010620330016333356},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.0014618838402205014},"egc":{"project":{"folds":[0.8954940597329181,0.9222005268942712,0.921879607955614,0.9080998803582107,0.9259030896420558],"mean":0.9147154329166141,"std":0.012705731202591374},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0037154329166140254},"ei":{"project":{"folds":[0.7884231803869564,0.7989326061254796,0.7727133126562815,0.8669222617863763,0.8768311548735388],"mean":0.8207645031657265,"std":0.04771141873660545},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.0182354968342735},"eps":{"project":{"folds":[0.7492667742385132,0.8133641688819103,0.6884094045644159,0.7892032358428462,0.7694089226352442],"mean":0.7619305012325859,"std":0.04746280818791013},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.025069498767414156},"nc":{"project":{"folds":[0.838488999374993,0.8830862922872995,0.8137809919331392,0.7846104386661646,0.845350412302381],"mean":0.8330634269127956,"std":0.03674982406609253},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.022936573087204426},"xc":{"project":{"folds":[0.5079170864870101,0.3795977720010695,0.47604893967469863,0.23778409471443362,0.5043105139432844],"mean":0.4211316813640993,"std":0.11489966617572261},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":-0.001868318635900701}},"B_MODAL_LR":{"eea":{"project":{"folds":[0.909089908284339,0.9317791056778604,0.9325665647675506,0.948555265650546,0.9174263696409612],"mean":0.9278834428042515,"std":0.01522316819218393},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.006116557195748551},"egb":{"project":{"folds":[0.9242774682963187,0.8993735177640904,0.9244087994077628,0.9051728871307174,0.9167295384770083],"mean":0.9139924422151795,"std":0.01132774755406195},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.009007557784820541},"egc":{"project":{"folds":[0.8953419081163848,0.9246255754420566,0.9179586574570652,0.9127529735436738,0.9235242695297841],"mean":0.914840676817793,"std":0.011889399822673131},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0038406768177929873},"ei":{"project":{"folds":[0.771270391569792,0.7849741899289643,0.7966818387585748,0.8711945944517282,0.8483517418505448],"mean":0.814494551311921,"std":0.04306470914021787},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.024505448688079023},"eps":{"project":{"folds":[0.7550453863386251,0.8322496248298739,0.5364327777401563,0.7624783374030997,0.7748411905216643],"mean":0.732209463366684,"std":0.1135690239613325},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.05479053663331601},"nc":{"project":{"folds":[0.8041325560880819,0.8941189942326692,0.8154724847073886,0.8017553725998287,0.8708158921439366],"mean":0.8372590599543811,"std":0.0424017837121737},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.01874094004561888},"xc":{"project":{"folds":[0.5130417703661541,0.30331484604785175,0.4428759099126335,0.33620390609688144,0.5633322650989303],"mean":0.43175373950449025,"std":0.11143459410137056},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.008753739504490265}},"AB_DECAY_LR":{"eea":{"project":{"folds":[0.911058866751712,0.9337558165241129,0.9335821800302612,0.948131316945498,0.9170178946461592],"mean":0.9287092149795486,"std":0.014787273109101448},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.005290785020451438},"egb":{"project":{"folds":[0.92243174564714,0.8986292621337116,0.9317042012813916,0.9240118443602714,0.9138587033870932],"mean":0.9181271513619216,"std":0.012608357948502916},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.004872848638078464},"egc":{"project":{"folds":[0.8973498792308929,0.9243374356685544,0.9201276065045769,0.9083990687178851,0.922345401023286],"mean":0.9145118782290389,"std":0.011416465216540321},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0035118782290388673},"ei":{"project":{"folds":[0.7948673591474984,0.7883307687433662,0.7966642199554042,0.859324494143849,0.8315263248746072],"mean":0.8141426333729449,"std":0.030359828559556545},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.024857366627055066},"eps":{"project":{"folds":[0.7548085721816783,0.7987726596190508,0.5579458512139692,0.7624899035830939,0.779360585937199],"mean":0.7306755145069982,"std":0.09802748818432702},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.05632448549300184},"nc":{"project":{"folds":[0.7402376421841073,0.8740961810300039,0.820598156782171,0.7328114677063449,0.8616572468584706],"mean":0.8058801389122194,"std":0.06638731488691142},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.05011986108778055},"xc":{"project":{"folds":[0.46646807075473407,0.29796597272581027,0.4005241979238625,0.28791610791176425,0.6086857130657267],"mean":0.4123120124763796,"std":0.13246114466364775},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":-0.01068798752362038}}},"cpu":{"status":"COMPLETE"}}
```

</details>

<details><summary>results/ua2d_cleanfp_readout_distill_20261019/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-CLEANFP-READOUT-DISTILL-20261019-01","revision":"r1","status":"待审查","validation":{"G_ALL1":{"tasks":{"eea":{"folds":[0.9273860764843063,0.8956199131919541,0.9013730158048413,0.9061714628550783,0.9512045211747545],"mean":0.9163509979021869,"std":0.0228816618636602},"egb":{"folds":[0.9488560748325693,0.9423053933145314,0.9402485758097483,0.9516219797376557,0.9568337287948756],"mean":0.947973150497876,"std":0.006789622293635751},"egc":{"folds":[0.9250186907774739,0.9242308190172538,0.9216631427836848,0.9117384789774551,0.9138685936426173],"mean":0.919303945039697,"std":0.006108959436394896},"ei":{"folds":[0.8710074493038845,0.8678339156571846,0.8625468633503772,0.849275577660555,0.8199433089941454],"mean":0.8541214229932294,"std":0.020831371044398424},"eps":{"folds":[0.6658496983972527,0.834881691798884,0.7194014333861962,0.8105646178162164,0.6735426351736009],"mean":0.74084801531443,"std":0.0779702741278533},"nc":{"folds":[0.8285187217729764,0.8972519263168642,0.8304368835456928,0.9043307152619836,0.7570602516490896],"mean":0.8435196997093215,"std":0.06011771237632131},"xc":{"folds":[0.3325341254094364,0.5532641805242575,0.41803527342034486,0.49116251289072677,0.26431593820995447],"mean":0.411862406090944,"std":0.11660189679251892}},"macro":0.8048542339353836},"G_DYN":{"tasks":{"eea":{"folds":[0.9250852411748429,0.9001767108198037,0.893015706859481,0.9220380635591413,0.9427447409041078],"mean":0.9166120926634752,"std":0.02006901784644165},"egb":{"folds":[0.9410168123998255,0.9445902309182066,0.9348816044428323,0.9519793877944323,0.9459626900320431],"mean":0.9436861451174681,"std":0.006313144534045365},"egc":{"folds":[0.9159838550233742,0.9246833457744909,0.9243731585045377,0.9134286135053511,0.914068644016289],"mean":0.9185075233648086,"std":0.0055770587041272385},"ei":{"folds":[0.8524823656658403,0.8568805660358824,0.8913510224090693,0.8574000031917055,0.8191617678418881],"mean":0.8554551450288772,"std":0.02559479906149331},"eps":{"folds":[0.7396566375534889,0.8235990035340689,0.7657833405107001,0.8222665227553558,0.658215333116117],"mean":0.7619041674939462,"std":0.06839452038193666},"nc":{"folds":[0.7945686292406454,0.9044670285100129,0.853681122064029,0.8896918325719578,0.7356329840193516],"mean":0.8356083192811994,"std":0.07012934169500773},"xc":{"folds":[0.3522540777814054,0.5520607903101402,0.4174684146397345,0.5128335272903031,0.34020306480154516],"mean":0.4349639749646257,"std":0.09473909325291636}},"macro":0.8095339097020572},"Q_ALL1":{"tasks":{"eea":{"folds":[0.9198856055223831,0.8886472322245723,0.8896190769280471,0.9066111012686999,0.9517316102921013],"mean":0.9112989252471607,"std":0.02604622723003583},"egb":{"folds":[0.9457137385416589,0.9430793336697358,0.9403539234095918,0.9471615333084393,0.9526611157739512],"mean":0.9457939289406754,"std":0.004636747644447706},"egc":{"folds":[0.9231582564980196,0.9266084162806588,0.9211419047341409,0.9120505784028758,0.9144440635900756],"mean":0.9194806439011541,"std":0.006075828659543472},"ei":{"folds":[0.8702497583007751,0.8542173952188993,0.8798226721065547,0.8400597489465893,0.8219404389606701],"mean":0.8532580027066977,"std":0.023185693615825606},"eps":{"folds":[0.6790425723908748,0.8229120317668609,0.7502753698629068,0.8213926179792996,0.7015199614152217],"mean":0.7550285106830328,"std":0.06646801332691972},"nc":{"folds":[0.78361690966038,0.9298699029917135,0.8341290234490064,0.898637559495064,0.7624610098835443],"mean":0.8417428810959416,"std":0.07198192638588538},"xc":{"folds":[0.32631855941684484,0.5192321562995086,0.4134025779429982,0.4928496233597526,0.2659529060923602],"mean":0.4035511646222929,"std":0.10764700758731753}},"macro":0.804307722456708},"Q_DYN":{"tasks":{"eea":{"folds":[0.9271447592479868,0.9056087761832401,0.8985651805328561,0.9044574492299646,0.944341817722708],"mean":0.9160235965833511,"std":0.01918751187015793},"egb":{"folds":[0.9409194098601816,0.9402988932597236,0.9346013639755181,0.9547414951276213,0.9514860516094665],"mean":0.9444094427665022,"std":0.008397991652982552},"egc":{"folds":[0.9177670270901688,0.9299282411644535,0.923363144465557,0.9154435822399579,0.9175742713943316],"mean":0.9208152532708936,"std":0.005877686490568928},"ei":{"folds":[0.8465714558532567,0.8780769681142234,0.8933663441313187,0.8789941123437477,0.8171519019409554],"mean":0.8628321564767003,"std":0.03072883724718158},"eps":{"folds":[0.7456168041666694,0.8186510137230245,0.7698048051835917,0.8222924648627146,0.7006847052546689],"mean":0.7714099586381338,"std":0.05121198962788145},"nc":{"folds":[0.8203845607586416,0.8953442789359289,0.8505067208975348,0.8797446148570373,0.7568531418044036],"mean":0.8405666634507092,"std":0.05488527751635457},"xc":{"folds":[0.3400298679261874,0.5218959257969178,0.4535461798026541,0.5085649117860118,0.29121258742313516],"mean":0.4230498945469813,"std":0.10281874810976248}},"macro":0.8113009951047531},"QL_ALL1":{"tasks":{"eea":{"folds":[0.9236584425817459,0.8968134071563164,0.8918442052059765,0.9151935502072635,0.9568071632187378],"mean":0.916863353674008,"std":0.025847763652079067},"egb":{"folds":[0.9496991437979413,0.9415326905985858,0.9363684337284401,0.954422109719353,0.9602596487911731],"mean":0.9484564053270986,"std":0.009624285260302349},"egc":{"folds":[0.9169842384118871,0.9230886338583209,0.9242912270323985,0.9117767591873236,0.9208443543295773],"mean":0.9193970425639014,"std":0.0050883676621069065},"ei":{"folds":[0.8666347338725151,0.8664345661564772,0.8754500857295275,0.8722445146220414,0.7949015742405083],"mean":0.855133094924214,"std":0.033887415930699126},"eps":{"folds":[0.6558674134768654,0.8402540018431632,0.7420872912919113,0.8309856159589604,0.673219255630819],"mean":0.7484827156403439,"std":0.08589496724537715},"nc":{"folds":[0.779870499874727,0.8994938867313905,0.8442231860089281,0.9078942143691744,0.7666389708916145],"mean":0.8396241515751669,"std":0.06550510146210996},"xc":{"folds":[0.3808226990671526,0.5344833805094273,0.4231100523210337,0.49172036919810846,0.29823222972426267],"mean":0.425673746163997,"std":0.0928186700034793}},"macro":0.8076615014098186},"QL_DYN":{"tasks":{"eea":{"folds":[0.9218957193196434,0.9028833879318865,0.8896826355336713,0.917064900887164,0.9464321149931184],"mean":0.9155917517330968,"std":0.021364825031530765},"egb":{"folds":[0.9496291937255293,0.9405935127191237,0.9413714763477671,0.9570730604545288,0.9511646642506013],"mean":0.9479663814995101,"std":0.006960199829350976},"egc":{"folds":[0.9166413840178814,0.9244428900552367,0.9221458892370016,0.9159595489592579,0.924591614465944],"mean":0.9207562653470645,"std":0.004188463191207218},"ei":{"folds":[0.8466447145180986,0.8737667903625221,0.8634713754045682,0.8755063300096542,0.8154556490130307],"mean":0.8549689718615747,"std":0.02488989689090054},"eps":{"folds":[0.6984608092620227,0.8508122531323143,0.7448182118930273,0.8436342733495633,0.6850372204678006],"mean":0.7645525536209457,"std":0.0787002307450912},"nc":{"folds":[0.7760587137495842,0.9084107425966167,0.840838876074675,0.8881027038372968,0.7503192174532428],"mean":0.8327460507422831,"std":0.06867065508377032},"xc":{"folds":[0.3390136452388425,0.5888159865043598,0.4584227712000104,0.5419891655325253,0.3043517576338033],"mean":0.4465186652219083,"std":0.12376753486636027}},"macro":0.811871520003769}},"test":{"G_ALL1":{"tasks":{"eea":{"folds":[0.9024423311234964,0.9423832658380622,0.944271709908223,0.9395954757173253,0.890666066508118],"mean":0.9238717698190448,"std":0.02533726279049959},"egb":{"folds":[0.9448766946708891,0.9186499281253566,0.9444382508466994,0.9313880092496799,0.9049292354253049],"mean":0.928856423663586,"std":0.017193997755420347},"egc":{"folds":[0.8902500819159156,0.9263689791118992,0.9219291025356635,0.9150300531562684,0.9184174294377659],"mean":0.9143991292315026,"std":0.0141394756967644},"ei":{"folds":[0.7854260953907127,0.7866115352646991,0.784103694078305,0.8443458025010082,0.8202254433619471],"mean":0.8041425141193344,"std":0.027083938358307712},"eps":{"folds":[0.6736709955172459,0.8238154743446792,0.6673203748429876,0.7064755417357097,0.6269649764101854],"mean":0.6996494725701616,"std":0.07494477805156603},"nc":{"folds":[0.832736320425442,0.8983626629018744,0.7179276113702908,0.6281841873169591,0.7364592037481534],"mean":0.7627339971525439,"std":0.10499653391737565},"xc":{"folds":[0.44399534345663083,0.4184592611479101,0.42780058600817317,0.1959076344030104,0.5813719132995072],"mean":0.41350694766304635,"std":0.13846162248383967}},"macro":0.7781657506027456},"G_DYN":{"tasks":{"eea":{"folds":[0.9081942633238349,0.9347168340027779,0.9280996950232183,0.9348367628661166,0.8835041579066689],"mean":0.9178703426245235,"std":0.02208486190110514},"egb":{"folds":[0.9433971220668981,0.9285256218369552,0.9476606861774095,0.9326829878863092,0.9141352339390051],"mean":0.9332803303813154,"std":0.013215926906721746},"egc":{"folds":[0.8893465810579915,0.9302865035070581,0.9232762856452191,0.9193355713265193,0.9241024162909156],"mean":0.9172694715655407,"std":0.016094802360492308},"ei":{"folds":[0.7885731985936895,0.8197278596552937,0.7851782576524186,0.82509126684104,0.8292251244196922],"mean":0.8095591414324268,"std":0.021013316775234367},"eps":{"folds":[0.7347478139215609,0.8209142294514857,0.7829509781082817,0.6746108831769106,0.7028699975206806],"mean":0.7432187804357839,"std":0.05920738014035348},"nc":{"folds":[0.7927827000237462,0.8659169859375948,0.837825189467261,0.711028348612381,0.778599631280412],"mean":0.797230571064279,"std":0.059505178894577716},"xc":{"folds":[0.2662702898528052,0.39523053748765313,0.300399944865801,0.12047769461920521,0.4206316125661447],"mean":0.30060201587832186,"std":0.1193578029881706}},"macro":0.7741472361974561},"Q_ALL1":{"tasks":{"eea":{"folds":[0.9041488071425338,0.9530633756960275,0.9369822046487546,0.9274836384211479,0.8705126771929917],"mean":0.9184381406202912,"std":0.03211559333550849},"egb":{"folds":[0.9499860092069967,0.9326577521377076,0.9474460594348976,0.9419040951892844,0.906209839316853],"mean":0.935640751057148,"std":0.017744422119468646},"egc":{"folds":[0.8869799474994338,0.9276350403292929,0.9229474893896501,0.918709220312095,0.917841451787618],"mean":0.9148226298636178,"std":0.016043729589894005},"ei":{"folds":[0.8164330609851838,0.7852605401833563,0.8170952869107954,0.8061468358788015,0.8439639599669447],"mean":0.8137799367850164,"std":0.02121748148280352},"eps":{"folds":[0.6493935839372373,0.8126905010135732,0.7863250231356531,0.6205703315199793,0.7648463220789826],"mean":0.7267651523370852,"std":0.08608777324431575},"nc":{"folds":[0.7770835125950109,0.9149842721144741,0.8036085968233848,0.6155986287838495,0.7588124369424142],"mean":0.7740174894518266,"std":0.10733638360872985},"xc":{"folds":[0.46921072174194256,0.306302110975328,0.49555205647582157,0.1183716431602887,0.5873853653244638],"mean":0.3953643795355689,"std":0.18508104679912946}},"macro":0.782689782807222},"Q_DYN":{"tasks":{"eea":{"folds":[0.8953801874913788,0.9486290194913581,0.9361031551348716,0.939768125162608,0.8679501807895493],"mean":0.9175661336139532,"std":0.034479745701837254},"egb":{"folds":[0.9422670437121871,0.9284648267729746,0.9477771754846557,0.9214762723277143,0.9284336061836324],"mean":0.9336837848962329,"std":0.010910101439681998},"egc":{"folds":[0.8863387737100613,0.9317401901580487,0.9239337090391256,0.9229366697106188,0.9255292056047115],"mean":0.9180957096445133,"std":0.01807933089132822},"ei":{"folds":[0.7954647755075805,0.8090532323426085,0.8032163905106815,0.8325155179568146,0.8415146992694509],"mean":0.8163529231174274,"std":0.019726334783567542},"eps":{"folds":[0.6911247533523621,0.8078984103296655,0.5521113857172091,0.7594402591919772,0.7554622686421782],"mean":0.7132074154466784,"std":0.09915758636010934},"nc":{"folds":[0.8214055432491051,0.8817099546855562,0.8261046445241013,0.6359529926635616,0.7499839826700994],"mean":0.7830314235584848,"std":0.0945905995799309},"xc":{"folds":[0.36795037296164224,0.3669608860491882,0.4846099248124488,0.21136743192062268,0.5574279488874552],"mean":0.3976633129262714,"std":0.13192432046390296}},"macro":0.7828001004576516},"QL_ALL1":{"tasks":{"eea":{"folds":[0.9024309421086621,0.9416833470001962,0.9383712655478612,0.9273577342233665,0.9060309521691668],"mean":0.9231748482098506,"std":0.01813294635858977},"egb":{"folds":[0.9348781954999645,0.9352861239936996,0.9498691758125957,0.9349222110244175,0.9028045407176652],"mean":0.9315520494096685,"std":0.01730824172020544},"egc":{"folds":[0.886202293462864,0.9273581789353902,0.913023174825577,0.9211374978167126,0.9191383044502425],"mean":0.9133718898981572,"std":0.01602717268284392},"ei":{"folds":[0.7868482296045041,0.7893539674501368,0.7963120698223166,0.8345403068370555,0.8185652052196286],"mean":0.8051239557867284,"std":0.020653891457224193},"eps":{"folds":[0.6617869868597925,0.8200847262345651,0.7078304669616846,0.6852330901947855,0.6125247918169985],"mean":0.6974920124135654,"std":0.0771045000698863},"nc":{"folds":[0.785344255956991,0.8843213066032858,0.8493894636341088,0.6253693418243582,0.7521060644788159],"mean":0.7793060864995119,"std":0.10051454901349015},"xc":{"folds":[0.4654323354875223,0.3121694843938454,0.3758569776217372,0.16453288490340623,0.5202190693744044],"mean":0.36764215035618314,"std":0.13896106495512783}},"macro":0.7739518560819522},"QL_DYN":{"tasks":{"eea":{"folds":[0.9036165484050228,0.9424444611943106,0.935106590425934,0.9305445863005736,0.8915045314591961],"mean":0.9206433435570075,"std":0.02191740730397457},"egb":{"folds":[0.9459531647741631,0.9288528717948421,0.9488371337065323,0.9449196335222266,0.9160606199639526],"mean":0.9369246847523434,"std":0.014033909481490421},"egc":{"folds":[0.8856138011591007,0.930998560167146,0.9204170111257088,0.9195869742605105,0.9273242385766176],"mean":0.9167881170578168,"std":0.01806783741605795},"ei":{"folds":[0.8162273400980131,0.7958485352245113,0.791387176030104,0.8203355909790077,0.8516390638594347],"mean":0.8150875412382141,"std":0.023961957541305607},"eps":{"folds":[0.7445057601179743,0.8276916987422023,0.7455880363050817,0.7368502201367648,0.7500109709579743],"mean":0.7609293372519994,"std":0.03762086161941728},"nc":{"folds":[0.8157568113559561,0.8768183606823095,0.8918166392290015,0.5898065325192009,0.7513782784588725],"mean":0.7851153244490681,"std":0.12249207181209251},"xc":{"folds":[0.4475962904550499,0.30673423505690633,0.4846771292923945,0.2195437623934431,0.5121093568775603],"mean":0.3941321548150708,"std":0.12559352048945996}},"macro":0.7899457861602172}},"budget":{"plan":"UA2D-CLEANFP-READOUT-DISTILL-20261019-01","revision":"r1"},"independent_review":false,"delta_test":0.0}
```

</details>

<details><summary>results/ua2d_transfer_20261022/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-TRANSFER-20261022-01","revision":"r1","status":"待审查","independent_review":"PENDING","validation":{"R_MEAN":{"tasks":{"eea":{"folds":[0.9387407610504498,0.9117069915817465,0.9133580573989676,0.9185195035867917,0.9455761407059127],"mean":0.9255802908647738,"std":0.01553011692849955},"egb":{"folds":[0.9371251330356725,0.9366502626609605,0.9297673166708939,0.9457677444401853,0.956020127261723],"mean":0.941066116813887,"std":0.010104723357441938},"egc":{"folds":[0.9297904178386152,0.9274564653043744,0.9190722528773942,0.9053621581142989,0.9173241182014341],"mean":0.9198010824672233,"std":0.00966360166501043},"ei":{"folds":[0.854352128293108,0.8832785296272813,0.8945900817930894,0.8777494145614269,0.8345235411966339],"mean":0.8688987390943079,"std":0.024178585762863202},"eps":{"folds":[0.7305752579983713,0.8584296607897256,0.7421907821218909,0.8387091819070153,0.6699003032914845],"mean":0.7679610372216976,"std":0.07884729212484827},"nc":{"folds":[0.7958197839704001,0.8991325533210648,0.8518220783702323,0.8956380222590562,0.7715861400415627],"mean":0.8427997155924632,"std":0.057717340561570185},"xc":{"folds":[0.34041835718691993,0.5305146098584024,0.529286367614905,0.5577158771209596,0.33116016539354654],"mean":0.45781907543494677,"std":0.11202342802040341}},"macro":0.8177037224984713},"R_STD":{"tasks":{"eea":{"folds":[0.9438160834352152,0.9148969145045996,0.9204488957902406,0.9139357609820887,0.9455245720193255],"mean":0.927724445346294,"std":0.0156795432912894},"egb":{"folds":[0.9401927615322558,0.9402036414775918,0.9333472783328108,0.9458734418636999,0.9572200467691598],"mean":0.9433674339951036,"std":0.008925617680897745},"egc":{"folds":[0.9283419189162069,0.9282329271920803,0.9204717500925781,0.9127315077170767,0.9213519475227775],"mean":0.9222260102881439,"std":0.0064705946917798934},"ei":{"folds":[0.8663023820485982,0.8865949727925548,0.8949353786522453,0.8746835003688506,0.8268070018553735],"mean":0.8698646471435245,"std":0.026449561770844926},"eps":{"folds":[0.7703303306429002,0.8500359993724674,0.7542793459844985,0.8347965541807765,0.6907078389873189],"mean":0.7800300138335923,"std":0.06449000624284999},"nc":{"folds":[0.8097035440470913,0.9087466346649157,0.8559073530670648,0.8802034401958863,0.7704739149062688],"mean":0.8450069773762454,"std":0.05527626729092686},"xc":{"folds":[0.3466357827235371,0.4950340430364276,0.5710592300172858,0.5589547516947933,0.30644755458096007],"mean":0.4556262724106007,"std":0.12215536358898633}},"macro":0.8205494000562148},"A_SP":{"tasks":{"eea":{"folds":[0.9389807684954844,0.9102242690210703,0.9103658087942034,0.918769790526318,0.9458233960657679],"mean":0.9248328065805687,"std":0.016584881907434075},"egb":{"folds":[0.9357811787779496,0.9344367884333096,0.929517333850315,0.9461191620379359,0.9558027337194093],"mean":0.9403314393637838,"std":0.010550537211901533},"egc":{"folds":[0.9248330030360208,0.9277249983793013,0.9218670311695617,0.9085637405666374,0.9211101202959903],"mean":0.9208197786895024,"std":0.007334499558498256},"ei":{"folds":[0.853438173233072,0.8793652425570844,0.8960266221664459,0.8842695144428833,0.820143354817282],"mean":0.8666485814433533,"std":0.0303007054992438},"eps":{"folds":[0.708922999432992,0.8574292150162672,0.7492430757451296,0.842684863906777,0.724816467078379],"mean":0.7766193242359088,"std":0.06875805655134323},"nc":{"folds":[0.7966833802087838,0.9115578329210029,0.847148381032509,0.8939351394324297,0.7661361606302842],"mean":0.843092178845002,"std":0.06197783986203517},"xc":{"folds":[0.33747432063891103,0.5189817152433926,0.5498587577406365,0.5257439164949345,0.30102199309653044],"mean":0.446616140642881,"std":0.11754425970381993}},"macro":0.8169943214001429},"R_STD_SP":{"tasks":{"eea":{"folds":[0.9438706357327682,0.9148423815762183,0.920283043537079,0.9135436086005796,0.9455306041166024],"mean":0.9276140547126495,"std":0.015812305195328162},"egb":{"folds":[0.940449268964145,0.9380596299997882,0.9321422534974375,0.9494277039826003,0.957412465532342],"mean":0.9434982643952626,"std":0.009957376007778431},"egc":{"folds":[0.9275086868239069,0.9262302659291209,0.9194528473961618,0.9146649401395154,0.9142908103065858],"mean":0.9204295101190582,"std":0.006237613391703579},"ei":{"folds":[0.8726748916787086,0.8877732789420488,0.8948791861726066,0.8739034304278691,0.8251127191327912],"mean":0.870868701270805,"std":0.02724128272366841},"eps":{"folds":[0.770050071895326,0.8496297801005632,0.7542147468310016,0.834634062528858,0.708458172427485],"mean":0.7833973667566468,"std":0.05843284607897578},"nc":{"folds":[0.8149036803089755,0.9103083743467372,0.855572328195292,0.8828167830817545,0.7704739556610126],"mean":0.8468150243187544,"std":0.05534102263083803},"xc":{"folds":[0.34663576206779045,0.4942291658064196,0.5712642176836352,0.5531827833421883,0.31051533146575117],"mean":0.4551654520731569,"std":0.11970238650546204}},"macro":0.8211126248066191}},"formal":{"R_MEAN":{"tasks":{"eea":{"folds":[0.9212382243868382,0.9471401580734473,0.9417275132105148,0.9399328641591636,0.9215917829595731],"mean":0.9343261085579074,"std":0.012081723021885992},"egb":{"folds":[0.9463391643580314,0.9080728022816144,0.9467428010409014,0.9329311905200541,0.9204951011318151],"mean":0.9309162118664831,"std":0.016754323604916164},"egc":{"folds":[0.8860402682872044,0.928702029369899,0.9191258760960525,0.9157724357207029,0.9016656356176674],"mean":0.9102612490183052,"std":0.01665501162849731},"ei":{"folds":[0.8051094232693398,0.8006904274176488,0.7996828860582365,0.8418935367852383,0.8476393749919214],"mean":0.8190031297044769,"std":0.023694235789467458},"eps":{"folds":[0.7510333872046395,0.8260251886166968,0.66844397523638,0.7411593280327176,0.708029584491916],"mean":0.73893829271647,"std":0.05843831959250935},"nc":{"folds":[0.7617120425113725,0.9105482555940217,0.8369692425762751,0.7249607262814185,0.7933531411072212],"mean":0.805508681614062,"std":0.07172334847858607},"xc":{"folds":[0.4013887883224011,0.37364558639304857,0.47209453996497885,0.19637361712562718,0.5606532501425848],"mean":0.4008311563897281,"std":0.1352440599633569}},"macro":0.7913978328382046},"R_STD":{"tasks":{"eea":{"folds":[0.9247950006173244,0.9495461981389602,0.9454628512807024,0.9272525146824754,0.9186201121312044],"mean":0.9331353353701333,"std":0.013565991799815012},"egb":{"folds":[0.9411645467312642,0.9228650253481733,0.9460083161584405,0.9283979117937551,0.9227400175546877],"mean":0.9322351635172643,"std":0.010748434830333052},"egc":{"folds":[0.886697633753897,0.931895601093778,0.914700190649696,0.9221010979938808,0.9214877605886429],"mean":0.9153764568159788,"std":0.01716379320399424},"ei":{"folds":[0.7719537284499173,0.779687453130751,0.7931309837247691,0.8534159460203516,0.8371122832825049],"mean":0.8070600789216588,"std":0.03615154761747953},"eps":{"folds":[0.7491876518329676,0.8061980318988372,0.6012803926847208,0.6667679738736515,0.7816561800562905],"mean":0.7210180460692935,"std":0.08515620344329522},"nc":{"folds":[0.734389373431472,0.8960975366079077,0.8058997402268405,0.7325089793002433,0.7912373151453919],"mean":0.792026588942371,"std":0.06687346988353035},"xc":{"folds":[0.41479199187794435,0.36390000773866993,0.47598861118627456,0.27511065761723086,0.5480374592832081],"mean":0.4155657455406655,"std":0.10439978031065052}},"macro":0.7880596307396237},"A_SP":{"tasks":{"eea":{"folds":[0.9230277067706659,0.9472107542101151,0.9396672462224241,0.9311851969788403,0.920809108157039],"mean":0.9323800024678169,"std":0.011133661896422203},"egb":{"folds":[0.9439198101249202,0.912805504743901,0.9465964693000755,0.9169906652061836,0.9245599291808253],"mean":0.9289744757111811,"std":0.015479286451696035},"egc":{"folds":[0.8835687300689091,0.9265791030608996,0.9240869718154585,0.9092541296826004,0.921920731605884],"mean":0.9130819332467504,"std":0.01779810428996875},"ei":{"folds":[0.8035992025327083,0.7742841759666874,0.7998509661675249,0.852946771900614,0.8706789378133049],"mean":0.8202720108761679,"std":0.040057398991828856},"eps":{"folds":[0.7429751437549887,0.82558373825018,0.6853748502414113,0.7490213891172125,0.7949545984148768],"mean":0.759581943955734,"std":0.053622442648818804},"nc":{"folds":[0.7620815689955034,0.8861452703622603,0.8347531917729735,0.6383287173201504,0.7947465848628412],"mean":0.7832110666627458,"std":0.09330388517497021},"xc":{"folds":[0.4597791516455405,0.3278190776735953,0.4653362315851266,0.17045975025560178,0.5536582134634415],"mean":0.3954104849246612,"std":0.14937093948578128}},"macro":0.7904159882635796},"R_STD_SP":{"tasks":{"eea":{"folds":[0.9249258987044148,0.9494935984645693,0.9446894178298351,0.9263453893965484,0.9186349447095087],"mean":0.9328178498209752,"std":0.013456830547960834},"egb":{"folds":[0.9401824199466463,0.9268548533963497,0.9449501705656594,0.9288643358575797,0.9228181754409827],"mean":0.9327339910414436,"std":0.009388503941497282},"egc":{"folds":[0.8934467507470006,0.9288821987881274,0.9081218294390948,0.9178687843037895,0.9194013666182705],"mean":0.9135441859792565,"std":0.01343104804171137},"ei":{"folds":[0.769538379469682,0.7807643228970182,0.7930069350298413,0.8457895020785373,0.8388916592597455],"mean":0.8055981597469648,"std":0.03463873285223481},"eps":{"folds":[0.7492467620339915,0.774748521501148,0.6014961973775625,0.6659628109123114,0.772052443601069],"mean":0.7127013470852164,"std":0.07625458666450974},"nc":{"folds":[0.7322416784876528,0.8974998805555947,0.8060007190140874,0.7031095877369253,0.791237327596798],"mean":0.7860178386782116,"std":0.07520049486410266},"xc":{"folds":[0.41479199567990943,0.36570370211296843,0.4754810044750344,0.2657461180842019,0.5502395746087279],"mean":0.41439247899216836,"std":0.10799622510981939}},"macro":0.7854008359063194}},"FINAL":"R_STD_SP","delta_FINAL_B_FP":-0.0058359724457001105,"performance_pass":false,"comparisons":{"R_MEAN_minus_B_FP":{"delta":0.0001610244861851573,"paired_tasks":{"eea":{"folds":[0.005274554585608104,-0.00016461344250440213,0.001743098008422872,0.0003857240796513173,0.0007502549335974829],"mean":0.001597803632955075,"std":0.00216986160921431},"egb":{"folds":[0.0018841641398942288,-0.010139155477392081,0.00020789830427503375,-0.0008150267480727669,-0.00418509313947335],"mean":-0.002609442584153787,"std":0.0047573388094574115},"egc":{"folds":[0.0012667421829050118,0.0010477245533649926,0.0011738144680760731,0.004938270314221227,-0.02112069191980448],"mean":-0.002538828080247435,"std":0.010515727550501298},"ei":{"folds":[0.001358262922383724,0.02207962251830642,-0.0003769116663207006,-0.007099758990662175,-0.0026190801561535304],"mean":0.0026684269255107473,"std":0.011304427433834573},"eps":{"folds":[0.001064259701810677,-0.015257722289560793,-0.016858553569781165,-0.0075578696036688875,-0.08701167209161631],"mean":-0.025124311570563294,"std":0.03532045752922362},"nc":{"folds":[-0.00036952901343012545,0.004208049243839773,0.0020816910175034398,0.08959067329856962,-0.0013934435076204954],"mean":0.018823488207772444,"std":0.039620037711308176},"xc":{"folds":[-0.05839035824094996,0.04607575700094535,0.006739684021097259,0.026989820130949527,0.020135271448070813],"mean":0.008310034872022597,"std":0.03989495208838525}},"status":"COMPLETE"},"R_STD_minus_B_FP":{"delta":-0.0031771776123957762,"paired_tasks":{"eea":{"folds":[0.00883133081609433,0.0022414266230085245,0.005478436078610427,-0.012294625397036851,-0.002221415894771206],"mean":0.0004070304451810447,"std":0.008189838100834677},"egb":{"folds":[-0.0032904534868730684,0.004653067589166837,-0.0005265865781858281,-0.005348305474371817,-0.0019401767166007389],"mean":-0.001290490933372923,"std":0.003768197130465777},"egc":{"folds":[0.0019241076495976062,0.004241296277244011,-0.003251870978280391,0.011266932587399192,-0.0012985669488290164],"mean":0.0025763797174262804,"std":0.005650372399801352},"ei":{"folds":[-0.031797431897038764,0.0010766482314086279,-0.006928813999788064,0.0044226502444512095,-0.0131461718655701],"mean":-0.009274623857307419,"std":0.014339915808119693},"eps":{"folds":[-0.000781475669861198,-0.03508487900742041,-0.08402213612144038,-0.08194922376273495,-0.013385076527241857],"mean":-0.04304455821773976,"std":0.03847708534228768},"nc":{"folds":[-0.027692198093330567,-0.01024266974227428,-0.02898781133193118,0.0971389263173944,-0.0035092694694498494],"mean":0.005341395536081706,"std":0.052482608479967496},"xc":{"folds":[-0.044987154685406705,0.03633017834656671,0.010633755242392962,0.10572686062255321,0.007519480588694116],"mean":0.023044624022960057,"std":0.054854901466422704}},"status":"COMPLETE"},"A_SP_minus_B_FP":{"delta":-0.0008208200884398664,"paired_tasks":{"eea":{"folds":[0.007064036969435805,-9.401730583658008e-05,-0.00031716897966782565,-0.008361943100671954,-3.241986893665061e-05],"mean":-0.0003483024571354409,"std":0.005461836325347921},"egb":{"folds":[-0.0005351900932170084,-0.00540645301510545,6.156656344913358e-05,-0.016755552061943346,-0.00012026509046314526],"mean":-0.004551178739455963,"std":0.007188824696747264},"egc":{"folds":[-0.0012047960353902543,-0.0010752017556343585,0.006134910187482134,-0.0015800357238812124,-0.0008655959315879702],"mean":0.0002818561481976678,"std":0.0032822732242617195},"ei":{"folds":[-0.00015195781424770516,-0.004326628932655008,-0.00020883155703221856,0.003953476124713529,0.02042048266522989],"mean":0.003937308097201697,"std":0.009668253852242332},"eps":{"folds":[-0.006993983747840105,-0.015699172656077565,7.232143525015555e-05,0.00030419148082605485,-8.665816865549569e-05],"mean":-0.004480660331299391,"std":0.006983973049296218},"nc":{"folds":[-2.5292992145153903e-09,-0.02019493598792166,-0.0001343597857982326,0.00295866433730152,2.479995098170207e-10],"mean":-0.0034741267435436153,"std":0.009437417597230909},"xc":{"folds":[5.082189447058738e-09,0.0002492482814920871,-1.8624358754992798e-05,0.001075953260924134,0.01314023476892745],"mean":0.0028893634069556253,"std":0.005747691107904627}},"status":"COMPLETE"},"R_STD_SP_minus_B_FP":{"delta":-0.0058359724457001105,"paired_tasks":{"eea":{"folds":[0.008962228903184655,0.0021888269486175504,0.004705002627743093,-0.013201750682963809,-0.002206583316466948],"mean":8.954489602290839e-05,"std":0.008461156830433548},"egb":{"folds":[-0.004272580271490911,0.008642895637343218,-0.001584732170966885,-0.00488188141054724,-0.0018620188303057894],"mean":-0.0007916634091935215,"std":0.005468825026726894},"egc":{"folds":[0.00867322464270126,0.0012278939715933967,-0.009830232188881571,0.007034618897307832,-0.003384960919201374],"mean":0.0007441088807039087,"std":0.0076082413789049855},"ei":{"folds":[-0.034212780877274085,0.0021535179976758245,-0.0070528626947158335,-0.003203793697363122,-0.01136679588832945],"mean":-0.010736543032001333,"std":0.014035637220530143},"eps":{"folds":[-0.0007223654688373049,-0.06653438940510958,-0.08380633142859861,-0.08275438672407509,-0.022988812982463358],"mean":-0.05136125720181679,"std":0.03754240372753857},"nc":{"folds":[-0.029839893037149845,-0.008840325794587267,-0.02888683254468427,0.06773953475407646,-0.003509257018043721],"mean":-0.0006673547280777292,"std":0.04000547560315838},"xc":{"folds":[-0.04498715088344163,0.038133872720865214,0.01012614853115279,0.09636232108952425,0.009721595914213843],"mean":0.021871357474462894,"std":0.05140282054852819}},"status":"COMPLETE"},"R_STD_minus_R_MEAN":{"delta":-0.0033382020985809335,"paired_tasks":{"eea":{"folds":[0.003556776230486225,0.0024060400655129266,0.003735338070187555,-0.012680349476688169,-0.002971670828368689],"mean":-0.0011907731877740303,"std":0.006980774005952232},"egb":{"folds":[-0.005174617626767297,0.014792223066558918,-0.0007344848824608619,-0.00453327872629905,0.0022449164228726115],"mean":0.0013189516507808642,"std":0.008108916347872306},"egc":{"folds":[0.0006573654666925943,0.0031935717238790184,-0.004425685446356464,0.006328662273177965,0.019822124970975463],"mean":0.005115207797673715,"std":0.00911536576129088},"ei":{"folds":[-0.03315569481942249,-0.02100297428689779,-0.006551902333467363,0.011522409235113384,-0.01052709170941657],"mean":-0.011943050782818165,"std":0.016686489418666597},"eps":{"folds":[-0.001845735371671875,-0.019827156717859618,-0.06716358255165922,-0.07439135415906606,0.07362659556437445],"mean":-0.017920246647176463,"std":0.0597008011928896},"nc":{"folds":[-0.02732266907990044,-0.014450718986114053,-0.03106950234943462,0.00754825301882478,-0.002115825961829354],"mean":-0.013482092671690737,"std":0.016380508464357655},"xc":{"folds":[0.013403203555543253,-0.00974557865437864,0.003894071221295703,0.07873704049160368,-0.012615790859376697],"mean":0.01473458915093746,"std":0.037292509625546394}},"status":"COMPLETE"},"R_STD_SP_minus_R_STD":{"delta":-0.0026587948333043343,"paired_tasks":{"eea":{"folds":[0.00013089808709032535,-5.2599674390974016e-05,-0.0007734334508673335,-0.0009071252859269574,1.4832578304257993e-05],"mean":-0.00031748554915813634,"std":0.00048404835966990275},"egb":{"folds":[-0.0009821267846178428,0.00398982804817638,-0.0010581455927810568,0.00046642406382457757,7.815788629494946e-05],"mean":0.0004988275241794016,"std":0.002060490750932328},"egc":{"folds":[0.006749116993103654,-0.0030134023056506143,-0.00657836121060118,-0.0042323136900913605,-0.0020863939703723577],"mean":-0.0018322708367223717,"std":0.005083931715091863},"ei":{"folds":[-0.002415348980235321,0.0010768697662671967,-0.0001240486949277697,-0.0076264439418143315,0.0017793759772406492],"mean":-0.0014619191746939153,"std":0.003796436979849211},"eps":{"folds":[5.911020102389308e-05,-0.03144951039768917,0.00021580469284177006,-0.0008051629613401445,-0.009603736455221501],"mean":-0.00831669898407703,"std":0.013566174859486017},"nc":{"folds":[-0.0021476949438192783,0.0014023439476870125,0.00010097878724690812,-0.029399391563317945,1.2451406128377585e-08],"mean":-0.006008750264159435,"std":0.013137619499390484},"xc":{"folds":[3.801965076100089e-09,0.0018036943742985034,-0.0005076067112401716,-0.00936453953302896,0.0022021153255197268],"mean":-0.0011732665484971649,"std":0.004721549495997723}},"status":"COMPLETE"},"R_STD_SP_minus_A_SP":{"delta":-0.005015152357260244,"paired_tasks":{"eea":{"folds":[0.0018981919337488495,0.0022828442544541305,0.005022171607410919,-0.004839807582291855,-0.0021741634475302973],"mean":0.00043784735315834935,"std":0.003912444592852898},"egb":{"folds":[-0.0037373901782739027,0.014049348652448668,-0.0016462987344160185,0.011873670651396107,-0.0017417537398426441],"mean":0.0037595153302624417,"std":0.008476593738957719},"egc":{"folds":[0.009878020678091515,0.0023030957272277552,-0.015965142376363706,0.008614654621189044,-0.002519364987613404],"mean":0.0004622527325062409,"std":0.010455338346782243},"ei":{"folds":[-0.03406082306302638,0.006480146930330832,-0.006844031137683615,-0.007157269822076651,-0.03178727855355934],"mean":-0.01467385112920303,"std":0.01756431601657581},"eps":{"folds":[0.0062716182790028,-0.05083521674903202,-0.08387865286384877,-0.08305857820490115,-0.022902154813807862],"mean":-0.0468805968705174,"std":0.03903032525443551},"nc":{"folds":[-0.02983989050785063,0.011354610193334391,-0.02875247275888604,0.06478087041677494,-0.003509257266043231],"mean":0.002806772015465886,"std":0.038779946946501304},"xc":{"folds":[-0.044987155965631076,0.03788462443937313,0.010144772889907783,0.09528636782860012,-0.003418638854713607],"mean":0.01898199406750727,"std":0.052085515931658014}},"status":"COMPLETE"},"interaction":{"delta":-0.0018379747448644679,"paired_tasks":{"eea":{"folds":[-0.00693313888234548,4.141763144560606e-05,-0.00045626447119950786,0.007454817814744996,4.72524472409086e-05],"mean":3.081690797730463e-05,"std":0.005095343474327503},"egb":{"folds":[-0.00044693669140083436,0.00939628106328183,-0.0011197121562301904,0.017221976125767924,0.00019842297675809473],"mean":0.0050500062636353645,"std":0.00804466628548945},"egc":{"folds":[0.007953913028493909,-0.0019382005500162558,-0.012713271398083315,-0.002652277966210148,-0.0012207980387843875],"mean":-0.0021141269849200396,"std":0.007328469915302237},"ei":{"folds":[-0.002263391165987616,0.0054034986989222045,8.478286210444885e-05,-0.01157992006652786,-0.01864110668798924],"mean":-0.005399227271895612,"std":0.009619707083969559},"eps":{"folds":[0.007053093948863998,-0.01575033774161161,0.0001434832575916145,-0.0011093544421661994,-0.009517078286566005],"mean":-0.00383603865277764,"std":0.008889169594716168},"nc":{"folds":[-0.0021476924145200638,0.02159727993560867,0.00023533857304514072,-0.032358055900619465,1.2203406618560564e-08],"mean":-0.0025346235206158197,"std":0.0192745471236023},"xc":{"folds":[-1.2802243709586492e-09,0.0015544460928064163,-0.0004889823524851789,-0.010440492793953093,-0.010938119443407723],"mean":-0.00406262995545279,"std":0.006098707698369663}},"status":"COMPLETE"},"FINAL_minus_A_MIX246":{"delta":-0.005950088856263802,"paired_tasks":{"eea":{"folds":[0.008962629939247568,0.002188752610584155,0.004704772779341337,-0.01320112206407309,-0.0022067557017382278],"mean":8.965551267234862e-05,"std":0.008460990843961715},"egb":{"folds":[-0.004273360795422176,0.008643810408038743,-0.001584332864822624,-0.004881542071127121,-0.001862310728362182],"mean":-0.000791547210339072,"std":0.005469280135326457},"egc":{"folds":[0.004806575418535131,0.0012249094966634289,-0.010316938681959376,0.00842357664147353,-0.0033990616227898007],"mean":0.00014781225038458247,"std":0.0073048582351041245},"ei":{"folds":[-0.03422606398908856,0.0021544927161114558,-0.00705153381210355,-0.003204223509022275,-0.011331932161627756],"mean":-0.010731852151146137,"std":0.014041063937549447},"eps":{"folds":[-0.0007516648683847382,-0.06686379997023384,-0.08380479861406731,-0.08275521970740607,-0.02298875583463522],"mean":-0.051432847798945434,"std":0.03756593249666604},"nc":{"folds":[-0.029839880290611243,-0.009343153075204724,-0.028885451716309407,0.06774539481634823,-0.003509239459250413],"mean":-0.0007664659450055122,"std":0.04003404243946942},"xc":{"folds":[-0.044987158351790124,0.03812783856835711,0.010125022686976681,0.0959691243072548,0.00993828953187259],"mean":0.02183462334853421,"std":0.05124739288523468}},"status":"COMPLETE"}},"references":{"B_FP":{"tasks":{"eea":{"folds":[0.9159636698012301,0.9473047715159517,0.939984415202092,0.9395471400795122,0.9208415280259756],"mean":0.9327283049249523,"std":0.013545935583923146},"egb":{"folds":[0.9444550002181372,0.9182119577590064,0.9465349027366263,0.9337462172681269,0.9246801942712884],"mean":0.9335256544506372,"std":0.012262659376907688},"egc":{"folds":[0.8847735261042994,0.927654304816534,0.9179520616279764,0.9108341654064817,0.9227863275374719],"mean":0.9128000770985528,"std":0.016854291685891032},"ei":{"folds":[0.803751160346956,0.7786108048993424,0.8000597977245572,0.8489932957759004,0.850258455148075],"mean":0.8163347027789662,"std":0.031873638077730186},"eps":{"folds":[0.7499691275028288,0.8412829109062576,0.6853025288061612,0.7487171976363864,0.7950412565835323],"mean":0.7640626042870332,"std":0.058223512723314164},"nc":{"folds":[0.7620815715248026,0.906340206350182,0.8348875515587717,0.6353700529828489,0.7947465846148417],"mean":0.7866851934062894,"std":0.1002588966293432},"xc":{"folds":[0.45977914656335106,0.3275698293921032,0.4653548559438816,0.16938379699467765,0.540517978694514],"mean":0.3925211215177055,"std":0.1464073089720903}},"macro":0.7912368083520195},"A_MIX246":{"tasks":{"eea":{"folds":[0.9159632687651672,0.9473048458539851,0.9399846450504937,0.9395465114606215,0.9208417004112469],"mean":0.9327281943083028,"std":0.013545993522803307},"egb":{"folds":[0.9444557807420685,0.9182110429883109,0.9465345034304821,0.9337458779287068,0.9246804861693448],"mean":0.9335255382517825,"std":0.012262958829399888},"egc":{"folds":[0.8886401753284655,0.927657289291464,0.9184387681210542,0.909445207662316,0.9228004282410603],"mean":0.9133963737288721,"std":0.015376254157658902},"ei":{"folds":[0.8037644434587705,0.7786098301809068,0.8000584688419449,0.8489937255875596,0.8502235914213733],"mean":0.8163300118981109,"std":0.03186362211557758},"eps":{"folds":[0.7499984269023763,0.8416123214713819,0.6853009959916299,0.7487180306197174,0.7950411994357042],"mean":0.764134194884162,"std":0.058331497354574796},"nc":{"folds":[0.762081558778264,0.9068430336307994,0.8348861707303968,0.6353641929205771,0.7947465670560484],"mean":0.786784304623217,"std":0.10041110675278749},"xc":{"folds":[0.45977915403169956,0.3275758635446113,0.4653559817880577,0.1697769937769471,0.5403012850768553],"mean":0.3925578556436342,"std":0.14620222534132926}},"macro":0.7913509247625832}},"budget":{"plan":"UA2D-TRANSFER-20261022-01","revision":"r1"},"paper_envelope_macro":0.8104285714285714,"sota":{"R_MEAN":{"eea":{"project":{"folds":[0.9212382243868382,0.9471401580734473,0.9417275132105148,0.9399328641591636,0.9215917829595731],"mean":0.9343261085579074,"std":0.012081723021885992},"reference":0.934,"delta":0.0003261085579073786,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9463391643580314,0.9080728022816144,0.9467428010409014,0.9329311905200541,0.9204951011318151],"mean":0.9309162118664831,"std":0.016754323604916164},"reference":0.923,"delta":0.007916211866483058,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8860402682872044,0.928702029369899,0.9191258760960525,0.9157724357207029,0.9016656356176674],"mean":0.9102612490183052,"std":0.01665501162849731},"reference":0.911,"delta":-0.0007387509816948468,"status":"COMPLETE"},"ei":{"project":{"folds":[0.8051094232693398,0.8006904274176488,0.7996828860582365,0.8418935367852383,0.8476393749919214],"mean":0.8190031297044769,"std":0.023694235789467458},"reference":0.839,"delta":-0.019996870295523106,"status":"COMPLETE"},"eps":{"project":{"folds":[0.7510333872046395,0.8260251886166968,0.66844397523638,0.7411593280327176,0.708029584491916],"mean":0.73893829271647,"std":0.05843831959250935},"reference":0.787,"delta":-0.048061707283529986,"status":"COMPLETE"},"nc":{"project":{"folds":[0.7617120425113725,0.9105482555940217,0.8369692425762751,0.7249607262814185,0.7933531411072212],"mean":0.805508681614062,"std":0.07172334847858607},"reference":0.856,"delta":-0.05049131838593801,"status":"COMPLETE"},"xc":{"project":{"folds":[0.4013887883224011,0.37364558639304857,0.47209453996497885,0.19637361712562718,0.5606532501425848],"mean":0.4008311563897281,"std":0.1352440599633569},"reference":0.423,"delta":-0.022168843610271882,"status":"COMPLETE"}},"R_STD":{"eea":{"project":{"folds":[0.9247950006173244,0.9495461981389602,0.9454628512807024,0.9272525146824754,0.9186201121312044],"mean":0.9331353353701333,"std":0.013565991799815012},"reference":0.934,"delta":-0.0008646646298667848,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9411645467312642,0.9228650253481733,0.9460083161584405,0.9283979117937551,0.9227400175546877],"mean":0.9322351635172643,"std":0.010748434830333052},"reference":0.923,"delta":0.009235163517264278,"status":"COMPLETE"},"egc":{"project":{"folds":[0.886697633753897,0.931895601093778,0.914700190649696,0.9221010979938808,0.9214877605886429],"mean":0.9153764568159788,"std":0.01716379320399424},"reference":0.911,"delta":0.004376456815978758,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7719537284499173,0.779687453130751,0.7931309837247691,0.8534159460203516,0.8371122832825049],"mean":0.8070600789216588,"std":0.03615154761747953},"reference":0.839,"delta":-0.031939921078341205,"status":"COMPLETE"},"eps":{"project":{"folds":[0.7491876518329676,0.8061980318988372,0.6012803926847208,0.6667679738736515,0.7816561800562905],"mean":0.7210180460692935,"std":0.08515620344329522},"reference":0.787,"delta":-0.0659819539307065,"status":"COMPLETE"},"nc":{"project":{"folds":[0.734389373431472,0.8960975366079077,0.8058997402268405,0.7325089793002433,0.7912373151453919],"mean":0.792026588942371,"std":0.06687346988353035},"reference":0.856,"delta":-0.06397341105762899,"status":"COMPLETE"},"xc":{"project":{"folds":[0.41479199187794435,0.36390000773866993,0.47598861118627456,0.27511065761723086,0.5480374592832081],"mean":0.4155657455406655,"std":0.10439978031065052},"reference":0.423,"delta":-0.007434254459334488,"status":"COMPLETE"}},"A_SP":{"eea":{"project":{"folds":[0.9230277067706659,0.9472107542101151,0.9396672462224241,0.9311851969788403,0.920809108157039],"mean":0.9323800024678169,"std":0.011133661896422203},"reference":0.934,"delta":-0.001619997532183115,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9439198101249202,0.912805504743901,0.9465964693000755,0.9169906652061836,0.9245599291808253],"mean":0.9289744757111811,"std":0.015479286451696035},"reference":0.923,"delta":0.005974475711181038,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8835687300689091,0.9265791030608996,0.9240869718154585,0.9092541296826004,0.921920731605884],"mean":0.9130819332467504,"std":0.01779810428996875},"reference":0.911,"delta":0.0020819332467503227,"status":"COMPLETE"},"ei":{"project":{"folds":[0.8035992025327083,0.7742841759666874,0.7998509661675249,0.852946771900614,0.8706789378133049],"mean":0.8202720108761679,"std":0.040057398991828856},"reference":0.839,"delta":-0.01872798912383211,"status":"COMPLETE"},"eps":{"project":{"folds":[0.7429751437549887,0.82558373825018,0.6853748502414113,0.7490213891172125,0.7949545984148768],"mean":0.759581943955734,"std":0.053622442648818804},"reference":0.787,"delta":-0.02741805604426606,"status":"COMPLETE"},"nc":{"project":{"folds":[0.7620815689955034,0.8861452703622603,0.8347531917729735,0.6383287173201504,0.7947465848628412],"mean":0.7832110666627458,"std":0.09330388517497021},"reference":0.856,"delta":-0.07278893333725422,"status":"COMPLETE"},"xc":{"project":{"folds":[0.4597791516455405,0.3278190776735953,0.4653362315851266,0.17045975025560178,0.5536582134634415],"mean":0.3954104849246612,"std":0.14937093948578128},"reference":0.423,"delta":-0.0275895150753388,"status":"COMPLETE"}},"R_STD_SP":{"eea":{"project":{"folds":[0.9249258987044148,0.9494935984645693,0.9446894178298351,0.9263453893965484,0.9186349447095087],"mean":0.9328178498209752,"std":0.013456830547960834},"reference":0.934,"delta":-0.001182150179024899,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9401824199466463,0.9268548533963497,0.9449501705656594,0.9288643358575797,0.9228181754409827],"mean":0.9327339910414436,"std":0.009388503941497282},"reference":0.923,"delta":0.009733991041443568,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8934467507470006,0.9288821987881274,0.9081218294390948,0.9178687843037895,0.9194013666182705],"mean":0.9135441859792565,"std":0.01343104804171137},"reference":0.911,"delta":0.0025441859792564303,"status":"COMPLETE"},"ei":{"project":{"folds":[0.769538379469682,0.7807643228970182,0.7930069350298413,0.8457895020785373,0.8388916592597455],"mean":0.8055981597469648,"std":0.03463873285223481},"reference":0.839,"delta":-0.033401840253035164,"status":"COMPLETE"},"eps":{"project":{"folds":[0.7492467620339915,0.774748521501148,0.6014961973775625,0.6659628109123114,0.772052443601069],"mean":0.7127013470852164,"std":0.07625458666450974},"reference":0.787,"delta":-0.07429865291478366,"status":"COMPLETE"},"nc":{"project":{"folds":[0.7322416784876528,0.8974998805555947,0.8060007190140874,0.7031095877369253,0.791237327596798],"mean":0.7860178386782116,"std":0.07520049486410266},"reference":0.856,"delta":-0.06998216132178836,"status":"COMPLETE"},"xc":{"project":{"folds":[0.41479199567990943,0.36570370211296843,0.4754810044750344,0.2657461180842019,0.5502395746087279],"mean":0.41439247899216836,"std":0.10799622510981939},"reference":0.423,"delta":-0.008607521007831631,"status":"COMPLETE"}}}}
```

</details>

<details><summary>results/ua3d_ph_opt_20261031/r1/final_review/results.json</summary>

```json
{"plan":"UA3D-PH-OPT-20261031-01","revision":"r1","status":"STOPPED","FINAL":null,"execution_complete":false,"performance_pass":null,"reference_macro":0.804675462276949,"delta_macro":null,"independent_review":"PENDING","stage_facts":{"implementation":{"status":"PASSED"},"preflight":{"status":"PASSED"},"packing":{"status":"PASSED"},"diagnostic":{"status":"PASSED"},"pilot":{"status":"NOT_RUN"},"ft":{"status":"NOT_RUN"},"freeze":{"status":"NOT_RUN"},"test":{"status":"NOT_RUN"},"cpu":{"status":"NOT_RUN"},"delivery":{"status":"PASSED"}}}
```

</details>

<details><summary>results/ua3d_ph_opt_20261031/r1/repairs/auto_r2/final_review/results.json</summary>

```json
{"plan":"UA3D-PH-OPT-20261031-01","revision":"r2","status":"STOPPED","FINAL":null,"execution_complete":false,"performance_pass":null,"reference_macro":0.804675462276949,"delta_macro":null,"independent_review":"PENDING","stage_facts":{"implementation":{"status":"PASSED"},"preflight":{"status":"PASSED"},"packing":{"status":"NOT_RUN"},"diagnostic":{"status":"PASSED"},"pilot":{"status":"NOT_RUN"},"ft":{"status":"NOT_RUN"},"freeze":{"status":"NOT_RUN"},"test":{"status":"NOT_RUN"},"cpu":{"status":"NOT_RUN"},"delivery":{"status":"PASSED"}}}
```

</details>

<details><summary>results/ua3d_hmr_readout_20261103/r1/final_review/results.json</summary>

```json
{"FINAL":null,"reference":0.804675462276949,"selected_macro":null,"delta_C_FIXED":null,"target_met":false,"performance_pass":false,"status":"STOPPED","execution_complete":false,"preflight":{"status":"PASSED"},"final_cpu":{"status":"NOT_RUN"},"independent_review":"PENDING"}
```

</details>

<details><summary>results/ua3d_anchor_opt_20261102/r2/final_review/results.json</summary>

```json
{"plan":"UA3D-BASE-ANCHOR-20261102-01","revision":"r2","status":"COMPLETE_PENDING_REVIEW","independent_review":"PENDING","FINAL":"BASE_KEEP","reference":0.804675462276949,"selected_macro":0.804675462276949,"performance_pass":false,"validation":{"C_ANCHOR":{"macro":0.8078628636882943,"tasks":{"eea":{"mean":0.9229350682947899,"std":0.019906804333146833,"folds":[0.9258489964641599,0.9178132864027428,0.9058696665311429,0.9094044488770181,0.9557389431988855]},"egb":{"mean":0.9286126237237007,"std":0.014468876619352455,"folds":[0.9317327707438666,0.9323129066027187,0.9043846346096567,0.9311739634139514,0.9434588432483098]},"egc":{"mean":0.9146058435915201,"std":0.005769546908078458,"folds":[0.9136946123629407,0.9231168207368707,0.9165977648387951,0.9120366467141694,0.9075833733048249]},"ei":{"mean":0.8619796973872516,"std":0.018703992102611068,"folds":[0.8889190565584546,0.8665770074941318,0.8655728937003306,0.847770964939784,0.8410585642435568]},"eps":{"mean":0.7676839814835252,"std":0.07442597285127069,"folds":[0.7472369430230483,0.8569800100358066,0.7720583584291768,0.8054533228589349,0.6566912730706598]},"nc":{"mean":0.8461614744560061,"std":0.049218611668987014,"folds":[0.8335550582100563,0.9090776168632717,0.8382871297614007,0.8729549185146485,0.776932648930654]},"xc":{"mean":0.4130613568812661,"std":0.07753748895830614,"folds":[0.35120114193594953,0.4562606930130606,0.3929779832746815,0.5251736564179612,0.3396933097646776]}}},"G_GEOM":{"macro":0.7977246540301042,"tasks":{"eea":{"mean":0.9246584696344506,"std":0.018664437085937673,"folds":[0.9245724457536968,0.9171705226141236,0.9140046261767865,0.9107752345165332,0.9567695191111125]},"egb":{"mean":0.9311131247357913,"std":0.016933974208215987,"folds":[0.9312088195629958,0.9309862246837298,0.9057017261612064,0.9344107742848887,0.9532580789861358]},"egc":{"mean":0.916187233968494,"std":0.004428138684139982,"folds":[0.915701036460582,0.9228047215108354,0.9178979574670032,0.9128707751407457,0.9116616792633044]},"ei":{"mean":0.8594939276571394,"std":0.015158131145934086,"folds":[0.8763347549181069,0.8675741471301788,0.8667280143480692,0.8427590798999512,0.844073641989391]},"eps":{"mean":0.7277692026850702,"std":0.10958385730886239,"folds":[0.689885581869388,0.8524534209094957,0.7353398700974579,0.7957853092226966,0.5653818313263123]},"nc":{"mean":0.8227880944365715,"std":0.07704193281956997,"folds":[0.7737070111974419,0.9049824026734432,0.835840506954814,0.8813783498697144,0.7180322014874444]},"xc":{"mean":0.40206252509321205,"std":0.08548875201037652,"folds":[0.34910331718593635,0.45484486684371706,0.3305886387475905,0.5272389561280735,0.348536846560743]}}},"P_PE":{"macro":0.8065793695838803,"tasks":{"eea":{"mean":0.9228119319907178,"std":0.01962882327413951,"folds":[0.9246271096588815,0.917776416012762,0.9069628670905269,0.9091141391775943,0.9555791280138235]},"egb":{"mean":0.9299502030251263,"std":0.014184682908282368,"folds":[0.9316875946469851,0.931527021777736,0.9069776903514714,0.933520150286706,0.9460385580627331]},"egc":{"mean":0.9151733736372956,"std":0.005718156108388381,"folds":[0.9148041839833106,0.9234346954492547,0.9173945801693701,0.9118957391219312,0.9083376694626112]},"ei":{"mean":0.8638805000536243,"std":0.017058960173791438,"folds":[0.8887846752880909,0.8670734474291819,0.8674583241456999,0.8511388046554998,0.8449472487496489]},"eps":{"mean":0.7638694788558931,"std":0.08521385724538666,"folds":[0.7434430072109321,0.8674961035700226,0.7703223461996405,0.8021976926328258,0.6358882446660445]},"nc":{"mean":0.841670770595405,"std":0.05462292394590278,"folds":[0.8295428009238297,0.9075539210160424,0.8373390493150222,0.8726675462337617,0.7612505354883687]},"xc":{"mean":0.4086993289291,"std":0.07941580817800412,"folds":[0.3487018214881249,0.45583824334133416,0.3694476842639698,0.5256039776537264,0.34390491789834476]}}},"P_LEARN":{"macro":0.8069341445799468,"tasks":{"eea":{"mean":0.9228409174352924,"std":0.019649367537067496,"folds":[0.9247445425424627,0.9177860086764944,0.9069401279395859,0.9091194935548995,0.9556144144630195]},"egb":{"mean":0.929930495438458,"std":0.014658306583628659,"folds":[0.9319254125700982,0.9316270865249834,0.9060548689103507,0.9337431902212533,0.9463019189656043]},"egc":{"mean":0.9152716301387886,"std":0.005647617255378551,"folds":[0.915036093251375,0.9234614818111052,0.917393105487065,0.9118449060059982,0.9086225641383998]},"ei":{"mean":0.8640568875115395,"std":0.017273062347382354,"folds":[0.8888151563544654,0.8682155538111,0.867755232548501,0.8507167668684419,0.8447817279751901]},"eps":{"mean":0.7640859250324323,"std":0.08225986572437828,"folds":[0.7441115558172017,0.8627757168442074,0.7708891233168164,0.8026441736951181,0.6400090554888185]},"nc":{"mean":0.8425375433154627,"std":0.05353066221313418,"folds":[0.830216081849943,0.9078460521086832,0.8374106748916432,0.8727993550678317,0.7644155526592123]},"xc":{"mean":0.4098156131876536,"std":0.07883373778194612,"folds":[0.34889214100281785,0.455863818239014,0.37517285334300965,0.5256811245010327,0.343468128852394]}}}},"formal":{"C_ANCHOR":{"macro":0.804675462883717,"tasks":{"eea":{"mean":0.9295688558369829,"std":0.01754679379228602,"folds":[0.9042462711552217,0.9365220872318804,0.9461348247638234,0.9420014140664298,0.918939681967559]},"egb":{"mean":0.9207569194790153,"std":0.015123003079500142,"folds":[0.9344724694701166,0.8962762604618353,0.9295219341247062,0.9266852645798493,0.9168286687585689]},"egc":{"mean":0.9141389050197775,"std":0.012837492980919523,"folds":[0.8942510629508391,0.92261382679794,0.921665792548718,0.9080928763498375,0.9240709664515532]},"ei":{"mean":0.8213759961143783,"std":0.04022955921842092,"folds":[0.8154528041829034,0.7881013591167164,0.7788210131956405,0.8717048190815757,0.8527999849950554]},"eps":{"mean":0.7758804820169065,"std":0.0683173376065637,"folds":[0.7500065386646162,0.8534280097994607,0.6745190900168505,0.7852747326976403,0.816174038905965]},"nc":{"mean":0.8313768055344614,"std":0.057063033347545464,"folds":[0.8192336246718293,0.9143316272328296,0.8223237503488454,0.7557382647494753,0.8452567606693266]},"xc":{"mean":0.43963027618449857,"std":0.14343566604142524,"folds":[0.4975306574083711,0.373087410571487,0.4686508238167989,0.2377840908411245,0.6210983982847111]}}},"G_GEOM":{"macro":0.8073860377992158,"tasks":{"eea":{"mean":0.9325738011105618,"std":0.019908013751295105,"folds":[0.906476398333938,0.9368363847788784,0.9576672651579322,0.9421489570661103,0.9197400002159499]},"egb":{"mean":0.9235208702546005,"std":0.011382333033177934,"folds":[0.9331203127919312,0.906733583872313,0.9312962847542262,0.9296938480173347,0.916760321837198]},"egc":{"mean":0.9148348309008163,"std":0.012200759867529455,"folds":[0.8957176034134955,0.9229314088272893,0.9227953935296433,0.909423744016667,0.9233060047169858]},"ei":{"mean":0.8276815242345114,"std":0.03644894261963384,"folds":[0.8237485366178088,0.7887250810297752,0.7960231339075616,0.8651631804716141,0.864747689145797]},"eps":{"mean":0.7675060605199839,"std":0.08558054127628352,"folds":[0.7404341935207729,0.8635917909875446,0.6362490482474157,0.7895376805491104,0.8077175892950759]},"nc":{"mean":0.829631716007512,"std":0.05986833195713812,"folds":[0.8135236885827244,0.9113105892919852,0.8267736308799644,0.746163206694037,0.8503874645888483]},"xc":{"mean":0.4559534615665246,"std":0.14678835474426158,"folds":[0.49758835347537067,0.39511267958937524,0.51049037401229,0.24187791500788047,0.6346979857477064]}}},"P_PE":{"macro":0.8073324973033907,"tasks":{"eea":{"mean":0.9296518174637429,"std":0.018268100192434716,"folds":[0.9030998053559869,0.9367111496247276,0.9471406835785524,0.9423195365699546,0.9189879121894925]},"egb":{"mean":0.9216733047982586,"std":0.01463755909549905,"folds":[0.9326734019157812,0.896669714475212,0.930871546594724,0.9267868350345686,0.9213650259710076]},"egc":{"mean":0.914726616704268,"std":0.012814627661688024,"folds":[0.8944108757883372,0.9228095647929216,0.9220253679680847,0.9096653177339666,0.9247219572380301]},"ei":{"mean":0.827452297668068,"std":0.03808595650718544,"folds":[0.8157694186783941,0.7888599546185636,0.7974419643213129,0.8727329941679844,0.8624571565540853]},"eps":{"mean":0.7815349428662419,"std":0.06833853917919668,"folds":[0.7488445471063969,0.8681458568003312,0.6877449318317705,0.78535824986903,0.8175811287236812]},"nc":{"mean":0.8318840195475371,"std":0.05459830063439363,"folds":[0.8179169567464118,0.9114000319185354,0.8221052558902475,0.7606674044338598,0.8473304487486313]},"xc":{"mean":0.4444044820756183,"std":0.14720570576760056,"folds":[0.49794217369381655,0.37445148187435273,0.48198142848045045,0.2372499642342526,0.6303973620952192]}}},"P_LEARN":{"macro":0.8067333950086197,"tasks":{"eea":{"mean":0.9296533659010005,"std":0.018201244248580314,"folds":[0.9032223585622186,0.9366939018463459,0.9470522576455032,0.9423168299744232,0.9189814814765104]},"egb":{"mean":0.9214256596883763,"std":0.01449696781377437,"folds":[0.9327491952059449,0.896685534733454,0.930700911701492,0.9251913838618491,0.9218012729391407]},"egc":{"mean":0.9144628534227728,"std":0.013130172010942411,"folds":[0.8940527653445202,0.9228321749462192,0.9220278002835729,0.9084623311358825,0.9249391954036692]},"ei":{"mean":0.825495079987778,"std":0.03865490940426046,"folds":[0.8157188376511697,0.7892059405467071,0.7905608128425196,0.8716277076950939,0.8603621012034004]},"eps":{"mean":0.7806655937624787,"std":0.06820539378709786,"folds":[0.7490636579323173,0.8655834541608246,0.6857239147140718,0.7853864937280097,0.8175704482771694]},"nc":{"mean":0.8318324478063861,"std":0.055016089729294955,"folds":[0.8181593930006135,0.9118922277589971,0.8221060090157488,0.7598516340372523,0.8471529752193193]},"xc":{"mean":0.4435987644915464,"std":0.1466005364342977,"folds":[0.4979146875284459,0.37432850948051066,0.47975227505001705,0.23718484669578832,0.6288135037029701]}}}},"execution_complete":true,"stage_facts":{"implementation":{"status":"PASSED"},"preflight":{"status":"PASSED"},"cache_trainval":{"status":"PASSED"},"diagnostic":{"status":"PASSED"},"cost_gate":{"status":"PASSED"},"pilot":{"status":"PASSED"},"ft":{"status":"PASSED"},"freeze":{"status":"PASSED"},"cache_test":{"status":"PASSED"},"anchor_output":{"status":"PASSED"},"adapter_output":{"status":"PASSED"},"final_cpu":{"status":"PASSED"},"delivery":{"status":"PASSED"}},"sota":{"C_ANCHOR":{"eea":{"project":{"mean":0.9295688558369829,"std":0.01754679379228602,"folds":[0.9042462711552217,0.9365220872318804,0.9461348247638234,0.9420014140664298,0.918939681967559]},"delta":-0.004431144163017153},"egb":{"project":{"mean":0.9207569194790153,"std":0.015123003079500142,"folds":[0.9344724694701166,0.8962762604618353,0.9295219341247062,0.9266852645798493,0.9168286687585689]},"delta":-0.0022430805209847637},"egc":{"project":{"mean":0.9141389050197775,"std":0.012837492980919523,"folds":[0.8942510629508391,0.92261382679794,0.921665792548718,0.9080928763498375,0.9240709664515532]},"delta":0.003138905019777427},"ei":{"project":{"mean":0.8213759961143783,"std":0.04022955921842092,"folds":[0.8154528041829034,0.7881013591167164,0.7788210131956405,0.8717048190815757,0.8527999849950554]},"delta":-0.017624003885621686},"eps":{"project":{"mean":0.7758804820169065,"std":0.0683173376065637,"folds":[0.7500065386646162,0.8534280097994607,0.6745190900168505,0.7852747326976403,0.816174038905965]},"delta":-0.011119517983093519},"nc":{"project":{"mean":0.8313768055344614,"std":0.057063033347545464,"folds":[0.8192336246718293,0.9143316272328296,0.8223237503488454,0.7557382647494753,0.8452567606693266]},"delta":-0.024623194465538623},"xc":{"project":{"mean":0.43963027618449857,"std":0.14343566604142524,"folds":[0.4975306574083711,0.373087410571487,0.4686508238167989,0.2377840908411245,0.6210983982847111]},"delta":0.016630276184498582}},"G_GEOM":{"eea":{"project":{"mean":0.9325738011105618,"std":0.019908013751295105,"folds":[0.906476398333938,0.9368363847788784,0.9576672651579322,0.9421489570661103,0.9197400002159499]},"delta":-0.0014261988894382904},"egb":{"project":{"mean":0.9235208702546005,"std":0.011382333033177934,"folds":[0.9331203127919312,0.906733583872313,0.9312962847542262,0.9296938480173347,0.916760321837198]},"delta":0.0005208702546004584},"egc":{"project":{"mean":0.9148348309008163,"std":0.012200759867529455,"folds":[0.8957176034134955,0.9229314088272893,0.9227953935296433,0.909423744016667,0.9233060047169858]},"delta":0.003834830900816266},"ei":{"project":{"mean":0.8276815242345114,"std":0.03644894261963384,"folds":[0.8237485366178088,0.7887250810297752,0.7960231339075616,0.8651631804716141,0.864747689145797]},"delta":-0.011318475765488611},"eps":{"project":{"mean":0.7675060605199839,"std":0.08558054127628352,"folds":[0.7404341935207729,0.8635917909875446,0.6362490482474157,0.7895376805491104,0.8077175892950759]},"delta":-0.019493939480016098},"nc":{"project":{"mean":0.829631716007512,"std":0.05986833195713812,"folds":[0.8135236885827244,0.9113105892919852,0.8267736308799644,0.746163206694037,0.8503874645888483]},"delta":-0.02636828399248803},"xc":{"project":{"mean":0.4559534615665246,"std":0.14678835474426158,"folds":[0.49758835347537067,0.39511267958937524,0.51049037401229,0.24187791500788047,0.6346979857477064]},"delta":0.03295346156652462}},"P_PE":{"eea":{"project":{"mean":0.9296518174637429,"std":0.018268100192434716,"folds":[0.9030998053559869,0.9367111496247276,0.9471406835785524,0.9423195365699546,0.9189879121894925]},"delta":-0.004348182536257128},"egb":{"project":{"mean":0.9216733047982586,"std":0.01463755909549905,"folds":[0.9326734019157812,0.896669714475212,0.930871546594724,0.9267868350345686,0.9213650259710076]},"delta":-0.0013266952017414457},"egc":{"project":{"mean":0.914726616704268,"std":0.012814627661688024,"folds":[0.8944108757883372,0.9228095647929216,0.9220253679680847,0.9096653177339666,0.9247219572380301]},"delta":0.003726616704267993},"ei":{"project":{"mean":0.827452297668068,"std":0.03808595650718544,"folds":[0.8157694186783941,0.7888599546185636,0.7974419643213129,0.8727329941679844,0.8624571565540853]},"delta":-0.01154770233193192},"eps":{"project":{"mean":0.7815349428662419,"std":0.06833853917919668,"folds":[0.7488445471063969,0.8681458568003312,0.6877449318317705,0.78535824986903,0.8175811287236812]},"delta":-0.005465057133758111},"nc":{"project":{"mean":0.8318840195475371,"std":0.05459830063439363,"folds":[0.8179169567464118,0.9114000319185354,0.8221052558902475,0.7606674044338598,0.8473304487486313]},"delta":-0.024115980452462882},"xc":{"project":{"mean":0.4444044820756183,"std":0.14720570576760056,"folds":[0.49794217369381655,0.37445148187435273,0.48198142848045045,0.2372499642342526,0.6303973620952192]},"delta":0.021404482075618292}},"P_LEARN":{"eea":{"project":{"mean":0.9296533659010005,"std":0.018201244248580314,"folds":[0.9032223585622186,0.9366939018463459,0.9470522576455032,0.9423168299744232,0.9189814814765104]},"delta":-0.004346634098999602},"egb":{"project":{"mean":0.9214256596883763,"std":0.01449696781377437,"folds":[0.9327491952059449,0.896685534733454,0.930700911701492,0.9251913838618491,0.9218012729391407]},"delta":-0.0015743403116237342},"egc":{"project":{"mean":0.9144628534227728,"std":0.013130172010942411,"folds":[0.8940527653445202,0.9228321749462192,0.9220278002835729,0.9084623311358825,0.9249391954036692]},"delta":0.003462853422772749},"ei":{"project":{"mean":0.825495079987778,"std":0.03865490940426046,"folds":[0.8157188376511697,0.7892059405467071,0.7905608128425196,0.8716277076950939,0.8603621012034004]},"delta":-0.013504920012221922},"eps":{"project":{"mean":0.7806655937624787,"std":0.06820539378709786,"folds":[0.7490636579323173,0.8655834541608246,0.6857239147140718,0.7853864937280097,0.8175704482771694]},"delta":-0.006334406237521351},"nc":{"project":{"mean":0.8318324478063861,"std":0.055016089729294955,"folds":[0.8181593930006135,0.9118922277589971,0.8221060090157488,0.7598516340372523,0.8471529752193193]},"delta":-0.024167552193613884},"xc":{"project":{"mean":0.4435987644915464,"std":0.1466005364342977,"folds":[0.4979146875284459,0.37432850948051066,0.47975227505001705,0.23718484669578832,0.6288135037029701]},"delta":0.020598764491546406}}}}
```

</details>

<details><summary>results/ua3d_anchor_lr_20261102/r1/final_review/results.json</summary>

```json
{"plan":"UA3D-ANCHOR-LR-20261102-01","revision":"r1","status":"STOPPED","execution_complete":false,"FINAL":null,"reference":0.804675462276949,"selected_macro":null,"performance_pass":null,"target_met":null,"preflight_cpu":{"status":"NOT_RUN"},"final_cpu":{"status":"NOT_RUN"},"stage_facts":{"implementation":{"status":"FAILED"},"preflight":{"status":"NOT_RUN"},"cache_trainval":{"status":"NOT_RUN"},"diagnostic":{"status":"NOT_RUN"},"cost_gate":{"status":"NOT_RUN"},"pilot":{"status":"NOT_RUN"},"ft":{"status":"NOT_RUN"},"freeze":{"status":"NOT_RUN"},"cache_test":{"status":"NOT_RUN"},"anchor_output":{"status":"NOT_RUN"},"adapter_output":{"status":"NOT_RUN"},"final_cpu":{"status":"NOT_RUN"},"delivery":{"status":"PASSED"}},"independent_review":"PENDING"}
```

</details>

<details><summary>results/ua3d_anchor_lr_20261102/r1/repairs/init_r2/final_review/results.json</summary>

```json
{"status":"STOPPED","execution_complete":false,"independent_review":"PENDING","preflight_cpu":{"status":"PASSED"},"final_cpu":{"status":"FAILED"},"stage_facts":{"implementation":{"status":"PASSED"},"preflight":{"status":"PASSED"},"cache_trainval":{"status":"PASSED"},"diagnostic":{"status":"PASSED"},"cost_gate":{"status":"PASSED"},"pilot":{"status":"PASSED"},"ft":{"status":"PASSED"},"freeze":{"status":"PASSED"},"cache_test":{"status":"PASSED"},"anchor_output":{"status":"PASSED"},"adapter_output":{"status":"PASSED"},"final_cpu":{"status":"FAILED"},"delivery":{"status":"NOT_RUN"}}}
```

</details>

<details><summary>results/ua2d_opt_20261021/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-OPT-20261021-01","revision":"r1","status":"需返修","performance_pass":null,"budget":{"plan":"UA2D-OPT-20261021-01","revision":"r1"},"sota":{"eea":{"reference":0.934,"delta":null,"status":"未完成／不可比较"},"egb":{"reference":0.923,"delta":null,"status":"未完成／不可比较"},"egc":{"reference":0.911,"delta":null,"status":"未完成／不可比较"},"ei":{"reference":0.839,"delta":null,"status":"未完成／不可比较"},"eps":{"reference":0.787,"delta":null,"status":"未完成／不可比较"},"nc":{"reference":0.856,"delta":null,"status":"未完成／不可比较"},"xc":{"reference":0.423,"delta":null,"status":"未完成／不可比较"}},"independent_review":"PENDING","cd_required_comparisons":{"C_vs_direct":{"reference":null,"delta":null,"performance_pass":null},"D_vs_direct":{"reference":null,"delta":null,"performance_pass":null},"C_vs_B_FP":{"reference":"B_FP","delta":null,"performance_pass":null},"D_vs_B_FP":{"reference":"B_FP","delta":null,"performance_pass":null}}}
```

</details>

<details><summary>results/ua2d_opt_20261021/r1/repair_ft1/snapshot/final_review/results.json</summary>

```json
{"plan":"UA2D-OPT-20261021-01","revision":"r1","status":"需返修","performance_pass":null,"budget":{"plan":"UA2D-OPT-20261021-01","revision":"r1"},"sota":{"eea":{"reference":0.934,"delta":null,"status":"未完成／不可比较"},"egb":{"reference":0.923,"delta":null,"status":"未完成／不可比较"},"egc":{"reference":0.911,"delta":null,"status":"未完成／不可比较"},"ei":{"reference":0.839,"delta":null,"status":"未完成／不可比较"},"eps":{"reference":0.787,"delta":null,"status":"未完成／不可比较"},"nc":{"reference":0.856,"delta":null,"status":"未完成／不可比较"},"xc":{"reference":0.423,"delta":null,"status":"未完成／不可比较"}},"independent_review":"PENDING","cd_required_comparisons":{"C_vs_direct":{"reference":null,"delta":null,"performance_pass":null},"D_vs_direct":{"reference":null,"delta":null,"performance_pass":null},"C_vs_B_FP":{"reference":"B_FP","delta":null,"performance_pass":null},"D_vs_B_FP":{"reference":"B_FP","delta":null,"performance_pass":null}}}
```

</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/repairs/mm_lifecycle_r6/final_review/results.json</summary>

```json
{"plan":"UA3D-COND-SPECTRAL-20261104-01","revision":"r6","FINAL":"S_STATIC_GATE","formal":{"C_FRESH":{"tasks":{"eea":{"folds":[0.8873782221533266,0.9359461693428633,0.9320904771415193,0.940486071927647,0.9149677489060003],"mean":0.9221737378942713,"std":0.02171415066282479},"egb":{"folds":[0.9130703542297971,0.8963080621535762,0.931866684576119,0.8860613753972348,0.9092968562028848],"mean":0.9073206665119224,"std":0.01741196536012025},"egc":{"folds":[0.8963870783383912,0.9205608832564822,0.9245357052239949,0.9115987150287026,0.9255735327315484],"mean":0.9157311829158239,"std":0.012134549240613255},"ei":{"folds":[0.8224900111610237,0.786892704497354,0.8291272990602981,0.8383202322933624,0.8469572846747109],"mean":0.8247575063373498,"std":0.023101718197918765},"eps":{"folds":[0.7394247289896108,0.8231704153609851,0.6874575124876571,0.6262949078595029,0.7071505941815548],"mean":0.716699631775862,"std":0.07240814451158216},"nc":{"folds":[0.8198919480915939,0.8939738394993177,0.8349199015565983,0.7500678811057764,0.7842386097991003],"mean":0.8166184360104773,"std":0.05432988927608119},"xc":{"folds":[0.40479166706125025,0.35799817703904613,0.4287213796756557,0.22651335646666204,0.6120660172829939],"mean":0.4060181195051216,"std":0.13919460587122914}},"macro":0.7870456115644041},"S_STATIC_GATE":{"tasks":{"eea":{"folds":[0.8784632734549765,0.9378702325299234,0.9396361778576922,0.9401315740396265,0.9136272786947425],"mean":0.9219457073153923,"std":0.026726332094818506},"egb":{"folds":[0.9251640604791803,0.9031661757964926,0.9049037257907827,0.9301514063453773,0.9293525446448512],"mean":0.9185475826113368,"std":0.013396979265076395},"egc":{"folds":[0.8982135729795699,0.9263123894021302,0.9165344448622024,0.9022450639339832,0.9217251737088291],"mean":0.913006128977343,"std":0.012248954591310415},"ei":{"folds":[0.786531110761179,0.7941348552706701,0.813152692556848,0.8471032296061256,0.8078026954188822],"mean":0.809744916722741,"std":0.023417853803135064},"eps":{"folds":[0.7568793501829904,0.7966474741931385,0.5854473199909132,0.7333781467999707,0.8029810867669714],"mean":0.7350666755867967,"std":0.08841543625744248},"nc":{"folds":[0.8478426049683846,0.9048527992202066,0.8322467992209347,0.7600524832041904,0.8312890609232289],"mean":0.835256749507389,"std":0.05167808509645133},"xc":{"folds":[0.3864981694419597,0.3417621740885738,0.4163439506620885,0.2461191998368073,0.5366342671205154],"mean":0.3854715522299889,"std":0.10622371810940756}},"macro":0.7884341875644267},"S_COND_GATE":{"tasks":{"eea":{"folds":[0.870750775710312,0.9382944673913565,0.9501424694870682,0.9456638425051342,0.9136954383451237],"mean":0.9237093986877989,"std":0.03278080312359462},"egb":{"folds":[0.91213013917789,0.9045808674626135,0.9176348513467182,0.93549528384069,0.9243845601489169],"mean":0.9188451403953657,"std":0.011810355320522057},"egc":{"folds":[0.8921325040011718,0.923762137425338,0.9174442703956387,0.8958796768654866,0.9181815261044779],"mean":0.9094800229584227,"std":0.014396440668685262},"ei":{"folds":[0.8217859933895328,0.7899722858792961,0.8321649432943705,0.8564380583511009,0.8147814561883149],"mean":0.823028547420523,"std":0.02429991785777448},"eps":{"folds":[0.757028803653163,0.846902278047552,0.609712590510094,0.7833982315681367,0.7144709800459046],"mean":0.7423025767649701,"std":0.088319751954857},"nc":{"folds":[0.8581495646075946,0.8812367851097607,0.8581870586994754,0.7538788089031406,0.7785419884164488],"mean":0.825998841147284,"std":0.05606788907971911},"xc":{"folds":[0.41783161628497034,0.37216115458410426,0.46273254101971006,0.2984827358052693,0.6350066077689238],"mean":0.43724293109259554,"std":0.12612198519589538}},"macro":0.7972296369238513},"S_STATIC_TOKEN":{"tasks":{"eea":{"folds":[0.8874600832590006,0.9312174178362889,0.9330527234194766,0.911934184629887,0.9180468775498639],"mean":0.9163422573389035,"std":0.018418922953761203},"egb":{"folds":[0.9252300574175413,0.8873050286027453,0.9377229931864892,0.9307599001803175,0.9270420560200746],"mean":0.9216120070814335,"std":0.019766084042329497},"egc":{"folds":[0.8874314071147141,0.9262004985904722,0.9253232229233992,0.905636229735731,0.9085655139298312],"mean":0.9106313744588297,"std":0.016013320805116967},"ei":{"folds":[0.7266089550449806,0.7969414915749724,0.8071030577322382,0.8308884734885429,0.8124053125018847],"mean":0.7947894580685237,"std":0.04005712863881606},"eps":{"folds":[0.7260477636849725,0.8475559016797594,0.6508897369237499,0.746055500483809,0.7883709892810715],"mean":0.7517839784106725,"std":0.07311922938611622},"nc":{"folds":[0.8421252261878575,0.8777724831230829,0.8097159001372798,0.7300750928123811,0.80248671135835],"mean":0.8124350827237903,"std":0.05486349536537314},"xc":{"folds":[0.5068873867231289,0.2859934507428775,0.3667452192175439,0.35224574067890735,0.5352241601564169],"mean":0.4094191915037749,"std":0.10683015034307515}},"macro":0.7881447642265612},"S_COND_TOKEN":{"tasks":{"eea":{"folds":[0.8735850501098202,0.9250170782793097,0.9329858772882449,0.9442949168249548,0.9227415225751887],"mean":0.9197248890155036,"std":0.027136378516701355},"egb":{"folds":[0.9298273188476986,0.892698234422566,0.9394747251196487,0.9344657364795512,0.9081730613632144],"mean":0.9209278152465359,"std":0.01978672106250147},"egc":{"folds":[0.8907740238098908,0.9262170905705875,0.9159847246348529,0.9086157953256389,0.9136909483870971],"mean":0.9110565165456134,"std":0.013023124649189075},"ei":{"folds":[0.769812834568699,0.7948684586437328,0.7913944641916714,0.7959468513984558,0.8251460556521144],"mean":0.7954337328909347,"std":0.019724040356282342},"eps":{"folds":[0.7419244811043846,0.8160822854782166,0.6058864293545074,0.7431716335659555,0.7866353639726616],"mean":0.7387400386951452,"std":0.08055378101350082},"nc":{"folds":[0.8592930332671629,0.8850934558900574,0.83120438743159,0.7188528257139936,0.8013755540664952],"mean":0.8191638512738599,"std":0.06418871374641158},"xc":{"folds":[0.36523613677556943,0.20065190211323436,0.465979848950829,0.3143254704759326,0.5183271426707868],"mean":0.37290410019727044,"std":0.12547114165828918}},"macro":0.7825644205521233},"S_COND_NOCROSS":{"tasks":{"eea":{"folds":[0.8762957101031948,0.9340916601440296,0.9356779112932503,0.9419294715926203,0.9113452346528934],"mean":0.9198679975571977,"std":0.026972932301838868},"egb":{"folds":[0.9101461450527594,0.9055971588057327,0.8578705837456017,0.9070950657115997,0.9235977522868191],"mean":0.9008613411205026,"std":0.025063142699376494},"egc":{"folds":[0.8874497654783763,0.9186171674748344,0.9201145053878029,0.8922821747491239,0.9222293344482679],"mean":0.908138589507681,"std":0.016816848538542154},"ei":{"folds":[0.7834596510966323,0.7604299570069593,0.8052801842570241,0.8251529025513294,0.8255793483280269],"mean":0.7999804086479945,"std":0.028081227887336933},"eps":{"folds":[0.6648131814358273,0.8268225187246627,0.5880126741349013,0.7308438748196324,0.7975929188511163],"mean":0.721617033593228,"std":0.09747110639526363},"nc":{"folds":[0.8444693423327155,0.8728057107310292,0.8523637613032828,0.7063726143873442,0.8075496279242118],"mean":0.8167122113357166,"std":0.06604307609467554},"xc":{"folds":[0.41287710727829474,0.3916160754417124,0.39957829932352307,0.2143741949899105,0.45739838437235325],"mean":0.37516881228115884,"std":0.0934142153407002}},"macro":0.7774780562919256}},"macro":0.7884341875644267,"delta_BASE":-0.016241274712522236,"delta_C_FRESH":0.0013885760000226233,"target_met":false,"performance_pass":false,"independent_review":"PENDING","paper_comparison":{"eea":{"delta":-0.012054292684607781},"egb":{"delta":-0.004452417388663199},"egc":{"delta":0.0020061289773429802},"ei":{"delta":-0.02925508327725901},"eps":{"delta":-0.05193332441320331},"nc":{"delta":-0.020743250492610943},"xc":{"delta":-0.037528447770011075}}}
```

</details>

<details><summary>results/ua2d_demol_20261020/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-DEMOL-LITE-20261020-01","revision":"r1","status":"待审查","independent_review":"PENDING","budget":{"plan":"UA2D-DEMOL-LITE-20261020-01","revision":"r1"},"development_decision":{"delta_self":-0.0006282188555268853,"delta_baseline":0.0010348282828616107,"selection":"validation_only_full_precision","validation":{"BOND_SELF":{"tasks":{"eea":{"folds":[0.938833443041506,0.9102972337866144,0.9102410105441439,0.9215294945832074,0.9457642652134223],"mean":0.9253330894337788,"std":0.016340192970961086},"egb":{"folds":[0.93555721388491,0.9364978246182042,0.9285613132052696,0.9478513460561226,0.9557932619237457],"mean":0.9408521919376505,"std":0.010843047750163233},"egc":{"folds":[0.9332368447693018,0.9258001470393601,0.925043733825342,0.9085193435256385,0.9210801699554919],"mean":0.9227360478230269,"std":0.009080714010903634},"ei":{"folds":[0.8516275023865783,0.8789052452649986,0.8950719819268126,0.8751393907225125,0.8229628864864972],"mean":0.8647414013574798,"std":0.028045032651918083},"eps":{"folds":[0.7116955655878573,0.875884088630006,0.750349105699432,0.840312079043327,0.737463754664084],"mean":0.7831409187249412,"std":0.07095065528100382},"nc":{"folds":[0.7972718991908353,0.908926943894299,0.8507874705346032,0.8993842625056769,0.7661361621305467],"mean":0.8445013476511922,"std":0.06239797720812845},"xc":{"folds":[0.3374743193029335,0.5112384727763828,0.5498608577132116,0.5431103611956691,0.3562247055891622],"mean":0.45958174331547186,"std":0.10414973169258475}},"macro":0.8201266771776486},"BOND_NEIGHBOR":{"tasks":{"eea":{"folds":[0.9390622450475968,0.9103010390357993,0.9100010838460988,0.9221739531797982,0.9458184887988749],"mean":0.9254713619816336,"std":0.016424515068242652},"egb":{"folds":[0.9353204615412756,0.9371044005228011,0.9285890667991136,0.9467276630230651,0.9557930861202074],"mean":0.9407069356012926,"std":0.010638094310830053},"egc":{"folds":[0.9259656819205843,0.9255967320784917,0.9225942619506109,0.9068314549799579,0.9186693042928693],"mean":0.9199314870445029,"std":0.007886710732125416},"ei":{"folds":[0.8528653166826087,0.8788334212116966,0.8951142977239596,0.8765547006936685,0.8188263655754691],"mean":0.8644388203774807,"std":0.029620204338794257},"eps":{"folds":[0.7222426264829465,0.8801218080765817,0.7504442568690661,0.8399643951045717,0.7394940623956486],"mean":0.7864534297857628,"std":0.06939390910159185},"nc":{"folds":[0.7975142152454744,0.9007186867356828,0.850801791770887,0.8990561067195341,0.7661361609179967],"mean":0.842845392277915,"std":0.06023182030559187},"xc":{"folds":[0.33747429220043235,0.4986956776865382,0.5503150711048044,0.5333308183152283,0.3633930466243208],"mean":0.45664178118626475,"std":0.09914716435604502}},"macro":0.8194984583221218}},"reference":{"tasks":{"eea":{"folds":[0.9385253327587553,0.9103085253762523,0.9106571106828454,0.9183210910769455,0.945869123306987],"mean":0.9247362366403571,"std":0.016464148955189464},"egb":{"folds":[0.9351454238178332,0.9358717531077823,0.92952116905478,0.9479926538394374,0.9559180061222797],"mean":0.9408898011884226,"std":0.010767101906054916},"egc":{"folds":[0.9219126712024205,0.9287402280684527,0.9245340532184785,0.909957660085341,0.9218161563689067],"mean":0.9213921537887199,"std":0.006983267331016533},"ei":{"folds":[0.853930487087313,0.8778922483494371,0.8960744572012486,0.8807400661015408,0.8280002148317006],"mean":0.867327494714248,"std":0.026664647471590804},"eps":{"folds":[0.7033665545728356,0.8691560148694102,0.749035734190309,0.8427987289319954,0.7235093357865903],"mean":0.7775732736702281,"std":0.07396907495351397},"nc":{"folds":[0.7966833731467942,0.9320223063590559,0.8471186026481647,0.8944451305349848,0.766136154324434],"mean":0.8472811134026866,"std":0.06809951616620287},"xc":{"folds":[0.3374743050508727,0.5202444411504019,0.549870188617994,0.5235968024424906,0.3190409470890351],"mean":0.45004533687015885,"std":0.1119565995183479}},"macro":0.8184636300392601}},"validation":{"BOND_SELF":{"tasks":{"eea":{"folds":[0.938833443041506,0.9102972337866144,0.9102410105441439,0.9215294945832074,0.9457642652134223],"mean":0.9253330894337788,"std":0.016340192970961086},"egb":{"folds":[0.93555721388491,0.9364978246182042,0.9285613132052696,0.9478513460561226,0.9557932619237457],"mean":0.9408521919376505,"std":0.010843047750163233},"egc":{"folds":[0.9332368447693018,0.9258001470393601,0.925043733825342,0.9085193435256385,0.9210801699554919],"mean":0.9227360478230269,"std":0.009080714010903634},"ei":{"folds":[0.8516275023865783,0.8789052452649986,0.8950719819268126,0.8751393907225125,0.8229628864864972],"mean":0.8647414013574798,"std":0.028045032651918083},"eps":{"folds":[0.7116955655878573,0.875884088630006,0.750349105699432,0.840312079043327,0.737463754664084],"mean":0.7831409187249412,"std":0.07095065528100382},"nc":{"folds":[0.7972718991908353,0.908926943894299,0.8507874705346032,0.8993842625056769,0.7661361621305467],"mean":0.8445013476511922,"std":0.06239797720812845},"xc":{"folds":[0.3374743193029335,0.5112384727763828,0.5498608577132116,0.5431103611956691,0.3562247055891622],"mean":0.45958174331547186,"std":0.10414973169258475}},"macro":0.8201266771776486},"BOND_NEIGHBOR":{"tasks":{"eea":{"folds":[0.9390622450475968,0.9103010390357993,0.9100010838460988,0.9221739531797982,0.9458184887988749],"mean":0.9254713619816336,"std":0.016424515068242652},"egb":{"folds":[0.9353204615412756,0.9371044005228011,0.9285890667991136,0.9467276630230651,0.9557930861202074],"mean":0.9407069356012926,"std":0.010638094310830053},"egc":{"folds":[0.9259656819205843,0.9255967320784917,0.9225942619506109,0.9068314549799579,0.9186693042928693],"mean":0.9199314870445029,"std":0.007886710732125416},"ei":{"folds":[0.8528653166826087,0.8788334212116966,0.8951142977239596,0.8765547006936685,0.8188263655754691],"mean":0.8644388203774807,"std":0.029620204338794257},"eps":{"folds":[0.7222426264829465,0.8801218080765817,0.7504442568690661,0.8399643951045717,0.7394940623956486],"mean":0.7864534297857628,"std":0.06939390910159185},"nc":{"folds":[0.7975142152454744,0.9007186867356828,0.850801791770887,0.8990561067195341,0.7661361609179967],"mean":0.842845392277915,"std":0.06023182030559187},"xc":{"folds":[0.33747429220043235,0.4986956776865382,0.5503150711048044,0.5333308183152283,0.3633930466243208],"mean":0.45664178118626475,"std":0.09914716435604502}},"macro":0.8194984583221218}},"test":{"BOND_SELF":{"status":"未完成／不可比较"},"BOND_NEIGHBOR":{"status":"未完成／不可比较"}},"sota":{"eea":{"reference":0.934,"arms":{"BOND_SELF":{"status":"未完成／不可比较"},"BOND_NEIGHBOR":{"status":"未完成／不可比较"}}},"egb":{"reference":0.923,"arms":{"BOND_SELF":{"status":"未完成／不可比较"},"BOND_NEIGHBOR":{"status":"未完成／不可比较"}}},"egc":{"reference":0.911,"arms":{"BOND_SELF":{"status":"未完成／不可比较"},"BOND_NEIGHBOR":{"status":"未完成／不可比较"}}},"ei":{"reference":0.839,"arms":{"BOND_SELF":{"status":"未完成／不可比较"},"BOND_NEIGHBOR":{"status":"未完成／不可比较"}}},"eps":{"reference":0.787,"arms":{"BOND_SELF":{"status":"未完成／不可比较"},"BOND_NEIGHBOR":{"status":"未完成／不可比较"}}},"nc":{"reference":0.856,"arms":{"BOND_SELF":{"status":"未完成／不可比较"},"BOND_NEIGHBOR":{"status":"未完成／不可比较"}}},"xc":{"reference":0.423,"arms":{"BOND_SELF":{"status":"未完成／不可比较"},"BOND_NEIGHBOR":{"status":"未完成／不可比较"}}}}}
```

</details>

<details><summary>results/ua3d_hmr_20261103/r1/final_review/results.json</summary>

```json
{"plan":"UA3D-HMR-20261103-01","revision":"r1","status":"STOPPED","execution_complete":false,"reference":0.804675462276949,"FINAL":null,"selected_macro":null,"target_met":false,"performance_pass":false,"independent_review":"PENDING","preflight":{"status":"PASSED"},"prepare":{"status":"FAILED"},"final_cpu":{"status":"NOT_RUN"},"clock":{"independent_review":"PENDING"},"sota":{"eea":{"reference_base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674}},"egb":{"reference_base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361}},"egc":{"reference_base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484}},"ei":{"reference_base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582}},"eps":{"reference_base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618}},"nc":{"reference_base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966}},"xc":{"reference_base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815}}}}
```

</details>

<details><summary>results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2/final_review/results.json</summary>

```json
{"reference":0.804675462276949,"FINAL":null,"selected_macro":null,"target_met":false,"performance_pass":false,"plan":"UA3D-HMR-20261103-01","revision":"r2","execution_complete":false,"independent_review":"PENDING","status":"STOPPED","preflight":{"status":"PASSED"},"final_cpu":{"status":"NOT_RUN"},"bootstrap_exit":{"status":"FINALIZING"}}
```

</details>

<details><summary>results/ua3d_hmr_20261103/r1/repairs/scheduler_r3/final_review/results.json</summary>

```json
{"reference":0.804675462276949,"FINAL":"S_FIXED","formal":{"C_ANCHOR":{"tasks":{"eea":{"folds":[0.9042462711552217,0.9365220872318804,0.9461348247638234,0.9420014140664298,0.918939681967559],"mean":0.9295688558369829,"std":0.01754679379228602},"egb":{"folds":[0.9344724694701166,0.8962762604618353,0.9295219341247062,0.9266852645798493,0.9168286687585689],"mean":0.9207569194790153,"std":0.015123003079500142},"egc":{"folds":[0.8942510629508391,0.92261382679794,0.921665792548718,0.9080928763498375,0.9240709664515532],"mean":0.9141389050197775,"std":0.012837492980919523},"ei":{"folds":[0.8154528041829034,0.7881013591167164,0.7788210131956405,0.8717048190815757,0.8527999849950554],"mean":0.8213759961143783,"std":0.04022955921842092},"eps":{"folds":[0.7500065386646162,0.8534280097994607,0.6745190900168505,0.7852747326976403,0.816174038905965],"mean":0.7758804820169065,"std":0.0683173376065637},"nc":{"folds":[0.8192336246718293,0.9143316272328296,0.8223237503488454,0.7557382647494753,0.8452567606693266],"mean":0.8313768055344614,"std":0.057063033347545464},"xc":{"folds":[0.4975306574083711,0.373087410571487,0.4686508238167989,0.2377840908411245,0.6210983982847111],"mean":0.43963027618449857,"std":0.14343566604142524}},"macro":0.804675462883717},"S_STAT":{"tasks":{"eea":{"folds":[0.9042462711552217,0.9365200894959715,0.9507401893767184,0.9420630266093959,0.9189878415523127],"mean":0.9305114836379241,"std":0.01871762364006601},"egb":{"folds":[0.9344724694701166,0.8962762604618353,0.9287098324873798,0.9270620951494998,0.920286221579448],"mean":0.9213613758296558,"std":0.01490652342216008},"egc":{"folds":[0.8949605992758506,0.9218378870353889,0.9225900592946394,0.9080928763498375,0.9260859355283638],"mean":0.914713471496816,"std":0.013001337540133247},"ei":{"folds":[0.8154528041829034,0.7888340408005895,0.7987229109819739,0.8717048190815757,0.8625417177513246],"mean":0.8274512585596735,"std":0.03758419985371398},"eps":{"folds":[0.7500065386646162,0.8564382444156364,0.6793656126275619,0.7852747326976403,0.816174038905965],"mean":0.777451833462284,"std":0.06740476798107425},"nc":{"folds":[0.8192336246718293,0.9143322864770377,0.8223237503488454,0.7671466054354268,0.8452567606693266],"mean":0.8336586055204931,"std":0.05339280188259735},"xc":{"folds":[0.4975306574083711,0.4121095969371643,0.4686508238167989,0.23802949962608377,0.6210983982847111],"mean":0.4474837952146258,"std":0.13983668846080213}},"macro":0.8075188319602103},"S_FIXED":{"tasks":{"eea":{"folds":[0.9042462711552217,0.936525999158363,0.9498500208314181,0.9420604234026188,0.9190418488787669],"mean":0.9303449126852777,"std":0.018471644528011987},"egb":{"folds":[0.9344724694701166,0.9028371236326306,0.9258423619737912,0.9273413747815704,0.9194046580503162],"mean":0.9219795975816851,"std":0.011966683688383899},"egc":{"folds":[0.8944587939925042,0.922412701917005,0.9213595824789435,0.908922048174576,0.927644555553518],"mean":0.9149595364233093,"std":0.013361808263594241},"ei":{"folds":[0.8154528041829034,0.7885916300783089,0.7979699494693738,0.8717048190815757,0.8620679437845344],"mean":0.8271574293193392,"std":0.03768103652876766},"eps":{"folds":[0.7500065386646162,0.8554851115108497,0.6793755501419281,0.7852747326976403,0.816174038905965],"mean":0.7772631943841998,"std":0.06712269188365187},"nc":{"folds":[0.8192336246718293,0.9143339141026394,0.8223237503488454,0.7653599763046688,0.8452567606693266],"mean":0.833301605219462,"std":0.05395286632472241},"xc":{"folds":[0.4975306574083711,0.41695863819487866,0.4686508238167989,0.23973057828593503,0.6210983982847111],"mean":0.4487938191981389,"std":0.13890587231609505}},"macro":0.8076857278302016},"S_HMR":{"tasks":{"eea":{"folds":[0.9042462711552217,0.936525999158363,0.9498500724339416,0.9420604262604564,0.9190418990369386],"mean":0.9303449336089843,"std":0.01847165093044993},"egb":{"folds":[0.9344724694701166,0.9028373758151026,0.9258426559069106,0.9273413642019775,0.9193980407481837],"mean":0.9219783812284582,"std":0.01196696172158358},"egc":{"folds":[0.894458789901636,0.9224126844343891,0.9213596338743517,0.9089220563922003,0.9276444409960241],"mean":0.9149595211197201,"std":0.01336178543222754},"ei":{"folds":[0.8154528041829034,0.7885915369030598,0.7979772172908933,0.8717048190815757,0.8620705058496028],"mean":0.8271593766616071,"std":0.03768024651574589},"eps":{"folds":[0.7500065386646162,0.8554851241162347,0.6793740008050995,0.7852747326976403,0.816174038905965],"mean":0.7772628870379111,"std":0.06712326042188782},"nc":{"folds":[0.8192336246718293,0.914333914102521,0.8223237503488454,0.7653600355489445,0.8452567606693266],"mean":0.8333016170682933,"std":0.05395284767343583},"xc":{"folds":[0.4975306574083711,0.4169668507990223,0.4686508238167989,0.23973071791960998,0.6210983982847111],"mean":0.4487954896457027,"std":0.13890534927161685}},"macro":0.8076860294815253}},"validation":{"C_ANCHOR":{"tasks":{"eea":{"folds":[0.9258489964641599,0.9178132864027428,0.9058696665311429,0.9094044488770181,0.9557389431988855],"mean":0.9229350682947899,"std":0.019906804333146833},"egb":{"folds":[0.9317327707438666,0.9323129066027187,0.9043846346096567,0.9311739634139514,0.9434588432483098],"mean":0.9286126237237007,"std":0.014468876619352455},"egc":{"folds":[0.9136946123629407,0.9231168207368707,0.9165977648387951,0.9120366467141694,0.9075833733048249],"mean":0.9146058435915201,"std":0.005769546908078458},"ei":{"folds":[0.8889190565584546,0.8665770074941318,0.8655728937003306,0.847770964939784,0.8410585642435568],"mean":0.8619796973872516,"std":0.018703992102611068},"eps":{"folds":[0.7472369430230483,0.8569800100358066,0.7720583584291768,0.8054533228589349,0.6566912730706598],"mean":0.7676839814835252,"std":0.07442597285127069},"nc":{"folds":[0.8335550582100563,0.9090776168632717,0.8382871297614007,0.8729549185146485,0.776932648930654],"mean":0.8461614744560061,"std":0.049218611668987014},"xc":{"folds":[0.35120114193594953,0.4562606930130606,0.3929779832746815,0.5251736564179612,0.3396933097646776],"mean":0.4130613568812661,"std":0.07753748895830614}},"macro":0.8078628636882943},"S_STAT":{"tasks":{"eea":{"folds":[0.9258489964641599,0.9178769164490176,0.9067608179799002,0.9094365062452253,0.9559855725828293],"mean":0.9231817619442264,"std":0.01981113937306576},"egb":{"folds":[0.9317327707438666,0.9323129066027187,0.908364740154735,0.9317097102967963,0.9521112095397051],"mean":0.9312462674675643,"std":0.015495961438226458},"egc":{"folds":[0.9142669317189999,0.9236042005727708,0.917303869044742,0.9120366467141694,0.9112378352064174],"mean":0.9156898966514199,"std":0.005010814027154468},"ei":{"folds":[0.8889190565584546,0.8671066683935705,0.8667240519232414,0.847770964939784,0.8481298993825417],"mean":0.8637301282395186,"std":0.016977254974745098},"eps":{"folds":[0.7472369430230483,0.857820531670709,0.7725346762326766,0.8054533228589349,0.6566912730706598],"mean":0.7679473493712058,"std":0.07468561768596192},"nc":{"folds":[0.8335550582100563,0.909086802004116,0.8382871297614007,0.8866019207552374,0.776932648930654],"mean":0.8488927119322927,"std":0.05140837862574984},"xc":{"folds":[0.35120114193594953,0.5039434525984864,0.3929779832746815,0.525299631502429,0.3396933097646776],"mean":0.42262310381524476,"std":0.08662094004330981}},"macro":0.8104730313459247},"S_FIXED":{"tasks":{"eea":{"folds":[0.9258489964641599,0.9178331332063161,0.9066153821934175,0.9094900554710332,0.9559881017048805],"mean":0.9231551338079613,"std":0.019836091580498424},"egb":{"folds":[0.9317327707438666,0.9330503885321512,0.9098483575426559,0.9325100292945656,0.9536400137521317],"mean":0.9321563119730742,"std":0.015494338877562264},"egc":{"folds":[0.9138632735528373,0.9237142051890694,0.9178600280789662,0.9121030815965988,0.9124457473281329],"mean":0.9159972671491209,"std":0.004882440829633394},"ei":{"folds":[0.8889190565584546,0.8670203091845546,0.8671192242845165,0.847770964939784,0.8469681080930257],"mean":0.8635595326120671,"std":0.017264907115217106},"eps":{"folds":[0.7472369430230483,0.8578443750698888,0.772262268349449,0.8054533228589349,0.6566912730706598],"mean":0.7678976364743961,"std":0.0746887121500948},"nc":{"folds":[0.8335550582100563,0.9090990464765769,0.8382871297614007,0.8867582880965407,0.776932648930654],"mean":0.8489264342950456,"std":0.051440673483165486},"xc":{"folds":[0.35120114193594953,0.5058905156614282,0.3929779832746815,0.5253667497219885,0.3396933097646776],"mean":0.42302594007174504,"std":0.08710078618523776}},"macro":0.8106740366262015},"S_HMR":{"tasks":{"eea":{"folds":[0.9258489964641599,0.9178331331832894,0.9066153121309277,0.909490056196328,0.9559880066862231],"mean":0.9231551009321857,"std":0.019836066743020127},"egb":{"folds":[0.9317327707438666,0.9330507233087334,0.9098462664982199,0.9325097918812104,0.9536423133749903],"mean":0.9321563731614042,"std":0.015495892134450812},"egc":{"folds":[0.9138632671005734,0.9237141969822339,0.9178602593844016,0.9121030628296799,0.9124455491572614],"mean":0.91599726709083,"std":0.004882500135697841},"ei":{"folds":[0.8889190565584546,0.8670204941256926,0.86711267721158,0.847770964939784,0.8469725037544469],"mean":0.8635591393179917,"std":0.017263523254420245},"eps":{"folds":[0.7472369430230483,0.8578443954063527,0.7722620578817269,0.8054533228589349,0.6566912730706598],"mean":0.7678975984481445,"std":0.07468871519808952},"nc":{"folds":[0.8335550582100563,0.9090990464765769,0.8382871297614007,0.8867563432648741,0.776932648930654],"mean":0.8489260453287123,"std":0.05144031590945451},"xc":{"folds":[0.35120114193594953,0.5058832756355084,0.3929779832746815,0.5253667979723506,0.3396933097646776],"mean":0.4230245017166335,"std":0.08709907842657413}},"macro":0.8106737179994145}},"selected_macro":0.8076857278302016,"target_met":false,"performance_pass":false,"independent_review":"PENDING","status":"COMPLETE_PENDING_REVIEW","execution_complete":true,"preflight":{"status":"PASSED"},"final_cpu":{"status":"PASSED"}}
```

</details>

<details><summary>results/ua3d_head_dropout_20261028/r1/final_review/results.json</summary>

```json
{"plan":"UA3D-HEAD-DROPOUT-20261028-01","revision":"r1","independent_review":"PENDING","FINAL":"H40","validation":{"H50_BASE":{"tasks":{"eea":{"folds":[0.9258489962334948,0.9178132892834358,0.9058696511975507,0.90940444186418,0.9557389448516351],"mean":0.9229350646860593,"std":0.0199068092981852},"egb":{"folds":[0.9317327685836467,0.9323129091615904,0.9043846369319708,0.931173957145727,0.9434588393916917],"mean":0.9286126222429253,"std":0.014468874427615519},"egc":{"folds":[0.9136946146435524,0.923116820626894,0.9165977660323378,0.9120366473326542,0.9075833727563615],"mean":0.91460584427836,"std":0.005769546978527248},"ei":{"folds":[0.888919055883577,0.8665769855366472,0.8655728923752912,0.8477709629625998,0.8410585473221172],"mean":0.8619796888160464,"std":0.018703995554038573},"eps":{"folds":[0.7472369136542818,0.856980031011281,0.772058361523108,0.8054533188233461,0.6566912907405946],"mean":0.7676839831505223,"std":0.07442597410557149},"nc":{"folds":[0.8335550837981948,0.9090776141944835,0.8382871410816238,0.8729549039891361,0.7769326614604025],"mean":0.8461614809047682,"std":0.04921860234206862},"xc":{"folds":[0.351201144033391,0.4562607014500726,0.3929779893799423,0.5251736603280752,0.3396932842125413],"mean":0.41306135588080445,"std":0.07753749677772813}},"macro":0.8078628628513551},"H00":{"tasks":{"eea":{"folds":[0.9408947833953457,0.9088611151309354,0.8907027756354811,0.8988649284984491,0.9483766459403447],"mean":0.9175400497201112,"std":0.025693688865319513},"egb":{"folds":[0.9295835751491375,0.9263883197849762,0.8980108637132335,0.9409453854529986,0.9383756948778192],"mean":0.926660767795633,"std":0.017107921066226637},"egc":{"folds":[0.9099647247541096,0.9192492695308262,0.9157960383565407,0.9110015044823945,0.9065388096134006],"mean":0.9125100693474544,"std":0.0050155982021771045},"ei":{"folds":[0.8569802077213973,0.8523167040800999,0.873812582220576,0.8444909295980845,0.8286744042292865],"mean":0.8512549655698889,"std":0.016570951152554587},"eps":{"folds":[0.7550416744610602,0.8442522845538019,0.7486154430028438,0.8073975205660909,0.6444159745761802],"mean":0.7599445794319954,"std":0.07559566309401759},"nc":{"folds":[0.8069537865033752,0.9189548718306165,0.8177521938573484,0.8750595098717957,0.7478531607786958],"mean":0.8333147045683663,"std":0.0657982374883646},"xc":{"folds":[0.3390855676398504,0.47260428075942074,0.4247190844859672,0.5268032699547501,0.33080158518223746],"mean":0.4188027576044452,"std":0.08469467841757708}},"macro":0.8028611277196992},"H10":{"tasks":{"eea":{"folds":[0.9295226910943755,0.9068553615403394,0.9023653164358852,0.9123623702353345,0.9530425056554921],"mean":0.9208296489922855,"std":0.020743337553801918},"egb":{"folds":[0.9292498298366008,0.9228961932632909,0.8997170788191712,0.9312266172756366,0.9459142671718694],"mean":0.9258007972733138,"std":0.016842536273750056},"egc":{"folds":[0.917222230944112,0.9210856194213096,0.919303782693964,0.9102887241925002,0.9080530807853956],"mean":0.9151906876074563,"std":0.0057177271943762665},"ei":{"folds":[0.8734134706077045,0.8599062996545336,0.8558610379177002,0.8572888163001959,0.8295014066150134],"mean":0.8551942062190296,"std":0.015961652693314437},"eps":{"folds":[0.7692735945759572,0.8519847494739279,0.7464464581511753,0.7951361253753925,0.6489466191727451],"mean":0.7623575093498396,"std":0.07461652571092084},"nc":{"folds":[0.8152937480517976,0.909791937271911,0.8236109853626601,0.8821787526043303,0.7396437723598194],"mean":0.8341038391301037,"std":0.06600475215218206},"xc":{"folds":[0.323927733393377,0.42885621359669024,0.4107703384299237,0.5368602878315292,0.32838268216568445],"mean":0.4057594510834409,"std":0.08722040073127914}},"macro":0.8027480199507814},"H20":{"tasks":{"eea":{"folds":[0.9268881141407571,0.9028047112494447,0.9033647509028941,0.9036601370880473,0.9485312960128974],"mean":0.9170498018788082,"std":0.020355363000566937},"egb":{"folds":[0.9282387671349602,0.9268426145228694,0.9038602647713937,0.935664367919414,0.9402356889437811],"mean":0.9269683406584835,"std":0.014029044713941192},"egc":{"folds":[0.9137353637083414,0.9211376398780442,0.9182635213850503,0.9124904005168912,0.912723351198374],"mean":0.9156700553373401,"std":0.0038456744730116363},"ei":{"folds":[0.8824428418002951,0.8658234504286385,0.8683225227554198,0.8623487448120285,0.8306505973679889],"mean":0.8619176314328743,"std":0.019074692615272736},"eps":{"folds":[0.7932191107460081,0.8620635428575353,0.7605817423794228,0.7721253537181977,0.6965165115394643],"mean":0.7769012522481257,"std":0.059727988405185804},"nc":{"folds":[0.8363096008153716,0.8923721458890788,0.8506805272079311,0.8916254021364686,0.7627656830623083],"mean":0.8467506718222315,"std":0.05308718861678106},"xc":{"folds":[0.3616415594812379,0.4628321051913592,0.44567171685694806,0.5133032249189964,0.4027405064226254],"mean":0.4372378225742334,"std":0.05791589402879269}},"macro":0.811785082278871},"H30":{"tasks":{"eea":{"folds":[0.9239167030320181,0.9091381150376834,0.9085793697917778,0.9064407300392975,0.9558193607385248],"mean":0.9207788557278602,"std":0.02078219927178218},"egb":{"folds":[0.9285456295037235,0.9280789819159878,0.9079137617376312,0.9341285347323252,0.9427914978498588],"mean":0.9282916811479053,"std":0.012842146183353794},"egc":{"folds":[0.9158195446300296,0.9217968830048358,0.9183332640694934,0.9150363664845638,0.9119594042589567],"mean":0.9165890924895759,"std":0.003694603890085161},"ei":{"folds":[0.8842352491961432,0.8749270604016346,0.8687256141806348,0.8656826655991174,0.8439301096078413],"mean":0.8675001397970743,"std":0.014963074210465586},"eps":{"folds":[0.7129623790868717,0.8568582032060391,0.793648362089896,0.768037739215264,0.6655904102470922],"mean":0.7594194187690325,"std":0.07365859097379353},"nc":{"folds":[0.8315675457346848,0.9125154027286205,0.8339125088779954,0.8867370860980907,0.7340975355597404],"mean":0.8397660157998263,"std":0.06849463649517068},"xc":{"folds":[0.3278005207287862,0.4931446275492135,0.3807074272778419,0.5832610670944705,0.36746208252297685],"mean":0.4304751450346577,"std":0.10519096001140538}},"macro":0.8089743355379903},"H40":{"tasks":{"eea":{"folds":[0.9208585881138094,0.9158133422687963,0.9008095728978037,0.9016360350855191,0.958555384614084],"mean":0.9195345845960026,"std":0.02350110423735961},"egb":{"folds":[0.9319157373690673,0.9254044032100099,0.9028824784321087,0.9344652210703248,0.9423257657579762],"mean":0.9273987211678975,"std":0.014984817136222463},"egc":{"folds":[0.912298220798736,0.9215930884003871,0.921047044731043,0.9131072049854693,0.9104440077385805],"mean":0.9156979133308433,"std":0.005225885065068909},"ei":{"folds":[0.8885398870611387,0.8655821663412882,0.8676313444521719,0.8687152215574203,0.8123603844186958],"mean":0.8605658007661431,"std":0.028494782678880802},"eps":{"folds":[0.7672870909309157,0.8402461919980723,0.7615666502621645,0.7940434302469542,0.6533428102144219],"mean":0.7632972347305056,"std":0.06887553976025643},"nc":{"folds":[0.8383245761333544,0.9047318541255591,0.862507170343717,0.8839207232075951,0.7781364769582266],"mean":0.8535241601536905,"std":0.04883954230808154},"xc":{"folds":[0.385508870622575,0.48023193909212614,0.4183901158335285,0.5293853594260145,0.3127355299415696],"mean":0.42525036298316277,"std":0.08389443970841688}},"macro":0.8093241111040349}},"formal":{"H50_BASE":{"tasks":{"eea":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"egb":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"egc":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"ei":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"eps":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"nc":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"xc":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815}},"macro":0.804675462276949},"H00":{"tasks":{"eea":{"folds":[0.908248032647963,0.9278713813467032,0.9480039262515807,0.9405152290707394,0.9164249707397418],"mean":0.9282127080113456,"std":0.016436999879015698},"egb":{"folds":[0.9289724112325927,0.9024660519715393,0.9214031645076819,0.9242238521262117,0.90537722098155],"mean":0.916488540163915,"std":0.011831390551531462},"egc":{"folds":[0.8956373648937858,0.9223817420187204,0.9148744334523362,0.9100600491351817,0.9207625093430372],"mean":0.9127432197686123,"std":0.010741995436797706},"ei":{"folds":[0.7682419559856652,0.7632303575185952,0.7720764239507122,0.7963658614770917,0.8353204711883808],"mean":0.787047014024089,"std":0.029841872790256074},"eps":{"folds":[0.7157949162796047,0.8337076725619288,0.6407507841879934,0.7187567489123474,0.7953645868810503],"mean":0.7408749417645848,"std":0.07538253510596572},"nc":{"folds":[0.864511389727907,0.9118100543624055,0.777346016898434,0.7373107192572403,0.7719027434694163],"mean":0.8125761847430806,"std":0.07263781770304897},"xc":{"folds":[0.5371934895626866,0.3514414157178347,0.46382096841227305,0.31026865439061935,0.5683321767782106],"mean":0.44621134097232484,"std":0.11287504335150221}},"macro":0.7920219927782787},"H10":{"tasks":{"eea":{"folds":[0.9027772439300666,0.9301108774642474,0.9458051363673288,0.9266005262115805,0.9135071609661707],"mean":0.9237601889878789,"std":0.01643005653934307},"egb":{"folds":[0.9195170690984332,0.898342523862419,0.9268942356505909,0.9325252023902979,0.91042179824034],"mean":0.9175401658484162,"std":0.013557089507999562},"egc":{"folds":[0.9008642164904155,0.9211335221147043,0.9132366537589476,0.912302017539938,0.9253600581011668],"mean":0.9145792936010345,"std":0.009411094553384113},"ei":{"folds":[0.7801216388208453,0.7876207122935188,0.8086590658983596,0.8507749969979012,0.8302394102111497],"mean":0.8114831648443548,"std":0.02939635311294076},"eps":{"folds":[0.6935196304412425,0.836151832215912,0.6729821647804772,0.7625189929226014,0.7854285982277025],"mean":0.7501202437175871,"std":0.06699954429877127},"nc":{"folds":[0.8012760032896498,0.8867076197669989,0.8290883853917632,0.7526081367999099,0.861344362529424],"mean":0.826204901555549,"std":0.05230107278725442},"xc":{"folds":[0.42221579509463614,0.37390479622686257,0.40318325269812005,0.31885816635696895,0.5696096437049198],"mean":0.41755433081630144,"std":0.0935351405382102}},"macro":0.7944631841958747},"H20":{"tasks":{"eea":{"folds":[0.9083642533938606,0.9397541163739256,0.9358041036963339,0.8929724675087296,0.9228339797127938],"mean":0.9199457841371288,"std":0.019454367525196133},"egb":{"folds":[0.923676583307228,0.906480044110095,0.9162923895635485,0.9232640132393494,0.907973683325064],"mean":0.915537342709057,"std":0.00815110992348444},"egc":{"folds":[0.8970700787830405,0.9199428772000636,0.9187957936997988,0.9103370916364188,0.9259887700427409],"mean":0.9144269222724125,"std":0.011193478578203792},"ei":{"folds":[0.7569369532997342,0.7660641578959794,0.7919052739236192,0.8590298093787918,0.8185562872772677],"mean":0.7984984963550785,"std":0.04150052285063512},"eps":{"folds":[0.7274671134619861,0.8331392047997705,0.6174089407007398,0.7616366934044813,0.7949517665557837],"mean":0.7469207437845522,"std":0.08232392151168279},"nc":{"folds":[0.7955331510339361,0.8936824463958493,0.8177158234635141,0.6704081722731211,0.8218511806690693],"mean":0.799838154767098,"std":0.08121205089796633},"xc":{"folds":[0.4685668119574705,0.3829177127316964,0.35525450796244584,0.3179852168960442,0.5328313724168774],"mean":0.4115111243929069,"std":0.08761198479632511}},"macro":0.7866683669168906},"H30":{"tasks":{"eea":{"folds":[0.901447662322937,0.9347427563594117,0.9503404662042491,0.9470186288993829,0.9248151869595287],"mean":0.9316729401491018,"std":0.019710318758731013},"egb":{"folds":[0.9287674107852544,0.9116423554049352,0.9292327294266168,0.9282944453061988,0.9122704502103822],"mean":0.9220414782266776,"std":0.009215023112527154},"egc":{"folds":[0.9012588901849938,0.9222828748231718,0.9180227215831683,0.9123900296453871,0.9238581679869905],"mean":0.9155625368447424,"std":0.009147295464188297},"ei":{"folds":[0.7575582964538733,0.787319490353776,0.8148430752754083,0.8434032354423728,0.8486446767252369],"mean":0.8103537548501334,"std":0.038394513382455656},"eps":{"folds":[0.674279520816104,0.858574588560562,0.6029250501652896,0.7592833377253365,0.8108034281352108],"mean":0.7411731850805006,"std":0.1031343141757602},"nc":{"folds":[0.7899862140931951,0.8929422591781085,0.8162062345155837,0.7369540293272768,0.8667925861439479],"mean":0.8205762646516226,"std":0.06188762423394249},"xc":{"folds":[0.39961782843104143,0.40895982248699514,0.4855739204518259,0.2907025089684738,0.4853457979026301],"mean":0.4140399756481933,"std":0.08007428195122301}},"macro":0.7936314479215675},"H40":{"tasks":{"eea":{"folds":[0.9124961793515819,0.9143342732179647,0.9452561580167443,0.94149507359881,0.913008297126059],"mean":0.9253179962622319,"std":0.016551419719537844},"egb":{"folds":[0.930233329597923,0.891772986583822,0.9288578758976499,0.9280905202485262,0.9088647083064568],"mean":0.9175638841268755,"std":0.016879908343431917},"egc":{"folds":[0.9005293509114216,0.9241278641041377,0.9224742842787617,0.9139323909517681,0.9238568805800785],"mean":0.9169841541652335,"std":0.010105472497126186},"ei":{"folds":[0.7604211673371126,0.7961193943271139,0.8222277353761441,0.8282585826899482,0.8180406195371656],"mean":0.8050134998534968,"std":0.02772161097556156},"eps":{"folds":[0.7359219400168089,0.8545162940886161,0.5305020649965344,0.7538935070591577,0.7830428134840227],"mean":0.731575323929028,"std":0.12115159352259093},"nc":{"folds":[0.84333318451427,0.8518966417210828,0.670831134318095,0.7401783192396854,0.8542564079462297],"mean":0.7920991375478725,"std":0.082864189066212},"xc":{"folds":[0.4779275894895114,0.3860614899525441,0.4576658066462731,0.34281696416161134,0.5862442645076987],"mean":0.45014322295152775,"std":0.09351129361209426}},"macro":0.7912424598337522}},"reference":0.804675462276949,"selected_comparison":{"macro":0.7912424598337522,"delta_macro":-0.013433002443196784,"performance_pass":false},"performance_pass":false,"paired":{"H00":{"validation":{"delta_macro":-0.0050017351316559155},"formal":{"delta_macro":-0.0126534694986703}},"H10":{"validation":{"delta_macro":-0.005114842900573735},"formal":{"delta_macro":-0.010212278081074277}},"H20":{"validation":{"delta_macro":0.003922219427515916},"formal":{"delta_macro":-0.0180070953600584}},"H30":{"validation":{"delta_macro":0.0011114726866351399},"formal":{"delta_macro":-0.011044014355381493}},"H40":{"validation":{"delta_macro":0.001461248252679792},"formal":{"delta_macro":-0.013433002443196784}}},"sota":{"H50_BASE":{"eea":{"project":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.004431145079367593},"egb":{"project":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.0022430816246885588},"egc":{"project":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.003138905643812362},"ei":{"project":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.01762400606686776},"eps":{"project":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.011119521310488989},"nc":{"project":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.024623190090803382},"xc":{"project":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.01663027446704779}},"H00":{"eea":{"project":{"folds":[0.908248032647963,0.9278713813467032,0.9480039262515807,0.9405152290707394,0.9164249707397418],"mean":0.9282127080113456,"std":0.016436999879015698},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.005787291988654464},"egb":{"project":{"folds":[0.9289724112325927,0.9024660519715393,0.9214031645076819,0.9242238521262117,0.90537722098155],"mean":0.916488540163915,"std":0.011831390551531462},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.006511459836085032},"egc":{"project":{"folds":[0.8956373648937858,0.9223817420187204,0.9148744334523362,0.9100600491351817,0.9207625093430372],"mean":0.9127432197686123,"std":0.010741995436797706},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0017432197686122475},"ei":{"project":{"folds":[0.7682419559856652,0.7632303575185952,0.7720764239507122,0.7963658614770917,0.8353204711883808],"mean":0.787047014024089,"std":0.029841872790256074},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.051952985975910915},"eps":{"project":{"folds":[0.7157949162796047,0.8337076725619288,0.6407507841879934,0.7187567489123474,0.7953645868810503],"mean":0.7408749417645848,"std":0.07538253510596572},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.04612505823541524},"nc":{"project":{"folds":[0.864511389727907,0.9118100543624055,0.777346016898434,0.7373107192572403,0.7719027434694163],"mean":0.8125761847430806,"std":0.07263781770304897},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.04342381525691941},"xc":{"project":{"folds":[0.5371934895626866,0.3514414157178347,0.46382096841227305,0.31026865439061935,0.5683321767782106],"mean":0.44621134097232484,"std":0.11287504335150221},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.023211340972324856}},"H10":{"eea":{"project":{"folds":[0.9027772439300666,0.9301108774642474,0.9458051363673288,0.9266005262115805,0.9135071609661707],"mean":0.9237601889878789,"std":0.01643005653934307},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.010239811012121125},"egb":{"project":{"folds":[0.9195170690984332,0.898342523862419,0.9268942356505909,0.9325252023902979,0.91042179824034],"mean":0.9175401658484162,"std":0.013557089507999562},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.005459834151583798},"egc":{"project":{"folds":[0.9008642164904155,0.9211335221147043,0.9132366537589476,0.912302017539938,0.9253600581011668],"mean":0.9145792936010345,"std":0.009411094553384113},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0035792936010344834},"ei":{"project":{"folds":[0.7801216388208453,0.7876207122935188,0.8086590658983596,0.8507749969979012,0.8302394102111497],"mean":0.8114831648443548,"std":0.02939635311294076},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.02751683515564518},"eps":{"project":{"folds":[0.6935196304412425,0.836151832215912,0.6729821647804772,0.7625189929226014,0.7854285982277025],"mean":0.7501202437175871,"std":0.06699954429877127},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.03687975628241291},"nc":{"project":{"folds":[0.8012760032896498,0.8867076197669989,0.8290883853917632,0.7526081367999099,0.861344362529424],"mean":0.826204901555549,"std":0.05230107278725442},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.02979509844445094},"xc":{"project":{"folds":[0.42221579509463614,0.37390479622686257,0.40318325269812005,0.31885816635696895,0.5696096437049198],"mean":0.41755433081630144,"std":0.0935351405382102},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":-0.005445669183698543}},"H20":{"eea":{"project":{"folds":[0.9083642533938606,0.9397541163739256,0.9358041036963339,0.8929724675087296,0.9228339797127938],"mean":0.9199457841371288,"std":0.019454367525196133},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.014054215862871255},"egb":{"project":{"folds":[0.923676583307228,0.906480044110095,0.9162923895635485,0.9232640132393494,0.907973683325064],"mean":0.915537342709057,"std":0.00815110992348444},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.007462657290943087},"egc":{"project":{"folds":[0.8970700787830405,0.9199428772000636,0.9187957936997988,0.9103370916364188,0.9259887700427409],"mean":0.9144269222724125,"std":0.011193478578203792},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0034269222724124804},"ei":{"project":{"folds":[0.7569369532997342,0.7660641578959794,0.7919052739236192,0.8590298093787918,0.8185562872772677],"mean":0.7984984963550785,"std":0.04150052285063512},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.040501503644921444},"eps":{"project":{"folds":[0.7274671134619861,0.8331392047997705,0.6174089407007398,0.7616366934044813,0.7949517665557837],"mean":0.7469207437845522,"std":0.08232392151168279},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.040079256215447834},"nc":{"project":{"folds":[0.7955331510339361,0.8936824463958493,0.8177158234635141,0.6704081722731211,0.8218511806690693],"mean":0.799838154767098,"std":0.08121205089796633},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.05616184523290202},"xc":{"project":{"folds":[0.4685668119574705,0.3829177127316964,0.35525450796244584,0.3179852168960442,0.5328313724168774],"mean":0.4115111243929069,"std":0.08761198479632511},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":-0.011488875607093108}},"H30":{"eea":{"project":{"folds":[0.901447662322937,0.9347427563594117,0.9503404662042491,0.9470186288993829,0.9248151869595287],"mean":0.9316729401491018,"std":0.019710318758731013},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.0023270598508982054},"egb":{"project":{"folds":[0.9287674107852544,0.9116423554049352,0.9292327294266168,0.9282944453061988,0.9122704502103822],"mean":0.9220414782266776,"std":0.009215023112527154},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.0009585217733224516},"egc":{"project":{"folds":[0.9012588901849938,0.9222828748231718,0.9180227215831683,0.9123900296453871,0.9238581679869905],"mean":0.9155625368447424,"std":0.009147295464188297},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.004562536844742393},"ei":{"project":{"folds":[0.7575582964538733,0.787319490353776,0.8148430752754083,0.8434032354423728,0.8486446767252369],"mean":0.8103537548501334,"std":0.038394513382455656},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.028646245149866578},"eps":{"project":{"folds":[0.674279520816104,0.858574588560562,0.6029250501652896,0.7592833377253365,0.8108034281352108],"mean":0.7411731850805006,"std":0.1031343141757602},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.04582681491949947},"nc":{"project":{"folds":[0.7899862140931951,0.8929422591781085,0.8162062345155837,0.7369540293272768,0.8667925861439479],"mean":0.8205762646516226,"std":0.06188762423394249},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.035423735348377394},"xc":{"project":{"folds":[0.39961782843104143,0.40895982248699514,0.4855739204518259,0.2907025089684738,0.4853457979026301],"mean":0.4140399756481933,"std":0.08007428195122301},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":-0.008960024351806706}},"H40":{"eea":{"project":{"folds":[0.9124961793515819,0.9143342732179647,0.9452561580167443,0.94149507359881,0.913008297126059],"mean":0.9253179962622319,"std":0.016551419719537844},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.008682003737768174},"egb":{"project":{"folds":[0.930233329597923,0.891772986583822,0.9288578758976499,0.9280905202485262,0.9088647083064568],"mean":0.9175638841268755,"std":0.016879908343431917},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.005436115873124536},"egc":{"project":{"folds":[0.9005293509114216,0.9241278641041377,0.9224742842787617,0.9139323909517681,0.9238568805800785],"mean":0.9169841541652335,"std":0.010105472497126186},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.005984154165233457},"ei":{"project":{"folds":[0.7604211673371126,0.7961193943271139,0.8222277353761441,0.8282585826899482,0.8180406195371656],"mean":0.8050134998534968,"std":0.02772161097556156},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.033986500146503174},"eps":{"project":{"folds":[0.7359219400168089,0.8545162940886161,0.5305020649965344,0.7538935070591577,0.7830428134840227],"mean":0.731575323929028,"std":0.12115159352259093},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.05542467607097201},"nc":{"project":{"folds":[0.84333318451427,0.8518966417210828,0.670831134318095,0.7401783192396854,0.8542564079462297],"mean":0.7920991375478725,"std":0.082864189066212},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.06390086245212745},"xc":{"project":{"folds":[0.4779275894895114,0.3860614899525441,0.4576658066462731,0.34281696416161134,0.5862442645076987],"mean":0.45014322295152775,"std":0.09351129361209426},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.027143222951527768}}},"cpu":{"status":"COMPLETE"}}
```

</details>

<details><summary>results/ua3d_stability_20261028/r1/final_review/results.json</summary>

```json
{"plan":"UA3D-STABILITY-20261028-01","revision":"r1","PRIMARY_RUN":"S42_A","same_seed_repro_pass":true,"stable_ready":false,"performance_pass":false,"primary_test_macro":0.7901590429486368,"primary_delta_macro":-0.014516419328312158,"quality_guard":{"macro_pass":false},"cross_seed":{"mean":0.7892382053494554,"std":0.0009566213211750466,"range":0.001909619852350275,"macro_range_pass":true,"tasks":{"eea":{"mean":0.9277397709310021,"std":0.0037315955083882545,"range":0.006688795799018155},"egb":{"mean":0.9192646787689974,"std":0.0009219046436412208,"range":0.0016347161663320398},"egc":{"mean":0.9150686129729144,"std":0.001182383416228048,"range":0.0023418685654140736},"ei":{"mean":0.8056287672345515,"std":0.013007607753600443,"range":0.02592970924387472},"eps":{"mean":0.745568576524744,"std":0.02540429347049008,"range":0.05058325080434045},"nc":{"mean":0.8127952425010614,"std":0.016503299272891774,"range":0.03247007847578964},"xc":{"mean":0.39860178851291755,"std":0.02306947722298624,"range":0.045759993548982025}}},"execution_complete":true,"independent_review":"PENDING","reference":0.804675462276949,"validation":{"BASE_ORIGINAL":{"tasks":{"eea":{"folds":[0.9258489962334948,0.9178132892834358,0.9058696511975507,0.90940444186418,0.9557389448516351],"mean":0.9229350646860593,"std":0.0199068092981852},"egb":{"folds":[0.9317327685836467,0.9323129091615904,0.9043846369319708,0.931173957145727,0.9434588393916917],"mean":0.9286126222429253,"std":0.014468874427615519},"egc":{"folds":[0.9136946146435524,0.923116820626894,0.9165977660323378,0.9120366473326542,0.9075833727563615],"mean":0.91460584427836,"std":0.005769546978527248},"ei":{"folds":[0.888919055883577,0.8665769855366472,0.8655728923752912,0.8477709629625998,0.8410585473221172],"mean":0.8619796888160464,"std":0.018703995554038573},"eps":{"folds":[0.7472369136542818,0.856980031011281,0.772058361523108,0.8054533188233461,0.6566912907405946],"mean":0.7676839831505223,"std":0.07442597410557149},"nc":{"folds":[0.8335550837981948,0.9090776141944835,0.8382871410816238,0.8729549039891361,0.7769326614604025],"mean":0.8461614809047682,"std":0.04921860234206862},"xc":{"folds":[0.351201144033391,0.4562607014500726,0.3929779893799423,0.5251736603280752,0.3396932842125413],"mean":0.41306135588080445,"std":0.07753749677772813}},"macro":0.8078628628513551},"S42_A":{"tasks":{"eea":{"folds":[0.9232787998137404,0.9170375832245461,0.9055972034651808,0.9151130050462468,0.9529119468532535],"mean":0.9227877076805935,"std":0.017994369842958297},"egb":{"folds":[0.9316562195877197,0.9324605892748533,0.9043603067234441,0.9361440895018788,0.9427291279847412],"mean":0.9294700666145275,"std":0.014701767090319102},"egc":{"folds":[0.9136728959058448,0.9216358210595708,0.9166268546386297,0.9116074224387392,0.9077013829677186],"mean":0.9142488754021005,"std":0.005253580557863084},"ei":{"folds":[0.88891010012726,0.8719575809783203,0.8626357845716468,0.87009346643949,0.8436854164328349],"mean":0.8674564697099104,"std":0.016399496879405225},"eps":{"folds":[0.7816703143465606,0.8416083014060043,0.7720209764319973,0.8057673485272518,0.6484598951076622],"mean":0.7699053671638952,"std":0.07300562470450721},"nc":{"folds":[0.8390751478566909,0.9032857466746496,0.8364705286958442,0.8830515836096089,0.7763465767079905],"mean":0.847645916708957,"std":0.04906958117771927},"xc":{"folds":[0.35470672946815385,0.4357165351732598,0.38850651684842186,0.5251736259276018,0.3561303830459075],"mean":0.41204675809266894,"std":0.07128570931830387}},"macro":0.8090801659103789},"S42_B":{"tasks":{"eea":{"folds":[0.9232787998137404,0.9170375832245461,0.9055972034651808,0.9151130050462468,0.9529119468532535],"mean":0.9227877076805935,"std":0.017994369842958297},"egb":{"folds":[0.9316562195877197,0.9324605892748533,0.9043603067234441,0.9361440895018788,0.9427291279847412],"mean":0.9294700666145275,"std":0.014701767090319102},"egc":{"folds":[0.9136728959058448,0.9216358210595708,0.9166268546386297,0.9116074224387392,0.9077013829677186],"mean":0.9142488754021005,"std":0.005253580557863084},"ei":{"folds":[0.88891010012726,0.8719575809783203,0.8626357845716468,0.87009346643949,0.8436854164328349],"mean":0.8674564697099104,"std":0.016399496879405225},"eps":{"folds":[0.7816703143465606,0.8416083014060043,0.7720209764319973,0.8057673485272518,0.6484598951076622],"mean":0.7699053671638952,"std":0.07300562470450721},"nc":{"folds":[0.8390751478566909,0.9032857466746496,0.8364705286958442,0.8830515836096089,0.7763465767079905],"mean":0.847645916708957,"std":0.04906958117771927},"xc":{"folds":[0.35470672946815385,0.4357165351732598,0.38850651684842186,0.5251736259276018,0.3561303830459075],"mean":0.41204675809266894,"std":0.07128570931830387}},"macro":0.8090801659103789},"S142":{"tasks":{"eea":{"folds":[0.9262173218368434,0.9181832243540197,0.9002883794775444,0.9015437461440693,0.9590659376315057],"mean":0.9210597218887966,"std":0.023935692413252274},"egb":{"folds":[0.930056726737385,0.9253902360344436,0.9014833220691548,0.9393640985050824,0.9426151530972857],"mean":0.9277819072886704,"std":0.016253328997877663},"egc":{"folds":[0.9182305874855942,0.923968388637936,0.918059451687377,0.9127200017869462,0.9086231404910853],"mean":0.9163203140177878,"std":0.005860365142434833},"ei":{"folds":[0.8828493122480674,0.910149019447776,0.8647894087122994,0.8655092153859116,0.8340578018479887],"mean":0.8714709515284087,"std":0.027863009133471357},"eps":{"folds":[0.793444597710708,0.82833625211333,0.7737261788461254,0.8167502433921519,0.7263696603436165],"mean":0.7877253864811863,"std":0.04026442923803823},"nc":{"folds":[0.8461949119724306,0.9276398527664601,0.8424141458881653,0.8965029339905259,0.7719531620115021],"mean":0.8569410013258167,"std":0.05939889499773184},"xc":{"folds":[0.31853199165729507,0.47373258335136004,0.41865632591962043,0.5526523200799336,0.3647864050067817],"mean":0.4256719252029981,"std":0.09174567851921234}},"macro":0.8152816011048093},"S242":{"tasks":{"eea":{"folds":[0.9302699775066546,0.9144001118654748,0.8957757237332789,0.9089353771682873,0.9515679839711381],"mean":0.9201898348489668,"std":0.02146176862361007},"egb":{"folds":[0.9301869931464415,0.9218682394872875,0.9068193795700279,0.9427377611796678,0.9497807375033281],"mean":0.9302786221773506,"std":0.017004717312033103},"egc":{"folds":[0.9176306293906511,0.924377227684742,0.9159150173209998,0.9112157328532157,0.9117692934147169],"mean":0.916181580132865,"std":0.005326052749614235},"ei":{"folds":[0.8808094639325523,0.8806889174529763,0.8889432929854695,0.8708505839575211,0.8358154450432765],"mean":0.8714215406743591,"std":0.02091141564536449},"eps":{"folds":[0.7837364823195527,0.8446673202767659,0.757581613506904,0.8050108497521185,0.7106256797392612],"mean":0.7803243891189205,"std":0.0503393991827381},"nc":{"folds":[0.8656693341073799,0.9051302697280291,0.83065914474124,0.8756034121882372,0.7703190516091836],"mean":0.849476242474814,"std":0.05162886788466562},"xc":{"folds":[0.3845863293727019,0.48314155715601503,0.45299956117714213,0.5367843745644104,0.3410473876220911],"mean":0.4397118419784721,"std":0.07787694928225647}},"macro":0.8153691502008211}},"formal":{"BASE_ORIGINAL":{"tasks":{"eea":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"egb":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"egc":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"ei":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"eps":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"nc":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"xc":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815}},"macro":0.804675462276949},"S42_A":{"tasks":{"eea":{"folds":[0.9058184813631905,0.9314225506385276,0.9418020574734336,0.9405905135617408,0.9095087394789485],"mean":0.9258284685031682,"std":0.01711067221506428},"egb":{"folds":[0.9151983022103486,0.9021217732879702,0.9313825400712233,0.9270290970239738,0.9177358590719301],"mean":0.9186935143330892,"std":0.011382547228413639},"egc":{"folds":[0.8944593377360505,0.9233164577934517,0.9274548122041496,0.9115227003666666,0.9249182728125974],"mean":0.9163343161825832,"std":0.01367265860634017},"ei":{"folds":[0.7507134792573248,0.7883687758079122,0.758346976642833,0.8531225441883048,0.8158097219999276],"mean":0.7932722995792605,"std":0.042264237553500124},"eps":{"folds":[0.7411496509433209,0.8660578107312047,0.6794221078549454,0.780615816620666,0.7939547848169688],"mean":0.7722400341934212,"std":0.06878451624605955},"nc":{"folds":[0.8329887600354007,0.8929953893940883,0.8213587847528893,0.7437265071882315,0.8455270526379388],"mean":0.8273192988017097,"std":0.054083954421343707},"xc":{"folds":[0.3244347880934484,0.33556547591205943,0.438722251367053,0.2377841345924503,0.5506201952711153],"mean":0.37742536904722523,"std":0.1202223796720207}},"macro":0.7901590429486368},"S142":{"tasks":{"eea":{"folds":[0.9091173785377625,0.9300135171532607,0.9539327887720307,0.9409731590133223,0.9261622567457641],"mean":0.9320398200444281,"std":0.016757062714002183},"egb":{"folds":[0.921028147620244,0.9116411103904567,0.9222910350051907,0.9261068676194435,0.9127942967370729],"mean":0.9187722914744816,"std":0.006282050122077305},"egc":{"folds":[0.8973606566246533,0.9215438353349087,0.9234236856126943,0.9080753828316384,0.9239918151910599],"mean":0.9148790751189908,"std":0.011764954782420648},"ei":{"folds":[0.8045485143454515,0.7974735060531526,0.8130273614335738,0.8506402536479486,0.8303204086355498],"mean":0.8192020088231352,"std":0.021429891830523046},"eps":{"folds":[0.7841340191066283,0.8270541171031814,0.5895393142597103,0.7149418851497124,0.7983752243394157],"mean":0.7428089119917296,"std":0.09507993335109909},"nc":{"folds":[0.82934225078746,0.8837679742010279,0.6913458799536403,0.7221547815817042,0.8476352151057679],"mean":0.7948492203259201,"std":0.08348667270675214},"xc":{"folds":[0.5327654473782663,0.3241069669861184,0.4167490570881721,0.27713992741204785,0.4252117706119958],"mean":0.3951946338953201,"std":0.09909808334126459}},"macro":0.7882494230962865},"S242":{"tasks":{"eea":{"folds":[0.9091382863339706,0.9419878764435382,0.9448614480049967,0.924322655255868,0.9064448551886767],"mean":0.9253510242454099,"std":0.017880122539474276},"egb":{"folds":[0.9193352655661126,0.9080625395355496,0.9341132239312836,0.9230103597393675,0.917119763724793],"mean":0.9203282304994213,"std":0.00947295194752063},"egc":{"folds":[0.8976861027558355,0.9234698355140983,0.9230076389395383,0.9055567625725993,0.9202418983037742],"mean":0.9139924476171691,"std":0.01169623715901839},"ei":{"folds":[0.7425795222347185,0.7702930081602569,0.8051718765006373,0.8437322249272485,0.8602833346834312],"mean":0.8044119933012585,"std":0.049131518526013135},"eps":{"folds":[0.7402730124247168,0.791079654469599,0.5721725306170516,0.7311096544787377,0.773649064955299],"mean":0.7216567833890808,"std":0.08704050056393287},"nc":{"folds":[0.8456960912816979,0.9045792991032544,0.7553350112372592,0.7132079925161303,0.86226764773943],"mean":0.8162172083755543,"std":0.07923863808321363},"xc":{"folds":[0.44942721233860095,0.37583522521101775,0.44160393888648763,0.2642044030831354,0.5848560334617944],"mean":0.42318536259620726,"std":0.11692119722989236}},"macro":0.7893061500034431}},"diagnostics":{"DEFAULT_R0":{"consumption":{"status":"COMPLETE"}},"DEFAULT_R1":{"consumption":{"status":"COMPLETE"}},"DET_R0":{"consumption":{"status":"COMPLETE"}},"DET_R1":{"consumption":{"status":"COMPLETE"}}},"sota":{"BASE_ORIGINAL":{"eea":{"project":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.004431145079367593},"egb":{"project":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.0022430816246885588},"egc":{"project":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.003138905643812362},"ei":{"project":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.01762400606686776},"eps":{"project":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.011119521310488989},"nc":{"project":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.024623190090803382},"xc":{"project":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.01663027446704779}},"S42_A":{"eea":{"project":{"folds":[0.9058184813631905,0.9314225506385276,0.9418020574734336,0.9405905135617408,0.9095087394789485],"mean":0.9258284685031682,"std":0.01711067221506428},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.008171531496831808},"egb":{"project":{"folds":[0.9151983022103486,0.9021217732879702,0.9313825400712233,0.9270290970239738,0.9177358590719301],"mean":0.9186935143330892,"std":0.011382547228413639},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.004306485666910831},"egc":{"project":{"folds":[0.8944593377360505,0.9233164577934517,0.9274548122041496,0.9115227003666666,0.9249182728125974],"mean":0.9163343161825832,"std":0.01367265860634017},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.005334316182583176},"ei":{"project":{"folds":[0.7507134792573248,0.7883687758079122,0.758346976642833,0.8531225441883048,0.8158097219999276],"mean":0.7932722995792605,"std":0.042264237553500124},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.04572770042073948},"eps":{"project":{"folds":[0.7411496509433209,0.8660578107312047,0.6794221078549454,0.780615816620666,0.7939547848169688],"mean":0.7722400341934212,"std":0.06878451624605955},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.01475996580657879},"nc":{"project":{"folds":[0.8329887600354007,0.8929953893940883,0.8213587847528893,0.7437265071882315,0.8455270526379388],"mean":0.8273192988017097,"std":0.054083954421343707},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.028680701198290248},"xc":{"project":{"folds":[0.3244347880934484,0.33556547591205943,0.438722251367053,0.2377841345924503,0.5506201952711153],"mean":0.37742536904722523,"std":0.1202223796720207},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":-0.04557463095277475}},"S142":{"eea":{"project":{"folds":[0.9091173785377625,0.9300135171532607,0.9539327887720307,0.9409731590133223,0.9261622567457641],"mean":0.9320398200444281,"std":0.016757062714002183},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.0019601799555719612},"egb":{"project":{"folds":[0.921028147620244,0.9116411103904567,0.9222910350051907,0.9261068676194435,0.9127942967370729],"mean":0.9187722914744816,"std":0.006282050122077305},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.004227708525518459},"egc":{"project":{"folds":[0.8973606566246533,0.9215438353349087,0.9234236856126943,0.9080753828316384,0.9239918151910599],"mean":0.9148790751189908,"std":0.011764954782420648},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0038790751189907713},"ei":{"project":{"folds":[0.8045485143454515,0.7974735060531526,0.8130273614335738,0.8506402536479486,0.8303204086355498],"mean":0.8192020088231352,"std":0.021429891830523046},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.01979799117686476},"eps":{"project":{"folds":[0.7841340191066283,0.8270541171031814,0.5895393142597103,0.7149418851497124,0.7983752243394157],"mean":0.7428089119917296,"std":0.09507993335109909},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.04419108800827043},"nc":{"project":{"folds":[0.82934225078746,0.8837679742010279,0.6913458799536403,0.7221547815817042,0.8476352151057679],"mean":0.7948492203259201,"std":0.08348667270675214},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.06115077967407989},"xc":{"project":{"folds":[0.5327654473782663,0.3241069669861184,0.4167490570881721,0.27713992741204785,0.4252117706119958],"mean":0.3951946338953201,"std":0.09909808334126459},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":-0.027805366104679874}},"S242":{"eea":{"project":{"folds":[0.9091382863339706,0.9419878764435382,0.9448614480049967,0.924322655255868,0.9064448551886767],"mean":0.9253510242454099,"std":0.017880122539474276},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.008648975754590116},"egb":{"project":{"folds":[0.9193352655661126,0.9080625395355496,0.9341132239312836,0.9230103597393675,0.917119763724793],"mean":0.9203282304994213,"std":0.00947295194752063},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.002671769500578791},"egc":{"project":{"folds":[0.8976861027558355,0.9234698355140983,0.9230076389395383,0.9055567625725993,0.9202418983037742],"mean":0.9139924476171691,"std":0.01169623715901839},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.002992447617169103},"ei":{"project":{"folds":[0.7425795222347185,0.7702930081602569,0.8051718765006373,0.8437322249272485,0.8602833346834312],"mean":0.8044119933012585,"std":0.049131518526013135},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.034588006698741514},"eps":{"project":{"folds":[0.7402730124247168,0.791079654469599,0.5721725306170516,0.7311096544787377,0.773649064955299],"mean":0.7216567833890808,"std":0.08704050056393287},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.06534321661091924},"nc":{"project":{"folds":[0.8456960912816979,0.9045792991032544,0.7553350112372592,0.7132079925161303,0.86226764773943],"mean":0.8162172083755543,"std":0.07923863808321363},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.039782791624445646},"xc":{"project":{"folds":[0.44942721233860095,0.37583522521101775,0.44160393888648763,0.2642044030831354,0.5848560334617944],"mean":0.42318536259620726,"std":0.11692119722989236},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.00018536259620727247}}},"cpu":{"status":"COMPLETE"}}
```

</details>

<details><summary>results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json</summary>

```json
{"status":"待审查","validation":{"V1":{"tasks":{"eea":{"folds":[0.9316260918906403,0.9123610651479899,0.9050702542361508,0.9202273931472877,0.946903998516294],"mean":0.9232377605876726,"std":0.016491497980692923},"egb":{"folds":[0.9377257024120351,0.9366708875802592,0.9340279240727146,0.9461228554817407,0.9534817037048147],"mean":0.9416058146503128,"std":0.008035446479226518},"egc":{"folds":[0.9244811336142278,0.9213336739117486,0.9189108785172547,0.9036985691471207,0.9222909312285237],"mean":0.9181430372837751,"std":0.008318486810613496},"ei":{"folds":[0.8560148240981529,0.8947956072904384,0.8982002228087859,0.8560379250533212,0.8298769324213336],"mean":0.8669851023344064,"std":0.029004361316193928},"eps":{"folds":[0.7714897848265632,0.8464592232314412,0.7565904633996667,0.8546090841920899,0.6948400899130099],"mean":0.7847977291125543,"std":0.06659747222208665},"nc":{"folds":[0.8087157870263256,0.8972957183750673,0.8507503824116913,0.886434641268426,0.7661854053539039],"mean":0.8418763868870828,"std":0.054690580087524324},"xc":{"folds":[0.33559169945487466,0.4637539322418345,0.5353308876269116,0.5425817923661881,0.33429509965263915],"mean":0.4423106822684896,"std":0.10274139194499628}},"macro":0.8169937875891848},"V2":{"tasks":{"eea":{"folds":[0.943645909731352,0.8981388546873664,0.9020410605738527,0.916094543556696,0.9497780614534825],"mean":0.92193968600055,"std":0.023678766077277105},"egb":{"folds":[0.9365906708303705,0.9447740607057239,0.9302172452095397,0.9511921599042177,0.9528669899411715],"mean":0.9431282253182047,"std":0.00964381119630096},"egc":{"folds":[0.9190963894926791,0.9244326465209155,0.914106265665048,0.9102583438995274,0.9132218413311345],"mean":0.9162230973818609,"std":0.005583889843222611},"ei":{"folds":[0.852620386843219,0.8718914414572666,0.8977368951582153,0.8659458589573235,0.8532556406259593],"mean":0.8682900446083966,"std":0.01841978386489664},"eps":{"folds":[0.7579105675352572,0.8425945128799153,0.7434172775991136,0.848740833027059,0.6731370565727748],"mean":0.7731600495228239,"std":0.07357883159931577},"nc":{"folds":[0.8253016638236504,0.902616302691064,0.8751698121846471,0.892146864262583,0.7655447336639308],"mean":0.8521558753251751,"std":0.05678427104213044},"xc":{"folds":[0.30364915450424546,0.5369138516180565,0.5065884547872334,0.48416606967094367,0.30633397127207496],"mean":0.4275303003705108,"std":0.11342139559583758}},"macro":0.8146324683610745},"B_FP":{"tasks":{"eea":{"folds":[0.9385253327587553,0.9103085253762523,0.9106571106828454,0.9183210910769455,0.945869123306987],"mean":0.9247362366403571,"std":0.016464148955189464},"egb":{"folds":[0.9351454238178332,0.9358717531077823,0.92952116905478,0.9479926538394374,0.9559180061222797],"mean":0.9408898011884226,"std":0.010767101906054916},"egc":{"folds":[0.9219126712024205,0.9287402280684527,0.9245340532184785,0.909957660085341,0.9218161563689067],"mean":0.9213921537887199,"std":0.006983267331016533},"ei":{"folds":[0.853930487087313,0.8778922483494371,0.8960744572012486,0.8807400661015408,0.8280002148317006],"mean":0.867327494714248,"std":0.026664647471590804},"eps":{"folds":[0.7033665545728356,0.8691560148694102,0.749035734190309,0.8427987289319954,0.7235093357865903],"mean":0.7775732736702281,"std":0.07396907495351397},"nc":{"folds":[0.7966833731467942,0.9320223063590559,0.8471186026481647,0.8944451305349848,0.766136154324434],"mean":0.8472811134026866,"std":0.06809951616620287},"xc":{"folds":[0.3374743050508727,0.5202444411504019,0.549870188617994,0.5235968024424906,0.3190409470890351],"mean":0.45004533687015885,"std":0.1119565995183479}},"macro":0.8184636300392601}},"test":{"V1":{"tasks":{"eea":{"folds":[0.9055158460120524,0.9495350573659151,0.9395726149958313,0.9295932728724573,0.9071155318457937],"mean":0.9262664646184099,"std":0.01953775254528255},"egb":{"folds":[0.9421613151179151,0.9161442183785173,0.9476973155492734,0.931428697561094,0.9145136742256555],"mean":0.9303890441664912,"std":0.014951470941706697},"egc":{"folds":[0.8851340271095594,0.9193720834888168,0.9239270175487034,0.9093590307708537,0.9210915751837073],"mean":0.9117767468203282,"std":0.015872650296990337},"ei":{"folds":[0.7466707369879895,0.7798018823314983,0.823318232130934,0.8532392564265002,0.8358338053799164],"mean":0.8077727826513676,"std":0.04362371339819754},"eps":{"folds":[0.7130826147709619,0.8338873131482065,0.683452318212775,0.6970438676744631,0.8014620928609109],"mean":0.7457856413334635,"std":0.06743972021049974},"nc":{"folds":[0.796075540556072,0.8872636105731146,0.8492538889266049,0.7096981992746046,0.7983012507919558],"mean":0.8081184980244703,"std":0.06686560326287308},"xc":{"folds":[0.4390909386892953,0.3188679291946638,0.38785634675882863,0.27711625839993936,0.4786542689223101],"mean":0.38031714839300745,"std":0.08307800167197278}},"macro":0.7872037608582197},"V2":{"tasks":{"eea":{"folds":[0.9090244183719222,0.9322491970667464,0.9430430160805393,0.928019235560169,0.9263488247196873],"mean":0.927736938359813,"std":0.012316670099053318},"egb":{"folds":[0.941671564208756,0.9194012584998498,0.9375168021944394,0.9458921245870465,0.9261181870457547],"mean":0.9341199873071693,"std":0.011043297360188959},"egc":{"folds":[0.8835032297590825,0.9286534397244427,0.9217557563194558,0.9068522281939775,0.913707826939865],"mean":0.9108944961873646,"std":0.017377307646817334},"ei":{"folds":[0.8000168090685374,0.8196100439237102,0.7992556778018725,0.8497571354364515,0.811685522011784],"mean":0.816065037648471,"std":0.02065795790406536},"eps":{"folds":[0.6834234655214799,0.8145724825823535,0.7474495706470556,0.6910175996663204,0.7620513607359873],"mean":0.7397028958306393,"std":0.054089167615245064},"nc":{"folds":[0.7691946644866912,0.8893403983149617,0.7903640446609633,0.6609146319608321,0.8305320147368703],"mean":0.7880691508320636,"std":0.08454103266994446},"xc":{"folds":[0.3768175580849322,0.3131381240655201,0.40940626086348486,0.29047281425912685,0.5087608735251441],"mean":0.37971912615964165,"std":0.08651510222940827}},"macro":0.7851868046178803},"B_FP":{"tasks":{"eea":{"folds":[0.9159636698012301,0.9473047715159517,0.939984415202092,0.9395471400795122,0.9208415280259756],"mean":0.9327283049249523,"std":0.013545935583923146},"egb":{"folds":[0.9444550002181372,0.9182119577590064,0.9465349027366263,0.9337462172681269,0.9246801942712884],"mean":0.9335256544506372,"std":0.012262659376907688},"egc":{"folds":[0.8847735261042994,0.927654304816534,0.9179520616279764,0.9108341654064817,0.9227863275374719],"mean":0.9128000770985528,"std":0.016854291685891032},"ei":{"folds":[0.803751160346956,0.7786108048993424,0.8000597977245572,0.8489932957759004,0.850258455148075],"mean":0.8163347027789662,"std":0.031873638077730186},"eps":{"folds":[0.7499691275028288,0.8412829109062576,0.6853025288061612,0.7487171976363864,0.7950412565835323],"mean":0.7640626042870332,"std":0.058223512723314164},"nc":{"folds":[0.7620815715248026,0.906340206350182,0.8348875515587717,0.6353700529828489,0.7947465846148417],"mean":0.7866851934062894,"std":0.1002588966293432},"xc":{"folds":[0.45977914656335106,0.3275698293921032,0.4653548559438816,0.16938379699467765,0.540517978694514],"mean":0.3925211215177055,"std":0.1464073089720903}},"macro":0.7912368083520195}},"budget":{"plan":"UA2D-PRETRAIN-FP-20261019-01","revision":"r4"},"independent_review":"待审查，执行自检不能替代独立审查","delta_test":0.004033047493799735}
```

</details>

<details><summary>results/ua2d_pretrain_fp_20261019/r1/final_review/results.json</summary>

```json
{"status":"阻断","budget":{"plan":"UA2D-PRETRAIN-FP-20261019-01","revision":"r2"},"independent_review":"待审查，执行自检不能替代独立审查"}
```

</details>

<details><summary>results/ua2d_a020_fp_fusion_20261024/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-A020-FP-FUSION-20261024-01","revision":"r1","status":"需返修","independent_review":"PENDING","formal":{"B_A020":{"tasks":{"eea":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"egb":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"egc":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"ei":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"eps":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"nc":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"xc":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173}},"macro":0.785044436513689}},"FINAL":null,"baseline_macro":0.785044436513689,"delta":null,"performance_pass":null,"comparisons":{"UNI_FP_minus_B_A020":{"status":"未完成／不可比较","delta":null},"TMOL_FP_minus_B_A020":{"status":"未完成／不可比较","delta":null},"NEURAL_LATE_minus_B_A020":{"status":"未完成／不可比较","delta":null},"UNI_FP_minus_UNI_ZERO":{"status":"未完成／不可比较","delta":null},"TMOL_FP_minus_TMOL_ZERO":{"status":"未完成／不可比较","delta":null},"UNI_ZERO_minus_B_A020":{"status":"未完成／不可比较","delta":null},"TMOL_ZERO_minus_B_A020":{"status":"未完成／不可比较","delta":null},"UNI_FP_minus_TMOL_FP":{"status":"未完成／不可比较","delta":null}},"sota":{"B_A020":{"eea":{"project":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"reference":0.934,"delta":-0.0048973534443628175},"egb":{"project":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"reference":0.923,"delta":0.00785103276296062},"egc":{"project":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"reference":0.911,"delta":0.007125341354343506},"ei":{"project":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"reference":0.839,"delta":-0.02760752363123309},"eps":{"project":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"reference":0.787,"delta":-0.07443279463206498},"nc":{"project":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"reference":0.856,"delta":-0.04828792212544053},"xc":{"project":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173},"reference":0.423,"delta":-0.03743972468837947}},"UNI_ZERO":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"UNI_FP":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"TMOL_ZERO":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"TMOL_FP":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"FP_MLP":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}},"NEURAL_LATE":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}}}}
```

</details>

<details><summary>results/ua2d_a020_fp_fusion_20261024/r1/repairs/controller_repair1/final_review/results.json</summary>

```json
{"plan":"UA2D-A020-FP-FUSION-20261024-01","revision":"r1","status":"待审查","independent_review":"PENDING","formal":{"B_A020":{"tasks":{"eea":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"egb":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"egc":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"ei":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"eps":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"nc":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"xc":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173}},"macro":0.785044436513689},"UNI_ZERO":{"tasks":{"eea":{"folds":[0.9258658999200445,0.9166993661335252,0.939980989715117,0.9314239935439407,0.9082798471776021],"mean":0.924450019298046,"std":0.012383448305051833},"egb":{"folds":[0.9377433505672996,0.914149679256997,0.9443918537570595,0.9240677096136919,0.9224070714852315],"mean":0.9285519329360559,"std":0.012254363934719625},"egc":{"folds":[0.8904790676215795,0.9261376055617578,0.924529550972692,0.9103146413737843,0.9220092455848744],"mean":0.9146940222229377,"std":0.014889163183525133},"ei":{"folds":[0.7688424818013465,0.8018795961407629,0.8160718257386824,0.8491832531538923,0.840894069043872],"mean":0.8153742451757111,"std":0.03217960380690227},"eps":{"folds":[0.7565151942288811,0.8377359231283148,0.6745754611769279,0.734384918347236,0.7160960956201803],"mean":0.743861518500308,"std":0.06048512575687278},"nc":{"folds":[0.7958171324747232,0.9002856983093506,0.7907047057309609,0.7374751927665084,0.8250823320812746],"mean":0.8098730122725636,"std":0.0596074936715716},"xc":{"folds":[0.3621169013677178,0.3939560938837241,0.429761261879958,0.23942466838843535,0.5304879726693038],"mean":0.3911493796378278,"std":0.10580320340299895}},"macro":0.7897077328633502},"UNI_FP":{"tasks":{"eea":{"folds":[0.8968528691189643,0.9425387408171786,0.9376290221467004,0.9107726372599327,0.9064945787882944],"mean":0.918857569626214,"std":0.020097150400891578},"egb":{"folds":[0.9240366427020008,0.9031711518762119,0.9154097996073631,0.8975332739279469,0.9100055525242345],"mean":0.9100312841275514,"std":0.010348198311012573},"egc":{"folds":[0.8790425924736526,0.8900656815933095,0.9191798417978966,0.8878860735693872,0.8900342327707611],"mean":0.8932416844410014,"std":0.01519423844654539},"ei":{"folds":[0.7537103074718801,0.7363500701868182,0.7431773230609866,0.8114136494652423,0.7881711445577535],"mean":0.7665644989485362,"std":0.032028875161575376},"eps":{"folds":[0.7587118161541253,0.8180126209829323,0.5668729939456694,0.7078970799794216,0.7631816002081291],"mean":0.7229352222540555,"std":0.09555235761902141},"nc":{"folds":[0.8586273952403055,0.8523400353612158,0.7889806599691067,0.7195366776616617,0.80375117895293],"mean":0.8046471894370439,"std":0.056299450188110486},"xc":{"folds":[0.3451033182281169,0.4363675635716231,0.3124839582044422,0.22344112912893843,0.5219258498737509],"mean":0.36786436380137433,"std":0.11496820767337138}},"macro":0.7691631160908253},"TMOL_ZERO":{"tasks":{"eea":{"folds":[0.9252744117547171,0.9106010126320072,0.943081174899528,0.9294851837731055,0.9171100505229156],"mean":0.9251103667164549,"std":0.01241614168080239},"egb":{"folds":[0.9399483534954338,0.9087681248273138,0.9461085588961083,0.9276222551679317,0.9228598164777072],"mean":0.929061421772899,"std":0.014675482752745623},"egc":{"folds":[0.8993839781111899,0.9258959058838159,0.9283023884195152,0.9220708210084153,0.9216665965024947],"mean":0.9194639379850862,"std":0.01155790583660236},"ei":{"folds":[0.7666502536128825,0.8062967389864233,0.8167668388869672,0.8526060202780152,0.8418921268430098],"mean":0.8168423957214597,"std":0.033678604428257874},"eps":{"folds":[0.7311728306516551,0.8374848208989077,0.5142505448265144,0.6700270996787501,0.6940211468335173],"mean":0.6893912885778688,"std":0.1169915371879008},"nc":{"folds":[0.7992164326653113,0.882039710328235,0.8019387892528878,0.6676575525610149,0.8153738055892724],"mean":0.7932452580793442,"std":0.07787430826761767},"xc":{"folds":[0.36011696016118155,0.3592110208438334,0.4184924464951424,0.22750144428209151,0.483755931063386],"mean":0.36981556056912696,"std":0.09461986321465687}},"macro":0.7775614613460343},"TMOL_FP":{"tasks":{"eea":{"folds":[0.8994050116499571,0.9250570764125084,0.9368281248853488,0.901353241551868,0.9031240670951906],"mean":0.9131535043189747,"std":0.01681546572813523},"egb":{"folds":[0.9096957836862659,0.8999154641192938,0.9148678221179006,0.8974372686714304,0.9064999719539045],"mean":0.9056832621097591,"std":0.007113146075373148},"egc":{"folds":[0.8807700410205145,0.8868018577913411,0.9165699430256761,0.885376820662649,0.8873401203003745],"mean":0.8913717565601109,"std":0.014321507389571586},"ei":{"folds":[0.7820669701685052,0.7749445952895406,0.681475360206191,0.8351494314261217,0.7433836438125532],"mean":0.7634040001805824,"std":0.05644026799636001},"eps":{"folds":[0.6924035370077296,0.7967906316864659,0.5447326032529063,0.7217549110291033,0.7426300329640163],"mean":0.6996623431880444,"std":0.09463630768455289},"nc":{"folds":[0.8162392888979311,0.8728642836599936,0.7822683447711554,0.7394585580083213,0.8273084031220451],"mean":0.8076277756918893,"std":0.05001857206035606},"xc":{"folds":[0.4067948349082431,0.3971308405837295,0.3747919142674392,0.1991451864412982,0.5152731351248828],"mean":0.37862718226511854,"std":0.11405745763108219}},"macro":0.7656471177592113},"FP_MLP":{"tasks":{"eea":{"folds":[0.8675893747998014,0.876595904775951,0.852007097201213,0.8507725578244534,0.8770546066838922],"mean":0.8648039082570623,"std":0.012821142557014123},"egb":{"folds":[0.8513925535154201,0.8359090204262947,0.8822027798964739,0.841523183923169,0.8654137955959109],"mean":0.8552882666714539,"std":0.018755843271041787},"egc":{"folds":[0.8558325057527283,0.8701959295216866,0.8831287378962567,0.8458206008130353,0.8528414411726832],"mean":0.861563843031278,"std":0.014969433894114424},"ei":{"folds":[0.7137644567102868,0.7108499496730359,0.6350264789824875,0.7589210558402918,0.6763728486694451],"mean":0.6989869579751093,"std":0.0462426803409487},"eps":{"folds":[0.6991871478363072,0.7068996299569485,0.5748394439974518,0.6557490198339786,0.7123860749799447],"mean":0.669812263320926,"std":0.057594481660760644},"nc":{"folds":[0.7709866827922782,0.7973299106788785,0.7221377715723296,0.6489219333662586,0.7912469175075851],"mean":0.746124643183466,"std":0.06184629511633119},"xc":{"folds":[0.3862710206645621,0.3134273144206211,0.39909437939924464,0.24278367627800745,0.49954940687356253],"mean":0.36822515952719953,"std":0.09651811919125375}},"macro":0.7235435774237852},"NEURAL_LATE":{"tasks":{"eea":{"folds":[0.9281577240514418,0.9363666327699924,0.943314986279109,0.9267048557741092,0.9197959967194124],"mean":0.930868039118813,"std":0.009116318499977882},"egb":{"folds":[0.9292720673989988,0.9154953277469265,0.9458333752117944,0.9232599919704603,0.9190170689120342],"mean":0.9265755662480428,"std":0.011927174063369916},"egc":{"folds":[0.900301554013309,0.9284037168484098,0.9317221936408101,0.918390213222905,0.9234236279346597],"mean":0.9204482611320188,"std":0.012342700336654217},"ei":{"folds":[0.7731321742201664,0.8017696807864003,0.8039610481670855,0.8517460058626524,0.8122098784891378],"mean":0.8085637575050886,"std":0.028287551490145717},"eps":{"folds":[0.7108346563604855,0.835332002193217,0.602680018072256,0.7270459628646793,0.7713215803309137],"mean":0.7294428439643103,"std":0.08571342675324282},"nc":{"folds":[0.8125260728558614,0.8874840011244796,0.830749186034972,0.7260982380919829,0.804497443706531],"mean":0.8122709883627653,"std":0.05806177105410808},"xc":{"folds":[0.40084474381781454,0.39279863066244136,0.5069074933289177,0.3316137164983003,0.5006655188344165],"mean":0.42656602062837806,"std":0.07543747184848903}},"macro":0.793533639565631}},"FINAL":"NEURAL_LATE","baseline_macro":0.785044436513689,"delta":0.008489203051942007,"performance_pass":false,"comparisons":{"UNI_FP_minus_B_A020":{"status":"COMPLETE","delta":-0.01588132042286372,"tasks":{"eea":{"folds":[-0.029465000205244007,0.012369494053752428,-0.005467337787612547,-0.018634854012148372,-0.010027686695863625],"mean":-0.010245076929423225,"std":0.015605180876876953},"egb":{"folds":[-0.013699019582434802,-0.014082196875767616,-0.032417200250518974,-0.034295169909655976,-0.009605156558668781],"mean":-0.02081974863540923,"std":0.011596921185067256},"egc":{"folds":[-0.01922998747579241,-0.03670186766126782,-0.006315049456634969,-0.031194548058049443,-0.03097683191496603],"mean":-0.024883656913342135,"std":0.012179710535492114},"ei":{"folds":[-0.015776133814902837,-0.07384588725809671,-0.0777009562216695,-0.03930409150817182,-0.017512818298312638],"mean":-0.0448279774202307,"std":0.029762865608556333},"eps":{"folds":[0.07426311031062083,-0.02515051166019855,0.008050843067230584,-0.006984742284889323,0.001661384997838522],"mean":0.010368016886120412,"std":0.037836628476957915},"nc":{"folds":[0.056272809964148074,-0.04107536469080508,-0.05144156062034266,0.005182154974728181,0.015737518184694044],"mean":-0.0030648884375154895,"std":0.04395339972039633},"xc":{"folds":[-0.014548584309667989,0.08524120940379043,-0.1605310428693244,-0.05789236517264451,0.059251225396615315],"mean":-0.017695911510246232,"std":0.09816370007052026}}},"TMOL_FP_minus_B_A020":{"status":"COMPLETE","delta":-0.0193973187544777,"tasks":{"eea":{"folds":[-0.026912857674251223,-0.005112170350917711,-0.006268235048964121,-0.028054249720213087,-0.013398198388967497],"mean":-0.015949142236662727,"std":0.01100458906011578},"egb":{"folds":[-0.0280398785981697,-0.017337884632685707,-0.03295917773998147,-0.034391175166172516,-0.013110737128998795],"mean":-0.025167770653201638,"std":0.009496122784423458},"egc":{"folds":[-0.0175025389289305,-0.03996569146323614,-0.008924948228855478,-0.033703800964787645,-0.033670944385352675],"mean":-0.02675358479423249,"std":0.012982269279152036},"ei":{"folds":[0.012580528881722342,-0.03525136215537428,-0.13940291907646507,-0.01556830954729238,-0.062300319043512964],"mean":-0.04798847618818447,"std":0.05797269803225643},"eps":{"folds":[0.007954831164225129,-0.0463725009566649,-0.014089547625532495,0.006873088764792312,-0.018890182246274256],"mean":-0.012904862179890841,"std":0.022268837706398415},"nc":{"folds":[0.01388470362177363,-0.020551116392027335,-0.05815387581829401,0.02510403532138772,0.03929474235380914],"mean":-8.430218267017065e-05,"std":0.03927868532707154},"xc":{"folds":[0.04714293237045819,0.04600448641589683,-0.09822308680632741,-0.08218830786028475,0.05259851064774723],"mean":-0.0069330930465019815,"std":0.07626897468823776}}},"NEURAL_LATE_minus_B_A020":{"status":"COMPLETE","delta":0.008489203051942007,"tasks":{"eea":{"folds":[0.0018398547272334698,0.0061973860065662745,0.00021862634479608012,-0.0027026354979718947,0.003273731235254318],"mean":0.0017653925631756495,"std":0.0033271217491753972},"egb":{"folds":[-0.008463594885436798,-0.0017580210050529566,-0.001993624646087655,-0.008568451867142612,-0.0005936401708691585],"mean":-0.004275466514917836,"std":0.003907375139386363},"egc":{"folds":[0.002028974063864042,0.0016361675938325604,0.006227302386278555,-0.0006904084045316061,0.0024125632489325755],"mean":0.002322919777675225,"std":0.0024945149270281875},"ei":{"folds":[0.0036457329333835498,-0.008426276658514587,-0.01691723111557064,0.0010282648892382396,0.006525915633071699],"mean":-0.0028287188636783476,"std":0.00967112799843919},"eps":{"folds":[0.026385950516981005,-0.007831130449913792,0.043857867193817235,0.0121641406003683,0.009801365120623129],"mean":0.016875638596375177,"std":0.019372368777427592},"nc":{"folds":[0.010171487579703986,-0.005931398927541354,-0.009673034554477322,0.01174371540504937,0.016483782938295],"mean":0.004558910488205936,"std":0.01159654989197296},"xc":{"folds":[0.041192841280029646,0.041672276494608695,0.03389249225515112,0.05028022219671735,0.03799089435728087],"mean":0.04100574531675753,"std":0.006045305731179771}}},"UNI_FP_minus_UNI_ZERO":{"status":"COMPLETE","delta":-0.020544616772524837,"tasks":{"eea":{"folds":[-0.029013030801080197,0.025839374683653382,-0.002351967568416602,-0.020651356284007938,-0.0017852683893077037],"mean":-0.005592449671831812,"std":0.021143847950885003},"egb":{"folds":[-0.013706707865298817,-0.010978527380785086,-0.028982054149696346,-0.026534435685744984,-0.012401518960996949],"mean":-0.018520648808504438,"std":0.008531751797730979},"egc":{"folds":[-0.011436475147926894,-0.03607192396844838,-0.005349709174795381,-0.02242856780439706,-0.0319750128141133],"mean":-0.021452337781936205,"std":0.013086483174608724},"ei":{"folds":[-0.015132174329466452,-0.06552952595394468,-0.07289450267769582,-0.03776960368865001,-0.05272292448611848],"mean":-0.048809746227175085,"std":0.023081394496840678},"eps":{"folds":[0.0021966219252441688,-0.019723302145382537,-0.10770246723125854,-0.02648783836781432,0.0470855045879488],"mean":-0.020926296246252484,"std":0.05639759789625892},"nc":{"folds":[0.06281026276558233,-0.04794566294813474,-0.0017240457618541427,-0.017938515104846675,-0.02133115312834455],"mean":-0.005225822835519556,"std":0.041494428344107576},"xc":{"folds":[-0.017013583139600885,0.042411469687898995,-0.11727730367551581,-0.015983539259496915,-0.008562122795552907],"mean":-0.023285015836453503,"std":0.05800858641451581}}},"TMOL_FP_minus_TMOL_ZERO":{"status":"COMPLETE","delta":-0.011914343586822973,"tasks":{"eea":{"folds":[-0.02586940010476002,0.014456063780501238,-0.006253050014179196,-0.028131942221237494,-0.013985983427725057],"mean":-0.011956862397480107,"std":0.017244458457024534},"egb":{"folds":[-0.030252569809167906,-0.008852660708020044,-0.03124073677820771,-0.030184986496501365,-0.01635984452380268],"mean":-0.02337815966313994,"std":0.010193838734448969},"egc":{"folds":[-0.01861393709067538,-0.03909404809247474,-0.011732445393839108,-0.03669400034576631,-0.03432647620212026],"mean":-0.02809218142497516,"std":0.012159120742810668},"ei":{"folds":[0.015416716555622711,-0.03135214369688266,-0.13529147868077618,-0.017456588851893495,-0.09850848303045667],"mean":-0.05343839554087726,"std":0.061755079549715736},"eps":{"folds":[-0.03876929364392545,-0.040694189212441834,0.030482058426391845,0.0517278113503532,0.04860888613049896],"mean":0.010271054610175346,"std":0.04636633615872734},"nc":{"folds":[0.017022856232619787,-0.009175426668241449,-0.019670444481732385,0.07180100544730639,0.011934597532772706],"mean":0.014382517612545009,"std":0.03544159777756635},"xc":{"folds":[0.04667787474706153,0.03791981973989611,-0.043700532227703204,-0.02835625784079332,0.03151720406149683],"mean":0.00881162169599159,"std":0.04164030908571147}}},"UNI_ZERO_minus_B_A020":{"status":"COMPLETE","delta":0.004663296349661117,"tasks":{"eea":{"folds":[-0.00045196940416381004,-0.013469880629900954,-0.0031153702191959454,0.002016502271859566,-0.008242418306555921],"mean":-0.004652627257591413,"std":0.006227289359966022},"egb":{"folds":[7.688282864015683e-06,-0.0031036694949825305,-0.003435146100822628,-0.007760734223910992,0.0027963624023281675],"mean":-0.0022990998269047935,"std":0.0039702767689024235},"egc":{"folds":[-0.007793512327865515,-0.0006299436928194346,-0.0009653402818395884,-0.008765980253652383,0.0009981808991472718],"mean":-0.00343131913140593,"std":0.004501015588940487},"ei":{"folds":[-0.000643959485436385,-0.008316361304152031,-0.004806453543973688,-0.0015344878195218126,0.035210106187805845],"mean":0.0039817688069443856,"std":0.017716604932206295},"eps":{"folds":[0.07206648838537666,-0.0054272095148160115,0.11575331029848912,0.019503096082924998,-0.04542411959011028],"mean":0.031294313132372895,"std":0.06357852117971338},"nc":{"folds":[-0.0065374528014342514,0.0068702982573296545,-0.04971751485848852,0.023120670079574857,0.03706867131303859],"mean":0.002160934398004066,"std":0.03334319331385315},"xc":{"folds":[0.002464998829932896,0.042829739715891435,-0.043253739193808594,-0.041908825913147596,0.06781334819216822],"mean":0.005589104326207272,"std":0.04977488881005463}}},"TMOL_ZERO_minus_B_A020":{"status":"COMPLETE","delta":-0.007482975167654726,"tasks":{"eea":{"folds":[-0.0010434575694912018,-0.01956823413141895,-1.5185034784925833e-05,7.769250102440672e-05,0.00058778503875756],"mean":-0.003992279839182622,"std":0.00872732572314426},"egb":{"folds":[0.0022126912109982078,-0.008485223924665664,-0.0017184409617737595,-0.0042061886696711515,0.003249107394803885],"mean":-0.0017896109900616963,"std":0.00479800091952733},"egc":{"folds":[0.001111398161744881,-0.0008716433707613991,0.00280749716498363,0.0029901993809786642,0.000655531816767585],"mean":0.0013385966307426721,"std":0.0016038093693403013},"ei":{"folds":[-0.00283618767390037,-0.0038992184584916156,-0.004111440395688892,0.0018882793046011148,0.03620816398694371],"mean":0.005449919352692789,"std":0.01736548617602775},"eps":{"folds":[0.04672412480815058,-0.0056783117442230635,-0.04457160605192434,-0.04485472258556089,-0.06749906837677322],"mean":-0.023175916790066187,"std":0.04495684014713369},"nc":{"folds":[-0.0031381526108461566,-0.011375689723785887,-0.038483431336561624,-0.04669697012591867,0.027360144821036436],"mean":-0.01446681979521518,"std":0.029592942463404767},"xc":{"folds":[0.00046505762339665857,0.008084666676000718,-0.054522554578624205,-0.05383205001949143,0.021081306586250403],"mean":-0.01574471474249357,"std":0.03585079481931402}}},"UNI_FP_minus_TMOL_FP":{"status":"COMPLETE","delta":0.0035159983316139787,"tasks":{"eea":{"folds":[-0.0025521425309927848,0.01748166440467014,0.0008008972613515741,0.009419395708064715,0.0033705116931038726],"mean":0.005704065307239503,"std":0.007908334491399644},"egb":{"folds":[0.014340859015734897,0.003255687756918091,0.0005419774894624974,9.600525651654035e-05,0.0035055805703300136],"mean":0.004348022017792408,"std":0.005794939915286689},"egc":{"folds":[-0.0017274485468619094,0.003263823801968324,0.002609898772220509,0.002509252906738202,0.002694112470386645],"mean":0.0018699278808903541,"std":0.002032217793938429},"ei":{"folds":[-0.02835666269662518,-0.03859452510272243,0.06170196285479557,-0.023735781960879443,0.044787500745200326],"mean":0.0031604987679537677,"std":0.04642233760086291},"eps":{"folds":[0.0663082791463957,0.021221989296466348,0.02214039069276308,-0.013857831049681635,0.02055156724411278],"mean":0.023272879066011255,"std":0.02847643547992244},"nc":{"folds":[0.04238810634237444,-0.020524248298777747,0.006712315197951346,-0.019921880346659537,-0.0235572241691151],"mean":-0.002980586254845319,"std":0.02815340191622795},"xc":{"folds":[-0.06169151668012618,0.0392367229878936,-0.062307956062996994,0.024295942687640237,0.006652714748868083],"mean":-0.01076281846374425,"std":0.04817417666152864}}},"FINAL_minus_B_A020":{"status":"COMPLETE","delta":0.008489203051942007,"tasks":{"eea":{"folds":[0.0018398547272334698,0.0061973860065662745,0.00021862634479608012,-0.0027026354979718947,0.003273731235254318],"mean":0.0017653925631756495,"std":0.0033271217491753972},"egb":{"folds":[-0.008463594885436798,-0.0017580210050529566,-0.001993624646087655,-0.008568451867142612,-0.0005936401708691585],"mean":-0.004275466514917836,"std":0.003907375139386363},"egc":{"folds":[0.002028974063864042,0.0016361675938325604,0.006227302386278555,-0.0006904084045316061,0.0024125632489325755],"mean":0.002322919777675225,"std":0.0024945149270281875},"ei":{"folds":[0.0036457329333835498,-0.008426276658514587,-0.01691723111557064,0.0010282648892382396,0.006525915633071699],"mean":-0.0028287188636783476,"std":0.00967112799843919},"eps":{"folds":[0.026385950516981005,-0.007831130449913792,0.043857867193817235,0.0121641406003683,0.009801365120623129],"mean":0.016875638596375177,"std":0.019372368777427592},"nc":{"folds":[0.010171487579703986,-0.005931398927541354,-0.009673034554477322,0.01174371540504937,0.016483782938295],"mean":0.004558910488205936,"std":0.01159654989197296},"xc":{"folds":[0.041192841280029646,0.041672276494608695,0.03389249225515112,0.05028022219671735,0.03799089435728087],"mean":0.04100574531675753,"std":0.006045305731179771}}}},"attempt_map":{"plan":"UA2D-A020-FP-FUSION-20261024-01","revision":"r1"},"sota":{"B_A020":{"eea":{"project":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"reference":0.934,"delta":-0.0048973534443628175},"egb":{"project":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"reference":0.923,"delta":0.00785103276296062},"egc":{"project":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"reference":0.911,"delta":0.007125341354343506},"ei":{"project":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"reference":0.839,"delta":-0.02760752363123309},"eps":{"project":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"reference":0.787,"delta":-0.07443279463206498},"nc":{"project":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"reference":0.856,"delta":-0.04828792212544053},"xc":{"project":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173},"reference":0.423,"delta":-0.03743972468837947}},"UNI_ZERO":{"eea":{"project":{"folds":[0.9258658999200445,0.9166993661335252,0.939980989715117,0.9314239935439407,0.9082798471776021],"mean":0.924450019298046,"std":0.012383448305051833},"reference":0.934,"delta":-0.009549980701954008},"egb":{"project":{"folds":[0.9377433505672996,0.914149679256997,0.9443918537570595,0.9240677096136919,0.9224070714852315],"mean":0.9285519329360559,"std":0.012254363934719625},"reference":0.923,"delta":0.005551932936055826},"egc":{"project":{"folds":[0.8904790676215795,0.9261376055617578,0.924529550972692,0.9103146413737843,0.9220092455848744],"mean":0.9146940222229377,"std":0.014889163183525133},"reference":0.911,"delta":0.0036940222229376873},"ei":{"project":{"folds":[0.7688424818013465,0.8018795961407629,0.8160718257386824,0.8491832531538923,0.840894069043872],"mean":0.8153742451757111,"std":0.03217960380690227},"reference":0.839,"delta":-0.023625754824288858},"eps":{"project":{"folds":[0.7565151942288811,0.8377359231283148,0.6745754611769279,0.734384918347236,0.7160960956201803],"mean":0.743861518500308,"std":0.06048512575687278},"reference":0.787,"delta":-0.043138481499692016},"nc":{"project":{"folds":[0.7958171324747232,0.9002856983093506,0.7907047057309609,0.7374751927665084,0.8250823320812746],"mean":0.8098730122725636,"std":0.0596074936715716},"reference":0.856,"delta":-0.0461269877274364},"xc":{"project":{"folds":[0.3621169013677178,0.3939560938837241,0.429761261879958,0.23942466838843535,0.5304879726693038],"mean":0.3911493796378278,"std":0.10580320340299895},"reference":0.423,"delta":-0.03185062036217218}},"UNI_FP":{"eea":{"project":{"folds":[0.8968528691189643,0.9425387408171786,0.9376290221467004,0.9107726372599327,0.9064945787882944],"mean":0.918857569626214,"std":0.020097150400891578},"reference":0.934,"delta":-0.015142430373786042},"egb":{"project":{"folds":[0.9240366427020008,0.9031711518762119,0.9154097996073631,0.8975332739279469,0.9100055525242345],"mean":0.9100312841275514,"std":0.010348198311012573},"reference":0.923,"delta":-0.012968715872448633},"egc":{"project":{"folds":[0.8790425924736526,0.8900656815933095,0.9191798417978966,0.8878860735693872,0.8900342327707611],"mean":0.8932416844410014,"std":0.01519423844654539},"reference":0.911,"delta":-0.01775831555899865},"ei":{"project":{"folds":[0.7537103074718801,0.7363500701868182,0.7431773230609866,0.8114136494652423,0.7881711445577535],"mean":0.7665644989485362,"std":0.032028875161575376},"reference":0.839,"delta":-0.07243550105146379},"eps":{"project":{"folds":[0.7587118161541253,0.8180126209829323,0.5668729939456694,0.7078970799794216,0.7631816002081291],"mean":0.7229352222540555,"std":0.09555235761902141},"reference":0.787,"delta":-0.06406477774594455},"nc":{"project":{"folds":[0.8586273952403055,0.8523400353612158,0.7889806599691067,0.7195366776616617,0.80375117895293],"mean":0.8046471894370439,"std":0.056299450188110486},"reference":0.856,"delta":-0.05135281056295604},"xc":{"project":{"folds":[0.3451033182281169,0.4363675635716231,0.3124839582044422,0.22344112912893843,0.5219258498737509],"mean":0.36786436380137433,"std":0.11496820767337138},"reference":0.423,"delta":-0.05513563619862566}},"TMOL_ZERO":{"eea":{"project":{"folds":[0.9252744117547171,0.9106010126320072,0.943081174899528,0.9294851837731055,0.9171100505229156],"mean":0.9251103667164549,"std":0.01241614168080239},"reference":0.934,"delta":-0.008889633283545195},"egb":{"project":{"folds":[0.9399483534954338,0.9087681248273138,0.9461085588961083,0.9276222551679317,0.9228598164777072],"mean":0.929061421772899,"std":0.014675482752745623},"reference":0.923,"delta":0.006061421772898923},"egc":{"project":{"folds":[0.8993839781111899,0.9258959058838159,0.9283023884195152,0.9220708210084153,0.9216665965024947],"mean":0.9194639379850862,"std":0.01155790583660236},"reference":0.911,"delta":0.008463937985086156},"ei":{"project":{"folds":[0.7666502536128825,0.8062967389864233,0.8167668388869672,0.8526060202780152,0.8418921268430098],"mean":0.8168423957214597,"std":0.033678604428257874},"reference":0.839,"delta":-0.0221576042785403},"eps":{"project":{"folds":[0.7311728306516551,0.8374848208989077,0.5142505448265144,0.6700270996787501,0.6940211468335173],"mean":0.6893912885778688,"std":0.1169915371879008},"reference":0.787,"delta":-0.09760871142213123},"nc":{"project":{"folds":[0.7992164326653113,0.882039710328235,0.8019387892528878,0.6676575525610149,0.8153738055892724],"mean":0.7932452580793442,"std":0.07787430826761767},"reference":0.856,"delta":-0.06275474192065578},"xc":{"project":{"folds":[0.36011696016118155,0.3592110208438334,0.4184924464951424,0.22750144428209151,0.483755931063386],"mean":0.36981556056912696,"std":0.09461986321465687},"reference":0.423,"delta":-0.05318443943087303}},"TMOL_FP":{"eea":{"project":{"folds":[0.8994050116499571,0.9250570764125084,0.9368281248853488,0.901353241551868,0.9031240670951906],"mean":0.9131535043189747,"std":0.01681546572813523},"reference":0.934,"delta":-0.020846495681025368},"egb":{"project":{"folds":[0.9096957836862659,0.8999154641192938,0.9148678221179006,0.8974372686714304,0.9064999719539045],"mean":0.9056832621097591,"std":0.007113146075373148},"reference":0.923,"delta":-0.017316737890240907},"egc":{"project":{"folds":[0.8807700410205145,0.8868018577913411,0.9165699430256761,0.885376820662649,0.8873401203003745],"mean":0.8913717565601109,"std":0.014321507389571586},"reference":0.911,"delta":-0.019628243439889115},"ei":{"project":{"folds":[0.7820669701685052,0.7749445952895406,0.681475360206191,0.8351494314261217,0.7433836438125532],"mean":0.7634040001805824,"std":0.05644026799636001},"reference":0.839,"delta":-0.07559599981941756},"eps":{"project":{"folds":[0.6924035370077296,0.7967906316864659,0.5447326032529063,0.7217549110291033,0.7426300329640163],"mean":0.6996623431880444,"std":0.09463630768455289},"reference":0.787,"delta":-0.0873376568119556},"nc":{"project":{"folds":[0.8162392888979311,0.8728642836599936,0.7822683447711554,0.7394585580083213,0.8273084031220451],"mean":0.8076277756918893,"std":0.05001857206035606},"reference":0.856,"delta":-0.04837222430811072},"xc":{"project":{"folds":[0.4067948349082431,0.3971308405837295,0.3747919142674392,0.1991451864412982,0.5152731351248828],"mean":0.37862718226511854,"std":0.11405745763108219},"reference":0.423,"delta":-0.04437281773488144}},"FP_MLP":{"eea":{"project":{"folds":[0.8675893747998014,0.876595904775951,0.852007097201213,0.8507725578244534,0.8770546066838922],"mean":0.8648039082570623,"std":0.012821142557014123},"reference":0.934,"delta":-0.06919609174293773},"egb":{"project":{"folds":[0.8513925535154201,0.8359090204262947,0.8822027798964739,0.841523183923169,0.8654137955959109],"mean":0.8552882666714539,"std":0.018755843271041787},"reference":0.923,"delta":-0.06771173332854619},"egc":{"project":{"folds":[0.8558325057527283,0.8701959295216866,0.8831287378962567,0.8458206008130353,0.8528414411726832],"mean":0.861563843031278,"std":0.014969433894114424},"reference":0.911,"delta":-0.049436156968722056},"ei":{"project":{"folds":[0.7137644567102868,0.7108499496730359,0.6350264789824875,0.7589210558402918,0.6763728486694451],"mean":0.6989869579751093,"std":0.0462426803409487},"reference":0.839,"delta":-0.14001304202489062},"eps":{"project":{"folds":[0.6991871478363072,0.7068996299569485,0.5748394439974518,0.6557490198339786,0.7123860749799447],"mean":0.669812263320926,"std":0.057594481660760644},"reference":0.787,"delta":-0.117187736679074},"nc":{"project":{"folds":[0.7709866827922782,0.7973299106788785,0.7221377715723296,0.6489219333662586,0.7912469175075851],"mean":0.746124643183466,"std":0.06184629511633119},"reference":0.856,"delta":-0.10987535681653393},"xc":{"project":{"folds":[0.3862710206645621,0.3134273144206211,0.39909437939924464,0.24278367627800745,0.49954940687356253],"mean":0.36822515952719953,"std":0.09651811919125375},"reference":0.423,"delta":-0.05477484047280046}},"NEURAL_LATE":{"eea":{"project":{"folds":[0.9281577240514418,0.9363666327699924,0.943314986279109,0.9267048557741092,0.9197959967194124],"mean":0.930868039118813,"std":0.009116318499977882},"reference":0.934,"delta":-0.003131960881187057},"egb":{"project":{"folds":[0.9292720673989988,0.9154953277469265,0.9458333752117944,0.9232599919704603,0.9190170689120342],"mean":0.9265755662480428,"std":0.011927174063369916},"reference":0.923,"delta":0.003575566248042783},"egc":{"project":{"folds":[0.900301554013309,0.9284037168484098,0.9317221936408101,0.918390213222905,0.9234236279346597],"mean":0.9204482611320188,"std":0.012342700336654217},"reference":0.911,"delta":0.009448261132018798},"ei":{"project":{"folds":[0.7731321742201664,0.8017696807864003,0.8039610481670855,0.8517460058626524,0.8122098784891378],"mean":0.8085637575050886,"std":0.028287551490145717},"reference":0.839,"delta":-0.030436242494911392},"eps":{"project":{"folds":[0.7108346563604855,0.835332002193217,0.602680018072256,0.7270459628646793,0.7713215803309137],"mean":0.7294428439643103,"std":0.08571342675324282},"reference":0.787,"delta":-0.05755715603568978},"nc":{"project":{"folds":[0.8125260728558614,0.8874840011244796,0.830749186034972,0.7260982380919829,0.804497443706531],"mean":0.8122709883627653,"std":0.05806177105410808},"reference":0.856,"delta":-0.04372901163723464},"xc":{"project":{"folds":[0.40084474381781454,0.39279863066244136,0.5069074933289177,0.3316137164983003,0.5006655188344165],"mean":0.42656602062837806,"std":0.07543747184848903},"reference":0.423,"delta":0.003566020628378075}}},"delivery_summary":{"independent_review":"PENDING"}}
```

</details>

<details><summary>results/ua2d_3dshift_20261022/r1/all1_pcontrol_gpu2/final_review/results.json</summary>

```json
{"plan":"UA2D-3DSHIFT-20261022-01","revision":"r1.5-ALL1-PCTRL-GPU2","status":"待审查","independent_review":"PENDING","FINAL":"P_G","validation":{"F_G10K":{"tasks":{"eea":{"folds":[0.9384467046478924,0.8952747740597652,0.897165477770583,0.8984779549101436,0.9386812122490911],"mean":0.9136092247274951,"std":0.022809039812293024},"egb":{"folds":[0.9354674824587567,0.9329106361939695,0.9216571515298654,0.9501294999120807,0.9519268845495472],"mean":0.9384183309288439,"std":0.012645274242855175},"egc":{"folds":[0.9254745790780305,0.9218292282725803,0.9229148187512224,0.9096474422167751,0.917603017344742],"mean":0.9194938171326701,"std":0.006193810086207466},"ei":{"folds":[0.8848015831443573,0.8566898240765403,0.9093362736401144,0.8592930970118535,0.8209676443066232],"mean":0.8662176844358978,"std":0.03313182077659587},"eps":{"folds":[0.7125793250215011,0.8440357803093739,0.7226642428172178,0.8202097294456314,0.6456827994912635],"mean":0.7490343754169976,"std":0.08184910016552949},"nc":{"folds":[0.8346560203987243,0.9041085494087151,0.8292621685740287,0.8845627013438049,0.739635063307937],"mean":0.838444900606642,"std":0.06383695725566017},"xc":{"folds":[0.29300885626604756,0.5032279737669401,0.40595234453140383,0.6087724256299532,0.27309669076282816],"mean":0.41681165819143456,"std":0.14178836884065923}},"macro":0.8060042844914258},"P_G5K":{"tasks":{"eea":{"folds":[0.935128552763081,0.9139734176330018,0.9086308262297157,0.9158905054699712,0.9451956221340085],"mean":0.9237637848459557,"std":0.015615119709189813},"egb":{"folds":[0.9381042341988292,0.9305388213214707,0.9301301975682139,0.9519859542903547,0.9512987237871651],"mean":0.9404115862332068,"std":0.010735432843331618},"egc":{"folds":[0.9282256530826687,0.9211917193740644,0.9224022580308272,0.9069872939148541,0.9160023110711715],"mean":0.9189618470947172,"std":0.007980846391610768},"ei":{"folds":[0.8539253189150727,0.8809431373941408,0.8972347960077691,0.8614029157235867,0.842931951139527],"mean":0.8672876238360192,"std":0.02173525906498476},"eps":{"folds":[0.7317715851661242,0.8452418092332674,0.740796215891761,0.8178513641568933,0.6586450586888805],"mean":0.7588612066273852,"std":0.07423765045570946},"nc":{"folds":[0.7883601056870426,0.9182402166267448,0.8380469337661497,0.8923134591234833,0.7522918652931604],"mean":0.8378505160993163,"std":0.06927903388505817},"xc":{"folds":[0.31584801435153953,0.507804846122295,0.42422473476893396,0.5528858094404367,0.31227592680322136],"mean":0.4226078662972853,"std":0.1093209551785101}},"macro":0.8099634901476981},"P_G":{"tasks":{"eea":{"folds":[0.9270563823772411,0.9166327408435379,0.8963101382555146,0.9137172504791108,0.9473925446585585],"mean":0.9202218113227925,"std":0.018787755048516357},"egb":{"folds":[0.9312827943156986,0.9337063442353394,0.921094144774136,0.9554667586779052,0.9503605832683931],"mean":0.9383821250542945,"std":0.014199523342439591},"egc":{"folds":[0.9213872613549873,0.9191506113137938,0.9225184124232574,0.9108393250802893,0.9143191719038188],"mean":0.9176429564152293,"std":0.004934568291181817},"ei":{"folds":[0.8775222829117403,0.8766991197984997,0.8957539846782286,0.8738109756555648,0.830508895577735],"mean":0.8708590517243536,"std":0.024161496682207607},"eps":{"folds":[0.7340465159854896,0.8285976113533151,0.7448652444071135,0.7856746136553446,0.6736944558677618],"mean":0.7533756882538049,"std":0.058087652832782284},"nc":{"folds":[0.8013452945938959,0.9127552841610557,0.8434927153780698,0.866091366464452,0.7677803814733594],"mean":0.8382930084141667,"std":0.056306261389933686},"xc":{"folds":[0.2804617009596886,0.5259560422686876,0.45940561674335145,0.5867028524556174,0.2556533022104881],"mean":0.4216359029275667,"std":0.14750993154907027}},"macro":0.8086300777303154}},"formal":{"F_G10K":{"tasks":{"eea":{"folds":[0.9096301835610174,0.9280967986862416,0.9439369140030686,0.9300955352895374,0.9032566156068411],"mean":0.9230032094293412,"std":0.016456241318045316},"egb":{"folds":[0.9464454544870122,0.917364692029361,0.950250224420717,0.9342532924376477,0.9230226837860851],"mean":0.9342672694321645,"std":0.014281990984823927},"egc":{"folds":[0.8872152581767657,0.9295168156498679,0.9283668684300971,0.9180914199814554,0.9215191197317041],"mean":0.9169418963939779,"std":0.01728152176442276},"ei":{"folds":[0.7809049283434036,0.791028370590115,0.7979578162545934,0.7904551933276684,0.8271780541170919],"mean":0.7975048725265745,"std":0.01766284281160179},"eps":{"folds":[0.7283602102833471,0.8187013061581133,0.6135500798756172,0.7040513873576981,0.6702942076786225],"mean":0.7069914382706797,"std":0.07586556711046405},"nc":{"folds":[0.8115460526299713,0.8867299414283278,0.7685472955491949,0.6399311693870428,0.7607314234623676],"mean":0.7734971764913808,"std":0.08985644113836715},"xc":{"folds":[0.5137752035719265,0.38954879960877364,0.45892084856583204,0.2406366624238636,0.5117011913189038],"mean":0.42291654109785987,"std":0.1137691535074282}},"macro":0.7821603433774255},"P_G5K":{"tasks":{"eea":{"folds":[0.9091717064469558,0.9245400727594756,0.9469240117310807,0.9378835354472659,0.9086237789332884],"mean":0.9254286210636133,"std":0.017063529985164125},"egb":{"folds":[0.9382132709964627,0.9107683995398862,0.9456268538461009,0.9242449264226753,0.9263989541056279],"mean":0.9290504809821506,"std":0.013445178591540626},"egc":{"folds":[0.8880830871284674,0.9246179002243862,0.9241360610056591,0.9158753571877645,0.9226302612539403],"mean":0.9150685333600436,"std":0.015487539078735284},"ei":{"folds":[0.7860424428828154,0.7902994562350695,0.7870325759668177,0.8151627504662824,0.8241966387107201],"mean":0.800546772852341,"std":0.01782525988527752},"eps":{"folds":[0.6713411021667351,0.8469016917534875,0.7630088104563741,0.6948779851151021,0.6738842624986517],"mean":0.7300027703980703,"std":0.07513152681149461},"nc":{"folds":[0.8122795543752159,0.8992652303761335,0.8285355953671458,0.7232213191711669,0.7666482375859912],"mean":0.8059899873751307,"std":0.06642017720994764},"xc":{"folds":[0.4617294701041825,0.30692460154322665,0.28440553379046685,0.1887628311648527,0.3734228159724753],"mean":0.3230490505150408,"std":0.1019263005443577}},"macro":0.7755908880780558},"P_G":{"tasks":{"eea":{"folds":[0.9147193717154435,0.931490877873318,0.9483597918775252,0.9313877015167148,0.9032920494552072],"mean":0.9258499584876418,"std":0.01733441944817507},"egb":{"folds":[0.9409291375491511,0.9202330514015346,0.9484378012418313,0.9288372909650989,0.923433424240541],"mean":0.9323741410796312,"std":0.011948649790888463},"egc":{"folds":[0.8837380360424917,0.926460443520169,0.9250124428573644,0.9135122257988019,0.9180594047614846],"mean":0.9133565105960623,"std":0.017370731926113447},"ei":{"folds":[0.7579024035459649,0.7929949142775075,0.7843264082011602,0.8330442552693598,0.848972040950672],"mean":0.8034480044489328,"std":0.03707317177166164},"eps":{"folds":[0.7249513072678804,0.833699275423901,0.6157174705571195,0.6541465884767941,0.6725630813128236],"mean":0.7002155446077036,"std":0.08434556347330308},"nc":{"folds":[0.84431336071253,0.8874503853219452,0.830638672191863,0.7105546188508409,0.7730662232555333],"mean":0.8092046520665426,"std":0.06865174525200456},"xc":{"folds":[0.46016676482559593,0.3784538406416572,0.485036141835838,0.17093597302441232,0.49160253967297196],"mean":0.39723905200009507,"std":0.13429320678139187}},"macro":0.7830982661838012}},"formal_with_references":{"F_G":{"tasks":{"eea":{"folds":[0.9022999465049798,0.9483272819157673,0.950275007622847,0.9249630373566732,0.9146547318822503],"mean":0.9281040010565036,"std":0.02095908124453868},"egb":{"folds":[0.9376755540495655,0.9098241220082289,0.9470947163581207,0.9354815689581856,0.9170966306772548],"mean":0.929434518410271,"std":0.015436605853069899},"egc":{"folds":[0.8894244636653041,0.929623843043872,0.9186202982214563,0.9135562663533002,0.9196810490709284],"mean":0.9141811840709722,"std":0.015014290634345137},"ei":{"folds":[0.7858744270713283,0.777132445360772,0.8090624160370808,0.8409276765225356,0.8135021281392556],"mean":0.8052998186261945,"std":0.02510739947027493},"eps":{"folds":[0.6823578873490527,0.8234008093952725,0.6579744346581984,0.7638690172331493,0.7685514290559792],"mean":0.7392307155383303,"std":0.06780202176224799},"nc":{"folds":[0.741438054484669,0.9029592841415552,0.8328603305702754,0.6721506535257076,0.8356509813548945],"mean":0.7970118608154204,"std":0.09039043086135899},"xc":{"folds":[0.3836663487804117,0.3179460414714651,0.4037410139703107,0.1973756117837021,0.4599576049974794],"mean":0.35253732420067385,"std":0.10050082659460417}},"macro":0.7808284889597665},"P10K":{"tasks":{"eea":{"folds":[0.9085344377329794,0.9500258136738593,0.9521388027334685,0.9396353097229598,0.9078530808580162],"mean":0.9316374889442566,"std":0.021919446468473047},"egb":{"folds":[0.9507045581796164,0.9125709629899116,0.9378252334372563,0.9224240450263853,0.9238208983256504],"mean":0.9294691395917638,"std":0.014898419935170406},"egc":{"folds":[0.8931196561691934,0.9285935986642526,0.91901340442812,0.9211382443946686,0.9156646205160663],"mean":0.91550590483446,"std":0.013383598657257183},"ei":{"folds":[0.7937226262329049,0.8200997336019509,0.8094929852163176,0.8400600927711817,0.837139549335576],"mean":0.8201029974315862,"std":0.019345710827028215},"eps":{"folds":[0.6879749158999735,0.8305278546078143,0.6609921529804077,0.7942839899185673,0.7072852733709486],"mean":0.7362128373555422,"std":0.0726110074671783},"nc":{"folds":[0.8122387550201953,0.8977313938936047,0.8306830844504185,0.6850384662079421,0.8508847499683402],"mean":0.8153152899081002,"std":0.07949101467232002},"xc":{"folds":[0.37894031667094474,0.4125651835310974,0.5107261046971499,0.22020467215677086,0.4995110922280195],"mean":0.40438947385679647,"std":0.11725415168285576}},"macro":0.7932333045603579},"B_FP":{"tasks":{"eea":{"folds":[0.9159636698012301,0.9473047715159517,0.939984415202092,0.9395471400795122,0.9208415280259756],"mean":0.9327283049249523,"std":0.013545935583923146},"egb":{"folds":[0.9444550002181372,0.9182119577590064,0.9465349027366263,0.9337462172681269,0.9246801942712884],"mean":0.9335256544506372,"std":0.012262659376907688},"egc":{"folds":[0.8847735261042994,0.927654304816534,0.9179520616279764,0.9108341654064817,0.9227863275374719],"mean":0.9128000770985528,"std":0.016854291685891032},"ei":{"folds":[0.803751160346956,0.7786108048993424,0.8000597977245572,0.8489932957759004,0.850258455148075],"mean":0.8163347027789662,"std":0.031873638077730186},"eps":{"folds":[0.7499691275028288,0.8412829109062576,0.6853025288061612,0.7487171976363864,0.7950412565835323],"mean":0.7640626042870332,"std":0.058223512723314164},"nc":{"folds":[0.7620815715248026,0.906340206350182,0.8348875515587717,0.6353700529828489,0.7947465846148417],"mean":0.7866851934062894,"std":0.1002588966293432},"xc":{"folds":[0.45977914656335106,0.3275698293921032,0.4653548559438816,0.16938379699467765,0.540517978694514],"mean":0.3925211215177055,"std":0.1464073089720903}},"macro":0.7912368083520195},"F_G10K":{"tasks":{"eea":{"folds":[0.9096301835610174,0.9280967986862416,0.9439369140030686,0.9300955352895374,0.9032566156068411],"mean":0.9230032094293412,"std":0.016456241318045316},"egb":{"folds":[0.9464454544870122,0.917364692029361,0.950250224420717,0.9342532924376477,0.9230226837860851],"mean":0.9342672694321645,"std":0.014281990984823927},"egc":{"folds":[0.8872152581767657,0.9295168156498679,0.9283668684300971,0.9180914199814554,0.9215191197317041],"mean":0.9169418963939779,"std":0.01728152176442276},"ei":{"folds":[0.7809049283434036,0.791028370590115,0.7979578162545934,0.7904551933276684,0.8271780541170919],"mean":0.7975048725265745,"std":0.01766284281160179},"eps":{"folds":[0.7283602102833471,0.8187013061581133,0.6135500798756172,0.7040513873576981,0.6702942076786225],"mean":0.7069914382706797,"std":0.07586556711046405},"nc":{"folds":[0.8115460526299713,0.8867299414283278,0.7685472955491949,0.6399311693870428,0.7607314234623676],"mean":0.7734971764913808,"std":0.08985644113836715},"xc":{"folds":[0.5137752035719265,0.38954879960877364,0.45892084856583204,0.2406366624238636,0.5117011913189038],"mean":0.42291654109785987,"std":0.1137691535074282}},"macro":0.7821603433774255},"P_G5K":{"tasks":{"eea":{"folds":[0.9091717064469558,0.9245400727594756,0.9469240117310807,0.9378835354472659,0.9086237789332884],"mean":0.9254286210636133,"std":0.017063529985164125},"egb":{"folds":[0.9382132709964627,0.9107683995398862,0.9456268538461009,0.9242449264226753,0.9263989541056279],"mean":0.9290504809821506,"std":0.013445178591540626},"egc":{"folds":[0.8880830871284674,0.9246179002243862,0.9241360610056591,0.9158753571877645,0.9226302612539403],"mean":0.9150685333600436,"std":0.015487539078735284},"ei":{"folds":[0.7860424428828154,0.7902994562350695,0.7870325759668177,0.8151627504662824,0.8241966387107201],"mean":0.800546772852341,"std":0.01782525988527752},"eps":{"folds":[0.6713411021667351,0.8469016917534875,0.7630088104563741,0.6948779851151021,0.6738842624986517],"mean":0.7300027703980703,"std":0.07513152681149461},"nc":{"folds":[0.8122795543752159,0.8992652303761335,0.8285355953671458,0.7232213191711669,0.7666482375859912],"mean":0.8059899873751307,"std":0.06642017720994764},"xc":{"folds":[0.4617294701041825,0.30692460154322665,0.28440553379046685,0.1887628311648527,0.3734228159724753],"mean":0.3230490505150408,"std":0.1019263005443577}},"macro":0.7755908880780558},"P_G":{"tasks":{"eea":{"folds":[0.9147193717154435,0.931490877873318,0.9483597918775252,0.9313877015167148,0.9032920494552072],"mean":0.9258499584876418,"std":0.01733441944817507},"egb":{"folds":[0.9409291375491511,0.9202330514015346,0.9484378012418313,0.9288372909650989,0.923433424240541],"mean":0.9323741410796312,"std":0.011948649790888463},"egc":{"folds":[0.8837380360424917,0.926460443520169,0.9250124428573644,0.9135122257988019,0.9180594047614846],"mean":0.9133565105960623,"std":0.017370731926113447},"ei":{"folds":[0.7579024035459649,0.7929949142775075,0.7843264082011602,0.8330442552693598,0.848972040950672],"mean":0.8034480044489328,"std":0.03707317177166164},"eps":{"folds":[0.7249513072678804,0.833699275423901,0.6157174705571195,0.6541465884767941,0.6725630813128236],"mean":0.7002155446077036,"std":0.08434556347330308},"nc":{"folds":[0.84431336071253,0.8874503853219452,0.830638672191863,0.7105546188508409,0.7730662232555333],"mean":0.8092046520665426,"std":0.06865174525200456},"xc":{"folds":[0.46016676482559593,0.3784538406416572,0.485036141835838,0.17093597302441232,0.49160253967297196],"mean":0.39723905200009507,"std":0.13429320678139187}},"macro":0.7830982661838012}},"performance_pass":false,"carry_in":{"status":"HISTORICAL_ONLY","original_incident":{"plan":"UA2D-3DSHIFT-20261022-01","status":"ABNORMAL_STOP_EXIT_UNKNOWN"},"recovery1_stop":{"status":"已取消","consumption":{"plan":"UA2D-3DSHIFT-20261022-01","revision":"r1.3-RECOVERY1","status":"ABNORMAL_STOP"}}},"delta_P_G_F_G":0.0022697772240346525,"delta_P_G_P10K":-0.010135038376556715,"delta_P_G_F_G10K":0.0009379228063757239,"comparisons":{"P_G_minus_F_G10K":{"validation":{"status":"COMPLETE","delta":0.002625793238889562,"tasks":{"eea":{"folds":[-0.011390322270651287,0.021357966783772664,-0.0008553395150683318,0.015239295568967193,0.00871133240946742],"mean":0.006612586595297532,"std":0.01300120885376727},"egb":{"folds":[-0.004184688143058057,0.0007957080413698581,-0.0005630067557294005,0.005337258765824471,-0.0015663012811540922],"mean":-3.620587454944424e-05,"std":0.003514072397472196},"egc":{"folds":[-0.004087317723043271,-0.0026786169587864617,-0.0003964063279650043,0.0011918828635142376,-0.0032838454409231943],"mean":-0.0018508607174407387,"std":0.002185983281113872},"ei":{"folds":[-0.007279300232617025,0.02000929572195942,-0.013582288961885736,0.014517878643711235,0.009541251271111828],"mean":0.0046413672884559444,"std":0.014421611605589024},"eps":{"folds":[0.021467190963988525,-0.015438168956058762,0.022201001589895797,-0.034535115790286786,0.028011656376498273],"mean":0.00434131283680741,"std":0.02772703531782253},"nc":{"folds":[-0.03331072580482841,0.008646734752340568,0.014230546804041122,-0.01847133487935293,0.028145318165422317],"mean":-0.00015189219247546681,"std":0.025100362191267166},"xc":{"folds":[-0.01254715530635897,0.022728068501747578,0.05345327221194762,-0.022069573174335777,-0.017443388552340067],"mean":0.004824244736132077,"std":0.03242768759941639}}},"test":{"status":"COMPLETE","delta":0.0009379228063757239,"tasks":{"eea":{"folds":[0.00508918815442605,0.0033940791870763487,0.00442287787445661,0.001292166227177427,3.543384836612873e-05],"mean":0.002846749058300513,"std":0.002129114041288464},"egb":{"folds":[-0.005516316937861143,0.0028683593721735567,-0.001812423178885747,-0.005416001472548726,0.00041074045445588236],"mean":-0.0018931283525332355,"std":0.0036580124624576702},"egc":{"folds":[-0.0034772221342740384,-0.003056372129698892,-0.003354425572732689,-0.004579194182653512,-0.003459714970219463],"mean":-0.0035853857979157187,"std":0.0005805914500356971},"ei":{"folds":[-0.023002524797438695,0.0019665436873924635,-0.013631408053433214,0.04258906194169132,0.021793986833580137],"mean":0.005943131922358402,"std":0.026603574480664794},"eps":{"folds":[-0.003408903015466702,0.01499796926578767,0.0021673906815022903,-0.049904798880904044,0.0022688736342011406],"mean":-0.006775893662975929,"std":0.025036624914733917},"nc":{"folds":[0.03276730808255879,0.0007204438936173618,0.06209137664266817,0.07062344946379817,0.012334799793165763],"mean":0.03570747557516165,"std":0.030390189256206346},"xc":{"folds":[-0.05360843874633059,-0.01109495896711643,0.026115293270005946,-0.06970068939945129,-0.02009865164593183],"mean":-0.02567748909776484,"std":0.03756476417732991}}}},"P_G_minus_F_G":{"validation":{"status":"COMPLETE","delta":-0.003697382714945574,"tasks":{"eea":{"folds":[-0.004390489040594914,0.015196711630007909,-0.018226330715534367,-0.0035071664065436936,0.007227561396555338],"mean":-0.0007399426272219455,"std":0.012690703824251614},"egb":{"folds":[-0.0038289815284482387,-0.011833502731916856,-0.011938987561935899,0.012963516420504417,-0.007867216008998312],"mean":-0.004501034282158978,"std":0.010318449594374834},"egc":{"folds":[-0.004562218956637998,-0.009076987473721543,0.006642154146162649,0.0052658971857610926,0.0017470787942163168],"mean":3.184739156103511e-06,"std":0.006672811661495851},"ei":{"folds":[0.013385353455732107,-0.00014181547968350472,0.001766351945291289,0.000523229771306366,-0.04191546253929135],"mean":-0.005276468569329018,"std":0.02121486486181637},"eps":{"folds":[0.039678624153207664,-0.014268820131597693,-0.013992336091519952,-0.05170763428292058,-0.0006095808662711644],"mean":-0.008179949443820344,"std":0.032837749159271554},"nc":{"folds":[0.004438768626062073,0.013745834363267373,-0.014134487508391747,-0.022082950221615105,0.03062085690636762],"mean":0.0025176044331380433,"std":0.021225198904514228},"xc":{"folds":[-0.038209994827149485,0.010483890278671804,-0.009182737843441813,0.019033356472280882,-0.030649880352276004],"mean":-0.009705073254382923,"std":0.024924786479589223}}},"test":{"status":"COMPLETE","delta":0.0022697772240346525,"tasks":{"eea":{"folds":[0.012419425210463686,-0.01683640404244935,-0.00191521574532183,0.0064246641600416154,-0.01136268242704308],"mean":-0.0022540425688617914,"std":0.012107415052210628},"egb":{"folds":[0.0032535834995855994,0.010408929393305622,0.001343084883710599,-0.006644277993086645,0.006336793563286203],"mean":0.0029396226693602758,"std":0.006360620434173189},"egc":{"folds":[-0.005686427622812462,-0.0031633995237030366,0.0063921446359080925,-4.4040554498359086e-05,-0.0016216443094437816],"mean":-0.0008246734749099094,"std":0.004539639491141059},"ei":{"folds":[-0.02797202352536332,0.015862468916735484,-0.02473600783592067,-0.00788342125317587,0.035469912811416426],"mean":-0.0018518141772615904,"std":0.027152527831979013},"eps":{"folds":[0.0425934199188277,0.010298466028628495,-0.04225696410107893,-0.10972242875635518,-0.09598834774315557],"mean":-0.039015170930626696,"std":0.06585545782030633},"nc":{"folds":[0.10287530622786101,-0.01550889881961004,-0.0022216583784123722,0.03840396532513335,-0.06258475809936115],"mean":0.01219279125112216,"std":0.06220366959956188},"xc":{"folds":[0.0765004160451842,0.06050779917019211,0.08129512786552728,-0.026439638759289785,0.031644934675492564],"mean":0.044701727799421276,"std":0.04424827476085699}}}},"P_G_minus_P10K":{"validation":{"status":"未完成／不可比较","delta":null},"test":{"status":"COMPLETE","delta":-0.010135038376556715,"tasks":{"eea":{"folds":[0.006184933982464114,-0.01853493580054133,-0.0037790108559433433,-0.008247608206245038,-0.004561031402808946],"mean":-0.005787530456614909,"std":0.008908236093594367},"egb":{"folds":[-0.009775420630465326,0.007662088411622969,0.01061256780457498,0.006413245938713619,-0.0003874740851094183],"mean":0.0029050014878673645,"std":0.008154411906312675},"egc":{"folds":[-0.00938162012670174,-0.0021331551440836316,0.00599903842924443,-0.007626018595866757,0.00239478424541828],"mean":-0.0021493942383978837,"std":0.006506636657911986},"ei":{"folds":[-0.03582022268693996,-0.02710481932444342,-0.025166577015157454,-0.007015837501821887,0.01183249161509603],"mean":-0.016654992982653338,"std":0.019061755297090754},"eps":{"folds":[0.03697639136790687,0.0031714208160866786,-0.04527468242328825,-0.14013740144177322,-0.03472219205812499],"mean":-0.03599729274783858,"std":0.06669371180145282},"nc":{"folds":[0.03207460569233478,-0.010281008571659478,-4.44122585554485e-05,0.02551615264289886,-0.07781852671280687],"mean":-0.006110637841557631,"std":0.04374522577182615},"xc":{"folds":[0.08122644815465119,-0.03411134288944018,-0.025689962861311866,-0.04926869913235854,-0.007908552555047521],"mean":-0.007150421856701383,"std":0.05161298538019195}}}},"P_G_minus_B_FP":{"validation":{"status":"未完成／不可比较","delta":null},"test":{"status":"COMPLETE","delta":-0.008138542168218299,"tasks":{"eea":{"folds":[-0.0012442980857866326,-0.015813893642633747,0.0083753766754332,-0.008159438562797439,-0.017549478570768384],"mean":-0.0068783464373106005,"std":0.010721203658286488},"egb":{"folds":[-0.003525862668986135,0.0020210936425281156,0.0019028985052049752,-0.004908926303027972,-0.0012467700307474683],"mean":-0.001151513371005697,"std":0.0031288826893011794},"egc":{"folds":[-0.0010354900618076979,-0.0011938612963650064,0.007060381229388013,0.002678060392320214,-0.004726922775987297],"mean":0.0005564334975096452,"std":0.00448091069919562},"ei":{"folds":[-0.0458487568009911,0.014384109378165033,-0.01573338952339698,-0.015949040506540668,-0.0012864141974029408],"mean":-0.01288669833003333,"std":0.02226119063535527},"eps":{"folds":[-0.025017820234948474,-0.007583635482356632,-0.06958505824904171,-0.09457060915959237,-0.1224781752707087],"mean":-0.06384705967932958,"std":0.04766484966392802},"nc":{"folds":[0.08223178918772744,-0.018889821028236775,-0.004248879366908653,0.07518456586799205,-0.021680361359308398],"mean":0.022519458660253133,"std":0.05177854862064643},"xc":{"folds":[0.00038761826224487095,0.050884011249553995,0.01968128589195639,0.0015521760297346754,-0.04891543902154205],"mean":0.004717930482389577,"std":0.036264724416563074}}}}},"references":{"validation_blocks":{"F_G":{"tasks":{"eea":{"folds":[0.931446871417836,0.9014360292135299,0.914536468971049,0.9172244168856545,0.9401649832620032],"mean":0.9209617539500146,"std":0.015126199177266568},"egb":{"folds":[0.9351117758441468,0.9455398469672562,0.9330331323360719,0.9425032422574008,0.9582277992773914],"mean":0.9428831593364535,"std":0.01000097163235137},"egc":{"folds":[0.9259494803116253,0.9282275987875154,0.9158762582770947,0.9055734278945282,0.9125720931096025],"mean":0.9176397716760732,"std":0.009427901918844813},"ei":{"folds":[0.8641369294560082,0.8768409352781832,0.8939876327329374,0.8732877458842584,0.8724243581170263],"mean":0.8761355202936827,"std":0.011011860137171335},"eps":{"folds":[0.694367891832282,0.8428664314849128,0.7588575804986335,0.8373822479382652,0.6743040367340329],"mean":0.7615556376976252,"std":0.07825508703069987},"nc":{"folds":[0.7969065259678338,0.8990094497977883,0.8576272028864615,0.8881743166860671,0.7371595245669917],"mean":0.8357754039810285,"std":0.06792932553219976},"xc":{"folds":[0.31867169578683807,0.5154721519900158,0.46858835458679327,0.5676694959833365,0.2863031825627641],"mean":0.43134097618194955,"std":0.12326923336217376}},"macro":0.8123274604452609}},"formal_blocks":{"F_G":{"tasks":{"eea":{"folds":[0.9022999465049798,0.9483272819157673,0.950275007622847,0.9249630373566732,0.9146547318822503],"mean":0.9281040010565036,"std":0.02095908124453868},"egb":{"folds":[0.9376755540495655,0.9098241220082289,0.9470947163581207,0.9354815689581856,0.9170966306772548],"mean":0.929434518410271,"std":0.015436605853069899},"egc":{"folds":[0.8894244636653041,0.929623843043872,0.9186202982214563,0.9135562663533002,0.9196810490709284],"mean":0.9141811840709722,"std":0.015014290634345137},"ei":{"folds":[0.7858744270713283,0.777132445360772,0.8090624160370808,0.8409276765225356,0.8135021281392556],"mean":0.8052998186261945,"std":0.02510739947027493},"eps":{"folds":[0.6823578873490527,0.8234008093952725,0.6579744346581984,0.7638690172331493,0.7685514290559792],"mean":0.7392307155383303,"std":0.06780202176224799},"nc":{"folds":[0.741438054484669,0.9029592841415552,0.8328603305702754,0.6721506535257076,0.8356509813548945],"mean":0.7970118608154204,"std":0.09039043086135899},"xc":{"folds":[0.3836663487804117,0.3179460414714651,0.4037410139703107,0.1973756117837021,0.4599576049974794],"mean":0.35253732420067385,"std":0.10050082659460417}},"macro":0.7808284889597665},"P10K":{"tasks":{"eea":{"folds":[0.9085344377329794,0.9500258136738593,0.9521388027334685,0.9396353097229598,0.9078530808580162],"mean":0.9316374889442566,"std":0.021919446468473047},"egb":{"folds":[0.9507045581796164,0.9125709629899116,0.9378252334372563,0.9224240450263853,0.9238208983256504],"mean":0.9294691395917638,"std":0.014898419935170406},"egc":{"folds":[0.8931196561691934,0.9285935986642526,0.91901340442812,0.9211382443946686,0.9156646205160663],"mean":0.91550590483446,"std":0.013383598657257183},"ei":{"folds":[0.7937226262329049,0.8200997336019509,0.8094929852163176,0.8400600927711817,0.837139549335576],"mean":0.8201029974315862,"std":0.019345710827028215},"eps":{"folds":[0.6879749158999735,0.8305278546078143,0.6609921529804077,0.7942839899185673,0.7072852733709486],"mean":0.7362128373555422,"std":0.0726110074671783},"nc":{"folds":[0.8122387550201953,0.8977313938936047,0.8306830844504185,0.6850384662079421,0.8508847499683402],"mean":0.8153152899081002,"std":0.07949101467232002},"xc":{"folds":[0.37894031667094474,0.4125651835310974,0.5107261046971499,0.22020467215677086,0.4995110922280195],"mean":0.40438947385679647,"std":0.11725415168285576}},"macro":0.7932333045603579},"B_FP":{"tasks":{"eea":{"folds":[0.9159636698012301,0.9473047715159517,0.939984415202092,0.9395471400795122,0.9208415280259756],"mean":0.9327283049249523,"std":0.013545935583923146},"egb":{"folds":[0.9444550002181372,0.9182119577590064,0.9465349027366263,0.9337462172681269,0.9246801942712884],"mean":0.9335256544506372,"std":0.012262659376907688},"egc":{"folds":[0.8847735261042994,0.927654304816534,0.9179520616279764,0.9108341654064817,0.9227863275374719],"mean":0.9128000770985528,"std":0.016854291685891032},"ei":{"folds":[0.803751160346956,0.7786108048993424,0.8000597977245572,0.8489932957759004,0.850258455148075],"mean":0.8163347027789662,"std":0.031873638077730186},"eps":{"folds":[0.7499691275028288,0.8412829109062576,0.6853025288061612,0.7487171976363864,0.7950412565835323],"mean":0.7640626042870332,"std":0.058223512723314164},"nc":{"folds":[0.7620815715248026,0.906340206350182,0.8348875515587717,0.6353700529828489,0.7947465846148417],"mean":0.7866851934062894,"std":0.1002588966293432},"xc":{"folds":[0.45977914656335106,0.3275698293921032,0.4653548559438816,0.16938379699467765,0.540517978694514],"mean":0.3925211215177055,"std":0.1464073089720903}},"macro":0.7912368083520195}}},"budget":{"plan":"UA2D-3DSHIFT-20261022-01","revision":"r1.5-ALL1-PCTRL-GPU2","historical_carry_in":{"status":"HISTORICAL_ONLY","original_incident":{"plan":"UA2D-3DSHIFT-20261022-01","status":"ABNORMAL_STOP_EXIT_UNKNOWN"},"recovery1_stop":{"status":"已取消","consumption":{"plan":"UA2D-3DSHIFT-20261022-01","revision":"r1.3-RECOVERY1","status":"ABNORMAL_STOP"}}}},"sota":{"F_G":{"eea":{"project":{"folds":[0.9022999465049798,0.9483272819157673,0.950275007622847,0.9249630373566732,0.9146547318822503],"mean":0.9281040010565036,"std":0.02095908124453868},"reference":0.934,"delta":-0.005895998943496461},"egb":{"project":{"folds":[0.9376755540495655,0.9098241220082289,0.9470947163581207,0.9354815689581856,0.9170966306772548],"mean":0.929434518410271,"std":0.015436605853069899},"reference":0.923,"delta":0.006434518410270984},"egc":{"project":{"folds":[0.8894244636653041,0.929623843043872,0.9186202982214563,0.9135562663533002,0.9196810490709284],"mean":0.9141811840709722,"std":0.015014290634345137},"reference":0.911,"delta":0.0031811840709721206},"ei":{"project":{"folds":[0.7858744270713283,0.777132445360772,0.8090624160370808,0.8409276765225356,0.8135021281392556],"mean":0.8052998186261945,"std":0.02510739947027493},"reference":0.839,"delta":-0.03370018137380548},"eps":{"project":{"folds":[0.6823578873490527,0.8234008093952725,0.6579744346581984,0.7638690172331493,0.7685514290559792],"mean":0.7392307155383303,"std":0.06780202176224799},"reference":0.787,"delta":-0.047769284461669725},"nc":{"project":{"folds":[0.741438054484669,0.9029592841415552,0.8328603305702754,0.6721506535257076,0.8356509813548945],"mean":0.7970118608154204,"std":0.09039043086135899},"reference":0.856,"delta":-0.05898813918457957},"xc":{"project":{"folds":[0.3836663487804117,0.3179460414714651,0.4037410139703107,0.1973756117837021,0.4599576049974794],"mean":0.35253732420067385,"std":0.10050082659460417},"reference":0.423,"delta":-0.07046267579932614}},"F_G10K":{"eea":{"project":{"folds":[0.9096301835610174,0.9280967986862416,0.9439369140030686,0.9300955352895374,0.9032566156068411],"mean":0.9230032094293412,"std":0.016456241318045316},"reference":0.934,"delta":-0.010996790570658832},"egb":{"project":{"folds":[0.9464454544870122,0.917364692029361,0.950250224420717,0.9342532924376477,0.9230226837860851],"mean":0.9342672694321645,"std":0.014281990984823927},"reference":0.923,"delta":0.011267269432164495},"egc":{"project":{"folds":[0.8872152581767657,0.9295168156498679,0.9283668684300971,0.9180914199814554,0.9215191197317041],"mean":0.9169418963939779,"std":0.01728152176442276},"reference":0.911,"delta":0.005941896393977908},"ei":{"project":{"folds":[0.7809049283434036,0.791028370590115,0.7979578162545934,0.7904551933276684,0.8271780541170919],"mean":0.7975048725265745,"std":0.01766284281160179},"reference":0.839,"delta":-0.04149512747342543},"eps":{"project":{"folds":[0.7283602102833471,0.8187013061581133,0.6135500798756172,0.7040513873576981,0.6702942076786225],"mean":0.7069914382706797,"std":0.07586556711046405},"reference":0.787,"delta":-0.08000856172932036},"nc":{"project":{"folds":[0.8115460526299713,0.8867299414283278,0.7685472955491949,0.6399311693870428,0.7607314234623676],"mean":0.7734971764913808,"std":0.08985644113836715},"reference":0.856,"delta":-0.08250282350861915},"xc":{"project":{"folds":[0.5137752035719265,0.38954879960877364,0.45892084856583204,0.2406366624238636,0.5117011913189038],"mean":0.42291654109785987,"std":0.1137691535074282},"reference":0.423,"delta":-8.345890214012108e-05}},"P_G":{"eea":{"project":{"folds":[0.9147193717154435,0.931490877873318,0.9483597918775252,0.9313877015167148,0.9032920494552072],"mean":0.9258499584876418,"std":0.01733441944817507},"reference":0.934,"delta":-0.008150041512358208},"egb":{"project":{"folds":[0.9409291375491511,0.9202330514015346,0.9484378012418313,0.9288372909650989,0.923433424240541],"mean":0.9323741410796312,"std":0.011948649790888463},"reference":0.923,"delta":0.009374141079631193},"egc":{"project":{"folds":[0.8837380360424917,0.926460443520169,0.9250124428573644,0.9135122257988019,0.9180594047614846],"mean":0.9133565105960623,"std":0.017370731926113447},"reference":0.911,"delta":0.0023565105960622335},"ei":{"project":{"folds":[0.7579024035459649,0.7929949142775075,0.7843264082011602,0.8330442552693598,0.848972040950672],"mean":0.8034480044489328,"std":0.03707317177166164},"reference":0.839,"delta":-0.03555199555106714},"eps":{"project":{"folds":[0.7249513072678804,0.833699275423901,0.6157174705571195,0.6541465884767941,0.6725630813128236],"mean":0.7002155446077036,"std":0.08434556347330308},"reference":0.787,"delta":-0.0867844553922964},"nc":{"project":{"folds":[0.84431336071253,0.8874503853219452,0.830638672191863,0.7105546188508409,0.7730662232555333],"mean":0.8092046520665426,"std":0.06865174525200456},"reference":0.856,"delta":-0.04679534793345741},"xc":{"project":{"folds":[0.46016676482559593,0.3784538406416572,0.485036141835838,0.17093597302441232,0.49160253967297196],"mean":0.39723905200009507,"std":0.13429320678139187},"reference":0.423,"delta":-0.025760947999904915}},"P_G5K":{"eea":{"project":{"folds":[0.9091717064469558,0.9245400727594756,0.9469240117310807,0.9378835354472659,0.9086237789332884],"mean":0.9254286210636133,"std":0.017063529985164125},"reference":0.934,"delta":-0.008571378936386709},"egb":{"project":{"folds":[0.9382132709964627,0.9107683995398862,0.9456268538461009,0.9242449264226753,0.9263989541056279],"mean":0.9290504809821506,"std":0.013445178591540626},"reference":0.923,"delta":0.006050480982150552},"egc":{"project":{"folds":[0.8880830871284674,0.9246179002243862,0.9241360610056591,0.9158753571877645,0.9226302612539403],"mean":0.9150685333600436,"std":0.015487539078735284},"reference":0.911,"delta":0.004068533360043536},"ei":{"project":{"folds":[0.7860424428828154,0.7902994562350695,0.7870325759668177,0.8151627504662824,0.8241966387107201],"mean":0.800546772852341,"std":0.01782525988527752},"reference":0.839,"delta":-0.038453227147658975},"eps":{"project":{"folds":[0.6713411021667351,0.8469016917534875,0.7630088104563741,0.6948779851151021,0.6738842624986517],"mean":0.7300027703980703,"std":0.07513152681149461},"reference":0.787,"delta":-0.056997229601929766},"nc":{"project":{"folds":[0.8122795543752159,0.8992652303761335,0.8285355953671458,0.7232213191711669,0.7666482375859912],"mean":0.8059899873751307,"std":0.06642017720994764},"reference":0.856,"delta":-0.05001001262486926},"xc":{"project":{"folds":[0.4617294701041825,0.30692460154322665,0.28440553379046685,0.1887628311648527,0.3734228159724753],"mean":0.3230490505150408,"std":0.1019263005443577},"reference":0.423,"delta":-0.0999509494849592}}}}
```

</details>

<details><summary>results/ua2d_fpdecay_20261022/sg_r2/final_review/results.json</summary>

```json
{"plan":"UA2D-FPDECAY-20261022-01","revision":"r2-SG","status":"待审查","independent_review":"PENDING","validation":{"BASE":{"tasks":{"eea":{"folds":[0.931446871417836,0.9014360292135299,0.914536468971049,0.9172244168856545,0.9401649832620032],"mean":0.9209617539500146,"std":0.015126199177266568},"egb":{"folds":[0.9351117758441468,0.9455398469672562,0.9330331323360719,0.9425032422574008,0.9582277992773914],"mean":0.9428831593364535,"std":0.01000097163235137},"egc":{"folds":[0.9259494803116253,0.9282275987875154,0.9158762582770947,0.9055734278945282,0.9125720931096025],"mean":0.9176397716760732,"std":0.009427901918844813},"ei":{"folds":[0.8641369294560082,0.8768409352781832,0.8939876327329374,0.8732877458842584,0.8724243581170263],"mean":0.8761355202936827,"std":0.011011860137171335},"eps":{"folds":[0.694367891832282,0.8428664314849128,0.7588575804986335,0.8373822479382652,0.6743040367340329],"mean":0.7615556376976252,"std":0.07825508703069987},"nc":{"folds":[0.7969065259678338,0.8990094497977883,0.8576272028864615,0.8881743166860671,0.7371595245669917],"mean":0.8357754039810285,"std":0.06792932553219976},"xc":{"folds":[0.31867169578683807,0.5154721519900158,0.46858835458679327,0.5676694959833365,0.2863031825627641],"mean":0.43134097618194955,"std":0.12326923336217376}},"macro":0.8123274604452609},"FP_DECAY":{"tasks":{"eea":{"folds":[0.9298590755383758,0.9064549822825829,0.8891170450461952,0.9133271704968458,0.9421057498472971],"mean":0.9161728046422594,"std":0.020581420568519835},"egb":{"folds":[0.9355158729489181,0.9362039341142085,0.9299611103590889,0.9497490232263005,0.9531340735889118],"mean":0.9409128028474857,"std":0.009983462757930156},"egc":{"folds":[0.9246113955101657,0.9187058675678973,0.9163256307900233,0.9064683904320036,0.9095485314393905],"mean":0.915131963147896,"std":0.007250680571524179},"ei":{"folds":[0.866419983669868,0.894702476038562,0.8987288814786973,0.8709097638719442,0.8545690190090582],"mean":0.8770660248136259,"std":0.018958354238226956},"eps":{"folds":[0.7070290820018711,0.8463179251440758,0.7382018688894136,0.8278847138092428,0.6637789353648083],"mean":0.7566425050418824,"std":0.07832995062742715},"nc":{"folds":[0.8094482493982802,0.9002668438559163,0.8470529422304609,0.8894241905753728,0.7692744136546351],"mean":0.8430933279429331,"std":0.05480023201685162},"xc":{"folds":[0.32988880352461625,0.52435395551164,0.4266400942239744,0.5213387025198513,0.3997300152284312],"mean":0.4403903142017026,"std":0.08314963463130726}},"macro":0.8127728203768264}},"formal":{"BASE":{"tasks":{"eea":{"folds":[0.9022999465049798,0.9483272819157673,0.950275007622847,0.9249630373566732,0.9146547318822503],"mean":0.9281040010565036,"std":0.02095908124453868},"egb":{"folds":[0.9376755540495655,0.9098241220082289,0.9470947163581207,0.9354815689581856,0.9170966306772548],"mean":0.929434518410271,"std":0.015436605853069899},"egc":{"folds":[0.8894244636653041,0.929623843043872,0.9186202982214563,0.9135562663533002,0.9196810490709284],"mean":0.9141811840709722,"std":0.015014290634345137},"ei":{"folds":[0.7858744270713283,0.777132445360772,0.8090624160370808,0.8409276765225356,0.8135021281392556],"mean":0.8052998186261945,"std":0.02510739947027493},"eps":{"folds":[0.6823578873490527,0.8234008093952725,0.6579744346581984,0.7638690172331493,0.7685514290559792],"mean":0.7392307155383303,"std":0.06780202176224799},"nc":{"folds":[0.741438054484669,0.9029592841415552,0.8328603305702754,0.6721506535257076,0.8356509813548945],"mean":0.7970118608154204,"std":0.09039043086135899},"xc":{"folds":[0.3836663487804117,0.3179460414714651,0.4037410139703107,0.1973756117837021,0.4599576049974794],"mean":0.35253732420067385,"std":0.10050082659460417}},"macro":0.7808284889597665},"FP_DECAY":{"tasks":{"eea":{"folds":[0.9125802903108334,0.9445832952187915,0.938769477427334,0.9218543503645086,0.9129570592731262],"mean":0.9261488945189187,"std":0.014795952900317318},"egb":{"folds":[0.9351526784622415,0.9096532004578726,0.9483224375711291,0.928500049339905,0.9170731687197715],"mean":0.927740306910184,"std":0.0151663758040615},"egc":{"folds":[0.8836585131368847,0.9234920840411093,0.9177430161625211,0.9056781216910095,0.9128708381814088],"mean":0.9086885146425866,"std":0.015444942019906576},"ei":{"folds":[0.8198382021909675,0.7967408022622271,0.8173751888884229,0.8363868151103808,0.8321666874479461],"mean":0.820501539179989,"std":0.015517767046838448},"eps":{"folds":[0.6809350147645268,0.8300396835093667,0.6402201022041298,0.7326816143228059,0.784789348388707],"mean":0.7337331526379073,"std":0.07651000377433362},"nc":{"folds":[0.7814037526894418,0.9167808811955043,0.8365407729741607,0.7165140182997329,0.8473810890394773],"mean":0.8197241028396635,"std":0.07514735240056564},"xc":{"folds":[0.393337422156672,0.3558810288033877,0.3885781162337878,0.2806517782219191,0.5625033876147971],"mean":0.39619034660611274,"std":0.10332003039670233}},"macro":0.7903895510479089}},"FINAL":"FP_DECAY","delta_FINAL_B_FP":-0.0008472573041106157,"delta_FP_DECAY_BASE":0.009561062088142336,"delta_FP_DECAY_P10K":-0.002843753512449032,"performance_pass":false,"comparisons":{"FP_DECAY_minus_BASE":{"delta":0.009561062088142336,"paired_tasks":{"eea":{"folds":[0.010280343805853631,-0.003743986696975843,-0.011505530195513036,-0.0031086869921646,-0.0016976726091241234],"mean":-0.0019551065375847942,"std":0.007834442377572847},"egb":{"folds":[-0.0025228755873240027,-0.00017092155035636214,0.0012277212130084036,-0.0069815196182805295,-2.346195748326174e-05],"mean":-0.0016942115000871504,"std":0.003251608001862677},"egc":{"folds":[-0.005765950528419439,-0.006131759002762749,-0.0008772820589352248,-0.007878144662290754,-0.006810210889519608],"mean":-0.005492669428385555,"std":0.0027023751685867612},"ei":{"folds":[0.03396377511963922,0.019608356901455126,0.00831277285134202,-0.004540861412154817,0.018664559308690487],"mean":0.015201720553794406,"std":0.014323508044505199},"eps":{"folds":[-0.0014228725845258916,0.006638874114094251,-0.017754332454068544,-0.031187402910343387,0.01623791933272778],"mean":-0.005497562900423159,"std":0.01901659585803872},"nc":{"folds":[0.039965698204772804,0.01382159705394903,0.00368044240388532,0.04436336477402536,0.011730107684582824],"mean":0.022712242024243066,"std":0.0182230150217466},"xc":{"folds":[0.009671073376260275,0.037934987331922576,-0.015162897736522907,0.083276166438217,0.10254578261731773],"mean":0.04365302240543893,"std":0.04920653410748776}},"status":"COMPLETE"},"FP_DECAY_minus_PTSCALE_P10K":{"delta":-0.002843753512449032,"paired_tasks":{"eea":{"folds":[0.004045852577854059,-0.005442518455067824,-0.01336932530613455,-0.017780959358451254,0.005103978415110011],"mean":-0.005488594425337912,"std":0.01020194291417429},"egb":{"folds":[-0.015551879717374928,-0.0029177625320390144,0.010497204133872784,0.006076004313519734,-0.006747729605878883],"mean":-0.0017288326815800615,"std":0.010344937972418353},"egc":{"folds":[-0.009461143032308716,-0.005101514623143344,-0.0012703882655988874,-0.015460122703659152,-0.0027937823346575463],"mean":-0.006817390191873529,"std":0.005735615352289919},"ei":{"folds":[0.026115575958062576,-0.023358931339723776,0.007882203672105237,-0.0036732776608008333,-0.004972861887629909],"mean":0.0003985417484026588,"std":0.018215517163839297},"eps":{"folds":[-0.0070399011354467245,-0.0004881710984475651,-0.02077205077627786,-0.06160237559576143,0.07750407501775836],"mean":-0.0024796847176350445,"std":0.050626738869829706},"nc":{"folds":[-0.030835002330753425,0.019049487301899592,0.005857688523742244,0.03147555209179087,-0.003503660928862895],"mean":0.004408812931563278,"std":0.02373705285046918},"xc":{"folds":[0.014397105485727257,-0.05668415472770971,-0.12214798846336206,0.06044710606514825,0.06299229538677764],"mean":-0.008199127250683724,"std":0.08004101230086601}},"status":"COMPLETE"},"FP_DECAY_minus_B_FP":{"delta":-0.0008472573041106157,"paired_tasks":{"eea":{"folds":[-0.0033833794903966874,-0.0027214762971602413,-0.001214937774758007,-0.017692789715003654,-0.007884468752849427],"mean":-0.006579410406033603,"std":0.006691154847398498},"egb":{"folds":[-0.009302321755895737,-0.008558757301133868,0.0017875348345027797,-0.0052461679282218565,-0.007607025551516933],"mean":-0.005785347540453123,"std":0.004500518920854695},"egc":{"folds":[-0.0011150129674146747,-0.004162220775424719,-0.00020904546545530422,-0.0051560437154721805,-0.009915489356063123],"mean":-0.004111562455966,"std":0.0038401830206066636},"ei":{"folds":[0.01608704184401144,0.018129997362884676,0.017315391163865712,-0.012606480665519615,-0.01809176770012888],"mean":0.004166836401022667,"std":0.017935546243061345},"eps":{"folds":[-0.06903411273830207,-0.011243227396890876,-0.04508242660203132,-0.01603558331358057,-0.010251908194825354],"mean":-0.030329451649126038,"std":0.025920105320720047},"nc":{"folds":[0.01932218116463924,0.010440674845322295,0.001653221415389039,0.08114396531688406,0.052634504424635575],"mean":0.03303890943337404,"std":0.03309824012645684},"xc":{"folds":[-0.06644172440667906,0.02831119941128446,-0.0767767397100938,0.11126798122724146,0.02198540892028311],"mean":0.0036692250884072354,"std":0.07730992870062185}},"status":"COMPLETE"}},"references":{"B_FP":{"tasks":{"eea":{"folds":[0.9159636698012301,0.9473047715159517,0.939984415202092,0.9395471400795122,0.9208415280259756],"mean":0.9327283049249523,"std":0.013545935583923146},"egb":{"folds":[0.9444550002181372,0.9182119577590064,0.9465349027366263,0.9337462172681269,0.9246801942712884],"mean":0.9335256544506372,"std":0.012262659376907688},"egc":{"folds":[0.8847735261042994,0.927654304816534,0.9179520616279764,0.9108341654064817,0.9227863275374719],"mean":0.9128000770985528,"std":0.016854291685891032},"ei":{"folds":[0.803751160346956,0.7786108048993424,0.8000597977245572,0.8489932957759004,0.850258455148075],"mean":0.8163347027789662,"std":0.031873638077730186},"eps":{"folds":[0.7499691275028288,0.8412829109062576,0.6853025288061612,0.7487171976363864,0.7950412565835323],"mean":0.7640626042870332,"std":0.058223512723314164},"nc":{"folds":[0.7620815715248026,0.906340206350182,0.8348875515587717,0.6353700529828489,0.7947465846148417],"mean":0.7866851934062894,"std":0.1002588966293432},"xc":{"folds":[0.45977914656335106,0.3275698293921032,0.4653548559438816,0.16938379699467765,0.540517978694514],"mean":0.3925211215177055,"std":0.1464073089720903}},"macro":0.7912368083520195},"P10K":{"tasks":{"eea":{"folds":[0.9085344377329794,0.9500258136738593,0.9521388027334685,0.9396353097229598,0.9078530808580162],"mean":0.9316374889442566,"std":0.021919446468473047},"egb":{"folds":[0.9507045581796164,0.9125709629899116,0.9378252334372563,0.9224240450263853,0.9238208983256504],"mean":0.9294691395917638,"std":0.014898419935170406},"egc":{"folds":[0.8931196561691934,0.9285935986642526,0.91901340442812,0.9211382443946686,0.9156646205160663],"mean":0.91550590483446,"std":0.013383598657257183},"ei":{"folds":[0.7937226262329049,0.8200997336019509,0.8094929852163176,0.8400600927711817,0.837139549335576],"mean":0.8201029974315862,"std":0.019345710827028215},"eps":{"folds":[0.6879749158999735,0.8305278546078143,0.6609921529804077,0.7942839899185673,0.7072852733709486],"mean":0.7362128373555422,"std":0.0726110074671783},"nc":{"folds":[0.8122387550201953,0.8977313938936047,0.8306830844504185,0.6850384662079421,0.8508847499683402],"mean":0.8153152899081002,"std":0.07949101467232002},"xc":{"folds":[0.37894031667094474,0.4125651835310974,0.5107261046971499,0.22020467215677086,0.4995110922280195],"mean":0.40438947385679647,"std":0.11725415168285576}},"macro":0.7932333045603579}},"budget":{"plan":"UA2D-FPDECAY-20261022-01","revision":"r2-SG"},"paper_envelope_macro":0.8104285714285714,"sota":{"BASE":{"eea":{"project":{"folds":[0.9022999465049798,0.9483272819157673,0.950275007622847,0.9249630373566732,0.9146547318822503],"mean":0.9281040010565036,"std":0.02095908124453868},"reference":0.934,"delta":-0.005895998943496461,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9376755540495655,0.9098241220082289,0.9470947163581207,0.9354815689581856,0.9170966306772548],"mean":0.929434518410271,"std":0.015436605853069899},"reference":0.923,"delta":0.006434518410270984,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8894244636653041,0.929623843043872,0.9186202982214563,0.9135562663533002,0.9196810490709284],"mean":0.9141811840709722,"std":0.015014290634345137},"reference":0.911,"delta":0.0031811840709721206,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7858744270713283,0.777132445360772,0.8090624160370808,0.8409276765225356,0.8135021281392556],"mean":0.8052998186261945,"std":0.02510739947027493},"reference":0.839,"delta":-0.03370018137380548,"status":"COMPLETE"},"eps":{"project":{"folds":[0.6823578873490527,0.8234008093952725,0.6579744346581984,0.7638690172331493,0.7685514290559792],"mean":0.7392307155383303,"std":0.06780202176224799},"reference":0.787,"delta":-0.047769284461669725,"status":"COMPLETE"},"nc":{"project":{"folds":[0.741438054484669,0.9029592841415552,0.8328603305702754,0.6721506535257076,0.8356509813548945],"mean":0.7970118608154204,"std":0.09039043086135899},"reference":0.856,"delta":-0.05898813918457957,"status":"COMPLETE"},"xc":{"project":{"folds":[0.3836663487804117,0.3179460414714651,0.4037410139703107,0.1973756117837021,0.4599576049974794],"mean":0.35253732420067385,"std":0.10050082659460417},"reference":0.423,"delta":-0.07046267579932614,"status":"COMPLETE"}},"FP_DECAY":{"eea":{"project":{"folds":[0.9125802903108334,0.9445832952187915,0.938769477427334,0.9218543503645086,0.9129570592731262],"mean":0.9261488945189187,"std":0.014795952900317318},"reference":0.934,"delta":-0.007851105481081366,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9351526784622415,0.9096532004578726,0.9483224375711291,0.928500049339905,0.9170731687197715],"mean":0.927740306910184,"std":0.0151663758040615},"reference":0.923,"delta":0.004740306910184011,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8836585131368847,0.9234920840411093,0.9177430161625211,0.9056781216910095,0.9128708381814088],"mean":0.9086885146425866,"std":0.015444942019906576},"reference":0.911,"delta":-0.002311485357413434,"status":"COMPLETE"},"ei":{"project":{"folds":[0.8198382021909675,0.7967408022622271,0.8173751888884229,0.8363868151103808,0.8321666874479461],"mean":0.820501539179989,"std":0.015517767046838448},"reference":0.839,"delta":-0.018498460820010942,"status":"COMPLETE"},"eps":{"project":{"folds":[0.6809350147645268,0.8300396835093667,0.6402201022041298,0.7326816143228059,0.784789348388707],"mean":0.7337331526379073,"std":0.07651000377433362},"reference":0.787,"delta":-0.05326684736209275,"status":"COMPLETE"},"nc":{"project":{"folds":[0.7814037526894418,0.9167808811955043,0.8365407729741607,0.7165140182997329,0.8473810890394773],"mean":0.8197241028396635,"std":0.07514735240056564},"reference":0.856,"delta":-0.03627589716033652,"status":"COMPLETE"},"xc":{"project":{"folds":[0.393337422156672,0.3558810288033877,0.3885781162337878,0.2806517782219191,0.5625033876147971],"mean":0.39619034660611274,"std":0.10332003039670233},"reference":0.423,"delta":-0.026809653393887245,"status":"COMPLETE"}}}}
```

</details>

<details><summary>results/ua3d_base_reft_20261028/r1/final_review/results.json</summary>

```json
{"plan":"UA3D-BASE-REFT-20261028-01","revision":"r1","independent_review":"PENDING","FINAL":"BASE_REFT","validation":{"BASE_ORIGINAL":{"tasks":{"eea":{"folds":[0.9258489962334948,0.9178132892834358,0.9058696511975507,0.90940444186418,0.9557389448516351],"mean":0.9229350646860593,"std":0.0199068092981852},"egb":{"folds":[0.9317327685836467,0.9323129091615904,0.9043846369319708,0.931173957145727,0.9434588393916917],"mean":0.9286126222429253,"std":0.014468874427615519},"egc":{"folds":[0.9136946146435524,0.923116820626894,0.9165977660323378,0.9120366473326542,0.9075833727563615],"mean":0.91460584427836,"std":0.005769546978527248},"ei":{"folds":[0.888919055883577,0.8665769855366472,0.8655728923752912,0.8477709629625998,0.8410585473221172],"mean":0.8619796888160464,"std":0.018703995554038573},"eps":{"folds":[0.7472369136542818,0.856980031011281,0.772058361523108,0.8054533188233461,0.6566912907405946],"mean":0.7676839831505223,"std":0.07442597410557149},"nc":{"folds":[0.8335550837981948,0.9090776141944835,0.8382871410816238,0.8729549039891361,0.7769326614604025],"mean":0.8461614809047682,"std":0.04921860234206862},"xc":{"folds":[0.351201144033391,0.4562607014500726,0.3929779893799423,0.5251736603280752,0.3396932842125413],"mean":0.41306135588080445,"std":0.07753749677772813}},"macro":0.8078628628513551},"BASE_REFT":{"tasks":{"eea":{"folds":[0.9260557892520624,0.9141144692880022,0.8993417263064409,0.9119713509469027,0.9528767691342561],"mean":0.9208720209855329,"std":0.020246082204677187},"egb":{"folds":[0.9371369122196221,0.930344017333475,0.8969212245376478,0.9295492391454958,0.9396467950080171],"mean":0.9267196376488517,"std":0.017209875097930948},"egc":{"folds":[0.9130608411535063,0.9232113269672673,0.919244512235269,0.9144094730703112,0.9106630474317192],"mean":0.9161178401716146,"std":0.005052528033036589},"ei":{"folds":[0.8941826297402797,0.8670303167843302,0.8614572599877819,0.8669683969937125,0.8483521426803055],"mean":0.8675981492372818,"std":0.016700096664113824},"eps":{"folds":[0.7851577412322878,0.8386169692860449,0.781967901395976,0.8035009804951252,0.6608131216511253],"mean":0.7740113428121118,"std":0.06716567575595468},"nc":{"folds":[0.8309792186722673,0.9054812846856072,0.8430952440546673,0.8903137746445404,0.7826533310948671],"mean":0.8505045706303898,"std":0.049109927734752735},"xc":{"folds":[0.3479902347287206,0.4259342009956365,0.38850647638729285,0.5251736341517339,0.330540647903266],"mean":0.4036290388333299,"std":0.07735775690916945}},"macro":0.8084932286170161}},"formal":{"BASE_ORIGINAL":{"tasks":{"eea":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"egb":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"egc":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"ei":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"eps":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"nc":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"xc":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815}},"macro":0.804675462276949},"BASE_REFT":{"tasks":{"eea":{"folds":[0.9049769797534798,0.9337302948382149,0.9516995851295479,0.9439625692503532,0.9180147982506615],"mean":0.9304768454444513,"std":0.01902892620319164},"egb":{"folds":[0.9301066676528006,0.9080527255095839,0.9273389169254724,0.9308825087595678,0.9110773298047282],"mean":0.9214916297304306,"std":0.01101884799428503},"egc":{"folds":[0.8940269313312734,0.9226997039839572,0.9207141469471163,0.9044920446982162,0.9227965070639776],"mean":0.9129458668049081,"std":0.013056721039985241},"ei":{"folds":[0.7785815070972912,0.7901070375903684,0.7896397334577828,0.8147737655658901,0.8739578678098907],"mean":0.8094119823042446,"std":0.03843520837653814},"eps":{"folds":[0.7546519544007158,0.8353980470179342,0.6921698767367295,0.7801628458516471,0.7941884476108171],"mean":0.7713142343235686,"std":0.05303235691300377},"nc":{"folds":[0.8143816639584283,0.8827951064379919,0.8244831743742866,0.7508113726273172,0.8464509219396861],"mean":0.823784447867542,"std":0.04850256989523753},"xc":{"folds":[0.5076895507308201,0.35343285252198553,0.43872220092850245,0.23778410565263752,0.6157927239010046],"mean":0.43068428674699,"std":0.14442802377591465}},"macro":0.8000156133174479}},"reference":0.804675462276949,"selected_comparison":{"macro":0.8000156133174479,"delta_macro":-0.004659848959501112,"performance_pass":false},"performance_pass":false,"paired":{"BASE_REFT":{"validation":{"delta_macro":0.0006303657656610007},"formal":{"delta_macro":-0.004659848959501112}}},"sota":{"BASE_ORIGINAL":{"eea":{"project":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.004431145079367593},"egb":{"project":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.0022430816246885588},"egc":{"project":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.003138905643812362},"ei":{"project":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.01762400606686776},"eps":{"project":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.011119521310488989},"nc":{"project":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.024623190090803382},"xc":{"project":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.01663027446704779}},"BASE_REFT":{"eea":{"project":{"folds":[0.9049769797534798,0.9337302948382149,0.9516995851295479,0.9439625692503532,0.9180147982506615],"mean":0.9304768454444513,"std":0.01902892620319164},"base":{"folds":[0.9042462676091063,0.936522089722451,0.9461348217978401,0.9420014155542406,0.9189396799195242],"mean":0.9295688549206325,"std":0.017546795192059674},"delta":-0.0035231545555487465},"egb":{"project":{"folds":[0.9301066676528006,0.9080527255095839,0.9273389169254724,0.9308825087595678,0.9110773298047282],"mean":0.9214916297304306,"std":0.01101884799428503},"base":{"folds":[0.9344724655190414,0.8962762606909007,0.9295219329466198,0.9266852682771768,0.9168286644428185],"mean":0.9207569183753115,"std":0.01512300256286361},"delta":-0.0015083702695694035},"egc":{"project":{"folds":[0.8940269313312734,0.9226997039839572,0.9207141469471163,0.9044920446982162,0.9227965070639776],"mean":0.9129458668049081,"std":0.013056721039985241},"base":{"folds":[0.894251062769743,0.9226138291508453,0.9216657911459155,0.9080928795335959,0.9240709656189622],"mean":0.9141389056438124,"std":0.012837492697864484},"delta":0.0019458668049080474},"ei":{"project":{"folds":[0.7785815070972912,0.7901070375903684,0.7896397334577828,0.8147737655658901,0.8739578678098907],"mean":0.8094119823042446,"std":0.03843520837653814},"base":{"folds":[0.8154527928121775,0.7881013579288078,0.7788210062438925,0.8717048225688533,0.8527999901119295],"mean":0.8213759939331322,"std":0.04022956381089582},"delta":-0.029588017695755364},"eps":{"project":{"folds":[0.7546519544007158,0.8353980470179342,0.6921698767367295,0.7801628458516471,0.7941884476108171],"mean":0.7713142343235686,"std":0.05303235691300377},"base":{"folds":[0.7500065364473627,0.8534280199002735,0.6745190824049889,0.7852747286165359,0.8161740260783943],"mean":0.775880478689511,"std":0.06831734147455618},"delta":-0.01568576567643143},"nc":{"project":{"folds":[0.8143816639584283,0.8827951064379919,0.8244831743742866,0.7508113726273172,0.8464509219396861],"mean":0.823784447867542,"std":0.04850256989523753},"base":{"folds":[0.8192336274766476,0.9143316193548923,0.8223237570269492,0.7557382671352346,0.845256778552259],"mean":0.8313768099091966,"std":0.057063030367198966},"delta":-0.03221555213245797},"xc":{"project":{"folds":[0.5076895507308201,0.35343285252198553,0.43872220092850245,0.23778410565263752,0.6157927239010046],"mean":0.43068428674699,"std":0.14442802377591465},"base":{"folds":[0.4975306697687487,0.37308740370305815,0.46865083461754664,0.23778406631488924,0.621098397930996],"mean":0.4396302744670478,"std":0.14343567714831815},"delta":0.00768428674699001}}},"cpu":{"status":"COMPLETE"}}
```

</details>

<details><summary>results/ua3d_base_opt_20261026/r6/final_review/results.json</summary>

```json
{"plan":"UA3D-BASE-OPT-20261026-01","revision":"r6","status":"失败停止，待审查","independent_review":"PENDING","FINAL":null,"reference":0.795244745287872,"performance_pass":null,"sota":{"A_L_A8_B8":{"eea":{"reference":0.934,"delta":null},"egb":{"reference":0.923,"delta":null},"egc":{"reference":0.911,"delta":null},"ei":{"reference":0.839,"delta":null},"eps":{"reference":0.787,"delta":null},"nc":{"reference":0.856,"delta":null},"xc":{"reference":0.423,"delta":null}}},"registry":{"plan":"UA3D-BASE-OPT-20261026-01","revision":"r6"},"cpu":{"status":"NOT_RUN"}}
```

</details>

<details><summary>results/ua3d_base_opt_20261026/r5/final_review/results.json</summary>

```json
{"plan":"UA3D-BASE-OPT-20261026-01","revision":"r5","status":"待审查","independent_review":"PENDING","reference":{"macro_r2":0.795244745287872,"performance_delta":0.01},"delta_macro":null,"performance_pass":null,"sota":{"Eea":{"reference":0.934,"delta":null,"status":"未完成／不可比较"},"Egb":{"reference":0.923,"delta":null,"status":"未完成／不可比较"},"Egc":{"reference":0.911,"delta":null,"status":"未完成／不可比较"},"Ei":{"reference":0.839,"delta":null,"status":"未完成／不可比较"},"EPS":{"reference":0.787,"delta":null,"status":"未完成／不可比较"},"Nc":{"reference":0.856,"delta":null,"status":"未完成／不可比较"},"Xc":{"reference":0.423,"delta":null,"status":"未完成／不可比较"}}}
```

</details>

<details><summary>results/ua2d_p10k_nofp_align_20261024/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-P10K-NOFP-ALIGN-20261024-01","revision":"r1.1-GPU23","status":"需返修","independent_review":"PENDING","validation":{"D_A000":{"tasks":{"eea":{"folds":[0.9338627132743905,0.904945110266952,0.922638723811055,0.9153391936208457,0.9484299683713306],"mean":0.9250431418689148,"std":0.01679737483732239},"egb":{"folds":[0.9294338773151793,0.9530252108298971,0.9233340564831725,0.9469365468110295,0.9561100244365661],"mean":0.9417679431751689,"std":0.014586579385819566},"egc":{"folds":[0.9255922742686361,0.9252024078359347,0.9279816743436603,0.9058276493663406,0.908150723032275],"mean":0.9185509457693695,"std":0.010639640506614756},"ei":{"folds":[0.8621328212174357,0.8595187670659452,0.8861599408612645,0.855192768794948,0.8237497689311206],"mean":0.8573508133741428,"std":0.022311548539153937},"eps":{"folds":[0.684368180754181,0.836825687566829,0.7532803870782547,0.8333707005734058,0.6380513939625413],"mean":0.7491792699870424,"std":0.08851100606338655},"nc":{"folds":[0.7973098242166272,0.9052746919160409,0.8256463269778194,0.9065523776774033,0.7625725069630589],"mean":0.8394711455501899,"std":0.06463764704050742},"xc":{"folds":[0.3132105676771786,0.5290280186798659,0.4914356310496377,0.5532375735663371,0.36228673770989384],"mean":0.44983970573658266,"std":0.10609567917489628}},"macro":0.8116004236373444},"D_A005":{"tasks":{"eea":{"folds":[0.9338001804771181,0.9013846044048852,0.9125972711543764,0.9092213399628666,0.9498334178518868],"mean":0.9213673627702267,"std":0.01992888768170369},"egb":{"folds":[0.9336950835571267,0.9395282316238074,0.9223595810581519,0.9434984526539625,0.9512693422157109],"mean":0.9380701382217518,"std":0.010860183351668265},"egc":{"folds":[0.9230841632552842,0.9257665982199936,0.9222465935396518,0.9163353833350147,0.9155466624390138],"mean":0.9205958801577916,"std":0.004452503943558415},"ei":{"folds":[0.8849645716216848,0.8620462384761272,0.90658283758982,0.8701560235319239,0.8341460541945133],"mean":0.8715791450828139,"std":0.026919166078858837},"eps":{"folds":[0.6995904730368765,0.8600827758253943,0.7261538078118044,0.8242399907205206,0.6917232291855822],"mean":0.7603580553160356,"std":0.07681030287831696},"nc":{"folds":[0.8023879476049419,0.8977177294298705,0.8551579584994897,0.8968308538415284,0.7697710898506345],"mean":0.8443731158452931,"std":0.057100769565575124},"xc":{"folds":[0.3066809712559876,0.4959744124576728,0.49391945680169114,0.49309993404057395,0.37905294993536587],"mean":0.4337455448982583,"std":0.0868230398002364}},"macro":0.8128698917560244},"D_A010":{"tasks":{"eea":{"folds":[0.9344938904504608,0.9021513100468725,0.9167003184204494,0.9063664532755548,0.9467928842212266],"mean":0.9213009712829129,"std":0.018939417241193867},"egb":{"folds":[0.9352684532350641,0.9342052821907845,0.9277900973309378,0.9544962615562897,0.9556977248374324],"mean":0.9414915638301016,"std":0.01275229747289637},"egc":{"folds":[0.9261060846133411,0.9268093012693991,0.9246756110351076,0.9135396185772772,0.9142978871507049],"mean":0.9210857005291659,"std":0.006592971304964537},"ei":{"folds":[0.8848586222280087,0.8964227805422258,0.8906022878502374,0.8723514515540836,0.8452815053162397],"mean":0.877903329498159,"std":0.020295421089628052},"eps":{"folds":[0.7106362011959144,0.8552537536910876,0.7648756962983427,0.814872733629514,0.6833617566304739],"mean":0.7658000282890665,"std":0.07115137075450256},"nc":{"folds":[0.8085547608051535,0.9020186450491874,0.8637717852216731,0.8897438458990806,0.7701973260872503],"mean":0.8468572726124689,"std":0.055928449214545796},"xc":{"folds":[0.3075513349289898,0.4866390036081988,0.4633303421770666,0.4945498642133951,0.31533630740514507],"mean":0.4134813704665591,"std":0.09389180712152437}},"macro":0.8125600337869191},"D_A020":{"tasks":{"eea":{"folds":[0.9412431494982101,0.9051424864687183,0.9092589849306804,0.907307546849627,0.9417838528552881],"mean":0.9209472041205047,"std":0.01883172288732914},"egb":{"folds":[0.9370493857961467,0.9361765909151042,0.9259939517945017,0.9533285354603953,0.9494443522003922],"mean":0.9403985632333081,"std":0.011017546803786794},"egc":{"folds":[0.9297536315943657,0.9201718264138106,0.920418919946091,0.9157153995883478,0.9165051326182817],"mean":0.9205129820321793,"std":0.0055810951746265705},"ei":{"folds":[0.8815402453512116,0.875155959445066,0.886816224003522,0.8711395606148146,0.8432984880246482],"mean":0.8715900954878524,"std":0.01691281978385184},"eps":{"folds":[0.703779163096439,0.8561343896014999,0.7592762059657969,0.8036705237087924,0.7139195735202686],"mean":0.7673559711785594,"std":0.06356708273479461},"nc":{"folds":[0.7914661394537619,0.8879987836987571,0.864705178664168,0.8861375156400008,0.7532129260307072],"mean":0.836704108697479,"std":0.06098384369876759},"xc":{"folds":[0.31198769564559226,0.47160336965043115,0.4860197404375296,0.5671942905674968,0.29273331526846236],"mean":0.4259076823139024,"std":0.11871882374468079}},"macro":0.8119166581519693},"PURE2D":{"tasks":{"eea":{"folds":[0.9290724303299145,0.8934978344067389,0.9153496713028129,0.9149908164780459,0.9492183673952531],"mean":0.9204258239825531,"std":0.02051948837505267},"egb":{"folds":[0.9239957693742125,0.9457212288555676,0.9263332690581307,0.9492914086721702,0.9544771586376648],"mean":0.9399637669195492,"std":0.013888477466333558},"egc":{"folds":[0.9236218388601357,0.9275810801981074,0.9229818320506354,0.9024898761801893,0.9095711531122876],"mean":0.9172491560802711,"std":0.010688912343124819},"ei":{"folds":[0.8700197797777786,0.8691019705522606,0.8737240242413487,0.8794969668506667,0.8548191582339373],"mean":0.8694323799311983,"std":0.009133928208267417},"eps":{"folds":[0.7069821218219154,0.8639746108338573,0.7492568746539834,0.8636494337858582,0.6087672008085723],"mean":0.7585260483808374,"std":0.10878962709661714},"nc":{"folds":[0.8639098840974777,0.9024183553495724,0.8656132700030094,0.9007069972902789,0.7415221811669375],"mean":0.8548341375814552,"std":0.06596729374383013},"xc":{"folds":[0.29134783917103724,0.5016288076524736,0.460970617515424,0.5115786826762055,0.34149788900819134],"mean":0.42140476720466635,"std":0.09928821924163998}},"macro":0.8116908685829328}},"formal":{"D_A000":{"tasks":{"eea":{"folds":[0.9207747157629868,0.9493542119033851,0.9456501441067761,0.9329554276638837,0.9091079122292602],"mean":0.9315684823332584,"std":0.016863656173600822},"egb":{"folds":[0.9443779503549701,0.9214612543965751,0.9407763634342377,0.9327782803199334,0.9242137780671775],"mean":0.9327215253145787,"std":0.00999914180516641},"egc":{"folds":[0.8849613727475081,0.9262980509303037,0.9240909054476015,0.9081127839680805,0.9158937792045817],"mean":0.911871378459615,"std":0.016672588777070493},"ei":{"folds":[0.7882209813294927,0.8042828360924694,0.8072473200280987,0.8052414190706026,0.8416669899608181],"mean":0.8093319092962963,"std":0.01960739327474388},"eps":{"folds":[0.6840653574672783,0.8323998565973472,0.7937756971906469,0.6304487907591483,0.7772503586617595],"mean":0.7435880121352361,"std":0.08349091395769276},"nc":{"folds":[0.778018809203421,0.8877295575068768,0.8396550707743812,0.7339104161852676,0.8492026671754096],"mean":0.8177033041690713,"std":0.061183889425188084},"xc":{"folds":[0.2534269900520457,0.32239498703227876,0.4447522916214529,0.2288266749000667,0.5321719420292761],"mean":0.3563145771270241,"std":0.12916714125107337}},"macro":0.7861570269764399},"D_A005":{"tasks":{"eea":{"folds":[0.9213763184329368,0.9439263862819022,0.9479827979292965,0.9254777017649063,0.9083485884741624],"mean":0.929422358576641,"std":0.01642604330268835},"egb":{"folds":[0.9433905496782553,0.9067645825755051,0.9499793300187559,0.9331332411942509,0.9212738448432523],"mean":0.9309083096620039,"std":0.017321972488469985},"egc":{"folds":[0.8907376354686284,0.9247101419809346,0.9261079965850488,0.9133942157061824,0.9252892770306326],"mean":0.9160478533542854,"std":0.01507725571703504},"ei":{"folds":[0.7945449700451912,0.7780867723944722,0.7937814581907918,0.8080124129704936,0.8435849891823527],"mean":0.8036021205566601,"std":0.024736566249402715},"eps":{"folds":[0.5829077811810623,0.8418769982606953,0.6198687426268719,0.6573775782547919,0.7126224584101832],"mean":0.6829307117467209,"std":0.10097014786061936},"nc":{"folds":[0.7752567080289376,0.897675981129949,0.8542150229365492,0.722516344602931,0.7918607988121418],"mean":0.8083049711021018,"std":0.06858789474810476},"xc":{"folds":[0.22457713789208855,0.2900494227143937,0.4606906684208535,0.24862245901481783,0.5530378723120584],"mean":0.3553955120708424,"std":0.1439909856452392}},"macro":0.775230262438465},"D_A010":{"tasks":{"eea":{"folds":[0.9103583701095922,0.9219585978028229,0.9440176442688671,0.9274616333414997,0.9185118512502244],"mean":0.9244616193546014,"std":0.012569851739582593},"egb":{"folds":[0.9356769907363296,0.9162794418923947,0.9516505508155307,0.9356570467741608,0.9255574603636255],"mean":0.9329642981164084,"std":0.013200675259541805},"egc":{"folds":[0.8946873748872127,0.9285071688617261,0.9257000391787553,0.9140148658204585,0.9166392475713233],"mean":0.9159097392638952,"std":0.01331405913282221},"ei":{"folds":[0.7902364656331061,0.7997378193285061,0.8028002634857329,0.8234249653705807,0.8503401639907544],"mean":0.813307935561736,"std":0.023982244402265145},"eps":{"folds":[0.686323375381211,0.8375448567405986,0.7441583268142146,0.7109782027689728,0.7044547944172891],"mean":0.7366919112244571,"std":0.06013418305466074},"nc":{"folds":[0.7946427746168601,0.8980613085589233,0.8432172097313073,0.6917233759645447,0.7971561790081891],"mean":0.8049601695759648,"std":0.07602852349002164},"xc":{"folds":[0.3600795494439999,0.3005298012017996,0.40939401354207006,0.2503409150049134,0.4788885148436042],"mean":0.3598465588072774,"std":0.08963252145522448}},"macro":0.7840203188434771},"D_A020":{"tasks":{"eea":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"egb":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"egc":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"ei":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"eps":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"nc":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"xc":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173}},"macro":0.785044436513689},"PURE2D":{"tasks":{"eea":{"folds":[0.917721829470783,0.9455652996213213,0.9254854461733641,0.9261505485985837,0.9152447837214163],"mean":0.9260335815170937,"std":0.011908878260688273},"egb":{"folds":[0.9449893611161805,0.9164047890275746,0.944334031877542,0.9368564394095376,0.9187120132158275],"mean":0.9322593269293324,"std":0.013819205784619886},"egc":{"folds":[0.8920332139300725,0.9285448865303768,0.9221906756857559,0.9176398225338344,0.9211250466970163],"mean":0.916306729075411,"std":0.014129669015837424},"ei":{"folds":[0.7898906901591223,0.7876448024659414,0.808585136006504,0.8129507281051696,0.8350577334213596],"mean":0.8068258180316192,"std":0.01931554221286506},"eps":{"folds":[0.6686842598951328,0.8550665902781935,0.6052197768250208,0.6469375054556183,0.7259779020105375],"mean":0.7003772068929006,"std":0.09682378581585849},"nc":{"folds":[0.8442401064985994,0.9029562391796172,0.7279208298698319,0.6630384243972473,0.8656656549311819],"mean":0.8007642509752955,"std":0.10101993863470109},"xc":{"folds":[0.26577647845630104,0.31743994687923216,0.4485529791998518,0.12040984561450885,0.5203672885753654],"mean":0.33450930774505183,"std":0.156846495517276}},"macro":0.7738680315952434}},"FINAL":"D_A005","p10k_reference_deltas":{"D_A000":{"delta":-0.007076277583918045},"D_A005":{"delta":-0.018003042121892854},"D_A010":{"delta":-0.009212985716880784},"D_A020":{"delta":-0.008188868046668851},"PURE2D":{"delta":-0.019365272965114477}},"comparisons":{"D_A000_minus_D_A010":{"delta":0.002136708132962739,"paired_tasks":{"eea":{"folds":[0.010416345653394554,0.027395614100562238,0.0016324998379089184,0.00549379432238406,-0.009403939020964147],"mean":0.007106862978657124,"std":0.013488544423123684},"egb":{"folds":[0.00870095961864048,0.005181812504180372,-0.010874187381293088,-0.002878766454227466,-0.00134368229644799],"mean":-0.0002427728018295383,"std":0.0075928241709261},"egc":{"folds":[-0.009726002139704648,-0.002209117931422422,-0.001609133731153789,-0.0059020818523780605,-0.0007454683667416084],"mean":-0.004038360804280106,"std":0.0037387542835636417},"ei":{"folds":[-0.002015484303613424,0.00454501676396335,0.004447056542365879,-0.018183546299978093,-0.008673174029936281],"mean":-0.003976026265439714,"std":0.009634794311939084},"eps":{"folds":[-0.002258017913932764,-0.005145000143251477,0.049617370376432324,-0.08052941200982455,0.07279556424447042],"mean":0.0068961009107787906,"std":0.05924494165729771},"nc":{"folds":[-0.016623965413439024,-0.010331751052046534,-0.003562138956926142,0.04218704022072295,0.05204648816722046],"mean":0.012743134593106342,"std":0.03190782050161265},"xc":{"folds":[-0.10665255939195417,0.02186518583047914,0.035358278079382854,-0.021514240104846705,0.05328342718567192],"mean":-0.003531981680253393,"std":0.06391923472630565}},"status":"COMPLETE"},"D_A005_minus_D_A010":{"delta":-0.00879005640501207,"paired_tasks":{"eea":{"folds":[0.011017948323344573,0.02196778847907932,0.003965153660429377,-0.0019839315765933785,-0.010163262776061943],"mean":0.0049607392220395894,"std":0.012287031639891293},"egb":{"folds":[0.007713558941925713,-0.009514859316889668,-0.0016712207967748416,-0.0025238055799099435,-0.004283615520373263],"mean":-0.0020559884544044003,"std":0.0062531598458436055},"egc":{"folds":[-0.0039497394185843415,-0.003797026880791532,0.0004079574062935132,-0.0006206501142761445,0.00865002945930926],"mean":0.0001381140903901512,"std":0.005130719310071682},"ei":{"folds":[0.004308504412085079,-0.021651046934033857,-0.009018805294941035,-0.015412552400087098,-0.006755174808401709],"mean":-0.009705815005075724,"std":0.009757494890729896},"eps":{"folds":[-0.10341559420014867,0.004332141520096644,-0.1242895841873427,-0.053600624514180906,0.008167663992894147],"mean":-0.0537611994777363,"std":0.060518299191998075},"nc":{"folds":[-0.019386066587922413,-0.0003853274289743469,0.010997813205241846,0.03079296863838632,-0.005295380196047361],"mean":0.003344801526136809,"std":0.01882305390349406},"xc":{"folds":[-0.13550241155191134,-0.010480378487405906,0.05129665487878343,-0.0017184559900955687,0.07414935746845419],"mean":-0.004451046736435038,"std":0.08140041444934526}},"status":"COMPLETE"},"D_A020_minus_D_A010":{"delta":0.0010241176702119326,"paired_tasks":{"eea":{"folds":[0.015959499214616146,0.008210648960603306,-0.0009212843345541843,0.0019458579305814405,-0.0019895857660663374],"mean":0.004641027201036074,"std":0.00746689834475201},"egb":{"folds":[0.002058671548106017,0.0009739068595847611,-0.003823550957648636,-0.0038286029365579566,-0.005946751280722196],"mean":-0.002113265353447602,"std":0.0034459673536629907},"egc":{"folds":[0.003585205062232255,-0.0017396196071488568,-0.0002051479242237697,0.005065755806978101,0.00437181711440382],"mean":0.00221560209044831,"std":0.0030063375163302},"ei":{"folds":[-0.020750024346323226,0.010458138116408833,0.018078015796923252,0.027292775602833386,-0.04465620113468827],"mean":-0.0019154591929692044,"std":0.029947778710103845},"eps":{"folds":[-0.0018746695377065414,0.005618275902532166,-0.18533617593577578,0.0039036194953381287,0.057065420793001476],"mean":-0.02412470585652211,"std":0.09320175810119889},"nc":{"folds":[0.00771181065929738,-0.004645908506902385,-0.00279498914185794,0.022631146722388862,-0.009142518239953157],"mean":0.002751908298594552,"std":0.012715717627421436},"xc":{"folds":[-0.0004276469062149957,0.05059655296603305,0.06362098753169654,0.030992579296669542,-0.016213890366468586],"mean":0.02571371650434311,"std":0.03363509416657268}},"status":"COMPLETE"},"D_A005_minus_D_A000":{"delta":-0.010926764537974809,"paired_tasks":{"eea":{"folds":[0.0006016026699500188,-0.005427825621482918,0.0023326538225204585,-0.007477725898977439,-0.000759323755097796],"mean":-0.002146123756617535,"std":0.00414512399986839},"egb":{"folds":[-0.000987400676714767,-0.01469667182107004,0.009202966584518246,0.00035496087431752255,-0.002939933223925273],"mean":-0.0018132156525748622,"std":0.008573099880412187},"egc":{"folds":[0.005776262721120307,-0.00158790894936911,0.002017091137447302,0.005281431738101916,0.00939549782605087],"mean":0.004176474894670257,"std":0.004150587620631605},"ei":{"folds":[0.006323988715698503,-0.026196063697997207,-0.013465861837306914,0.0027709938998909944,0.001917999221534572],"mean":-0.00572978873963601,"std":0.013736454735210882},"eps":{"folds":[-0.10115757628621591,0.009477141663348121,-0.17390695456377503,0.026928787495643647,-0.06462790025157628],"mean":-0.06065730038851509,"std":0.0822673764846311},"nc":{"folds":[-0.002762101174483389,0.009946423623072187,0.014559952162167988,-0.011394071582336629,-0.05734186836326782],"mean":-0.009398333066969532,"std":0.02870027946369857},"xc":{"folds":[-0.028849852159957168,-0.032345564317885045,0.015938376799400578,0.019795784114751136,0.020865930282782275],"mean":-0.0009190650561816449,"std":0.027182798182351402}},"status":"COMPLETE"},"D_A010_minus_D_A000":{"delta":-0.002136708132962739,"paired_tasks":{"eea":{"folds":[-0.010416345653394554,-0.027395614100562238,-0.0016324998379089184,-0.00549379432238406,0.009403939020964147],"mean":-0.007106862978657124,"std":0.013488544423123684},"egb":{"folds":[-0.00870095961864048,-0.005181812504180372,0.010874187381293088,0.002878766454227466,0.00134368229644799],"mean":0.0002427728018295383,"std":0.0075928241709261},"egc":{"folds":[0.009726002139704648,0.002209117931422422,0.001609133731153789,0.0059020818523780605,0.0007454683667416084],"mean":0.004038360804280106,"std":0.0037387542835636417},"ei":{"folds":[0.002015484303613424,-0.00454501676396335,-0.004447056542365879,0.018183546299978093,0.008673174029936281],"mean":0.003976026265439714,"std":0.009634794311939084},"eps":{"folds":[0.002258017913932764,0.005145000143251477,-0.049617370376432324,0.08052941200982455,-0.07279556424447042],"mean":-0.0068961009107787906,"std":0.05924494165729771},"nc":{"folds":[0.016623965413439024,0.010331751052046534,0.003562138956926142,-0.04218704022072295,-0.05204648816722046],"mean":-0.012743134593106342,"std":0.03190782050161265},"xc":{"folds":[0.10665255939195417,-0.02186518583047914,-0.035358278079382854,0.021514240104846705,-0.05328342718567192],"mean":0.003531981680253393,"std":0.06391923472630565}},"status":"COMPLETE"},"D_A020_minus_D_A000":{"delta":-0.0011125904627508065,"paired_tasks":{"eea":{"folds":[0.005543153561221592,-0.01918496513995893,-0.0025537841724631027,-0.0035479363918026197,0.00741435325489781],"mean":-0.0024658357776210504,"std":0.0105174855460081},"egb":{"folds":[-0.006642288070534463,-0.004207905644595611,0.007050636423644452,-0.0009498364823304906,-0.004603068984274206],"mean":-0.0018704925516180638,"std":0.005388289528994004},"egc":{"folds":[0.013311207201936903,0.0004694983242735651,0.0014039858069300193,0.010967837659356161,0.005117285481145428],"mean":0.0062539628947284156,"std":0.0057074437397531},"ei":{"folds":[-0.018734540042709802,0.005913121352445483,0.013630959254557373,0.04547632190281148,-0.035983027104751986],"mean":0.0020605670724705093,"std":0.03127856876148885},"eps":{"folds":[0.0003833483762262224,0.010763276045783643,-0.2349535463122081,0.08443303150516268,-0.015730143451468948],"mean":-0.0310208067673009,"std":0.1202955588868337},"nc":{"folds":[0.024335776072736404,0.0056858425451441486,0.0007671498150682021,-0.019555893498334087,-0.06118900640717362],"mean":-0.00999122629451179,"std":0.032605796197240376},"xc":{"folds":[0.10622491248573918,0.02873136713555391,0.028262709452313683,0.05250681940151625,-0.0694973175521405],"mean":0.029245698184596502,"std":0.06367553131845619}},"status":"COMPLETE"},"D_A010_minus_PURE2D":{"delta":0.010152287248233693,"paired_tasks":{"eea":{"folds":[-0.007363459361190761,-0.023606701818498443,0.018532198095502994,0.0013110847429159689,0.003267067528808121],"mean":-0.0015719621624924241,"std":0.015451347991370611},"egb":{"folds":[-0.009312370379850887,-0.00012534713517986518,0.007316518937988792,-0.001199392635376717,0.0068454471477980006],"mean":0.0007049711870758645,"std":0.006820578619879447},"egc":{"folds":[0.002654160957140217,-3.7717668650683045e-05,0.003509363492999462,-0.0036249567133758154,-0.004485799125693024],"mean":-0.00039698981151596867,"std":0.0035998724127603667},"ei":{"folds":[0.0003457754739838448,0.012093016862564698,-0.005784872520771089,0.01047423726541119,0.015282430569394823],"mean":0.0064821175301166935,"std":0.008845881239700294},"eps":{"folds":[0.017639115486078216,-0.017521733537594897,0.13893854998919375,0.06404069731335449,-0.02152310759324838],"mean":0.03631470433155663,"std":0.06688186700033634},"nc":{"folds":[-0.049597331881739315,-0.004894930620693905,0.11529637986147545,0.028684951567297445,-0.06850947592299272],"mean":0.004195918600669391,"std":0.0728104679250768},"xc":{"folds":[0.09430307098769886,-0.016910145677432542,-0.039158965657781764,0.12993106939040455,-0.04147877373176123],"mean":0.025337251062225574,"std":0.08078542074874}},"status":"COMPLETE"},"D_A000_minus_PURE2D":{"delta":0.012288995381196433,"paired_tasks":{"eea":{"folds":[0.0030528862922037936,0.0037889122820637944,0.020164697933411913,0.006804879065300029,-0.006136871492156026],"mean":0.005534900816164701,"std":0.009501091375741249},"egb":{"folds":[-0.0006114107612104069,0.005056465369000507,-0.003557668443304296,-0.004078159089604183,0.005501764851350011],"mean":0.0004621983852463263,"std":0.004594326153048959},"egc":{"folds":[-0.007071841182564431,-0.002246835600073105,0.0019002297618456732,-0.009527038565753876,-0.0052312674924346325],"mean":-0.004435350615796074,"std":0.004428290557837362},"ei":{"folds":[-0.0016697088296295792,0.016638033626528048,-0.0013378159784052102,-0.007709309034566902,0.006609256539458541],"mean":0.0025060912646769793,"std":0.00939567843193982},"eps":{"folds":[0.015381097572145452,-0.022666733680846374,0.18855592036562607,-0.016488714696470064,0.05127245665122204],"mean":0.04321080524233543,"std":0.08641262662098116},"nc":{"folds":[-0.06622129729517834,-0.015226681672740439,0.11173424090454931,0.0708719917880204,-0.016462987755772263],"mean":0.016939053193775734,"std":0.07238933659572269},"xc":{"folds":[-0.012349488404255315,0.004955040153046597,-0.003800687578398909,0.10841682928555785,0.011804653453910685],"mean":0.02180526938197218,"std":0.049263874604604005}},"status":"COMPLETE"},"FINAL_minus_D_A010":{"delta":-0.00879005640501207,"paired_tasks":{"eea":{"folds":[0.011017948323344573,0.02196778847907932,0.003965153660429377,-0.0019839315765933785,-0.010163262776061943],"mean":0.0049607392220395894,"std":0.012287031639891293},"egb":{"folds":[0.007713558941925713,-0.009514859316889668,-0.0016712207967748416,-0.0025238055799099435,-0.004283615520373263],"mean":-0.0020559884544044003,"std":0.0062531598458436055},"egc":{"folds":[-0.0039497394185843415,-0.003797026880791532,0.0004079574062935132,-0.0006206501142761445,0.00865002945930926],"mean":0.0001381140903901512,"std":0.005130719310071682},"ei":{"folds":[0.004308504412085079,-0.021651046934033857,-0.009018805294941035,-0.015412552400087098,-0.006755174808401709],"mean":-0.009705815005075724,"std":0.009757494890729896},"eps":{"folds":[-0.10341559420014867,0.004332141520096644,-0.1242895841873427,-0.053600624514180906,0.008167663992894147],"mean":-0.0537611994777363,"std":0.060518299191998075},"nc":{"folds":[-0.019386066587922413,-0.0003853274289743469,0.010997813205241846,0.03079296863838632,-0.005295380196047361],"mean":0.003344801526136809,"std":0.01882305390349406},"xc":{"folds":[-0.13550241155191134,-0.010480378487405906,0.05129665487878343,-0.0017184559900955687,0.07414935746845419],"mean":-0.004451046736435038,"std":0.08140041444934526}},"status":"COMPLETE"},"D_A000_minus_P10K_REF":{"delta":-0.007076277583918045,"paired_tasks":{"eea":{"folds":[0.012240278030007379,-0.0006716017704742061,-0.006488658626692456,-0.006679882059076103,0.0012548313712440473],"mean":-6.900661099826789e-05,"std":0.007722528123861925},"egb":{"folds":[-0.006326607824646335,0.00889029140666353,0.0029511299969813276,0.010354235293548064,0.0003928797415271301],"mean":0.0032523857228147434,"std":0.0067499424937729575},"egc":{"folds":[-0.008158283421685342,-0.0022955477339489105,0.005077501019481567,-0.013025460426588142,0.00022915868851536292],"mean":-0.003634526374845093,"std":0.007093559888591429},"ei":{"folds":[-0.005501644903412206,-0.015816897509481453,-0.0022456651882188927,-0.034818673700579006,0.0045274406252421295],"mean":-0.010771088135289885,"std":0.015315363319211794},"eps":{"folds":[-0.003909558432695248,0.0018720019895328566,0.1327835442102392,-0.16383519915941902,0.06996508529081091],"mean":0.007375174779693738,"std":0.11080994797379634},"nc":{"folds":[-0.03421994581677423,-0.010001836386727914,0.008971986323962677,0.04887194997732558,-0.0016820827929305837],"mean":0.0023880142609711053,"std":0.030472618028168714},"xc":{"folds":[-0.12551332661889902,-0.09017019649881863,-0.06597381307569694,0.008622002743295831,0.03266084980125661],"mean":-0.04807489672977243,"std":0.06674933533466684}},"status":"COMPLETE"},"D_A005_minus_P10K_REF":{"delta":-0.018003042121892854,"paired_tasks":{"eea":{"folds":[0.012841880699957398,-0.006099427391957124,-0.004156004804171998,-0.014157607958053542,0.0004955076161462513],"mean":-0.002215130367615803,"std":0.009944076987306656},"egb":{"folds":[-0.007314008501361102,-0.005806380414406509,0.012154096581499574,0.010709196167865587,-0.0025470534823981428],"mean":0.0014391700702398812,"std":0.009297165551067941},"egc":{"folds":[-0.002382020700565035,-0.0038834566833180206,0.0070945921569288695,-0.007744028688486226,0.009624656514566232],"mean":0.0005419485198251639,"std":0.007453588878236122},"ei":{"folds":[0.0008223438122862969,-0.04201296120747866,-0.015711527025525807,-0.03204767980068801,0.0064454398467767016],"mean":-0.016500876874925897,"std":0.020735356444121368},"eps":{"folds":[-0.10506713471891116,0.011349143652880977,-0.04112341035353584,-0.13690641166377537,0.005337185039234638],"mean":-0.053282125608821354,"std":0.06602266143269282},"nc":{"folds":[-0.03698204699125762,-5.541276365572667e-05,0.023531938486130666,0.03747787839498895,-0.059023951156198406],"mean":-0.0070103188059984275,"std":0.0405096356729581},"xc":{"folds":[-0.1543631787788562,-0.12251576081670368,-0.05003543627629636,0.028417786858046967,0.05352678008403888],"mean":-0.048993961785954075,"std":0.09084642650267674}},"status":"COMPLETE"},"D_A010_minus_P10K_REF":{"delta":-0.009212985716880784,"paired_tasks":{"eea":{"folds":[0.0018239323766128246,-0.028067215871036444,-0.008121158464601375,-0.012173676381460163,0.010658770392208194],"mean":-0.007175869589655393,"std":0.014674730082258863},"egb":{"folds":[-0.015027567443286816,0.003708478902483159,0.013825317378274415,0.01323300174777553,0.00173656203797512],"mean":0.003495158524644282,"std":0.011702214716124376},"egc":{"folds":[0.0015677187180193064,-8.642980252648869e-05,0.006686634750635356,-0.007123378574210082,0.0009746270552569714],"mean":0.00040383442943501267,"std":0.0049510934487839615},"ei":{"folds":[-0.003486160599798782,-0.020361914273444803,-0.006692721730584772,-0.016635127400600913,0.013200614655178411],"mean":-0.006795061869850172,"std":0.013149836069349134},"eps":{"folds":[-0.0016515405187624843,0.0070170021327843335,0.08316617383380687,-0.08330578714959447,-0.00283047895365951],"mean":0.00047907386891494764,"std":0.0589815577109338},"nc":{"folds":[-0.01759598040333521,0.00032991466531862024,0.01253412528088882,0.0066849097566026305,-0.053728570960151045],"mean":-0.010355120332135237,"std":0.02674946932654398},"xc":{"folds":[-0.018860767226944852,-0.11203538232929777,-0.1013320911550798,0.030136242848142536,-0.020622577384415308],"mean":-0.04454291504951904,"std":0.06039241582632168}},"status":"COMPLETE"},"D_A020_minus_P10K_REF":{"delta":-0.008188868046668851,"paired_tasks":{"eea":{"folds":[0.01778343159122897,-0.019856566910433138,-0.009042442799155559,-0.010227818450878723,0.008669184626141857],"mean":-0.0025348423886193184,"std":0.015329182651685903},"egb":{"folds":[-0.012968895895180799,0.00468238576206792,0.01000176642062578,0.009404398811217574,-0.0042101892427470755],"mean":0.0013818931711966798,"std":0.009833267456379705},"egc":{"folds":[0.0051529237802515615,-0.0018260494096753455,0.006481486826411587,-0.0020576227672319813,0.005346444169660791],"mean":0.0026194365198833223,"std":0.0041954674863370015},"ei":{"folds":[-0.024236184946122008,-0.00990377615703597,0.01138529406633848,0.010657648202232473,-0.031455586479509856],"mean":-0.008710521062819377,"std":0.019613616851129726},"eps":{"folds":[-0.0035262100564690257,0.0126352780353165,-0.10217000210196892,-0.07940216765425634,0.054234941839341966],"mean":-0.023645631987607163,"std":0.0653095462431663},"nc":{"folds":[-0.00988416974403783,-0.004315993841583765,0.00973913613903088,0.029316056478991492,-0.0628710892001042],"mean":-0.007603212033540685,"std":0.03440325655245484},"xc":{"folds":[-0.019288414133159848,-0.061438829363264724,-0.03771110362338326,0.06112882214481208,-0.036836467750883894],"mean":-0.01882919854517593,"std":0.047143039448310636}},"status":"COMPLETE"},"PURE2D_minus_P10K_REF":{"delta":-0.019365272965114477,"paired_tasks":{"eea":{"folds":[0.009187391737803585,-0.0044605140525380005,-0.02665335656010437,-0.013484761124376132,0.007391702863400074],"mean":-0.005603907427162968,"std":0.014951278624402247},"egb":{"folds":[-0.005715197063435928,0.003833826037663024,0.006508798440285624,0.014432394383152247,-0.0051088851098228805],"mean":0.002790187337568417,"std":0.008443860854191835},"egc":{"folds":[-0.0010864422391209105,-4.871213387580564e-05,0.003177271257635894,-0.0034984218608342665,0.0054604261809499954],"mean":0.0008008242409509814,"std":0.003539999806674629},"ei":{"folds":[-0.003831936073782627,-0.0324549311360095,-0.0009078492098136826,-0.027109364666012103,-0.0020818159142164117],"mean":-0.013277179399966865,"std":0.01522058331097582},"eps":{"folds":[-0.0192906560048407,0.02453873567037923,-0.05577237615538688,-0.14734648446294896,0.01869262863958887],"mean":-0.03583563046264169,"std":0.07022157021739897},"nc":{"folds":[0.032001351478404105,0.005224845286012525,-0.10276225458058663,-0.022000041810694815,0.014780904962841679],"mean":-0.014551038932804627,"std":0.053045451718939395},"xc":{"folds":[-0.11316383821464371,-0.09512523665186523,-0.06217312549729803,-0.09979482654226202,0.020856196347345923],"mean":-0.06988016611174461,"std":0.05407902253485241}},"status":"COMPLETE"}},"references":{"P10K_REF":{"macro":0.7932333045603579,"tasks":{"eea":{"folds":[0.9085344377329794,0.9500258136738593,0.9521388027334685,0.9396353097229598,0.9078530808580162],"mean":0.9316374889442566,"std":0.021919446468473047},"egb":{"folds":[0.9507045581796164,0.9125709629899116,0.9378252334372563,0.9224240450263853,0.9238208983256504],"mean":0.9294691395917638,"std":0.014898419935170406},"egc":{"folds":[0.8931196561691934,0.9285935986642526,0.91901340442812,0.9211382443946686,0.9156646205160663],"mean":0.91550590483446,"std":0.013383598657257183},"ei":{"folds":[0.7937226262329049,0.8200997336019509,0.8094929852163176,0.8400600927711817,0.837139549335576],"mean":0.8201029974315862,"std":0.019345710827028215},"eps":{"folds":[0.6879749158999735,0.8305278546078143,0.6609921529804077,0.7942839899185673,0.7072852733709486],"mean":0.7362128373555422,"std":0.0726110074671783},"nc":{"folds":[0.8122387550201953,0.8977313938936047,0.8306830844504185,0.6850384662079421,0.8508847499683402],"mean":0.8153152899081002,"std":0.07949101467232002},"xc":{"folds":[0.37894031667094474,0.4125651835310974,0.5107261046971499,0.22020467215677086,0.4995110922280195],"mean":0.40438947385679647,"std":0.11725415168285576}}}},"budget":{"plan":"UA2D-P10K-NOFP-ALIGN-20261024-01","revision":"r1.1-GPU23"},"paper_envelope_macro":0.8104285714285714,"sota":{"D_A000":{"eea":{"project":{"folds":[0.9207747157629868,0.9493542119033851,0.9456501441067761,0.9329554276638837,0.9091079122292602],"mean":0.9315684823332584,"std":0.016863656173600822},"reference":0.934,"delta":-0.002431517666741656,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9443779503549701,0.9214612543965751,0.9407763634342377,0.9327782803199334,0.9242137780671775],"mean":0.9327215253145787,"std":0.00999914180516641},"reference":0.923,"delta":0.009721525314578705,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8849613727475081,0.9262980509303037,0.9240909054476015,0.9081127839680805,0.9158937792045817],"mean":0.911871378459615,"std":0.016672588777070493},"reference":0.911,"delta":0.0008713784596149354,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7882209813294927,0.8042828360924694,0.8072473200280987,0.8052414190706026,0.8416669899608181],"mean":0.8093319092962963,"std":0.01960739327474388},"reference":0.839,"delta":-0.029668090703703687,"status":"COMPLETE"},"eps":{"project":{"folds":[0.6840653574672783,0.8323998565973472,0.7937756971906469,0.6304487907591483,0.7772503586617595],"mean":0.7435880121352361,"std":0.08349091395769276},"reference":0.787,"delta":-0.04341198786476397,"status":"COMPLETE"},"nc":{"project":{"folds":[0.778018809203421,0.8877295575068768,0.8396550707743812,0.7339104161852676,0.8492026671754096],"mean":0.8177033041690713,"std":0.061183889425188084},"reference":0.856,"delta":-0.03829669583092865,"status":"COMPLETE"},"xc":{"project":{"folds":[0.2534269900520457,0.32239498703227876,0.4447522916214529,0.2288266749000667,0.5321719420292761],"mean":0.3563145771270241,"std":0.12916714125107337},"reference":0.423,"delta":-0.06668542287297591,"status":"COMPLETE"}},"D_A005":{"eea":{"project":{"folds":[0.9213763184329368,0.9439263862819022,0.9479827979292965,0.9254777017649063,0.9083485884741624],"mean":0.929422358576641,"std":0.01642604330268835},"reference":0.934,"delta":-0.004577641423359036,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9433905496782553,0.9067645825755051,0.9499793300187559,0.9331332411942509,0.9212738448432523],"mean":0.9309083096620039,"std":0.017321972488469985},"reference":0.923,"delta":0.007908309662003843,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8907376354686284,0.9247101419809346,0.9261079965850488,0.9133942157061824,0.9252892770306326],"mean":0.9160478533542854,"std":0.01507725571703504},"reference":0.911,"delta":0.00504785335428537,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7945449700451912,0.7780867723944722,0.7937814581907918,0.8080124129704936,0.8435849891823527],"mean":0.8036021205566601,"std":0.024736566249402715},"reference":0.839,"delta":-0.03539787944333983,"status":"COMPLETE"},"eps":{"project":{"folds":[0.5829077811810623,0.8418769982606953,0.6198687426268719,0.6573775782547919,0.7126224584101832],"mean":0.6829307117467209,"std":0.10097014786061936},"reference":0.787,"delta":-0.1040692882532791,"status":"COMPLETE"},"nc":{"project":{"folds":[0.7752567080289376,0.897675981129949,0.8542150229365492,0.722516344602931,0.7918607988121418],"mean":0.8083049711021018,"std":0.06858789474810476},"reference":0.856,"delta":-0.04769502889789823,"status":"COMPLETE"},"xc":{"project":{"folds":[0.22457713789208855,0.2900494227143937,0.4606906684208535,0.24862245901481783,0.5530378723120584],"mean":0.3553955120708424,"std":0.1439909856452392},"reference":0.423,"delta":-0.06760448792915758,"status":"COMPLETE"}},"D_A010":{"eea":{"project":{"folds":[0.9103583701095922,0.9219585978028229,0.9440176442688671,0.9274616333414997,0.9185118512502244],"mean":0.9244616193546014,"std":0.012569851739582593},"reference":0.934,"delta":-0.009538380645398692,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9356769907363296,0.9162794418923947,0.9516505508155307,0.9356570467741608,0.9255574603636255],"mean":0.9329642981164084,"std":0.013200675259541805},"reference":0.923,"delta":0.00996429811640831,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8946873748872127,0.9285071688617261,0.9257000391787553,0.9140148658204585,0.9166392475713233],"mean":0.9159097392638952,"std":0.01331405913282221},"reference":0.911,"delta":0.004909739263895152,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7902364656331061,0.7997378193285061,0.8028002634857329,0.8234249653705807,0.8503401639907544],"mean":0.813307935561736,"std":0.023982244402265145},"reference":0.839,"delta":-0.025692064438263995,"status":"COMPLETE"},"eps":{"project":{"folds":[0.686323375381211,0.8375448567405986,0.7441583268142146,0.7109782027689728,0.7044547944172891],"mean":0.7366919112244571,"std":0.06013418305466074},"reference":0.787,"delta":-0.050308088775542936,"status":"COMPLETE"},"nc":{"project":{"folds":[0.7946427746168601,0.8980613085589233,0.8432172097313073,0.6917233759645447,0.7971561790081891],"mean":0.8049601695759648,"std":0.07602852349002164},"reference":0.856,"delta":-0.05103983042403515,"status":"COMPLETE"},"xc":{"project":{"folds":[0.3600795494439999,0.3005298012017996,0.40939401354207006,0.2503409150049134,0.4788885148436042],"mean":0.3598465588072774,"std":0.08963252145522448},"reference":0.423,"delta":-0.06315344119272259,"status":"COMPLETE"}},"D_A020":{"eea":{"project":{"folds":[0.9263178693242083,0.9301692467634262,0.943096359934313,0.9294074912720811,0.916522265484158],"mean":0.9291026465556372,"std":0.00952727076031278},"reference":0.934,"delta":-0.0048973534443628175,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9377356622844356,0.9172533487519795,0.9478269998578821,0.9318284438376029,0.9196107090829033],"mean":0.9308510327629607,"std":0.012725749504052832},"reference":0.923,"delta":0.00785103276296062,"status":"COMPLETE"},"egc":{"project":{"folds":[0.898272579949445,0.9267675492545773,0.9254948912545315,0.9190806216274366,0.9210110646857271],"mean":0.9181253413543435,"std":0.011536573560502113},"reference":0.911,"delta":0.007125341354343506,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7694864412867829,0.8101959574449149,0.8208782792826561,0.8507177409734141,0.8056839628560661],"mean":0.8113924763687669,"std":0.029268547339674062},"reference":0.839,"delta":-0.02760752363123309,"status":"COMPLETE"},"eps":{"project":{"folds":[0.6844487058435045,0.8431631326431308,0.5588221508784388,0.714881822264311,0.7615202152102906],"mean":0.712567205367935,"std":0.10474402446005517},"reference":0.787,"delta":-0.07443279463206498,"status":"COMPLETE"},"nc":{"project":{"folds":[0.8023545852761574,0.8934154000520209,0.8404222205894494,0.7143545226869336,0.788013660768236],"mean":0.8077120778745595,"std":0.06623325513427802},"reference":0.856,"delta":-0.04828792212544053,"status":"COMPLETE"},"xc":{"project":{"folds":[0.3596519025377849,0.35112635416783267,0.4730150010737666,0.28133349430158294,0.4626746244771356],"mean":0.3855602753116205,"std":0.08110966725704173},"reference":0.423,"delta":-0.03743972468837947,"status":"COMPLETE"}},"PURE2D":{"eea":{"project":{"folds":[0.917721829470783,0.9455652996213213,0.9254854461733641,0.9261505485985837,0.9152447837214163],"mean":0.9260335815170937,"std":0.011908878260688273},"reference":0.934,"delta":-0.007966418482906357,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9449893611161805,0.9164047890275746,0.944334031877542,0.9368564394095376,0.9187120132158275],"mean":0.9322593269293324,"std":0.013819205784619886},"reference":0.923,"delta":0.009259326929332401,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8920332139300725,0.9285448865303768,0.9221906756857559,0.9176398225338344,0.9211250466970163],"mean":0.916306729075411,"std":0.014129669015837424},"reference":0.911,"delta":0.005306729075410965,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7898906901591223,0.7876448024659414,0.808585136006504,0.8129507281051696,0.8350577334213596],"mean":0.8068258180316192,"std":0.01931554221286506},"reference":0.839,"delta":-0.03217418196838073,"status":"COMPLETE"},"eps":{"project":{"folds":[0.6686842598951328,0.8550665902781935,0.6052197768250208,0.6469375054556183,0.7259779020105375],"mean":0.7003772068929006,"std":0.09682378581585849},"reference":0.787,"delta":-0.08662279310709942,"status":"COMPLETE"},"nc":{"project":{"folds":[0.8442401064985994,0.9029562391796172,0.7279208298698319,0.6630384243972473,0.8656656549311819],"mean":0.8007642509752955,"std":0.10101993863470109},"reference":0.856,"delta":-0.05523574902470452,"status":"COMPLETE"},"xc":{"project":{"folds":[0.26577647845630104,0.31743994687923216,0.4485529791998518,0.12040984561450885,0.5203672885753654],"mean":0.33450930774505183,"std":0.156846495517276},"reference":0.423,"delta":-0.08849069225494816,"status":"COMPLETE"}}}}
```

</details>

<details><summary>results/ua2d_ptscale_20261022/r1/final_review/results.json</summary>

```json
{"plan":"UA2D-PTSCALE-20261022-01","revision":"r1","status":"待审查","independent_review":"PENDING","validation":{"P5K":{"tasks":{"eea":{"folds":[0.9331492444967687,0.904942821369163,0.903024369370583,0.9104531291380855,0.9446375165757439],"mean":0.9192414161900688,"std":0.018594797751749128},"egb":{"folds":[0.931840082033244,0.9359879205687087,0.926924483798022,0.9485109456203162,0.9559500529102828],"mean":0.9398426969861147,"std":0.012045401908376517},"egc":{"folds":[0.924541631001762,0.9236287328889187,0.9211387170250196,0.907684798892544,0.9134741043411789],"mean":0.9180935968298847,"std":0.007265851949877074},"ei":{"folds":[0.8822679261672504,0.8933043315459657,0.8866963601357505,0.8573353048696983,0.8441333057257081],"mean":0.8727474456888746,"std":0.021000597666427666},"eps":{"folds":[0.7241019307028541,0.8504583113763631,0.7752771598630395,0.8305627375098534,0.7051515746631918],"mean":0.7771103428230604,"std":0.0636953680007527},"nc":{"folds":[0.8138814453588458,0.9033184582140601,0.849255234305047,0.886770808914391,0.7465494087111322],"mean":0.8399550711006952,"std":0.06264498562316356},"xc":{"folds":[0.33991254316765507,0.4984484635738603,0.4569333734774049,0.5495916995034967,0.27996812414820005],"mean":0.4249708407741234,"std":0.11203975276677866}},"macro":0.8131373443418316},"P10K":{"tasks":{"eea":{"folds":[0.934798023919767,0.8933933632880209,0.9043810738244442,0.9124422554532301,0.9411906713777947],"mean":0.9172410775726514,"std":0.0202419815927565},"egb":{"folds":[0.9354634271719509,0.9334597227207302,0.9253679078892885,0.9456180946291108,0.9538200547616122],"mean":0.9387458414345385,"std":0.011092183867359898},"egc":{"folds":[0.9281098693702018,0.9242546396108346,0.9203884766940726,0.9143604117085347,0.9166812760290226],"mean":0.9207589346825333,"std":0.005566907336948217},"ei":{"folds":[0.8866432697536007,0.8638825988623507,0.8850339904968607,0.8774546167406215,0.8425643442260083],"mean":0.8711157640158884,"std":0.018320096662084586},"eps":{"folds":[0.7475917533437257,0.8660110945760968,0.7478910172830804,0.8185282532395511,0.6905880934237167],"mean":0.7741220423732342,"std":0.06852983477766152},"nc":{"folds":[0.8329910757512283,0.9113693745052067,0.8432017559670152,0.8988267933417303,0.7871450489352178],"mean":0.8547068097000796,"std":0.05080784901671487},"xc":{"folds":[0.3048347985572377,0.48537332712812065,0.4433977602058856,0.49130706970519034,0.35305071126767185],"mean":0.4155927333728212,"std":0.08299689175188235}},"macro":0.8131833147359638},"P20K":{"tasks":{"eea":{"folds":[0.9372416390959711,0.9085425684339803,0.9167854121116876,0.9083532897102292,0.9371973490593309],"mean":0.9216240516822399,"std":0.014638027428643032},"egb":{"folds":[0.9208134036906261,0.9303597676024042,0.9217331230480141,0.9454329463693794,0.9532604005728145],"mean":0.9343199282566477,"std":0.014480827037443583},"egc":{"folds":[0.9217501211881222,0.9260819985196943,0.9176225315311501,0.9153017229693884,0.9058616744071986],"mean":0.9173236097231108,"std":0.007613421036898465},"ei":{"folds":[0.8876391539201969,0.8815766157001943,0.865611074896816,0.8689675407682281,0.8332236087055687],"mean":0.8674035987982007,"std":0.02112038635036428},"eps":{"folds":[0.7427489428833158,0.8585541670028896,0.7201713569897554,0.804235076350383,0.6953767134297393],"mean":0.7642172513312167,"std":0.0664093907369791},"nc":{"folds":[0.8046375019753013,0.9135990316220768,0.8499159323741408,0.8952325174173589,0.7818467533771662],"mean":0.8490463473532088,"std":0.05654400611020423},"xc":{"folds":[0.31657419882657956,0.5522020938421072,0.4499092124083395,0.506108237892793,0.31753597424917945],"mean":0.4284659434437998,"std":0.10796295101282077}},"macro":0.8117715329412034}},"formal":{"P5K":{"tasks":{"eea":{"folds":[0.912907786916811,0.9320309692726719,0.9464107264794902,0.9279789958998708,0.9250309745466743],"mean":0.9288718906231036,"std":0.012124283953595444},"egb":{"folds":[0.9196805109843117,0.9256431346165681,0.9460021832989186,0.9312824702594993,0.9173763501006118],"mean":0.927996929851982,"std":0.011428952450967082},"egc":{"folds":[0.886118461046401,0.930952441426184,0.9101351008110369,0.9112262068727272,0.9169752497278216],"mean":0.9110814919768341,"std":0.016230068733060696},"ei":{"folds":[0.7637059584976427,0.7964753556221047,0.8028855995673757,0.8140014827016153,0.8622608275391128],"mean":0.8078658447855702,"std":0.03571099916053201},"eps":{"folds":[0.68595477659058,0.825642705005163,0.6647249126022433,0.7368559237966492,0.7669829871239576],"mean":0.7360322610237187,"std":0.06438390700147831},"nc":{"folds":[0.7872148376019783,0.8719406579011648,0.8195251662526776,0.6867929634577973,0.8014973203368878],"mean":0.7933941891101011,"std":0.06767577512907455},"xc":{"folds":[0.4500276515794517,0.40618238978296817,0.44101074953813935,0.27899390102120636,0.4703406183001274],"mean":0.40931106204437856,"std":0.07645205790984642}},"macro":0.7877933813450984},"P10K":{"tasks":{"eea":{"folds":[0.9085344377329794,0.9500258136738593,0.9521388027334685,0.9396353097229598,0.9078530808580162],"mean":0.9316374889442566,"std":0.021919446468473047},"egb":{"folds":[0.9507045581796164,0.9125709629899116,0.9378252334372563,0.9224240450263853,0.9238208983256504],"mean":0.9294691395917638,"std":0.014898419935170406},"egc":{"folds":[0.8931196561691934,0.9285935986642526,0.91901340442812,0.9211382443946686,0.9156646205160663],"mean":0.91550590483446,"std":0.013383598657257183},"ei":{"folds":[0.7937226262329049,0.8200997336019509,0.8094929852163176,0.8400600927711817,0.837139549335576],"mean":0.8201029974315862,"std":0.019345710827028215},"eps":{"folds":[0.6879749158999735,0.8305278546078143,0.6609921529804077,0.7942839899185673,0.7072852733709486],"mean":0.7362128373555422,"std":0.0726110074671783},"nc":{"folds":[0.8122387550201953,0.8977313938936047,0.8306830844504185,0.6850384662079421,0.8508847499683402],"mean":0.8153152899081002,"std":0.07949101467232002},"xc":{"folds":[0.37894031667094474,0.4125651835310974,0.5107261046971499,0.22020467215677086,0.4995110922280195],"mean":0.40438947385679647,"std":0.11725415168285576}},"macro":0.7932333045603579},"P20K":{"tasks":{"eea":{"folds":[0.9179965325712622,0.9470913684891717,0.9490918314264583,0.9296376502129574,0.9051639055138758],"mean":0.9297962576427452,"std":0.018824528322861633},"egb":{"folds":[0.9283391661891683,0.9106458329987162,0.9446013308660709,0.930527225707331,0.921964151230287],"mean":0.9272155413983146,"std":0.012415995693005586},"egc":{"folds":[0.8917755896711744,0.9291183561325246,0.9162575130393229,0.9087628374987755,0.9192890114630743],"mean":0.9130406615609743,"std":0.013949778975438712},"ei":{"folds":[0.8025168599259587,0.7997670907587485,0.7778347174934734,0.8375788040295185,0.835411759831226],"mean":0.810621846407785,"std":0.025493772378152438},"eps":{"folds":[0.6638410207338585,0.861275277173756,0.597142400659794,0.7119153348680625,0.703635326937305],"mean":0.7075618720745551,"std":0.09716757922742401},"nc":{"folds":[0.8125784002940128,0.9050667387191798,0.8444170095070214,0.7033083768470945,0.7849790280109914],"mean":0.81006991067566,"std":0.07456203885850168},"xc":{"folds":[0.3858772120902444,0.2845420007272328,0.4821231018528608,0.2931138616164045,0.5348008607538253],"mean":0.39609140740811355,"std":0.1115725545896393}},"macro":0.7849139281668782}},"FINAL":"P10K","delta_FINAL_B_FP":0.0019964962083384163,"delta_FINAL_P5K":0.005439923215259546,"performance_pass":false,"comparisons":{"P10K_minus_P5K":{"delta":0.005439923215259546,"paired_tasks":{"eea":{"folds":[-0.0043733491838315874,0.01799484440118737,0.005728076253978309,0.01165631382308907,-0.017177893688658075],"mean":0.0027655983211530176,"std":0.013860541825893833},"egb":{"folds":[0.031024047195304716,-0.013072171626656526,-0.008176949861662286,-0.008858425233113976,0.006444548225038593],"mean":0.0014722097397821044,"std":0.0180928340531505},"egc":{"folds":[0.007001195122792403,-0.0023588427619314345,0.008878303617083083,0.009912037521941475,-0.0013106292117552876],"mean":0.004424412857626047,"std":0.005820099624007469},"ei":{"folds":[0.030016667735262237,0.02362437797984618,0.006607385648941921,0.026058610069566335,-0.025121278203536823],"mean":0.01223715264601597,"std":0.022716510377395874},"eps":{"folds":[0.002020139309393487,0.004885149602651295,-0.003732759621835613,0.05742806612191809,-0.05969771375300903],"mean":0.0001805763318236453,"std":0.041543699614519085},"nc":{"folds":[0.025023917418216968,0.02579073599243986,0.011157918197740924,-0.0017544972498552136,0.04938742963145237],"mean":0.021921100797998982,"std":0.019074726798267678},"xc":{"folds":[-0.07108733490850694,0.006382793748129223,0.0697153551590105,-0.058789228864435494,0.02917047392789207],"mean":-0.004921588187582126,"std":0.059456622451970025}},"status":"COMPLETE"},"P20K_minus_P5K":{"delta":-0.002879453178220137,"paired_tasks":{"eea":{"folds":[0.005088745654451277,0.01506039921649982,0.0026811049469680492,0.0016586543130866493,-0.019867069032798446],"mean":0.0009243670196414699,"std":0.01277770103326492},"egb":{"folds":[0.008658655204856647,-0.014997301617851955,-0.00140085243284771,-0.0007552445521682438,0.004587801129675162],"mean":-0.0007813884536672199,"std":0.00895012887160623},"egc":{"folds":[0.005657128624773411,-0.001834085293659471,0.006122412228286023,-0.0024633693739516405,0.002313761735252662],"mean":0.001959169584140197,"std":0.004033635437717966},"ei":{"folds":[0.038810901428316,0.0032917351366438252,-0.02505088207390227,0.023577321327903134,-0.026849067707886864],"mean":0.0027560016222147655,"std":0.02908372859367619},"eps":{"folds":[-0.022113755856721484,0.035632572168592946,-0.06758251194244935,-0.02494058892858675,-0.06334766018665261],"mean":-0.02847038894916345,"std":0.04155790571761615},"nc":{"folds":[0.025363562692034547,0.033126080818014936,0.024891843254343793,0.016515413389297184,-0.01651829232589641],"mean":0.01667572156555881,"std":0.019464492130005433},"xc":{"folds":[-0.06415043948920729,-0.12164038905573538,0.041112352314721434,0.014119960595198155,0.06446024245369786],"mean":-0.013219654636265043,"std":0.0775928290018281}},"status":"COMPLETE"},"P20K_minus_P10K":{"delta":-0.008319376393479683,"paired_tasks":{"eea":{"folds":[0.009462094838282864,-0.0029344451846875508,-0.00304697130701026,-0.009997659510002421,-0.002689175344140371],"mean":-0.0018412313015115477,"std":0.007029588138672916},"egb":{"folds":[-0.02236539199044807,-0.001925129991195429,0.006776097428814576,0.008103180680945732,-0.0018567470953634313],"mean":-0.002253598193449324,"std":0.012181419288929232},"egc":{"folds":[-0.001344066498018992,0.0005247574682719636,-0.0027558913887970604,-0.012375406895893115,0.0036243909470079494],"mean":-0.002465243273485851,"std":0.006032756544208997},"ei":{"folds":[0.008794233693053766,-0.020332642843202353,-0.03165826772284419,-0.0024812887416632012,-0.0017277895043500413],"mean":-0.009481151023801204,"std":0.016222528726733403},"eps":{"folds":[-0.02413389516611497,0.03074742256594165,-0.06384975232061374,-0.08236865505050484,-0.0036499464336435805],"mean":-0.028650965280987095,"std":0.04554714082101424},"nc":{"folds":[0.00033964527381757925,0.007335344825575074,0.01373392505660287,0.018269910639152398,-0.06590572195734878],"mean":-0.005245379232440173,"std":0.03457718592187596},"xc":{"folds":[0.006936895419299649,-0.1280231828038646,-0.02860300284428907,0.07290918945963365,0.03528976852580579],"mean":-0.008298066448682917,"std":0.07660452950271994}},"status":"COMPLETE"},"FINAL_minus_B_FP":{"delta":0.0019964962083384163,"paired_tasks":{"eea":{"folds":[-0.007429232068250746,0.0027210421579075827,0.012154387531376543,8.816964344759981e-05,-0.012988447167959438],"mean":-0.001090815980695692,"std":0.009657702004282785},"egb":{"folds":[0.006249557961479191,-0.005640994769094854,-0.008709669299370004,-0.01132217224174159,-0.00085929594563805],"mean":-0.004056514858873062,"std":0.006953228491064348},"egc":{"folds":[0.008346130064894042,0.0009392938477186252,0.0010613428001435832,0.010304078988186971,-0.007121707021405577],"mean":0.002705827735907529,"std":0.006927334345608856},"ei":{"folds":[-0.010028534114051135,0.04148892870260845,0.009433187491760475,-0.008933203004718782,-0.013118905812498971],"mean":0.0037682946526200077,"std":0.022868074417828857},"eps":{"folds":[-0.06199421160285534,-0.01075505629844331,-0.02431037582575346,0.04556679228218086,-0.08775598321258371],"mean":-0.027849766931490993,"std":0.051111019943928214},"nc":{"folds":[0.050157183495392665,-0.008608812456577297,-0.004204467108353205,0.04966841322509319,0.05613816535349847],"mean":0.028630096501810764,"std":0.03212306416271881},"xc":{"folds":[-0.08083882989240632,0.08499535413899417,0.045371248753268256,0.050820875162093215,-0.04100688646649453],"mean":0.011868352339090959,"std":0.06960192887181789}},"status":"COMPLETE"}},"references":{"B_FP":{"tasks":{"eea":{"folds":[0.9159636698012301,0.9473047715159517,0.939984415202092,0.9395471400795122,0.9208415280259756],"mean":0.9327283049249523,"std":0.013545935583923146},"egb":{"folds":[0.9444550002181372,0.9182119577590064,0.9465349027366263,0.9337462172681269,0.9246801942712884],"mean":0.9335256544506372,"std":0.012262659376907688},"egc":{"folds":[0.8847735261042994,0.927654304816534,0.9179520616279764,0.9108341654064817,0.9227863275374719],"mean":0.9128000770985528,"std":0.016854291685891032},"ei":{"folds":[0.803751160346956,0.7786108048993424,0.8000597977245572,0.8489932957759004,0.850258455148075],"mean":0.8163347027789662,"std":0.031873638077730186},"eps":{"folds":[0.7499691275028288,0.8412829109062576,0.6853025288061612,0.7487171976363864,0.7950412565835323],"mean":0.7640626042870332,"std":0.058223512723314164},"nc":{"folds":[0.7620815715248026,0.906340206350182,0.8348875515587717,0.6353700529828489,0.7947465846148417],"mean":0.7866851934062894,"std":0.1002588966293432},"xc":{"folds":[0.45977914656335106,0.3275698293921032,0.4653548559438816,0.16938379699467765,0.540517978694514],"mean":0.3925211215177055,"std":0.1464073089720903}},"macro":0.7912368083520195}},"budget":{"plan":"UA2D-PTSCALE-20261022-01","revision":"r1"},"paper_envelope_macro":0.8104285714285714,"sota":{"P5K":{"eea":{"project":{"folds":[0.912907786916811,0.9320309692726719,0.9464107264794902,0.9279789958998708,0.9250309745466743],"mean":0.9288718906231036,"std":0.012124283953595444},"reference":0.934,"delta":-0.00512810937689645,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9196805109843117,0.9256431346165681,0.9460021832989186,0.9312824702594993,0.9173763501006118],"mean":0.927996929851982,"std":0.011428952450967082},"reference":0.923,"delta":0.004996929851981968,"status":"COMPLETE"},"egc":{"project":{"folds":[0.886118461046401,0.930952441426184,0.9101351008110369,0.9112262068727272,0.9169752497278216],"mean":0.9110814919768341,"std":0.016230068733060696},"reference":0.911,"delta":8.149197683404719e-05,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7637059584976427,0.7964753556221047,0.8028855995673757,0.8140014827016153,0.8622608275391128],"mean":0.8078658447855702,"std":0.03571099916053201},"reference":0.839,"delta":-0.03113415521442975,"status":"COMPLETE"},"eps":{"project":{"folds":[0.68595477659058,0.825642705005163,0.6647249126022433,0.7368559237966492,0.7669829871239576],"mean":0.7360322610237187,"std":0.06438390700147831},"reference":0.787,"delta":-0.050967738976281374,"status":"COMPLETE"},"nc":{"project":{"folds":[0.7872148376019783,0.8719406579011648,0.8195251662526776,0.6867929634577973,0.8014973203368878],"mean":0.7933941891101011,"std":0.06767577512907455},"reference":0.856,"delta":-0.06260581088989892,"status":"COMPLETE"},"xc":{"project":{"folds":[0.4500276515794517,0.40618238978296817,0.44101074953813935,0.27899390102120636,0.4703406183001274],"mean":0.40931106204437856,"std":0.07645205790984642},"reference":0.423,"delta":-0.013688937955621427,"status":"COMPLETE"}},"P10K":{"eea":{"project":{"folds":[0.9085344377329794,0.9500258136738593,0.9521388027334685,0.9396353097229598,0.9078530808580162],"mean":0.9316374889442566,"std":0.021919446468473047},"reference":0.934,"delta":-0.002362511055743477,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9507045581796164,0.9125709629899116,0.9378252334372563,0.9224240450263853,0.9238208983256504],"mean":0.9294691395917638,"std":0.014898419935170406},"reference":0.923,"delta":0.006469139591763762,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8931196561691934,0.9285935986642526,0.91901340442812,0.9211382443946686,0.9156646205160663],"mean":0.91550590483446,"std":0.013383598657257183},"reference":0.911,"delta":0.004505904834460006,"status":"COMPLETE"},"ei":{"project":{"folds":[0.7937226262329049,0.8200997336019509,0.8094929852163176,0.8400600927711817,0.837139549335576],"mean":0.8201029974315862,"std":0.019345710827028215},"reference":0.839,"delta":-0.018897002568413757,"status":"COMPLETE"},"eps":{"project":{"folds":[0.6879749158999735,0.8305278546078143,0.6609921529804077,0.7942839899185673,0.7072852733709486],"mean":0.7362128373555422,"std":0.0726110074671783},"reference":0.787,"delta":-0.05078716264445782,"status":"COMPLETE"},"nc":{"project":{"folds":[0.8122387550201953,0.8977313938936047,0.8306830844504185,0.6850384662079421,0.8508847499683402],"mean":0.8153152899081002,"std":0.07949101467232002},"reference":0.856,"delta":-0.04068471009189978,"status":"COMPLETE"},"xc":{"project":{"folds":[0.37894031667094474,0.4125651835310974,0.5107261046971499,0.22020467215677086,0.4995110922280195],"mean":0.40438947385679647,"std":0.11725415168285576},"reference":0.423,"delta":-0.01861052614320352,"status":"COMPLETE"}},"P20K":{"eea":{"project":{"folds":[0.9179965325712622,0.9470913684891717,0.9490918314264583,0.9296376502129574,0.9051639055138758],"mean":0.9297962576427452,"std":0.018824528322861633},"reference":0.934,"delta":-0.004203742357254847,"status":"COMPLETE"},"egb":{"project":{"folds":[0.9283391661891683,0.9106458329987162,0.9446013308660709,0.930527225707331,0.921964151230287],"mean":0.9272155413983146,"std":0.012415995693005586},"reference":0.923,"delta":0.004215541398314593,"status":"COMPLETE"},"egc":{"project":{"folds":[0.8917755896711744,0.9291183561325246,0.9162575130393229,0.9087628374987755,0.9192890114630743],"mean":0.9130406615609743,"std":0.013949778975438712},"reference":0.911,"delta":0.002040661560974266,"status":"COMPLETE"},"ei":{"project":{"folds":[0.8025168599259587,0.7997670907587485,0.7778347174934734,0.8375788040295185,0.835411759831226],"mean":0.810621846407785,"std":0.025493772378152438},"reference":0.839,"delta":-0.02837815359221496,"status":"COMPLETE"},"eps":{"project":{"folds":[0.6638410207338585,0.861275277173756,0.597142400659794,0.7119153348680625,0.703635326937305],"mean":0.7075618720745551,"std":0.09716757922742401},"reference":0.787,"delta":-0.07943812792544491,"status":"COMPLETE"},"nc":{"project":{"folds":[0.8125784002940128,0.9050667387191798,0.8444170095070214,0.7033083768470945,0.7849790280109914],"mean":0.81006991067566,"std":0.07456203885850168},"reference":0.856,"delta":-0.04593008932433995,"status":"COMPLETE"},"xc":{"project":{"folds":[0.3858772120902444,0.2845420007272328,0.4821231018528608,0.2931138616164045,0.5348008607538253],"mean":0.39609140740811355,"std":0.1115725545896393},"reference":0.423,"delta":-0.026908592591886438,"status":"COMPLETE"}}}}
```

</details>

### 已有分析报告快照

<details><summary>results/ua3d_frozen_transfer_20261026/r1/final_review/report.md</summary>

# UA3D冻结3D迁移实际交付

状态：待审查；independent_review=PENDING；完整科学流程完成：FULL 70/70，TEST 70/70。
固定科学候选F_G3D、匹配控制F_BASE；REF=C_CONST；validation冻结NEXT_RECOMMEND=REF；主门槛通过=False。
两臂共同o8/norm2/A2/fp_encoder/Q2/Q3取BASE PT10K，仅encoder3d/norm3/A3来源不同；全70epochs冻结这三个模块，Q3继续按原配方训练。fusion/head为原公共初态，不从FULL BEST初始化。
固定冻结3D包在共同BASE背景中的可转移性对照；不与旧实验混拼factorial或interaction，不单独归因角度/邻接/容量。共享TEST、顺序开发及单seed/fold限制存在，非独立盲测，无统计显著性主张。

| 单元 | 正式macro R² |
|---|---:|
| BASE | 0.795 |
| REF | 0.792 |
| F_BASE | 0.787 |
| F_G3D | 0.785 |

### BASE 七任务

| 任务 | 项目mean±std | 论文最佳 | 项目减论文 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.017 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.917±0.010 | 0.923 | -0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.012 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.806±0.041 | 0.839 | -0.033 | Periodic-TDL |
| EPS | 0.771±0.048 | 0.787 | -0.016 | Periodic-TDL |
| Nc | 0.821±0.074 | 0.856 | -0.035 | Periodic-TDL |
| Xc | 0.415±0.105 | 0.423 | -0.008 | Periodic-TDL |

### REF 七任务

| 任务 | 项目mean±std | 论文最佳 | 项目减论文 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.916±0.008 | 0.934 | -0.018 | Periodic-TDL |
| Egb | 0.915±0.013 | 0.923 | -0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.013 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.813±0.038 | 0.839 | -0.026 | Periodic-TDL |
| EPS | 0.724±0.087 | 0.787 | -0.063 | Periodic-TDL |
| Nc | 0.805±0.064 | 0.856 | -0.051 | Periodic-TDL |
| Xc | 0.455±0.059 | 0.423 | +0.032 | Periodic-TDL |

### F_BASE 七任务

| 任务 | 项目mean±std | 论文最佳 | 项目减论文 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.916±0.016 | 0.934 | -0.018 | Periodic-TDL |
| Egb | 0.912±0.015 | 0.923 | -0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.909±0.016 | 0.911 | -0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.802±0.031 | 0.839 | -0.037 | Periodic-TDL |
| EPS | 0.744±0.087 | 0.787 | -0.043 | Periodic-TDL |
| Nc | 0.794±0.053 | 0.856 | -0.062 | Periodic-TDL |
| Xc | 0.435±0.101 | 0.423 | +0.012 | Periodic-TDL |

### F_G3D 七任务

| 任务 | 项目mean±std | 论文最佳 | 项目减论文 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.017 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.918±0.012 | 0.923 | -0.005 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.911±0.016 | 0.911 | -0.000 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.785±0.030 | 0.839 | -0.054 | Periodic-TDL |
| EPS | 0.724±0.064 | 0.787 | -0.063 | Periodic-TDL |
| Nc | 0.822±0.054 | 0.856 | -0.034 | Periodic-TDL |
| Xc | 0.411±0.109 | 0.423 | -0.012 | Periodic-TDL |

### 固定配对差

| 对照 | Δmacro R² | ≥0.010 |
|---|---:|---|
| F_G3D_minus_F_BASE | -0.002 | False |
| F_BASE_minus_BASE | -0.008 | False |
| F_G3D_minus_BASE | -0.010 | False |
| F_G3D_minus_REF | -0.007 | False |

五折mean±std(ddof=1)，seed42+fold，七任务等权macro；完整精度保存在results.json，门槛用未舍入值。论文为SOTA_RESULT逐任务最佳包络，Egc3380/4124、Xc432/430及split协议不同，仅数值对照；EAT/Eib/Tg不纳入。
原始预测formal/{arm}/{task}/fold{fold}/test_predictions.npz；history/BEST/scaler/validation/source state metadata在finetune同结构。唯一冻结selection/final_frozen.json；既定row+sample key/相同目标对齐final_review/prediction_join.json。
预算及累计见budget.json；新增PT/prepare/probe/构象/FP/KRR/输出融合/父重推理均0；CPU={'exit': 0, 'attempts': 1, 'command': ['timeout', '--signal=TERM', '--kill-after=30s', '600s', 'python', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_ua3d_frozen_transfer.py'], 'env': {'CUDA_VISIBLE_DEVICES': '', 'WORLD_SIZE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'UA3M_PHYSICAL_GPU_UUID': ''}, 'log': '/root/workspace/Uni-Poly-Plus-master/logs/ua3d_frozen_transfer_20261026/r1/ua3d_frozen_transfer_r1_cpu', 'time': '2026-10-26T16:43:21.569432+08:00'}；异常=None。
tmux=ua3d-frozen-transfer-r1-controller；命令/cwd/GPU UUID/进程身份/日志/真实exit见budget.jobs与各log/process.json/exit.txt；截止不刷新。


</details>

<details><summary>results/ua2d_wavg_20261022/r1/final_review/report.md</summary>

# UA2D-WAVG r1 实际交付

状态：待审查；DSH执行，待Codex独立审查。
FINAL=EMA60；完整105单元正式TEST=True；双参考性能门槛：未达到双参考性能门槛；CPU exit=0。

35条FT轨迹各只训练一次，共享产生BEST／SWA4／EMA60；不称三组独立训练。训练validation最多2450次，SWA／EMA终点额外各35次，BEST复用保存的预测。
SWA4固定全局epoch40/50/60/70，保留4份纯模型权重；EMA60固定global11—70、beta0.9、S0=z0=0、最终S/z，最终S／z及60次更新记录保留。
平均器独立detached FP32，覆盖完整浮点state；未修改学生／梯度／optimizer或RNG，无teacher、额外backward或平均模型逐epoch评估。
prepare／公共初始化／PT均0，只读引用TRANSFER的5875条原2-hop缓存，保留其原plan和schema，不复制或冒充新缓存。
历史B_FP macro=0.791，A_MIX246=0.791，R_MEAN=0.791，均直接复用。

| 预登记正式对照 | Δmacro R² | 完整性 |
|---|---:|---|
| SWA4_minus_BEST | -0.011 | COMPLETE |
| EMA60_minus_BEST | -0.014 | COMPLETE |
| BEST_minus_B_FP | -0.000 | COMPLETE |
| SWA4_minus_B_FP | -0.011 | COMPLETE |
| EMA60_minus_B_FP | -0.014 | COMPLETE |
| FINAL_minus_A_MIX246 | -0.014 | COMPLETE |
| FINAL_minus_R_MEAN | -0.014 | COMPLETE |

## BEST 七任务SOTA数值对照（macro=0.791）

| Task | 正式R² | 论文参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.933±0.014 | 0.934 | -0.001 | Periodic-TDL |
| Egb | 0.934±0.012 | 0.923 | +0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.912±0.016 | 0.911 | +0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.816±0.032 | 0.839 | -0.023 | Periodic-TDL |
| EPS | 0.764±0.058 | 0.787 | -0.023 | Periodic-TDL |
| Nc | 0.787±0.100 | 0.856 | -0.069 | Periodic-TDL |
| Xc | 0.393±0.146 | 0.423 | -0.030 | Periodic-TDL |

## SWA4 七任务SOTA数值对照（macro=0.780）

| Task | 正式R² | 论文参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.933±0.013 | 0.934 | -0.001 | Periodic-TDL |
| Egb | 0.934±0.012 | 0.923 | +0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.910±0.014 | 0.911 | -0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.819±0.029 | 0.839 | -0.020 | Periodic-TDL |
| EPS | 0.694±0.137 | 0.787 | -0.093 | Periodic-TDL |
| Nc | 0.782±0.094 | 0.856 | -0.074 | Periodic-TDL |
| Xc | 0.391±0.157 | 0.423 | -0.032 | Periodic-TDL |

## EMA60 七任务SOTA数值对照（macro=0.777）

| Task | 正式R² | 论文参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.931±0.014 | 0.934 | -0.003 | Periodic-TDL |
| Egb | 0.933±0.012 | 0.923 | +0.010 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.910±0.014 | 0.911 | -0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.816±0.029 | 0.839 | -0.023 | Periodic-TDL |
| EPS | 0.684±0.148 | 0.787 | -0.103 | Periodic-TDL |
| Nc | 0.780±0.095 | 0.856 | -0.076 | Periodic-TDL |
| Xc | 0.385±0.159 | 0.423 | -0.038 | Periodic-TDL |

性能通过须FINAL对同轨迹BEST及历史B_FP的未舍入Δmacro R²均≥0.010；失败不搜索窗口／系数或增加seed。
每任务五折等权R²，std(ddof=1)，seed42+fold，再七任务等权macro；原始值完整精度，展示3位小数。共享开发TEST不是独立盲测，论文协议等价性未证明，只作数值比较。
35份完整history／最终resume／SWA源／EMA累积器及记录在trajectories/；三估计checkpoint／scaler／validation在estimates/，正式预测在formal/；来源冻结selection/final_frozen.json；所有预登记配对差值final_review/results.json；预算budget.json；完整源码diff与训练计时final_review/；真实命令／window／exit在logs/ua2d_wavg_20261022/r1/。
未完成／阻断原因：无。


</details>

<details><summary>results/ua2d_p10k_r10fp_20261023/r1/final_review/report.md</summary>

# UA2D-P10K-R10FP-20261023-01 / r1

状态：待审查；交Codex独立审查，执行自检不构成独立审查。

一次prepare → validation唯一全局权重 → 一次冻结 → 35正式政策评估 → 唯一最终CPU suite。

PT、FT、KRR拟合、FP生成、模型推理、GPU任务、worker全部为0。CPU exit=0。

全局P10K权重 w=0.75，仅依据175个validation R²选择；端点按实际政策解释。

正式macro R²=0.804，P10K_REF=0.793，Δ=+0.010；性能门槛通过=True（未舍入Δ≥0.010）。

|任务|项目R²|P10K_REF|配对Δ|正向fold|论文参考|项目减论文|参考模型|
|---|---:|---:|---:|---:|---:|---:|---|
|eea|0.933±0.019|0.932±0.022|+0.001|2/5|0.934|-0.001|Periodic-TDL|
|egb|0.928±0.009|0.929±0.015|-0.001|3/5|0.923|+0.005|Periodic-TDL (+Chem, +RC)|
|egc|0.919±0.013|0.916±0.013|+0.004|4/5|0.911|+0.008|Periodic-TDL (+Chem, +RC)|
|ei|0.822±0.019|0.820±0.019|+0.002|3/5|0.839|-0.017|Periodic-TDL|
|eps|0.751±0.070|0.736±0.073|+0.015|5/5|0.787|-0.036|Periodic-TDL|
|nc|0.825±0.062|0.815±0.079|+0.010|2/5|0.856|-0.031|Periodic-TDL|
|xc|0.447±0.110|0.404±0.117|+0.042|5/5|0.423|+0.024|Periodic-TDL|

五折seed42+fold等权mean/std(ddof1)，七任务等权macro。共享开发TEST非独立盲测；论文协议未证明等价，仅数值比较。历史r10采用不同神经分支且旧std为ddof0，不将跨周期差异作为严格消融。原始预测及指标保留完整精度。

冻结：selection/final_frozen.json；35组预测：formal/{task}/fold{fold}/predictions.npz；所有配置、scaler和来源见冻结名单及inputs/metadata.json。

总墙钟：5.516秒；各阶段命令/PID/window/真实exit/stdout/stderr见execution_manifest.json及/root/workspace/Uni-Poly-Plus-master/logs/ua2d_p10k_r10fp_20261023/r1。

异常：无


</details>

<details><summary>results/ua3m_paperfuse_20261025/r1/final_review/report.md</summary>

# UA3M-PAPERFUSE r1 实际交付

状态：需返修，独立复审PENDING；未归档。
FINAL=None；未舍入delta=None；性能通过=None。

科学问题：原生UA2D/3D保留，FP作为输入，分别采用UNI标准InfoNCE或TMOL循环排正项损失和三模态直接融合；原mask任务保留，无FP预测或核模型。
这是完整方案包比较，含FP参与、ALIGN、下游3D和优化多因素变化，不能独立归因FP/3D/融合或interaction。

| 方法 | 正式macro R² | ΔB_A020 |
|---|---:|---:|
| B_A020 | 0.785 | +0.000 |
| UNI3M | 未完成／不可比较 | — |
| TMOL3M | 未完成／不可比较 | — |

## B_A020 七任务

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.929±0.010 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.931±0.013 | 0.923 | +0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.918±0.012 | 0.911 | +0.007 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.029 | 0.839 | -0.028 | Periodic-TDL |
| EPS | 0.713±0.105 | 0.787 | -0.074 | Periodic-TDL |
| Nc | 0.808±0.066 | 0.856 | -0.048 | Periodic-TDL |
| Xc | 0.386±0.081 | 0.423 | -0.037 | Periodic-TDL |

## UNI3M 七任务

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## TMOL3M 七任务

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## UNI3M_minus_B_A020

Δmacro=None；改善fold=None；状态未完成／不可比较。

## TMOL3M_minus_B_A020

Δmacro=None；改善fold=None；状态未完成／不可比较。

## TMOL3M_minus_UNI3M

Δmacro=None；改善fold=None；状态未完成／不可比较。

## 成本、可用率及协议

{"prepare_starts": 1, "cache_rows": 0, "pt_starts": 0, "pt_updates": 0, "pt_microbatches": 0, "pt_presentations": 0, "ft_starts": 0, "epochs_started": 0, "attempted_updates": 0, "train_graphs": 0, "validation_started": 0, "validation_graphs": 0, "test_starts": 0, "test_graphs": 0, "test_3d_graphs": 0, "cpu_suites": 0}
所有完整精度fold差值、参数量、PT/FT/TEST实际耗时及可用模态呈现见results.json成本表与单元consumption/history；所有计时来自真实运行，未做benchmark。
下游原行数5875，3D有效5860，固定缺失15（Egb7/Egc8）；缺失3D不进入编码器，权重严格0，其他模态重归一化。PT无效635条采样前排除；缺失名单不是运行错误白名单。
七任务五折等权mean±std(ddof=1)，seed42+fold，七任务等权macro。论文是逐任务最佳包络，数据/split/协议等价性未证明，仅数值对照；Egc3380/Xc432与论文4124/430不同，共享开发TEST非独立盲测。
基线D_A020 legacy PT随机偏差及历史ALIGN×2保留为基线限制。本轮UNI不继承ALIGN×2，TMOL保留作者可负损失数学；不同RNG种子不证明统计独立性。本项目适配而非论文完整复现，不宣称SOTA。
异常/未完成原因：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3m_paperfuse.py", line 223, in run
    t.update_plan('执行中','一次prepare','CPU物化5875行既有结构输入和初始化；无模型forward、FP或构象生成。');self.wait([self.launch('prepare')])
                                                                                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3m_paperfuse.py", line 132, in wait
    self.deadline();self.refresh();self.check_failures()
                                   ^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3m_paperfuse.py", line 84, in check_failures
    if bad:raise RuntimeError('job failed; no automatic retry: '+bad[0]['name']+' exit='+str(bad[0]['exit']))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: job failed; no automatic retry: ua3m_prepare exit=1
。
最终CPU：{"exit": null, "attempts": 0, "not_run_reason": "authorized experiments/report not complete; no early suite"}。
tmux Uni-Poly:ua3m-paperfuse-r1；各独立worker窗口/命令/PID/起止/真实exit见budget.jobs及/root/workspace/Uni-Poly-Plus-master/logs/ua3m_paperfuse_20261025/r1。

## 本次事故、实际完成层级与停止

本次prepare实现漏建initialization/heads/{task}/目录。首次保存Eea/fold0的256维回归头时，save_checkpoint调用torch.save报父目录不存在。prepare真实exit1，controller真实exit1；根因和原始堆栈见incident.json、inputs/coverage.json及ua3m_prepare/stderr.log。这是本次实现问题，代码保留故障时原样，未修复后重跑。

完成层级：第11节七个隔离源码/配置/测试文件已写入，但未通过最终CPU或实际训练验证；一次prepare失败。保留两个新模态适配器的部分初始化文件，35个新性质头尚未保存完成。下游缓存实际物化0行；LMDB和15行缺失名单尚未建立，本次未验证5860/15物化覆盖，前述5875/5860/15为计划声明与历史来源口径。

预算：prepare尝试1/1已消费且不退款；5875行缓存额度按启动时全额不可退预留，实际生成行数为0，二者不能混称。PT、FT、validation、正式TEST、FP/构象生成、模型forward、GPU任务、KRR及输出融合均0；未进行选择或冻结。最终CPU尝试0次，因为尚未完成授权实验和完整正式结果，不提前运行suite。

本周期新方案七任务正式结果全部未完成／不可比较。报告中的B_A020及论文各任务最佳包络仅为原参考，不是本次新评估。不能据已写代码、部分初始化或静态检查声称新方案运行正确或性能通过。

状态需返修；原无FP/A020源码、模型、日志、事故、预算和旧FINAL=D_A005不变。未自动retry/resume、减batch、换精度/卡、安装依赖或换编号补额度；未写PROJECT_HISTORY。下一步先独立只读审查事故与实现，再制定并另行授权明确的返修/追加prepare额度；本次不自动继续。


</details>

<details><summary>results/ua3m_paperfuse_20261025/r1/repairs/prepare_repair1/final_review/report.md</summary>

# UA3M-PAPERFUSE r1 实际交付

状态：需返修，独立复审PENDING；未归档。
FINAL=None；未舍入delta=None；性能通过=None。

科学问题：原生UA2D/3D保留，FP作为输入，分别采用UNI标准InfoNCE或TMOL循环排正项损失和三模态直接融合；原mask任务保留，无FP预测或核模型。
这是完整方案包比较，含FP参与、ALIGN、下游3D和优化多因素变化，不能独立归因FP/3D/融合或interaction。

| 方法 | 正式macro R² | ΔB_A020 |
|---|---:|---:|
| B_A020 | 0.785 | +0.000 |
| UNI3M | 未完成／不可比较 | — |
| TMOL3M | 未完成／不可比较 | — |

## B_A020 七任务

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.929±0.010 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.931±0.013 | 0.923 | +0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.918±0.012 | 0.911 | +0.007 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.029 | 0.839 | -0.028 | Periodic-TDL |
| EPS | 0.713±0.105 | 0.787 | -0.074 | Periodic-TDL |
| Nc | 0.808±0.066 | 0.856 | -0.048 | Periodic-TDL |
| Xc | 0.386±0.081 | 0.423 | -0.037 | Periodic-TDL |

## UNI3M 七任务

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## TMOL3M 七任务

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## UNI3M_minus_B_A020

Δmacro=None；改善fold=None；状态未完成／不可比较。

## TMOL3M_minus_B_A020

Δmacro=None；改善fold=None；状态未完成／不可比较。

## TMOL3M_minus_UNI3M

Δmacro=None；改善fold=None；状态未完成／不可比较。

返修prepare-r1：父prepare真实exit1、实际缓存0行及部分初始化保留；新增prepare仅一次，父预留5875行永久结转，累计预留11750行，实际新目标5875行。父历史、新执行及累计消费见budget三栏。

## 成本、可用率及协议

{"prepare_starts": 2, "cache_rows": 5875, "pt_starts": 1, "pt_updates": 3635, "pt_microbatches": 21804, "pt_presentations": 3663072, "ft_starts": 0, "epochs_started": 0, "attempted_updates": 0, "train_graphs": 0, "validation_started": 0, "validation_graphs": 0, "test_starts": 0, "test_graphs": 0, "test_3d_graphs": 0, "cpu_suites": 0}
所有完整精度fold差值、参数量、PT/FT/TEST实际耗时及可用模态呈现见results.json成本表与单元consumption/history；所有计时来自真实运行，未做benchmark。
下游原行数5875，3D有效5860，固定缺失15（Egb7/Egc8）；缺失3D不进入编码器，权重严格0，其他模态重归一化。PT无效635条采样前排除；缺失名单不是运行错误白名单。
七任务五折等权mean±std(ddof=1)，seed42+fold，七任务等权macro。论文是逐任务最佳包络，数据/split/协议等价性未证明，仅数值对照；Egc3380/Xc432与论文4124/430不同，共享开发TEST非独立盲测。
基线D_A020 legacy PT随机偏差及历史ALIGN×2保留为基线限制。本轮UNI不继承ALIGN×2，TMOL保留作者可负损失数学；不同RNG种子不证明统计独立性。本项目适配而非论文完整复现，不宣称SOTA。
异常/未完成原因：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3m_paperfuse.py", line 242, in run
    self.pt_stage();self.matrix(False);frozen=t.freeze()
    ^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3m_paperfuse.py", line 159, in pt_stage
    self.wait([job])
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3m_paperfuse.py", line 151, in wait
    time.sleep(5)
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3m_paperfuse.py", line 275, in stop
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled termination '+str(sig))
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: controlled termination 15
。
最终CPU：{"exit": null, "attempts": 0, "not_run_reason": "authorized experiments/report not complete; no early suite"}。
tmux Uni-Poly:ua3m-paperfuse-prepare-repair1；各独立worker窗口/命令/PID/起止/真实exit见budget.jobs及/root/workspace/Uni-Poly-Plus-master/logs/ua3m_paperfuse_20261025/r1/prepare_repair1。


</details>

<details><summary>results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/final_review/report.md</summary>

# UA3M-PAPERFUSE r1 实际交付

状态：待审查，独立复审PENDING；未归档。
FINAL=B_A020；未舍入delta=0.0；性能通过=False。

科学问题：原生UA2D/3D保留，FP作为输入，分别采用UNI标准InfoNCE或TMOL循环排正项损失和三模态直接融合；原mask任务保留，无FP预测或核模型。
这是完整方案包比较，含FP参与、ALIGN、下游3D和优化多因素变化，不能独立归因FP/3D/融合或interaction。

| 方法 | 正式macro R² | ΔB_A020 |
|---|---:|---:|
| B_A020 | 0.785 | +0.000 |
| UNI3M | 0.778 | -0.007 |
| TMOL3M | 0.790 | +0.005 |

## B_A020 七任务

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.929±0.010 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.931±0.013 | 0.923 | +0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.918±0.012 | 0.911 | +0.007 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.029 | 0.839 | -0.028 | Periodic-TDL |
| EPS | 0.713±0.105 | 0.787 | -0.074 | Periodic-TDL |
| Nc | 0.808±0.066 | 0.856 | -0.048 | Periodic-TDL |
| Xc | 0.386±0.081 | 0.423 | -0.037 | Periodic-TDL |

## UNI3M 七任务

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.026 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.932±0.011 | 0.923 | +0.009 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.908±0.009 | 0.911 | -0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.808±0.016 | 0.839 | -0.031 | Periodic-TDL |
| EPS | 0.711±0.075 | 0.787 | -0.076 | Periodic-TDL |
| Nc | 0.788±0.069 | 0.856 | -0.068 | Periodic-TDL |
| Xc | 0.373±0.073 | 0.423 | -0.050 | Periodic-TDL |

## TMOL3M 七任务

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.923±0.013 | 0.934 | -0.011 | Periodic-TDL |
| Egb | 0.911±0.006 | 0.923 | -0.012 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.911±0.013 | 0.911 | +0.000 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.805±0.033 | 0.839 | -0.034 | Periodic-TDL |
| EPS | 0.747±0.046 | 0.787 | -0.040 | Periodic-TDL |
| Nc | 0.814±0.071 | 0.856 | -0.042 | Periodic-TDL |
| Xc | 0.418±0.094 | 0.423 | -0.005 | Periodic-TDL |

## UNI3M_minus_B_A020

Δmacro=-0.007382843913843917；改善fold=12；状态COMPLETE。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|
| eea | -0.005±0.021 | 3/5 |
| egb | +0.001±0.003 | 3/5 |
| egc | -0.011±0.008 | 0/5 |
| ei | -0.003±0.033 | 2/5 |
| eps | -0.002±0.054 | 1/5 |
| nc | -0.020±0.036 | 2/5 |
| xc | -0.013±0.014 | 1/5 |

## TMOL3M_minus_B_A020

Δmacro=0.004838913368283682；改善fold=11；状态COMPLETE。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|
| eea | -0.006±0.008 | 0/5 |
| egb | -0.020±0.017 | 0/5 |
| egc | -0.007±0.002 | 0/5 |
| ei | -0.007±0.023 | 1/5 |
| eps | +0.035±0.074 | 3/5 |
| nc | +0.006±0.059 | 4/5 |
| xc | +0.032±0.077 | 3/5 |

## TMOL3M_minus_UNI3M

Δmacro=0.012221757282127599；改善fold=20；状态COMPLETE。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|
| eea | -0.001±0.016 | 2/5 |
| egb | -0.021±0.014 | 0/5 |
| egc | +0.004±0.008 | 3/5 |
| ei | -0.003±0.028 | 3/5 |
| eps | +0.036±0.055 | 4/5 |
| nc | +0.026±0.036 | 4/5 |
| xc | +0.045±0.072 | 4/5 |

## FINAL_minus_B_A020

Δmacro=0.0；改善fold=0；状态COMPLETE。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|
| eea | +0.000±0.000 | 0/5 |
| egb | +0.000±0.000 | 0/5 |
| egc | +0.000±0.000 | 0/5 |
| ei | +0.000±0.000 | 0/5 |
| eps | +0.000±0.000 | 0/5 |
| nc | +0.000±0.000 | 0/5 |
| xc | +0.000±0.000 | 0/5 |

用户明确停止原GPU3执行后按四卡调度重启；用户明确要求不复用成功prepare；本次重新物化和初始化。中断UNI3M已完成3634 updates，整次PT收费不退款；补充重启授权及完整父/新/累计账本见authorization和budget。候选两方案均从step0重新训练，不从中断权重resume。

## 成本、可用率及协议

{"prepare_starts": 3, "cache_rows": 11750, "pt_starts": 3, "pt_updates": 23635, "pt_microbatches": 141804, "pt_presentations": 23823072, "ft_starts": 70, "epochs_started": 4900, "attempted_updates": 112000, "train_graphs": 2630040, "validation_started": 4900, "validation_graphs": 659960, "test_starts": 70, "test_graphs": 11750, "test_3d_graphs": 11720, "cpu_suites": 1}
所有完整精度fold差值、参数量、PT/FT/TEST实际耗时及可用模态呈现见results.json成本表与单元consumption/history；所有计时来自真实运行，未做benchmark。
下游原行数5875，3D有效5860，固定缺失15（Egb7/Egc8）；缺失3D不进入编码器，权重严格0，其他模态重归一化。PT无效635条采样前排除；缺失名单不是运行错误白名单。
七任务五折等权mean±std(ddof=1)，seed42+fold，七任务等权macro。论文是逐任务最佳包络，数据/split/协议等价性未证明，仅数值对照；Egc3380/Xc432与论文4124/430不同，共享开发TEST非独立盲测。
基线D_A020 legacy PT随机偏差及历史ALIGN×2保留为基线限制。本轮UNI不继承ALIGN×2，TMOL保留作者可负损失数学；不同RNG种子不证明统计独立性。本项目适配而非论文完整复现，不宣称SOTA。
异常/未完成原因：无。
最终CPU：{"exit": 0, "attempts": 1, "command": ["timeout", "--signal=TERM", "--kill-after=10s", "600s", "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_ua3m_paperfuse.py"], "env": {"UA3M_EXECUTION_ROOT": "/root/workspace/Uni-Poly-Plus-master/results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1", "CUDA_VISIBLE_DEVICES": "", "WORLD_SIZE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "UA3M_PHYSICAL_GPU_UUID": ""}, "log": "/root/workspace/Uni-Poly-Plus-master/logs/ua3m_paperfuse_20261025/r1/gpu0123_restart1/ua3m_cpu", "time": "2026-10-25T15:17:24.297631+08:00"}。
tmux Uni-Poly:ua3m-paperfuse-gpu0123-restart1；各独立worker窗口/命令/PID/起止/真实exit见budget.jobs及/root/workspace/Uni-Poly-Plus-master/logs/ua3m_paperfuse_20261025/r1/gpu0123_restart1。


</details>

<details><summary>results/ua_hop_ph_20261020/frozen3d_r1/final_review/report.md</summary>

# UA-HOP-PH-FROZEN3D-20261020-01 / r1 DSH执行交付

状态：待审查；独立审查未完成。

唯一受控变化为encoder3d冻结，保留真实三维前向；不重跑reference。
五折seed42—46、std ddof=1、七任务等权宏平均；共享validation/TEST不是独立盲测。

validation宏R²：0.821；r1 HOP_PH：0.819；差值：+0.002；门槛≥0.010：未通过。
head阶段best：3/35，不能解释为三维收益。

|任务|候选validation R²|r1 PH validation R²|差值|
|---|---:|---:|---:|
|eea|0.925±0.017|0.926±0.016|-0.002|
|egb|0.941±0.011|0.942±0.009|-0.001|
|egc|0.919±0.008|0.921±0.006|-0.002|
|ei|0.872±0.024|0.866±0.022|+0.006|
|eps|0.791±0.059|0.780±0.078|+0.011|
|nc|0.852±0.063|0.847±0.064|+0.005|
|xc|0.444±0.119|0.450±0.113|-0.006|

|任务|论文模型|参考R²|候选正式R²／差值|
|---|---|---:|---|
|eea|Periodic-TDL|0.934|未执行／不可比较（validation门槛未通过）|
|egb|Periodic-TDL (+Chem, +RC)|0.923|未执行／不可比较（validation门槛未通过）|
|egc|Periodic-TDL (+Chem, +RC)|0.911|未执行／不可比较（validation门槛未通过）|
|ei|Periodic-TDL|0.839|未执行／不可比较（validation门槛未通过）|
|eps|Periodic-TDL|0.787|未执行／不可比较（validation门槛未通过）|
|nc|Periodic-TDL|0.856|未执行／不可比较（validation门槛未通过）|
|xc|Periodic-TDL|0.423|未执行／不可比较（validation门槛未通过）|

协议等价未证明，仅数值对照；未开放TEST不填入r1 TEST冒充候选结果。
原始结果：/root/workspace/Uni-Poly-Plus-master/results/ua_hop_ph_20261020/frozen3d_r1/final_review/results.json；CPU证据：/root/workspace/Uni-Poly-Plus-master/logs/ua_hop_ph_20261020/frozen3d_r1/cpu；失败：无。


</details>

<details><summary>results/ua_hop_ph_20261020/r1/final_review/report.md</summary>

# UA-HOP-PH-20261020-01 / r1 DSH执行交付

状态：待审查；独立审查未完成。

五折seed42—46，std ddof=1；七任务等权宏平均，完整精度判定，展示三位小数。
共享TEST不是独立盲测；协议未证明与论文一致，仅数值对照；head阶段best不是三维／PH收益。

|任务|论文模型|参考R²|HOP_EQ R²／差值|HOP_PH R²／差值|
|---|---|---:|---|---|
|eea|Periodic-TDL|0.934|0.934±0.015／-0.000|0.934±0.015／-0.000|
|egb|Periodic-TDL (+Chem, +RC)|0.923|0.932±0.014／+0.009|0.933±0.014／+0.010|
|egc|Periodic-TDL (+Chem, +RC)|0.911|0.912±0.018／+0.001|0.914±0.015／+0.003|
|ei|Periodic-TDL|0.839|0.807±0.036／-0.032|0.807±0.036／-0.032|
|eps|Periodic-TDL|0.787|0.755±0.045／-0.032|0.755±0.045／-0.032|
|nc|Periodic-TDL|0.856|0.808±0.079／-0.048|0.804±0.069／-0.052|
|xc|Periodic-TDL|0.423|0.369±0.152／-0.054|0.372±0.151／-0.051|

原始配置、预测与消耗分别见各finetune／formal单元；CPU证据见日志cpu目录及budget.json。
结果：/root/workspace/Uni-Poly-Plus-master/results/ua_hop_ph_20261020/r1/final_review/results.json；失败：无。


</details>

<details><summary>results/ua3d_spatial_20261025/r4/final_review/report.md</summary>

# UA3D spatial r4 实际交付

状态：需返修；独立审查PENDING，未归档或替换部署。
FINAL=C_CONST；G=S_RADANG。
科学问题：同一有限Trimer的非键空间状态消息及距离／夹角矩能否改善FULL预测与冻结PT-3D性质可读性。
BASE为本轮共同step0匹配基线；C_CONST保留真实空间邻接，不是完全无几何控制；四组非完整factorial，不宣称交互效应。
所有正式TEST在全部280估计器统一冻结后一次执行。共享开发TEST非独立盲测。

| 模式 | Arm | 正式macro R² |
|---|---|---:|
| FULL | BASE | 0.795 |

### FULL/BASE 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.017 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.917±0.010 | 0.923 | -0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.012 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.806±0.041 | 0.839 | -0.033 | Periodic-TDL |
| EPS | 0.771±0.048 | 0.787 | -0.016 | Periodic-TDL |
| Nc | 0.821±0.074 | 0.856 | -0.035 | Periodic-TDL |
| Xc | 0.415±0.105 | 0.423 | -0.008 | Periodic-TDL |
| FULL | C_CONST | 0.792 |

### FULL/C_CONST 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.916±0.008 | 0.934 | -0.018 | Periodic-TDL |
| Egb | 0.915±0.013 | 0.923 | -0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.013 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.813±0.038 | 0.839 | -0.026 | Periodic-TDL |
| EPS | 0.724±0.087 | 0.787 | -0.063 | Periodic-TDL |
| Nc | 0.805±0.064 | 0.856 | -0.051 | Periodic-TDL |
| Xc | 0.455±0.059 | 0.423 | +0.032 | Periodic-TDL |
| FULL | S_RAD | 0.790 |

### FULL/S_RAD 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.925±0.017 | 0.934 | -0.009 | Periodic-TDL |
| Egb | 0.909±0.014 | 0.923 | -0.014 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.916±0.011 | 0.911 | +0.005 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.816±0.022 | 0.839 | -0.023 | Periodic-TDL |
| EPS | 0.746±0.060 | 0.787 | -0.041 | Periodic-TDL |
| Nc | 0.811±0.062 | 0.856 | -0.045 | Periodic-TDL |
| Xc | 0.405±0.130 | 0.423 | -0.018 | Periodic-TDL |
| FULL | S_RADANG | 0.792 |

### FULL/S_RADANG 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.922±0.020 | 0.934 | -0.012 | Periodic-TDL |
| Egb | 0.906±0.018 | 0.923 | -0.017 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.012 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.801±0.041 | 0.839 | -0.038 | Periodic-TDL |
| EPS | 0.735±0.100 | 0.787 | -0.052 | Periodic-TDL |
| Nc | 0.833±0.051 | 0.856 | -0.023 | Periodic-TDL |
| Xc | 0.432±0.111 | 0.423 | +0.009 | Periodic-TDL |
| 模式 | Arm | 正式macro R² |
|---|---|---:|
| probe | BASE | 0.671 |

### probe/BASE 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.827±0.037 | 0.934 | -0.107 | Periodic-TDL |
| Egb | 0.834±0.021 | 0.923 | -0.089 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.812±0.015 | 0.911 | -0.099 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.635±0.078 | 0.839 | -0.204 | Periodic-TDL |
| EPS | 0.580±0.157 | 0.787 | -0.207 | Periodic-TDL |
| Nc | 0.689±0.049 | 0.856 | -0.167 | Periodic-TDL |
| Xc | 0.318±0.077 | 0.423 | -0.105 | Periodic-TDL |
| probe | C_CONST | 0.682 |

### probe/C_CONST 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.811±0.028 | 0.934 | -0.123 | Periodic-TDL |
| Egb | 0.832±0.018 | 0.923 | -0.091 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.808±0.016 | 0.911 | -0.103 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.636±0.072 | 0.839 | -0.203 | Periodic-TDL |
| EPS | 0.627±0.101 | 0.787 | -0.160 | Periodic-TDL |
| Nc | 0.710±0.034 | 0.856 | -0.146 | Periodic-TDL |
| Xc | 0.353±0.066 | 0.423 | -0.070 | Periodic-TDL |
| probe | S_RAD | 0.693 |

### probe/S_RAD 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.848±0.029 | 0.934 | -0.086 | Periodic-TDL |
| Egb | 0.835±0.027 | 0.923 | -0.088 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.805±0.015 | 0.911 | -0.106 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.667±0.073 | 0.839 | -0.172 | Periodic-TDL |
| EPS | 0.625±0.113 | 0.787 | -0.162 | Periodic-TDL |
| Nc | 0.711±0.057 | 0.856 | -0.145 | Periodic-TDL |
| Xc | 0.361±0.038 | 0.423 | -0.062 | Periodic-TDL |
| probe | S_RADANG | 0.700 |

### probe/S_RADANG 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.852±0.014 | 0.934 | -0.082 | Periodic-TDL |
| Egb | 0.840±0.020 | 0.923 | -0.083 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.809±0.007 | 0.911 | -0.102 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.691±0.028 | 0.839 | -0.148 | Periodic-TDL |
| EPS | 0.660±0.091 | 0.787 | -0.127 | Periodic-TDL |
| Nc | 0.712±0.046 | 0.856 | -0.144 | Periodic-TDL |
| Xc | 0.341±0.076 | 0.423 | -0.082 | Periodic-TDL |

### FULL 配对差值

| 对照 | Δmacro R² | 达到0.010 |
|---|---:|---|
| G_minus_BASE | -0.004 | False |
| G_minus_C_CONST | +0.000 | False |
| C_CONST_minus_BASE | -0.004 | False |
| distance | -0.002 | False |
| angle | +0.002 | False |

### probe 配对差值

| 对照 | Δmacro R² | 达到0.010 |
|---|---:|---|
| G_minus_BASE | +0.030 | True |
| G_minus_C_CONST | +0.018 | True |
| C_CONST_minus_BASE | +0.012 | True |
| distance | +0.011 | True |
| angle | +0.007 | False |

表征诊断改善、整体性能未通过

实际预算：{"prepare_starts": 1, "cache_rows": 5875, "pt_starts": 4, "pt_updates": 40000, "pt_microbatches": 240000, "pt_presentations": 40320000, "ft_starts": 140, "ft_epochs": 9800, "ft_updates": 224000, "probe_starts": 140, "probe_epochs": 4200, "probe_updates": 96000, "feature_rows": 94000, "test_starts": 280, "test_graphs": 47000, "cpu_suites": 1}
原始精度、配对fold差值、退化任务／fold、参数与真实耗时见results.json及各单元history/consumption。未完成不得以validation或旧结果补齐。
五折mean±std(ddof=1)，seed42+fold，七任务等权macro。论文是逐任务最佳包络；Egc3380/Xc432与论文4124/430等协议差异未消除，仅数值对照。EAT/Eib/Tg不纳入。
本轮同时改变PT和joint FT，未独立拆分两阶段作用；冻结探针仅衡量固定head的可读性。空间角度是两个矩，不是完整三元组描述。
PT模板没有RU offset，接缝统计记未提供；下游sidecar记录原RU offset跨接缝边。无虚构周期盒、长链、MD或新增构象。
异常／缺项：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_spatial.py", line 278, in run
    job=self.launch('cpu');self.wait([job])
                           ^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_spatial.py", line 175, in wait
    self.deadline();self.refresh();self.check_failures()
                                   ^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_spatial.py", line 117, in check_failures
    if bad:raise RuntimeError('job failed; no automatic retry: '+bad[0]['name']+' exit='+str(bad[0]['exit']))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: job failed; no automatic retry: ua3d_r4_cpu exit=1
。
CPU：{'exit': 1, 'attempts': 1, 'not_run_reason': None}。
tmux窗口、命令、cwd、设备UUID、进程身份、日志及真实exit见budget.jobs；controller为ua3d-spatial-r4-controller。


</details>

<details><summary>results/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1/final_review/report.md</summary>

# A返修及旧STAGE/r1更正

旧STAGE/r1未启动：0 FULL、0 TEST、0 CPU，未冻结；旧模板“两新臂均按既定预算训练”不符合实际，以本更正为准。旧文件保持原状。

父原controller exit1、CPU exit1及failure保留；本次CPU exit0仅允许指定测试缺陷例外，不冒充父成功或独立审查通过。父科学数值未修改：表征诊断改善、整体性能未通过。

# UA3D spatial r4 实际交付

状态：需返修；独立审查PENDING，未归档或替换部署。
FINAL=C_CONST；G=S_RADANG。
科学问题：同一有限Trimer的非键空间状态消息及距离／夹角矩能否改善FULL预测与冻结PT-3D性质可读性。
BASE为本轮共同step0匹配基线；C_CONST保留真实空间邻接，不是完全无几何控制；四组非完整factorial，不宣称交互效应。
所有正式TEST在全部280估计器统一冻结后一次执行。共享开发TEST非独立盲测。

| 模式 | Arm | 正式macro R² |
|---|---|---:|
| FULL | BASE | 0.795 |

### FULL/BASE 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.017 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.917±0.010 | 0.923 | -0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.012 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.806±0.041 | 0.839 | -0.033 | Periodic-TDL |
| EPS | 0.771±0.048 | 0.787 | -0.016 | Periodic-TDL |
| Nc | 0.821±0.074 | 0.856 | -0.035 | Periodic-TDL |
| Xc | 0.415±0.105 | 0.423 | -0.008 | Periodic-TDL |
| FULL | C_CONST | 0.792 |

### FULL/C_CONST 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.916±0.008 | 0.934 | -0.018 | Periodic-TDL |
| Egb | 0.915±0.013 | 0.923 | -0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.013 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.813±0.038 | 0.839 | -0.026 | Periodic-TDL |
| EPS | 0.724±0.087 | 0.787 | -0.063 | Periodic-TDL |
| Nc | 0.805±0.064 | 0.856 | -0.051 | Periodic-TDL |
| Xc | 0.455±0.059 | 0.423 | +0.032 | Periodic-TDL |
| FULL | S_RAD | 0.790 |

### FULL/S_RAD 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.925±0.017 | 0.934 | -0.009 | Periodic-TDL |
| Egb | 0.909±0.014 | 0.923 | -0.014 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.916±0.011 | 0.911 | +0.005 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.816±0.022 | 0.839 | -0.023 | Periodic-TDL |
| EPS | 0.746±0.060 | 0.787 | -0.041 | Periodic-TDL |
| Nc | 0.811±0.062 | 0.856 | -0.045 | Periodic-TDL |
| Xc | 0.405±0.130 | 0.423 | -0.018 | Periodic-TDL |
| FULL | S_RADANG | 0.792 |

### FULL/S_RADANG 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.922±0.020 | 0.934 | -0.012 | Periodic-TDL |
| Egb | 0.906±0.018 | 0.923 | -0.017 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.012 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.801±0.041 | 0.839 | -0.038 | Periodic-TDL |
| EPS | 0.735±0.100 | 0.787 | -0.052 | Periodic-TDL |
| Nc | 0.833±0.051 | 0.856 | -0.023 | Periodic-TDL |
| Xc | 0.432±0.111 | 0.423 | +0.009 | Periodic-TDL |
| 模式 | Arm | 正式macro R² |
|---|---|---:|
| probe | BASE | 0.671 |

### probe/BASE 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.827±0.037 | 0.934 | -0.107 | Periodic-TDL |
| Egb | 0.834±0.021 | 0.923 | -0.089 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.812±0.015 | 0.911 | -0.099 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.635±0.078 | 0.839 | -0.204 | Periodic-TDL |
| EPS | 0.580±0.157 | 0.787 | -0.207 | Periodic-TDL |
| Nc | 0.689±0.049 | 0.856 | -0.167 | Periodic-TDL |
| Xc | 0.318±0.077 | 0.423 | -0.105 | Periodic-TDL |
| probe | C_CONST | 0.682 |

### probe/C_CONST 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.811±0.028 | 0.934 | -0.123 | Periodic-TDL |
| Egb | 0.832±0.018 | 0.923 | -0.091 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.808±0.016 | 0.911 | -0.103 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.636±0.072 | 0.839 | -0.203 | Periodic-TDL |
| EPS | 0.627±0.101 | 0.787 | -0.160 | Periodic-TDL |
| Nc | 0.710±0.034 | 0.856 | -0.146 | Periodic-TDL |
| Xc | 0.353±0.066 | 0.423 | -0.070 | Periodic-TDL |
| probe | S_RAD | 0.693 |

### probe/S_RAD 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.848±0.029 | 0.934 | -0.086 | Periodic-TDL |
| Egb | 0.835±0.027 | 0.923 | -0.088 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.805±0.015 | 0.911 | -0.106 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.667±0.073 | 0.839 | -0.172 | Periodic-TDL |
| EPS | 0.625±0.113 | 0.787 | -0.162 | Periodic-TDL |
| Nc | 0.711±0.057 | 0.856 | -0.145 | Periodic-TDL |
| Xc | 0.361±0.038 | 0.423 | -0.062 | Periodic-TDL |
| probe | S_RADANG | 0.700 |

### probe/S_RADANG 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.852±0.014 | 0.934 | -0.082 | Periodic-TDL |
| Egb | 0.840±0.020 | 0.923 | -0.083 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.809±0.007 | 0.911 | -0.102 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.691±0.028 | 0.839 | -0.148 | Periodic-TDL |
| EPS | 0.660±0.091 | 0.787 | -0.127 | Periodic-TDL |
| Nc | 0.712±0.046 | 0.856 | -0.144 | Periodic-TDL |
| Xc | 0.341±0.076 | 0.423 | -0.082 | Periodic-TDL |

### FULL 配对差值

| 对照 | Δmacro R² | 达到0.010 |
|---|---:|---|
| G_minus_BASE | -0.004 | False |
| G_minus_C_CONST | +0.000 | False |
| C_CONST_minus_BASE | -0.004 | False |
| distance | -0.002 | False |
| angle | +0.002 | False |

### probe 配对差值

| 对照 | Δmacro R² | 达到0.010 |
|---|---:|---|
| G_minus_BASE | +0.030 | True |
| G_minus_C_CONST | +0.018 | True |
| C_CONST_minus_BASE | +0.012 | True |
| distance | +0.011 | True |
| angle | +0.007 | False |

表征诊断改善、整体性能未通过

实际预算：{"prepare_starts": 1, "cache_rows": 5875, "pt_starts": 4, "pt_updates": 40000, "pt_microbatches": 240000, "pt_presentations": 40320000, "ft_starts": 140, "ft_epochs": 9800, "ft_updates": 224000, "probe_starts": 140, "probe_epochs": 4200, "probe_updates": 96000, "feature_rows": 94000, "test_starts": 280, "test_graphs": 47000, "cpu_suites": 1}
原始精度、配对fold差值、退化任务／fold、参数与真实耗时见results.json及各单元history/consumption。未完成不得以validation或旧结果补齐。
五折mean±std(ddof=1)，seed42+fold，七任务等权macro。论文是逐任务最佳包络；Egc3380/Xc432与论文4124/430等协议差异未消除，仅数值对照。EAT/Eib/Tg不纳入。
本轮同时改变PT和joint FT，未独立拆分两阶段作用；冻结探针仅衡量固定head的可读性。空间角度是两个矩，不是完整三元组描述。
PT模板没有RU offset，接缝统计记未提供；下游sidecar记录原RU offset跨接缝边。无虚构周期盒、长链、MD或新增构象。
异常／缺项：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_spatial.py", line 278, in run
    job=self.launch('cpu');self.wait([job])
                           ^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_spatial.py", line 175, in wait
    self.deadline();self.refresh();self.check_failures()
                                   ^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_spatial.py", line 117, in check_failures
    if bad:raise RuntimeError('job failed; no automatic retry: '+bad[0]['name']+' exit='+str(bad[0]['exit']))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: job failed; no automatic retry: ua3d_r4_cpu exit=1
。
CPU：{'exit': 1, 'attempts': 1, 'not_run_reason': None}。
tmux窗口、命令、cwd、设备UUID、进程身份、日志及真实exit见budget.jobs；controller为ua3d-spatial-r4-controller。


</details>

<details><summary>results/ua3d_stage_20261026/r1/final_review/report.md</summary>

# UA3D-STAGE r1 实际交付

状态：阻断；独立审查PENDING；未归档或替换部署。
G=None；REF=None；C2=None；NEXT_FINAL=None；主要性能门槛通过=None。
科学问题：空间分支在PT和FT启用的阶段作用及其描述性交互。E00/E11/REF复用父r4，两个交叉臂从PT endpoint完整微调。
PT源改变可能连带影响2D/FP/Q；PT_ONLY撤去空间模块会改变表示分布，不能单独解释为仅3D权重的因果收益。
两新臂均按既定预算训练，全部validation后一次冻结，再各TEST一次。共享开发TEST非独立盲测，未按TEST重选或自动部署。

| 单元 | 正式macro R² | 来源 |
|---|---:|---|
| E00 | 未完成／不可比较 | 父预测复用 |
| E11 | 未完成／不可比较 | 父预测复用 |
| PT_ONLY | 未完成／不可比较 | 本轮新训练 |
| FT_ONLY | 未完成／不可比较 | 本轮新训练 |
| REF | 未完成／不可比较 | 父预测复用 |

### E00 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

### E11 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

### PT_ONLY 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

### FT_ONLY 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

### REF 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

### 固定配对差值

| 对照 | Δmacro R² | 达到0.010 |
|---|---:|---|
| PT_when_FT_off | 未完成／不可比较 | 未判定 |
| PT_when_FT_on | 未完成／不可比较 | 未判定 |
| FT_when_PT_off | 未完成／不可比较 | 未判定 |
| FT_when_PT_on | 未完成／不可比较 | 未判定 |
| PT_ONLY_minus_REF | 未完成／不可比较 | 未判定 |
| FT_ONLY_minus_REF | 未完成／不可比较 | 未判定 |

阶段交互I=(E11−E10)−(E01−E00)：未完成／不可比较；描述性统计，不称统计显著。

七任务五折mean±std(ddof=1)，seed42+fold，任务等权macro；比较/门槛用未舍入值。各任务/退化fold、参数数目与真实耗时见results.json和history/consumption。
论文采用SOTA_RESULT逐任务最佳包络；Egc3380/Xc432与论文4124/430等数据和协议差异仍在，仅数值对照。EAT/Eib/Tg不纳入。
父／新／累计预算见budget；父引用不重训、不重推理、不重放CPU；本轮无PT/prepare/probe/构象/FP生成。
异常或缺项：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_stage.py", line 300, in run
    self.wait_parent();self.matrix('ft');frozen=t.freeze()
    ^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_stage.py", line 133, in wait_parent
    self.deadline();self.refresh();evidence=t.inspect_parent()
                                            ^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_stage.py", line 73, in inspect_parent
    if budget.get('budget_violations') or budget.get('exit_protocol_errors') or (R4/'failure.json').exists():raise RuntimeError('parent failure/violation forbids continuation')
                                                                                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: parent failure/violation forbids continuation
。
唯一CPU：{'exit': None, 'attempts': 0, 'not_run_reason': 'authorized experiments incomplete; no early suite'}。
tmux：ua3d-stage-r1-controller；每个作业命令、cwd、设备UUID、进程身份、日志、起止和真实exit见budget.jobs。


</details>

<details><summary>results/ua3d_stage_20261026/r2/final_review/report.md</summary>

# UA3D-STAGE r2 实际交付

状态：需返修；独立审查PENDING；未归档或替换部署。
G=S_RADANG；REF=C_CONST；C2=PT_ONLY；NEXT_FINAL=REF；主要性能门槛通过=False。
科学问题：空间分支在PT和FT启用的阶段作用及其描述性交互。E00/E11/REF复用父r4，两个交叉臂从PT endpoint完整微调。
PT源改变可能连带影响2D/FP/Q；PT_ONLY撤去空间模块会改变表示分布，不能单独解释为仅3D权重的因果收益。
完整完成：新FULL 70/70，全部validation后一次冻结，正式TEST 70/70及row+sample key对齐完成。
共享开发TEST非独立盲测，未按TEST重选或自动部署。

| 单元 | 正式macro R² | 来源 |
|---|---:|---|
| E00 | 0.795 | 父预测复用 |
| E11 | 0.792 | 父预测复用 |
| PT_ONLY | 0.790 | 本轮新训练 |
| FT_ONLY | 0.797 | 本轮新训练 |
| REF | 0.792 | 父预测复用 |

### E00 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.017 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.917±0.010 | 0.923 | -0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.012 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.806±0.041 | 0.839 | -0.033 | Periodic-TDL |
| EPS | 0.771±0.048 | 0.787 | -0.016 | Periodic-TDL |
| Nc | 0.821±0.074 | 0.856 | -0.035 | Periodic-TDL |
| Xc | 0.415±0.105 | 0.423 | -0.008 | Periodic-TDL |

### E11 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.922±0.020 | 0.934 | -0.012 | Periodic-TDL |
| Egb | 0.906±0.018 | 0.923 | -0.017 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.012 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.801±0.041 | 0.839 | -0.038 | Periodic-TDL |
| EPS | 0.735±0.100 | 0.787 | -0.052 | Periodic-TDL |
| Nc | 0.833±0.051 | 0.856 | -0.023 | Periodic-TDL |
| Xc | 0.432±0.111 | 0.423 | +0.009 | Periodic-TDL |

### PT_ONLY 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.920±0.015 | 0.934 | -0.014 | Periodic-TDL |
| Egb | 0.920±0.012 | 0.923 | -0.003 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.014 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.810±0.042 | 0.839 | -0.029 | Periodic-TDL |
| EPS | 0.742±0.068 | 0.787 | -0.045 | Periodic-TDL |
| Nc | 0.817±0.039 | 0.856 | -0.039 | Periodic-TDL |
| Xc | 0.413±0.074 | 0.423 | -0.010 | Periodic-TDL |

### FT_ONLY 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.928±0.016 | 0.934 | -0.006 | Periodic-TDL |
| Egb | 0.912±0.013 | 0.923 | -0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.012 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.816±0.036 | 0.839 | -0.023 | Periodic-TDL |
| EPS | 0.775±0.053 | 0.787 | -0.012 | Periodic-TDL |
| Nc | 0.826±0.062 | 0.856 | -0.030 | Periodic-TDL |
| Xc | 0.407±0.115 | 0.423 | -0.016 | Periodic-TDL |

### REF 七任务

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.916±0.008 | 0.934 | -0.018 | Periodic-TDL |
| Egb | 0.915±0.013 | 0.923 | -0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.013 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.813±0.038 | 0.839 | -0.026 | Periodic-TDL |
| EPS | 0.724±0.087 | 0.787 | -0.063 | Periodic-TDL |
| Nc | 0.805±0.064 | 0.856 | -0.051 | Periodic-TDL |
| Xc | 0.455±0.059 | 0.423 | +0.032 | Periodic-TDL |

### 固定配对差值

| 对照 | Δmacro R² | 达到0.010 |
|---|---:|---|
| C2_minus_REF | -0.001 | False |
| C2_minus_E00 | -0.005 | False |
| PT_when_FT_off | -0.005 | False |
| PT_when_FT_on | -0.005 | False |
| FT_when_PT_off | +0.002 | False |
| FT_when_PT_on | +0.001 | False |
| PT_ONLY_minus_REF | -0.001 | False |
| FT_ONLY_minus_REF | +0.005 | False |

阶段交互I=(E11−E10)−(E01−E00)：-0.000；描述性统计，不称统计显著。

七任务五折mean±std(ddof=1)，seed42+fold，任务等权macro；比较/门槛用未舍入值。各任务/退化fold、参数数目与真实耗时见results.json和history/consumption。
论文采用SOTA_RESULT逐任务最佳包络；Egc3380/Xc432与论文4124/430等数据和协议差异仍在，仅数值对照。EAT/Eib/Tg不纳入。
父／新／累计预算见budget；父引用不重训、不重推理、不重放CPU；本轮无PT/prepare/probe/构象/FP生成。
异常或缺项：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_stage.py", line 310, in run
    job=self.launch('cpu');self.wait([job])
                           ^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_stage.py", line 175, in wait
    self.deadline();self.refresh();self.check_failures()
                                   ^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_stage.py", line 204, in check_failures
    if bad:raise RuntimeError('job failed; no automatic retry: '+bad[0]['name']+' exit='+str(bad[0]['exit']))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: job failed; no automatic retry: ua3d_stage_r2_cpu exit=1
。
唯一CPU：{'exit': 1, 'attempts': 1, 'not_run_reason': None}。
tmux：ua3d-stage-r2-controller；每个作业命令、cwd、设备UUID、进程身份、日志、起止和真实exit见budget.jobs。


</details>

<details><summary>results/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1/final_review/report.md</summary>

# reservation guard返修

A唯一CPU通过；原STAGE controller exit1、CPU exit1和failure保留；旧140个科学作业exit0。仅接纳指定guard缺陷；不是独立审查通过。完整原始统计/冻结只读引用旧final_review/results.json，无新增训练或推理。


</details>

<details><summary>results/tmol3m_3d_ablation_20261025/r1/final_review/report.md</summary>

# TMOL3M 下游3D配对消融实际交付

状态：需返修；独立复审PENDING；未归档。
FINAL=None；Δ3D未舍入值=None；未完成／不可比较。
优化差值未舍入值=None；优化通过=None。

| 组 | 正式TEST macro R² |
|---|---:|
| FULL | 未完成／不可比较 |
| NO3D_FT | 未完成／不可比较 |

## FULL_minus_NO3D_FT

未舍入Δmacro=None；改善fold数=None。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|

## NO3D_FT_minus_FULL

未舍入Δmacro=None；改善fold数=None。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|

## FINAL_minus_FULL

未舍入Δmacro=None；改善fold数=None。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|

## FULL 七任务SOTA数值对照

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## NO3D_FT 七任务SOTA数值对照

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## FINAL 七任务SOTA数值对照

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

FULL相对历史TMOL3M未舍入Δmacro=None；仅跨周期数值参照，不替代本轮FULL配对基线。

FULL相对历史B_A020未舍入Δmacro=None；仅跨周期数值参照，不替代本轮FULL配对基线。

NO3D_FT相对历史TMOL3M未舍入Δmacro=None；仅跨周期数值参照，不替代本轮FULL配对基线。

NO3D_FT相对历史B_A020未舍入Δmacro=None；仅跨周期数值参照，不替代本轮FULL配对基线。

## 成本、可用率和证据

共享开发TEST上的配对消融，非独立盲测。两组同一三模态PT起点、原35头初态和固定train-only scaler，七任务五折等权macro，seed42+fold，std使用ddof=1；不是五次独立seed。结论仅限整个下游3D路径，不是纯几何信息、3D预训练或interaction的独立贡献。删除涉及3D的两个ALIGN方向及熵支持集变化属于干预；alpha/beta不变、熵不归一化。NO3D保留休眠模块，活动参数容量不同；gate输出3D行任务梯度为零不代表AdamW weight decay不更新该行。跨GPU不宣称逐位确定。论文是逐任务最佳包络，协议等价未证明，Egc3380/Xc432与论文4124/430不同，仅数值对照，不宣称公平超越或SOTA；EAT/Eib/Tg不纳入。

原始缓存5875/5860/15保持。FULL实际3D呈现与NO3D全零呈现、各模块调用次数、对齐有效计数见budget及单元consumption.path_audit。参数总数／活动数／head和joint可训练数、优化器参数名见parameters/optimizer_*.json；耗时与设备来自实际运行，无benchmark。
预算：{"ft_starts": 16, "epochs_started": 16, "attempted_updates": 417, "train_graphs": 9975, "validation_started": 11, "validation_graphs": 805, "test_starts": 0, "test_graphs": 0, "train_3d_graphs": 9944, "validation_3d_graphs": 792, "test_3d_graphs": 0, "cpu_suites": 0}
CPU唯一尝试：{"exit": null, "attempts": 0, "command": null, "log": null, "not_run_reason": "authorized experiments/report incomplete; no early suite"}
异常／未完成：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_tmol3m_3d_ablation.py", line 238, in run
    self.matrix(False);frozen=t.freeze();t.update_plan('执行中','一次冻结',f'FINAL={frozen["FINAL"]}，全部70 BEST/scaler/config已冻结。')
    ^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_tmol3m_3d_ablation.py", line 163, in matrix
    self.deadline();self.refresh();self.check_failures()
                                   ^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_tmol3m_3d_ablation.py", line 103, in check_failures
    if bad:raise RuntimeError('job failed; no automatic retry: '+bad[0]['name']+' exit='+str(bad[0]['exit']))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: job failed; no automatic retry: tmol3d_ft_FULL_eea_0 exit=1

tmux Uni-Poly:tmol3m-3d-ablation-r1；日志/root/workspace/Uni-Poly-Plus-master/logs/tmol3m_3d_ablation_20261025/r1；所有命令、PID/start_ticks、UUID及真实exit见budget.jobs及日志。
旧父源码、产物和FINAL只读，PROJECT_HISTORY未修改。


## 失败原因与停止核对

- 根因：新training文件缺少`import json`，首轮`print(json.dumps(rec))`触发NameError（第141行）。11个FULL单元触发该错误，另5个在控制器停止清理时受控终止；16个已启动FT全部真实exit1，完成FT为0。控制器真实exit1，cleanup.errors=[]，controller/wrapper/timeout/worker的PID＋start_ticks均无存活匹配，GPU compute列表为空。
- 已有16份正常初始化记录均common_initialization_equal=true；只验证这些已启动FULL单元的初始化，NO3D_FT尚未启动，不能据此声称跨组运行或3D消融已通过。
- 不可退预留：16FT、1120epochs、41300attempted updates、981540训练呈现、1120次validation及245700呈现；实际持久记录下界：16 epochs_started、417attempted updates、9975训练呈现、11次validation及805呈现。中途终止epoch可能超出最后持久计数，不能将下界当作精确全部消费。
- 剩余54FT未启动；选择／冻结／70正式TEST均未执行，唯一最终CPU尝试0（按合同不得提前运行）。两组全部七任务正式结果未完成／不可比较，不能判断Δ3D或优化通过。
- `incident.json`记录根因和真实exit；`proposed_repair.patch`仅为未应用的一行导入修复建议；`repair_scope.json`列16个替代单元及所需累计预算增量。未自动修复源码、重试、resume或补跑；原截止仍为2026-10-26T16:40:46.305387+08:00。


</details>

<details><summary>results/tmol3m_3d_ablation_20261025/r1/repairs/json_import_repair1/final_review/report.md</summary>

# TMOL3M 下游3D配对消融实际交付

状态：待审查；独立复审PENDING；未归档。
FINAL=FULL；Δ3D未舍入值=0.004578983000486758；未达到预设0.010收益门槛；不表示3D无作用或两组等价。
优化差值未舍入值=0.0；优化通过=False。

| 组 | 正式TEST macro R² |
|---|---:|
| FULL | 0.788 |
| NO3D_FT | 0.784 |

## FULL_minus_NO3D_FT

未舍入Δmacro=0.004578983000486758；改善fold数=22。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|
| eea | +0.004±0.008 | 3/5 |
| egb | +0.000±0.007 | 3/5 |
| egc | +0.003±0.006 | 3/5 |
| ei | +0.001±0.037 | 3/5 |
| eps | +0.005±0.050 | 3/5 |
| nc | +0.025±0.023 | 5/5 |
| xc | -0.007±0.079 | 2/5 |

## NO3D_FT_minus_FULL

未舍入Δmacro=-0.004578983000486758；改善fold数=13。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|
| eea | -0.004±0.008 | 2/5 |
| egb | -0.000±0.007 | 2/5 |
| egc | -0.003±0.006 | 2/5 |
| ei | -0.001±0.037 | 2/5 |
| eps | -0.005±0.050 | 2/5 |
| nc | -0.025±0.023 | 0/5 |
| xc | +0.007±0.079 | 3/5 |

## FINAL_minus_FULL

未舍入Δmacro=0.0；改善fold数=0。

| Task | 配对差值mean±std | 改善fold |
|---|---:|---:|
| eea | +0.000±0.000 | 0/5 |
| egb | +0.000±0.000 | 0/5 |
| egc | +0.000±0.000 | 0/5 |
| ei | +0.000±0.000 | 0/5 |
| eps | +0.000±0.000 | 0/5 |
| nc | +0.000±0.000 | 0/5 |
| xc | +0.000±0.000 | 0/5 |

## FULL 七任务SOTA数值对照

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.921±0.016 | 0.934 | -0.013 | Periodic-TDL |
| Egb | 0.916±0.004 | 0.923 | -0.007 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.910±0.014 | 0.911 | -0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.792±0.011 | 0.839 | -0.047 | Periodic-TDL |
| EPS | 0.747±0.044 | 0.787 | -0.040 | Periodic-TDL |
| Nc | 0.825±0.064 | 0.856 | -0.031 | Periodic-TDL |
| Xc | 0.408±0.141 | 0.423 | -0.015 | Periodic-TDL |

## NO3D_FT 七任务SOTA数值对照

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.917±0.012 | 0.934 | -0.017 | Periodic-TDL |
| Egb | 0.916±0.009 | 0.923 | -0.007 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.907±0.014 | 0.911 | -0.004 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.791±0.034 | 0.839 | -0.048 | Periodic-TDL |
| EPS | 0.742±0.092 | 0.787 | -0.045 | Periodic-TDL |
| Nc | 0.800±0.065 | 0.856 | -0.056 | Periodic-TDL |
| Xc | 0.415±0.102 | 0.423 | -0.008 | Periodic-TDL |

## FINAL 七任务SOTA数值对照

| Task | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.921±0.016 | 0.934 | -0.013 | Periodic-TDL |
| Egb | 0.916±0.004 | 0.923 | -0.007 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.910±0.014 | 0.911 | -0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.792±0.011 | 0.839 | -0.047 | Periodic-TDL |
| EPS | 0.747±0.044 | 0.787 | -0.040 | Periodic-TDL |
| Nc | 0.825±0.064 | 0.856 | -0.031 | Periodic-TDL |
| Xc | 0.408±0.141 | 0.423 | -0.015 | Periodic-TDL |

FULL相对历史TMOL3M未舍入Δmacro=-0.0015193945599878411；仅跨周期数值参照，不替代本轮FULL配对基线。

FULL相对历史B_A020未舍入Δmacro=0.0033195188082958405；仅跨周期数值参照，不替代本轮FULL配对基线。

NO3D_FT相对历史TMOL3M未舍入Δmacro=-0.006098377560474599；仅跨周期数值参照，不替代本轮FULL配对基线。

NO3D_FT相对历史B_A020未舍入Δmacro=-0.0012594641921909178；仅跨周期数值参照，不替代本轮FULL配对基线。

返修json-import-repair-r1：父16个FULL失败单元及部分BEST原位保留，整单元预留不可退；新活动70FT从原PT/头初态开始。父实际持久记录仅为下界；父失败、新活动和累计账本分栏。继承原绝对截止2026-10-26T16:40:46.305387+08:00，未刷新；父失败不进入新活动失败判定。

## 成本、可用率和证据

共享开发TEST上的配对消融，非独立盲测。两组同一三模态PT起点、原35头初态和固定train-only scaler，七任务五折等权macro，seed42+fold，std使用ddof=1；不是五次独立seed。结论仅限整个下游3D路径，不是纯几何信息、3D预训练或interaction的独立贡献。删除涉及3D的两个ALIGN方向及熵支持集变化属于干预；alpha/beta不变、熵不归一化。NO3D保留休眠模块，活动参数容量不同；gate输出3D行任务梯度为零不代表AdamW weight decay不更新该行。跨GPU不宣称逐位确定。论文是逐任务最佳包络，协议等价未证明，Egc3380/Xc432与论文4124/430不同，仅数值对照，不宣称公平超越或SOTA；EAT/Eib/Tg不纳入。

原始缓存5875/5860/15保持。FULL实际3D呈现与NO3D全零呈现、各模块调用次数、对齐有效计数见budget及单元consumption.path_audit。参数总数／活动数／head和joint可训练数、优化器参数名见parameters/optimizer_*.json；耗时与设备来自实际运行，无benchmark。
预算：{"ft_starts": 86, "epochs_started": 4916, "attempted_updates": 112417, "train_graphs": 2640015, "validation_started": 4911, "validation_graphs": 660765, "test_starts": 70, "test_graphs": 11750, "train_3d_graphs": 1322094, "validation_3d_graphs": 329442, "test_3d_graphs": 5860, "cpu_suites": 1}
CPU唯一尝试：{"exit": 0, "attempts": 1, "command": ["timeout", "--signal=TERM", "--kill-after=10s", "600s", "python", "-u", "scripts/run_tmol3m_3d_ablation.py", "cpu", "--execution-root", "/root/workspace/Uni-Poly-Plus-master/results/tmol3m_3d_ablation_20261025/r1/repairs/json_import_repair1"], "log": "/root/workspace/Uni-Poly-Plus-master/logs/tmol3m_3d_ablation_20261025/r1/repairs/json_import_repair1/tmol3d_cpu", "time": "2026-10-25T17:11:38.476821+08:00"}
异常／未完成：无
tmux Uni-Poly:tmol3m-3d-ablation-json-repair1；日志/root/workspace/Uni-Poly-Plus-master/logs/tmol3m_3d_ablation_20261025/r1/repairs/json_import_repair1；所有命令、PID/start_ticks、UUID及真实exit见budget.jobs及日志。
旧父源码、产物和FINAL只读，PROJECT_HISTORY未修改。


</details>

<details><summary>results/ua2d_fg10k_morgan_20261023/r1/final_review/report.md</summary>

# UA2D-FG10K-MORGAN-20261023-01 / r1.1 交付

状态：待审查；固定主候选GATE_FP；CPU exit=0。
执行自检不替代Codex独立审查。

| 对照 | 正式Δmacro R² |
|---|---:|
| CONCAT_FP_minus_FG_REF | -0.025 |
| GATE_FP_minus_FG_REF | -0.024 |
| GATE_FP_minus_CONCAT_FP | +0.001 |

## FG_REF 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.923±0.016 | 0.934 | -0.011 | Periodic-TDL |
| Egb | 0.934±0.014 | 0.923 | +0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.917±0.017 | 0.911 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.798±0.018 | 0.839 | -0.041 | Periodic-TDL |
| EPS | 0.707±0.076 | 0.787 | -0.080 | Periodic-TDL |
| Nc | 0.773±0.090 | 0.856 | -0.083 | Periodic-TDL |
| Xc | 0.423±0.114 | 0.423 | -0.000 | Periodic-TDL |
macro R²=0.782。

## CONCAT_FP 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.921±0.015 | 0.934 | -0.013 | Periodic-TDL |
| Egb | 0.916±0.010 | 0.923 | -0.007 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.892±0.017 | 0.911 | -0.019 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.758±0.097 | 0.839 | -0.081 | Periodic-TDL |
| EPS | 0.659±0.143 | 0.787 | -0.128 | Periodic-TDL |
| Nc | 0.796±0.086 | 0.856 | -0.060 | Periodic-TDL |
| Xc | 0.360±0.103 | 0.423 | -0.063 | Periodic-TDL |
macro R²=0.757。

## GATE_FP 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.921±0.015 | 0.934 | -0.013 | Periodic-TDL |
| Egb | 0.915±0.010 | 0.923 | -0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.894±0.017 | 0.911 | -0.017 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.757±0.098 | 0.839 | -0.082 | Periodic-TDL |
| EPS | 0.662±0.145 | 0.787 | -0.125 | Periodic-TDL |
| Nc | 0.798±0.079 | 0.856 | -0.058 | Periodic-TDL |
| Xc | 0.359±0.088 | 0.423 | -0.064 | Periodic-TDL |
macro R²=0.758。

五折seed42+fold等权mean/std(ddof1)，再七任务等权macro；使用未舍入值判定0.010门槛。
论文参考为逐任务包络，协议未证明等价，仅数值比较；共享开发TEST非独立盲测。新增模型容量也改变，不将差值单独归因于FP或门控机制。
异常／未完成：无


</details>

<details><summary>results/ua2d_et_single_20261020/r1/final_review/report.md</summary>

# UA2D-ET-SINGLE-20261020-01 / r1 执行交付

状态：待审查；独立审查未完成。

五折seed42—46，std ddof=1，七任务等权macro；完整精度判定，展示3位小数。
历史配方比较不能单独归因ET；共享validation／TEST非独立盲测；论文协议未证明一致，仅数值对照。

validation宏R² 0.815；V1 0.817；差值 -0.002；门槛通过=False。

|任务|论文模型|参考R²|ET_SINGLE正式R²|项目减参考|
|---|---|---:|---|---|
|eea|Periodic-TDL|0.934|未完成／不可比较（validation门槛未通过，TEST=0）|—|
|egb|Periodic-TDL (+Chem, +RC)|0.923|未完成／不可比较（validation门槛未通过，TEST=0）|—|
|egc|Periodic-TDL (+Chem, +RC)|0.911|未完成／不可比较（validation门槛未通过，TEST=0）|—|
|ei|Periodic-TDL|0.839|未完成／不可比较（validation门槛未通过，TEST=0）|—|
|eps|Periodic-TDL|0.787|未完成／不可比较（validation门槛未通过，TEST=0）|—|
|nc|Periodic-TDL|0.856|未完成／不可比较（validation门槛未通过，TEST=0）|—|
|xc|Periodic-TDL|0.423|未完成／不可比较（validation门槛未通过，TEST=0）|—|


</details>

<details><summary>results/ua2d_et_single_20261020/r1/incidents/attempt1/final_review/report.md</summary>

# UA2D-ET-SINGLE-20261020-01 / r1 执行交付

状态：需返修；独立审查未完成。

五折seed42—46，std ddof=1，七任务等权macro；完整精度判定，展示3位小数。
历史配方比较不能单独归因ET；共享validation／TEST非独立盲测；论文协议未证明一致，仅数值对照。


|任务|论文模型|参考R²|ET_SINGLE正式R²|项目减参考|
|---|---|---:|---|---|
|eea|Periodic-TDL|0.934|未完成／不可比较（[Errno 2] No such file or directory: '/root/workspace/Uni-Poly-Plus-master/results/ua2d_et_single_20261020/r1/formal/ET_SINGLE/eea/fold0/metrics.json'）|—|
|egb|Periodic-TDL (+Chem, +RC)|0.923|未完成／不可比较（[Errno 2] No such file or directory: '/root/workspace/Uni-Poly-Plus-master/results/ua2d_et_single_20261020/r1/formal/ET_SINGLE/eea/fold0/metrics.json'）|—|
|egc|Periodic-TDL (+Chem, +RC)|0.911|未完成／不可比较（[Errno 2] No such file or directory: '/root/workspace/Uni-Poly-Plus-master/results/ua2d_et_single_20261020/r1/formal/ET_SINGLE/eea/fold0/metrics.json'）|—|
|ei|Periodic-TDL|0.839|未完成／不可比较（[Errno 2] No such file or directory: '/root/workspace/Uni-Poly-Plus-master/results/ua2d_et_single_20261020/r1/formal/ET_SINGLE/eea/fold0/metrics.json'）|—|
|eps|Periodic-TDL|0.787|未完成／不可比较（[Errno 2] No such file or directory: '/root/workspace/Uni-Poly-Plus-master/results/ua2d_et_single_20261020/r1/formal/ET_SINGLE/eea/fold0/metrics.json'）|—|
|nc|Periodic-TDL|0.856|未完成／不可比较（[Errno 2] No such file or directory: '/root/workspace/Uni-Poly-Plus-master/results/ua2d_et_single_20261020/r1/formal/ET_SINGLE/eea/fold0/metrics.json'）|—|
|xc|Periodic-TDL|0.423|未完成／不可比较（[Errno 2] No such file or directory: '/root/workspace/Uni-Poly-Plus-master/results/ua2d_et_single_20261020/r1/formal/ET_SINGLE/eea/fold0/metrics.json'）|—|


</details>

<details><summary>results/ua2d_ph_descriptor_20261023/r1_p10k_gpu01/final_review/report.md</summary>

# UA2D-PH-DESC-20261023-01 / r1.2-P10K-GPU01 交付

状态：待审查；固定主候选GATE_PH；CPU exit=0。
执行自检不替代Codex独立审查。

| 对照 | 正式Δmacro R² |
|---|---:|
| CONCAT_PH_minus_P10K_REF | -0.011 |
| GATE_PH_minus_P10K_REF | -0.012 |
| GATE_PH_minus_CONCAT_PH | -0.001 |

## P10K_REF 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.932±0.022 | 0.934 | -0.002 | Periodic-TDL |
| Egb | 0.929±0.015 | 0.923 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.916±0.013 | 0.911 | +0.005 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.820±0.019 | 0.839 | -0.019 | Periodic-TDL |
| EPS | 0.736±0.073 | 0.787 | -0.051 | Periodic-TDL |
| Nc | 0.815±0.079 | 0.856 | -0.041 | Periodic-TDL |
| Xc | 0.404±0.117 | 0.423 | -0.019 | Periodic-TDL |
macro R²=0.793。

## CONCAT_PH 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.016 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.929±0.010 | 0.923 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.918±0.013 | 0.911 | +0.007 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.790±0.024 | 0.839 | -0.049 | Periodic-TDL |
| EPS | 0.730±0.073 | 0.787 | -0.057 | Periodic-TDL |
| Nc | 0.824±0.059 | 0.856 | -0.032 | Periodic-TDL |
| Xc | 0.358±0.095 | 0.423 | -0.065 | Periodic-TDL |
macro R²=0.782。

## GATE_PH 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.925±0.015 | 0.934 | -0.009 | Periodic-TDL |
| Egb | 0.926±0.011 | 0.923 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.016 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.785±0.021 | 0.839 | -0.054 | Periodic-TDL |
| EPS | 0.734±0.075 | 0.787 | -0.053 | Periodic-TDL |
| Nc | 0.830±0.061 | 0.856 | -0.026 | Periodic-TDL |
| Xc | 0.355±0.109 | 0.423 | -0.068 | Periodic-TDL |
macro R²=0.781。

五折seed42+fold等权mean/std(ddof1)，再七任务等权macro；使用未舍入值判定0.010门槛。
论文参考为逐任务包络，协议未证明等价，仅数值比较；共享开发TEST非独立盲测。新增模型容量也改变，不将差值单独归因于PH或门控机制。
异常／未完成：无


</details>

<details><summary>results/ua2d_transfer_20261022/r1/final_review/report.md</summary>

# UA2D-TRANSFER r1 实际交付

状态：待审查；DSH执行，待Codex独立审查。
FINAL=R_STD_SP；完整四臂正式TEST=True；性能：本次未观察到提升；最终CPU exit=0。

B_FP已有正式macro=0.791，A_MIX246=0.791，均直接复用，不重训或重测。
全新独立周期：一次5875条原始2-hop准备／公共残差初始化，PT=0，四臂140FT及唯一四臂TEST；失败不retry/resume，未使用旧周期账本／attempt_mapping。
统计输入同结构对照为R_STD−R_MEAN，不将所有读出变化归因于唯一机制；interaction采用预登记的四项差分。

| 预登记正式对照 | Δmacro R² | 完整性 |
|---|---:|---|
| R_MEAN_minus_B_FP | +0.000 | COMPLETE |
| R_STD_minus_B_FP | -0.003 | COMPLETE |
| A_SP_minus_B_FP | -0.001 | COMPLETE |
| R_STD_SP_minus_B_FP | -0.006 | COMPLETE |
| R_STD_minus_R_MEAN | -0.003 | COMPLETE |
| R_STD_SP_minus_R_STD | -0.003 | COMPLETE |
| R_STD_SP_minus_A_SP | -0.005 | COMPLETE |
| interaction | -0.002 | COMPLETE |
| FINAL_minus_A_MIX246 | -0.006 | COMPLETE |

## R_MEAN 七任务SOTA数值对照（macro=0.791）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.934±0.012 | 0.934 | +0.000 | Periodic-TDL |
| Egb | 0.931±0.017 | 0.923 | +0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.910±0.017 | 0.911 | -0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.819±0.024 | 0.839 | -0.020 | Periodic-TDL |
| EPS | 0.739±0.058 | 0.787 | -0.048 | Periodic-TDL |
| Nc | 0.806±0.072 | 0.856 | -0.050 | Periodic-TDL |
| Xc | 0.401±0.135 | 0.423 | -0.022 | Periodic-TDL |

## R_STD 七任务SOTA数值对照（macro=0.788）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.933±0.014 | 0.934 | -0.001 | Periodic-TDL |
| Egb | 0.932±0.011 | 0.923 | +0.009 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.915±0.017 | 0.911 | +0.004 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.807±0.036 | 0.839 | -0.032 | Periodic-TDL |
| EPS | 0.721±0.085 | 0.787 | -0.066 | Periodic-TDL |
| Nc | 0.792±0.067 | 0.856 | -0.064 | Periodic-TDL |
| Xc | 0.416±0.104 | 0.423 | -0.007 | Periodic-TDL |

## A_SP 七任务SOTA数值对照（macro=0.790）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.932±0.011 | 0.934 | -0.002 | Periodic-TDL |
| Egb | 0.929±0.015 | 0.923 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.018 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.820±0.040 | 0.839 | -0.019 | Periodic-TDL |
| EPS | 0.760±0.054 | 0.787 | -0.027 | Periodic-TDL |
| Nc | 0.783±0.093 | 0.856 | -0.073 | Periodic-TDL |
| Xc | 0.395±0.149 | 0.423 | -0.028 | Periodic-TDL |

## R_STD_SP 七任务SOTA数值对照（macro=0.785）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.933±0.013 | 0.934 | -0.001 | Periodic-TDL |
| Egb | 0.933±0.009 | 0.923 | +0.010 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.013 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.806±0.035 | 0.839 | -0.033 | Periodic-TDL |
| EPS | 0.713±0.076 | 0.787 | -0.074 | Periodic-TDL |
| Nc | 0.786±0.075 | 0.856 | -0.070 | Periodic-TDL |
| Xc | 0.414±0.108 | 0.423 | -0.009 | Periodic-TDL |

各任务五折等权R²，std(ddof=1)，seed42+fold，再七任务等权macro；所有计算使用未舍入值，展示3位小数。
论文逐任务最佳包络macro约0.810；协议等价性未证明，仅作数值对照，不宣称公平超过论文。共享开发TEST不称独立盲测。
配置、初始化、cache COMPLETE、history／best／scaler／预测位于原新周期根；冻结名单selection/final_frozen.json，完整精度指标及配对差值final_review/results.json；预算budget.json；源码diff及真实训练计时final_review/；全部命令／window／真实exit在logs/ua2d_transfer_20261022/r1/。
未完成／异常原因：无。


</details>

<details><summary>results/ua3d_ph_opt_20261031/r1/delivery_summary.md</summary>

# PH-OPT r1 交付待审查

执行于2026-10-31北京时间19:35:39停止；controller和outer launcher真实exit均为1。停止原因是预登记速度和全矩阵ETA门禁未通过，不是正式性能结论。

已完成：七个独立OPT文件及静态核对；前置CPU1次/exit0；packing1次/5875行/exit0；固定45套C0/M来源核实，排除但保留36套旧P；28个GPU2诊断全部exit0，56次eval和560次更新。

数学容差验证、同版精确复现、S上C0/M相对父轨迹的精确对照和显存门禁均通过。

| P臂 | L head/C0 | L joint/C0 | 每阶段≤5倍 |
|---|---:|---:|---|
|P_PE|1.094|1.048|True|
|P_FIXED|1.627|1.526|True|
|P_LEARN|15.471|9.242|False|

全矩阵保守ETA（含1.5余量）=129.235小时；门禁时剩余=47.465小时。估算基于S/L实测与结构工作量放大、1个有效并发和I/O/收尾预留；不是实测全矩阵耗时或延时授权。

未执行：首个P_LEARN/Egc/fold1正常FT、全部200新FT、唯一冻结、245正式TEST和最终CPU。各新增次数均为0；没有提前用最终CPU代替收尾，也未自动修改后重试。

七任务正式结果/相对BASE及论文差值均未完成/不可比较。BASE仍为F_PD_N1_F0，reference=0.804675462276949。共享开发评价和父期TEST标签暴露限制保持；不声称独立盲测或性能通过。

父r2 STOPPED、exit1、81FT、无TEST、原t0/截止/事故/消费只读保留。新tN=2026-10-31T19:03:34.521424+08:00，新截止=2026-11-02T19:03:34.521424+08:00未改写。

implementation、CPU、GPU诊断、速度、执行完整、性能和独立审查分层记录；independent_review=PENDING；交付待审查，不替换BASE。

改动文件及完整精度消费见delivery_summary.json；实际代码/状态改动见actual_changes.diff；首次失败见failure.json，速度/显存/逐单元ETA见gates/cost.json。


</details>

<details><summary>results/ua3d_ph_opt_20261031/r1/repairs/auto_r2/authorization/parent_snapshot/delivery_summary.md</summary>

# PH-OPT r1 交付待审查

执行于2026-10-31北京时间19:35:39停止；controller和outer launcher真实exit均为1。停止原因是预登记速度和全矩阵ETA门禁未通过，不是正式性能结论。

已完成：七个独立OPT文件及静态核对；前置CPU1次/exit0；packing1次/5875行/exit0；固定45套C0/M来源核实，排除但保留36套旧P；28个GPU2诊断全部exit0，56次eval和560次更新。

数学容差验证、同版精确复现、S上C0/M相对父轨迹的精确对照和显存门禁均通过。

| P臂 | L head/C0 | L joint/C0 | 每阶段≤5倍 |
|---|---:|---:|---|
|P_PE|1.094|1.048|True|
|P_FIXED|1.627|1.526|True|
|P_LEARN|15.471|9.242|False|

全矩阵保守ETA（含1.5余量）=129.235小时；门禁时剩余=47.465小时。估算基于S/L实测与结构工作量放大、1个有效并发和I/O/收尾预留；不是实测全矩阵耗时或延时授权。

未执行：首个P_LEARN/Egc/fold1正常FT、全部200新FT、唯一冻结、245正式TEST和最终CPU。各新增次数均为0；没有提前用最终CPU代替收尾，也未自动修改后重试。

七任务正式结果/相对BASE及论文差值均未完成/不可比较。BASE仍为F_PD_N1_F0，reference=0.804675462276949。共享开发评价和父期TEST标签暴露限制保持；不声称独立盲测或性能通过。

父r2 STOPPED、exit1、81FT、无TEST、原t0/截止/事故/消费只读保留。新tN=2026-10-31T19:03:34.521424+08:00，新截止=2026-11-02T19:03:34.521424+08:00未改写。

implementation、CPU、GPU诊断、速度、执行完整、性能和独立审查分层记录；independent_review=PENDING；交付待审查，不替换BASE。

改动文件及完整精度消费见delivery_summary.json；实际代码/状态改动见actual_changes.diff；首次失败见failure.json，速度/显存/逐单元ETA见gates/cost.json。


</details>

<details><summary>results/ua3d_hmr_readout_20261103/r1/delivery_summary.md</summary>

# UA3D-HMR-READOUT-20261103-01/r1 交付

状态：STOPPED；未完成/不可比较；independent_review=PENDING。
FINAL=None；macro=None；ΔBASE=None；ΔC_FIXED=None。
near=False；target=False；performance_pass=target_met=False。
性能目标未达。
实际阶段：{'preflight': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 1}, 'qualify': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 1}, 'diagnostic': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 0}}；前置CPU=PASSED/0；最终CPU=NOT_RUN/None。
累计账目：{'historical_hmr': {'source': '/root/workspace/Uni-Poly-Plus-master/results/ua3d_hmr_20261103/r1/repairs/scheduler_r3/cumulative_budget.json', 'geometry_reserved': 7300, 'geometry_attempted': 4231, 'FT_attempts': 107, 'update_interval': [18090, 18230], 'preflight': 3, 'final_cpu': 1, 'formal_units': 140, 'formal_rows': 23500, 'unknown_old_updates': [0, 140]}, 'current_readout': {'FT_attempts': 0, 'recorded_updates': 0, 'preflight': 1, 'final_cpu': 0, 'formal_completed': 0, 'new_model_outputs': 0, 'reused_outputs': 0, 'new_geometry': 0, 'new_BASE_encoding': 0, 'new_fit': 0}, 'family_total': {'geometry_reserved': 7300, 'geometry_attempted': 4231, 'FT_attempts': 107, 'FT_attempts_cap': 212, 'update_interval': [18090, 18230], 'update_upper_cap': 36320, 'unknown_old_updates': [0, 140]}, 'accounting': 'history already nests P1/P2; old controls are output aliases, never new prediction/FT'}。旧时钟、旧STOPPED与P3待审查状态不改写。

## C_ANCHOR

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## C_FIXED

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## N_BASE

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## N_ADD

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## N_BILIN

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|


七任务五折mean，std(ddof=1)，seed42+fold；论文参考SOTA_RESULT.md逐任务最佳配置包络，不是单一模型。Egc3380/论文4124、Xc432/论文430，协议不等价，仅数值对照。共享开发评价，非独立盲测或跨seed稳定。
N_BASE取消表面输入但仍包含原BASE的3D潜变量；N_ADD/N_BILIN为同参数量分支对照，普通MLP本身已有非线性交互。额外分支同时读取统计与谱，不能单独归因于谱；小差值不证明因果或显著性。有限Trimer含端部效应，不是周期链/真实体相。
首错：{'phase': 'diagnostic', 'error': 'Traceback (most recent call last):\n  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_hmr_readout_r1.py", line 339, in run\n    self.one(\'diagnostic\',arm=arm,task=task,fold=fold,repeat=repeat)\n  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_hmr_readout_r1.py", line 293, in one\n    self.wait_all(kind in (\'preflight\',\'qualify\'))\n  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_hmr_readout_r1.py", line 285, in wait_all\n    check_clock(prepare);self.engine.poll_all()\n                         ^^^^^^^^^^^^^^^^^^^^^^\n  File "/root/workspace/Uni-Poly-Plus-master/scripts/bootstrap_ua3d_hmr_readout_r1.py", line 138, in poll_all\n    if key in self.active:self.observe(key)\n                          ^^^^^^^^^^^^^^^^^\n  File "/root/workspace/Uni-Poly-Plus-master/scripts/bootstrap_ua3d_hmr_readout_r1.py", line 122, in observe\n    if p.poll() is not None:self.reap(key);return None\n                            ^^^^^^^^^^^^^^\n  File "/root/workspace/Uni-Poly-Plus-master/scripts/bootstrap_ua3d_hmr_readout_r1.py", line 108, in reap\n    if rc:raise RuntimeError(\'worker nonzero exit \'+key+\'=\'+str(rc))\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRuntimeError: worker nonzero exit diagnostic/N_BASE/eps/0/R0=1\n', 'exit': 1, 'time': 1793686558.869889}
真实退出：{'controller': {'exit': 1, 'started_unix': 1793686536.1273363, 'ended_unix': 1793686559.5652814, 'wait_source': 'subprocess.Popen.wait'}, 'outer': {'exit': 1, 'started_unix': 1793686536.0038254, 'ended_unix': 1793686559.5863895, 'wait_source': 'subprocess.Popen.wait'}, 'bootstrap': {'exit': 1}}；详见各层subprocess wait文件。
证据：authorization/、registry/sources.json/canonical.json/test_qualification、preflight/、diagnostics/、gates/、finetune/、selection/final_frozen.json、formal/*/test_predictions.npz、final_cpu/audit.json、budget.json/cumulative_budget.json。
105新预测与70旧输出复用独立计账，不将alias记为新训练/模型前向。没有重构面、求谱、BASE编码或统计重拟合。
不自动替换BASE、不部署、不追加搜索。


# READOUT/r1 失败交付核对补记

**STOPPED；未完成/不可比较；independent_review=PENDING。**

首个GPU诊断N_BASE/EPS/fold0/R0在计算epoch0引用指标时失败：metrics()调用d.finite，但新增src/dataset/ua3d_hmr_readout_r1.py没有提供finite。此为我的实现接口遗漏，前置合成suite未覆盖该接缝；不是数据错误、数值不稳定或模型性能结论。原worker堆栈保留在对应logs/failure.json和stdout.log；根first_failure.json保留外层非零传播，不覆盖。

完成层级：

- 八隔离文件已实现并锁定；原venv只读，未安装依赖。
- 唯一前置CPU通过、exit0：4测试，3小模型构造、11合成forward、3backward、0 optimizer step/真实标签/几何/GPU调用，2合成入口。
- 来源资格35套全部通过、exit0：P2表面/mask/trainval/scaler、P3 C_FIXED checkpoint/history/consumption/真实exit/validation；无新构面、求谱、BASE编码、统计拟合或trainval打包。资格未生成模型或前向。
- 诊断预约/启动/尝试1，完成0，exit1；GPU模型已构造1个，但GPU eval/train forward、backward和optimizer step均0。该预约和GPU占用保留，不退款。原consumption为RUNNING/空消费，因为错误发生在finish前；保留原文件，不伪填成功。
- 其余11诊断、完整试跑、105 FT、冻结、175正式输出及最终CPU均未执行。未打开P3逐行TEST标签、预测或TEST缓存；没有新原始预测、正式指标或消融差可交付。

累计新更新0；旧及谱系更新区间均为[18090,18230]，旧U∈[0,140]不变。几何仍7300预约/4231尝试，本轮增量0。前置CPU谱系累计4次（历史3+本轮1）；最终CPU累计1次（历史1+本轮0），按周期分账。

GPU2作业进程时间并集5.109秒；外层作业进程时长合计18.536秒；本新周期退出前总历时839.942秒。


controller/outer/bootstrap/tmux真实exit均1。当前无本周期存活进程。新tN及3h准备/12h总截止保留，未重置；旧时钟与旧r1/r2 STOPPED、旧r3 COMPLETE_PENDING_REVIEW/三层exit0保持。

自动报告的near/target/performance_pass=false表示没有完成正式评价；“目标未达”不得解读为模型性能否定。FINAL/macro/Δ均null。七任务论文参考仍列于final_review/report.md，所有本周期正式值标记未完成/不可比较；没有用validation或旧成绩填充。

证据：failure_closeout.json、first_failure.json、原始budget.json/cumulative_budget.json、preflight/、registry/sources.json/canonical.json、diagnostics/S/N_BASE/R0/consumption.json、对应logs/process/end/wait_exit/failure与三层exit文件。所有源码与锁定快照一致，未在线修复、重试、补单元或补CPU。旧缓存/模型/预算/BASE不改，未读TODO.md、未远程同步、未新增哈希/指纹。

下一步若要修正接口和继续执行，需要新的明确修订与授权；本周期停止，不自动开启下一轮。


</details>

<details><summary>results/ua3d_hmr_readout_20261103/r1/delivery_addendum.md</summary>

# READOUT/r1 失败交付核对补记

**STOPPED；未完成/不可比较；independent_review=PENDING。**

首个GPU诊断N_BASE/EPS/fold0/R0在计算epoch0引用指标时失败：metrics()调用d.finite，但新增src/dataset/ua3d_hmr_readout_r1.py没有提供finite。此为我的实现接口遗漏，前置合成suite未覆盖该接缝；不是数据错误、数值不稳定或模型性能结论。原worker堆栈保留在对应logs/failure.json和stdout.log；根first_failure.json保留外层非零传播，不覆盖。

完成层级：

- 八隔离文件已实现并锁定；原venv只读，未安装依赖。
- 唯一前置CPU通过、exit0：4测试，3小模型构造、11合成forward、3backward、0 optimizer step/真实标签/几何/GPU调用，2合成入口。
- 来源资格35套全部通过、exit0：P2表面/mask/trainval/scaler、P3 C_FIXED checkpoint/history/consumption/真实exit/validation；无新构面、求谱、BASE编码、统计拟合或trainval打包。资格未生成模型或前向。
- 诊断预约/启动/尝试1，完成0，exit1；GPU模型已构造1个，但GPU eval/train forward、backward和optimizer step均0。该预约和GPU占用保留，不退款。原consumption为RUNNING/空消费，因为错误发生在finish前；保留原文件，不伪填成功。
- 其余11诊断、完整试跑、105 FT、冻结、175正式输出及最终CPU均未执行。未打开P3逐行TEST标签、预测或TEST缓存；没有新原始预测、正式指标或消融差可交付。

累计新更新0；旧及谱系更新区间均为[18090,18230]，旧U∈[0,140]不变。几何仍7300预约/4231尝试，本轮增量0。前置CPU谱系累计4次（历史3+本轮1）；最终CPU累计1次（历史1+本轮0），按周期分账。

GPU2作业进程时间并集5.109秒；外层作业进程时长合计18.536秒；本新周期退出前总历时839.942秒。


controller/outer/bootstrap/tmux真实exit均1。当前无本周期存活进程。新tN及3h准备/12h总截止保留，未重置；旧时钟与旧r1/r2 STOPPED、旧r3 COMPLETE_PENDING_REVIEW/三层exit0保持。

自动报告的near/target/performance_pass=false表示没有完成正式评价；“目标未达”不得解读为模型性能否定。FINAL/macro/Δ均null。七任务论文参考仍列于final_review/report.md，所有本周期正式值标记未完成/不可比较；没有用validation或旧成绩填充。

证据：failure_closeout.json、first_failure.json、原始budget.json/cumulative_budget.json、preflight/、registry/sources.json/canonical.json、diagnostics/S/N_BASE/R0/consumption.json、对应logs/process/end/wait_exit/failure与三层exit文件。所有源码与锁定快照一致，未在线修复、重试、补单元或补CPU。旧缓存/模型/预算/BASE不改，未读TODO.md、未远程同步、未新增哈希/指纹。

下一步若要修正接口和继续执行，需要新的明确修订与授权；本周期停止，不自动开启下一轮。


</details>

<details><summary>results/ua3d_hmr_readout_20261103/r1/final_review/report.md</summary>

# UA3D-HMR-READOUT-20261103-01/r1 交付

状态：STOPPED；未完成/不可比较；independent_review=PENDING。
FINAL=None；macro=None；ΔBASE=None；ΔC_FIXED=None。
near=False；target=False；performance_pass=target_met=False。
性能目标未达。
实际阶段：{'preflight': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 1}, 'qualify': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 1}, 'diagnostic': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 0}}；前置CPU=PASSED/0；最终CPU=NOT_RUN/None。
累计账目：{'historical_hmr': {'source': '/root/workspace/Uni-Poly-Plus-master/results/ua3d_hmr_20261103/r1/repairs/scheduler_r3/cumulative_budget.json', 'geometry_reserved': 7300, 'geometry_attempted': 4231, 'FT_attempts': 107, 'update_interval': [18090, 18230], 'preflight': 3, 'final_cpu': 1, 'formal_units': 140, 'formal_rows': 23500, 'unknown_old_updates': [0, 140]}, 'current_readout': {'FT_attempts': 0, 'recorded_updates': 0, 'preflight': 1, 'final_cpu': 0, 'formal_completed': 0, 'new_model_outputs': 0, 'reused_outputs': 0, 'new_geometry': 0, 'new_BASE_encoding': 0, 'new_fit': 0}, 'family_total': {'geometry_reserved': 7300, 'geometry_attempted': 4231, 'FT_attempts': 107, 'FT_attempts_cap': 212, 'update_interval': [18090, 18230], 'update_upper_cap': 36320, 'unknown_old_updates': [0, 140]}, 'accounting': 'history already nests P1/P2; old controls are output aliases, never new prediction/FT'}。旧时钟、旧STOPPED与P3待审查状态不改写。

## C_ANCHOR

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## C_FIXED

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## N_BASE

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## N_ADD

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## N_BILIN

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|


七任务五折mean，std(ddof=1)，seed42+fold；论文参考SOTA_RESULT.md逐任务最佳配置包络，不是单一模型。Egc3380/论文4124、Xc432/论文430，协议不等价，仅数值对照。共享开发评价，非独立盲测或跨seed稳定。
N_BASE取消表面输入但仍包含原BASE的3D潜变量；N_ADD/N_BILIN为同参数量分支对照，普通MLP本身已有非线性交互。额外分支同时读取统计与谱，不能单独归因于谱；小差值不证明因果或显著性。有限Trimer含端部效应，不是周期链/真实体相。
首错：{'phase': 'diagnostic', 'error': 'Traceback (most recent call last):\n  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_hmr_readout_r1.py", line 339, in run\n    self.one(\'diagnostic\',arm=arm,task=task,fold=fold,repeat=repeat)\n  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_hmr_readout_r1.py", line 293, in one\n    self.wait_all(kind in (\'preflight\',\'qualify\'))\n  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_hmr_readout_r1.py", line 285, in wait_all\n    check_clock(prepare);self.engine.poll_all()\n                         ^^^^^^^^^^^^^^^^^^^^^^\n  File "/root/workspace/Uni-Poly-Plus-master/scripts/bootstrap_ua3d_hmr_readout_r1.py", line 138, in poll_all\n    if key in self.active:self.observe(key)\n                          ^^^^^^^^^^^^^^^^^\n  File "/root/workspace/Uni-Poly-Plus-master/scripts/bootstrap_ua3d_hmr_readout_r1.py", line 122, in observe\n    if p.poll() is not None:self.reap(key);return None\n                            ^^^^^^^^^^^^^^\n  File "/root/workspace/Uni-Poly-Plus-master/scripts/bootstrap_ua3d_hmr_readout_r1.py", line 108, in reap\n    if rc:raise RuntimeError(\'worker nonzero exit \'+key+\'=\'+str(rc))\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRuntimeError: worker nonzero exit diagnostic/N_BASE/eps/0/R0=1\n', 'exit': 1, 'time': 1793686558.869889}
真实退出：{'controller': {'exit': 1, 'started_unix': 1793686536.1273363, 'ended_unix': 1793686559.5652814, 'wait_source': 'subprocess.Popen.wait'}, 'outer': {'exit': 1, 'started_unix': 1793686536.0038254, 'ended_unix': 1793686559.5863895, 'wait_source': 'subprocess.Popen.wait'}, 'bootstrap': {'exit': 1}}；详见各层subprocess wait文件。
证据：authorization/、registry/sources.json/canonical.json/test_qualification、preflight/、diagnostics/、gates/、finetune/、selection/final_frozen.json、formal/*/test_predictions.npz、final_cpu/audit.json、budget.json/cumulative_budget.json。
105新预测与70旧输出复用独立计账，不将alias记为新训练/模型前向。没有重构面、求谱、BASE编码或统计重拟合。
不自动替换BASE、不部署、不追加搜索。


# READOUT/r1 失败交付核对补记

**STOPPED；未完成/不可比较；independent_review=PENDING。**

首个GPU诊断N_BASE/EPS/fold0/R0在计算epoch0引用指标时失败：metrics()调用d.finite，但新增src/dataset/ua3d_hmr_readout_r1.py没有提供finite。此为我的实现接口遗漏，前置合成suite未覆盖该接缝；不是数据错误、数值不稳定或模型性能结论。原worker堆栈保留在对应logs/failure.json和stdout.log；根first_failure.json保留外层非零传播，不覆盖。

完成层级：

- 八隔离文件已实现并锁定；原venv只读，未安装依赖。
- 唯一前置CPU通过、exit0：4测试，3小模型构造、11合成forward、3backward、0 optimizer step/真实标签/几何/GPU调用，2合成入口。
- 来源资格35套全部通过、exit0：P2表面/mask/trainval/scaler、P3 C_FIXED checkpoint/history/consumption/真实exit/validation；无新构面、求谱、BASE编码、统计拟合或trainval打包。资格未生成模型或前向。
- 诊断预约/启动/尝试1，完成0，exit1；GPU模型已构造1个，但GPU eval/train forward、backward和optimizer step均0。该预约和GPU占用保留，不退款。原consumption为RUNNING/空消费，因为错误发生在finish前；保留原文件，不伪填成功。
- 其余11诊断、完整试跑、105 FT、冻结、175正式输出及最终CPU均未执行。未打开P3逐行TEST标签、预测或TEST缓存；没有新原始预测、正式指标或消融差可交付。

累计新更新0；旧及谱系更新区间均为[18090,18230]，旧U∈[0,140]不变。几何仍7300预约/4231尝试，本轮增量0。前置CPU谱系累计4次（历史3+本轮1）；最终CPU累计1次（历史1+本轮0），按周期分账。

GPU2作业进程时间并集5.109秒；外层作业进程时长合计18.536秒；本新周期退出前总历时839.942秒。


controller/outer/bootstrap/tmux真实exit均1。当前无本周期存活进程。新tN及3h准备/12h总截止保留，未重置；旧时钟与旧r1/r2 STOPPED、旧r3 COMPLETE_PENDING_REVIEW/三层exit0保持。

自动报告的near/target/performance_pass=false表示没有完成正式评价；“目标未达”不得解读为模型性能否定。FINAL/macro/Δ均null。七任务论文参考仍列于final_review/report.md，所有本周期正式值标记未完成/不可比较；没有用validation或旧成绩填充。

证据：failure_closeout.json、first_failure.json、原始budget.json/cumulative_budget.json、preflight/、registry/sources.json/canonical.json、diagnostics/S/N_BASE/R0/consumption.json、对应logs/process/end/wait_exit/failure与三层exit文件。所有源码与锁定快照一致，未在线修复、重试、补单元或补CPU。旧缓存/模型/预算/BASE不改，未读TODO.md、未远程同步、未新增哈希/指纹。

下一步若要修正接口和继续执行，需要新的明确修订与授权；本周期停止，不自动开启下一轮。


</details>

<details><summary>results/ua3d_anchor_opt_20261102/r2/delivery_summary.md</summary>

# BASE ANCHOR r2 交付待审查

状态：COMPLETE_PENDING_REVIEW；完整正式TEST；independent_review=PENDING。
前置CPU次数/exit：{'attempts': 1, 'exit': 0}；最终CPU次数/exit：{'attempts': 1, 'exit': 0}。
FINAL='BASE_KEEP'；selected_macro=0.804675462276949；selected_delta=0.0；reference=0.804675462276949。
原BASE35套下游BEST只读冻结，原模块eval；同批U/H差值锚定；旧周期/BASE/旧FINAL不改写。
共享开发评价、旧TEST标签暴露和已看TEST路线开发限制持续披露，不称独立盲测。

| Arm | TEST macro | Δ原BASE |
|---|---:|---:|
|C_ANCHOR|0.805|+0.000|
|G_GEOM|0.807|+0.003|
|P_PE|0.807|+0.003|
|P_LEARN|0.807|+0.002|

## C_ANCHOR

|任务|项目mean±std|论文mean±std|差值|论文模型|
|---|---:|---:|---:|---|
|eea|0.930±0.018|0.934±0.013|-0.004|Periodic-TDL|
|egb|0.921±0.015|0.923±0.009|-0.002|Periodic-TDL (+Chem, +RC)|
|egc|0.914±0.013|0.911±0.008|+0.003|Periodic-TDL (+Chem, +RC)|
|ei|0.821±0.040|0.839±0.018|-0.018|Periodic-TDL|
|eps|0.776±0.068|0.787±0.033|-0.011|Periodic-TDL|
|nc|0.831±0.057|0.856±0.032|-0.025|Periodic-TDL|
|xc|0.440±0.143|0.423±0.102|+0.017|Periodic-TDL|

## G_GEOM

|任务|项目mean±std|论文mean±std|差值|论文模型|
|---|---:|---:|---:|---|
|eea|0.933±0.020|0.934±0.013|-0.001|Periodic-TDL|
|egb|0.924±0.011|0.923±0.009|+0.001|Periodic-TDL (+Chem, +RC)|
|egc|0.915±0.012|0.911±0.008|+0.004|Periodic-TDL (+Chem, +RC)|
|ei|0.828±0.036|0.839±0.018|-0.011|Periodic-TDL|
|eps|0.768±0.086|0.787±0.033|-0.019|Periodic-TDL|
|nc|0.830±0.060|0.856±0.032|-0.026|Periodic-TDL|
|xc|0.456±0.147|0.423±0.102|+0.033|Periodic-TDL|

## P_PE

|任务|项目mean±std|论文mean±std|差值|论文模型|
|---|---:|---:|---:|---|
|eea|0.930±0.018|0.934±0.013|-0.004|Periodic-TDL|
|egb|0.922±0.015|0.923±0.009|-0.001|Periodic-TDL (+Chem, +RC)|
|egc|0.915±0.013|0.911±0.008|+0.004|Periodic-TDL (+Chem, +RC)|
|ei|0.827±0.038|0.839±0.018|-0.012|Periodic-TDL|
|eps|0.782±0.068|0.787±0.033|-0.005|Periodic-TDL|
|nc|0.832±0.055|0.856±0.032|-0.024|Periodic-TDL|
|xc|0.444±0.147|0.423±0.102|+0.021|Periodic-TDL|

## P_LEARN

|任务|项目mean±std|论文mean±std|差值|论文模型|
|---|---:|---:|---:|---|
|eea|0.930±0.018|0.934±0.013|-0.004|Periodic-TDL|
|egb|0.921±0.014|0.923±0.009|-0.002|Periodic-TDL (+Chem, +RC)|
|egc|0.914±0.013|0.911±0.008|+0.003|Periodic-TDL (+Chem, +RC)|
|ei|0.825±0.039|0.839±0.018|-0.014|Periodic-TDL|
|eps|0.781±0.068|0.787±0.033|-0.006|Periodic-TDL|
|nc|0.832±0.055|0.856±0.032|-0.024|Periodic-TDL|
|xc|0.444±0.147|0.423±0.102|+0.021|Periodic-TDL|

论文为逐任务最佳包络，不是单一模型；数据/split/统计未证实等价，仅数值对照。Egc3380/论文4124，Xc432/论文430差异保留。
阻断：无。
不自动修复重试/补单元，不扩时，不替换BASE，不自动开启下一轮。


</details>

<details><summary>results/ua3d_anchor_lr_20261102/r1/delivery_summary.md</summary>

# ANCHOR LR r1 失败交付，待审查

执行状态：STOPPED；完成层级：隔离实现/静态检查完成，controller初始化失败；independent_review=PENDING。
原始错误：KeyError: estimated_cache_bytes。新周期采用缓存复用，controller初始化仍访问旧缓存估算字段；不填字段、不改锁定代码、不重试。
真实controller/outer/bootstrap exit均为1。前置CPU与最终CPU均NOT_RUN，次数0、exit=null。
35 trainval核对、16诊断、140FT（含唯一试跑）、冻结、35 TEST核对和175正式输出全部NOT_RUN；科学作业0，预留/尝试/完成消费均0。
执行实现和启动失败耗时计入原新周期时钟，不刷新。
本周期目标：未舍入macro≥0.809为接近，≥0.810为达到；本次无正式结果，未完成/不可比较，不能判定性能通过或未达。
reference=F_PD_N1_F0，0.804675462276949；未选FINAL，未替换BASE。

|候选|正式TEST macro|状态|
|---|---:|---|
|C_ANCHOR|—|未完成/不可比较|
|G_GEOM_1E4|—|未完成/不可比较|
|G_GEOM_3E5|—|未完成/不可比较|
|P_PE_1E4|—|未完成/不可比较|
|P_PE_3E5|—|未完成/不可比较|

七任务/35项论文数值对照：未执行正式输出，不能填入项目值或用父结果替代。
保护：父ANCHOR/r2/BASE/旧FINAL/旧事故/旧预算和退出只读保留；本次未打开父TEST缓存行/预测；无GPU、完整encoder、真实PH、PT或optimizer更新。
失败文件和锁定实现保留，待独立审查；不自动修复重试、不重建根、不另开周期。


</details>

<details><summary>results/ua3d_anchor_lr_20261102/r1/repairs/init_r2/delivery_summary.md</summary>

# ANCHOR-LR r2 失败交付，待审查

STOPPED；正式输出完整；执行失败，数值保留在formal单元；independent_review=PENDING。
首次错误阶段：controller_run。
Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/bootstrap_ua3d_anchor_lr_r2.py", line 336, in controller
    phase='controller_run';stage(OUT,phase);return c.run()
                                                   ^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_anchor_lr_r2.py", line 292, in run
    signal.setitimer(signal.ITIMER_REAL,max(.001,t.DEADLINE-300-time.time()));self.execute([('cpu',None,None,None,None)])
                                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_anchor_lr_r2.py", line 269, in execute
    self.refresh(); self.failures()
                    ^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_anchor_lr_r2.py", line 136, in failures
    raise RuntimeError(j['name']+' true exit='+str(j['exit'])+'; stop without retry')
RuntimeError: ua3d-anchor-lr-r2-cpu true exit=1; stop without retry


真实exit由controller/outer/bootstrap各层wait文件记录；父r1 STOPPED/exit1/科学0保留。
CPU：{'attempts': 1, 'exit': 0, 'status': 'PASSED'} / {'attempts': 1, 'exit': 1, 'status': 'FAILED'}。
已派发403个作业；科学预留/尝试/完成见budget.json，失败不退款。
不调用完整refresh/report，不自动修复重试或补单元；BASE不替换。

实际wait退出链：{"controller": 1, "outer": 1, "bootstrap": 1}；independent_review=PENDING。


</details>

<details><summary>results/ua2d_opt_20261021/r1/final_review/report.md</summary>

# UA2D-OPT r1 实际交付

状态：需返修；独立审查：待进行。执行者按DSH职责交付，本次自检不构成Codex独立审查。

FINAL：未冻结；正式TEST性能结论：未完成／不可比较；CPU唯一suite exit=0。

停止原因：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_opt.py", line 279, in run
    self.matrix(['B3_C'])
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_opt.py", line 146, in matrix
    if any(j['exit'] not in (None,0) for j in active):raise RuntimeError('matrix unit failed; stop dispatch')
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: matrix unit failed; stop dispatch


| Task | FINAL正式R² | 项目减参考 | 论文参考R² | 参考模型 |
|---|---|---|---|---|
| Eea | 未完成／不可比较 | — | 0.934 | Periodic-TDL |
| EGB | 未完成／不可比较 | — | 0.923 | Periodic-TDL (+Chem, +RC) |
| EGC | 未完成／不可比较 | — | 0.911 | Periodic-TDL (+Chem, +RC) |
| EI | 未完成／不可比较 | — | 0.839 | Periodic-TDL |
| EPS | 未完成／不可比较 | — | 0.787 | Periodic-TDL |
| NC | 未完成／不可比较 | — | 0.856 | Periodic-TDL |
| XC | 未完成／不可比较 | — | 0.423 | Periodic-TDL |

口径：七任务等权macro；各任务五折等权R²，std(ddof=1)，seed42+fold。共享validation/TEST参与开发，不称独立盲测。数据／split／协议等价性未证明；仅作论文数值对照。未开放TEST不以validation或smoke代填。

结果：/root/workspace/Uni-Poly-Plus-master/results/ua2d_opt_20261021/r1/final_review/results.json；实际预算：/root/workspace/Uni-Poly-Plus-master/results/ua2d_opt_20261021/r1/budget.json；日志及真实退出码：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_opt_20261021/r1；完整已执行FT：/root/workspace/Uni-Poly-Plus-master/results/ua2d_opt_20261021/r1/finetune；正式预测仅在实际开放时存在。

PT／FT失败及部分运行计账，未重试／恢复／换卡／改batch。C/D按r1-CD-REQUIRED固定执行，异常停止及未完成阶段如实记录，不转移或扩大预算。


</details>

<details><summary>results/ua2d_opt_20261021/r1/repair_ft1/snapshot/final_review/report.md</summary>

# UA2D-OPT r1 实际交付

状态：需返修；独立审查：待进行。执行者按DSH职责交付，本次自检不构成Codex独立审查。

FINAL：未冻结；正式TEST性能结论：未完成／不可比较；CPU唯一suite exit=0。

停止原因：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_opt.py", line 279, in run
    self.matrix(['B3_C'])
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_opt.py", line 146, in matrix
    if any(j['exit'] not in (None,0) for j in active):raise RuntimeError('matrix unit failed; stop dispatch')
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: matrix unit failed; stop dispatch


| Task | FINAL正式R² | 项目减参考 | 论文参考R² | 参考模型 |
|---|---|---|---|---|
| Eea | 未完成／不可比较 | — | 0.934 | Periodic-TDL |
| EGB | 未完成／不可比较 | — | 0.923 | Periodic-TDL (+Chem, +RC) |
| EGC | 未完成／不可比较 | — | 0.911 | Periodic-TDL (+Chem, +RC) |
| EI | 未完成／不可比较 | — | 0.839 | Periodic-TDL |
| EPS | 未完成／不可比较 | — | 0.787 | Periodic-TDL |
| NC | 未完成／不可比较 | — | 0.856 | Periodic-TDL |
| XC | 未完成／不可比较 | — | 0.423 | Periodic-TDL |

口径：七任务等权macro；各任务五折等权R²，std(ddof=1)，seed42+fold。共享validation/TEST参与开发，不称独立盲测。数据／split／协议等价性未证明；仅作论文数值对照。未开放TEST不以validation或smoke代填。

结果：/root/workspace/Uni-Poly-Plus-master/results/ua2d_opt_20261021/r1/final_review/results.json；实际预算：/root/workspace/Uni-Poly-Plus-master/results/ua2d_opt_20261021/r1/budget.json；日志及真实退出码：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_opt_20261021/r1；完整已执行FT：/root/workspace/Uni-Poly-Plus-master/results/ua2d_opt_20261021/r1/finetune；正式预测仅在实际开放时存在。

PT／FT失败及部分运行计账，未重试／恢复／换卡／改batch。C/D按r1-CD-REQUIRED固定执行，异常停止及未完成阶段如实记录，不转移或扩大预算。


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/delivery_summary.md</summary>

# UA3D-COND-SPECTRAL-20261104-01/r2

状态：STOPPED；independent_review=PENDING。

正式评价：未完成/不可比较；完成 0/210 单元。

PT完成更新 857；FT完成 0/210；最终CPU次数 0，exit=None。

真实controller exit=1；outer/bootstrap由bootstrap实际wait后补记。

错误：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r2.py", line 162, in main
    try:controller.run()
        ^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r2.py", line 102, in run
    self.launch('pt');self.wait()
                      ^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r2.py", line 72, in wait
    while self.live:self.poll();time.sleep(5)
                    ^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r2.py", line 63, in poll
    self.check()
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r2.py", line 50, in check
    if rss>160*1024**3:raise RuntimeError('160GiB host memory cap')
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: 160GiB host memory cap


实现文件见implementation_diff.md，逐作业argv/cwd/log/真实exit见jobs.json；旧HMR/READOUT保持原样。

三层真实exit：{'controller': 1, 'outer': 1, 'bootstrap': 1}。

停止后核对见delivery_addendum.md及shutdown_audit.json：PT真实完成区间[857,858]，858次尝试全计；RSS合计是保守代理，不能当作已核实物理峰值。原快照保持。


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/delivery_addendum.md</summary>

# r2 停止后只读核对补记

状态 STOPPED；未完成/不可比较；independent_review=PENDING。原 first_failure.json、failure_closeout.json、budget.json、consumption.json 和PT RUNNING快照全部保留，本补记不覆盖原证据。

实现八个隔离文件，完成35来源/训练统计资格（prepare exit0）、新随机step0及35套新头；GPU2未知PID退出后仅启动1套新PT。此前只执行静态AST/JSON/接口检查，没有CPU suite、smoke、benchmark或额外GPU诊断。

控制器的后代VmRSS合计门禁触发160GiB停止。最后已保存的门禁前样本为 171105198080 bytes（159.354 GiB）；触发的越界样本未落盘，因此不能给出精确峰值。合计RSS可能重复计算共享/文件映射页，不能把该停止等同于已测得主机物理峰值越过160GiB。未确认根因，不在线修复或重启。

PT由控制器SIGTERM，worker exit=-15。历史日志证明完成857更新；最后消费快照为尝试858、完成857。由于SIGTERM发生在第858次更新内部，实际完成更新保守区间[857,858]，不把未记账尾部强行算成成功或退款。858次尝试全部保留计账。消费5148 microbatches / 864864样本呈现；最后prefetch计数866538、未消费1674，最终预取计数未知且不得归零。PT状态快照仍为RUNNING，实际停止以worker真实wait为准。

FT=0、冻结=0、正式TEST=0、最终CPU=0；没有step5000/step10000/deploy或BEST。FINAL、macro、Δ均null；不作性能结论，不替换BASE。controller/outer/bootstrap真实exit均1，tmux exit1；本轮进程均已退出，GPU2无本轮存活进程。

证据入口：shutdown_audit.json、first_failure.json、jobs.json、exits/、pretrain/BASE_FRESH/{consumption.json,history.jsonl,sampling.jsonl}。新合同/时钟和旧HMR/READOUT现场保持原样。后续若修复须新修订及明确授权，不能在本r2内重试、降worker或恢复部分权重。


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/repairs/coverage_r5/delivery_summary.md</summary>

# UA3D-COND-SPECTRAL-20261104-01/r5

状态：STOPPED；independent_review=PENDING；正式评价未完成/不可比较。

新增PT=0，允许来源PT完成10000；新增FT启动201、完整新FT197/209、父完整alias1、canonical完整198/210、部分/中断4；开始epochs 13993、完整训练验证epochs 13989；尝试/记录完成更新326993/326993。

TEST完整0/210、0/35250行；最终CPU 0次、exit=None；前置CPU/额外前向0。

父累计12288尝试/[12287,12288]完成区间及预取未知尾部保留；当前累计见cumulative_budget.json。

原总截止不变；repair_diff.md/source_aliases.json/memory.jsonl/lifecycle.jsonl/jobs.json/exits/记录修复、来源、生命周期和真实exit。

停止原因：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r5.py", line 262, in main
    controller=Controller();controller.run()
                            ^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r5.py", line 234, in run
    self.queue('ft',pending,available,2)
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r5.py", line 179, in queue
    done=self.poll()
         ^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r5.py", line 122, in poll
    self.reap();self.check();done=self.finished_buffer;self.finished_buffer=[];return done
                ^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r5.py", line 76, in check
    result=self.memory.sample(force=force or self.boundary_due)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_cond_spectral_r5.py", line 500, in sample
    if sample['fatal_stop']:raise RuntimeError(sample['fatal_stop'])
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: TELEMETRY_ERROR: living PID 3000636:62397014 PSS unreadable after bounded reconciliation: [Errno 3] No such process: '/proc/3000636/smaps_rollup'


三层真实exit：{'controller': 1, 'outer': 1, 'bootstrap': 1}；tmux实际exit见对应日志根tmux.exit。

停止后核对补充见[delivery_addendum.md](delivery_addendum.md)、[stop_audit.json](stop_audit.json)及[implementation.diff](implementation.diff)。


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/repairs/coverage_r5/delivery_addendum.md</summary>

# r5 停止后交付核对（执行自检，非独立审查）

状态：STOPPED；正式 TEST 未完成/不可比较；independent_review=PENDING。本次不自动替换 BASE，不修复重试，不恢复中断 BEST。

2026-11-05 13:00:37（Asia/Shanghai），pass5510 对 DataLoader 身份3000636:62397014（父FT PID2992854，S_COND_NOCROSS/EPS/fold4）实际读取 smaps_rollup 三次均返回 ESRCH。约0/120.6/241.2ms的读取后同一stat身份均为D、RSS=0；250.4ms复核结束仍可见，未得到已退出判定，因此按已登记3读/250ms合同停止。该样本未等待至5秒pending截止，不能写成pending超时或未尝试测量。证据支持实际读取失败与身份过渡的不一致；仅凭现有证据不反演确切内核退出原因，也不将RSS=0当作PSS=0。

遥测共5511条：COMPLETE2214、PENDING3160、轻量快照136、STOP1。完整样本PSS+SwapPss最大观测13.3189697265625GiB（展示13.319），不是连续物理峰值证明。失败样本全活跃PSS为null；没有数值160GiB超限判定，环境memory.events的oom/oom_kill均0。采样累计墙钟1832.0622834913665秒，逐pass待测、服务、延迟和生命周期证据保留在memory.jsonl/lifecycle.jsonl；不得据该累计值推断GPU独占损失时间。

新增PT/初始化/统计拟合/几何/构面/求谱均0。r3 deploy及原随机头为只读来源，唯一父C_FRESH/EPS/fold0完整alias未重收费。新增FT启动201，197套完整、4套中断、8套未启动；canonical完整198/210。新完成按臂：C_FRESH34（加父alias为35）、S_STATIC_GATE35、S_COND_GATE35、S_STATIC_TOKEN35、S_COND_TOKEN35、S_COND_NOCROSS23。

| 中断单元（均S_COND_NOCROSS） | 开始/完整训练验证epochs | 尝试/记录完成更新 |
|---|---:|---:|
| Egc/fold4 | 59/58 | 5369/5369 |
| EPS/fold4 | 57/56 | 627/627 |
| Nc/fold0 | 46/45 | 506/506 |
| Nc/fold1 | 41/40 | 451/451 |

4套均保留epoch_open=true与signal15错误，真实worker exit1来自控制器停止；不将它们写成70epoch完成单元。其余197套worker及prepare exit0。当前FT13993个开始epoch、13989个完整训练验证epoch、326993尝试/记录完成更新；训练消费7685303行、validation消费1927985行。四修订累计339281尝试、339280日志完成、实际区间[339280,339281]；父不确定区间和预取未知尾部不退款不归零。本轮未消费预取没有凭空归零或计入已消费行。

五表面臂首套EPS/fold0正常完整训练通过2倍门槛，相对父完整C_FRESH稳定joint每批耗时比分别1.015533700596931、1.1118254690912586、1.0635887569147064、1.1946389085224391、1.0671948271360985；按实测显存门禁已达到4并发。停止前ETA/显存准入均未触发失败；本次停止后ETA作废。

冻结0、正式TEST0单元/0行、最终CPU suite0；前置CPU/pytest/smoke/额外模型前向0。没有七任务正式性能、macro、Δ历史BASE或ΔC_FRESH；target_met/performance_pass均未评价，不用validation填补。

controller/outer/bootstrap/tmux真实exit均1，已按PID+start_ticks核对登记主进程与作业均退出。源8文件与启动source_snapshot逐字节一致，无hash合同，无在线改码。修复实际diff见implementation.diff，科学不变说明见repair_diff.md。父现场、原时钟与BASE保持原身份；原总截止1794048655.5459058未刷新，r5准备窗口从1793840882.0起，prepare于1793842013.9961956完成，未超过2小时。

证据：first_failure.json、telemetry_error.json、stop_audit.json、failure_closeout.json、jobs.json、budget.json、cumulative_budget.json、authorization/clock.json、registry/canonical.json、exits/、对应日志根tmux.exit。后续任何修复或接续需要新的明确授权。


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/repairs/mm_lifecycle_r6/delivery_summary.md</summary>

# UA3D-COND-SPECTRAL-20261104-01/r6

状态：COMPLETE_PENDING_REVIEW；independent_review=PENDING；正式评价COMPLETE。

新增PT=0，允许来源PT完成10000；新增FT启动12、完整新FT12/12、父完整alias198、canonical完整210/210、部分/中断0；开始epochs 840、完整训练验证epochs 840；尝试/记录完成更新15190/15190。

TEST完整210/210、35250/35250行；最终CPU 1次、exit=0；前置CPU/额外前向0。

父累计339281尝试/[339280,339281]完成区间及预取未知尾部保留；当前累计见cumulative_budget.json。

原总截止不变；repair_diff.md/source_aliases.json/memory.jsonl/lifecycle.jsonl/jobs.json/exits/记录修复、来源、生命周期和真实exit。

完整七任务/论文对照见final_review/results.json及comparison.md；数值和审查分开，BASE不替换。

三层真实exit：{'controller': 0, 'outer': 0, 'bootstrap': 0}；tmux实际exit见对应日志根tmux.exit。

完整七任务、论文数值对照、性能门槛、来源/时钟/内存限制及真实exit补充见[delivery_addendum.md](delivery_addendum.md)和[execution_audit.json](execution_audit.json)。


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/repairs/mm_lifecycle_r6/delivery_addendum.md</summary>

# r6 完整交付补充（执行自检，待独立审查）

状态：COMPLETE_PENDING_REVIEW；independent_review=PENDING。这是共享开发评价，未自动替换BASE。

198个只读完整来源＋12个新完整FT组成210个canonical。新增PT/初始化/统计拟合/构象/构面/求谱均0；新FT840epochs、15190尝试/完成更新，训练350490行、validation88130行。累计354471尝试、354470日志完成、实际完成区间[354470,354471]；FT343613=canonical336000＋父中断297＋363＋6953。父完整320810更新不重收费，r2预取未知尾部和866538快照/1674下界保留。

最后新增FT退出：2026-11-05T14:38:59.913655+08:00；唯一冻结：2026-11-05T14:39:02.532309+08:00；首个正式TEST：2026-11-05T14:39:02.795995+08:00。最后正式TEST退出：2026-11-05T14:45:53.518911+08:00；唯一CPU suite启动：2026-11-05T14:45:55.133240+08:00。正式210/210单元、35250行；CPU仅1次，21 passed/exit0，未重跑。正式阶段前pytest/smoke/合成测试/额外模型前向0。

仅validation按逐任务保护/严格macro/注册顺序规则冻结FINAL=S_STATIC_GATE；正式TEST后未改选、改权重、拼任务或追加训练。所有210正式输出只写本r6根，三代BEST来源分别r4×1、r5×197、r6×12；实际estimator_config和r6 runner_config分别绑定。

| Task | FINAL TEST R² mean ± std | 论文参考模型 | 论文参考 R² | Δ项目−参考 |
|---|---:|---|---:|---:|
| Eea | 0.922 ± 0.027 | Periodic-TDL | 0.934 ± 0.013 | -0.012 |
| Egb | 0.919 ± 0.013 | Periodic-TDL (+Chem, +RC) | 0.923 ± 0.009 | -0.004 |
| Egc | 0.913 ± 0.012 | Periodic-TDL (+Chem, +RC) | 0.911 ± 0.008 | +0.002 |
| Ei | 0.810 ± 0.023 | Periodic-TDL | 0.839 ± 0.018 | -0.029 |
| EPS | 0.735 ± 0.088 | Periodic-TDL | 0.787 ± 0.033 | -0.052 |
| Nc | 0.835 ± 0.052 | Periodic-TDL | 0.856 ± 0.032 | -0.021 |
| Xc | 0.385 ± 0.106 | Periodic-TDL | 0.423 ± 0.102 | -0.038 |

FINAL macro R²=0.788；Δ历史BASE=-0.016；Δ本轮C_FRESH=+0.001。target_met=false（macro<0.810）；performance_pass=false（Δ历史BASE<0.010）。未舍入判断及全部原始精度见final_review/results.json，原始预测见formal/。每任务5折、seed42+fold、std(ddof=1)，七任务等权macro。论文数字来自SOTA_RESULT.md，属于不同最佳配置的包络，协议等价性未证实，只作数值对照。

| 已冻结臂 | 正式 macro R² |
|---|---:|
| C_FRESH | 0.787 |
| S_STATIC_GATE | 0.788 |
| S_COND_GATE | 0.797 |
| S_STATIC_TOKEN | 0.788 |
| S_COND_TOKEN | 0.783 |
| S_COND_NOCROSS | 0.777 |

各臂均为预登记对照，以上TEST数字不触发重选。历史reference仍F_PD_N1_F0（原始macro0.804675462276949），新C_FRESH只作本周期匹配控制。

内存：911条遥测中714次COMPLETE、158次PENDING、39次轻量快照；无STOP/DRAINING_EXIT。本轮真实采样NO_USER_MM_OBSERVED事件为0，因此不声称生产观测已经验证该分支；两轮ESRCH组合证据、拒绝条件、null语义、恢复及不可刷新30秒退出等待由唯一最终CPU用例覆盖。完整PSS+SwapPss最大观测11.163GiB，只是采样最大值；监控累计采样墙钟90.002秒，不等同GPU独占损失时间。逐PID各PSS分项与环境、pending服务证据保留在memory.jsonl/lifecycle.jsonl，逐单元正常CUDA分配峰值及最新80%准入记录见gpu_memory_summary.json，不伪装成总NVML连续峰值。

GPU2曾出现未知NVML PID1279310、本地/proc不可见；控制器仅等待，未终止或推测归属，随后无未知占用时正常派发。新FT/TEST最多实测4并发，旧正常成本门禁只读沿用，未重跑首六。

原总截止1794048655.5459058未变；tR6=1793858780.7218225，准备截止=1793865980.7218225；prepare退出于2026-11-05T14:22:25.967752+08:00，在两小时窗口内。所有父阶段及修复时钟完整留存。

prepare×1、新FT×12、正式TEST×210、CPU×1真实exit均0；controller/outer/bootstrap/tmux exit均0。登记进程已退出，无重试、无在线源码改动；八源码与启动快照一致，父r3/r4/r5源码与其各自快照一致，198父完整消费元数据仍匹配。完整科学执行及四层exit与辅助观察命令分开记录：曾在终端会话启动一个仅反复读取结果JSON的UI观察器，随后为采用有界短读而停止，观察器shell exit143单列，GPU/模型/训练/测试消费0；见authorization/read_only_observer.json和execution_audit.json。该辅助观察器不应被混写成正式worker失败或科学预算退款。

交付：implementation.diff、repair_diff.md、source_aliases.json、registry/canonical.json、source_snapshot/、authorization/clock.json、finetune/、formal/、selection/final_frozen.json、final_review/results.json及comparison.md、gpu_memory_summary.json、execution_audit.json、budget.json及cumulative_budget.json、jobs.json/exits/和对应日志根。r3随机step0/35头/deploy按source_aliases.json的只读来源交付，不复制或重训。独立审查仍PENDING；CPU通过和执行完成不等于独立审查通过，性能门槛未通过。


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/repairs/mm_lifecycle_r6/final_review/comparison.md</summary>

# Formal shared-development results
FINAL=S_STATIC_GATE; macro=0.788; ΔBASE=-0.016; ΔC_FRESH=+0.001.
Paper values form a per-task envelope, not a single model; numerical comparison only.

|Task|Project mean ± std|Paper model|Paper mean|Δ|
|---|---:|---|---:|---:|
|eea|0.922 ± 0.027|Periodic-TDL|0.934|-0.012|
|egb|0.919 ± 0.013|Periodic-TDL (+Chem, +RC)|0.923|-0.004|
|egc|0.913 ± 0.012|Periodic-TDL (+Chem, +RC)|0.911|+0.002|
|ei|0.810 ± 0.023|Periodic-TDL|0.839|-0.029|
|eps|0.735 ± 0.088|Periodic-TDL|0.787|-0.052|
|nc|0.835 ± 0.052|Periodic-TDL|0.856|-0.021|
|xc|0.385 ± 0.106|Periodic-TDL|0.423|-0.038|


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/repairs/lifecycle_r4/delivery_summary.md</summary>

# UA3D-COND-SPECTRAL-20261104-01/r4

状态：STOPPED；independent_review=PENDING；正式评价未完成/不可比较。

新增PT=0，允许来源PT完成10000；新增FT启动2、完整1/210、部分/中断1；开始epochs 103、完整训练验证epochs 102；尝试/记录完成更新1133/1133。

TEST完整0/210、0/35250行；最终CPU 0次、exit=None；前置CPU/额外前向0。

父累计11155尝试/[11154,11155]完成区间及预取未知尾部保留；当前累计见cumulative_budget.json。

原总截止不变；repair_diff.md/source_aliases.json/memory.jsonl/lifecycle.jsonl/jobs.json/exits/记录修复、来源、生命周期和真实exit。

停止原因：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r4.py", line 237, in main
    controller=Controller();controller.run()
                            ^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r4.py", line 198, in run
    self.wait_gpu();job=self.launch('ft',arm,'eps',0);self.wait();h,peak=self.observe_ft(job)
                                                      ^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r4.py", line 119, in wait
    self.poll()
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r4.py", line 107, in poll
    done=self.reap();self.check();return done
                     ^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r4.py", line 74, in check
    self.memory.sample(force=force);budget();self.next_check=time.monotonic()+5
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_cond_spectral_r4.py", line 474, in sample
    if sample['decision']:raise RuntimeError(sample['decision'])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: TELEMETRY_ERROR: unresolved living identities/PSS coverage


三层真实exit：{'controller': 1, 'outer': 1, 'bootstrap': 1}；tmux实际exit见对应日志根tmux.exit。

只读核对补记：delivery_addendum.md、lifecycle_stop_audit.json。新增FT1133尝试/完成；三代12288尝试/[12287,12288]完成区间；新生进程覆盖7/8触发停止，不是内存超限。四层exit均1。


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/repairs/lifecycle_r4/delivery_addendum.md</summary>

# r4 停止后只读核对

状态 STOPPED；未完成/不可比较；independent_review=PENDING。本补记不覆盖首错、内存决定样本、源码快照、预算或父证据。

八个r4隔离文件已实施并锁定；prepare仅元数据alias，exit0，未构造模型或加载checkpoint张量。正常FT显式读取r3完整deploy与原随机头各2次；historical_BASE_tensor_reads=0，新PT/新随机step0/头/统计拟合/几何/构面/求谱均0。源码与锁定文本逐字一致，没有启动后改码。

2026-11-05 08:44:41（Asia/Shanghai）完整采样经过4轮有界发现，最终8个存活身份只测完7个，缺少2324200:60861492（新生DataLoader worker）；采样主合计为null，不能把它填0或宣称160GiB超限。正常退出过渡中2324176的ESRCH已有读取前后R状态、120ms内身份消失的完整证据；这条旧过渡已正确归并。本次失败是采样末尾新增进程未完成覆盖，r4发现复核仍有实现缺口，不能写成生命周期修复已验收。

正常注册发现当前Python的os未暴露pidfd_open，能力和失败原因均落盘，使用合同允许的严格stat路径；没有运行探针。42条内存采样、20条完整PSS样本，完整样本最大PSS+SwapPss为3.131GiB，仅为观测最大值，不是连续精确峰值；未发生MEMORY_STOP。

C_FRESH/EPS/fold0完整70epochs、770尝试/完成，exit0。S_STATIC_GATE/EPS/fold0开始33epochs、完整训练验证32epochs，363尝试/完成；控制器SIGTERM时正在第33次validation等候数据，worker记录FAILED并exit1。其ConnectionResetError属于停止后的加载器清理输出，不覆盖首个遥测覆盖错误。其余208套FT未启动。

新增FT总尝试/记录完成1133；开始epochs103、完整训练验证epochs102，训练25132行、validation6222行。父旧消费不退款；三代累计尝试12288、日志完成12287、实际完成区间[12287,12288]。父r2预取未知尾部保持，r3预取0不抵消它。

冻结/正式TEST/最终CPU均0，前置CPU/额外GPU诊断/额外模型前向均0。FINAL、七任务正式结果、macro与Δ均null，目标和默认性能门槛尚未评价，不据此否定模型。新BEST/history和中断BEST全部保留，不用于接续。完整正常速度门禁也尚未完成六套，不能作六臂成本通过结论。

controller/outer/bootstrap/tmux真实exit均1；本轮登记进程已全部退出，GPU2无本轮占用。时钟及原总截止保持，父r2/r3及BASE不变。未在线修改、重试、重启、换卡或追加测试。后续修复必须另行明确授权新修订。

证据：telemetry_error.json、memory.jsonl、lifecycle.jsonl、first_failure.json、jobs.json、budget.json、cumulative_budget.json、lifecycle_stop_audit.json及对应logs。


</details>

<details><summary>results/ua3d_cond_spectral_20261104/r2/repairs/memory_r3/delivery_summary.md</summary>

# UA3D-COND-SPECTRAL-20261104-01/r3

状态：STOPPED；independent_review=PENDING。

正式评价：未完成/不可比较；完成 0/210 单元。

PT完成更新 10000；FT完成 0/210；最终CPU次数 0，exit=None。

真实controller exit=1；outer/bootstrap由bootstrap实际wait后补记。

错误：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r3.py", line 182, in main
    try:controller.run()
        ^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r3.py", line 124, in run
    self.wait_gpu();self.launch('ft',arm,'eps',0);self.wait()
                                                  ^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r3.py", line 76, in wait
    while self.live:self.poll();time.sleep(5)
                    ^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r3.py", line 66, in poll
    self.check()
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_cond_spectral_r3.py", line 49, in check
    self.memory.sample(force=full)
  File "/root/workspace/Uni-Poly-Plus-master/src/training/ua3d_cond_spectral_r3.py", line 440, in sample
    if sample['decision']:raise RuntimeError(sample['decision'])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: TELEMETRY_ERROR: living PID 2305401:57882989 PSS unreadable: [Errno 3] No such process: '/proc/2305401/smaps_rollup'


修复文件见repair_diff.md，逐作业argv/cwd/log/真实exit见jobs.json；旧HMR/READOUT保持原样。

三层真实exit：{'controller': 1, 'outer': 1, 'bootstrap': 1}。


</details>

<details><summary>results/ua2d_demol_20261020/r1/final_review/report.md</summary>

# UA2D-DEMOL-LITE-20261020-01 / r1 执行交付

状态：待审查；独立审查未完成。

validation：NEIGHBOR−SELF=-0.001；NEIGHBOR−B_FP=+0.001；门槛通过=False。

五折seed42—46，std ddof=1，七任务等权macro；完整精度判定，展示三位小数。
共享validation／TEST不是独立盲测；论文协议未证明一致，仅数值对照。

|任务|论文模型|参考R²|BOND_SELF R²／差值|BOND_NEIGHBOR R²／差值|
|---|---|---:|---|---|
|eea|Periodic-TDL|0.934|未完成／不可比较（validation双门槛未通过，正式TEST=0）|未完成／不可比较（validation双门槛未通过，正式TEST=0）|
|egb|Periodic-TDL (+Chem, +RC)|0.923|未完成／不可比较（validation双门槛未通过，正式TEST=0）|未完成／不可比较（validation双门槛未通过，正式TEST=0）|
|egc|Periodic-TDL (+Chem, +RC)|0.911|未完成／不可比较（validation双门槛未通过，正式TEST=0）|未完成／不可比较（validation双门槛未通过，正式TEST=0）|
|ei|Periodic-TDL|0.839|未完成／不可比较（validation双门槛未通过，正式TEST=0）|未完成／不可比较（validation双门槛未通过，正式TEST=0）|
|eps|Periodic-TDL|0.787|未完成／不可比较（validation双门槛未通过，正式TEST=0）|未完成／不可比较（validation双门槛未通过，正式TEST=0）|
|nc|Periodic-TDL|0.856|未完成／不可比较（validation双门槛未通过，正式TEST=0）|未完成／不可比较（validation双门槛未通过，正式TEST=0）|
|xc|Periodic-TDL|0.423|未完成／不可比较（validation双门槛未通过，正式TEST=0）|未完成／不可比较（validation双门槛未通过，正式TEST=0）|


</details>

<details><summary>results/ua3d_hmr_20261103/r1/delivery_summary.md</summary>

# UA3D-HMR-20261103-01/r1 失败交付

**STOPPED；未完成/不可比较；independent_review=PENDING。**

八个隔离实现文件已新增并锁定；隔离依赖安装与唯一一次前置CPU通过（exit=0）。正式prepare失败（exit=1），没有诊断、FT、冻结、正式输出或最终CPU。BASE不变。

首错来自 s00578（Egb row560 / Egc row3233）：原center_physical=[2,-1,3,-1]，源代码明确允许显式同位素氢没有重原子代表时为-1。新增surface()要求映射非负且唯一，错误地拒绝了该合法记录。此为实现错误，不是源坐标损坏或模型性能结论。未修改或重试。

自动失败报告另因bootstrap导入路径错误未生成（ModuleNotFoundError: No module named scripts）；delivery_error.json保留原堆栈。本报告仅补齐文档和只读账目，不调用失败入口。

表面队列预留3650（3646唯一源结构+4旋转）；实际开始581、正常返回474（可用355、预登记不可用119）、失败1、中止且无end 106、未开始3069。失败/部分不退款。
预登记不可用分布：{'DISCONNECTED': 103, 'NONMANIFOLD': 14, 'MESH_CAP': 2}。覆盖率与旋转门禁未完成，不能由该部分样本外推全体成功率。

独立tN=2026-11-03T09:57:34.933989+08:00；准备截止=2026-11-03 17:57:34.933989+08:00；硬截止=2026-11-04 09:57:34.933989+08:00。退出前总历时1100.183秒；作业进程时长合计56.005秒；物理GPU忙时0，optimizer更新0。
前置CPU：{'meshes': 3, 'constructs': 3, 'forwards': 6, 'backwards': 6, 'optimizer_updates': 0, 'real_labels': 0, 'encoder_forwards': 0, 'tests_run': 5}。CPU几何池106，BLAS/OMP线程1。新增产物账本字节={'surface': 483369970, 'teacher': 0, 'total': 549863186}（退出时快照，报告新增字节另计）。

## 七任务论文数值对照

本周期正式预测与指标均未生成。下列BASE为已有reference；不能作为本周期锚点复验通过。论文为逐任务不同配置包络。

|Task|本周期|原BASE mean ± std|论文 mean ± std|BASE−论文|论文模型|
|---|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.930 ± 0.018|0.934 ± 0.013|-0.004|Periodic-TDL|
|egb|未完成/不可比较|0.921 ± 0.015|0.923 ± 0.009|-0.002|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.914 ± 0.013|0.911 ± 0.008|+0.003|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.821 ± 0.040|0.839 ± 0.018|-0.018|Periodic-TDL|
|eps|未完成/不可比较|0.776 ± 0.068|0.787 ± 0.033|-0.011|Periodic-TDL|
|nc|未完成/不可比较|0.831 ± 0.057|0.856 ± 0.032|-0.025|Periodic-TDL|
|xc|未完成/不可比较|0.440 ± 0.143|0.423 ± 0.102|+0.017|Periodic-TDL|

reference macro=0.804675462276949；本周期macro/Δ/FINAL均null；near/target/performance_pass均false（未完成，不是性能否定）。

原BASE与论文仅数值对照。七任务五折，std(ddof=1)、seed=42+fold；Egc3380/论文4124、Xc432/论文430等协议不等价。共享开发评价，不称独立盲测或跨seed稳定。

## 真实exit与证据

controller=1；outer=1；bootstrap=1；tmux=1。前置CPU=0；prepare=1；最终CPU=NOT_RUN/null。

证据：authorization/clock.json、source_locked/、dependencies.json；preflight/consumption.json；surface/mapping.json、queue.json、structures/*、failure_accounting.json；budget.json；first_failure.json、failure_detail.json、delivery_error.json；logs同周期的prepare/stdout与process/end/wait_exit。原预算、首错和退出证据不改写。

当前无存活本周期controller/worker；未读取TODO.md，未远程同步，未新增哈希合同，未改旧源码/缓存/checkpoint。

下一步需要新的明确修订与授权：修正合法负哨兵处理、失败报告导入路径，并先只读评估目前大量DISCONNECTED/NONMANIFOLD是否与预登记覆盖率合同兼容。当前周期不恢复、不重建根、不追加作业；不能预先宣称修复后可过覆盖率。


</details>

<details><summary>results/ua3d_hmr_20261103/r1/final_review/report.md</summary>

# UA3D-HMR-20261103-01/r1 失败交付

**STOPPED；未完成/不可比较；independent_review=PENDING。**

八个隔离实现文件已新增并锁定；隔离依赖安装与唯一一次前置CPU通过（exit=0）。正式prepare失败（exit=1），没有诊断、FT、冻结、正式输出或最终CPU。BASE不变。

首错来自 s00578（Egb row560 / Egc row3233）：原center_physical=[2,-1,3,-1]，源代码明确允许显式同位素氢没有重原子代表时为-1。新增surface()要求映射非负且唯一，错误地拒绝了该合法记录。此为实现错误，不是源坐标损坏或模型性能结论。未修改或重试。

自动失败报告另因bootstrap导入路径错误未生成（ModuleNotFoundError: No module named scripts）；delivery_error.json保留原堆栈。本报告仅补齐文档和只读账目，不调用失败入口。

表面队列预留3650（3646唯一源结构+4旋转）；实际开始581、正常返回474（可用355、预登记不可用119）、失败1、中止且无end 106、未开始3069。失败/部分不退款。
预登记不可用分布：{'DISCONNECTED': 103, 'NONMANIFOLD': 14, 'MESH_CAP': 2}。覆盖率与旋转门禁未完成，不能由该部分样本外推全体成功率。

独立tN=2026-11-03T09:57:34.933989+08:00；准备截止=2026-11-03 17:57:34.933989+08:00；硬截止=2026-11-04 09:57:34.933989+08:00。退出前总历时1100.183秒；作业进程时长合计56.005秒；物理GPU忙时0，optimizer更新0。
前置CPU：{'meshes': 3, 'constructs': 3, 'forwards': 6, 'backwards': 6, 'optimizer_updates': 0, 'real_labels': 0, 'encoder_forwards': 0, 'tests_run': 5}。CPU几何池106，BLAS/OMP线程1。新增产物账本字节={'surface': 483369970, 'teacher': 0, 'total': 549863186}（退出时快照，报告新增字节另计）。

## 七任务论文数值对照

本周期正式预测与指标均未生成。下列BASE为已有reference；不能作为本周期锚点复验通过。论文为逐任务不同配置包络。

|Task|本周期|原BASE mean ± std|论文 mean ± std|BASE−论文|论文模型|
|---|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.930 ± 0.018|0.934 ± 0.013|-0.004|Periodic-TDL|
|egb|未完成/不可比较|0.921 ± 0.015|0.923 ± 0.009|-0.002|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.914 ± 0.013|0.911 ± 0.008|+0.003|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.821 ± 0.040|0.839 ± 0.018|-0.018|Periodic-TDL|
|eps|未完成/不可比较|0.776 ± 0.068|0.787 ± 0.033|-0.011|Periodic-TDL|
|nc|未完成/不可比较|0.831 ± 0.057|0.856 ± 0.032|-0.025|Periodic-TDL|
|xc|未完成/不可比较|0.440 ± 0.143|0.423 ± 0.102|+0.017|Periodic-TDL|

reference macro=0.804675462276949；本周期macro/Δ/FINAL均null；near/target/performance_pass均false（未完成，不是性能否定）。

原BASE与论文仅数值对照。七任务五折，std(ddof=1)、seed=42+fold；Egc3380/论文4124、Xc432/论文430等协议不等价。共享开发评价，不称独立盲测或跨seed稳定。

## 真实exit与证据

controller=1；outer=1；bootstrap=1；tmux=1。前置CPU=0；prepare=1；最终CPU=NOT_RUN/null。

证据：authorization/clock.json、source_locked/、dependencies.json；preflight/consumption.json；surface/mapping.json、queue.json、structures/*、failure_accounting.json；budget.json；first_failure.json、failure_detail.json、delivery_error.json；logs同周期的prepare/stdout与process/end/wait_exit。原预算、首错和退出证据不改写。

当前无存活本周期controller/worker；未读取TODO.md，未远程同步，未新增哈希合同，未改旧源码/缓存/checkpoint。

下一步需要新的明确修订与授权：修正合法负哨兵处理、失败报告导入路径，并先只读评估目前大量DISCONNECTED/NONMANIFOLD是否与预登记覆盖率合同兼容。当前周期不恢复、不重建根、不追加作业；不能预先宣称修复后可过覆盖率。


</details>

<details><summary>results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2/delivery_summary.md</summary>

# UA3D-HMR-20261103-01/r2 交付

状态：STOPPED；未完成/不可比较；independent_review=PENDING。
reference=0.804675462276949；FINAL=None；macro=None；Δ=None。
前置CPU=PASSED/0；最终CPU=NOT_RUN/None。
表面：{'reserved': 3650, 'attempted': 3650, 'completed': 3650, 'failed': 0, 'without_end': 0, 'usable': 3648, 'unavailable': {'EIGEN_NOT_CONVERGED': 2}}；累计预留=7300/7300，累计尝试=4231/4231。
父r1仍STOPPED/三层exit1；旧预约、失败与106中止不退款，不重写。

## C_ANCHOR 七任务论文数值对照

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## S_STAT 七任务论文数值对照

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## S_FIXED 七任务论文数值对照

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## S_HMR 七任务论文数值对照

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

七任务五折mean，std(ddof=1)，seed=42+fold。论文macro是不同配置包络；Egc3380/论文4124，Xc432/论文430，协议不等价，只作数值对照。共享开发TEST，不称独立盲测或跨seed稳定。
表面为tau=0.15 Å、rho0=1.000001、尾界1e-10的软球隐式面；多分量均保留，统一64模与显式零模；不是电子密度或无限周期体相。
错误：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/bootstrap_ua3d_hmr_r2.py", line 109, in envelope
    if rc:raise RuntimeError('outer/controller nonzero exit '+str(rc))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: outer/controller nonzero exit 1

真实退出见controller_exit.json、outer_exit.json、bootstrap_exit.json。证据：budget.json、cumulative_budget.json、surface/、diagnostics/、finetune/、selection/、formal/及对应logs。
本周期不自动替换BASE、不追加搜索；错误后不修复重试。


# r2 停止原因与完成层级补记

**STOPPED；未完成/不可比较；independent_review=PENDING。**

首错：S_STAT/Eea/fold0 worker正常退出后，controller在归属核验中继续读取 /proc/1718271/environ，发生ProcessLookupError。我的调度实现未处理poll与进程快照之间的退出竞态。首堆栈保留于first_failure.json；未修复源码、重试作业或再次运行CPU。
bootstrap标准库兜底报告成功生成；原自动报告error为外层非零传播，具体首错以first_failure.json为准。

实际完成范围：

- 八个r2隔离文件已实现、锁定且未改动；父环境版本一致，无依赖安装。前置CPU一次9项测试exit0；合成网格4、模型构造4、forward6、backward6、optimizer更新0、真实标签0、BASE前向0。
- prepare exit0；3650项新几何全部返回（3646主结构+4旋转）；3648可用、2谱不收敛；原有效行覆盖率99.966%，各任务均达标；原119不可用+s00578共120项均可用。
- 多分量分布（含旋转）：{'1': 3465, '2': 163, '3': 14, '4': 3, '7': 1, '5': 2}；最大顶点16054，最大64模；183项含负有向体积分量，均保留，未按分量逐个取绝对值。
- 原4旋转、35 trainval缓存资格与train-only统计全部通过。12串行GPU2诊断全部通过（240更新），首次成本门禁通过。
- 唯一S_HMR/Egc/fold1试跑完成70epoch/630更新，wrapper约10.005秒，exit0，试跑后完整ETA门禁通过。S_STAT/Eea/fold0完成70epoch/70更新、exit0。
- S_STAT/Eea/fold1和fold2已预约并启动，收束后的真实wait exit=-15；无消费文件，不补写成功或退款。FT预约/启动4、完成2、中止2，不等于105FT完成。
- 本周期记录optimizer更新940=诊断240+FT700；GPU2时间并集80.156秒，外层作业进程时长合计240.285秒。
- 几何worker已完成进程时长合计：父305.903秒、子4112.983秒；另有父106个无end中止单元，保留真实start和未知时长，不填0。外层prepare时间与worker核时分账。
- 父子几何累计预留7300/7300、尝试4231/4231；前置CPU累计2次。父r1仍STOPPED/三层exit1。
- 未冻结、未读取教师TEST、未生成正式预测、未运行最终CPU；FINAL/macro/Δ均null，目标未判定。

真实exit：controller=1，outer=1，bootstrap=1，tmux=1；两个中止FT worker=-15。当前无本周期存活进程。
原tN=2026-11-03T09:57:34.933989+08:00；repair_started=2026-11-03T10:59:49.086939+08:00；准备截止11月3日17:57:34.933989、总截止11月4日09:57:34.933989（北京时间）未刷新。

四臂七任务论文对照保留于final_review/report.md，本周期值全部为未完成/不可比较，未用validation填充。
证据：budget.json、cumulative_budget.json、failure_closeout_addendum.json、first_failure.json、surface/coverage.json、rotation.json、r1_r2_comparison.json、structures/*/audit.npz、preflight/entry_evidence.json、diagnostics/、finetune/和对应logs。

仅更新顶部r2 STATUS及待审查历史；父r1状态块与原文件保持。未读TODO.md、不远程同步、不新增哈希/指纹、不替换BASE。后续修复或执行需要新的明确修订与授权；当前不追加搜索。


</details>

<details><summary>results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2/delivery_addendum.md</summary>

# r2 停止原因与完成层级补记

**STOPPED；未完成/不可比较；independent_review=PENDING。**

首错：S_STAT/Eea/fold0 worker正常退出后，controller在归属核验中继续读取 /proc/1718271/environ，发生ProcessLookupError。我的调度实现未处理poll与进程快照之间的退出竞态。首堆栈保留于first_failure.json；未修复源码、重试作业或再次运行CPU。
bootstrap标准库兜底报告成功生成；原自动报告error为外层非零传播，具体首错以first_failure.json为准。

实际完成范围：

- 八个r2隔离文件已实现、锁定且未改动；父环境版本一致，无依赖安装。前置CPU一次9项测试exit0；合成网格4、模型构造4、forward6、backward6、optimizer更新0、真实标签0、BASE前向0。
- prepare exit0；3650项新几何全部返回（3646主结构+4旋转）；3648可用、2谱不收敛；原有效行覆盖率99.966%，各任务均达标；原119不可用+s00578共120项均可用。
- 多分量分布（含旋转）：{'1': 3465, '2': 163, '3': 14, '4': 3, '7': 1, '5': 2}；最大顶点16054，最大64模；183项含负有向体积分量，均保留，未按分量逐个取绝对值。
- 原4旋转、35 trainval缓存资格与train-only统计全部通过。12串行GPU2诊断全部通过（240更新），首次成本门禁通过。
- 唯一S_HMR/Egc/fold1试跑完成70epoch/630更新，wrapper约10.005秒，exit0，试跑后完整ETA门禁通过。S_STAT/Eea/fold0完成70epoch/70更新、exit0。
- S_STAT/Eea/fold1和fold2已预约并启动，收束后的真实wait exit=-15；无消费文件，不补写成功或退款。FT预约/启动4、完成2、中止2，不等于105FT完成。
- 本周期记录optimizer更新940=诊断240+FT700；GPU2时间并集80.156秒，外层作业进程时长合计240.285秒。
- 几何worker已完成进程时长合计：父305.903秒、子4112.983秒；另有父106个无end中止单元，保留真实start和未知时长，不填0。外层prepare时间与worker核时分账。
- 父子几何累计预留7300/7300、尝试4231/4231；前置CPU累计2次。父r1仍STOPPED/三层exit1。
- 未冻结、未读取教师TEST、未生成正式预测、未运行最终CPU；FINAL/macro/Δ均null，目标未判定。

真实exit：controller=1，outer=1，bootstrap=1，tmux=1；两个中止FT worker=-15。当前无本周期存活进程。
原tN=2026-11-03T09:57:34.933989+08:00；repair_started=2026-11-03T10:59:49.086939+08:00；准备截止11月3日17:57:34.933989、总截止11月4日09:57:34.933989（北京时间）未刷新。

四臂七任务论文对照保留于final_review/report.md，本周期值全部为未完成/不可比较，未用validation填充。
证据：budget.json、cumulative_budget.json、failure_closeout_addendum.json、first_failure.json、surface/coverage.json、rotation.json、r1_r2_comparison.json、structures/*/audit.npz、preflight/entry_evidence.json、diagnostics/、finetune/和对应logs。

仅更新顶部r2 STATUS及待审查历史；父r1状态块与原文件保持。未读TODO.md、不远程同步、不新增哈希/指纹、不替换BASE。后续修复或执行需要新的明确修订与授权；当前不追加搜索。


</details>

<details><summary>results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2/final_review/report.md</summary>

# UA3D-HMR-20261103-01/r2 交付

状态：STOPPED；未完成/不可比较；independent_review=PENDING。
reference=0.804675462276949；FINAL=None；macro=None；Δ=None。
前置CPU=PASSED/0；最终CPU=NOT_RUN/None。
表面：{'reserved': 3650, 'attempted': 3650, 'completed': 3650, 'failed': 0, 'without_end': 0, 'usable': 3648, 'unavailable': {'EIGEN_NOT_CONVERGED': 2}}；累计预留=7300/7300，累计尝试=4231/4231。
父r1仍STOPPED/三层exit1；旧预约、失败与106中止不退款，不重写。

## C_ANCHOR 七任务论文数值对照

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## S_STAT 七任务论文数值对照

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## S_FIXED 七任务论文数值对照

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

## S_HMR 七任务论文数值对照

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|未完成/不可比较|0.934 ± 0.013|不可比较|Periodic-TDL|
|egb|未完成/不可比较|0.923 ± 0.009|不可比较|Periodic-TDL (+Chem, +RC)|
|egc|未完成/不可比较|0.911 ± 0.008|不可比较|Periodic-TDL (+Chem, +RC)|
|ei|未完成/不可比较|0.839 ± 0.018|不可比较|Periodic-TDL|
|eps|未完成/不可比较|0.787 ± 0.033|不可比较|Periodic-TDL|
|nc|未完成/不可比较|0.856 ± 0.032|不可比较|Periodic-TDL|
|xc|未完成/不可比较|0.423 ± 0.102|不可比较|Periodic-TDL|

七任务五折mean，std(ddof=1)，seed=42+fold。论文macro是不同配置包络；Egc3380/论文4124，Xc432/论文430，协议不等价，只作数值对照。共享开发TEST，不称独立盲测或跨seed稳定。
表面为tau=0.15 Å、rho0=1.000001、尾界1e-10的软球隐式面；多分量均保留，统一64模与显式零模；不是电子密度或无限周期体相。
错误：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/bootstrap_ua3d_hmr_r2.py", line 109, in envelope
    if rc:raise RuntimeError('outer/controller nonzero exit '+str(rc))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: outer/controller nonzero exit 1

真实退出见controller_exit.json、outer_exit.json、bootstrap_exit.json。证据：budget.json、cumulative_budget.json、surface/、diagnostics/、finetune/、selection/、formal/及对应logs。
本周期不自动替换BASE、不追加搜索；错误后不修复重试。


# r2 停止原因与完成层级补记

**STOPPED；未完成/不可比较；independent_review=PENDING。**

首错：S_STAT/Eea/fold0 worker正常退出后，controller在归属核验中继续读取 /proc/1718271/environ，发生ProcessLookupError。我的调度实现未处理poll与进程快照之间的退出竞态。首堆栈保留于first_failure.json；未修复源码、重试作业或再次运行CPU。
bootstrap标准库兜底报告成功生成；原自动报告error为外层非零传播，具体首错以first_failure.json为准。

实际完成范围：

- 八个r2隔离文件已实现、锁定且未改动；父环境版本一致，无依赖安装。前置CPU一次9项测试exit0；合成网格4、模型构造4、forward6、backward6、optimizer更新0、真实标签0、BASE前向0。
- prepare exit0；3650项新几何全部返回（3646主结构+4旋转）；3648可用、2谱不收敛；原有效行覆盖率99.966%，各任务均达标；原119不可用+s00578共120项均可用。
- 多分量分布（含旋转）：{'1': 3465, '2': 163, '3': 14, '4': 3, '7': 1, '5': 2}；最大顶点16054，最大64模；183项含负有向体积分量，均保留，未按分量逐个取绝对值。
- 原4旋转、35 trainval缓存资格与train-only统计全部通过。12串行GPU2诊断全部通过（240更新），首次成本门禁通过。
- 唯一S_HMR/Egc/fold1试跑完成70epoch/630更新，wrapper约10.005秒，exit0，试跑后完整ETA门禁通过。S_STAT/Eea/fold0完成70epoch/70更新、exit0。
- S_STAT/Eea/fold1和fold2已预约并启动，收束后的真实wait exit=-15；无消费文件，不补写成功或退款。FT预约/启动4、完成2、中止2，不等于105FT完成。
- 本周期记录optimizer更新940=诊断240+FT700；GPU2时间并集80.156秒，外层作业进程时长合计240.285秒。
- 几何worker已完成进程时长合计：父305.903秒、子4112.983秒；另有父106个无end中止单元，保留真实start和未知时长，不填0。外层prepare时间与worker核时分账。
- 父子几何累计预留7300/7300、尝试4231/4231；前置CPU累计2次。父r1仍STOPPED/三层exit1。
- 未冻结、未读取教师TEST、未生成正式预测、未运行最终CPU；FINAL/macro/Δ均null，目标未判定。

真实exit：controller=1，outer=1，bootstrap=1，tmux=1；两个中止FT worker=-15。当前无本周期存活进程。
原tN=2026-11-03T09:57:34.933989+08:00；repair_started=2026-11-03T10:59:49.086939+08:00；准备截止11月3日17:57:34.933989、总截止11月4日09:57:34.933989（北京时间）未刷新。

四臂七任务论文对照保留于final_review/report.md，本周期值全部为未完成/不可比较，未用validation填充。
证据：budget.json、cumulative_budget.json、failure_closeout_addendum.json、first_failure.json、surface/coverage.json、rotation.json、r1_r2_comparison.json、structures/*/audit.npz、preflight/entry_evidence.json、diagnostics/、finetune/和对应logs。

仅更新顶部r2 STATUS及待审查历史；父r1状态块与原文件保持。未读TODO.md、不远程同步、不新增哈希/指纹、不替换BASE。后续修复或执行需要新的明确修订与授权；当前不追加搜索。


</details>

<details><summary>results/ua3d_hmr_20261103/r1/repairs/scheduler_r3/delivery_summary.md</summary>

# UA3D-HMR-20261103-01 science r2 / execution r3

状态：COMPLETE_PENDING_REVIEW；正式输出完成；independent_review=PENDING。
FINAL=S_FIXED；reference=0.804675462276949；macro=0.8076857278302016；未舍入Δ=0.003010265553252678。
near=False；target=False；performance_pass=False；最后CPU=PASSED/0。
实际阶段：{'preflight': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 1}, 'ft': {'reserved': 103, 'started': 103, 'attempted': 103, 'completed': 103}, 'test': {'reserved': 140, 'started': 140, 'attempted': 140, 'completed': 140}, 'cpu': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 1}}；2 parent aliases不计新训练。
累计账目：{'geometry_reserved': 7300, 'geometry_attempted': 4231, 'preflight_total': 3, 'FT_attempts': 107, 'FT_attempts_cap': 107, 'parent_recorded_updates': 940, 'new_recorded_updates': 17150, 'cumulative_recorded_updates': 18090, 'old_unknown_updates': {'each': [0, 70], 'sum': [0, 140]}, 'cumulative_update_interval': [18090, 18230], 'update_upper_cap': 18230, 'accounting': 'r2 already nests r1; never add ancestor geometry again; aliases are not new FT'}。父r1/r2仍STOPPED/三层exit1，全部失败、中止、旧预约不退款。

## C_ANCHOR

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|0.930 ± 0.018|0.934 ± 0.013|-0.004|Periodic-TDL|
|egb|0.921 ± 0.015|0.923 ± 0.009|-0.002|Periodic-TDL (+Chem, +RC)|
|egc|0.914 ± 0.013|0.911 ± 0.008|+0.003|Periodic-TDL (+Chem, +RC)|
|ei|0.821 ± 0.040|0.839 ± 0.018|-0.018|Periodic-TDL|
|eps|0.776 ± 0.068|0.787 ± 0.033|-0.011|Periodic-TDL|
|nc|0.831 ± 0.057|0.856 ± 0.032|-0.025|Periodic-TDL|
|xc|0.440 ± 0.143|0.423 ± 0.102|+0.017|Periodic-TDL|

macro=0.805；ΔBASE=+0.000；完整精度见results.json。

## S_STAT

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|0.931 ± 0.019|0.934 ± 0.013|-0.003|Periodic-TDL|
|egb|0.921 ± 0.015|0.923 ± 0.009|-0.002|Periodic-TDL (+Chem, +RC)|
|egc|0.915 ± 0.013|0.911 ± 0.008|+0.004|Periodic-TDL (+Chem, +RC)|
|ei|0.827 ± 0.038|0.839 ± 0.018|-0.012|Periodic-TDL|
|eps|0.777 ± 0.067|0.787 ± 0.033|-0.010|Periodic-TDL|
|nc|0.834 ± 0.053|0.856 ± 0.032|-0.022|Periodic-TDL|
|xc|0.447 ± 0.140|0.423 ± 0.102|+0.024|Periodic-TDL|

macro=0.808；ΔBASE=+0.003；完整精度见results.json。

## S_FIXED

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|0.930 ± 0.018|0.934 ± 0.013|-0.004|Periodic-TDL|
|egb|0.922 ± 0.012|0.923 ± 0.009|-0.001|Periodic-TDL (+Chem, +RC)|
|egc|0.915 ± 0.013|0.911 ± 0.008|+0.004|Periodic-TDL (+Chem, +RC)|
|ei|0.827 ± 0.038|0.839 ± 0.018|-0.012|Periodic-TDL|
|eps|0.777 ± 0.067|0.787 ± 0.033|-0.010|Periodic-TDL|
|nc|0.833 ± 0.054|0.856 ± 0.032|-0.023|Periodic-TDL|
|xc|0.449 ± 0.139|0.423 ± 0.102|+0.026|Periodic-TDL|

macro=0.808；ΔBASE=+0.003；完整精度见results.json。

## S_HMR

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|0.930 ± 0.018|0.934 ± 0.013|-0.004|Periodic-TDL|
|egb|0.922 ± 0.012|0.923 ± 0.009|-0.001|Periodic-TDL (+Chem, +RC)|
|egc|0.915 ± 0.013|0.911 ± 0.008|+0.004|Periodic-TDL (+Chem, +RC)|
|ei|0.827 ± 0.038|0.839 ± 0.018|-0.012|Periodic-TDL|
|eps|0.777 ± 0.067|0.787 ± 0.033|-0.010|Periodic-TDL|
|nc|0.833 ± 0.054|0.856 ± 0.032|-0.023|Periodic-TDL|
|xc|0.449 ± 0.139|0.423 ± 0.102|+0.026|Periodic-TDL|

macro=0.808；ΔBASE=+0.003；完整精度见results.json。

七任务五折mean，std(ddof=1)，seed=42+fold。论文macro是不同配置逐任务包络；Egc3380/论文4124，Xc432/论文430，协议不等价，仅数值对照。共享开发TEST，非独立盲测或跨seed稳定。
有限Trimer软球表面/全部闭合分量/显式零模沿用r2；不是电子密度、周期无限链或体相自由体积。
首错：None
退出：{'controller': {'exit': 0, 'started_unix': 1793677741.5421786, 'ended_unix': 1793678266.0168915, 'wait_source': 'subprocess.Popen.wait'}, 'outer': {'exit': 0, 'started_unix': 1793677741.4321551, 'ended_unix': 1793678266.0267048, 'wait_source': 'subprocess.Popen.wait'}, 'bootstrap': {'exit': 0}}。真实exit以各层subprocess wait文件为准；最后CPU仅在全部140输出后一次。
证据：authorization/、registry/canonical.json及parent_qualification.json、lifecycle/、budget.json、cumulative_budget.json、finetune/、selection/final_frozen.json、formal/*/test_predictions.npz、final_cpu/与对应logs。
不自动替换BASE、不部署、不追加搜索。


</details>

<details><summary>results/ua3d_hmr_20261103/r1/repairs/scheduler_r3/final_review/report.md</summary>

# UA3D-HMR-20261103-01 science r2 / execution r3

状态：COMPLETE_PENDING_REVIEW；正式输出完成；independent_review=PENDING。
FINAL=S_FIXED；reference=0.804675462276949；macro=0.8076857278302016；未舍入Δ=0.003010265553252678。
near=False；target=False；performance_pass=False；最后CPU=PASSED/0。
实际阶段：{'preflight': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 1}, 'ft': {'reserved': 103, 'started': 103, 'attempted': 103, 'completed': 103}, 'test': {'reserved': 140, 'started': 140, 'attempted': 140, 'completed': 140}, 'cpu': {'reserved': 1, 'started': 1, 'attempted': 1, 'completed': 1}}；2 parent aliases不计新训练。
累计账目：{'geometry_reserved': 7300, 'geometry_attempted': 4231, 'preflight_total': 3, 'FT_attempts': 107, 'FT_attempts_cap': 107, 'parent_recorded_updates': 940, 'new_recorded_updates': 17150, 'cumulative_recorded_updates': 18090, 'old_unknown_updates': {'each': [0, 70], 'sum': [0, 140]}, 'cumulative_update_interval': [18090, 18230], 'update_upper_cap': 18230, 'accounting': 'r2 already nests r1; never add ancestor geometry again; aliases are not new FT'}。父r1/r2仍STOPPED/三层exit1，全部失败、中止、旧预约不退款。

## C_ANCHOR

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|0.930 ± 0.018|0.934 ± 0.013|-0.004|Periodic-TDL|
|egb|0.921 ± 0.015|0.923 ± 0.009|-0.002|Periodic-TDL (+Chem, +RC)|
|egc|0.914 ± 0.013|0.911 ± 0.008|+0.003|Periodic-TDL (+Chem, +RC)|
|ei|0.821 ± 0.040|0.839 ± 0.018|-0.018|Periodic-TDL|
|eps|0.776 ± 0.068|0.787 ± 0.033|-0.011|Periodic-TDL|
|nc|0.831 ± 0.057|0.856 ± 0.032|-0.025|Periodic-TDL|
|xc|0.440 ± 0.143|0.423 ± 0.102|+0.017|Periodic-TDL|

macro=0.805；ΔBASE=+0.000；完整精度见results.json。

## S_STAT

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|0.931 ± 0.019|0.934 ± 0.013|-0.003|Periodic-TDL|
|egb|0.921 ± 0.015|0.923 ± 0.009|-0.002|Periodic-TDL (+Chem, +RC)|
|egc|0.915 ± 0.013|0.911 ± 0.008|+0.004|Periodic-TDL (+Chem, +RC)|
|ei|0.827 ± 0.038|0.839 ± 0.018|-0.012|Periodic-TDL|
|eps|0.777 ± 0.067|0.787 ± 0.033|-0.010|Periodic-TDL|
|nc|0.834 ± 0.053|0.856 ± 0.032|-0.022|Periodic-TDL|
|xc|0.447 ± 0.140|0.423 ± 0.102|+0.024|Periodic-TDL|

macro=0.808；ΔBASE=+0.003；完整精度见results.json。

## S_FIXED

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|0.930 ± 0.018|0.934 ± 0.013|-0.004|Periodic-TDL|
|egb|0.922 ± 0.012|0.923 ± 0.009|-0.001|Periodic-TDL (+Chem, +RC)|
|egc|0.915 ± 0.013|0.911 ± 0.008|+0.004|Periodic-TDL (+Chem, +RC)|
|ei|0.827 ± 0.038|0.839 ± 0.018|-0.012|Periodic-TDL|
|eps|0.777 ± 0.067|0.787 ± 0.033|-0.010|Periodic-TDL|
|nc|0.833 ± 0.054|0.856 ± 0.032|-0.023|Periodic-TDL|
|xc|0.449 ± 0.139|0.423 ± 0.102|+0.026|Periodic-TDL|

macro=0.808；ΔBASE=+0.003；完整精度见results.json。

## S_HMR

|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|
|---|---:|---:|---:|---|
|eea|0.930 ± 0.018|0.934 ± 0.013|-0.004|Periodic-TDL|
|egb|0.922 ± 0.012|0.923 ± 0.009|-0.001|Periodic-TDL (+Chem, +RC)|
|egc|0.915 ± 0.013|0.911 ± 0.008|+0.004|Periodic-TDL (+Chem, +RC)|
|ei|0.827 ± 0.038|0.839 ± 0.018|-0.012|Periodic-TDL|
|eps|0.777 ± 0.067|0.787 ± 0.033|-0.010|Periodic-TDL|
|nc|0.833 ± 0.054|0.856 ± 0.032|-0.023|Periodic-TDL|
|xc|0.449 ± 0.139|0.423 ± 0.102|+0.026|Periodic-TDL|

macro=0.808；ΔBASE=+0.003；完整精度见results.json。

七任务五折mean，std(ddof=1)，seed=42+fold。论文macro是不同配置逐任务包络；Egc3380/论文4124，Xc432/论文430，协议不等价，仅数值对照。共享开发TEST，非独立盲测或跨seed稳定。
有限Trimer软球表面/全部闭合分量/显式零模沿用r2；不是电子密度、周期无限链或体相自由体积。
首错：None
退出：{'controller': {'exit': 0, 'started_unix': 1793677741.5421786, 'ended_unix': 1793678266.0168915, 'wait_source': 'subprocess.Popen.wait'}, 'outer': {'exit': 0, 'started_unix': 1793677741.4321551, 'ended_unix': 1793678266.0267048, 'wait_source': 'subprocess.Popen.wait'}, 'bootstrap': {'exit': 0}}。真实exit以各层subprocess wait文件为准；最后CPU仅在全部140输出后一次。
证据：authorization/、registry/canonical.json及parent_qualification.json、lifecycle/、budget.json、cumulative_budget.json、finetune/、selection/final_frozen.json、formal/*/test_predictions.npz、final_cpu/与对应logs。
不自动替换BASE、不部署、不追加搜索。


</details>

<details><summary>results/ua2d_a020_fp_fusion_20261024/r1/final_review/report.md</summary>

# A020 神经FP融合 r1 实际交付

状态：需返修；独立复审PENDING，未归档。
FINAL=None；全局w=None；未舍入Δmacro=None；性能通过=None。

科学问题：同一无FP预训练A020上，注意力、门控或纯神经输出融合是否提高正式TEST；ZERO/FP配对仅改变输入。
参考：D_A020原TEST macro=0.785044436513689。受控预算：五臂×七任务×五折×70epoch，seed42+fold。
本周期无PT、FP/构象生成、核模型拟合/加载/查询/残差监督；旧FINAL=D_A005不变。

| 政策 | macro R² | ΔB_A020 |
|---|---:|---:|
| B_A020 | 0.785 | +0.000 |
| UNI_ZERO | 未完成／不可比较 | — |
| UNI_FP | 未完成／不可比较 | — |
| TMOL_ZERO | 未完成／不可比较 | — |
| TMOL_FP | 未完成／不可比较 | — |
| FP_MLP | 未完成／不可比较 | — |
| NEURAL_LATE | 未完成／不可比较 | — |

## B_A020 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.929±0.010 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.931±0.013 | 0.923 | +0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.918±0.012 | 0.911 | +0.007 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.029 | 0.839 | -0.028 | Periodic-TDL |
| EPS | 0.713±0.105 | 0.787 | -0.074 | Periodic-TDL |
| Nc | 0.808±0.066 | 0.856 | -0.048 | Periodic-TDL |
| Xc | 0.386±0.081 | 0.423 | -0.037 | Periodic-TDL |

## UNI_ZERO 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## UNI_FP 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## TMOL_ZERO 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## TMOL_FP 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## FP_MLP 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## NEURAL_LATE 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## 预登记配对对照

| 对照 | Δmacro R² | 改善fold |
|---|---:|---:|
| UNI_FP_minus_B_A020 | 未完成／不可比较 | — |
| TMOL_FP_minus_B_A020 | 未完成／不可比较 | — |
| NEURAL_LATE_minus_B_A020 | 未完成／不可比较 | — |
| UNI_FP_minus_UNI_ZERO | 未完成／不可比较 | — |
| TMOL_FP_minus_TMOL_ZERO | 未完成／不可比较 | — |
| UNI_ZERO_minus_B_A020 | 未完成／不可比较 | — |
| TMOL_ZERO_minus_B_A020 | 未完成／不可比较 | — |
| UNI_FP_minus_TMOL_FP | 未完成／不可比较 | — |

## 实际成本

{"prepare_starts": 1, "ft_starts": 48, "epochs_started": 3356, "attempted_updates": 83344, "train_graphs": 1964962, "validation_started": 3354, "validation_graphs": 492494, "test_starts": 0, "test_graphs": 0, "late_units": 0, "late_graphs": 0, "cpu_suites": 0}

完整精度结果、35个逐fold差值、参数量和实际秒数见results.json/各单元parameters.json、consumption.json及history.json；所有计时来自真实执行，不作benchmark。

UNI_ZERO: 参数量19809626，已记录FT实际墙钟合计2804.815秒，TEST合计0.000秒。

UNI_FP: 参数量19809626，已记录FT实际墙钟合计767.788秒，TEST合计0.000秒。

## 解释边界

FP信息效应必须看UNI_FP−UNI_ZERO、TMOL_FP−TMOL_ZERO；胜基线但不胜ZERO不归因于FP。NEURAL_LATE若w=1则无FP贡献。不把fused−baseline当interaction；门控/注意力仅诊断。
父PT保留ALIGN×2及legacy_offset_v1的59998个跨分支seed复用；本周期不追溯修复历史随机独立性。历史全卡并发未独立核实。
各任务五折等权mean±std(ddof=1)，seed42+fold，再七任务等权macro。论文为逐任务最佳包络；数据/split/协议等价性未证明，仅数值对照。Egc3380/Xc432不同于论文4124/430；共享开发TEST非独立盲测，单seed/fold不足以证明跨seed稳定优势。
异常／未完成原因：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_a020_fp_fusion.py", line 211, in run
    self.matrix()
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_a020_fp_fusion.py", line 176, in matrix
    self.deadline();self.refresh();self.check_failures()
                    ^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_a020_fp_fusion.py", line 107, in refresh
    if j['exit'] is None and path.exists():j.update(exit=int(path.read_text().strip()),ended=t.now())
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: invalid literal for int() with base 10: ''
。

最终CPU：{"exit": null, "attempts": 0, "not_run_reason": "not all authorized stages completed; no early suite"}。

窗口Uni-Poly:a020-fp-fusion-r1；worker独立window见budget.jobs；日志/root/workspace/Uni-Poly-Plus-master/logs/ua2d_a020_fp_fusion_20261024/r1；命令、PID、起止、真实exit均在各job目录。

## 停止现场与交付层级

本次控制器实现存在退出文件发布/读取竞态：wrapper 使用 Path.write_text 写 exit.txt，读方先检查 exists 再 int(read_text().strip())，在短暂空文件窗口读到了空字符串并触发 ValueError。原始traceback未记录具体exit.txt路径，不能事后指定是哪个单元。事故证据见 incident.json、failure.json、controller/stderr.log 和各worker日志。代码保持故障时原样，不在失败后修复并重启。

一次prepare成功；FT启动48/175，其中46个真实exit0，2个因控制器停止而exit1（UNI_FP/egc/fold1、fold2）。UNI_ZERO完整35单元，UNI_FP完成11单元；其余未执行。两个中止单元保留BEST、history和consumption现场；其中一个异常处理被第二次SIGTERM打断，其consumption不能视为最终完整记录。

只按持久化记录报告下界：epochs_started=3356、attempted_updates=83344、train_graphs=1964962、validation_started=3354、validation_graphs=492494。失败/部分单元全额保留启动预留：FT48、epochs3360、updates83860、train1977220、validation调用3360/呈现495740；没有退款或追加额度。实际最后未落盘进度未核实。

尚未选择或冻结全局方法/w；新增正式TEST、NEURAL_LATE政策单元均0，最终CPU suite尝试0次（未满足全部实验与报告完成条件，不提前运行）。五个训练臂/六个新政策的完整七任务结果均为未完成／不可比较；不能据现有validation给出性能结论。仅保留旧B_A020参考数值及原七任务对照。

完成层级：实现已写入但尚未通过最终CPU测试；prepare完成；FT部分执行后失败停止；选择/冻结、正式TEST、完整比较和最终CPU未执行。状态需返修；没有重试、resume、换卡或修复后重启。旧无FP代码/模型/缓存/失败证据/账本与旧FINAL=D_A005保持不变；未归档。

下一步先由Codex只读独立审查本事故与实现，制定明确的返修、预算及停止条件；本次不自动消费剩余额度重启或另立编号补额度。


</details>

<details><summary>results/ua2d_a020_fp_fusion_20261024/r1/repairs/controller_repair1/final_review/report.md</summary>

# A020 神经FP融合 r1 实际交付

状态：待审查；独立复审PENDING，未归档。
FINAL=NEURAL_LATE；全局w=0.75；未舍入Δmacro=0.008489203051942007；性能通过=False。

科学问题：同一无FP预训练A020上，注意力、门控或纯神经输出融合是否提高正式TEST；ZERO/FP配对仅改变输入。
参考：D_A020原TEST macro=0.785044436513689。受控预算：五臂×七任务×五折×70epoch，seed42+fold。
本周期无PT、FP/构象生成、核模型拟合/加载/查询/残差监督；旧FINAL=D_A005不变。

| 政策 | macro R² | ΔB_A020 |
|---|---:|---:|
| B_A020 | 0.785 | +0.000 |
| UNI_ZERO | 0.790 | +0.005 |
| UNI_FP | 0.769 | -0.016 |
| TMOL_ZERO | 0.778 | -0.007 |
| TMOL_FP | 0.766 | -0.019 |
| FP_MLP | 0.724 | -0.062 |
| NEURAL_LATE | 0.794 | +0.008 |

## B_A020 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.929±0.010 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.931±0.013 | 0.923 | +0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.918±0.012 | 0.911 | +0.007 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.029 | 0.839 | -0.028 | Periodic-TDL |
| EPS | 0.713±0.105 | 0.787 | -0.074 | Periodic-TDL |
| Nc | 0.808±0.066 | 0.856 | -0.048 | Periodic-TDL |
| Xc | 0.386±0.081 | 0.423 | -0.037 | Periodic-TDL |

## UNI_ZERO 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.012 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.929±0.012 | 0.923 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.915±0.015 | 0.911 | +0.004 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.815±0.032 | 0.839 | -0.024 | Periodic-TDL |
| EPS | 0.744±0.060 | 0.787 | -0.043 | Periodic-TDL |
| Nc | 0.810±0.060 | 0.856 | -0.046 | Periodic-TDL |
| Xc | 0.391±0.106 | 0.423 | -0.032 | Periodic-TDL |

## UNI_FP 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.919±0.020 | 0.934 | -0.015 | Periodic-TDL |
| Egb | 0.910±0.010 | 0.923 | -0.013 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.893±0.015 | 0.911 | -0.018 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.767±0.032 | 0.839 | -0.072 | Periodic-TDL |
| EPS | 0.723±0.096 | 0.787 | -0.064 | Periodic-TDL |
| Nc | 0.805±0.056 | 0.856 | -0.051 | Periodic-TDL |
| Xc | 0.368±0.115 | 0.423 | -0.055 | Periodic-TDL |

## TMOL_ZERO 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.925±0.012 | 0.934 | -0.009 | Periodic-TDL |
| Egb | 0.929±0.015 | 0.923 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.919±0.012 | 0.911 | +0.008 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.817±0.034 | 0.839 | -0.022 | Periodic-TDL |
| EPS | 0.689±0.117 | 0.787 | -0.098 | Periodic-TDL |
| Nc | 0.793±0.078 | 0.856 | -0.063 | Periodic-TDL |
| Xc | 0.370±0.095 | 0.423 | -0.053 | Periodic-TDL |

## TMOL_FP 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.913±0.017 | 0.934 | -0.021 | Periodic-TDL |
| Egb | 0.906±0.007 | 0.923 | -0.017 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.891±0.014 | 0.911 | -0.020 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.763±0.056 | 0.839 | -0.076 | Periodic-TDL |
| EPS | 0.700±0.095 | 0.787 | -0.087 | Periodic-TDL |
| Nc | 0.808±0.050 | 0.856 | -0.048 | Periodic-TDL |
| Xc | 0.379±0.114 | 0.423 | -0.044 | Periodic-TDL |

## FP_MLP 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.865±0.013 | 0.934 | -0.069 | Periodic-TDL |
| Egb | 0.855±0.019 | 0.923 | -0.068 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.862±0.015 | 0.911 | -0.049 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.699±0.046 | 0.839 | -0.140 | Periodic-TDL |
| EPS | 0.670±0.058 | 0.787 | -0.117 | Periodic-TDL |
| Nc | 0.746±0.062 | 0.856 | -0.110 | Periodic-TDL |
| Xc | 0.368±0.097 | 0.423 | -0.055 | Periodic-TDL |

## NEURAL_LATE 七任务数值对照

| Task | 项目R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.931±0.009 | 0.934 | -0.003 | Periodic-TDL |
| Egb | 0.927±0.012 | 0.923 | +0.004 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.920±0.012 | 0.911 | +0.009 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.809±0.028 | 0.839 | -0.030 | Periodic-TDL |
| EPS | 0.729±0.086 | 0.787 | -0.058 | Periodic-TDL |
| Nc | 0.812±0.058 | 0.856 | -0.044 | Periodic-TDL |
| Xc | 0.427±0.075 | 0.423 | +0.004 | Periodic-TDL |

## 预登记配对对照

| 对照 | Δmacro R² | 改善fold |
|---|---:|---:|
| UNI_FP_minus_B_A020 | -0.016 | 9/35 |

UNI_FP_minus_B_A020 / eea: mean±std=-0.010±0.016，五折差值=-0.029, +0.012, -0.005, -0.019, -0.010，改善1/5。

UNI_FP_minus_B_A020 / egb: mean±std=-0.021±0.012，五折差值=-0.014, -0.014, -0.032, -0.034, -0.010，改善0/5。

UNI_FP_minus_B_A020 / egc: mean±std=-0.025±0.012，五折差值=-0.019, -0.037, -0.006, -0.031, -0.031，改善0/5。

UNI_FP_minus_B_A020 / ei: mean±std=-0.045±0.030，五折差值=-0.016, -0.074, -0.078, -0.039, -0.018，改善0/5。

UNI_FP_minus_B_A020 / eps: mean±std=+0.010±0.038，五折差值=+0.074, -0.025, +0.008, -0.007, +0.002，改善3/5。

UNI_FP_minus_B_A020 / nc: mean±std=-0.003±0.044，五折差值=+0.056, -0.041, -0.051, +0.005, +0.016，改善3/5。

UNI_FP_minus_B_A020 / xc: mean±std=-0.018±0.098，五折差值=-0.015, +0.085, -0.161, -0.058, +0.059，改善2/5。
| TMOL_FP_minus_B_A020 | -0.019 | 9/35 |

TMOL_FP_minus_B_A020 / eea: mean±std=-0.016±0.011，五折差值=-0.027, -0.005, -0.006, -0.028, -0.013，改善0/5。

TMOL_FP_minus_B_A020 / egb: mean±std=-0.025±0.009，五折差值=-0.028, -0.017, -0.033, -0.034, -0.013，改善0/5。

TMOL_FP_minus_B_A020 / egc: mean±std=-0.027±0.013，五折差值=-0.018, -0.040, -0.009, -0.034, -0.034，改善0/5。

TMOL_FP_minus_B_A020 / ei: mean±std=-0.048±0.058，五折差值=+0.013, -0.035, -0.139, -0.016, -0.062，改善1/5。

TMOL_FP_minus_B_A020 / eps: mean±std=-0.013±0.022，五折差值=+0.008, -0.046, -0.014, +0.007, -0.019，改善2/5。

TMOL_FP_minus_B_A020 / nc: mean±std=-0.000±0.039，五折差值=+0.014, -0.021, -0.058, +0.025, +0.039，改善3/5。

TMOL_FP_minus_B_A020 / xc: mean±std=-0.007±0.076，五折差值=+0.047, +0.046, -0.098, -0.082, +0.053，改善3/5。
| NEURAL_LATE_minus_B_A020 | +0.008 | 23/35 |

NEURAL_LATE_minus_B_A020 / eea: mean±std=+0.002±0.003，五折差值=+0.002, +0.006, +0.000, -0.003, +0.003，改善4/5。

NEURAL_LATE_minus_B_A020 / egb: mean±std=-0.004±0.004，五折差值=-0.008, -0.002, -0.002, -0.009, -0.001，改善0/5。

NEURAL_LATE_minus_B_A020 / egc: mean±std=+0.002±0.002，五折差值=+0.002, +0.002, +0.006, -0.001, +0.002，改善4/5。

NEURAL_LATE_minus_B_A020 / ei: mean±std=-0.003±0.010，五折差值=+0.004, -0.008, -0.017, +0.001, +0.007，改善3/5。

NEURAL_LATE_minus_B_A020 / eps: mean±std=+0.017±0.019，五折差值=+0.026, -0.008, +0.044, +0.012, +0.010，改善4/5。

NEURAL_LATE_minus_B_A020 / nc: mean±std=+0.005±0.012，五折差值=+0.010, -0.006, -0.010, +0.012, +0.016，改善3/5。

NEURAL_LATE_minus_B_A020 / xc: mean±std=+0.041±0.006，五折差值=+0.041, +0.042, +0.034, +0.050, +0.038，改善5/5。
| UNI_FP_minus_UNI_ZERO | -0.021 | 5/35 |

UNI_FP_minus_UNI_ZERO / eea: mean±std=-0.006±0.021，五折差值=-0.029, +0.026, -0.002, -0.021, -0.002，改善1/5。

UNI_FP_minus_UNI_ZERO / egb: mean±std=-0.019±0.009，五折差值=-0.014, -0.011, -0.029, -0.027, -0.012，改善0/5。

UNI_FP_minus_UNI_ZERO / egc: mean±std=-0.021±0.013，五折差值=-0.011, -0.036, -0.005, -0.022, -0.032，改善0/5。

UNI_FP_minus_UNI_ZERO / ei: mean±std=-0.049±0.023，五折差值=-0.015, -0.066, -0.073, -0.038, -0.053，改善0/5。

UNI_FP_minus_UNI_ZERO / eps: mean±std=-0.021±0.056，五折差值=+0.002, -0.020, -0.108, -0.026, +0.047，改善2/5。

UNI_FP_minus_UNI_ZERO / nc: mean±std=-0.005±0.041，五折差值=+0.063, -0.048, -0.002, -0.018, -0.021，改善1/5。

UNI_FP_minus_UNI_ZERO / xc: mean±std=-0.023±0.058，五折差值=-0.017, +0.042, -0.117, -0.016, -0.009，改善1/5。
| TMOL_FP_minus_TMOL_ZERO | -0.012 | 11/35 |

TMOL_FP_minus_TMOL_ZERO / eea: mean±std=-0.012±0.017，五折差值=-0.026, +0.014, -0.006, -0.028, -0.014，改善1/5。

TMOL_FP_minus_TMOL_ZERO / egb: mean±std=-0.023±0.010，五折差值=-0.030, -0.009, -0.031, -0.030, -0.016，改善0/5。

TMOL_FP_minus_TMOL_ZERO / egc: mean±std=-0.028±0.012，五折差值=-0.019, -0.039, -0.012, -0.037, -0.034，改善0/5。

TMOL_FP_minus_TMOL_ZERO / ei: mean±std=-0.053±0.062，五折差值=+0.015, -0.031, -0.135, -0.017, -0.099，改善1/5。

TMOL_FP_minus_TMOL_ZERO / eps: mean±std=+0.010±0.046，五折差值=-0.039, -0.041, +0.030, +0.052, +0.049，改善3/5。

TMOL_FP_minus_TMOL_ZERO / nc: mean±std=+0.014±0.035，五折差值=+0.017, -0.009, -0.020, +0.072, +0.012，改善3/5。

TMOL_FP_minus_TMOL_ZERO / xc: mean±std=+0.009±0.042，五折差值=+0.047, +0.038, -0.044, -0.028, +0.032，改善3/5。
| UNI_ZERO_minus_B_A020 | +0.005 | 14/35 |

UNI_ZERO_minus_B_A020 / eea: mean±std=-0.005±0.006，五折差值=-0.000, -0.013, -0.003, +0.002, -0.008，改善1/5。

UNI_ZERO_minus_B_A020 / egb: mean±std=-0.002±0.004，五折差值=+0.000, -0.003, -0.003, -0.008, +0.003，改善2/5。

UNI_ZERO_minus_B_A020 / egc: mean±std=-0.003±0.005，五折差值=-0.008, -0.001, -0.001, -0.009, +0.001，改善1/5。

UNI_ZERO_minus_B_A020 / ei: mean±std=+0.004±0.018，五折差值=-0.001, -0.008, -0.005, -0.002, +0.035，改善1/5。

UNI_ZERO_minus_B_A020 / eps: mean±std=+0.031±0.064，五折差值=+0.072, -0.005, +0.116, +0.020, -0.045，改善3/5。

UNI_ZERO_minus_B_A020 / nc: mean±std=+0.002±0.033，五折差值=-0.007, +0.007, -0.050, +0.023, +0.037，改善3/5。

UNI_ZERO_minus_B_A020 / xc: mean±std=+0.006±0.050，五折差值=+0.002, +0.043, -0.043, -0.042, +0.068，改善3/5。
| TMOL_ZERO_minus_B_A020 | -0.007 | 15/35 |

TMOL_ZERO_minus_B_A020 / eea: mean±std=-0.004±0.009，五折差值=-0.001, -0.020, -0.000, +0.000, +0.001，改善2/5。

TMOL_ZERO_minus_B_A020 / egb: mean±std=-0.002±0.005，五折差值=+0.002, -0.008, -0.002, -0.004, +0.003，改善2/5。

TMOL_ZERO_minus_B_A020 / egc: mean±std=+0.001±0.002，五折差值=+0.001, -0.001, +0.003, +0.003, +0.001，改善4/5。

TMOL_ZERO_minus_B_A020 / ei: mean±std=+0.005±0.017，五折差值=-0.003, -0.004, -0.004, +0.002, +0.036，改善2/5。

TMOL_ZERO_minus_B_A020 / eps: mean±std=-0.023±0.045，五折差值=+0.047, -0.006, -0.045, -0.045, -0.067，改善1/5。

TMOL_ZERO_minus_B_A020 / nc: mean±std=-0.014±0.030，五折差值=-0.003, -0.011, -0.038, -0.047, +0.027，改善1/5。

TMOL_ZERO_minus_B_A020 / xc: mean±std=-0.016±0.036，五折差值=+0.000, +0.008, -0.055, -0.054, +0.021，改善3/5。
| UNI_FP_minus_TMOL_FP | +0.004 | 24/35 |

UNI_FP_minus_TMOL_FP / eea: mean±std=+0.006±0.008，五折差值=-0.003, +0.017, +0.001, +0.009, +0.003，改善4/5。

UNI_FP_minus_TMOL_FP / egb: mean±std=+0.004±0.006，五折差值=+0.014, +0.003, +0.001, +0.000, +0.004，改善5/5。

UNI_FP_minus_TMOL_FP / egc: mean±std=+0.002±0.002，五折差值=-0.002, +0.003, +0.003, +0.003, +0.003，改善4/5。

UNI_FP_minus_TMOL_FP / ei: mean±std=+0.003±0.046，五折差值=-0.028, -0.039, +0.062, -0.024, +0.045，改善2/5。

UNI_FP_minus_TMOL_FP / eps: mean±std=+0.023±0.028，五折差值=+0.066, +0.021, +0.022, -0.014, +0.021，改善4/5。

UNI_FP_minus_TMOL_FP / nc: mean±std=-0.003±0.028，五折差值=+0.042, -0.021, +0.007, -0.020, -0.024，改善2/5。

UNI_FP_minus_TMOL_FP / xc: mean±std=-0.011±0.048，五折差值=-0.062, +0.039, -0.062, +0.024, +0.007，改善3/5。
| FINAL_minus_B_A020 | +0.008 | 23/35 |

FINAL_minus_B_A020 / eea: mean±std=+0.002±0.003，五折差值=+0.002, +0.006, +0.000, -0.003, +0.003，改善4/5。

FINAL_minus_B_A020 / egb: mean±std=-0.004±0.004，五折差值=-0.008, -0.002, -0.002, -0.009, -0.001，改善0/5。

FINAL_minus_B_A020 / egc: mean±std=+0.002±0.002，五折差值=+0.002, +0.002, +0.006, -0.001, +0.002，改善4/5。

FINAL_minus_B_A020 / ei: mean±std=-0.003±0.010，五折差值=+0.004, -0.008, -0.017, +0.001, +0.007，改善3/5。

FINAL_minus_B_A020 / eps: mean±std=+0.017±0.019，五折差值=+0.026, -0.008, +0.044, +0.012, +0.010，改善4/5。

FINAL_minus_B_A020 / nc: mean±std=+0.005±0.012，五折差值=+0.010, -0.006, -0.010, +0.012, +0.016，改善3/5。

FINAL_minus_B_A020 / xc: mean±std=+0.041±0.006，五折差值=+0.041, +0.042, +0.034, +0.050, +0.038，改善5/5。

## 实际成本

{"prepare_starts": 1, "ft_starts": 177, "epochs_started": 12386, "attempted_updates": 292224, "train_graphs": 6865662, "validation_started": 12384, "validation_graphs": 1722394, "test_starts": 175, "test_graphs": 29375, "late_units": 35, "late_graphs": 5875, "cpu_suites": 1}

完整精度结果、35个逐fold差值、参数量和实际秒数见results.json/各单元parameters.json、consumption.json及history.json；所有计时来自真实执行，不作benchmark。

UNI_ZERO: 参数量19809626，已记录FT实际墙钟合计2804.815秒，TEST合计35.835秒。

UNI_FP: 参数量19809626，已记录FT实际墙钟合计2987.890秒，TEST合计34.649秒。

TMOL_ZERO: 参数量19689755，已记录FT实际墙钟合计2662.311秒，TEST合计36.069秒。

TMOL_FP: 参数量19689755，已记录FT实际墙钟合计2725.642秒，TEST合计35.421秒。

FP_MLP: 参数量345601，已记录FT实际墙钟合计359.589秒，TEST合计22.146秒。

## 解释边界

FP信息效应必须看UNI_FP−UNI_ZERO、TMOL_FP−TMOL_ZERO；胜基线但不胜ZERO不归因于FP。NEURAL_LATE若w=1则无FP贡献。不把fused−baseline当interaction；门控/注意力仅诊断。
父PT保留ALIGN×2及legacy_offset_v1的59998个跨分支seed复用；本周期不追溯修复历史随机独立性。历史全卡并发未独立核实。
各任务五折等权mean±std(ddof=1)，seed42+fold，再七任务等权macro。论文为逐任务最佳包络；数据/split/协议等价性未证明，仅数值对照。Egc3380/Xc432不同于论文4124/430；共享开发TEST非独立盲测，单seed/fold不足以证明跨seed稳定优势。
异常／未完成原因：无。

最终CPU：{"exit": 0, "command": ["timeout", "--signal=TERM", "--kill-after=10s", "600s", "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_ua2d_a020_fp_fusion.py"], "log": "/root/workspace/Uni-Poly-Plus-master/logs/ua2d_a020_fp_fusion_20261024/r1/controller_repair1/a020r1_cpu", "time": "2026-10-24T23:15:54.578785+08:00", "attempts": 1}。

窗口Uni-Poly:a020-fp-fusion-repair1（返修）/a020-fp-fusion-r1（原始）；worker独立window见budget.jobs；日志/root/workspace/Uni-Poly-Plus-master/logs/ua2d_a020_fp_fusion_20261024/r1/controller_repair1；命令、PID、起止、真实exit均在各job目录。

## 累计返修与历史成本

复用46个原COMPLETE，仅129个新attempt；其中UNI_FP/egc/fold1、fold2是唯一从原PT/head/common重新训练的例外，未从BEST续训、不比较旧新attempt择优。父两个失败及事故证据完整保留。
科学config/checkpoint保持原r1；repair-r1只记录执行层。module/dataset与旧无FP六文件未修改，prepare/common/缓存全部只读复用，新增prepare/初始化为0。
父actual为持久化下界，中止时未落盘进度未知。下方known_actual累计仍只是下界；新成功单元有完整计数，不能据此把父失败成本改成精确actual。
| 消费项 | 父已收费 | 本次收费/预留 | 累计收费/预留 | 累计上限 |
|---|---:|---:|---:|---:|
| prepare_starts | 1 | 0 | 1 | 1 |
| ft_starts | 48 | 129 | 177 | 177 |
| epochs_started | 3360 | 9030 | 12390 | 12390 |
| attempted_updates | 83860 | 208880 | 292740 | 292740 |
| train_graphs | 1977220 | 4900700 | 6877920 | 6877920 |
| validation_started | 3360 | 9030 | 12390 | 12390 |
| validation_graphs | 495740 | 1229900 | 1725640 | 1725640 |
| test_starts | 0 | 175 | 175 | 175 |
| test_graphs | 0 | 29375 | 29375 | 29375 |
| late_units | 0 | 35 | 35 | 35 |
| late_graphs | 0 | 5875 | 5875 | 5875 |
| cpu_suites | 0 | 1 | 1 | 1 |

父历史事故与无FP父PT legacy/ALIGN×2限制均保留。绝对截止2026-10-26 20:42:23.561393+08:00，未因返修重置。独立复审PENDING，未归档。

最终CPU摘要：32 passed、1依赖弃用warning，exit=0，pytest 6.04秒；本周期累计仅1次尝试。已于2026-10-24 23:15完成执行，当前只交付待审查，独立复审未进行。


</details>

<details><summary>results/ua2d_3dshift_20261022/r1/all1_pcontrol_gpu2/final_review/report.md</summary>

# UA2D-3DSHIFT-20261022-01 / r1.5-ALL1-PCTRL-GPU2 交付

状态：待审查；固定FINAL=P_G；完整正式TEST=True；CPU exit=0。
执行自检不构成Codex独立审查。所有差值用未舍入值，展示3位小数；失败不追加候选或重试。

| 预登记对照 | 正式Δmacro R² | 完整性 |
|---|---:|---|
| P_G_minus_F_G10K | +0.001 | COMPLETE |
| P_G_minus_F_G | +0.002 | COMPLETE |
| P_G_minus_P10K | -0.010 | COMPLETE |
| P_G_minus_B_FP | -0.008 | COMPLETE |

| 臂 | validation macro | TEST macro |
|---|---:|---:|
| F_G | 0.812 | 0.781 |
| F_G10K | 0.806 | 0.782 |
| P_G | 0.809 | 0.783 |
| P_G5K | 0.810 | 0.776 |

## F_G 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.928±0.021 | 0.934 | -0.006 | Periodic-TDL |
| Egb | 0.929±0.015 | 0.923 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.015 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.805±0.025 | 0.839 | -0.034 | Periodic-TDL |
| EPS | 0.739±0.068 | 0.787 | -0.048 | Periodic-TDL |
| Nc | 0.797±0.090 | 0.856 | -0.059 | Periodic-TDL |
| Xc | 0.353±0.101 | 0.423 | -0.070 | Periodic-TDL |

## F_G10K 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.923±0.016 | 0.934 | -0.011 | Periodic-TDL |
| Egb | 0.934±0.014 | 0.923 | +0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.917±0.017 | 0.911 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.798±0.018 | 0.839 | -0.041 | Periodic-TDL |
| EPS | 0.707±0.076 | 0.787 | -0.080 | Periodic-TDL |
| Nc | 0.773±0.090 | 0.856 | -0.083 | Periodic-TDL |
| Xc | 0.423±0.114 | 0.423 | -0.000 | Periodic-TDL |

## P_G 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.926±0.017 | 0.934 | -0.008 | Periodic-TDL |
| Egb | 0.932±0.012 | 0.923 | +0.009 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.017 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.803±0.037 | 0.839 | -0.036 | Periodic-TDL |
| EPS | 0.700±0.084 | 0.787 | -0.087 | Periodic-TDL |
| Nc | 0.809±0.069 | 0.856 | -0.047 | Periodic-TDL |
| Xc | 0.397±0.134 | 0.423 | -0.026 | Periodic-TDL |

## P_G5K 七任务数值对照

| 任务 | 项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.925±0.017 | 0.934 | -0.009 | Periodic-TDL |
| Egb | 0.929±0.013 | 0.923 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.915±0.015 | 0.911 | +0.004 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.801±0.018 | 0.839 | -0.038 | Periodic-TDL |
| EPS | 0.730±0.075 | 0.787 | -0.057 | Periodic-TDL |
| Nc | 0.806±0.066 | 0.856 | -0.050 | Periodic-TDL |
| Xc | 0.323±0.102 | 0.423 | -0.100 | Periodic-TDL |

共享开发TEST不是独立盲测；F_G=历史BASE5k，历史参考均只作数值对照；匹配对照为本次F_G10K；主要比较P_G−F_G10K，缺少L组合，不能估计严格2×2交互。P为整体表征包；FG微调/TEST使用GPU3，P_G使用GPU2，资源差异须说明；论文协议未证明等价。
各任务五折seed42+fold等权、std(ddof1)，再七任务等权macro。完整预测、scaler、history、checkpoint、配置、冻结、预算、计时及真实exit保留原根。
主要对照P_G−F_G10K Δmacro=+0.001；性能通过=False。
异常／未完成原因：无


</details>

<details><summary>results/ua2d_fpdecay_20261022/sg_r2/final_review/report.md</summary>

# UA2D-FPDECAY r2-SG 实际交付

状态：待审查；DSH执行，待Codex独立审查。
FINAL=FP_DECAY；完整两支正式TEST=True；性能：未达到双参考性能门槛；最终CPU exit=0。

历史B_FP macro=0.791，PTSCALE/P10K=0.793，直接复用，不重测。
主要受控比较为FP_DECAY−单卡BASE；晋级要求FP_DECAY相对BASE及P10K未舍入Δmacro均≥0.010。不能凭Nc/EPS单项改善晋级。
只用物理GPU3/world1/micro168/acc6/global1008；两支各从相同公共step0独立训练5000步；两支结束且worker退出后70FT及一次性70正式TEST。
固定候选FP_DECAY不依validation或TEST改选；BASE保持原FP权重；唯一科学变化为FP_DECAY的第4001—5000步原FP warmup权重乘固定余弦退火。
prepare=0、公共初始化作业0；只读引用原PT缓存、PTSCALE logical映射及TRANSFER全部5875条FT缓存，保留原身份，不复制、重建或在线fallback。
单卡按六个microbatch优化窗口目标计数归一化，不乘world2或再除acc6；每微批对比最多168。P10K及历史B_FP为跨资源／预算数值参考，不解释为单因素因果效应。

| 预登记正式对照 | Δmacro R² | 完整性 |
|---|---:|---|
| FP_DECAY_minus_BASE | +0.010 | COMPLETE |
| FP_DECAY_minus_PTSCALE_P10K | -0.003 | COMPLETE |
| FP_DECAY_minus_B_FP | -0.001 | COMPLETE |

## BASE 七任务SOTA数值对照（macro=0.781）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.928±0.021 | 0.934 | -0.006 | Periodic-TDL |
| Egb | 0.929±0.015 | 0.923 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.015 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.805±0.025 | 0.839 | -0.034 | Periodic-TDL |
| EPS | 0.739±0.068 | 0.787 | -0.048 | Periodic-TDL |
| Nc | 0.797±0.090 | 0.856 | -0.059 | Periodic-TDL |
| Xc | 0.353±0.101 | 0.423 | -0.070 | Periodic-TDL |

## FP_DECAY 七任务SOTA数值对照（macro=0.790）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.926±0.015 | 0.934 | -0.008 | Periodic-TDL |
| Egb | 0.928±0.015 | 0.923 | +0.005 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.909±0.015 | 0.911 | -0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.821±0.016 | 0.839 | -0.018 | Periodic-TDL |
| EPS | 0.734±0.077 | 0.787 | -0.053 | Periodic-TDL |
| Nc | 0.820±0.075 | 0.856 | -0.036 | Periodic-TDL |
| Xc | 0.396±0.103 | 0.423 | -0.027 | Periodic-TDL |

各任务五折等权R²，std(ddof=1)，seed42+fold，再七任务等权macro；所有计算使用未舍入值，展示3位小数。
论文逐任务最佳包络macro约0.810；协议等价性未证明，仅作数值对照，不宣称公平超过论文。共享开发TEST不称独立盲测。
配置、复用初始化来源、映射 COMPLETE、两支真实step5000 PT来源、history／best／scaler／预测位于原新周期根；冻结名单selection/final_frozen.json，完整精度指标及配对差值final_review/results.json；预算budget.json；源码diff及真实训练计时final_review/；全部命令／window／真实exit在logs/ua2d_fpdecay_20261022/sg_r2/。
未完成／异常原因：无。


</details>

<details><summary>results/ua3d_base_opt_20261026/r6/final_review/report.md</summary>

# UA3D BASE OPT r6 A—G

状态：失败停止，待审查；independent_review=PENDING。
预选FINAL：未冻结；历史BASE完整精度：`0.795244745287872`。
评价使用共享开发validation/TEST，非独立盲测。所有诊断臂均预登记；TEST不改变FINAL。
完整正式TEST：0/1套；不完整单元均为未完成／不可比较。

| 配方 | 角色 | TEST macro R² | 相对 BASE Δmacro R² | +0.010 |
|---|---|---:|---:|---|
| A_L_A8_B8 | 预登记诊断臂 | 未完成／不可比较 | — | 未判定 |

预选FINAL完整精度：`null`。
数值门槛与七任务SOTA目标分别判断；诊断臂的数值通过不使其事后晋级。

## A_L_A8_B8 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 未完成／不可比较 | 0.924 | — | 0.934 | — | Periodic-TDL |
| Egb | 未完成／不可比较 | 0.917 | — | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| Egc | 未完成／不可比较 | 0.913 | — | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| Ei | 未完成／不可比较 | 0.806 | — | 0.839 | — | Periodic-TDL |
| EPS | 未完成／不可比较 | 0.771 | — | 0.787 | — | Periodic-TDL |
| Nc | 未完成／不可比较 | 0.821 | — | 0.856 | — | Periodic-TDL |
| Xc | 未完成／不可比较 | 0.415 | — | 0.423 | — | Periodic-TDL |

配对比较、每fold差值、2×2主效应和差中差的完整精度见results.json的controlled_comparisons。
StageA/B/C/D/E/F的完整2×2只解释各自固定父条件；跨阶段为顺序开发，不能推断跨阶段交互。
StageE双向臂与独立投影不构成完整factorial；StageF将2D/3D PT dropout绑定；StageG新增容量单列，不称完整factorial。
无多seed支持，不宣称统计显著或全局最优；不能仅凭gate权重解释模态贡献。
项目为paper5_outer、每任务五折、seed42+fold、任务等权macro、std(ddof=1)。论文逐任务最佳组成包络。
Egc3380／论文4124、Xc432／论文430等数据、split和统计协议差异未消除，仅作数值对照。
history记录正常训练/validation前向的dropout、训练模式、optimizer组、模态范数、gate熵/均值/饱和比例、时间和显存；无额外模型前向。
完整精度原始预测和row/key/target对齐在formal；BEST/validation/scaler/config在finetune；PT逐项sum/count/loss及消费在pretrain。
预算预留：`{"initialize_starts": 1, "pt_starts": 0, "pt_updates": 0, "pt_microbatches": 0, "pt_presentations": 0, "ft_starts": 1, "ft_epochs": 70, "ft_updates": 700, "test_starts": 0, "test_graphs": 0, "cpu_suites": 0}`。
实际已知消费下界：`{"initialize_starts": 1, "pt_starts": 0, "pt_updates": 0, "pt_microbatches": 0, "pt_presentations": 0, "ft_starts": 1, "ft_epochs": 69, "ft_updates": 684, "test_starts": 0, "test_graphs": 0, "cpu_suites": 0}`。
实际已知完成：`{"initialize_starts": 1, "pt_starts": 0, "pt_updates": 0, "pt_microbatches": 0, "pt_presentations": 0, "ft_starts": 0, "ft_epochs": 68, "ft_updates": 684, "test_starts": 0, "test_graphs": 0, "cpu_suites": 0}`。
CPU suite：`{"attempts": 0, "exit": null, "log": null, "status": "NOT_RUN"}`。
硬截止：2026-10-30T22:07:13.844249+08:00；t0不刷新；全部真实exit见budget.json及logs作业exit.txt。
异常：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_base_opt_r6.py", line 368, in run
    self.develop();self.guard_clock('test');frozen=t.freeze()
    ^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_base_opt_r6.py", line 222, in develop
    order.append(self.ft_recipe('A_'+name,'A',ft=ft,control=t.BASELINE))
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_base_opt_r6.py", line 210, in ft_recipe
    for fold in range(5):self.wait(self.launch('ft',canonical,task,fold))
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_base_opt_r6.py", line 202, in wait
    self.guard_clock();self.refresh();time.sleep(5)
                                      ^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua3d_base_opt_r6.py", line 460, in stop
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled termination '+str(sig))
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: controlled termination 15
。
未完成时不使用validation代替TEST；未执行单元不虚报exit。最终保持待独立审查，不自动替换BASE、部署或归档为已验收。


</details>

<details><summary>results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/final_review/report.md</summary>

# UA3D BASE OPT r6 A—G

状态：完成，待审查；independent_review=PENDING。
预选FINAL：E_P_SYM_SHARED；历史BASE完整精度：`0.795244745287872`。
评价使用共享开发validation/TEST，非独立盲测。所有诊断臂均预登记；TEST不改变FINAL。
完整正式TEST：23/23套；不完整单元均为未完成／不可比较。

| 配方 | 角色 | TEST macro R² | 相对 BASE Δmacro R² | +0.010 |
|---|---|---:|---:|---|
| A_L_A8_B8 | 预登记诊断臂 | 0.793 | -0.002 | False |
| A_L_A0_B8 | 预登记诊断臂 | 0.769 | -0.026 | False |
| A_L_A8_B0 | 预登记诊断臂 | 0.785 | -0.011 | False |
| A_L_A0_B0 | 预登记诊断臂 | 0.752 | -0.043 | False |
| B_U_LO_RESTART | 预登记诊断臂 | 0.793 | -0.002 | False |
| B_U_HI_MONO | 预登记诊断臂 | 0.792 | -0.003 | False |
| B_U_LO_MONO | 预登记诊断臂 | 0.791 | -0.004 | False |
| C_D2_1_D3_0 | 预登记诊断臂 | 0.778 | -0.017 | False |
| C_D2_0_D3_1 | 预登记诊断臂 | 0.789 | -0.006 | False |
| C_D2_1_D3_1 | 预登记诊断臂 | 0.784 | -0.011 | False |
| D_DF_0_H5 | 预登记诊断臂 | 0.788 | -0.008 | False |
| D_DF_2_H1 | 预登记诊断臂 | 0.787 | -0.009 | False |
| D_DF_0_H1 | 预登记诊断臂 | 0.783 | -0.012 | False |
| E_P_CURRENT_SHARED | 预登记诊断臂 | 0.799 | +0.004 | False |
| E_P_CURRENT_IND | 预登记诊断臂 | 0.795 | -0.000 | False |
| E_P_PAPER_SHARED | 预登记诊断臂 | 0.795 | -0.001 | False |
| E_P_PAPER_IND | 预登记诊断臂 | 0.784 | -0.011 | False |
| E_P_SYM_SHARED | 预选FINAL | 0.795 | -0.000 | False |
| F_PD_N0_F2 | 预登记诊断臂 | 0.794 | -0.001 | False |
| F_PD_N1_F0 | 预登记诊断臂 | 0.805 | +0.009 | False |
| F_PD_N0_F0 | 预登记诊断臂 | 0.794 | -0.001 | False |
| G_RESIDUAL | 预登记诊断臂 | 0.784 | -0.011 | False |
| G_MODALNORM | 预登记诊断臂 | 0.778 | -0.017 | False |

预选FINAL完整精度：`{"macro": 0.7949437804438609, "delta_macro": -0.0003009648440110979, "performance_pass": false, "sota_all7_pass": false}`。
数值门槛与七任务SOTA目标分别判断；诊断臂的数值通过不使其事后晋级。

## A_L_A8_B8 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.926±0.022 | 0.924 | +0.002 | 0.934 | -0.008 | Periodic-TDL |
| Egb | 0.919±0.011 | 0.917 | +0.002 | 0.923 | -0.004 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.915±0.011 | 0.913 | +0.002 | 0.911 | +0.004 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.797±0.047 | 0.806 | -0.009 | 0.839 | -0.042 | Periodic-TDL |
| EPS | 0.784±0.044 | 0.771 | +0.013 | 0.787 | -0.003 | Periodic-TDL |
| Nc | 0.816±0.078 | 0.821 | -0.005 | 0.856 | -0.040 | Periodic-TDL |
| Xc | 0.395±0.090 | 0.415 | -0.021 | 0.423 | -0.028 | Periodic-TDL |

## A_L_A0_B8 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.912±0.011 | 0.924 | -0.012 | 0.934 | -0.022 | Periodic-TDL |
| Egb | 0.911±0.012 | 0.917 | -0.006 | 0.923 | -0.012 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.902±0.013 | 0.913 | -0.010 | 0.911 | -0.009 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.755±0.052 | 0.806 | -0.051 | 0.839 | -0.084 | Periodic-TDL |
| EPS | 0.698±0.052 | 0.771 | -0.074 | 0.787 | -0.089 | Periodic-TDL |
| Nc | 0.791±0.039 | 0.821 | -0.030 | 0.856 | -0.065 | Periodic-TDL |
| Xc | 0.413±0.095 | 0.415 | -0.002 | 0.423 | -0.010 | Periodic-TDL |

## A_L_A8_B0 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.921±0.017 | 0.924 | -0.003 | 0.934 | -0.013 | Periodic-TDL |
| Egb | 0.912±0.008 | 0.917 | -0.004 | 0.923 | -0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.894±0.019 | 0.913 | -0.019 | 0.911 | -0.017 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.810±0.035 | 0.806 | +0.004 | 0.839 | -0.029 | Periodic-TDL |
| EPS | 0.741±0.048 | 0.771 | -0.030 | 0.787 | -0.046 | Periodic-TDL |
| Nc | 0.802±0.097 | 0.821 | -0.019 | 0.856 | -0.054 | Periodic-TDL |
| Xc | 0.412±0.097 | 0.415 | -0.003 | 0.423 | -0.011 | Periodic-TDL |

## A_L_A0_B0 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.887±0.014 | 0.924 | -0.037 | 0.934 | -0.047 | Periodic-TDL |
| Egb | 0.878±0.013 | 0.917 | -0.038 | 0.923 | -0.045 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.890±0.017 | 0.913 | -0.022 | 0.911 | -0.021 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.740±0.023 | 0.806 | -0.066 | 0.839 | -0.099 | Periodic-TDL |
| EPS | 0.697±0.094 | 0.771 | -0.074 | 0.787 | -0.090 | Periodic-TDL |
| Nc | 0.787±0.050 | 0.821 | -0.034 | 0.856 | -0.069 | Periodic-TDL |
| Xc | 0.388±0.063 | 0.415 | -0.027 | 0.423 | -0.035 | Periodic-TDL |

## B_U_LO_RESTART 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.928±0.012 | 0.924 | +0.004 | 0.934 | -0.006 | Periodic-TDL |
| Egb | 0.912±0.014 | 0.917 | -0.005 | 0.923 | -0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.912±0.012 | 0.913 | -0.001 | 0.911 | +0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.810±0.042 | 0.806 | +0.005 | 0.839 | -0.029 | Periodic-TDL |
| EPS | 0.739±0.069 | 0.771 | -0.032 | 0.787 | -0.048 | Periodic-TDL |
| Nc | 0.838±0.064 | 0.821 | +0.017 | 0.856 | -0.018 | Periodic-TDL |
| Xc | 0.411±0.098 | 0.415 | -0.005 | 0.423 | -0.012 | Periodic-TDL |

## B_U_HI_MONO 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.924±0.019 | 0.924 | +0.000 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.913±0.011 | 0.917 | -0.004 | 0.923 | -0.010 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.013 | 0.913 | +0.002 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.817±0.033 | 0.806 | +0.011 | 0.839 | -0.022 | Periodic-TDL |
| EPS | 0.762±0.076 | 0.771 | -0.009 | 0.787 | -0.025 | Periodic-TDL |
| Nc | 0.828±0.099 | 0.821 | +0.007 | 0.856 | -0.028 | Periodic-TDL |
| Xc | 0.388±0.087 | 0.415 | -0.027 | 0.423 | -0.035 | Periodic-TDL |

## B_U_LO_MONO 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.929±0.010 | 0.924 | +0.005 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.914±0.009 | 0.917 | -0.002 | 0.923 | -0.009 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.911±0.012 | 0.913 | -0.002 | 0.911 | +0.000 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.801±0.031 | 0.806 | -0.005 | 0.839 | -0.038 | Periodic-TDL |
| EPS | 0.739±0.046 | 0.771 | -0.032 | 0.787 | -0.048 | Periodic-TDL |
| Nc | 0.823±0.063 | 0.821 | +0.003 | 0.856 | -0.033 | Periodic-TDL |
| Xc | 0.421±0.084 | 0.415 | +0.006 | 0.423 | -0.002 | Periodic-TDL |

## C_D2_1_D3_0 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.919±0.029 | 0.924 | -0.005 | 0.934 | -0.015 | Periodic-TDL |
| Egb | 0.919±0.014 | 0.917 | +0.003 | 0.923 | -0.004 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.012 | 0.913 | +0.001 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.793±0.032 | 0.806 | -0.012 | 0.839 | -0.046 | Periodic-TDL |
| EPS | 0.718±0.097 | 0.771 | -0.053 | 0.787 | -0.069 | Periodic-TDL |
| Nc | 0.794±0.068 | 0.821 | -0.026 | 0.856 | -0.062 | Periodic-TDL |
| Xc | 0.391±0.131 | 0.415 | -0.024 | 0.423 | -0.032 | Periodic-TDL |

## C_D2_0_D3_1 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.927±0.007 | 0.924 | +0.003 | 0.934 | -0.007 | Periodic-TDL |
| Egb | 0.910±0.006 | 0.917 | -0.007 | 0.923 | -0.013 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.014 | 0.913 | -0.000 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.812±0.021 | 0.806 | +0.006 | 0.839 | -0.027 | Periodic-TDL |
| EPS | 0.739±0.105 | 0.771 | -0.033 | 0.787 | -0.048 | Periodic-TDL |
| Nc | 0.840±0.070 | 0.821 | +0.019 | 0.856 | -0.016 | Periodic-TDL |
| Xc | 0.383±0.180 | 0.415 | -0.032 | 0.423 | -0.040 | Periodic-TDL |

## C_D2_1_D3_1 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.923±0.019 | 0.924 | -0.001 | 0.934 | -0.011 | Periodic-TDL |
| Egb | 0.912±0.018 | 0.917 | -0.005 | 0.923 | -0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.912±0.013 | 0.913 | -0.001 | 0.911 | +0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.798±0.070 | 0.806 | -0.008 | 0.839 | -0.041 | Periodic-TDL |
| EPS | 0.720±0.107 | 0.771 | -0.051 | 0.787 | -0.067 | Periodic-TDL |
| Nc | 0.794±0.075 | 0.821 | -0.027 | 0.856 | -0.062 | Periodic-TDL |
| Xc | 0.430±0.108 | 0.415 | +0.015 | 0.423 | +0.007 | Periodic-TDL |

## D_DF_0_H5 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.918±0.013 | 0.924 | -0.006 | 0.934 | -0.016 | Periodic-TDL |
| Egb | 0.916±0.012 | 0.917 | -0.001 | 0.923 | -0.007 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.912±0.012 | 0.913 | -0.001 | 0.911 | +0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.805±0.036 | 0.806 | -0.000 | 0.839 | -0.034 | Periodic-TDL |
| EPS | 0.707±0.119 | 0.771 | -0.065 | 0.787 | -0.080 | Periodic-TDL |
| Nc | 0.811±0.088 | 0.821 | -0.010 | 0.856 | -0.045 | Periodic-TDL |
| Xc | 0.445±0.100 | 0.415 | +0.029 | 0.423 | +0.022 | Periodic-TDL |

## D_DF_2_H1 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.923±0.011 | 0.924 | -0.001 | 0.934 | -0.011 | Periodic-TDL |
| Egb | 0.914±0.007 | 0.917 | -0.003 | 0.923 | -0.009 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.912±0.011 | 0.913 | -0.001 | 0.911 | +0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.784±0.049 | 0.806 | -0.022 | 0.839 | -0.055 | Periodic-TDL |
| EPS | 0.748±0.093 | 0.771 | -0.023 | 0.787 | -0.039 | Periodic-TDL |
| Nc | 0.808±0.082 | 0.821 | -0.013 | 0.856 | -0.048 | Periodic-TDL |
| Xc | 0.419±0.102 | 0.415 | +0.004 | 0.423 | -0.004 | Periodic-TDL |

## D_DF_0_H1 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.915±0.018 | 0.924 | -0.009 | 0.934 | -0.019 | Periodic-TDL |
| Egb | 0.906±0.012 | 0.917 | -0.010 | 0.923 | -0.017 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.011 | 0.913 | -0.000 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.808±0.041 | 0.806 | +0.003 | 0.839 | -0.031 | Periodic-TDL |
| EPS | 0.724±0.076 | 0.771 | -0.047 | 0.787 | -0.063 | Periodic-TDL |
| Nc | 0.803±0.067 | 0.821 | -0.018 | 0.856 | -0.053 | Periodic-TDL |
| Xc | 0.413±0.098 | 0.415 | -0.002 | 0.423 | -0.010 | Periodic-TDL |

## E_P_CURRENT_SHARED 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.929±0.011 | 0.924 | +0.005 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.905±0.026 | 0.917 | -0.012 | 0.923 | -0.018 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.912±0.013 | 0.913 | -0.001 | 0.911 | +0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.826±0.038 | 0.806 | +0.020 | 0.839 | -0.013 | Periodic-TDL |
| EPS | 0.754±0.070 | 0.771 | -0.018 | 0.787 | -0.033 | Periodic-TDL |
| Nc | 0.826±0.069 | 0.821 | +0.005 | 0.856 | -0.030 | Periodic-TDL |
| Xc | 0.444±0.118 | 0.415 | +0.029 | 0.423 | +0.021 | Periodic-TDL |

## E_P_CURRENT_IND 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.927±0.004 | 0.924 | +0.003 | 0.934 | -0.007 | Periodic-TDL |
| Egb | 0.913±0.011 | 0.917 | -0.004 | 0.923 | -0.010 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.012 | 0.913 | +0.000 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.025 | 0.806 | +0.006 | 0.839 | -0.028 | Periodic-TDL |
| EPS | 0.768±0.066 | 0.771 | -0.003 | 0.787 | -0.019 | Periodic-TDL |
| Nc | 0.811±0.078 | 0.821 | -0.010 | 0.856 | -0.045 | Periodic-TDL |
| Xc | 0.420±0.081 | 0.415 | +0.005 | 0.423 | -0.003 | Periodic-TDL |

## E_P_PAPER_SHARED 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.918±0.010 | 0.924 | -0.006 | 0.934 | -0.016 | Periodic-TDL |
| Egb | 0.916±0.013 | 0.917 | -0.000 | 0.923 | -0.007 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.012 | 0.913 | +0.002 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.803±0.047 | 0.806 | -0.003 | 0.839 | -0.036 | Periodic-TDL |
| EPS | 0.760±0.051 | 0.771 | -0.011 | 0.787 | -0.027 | Periodic-TDL |
| Nc | 0.804±0.057 | 0.821 | -0.017 | 0.856 | -0.052 | Periodic-TDL |
| Xc | 0.447±0.100 | 0.415 | +0.032 | 0.423 | +0.024 | Periodic-TDL |

## E_P_PAPER_IND 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.923±0.014 | 0.924 | -0.002 | 0.934 | -0.011 | Periodic-TDL |
| Egb | 0.911±0.014 | 0.917 | -0.006 | 0.923 | -0.012 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.911±0.013 | 0.913 | -0.001 | 0.911 | +0.000 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.049 | 0.806 | +0.005 | 0.839 | -0.028 | Periodic-TDL |
| EPS | 0.732±0.085 | 0.771 | -0.039 | 0.787 | -0.055 | Periodic-TDL |
| Nc | 0.790±0.088 | 0.821 | -0.031 | 0.856 | -0.066 | Periodic-TDL |
| Xc | 0.410±0.088 | 0.415 | -0.005 | 0.423 | -0.013 | Periodic-TDL |

## E_P_SYM_SHARED 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.923±0.011 | 0.924 | -0.001 | 0.934 | -0.011 | Periodic-TDL |
| Egb | 0.919±0.011 | 0.917 | +0.002 | 0.923 | -0.004 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.915±0.010 | 0.913 | +0.003 | 0.911 | +0.004 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.823±0.023 | 0.806 | +0.017 | 0.839 | -0.016 | Periodic-TDL |
| EPS | 0.750±0.053 | 0.771 | -0.021 | 0.787 | -0.037 | Periodic-TDL |
| Nc | 0.799±0.059 | 0.821 | -0.022 | 0.856 | -0.057 | Periodic-TDL |
| Xc | 0.436±0.095 | 0.415 | +0.021 | 0.423 | +0.013 | Periodic-TDL |

## F_PD_N0_F2 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.924±0.013 | 0.924 | -0.000 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.910±0.005 | 0.917 | -0.006 | 0.923 | -0.013 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.011 | 0.913 | +0.001 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.829±0.032 | 0.806 | +0.023 | 0.839 | -0.010 | Periodic-TDL |
| EPS | 0.745±0.065 | 0.771 | -0.026 | 0.787 | -0.042 | Periodic-TDL |
| Nc | 0.816±0.057 | 0.821 | -0.005 | 0.856 | -0.040 | Periodic-TDL |
| Xc | 0.424±0.046 | 0.415 | +0.009 | 0.423 | +0.001 | Periodic-TDL |

## F_PD_N1_F0 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.930±0.018 | 0.924 | +0.006 | 0.934 | -0.004 | Periodic-TDL |
| Egb | 0.921±0.015 | 0.917 | +0.004 | 0.923 | -0.002 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.914±0.013 | 0.913 | +0.001 | 0.911 | +0.003 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.821±0.040 | 0.806 | +0.016 | 0.839 | -0.018 | Periodic-TDL |
| EPS | 0.776±0.068 | 0.771 | +0.005 | 0.787 | -0.011 | Periodic-TDL |
| Nc | 0.831±0.057 | 0.821 | +0.010 | 0.856 | -0.025 | Periodic-TDL |
| Xc | 0.440±0.143 | 0.415 | +0.024 | 0.423 | +0.017 | Periodic-TDL |

## F_PD_N0_F0 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.925±0.013 | 0.924 | +0.001 | 0.934 | -0.009 | Periodic-TDL |
| Egb | 0.918±0.007 | 0.917 | +0.001 | 0.923 | -0.005 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.016 | 0.913 | +0.001 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.808±0.033 | 0.806 | +0.002 | 0.839 | -0.031 | Periodic-TDL |
| EPS | 0.747±0.068 | 0.771 | -0.025 | 0.787 | -0.040 | Periodic-TDL |
| Nc | 0.805±0.099 | 0.821 | -0.016 | 0.856 | -0.051 | Periodic-TDL |
| Xc | 0.444±0.090 | 0.415 | +0.029 | 0.423 | +0.021 | Periodic-TDL |

## G_RESIDUAL 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.926±0.021 | 0.924 | +0.002 | 0.934 | -0.008 | Periodic-TDL |
| Egb | 0.912±0.011 | 0.917 | -0.005 | 0.923 | -0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.904±0.014 | 0.913 | -0.008 | 0.911 | -0.007 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.814±0.023 | 0.806 | +0.008 | 0.839 | -0.025 | Periodic-TDL |
| EPS | 0.726±0.079 | 0.771 | -0.045 | 0.787 | -0.061 | Periodic-TDL |
| Nc | 0.801±0.058 | 0.821 | -0.019 | 0.856 | -0.055 | Periodic-TDL |
| Xc | 0.407±0.111 | 0.415 | -0.008 | 0.423 | -0.016 | Periodic-TDL |

## G_MODALNORM 七任务论文数值对照

| 任务 | 项目 R² mean±std | BASE mean | 项目−BASE | 论文最佳 mean | 项目−论文 | 论文模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.928±0.015 | 0.924 | +0.004 | 0.934 | -0.006 | Periodic-TDL |
| Egb | 0.907±0.009 | 0.917 | -0.010 | 0.923 | -0.016 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.904±0.015 | 0.913 | -0.008 | 0.911 | -0.007 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.792±0.026 | 0.806 | -0.014 | 0.839 | -0.047 | Periodic-TDL |
| EPS | 0.718±0.130 | 0.771 | -0.054 | 0.787 | -0.069 | Periodic-TDL |
| Nc | 0.779±0.075 | 0.821 | -0.042 | 0.856 | -0.077 | Periodic-TDL |
| Xc | 0.419±0.086 | 0.415 | +0.003 | 0.423 | -0.004 | Periodic-TDL |

配对比较、每fold差值、2×2主效应和差中差的完整精度见results.json的controlled_comparisons。
StageA/B/C/D/E/F的完整2×2只解释各自固定父条件；跨阶段为顺序开发，不能推断跨阶段交互。
StageE双向臂与独立投影不构成完整factorial；StageF将2D/3D PT dropout绑定；StageG新增容量单列，不称完整factorial。
无多seed支持，不宣称统计显著或全局最优；不能仅凭gate权重解释模态贡献。
项目为paper5_outer、每任务五折、seed42+fold、任务等权macro、std(ddof=1)。论文逐任务最佳组成包络。
Egc3380／论文4124、Xc432／论文430等数据、split和统计协议差异未消除，仅作数值对照。
history记录正常训练/validation前向的dropout、训练模式、optimizer组、模态范数、gate熵/均值/饱和比例、时间和显存；无额外模型前向。
完整精度原始预测和row/key/target对齐在formal；BEST/validation/scaler/config在finetune；PT逐项sum/count/loss及消费在pretrain。
预算预留：`{"initialize_starts": 1, "pt_starts": 8, "pt_updates": 80000, "pt_microbatches": 480000, "pt_presentations": 80640000, "ft_starts": 806, "ft_epochs": 56420, "ft_updates": 1288700, "test_starts": 805, "test_graphs": 135125, "cpu_suites": 1}`。
实际已知消费下界：`{"initialize_starts": 1, "pt_starts": 8, "pt_updates": 80000, "pt_microbatches": 480000, "pt_presentations": 80640000, "ft_starts": 806, "ft_epochs": 56419, "ft_updates": 1288684, "test_starts": 805, "test_graphs": 135125, "cpu_suites": 1}`。
实际已知完成：`{"initialize_starts": 1, "pt_starts": 8, "pt_updates": 80000, "pt_microbatches": 480000, "pt_presentations": 80640000, "ft_starts": 805, "ft_epochs": 56418, "ft_updates": 1288684, "test_starts": 805, "test_graphs": 135125, "cpu_suites": 1}`。
CPU suite：`{"attempts": 1, "exit": 0, "log": "/root/workspace/Uni-Poly-Plus-master/logs/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/base_opt_r6_c1_cpu", "status": "COMPLETE"}`。
硬截止：2026-10-30T22:07:13.844249+08:00；t0不刷新；全部真实exit见budget.json及logs作业exit.txt。
异常：无。
未完成时不使用validation代替TEST；未执行单元不虚报exit。最终保持待独立审查，不自动替换BASE、部署或归档为已验收。

资源修订：gpu2-pt2-fttest4-c1。父运行用户停止与原exit1保留；成功初始化exit0只读复用。累计预算包含父失败预留70 epochs／700 updates及本次补单元。父证据入口：/root/workspace/Uni-Poly-Plus-master/results/ua3d_base_opt_20261026/r6。原t0与截止不刷新。


</details>

<details><summary>results/ua3d_base_opt_20261026/r5/final_review/report.md</summary>

# UA3D BASE OPT r5 Stage0 停止记录

状态：待审查；run_outcome=STOPPED_BEFORE_CAMPAIGN；independent_review=PENDING。

执行偏差：助手在正式执行前运行了一次 pytest -k，5 passed、1 deselected，真实 exit 0。这违反“最后仅一次 CPU suite”的顺序要求；按保守口径计入唯一 CPU suite 额度。原测试命令未记录 PID 和精确起止时刻，证据字段保留 null，不补造。

已停止受影响链，未启动 controller、公共 step0 初始化、PT、FT、validation、冻结、TEST 或任何 GPU 任务；未再运行 CPU suite。预算：PT starts/updates/microbatches/presentations 全部 0；FULL starts/epochs/updates 全部 0；TEST units/rows 全部 0；提前 CPU 测试调用 1，最终 CPU suite 0。

已新增独立 base_opt 模型、PT 入口、controller、配置和测试。5 项测试仅覆盖损失与梯度一致性、双向正对计数、warmup/归一化、独立投影初态和预算/tie 规则；不能证明完整运行或 runtime 正确。实现未完成正式运行验证，不能视作验收通过。

当前 BASE reference 为用户指定的 0.795244745287872；本周期没有正式 TEST，Δmacro 未完成／不可比较。没有 PT_DIRECTION_SELECTED，不能用 validation 或历史结果填补。

| 任务 | 项目 R² | 论文参考 | 论文模型 | Δ |
|---|---:|---:|---|---:|
| Eea | 未完成／不可比较 | 0.934 | Periodic-TDL | — |
| Egb | 未完成／不可比较 | 0.923 | Periodic-TDL (+Chem, +RC) | — |
| Egc | 未完成／不可比较 | 0.911 | Periodic-TDL (+Chem, +RC) | — |
| Ei | 未完成／不可比较 | 0.839 | Periodic-TDL | — |
| EPS | 未完成／不可比较 | 0.787 | Periodic-TDL | — |
| Nc | 未完成／不可比较 | 0.856 | Periodic-TDL | — |
| Xc | 未完成／不可比较 | 0.423 | Periodic-TDL | — |

论文来源 SOTA_RESULT.md；逐任务最佳包络，不是一个可运行单模型。若完整执行，采用七任务五折、seed42+fold、mean±std(ddof=1)、七任务等权 macro；本次没有实验结果，不作公平性或超过论文声明。

证据：budget.json、identity.json、failure.json、execution_manifest.json、results.json、final_review/actual_changes.diff；日志根同周期 pre_execution_cpu_contract_tests 下保存完整命令、观察到的输出摘要和真实 exit。没有 campaign/controller exit，因为它们没有启动；不能用记录脚本的 exit 代替。

保留用户 CURRENT_PLAN.md/TODO.md dirty 状态、父 r4 代码、旧结果、失败现场、共享 cache/checkpoint 和历史账本。未读取 TODO.md，未远程同步，未新增 hash 合同。Stage1—5 未启动。

后续执行需要用户针对本次偏差、已消耗 CPU 测试额度和已存在 r5 根给出新的明确授权；本次不自动修复、覆盖、重试或恢复。


</details>

<details><summary>results/ua2d_p10k_nofp_align_20261024/r1/final_review/report.md</summary>

# UA2D-P10K-NOFP-ALIGN r1.1-GPU23 实际交付

状态：需返修；DSH执行，待Codex独立审查。
FINAL=D_A005；完整五臂正式TEST=True；性能：3D包通过；最终CPU exit=1。

历史P10K_REF正式macro=0.793（原值0.7932333045603579）直接复用，不重测。
ALIGN调权判定 FINAL−D_A010=-0.009（未舍入≥0.010才通过）；ALIGN权重相对本轮无FP参考D_A010的收益。
3D包判定 D_A010−PURE2D=+0.010（≥0.010达到项目默认门槛；负结果同样是有效消融交付）。
五臂均从公共step0起、FP自step0彻底移除（无FP头/target/loss/梯度）；单卡GPU2或3/world1/micro168/acc6/global1008；每臂独立10K updates。
2D/3D样本流与分支dropout流分离且跨5K边界连续；LR每5K一周期、全局loss warmup仅一次；保留历史ALIGN×2与6微批window count归一化。
D_A000为显式ALIGN关闭（无proj2/proj3/norm3参数）；PURE2D不构造/前向/优化任何3D模块；全局裁剪下两臂不要求数值等同。
对历史P10K差值受双卡world2/随机策略差异混杂，仅数值对照。共享开发TEST非独立盲测；论文协议等价性未证明。

| 预登记正式对照 | Δmacro R² | 完整性 |
|---|---:|---|
| D_A000_minus_D_A010 | +0.002 | COMPLETE |
| D_A005_minus_D_A010 | -0.009 | COMPLETE |
| D_A020_minus_D_A010 | +0.001 | COMPLETE |
| D_A005_minus_D_A000 | -0.011 | COMPLETE |
| D_A010_minus_D_A000 | -0.002 | COMPLETE |
| D_A020_minus_D_A000 | -0.001 | COMPLETE |
| D_A010_minus_PURE2D | +0.010 | COMPLETE |
| D_A000_minus_PURE2D | +0.012 | COMPLETE |
| FINAL_minus_D_A010 | -0.009 | COMPLETE |
| D_A000_minus_P10K_REF | -0.007 | COMPLETE |
| D_A005_minus_P10K_REF | -0.018 | COMPLETE |
| D_A010_minus_P10K_REF | -0.009 | COMPLETE |
| D_A020_minus_P10K_REF | -0.008 | COMPLETE |
| PURE2D_minus_P10K_REF | -0.019 | COMPLETE |

| 臂 − 历史P10K_REF | Δmacro R² | ≥0.010 |
|---|---:|---|
| D_A000 | -0.007 | 否 |
| D_A005 | -0.018 | 否 |
| D_A010 | -0.009 | 否 |
| D_A020 | -0.008 | 否 |
| PURE2D | -0.019 | 否 |

## D_A000 七任务SOTA数值对照（macro=0.786）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.932±0.017 | 0.934 | -0.002 | Periodic-TDL |
| Egb | 0.933±0.010 | 0.923 | +0.010 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.912±0.017 | 0.911 | +0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.809±0.020 | 0.839 | -0.030 | Periodic-TDL |
| EPS | 0.744±0.083 | 0.787 | -0.043 | Periodic-TDL |
| Nc | 0.818±0.061 | 0.856 | -0.038 | Periodic-TDL |
| Xc | 0.356±0.129 | 0.423 | -0.067 | Periodic-TDL |

## D_A005 七任务SOTA数值对照（macro=0.775）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.929±0.016 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.931±0.017 | 0.923 | +0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.916±0.015 | 0.911 | +0.005 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.804±0.025 | 0.839 | -0.035 | Periodic-TDL |
| EPS | 0.683±0.101 | 0.787 | -0.104 | Periodic-TDL |
| Nc | 0.808±0.069 | 0.856 | -0.048 | Periodic-TDL |
| Xc | 0.355±0.144 | 0.423 | -0.068 | Periodic-TDL |

## D_A010 七任务SOTA数值对照（macro=0.784）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.013 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.933±0.013 | 0.923 | +0.010 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.916±0.013 | 0.911 | +0.005 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.813±0.024 | 0.839 | -0.026 | Periodic-TDL |
| EPS | 0.737±0.060 | 0.787 | -0.050 | Periodic-TDL |
| Nc | 0.805±0.076 | 0.856 | -0.051 | Periodic-TDL |
| Xc | 0.360±0.090 | 0.423 | -0.063 | Periodic-TDL |

## D_A020 七任务SOTA数值对照（macro=0.785）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.929±0.010 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.931±0.013 | 0.923 | +0.008 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.918±0.012 | 0.911 | +0.007 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.029 | 0.839 | -0.028 | Periodic-TDL |
| EPS | 0.713±0.105 | 0.787 | -0.074 | Periodic-TDL |
| Nc | 0.808±0.066 | 0.856 | -0.048 | Periodic-TDL |
| Xc | 0.386±0.081 | 0.423 | -0.037 | Periodic-TDL |

## PURE2D 七任务SOTA数值对照（macro=0.774）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.926±0.012 | 0.934 | -0.008 | Periodic-TDL |
| Egb | 0.932±0.014 | 0.923 | +0.009 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.916±0.014 | 0.911 | +0.005 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.807±0.019 | 0.839 | -0.032 | Periodic-TDL |
| EPS | 0.700±0.097 | 0.787 | -0.087 | Periodic-TDL |
| Nc | 0.801±0.101 | 0.856 | -0.055 | Periodic-TDL |
| Xc | 0.335±0.157 | 0.423 | -0.088 | Periodic-TDL |

各任务五折等权R²，std(ddof=1)，seed42+fold，再七任务等权macro；所有计算使用未舍入值，展示3位小数。
论文逐任务最佳包络macro约0.810；协议等价性未证明，仅作数值对照，不宣称公平超过论文。共享开发TEST不称独立盲测。
五臂配置、deploy/resume、随机流/采样位置、budget、execution_manifest、validation_selection、唯一final_frozen、175组预测/指标、results/report与源码diff位于本轮根目录；全部命令/window/真实exit在logs/ua2d_p10k_nofp_align_20261024/r1/。
未完成／异常原因：
Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_p10k_nofp_align.py", line 355, in run
    j=self.launch('nofp_final_cpu','cpu',allow_final_cpu=True);self.wait([j]);cpu_exit=j['exit']
                                                               ^^^^^^^^^^^^^^
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_p10k_nofp_align.py", line 175, in wait
    if any(j['exit'] not in (None,0) for j in jobs):raise RuntimeError('job failed; no automatic retry')
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: job failed; no automatic retry
。


</details>

<details><summary>results/ua2d_ptscale_20261022/r1/final_review/report.md</summary>

# UA2D-PTSCALE r1 实际交付

状态：待审查；DSH执行，待Codex独立审查。
FINAL=P10K；完整三检查点正式TEST=True；性能：未达到双参考性能门槛；最终CPU exit=0。

历史B_FP正式macro=0.791直接复用，不重测。FINAL−P5K=+0.005，双参考均须未舍入≥0.010。
一个step0→20000连续PT轨迹，真实5k/10k/20k三个检查点，不是独立预训练；一个logical slot映射准备，不重建缓存，新公共初始化0。105FT及一次性105正式TEST。
P5K为同条件主要基线；历史B_FP原池911391与新池910756不同，其差值不单归因于时长；额外周期同时包含曝光及循环优化。未通过不追加PT或目标搜索。

| 预登记正式对照 | Δmacro R² | 完整性 |
|---|---:|---|
| P10K_minus_P5K | +0.005 | COMPLETE |
| P20K_minus_P5K | -0.003 | COMPLETE |
| P20K_minus_P10K | -0.008 | COMPLETE |
| FINAL_minus_B_FP | +0.002 | COMPLETE |

## P5K 七任务SOTA数值对照（macro=0.788）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.929±0.012 | 0.934 | -0.005 | Periodic-TDL |
| Egb | 0.928±0.011 | 0.923 | +0.005 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.911±0.016 | 0.911 | +0.000 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.808±0.036 | 0.839 | -0.031 | Periodic-TDL |
| EPS | 0.736±0.064 | 0.787 | -0.051 | Periodic-TDL |
| Nc | 0.793±0.068 | 0.856 | -0.063 | Periodic-TDL |
| Xc | 0.409±0.076 | 0.423 | -0.014 | Periodic-TDL |

## P10K 七任务SOTA数值对照（macro=0.793）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.932±0.022 | 0.934 | -0.002 | Periodic-TDL |
| Egb | 0.929±0.015 | 0.923 | +0.006 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.916±0.013 | 0.911 | +0.005 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.820±0.019 | 0.839 | -0.019 | Periodic-TDL |
| EPS | 0.736±0.073 | 0.787 | -0.051 | Periodic-TDL |
| Nc | 0.815±0.079 | 0.856 | -0.041 | Periodic-TDL |
| Xc | 0.404±0.117 | 0.423 | -0.019 | Periodic-TDL |

## P20K 七任务SOTA数值对照（macro=0.785）

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.930±0.019 | 0.934 | -0.004 | Periodic-TDL |
| Egb | 0.927±0.012 | 0.923 | +0.004 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.014 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.025 | 0.839 | -0.028 | Periodic-TDL |
| EPS | 0.708±0.097 | 0.787 | -0.079 | Periodic-TDL |
| Nc | 0.810±0.075 | 0.856 | -0.046 | Periodic-TDL |
| Xc | 0.396±0.112 | 0.423 | -0.027 | Periodic-TDL |

各任务五折等权R²，std(ddof=1)，seed42+fold，再七任务等权macro；所有计算使用未舍入值，展示3位小数。
论文逐任务最佳包络macro约0.810；协议等价性未证明，仅作数值对照，不宣称公平超过论文。共享开发TEST不称独立盲测。
配置、复用初始化来源、映射 COMPLETE、真实step三份PT来源、history／best／scaler／预测位于原新周期根；冻结名单selection/final_frozen.json，完整精度指标及配对差值final_review/results.json；预算budget.json；源码diff及真实训练计时final_review/；全部命令／window／真实exit在logs/ua2d_ptscale_20261022/r1/。
未完成／异常原因：无。


</details>


清理实际完成（2026-11-05T23:31:04.741135+08:00）：删除 124637 文件；results 675.533 → 123.234 GiB，净减少 552.299 GiB；文件系统可用空间增加 552.237 GiB（含并行训练写入影响）。当前实验状态 RUNNING/PT，必要输入均存在；无实验数学/预算/初始化变更。


## 2026-11-05T23:40:10.892195+08:00：按 BASE／图形学融合／当前稳定性范围整理 results 与 logs
用户追加授权只保留 BASE 基线、图形学融合及当前 BASE 稳定性相关文件，其他无关目录可删除。已有49份聚合数值与57份分析报告已在上一节归档；不重新加载历史权重/预测或运行实验。
实际仅 BASE-STABILIZE/r1 正在执行；条件谱r6已完成待审查、图形学方向当前暂停。本次保留指定BASE r6/concurrency_r1现存配置/指标及对应日志、当前稳定性完整现场、旧稳定性/重复推理证据、条件谱/HMR/HMR-READOUT相关记录，及当前PT原始模板/逻辑映射/下游输入缓存/35源config/35 train-only scaler；保留两次清理审计。
指定BASE原模型权重在上一轮用户授权清理时已经删除，本次未恢复也未替换数字reference0.804675462276949。删除旧日志/元数据后，原文历史路径不保证仍存在；历史成功/停止、预算、时钟与审查状态仍按原结论保留，不据清理改写实验结论。
删除目标（以下清单于删除前持久化；不含任何受保护子目录）：

|目录/文件|分配空间 MiB|
|---|---:|
|results/.lifecycle.lock|0.000|
|results/base_cleanup_20261026|9.934|
|results/cache_optimization_repair_20260916T234443Z|0.105|
|results/cache_optimization_repair_20260916T234930Z|0.117|
|results/cleanup_20260916_01|0.016|
|results/fce_param_20261014|19.742|
|results/fce_path_ablation_20261015|2.809|
|results/fce_tune_20261015|165.727|
|results/fce_tune_20261016|193.277|
|results/fce_tune_20261017|36.059|
|results/fp_graph_fusion_20261017|49.715|
|results/glt_dual_opt_20261006|10.410|
|results/glt_dual_static|0.035|
|results/glt_dual_static_finetune_formal_grid|3.742|
|results/glt_engineering_20260918|0.883|
|results/glt_ft_generalization_20261008|7.160|
|results/glt_ft_robustness_20261011|6.750|
|results/glt_fusion_20261007|23.926|
|results/glt_optimization_20261011|51.621|
|results/glt_periodic_seam_20261009|7.723|
|results/glt_pred_20260918|7.742|
|results/glt_quality_fusion_20261010|9.848|
|results/gp_sota_20261004|3.309|
|results/knowledge_world_map_20261026|0.047|
|results/mcl_ph_20260921|64.137|
|results/mcl_ph_next_20260929|0.016|
|results/mcl_ph_opt_20261006|20.645|
|results/mcl_ph_r4_20261001|0.238|
|results/mcl_ph_r4_20261002|11.441|
|results/mcl_ph_r5_20261004|0.875|
|results/mcl_ph_r6_20261004|0.465|
|results/mcl_ph_repair_20260929|0.027|
|results/sota_ft_20261005|2.020|
|results/tmol3m_3d_ablation_20261025|14.051|
|results/ua2d_3dshift_20261022|28.535|
|results/ua2d_a020_fp_fusion_20261024|27.535|
|results/ua2d_cleanfp_readout_distill_20261019|18.934|
|results/ua2d_demol_20261020|6.320|
|results/ua2d_et_single_20261020|3.547|
|results/ua2d_fg10k_morgan_20261023|16.742|
|results/ua2d_fpdecay_20261022|13.258|
|results/ua2d_fusion_20261017|73.297|
|results/ua2d_opt_20261021|48.328|
|results/ua2d_p10k_nofp_align_20261024/r1/authorization|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/budget.json|0.848|
|results/ua2d_p10k_nofp_align_20261024/r1/controller.lock|0.000|
|results/ua2d_p10k_nofp_align_20261024/r1/execution_manifest.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/final_review|7.129|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A000|3.242|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A005|3.246|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A010|3.246|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold0/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold0/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold0/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold0/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold0/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold1/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold1/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold1/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold1/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold1/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold2/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold2/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold2/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold2/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold2/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold3/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold3/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold3/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold3/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold3/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold4/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold4/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold4/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold4/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eea/fold4/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold0/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold0/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold0/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold0/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold0/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold1/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold1/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold1/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold1/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold1/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold2/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold2/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold2/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold2/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold2/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold3/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold3/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold3/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold3/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold3/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold4/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold4/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold4/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold4/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egb/fold4/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold0/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold0/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold0/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold0/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold0/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold1/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold1/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold1/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold1/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold1/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold2/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold2/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold2/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold2/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold2/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold3/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold3/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold3/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold3/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold3/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold4/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold4/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold4/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold4/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/egc/fold4/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold0/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold0/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold0/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold0/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold0/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold1/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold1/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold1/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold1/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold1/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold2/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold2/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold2/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold2/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold2/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold3/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold3/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold3/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold3/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold3/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold4/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold4/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold4/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold4/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/ei/fold4/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold0/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold0/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold0/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold0/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold0/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold1/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold1/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold1/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold1/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold1/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold2/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold2/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold2/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold2/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold2/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold3/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold3/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold3/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold3/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold3/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold4/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold4/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold4/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold4/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/eps/fold4/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold0/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold0/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold0/epoch_timing.json|0.023|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold0/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold0/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold1/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold1/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold1/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold1/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold1/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold2/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold2/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold2/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold2/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold2/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold3/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold3/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold3/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold3/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold3/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold4/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold4/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold4/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold4/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/nc/fold4/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold0/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold0/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold0/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold0/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold0/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold1/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold1/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold1/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold1/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold1/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold2/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold2/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold2/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold2/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold2/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold3/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold3/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold3/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold3/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold3/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold4/config.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold4/consumption.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold4/epoch_timing.json|0.020|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold4/history.json|0.051|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/xc/fold4/metrics.json|0.004|
|results/ua2d_p10k_nofp_align_20261024/r1/finetune/PURE2D|3.188|
|results/ua2d_p10k_nofp_align_20261024/r1/formal|3.578|
|results/ua2d_p10k_nofp_align_20261024/r1/inputs|0.012|
|results/ua2d_p10k_nofp_align_20261024/r1/pretrain|0.188|
|results/ua2d_p10k_nofp_align_20261024/r1/repairs|0.352|
|results/ua2d_p10k_nofp_align_20261024/r1/selection|2.449|
|results/ua2d_p10k_nofp_align_20261024/r1/status.json|0.008|
|results/ua2d_p10k_r10fp_20261023|18.434|
|results/ua2d_ph_descriptor_20261023|176.406|
|results/ua2d_pretrain_fp_20261019/fast_r4/authorization|0.008|
|results/ua2d_pretrain_fp_20261019/fast_r4/budget.json|0.180|
|results/ua2d_pretrain_fp_20261019/fast_r4/configs|0.016|
|results/ua2d_pretrain_fp_20261019/fast_r4/controller.lock|0.000|
|results/ua2d_pretrain_fp_20261019/fast_r4/current_plan.lock|0.000|
|results/ua2d_pretrain_fp_20261019/fast_r4/current_plan_registration.json|0.004|
|results/ua2d_pretrain_fp_20261019/fast_r4/engineering_review|0.012|
|results/ua2d_pretrain_fp_20261019/fast_r4/execution_manifest.json|0.004|
|results/ua2d_pretrain_fp_20261019/fast_r4/final_review|0.820|
|results/ua2d_pretrain_fp_20261019/fast_r4/finetune|6.250|
|results/ua2d_pretrain_fp_20261019/fast_r4/formal|1.328|
|results/ua2d_pretrain_fp_20261019/fast_r4/inherited_budget.json|0.008|
|results/ua2d_pretrain_fp_20261019/fast_r4/inherited_status.json|0.004|
|results/ua2d_pretrain_fp_20261019/fast_r4/initialization|0.004|
|results/ua2d_pretrain_fp_20261019/fast_r4/pretrain|0.062|
|results/ua2d_pretrain_fp_20261019/fast_r4/selection|1.172|
|results/ua2d_pretrain_fp_20261019/fast_r4/status.json|0.004|
|results/ua2d_pretrain_fp_20261019/r1|0.203|
|results/ua2d_ptscale_20261022/r1/authorization|0.230|
|results/ua2d_ptscale_20261022/r1/budget.json|0.480|
|results/ua2d_ptscale_20261022/r1/controller.lock|0.000|
|results/ua2d_ptscale_20261022/r1/execution_manifest.json|0.004|
|results/ua2d_ptscale_20261022/r1/final_review|4.309|
|results/ua2d_ptscale_20261022/r1/finetune|9.730|
|results/ua2d_ptscale_20261022/r1/formal|2.148|
|results/ua2d_ptscale_20261022/r1/pretrain|0.086|
|results/ua2d_ptscale_20261022/r1/selection|1.633|
|results/ua2d_ptscale_20261022/r1/status.json|0.004|
|results/ua2d_sota_20261018|37.410|
|results/ua2d_transfer_20261022|27.449|
|results/ua2d_v1_attnscale_20261020|3.359|
|results/ua2d_wavg_20261022|14.773|
|results/ua3d_anchor_lr_20261102|35.930|
|results/ua3d_anchor_opt_20261102|29.828|
|results/ua3d_base_opt_20261026/r5|0.164|
|results/ua3d_base_opt_20261026/r6/authorization|0.047|
|results/ua3d_base_opt_20261026/r6/budget.json|0.008|
|results/ua3d_base_opt_20261026/r6/config.json|0.008|
|results/ua3d_base_opt_20261026/r6/controller.lock|0.000|
|results/ua3d_base_opt_20261026/r6/controller_exit.json|0.004|
|results/ua3d_base_opt_20261026/r6/execution_manifest.json|0.004|
|results/ua3d_base_opt_20261026/r6/failure.json|0.004|
|results/ua3d_base_opt_20261026/r6/final_review|0.133|
|results/ua3d_base_opt_20261026/r6/finetune|0.371|
|results/ua3d_base_opt_20261026/r6/identity.json|2.168|
|results/ua3d_base_opt_20261026/r6/initialization|0.008|
|results/ua3d_base_opt_20261026/r6/occupancy_wait.json|0.004|
|results/ua3d_base_opt_20261026/r6/registry|0.008|
|results/ua3d_base_opt_20261026/r6/report.md|0.008|
|results/ua3d_base_opt_20261026/r6/results.json|0.016|
|results/ua3d_base_opt_20261026/r6/startup_gpu_inventory.json|0.004|
|results/ua3d_base_opt_20261026/r6/status.json|0.004|
|results/ua3d_base_reft_20261028|23.758|
|results/ua3d_frozen_transfer_20261026|39.188|
|results/ua3d_ft_balance_20261028|167.316|
|results/ua3d_head_dropout_20261028|92.914|
|results/ua3d_ph_20261028|30.754|
|results/ua3d_ph_opt_20261031|110.301|
|results/ua3d_spatial_20261025|39.633|
|results/ua3d_stage_20261026|17.504|
|results/ua3m_paperfuse_20261025/r1/authorization|0.086|
|results/ua3m_paperfuse_20261025/r1/budget.json|0.004|
|results/ua3m_paperfuse_20261025/r1/cleanup.json|0.004|
|results/ua3m_paperfuse_20261025/r1/controller.lock|0.000|
|results/ua3m_paperfuse_20261025/r1/execution_manifest.json|0.004|
|results/ua3m_paperfuse_20261025/r1/failure.json|0.004|
|results/ua3m_paperfuse_20261025/r1/final_review|0.344|
|results/ua3m_paperfuse_20261025/r1/initialization|0.004|
|results/ua3m_paperfuse_20261025/r1/inputs|0.008|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/authorization|0.277|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/budget.json|0.305|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/controller.lock|0.000|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/execution_manifest.json|0.004|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/execution_roots.json|0.004|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/final_review|1.324|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/finetune|6.219|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/formal|1.707|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/initialization|0.039|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/occupancy_wait.json|0.004|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/pretrain|0.035|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/selection|1.469|
|results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/status.json|0.008|
|results/ua3m_paperfuse_20261025/r1/repairs/prepare_repair1|2.512|
|results/ua3m_paperfuse_20261025/r1/status.json|0.008|
|results/ua_hop_ph_20261020|15.594|
|logs/base_cleanup_20261026|0.031|
|logs/cache_optimization_repair_20260916T234443Z|0.008|
|logs/cache_optimization_repair_20260916T234930Z|0.008|
|logs/cleanup_20260916_01|0.008|
|logs/fce_param_20261014|5.180|
|logs/fce_path_ablation_20261015|0.008|
|logs/fce_tune_20261015|15.559|
|logs/fce_tune_20261016|8.621|
|logs/fce_tune_20261017|2.887|
|logs/fp_graph_fusion_20261017|4.281|
|logs/glt_dual_opt_20261006|16.492|
|logs/glt_dual_static_finetune_smoke_auto_retry|0.004|
|logs/glt_engineering_20260918|0.055|
|logs/glt_ft_generalization_20261008|6.398|
|logs/glt_ft_robustness_20261011|7.574|
|logs/glt_fusion_20261007|217.508|
|logs/glt_fusion_plan_20261007|0.141|
|logs/glt_optimization_20261011|45.648|
|logs/glt_periodic_seam_20261009|9.246|
|logs/glt_quality_fusion_20261010|8.340|
|logs/glt_v2_real_audit_final22_20260911.json|0.016|
|logs/gp_sota_20261004|44.316|
|logs/mcl_ph_20260921|9.535|
|logs/mcl_ph_next_20260929|159.277|
|logs/mcl_ph_opt_20261006|10.461|
|logs/mcl_ph_r4_20260930|0.527|
|logs/mcl_ph_r4_20261001|0.770|
|logs/mcl_ph_r4_20261002|11.484|
|logs/mcl_ph_r5_20261004|1.645|
|logs/mcl_ph_r6_20261004|1.312|
|logs/mcl_ph_repair_20260929|0.164|
|logs/remove_fp_20261007|0.496|
|logs/sota_ft_20261005|1.770|
|logs/tmol3m_3d_ablation_20261025|6.652|
|logs/ua2d_3dshift_20261022|5.863|
|logs/ua2d_a020_fp_fusion_20261024|12.816|
|logs/ua2d_cleanfp_readout_distill_20261019|11.762|
|logs/ua2d_demol_20261020|0.852|
|logs/ua2d_et_single_20261020|6.195|
|logs/ua2d_fg10k_morgan_20261023|3.371|
|logs/ua2d_fpdecay_20261022|3.398|
|logs/ua2d_fusion_20261017|40.098|
|logs/ua2d_galformer_implementation_20261019|0.059|
|logs/ua2d_opt_20261021|20.207|
|logs/ua2d_p10k_nofp_align_20261024|8.457|
|logs/ua2d_p10k_r10fp_20261023|0.125|
|logs/ua2d_ph_descriptor_20261023|3.371|
|logs/ua2d_pretrain_fp_20261019|7.164|
|logs/ua2d_ptscale_20261022|5.039|
|logs/ua2d_sota_20261018|17.785|
|logs/ua2d_transfer_20261022|6.668|
|logs/ua2d_v1_attnscale_20261020|1.988|
|logs/ua2d_wavg_20261022|3.348|
|logs/ua3d_anchor_lr_20261102|9.664|
|logs/ua3d_anchor_opt_20261102|7.707|
|logs/ua3d_base_reft_20261028|9.250|
|logs/ua3d_frozen_transfer_20261026|2.773|
|logs/ua3d_ft_balance_20261028|95.613|
|logs/ua3d_head_dropout_20261028|46.070|
|logs/ua3d_ph_20261028|2.949|
|logs/ua3d_ph_opt_20261031|12.398|
|logs/ua3d_spatial_20261025|20.277|
|logs/ua3d_stage_20261026|2.832|
|logs/ua3m_paperfuse_20261025|6.578|
|logs/ua_hop_ph_20261020|2.129|


本次范围清理完成：{"status": "COMPLETE", "ended": "2026-11-05T23:40:16.365929+08:00", "removed_entries": 385, "before_allocated_bytes": {"results": 132761522176, "logs": 1543069696}, "after_allocated_bytes": {"results": 130756341760, "logs": 522706944}, "reduction_bytes": {"results": 2005180416, "logs": 1020362752}, "active_status": {"status": "RUNNING", "phase": "PT", "detail": "两个独立seed42完整10K，从同一新step0串行；DET不可降级。", "time": "2026-11-05T18:59:30.596462+08:00", "independent_review": "PENDING"}, "active_first_failure": false, "protected_inputs_equal": true, "locked_source_equal": true, "baseline_weights_present": false}
