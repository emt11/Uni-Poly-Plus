# Uni-Poly 项目实验周期归档

本文件记录已关闭周期的当时实现、正式结果与审查结论；当前授权和后续修复状态以 [CURRENT_PLAN.md](CURRENT_PLAN.md) 为准。下列 r13 数值是历史产物的观测值，不表示其角度监督、PH 系数域或当前八任务运行合同已通过修复后的验收。

## 2026-09-28｜MCL-PH-20260921-01 / r13-PAPER5：官方五折五臂实验收口

### 授权与范围

- 用户授权使用 Periodic-TDL 官方五折 `paper5_outer`，比较 `GLT_REF`、`O8_ONLY`、`M_CAT`、`M_GATE`、`M_XATTN` 五臂。
- 官方可比任务为 Eea、Egb、Ei、EPS、Nc；每臂五折，共 125 个 unit，最多 8,750 epochs。
- r12 项目自建五折运行已被取消并与本轮隔离；r12 失败现场不并入本轮结果。
- **scientific question**：在同一五任务五折评价下，三种 MCL-PH 融合臂的测试 R² 是否优于同期 GLT_REF 与 O8_ONLY？
- **reference / controlled change**：同期 GLT_REF、O8_ONLY 是项目内参照；CAT、GATE、XATTN 更换融合及对应预训练包，改变了多项因素，不能从臂间差值单独归因 PH 或交互效应。
- **comparison budget**：五臂各五任务×五折、每 unit 10 个 head-only 加 60 个 joint epoch，按 inner-validation RMSE 选优后读取一次 outer-test；原运行采用各臂对应的 5k 预训练包。项目内五臂按同一历史评价协议汇总；与论文表仅作数值对照。

### 执行与验收证据

- 结果文件：[results/mcl_ph_20260921/p2/paper5_official_r13_1/paper5_test.json](results/mcl_ph_20260921/p2/paper5_official_r13_1/paper5_test.json)。
- `status=PASS`、`units_expected=125`、`units_accepted=125`、`units_rejected=0`、`outer_test=RUN`。
- launcher 真实退出码为 `0`：[logs/mcl_ph_20260921/paper5_official_r13_1_launcher.exit](logs/mcl_ph_20260921/paper5_official_r13_1_launcher.exit)。
- 125 个 unit 的 `run.json`、`runtime.json`、`metrics.json`、`best.pt`、验证预测和测试预测均存在于 [results/mcl_ph_20260921/p2/paper5_official_r13_1/](results/mcl_ph_20260921/p2/paper5_official_r13_1/)。
- 本条为既有产物的归档；本轮未重新训练、未重新访问 outer-test、未做 OOF/refit、未增加 seed，也未逐个重新 strict-load 125 个 `best.pt`。
- 以上 `PASS` 是历史聚合合同及既有预测的验收状态，不等于修复后的数据、监督与 artifact 合同通过；当前已核实的限制见文末 2026-09-29 更正。

### Outer-test R²

| 臂 | macro5 outer-test R² |
| --- | ---: |
| GLT_REF | 0.8366660207475807 |
| O8_ONLY | 0.8357671864188265 |
| M_CAT | 0.8293898878677041 |
| M_GATE | 0.8294560980524395 |
| M_XATTN | 0.8366822551610775 |

### 与论文逐任务最佳均值的七任务对照

本项目列选用 r13 的 **M_XATTN** 历史 outer-test 预测 R²：每项为五个 outer fold 的 R² 算术平均，每折一次训练，未做额外 seed 汇总；`macro5` 是这五个任务均值的算术平均，不是七任务结果，也不是 pooled R²。数据／split 对齐仅已核实五个发布任务的历史行身份与官方折；本项目模型、预训练、几何及评估细节与论文各模型不完全一致，论文表的均值及标准差也不能与本项目单次五折均值视为相同统计口径。正差只表示本表数值较高，不代表公平比较或超过论文方法。项目结果与 fold 明细见上方 `paper5_test.json`，论文参考取自 [SOTA_RESULT.md](SOTA_RESULT.md) 的逐任务最佳 R² 均值。

| 任务 | 本项目 M_XATTN R² | 论文最佳模型 | 论文 R² 均值 | 差值：本项目−论文 |
| --- | ---: | --- | ---: | ---: |
| Eea | 0.924071 | Periodic-TDL | 0.934 | −0.009929 |
| Egb | 0.928568 | Periodic-TDL (+Chem, +RC) | 0.923 | +0.005568 |
| Egc | 未完成／不可比较：r13 无正式五折预测 | Periodic-TDL (+Chem, +RC) | 0.911 | — |
| Ei | 0.791783 | Periodic-TDL | 0.839 | −0.047217 |
| EPS | 0.733418 | Periodic-TDL | 0.787 | −0.053582 |
| Nc | 0.805572 | Periodic-TDL | 0.856 | −0.050428 |
| Xc | 未完成／不可比较：r13 无正式五折预测 | Periodic-TDL | 0.423 | — |

EAT 不在论文表中；Eib、Tg 不在当前项目任务集合中。Egc、Xc 虽有活动 split manifest，不能替代缺失的 r13 正式预测。

### 结果解释

- `M_XATTN - GLT_REF = +0.0000162344134968`；25 个同任务同折配对中 12 个为正，不能称明确或稳定增益。
- `O8_ONLY`、`M_CAT`、`M_GATE` 的 macro5 均低于 `GLT_REF`；`M_CAT` 与 `M_GATE` 分别低约 0.00728 和 0.00721。
- 五折曾参与项目路线开发，因此不能作为独立盲测；五臂差值不能单独归因于 PH 机制。
- 本轮没有候选满足继续扩展实验所需的稳定优势证据，MCL-PH 路线 STOP，不追加正式训练。

### 最终状态与后续

- r13-PAPER5 历史运行已结束，正式观测值保留；当时的停止追加训练决定不等于当前实现已通过正确性审查。
- 后续修复路线及其授权状态见 [CURRENT_PLAN.md](CURRENT_PLAN.md)；本条历史归档不授权重训或正式复评。


## 2026-09-29｜ROUTE-CONSOLIDATION-20260929-01 / r2：活动路线收敛与 `paper5_outer` 统一

### 授权与目标

- 用户授权将项目收敛为 GLT dual 参考路线与 MCL-PH 正式路线，关闭其余路线；本周期禁止训练、GPU、worker、缓存重建和正式实验。
- 八个任务 `eat / eea / egb / egc / ei / eps / nc / xc` 统一使用 `paper5_outer`。

### 实际架构与实现

- `scripts/create_mips_split_manifests.py` 现在只生成 `data/splits/mips_paper5_outer/`，固定
  `KFold(seed=100)` + sorted outer-train 上的 `train_test_split(seed=42)`，manifest 保存任务、样本顺序 SHA-256、五折索引、互斥和完整覆盖断言。
- GLT dual 保留 O8、3D、`concat`/`kfuse` 和现有微调逻辑；MCL-PH 保留
  `glt_ref/o8_only/m_cat/m_gate/m_xattn`，使用 Periodic-TDL head-only 10 epoch + joint
  阶段，按 validation RMSE 选优。EGC 为 60 总 epoch，其余七任务为 70；EAT 是项目适配任务。
- 新增 `src/training/split_protocol.py` 作为两条路线的 split 校验入口；`src/utils.py` 删除旧
  MTS trainer 分支，仅保留两条活动路线所需的 scaler、训练、评估和 scheduler 工具。
- MCL-PH 当前聚合器以八任务 Macro8 为期望范围，不能直接验收 r13 五任务的历史 metadata；活动配置和测试改用 `results/mcl_ph_20260921/shared_pretrain/` 下的共享初始化元数据。该目录缺失 `pretrain_split_v1.npz`，预训练恢复合同未完整核实。

### 删除与保留

- 删除 MTS/Distill、GLT-Galformer、GLT-PRED、Atomic-PC/EQ3D/DND、旧 PH fusion、退役
  MCL-PH P2/r10r3 入口及其配置、模块、测试、日志、结果和旧 split 目录。
- 保留活动路线代码、共享缓存、必要 checkpoint、Periodic-TDL 五任务来源快照、论文来源和
  历史审计文档；原始 `results/glt_pred_20260918/` 因缺失索引 artifact 暂保留为依赖证据。

### 验证与结果

- 八个 manifest 生成成功；Eea/Egb/Ei/EPS/Nc 五任务的索引与官方五折文件完全一致，八任务
  每折互斥且完整覆盖。
- `compileall` 和活动入口导入检查通过。
- 定向 CPU 测试：GLT dual/MCL-PH 合同与聚合 `93 passed`；MCL-PH view/pretrain `37 passed, 1 skipped`；缓存和读写 `82 passed, 3 skipped`；PH 独立定义 `63 passed`。这些是实现验证，不是性能实验。
- 未运行训练、GPU、worker、outer-test 或正式实验，未产生新的模型性能结果。

### 审查结论与后续

- 当时核对了 diff、活动 import 闭包、旧路线引用、manifest 和定向测试；这些验证只支持入口收敛与新 manifest 的局部实现结论，不能证明冻结 cohort 与新 split 的 SHA 绑定兼容，也不能证明 PH 目标正确。
- 本周期的实现与文档工作已结束，没有产生新正式预测；`pretrain_split_v1.npz` 缺失及后续发现的其他风险，按下方更正和当前计划处理。

## 2026-09-29｜历史归档更正：当前合同与 r13 结果的适用边界

此节是对已关闭周期结论的静态更正，不是新模型或实验周期；未修改历史预测、checkpoint、聚合文件或任何训练产物。依据 [CURRENT_PLAN.md](CURRENT_PLAN.md) 对旧实现与产物的核查；2026-09-29 本次文档复核时，工作树已有尚未独立审查的代码改动，不能把它们写成修复完成：

- r13 所用旧实现的 `mcl_ph_collate` 对角度键行使用原子偏移；旧 `persistence_pairs` 未显式指定 F2，本机 GUDHI 默认 F11。当前工作树已出现键偏移和显式 F2 的未提交修改，本次未对其运行测试或验收。r13 分数仍是旧实现的观测值，不能称为正确角度监督或 F2 PH 实现的性能；错误对真实训练的影响幅度未核实。
- 冻结 downstream cohort 绑定旧 split SHA；新 `paper5_outer` manifest 的文件 SHA 不匹配，旧 manifest 现已删除。五个历史任务的官方折索引对齐，不等于当前冻结 cohort 来源合同可运行。GLT_REF 新 predictor 的历史初始身份未保存；Xc 的当前 scaler 与学习率记录有不一致，但 r13 未运行 Xc。
- 除缺失的 `pretrain_split_v1.npz` 外，旧包的 PH 定义及缓存身份、历史五任务只读验收入口也待独立核实。先前“artifact 缺失是唯一未解决项”“暂无后续执行”的判断已失效；阶段 A 在当前计划中的状态仍为“待授权”，本次文档修改不改变授权或执行状态。具体范围、验证预算与停止条件以当前计划为准。
