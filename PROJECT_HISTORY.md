# Uni-Poly 项目实验周期归档

实验结果展示统一保留三位小数；选模、差值和门槛判断使用原始完整精度指标，舍入显示为0.000不代表原始差值严格为零。

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
| GLT_REF | 0.837 |
| O8_ONLY | 0.836 |
| M_CAT | 0.829 |
| M_GATE | 0.829 |
| M_XATTN | 0.837 |

### 与论文逐任务最佳均值的七任务对照

本项目列选用 r13 的 **M_XATTN** 历史 outer-test 预测 R²：每项为五个 outer fold 的 R² 算术平均，每折一次训练，未做额外 seed 汇总；`macro5` 是这五个任务均值的算术平均，不是七任务结果，也不是 pooled R²。数据／split 对齐仅已核实五个发布任务的历史行身份与官方折；本项目模型、预训练、几何及评估细节与论文各模型不完全一致，论文表的均值及标准差也不能与本项目单次五折均值视为相同统计口径。正差只表示本表数值较高，不代表公平比较或超过论文方法。项目结果与 fold 明细见上方 `paper5_test.json`，论文参考取自 [SOTA_RESULT.md](SOTA_RESULT.md) 的逐任务最佳 R² 均值。

| 任务 | 本项目 M_XATTN R² | 论文最佳模型 | 论文 R² 均值 | 差值：本项目−论文 |
| --- | ---: | --- | ---: | ---: |
| Eea | 0.924 | Periodic-TDL | 0.934 | −0.010 |
| Egb | 0.929 | Periodic-TDL (+Chem, +RC) | 0.923 | +0.006 |
| Egc | 未完成／不可比较：r13 无正式五折预测 | Periodic-TDL (+Chem, +RC) | 0.911 | — |
| Ei | 0.792 | Periodic-TDL | 0.839 | −0.047 |
| EPS | 0.733 | Periodic-TDL | 0.787 | −0.054 |
| Nc | 0.806 | Periodic-TDL | 0.856 | −0.050 |
| Xc | 未完成／不可比较：r13 无正式五折预测 | Periodic-TDL | 0.423 | — |

EAT 不在论文表中；Eib、Tg 不在当前项目任务集合中。Egc、Xc 虽有活动 split manifest，不能替代缺失的 r13 正式预测。

### 结果解释

- `M_XATTN - GLT_REF = +0.000`；25 个同任务同折配对中 12 个为正，不能称明确或稳定增益。
- `O8_ONLY`、`M_CAT`、`M_GATE` 的 macro5 均低于 `GLT_REF`；`M_CAT` 与 `M_GATE` 分别低约 0.007 和 0.007。
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


## 2026-10-02｜MCL-PH-NEXT-20260929-02 / r4：直接来源、快速预训练与七任务正式复评

### 授权、范围与审查

- 用户授权执行r4，并在最后明确要求立即停止旧慢任务、以最快速度自主完成预训练和微调。周期于2026-10-02 08:11:34 UTC结束训练，08:14:01 UTC独立审查PASS/exit0；[CURRENT_PLAN.md 的R4-20](CURRENT_PLAN.md)保存授权、现场、各阶段证据和停止/恢复实账。
- Scientific question/reference：同MCL包O8与完整XATTN的开发诊断、固定/可学习残差的单改动比较，以及整体路线对新GLT参考的验证筛选。D1–D3只使用Ei/EPS/Nc开发fold0/1；全局冻结后复评七任务。来源重建/语义修复和不同预训练模型/目标均与旧r13不同，不能把新旧差值当PH单因素效应。

### 实际架构与输入

- GLT_REF为O8+3D concat（预测head输入1024维），预训练atom/geometry/FP目标权重1/1/0.1、geometry_head_norm启用；FP仅为预训练目标，不是微调额外输入。
- MCL为O8+三个独立SchNet距离专家（2/3/4 Å）+PH路由器，前500步dense再Top-2；PH为F2、31个半径1.5..4.5 Å×5列，最大边4.5 Å、H0/H1。按图attention保持图内softmax和0.1固定残差，512维预测head；空图/BF16赋值修复保留。attention显存降低但对应benchmark延迟退化，未认定其提速。预训练继续atomCE+专家geometry+balance0.001，几何损失未增加通向router/XATTN的新监督，未新增全局预训练任务。
- PI1M direct训练911391独立样本，存量三聚体构象；同(global4×micro84×accum3=1008)和seed42的两个5000步包，各5040000噪声展示位置。复用现存F2 PH position缓存与train统计，不生成新构象、不把每分子固定缓存改成展示噪声协议。
- 下游使用已独立语义审计的direct source，`paper5_outer`七任务共5875样本（Eea368/Egb561/Egc3380/Ei370/EPS382/Nc382/Xc432）；所有正式折有保存的sample keys/indices/raw truth/prediction。

### 工程加速与验证

- 合并同一记录的decode/严格验证、复用本次已验证identity；排序后首个合法原子映射等价min(valid)，完整身份校验不变。CPU预取独立generator隔离模型RNG，GLT32/MCL64worker；normal只读LMDB readahead减少I/O等待。微调固定clean CPU缓存按次clone并fresh y，避免GPU迁移或标签缩放污染。
- 保存恢复通过真实四rank1000条日志逐值重放；所有CPU输入/PH/masks/labels逐值和RNG验证通过。相关mapping/IO、identity、finetune缓存/复用、campaign门槛测试分别15/28/6/4项通过。全仓测试、多seed及预算外候选未执行。
- 两个最终包实际step5000、next_position5040000，direct来源/采样/共同初始化/统计/缓存数组绑定、四rankRNG、finite模型/optimizer、截断scheduler及完整encoder导出已独立核实。原科学配方BF16、LR2e-4/wd0、mask0.30/noise0.03、warmup2000/schedule20000保留；5000步为20000 schedule截断，未缩短为提速。
- 微调FP32/batch24，seed42+fold，head10epoch后joint；Egc合计60epochs、其余70；train-onlyStandardScaler（Xc identity）。head初态按保存tensor审查：512维统一seed20260921；GLT原1024维head为seed42+fold。仅改变D2残差是否可学习；数据、标签、初始化、epoch、LR/wd和调度沿计划。

### 开发选择、预算和产物

- 开发23 starts/1610epochs，开发自身outer-test均NOT_RUN；D2收益−1.855%、1/3正，拒绝可学习残差；D3收益1.8693%、2/6正未过4/6门槛，Ei/EPS/Nc保留GLT。07:50:07 UTC全局冻结Eea为固定M_XATTN，另六项为GLT；任何正式test预测均在freeze之后，没有依据test扩大候选。
- 正式40逻辑units/2750epochs，其中6个GLT开发best零训练评估（原开发目录仍NOT_RUN），新34 starts/2330epochs；总57不重复微调starts/3940epochs，另有前置2epoch smoke。最终预训练10000步，取消585+250=835已记录步另计，物理完成记录至少10835步另有前置2步smoke及未核实在途步；取消开销公开保留。
- 63个开发/正式单元全部exit0及独立实际产物审查PASS；回读checkpoint有限性、初态一致、真实来源标签/keys、fold覆盖、仅train拟合scaler、history/best和预测复算。审查本身model calls/updates=0；首轮审查脚本head命名错误exit1已保留，02修正脚本后通过，没有改训练产物。
- 证据：[独立审查report](logs/mcl_ph_r4_20261002/final_review_02/report.json)、[运行账本](logs/mcl_ph_r4_20261002/fast_campaign_04/units.json)、[冻结路线](results/mcl_ph_r4_20261002/fast_campaign_04/frozen_selection.json)、[七任务实际结果](results/mcl_ph_r4_20261002/fast_campaign_04/seven_task_results.json)。预训练完整deploy/resume分别在 `results/mcl_ph_r4_20261002/r3_glt_ref_fast_ra_01/` 和 `r3_m_xattn_fast_01/`；训练/验收日志位于同名logs目录。
- 所有已有改动、失败/取消证据和删除状态保留；没有清理、pull、commit或push。此前历史周期架构与结果继续保留，不重写其科学含义。

### 正式预测与七任务论文数值对照

|任务|冻结路线|GLT R²均值|冻结路线R²均值|折间std（ddof=0）|论文参考模型|论文R²均值|项目−论文|
|---|---|---:|---:|---:|---|---:|---:|
|Eea|m_xattn|0.917|0.928|0.016|Periodic-TDL|0.934|-0.006|
|Egb|glt_ref|0.918|0.918|0.014|Periodic-TDL (+Chem, +RC)|0.923|-0.005|
|Egc|glt_ref|0.901|0.901|0.014|Periodic-TDL (+Chem, +RC)|0.911|-0.010|
|Ei|glt_ref|0.803|0.803|0.027|Periodic-TDL|0.839|-0.036|
|EPS|glt_ref|0.717|0.717|0.053|Periodic-TDL|0.787|-0.070|
|Nc|glt_ref|0.800|0.800|0.046|Periodic-TDL|0.856|-0.056|
|Xc|glt_ref|0.226|0.226|0.219|Periodic-TDL|0.423|-0.197|

- 所有项目均值为同一预登记 `paper5_outer` 的五折 R²算术均值，一组初始化规则 seed42+fold；不是 pooled R²、多seed均值或独立盲测。折间std采用ddof=0，不应当作论文多次实验的方差估计。
- 论文逐任务最佳模型/均值按现存 `SOTA_RESULT.md` 独立读取；差值=本项目冻结路线−论文，正值表示本项目数值较高。本轮七项差值均为负。论文与项目的数据版本、预训练预算和评估/统计协议等价性未核实，只作数值对照，不能宣称公平超过论文。
- Eea固定M_XATTN相对本轮GLT的R²均值差为+0.012；这是一组初始化的五折观测。GLT concat输入1024维、MCL输入512维，3D编码器、融合和预训练目标不同，不能归因于PH单因素或称稳定优势。Ei/EPS/Nc未晋级，最终保留GLT；本周期未验证这三个任务的正式预测改善。Xc差距最大（−0.197），具体原因未核实。
- 正式预测与每折metrics/checkpoint：`results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/{arm}/{task}/foldN/`；具体35个冻结路线prediction路径列于 [seven_task_results.json](results/mcl_ph_r4_20261002/fast_campaign_04/seven_task_results.json)，额外5个Eea GLT参考也完整保存。EAT/Eib/Tg不纳入七任务对照。

### 周期结论

已完成既定预训练/微调及工程加速，未证实整体预测改善或PH单因素收益。Eea相对本轮GLT的五折增益仅是本轮观测；七任务均低于静态论文参考，协议差异未核实。没有剩余排队训练，预算外损失、PH化学分型、周期/多构象和额外调参未执行。


## 2026-10-04｜MCL-PH-NEXT-20261004-01 / r5：Xc目标尺度对照与Nc路线确认

### 授权、实际实现及范围

- 用户明确授权执行CURRENT_PLAN第一执行包A0–A4/N3；Codex实施并独立审查。2026-10-04 10:35:36（Asia/Shanghai）训练/预测结束，10:41:04独立审查PASS/exit0。N4的FP辅助监督、PH消融、调度或周期几何未执行。
- Scientific question/reference：Xc同GLT5000步包只把identity目标改为train-only StandardScaler，比较r4 GLT同fold；Nc复用固定0.1残差M_XATTN，在新增开发fold2/3/4与r4 GLT确认。Nc属于整体路线对照，不是PH单因素消融。
- 实际架构保留r4：GLT O8+3D concat/1024维head；MCL O8+三个SchNet专家+F2 PH路由+图内XATTN/512维head。新增显式r5 scope、target-transform、frozen合同、promotion scaler身份及有界监督/验收入口。没有修改预训练模型或目标，没有新预训练、构象、PI1M、统计或PH trajectory。
- 数据为原direct source共5875样本，原paper5_outer、seed42+fold及各路线原head初态；FP32、batch24、head10+joint60、任务原LR/wd/dropout/schedule完全保留。Xc scaler仅各fold train拟合，在原尺度validation RMSE上选best。

### 开发决策、验证与预算

- Xc开发fold0/1相对validation RMSE收益+4.310%、+3.429%，均值+3.870%，通过原门槛。Nc新增fold2/3/4为−3.682%、−4.679%、−2.611%，均值−3.658%，0/3正，未晋级；不追加搜索或改门槛。全局冻结后只生成Xc候选正式test预测，Nc保留r4 GLT。
- 8次完整训练/560epochs/6510updates：开发5×70，Xc正式新增3×70；Xc开发fold0/1精确checkpoint复用、0新增更新。另有1次2epoch/24update smoke，共9训练starts/562epochs/6534updates。未改变六任务直接复用r4的30份正式预测；Nc候选本轮五折正式预测未执行。
- 局部14项测试通过。前置batch检查脚本因tuple API错误失败，实际1 forward/0 backward/0updates；修正后未重跑，单独零update backward未完成，由原预算内smoke覆盖实际有限forward/backward。fixture shape失败及后续修复日志也保留，不隐去失败开销。全仓测试、多seed及额外消融未执行。
- 独立CPU审查11个新保存单元、七任务35份最终预测；核对原始来源标签/keys/折索引、train-only scaler、初态逐张量、checkpoint有限性和配置、history/best/预算、冻结/test时间顺序和promotion权重，重算门槛及R²/RMSE。额外model calls/updates=0，PASS。

### 正式结果与七任务论文数值对照

|任务|来源／路线|R²均值|折间std|RMSE折均值|项目−r4|论文最佳模型|论文R²|项目−论文|
|---|---|---:|---:|---:|---:|---|---:|---:|
|Eea|r4 m_xattn|0.928|0.016|0.286|+0.000|Periodic-TDL|0.934|-0.006|
|Egb|r4 glt_ref|0.918|0.014|0.552|+0.000|Periodic-TDL (+Chem, +RC)|0.923|-0.005|
|Egc|r4 glt_ref|0.901|0.014|0.489|+0.000|Periodic-TDL (+Chem, +RC)|0.911|-0.010|
|Ei|r4 glt_ref|0.803|0.027|0.436|+0.000|Periodic-TDL|0.839|-0.036|
|EPS|r4 glt_ref|0.717|0.053|0.587|+0.000|Periodic-TDL|0.787|-0.070|
|Nc|r4 glt_ref|0.800|0.046|0.105|+0.000|Periodic-TDL|0.856|-0.056|
|Xc|r5 GLT standard|0.319|0.192|19.300|+0.093|Periodic-TDL|0.423|-0.104|

- 均值为paper5_outer五折R²算术均值，std为折间ddof=0，seed42+fold；RMSE列为原尺度五折均值，不是pooled或多seed统计。Xc是r5新结果，另六项明确为r4复用。七任务平均R²从0.756242到0.769584（+0.013）；Xc+0.093，fold3仍为−0.043，改善仅限当前五折观测，不称稳定优势。Nc新候选没有通过门槛，未证明正式改善。
- 论文最佳模型及R²按SOTA_RESULT.md逐项读取；项目−论文为差值，正值表示项目数值较高。七项仍均为负。论文数据/split/预训练与统计协议等价性未核实，只有数值对照；当前split已用于开发，不是独立盲测，不宣称公平超过论文。EAT/Eib/Tg不纳入七任务对照。
- 证据：[独立审查](logs/mcl_ph_r5_20261004/final_review_01/report.json)、[七任务结果及全部预测路径](results/mcl_ph_r5_20261004/seven_task_results.json)、[冻结路线](results/mcl_ph_r5_20261004/frozen_selection.json)、[运行账本](logs/mcl_ph_r5_20261004/campaign_01/units.json)、[CURRENT_PLAN N6/N7](CURRENT_PLAN.md)。结果根results/mcl_ph_r5_20261004/，日志根logs/mcl_ph_r5_20261004/；tmux Uni-Poly:mcl_r5_campaign及mcl_r5_final_review已结束。已有改动/日志/删除状态全部保留，无pull/commit/push或清理。

### 周期结论

第一执行包已完成并审查归档；Xc train-only目标标准化在本轮五折复评改善，Nc固定M_XATTN被新增三折验证拒绝。没有本轮排队或运行中的训练，N4后续机制候选未启动。

## R6周期结束与独立审查（MCL-PH-NEXT-20261004-02 / r1）

**状态：已完成；候选未晋级，保留r5。** 授权来源为本轮用户明确执行指令。scientific question是只降低Xc joint backbone LR能否改善r5 standard；reference为r5 GLT standard而非r4 identity；controlled change仅2e-4→1e-4，原包/初态、train-only scaler、source/split、seed42+fold、FP32、batch24、head10+joint60、head LR、wd、dropout与调度不变。无混合科学改动。

|fold|r5 validation RMSE|r6 validation RMSE|相对收益|r6 best epoch|
|---|---:|---:|---:|---:|
|0|17.904|17.904|+0.000%|4|
|1|17.398|16.778|+3.567%|54|

- 两折平均收益+1.784%，只有1/2为正；未达到预登记≥2%且2/2正。fold0 best仍为head阶段epoch4，该阶段不受joint LR改动影响。没有重新解释门槛或追加LR搜索。
- 新训练实际：2次完整/140epochs/1680updates；唯一smoke 2epochs/24updates；合计3个物理starts/142epochs/1704updates。无失败训练开销、额外GPU探针或预训练/构象/PH缓存更新。硬上限5次/350epochs/4200updates另加smoke未突破。
- 两个开发目录outer-test均NOT_RUN；冻结决策`results/mcl_ph_r6_20261004/frozen_selection.json`只允许保留r5、allowed_units为空。候选fold2/3/4、正式test和实际promotion未执行（门槛拒绝），没有新候选正式性能结论。七任务全部复用已验收预测，0新增推理；Xc复用r5，其余六任务为r5汇总中的r4来源。
- 实现：`scripts/finetune_mcl_ph.py`最小显式r6 LR覆盖，旧入口默认保持；`scripts/run_mcl_ph_r6.py`有界监督器，`scripts/audit_mcl_ph_r6.py`单元验收，`scripts/review_mcl_ph_r6.py`独立CPU审查，`tests/test_mcl_ph_r6.py`；候选值合同`results/mcl_ph_r6_20261004/candidate.json`。metrics/checkpoint/冻结/promotion绑定实际LR、完整配方、config实际内容、source metadata、split、逐折seed和原预训练包；实际LR组及每epoch调度逐值核实。
- 验证：local_tests_01 12 passed、gate_tests_01 5 passed（有重叠，不合计唯一数）；覆盖LR组/调度、旧默认、scope、冻结拒绝及CPU fixture零更新promotion。唯一GPU smoke PASS。独立审查`Uni-Poly:mcl_r6_final_review_02` exit0/PASS，报告`logs/mcl_ph_r6_20261004/final_review_02/report.json`：回读3个新单元与35份正式预测，核对raw labels/keys/split、train-only scaler、逐折初始head完全相同、模型finite、history/best、实际LR/调度和预算，复算全部R²/RMSE与论文表；0额外model calls/updates。实际promotion/新test冻结时序因拒绝分支无执行对象，只有CPU合同测试通过。
- 首轮独立审查final_review_01 exit1：全部检查完成后，报告中的NumPy bool无法JSON序列化；修为Python bool后在02重跑CPU审查。失败日志和当时脚本保留；没有修改训练产物或自动重试训练。
- 执行window：`Uni-Poly:mcl_r6_campaign`、`mcl_r6_0_smoke_xc_0`、`mcl_r6_0_development_xc_0`、`mcl_r6_1_development_xc_1`及审查window；全部已结束，无活动本轮trainer/supervisor。日志根`logs/mcl_ph_r6_20261004/`，逐单元`units/{phase}/xc_standard/foldN/`保留command/cwd/stdout/stderr/exit；账本`campaign_01/units.json`与实时/结束状态`campaign_01/status.json`。结果根`results/mcl_ph_r6_20261004/`，正式汇总`seven_task_results.json`列明35个复用预测路径。

### 七任务正式结果与论文数值对照（全部复用）

|任务|正式结果来源|项目R²|论文最佳模型|论文R²|项目−论文|
|---|---|---:|---|---:|---:|
|Eea|r4 m_xattn（复用）|0.928|Periodic-TDL|0.934|-0.006|
|Egb|r4 glt_ref（复用）|0.918|Periodic-TDL (+Chem, +RC)|0.923|-0.005|
|Egc|r4 glt_ref（复用）|0.901|Periodic-TDL (+Chem, +RC)|0.911|-0.010|
|Ei|r4 glt_ref（复用）|0.803|Periodic-TDL|0.839|-0.036|
|EPS|r4 glt_ref（复用）|0.717|Periodic-TDL|0.787|-0.070|
|Nc|r4 glt_ref（复用）|0.800|Periodic-TDL|0.856|-0.056|
|Xc|r5 GLT standard（复用）|0.319|Periodic-TDL|0.423|-0.104|

- 统计口径：paper5_outer，五折R²算术均值、seed42+fold，折间std ddof=0；RMSE为原尺度逐折指标。七任务平均R²仍0.769584，Xc仍0.319096，全部相对r5差值0。候选只有两折validation，不用它补齐正式test。
- 论文参考为SOTA_RESULT.md各任务最佳R²均值及对应模型；差值=项目−论文，正值表示项目较高。项目direct source与split已核实；论文数据版本、split、预训练预算和评估/统计协议等价性未核实，仅作数值对照，不称公平超过论文、独立盲测或稳定优势。已有test参与方案开发。EAT/Eib/Tg不纳入七任务表。
- 全仓测试、多seed、额外消融未执行。已有改动、四个文档删除状态、旧正式产物、失败/取消日志均保留；未删除任何文件，无pull/commit/push。r5完整交接保持。独立审查通过后归档PROJECT_HISTORY，本轮无待执行步骤。


## 2026-10-04｜GP-SOTA-20261004-01 / r3：Nc限定接续与异常保存保护

**状态：已完成并独立审查归档；GP/F0候选未晋级。** 本节关闭r3限定Nc周期及其工程返修。原r2 A包仍未完成，GP_A800步事故最终checkpoint未恢复，SOTA目标未达成。授权来源为用户“请你解决问题并执行计划！！！”及后续“请你直接执行该计划，无需授权，执行审查执行！”；Codex实施，未参与实现的独立agent审查，非人工第三方审查。

### 科学问题、实际架构与训练合同

- scientific question：已有GP/Nc fold0/1的收益能否在fold2/3/4确认；同容量GU两折用于路由机制诊断。reference：已核实历史Nc GLT／incumbent（同一glt_ref）。controlled change：GP/GU的PH自适应路由与必要均衡项；下游均F0。comparison budget：原direct source、paper5_outer、seed42+fold、相同train-only scaler、匹配原head初态与每折70epoch／770updates。历史GLT的四个FP step0仍未核实；GP对GLT含容量／融合／梯度路径／初始化混杂，不作PH单因素归因。
- 实际架构见[GPEncoder](src/modules/glt_mcl_ph.py)：保留O8 512维原子与Galformer风格512维物理键角通道；三个2／3／4 Å SchNet专家输出128维原子状态，PH trajectory驱动尺度权重，端点对称384维汇合经384→512→512 adaptor，以0.1残差加到键状态。中心原子／中心键池化后拼接1024维，性质head为1024→512→1。GP前500步dense、其后Top-2；GU等权1/3且balance为0。
- 输入为联合canonical topology、存量三聚体坐标／键角、PH trajectory／索引及validity；不向encoder输入属性真值或FP目标。预训练atom CE＋长度／角重建＋0.1 FP BCE，GP再加0.001 balance；GP_A只追加0.1 alignment但本周期不消费其失败包。下游输出标准化标量，逆变换回原尺度；几何无效分支按既有fallback规则处理。模型来源是项目改造，不称原论文完整复现。
- 205公共step0的新合同已验收；四个FP参数是当前确定性重建的新初态，历史相等未核实。复用已验收GP／GU各5000步完整包，4GPU／全局batch1008、lr2e-4、warmup2000、total schedule20000、bf16的原内容合同保持；原固定1024行validation每包一次，没有重跑。
- 微调F0：FP32、batch24、AdamW、train-only StandardScaler、标准化MSE；head10epoch（lr3e-4）、joint60epoch（backbone1e-4、head1e-3）、wd0.02、head dropout0.1、原两阶段cosine调度；验证按原尺度RMSE选best。未启用F1、额外seed／task／HPO。

### 实际实验与停止结论

- 旧GP/Nc fold0/1初筛平均validation RMSE收益+7.075%；r3仅新增GU/Nc fold0/1及GP/Nc fold2/3/4五个完整单元，共5starts／350epochs／3850updates。GU两折的GP相对收益为4.570646%、0.161%，平均2.365880%，仅开发观测，不称稳定机制收益。候选在确认折启动前锁定GP/F0，GU不能晋级。
- 新确认折相对GLT／incumbent收益−3.736%、−0.404%、−1.199%；三折均值−1.780%、0/3正、最差−3.736%。五折开发均值+1.762%不能替代三折均值≥1%、至少2/3正、最差≥−2%的联合要求，候选拒绝。没有新冻结权重或正式test预测，promotion／test-only均未执行。
- 累计本GP周期完整FT13starts／910epochs／10010updates；正式预训练3starts／2完整包／10800updates（GP5000、GU5000、GP_A800）。原pretrain smoke8updates、探针10次0updates、FT smoke2epochs／24updates、固定预训练validation2048行均未增加。七任务正式结果仍全部复用旧预测。GP_A未重试，0字节resume_00800.pt.tmp及历史RUNNING runtime作为事故现场保留，不能据此声称活动训练或最终状态已核实。

### 工程返修、验证及独立审查

- 先前R2-STOP-003修复将所有成功消费读盘核验移至gate解除前，成功发布后无读盘降级；十类四CPU rank与14项scope测试为已有保存证据。本次独立审查发现并完成四处最小返修：FT只向成功独占创建的输出写失败metadata；初始保存stat异常统一广播FAILED且未知冲突不改旧文件；接续控制器前置拒绝无ledger／终止权限，独占claim成功后只从自己的新日志完整command选择子进程；metadata.files须dict，非法schema闭合拒绝。科学模型／配方没有变动。
- 本次新增CPU验证19条：两类初始stat场景各一次四gloo rank（2 passed／5.85秒）；真实FT入口八类（8 passed／5.15秒）；真实控制器重入／claim／owner七条及malformed reader＋mock collective一条（8 passed／4.17秒）；完整command选择一条（1 passed／4.21秒）。无失败或重跑；第三方弃用warning保留。后两组停止行为只模拟ps／kill／collective，无实际进程终止。没有模型构造、GPU／forward-backward／optimizer或新增科学推理。生产NCCL、系统崩溃持久性未新增验证；gate删除不宣称持久化，崩溃后重现会保守拒绝。
- 独立agent未参与代码实施，直接核对原六文件diff和最终七文件返修diff／hash，回读十二类保存场景／新CLI证据、GP／GU完整包、七个Nc开发单元、35份正式预测、原始标签／split／scaler／head初态／完整LR轨迹／best与预算。未调用实施者audit、自检或review入口，未重跑科学实验。398保护文件哈希及无关改动保持。审查首轮脚本误把GP_A runtime预期为不存在而exit1，依据事故的实际遗留RUNNING记录更正后run_02 exit0，首轮证据保留；这不是科学运行重试。最终审查final_run_01另因将pytest current软链接计为额外场景而exit1，识别软链接后final_run_02通过；两次审查脚本更正记录与日志均保留，未改测试或实验文件。最终独立结论通过，允许归档，不放行训练。

### 七任务正式结果与论文数值对照

|任务|新GP正式结果|保留项目R²|论文最佳模型|论文R²|项目−论文|
|---|---|---:|---|---:|---:|
|Eea|未完成／不可比较：原r2停止，r3未新增该任务实验|0.928|Periodic-TDL|0.934|-0.006|
|Egb|未完成／不可比较：原r2停止，r3未新增该任务实验|0.918|Periodic-TDL (+Chem, +RC)|0.923|-0.005|
|Egc|未完成／不可比较：原r2停止，r3未新增该任务实验|0.901|Periodic-TDL (+Chem, +RC)|0.911|-0.010|
|Ei|未完成／不可比较：原r2停止，r3未新增该任务实验|0.803|Periodic-TDL|0.839|-0.036|
|EPS|未完成／不可比较：原r2停止，r3未新增该任务实验|0.717|Periodic-TDL|0.787|-0.070|
|Nc|未完成／不可比较：Nc确认未过门槛，test未执行|0.800|Periodic-TDL|0.856|-0.056|
|Xc|未完成／不可比较：原r2停止，r3未新增该任务实验|0.319|Periodic-TDL|0.423|-0.104|

七项保留结果全部来自既有35份已验收正式预测，paper5_outer五折R²算术均值、seed42+fold、折间std ddof=0；逐折数值、std和预测路径见[七任务汇总](results/gp_sota_20261004/seven_task_results_r3.json)与独立报告。上述表不是本轮新GP正式成绩。论文数据版本／split／训练和评估协议等价性未核实，仅作数值对照；历史test参与过开发，不称独立盲测、公平超过论文或稳定优势。EAT／Eib／Tg不纳入七任务表。

证据：[独立审查](logs/gp_sota_20261004/nc_r3_independent_review_01/report.json)、[原Nc门槛](results/gp_sota_20261004/nc_r3_confirmation_gate.json)、[最终返修diff](logs/gp_sota_20261004/nc_r3_closure_01/actual_repair.diff)、[预算账本](logs/gp_sota_20261004/status.json)、[当前交接](CURRENT_PLAN.md)。科学日志nc_r3_continuation_01、R3_Nc_GU_01及两折、R3_Nc_GP_01及三折全部exit0；CPU验证／独立审查／交接均在tmux Uni-Poly独立window，command／cwd／stdout／stderr／起止／exit保留。当前无活动trainer/controller，仅ACTIVE_READ_ONLY监控。旧日志／结果／删除状态保留，无删除／pull／commit／push。


## GLT-DUAL-OPT-20261006-01 / r1（2026-10-07，独立审查PASS关闭）

用户授权完整固定矩阵，并后续明确授权修复最终测试导入错误继续。工程/实验完整性独立PASS；所有候选未满足validation固定门槛，保留I，完整否定结果归档。论文协议等价性未核实，不能称达到SOTA、公平超越或稳定优势。

### 实际架构与初始化

保留O8周期原子编码器＋Galformer物理三聚体、原数据/split/初始化head/优化器调度。R_INCIDENT使用历史GLT parent，仅将中心内部键读出改为中心重原子入射键均值。B_FP从201张量旧common及本轮seed42生成FP四张量形成完整205 step0，不称已证明历史FP初态。A_ALIGN复制所有共同初态，移除FP并使用已有多正例ALIGN（不是追加FP）。F_TOKEN添加四头原子-query/物理键-key-value token注意力及残差；对所有重复单元对应原子副本实施3D元素遮蔽，与融合机制共同改变，存在混杂。M_TOKEN_CE与F相同结构/遮蔽/完整初态，只将atom CE读入融合原子状态，M−F作为直接对照而非组合交互效应。G_SEAM_GEO只改变几何监督，按中心端点数/2加权键长与覆盖中心共享原子的键角，保持内部读出。F/M新增融合seed20261006相同，共有张量精确复制，完整step0显式导出，实际预训练/导出/微调消费模式且严格加载。

预训练五包各5000updates，四rank、有效batch1008、bf16、LR2e-4、warmup2000、schedule20000、mask0.3、noise0.03。A ALIGN固定warmup/temperature，其他FP，原O8/geometry权重保持。下游每arm×七task×五fold，train-only StandardScaler、seed42+fold、FP32/batch24；head10+joint60（Egc joint50），head/global-best按原尺度validation RMSE，原LR/wd/dropout/schedule，无E5。

### 真实预算、测试、冻结及审查

215科学训练starts（5预训练＋210FT）、14400FTepochs、25000预训练updates＋308700FTupdates＝333700总updates，全部完成，训练失败重试0。正式210units，各best一次test-only、0updates。smoke/probe/benchmark/额外预训练validation均0。首个最终pytest因辅助导入失败5passed/2failed，用户后续授权最小修复后7passed；累计2calls，首失败证据未覆盖。独立agent发现旧CleanLabeledDataset无效opt_mode赋值，删除并静态复核，不新增动态测试，不影响实际走CleanDualDataset的210结果。独立报告保留该发现与局限。

selection_before_test于全部新test前冻结，严格相对主要参考及I的validation七任务平均收益≥1%、≥5/7任务正、最差任务≥−2%。全部候选未满足，chosen=I，不按test重选或拼接fold赢家；新test成绩仅完整对照。A主要参考平均0.9954%且I正4/7；F/M最差任务约−3.992%/−11.649%，R/G平均负。

实际证据：[独立审查](results/glt_dual_opt_20261006/r1/final_review/independent_report.json)、[210-fold汇总及预测路径](results/glt_dual_opt_20261006/r1/final_review/seven_task_results.json)、[自检](results/glt_dual_opt_20261006/r1/final_review/report.json)、[冻结selection](results/glt_dual_opt_20261006/r1/selection_before_test.json)、[预算](results/glt_dual_opt_20261006/r1/consumption.json)、[实际diff](logs/glt_dual_opt_20261006/r1/actual_code.diff)。日志根logs/glt_dual_opt_20261006/r1/，tmux Uni-Poly各campaign/pretrain/dev/formal/continuation_01/closeout_01/independent_review_01独立window均保存command/cwd/stdout/stderr/start/end/exit。旧修改、删除状态与历史结果保留，未pull/commit/push，不新增哈希验证。

### 保留部署路线I与论文七项对照

|任务|I R²|论文最佳模型|论文R²|I−论文|
|---|---:|---|---:|---:|
|Eea|0.928|Periodic-TDL|0.934|−0.006|
|Egb|0.918|Periodic-TDL (+Chem, +RC)|0.923|−0.005|
|Egc|0.901|Periodic-TDL (+Chem, +RC)|0.911|−0.010|
|Ei|0.803|Periodic-TDL|0.839|−0.036|
|EPS|0.717|Periodic-TDL|0.787|−0.070|
|Nc|0.800|Periodic-TDL|0.856|−0.056|
|Xc|0.319|Periodic-TDL|0.423|−0.104|

I来自历史正式预测：Eea r4 M_XATTN，其余GLT（Xc r5standard）。本轮全部新六组结果如下，不能据其中单任务数值重选部署。五折算术均值，seed42+fold，std ddof=0，非多seed；源/split/统计协议与论文等价性未核实，且历史test参与开发，不称独立盲测。原SOTA_RESULT未修改。

### 六组完整七任务正式结果

冻结选择：I；全部validation候选未满足所有门槛。新正式结果仅用于完整科学报告，未据test重选。

五折算术均值，seed42+fold，std ddof=0。论文协议等价性未核实，仅数值对照；不是独立盲测或多seed稳定优势。

#### R_INCIDENT

|任务|R²均值|R² std|RMSE均值|RMSE std|论文最佳模型|论文R²|项目−论文|
|---|---:|---:|---:|---:|---|---:|---:|
|eea|0.917|0.011|0.311|0.028|Periodic-TDL|0.934|-0.017|
|egb|0.921|0.013|0.542|0.054|Periodic-TDL (+Chem, +RC)|0.923|-0.002|
|egc|0.902|0.014|0.487|0.032|Periodic-TDL (+Chem, +RC)|0.911|-0.009|
|ei|0.803|0.020|0.435|0.074|Periodic-TDL|0.839|-0.036|
|eps|0.707|0.061|0.595|0.033|Periodic-TDL|0.787|-0.080|
|nc|0.814|0.054|0.101|0.009|Periodic-TDL|0.856|-0.042|
|xc|0.344|0.106|19.070|1.431|Periodic-TDL|0.423|-0.079|

相对H/I及主要参考的逐折收益、正收益折数和最差折见report.json；各五折R²/RMSE明细与实际预测路径见seven_task_results.json。

#### B_FP

|任务|R²均值|R² std|RMSE均值|RMSE std|论文最佳模型|论文R²|项目−论文|
|---|---:|---:|---:|---:|---|---:|---:|
|eea|0.917|0.012|0.311|0.032|Periodic-TDL|0.934|-0.017|
|egb|0.918|0.014|0.552|0.053|Periodic-TDL (+Chem, +RC)|0.923|-0.005|
|egc|0.901|0.014|0.489|0.032|Periodic-TDL (+Chem, +RC)|0.911|-0.010|
|ei|0.803|0.027|0.436|0.081|Periodic-TDL|0.839|-0.036|
|eps|0.717|0.053|0.587|0.038|Periodic-TDL|0.787|-0.070|
|nc|0.800|0.046|0.105|0.004|Periodic-TDL|0.856|-0.056|
|xc|0.319|0.192|19.300|2.373|Periodic-TDL|0.423|-0.104|

相对H/I及主要参考的逐折收益、正收益折数和最差折见report.json；各五折R²/RMSE明细与实际预测路径见seven_task_results.json。

#### A_ALIGN

|任务|R²均值|R² std|RMSE均值|RMSE std|论文最佳模型|论文R²|项目−论文|
|---|---:|---:|---:|---:|---|---:|---:|
|eea|0.928|0.015|0.289|0.033|Periodic-TDL|0.934|-0.006|
|egb|0.926|0.013|0.526|0.044|Periodic-TDL (+Chem, +RC)|0.923|+0.003|
|egc|0.915|0.016|0.453|0.039|Periodic-TDL (+Chem, +RC)|0.911|+0.004|
|ei|0.787|0.029|0.451|0.077|Periodic-TDL|0.839|-0.052|
|eps|0.747|0.042|0.557|0.052|Periodic-TDL|0.787|-0.040|
|nc|0.807|0.076|0.102|0.011|Periodic-TDL|0.856|-0.049|
|xc|0.357|0.179|18.745|2.114|Periodic-TDL|0.423|-0.066|

相对H/I及主要参考的逐折收益、正收益折数和最差折见report.json；各五折R²/RMSE明细与实际预测路径见seven_task_results.json。

#### F_TOKEN

|任务|R²均值|R² std|RMSE均值|RMSE std|论文最佳模型|论文R²|项目−论文|
|---|---:|---:|---:|---:|---|---:|---:|
|eea|0.924|0.018|0.295|0.029|Periodic-TDL|0.934|-0.010|
|egb|0.923|0.006|0.538|0.036|Periodic-TDL (+Chem, +RC)|0.923|-0.000|
|egc|0.908|0.014|0.472|0.031|Periodic-TDL (+Chem, +RC)|0.911|-0.003|
|ei|0.807|0.015|0.431|0.071|Periodic-TDL|0.839|-0.032|
|eps|0.739|0.049|0.564|0.044|Periodic-TDL|0.787|-0.048|
|nc|0.800|0.064|0.104|0.010|Periodic-TDL|0.856|-0.056|
|xc|0.354|0.174|18.807|2.116|Periodic-TDL|0.423|-0.069|

相对H/I及主要参考的逐折收益、正收益折数和最差折见report.json；各五折R²/RMSE明细与实际预测路径见seven_task_results.json。

#### M_TOKEN_CE

|任务|R²均值|R² std|RMSE均值|RMSE std|论文最佳模型|论文R²|项目−论文|
|---|---:|---:|---:|---:|---|---:|---:|
|eea|0.920|0.016|0.303|0.029|Periodic-TDL|0.934|-0.014|
|egb|0.938|0.007|0.483|0.024|Periodic-TDL (+Chem, +RC)|0.923|+0.015|
|egc|0.912|0.014|0.462|0.035|Periodic-TDL (+Chem, +RC)|0.911|+0.001|
|ei|0.789|0.021|0.449|0.067|Periodic-TDL|0.839|-0.050|
|eps|0.738|0.051|0.565|0.044|Periodic-TDL|0.787|-0.049|
|nc|0.811|0.056|0.102|0.007|Periodic-TDL|0.856|-0.045|
|xc|0.299|0.130|19.705|1.873|Periodic-TDL|0.423|-0.124|

相对H/I及主要参考的逐折收益、正收益折数和最差折见report.json；各五折R²/RMSE明细与实际预测路径见seven_task_results.json。

#### G_SEAM_GEO

|任务|R²均值|R² std|RMSE均值|RMSE std|论文最佳模型|论文R²|项目−论文|
|---|---:|---:|---:|---:|---|---:|---:|
|eea|0.926|0.016|0.292|0.030|Periodic-TDL|0.934|-0.008|
|egb|0.923|0.015|0.537|0.078|Periodic-TDL (+Chem, +RC)|0.923|+0.000|
|egc|0.906|0.013|0.478|0.028|Periodic-TDL (+Chem, +RC)|0.911|-0.005|
|ei|0.800|0.014|0.437|0.059|Periodic-TDL|0.839|-0.039|
|eps|0.736|0.039|0.568|0.018|Periodic-TDL|0.787|-0.051|
|nc|0.812|0.059|0.101|0.013|Periodic-TDL|0.856|-0.044|
|xc|0.310|0.123|19.533|1.568|Periodic-TDL|0.423|-0.113|

相对H/I及主要参考的逐折收益、正收益折数和最差折见report.json；各五折R²/RMSE明细与实际预测路径见seven_task_results.json。


## GLT-FT-GENERALIZATION-20261008-01 / r1（已完成）

归档时间：2026-10-09T00:18:17.470945+08:00；执行Codex；独立只读审查 `/root/final_readonly_review` PASS，仅本周期实际授权范围。原审查全文：[/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/final_review/independent_review_raw.txt](/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/final_review/independent_review_raw.txt)。授权来自用户2026-10-08明确指令，执行于2026-10-08 22:30:14～23:57:28（Asia/Shanghai）；未pull/commit/push。

实际架构保持旧B0：O8/Galformer各6层512维，concat1024维读出，1024→512 GELU/dropout→1预测头，无MD输入；唯一parent为旧B0 deploy_05000.pt，35fold head初态逐张量复用。参考为旧B0七任务五折validation与正式预测，未重跑。科学问题是三个独立单因素FT变化能否改善泛化：PARTIAL只开放joint最后两层encoder/读出LN/head；LOW_LR仅joint backbone初始LR×0.3；REG仅joint backbone WD加0.01。其余数据/split/seed42+fold/batch24FP32workers0/train-only scaler（含Xc）/head10epochs/joint Egc50其他60/调度与best选择不变。实际optimizer成员、冻结权重及头初态经独立审查核实。

**原训练过程中未执行PT validation，本次为事后补评**；旧周期需返修及原时序未满足事实不追溯改写。十二组resume_05000.pt完整任务头strict加载、eval/no_grad，原fixed P_val1024/组只评一次，四rank全局pool336/336/336/16，无padding；原mask/noise与独立MD/FGR RNG、MD0.5遮蔽视图保留，train-only统计只读。按原损失定义全球sum/count报告atom CE、原尺度几何误差及启用附加目标/有效计数。没有optimizer/scheduler/backward、checkpoint写回、MD/构象重算；不按不同目标配方总loss排名或改选。

实际消耗：新PT训练starts/updates=0/0；PT终点评估12/12288有效样本呈现；科学FT105starts、7200epochs、154350 attempted=completed optimizer updates；正式test105次，各一次；失败0/自动重试0。唯一最终工程pytest14passed、1warning、4.80s、exit0，覆盖新终点评估读取归约、PARTIAL冻结及LR/WD组；旧52passed复用不重跑。十二组补评与105FT validation完成后才pytest，成功后才105正式预测。

validation采用35fold原单位RMSE，逐fold相对收益任务均值再7任务macro；门槛macro≥1%、改善≥5/7、最差≥−2%。PARTIAL −4.704%/0/7/最差−11.642%；LOW_LR −3.112%/1/7/−6.773%；REG −0.674%/3/7/−1.926%，均未通过。冻结科学路线B0；B0对历史I +0.387%/4/7/−3.454%亦不通过，部署保留I。较高test观测未用于选择，不证明本轮方案获得符合门槛的过拟合改善。

代码实际delta：[/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/final_review/actual_code_diff.patch](/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/final_review/actual_code_diff.patch)。修改scripts/finetune_mcl_ph.py、scripts/run_glt_fusion_unit.py；新增scripts/evaluate_glt_pt_endpoint.py、run_glt_ft_generalization_unit.py、select_glt_ft_generalization.py、run_glt_ft_generalization.py、tests/test_glt_ft_generalization.py；本周期文档CURRENT_PLAN/PROJECT_HISTORY，保留此前工作树修改/删除。

日志：/root/workspace/Uni-Poly-Plus-master/logs/glt_ft_generalization_20261008/r1/（每单元command/cwd/start/end/stdout/stderr/真实exit）；tmux session Uni-Poly windows glt_ftg_r1_controller、ptval、ft、final、formal各独立执行，已全部结束。唯一工程测试证据/root/workspace/Uni-Poly-Plus-master/logs/glt_ft_generalization_20261008/r1/final/。冻结选择：[/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/selection_before_test.json](/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/selection_before_test.json)。PT分项、sum/count、有效目标计数：[/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/pt_endpoint/summary.json](/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/pt_endpoint/summary.json) 和summary.md，各组evaluation_spec/metrics/rank进度保留。逐折结果/预测路径：[/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/final_review/formal_fold_details.csv](/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/final_review/formal_fold_details.csv)；完整精度mean/std和逐fold：[/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/final_review/seven_task_results.json](/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/final_review/seven_task_results.json)。本周期正式预测/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/formal/<arm>/<task>/fold0～4/test_predictions.npz；旧B0复用results/glt_fusion_20261007/r1/formal/B0/<task>/fold0～4/test_predictions.npz。

以下为本周期完整七任务结果与SOTA_RESULT七项最佳参考模型的数值对照，保留原汇总口径：

### 本周期完整正式结果

三个候选各35fold正式TEST，共105次；B0为上一周期35次正式结果复用，无重放。五折算术mean±std(ddof0)，seed42+fold。

所有候选均未通过validation三门槛，冻结科学路线B0、部署I；test数字不用于改选。共享fold/旧test参与开发，非盲测或多seed稳定优势。论文Egc/Xc样本数与Xc scaling不同，精确split等价性未核实，仅作数值对照。

#### R² mean ± std

| 组 | eea | egb | egc | ei | eps | nc | xc | 宏R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.916 ± 0.012 | 0.926 ± 0.013 | 0.900 ± 0.018 | 0.803 ± 0.020 | 0.737 ± 0.073 | 0.802 ± 0.040 | 0.292 ± 0.198 | 0.768 |
| FT_PARTIAL | 0.917 ± 0.017 | 0.913 ± 0.018 | 0.897 ± 0.010 | 0.784 ± 0.038 | 0.719 ± 0.100 | 0.818 ± 0.055 | 0.338 ± 0.137 | 0.769 |
| FT_LOW_LR | 0.921 ± 0.012 | 0.915 ± 0.013 | 0.905 ± 0.014 | 0.808 ± 0.022 | 0.745 ± 0.058 | 0.811 ± 0.077 | 0.358 ± 0.157 | 0.780 |
| FT_REG | 0.917 ± 0.012 | 0.926 ± 0.013 | 0.903 ± 0.017 | 0.803 ± 0.020 | 0.743 ± 0.078 | 0.802 ± 0.040 | 0.351 ± 0.193 | 0.778 |

#### 原单位 RMSE mean ± std

| 组 | eea | egb | egc | ei | eps | nc | xc |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.311 ± 0.022 | 0.523 ± 0.034 | 0.492 ± 0.040 | 0.435 ± 0.076 | 0.561 ± 0.041 | 0.105 ± 0.007 | 19.664 ± 2.619 |
| FT_PARTIAL | 0.310 ± 0.028 | 0.570 ± 0.055 | 0.500 ± 0.021 | 0.452 ± 0.064 | 0.578 ± 0.068 | 0.100 ± 0.008 | 19.122 ± 2.016 |
| FT_LOW_LR | 0.302 ± 0.025 | 0.561 ± 0.036 | 0.481 ± 0.032 | 0.429 ± 0.071 | 0.555 ± 0.039 | 0.101 ± 0.014 | 18.785 ± 2.313 |
| FT_REG | 0.311 ± 0.022 | 0.523 ± 0.034 | 0.484 ± 0.039 | 0.435 ± 0.076 | 0.553 ± 0.047 | 0.105 ± 0.007 | 18.835 ± 2.792 |

#### 七项论文参考及候选差值

| 任务 | 论文最佳模型 | 论文R² | B0−论文 | PARTIAL−论文 | LOW_LR−论文 | REG−论文 |
|---|---|---:|---:|---:|---:|---:|
| eea | Periodic-TDL | 0.934 | -0.018 | -0.017 | -0.013 | -0.017 |
| egb | Periodic-TDL (+Chem, +RC) | 0.923 | +0.003 | -0.010 | -0.008 | +0.003 |
| egc | Periodic-TDL (+Chem, +RC) | 0.911 | -0.011 | -0.014 | -0.006 | -0.008 |
| ei | Periodic-TDL | 0.839 | -0.036 | -0.055 | -0.031 | -0.036 |
| eps | Periodic-TDL | 0.787 | -0.050 | -0.068 | -0.042 | -0.044 |
| nc | Periodic-TDL | 0.856 | -0.054 | -0.038 | -0.045 | -0.054 |
| xc | Periodic-TDL | 0.423 | -0.131 | -0.085 | -0.065 | -0.072 |

每组逐折指标/实际预测路径见 formal_fold_details.csv 及 seven_task_results.json；各候选预测目录为 `results/glt_ft_generalization_20261008/r1/formal/<arm>/<task>/fold0～4/test_predictions.npz`，B0目录为旧周期formal/B0。

PT事后补评见 [summary.md](/root/workspace/Uni-Poly-Plus-master/results/glt_ft_generalization_20261008/r1/pt_endpoint/summary.md)；原训练过程无PT validation事实保留，本次不伪造时序。

## GLT-PERIODIC-SEAM-20261009-01 / r1（已完成）

归档时间：2026-10-10T00:54:30.498004+08:00。用户明确授权Codex实现及完整实验；后续追加三卡/单卡/双卡资源调整、三个中断FT单元完整重跑及三位小数展示。授权实验于北京时间2026-10-09 23:22:27结束，独立只读审查代理`/root/periodic_seam_readonly_review`未参与实现，结论PASS；[审查原文](/root/workspace/Uni-Poly-Plus-master/results/glt_periodic_seam_20261009/r1/final_review/independent_review_raw.md)。审查复用已有日志/JSON/140份NPZ，未新增测试或实验。状态依据实际证据，部署I保留，未pull/commit/push。

实际架构：复用原O8/Galformer各6层、512维编码，token fusion后1024维图级表示与原两阶段属性预测头。共有初态张量复用旧common_step0，各fold预测头复用旧B0对应head_initial；新增Gaussian角度进入状态的Linear(128,512)，仅angle_value weight/bias共66048参数，seed20261009初始化一次并四臂共享。完整pretrainer参数40846686。无MD、ALIGN、FGR、local、KFuse、torsion实验。

F_RAW/F_CE采用finite_incident：完整开放Trimer物理键/线图，按中心原子全部入射键读出；内部长度权重1、左右接缝半权重，真实中心角各计一次。P_RAW/P_CE采用periodic_template：无向(i,j,s)键身份和无限覆盖邻接的一/二跳image关系，保留self-image、双入射、同编号不同shift及合法路径多重性；真实局部观测先分别编码再按权重汇聚canonical状态，预测映射回逐物理观测并保持同一监督查询/权重。角度投影进入状态而非仅attention bias；all_copies_unknown应用全部复制且周期类型来自masked元素。raw CE明确为负对照，PT fusion梯度为0；fused CE从融合原子状态监督，真实日志和唯一测试显示融合/3D梯度路径。新配置完整strict恢复，旧缺字段按legacy finite_internal/angle-off；专用FT入口适用新模型，未将旧通用入口宣称兼容。

科学问题：完整接缝/角度配方相对旧T_RAW/T0是否改善、周期局部模板相对F是否改善、fused CE相对raw是否改善及固定2×2组合差分。参考旧B0/T_RAW/T0只读复用，不重训或补评；F同时改变接缝读出/监督和角度增量，P同时改变周期状态组织和观测汇聚，不能归因单因素。四臂保持PT数据/采样/初态和5000更新、固定FT split/seed/epoch/原optimizer及头初态；计算成本并不完全相同。共享fold已用于开发，不称盲测或多seed稳定效果。

预算：四臂PT各5000，共20000updates、20160000训练呈现；step1000/2500/5000共12次fixed1024validation、12288呈现、无padding，不选checkpoint或早停。140完整FT选择矩阵：七任务五折、seed42+fold、batch24FP32workers0、train-only StandardScaler（含Xc）、head10+joint Egc50/其他60；9600epochs/205800updates。用户立即停止导致3个P_RAW/Egc/fold2、3、4中断，原现场保留并明确授权重跑；新目录finetune_gpu3_restart，共143starts、9764完整epochs、220724持久化记录updates。三个中断epoch0～273未持久化updates无法精确核实，实际累计区间220724～220997，在追加预算内。无自动重试或其他预算扩张。

全部FT validation完成后冻结选择，冻结结束北京时间23:07:59，首个正式预测23:08:09；140个正式预测各一次，无test重选。唯一CPU命令`timeout 300s python -m pytest -q tests/test_glt_periodic_seam.py`，CUDA_VISIBLE_DEVICES为空，19passed/1warning/11.42s/exit0，[原测试日志](/root/workspace/Uni-Poly-Plus-master/logs/glt_periodic_seam_20261009/r1/final/stdout.log)。旧测试和实验不重跑；独立审查未追加测试。

### Validation选择及正式结果

门槛：对B0逐fold RMSE相对收益按task平均再七任务等权macro，要求macro≥1%、改善≥5/7、最差≥−2%。正收益表示RMSE降低。

|候选|macro收益|改善任务数|最差任务收益|全部门槛|
|---|---:|---:|---:|---|
|F_RAW|-0.628%|3/7|-3.969%|未通过|
|F_CE|+2.381%|5/7|-4.400%|未通过|
|P_RAW|+0.033%|5/7|-4.062%|未通过|
|P_CE|+2.399%|5/7|-3.215%|未通过|

全部候选未过完整门槛，科学选择S=B0，部署I保留；正式test数值不用于修改选择。以下四臂为140次正式预测的五折R² mean±std(ddof0)，按用户要求展示三位小数，原完整精度保留。

|任务|F_RAW|F_CE|P_RAW|P_CE|
|---|---:|---:|---:|---:|
|Eea|0.917±0.010|0.924±0.018|0.923±0.013|0.928±0.019|
|Egb|0.924±0.016|0.941±0.012|0.929±0.013|0.936±0.006|
|Egc|0.904±0.018|0.914±0.012|0.902±0.016|0.910±0.012|
|Ei|0.803±0.023|0.799±0.023|0.789±0.013|0.808±0.018|
|EPS|0.723±0.054|0.776±0.046|0.717±0.073|0.707±0.064|
|Nc|0.781±0.057|0.821±0.059|0.819±0.048|0.815±0.064|
|Xc|0.317±0.197|0.324±0.160|0.277±0.124|0.309±0.125|

按B0 RMSE归一化的固定2×2组合差分D等权macro=0.036；定义与逐fold分布见本周期JSON。这是固定seed/fold组合观测，不是多seed稳定交互结论。

### 选定S与七任务论文参考

本轮S=B0，以下项目列只读复用旧B0正式结果，不是新增B0实验。

|任务|S R² mean±std|论文最佳模型|论文R²|项目−论文|
|---|---:|---|---:|---:|
|Eea|0.916±0.012|Periodic-TDL|0.934|-0.018|
|Egb|0.926±0.013|Periodic-TDL (+Chem,+RC)|0.923|+0.003|
|Egc|0.900±0.018|Periodic-TDL (+Chem,+RC)|0.911|-0.011|
|Ei|0.803±0.020|Periodic-TDL|0.839|-0.036|
|EPS|0.737±0.073|Periodic-TDL|0.787|-0.050|
|Nc|0.802±0.040|Periodic-TDL|0.856|-0.054|
|Xc|0.292±0.198|Periodic-TDL|0.423|-0.131|

统计口径：paper5_outer、seed42+fold、五折算术mean/std(ddof0)。共享validation/test参与开发；项目Egc3380/Xc432与所转录论文参考4124/430不同，精确split及Xc scaling等价性未核实；仅数值对照，不宣称公平超过论文。局部模板未证明远端欧氏坐标、全局周期构象或重新生成构象的RSIT等价性。这些不在本轮追加实验授权内。

实际修改：src/dataset/glt_dual.py、glt_dual_pretrain.py及新glt_periodic_geometry.py；src/modules/glt_dual.py、glt_dual_pretrain.py；src/training/glt_dual_runtime.py；scripts/pretrain_glt_dual.py、evaluate_glt_pt_endpoint.py、finetune_mcl_ph.py及新run_glt_periodic_seam.py、run_glt_periodic_seam_unit.py、select_glt_periodic_seam.py；新tests/test_glt_periodic_seam.py。用户展示追加修改collect_glt_dual_opt.py、run_mcl_ph_r4_fast.py、select_glt_periodic_seam.py的.3f格式以及七份结果Markdown，只改展示，原metrics及选模精度不变。工作树含先前修改，不能将整个HEAD diff算成本周期。CURRENT_PLAN/PIPELINE已完成交接，当前周期模型历史在独立PASS后才归档。

证据：日志根logs/glt_periodic_seam_20261009/r1，产物根results/glt_periodic_seam_20261009/r1；controller、controller_123、controller_gpu3因授权资源切换/停止exit143，最终controller_03 exit0。所有长命令/GPU/worker由tmux Uni-Poly独立window保留command/cwd/start/end/stdout/stderr/exit；最终窗口glt_ps_r1_controller_03已退出。未操作其他GPU占用进程。

[执行交付](/root/workspace/Uni-Poly-Plus-master/results/glt_periodic_seam_20261009/r1/final_review/delivery.md)；[四臂正式R²/RMSE及validation表](/root/workspace/Uni-Poly-Plus-master/results/glt_periodic_seam_20261009/r1/final_review/formal_results.md)；[原精度、配对/组合差异及预测路径](/root/workspace/Uni-Poly-Plus-master/results/glt_periodic_seam_20261009/r1/final_review/seven_task_results.json)；[test前选择冻结](/root/workspace/Uni-Poly-Plus-master/results/glt_periodic_seam_20261009/r1/selection_before_test.json)；[累计台账](/root/workspace/Uni-Poly-Plus-master/results/glt_periodic_seam_20261009/r1/consumption.json)；原中断resource_stop_gpu3.json、追加授权resume_gpu3.json及所有资源交接记录保留。

本轮已授权执行与审查事项均完成；无待运行单元，无需要追加的测试，未训练方案三/MD/ALIGN/FGR/KFuse/torsion或扩大任务/seed/调参。


## 2026-10-14｜GLT-OPTIMIZATION-20261011-01 / r1—r9 精简归档

迁移授权：用户本条明确要求重写 CURRENT_PLAN.md，并将原计划实验结果精简写入本文件。此节替代旧计划的长交接历史，不覆盖此前任何归档，不改旧 checkpoint／日志／预测。未关闭独立审查的 S3、N1 以执行摘要快照保留；S2 的审查来源是用户报告，不能冒充本轮已独立核实。S5 记录上一轮已完成的只读审查。

研究参考为历史 F_CE（宏 R² 0.786），统一 FT 融合参考 U0_FUSED 为 0.787，UA_2D 候选为 0.788；全部候选均未达到相对历史 F_CE 宏差严格大于 0.010 的开发门槛，研究参考与部署 I 未替换。原完整精度、配置及预测继续保存在各周期结果目录。

### 实际实现、状态与实耗

|周期|PT启动／updates|FT启动／epochs／updates|正式预测|唯一 CPU 测试|审查状态|
|---|---:|---:|---:|---|---|
|S1|0／0|70／4,800／102,900|70|7 passed，5.36秒，exit0|已完成；独立审查通过|
|S2|0／0|70／4,800／102,900|70|10 passed，7.34秒，exit0|执行完成；用户提供审查通过，过程未独立核实|
|S3|3／15,000|105／7,200／154,350|105|11 passed，12.82秒，exit0|执行完成；独立审查未关闭|
|U1|1／5,000|140／9,800／224,000|140|11 passed，10.01秒，exit0|已完成；独立审查通过|
|F1|0／0|35／2,450／56,000|35|6 passed，6.73秒，exit0|已完成；独立审查通过|
|N1|0／0|70／4,900／112,000|70|10 passed，4.62秒，exit0|执行完成；旧文档独立审查未关闭|
|S4|1／5,000|35／2,450／56,000|35|11 passed，9.69秒，exit0|已完成；独立审查通过|
|S5|1／5,000|35／2,450／56,000|35|9 passed，6.94秒，exit0|已完成；2026-10-14 对话独立审查通过|

八周期新增实耗合计：PT 6 次／30,000 updates；FT 560 次／38,850 epochs／864,150 updates；正式预测 560 次；最终 CPU pytest 8 次，均 exit0，各有一次依赖弃用 warning。复用旧参考不重复计入。

统计与控制：七任务各五折，seed42＋fold，std ddof=0，先任务内汇总再七任务等权宏平均；原单位 RMSE。S1—S3 沿用旧任务特定 FT（Egc joint50、Xc LR／WD／dropout 特例），U1／F1／N1／S4／S5 使用统一 uniform_u1（head10＋joint60、batch24、FP32、encoder dropout0、head dropout0.1、WD0.02、backbone LR1e-4、train-only StandardScaler；N1 是明确不缩放例外）。四 rank PT 与两 rank PT不能视为相同随机流；S3 还有端点容量差。共享 validation／TEST 已用于开发，论文数据／split／标签处理及统计协议等价性未完全核实，仅数值对照。

各周期实际实现：
- **S1**：joint 每图 p=0.20，分别关闭融合更新／三维读出；有效性分开，读出在 norm3 后置零，原无效几何不启用。
- **S2**：原 F_CE parent 上仅改变 FT 跨注意力连接；L 为真实入射键，B 保留全图候选并给入射项加 ln2。
- **S3**：三个两 rank PT；M_G 全图、M_L 入射、M_A 入射加编码前端点输入；三者都有新增 524288 参数端点模块，M_G／M_L 输入零，M_G 不等于原 F_CE 架构。
- **U1**：新增四 rank P_ALIGN：融合前 raw O8／geometry 图表示、双向多正 InfoNCE、跨 rank 可微 gather，两个投影共 328448 参数；FT 统一，纯二维路径仅 O8／norm2／512维 head。
- **F1**：复用原 F_CE parent，新增 rank64 图级乘性交互残差 98944 参数，末层零初态；二维侧已含 token fusion 信息；无新 PT。
- **N1**：固定原 parent／head 与 uniform_u1，只改成原单位标签 MSE，无 scaler 拟合或 head 仿射补偿；同配方下均劣于缩放对照。
- **S4**：两 rank 原 F_CE 路线新增物理有限 Trimer 四重原子路径；cos φ／cos(2φ)、退化 mask、有效均值、小 MLP 与零初态 scalar 共 33473 参数；不增加 torsion 预测目标。
- **S5**：两 rank 无 torsion F_CE，新增参数 0；公共 encoder／PT heads 初态与 S4 逐 tensor 一致，同 world／micro／accum／目标／统一 FT，复用 S4 保存结果做匹配分支比较。

### 七任务完整正式结果（R² mean±std）

八周期16个条件；另附旧先行D／G／DG三条件，后者未关闭审查且不计入八周期实耗。

|周期／arm|Eea|Egb|Egc|Ei|EPS|Nc|Xc|宏 R²|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|S1/F|0.932±0.021|0.935±0.010|0.908±0.019|0.802±0.009|0.765±0.063|0.808±0.052|0.299±0.127|0.779|
|S1/R|0.928±0.016|0.941±0.009|0.913±0.014|0.802±0.034|0.765±0.051|0.812±0.059|0.321±0.121|0.783|
|S2/L_FT|0.921±0.015|0.933±0.015|0.906±0.017|0.805±0.018|0.723±0.065|0.813±0.088|0.282±0.118|0.769|
|S2/B_FT|0.924±0.010|0.934±0.007|0.913±0.015|0.794±0.039|0.769±0.040|0.798±0.066|0.309±0.112|0.777|
|S3/M_G|0.924±0.020|0.935±0.006|0.912±0.013|0.808±0.016|0.715±0.067|0.796±0.057|0.302±0.106|0.770|
|S3/M_L|0.923±0.017|0.925±0.008|0.901±0.016|0.783±0.031|0.730±0.065|0.830±0.042|0.327±0.141|0.774|
|S3/M_A|0.922±0.011|0.929±0.007|0.908±0.014|0.792±0.040|0.751±0.052|0.824±0.046|0.321±0.147|0.778|
|U1/U0_2D|0.925±0.020|0.931±0.009|0.904±0.016|0.809±0.024|0.732±0.062|0.815±0.045|0.331±0.108|0.778|
|U1/UA_2D|0.927±0.016|0.926±0.010|0.905±0.016|0.803±0.029|0.768±0.061|0.813±0.065|0.373±0.107|0.788|
|U1/U0_FUSED|0.924±0.018|0.941±0.012|0.912±0.020|0.799±0.023|0.776±0.046|0.821±0.059|0.336±0.119|0.787|
|U1/UA_FUSED|0.924±0.017|0.933±0.005|0.909±0.017|0.805±0.030|0.697±0.091|0.803±0.079|0.326±0.087|0.771|
|F1/F1_MUL|0.921±0.016|0.937±0.008|0.914±0.013|0.804±0.034|0.793±0.033|0.792±0.060|0.294±0.127|0.779|
|N1/F_CE（不缩放）|0.927±0.016|0.936±0.012|0.913±0.019|0.763±0.077|0.755±0.055|0.828±0.046|0.231±0.163|0.765|
|N1/UA_2D（不缩放）|0.933±0.023|0.928±0.009|0.904±0.013|0.799±0.026|0.716±0.067|0.808±0.062|0.362±0.131|0.779|
|S4/T_FCE|0.924±0.020|0.932±0.016|0.917±0.011|0.797±0.022|0.737±0.018|0.791±0.067|0.370±0.107|0.781|
|S5/F_CE_2R|0.928±0.014|0.931±0.015|0.912±0.013|0.799±0.018|0.685±0.103|0.787±0.082|0.326±0.131|0.767|
|先行/D|0.925±0.015|0.941±0.006|0.909±0.014|0.799±0.023|0.762±0.055|0.846±0.061|0.343±0.105|0.789|
|先行/G|0.916±0.011|0.937±0.010|0.915±0.011|0.799±0.027|0.747±0.053|0.803±0.053|0.319±0.134|0.777|
|先行/DG|0.926±0.017|0.938±0.009|0.910±0.014|0.781±0.020|0.747±0.064|0.808±0.081|0.316±0.079|0.775|

### 原单位 RMSE mean±std

|周期／arm|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|S1/F|0.277±0.041|0.493±0.037|0.471±0.043|0.435±0.060|0.532±0.062|0.103±0.007|19.697±1.729|
|S1/R|0.289±0.032|0.471±0.035|0.458±0.034|0.435±0.075|0.534±0.049|0.101±0.010|19.373±1.602|
|S2/L_FT|0.302±0.031|0.499±0.050|0.477±0.037|0.430±0.054|0.581±0.063|0.099±0.013|19.936±1.574|
|S2/B_FT|0.297±0.026|0.499±0.036|0.459±0.035|0.443±0.078|0.534±0.062|0.105±0.013|19.578±1.610|
|S3/M_G|0.295±0.036|0.493±0.041|0.463±0.030|0.429±0.064|0.591±0.085|0.106±0.010|19.676±1.404|
|S3/M_L|0.298±0.038|0.528±0.016|0.490±0.036|0.454±0.073|0.570±0.041|0.097±0.008|19.244±1.854|
|S3/M_A|0.300±0.029|0.515±0.021|0.471±0.031|0.447±0.089|0.551±0.045|0.098±0.006|19.330±1.969|
|U1/U0_2D|0.293±0.039|0.509±0.033|0.483±0.036|0.428±0.075|0.570±0.057|0.101±0.006|19.246±1.421|
|U1/UA_2D|0.289±0.026|0.528±0.045|0.479±0.036|0.433±0.066|0.530±0.066|0.101±0.010|18.627±1.526|
|U1/U0_FUSED|0.295±0.029|0.471±0.061|0.460±0.048|0.439±0.074|0.522±0.041|0.099±0.008|19.157±1.448|
|U1/UA_FUSED|0.296±0.033|0.501±0.026|0.469±0.038|0.432±0.071|0.606±0.109|0.103±0.013|19.363±1.313|
|F1/F1_MUL|0.301±0.032|0.485±0.044|0.456±0.031|0.432±0.076|0.506±0.057|0.107±0.012|19.760±1.739|
|N1/F_CE（不缩放）|0.290±0.030|0.485±0.029|0.458±0.044|0.472±0.107|0.544±0.038|0.097±0.006|20.591±1.958|
|N1/UA_2D（不缩放）|0.274±0.043|0.520±0.035|0.482±0.027|0.438±0.062|0.590±0.083|0.102±0.011|18.744±1.573|
|S4/T_FCE|0.295±0.034|0.503±0.054|0.450±0.027|0.439±0.055|0.571±0.045|0.106±0.011|18.665±1.235|
|S5/F_CE_2R|0.288±0.028|0.509±0.078|0.461±0.029|0.437±0.059|0.618±0.117|0.107±0.015|19.283±1.726|
|先行/D|0.295±0.029|0.472±0.025|0.468±0.033|0.437±0.056|0.537±0.071|0.091±0.010|19.077±1.346|
|先行/G|0.312±0.026|0.485±0.049|0.453±0.026|0.439±0.079|0.554±0.054|0.104±0.010|19.382±1.939|
|先行/DG|0.293±0.037|0.481±0.031|0.466±0.034|0.457±0.065|0.556±0.076|0.101±0.013|19.502±1.118|

### 七任务论文参考与差值

|任务|论文最佳模型|参考 R²|
|---|---|---:|
|Eea|Periodic-TDL|0.934|
|Egb|Periodic-TDL (+Chem, +RC)|0.923|
|Egc|Periodic-TDL (+Chem, +RC)|0.911|
|Ei|Periodic-TDL|0.839|
|EPS|Periodic-TDL|0.787|
|Nc|Periodic-TDL|0.856|
|Xc|Periodic-TDL|0.423|

下表为项目 R² 均值减论文参考；所有差值先用未舍入值计算，再保留三位小数。正值仅表示数值较高。

|周期／arm|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|S1/F|-0.002|+0.012|-0.003|-0.037|-0.022|-0.048|-0.124|
|S1/R|-0.006|+0.018|+0.002|-0.037|-0.022|-0.044|-0.102|
|S2/L_FT|-0.013|+0.010|-0.005|-0.034|-0.064|-0.043|-0.141|
|S2/B_FT|-0.010|+0.011|+0.002|-0.045|-0.018|-0.058|-0.114|
|S3/M_G|-0.010|+0.012|+0.001|-0.031|-0.072|-0.060|-0.121|
|S3/M_L|-0.011|+0.002|-0.010|-0.056|-0.057|-0.026|-0.096|
|S3/M_A|-0.012|+0.006|-0.003|-0.047|-0.036|-0.032|-0.102|
|U1/U0_2D|-0.009|+0.008|-0.007|-0.030|-0.055|-0.041|-0.092|
|U1/UA_2D|-0.007|+0.003|-0.006|-0.036|-0.019|-0.043|-0.050|
|U1/U0_FUSED|-0.010|+0.018|+0.001|-0.040|-0.011|-0.035|-0.087|
|U1/UA_FUSED|-0.010|+0.010|-0.002|-0.034|-0.090|-0.053|-0.097|
|F1/F1_MUL|-0.013|+0.014|+0.003|-0.035|+0.006|-0.064|-0.129|
|N1/F_CE（不缩放）|-0.007|+0.013|+0.002|-0.076|-0.032|-0.028|-0.192|
|N1/UA_2D（不缩放）|-0.001|+0.005|-0.007|-0.040|-0.071|-0.048|-0.061|
|S4/T_FCE|-0.010|+0.009|+0.006|-0.042|-0.050|-0.065|-0.053|
|S5/F_CE_2R|-0.006|+0.008|+0.001|-0.040|-0.102|-0.069|-0.097|
|先行/D|-0.009|+0.018|-0.002|-0.040|-0.025|-0.010|-0.080|
|先行/G|-0.018|+0.014|+0.004|-0.040|-0.040|-0.053|-0.104|
|先行/DG|-0.008|+0.015|-0.001|-0.058|-0.040|-0.048|-0.107|

### 结果解释、停止／恢复及证据

- S1：分路径 dropout 未达标；组合差分只解释共享掩码配方，不证明稳定互补。S2：局部限制与 ln2 bias 均低于 F_CE。S3：M_L−M_G 为 +0.004、M_A−M_L 为 +0.004，但所有宏值均低于历史参考。
- U1：ALIGN 对二维路径宏增量 +0.010，对融合路径 -0.016。UA_2D FT 为 19,262,233 参数，融合模型为 39,876,344；forward 为纯二维，但旧 CPU 恢复临时构造 dual，reader 仍解码含 Trimer 的记录，独立二维工程交付留到 E0。
- F1：固定 rank64 残差未改善宏结果，不能外推为所有融合模块无效。N1：不缩放宏差相对匹配缩放条件分别为 −0.022／−0.009；只解释固定配方，不证明所有原单位训练不可用。
- S4／S5：T_FCE−F_CE_2R 宏差 +0.014，但 T 相对历史 F_CE为 −0.005；仍有 33,473 参数容量混杂。Xc TEST 差 +0.044（4／5折正），validation 差 -0.012（5／5折负）；EPS fold4 与 Xc fold2合计约83%的宏差。不能删除折、按任务拼接或据此确认稳定收益。
- U1 首次初始化因 head 父目录缺失 exit1，科学消耗0；用户授权后 recovery1 完整完成。N1 资源调度中断后按授权接管既有完成单元，只启动未启动单元；旧事故证据保留。S4 四卡方案被授权两卡资源修订覆盖，没有因此重跑历史 F_CE。S5 于北京时间2026-10-14 13:26:02结束、exit0；此前对话审查核实35个有限 best、70份 validation／TEST预测指标误差0、无 GPU 作业重叠、冻结时序与预算正确。

|周期|完整精度结果与执行交接|日志根|
|---|---|---|
|S1|[结果](results/glt_optimization_20261011/r1/S1/final_review/seven_task_results.json)／[交接](results/glt_optimization_20261011/r1/S1/final_review/delivery.md)|logs/glt_optimization_20261011/r1/S1/|
|S2|[结果](results/glt_optimization_20261011/r2/S2/final_review/seven_task_results.json)／[交接](results/glt_optimization_20261011/r2/S2/final_review/delivery.md)|logs/glt_optimization_20261011/r2/S2/|
|S3|[结果](results/glt_optimization_20261011/r3/S3/final_review/seven_task_results.json)／[交接](results/glt_optimization_20261011/r3/S3/final_review/delivery.md)|logs/glt_optimization_20261011/r3/S3/|
|U1|[结果](results/glt_optimization_20261011/r4/U1/final_review/recovery1/seven_task_results.json)／[交接](results/glt_optimization_20261011/r4/U1/final_review/recovery1/delivery.md)|logs/glt_optimization_20261011/r4/U1/|
|F1|[结果](results/glt_optimization_20261011/r5/F1/final_review/seven_task_results.json)／[交接](results/glt_optimization_20261011/r5/F1/final_review/delivery.md)|logs/glt_optimization_20261011/r5/F1/|
|N1|[结果](results/glt_optimization_20261011/r7/N1/final_review/recovery1/seven_task_results.json)／[交接](results/glt_optimization_20261011/r7/N1/final_review/recovery1/delivery.md)|logs/glt_optimization_20261011/r7/N1/|
|S4|[结果](results/glt_optimization_20261011/r6/S4/final_review/seven_task_results.json)／[交接](results/glt_optimization_20261011/r6/S4/final_review/delivery.md)|logs/glt_optimization_20261011/r6/S4/|
|S5|[结果](results/glt_optimization_20261011/r9/S5/final_review/seven_task_results.json)／[交接](results/glt_optimization_20261011/r9/S5/final_review/delivery.md)|logs/glt_optimization_20261011/r9/S5/|

旧先行证据另计：FT-ROBUSTNESS 的 D／G／DG 宏 R² 为 0.789／0.777／0.775，105 FT／7200epochs／154350updates／105正式预测，最终5 passed、5.60秒、exit0；数值已有核对，整个周期审查／归档未关闭。[结果](results/glt_ft_robustness_20261011/r1/final_review/seven_task_results.json)。严格收敛筛选曾把 PT train 从911391减到9623，下游仅319／3655结构保留3D，同时改变分布与覆盖；r2最小工程返修通过不等于整个质量周期已关闭。[质量汇总](results/glt_quality_fusion_20261010/r1/quality_completed/summary.json)／[工程审查](results/glt_quality_fusion_20261010/r2/final_review/independent_review.md)。这些旧周期不混入上面八周期总额。

旧计划覆盖关系：已执行 S1／S2／S3／U1／F1／N1／S4／S5 不再列为待运行；未实施 E0／E1、二维 ALIGN 后续、FiLM／两 token attention、有限非键距离、PT schedule／图级读出目标及保持覆盖的质量权重均迁入 CURRENT_PLAN.md r10。旧局部结构与路径 dropout 组合仍保留为0预算候选。用户本条仅授权两份文档修改，新代码／实验消耗0；模型及部署不变。

## 2026-10-17｜FCE-SOTA 与 FP-GRAPH 结果迁移：UA_2D 新计划的历史依据

### 授权、状态与证据边界

用户当前明确要求覆盖 CURRENT_PLAN.md，并将原计划实验结果写入本文件。本节迁移原 FCE-SOTA-20261015-01 的 T1—r16 与 FP-GRAPH-20261017/f1 结果，同时保留原参数周期及 U1／UA_2D 参考。此前归档全部保留。旧 GLT-OPTIMIZATION 的架构路线 r10／E0／A1 与本节“r10 全局预测融合”是不同周期，不能混称。

**审查通过的周期归档实际结论；其余按用户此次迁移要求保存为待审查执行快照，不视为关闭周期或独立审查通过。** T1／T2、r15 的独立审查状态沿用被替换交接的明确记录，本轮没有重复独立审查；T3a 科学审查通过、交付补交在 T3b 执行后复核完成，不把补交复核升级成全周期独立审查。r16 与 FP-GRAPH f1 的独立审查仍未进行，后续交接保留在 CURRENT_PLAN.md。

本轮直接读取14份完整七任务正式汇总、原参数周期汇总、U1汇总、Xc／Ei单任务正式汇总、completion与已有CPU日志，按完整精度格式化下面的数值及论文差。没有重算原预测指标、重放模型、重新strict恢复checkpoint、运行pytest、生成FP、拟合或新增实验；本节不是一次新的完整科学审查。既有JSON中的NOT_PERFORMED／待审查保留原状，不覆盖旧数据、split、checkpoint、预测、日志或事故记录。

### 科学问题、实际实现与已消费预算

以下每行检验该 controlled change 是否改善所列 reference；comparison budget 固定原周期的数据／split、各fold初态／训练与评价条件。表中 FT 数为新训练，不把复用结果重复计数；PT updates 是DDP逻辑更新，不按rank重复相加。Xc／Ei仅各自单任务五折，其余完整周期为七任务五折。

|周期|reference 与 controlled change（scientific question）|新 PT／updates|新 FT／epochs／updates|新正式单元|新 KRR fit|已有 CPU 结果|交接状态|
|---|---|---:|---:|---|---:|---|---|
|T1|原 F_CE／R0；改变 head dropout 与 backbone WD，再本 fold validation 选配方|0|140／9,600／205,800|62|0|10 passed|已完成；原交接记录独立审查通过|
|T2|原 F_CE／R0；仅 Galformer joint 初始 LR×0.3|0|70／4,800／102,900|70|0|12 passed|已完成；原交接记录独立审查通过|
|T3a／r4|同 step0、两 rank、R0；schedule_total_steps 20000 对 5000|2／10,000|70／4,800／102,900|70|0|7 passed|科学独立审查通过；交付曾需返修，T3b 补交复核完成|
|T3b／r5|复用 T3_REF；只把 geometry loss 权重1改0.5|1／5,000|35／2,400／51,450|35；复用35|0|10 passed|待审查；执行后复核通过|
|r6|G05／R0；只减小 Galformer joint LR|0|35／2,400／51,450|35；复用35|0|10 passed|待审查；执行后复核通过|
|r7／Xc|固定 G05；Xc 的 Morgan radius2 KRR＋0.5融合|0|0|5|15|8 passed|待审查；执行后复核与研究应用判断通过|
|r8／Ei|固定 G05；Ei 复用同一结构核规则与0.5融合|0|0|5|15累计|11 passed|待审查；两次事故后接续完成并复核|
|r9|固定 G05；扩展七任务统一 KRR／α选择／0.5输出政策|0|0|25；复用10|75|9 passed|待审查；执行后复核通过|
|r10|固定 r9 双分支及 α；仅 validation 选一个全局 w|0|0|35数组算术单元|0|7 passed|待审查；只读结果复核通过|
|r11|固定 w=0.75；α选择目标从0.5融合对齐到0.75|0|0|8；复用27|8|7 passed|待审查；执行后只读复核通过|
|r12|固定神经／α／w；Morgan radius2改radius3|0|0|35|35|6 passed|待审查；执行后只读复核通过|
|r13|G05／R0；MSE改为2×Huber(delta1)，固定0.75后融合|0|35／2,400／51,450|35|0|14 passed|待审查；执行后只读复核通过|
|r14|G05／R0／MSE；joint仅更新encoder LayerNorm affine与head|0|35／2,400／51,450|35|0|16 passed|待审查；执行后只读复核通过|
|r15|G05匹配续训；CONT_REF对图级teacher表征蒸馏，固定r10核政策|2／10,000|70／4,800／102,900|70；140数组|0|原9 passed／1 failed；返修指定项1 passed|已完成、不晋级；原交接记录返修后独立审查通过|
|r16|G05上同容量34368参数；有限Trimer三键端点常数距离对真实距离|0|70／4,800／102,900|70；140数组|0|11 passed|待审查；独立审查未进行|
|FP-GRAPH f1|G05同初态；常量FP注意力／真实FP均匀／真实FP注意力|0|105／7,200／154,350|105神经；不新做KRR后融合|0|10 passed|待审查；独立审查未进行|

参数周期背景另见[完整精度结果](results/fce_param_20261014/r2/final_review/seven_task_results.json)：140FT／9,600epochs／205,800updates，正式R0／B_LR各35单元；validation选出的B_LR在正式macro低于R0。该周期completion记载CPU启动2次，不能改称一次；本轮未重测。

实现要点：

- 原 F_CE：O8 与 Galformer 各6层／hidden512；原子到物理键token融合，norm2／norm3后concat1024，1024→512→1 head；共39,876,344 FT参数。旧四rank parent与新两rank parent的比较有随机流差异。
- T1增加dropout／WD并在每fold自己的validation选参，最终选中R0／D／W／DW为8／15／6／6。T2／r6的低LR只作用于Galformer，不是降低整个backbone。
- T3a两臂同原step0、global1008、两rank、各5000updates，只改schedule长度；T3b沿用T3_REF条件，只改geometry权重为0.5得到G05。不得把G05写成UA的预训练parent，UA来自四rank P_ALIGN。
- r7—r9是微调后的Morgan radius2／2048 binary、Tanimoto KRR；α∈{0.1,1,10}由本fold固定0.5融合validation RMSE选取，train-only scaler逆变换到原单位后融合。核单分支分数只作该选择政策下的诊断。
- r10固定r9全部模型与α，只在{0,0.25,0.5,0.75,1}中按35个validation R²等权平均选择一个全局神经权重，exact tie选较大值；选中0.75。七任务／五折相同，不按TEST回退或拼接。新增35组输出均为数组算术，0FT／0fit／0模型推理。公共实现为src/training/fce_kernel.py，经fce_param.py导出。
- r11在0.75融合下重新选α，8项改变、8新fit，其余27项复用；r12固定α和权重，只将radius改3。r13标准化目标使用2×Huber(delta1)，r14只开放encoder LayerNorm affine 26,624参数与head525,313参数，其余编码器冻结。均保留原神经validation RMSE选best。
- r15两臂从G05匹配续训各5000新updates，比较图级teacher目标；候选与CONT_REF均使用相同后续FT和固定r10核输出。r16同容量新模块34,368参数、零残差初态，只改变新增三键端点距离为常数3.0 Å或实际Trimer距离，不新增构象／PT。
- FP-GRAPH f1新增2,033,152参数，总41,909,496；self／geometry／FP三候选，V投影零初态，geometry无效时整候选含bias屏蔽。FP_NULL_KFUSE输入每位0.5，FP_REAL_MEAN严格均匀有效模态权重，FP_REAL_KFUSE可学习注意力，三臂均计算Q/K/V；原G05 parent及fold head一致。head阶段新模块和head共同训练，禁用旧head特征缓存；joint全参数。105正式结果均为神经预测，本周期没有新KRR融合输出。

T1／T2／T3a／T3b及后续F_CE神经周期沿用原任务配方：head10、joint一般60／Egc50，Xc保留其原LR／WD／dropout特例；U1／UA_2D是uniform_u1，七任务全部head10＋joint60、统一LR／WD／dropout。这些差异必须在UA迁移比较中保留说明，不能把跨parent结果差归因为单一结构因素。

### 七任务正式 R² mean±std

下列29条政策均有完整七任务汇总；R0在参数周期／T1／T2重复出现且保存数值相同，合并展示不代表复用了当时的训练预算。U1／UA_2D已在2026-10-14归档，此处仅作为新路线参考重列。

|周期／政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|宏 R²|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|U1／UA_2D（已有归档参考）|0.927±0.016|0.926±0.010|0.905±0.016|0.803±0.029|0.768±0.061|0.813±0.065|0.373±0.107|0.788|
|参数周期／R0（同 T1_R0、T2_R0）|0.927±0.018|0.940±0.014|0.911±0.017|0.790±0.031|0.766±0.048|0.820±0.054|0.305±0.144|0.780|
|参数周期／B_LR|0.921±0.012|0.929±0.012|0.911±0.008|0.798±0.031|0.764±0.044|0.846±0.044|0.234±0.140|0.772|
|T1／TUNED|0.927±0.018|0.940±0.003|0.912±0.017|0.794±0.046|0.771±0.047|0.830±0.059|0.270±0.183|0.778|
|T2／G_LR|0.918±0.020|0.937±0.007|0.911±0.012|0.793±0.036|0.764±0.046|0.825±0.058|0.354±0.106|0.786|
|T3a／REF|0.927±0.014|0.931±0.012|0.913±0.015|0.790±0.012|0.670±0.108|0.770±0.121|0.314±0.148|0.759|
|T3a／COS5K|0.922±0.015|0.927±0.015|0.907±0.014|0.794±0.020|0.716±0.066|0.773±0.067|0.307±0.152|0.764|
|T3b／G05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|r6／G05_GLR|0.925±0.012|0.928±0.011|0.909±0.012|0.779±0.047|0.759±0.037|0.834±0.054|0.359±0.114|0.785|
|r9／KRR（诊断）|0.864±0.007|0.860±0.026|0.884±0.014|0.741±0.028|0.683±0.087|0.762±0.052|0.392±0.067|0.741|
|r9／固定等权融合|0.916±0.012|0.912±0.013|0.914±0.012|0.801±0.019|0.768±0.026|0.817±0.044|0.430±0.088|0.794|
|r10／全局权重融合|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|
|r11／对齐 α|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.776±0.050|0.814±0.049|0.411±0.122|0.796|
|r12／radius3|0.923±0.017|0.922±0.013|0.916±0.012|0.808±0.017|0.775±0.056|0.814±0.053|0.408±0.126|0.795|
|r13／Huber 神经|0.920±0.010|0.933±0.005|0.914±0.011|0.779±0.024|0.727±0.052|0.815±0.055|0.316±0.063|0.772|
|r13／Huber 融合|0.922±0.012|0.931±0.005|0.918±0.012|0.791±0.018|0.745±0.049|0.825±0.044|0.378±0.059|0.787|
|r14／LayerNorm 神经|0.890±0.013|0.906±0.019|0.881±0.011|0.763±0.057|0.688±0.058|0.777±0.084|0.291±0.165|0.742|
|r14／LayerNorm 融合|0.902±0.014|0.915±0.014|0.897±0.011|0.781±0.049|0.716±0.050|0.794±0.066|0.346±0.123|0.764|
|r15／CONT_REF_neural|0.924±0.009|0.927±0.013|0.913±0.014|0.800±0.029|0.751±0.062|0.810±0.060|0.320±0.124|0.778|
|r15／CONT_REF_fused|0.926±0.008|0.927±0.009|0.917±0.013|0.810±0.018|0.764±0.054|0.821±0.049|0.368±0.099|0.790|
|r15／GRAPH_DISTILL_neural|0.923±0.012|0.934±0.006|0.909±0.011|0.787±0.008|0.697±0.066|0.816±0.067|0.371±0.067|0.777|
|r15／GRAPH_DISTILL_fused|0.923±0.011|0.931±0.005|0.915±0.009|0.797±0.007|0.723±0.063|0.827±0.045|0.410±0.064|0.790|
|r16／NB_CONST_neural|0.923±0.017|0.929±0.008|0.912±0.012|0.796±0.023|0.669±0.101|0.797±0.072|0.351±0.130|0.768|
|r16／NB_CONST_fused|0.924±0.016|0.928±0.004|0.917±0.013|0.805±0.022|0.711±0.078|0.812±0.056|0.398±0.109|0.785|
|r16／NB_REAL_neural|0.925±0.012|0.933±0.004|0.912±0.014|0.783±0.023|0.715±0.061|0.819±0.053|0.273±0.170|0.766|
|r16／NB_REAL_fused|0.927±0.010|0.932±0.005|0.917±0.013|0.794±0.025|0.735±0.061|0.824±0.049|0.336±0.134|0.781|
|FP-GRAPH f1／FP_NULL_KFUSE|0.922±0.022|0.928±0.011|0.910±0.016|0.802±0.024|0.735±0.047|0.809±0.087|0.341±0.121|0.778|
|FP-GRAPH f1／FP_REAL_MEAN|0.920±0.019|0.917±0.011|0.893±0.011|0.782±0.048|0.750±0.063|0.806±0.072|0.366±0.098|0.776|
|FP-GRAPH f1／FP_REAL_KFUSE|0.882±0.025|0.899±0.011|0.883±0.014|0.735±0.029|0.683±0.090|0.788±0.068|0.302±0.093|0.739|

统计：每任务五折算术mean／std(ddof=0)，神经seed42+fold、每fold一次训练；先任务内汇总再七任务等权macro，不是pooled R²或每fold多seed。全部数值先从未舍入值计算，再保留3位小数；展示相同不代表原精度相同。

### 对应原单位 RMSE mean±std

RMSE只在同一任务内比较，差值为候选减参考时负值较好；不跨物理单位平均RMSE，不采用旧r10报告中的跨任务RMSE均值作为排名。

|周期／政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|U1／UA_2D（已有归档参考）|0.289±0.026|0.528±0.045|0.479±0.036|0.433±0.066|0.530±0.066|0.101±0.010|18.627±1.526|
|参数周期／R0（同 T1_R0、T2_R0）|0.288±0.031|0.473±0.059|0.463±0.040|0.448±0.073|0.534±0.049|0.099±0.007|19.575±2.020|
|参数周期／B_LR|0.303±0.021|0.515±0.054|0.466±0.017|0.439±0.071|0.537±0.046|0.092±0.007|20.623±2.235|
|T1／TUNED|0.288±0.029|0.477±0.032|0.461±0.040|0.442±0.085|0.528±0.040|0.096±0.009|19.994±2.274|
|T2／G_LR|0.306±0.031|0.485±0.027|0.464±0.030|0.442±0.064|0.537±0.040|0.097±0.009|18.929±1.601|
|T3a／REF|0.291±0.029|0.508±0.059|0.458±0.035|0.447±0.058|0.633±0.124|0.109±0.020|19.450±2.076|
|T3a／COS5K|0.299±0.028|0.520±0.054|0.474±0.032|0.442±0.059|0.589±0.082|0.111±0.013|19.538±2.067|
|T3b／G05|0.303±0.036|0.534±0.066|0.463±0.032|0.441±0.061|0.541±0.110|0.105±0.008|18.596±2.012|
|r6／G05_GLR|0.294±0.021|0.517±0.028|0.470±0.027|0.456±0.059|0.544±0.042|0.095±0.010|18.817±1.305|
|r9／KRR（诊断）|0.398±0.022|0.727±0.105|0.532±0.028|0.497±0.072|0.619±0.066|0.115±0.010|18.375±0.649|
|r9／固定等权融合|0.310±0.014|0.577±0.075|0.458±0.029|0.435±0.059|0.537±0.051|0.101±0.007|17.784±1.097|
|r10／全局权重融合|0.294±0.025|0.539±0.068|0.450±0.031|0.429±0.058|0.527±0.078|0.100±0.006|18.017±1.559|
|r11／对齐 α|0.294±0.025|0.539±0.068|0.450±0.031|0.429±0.057|0.525±0.080|0.101±0.007|18.030±1.598|
|r12／radius3|0.296±0.024|0.542±0.071|0.452±0.029|0.428±0.058|0.525±0.087|0.101±0.008|18.070±1.651|
|r13／Huber 神经|0.305±0.015|0.502±0.036|0.457±0.026|0.457±0.047|0.578±0.059|0.101±0.010|19.527±0.913|
|r13／Huber 融合|0.299±0.016|0.510±0.049|0.446±0.029|0.445±0.049|0.557±0.055|0.098±0.008|18.611±0.858|
|r14／LayerNorm 神经|0.359±0.031|0.592±0.083|0.537±0.019|0.471±0.073|0.619±0.072|0.110±0.011|19.761±2.470|
|r14／LayerNorm 融合|0.337±0.026|0.567±0.076|0.500±0.023|0.454±0.071|0.592±0.065|0.106±0.009|19.028±1.897|
|r15／CONT_REF_neural|0.297±0.025|0.520±0.052|0.460±0.032|0.434±0.052|0.549±0.072|0.101±0.013|19.405±1.842|
|r15／CONT_REF_fused|0.294±0.019|0.525±0.053|0.449±0.030|0.425±0.053|0.536±0.064|0.099±0.009|18.730±1.527|
|r15／GRAPH_DISTILL_neural|0.299±0.019|0.499±0.028|0.469±0.027|0.450±0.055|0.607±0.072|0.100±0.015|18.703±0.914|
|r15／GRAPH_DISTILL_fused|0.299±0.017|0.509±0.044|0.455±0.022|0.440±0.054|0.580±0.065|0.097±0.008|18.113±0.830|
|r16／NB_CONST_neural|0.297±0.025|0.516±0.042|0.463±0.028|0.441±0.067|0.629±0.082|0.105±0.011|18.949±2.033|
|r16／NB_CONST_fused|0.294±0.022|0.522±0.045|0.449±0.030|0.431±0.065|0.591±0.071|0.102±0.008|18.255±1.735|
|r16／NB_REAL_neural|0.294±0.025|0.501±0.038|0.461±0.033|0.457±0.079|0.589±0.048|0.100±0.009|19.985±2.199|
|r16／NB_REAL_fused|0.291±0.017|0.509±0.048|0.448±0.032|0.444±0.076|0.567±0.049|0.098±0.009|19.134±1.803|
|FP-GRAPH f1／FP_NULL_KFUSE|0.298±0.034|0.519±0.030|0.465±0.037|0.431±0.034|0.571±0.065|0.101±0.015|19.115±1.918|
|FP-GRAPH f1／FP_REAL_MEAN|0.303±0.027|0.557±0.055|0.510±0.022|0.451±0.055|0.550±0.056|0.102±0.011|18.727±1.124|
|FP-GRAPH f1／FP_REAL_KFUSE|0.367±0.026|0.615±0.041|0.532±0.027|0.500±0.055|0.620±0.092|0.107±0.008|19.683±1.251|

### 论文七任务参考与逐项差值

参考来自当前SOTA_RESULT.md的每任务最佳R²均值及对应模型；不是单个论文模型的七任务联合表现。

|任务|论文最佳模型|论文 R²|
|---|---|---:|
|Eea|Periodic-TDL|0.934|
|Egb|Periodic-TDL (+Chem, +RC)|0.923|
|Egc|Periodic-TDL (+Chem, +RC)|0.911|
|Ei|Periodic-TDL|0.839|
|EPS|Periodic-TDL|0.787|
|Nc|Periodic-TDL|0.856|
|Xc|Periodic-TDL|0.423|

下表为本项目R²均值减对应论文参考，原精度计算后格式化，正值表示本项目数值较高；负零仍表示原精度差为负。

|周期／政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|U1／UA_2D（已有归档参考）|-0.007|+0.003|-0.006|-0.036|-0.019|-0.043|-0.050|
|参数周期／R0（同 T1_R0、T2_R0）|-0.007|+0.017|+0.000|-0.049|-0.021|-0.036|-0.118|
|参数周期／B_LR|-0.013|+0.006|-0.000|-0.041|-0.023|-0.010|-0.189|
|T1／TUNED|-0.007|+0.017|+0.001|-0.045|-0.016|-0.026|-0.153|
|T2／G_LR|-0.016|+0.014|+0.000|-0.046|-0.023|-0.031|-0.069|
|T3a／REF|-0.007|+0.008|+0.002|-0.049|-0.117|-0.086|-0.109|
|T3a／COS5K|-0.012|+0.004|-0.004|-0.045|-0.071|-0.083|-0.116|
|T3b／G05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|r6／G05_GLR|-0.009|+0.005|-0.002|-0.060|-0.028|-0.022|-0.064|
|r9／KRR（诊断）|-0.070|-0.063|-0.027|-0.098|-0.104|-0.094|-0.031|
|r9／固定等权融合|-0.018|-0.011|+0.003|-0.038|-0.019|-0.039|+0.007|
|r10／全局权重融合|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|
|r11／对齐 α|-0.010|-0.000|+0.006|-0.032|-0.011|-0.042|-0.012|
|r12／radius3|-0.011|-0.001|+0.005|-0.031|-0.012|-0.042|-0.015|
|r13／Huber 神经|-0.014|+0.010|+0.003|-0.060|-0.060|-0.041|-0.107|
|r13／Huber 融合|-0.012|+0.008|+0.007|-0.048|-0.042|-0.031|-0.045|
|r14／LayerNorm 神经|-0.044|-0.017|-0.030|-0.076|-0.099|-0.079|-0.132|
|r14／LayerNorm 融合|-0.032|-0.008|-0.014|-0.058|-0.071|-0.062|-0.077|
|r15／CONT_REF_neural|-0.010|+0.004|+0.002|-0.039|-0.036|-0.046|-0.103|
|r15／CONT_REF_fused|-0.008|+0.004|+0.006|-0.029|-0.023|-0.035|-0.055|
|r15／GRAPH_DISTILL_neural|-0.011|+0.011|-0.002|-0.052|-0.090|-0.040|-0.052|
|r15／GRAPH_DISTILL_fused|-0.011|+0.008|+0.004|-0.042|-0.064|-0.029|-0.013|
|r16／NB_CONST_neural|-0.011|+0.006|+0.001|-0.043|-0.118|-0.059|-0.072|
|r16／NB_CONST_fused|-0.010|+0.005|+0.006|-0.034|-0.076|-0.044|-0.025|
|r16／NB_REAL_neural|-0.009|+0.010|+0.001|-0.056|-0.072|-0.037|-0.150|
|r16／NB_REAL_fused|-0.007|+0.009|+0.006|-0.045|-0.052|-0.032|-0.087|
|FP-GRAPH f1／FP_NULL_KFUSE|-0.012|+0.005|-0.001|-0.037|-0.052|-0.047|-0.082|
|FP-GRAPH f1／FP_REAL_MEAN|-0.014|-0.006|-0.018|-0.057|-0.037|-0.050|-0.057|
|FP-GRAPH f1／FP_REAL_KFUSE|-0.052|-0.024|-0.028|-0.104|-0.104|-0.068|-0.121|

项目使用downstream_direct_01.staging与mips_paper5_outer，七任务样本数368／561／3380／370／382／382／432；历史记录的论文Egc／Xc为4124／430，论文原数据、精确split、预处理和评价协议等价性本轮未核实。共享validation／TEST已经参与开发，只作探索性数值对照，不称独立盲测、公平超过论文、统计显著或多seed稳定优势。EAT不在论文表中，Eib／Tg不在项目当前任务中，均不纳入七项比较。

### r7／r8 单任务正式结果与缺项

r7只运行Xc，r8只运行Ei；每task五折、std ddof=0、固定0.5G05＋0.5KRR。原单位RMSE及论文差如下，缺失任务不借用其他周期补成新macro。

|任务|r7 融合 R²／RMSE mean±std|r8 融合 R²／RMSE mean±std|论文最佳模型|论文 R²|r7−论文|r8−论文|
|---|---|---|---|---:|---|---|
|Eea|未完成／不可比较，未运行|未完成／不可比较，未运行|Periodic-TDL|0.934|不可比较|不可比较|
|Egb|未完成／不可比较，未运行|未完成／不可比较，未运行|Periodic-TDL (+Chem, +RC)|0.923|不可比较|不可比较|
|Egc|未完成／不可比较，未运行|未完成／不可比较，未运行|Periodic-TDL (+Chem, +RC)|0.911|不可比较|不可比较|
|Ei|未完成／不可比较，未运行|0.801±0.019／0.435±0.059|Periodic-TDL|0.839|不可比较|-0.038|
|EPS|未完成／不可比较，未运行|未完成／不可比较，未运行|Periodic-TDL|0.787|不可比较|不可比较|
|Nc|未完成／不可比较，未运行|未完成／不可比较，未运行|Periodic-TDL|0.856|不可比较|不可比较|
|Xc|0.430±0.088／17.784±1.097|未完成／不可比较，未运行|Periodic-TDL|0.423|+0.007|不可比较|

核分支诊断：r7的Xc R²0.392±0.067、RMSE18.375±0.649；r8的Ei R²0.741±0.028、RMSE0.497±0.072。r7融合−G05 R²+0.059、RMSE−0.812，4/5折改善；r8分别+0.005／−0.006，未达到其+0.010里程碑。r9后来完整覆盖七任务并复用这10个正式单元，只有r9可报告新的七任务macro。

### 结果判断与保留事项

- T1_TUNED较R0宏差−0.002，不晋级；T2_G_LR宏差+0.006，但主要来自Xc，其余六任务均差−0.001，不达到+0.010里程碑。T3a COS5K较REF宏差+0.005，不晋级；T3b G05较匹配REF为+0.023，但较历史四rank F_CE仍低，不能外推为所有parent的改进。r6 G05_GLR较G05约+0.002，Xc改善伴随其他退化，不按任务拼赢家。
- r9宏0.794较G05为+0.011；r10为0.796，较r9+0.002、较G05+0.014。r10的validation宏0.816不等于正式宏；论文七项最佳均值包络宏为0.810，也不是本项目实测。r10改善r9的5/7任务，Xc由0.430降至0.412，不能只报告有利任务。
- r10论文数值达标仅1/7（Egc）；Egb原精度低于0.923，差展示−0.000，不能按三位小数判为达标。r11和r10均显示0.796，但r11−r10为负、展示−0.000；r12较r11／r10均约−0.001。停止原α／w／radius配方的小网格，不追加测试后择优。
- r13神经／融合macro0.772／0.787，相对G05／r10为−0.011／−0.009；r14为0.742／0.764，相对为−0.040／−0.032。不晋级，不追加Huberδ或LayerNorm-only微网格。
- r15候选／参照融合均显示0.790，原精度候选−参照为−0.001，候选−r10为−0.007；去掉Xc后差−0.008，目标优化成功不等于下游整体改善。原交接明确返修后独立审查通过，按负结果收口。
- r16的真实／常数距离融合macro0.781／0.785，真实−常数−0.004、真实−r10−0.015，两项+0.010里程碑均未达到；独立审查未进行，不自动晋级。
- FP_GRAPH_F1主候选REAL_KFUSE宏0.739，较NULL−0.039、较MEAN−0.037、较r10−0.057，七任务均低于NULL；不按TEST改选MEAN或NULL作为主候选。该结果不支持直接把原三候选注意力照搬到UA，仍不能否定所有轻量FP残差方案。attention和融合输出差不构成交互效应证据。
- U1／UA_2D宏0.788，原模型19,262,233参数；已有纯二维构造／恢复代码，但旧reader仍解码含Trimer的存量记录。旧GLT架构r10的E0交付有2 failed，A1仅初始化PASS，PT／FT未完成；保留事故，不冒称完成，也不为归档重跑或伪造历史入口。
- T3a两个收尾helper失败与一条FT原metadata缺项仍保留。T3b已补交真实退出索引，缺失start／end保留null，没有伪造时长或重训科学单元。r8首次因把config.seed规则字符串误当整数在fit前失败；recovery1完成15fit后在冻结前因身份清单缺行失败；recovery2复用15fit完成正式预测，不重复相加额度。
- r15原CPU为9 passed／1 failed，原因是fixture传字符串而接口要求device.type；返修两处device输入后指定nodeid仅运行一次，1 passed／exit0，其他9项复用。并移除4项旧hash身份门槛，字段删除只作静态审查，没有重跑PT。保留原执行曾使用旧hash门槛的偏差；本轮没有新增或执行任何哈希验证。
- 所有周期未达成七任务全部论文数值目标；保留研究参照G05及r10统一预测政策。新UA路线先验证预测融合迁移，再评估匹配的轻量FP输入；这是新计划，当前没有新实验结果。

### 原始日志、产物与下一步

|周期|完整精度汇总／交付|日志根|
|---|---|---|
|T1|[正式结果](results/fce_tune_20261015/t1/final_review/seven_task_results.json)；`results/fce_tune_20261015/t1/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261015/t1/`|
|T2|[正式结果](results/fce_tune_20261015/t2/final_review/seven_task_results.json)；`results/fce_tune_20261015/t2/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261015/t2/`|
|T3a／r4|[正式结果](results/fce_tune_20261015/t3/final_review/seven_task_results.json)；`results/fce_tune_20261015/t3/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261015/t3/`|
|T3b／r5|[正式结果](results/fce_tune_20261015/t3b/final_review/seven_task_results.json)；`results/fce_tune_20261015/t3b/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261015/t3b/`|
|r6|[正式结果](results/fce_tune_20261016/g05_glr/final_review/seven_task_results.json)；`results/fce_tune_20261016/g05_glr/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/g05_glr/`|
|r7／Xc|[正式结果](results/fce_tune_20261016/xc_kernel/final_review/xc_results.json)；`results/fce_tune_20261016/xc_kernel/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/xc_kernel/`|
|r8／Ei|[正式结果](results/fce_tune_20261016/ei_kernel/recovery2/final_review/ei_results.json)；`results/fce_tune_20261016/ei_kernel/recovery2/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/ei_kernel/recovery2/`|
|r9|[正式结果](results/fce_tune_20261016/all_task_kernel/final_review/seven_task_results.json)；`results/fce_tune_20261016/all_task_kernel/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/all_task_kernel/`|
|r10|[正式结果](results/fce_tune_20261016/global_blend/final_review/seven_task_results.json)；`results/fce_tune_20261016/global_blend/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/global_blend/`|
|r11|[正式结果](results/fce_tune_20261016/aligned_alpha/final_review/seven_task_results.json)；`results/fce_tune_20261016/aligned_alpha/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/aligned_alpha/`|
|r12|[正式结果](results/fce_tune_20261016/radius3/final_review/seven_task_results.json)；`results/fce_tune_20261016/radius3/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/radius3/`|
|r13|[正式结果](results/fce_tune_20261016/huber_ft/final_review/seven_task_results.json)；`results/fce_tune_20261016/huber_ft/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/huber_ft/`|
|r14|[正式结果](results/fce_tune_20261016/norm_ft/final_review/seven_task_results.json)；`results/fce_tune_20261016/norm_ft/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/norm_ft/`|
|r15|[正式结果](results/fce_tune_20261016/graph_distill/final_review/seven_task_results.json)；`results/fce_tune_20261016/graph_distill/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261016/graph_distill/`|
|r16|[正式结果](results/fce_tune_20261017/nonbonded_ft/final_review/seven_task_results.json)；`results/fce_tune_20261017/nonbonded_ft/` 内保留冻结、completion、delivery及原数组|`logs/fce_tune_20261017/nonbonded_ft/`|
|FP-GRAPH f1|[正式结果](results/fp_graph_fusion_20261017/f1/final_review/seven_task_results.json)；`results/fp_graph_fusion_20261017/f1/` 内保留冻结、completion、delivery及原数组|`logs/fp_graph_fusion_20261017/f1/`|

参数周期日志根`logs/fce_param_20261014/r2/`；U1参考结果根`results/glt_optimization_20261011/r4/U1/`、日志根`logs/glt_optimization_20261011/r4/U1/`。r8的表中日志根是recovery2，首次失败及recovery1现场保留在同级目录。

长任务的既有tmux记录分别为`Uni-Poly:fce_t1_*`、`fce_t2_*`、`fce_t3_*`、`fce_t3b_*`、`fce_g05_glr_*`、`fce_xc_kernel_*`、`fce_ei_kernel_*`、`fce_all_kernel_*`、`fce_global_blend_*`、`fce_aligned_alpha_*`、`fce_radius3_*`、`fce_huber_*`、`fce_norm_*`、`fce_graph_distill_*`、`fce_nonbonded_*`、`fp_graph_f1_*`；这些是历史执行索引，本轮没有查询或恢复进程。

r15返修证据：[交付](results/fce_tune_20261016/graph_distill/repair_device_fixture/delivery.md)、[增量](results/fce_tune_20261016/graph_distill/repair_device_fixture/repair_only_changes.diff)、[指定测试日志](logs/fce_tune_20261016/graph_distill/repair_device_fixture/stdout.log)。其原completion仍保留需返修／test_exit1，不能据本节回写为原suite全通过。r16／FP f1各自唯一CPU日志为`logs/fce_tune_20261017/nonbonded_ft/final_cpu/`与`logs/fp_graph_fusion_20261017/f1/final_cpu/`，分别11 passed／10 passed、exit0；自检通过不替代尚未进行的独立审查。

本次只修改CURRENT_PLAN.md和本文件，新增PT／FT／fit／模型推理／正式融合评价／pytest／GPU均0，没有文件删除或旧产物改写。UA新周期和旧待审查事项由CURRENT_PLAN.md继续交接；未消费或已消费的旧额度都不因文档迁移重新生效。


## 2026-10-18｜UA2D-SOTA-20261018-01 / r2 F0：Morgan条件化融合开发阶段关闭

- **状态与审查**：已完成，仅关闭已授权F0开发和一次FP专项CPU测试。2026-10-18T16:07:52+08:00 Codex静态独立审查通过；完整SOTA周期未完成，原主线与F1、正式TEST继续暂停。基于本地dev / `3503fbe780a5399410d6db6055305925765c3ff8`及当时未提交工作树，没有远程同步。
- **scientific question / reference**：固定原UA各任务各折已训练O8及head，当前Morgan信息能否通过特征融合产生≥0.010的validation宏平均增量？原UA validation宏平均为0.812；NULL另控制head继续训练和全局适配收益。
- **实际架构 / controlled change**：复用35份norm2后raw h缓存；固定radius2/2048-bit Morgan。新增2048→32→1024适配器（99,360参数），输出`γ=0.1tanh(vγ)`、`β=0.1tanh(vβ)`。NULL用常量0.5输入及完整FiLM；SHIFT用真实FP的`h+β`；FILM用`(1+γ)⊙h+β`。输出层零初始化。仅训练adapter与同一核心的已训练head，O8/readout/norm2无更新。SHIFT的γ未参与预测，有效容量较低；FP信息控制以FILM−NULL为准。
- **comparison budget / 执行**：三臂各七任务五折、seed42+fold、50 epochs、batch24、FP32；AdamW3e-4/WD0.02，cosine50/min3e-6、clip5、head dropout0.1。逐fold按validation原单位RMSE选best，允许epoch0，不early stop。实际105 starts、5250 epochs、attempted/completed updates=120000/120000；未增加PT、KRR、FP、PH、构象、encoder forward或TEST。

### Validation结果及开发结论

| 臂 | 七任务五折validation宏平均R² | 相对原UA |
|---|---:|---:|
| NULL | 0.814 | +0.002 |
| SHIFT | 0.815 | +0.003 |
| FILM | 0.815 | +0.003 |

完整精度选SHIFT；FILM−NULL为+0.001，FILM−SHIFT为-0.000（小负差值舍入）。未达到+0.010开发晋级门槛，不采用为最终融合系统。NULL有19/35、SHIFT和FILM各18/35单元选epoch0。该结果未证明FP提供稳定收益，也不能否定其他未执行的FP融合机制。

### 七任务正式SOTA对照

下表前三列全部为validation均值±五折总体标准差(ddof=0)，仅记录实际开发观测；不能与论文正式数值混作性能对照。本批没有正式TEST，项目TEST R²与差值七项均不可比较。

| 任务 | NULL validation R² | SHIFT validation R² | FILM validation R² | 论文R² | 论文模型 | 本项目TEST R² | 项目−论文 |
|---|---:|---:|---:|---:|---|---|---|
| Eea | 0.924±0.016 | 0.924±0.016 | 0.924±0.015 | 0.934±0.013 | Periodic-TDL | 未完成／不可比较 | — |
| Egb | 0.942±0.013 | 0.941±0.012 | 0.941±0.012 | 0.923±0.009 | Periodic-TDL (+Chem, +RC) | 未完成／不可比较 | — |
| Egc | 0.908±0.005 | 0.911±0.007 | 0.911±0.007 | 0.911±0.008 | Periodic-TDL (+Chem, +RC) | 未完成／不可比较 | — |
| Ei | 0.848±0.023 | 0.848±0.023 | 0.848±0.023 | 0.839±0.018 | Periodic-TDL | 未完成／不可比较 | — |
| EPS | 0.771±0.082 | 0.772±0.081 | 0.772±0.081 | 0.787±0.033 | Periodic-TDL | 未完成／不可比较 | — |
| Nc | 0.852±0.050 | 0.853±0.051 | 0.853±0.051 | 0.856±0.032 | Periodic-TDL | 未完成／不可比较 | — |
| Xc | 0.455±0.098 | 0.455±0.098 | 0.454±0.098 | 0.423±0.102 | Periodic-TDL | 未完成／不可比较 | — |

统计为5fold等权、七任务等权，无额外seed；论文参考来自`SOTA_RESULT.md`，数据/split/预训练/seed统计未证明一致，历史TEST曾参与路线开发。EAT/Eib/Tg不纳入，不宣称公平超过论文。

### 独立审查证据、测试与限制

- 三份当前实现与交付`implementation.diff`一致；105单元配置、split/行身份、epoch记录、预算和exit已核对。读取保存的validation预测重算R²/RMSE，和记录最大绝对差0；没有重放模型forward或实验。
- 最终唯一CPU日志为10 passed、1 warning、4.58s、exit0；warning为依赖弃用提示。本轮不重跑测试，不重新strict-load105个best包；审查通过不等于部署恢复或正式TEST通过。
- [validation报告](results/ua2d_sota_20261018/r2/fp_fusion/final_review/validation_report.md)、[完整精度开发汇总](results/ua2d_sota_20261018/r2/fp_fusion/development/validation_summary.json)、[SOTA缺项表](results/ua2d_sota_20261018/r2/fp_fusion/final_review/seven_task_comparison.md)、[交付diff](results/ua2d_sota_20261018/r2/fp_fusion/final_review/implementation.diff)、[CPU日志](results/ua2d_sota_20261018/r2/fp_fusion/final_review/final_cpu.log)。结果根`results/ua2d_sota_20261018/r2/fp_fusion/`，日志根`logs/ua2d_sota_20261018/r2/fp_fusion/`；历史tmux窗口为`Uni-Poly:ua2d_sota_fp_r2`、`film_r2_<mode>_<task>_<fold>`、`film_r2_final_cpu`，本轮未查询/操作进程。
- 原execution_manifest的待审查状态保留为交付快照，独立审查状态以本归档及CURRENT_PLAN为准。不重跑F0，不自动启动F1/TEST或原P/A/T/M/I；下一步恢复须明确授权并保留实际累计预算。

## 2026-10-20｜UA-HOP-PH-20261020-01 / r1：四hop与空间PH路由负结果收口

**状态：已完成；Codex独立审查通过；性能门槛未通过，不晋级。** 用户2026-10-20明确要求Codex亲自记录审查并制定下一步计划，本节据此归档。审查结论已同时写入CURRENT_PLAN.md H16；不改变其他待审查周期状态。

- 审查主体：Codex；上轮独立审查为2026-10-20 17:07—17:11（Asia/Shanghai）；本轮17:30起按用户要求亲自落盘，并复核交付汇总未变化。依据本地 dev / `3503fbe780a5399410d6db6055305925765c3ff8` 加已有 dirty 工作区；不是对干净提交的验收。用户本轮授权审查记录、归档及下一步计划，不授权新训练。
- 科学问题：相同 UA、四尺度适配器和训练条件下，空间 PH 条件路由是否优于固定等权。reference 为 HOP_EQ，controlled change 为 PH router及其额外20484参数；与纯二维 V1＋FP 的差值另含推理期三维输入和适配器，不能单独归因 PH。
- 实际实现：保留 B_FP step5000 的 O8、MaskedGalformer3D、旧 attention 1/√512 与原 mean/norm2/head；末端四套独立LN/QKV构成物理键线图 exact 1／2／3／4-hop；全部最短路径先归约后进入共享18→32→4 bias MLP；等权或155→128→4空间PH dense路由后，经中央真实incidence均值、零初始化W_out和投影后完整屏蔽注入二维原子表示。合法无几何采用零增量，不新增构象。
- 审查范围：已静态检查 `src/dataset/ua_hop_ph.py`、`src/modules/ua_hop_ph.py`、`src/training/ua_hop_ph.py`、`scripts/run_ua_hop_ph.py`、`tests/test_ua_hop_ph.py` 及交付diff；旧 `glt_dual.py`／`glt_dual_pretrain.py` 改动此前已存在，本次未归为r1新修改。核对70个history／best阶段、冻结记录、配置／样本键／split索引、140份既有validation／TEST预测；保存预测重算R²与记录最大差为0，没有新增模型预测。抽查Egc/fold0 joint与Nc/fold4 head的完整checkpoint参数；未重放70个模型forward。
- 实际消耗：0 PT；70 FT；4900 started/completed epochs及validations；attempted/completed updates均112000；70正式TEST、11750图；prepare／FT／TEST／CPU共142个作业exit均0。GPU2／3每卡最高两路；冻结时间16:55:53早于首个TEST启动；controller最终exit0，结束16:59:15。实际controller窗口为 `Uni-Poly/@1200`，交付文本的 `@0` 为元数据笔误，不代表额外启动。
- 最终唯一CPU证据：`logs/ua_hop_ph_20261020/r1/cpu/stdout.log`，8 passed、1个依赖弃用warning、8.72秒、exit0；作业窗口@1342，包含启动的墙钟约15秒，未超300秒。本轮不重跑suite或实验。
- 性能：HOP_EQ validation／TEST宏R²为0.819／0.788；HOP_PH为0.819／0.788。PH−EQ的未舍入validation和TEST差值均为正且小于0.001，展示均为+0.000；validation支持条件满足，TEST增量未达≥0.010，预设总门槛不通过。
- 历史参照：HOP_EQ相对V1＋FP／r10的TEST差值为−0.003／−0.008；HOP_PH为−0.003／−0.008。历史来源分别为 `results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json`、`results/fce_tune_20261016/global_blend/final_review/seven_task_results.json`；本次不改变这些周期的既有审查状态。
- 两臂各有4／35折选head-only best；这些折W_out仍为零，只能按二维预测解释。输入准备为3654个唯一身份（3646有效、8合法无几何），5875个任务记录；中央incidence根键四跳周期覆盖95.022%，该比例不是分子覆盖率，也不能单独解释性能差。
- 统计：各任务5fold、seed42—46、std ddof=1，再七任务等权macro；原始预测／指标保持完整精度，展示三位小数。共享validation／TEST已经参与路线开发，非独立盲测；与论文数据、split及统计等价性未证明，仅作数值对照。Eea显示0.934但两臂未舍入值均低于参考；两臂均只有Egb、Egc两项数值达到参考，SOTA目标未完成。EAT／Eib／Tg不纳入。

| 任务 | 论文最佳模型 | 参考 R² | HOP_EQ R²±std | EQ−参考 | HOP_PH R²±std | PH−参考 |
|---|---|---:|---:|---:|---:|---:|
| Eea | Periodic-TDL | 0.934 | 0.934±0.015 | -0.000 | 0.934±0.015 | -0.000 |
| Egb | Periodic-TDL (+Chem, +RC) | 0.923 | 0.932±0.014 | +0.009 | 0.933±0.014 | +0.010 |
| Egc | Periodic-TDL (+Chem, +RC) | 0.911 | 0.912±0.018 | +0.001 | 0.914±0.015 | +0.003 |
| Ei | Periodic-TDL | 0.839 | 0.807±0.036 | -0.032 | 0.807±0.036 | -0.032 |
| EPS | Periodic-TDL | 0.787 | 0.755±0.045 | -0.032 | 0.755±0.045 | -0.032 |
| Nc | Periodic-TDL | 0.856 | 0.808±0.079 | -0.048 | 0.804±0.069 | -0.052 |
| Xc | Periodic-TDL | 0.423 | 0.369±0.152 | -0.054 | 0.372±0.151 | -0.051 |

- 原始结果与产物：`results/ua_hop_ph_20261020/r1/final_review/results.json`、`best_stage_summary.json`、`code_diff.patch`、`plan_diff.patch`；每臂每任务五折的 `finetune/`、`formal/`、`selection/final_frozen.json`、`sidecar/coverage_summary.json`、`initialization/`、`execution_manifest.json` 和 `budget.json` 全部保留。
- 日志：`logs/ua_hop_ph_20261020/r1/`，含controller／142个prepare、FT、TEST及CPU作业记录；实际controller为Uni-Poly/@1200，CPU为@1342。本轮不重启或干预这些历史进程。
- 收口：保留r1为已执行负结果，不替换V1＋FP或r10，不追加seed、训练或测试，不修改原始产物／旧审查快照。后续仅提出三维编码器冻结的单因素计划，状态待授权，唯一当前交接以CURRENT_PLAN.md F1—F11为准。
- 本轮只修改CURRENT_PLAN.md和本归档，新增代码、PT、FT、模型预测、CPU suite和GPU消耗均0，没有删除文件、哈希／校验和或远程同步。

## 2026-10-20｜UA-HOP-PH-FROZEN3D-20261020-01 / r1：冻结三维编码器开发负结果收口

**状态：已完成；Codex独立审查通过；开发门槛未通过，不晋级；正式TEST=0。** 当前用户要求Codex审查并制定下一步计划，本节归档实际实现与开发结果；CURRENT_PLAN.md F12为对应审查记录。

- 审查主体：Codex；检查时间2026-10-20 19:01起（Asia/Shanghai）。依据本地dev／`3503fbe780a5399410d6db6055305925765c3ff8`加已有dirty工作区，以及FROZEN3D的实际源码、交付diff和产物；用户授权本轮审查落盘、归档、规划与提示词，未授权新训练。
- 科学问题／reference：在相同初始化、数据与10＋60 epochs下，只冻结B_FP的encoder3d是否优于r1 HOP_PH全量joint更新。候选F3D_PH，原r1 HOP_PH的35套正式FT validation记录只读复用；原reference未重训或重推理。
- 实际实现与检查：静态查看 `src/modules/ua_hop_ph_frozen3d.py`、`src/training/ua_hop_ph_frozen3d.py`、`scripts/run_ua_hop_ph_frozen3d.py`、`tests/test_ua_hop_ph_frozen3d.py`。encoder3d全阶段requires_grad=false／eval，不入optimizer，真实3D前向保留；O8／norm2与新适配器／router／head在joint按原LR训练。共享输入／初态只读来自r1，新的OUT隔离，没有输入准备或embedding缓存。
- 执行证据：35 FT全部完成；started/completed epochs及validations均2450；attempted/completed updates均56000；TEST starts／graphs均0；CPU suite1次。35 FT＋1 CPU共36个作业exit均0，GPU2／3峰值各两路。FT从17:50:34至18:38:54，controller为Uni-Poly/@1343，18:39:05真实exit0；当前controller窗口为bash，无继续派发证据。
- 选择与数据核对：35个history的完整70 epochs、严格validation best、配置／阶段／scaler及保存预测一致；4714条既有validation预测的fold索引和sample key与manifest一致，保存预测重算R²与记录最大差0。无正式预测文件、无final_frozen.json，符合未通过门槛时不开放TEST的规定。
- checkpoint抽查：Egc/fold0 best为joint epoch59，encoder3d的103个状态张量与B_FP parent逐项torch.equal且状态有限；这支持冻结实现，不代替对全部35套参数的重新恢复。未执行模型forward或重跑训练／预测。
- 唯一CPU日志：`logs/ua_hop_ph_20261020/frozen3d_r1/cpu/stdout.log`，5 passed、1个依赖弃用warning、6.86秒、exit0；作业窗口@1379，包含启动墙钟约10秒，未超300秒。本轮没有重跑测试或扩大预算。
- 结果：F3D_PH validation宏R² 0.821，r1 HOP_PH 0.819，差值+0.002；完整精度差值小于0.010，开发门槛未通过。Ei／EPS／Nc三项validation提高，Eea／Egb／Egc／Xc下降；EPS单任务+0.011不构成改变全局门槛或按任务拼接的依据。head阶段best为3／35，不作三维收益解释。
- 统计与正式结果限制：每任务5fold、seed42—46、std ddof=1，再七任务等权macro，完整精度计算后三位小数展示；共享validation已经用于开发。新候选没有TEST，七任务正式R²及项目减论文差值全部不可比较，不填入r1、V1＋FP或validation数值代替。

| 任务 | F3D_PH validation R²±std | r1 HOP_PH validation R²±std | 候选−参考 |
|---|---:|---:|---:|
| Eea | 0.925±0.017 | 0.926±0.016 | -0.002 |
| Egb | 0.941±0.011 | 0.942±0.009 | -0.001 |
| Egc | 0.919±0.008 | 0.921±0.006 | -0.002 |
| Ei | 0.872±0.024 | 0.866±0.022 | +0.006 |
| EPS | 0.791±0.059 | 0.780±0.078 | +0.011 |
| Nc | 0.852±0.063 | 0.847±0.064 | +0.005 |
| Xc | 0.444±0.119 | 0.450±0.113 | -0.006 |

| 任务 | 论文最佳模型 | 参考R² | F3D_PH正式R²／项目−参考 |
|---|---|---:|---|
| Eea | Periodic-TDL | 0.934 | 未执行／不可比较（validation门槛未通过，TEST=0） |
| Egb | Periodic-TDL (+Chem, +RC) | 0.923 | 未执行／不可比较（validation门槛未通过，TEST=0） |
| Egc | Periodic-TDL (+Chem, +RC) | 0.911 | 未执行／不可比较（validation门槛未通过，TEST=0） |
| Ei | Periodic-TDL | 0.839 | 未执行／不可比较（validation门槛未通过，TEST=0） |
| EPS | Periodic-TDL | 0.787 | 未执行／不可比较（validation门槛未通过，TEST=0） |
| Nc | Periodic-TDL | 0.856 | 未执行／不可比较（validation门槛未通过，TEST=0） |
| Xc | Periodic-TDL | 0.423 | 未执行／不可比较（validation门槛未通过，TEST=0） |

- 原始产物：`results/ua_hop_ph_20261020/frozen3d_r1/`，包括35份FT完整best／resume／scaler／history／validation预测、sources、execution_manifest、budget、selection/development_decision.json及final_review结果／CPU／diff。日志根 `logs/ua_hop_ph_20261020/frozen3d_r1/`，controller @1343，CPU @1379，真实退出码保留。
- 没有正式TEST、没有final_frozen，不伪造候选七任务正式结果。原交付metadata中的PENDING仅代表DSH交付时点，独立审查状态以本归档及CURRENT_PLAN F12为准。保留全部原始产物，不追加训练或测试，不以单任务EPS改善重写宏门槛。
- 下一步为固定每折已微调B_FP二维预测、比较REAL／NULL几何残差的新方案，见CURRENT_PLAN A1—A11，状态待授权；不恢复旧r1／FROZEN3D或其他历史队列。本轮仅修改两份文档，新实现／PT／FT／模型预测／CPU／GPU消耗均0，无删除、哈希或远程操作。

## 2026-10-24｜近几日实验集中独立审查与归档

### 审查范围、版本与完成层级

- 审查 ID：UA2D-RECENT-REVIEW-20261024-01／r1；主体 Codex。用户本轮明确要求审查近几天所有实验并写入本文件；授权范围为审查与文档归档，没有新增实现、训练或评估授权。
- 检查时间：2026-10-24 10:36 起（Asia/Shanghai）。基于本地 dev／3503fbe780a5399410d6db6055305925765c3ff8 及已有 dirty 工作区。开始时 AGENTS.md、CURRENT_PLAN.md、PROJECT_HISTORY.md、共享模型已有修改，另有大量未跟踪实验源码与既有删除；全部保留。已查看相关 diff、源码、计划、SOTA_RESULT.md、实际交付和预算，不把 HEAD 当作全部运行源码。
- 范围：10 月 18—23 日实际执行的 UA2D 工作，重点为 10 月 19—23 日的 16 个交付周期，另补记已取消的旧 SOTA 主线；共 17 个周期／工作包。HOP_PH、FROZEN3D 及 10 月 18 日 F0 已有独立归档，复用其记录并纳入索引，不重复计费或伪造新增周期。10 月 17 日以前的 FCE／r9／r10 等仍见前文既有归档。
- 16 个交付周期合计：PT 启动 26 次（含历史失败／停止尝试），FT 启动 1,443 次，其中完整 FT 1,435 次、OPT 历史失败 8 次；正式评估 1,295 单元，包含 36 个模型输出臂的 1,260 次评估和 1 个数组融合政策的 35 次评估；已有 CPU suite 共 20 次。旧 SOTA 主线及其已归档 F0 的消费另列，不混入上述合计。
- 本轮只读核对 1,435 份完整 70-epoch history；OPT 按成功 attempt 映射取 280 个单元，未把失败目录当作成功记录。核对正式预测的配置、fold／seed、样本键、原始行号和目标，重算 1,260 组既有模型预测的 R²，结果与保存值差异为 0；各臂的五折均值、样本标准差和七任务 macro 一致。另按原始行号与 sample key 联合对齐，核对 35 组数组融合预测及 175 个既有 validation 权重分数，融合预测最大差异为 0。
- 已核对相关实际实现、交付 diff／配置、预算与作业真实退出证据；原有最终 CPU 测试结果见下表，没有重跑。既有 FPDECAY、HOP_PH、FROZEN3D 的详细参数／数据审查记录一并复用。审查没有重新恢复全部 checkpoint 或执行数值 forward，不声称逐张量全面认证。
- 下文“已完成、审查通过”指限定执行范围及其结果证据通过独立审查；性能结论另行判定。原始 JSON 的 PENDING、旧计划的待执行、事故日志和未知退出码保留为历史快照，当前结论以本节及 CURRENT_PLAN.md 的同 ID 审查记录为准。L、额外 30K／40K PT 等未执行方案不列为实验结果。

### 各周期的科学问题、实际实现与判定

**SOTA 主线／r1、r1_retest1（10 月 18—19 日）：已取消，完整实验未完成。** 该旧路线包含 T_RAWCE、T_AL05、M_ALL4 等旧预训练与下游开发分支，不等同于后来的 V1／B_FP。retest1 的两支 PT runtime 各记录完成 5,000 steps，M_ALL4 被用户停止；账本累计 PT 启动 3、attempted updates 11,845、completed updates 11,844，FT 启动 284、begun epochs 19,786／completed epochs 19,782，KRR 拟合 315、特征单元 35，正式神经／核 TEST 与最终 CPU 均为 0。更早 r1 单独记录 FT 启动 53、epochs 3,577、KRR 拟合 315、特征单元 36，其中一次 FP 旧表列不匹配失败；不将这些消费重置或称为 retest 新增。2026-10-19 11:27 的用户停止记录明确禁止自动继续。该部分仅完成停止／消费证据审查，未逐单元认证其开发产物，不作为完整实验或性能通过归档。原 r2 F0 的 105 次适配与 CPU 结果已经在本文件 2026-10-18 节归档，TEST=0，保持原结论。

**FP／PRETRAIN-FP fast_r4（10 月 19 日）：已完成，执行与结果审查通过；新增 FP 未达性能门槛。** 科学问题是在纯二维下游路径中，比较 masked atom／3D bond CE＋图级 ALIGN 的 V1、几何重建任务包 V2，以及 V1 加真实 Morgan FP 监督的 B_FP。三支成功 PT 均从公共 step0 完整训练 5,000 updates，各 35 FT／35 TEST；数据、split、公共下游 head 与 10＋60 epochs 配方相同。B_FP 使用 masked 2D states 的 mean＋norm2 预测完整 2048-bit Morgan，balanced BCE；历史 ALIGN／FP 权重各 0.1 并 warmup 1,000，双向 ALIGN 使用 sum 口径。下游不输入 FP 或 3D。TEST：V1 0.787，V2 0.785，B_FP 0.791；B_FP−V1 +0.004，V2−V1 −0.002。B_FP 可保留为后续已使用的数值参考，不能写成达到 +0.010 的改进。最早 collate NameError、一次失败 CPU、用户停止的 V1_attempt2 均保留并进入累计账本；后者日志完成约 1,390 steps，计费采用预算记录，不拿每个 rank 的相同步数重复相加。

**CLEAN／CLEANFP-READOUT-DISTILL（10 月 19—20 日）：已完成，审查通过；固定候选未晋级，存在限定的内部正向对照。** 六臂为 G／Q／QL × ALL1／DYN，每臂 5,000 PT＋35 FT＋35 TEST。实际实现同时进行 masked 与 clean 的 2D／3D forward：masked 路径负责恢复 CE，clean 2D 路径预测完整 FP，clean 2D／3D 图表示做 ALIGN；因此它同时改变 FP 输入和 ALIGN 输入，不能单独归因为“clean FP”。G 为 mean readout；Q 为零门控残差的 query readout；QL 进一步用 clean atom 与 stop-gradient incident 3D bond teacher 对齐。ALL1 的各任务系数为 1，但保留旧 ALIGN×2 口径，不能与后来 uniform1_align_mean_v1 混称。DYN 按训练损失历史每 20 窗口调整，权重限制 0.25—4，保持总和并平滑，不读取 TEST。固定候选 G_ALL1 TEST 0.778，低于历史 B_FP；原 validation 晋级条件也未通过。QL_DYN TEST 0.790，较 QL_ALL1 0.774 提高 +0.016，按当前默认 TEST 数值门槛，该匹配内部权重对照达到门槛；但它仍低于 B_FP，且不改变原冻结候选或原周期晋级结论。不能写成“六臂所有对照都没有收益”，也不能按 TEST 将 QL_DYN 追认为最终候选。

**ATTN／V1_ATTN64（10 月 20 日）：已完成，审查通过；性能未通过。** 将 O8／3D attention 的 QK 缩放由 1/√512 改为 1/√64，结构偏置在缩放之后加入、保持不缩放。新增 1 支 5,000-step PT＋35 FT／35 TEST；历史 V1 为 reference。TEST 0.782，相对 V1 −0.005、B_FP −0.009。该结果没有支持调整缩放即可改善迁移；与历史 PT 执行条件的差异限制因果解释，不据此宣称缩放对任何设置均无效。

**DEMOL（10 月 20 日）：已完成限定开发范围，审查通过；正式性能未完成／不可比较。** 从同一 B_FP 恢复 O8／norm2 与公共 head，PT=0；对比 BOND_SELF／BOND_NEIGHBOR，各 35 FT。原子端点和 bond chemistry 构造 128 维键适配器，经过自身或 1-hop 键邻域聚合后，通过零初始化投影回注入原子；两臂适配器结构匹配。head 10 epochs 冻结适配器，joint 60 epochs 使用 O8 1e−4、adapter 3e−4、head 1e−3。validation macro：SELF 0.820、NEIGHBOR 0.819，NEIGHBOR−SELF −0.001，原开发触发条件未满足，正式 TEST=0。不能将 validation 差值当成 TEST 负结果。

**ET／ET_SINGLE repair1（10 月 20—21 日）：修复后的限定开发范围已完成，审查通过；正式性能未完成／不可比较。** 用 TorchMD 风格 6×256 equivariant atom encoder 替代 3D bondline teacher，以半径 4 Å、真实重原子键及 self 的并集构图；保留 2D CE、3D bond-type CE 与图级 ALIGN，下游仍纯 2D。原首次 PT 因 atomic/template physical count mismatch 失败，真实退出 137；repair1 经授权排除 635 条 geometry_invalid 的整条记录，911,391→910,756，不让它们只参与二维目标。修复 PT 完成 5,000 updates、35 FT 完成；validation macro 0.815，相对 V1 −0.002，原触发条件未满足，TEST=0。早先“修复仅到 51 steps”的进度快照已过时。两次 prepare／PT、两次既有 CPU 与失败消费保留；不同 teacher 与筛选后的 cohort 同时变化，不能归因于单独的 ET 算子。

**HOP／UA-HOP-PH r1 与 F3D／FROZEN3D（10 月 20 日）：沿用前文审查，通过证据归档，均未形成晋级结果。** HOP 在同一 B_FP parent 上比较四 hop 等权与基于空间 PH 的路由，共 70 FT／70 TEST，HOP_EQ 0.788、HOP_PH 0.788，差值展示 +0.000；不能称为 0.010 改进。FROZEN3D 另做 35 FT、TEST=0，实际保留 3D forward 而仅冻结 encoder3d；validation 差值 +0.002，正式性能未完成／不可比较。两周期的实现、参数冻结抽查、GPU 峰值与论文对照已见本文件对应 2026-10-20 独立归档，不再重复记为新增实验。

**OPT／r1-FT-REPAIR1（10 月 21—22 日）：修复后已完成，审查通过；性能未通过。** A 比较末两层迁移和 2／4／6 层读出；B 比较固定 2／3-hop 与元素 E／化学上下文 C 目标；C 为六层 hop schedule [1,1,2,2,3,3]；D 将所选 B2_C 与 MIX246 组合。实际 MIX246 为 p6＋0.25 tanh(a2)(p2−p6)＋0.25 tanh(a4)(p4−p6)，a2／a4 零初始化，属于有界残差混合，绝非 softmax 权重混合。用户后续授权 C、D 无论 B 结果均执行，不能再按早期 FT_POLICY=NONE 宣称 D 未跑。实际 8 臂各 35 FT 完成；按最后授权政策，一次冻结 A_MIX246、B3_E、B2_C、C_SCHED_E、D_INTEGRATE 的 175 个正式单元。FINAL=A_MIX246 TEST 0.791，较 B_FP +0.000；C−B3_E −0.011，D−B2_C +0.000，均未达到门槛。A_TOP2、B2_E、B3_C 没有正式 TEST，因此不能报告完整 TEST 2×2 hop×target 主效应或交互效应。审查使用 repair_ft1 的结果和 attempt_mapping，而非旧失败 final_review 的空结果。

OPT 历史 B3_C 的 8 次失败各有真实退出：Egc folds 0／1 为 124，Egc folds 2／3、Ei fold4、EPS fold1 为 137，Egc fold4、EPS fold0 为 1。授权修复为新目录各一次完整 fresh FT，没有覆盖失败现场。最终 288 次 FT 启动包含 280 次成功和 8 次失败；失败单元按整单元保守预留替代已知下界计费，共 20,160 epochs／validations、482,090 attempted updates，不能误写成全部成功完成的训练量。B3_C／C 的 LMDB 预读与计时仅为工程变化，实际加载／耗时见原报告；没有根据静态检查预先断言提速。

**TRANSFER（10 月 22 日）：已完成，审查通过；性能未通过。** 同一 B_FP、同一 2-hop 输入及公共 head，比较 R_MEAN、R_STD、A_SP、R_STD_SP，各 35 FT／TEST，PT=0。R_MEAN／R_STD 使用匹配的零输出残差适配器，后者读 population std；A_SP 在共同 mean 路径施加系数 0.5e−3 的 L2-SP 锚定，组合臂合并两因素。R_MEAN 0.791、R_STD 0.788、A_SP 0.790、R_STD_SP 0.785；固定 FINAL=R_STD_SP，相对 B_FP −0.006。R_STD−R_MEAN 是匹配适配器容量的读出对照；其他跨配置差值不统一解释成“std 的效果”。

**WAVG（10 月 22 日）：已完成，审查通过；权重平均未达性能门槛。** 仅 35 条完整 70-epoch FT 轨迹，各导出 validation BEST、全局 epochs 40／50／60／70 的 SWA4，以及全局 epochs 11—70（joint 60 轮）的 β=0.9、S0=z0=0、最终 S/z 的 EMA60；浮点状态 FP32 累加，非浮点 buffer 必须一致。不是 105 次 FT；额外 70 次 endpoint validation 已计入 2,520 次 validation。一次冻结 105 个估计器后正式 TEST：BEST 0.791，SWA4 0.780（−0.011），EMA60 0.777（−0.014）。固定 FINAL=EMA60，不根据 TEST 改选 BEST 为新优化成果。

**PTSCALE（10 月 22 日）：已完成，审查通过；10K 是本周期冻结参考，扩大步数未达到提升门槛。** 同一条从 step0 连续到 20,000 的 PT 轨迹保存 5K／10K／20K；不是三次独立 PT，也不把 5K／10K checkpoint 当成另外训练消费。保持旧 masked FP、ALIGN／FP 0.1 warmup 配方，5K 学习率周期重复；因此延长训练同时增加数据曝光和学习率周期数。三组各 35 FT／35 TEST：P5K 0.788、P10K 0.793、P20K 0.785；P10K−P5K +0.005，P20K−P10K −0.008，P10K−B_FP +0.002。正式 reference 是 step10000/deploy.pt，不是 0.893，也不是 3DSHIFT 的 P_G。

**FPDECAY／r2-SG（10 月 22 日）：已完成，审查通过；严格未通过性能门槛。** 单物理 GPU3、world1、micro168×acc6、global1008；BASE 与 FP_DECAY 各独立 5K PT＋35 FT／35 TEST，使用同一采样、初始化与目标计数口径。FP_DECAY 在前 4,000 updates 保留原 FP，4,000—5,000 余弦退火到 0，其余保持匹配。BASE TEST 0.781，FP_DECAY 0.790，差值展示 +0.010，但原始未舍入值严格小于 0.010，性能判定为未通过；相对 P10K −0.003、B_FP −0.001。既有 CURRENT_PLAN 第 2 节对 PT 日志、全部微批采样、resume／deploy、预测及源码的详细核对继续有效，不重复跑测试。

**SHIFT／3DSHIFT r1.5-ALL1-PCTRL-GPU2（10 月 22—23 日）：新修订已完成，审查通过；P 未达到性能门槛。** 科学问题是周期 canonical bond shared state＋相对 image shift covering graph＋每层零初始化几何 value 的 P 表征包，相对有限 Trimer teacher 是否改善纯二维迁移。共同取消 ALIGN×2，所有激活目标权重为 1、无 loss warmup，policy=uniform1_align_mean_v1；FP 仍是 masked 2D→完整 Morgan，L 关闭。F_G10K 与 P_G10K 从公共 step0 各独立 10,000 updates；P_G5K 只读出同一 P_G 轨迹已保留的 step5K，增加 35 FT／TEST 而无新增 PT。最终总计 105 FT／105 TEST，逐臂冻结并保留 all_evaluations_frozen 记录，固定 FINAL=P_G。TEST F_G10K 0.782、P_G5K 0.776、P_G10K 0.783；P_G10K−F_G10K +0.001，P_G10K−PTSCALE/P10K −0.010。后者还存在 objective policy／资源等变化，只作数值参考，不是 P 的单因素对照。

SHIFT 早期 original_P_G 异常停止，真实退出与原因缺失，仍为未核实；recovery1 完成 629／尝试 630 后于 10 月 23 日 08:54 按用户要求取消。累计账本包含旧 2 次 PT 与新 2 次 PT，保守计费 30,630 updates，已知实际下界 26,009，不把保守预留称为全部完成。原 prepare 已完成 910,756 条 sidecar，后续只读复用。新两支 PT 及 105 FT／TEST、最终 CPU 的退出证据完整；旧未知退出不伪造为 0。历史 PT 使用当时明确授权的 GPU2，F_G 与 P_G5K 下游用 GPU3、P_G10K 用 GPU2，硬件差异保留。没有执行 L，不宣称完成 P×L 消融或交互效应。

**FPM／F_G10K＋Morgan concat／gate（10 月 23 日）：已完成，审查通过；性能下降。** 只复用上项 F_G10K step10000，PT=0；真实 binary Morgan 2048→128→128＋LN，concat 等价于旧 head 的 UA512 线性输出加零初始化 FP128→512 投影；gate 用 [UA,z]→64→128 的 2sigmoid，零末层使初始 gate=1。只新增 CONCAT_FP／GATE_FP 各 35 FT／TEST，head／joint 保持 10＋60 epochs；FP 提前从既有 SQLite 缓存读取，5,875 行、3,654 结构，无在线 FP 生成。TEST 0.757／0.758，对 F_G10K 分别 −0.025／−0.024。两支均为负结果；不能据此否定独立 KRR 的预测级融合。实际 GPU3 总并发最多 4 路，旧计划中的每组并发表述以其后执行授权为准。

**PH／P10K＋PH descriptor concat／gate（10 月 23 日）：已完成，审查通过；性能下降。** reference 明确为 PTSCALE/P10K TEST 0.793，PT=0。有限 Trimer 的 ALL-heavy／C／C+N／C+O 四组点云，F₂ VR filtration、截断 4.5 Å、H0／H1，输出 412 维 Betti／有限条形统计／右删失信息；它是分子描述符，不是无限周期 PH 或额外预训练通道。一次 prepare 3,654 结构（3,646 有效、8 无效），耗时 44.8 秒；无效样本保留，encoder 后置零其 PH 分支。每折仅有效 train 行拟合 log1p 后的特征 scaler。412→128→128＋LN、零投影 concat 或初始 gate=1 的通道 gate；共 70 FT／70 TEST，无新 PT。TEST CONCAT_PH 0.782、GATE_PH 0.781，分别较 reference −0.011／−0.012。Nc 的局部改善不改变宏门槛，不按任务拼接。实际执行 GPU2 总并发 4，是后续授权的交付配置；目录名 gpu01 与旧 PH_PLAN 的 GPU0／1 不代表实际用了两卡。既有保存 barcodes、feature scaler 与预测构成可追溯证据，不新增 PH 计算或填补几何。

**R10／P10K＋r10-style 全局预测融合（10 月 23 日）：已完成，独立审查通过；本轮固定主候选达到性能门槛。** reference 为 PTSCALE/P10K 的原始单位正式预测，FP 分支只读复用 r9 all_task_kernel 的已拟合 KRR 及既有预测，保持其原 train-only scaler、核、alpha 和 split；不是复用旧 r10 已混合的输出再混合。一次 prepare 对齐 35 validation 单元；在固定 w∈{0,0.25,0.5,0.75,1} 上比较 175 个 validation R²，唯一全局 w=0.75。一次冻结后仅评估政策 y=0.75·P10K＋0.25·KRR 的 35 个 TEST 单元，不比较其他 w 的 TEST，不按 task／fold 选权重、不重选 alpha。

R10 正式 macro R² 0.804，reference 0.793，配对差值展示 +0.010，原始未舍入值达到 0.010 门槛；由完整精度判定为性能通过，不能用两个已舍入 macro 相减代替原差值。EPS、Nc、Xc 的均值分别改善 +0.015、+0.010、+0.042；24／35 folds 为正。新 PT、FT、KRR 拟合、FP 生成、核查询、模型推理、GPU 和 worker 全为 0；35 次正式评估是保存数组的政策评估，不是 35 次重新推理。CPU 24 passed、exit0；全流程 5.516 秒。融合收益支持两类预测在本协议下互补，不能自动称为表征交互效应、独立盲测优势或超过全部论文任务。

### 实际预算、退出与已有 CPU 证据

| ID | PT 启动／计费 updates | FT 启动／完整单元 | 正式 TEST 单元 | CPU 原有证据 |
|---|---:|---:|---:|---|
| FP | 5／16,392 | 105／105 | 105 | 3 次：首次 7 passed／1 failed，exit 1；返修 10 passed；fast_r4 6 passed |
| CLEAN | 6／30,000 | 210／210 | 210 | 7 passed |
| ATTN | 1／5,000 | 35／35 | 35 | 8 passed |
| DEMOL | 0／0 | 70／70 | 0 | 5 passed |
| ET | 2／5,001 | 35／35 | 0 | 6 passed；repair1 5 passed |
| HOP | 0／0 | 70／70 | 70 | 8 passed（既有归档） |
| F3D | 0／0 | 35／35 | 0 | 5 passed（既有归档） |
| OPT | 5／25,000 | 288／280 | 175 | 23 passed；FT 修复后 28 passed |
| TRANSFER | 0／0 | 140／140 | 140 | 14 passed |
| WAVG | 0／0 | 35／35 | 105 | 8 passed |
| PTSCALE | 1／20,000 | 105／105 | 105 | 11 passed |
| FPDECAY | 2／10,000 | 70／70 | 70 | 9 passed |
| SHIFT | 4／30,630 | 105／105 | 105 | 26 passed |
| FPM | 0／0 | 70／70 | 70 | 9 passed |
| PH | 0／0 | 70／70 | 70 | 11 passed |
| R10 | 0／0 | 0／0 | 35 | 24 passed |

上述 CPU 的成功项均有原日志 exit0；FP 首次 CPU 的 exit1 保留，不由后续通过覆盖。预算 PT updates 是各累计账本口径，包含已说明的失败／预取／未知退出保守计费，不能直接当作净成功更新量。OPT 成功 FT 为 280 而非 288；WAVG 只有 35 FT；PTSCALE 只有 1 条 PT；SHIFT 的 P_G5K 不新增 PT；R10 不新增任何 FT。各周期日志和命令、cwd、tmux window、PID、真实退出见对应 execution_manifest／budget 的 job 记录和 logs 同名根。审查不根据旧 tmux 窗口是否存在推断仍在运行。

### 七任务正式 TEST、论文参考与逐项差值

统一口径：每任务 5 folds，seed=42+fold，等权 mean±std（ddof=1），再七任务等权 macro；原始预测与指标保留完整精度。下表每格为“项目 R²±std（项目−论文）”，均按未舍入值计算后展示 3 位小数；macro 没有另造标准差。R10 的统计由同一 35 单元重算，与此口径一致；历史 r9／r10 原文曾用 ddof=0，不将旧标准差直接作本轮配对不确定性。

| 任务 | 论文参考 R² | 对应模型 |
|---|---:|---|
| Eea | 0.934 | Periodic-TDL |
| Egb | 0.923 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.911 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.839 | Periodic-TDL |
| EPS | 0.787 | Periodic-TDL |
| Nc | 0.856 | Periodic-TDL |
| Xc | 0.423 | Periodic-TDL |

七项论文最佳均值的等权包络 macro 为 0.810，来自不同模型配置，不是一个可运行单模型。项目 Egc 3,380 对论文 4,124、Xc 432 对论文 430，且数据／split／评估协议等价性未获证明；以下只作数值对照。共享开发 TEST 已参与路线开发，不称独立盲测；R10 数值高于 Egb／Egc／Xc 三项参考，其他四项仍低于参考，不能宣布整体 SOTA。EAT、Eib、Tg 不纳入。

| 周期／臂 | Eea | Egb | Egc | Ei | EPS | Nc | Xc | macro R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP／V1 | 0.926±0.020（-0.008） | 0.930±0.015（+0.007） | 0.912±0.016（+0.001） | 0.808±0.044（-0.031） | 0.746±0.067（-0.041） | 0.808±0.067（-0.048） | 0.380±0.083（-0.043） | 0.787 |
| FP／V2 | 0.928±0.012（-0.006） | 0.934±0.011（+0.011） | 0.911±0.017（-0.000） | 0.816±0.021（-0.023） | 0.740±0.054（-0.047） | 0.788±0.085（-0.068） | 0.380±0.087（-0.043） | 0.785 |
| FP／B_FP | 0.933±0.014（-0.001） | 0.934±0.012（+0.011） | 0.913±0.017（+0.002） | 0.816±0.032（-0.023） | 0.764±0.058（-0.023） | 0.787±0.100（-0.069） | 0.393±0.146（-0.030） | 0.791 |
| CLEAN／G_ALL1 | 0.924±0.025（-0.010） | 0.929±0.017（+0.006） | 0.914±0.014（+0.003） | 0.804±0.027（-0.035） | 0.700±0.075（-0.087） | 0.763±0.105（-0.093） | 0.414±0.138（-0.009） | 0.778 |
| CLEAN／G_DYN | 0.918±0.022（-0.016） | 0.933±0.013（+0.010） | 0.917±0.016（+0.006） | 0.810±0.021（-0.029） | 0.743±0.059（-0.044） | 0.797±0.060（-0.059） | 0.301±0.119（-0.122） | 0.774 |
| CLEAN／Q_ALL1 | 0.918±0.032（-0.016） | 0.936±0.018（+0.013） | 0.915±0.016（+0.004） | 0.814±0.021（-0.025） | 0.727±0.086（-0.060） | 0.774±0.107（-0.082） | 0.395±0.185（-0.028） | 0.783 |
| CLEAN／Q_DYN | 0.918±0.034（-0.016） | 0.934±0.011（+0.011） | 0.918±0.018（+0.007） | 0.816±0.020（-0.023） | 0.713±0.099（-0.074） | 0.783±0.095（-0.073） | 0.398±0.132（-0.025） | 0.783 |
| CLEAN／QL_ALL1 | 0.923±0.018（-0.011） | 0.932±0.017（+0.009） | 0.913±0.016（+0.002） | 0.805±0.021（-0.034） | 0.697±0.077（-0.090） | 0.779±0.101（-0.077） | 0.368±0.139（-0.055） | 0.774 |
| CLEAN／QL_DYN | 0.921±0.022（-0.013） | 0.937±0.014（+0.014） | 0.917±0.018（+0.006） | 0.815±0.024（-0.024） | 0.761±0.038（-0.026） | 0.785±0.122（-0.071） | 0.394±0.126（-0.029） | 0.790 |
| ATTN／V1_ATTN64 | 0.922±0.014（-0.012） | 0.933±0.011（+0.010） | 0.910±0.016（-0.001） | 0.819±0.023（-0.020） | 0.713±0.048（-0.074） | 0.810±0.059（-0.046） | 0.367±0.084（-0.056） | 0.782 |
| HOP／HOP_EQ | 0.934±0.015（-0.000） | 0.932±0.014（+0.009） | 0.912±0.018（+0.001） | 0.807±0.036（-0.032） | 0.755±0.045（-0.032） | 0.808±0.079（-0.048） | 0.369±0.152（-0.054） | 0.788 |
| HOP／HOP_PH | 0.934±0.015（-0.000） | 0.933±0.014（+0.010） | 0.914±0.015（+0.003） | 0.807±0.036（-0.032） | 0.755±0.045（-0.032） | 0.804±0.069（-0.052） | 0.372±0.151（-0.051） | 0.788 |
| OPT／A_MIX246 | 0.933±0.014（-0.001） | 0.934±0.012（+0.011） | 0.913±0.015（+0.002） | 0.816±0.032（-0.023） | 0.764±0.058（-0.023） | 0.787±0.100（-0.069） | 0.393±0.146（-0.030） | 0.791 |
| OPT／C_SCHED_E | 0.933±0.017（-0.001） | 0.932±0.019（+0.009） | 0.912±0.014（+0.001） | 0.795±0.041（-0.044） | 0.732±0.067（-0.055） | 0.812±0.066（-0.044） | 0.336±0.117（-0.087） | 0.779 |
| OPT／D_INTEGRATE | 0.931±0.025（-0.003） | 0.936±0.015（+0.013） | 0.915±0.016（+0.004） | 0.794±0.042（-0.045） | 0.737±0.058（-0.050） | 0.795±0.087（-0.061） | 0.374±0.066（-0.049） | 0.783 |
| OPT／B3_E | 0.930±0.018（-0.004） | 0.918±0.023（-0.005） | 0.911±0.016（+0.000） | 0.812±0.027（-0.027） | 0.751±0.055（-0.036） | 0.820±0.074（-0.036） | 0.390±0.123（-0.033） | 0.790 |
| OPT／B2_C | 0.931±0.025（-0.003） | 0.936±0.015（+0.013） | 0.914±0.016（+0.003） | 0.794±0.042（-0.045） | 0.737±0.058（-0.050） | 0.795±0.087（-0.061） | 0.374±0.066（-0.049） | 0.783 |
| TRANSFER／R_MEAN | 0.934±0.012（+0.000） | 0.931±0.017（+0.008） | 0.910±0.017（-0.001） | 0.819±0.024（-0.020） | 0.739±0.058（-0.048） | 0.806±0.072（-0.050） | 0.401±0.135（-0.022） | 0.791 |
| TRANSFER／R_STD | 0.933±0.014（-0.001） | 0.932±0.011（+0.009） | 0.915±0.017（+0.004） | 0.807±0.036（-0.032） | 0.721±0.085（-0.066） | 0.792±0.067（-0.064） | 0.416±0.104（-0.007） | 0.788 |
| TRANSFER／A_SP | 0.932±0.011（-0.002） | 0.929±0.015（+0.006） | 0.913±0.018（+0.002） | 0.820±0.040（-0.019） | 0.760±0.054（-0.027） | 0.783±0.093（-0.073） | 0.395±0.149（-0.028） | 0.790 |
| TRANSFER／R_STD_SP | 0.933±0.013（-0.001） | 0.933±0.009（+0.010） | 0.914±0.013（+0.003） | 0.806±0.035（-0.033） | 0.713±0.076（-0.074） | 0.786±0.075（-0.070） | 0.414±0.108（-0.009） | 0.785 |
| WAVG／BEST | 0.933±0.014（-0.001） | 0.934±0.012（+0.011） | 0.912±0.016（+0.001） | 0.816±0.032（-0.023） | 0.764±0.058（-0.023） | 0.787±0.100（-0.069） | 0.393±0.146（-0.030） | 0.791 |
| WAVG／SWA4 | 0.933±0.013（-0.001） | 0.934±0.012（+0.011） | 0.910±0.014（-0.001） | 0.819±0.029（-0.020） | 0.694±0.137（-0.093） | 0.782±0.094（-0.074） | 0.391±0.157（-0.032） | 0.780 |
| WAVG／EMA60 | 0.931±0.014（-0.003） | 0.933±0.012（+0.010） | 0.910±0.014（-0.001） | 0.816±0.029（-0.023） | 0.684±0.148（-0.103） | 0.780±0.095（-0.076） | 0.385±0.159（-0.038） | 0.777 |
| PTSCALE／P5K | 0.929±0.012（-0.005） | 0.928±0.011（+0.005） | 0.911±0.016（+0.000） | 0.808±0.036（-0.031） | 0.736±0.064（-0.051） | 0.793±0.068（-0.063） | 0.409±0.076（-0.014） | 0.788 |
| PTSCALE／P10K | 0.932±0.022（-0.002） | 0.929±0.015（+0.006） | 0.916±0.013（+0.005） | 0.820±0.019（-0.019） | 0.736±0.073（-0.051） | 0.815±0.079（-0.041） | 0.404±0.117（-0.019） | 0.793 |
| PTSCALE／P20K | 0.930±0.019（-0.004） | 0.927±0.012（+0.004） | 0.913±0.014（+0.002） | 0.811±0.025（-0.028） | 0.708±0.097（-0.079） | 0.810±0.075（-0.046） | 0.396±0.112（-0.027） | 0.785 |
| FPDECAY／BASE | 0.928±0.021（-0.006） | 0.929±0.015（+0.006） | 0.914±0.015（+0.003） | 0.805±0.025（-0.034） | 0.739±0.068（-0.048） | 0.797±0.090（-0.059） | 0.353±0.101（-0.070） | 0.781 |
| FPDECAY／FP_DECAY | 0.926±0.015（-0.008） | 0.928±0.015（+0.005） | 0.909±0.015（-0.002） | 0.821±0.016（-0.018） | 0.734±0.077（-0.053） | 0.820±0.075（-0.036） | 0.396±0.103（-0.027） | 0.790 |
| SHIFT／F_G10K | 0.923±0.016（-0.011） | 0.934±0.014（+0.011） | 0.917±0.017（+0.006） | 0.798±0.018（-0.041） | 0.707±0.076（-0.080） | 0.773±0.090（-0.083） | 0.423±0.114（-0.000） | 0.782 |
| SHIFT／P_G5K | 0.925±0.017（-0.009） | 0.929±0.013（+0.006） | 0.915±0.015（+0.004） | 0.801±0.018（-0.038） | 0.730±0.075（-0.057） | 0.806±0.066（-0.050） | 0.323±0.102（-0.100） | 0.776 |
| SHIFT／P_G10K | 0.926±0.017（-0.008） | 0.932±0.012（+0.009） | 0.913±0.017（+0.002） | 0.803±0.037（-0.036） | 0.700±0.084（-0.087） | 0.809±0.069（-0.047） | 0.397±0.134（-0.026） | 0.783 |
| FPM／CONCAT_FP | 0.921±0.015（-0.013） | 0.916±0.010（-0.007） | 0.892±0.017（-0.019） | 0.758±0.097（-0.081） | 0.659±0.143（-0.128） | 0.796±0.086（-0.060） | 0.360±0.103（-0.063） | 0.757 |
| FPM／GATE_FP | 0.921±0.015（-0.013） | 0.915±0.010（-0.008） | 0.894±0.017（-0.017） | 0.757±0.098（-0.082） | 0.662±0.145（-0.125） | 0.798±0.079（-0.058） | 0.359±0.088（-0.064） | 0.758 |
| PH／CONCAT_PH | 0.924±0.016（-0.010） | 0.929±0.010（+0.006） | 0.918±0.013（+0.007） | 0.790±0.024（-0.049） | 0.730±0.073（-0.057） | 0.824±0.059（-0.032） | 0.358±0.095（-0.065） | 0.782 |
| PH／GATE_PH | 0.925±0.015（-0.009） | 0.926±0.011（+0.003） | 0.914±0.016（+0.003） | 0.785±0.021（-0.054） | 0.734±0.075（-0.053） | 0.830±0.061（-0.026） | 0.355±0.109（-0.068） | 0.781 |
| R10／P10K_R10FP_GLOBAL | 0.933±0.019（-0.001） | 0.928±0.009（+0.005） | 0.919±0.013（+0.008） | 0.822±0.019（-0.017） | 0.751±0.070（-0.036） | 0.825±0.062（-0.031） | 0.447±0.110（+0.024） | 0.804 |

没有完整正式预测的工作逐项登记如下；全部写“未完成／不可比较”，不以 validation、其他 checkpoint 或其他周期结果补齐：

| 任务 | SOTA 参考 R² | SOTA 主线 r1／retest1 | DEMOL 两臂 | ET_SINGLE repair1 | FROZEN3D | OPT：A_TOP2／B2_E／B3_C |
|---|---:|---|---|---|---|---|
| Eea | 0.934 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 |
| Egb | 0.923 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 |
| Egc | 0.911 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 |
| Ei | 0.839 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 |
| EPS | 0.787 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 |
| Nc | 0.856 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 |
| Xc | 0.423 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 | 未完成／不可比较 |

缺项原因：旧 SOTA 主线被用户停止；DEMOL、ET_SINGLE repair1、FROZEN3D 的原开发触发条件未满足，实际 TEST=0；OPT 三臂不在最后冻结的正式 TEST 名单。10 月 18 日已归档 F0 同样七项不可比较，原因及消耗见前文，不重复表列或补测。

### 原始证据索引、归档结论与后续边界

| ID | 计划／修订 | 结果入口 | 实际实现／审查材料 |
|---|---|---|---|
| FP | UA2D-PRETRAIN-FP-20261019-01／r4 | [results.json](results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json) | [预训练模块](src/modules/ua2d_galformer_pretrain.py)、[fast worker](scripts/pretrain_ua2d_fast.py)、[实现与事故报告](results/ua2d_pretrain_fp_20261019/fast_r4/final_review/implementation_review.md) |
| CLEAN | UA2D-CLEANFP-READOUT-DISTILL-20261019-01／r1 | [results.json](results/ua2d_cleanfp_readout_distill_20261019/r1/final_review/results.json) | [预训练模块](src/modules/ua2d_cleanfp_pretrain.py)、[动态权重与 FT](src/training/ua2d_readout_distill.py) |
| ATTN | UA2D-V1-ATTNSCALE-20261020-01／r1 | [results.json](results/ua2d_v1_attnscale_20261020/r1/final_review/results.json) | [共享 attention 修改](src/modules/glt_dual.py)、[独立训练入口](src/training/ua2d_attnscale.py) |
| DEMOL | UA2D-DEMOL-LITE-20261020-01／r1 | [results.json](results/ua2d_demol_20261020/r1/final_review/results.json) | [模型](src/modules/ua2d_demol.py)、[交付 diff](results/ua2d_demol_20261020/r1/final_review/code_diff.patch) |
| ET | UA2D-ET-SINGLE-20261020-01／r1 | [results.json](results/ua2d_et_single_20261020/r1/final_review/results.json) | [ET 模块](src/modules/ua2d_et_single.py)、[修复授权与执行](scripts/resume_ua2d_et_single_repair1.py) |
| HOP | UA-HOP-PH-20261020-01／r1 | [results.json](results/ua_hop_ph_20261020/r1/final_review/results.json) | [模型](src/modules/ua_hop_ph.py)，本文件 2026-10-20 既有独立审查 |
| F3D | UA-HOP-PH-FROZEN3D-20261020-01／r1 | [results.json](results/ua_hop_ph_20261020/frozen3d_r1/final_review/results.json) | [模型](src/modules/ua_hop_ph_frozen3d.py)，本文件 2026-10-20 既有独立审查 |
| OPT | UA2D-OPT-20261021-01／r1-FT-REPAIR1 | [results.json](results/ua2d_opt_20261021/r1/repair_ft1/results.json) | [模型](src/modules/ua2d_opt.py)、[成功 attempt 映射](results/ua2d_opt_20261021/r1/repair_ft1/attempt_mapping.json)、[修复 diff](results/ua2d_opt_20261021/r1/repair_ft1/actual_diff.patch) |
| TRANSFER | UA2D-TRANSFER-20261022-01／r1 | [results.json](results/ua2d_transfer_20261022/r1/final_review/results.json) | [模型](src/modules/ua2d_transfer.py)、[交付 diff](results/ua2d_transfer_20261022/r1/final_review/actual_source_diff.patch) |
| WAVG | UA2D-WAVG-20261022-01／r1 | [results.json](results/ua2d_wavg_20261022/r1/final_review/results.json) | [训练与平均](src/training/ua2d_wavg.py)、[平均来源记录](results/ua2d_wavg_20261022/r1/final_review/estimator_sources.json) |
| PTSCALE | UA2D-PTSCALE-20261022-01／r1 | [results.json](results/ua2d_ptscale_20261022/r1/final_review/results.json) | [PT worker](scripts/pretrain_ua2d_ptscale.py)、[完整训练与评估](src/training/ua2d_ptscale.py) |
| FPDECAY | UA2D-FPDECAY-20261022-01／r2-SG | [results.json](results/ua2d_fpdecay_20261022/sg_r2/final_review/results.json) | [SG 训练与权重](src/training/ua2d_fpdecay_sg.py)，CURRENT_PLAN.md 第 2 节既有深度审查 |
| SHIFT | UA2D-3DSHIFT-20261022-01／r1.5-ALL1-PCTRL-GPU2 | [results.json](results/ua2d_3dshift_20261022/r1/all1_pcontrol_gpu2/final_review/results.json) | [周期几何](src/modules/ua2d_3dshift.py)、[完整交付 diff](results/ua2d_3dshift_20261022/r1/all1_pcontrol_gpu2/final_review/actual_source_diff.patch) |
| FPM | UA2D-FG10K-MORGAN-20261023-01／r1.1 | [results.json](results/ua2d_fg10k_morgan_20261023/r1/final_review/results.json) | [FP 模型](src/modules/ua2d_fg10k_morgan.py)、[交付 diff](results/ua2d_fg10k_morgan_20261023/r1/final_review/actual_source_diff.patch) |
| PH | UA2D-PH-DESC-20261023-01／r1.2-P10K-GPU01 | [results.json](results/ua2d_ph_descriptor_20261023/r1_p10k_gpu01/final_review/results.json) | [PH 描述符](src/dataset/ua2d_ph_descriptor.py)、[融合模型](src/modules/ua2d_ph_descriptor.py) |
| R10 | UA2D-P10K-R10FP-20261023-01／r1 | [results.json](results/ua2d_p10k_r10fp_20261023/r1/final_review/results.json) | [冻结数组融合](src/training/ua2d_p10k_r10fp.py)、[交付 diff](results/ua2d_p10k_r10fp_20261023/r1/final_review/actual_source_diff.patch) |

每个 results 入口所处周期根同时保留 budget.json、execution_manifest 等实际执行记录；报告、CPU 日志和原始产物路径以相邻 final_review/report.md 或表中专项报告为准。OPT 的正式预测在原 r1/formal，修复尝试在 repair_ft1/attempts，由 attempt_mapping／artifact_index 关联；不能只扫描一个目录推断缺折。SHIFT 的正式冻结文件名是 selection/all_evaluations_frozen.json 及各 frozen_*.json，不要求另造 final_frozen.json。

取消与事故补充入口：[旧 SOTA 初次消费](results/ua2d_sota_20261018/r1/budget.json)、[retest1 停止授权与动作](results/ua2d_sota_20261018/r1_retest1/user_stop_20261019.json)、[retest1 消费](results/ua2d_sota_20261018/r1_retest1/budget.json)、[ET 原失败与修复报告](results/ua2d_et_single_20261020/r1/final_review/report.md)、[SHIFT 累计及未知退出证据](results/ua2d_3dshift_20261022/r1/all1_pcontrol_gpu2/budget.json)。工程实现阶段的静态检查不代替这些运行结果。

归档结论：16 个交付周期的实际实现／结果证据按上述限定范围审查通过并收口；旧 SOTA 主线只归档取消、部分消费和证据限制，完整实验未完成。达到本轮主性能门槛的固定候选是 P10K_R10FP_GLOBAL；CLEAN 的 QL_DYN−QL_ALL1 仅作为匹配内部对照的数值正结果保留。其余完整候选未达到各自门槛，开发阶段未做 TEST 的工作维持不可比较。不自动重新开放任何历史队列、不补跑、不修改原结果或 checkpoint，不按 TEST 改选任务、权重或父模型。

本次只向 PROJECT_HISTORY.md 和 CURRENT_PLAN.md 写入审查归档；新增 prepare／PT／FT／TEST／CPU suite／GPU 使用均为 0，未读 TODO.md、未做哈希／校验和、未进行远程操作，未删除文件。后续执行资源遵守当前 AGENTS.md 的物理 GPU3、PT 最多 2 路／FT 最多 4 路；本次归档本身不产生执行授权。
