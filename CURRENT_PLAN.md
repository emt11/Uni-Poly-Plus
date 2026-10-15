# F_CE 参数优化审查与候选计划

计划 ID：FCE-PARAM-20261014-01 / r4。日期：2026-10-14（北京时间）。本次用户明确授权更新并发及自主接续，科学配方和累计预算沿用r2。

状态：**待审查**。授权来源：用户明确要求接续现有 FCE 周期；FT、validation-only 选优、冻结、70 个正式预测和最终 CPU 测试已由实际产物记录，独立审查仍待完成。

**唯一执行对象是本文件第4节首轮四臂，遵循第6／7节；第5节及CURRENT_PLAN.md中的旧GLT队列不执行、不恢复、不推进。** CURRENT_PLAN仅作为FCE交接和资源核对；不读TODO。r2科学runtime、配置、初始化、source_used、所有成功产物保留，不另开实验、不重置预算。

## 1. 审查结论与目标

**优先保留原 F_CE 预训练包和结构，调微调学习率及其调度，再研究正则化。当前证据不足以支持先加宽、加深或添加新的融合模块。**

具体优先级：

1. 联合训练 head LR：当前 1e-3，首选检验 3e-4，减小解冻阶段的更新幅度。
2. 联合训练 backbone LR：当前一般任务 1e-4、Xc 2e-4，首选分别检验 3e-5、6e-5。
3. joint scheduler：当前每 10 epoch 重启，首选检验覆盖整个 joint 阶段的单段 cosine。
4. 在上述结果基础上，才考虑 head dropout、backbone weight decay、分模块 LR；不同时叠加所有改动。
5. 预训练schedule保留为后续研究方向；仅在已有完整结果时只读参考P1，不等待、启动或恢复旧计划P1。本轮PT预算为0。

本轮唯一性能有效性判据改为：**正式预测的七任务宏平均 R² 相对本轮匹配基线 R0 提升至少0.010，即 Δmacro R² ≥ 0.010**。这里是R²绝对差，不是相对提升1%；等于0.010也有效，必须使用未舍入值判定。各任务先对五折R²取算术平均，再七任务等权平均，不按样本数加权。

七任务逐项达到论文参考仍作为长期目标及报告项，**不再是本轮有效的必要条件**。撤销r1的逐任务退化门槛及Egb／Egc保护门槛；出现个别任务退化时如实展示，但不额外否决已达到宏平均门槛的候选。Xc、Ei、Nc仍是主要缺口。历史F_CE宏值仅作额外数值参照，不能在结果出来后改换基线以通过门槛。

### 1.1 当前正式结果与差距

基线取历史四 rank PT 的 **F_CE**，不是两 rank F_CE_2R，也不是 UA_2D 或当前 A1。数据来自 [七任务汇总](results/glt_periodic_seam_20261009/r1/final_review/seven_task_results.json) 与各折 `formal/F_CE/<task>/foldN/metrics.json`。本轮确认 35 个正式预测文件存在、逐折保存指标均值与汇总一致，并复用 [既有独立审查记录](results/glt_periodic_seam_20261009/r1/final_review/independent_review.json)；没有重算预测、重新 strict-load 35 个 best 或重跑历史测试。

|任务|F_CE R² mean±std|论文参考模型|论文 R² mean±std|F_CE−参考|判断|
|---|---:|---|---:|---:|---|
|Eea|0.924±0.018|Periodic-TDL|0.934±0.013|-0.010|接近，继续改善|
|Egb|0.941±0.012|Periodic-TDL (+Chem, +RC)|0.923±0.009|+0.018|数值已达到，报告后续变化|
|Egc|0.914±0.012|Periodic-TDL (+Chem, +RC)|0.911±0.008|+0.003|余量较小，报告后续变化|
|Ei|0.799±0.023|Periodic-TDL|0.839±0.018|-0.040|主要优化任务|
|EPS|0.776±0.046|Periodic-TDL|0.787±0.033|-0.011|接近，但折间波动较大|
|Nc|0.821±0.059|Periodic-TDL|0.856±0.032|-0.035|主要优化任务|
|Xc|0.324±0.160|Periodic-TDL|0.423±0.102|-0.099|最大缺口，优先降低不稳定性|

口径：每任务五个 outer folds、seed42+fold，每折按 inner-validation 原单位 RMSE 选 best；R² 为五折算术平均，std 为 ddof=0，不是 pooled R²，也不是每折多个 seed。七任务等权 macro 为 0.786，参考最佳值包络均值为 0.810；后者不对应一个论文模型。所有差值由原始数值计算后保留三位小数。

协议限制：共享 TEST 已参与历史开发，不是独立盲测。当前 split manifest 的 Egc 为 3380 条、Xc 为 432 条；历史交付记录指出论文对应为 4124／430，论文原始数据及精确 split 等价性本轮未独立核实。本文仅对照 [SOTA_RESULT.md](SOTA_RESULT.md) 的数值，不宣称公平超过论文。EAT、Eib、Tg 不纳入七任务目标。

## 2. 当前模型和实际参数

实现依据：[模型](src/modules/glt_dual.py)、[token fusion](src/modules/glt_dual_opt.py)、[微调函数](scripts/finetune_mcl_ph.py)、[F_CE 专用入口](scripts/run_glt_periodic_seam_unit.py)、[PT 配置](results/glt_periodic_seam_20261009/r1/initialization/F_CE.json)。本轮读取了相关实际 diff；工作树已有大量改动，不能将当前源码直接等同于历史运行快照。

|部分|当前 F_CE 配方|调参含义|
|---|---|---|
|二维 O8／三维 Galformer|各 6 层、hidden 512、FFN 2048|改变层数／宽度会改变权重形状，不是复用 parent 的首轮微调参数|
|几何|finite_incident Trimer、angle_value、incident readout|首轮固定，不更换输入语义|
|token fusion|atom query 对全图物理 bond tokens；4 heads×32；残差 scale 为可学习参数|0.1 是构造初值，不是已核实的预训练末值；恢复 parent 后不得重置|
|图级预测|融合原子 mean pool＋三维 incident 图表示，各自 LayerNorm 后 concat；1024→512→1，GELU|首轮固定 head 形状和初态，不搜索宽度|
|PT 目标|fused atom CE＋length/angle geometry，权重 [1,1,0]|无 ALIGN、torsion、MD、endpoint 或路径 dropout|
|PT 优化|AdamW LR2e-4、WD0；warmup2000、cosine total20000，实际 stop5000|固定步数下 schedule 与终点不匹配，值得后续研究|
|PT 输入／预算|mask0.30、noise0.03；micro84×accum3×world4＝global1008；5000 updates|首轮 FT 不重新 PT；不能把两 rank 重训当作原 parent|
|FT 公共条件|batch24、FP32、workers0、MSE、train-only StandardScaler（含 Xc）、clip5|固定数据、scaler、batch 和随机种子|
|head-only|10 epochs，LR3e-4，Cosine T_max10／eta_min3e-5|保持冻结 encoder 且 eval；encoder dropout 不参与该阶段|
|joint|一般任务60 epochs，Egc50；head LR1e-3；backbone一般1e-4、Xc2e-4|当前仅 backbone／head 两组，没有二维／三维差分 LR|
|joint scheduler|WarmRestarts T_0=10、T_mult=1、eta_min1e-6，每 epoch step|优先比较取消重复升温，不先增加训练长度|
|WD／dropout|一般任务 WD0.02、head dropout0.1；Egc WD0、head dropout0；Xc WD1e-4、head dropout0.3；encoder dropout全部0|不能把统一 uniform_u1 当作历史 F_CE 原配方|

WD 当前施加于组内所有可训练参数，包括 bias／norm；本轮不同时改变 decay 分组方式。head→joint 会重新建立 AdamW，head 动量状态也重置。首轮只改 LR 或 scheduler，保留重建 optimizer 行为，以避免额外混杂。

历史 parent：[deploy_05000.pt](results/glt_periodic_seam_20261009/r1/pretrain/F_CE/deploy_05000.pt)。共同 head 来自 `results/glt_fusion_20261007/r1/finetune/B0/<task>/foldN/head_initial.pt`。二者本轮均定位到实际文件。

## 3. 为什么先调这些参数

### 3.1 解冻阶段存在共同的性能下降

本轮按 [selection_before_test.json](results/glt_periodic_seam_20261009/r1/selection_before_test.json) 指向的 35 个训练目录读取 `progress.json`：**35/35 个单元在 epoch10→11，即 head-only→joint 时 validation R² 下降**。

head-only 最后一个 epoch 实际使用约 3.66e-5，完成后 scheduler 记录为 3e-5；joint 首个 epoch 重新使用 head LR1e-3，约为前一轮实际 LR 的 27 倍，且同时解冻全部 encoder、重置 AdamW。注意日志 `learning_rates` 在 scheduler.step 后记录，代表下一轮 LR，不能直接当作刚完成 epoch 的 LR。

例：EPS fold0 的 validation R² 从 0.612 降到 -1.458；Xc fold2 从 0.252 降到 -0.820。它们之后会恢复，因此不能据此断言训练错误或低 LR 必然提高正式结果；但这是检验更新幅度的直接依据。

### 3.2 不支持先统一增加 epoch

|任务|五折 validation 最佳 epoch（总 epoch 编号）|最佳 validation R²−末轮 R² 的五折均值|
|---|---|---:|
|Eea|30、30、64、19、33|0.019|
|Egb|44、69、29、30、30|0.005|
|Egc|58、30、47、58、20|0.005|
|Ei|57、68、70、32、46|0.012|
|EPS|15、59、20、46、55|0.094|
|Nc|32、47、25、32、26|0.093|
|Xc|18、57、44、69、56|0.143|

EPS／Nc／Xc 的末轮退化较明显，符合过拟合或优化波动的现象，但单凭这些曲线不能区分两者；best 已按 validation 保存，末轮退化不是当前正式预测误用了 last checkpoint。先稳定更新，再决定是否需要延长训练。Ei 有较晚的最佳轮次，但只有一折在最终轮，不能直接推广为七任务都欠训练。

数据规模也支持谨慎微调：fold0 的 train/validation/test 数量分别为 Eea235/59/74、Egb358/90/113、Egc2163/541/676、Ei236/60/74、EPS244/61/77、Nc244/61/77、Xc276/69/87。小验证集上的 best 波动不能靠反复查看 TEST 来解决。

### 3.3 既有结果能回答什么

- [U1 统一 FT](results/glt_optimization_20261011/r4/U1/final_review/recovery1/seven_task_results.json)：原 parent 的 U0_FUSED 宏 R² 0.787；Xc 为 0.336，相对历史 0.324 增加 0.011。Xc 同时改了 backbone LR、WD、head dropout，Egc 还改变了 epoch，因此不能归因于单一参数，也不应把整个 uniform_u1 无条件替换为基线。
- [旧低 LR 实验](results/glt_ft_generalization_20261008/r1/final_review/seven_task_results.json)：B0 上 FT_LOW_LR 将 backbone LR 乘 0.3，宏 R² 从 0.768 到 0.780，Xc 从 0.292 到 0.358，但 Egb 下降。它支持一个候选方向，**不构成 F_CE 上已验证的结论**；旧入口实际绑定 B0 parent。
- [F_CE 路径 dropout](results/glt_ft_robustness_20261011/r1/final_review/seven_task_results.json)：D 的 Nc 为 0.846，但 EPS 为 0.762、Egc 为 0.909，不能完成七任务目标。token／readout 路径 dropout 与普通 head dropout 不等价。
- [原 N1 结果](results/glt_optimization_20261011/r7/N1/final_review/recovery1/seven_task_results.json)：取消 target standardization 的 F_CE 宏 R² 为 0.765，Xc 为 0.231。首轮继续 train-only standardization，不通过改标签尺度来扩展搜索。

上述候选只读复用保存结果；本轮没有对这些历史周期重新出具独立通过结论。不能按 TEST 将不同候选的最好任务拼成一个模型。

### 3.4 预训练 schedule 值得排在 FT 后

[实际 PT 日志](results/glt_periodic_seam_20261009/r1/pretrain/F_CE/records_rank0.jsonl) 的 step5000 LR 为约 1.866e-4，仍为峰值的 93.3%。5000 步中 warmup 占 40%，尚未经历低 LR 收尾；这是 schedule 预算不匹配的事实，不等同于已证明欠收敛。

旧计划曾列P1研究total20000→5000，但旧队列已取消，本轮不启动或恢复它；若已有完整结果，可只读参考，否则标未完成。其拟议两rank匹配参考不能直接把相对历史四rank F_CE的变化归因schedule。CE／geometry的loss单位不同，也不能仅因数值大小就断言梯度失衡。

## 4. 首轮拟议实验：固定 F_CE 的三个单因素比较

本节科学方案已由r2实际执行；r3保持单一模型结构和公共参数变更规则，不新增task-specific搜索、混用parent或组合候选。采用历史每任务配方，保留Xc／Egc原有特殊设置。r3接续时从实际账本确定已完成与剩余单元，不以本文快照作为固定剩余数量。

|候选|scientific question|reference|controlled change|comparison budget|
|---|---|---|---|---|
|R0|当前固定运行版本下的匹配基线|历史 F_CE 仅作数值参考|同一 parent、原 FT 配方；建立本轮参考|七任务×五折，head10＋joint60，Egc joint50|
|H_LR|head 更新过大是否损伤微调？|R0|只将 joint head LR 1e-3→3e-4|同 R0；head-only LR、backbone LR、scheduler 均不变|
|B_LR|减小预训练 backbone 更新是否改善泛化？|R0|只将 joint backbone LR 乘0.3；一般3e-5、Xc6e-5|同 R0；head LR、WD、scheduler 均不变|
|S_COS|减少重复升温是否改善泛化？|R0|只将 joint scheduler 改为单段 CosineAnnealingLR，T_max为joint epoch数，eta_min1e-6|同 R0；两组起始 LR 不变；scheduler 每 epoch 末 step|

三项分别与 R0 比较，不把 H_LR 与 B_LR 同时降低后称单因素。S_COS 的最后一个训练 epoch 使用倒数第二个调度点，末次 step 到 eta_min；明确这一约定，不额外补跑 epoch。第一轮不加入 joint warmup，以免同时改变多个调度因素。

### 4.1 固定条件与最小实现范围

- 固定原 F_CE step5000 parent、原 head 初态、seed42+fold、七任务各 fold0—4、原 train/validation/test 索引、train-only scaler、loss、batch、精度、clip、epoch、采样和模型结构。
- 开始首轮时锁定一个实际运行版本。若复用 r10 的 head-only cache，则四臂一致启用；global fusion 保留 loop。不能把 cache／运行版本变化与某个候选绑定。
- 现有 `uniform_u1` 会拒绝 LR override；需要最小的显式 FT recipe 传递与保存，不能偷偷修改该 profile。`ft_candidate=FT_LOW_LR` 虽已有 backbone倍率逻辑，旧专用入口绑定 B0，不能直接调用旧 controller 作为 F_CE 实验。
- 拟议修改限于实际 FT 函数所需参数接线、一个本轮入口／汇总及相应窄测试；配置保存实际 optimizer groups、dropout、scheduler、parent、head、scaler、运行版本。不修改共享 encoder 结构、旧严格恢复协议或旧结果。
- 保留 fusion 训练后的 scale 和全部共同权重，不在微调时重置 alpha／norm／angle 模块。不新增训练缓存体系或通用搜索框架。

### 4.2 验证选优、正式预测和有效性判定（r2替代旧规则）

1. 每个 fold 内仍按原单位 validation RMSE 选择 best；四臂所有单元完成后才选参数。候选筛选不读取其 TEST。
2. 使用各折validation选中best的R²，先在每task五折平均，再对七task等权平均，得到validation macro R²。按未舍入的validation macro R²从H_LR、B_LR、S_COS中选一个最高者；完全相同时依次优先H_LR、B_LR、S_COS。不再按相对RMSE改善筛选，也不施加逐任务退化门槛。
3. 即使最高候选的validation宏差不足0.010或为负，也按上述固定规则选定它并记录该事实，完成唯一一次正式复评；validation用于选参数，不提前宣称正式有效。四臂未完整完成时不使用剩余臂替换完整比较，不据缺失折作判定。
4. 同时展示每任务／每折validation R²与RMSE、改善折数、最佳epoch及train/validation曲线；个别任务退化必须如实报告。RMSE下降为好，但不参与本轮唯一有效性门槛。
5. 全部140个FT单元完成后，冻结一个全任务候选及R0的全部best、recipe、scaler，再对R0与该候选各35折进行正式预测，共70次。不对其余候选查看TEST，不因TEST未达标改选第二名、追加组合或继续搜索。
6. **正式有效 ⇔ macro_TEST_R²(candidate)−macro_TEST_R²(R0) ≥ 0.010**。只有两臂完整七任务五折且无影响结果的正确性问题时才判定；低于门槛为“本轮未证实有效”，结果不完整则为“未完成／不可判定”。使用原始精度判定、三位小数展示，不能因显示为0.010就判通过；同时保存原精度差与判定布尔值。历史F_CE宏差另外报告，不替换R0作为主判据。
7. 最终按§1.1逐项报告候选、R0、论文参考和差值，以及宏R²、宏差、是否有效。七项全部达到参考才另外称“七项数值目标达到”；有效不等于七项SOTA达标。共享TEST仍不是独立盲测，本轮结论不代表统计显著或多seed稳定优势，不自动替换部署。

### 4.3 全周期累计硬预算（r2＋r3共用，不重新计数）

|范围|单个 arm|首轮上限|
|---|---:|---:|
|PT|0|0|
|FT 单元|35|140（R0＋3候选）|
|FT epochs|2400|9600|
|FT optimizer updates|51450|205800|
|正式预测|35|70（R0＋validation最高候选）|
|额外 seed／task／fold／ensemble|0|0|

epoch／update 数依据原 F_CE 35 个保存 metrics 合计：Eea3500、Egb5250、Egc27300、Ei3500、EPS3850、Nc3850、Xc4200 updates。不是 uniform_u1 的56000 updates／arm。各候选都完整五折，不以部分 fold smoke 作性能结论。新 R0 用于匹配运行版本，不重跑历史实验来验收其旧结果；若已有**完全同版本同配方** R0，可只读复用并扣减新训练预算，不另加候选。

所有长任务在tmux session `Uni-Poly`独立window，新调度控制器`fce_param_r4_controller`。仅物理GPU2、3：FT／正式预测每卡以4路为目标，正常joint遥测支持时按4→6→8提升，每卡最多8、全局最多16；已有单元计入槽位。各进程CUDA_VISIBLE_DEVICES=2或3、内部cuda:0，独立model／optimizer／RNG／dataset／输出，保持batch24、FP32、workers0、OMP4、BLAS/MKL1，不用DDP或MPS，不改变科学配方。具体资源、观察及回退规则见§7。

继续使用results/fce_param_20261014/r2/与logs/fce_param_20261014/r2/；本次调度源码／接管／遥测独立保存scheduler_r4/。旧source_used／初始化／metadata／checkpoint／预测／失败现场不覆盖；成功不重训、在途不重派。只有唯一控制器可派发、冻结、汇总和写交接。

## 5. 后续参数空间：排序建议，尚不排队执行

只有首轮回答完三个问题后，才从以下选择一个完整周期并明确候选数及预算；本节当前执行预算全部为0，不形成笛卡尔积搜索。

|顺序|参数方向|建议小范围|适用证据与注意事项|
|---|---|---|---|
|1|组合已有效的 LR／调度|仅组合首轮 validation 支持的设置，保留匹配单改动参照|组合效果未经测试；不能把三个单因素收益相加|
|2|head dropout|一般任务0.1→0.2；Xc0.3→0.1或0.2；Egc0→0.1|先单改 dropout；U1 的 Xc 改善不能单独归因 dropout|
|3|backbone WD|一般0.02→0.01或0.05；Xc1e-4→0.01；Egc0→0.01|head WD 保持参考；低 LR 时 AdamW 衰减总效应也随 LR 改变，不能只比较名义 WD|
|4|分模块 LR|O8保留获选 LR，Galformer单独乘0.3；token fusion／norm保留参考|只检验三维分支适应强度，不同时重设 scale 或关闭路径；需显式无遗漏／无重复参数组|
|5|部分冻结|冻结低层、保留高层及融合更新|旧 B0 FT_PARTIAL 未显示全面收益，优先级低于 LR；不能将旧冻结列表直接视为 F_CE 的合适策略|
|6|PT schedule|已有完整P1结果才只读参考；否则后续另定方案|本轮不恢复P1；后续需保持world／micro／accum／FT匹配，不作跨rank因果比较|
|7|PT mask／geometry权重|分别考虑mask0.30→0.15，或geometry权重1→0.5／2|每次仅改一个；noise0.03先保持。需新PT，先明确目标和预算再训练|

不优先增加 hidden512／6层／attention heads、预测 head 容量、训练 epochs、batch、ensemble、ALIGN、torsion、FiLM 或非键几何。前几项可能破坏 parent 权重兼容性或改变优化预算；后几项是结构／目标路线，当前计划已有独立问题。本文的目标是先回答现有 F_CE 参数是否合适。

如统一规则仍不能同时覆盖七项，可另行讨论**按任务、只凭 inner-validation**选择有限配方的后续方案；必须明确它改变了当前“不按任务改配方”的研究约束，不能在本轮自动实施，更不能从历史 TEST 最好项拼接。届时对照应称同一架构的逐任务调参结果，不称单一统一训练配方。

## 6. 停止条件、最终一次性检查与交接

r1／r2文档轮实际完成模型、参数、既有结果与曲线审查。r3文档轮新增只读检查：当前进程、nvidia-smi遥测、tmux、CPU／内存、FCE控制器代码及已有执行日志；只修改PLAN.MD，没有改代码或调度、启动／终止训练、执行pytest或重新预测。未读取TODO.md，未修改CURRENT_PLAN.md／PROJECT_HISTORY.md／SOTA_RESULT.md。

本轮最终检查仅一次：审阅 `PLAN.MD` 实际 diff、Markdown 格式、现有证据链接及预算文字。无代码测试需求。

未来已授权执行的停止条件：parent／head／scaler／schema／split不匹配，NaN/Inf，writer冲突，预算超出，正式产物覆盖风险，或资源未协调。停止受影响的运行、保留现场，不静默跳过失败折，不用减少fold或换seed补齐结果。当前计划中的运行状态变化本身不构成本轮启动或停止其他任务的授权。

未来最小代码实现与授权实验完成后，集中一次最终CPU测试，空CUDA_VISIBLE_DEVICES、timeout300秒：原recipe／scheduler／恢复／选优测试之外，仅增加本次调度正确性问题——每卡最多8／全局最多16、4→6→8的正常joint观察和资源回退、同unit不重复派发、已有成功跳过、在途接管／失败不误重跑、陌生进程拒绝新派发、冻结仅一次、累计预算不重置。调度用CPU模拟进程状态，不启动GPU测试。复用已有模型／缓存证据，不重放全仓回归、阶段smoke或benchmark。仍使用本周期原最终测试额度，不为并发另开测试轮次。

独立审查复用最终实际diff、唯一测试结果、已有训练日志和预测产物，不重跑训练。只有实验真正完成且审查通过后，才在 CURRENT_PLAN.md 写齐七任务结论，并将实际模型和结果归档 PROJECT_HISTORY.md；本文纯文档审查无需模型历史归档。

**当前交付结论：r3支持在GPU2、3各2路、总4路接续原周期；科学矩阵和有效性门槛不变。现有r2在运行，本轮文档修改新增训练消耗0，多进程实际提速尚未验证。**

## 7. 无人值守连续执行与动态调度（r4当前授权）

只完成第4节固定矩阵，按R0→H_LR→B_LR→S_COS顺序派发；全部FT完成后按validation选唯一候选、冻结，再执行R0与候选共70正式预测，最后集中一次最终CPU测试及汇总。预算从r2累计：140 FT、9600 epochs、205800 attempted updates、70正式预测；失败计入，不增加PT／seed／候选或搜索。12小时不是终止期限，不等用户逐阶段确认。

### 7.1 安全接管与唯一writer

1. 核实实际控制器PID／命令／cwd、在途unit、启动器及exit／metrics／updates；若周期已完成，仅报告，不重启。
2. 准备唯一接续控制器，以普通文件锁和实际进程身份防止双主体；不能原样再运行r2入口、重建初始化或source_used。
3. 只暂停旧FCE控制器进程本身的派发，不向进程组、正常训练unit、日志启动器或其他任务发信号；在途unit及启动器自然结束并写真实exit后，旧controller单独退出，新controller接管。
4. 当前实际接管现场：r2 controller PID1643590已单独SIGSTOP，Egc fold0／1自然运行；r3调度器仅准备、未启动。本次r4继承该暂停现场，不另停训练。旧GLT仍已取消。
5. 接续读取现有r2全部成功／失败／在途／冻结／预测／测试状态；仅成功exit0且产物齐全才跳过成功。未退出不能因缺metrics当作待执行，拒绝同unit多writer。冻结、TEST和最终测试已完成时不得重做。
6. 保留r2运行版本和目录身份；新调度记录scheduler_r4/，旧调度记录及源码保留。旧controller退出后才接管其交接实时区段。

### 7.2 FT／正式预测并发、遥测和资源边界

- 仅物理GPU2、3；每卡以4路为目标，支持时按4→6→8增加，最多8／卡、16总路，已有单元计入；独立状态和输出。batch、FP32、workers0及GPU单元OMP4、BLAS/MKL1均不变。
- 每次提高后，正常joint训练观察至少约180秒；不得用head-only或单帧显存作下一次升档依据。现有任务持续推进、正常joint数据足够、资源有余量且无明确可比吞吐退化才提升一级，否则保持当前级别。持续高负载或吞吐不再改善时保持有效并发，不追求填满显存，不假定翻倍并发翻倍提速。
- 每次启动新槽位要求GPU空闲显存≥12 GiB、主机MemAvailable≥32 GiB，无无法归属GPU进程；显存余量不保证峰值安全。利用命令／cwd／环境／启动记录及可核实主机/容器PID映射区分本周期和陌生进程，不仅按PID数字、数量或同时出现推定归属。无法归属不增槽，继续已知在途任务或空卡一路安全执行，不杀别的作业、不换GPU0／1。
- 显存占用≥80%、MemAvailable<32 GiB或可比任务／joint阶段的总吞吐明确下降时，停止补充高位槽，在途自然完成后降低一级；恢复后只有再次满足180秒观察和资源条件才重新升档，不频繁试探。达到高负载、吞吐平台或边界时保持有效并发。
- 每卡槽位／unit阶段、进度、真实exit、NVML进程映射证据、GPU显存和利用率、CPU／主机可用内存、重叠时段及实际完成吞吐持续写入遥测。ETA按真实完成／近期更新吞吐估计，不机械除以4或8；区分task／head／joint和等待，无足够可比数据不宣称提速。
- 不运行额外benchmark、探针、重复训练或训练验证并发。真实OOM、NaN/Inf、writer冲突停止受影响运行、保留现场并降低后续并发；不静默重跑、不减batch、不改数据语义或扩大预算。成功不重训，失败消耗累计；仅完整可信model／optimizer／scheduler／RNG／采样状态可精确恢复，无此状态且预算不足则记录阻断，继续不受影响的已定义单元，缺折不判有效。

### 7.3 PT边界

当前FCE首轮PT预算为0，无已定义且授权的本轮PT候选，不启动任何PT，不恢复旧计划。只有未来本文件明确授权且有候选配置／预算的PT周期，才适用每卡2～3进程上限：两rank实验成组占GPU2、3各一rank，最多3组同时，先2组、正常吞吐／资源支持才3组；独立端口／rendezvous／日志／checkpoint，不改变world、micro、accum、global batch，不与FT／正式预测混跑。该条仅界定调度，不产生新的科学授权。

### 7.4 测试、汇总与交付

调度槽位／动态升级180秒／回退／PID归属／接管／防重复／冻结一次／累计预算检查合并进原本周期最终CPU测试，空CUDA_VISIBLE_DEVICES、timeout300秒。正常一次；必要修复仅受影响项再一次≤300秒，累计最多2次。无阶段smoke、额外GPU测试、全仓回归或hash验证。

最终写CURRENT_PLAN：累计四臂完成数、epochs／attempted updates／正式预测、真实CPU结果、选定唯一候选、R0／候选正式宏R²／原精度差值与≥0.010有效判定、完整七任务变化与论文数值对照、实际并发及可核实吞吐、tmux／命令／环境／日志／checkpoint／预测／冻结／接管路径。独立审查未进行标待审查，不归档模型历史、不改SOTA_RESULT／部署，不commit／push；不把启动写为完成。

## 8. 当前执行授权摘要

用户本轮明确要求“请开始修改调度并接续执行，完成后再交付”，授权以上r4调度条款，覆盖r3每卡2路限制；仅接续既有首轮四臂，科学预算与判定不变，无其他实验授权。该授权已同步CURRENT_PLAN；执行状态以实际日志和产物为准。


## FCE r4 PID归属修复（用户“请解决／请继续”授权）

原因：NVML PID与容器PID不同，容器NSpid只包含当前命名空间编号；本环境未发现可用宿主proc映射源。旧策略把无法映射当作本周期任务不可调度，造成每卡一路。没有伪造映射或按数量推断归属。

本次最小修复改用两类独立证据：本地unit必须匹配启动记录中的完整argv、cwd、GPU字段、实际CUDA_VISIBLE_DEVICES以及stdout/stderr打开文件，记录PID和start_ticks；NVML继续提供整卡真实资源，未映射PID保留为unmatched_nvml_pids，nvml_mapping_verified=false。已有本周期可信在途unit时，允许在原显存／内存／并发边界内填槽，不再以无法取得的host PID映射作为硬前提。空卡接管若仍有NVML未知占用且没有本周期可信unit则拒绝；任一本地unit身份核实失败也拒绝新增。该方案不证明整卡没有其他容器作业，不将未知PID冒充本周期进程、不向其发信号；其他占用已包含在整卡空闲显存与负载中。此条覆盖旧“任意NVML PID无法映射都禁止增加槽位”的规定，其他科学预算及4→6→8边界不变。

仅暂停已核实r4控制器本身（PID1654602），所有在途unit自然完成；旧控制器及源码现场保留于results/fce_param_20261014/r2/scheduler_r4/pid_repair1/。修复仅涉及scripts/run_fce_param_r4.py、tests/test_fce_param_scheduler.py、PLAN.MD和CURRENT_PLAN.md，不改训练unit／recipe／模型。新控制器使用原scheduler_r4/controller.lock及原r2产物，累计预算不重置。

本次修复完成后一次窄最终CPU测试：tests/test_fce_param_scheduler.py，空CUDA_VISIBLE_DEVICES、timeout300秒、tmux Uni-Poly:fce_r2_final_cpu_pidrepair1。这是本周期两次CPU额度中的第一次；原全周期结束仅运行tests/test_fce_param.py，复用本次调度结果，不再重复调度suite。两次合计上限保持2，无额外训练／GPU探针／benchmark。真实结果后续写入。


### PID归属修复实际交付记录

2026-10-14T13:44:47.229273+00:00：修复已生效，科学周期仍执行中。GPU2、3各4路，8个unit全部joint训练；每卡4个本地启动身份核实通过，holds均为空。GPU2空闲40150 MiB、GPU3空闲39737 MiB，瞬时利用率均99%；该观测不作为吞吐提速结论。后续4→6→8仍按正常joint≥180秒及持续负载／资源条件自动决定，高负载保持当前有效并发。host NVML映射仍未取得，记录为未知，没有冒充已验证。

最终调度CPU检查8 passed、1 warning、4.56秒、exit0，日志logs/fce_param_20261014/r2/final_cpu_pidrepair1/；未新增GPU测试或benchmark。在途unit自然完成，旧controller计划退出-15，unit信号次数0。新控制器Uni-Poly:fce_param_r4_pidrepair1_controller，日志logs/fce_param_20261014/r2/scheduler_r4/pid_repair1/controller/，只接续PLAN.MD既有矩阵，已完成项不重训。实际接管与并发快照见results/fce_param_20261014/r2/scheduler_r4/pid_repair1/{takeover,repair_completion}.json，原科学产物results/fce_param_20261014/r2/。


## 用户紧急降载：每卡最多两路（2026-10-14）

用户明确要求降低过载，并指定GPU2、3各二路。本条覆盖此前4→6→8：每卡固定上限2，总上限4，不再自动升档；资源不足可少于2。旧控制器1666617已暂停派发，在途最多保留每卡2个活跃进程，多余进程如有则SIGSTOP保留内存状态，按原进度顺序SIGCONT接续，绝不重训。暂停进程仍占显存，不算活跃计算槽位。待旧队列自然完成后仅退出旧控制器，再启动两路调度。科学矩阵、累计预算、有效门槛不变。交接证据results/fce_param_20261014/r2/scheduler_r4/two_slots/，日志logs/fce_param_20261014/r2/scheduler_r4/two_slots/。

两路限制已生效：2026-10-14T13:51:36.276275+00:00，GPU2、3目标均2，实际槽位{'2': 2, '3': 2}。新控制器PID1670439，窗口Uni-Poly:fce_param_two_slots_controller；旧队列全部自然结束，暂停unit已全部恢复完成，交接exit0。最终仅核对实际diff、语法、两槽派发边界及level2不能自动升级，均通过；未重复运行pytest或GPU测试，未扩展科学预算。证据scheduler_r4/two_slots/delivery.json及scheduler.diff。

## 9. 两小时后自动接续计划（r5 handoff）

### 9.1 触发条件与当前基线

本节是对当前已授权 FCE r4 周期的无人值守接续，不新增模型、任务、fold、seed、PT 或候选预算。当前实况以 `results/fce_param_20261014/r2/status.json` 和 `results/fce_param_20261014/r2/scheduler_r4/status.json` 为准；最近核实为 2026-10-14 14:05 UTC：阶段 `FT:H_LR`，FT 45/140，epochs 3252/9600，attempted updates 74174/205800，正式预测 0/70，GPU2/3 各 2 路，失败 0，估计剩余 FT 约 1.81 小时。

唯一触发器是 `scripts/run_fce_param_r4.py` 的当前控制器在 `Uni-Poly:fce_param_two_slots_controller` 中完成本周期 FT 队列，或其 `completion.json`／`campaign_failure.json` 明确出现。不得按墙钟时间直接抢跑；“约两小时”只是观察窗口，不是完成判据。

### 9.2 正常自动链路

1. 继续使用现有 `results/fce_param_20261014/r2/`、`logs/fce_param_20261014/r2/`、当前 recipe、parent、head、scaler、split 和两路上限。已成功单元跳过，在途单元自然结束；不重启、不重训、不改变 GPU2/3 各 2 路限制。
2. 只有四臂 R0、H_LR、B_LR、S_COS 的 140 个 FT 单元全部以可信 `exit=0` 且必需产物齐全时，运行既有 validation-only 选择：读取 `validation_predictions.npz`，按五折 task mean 后七任务等权 macro R² 选唯一候选；禁止读取任何候选 TEST 结果，禁止事后换候选。
3. 选择冻结成功后，立即执行且只执行 R0 与唯一候选的 70 个正式预测（各七任务五折），复用冻结的 best、recipe、scaler、split 和 test indices。正式预测完成后运行尚未完成的唯一最终 CPU 检查：`CUDA_VISIBLE_DEVICES='' timeout 300s python -m pytest -q tests/test_fce_param.py`；此前已通过的 scheduler 窄测不重复运行。
4. 将原始精度的 candidate−R0 macro TEST R²、七任务逐项结果、论文参考对照、fold/seed 口径、并发与吞吐证据写入 `results/fce_param_20261014/r2/final_review/`，并由控制器追加 `CURRENT_PLAN.md` 最终记录。正式有效只按既有门槛 Δmacro TEST R² ≥ 0.010 判定；未满足则记录“本轮未证实有效”，不追加搜索。
5. 控制器完成后状态保持“待审查”（即使 CPU 测试通过）；独立审查模型 diff、日志、冻结文件、70 个预测和唯一测试结果后，才可把结论归档到 `CURRENT_PLAN.md`／`PROJECT_HISTORY.md`。不得把控制器退出或训练完成写成实验已审查。

### 9.3 失败与阻断分支

- 若出现 `campaign_failure.json`、任何 FT 失败、必需文件缺失、schema／parent／head／scaler／split 不匹配、NaN/Inf、writer 冲突或预算异常：停止受影响链路，保留现场，在 `CURRENT_PLAN.md` 标记“阻断”或“需返修”，不得减少 fold、换 seed、静默重跑或进入正式预测。
- 若控制器在 FT 完成后退出但没有生成 selection，先核对唯一锁、进程和账本；只允许恢复同一控制器的未完成步骤，不得重新初始化或重新派发成功单元。
- 若已有可信 `completion.json`，只读复用其状态和产物，不重做正式预测或 CPU 测试。
- 本节完成前不启动 §5 的后续参数空间。后续 dropout、WD、分模块 LR 或 PT schedule 仍需新的明确授权和独立预算。

### 9.4 交给执行模型的验收清单

执行模型必须按顺序报告：FT 四臂计数和失败数；validation 选定候选及冻结文件；R0／候选 70 个正式预测计数；七任务与 macro R² 原始差值及三位小数展示；最终 CPU 测试命令、exit 和日志路径；`CURRENT_PLAN.md` 状态；独立审查是否完成。任何一项缺失都只能报告“未完成／不可判定”，不能宣称性能提升或自动进入下一轮实验。

## FCE r4 实际完成与只读收口核对（2026-10-15，北京时间）

状态：**待审查**。本段依据当前 `completion.json`、`selection_before_test.json`、正式预测、metrics、日志和 Xc 诊断补录实际证据；未启动 controller、训练、正式预测或已通过测试。`completion.json` 的 `independent_review` 仍为 `NOT_PERFORMED`，本段只读核对不等同于归档审查或部署。

### 执行计数与触发链路

- FT 四臂 R0/H_LR/B_LR/S_COS：140/140；失败 0；epochs 9600；attempted/completed updates 205800/205800。
- validation-only 冻结：`results/fce_param_20261014/r2/selection_before_test.json`，候选 **B_LR**；validation macro R²（未舍入）R0 0.808423、H_LR 0.809283、B_LR 0.814683、S_COS 0.807612。冻结文件标记 `frozen=true`，正式两臂为 `test_arms=[R0,B_LR]`；没有读取未选候选 TEST 进行改选。
- 正式预测：R0 35/35、B_LR 35/35，共 70/70；FT 四臂未选候选无正式 TEST 目录。
- `results/fce_param_20261014/r2/campaign_failure.json` 不存在；当前无 fce controller 或训练进程，`controller.lock` 保留历史 PID 记录但无活动 writer。

### 正式结果（原始值用于判定，展示保留三位）

|任务|R0 TEST R²|B_LR TEST R²|B_LR−R0|
|---|---:|---:|---:|
|Eea|0.927|0.921|-0.007|
|Egb|0.940|0.929|-0.010|
|Egc|0.911|0.911|-0.000|
|Ei|0.790|0.798|+0.008|
|EPS|0.766|0.764|-0.002|
|Nc|0.820|0.846|+0.025|
|Xc|0.305|0.234|-0.071|
|macro|0.780|0.772|-0.008|

原始 macro 差值为 -0.008234759742148379，`effective=false`，因此 Δmacro R² ≥ 0.010 **未满足**；本轮未证实有效，不能表述为性能提升。逐折 R²、RMSE、原始差值和论文参考字段见 `results/fce_param_20261014/r2/final_review/seven_task_results.json` 与 `delivery.md`。论文参考为 Eea 0.934、Egb 0.923、Egc 0.911、Ei 0.839、EPS 0.787、Nc 0.856、Xc 0.423；本项目与论文数据规模、split/scaler 等价性未核实，且共享 TEST 参与开发，只作数值对照，不宣称公平比较。

### 产物、绑定与 Xc 核对

只读核对了 140 个 FT 目录和 70 个正式目录：必需 `metrics.json`、`best.pt`、`scaler.json`、`progress.json`、`updates.json`、`validation_predictions.npz` 及正式 `test_predictions.npz` 均存在；FT/formal 日志 exit 文件分别为 140/140、70/70 且全部为 0。预测数组和 metrics 数值均为有限值；未发现 NaN/Inf、缺失产物或 writer 冲突。正式 `indices` 与冻结 selection 一致，R0/B_LR 的 parent、head_initial、best、scaler、split、sample binding 复用冻结记录，未发现绑定不一致。具体路径为 `results/fce_param_20261014/r2/finetune/`、`formal/`、`logs/fce_param_20261014/r2/`。

Xc 只读诊断见 `results/fce_param_20261014/r2/final_review/xc_diagnostic.md`：R0 validation/TEST 均值为 0.411035/0.304905，B_LR 为 0.449093/0.233707；B_LR validation−TEST gap 为 -0.215386，R0 为 -0.106130。B_LR 在 4/5 folds 的 TEST R² 低于 validation，R0 亦为 4/5；五折 TEST 差值为 -0.1695、-0.0758、+0.0245、+0.1489、+0.1103。现有证据支持泛化/校准不稳定，未支持预测文件错配或 scaler/split 错误归因。

### 测试与后续边界

最终 CPU 测试为 `CUDA_VISIBLE_DEVICES='' timeout 300s python -m pytest -q tests/test_fce_param.py`，exit 0，8 passed、1 warning；日志为 `logs/fce_param_20261014/r2/final_cpu/{command.json,execution.json,stdout.log,stderr.log,exit.txt}`。此前 scheduler 窄测仍只复用 `logs/fce_param_20261014/r2/final_cpu_pidrepair1/` 的 exit 0 证据，未重复运行。

本周期保持“待审查”；不启动 dropout、WD、分模块 LR、预训练或新的正式 TEST 周期。任何新周期需先获得明确授权并写清 scientific question、reference、controlled change、comparison budget、validation-only 规则、独立评估协议和停止条件。

### 独立审查记录（2026-10-15，北京时间）

本次按当前产物进行一次只读独立审查，未启动 controller、训练、正式预测或已通过的 CPU 测试。`completion.json` 存在且状态为“待审查”；`campaign_failure.json` 不存在；controller.lock 中的历史 PID 1670439 已退出，当前未发现 FCE controller、训练进程或 formal writer。工作树的其他改动未触碰。

- 计数与预算：FT 四臂 140/140、9600/9600 epochs、attempted/completed updates 205800/205800、失败 0；正式预测 R0 35/35 与 B_LR 35/35，共 70/70；CPU 测试累计 2 次，最后一次 `tests/test_fce_param.py` exit 0（8 passed、1 warning）。
- 选择与绑定：`selection_before_test.json` 显示 140 个 validation 单元，四臂各 35，`frozen=true`，唯一候选 B_LR，正式臂仅 R0/B_LR；未选 H_LR、S_COS 没有正式 TEST。逐单元检查了 parent、split、source 和 runtime revision；正式预测 indices 与冻结记录一致，R0/B_LR 的 indices、sample_keys、y_true、scaler_mean、scaler_scale 逐 fold 一致。
- 产物与数值：140 个 FT 目录与 70 个 formal 目录的必需文件齐全，日志 exit 文件为 140/140 与 70/70 个 0；预测数组均为有限值。70 份 formal `metrics.json` 的 R²/RMSE 与 `test_predictions.npz` 原数组重算一致，状态均为 PASS；未发现 NaN/Inf、schema／parent／head／scaler／split 不匹配、writer 冲突或覆盖风险。
- 正式判定：R0 macro R² 0.7798443470663116，B_LR 0.7716095873241632，原始差值 -0.008234759742148379，`effective=false`；因此本轮未证实有效，不能写成性能提升或部署完成。七任务论文参考对照及 fold／seed 口径沿用本节前述 `seven_task_results.json` 与 `delivery.md`，仅作共享 TEST 开发协议下的数值对照。
- Xc：R0 validation/TEST 均值 0.411035/0.304905，B_LR 0.449093/0.233707；B_LR gap -0.215386，R0 gap -0.106130，二者均有 4/5 folds 的 TEST 低于 validation。逐折预测分布、残差、scaler、样本数和 sample key 见 `final_review/xc_diagnostic.md`；现有证据支持泛化／校准不稳定，不支持文件错配或 scaler／split 错误归因。

审查结论：状态仍为**待审查**，本周期结果可供用户审阅但尚未归档为模型历史或部署结果。下一步仅允许只读收口；dropout、WD、分模块 LR、预训练及新的正式 TEST 周期均需新的明确用户授权，并预先写明 scientific question、reference、controlled change、comparison budget、validation-only 规则、独立评估协议和停止条件。

## FCE 下一步执行计划 N1（2026-10-15，待授权）

### 计划状态与目标

状态：**待授权**。本节只定义下一周期，不自动启动训练、正式预测或预训练。

scientific question：在独立 outer holdout 下，降低 joint head learning rate 是否能减小解冻阶段的更新冲击并改善跨折泛化？

reference：同一新 outer split、同一 parent、同一运行版本下的 R0 原配方。当前 FCE r4 的 B_LR TEST 结果只作为历史背景，不用于事后选择 N1 候选。

controlled change：只将 joint head LR 从 1e-3 改为 3e-4（H_LR）；不改变预训练 parent、backbone LR、scheduler、weight decay、dropout、epoch、batch、seed 规则、模型结构或数据语义。

### 评估协议与预算

1. 在任何 N1 训练前，先固定一个未参与 FCE r4 选择的新 outer holdout／split manifest，并记录生成规则、样本计数和任务覆盖。若无法建立独立 holdout，N1 不得声称确认性结果，只能退回探索性方案并重新授权。
2. N1 重新训练 R0 与 H_LR 两臂，七任务各五折；不得复用 FCE r4 的 H_LR checkpoint 作为 N1 的新候选结果。每臂 35 个 FT 单元，共 70 个；每臂上限 2400 epochs、51450 attempted updates，共 4800 epochs、102900 attempted updates。
3. 每折仍按 validation RMSE 选择 best；在正式 TEST 前冻结 R0 与 H_LR 的 recipe、parent、head、scaler、split、best checkpoint 和 test indices。正式预测只执行两臂共 70 折，不允许看 TEST 后换候选。
4. N1 的确认性有效门槛沿用 Δmacro TEST R² ≥ 0.010，但必须同时报告七任务变化、fold/seed 口径、独立 holdout 规则及与 FCE r4 协议差异。若使用共享旧 TEST，则有效性只能标为探索性，不能替换部署。
5. N1 不含 PT、额外 seed、ensemble、dropout/WD/分模块 LR 组合或第二候选；失败单元计入预算，不减 fold、不静默重跑。

### 实施边界与停止条件

- 先完成 split manifest 和 N1 recipe 的只读审查，再等待明确启动授权；split、parent、head、scaler、schema 或 runtime revision 任一不匹配时停止。
- 训练和正式预测必须使用 tmux `Uni-Poly` 独立窗口；GPU2、GPU3 每卡最多 2 路、总并发最多 4 路，沿用 batch24、FP32、workers0 和既有资源边界。
- 出现 NaN/Inf、writer 冲突、checkpoint 覆盖风险、测试泄漏或预算超出时，停止受影响运行并保留现场。
- N1 完成后只提交结果和独立审查材料；未通过 Δmacro R² 门槛时不得追加 H_LR 组合、回看其他候选 TEST 或启动预训练。

### N1 验收交付

必须交付：新 split manifest 及协议说明；R0/H_LR 的 70 个 FT 结果和 70 个正式预测；validation 冻结文件；七任务 R²/RMSE、macro R² 原始差值和三位小数展示；独立 holdout 与旧 FCE r4 的差异；最终 CPU 测试真实 exit；日志、checkpoint、预测路径；独立审查结论。所有新训练和正式预测均以新的明确授权为前提。

### N1 第一阶段只读审查（2026-10-15，北京时间）

本阶段未启动训练、正式预测或测试，也未读取 TODO.md 或以 PLAN.MD 作为计划来源。r4 现有完成产物保持不变：`completion.json` 为“待审查”，FT 140/140、正式预测 70/70、失败 0；冻结候选为 B_LR；`campaign_failure.json` 不存在；当前无 FCE controller 或训练进程。

#### r4 产物与 N1 recipe 核对

- 140 个 r4 FT 单元的 parent 均为 `results/glt_periodic_seam_20261009/r1/pretrain/F_CE/deploy_05000.pt`，head initial、scaler、best checkpoint 均存在；runtime revision 均为 `fce_param_r2_cache_global_loop_v1`，seed 规则为 `42+fold`，batch 24、FP32、workers 0，数据语义为 `train_only_standard`。
- N1 的 R0/H_LR 控制变量已由 r4 recipe 明确：R0 joint head LR 为 1e-3，H_LR 为 3e-4；两者共用 parent、backbone LR、scheduler、WD、dropout、head 阶段、结构和 seed 规则。按任务保留既有差异：Egc 为 50 joint epochs、WD 0、head dropout 0；Xc 为 backbone LR 2e-4、WD 1e-4、head dropout 0.3；其余任务为 60 joint epochs、backbone LR 1e-4、WD 0.02、head dropout 0.1。预算仍是每臂最多 2400 epochs／51450 attempted updates，两臂合计最多 4800／102900。

#### 独立 outer holdout 判定

检查对象为 `data/raw/smi_{eea,egb,egc,ei,eps,nc,xc}.csv`、`data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging/{task}.jsonl` 和 `data/splits/mips_paper5_outer/{task}.json`。七任务计数如下：Eea 368、Egb 561、Egc 3380、Ei 370、EPS 382、Nc 382、Xc 432。每个现有 manifest 的五折 train／validation／test 并集均覆盖该任务的全部样本，未使用行数均为 0；fold0 计数分别为 Eea 235/59/74、Egb 358/90/113、Egc 2163/541/676、Ei 236/60/74、EPS 244/61/77、Nc 244/61/77、Xc 276/69/87。

因此无法从当前七任务数据建立同时满足以下条件的新 manifest：样本未参与 FCE r4 的训练、validation 或历史共享 TEST 开发；外层 holdout 与训练和候选选择完全隔离；仍覆盖七任务并支持五折比较。换一个随机种子或从同一全量行重新切分，只会产生重叠样本，不能被记录为独立 outer holdout。当前没有可交付的有效 N1 split manifest，也没有写入一个会造成错误独立性声明的候选 manifest。

结论：N1 保持**待授权**，但确认性执行被数据边界阻断；在获得新的、真正未参与 r4 开发的外部／新增样本及其固定 split manifest 前，N1 **只能进行探索性周期**。不得启动第二阶段训练。若用户明确接受探索性协议，必须另行授权，并在结果中明确标记不能替换部署、不能宣称独立确认性有效；若要求确认性结论，则需先提供新的独立 outer holdout 数据或等价评估协议。
