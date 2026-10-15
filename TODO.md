+ 多模块距离
+ 预训练任务优化
+ 2D/3D 融合模块
+ 2D/3D 是否需要对齐
+ 标签放缩
+ hop 值调整

+ 基线模型：F_CE、UA_2D




# 当前执行：FCE-PARAM-20261014-01 / r4（r2科学产物接续）

状态：**执行中**。授权来源：用户2026-10-14本轮明确要求完整执行PLAN.MD / r2及§7无人值守安排，授权最小实现、四臂完整实验、汇总及必要工程修复，12小时不是停止期限。

当前仅执行FCE参数周期；下方旧GLT r10已取消的状态与证据保持，不恢复旧队列。实际检查无旧python训练／控制器，tmux Uni-Poly可用；GPU2空闲，GPU3派发前检查，不操作GPU0、1或旧进程。

基线：原四rank F_CE step5000 parent；公共B0 1024维head；匹配新R0。scientific question：joint head LR、backbone LR、单段cosine三个单因素是否改善泛化。reference为新R0，历史F_CE仅额外数值参照。四臂同一运行版本，原七任务五折、seed42+fold、split、train-only scaler、batch24、FP32、workers0、任务dropout／WD、head10及joint60（Egc50）；global loop与冻结head cache统一启用。

允许新增本周期薄recipe接线、入口、控制器、选优汇总和窄CPU测试，保存本周期日志／checkpoint／预测及修改本文。不修改共享encoder结构、原数据、旧产物、SOTA_RESULT、部署或旧队列；不读TODO、不委派、不commit/push、不做hash验证、PT、新seed、后续搜索、阶段smoke或benchmark。

硬预算：FT≤140、epochs≤9600、optimizer attempted updates≤205800（失败计入），正式预测≤70。R0→H_LR→B_LR→S_COS，每臂35；全部完成后按各折validation RMSE best的R²，先五折均值再七任务等权均值，候选最高优先H_LR/B_LR/S_COS；冻结后仅R0和候选各35正式预测。TEST不参与选优。唯一有效判据为完整正式候选宏R²−R0宏R²原精度≥0.010，无逐任务退化否决。

GPU仅物理2、3，单元CUDA_VISIBLE_DEVICES=2或3、内部cuda:0，每卡一个总并发≤2。长任务在Uni-Poly独立window，资源占用等待。产物results/fce_param_20261014/r2/，日志logs/fce_param_20261014/r2/；controller window fce_param_r2_controller。持续记录真实exit、预算、已完成／失败／运行／等待与基于正常耗时ETA。成功不重跑；失败现场保留，只有完整可信状态可精确恢复；无此状态且预算不足则阻断该单元、继续不受影响矩阵，不以缺折判有效。schema／身份／split不匹配、NaN/Inf、writer冲突、覆盖或预算风险停止受影响运行。

最终一次CPU测试≤300秒、空CUDA_VISIBLE_DEVICES，覆盖optimizer配方、scheduler、默认路径、严格恢复绑定、validation选优与TEST隔离；必要修复仅受影响项一次复验≤300秒，累计最多2次。当前测试0、FT0、epochs0、updates0、预测0；尚不可估计ETA。独立审查未进行，最终须标待审查，不归档模型历史。

执行记录与最终七任务表随后追加。下方为旧周期交接全文，保留原授权、取消及事故证据。

---

# GLT-OPTIMIZATION-20261011-01 / r10

修订日期：2026-10-14。状态：**已取消**（用户于北京时间20:42要求“请停止执行”；本轮控制器、PT与worker已退出，停止位置与证据见§16，已有日志与产物保留）。

当前指令：用户要求停止执行，覆盖§14／15的自动继续与自主恢复授权；没有新的恢复指令，不启动训练、微调、正式预测或后续周期。

历史执行授权（已被本次停止指令覆盖）：用户明确要求使用物理GPU2、3完整执行本计划，问题由Codex自行修复、无需再次授权（§15），覆盖初始E0范围及逐周期等待授权限制。执行E1、A1、A2、P1、F2、F3、G0、G1、Q的已定义最小实现与固定实验，保留科学硬预算；P2未定义、预算0。初始E0授权与事故范围保留于§12／13。不修改原数据、旧checkpoint／正式预测、PROJECT_HISTORY.md或SOTA_RESULT.md，不读取TODO.md。

本文件替代 r9 的长交接；八个已执行周期及事故／审查状态见 [精简历史](PROJECT_HISTORY.md)。完整旧结果／配置／日志继续保留。S3、N1 与旧 FT-ROBUSTNESS／质量周期的审查收口仍待处理；S2审查来源是用户报告，不将其记为本轮独立审查。S5本对话独立审查已通过，结束时待审查产物保留为原快照。

## 1. 当前参考、目标与决策依据

|参考|用途／状态|宏 TEST R²|已保存证据|
|---|---|---:|---|
|历史 F_CE|研究性能参考，原四 rank PT；保持|0.786|[配置](results/glt_periodic_seam_20261009/r1/initialization/F_CE.json)／[parent](results/glt_periodic_seam_20261009/r1/pretrain/F_CE/deploy_05000.pt)|
|U0_FUSED|原 F_CE parent＋统一 FT 的历史机制参考|0.787|[U1四臂结果](results/glt_optimization_20261011/r4/U1/final_review/recovery1/seven_task_results.json)|
|UA_2D|3D辅助PT、只用二维预测的候选；未晋级|0.788|[U1执行交接](results/glt_optimization_20261011/r4/U1/final_review/recovery1/delivery.md)|
|F_CE_2R|两 rank 无 torsion 匹配参考；不替换历史基线|0.767|[S5结果](results/glt_optimization_20261011/r9/S5/final_review/seven_task_results.json)|
|T_FCE|两 rank torsion 候选；不替换部署|0.781|[S4结果](results/glt_optimization_20261011/r6/S4/final_review/seven_task_results.json)|

F_CE保留 O8／Galformer 各六层、512维，finite_incident Trimer、angle_value、incident readout、全图 atom-to-bond token fusion、fused atom CE，最终两个512维图表示 concat为1024维。它包含 token fusion 与图级读出两层融合语义。历史 F_CE 无 ALIGN、torsion、MD、endpoint或路径dropout。

现有证据优先支持继续研究二维可用性，并减少工程冗余。U1中 ALIGN−无ALIGN 的二维宏差为 +0.010，融合宏差为 −0.016；S5中 torsion匹配宏差 +0.014，但Xc validation五折均负、EPS／Xc两折贡献约83%宏差，稳定性未确认。局部融合、分路径dropout、F1乘性交互均未达到原性能门槛；不按任务改配方或拼接候选。

目标分开验收：E0／E1解决恢复、输入复用与计算成本；科学周期检验唯一目标／模块／几何因素。工程改动不自动意味着R²提升，候选开发数值不等于独立泛化证据。

## 2. 执行顺序与授权边界

建议顺序：**E0 → E1 → A1 → 独立审查 → 单独选择一个后续科学周期**。E0／E1结束后固定运行版本；任何周期不混用优化前后代码或中途替换实现。不能复用旧运行条件时，在本计划已列预算内构造同版本参考，不自行补训或增加预算。

|周期|唯一问题|优先级／依赖|状态|
|---|---|---|---|
|E0|reader target／身份复用、严格构造、独立二维模型接口|repair1单次CPU复验通过，首次失败保留，尚未独立审查|待审查|
|E1|跨注意力批量计算与 head-only 特征缓存|修复复验及GPU有界对照见§15|待审查|
|A1|二维 ALIGN 下 fused CE 与 raw O8 CE 的差别|按用户请求停止；FUSED完成5000更新、RAW完成92更新，完整周期未完成|已取消|
|A2|双向 ALIGN 与仅二维anchor方向的差别|未启动；按用户请求停止本轮执行|已取消|
|P1|5000更新预算内走完整LR schedule是否更合适|未启动；按用户请求停止本轮执行|已取消|
|F2|固定零初态FiLM残差是否改善原融合FT|未启动；按用户请求停止本轮执行|已取消|
|F3|两图token attention是否改善原融合FT|未启动；按用户请求停止本轮执行|已取消|
|G0|torsion分支对第二个PT随机流是否仍有收益|未启动；按用户请求停止本轮执行|已取消|
|G1|有限非键距离输入是否有益|未启动；按用户请求停止本轮执行|已取消|
|Q|保持覆盖的几何质量权重是否有益|未启动；按用户请求停止本轮执行|已取消|
|P2|更长PT或图级读出目标|保留研究问题，尚无可直接执行实验|待授权，预算0|

以上是队列，不是整条流水线授权。每次仅执行用户明确授权的一个完整周期；状态由实际证据更新。默认Codex规划／审查、ZCode执行；用户指定Codex实现时以其指令为准，不委派代理、不并行写同一文件／产物。

## 3. 所有科学周期共用的固定条件

### 3.1 数据、初始化与资源

- PT source：`data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging`；索引：[原train索引](results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json)，train 911391，沿用 overlap排除；不重新抽池。
- PT公共初态：[原 F_CE step0](results/glt_periodic_seam_20261009/r1/initialization/F_CE_step0.pt)，完整恢复公共encoder及原PT heads；所有科学候选从step0训练，不从旧step5000继续。新增模块独立CPU初始化并保存，不能推进公共RNG。
- PT只用物理GPU2、3，`CUDA_VISIBLE_DEVICES=2,3`，两rank、内部`cuda:LOCAL_RANK`；micro84×accum6×world2＝global1008，BF16，prep_workers8/rank，clip1。
- 默认seed42、AdamW lr2e-4／WD0、warmup2000、cosine总20000、end_lr1e-9、固定stop5000；mask0.30、noise0.03、原CE＋length-angle权重[1,1,0]。P1只改变声明的schedule字段；G0只改变两条件共同的PT seed。
- PT过程validation固定step1000／2500／5000，每次1024、两rank各512，共3072；恢复model模式和RNG，不选parent或早停，无padding、bank、queue、额外validation遍历。
- 下游 source：`data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging`；split：`data/splits/mips_paper5_outer/<task>.json`；固定 Eea／Egb／Egc／Ei／EPS／Nc／Xc各5折。
- FT／预测只用物理GPU2、3，每单元单卡、`CUDA_VISIBLE_DEVICES=2`或`3`、内部`cuda:0`；每卡最多一个GPU单元，总并发最多2。PT与FT不并行。资源占用时等待，不换卡或终止其他任务。
- 融合head复用`results/glt_fusion_20261007/r1/finetune/B0/<task>/foldN/head_initial.pt`；二维head复用`results/glt_optimization_20261011/r4/U1/initialization/heads_2d/<task>/foldN.pt`。不同输入宽度的head不混用，不重新抽初始化。

### 3.2 全任务统一FT与预算单位

统一`uniform_u1`：seed42＋fold、batch24、FP32、workers0、MSE、train-only StandardScaler（含Xc）、head10＋joint60、无early stopping；encoder dropout0、head dropout0.1、modality/path dropout0。AdamW全组WD0.02，head LR3e-4／1e-3、joint backbone LR1e-4、clip5；head Cosine(T_max10,eta_min3e-5)，joint WarmRestarts(T_0=10,T_mult1,eta_min1e-6)，每epoch step；原单位validation RMSE选best。

每个完整arm为35 FT／2450 epochs（head350、joint2100）／56000 updates／35正式预测。七任务updates依次3500、5250、31850、3500、3850、3850、4200。一个默认PT为5000 updates／5040000训练呈现／3次过程validation共3072；这些是新周期拟议上限，当前消耗0。

新增包明确绑定计划／周期、架构revision、输入与预测路径、CE入口／ALIGN方向、world／micro／accum／global、运行版本、source／索引、common step0、head、parent step及scaler。原U1 revision只支持其旧fused CE／双向契约；A1／A2必须新增严格revision，不能放宽旧契约或用strict=False丢键。

## 4. E0：输入复用、严格构造与独立二维接口

|scientific question／工程问题|reference|controlled change|comparison budget|
|---|---|---|---|
|消除重复解析与BRICS计算，同时保持训练输入|现有reader／prepare|直接传递同条记录已有target／identity|固定最多16个既有记录、相同seed／position；无训练|
|恢复正确模型并提供纯二维接口|现有严格包加载与UA_2D路径|小型metadata构造函数及直接O8／norm2恢复|固定已保存包与二维输入；无新参数或训练|

当前`DirectMCLSource.validated_pretrain_fields_for`已能返回六字段，包含target与已解析identity；`pretrain_source_fields`已有include_identity接口。热路径仍用include_target=False和五字段，prepare随后重新解析及调用chemical_groups（LRU512，命中时可避免完整BRICS重算）；优化应复用这些已存在接口。

E0三个子项：

1. **E0-R**：inline、worker和过程validation传入已有target／identity；prepare接收validated identity。保留BRICS组及组内顺序、canonical编号、样本流、motif mask、noise和all-copies masking，不改源数据、不生成缓存、不重复读取record。字段不合法或身份／schema不符时停止；仅旧来源确实无缓存时沿用原计算路径并记录来源。
2. **E0-M**：metadata明确恢复fusion／geometry revision、geometry_mode、angle_value、readout、连接／endpoint／torsion／ALIGN设置；先正确构造再strict-load。仅支持已存在且可核实的包类型；legacy缺省只适用于旧schema，未知revision／新包缺字段拒绝。先接通实际FT入口，不授权LoRA／ridge／head搜索，不建立注册框架。
3. **E0-2D**：直接恢复O8／norm2和512维head；可从符合原canonical schema的纯二维topology运行，无须构造Galformer或提供几何字段。保持原原子特征、pooling、模型输出及身份；旧direct仓库若仍需整record解码，明确其I/O限制，不声称已完成独立二维数据仓库。无需派生全量2D数据或写正式导出模型，导出／恢复样例仅在最终测试临时目录。

最小修改范围：`src/training/glt_dual_runtime.py`、`src/dataset/glt_dual_pretrain.py`、`src/dataset/mcl_ph_direct.py`、`src/modules/glt_dual.py`、`src/modules/glt_dual_pretrain.py`及现存PT／过程validation／FT入口中的必要调用；一个窄测试文件。共享O8／Galformer层、损失定义及旧包不改。

最终一次CPU pytest≤300秒、空CUDA_VISIBLE_DEVICES；最多16个既有record，覆盖旧／新groups及mask/noisy输入／labels一致、cached路径不再调用重复BRICS或解析、inline／worker样本方法传递（不启动worker）、缓存异常、严格包拒绝、纯二维CPU输出与旧UA_2D路径一致及无3D输入。PT／FT／GPU／worker启动／正式预测均0。只报告语义与复用行为，不承诺总训练加速。

## 5. E1：批量跨注意力与head-only缓存

|工程问题|reference|controlled change|comparison budget|
|---|---|---|---|
|减少逐图索引、GPU有效性读取与小kernel|当前BondTokenFusion|长度分桶批量attention，按需返回平均attention|固定权重、输入和连接模式，最终一次输出／梯度及计时对照|
|减少head-only重复编码|现有在线features|只缓存冻结且eval的encoder特征|相同head初态、输入、标签与shuffle／dropout RNG，无optimizer更新|

保留图隔离、有效性、Q/K/V、4×32 heads、原FP32 score／softmax与缩放；padding和非候选项不能进入softmax。支持已有global／incident-only／ln2 bias；空集合／无效图按原fallback。F_CE不需要local loss时不保存平均attention，旧需要attention的接口继续显式请求。

head-only缓存支持实际512／1024维路径，只在encoder冻结且eval时使用；绑定parent、架构／运行版本、输入视图、dtype、样本索引，不存y、不共享scaler、不预编码TEST。joint开始立即失效；其他parent／模式不得复用。新科学周期正常缓存构建计入其原流程，不单独重复编码作验收；不建立跨周期永久缓存服务。

最小范围：`src/modules/glt_dual_opt.py`的BondTokenFusion、实际head-only adapter／FT调用；先按`src/modules/glt_adaptation.py`与`scripts/finetune_mcl_ph.py`的现有接口定位，必要薄接线；一个窄CPU测试和一个有界GPU对照入口。不改参数结构、PT／FT科学配置或实现通用attention框架。

全部实现完成后集中一次最终阶段：CPU pytest≤300秒；随后仅物理GPU3一次计时≤300秒，空闲时执行，不启动worker或optimizer。固定既有batch最多24个、每批≤24样本；每模式warmup4对＋测量8对旧／新fusion forward-backward，三模式共36对／72次，不额外warmup或追测。FP32输出／梯度容差预先固定atol1e-5、rtol1e-4，失败不得放宽。

记录batch token分布、显存、耗时／吞吐及容差；缓存生命周期／标签隔离／head RNG在同一次CPU suite覆盖，不构建全任务缓存。没有实测提速则不宣称加速；浮点归约变化不承诺bitwise或最终训练R²等同。E0／E1后固定版本；下列A1／P1／F2／G0自建同版本参考，旧结果只作历史对照。

## 6. A1／A2：二维ALIGN后续（首选性能路线）

### 6.1 A1：只改变原子CE入口

|scientific question|reference|controlled change|comparison budget|
|---|---|---|---|
|ALIGN下直接监督raw O8是否改善可独立用的二维表示？|新A1_FUSEDCE_2D|A1_RAWCE_2D仅将CE输入fused_atom_states改为raw atom_states|相同新两rank、step0／投影／样本／目标权重／FT与预测路径|

固定两个新PT＋两个纯二维FT arms，不把旧四rank UA_2D当匹配参考。两者均无torsion／endpoint／gate／MD／local／质量权重，保持原global token fusion模块及原geometry损失；ALIGN取融合前raw graph_2d与incident graph_3d。

沿用512→256→128投影、L2归一化、双向multi-positive InfoNCE、temperature0.10、weight0.10从step1到1000线性warmup；同身份为多正、无效几何排除、全局anchor归一化与跨rank可微gather。两个新投影按原独立seed314161初始化并共享step0，仅新增328448 PT参数；两条件FT纯二维参数仍为19262233。

两rank每次forward最多168个候选，accum6不形成1008候选或negative bank。原无效几何仍参与二维CE，不删除标签。RAW条件中token fusion缺少直接PT损失梯度是预期结果，不能悄悄加辅助目标补偿；其FT不使用fusion。FUSED条件继续保留原梯度路径。

最小范围：已有pretrainer CE／ALIGN revision必要接线、严格metadata／过程validation与新unit／controller／selector；复用E0纯二维构造与原训练评估，不改共享encoder层。最终一次CPU pytest≤300秒，覆盖新revision拒绝、共享初态／投影RNG、CE入口与梯度路径、多正／无效mask及全局计数、纯二维输入／恢复、完整冻结与预测身份。

硬预算：2PT／10000updates／10080000呈现；6PT validation／6144；70FT／4900epochs／112000updates；70正式预测。只回答CE入口对二维ALIGN的影响，不额外重跑融合FT或无ALIGN四臂矩阵。报告candidate−reference及validation逐折差；旧UA_2D仅历史数值参考。

### 6.2 A2：只改变ALIGN方向

参考固定A1_FUSEDCE_2D及其完整配方，不自动换成TEST较高的arm。候选保留二维anchor查询三维keys方向，去掉反方向；两端仍求导，不加入detach／teacher／EMA。双向参考按两个方向有效anchor平均；单向候选按该方向全局有效anchor平均，weight／temperature／候选池／其他损失保持一致。

此处“单向”指anchor方向，不表示三维编码器被冻结。允许范围仅ALIGN方向、归一化与新严格revision／入口；最终CPU一次≤300秒覆盖方向、两端梯度及DDP计数／冻结。拟议1PT／5000updates／5040000呈现；3validation／3072；35FT／2450epochs／56000updates／35正式预测。同版本参考不完整或配方改变则停止修订，不自动增训。

## 7. P1／P2：预训练配方后续

**P1**问题：固定5000更新预算，让cosine schedule在5000结束是否优于当前总20000、stop5000的配方？新建无ALIGN／无torsion原F_CE结构的`P1_BASE20K`与`P1_SCHED5K`；唯一差异为schedule_total_steps=20000／5000。warmup2000、stop5000、lr／end_lr、公共初态、seed42、数据／目标及统一融合FT均相同；不同时改变PT长度或几何输入。

两条件新建同运行版本参考，旧F_CE_2R只作历史参考。无新增参数；PT1000／2500／5000 validation仅诊断，两个parent均固定step5000，不选中途checkpoint。最小范围为新配置／严格绑定／必要scheduler接线及薄执行入口；最终CPU一次≤300秒覆盖LR边界、更新计数、数据及初始化控制、parent选择与恢复。预算与A1相同：2PT／10000updates，6validation／6144，70FT／4900epochs／112000updates，70正式预测。

**P2保留但预算0**：更长PT、其他warmup配方、图级三维读出目标。必须先写明唯一新增目标或训练长度、额外forward／参数／数据及匹配参考，单独授权；不能把更长PT称等预算结构收益。图级目标不能利用属性TEST标签，也不能与ALIGN方向、非键输入同时改变。本计划不安排schedule网格或自动续训。

## 8. F2／F3：其他图级融合模块

### 8.1 F2：固定零初态FiLM残差

|scientific question|reference|controlled change|comparison budget|
|---|---|---|---|
|显式条件调制是否改善原融合FT？|新F2_BASE|F2_FILM仅新增图级FiLM残差|同原F_CE step5000 parent／原1024维head／同版本统一FT，无新PT|

两arm均保留原全图token fusion与concat。先一次取得原norm2后的h₂及norm3且有效性mask后的h₃；h₂已含3D融合信息。固定MLP512→64→1024（GELU）输出γ／β，更新h₂为h₂＋γ(h₃)⊙h₂＋β(h₃)，再与h₃ concat。末层weight／bias零初态，独立CPU seed314179；按两层含bias结构预计新增99392参数，正常构造时记录真实数量，不另设探针。

原无效几何在整个残差输出后严格置零，包括bias，不detach有效通路、不改变输入encoded对象。原head不换、不加投影归一化／gate／新PT目标。head-only冻结新增模块，joint按backbone组训练。

同版本新建F2_BASE的35FT与候选35FT，避免E1改变归约后直接复用旧U0_FUSED当严格机制参考。预算PT0、70FT／4900epochs／112000updates、70正式预测；最终CPU一次≤300秒覆盖零初态输出／RNG、bias后mask、梯度／参数归组、strict恢复／完整冻结。新增容量属于候选整体，不宣称排除容量。

### 8.2 F3：图级两个token attention（后续替代）

F2审查后才决定是否授权。参考固定F2_BASE，候选把已归一化h₂／h₃作为两个token，做一次4heads×32维Q/K/V self-attention，128→512输出投影零初态，独立CPU seed314181；更新后仍concat1024。含bias Q/K/V及输出投影预计新增263040参数，无额外LayerNorm或FFN；无效图在两token整体更新输出后mask。每个query面对两个keys，不能退化为单key、恒为1的选择权重。

只改变该模块；无新PT，不叠加FiLM／ALIGN／torsion。拟议35FT／2450epochs／56000updates／35预测，最终CPU一次≤300秒覆盖两token语义、零初态、mask及严格恢复。参考必须同版本完整；不自动搜索head数、宽度或多个模块，不把F2／F3差直接解释为等容量机制对照。

## 9. G0／G1／Q：几何信息与质量后续

### 9.1 G0：固定第二个PT随机流的torsion匹配确认

保留此前对话S6建议，正式队列命名G0。固定PT seed43，两条件都从原公共step0恢复，共同保持原CE／length-angle／schedule及统一FT；FT仍seed42＋fold和原head，只考察PT随机流敏感性，不称全面独立初始化复现。

新建`G0_BASE`无torsion与`G0_TORSION`使用S4物理分支，保留原五个tensor／33473参数、cos φ／cos(2φ)、退化mask与初始化seed314167。双方使用同E0／E1版本。旧seed42的S4／S5只作历史对照；运行版本不同时不能直接合并为同协议多seed统计。

预算2PT／10000updates／10080000呈现，6validation／6144，70FT／4900epochs／112000updates／70预测；最终CPU一次≤300秒只覆盖新增seed／公共初态／成对配置、metadata与参考绑定，不重跑已验证的S4几何suite。报告validation／TEST全部折、宏贡献及相对历史参考，不能仅凭改善折或macro确认泛化；无自动第三seed。

### 9.2 G1：有限非键距离，作为torsion之外的独立问题

reference固定已完成同版本`P1_BASE20K`，候选保持其无ALIGN／无torsion配方，仅新增来自原有限Trimer坐标的非键距离输入。固定真实四重原子路径，其端点物理原子不相邻、拓扑最短路为3；按中央物理键及无向端点对去重，保留接缝image身份，禁止跨图或虚构无限几何。

对端点距离用16维Gaussian特征，中心固定为0至6 Å等间距、σ＝0.4 Å，距离不裁剪、超出中心范围不判无效；经16→64→512 MLP（末层零初态、独立CPU seed314173），按有效观测均值更新中央键状态，再进入原Galformer。不加空间cutoff／top-k／多构象，不增加非键距离预测目标；合法无关系样本保留原geometry_valid／标签，退化和无效距离显式mask。

最小范围为该输入／关系构造、小分支与严格metadata／执行入口；不重生成坐标或回写源数据。拟议1PT／5000updates／5040000呈现，3validation／3072，35FT／2450epochs／56000updates／35预测。最终CPU一次≤300秒覆盖纯二面角变化可区分、刚体不变、physical编号／接缝／去重、空关系mask、梯度与恢复。新增参数容量属于分支整体；参考缺失时停止，不自行加参考PT。

### 9.3 Q：保持覆盖的质量权重，继续暂缓

保留旧Q假设，改为统一FT、同版本P1_BASE20K参考，不再使用旧任务特定M_L。仅原所选构象converged权重1、有效未收敛权重0.5；原无效仍不参与geometry loss，CE／属性标签不变。几何loss按各图原L_geo的Σ(q_g L_geo,g)／Σq_g，保留全局DDP／累积窗口归一化，不能在图内归一化时把q抵消。

不得应用旧eligible严格筛选；读取可核实原QC字段，缺失或身份不符停止，不按文件名、长度或loss猜测。只有单独授权时才最多一次读取原source并写独立QC小sidecar，无坐标派生。拟议1PT／5000updates、3validation／3072、35FT／2450epochs／56000updates／35预测；最终CPU一次≤300秒覆盖覆盖不变、QC身份、权重／全局归一化及缺失异常。权重不保证更高质量几何有效，保持暂缓。

## 10. 拟议预算、路径及停止／交付

### 10.1 按周期独立预算（E0实耗见 §12；其余新增消耗0）

|周期|PT启动／updates|PT validation次／呈现|FT／epochs／updates|正式预测|最终一次验证|
|---|---:|---:|---:|---:|---|
|E0|0／0|0／0|0／0／0|0|CPU≤300秒，≤16既有record|
|E1|0／0|0／0|0／0／0|0|CPU≤300秒＋GPU3一次≤300秒，36对fusion对照|
|A1|2／10000|6／6144|70／4900／112000|70|CPU一次≤300秒|
|A2|1／5000|3／3072|35／2450／56000|35|CPU一次≤300秒|
|P1|2／10000|6／6144|70／4900／112000|70|CPU一次≤300秒|
|F2|0／0|0／0|70／4900／112000|70|CPU一次≤300秒|
|F3|0／0|0／0|35／2450／56000|35|CPU一次≤300秒|
|G0|2／10000|6／6144|70／4900／112000|70|CPU一次≤300秒|
|G1|1／5000|3／3072|35／2450／56000|35|CPU一次≤300秒|
|Q|1／5000|3／3072|35／2450／56000|35|CPU一次≤300秒|
|P2／其他未定组合|0／0|0／0|0／0／0|0|未安排|

不把此表相加作为已授权总预算，不转移周期额度。A2／F3／G1／Q依赖参考闭环且运行版本完全一致；不一致时停止修订。任何周期不新增task／fold、概率或结构网格、ensemble、重新构象优化、多构象、源样本过滤。旧局部结构与路径dropout组合继续保留，预算0。

拟议每周期独立产物`results/glt_optimization_20261011/r10/<cycle>/`、日志`logs/glt_optimization_20261011/r10/<cycle>/`；E0已按本轮授权创建，其余目录未创建。非空目录／旧正式产物不得覆盖，不自动改revision重试。所有GPU、worker、长命令及训练产物写入在tmux session `Uni-Poly`独立window，命名`glt_opt_r10_<cycle>_gpu23_<stage>`；记录command、cwd、environment、stdout／stderr、start／end及真实exit。tmux不可用则阻断。

### 10.2 执行与唯一最终测试

开始前读AGENTS／本计划、git status及相关diff，保护已有改动，只读实际运行依赖。每周期最小实现全部完成后直接执行完整已授权流程；无阶段测试、smoke、probe、预跑benchmark、旧suite回归或全仓测试。

科学周期全部FT结束后冻结完整best、config／运行版本、parent、head来源、scaler、best_epoch及test索引，再每best正式预测一次；不启动不完整矩阵。一次汇总后仅最终CPU pytest一次≤300秒、空CUDA_VISIBLE_DEVICES，覆盖本周期新正确性问题；已有证据能回答的不重测。E1独有的GPU对照只在同一最终阶段执行一次。

NaN／Inf、source／schema／checkpoint不符、writer冲突、覆盖风险、超预算或中断时停止受影响与依赖运行，保留现场及真实消耗；不静默丢异常、自动重试或扩大预算。最终测试失败／超时写需返修，保留唯一结果，不追加测试掩盖。完成后待审查；Codex独立审查复用实际diff／唯一测试／日志／预测，不重复实验，自检不替代独立审查。

### 10.3 判定、七任务论文对照与交接

每任务五折R² mean/std(ddof0)、原单位RMSE与配对差，七任务等权宏平均；展示统一3位小数并保留末尾零，统计／best／门槛使用完整精度。R²差正值较好，RMSE差负值较好。成对参考报告validation／TEST逐折差与宏贡献，不将五折当五个独立初始化，不删除不利折或按任务拼接。

机制结论以各周期匹配reference为准；历史开发门槛仍为候选宏R²−历史F_CE宏值严格>0.010，历史基值从已保存原精度结果读取，不能使用展示0.786判定。通过门槛也只形成候选建议，研究基线／部署I更换另需授权。共享TEST已用于开发，不能称盲测，不按其追加实验。

每周期CURRENT_PLAN审查结论与通过后的历史归档，列齐以下七任务最佳论文模型／参考R²、项目R²及项目减论文；没有完整正式结果写未完成／不可比较及原因。数据、split或统计协议不等价时只作数值对照。

|任务|论文最佳模型|论文参考R²|
|---|---|---:|
|Eea|Periodic-TDL|0.934|
|Egb|Periodic-TDL (+Chem, +RC)|0.923|
|Egc|Periodic-TDL (+Chem, +RC)|0.911|
|Ei|Periodic-TDL|0.839|
|EPS|Periodic-TDL|0.787|
|Nc|Periodic-TDL|0.856|
|Xc|Periodic-TDL|0.423|

交付实际修改文件／diff、唯一最终测试、完整七任务／论文数值对照、实耗、tmux／日志／parent／best／预测路径及工程／性能分别结论。科学周期结束即停止；候选形成后独立确认数据／协议另行规划授权，不在本计划扩大split、seed或预算。

## 11. 前次规划文档交接（历史记录）

实际修改仅CURRENT_PLAN.md重写与PROJECT_HISTORY.md精简追加，旧历史原文保留；没有删除代码／数据／checkpoint／日志／正式预测。r1—r9八周期的新结果与旧未关闭审查状态已迁移，不再混杂旧“待授权／尚未启动”快照。原未实施路线均在本修订保留或明确合并到对应研究问题。

本轮实现／PT／FT／正式预测／pytest／GPU／worker消耗均0；未pull／commit／push，不执行哈希／校验和验证，不委派代理。最终只集中检查一次本轮文档diff、Markdown格式、现有引用、结果格式及预算。检查结论以本轮交付为准；本次文档编制不授权E0或其他周期启动。

## 12. E0 原执行交接（2026-10-14首次失败快照；最新见§13）

### 12.1 状态、授权与基线

**需返修**。三个子项属于同一工程周期；最小实现与唯一最终CPU测试已执行，整体未通过，未形成“待审查／已完成”结论。用户明确要求失败停止、不追加测试，本轮保留失败版本和唯一结果；后续修复与复验需要另行授权，不自动进入E1、A1或其他周期。自检不替代独立审查；尚未独立审查。

授权来源：本轮用户明确授权GLT-OPTIMIZATION-20261011-01 / r10 / E0-R、E0-M、E0-2D最小实现、一个窄测试文件、一次CPU pytest≤300秒和本文件交接。允许／禁止修改、预算及停止条件按 §4 和用户指令执行。基线为开始时工作区实际内容，已有大量tracked／untracked改动，均保留；本轮前源文件快照在 `results/glt_optimization_20261011/r10/E0/baseline/`，既有tracked diff单独保存，不把既有改动计为本轮实现。

### 12.2 实际修改与实现层级

- `src/training/glt_dual_runtime.py`
- `src/dataset/glt_dual_pretrain.py`
- `src/dataset/mcl_ph_direct.py`
- `src/modules/glt_dual.py`
- `src/modules/glt_dual_pretrain.py`
- `scripts/pretrain_glt_dual.py`
- `scripts/evaluate_glt_pt_endpoint.py`
- `scripts/run_glt_align_2d_unit.py`
- `tests/test_glt_e0.py`
- `CURRENT_PLAN.md`：授权、执行状态及本节交接。

E0-R：新增 `prepare_source_sample`，一次 `pretrain_source_fields(..., include_target=True, include_identity=True)` 携带同条record已有static／target／validated identity，接通inline PT、RankMicrobatchStream样本方法、endpoint和过程validation。Direct reader原有身份、center mapping、geometry有效性校验保留，新增仅为已解析identity附加source／schema关联信息。prepare接收identity和已有BRICS groups，缓存路径无重复解析／BRICS；非direct旧来源无target时才计算，样本与collate labels记录 `preparation_source`／`brics_source`。

**E0-R尚未通过**：新增target键集合检查只允许 `brics_groups`，但实际旧direct record还含 `fingerprint_packed`。这属于本轮实现未覆盖实际旧缓存schema，不是源数据损坏。唯一测试首次使用合法旧target即在 `src/dataset/glt_dual_pretrain.py:557` 抛出 `ValueError: invalid pretrain target cache schema`；影响旧／新mask、noise、labels完整对照及worker／inline／过程validation传递测试的后续覆盖。未静默回退、删除字段、重写数据或追加测试。原groups与cached groups的前置四条记录对照已执行，但不能据此写整体语义等价通过。

E0-M：新增 `deployment_model_options`／`restore_deployment`，先读取fusion／geometry revision、geometry_mode、angle_value、readout、连接、endpoint、torsion、ALIGN metadata再构造，复用现有严格加载和U1／S4／S5原契约；名称／shape／dtype不符及未知revision拒绝。只读核实原F_CE包的旧metadata实际不含gate与新版ALIGN字段，以其精确旧键集合和原语义限定缺省值；新版缺字段拒绝。legacy缺省只应用明确legacy input_schema，不使用strict=False。四个实际包F_CE／UA／S4／S5的恢复及拒绝测试通过。接通U1实际fused／pure_2d构造；未扩展其他历史FT入口。`scripts/finetune_mcl_ph.py`本轮尝试接线后已恢复为开始时快照，最终无本轮增量。

E0-2D：新增 `Pure2DGLTModel`、`restore_pure_2d`、`strict_module_state`。模型只有原O8、norm2、512维head，`forward_topology`可接收符合原canonical字段契约的纯二维topology；复用既有二维样本构造／collate，不需geometry字段或构造Galformer。完整F_CE／U1包先按明确198-tensor名称／shape／FP32 schema核对，再按精确子模块键严格加载；其他提取架构拒绝，不任意过滤参数。已有UA包和原eea/fold0二维head的CPU输出与旧UA_2D路径bitwise一致，无3D构造测试通过；原pooling、head、标准化输出和外部train-only scaler不变。恢复样例仅在最终pytest临时目录写入，没有正式导出模型。

I/O限制：旧direct source仍通过 `record()` 解码整条record，包含Trimer／static／target；本轮没有建立独立二维数据仓库，也没有派生全量二维数据。发布器的 `select_record_fields` 本就不保存历史schema标签；新增检查接受原canonical显式标签或现有untagged direct字段契约，未知显式标签拒绝。

### 12.3 唯一最终测试与停止记录

```bash
CUDA_VISIBLE_DEVICES= timeout 300s python -m pytest -q tests/test_glt_e0.py
```

- 结果：**2 failed、8 passed、1 dependency deprecation warning**；pytest内部耗时 **12.68秒**，launcher墙钟 **14.079秒**，真实exit **1**，未超时。
- 北京时间：2026-10-14 **15:18:03.249—15:18:17.329**；机器可读日志保存UTC原始时间。
- 失败项：`test_existing_cached_groups_mask_noise_labels_and_no_reparse`、`test_inline_worker_method_and_process_validation_transfer`；共同原因是上述合法旧target的额外 `fingerprint_packed` 字段被拒绝。
- 通过项：reader同record字段复用／缺失target拒绝、错误缓存／identity／schema拒绝、四类实际包的metadata构造和严格恢复、明确legacy恢复和未知包拒绝、纯二维输出／无几何输入／无Galformer／临时恢复样例／scaler保留。
- 不把未走到的inline／过程validation完整对照、缓存命中后的完整prepare对照写成通过。8项通过不抵消2项失败；未修复后重测、未追加smoke／探针／旧suite／全仓测试。

实际预算：CPU pytest **1次／14.079秒墙钟**；既有record **4条**（source indices 0、1、2、3，每条解码一次），后续比较复用内存。PT **0**、过程PT validation实际运行 **0**、FT／optimizer updates **0**、正式预测 **0**、GPU命令及计算 **0**、worker启动 **0**、新数据派生／生产缓存／正式模型导出 **0**。另有CPU只读checkpoint metadata读取，不属于训练或性能实验。全部本轮模型／测试入口空 `CUDA_VISIBLE_DEVICES`。

### 12.4 日志、交接与审查结论

- tmux：`Uni-Poly:glt_opt_r10_E0_cpu_final`，独立window保留；cwd `/root/workspace/Uni-Poly-Plus-master`。
- 命令、environment、起止时间、真实exit：`logs/glt_optimization_20261011/r10/E0/execution.json`。
- stdout／stderr：同目录 `pytest.stdout.log`／`pytest.stderr.log`（stderr为空，失败详情在stdout）。
- 工程交接：`results/glt_optimization_20261011/r10/E0/`；本轮增量 `incremental.diff`、修改统计 `changed_files.json`、record预算 `record_budget.json`、交付 `delivery.md`、单次执行器 `run_final.py`。
- 本轮未读改TODO.md，未改PROJECT_HISTORY.md／SOTA_RESULT.md／共享O8与Galformer层／原数据／旧checkpoint／正式预测；无删除，无hash／校验和验证，无pull／commit／push，无代理。

执行方结论：**需返修**，不能作为已通过的E0运行版本交接。下一步应基于现有失败证据修正旧target schema兼容（保留合法旧字段，继续严格拒绝未知／损坏schema），另行授权最终复验；本轮不继续修改实现或消耗测试预算。

本周期仅工程实现／CPU自检，没有新性能实验，不比较R²、不宣称总训练加速。七任务均无本周期正式预测结果：

|任务|本周期项目R²|论文数值差|状态／原因|
|---|---|---|---|
|Eea|—|—|未完成／不可比较：E0未授权正式实验|
|Egb|—|—|未完成／不可比较：E0未授权正式实验|
|Egc|—|—|未完成／不可比较：E0未授权正式实验|
|Ei|—|—|未完成／不可比较：E0未授权正式实验|
|EPS|—|—|未完成／不可比较：E0未授权正式实验|
|Nc|—|—|未完成／不可比较：E0未授权正式实验|
|Xc|—|—|未完成／不可比较：E0未授权正式实验|

§10.3论文参考表保留，不将历史开发指标或本轮CPU输出当成本周期正式结果，也不宣称公平比较或超过论文。独立审查尚未进行，未更新模型历史或部署。


## 13. E0 repair1：兼容修复与新授权单次复验

授权来源：用户报告旧target被过严检查拒绝后，明确要求“请你自行执行，不需要交付”；本次由Codex修正该缺口并完成一次新的最终CPU复验，不整理额外交付报告。沿用用户“尝试使用更多CPU”的要求，PyTorch／OMP线程16、BLAS／MKL线程1，worker启动仍0，CUDA_VISIBLE_DEVICES为空。

状态：**待审查**（仅修复及CPU复验完成，独立审查未执行）。仅修改src/dataset/glt_dual_pretrain.py、tests/test_glt_e0.py和本文；允许旧target携带已知fingerprint_packed字段，保持字段及其内容、不将它作为输入或重算；继续要求brics_groups，拒绝未知字段、损坏分组与身份不匹配。现有BRICS缓存reader本就跳过该旧fingerprint字段，不新增其无关内容校验。最终suite同时比较旧含fingerprint和新仅BRICS两种缓存的mask／noise／labels、不重复解析或BRICS计算及原字段不被修改。其他metadata／纯二维实现不改。

新授权预算：最终CPU pytest一次≤300秒，仍仅tests/test_glt_e0.py，既有indices0—3四条record各解码一次后内存复用；新增PT／FT／optimizer／正式预测／GPU／worker均0。不运行阶段性测试、smoke、benchmark或旧suite；失败保留唯一结果，不追加重测，不启动E1或其他周期。原首次失败测试记录与原代码增量均不覆盖；新增现场与日志为results/glt_optimization_20261011/r10/E0/repair1/和logs/glt_optimization_20261011/r10/E0/repair1/，失败版本代码／测试／本文快照保存在repair1/before/。

tmux：Uni-Poly:glt_opt_r10_E0_repair1_cpu_final；唯一复验命令、cwd、environment、stdout／stderr、起止时间及真实exit保存到repair1日志。原测试1次与新授权复验分开记账；不得把原失败改写成通过。自检不替代独立审查；PROJECT_HISTORY.md、SOTA_RESULT.md、原样本／checkpoint／部署不改。


### 13.1 实际复验完成记录

唯一新授权复验已在上述tmux窗口执行：**10 passed、1 dependency deprecation warning、9.25秒，真实exit0**；launcher墙钟10.524秒，未超时。北京时间2026-10-14 15:34:26.097—15:34:36.622。原先两项失败现在通过；两种target schema的原groups／mask／noise／labels一致、prepare无重复解析或BRICS、字段保持不变、未知字段／错误分组仍拒绝，其他metadata／纯二维测试亦通过。

本次PyTorch CPU线程实际为16，既有record解码4条（原indices0—3），新worker／optimizer／GPU为0，PT／FT／正式预测仍0。累计首次失败测试1次＋本次单独授权复验1次，共2次测试、8次record解码，涉及同4条唯一记录；不覆盖原失败的execution.json／stdout／record_budget或incremental.diff，不把本次复验当作原测试未失败。

状态待审查；当前实现不能写独立审查通过或性能提升。仅维护本文与必要原始日志，没有额外交付报告；E1、A1或其他周期未启动，F_CE／部署I保持。原direct整record解码限制不变。


## 14. r10 整计划执行授权（2026-10-14）

用户明确要求“请你执行完成这个计划！！！CURRENT_PLAN.md”，覆盖此前各已定义周期待授权和每周期结束不自动启动的编制限制。本次由Codex按依赖顺序执行E1、A1、A2、P1、F2、F3、G0、G1及可核实原QC条件下的Q；P2／未定义组合预算仍0，不自行设计额外实验。每周期保留原硬预算及一次最终验证；异常停止受影响及依赖周期，不自动重试。科研结果不得改变固定候选或删除折；已完成E0不重测，自检与独立审查分开记录，不宣称新独立审查已完成。

本轮物理GPU2、3；E1最终对照仅GPU3。沿用更多CPU授权，最终CPU验证16线程，BLAS／MKL线程1；PT保持既定每rank8个prep worker，FT workers0，不因CPU资源扩数据／训练预算。E1开始前已读AGENTS／完整计划、status与相关实际入口，保留已有改动和E0事故证据；不读改TODO、不委派代理、不改旧产物、SOTA或部署。

E1专属新增路径results/glt_optimization_20261011/r10/E1/及logs/glt_optimization_20261011/r10/E1/，既有源码快照before/用于识别本次增量，不做哈希。E1实施后仅一次CPU≤300秒及一次GPU3≤300秒最终对照，GPU每模式4对warmup＋8对测量，共36对／72次fusion forward-backward，无worker或optimizer。其他周期尚未启动；真实阶段／预算随后追加。


E1启动准备记录：原artifact-local启动器缺少项目sys.path，在require_tmux导入前exit1；CPU测试／GPU对照均未启动，验证预算消耗0。原controller.stderr.log／exit及launcher_before_path_fix.py保留，已停止该启动器并修正导入路径。原验证单元尚未启动，按本条完整计划授权仅启动首次CPU／GPU验证，不重测任何已启动单元。新控制器日志controller_setup_fix.*不覆盖原错误证据。


### 14.1 E1本次最终验证记录

状态：**需返修**。源码默认仍loop，仅显式bucket路径采用分桶；不需要local loss的调用可跳过平均attention，旧接口保持。head-only特征在冻结eval时缓存，标签仍读原dataset，原sampler和RNG保留，joint前失效。CPU／GPU真实结果见logs/glt_optimization_20261011/r10/E1/final_cpu/及final_gpu/；预算见E1/consumption.json及gpu_consumption.json。GPU输入为24条既有Eea训练记录的真实图规模上的固定人工状态，8种batch，fusion-only测量；没有额外GPU编码或optimizer，不能当作端到端加速结论。没有独立审查，不覆盖旧产物，失败不追加验证。


### 14.2 E1最终CPU失败及最小夹具修正（未复验）

CPU唯一执行结果：**8 passed、2 failed、1 dependency deprecation warning，4.25秒、真实exit1**，launcher墙钟5.389秒，16 CPU线程、空CUDA_VISIBLE_DEVICES。失败仅在新增人工测试夹具把张量命名为Data.values，属性读取返回PyG BaseStorage.values方法，Linear因此TypeError；512／1024两项缓存测试在cache.build处中止，后续RNG／sampler／标签／梯度／失效完整检查未走到，不能宣称缓存通过。其余8项（四种attention／gate输出及全部梯度、空incidence与bias、输入不变及可省略attention、实际dual读出不变、缺失绑定拒绝）通过。

已保留原唯一失败stdout／stderr／execution、E1/failure.json、首次incremental.diff及failed测试版本test_glt_e1_failed_fixture.py；仅把人工夹具字段改名为e1_input，不改任何既有样本／标签／预算或shared实现以绕过失败。修正未测试，没有再次pytest或追加smoke，整体仍需返修，不作为已通过E1运行版本。GPU对照0（36对预算尚未消耗），PT0、FT0、optimizer0、worker0、正式预测0；CPU1次。

本条整计划授权仍记录为有效；A1／A2／P1／F2／F3／G0／G1／Q待执行，但依赖E1最终复验通过，当前均未实现／未启动。依据§10.2“最终测试失败／超时写需返修，保留唯一结果，不追加测试掩盖”，本次停止受影响及依赖阶段；另行授权恢复后只能执行明确的新复验预算，不把失败改写成通过。P2仍预算0，不虚构方案；F_CE／部署I、旧科学产物保持。

必要证据：logs/glt_optimization_20261011/r10/E1/final_cpu/stdout.log／execution.json、E1/failure.json／consumption.json。本轮没有额外交付报告，源码与测试现场保留；独立审查尚未执行。

## 15. 自主修复与整计划恢复授权（2026-10-14）

授权来源：用户明确要求“使用 GPU 2、3 执行计划，遇到问题自行解决，不需要我授权”。覆盖此前问题后等待再次授权与不自动恢复的限制。Codex继续全部已定义周期，必要修复和修复后的窄最终复验自行完成；每次失败现场、修复原因、真实exit与累计消耗分别保留，不覆盖原测试／checkpoint／预测。E1新修复复验一次≤300秒；GPU原36对上限不扩张。科学task／fold／seed／epoch／optimizer／预测总预算保持§10.1，不因恢复扩大；P2仍未定义、预算0。不可修复的外部依赖或实质数据不一致仍如实记录，不能伪造通过。

物理GPU仅2、3，PT两rank占两卡，FT／正式预测每卡一单元最多两路；不与PT并行。更多CPU仅用于既定prep worker和CPU线程，不扩样本。E0已有通过证据不重测；E1原失败及未复验修正保留，新日志results／logs的E1/repair1/不覆盖非空目录。后续周期最小实现完成后执行完整固定矩阵，最后集中本周期一次CPU正确性测试，失败后必要修复复验另记账。无阶段smoke／额外benchmark／哈希验证／代理，不读TODO，不改旧正式产物或部署。

E1修复结果：CPU 10 passed、1 warning、4.26秒，exit0；GPU3最终有界对照36/36对、72次fusion forward-backward，exit0。global／incident-only／ln2 bias计时速度比分别0.589／1.925／1.385；global分桶反而较慢，后续全图路线明确保留loop，仅启用省略无用平均attention和head-only缓存，运行版本固定glt_r10_e0_cache_global_loop_v1。fusion-only人工状态计时不外推端到端吞吐。新周期实现文件与实际入口见scripts/run_glt_r10.py、scripts/run_glt_r10_unit.py、scripts/select_glt_r10.py、src/modules/glt_r10.py、src/dataset/glt_r10.py；复用原PT、FT epoch、scaler及评估逻辑，旧metadata契约保留。

完整执行顺序固定E1→A1→A2→P1→F2→F3→G0→G1→Q；已定义新增科学总预算9 PT／45000 PT updates／45360000训练呈现、27次PT validation／27648呈现、420 FT／29400 epochs／672000 updates、420次正式预测，各周期独立完整冻结后预测，最后各自一次窄CPU测试。Q只从已完成raw selected QC ledger生成小权重／身份sidecar，复用原records身份行，无源record整池重读、坐标生成或旧eligible筛选。P2保留问题，未定义、预算0。

E1 repair2：接入真实FT后静态发现缓存默认PyG拼接会改变column标签形状，引起MSE广播；尚无科学训练消耗。cache只记录原collate的标签尾部shape、不存y，逐batch继续取原dataset标签。按本条自主修复授权，新增一次≤300秒窄CPU复验，仅覆盖真实标签shape／MSE／head梯度及受影响512／1024缓存生命周期；原融合测试／GPU计时不重跑。现场与真实结果保留logs/glt_optimization_20261011/r10/E1/repair2/final_cpu/；先完成该复验后再启动A1。

repair2窄复验完成：4 passed、2 warnings、4.91秒，真实exit0；其中一条warning来自人工flat标签对照，真实column标签形状／loss／梯度对照通过。旧首次失败、repair1及本次复验分别保留，不回写旧结论。整计划自动控制器已在Uni-Poly:glt_r10_campaign_controller启动，日志logs/glt_optimization_20261011/r10/campaign/controller/；依次执行各周期固定矩阵，不需要新的用户授权。完整运行后自动更新本文、保存正式结果／预算／最终测试，独立审查仍另记。

A1初始化已完成、首个PT真实开始。启动后补齐Q专有诊断的加权raw numerator／浮点denominator及汇总中逐任务宏贡献；所有无QC条件（含正在运行A1）的训练算式、归约、参数与RNG不变。原启动source_used副本保留；该报告修订副本及逐文件diff在A1/source_reporting_revision1/，不把其写成A1 PT已加载的新源码。尚无FT或正式预测启动；不追加阶段测试。

2026-10-14北京时间17:17执行快照：A1两个公共step0／严格配置已保存，初始化真实exit0；A1_FUSEDCE_2D两rank PT已完成156/5000更新，下一更新进行中，未见运行错误。实际动态PT预算在A1/pretrain/A1_FUSEDCE_2D/consumption.json，逐rank训练日志在records_rank0.jsonl／records_rank1.jsonl；对应tmux为Uni-Poly:glt_r10_A1_pt_A1_FUSEDCE_2D。此时FT0、正式预测0；其余周期由同一控制器按顺序推进，不把启动写成完成。


E1 repair1执行完成：CPU／GPU真实exit均0，结果与累计预算分别见E1/repair1/execution.json和E1/gpu_comparison.json；旧失败保留，独立审查未执行。

A1：执行中。完整固定周期已启动；不做阶段测试。实际预算见results/glt_optimization_20261011/r10/A1/consumption.json。

A1：需返修。RuntimeError('controller interrupted by signal 15')；现场保留，后续依赖暂停。实际预算见results/glt_optimization_20261011/r10/A1/consumption.json。

## 16. 用户主动停止执行（2026-10-14）

最新指令：“请停止执行”。北京时间20:42:06向已核实身份的控制器PID1528609发送SIGTERM；控制器停止调度并向当前torchrun进程组发送SIGTERM。控制器于20:42:07.096退出，RAW预训练启动器于20:42:07.017退出，真实exit均为1。控制器、torchrun、两rank与prep worker已退出；停止后进程核对无本轮剩余进程，GPU2、3的compute-apps查询为空。本次为用户主动中止，不自动修复或重启，也不启动A2或其他周期。

停止位置：A1_FUSEDCE_2D已完成5000／5000更新、3次PT过程validation／3072条呈现，预训练真实exit0；deploy_05000.pt与resume_05000.pt保留。A1_RAWCE_2D完成92／5000更新，第93次已尝试但未完成；该条件没有保存训练checkpoint。累计PT启动2次、尝试5093更新、完成5092更新；微调、正式预测和A1最终CPU测试均0。A1完整对照周期未完成，没有本周期七任务正式结果，不能作性能比较。E0／E1已有复验证据仍保留，独立审查未执行。

原始execution.json与controller traceback真实记录FAILED／exit1，RAW的consumption.json停留在停止前RUNNING快照；均保留，不覆写成成功。控制器自动写入的“需返修”是SIGTERM中断处理结果，最新计划状态以本节的用户主动停止／已取消为准，不能据此自动恢复。

运行时tmux窗口：Uni-Poly:glt_r10_campaign_controller、Uni-Poly:glt_r10_A1_pt_A1_FUSEDCE_2D、Uni-Poly:glt_r10_A1_pt_A1_RAWCE_2D；停止后实际核对，这三个窗口已随启动器退出关闭，session Uni-Poly仍保留。命令与cwd、起止时间、真实exit及stdout／stderr位于logs/glt_optimization_20261011/r10/campaign/controller/与logs/glt_optimization_20261011/r10/A1/pt_A1_RAWCE_2D/；计数、checkpoint及中断现场位于results/glt_optimization_20261011/r10/A1/和campaign_failure.json。本次只停止运行并更新当前交接，不删除日志或产物、不改训练代码、不运行新测试／实验。


## FCE r4 最新调度授权（覆盖r3上限）

用户明确要求更新PLAN调度并自主接续，唯一范围第4节首轮四臂；旧GLT队列和第5节不执行。每卡目标4路，正常joint≥180秒及资源／吞吐支持才4→6→8，GPU2、3每卡≤8、总≤16；原batch／FP32／workers／OMP4／BLAS1／模型／RNG／runtime及全部r2产物不改。PT预算仍0，不恢复旧计划。预算自r2累计140 FT／9600 epochs／205800 attempted updates／70正式预测，TEST不改选。

旧r2控制器PID1643590已单独暂停派发，在途Egc两折自然运行；r3控制器仅准备未启动，本次沿用该安全接管现场。新scheduler_r4/仅记调度，不另开训练周期。PID归属无法核实不加槽；显存≥80%／MemAvailable<32 GiB／可比joint吞吐下降时停止高槽补充、自然降低一级。原一次最终CPU合并调度检查，无阶段测试或benchmark，不读TODO、不委派、不hash、不commit/push。独立审查仍未进行。


## FCE r4 接续授权与安全接管

用户明确授权按PLAN.MD r4仅接续第4节四臂周期、第6／7节。旧GLT及第5节预算0、不执行。仅调整调度，每卡目标4→6→8、最多8、全局最多16；batch24、FP32、workers0、OMP4、BLAS/MKL1、训练runtime／配置／初始化／种子均不变。沿用全部r2账本与目录，不重置140 FT／9600 epochs／205800 attempted updates／70预测。

旧FCE controller PID 1643590仅单独SIGSTOP停止派发，所有在途unit及启动器自然结束并落真实exit后，单独SIGTERM/SIGCONT使旧controller退出，真实exit -15（计划接管退出，不是训练失败）。未向unit或进程组发信号。接管证据scheduler_r4/takeover.json，新controller Uni-Poly:fce_param_r4_controller；唯一文件锁scheduler_r4/controller.lock。

资源边界：每次派发空闲显存≥12 GiB、主机可用内存≥32 GiB，陌生NVML进程拒绝派发；PID namespace映射不确定不增槽。每次升档后至少180秒正常joint；显存占用≥80%／可用RAM<32 GiB／可比joint吞吐恶化时降低一级，在途自然结束。最终一次CPU合并原recipe suite与调度模拟suite，总额度不新增。


## FCE r4 PID归属修复（用户“请解决／请继续”授权）

原因：NVML PID与容器PID不同，容器NSpid只包含当前命名空间编号；本环境未发现可用宿主proc映射源。旧策略把无法映射当作本周期任务不可调度，造成每卡一路。没有伪造映射或按数量推断归属。

本次最小修复改用两类独立证据：本地unit必须匹配启动记录中的完整argv、cwd、GPU字段、实际CUDA_VISIBLE_DEVICES以及stdout/stderr打开文件，记录PID和start_ticks；NVML继续提供整卡真实资源，未映射PID保留为unmatched_nvml_pids，nvml_mapping_verified=false。已有本周期可信在途unit时，允许在原显存／内存／并发边界内填槽，不再以无法取得的host PID映射作为硬前提。空卡接管若仍有NVML未知占用且没有本周期可信unit则拒绝；任一本地unit身份核实失败也拒绝新增。该方案不证明整卡没有其他容器作业，不将未知PID冒充本周期进程、不向其发信号；其他占用已包含在整卡空闲显存与负载中。此条覆盖旧“任意NVML PID无法映射都禁止增加槽位”的规定，其他科学预算及4→6→8边界不变。

仅暂停已核实r4控制器本身（PID1654602），所有在途unit自然完成；旧控制器及源码现场保留于results/fce_param_20261014/r2/scheduler_r4/pid_repair1/。修复仅涉及scripts/run_fce_param_r4.py、tests/test_fce_param_scheduler.py、PLAN.MD和CURRENT_PLAN.md，不改训练unit／recipe／模型。新控制器使用原scheduler_r4/controller.lock及原r2产物，累计预算不重置。

本次修复完成后一次窄最终CPU测试：tests/test_fce_param_scheduler.py，空CUDA_VISIBLE_DEVICES、timeout300秒、tmux Uni-Poly:fce_r2_final_cpu_pidrepair1。这是本周期两次CPU额度中的第一次；原全周期结束仅运行tests/test_fce_param.py，复用本次调度结果，不再重复调度suite。两次合计上限保持2，无额外训练／GPU探针／benchmark。真实结果后续写入。


PID归属修复已接管：调度CPU最终检查8 passed、1 warning、4.56秒、exit0；在途unit自然完成，旧r4控制器单独退出（exit -15，计划交接，非训练失败）。新控制器窗口Uni-Poly:fce_param_r4_pidrepair1_controller；日志logs/fce_param_20261014/r2/scheduler_r4/pid_repair1/controller/。仍只执行既有FCE矩阵，不恢复旧GLT。


## 用户紧急降载：每卡最多两路（2026-10-14）

用户明确要求降低过载，并指定GPU2、3各二路。本条覆盖此前4→6→8：每卡固定上限2，总上限4，不再自动升档；资源不足可少于2。旧控制器1666617已暂停派发，在途最多保留每卡2个活跃进程，多余进程如有则SIGSTOP保留内存状态，按原进度顺序SIGCONT接续，绝不重训。暂停进程仍占显存，不算活跃计算槽位。待旧队列自然完成后仅退出旧控制器，再启动两路调度。科学矩阵、累计预算、有效门槛不变。交接证据results/fce_param_20261014/r2/scheduler_r4/two_slots/，日志logs/fce_param_20261014/r2/scheduler_r4/two_slots/。

两路交接：旧队列已按每卡最多2路自然完成，暂停unit全部恢复并落真实exit；旧控制器计划退出。新控制器Uni-Poly:fce_param_two_slots_controller，累计预算继续沿用原r2账本。

<!-- FCE_R2_LIVE -->

FCE r4实时记录：2026-10-14T16:03:58.137229+00:00 UTC，阶段最终CPU测试；FT 140/140，epochs 9600/9600，attempted updates 205800/205800，正式预测 70/70；当前并发0，并发目标{2: 2, 3: 2}、等待原因{2: None, 3: None}，按真实近期吞吐估计剩余FT 尚不可估计。槽位／进程归属／显存／主机内存／重叠时段见scheduler_r4/status.json、progress.jsonl。


## FCE r4 全周期最终记录

状态：**待审查**。140 FT及70正式预测实际完成；最终CPU真实exit 0。独立审查未进行，未归档模型历史。累计预算：{"ft_starts": 140, "ft_completed": 140, "ft_epochs": 9600, "ft_attempted_updates": 205800, "ft_completed_updates": 205800, "ft_failed": 0, "formal_starts": 70, "formal_completed": 70, "cpu_test_starts": 2, "pt_starts": 0}。

候选：B_LR；状态：待审查（独立审查未进行）。
R0 macro R² 0.780；候选 0.772；差值 -0.008；本轮未证实有效。
原精度值与判定保存seven_task_results.json；先五折均值、后七任务等权均值，std ddof=0，seed42+fold。

|任务|R0 R²±std|候选R²±std|候选−R0|论文模型|论文R²|候选−论文|
|---|---:|---:|---:|---|---:|---:|
|eea|0.927±0.018|0.921±0.012|-0.007|Periodic-TDL|0.934|-0.013|
|egb|0.940±0.014|0.929±0.012|-0.010|Periodic-TDL (+Chem, +RC)|0.923|+0.006|
|egc|0.911±0.017|0.911±0.008|-0.000|Periodic-TDL (+Chem, +RC)|0.911|-0.000|
|ei|0.790±0.031|0.798±0.031|+0.008|Periodic-TDL|0.839|-0.041|
|eps|0.766±0.048|0.764±0.044|-0.002|Periodic-TDL|0.787|-0.023|
|nc|0.820±0.054|0.846±0.044|+0.025|Periodic-TDL|0.856|-0.010|
|xc|0.305±0.144|0.234±0.140|-0.071|Periodic-TDL|0.423|-0.189|

历史F_CE宏R² 0.786；候选−历史 -0.014；R0−历史 -0.006（仅额外数值参照，不用于有效门槛）。

共享TEST已用于历史开发，不是独立盲测；Egc／Xc数据规模与论文记录不同，精确split等价性未核实，仅数值对照，不宣称公平比较、多seed稳定或统计显著。
checkpoint：finetune/<arm>/<task>/foldN/best.pt；预测：formal/<arm>/<task>/foldN/test_predictions.npz；冻结selection_before_test.json；日志logs/fce_param_20261014/r2/。

## FCE r4 独立审查与后续计划（2026-10-15）

### 独立审查结论

本次审查只读复用实际日志、冻结文件、checkpoint metadata、预测文件和唯一最终 CPU 测试，不重跑训练、正式预测或 pytest。

- 执行完整性：`completion.json` 为 `待审查`；140/140 FT 单元完成，失败 0；70/70 正式预测完成；最终 `tests/test_fce_param.py` exit 0，日志为 `logs/fce_param_20261014/r2/final_cpu/`。
- validation 选优：`selection_before_test.json` 标记 `outer_test=NOT_RUN`、140 个 unit、四臂各 35 个；B_LR validation macro R²=0.814683，高于 H_LR=0.809283、R0=0.808423、S_COS=0.807612，选择逻辑符合预注册规则。
- 正式结果一致性：从 R0／B_LR 70 个 `test_predictions.npz` 的原始 `y_true/y_pred` 重算 R² 和 RMSE，与各 `metrics.json` 及 `seven_task_results.json` 在容差内一致；每个 task/fold 的 R0 与 B_LR `indices`、`sample_keys`、`y_true`、scaler 均一致；绑定的 parent、head、best 和 test indices 均存在，未发现 NaN/Inf。
- 科学结论：R0 macro TEST R²=0.779844，B_LR=0.771610，原精度差值=-0.008235；低于 Δmacro R²≥0.010，有效判定为 false。本轮应表述为“本轮未证实有效”，不归档为性能提升，不替换部署。
- 主要风险：B_LR 的 validation 宏提升没有外推到 TEST；Xc 从 R0 的 0.305 降至 0.234，差值 -0.071，且 Xc 的 TEST−validation gap 从 R0 的 -0.106 扩大到 B_LR 的 -0.215。共享 TEST 已参与历史开发，后续不能据此反复挑选候选或把 H_LR/S_COS 的未测试结果补成确认性结论。

审查结论：**产物与计算链路一致，科学有效性未通过；周期保持“待审查”直至本节记录被项目负责人确认，当前没有新的运行授权。**

### 下一步计划 N0（待授权，不自动启动）

1. **先完成周期收口**：保留现有 `results/fce_param_20261014/r2/`、`logs/fce_param_20261014/r2/`、冻结文件和失败现场；将 FCE r4 的七任务表、协议限制和未通过门槛纳入独立审查记录。不得删除、覆盖或重写历史指标。
2. **只读 Xc 稳定性诊断**：逐 fold 对照 R0/B_LR 的 validation 与 TEST best epoch、预测均值/方差、残差、scaler、样本数量和异常样本键；确认 Xc 下降来自泛化、折分波动还是数据/预测绑定问题。该阶段不读取其他候选 TEST，不训练，不改 split。
3. **预注册下一科学问题**：若 Xc 诊断未发现实现或产物错误，再选择一个单因素方向。优先研究 head LR 或 head dropout，暂不再降低 backbone LR，也不同时改预训练；固定 parent、数据、split、seed、epoch、batch 和评估协议。候选必须在正式 TEST 前冻结，不能根据本轮 B_LR 的 TEST 结果事后拼接。
4. **评估协议限制**：由于当前共享 TEST 已参与开发，下一周期若继续使用该 TEST，只能标为探索性数值对照；若要作确认性结论，应预先建立未参与选择的新 outer holdout 或等价独立评估协议，并重新定义 R0 与预算。
5. **授权边界**：N0 的文档收口和只读诊断可作为审查工作；任何新训练、正式预测、预训练、额外 seed、split 改动或部署替换都需要新的明确授权，不由本记录自动触发。

### N0 验收

必须交付 Xc 七折/五折诊断表、R0/B_LR validation 与 TEST 原始指标路径、是否发现实现错误的结论、下一候选的 scientific question/reference/controlled change/comparison budget，以及新的停止条件。缺少独立评估协议或诊断证据时，状态保持“待授权”，不得启动新实验。

### N0 只读诊断实际完成记录（2026-10-15）

已完成 `results/fce_param_20261014/r2/final_review/xc_diagnostic.md`。诊断读取 R0/B_LR 的 Xc 五折 validation／TEST 预测、metrics、scaler 和冻结绑定；未训练、未重跑正式预测、未运行新增测试。

结论：未发现 split／checkpoint／scaler／sample binding 或 NaN/Inf 问题。R0 Xc TEST R²=0.304905，B_LR=0.233707，差值=-0.071198；B_LR validation=0.449093、TEST=0.233707，validation→TEST gap=-0.215386；R0 gap=-0.106130。B_LR 在 4/5 fold 的 TEST R² 低于 validation，且 fold0、fold2、fold3 TEST R²分别为0.077、0.297、0.103。该证据支持“泛化不稳定”，不支持“预测产物错配”。

N0 已完成，后续仍为**待授权**：暂停新的超参数搜索、正式 TEST、预训练和部署替换；先按新的独立评估协议预注册单因素候选。若继续使用当前共享 TEST，只能报告探索性数值对照。
