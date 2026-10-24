# UA2D 优化总计划：微调、传播尺度与化学监督

> 计划 ID：UA2D-OPT-20261021-01 / r1，最新执行政策修订：r1-CD-REQUIRED。原周期状态：执行中；新增政策实现与控制器交接：待执行。本轮Codex仅修订计划，未修改运行源码或停止作业。
> 当前执行授权：用户本对话明确授权第1—12节完整有限计划及第13节执行提示词；原规划快照保留供追溯。实际执行／预算以第12节最新登记与本轮产物为准。
> 最新速度约束（2026-10-21）：用户要求“保证预训练的速度；需要在线计算的数据请先构建缓存”。确定性重型数据处理必须在唯一准备作业中完成并落盘，PT不得回退为在线构图／标签生成。细则见第7.1节，训练、评估及准备预算不增加。
> 最新性能判读（2026-10-21）：用户明确“判读是否有提升应该从TEST来看而不是Validate”。提升与性能通过仅据完整正式TEST判断；validation用于checkpoint、开发触发和主候选提名，不得因validation增量不足拒绝已预注册名单的正式TEST。第9节及第13节据此修订，PT／FT／TEST预算上限保持不变。
> C读取加速扩展（2026-10-21）：r1-C-IO1见第7.2节及第13.1节；其工程授权已登记，部署助手等待B3_C正常结束后的FT窗口。该扩展复用现有缓存，schedule和预算不变；是否部署／实际加载以真实记录为准。
> 最新用户变更（2026-10-21）：用户明确“请调整为无论B结果如何，都执行C、D”。C与D均改为必执行，不再受validation增量门槛或旧FT_POLICY=NONE阻止；只保留目标／parent／政策的validation选择。见第6节及第13.2节。此变更不豁免异常停止和预算期限，也不重跑已完成／在途单元；旧授权原文和旧决策产物保留为历史依据。
> 完整正常流程为A→B四臂→一个C→一个D→统一冻结／正式TEST→唯一最终CPU suite。当前从已执行阶段接续，禁止按完整入口从A重放。文末提示词交给DSH落实修订，计划更新不等于内存中的旧controller已采用新规则。

## 1. 版本、检查范围与交接边界

- 检查起于2026-10-21 07:48，核心状态快照截至07:52，Asia/Shanghai。目录 /root/workspace/Uni-Poly-Plus-master；分支 dev；HEAD 3503fbe780a5399410d6db6055305925765c3ff8。依据为该提交加现有 dirty 工作区，不能视为干净提交的验收。
- 已读本地 AGENTS.md、本文件覆盖前交接、SOTA_RESULT.md、相关实际 diff、O8／周期 topology／预训练与 FT 源码、初始化路径、正式结果汇总、最新 DEMOL／ET_SINGLE 交付及旧 HOP 审查记录。本次没有模型 forward、checkpoint 全量加载、测试、训练或 GPU 操作。
- 已有修改包括 AGENTS.md、CURRENT_PLAN.md、PROJECT_HISTORY.md、TODO.md、glt_dual.py、glt_dual_pretrain.py；既有文档及 result_contract.py 删除；大量 UA2D／HOP／ET 未跟踪文件。全部保护；没有读取 TODO.md，没有远程同步或新增哈希／校验和／文件指纹验证。
- 本轮仅修改 CURRENT_PLAN.md。新增实现、准备、PT、FT、TEST、CPU suite 消耗均为 0；新结果／日志根尚未创建。
- 当前未查询实际 GPU／rank／controller，也不据历史快照认定资源空闲。执行者须在获得授权后核对相关 writer／入口；无权启动、停止或恢复无关旧进程。
- 本文件覆盖旧长计划，将当前队列统一为本计划。旧 V1_CONTEXT、几何残差及 ET／DEMOL 旧提示词不再作为新派发入口；已有授权、事故消耗、产物和待审查状态保留，不清账，不追溯改写结果。PROJECT_HISTORY.md 中的旧节号属于历史记录，不是当前授权。
- 本轮是规划和证据摘要，不办理 DEMOL、ET_SINGLE、旧 PRETRAIN-FP 等周期的完整独立验收或历史归档。

## 2. 已有证据与科学目标

### 2.1 固定参考

| 身份 | 固定来源／用途 | validation 宏 R² | TEST 宏 R² |
|---|---|---:|---:|
| 工作基线 B_FP | fast_r4/B_FP；本轮下游基线与 A 阶段 parent | 0.818 | 0.791 |
| 配方参考 V1 | fast_r4/V1；元素恢复＋Galformer＋ALIGN，无 FP | 0.817 | 0.787 |
| 项目参考 r10 | G05 神经预测与 Morgan-KRR 按 0.75／0.25 融合 | 0.816 | 0.796 |
| 论文数值目标 | SOTA_RESULT 七任务最佳均值包络 | — | 0.810 |

- [V1／B_FP 正式结果](results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json)及原 finetune／formal／selection 全部只读。
- [r10 正式结果](results/fce_tune_20261016/global_blend/final_review/seven_task_results.json)，读取 macro.global.r2；不与其他同名 r10 混用。
- B_FP 被指定为工作基线不改变原 FP 周期未过增量门槛的结论。上述原周期尚未完成的独立审查状态继续保留。
- r10 含三维神经组件及核分支，本计划最终部署为纯二维模型；其差值只作项目数值参照，不单独归因某个模块。

### 2.2 最新历史交付，不恢复旧队列

| 周期 | 已读取的结果／消耗 | 当前记录状态 |
|---|---|---|
| V1_ATTN64 | validation 0.817／TEST 0.782；未提供优于 V1 的证据 | 待审查 |
| clean-FP 六臂 | 最高 validation 0.812／TEST 0.790；不替代 B_FP | 待审查 |
| HOP_EQ／HOP_PH | TEST 均约 0.788；PH 增量未达 0.010 | 已完成，独立审查通过，性能门槛未通过 |
| FROZEN3D | validation 0.821，对原 HOP_PH +0.002；TEST=0 | 已完成，独立审查通过，开发门槛未通过 |
| DEMOL-LITE | SELF／NEIGHBOR validation 0.820／0.819；NEIGHBOR−SELF -0.001，NEIGHBOR−B_FP +0.001；70 FT／4900 epochs／112000 updates，TEST=0，CPU 1 次 exit0 | 待审查，交付状态未完整独立核实 |
| ET_SINGLE repair1 | validation 0.815，对 V1 -0.002；累计 PT 2 次／attempted 5001／预算计呈现 5042352；35 FT／2450 epochs／56000 updates；TEST=0，CPU 累计2次 exit0 | 待审查，交付状态未完整独立核实 |

- [DEMOL 结果](results/ua2d_demol_20261020/r1/final_review/results.json)、[DEMOL 交付](results/ua2d_demol_20261020/r1/final_review/report.md)、[ET 结果](results/ua2d_et_single_20261020/r1/final_review/results.json)、[ET 交付](results/ua2d_et_single_20261020/r1/final_review/report.md)。前两者三位小数差值均来自完整精度，不能用展示值相减回算。
- DEMOL／ET 七任务正式结果均为“未完成／不可比较”，原因是 validation 门槛未通过，未开放 TEST；不以 validation 代填。
- ET 首次失败为 atomic/template physical count mismatch，发生于合法无几何记录的错误长度比较。repair1 获单次明确恢复授权；旧失败现场、incidents／attempt1、orphan_stop、原 rank 计数和保守呈现预留必须保留。
- ET 最新已登记用户要求“注意 3D 失败样本直接弃用！！！”；repair1 从 911391 条中剔除 635 条 geometry_invalid，保留 910756 条。新 PT 统一采用这份有效池，所有匹配臂一致整条排除失败三维记录；历史 V1 曾保留其二维监督，因此必须新增匹配 reference，不能将新旧差值单独归因 hop。
- 旧 HOP 是三维键线图末端多 hop＋几何／PH；本轮 hop 是二维原子 lifted 周期图传播半径。负结果不能跨对象直接外推。

### 2.3 核心问题与目标

当前下游为 O8 → 原子 mean → norm2 → 属性 head。O8 为六层、512维、8heads，每层显式关系最多2-hop；堆叠六层不等于独立的精确12-hop分支。

本计划依次回答：

1. 小样本 FT 中，限制更新范围或保留浅／中层表示，能否提高迁移？
2. 相同数据、初始化与预算下，2→3-hop 和元素→chemical-context 各自贡献多少，是否存在组合交互？
3. 相同目标与预算下，分层1／2／3-hop相对固定3-hop是否改善？
4. 将 A 阶段提名的 FT 配方迁移到新的 PT 主候选，是否仍有正式TEST增量？

开发后的模型采用单一全局政策，不按TEST或task拼装模型。r1-CD-REQUIRED取消C／D的0.005开发执行门槛：不论B增量或A旧门槛是否达标，都完成一个C和一个D；validation只选目标、parent、FT政策、checkpoint及FINAL。正式TEST宏R²差值>0才可报告本次观察到正增量，≥0.010才达到性能通过门槛；0<差值<0.010须写“正增量但未达门槛”，没有完整正式TEST则“未完成／不可比较”。论文包络宏平均及七项分别达标另行判断，不承诺SOTA。

机制动机分别见 [Jumping Knowledge](https://arxiv.org/abs/1806.03536)、[MixHop](https://proceedings.mlr.press/v97/abu-el-haija19a.html) 和 [Periodic-TDL 的 context 监督](https://arxiv.org/html/2605.26833v1)。本项目改造与其原方法不同，论文结果不代替本项目实验。

## 3. 固定数据、初始化与新产物身份

### 3.1 数据与身份

- PT 唯一原 source：data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging/；原 index：results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json。
- 旧只读 prepared_cache：results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache/。已读 metadata 为 COMPLETE、911391 train rows；这不代表逐条独立核验全部缓存。
- 有效池只读来源：results/ua2d_et_single_20261020/r1/sampling_repair1/eligible_indices.npy 及 metadata.json／excluded.jsonl。indices 是旧 prepared_cache 的 logical indices，不是可直接替代 source_row 的编号；必须经原 source_rows.npy／keys.npy 解析。metadata 为 910756 有效、635 剔除。
- PT 每臂采用同一 OrderedSampleStream(N=910756, seed42)，源顺序、噪声／腐蚀位置和样本身份一致。2D 与 3D 原随机流保留；新初始化使用独立 RNG，不移动既有流。预训练不读取下游标签，不做 PT validation 或 checkpoint 搜索，只使用 step5000。
- 身份错误、损坏、非有限几何不得伪装为无几何并筛掉；有效池内再次发现失败几何即停，不静默缩小池。A／FT 使用完整原二维 task 数据，不因缺少三维而删除下游记录。
- 下游 source：data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging/。任务仅 eea／egb／egc／ei／eps／nc／xc，fold0—4，seed42+fold，原 data/splits/mips_paper5_outer/{task}.json；不新增 split、task、fold、seed、样本或构象。

| Task | 样本数 | 每五折合计更新：每臂70epochs |
|---|---:|---:|
| Eea | 368 | 3500 |
| Egb | 561 | 5250 |
| Egc | 3380 | 31850 |
| Ei | 370 | 3500 |
| EPS | 382 | 3850 |
| Nc | 382 | 3850 |
| Xc | 432 | 4200 |
| 合计 | 5875 个任务记录 | 56000 |

每臂35单元、2450epochs／validations；每轮五折合计18786个train记录、4714个validation记录、5875个TEST记录。指标按任务／fold计算，不能把这些合计当作独立分子数。

### 3.2 初始化

- A 唯一 PT parent：results/ua2d_pretrain_fp_20261019/fast_r4/pretrain/B_FP/deploy.pt，原 r4／mask_align／FP enabled／step5000／attention512。下游只恢复 O8／norm2，丢弃 FP、3D 及预训练辅助头。
- B／C 均从共同 step0 开始，不从已训练 V1、B_FP、ET 或前一候选续训。只读公共初态为 results/ua2d_pretrain_fp_20261019/fast_r4/initialization/common_step0.pt 和 initialization/task_heads/V1.pt。
- 在本轮 initialization/ 中保存一次统一 max-hop3 的扩展 step0：同名同形状旧张量严格导入；SPD前三行、旧path投影和bond-position前两槽逐项复制，新增行／槽／projection用独立 seed20261021 初始化。所有 B／C 使用同一扩展初态；扩展代码不改变旧默认。
- 四臂二维 head 均为512→256→类别数；元素101类，context词表大小由固定PT-train决定。context两臂共享同一初态，元素两臂共享原元素head初态；类别数不同是目标变化的一部分，不声称等预训练辅助头参数量。
- 公共 FT head：results/glt_optimization_20261011/r4/U1/initialization/heads_2d/{task}/fold{fold}.pt，共35份。D 重新从选定 PT package＋相同原 head 开始，不能从其他 FT best 续训或继承 A 学到的权重。
- 本轮未加载大 checkpoint 独立验证其中全部张量；未来执行时在真实初始化／恢复路径严格检查 schema、形状和来源，不另跑前置探针。

### 3.3 产物根

- 固定结果根 results/ua2d_opt_20261021/r1/；固定日志根 logs/ua2d_opt_20261021/r1/。本轮检查时均不存在，不创建。
- 所有新缓存／词表／配置／公共初态／预算／决策／checkpoint 写入新根；旧结果与旧缓存只读。不得改根规避冲突、失败或累计预算。
- 模型完整schema记录 max_hops、六层hop schedule、目标、vocab路径、读出／系数定义、可训练范围、attention、parent／初始化、task／fold／seed、best阶段、scaler与完整state。禁止宽松忽略缺键或将max-hop3包送入旧max-hop2 loader。

## 4. 阶段 A：复用 B_FP 的两种下游优化

Reference 固定为旧 B_FP 对应35套FT validation，直接复用，不重训或重推理。原配方记为 A_NONE，无新增消耗。仅两个新增臂，不能组合两项变化。

| Arm | 唯一变化 | PT | FT |
|---|---|---:|---:|
| A_TOP2 | joint冻结embedding／全部bias模块／前四层，只训练最后两层、norm2、head | 0 | 35 |
| A_MIX246 | 原全量joint不变，在读出前加入第2／4／6层图表示的两个尺度系数 | 0 | 35 |

- A_TOP2：head阶段仍仅head；joint时 atom_embedding、SPD、path_bias、bond_bias、layers[0:4] 全部冻结并保持eval；layers[4:6]、norm2、head按原组LR训练。冻结模块不得被共享trainability helper再次解冻。
- A_MIX246：令 p2／p4／p6 为对应层原子mean；r = p6 + 0.250·tanh(a2)·(p2−p6) + 0.250·tanh(a4)·(p4−p6)，再进原norm2／head。a2=a4=0，因此初态等于原读出；两标量只在joint训练，按backbone LR／WD分组。原O8仍全量joint；不新增query、MLP、PH或3D融合。
- A_MIX246改变表示组合，不能叫“精确4／8／12-hop分支”，也不与旧query-readout实验混称。
- 两臂采用第8节原FT配方。70FT全部完成后，按完整精度validation macro在TOP2／MIX246中固定唯一D_FT_POLICY，精确同分优先TOP2，不再要求相对B_FP增量≥0.005。已核实A_MIX246优于A_TOP2，因此本次D_FT_POLICY固定MIX246，不根据B或TEST重选。
- 已有selection/a_policy.json中的FT_POLICY=NONE是旧执行门槛的真实历史结果，原样保留，不覆盖伪装为新决策。修订后的有效D政策另存selection/cd_required.json，并写入conditional_decisions.json的D.FT_POLICY；D的policy／config／strict restore须读有效政策，不能继续从旧a_policy.json取NONE。A本轮正式性能仍待TEST，不能据validation差值判断提升。
- A阶段不做阶段性正式TEST；若A被第9节提名，最终统一批次仍须正式评估，不另设validation增量门槛。

## 5. 阶段 B：2／3-hop × 元素／context 的匹配四臂

四臂都是完整PT＋FT，不以历史2-hop结果代替匹配reference。A选出的FT政策暂不带入，统一使用原完整joint／末层mean。

| Arm | 六层显式最大hop | 二维恢复目标 | 额外FP／图级任务 | PT／FT |
|---|---|---|---|---|
| B2_E | 2,2,2,2,2,2 | 原元素CE | 无 | 1×5000／35 |
| B3_E | 3,3,3,3,3,3 | 原元素CE | 无 | 1×5000／35 |
| B2_C | 2,2,2,2,2,2 | 一跳chemical-context CE | 无 | 1×5000／35 |
| B3_C | 3,3,3,3,3,3 | 同一chemical-context CE | 无 | 1×5000／35 |

### 5.1 hop语义与必要适配

- hop是无限lifted周期化学图上的最短键路径长度，不是Å距离、layer数量或canonical闭环最短路。保留(source atom, target atom, relative image shift)和不同物理image的重数；shift用于建立真实关系，不新增绝对胞编号输入。
- 统一从真实周期键模板生成≤3-hop关系，存路径宽度4／键化学宽度3；四臂使用相同节点顺序、原137维mips_x、backbone标记和graph_available，不能因max_hops重算而改变这些基础特征。
- B2臂每层仅保留SPD≤2，B3保留SPD≤3；self SPD0保留。必须对source／target／bias及对应路径行同步筛选后再attention softmax，不得只把bias置零或在归一化后将远边消息置零。
- 同步适配BFS、path_shift、SPD词表0—3、path节点projection四槽、bond-position三槽、collator、prepared cache、FT输入与恢复schema。旧固定2-hop的guards／宽度只在隔离新入口扩展，不修改旧默认。
- 较长路径必须保留旧最短路径选择规则。多个同距路径不在本轮改成另一种采样／全路径聚合，避免引入额外因素。
- 四臂注册相同max-hop3主干参数；实际使用槽位／梯度与计算量会随hop变化，不宣称等有效容量或等FLOPs。3D Galformer及其原2-hop line关系完全不变。
- 新hop chemistry必须来自可覆盖所需image的真实无限键模板；旧三聚体中无观测的路径不能伪造、环化或截断。输入语义不足时停止受影响准备，不生成新构象扩展三维覆盖。
- 记录实际relation数量、路径覆盖、吞吐、显存和时间。这些数据来自唯一真实执行，不预跑benchmark或额外模型forward。

### 5.2 context标签

- 用干净周期一跳邻接，为每个中心原子构造“中心元素＋按邻居元素／真实键型统计的计数”类别。元素索引遵循原101类约定；键型遵循现有5类；排序、分隔和计数规则固定并写入vocab metadata。
- 包含真实seam邻居；同一物理观察副本去重，不同物理邻居的多重性保留。周期自映射的不同image不能按canonical ID粗暴合并。
- 只用固定910756条PT-train建词表，不读取下游val／TEST来建词表、合并长尾或选阈值；保留train全部类别，不按下游结果调词表。首次确定性遍历次序固定，类别映射在PT前冻结。
- 预测位置与原atom_selected／40%及80/10/10动作一致；标签仅作监督，不写回受损输入。替换元素CE，不同时保留元素CE，不增加官能团／FP任务。
- 损失先分子内归约，再按optimizer-window全局有效分母归约，二维权重仍1.000；3D CE／ALIGN按第7节不变。辅助分类头容量变化必须报告。

### 5.3 预注册比较

令 V(arm) 为完整精度七任务validation macro；正式TEST开放后使用同样定义计算 T：

- 元素目标下的hop效应：V(B3_E)−V(B2_E)。
- context目标下的hop效应：V(B3_C)−V(B2_C)。
- 2-hop下的目标效应：V(B2_C)−V(B2_E)。
- 3-hop下的目标效应：V(B3_C)−V(B3_E)。
- 交互项：I = [V(B3_C)−V(B2_C)] − [V(B3_E)−V(B2_E)]。该差分来自匹配2×2消融；组合−baseline本身不代表interaction。
- 全部报告逐task／fold配对差值、五折均值与std；不凭单次seed、系数或训练loss宣称稳定机制。I为估计值，不自动等于有统计支持的交互。
- B2_E与旧V1的差异含有效池和统一路径模块变化；B2_E是本轮因子比较reference，不宣称重现旧V1或改变项目基线。
- B四臂完整140FT结束后一次记录selection/b_factorial.json；B阶段不看本轮正式TEST，不依据部分task停止或改臂。

## 6. 阶段 C、D：固定执行（r1-CD-REQUIRED）

### 6.1 C：必执行一个分层hop候选

- 在B3_E与B3_C中选validation macro更高者，精确同分优先E，目标记为U。
- 无论V(B3_U)−V(B2_U)为正、为零或为负，都执行唯一C_SCHED_U；0.005仅可作为历史规则的对照字段，不能参与新执行分支。conditional_decisions.json记录C.trigger=true、required=true及本次用户变更原因，保留实际未舍入差值，不伪造旧门槛通过。
- C唯一变化为六层hop schedule改成1,1,2,2,3,3；context或元素目标U、全部模块／参数／初态、有效池、5000 PT updates、原FT政策均与对应B3_U一致。逐层真正筛选attention关系。
- 从第3节共同step0重新PT，不从B3_U训练结束包继续。正常完整执行固定1PT＋35FT；仅执行E／C中已选的一个，不运行两个C。
- C在其首次PT启动前按第7.2节启用同一读取加速策略；IO修改不改变模型计算，也不增加候选。
- 对B3_U的差值仅解释传播顺序变化；C失败不追加另一目标、其他schedule或4-hop。C不做正式TEST。

### 6.2 D：将开发提名的FT政策迁移到新PT模型

- 从完整B四臂及必执行C中，按validation macro选择唯一PT_WINNER；精确同分固定次序B2_E、B3_E、B2_C、B3_C、C。即使全部低于历史baseline也必须选定一个，不以增量门槛跳过D。
- 固定执行D_INTEGRATE的35FT／0PT。PT parent为PT_WINNER的step5000；唯一FT变化是采用本次已固定的D_FT_POLICY=MIX246，其他FT条件与PT_WINNER原35FT一致。conditional_decisions.json记录D.trigger=true、required=true、FT_POLICY=MIX246及旧a_policy.json路径；旧NONE不再是跳过条件。
- 重新从该PT package和原35公共属性head初始化；MIX的a2／a4重置0，TOP2按政策冻结；不得继承A或PT_WINNER的FT best。
- D必须使用新PT package＋公共head重新FT；禁止以“复用PT_WINNER的FT结果”替代，也禁止将NONE作为新D政策导致原配方重复。
- D−PT_WINNER估计同一PT下的FT政策增量；与A之间的迁移不能被称为已有完整交互消融。
- D不追加TOP2＋MIX组合、task例外或第二次FT重试，不做阶段性TEST。
- C／D固定执行，validation只选择U、PT_WINNER和最终候选，不决定是否运行。正常完成后C、D均进入第9.2节预注册TEST名单；异常、产物不完整或期限耗尽仍按第11节停止，不因“必执行”而自动补跑、忽略失败或扩预算。

### 6.3 r1-CD-REQUIRED：实现范围与一次显式控制器交接

- 修订ID：UA2D-OPT-20261021-01 / r1-CD-REQUIRED。用户本条明确要求“请调整为无论B结果如何，都执行C、D”，覆盖旧C的0.005门槛及旧D的NONE跳过规则。Codex本轮完成规划修订，DSH按第13.2节落实必要实现与交接；本轮未修改源码、发送信号或派发训练。原停止条件／最高预算不变，“无论结果”指完整有限指标的高低，不授权忽略失败、NaN/Inf、产物不匹配或超时。
- 检查依据：北京时间2026-10-21 22:04—22:07，本地dev／3503fbe780a5399410d6db6055305925765c3ff8，加现有dirty工作区；已读AGENTS、SOTA_RESULT、相关diff及实际入口／决策／IO助手。A70FT、B前3臂PT＋105FT已完成，B3_C PT在途，C／D／TEST／最终CPU均未启动。主controller仍为Uni-Poly:@1545、PID1195532，B3_C为@1682；B3C-IO1已实际加载，C-IO1助手@1646等待部署。快照不代替部署时的实际核对，既有修改、事故和产物均保留。
- 科学问题与受控变化：B四臂匹配比较不变；C固定相同目标下的分层schedule，D固定相同PT parent下的MIX246读出政策。新增变化仅为取消执行门槛，并将C／D和直接reference预注册纳入唯一最终TEST。既有数据、split、seed、初始化、训练及评估口径不变；不增加第二个C目标或第二个D政策，不作额外搜索。
- 允许最小修改src/training/ua2d_opt.py、scripts/run_ua2d_opt.py及原最终tests/test_ua2d_opt.py的必要夹具，并新增本次一次性交接／证据辅助。保留原r1结果根、PLAN身份和已有checkpoint／cache／FT config的revision=r1；只追加execution_policy_revision等政策元数据，不能全局改旧config导致aggregate或strict restore拒绝存量结果。C-IO1的scripts/pretrain_ua2d_opt.py由原助手独占部署，本修订不与之同时修改；dataset、模型和缓存不改。
- 有效决策：保留旧a_decision／a_policy.json历史含义；新政策另写selection/cd_required.json，含用户依据、修订ID、C.required=true、D.required=true、D_FT_POLICY=MIX246、A winner／来源和旧NONE记录路径。C的U按第6.1节选择，conditional_decisions.json的C.trigger与D.trigger均为true，实际差值可保留但不再作为分支。D的policy／config读取该文件中的D.FT_POLICY=MIX246，不再读旧a_policy.json的NONE；parent／schedule仍由PT_WINNER派生，D必须从该PT package＋原公共head重新初始化，MIX标量重置0。
- 内存规则限制：当前controller.run、c_decision及a政策已在内存加载旧分支，仅改磁盘或计划不会令其执行新C／D。原controller_handoff_repair1只接受A阶段47单元且claim已使用，不能重用或改写该旧交接。必须实现本修订专用、一次显式交接入口，不新增通用自动resume／retry。
- 部署顺序：先让在途B3_C PT自然完成；等待B3C-IO1助手正常汇总退出，C-IO1助手在B3_C FT窗口完成预读代码增量并确认deployed_at，再进行controller交接。确认其已完成源码写入、主controller仍在B3_C的FT矩阵，且C／D／TEST／最终CPU均无启动记录／进程／输出冲突。若B3_C失败、窗口已错过或C等已启动，不按另一阶段静默接管、补训或重放；保留现场报告，由Codex修订。
- 交接时由DSH核对旧controller和C-IO1助手的实际PID／启动身份／命令／cwd，在旧dispatcher处于稳定FT矩阵等待时，只暂停其派发及C助手的监控循环，训练／FT／worker不发信号。冻结后核对每个已记启动的在途job都有实际window／wrapper／worker或真实退出证据，不能采纳记账后尚未实际launch的半派发状态；若发现该边界，恢复旧dispatcher完成同一原始派发后再安全冻结，不删条目、不重新启动训练。将授权、原budget／manifest／timing／作业身份及clock快照存入controller_handoff_cd_required/。旧dispatcher退出须避免进入其finally停止健康worker；允许本次记录充分的一次行政替换，只处置已核实的旧controller，不把行政退出计为训练失败，不信号整组或tmux session。交接失败保留现场，不自动重复接管。
- 新controller在Uni-Poly独立window取得原controller.lock的唯一写锁，使用controller_handoff_cd_required/request.json及独占一次claim，继承原全部jobs、已启动身份、成功／在途／真实exit、预算上限和wallclock_started_unix=1792541198.175118。更新manifest的新writer／命令／window及政策ID后再恢复C助手；暂停期间不得让助手误判controller退出。不用旧controller日志目录，新增logs/ua2d_opt_20261021/r1/controller_cd_required/，原@1545及其日志保留。
- 新运行入口必须从B3_C的未完成FT矩阵接续：采纳已完成／在途单元，只派发从未启动的剩余单元；不再执行A、准备、初始化、前3个B臂或B3_C PT。完整B3_C FT结束后从既有产物汇总A／B validation，执行固定C→汇总C并选PT_WINNER→固定D→冻结及第9.2节唯一TEST→原唯一CPU。失败／部分启动不重新排队，不能从完整旧run()入口重放。C助手只继续检查新conditional_decisions的trigger=true及实际加载，不接管派发／主预算；其旧报告模板中的条件措辞由新修订记录解释，不覆盖旧报告伪装历史。
- 新最终决策函数／冻结／交付须同步第9.2节：FINAL仍在原全局候选集合中按validation选择，C／D及直接reference固定纳入TEST；先按身份去重再检查≤6臂／210单元／35250图，不能把A追加到6臂集合使预算变245次。FINAL为A时使用第9.2节的最多5臂分支。首个TEST之前一次冻结全部名单；不得看TEST后追加C／D或改选FINAL。
- 正常剩余范围：当前B3_C仅完成原PT余步及35FT，新增C为1PT＋35FT，新增D为0PT＋35FT；全周期仍最多5PT／25000updates／25200000呈现、280FT／19600epochs／448000updates、210TEST／35250图和1CPU suite。仅GPU2／3，PT world2、micro84／rank、acc6、global1008、4workers／rank／prefetch2，FT／TEST每卡最多4路，原阶段期限不变；72小时截止仍为北京时间2026-10-24 08:06:38，不因交接重置。
- 验证与交付：本轮仅检查最终文档diff／格式。DSH仅静态语法／最小diff检查，不增加smoke／benchmark／阶段测试；把“负增量仍执行C、旧NONE仍用MIX的D、全局择优与两种TEST名单上限、一次交接保留账本／在途身份及不重放”等必要用例放入原唯一最终CPU suite，收尾一次、≤300秒。保存selection/cd_required.json、controller_handoff_cd_required/、实际配置／预测／命令／exit／预算、final_review/cd_required_report.md及独立的实际diff。C／D对B_FP和直接reference的正式差值与七任务SOTA数值对照据第9节报告，状态待审查，交Codex独立审查，不自行归档PROJECT_HISTORY。

## 7. 统一准备与 PT 配方

- 一次新输入准备作业，墙钟≤14400秒。最多准备910756条有效PT记录和5875个下游任务记录；保留原数据身份、source_row／sample_key及拆分，输入缓存不读取下游标签。仅构建hop3拓扑、context词表／标签及必需索引；旧prepared_cache的3D模板只读复用。
- 不重新生成FP／PH／坐标／构象／split，不扩大数据池，不安装或升级依赖。必要的多进程准备最多16 CPU workers，PT每rank最多4预取workers／每worker预取2；任务数固定，预取也计呈现预算。
- 每个B／C臂仅1次PT启动、5000 attempted/completed optimizer updates目标、最多5040000次源样本呈现；global batch1008，物理GPU2／3、world2、microbatch84／rank、accumulation6。单微批跨rank对比候选最多168，累积不会将其变为1008。
- BF16；attention logits、几何关键运算、loss／全局有效分母FP32。AdamW peak2e-4／weight_decay0／clip1；LR warmup500，余弦至step5000的1e-6。
- 二维CE权重1.000、三维原25755类线token CE权重1.000；原mask_align、fusion=none、fp_enabled=false、mask_ratio0.400、噪声0.030Å。保持原token周期组同步腐蚀、原正权重监督范围和分子归约。
- 图级双向InfoNCE与旧V1相同，保留现有双向求和约定；温度0.100，系数前1000updates线性升至0.100。QK缩放保持1/√512，结构bias不随之缩放。不加入ET、3D→2D融合、局部蒸馏或新的任务权重控制。
- 每臂PT硬期限18000秒，成功臂只用step5000；不选中途checkpoint，不加PT验证，不自动resume或减步数。
- 记录consumed／prepared／prefetch-in-flight及可核实下界。成功完整呈现最多5040000；失败也计已准备／已呈现记录，不把未落盘值写成精确0，不清账补额度。

### 7.1 缓存前置与训练速度硬约束（2026-10-21补充）

用户本次补充约束在本计划既有执行授权范围内生效，不另加准备启动次数、训练或测试。本节依据本地dev／3503fbe及当前实现中的准备、dataset、PT入口只读检查制定；当前执行登记和已有实现全部保留。本次只修改本计划，不与执行者同时修改训练源码或写训练产物。

**在PT启动前完成一次准备、冻结静态缓存，全部B／C臂共用，禁止每个epoch／实验臂重复生成。** 一次准备可含顺序扫描、词表冻结和数值编码子步骤，均计入原一次准备启动、910756条有效PT／5875个下游任务记录、14400秒和全程72小时预算，不能另开无预算缓存任务。允许一次准备内部读取已写的临时context签名以编码，不重新读取源生成第二套样本；旧3D模板只读复用，不重新生成坐标／构象。

| 必须准备期计算并持久化的数据 | PT读取要求 |
|---|---|
| normalized identity、logical index→source_row／sample_key、canonical／physical image映射、有效池与真实键模板 | 数值索引直接查找；不在线解析SMILES、RDKit规范化或源身份重建 |
| 周期BFS、最短路径／shift、SPD、边及路径顺序、三步14维键化学特征和padding mask | 直接读张量；不在线extend、构图、遍历RDKit键或重建路径 |
| ≤1／≤2／≤3-hop的关系行索引、相应source／target及路径索引视图、图内offset规则 | 直接按固定schedule选择预建视图；允许batch offset与张量gather，不逐样本重复nonzero／Python关系筛选或重新排序 |
| 一跳context计数／签名、train-only词表及每个原子的整数context类别ID | 签名和词表保留供追溯，热路径直接读整数labels；禁止在__getitem__的缓存miss或每次worker启动后逐原子做字符串→ID转换 |
| 干净元素／原25755类线token标签、周期腐蚀分组／group_first、物理端点／角度triplet／incidence索引 | 复用或数值缓存；不在线Python重分组、反复构造标签或搜索邻接 |
| 必需的干净坐标及训练无关静态几何字段、用于归约的固定图索引／有效标记 | 读取原模板或联合数值记录；不扫描原source重建，不在forward反复计算固定计数／映射 |

- 缓存采用批量读取友好的二进制张量记录或mmap／LMDB；保存dtype、shape、schema、source绑定、eligible顺序、vocab类别映射与覆盖计数。不保存依赖文本解析才能训练的主数据。context签名为可选追溯字段，整数标签才是训练入口。
- 优先准备可直接组batch的联合数值模板或已解析的数值sidecar，减少每次读取旧／新模板后的重复解码、字段重建和小文件IO；不能用进程内LRU代替持久化准备。若复用双缓存，身份join和字段映射须已固定，冷读不得重新计算拓扑或化学标签。
- 缓存由单writer在原准备作业内写入，完成覆盖及正常schema／范围检查后最后写COMPLETE；PT正常初始化只接受COMPLETE且schema／eligible／vocab相符的输入。缓存缺键、缺整数标签、视图不全或身份冲突即按停止条件报告，禁止在线fallback、边训练边补写或将异常样本跳过。上述为实际准备与加载逻辑，不新增前置probe／hash检查。
- **保留在线且矢量化的内容：** 按原seed／position生成的新噪声与80/10/10掩码、donor选择，以及依赖本次噪声的键长／键角；仅用已缓存端点与triplet索引计算。保留batch拼接／offset、必要张量gather、损失有效分母与模型forward。不得为提速冻结随机增强、复用同一份受损输入，或缓存依赖可学习参数的Gaussian／embedding／attention bias结果，否则会改变目标或梯度。
- 持久worker、按既定顺序的有界异步预取、批量二进制IPC、pinned memory与non_blocking H2D采用旧fast入口的可复用方式；仍最多每rank4workers／每worker预取2，限制在途内存。仅传实际目标必需字段，不能改变随机流、sample顺序、有效样本归约或预算计数。
- 正常训练中记录data_wait、H2D／compute（能低开销区分时）、step时间、graphs/s、缓存hit／miss、IO与峰值显存；每100个真实updates汇总一次即可，保留原必要消耗账本。不得为计时增加forward、每微批强制cuda.synchronize、额外GPU探针或预跑benchmark；不足的计时项标未分离，不伪造精确值。
- 将缓存生成耗时／体积、实际PT吞吐、data_wait占比及每臂≤18000秒的遵守情况写入交付。缓存保证确定性重计算移出热路径，实际速度依据真实执行报告，不预先声称已达到某个step耗时；瓶颈不得通过改batch、减步数、额外准备启动或放宽预算掩盖。

### 7.2 C读取加速扩展：r1-C-IO1（2026-10-21）

- 计划ID：UA2D-OPT-20261021-01 / r1-C-IO1。工程授权已于2026-10-21 19:54登记，@1646助手待执行／尚未部署；原r1科学周期仍执行中。Codex负责计划及独立审查，DSH负责实现、部署、监控与事实交付。最新r1-CD-REQUIRED只覆盖旧C执行门槛和D跳过规则；本节IO改动、部署顺序和预算继续保留。
- 检查快照：北京时间2026-10-21 19:40，本地dev／3503fbe780a5399410d6db6055305925765c3ff8，加既有dirty工作区。已核对AGENTS、SOTA_RESULT、相关diff、当前PT／dataset／controller、B3C-IO1暂存补丁及deploy_and_report.py。B2_C PT在途，B3_C和C尚未启动；B3C-IO1助手已准备但未部署。唯一数值缓存COMPLETE，含910756条PT和5875条下游记录、2423类context及1／2／3-hop关系视图；这些是检查快照，不代表未来实时状态。已有修改／删除／未跟踪文件均保留，未读TODO、未远程同步或执行哈希验证。
- 工程问题与reference：在不改变样本或模型计算的条件下，实际触发的C能否通过复用B3C-IO1的只读LMDB预读减少加载等待。reference为真实B3_C及既有B3_E／B2_C日志；它们的schedule、目标、运行时共享负载和缓存冷热可能不同，耗时只作工程观测，不构成隔离的速度因果比较。本方案不承诺提速倍数，不增加对照重跑。
- 唯一受控工程变化：仅C_SCHED_E／C_SCHED_C中按第6.1节选中的一个、且尚未启动的PT，继承两份只读LMDB的readahead=True和loader_wait_seconds／batch_decode_seconds计时。B3_C保留原B3C-IO1策略。已有低层read_policy字符串b3c_io1_readahead可兼容复用，无需为名称重写dataset；默认original和其余臂保持不变。C的engineering记录使用io_plan=UA2D-OPT-20261021-01/r1-C-IO1、io_scope=<实际C臂>_PT_only和实际读取参数，不能继续标成B3_C_PT_only。
- 数据与计算边界：复用当前统一max-hop3缓存及opt_hop1_rows／opt_hop2_rows／opt_hop3_rows，六层schedule仍为1,1,2,2,3,3；不改为独立1-hop缓存，不重建拓扑、context或任何缓存，不复制／全库预热。readonly、lock=False、bytes解码、事务生命周期、LRU、完整性检查及异常行为不变，不启用buffers=True。target、词表、样本身份／顺序、seed／position、腐蚀／噪声、共同step0、模型schema、损失、精度和优化算术均不变；从原E_step0或C_step0开始，不继承B3训练权重。
- 最小实现：基于B3C-IO1实际已部署版本，在scripts/pretrain_ua2d_opt.py统一定义显式适用集合B3_C、C_SCHED_E、C_SCHED_C，并使read_policy选择、engineering写入、loader／decode计时及timing输出的适用条件一致。B3_C原方案ID和历史报告保留，C标记本扩展ID。src/dataset/ua2d_opt.py只读复用已部署的策略接口；不改模型、FT、controller、候选选择或最终TEST逻辑。允许新增本工程的部署／汇总辅助与证据文件，辅助不得派发训练或写主预算。
- 部署顺序与单writer：保留Uni-Poly:@1645的原B3C-IO1助手、其staged／before和原报告，不修改其正在等待的源码快照，不在它部署前覆盖运行源码。待原B3C-IO1完成部署、B3_C PT正常exit0且rank／worker退出，利用B3_C的35FT在途窗口，由DSH基于当时实际源码准备并一次部署C增量；部署前再次确认无C启动记录／进程／输出目录，且无其他源码writer。此时IO1助手仅汇总其自身报告，不会再次写源码。若错过窗口、C已启动或已有异常，记录未生效／需返修，不热切换、不重启／恢复，不创建第二controller或绕过原停止规则。若实际前置状态与本快照不符，报告具体缺项，由Codex修订，不能静默另建执行通路。
- 选择与预算：按第6.1节在B3_E／B3_C选U，同分E，不再检查0.005，固定执行一个C；工程部署、实际加载与训练完成分开记录。C固定1次PT／5000updates／5040000呈现，加原35FT／2450epochs／56000updates，七任务fold0—4、seed42+fold不变；D按第6.2节必执行MIX246的35FT。本IO扩展额外准备、初始化、PT、FT、TEST及CPU suite预算均0，r1-CD-REQUIRED使用原预留C／D额度；原全程最高预算不增加，不能将旧NONE解释为当前D仍跳过。
- 资源与期限：仅GPU2／3，一个双卡PT、world2、micro84／rank、accumulation6、global batch1008；每rank4workers／每worker预取2，PT与FT／TEST不混跑，FT／TEST每卡最多4路。C PT仍≤18000秒，FT单元≤1800秒；72小时仍自北京时间2026-10-21 08:06:38起至10月24日08:06:38止，等待和部署也计入，不延长或重置。长命令、辅助worker及产物写入在tmux Uni-Poly独立window，保留命令／cwd／日志／真实exit。
- 验证、停止与审查：本轮规划仅检查最终文档diff与格式；DSH部署前仅做静态语法和最小diff检查，不运行额外smoke／benchmark／前置probe／基线重放或测试套件。原唯一最终CPU suite仍在周期收尾一次执行、≤300秒，失败不重测。真实C配置与正常训练日志确认策略加载；按全程和连续500updates窗口汇总step均值／中位／P90、loader等待、decode、原data_seconds、graphs/s和总墙钟，不新增CUDA同步，不将data_seconds称纯IO。异常未执行、缺字段或部分运行分别如实说明；不根据少数快慢步自动切策略／重试。NaN/Inf、OOM、缺键、schema／身份／writer／覆盖冲突及期限问题遵守原第11节停止规则。最终模型效果仍仅按第9节正式TEST判读，不以速度改变C／D必执行政策或候选选择。
- 交付：沿用原results／logs根；新增final_review/c_io1/保存实际授权、最小diff、部署时间、读取策略、预算快照、辅助命令／window／exit及全程／窗口统计，final_review/c_io1_report.md说明部署、C触发／实际加载及工程效果。保留b3c_io1及其报告，不覆盖或合并改写旧证据；C的pretrain/<实际臂>/目录仅由真实PT入口创建。DSH仅更新协调后的执行记录，交付标待审查，由Codex独立核对；不自行归档PROJECT_HISTORY。

## 8. 统一 FT 与原始预测

- 35单元矩阵固定；每单元10epochs仅head，60epochs joint，标准化MSE。train-only StandardScaler，不用val／TEST拟合变换；预测和标签均展平为一维防止广播。
- batch24、FP32、workers0、drop_last=false、WD0.020、clip5；encoder dropout0、head dropout0.100，不搜索batch／精度／dropout。
- head-only LR3e-4、CosineAnnealingLR(T_max=10,eta_min=3e-5)；joint backbone／norm2 LR1e-4，head LR1e-3，CosineAnnealingWarmRestarts(T_0=10,T_mult=1,eta_min=1e-6)。A／D的冻结或两个尺度标量按第4节明确例外。
- 每epoch一次正常validation，无early stopping。两个阶段共同按完整精度validation R²严格提高保存best；exact tie保留较早epoch，不能按末轮替代best。
- 每单元更新数固定70×ceil(n_train/24)，每完整35单元总56000；保存完整70行history、best阶段、原单位validation预测、scaler、source／split／行索引／sample_key、真实消耗与退出码。
- 完整best必须保存新旧有效模块、schedule／readout／trainability／attention／schema；正式预测strict恢复完整best及原scaler。部署只用二维拓扑，不能要求ET／Galformer／FP输入。
- 每FT单元硬期限1800秒；预留resume只保存现场，不授权自动恢复。没有额外checkpoint平均、seed集成、KRR或按TEST回退。

## 9. 全局选择、正式一次性 TEST 与结果判断

### 9.1 最终政策冻结

- A、B、唯一C和D的280FT全部正常结束后，方可做最终选择。任何候选不完整，不以其部分task均值参与排名；异常按第11节停止。
- 主候选全集仅A_TOP2、A_MIX246、B四臂、实际执行的C和D。选择完整精度validation宏平均最大的新候选作为FINAL；精确同分次序A_TOP2、A_MIX246、B2_E、B3_E、B2_C、B3_C、C、D。B_FP仅作固定历史reference，不参与本轮主候选排名以阻止新候选进入TEST。
- 一个FINAL身份用于所有task／fold，不逐task拼接。每折best epoch仍按该折validation选择。
- 正常完成全部实际FT且名单／best／schema／scaler／split与评估预算完整后，冻结FINAL及第9.2节规定的reference集合，并执行其唯一正式TEST批次。删除原V(FINAL)−V(B_FP)≥0.010及V(FINAL)≥V(B2_E)准入条件；即使validation没有提高或低于baseline，预注册名单也必须正式TEST后才能判读性能。
- TEST仍只因真实执行异常、产物不完整、身份／schema不匹配或预算／期限耗尽而停止；具体原因据实记录，不能将validation不足写成性能未通过或TEST=0的理由。
- 冻结文件selection/final_frozen.json保存完整候选验证结果、C／D必执行的用户依据及目标选择、FINAL、有效D_FT_POLICY、PT_WINNER、全部所需best／scaler／配置／TEST indices和输出路径，时间必须早于第一个TEST作业。旧a_policy.json保留并与有效政策区分；读旧TEST数值作研究背景不授权用其选择本轮政策。

### 9.2 唯一正式TEST批次

- 所有本轮正式TEST只在最终冻结后统一一次开展；每个实际模型／task／fold仅一次，不重评、不看结果后追加模型。
- 为使必执行C／D都获得正式性能证据，两者不受validation排名影响，均纳入本次唯一TEST批次，名单在首个TEST前确定，不临时补评。
- FINAL为B／C／D时，固定评估B四臂＋唯一C＋D，共6臂／210单元／35250图；C的直接reference=B3_U及D的PT_WINNER均在其中。
- FINAL为A臂时，评估{FINAL、唯一C、D、B3_U、PT_WINNER}，按完整模型身份去重，最多5臂／175单元／29375图；保留A作为全局FINAL的可能性，同时覆盖C／D及各自直接reference。该分支没有完整B四臂TEST时，不作正式2×2因子结论，不再追加B臂超过既定名单。
- 该名单由validation和祖先关系在TEST前固定，不依据TEST决定增加、删除或改选臂。不重跑历史B_FP／V1／r10；旧预测／指标只读复用。
- 每TEST单元≤300秒；输出完整原始y_true／y_pred、task／fold／seed、indices／sample_keys、best来源、scaler、原单位R²／RMSE。归一化不能再fit。
- 首要正式增量：T(FINAL)−T(B_FP)≥0.010才称达到本计划性能门槛；同时报告对V1／r10、名单内B2_E及FINAL直接reference的差值。另固定报告C−B3_U、D−PT_WINNER及C／D各自相对B_FP的TEST差值，应用同一0.010门槛；未评估reference标不可比较，不用validation替代。负增量和个别退化照实报告，不事后换FINAL。
- 判读按未舍入TEST差值区分：>0是本次正增量，≥0.010为性能通过，≤0为本次未观察到提升；正增量低于0.010不得写成达标，单组五折也不称稳定优势。未纳入预注册正式名单的候选只保留validation开发结果，正式性能“未完成／不可比较”，不得据其validation下降判无效。
- 若B四臂已正式评估，用T重复第5.3节的四个单因素差值和interaction差分；若未开放TEST，这些正式结论均未完成。

### 9.3 SOTA口径与结论层级

所有均值／std／差值展示3位小数并保留末尾零；择优、差分与门槛用原始完整精度。每task五折R²等权mean、std(ddof=1)，再七任务等权macro；不拼接预测重算宏R²。每折一次seed42—46，不能将fold std称为独立seed稳定性证据；r10原std口径差异单列。

区分：实现完成、最终CPU通过、validation开发触发、正式TEST正增量、正式性能门槛通过、论文数值目标、独立审查及归档。validation只决定开发选择，不能充当提升证据。达到+0.010不等于达到0.810，达到包络macro不等于七项全部达到参考。

论文与项目数据／split／版本等价性未证明，已知项目Egc／Xc样本3380／432，论文公开v1为4124／430；共享validation／TEST参与开发，不称独立盲测。始终以SOTA_RESULT.md用户表为本地数值目标，不把“数值达到”写成公平超过论文。

| Task | 论文最佳模型 | 参考R² | 历史V1 | 历史B_FP | B_FP−参考 | 本轮FINAL正式R²／差值 |
|---|---|---:|---:|---:|---:|---|
| Eea | Periodic-TDL | 0.934 | 0.926 | 0.933 | -0.001 | 未完成／不可比较：尚未授权执行 |
| Egb | Periodic-TDL (+Chem, +RC) | 0.923 | 0.930 | 0.934 | +0.011 | 未完成／不可比较：尚未授权执行 |
| Egc | Periodic-TDL (+Chem, +RC) | 0.911 | 0.912 | 0.913 | +0.002 | 未完成／不可比较：尚未授权执行 |
| Ei | Periodic-TDL | 0.839 | 0.808 | 0.816 | -0.023 | 未完成／不可比较：尚未授权执行 |
| EPS | Periodic-TDL | 0.787 | 0.746 | 0.764 | -0.023 | 未完成／不可比较：尚未授权执行 |
| Nc | Periodic-TDL | 0.856 | 0.808 | 0.787 | -0.069 | 未完成／不可比较：尚未授权执行 |
| Xc | Periodic-TDL | 0.423 | 0.380 | 0.393 | -0.030 | 未完成／不可比较：尚未授权执行 |
| 七任务等权macro | 逐任务最佳包络 | 0.810 | 0.787 | 0.791 | -0.019 | 未完成／不可比较 |

周期结束必须更新本节七项正式结果、项目减参考、fold／seed口径、模型身份和日志／产物路径。TEST未开放／结果不完整时保留“未完成／不可比较”并写具体原因；不能用validation或smoke补齐。EAT、Eib、Tg不纳入。独立审查通过后才由Codex归档PROJECT_HISTORY；本轮规划不预先归档。

## 10. 总预算与资源

2026-10-21 TEST判读修订仅移除validation性能准入门槛；本节既有PT／FT／TEST／CPU及墙钟上限不增加。未进入预冻结正式名单的候选没有TEST结论，不补评来扩展预算。

| 阶段 | PT starts／updates／源呈现 | FT单元 | epochs／validation | FT attempted updates |
|---|---|---:|---:|---:|
| A 两臂 | 0／0／0 | 70 | 4900 | 112000 |
| B 四臂 | 4／20000／20160000 | 140 | 9800 | 224000 |
| C 固定一个分层臂 | 1／5000／5040000 | 35 | 2450 | 56000 |
| D 固定一个组合 | 0／0／0 | 35 | 2450 | 56000 |
| 最大总额 | 5／25000／25200000 | 280 | 19600 | 448000 |

- 用户本次将原已预留的C／D额度改为固定执行，正常完成为5PT、280FT；没有增加原最高预算。实际消耗以活动账本为准，旧历史“未触发／0消耗”不代表本修订下可跳过；任何剩余额度不能转移成额外臂、重试、seed或更长训练。
- 输入准备最多1次，910756条PT＋5875个下游任务记录，≤14400秒；初始化生成1次，无模型预跑。PT每臂≤18000秒，FT每单元≤1800秒；正式TEST最多210次／35250图，每单元≤300秒。
- PT仅物理GPU2／3，一个双卡作业、每卡1个rank，B／C各臂依次执行；FT和TEST每卡最多4路。PT与FT／TEST分阶段，不混跑；不使用GPU0／1，不自行换卡、改batch、精度或并行配方。准备最多16CPU workers，FT workers0。
- 所有预计超过一分钟、GPU、worker或写训练产物的命令在tmux session Uni-Poly独立window。建议controller名ua2d_opt_r1；每个作业保存实际window ID／命令／cwd／环境／stdout／stderr／起止时间／真实exit。
- 自首次准备或A训练启动的较早时点开始，整个周期墙钟≤72小时；不因等待、失败或重启重置。阶段上限和总上限同时适用，不保证预算全部消耗或性能达标。
- 所有部分启动、失败、started epoch／validation、attempted update及预取呈现都计账。无retry／resume／替代根／自动加额度；一个单元中断不能从零补跑或补齐epoch。
- 唯一最终针对性CPU suite最多1次／总墙钟≤300秒，失败也占次数。仅在本计划全部实际执行及正式评估结束、或异常安全收尾后运行；不逐阶段运行测试，不重放旧suite、smoke、前置probe或benchmark。

## 11. 修改范围、停止条件与唯一最终测试

### 11.1 允许范围

未来获授权后，仅新增隔离的UA2D-OPT数据适配、hop3／context／readout模型、初始化／恢复、PT／FT／TEST入口、controller／账本／决策／汇总和必要CPU夹具，以及本文件执行登记。可采用 src/dataset/ua2d_opt.py、src/modules/ua2d_opt.py、src/training/ua2d_opt.py、scripts/run_ua2d_opt.py、scripts/prepare_ua2d_opt.py、scripts/pretrain_ua2d_opt.py、tests/test_ua2d_opt.py 等新文件；当前尚不存在的建议路径不是已交付实现。

复用现有稳定算术与数据读取时用显式参数，不修改旧模块全局OUT／LOG或恢复历史controller。旧默认行为、缓存、checkpoint、split、正式预测、事故证据、AGENTS和SOTA_RESULT不修改；不顺手重构／清理／删除文件。不安装或升级依赖。

官能团监督、等价RU一致性、4-hop及更大半径、更多schedule、二维键hop加深、PH／MCL／ET追加、动态权重、FP输入／核融合、局部蒸馏及新seed均为0预算研究备选。它们需要后续依据与另行授权，不自动串入本计划。

### 11.2 停止条件

遇到影响运行／结果的schema／checkpoint／vocab／原子映射／periodic image／split／scaler不匹配、有效池内失败几何、NaN／Inf、OOM、writer冲突、正式覆盖风险或任一预算／硬期限耗尽：停止受影响运行及新增派发，保留现场和真实消耗，记录需返修／阻断及影响。不得丢弃异常后继续、放宽检查、换parent或自动修复重跑。

只有依据命令／cwd／祖先关系明确属于本周期的进程才可在异常停止时处理；不干预无关旧作业。其他健康在途单元保留或安全收尾，不以停止监控代替停止controller。

### 11.3 最终CPU suite与审查

唯一最终CPU suite只覆盖本次实际实现的正确性：

- 周期1／2／3-hop最短路径、不同image重数、跨图offset、padding与逐层候选关系排除；固定B2行为和旧有效2-hop算术兼容。
- context干净标签／受损输入分离、词表train-only与周期邻接计数；四臂源样本／随机流一致，失败三维记录整条排除。
- TOP2冻结／eval／optimizer覆盖；MIX初始等于原读出及梯度路径；固定schedule／D政策完整恢复，D不得读取旧NONE或继承FT best。
- 全局窗口归约、完整新schema strict-load、pure2D预测、全局选择／必执行C及D／预算／TEST完整性与名单上限；本修订显式交接保留成功及在途单元、累计账本和一次claim，不重放准备／PT。使用小型CPU夹具，不用额外真实训练或GPU预跑代替；只补入原唯一最终suite，不提前或重复运行。
- 保留唯一命令、日志、真实exit、用时和适用范围；失败不重测。实现者自检通过后仍交Codex独立审查，不自行宣称独立验收或科学成功。

## 12. 交付路径与执行登记

| 交付内容 | 固定相对路径 |
|---|---|
| 授权、writer、命令、版本、原范围 | execution_manifest.json |
| 总额／阶段额／已启动／已完成／失败／预取／墙钟 | budget.json、status.json、timing.json |
| 原source／logical index映射及有效池来源 | inputs/sources.json、inputs/coverage.json |
| 统一hop3／逐hop关系视图、整数context标签与词表、静态缓存COMPLETE元数据 | inputs/hop3/、inputs/context/、inputs/cache_manifest.json |
| 缓存准备耗时／体积、实际data_wait／step耗时／吞吐／显存 | timing.json、final_review/throughput.json |
| 公共扩展初态与新旧张量来源 | initialization/、configs/ |
| B／C step5000及现场 | pretrain/{arm}/deploy.pt、resume.pt、rank counters |
| 全部实际FT完整best／scaler／history／validation预测 | finetune/{arm}/{task}/fold{fold}/ |
| A历史政策、B因子表、C／D必执行有效政策和最终冻结 | selection/a_policy.json、b_factorial.json、cd_required.json、conditional_decisions.json、final_frozen.json |
| 唯一批次正式原始预测／指标 | formal/{arm}/{task}/fold{fold}/ |
| 完整精度结果、逐task／fold差值、7任务对照、CPU、实际diff | final_review/ |
| 命令／cwd／日志／真实exit／tmux | logs/ua2d_opt_20261021/r1/{job}/ |

上述相对路径均在结果根results/ua2d_opt_20261021/r1/下，日志行除外。未授权不创建；未触发阶段不得伪造checkpoint或正式结果；没有TEST时不创建虚假final_frozen。

执行完成后状态先为“待审查”；遇问题据证据标“需返修／阻断”。交付必须区分已实现、已启动、执行中、失败、开发门槛未过、TEST未开放、CPU结果和正式性能结论。独立审查确认实际diff／一次最终测试／日志产物及七任务对照后，才判断完成并按规则归档。

### 本轮文档交付登记

- 用户授权：覆盖CURRENT_PLAN.md并提供执行提示词。
- 新计划状态：待授权；准备／PT／FT／TEST／CPU suite实际消耗均0。
- 已完成本地版本、相关diff、已有交接／源码／结果核对；仅修改本文件。
- 本文件git diff --check通过；6个本地Markdown链接均存在，提示词围栏配对且未使用单美元数学定界符。仅做文档diff／格式检查，没有运行项目测试或实验。未来运行前须核对届时工作区与writer，本快照不代替实时状态。

### 2026-10-21 TEST判读修订交接

- 用户最新依据：“另外判读是否有提升应该从TEST来看而不是Validate！”；本节覆盖本计划旧validation→TEST增量准入规则，不追溯改写历史其他周期记录，不扩大当前已授权预算。
- 2026-10-21 08:10起只读核对时，本周期已登记执行中／A，训练代码的final_decision／freeze_final及交付缺项原因仍包含旧validation gate。本轮仅修订本文件，未并行修改执行者正在维护的源码或重启controller；不得把文档修改视为运行规则已生效。
- 执行方必须在最终冻结前同步主候选排名（排除B_FP竞争）、TEST名单生成／freeze逻辑和交付原因：只检查矩阵／产物／预算完整，不设validation性能准入；未完成正式TEST不判性能失败。若controller已在内存加载旧函数，须明确核对实际运行采用的策略；不能仅修改磁盘函数后宣称已生效，也不得为此重训／重置计数／自动恢复运行。
- A／B及已触发C／D的成功或在途单元保留，0.005仅控制后续开发资源；正式名单和210次上限不变，正式TEST后不改候选。本轮没有新增模型预测、测试或训练消耗。

### 2026-10-21 自主修正与控制器交接（最新落实记录）

- 授权来源：用户明确“请你自行修正！”，对应上轮发现的旧validation→TEST gate和context加载期字符串编码。本次授权用于最小实现修正及一次显式控制器交接，不增加task／fold／seed、PT／FT／TEST／CPU预算，不自动重试训练。
- 已修正src/training/ua2d_opt.py、scripts/run_ua2d_opt.py：B_FP不再竞争本轮FINAL；validation仅提名新候选，完整矩阵冻结后按预注册名单一次TEST，不设validation性能准入。正式TEST差值>0记录正增量、≥0.010才性能通过；未完成TEST的performance_pass为未判定，不用validation填负结果。
- 已修正scripts/prepare_ua2d_opt.py、src/dataset/ua2d_opt.py、src/modules/ua2d_opt.py及scripts/pretrain_ua2d_opt.py：唯一准备作业将每原子context整数ID、≤1／2／3-hop关系行索引持久化；PT／FT collator按relation offset组合缓存视图，O8直接gather已缓存关系。PT只读整数标签，缺失或不完整缓存禁止启动／禁止在线fallback；不改变腐蚀随机流、模型参数或原训练配方。静态缓存最后发布inputs/cache_manifest.json的COMPLETE及来源／词表／覆盖信息，记录实际缓存体积与PT数据等待／吞吐。
- 08:27:26仅暂停旧controller PID1184170的派发，未向任何FT worker发信号；冻结时47个已派发A单元完整保留。08:33:01核对旧PID／启动身份／cwd／命令后，显式替换该dispatcher，避免旧finally停止健康作业；旧window @1495及原日志保留。旧controller的行政退出不是训练单元失败。
- 新controller为Uni-Poly/@1545，PID1195532，08:33:06起按原账本继续；47个旧单元的名称、started、log、window均继承，无重复派发。预算上限不变，wallclock_started_unix仍1792541198.175118，即北京时间2026-10-21 08:06:38，交接耗时计入原72小时。
- 新实际命令为/opt/conda/bin/python -u scripts/run_ua2d_opt.py controller --handoff results/ua2d_opt_20261021/r1/controller_handoff_repair1/request.json；cwd为本项目根；日志logs/ua2d_opt_20261021/r1/controller_repair1/。CLI仅接受这一次明确授权的repair1账本交接，不提供训练自动resume／retry；claim.json防止重复接管。
- 现场及交接依据：results/ua2d_opt_20261021/r1/controller_handoff_repair1/中的request.json、冻结budget／manifest／status／timing、claim.json及activation.json；原implementation_before_start.patch和新增implementation_repair1.patch保留。最新进度仍以活动budget.json／status.json及文末执行登记为准。
- 验证层级：已完成七个修改源码／夹具的静态语法检查，新增实际FT已正常开始推进，继承单元身份及预算／时间未重置。本轮没有运行专项测试或模型前置probe；必要的新用例已补入tests/test_ua2d_opt.py，仍在原周期结束时唯一CPU suite集中运行，不能提前称测试通过。输入准备和PT在本记录时尚未启动，因此不能声称缓存已经构建或预训练速度已经实测。
- 本次仅落实修正与交接，不办理本周期独立审查／PROJECT_HISTORY归档。前节“代码仍待同步”是08:10只读快照，其当前实现状态由本节更新。
- 08:39查询预计结束时间时发现并修正上一轮准备代码中的局部变量遮蔽：将context数值记录的变量record改为numeric_record，避免覆盖准备进度函数record()并在首批缓存写入后报错。修正时唯一准备尚未启动，新准备子进程将直接加载修改，无需交接controller或重跑单元；只做AST语法检查，未增加测试／实验启动，原预算和计时不变。

## 13. 可复制给 DSH 的执行提示词

以下提示词用于用户另行明确下达执行指令；本次文档交付不代表已经发出执行授权。

~~~text
请执行CURRENT_PLAN.md中UA2D-OPT-20261021-01 / r1及r1-CD-REQUIRED的完整有限范围。我授权其中明确列出的实现、一次输入准备、A／B／一个C／一个D固定阶段、最终一次性正式TEST及唯一最终CPU suite；不授权其他历史队列、额外搜索或自动重试。当前已有周期从第13.2节安全交接接续，不从A重放；先登记实际授权、时间、执行者和writer，不再重复询问同一范围的启动许可。

基于本地dev／3503fbe780a5399410d6db6055305925765c3ff8及现有dirty工作区，重新核对AGENTS、CURRENT_PLAN、SOTA_RESULT、相关diff和相关运行writer。保护所有已有修改／删除／未跟踪文件及旧事故；不读TODO、不远程同步、不新增哈希／校验和／文件指纹，不安装／升级依赖。DEMOL和ET旧交付仍待独立审查，不恢复它们或旧V1_CONTEXT／几何残差队列。

正常顺序为A→B四臂→一个C→一个D→最终冻结／TEST→唯一CPU；保留当前所有已完成／在途单元，不重跑A。A只读fast_r4/B_FP step5000，TOP2／MIX246各35FT已经完成，旧B_FP指标直接复用。根据A完整精度validation选择D政策，当前固定MIX246，不以旧0.005门槛或a_policy.json中的NONE跳过D；旧决策原样保留，新有效政策另存cd_required.json／conditional_decisions.json。

所有新PT只读原prepared_cache及其source/index，统一复用ET repair1的910756 eligible logical indices，整条排除635条失败3D；经原source_rows／keys映射，不把logical index当source_row。一次新准备构建≤3-hop拓扑及train-only一跳chemical-context标签／词表，旧3D模板只读，不生成坐标／构象／FP／PH／split。身份错误停止，不能降级或静默筛样本。

保证数据供给速度，严格执行第7.1节：周期BFS／路径／键化学、≤1/2/3-hop关系视图、身份映射、context计数／词表及每原子整数类别ID、周期分组／端点／角度索引在唯一准备作业内数值化落盘，COMPLETE后全部B/C只读复用。禁止在__getitem__／forward做RDKit构图、context字符串转ID、Python重分组或缓存miss在线重算；缺缓存即报告，不边训边建。在线仅保留既定随机增强、噪声相关矢量化几何、batch offset／gather及模型与归约计算，不冻结增强或缓存可学习结果。沿用有界worker预取、二进制IPC与pinned H2D，在真实训练低开销记录data_wait和吞吐，不加benchmark／probe。缓存准备仍只1次≤14400秒，训练及全程预算不增加。

B为匹配2×2：B2_E／B3_E／B2_C／B3_C，四臂分别完整1PT×5000updates＋35FT。统一max-hop3模块和共同扩展step0：旧同形状参数、SPD前三行／旧path槽严格复制，新增参数seed20261021隔离初始化；类别head按目标区分并共享对应初态。所有四臂相同有效池、源顺序／腐蚀、3D Galformer、ALIGN、mean读出和原FT，不带入A政策。2D hop按真实lifted最短化学路径，保留image重数，≤2或≤3必须在attention前筛掉关系；同步路径宽度／bond化学／SPD／collator／cache／完整schema，不能只改max_hops或将bias置零。报告四个单因素差值及匹配交互差分，不将组合−baseline当interaction。

B全部完成后，只按validation在B3_E／B3_C选U，ties选E；无论与B2_U差值如何，都从原共同step0执行唯一C_SCHED_U（1,1,2,2,3,3），1PT＋35FT，并沿用C-IO1读取加速。再从B四臂及C选唯一PT_WINNER，对其PT package＋原公共head重新执行D的35FT／0PT，固定MIX246，标量从0开始，不继承FT best或A已学系数。C／D的trigger均true，required=true并注明用户变更；不伪造旧门槛达标，不追加其他目标、hop或政策。

PT固定GPU2/3 world2 micro84 acc6，BF16／关键lossFP32，5000updates／5040000呈现／每臂≤18000秒；AdamW peak2e-4 WD0 clip1 warmup500 cosine end1e-6。元素或context二维CE替换，原3D CE＋双向InfoNCE温度0.100、前1000步升到权重0.100；mask0.400、噪声0.030Å、FP关闭、attention1/√512不变。对比候选单微批最多168，不将累积算作1008候选。

FT固定七任务5fold seed42+fold、原split／35公共head、train-only scaler；10+60epochs、batch24 FP32 workers0、原LR／scheduler／WD／dropout／clip，按文档A/D例外控制训练范围。每epoch一次validation、完整精度严格择best，不early-stop。每35单元2450epochs／56000updates，单元≤1800秒；完整记录history／best阶段／预测／scaler／source／schema。

硬上限：1次准备（910756 PT＋5875下游记录，≤14400秒）；最多5PT／25000updates／25200000呈现；280FT／19600epochs及validations／448000尝试更新；正式TEST最多210单元／35250图，单元≤300秒；最终针对性CPU suite仅1次≤300秒。C／D使用原预留额度固定执行，任何剩余不转移。总墙钟自北京时间10月21日08:06:38起≤72小时，截止10月24日08:06:38，不重置。失败和部分工作含预取都计预算，不retry/resume/换卡/改batch或精度/减步数/扩预算。

仅物理GPU2/3，PT一个双卡作业、各臂依次，FT/TEST每卡最多4路，PT与FT/TEST不混跑。长任务／GPU／worker／训练写入均在tmux Uni-Poly独立window。新根results/ua2d_opt_20261021/r1及对应logs；保留真实命令/cwd/window ID/log/exit，单writer。schema／身份／split／scaler／有效池几何／NaN/Inf／OOM／writer／覆盖风险／期限问题立即按第11节停止受影响流程和新派发、保留现场，不静默补救。不做阶段smoke、前置probe、benchmark或旧suite重放。

全部280FT完成，仅用validation提名并冻结唯一全局新候选FINAL；B_FP只作reference，不得以validation增量不足拒绝正式TEST。C／D及各自直接reference均进入唯一批次：FINAL为B／C／D时固定B四臂＋C＋D，共6臂／210次；FINAL为A时按{FINAL、C、D、B3_U、PT_WINNER}去重，最多5臂／175次，不能将A额外加到6臂集合。所有名单及best／scaler／schema／TEST indices在首个TEST前一次冻结，各单元只评一次，不根据TEST改选／补训／拼task。唯一性能证据为完整正式TEST：Δ>0报告本次正增量，Δ≥0.010才性能通过；报告C−B3_U、D−PT_WINNER及两者对B_FP差值，未正式评估只写不可比较。之后唯一最终CPU suite收尾执行一次，失败不重测。

完整精度报告五折mean/std(ddof1)、七任务等权macro、单因素和交互差分、FINAL相对B_FP/V1/r10及直接reference的差值与≥0.010门槛；展示统一3位小数。逐项更新Eea/Egb/Egc/Ei/EPS/Nc/Xc对SOTA_RESULT的正式数值表、模型、差值、统计口径及日志产物，未开放TEST写未完成／不可比较，不用validation补齐。共享TEST非独立盲测、数据／协议未证实等价，只作论文数值对照。交付第12节全部实际产物、预算、时间、真实exit和diff，状态待审查；DSH自检不代替Codex独立审查，不自行归档PROJECT_HISTORY。
~~~

### 13.1 C读取加速扩展：可复制给DSH的执行提示词

以下提示词用于用户下达r1-C-IO1工程实施指令；本轮规划未启动实现或部署，原r1训练授权及运行状态保持不变。

~~~text
请实施CURRENT_PLAN.md第7.2节UA2D-OPT-20261021-01 / r1-C-IO1。我授权其中明确限定的C读取加速实现、一次安全部署、正常运行监控和事实交付，不授权新增候选、训练重试、缓存准备或测试预算。Codex负责规划和独立审查；你作为DSH只按计划执行，不自行改变科研范围或宣布审查通过。

基于dev／3503fbe及现有dirty工作区，先只读核对AGENTS、CURRENT_PLAN、第7.2节前置状态及实际writer。保留既有controller和Uni-Poly:@1645的B3C-IO1助手、暂存补丁与报告，不抢先修改它将校验／覆盖的源码，也不新起controller或重启B2_C／B3_C。

先准备C扩展方案；待B3C-IO1已实际部署、B3_C PT正常exit0且rank／worker退出，利用B3_C的35FT在途窗口，基于真实已部署源码一次增量修改scripts/pretrain_ua2d_opt.py。部署前再次确认C没有启动记录、进程或输出目录，且没有其他源码writer。若窗口错过、C已启动、前置策略缺失或周期异常，保留现场报告，不热切换、不重启、不绕过原停止规则。

统一用显式适用集合B3_C／C_SCHED_E／C_SCHED_C控制read_policy、engineering、loader_wait_seconds、batch_decode_seconds及timing输出。复用已部署的b3c_io1_readahead接口和两份只读LMDB；B3_C保留原IO1身份，实际C使用r1-C-IO1及准确的<臂名>_PT_only范围。默认original和其他臂不变，不修改dataset／模型／controller／FT／选择／TEST逻辑。

C按最新第6节必执行一个目标，不再使用旧0.005门槛；schedule固定1,1,2,2,3,3，复用统一max-hop3缓存和1／2／3-hop视图，不重建／复制／全库预热缓存。样本、随机流、初始化、损失、精度、优化器与LR保持原值。资源和预算严格按第7.2节，不增加最高预算；D也使用原预留35FT固定执行。该IO助手不负责控制器交接或修改科学决策，r1-CD-REQUIRED由第13.2节落实。

不增加benchmark、smoke、前置probe或阶段CPU suite；只做静态语法与最小diff检查，原唯一最终CPU仍在周期收尾执行。用原定C真实训练的配置及全程／连续500步窗口日志记录加载策略、step均值／中位／P90、loader等待、decode、原data_seconds、吞吐和总墙钟；不新增CUDA同步，不预先宣称提速。异常未执行或部分运行如实说明，不造产物或补跑。

在tmux Uni-Poly独立window保留部署／汇总命令、cwd、日志和真实exit；辅助只写自身证据，不派发训练或改写主预算。所有异常遵守原停止规则，不自动retry/resume、换卡、改batch、减步数或扩预算。

交付原结果根final_review/c_io1/和final_review/c_io1_report.md，保留B3C-IO1全部旧记录。CURRENT_PLAN仅协调更新事实执行记录，不与controller并发写入，不覆盖科研规划正文。工程交付标待审查，交Codex独立审查；正式模型性能仍按原冻结名单和唯一TEST判定，不自行归档PROJECT_HISTORY。不读TODO、不远程同步、不新增哈希验证或依赖。
~~~

### 13.2 C／D必执行修订：可复制给DSH的执行提示词

本提示词供DSH落实用户本次C／D必执行要求。它覆盖第13节旧条件分支及第13.1节与之冲突的执行门槛；既有IO助手的工程边界保持不变。当前Codex仅更新计划，内存中的旧controller尚未替换，不能宣称修订已生效。

~~~text
请实施CURRENT_PLAN.md的UA2D-OPT-20261021-01 / r1-CD-REQUIRED，按第6.1—6.3、7.2、9.2和本节接续现有周期。我要求无论B的完整validation增量如何，都执行一个C和一个D；授权落实所需的最小源码修改、一次显式controller交接、原预算内剩余训练／唯一TEST及原唯一最终CPU，不授权重训、额外候选、自动retry/resume或扩预算。Codex负责规划与独立审查，你作为DSH负责实现、执行和事实交付。

基于dev／3503fbe及现有dirty工作区，核对AGENTS、当前计划、实际diff、进程、预算和IO助手。先准备修改scripts/run_ua2d_opt.py、src/training/ua2d_opt.py及原最终suite必要夹具；保持存量r1配置／schema／结果根和全部旧产物，不安装依赖、不读TODO、不远程同步、不新增哈希验证。本任务不与C-IO1助手同时修改scripts/pretrain_ua2d_opt.py，不修改dataset／模型／缓存。

C在B3_E／B3_C中按完整validation macro选U，同分E；不论B3_U−B2_U是否≥0.005，都从原E_step0或C_step0执行唯一C_SCHED_U，schedule=1,1,2,2,3,3，完整1PT＋35FT，并保留C-IO1预读与计时。不得执行两个C目标。

D固定完整35FT／0PT。A现有完整validation的winner为A_MIX246，因此D_FT_POLICY固定MIX246，不再受旧NONE阻止。保留a_policy.json原值；另存selection/cd_required.json，并在conditional_decisions.json中令C.trigger／D.trigger及required均true，记录用户变更原因和实际差值。D的policy／config必须读新有效D.FT_POLICY，而非旧a_policy.json的NONE。PT_WINNER仍从完整B四臂＋C按validation选择，从其step5000＋原公共head重新FT，MIX标量重置0，不继承FT best或A系数。

当前controller已加载旧分支，仅改磁盘不会生效；不得重用A阶段repair1交接入口或从完整run()重放。按第6.3节实现controller_handoff_cd_required/request.json的一次专用claim、原锁唯一writer及从B3_C未完成FT接续的入口。

等待B3_C PT正常结束、rank／worker退出、B3C-IO1助手正常汇总退出，且C-IO1助手已在B3_C FT窗口完成源码部署。确认仍在B3_C FT矩阵、无C／D／TEST／CPU启动。核对身份后，在稳定等待点暂停旧dispatcher及C助手的监控循环；训练worker不发信号。冻结后检查无半派发状态，完整保存作业／预算／计时现场。仅行政替换旧dispatcher，避免旧finally停止健康作业，不信号整组；记录真实exit与原因。

新controller在Uni-Poly独立window获得原锁，继承所有成功／在途job身份和原wallclock_started_unix=1792541198.175118，更新manifest的新writer／命令／window及政策ID后再恢复C助手。仅派发B3_C从未启动的剩余FT，随后汇总已有A／B，固定执行C→D→冻结／TEST→唯一CPU。绝不重跑准备、初始化、A、前三个B或B3_C PT；失败／部分单元不重排。若错过交接阶段或存在异常，保留现场报告，不静默另建通路或重复接管。

正常280FT完成后按原全局validation次序选FINAL，但C／D均固定纳入唯一正式TEST：FINAL为B／C／D时名单B四臂＋C＋D，共6臂／210单元；FINAL为A时名单{FINAL、C、D、B3_U、PT_WINNER}去重，最多5臂／175单元。先冻结全部best／scaler／config／indices及名单，首个TEST后不加模型、不改选。报告C−B3_U、D−PT_WINNER及C／D对B_FP的正式差值，Δ>0仅正增量，Δ≥0.010才性能通过；缺项如实标不可比较。

保持原5PT／25000updates／25200000呈现、280FT／19600epochs／448000updates、210TEST／35250图、1CPU suite上限。仅GPU2/3，PT world2 micro84 acc6 global1008 workers4/rank prefetch2，FT/TEST每卡最多4路，阶段不混跑；C PT≤18000秒，FT单元≤1800秒，72小时截止仍2026-10-24 08:06:38。必执行不豁免NaN/Inf、OOM、schema/身份/覆盖/writer冲突、失败或超时等原停止规则，不换卡、改batch、减步数或自动重试。

不加smoke、benchmark、前置probe或阶段suite。必要的必执行决策／D有效政策／去重名单与上限／一次交接保留存量的用例只补入原最终CPU suite，全部实际执行结束或安全异常收尾后一次运行、≤300秒，失败不重测。部署只检查静态语法和最小diff，不以自检代替Codex独立审查。

交付selection/cd_required.json、controller_handoff_cd_required/、logs/.../controller_cd_required/及final_review/cd_required_report.md，保留旧controller／IO助手／a_policy／原事故全部证据，不覆盖旧报告伪装历史。协调更新CURRENT_PLAN的实际授权与执行记录，明确旧规则、新政策及实际生效时间。最终附完整配置、预测、七任务SOTA数值对照、预算、实际diff、命令／窗口／真实exit，状态待审查，交Codex审查后再归档。
~~~

### UA2D-OPT r1 实际授权与执行登记

- 登记时间：2026-10-21T07:59:41.716573+08:00；执行者／初始writer：Codex；状态：执行中（实现）。
- 用户授权原文：

> 请执行 CURRENT_PLAN.md 中 UA2D-OPT-20261021-01 / r1 的完整优化计划，严格遵守第1—12节的范围、预算、触发条件和停止规则。我授权其中明确列出的实现、准备、训练及评估，不授权其他历史队列或额外搜索。先登记本条授权，再按文档执行，不重复询问同一范围的启动许可。
>
> 顺序为 A→B→满足条件的C／D→最终冻结与一次性正式TEST→唯一最终CPU suite。所有选择仅依据validation，不根据TEST改选、补训或拼接任务。
>
> 仅GPU2／3，tmux Uni-Poly独立window；最多5次PT、280个FT单元，总墙钟72小时。失败和部分运行计预算，不自动retry/resume、换卡、改batch或扩大范围。
>
> 保护现有改动、旧结果及事故证据；不读TODO、不远程同步、不新增哈希验证。完整交付配置、预测、日志、真实退出码、预算、实际diff及七任务SOTA对照，标为待审查，交Codex独立审查。其余具体要求以该计划正文及第13节完整提示词为准。

- 基于dev／3503fbe780a5399410d6db6055305925765c3ff8及既有dirty工作区，已有修改／删除／未跟踪文件保留；未读TODO、无远程同步／哈希验证。
- 固定结果／日志根已创建用于授权登记；准备／初始化／PT／FT／TEST／CPU启动均0；72小时尚未起算。
- tmux Uni-Poly存在；旧SOTA controller PID125308／127331仍在，须只读确认其资源与writer，不恢复或干预旧队列。
- execution_manifest.json保留授权原文、时间、执行者及初始工作区状态。

### UA2D-OPT 最新执行状态

- 时间：2026-10-22T09:36:06.998319+08:00；状态：待审查；阶段：r1-FT-REPAIR1。
- # r1-FT-REPAIR1 实际交付

状态：待审查；DSH执行，待Codex独立审查。
FINAL：A_MIX246；完整正式TEST：True；CPU exit=0；正式性能：正增量但未达门槛。

旧197个FT job／189个成功结果／8个失败及其真实exit原样保留，8个指定失败仅各一次独立fresh attempt，从PT step5000及原公共head完整70epochs。
原8失败按560epochs／560validations／34090updates替代已知下界保守计账，不重复相加；原3份RUNNING残留在snapshot/incidents.json标明已中断，原记录未改。
完整配置、checkpoint、scaler、history、预测及实际成功attempt见attempt_mapping.json和artifact_index.json；命令、window和真实exit见budget.json及logs/ua2d_opt_20261021/r1/repair_ft1/。
旧事故快照与报告见snapshot/；实际源码差异见actual_diff.patch；本次CPU只在收尾执行一次，不重测。
完整validation臂：A_TOP2, A_MIX246, B2_E, B3_E, B2_C, B3_C, C_SCHED_E, D_INTEGRATE；部分结果另列validation.json，不以空表抹去已完成结果。

| 正式比较 | Δmacro R² | 判读 |
|---|---:|---|
| C_vs_direct | -0.011 | 本次未观察到提升 |
| D_vs_direct | +0.000 | 正增量但未达门槛 |
| C_vs_B_FP | -0.012 | 本次未观察到提升 |
| D_vs_B_FP | -0.008 | 本次未观察到提升 |
| FINAL_vs_B_FP | +0.000 | 正增量但未达门槛 |

B_FP既有正式TEST macro R²=0.791；不重测，差值使用完整精度。

## A_MIX246 七任务SOTA数值对照

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.933±0.014 | 0.934 | -0.001 | Periodic-TDL |
| Egb | 0.934±0.012 | 0.923 | +0.011 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.015 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.816±0.032 | 0.839 | -0.023 | Periodic-TDL |
| EPS | 0.764±0.058 | 0.787 | -0.023 | Periodic-TDL |
| Nc | 0.787±0.100 | 0.856 | -0.069 | Periodic-TDL |
| Xc | 0.393±0.146 | 0.423 | -0.030 | Periodic-TDL |

## C_SCHED_E 七任务SOTA数值对照

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.933±0.017 | 0.934 | -0.001 | Periodic-TDL |
| Egb | 0.932±0.019 | 0.923 | +0.009 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.912±0.014 | 0.911 | +0.001 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.795±0.041 | 0.839 | -0.044 | Periodic-TDL |
| EPS | 0.732±0.067 | 0.787 | -0.055 | Periodic-TDL |
| Nc | 0.812±0.066 | 0.856 | -0.044 | Periodic-TDL |
| Xc | 0.336±0.117 | 0.423 | -0.087 | Periodic-TDL |

## D_INTEGRATE 七任务SOTA数值对照

| Task | 正式R² | 参考R² | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.931±0.025 | 0.934 | -0.003 | Periodic-TDL |
| Egb | 0.936±0.015 | 0.923 | +0.013 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.915±0.016 | 0.911 | +0.004 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.794±0.042 | 0.839 | -0.045 | Periodic-TDL |
| EPS | 0.737±0.058 | 0.787 | -0.050 | Periodic-TDL |
| Nc | 0.795±0.087 | 0.856 | -0.061 | Periodic-TDL |
| Xc | 0.374±0.066 | 0.423 | -0.049 | Periodic-TDL |

口径：各任务五折等权R²，std(ddof=1)，seed42+fold；七任务等权macro，原始数值完整精度，展示3位小数。
协议等价性未证明，仅作论文数值对照；共享TEST不称独立盲测。完整B四臂未正式评估时不宣称2×2因子TEST结论。
未完成／阻断原因：无。
；当前交付指向repair_ft1/，原失败交付及原始exit保留。DSH自检不替代Codex独立审查。
- 结果根：/root/workspace/Uni-Poly-Plus-master/results/ua2d_opt_20261021/r1；日志根：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_opt_20261021/r1。实现者自检不构成独立审查。

<!-- UA2D_OPT_EXECUTION_END -->

### B3C-IO1 工程执行记录

- 2026-10-21T20:56:58.461761+08:00；UA2D-OPT-20261021-01 / r1-B3C-IO1获本条用户明确授权，仅部署既定局部工程方案；在B2_C PT正常exit0后的FT窗口替换两个指定源码，B3_C尚未启动。默认original，仅B3_C PT启用两份LMDB readahead=True并新增loader／decode计时。既有controller、缓存、配置身份、预算及72小时起点不变；无新增准备、PT、测试或retry/resume。最小diff及后续真实训练全程统计交付至final_review/b3c_io1_report.md；独立审查待Codex进行。

### C-IO1 工程执行记录

- 2026-10-21T22:54:37.110075+08:00；用户已明确授权第7.2节及第13.1节r1-C-IO1，由DSH执行、Codex独立审查。B3C-IO1已实际部署，原B3_C PT真实exit0、两rank完整5000步且已观察rank／worker均退出；在其35FT在途窗口、确认C无启动记录／进程／输出目录后，基于当时实际源码一次增量部署。仅修改scripts/pretrain_ua2d_opt.py，统一显式适用集合B3_C／C_SCHED_E／C_SCHED_C；C复用既有readahead策略与loader／decode计时，C engineering使用自身方案ID与实际臂范围。保留controller、B3C助手及其全部旧证据，dataset与缓存只读复用；C触发、schedule、样本、初始化、配方及预算不变，D仍跳过，无额外训练／准备／测试或retry/resume。部署与C实际触发／加载分开记录；实际统计及事实交付见final_review/c_io1_report.md，待Codex审查，不作独立审查结论。

### r1-CD-REQUIRED 实际授权与交接登记

- 2026-10-21T22:54:39.759880+08:00；用户本条明确授权第6.1—6.3节及第13.2节，DSH执行、Codex独立审查。C取消0.005启动门槛并固定执行一个目标，D固定MIX246重新35FT；旧a_policy.json保留，selection/cd_required.json由新controller登记有效政策。B3_C PT正常完成，B3C-IO1助手正常汇总退出，C-IO1已于2026-10-21T22:54:37.100854+08:00完成部署；在其35FT稳定在途窗口，仅暂停旧dispatcher PID1195532和C监控助手PID1459499，未向训练worker发信号。冻结账本／作业／命令／clock及授权存controller_handoff_cd_required/，三份最小源码／夹具补丁部署完成；不重放准备、初始化或已启动训练。新writer生效及助手恢复时间以activation.json和manifest为准；原累计预算与72小时起点不变。交接及最终交付见final_review/cd_required_report.md，待Codex审查。

### r1-FT-REPAIR1 用户授权与返修接续登记

- 2026-10-22T08:04:01.866001+08:00：用户本条明确授权一次有限FT返修，DSH实现／执行／记录，Codex独立审查。原现场197FT启动、189成功、8失败，4个B PT各5000完成；C／D／TEST未启动，FINAL未冻结，旧CPU已23 passed。未发现本轮活跃writer／训练；原唯一writer锁可独占取得。旧结果、失败目录、8个非零exit和3个RUNNING残留原样保留，快照见repair_ft1/snapshot/。
- 仅指定8失败逻辑单元各全新attempt完整70epoch一次；另13个从未启动B3_C，随后固定一个C和MIX246的D。总上限修订为288FT／20160epochs及validations／482090 attempted updates／2CPU，PT和TEST上限不变；本次最多91FT／6370epochs／156450 updates、1C PT、1CPU。旧8失败以560epochs／560validations／34090updates替代其旧下界计账，单列实际已知与不确定部分，不重复相加、不转移。
- 起点1792541198.175118及北京时间2026-10-24 08:06:38截止保持不变。授权、实际diff、attempt映射、配置／预测／预算／退出及本次最终CPU交付至repair_ft1/，日志logs/ua2d_opt_20261021/r1/repair_ft1/；当前交接以本修订记录为准，状态：执行中（实现）。禁止新增retry/resume、测试轮次或研究范围；最终待Codex审查，不归档PROJECT_HISTORY。

### r1-FT-REPAIR1 前事故记录原文保留

以下为返修授权前实际失败交付原文；当前状态以新的最新执行状态及repair_ft1/status.json为准。

- 时间：2026-10-21T23:42:08.701791+08:00；状态：需返修；阶段：执行收尾。
- # UA2D-OPT r1 实际交付

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


# r1-CD-REQUIRED 实际执行交付

状态：需返修；独立审查：待Codex审查。DSH执行自检不构成独立审查。

旧a_policy.json保留，旧NONE仅为历史开发决策；有效政策另存selection/cd_required.json。C无validation增量准入门槛，D固定MIX246。
仅一次显式controller交接，从B3_C未完成FT接续；原准备、初始化、成功和在途单元、累计预算及72小时起点均保留。
C及D使用原预留额度；无新增候选、重复准备、训练retry/resume或额外测试。命令／窗口／真实exit及消耗见budget.json和controller_handoff_cd_required/。

FINAL：未冻结；CPU exit=0。

| 正式对照 | Δmacro R² | 依据未舍入值的记录 |
|---|---:|---|
| C_vs_direct | 未完成／不可比较 | 未完成／不可比较 |
| D_vs_direct | 未完成／不可比较 | 未完成／不可比较 |
| C_vs_B_FP | 未完成／不可比较 | 未完成／不可比较 |
| D_vs_B_FP | 未完成／不可比较 | 未完成／不可比较 |

## C未确定 七任务论文数值对照

| Task | 正式R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| eps | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

## D_INTEGRATE 七任务论文数值对照

| Task | 正式R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| eea | 未完成／不可比较 | 0.934 | — | Periodic-TDL |
| egb | 未完成／不可比较 | 0.923 | — | Periodic-TDL (+Chem, +RC) |
| egc | 未完成／不可比较 | 0.911 | — | Periodic-TDL (+Chem, +RC) |
| ei | 未完成／不可比较 | 0.839 | — | Periodic-TDL |
| eps | 未完成／不可比较 | 0.787 | — | Periodic-TDL |
| nc | 未完成／不可比较 | 0.856 | — | Periodic-TDL |
| xc | 未完成／不可比较 | 0.423 | — | Periodic-TDL |

统计口径：每任务五折R²等权mean、std(ddof=1)，seed42+fold，再七任务等权macro；差值与门槛使用完整精度。
数据／split／协议等价性未证明，仅作论文数值对照；共享TEST不称独立盲测。完整B四臂未正式评估时不声称2×2 TEST因子结论。
完整配置、原始预测、scaler、history、selection、预算、七任务FINAL表及完整精度配对差值见final_review/results.json、report.md及原产物路径。
异常：Traceback (most recent call last):
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_opt.py", line 279, in run
    self.matrix(['B3_C'])
  File "/root/workspace/Uni-Poly-Plus-master/scripts/run_ua2d_opt.py", line 146, in matrix
    if any(j['exit'] not in (None,0) for j in active):raise RuntimeError('matrix unit failed; stop dispatch')
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: matrix unit failed; stop dispatch
。

- 结果根：/root/workspace/Uni-Poly-Plus-master/results/ua2d_opt_20261021/r1；日志根：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_opt_20261021/r1。实现者自检不构成独立审查。



## UA2D-TRANSFER-20261022-01 / r1：当前授权与执行交接

- 2026-10-22T10:16:45.316414+08:00；用户本条明确授权，DSH负责实现／执行／监控／记录／交付，Codex负责规划及独立审查。当前基于dev／3503fbe780a5399410d6db6055305925765c3ff8及既有dirty工作区；新结果与日志根此前不存在，未发现本轮或旧OPT活跃writer。用户说明上一轮已独立审查，现有原交付metadata仍PENDING，按本条明确授权推进，本执行不代填旧审查。
- 唯一parent为fast_r4/pretrain/B_FP/deploy.pt，原2-hop O8／6层512维／denominator512／norm2与35公共head；七任务5fold、seed42+fold，原split、scaler、10+60epochs、batch24、FP32、workers0及原优化器配方固定。B_FP和A_MIX246仅复用既有正式结果。
- 四臂R_MEAN、R_STD、A_SP、R_STD_SP全部各35FT；残差LN512→Linear128→GELU→Linear512，末层零初始化、独立固定RNG；population标准差使用sqrt(v+1e-6)-sqrt(1e-6)，缺失图残差0。SP仅joint对o8/norm2施加0.5×1e-3×Σ||θ−θ0||²，θ0 GPU FP32 detached常量；不改WD、不加teacher。
- 新账本：prepare1／公共初始化1／PT0／140FT／9800epochs及validations／224000attempted updates／140TEST及23500图／1最终CPU≤300秒。一次原始2-hop准备5875条≤3600秒，完整发布COMPLETE后才训练；FT≤7200秒，TEST≤300秒，总墙钟从首次准备启动起≤24小时。仅GPU2／3、FT或TEST每卡≤4路，各阶段不混跑。失败及部分执行计预算，不retry/resume、换卡、减epochs或转移额度。
- 140FT完整后按validation提名唯一FINAL，同分R_MEAN→R_STD→A_SP→R_STD_SP；四臂无准入门槛，一次预冻结后全部唯一正式TEST。预登记：各臂−B_FP、STD−MEAN、STD_SP−STD、STD_SP−A_SP、STD_SP−STD−A_SP+B_FP interaction，以及FINAL−A_MIX246。仅完整正式TEST判读，FINAL对B_FP≥0.010才性能通过；七任务SOTA及包络只作协议有限制的数值比较。
- 仅新增隔离模块／训练／准备／入口／专项测试及本记录，不修改旧模型、dataset、controller、结果或预算；无额外预训练／搜索／probe／smoke／benchmark／阶段测试。部署只做静态语法与最小diff，最终一次新suite集中收尾。当前状态：执行中（实现）；交付根results/ua2d_transfer_20261022/r1/，logs/ua2d_transfer_20261022/r1/；最终待Codex审查，不自行归档PROJECT_HISTORY。

### UA2D-TRANSFER 最新执行状态

- 2026-10-22T11:01:41.898349+08:00；状态：待审查；阶段：执行交付。
- # UA2D-TRANSFER r1 实际交付

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

- 交付：/root/workspace/Uni-Poly-Plus-master/results/ua2d_transfer_20261022/r1；日志：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_transfer_20261022/r1；DSH执行自检不构成Codex独立审查。

<!-- UA2D_TRANSFER_EXECUTION_END -->


## UA2D-WAVG-20261022-01 / r1：当前授权与交接

- 2026-10-22T11:41:21.983172+08:00；用户本条明确授权，DSH实现／执行／监控／记录／交付，Codex规划及独立审查。基于dev／3503fbe780a5399410d6db6055305925765c3ff8及dirty工作区；新结果／日志／源码无冲突，未发现同任务或TRANSFER活跃writer，旧记录完整保留。
- 科学问题：同一35条FT轨迹的固定权重平均能否优于其全70epoch validation BEST？唯一B_FP PT step5000＋35公共head，原2-hop mean→norm2、512／6层／denominator512、纯二维；原七任务5fold、seed42+fold、scaler及10+60epochs／batch24／FP32／workers0／MSE与AdamW配方不变，无残差／STD／MIX／SP／teacher。
- 只读引用TRANSFER原inputs/，保留原plan及ua2d-transfer-input-2hop-r1 schema、5875样本映射；不复制或重建缓存，不在线fallback。prepare／公共初始化／PT均0。每条轨迹同时产出BEST、SWA4（全局epoch40/50/60/70等权）及EMA60（全局11—70，beta0.9、S0=z0=0、最终S/z）；全部浮点state使用独立detached FP32缓冲，非浮点buffer一致保留，不改变学生／optimizer／梯度／RNG。保留4份SWA纯权重及EMA最终S／z／60次记录，新checkpoint schema明确类型及来源。
- 硬上限：35FT／2450epochs／56000updates／2520validations（训练2450＋终点70）／105TEST及17625图／1最终CPU≤300秒；FT含终点评估≤7200秒，TEST≤300秒，总墙钟从首次FT启动≤12小时。仅GPU2／3，每卡FT或TEST≤4路，不混阶段；异常、覆盖／writer冲突、预算／期限耗尽即停止，无retry/resume／换卡／改batch／减epoch补救。
- 全35轨迹及终点评估完成后仅按validation在SWA4／EMA60提名FINAL，同分SWA4；BEST仅直接对照。无准入门槛，一次冻结三估计全部105个单元，再唯一正式TEST；不依TEST改选。预登记SWA4−BEST、EMA60−BEST、各估计−历史B_FP、FINAL−A_MIX246／R_MEAN；FINAL对同轨迹BEST及历史B_FP均≥0.010才性能通过。五折等权、std(ddof1)、七任务等权macro，未舍入值计算，显示3位小数；共享开发TEST及同轨迹三估计，不称独立盲测或三组独立训练。
- 当前状态：执行中（实现）；仅新增隔离训练／平均／调度／strict恢复和专项测试及本记录，旧源码、缓存、结果／预算均不改。部署只做AST／最小diff，不加probe／smoke／benchmark／阶段suite；原实际流程收尾后仅一次新CPU suite。交付results/ua2d_wavg_20261022/r1/及logs/ua2d_wavg_20261022/r1/，包含来源、冻结、全部预测／比较、计时／exit／diff和七任务SOTA数值对照，最终待Codex审查，不自行归档。

### UA2D-WAVG 最新执行状态

- 2026-10-22T12:02:42.427622+08:00；状态：待审查；阶段：执行交付。
- # UA2D-WAVG r1 实际交付

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

- 交付：/root/workspace/Uni-Poly-Plus-master/results/ua2d_wavg_20261022/r1；日志：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_wavg_20261022/r1；DSH执行自检不构成Codex独立审查。

<!-- UA2D_WAVG_EXECUTION_END -->


## UA2D-PTSCALE-20261022-01 / r1：当前授权与交接

- 2026-10-22T13:01:51.829984+08:00；用户本条完整提示词明确授权实现、执行、监控和交付；Codex科研设计及独立审查，DSH执行自检不构成独立审查。基于dev／3503fbe780a5399410d6db6055305925765c3ff8及现有dirty工作区；新根／同任务writer无冲突。WAVG执行审查通过、性能未通过以用户本条为授权依据，保留全部负结果与原交付记录。
- 科学问题：固定B_FP目标、模型及FT配方，连续增加固定5k预训练周期能否提升完整正式TEST？同一step0→20000双卡PT轨迹保存真实5000／10000／20000步P5K／P10K／P20K；不从历史step5000接续，不称三个独立PT。
- 只读引用fast_r4 prepared_cache（schema ua2d_prepared_templates_r4、911391条）、ET repair1 eligible（910756 logical slots、整条排除635失败几何）；logical=eligible[stream.index_at(global_position)]，key=keys[logical]、source_row=source_rows[logical]。一次CPU准备仅物化映射／来源并发布COMPLETE，不构图／FP／构象／dense FP。FT只读引用TRANSFER inputs原plan／schema、全部5875条原2-hop，PT筛选不传播到FT。无在线构建或缺缓存fallback。
- PT保持原B_FP mask_align／FP、初始化common_step0及V1 mask head、FP seed20261019、PT seed42；GPU2／3 world2 micro84／rank acc6 global1008、BF16及原关键FP32／DDP no_sync／全局计数归一化；每微批对比最多168。原5k LR以u%5000循环，optimizer／RNG／采样不重置，ALIGN／FP只按全局u最初1000步warmup。每rank4worker、prefetch2、bytes IPC、pinned H2D、LMDB预读；真实训练记录每步及每500步窗口loader／decode／step计时，无额外CUDA同步。
- PT正常exit0且rank／worker退出后才105FT；各P检查点七任务五折seed42+fold，从各PT及原35head初始化，原纯2D mean→norm2 O8 6层512 denominator512、70epochs／batch24／FP32／workers0／MSE及WAVG BEST原优化器配方。epoch开始预留、结束结账，history逐epoch、best严格改善、resume正常结束一次；无MIX／STD／SP／EMA／SWA等变化。
- 硬预算：prepare1≤3600秒，公共初始化0，PT1／20000updates／20160000训练呈现≤43200秒；FT105／7350epochs／168000updates／7350validations，各≤7200秒；正式TEST105／17625图，各≤300秒；新增专项CPU一次≤300秒。总墙钟首次prepare起≤24小时；失败部分计账，读取／预取另记。仅GPU2／3、FT或TEST每卡≤4路、阶段不混跑；全部长任务tmux Uni-Poly独立window、真实命令／cwd／PID／日志／exit。异常／非有限／身份冲突／覆盖／预算／期限耗尽停止，无retry/resume／换卡／改batch／减epoch补救。
- 全105FT后仅validation在P10K／P20K选FINAL，同分P10K；P5K仅同条件对照。一次冻结全部105单元及FINAL后统一唯一正式TEST，无validation准入，不依TEST改选／补训／拼任务。预登记P10K−P5K、P20K−P5K、P20K−P10K、FINAL−历史B_FP；FINAL对P5K及历史B_FP未舍入Δmacro均≥0.010才性能通过。历史池911391与新池910756不同，不将其差异单归因于时长；额外周期含额外曝光及固定循环优化。五折等权、std(ddof1)、七任务等权macro、展示3位小数；共享开发TEST不是独立盲测；七任务SOTA仅协议未证明等价下数值对照。
- 修改范围仅新增本轮数据／PT／FT／调度／strict恢复与专项测试及本记录；旧模型数学、源码、缓存、结果、预算与事故证据保留。不读TODO／远程操作／安装依赖／新增哈希指纹。部署前仅AST及最小diff；流程实际完成或安全异常收尾后新增专项CPU suite一次，覆盖logical映射、rank／采样连续、循环LR／全局warmup、预算、真实step恢复／来源；不重跑旧suite。
- 当前状态：执行中（实现）；交付results/ua2d_ptscale_20261022/r1及logs/ua2d_ptscale_20261022/r1，含三份PT来源、完整配置／history／BEST／scaler／预测／冻结／预算／计时／真实exit／源码diff／七任务SOTA，最终待Codex独立审查，不自行归档PROJECT_HISTORY。

### UA2D-PTSCALE 最新执行状态

- 2026-10-22T17:12:13.382607+08:00；状态：待审查；阶段：执行交付。
- # UA2D-PTSCALE r1 实际交付

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

- 交付：/root/workspace/Uni-Poly-Plus-master/results/ua2d_ptscale_20261022/r1；日志：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_ptscale_20261022/r1；DSH执行自检不构成Codex独立审查。

<!-- UA2D_PTSCALE_EXECUTION_END -->


## UA2D-FPDECAY-20261022-01 / r2-SG：当前授权与交接

- 2026-10-22T17:39:07.260027+08:00；授权来源为本会话用户本条完整执行提示词；Codex负责科研设计及独立审查，DSH负责隔离实现、执行、记录及交付。基于dev／3503fbe780a5399410d6db6055305925765c3ff8和既有dirty工作区。本修订替代此前FPDECAY双卡版本，不运行其他候选或历史队列；旧修改／控制器／缓存／结果／事故证据均保留。
- PTSCALE本次依据与审查记录：实际105FT／105正式TEST已结束，controller exit0、CPU exit0，validation提名P10K；正式macro P5K=0.788、P10K=0.793、P20K=0.785，P10K−P5K=+0.005、相对历史B_FP=+0.002，未达双参考0.010门槛。用户以Codex本次审查为规划依据；现有独立审查metadata仍PENDING，未提供单独审查文本，其独立审查结论未独立核实，DSH不补写“审查通过”或归档。原完整精度快照authorization/ptscale_reference_snapshot.json。
- 科学问题：FP辅助监督后期退火是否改善通用二维表征及完整七任务TEST macro？固定BASE对照、FP_DECAY预登记候选，两支相同step0／PT seed42各独立5000步，不从任何训练后检查点续训。唯一变化为FP_DECAY的原FP warmup权重乘g(t)：t≤4000为1；4000<t≤5000为0.5×(1+cos(π×(t−4000)/1000))，t为将执行update编号1—5000。ALIGN及其warmup、其他损失和模型不变；不搜索起点／曲线／系数。
- 仅物理GPU3，CUDA_VISIBLE_DEVICES=3、world1、micro168×acc6=global1008，每微批对比最多168；单卡本优化窗口目标计数归一化，无双卡乘数或额外除acc6。两个PT串行，同一卡不模拟world2。FP仅PT监督，下游纯二维。OOM或其他失败停止，不变更micro/acc／对比池或retry/resume。
- 只读引用fast_r4 prepared_cache、PTSCALE r1 inputs（保留原plan／schema，910756个eligible logical slots）、TRANSFER r1 inputs（原plan／schema、完整5875二维2hop）；635失败几何整条排除全部PT目标，eligible不作source_row。prepare=0、新公共初始化0；复用common_step0与V1 mask head，FP seed20261019；不复制／改写／重建缓存，不在线图／FP／构象／缺失fallback。单卡每micro连续168个global positions，同seed／增强及初始RNG；ordered4worker、prefetch2、bytes IPC、pinned H2D、LMDB预读，仅实际PT全程／500步计时，无额外CUDA同步或benchmark。
- 其余PT原B_FP mask_align／fusion none／FP enabled／denominator512、mask／ALIGN／精度／AdamW／5k LR固定；保存两支真实step5000 deploy、完整resume及来源。两PT正常结束且workers退出后全部70FT；ARCHITECTURE及PROFILE逐字段复用src/training/ua2d_ptscale.py，原七任务×5fold／seed42+fold、公共heads／split／train-only scaler、10head+60joint／batch24／FP32／workers0／MSE／AdamW／LR／scheduler／dropout／clip，70epochs统一严格validation择BEST，无early-stop／任务分支。
- 全70FT后一次冻结两支全部BEST／scaler／config／split／indices及预登记候选FP_DECAY，再唯一正式TEST70单元，无validation准入或TEST后改选／补训／拼任务。主要受控差值FP_DECAY−BASE；另外FP_DECAY−PTSCALE/P10K及−历史B_FP仅跨资源／预算数值参考。晋级要求对单卡BASE及P10K未舍入Δmacro均≥0.010；不能凭Nc/EPS单项晋级。五折等权std(ddof1)、七任务等权macro、完整精度计算、展示3位小数，共享开发TEST不是独立盲测；逐项SOTA对照，协议不等价只数值比较。
- 硬上限：prepare0／新公共初始化0；PT2／10000updates／10080000训练呈现，各≤21600秒；FT70／4900epochs／112000updates／4900validations，各≤7200秒；TEST70／11750图，各≤300秒；新增专项CPU仅一次≤300秒。总墙钟首次PT启动≤24小时，不重置。仅GPU3，PT串行、FT或TEST≤4路，阶段不混跑；失败／部分计账，读取／预取单独记录。所有GPU／worker／训练写入及长任务tmux Uni-Poly独立window，保存cwd／实际命令／PID／window／日志／真实exit。异常／非有限／身份或step不匹配／writer冲突／覆盖／预算或期限耗尽，保留现场停止，无扩预算／换卡／改配方。
- 仅新增本轮数据／PT／FT／调度／strict恢复及专项测试和本记录；不改旧源码、共享模型数学、缓存，不读TODO／远程／安装／哈希指纹。部署前仅AST和最小diff；流程正常或安全异常收尾后唯一新增CPU suite覆盖单卡归一化／连续采样／退火边界／初始化恢复／预算／冻结，不重跑旧suite或阶段测试。
- 当前状态：执行中（实现）；新根results/ua2d_fpdecay_20261022/sg_r2及logs/ua2d_fpdecay_20261022/sg_r2此前无冲突，同产物writer不存在，GPU3未见活跃训练。交付配置、PT来源、全部history／BEST／scaler／validation与TEST预测、冻结／预算／真实diff／全程及窗口计时／退出码／七任务SOTA。最终待Codex审查，不代替独立审查或归档；未通过即结束，不自动启动路径编码或其他搜索。

### UA2D-FPDECAY r2-SG 最新执行状态

- 2026-10-22T20:48:06.707108+08:00；状态：待审查；阶段：执行交付。
- # UA2D-FPDECAY r2-SG 实际交付

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

- 交付：/root/workspace/Uni-Poly-Plus-master/results/ua2d_fpdecay_20261022/sg_r2；日志：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_fpdecay_20261022/sg_r2；DSH执行自检不构成Codex独立审查。

<!-- UA2D_FPDECAY_SG_EXECUTION_END -->
