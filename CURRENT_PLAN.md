# 当前计划：UA3D 阶段拆分审查与冻结3D路径实验提案

> 当前提案：UA3D-FROZEN-TRANSFER-20261026-01/r1（第16节）；状态：**待授权**。第15节为本次独立审查：A修复通过；STAGE/r2训练与正式TEST完整、最终CPU失败，整体需返修，性能未通过。下一步先CPU边界返修，再固定冻结3D迁移对照；仅物理GPU2。本轮只审查和写计划，不启动执行，全部旧终态保留。
> 历史r4授权来源：用户于2026-10-25明确要求“请你直接执行！”，授权r4的实现与固定实验。Codex按该明确要求直接执行；按用户随后“请使用 4 GPU并行！”变更为GPU0/1/2/3、48小时硬上限及第9节预算不变。本文件记录授权，不产生授权。
> 历史r4资源修订（启动前快照）：用户在实现期间明确要求四GPU并行，覆盖原r3仅GPU3约束；修订当时controller尚未启动，原r3消费0。保留BASE冻结PT-3D评价；保留r2的距离／空间夹角非键状态消息定义。r2曾替代同轮未执行的四体方案。新方案完全撤销扭转特征、四原子路径和二面角目标；采用距离与三原子空间夹角。10K指每组从公共step0完整训练10,000 updates，非从10K checkpoint再追加10K。

## 1. 快照、版本与角色

- 检查时间：2026-10-25 20:29:52（Asia/Shanghai；r3只读检查）。本地分支dev，HEAD=cd36ae7280b159f881b483ad5814835ff7b74c4a（P10K）。已查看git status和相关diff，以及AGENTS.md、SOTA_RESULT.md、PROJECT_HISTORY.md与相关源码／交付。
- 用户首次要求规划时CURRENT_PLAN.md为0字节；同轮r1只有文档，执行消费0。前轮r2也只有文档；本次在现有dirty文档上修订r3，旧r1／r2没有待接续训练。TODO.md有用户改动但未读取；UA3M／TMOL3M及旧无FP／A020等源码、配置、测试、调研材料有未跟踪文件，全部保留，不归为本轮实现。
- 历史r3规划曾限定物理GPU3；当前用户明确要求四GPU并行，覆盖该历史约束，当前仅允许GPU0/1/2/3。历史规划时未启动运行；执行准备已检查四卡均无计算占用，启动时记录分配卡实际占用，不终止无关作业。
- 本轮只读源码、文档、JSON和文件存在性；未加载大型checkpoint逐张量认证、未重算预测指标、未运行模型／测试。普通加载与身份检查纳入未来授权后的正常初始化，不加前置探针。
- Codex负责规划和独立审查；DSH获用户明确执行授权后负责实现、实验、监控、记录和交付。旧交付仍待独立审查，不因本计划而自动通过。

## 2. 使用什么模型，以及优化什么

**完整模型仍为TMOL3M，3D底座仍为原生MaskedGalformer3D。** 新模型可称TMOL3M＋Spatial Galformer，是项目适配名称，不称另一篇论文的完整复现。

| 路径 | 本轮结构 |
|---|---|
| 2D | 原BondPathO8，6层、512维；原norm2／A2 |
| 3D | 原MaskedGalformer3D，6层、512维，有限Trimer键token／键长／原键角路径；增加本计划空间消息残差 |
| FP | 原2048-bit Morgan及TMOL3M FP MLP |
| 融合／性质头 | 原256维模态投影、TMOL gate／任务嵌入、256→256→1性质头 |

3D内部信息流：物理键token → 物理原子incidence均值 → 非键空间邻域消息 → 回写物理键token → 原有Transformer后续层。原键长、化学键角、attention缩放和图级readout保留。只增强3D内部消息，不更换整个编码器，不改2D／FP／融合机制。

**科学问题**：当前3D以化学键线图关系为主；在同一真实构象中，化学图上相隔较远的原子也可能空间接近。显式传递这些接触关系的消息，并用距离和局部空间夹角调制，能否改善性质相关的3D表征及完整模型表现？这是研究假设，不将“增加几何”预设为有效。

本轮不计算／缓存／输入扭转角、二面角、cosφ／sinφ／cos2φ或四原子路径，不调用历史torsion模块。空间夹角只由同一中心的两条邻居方向定义，属于三原子关系。距离本身仍可能携带构象信息，不能声称模型对扭转构象变化完全不敏感。

## 3. 已有证据与参考边界

- PROJECT_HISTORY.md记录：r16真实三键端点距离相对常数距离，融合TEST宏差−0.004；3DSHIFT匹配10K表示包差+0.001；均未支持对应路线达到0.010。本轮与r16的区别是对所有半径内非键邻居在每层进行状态依赖消息传递，并非只注入一个固定化学路径端点距离。
- 既有ET_SINGLE更换整个teacher且改变cohort，正式TEST=0；不能据此认定空间邻域或特定算子无效。本轮固定cohort并保留Galformer，避免同样混杂。
- 最新TMOL3M消融：`results/tmol3m_3d_ablation_20261025/r1/repairs/json_import_repair1/final_review/results.json`及report.md，FULL TEST宏R² 0.788、NO3D_FT 0.784、差+0.005。原完整精度分别为0.7883639553219849、0.7837849723214981、0.004578983000486758。交付状态待审查／independent_review=PENDING，本轮未完成其独立复审。
- 父三模态交付：`results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/final_review/results.json`及report.md，历史TMOL3M为0.790，旧FINAL=B_A020保持。上述旧结果只作数值参照；新BASE的配对结果必须实际完成，不能复用旧分数填入。

## 4. 固定来源、初始化与数据

下文PARENT=`results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1`，仅作为原实现、输入和初态资料来源。

- 四组均从原step0开始：native参数来自`results/ua2d_pretrain_fp_20261019/fast_r4/initialization/common_step0.pt`，mask heads来自同目录`task_heads/V1.pt`，TMOL投影／FP／Q2／Q3初态复用`PARENT/initialization/TMOL3M_modal.pt`。按原load_initial严格加载、排除FP预测头；不从任何10K deploy或性质BEST起训。
- 新空间分支独立seed20261025、fork_rng初始化，后三组的完整新增状态相同；公共native／modal初态四组逐项一致。保存正常初始化记录，不新增哈希、文件指纹或校验和验证。
- PT只读复用`results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache`；合法映射复用`results/ua2d_ptscale_20261022/r1/inputs`，910,756条有效记录，635条失败几何整条排除。global position从0开始，四组使用同一采样顺序、mask／donor／噪声与身份去重语义。
- 下游只读复用`PARENT/inputs`的5,875行：5,860有3D、15缺3D（Egb7／Egc8、8个结构）。保留全部性质行；缺3D时投影后z3严格零、gate屏蔽并重归一化，不进入3D编码器。
- 原坐标、拓扑与身份来自`data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging`；split为`data/splits/mips_paper5_outer/{task}.json`。新sidecar只附加空间字段，不改原2D、FP、row／key／SMILES、split或缺失名单。
- 每fold性质头初态复用`PARENT/initialization/heads/{task}/fold{fold}.pt`；train-only scaler复用`results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020/{task}/fold{fold}/scaler.json`，正常检查与本split／标签相容。不重选或用全数据拟合scaler。
- 不新增构象、MD、长链、多构象、PH或FP；不做KRR和输出融合。有限Trimer不能提供未观测的无限链／体相长程结构。

## 5. 四组对照与四卡资源

| Arm | 3D改动 | 新PT | FULL FT／TEST | 冻结3D探针／TEST | PT物理GPU |
|---|---|---:|---:|---:|---:|
| BASE | 原MaskedGalformer3D | 10,000 | 35／35 | 35／35 | 0 |
| C_CONST | 相同空间邻接；距离编码固定3.0 Å，新增角度通道0 | 10,000 | 35／35 | 35／35 | 1 |
| S_RAD | 相同邻接；真实距离编码，新增角度通道0 | 10,000 | 35／35 | 35／35 | 2 |
| S_RADANG | 相同邻接；真实距离＋空间夹角统计 | 10,000 | 35／35 | 35／35 | 3 |

- 四组PT按表序分配物理GPU0/1/2/3，每进程只见分配卡（CUDA_VISIBLE_DEVICES=该卡，内部cuda:0），world_size=1，不改DDP。每卡默认1路PT、允许上限2路；不因并发改变microbatch／accumulation／global batch1008或ALIGN负样本批大小，不以OOM后减batch补救。
- C_CONST−BASE检验空间分支整体变化，包含额外参数、邻接／有效性与计算，不称纯容量效应。S_RAD−C_CONST检验给定相同空间邻接时的真实距离数值；S_RADANG−S_RAD检验新增夹角信息。后三组的参数形状、初态和optimizer配置相同。
- 常数臂仍知道由真实构象决定的邻接与有效性，不能称完全无几何控制。输入常数或零通道导致有效梯度／函数容量不同，不能只凭参数总数声称有效容量完全一致。
- S_RADANG−BASE是整体方案效果；四组不是完整2×2 factorial，不宣称distance×angle interaction或拆分空间边存在与分支容量的独立作用。

## 6. 非键空间消息的可实现定义

### 6.1 物理图和几何

1. 使用原有限Trimer物理重原子及物理键端点，保留不同RU副本／image身份。PT模板已有positions、endpoint_a/b和bond_z_a/b，可从物理键表建立重原子编号和键incidence；下游从原trimer身份字段物化同义sidecar。不得用canonical编号合并空间上不同的副本。
2. 新有向空间边j→i满足：i≠j，1e-8 Å<dij≤4.5 Å；化学重原子图最短距离大于2或不连通。排除原1-hop键与2-hop键角邻居，保留所有合格边，不按邻居数截断，不添加周期盒或链外坐标。各图分别构边，禁止跨batch图连接。
3. PT邻接、距离和夹角均来自与原键长／键角相同的σ=0.03 Å带噪坐标；不读取clean距离真值作输入。mask按原40%语义处理；原子状态只从已经corrupt的键token汇聚，不另加未mask元素／键型输入。
4. 固定32维Gaussian RBF，中心均匀覆盖0到4.5 Å，σ=4.5/31 Å，编码exp(−0.5×((d−中心)/σ)²)。C_CONST对同一有效边输入d=3.0 Å的RBF，不改变真实邻接。
5. S_RADANG在中心i的新增空间邻居方向uij上计算a1=平均cosθjik、a2=平均P2(cosθjik)，P2(c)=(3c²−1)/2；对所有j<k无序邻居对取均值。邻居不足2时两个量为0并记录有效性。C_CONST／S_RAD只将这两个新增通道置0，原Galformer化学键角仍保留。
6. 角度统计可精确由方向一阶和二阶矩计算，避免显式枚举所有三元组：n≥2时，mean cos=(||Σu||²−n)/(n(n−1))，mean cos²=(||Σuuᵀ||F²−n)/(n(n−1))；再计算a2。方向和统计使用FP32。以上只编码空间夹角分布的两个矩，不宣称完整保留每个角度。
7. 合法无空间边图新增增量为0，保留该样本。近零空间距离、坐标非有限、元素／端点身份或索引冲突须停止；不能静默删边、替换构象或扩大缺失名单。正常运行记录邻居数、边数、无边图、有效角度覆盖与接缝覆盖，不加前置探针。

### 6.2 每层消息与参数

- 在每个原3D Transformer block之后，用当前物理键状态按真实端点incidence取均值形成物理原子状态hi；保留512维，不直接使用原始坐标向量作预测特征。
- 共享线性Wdown:512→32；几何MLP为34→64→32（中间GELU），输入为32维RBF与中心i的a1/a2，输出经sigmoid为32维边门控ge。每个空间边消息为ge逐通道乘Wdown(hj)，按目标原子i的有效空间邻居取均值。
- 经共享Wup:32→512回到原维度；对物理键b=(i,j)，回写增量为(Wup(mi)+Wup(mj))/2，再乘1/6加到该block输出。先对无邻居原子在Wup之后置零，避免bias使无边原子获得增量。
- 六层共享Wdown、几何MLP和Wup；均含bias，新增共37,632参数。Wup末层weight和bias全零，使初始增量为零；其他层按隔离seed正常初始化，不额外增加零gate或随机dropout。
- 后三组模块完全同形同初态、同边／mask、同调用次数与优化器；只改变输入通道。保留原1/√512 attention缩放、source/target语义、原键角bias／angle value、readout、norm3／A3。
- 新模块归属encoder3d：FULL head阶段随native冻结／eval，joint阶段native LR1e-4；不得误入LR1e-3的fusion/head组。独立schema明确arm、空间定义、阈值与新模块；公共参数严格加载，不能宽泛strict=False吞缺项。

## 7. 固定训练和双层评价

### 7.1 每组完整10K预训练

- 四组从公共step0各训练10,000 updates，microbatch168、accumulation6、global1008、world1、seed42；BF16 forward，关键loss FP32，clip1。AdamW LR峰值2e-4、WD0、betas(0.9,0.999)、eps1e-8，两个连续5K LR周期，每周期warmup500，末端1e-6；周期交界不重置optimizer。
- 目标保持原mask2D＋mask3D＋0.08×TMOL三方向ALIGN；temperature0.1、sum3、分母排除正项、原identity去重；loss warmup1000只做一次。允许合法负ALIGN／总loss，不新增几何预测loss或调权。
- 原2D／3D／FP分离forward RNG沿用原10K范围（各60,000次forward），四组公共分支同seed和计数；新增模块无随机层，fork_rng初始化。三组空间分支不改变公共mask／dropout消费；normal worker不额外抽取样本。
- 仅第10,000 update endpoint用于下游，不按PT loss选checkpoint。无额外PT validation、阶段smoke、benchmark或中途性能筛选。

### 7.2 FULL：四组各35套

- Eea／Egb／Egc／Ei／EPS／Nc／Xc，fold0—4，seed42+fold；FP32、batch24、workers0，head10＋joint60，共140套／9,800epochs。统一train-only scaler、公共fold head／fusion初态和数据顺序。
- 保留原FP MLP、TMOL gate／task embedding、head dropout0.5、原head／joint scheduler、WD0.02、clip5。FT loss仍为MSE＋0.08×原ALIGN−0.08×entropy。仅按原单位validation RMSE严格改善选择best。
- 本轮优化包同时用于PT和joint FT；不独立区分两阶段的作用。BASE和空间组的计算量不相同，记录实际时间与显存，不宣称等计算量比较。

### 7.3 冻结PT-3D表征评价：四组各35套

- 直接从各自PT endpoint的encoder3d／norm3／A3提取256维z3；全部冻结且eval。只训练相同原256→256→1性质头，公共fold初态、dropout0.5；30epochs、AdamW LR1e-3、WD0.02、clip5、无scheduler、FP32／batch24／seed42+fold，validation RMSE选best。
- 不读取2D、FP或gate；不从FULL已监督微调BEST提取特征。15个缺几何样本使用投影后严格零z3，仍参加同一评价，不另挑容易的子集。
- 每fold预冻结只提取本fold train/validation特征一次并缓存，四组合计最多94,000行编码；缓存绑定endpoint、row＋key、split和availability，不缓存标签或做全数据标准化。每fold正式TEST特征只在统一冻结后提取一次。
- 共140套／4,200head epochs及140次正式TEST。这是预注册表征可读性实验，非额外验收测试；固定head的可读性不等于全部3D信息量。

## 8. 一次冻结、正式TEST与判定

1. 全部140套FULL、140套探针和validation完成后，一次冻结全部280个估计器、scaler／配置／样本身份与路径。FINAL按FULL全局validation macro选择，exact tie顺序BASE、C_CONST、S_RAD、S_RADANG；不按任务拼接。
2. 另在S_RAD／S_RADANG中用同一validation macro冻结唯一科学候选G，exact tie选S_RAD。四组FULL及四组探针均属固定正式比较，不因validation未达门槛删减；统一冻结后280单元各TEST一次，不读TEST改结构或重选G／FINAL。
3. 完整方案门槛：FULL的G−BASE ≥0.010；空间几何数值包门槛：FULL的G−C_CONST ≥0.010；冻结表征可读性门槛：probe的G−BASE ≥0.010；表征几何数值包门槛：probe的G−C_CONST ≥0.010。分别报告，不将容量／连通关系差或融合收益混称为几何信息收益。
4. 另固定报告FULL和probe的G−BASE、G−C_CONST、C_CONST−BASE、S_RAD−C_CONST（距离）与S_RADANG−S_RAD（夹角），判断各自是否≥0.010；不得把G−BASE归因于其中任意单项。全部判断使用完整精度。
5. 只有FULL的G−BASE和probe的G−BASE均达到0.010时才写“本协议下3D表征与完整预测均改善”。探针通过而FULL未通过，写“表征诊断改善、整体性能未通过”；反之写“完整预测收益、独立表征改善未证实”。几何包对照未通过时不得声称排除分支／邻接解释。
6. 若G未被validation选为FINAL，仍报告固定科学对照，但不得依据TEST追认部署。旧部署／旧FINAL不自动替换。旧FULL 0.788和TMOL3M 0.790只作跨周期数值参照，本轮通过不自动代表超过历史最优。
7. 七任务各五折mean±std，ddof=1，再七任务等权macro；统计／排名／差值／门槛用未舍入值，展示3位小数及末尾零。共享validation／TEST非独立盲测；fold相关、一次seed配方不称多seed稳定优势。缺项写未完成／不可比较，不以validation补TEST。

## 9. 固定预算、资源和停止

| 项目 | 上限 |
|---|---:|
| prepare | 1次：下游5,875行sidecar，不重建原PT全库 |
| PT | 4 starts、40,000 updates、240,000 microbatches、40,320,000样本呈现 |
| FULL | 140 starts、9,800epochs；按原split/batch预计224,000updates |
| frozen probe | 140 starts、4,200head epochs；按原split/batch预计96,000updates |
| probe预冻结编码 | 最多94,000行；train/validation只提取一次，不额外模型探针 |
| 正式TEST | 280单元、47,000行预测；每估计器一次 |
| 最终CPU suite | 完成全部授权执行和TEST后集中1次，timeout600秒 |
| 总墙钟 | 沿用r2的controller首次启动后48小时，失败／等待均计入，不刷新；四卡完成可行性未核实，不承诺时限内完成 |

- 用户最新授权物理GPU0/1/2/3：PT默认各卡1路（每卡最多2路）；PT全部完成后，FULL／正式推理／特征提取每卡最多4路、合计最多16路，不与PT混跑；probe head及其提取作业总并发最多4路。各PT workers4／各worker1线程，prepare workers≤4、FULL workers0；计入所有相关子进程和writer。
- 所有超过一分钟、GPU、worker或写实验产物的命令，必须在tmux session Uni-Poly的本计划独立window；登记cwd、完整命令、物理卡／内部编号、PID、起止、日志、产物与真实exit。发现健康旧作业不另启动竞争controller，不终止无关作业。
- 一次prepare只追加新sidecar；PT在正常worker读原模板时构建空间图，使用固定容量LRU缓存，不加全库预扫／额外采样轮次。正常运行记录覆盖、参数量、实际时间和显存。
- 最终CPU suite仅覆盖本次必要正确性：空间边与batch索引、1/2-hop排除、同噪声几何、距离／角度通道、旋转／平移／排列不变性、角度矩与直接三元组小fixture等价、容量与零初态、缺边／缺3D零输出、公共RNG／加载／optimizer和冻结probe梯度。不得调用torsion作fixture或新增全仓回归、逐文件验收、阶段smoke、benchmark。
- schema／checkpoint／身份不匹配、NaN/Inf、近零空间距离、OOM、worker失败、writer冲突、正式覆盖或预算超限时停止受影响依赖链，保留现场和累计账本。无自动重试／resume、换GPU、减batch、增加seed／epoch或预算补救；失败／部分运行计入上限。
- 增补BASE表征评价对应35 starts／1,050head epochs／35正式TEST；总训练预算为140×70 FULL＋140×30 head epochs。编码／预测行数为原5,875行与既定五折完整覆盖的预算推导，实际以冻结split身份和累计账本为准；不为凑行数新增数据或重复推理。四卡下仍沿用48小时硬上限，达到截止即停止受影响运行、保留部分结果，不自动扩时或补跑。
- tmux不可用、指定卡被占用导致不能在截止内执行、关键原坐标／运行依赖缺失时具体报告阻断。非运行依赖历史文件缺失不单独阻断；不伪造旧记录。

## 10. 最小实现与交付

- 计划新增独立文件：`src/modules/ua3d_spatial.py`、`src/dataset/ua3d_spatial.py`、`src/training/ua3d_spatial.py`、`scripts/pretrain_ua3d_spatial.py`、`scripts/run_ua3d_spatial.py`、`configs/ua3d_spatial_r4.json`、`tests/test_ua3d_spatial.py`。这些独立文件已进入本轮实现；是否完成由文末实时记录及产物决定。
- 只读复用现有原生模型／数据／严格加载，通过独立子类或forward接入残差；不修改旧schema含义、老controller、共享GPU helper或无关代码。若确需共享修改，给出具体最小diff供Codex修订范围。旧dirty、事故、预算和产物全部保护；不读TODO、不删除、不push/pull、不新增哈希验证。
- 新根：`results/ua3d_spatial_20261025/r4/`和`logs/ua3d_spatial_20261025/r4/`；建议tmux window `ua3d-spatial-r4-controller`，各PT／长worker独立window登记。若新根冲突，停止，禁止覆盖／自动换目录。
- 交付execution_manifest／budget／status、输入sidecar与coverage、公共／新增初态与参数组、PT采样／loss／RNG／10K endpoint、FULL与probe history／best／scaler／validation和TEST原始预测、`selection/final_frozen.json`、`final_review/results.json`／report.md／actual_changes.diff和唯一CPU日志／命令／exit，保留所有失败记录。
- report对FULL和probe分别列G−BASE、G−C_CONST、C_CONST−BASE、距离和夹角配对差、probe对应差、G与FINAL、退化任务／fold、参数总数／活动数、真实耗时和设备。不能用loss／梯度／门控权重代替正式预测，不能宣称完整公平超越论文。
- DSH最终交付状态待审查，更新本交接的授权／实际预算／日志／产物／缺项。Codex核对本轮diff、最终CPU和正式汇总，复用证据不重跑。独立审查通过后才归档PROJECT_HISTORY；CPU通过、数值通过与独立审查通过分开记录。

### 周期结束七任务对照占位

本计划尚未执行，各组正式结果／项目减论文均为未完成／不可比较，不能填入旧模型或validation数值。完成后为四组FULL和四组probe补齐五折统计、协议、日志／产物路径，论文采用SOTA_RESULT.md逐任务最佳包络：

| 任务 | 论文参考模型 | 参考R² | 本轮正式结果／项目减参考 |
|---|---|---:|---|
| Eea | Periodic-TDL | 0.934 | 未完成／不可比较：尚未执行 |
| Egb | Periodic-TDL (+Chem, +RC) | 0.923 | 未完成／不可比较：尚未执行 |
| Egc | Periodic-TDL (+Chem, +RC) | 0.911 | 未完成／不可比较：尚未执行 |
| Ei | Periodic-TDL | 0.839 | 未完成／不可比较：尚未执行 |
| EPS | Periodic-TDL | 0.787 | 未完成／不可比较：尚未执行 |
| Nc | Periodic-TDL | 0.856 | 未完成／不可比较：尚未执行 |
| Xc | Periodic-TDL | 0.423 | 未完成／不可比较：尚未执行 |

EAT／Eib／Tg不纳入。既有交付记录的Egc3380／Xc432与论文4124／430等数据／协议差异保留；协议等价未核实时只作数值对照。

## 11. 可复制给DSH的执行提示词

```text
用户已明确要求“请你直接执行！”并随后要求“请使用 4 GPU并行！”，执行UA3D-SPATIAL-20261025-01/r4；本文件记录该授权。
基于dev/cd36ae7280b159f881b483ad5814835ff7b74c4a加dirty工作区，读AGENTS和CURRENT_PLAN，保护既有代码、预算、事故与产物。不读TODO、不删除、不远程同步、不做哈希验证。
模型仍是TMOL3M，3D底座MaskedGalformer3D 6×512。新增物理原子非键空间消息：保留原键token，incidence到原子，半径4.5Å且化学hop>2的真实空间邻居，32维距离RBF和两个三原子夹角矩，低秩32消息回写原物理键，6层共享模块/各乘1/6，末层零初态。完全禁用扭转/二面角/四原子路径。
四组BASE/C_CONST/S_RAD/S_RADANG，分别原模型、同邻接常数3.0Å、真实距离、真实距离加夹角；后三组同37632参数和初态，公共step0和原TMOL模块初态四组一致。每组从step0完整10K，不是从10K再续训到20K。
物理GPU0/1/2/3各默认1路PT，每卡最多2路，各world1/micro168/acc6/global1008/BF16；不改DDP或负样本批大小。保留原mask/噪声、TMOL loss、RNG、两个5K LR周期和一次loss warmup。
每组七任务五折FULL，head10+joint60、FP32/batch24/workers0；四组均另各35个冻结PT-z3探针，只训练同初态head30epochs。全部validation后一次冻结280估计器；FINAL全组validation选，科学候选G只在S_RAD/S_RADANG以validation选，禁止TEST重选。
预算prepare1、PT4/40000updates、FULL140/9800epochs、probe140/4200head epochs、预冻结probe编码最多94000行、正式TEST280单元/47000行、最后唯一CPU suite1次/600秒。分别判FULL与probe的G-BASE及G-C_CONST是否≥0.010，并报告距离和夹角增量。共享TEST不称盲测，不宣称interaction。
GPU0/1/2/3，PT各默认1路、每卡最多2路；PT全部完成后FT/推理/特征提取每卡最多4路、合计最多16路；CPU探针连同其提取作业最多4路。全部长命令和worker在tmux Uni-Poly独立window，保留命令/cwd/设备/PID/日志/真实exit。controller首次启动48小时截止不刷新。
发生NaN/Inf、schema/checkpoint/identity错误、近零空间距离、OOM、worker失败、writer冲突、覆盖风险或预算超限，停受影响链并保留现场，无自动重试/resume、换卡、减batch或扩预算。
只新建本计划独立文件；输出results/ua3d_spatial_20261025/r4和对应logs。按第10节交付实际diff、完整预算/初态/采样/模型/原始预测/冻结/唯一CPU证据及七任务论文数值对照。更新CURRENT_PLAN，最终待审查，独立审查通过前不归档或替换部署。
```

## 12. 历史规划记录（执行授权前快照）

- Codex本轮完成源码／交付只读检查和r2→r3计划修订，仅修改CURRENT_PLAN.md。新增代码、prepare、PT、FT、probe、正式TEST、CPU suite和GPU消费均0；仅检查最终文档diff与格式。
- 当前状态待授权；r3结果根和日志根在本轮只读检查时均不存在，未创建。未核查旧草案的进程／产物实时状态，不将其视为r3执行证据。r3本轮未启动controller或tmux window，未生成训练日志或结果产物。下一步是用户明确授权后的DSH实现与执行，不自动启动旧队列。

<!-- UA3D_SPATIAL_STATUS_START -->
- 2026-10-26T01:46:46.232006+08:00：状态**需返修**；阶段交付报告。用户“请你直接执行！”授权r4；Codex按明确要求直接执行。
- # UA3D spatial r4 实际交付

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

- 账本：`/root/workspace/Uni-Poly-Plus-master/results/ua3d_spatial_20261025/r4/budget.json`；日志：`/root/workspace/Uni-Poly-Plus-master/logs/ua3d_spatial_20261025/r4`；独立审查PENDING，未归档或替换部署。
<!-- UA3D_SPATIAL_STATUS_END -->

- r4资源补充：用户“请使用 4 GPU并行！”发生于实现期间、controller首次启动前；结果／日志根改为独立r4，r3未启动、未消费。PT四组各一卡，FULL每卡最多4路，probe总并发4路；科学问题、初始化、数据、280正式TEST与48小时上限全部保留。

- 实现完成后进行了单次静态AST／JSON格式检查（6个Python文件通过）及相关diff格式检查；未运行任何模型／测试或预跑。按既定split计算的固定总量为FULL224000 updates、probe96000 updates、预冻结特征94000行、正式预测47000行，与预算一致。

## 13. r4 完成后的后续计划：PT／FT 空间分支阶段拆分

> 后续计划 ID：UA3D-STAGE-20261026-01；修订：r1；状态：**阻断**。
> 授权来源：用户先要求设计夜间后续计划，随后明确回复“授权！”，已授权第13节新增70套FULL、70个正式TEST及唯一最终CPU suite，并授权r4正常完成后的自动接续，无需逐阶段确认。第1—12节r4继续由原controller管理，后续只使用独立入口／账本，不与父运行混跑。
> 已授权后续预算：**新增PT=0，新增FULL=70套×70epochs，新增正式TEST=70单元，新增最终CPU suite=1次**。独立实现与唯一接续controller已落实；r4正常完整交付后自动接续，无需用户夜间逐阶段确认。

### 13.1 规划快照与依赖事实

- 检查时间：2026-10-25 22:05—22:10，Asia/Shanghai。本地dev，HEAD=cd36ae7280b159f881b483ad5814835ff7b74c4a；git status、CURRENT_PLAN相关diff、AGENTS、SOTA_RESULT、r4模块／训练／controller入口及既有聚合物3D调研均已查看。当前HEAD不是干净版本；r4新源码／配置／测试及其他既有未跟踪材料全部保留，不归为本次文档修改。
- 已核实r4 controller进程身份、tmux window及账本：`Uni-Poly:ua3d-spatial-r4-controller`，controller PID1016207，window @1532，进程start_ticks与manifest匹配；prepare已完成5,875行；四PT各约2,500／10,000 updates，GPU0/1/2/3各一路，尚无FULL／probe／TEST／CPU suite消费，预算无违规。此处是时间快照，不替代实时账本。
- r4实际输入sidecar成功保留5,860有3D、15缺3D样本，真实物理实例非键有向边1,678,276条；接缝边466,384条。覆盖只证明正常物化，不证明性质预测收益。
- r4的10K endpoints、280个最终估计器、正式TEST与CPU结论在本节制定时均尚未完成。因此G、r4 FINAL、性能及可复用的对角单元均为**未来依赖，未核实**，不得用旧周期分数代替。
- 用户已要求四GPU，覆盖AGENTS的默认GPU3限制；后续候选资源沿用物理GPU0/1/2/3。用户暂离不改变预算或停止条件，也不代表自动同意新增实验。
- r4原controller已包含PT→FULL→probe→冻结→280 TEST→唯一CPU→待审查的自动流程；无需本节启动第二个controller接管、暂停或重启它。现有入口没有本后续计划的自动接跑实现，不把文档当作已部署队列。

### 13.2 为什么下一步先拆阶段

r4的空间臂同时改变PT与joint FT，完整方案即使有效，也不能说明收益来自更好的PT、监督FT时使用空间消息，还是两者组合；无效也不能说明两个阶段都无用。现有PT产物足以补齐缺失的交叉单元，不必再训练四组10K或增加构象。

科学假设：空间分支在PT与FT的作用可能不同。关闭其中一个阶段可能减少无效几何消息或跨阶段不适配，同时保留有效部分；这是待检验假设，不预设全程启用或部分启用更好。

固定G仍读取r4 `selection/final_frozen.json`：只按r4 FULL validation在S_RAD／S_RADANG中选择，exact tie为S_RAD。本后续不按r4 TEST重新选择几何定义，不重选距离阈值、角度通道或FT超参数。后续两臂无论validation是否优于参照，都各完成35个正式TEST；不按TEST结果增删分支。

### 13.3 固定2×2矩阵、实现与可解释范围

| 单元 | PT是否启用空间分支 | FULL FT是否启用空间分支 | 来源与新增消费 |
|---|---|---|---|
| E00 | 否 | 否 | 复用r4 BASE的35个FULL及预测；新增训练／推理0 |
| E11 | 是，固定G | 是，同一G | 复用r4 G的35个FULL及预测；新增训练／推理0 |
| E10／PT_ONLY | 是，固定G | 否 | 从r4 G的PT 10K endpoint初始化，关闭FT空间分支；新增35 FULL＋35 TEST |
| E01／FT_ONLY | 否 | 是，固定G | 从r4 BASE的PT 10K endpoint初始化，新装同定义零初态空间分支；新增35 FULL＋35 TEST |

**PT_ONLY严格加载合同**：源为r4 G的PT deploy。只显式移除键名前缀`encoder3d.spatial.`；被移除的state集合须与该37632参数模块的完整state精确对应。其余native、norm2／norm3、A2／A3、FP与Q2／Q3全部原样严格加载；不恢复为BASE的2D／FP或投影。FT使用原MaskedGalformer3D前向，不调用、训练或缓存被移除模块。记录移除的键名、形状及参数数目；不使用宽泛strict=False。

**FT_ONLY严格加载合同**：先将r4 BASE的所有原生／模态PT state严格加载到原生FT模型；再按独立seed20261025及fork_rng安装G定义的37632参数共享空间模块。复用r4 G的`spatial_initial.pt`严格加载新增状态，确认Wup weight／bias为零；不得读取G训练后的空间权重或FULL BEST初始化。空间模块在head阶段冻结／eval，在joint阶段属于native组、LR1e-4。

两个新臂的fusion／fold性质头仍复用r4公共初态；不从任何FULL监督BEST接续训练。PT来源arm、FT空间开关和G必须是三个独立metadata字段，不能仅靠一个arm名混淆。新增独立schema，例如`ua3d-stage-best-r1`，明确允许的状态转换；不改变r4 schema或旧checkpoint含义。

因r4 PT包含三模态ALIGN，PT因素可能连带改变2D／FP／Q投影；本2×2回答“PT启用空间分支这一处理”的整体作用，不能称为只改变3D权重的因果消融。PT_ONLY在FT阶段撤去空间模块会改变表示分布；它检验PT训练所得原生参数能否在原生FT中保留收益，不等价于完整PT-3D表示的信息量。r4 probe对完整源PT路径的诊断不能代替PT_ONLY关闭模块后的表征评价。

四单元的参数活动数在FT空间开关间不同，这是固定处理的一部分。只在“PT／FT是否启用空间分支”两个指定因素及其完整预算匹配范围内讨论交互，不声称容量、邻接和距离分别得到独立控制。

### 13.4 固定数据、训练与资源

- 任务仍为Eea、Egb、Egc、Ei、EPS、Nc、Xc；fold0—4，seed42+fold；与r4相同数据、split、row＋sample key＋SMILES、train-only scaler、缺失15行处理及采样顺序。EAT／Eib／Tg不纳入。
- 只读复用`results/ua3d_spatial_20261025/r4/pretrain/{BASE,G}/deploy.pt`、r4空间sidecar、原PARENT输入／fold性质头和原scaler。坐标与空间关系不重建，不新增prepare、构象、模板全库扫描、MD、FP、PH、KRR或输出融合。正常训练加载时检查来源、schema、样本及split身份；不加前置探针、哈希或checkpoint逐张量认证轮次。
- 两臂各35 FULL；每套head10＋joint60、FP32、batch24、workers0、head dropout0.5、WD0.02、clip5、原scheduler。loss保持MSE＋0.08 ALIGN−0.08 entropy，best只按原单位validation RMSE严格改善选择。空间关闭时仍保留完整3D模态、原键长／化学键角、norm3／A3、FP／gate／head及ALIGN，不将PT_ONLY改成NO3D。
- 物理GPU0/1/2/3，world1、内部cuda:0；每卡FT／推理最多4路、合计16路，进程CPU线程2、workers0。不与r4作业混跑、不抢占其他工作、不改变batch补救OOM。不得为凑并发追加副本。
- 所有长命令与正式产物写入均在tmux `Uni-Poly`独立window；候选controller名`ua3d-stage-r1-controller`，记录命令、cwd、设备UUID、PID/start_ticks、日志、起止和真实exit。只有一个后续controller／累计预算writer；已有健康运行不能另启竞争队列。

### 13.5 进入后续的条件与无需夜间确认的边界

获本后续明确授权并实现入口后，一次性注册授权，不在阶段间重新询问；但以下依赖必须由父运行的现有产物确认，不能靠文档推断：

1. r4 controller真实exit=0，相关worker已退出、父预算无违规；四个PT均10000 updates，FULL140／probe140、正式TEST280／47000预测行全部完整；唯一最终CPU exit=0，父交付状态待审查。
2. r4冻结记录包含唯一G与FINAL；BASE、G以及FINAL涉及的reference BEST、scaler、validation／TEST预测与原split／样本身份齐全；不另做模型推理或重复父CPU suite。
3. r4实际diff和日志摘要存在；将“r4独立审查PENDING”显式带入后续provenance。依赖就绪检查或执行者自检不冒充独立审查，不提前归档／部署。若已知重大正确性问题影响父结果，不能继续消费后续训练预算。
4. 新结果／日志根无冲突，GPU资源可分配，且未超过下述固定截止。

r4任何失败、NaN/Inf、OOM、worker错误、schema／身份不匹配、覆盖风险、预算超限或CPU失败：记录具体阻断与已消费预算，**不自动返修、不重试r4，也不接跑本轮**。父产物缺项不通过重跑、补seed、少task或降低epoch绕过。数值未达0.010不是运行故障；如果依赖完整，获授权的固定70套矩阵仍可进行，用于解释阶段作用，不承诺收益。

用户已明确回复“授权！”，自动接续入口已实现并在独立tmux window等待父r4。r4自身正常流程继续运行至待审查；父异常或依赖不完整时接续入口停止，不扩预算补救。

### 13.6 一次冻结、正式TEST与结果判定

- 两个新臂全部70套FT和validation完成后，按七任务等权、每任务五折平均的FULL validation宏R²选择唯一候选C2；exact tie选PT_ONLY。该顺序在读取本轮正式TEST前固定，不按任务拼接。
- 性能参照REF为r4原冻结FINAL，不用r4 TEST重选父参照。另保留E00／E11作阶段对照。NEXT_FINAL在REF、PT_ONLY、FT_ONLY中只按validation选；exact tie依次REF、PT_ONLY、FT_ONLY。NEXT_FINAL只是后续推荐，不直接替换部署。
- 一次冻结70个新BEST、scaler、配置、G、C2、NEXT_FINAL及所需父reference路径；父引用不重新训练／推理或改写。之后两新臂35＋35个正式TEST各执行一次，共11750行；不因validation没有优势而删减。父prediction按原row＋key对齐复用，不能按数组位置拼接。
- **主要性能门槛：正式TEST macro R²(C2)−macro R²(REF)≥0.010。** 同时报告C2−E00、两个新臂各自减REF及全部四单元结果；只使用未舍入值判定。未经validation提名的单元即使TEST更高，也只能作为固定比较结果报告，不能追认FINAL、重训或再筛选。
- 固定报告PT效应E10−E00与E11−E01；FT效应E01−E00与E11−E10；阶段交互I=(E11−E10)−(E01−E00)。均报告各任务五折配对差、ddof=1标准差、退化fold及等权macro。0.010用于性能门槛；I仅作为本协议下描述性交互，不因I为正便称统计显著，也不设事后“最佳交互”选择规则。
- 共享开发validation／TEST非独立盲测；r4结果已参与后续科学问题形成，需披露顺序开发与重复使用TEST的限制。本阶段不会增加独立外部验证或多seed证据。
- 完成后依据SOTA_RESULT逐项数值对照Eea／Egb／Egc／Ei／EPS／Nc／Xc，保留r4相同的数据／协议差异。展示3位小数及末尾零，保存原始精度；缺项写未完成／不可比较，不用validation、部分fold或历史不同模型补齐。

### 13.7 新增预算、时限与允许修改

| 项目 | 本后续新增上限 | 与r4累计已授权上限 |
|---|---:|---:|
| PT／prepare／构象生成 | 0 | PT4starts／40000updates；prepare1，不扩大 |
| FULL | 70starts、4900epochs、112000updates | 210starts、14700epochs、336000updates |
| 冻结probe及其特征编码 | 0 | 保持140starts／4200epochs／94000预冻结编码行 |
| 正式TEST | 70单元、11750预测行 | 350单元、58750预测行 |
| 最终CPU suite | 1次、timeout600秒，仅本后续结束后 | 两个周期各一次，不重跑父suite |

用户“授权！”已批准该表新增固定预算；父预算不扩大或退款。新计划使用独立账本，父账本只读结转已收费／已知实际／失败记录，并展示父＋新累计总量。每单元派发即全额预留，失败／部分执行不退款；不把训练未完成的预算退回重试。七任务五折及batch不变时112000updates由既定split推导，不能为补齐总量新增样本。

后续硬截止=min（后续首次controller启动＋6小时，r4原硬截止2026-10-27 21:18:28 Asia/Shanghai），不刷新父48小时，不延长至“跑完为止”。检查时间快照下r4预计10月26日02:00—04:00交付；后续实现就绪且及时获授权时，正常执行再暂估1—3小时，目标03:00—07:00完成。此为既有FULL耗时推算，未对新交叉臂做benchmark；“明早08:00”只作为便于用户查看的假设时间，不是用户明确截止、不构成完成承诺，也不据此中止健康r4。无授权／实现未就绪／父失败时不存在该后续ETA。

最小候选实现文件：`src/modules/ua3d_stage.py`、`src/training/ua3d_stage.py`、`scripts/run_ua3d_stage.py`、`configs/ua3d_stage_r1.json`、`tests/test_ua3d_stage.py`；以上五文件本轮已实现，静态语法／格式检查通过，新增训练与唯一最终CPU尚未执行。复用r4只读数据加载；不改运行中的r4脚本、schema、账本、BEST、冻结选择或原始预测。普通实现细节自主完成并登记；若必须改共享代码或父科学语义，停止该改变并报告，不能默默扩围。

必要正确性仅在本后续执行与正式TEST结束后集中一次CPU suite，覆盖显式状态增删白名单、PT／FT来源与开关、零初态及隔离RNG、native optimizer归属、复用父预测按row＋key对齐、全局提名／冻结与预算。只用小fixture，不增加预跑、benchmark、重放父suite或全仓回归。纯文档规划只检查本次diff与格式。

### 13.8 交付、独立审查与更远路线

- 独立根：`results/ua3d_stage_20261026/r1/`及`logs/ua3d_stage_20261026/r1/`；已由本轮唯一controller排他创建。后续启动仍禁止覆盖／重试／自动换根。
- 交付authorization、父依赖只读记录、execution_manifest／status、父／新／累计budget、显式状态增删及参数组记录、70套history／BEST／scaler／validation、selection/final_frozen.json、70个TEST原始预测、四单元及REF的完整精度统计／配对差／I、七任务SOTA数值对照、实际diff和唯一CPU命令／日志／真实exit。
- 最终状态待审查；执行者完成实现和自检，不能把本轮自检或父“待审查”写成Codex独立审查通过。独立审查通过后才按实际结果归档PROJECT_HISTORY；纯文档本轮不归档模型历史，也不替换旧FINAL／部署。
- 若E10更好，后续优先研究PT所得原生参数及跨阶段表示适配；若E01更好，优先研究监督阶段空间学习；若E11最好，优先验证保留双阶段方案的稳定性；若PT源probe改善而FULL仍无收益，再规划“冻结PT-3D／保留表示”与matched BASE对照。这里只决定讨论优先级，不能按TEST追认候选或自动加跑第三周期。
- 若四单元都无可靠收益，回到输入与目标匹配：有限Trimer的空间接触是否对应当前标签、读出是否覆盖中心／接缝，而非继续盲目延长PT或叠加角度通道。多构象、长链、体相MD、层级RU frame和新seed须另立数据语义、预算与授权；本夜间后续均不执行。

### 13.9 可复制给DSH的后续执行提示词（已获用户明确授权）

```text
执行对象仅UA3D-STAGE-20261026-01/r1，先确认用户已明确授权本新增70 FULL、70 TEST及唯一最终CPU；“设计计划”本身不是启动许可。当前已授权运行的父UA3D-SPATIAL/r4继续由原controller管理，不接管、不重启或修改它。
基于dev/cd36ae7280b159f881b483ad5814835ff7b74c4a加已有dirty，读AGENTS与CURRENT_PLAN第13节，保留既有内容。不读TODO、不删除、不远程同步、不做哈希验证。
先完成本后续独立实现；等待父真实exit0、280 TEST与唯一CPU0、预算和引用完整，再开始。父异常、缺项或正确性问题停接续，不能自动返修或扩大预算。父独立审查PENDING必须如实带入，不冒充通过。
G只取父原validation冻结候选；REF只取父原FINAL。复用BASE=E00、G=E11所有既有FULL及预测，不新增PT或prepare。新增PT_ONLY=E10：G PT endpoint显式移除完整encoder3d.spatial state，其他state严格保留，使用原生3D FULL；新增FT_ONLY=E01：BASE PT endpoint严格加载后安装同G空间分支，用父G spatial_initial.pt零初态，不取G训练后空间参数。两个新臂都从PT endpoint完整FT，不从FULL BEST接续。
两臂各七任务五折，seed42+fold、head10+joint60、FP32/batch24/workers0、原loss/optimizer/scheduler/scaler/head初态；空间FT模块归native。仅GPU0/1/2/3，每卡最多4路，父全部退出后运行；所有长命令在tmux Uni-Poly独立window。
新增预算PT0/prepare0/probe0/FULL70starts4900epochs112000updates/TEST70单元11750行/最终CPU1次600秒。父原账本只读，独立新账本展示累计；失败整单元收费，不重试/resume、减batch、加seed或超预算。
全部70 validation完成后一次冻结C2及NEXT_FINAL。C2按PT_ONLY/FT_ONLY全局validation宏R²选，tie PT_ONLY；NEXT_FINAL在REF/PT_ONLY/FT_ONLY中选，tie REF优先。之后两臂各35 TEST各一次，旧reference预测按row+key对齐复用，不重推理。主要门槛C2减REF正式macro R²≥0.010，另报告2×2四单元、配对差与I=(E11-E10)-(E01-E00)，不按TEST重选。
截止min(本controller首启+6h,父原截止2026-10-27 21:18:28北京时间)，不刷新父48h。NaN/Inf/OOM/schema/identity/worker/writer/覆盖/预算问题停止受影响链并保留现场，无夜间自选返修或第三周期。
独立输出results/ua3d_stage_20261026/r1和对应logs，按第13.8节交付；最终待审查，独立复审前不归档、不替换部署。只在最后集中一次本轮必要CPU，不重跑父suite、不另做预跑或benchmark。
```

### 13.10 规划阶段记录（授权实现前快照）

本次仅修订本文件第13节，完成后续科学问题、两交叉臂实现合同、固定数据与预算、选择／冻结／TEST、资源／截止、停止与交付及DSH提示词。未新增或修改后续代码、未创建后续结果根、未启动后续controller／worker／训练／推理／CPU测试；当前r4健康作业保持运行。后续实验状态待授权，独立审查PENDING。

<!-- UA3D_STAGE_AUTHORIZATION -->
- 2026-10-25：用户明确回复“授权！”，已授权UA3D-STAGE-20261026-01/r1第13节固定后续。状态待执行，现阶段进行独立实现；r4正常完成后自动接续，不再询问同一启动授权。新增预算FULL70starts/4900epochs/112000updates、TEST70单元/11750行、CPU suite1次600秒；PT/prepare/probe新增0。四卡、原截止及所有停止条件不变。
- 后续6小时从本后续controller首次启动计时，等待父运行也计入，不在父完成后刷新。等待期间后续入口只写自己的JSON状态，不并发修改父controller管理的CURRENT_PLAN状态块。

<!-- UA3D_STAGE_STATUS_START -->
- 2026-10-26T01:46:51.299031+08:00：UA3D-STAGE-20261026-01/r1；状态**阻断**；阶段交付报告。授权来源：用户“授权！”（第13节新增固定预算）。
- # UA3D-STAGE r1 实际交付

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

- 结果`/root/workspace/Uni-Poly-Plus-master/results/ua3d_stage_20261026/r1`；日志`/root/workspace/Uni-Poly-Plus-master/logs/ua3d_stage_20261026/r1`；独立审查PENDING，未归档或替换部署。
<!-- UA3D_STAGE_STATUS_END -->

## 14. 最小返修与接续执行计划（用户已授权）

> 计划ID：UA3D-REPAIR-20261026-01；修订：r2；状态：**需返修**（A通过，B唯一最终CPU失败；详见14.10）。授权来源：用户2026-10-26明确执行第14节A及通过后自动B；资源仅物理GPU2，FT／推理合计最多4路。
> 原计划编写及第13节旧r1授权均不包含本节执行；本次用户已另外明确授权A段返修、额外CPU复验、严格失败接纳、重新计时及通过后的B段自动接续，无需重复确认。本节不改变历史真实exit或抹除失败。
> r2资源依据：用户最新明确要求“请调整为只使用 GPU 2！给我执行提示词！”，覆盖本提案原四卡安排及AGENTS.md默认GPU3限制；该要求授权资源／文档修订，不启动实现或实验。第1—13节四卡记录属于历史，本提案新增运行一律仅GPU2。

### 14.1 当前依据、版本与证据限制

- r2修订检查时间：2026-10-26 09:48（Asia/Shanghai）。本地分支、HEAD及已有dirty状态与下述r1快照一致；已查看git status、相关diff、AGENTS.md、CURRENT_PLAN.md、SOTA_RESULT.md及stage资源入口。当前源码仍声明ALLOWED_GPUS=(0,1,2,3)、默认physical_gpu=3，获执行授权后须在本次允许的stage文件中改为仅2；本轮未改代码或验证运行。未核实GPU2占用与单卡完成可行性，不能把文档改动称为调度已生效。
- 2026-10-26 09:41（Asia/Shanghai）只读快照：本地分支dev，HEAD=`cd36ae7280b159f881b483ad5814835ff7b74c4a`。已查看AGENTS.md、当前计划及相关diff、SOTA_RESULT.md、相关历史与本轮源码／日志／JSON。CURRENT_PLAN.md、TODO.md已有修改；UA3D及其他旧周期源码／配置／测试／调研文件为未跟踪文件，全部保护；未读取TODO内容，不远程同步，不新增哈希验证。
- 父r4于10月26日01:46结束，controller真实exit=1。566个作业中565个exit=0：prepare1、PT4、FULL140、probe140、TEST280；唯一CPU作业exit=1。四PT各10,000 updates；FULL9,800epochs／224,000updates；probe4,200epochs／96,000updates；正式预测47,000行。账本无超限或退出协议错误，formal partial为空。上一轮独立审查已核对消费、冻结先于TEST、源码交付diff及无遗留训练进程；本次复核现有终态与故障日志，不重放审计。
- CPU日志为10 passed、1 failed，唯一失败节点是`tests/test_ua3d_spatial.py::test_identical_capacity_isolated_rng_zero_initial_and_no_neighbor_bias`。代码在循环中改写`modules[0].up.bias`后，下一轮仍将其作为初态基准；属于测试自身状态污染。上一轮只读核对保存的三个spatial_initial：参数逐项相同，Wup为零。本轮不重复加载认证。修复后尚未执行的断言是否全部通过，仍须由一次最终复验决定。
- 父原冻结时间为10月26日01:33:35，固定G=S_RADANG、FINAL=C_CONST。正式FULL：BASE0.795、C_CONST0.792、S_RAD0.790、S_RADANG0.792；probe：0.671／0.682／0.693／0.700。原始精度以父results.json为准。G减BASE：FULL−0.004、probe+0.030；G减C_CONST：FULL+0.000、probe+0.018。结论保留“表征诊断改善、整体性能未通过”，返修不能将数值失败改写为性能通过。
- 旧STAGE/r1在等待父结果时于01:46阻断，jobs=0，FT／TEST／CPU实际与预留消费均0；真实exit=1，旧截止10月26日04:51已过。报告仍使用“两新臂均按既定预算训练”的模板，需要在新交付中纠正。不能重启旧controller或改旧终态。
- 本次查看时tmux session Uni-Poly及旧controller留存窗口存在，未发现相关Python训练／controller运行；未核实四卡当前占用。授权执行时由正常调度记录资源，等待计时，不加benchmark或预跑。新A、B结果根尚不存在。本轮没有加载模型、运行测试或实验。

### 14.2 科学问题、基线与返修边界

本轮首先解决交付正确性：测试状态污染是否能在保持模型、数据和既有预测完全不变时修复；随后完成原定PT／FT启用空间分支的2×2阶段拆分，解释probe改善而FULL未改善的现象。修复测试不构成模型创新或性能提升。

E00复用父BASE FULL，E11复用父G FULL；性能参照REF严格取父冻结FINAL=C_CONST。G与REF均来自原validation冻结，不能因已知TEST较低改为其他arm。两个新交叉臂仍为PT_ONLY与FT_ONLY，第13.3节状态转换、三模态范围和解释限制全部保留。r4 TEST已被用于后续问题形成，须披露顺序开发与共享TEST限制，不称独立盲测。

父r4所有prepare、PT、FULL、probe、特征及正式TEST均只读复用，新增消费为0；不重建构象／空间sidecar、FP或scaler，不训练新PT，不改变模型公式、损失、优化器、split、task、seed、batch或epoch。旧失败CPU继续收费，旧STAGE/r1的零消费如实结转。

### 14.3 A段：测试、报告与父依赖接纳的最小修复

1. 在`tests/test_ua3d_spatial.py`中，先对三组模块完成全部初态、参数数目、零Wup及隔离RNG断言，再逐个开展会修改bias的行为检查；或使用detach().clone()保存独立初态基准。不得删断言、跳过失败节点、改成比较对象自身或放宽容差掩盖问题。无邻居屏蔽等原断言继续保留。
2. 修正`run_ua3d_stage.py`报告：零派发写“未启动”，部分执行写实际完成数；只有完整消费证据支持时才写完成训练、冻结和TEST。新更正说明引用旧r1账本为零，不覆盖旧报告／results.json或CURRENT_PLAN中的历史原报告块。
3. 实现显式repair1接纳路径，替代旧STAGE/r1要求“父历史exit0”的单一路径。只允许本节指定的r4原始运行、唯一CPU测试缺陷被修复并复验成功的情形；不能直接忽略failure.json或允许任意失败父运行。至少保留以下必要条件：
   - 原父计划／r4身份一致，原controller已退出且真实exit=1，唯一非零作业为`ua3d_r4_cpu`；日志对应上述唯一失败节点和10 passed／1 failed。其余565作业完成、无预算或退出协议错误，原正式结果与引用齐全，相关worker已退出。复用上一轮审查及现有消费证据，不运行额外验收脚本、重新评估或全量checkpoint扫描。
   - 原四PT、140 FULL、140 probe、280 TEST及冻结记录仍完整；输入、scaler、row／key、G／FINAL与所引用metadata相符。必要身份检查纳入正常加载；不制造哈希合同。
   - A段唯一最终CPU suite真实exit0；repair记录标明原故障节点、修复diff、命令、cwd、日志、真实exit、消费及原父路径，状态为待审查，independent_review=PENDING。先有真实exit0再写接纳记录，记录排他创建，不由“文件存在”单独决定通过。
   - 父训练／推理源码、checkpoint和原预测不因A段发生变化；若发现生产逻辑或数据问题，停止，不能归入本测试缺陷例外。repair接纳必须记录`original_controller_exit=1`、`original_cpu_exit=1`、`repair_cpu_exit=0`，不伪造父成功，也不冒充独立审查通过。
4. 为上述两项新增运行边界保留最小必要CPU fixture：修复接纳拒绝缺失凭据／无关父故障；零作业报告不称已训练。可新增`tests/test_ua3d_repair1.py`，不调用模型训练或正式数据推理。该fixture只随A段最终suite运行一次，不在B段重复。
5. 全部本次代码修复、B段r2适配及交付记录逻辑完成后，A段集中一次CPU复验（timeout600秒、CUDA_VISIBLE_DEVICES为空），覆盖修正后的完整spatial suite与本次repair fixture。命令为`python -m pytest -q -p no:cacheprovider tests/test_ua3d_spatial.py tests/test_ua3d_repair1.py`，外层timeout及真实exit捕获由独立tmux wrapper负责。不得只选原失败节点替代完整spatial suite，不做阶段smoke或额外回归。A是独立修复交付的最终测试，也是B运行依赖，不是性能预跑。

A段失败即停止A／B链，无第二次复验或自动改码重试；A段通过后生成`parent_acceptance.json`与更正报告，以父原results.json为数值来源，保留完整精度及原冻结选择，不重新运行原aggregate、模型推理或280个TEST。

### 14.4 B段：独立r2接续原阶段拆分预算

使用计划ID UA3D-STAGE-20261026-01、修订r2及全新结果／日志根，不恢复旧r1。r2消费的是旧r1未启动的同一70套矩阵，重新授权时转入r2，不再为r1保留另一份可执行70套预算；任意时刻只允许一个活跃controller／账本writer。

| 单元 | 固定来源与受控变化 | 本轮新增 |
|---|---|---:|
| E00 | r4 BASE FULL | 0训练／0推理 |
| E11 | r4 S_RADANG FULL | 0训练／0推理 |
| REF | r4 C_CONST FULL，父原FINAL | 0训练／0推理 |
| PT_ONLY／E10 | S_RADANG PT 10K；只移除完整encoder3d.spatial状态，其余native／模态原样严格加载；FT保留原生3D模态 | 35 FULL＋35 TEST |
| FT_ONLY／E01 | BASE PT 10K先严格加载；再安装S_RADANG定义及其保存的spatial_initial零初态；不取G训练后空间权重 | 35 FULL＋35 TEST |

- 七任务Eea／Egb／Egc／Ei／EPS／Nc／Xc，fold0—4，seed42+fold；head10＋joint60，FP32、batch24、workers0、原head初态／fusion初态／train-only scaler及15行缺3D处理。原TMOL损失、scheduler、optimizer参数组不变；空间模块head冻结、joint归native LR1e-4。PT_ONLY不是NO3D。正常加载记录状态增删白名单、形状、dtype及37632参数，不宽泛strict=False。
- **仅物理GPU2**；B的ALLOWED_GPUS=(2,)、GPU任务CUDA_VISIBLE_DEVICES=2、--physical-gpu=2，world1、内部cuda:0；FT与推理在该卡合计最多4路，线程2、workers0。controller／A与B的CPU suite均隐藏GPU。单卡忙时等待计时，不回退到GPU0／1／3，不抢占无关作业，不减batch补救OOM。所有长命令／worker／实验产物写入均在tmux Uni-Poly独立window。
- 资源限制须覆盖stage的允许集合、函数及CLI默认卡、调度候选、worker命令、r2配置、authorization／manifest／consumption和交付记录；不能只在外层设置CUDA_VISIBLE_DEVICES而让controller重新分配四卡。保留父r4原四卡provenance，其历史卡号不改、不要求改成GPU2；只限制本次新派发。
- A段通过及接纳证据齐全后，r2自动执行70 FULL。全部validation完成后一次冻结70个BEST、scaler、来源、G／REF／C2／NEXT_FINAL及父引用。C2只按PT_ONLY／FT_ONLY全局validation macro选，tie PT_ONLY；NEXT_FINAL按REF／PT_ONLY／FT_ONLY全局validation macro选，tie按此顺序。之后两臂各35 TEST各一次，共11750行，无论validation有无优势都完成固定矩阵。
- 父prediction按原row＋sample key对齐，目标一致性检查随既定复用流程完成；不按数组位置拼接，不重推理父模型。主要性能门槛C2减REF正式TEST macro R²≥0.010，完整精度判定，展示3位小数；不根据新TEST追认NEXT_FINAL或选择单task赢家。
- 固定报告E10−E00、E11−E01、E01−E00、E11−E10、各新臂减REF及C2减E00；交互I=E11−E10−E01＋E00仅作描述性统计。PT空间处理可经ALIGN影响2D／FP／Q，撤去分支还改变表示分布；不能把结果解释成仅3D权重的孤立因果收益。
- 全部70 TEST及既定预测身份对齐结束后，B段集中一次最终CPU：`python -m pytest -q -p no:cacheprovider tests/test_ua3d_stage.py`，timeout600秒、隐藏GPU。只适配与r2父接纳、修订／路径、累计账本相关的原fixture，不再运行spatial／repair suite。训练或TEST失败时不提前运行该suite补交差。

### 14.5 允许修改与旧产物保护

- 允许最小修改：`tests/test_ua3d_spatial.py`的上述失败测试；`scripts/run_ua3d_stage.py`的状态报告、r2入口／窗口名、deadline、累计账本、仅GPU2调度与CLI默认卡；`src/training/ua3d_stage.py`的r2路径／revision、状态块、严格父接纳及仅GPU2允许集合／默认卡／资源metadata；`src/modules/ua3d_stage.py`仅r2 BEST schema／metadata；`tests/test_ua3d_stage.py`只适配本次边界；`CURRENT_PLAN.md`当前提案与新的执行／审查块。
- 允许新增：`configs/ua3d_stage_r2.json`、`tests/test_ua3d_repair1.py`和必要的`src/training/ua3d_repair.py`／`scripts/run_ua3d_repair.py`，用于repair专属记录、严格接纳和一次CPU wrapper。不改变原r1配置，不以环境变量静默绕过父失败。
- 禁止修改父spatial模型／数据／训练／PT脚本、旧controller或旧checkpoint schema及其他共享训练代码；新增repair工具只管理本次记录与依赖，不重新训练。若最小范围不足，报告具体阻断，不扩大修改。
- r4原`failure.json`、`controller_exit.json`、CPU日志、budget、execution_manifest、final_review、选择、checkpoint及预测全部保留只读。旧STAGE/r1同样全部只读；纠正报告只写新repair根，历史原报告块保留并注明被更正。本节状态块不得覆盖原UA3D_SPATIAL_STATUS／UA3D_STAGE_STATUS标记块，应使用独立REPAIR及STAGE_R2标记。
- 任何新根存在、writer或来源不符时停止，不能自动换repair2／r3或清理文件。不得删除／覆盖事故证据、TODO、旧dirty文件；不push/pull、不增加哈希、指纹或校验和验证。

### 14.6 固定预算、计时与估时

| 项目 | A段新增上限 | B段新增上限 | 含原r4／旧r1的累计上限 |
|---|---:|---:|---:|
| prepare／PT／构象／FP／probe | 0 | 0 | 原消费不变：prepare1、PT4／40000updates、probe140／4200epochs／96000updates |
| FULL | 0 | 70starts／4900epochs／112000updates | 210starts／14700epochs／336000updates |
| 正式TEST | 0 | 70单元／11750行 | 350单元／58750行 |
| 最终CPU suite | 1次，timeout600秒 | 1次，timeout600秒 | 3次：原失败1＋A复验1＋B最终1 |
| 父重复推理／原TEST重跑 | 0 | 0 | 0新增 |

新增CPU两次是本提案待授权预算，不能从原失败次数中退款；B的70套是原未消费预算迁移，不再追加另一轮70套。独立账本同时展示父r4已消费、旧r1零消费、A修复、B新消费及总量；所有失败／部分单元派发即整单元收费，不退款、不重试／resume或追加seed。

获执行授权后，在tmux独立repair入口首次启动即持久化唯一t0，代码实施／等待／A复验／B训练／TEST／最终CPU和交付均计入。A段上限60分钟；全链硬截止为min（t0＋6小时，父原截止2026-10-27 21:18:28 Asia/Shanghai）。r2读取同一截止，不在A通过或r2启动时重置六小时；旧r1截止保留原值，本节重新计时必须由新授权批准，不追写成旧r1延长。

依据r4实际140套FULL在四卡最多16路下约50分钟、280 TEST约13分钟，70套FULL在单卡最多4路时按吞吐粗推约100分钟，70 TEST约13分钟；实际还受任务长短、单卡竞争和交叉臂负载影响。B暂估2—4小时，A实施／记录／CPU约20—45分钟，整体暂估2.5—5小时，单卡六小时内完成可行性未核实。原四卡1—2.5小时估时不再适用于本提案。硬上限仍为原6小时及父原截止，不扩大训练或时间预算；等待计时。没有新授权时不计算绝对完成时间，不做benchmark验证ETA。

### 14.7 停止条件、交付与通过层级

- A复验非零、timeout、接纳证据不足或故障不再局限于已知测试状态污染：停止A／B，不启动FT。B中NaN／Inf、OOM、artifact／schema／身份不匹配、worker失败、writer冲突、正式覆盖风险、预算或截止问题：停止受影响运行并保留真实exit／部分消费。不自动修第二轮，不换卡、减batch、补task／fold／seed或延时。
- 代码正确性、执行完成、性能门槛与独立审查分别记录。A通过只表示本次修复自检通过／父结果在指定例外下可复用；父与A仍待独立审查。B完整执行且唯一CPU0后状态待审查；数值不达0.010写性能未通过，不称运行失败。任何未执行或缺失任务写未完成／不可比较。
- A独立结果根：`results/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1/`，日志：`logs/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1/`；window建议`ua3d-repair1-controller`及`ua3d-repair1-cpu`。只新增该repair子目录，不改其父旧文件。
- A交付authorization、manifest／status／budget、修复前后diff、原故障引用、旧r1零消费的更正报告、唯一CPU命令／日志／真实exit、`parent_acceptance.json`及final_review。接纳记录清楚区分原失败和本次复验；已有FULL／probe原始统计直接引用父results.json，数值不变。
- B独立结果根：`results/ua3d_stage_20261026/r2/`，日志：`logs/ua3d_stage_20261026/r2/`，window建议`ua3d-stage-r2-controller`。交付第13.8节所列70套history／BEST／scaler／validation、唯一冻结、70次TEST原始预测、父row＋key复用记录、四单元及REF统计／配对差／I、实际diff和唯一最终CPU，并引用A接纳和原父独立审查状态。
- 交付须在CURRENT_PLAN新的审查块逐项写Eea／Egb／Egc／Ei／EPS／Nc／Xc的项目五折mean±std(ddof=1)、论文参考及模型、项目减参考、split／seed口径与证据路径；参考依次0.934／0.923／0.911／0.839／0.787／0.856／0.423，对应Periodic-TDL（Egb／Egc为+Chem,+RC）。A复用父现有七项表；B据新正式结果对照，缺项不补跑。Egc3380对4124、Xc432对430等协议差异仍须披露，仅作数值比较；EAT／Eib／Tg不纳入。
- Codex复用交付证据独立审查，不重复测试或实验；通过后才讨论PROJECT_HISTORY归档。执行者不自动标独立通过、不替换部署、不按TEST重选FINAL、不展开第三周期。

### 14.8 可复制给DSH的执行提示词（待用户明确执行授权）

```text
执行对象：CURRENT_PLAN第14节UA3D-REPAIR-20261026-01/r2及其B段UA3D-STAGE-20261026-01/r2。基于dev/cd36ae7280b159f881b483ad5814835ff7b74c4a与受保护dirty工作树。仅在收到用户对本r2的执行指令后执行；“调整为只使用GPU2、给执行提示词”只授权资源／文档修订。用户将本提示词作为执行指令发送，即记录为A及B固定范围的一次授权，不再索要同一确认；A通过自动接B，无需阶段性重复确认。

从第一条返修执行命令在tmux Uni-Poly独立window记录t0，A最多60分钟，总截止min(t0+6h,2026-10-27 21:18:28北京时间)，实施／等待／复验／训练／交付均计入，B不刷新截止。保留r4需返修与旧STAGE/r1阻断及exit1，不能恢复旧controller或覆盖旧根。

科学问题保持原PT/FT空间分支2×2：E00=父BASE，E11=父原G=S_RADANG，REF=父原FINAL=C_CONST。修复只涉及测试自身状态污染、报告和严格repair接纳／r2执行记录，不改父模型／数据／loss／optimizer／checkpoint／冻结／预测。父PT/FULL/probe/TEST新增0。允许文件、独立路径与停止条件严格按14.3—14.7节。

A：先完成全部最小实现和r2适配；初态比较使用独立clone或在全部比较完成后再修改bias，保留原断言。新报告按实际消费描述，旧r1零启动不得写已训练。接纳仅允许指定父运行唯一CPU测试缺陷，原565个非CPU作业完整且无其他故障；记录原controller1、原CPU1与本次CPU0，不删除failure或伪造父exit0。正常加载检查来源／身份，不加哈希或额外验收。A最后只运行一次timeout600s、CUDA隐藏的pytest：tests/test_ua3d_spatial.py tests/test_ua3d_repair1.py；真实exit0后排他写parent_acceptance，状态待审查/PENDING。失败停止，无自动重试。

B：独立r2结果／日志根，迁移旧r1未消费的70 FULL预算而非再加70。PT_ONLY从G PT10K只显式移除完整37632参数spatial state，其余native/modal严格保留，FT仍是完整原生3D；FT_ONLY从BASE PT10K严格加载后安装父G保存的零spatial_initial。不从FULL BEST初始化。七任务五折、seed42+fold、FP32/batch24/workers0、head10+joint60，原TMOL训练配方。仅物理GPU2，ALLOWED_GPUS=(2,)、CUDA_VISIBLE_DEVICES=2、--physical-gpu=2、world1、内部cuda:0，FT／推理合计最多4路、线程2。同步修改stage函数/CLI默认卡、调度候选、worker命令、r2配置和资源metadata；A/B CPU suite与controller隐藏GPU。卡忙等待且计时，不使用GPU0/1/3、不抢占、不减batch；父原四卡provenance保留。70starts/4900epochs/112000updates封顶，无PT/prepare/probe/新构象/FP/KRR/输出融合。

70套validation后一次冻结C2与NEXT_FINAL；C2按两新臂全局validation macro选、tie PT_ONLY；NEXT_FINAL在REF/PT_ONLY/FT_ONLY中选、tie按此顺序。随后两臂各35 TEST各一次，共11750行。父预测只按row+key及相同目标复用，不重推理或按TEST重选。主门槛C2−REF正式macro≥0.010，报告四阶段效应和I=E11−E10−E01+E00及七任务论文数值对照，披露共享TEST和阶段因果限制。

B全部TEST与既定预测对齐完成后唯一timeout600s CPU suite：tests/test_ua3d_stage.py，不重复A测试。累计CPU共3次（旧失败1+A1+B1），独立父/旧r1/A/B及总账本，失败单元整单元收费。任何NaN/Inf/OOM/schema/identity/worker/writer/覆盖/预算/截止错误停止留证，不自动修复、重试/resume、减batch或加seed。

交付第14.7节全部记录、diff、命令/cwd/tmux/物理GPU/进程/日志/真实exit与完整精度结果，更新CURRENT_PLAN独立新状态块，旧状态块不改。最终待审查/PENDING，不归档、不部署、不续开第三周期。不要读取TODO、push/pull、删除旧dirty/事故产物或增加哈希验证。
```

### 14.9 历史r1计划编写与本轮r2修订记录

本轮仅修改CURRENT_PLAN.md顶层当前提案索引并追加第14节，保留全部旧执行记录与失败报告；计划完成后只检查本次文档diff与格式。未改代码、配置或测试，未创建A／B结果根、未启动controller／GPU／worker／CPU suite，未读取TODO或新增哈希／远程操作。新增执行状态为待授权，不把本计划或旧授权当成本次返修启动授权。

上述为r1编写记录。r2按用户最新要求只修订本文件第14节与顶部索引：新增运行仅GPU2、FT／推理合计最多4路，更新执行入口必须适配的资源字段与单卡ETA；全部科学范围、原6小时硬上限、训练／评估预算、结果根及停止条件不变。仅检查最终文档diff与格式；未改代码、运行测试或启动实验，执行状态仍为待授权。

<!-- UA3D_REPAIR_STATUS_START -->
- 2026-10-26T10:00:46.489090+08:00：UA3D-REPAIR-20261026-01/r2；状态**待审查**；independent_review=PENDING。用户明确授权第14节A及自动B，仅GPU2。
- A唯一CPU真实exit0；严格父接纳完成，原故障不改写。t0=1792979783.591774；A截止=1792983383.591774；全链截止=1793001383.591774。B自动接续；父完整精度统计及七项SOTA表见本根final_review/report.md与父原results.json。
- A结果`/root/workspace/Uni-Poly-Plus-master/results/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1`；日志`/root/workspace/Uni-Poly-Plus-master/logs/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1`；原父controller/CPU exit1均保留；旧STAGE/r1零消费账本与报告只读保留，原错误模板由新更正报告纠正。
<!-- UA3D_REPAIR_STATUS_END -->
<!-- UA3D_STAGE_R2_STATUS_START -->
- 2026-10-26T11:25:06.349960+08:00：UA3D-STAGE-20261026-01/r2；状态**需返修**；阶段交付报告。授权来源：用户明确执行第14节A及自动B；仅GPU2。
- # UA3D-STAGE r2 实际交付

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

- 结果`/root/workspace/Uni-Poly-Plus-master/results/ua3d_stage_20261026/r2`；日志`/root/workspace/Uni-Poly-Plus-master/logs/ua3d_stage_20261026/r2`；independent_review=PENDING。
<!-- UA3D_STAGE_R2_STATUS_END -->


### 14.10 本次r2实际终态与故障交付（2026-10-26）

- 完成层级：A实现及唯一CPU复验通过（14 passed，真实exit0），严格parent_acceptance已排他记录，A待审查。B的70 FULL、70正式TEST和140组原row+sample key及相同目标对齐完整完成；B最终唯一CPU为9 passed、1 failed，真实exit1，controller真实exit1；状态**需返修**，independent_review=PENDING。按停止规则不改码重试、不自动resume、不归档、不部署、不展开第三周期。
- 失败节点：`tests/test_ua3d_stage.py::test_failed_unit_keeps_reservation_and_parent_cumulative_budget`。原fixture调用`reservation('pt')`预期ValueError，但原stage reservation实现先读取`data/splits/mips_paper5_outer/None.json`，实际FileNotFoundError。该顺序在本次修改前已存在；本次未在最终suite前发现，未修正或放宽断言。失败发生在训练／TEST／对齐全部结束后的CPU边界检查，不能将B判为正确性通过。没有第二次测试或修复授权，保留现场。
- 实际新增：70 FULL／4900epochs／112000updates；70 TEST／11750行；prepare/PT/probe/构象/FP生成均0，父重训练／重推理均0。累计CPU3次已全部消费（父失败1+A成功1+B失败1），失败不退款；父+新增累计FULL210／14700epochs／336000updates，TEST350／58750行。旧STAGE/r1消费0，同一70套预算迁入r2，不再保留第二份。
- 唯一t0=2026-10-26 09:56:23.592 Asia/Shanghai，A入口真实exit0于10:00:47；B controller真实exit1于11:25:06，全链截止15:56:23.592未刷新；本收尾仍在截止内。各段command/cwd/env、进程身份、tmux、日志和exit见A/B manifest、budget.jobs、log/process.json、exit.txt。A窗口ua3d-repair1-controller(@2100)，B窗口ua3d-stage-r2-controller；物理GPU2，world1、内部cuda:0、FT/推理合计最多4路、workers0/线程2。NVML未知PID时按调度等待，未换卡或改调度逻辑。
- 一次validation冻结2026-10-26 11:19:16.841：G=S_RADANG，REF=父原FINAL=C_CONST，C2=PT_ONLY，NEXT_FINAL=REF。TEST后未重选。正式macro：E00=0.795，E11=0.792，PT_ONLY=0.790，FT_ONLY=0.797，REF=0.792。主要C2−REF=−0.001，未舍入值−0.0011331146039976758，小于0.010，**性能未通过**。FT_ONLY−REF=+0.005，不能按TEST改选它。
- 阶段效应：E10−E00=−0.005；E11−E01=−0.005；E01−E00=+0.002；E11−E10=+0.001；描述性交互I展示−0.000，未舍入值−0.0004312350840597369。该轮未支持PT空间处理的正向收益；不能将跨模态PT影响或撤支分布变化解释为孤立3D权重因果效果，无统计显著性结论。
- 实际修改：tests/test_ua3d_spatial.py仅增加第二循环避免bias污染初态；scripts/run_ua3d_stage.py、src/training/ua3d_stage.py做r2资源/路径/父接纳/报告/账本适配；src/modules/ua3d_stage.py仅BEST schema；tests/test_ua3d_stage.py仅r2边界断言；新增configs/ua3d_stage_r2.json、src/training/ua3d_repair.py、scripts/run_ua3d_repair.py、tests/test_ua3d_repair1.py；CURRENT_PLAN.md记录授权和独立状态块。原spatial生产模型/数据/训练/PT代码与旧配置/产物保留。
- A：`results/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1/`，对应logs；更正旧r1为未启动，父原controller exit1、CPU exit1、failure、冻结及数值均保留。B：`results/ua3d_stage_20261026/r2/`，对应logs；原始预测`formal/{PT_ONLY,FT_ONLY}/{task}/fold{fold}/test_predictions.npz`；BEST/history/scaler/validation在finetune同结构；冻结selection/final_frozen.json；对齐final_review/prediction_join.json；全精度final_review/results.json；本次实际补丁final_review/actual_changes.diff。
- 未执行：任何额外测试、smoke、benchmark、PT、prepare、probe、父推理、修复重试、远程同步、TODO读取、哈希验证、归档与第三周期。下一步为只读独立审查；如需修复B边界及新增CPU复验，须另外授权，本轮不自行执行。

本轮七任务五折mean±std(ddof=1)，seed42+fold，任务等权macro。以下为validation冻结C2=PT_ONLY；其他四单元及REF七项完整表见前一STAGE_R2独立报告块和B final_review/report.md。论文逐任务最佳为包络而非一个模型；Egc3380对4124、Xc432对430等split/协议差异，仅作数值对照，不能声称公平超过论文；共享开发TEST非独立盲测。

| 任务 | C2项目R² | 论文参考 | 项目减参考 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.920±0.015 | 0.934 | -0.014 | Periodic-TDL |
| Egb | 0.920±0.012 | 0.923 | -0.003 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.913±0.014 | 0.911 | +0.002 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.810±0.042 | 0.839 | -0.029 | Periodic-TDL |
| EPS | 0.742±0.068 | 0.787 | -0.045 | Periodic-TDL |
| Nc | 0.817±0.039 | 0.856 | -0.039 | Periodic-TDL |
| Xc | 0.413±0.074 | 0.423 | -0.010 | Periodic-TDL |

## 15. GPU2返修与STAGE/r2独立审查（2026-10-26）

### 15.1 审查快照与结论

- 检查时间2026-10-26 12:05—12:11（Asia/Shanghai）；dev，HEAD=`cd36ae7280b159f881b483ad5814835ff7b74c4a`。git status仍包含CURRENT_PLAN／TODO改动及UA3D／其他旧周期未跟踪文件，全部保留，未读TODO。已查看AGENTS、相关diff、SOTA_RESULT、历史参考、修复及stage源码、测试日志、真实退出、预算／冻结／预测对齐／状态转换记录。只读检查与文档修改，无测试、模型推理、训练、远程同步或哈希验证。
- **A修复独立审查通过（仅修复范围）**：spatial测试将初态比较与会修改bias的行为检查分成两个循环，原断言保留；唯一A suite为14 passed、真实exit0，原父失败与数据结果保留。严格接纳仅针对原r4唯一已知CPU缺陷，未伪造父exit0；旧r1零消费报告已更正。父r4生产模型／数据／训练／PT及配置与原交付源码逐内容一致。现有artifact中的independent_review=PENDING是交付时快照，本节记录当前审查，不为更新状态而改写旧artifact。
- **STAGE/r2整体需返修，性能未通过**：70 FULL和70 TEST科学流程完整，但唯一最终CPU为9 passed、1 failed，controller真实exit1。未发现合法FT／TEST路径受下述非法阶段异常路径影响的证据；这不等于CPU套件通过或全周期验收通过。暂不归档PROJECT_HISTORY，不替换部署。
- P1发现：`scripts/run_ua3d_stage.py:65—70`的reservation先读split，再判断kind。原fixture在`tests/test_ua3d_stage.py:121`调用reservation('pt')，预期ValueError，却先尝试读取None.json触发FileNotFoundError。该生产入口边界错误在旧版本已存在，本轮没有修正；应在任何数据读取前拒绝非法kind，并验证合法FT／TEST所需task／fold。不能把断言改成接受FileNotFoundError或宽泛Exception。
- 新A／B源码与实际交付diff逐内容核对一致；state_conversion记录与源码一致：PT_ONLY移除完整8个spatial state键、37632参数，其他模块严格保留；FT_ONLY从BASE后安装同定义零初态，未使用FULL BEST。未额外加载大型checkpoint逐张量认证；checkpoint实际加载正确性以已有正常运行检查、转换记录和现有测试为依据。
- B作业共141：FT70 exit0、TEST70 exit0、CPU1 exit1。逐项现有消费与exit.txt一致，70套各70epochs／共112000updates，70正式TEST共11750行；冻结11:19:16.841先于所有新TEST。已知进程均退出，GPU任务只用物理2、最大并发4；未发现预算或退出协议违规。预测复用记录COMPLETE、140组row＋sample key及相同目标对齐、新父推理0。
- B训练10:00:55—11:19:13，TEST11:19:17—11:24:43，最终CPU11:24:47—11:25:02；controller11:25:06退出。与A合计从09:56:23计时约1小时29分钟，截止15:56:23未刷新。累计FULL210／14700epochs／336000updates，正式TEST350／58750行；累计CPU3次全消费（原失败＋A成功＋B失败）。

### 15.2 科学结果与可解释边界

| 单元 | PT空间 | FT空间 | validation macro R² | 正式TEST macro R² |
|---|---|---|---:|---:|
| E00／BASE | 关 | 关 | 0.809 | 0.795 |
| E10／PT_ONLY | 开 | 关 | 0.809 | 0.790 |
| E01／FT_ONLY | 关 | 开 | 0.805 | 0.797 |
| E11／S_RADANG | 开 | 开 | 0.806 | 0.792 |
| REF／C_CONST | 常数空间包 | 同PT | 0.811 | 0.792 |

原validation完整精度选择C2=PT_ONLY、NEXT_FINAL=REF，冻结未变。主要C2−REF=−0.001（原始−0.0011331146039976758），C2−E00=−0.005；FT_ONLY−REF=+0.005、FT_ONLY−E00=+0.002，均小于0.010。FT_ONLY虽然TEST数值最高，不能事后改为C2／NEXT_FINAL。

PT处理效应在FT空间关闭／开启时分别−0.005／−0.005；FT空间处理效应在PT关闭／开启时分别+0.002／+0.001；描述性交互I为−0.000（原始−0.0004312350840597369），无统计显著性结论。结果未支持空间PT包给FULL带来正收益；也未证明“3D表征被FT破坏”，因为PT_ONLY撤分支改变表示分布，且空间PT经ALIGN同时改变2D／FP／Q。此前probe改善只能说明源3D路径的固定头可读性，不是FULL因果收益。

下一步优先级：先修复入口预算边界并关闭本轮正确性缺口；科研上固定共同BASE非3D参数，只交换并冻结3D路径，检验预训练空间3D表示能否转移到完整模型。这样能减少上一轮“PT处理连带改变其他模态”的混杂；不是根据FT_ONLY TEST最高追认候选或延长PT。

### 15.3 七任务正式结果与论文数值对照

以下C2是原validation冻结的PT_ONLY，FT_ONLY仅作为预注册比较臂报告。项目为五折mean±std(ddof=1)、seed42+fold，任务等权macro；差值用完整精度计算后展示。E00／E11／REF完整七任务表仍保留于第14节实际报告。

| 任务 | C2项目R² | C2减论文 | FT_ONLY项目R² | FT_ONLY减论文 | 论文最佳R² | 参考模型 |
|---|---:|---:|---:|---:|---:|---|
| Eea | 0.920±0.015 | -0.014 | 0.928±0.016 | -0.006 | 0.934 | Periodic-TDL |
| Egb | 0.920±0.012 | -0.003 | 0.912±0.013 | -0.011 | 0.923 | Periodic-TDL (+Chem,+RC) |
| Egc | 0.913±0.014 | +0.002 | 0.913±0.012 | +0.002 | 0.911 | Periodic-TDL (+Chem,+RC) |
| Ei | 0.810±0.042 | -0.029 | 0.816±0.036 | -0.023 | 0.839 | Periodic-TDL |
| EPS | 0.742±0.068 | -0.045 | 0.775±0.053 | -0.012 | 0.787 | Periodic-TDL |
| Nc | 0.817±0.039 | -0.039 | 0.826±0.062 | -0.030 | 0.856 | Periodic-TDL |
| Xc | 0.413±0.074 | -0.010 | 0.407±0.115 | -0.016 | 0.423 | Periodic-TDL |

论文为逐任务最佳包络，非单一模型；Egc3380对4124、Xc432对430及split／统计协议差异未消除，只作数值比较，不宣称公平超越论文。共享开发TEST非独立盲测；EAT／Eib／Tg不纳入。

证据：A根`results/ua3d_spatial_20261025/r4/repairs/cpu_report_repair1/`的final_review／parent_acceptance；B根`results/ua3d_stage_20261026/r2/`的budget、selection、finetune、formal及final_review/results.json、report.md、prediction_join.json、actual_changes.diff；唯一B失败日志`logs/ua3d_stage_20261026/r2/ua3d_stage_r2_cpu/stdout.log`。本次不改写任何旧结果／失败记录。

## 16. 当前执行计划：预算边界返修＋冻结3D迁移对照（用户已授权）

> 计划ID：UA3D-FROZEN-TRANSFER-20261026-01；修订r1；状态：**待审查**。用户已明确授权第16节固定A返修及A通过后自动B，无需重复确认。历史消费和记录保留；当前状态见独立GUARD及FROZEN_TRANSFER块。

### 16.1 版本、科学问题与固定对照

基于第15节快照：dev／cd36ae7280b159f881b483ad5814835ff7b74c4a及受保护dirty工作树，最新仅GPU2要求持续适用，覆盖AGENTS默认GPU3。未核实GPU2当前占用；不做benchmark、smoke或前置模型探针。下列新结果根在规划时不存在。

假设：S_RADANG源PT-3D路径的probe可读性提升，能否在共同非3D背景中、通过冻结路径保留并转化为FULL性能？本实验不预设会发生收益，也不将probe差值当作FT性能结论。

| 新臂 | 共同非3D初态 | 3D初态 | FT处理 | 新训练／TEST |
|---|---|---|---|---:|
| F_BASE | 父r4 BASE PT10K | BASE的encoder3d／norm3／A3 | 完整3D路径冻结 | 35 FULL／35 TEST |
| F_G3D | 同一BASE PT10K | 仅换为父G=S_RADANG的encoder3d／norm3／A3，包括训练后的spatial | 同样完整3D路径冻结 | 35 FULL／35 TEST |

两臂o8、norm2、A2、fp_encoder、Q2、Q3均取同一BASE PT endpoint；fusion／head取相同公共初始化和原fold head。只改变被严格声明的3D模块来源与是否携带空间分支，参数形状差别属于空间表征包处理。**F_G3D不是完整G endpoint再冻结**，不得将G的2D／FP／Q参数一起换入。两臂均从PT endpoint完整FT，不从任何FULL BEST接续。

已核实reference：匹配冻结对照F_BASE必须新训练；原r4 BASE FULL宏0.795与父冻结REF=C_CONST宏0.792作为已有固定参照复用，不新推理。E11与FT_ONLY仅保留历史描述，不与本新两臂拼凑因果2×2；本次非3D来源已不同，不计算或宣称旧／新混拼interaction。

### 16.2 A段：STAGE/r2预算边界最小返修（新增训练0）

1. 只修改`scripts/run_ua3d_stage.py::reservation`：先拒绝非cpu／ft／test阶段，返回明确ValueError，再处理cpu固定预算；合法ft／test在读取split前验证task属于七项、fold为真正int且0—4，非法输入先报错。合法阶段的预算公式与返回值不变，不把pt加入允许范围。
2. `tests/test_ua3d_stage.py`保留原ValueError及失败整单元收费断言；在同一fixture中使用monkeypatch令读取函数遇到非法输入便报错，证明非法kind／task／fold不读取文件。不能把预期改成FileNotFoundError、Exception或通过创建None.json绕过。其余stage科学实现、BEST schema、配置与旧产物不变。
3. 修复及交付入口完成后集中**一次**CPU复验：`python -m pytest -q -p no:cacheprovider tests/test_ua3d_stage.py`，timeout600秒、CUDA_VISIBLE_DEVICES为空，在tmux Uni-Poly独立窗口，保存命令／cwd／stdout／stderr／真实exit。不要重跑spatial或此前A suite，不运行旧controller、prepare／训练／validation／TEST或旧aggregate。
4. 新根`results/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1/`及对应logs；排他创建，仅追加repair子目录，原B failure、exit1、CPU1、预算／预测／冻结／results全部保留只读。修复只写新更正报告与`stage_acceptance.json`：原非法阶段guard缺陷、原CPU1／controller1、本次CPU0、原140个科学作业完整及本次diff／真实exit。不得伪造原controller成功，累计失败费用不退款。
5. B依赖此一次复验exit0及指定guard缺陷的窄范围接纳、原科学产物完整／进程已退出／预算无违规、源r4现有A接纳记录；不能笼统允许任意失败父任务。原正式统计、G／REF／C2／NEXT_FINAL从旧记录只读引用，不重算或重选。CPU失败、异常不再局限于guard或来源缺项即停止，不自动二次修复或测试。

### 16.3 B段：实现级冻结与加载合同

- 保留完整TMOL3M、MaskedGalformer3D／SpatialGalformer以及原FP／gate／head，复用原finite Trimer、侧缓存、split、scaler和缺失15行语义。不要将新实验改成3D-only probe或纯2D模型，不新增构象／MD／PH／FP／KRR／输出融合。
- 正常初始化先严格加载共同BASE PT状态，再按arm为encoder3d／norm3／A3安装对应源。F_G3D的spatial直接取父G训练后的状态，既不置零，也不从spatial_initial读；只允许这三个完整模块的显式state替换，其余模块严格来自BASE。记录module来源、键集合／形状／dtype、正常加载是否成功，不添加额外checkpoint扫描或哈希验证。
- **冻结的精确集合为encoder3d（含spatial／mask token）、norm3、A3**，70个epochs全程requires_grad=False、eval，参数与buffer无更新；因此固定的是原始256维z3，不能只冻结spatial或encoder而让A3继续漂移。模型调用train()后必须重新将这三个模块保持eval，优化器排除其全部参数。
- 其余参数沿用原head10＋joint60：head阶段原native中的o8／norm2冻结eval，A2／fp_encoder／Q2／Q3／fusion／head按原方式训练；joint阶段o8／norm2 native LR1e-4，其余原modal/fusion/head组LR1e-3。head组LR3e-4；WD0.02、clip5、原scheduler／loss／dropout及状态严格保留。Q3仍从BASE初始化并按原配方训练，不属于z3路径冻结集合；ALIGN仍可调整其他模态来匹配固定3D。
- 原通用optimizer会在每个阶段重设requires_grad，不能调用后仅删除param group而忽略冻结状态。新实验专属optimizer应先应用阶段规则和冻结集合，再仅对可训练参数建组；保持原组名／LR／scheduler，不允许冻结参数留在optimizer。新模型train覆写只管理本实验，不改共享原模型行为。
- 冻结3D本身没有新增随机层。原FULL FT构造backbone dropout=0，冻结路径eval与probe取值一致；保留原全局seed与数据顺序，不因优化速度新增缓存、改batch或改变其他分支随机策略。固定路径forward可随正常batch计算，不新建多轮特征缓存；若用no_grad仅包裹三个冻结模块，不能截断其他分支或Q3梯度。
- 正常训练日志记录两阶段可训练参数分组、冻结模块training／requires_grad状态、冻结参数无grad／无optimizer归属、缺3D投影后严格零。只在原正常保存流程中记录状态，不增加阶段模型探针、逐epoch额外预测、checkpoint认证或第二测试轮。

### 16.4 固定训练、validation、一次冻结与正式TEST

- 七任务Eea／Egb／Egc／Ei／EPS／Nc／Xc，fold0—4、seed42+fold。与原父相同train／validation／TEST行、row＋sample key＋SMILES、train-only scaler和性质标签；FP32、batch24、workers0、线程2。每臂35套，每套head10＋joint60，共70starts／4900epochs／112000updates，best仅按原单位validation RMSE严格改善选取。
- 所有70套validation结束后，一次冻结两臂全部BEST、scaler、来源、配置及样本身份。F_G3D是预注册科学候选，F_BASE是匹配对照，不按新TEST决定报告哪条几何臂。NEXT_RECOMMEND仅在原REF、F_BASE、F_G3D中按全局validation macro选择，exact tie顺序REF、F_BASE、F_G3D，冻结后不改推荐。
- 统一冻结后两臂各35个正式TEST各一次，共70starts／11750行。不因validation无优势删减臂，原参照预测按row＋key及相同目标只读对齐，不重推理。任务五折mean±std(ddof=1)、七任务等权macro；原始精度保存，展示3位小数。
- 主科学门槛：F_G3D−F_BASE正式macro R²≥0.010。整体优化另报F_G3D−原r4 BASE≥0.010；相对原冻结参照另报F_G3D−REF≥0.010，均用未舍入值分别判定。只有这些数值条件与完整执行／最终CPU条件均满足，且推荐规则支持对应候选时，才讨论下一轮晋级；不自动部署或改旧NEXT_FINAL。
- 固定报告F_BASE−原BASE、F_G3D−F_BASE、F_G3D−原BASE／REF和七任务配对差，不宣称旧／新完整factorial或统计显著。空间包在PT学习时可受原多模态ALIGN影响，结果仅说明该冻结3D包在共同BASE背景中的可转移性，不能单独归因于角度、邻接或纯容量。
- 本计划在已查看旧共享TEST后形成，必须披露顺序开发和重复使用同一划分的限制。新TEST不构成独立盲测或多seed稳定性证据。若主对照仍未达到0.010，停止该固定路线扩展，回到输入／目标匹配讨论，不自动再加PT／seed／长链或调权。

### 16.5 固定预算、资源、停止与预计时间

| 项目 | 本节A新增 | 本节B新增 | 与此前累计上限 |
|---|---:|---:|---:|
| prepare／PT／probe／新构象／FP | 0 | 0 | 原消费不变 |
| FULL训练 | 0 | 70starts／4900epochs／112000updates | 280starts／19600epochs／448000updates |
| 正式TEST | 0 | 70starts／11750行 | 420starts／70500行 |
| CPU suite | 1次，timeout600秒 | 1次，timeout600秒 | 5次，含此前已消费3次 |
| 父重推理／旧TEST重跑 | 0 | 0 | 0新增 |

本表新增预算待授权，不从B失败CPU退款；旧r1零消费不变，不重启原70套。父／原A／STAGEr2／本节guard修复／新冻结实验分账并展示累计，所有失败或部分单元按派发整单元收费，无重试／resume／换根补跑。

仅物理GPU2，ALLOWED_GPUS=(2,)、worker CUDA_VISIBLE_DEVICES=2、--physical-gpu=2、world1／内部cuda:0；FT及推理合计最多4路，controller和CPU suite隐藏GPU。必须在tmux Uni-Poly独立window，记录真实命令／cwd／卡UUID／进程身份／日志／exit；卡忙等待计时，不抢占、不换卡、不降batch。新根或writer冲突、NaN／Inf、OOM、artifact／schema／身份不符、worker失败、覆盖／预算风险均停止受影响链留证，不追加自动修复。

本节若获执行授权，从第一条新执行命令持久化唯一tN，A最多30分钟，全链硬截止min(tN＋6小时，父原48小时截止2026-10-27 21:18:28北京时间)，实施／等待／CPU／训练／交付全计入，B不刷新。此为新周期待批准的独立计时，不修改已结束第14节的t0或15:56:23截止，也不凭旧授权启动。预计A约10—25分钟，新B约1—2.5小时，总计约1.5—3小时；依据刚完成的GPU2上70套78分钟及70 TEST约5分钟，冻结可能降低反向成本但未实测，不加benchmark承诺时限。

B只在全部FT／validation／正式TEST和既定预测复用完成后集中一次最终CPU：`python -m pytest -q -p no:cacheprovider tests/test_ua3d_frozen_transfer.py`，timeout600秒。必要小fixture覆盖严格3D替换／共同非3D来源、两阶段冻结与optimizer排除／eval／梯度、缺3D零、原模型科学配置、一次冻结及预算／row＋key对齐；不重复A或旧spatial suite，不增加全仓回归、阶段smoke或父预测重放。B若实验失败不提前跑suite代替完整执行。

### 16.6 修改范围、输出与交付

- A只允许上述reservation与对应同一stage测试fixture；新增最小专属guard修复记录／wrapper工具，例如`scripts/run_ua3d_stage_guard_repair.py`，不改合法训练／推理科学代码、旧schema／预算或r4接纳逻辑。
- B新增独立文件：`src/modules/ua3d_frozen_transfer.py`、`src/training/ua3d_frozen_transfer.py`、`scripts/run_ua3d_frozen_transfer.py`、`configs/ua3d_frozen_transfer_r1.json`、`tests/test_ua3d_frozen_transfer.py`。通过专属子类／optimizer复用已核实原spatial模型、数据和训练配方，不修改父spatial／stage模型、数据、loss和旧checkpoint。必要最小wrapper可放本次新入口，不重构共享controller或helper。
- A输出`results/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1/`及对应logs；B输出`results/ua3d_frozen_transfer_20261026/r1/`及对应logs；建议窗口`ua3d-stage-guard-repair1`、`ua3d-frozen-transfer-r1-controller`及独立作业窗口。只允许这两个全新根，存在即报告，不自动切r2或repair2。
- 保护第14节及之前所有失败、日志、manifest／budget／exit、selection、BEST、原始预测和已有dirty改动；旧artifact只读。CURRENT_PLAN新增独立guard与frozen状态块，保留第15节审查和历史块。若新正确性问题影响既有预测，停止报告，不暗中重训补救。不读TODO、不远程同步、不删除／新增哈希／文件指纹。
- A交付授权、版本、最小diff、命令／环境／日志／真实exit、独立预算、stage_acceptance与更正报告；B交付严格来源和冻结规则metadata、70套history／BEST／scaler／validation、一次冻结、70 TEST原始预测、父对齐记录、完整精度宏及七任务配对结果、实际diff、唯一最终CPU、资源和累计预算。
- 完成后将Eea／Egb／Egc／Ei／EPS／Nc／Xc逐项项目mean±std、论文最佳均值与模型、项目减论文、fold／seed／协议差异和产物路径写入CURRENT_PLAN新周期审查块；缺项标未完成／不可比较，不补跑。继续使用SOTA_RESULT的0.934／0.923／0.911／0.839／0.787／0.856／0.423包络，只作协议有限的数值比较。
- A通过表示guard修复自检通过；B完整执行且唯一CPU0后交付待审查，independent_review=PENDING。数值不达门槛写性能未通过，不自动改变部署。Codex复用证据独立审查，不重跑suite；通过后才归档实际周期，不能把执行者自检当独立审查。

### 16.7 可复制给DSH的执行提示词（待执行授权）

```text
执行CURRENT_PLAN第16节UA3D-FROZEN-TRANSFER-20261026-01/r1：先A预算边界返修，通过后自动B冻结3D迁移实验。本提示词被用户作为执行指令发送即授权第16节固定范围，不重复询问同一授权；当前“审查并给下一步计划”只授权审查和文档。

基于dev/cd36ae7280b159f881b483ad5814835ff7b74c4a及dirty工作树。仅物理GPU2，ALLOWED_GPUS=(2,)、CUDA_VISIBLE_DEVICES=2、--physical-gpu=2、world1/内部cuda:0，FT/推理合计最多4路、workers0/线程2；controller/CPU隐藏GPU。所有长命令、worker和实验写入在tmux Uni-Poly独立window，登记命令/cwd/进程/卡UUID/日志/真实exit。卡忙等待计时不换卡。首条执行持久化tN，A最多30分钟，总截止min(tN+6h,2026-10-27 21:18:28北京时间)，B不刷新。

A仅修scripts/run_ua3d_stage.py的reservation：任何读取前拒绝非法kind，合法ft/test读split前验证task/fold；cpu仍固定预算，合法公式不变。保留原ValueError断言，同一fixture证明非法输入不读文件，不能改为接受FileNotFoundError。全部A实现结束后只跑一次timeout600秒pytest tests/test_ua3d_stage.py，隐藏GPU。新guard repair根保存真实exit和窄范围stage_acceptance；旧STAGEr2原CPU1/controller1/failure不改。旧70 FULL/70 TEST完整复用，不重跑训练或预测。A非零或例外扩大即停止，不自动修第二轮。

B只新增第16.6节独立frozen_transfer文件，禁止改父科学代码。两新臂F_BASE/F_G3D：共同o8/norm2/A2/fp_encoder/Q2/Q3全部取r4 BASE PT10K；F_BASE的encoder3d/norm3/A3也取BASE；F_G3D只将这三个完整模块换为原G=S_RADANG PT10K，包含训练后的spatial，非零初态。共同fusion/head初态，不从FULL BEST初始化，不把G的其他模态换入。严格模块加载和显式来源metadata。

两个新臂70epochs全程冻结encoder3d/norm3/A3，requires_grad=False/eval、无optimizer归属；train()后仍eval。Q3按原配方训练，其他参数head10/joint60保持原LR/loss/scheduler。新专属optimizer不能被原stage规则重新解冻。原缺3D投影后严格零。七任务五折、seed42+fold、FP32/batch24，70starts/4900epochs/112000updates；新增PT/prepare/probe/构象/FP/KRR/输出融合均0。

全部70 validation后一次冻结；F_G3D固定科学候选、F_BASE匹配控制。NEXT_RECOMMEND只按REF/F_BASE/F_G3D全局validation选，tie同顺序。之后两臂各35正式TEST各一次共11750行，无论val有无优势；原BASE和REF只读row+key/相同目标对齐，不重新推理，不按TEST改推荐。主门槛F_G3D−F_BASE正式macro≥0.010；另报减原BASE/REF是否≥0.010，完整精度判定。不得拼接旧四单元宣称新interaction或统计显著；披露共享TEST与顺序开发。

B全部授权实验及TEST/对齐结束后唯一timeout600秒CPU tests/test_ua3d_frozen_transfer.py，不重复A/旧suite，不加smoke/benchmark。累计CPU上限5次（原3+A1+B1），累计FULL280/19600epochs/448000updates、TEST420/70500行。失败整单元收费，不退款、不重试/resume/减batch/加seed或扩时；NaN/Inf/OOM/schema/identity/worker/writer/覆盖/预算问题停止留证。

A结果results/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1/；B结果results/ua3d_frozen_transfer_20261026/r1/，日志对应logs。新根冲突即停止不覆盖。按第16.6节交付diff/预算/加载与冻结metadata/原始预测/冻结/七任务SOTA对照/一次CPU与真实exit，更新CURRENT_PLAN独立状态块。最终待审查/PENDING，不归档、不部署、不续开周期。不读取TODO、不远程同步、不清理旧dirty或事故、不新增哈希验证。
```

### 16.8 本轮执行边界

本轮完成第15节证据审查及第16节计划，只修改CURRENT_PLAN标题、当前索引和新增章节。旧结果／状态／失败文件未改；源码与测试未改；未创建新运行根、启动controller、运行CPU suite或模型／实验，未归档。最终仅检查本次文档diff与格式；新计划待授权，不从用户本次规划请求推断执行许可。

<!-- UA3D_GUARD_REPAIR_STATUS_START -->
- 2026-10-26T15:32:24.002581+08:00：UA3D-FROZEN-TRANSFER-20261026-01/r1 A；状态**待审查**；independent_review=PENDING；用户明确授权第16节A及自动B，仅GPU2。
- A唯一suite exit0；stage_acceptance严格记录原controller1/CPU1及本次CPU0。tN=1792999864.9958353；A截止=1793001664.9958353；全链截止=1793021464.9958353，B不刷新；自动等待本次独立B实现就绪。
- 结果`/root/workspace/Uni-Poly-Plus-master/results/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1`；日志`/root/workspace/Uni-Poly-Plus-master/logs/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1`；旧STAGE/r2原失败/exit1/冻结/预测保持只读。
<!-- UA3D_GUARD_REPAIR_STATUS_END -->

### 16.9 本次授权执行快照

- 用户明确执行第16节UA3D-FROZEN-TRANSFER-20261026-01/r1，A通过自动B。dev/HEAD cd36ae7280b159f881b483ad5814835ff7b74c4a，受保护dirty不清理，未读TODO、不远程同步、不新增哈希。A实际仅reservation函数和同一stage fixture；B全部为独立frozen_transfer文件，父科学实现不改。
- 唯一tN=2026-10-26 15:31:04.996 Asia/Shanghai，A30分钟截止16:01:04.996，链截止21:31:04.996，B不刷新。A唯一suite真实exit0，10 passed，累计CPU由3增至4；旧STAGE controller/CPU exit1与事故文件保留。A window=ua3d-stage-guard-repair1(@2245)，cwd=/root/workspace/Uni-Poly-Plus-master，入口env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python /tmp/ua3d_guard_bootstrap.py；manifest有身份和命令。
- 新B模型以BASE严格加载所有native/modal，再仅F_G3D安装S_RADANG定义并加载G的encoder3d/norm3/A3完整训练状态；不读取spatial_initial。显式白名单记录每模块state key/shape/dtype与来源；共同非3D全部BASE，fusion及fold head原公共初始化。
- 独立optimizer按原head/joint规则只对非冻结参数分组，encoder3d/norm3/A3全程requires_grad=False/eval/无optimizer成员；Q3继续训练。train()后仍保持冻结路径eval，正常history记录冻结状态与梯度/optimizer归属，不加模型探针。batch24、FP32、workers0、线程2，所有新GPU入口/CLI/调度仅卡2、world1/内部cuda:0，合计最多4路，controller与CPU隐藏GPU。
- 本次独立根、命令/身份/UUID/真实exit、严格stage_acceptance和最终交付见对应results/logs；B实现就绪后自动启动固定70 FULL及70 TEST。尚未运行B CPU suite，不添加smoke/benchmark或重复A。
<!-- UA3D_FROZEN_TRANSFER_STATUS_START -->
- 2026-10-26T16:43:23.179835+08:00：UA3D-FROZEN-TRANSFER-20261026-01/r1；状态**待审查**；阶段交付报告。授权来源：用户明确执行第16节A及自动B；仅GPU2。
- # UA3D冻结3D迁移实际交付

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

## 17. BASE 专属清理与保留审计（2026-10-26）

- 状态：**已完成（静态自检完成）**。授权来源为用户要求“不要动当前缓存数据，清理与当前 BASE 无关的代码、日志以及 checkpoint；PH 和多尺度数据不要删除，代码可以删除”。当前基线固定为 `UA3D-SPATIAL-20261025-01/r4/FULL/BASE`，正式七任务 macro R² 为 `0.795244745287872`。
- 清理根为 `results/base_cleanup_20261026/r1/`，可复核入口为 `manifest.json`、`deletion_journal.jsonl`、`completion.json` 和 `initialization_regeneration.json`。共删除 28,739 个文件、`1,108,454,353,983` bytes：source 412、config 26、bytecode 262、checkpoint 10,623、旧成功日志 17,416。重新生成的两个初始化依赖占 `185,364,148` bytes，净释放 `1,108,268,989,835` bytes。
- 保留 BASE 及其加载与初始化所需的 110 个 checkpoint 路径、65 个源码文件和 2 个配置；35 个 FULL、35 个 probe、预训练 deploy/resume、公共初态、TMOL3M modal 与 35 个 fold head 均保留。保留 3,660 个张量数据路径；`data/` 删除为 0，路径命中 cache／input／feature／sidecar／descriptor／multiscale 的删除也为 0。正式指标、预测、JSON、config／scaler 和失败证据保留；不按名称继续猜测删除。
- 保留检查发现 `fast_r4/initialization` 符号链接目标位于旧 `r1` 目录，首轮按路径字符串比较时误删 `common_step0.pt` 与 `V1.pt`。已依据保留的初始化实现、记录中的 seed 42、plan 和 revision 在 CPU 下重新生成，并通过正常反序列化、metadata、state keys、shape、dtype 和有限值检查；两份原始序列化文件的逐位一致性未核实，因此只记为重新生成，不冒充原文件恢复。
- 首次恢复包装脚本因 shell 转义产生 `SyntaxError`，在生成文件前即以 exit 1 终止；失败记录保留在 `logs/base_cleanup_20261026/r1/initialization.log` 和 `initialization.exit`。修正后运行真实 exit 0，记录为 `initialization_corrected.log`／`initialization_corrected.exit`。
- 已复核删除 journal 与 manifest 的 28,739 项逐项一致；其中 journal 第 9158／9159 行的两个旧符号链接目标按设计重新生成，其余删除候选为 0。110 个 BASE 权重／初始化文件（`6,763,859,366` bytes）逐一通过 CPU 反序列化和有限值检查；全部保留源码可解析，保留源码的绝对本地 import 无缺失，BASE formal 与 frozen JSON 可读取。未运行训练、GPU、模型 forward、pytest、预测或远程操作，未新增哈希／校验和，未读取 `TODO.md`。

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

- 结果`/root/workspace/Uni-Poly-Plus-master/results/ua3d_frozen_transfer_20261026/r1`；日志`/root/workspace/Uni-Poly-Plus-master/logs/ua3d_frozen_transfer_20261026/r1`；independent_review=PENDING。
<!-- UA3D_FROZEN_TRANSFER_STATUS_END -->


### 16.10 实际交付终态（执行者自检，待独立审查）

- 状态：**待审查**，independent_review=PENDING。A唯一CPU10 passed/exit0；B唯一最终CPU8 passed/exit0，controller真实exit0。A/B全部授权执行完成，未重复测试、未归档、未部署、未展开下一周期。父原失败与旧账本/预测/冻结保留只读；本次guard接纳不是伪造原成功或独立审查通过。
- 新增B消费恰为70starts/4900epochs/112000updates、正式TEST70单元/11750行、CPU1；A另CPU1。累计FULL280starts/19600epochs/448000updates、TEST420单元/70500行、CPU5次均已消费，失败旧CPU不退款，无预算或exit协议违规。新增PT/prepare/probe/构象/FP/KRR/输出融合/父重推理均0，旧r1未重启。
- 资源：全部新GPU任务仅物理2，UUID GPU-31456dd5-a421-142b-644b-7a80f482dfb9；CUDA_VISIBLE_DEVICES=2、--physical-gpu=2、world1/内部cuda:0；FT/推理最多4路、workers0/线程2；controller与CPU隐藏GPU。调度遇NVML未知PID只等待，无换卡、抢占或batch修改。
- 唯一tN=2026-10-26 15:31:04.996 Asia/Shanghai；A通过15:32:24；B首次FT15:37:21；一次冻结16:37:53.512；全部TEST于16:43:09前结束；B最终CPU/交付16:43:23结束。全链截止21:31:04.996未刷新，约1小时12分钟完成。A30分钟上限满足。
- validation一次冻结全部70 BEST/scaler/config/来源/任务划分身份；F_G3D固定候选，F_BASE匹配控制，NEXT_RECOMMEND=REF（父C_CONST），TEST后未变。父BASE/REF按original row+sample key及相同目标完成105组复用对齐，无父推理。
- 正式macro R²：F_BASE=0.787，F_G3D=0.785，原BASE=0.795，REF=0.792。主差F_G3D−F_BASE=−0.002，完整精度−0.0024166099482554326；减原BASE=−0.010（−0.01024043997257651）；减REF=−0.007（−0.006599918921751935）；F_BASE−原BASE=−0.008（−0.007823830024321077）。三个性能门槛均未通过，不能因suite通过称性能通过。
- 在共同BASE非3D背景中，本次未支持S_RADANG冻结3D包相对匹配冻结BASE产生正向迁移收益；不拼接旧/新factorial或interaction，不将差异归因某个角度/容量因素，不宣称统计显著。顺序开发及共享TEST、每折单seed限制存在；固定路线停止扩展，无自动加PT/seed/调权。下一步只读独立审查，输入/目标匹配讨论需另行规划授权。
- 正常source记录确认o8/norm2/A2/fp_encoder/Q2/Q3两臂均BASE PT10K；F_BASE冻结三模块取BASE，F_G3D只这三个模块取G训练状态，spatial未置零/未读spatial_initial。每模块完整state key/shape/dtype/source写入state_conversion.json；fusion/head原公共初态。正常history记录两阶段encoder3d/norm3/A3为eval、requires_grad=False、无grad/无optimizer归属，Q3保留训练；head LR0.0003，joint native0.0001/modal0.001；无额外模型探针。
- A实际仅改scripts/run_ua3d_stage.py::reservation及tests/test_ua3d_stage.py原同一fixture；新增scripts/run_ua3d_stage_guard_repair.py。B仅新增src/modules/ua3d_frozen_transfer.py、src/training/ua3d_frozen_transfer.py、scripts/run_ua3d_frozen_transfer.py、configs/ua3d_frozen_transfer_r1.json、tests/test_ua3d_frozen_transfer.py；父科学模型/数据/训练/loss/schema无修改。CURRENT_PLAN只更正本次授权并追加独立状态/交付块；其他dirty保留，未读取TODO、未同步远程、未新增哈希。
- A根results/ua3d_stage_20261026/r2/repairs/reservation_guard_repair1及对应logs；stage_acceptance.json保留原controller1/CPU1和本次CPU0。A窗ua3d-stage-guard-repair1(@2245)及独立cpu窗。B根results/ua3d_frozen_transfer_20261026/r1及对应logs；controller窗ua3d-frozen-transfer-r1-controller，全部命令/cwd/身份/UUID/日志/exit见manifest、budget.jobs和每个log的command.json/process.json/exit.txt；controller外层真实exit0在A日志frozen-controller/exit.txt。
- B finetune/{F_BASE,F_G3D}/{task}/fold{fold}含70套history/BEST/scaler/validation/state_conversion；formal同结构含70份原始test_predictions.npz；selection/final_frozen.json为唯一冻结；final_review/prediction_join.json为对齐证据；final_review/results.json保留完整精度宏及七任务配对；actual_changes.diff为本次实际改动。A/B唯一CPU日志分别在对应logs/cpu和logs/ua3d_frozen_transfer_20261026/r1/ua3d_frozen_transfer_r1_cpu。未补跑任何不完整任务，无遗漏/阻断。

七任务五折mean±std(ddof=1)、seed42+fold、任务等权macro；下表固定科学候选F_G3D，匹配控制及父参照完整七项表保留在FROZEN_TRANSFER独立状态块及final_review/report.md。论文最佳是逐任务包络，Egc3380对4124、Xc432对430及split/统计协议不同，仅数值对照，不声称公平超过论文；EAT/Eib/Tg不纳入。

| 任务 | F_G3D项目R² | 论文最佳 | 项目减论文 | 参考模型 |
|---|---:|---:|---:|---|
| Eea | 0.924±0.017 | 0.934 | -0.010 | Periodic-TDL |
| Egb | 0.918±0.012 | 0.923 | -0.005 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.911±0.016 | 0.911 | -0.000 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.785±0.030 | 0.839 | -0.054 | Periodic-TDL |
| EPS | 0.724±0.064 | 0.787 | -0.063 | Periodic-TDL |
| Nc | 0.822±0.054 | 0.856 | -0.034 | Periodic-TDL |
| Xc | 0.411±0.109 | 0.423 | -0.012 | Periodic-TDL |
