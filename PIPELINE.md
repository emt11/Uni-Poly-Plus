# Uni-Poly 当前可用路线完整流程

> 历史综合实现说明，包含退役/旧分支及其当时状态。2026-10-17当前主线的完整流程以[MODEL_ARCHITECTURE.md](docs/research/MODEL_ARCHITECTURE.md)、[CHATGPT_HANDOFF.md](CHATGPT_HANDOFF.md)及实际源码为准：FP仅在G2下游作为轻量GATE输入，P2预训练仍无FP目标。历史“无FP”或“尚未执行”描述不能覆盖后续实验事实；实际执行索引见[EXPERIMENT_CATALOG.md](docs/research/EXPERIMENT_CATALOG.md)。

实验结果展示统一保留三位小数；选模、差值和门槛判断使用原始完整精度指标，舍入显示为0.000不代表原始差值严格为零。

核对日期：2026-10-07（Asia/Shanghai）。本文依据当前工作树中的模型、数据接口、训练入口、实际配置和现有结果记录编写，覆盖 GLT Dual、MCL-PH、GP 混合模型及其消融／适配能力。原文中的身份映射、构象、PH、几何标签和缓存语义保留并同步到当前实现；2026-09-30“r4尚未训练”的状态由已完成周期的证据替代。

2026-10-07依据用户“将当前所有路线中的FP预训练任务删除”更新代码、配置与本文：当前预训练不再提供FP预测头、BCE损失或Morgan标签。MCL-PH和O8消融原本没有FP。既有checkpoint和实验结果保留其原始任务口径，本轮不启动重训或正式推理；最终工程测试不等于性能复评。执行授权、预算和停止条件以用户指令及[CURRENT_PLAN.md](CURRENT_PLAN.md)为准；本文不授权重新运行已关闭的实验控制器。

当前 GLT 基线为 **F_CE**（2026-10-10 用户明确指定）。后续 GLT 比较默认参考其已核实配置、PT parent 和七任务五折结果；当前授权与基线绑定见 [CURRENT_PLAN.md](CURRENT_PLAN.md)。F_CE 的七任务等权宏 TEST R² 为 **0.786**（paper5_outer、seed42+fold、五折 mean；原精度见 [正式结果](results/glt_periodic_seam_20261009/r1/final_review/seven_task_results.json) 的 `arms.F_CE`）。历史周期冻结选择 S=B0 和部署 I 的记录保留。

## 1. 路线总览与使用状态

“路线”是模型与完整训练／预测接口；“arm”是同一实现中的受控变化；“配方”改变微调行为，不一定改变模型。代码可构造、存在预训练包、具有完整正式结果、当前被保留使用，是四种不同状态。

|路线／能力|核心结构|当前实现／结果状态|
|---|---|---|
|GLT F_CE（当前基线）|O8＋完整Trimer接缝几何／角度状态增量＋token fusion，融合原子CE监督|当前GLT研究基线；PT 5000步、七任务35折正式结果及独立审查已完成|
|GLT Dual concat／glt_ref|O8周期原子＋Galformer物理键角，图级拼接|历史参考；除Eea外的保留部署I使用此模型，Xc使用r5 standard结果|
|GLT Dual kfuse|O8原子＋一个3D图级knowledge token|代码和配置保留；不是本轮七任务保留模型，当前完整35折证据未在本轮核实|
|GLT Dual优化六组|R_INCIDENT、B_FP、A_ALIGN、F_TOKEN、M_TOKEN_CE、G_SEAM_GEO|六组各35折正式结果已完成；独立审查PASS，均未满足冻结validation晋级条件|
|MCL-PH cat／gate／xattn|O8＋三尺度SchNet专家＋PH路由＋原子融合|三种融合实现存在；旧r13有五任务正式结果，当前direct来源的XATTN保留Eea五折结果|
|O8消融|仅O8＋512维属性head|有GLT来源o8_only和MCL来源m_o8_only；二者预训练来源不同|
|MCL XATTN可学习残差|原XATTN残差常数0.1改为可学习标量|m_xattn_scale为开发arm；没有替代当前保留路线|
|MCL几何目标对照|M0 expert／Mmix expert_mixed_50|两个5000-step包及部分开发结果存在；候选拒绝，旧周期仍待独立审查，没有新正式test|
|GP／GU|保留GLT双编码器，专家端点残差接入3D键状态|两者完整5000-step包存在；有部分任务／fold开发结果，Nc确认失败，无新七任务正式结果|
|GP_A|GP的atom／geometry基础上追加ALIGN|实现存在；正式尝试在800 updates后发生保存事故，未完成5000-step包|
|GLT下游full／head／LoRA／Ridge|同GLT encoder的不同适配方式|通用入口支持；Ridge仅validation-only，不能按现有入口作正式test|
|GLT可选FGR／environment／torsion|额外预训练目标或3D输入能力|代码分支存在，依赖专用统计／词表／配置；当前完整七任务结果未核实|
|F0／SP／E5／GP-F1|既有模型上的微调／预测配方|实现保留，但入口受历史scope限定；不是新的预训练主干，也没有自动获得新执行授权|

项目不是MIPS、Galformer或MI-MoE任一论文的完整复现。GLT和GP实际使用两个模态作下游预测；MCL-PH以距离专家替代Galformer 3D通道。MIPSLocalGraphEncoder及部分旧名称仍可作为共享组件存在，不代表已经关闭的MTS／Distill、GLT-PRED、Atomic-PC等路线重新成为活动训练入口。

## 2. 共同端到端流程

~~~mermaid
flowchart TD
    A["P-SMILES与任务属性表"] --> B["规范RU身份、canonical拓扑与显式映射"]
    B --> C["开放Trimer单构象与物理键角static"]
    B --> S["PI1M索引及paper5_outer任务划分"]
    C --> D["只读direct source"]
    S --> D
    D --> V["预训练：motif mask与noisy view；标签来自clean结构"]
    V --> G["GLT：O8与Galformer"]
    V --> M["MCL-PH：O8与PH路由专家"]
    V --> P["GP：GLT与PH专家键残差"]
    G --> L["各路线自监督目标与optimizer updates"]
    M --> L
    P --> L
    L --> E["resume训练状态与deploy encoder包"]
    E --> F["clean下游输入；加载encoder与任务head初态"]
    F --> H["train-only scaler；head阶段与joint阶段"]
    H --> K["按原尺度validation RMSE保存best"]
    K --> T["冻结候选后恢复best；正式test预测"]
    T --> R["每任务五折指标、论文数值对照与独立审查"]
~~~

预训练不消费属性标签。下游训练只消费本fold的train／validation；正式预测使用保存的best及对应scaler。训练日志中的过程validation属于实验流程，不是额外工程测试。纯encoder deploy尚不足以预测属性，必须配套下游head、scaler和正确的数据接口。

## 3. 数据、构象、身份与划分

### 3.1 当前实际来源

|输入|当前位置|消费内容|
|---|---|---|
|PI1M direct source|data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging/|records.jsonl、行offset、结构LMDB、预训练身份／角色、topology、Trimer、static与预训练目标|
|下游direct source|data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging/|各task JSONL／offset、原始行号与标签、五折角色及共享结构|
|direct预训练索引|[summary.json](results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json)|同目录pretrain_indices.npz中的train／validation／fixed_validation源行索引|
|任务split|data/splits/mips_paper5_outer/{task}.json|五折train_indices／validation_indices／test_indices|
|MCL几何统计|data/processed/mcl_ph_r4_20261001/statistics_train_full.npz|P_train的长度／非键距离归一化统计及PH统计记录|
|MCL noisy PH cache|data/processed/mcl_ph_r4_20261001/trajectory_train_5040000/|按展示位置保存31×5轨迹；可选加速，不能改变noisy view|
|历史共享GLT初态|results/mcl_ph_20260921/shared_pretrain/common_init_v1.pt|201个共享张量，不含FP四个参数|

目录后缀.staging并不代表当前仍未发布。当前direct_source.json记录PASS且reader要求.frozen；这些来源已被r4、r5与GLT优化周期实际消费。不能用2026-09-30的缺口描述替代当前状态。

当前记录：PI1M源行959588，排除与下游重叠身份229，合格959359；P_train为911391，P_validation为47968，fixed_validation为其中1024行子集。下游包含6265行／3655个规范结构；其中七任务5875行，EAT另390行。

|任务|Eea|Egb|Egc|Ei|EPS|Nc|Xc|EAT|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|属性行数|368|561|3380|370|382|382|432|390|

PI1M有958696个有效几何、663个失败身份；下游3655个结构中3647个有效、8个失败。失败结构保留身份和任务行，以无几何分支处理，不能从fold中删除或伪造坐标。来源依据：[PI1M metadata](data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging/direct_source.json)、[下游metadata](data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging/direct_source.json)。

### 3.2 身份与索引空间

canonical RU原子、normalized RU的base atom、Trimer全原子、Trimer重原子和物理键token使用不同索引。通过canonical_to_trimer_base_atom_id、RU offset、mips_to_trimer_central_index和static端点坐标索引连接，不能按数组位置推断对应关系。

同一规范结构可跨task或在同一task多行出现：结构与几何可以共享，标签与fold角色仍按(task, original_row)保留。数据接口是[open_source](src/training/glt_dual_runtime.py)、[DirectMCLSource](src/dataset/mcl_ph_direct.py)；没有direct_source.json时才进入旧cohort／FrozenDualLayerSource路径。当前loader已支持direct索引schema；旧JSON默认路径仍可能指向历史来源，不能只依据配置文件名判断实际数据。

### 3.3 Trimer构造

[trimer_mcl.py](src/dataset/trimer_mcl.py)从规范P-SMILES构造真实接缝连接的开放三个RU，并添加显式H。ETKDGv3＋MMFF94使用first-valid单构象：每轮最多4个候选、最多2轮、全样本共用60秒截止；ETKDG和MMFF松弛各最多200 iterations，不做能量排名。保存全原子坐标、RU身份、中心映射、共价键与几何状态。

GLT的物理键token和MCL／PH都投影到重原子；显式H用于构象生成但不进入这些3D表示。canonical拓扑可能含显式同位素H，必须保留其真实映射与对应fallback语义。

这是有限Trimer的单个构象，不是无限周期链、多构象平均或链间体相环境。训练reader不在线生成坐标。首次建立新来源的构建／发布入口包括build_mcl_ph_next_indices.py、build_mcl_ph_full_geometry.py、publish_mcl_ph_pi1m_direct.py、publish_mcl_ph_downstream_direct.py；现有已发布来源无需因新模型实验重新构建。

### 3.4 paper5_outer

[create_mips_split_manifests.py](scripts/create_mips_split_manifests.py)的实际规则：

~~~text
outer: KFold(n_splits=5, shuffle=True, random_state=100)
inner: train_test_split(sorted outer_train, test_size=0.20,
                        shuffle=True, random_state=42)
~~~

每折约64% train、16% validation、20% test；划分单位是任务CSV行，不是scaffold或规范结构分组。行互斥不保证结构互斥。这些fold已参与路线开发，不能称独立盲测。完整七任务评价为7×5＝35个task/fold；EAT不在当前七任务论文对照中，Eib／Tg不是当前项目任务。

## 4. GLT Dual：输入、编码与输出

依据：[模型](src/modules/glt_dual.py)、[数据](src/dataset/glt_dual.py)、[物理line graph](src/dataset/periodic_line_glt_complete.py)、[static](src/dataset/glt_dual_static.py)。

### 4.1 输入与张量接口

设N为batch内canonical RU原子数，A为所有Trimer重原子数，B为物理重原子键token数，G为图数。

|输入字段|含义／典型形状|
|---|---|
|mips_x／mips_backbone_mask|化学特征N×137及N个backbone标记；模型拼为N×138|
|lga_edge_index／lga_spd|周期lifted局部关系2×R及hop 0／1／2|
|lga_path_index／lga_path_mask|最多两hop的原子路径，宽度3|
|bond_path_features／bond_path_mask|二维路径键化学特征R×2×14及有效位置|
|bond_z_a／bond_z_b／bond_type|真实物理键两端元素与共价键类型|
|bond_distance／bond_center／bond_batch|每键长度、两端都位于中心RU的mask、图编号|
|line_source／line_target／line_path|物理line graph一／二hop关系；路径宽度3|
|line_angle／line_mask／line_path_group|每路径最多两个角度及等长路径的聚合关系|
|geometry_valid／graph_available|几何与二维拓扑有效性；无几何不伪造有效3D状态|
|incidence_index|incident模式新增的键→中心重原子映射，来自真实身份|
|bond_canonical_a／bond_canonical_b|F/M遮蔽物理副本端点所需的canonical映射|

输入没有属性真值或FP目标。监督单独放在labels；下游dataset附带y供loss使用，encoder不读取y。

### 4.2 O8二维主干

BondPathO8以138→512原子投影开始，使用六层512维、八头Transformer。attention bias由SPD、初始原子路径表示和14维键路径化学特征组成；周期关系保留位移身份和backbone标记，不把整个Trimer复制为二维状态。

输出atom_states为N×512，按canonical_graph_index均值池化得到graph_2d为G×512。当前SourceAttention512采用source-Q、target-K及incoming softmax，缩放512^(-1/2)；这是实际实现，不应仅因与常见head-dim缩放不同就判错。

### 4.3 Galformer风格三维主干

GalformerTrimer3D以真实物理重原子键为token。两端元素one-hot投影与256维条件Gaussian距离basis经MLP生成512维键状态；键角路径的Gaussian与位置MLP提供八头bias。六层512维Transformer沿物理line graph传播；等长最短路径先按关系聚合，padding与synthetic self不成为几何监督目标。

输出bond_states为B×512；默认graph_3d为中心内部键均值G×512。跨RU连接键参与编码和消息传播，但默认不进入直接池化及中心内部几何监督。

### 4.4 三种融合与属性输出

|fusion_mode|计算与读出|属性head|
|---|---|---|
|concat|分别LayerNorm(graph_2d／graph_3d)，拼成G×1024|1024→512→1|
|kfuse|一个3D图级token广播到原子，通过OriginalMIPSAttentiveFusion，原子均值池化G×512|512→512→1|
|token|二维原子query与同图全部物理键key/value作四头×32跨注意力；残差投回512；融合原子池化后与原3D图表示拼接|1024→512→1|

KFuse当前只有一个knowledge token，attention softmax恒为1；不能描述为在多个键之间选择。token融合由[BondTokenFusion](src/modules/glt_dual_opt.py)实现，标量残差初值0.1可学习，同图隔离，一个forward计算一次供后续目标复用。两者不是同一机制。

无几何时concat／token的3D图通道为零；token不产生3D残差，kfuse返回原二维原子状态。最终forward输出G×1标量；微调若使用StandardScaler，这是标准化目标空间，评估再逆变换。

### 4.5 GLT预训练任务

[DualPretrainer](src/modules/glt_dual_pretrain.py)的常用目标是：

$$
L_{\mathrm{GLT}}=L_{\mathrm{atom}}+L_{\mathrm{length}}+
L_{\mathrm{angle}}+\lambda_{\mathrm{third}}L_{\mathrm{third}}.
$$

|任务|监督输入／输出|损失与有效性|
|---|---|---|
|masked atom|约30% BRICS motif mask；AtomGraphDecoder两层邻居聚合后输出101类元素logits；clean元素为标签|仅masked原子CE，先每图均值；没有mask目标的图不进入分母|
|bond length|noisy键状态预测clean中心内部键长|原尺度平方误差，先每图均值|
|bond angle|两条键状态的和／绝对差预测clean cos(angle)，输出Tanh|原尺度平方误差；真实无向one-hop角一次；无角图仅有length项|
|ALIGN，third_task=align|2D／3D图表示分别512→256→128并L2 normalize|双向多正例InfoNCE；相同结构身份可有多个正例，invalid／padding不入负例池|
|none（默认）|不构造额外目标|仅atom与geometry；第三项系数／计数均为0|

第三项仅允许none／fgr／align，指定fp直接报不支持。普通GLT配置明确third_task=none及loss_weights=[1,1,0]；不再生成七RU Morgan目标，sample／collate也不返回fingerprint字段。旧direct记录的fingerprint_packed仅作为历史存储字段保留，不进入当前训练；新BRICS target构建和chunk写出不生成它，reader忽略旧manifest中的该数组。

ALIGN的temperature为0.1；常用入口前1000 steps将第三项权重线性升至0.1。负例池来自同一microbatch跨rank的有效样本，四rank×84最多336；accumulation三次不是1008个同时参与对比的样本。

geometry_head_norm只给length／angle decoder前的键状态加非仿射LayerNorm，不改变下游raw3D表示或encoder维度。常用正式包启用此选项。监督decoder在deploy时丢弃，不把预训练CE输出当作属性预测。

### 4.6 六组GLT的当前目标与历史结果边界

|arm|encoder／读出|atom CE|当前额外目标与3D元素输入|
|---|---|---|---|
|R_INCIDENT|历史5000-step concat包，只把3D池化改为中心重原子入射键均值再原子均值|沿用历史预训练|本轮无新增预训练|
|B_FP（历史arm ID）|concat＋内部键读出|未融合O8|当前none；原3D元素输入|
|A_ALIGN|与B相同encoder结构／读出|未融合O8|当前在atom／geometry之上追加ALIGN；原3D元素输入|
|F_TOKEN|多键token融合＋内部键读出|未融合O8|当前none；masked原子在全部RU副本的端点元素置unknown，派生Gaussian type同步变化|
|M_TOKEN_CE|与F相同结构、初始化、mask、读出|融合原子状态|与F相同|
|G_SEAM_GEO|concat＋内部键读出|未融合O8|当前none；仅改变geometry监督覆盖|

G的长度权重为中心端点数量／2：内部键1、单中心端点连接键0.5、外侧键0；每图按有效权重归一化。角监督要求共享物理原子属于中心RU，允许接缝键，不要求两条键都为内部键。R与G分别改变读出和监督，不能混为同一个周期修复。

当前step0由201个公共encoder／atom／geometry张量组成，A另加ALIGN投影，F/M另加共同token融合参数；新模型不存在FP head。旧r1六组35折和初始化说明属于删除前实际实验：B/F/M/G当时使用FP，A当时替换FP；旧205张量包保留，不改写其metadata或结果。B_FP仅保留为历史arm／路径ID，不是可选择的当前FP目标。

删除FP后，concat的LayerNorm读出、kfuse与F_TOKEN融合模块不再由FP直接监督；若仅使用原始O8 CE＋3D geometry，它们没有独立有效融合梯度，F_TOKEN不能据此称为已训练的跨模态融合。M_TOKEN_CE通过融合原子CE继续训练token模块；FGR使用融合context，ALIGN监督两分支图表示。本文不为弥补删除效果擅自新增损失或实验。

F相对B同时改变融合与元素遮蔽，存在混杂；M−F只改变CE入口。六组不能任意组合成未经执行的新路线。F_RAW3D目前仅为对话中的后续建议，尚未列入当前控制器ARMS或形成实验结果。

实际流程：[run_glt_dual_opt.py](scripts/run_glt_dual_opt.py)建立step0／配置→调用GLT预训练→development train/validation→selection_before_test→最终工程测试→formal test-only→[finish](scripts/finish_glt_dual_opt.py)／[collect](scripts/collect_glt_dual_opt.py)。源码中的输出根和预算固定为已关闭r1，不能把再次启动旧controller当作新周期执行。

## 5. MCL-PH：PH路由、多尺度专家与原子融合

依据：[MCL模型](src/modules/mcl_ph.py)、[数据view](src/dataset/mcl_ph_view.py)、[预训练](src/modules/mcl_ph_pretrain.py)。

### 5.1 输入与编码数据流

~~~mermaid
flowchart LR
    T["canonical RU拓扑"] --> O["O8原子 N×512"]
    X["同一Trimer重原子view"] --> PH["31×5 PH与图统计"]
    X --> EX["2 / 3 / 4 Å独立专家；各A×128"]
    PH --> RT["每图三个路由权重"]
    EX --> MX["按图权重混合 A×128"]
    RT --> MX
    MX --> CT["映射中心重原子；N×128 tokens"]
    O --> FU["cat / gate / xattn；N×512"]
    CT --> FU
    FU --> PO["原子均值池化 G×512与任务head"]
    O --> CE["共享元素classifier"]
    FU --> CE
    EX --> GE["共享local与nonbond decoders"]
~~~

新增输入包括mcl_z／mcl_charge／mcl_aromatic、mcl_pos、各尺度mcl_edge_index／distance／type／scale、mcl_trajectory为G×31×5、mcl_atom_batch、mcl_central_index、mcl_bond_index及mcl_geometry_valid／mcl_readout_valid。

这一路线不使用Galformer3D编码器。共享dual-static只是提供真实键／角及索引接口，不意味着共享GLT的3D权重。encoder输出融合原子状态N×512；下游不是拼接两个图级512维向量。

### 5.2 PH轨迹语义

点云为整个Trimer的重原子，未按元素／电荷分型。半径为1.5至4.5 Å、步长0.1，共31点。GUDHI Vietoris–Rips用最大边长4.5 Å，simplex dimension2，在F₂上计算H₀／H₁，保留death=inf的右删失区间；存活条件birth≤r<death。半径图E−V+C不能替代VR H₁。

|轨迹列|定义|
|---|---|
|randic|无向边度关联和乘2／重原子数|
|normalized_wiener|连通分量hop-distance的Wiener归一化，按分量节点对数加权|
|efficiency|有序不同节点对1／hop-distance均值，不可达贡献0|
|betti0_per_atom|存活H₀区间数／重原子数|
|betti1_per_interval|存活H₁区间数／max(1, H₁区间总数)|

前三列是半径图统计，后两列是PH。batch为float32 G×31×5。虽然统计文件含router_mean／router_std，当前router直接消费轨迹，没有额外按这些统计标准化。

### 5.3 三尺度专家与路由

三个独立SchNetExpert固定cutoff 2／3／4 Å，各hidden128、三个interaction blocks。每个专家的邻域是所有d≤cutoff的重原子对，双向储存，包含共价与非键关系；边特征为64维固定RBF＋8维键类别embedding，乘cosine cutoff envelope。节点为元素／电荷／芳香性embedding之和；不更新坐标，没有向量状态。

TopologyRouter为Flatten155→Linear128→GELU→Linear3；预训练updates1–500使用dense softmax，之后Top-2重新softmax。每图一组权重应用于该图所有重原子；专家在当前实现中全部计算，Top-2权重不等于只执行两个专家。

mixed状态A×128经真实中心映射变为canonical tokens N×128。PH用于尺度路由，没有直接PH属性head或PH预测loss。fusion不再次读取各专家或PH曲线选择尺度。

### 5.4 三种原子融合

|模式|实际行为|
|---|---|
|cat|对二维状态／中心3D token分别LayerNorm，128扩展到512，原子级拼接1024后投影512|
|gate|GMU式128维channel gate，输入二维／三维投影及乘积／绝对差；输出h₂＋0.1×投影残差|
|xattn|O8原子query与同图所有中心原子3D tokens作四头×32跨注意力；输出h₂＋0.1×残差|

MCL-XATTN的key/value是中心原子tokens，GLT-F/M的key/value是物理键tokens，不能互换。MCL按图／长度bucket计算attention，并分query chunk；不同图没有交叉attention。m_xattn_scale只在加载原包后将常数0.1启用为可学习标量，不属于原预训练包。

几何不可用或canonical原子无法完整映射到重原子时，融合返回原O8状态；身份／索引错误报错，不能伪装成普通无几何fallback。合法同位素H的重原子读出缺口按该语义处理。

### 5.5 预训练view与监督任务

build_mcl_ph_view只抽取一次随机view：先约30% motif mask，再从全原子坐标抽sigma0.03 Å Gaussian noise；专家邻接、RBF、PH和非键候选来自同一个noisy view。masked二维原子的对应base atom在全部RU重原子副本中同步遮蔽元素／电荷／芳香性。标签来自clean坐标与clean元素，不删除点云节点。

监督为：

$$
L_{\mathrm{MCL}}=L_{\mathrm{atom}}+L_{\mathrm{geo}}+
0.001\left(3\sum_{k=1}^{3}\bar p_k^2-1\right).
$$

|任务|实现细节|
|---|---|
|atom CE|一个共享Linear512→101 classifier分别读取O8状态与融合状态；有几何图取0.5 CE₂D＋0.5 CE融合，无几何图取CE₂D；仅masked原子|
|local geometry|三个专家共享LocalGeometryDecoder；预测clean中心内部键长与内部真实角cosine；使用Huber delta0.5，各图内可用length／angle等权，再三专家等权|
|nonbond distance|至少一端是中心重原子、排除直接共价键；从noisy view按(0,2]／(2,3]／(3,4] Å各最多32对抽样；使用对应尺度专家与共享NonBondDecoder预测clean距离|
|geometry组合|同图local／nonbond都有时各0.5；只有一项则使用该项；空目标不进入有效图分母|
|router balance|有效读出图的dense概率全局均值；采用Top-2前的概率，系数0.001|

长度与非键距离用P_train统计标准化，std下限1e-6；角仍预测cosine。主损失分母按一个optimizer update内所有rank／accumulation的有效图总数；balance按分布式microstep计算并在accumulation内平均，两者不混为同一统计口径。

MCL-PH当前没有FP、ALIGN或MD200第三目标。M0使用local_objective=expert；Mmix使用expert_mixed_50，即local部分取0.5×三专家平均＋0.5×mixed状态预测；nonbond与其余任务不变。混合几何监督不同于追加FP或直接PH监督。

### 5.6 数据collate、缓存与初始化

mcl_ph_collate分别累加canonical atom、heavy atom、physical bond offsets；angle pair索引指向两条物理键，不能使用heavy atom offset。它移除MCL不读取的GLT端点元素／line字段，避免下游误消费未遮蔽副本。

预训练初始化只从GLT common复制O8；新专家、router、atom／geometry heads使用shared_new_init_v2共同初态，fusion按各自模式初始化。不能把GLT geometry decoder权重强行加载到MCL的128维专家decoder。

PH trajectory cache按absolute presentation position保存，不是每分子一条永久曲线。其seed、mask、sigma、数据顺序和展示位置需与训练实际语义一致；只替换轨迹计算，不替换noisy坐标、边或目标。5040000×31×5个float32约2.91 GiB。clean下游view使用sigma0、ratio=None，并按结构缓存CPU样本；collate前克隆，避免PyG .to()修改缓存。

预训练入口为[pretrain_mcl_ph.py](scripts/pretrain_mcl_ph.py)，需要config、cohort/cache/static、statistics、shared-new-init；trajectory-cache可选。当前direct训练配置来自r4实际产物，configs/mts/mcl_ph_*.json的默认历史索引／统计路径不代表当前正式包实际配置。

## 6. GP混合路线：GLT键通道上的PH专家残差

依据：[GPEncoder](src/modules/glt_mcl_ph.py)、[GPPretrainer](src/modules/glt_mcl_ph_pretrain.py)、[联合view](src/dataset/glt_mcl_ph.py)。

GP保留O8＋Galformer，另计算MCL三个专家与PH路由。混合重原子状态128维经物理键的两个端点取值，LayerNorm后构造[和、绝对差、乘积]共384维，通过384→512→512 EndpointAdaptor，按固定0.1残差加入Galformer输出键状态：

$$
h_b^{\mathrm{GP}}=h_b^{\mathrm{GLT}}+0.1\,
f\!\left([z_a+z_b,\ |z_a-z_b|,\ z_a z_b]\right).
$$

中心内部键池化仍512维，与原O8图表示拼1024；属性head1024→512→1。该残差发生在Galformer六层之后，不把专家token插入其每层attention，也不是MCL原子XATTN。

prepare_gp_sample通过同一个MCLPHView产生mask与noise，供Galformer、专家、PH共用；gp_collate同时处理GLT与MCL字段，schema为gp-joint-view-v2。GLT物理键与MCL键端点必须实际对应；不以schema名称替代端点映射。

|route|路由|预训练任务|
|---|---|---|
|GP|PH自适应dense→Top-2|atom CE＋length／angle＋0.001 balance|
|GU|专家权重固定1/3，保留同容量专家／adaptor|atom CE＋length／angle；balance为0|
|GP_A|同GP|在GP当前任务上追加0.1 ALIGN|

GP_A的ALIGN投影为512→512→128，和普通DualPretrainer的512→256→128不同；两个实现不能仅按ALIGN名称视作完全同构。

当前公共初始化使用201个非FP张量，加独立新增专家／router／adaptor；新包schema为gp-common-init-no-fp-v1。GPPretrainer亦可把旧gp-sota-common-init-v2显式去掉fp_head四参数后作为新训练初始化，不能据此恢复旧optimizer或称新包仍使用FP。[build_gp_common_step0.py](scripts/build_gp_common_step0.py)要求指定新的--output目录，不再覆盖历史205张量包。预训练入口[pretrain_glt_mcl_ph.py](scripts/pretrain_glt_mcl_ph.py)支持GP／GU／GP_A、resume、固定四rank；下游入口[finetune_glt_mcl_ph.py](scripts/finetune_glt_mcl_ph.py)支持development／formal及F0／F1配方，正式阶段复用开发best和train-only scaler。

现存GP／GU各5000-step包实际使用删除前含FP的目标，GP_A当时也保留FP；当前无FP实现尚无新预训练或正式结果。GP／GU各5000-step完整包实际存在；GP_A正式尝试只完成800 updates且没有5000-step deploy，不能用step0／smoke包或事故tmp替代。Nc确认折失败，GP未晋级，无新七任务正式分数。入口绑定历史GP scope／输出根及已消耗预算，文档不要求重建历史验收或重新启动其守卫流程。

## 7. O8消融与GLT附加能力

### 7.1 两种O8来源

|arm|预训练来源／加载|微调输入与输出|
|---|---|---|
|o8_only|从GLT deploy抽取O8张量|仅二维topology；N×512均值池化后新512维head|
|m_o8_only|从MCL deploy抽取同包O8张量|同上；用于判断MCL预训练产生的二维主干贡献|

两者都不实例化3D分支，但预训练监督历史不同。与glt_ref比较还涉及512维与1024维head差异；不能把差值直接称为3D或PH的单因素效应。

MCL的fusion_mode=none在模型层可构造，但当前pretrain_mcl_ph.py只允许cat／gate／xattn；实际O8消融入口是上述下游arm，不是已支持的新none预训练命令。

### 7.2 预训练可选分支

|能力|输入／目标与实现|当前边界|
|---|---|---|
|third_task=fgr|真实Trimer中canonical原子SPD2／3距离目标；端点原子状态＋融合图context作对称decoder；归一化目标、Huber delta0.5|需要fgr_mu／sigma／max_pairs等配置；不能描述为额外FP；完整当前35折结果未核实|
|atom_target=environment|按周期bond table和env_vocab生成环境类别，以对应类别数替换atom head输出|需要实际词表和共有参数加载策略，不是元素101类CE|
|torsion_mode|物理重原子quadruplets的cos(phi)、cos(2phi)经小MLP与初值零的gate加入3D token|改变encoder输入及参数，不能直接加载无torsion包或声称新增torsion监督已完成|
|third_task=none|不启用FGR／ALIGN；FP已删除|模型分支已有，不等于为本周期训练了完整结果|

这些是源码可选能力。其词表、统计、初始态、模型package及完整动态流程须由实际配置与产物说明；本文未对它们作模型调用或将它们纳入新预算。当前GLT／MCL／GP均不启用MD200。

## 8. 共同预训练执行与保存

### 8.1 已采用的配方

GLT r4／r1、MCL r4／M0/Mmix、GP主要包均采用5000 optimizer updates、seed42、四rank、microbatch84、accumulation3，有效global batch1008；bf16、AdamW、lr2e-4、weight_decay0、warmup2000、总cosine schedule20000、end_lr1e-9。5000是停止位置，schedule并非压缩到5000。

OrderedSampleStream按固定源索引与seed维护展示顺序。一个update的样本位置从step×1008开始，按microstep、rank和local位置分配；5000步消费5040000个展示位置，不代表5040000个不同结构。BRICS mask先于noise，改变抽样次序会改变输入。

当前GLT默认第三项权重0，只有显式FGR／ALIGN时使用对应额外权重（常用0.1）；MCL balance为0.001；GP按第6节。上面已完成周期的训练预算不代表本轮新增训练。没有属性回归loss进入上述预训练。

### 8.2 package边界

|产物|用途与内容|
|---|---|
|完整step0|各模块初态、新增／删除head及配置；用于同预算从0开始的比较|
|resume_XXXXX.pt|完整pretrainer、optimizer、scheduler／step、各rank RNG、数据位置与模式；用于真正续训|
|deploy_XXXXX.pt|encoder与融合参数、architecture／mode／step等语义；用于下游初始化，不含预训练supervision heads|
|下游best.pt|encoder＋任务head完整状态、scaler、任务／fold／best epoch及来源；用于正式预测|

GLT deploy去掉predictor参数并由下游重建任务head；MCL deploy保存encoder的O8／experts／router／fusion，不带属性head；GP deploy保存GPEncoder并带route/schema。三者不能交叉按文件名加载。

当前无FP模型的完整resume参数和optimizer组与旧含FP包不同，旧包不能直接续训；旧deploy encoder结构未因删除监督head改变，仍可用作下游初始化，其历史训练来源须如实注明。新预训练与性能评估需要新的授权与输出目录。

GLT r1新增readout／fusion／geometry target／third_task／element mask等metadata，并由实际入口消费。MCL导出包括PH语义、cutoffs和推理Top-2模式。加载只需要真实模型结构、参数名／形状和运行语义兼容；不建立新的指纹保护合同。

### 8.3 实际输入依赖与历史接口

老configs、cohort、词表和旧schema不自动与direct来源兼容。legacy reader及部分脚本仍存在摘要验证分支；例如PI1M reader的verify_key_digest，旧cohort／sample index路径和部分历史收口逻辑。本文没有运行这些分支，也不要求其成为执行或验收步骤；按[AGENTS.md](AGENTS.md)不得新增或执行哈希／校验和验证。不能写成“整个仓库已清除所有此类代码”。

普通真实输入不匹配、NaN/Inf、输出冲突按实际错误停止受影响运行。历史归档缺失不能单独阻断工作，不能为满足旧合同补造文件。

## 9. 当前正式微调：Periodic-TDL两阶段配方

实际公共实现为[run_periodic_tdl_epochs](scripts/finetune_mcl_ph.py)。GLT r1控制器、MCL下游及GP下游都复用它；JSON中的旧epochs／LR默认不覆盖这里的实际正式行为。

### 9.1 任务模型与head

|模型|encoder图级输出|属性head|
|---|---|---|
|GLT concat／token／GP|G×1024|Linear1024→512→GELU→Dropout→Linear512→1|
|GLT kfuse|G×512|Linear512→512→GELU→Dropout→Linear512→1|
|MCL／两种O8-only|G×512|LayerNorm→Linear512→256→GELU→Dropout→Linear256→1|

head不能在不同输入维度间强行共享。GLT r1六组同task/fold复用历史GLT head_initial；MCL/O8各512维arm使用build_head的共享初态。对照必须记录实际初始化来源。

### 9.2 数据与目标

clean view不加mask／noise，现有坐标直接生成或缓存输入；geometry失败保留2D fallback。开发时仅打开train＋validation的原始行，通过compact索引映射到dataset局部行号。DataLoader使用FP32、batch24、workers0、每fold seed42+fold、train shuffle，validation不shuffle。

StandardScaler仅以本fold train标签拟合；训练loss为标准化MSE，validation预测逆变换后按原尺度RMSE选best。Xc在r4曾用identity，r5及GLT r1使用train-only standard；这两种结果不能混用参考。其他七任务默认standard，入口仍有明确identity能力。

### 9.3 优化器、训练阶段与选best

|参数|实际正式配方|
|---|---|
|head阶段|冻结encoder并保持eval，训练head10 epochs；lr3e-4|
|joint阶段|解冻encoder；Egc50 epochs，其他任务60；head lr1e-3|
|backbone lr|通常1e-4；Xc2e-4；r6 Xc候选仅降为1e-4且未晋级|
|weight decay／head dropout|Egc 0／0；Xc 1e-4／0.3；其他0.02／0.1|
|encoder dropout|当前正式微调逻辑关闭；head保留任务dropout|
|head scheduler|CosineAnnealingLR，T_max10，eta_min3e-5|
|joint scheduler|CosineAnnealingWarmRestarts，T_0=10，T_mult=1，eta_min1e-6|
|gradient clip|5|
|选择|head与joint全部epoch中原尺度validation RMSE最小的一个best|
|正式预测|恢复这个best与保存scaler，test一次；不额外重训best|

总epochs：Egc60，其余70；全七任务五折为2400 FT epochs、batch24下51450 optimizer updates。训练阶段每epoch的validation是原配方必需流程，不附加smoke或新的测试轮次。

### 9.4 入口行为

finetune_mcl_ph.py支持glt_ref、o8_only、m_cat、m_gate、m_xattn及开发arm m_o8_only／m_xattn_scale；stage为development／paper5_outer／历史smoke。复用开发best的正式分支需要实际冻结选择与同fold来源。

run_glt_dual_opt.py的development／formal分支支持完整六组，formal只加载已保存best，epochs／updates为0；其selection在任何新test之前冻结。GP的formal分支同样复用开发状态；不要假定所有入口的stage名称相同或都能接收新arm。

## 10. 其他微调与预测配方

### 10.1 通用GLT入口的full／head／LoRA／Ridge

[finetune_glt_dual.py](scripts/finetune_glt_dual.py)和[glt_adaptation.py](src/modules/glt_adaptation.py)是另一个下游入口：

|adaptation|训练内容|输出与限制|
|---|---|---|
|full|O8／Galformer及融合／属性head全量训练|按该入口config分组LR与scheduler；不是上述head10＋joint配方|
|head|冻结encoder，保持eval，只训练预测head|使用fusion_head_lr|
|lora|先载入deploy，再把O8／Galformermerged QKV替换为Q/V低秩更新；K与base冻结，任务head可训练|默认rank8／alpha8／dropout0，B矩阵零初始化；恢复须重建相同adapter结构|
|ridge|冻结encoder收集图特征，用sklearn Ridge拟合train目标|当前只允许validation-only，不创建正式test分支|

configs/mts/glt_dual_three_task_*.json中，这个入口的encoder_lr1e-5、fusion_head_lr1e-4、batch32、eval_batch64、epochs100、warmup5、patience10；实际执行以run_config为准。它按validation RMSE和patience早停，不能拿同名GLT输出当作Periodic-TDL同预算对照。

通用入口的development目前限定Xc／EPS／EAT、fold0/1；formal-shard为单task/fold正式单元。Ridge禁止正式模式。不能因为帮助参数支持某个能力就认定当前七任务训练、正式推理接口和产物都完整。

### 10.2 SP、snapshot E5与GP-F1

|配方|科学改动／实现|实际范围|
|---|---|---|
|F0|原Periodic-TDL single-best|GLT／GP开发对照使用；同名配方不是新encoder|
|SP|标准化MSE加(alpha/2)×非head参数相对预训练初态的平方偏移，alpha0.001|finetune_sota_ft.py原入口限定EPS/Nc及历史scope；MCL优化R1有Nc后三折SP-single接续|
|E5|每个完整joint restart cycle选validation-best，取最后min(5,完整cycle数)的原尺度预测等权平均|需要多个任务head/checkpoint与各自预测；不是一次single-best推理；优化R1明确关闭E5|
|GP-F1|公共O8／Galformer／norm的joint LR乘0.3，新experts／router／adaptor用原backbone LR，head1e-3|GP入口实现，受任务和scope限制；已关闭Nc r3用F0而非F1|

[finetune_sota_ft.py](scripts/finetune_sota_ft.py)只有smoke／development，没有正式test loader；scope是实际运行限制。SP／E5的部分开发观测不能填补七任务35-fold正式结果，也不自动等于当前可发布模型。

## 11. 入口与产物导航

### 11.1 数据到训练入口

|阶段／路线|实际入口或公共函数|关键消费／产物|
|---|---|---|
|数据索引／划分|scripts/build_mcl_ph_next_indices.py、scripts/create_mips_split_manifests.py|direct源行与paper5_outer|
|结构准备与发布|scripts/build_mcl_ph_full_geometry.py、scripts/publish_mcl_ph_pi1m_direct.py、scripts/publish_mcl_ph_downstream_direct.py|已有坐标、topology／Trimer／static、身份与offset|
|MCL统计与cache|scripts/build_mcl_ph_statistics.py、scripts/build_mcl_ph_trajectory_cache.py|仅指定P_train和noisy展示；当前产物可复用|
|GLT预训练|scripts/pretrain_glt_dual.py|config、cohort/cache、static、sample index、common/step0或resume→GLT package；无FP target依赖|
|MCL预训练|scripts/pretrain_mcl_ph.py|config、direct source、统计、shared-new-init、可选trajectory→MCL package|
|GP预训练|scripts/pretrain_glt_mcl_ph.py|route、GP共同初态、direct source、PH cache→GP package|
|GLT／MCL正式微调|scripts/finetune_mcl_ph.py、scripts/run_glt_dual_opt.py|指定arm、task/fold、deploy、split、head、scaler→best与预测|
|GP微调|scripts/finetune_glt_mcl_ph.py|route／recipe／stage／scope／checkpoint／reuse|
|旧通用GLT适配|scripts/finetune_glt_dual.py|config／checkpoint、输入来源、adaptation／task/fold|
|SOTA微调配方|scripts/finetune_sota_ft.py|受限scope与F0／SP；开发预测及可选E5|
|现有结果收口|scripts/collect_glt_dual_opt.py及各周期aggregate／finish脚本|读取已保存预测／日志；不产生新的模型结果|

表中给的是接口导航，不是要本轮执行的命令清单。run_mcl_ph_r4_fast.py、run_mcl_ph_r5.py、run_mcl_ph_r6.py、run_mcl_ph_opt.py、run_gp_*、run_glt_dual_opt.py均带历史固定路径、arm集合、阶段或预算；重启可能冲突或覆盖历史运行语义。新周期应使用其明确授权的配置／入口，不能把旧scope当作新授权。

### 11.2 当前package和正式结果位置

|用途|实际路径|
|---|---|
|GLT历史参考deploy|results/mcl_ph_r4_20261002/r3_glt_ref_fast_ra_01/deploy_05000.pt|
|MCL当前XATTN deploy|results/mcl_ph_r4_20261002/r3_m_xattn_fast_01/deploy_05000.pt|
|GLT r1五包deploy|results/glt_dual_opt_20261006/r1/pretrain/{B_FP,A_ALIGN,F_TOKEN,M_TOKEN_CE,G_SEAM_GEO}/deploy_05000.pt|
|MCL M0／Mmix deploy|results/mcl_ph_opt_20261006/r3/R2/pretrain/{M0,Mmix}/deploy_05000.pt|
|GP／GU deploy|results/gp_sota_20261004/pretrain/{GP,GU}/attempt01/deploy_05000.pt|
|GLT r1开发best|results/glt_dual_opt_20261006/r1/development/{arm}/{task}/foldN/best.pt|
|GLT r1正式预测|results/glt_dual_opt_20261006/r1/formal/{arm}/{task}/foldN/test_predictions.npz|
|当前Eea保留结果|results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/m_xattn/eea/foldN/|
|当前Egb／Egc／Ei／EPS／Nc保留结果|results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/glt_ref/{task}/foldN/|
|当前Xc保留结果|results/mcl_ph_r5_20261004/paper5_outer/xc_standard/glt_ref/xc/foldN/|

占位符表示同一组已有目录的结构，不声称每种组合都有产物。尤其GP、MCL开发变体、FGR和适配能力不因路径模式存在就被视为完整正式结果。

### 11.3 正式单元产物

典型下游产物包含best.pt、head_initial.pt、validation_predictions.npz、test_predictions.npz、metrics.json及训练history／progress／runtime。不同入口的字段、indices命名和目录布局不同，应读取实际schema。

预测须能对应task／fold、样本key／原始行、真实标签、scaler、best epoch与实际模型配置。standard指标在逆变换后的原尺度计算。仅有deploy、loss曲线、注意力权重或一个PASS字段，不能说明正式预测完整或性能提升。

## 12. 属性推理与结果解释

完整属性推理顺序：

1. 根据下游best的architecture、fusion/readout、route及adapter模式构造相同模型；恢复encoder和属性head。
2. 使用同一拓扑／Trimer／static身份接口建立clean输入；MCL／GP按clean几何生成PH及专家关系，GLT按clean键角构造token。
3. 进入eval，关闭mask／noise，按保存scaler解释输出；standard预测逆变换回任务原单位，identity输出保持原尺度。
4. 对有标签任务计算原尺度R²／RMSE／MAE，保存样本身份和预测。无标签新结构只需要结构输入与任务best/scaler，不把y喂给模型。

当前正式入口主要是已有来源上的任务评估，并非自动接收任意SMILES的公开预测CLI；任意新P-SMILES仍需合法结构输入准备。LoRA需恢复adapter参数，Ridge需其拟合系数／截距，E5需各component状态与平均规则；不能只提供一个pretrain deploy。

五折R²／RMSE用算术均值、std ddof=0；seed42+fold不是每fold多seed重复。pooled OOF指标必须另列，不替代五折均值。RMSE收益正值表示降低，R²差值正值表示增加。论文表以[SOTA_RESULT.md](SOTA_RESULT.md)为来源，协议等价性未核实，只作数值对照。

## 13. 当前结果与证据边界

截至本次文档更新，GLT-DUAL-OPT-20261006-01/r1已关闭：五包各5000步＋六组210个FT单元，共215个科学训练单元、14400 FT epochs、333700总updates；210个正式预测单元完成。最终测试首轮5 passed／2 failed，用户后续修复授权后7 passed，累计2 calls。旧数据类的单行兼容性修复发生在7pass之后，仅独立静态复核，不能称为动态回归通过。

[独立审查](results/glt_dual_opt_20261006/r1/final_review/independent_report.json)为PASS；[selection_before_test](results/glt_dual_opt_20261006/r1/selection_before_test.json)冻结chosen=I，所有新候选未满足完整validation晋级门槛，未根据test改选。完整六组42项结果见[results.md](results/glt_dual_opt_20261006/r1/final_review/results.md)。

当前保留路线I是按任务明确选择的结果集，不是一个统一的新encoder或由test挑选出的各折赢家：

|任务|保留模型／来源|五折R²|论文最佳模型|论文R²|项目−论文|
|---|---|---:|---|---:|---:|
|Eea|r4 M_XATTN|0.928|Periodic-TDL|0.934|−0.006|
|Egb|r4 GLT_REF|0.918|Periodic-TDL (+Chem, +RC)|0.923|−0.005|
|Egc|r4 GLT_REF|0.901|Periodic-TDL (+Chem, +RC)|0.911|−0.010|
|Ei|r4 GLT_REF|0.803|Periodic-TDL|0.839|−0.036|
|EPS|r4 GLT_REF|0.717|Periodic-TDL|0.787|−0.070|
|Nc|r4 GLT_REF|0.800|Periodic-TDL|0.856|−0.056|
|Xc|r5 GLT standard|0.319|Periodic-TDL|0.423|−0.104|

路线I与r1的H参考不同：H的Eea也用GLT，I的Eea用M_XATTN。五折seed42+fold、算术均值；这些结果不能称公平超过论文或跨seed稳定优势。

MCL M0/Mmix的[旧收口报告](results/mcl_ph_opt_20261006/r3/final_review/report_closeout.json)记录候选拒绝及无新正式预测，且实施者收口不等于独立审查。GP Nc的[独立报告](logs/gp_sota_20261004/nc_r3_independent_review_01/report.json)记录确认失败及保留I。GP_A800步事故、旧失败日志和不完整artifact保持原状，不以本文补齐其状态或创建归档。

本文件描述当前实现；[PROJECT_HISTORY.md](PROJECT_HISTORY.md)记录已关闭周期的实际架构与结果；CURRENT_PLAN记录最新授权／交接。没有完整正式结果的路线写“未完成／未核实”，不能借用I或其他模型成绩填补。后续性能判定应满足用户要求的完整七任务五折，不把旧两折开发筛选重新写成当前验收流程。

## 14. 本次FP删除的交付范围

本次修改模型监督head、loss、sample／collate、输入流、GLT／GP预训练入口、共同初始化与GLT配置，并同步必要测试和本文。没有删除文件，没有改写已有数据、checkpoint、预测或历史实验表；MCL-PH模型与下游属性head保持原有语义。

工程测试只在全部修改完成后集中调用一次，范围覆盖无FP目标、实际数据标签／BRICS、梯度、ALIGN／FGR保留、GP三个arm与初始化／deploy边界；FP删除r1的首次集中测试为41 passed／1 failed，辅助测试NameError已修正。用户随后明确要求重新测试GLT，本轮r2仅GLT四个测试文件一次集中复测33 passed、16.60s、退出码0，原失败项及标签对照修正均动态通过；[本轮报告](logs/remove_fp_20261007/glt_retest_r2/report.json)与CURRENT_PLAN保留两次不同范围的记录。GP／MCL未在r2复测，不宣称全路线整体PASS。本次没有正式训练或7×5性能复评，现有论文对照仍属于历史结果。

## GLT 周期接缝周期（2026-10-09，已完成）

当前基线绑定为 F_CE：配置 [F_CE.json](results/glt_periodic_seam_20261009/r1/initialization/F_CE.json)，编码器初始化 parent [deploy_05000.pt](results/glt_periodic_seam_20261009/r1/pretrain/F_CE/deploy_05000.pt)。属性预测使用 `finetune/F_CE/<task>/foldN/best.pt`，正式预测在 `formal/F_CE/<task>/foldN/test_predictions.npz`，均位于本周期产物根目录；路径及 scaler 见 [原冻结记录](results/glt_periodic_seam_20261009/r1/selection_before_test.json) 的 F_CE 单元。原实验按 validation 冻结 S=B0；后续用户指定 F_CE 为研究基线的记录以 CURRENT_PLAN.md 为准。

授权及当前状态以 `CURRENT_PLAN.md` 的 GLT-PERIODIC-SEAM-20261009-01/r1 为准。固定入口为 `scripts/run_glt_periodic_seam.py`，单元入口为 `scripts/run_glt_periodic_seam_unit.py`，选择／结果汇总为 `scripts/select_glt_periodic_seam.py`；各长命令由 `Uni-Poly` 独立 window 留存真实 exit、cwd、command 和 stdout/stderr。

`geometry_revision=glt_periodic_seam_r1` 明确区分 `finite_incident` 与 `periodic_template`，二者均使用 incident readout 和真实一 hop 角度进入 token 状态的 Gaussian→512 投影。后者在冻结 Trimer 中提取中心局部观测，用 `(i,j,s)` 键身份与覆盖图上的相对 image 建立两跳 line relation；先分别编码左右键长观测，再汇聚周期共享状态，几何头映射回逐物理观测并保留接缝半权重。局部模板不提供远端欧氏坐标。

四臂 F_RAW／F_CE／P_RAW／P_CE 固定 token fusion；raw CE 为明确负对照，fused CE 通过 fused atom states 监督融合。旧 checkpoint 缺失几何字段时按 legacy finite_internal／angle-off 恢复，新包记录几何配置、schema、目标与 mask 语义。通用旧 FT 入口不自动适用新模型；本周期专用入口完整恢复配置并复用原两阶段训练函数和 B0 fold 的预测头初态。

本周期路径为 `results/glt_periodic_seam_20261009/r1/` 和 `logs/glt_periodic_seam_20261009/r1/`。工程测试只有最终一次 `timeout 300s python -m pytest -q tests/test_glt_periodic_seam.py`，已完成一次：19 passed、1 warning、11.42s、exit0。模型历史只在独立审查通过后归档；实际完成状态与预算请读取当前交接记录和本周期台账。

本周期授权执行已完成、唯一 pytest exit=0；独立审查尚未执行。实际预算、七任务结果及预测路径见 `results/glt_periodic_seam_20261009/r1/final_review/delivery.md`，状态为待审查。

本周期独立只读审查PASS，CURRENT_PLAN状态已完成，实际架构与完整结果已归档PROJECT_HISTORY。独立审查原文位于results/glt_periodic_seam_20261009/r1/final_review/independent_review_raw.md；无追加测试或实验。此前“尚未审查”交付语句为当时记录。
