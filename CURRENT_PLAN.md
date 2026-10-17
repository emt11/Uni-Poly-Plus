# UA_2D＋FP GATE 融合优化计划：保留 r10 统一预测政策

计划 ID：**UA2D-FUSION-20261017-01 / r7（GPU2/3、PT最多2路、FT最多4路总并发）**。日期：2026-10-17（北京时间）。

状态：**执行中（P2＋G2）**。用户已授权新预训练20,000 step、每5,000 step保存，使用GPU2/3，PT最多2路、FT最多4路总并发；实际依据见第26节与P2日志。

本文件替代原F_CE长计划；第1—24节为历次规划与现场快照，发生冲突时以第26节r7为准。旧G1累计43次启动、19个旧parent完整FT只读保留；P2 r5因GPU2句柄异常在PT启动前停止。随后GPU3单卡草案仅改代码、没有启动；用户最新已改为GPU2/3范围，两卡现在均可正常查询。本轮新根重新执行PT20k及G2完整70FT，不重跑已完成旧周期，不读TODO。

## 1. 先固定事实与优化目标

- **r10 统一预测融合的正式七任务 macro R² 为 0.796，validation macro R² 为 0.816。** r10 是 G05 神经预测与 Morgan/Tanimoto KRR 的原单位加权输出，不是新的神经 encoder；35 个 validation 单元选择的统一神经权重为 0.75。
- **原 UA_2D 的正式 macro R² 为 0.788。** 它来自 U1／P_ALIGN step5000，预测路径只有 O8、norm2、512→512→1 head，共 19,262,233 参数。预训练用过三维 ALIGN 监督，不代表下游 forward 含三维输入。
- [SOTA_RESULT.md](SOTA_RESULT.md) 的七任务逐项最佳均值包络 macro 为 **0.810**，不是同一论文模型的整体结果。用户所说“宏平均 0.81”暂不能绑定到一个已核实的项目正式模型；不混用 validation、正式结果和论文包络。
- 首要目标是回答“r10 的预测政策能否迁移到 UA_2D”；当前G1改用可学习GATE检验FP特征注入，固定r10的0.75预测权重；原A权重迁移问题保留为历史授权，不作为G1前置依赖。达到 0.810 是后续正式结果的数值目标，不预先承诺。最终仍逐项检查七个任务，不能用 macro 达标替代七项达标。

已核实基线与论文数值对照如下：

|任务|UA_2D 正式 R² mean±std|r10 正式 R² mean±std|论文最佳模型|论文 R²|UA_2D−论文|r10−论文|
|---|---:|---:|---|---:|---:|---:|
|Eea|0.927±0.016|0.924±0.017|Periodic-TDL|0.934|-0.007|-0.010|
|Egb|0.926±0.010|0.923±0.012|Periodic-TDL (+Chem, +RC)|0.923|+0.003|-0.000|
|Egc|0.905±0.016|0.917±0.013|Periodic-TDL (+Chem, +RC)|0.911|-0.006|+0.006|
|Ei|0.803±0.029|0.807±0.016|Periodic-TDL|0.839|-0.036|-0.032|
|EPS|0.768±0.061|0.775±0.049|Periodic-TDL|0.787|-0.019|-0.012|
|Nc|0.813±0.065|0.816±0.049|Periodic-TDL|0.856|-0.043|-0.040|
|Xc|0.373±0.107|0.412±0.120|Periodic-TDL|0.423|-0.050|-0.011|

口径：每任务五折、神经 seed42+fold、每折一次训练；std 为 ddof=0，先任务内五折算术均值，再七任务等权 macro。Egb 的 r10−论文显示为 −0.000，原精度仍为负，不能据展示值判定达标。UA_2D／r10 当前分别为 1/7、1/7 项数值达到论文参考。R²、RMSE、均值、标准差及差值均从完整精度计算后展示三位小数。

数据为 `data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging/`，split 为 `data/splits/mips_paper5_outer/`。项目七任务样本数为 368／561／3380／370／382／382／432；历史文档记载论文 Egc／Xc 为 4124／430，该论文原始数据及精确 split 等价性本轮未核实。共享 validation／TEST 已参与开发，所有后续结果属于探索性比较，不称独立盲测、公平超过论文或多 seed 稳定优势。EAT／Eib／Tg 不纳入本周期。

## 2. 当前实现与迁移边界

### 2.1 已有实际依赖

|对象|当前实际路径／事实|
|---|---|
|UA parent|[P_ALIGN deploy_05000.pt](results/glt_optimization_20261011/r4/U1/pretrain/P_ALIGN/deploy_05000.pt)，存在；不改为 G05 parent|
|UA 配置、head 初态与冻结|[U1 初始化](results/glt_optimization_20261011/r4/U1/initialization/UA_2D.json)、`initialization/heads_2d/<task>/foldN.pt`、[evaluation_before_test.json](results/glt_optimization_20261011/r4/U1/evaluation_before_test.json)|
|UA FT 与正式结果|U1 的 `finetune/UA_2D/<task>/foldN/`、`formal/UA_2D/<task>/foldN/`；35 份 validation NPZ、35 份 TEST NPZ、两组各 35 份 metrics 均存在|
|UA 汇总|[U1 recovery1 正式汇总](results/glt_optimization_20261011/r4/U1/final_review/recovery1/seven_task_results.json)|
|纯二维构造|[Pure2DGLTModel／restore_pure_2d](src/modules/glt_dual.py)、[build_glt_2d_sample／glt_2d_collate](src/dataset/glt_dual.py) 已有实现|
|r10 冻结与预测来源|[selection_before_test.json](results/fce_tune_20261016/global_blend/selection_before_test.json)、[正式汇总](results/fce_tune_20261016/global_blend/final_review/seven_task_results.json)|
|KRR 实现|[xc_kernel.py](src/training/xc_kernel.py)：radius2、2048 位 binary Morgan、Tanimoto、α∈{0.1,1,10}；r10 沿用 r9 已选 α|
|公共数组融合|[fce_kernel.py](src/training/fce_kernel.py) 的 `blend_arrays`、`select_global_weight`、双重身份对齐；旧 runner 的 G05 专属绑定不能直接冒用为 UA|
|当前旧 FP 方案|[fp_graph_fusion.py](src/modules/fp_graph_fusion.py) 是 F_CE 三候选图级注意力，新增 2,033,152 参数；不是 UA 专用模块|

本轮只读核对：UA 与 r10 内嵌 r9 的 35 份 scaler 记录及 TEST indices 一致；r10 的 35 份所选 α validation 缓存存在。只读检查了数组字段 schema，尚未逐数组核对数值身份或计算 UA 新融合指标。执行时在正常数据加载中按 task／fold／original_row／sample_key 对齐并检查 truth、scaler 和 α，不能按数组位置盲拼；这不是另设预跑验收。

当前纯二维 forward 不构造 Galformer；已有 U1 reader 仍会解码包含 Trimer 的存量记录。本计划不为解决存储解码开销重建数据或扩大成 E0 工程项目，也不把存量 reader 的解码等同于模型使用几何。

### 2.2 哪些旧修改迁移、哪些不叠加

|旧修改|本轮设计决定及理由|
|---|---|
|r9／r10 的 Morgan KRR 与统一预测权重规则|优先迁移；直接利用已有独立预测分支，首轮无需重训 UA|
|G05 的 geometry loss 权重 0.5|不迁移到 UA 下游；这是三维预训练干预，换 parent 会同时丢失原 ALIGN 条件|
|Galformer 分组低 LR、非键距离、三维图级蒸馏|不加到纯二维 forward；属于另一个科学问题|
|Huber、LayerNorm-only、radius3 或细化 α／w|不叠加；F_CE 固定配方已有负结果，尚无 UA 证据，不把它们当已证实改进|
|FP_GRAPH_F1 三模态注意力|不直接照搬；保存结果中 NULL／MEAN／KFUSE macro 分别为 0.778／0.776／0.739，且独立审查未关闭|
|新的 FP 融合|G1采用小容量、零初态FP残差＋样本级sigmoid GATE；两臂同门控架构，固定r10的0.75输出政策|

旧 FP 负结果只否定其已执行配方，不能据此断言所有 FP 融合无效。注意力值不作为贡献证明；本计划也不以 `fused−baseline` 命名 interaction。

## 3. A 周期：把 r10 预测融合应用于 UA_2D

### 3.1 科学问题与政策

|项目|预先约定|
|---|---|
|scientific question|不改 UA 神经模型，r10 的统一预测政策能否改善原 UA_2D？在相同分支下，重新用 UA validation 选择全局权重是否优于直接沿用 0.75？|
|reference|已核实 U1／UA_2D macro 0.788；r10 macro 0.796 作项目参照|
|controlled change|A_FIXED 只把 r10 的神经预测替换成 UA_2D，权重固定 0.75；A_GLOBAL 与 A_FIXED 只差全局权重选择|
|comparison budget|同七任务五折、原 UA best／head／scaler／split／seed；完全复用 r10 指定的 KRR 模型、α 及原单位数组；不新增训练或拟合|

对每个 task／fold，只在原单位计算：

$$
\hat y_{A,w}=w\hat y_{\mathrm{UA\_2D}}+(1-w)\hat y_{\mathrm{KRR,r10}}
$$

- **A_FIXED**：全 35 单元固定 w=0.75，作为 r10 结果权重的直接迁移。
- **A_GLOBAL（主政策）**：按 r10 的同一算法，在 w∈{0,0.25,0.5,0.75,1} 中选择一个全局值；目标为 35 个 validation R² 的算术平均，完全同分选更大的神经权重。权重不得按任务或 fold 切换。
- 神经 best 保持原 UA 的 validation RMSE 选择；KRR α 保持 r10／r9 的既有选择，不重新按 UA 调 α。其 α 曾按 G05 的 0.5 融合选择，这是迁移限制，不包装成 UA 最优核方案。
- 本轮同时预先登记 A_FIXED 与 A_GLOBAL；只评估这两个政策，不把全部五种 w 都读取 TEST 后择优。若二者权重相同，直接复用同一输出；w=0 或 1 时复用已有对应分支结果，不重复模型预测。
- 原 r10 与 A_FIXED 的比较包含 parent、ALIGN 与 FT 配方等差异，只能比较完整系统的观测值，不能把差值单独归因为“去掉三维”。

### 3.2 最小实现与正常执行顺序

原A授权清单为 `src/training/ua2d_fusion.py`、`scripts/run_ua2d_global_blend.py`、`tests/test_ua2d_global_blend.py`。r2制定时仅发现第一个辅助文件；runner与测试尚未见，不能把A作为已完成或直接执行不存在的入口。复用现有低层数组融合与对齐，给 UA 使用自己的政策 metadata，不修改旧 G05 校验器以接受伪装身份。

顺序：读取两分支 validation → 计算 175 个网格评分 → 写入完整 35 单元来源、原 UA best／scaler／α、A_FIXED 与唯一 A_GLOBAL 权重并全局冻结 → 才读取本周期正式来源数组 → 输出两政策指标和差值。UA 的 `y_pred` 与核缓存的 `y_krr` 显式适配；禁用 clipping、重新缩放和 TEST 拟合。历史 TEST 已暴露的事实不因新冻结而改变。

### 3.3 A 的拟议硬预算

|项目|新消费上限|
|---|---:|
|PT／FT／KRR fit／特征生成／模型推理／GPU／worker|全部 0|
|validation 单元／网格评分|35／175|
|预登记正式政策|2；每政策 35 单元，共至多 70 个数组算术评估单元|
|新正式融合数组|至多 70 组；同权重或已有等价分支直接复用|
|最终 CPU 测试|仅新增 A 测试 1 次，空 CUDA、timeout300秒；人工夹具，无真实数据预测重放|

拟议结果根 `results/ua2d_fusion_20261017/a1_prediction/`，日志根 `logs/ua2d_fusion_20261017/a1_prediction/`，tmux `Uni-Poly:ua2d_a1_*`。r2只读检查时这两处A目录仍不存在，未发现相关活动作业；不据此推断旧A全部科学消费已核实。

## 4. G1 周期：UA_2D＋FP 的 GATE 融合（替代原 B 固定残差提案）

本修订按用户当前要求把原 B 的固定系数 FP 残差改为**样本级、可学习 sigmoid GATE**。r10 的预测政策保留为固定 0.75 神经＋0.25 KRR，使 GATE 的神经改造与全局预测权重搜索分开。G1 不依赖尚未完成的 A_GLOBAL，不新增 A 的权重搜索，也不消费其剩余预算；原 A 授权及部分实现保留，执行记录见第11—12节。

### 4.1 科学问题、对照与 GATE 定义

|项目|预先约定|
|---|---|
|scientific question|在可学习 GATE 架构下，真实 FP 信息是否优于同容量常量 FP，并改善 UA_2D 神经模型及固定 r10 预测政策的观测结果？|
|reference|原 UA_2D macro 0.788、r10 macro 0.796；UA_FIXED75 为相同旧 UA 预测与 r10 KRR 的固定0.75融合参照；GATE_NULL 为匹配容量对照|
|controlled change|GATE_REAL 对 GATE_NULL 唯一有意差异为 FP 输入取真实 Morgan bits 或每位0.5；两臂都有相同可学习 gate|
|comparison budget|同 P_ALIGN step5000、原二维 head 初态、uniform_u1、split、seed、scaler、模块初始化和训练步数；KRR 模型／α固定，输出权重全程0.75，不选新权重|

设 a 是 O8 原子 mean pool 后经 norm2 的512维图表示；x为2048位FP或常量对照。采用单样本确定性归一化，再用32维FP瓶颈形成门控输入：

$$
p=\sqrt{512}\,\frac{x}{\lVert x\rVert_2},\qquad
q=\operatorname{GELU}(Up),\qquad
\Delta=Vq
$$

$$
g=\operatorname{sigmoid}\!\left(W_g[a;q]+b_g\right),\qquad
a'=a+g\,\Delta
$$

$$
\hat y_{\mathrm{neural,std}}=H(a'),\qquad
\hat y_{\mathrm{G1}}=0.75\hat y_{\mathrm{neural,raw}}+0.25\hat y_{\mathrm{KRR,r10}}
$$

g 的 batch 形状为[B,1]，逐样本产生一个门值并广播到512维；不是任务／fold单独拟合的标量，也不是预测数组的0.75权重。U为2048→32，V为32→512，均含bias；W_g输入维度512＋32＝544、输出1。H严格复用原512→512→1 head初态。首轮只实现此种门，不搜索向量门、多层门或额外attention。

初始化：U常规初始化，V的weight／bias为0；W_g为0、b_g=log(0.1/0.9)，使初始g=0.1、Δ=0、a'=a。所有task／fold／两臂在隔离CPU RNG中共用seed20261017的新模块初态，不消耗FT的42+fold随机流。第一步U／gate梯度可因Δ=0而为0，V必须能得到梯度；V更新后U和gate应存在有效梯度路径，不要求每个真实样本的梯度都非零。

按形状推算，FP瓶颈82,464参数，scalar gate新增545参数，合计**83,009**；总模型**19,345,242**，head阶段新模块＋原head为**346,178**。这些是设计数，尚未实例化核实，执行时记录实际参数与优化器分组。门控制FP残差注入强度；真实／常量两臂只检验FP信息，不足以证明可学习门优于固定残差，未另加第三臂消融。

FP沿用旧FEATURE_SPEC的结构定义：Morgan radius2、2048位binary、chirality／bond type／ring membership开启，保留normalized_smiles中的dummy connection atoms；不把字典中的历史0.5预测权重当作G1政策。非法／全零FP报错。GATE_NULL每位0.5并采用同一归一化；两臂availability、容量、gate及初始化一致。常量臂仍有可学习公共偏置与由a决定的门，不能把它称作无容量或固定门对照。

### 4.2 微调、数据与梯度接线

- 两臂均从原UA parent与各fold原二维head初态开始，不从已微调UA best续训；不换G05 parent，不加三维forward、新构象或PT。
- 严格沿用uniform_u1：所有七任务head10＋joint60、batch24、FP32、workers0、标准化MSE、train-only StandardScaler、clip5；encoder dropout0、head dropout0.1、WD0.02。head LR3e-4、CosineAnnealingLR T_max10／eta_min3e-5；joint encoder LR1e-4、head LR1e-3，WarmRestarts T_0=10／T_mult1／eta_min1e-6。Egc仍joint60，Xc不套F_CE特例。
- head阶段冻结O8／norm2且置eval，训练FP瓶颈、gate与原回归器；joint开放全部参数。将整个FP-GATE与回归器共同注册在model.head中，复用已有head识别及优化器逻辑，参数不得重复注册或遗漏。
- 禁用旧head feature cache，避免缓存含可训练gate／FP残差的过期特征；冻结encoder在线前向，两臂一致，不新建缓存框架。features与forward使用同一GATE路径，head阶段不得把GATE冻结在encoder一侧。
- 使用窄dataset／collate wrapper，显式附加fp_bits[B,2048]，按original_row、sample_key、normalized_smiles绑定；旧glt_2d_collate不会自动保留任意新增字段。
- FP确定性缓存仅放新周期根，按normalized_smiles文本索引；单个进程负责落盘，其他进程不得并写同一缓存。每fold正常FT只加载train／validation；全部best全局冻结后才加载本周期formal的TEST输入，不预跑真实样本probe。
- 每fold仍按原单位纯神经validation RMSE严格改善选best，不按融合指标挑epoch。全部70项best、scaler、FP／gate政策、初态、KRR来源和固定0.75权重先全局冻结，再每best做一次正式神经预测。
- 同一份新神经原单位数组用于计算固定0.75融合；KRR仅复用r10数组，无新fit、forward或α选择。唯一主候选GATE_REAL，GATE_NULL仅作匹配对照，不按TEST在臂、任务或fold间切换。
- 为比较相同输出政策下的模型改动，额外复用旧UA及r10 KRR数组形成UA_FIXED75：有完整A_FIXED结果则直接复用；否则在本周期全局冻结中预登记后，仅做35个CPU数组算术单元，不执行A_GLOBAL网格或额外模型推理。未完成A辅助代码的Phase/FreezeGate只是阶段保护，不等于本节的可学习神经GATE。

### 4.3 原 r2 实现范围、授权预算与产物（已消费部分，r3接续提案见第17节）

原r2授权的以下五个GATE专用文件现已存在，且与失败执行的source_used逐文件直接内容比较一致；当前不将其视为已经最终测试通过：

- `src/modules/ua2d_fp_gate.py`
- `src/training/ua2d_fp_gate.py`
- `scripts/run_ua2d_fp_gate_unit.py`
- `scripts/run_ua2d_fp_gate.py`
- `tests/test_ua2d_fp_gate.py`

复用现有纯二维模型、训练循环与低层数组工具。已有A辅助文件`src/training/ua2d_fusion.py`、旧F_CE／FP-GRAPH源码和产物保持只读，不为G1改写未完成A的逻辑；若确有超出清单的必要运行依赖，先记录具体影响及授权边界。原r2实现已执行并失败；不得直接重启其现有入口，必须先按第17节修复并取得接续授权。

|项目|新消费上限|
|---|---:|
|神经FT|2臂×7任务×5折＝70|
|epochs／attempted及completed optimizer updates|4,900／112,000；每臂2,450epochs／56,000updates|
|训练流程必需validation|每epoch一次，共4,900次|
|正式神经预测|70；每best一次|
|GATE正式数组|140组：70神经＋70固定0.75融合|
|UA_FIXED75参照|最多35个CPU数组算术单元／35组新数组；已有完整同政策结果则复用|
|新增正式数组总上限|175组；参照不增加GPU模型预测|
|FT图材料化／TEST图材料化|47,000／11,750；仅正常数据加载|
|新FP计算|按不同结构缓存，最多5,875次；记录生成与命中|
|PT／teacher／新KRR fit或forward／新构象／额外seed／worker|全部0|
|最终CPU测试|1次新增GATE测试，空CUDA、timeout300秒，人工小夹具|

结果根：`results/ua2d_fusion_20261017/g1_fp_gate/`；日志根：`logs/ua2d_fusion_20261017/g1_fp_gate/`；tmux：`Uni-Poly:ua2d_g1_*`。原`b1_fp_residual/`未执行；G1现有根已包含失败现场，保持只读，后续仅用第17节新recovery根，不能覆盖或重置。

### 4.4 GPU并发：GPU 2、3各最多4个实验

- 仅使用物理GPU2、GPU3。**GPU2同时最多4个实验，GPU3同时最多4个实验，两卡合计最多8个实验**。每个FT或正式神经预测单元是单卡、单进程实验，CUDA_VISIBLE_DEVICES固定为2或3，不启用DDP或额外worker。
- 上限包含本项目已在该卡运行的其他实验及本控制器已预留、尚未建立CUDA上下文的单元；不能只数本周期进程，也不能把“每个window最多4个”解释成每卡无限多组。
- 单个控制器负责派发与槽位记账，在启动前先预留槽位，记录物理GPU、实验单元、PID、启动和真实退出；确认真实退出才释放。避免并发派发竞争同一槽位或同一产物目录。
- 资源不足就等待，不抢占、终止或迁移无关作业；4是上限，不为填满并发强行运行。若发生OOM或数值异常，按停止条件保留现场，不自动缩batch、改变配方或重跑。
- 派发和退出记录足以复核每卡峰值≤4、总峰值≤8；不为此启动独立定时监控、GPU benchmark或额外训练probe。

## 5. 如何判断进展，何时继续优化

原A若另行接续，仍按第3节报告其两政策。G1报告GATE_REAL−GATE_NULL（同GATE架构下的FP信息）、GATE_REAL神经−原UA（完整神经改造）、GATE_REAL固定融合−UA_FIXED75（固定预测政策下的模型改造），及GATE_REAL固定融合−r10（跨parent／训练条件的系统数值比较）。均报告七项均值、逐折差、退化任务、validation→TEST差与收益集中度。门值仅作观测诊断，不作因果贡献或独立的gate优势证据。

macro 正增量仅称本协议下的观测改善；沿用旧 +0.010 作为探索性进展里程碑，分别列出机制比较与对 r10 的项目比较，不能代替用户的 0.810 目标或七项论文目标。不设按 TEST 拼赢家的晋级规则。原单位 RMSE 差为候选减参考，负值较好；不跨物理单位平均 RMSE。

GATE_REAL若不能在匹配GATE_NULL与UA_FIXED75参照上给出整体正向证据，本固定方案收口，不自动追加rank／门宽／门初值／FP半径／注意力网格。只在已有结果定位到具体问题后再提出一个受控优化周期：例如更新强度或图级监督，二者不同时改。本轮没有为此授权代码、PT、延长训练或任何新预算。

## 6. 最终一次性测试、停止条件与交付

本轮为失败只读复核与CURRENT_PLAN维护，完成后集中一次检查本轮文档实际diff、格式、引用与预算算术，不运行pytest或科学评估。r3若获授权，将最小返修及完整接续合并一轮：全部实现、新70FT、正式预测及参照算术完成后，仅运行一次GATE CPU测试。原G1测试未运行，不另加返修前测、smoke、旧回归、benchmark或GPU probe，也不重跑A。原A若独立接续，仍遵守其独立周期的最终一次性检查，不把它附加成G1的前置验收。

A 的测试回答：双重身份乱序对齐、单一全局 w／tie、冻结先于 TEST、α 不变、0 fit 和等价结果复用是否正确。G1的测试回答：初始g=0.1且与原UA函数一致、[B,1]门及广播、真实／常量同容量、V更新前后梯度路径、head／joint参数分组、FP collate绑定、可训练特征未缓存、固定0.75融合及并发槽位计数是否正确；全部使用人工小夹具，不重放正式实验。唯一命令：`CUDA_VISIBLE_DEVICES='' timeout 300s /opt/conda/bin/python -m pytest -q tests/test_ua2d_fp_gate.py`。

运行中遇 parent／head／schema／scaler／样本身份不匹配、NaN／Inf、FP 非法、梯度路径错误、writer 冲突、正式目录覆盖、冻结前读取 TEST 或预算超限，停止受影响运行及后续依赖派发，保留日志和现场并报告；不静默跳样本、修正异常或自动重跑。仅历史归档缺失不单独阻断正常执行。

所有预计超过一分钟、使用 GPU、启动 worker 或写训练产物的命令必须在 tmux session `Uni-Poly` 的独立 window 运行，保留 command、cwd、环境、日志、产物和真实 exit；tmux 不可用则报告具体阻断。新根存在时拒绝覆盖。不得新增或执行任何哈希／校验和验证。不得删除、部署、pull、commit、push，或改写 SOTA 表。

正式交付至少包含 `selection_before_test.json`、`completion.json`、实际消费账本、`final_review/seven_task_results.json`、`delivery.md`、逐 fold 差、模型／数组路径索引、实际增量 diff、唯一最终 CPU 日志及真实退出记录。自检不能代替独立审查；独立审查复用现有证据，不重跑实验或测试。

## 7. 本周期七任务审查表与当前结果

以下为 A／G1 新结果占位，不能用旧参考、smoke 或缺项补齐。每个实际完成的政策均需独立报告 R²／原单位 RMSE mean±std、对原 UA／r10 的差及下表论文差；五折／seed42+fold、ddof=0、日志与产物路径必填。A只有部分实现、完整执行未核实；G1已启动8个FT、完成0个，无正式结果，全部七任务未完成／不可比较。现无可归档的新性能结果。G1同时填写GATE_NULL、UA_FIXED75及原UA／r10的完整对照，不借用A_GLOBAL充当结果。

|任务|A_GLOBAL 正式 R²|GATE_REAL 神经／融合 R²|论文最佳模型|论文 R²|候选−论文|状态／原因|
|---|---|---|---|---:|---|---|
|Eea|未完成|未完成|Periodic-TDL|0.934|不可比较|A完整执行未核实；G1失败，无正式结果|
|Egb|未完成|未完成|Periodic-TDL (+Chem, +RC)|0.923|不可比较|A完整执行未核实；G1失败，无正式结果|
|Egc|未完成|未完成|Periodic-TDL (+Chem, +RC)|0.911|不可比较|A完整执行未核实；G1失败，无正式结果|
|Ei|未完成|未完成|Periodic-TDL|0.839|不可比较|A完整执行未核实；G1失败，无正式结果|
|EPS|未完成|未完成|Periodic-TDL|0.787|不可比较|A完整执行未核实；G1失败，无正式结果|
|Nc|未完成|未完成|Periodic-TDL|0.856|不可比较|A完整执行未核实；G1失败，无正式结果|
|Xc|未完成|未完成|Periodic-TDL|0.423|不可比较|A完整执行未核实；G1失败，无正式结果|

## 8. r1 规划与历史迁移记录（历史快照，r2更新见第12节）

|事项|实际状态与依据|
|---|---|
|开始现场|已读 git status、相关 diff；工作树已有 CURRENT_PLAN、共享 PT／FT 源码修改及多个未跟踪实验文件，全部保护|
|静态定位|读取 UA／r10／FP 科学源码、冻结记录、实际正式汇总、completion 和已有 CPU 日志；无模型 forward|
|迁移可行性|UA 两类预测各35份、metrics各35份存在；35份 scaler／TEST indices 与 r10 对应一致；核 validation 缓存35份存在|
|文档修改|覆盖本文件；向 PROJECT_HISTORY 追加旧结果及待审查快照，不覆盖原有归档|
|新科学消费|PT0、FT0、fit0、FP生成0、模型推理0、正式新融合评估0、GPU0、worker0、pytest0|
|新运行日志／产物|r1规划时无；A部分实现及G1拟议路径见第11—12节|
|删除／保留|无文件删除；旧实验代码、checkpoint、日志、预测和事故现场全部保留|
|最终检查|两份文档完成后集中核对，结果见本轮交付；不把规划写成实现或实验完成|

## 9. 历史结果迁移的范围与证据边界

迁入历史的范围：原参数周期参考、T1／T2／T3a／T3b、r6—r16，以及 FP-GRAPH f1；U1／UA_2D 仍引用原已归档结果。本轮从现有完整精度汇总格式化 R²／RMSE／论文差，复用已有 completion 和 CPU 日志，没有重算原预测指标、恢复 checkpoint 或重测科学代码。

历史中的独立审查结论沿用被替换交接的明确记录，注明本轮未重复独立审查；其余只称执行／复核快照。旧 completion 的 NOT_PERFORMED 和 r15 原 test_exit=1 均不回写成通过，r15 的指定返修证据单列。不因计划更换丢弃负结果、失败尝试或未完成事项。未运行的 E0／A1 后续、其他 PT／几何／质量候选均不自动继承预算。

## 10. 尚未关闭的历史审查交接

- T1／T2 原交接明确记录独立审查通过；T3a 科学审查通过后交付问题在 T3b 补交，补交复核由执行主体完成，不升级为全周期独立审查关闭。r15 原交接明确记录返修后独立审查通过并按负结果关闭。本轮只迁移状态，不重新作这些审查。
- T3b、r6、r7、r8、r9、r10、r11、r12、r13、r14 保留“待审查／执行后复核通过”；没有剩余授权科学单元，不重复消费其训练、fit、预测或 CPU 测试预算。
- r16 与 FP-GRAPH f1 保留**待审查**：各自完成70／105 FT和正式神经预测，已有唯一 CPU 为11 passed／10 passed、exit0；本轮仅读取报告和日志，尚未对全部源码、best 和预测作独立审查。不能把这次历史迁移当作审查通过。
- 历史 E0 有2 failed交付、A1 仅初始化 PASS；这属于旧 GLT 架构路线的 r10，与本计划引用的 F_CE 全局预测融合 r10 不是同一个周期。旧事故完整保留，不把它们当成 UA 融合已经实现或必须重跑的前置依赖。


## 11. A 周期授权执行记录（2026-10-17，原交接快照）

已核对 git status 与 CURRENT_PLAN 相关 diff，保护全部既有改动，未读取 TODO.md。仅新增三个指定文件并维护本计划；旧源码、产物和事故现场只读。最终测试仅在科学执行完成后运行指定 pytest 一次，预算300秒、CPU、人工夹具。结果和日志根已预登记，拒绝覆盖。自检后交付待独立审查，不提前归档历史。

## 12. r2：用户要求改为 GATE 与 GPU2／3各4并发

- 授权来源：用户当前明确要求“修改计划为GATE融合，以及GPU2、3每路最多四个实验并发，重新给执行提示词”。本轮完成计划修订与提示词；G1代码和科学预算仍为待执行授权提案，不在本轮启动。
- r1第11节“新增三个文件”等表述保留为原交接，不能视为本轮已独立核实。当前仅见`src/training/ua2d_fusion.py`，未见A runner／测试；结果／日志根不存在，tmux未发现ua2d窗口，相关进程列表未发现UA实验writer。未重跑或清理部分实现。
- 当前约定：GATE指UA图表示与FP特征之间的样本级sigmoid残差门；固定r10的0.75神经＋0.25KRR政策。G1不搜索新w，不依赖A_GLOBAL完成。两臂仍70FT，不新增第三臂；仅新增最多35组UA_FIXED75参照数组，无新神经预测。
- 新GATE设计新增83,009参数；每卡最多4个单卡实验、两卡最多8个，包含其他活动实验与待建立CUDA上下文的预留单元。代码和实际参数／峰值尚未执行核实。
- 本轮修改文件仅CURRENT_PLAN.md，PROJECT_HISTORY.md保持原有结果记录；没有新代码、训练、预测、FP计算或pytest。旧A文件、checkpoint、数据、日志、正式结果与事故现场全部保留，无文件删除。
- 实际G1日志／产物：尚无；第4.3节为拟议路径。本轮不创建训练目录、不改动现有tmux或进程，也不启动定时监控。
- 最终检查：本轮文档完成后集中一次核对实际diff、格式、引用、GATE公式／参数数量与并发约束；结果在本轮交付说明。后续科学完成后状态为待审查，自检不写成独立审查通过。


## 13. G1 当前授权执行记录

开始核对 git status、相关 diff、计划与已有二维训练入口；全部既有改动保护，不读 TODO.md。仅新增第4.3节五文件并维护本计划。共享源码和旧产物只读；单控制器负责FP落盘及GPU2／3槽位，最终CPU测试仅在全部科学执行完成后一次运行。未独立审查，不提前归档。


## 14. G1 实际交付记录

执行在finetune停止：RuntimeError('GPU concurrency exceeded')。日志/root/workspace/Uni-Poly-Plus-master/logs/ua2d_fusion_20261017/g1_fp_gate/controller/，failure.json与消费账本在/root/workspace/Uni-Poly-Plus-master/results/ua2d_fusion_20261017/g1_fp_gate/。没有自动重试或追加预算；未完成任务不可比较，不能报告性能或论文差。


## 15. G1 失败现场补充与七任务交接（2026-10-17）

G1 状态：需返修；本次科学执行已停止，没有自动重试。

触发：2026-10-17 14:55（北京时间），FT阶段GPU concurrency exceeded。控制器真实exit=1；8个本周期子进程均已确认exit=-15，无遗留G1 writer。
派发：GPU2为Eea fold0/2/4、Egb fold1；GPU3为Eea fold1/3、Egb fold0/2。两臂共70FT中仅启动8个GATE_REAL、完成0个；其余未派发。

已新增五个授权GATE文件并维护CURRENT_PLAN.md；实际可学习sigmoid门、32维瓶颈、零V、初始g=0.1；实现未经最终CPU测试，不称通过。
账本已记录FP生成586、optimizer attempted/completed 24/21；完整epochs0、正式预测0、融合/参照数组0、CPU测试0。PT/teacher/KRR fit/forward/额外seed/worker均0。

证据限制：并发触发时的observed PID集合未落盘，无法独立区分重复计数、外部实验或真实超限；concurrency.json仅为最后一次成功计数快照，不是触发时实际峰值证明。
证据限制：ft_graphs=0是单元末尾input_rows.json尚未发布的计数缺口，实际已部分材料化；实际图数未核实，不能表述为0。
证据限制：attempted/completed=24/21为已落盘hook计数；SIGTERM可能发生于更新与记账之间，不等于完整科学消费已精确恢复。
证据限制：参数数量在build_model构造中校验通过后进入训练，但无单元完成首epoch，parameter_groups.json未发布；不伪造完整参数分组产物。

|任务|G1 REAL／NULL神经及融合|UA_FIXED75|论文最佳模型|论文R²|候选−论文|
|---|---|---|---|---:|---|
|Eea|未完成／不可比较|未执行|Periodic-TDL|0.934|不可比较：无正式结果|
|Egb|未完成／不可比较|未执行|Periodic-TDL (+Chem, +RC)|0.923|不可比较：无正式结果|
|Egc|未完成／不可比较|未执行|Periodic-TDL (+Chem, +RC)|0.911|不可比较：无正式结果|
|Ei|未完成／不可比较|未执行|Periodic-TDL|0.839|不可比较：无正式结果|
|EPS|未完成／不可比较|未执行|Periodic-TDL|0.787|不可比较：无正式结果|
|Nc|未完成／不可比较|未执行|Periodic-TDL|0.856|不可比较：无正式结果|
|Xc|未完成／不可比较|未执行|Periodic-TDL|0.423|不可比较：无正式结果|

以下仅复用已有正式汇总，不是本轮新评估；5折seed42+fold、std ddof=0、三位小数，不跨单位平均RMSE。

|任务|原UA R² mean±std|原UA RMSE mean±std|r10 R² mean±std|r10 RMSE mean±std|
|---|---:|---:|---:|---:|
|Eea|0.927±0.016|0.289±0.026|0.924±0.017|0.294±0.025|
|Egb|0.926±0.010|0.528±0.045|0.923±0.012|0.539±0.068|
|Egc|0.905±0.016|0.479±0.036|0.917±0.013|0.450±0.031|
|Ei|0.803±0.029|0.433±0.066|0.807±0.016|0.429±0.058|
|EPS|0.768±0.061|0.530±0.066|0.775±0.049|0.527±0.078|
|Nc|0.813±0.065|0.101±0.010|0.816±0.049|0.100±0.006|
|Xc|0.373±0.107|18.627±1.526|0.412±0.120|18.017±1.559|

已有原UA／r10 macro R²：0.788／0.796。G1候选对照、逐fold差、退化任务、validation→TEST差及收益集中度均未完成，不以中断训练或旧结果填充。
共享validation／TEST已参与开发，仅探索性数值比较；与论文数据、split及评估统计协议等价性未核实。门值不作因果贡献，当前两臂设计也不能证明GATE优于固定残差。
冻结记录、完整参数分组、正式预测和唯一CPU测试均未形成。原failure.json、completion.json、原始consumption与退出记录保留；本补充明确记录缺口，不改写为通过。
日志：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_fusion_20261017/g1_fp_gate/controller/、ft_*/；tmux Uni-Poly:ua2d_g1_*。实际diff在final_review/g1_failure_delivery.diff；既有源码与产物未修改，不接续A_GLOBAL或追加候选。
下一步需先修复并发现场证据与消费记录缺口；任何继续执行都须明确承接已失败8次启动及其实际消费，不自动重置预算或重跑。

## 16. G1 失败只读复核（Codex，2026-10-17）

**结论：需返修成立；不能把本次异常定性为已证实的并发误报。** 本轮未参与G1科学代码实现；直接读取当前五文件、source_used、failure／completion／消费账本、dispatch、8份单元execution／exit、controller原始日志和只读FP缓存。五份当前文件与执行副本直接内容相同，没有执行哈希、pytest、模型构造／forward、数据材料化或实验重放。仅维护本计划，不修改PROJECT_HISTORY或旧产物。

### 16.1 已核实事实

|对象|核实结果|
|---|---|
|退出|controller真实exit1；8个FT单元execution与exit.txt均为−15；这9个工作PID当前均不存在，对应tmux pane均dead；未发现G1遗留writer|
|派发|仅8个GATE_REAL：Eea fold0—4，Egb fold0—2；GPU2／3各派发4个，无第9次派发记录|
|最后成功快照|`concurrency.json`为14:55:31.973（北京时间），两卡各4、总8，8个实际child PID已经全部绑定；不是异常触发时观测或实际峰值证明|
|异常位置|`schedule`调用`observe_gpus`后，`Slots.record`抛出`GPU concurrency exceeded`；原始traceback与当前源码一致|
|现场丢失路径|`record`在更新峰值前即抛出，controller在本轮尾部才写`concurrency.json`；catch先stop_own清空active，再保存failure，故触发时observed／active证据没有落盘|
|单元结果|`ft/`内8个目录无best.pt、resume、scaler.json、progress.json或metrics.json；不能断点恢复optimizer／scheduler／RNG，不能称8个任务完成|
|更新记录|6个单元有updates.json，合计attempted24／completed21；另2个单元没有更新文件，实际消费不能因此填0；无完整epoch交付记录|
|输入记录|input_rows.json只在成功尾部写入，8个目录均无该文件；ledger据文件缺失给出ft_graphs0，与已发生训练不一致，该0只代表缺记录|
|FP缓存|只读SQLite查询bits表恰有586条记录，value长度均2048字节；这是缓存行数证据，未重新生成或重新验证全部位向量|
|正式阶段|无全局冻结、formal目录、候选／参照正式数组、最终CPU日志；已有final_cpu_not_run.json明确NOT_RUN|

异常发生前8个child PID已在成功快照中绑定，因此**不能直接用“尚未绑定PID的启动窗口”解释本次失败**。当前occupancy按“自身预留实验数＋observed中未归属自身的PID数”计数；observed合并nvidia-smi和按cmdline／CUDA_VISIBLE_DEVICES筛选的/proc进程。源码存在归属／采样时间窗的脆弱点，但触发时原始PID集合已丢失，无法确认是否为包装／子进程重复计数、PID视图差异、真实外部实验或其他分类问题；不事后伪造根因或峰值。正确返修首先是保留当时证据并使归属／去重可靠，不能删除外部进程计数、提高4路上限或把“自有8路”当全卡真实占用。

### 16.2 失败消费的已知值与保守占用

|失败单元|原train＋validation图数上界|落盘attempted|落盘completed|
|---|---:|---:|---:|
|GATE_REAL／Eea fold0|294|10|10|
|GATE_REAL／Eea fold1|294|1|0|
|GATE_REAL／Eea fold2|294|5|5|
|GATE_REAL／Eea fold3|295|6|6|
|GATE_REAL／Eea fold4|295|1|0|
|GATE_REAL／Egb fold0|448|1|0|
|GATE_REAL／Egb fold1|449|未落盘|未落盘|
|GATE_REAL／Egb fold2|449|未落盘|未落盘|
|合计|2,818|已知24|已知21|

2,818来自这8个fold的已冻结train／validation行数，是旧图材料化的**保守预算占用上界**，不是恢复出的实际图数。代码每index只在首次正常读取构造并缓存，本次每单元只有一个进程、workers0、没有重试；不为恢复精确图数重新加载图或跑旧单元。

更新前／后hook在同一进程中顺序计数并写盘，写盘返回后才继续下一个step。SIGTERM可处于一次更新与记账之间。按已读单步顺序路径，旧attempted保守上界为24＋8＝32，completed保守上界21＋8＝29；预算提案统一以**32次旧更新占用**承接两类计数，保留原24／21作为已知值，不能把32伪装成实测。如果实际执行路径存在超出此界的证据，停止并重新说明预算，不静默吸收。该保守界不是补造精确训练记录。

原FT启动上限70已消费8，剩余62次；0个完整best且没有resume依赖，若维持该启动上限，就无法补齐70个正式FT单元。完成原科学矩阵所需的新70次包含8个失败单元从原初态重启及62个未启动单元，必须明确增加累计启动额度，不能把失败8次从账本消失。

## 17. r3 最小返修与接续提案（待授权）

### 17.1 科学范围不变

|项目|约定|
|---|---|
|scientific question|仍为同GATE架构下真实FP相对常量FP的作用；本轮返修不引入新科学候选|
|reference|原UA、r10、UA_FIXED75及GATE_NULL；没有G1完整性能结果可作新基线|
|controlled change|仅修正并发进程归属／快照时序、消费记录、停止清理和recovery绑定，不改模型／输入／loss／训练配方|
|comparison budget|原七任务五折、两臂、parent／head／gate初态、seed42+fold、head10＋joint60、batch24／FP32／workers0及固定0.75预测政策全部保持|

返修允许修改：`src/training/ua2d_fp_gate.py`、`scripts/run_ua2d_fp_gate.py`、`scripts/run_ua2d_fp_gate_unit.py`、`tests/test_ua2d_fp_gate.py`及本计划。模型文件`src/modules/ua2d_fp_gate.py`和共享科学源码不改；不加入第三臂、额外seed、缩batch、换loss或新权重网格。优先原入口窄改，不建通用调度／审计框架。

### 17.2 并发与故障现场

- 保持物理GPU2、GPU3各最多4个实验、总最多8个，包含已存在的其他实验和本控制器预留单元。一个自有实验的预留、wrapper、真实worker和CUDA上下文只能归属同一个实验；按实际execution／进程关系识别，不把任意同名Python自动视为自身。
- 先持久化实验派发意图和槽位预留，再启动wrapper；真实worker PID发布后绑定原实验。采集GPU进程期间若启动信息变化，先协调最新绑定再分类，不能将同一自有单元同时算作预留和外部PID。无法解释的PID保留为未归属／外部，不通过忽略它绕过上限。
- 正常派发／回收循环保存本次观测的nvidia原始PID、/proc筛选PID、wrapper／worker归属、待启动预留、外部或未归属PID、每卡计算值和时间；分类或上限异常时**先落盘同一份异常快照，再抛错和清理自身子进程**。保留停机前与停机后两份状态，不用清空后的active替代现场。
- PID视图、进程归属、外部真实超限等无法解释时，按原停止条件停止本周期，不冒称根因已解决。仅修改日志而保留已知可重复计数时间窗不算完成返修；也不能仅把并发降为1规避原4路要求。
- 不建立额外GPU探针或定时监控；复用已有正常控制循环。任何停止仅针对本控制器已绑定的自有单元，保留真实退出，确认退出后才释放槽位。

### 17.3 消费与中断记录

- 区分“旧失败已知计数”“旧失败保守占用”“本次新执行”“累计预算占用”。ledger不能每次只从当前jobs清零重算而丢掉8次旧启动、旧FP或旧输入消耗；缺文件用unknown／保守占用表达，不再填0。
- 在正常数据加载时增量记录每单元材料化attempted／completed／in-flight及原行与身份，在记录异常前先发布现场；不能等70个epoch成功结束才写input_rows。每单元独立writer，不让多进程写同一文件。
- 更新前／后hook记录明确step序号与计数；SIGTERM用受控停止路径尽可能在正常边界落盘，保留在途状态。硬终止无法精确判断时保留unknown及保守占用，不能承诺物理更新绝对精确，更不能以写完账本为由继续额外训练。
- 首次正常step可尽早保存实际trainability与可取得的优化器证据；若接口尚无完整分组证据，就标未形成，不能把设计参数数当运行证据。不要为此修改共享训练循环或额外执行forward。
- 旧8个失败单元没有resume，恢复方案是从原parent／head与已有gate初态开始新的完整FT。新的成功70项用于科学汇总；旧失败仅进入消费和事故索引，不混进性能统计。最终完整性检查仍要求新矩阵恰好70个完成FT、112,000更新及原正式预测矩阵，不能删除检查来适应含失败的累计计数。

### 17.4 接续预算：明确承接失败消费

以下均为**新授权提案，当前尚未执行**。原7任务、5fold、2臂、epoch和seed范围不变；增加的仅是事故后的启动与已消耗工作占用。

|项目|旧失败已知／保守占用|recovery新执行上限|累计上限／解释|
|---|---|---:|---|
|FT启动|已启动8|70：8个重启＋62个首次启动|78；不是原70次额度重置|
|FT完成|0|70|70个完整科学单元|
|完整训练epoch|未形成epoch交付，旧片段另计|4,900|新科学矩阵仍4,900；旧片段不能解释为零消费|
|attempted／completed updates|落盘24／21；旧预算各保守占用32|各112,000|两类累计预算各112,032；旧实际精确值仍未核实|
|FT图材料化|实际未知；保守占用2,818|47,000|49,818；2,818不是旧实测图数|
|FP生成|586条已存缓存|至多5,289次新计算|5,875；复制或读取旧bits不计新计算|
|正式神经预测／GATE数组|0／0|70／140|70／140|
|UA_FIXED75算术／数组|0／0|最多35／35|最多35／35；已有等价结果可复用|
|TEST图材料化|0|11,750|11,750|
|最终CPU检查|0|1次，timeout300秒|1；不补跑故障版测试或追加smoke|
|PT／teacher／新KRR fit或forward／新构象／额外seed／worker|0|0|全部0|

新结果根：`results/ua2d_fusion_20261017/g1_fp_gate_recovery1/`；日志根：`logs/ua2d_fusion_20261017/g1_fp_gate_recovery1/`；tmux：`Uni-Poly:ua2d_g1_recovery1_*`。当前这些recovery根不存在，不在本轮创建。

旧`g1_fp_gate/`结果／日志、failure、completion、source_used和消费账本保持只读。原初始化、绑定与586条FP按真实内容和既有policy／行身份正常绑定后复用到新根；不重算已有FP，不新增哈希验证。新FPWriter需支持在新根承接已存bits表，不能复制后仍执行无条件CREATE TABLE或把旧586次生成计数重置为0。新旧根的OUT／LOG／单元／freeze引用需显式一致，避免某个import仍向旧根写入。

失败若再次发生，本提案不自动追加第79次启动或第二次恢复；保存新现场、停止受影响运行并报告。

### 17.5 唯一最终测试及交付

全部返修与新70FT、全局冻结、正式预测及UA_FIXED75算术完成后，只执行一次：

```bash
CUDA_VISIBLE_DEVICES='' timeout 300s /opt/conda/bin/python -m pytest -q tests/test_ua2d_fp_gate.py
```

原9项测试尚未运行，最终同一suite中保留原GATE正确性问题，并补齐与本次故障对应的人工夹具：自身预留／真实PID交接、observed重复来源、未归属外部实验、超限先保存快照、缺失或中断输入记录、旧失败预算承接和recovery路径拒绝旧根写入。使用假PID／人工数据，不读真实GPU、不启动worker、无模型重放；不是另一轮预跑。不得因本次实现仅运行期失败而把原GATE9项写成已通过。

交付仍为七任务R²／原单位RMSE、论文对应模型／R²及差、逐fold结果与协议限制，完整原精度计算后3位小数、std ddof=0；本周期当前全部七项依旧未完成／不可比较。额外交付旧／新／累计预算、旧消费unknown标记与保守占用、异常前原始PID和每卡峰值／总峰值、source增量及全部真实退出。最终状态为待审查或具体失败；独立审查复用这些证据，不重跑实验／测试，审查通过后才归档PROJECT_HISTORY。

## 18. 本轮复核执行记录与后续边界

- 当前状态仍需返修；本轮只完成失败只读复核、预算核算和计划维护，没有修复代码或重启。
- 已核实五份科学源码与source_used内容一致、9个工作PID已退出、8个FT均无可恢复checkpoint、FP缓存586条。未核实本次异常触发时真实PID集合与根因，不把最后成功并发快照当作触发时峰值。
- 新增科学消费：PT0、FT0、FP生成0、fit0、模型推理0、正式预测0、GPU作业0、pytest0。未创建recovery目录、未改旧产物、没有删除或git提交操作。
- 唯一文档最终检查在本节完成后集中进行，核对本轮实际diff、格式、现存引用及保守预算算术。已有失败报告和原日志继续保留；本轮不把文档检查表述为科学测试通过。
- 后续执行需明确下达“按r3第17节修复并完整接续，接受累计78次启动、旧更新32次及旧图2,818次保守占用”等具体范围。当前用户交接状态不自动等于允许重试，本计划也不自行产生该授权。


## 19. r3 recovery1授权与执行

已核对git status、相关diff和旧失败记录，保护所有既有改动、不读TODO。本条用户确认第17节全部返修／接续预算，不需重复确认。仅修改四个获准G1辅助文件和本计划，模型文件及共享科学源码只读。最终测试仅在全部科学执行完成后一次运行。


## 20. G1 recovery1 实际交付记录

执行在finetune停止：RuntimeError('GPU concurrency exceeded')。日志/root/workspace/Uni-Poly-Plus-master/logs/ua2d_fusion_20261017/g1_fp_gate_recovery1/controller/，failure.json与消费账本在/root/workspace/Uni-Poly-Plus-master/results/ua2d_fusion_20261017/g1_fp_gate_recovery1/。没有自动重试或追加预算；未完成任务不可比较，不能报告性能或论文差。


## 21. recovery1再次停止：证据、累计消费与七任务交接

G1 r3 recovery1：需返修；第二次异常后已停止，不自动追加启动或恢复。

授权：r3第17节，新70个完整FT、累计启动上限78；旧更新32次／旧图2818次保守占用，旧实际未知保留，复用586条FP。
结果：2026-10-17 15:40（北京时间）FT阶段再次触发GPU concurrency exceeded。仅启动8个REAL，完成0个FT；NULL未派发，无完整epoch、best或正式结果。controller真实exit1；8个worker受控退出均为exit1（InterruptedError），state_after_stop.active为空；无遗留G1 writer。

本次返修仍未解决环境PID视图映射：NVML每卡4个479xxx PID与/proc中每卡4个2934xxx自有PID不相交，NSpid仅给出/proc自身PID，不能建立归属。分类将未归属NVML PID额外计入，失败快照为每卡8、总16；该数字不是实际并发证明。最后通过校验的槽位计数为每卡4、总8。未将算术偏移假设当作归属证据。
与旧事故相比，本次保存了原始NVML／proc、wrapper／worker归属、触发快照、清理前后状态、参数数量、首次trainability、增量输入和在途更新证据。完整优化器分组未形成，不能用参数设计数冒充完整运行分组。

|消费项|旧已知|旧预算占用|本次新执行|累计预算占用|
|---|---:|---:|---:|---:|
|FT启动|8|8|8|16|
|FT完成|0|0|0|0|
|attempted更新|24|32|26|58|
|completed更新|21|32|26|58|
|FT图材料化|实际未知|2818|768|3586|
|FP生成|586|586|0|586|
|正式预测／融合／参照数组|0|0|0|0|
|最终CPU测试|0|0|0|0|

本次768次输入attempted/completed均已记录、inflight为空；新更新hook合计26/26。两单元尚未形成updates.json，但interrupted_state.json明确在受控停止边界记录0/0，原账本仍保留缺失更新文件标记。旧图、旧精确更新及旧片段epoch持续标unknown。
新FP计算0；全部586条旧记录复制至新缓存并按实际访问复用。PT／teacher／新KRR fit/forward／新构象／额外seed／worker全部0。没有读取本周期TEST图或预测数组，没有A_GLOBAL搜索。

|任务|REAL／NULL神经及融合|UA_FIXED75|论文最佳模型|论文R²|候选−论文|
|---|---|---|---|---:|---|
|Eea|未完成／不可比较|未执行|Periodic-TDL|0.934|不可比较：无正式结果|
|Egb|未完成／不可比较|未执行|Periodic-TDL (+Chem, +RC)|0.923|不可比较：无正式结果|
|Egc|未完成／不可比较|未执行|Periodic-TDL (+Chem, +RC)|0.911|不可比较：无正式结果|
|Ei|未完成／不可比较|未执行|Periodic-TDL|0.839|不可比较：无正式结果|
|EPS|未完成／不可比较|未执行|Periodic-TDL|0.787|不可比较：无正式结果|
|Nc|未完成／不可比较|未执行|Periodic-TDL|0.856|不可比较：无正式结果|
|Xc|未完成／不可比较|未执行|Periodic-TDL|0.423|不可比较：无正式结果|

已有正式参考仅复用原汇总，未重放预测；五折seed42+fold、ddof=0，三位小数，RMSE不跨单位平均。

|任务|原UA R² mean±std|原UA RMSE mean±std|r10 R² mean±std|r10 RMSE mean±std|
|---|---:|---:|---:|---:|
|Eea|0.927±0.016|0.289±0.026|0.924±0.017|0.294±0.025|
|Egb|0.926±0.010|0.528±0.045|0.923±0.012|0.539±0.068|
|Egc|0.905±0.016|0.479±0.036|0.917±0.013|0.450±0.031|
|Ei|0.803±0.029|0.433±0.066|0.807±0.016|0.429±0.058|
|EPS|0.768±0.061|0.530±0.066|0.775±0.049|0.527±0.078|
|Nc|0.813±0.065|0.101±0.010|0.816±0.049|0.100±0.006|
|Xc|0.373±0.107|18.627±1.526|0.412±0.120|18.017±1.559|

原UA／r10 macro R²为0.788／0.796。所有G1候选差、逐fold差、退化任务、validation→TEST差及收益集中度均不可计算，不用旧结果补齐。
共享validation／TEST已参与开发，仅探索性数值比较；论文数据／split／评估统计等价性未核实。门值不作因果贡献，两臂也不能证明GATE优于固定残差。
修改仅四个授权G1辅助文件及CURRENT_PLAN；模型、共享源码、旧现场只读。新增人工夹具已写入但从未运行，不能称测试通过。
日志：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_fusion_20261017/g1_fp_gate_recovery1/controller/、ft_*/、failure_delivery/；tmux Uni-Poly:ua2d_g1_recovery1_*。冻结、正式预测、最终CPU日志均未形成；final_cpu_not_run.json明确NOT_RUN。
证据：failure_diagnostic.json、input_and_interrupt_index.json、parameter_groups.json、seven_task_results.json、prediction_index.json、consumption.json、state_before_stop.json、state_after_stop.json、concurrency_observations.jsonl及真实exit。
独立审查未进行，不归档PROJECT_HISTORY、不改SOTA_RESULT、不提交或推送。后续须先解决PID视图归属；本次授权不允许第三次尝试，未启动其他周期。


## 22. r4 当前自主执行授权与修复

用户当前指令覆盖原“第三次尝试需再确认”的限制：直接修复并执行，无需再次询问同一启动预算。模型、数据、split、P_ALIGN step5000、原head、gate初态、seed、uniform_u1与固定0.75预测政策均保持，目标仍是70个完整FT、70正式神经预测／140组神经与融合数组、最多35组UA_FIXED75参照、最终一次CPU suite。没有新增科学候选、PT目标或GPU probe。新预训练目标／预算尚无具体定义，当前先复用已完成预训练，不把复用写成新增PT。

本轮已确认当前NVML与容器/proc为不同PID视图，NSpid只有本地PID。单靠NSpid、PID数值相等或算术偏移无法归属。修复采用每个真实FT／正式预测单元所需的首次CUDA初始化握手：CPU模型恢复后请求放行，控制器保存NVML基线，仅放行一个自身模型的model.to(cuda)，worker同步该正常操作并发布ready，然后以基线到ready采样间唯一新出现的NVML PID绑定已知worker；绑定成功后才允许其训练／正式预测。其他自身单元等待，不同时创建CUDA上下文。此处只串行首次CUDA初始化，完成绑定后的实验仍各卡最多4并发。

若出现0个或多个新增NVML PID，身份不明确即保存原始前后证据并停止，不能猜偏移或忽略外部进程。不同视图中尚未映射的NVML PID使用独立标识计数，避免与本地PID数值碰撞。已有进程、预留槽位、真实退出、旧现场保护及增量输入记录继续保留。修复仅修改4个G1辅助文件／测试及本计划；模型与共享科学源码保持只读，不新增预跑或阶段性测试。

### 22.1 承接两次失败的执行预算

|项目|已消费／保守占用|本轮新执行上限|累计上限|
|---|---:|---:|---:|
|FT启动|16|70|86|
|完整FT|0|70|70|
|新科学矩阵epochs|旧片段unknown保留|4,900|4,900完整epochs＋旧片段|
|attempted／completed更新|落盘合计50／47；预算各占58|各112,000|各112,058|
|FT图材料化|初次unknown按2,818占用＋recovery1实记768＝3,586|47,000|50,586|
|FP生成|586|最多5,289|5,875|
|正式神经预测／GATE数组|0／0|70／140|70／140|
|UA_FIXED75算术／数组|0／0|最多35／35|最多35／35|
|TEST图材料化|0|11,750|11,750|
|最终CPU检查|0|1次、timeout300秒|1|
|新增PT／teacher／KRR fit或forward／新构象／额外seed／worker|0|0|0|

结果根：`results/ua2d_fusion_20261017/g1_fp_gate_recovery2/`；日志根：`logs/ua2d_fusion_20261017/g1_fp_gate_recovery2/`；tmux：`Uni-Poly:ua2d_g1_recovery2_*`。旧g1_fp_gate及recovery1结果／日志一律只读。旧586条FP从原已存缓存复制复用，初始化与源绑定沿用原G1；消费记录明确保留原始unknown，不以0替代。修复前文件暂存`/tmp/ua2d_g1_recovery2_before/`，执行时保存到新根source_before，源码及参数不得并行改写。

只执行正常训练与正式评估；全部科学执行完成后在tmux独立final_cpu窗口一次运行现有GATE suite（含本轮PID视图／唯一CUDA出生及预算人工夹具），不提前运行测试、benchmark或试训练。当前测试尚未运行。启动／完成／失败以真实日志更新，完成交付仍需如实区分执行自检和独立审查。


## 23. G1 recovery2 实际交付记录

执行在finetune停止：ValueError('not enough values to unpack (expected 2, got 1)')。日志/root/workspace/Uni-Poly-Plus-master/logs/ua2d_fusion_20261017/g1_fp_gate_recovery2/controller/，failure.json与消费账本在/root/workspace/Uni-Poly-Plus-master/results/ua2d_fusion_20261017/g1_fp_gate_recovery2/。没有自动重试或追加预算；未完成任务不可比较，不能报告性能或论文差。


## 24. r5 当前执行：P2新预训练20k与G2微调

授权来源：用户“请你直接解决这个问题，并执行对应的预训练和微调！！无需授权”；澄清另开预训练周期；最新明确“使用 step 20000，每 5k 保存一次 checkpoint”。不再复用step5000作为新实验parent，也不重复要求预算授权。旧recovery3的51单元接续草案尚未启动，现由本节新parent方案替代。

### 24.1 科学目标与比较范围

|项目|本次固定方案|
|---|---|
|scientific question|完整ALIGN20k预训练后，UA_2D上的同容量GATE结构中，真实Morgan FP能否改善常量FP对照及固定r10预测融合的正式观测值？|
|reference|已核实旧UA_2D macro R² 0.788、r10 macro R² 0.796；同20k parent的GATE_NULL作FP信息的匹配对照；UA_FIXED75仅数组算术参照|
|controlled change|GATE_REAL与GATE_NULL仅FP取真实bits或常量0.5；新PT保持原U1原子掩码CE、几何重建、2D/3D ALIGN目标及权重，不加入FP预训练目标|
|comparison budget|两GATE臂共用P2 step20000、原head初态、split、seed42+fold、train-only scaler、uniform_u1、70epoch与GATE初态；全35单元固定神经0.75＋已有KRR0.25|

新PT从原`U1/initialization/P_ALIGN_step0.pt`的模型张量重新开始，优化器重新创建，不从已训练step5000续训；原step0仅重绑定新配置到新目录，旧文件只读。FP GATE仍在下游微调中加入；PT未新增FP teacher、核蒸馏或新loss。预训练原方未包含FGR（third_task=none），不将ALIGN与FGR混称。

只使用物理GPU 2、3、DDP world=2、每rank microbatch=84、accumulation=6，保持global batch=1008；seed42、BF16、AdamW lr0.0002、weight_decay0、clip1、warmup2000、schedule_total_steps20000、end_lr1e-9；mask0.3、noise_sigma0.03、geometry/atom权重1/1、ALIGN weight0.1、warmup1000、temperature0.1。与旧world4/accum3相比，每次前向ALIGN候选池从336变168，分rank RNG轨迹也不同；因此新旧差异混杂PT长度、DDP划分和GATE，不能称为训练长度的单因素消融。

固定保存step5000、10000、15000、20000的完整resume与deploy两类checkpoint，共4组8个文件；不覆盖旧checkpoint，不做4个checkpoint的下游搜索。G2只使用deploy_20000.pt，checkpoint不依据下游TEST选择。保留原训练内过程validation（1000/2500/5000各1024，共3072次呈现）；这是原训练流程，不另加smoke、benchmark或重复评估。

### 24.2 新预算与旧消费分列

|项目|P2＋G2本轮预算|历史消费及合计说明|
|---|---:|---|
|新PT周期／optimizer updates|1／20,000|历史G1 PT为0；不计入既有U1的旧PT消费|
|PT训练呈现|20,160,000|1008×20000；每rank 10,080,000，16个CPU预取worker（每rank8），不作为GPU实验路数|
|PT过程validation|3次／3,072呈现|固定原1024集合；只观测、不选checkpoint|
|G2完整FT|70（两臂×7任务×5fold）|旧G1累计启动43、完成19；新parent需要全新70FT，累计启动上限113；旧19只读保留，不拼入新parent矩阵|
|G2 epochs／optimizer attempted和completed各上限|4,900／112,000|旧更新保守30,848，合计每项142,848；旧实际已知attempted30,840／completed30,837，初次unknown继续保留|
|G2 FT图材料化|47,000|旧保守24,135，累计71,135；旧图actual unknown不改写为精确值|
|G2正式神经推理／新神经与融合数组|70／140组|旧G1未正式推理；test图11,750；旧35组KRR只读复用，禁止新fit或forward|
|UA_FIXED75参照|至多35组算术数组|已有兼容A_FIXED则复用；不扩展A_GLOBAL|
|FP|复用旧缓存3,414条，新增上限2,461|保持字面normalized_smiles键及FP规范；累计生成上限5,875；不为PT生成FP|
|最终CPU suite|1次，timeout300秒、空CUDA|旧G1至今0次；全部科学执行结束后才运行GATE/PID/消费/CSV/PT配置人工夹具；不得提前调用pytest或重复验证|
|新seed／新KRR fit或forward／新构象／teacher|0|保持原科学边界|

GPU约束仍为2、3每卡最多4路、总8路，包含外部作业与预留。PT阶段不并行G2 FT；PT用NVML实际context PID计数，不将容器/proc PID再次相加；PT两个rank的RNG保存可能在两卡均创建context，最多2个本任务context PID/卡。PT真实退出且两卡context清空后再启动G2；G2采用序列化第一次必需`model.to(cuda)`出生握手绑定NVML和/proc，训练本身可4/卡并行。

### 24.3 调度修复与当前设备问题

已实现CUDA出生握手；recovery2已完成19FT，未再次触发原PID重复计数。recovery2最终在GPU映射CSV拆行处抛`ValueError`，启动27、完成19、更新30,790／30,790、图20,549、FP新增2,828；与更早失败累计数见上表。旧非CSV原始行没有保存，不能事后声称已核实当时硬件原因。

本次正常设备查询实际发现`Unable to determine the device handle for GPU2: 0000:98:00.0: Unknown Error`，GPU2从有效index/uuid返回缺失，GPU3可识别。`/proc/driver/nvidia/gpus/0000:98:00.0/information`仍记载UUID `GPU-31456dd5-a421-142b-644b-7a80f482dfb9`。容器`/sys`挂载只读，CapEff为00000000a80425fb、不含CAP_SYS_ADMIN；dmesg读取被拒绝，内核Xid和物理根因未核实。该硬件/驱动句柄故障与PID命名空间是不同问题。

新的CSV读取支持空行、引号和两字段，GPU映射在正常首次查询后缓存。错误行、缺失GPU2/3、非零exit或stderr均带原stdout/stderr停止；禁止通过丢弃故障行、把缺卡当空闲或只跑GPU3来冒充两卡执行。定向设备恢复仅尝试GPU2 PCI地址的`nvidia-smi --gpu-reset -i 0000:98:00.0`一次，不重启驱动或宿主机、不操作GPU0/1、不改只读sysfs；日志`logs/ua2d_fusion_20261017/gpu2_device_repair/`，tmux `Uni-Poly:ua2d_gpu2_device_repair`。

允许修改：GATE运行辅助、controller/unit、最终CPU测试、P2独立配置/接续入口，以及共享PT入口中仅由ua2d_pt_revision触发的配置校验、deployment绑定和两rank过程validation适配。共享入口中原有T3/T3b及其他改动保留。禁止修改旧模型张量、旧训练/预测产物、数据/split、核拟合政策；不运行无关预检、hash、额外候选，不删除文件，不commit/push。

### 24.4 运行入口、产物与停止规则

- P2配置/训练：`results/ua2d_fusion_20261017/p2_align20k/initialization/P_ALIGN.json`、`pretrain/P_ALIGN/`；日志`logs/ua2d_fusion_20261017/p2_align20k/`；tmux `Uni-Poly:ua2d_p2_align20k`。入口`/opt/conda/bin/python scripts/run_ua2d_pt20k.py --launch`；命令、cwd、环境、真实exit、stdout/stderr落盘。
- 成功后自动启动G2入口`/opt/conda/bin/python scripts/run_ua2d_fp_gate.py --launch`，新根`results/ua2d_fusion_20261017/g2_fp_gate_pt20k/`、同名logs根、tmux `Uni-Poly:ua2d_g2_pt20k_*`。未完成P2、缺四组checkpoint或parent metadata不匹配时不得启动FT。
- 旧G1三个根只读保留，源码修复前快照`/tmp/ua2d_g1_recovery3_before/`；共享PT原改动由本次补丁反向还原到`/tmp/ua2d_pt20k_before/`供实际diff（不将其称原运行快照）。这些快照是审查依据，不是hash合同。
- 运行中的NaN/Inf、真实超限、CUDA身份归属歧义、device错误、artifact/checkpoint/schema不匹配、writer冲突或覆盖风险：停止受影响运行，保留原始证据和inflight，不静默丢弃，不自动扩大预算或改科学方案。GPU2无法恢复时记“阻断”，新PT/FT启动应为0；不为已知设备错误消费正式启动。
- 最终一次CPU suite仅在P2＋G2科学执行完成后运行；若训练受设备阻断则如实NOT_RUN。正式完成后再核对实际diff、日志和七任务正式预测，独立审查尚未完成前不将新G2结果归档PROJECT_HISTORY。

### 24.5 当前七任务审查结论

本轮是实现与运行准备，尚无新parent正式预测；以下逐项完整列示，不能以旧19个FT或训练loss补齐。

|任务|P2/G2本项目R²|论文参考R²|论文模型|本项目−论文|状态与原因|
|---|---|---:|---|---|---|
|Eea|未完成／不可比较|0.934|Periodic-TDL|不可计算|新PT与新parent正式预测未完成|
|Egb|未完成／不可比较|0.923|Periodic-TDL (+Chem, +RC)|不可计算|同上|
|Egc|未完成／不可比较|0.911|Periodic-TDL (+Chem, +RC)|不可计算|同上|
|Ei|未完成／不可比较|0.839|Periodic-TDL|不可计算|同上|
|EPS|未完成／不可比较|0.787|Periodic-TDL|不可计算|同上|
|Nc|未完成／不可比较|0.856|Periodic-TDL|不可计算|同上|
|Xc|未完成／不可比较|0.423|Periodic-TDL|不可计算|同上|

计划口径仍为每任务5fold、seed42+fold、std ddof=0、任务等权macro；原精度计算后保留3位小数。项目数据/split与论文等价性未核实，Egc/Xc历史样本量有差异；共享validation/TEST已参与开发，最终也只能数值对照。没有新科学结果可宣称提升，没有独立审查结论。


### P2 实际运行记录

P2未能完成：RuntimeError("Malformed nvidia-smi CSV; raw='Unable to determine the device handle for GPU2: 0000:98:00.0: Unknown Error\\n0, GPU-d57d4b9e-0c37-7854-f08b-7bf43759717c\\n1, GPU-c5a42592-e8f2-78f6-c1f9-e3da2a0ea4ec\\n3, GPU-665c4372-2ad2-31f1-7b34-8ec763ef6b38\\n'")。PT启动0次；未启动G2、未运行最终CPU测试。证据：/root/workspace/Uni-Poly-Plus-master/results/ua2d_fusion_20261017/p2_align20k/failure.json，日志/root/workspace/Uni-Poly-Plus-master/logs/ua2d_fusion_20261017/p2_align20k/。


### 24.6 r5 本轮实际收口：设备阻断，未启动新训练

2026-10-17 08:19:41—08:19:42 UTC：GPU2定向reset返回exit=6、`No devices were found`；前后正常查询均保留GPU2 Unknown Error。修复window `Uni-Poly:ua2d_gpu2_device_repair`；真实命令、原stdout/stderr与退出见该logs根的execution.json、stdout.log、exit.txt（汇总exit1）。没有操作GPU0/1，未执行PCI unbind、驱动卸载或宿主机重启。宿主机内核/硬件根因仍未核实。

2026-10-17 08:20:27 UTC：在`Uni-Poly:ua2d_p2_align20k`执行真实P2接续入口，已落盘新`initialization/P_ALIGN.json`与从原step0重绑定的164,784,313字节初态；没有创建预训练优化器或训练worker。controller在设备查询中识别并保留完整非CSV错误，按设备依赖停止、真实exit=1。`torchrun_pid=null`，PT starts=0、attempted=0、completed=0，G2 FT starts=0；FP新增0、图材料化0、新正式预测0、最终CPU suite 0次（NOT_RUN）。这不是20k预训练完成，也不是模型测试通过。

实际证据：`results/ua2d_fusion_20261017/p2_align20k/failure.json`、`logs/ua2d_fusion_20261017/p2_align20k/execution.json`、`command.json`、`stderr.log`、`exit.txt`。run时源码在P2/source_used；后续仅补入“零启动准备态接续”及最终静态交付记录，最终源码另存final_review/source_final，不覆盖原run源码与failure。

新增接续选项`--continue-prepared`：仅当原状态为阻断、PT starts=0、尚无pretrain目录或launch intent，且准备好的step0/config仍吻合时允许；新日志独立放在`p2_align20k/continuationN/`，新设备/失败证据放在`p2_align20k/controller_attempts/continuationN/`，旧failure和logs只读保留。真正spawn前排他写入pretrain_launch_intent.json，防止第二controller花费新PT预算。当前未运行该选项、没有任何自动重试。GPU2从宿主机恢复后，接续命令为：

```bash
/opt/conda/bin/python scripts/run_ua2d_pt20k.py --launch --continue-prepared
```

此命令会在Uni-Poly独立window运行；成功训练20k并保存四组checkpoint后自动启动完整G2 70FT、正式预测、固定融合和唯一最终CPU suite。当前容器缺少设备管理能力且/sys只读，无法完成GPU2恢复；需先由宿主机侧恢复PCI 0000:98:00.0的设备可用性。该条件是实际运行阻断，不是待用户批准预算或科学方案。

交付层级：GATE/PID/CSV与20k新parent接续实现已写入，设备错误停止路径已实际执行；新PT/G2训练与CPU suite均未执行，不能声称训练路径已验证。七任务仍全部未完成／不可比较，数值与协议口径见24.5。没有独立审查，不新增PROJECT_HISTORY归档；该文件本轮开始前已有的改动保留。最终只复核实际diff与格式，并保存已有进程/退出证据，不重放模型或测试。无文件删除或git提交操作。


## 26. r7：GPU2/3重新执行，PT最多2路、FT最多4路总并发

本节覆盖第24节r5的当前设备阻断、输出根及旧8路并发安排；旧现场保持只读。授权：用户最新“请你只使用 GPU 2、3，预训练最多 2 路实验并发，微调最多 4 路实验并发，重新执行，无需授权；启动后没有问题停止监控”。此前20,000 step、每5,000 step保存以及七任务GATE融合目标保持。

### 26.1 科学方案与新目录

|项目|预先固定条件|
|---|---|
|scientific question|新ALIGN20k预训练后，同容量GATE下真实FP信息是否优于常量FP，并改善神经预测与固定r10融合的正式观测值？|
|reference|旧UA_2D macro R² 0.788、r10 macro R² 0.796；同新parent的GATE_NULL为FP信息匹配对照；UA_FIXED75仅旧数组算术参照|
|controlled change|REAL/NULL仅Morgan bits取真实值或常量0.5；相同GATE、原head初态、训练配方、split、seed、scaler、20k parent；统一神经0.75＋原KRR0.25|
|comparison budget|PT从原U1 P_ALIGN step0重新开始，仅原CE/geometry/ALIGN目标；20,000更新，world2、micro84、accum6、global batch1008，保留原warmup/schedule/loss/seed；下游两臂×7任务×5fold=70FT，每个head10＋joint60epoch|

P2结果根：`results/ua2d_fusion_20261017/p2_align20k_r7/`；日志同名logs根；tmux `Uni-Poly:ua2d_p2_align20k_r7`；入口`/opt/conda/bin/python scripts/run_ua2d_pt20k.py --launch`。保存5k/10k/15k/20k的resume＋deploy四组，固定20k用于下游，不做checkpoint下游搜索。原训练内过程validation仍为1000/2500/5000各1024，共3072；无额外smoke、probe或benchmark。

G2新结果根：`results/ua2d_fusion_20261017/g2_fp_gate_pt20k_r7/`；日志同名logs根；tmux `Uni-Poly:ua2d_g2_r7_*`。P2成功并保存四组checkpoint、两卡原context退出后自动启动完整70FT；全70个validation best冻结后才做70正式预测与140组神经/融合数组，再计算至多35组UA_FIXED75参照，最后唯一CPU suite。GATE架构、FP规范、全局权重0.75、原核α等沿用第24节，无新KRR fit/forward或额外seed。

PT两rank导致ALIGN每次前向候选池为168，旧四rank为336，RNG划分也不同；新旧结果仍混杂训练长度、DDP划分与GATE，不能归因于PT长度一项。前一条GPU3-only请求形成的单rank/accum12草案没有生成实验目录、没有训练消费，最新两卡要求已覆盖该草案。

### 26.2 并发与预算

仅物理GPU2、3；不使用GPU0/1。预训练预算仍只有一个新科学周期（20,000更新、20,160,000训练呈现、8个CPU预取worker/rank）；DDP两rank各用一张卡，一次同时运行2个训练rank，未额外新增第二PT候选。PT阶段不并行FT。PT运行期按两卡NVML PID集合的去重并集限制至多2个GPU训练进程；保存RNG时同一rank可能在两卡均有context，不能将同一NVML PID重复当两路。

微调与正式神经预测**两卡合计最多4路**，含可观测外部实验和预留；Slots总上限4，派发只从2/3选择，正常空闲情况下均衡为2路/卡。保留CUDA出生握手、真实PID/退出绑定与失败消费记录。最终CPU suite为CPU，不占GPU实验槽位；只在完整实验结束后运行一次。

本轮预算仍为新70FT、4900epoch、attempted/completed各112,000、FT图47,000、正式test图11,750、FP复用3414/新增至多2461、数组至多175、最终CPU suite1次（空CUDA/timeout300秒）。旧G1保守消费43启动、19完成、更新各30,848、图24,135、FP3414分列保留，累计FT启动上限113、更新各142,848、图71,135、FP5875。旧实际unknown不改写为0。r5 P2启动0，故重新启动本P2未增加第二个科学周期消费。

### 26.3 实现、运行及停止监控

允许修改仅为P2/G2的路径、配置/parent绑定、GPU观察范围、并发上限、真实运行记录、最终CPU夹具和当前计划。原已有源码改动、旧产物/data/split只读保留，r5快照保存在`/tmp/ua2d_r7_before/`并在实际执行根复制source_before/source_used。没有文件删除、hash校验或git提交。

最新读取：GPU2 UUID `GPU-31456dd5-a421-142b-644b-7a80f482dfb9`，GPU3 UUID `GPU-665c4372-2ad2-31f1-7b34-8ec763ef6b38`，两卡当前均能正常查询。不能由此推断上次内核/硬件根因已解决；本次不再执行设备reset。NVML查询显式限定`-i 2,3`；设备/CSV错误仍带原始输出停止，不以跳过错误行继续。

Codex只观察正式训练起始记录，确认优化器更新正常推进且日志无已出现的运行异常后，按用户要求停止主动监控并结束本轮对话。训练继续在tmux中运行，P2→G2→正式评估→唯一最终CPU suite由队列控制器自动接续；控制器必要的派发、限流与错误处理属于实际执行流程，不是额外实验或模型验证。启动正常不等于20k/FT/最终测试完成；状态保持“执行中”。不设置额外smoke、阶段测试或预跑。完整正式结果可用后由已有报告入口更新CURRENT_PLAN；独立审查未完成前不归档新G2到PROJECT_HISTORY。

NaN/Inf、真实并发超限、CUDA身份歧义、设备故障、checkpoint/schema不匹配、writer冲突或覆盖风险时停止受影响运行、保留inflight和证据，不静默重试或扩展科学预算。旧样本/epoch/fold/seed不扩大。

### 26.4 七任务当前审查口径

尚无本轮新parent正式预测，均为未完成／不可比较：Eea（论文0.934，Periodic-TDL）、Egb（0.923，Periodic-TDL +Chem,+RC）、Egc（0.911，同前）、Ei（0.839，Periodic-TDL）、EPS（0.787，Periodic-TDL）、Nc（0.856，Periodic-TDL）、Xc（0.423，Periodic-TDL）。本项目R²及项目−论文差值全部不可计算，原因是新PT/FT与正式预测尚未完成，不能用旧19FT或训练loss补齐。

最终每任务五折、seed42+fold、std ddof=0，先用原精度计算再显示三位小数；任务等权macro。共享validation/TEST已参与开发，论文数据/split/协议等价性未核实，Egc/Xc历史样本量有差异，最终仅做数值对照。当前未运行CPU suite、未进行独立审查。


### P2 实际运行记录

新P2已启动：/root/workspace/Uni-Poly-Plus-master/logs/ua2d_fusion_20261017/p2_align20k_r7/command.json；预训练上限20,000步，checkpoint每5,000步；完成后自动启动G2完整70FT。


### 26.5 r7启动实际记录与停止主动监控

2026-10-17T19:50:46.467089+08:00（北京时间）完成本轮启动交接。首次launcher在tmux has-session前发现默认server不存在，未产生PT/FT；该证据已保存`logs/ua2d_fusion_20261017/p2_align20k_r7_launch_pre_tmux.json`，随后重建`Uni-Poly`会话，在独立window正式启动，未绕过tmux要求。

正式控制器`Uni-Poly:ua2d_p2_align20k_r7`，controller PID22908、torchrun PID22949，CUDA_VISIBLE_DEVICES=2,3，PT starts=1。停止主动监控时已观察两rank正常完成step14，消费快照attempted15/completed14表示第15次正常在途更新；loss、ALIGN及梯度日志均为有限值。最后NVML已有快照显示两卡各1个训练context，无新增运行错误；stderr仅记录NCCL旧环境变量弃用与DDP find_unused_parameters性能提示。它们不是科学测试结论。

状态继续为“执行中”，20k PT、四组checkpoint和70FT尚未完成；当时FT starts=0、正式预测0、最终CPU suite NOT_RUN。当前无性能改善或论文差值可报告，七任务仍按26.4未完成／不可比较。依用户最新要求，Codex此处停止主动轮询，不等待20k或FT完成；队列控制器继续正常派发、限流、错误处理与P2成功后的自动接续，不增加实验。没有提前运行CPU测试或额外smoke。新结果未独立审查，不归档PROJECT_HISTORY。

启动证据：`results/ua2d_fusion_20261017/p2_align20k_r7/startup_status.json`；真实训练日志`logs/ua2d_fusion_20261017/p2_align20k_r7/stdout.log`、`stderr.log`；参数/cwd/环境在`command.json`，真实运行状态在`execution.json`（运行中exit为null，不伪造完成）。实际rank记录在`pretrain/P_ALIGN/records_rank0.jsonl`及rank1同名文件，消费在`pretrain/P_ALIGN/consumption.json`，并发观察在P2根的concurrency.json与runtime_observations.jsonl。

所有科学源码在启动前已固定并存source_used/source_before；本节是启动后的文档交接。微调后续由`Uni-Poly:ua2d_g2_r7_*`自动执行，限流两卡总计4路，报告与正式数组将写入`results/ua2d_fusion_20261017/g2_fp_gate_pt20k_r7/`；一次CPU suite仍由该控制器在完整科学阶段结束后运行。startup不是完整实验验收，尚未执行最终测试或独立审查。


## 28. 2026-10-17：ChatGPT网页版科研交接与远程同步

用户当前授权：总结所有已执行实验与当前完整流程/架构，用于ChatGPT网页版延续科研；ChatGPT每次规划前读取最新远程仓库及最近更新并审查，规划后给具体计划和Codex执行提示词；Codex负责执行；每次代码修改完成必须push到emt11/Uni-Poly-Plus，允许自行确定上传范围并修改.gitignore。本节是工作流/资料交接，不新增科学候选或预算，r7仍继续执行。

### 实际远程与本地审查

首次核对远程默认分支与当前dev HEAD均为059d084aa6bf89ded67f9123e8c7991da0027163（Tuning point 1，2026-10-15 08:43:03北京时间），最近提交和实际diff已读取；fetch仅更新远程引用，不pull或覆盖工作树。此前本地FCE/T1至r16、FP-GRAPH、UA GATE/P2等源码与测试尚未同步；本轮按用户允许上传源码、必要文档和精简结果，不上传大型训练产物。原修改全部保留，不读取/修改TODO，不删除文件，不force-push。

新材料：CHATGPT_HANDOFF.md、CHATGPT_START_PROMPT.md、docs/research/MODEL_ARCHITECTURE.md、EXPERIMENT_CATALOG.md、EXPERIMENT_RESULTS.md、REMOTE_WORKFLOW.md，以及evidence内原精度统计、消费/当前状态快照、现存目录索引。43份源汇总/126条政策含重复参照和旧五/八任务方案，不等于126个新独立实验；31个现存results根全部列出，已删除或无法核实的旧单元不伪造。快照时间见各JSON，本轮只读摘录，没有重新模型评估。

AGENTS已将默认分工改为ChatGPT规划/独立审查、Codex执行，并加入每次规划先审查远程最新版本、代码修改批次完成后commit/push的长期要求；失败如实报告，不将推送当测试通过。更新.gitignore排除本地凭据、weights/数组/数据库等大文件；docs中的JSON/CSV/Markdown精简证据可提交。既有被跟踪的TODO历史不会因ignore自动删除，本轮不读取或提交其变化。

### 一项静态必要修正

梳理20k新parent的真实调用链发现：G2 build_model虽然要求package step20000，调用restore_pure_2d却仍使用默认expected_step5000。现仅显式传入expected_step=20000；不改变模型、输入、优化器、数据/split、步数或其他科学预算。当前P2训练中只使用该模块的环境/观察辅助，未调用此G2构造；未来G2新子进程加载修正并保存实际源码。没有重启PT、追加FT或提前运行最终CPU suite，静态修正不等于动态验证通过。

### 证据更新与范围限制

现存旧r10/A1消费实际为2个PT启动、5093 attempted/5092 completed、FT0，signal15停止，不再沿用“仅初始化PASS”的旧摘要。E0保存8passed/2failed，E1既有失败与后存36对GPU比较/72次forward-backward但0更新的PASS文件并存，早期consumption仍是GPU0，未核实全周期关闭。FCE路径诊断p1有19份FT指标、formal_test明确FORBIDDEN，不能当完整正式七任务。上述仅保留实际发现及冲突，没有重放或补签独立审查。

当前P2进度及r7消费快照见docs/research/evidence/execution_status.json；G2正式结果仍未完成，七任务仍按第26.4节不可比较。没有新模型/闭合实验结果归档到PROJECT_HISTORY，已有231行历史迁移改动随本轮一并同步。SOTA_RESULT内容保持原参考。

交付最终检查只覆盖本次文档/发布差异、链接/JSON/精简表格与待上传源码的静态语法，不启动模型或pytest；它不是r7唯一最终CPU suite。训练继续，Codex不恢复持续监控。源码、已有实验测试与记录按本次批准同步为“运行中/待审查”；整个科学周期的最终测试和独立审查仍待实际执行。
