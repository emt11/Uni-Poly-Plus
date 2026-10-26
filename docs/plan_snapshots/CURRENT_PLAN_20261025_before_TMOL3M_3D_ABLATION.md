# 当前计划：UA 2D＋UA 3D＋FP 的 Uni-Poly／TMolNet 三模态方案

> 最新索引（2026-10-25T15:17:24.452785+08:00）：父UA3M-PAPERFUSE/r1保持**需返修**；返修UA3M-PAPERFUSE-REPAIR-20261025-01/gpu0123-r1已获用户明确授权，状态**待审查**，阶段交付报告。独立返修状态块见文末；原失败与旧FINAL保留，独立复审PENDING。

> 当前资源修订：用户已明确允许物理GPU 0、1、2、3全部使用，资源修订gpu0123-r1见第23节。此条覆盖本文件及AGENTS中本周期旧的“仅GPU3”资源限制；当前作业实际设备仍以运行记录为准，四卡调度尚未落实，不能由计划改动推定已切换。

## 1. 检查依据、授权和旧记录

- 制定快照：2026-10-25 10:12:26起（Asia/Shanghai）；本地dev/cd36ae7280b159f881b483ad5814835ff7b74c4a（P10K）。已查看AGENTS、原CURRENT_PLAN、相关diff、SOTA_RESULT、当前UA编码器／mask任务／缓存／训练实现、已审查正式结果及论文／作者源码。
- 覆盖前完整文本已保存至 [旧交接快照](docs/plan_snapshots/CURRENT_PLAN_20261025_before_UA3M_PAPERFUSE.md)。该文件只保存历史交接；旧执行提示词不得当作新周期授权，旧预算、事故、checkpoint和正式结果继续保留在原目录。PROJECT_HISTORY本轮不修改。
- 用户原要求为类似Uni-Poly和TMolNet的架构，仅保留“当前UA路线2D、当前UA路线3D、FP”；2D／3D模态不动，ALIGN参照相应论文，删除FP预测；“完整实现以上方案到CURRENT_PLAN.md（覆盖它）”。本次交付是完整实现规格、实验合同及DSH提示词；新增代码、PT、FT、模型推理、GPU和CPU suite消费均0。
- 用户对缺3D的异步答复原文：“保留全部样本，缺3D时屏蔽该模态并重归一化融合权重（推荐）”。这是本计划的数据语义依据，不单独产生实验启动授权。
- 当前已有CURRENT_PLAN、用户TODO改动，以及11个未跟踪源码／测试文件（旧无FP六个、A020融合五个）。全部保护，不读取TODO，不同步远端，不新增哈希、校验和或文件指纹合同。新实现仅使用独立ua3m_paperfuse文件及新结果根。

## 2. 科学问题、参考和范围

科学问题：保留UA两个原生编码器及其自监督任务，在预训练中将真实FP作为第三模态参与对齐，再在下游同时融合2D、3D、FP，能否改善七任务正式TEST？分别评估Uni-Poly式注意力融合与TMolNet式任务条件门控两个完整方案包。

| 项目 | 固定范围 |
|---|---|
| 候选 | UNI3M、TMOL3M，共2个；不追加ZERO／消融／学习率或超参网格 |
| 三个模态 | UA 2D、UA 3D、Morgan FP；没有SMILES语言编码器、caption或第四模态 |
| 2D／3D保留项 | 编码器内部结构、512维、6层、原UA注意力尺度、几何语义、mask任务及目标类别 |
| 变化项 | 外接256维模态适配／对齐模块、FP输入路径、三模态融合和256维性质头 |
| 禁止项 | FP位预测头／BCE／FP reconstruction target、KRR及其预测／蒸馏／核近似、后处理输出融合 |
| 下游输入 | 实际使用2D＋3D＋FP；缺3D按第5节屏蔽。部署不再声称只需2D输入 |
| 参考 | 已核实D_A020，ALIGN名义0.20、无FP预测预训练，正式macro=0.785044436513689 |

基线证据：results/ua2d_p10k_nofp_align_20261024/r1/final_review/results.json中的formal.D_A020；其35组FT BEST／scaler／validation／正式TEST只读引用。基线每任务均值见第14节。原D_A020父PT目录为results/ua2d_p10k_nofp_align_20261024/r1/pretrain/D_A020/，仅作参考，不作为新10K预训练的已训练起点。

两个新候选从共同step0重新PT各10K。不能在已有D_A020或FP-on P10K上追加训练再宣称同预算。该比较同时改变FP参与方式、ALIGN、融合、下游3D及部分优化设置，是完整方案比较，不能单独归因FP、3D、注意力或门控。UNI3M与TMOL3M也含不同ALIGN／正则等因素，不是纯融合层消融。

上一周期A020三模态以外的结果保留：NEURAL_LATE正式macro=0.793533639565631，ΔD_A020=0.008489203051942007，未过门槛；它只作历史数值参考，不进入本轮FINAL提名，不重新推理。旧FINAL=D_A005继续保留，原故障及177次启动不被新周期抹去。

## 3. 论文与作者实现：已核实内容和适配边界

| 来源 | 已核实依据 | 本项目采用 |
|---|---|---|
| [Uni-Poly正式论文](https://www.nature.com/articles/s41524-025-01652-z)及[作者预印本](https://assets-eu.researchsquare.com/files/rs-5795287/v1_covered_6aa1b06d-b6a3-4e51-85c7-f766401caeec.pdf)第20—23页 | 公共维数256、模态投影、多头注意力＋FFN、注意力池化、跨模态InfoNCE、温度0.07 | 原生UA两路替换原编码器，只保留三个模态，直接从融合向量回归 |
| [TMolNet正式论文](https://link.springer.com/article/10.1007/s11030-025-11350-z)及[作者模型源码](https://github.com/hchchc465/TMolNet/blob/master/TMolNet/model/TMolNet.py) | 模态256维，门控隐藏512、任务嵌入8；模态循环对比、熵正则 | 三模态门控输入776→512→3，保留作者损失数学 |
| [TMolNet配置](https://github.com/hchchc465/TMolNet/blob/master/TMolNet/parser_args.py)及[FP编码器](https://github.com/hchchc465/TMolNet/blob/master/TMolNet/model/fp_encode.py) | latent_dim=256；alpha=beta=0.08、T=0.1；FP隐藏512／256、输出256、dropout0.2 | 统一保留项目Morgan2048输入，替换原1248维混合指纹；其余网络尺寸按源码 |

核对方式：GitHub连接器不可用，改为只读访问作者公开仓库／raw源码和论文，没有clone、pull或push。Uni-Poly论文所列仓库当前返回404，未取得可核对源码；论文未明确的头数、FFN宽度、投影激活及头dropout等，在下节明示为项目固定配置，不能称为论文原值。TMolNet公开入口存在模块路径／参数缺项，本项目只移植可核对的模块和损失，不照搬其不完整训练入口。

本轮是两篇方法的UA三模态适配，不是原论文完整复现。Uni-Poly的1024位FP与TMolNet原混合FP均统一替换为现有Morgan2048，避免另加指纹类型因素。

## 4. 保持原样的UA组件和初始化

### 4.1 原生编码器及mask任务

- 2D：src/modules/glt_dual.py的BondPathO8，2-hop、6层、512维，原attention_scale_denominator=512；图表示为原mean_pool＋norm2。
- 3D：src/modules/ua2d_galformer_pretrain.py的MaskedGalformer3D，原有限trimer／finite_incident、距离与角度路径、angle value及incident_pool，6层、512维，同样保留原UA注意力尺度。PT传原mask labels；FT／推理不施加mask或几何噪声。
- 2D mask头沿用512→256→GELU→101；3D mask头沿用512→256→GELU→25755。40% corruption、noise_sigma=0.03、物理token分组及bond_length_weight、每图聚合和有效计数均沿用原UA。
- mask头只消费各自原生token表示，不读取FP或融合结果。FP不能绕过mask任务直接给decoder提供答案。
- h2=norm2(mean_pool(O8 atoms))、h3=norm3(incident_pool(3D bonds))，均为512维。后续256维适配全部位于编码器外部，不把原UA内部层改成256维。
- 不增加torsion、PH、MD200、几何生成、噪声重建或新的2D／3D任务。PT编码器dropout0.1；FT编码器dropout0为本项目沿用UA二维微调节奏的选择，不冒称已有相同3D下游实验。

### 4.2 共同step0和新组件

共同step0：results/ua2d_pretrain_fp_20261019/fast_r4/initialization/common_step0.pt；
mask头初态：同目录task_heads/V1.pt。

按旧load_nofp_initialization的显式来源合并逻辑，精确载入o8、norm2、encoder3d、norm3、atom_head、line_head；列明并排除旧proj2／proj3，因为新的ALIGN投影定义不同。原初始化不含FP训练监督；禁止strict=False吞不匹配、加载FP-on已训练模型或已有D_A020权重替代step0。

新投影、FP编码器、TMOL对比头按隔离初始化seed20261025生成，恢复外部RNG。两候选同名原生UA参数必须来自同一初态；结构不同的新组件不宣称逐参数匹配。各task/fold新增256维回归头初态一次生成并在两候选间共享同形Linear权重，记录独立初始化seed=20261025＋100×task_index＋fold；训练seed仍只有42+fold，不构成追加训练seed。

原512维下游头不能直接加载为256维新头。PT不构造未接入任何loss的融合模块／性质头；它们在FT时初始化。PT部署包必须保留两个编码器、norm、所有有效模态投影／FP编码器／对比头；FT显式剥离mask头，按模块清单严格加载。

## 5. 数据、FP和缺3D处理

### 5.1 PT输入

- 标签与模板源：data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging。
- 只读模板：results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache/，原schema ua2d_prepared_templates_r4。
- 只读逻辑映射：results/ua2d_ptscale_20261022/r1/inputs/，eligible_logical.npy／keys.npy／source_rows.npy；有效910756条，原911391条中排除635条，eligible数值为逻辑slot，不是原始source row。
- 继续在采样前排除原已知几何无效记录，不能让它们参与任何PT目标，也不能在PT静默退成2D＋FP。
- 从template.fp_bits按little bitorder解包2048位，作为data.morgan_fp输入。旧metadata中的loss=BCE、head、weight字段只描述历史缓存用途，本轮不继承；新schema明确fp_as_input=true、fp_prediction=false。
- 新labels不得含fp_target，模型不得含FP预测头或BCE分支。FP编码器在PT通过跨模态ALIGN获得梯度，2D／3D mask loss不经FP路径。
- Morgan binary固定radius2、2048-bit、chirality和bond types开启，保留dummy连接原子的clean normalized_smiles；不改packed位序，不缩到1024，不新增FP生成。

### 5.2 下游输入和一次prepare

标签源：data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging；
split：data/splits/mips_paper5_outer/{task}.json；
任务顺序：eea、egb、egc、ei、eps、nc、xc，共5875条。

只读引用现有2D缓存results/ua2d_transfer_20261022/r1/inputs及FP缓存results/ua2d_fg10k_morgan_20261023/r1/inputs。新周期一次CPU prepare从已有DirectMCLSource／LMDB几何构造本周期2D＋3D＋FP输入缓存；使用当前UA几何builder及同样距离单位／normalization，不能替换为SchNet／GeoGNN构象或重新运行RDKit。旧缓存不改，禁止训练中在线构造／生成fallback。

prepare最多物化5875条输入，新缓存构建1次；新FP生成、构象生成、模型forward均0。PT模板及逻辑映射只建立引用，不重建911391条模板。按task＋original_row＋sample_key＋normalized_smiles连接，保留原split，cache不存性质target或拟合label统计。

### 5.3 用户已选缺失策略

静态源中Egb 7条、Egc 8条共15条／8个结构无有效3D，其他五任务无此缺失。保留全部5875条；3D可用行5860条。缺失名单固定来自原geometry_status，prepare记录具体row/key及原因，不能因后续异常随意扩大白名单。

- 三模态顺序固定[2D,3D,FP]；每行availability=[true,geometry_available,true]。
- 无有效3D的样本不进入3D编码器，不伪造坐标或bond token；对有效3D子batch计算后scatter回原行，缺失位置用有限零占位并始终带mask。
- UNI3M：MHA的3D key/value被mask；注意力池化的3D logit置−∞。缺失query的结果也不得被池化，最终3D权重为0，其他两个权重和为1。
- TMOL3M：3D gate logit在softmax前置−∞，其权重严格为0，另外两路重新归一化；熵只累加实际权重，0·log(0)按0处理。
- 涉及3D的ALIGN仅取有效交集；2D–FP配对仍使用全部可用行。每对对比去重后不足2个身份时，该对loss与count记0，不能产生空负样本NaN或丢弃该行的性质MSE。
- 原PASS构象若出现新的schema／物理原子计数／角度路径不匹配，应停止prepare，不把它追加为“缺3D”来绕过故障。
- 所有任务／fold／seed及TEST样本保持原样，因此可直接与原B_A020正式预测比较。一个候选的训练／validation呈现中，缺3D分别2870／1330次，有效3D分别1312150／328650次；TEST中缺3D15条。

## 6. UNI3M：Uni-Poly式直接融合

### 6.1 投影和融合

~~~text
h2 [B,512] → Linear(512,256) → z2 [B,256]
h3 [B,512] → Linear(512,256) → z3 [B,256]
FP [B,2048] → Linear(2048,256) → zFP [B,256]
stack(z2,z3,zFP) → [B,3,256]
→ 1层 MHA＋FFN＋残差／LayerNorm
→ 三模态注意力加权池化 → f [B,256]
→ Linear(256,256) → ReLU → Dropout(0.1) → Linear(256,1)
~~~

- d=256和温度0.07来自论文。单层、8头、FFN宽1024、GELU、dropout0.1、post-norm是论文未核实参数的项目固定配置，无参数搜索。
- 融合MHA使用标准256维／8头的注意力尺度；不能误把原UA内部attention denominator=512套给融合层。原UA内部尺度保持不动。
- T1=LN(T＋MHA(T,T,T))；T2=LN(T1＋FFN(T1))。无额外modality embedding。
- 三个模态各有Linear(256,1)打分，按论文的模态加权方式softmax，f=Σa_m·T2_m；按第5节mask。
- 性质头直接消费f。没有H(h＋R·u)的512维旁路、零初始化残差投影或两个性质预测的后处理加权。

### 6.2 UNI ALIGN和训练loss

对齐使用融合前z2、z3、zFP，余弦相似度、τ=0.07，正样本为同一sample identity。每个有向配对采用包含正样本的InfoNCE：

$$
\ell_{a\to b}
= -\frac{1}{n}\sum_i
\log \frac{\exp(s_{ii})}{\sum_j\exp(s_{ij})},
\qquad
s_{ij}=\operatorname{cos}(z_i^a,z_j^b)/0.07.
$$

三个无向模态对均做双向，共6个有向项，L_ALIGN_UNI为6项均值。稳定实现使用logsumexp。每对先按既有identity语义稳定保留重复身份的首项，避免同一身份成为负样本；不能按FP相似度删负样本。

PT：L_mask2D＋L_mask3D＋0.20·r(t)·L_ALIGN_UNI。
FT：只用性质MSE，融合模块和全部有效模态最终共同微调。

0.20是项目在保留UA mask任务时的固定混合系数，沿用用户此前名义参考值；不是论文声明的原值。不保留旧ALIGN×2，也不把6项均值额外乘2。r(t)=min(1,t/1000)，仅全局开头一次。PT不构造MHA／池化／性质头；对齐训练原生编码器、norm和三个模态投影，FT再构造融合层。

## 7. TMOL3M：TMolNet式任务条件门控

### 7.1 表示、对比投影、门控

~~~text
h2 [B,512] → A2=Linear(512,256) → z2 [B,256]
h3 [B,512] → A3=Linear(512,256) → z3 [B,256]

FP2048 → Linear(2048,512) → ReLU → Dropout(0.2)
       → Linear(512,256)  → ReLU → Dropout(0.2)
       → Linear(256,256) → zFP [B,256]

Q2,Q3：Linear(256,256) → ReLU → Linear(256,256)
q2=Q2(z2), q3=Q3(z3), qFP=Q2(zFP)

任务元数据 → TaskEmbedding(8维)
concat(z2,z3,zFP,T8) [B,776]
→ Linear(776,512) → ReLU → Linear(512,3) → masked softmax
→ f=Σa_m·z_m [B,256]
→ Linear(256,256) → ReLU → Dropout(0.5) → Linear(256,1)
~~~

- 256、门控512、任务8维、FP隐藏512／256、FP dropout0.2及性质头dropout0.5均有作者代码依据。A2/A3是UA原生512维替换作者256维编码器时必需的外部适配。
- 作者源码的FP与graph共享对比投影，本项目显式保留qFP=Q2(zFP)，不再注册一个重复参数别名；Q3独立。门控使用z，ALIGN使用q，二者不得混淆。
- TaskEmbedding采用作者mode=add：任务数量嵌入＋任务类型嵌入，输出8维。每个独立性质模型num_tasks=1，索引task_num=0、task_type=1（regression）；不输入标签数值或validation／TEST统计。PT没有任务门控。
- 单任务内该向量为常量，不能声称已经证明跨七任务共享或任务迁移。任务名称one-hot不替换该定义。
- 本轮不引入SMILES编码器或混合指纹；gate输出从原4维改为3维。直接从f回归，无原图残差旁路。

### 7.2 TMOL ALIGN及熵项：保留作者的数学差异

模态顺序固定[2D,3D,FP]；有向循环为(2D→FP)、(3D→2D)、(FP→3D)。使用作者cl_loss排除正样本的分母，τ=0.1：

$$
s_{ij} =
\frac{q_i^a\cdot q_j^b}
{\|q_i^a\|\|q_j^b\|+10^{-8}}\,/\,0.1,
\qquad
\ell_{a\to b}
=\frac1n\sum_i
\left[\log\left(\sum_{j\ne i}\exp(s_{ij})+10^{-8}\right)-s_{ii}\right].
$$

L_ALIGN_TMOL为3个循环项的**和**，不自动改成6项双向均值。按第5节有效性和identity去重，n＜2时记零有效项。实现以off-diagonal logsumexp及logaddexp(logden,log(1e−8))保持数值稳定，不直接exp大logit再相减。

这是作者代码的decoupled／InfoNCE-style损失，不等于UNI的标准InfoNCE；数值可为负。负ALIGN或负总loss本身不作为失败，NaN/Inf仍必须停止。不得为了让loss非负而静默把正项加回分母或clip为0。

门控熵H=−mean_i Σ_m a_im·log(a_im＋1e−8)。作者loss中的−βH鼓励较高熵／平衡使用，不是鼓励稀疏。

- PT：L_mask2D＋L_mask3D＋0.08·r(t)·L_ALIGN_TMOL。
- FT：L_property_MSE＋0.08·L_ALIGN_TMOL−0.08·H。
- α=β=0.08、τ=0.1来自作者源码；PT的全局1000-update对齐warmup是本项目保留UA训练节奏的适配，FT两系数固定，不另warmup或调参。
- PT只训练编码器／norm／A2／A3／FP编码器／Q2／Q3及原mask头，不构造没有性质监督的gate或性质头，熵项在PT为0。FT引入门控及熵项，并按作者方案继续模态对比。
- validation／TEST只预测性质，不计算用于训练的跨样本对比更新；不让推理结果依赖同batch其他分子的特征或标签。

## 8. 共同PT执行规格

- 每候选从同一原生step0训练10000 updates，world1、micro168、accumulation6、有效batch1008，seed42；两个候选共20000 updates／20160000样本呈现。
- 每个微批的对比候选最多168，不能把6个微批合为1008负样本池；六微批只做梯度累积。mask任务沿用原task_sum及有效计数；ALIGN逐有向项积累anchor sum/count后按各方案的均值／和定义组合，不重复除accumulation或乘world。
- 两个mask权重恒为1，原任务计数和几何权重不改。各ALIGN权重仅按其规定r(t)在全局开头warmup一次，5000边界不重置。
- LR每5000 updates一周期：前500 warmup到2e−4，其后cosine到1e−6，共两周期；AdamW，weight_decay0、betas(0.9,0.999)、eps1e−8、clip1。优化器、数据采样和RNG连续，不在周期边界重建optimizer。
- 沿用UA BF16前向和关键loss／相似度／logsumexp的FP32计算，FP位输入解包检查为真实binary。不得自动改精度、batch、累积或epoch应对失败。
- 使用明确的新三模态模型随机流策略disjoint_modal_v1：B=86400010；各模态本地forward计数k范围0≤k＜60000，2D seed=B＋2k，3D seed=B＋2k＋1，FP seed=B＋120000＋k。2D／3D沿用已修复v2公式，FP使用不相交区间。记录policy／B／计数，5000边界不断流；终态计数60000只作保存，不再forward。不同seed不被描述为统计独立性证明。
- 数据侧mask／geometry随机流沿用当前按样本position分离的生成器；新增FP随机流不能改变原采样顺序或2D／3D corruption。模型初始化使用隔离RNG。
- 有序PT数据workers4、每worker线程1，单GPU主进程线程最多2。每个候选60000个微批，不追加PT验证forward或中途FT。保存step10000 deploy／取证resume和全部实际日志；resume文件不产生恢复授权。

## 9. 下游微调、选择和正式TEST

- 每候选7任务×fold0—4，共35个FT；seed=42+fold。固定70轮：10 head＋60 joint；batch24、FP32、workers0、drop_last=false、AdamW、weight_decay0.02、clip5。
- head阶段冻结原生2D／3D编码器和norm2／norm3；模态适配、FP编码器、有效对比头、融合和性质头训练，lr=3e−4，CosineAnnealingLR(T_max10,eta_min3e−5)。
- joint阶段原生编码器／norm lr=1e−4，其他有效模块lr=1e−3；CosineAnnealingWarmRestarts(T_0=10,T_mult=1,eta_min1e−6)。按原UA阶段切换重建optimizer，PT周期LR不带入FT。
- UNI按第6节FT loss，TMOL按第7节FT loss。训练只读取train标签和train批内对比；每epoch恰一次validation，BEST按性质validation R²严格提高选择，完全平局保留较早epoch，保存该次预测，不再额外运行BEST validation。
- 所有任务包括Xc均只读继承B_A020对应train-only StandardScaler；检查train身份与scaler字段，不用validation／TEST拟合。模型输出标准化性质，正式预测统一逆变换为原单位。
- 70个FT全部完成后，按七任务、五折等权validation macro R²，在B_A020、UNI3M、TMOL3M中选一个全局FINAL；完全平局顺序B_A020、UNI3M、TMOL3M。没有后处理全局w或FP权重网格，不按任务挑赢家；融合层内的模态权重由训练学习。
- 一次性冻结70个BEST的路径／epoch／scaler、35个B_A020引用、全局FINAL、输入和缺失mask身份；随后两个候选各35个正式TEST均执行一次。B_A020旧正式预测只读复用，无基线新增推理。
- 预登记比较：UNI3M−B_A020、TMOL3M−B_A020、TMOL3M−UNI3M、FINAL−B_A020。逐task/fold配对，报告改善fold数、五折mean±std(ddof=1)、七任务macro；差值均用未舍入值。
- 默认性能通过为FINAL正式macro−0.785044436513689≥0.010，即原值macro≥0.795044436513689。未达门槛如实交付，不能据TEST回调损失、温度、mask、dropout、epoch、方法或任务赢家。

## 10. 固定总预算

| 项目 | 每候选上限 | 本周期累计上限 |
|---|---:|---:|
| 输入／初始化prepare | 共享 | 1 |
| 下游新缓存物化 | 共享 | 5875行／1次；3D有效5860、缺失15 |
| 新FP／构象生成 | 0 | 0 |
| PT启动／updates | 1／10000 | 2／20000 |
| PT微批／样本呈现 | 60000／10080000 | 120000／20160000 |
| FT启动／epochs | 35／2450 | 70／4900 |
| FT attempted updates | 56000 | 112000 |
| FT训练样本呈现 | 1315020 | 2630040 |
| validation调用／呈现 | 2450／329980 | 4900／659960 |
| 正式TEST单元／呈现 | 35／5875 | 70／11750 |
| 其中有效3D TEST呈现 | 5860 | 11720 |
| 基线新推理／输出融合政策 | 0／0 | 0／0 |
| KRR／核近似／FP预测任务 | 0 | 0 |
| 最终CPU suite尝试 | 共享 | 1 |

这是两个新三模态方案的预算提案，不给旧失败周期补额度，也不抹去其177次启动等历史消费。每个作业启动时不可退地预留完整单元额度，失败／部分运行／timeout均计入；不以actual少于预留退款，不另起编号清零。所有预测评估都是模型实际forward，不能沿用历史缓存融合的“零新增推理”描述。

## 11. 实现文件、接口与允许修改

未来获执行授权后，仅新增下列隔离文件，更新本计划及新产物：
- src/modules/ua3m_paperfuse.py：原生UA包装、三模态投影、Uni3MFusion、Tmol3MGating、两个ALIGN实现、checkpoint显式加载。
- src/dataset/ua3m_paperfuse.py：原缓存读取、FP输入解包、下游三模态缓存、有效3D子batch、availability及身份连接。
- src/training/ua3m_paperfuse.py：loss sum/count、PT／FT、一次选择冻结、正式指标和预算语义。
- scripts/pretrain_ua3m_paperfuse.py、scripts/run_ua3m_paperfuse.py：明确的新PT worker和单writer controller。
- configs/ua3m_paperfuse_r1.json、tests/test_ua3m_paperfuse.py。

建议接口：encode_modalities(data)返回z2/z3/zFP/availability；contrastive_features返回UNI的z或TMOL的q；fuse返回f与[B,3]权重；predict只用输入和固定任务元数据；training_loss才允许接labels。PT／FT schema分别明确，FP是输入而非prediction target。

可只读导入现有原生编码器、builder、task_sum和已修复原子退出／PID门控逻辑。不得修改旧无FP六文件、旧A020五文件、数据、缓存、AGENTS、SOTA_RESULT、PROJECT_HISTORY或模型历史。不得调用旧controller入口、旧默认OUT、旧aggregate/deliver来写旧产物；需要共享源码改动时停止并说明具体依赖，不能擅自扩范围。

## 12. 资源、调度和停止条件

- 仅物理GPU3。PT固定最多1路，FT／TEST最多4路独立world1 GPU worker；阶段不混跑。不把shell、CPU wrapper或I/O worker重复当GPU任务。
- 外部／未知compute PID、nvidia-smi失败或host/container PID无法核对均暂停新增派发；未能派发不弹出任务、不创建训练输出、不计启动。不能换GPU或把未知占用当空闲。
- 所有执行在tmux session Uni-Poly独立window ua3m-paperfuse-r1，worker独立window；单writer，保留cwd、命令、PID＋start_ticks、stdout/stderr、开始／结束时间和真实exit。
- 一次prepare→2 PT→70 FT及既定validation→一次选择／冻结→70正式TEST→完整报告→唯一最终CPU。禁止前置pytest、阶段smoke、benchmark、逐文件验收、compileall或追加回放审计。
- 总墙钟48小时，从新周期明确执行授权起算并包含实现、等待和停止。prepare≤3600秒，单PT／FT≤21600秒，单TEST≤1800秒，CPU≤600秒，均受剩余总墙钟约束。
- wrapper退出值必须以同目录临时文件＋os.replace发布；reader只接受合法终态，空／缺失不能int，收尾等待至多10秒，坏值／已死无exit带具体job／路径／身份报错。首个终止后屏蔽重复信号打断收尾，清理只处理核实身份的自有进程组，30秒有界；不吞异常当成功。
- artifact／checkpoint／schema、原PASS几何、row/key或FP位语义不匹配，NaN/Inf，异常退出，writer冲突、正式覆盖风险或预算超限，停止受影响队列及后续阶段并保留现场。
- 不自动retry／resume，不减batch、不换精度／设备、不增seed／epoch／模态／对照，不安装升级依赖补救。TMOL负对比loss是预设数学，不单独触发停止。资源／时间无法继续交阻断；实现或结果问题交需返修；完整执行后交待审查，性能单独判定。

## 13. 唯一最终CPU测试

全部实现、授权实验、正式结果及报告完成后，只尝试一次：

~~~bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout --signal=TERM --kill-after=10s 600s python -m pytest -q -p no:cacheprovider tests/test_ua3m_paperfuse.py
~~~

必要覆盖集中于本次问题：原UA结构／mask语义和公共初始化；FP仅输入且没有预测头／BCE；两种256维融合及梯度；UNI含正项的6方向均值、TMOL排正项的3方向和及可负性；熵负号；FP与2D共享Q2；缺3D权重严格0／其他归一化、有效子batch和固定15行；PT全排无效几何；行/key连接及train-only scaler；任务元数据不读标签；PT／FT阶段optimizer有效参数覆盖；RNG有限域无重复与5K边界连续；冻结前禁止TEST、预算、原子退出／重复终止和模拟占用门控。

使用合成CPU输入、少量metadata及模拟进程／时间；不运行GPU、RDKit生成、真实训练或大checkpoint推理，不做60000次模型forward。保留有效断言，不skip／xfail或放宽容差。失败保留真实exit和消费，停止且不重跑。

## 14. 交付、SOTA与独立审查

结果根：results/ua3m_paperfuse_20261025/r1/；
日志根：logs/ua3m_paperfuse_20261025/r1/。

交付authorization/user_authorization.json、基于版本／已有改动、参考来源／适配参数表、公共初始化引用及新头初态、inputs metadata／geometry_missing名单／coverage、两个PT config／日志／deploy及取证resume、70个FT config／scaler／BEST／70轮history及BEST validation预测、selection/validation_selection.json和final_frozen.json、70个正式TEST预测、budget.json、execution_manifest.json、最终CPU真实exit及完整日志、final_review/results.json／report.md／actual_source_diff.patch。

逐方案列任务mean±std、macro、原值差、改善fold、模态可用率、参数量、PT／FT／推理真实成本及缺失3D处理。预测保留原单位、完整精度、original_row、sample_key和availability，不复制旧模型充当新训练。以下参考必须逐项写入周期结果及审查结论：

| 任务 | B_A020正式TEST参考 | SOTA_RESULT参考均值 | 对应论文模型 |
|---|---:|---:|---|
| Eea | 0.929±0.010 | 0.934 | Periodic-TDL |
| Egb | 0.931±0.013 | 0.923 | Periodic-TDL (+Chem, +RC) |
| Egc | 0.918±0.012 | 0.911 | Periodic-TDL (+Chem, +RC) |
| Ei | 0.811±0.029 | 0.839 | Periodic-TDL |
| EPS | 0.713±0.105 | 0.787 | Periodic-TDL |
| Nc | 0.808±0.066 | 0.856 | Periodic-TDL |
| Xc | 0.386±0.081 | 0.423 | Periodic-TDL |

每项写新项目值、参考值、新项目减论文参考、五折／seed口径和产物路径；未完成标未完成／不可比较。论文为各任务最佳均值包络，数据／split／协议等价性未证明，只能数值对照。共享TEST参与路线开发，不称独立盲测；D_A020父PT legacy随机偏差等历史限制保留。本轮与基线多因素变化，不能据提升证明FP／3D独立贡献或interaction，也不宣称原论文完整复现／SOTA。

执行方交待审查，由Codex复用现有证据独立审查，不重跑测试／实验；审查通过前不写PROJECT_HISTORY，不自动开启下一周期。

## 15. 可复制给DSH的执行提示词

以下为待授权模板；用户实际发送才构成新周期执行授权，不能从本文件存在推定已授权。

~~~text
现在授权你执行CURRENT_PLAN.md：
UA3M-PAPERFUSE-20261025-01 / r1。
工作目录/root/workspace/Uni-Poly-Plus-master，基于dev/cd36ae7280b159f881b483ad5814835ff7b74c4a及当前已有改动。先读AGENTS和本计划、查看git status与相关diff，记录本条授权原文、时间、版本和已有改动。不读TODO、不同步远端、不做哈希验证。旧交接快照、模型、数据、失败证据和账本保留，不启动旧controller；旧FINAL=D_A005不变。

实现且仅实现UNI3M和TMOL3M，模态固定UA 2D、UA 3D、Morgan FP。原生两路512维/6层、几何与mask任务保持第4节；新候选都从共同step0+V1 heads各训练10K，不加载已训练A020/FP-on模型冒充同预算。FP仅作为真实结构输入参与ALIGN和下游性质预测，删除FP预测头/BCE/target；不使用KRR、蒸馏、核近似或输出层预测融合。

所有具体维度和loss以第6—7节为准：两方案公共空间256维；UNI三token、1层8头、FFN1024、模态注意力池化、直接256维性质头，标准InfoNCE温度0.07，三对双向6项均值；PT系数0.20全局warmup1000一次，FT性质MSE。TMOL采用作者FP MLP2048→512→256→256，FP dropout0.2；任务embedding8，gate776→512→3，性质头256→256→1/dropout0.5；Q2/Q3为256→256→256，FP与2D共享Q2；循环(2D→FP,3D→2D,FP→3D)，温度0.1、排除正项分母、三项求和，允许负loss。PT系数0.08全局warmup1000一次；FT=MSE+0.08*ALIGN−0.08*gate entropy，不误写成鼓励稀疏。PT不构造无监督的gate/MHA/性质头，FT再初始化；不保留旧512维残差旁路。

数据与缺失策略明确批准：保留七任务5875条及原split；Egb7条/Egc8条缺3D时不运行该行3D编码器，mask其attention/gate/pooling并重新归一化其他模态权重，不能删除TEST行或生成新构象。有效3D子batch后scatter回原行；原PASS新schema/物理计数异常停止，不能扩大缺失白名单。涉及3D的ALIGN仅用有效交集，不足2个身份记零有效对比，性质MSE仍保留。PT仍用原910756有效逻辑slot，635条无效几何不参加任何PT目标。

一次CPU prepare只读复用原PT模板、逻辑映射、2D/FP缓存和DirectMCLSource既有3D，物化新下游缓存5875行及公共初始化/256维新头；模型forward、FP生成、构象生成均0。不重建PT模板，不在线fallback，不按数组位置拼接；task+original_row+sample_key+SMILES连接。旧FP缓存历史BCE/head字段不继承，新labels不含fp_target。

固定预算：prepare1；PT2次/20000 updates/120000微批/20160000样本呈现；FT70次/4900 epochs/112000 attempted updates/2630040 train呈现；validation4900次/659960呈现；正式TEST70单元/11750呈现；唯一最终CPU1次。FP/构象生成、KRR、基线新推理、输出融合政策均0。失败/部分/timeout计消费，不退款、不retry/resume、不增臂/seed/epoch或另起编号领取额度。

PT每候选world1、micro168×acc6、seed42、两轮5K LR周期：每周期500 warmup到2e-4，再cosine至1e-6；AdamW wd0/betas(.9,.999)/eps1e-8/clip1，UA BF16与关键FP32 loss，optimizer/采样/RNG连续，ALIGN只在全局开头warmup一次。对比候选每微批最多168，不能做1008负样本池。使用第8节三模态disjoint_modal_v1：B=86400010，2D=B+2k、3D=B+2k+1、FP=B+120000+k，k<60000；不把不同seed宣称为统计独立。

每方案7任务×5fold，seed42+fold、batch24、FP32、workers0、70轮10head+60joint，optimizer/scheduler及train-only scaler按第9节，BEST仅validation性质R²严格提高/平局较早。所有FT完成后，只在B_A020/UNI3M/TMOL3M中选唯一全局FINAL，val macro完全平局优先B_A020、UNI3M、TMOL3M。一次冻结70个BEST、35基线引用和缺失mask，再完成两个方案各35个正式TEST；不据TEST重选或拼任务赢家。正式门槛仍为未舍入FINAL−0.785044436513689≥0.010。

仅物理GPU3；PT最多1路，FT/TEST最多4路独立GPU worker，不混阶段；CPU主进程线程2，PT有序I/O workers4各线程1，FT workers0。外部/未知PID、查询失败或namespace不能对应均busy，等待且不提前计启动，不换卡。tmux Uni-Poly独立window ua3m-paperfuse-r1，worker独立window，单writer，保留命令/cwd/PID身份/日志/真实exit。总墙钟48h含实现等待；prepare3600秒、单PT/FT21600秒、TEST1800秒、CPU600秒且受剩余总时间约束。

只新增第11节七个隔离实现/配置/测试文件及新产物，更新CURRENT_PLAN；不改旧无FP/A020源码、数据、缓存、AGENTS、SOTA_RESULT或PROJECT_HISTORY。原子exit发布、有界读取和重复终止安全收尾按第12节，不复制旧控制器竞态。身份/schema/原PASS几何/NaN/Inf/异常exit/预算/writer或覆盖问题停止受影响队列和后续阶段，保留现场，不减batch换精度或安装升级依赖。

执行顺序：实现→一次prepare→2 PT→70 FT及既定validation→一次选择冻结→70正式TEST→完整报告→唯一最终CPU。全部完成后只尝试一次：
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout --signal=TERM --kill-after=10s 600s python -m pytest -q -p no:cacheprovider tests/test_ua3m_paperfuse.py
只覆盖第13节必要合成CPU正确性，不跑阶段测试/smoke/compileall/benchmark/全仓回归，不skip/xfail，失败不重跑。

交付results/ua3m_paperfuse_20261025/r1/，日志logs/ua3m_paperfuse_20261025/r1/。按第14节交授权、依据/适配表、输入和15条缺失名单、初始化、PT、70FT、一次冻结、70正式预测、budget/manifest、CPU真实exit与report/results/source diff；列七任务、配对差值、SOTA参考模型/数值/差值/协议限制、缺失可用率和真实成本。完整后交待审查，性能独立判定；失败需返修、资源时间阻断，不自动归档，由Codex复审。
~~~

## 16. 本次文档交付记录

本次完整覆盖CURRENT_PLAN，并保存覆盖前全文至第1节快照。已固定两种架构、论文／代码依据、ALIGN差异、缺3D策略、初始化、数据、训练／评估预算、资源、停止条件及DSH执行提示词。Uni-Poly未核实参数已标为项目选择，未伪称论文原值。

本次实际实验消费全部0；没有修改源码、模型、数据、旧产物、PROJECT_HISTORY或用户TODO。只检查最终文档diff与格式。新周期实现及实验状态仍为待授权，旧周期状态和事故以快照及原实际产物为准。

## 17. 当前授权执行记录

- 2026-10-25T10:32:33.682164+08:00：收到当前完整执行授权并保存原文，基于 dev/cd36ae7280b159f881b483ad5814835ff7b74c4a 及当前已有改动。before、status及相关diff见 `results/ua3m_paperfuse_20261025/r1/authorization`。保护旧11个源码/测试、文档快照和用户TODO（不读），不远程同步、不新增哈希验证。仅新增第11节指定文件和本周期产物；48小时从本授权登记起连续计时。
- 状态执行中，tmux `Uni-Poly:ua3m-paperfuse-r1`；执行方自检不代替后续Codex独立复审，不归档。

<!-- UA3M_PAPERFUSE_STATUS_START -->
- 2026-10-25T10:44:34.552096+08:00；状态：**需返修**；阶段：交付报告。
- # UA3M-PAPERFUSE r1 实际交付

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

- 账本 `/root/workspace/Uni-Poly-Plus-master/results/ua3m_paperfuse_20261025/r1/budget.json`；日志 `/root/workspace/Uni-Poly-Plus-master/logs/ua3m_paperfuse_20261025/r1`；独立复审PENDING，未归档。
<!-- UA3M_PAPERFUSE_STATUS_END -->

### UA3M r1 事故收口与需返修交付

- 2026-10-25T10:46:02.426988+08:00：prepare与controller真实exit均1。根因为本次prepare漏建 `initialization/heads/eea/`，首次保存fold0头时torch.save报父目录不存在；源码保持故障时原样，没有修复后重跑。
- 一次prepare额度已消费；缓存实际物化0行（5875行只是完整不可退启动预留），仅留下两个模态适配器部分初始化。35头、下游缓存、15行缺失名单尚未完整建立，不能将计划覆盖数当作本次已验证覆盖。
- 新增PT/FT/validation/正式TEST/FP或构象生成/模型forward/GPU/KRR全部0，选择/冻结未执行，最终CPU尝试0；新方案全部七任务性能未完成／不可比较。原参考及协议限制保留。
- 状态**需返修**；当前相关controller/worker已退出，无自动retry/resume，不退款或换编号补额度，不归档。事故 `results/ua3m_paperfuse_20261025/r1/final_review/incident.json`；报告 `results/ua3m_paperfuse_20261025/r1/final_review/report.md`；日志 `logs/ua3m_paperfuse_20261025/r1`；tmux `Uni-Poly:ua3m-paperfuse-r1`。下一步独立只读审查后另行制定返修与prepare额度授权，不能自行重启。

## 18. Codex独立审查：prepare首次保存失败

### 18.1 版本、证据和完成层级

- 审查快照：2026-10-25 10:50—10:56（Asia/Shanghai）；本地dev/cd36ae7280b159f881b483ad5814835ff7b74c4a。CURRENT_PLAN及用户TODO已有改动，旧11个及本周期7个源码／配置／测试文件未跟踪，旧文档快照保留。未读取TODO，未修改任何源码或运行测试、prepare、模型forward、GPU任务。
- 核对了AGENTS、原科学合同、七个实际实现文件、SOTA_RESULT、授权、budget／manifest、failure／incident、coverage、prepare与controller真实exit、日志及部分目录。七个交付文件与final_review/actual_source_diff.patch逐文本一致，支持失败后未修源码的报告。
- prepare/exit.txt及end.json均为1；controller/exit.txt为1。budget只有一个CPU prepare作业，CUDA_VISIBLE_DEVICES为空；对应controller、wrapper、timeout和worker的四个PID＋start_ticks均无存活匹配进程。
- 实际只留下initialization/UNI3M_modal.pt、initialization/TMOL3M_modal.pt和inputs/coverage.json；没有保存好的性质头、graphs.lmdb、rows.json、geometry_missing.json或完整metadata。pretrain／finetune／selection／formal目录均不存在。CPU尝试0；FINAL未选择／冻结。

### 18.2 发现与影响

1. **P1：确认的运行缺陷。** src/training/ua3m_paperfuse.py第86行只创建initialization根目录，第92行直接保存initialization/heads/eea/fold0.pt，漏建heads/{task}父目录。调用的src/training/glt_dual_runtime.py第536—540行save_checkpoint只负责临时文件与rename，不创建目录；真实torch.save堆栈与此完全吻合。首次保存前失败，LMDB直到第97行才会打开，因此缓存物化实际为0，不是用计划行数推算。
2. **P1：旧目录及额度不能直接复用为重启许可。** 旧inputs和initialization目录已存在，prepare对其使用exist_ok=False；controller也有一次性lock／budget守卫。只补mkdir后重跑旧入口会撞既有现场，删除目录或清账则违反证据保护与不可退预算。返修须有独立输出根、明确的第二次prepare授权及累计账本。
3. **必要覆盖缺项。** 当前测试有模型／ALIGN／缺失mask／预算／退出门控等合成覆盖，但没有首次保存35个嵌套头目录的回归，也没有覆盖新旧执行root在prepare、PT、FT、freeze、TEST中的一致解析。最终suite尚未运行，不能把已写测试当作通过。返修集中补这类真实故障覆盖，不追加前置suite或smoke。

其他checkpoint保存点已读：两个modal文件的父initialization已创建；PT deploy／resume的作业目录、FT BEST的单元目录均在保存前创建。未确认第二个相同的漏建目录现场；不以此承诺后续准备、几何构造或模型训练一定成功。

静态核对范围内，两个原生UA分支／mask头、UNI六方向均值、TMOL排正项三方向和及熵负号、FP输入且无预测头、Q2共享、有效3D子batch、两阶段FT、validation选择及冻结边界与计划一致。尚未执行的内容包括5875行读写及15条缺失完整物化、PT与FT真实forward、优化器轨迹、正式预测和CPU suite，均不得升级为已验证。

### 18.3 预算和性能状态

父已收费prepare_starts=1、cache_rows=5875；实际cache_rows=0，其他科学／评估计数全0。5875是完整不可退预留，不是已生成行数，也不能因实际0退款。两个modal初始化文件仅是部分输入准备，不是预训练checkpoint。

| 任务 | B_A020历史参考 | 论文最佳参考 | 对应模型 | UNI3M／TMOL3M本周期 |
|---|---:|---:|---|---|
| Eea | 0.929±0.010 | 0.934 | Periodic-TDL | 未完成／不可比较 |
| Egb | 0.931±0.013 | 0.923 | Periodic-TDL (+Chem, +RC) | 未完成／不可比较 |
| Egc | 0.918±0.012 | 0.911 | Periodic-TDL (+Chem, +RC) | 未完成／不可比较 |
| Ei | 0.811±0.029 | 0.839 | Periodic-TDL | 未完成／不可比较 |
| EPS | 0.713±0.105 | 0.787 | Periodic-TDL | 未完成／不可比较 |
| Nc | 0.808±0.066 | 0.856 | Periodic-TDL | 未完成／不可比较 |
| Xc | 0.386±0.081 | 0.423 | Periodic-TDL | 未完成／不可比较 |

参考来自原正式结果及SOTA_RESULT，不是本次新评估；原五折mean±std、ddof=1口径不变，论文为逐任务最佳包络，协议等价性未证明。新方案没有TEST，不能宣称运行或性能通过。旧FINAL=D_A005已核对仍保留。

**独立结论：事故和零训练消费得到核实，维持需返修；不通过实现／性能审查，不归档。** 现有不足可以提出最小修复方案，但不授权自动重跑。5875／5860／15仍为计划及历史源口径，本次prepare未走到覆盖核对阶段。

## 19. 当前返修计划：最小修复、追加一次prepare并完成原矩阵

### 19.1 身份、授权与不变范围

- ID：UA3M-PAPERFUSE-REPAIR-20261025-01；修订prepare-r1；状态：**执行中**。用户本轮明确授权执行第19节并批准唯一prepare增量；本计划是原UA3M-PAPERFUSE/r1的累计返修，不是新科学周期或预算清零。授权原文及执行证据记录见第22节和返修authorization。
- 提议仅增加一次定额prepare及5875行完整预留，保留原失败不退款；其余原定2 PT、70 FT、70 TEST及1次最终CPU均尚未消费，保持原上限并在返修中执行。不追加候选、task、fold、seed、epoch、超参网格或额外测试轮次。
- 第2—14节科学合同保持不变：UNI3M／TMOL3M、原生UA 2D／3D、Morgan输入且无FP预测、各自ALIGN和loss、公共step0、10K PT、256维融合、15条缺3D保留并屏蔽、基线原值0.785044436513689及正式Δmacro≥0.010。只允许目录和执行root／账本适配，不借返修改变模型、loss或数据。

### 19.2 最小修复和旧现场保护

1. 在返修目录保存training、controller、PT worker、tests及当前计划的before文本，引用原budget／manifest／failure／incident／source diff和coverage。原r1所有旧inputs、两份modal初始化、日志、预算、manifest、lock和final_review只读；不删除、移动、覆盖或将coverage改成COMPLETE。
2. 在本周期training的初始化保存路径创建heads/{task}父目录，再保存各fold；可以使用局部保存辅助函数，但不得修改共享glt_dual_runtime.save_checkpoint、旧源码或模型定义。检查本周期所有checkpoint保存点的父目录契约，保留合法BEST更新语义，不给初始化文件增加静默覆盖。
3. 新增显式返修入口及唯一执行root：results/ua3m_paperfuse_20261025/r1/repairs/prepare_repair1/。新prepare、初始化、缓存、PT、FT、selection、formal、预算、manifest、报告全部写入该子目录；不能直接重跑旧controller或删除旧lock绕过守卫。
4. 用同一个显式root解析prepare、head_path、modal_path、parent_path、unit_path、check_prepared、freeze、TEST及报告。PT worker和FT／TEST子进程均通过明确参数启用相同root，不能只修改controller父进程全局变量。原科学PLAN／revision=r1、checkpoint schema及m.CONFIG保持兼容；prepare-r1只记录执行修订，默认旧入口的已有数据解释不变。
5. 新prepare按原seed完整生成两个modal初态和35个性质头，不把旧两份部分初始化当作已完成prepare，也不另抽seed择优。旧公共step0、V1头、数据、PT模板／逻辑映射、2D／FP缓存只读复用。新的物化目标仍为5875行，不能把累计预留11750当作缓存目标。
6. 账本分父历史、新执行、累计三栏：原失败prepare永久结转，不能进入新活动队列反复触发check_failures，也不能被当成可恢复作业。新prepare仅允许一次；失败后即停止，没有第三次额度。dataset的完整覆盖要求仍为5875／5860／15，不能为了通过而放宽。
7. 本次仅允许最小修改src/training/ua3m_paperfuse.py、scripts/run_ua3m_paperfuse.py、scripts/pretrain_ua3m_paperfuse.py、tests/test_ua3m_paperfuse.py及CURRENT_PLAN。module、dataset、科学配置和旧无FP／A020文件不改；若出现必须改科学实现或数据语义的问题，停止并交证据，不擅自扩范围。无必要不另建通用框架。

### 19.3 累计预算、时间与资源

| 项目 | 原上限 | 父已收费 | 本次新增上限 | 修订后累计上限 |
|---|---:|---:|---:|---:|
| prepare启动 | 1 | 1 | 1 | 2 |
| cache_rows完整预留 | 5875 | 5875 | 5875 | 11750 |
| PT启动／updates | 2／20000 | 0／0 | 2／20000 | 2／20000 |
| PT微批／呈现 | 120000／20160000 | 0／0 | 120000／20160000 | 120000／20160000 |
| FT启动／epochs | 70／4900 | 0／0 | 70／4900 | 70／4900 |
| FT attempted updates | 112000 | 0 | 112000 | 112000 |
| 训练呈现 | 2630040 | 0 | 2630040 | 2630040 |
| validation调用／呈现 | 4900／659960 | 0／0 | 4900／659960 | 4900／659960 |
| 正式TEST单元／呈现 | 70／11750 | 0／0 | 70／11750 | 70／11750 |
| 有效3D TEST呈现 | 11720 | 0 | 11720 | 11720 |
| 最终CPU尝试 | 1 | 0 | 1 | 1 |

新增prepare和cache预留是本次需要明确批准的唯一上限增量。父actual cache_rows=0，新prepare成功后的实际总物化行数为5875；父5875预留继续保留，不能混写成实际11750行或退回额度。原科学上限、零FP／构象生成、零KRR、零基线推理、零输出融合政策不变。

本周期允许物理GPU 0、1、2、3，按第23节gpu0123-r1调度：PT每卡最多1路、全局最多2路（仅原两个候选）；FT／TEST每卡最多4路、全局最多16路。各作业world1、单卡运行，阶段不混跑；每GPU作业主线程2、每PT有序I/O workers4各线程1、FT workers0。逐卡核对占用与PID身份，已核实空闲且有槽位的授权卡可派发新作业；查询失败、外部／未知占用或namespace不能对应的卡视busy，不提前计启动。作业启动后固定其物理卡，不迁移、重启或重复计费。

tmux Uni-Poly独立window ua3m-paperfuse-prepare-repair1，worker独立window、单writer，保留命令／cwd／PID身份／日志／真实exit。沿用父48小时绝对截止：2026-10-27 10:32:33.682164+08:00，不因返修重置；修复和等待均计入。单prepare3600秒、PT／FT21600秒、TEST1800秒、CPU600秒，并受剩余总时间约束。授权到达时已超截止则交阻断，不默默续期。

### 19.4 执行顺序及唯一最终测试

获明确授权后：保存现场／最小修复／登记父消费及新root→唯一新增prepare→2 PT→70 FT及既定validation→一次选择冻结→70正式TEST→完整报告→唯一最终CPU。没有提前suite、smoke、compileall、benchmark或模型探针；原失败准备不是可恢复checkpoint，不使用自动resume。

只有新prepare真实exit0且新cache完整metadata、5875／5860／15物化覆盖、35头和两份modal初态均齐备后才能启动PT。运行中若原PASS几何不匹配，不允许扩大15条缺失名单或跳样本；按原停止条件保留现场。

validation的全局方法选择仍为B_A020、UNI3M、TMOL3M，完全平局按此顺序；一次冻结70个BEST、35个基线引用及缺失mask后再TEST，不按TEST回调结构、loss、温度、epoch、方法或任务赢家。正式性能门槛、七任务统计和SOTA口径均按原计划。

实现及所有授权实验／报告完成后，只尝试一次：

~~~bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout --signal=TERM --kill-after=10s 600s python -m pytest -q -p no:cacheprovider tests/test_ua3m_paperfuse.py
~~~

保留原有效覆盖，集中新增必要合成CPU回归：空临时root下创建并首次保存35个嵌套头路径；不覆盖已有初始化；一条有效3D和一条缺3D的缓存序列化／解码／collate闭环；原旧root只读与prepare／PT／FT／freeze／TEST新root一致；父失败与当前队列分离；原prepare1／cache5875与返修累计prepare2／cache11750分别正确，实际0＋5875不等于预留11750。使用小型对象及模拟输入，不读取大checkpoint、不构建真实5875行测试缓存或生成构象，不增加真实forward、GPU或阶段测试，不skip／xfail或放宽科学断言。

本次任一新失败、timeout、schema／身份／PASS几何／NaN/Inf／预算或覆盖／writer风险，停止受影响队列及后续阶段；不retry、resume、换卡、改batch／精度、装依赖或另起编号补额度。实现／结果故障交需返修，资源／绝对截止无法继续交阻断；全部完成交待审查，性能独立判定，不自动归档。

### 19.5 交付

结果根：results/ua3m_paperfuse_20261025/r1/repairs/prepare_repair1/；日志根：logs/ua3m_paperfuse_20261025/r1/prepare_repair1/。

交授权及追加prepare例外、before、parent_evidence（原budget／manifest对象和事故路径）、repair_diff、执行root映射、父／新增／累计budget、execution_manifest、完整新inputs及缺失名单／coverage、初始化、2 PT、70 FT、一次冻结、70正式预测、最终CPU真实exit与完整日志、report/results。科学产物完整要求同第14节，报告明确原失败、0行实际、部分初始化及新增一次prepare，不把旧账本清零或改写成成功。

七任务逐项列项目／SOTA参考／对应模型／差值及协议限制；未完成标未完成／不可比较。保持旧FINAL=D_A005，父final_review、输入现场、日志、预算和manifest只读，PROJECT_HISTORY不改。最终由Codex复用现有证据独立审查，不重跑实验或CPU suite。

## 20. 可复制给DSH的返修执行提示词

以下为待授权模板；用户实际发送才构成最小修复、追加一次prepare及剩余原矩阵的执行授权。读取本文件不构成失败后的重启许可。

~~~text
现在授权你执行CURRENT_PLAN.md第19节：
UA3M-PAPERFUSE-REPAIR-20261025-01 / prepare-r1。
这是原UA3M-PAPERFUSE/r1的累计返修，不能另起科学周期清零预算。目录/root/workspace/Uni-Poly-Plus-master，基于dev/cd36ae7280b159f881b483ad5814835ff7b74c4a及当前已有改动。先读AGENTS和第18—19节，查看status/相关diff；记录本条授权原文、时间、版本、父消费和已有改动。不读TODO、不远程同步、不做哈希验证，旧产物和FINAL=D_A005保留。

明确批准唯一预算增量：新增prepare最多1次、完整cache预留5875行；连同父失败累计prepare最多2次、cache_rows预留11750。父prepare exit1及5875预留不退款，父实际缓存0行。新的实际缓存目标仍为5875行，有效3D5860、缺失15；不得把累计预留当作目标行数。新prepare再次失败立即停止，无第三次额度。

先保存training/controller/PT worker/tests及计划before文本，引用原budget/manifest/incident/failure/coverage/source diff。旧r1 inputs、两个partial modal文件、日志、账本、lock及final_review只读，不删除、移动、覆盖或伪写COMPLETE。修复本周期training首次保存heads/{task}/fold的父目录创建；不改共享save_checkpoint或科学实现。

新增显式repair入口，所有新prepare/初始化/cache/PT/FT/selection/formal/账本/报告写results/ua3m_paperfuse_20261025/r1/repairs/prepare_repair1/，日志logs/ua3m_paperfuse_20261025/r1/prepare_repair1/。统一解析执行root并显式传给PT和FT/TEST子进程，不能只改controller全局路径。旧r1科学PLAN/revision/checkpoint schema/m.CONFIG保持，prepare-r1只记执行修订。新prepare按原seed完整生成两份modal和35个头；旧部分初始化不冒充完成，不额外抽seed。旧失败只计历史，不污染新活动队列，原budget不改。

只允许最小修改本周期training、controller、PT worker、tests及CURRENT_PLAN；module、dataset、科学配置、旧无FP/A020源码、数据/缓存、AGENTS、SOTA_RESULT和PROJECT_HISTORY不改。若须改变模型/loss/数据语义则停止交证据，不擅自扩范围。

原第2—14节科学合同全部不变：UNI3M/TMOL3M三模态、原UA2D/3D及mask任务、FP作为输入且无FP预测，作者对应ALIGN/门控/熵，step0+V1、各10K PT、256维融合、原7任务×5fold/seed42+fold、70轮及原优化器/调度。缺3D保留15行、屏蔽并重归一化，PT635无效行全排；不得新生成FP/构象、用KRR/蒸馏/核近似或输出融合。TMOL负loss允许，NaN/Inf仍停止。

本次剩余原科学上限：PT2次/20000 updates/120000微批/20160000呈现；FT70次/4900 epochs/112000 updates/2630040 train呈现；validation4900次/659960呈现；正式TEST70单元/11750呈现，其中有效3D11720；最终CPU累计仅1次、父尝试0。除prepare/cache两项外累计上限不增加；失败/部分/timeout不退款，不另起编号补额度。

允许物理GPU0/1/2/3，资源修订gpu0123-r1以第23节为准：PT每卡最多1路、全局最多2路；FT/TEST每卡最多4路、全局最多16路。每个作业单卡world1，不混阶段、不改DDP或有效batch；CUDA_VISIBLE_DEVICES仅为该作业分配的一张物理卡。每作业主线程2、每PT有序I/O workers4各线程1、FT workers0。逐卡核对占用；busy卡等待，其他已核实可用卡可派发尚未启动的原矩阵单元，不提前计启动。当前运行中的GPU3作业保持原卡运行，不中断、迁移或重试；由原执行方完成单writer无损交接，禁止另起并行controller。tmux Uni-Poly独立window、worker独立window，保留命令/cwd/PID身份/物理卡映射/日志/真实exit。沿用父48h绝对截止2026-10-27 10:32:33.682164+08:00，不重置；prepare3600秒、PT/FT21600秒、TEST1800秒、CPU600秒且受剩余总时间约束。

顺序：授权/保存现场/最小修复/父消费与新root登记→唯一新增prepare→2PT→70FT及既定validation→一次选择冻结→70正式TEST→完整报告→唯一最终CPU。新prepare必须真实exit0且完整inputs/35头/两modal/5875覆盖及15缺失名单齐备才能启动PT，不做前置suite、smoke、模型探针、compileall、benchmark或全仓回归。

FINAL仍在B_A020/UNI3M/TMOL3M按等权validation macro选择，完全平局依次优先此顺序；冻结70BEST、35基线引用及缺失mask后TEST，不能据TEST改方法/epoch/loss/温度或拼任务赢家。正式门槛仍为未舍入FINAL−0.785044436513689≥0.010，原单位、3位小数展示、ddof1和SOTA对照按原合同。

全部授权实现/实验/报告结束后，只尝试一次：
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout --signal=TERM --kill-after=10s 600s python -m pytest -q -p no:cacheprovider tests/test_ua3m_paperfuse.py
保留原覆盖，加19.4规定的嵌套头首次保存、初始化不覆盖、缓存codec/collate、新旧root、历史失败分离及累计预算必要合成CPU回归，不读大checkpoint或生成真实测试缓存，不skip/xfail。失败不重跑。

按19.5交授权、before、父证据、repair_diff、root映射、父/新增/累计账本、manifest、新完整输入/初始化、2PT/70FT/一次冻结/70TEST、CPU真实exit及report/results。保留原exit1、缓存实际0及旧部分文件，七任务均写结果/SOTA模型/差值/协议限制。任何新异常或覆盖/writer/预算/schema问题停止，不retry/resume、改batch/精度/卡或装依赖。完整交待审查，性能独立判定；故障需返修、外部条件或截止阻断，不自动归档，由Codex独立复审。
~~~

## 21. 本次审查与计划交付

- 本次仅修改CURRENT_PLAN的当前索引并新增第18—21节，保留原科学合同、授权和失败快照；源码、模型、旧产物、预算、日志、PROJECT_HISTORY及用户TODO均未修改。
- 新增prepare、PT、FT、validation、TEST、模型forward、GPU、FP／构象生成、KRR和CPU suite均0；只读检查证据及最终文档diff／格式。
- 父周期需返修，prepare-r1返修计划待授权。必须由用户实际发送第20节或同等明确范围授权，才可修复并执行新增一次prepare及剩余原矩阵；不得从本次审查推定已获重启许可。


## 22. prepare-r1 用户授权与返修执行记录

- 用户本轮明确授权第19节及补充要求，批准最多追加1次prepare及5875行完整预留；累计prepare2次、缓存预留11750行，父失败不退款。其余2 PT／70 FT／70 TEST／一次最终CPU维持原合同。基线D_A020正式macro R²=0.785044436513689，多因素完整方案比较，不归因FP／3D／融合层单独贡献。
- 基于本地dev/cd36ae7280b159f881b483ad5814835ff7b74c4a及既有dirty工作区；本次只修改training、controller、PT worker、tests及本文件。已有其他文件保持；未读取TODO，未远程同步、未做哈希验证。父prepare exit1、实际缓存0行；PT／FT／TEST／CPU启动均0，父现场只读保留。
- 返修结果：`results/ua3m_paperfuse_20261025/r1/repairs/prepare_repair1/`；日志：`logs/ua3m_paperfuse_20261025/r1/prepare_repair1/`。before、父budget／manifest／failure／coverage／incident引用及对象位于authorization，修复增量见final_review/repair_diff.patch，执行root映射见execution_roots.json。
- 最小修复：本周期初始化父目录创建及独占保存防覆盖；显式root传入prepare、PT、FT、TEST及wrapper；父历史与活动队列分离，父／新／累计账本；新旧独立状态块保护，顶部索引更新。module、dataset、科学配置及共享save_checkpoint不改；BEST合法更新保留。
- 执行顺序严格为唯一新增prepare→2 PT→70 FT及既定validation→一次冻结→70正式TEST→完整报告→一次最终CPU。新增合成CPU覆盖仅在最终一次运行；未做前置suite、smoke、compileall或benchmark。
- 仅物理GPU3，PT1、FT／TEST4，阶段不混跑；tmux `Uni-Poly:ua3m-paperfuse-prepare-repair1`。沿用绝对截止2026-10-27 10:32:33.682164+08:00，原单作业超时不变；新失败即停，不retry/resume或扩预算。完整交付为待审查，PROJECT_HISTORY及旧FINAL=D_A005不改。
- 第19节原待授权文字和第21节属于授权前快照；当前明确授权以本节和最新索引为准。性能仍须按冻结FINAL的未舍入正式TEST相对基线提升≥0.010判断；未完成不作性能结论。

<!-- UA3M_PAPERFUSE_REPAIR1_STATUS_START -->
- 2026-10-25T12:25:53.479765+08:00；状态：**需返修**；阶段：交付报告。
- # UA3M-PAPERFUSE r1 实际交付

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

- 账本 `/root/workspace/Uni-Poly-Plus-master/results/ua3m_paperfuse_20261025/r1/repairs/prepare_repair1/budget.json`；日志 `/root/workspace/Uni-Poly-Plus-master/logs/ua3m_paperfuse_20261025/r1/prepare_repair1`；独立复审PENDING，未归档。
<!-- UA3M_PAPERFUSE_REPAIR1_STATUS_END -->

## 23. 用户资源修订：四张GPU可用（gpu0123-r1）

- 授权来源：用户当前原文“请调整为 4 个 GPU 都可以使用！”。这是本周期可用设备范围的明确调整，覆盖旧仅GPU3条款；原实现／实验授权继续有效，不重置预算或截止。本次只修改CURRENT_PLAN，未修改AGENTS、源码、配置、运行账本或模型，未启动、停止或迁移作业。
- 检查快照：2026-10-25 12:23:57（Asia/Shanghai），dev/cd36ae7280b159f881b483ad5814835ff7b74c4a及已有dirty改动。已看AGENTS、SOTA_RESULT、相关diff及当前controller／PT／路径实现。当前返修controller PID739004，UNI3M PT worker PID739294在GPU3运行；持久计数completed_updates=3504，仅为该时刻快照。第22节、旧授权文本及旧状态中的GPU3描述保留历史含义；设备当前是否空闲尚未逐卡查询，不声称已具备四卡实际调度能力。

| 阶段 | 每物理卡并发上限 | 全局并发上限 | 单作业规格 |
|---|---:|---:|---|
| PT | 1 | 2 | world1，micro168×acc6；仅UNI3M／TMOL3M两个原候选 |
| FT／正式TEST | 4 | 16 | world1，batch24；FT和TEST分阶段 |
| prepare／最终CPU | 不占GPU | 原合同 | CUDA_VISIBLE_DEVICES为空 |

两PT可在不同可用卡并行；预训练全部完成后再进入FT，70 FT全部完成并一次冻结后再进入正式TEST。不为占满四卡增加候选、seed、epoch或作业；预算仍为累计prepare2／缓存预留11750、PT2／20000 updates、FT70／4900 epochs、validation4900、正式TEST70、最终CPU1次，所有呈现／update细项保持第19.3节上限。绝对截止仍为2026-10-27 10:32:33.682164+08:00。

### 23.1 DSH落实范围与运行交接

1. 资源调度改动限定在原允许的training、controller、PT worker及tests；模型、loss、数据、m.CONFIG和科学checkpoint schema不改。支持显式physical_gpu参数及GPU0—3白名单；controller、wrapper、PT、FT和TEST子进程传递同一分配，每作业CUDA_VISIBLE_DEVICES只含一张卡，内部cuda:0对应所分配物理卡。
2. 在本周期controller内实现逐卡UUID／compute PID／host-container身份核对及槽位分配，不修改旧共享controller／占用helper。按实际占用从已核实可用卡派发；单卡busy不把其他空闲卡判busy，无法逐卡定位的查询异常则暂停所有新派发。运行中的自有作业计入对应卡槽位；wrapper、CPU worker不重复计GPU任务。物理设备记录为执行元数据，不因FT与TEST分配到不同GPU而误判科学配置或BEST／冻结身份不一致。
3. 保留正在运行的UNI3M及原队列、账本和锁，由原执行方在无新增派发的安全边界完成单writer交接；不终止训练、不resume／重试、不启动第二controller争抢剩余矩阵。修改磁盘源码不会更新已运行controller，必须记录实际接管机制和生效时点；不能无损交接时保持现有作业继续，报告四卡调度尚未落实，不宣称已切换。
4. 新manifest／job spec／consumption／日志记录allowed_gpus=[0,1,2,3]、实际physical_gpu、逻辑映射、每卡槽位和本资源修订；保留生效前的实际GPU3记录，不追溯改写。运行状态块继续由执行方单writer维护；本次利用其等待PT完成期间修改静态条款，保留原状态块及顶部运行状态原文。
5. 复用已有验证证据；确需补充的逐卡门控、槽位、设备参数传递及跨卡FT／TEST配置兼容合成CPU断言，合并到原唯一最终suite，不增加测试轮次、GPU探针或benchmark。异常停止规则、累计失败收费及交付待审查均沿用原合同。

### 23.2 可复制给DSH的资源调整提示词

~~~text
在当前已授权UA3M-PAPERFUSE-REPAIR-20261025-01/prepare-r1内，落实CURRENT_PLAN第23节gpu0123-r1：用户已明确允许物理GPU0、1、2、3。仅调整设备调度，原科学矩阵、预算、数据、batch、精度、world1和绝对截止不变；无需再次询问已批准设备范围。
保留当前GPU3上的UNI3M训练及原controller/账本/队列，由原执行方无损交接单writer；不得杀训练后重启、resume、重复派发、另开controller或重做prepare。不能无损交接则继续保留现有作业并如实报告，不伪称四卡已生效。
按23.1仅修改本周期training/controller/PT worker/tests，显式传递physical_gpu并逐卡核对UUID/PID/槽位；每作业只暴露一张GPU，内部cuda:0、world1。PT每卡1路、全局2路，FT/TEST每卡4路、全局16路，阶段不混跑；busy卡不阻塞其他已核实可用卡。不得把GPU分配变化当作模型或冻结身份变化。
仅派发原剩余单元；已有启动和失败消费全部结转。仍为2 PT/70 FT/一次冻结/70正式TEST/一次最终CPU，无额外seed或测试；截止2026-10-27 10:32:33.682164+08:00不重置。保留旧失败、实际GPU3历史及父产物，记录新分配、生效时点和资源revision，按原交付目录交待审查，由Codex独立复审。
~~~


## 24. 用户要求停止并四卡重新执行：gpu0123-r1

- 授权原文：“停止当前执行，立刻使用 CURRENT_PLAN.md 中的 4 GPU 重新执行！”；说明中断PT收费不退款及重启增量后，用户进一步明确：“重新执行，不需要复用！，”；因此按用户最新指令停止旧执行，并重新prepare、初始化和两个候选step0训练。该明确指令覆盖第23节保留运行中UNI3M／不得重启／无损接管约束，也覆盖旧prepare及PT累计上限；不扩大FT／TEST／CPU预算，不修改科学配置。
- 旧controller PID739004已按start_ticks核对后SIGTERM，真实exit1；清理记录errors=[]，训练及子进程均已退出。UNI3M完成3634、attempted3635 updates，21804微批、3663072实际样本呈现，整次10000 updates及60000微批／10080000呈现预留不退款。旧prepare成功5875/5860/15及35头/2modal保留为历史，不复用为本次完成产物。停止是用户要求，不代表模型数值失败。
- 新执行root `results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/`，日志 `logs/ua3m_paperfuse_20261025/r1/gpu0123_restart1/`；tmux `Uni-Poly:ua3m-paperfuse-gpu0123-restart1`。所有本次prepare／初始化／训练／冻结／正式TEST／账本／报告写新root；不覆盖旧现场、不读旧部分PT权重、不resume。原输入数据／FP／几何／共同step0来源仍按科学合同只读使用，不新生成FP或构象。
- 本次新增prepare最多1次、完整预留5875行；累计prepare上限3次，缓存预留17625行；新实际目标仍5875行（5860有效3D、15缺失），成功后的累计实际物化11750行。PT本次2次各10000 updates；累计PT上限3次／30000 updates／180000微批／30240000样本呈现，其中旧中断占1次完整预留。累计实际与预留分栏；父历史含原失败和用户中断，不进入新活动队列。
- FT累计70次／4900epochs／112000 attempted updates／2630040训练呈现、validation4900次／659960呈现、正式TEST70单元／11750呈现及11720有效3D呈现、最终CPU最多1次均不增加。原科学矩阵、损失、初始化seed、batch／精度／world1、全局validation选择与一次冻结、正式TEST门槛及七任务SOTA口径不变。
- 设备0/1/2/3逐卡UUID／compute PID／身份核对，PT每卡1路全局2路，FT／TEST每卡4路全局16路；每作业CUDA_VISIBLE_DEVICES只含分配卡，内部cuda:0。PT两候选在不同卡并行；PT全部完成才FT，FT全部完成且冻结才TEST。物理卡记录为执行元数据，不属于科学config，FT与TEST可分配不同卡且不改变冻结身份。未知占用只阻塞对应卡，无法定位的查询失败暂停所有新派发。
- 唯一最终CPU增加必要合成四卡UUID／busy／槽位／设备与科学config分离断言，合并原suite，仅在全部实验及报告结束后一次尝试。本轮未运行前置suite、smoke、compileall、benchmark或GPU探针。
- 绝对截止仍为2026-10-27 10:32:33.682164+08:00；单作业超时原值不变。新失败即停，无其他retry/resume／自动补额度；完整交付待审查，不自动归档。旧FINAL=D_A005、PROJECT_HISTORY及原父现场保持。
- 授权、旧停止账本及before见新root/authorization；执行root映射、manifest、累计账本、源diff及报告均写新root。第23节当前资源范围继续有效，其预算及不重启文字以本节用户最新授权为准。

<!-- UA3M_PAPERFUSE_GPU0123_STATUS_START -->
- 2026-10-25T15:17:24.452544+08:00；状态：**待审查**；阶段：交付报告。
- # UA3M-PAPERFUSE r1 实际交付

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

- 账本 `/root/workspace/Uni-Poly-Plus-master/results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/budget.json`；日志 `/root/workspace/Uni-Poly-Plus-master/logs/ua3m_paperfuse_20261025/r1/gpu0123_restart1`；独立复审PENDING，未归档。
<!-- UA3M_PAPERFUSE_GPU0123_STATUS_END -->
