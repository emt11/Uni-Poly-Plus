近两年聚合物/分子属性预测：面向 MIPS 的多尺度 3D 融合深度研究
Executive Summary

当前证据最支持的方向不是继续优化 C5/S4/S45 的聚合权重，而是把 距离尺度本身提升为独立的表示/路由维度。2026 年的 Periodic-TDL 在聚合物上以周期 Vietoris–Rips filtration 做跨尺度层级消息传递，在 9 个任务中取得全部最高 (R^2)；MI-MoE 则用多个距离 cutoff 专家和拓扑 gate 显式选择短/中/长程相互作用。二者与当前 MIPS 的 O8 + periodic GLT + Spatial 结构高度互补。

我最推荐的新主线不是单独 SPG，而是：

[ \boxed{\textbf{O8 Anchor + Periodic Multiscale Geometry Experts + Sparse Routing + Pretrained Fusion}} ]

其中 GLT 保留为 bonded/local expert；Spatial 不再只有一个 C5，而升级成嵌套距离专家；再借鉴 MoleBLEND 做 atom-relation 级预训练对齐近两年聚合物/分子属性预测：面向 MIPS 的多尺度 3D 融合深度研究
Executive Summary

当前证据最支持的方向不是继续优化 C5/S4/S45 的聚合权重，而是把 距离尺度本身提升为独立的表示/路由维度。2026 年的 Periodic-TDL 在聚合物上以周期 Vietoris–Rips filtration 做跨尺度层级消息传递，在 9 个任务中取得全部最高 (R^2)；MI-MoE 则用多个距离 cutoff 专家和拓扑 gate 显式选择短/中/长程相互作用。二者与当前 MIPS 的 O8 + periodic GLT + Spatial 结构高度互补。

我最推荐的新主线不是单独 SPG，而是：

[ \boxed{\textbf{O8 Anchor + Periodic Multiscale Geometry Experts + Sparse Routing + Pretrained Fusion}} ]

其中 GLT 保留为 bonded/local expert；Spatial 不再只有一个 C5，而升级成嵌套距离专家；再借鉴 MoleBLEND 做 atom-relation 级预训练对齐，借鉴 MuMo 做“先建立 O8 主表示、后注入几何”的 progressive fusion。

这一方案是文献驱动的项目新假设，尚未被任何论文直接在 MIPS 上验证；但它比 SPG-only、BP-MCL gate engineering、继续 cutoff/shell 搜索更直接针对你当前已观察到的“Spatial 有信息但几乎没有 downstream leverage”。
检索策略与候选论文

本轮检索时间范围设为 2024-01-01—2026-08-28，并对你指定的 GraphMVP、Unified 2D/3D、PAMNet、MXMNet 等 2020–2023 基础工作回溯。检索入口覆盖 arXiv、Google Scholar 交叉索引、PubMed，以及 ACL Anthology、IEEE Xplore 的分子/材料相关结果；高相关命中实际主要集中于 arXiv、NeurIPS/ICLR/ICML proceedings、PubMed/ACS 和 Nature 系期刊。

核心英文检索式包括 polymer property prediction 3D multiscale geometry、molecular multiscale distance cutoff mixture experts、local nonlocal molecular geometric GNN、2D 3D multimodal molecular pretraining fusion、periodic polymer geometric representation、long-range molecular interaction GNN、atom relation multimodal blending；中文对应为“聚合物 属性预测 多尺度 三维”“分子 局部 非局部 距离融合”“周期聚合物 3D 图神经网络”“二维三维 跨模态 预训练”“长程相互作用 分子图网络”。
Top-20 候选

“兼容性”是我按你当前 O8 topology + periodic GLT bonded + Spatial nonbonded 评估的工程/概念适配度，不是论文指标。
论文	年份/来源	核心方法	代码	MIPS兼容性
Periodic Topological Deep Learning for Polymer Design and Discovery	2026 arXiv	periodic VR filtration + hierarchical simplicial MP，多 cutoff 联合传播	未核实	5.0
Topology-Aware Multiscale Mixture of Experts / MI-MoE	2026 arXiv	多 distance-cutoff experts + persistent-topology gate + sparse routing	未核实	5.0
MIPS	2025 arXiv	infinite-polymer O8/topology + 3D descriptors + cross-attention	是	5.0
MuMo: Structure-Aware Fusion with Progressive Injection	NeurIPS 2025	unified 2D/3D graph、多尺度 global/subgraph、progressive cross-attention injection	是	4.8
MoleBLEND	ICLR 2024	atom-relation 级 blend-then-predict，细粒度 2D/3D 预训练	是	4.8
Neural P³M	NeurIPS 2024	atom short-range + mesh/Fourier long-range	是	4.5
MMPolymer	CIKM 2024	polymer 1D+3D、多任务预训练、coordinate recovery + alignment	是	4.5
Molecular Topological Deep Learning	ACS Nano 2026	多尺度 simplicial complexes / higher-order interactions	未核实	4.5
EMPP	ICLR 2025	equivariant masked-position prediction，物理启发 3D SSL	是	4.3
Uni-Poly	npj Computational Materials 2025	SMILES+2D+3D+fingerprint+text multimodal polymer model	未核实	4.0
MSG-Pre	PLOS ONE 2025	atom/group/conformer 多尺度 + adaptive attention + hierarchical contrastive	未核实	4.0
Uni-Mol2	NeurIPS 2024	atom/graph/geometry two-track Transformer，大规模 3D pretraining	是	3.8
GeoMFormer	ICML 2024	invariant/equivariant dual streams + cross-attention	是	3.8
3DMRL	2025	global 2D/3D contrastive + local relative-geometry prediction	未核实	3.8
SubGDiff	NeurIPS 2024	subgraph-aware 3D diffusion pretraining	是	3.3
PolyFusionAgent / PolyFusion	2026 arXiv	sequence/topology/3D/fingerprint 大规模 shared-space alignment	未核实	3.3
GraphMVP	ICLR 2022，基础	2D/3D correspondence + contrastive/generative pretraining	是	4.2
Unified 2D/3D Pre-training	2022，基础	coordinates/distances 与 atom states 联合预训练和跨模态恢复	有实现线索	4.2
PAMNet	Scientific Reports 2023，基础	physics-aware local/nonlocal multiplex	是	4.8
MXMNet	2020/21，基础	local covalent + global noncovalent multiplex，distance+angle 分工	是	4.7

这里最值得注意的是：**2025 年 Uni-Poly 自己在 Discussion 中明确承认现有 polymer representation 缺少 multi-scale polymer structure，指出单体、分子量分布、chain entanglement、aggregate structure 跨尺度共同影响性质。**这与我们现在从 MIPS 上观察到的问题高度一致。
关键论文深析
Periodic-TDL

这是当前对你最重要的新论文。它构建周期 Vietoris–Rips filtration

[ VR_{\epsilon_1}\subset VR_{\epsilon_2}\subset VR_{\epsilon_3}, ]

不是分别跑几个 cutoff 后简单 concat，而是在 HSMP 中跨 filtration level 层级传播；高阶 simplex 同时捕获 pairwise 和 multi-body interaction。模型在 PI1M 上预训练，并在 9 个 polymer tasks 上做五折评估。

其 RMSE 在 9 个任务中 8 个最低、(R^2) 全部最高；例如 (E_{ea}=0.294)、(E_i=0.406)、EPS=0.535、(X_c=18.22)，均优于文中 MMPolymer 等基线；删除 periodic/hierarchical/pretraining 等组件会退化。

**对 MIPS 的启示：**你之前 S4/S45 失败不能否定 multiscale，因为 Periodic-TDL 的 scale 是“嵌套结构 + 跨尺度 message passing”，而不是 shell pooling。这是完全不同的假设。
MI-MoE

MI-MoE 是目前最直接的“多尺度距离”论文：不同 expert 使用不同 distance cutoff，分别学习 short/mid/long-range，再用 filtration/persistent-homology 得到的 topology descriptor 做 routing。作者强调其是可插入不同 3D backbone 的模块。

论文在分子和聚合物任务都报告提升；其关键 ablation 显示 sparse expert selection 往往优于 dense 混合，说明把所有距离统一平均可能会稀释 scale-specific signal。

**对 MIPS 的启示：**当前 C5/CORA 实质上仍只有一个 Spatial encoder；真正值得测的是“独立 cutoff experts + routing”，而不是继续改变单一 C5 内的聚合函数。
MoleBLEND

MoleBLEND明确指出仅在 molecule-level 对齐 2D/3D 太粗；其 blend-then-predict 把不同模态的 atom relation 先融合成统一 relation matrix，再恢复各自 2D/3D relation。

它没有 shared/private 分解，也不依赖 downstream 才学 fusion：跨模态 interface 本身就是 pretraining task 的一部分。这直接对应你当前 GLT 融合 adapter 没充分参与预训练的问题。

**对 MIPS 的启示：**最自然的 anchor 不是 graph embedding，而是 canonical atom pair：O8 的 SPD/bond relation、GLT bonded distance/angle、Spatial nonbonded distance都可以在 pair/relation level 对齐。
MuMo

MuMo 把 bond length、bond angle 和拓扑统一到 structural graph，并额外构建 global graph 与 geometry-aware substructure graph，再通过 gate 融合 local/global node representations。

更值得借鉴的是 Progressive Injection：早期主 stream 独立形成稳定语义，后期通过双向 cross-attention 注入 structural prior，而不是一开始把所有模态硬混在一起。

**对 MIPS 的启示：**O8 可以真正作为 anchor；GLT+Spatial 先形成 geometry prior，再在 O8 后几层逐层注入，比末端一次 residual 更可能产生 leverage。
PAMNet / MXMNet

PAMNet/MXMNet 的核心不是“4 Å/5 Å”，而是分子力学意义上的：

[ E=E_{\rm local}+E_{\rm nonlocal}. ]

local plex 处理 covalent/local geometry，并使用更丰富 angle 信息；global/nonlocal plex 面向较远 pairwise interaction。

**对 MIPS 的启示：**你的 GLT 与 Spatial 天然已经对应 local/nonlocal 两个 plex。因此 PAMNet-like multiplex 不需要推翻现有工程，只需要把“两个 residual branch”升级成真正 cross-scale interaction。
Neural P³M

Neural P³M 的出发点是普通 geometric GNN 对大体系 long-range interaction 不足；其做法是保留 atom-scale short-range GNN，同时加入 mesh-scale long-range representation并在二者间交换信息。论文报告在 OE62 上集成不同 backbone 后平均改善约 22%，并在 MD22 上改善能量/力预测。

**对 MIPS 的启示：**如果 C5 的核心问题是 receptive field 根本太短，而不是 aggregation，那么 mesh/Fourier expert 比继续扩 5→6 Å 更有理论意义。不过你当前 trimer/周期数据能否支撑真正长程场仍是高风险项。
MMPolymer

MMPolymer 是 polymer-specific 的重要基线：P-SMILES/1D 与 3D structure 联合预训练，同时做 masked token、3D coordinate recovery 和 cross-modal latent alignment。

它证明了 polymer 3D supervision 可以直接进入 pretraining，而不是只在 downstream 加一个 geometry branch。其官方代码公开，仓库标注 GPL-3.0。

**对 MIPS 的启示：**你 SPG-A 的科学动机仍然成立，但 geometry objective 更适合从完整新训练周期开始，而不是 probe-only checkpoint 上 fresh optimizer continuation。
EMPP / Uni-Mol2

EMPP 在 ICLR 2025 提出 equivariant masked-position prediction，用被遮蔽原子位置而非简单 attribute reconstruction 作为几何 SSL，并强调其与 intramolecular potential/force 的物理联系。

Uni-Mol2 则以 two-track Transformer 同时处理 atom、graph、geometry，并扩展到 800M conformations 和最高 1.1B 参数；官方同时发布代码和预训练权重，MIT license。

**对 MIPS 的启示：**若将来重新做 geometry pretraining，masked coordinate/relative-position objective 比“把输入 distance mean 再预测回来”更难形成 shortcut；但完全采用 Uni-Mol2 级规模对你的项目成本过高。
面向 MIPS 的四个 SOTA 候选方案

以下“概率”均是研究优先级先验，不是统计学概率。MIPS 原论文的 3D 模态是 monomer 3D descriptors 经线性投影后通过 cross-attention 注入 atom topology；论文 ablation 显示加入 3D 后八项均优于 2D+BE。 你的 GLT 已把这一步推进到 atom-aligned bonded geometry，因此下一跳应集中在 多尺度 interaction 与预训练 fusion。
方案	核心	优先级/成本	first smoke
A. Periodic Multiscale Sparse Experts（首选）	GLT=local expert；新增多个嵌套 nonbonded cutoff experts；top-k sparse gate；各 expert 有独立 adapter	高 / 中高	Base-5k 权重、冻结 O8/GLT；500 steps，只训练新 experts/router/adapters；固定 EI fold + 一组预训练 relation probe；要求 gate 非塌缩、same-checkpoint expert-zeroing 改变预测、无 Base drift
B. RelationBlend Pretraining	O8 SPD/bond、GLT bonded geometry、Spatial distance 在 canonical atom-pair level blend/reconstruct	高 / 中	新预训练周期或 frozen-backbone adapter smoke；500 steps；检查 masked relation recovery、2D→3D/3D→2D probe、adapter grad、fusion ablation
C. Progressive Multiplex Injection	GLT+Spatial 先组成 geometry prior；在 O8 后半层通过 atom-wise cross-attention逐层注入	高 / 中高	Base-5k frozen lower O8，仅解冻末 2 层+fusion；500 steps；比较 residual-on/off、attention entropy、O8 representation drift
D. Neural-P³M Long-range Expert	GLT 短程 + atom/mesh Fourier long-range + O8 anchor	中 / 高	先只做 long-range auxiliary probe；500 steps；验证 mesh branch 对 (>5) Å pair perturbation敏感且对 local bond扰动不冗余
首选方案的增益机理

A 的核心区别是把你过去的

[ C5=\mathrm{Mean}{m_{ij}:d_{ij}<5\AA} ]

改为：
[ h_i^{3D}

\sum_{k\in TopK} g_{ik}, A_k!\left(E_k(G_{R_k})_i\right), ]

其中 (R_k) 是嵌套 cutoff graph，而不是互斥 shell。每个尺度拥有自己的 message-passing trajectory，因此 3 Å 信息不会先与 7 Å 信息平均掉；gate 再根据 O8 topology、尺度 degree/filtration statistics 选择专家。这最接近 MI-MoE 与 Periodic-TDL 的共同有效因素。

主要风险是计算量、长 cutoff 噪声和 expert collapse。特别是你的 trimer/periodic construction 若无法物理可信地提供 (>5) Å environment，则方案 D/A 的远程部分会变成伪长程，因此 prestart 必须先做 periodic-distance coverage audit。

B解决的不是距离范围，而是融合接口：MoleBLEND 提示 relation-level alignment 比 graph-level alignment更细。 风险是过强 reconstruction 再次让几何分支学捷径，因此必须采用 corrupted input / clean target、cross-modal recovery，并避免重复 SPG-A 的 probe-only continuation。

C解决你目前最明显的“leverage”问题：MuMo 的 delayed/progressive injection 表明，主模态可以先独立形成表示，再逐层接受结构 prior。 风险是 cross-attention 参数多、在小 polymer tasks 上过拟合，所以建议 adapter bottleneck + scalar residual scale，而不是 Full BP-MCL 大 router。

D最激进。Neural P³M 给出了 short/long-range 在 atom/mesh 两种尺度上并行计算的强证据。 但将它迁移到 polymer repeat-unit/trimer representation 是未证实假设，只有在 A 显示 (>5) Å expert 确有独立 leverage 后才值得投入。
推荐架构与预训练路径

综合 Periodic-TDL、MI-MoE、PAMNet、MoleBLEND 与 MuMo，我建议把下一代模型称为暂定的 MIPS-PM3D（Periodic Multiscale 3D）。这是项目新方案，不是任何单篇论文的复现。

Polymer / canonical atoms

O8 topology anchor

Periodic GLT
bonded / local 3D

Spatial Expert R1
short nonbonded

Spatial Expert R2
mid-range

Spatial Expert R3
long-range

Local Adapter

Scale Adapter

Scale Adapter

Scale Adapter

Sparse topology-aware router
Top-k

Multiscale Geometry Prior

Progressive Injection

Canonical fused atom states

Pooling + MD200

Graph adapter / property head

关键是 O8 不参与 softmax 竞争；它仍是 anchor：
[ h_i^{t+1}

h_i^{O8,t} + \gamma_t, CrossAttn \left( h_i^{O8,t}, h_i^{multi3D} \right). ]

而不是：

[ \alpha_OO8+\alpha_GGLT+\alpha_SSpatial. ]

第二张图显示为什么 fusion adapter 必须进入预训练，而不是像当前 GLT-v2 一样等到 downstream 才弱学习：

O8 relations
bond / SPD / topology

Relation Blend

GLT
bond distance / angle

Multiscale Spatial
nested distances

Multiscale geometry encoders

Pretrained scale adapters

Sparse router + geometry prior

Cross-modal relation recovery

Pretraining losses
masked relation + geometry + topology

Fine-tuning

Pretrained adapter + small gate

Residual / progressive injection into O8

这同时解决三件事：multi-scale distance 不再等于 shell engineering；geometry adapter 在 pretraining 中获得梯度；O8 anchor 不被随机 3D branch 覆盖。
与既有方案对比及复现资源
架构路线对比

“(R^2) 改进概率”指相对于你当前 matched MIPS baseline 出现可重复正 delta的主观先验，不代表达到论文 SOTA 的统计概率。
路线	真多尺度	预训练fusion接口	O8 anchor	新预训练	可解释性	复杂度	正 (R^2) delta 先验
Periodic Multiscale Sparse Experts	是	是	是	最终需要	高	中高	~65%
RelationBlend	可结合	是	是	是	中高	中	~60%
Progressive Multiplex Injection	是	是	是	推荐	中	中高	~55%
Neural-P³M-like	是，长程	是	是	是	高	高	~40%
SPG	否，本质 shared/private	是	是	是	高	中	~35%
BP-MCL-Lite	否	否/弱	是	否	高	低	~20%
Full BP-MCL	本身否	可选	是	可选	中	高	~30%
PAMNet-like multiplex	是	原版联合训练	可保留	推荐	高	中高	~50%

你已有的 S4/S45、PNA、Mean+STD、shell attention、CORA 全部没有形成稳定优势，因此我会把“再改单一 C5 aggregator”的先验降到很低；这是基于你提供的项目实验，而非外部文献。
可复现资源

MIPS 原论文公开代码；MMPolymer 官方仓库公开且标 GPL-3.0；MoleBLEND 的公开实现包含 MIT license；Uni-Mol/Uni-Mol2 提供代码、预训练权重并使用 MIT license；Neural P³M、PAMNet 也有公开实现。
资源	用途	状态/许可
PL1M / PI1M	MIPS/Periodic-TDL 百万级 polymer pretraining	论文公开使用；具体数据许可应在下载前再次核验
MIPS 8 tasks	与当前项目最直接 benchmark	原论文公开数据协议/代码
QM9	小分子 quantum-property / 3D geometry	广泛公开；适合先验证 multiscale encoder
PCQM4M(v2)	大规模 quantum property	OGB benchmark；适合 scale-up，而非第一 smoke
MD17 / MD22	force/energy、长程几何测试	公开科研 benchmark；Neural P³M 使用 MD22 验证长程能力
GEOM	多 conformer 3D pretraining	公开；SubGDiff 等采用，可用于 geometry SSL
Uni-Mol2 weights	强 3D representation 对照	公开，MIT

值得特别提醒：Periodic-TDL 虽然目前对 MIPS 最相关，但其 3D 坐标同样基于 repeat-unit pSMILES 生成并经 UFF 优化，而且论文自己指出它尚不能显式编码 distant repeat units。 因此它证明的是**“periodic hierarchical multiscale representation 很值得研究”**，并没有证明当前 repeat-unit geometry 已经充分描述真实长链 packing。
最终推荐与下一步执行

首选：Periodic Multiscale Sparse Experts + pretrained relation-level adapter。

也就是优先吸收：

[ \boxed{ \text{Periodic-TDL 的层级尺度} + \text{MI-MoE 的独立专家/稀疏路由} + \text{MoleBLEND 的关系级预训练} } ]

然后才吸收 MuMo 的 progressive injection。这个组合在文献中没有被作为一个模型验证过，因此必须明确标注：

    假设 / 未证实：预计比当前 C5 residual 更适合 MIPS，但尚无直接 MIPS 实验证据。

**备选：PAMNet-like local/nonlocal multiplex + MuMo progressive injection。**它少一些多-cutoff复杂性，更容易从现有 GLT + C5 改造，是工程风险较低的 Plan B。PAMNet 的 local/nonlocal 分解有明确的物理先验，而 MuMo 为“保持主 stream、延迟注入结构 prior”提供了近期证据。

接下来的三个具体行动应是：先做 periodic-distance coverage audit，确认当前 trimer/PBC 是否真的支持 5 Å 以上的可信 nonbonded relation；随后实现只增加 multiscale experts/router/adapters、冻结原 Base 的 500-step mechanism smoke；若 same-checkpoint scale intervention 确认有 leverage，再启动全新的 joint-pretraining cycle，避免重用已经出现 pathology 的 5k probe-only weights + fresh Adam/scheduler continuation。
500-step CODEX 提示词草案

text

任务：MIPS-PM3D-A0 — Frozen-Base Periodic Multiscale Expert 500-step Smoke

目标：
验证“独立距离尺度专家 + sparse routing”是否比现有单 C5 branch
产生真实、atom-specific、same-checkpoint prediction leverage。
这不是正式性能实验。

BASE：
- 使用当前正式 MTS-GLT-v2-Base-5k model weights。
- O8、GLT、MD200 全部冻结，禁止 weight drift。
- 不做 probe-only backbone continuation。
- 保留当前 canonical atom identity contract。

PRESTART：
1. 审计 periodic/trimer pair-distance coverage：
   histogram 到至少 10 Å；
   按 SPD、PBC shift、重复单元来源统计。
2. 若 >5 Å pair 无物理/周期可信性，STOP = RANGE_INVALID。
3. 固定 train/val hashes 与 Base predictions。

NEW BRANCH：
- 保留 GLT = bonded/local expert。
- 非键接要求 SPD>=4。
- 建立 nested experts，而非互斥 shell：
  E4: d<=4 Å
  E5: d<=5 Å
  E7: d<=7 Å
  （只有 coverage audit 通过才允许 E7）
- 每个 expert 独立 message aggregation；
  不允许先将三尺度 message 平均。
- shared continuous RBF distance encoding；
  scale-specific adapter -> 512 O8 residual space。
- router 输入仅：
  frozen O8 atom state +
  per-scale valid/count statistics。
- top-k=2 sparse routing。
- O8 不进入 softmax competition。

FUSION：
h_new = h_Base + gamma * sum_k g_ik A_k(h_ik)
gamma 小初始化但不得阻断 expert gradient。
geometry-invalid 精确回退 Base。

TRAIN：
- 只训练 new experts / adapters / router / gamma。
- 500 optimizer steps。
- 使用一个固定代表性 downstream smoke split；
  不做 8x5，不宣称性能提升。
- C5-static 参数量匹配 control 必须同时构造。

LOG：
step 0/20/50/100/200/500：
loss、LR、expert grad、adapter grad、gamma、
gate entropy、expert usage、residual/base norm ratio。

SAME-CHECKPOINT INTERVENTIONS：
A. gamma -> 0
B. force E5 only
C. remove E7
D. gate -> graph mean
E. permute scale assignment while preserving messages

SUCCESS：
- 无 NaN/OOM/Base drift；
- >=2 experts 获得稳定非零 usage；
- router 不 collapse 到单 expert；
- residual/base norm ratio >= 0.01；
- normal vs gamma=0 prediction delta 明显非零；
- normal vs E5-only delta 明显非零；
- 至少一个 outer-scale removal 对 prediction/validation metric 有一致影响。
不以单次 R² 提升作为唯一成功条件。

STOP：
500 steps 后立即停止。
不得启动 formal 8x5、SPG、Full BP-MCL 或新 cutoff sweep。
输出 RETURN_PACKET，给出 SCALE_LEVER_PRESENT /
SCALE_LEVER_WEAK / NO_SCALE_LEVER / RANGE_INVALID。

最关键的研究路线变化可以浓缩成一句话：

[ \boxed{ \textbf{不要再问“5 Å 内该怎么平均”，而要问“不同物理距离尺度是否应该拥有独立表示，并由模型选择何时使用”。} } ]

目前近两年的文献中，Periodic-TDL、MI-MoE、MoleBLEND、MuMo 四条证据链恰好分别对应你的四个瓶颈：周期多尺度、尺度选择、细粒度预训练对齐、以及 downstream fusion leverage。它们的组合，是我认为当前最有希望把 MIPS 从“强 O8 + 弱几何 residual”升级成真正 2D topology + multiscale 3D cooperative model 的方向。，借鉴 MuMo 做“先建立 O8 主表示、后注入几何”的 progressive fusion。

这一方案是文献驱动的项目新假设，尚未被任何论文直接在 MIPS 上验证；但它比 SPG-only、BP-MCL gate engineering、继续 cutoff/shell 搜索更直接针对你当前已观察到的“Spatial 有信息但几乎没有 downstream leverage”。
检索策略与候选论文

本轮检索时间范围设为 2024-01-01—2026-08-28，并对你指定的 GraphMVP、Unified 2D/3D、PAMNet、MXMNet 等 2020–2023 基础工作回溯。检索入口覆盖 arXiv、Google Scholar 交叉索引、PubMed，以及 ACL Anthology、IEEE Xplore 的分子/材料相关结果；高相关命中实际主要集中于 arXiv、NeurIPS/ICLR/ICML proceedings、PubMed/ACS 和 Nature 系期刊。

核心英文检索式包括 polymer property prediction 3D multiscale geometry、molecular multiscale distance cutoff mixture experts、local nonlocal molecular geometric GNN、2D 3D multimodal molecular pretraining fusion、periodic polymer geometric representation、long-range molecular interaction GNN、atom relation multimodal blending；中文对应为“聚合物 属性预测 多尺度 三维”“分子 局部 非局部 距离融合”“周期聚合物 3D 图神经网络”“二维三维 跨模态 预训练”“长程相互作用 分子图网络”。
Top-20 候选

“兼容性”是我按你当前 O8 topology + periodic GLT bonded + Spatial nonbonded 评估的工程/概念适配度，不是论文指标。
论文	年份/来源	核心方法	代码	MIPS兼容性
Periodic Topological Deep Learning for Polymer Design and Discovery	2026 arXiv	periodic VR filtration + hierarchical simplicial MP，多 cutoff 联合传播	未核实	5.0
Topology-Aware Multiscale Mixture of Experts / MI-MoE	2026 arXiv	多 distance-cutoff experts + persistent-topology gate + sparse routing	未核实	5.0
MIPS	2025 arXiv	infinite-polymer O8/topology + 3D descriptors + cross-attention	是	5.0
MuMo: Structure-Aware Fusion with Progressive Injection	NeurIPS 2025	unified 2D/3D graph、多尺度 global/subgraph、progressive cross-attention injection	是	4.8
MoleBLEND	ICLR 2024	atom-relation 级 blend-then-predict，细粒度 2D/3D 预训练	是	4.8
Neural P³M	NeurIPS 2024	atom short-range + mesh/Fourier long-range	是	4.5
MMPolymer	CIKM 2024	polymer 1D+3D、多任务预训练、coordinate recovery + alignment	是	4.5
Molecular Topological Deep Learning	ACS Nano 2026	多尺度 simplicial complexes / higher-order interactions	未核实	4.5
EMPP	ICLR 2025	equivariant masked-position prediction，物理启发 3D SSL	是	4.3
Uni-Poly	npj Computational Materials 2025	SMILES+2D+3D+fingerprint+text multimodal polymer model	未核实	4.0
MSG-Pre	PLOS ONE 2025	atom/group/conformer 多尺度 + adaptive attention + hierarchical contrastive	未核实	4.0
Uni-Mol2	NeurIPS 2024	atom/graph/geometry two-track Transformer，大规模 3D pretraining	是	3.8
GeoMFormer	ICML 2024	invariant/equivariant dual streams + cross-attention	是	3.8
3DMRL	2025	global 2D/3D contrastive + local relative-geometry prediction	未核实	3.8
SubGDiff	NeurIPS 2024	subgraph-aware 3D diffusion pretraining	是	3.3
PolyFusionAgent / PolyFusion	2026 arXiv	sequence/topology/3D/fingerprint 大规模 shared-space alignment	未核实	3.3
GraphMVP	ICLR 2022，基础	2D/3D correspondence + contrastive/generative pretraining	是	4.2
Unified 2D/3D Pre-training	2022，基础	coordinates/distances 与 atom states 联合预训练和跨模态恢复	有实现线索	4.2
PAMNet	Scientific Reports 2023，基础	physics-aware local/nonlocal multiplex	是	4.8
MXMNet	2020/21，基础	local covalent + global noncovalent multiplex，distance+angle 分工	是	4.7

这里最值得注意的是：**2025 年 Uni-Poly 自己在 Discussion 中明确承认现有 polymer representation 缺少 multi-scale polymer structure，指出单体、分子量分布、chain entanglement、aggregate structure 跨尺度共同影响性质。**这与我们现在从 MIPS 上观察到的问题高度一致。
关键论文深析
Periodic-TDL

这是当前对你最重要的新论文。它构建周期 Vietoris–Rips filtration

[ VR_{\epsilon_1}\subset VR_{\epsilon_2}\subset VR_{\epsilon_3}, ]

不是分别跑几个 cutoff 后简单 concat，而是在 HSMP 中跨 filtration level 层级传播；高阶 simplex 同时捕获 pairwise 和 multi-body interaction。模型在 PI1M 上预训练，并在 9 个 polymer tasks 上做五折评估。

其 RMSE 在 9 个任务中 8 个最低、(R^2) 全部最高；例如 (E_{ea}=0.294)、(E_i=0.406)、EPS=0.535、(X_c=18.22)，均优于文中 MMPolymer 等基线；删除 periodic/hierarchical/pretraining 等组件会退化。

**对 MIPS 的启示：**你之前 S4/S45 失败不能否定 multiscale，因为 Periodic-TDL 的 scale 是“嵌套结构 + 跨尺度 message passing”，而不是 shell pooling。这是完全不同的假设。
MI-MoE

MI-MoE 是目前最直接的“多尺度距离”论文：不同 expert 使用不同 distance cutoff，分别学习 short/mid/long-range，再用 filtration/persistent-homology 得到的 topology descriptor 做 routing。作者强调其是可插入不同 3D backbone 的模块。

论文在分子和聚合物任务都报告提升；其关键 ablation 显示 sparse expert selection 往往优于 dense 混合，说明把所有距离统一平均可能会稀释 scale-specific signal。

**对 MIPS 的启示：**当前 C5/CORA 实质上仍只有一个 Spatial encoder；真正值得测的是“独立 cutoff experts + routing”，而不是继续改变单一 C5 内的聚合函数。
MoleBLEND

MoleBLEND明确指出仅在 molecule-level 对齐 2D/3D 太粗；其 blend-then-predict 把不同模态的 atom relation 先融合成统一 relation matrix，再恢复各自 2D/3D relation。

它没有 shared/private 分解，也不依赖 downstream 才学 fusion：跨模态 interface 本身就是 pretraining task 的一部分。这直接对应你当前 GLT 融合 adapter 没充分参与预训练的问题。

**对 MIPS 的启示：**最自然的 anchor 不是 graph embedding，而是 canonical atom pair：O8 的 SPD/bond relation、GLT bonded distance/angle、Spatial nonbonded distance都可以在 pair/relation level 对齐。
MuMo

MuMo 把 bond length、bond angle 和拓扑统一到 structural graph，并额外构建 global graph 与 geometry-aware substructure graph，再通过 gate 融合 local/global node representations。

更值得借鉴的是 Progressive Injection：早期主 stream 独立形成稳定语义，后期通过双向 cross-attention 注入 structural prior，而不是一开始把所有模态硬混在一起。

**对 MIPS 的启示：**O8 可以真正作为 anchor；GLT+Spatial 先形成 geometry prior，再在 O8 后几层逐层注入，比末端一次 residual 更可能产生 leverage。
PAMNet / MXMNet

PAMNet/MXMNet 的核心不是“4 Å/5 Å”，而是分子力学意义上的：

[ E=E_{\rm local}+E_{\rm nonlocal}. ]

local plex 处理 covalent/local geometry，并使用更丰富 angle 信息；global/nonlocal plex 面向较远 pairwise interaction。

**对 MIPS 的启示：**你的 GLT 与 Spatial 天然已经对应 local/nonlocal 两个 plex。因此 PAMNet-like multiplex 不需要推翻现有工程，只需要把“两个 residual branch”升级成真正 cross-scale interaction。
Neural P³M

Neural P³M 的出发点是普通 geometric GNN 对大体系 long-range interaction 不足；其做法是保留 atom-scale short-range GNN，同时加入 mesh-scale long-range representation并在二者间交换信息。论文报告在 OE62 上集成不同 backbone 后平均改善约 22%，并在 MD22 上改善能量/力预测。

**对 MIPS 的启示：**如果 C5 的核心问题是 receptive field 根本太短，而不是 aggregation，那么 mesh/Fourier expert 比继续扩 5→6 Å 更有理论意义。不过你当前 trimer/周期数据能否支撑真正长程场仍是高风险项。
MMPolymer

MMPolymer 是 polymer-specific 的重要基线：P-SMILES/1D 与 3D structure 联合预训练，同时做 masked token、3D coordinate recovery 和 cross-modal latent alignment。

它证明了 polymer 3D supervision 可以直接进入 pretraining，而不是只在 downstream 加一个 geometry branch。其官方代码公开，仓库标注 GPL-3.0。

**对 MIPS 的启示：**你 SPG-A 的科学动机仍然成立，但 geometry objective 更适合从完整新训练周期开始，而不是 probe-only checkpoint 上 fresh optimizer continuation。
EMPP / Uni-Mol2

EMPP 在 ICLR 2025 提出 equivariant masked-position prediction，用被遮蔽原子位置而非简单 attribute reconstruction 作为几何 SSL，并强调其与 intramolecular potential/force 的物理联系。

Uni-Mol2 则以 two-track Transformer 同时处理 atom、graph、geometry，并扩展到 800M conformations 和最高 1.1B 参数；官方同时发布代码和预训练权重，MIT license。

**对 MIPS 的启示：**若将来重新做 geometry pretraining，masked coordinate/relative-position objective 比“把输入 distance mean 再预测回来”更难形成 shortcut；但完全采用 Uni-Mol2 级规模对你的项目成本过高。
面向 MIPS 的四个 SOTA 候选方案

以下“概率”均是研究优先级先验，不是统计学概率。MIPS 原论文的 3D 模态是 monomer 3D descriptors 经线性投影后通过 cross-attention 注入 atom topology；论文 ablation 显示加入 3D 后八项均优于 2D+BE。 你的 GLT 已把这一步推进到 atom-aligned bonded geometry，因此下一跳应集中在 多尺度 interaction 与预训练 fusion。
方案	核心	优先级/成本	first smoke
A. Periodic Multiscale Sparse Experts（首选）	GLT=local expert；新增多个嵌套 nonbonded cutoff experts；top-k sparse gate；各 expert 有独立 adapter	高 / 中高	Base-5k 权重、冻结 O8/GLT；500 steps，只训练新 experts/router/adapters；固定 EI fold + 一组预训练 relation probe；要求 gate 非塌缩、same-checkpoint expert-zeroing 改变预测、无 Base drift
B. RelationBlend Pretraining	O8 SPD/bond、GLT bonded geometry、Spatial distance 在 canonical atom-pair level blend/reconstruct	高 / 中	新预训练周期或 frozen-backbone adapter smoke；500 steps；检查 masked relation recovery、2D→3D/3D→2D probe、adapter grad、fusion ablation
C. Progressive Multiplex Injection	GLT+Spatial 先组成 geometry prior；在 O8 后半层通过 atom-wise cross-attention逐层注入	高 / 中高	Base-5k frozen lower O8，仅解冻末 2 层+fusion；500 steps；比较 residual-on/off、attention entropy、O8 representation drift
D. Neural-P³M Long-range Expert	GLT 短程 + atom/mesh Fourier long-range + O8 anchor	中 / 高	先只做 long-range auxiliary probe；500 steps；验证 mesh branch 对 (>5) Å pair perturbation敏感且对 local bond扰动不冗余
首选方案的增益机理

A 的核心区别是把你过去的

[ C5=\mathrm{Mean}{m_{ij}:d_{ij}<5\AA} ]

改为：
[ h_i^{3D}

\sum_{k\in TopK} g_{ik}, A_k!\left(E_k(G_{R_k})_i\right), ]

其中 (R_k) 是嵌套 cutoff graph，而不是互斥 shell。每个尺度拥有自己的 message-passing trajectory，因此 3 Å 信息不会先与 7 Å 信息平均掉；gate 再根据 O8 topology、尺度 degree/filtration statistics 选择专家。这最接近 MI-MoE 与 Periodic-TDL 的共同有效因素。

主要风险是计算量、长 cutoff 噪声和 expert collapse。特别是你的 trimer/periodic construction 若无法物理可信地提供 (>5) Å environment，则方案 D/A 的远程部分会变成伪长程，因此 prestart 必须先做 periodic-distance coverage audit。

B解决的不是距离范围，而是融合接口：MoleBLEND 提示 relation-level alignment 比 graph-level alignment更细。 风险是过强 reconstruction 再次让几何分支学捷径，因此必须采用 corrupted input / clean target、cross-modal recovery，并避免重复 SPG-A 的 probe-only continuation。

C解决你目前最明显的“leverage”问题：MuMo 的 delayed/progressive injection 表明，主模态可以先独立形成表示，再逐层接受结构 prior。 风险是 cross-attention 参数多、在小 polymer tasks 上过拟合，所以建议 adapter bottleneck + scalar residual scale，而不是 Full BP-MCL 大 router。

D最激进。Neural P³M 给出了 short/long-range 在 atom/mesh 两种尺度上并行计算的强证据。 但将它迁移到 polymer repeat-unit/trimer representation 是未证实假设，只有在 A 显示 (>5) Å expert 确有独立 leverage 后才值得投入。
推荐架构与预训练路径

综合 Periodic-TDL、MI-MoE、PAMNet、MoleBLEND 与 MuMo，我建议把下一代模型称为暂定的 MIPS-PM3D（Periodic Multiscale 3D）。这是项目新方案，不是任何单篇论文的复现。

Polymer / canonical atoms

O8 topology anchor

Periodic GLT
bonded / local 3D

Spatial Expert R1
short nonbonded

Spatial Expert R2
mid-range

Spatial Expert R3
long-range

Local Adapter

Scale Adapter

Scale Adapter

Scale Adapter

Sparse topology-aware router
Top-k

Multiscale Geometry Prior

Progressive Injection

Canonical fused atom states

Pooling + MD200

Graph adapter / property head

关键是 O8 不参与 softmax 竞争；它仍是 anchor：
[ h_i^{t+1}

h_i^{O8,t} + \gamma_t, CrossAttn \left( h_i^{O8,t}, h_i^{multi3D} \right). ]

而不是：

[ \alpha_OO8+\alpha_GGLT+\alpha_SSpatial. ]

第二张图显示为什么 fusion adapter 必须进入预训练，而不是像当前 GLT-v2 一样等到 downstream 才弱学习：

O8 relations
bond / SPD / topology

Relation Blend

GLT
bond distance / angle

Multiscale Spatial
nested distances

Multiscale geometry encoders

Pretrained scale adapters

Sparse router + geometry prior

Cross-modal relation recovery

Pretraining losses
masked relation + geometry + topology

Fine-tuning

Pretrained adapter + small gate

Residual / progressive injection into O8

这同时解决三件事：multi-scale distance 不再等于 shell engineering；geometry adapter 在 pretraining 中获得梯度；O8 anchor 不被随机 3D branch 覆盖。
与既有方案对比及复现资源
架构路线对比

“(R^2) 改进概率”指相对于你当前 matched MIPS baseline 出现可重复正 delta的主观先验，不代表达到论文 SOTA 的统计概率。
路线	真多尺度	预训练fusion接口	O8 anchor	新预训练	可解释性	复杂度	正 (R^2) delta 先验
Periodic Multiscale Sparse Experts	是	是	是	最终需要	高	中高	~65%
RelationBlend	可结合	是	是	是	中高	中	~60%
Progressive Multiplex Injection	是	是	是	推荐	中	中高	~55%
Neural-P³M-like	是，长程	是	是	是	高	高	~40%
SPG	否，本质 shared/private	是	是	是	高	中	~35%
BP-MCL-Lite	否	否/弱	是	否	高	低	~20%
Full BP-MCL	本身否	可选	是	可选	中	高	~30%
PAMNet-like multiplex	是	原版联合训练	可保留	推荐	高	中高	~50%

你已有的 S4/S45、PNA、Mean+STD、shell attention、CORA 全部没有形成稳定优势，因此我会把“再改单一 C5 aggregator”的先验降到很低；这是基于你提供的项目实验，而非外部文献。
可复现资源

MIPS 原论文公开代码；MMPolymer 官方仓库公开且标 GPL-3.0；MoleBLEND 的公开实现包含 MIT license；Uni-Mol/Uni-Mol2 提供代码、预训练权重并使用 MIT license；Neural P³M、PAMNet 也有公开实现。
资源	用途	状态/许可
PL1M / PI1M	MIPS/Periodic-TDL 百万级 polymer pretraining	论文公开使用；具体数据许可应在下载前再次核验
MIPS 8 tasks	与当前项目最直接 benchmark	原论文公开数据协议/代码
QM9	小分子 quantum-property / 3D geometry	广泛公开；适合先验证 multiscale encoder
PCQM4M(v2)	大规模 quantum property	OGB benchmark；适合 scale-up，而非第一 smoke
MD17 / MD22	force/energy、长程几何测试	公开科研 benchmark；Neural P³M 使用 MD22 验证长程能力
GEOM	多 conformer 3D pretraining	公开；SubGDiff 等采用，可用于 geometry SSL
Uni-Mol2 weights	强 3D representation 对照	公开，MIT

值得特别提醒：Periodic-TDL 虽然目前对 MIPS 最相关，但其 3D 坐标同样基于 repeat-unit pSMILES 生成并经 UFF 优化，而且论文自己指出它尚不能显式编码 distant repeat units。 因此它证明的是**“periodic hierarchical multiscale representation 很值得研究”**，并没有证明当前 repeat-unit geometry 已经充分描述真实长链 packing。
最终推荐与下一步执行

首选：Periodic Multiscale Sparse Experts + pretrained relation-level adapter。

也就是优先吸收：

[ \boxed{ \text{Periodic-TDL 的层级尺度} + \text{MI-MoE 的独立专家/稀疏路由} + \text{MoleBLEND 的关系级预训练} } ]

然后才吸收 MuMo 的 progressive injection。这个组合在文献中没有被作为一个模型验证过，因此必须明确标注：

    假设 / 未证实：预计比当前 C5 residual 更适合 MIPS，但尚无直接 MIPS 实验证据。

**备选：PAMNet-like local/nonlocal multiplex + MuMo progressive injection。**它少一些多-cutoff复杂性，更容易从现有 GLT + C5 改造，是工程风险较低的 Plan B。PAMNet 的 local/nonlocal 分解有明确的物理先验，而 MuMo 为“保持主 stream、延迟注入结构 prior”提供了近期证据。

接下来的三个具体行动应是：先做 periodic-distance coverage audit，确认当前 trimer/PBC 是否真的支持 5 Å 以上的可信 nonbonded relation；随后实现只增加 multiscale experts/router/adapters、冻结原 Base 的 500-step mechanism smoke；若 same-checkpoint scale intervention 确认有 leverage，再启动全新的 joint-pretraining cycle，避免重用已经出现 pathology 的 5k probe-only weights + fresh Adam/scheduler continuation。
500-step CODEX 提示词草案

text

任务：MIPS-PM3D-A0 — Frozen-Base Periodic Multiscale Expert 500-step Smoke

目标：
验证“独立距离尺度专家 + sparse routing”是否比现有单 C5 branch
产生真实、atom-specific、same-checkpoint prediction leverage。
这不是正式性能实验。

BASE：
- 使用当前正式 MTS-GLT-v2-Base-5k model weights。
- O8、GLT、MD200 全部冻结，禁止 weight drift。
- 不做 probe-only backbone continuation。
- 保留当前 canonical atom identity contract。

PRESTART：
1. 审计 periodic/trimer pair-distance coverage：
   histogram 到至少 10 Å；
   按 SPD、PBC shift、重复单元来源统计。
2. 若 >5 Å pair 无物理/周期可信性，STOP = RANGE_INVALID。
3. 固定 train/val hashes 与 Base predictions。

NEW BRANCH：
- 保留 GLT = bonded/local expert。
- 非键接要求 SPD>=4。
- 建立 nested experts，而非互斥 shell：
  E4: d<=4 Å
  E5: d<=5 Å
  E7: d<=7 Å
  （只有 coverage audit 通过才允许 E7）
- 每个 expert 独立 message aggregation；
  不允许先将三尺度 message 平均。
- shared continuous RBF distance encoding；
  scale-specific adapter -> 512 O8 residual space。
- router 输入仅：
  frozen O8 atom state +
  per-scale valid/count statistics。
- top-k=2 sparse routing。
- O8 不进入 softmax competition。

FUSION：
h_new = h_Base + gamma * sum_k g_ik A_k(h_ik)
gamma 小初始化但不得阻断 expert gradient。
geometry-invalid 精确回退 Base。

TRAIN：
- 只训练 new experts / adapters / router / gamma。
- 500 optimizer steps。
- 使用一个固定代表性 downstream smoke split；
  不做 8x5，不宣称性能提升。
- C5-static 参数量匹配 control 必须同时构造。

LOG：
step 0/20/50/100/200/500：
loss、LR、expert grad、adapter grad、gamma、
gate entropy、expert usage、residual/base norm ratio。

SAME-CHECKPOINT INTERVENTIONS：
A. gamma -> 0
B. force E5 only
C. remove E7
D. gate -> graph mean
E. permute scale assignment while preserving messages

SUCCESS：
- 无 NaN/OOM/Base drift；
- >=2 experts 获得稳定非零 usage；
- router 不 collapse 到单 expert；
- residual/base norm ratio >= 0.01；
- normal vs gamma=0 prediction delta 明显非零；
- normal vs E5-only delta 明显非零；
- 至少一个 outer-scale removal 对 prediction/validation metric 有一致影响。
不以单次 R² 提升作为唯一成功条件。

STOP：
500 steps 后立即停止。
不得启动 formal 8x5、SPG、Full BP-MCL 或新 cutoff sweep。
输出 RETURN_PACKET，给出 SCALE_LEVER_PRESENT /
SCALE_LEVER_WEAK / NO_SCALE_LEVER / RANGE_INVALID。

最关键的研究路线变化可以浓缩成一句话：

[ \boxed{ \textbf{不要再问“5 Å 内该怎么平均”，而要问“不同物理距离尺度是否应该拥有独立表示，并由模型选择何时使用”。} } ]

目前近两年的文献中，Periodic-TDL、MI-MoE、MoleBLEND、MuMo 四条证据链恰好分别对应你的四个瓶颈：周期多尺度、尺度选择、细粒度预训练对齐、以及 downstream fusion leverage。它们的组合，是我认为当前最有希望把 MIPS 从“强 O8 + 弱几何 residual”升级成真正 2D topology + multiscale 3D cooperative model 的方向。