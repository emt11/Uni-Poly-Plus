# 研究计划二：三模态查询表面 patch 与原子／键—表面局部交互

> 计划 ID：UA3D-LOCAL-SURFACE-20261104-01；修订 r1；编制日期 2026-11-04。状态：**研究计划已编制，尚未实施或运行**。
>
> 本次只授权编写计划。本文不替代 [CURRENT_PLAN.md](../../CURRENT_PLAN.md)，不恢复任何 STOPPED 周期，不授权实现、训练或 TEST。未来选定后，将完整合同登记到 CURRENT_PLAN.md，由获用户执行授权的 DSH 实施，Codex 独立审查。
>
> 同组计划：[条件化谱传播](BASE_CONDITIONAL_SPECTRAL_PLAN.md)、[体素与连续场](BASE_VOXEL_CONTINUOUS_FIELD_PLAN.md)。三个方向不是已授权的串行／并行实验清单。

## 1. 基础、目标与研究问题

基于 dev / 2992e3cd095ba636c0f120251c43ca435ae3ea89 及受保护 dirty worktree；编制时已查看状态、相关 diff、CURRENT_PLAN、历史和实际模型接口。TODO.md、用户 PH.md 不属于本次改写范围。

reference 为 F_PD_N1_F0 七任务五折 BEST，正式 macro R² 完整值 0.804675462276949，展示 **0.805**；independent_review=PENDING。B=results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/。

研究目标为正式 macro≥0.810，记录 target_met；默认 performance_pass 仍要求 ΔBASE≥0.010，即未舍入 macro≥0.814675462276949。共享开发 TEST 不称独立盲测。

**两个递进问题：**

1. 保留局部表面 patch，并让 z₂、z₃、zFP 分别查询，是否优于表面全局池化？
2. 在骨干仍能保留局部状态时，引入原子／键—表面双向通信，是否优于只有模态级查询或单向映射？

最终预测来自更新后的多模态表示与联合性质头，不使用 ŷBASE＋Δ，不输入教师预测。允许修改 BASE 的前向路径、局部交互、融合，并联合微调编码器。

## 2. 论文依据与证据边界

| 编号与来源 | 原论文提供的依据 | 本计划的迁移 |
|---|---|---|
| L1：[AtomSurf: Surface Representation for Learning on Protein Structures](https://arxiv.org/html/2309.16519)，ICLR 2025，首稿 2023 | 用空间二部图让图节点与表面顶点逐层双向通信。 | 主要架构来源；迁移到重复单元／Trimer，保留 FP 全局分支。 |
| L2：[ProteinF3S: boosting enzyme function prediction by fusing protein sequence, structure, and surface](https://yiheng-zhu.github.io/Yiheng/papers/8/ProteinF3S.pdf)，Briefings in Bioinformatics 2025，DOI 10.1093/bib/bbae695 | 多尺度双向结构—表面融合，再融合序列；提供融合消融。 | 局部交互先进行，Morgan 等全局表示后融合。 |
| L3：[Surface-VQMAE: Vector-quantized Masked Auto-encoders on Molecular Surfaces](https://proceedings.mlr.press/v235/wu24o.html)，ICML 2024 | 表面 patch、SurfFormer 与表面自监督。 | 采用 patch token 接口；首轮不复现其大规模预训练。 |
| L4：[SurfBind: Deciphering Fingerprints of 3D Molecular Surfaces for Accurate Epitope Prediction](https://arxiv.org/html/2606.23830v1)，2026 作者稿 | 表面局部编码、patch 注意力和条件交叉注意力。 | 将条件来源改为 BASE 三模态；没有抗体输入，也不使用蛋白任务标签。 |
| L5：[dMaSIF: Fast End-to-End Learning on Protein Surfaces](https://openaccess.thecvf.com/content/CVPR2021/html/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.html)，CVPR 2021，较早基础 | 有向表面点云与准测地卷积。 | 局部表面编码的基础参考，不纳入近三年新论文计数。 |

以上主要是蛋白任务证据，不能证明聚合物七任务会受益。本文的三模态 query、聚合物原子映射、轻量消息算子和两级插入位置均是项目适配，不能称完整复现原模型。

ProteinF3S 的简单拼接不总优于结构单模态；因此本文同时保留池化、query、单向和双向控制，而不是预设注意力必然有效。

## 3. 数据、初始化与资产边界

- 七任务顺序 Eea、Egb、Egc、Ei、EPS、Nc、Xc；data/splits/mips_paper5_outer/{task}.json；fold0—4，seed=42+fold。
- 5,875 原行全保留，跨折 train／validation／TEST 分别18,786／4,714／5,875行；沿用原 target scaler 和原 row／sample_key 身份。
- 所有新训练从 B/pretrain/F_PD_N1_F0/deploy.pt 起步；原始头、输入和 scaler 从 B/finetune/F_PD_N1_F0/{task}/fold{fold}/config.json 解析。禁止从下游 BEST 或旧适配器接续。
- 复用原PT schema验证及显式键转换，只排除已登记的PT辅助头；新增交互模块单独初始化。禁止用strict=False忽略无法解释的键或形状问题。
- 只读表面源 P2=results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2/。从其来源 manifest 解析 audit.npz 的 vertices、faces、mass、normals、H、K、owner、component，以及原物理坐标／元素／RU／base_id。
- 新 patch sidecar 可从既有网格派生；不重新构面、求谱或生成新构象。原失败／缺3D名单保留，缺表面行不删除。
- 现有 z 图级缓存不包含所需局部状态。联合训练必须读取当前模型状态；不能把其他 fold 的下游节点表示共享给本 fold。
- label-free 几何 sidecar 可以按原身份去重；任何可训练编码、统计拟合或参数选择只使用对应 fold 的训练／validation 角色。TEST 神经特征和标签在冻结后才读取／生成。

## 4. 表面点与 patch 的统一定义

### 4.1 面积权重与采样

每个原网格最多保留 N=1,024 个代表顶点；不足则全部保留并 padding、显式 mask。代表点采用固定、label-free、基于距离的 farthest-point sampling，起点由原 owner／vertex 身份顺序确定。所有臂共用同一份 sidecar。

每个原顶点分配到同一连通分量内最近代表点；将其质量矩阵对角权重累加到代表点。相同权重用于聚合原特征、中心归属和法向，法向汇总后归一化。总面积与中心面积保留，代表点坐标仍取原表面点。

特征基于原16通道定义，包括元素族、半径、曲率、中心归属、HKS；面积用于汇聚，不把面积全归一化为1后丢弃真实尺寸。

若连通分量数超过1,024、有效面积丢失或法向无法可靠归一化，停止；不静默丢分量。该采样是表示近似，不能称精确保留原曲面微分算子。

### 4.2 局部表面编码

固定两个 d=128 表面块，各点最多16邻居；仅在同一连通分量内构边。用距离 RBF 与以下标量描述局部几何：

$$
\|r_{ij}\|,\qquad n_i^\top n_j,\qquad
n_i^\top\widehat r_{ij},\qquad n_j^\top\widehat r_{ij}.
$$

这些标量不依赖全局坐标轴。距离使用16个 Gaussian RBF，中心0—4Å，超过4Å的边不进入局部块。边网络产生 attention／消息权重，再接逐点 MLP128→256→128、LayerNorm与0.1 dropout。

这是受局部表面学习启发的轻量算子，不宣称与 dMaSIF 原卷积数值相同。欧氏近邻即便有法向特征，也不能保证等于真实测地邻域；记录薄片两侧误连接风险与连通分量分布。

### 4.3 Patch

在代表点上选最多64个中心，每个 patch 取同分量内最近32点；保留重叠和点掩码，以面积权重汇聚为128维 token。中心点 FPS 仍使用距离和原身份打破并列，不采用依赖坐标轴的 Morton 顺序。

同时保留 patch 的面积、中心归属比例、面积加权曲率和相对距离信息。对 token 的任何绝对位置编码禁止直接使用全局 xyz；patch 间采用距离／法向不变量。

## 5. 路线A：三个模态 token 查询表面 patch

令 Z=[z₂,z₃,zFP]，S 为最多64个表面 token，Q／K／V 投影维度均128，4 heads，单层：

$$
R=\operatorname{Attention}(ZW_Q,SW_K,SW_V).
$$

得到三份不同的表面摘要 R₂、R₃、RFP。将每个 [zᵢ,Rᵢ] 经同形 384→256 投影，得到更新的模态 token；另保留面积加权表面全局 token256，四路 gate 后直接预测。

- 原缺失模态仍屏蔽；表面不可用时不进入全 mask softmax，直接绕过查询，保留本轮三模态路径。
- 不把表面平均成一个 token 再称局部检索；不把这条路线称原子级融合。
- Pool 对照也执行同形 token 更新投影，但把每个 Rᵢ 设为同一个面积汇聚摘要。参数总量与有效容量差异在交付中列明。
- 本轮不加入 masked pretraining。若未来加入，所有对应模态须同步遮蔽同一区域，避免用未遮蔽的全分子向量直接泄露重建目标。

## 6. 路线B：原子／键—表面双向局部交互

### 6.1 物理身份映射

2D 分支输出 canonical 原子状态 h₂，3D 分支输出物理键／线图 token h₃b。不得将键 token 下标当原子下标。

1. 用既有 physical_base_id／RU offset 将2D重复单元状态展开到各有效 Trimer 重原子副本，保留副本身份；展开不代表三副本最初拥有不同的2D化学上下文。
2. 用真实键端点 incidence 将3D键状态均值汇聚到物理原子；端点与 mask 按原物理键 schema 解析，不使用 canonical 编号合并空间副本。
3. 原子状态为两路128维投影的融合，另有元素／可用性／RU相对身份。未匹配 H 与 center_physical=-1 按合法负哨兵处理，不伪造重原子。
4. 每个表面代表点连接距离≤4.5Å的最多4个物理原子，按距离与原身份排序；原子数不足时只保留真实邻居，无邻居点显式 mask。
5. 原子—表面边使用距离 RBF 与表面法向相对位移，不输入全局轴向坐标。边不得跨样本。

### 6.2 两级通信与回写

在原2D／3D的第5、第6层后各放一个交互块，不改变其512维主宽度；局部通信宽度128，两个位置使用独立参数。

$$
S^{l+1}=E_s^l\!\left(S^l,\operatorname{MP}_{a\to s}^l(H_a^l)\right),
\qquad
\widetilde H_a^l=E_a^l\!\left(H_a^l,
\operatorname{MP}_{s\to a}^l(S^{l+1})\right).
$$

对3D，原子增量经128→512投影，再按每条物理键两个端点的均值回写键 token。对2D，只把中心 RU 中合法对应原子的表面消息映射回 canonical 原子；不把不同副本未经区分地相加。

消息回写投影可零初始化以稳定训练；这是骨干内部特征连接，最终仍是联合表示预测，不是独立的 ŷBASE＋Δ 模型。

两个位置均保留更新后的状态供后续层使用，再由原 readout 得到 z₂／z₃，结合 FP 与表面全局 token 做四路 gate。FP 保留全局化学角色，不伪装成逐原子监督。

### 6.3 参数冻结与缓存限制

- head阶段冻结骨干参数，但冻结模块之后若接受了可训练消息，必须保留对输入的 autograd；不能把整个后续骨干置于 no_grad 后声称双向模块可学习。
- joint阶段更新原骨干与局部模块。被修改层之后的神经特征不作跨 epoch 缓存。
- 允许在新 wrapper 中显式组织原 layers 与新增模块。不能修改旧源文件、以全局 monkey patch 改其他实验，或用最终图级向量假冒中间节点。
- 若现有 schema 无法建立上述物理映射，资格阶段停止；不通过截断、猜索引、忽略 unmatched 或更换输入版本继续。

## 7. 六臂固定实验矩阵

| 臂 | 定义 | 对应来源／问题 |
|---|---|---|
| C_REFT | 原 BASE 配方，同初始化重训 | 训练与预算控制 |
| P_POOL | 同一局部表面编码器，面积池化后联合预测 | 排除仅新增表面编码器的混杂 |
| P_QUERY | 三个模态 query 查询64个 patch | L3／L4启发的模态级局部选择 |
| L_FORWARD | 两级原子→表面，关闭反向回写 | L1／L2启发的单向对照 |
| L_BIDIR | 两级原子↔表面，回写2D原子／3D键 | 主要局部融合候选 |
| L_BIDIR_CONSTEDGE | 与 L_BIDIR 同邻接、同参数；边距离／法向数值换为固定有限占位 | 检验给定空间邻接时的几何数值作用 |

L_FORWARD 仍注册同形反向模块，但反向增量禁用；有效容量不同必须报告。CONSTEDGE 保留真实空间邻接，不能称无几何模型。P_QUERY−P_POOL检验此实现的查询方式；L_BIDIR−L_FORWARD包含反向通信与其有效容量；L_BIDIR−CONSTEDGE控制拓扑，检验边几何数值。P_QUERY与L_BIDIR同时改变局部层级／算子，不能作单因素因果比较。

每臂35套、全部独立从共同 PT deploy／任务头起步。不同架构共享参数尽可能保持同初态和批顺序；新增模块隔离初始化，保存每臂 step0 状态，禁止读父部分 BEST。

## 8. 训练、选模与冻结

固定 batch24、FP32、70epochs=head10＋joint60、drop_last=False。AdamW、weight_decay0.02、betas(0.9,0.999)、eps1e-8，clip5。

- native joint lr1e-4；原模态／gate／head lr=head3e-4、joint1e-3；新增表面、query、局部消息 lr=3e-4。
- head10 cosine最低3e-5；joint60使用10-epoch cosine warm restart，Tmult1、最低1e-6。只在 head→joint 重建优化器。
- 原 native dropout0、FP0.2、性质头0.5；新增模块0.1。原三方向 ALIGN0.08 保留。
- loss=MSE＋0.08·ALIGN−0.08·Hnative。Hnative只计算 gate 原三模态权重重新归一化后的熵，不把第四路最大熵增加混入。
- 每 epoch一次完整validation，原单位RMSE严格下降更新BEST；同分保留早epoch，不early-stop，不追加种子或自监督阶段。
- 原模块数值／确定性与模式设置继承r6，不同时调整TF32等因素；新增模块隔离RNG。五折seed系列不作为额外多seed稳健性证明。

完成210套训练后，全局候选按七任务validation等权macro选择；须严格优于原BASE已存validation macro，且逐任务mean≥BASE−0.010。精确同分优先C_REFT，再按表中顺序。无合格候选则BASE_KEEP。

一次写selection/final_frozen.json，锁定全局臂、35套BEST、所有对照、配置、映射／sidecar版本、scaler、规则和reference。冻结前不读取逐行TEST标签／预测，不生成TEST神经表示。冻结后在同一正式阶段完成六新臂＋原BASE alias，各task/fold一次；不按任务或折拼接胜者。

报告七任务mean±std(ddof=1)、等权macro、全精度ΔBASE及机制配对；展示保留三位小数。按SOTA_RESULT.md列出各任务参考均值、模型、差值及协议差异；不把论文逐任务包络当单模型。

## 9. 固定资源、预算与速度门禁

| 项目 | 上限 |
|---|---:|
| 新PT／构象／构面／谱分解 | 全部0 |
| 新派生几何sidecar | 原行上限5,875，按既有几何身份去重 |
| CPU旋转几何审计 | 4个预先固定有效样本×3个固定刚体旋转=12项，不产生新构象 |
| 新FT／epochs／optimizer updates | 210套／14,700／336,000 |
| 正式单元／预测行 | 245／41,125，含BASE alias |
| 前置CPU suite／额外GPU诊断 | 0／0 |
| 最终CPU suite | 正式阶段后至多1次 |
| 总硬截止／准备阶段 | 新授权t0起72h／最多8h |

- 仅物理GPU2；编制时UUID GPU-31456dd5-a421-142b-644b-7a80f482dfb9、显存49,140MiB。CUDA_VISIBLE_DEVICES=2，内部cuda:0、WORLD_SIZE1，禁用其他卡。
- 编制时112个affinity CPU、约251.5GiB主机RAM；运行服从实际affinity／cgroup。几何准备最多96个单线程worker，BLAS／OpenMP1线程，调度保留8核，按每作业峰值内存进一步降低并发。主机本周期峰值上限160GiB。
- 几何采样／邻接／patch为CPU并行一次性产物；GPU输入采用内存映射、按样本拼接、padding mask和pinned预取。GPU训练不在每个epoch重新跑FPS／KDTree。
- GPU FT＋validation与正式TEST分阶段，最多4worker；起始1路，使用正常训练已发生的显存／时间观测，满足总峰值≤可用显存80%时才按1→2→4提高。GPU worker主线程2、DataLoader4worker且各1线程，不改batch。
- 前3套正常单元分别覆盖C_REFT、P_QUERY、L_BIDIR，均计入210。用其正常epoch记录估算成本；不另跑benchmark。conservative ETA=1.5×按稳定实测吞吐推算剩余工作＋2h收尾；超过72h剩余时间即STOPPED，不删慢臂凑完整。
- CPU旋转样本从eea、eps、nc、xc各取原indices最小、表面PASS且物理身份互异的样本；固定绕x／y／z轴37°／73°／121°。只核对派生距离、法向不变量、面积与身份；最终CPU suite再验证模块旋转／排列一致性，不提前运行模型smoke。
- 所有worker在tmux Uni-Poly独立window执行，保留命令／cwd／日志／真实exit。调度应以poll/wait确认退出；/proc采样ENOENT／ESRCH归并为对象已退出，不把遥测竞态变成训练重试。

预算包含失败／中止的尝试和更新，无自动退款。旧HMR累计账本保留，新增sidecar／FT消费另列，不把旧失败现场改写为成功。

## 10. 允许修改、停止与交付

仅新增本计划自己的configs、dataset、model wrapper、training、prepare／controller／bootstrap、测试和说明；旧BASE／HMR／READOUT代码与结果只读。允许替换融合及包装第5／6层的局部通信，允许联合微调原权重；禁止改标签、split、原cache和旧checkpoint。

拟实施文件为configs/ua3d_local_surface_r1_gpu2.json、src/dataset/ua3d_local_surface_r1.py、src/modules/ua3d_local_surface_r1.py、src/training/ua3d_local_surface_r1.py、scripts/prepare_ua3d_local_surface_r1.py、scripts/run_ua3d_local_surface_r1.py、scripts/bootstrap_ua3d_local_surface_r1.py及tests/test_ua3d_local_surface_r1.py；目前均未创建。先静态检查生产epoch0／metrics／finite和层级包装接口，不通过提前跑模型smoke替代接口定义。

拟结果根results/ua3d_local_surface_20261104/r1/，日志根logs/ua3d_local_surface_20261104/r1/；当前不创建。若已存在不明或活动现场，停止，不覆盖或自动换修订。

停止条件：节点／键／RU／owner映射不一致；合法负哨兵被错误解释；有效输入或梯度NaN/Inf；面积／连通分量丢失；反向消息无合法梯度通路；writer冲突、OOM、预算／ETA超限、真实worker失败。不得在线修复重试、换GPU、缩batch或扩大缺失集合。

最终CPU suite只运行一次，覆盖：

- 物理键incidence、中心RU回写、batch隔离与缺失模态。
- 面积守恒、旋转／顶点重排一致性、padding不参与attention。
- query与局部双向模块的梯度可达；冻结参数时对输入的梯度仍存在。
- 生产metrics接口、原单位预测、原indices／sample_keys、冻结及预算汇总。

交付路径内至少包含sources.json、geometry_manifest.json、预算及累计消费、实际diff／参数／优化器分组、所有BEST／history、一次冻结清单、正式预测、final_review/results.json与delivery_summary.md。

机制分析只用已登记训练／validation日志或冻结后的固定正式输出：query分布、各方向消息强度、几何边消融、缺失组覆盖，不额外生成可被重新选模的TEST变体。正式报告说明完成层级、七任务对照、原始macro／差值、三个真实exit及independent_review=PENDING。正式TEST不完整则“未完成/不可比较”。

本周期不加入多构象、长链、MD、电势量化计算或masked预训练，也不自动叠加计划一的条件谱模块。若希望组合，先登记新的组合消融与预算；不能看完本周期TEST后临时加入。
