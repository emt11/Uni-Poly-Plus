# 近两年分子／聚合物属性预测论文调研：对 UA 路线的借鉴

> 整理日期：2026-10-25，Asia/Shanghai。文档整理检查时间：17:53:43。
> 范围：保存本次对话第一轮文献调研，以 UA_2D 为主要迁移基线。
> 状态：调研文档已整理；研究建议尚未在 UA 中验证，不构成新的实现或实验执行计划。
> 配套文档：[聚合物 3D 构象与重复链表征调研](POLYMER_3D_CHAIN_REPRESENTATION_SURVEY_20261025.md)。

## 1. 主要判断

对当前 UA，最值得研究的方向是：显式基团层级表示、任务适配，以及更有针对性的 3D→2D 知识迁移。继续增加模态、拼接描述符或扩大模型，当前项目证据下优先级较低。

- 若目标是形成可解释的结构改造，优先研究“周期结构中的基团层级表示”。
- 若目标是先以较低成本寻找突破，优先研究低秩适配与任务相关读出。
- 对预训练目标，优先明确 ALIGN 的正负样本语义与教师监督内容，而不是仅调整损失系数。
- 所有迁移判断都是研究假设，不能将原论文效果换算为 UA 的预期提升。

## 2. 检索范围、版本与证据限制

### 2.1 时间与文献选择

检索窗口为 2024-10-25 至 2026-10-25。优先纳入正式期刊、会议版本；对预印本早于窗口、正式发表位于窗口内的论文，单独说明时间。尚未核实正式发表的研究按预印本评价。

这是围绕 UA 的定向筛选，并非覆盖全部文献的系统综述。研究价值按科学问题、与现有 UA 的差异、迁移成本及可验证性判断，不按发表级别直接推断项目收益。

本报告整理已有两轮调研中核实的论文页面、正文、补充材料、作者预印本和部分作者公开仓库信息。整理本文件时没有重新进行全量网络检索。

### 2.2 本地依据

- 分支：`dev`；HEAD：`cd36ae7280b159f881b483ad5814835ff7b74c4a`，提交说明 `P10K`。
- HEAD 不是完整工作区实现版本：检查时 `CURRENT_PLAN.md` 和用户 `TODO.md` 已有 tracked 改动，UA3M、TMOL3M 消融及旧无 FP／A020 融合有未跟踪代码、配置、脚本和测试。全部保留；未读取 `TODO.md`。
- 当前交接：[CURRENT_PLAN.md](../../CURRENT_PLAN.md)，整理时记录的是 TMOL3M 下游 3D 配对消融，状态为待审查。本报告不修改其授权、状态或结论。
- 依据当前源码及既有结果记录进行分析；未重跑训练、推理、测试或指标评估，不构成相关实验的新增独立审查。
- 调研中的实验状态为相应记录快照，后续应以当前交接和实际产物为准。

## 3. UA_2D 已有能力与剩余缺口

当前 [Pure2DGLTModel](../../src/modules/glt_dual.py) 的下游路径是：

`周期拓扑 → BondPathO8（6层、512维、2-hop关系）→ 原子均值池化 → LayerNorm → MLP`

它已有周期连接、路径与键化学偏置，也已有多个利用 3D 辅助预训练的版本。因此，论文中的“引入周期性”“使用 3D”“多模态对齐”，不能直接算作 UA 的新增贡献。

更具体的机会包括：

1. 原子与整图之间，缺少显式基团／片段层级。
2. 不同性质最终都经过同类均值读出，任务相关结构的选择比较间接。
3. 图级 ALIGN 不保证学生学到性质相关的局部几何知识。
4. 现有周期表示是否覆盖不同重复单元选取方式，需要区别于普通原子编号不变性。

### 3.1 现有结果对研究选择的约束

既有三模态[交付报告](../../results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/final_review/report.md)及[完整精度结果](../../results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/final_review/results.json)记录：

| 方法 | 正式 TEST macro R² | 相对 B_A020 的 Δmacro R² |
|---|---:|---:|
| B_A020 | 0.785 | +0.000 |
| UNI3M | 0.778 | −0.007 |
| TMOL3M | 0.790 | +0.005 |

两个候选均未达到项目 0.010 门槛，父交付仍为待审查。比较同时改变 FP 参与方式、ALIGN、下游 3D、融合和部分优化设置，不能将差值独立归因于某个模态或融合机制。

这些结果不证明多模态无效，但没有支持“增加融合复杂度即可解决当前问题”。同样，PH 描述符的拼接／门控结果不能检验完整的高阶几何消息传递架构。

## 4. 重点论文与迁移价值

以下优先级指对当前 UA 的迁移价值与成本，不代表论文质量排名，也不代表已证明能提升 UA。

| 编号 | 论文与发表来源 | 核心价值 | 对 UA 的判断 |
|---|---|---|---|
| R1 | [Molecular Motif Learning as a pretraining objective for molecular property prediction（MotiL）](https://www.nature.com/articles/s41467-025-66685-w)，Nature Communications，2025 | 整分子与 motif 双尺度学习；参数扰动保留完整输入结构 | 高：补足原子—整图之间的层级 |
| R2 | [Fragment-based representation learning for polymers and heterogeneous formulations（CGMNet）](https://www.nature.com/articles/s41524-026-02357-7)，npj Computational Materials，2026 | 以化学片段及其连接关系连接小分子预训练与聚合物预测 | 高：直接支持片段作为聚合物表示接口 |
| R3 | [MIPT: Multilevel Informed Prompt Tuning for Robust Molecular Property Prediction](https://proceedings.mlr.press/v267/chen25cu.html)，ICML 2025 | 节点低秩适配、图级 prompt、任务相关信息提取 | 高：适合低成本迁移；原文主要证据来自分类 |
| R4 | [Periodicity-aware deep learning for polymers（PerioGT）](https://www.nature.com/articles/s43588-025-00903-9)，Nature Computational Science，2025 | 周期先验驱动的对比预训练、周期 prompt 和条件节点 | 高但重叠较多：必须找到 UA 的剩余缺口 |
| R5 | [A probability contrastive learning framework for 3D molecular representation learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6adaf0cbeba11705d4ea67a62044f63d-Abstract-Conference.html)，NeurIPS 2024 | 概率权重缓解错误正负样本对 | 中高：研究 ALIGN 的监督语义 |
| R6 | [3D Denoisers Are Good 2D Teachers: Molecular Pretraining via Denoising and Cross-Modal Distillation（D&D）](https://doi.org/10.1609/aaai.v39i1.31986)，AAAI 2025；预印本始于2023 | 先训练去噪教师，再冻结教师蒸馏到 2D 学生 | 中高：贴合纯 2D 部署，但需区别于已有局部蒸馏 |
| R7 | [Exploring Molecular Pretraining Model at Scale（Uni-Mol2）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/53923bb44655a7defb31c7744c01b62b-Abstract-Conference.html)，NeurIPS 2024；预印本始于2024年6月 | 双轨 Transformer 整合原子、图和几何信息，并研究规模规律 | 中：结构设计比直接扩大规模更值得借鉴 |
| R8 | [Beyond Atoms: Enhancing Molecular Pretrained Representations with 3D Space Modeling（SpaceFormer）](https://proceedings.mlr.press/v267/lu25e.html)，ICML 2025 | 对原子之外的三维空间进行离散化和建模 | 中：偏长期教师路线 |
| R9 | [Quadruple Attention in Many-body Systems for Accurate Molecular Property Predictions（MABNet）](https://proceedings.mlr.press/v267/rao25a.html)，ICML 2025 | 显式建模四体相互作用 | 中：偏几何教师增强；主要证据来自能量／力任务 |
| R10 | [Kolmogorov–Arnold graph neural networks for molecular property prediction（KA-GNN）](https://www.nature.com/articles/s42256-025-01087-7)，Nature Machine Intelligence，2025 | 在嵌入、消息传递和读出中使用 Fourier-KAN | 低到中：函数表达能力值得研究，但不是聚合物专用先验 |
| R11 | [Periodic Topological Deep Learning for Polymer Design and Discovery（Periodic-TDL）](https://arxiv.org/abs/2605.26833)，2026预印本 | 周期距离、跨尺度单纯形消息传递、高阶关系 | 与项目高度相关；本次核实来源为预印本，不能列作正式顶刊成果 |

补充来源：[Unified multimodal multidomain polymer representation for property prediction（Uni-Poly）](https://www.nature.com/articles/s41524-025-01652-z)，npj Computational Materials，2025。当前项目已做其三模态适配，不能再次将相同融合设计作为未尝试的新方向。

## 5. 优先方向一：显式组织化学基团及其连接关系

### 5.1 论文机制

MotiL 同时学习整个分子与内部 motif 的表示。其 bi-scaled 阶段使用 Barlow Twins 风格目标，并通过参数剪枝形成编码视图，避免依赖破坏化学结构的输入图增强。完整方法还包括 diffusion priming；只移植 motif 学习不能称为完整复现。

CGMNet 以化学片段为节点，保留片段之间的连接，从小分子预训练迁移到聚合物预测。其结果也提示片段划分存在性质依赖性，没有证明某一种分片适合所有性质。

### 5.2 UA 适配假设

保留 O8，增加以下层级：

`O8原子表示 → 化学基团池化 → 基团连接图 → 基团间消息传递 → 回注原子／参与读出`

让模型更直接地表达芳环与连接基团的排列、极性基团的主链／侧链位置、共轭片段连接与中断、局部刚性与柔性片段组合。

这不同于整体预测 Morgan 指纹或一跳化学上下文。辅助标签提供监督，但不自动建立片段之间的显式结构关系。

### 5.3 与既有实现的差异及验证边界

- 项目已有 BRICS 分组相关代码，不能把使用 BRICS 本身称为创新。
- 跨重复单元边界的基团必须正确处理，不能让 `*` 人为切断真实结构。
- 相同基团处于不同上下文时，表示不应被无条件强迫完全相同。
- 科学问题应是“保持周期连接语义的基团层级，是否优于同容量普通原子适配器”。
- 容量匹配对照是必要的，否则收益可能只是新增参数带来的。

评价：最适合发展为 UA 主线结构改造。原子—基团层级可以与配套调研中的重复单元—链层级衔接，但两部分的贡献应分别验证。

## 6. 优先方向二：低秩适配与任务相关读出

### 6.1 论文机制

MIPT 冻结预训练主干，通过低秩矩阵更新节点变换，从节点表示构造图级 prompt，并利用局部与整体表示的一致性辅助学习。原文还根据损失分布进行噪声筛选。

### 6.2 UA 适配假设

固定同一个 UA_2D checkpoint，在选定线性变换中加入低秩适配，让读出选择当前性质相关的原子／片段信息。

与已做的末两层微调、层间读出混合相比，新增问题是限制更新自由度并建立显式任务适配路径。当前不同任务分别训练时，可以各自学习适配模块；这不需要自动扩展为跨任务联合训练。

### 6.3 适用限制

MIPT 主要验证八个 MoleculeNet 分类数据集，没有直接证明对当前聚合物回归有效。尤其不能直接照搬“高损失样本不可靠”的筛选：困难、稀少而重要的回归样本可能被误伤。

最值得先研究的问题是：在相同 checkpoint、split 和训练预算下，低秩适配与任务相关读出能否改善全量微调的泛化。保留全部训练样本和原回归目标，使适配机制的影响可解释。

评价：适合先做成本较低的迁移研究，实施成本低于重新设计整套预训练。

## 7. 优先方向三：重新审视 ALIGN 的样本关系

R5 的核心问题是：哪些分子应该靠近、哪些应该分开，是否能仅由样本身份决定。论文通过贝叶斯建模学习样本对权重，以缓解错误正负样本。

当前 UA 的 [alignment_loss](../../src/modules/glt_dual_pretrain.py) 已支持同身份多正样本，不能把“支持多正样本”当作新方案。剩余问题包括：

- 不同字符串是否对应等价周期结构。
- 相似局部化学环境是否被整图身份规则一概视作负样本。
- 某个几何视图是否不可靠，或包含不适合迁移的细节。

可研究部分负样本的权重，或不依赖显式负样本的表示约束。MotiL 的去冗余目标是另一类参考，但改损失与加 motif 层级应分别比较。

结构相似不代表性质相同。不能将所有同 scaffold 或相似 FP 的聚合物直接设为正样本，否则可能抹去性质预测最需要区分的细微变化。用于构造权重的规则或模型也不能读取 TEST 标签。

评价：比继续扫 ALIGN 系数更明确的预训练目标问题，但目前没有 UA 正式收益证据。

## 8. 优先方向四：固定去噪教师向纯 2D 学生迁移

D&D 的步骤是先通过坐标去噪训练 3D 教师，再冻结教师，用图级或原子级表示回归训练 2D 学生，下游丢弃教师。可读[作者预印本](https://arxiv.org/abs/2309.04062)。

UA 已有 QL 局部蒸馏和 ET_SINGLE 等相关尝试，不能再次笼统建议“加入原子级 3D 蒸馏”。既有结果及边界见 [PROJECT_HISTORY.md](../../PROJECT_HISTORY.md)，相关实现见 [cleanFP 预训练模块](../../src/modules/ua2d_cleanfp_pretrain.py)。

剩余科学问题是：固定且经过独立目标训练的教师，是否比与学生共同变化的教师提供更稳定、有效的监督。

- UA 3D 路径以键表示为主，D&D 的 node 蒸馏依赖原子一一对应。
- 将 incident bond 聚合到原子是一种项目适配，不等同于原文原子教师。
- 一张 2D 图可能对应多个构象；要求学生复制某一个偶然构象的全部细节未必合理。
- 构象间稳定的表示或构象分布统计可能更适合作为目标，但这是待验证建议。

评价：贴合纯 2D 部署，适合作为后续教师路线；成本和归因难度高于前两条。

## 9. 优先方向五：扣除既有能力后的周期等价视图学习

PerioGT 的启发是周期先验如何参与学习，以及如何将等价重复单元中的信息用于下游 prompt。参考[作者仓库](https://github.com/wuyuhui-zju/PerioGT)。

需要分别考察原子编号变化、同一重复单元的 SMILES 写法变化、同一无限链的重复单元边界变化，以及用不同大小的单元描述同一周期链。这些不是同一个不变性问题，RDKit canonical SMILES 不能自动解决全部周期等价性。

应先通过源码确定哪些变换已被当前 canonical topology 消除。若两个视图最终产生相同输入，额外一致性目标没有新增监督信息。

若仍有缺口，可借鉴等价周期视图一致性、跨边界等价 motif 聚合和单位归一化的周期 prompt。重复单元扩张不能直接配合 sum pooling，否则混入单元大小；改变立体化学、组成或连接方式后也不能继续称为等价增强。

评价：相关性很高，优先级取决于现有表示尚未覆盖的实际缺口。本次对 PerioGT 的依据为官方摘要、补充材料和作者仓库，不宣称已取得全部付费正文实现细节。

## 10. 其他架构的适配价值

| 方法 | 值得借鉴的具体点 | 当前限制 |
|---|---|---|
| Uni-Mol2 | 研究逐层更新的关系表示，而非仅固定关系偏置 | UA 已有路径偏置；新增价值须来自关系更新机制。大规模结果不能外推到单卡小模型 |
| SpaceFormer | 用空间 token 表达原子之外的环境，再通过教师迁移给 2D | 有限链周围空间不等于体相堆积，需要确认现有几何可支持的语义 |
| MABNet | 在教师中补充四体／扭转相关信息 | 能量和力结果不能外推为七任务 R² 改善；需要新的几何语义与预算 |
| KA-GNN | 先考虑容量匹配的局部函数变换替换 | 原文在多处使用 KAN；单换回归头是局部适配。Fourier 基函数不自动构成聚合物周期先验 |
| Periodic-TDL | 跨尺度结构间的信息传播 | 已有 PH 描述符融合没有检验完整高阶消息传递；全面移植属于较大路线变化 |
| Uni-Poly | 多模态统一表示及信息互补 | 已做三模态适配；原文还使用 SMILES、caption 等信息，当前适配结果不能否定原文完整方法 |

## 11. 研究优先级与受控比较原则

| 顺序 | 研究方向 | 核心问题 |
|---|---|---|
| 1 | 周期语义下的显式 motif 层级 | 化学片段及其连接是否优于同容量普通适配器 |
| 2 | 低秩适配与任务相关读出 | 当前性能是否受限于微调和读出方式 |
| 3 | ALIGN 样本关系／去冗余目标 | 当前目标是否压制应保留的相似化学信息 |
| 4 | 固定去噪教师→纯 2D 学生 | 教师质量和稳定性是否比增加联合对齐更重要 |
| 5 | 周期等价视图学习 | canonical 表示之外还剩哪些真实不变性缺口 |

以上是研究建议，不是已授权实验矩阵。若后续制定执行计划，需要另行固定数据、预算、资源、停止条件及交付合同。

比较时应绑定具体 UA_2D checkpoint，不能将 U1、B_FP、P10K 和无 FP 的 A020 混称为同一基线。[P10K checkpoint](../../results/ua2d_ptscale_20261022/r1/pretrain/checkpoints/step10000/deploy.pt)对应的既有正式 macro R² 为 0.793；其 Morgan 辅助目标、训练时长与输出融合的效应需要区分。

同数据、split、初始化和预算下改变目标因素；验证集选择后冻结，再进行正式 TEST。七任务为 Eea、Egb、Egc、Ei、EPS、Nc、Xc，各任务五折汇总后等权 macro。默认性能门槛使用未舍入的 Δmacro R² ≥ 0.010；展示均值、标准差和差值时保留三位小数。

多模块组合若要宣称交互收益，需要组合消融；不得用 fused−baseline 直接解释 interaction。共享开发 TEST 不称独立盲测；与 [SOTA_RESULT.md](../../SOTA_RESULT.md)协议不一致时只作数值对照。

## 12. 本地证据入口与完成层级

- [UA 编码器与纯 2D 下游](../../src/modules/glt_dual.py)。
- [ALIGN 实现](../../src/modules/glt_dual_pretrain.py)。
- [已有统计读出／L2-SP 迁移](../../src/modules/ua2d_transfer.py)。
- [已有键适配器](../../src/modules/ua2d_demol.py)。
- [三模态实现](../../src/modules/ua3m_paperfuse.py)。
- [Morgan 融合交付](../../results/ua2d_fg10k_morgan_20261023/r1/final_review/report.md)。
- [PH 描述符融合交付](../../results/ua2d_ph_descriptor_20261023/r1_p10k_gpu01/final_review/report.md)。
- [已归档实现与实验结论](../../PROJECT_HISTORY.md)。

本次完成文献调研整理与本地链接、格式检查；新增代码、训练、构象生成、推理和测试均为零。文档不改变任何实验的待审查、需返修或归档状态。
