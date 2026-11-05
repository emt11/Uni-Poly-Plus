# 研究计划三：体素与连续分子场的多模态联合预测

> 计划 ID：UA3D-MOLECULAR-FIELD-20261104-01；修订 r1；编制日期 2026-11-04。状态：**研究计划已编制，尚未实施或运行**。
>
> 本次授权为编写计划，未授权实现、训练、TEST、DFT或下载大型权重。本文不替代 [CURRENT_PLAN.md](../../CURRENT_PLAN.md)，不恢复停止现场。选中后先登记完整执行合同，再由获授权的DSH实施、Codex独立审查。
>
> 同组计划：[条件化表面谱](BASE_CONDITIONAL_SPECTRAL_PLAN.md)、[局部表面](BASE_LOCAL_SURFACE_PLAN.md)。三份不是已授权连续执行的总实验。

## 1. 版本、reference、目标与信息边界

- 基于dev / 2992e3cd095ba636c0f120251c43ca435ae3ea89及受保护dirty worktree；已读当前计划、历史、相关diff、实际BASE与几何接口。
- reference为F_PD_N1_F0七任务五折BEST，正式macro R²原值0.804675462276949、展示 **0.805**，independent_review=PENDING。
- B=results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/。新训练从B/pretrain/F_PD_N1_F0/deploy.pt及原始task/fold头起步；不继续旧下游BEST。
- 研究目标target_met：正式macro≥0.810；默认performance_pass另要求ΔBASE≥0.010，即未舍入macro≥0.814675462276949。结果为共享开发评价，不称独立盲测。
- 保留2D／3D／FP，新增场表示后联合预测，允许微调原编码器和融合层。所有候选直接输出ŷ，不采用ŷBASE＋Δ或输出集成。

**本轮要回答：**已有3D分支之外，空间场的表示与归纳偏置是否仍有增量？如果有，是廉价体积统计已足够，还是局部体素／连续查询及条件化融合更有价值？

原子代理场是由已有元素、半径和构象确定的新表征，不是新增物理测量。DFT-COSMO等场包含额外量化计算信息，不能和廉价代理场混为一类。本轮默认不生成DFT电子密度、电势、Fukui或COSMO。

## 2. 论文来源、证据强度与本计划的改动

| 编号与论文 | 已有证据 | 本计划采用／不声称什么 |
|---|---|---|
| F1：[ML Workflows for Screening Degradation-Relevant Properties of Forever Chemicals](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202523817)，Advanced Science 2026 | 原子属性体素场、FFT自相关、PCA与GPR预测全局性质。 | 借鉴空间相关描述符；更换代理属性并作径向压缩，属于项目适配。 |
| F2：[Prop3D: Geometry-aware lightweight convolutional network for efficient molecular property prediction](https://www.nature.com/articles/s41598-025-22049-4)，Scientific Reports 2025 | 分解式3D大卷积核与注意力，用于分子属性预测。 | 借鉴轻量体素编码；下面的分块处理与BASE条件化不是原模型复现。 |
| F3：[HyperDiffusionFields／HyDiF](https://arxiv.org/html/2510.18122v1)，2025预印本 | 连续分子距离／方向场，冻结表示与ChemBERTa-2融合有属性实验。 | 连续场与化学模态互补的直接依据；本轮不重训其大型扩散超网络。 |
| F4：[MolField: Molecular Representations in Implicit Functional Space via Hyper-Networks](https://arxiv.org/html/2601.22327v1)，2026预印本 | 隐式分子函数、权重token与超网络，包含属性预测。 | 借鉴连续函数学习；本轮采用小型条件解码器，不称完整MolField。 |
| F5：[SigmaFormer: Augmenting transformer encoders with COSMO sigma profiles for pure component property prediction](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.70450)，AIChE Journal 2026 | 将53维COSMO描述符与768维化学表示后期融合。 | 支持保留简单后期融合控制；真正COSMO计算不在首轮范围。 |
| F6：[FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)，AAAI 2018，较早基础 | 条件信息控制逐通道缩放／偏移。 | BASE调制体素通道的通用算法依据，非分子性能证据。 |

F3有几何／化学融合的直接实验；F5有量化场／化学融合实验。它们未验证当前已有3D的BASE，也没有证明聚合物目标能达到0.810。本轮所有具体通道、分块、距离坐标、融合组合及轻量INR正则均为项目方案。

## 3. 数据与初始化合同

七任务Eea、Egb、Egc、Ei、EPS、Nc、Xc；原split为data/splits/mips_paper5_outer/{task}.json；fold0—4、seed=42+fold。5,875原行不变，跨折train18,786、validation4,714、TEST5,875。

原输入、目标scaler、头初始化路径逐项解析B/finetune/F_PD_N1_F0/{task}/fold{fold}/config.json。按indices／sample_keys联合对齐，不改标签、cohort或split。

初始化复用原PT schema验证与显式键转换，只排除已登记PT辅助头；新增模块隔离初始化。不用strict=False忽略未解释缺键／多键或形状不匹配。

原Trimer物理坐标／元素／半径表只读复用；不新建构象。场可在表面构建失败、但原3D有效的行上计算，不把旧表面失败强行扩展成场失败。原15条缺3D行保留，mask场模态后由本轮可用模态预测，不保证等于旧BASE。

纯确定性几何／场sidecar可按已有物理身份去重；PCA、标准化、可学习编码、辅助场拟合均仅用当前fold训练行。validation只选模；冻结前不读逐行TEST标签／预测或计算TEST神经表示。

## 4. 所有场路线的共同代理场

### 4.1 可直接取得的三个通道

采用现有半径rᵢ、原子序数Zᵢ和坐标pᵢ，不依赖尚未核实的电负性／离子化能表：

$$
\rho_c(x)=\sum_i a_{ic}
\exp\!\left[-\frac{\|x-p_i\|^2}{2\sigma_i^2}\right],
\qquad
\sigma_i=r_i/2,\qquad
(a_{i0},a_{i1},a_{i2})=(1,Z_i,r_i).
$$

三个通道分别是占据代理、原子序数加权代理、半径加权代理。称为代理密度／属性场，不称真实电子密度、电负性场或DFT电势。三路线读同一物理定义，避免同时更换输入信息来源。

每个Gaussian在相对幅度低于1e-6处截断；记录截断定义、物理网格间距、原点和尺寸。浮点计算FP32，不能不记录地改σ、单元或元素映射。

### 4.2 共同保存的尺寸信息

保留三个场的积分、中心RU归属、物理包围尺寸、回转半径及可用性。不将所有分子缩放到同一单位盒后忽略尺度。

训练统计按fold训练行拟合；常量列用scale1并记录。任何半径／元素缺失或不支持的物理身份在准备阶段停止，不能猜值或删样本。

## 5. 路线A：FFT空间相关与两种融合

### 5.1 特征

使用0.5Å物理间距，根据分子支持范围建立可变尺寸网格。各轴做足够零填充，以至少2n−1长度计算线性自相关，避免把单分子边界误作周期边界：

$$
K_c(r)=\mathcal F^{-1}
\left[\overline{\mathcal F(\rho_c)}\mathcal F(\rho_c)\right](r).
$$

按物理位移长度径向汇聚：0—64Å每0.5Å一箱，另有>64Å的overflow箱；不丢长距离贡献。各箱对相关值乘体元后按该箱位移格点数取平均，空箱为0并保留可用性记录；另保留零位相关与场积分，不能把分子尺寸全部归一化掉。

每通道129箱，三个通道387维；再拼三个场积分、三个零位相关、三个包围尺寸、回转半径，共397维。标准化后训练集PCA降至128维；有效秩不足则补零并保留mask，不扩大拟合数据。

径向汇聚换取旋转稳定性，也丢失方向信息。离散网格只能近似旋转不变；FFT本身不提供旋转不变性。首轮不增加跨场互相关，以免扩大特征与候选预算。

### 5.2 融合

- FFT_LATE：将128维场特征投影为zF256，与三个BASE模态一起四路gate，再由性质头预测。
- FFT_FILM：相同zF编码，经BASE上下文产生γ、β并调制场隐层，再接同形四路gate。γ初始化为1、β为0；条件MLP为771→128→256，输出分别形成128维γ／β。
- LATE保留同形条件模块但输入固定零上下文；两者共同初态、参数形状与场输入相同，仍须说明有效条件函数容量不同。

若FFT网格单轴原始尺寸>256或填充后总格点>2²⁷，或预测单任务工作集超过内存合同，准备阶段停止。不自动提高间距、截断长分子或改用周期FFT。

## 6. 路线B：轻量体素编码与BASE条件化

### 6.1 分块与物理覆盖

维持0.75Å间距，以32³网格构成24Å边长tile；支持范围按第4节Gaussian截断定义覆盖。只保留与至少一个原子支持球相交的tile，每分子最多64个。

不通过缩小分子装入固定盒，不丢溢出tile。tile数量超过64则停止资格，不临时扩大缺场名单。tile原点、元素支持和中心RU覆盖显式登记。

每tile三个通道；CPU一次生成、分块压缩落盘，训练并行解压、pinned预取。缓存预估／实测上限96GiB；超过即停止，不把每epoch重复生成全部场作为隐性替代。

### 6.2 轻量编码器

每tile先以3×3×3卷积做3→16通道映射，再用两个Prop3D启发的分解式大核块：1×1通道混合＋沿三个轴的depthwise 7长核，通道16→32→64；每块后接通道／空间注意力与0.1 dropout。

tile汇聚为128维token，再用tile中心间物理距离RBF进行一层4-head注意力；按有效支持与mask汇聚为zF256。禁止只平均tile却声称保留完整全局布局。

固定tile microbatch16，梯度checkpoint分块，物理分子batch仍24；一次分子更新之前合并其全部tile，不能把tile当独立标注样本或改变目标权重。

体素分支采用BF16 autocast、FP32归一化／损失／优化器状态；BASE路径继续FP32。VOX_LATE与VOX_FILM设置完全相同，精度差异在实际diff中报告。

### 6.3 两种融合

- VOX_LATE：所有tile采用同一编码器，获得zF后四路gate联合预测。
- VOX_FILM：用BASE上下文771→128→192，为32／64通道两个体素块产生共192个γ／β，先调制空间特征，再汇聚与gate；其通道尺度选择为分子条件化，来自F6通用方法。
- LATE也保留同形调制模块，输入固定零上下文；不能因关闭条件化删除计算路径或改变网络宽度。

训练每个样本使用预登记的24种立方体轴向旋转之一，seed和epoch决定，直接做网格转置／翻转；同一样本的全部tile、tile中心与RU几何使用同一旋转，两体素臂共享同一变换。此操作不等于任意SO(3)不变。正式TEST固定原场一次，不做旋转集成或事后增加TTA。

## 7. 路线C：连续场查询与轻量隐式解码正则

### 7.1 无需体素化的连续查询

查询点由物理原子位置，以及8Å内原子对的1/4、1/2、3/4插值点构成；按原物理身份去重后，用距离FPS选最多1,024点，不足则padding。插值点随分子刚体变换，避免依赖全局坐标轴的固定查询方向。

每点读取三种解析代理场、八个元素族最近距离及其存在mask，并保留到分子／中心RU质心的距离。最近距离以16Å封顶，缺失元素为16Å并置mask；这不是丢弃样本。

由距离FPS选最多64个查询中心，聚合其邻近查询，得到128维场token；局部块与token注意力只使用距离RBF和标量特征。让BASE三个模态作为query读取这些token，再与场全局表示联合预测。该路线是连续解析场的采样编码，不宣称已训练出HyDiF基础模型。

### 7.2 避免条件解码的坐标歧义

若global表示是旋转不变的，却要求它重建任意朝向的xyz场，会产生不可辨识问题。因此小型INR不用未定规范坐标的xyz作为唯一输入。

预先从物理原子中以距离FPS选最多4个锚点，按原身份打破并列；查询坐标定义为到这些锚点的四个物理距离，附锚点可用mask，记为ξ(q)。不足4原子时保留真实锚点并padding。它是连续、刚体不变的距离坐标；退化锚点不能支持唯一恢复任意三维向量场，本轮只重建标量代理场。

### 7.3 同形辅助目标消融

两连续场臂训练时共用同一80%上下文／20%重建查询划分，划分由fold／epoch／sample_key决定；context编码器只看80%查询的场数值，held-out查询的场值仅用于辅助真值。

共享三层64宽的条件INR：

$$
\widehat\rho(q)=D_\theta\!\left(\xi(q);
\operatorname{FiLM}_\eta[z_2,z_3,z_{\mathrm{FP}},z_F]\right).
$$

解码器不读held-out点的ρ、最近元素距离或其他可直接恢复目标的场值。训练上下文是一个全局瓶颈，而非将目标逐点传入。

- CONT_QUERY：相同场编码／query融合和同形解码器，辅助权重0。
- CONT_QUERY_INR：额外加入0.05×三代理场标准化后的重建MSE；场目标标准化只用训练上下文／训练查询拟合。
- 两臂每批仍一次optimizer update；解码前向／反向及查询数消费分别登记。validation／TEST属性预测均使用全部已登记查询，不读辅助真值选择checkpoint。

这一小型辅助任务是受F3／F4启发的项目假设，不等价于其生成预训练，也不证明获得真实量子场。它检验的是场重建约束是否帮助当前联合表示。

## 8. 七臂固定矩阵与受控比较

| 顺序／臂 | 场编码 | 融合／辅助目标 | 主要比较 |
|---|---|---|---|
| 0：C_REFT | 无 | 原BASE | 同起点、同训练预算控制 |
| 1：FFT_LATE | 固定FFT＋训练集PCA | 后期gate | 廉价体积描述符增量 |
| 2：FFT_FILM | 与1相同 | BASE条件化 | 2−1检验特征条件化 |
| 3：VOX_LATE | 轻量体素CNN | 后期gate | 学习型体素路线 |
| 4：VOX_FILM | 与3相同 | BASE条件化卷积通道 | 4−3检验空间编码前的调制 |
| 5：CONT_QUERY | 连续场采样编码 | 三模态查询，重建权重0 | 连续场联合预测 |
| 6：CONT_QUERY_INR | 与5相同 | 重建权重0.05 | 6−5检验辅助约束 |

七臂各35套，从同一PT deploy／原task/fold头起训。1／3／5之间同时改变表示、容量与采样方式，只能比较整体方案；没有相配的表示／融合组合，不能声称某一单因素已被证明更优。

## 9. 训练、冻结与正式TEST

固定batch24、drop_last=False、70epochs=head10＋joint60；BASE参数冻结／解冻与原F配方一致，mask_token继续冻结。无新增大规模PT、无额外seed、无epoch或超参搜索。

- AdamW，weight_decay0.02、betas(0.9,0.999)、eps1e-8、clip5。
- head原模态／gate／head lr3e-4；joint原native1e-4、原modal／head1e-3；新增场编码／FiLM／query／INR均3e-4。
- head10 cosine最低3e-5；joint60按T₀10、Tmult1的cosine warm restart、最低1e-6。阶段切换重建一次optimizer。
- 原native dropout0、FP0.2、性质头0.5；新增模块0.1。
- 基础loss=MSE＋0.08·原三方向ALIGN−0.08·Hnative；Hnative为gate前三路可用权重重归一化后的熵。第6臂另加上述0.05场重建项。
- 每epoch完整validation一次，原单位RMSE严格下降更新各task/fold BEST；同分保留较早epoch。
- 原BASE数值／确定性和模块模式继承r6，不同时改变TF32；新增模块隔离RNG。两体素臂仅新增路径BF16的差异已登记；五折seed系列不是额外多seed复现。

245套FT全部完成后按七任务validation等权macro选择单一全局臂。候选须严格优于旧BASE已存validation macro，且各任务validation mean≥BASE−0.010；同分优先C_REFT，再按表中顺序。无合格者BASE_KEEP。

一次冻结FINAL及所有候选／控制的35套BEST、scaler、场参数、训练统计／PCA、query／tile定义、参考与目标，保存selection/final_frozen.json。冻结前不生成新TEST神经特征或读取逐行TEST标签／预测。

冻结后统一正式评估七新臂及原BASE alias，各单元一次。原BASE仅在此时读取既有正式预测，不重新选checkpoint。不按task/fold拼胜者，不在TEST后改场权重、网格、PCA维度、aux系数或旋转策略。

输出七任务R² mean±std(ddof=1)、等权macro、全精度ΔBASE；展示三位小数。按SOTA_RESULT.md列出逐任务最佳参考／模型／差值，解释论文配置包络与协议差异。

## 10. 预算、CPU吞吐与GPU2资源合同

| 项目 | 上限 |
|---|---:|
| 新构象／构面／谱分解／大型PT／量化计算 | 全部0 |
| 原始行级派生sidecar | 3类×5,875=17,625项上限，按已有物理身份去重 |
| CPU旋转几何／场审计 | 固定4样本×3旋转×3类=36项 |
| 每分子CNN tiles／连续查询点 | 64／1,024 |
| train-only PCA | 35套，FFT两臂共享；其他统计按fold和数学定义分别登记 |
| 新FT／epochs／optimizer updates | 245／17,150／392,000 |
| 正式单元／预测行 | 280／47,000，含35 BASE alias |
| 前置CPU suite／额外GPU诊断 | 0／0 |
| 最终CPU suite | 正式阶段后至多1次 |
| 新周期硬截止／准备阶段 | 新授权t0起72h／最多8h |
| 缓存／主机峰值内存 | 96GiB／160GiB |

更新依据为batch24、每臂35单元每epoch合计800批，70epochs为56,000更新。任何新增查询／tile编码前向及aux解码开销不能以optimizer步数相同为由省略，另计点／tile presentations和耗时。

- 仅物理GPU2：编制时UUID GPU-31456dd5-a421-142b-644b-7a80f482dfb9、49,140MiB。CUDA_VISIBLE_DEVICES=2，内部cuda:0，WORLD_SIZE1；控制器隐藏GPU，禁止其他卡。
- 编制时CPU affinity112、主机RAM约251.5GiB。实际按affinity／cgroup取较小可用额度；预留8核供控制／I/O。
- FFT最多32worker，每worker至多3计算线程；按填充后的实际工作集，保证worker总预测内存≤128GiB，不把112个大FFT同时派发。
- 普通场／query／tile准备最多96个单线程worker、BLAS／OpenMP1；与FFT共享CPU和内存令牌，不分别占满。压缩写入设有界队列和磁盘水位，避免生产速度超过落盘。
- GPU worker主线程2、DataLoader4worker且各1线程，prefetch2、pinned memory；训练时从压缩cache读取，不在GPU空闲时串行重新做全分子FFT。
- FT＋validation与正式TEST阶段互斥，并发最多4。先1路，利用正常单元观测按1→2→4提高，预计显存总峰值≤可用80%；不额外benchmark，不改分子batch或tile microbatch。
- 最先3套正常FT覆盖C_REFT、VOX_FILM、CONT_QUERY_INR，均在245套内。conservative ETA=1.5×实测稳定速度推算剩余工作＋2h收尾；包括场准备、I/O、正式输出和最终CPU。超截止即停止，不自动放弃慢路线来保留快路线的正式结论。
- CPU旋转样本从eea、eps、nc、xc各取原indices最小且原3D有效的不同物理样本；固定绕x／y／z轴37°／73°／121°，随分子变换查询与tile几何，记录离散误差。该审计不运行新模型或选模。
- 所有长命令／worker在tmux Uni-Poly独立window，保存cwd／argv／日志／真实exit。/proc遥测竞态结合poll/wait归并；真实worker失败不吞掉、不重试。

这些是拟执行上限，不是已测ETA或已消费预算。失败、部分训练与中止均计账，不自动刷新时钟或退款。

## 11. 未来扩展：不进入本轮候选与预算

### 11.1 预训练HyDiF／MolField表示

先只读核对官方权重、许可证、元素集合、构象／尺度、外部预训练数据重叠与输入schema。当前没有确认其权重能直接覆盖本项目；不把可下载或可运行写成既成事实。

若可用，后续新修订可增加冻结field encoder＋后期融合、field token cross-attention，并设同容量控制。完整HyDiF扩散超网络或MolField训练不自动纳入本周期；借鉴其连续场思想不要求先复现其全部预训练。

### 11.2 真正量化场／SigmaFormer式融合

后续可在明确的CPU量化预算内计算DFT-COSMO的σ-profile、面积／体积，再做53维输入融合；需要登记分子封端／电荷／多重度、构象、基组、收敛和失败掩码。

若只用现有代理场或廉价部分电荷，必须以代理名称报告，不能称COSMO复现。DFT任务和失败重算均需新预算，不能以“CPU尽量多用”为无限计算授权。

### 11.3 进一步组合

谱—体积一致性、原子特征Gaussian splatting、跨场互相关和多构象均可成为后续问题，但本轮不叠加。先确认场输入与融合的增量，再登记组合消融，不能查看TEST后追加。

## 12. 实施边界、停止与交付

只新增本计划自己的configs、field dataset／preprocessor、model wrapper、training／controller／bootstrap及必要测试；原BASE、旧HMR／READOUT、已有cache／checkpoint／结果根只读。允许在新模型联合微调原编码器与替换融合；不改原标签／split，不新增hash或指纹合同。

拟实施文件为configs/ua3d_molecular_field_r1_gpu2.json、src/dataset/ua3d_molecular_field_r1.py、src/modules/ua3d_molecular_field_r1.py、src/training/ua3d_molecular_field_r1.py、scripts/prepare_ua3d_molecular_field_r1.py、scripts/run_ua3d_molecular_field_r1.py、scripts/bootstrap_ua3d_molecular_field_r1.py及tests/test_ua3d_molecular_field_r1.py；目前均未创建。准备前静态核对生产metrics／finite依赖，不能沿用旧READOUT缺失接口。

拟结果根results/ua3d_molecular_field_20261104/r1/，日志根logs/ua3d_molecular_field_20261104/r1/；目前不创建。已存在同名现场即停止，不覆盖或自动r2。

停止条件：身份／单位／schema不匹配；有效场或梯度非有限；线性／周期FFT混淆；体素截断、tile丢失、cache／内存／ETA超限；PCA／aux用了非训练数据；writer冲突、OOM或真实worker非零退出。停止后保留原exit／消费，不在线修改分辨率、删样本、缩batch、换卡重试。

唯一最终CPU suite在正式阶段后集中验证：

- Gaussian通道／单位、非周期FFT与小数组直接相关的一致性、overflow径向箱、训练集PCA作用域。
- tile完整覆盖、chunk聚合与分子batch权重、缺3D屏蔽。
- 连续查询刚体不变量、锚点／padding、held-out目标不进入context、条件解码梯度。
- 生产metrics接口、原单位指标、row-key／split、一次冻结和账本聚合。

交付至少包括sources.json、field_manifest.json、实际diff／参数／分组、按fold统计／PCA、field／tile／query消费、预算与累计账本、BEST／history、一次冻结清单、正式预测、final_review/results.json与delivery_summary.md。

说明完成层级、七任务／论文对照、全精度macro／差值、目标与默认门槛、真实controller／outer／bootstrap exit、资源与未执行内容；independent_review=PENDING。任何正式TEST不完整的周期写“未完成/不可比较”。

若一并执行三方向，需先登记总预算和统一的validation选路线／一次全局冻结协议，不得逐方向看TEST后继续择优。有限Trimer的任何场仍不能代表真实聚合物堆积或完整体相机制。
