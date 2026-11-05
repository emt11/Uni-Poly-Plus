# 研究计划一：BASE 条件化的表面谱传播与联合预测

> 计划 ID：UA3D-COND-SPECTRAL-20261104-01；修订 r2；资源修订 gpu2-pt1-fttest4-fresh-condspec-r2；编制日期 2026-11-04。状态：**已按用户“重新预训练 BASE，不复用”修订，尚未实施或运行**。
>
> 本次动作是修订计划并提供执行提示词，未启动训练。r2撤销r1“PT=0、继承旧deploy／任务头、旧BASE可回退”的条款；r1未实施，本次只读确认其拟结果根不存在。本文不替代 [CURRENT_PLAN.md](../../CURRENT_PLAN.md)，不恢复已停止的HMR／READOUT。未来执行时，先保留旧CURRENT_PLAN终态并将本修订登记到CURRENT_PLAN.md，再由获用户执行授权的DSH实施；Codex负责规划与独立审查。
>
> 同组计划：[局部表面融合](BASE_LOCAL_SURFACE_PLAN.md)、[体素与连续场](BASE_VOXEL_CONTINUOUS_FIELD_PLAN.md)。三份是可分别选择的科学周期，不表示已授权连续执行三个周期。

## 1. 版本、reference 与目标

- 基于 dev / 2992e3cd095ba636c0f120251c43ca435ae3ea89 及编制时受保护的 dirty worktree。已查看分支、状态、相关文档 diff 和模型／数据接口；不将未跟踪的旧实现归为本计划产物。
- reference 固定为 F_PD_N1_F0 的七任务五折 BEST 集，正式 macro R² 完整值为 0.804675462276949，展示为 **0.805**；independent_review=PENDING。
- BASE 根 B：results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/。
- **所有模型参数从本轮新随机初始化开始。** 不加载历史common_step0、PT／FT／BEST／deploy、mask heads、modal权重或性质头；也不加载旧BASE的z／神经特征或教师预测缓存。原代码定义可以引用，旧模型张量不作为输入。
- 本轮先新训1套BASE完整10K PT，再以本轮新生成的同一PT终点初始化六臂。这是为公平对照共享本轮初态，不是继承历史训练结果；不是每臂各做1次PT。
- 各task/fold新随机初始化35套性质头，并在六臂间共享对应初态；shuffle／训练seed仍为42+fold。新头初始化seed固定为2026110400＋10×task_index＋fold，task_index按本文七任务顺序。
- 只读复用原始数据、split、物理几何／表面谱与同fold的train-only target scaler。数据与scaler来源从B/finetune/F_PD_N1_F0/{task}/fold{fold}/config.json解析；不得据此调用旧的模型初始化入口。
- 新PT／FT schema分别为ua3d-cond-spectral-pretrain-r2／ua3d-cond-spectral-best-r2，显式验证本计划ID、r2、来源、step及键集合；不能冒充旧r6包。只移除本轮已登记的PT辅助头；禁止以strict=False吞掉未解释的缺键／多键。
- 新控制组名C_FRESH；它不替换用户指定的历史BASE reference。候选相对C_FRESH衡量本轮表面增量，相对旧BASE的完整精度差值衡量历史数值进展，两者分开。
- 研究目标：完整正式 TEST macro R² ≥0.810；记录 target_met。项目默认性能门槛另为 ΔBASE≥0.010，即未舍入 macro≥0.814675462276949；记录 performance_pass。两者分开，达到 0.810 不自动写成默认性能通过。
- 共享 validation／TEST 已用于路线开发，本周期属于共享开发评价，不称独立盲测。

## 2. 科学问题与已有证据

**主要问题：分子的 BASE 表示能否决定其表面上的有效传播尺度，并通过联合训练改善性质预测？**

主模型直接预测 y，不采用 ŷBASE＋Δ，不把 BASE 预测值作为输入，也不使用教师预测蒸馏。可以修改原三模态融合层、增加表面编码器，并在联合阶段更新原 2D／3D 编码器。

现有 HMR/r3 冻结 S_FIXED 正式 macro 为 0.808，原值 0.8076857278302016；其可学习滤波与固定滤波结果接近，且缺少已完成的 BASE-only 残差匹配对照。这只支持重新检验传播与融合，不能证明表面无效或条件化有效。旧 READOUT/r1 因指标接口缺失停止，未产生新性能结论。

本轮预登记假设：

1. 在相同表面、读出和训练条件下，按分子决定扩散时间优于全体分子共享时间。
2. 条件化传播与模态 token 交互可能互补；通过 2×2 对照检验，不能由合计提升推断交互作用。
3. 谱通道的相关关系可能补充单通道能量；另设去交叉项对照。

## 3. 论文依据及迁移边界

| 编号与论文 | 已有算法／证据 | 本计划采用什么 |
|---|---|---|
| S1：[HMR: Learning Harmonic Molecular Representations on Riemannian Manifold](https://arxiv.org/abs/2303.15520)，ICLR 2023，较早基础 | 分子表面上的谐波表示与谱传播，包含 QM9 回归。 | 表面特征与谱滤波的分子任务依据。 |
| S2：[DiffusionNet: Discretization Agnostic Learning on Surfaces](https://nmwsharp.com/research/diffusion-net/)，SIGGRAPH 2022，较早基础 | 学习连续扩散时间，控制曲面传播范围。 | 可学习热核与物理传播尺度的算子依据。 |
| S3：[AtomSurf: Surface Representation for Learning on Protein Structures](https://arxiv.org/html/2309.16519)，ICLR 2025，首稿 2023 | 讨论分子物理尺寸与曲面扩散尺度，并融合图和表面。 | 不盲目归一化尺寸；把化学表示和表面传播联系起来。 |
| S4：[FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)，AAAI 2018，通用条件化基础 | 条件信息控制网络中的逐通道变换。 | 说明条件化算子的设计来源，不作为分子预测性能证据。 |
| S5：[SurfBind: Deciphering Fingerprints of 3D Molecular Surfaces for Accurate Epitope Prediction](https://arxiv.org/html/2606.23830v1)，2026 作者稿 | 条件交叉注意力和表面 patch 建模，用于表位预测。 | 为表面与化学 token 交互提供参考；任务与聚合物回归不同。 |

**文献支持不等于本方案已发表或已验证。** 用 BASE 产生逐分子扩散时间、下面的紧凑跨通道读出及其组合，均为本项目提出的适配假设。热核扩散与生成模型的去噪扩散不是同一算法。

## 4. 数据与可复用资产

- 任务固定 Eea、Egb、Egc、Ei、EPS、Nc、Xc；内部名称 eea、egb、egc、ei、eps、nc、xc。
- split 固定 data/splits/mips_paper5_outer/{task}.json，fold0—4，seed=42+fold。共 5,875 原行；跨 35 单元 train／validation／TEST 行数分别为 18,786／4,714／5,875。
- 按原 indices、sample_keys 与 split 对齐，不用数组位置代替身份；目标标准化仅继承该 fold 的原训练统计。
- 只读谱源 P2：results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2/。
- 复用 P2/surface/compact/{task}.npz 中 λ、C、a、A、Ac、stat、mode_mask 和原可用性。定义 C=ΦᵀMF、a=ΦᵀMw；F 为现有 16 通道，w 为中心单元表面归属。
- 本轮重建构象、构面、求谱预算均为零。已有 64 模截断、分量零模和合法负哨兵语义保留；不因新约束拒绝合法的 center_physical=-1。
- 原缺 3D 行与已有谱失败行全部保留。缺失表面时屏蔽表面 token，使用本轮模型的可用原模态预测；不宣称精确回退旧 BASE 数值。
- 不沿用旧下游 BEST 的 z 缓存作为联合训练输入。z 必须由当前模型产生；旧缓存只可作字段定义参考。
- 原 stat16 的均值／标准差只在各 fold 有效训练行拟合；动态 u／q／G 拼接后使用逐样本 LayerNorm，不拟合随网络参数变化的谱输出均值，也不沿用旧 spec scaler。

### 4.1 必须完成的全新BASE预训练

1. seed=42，新构造原512维、6层2D／3D编码器、norm、A2／A3、FP MLP、共享Q2／Q3及两类mask head；全部从新随机初始化产生，不调用会读历史权重的load_initial／initialize_ft入口。不直接运行含旧初始化／alias分支的历史训练脚本，只引用纯模型、数学和数据接口。本轮step0保存到新根initialization/common_step0.pt，只作为本轮记录。
2. 无标签池保持原910,756条有效记录、635条无效几何不进入任何目标。原PT模板来自results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache，映射来自results/ua2d_ptscale_20261022/r1/inputs；这些仅含原始／确定性特征，资格阶段确认不含学得的BASE表示。
3. 固定1套PT，10,000 optimizer updates，microbatch168、accumulation6、global batch1008，合计60,000 microbatches和10,080,000样本呈现。从global position0开始，既有按position决定的采样／mask／噪声语义不变，不从某个历史step继续。
4. 原BASE的PT配方：native dropout0.1、FP dropout0；mask2D＋mask3D，三方向2D→FP、3D→2D、FP→3D，各ALIGN权重0.08并在前1,000更新warmup；FP只作输入，没有FP预测头或重建目标。
5. mask比例0.4，mask／random／keep为0.8／0.1／0.1，原3D噪声σ=0.03Å；这是原预训练扰动，不计作新构象生成。当前表面条件化模块在下游联合阶段训练，不另做六套表面PT。
6. PT使用BF16前向、FP32归一化／loss／优化器状态；AdamW，peak lr2e-4、weight_decay0、betas(0.9,0.999)、eps1e-8、clip1。两个连续5K学习率周期，各warmup500、最低1e-6；跨5K不重建优化器。
7. 原三个模态RNG计数从0到60,000，各分支终点都必须为60,000；不能为诊断额外消耗PT前向。保存本轮step0、step5000、step10000／deploy及真实消费；不自动恢复中间checkpoint。
8. 35个task/fold的FT只加载本轮pretrain/BASE_FRESH/deploy.pt和本轮新头，全部独立训练70epochs。历史BASE的权重／逐行预测不成为候选、初始化或回退；历史结果只读已公布汇总用于对照。

## 5. 主架构：条件化传播先于表面汇聚

### 5.1 BASE 化学上下文

保留原 2D／3D／FP 三分支和 256 维接口。上下文来自三个投影后的表示，加入原可用性掩码；不读取 y、ŷBASE、validation 标签或任务测试统计。

每模态 LayerNorm 后拼接三个向量及availability3，再由 771→128→64 的 MLP 产生条件偏移。缺失 3D 表示在进入上下文前仍严格置零。条件网络在同一 task 的五折分别训练。

### 5.2 逐分子、逐通道扩散时间

令 d=64，条件化模型采用：

$$
t_c(z)=t_{\min}+(t_{\max}-t_{\min})
\operatorname{sigmoid}\!\left(\beta_c+f_\theta(z)_c\right),
\qquad t_{\min}=0.0625,\quad t_{\max}=64\ {\rm \AA^2}.
$$

64 通道分四组，初始时间分别为 0.25、1、4、16 Å²；条件 MLP 输出层置零，使条件化与共享时间对照的初始滤波一致。其余层正常初始化，β 可学习。

共享时间对照具有同形条件 MLP，但输入为固定零上下文；仍执行相同计算。它不是完全相同的有效函数容量，报告时说明。

令 W 为 16→64 无偏置投影，传播后的谱系数为：

$$
B_{kc}(z)=e^{-\lambda_k t_c(z)}
\sum_d C_{kd}W_{dc}.
$$

这等价于对投影后的表面特征进行热传播，再做下述读出；不需要在线恢复全部网格顶点。mode_mask 在指数和汇聚中一致生效。

λ 保留 Å⁻²，时间保留 Å²，不做单位表面积缩放。重根子空间使用一致谱权重与已有分组规则，禁止直接将有符号特征向量拼到 MLP。

### 5.3 不能只做全局平均

闭合曲面的纯热扩散保持面积积分；只做全局面积均值，时间参数可能无法影响输出。因此采用三个互补读出：

$$
u_c=\frac{\sum_k a_kB_{kc}}{A_c},\qquad
q_c=\frac{\sum_{\lambda_k>0}B_{kc}^2}{A},\qquad
G=\frac{(BP)^\top M_+(BP)}{A}.
$$

P 为 64→8 的无偏置投影，M₊ 是仅保留正模的对角掩码。这里 M₊ 不是网格质量矩阵。G 取上三角共 36 维；u、q 各 64 维，合计164维先作LayerNorm，再拼训练集标准化的stat16，合计180维，经180→128→256表面头得到zS。

u 提供中心单元加权，q 与 G 保留非恒定模式。记录时间梯度、边界饱和率、非零模能量及通道相关，作为机制诊断；这些量不替代预测性能。

### 5.4 两种联合预测融合

**GATE：四模态门控。** 用 z₂、z₃、zFP、zS 及可用性产生四路权重，融合为 256 维，再由性质头预测。原 gate 的公共参数显式复制，新增列正常登记；第四路 softmax 会改变起始预测，不宣称初始化严格等于 BASE。

**TOKEN：四 token 自注意力＋同形门控。** 四个 256 维 token 经一个 4-head、FFN512 的 Transformer block，再使用相同四路门控和性质头。固定 dropout=0.1；不扩大层数搜索。

两者都输出 H(fjoint)，没有额外 ŷBASE 支路。内部标准神经网络残差连接不等同于旧的预测残差合同。

## 6. 固定实验矩阵

| 顺序／臂 | 时间条件 | 表面交叉项 | 融合 | 用途 |
|---|---|---|---|---|
| 0：C_FRESH | 无表面 | 无 | 原三路 gate | 本轮全新PT、同初始化与预算的BASE配方控制 |
| 1：S_STATIC_GATE | 固定上下文，共享时间 | 有 | 四路 GATE | 条件化的直接对照 |
| 2：S_COND_GATE | 当前分子的 BASE 上下文 | 有 | 四路 GATE | 主候选 |
| 3：S_STATIC_TOKEN | 固定上下文，共享时间 | 有 | 四 token TOKEN | 2×2 对照 |
| 4：S_COND_TOKEN | 当前分子的 BASE 上下文 | 有 | 四 token TOKEN | 条件化＋交互候选 |
| 5：S_COND_NOCROSS | 当前分子的 BASE 上下文 | G 的36维置零 | 四路 GATE | 跨通道项消融，保留输入形状 |

每臂完整7×5，均从本轮新BASE PT deploy和本轮新随机任务头起训；不从历史或其他臂BEST继续。对应公共参数初态、shuffle、dropout seed保持一致，新增模块在隔离RNG中初始化。

主要配对：2−1、4−3 检验条件化；(4−3)−(2−1) 描述此设置下的融合交互；2−5 检验交叉读出；各候选−0 为完整架构变化，不能称单因素因果。置零消融存在有效容量差异，不以参数数量相同声称完全匹配。

## 7. 训练配方与初始化

固定 batch24、drop_last=False、FP32，全部 70 epochs：head10＋joint60。每个数据批只做该模型必要的正常前向与一次更新，不额外启动 smoke／速度测试。

- head10：冻结原 o8、norm2、encoder3d、norm3 参数；训练模态投影／FP、融合、性质头及新谱模块。保留原 head 阶段模式语义。
- joint60：解冻上述 BASE 模块，原 encoder3d.mask_token 继续冻结；时间由当前 BASE 表示决定，性质损失能反传到原编码器。
- AdamW，weight_decay=0.02，betas=(0.9,0.999)，eps=1e-8，梯度 clip=5。
- 原 modal／fusion／head 学习率 head=3e-4、joint=1e-3；native joint=1e-4；新表面／条件网络 head=3e-4、joint=3e-4，β joint=1e-4。
- head 用 10-epoch cosine，最低 3e-5；joint 用 T₀=10、Tmult=1 的 cosine warm restart，各组最低 1e-6。优化器在阶段切换时重建，阶段内不重置。
- 原 dropout：2D=0、3D=0、FP=0.2、性质头=0.5；新增模块=0.1。
- 原三模态 ALIGN 保留 0.08，方向 2D→FP、3D→2D、FP→3D；不临时加入表面对比目标。
- loss=MSE＋0.08·ALIGN−0.08·Hnative。四模态时Hnative由四路权重的前三路在原可用模态内重新归一化后计算，避免因增加一路而抬高最大熵；C_FRESH为原三路熵。TOKEN采用其末端同形gate的权重。
- 每 epoch 一次完整 validation，按原单位 RMSE 严格下降保存 task/fold BEST；同分保留较早 epoch。保存完整 history，不提前停止或追加 epoch。
- 随机性／确定性、数据顺序和原模块模式沿用r6生产配方，新增模块隔离RNG；不得借本轮同时调整TF32等数值设置。五折seed联动不是额外五次独立seed复现。

## 8. 一次冻结与正式评价

1. 六臂全部 210 套完整训练并核对消费后，以每任务五折 R² 均值、再七任务等权得到 validation macro。
2. 本轮开发控制为C_FRESH：表面候选须严格优于C_FRESH的validation macro，且各任务validation mean≥C_FRESH−0.010；从合格者取最高全精度macro，精确同分按矩阵顺序。无合格表面候选则FINAL=C_FRESH，明确表面开发门槛未通过；不回退旧BASE。C_FRESH只用于本轮选择控制，不改第1节历史性能reference。
3. 冻结一个全局配方及其 35 套 BEST，连同所有对照 checkpoint、epoch、scaler、配置、候选列表、规则和 reference，写入 selection/final_frozen.json。不能按 task/fold 拼不同臂。
4. 冻结前不读逐行TEST标签、预测，不生成任何新TEST模型表示；冻结后统一完成六新臂正式评价，共210单元，各一次。不加载历史BASE权重或逐行预测，不创建BASE alias正式输出；历史对照直接引用已公布汇总及证据路径。
5. 使用同一 split、完整样本、原单位 R²，报告每任务 mean±std(ddof=1)、等权 macro、相对 BASE 的未舍入差值；展示均保留三位小数。
6. 按 SOTA_RESULT.md 比较七任务逐任务最佳均值、对应模型和差值。论文包络不是单模型，split 等价性未证实时仅作数值对照。
7. 正式结果出来后不得修改时间范围、融合权重、候选、epoch 或 alpha。全部单元正式评价后运行唯一最终 CPU suite；通过也只交付待独立审查。

## 9. 固定预算与单卡速度合同

| 项目 | 本周期上限 |
|---|---:|
| 新PT | 1套，从新随机step0完整训练10,000更新 |
| PT microbatches／样本呈现 | 60,000／10,080,000 |
| 新构象／构面／求谱 | 0／0／0 |
| 新随机性质头初态 | 35套，各在六臂间共享 |
| 新 FT | 6×35=210 套 |
| 训练 epochs | 14,700 |
| FT optimizer updates | 336,000 |
| PT＋FT optimizer updates | 346,000，含失败单元的实际尝试，不退款 |
| 正式单元 | 210，全部为本轮新模型 |
| 正式预测行 | 35,250；不创建旧BASE alias或重复TEST |
| train-only stat16 拟合 | 35套；不拟合动态谱输出统计 |
| 前置 CPU suite／额外 GPU 诊断 | 0／0 |
| 最终 CPU suite | 至多1次，正式阶段后 |
| 总时钟／准备阶段／PT阶段 | 新授权controller t0起72h／准备≤4h／PT起点后≤12h |

训练步数依据现有 split：batch24 时 35 单元每 epoch 共800批，单臂70epochs为56,000更新。未来正常元数据核对若不同，停止，不悄悄修改上限。

- 仅物理 GPU2，编制时 UUID GPU-31456dd5-a421-142b-644b-7a80f482dfb9，报告名称 RTX4090、显存49,140MiB。启动时再次只读确认；CUDA_VISIBLE_DEVICES=2，内部 cuda:0，WORLD_SIZE=1，禁止其他卡。
- 编制时 CPU affinity=112、主机 RAM≈251.5GiB；运行以 affinity／cgroup 实际额度为准。控制器不见 GPU。CPU 数字不是已授权新预算消费。
- PT仅1个GPU worker，main_threads2、DataLoader8worker、每worker／BLAS1线程、prefetch2。PT种子由global position决定，worker并行不能改变样本顺序／mask／噪声；prefetched与consumed分开记账。PT与FT／validation／TEST阶段互斥。
- FT＋validation／正式 TEST 的并发上限均4，两个阶段互斥；初始1路，以最先正常 FT 的已发生显存／吞吐观测，按1→2→4提高并发。总预计峰值须≤可用显存80%，否则保持当前并发；不新增探针、不改 batch、不因 OOM 降 batch 重试。
- FT／TEST每GPU worker主线程2、DataLoader4worker，每loader／BLAS1线程，prefetch2。PT与FT均实现自定义ModalBatch／图／标签的实际递归pin_memory与non_blocking传输，不能只设置无效的pin_memory=True；使用有界预取隐藏CPU解码／H2D。参数变化期间不缓存失效的z。
- 35套来源／训练统计可在CPU并行处理，普通准备worker至多96、每个1线程，按实际内存和I/O限制；本周期主机峰值上限160GiB。GPU阶段优先并行准备原图与批图，不重复构面或增加无用前向来提高利用率。
- PT前100次正常更新用于采集稳定时间，不增加诊断；排除首20次冷启动，使用后80次统计，随后每500更新刷新。保守PT剩余估计超出PT的12h阶段截止或总72h截止即停止。
- FT前6个正常单元固定为六臂各自EPS/fold0，串行完整训练70epochs，直接成为各自35套中的正式单元，不另做试跑、不重训。以joint阶段完整epoch训练墙钟／批数的稳定中位数比较同task/fold／同并发，剔除首2个joint epoch，正常epoch边界同步计时，不逐batch额外同步：五个表面臂各不得超过C_FRESH的2.0倍，否则按成本门禁停止。
- 六套通过后再按显存门禁提高FT并发。任务间成本先用历史r6对应task/fold的时间元数据作先验、结合本轮倍率上界；取得本轮相应任务实测后替换，不把EPS速度直接推广所有任务。
- conservative ETA=1.5×按实测稳定吞吐／实际并发推算的剩余时长＋固定2h收尾，包含验证、TEST、最终CPU和交付；每个PT观察窗口／FT单元结束更新。超过硬截止即停止，不能砍慢臂、减fold或改batch凑完整。
- 工程目标为在24h内交付，硬上限仍72h；目标不是实测耗时承诺。历史同配方PT消费记录为13,446.190360797002秒、10K更新，约3.735h；只作成本先验，出处B/pretrain/F_PD_N1_F0/consumption.json，不据此加载模型或承诺新运行时间。
- 所有 GPU／worker 命令在 tmux session Uni-Poly 的独立 window 中执行，记录 cwd、argv、日志、PID、controller／outer／bootstrap真实exit。/proc 的 ENOENT／ESRCH 仅表示遥测对象已退出，须结合 poll/wait 归并，不能掩盖真实 worker 错误。

## 10. 修改边界、停止条件与交付

实施时仅新增本计划自己的 configs、dataset sidecar、model wrapper、training、prepare／controller／bootstrap 和必要测试文件；原 BASE、旧 HMR／READOUT 源码及所有旧结果根只读。允许在新 wrapper 中替换融合、微调全部 BASE 参数，禁止修改旧 checkpoint、标签、split 或共享缓存。

拟实施文件为configs/ua3d_cond_spectral_r2_gpu2.json、src/dataset/ua3d_cond_spectral_r2.py、src/modules/ua3d_cond_spectral_r2.py、src/training/ua3d_cond_spectral_r2.py、scripts/prepare_ua3d_cond_spectral_r2.py、scripts/run_ua3d_cond_spectral_r2.py、scripts/bootstrap_ua3d_cond_spectral_r2.py及tests/test_ua3d_cond_spectral_r2.py；此为文件规划，目前均未创建。实施前静态核对真实epoch0／validation metrics依赖，确认finite helper、schema和导入均有定义，不复制旧READOUT缺失接口。

拟产物根R=results/ua3d_cond_spectral_20261104/r2/；日志根logs/ua3d_cond_spectral_20261104/r2/。新PT为R/pretrain/BASE_FRESH/deploy.pt，新头为R/initialization/heads/{task}/fold{fold}.pt。目前不创建。已存在同名现场时停止，不覆盖或自动换修订。r1未执行，不将其标成失败或虚构父消费。

必须停止的情况：来源 schema／身份不一致、有效数值 NaN/Inf、时间梯度因detach／no_grad等实现断链而不可达、writer冲突、显存溢出、预算／ETA超限、任何 worker 非零退出。正常零初始化导致的首步零梯度不单独判为断链。普通 /proc 采样竞态按上节处理。不得自动修复重试、换卡、扩大缺失名单或刷新旧父时钟。

最终 CPU suite 验证：紧凑读出与小型显式热算子的一致性；重根旋转／符号不变性；条件时间的梯度通路；缺模态屏蔽；真实生产 metrics 接口；split／row-key、冻结及预算聚合。只在最后集中运行一次；此前以静态接口审查和正常训练内的断言发现问题。

交付至少包含：

- implementation_diff.md：实际文件、架构、参数与优化器分组、文献借鉴和本项目新增部分。
- sources.json、budget.json、cumulative_budget.json、consumption.json：历史仅作证据引用，本轮随机初始化／新PT与FT消费分栏；失败尝试计费，未知历史值保持未知。
- 新step0、35新头、PT5000／10000终点及其完整计数；明确历史模型权重／z／教师缓存读取为0。
- selection/final_frozen.json、35折 BEST 定位、原单位正式预测与 final_review/results.json。
- 七任务对照、全精度 macro／Δ、目标与默认性能门槛两个标志、阶段与真实 exit。
- 传播时间分布／梯度／饱和、交叉项及2×2配对结果，不把机制图当性能证据。
- delivery_summary.md；停止时另有 first_failure.json 与 failure_closeout.json。未完成正式 TEST 写“未完成/不可比较”；完整交付仍为 independent_review=PENDING。

若三方向一并执行，必须先登记三方向总预算和统一的 validation 选路线／一次全局冻结规则；不能逐个查看 TEST 后再决定下一方向。本文初始范围不追加 FiLM 候选、更多时间范围、种子或顶点级多层 DiffusionNet；这些是后续新修订的问题。
