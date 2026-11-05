# 当前计划：先稳定 BASE 的重训质量与跨种子表现

> 计划 ID：UA3D-BASE-STABILIZE-20261105-01；revision=r1；resource_revision=gpu2-pt1-fttest4-base-stabilize-r1。
>
> 状态：**RUNNING**；阶段 PT；更新时间 2026-11-05T18:59:30.596286+08:00。
>
> 用户最新要求为“先稳定当前 BASE”。本周期暂停表面谱、局部表面／点云、体素与分子场融合。旧条件谱 r6 为 COMPLETE_PENDING_REVIEW，性能未通过；不再作工程 r7 接续或事后改选。
>
> 当前执行合同统一保存在本文件。2026-11-05 按用户要求整理 docs，仅保留下列三份图形学研究文档；执行计划镜像、提示词、审查文档及旧快照从 docs 删除。原启动授权文本已保存在 [authorization](results/ua3d_base_stabilize_20261105/r1/authorization/)，结果与日志保留。本次整理不改变既有执行合同、预算、资源、状态或冻结规则。

<!-- BASE_STABILIZE_R1_STATUS_START -->
2026-11-05T18:59:30.596317+08:00 RUNNING／PT：两个独立seed42完整10K，从同一新step0串行；DET不可降级。
independent_review=PENDING；BASE不替换。
<!-- BASE_STABILIZE_R1_STATUS_END -->

## 保留的三个图形学研究方向

| 研究方向 | 唯一研究文档 | 当前安排 |
|---|---|---|
| 条件化表面谱传播与融合 | [BASE_CONDITIONAL_SPECTRAL_PLAN.md](docs/research/BASE_CONDITIONAL_SPECTRAL_PLAN.md) | 保留原 r2 研究方案；后续 r6 已交付待审查、性能未通过，当前暂停，不接续或重跑。 |
| 局部表面／点云与原子／键交互 | [BASE_LOCAL_SURFACE_PLAN.md](docs/research/BASE_LOCAL_SURFACE_PLAN.md) | 研究计划，尚未实施；当前暂停，恢复前另行明确授权。 |
| 体素与连续分子场 | [BASE_VOXEL_CONTINUOUS_FIELD_PLAN.md](docs/research/BASE_VOXEL_CONTINUOUS_FIELD_PLAN.md) | 研究计划，尚未实施；当前暂停，恢复前另行明确授权。 |

三份文档保留各自编制时的原文与版本，实际进展和当前授权以本文件及运行产物为准。正在执行的是以下 BASE 稳定化 r1；已有 [status.json](results/ua3d_base_stabilize_20261105/r1/status.json) 记录 RUNNING／PT，[prepare_consumption.json](results/ua3d_base_stabilize_20261105/r1/prepare_consumption.json) 记录 prepare 完成。尚无本周期完整正式 TEST，不作性能结论。

## 1. 基于版本、历史证据与问题定义

仓库 /root/workspace/Uni-Poly-Plus-master；基于 dev / 2992e3cd095ba636c0f120251c43ca435ae3ea89 与当前受保护 dirty worktree。已核对分支、status、相关差异、CURRENT_PLAN、原 BASE 配置和实现、旧稳定性／重复推理产物，以及条件谱 r6 的完整精度指标、冻结时序、预算和真实 exit。不得将任何未提交文件视为可覆盖的临时文件。

以下路径均相对于仓库根：

- B=results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1/。
- T=results/ua3d_base_test_repeat_20261029/r2_gpu3/。
- D=results/ua3d_stability_20261028/r1/。
- P=results/ua3d_cond_spectral_20261104/r2/；P3=P/repairs/memory_r3/；P6=P/repairs/mm_lifecycle_r6/。
- R=results/ua3d_base_stabilize_20261105/r1/；日志根 logs/ua3d_base_stabilize_20261105/r1/。两根在原计划编制时不存在，现已由本周期执行创建，继续按原合同保护，不覆盖或另起运行。

历史核对的直接证据入口：T/results.json、D/final_review/results.json、P6/final_review/results.json、P6/execution_audit.json 与 P6/delivery_addendum.md。以下结论保留原审查时的证据边界；本次文档整理未重跑模型或补发独立审查结论。

既有证据必须用于缩小问题，不能重新把推理、同种子重训和跨种子泛化统称“随机不稳定”：

| 对象 | 已核实结果 | 实际含义 |
|---|---|---|
| 指定 BASE：F_PD_N1_F0 | macro R²=0.805；原值 0.804675462276949 | 继续作为唯一性能 reference，未被替换。 |
| 原35个 BEST 的重复 TEST | 原值 0.8046754656116685；差值 3.3347195005362096e-09 | 同权重推理数值基本稳定，但非逐位相同；该次在 GPU3 的历史授权不允许本轮使用 GPU3。 |
| 旧 DET 稳定性实验 | 35/35 同种子训练对比较通过；三组 macro=0.790/0.788/0.789 | 确定性已在旧 FT 路径得到证据；重训质量与部分任务跨种子表现仍不合格。 |
| 条件谱 r6 无表面对照 C_FRESH | 原值 0.7870456115644041；validation=0.8101093105117524 | 新底座的正式质量下降；较高 validation 没有保证正式 TEST 复现。 |
| 条件谱 r6 冻结 S_STATIC_GATE | 原值 0.7884341875644267；ΔBASE=−0.016241274712522236；ΔC_FRESH=+0.0013885760000226233 | 完整正式评价已完成，性能未通过；不能据其他臂的事后 TEST 改选 FINAL。 |

旧 D 的 macro 跨种子 std=0.001、range=0.002（完整精度见D/final_review/results.json），整体并非大幅抖动；但 Ei/EPS/Nc/Xc 的任务五折均值跨种子 range 分别为 0.026/0.051/0.032/0.046。任务之间可能相互抵消，不能只检查 macro。

本周期回答：恢复可追溯的 BASE 随机初始化与训练定义后，是否能通过同种子可重复性检验；固定的后期权重平均能否在不改变架构、损失、更新数的条件下，减小 FT 随机性和 BEST 选点的敏感性，同时保住接近原 BASE 的正式质量？

## 2. Reference、受控变化与证据边界

性能 reference 固定 F_PD_N1_F0 / 0.804675462276949。原 reference 是共享开发 TEST 后由用户指定的模型集，independent_review=PENDING；本周期不能自动替换它。

本周期保持 2D BondPathO8 512维×6层＋有限 Trimer 3D Galformer 512维×6层＋Morgan2048 FP，三路各256维，TMOL3M gate＋性质头。没有新表面输入、教师预测、残差预测头、FP 重建、额外模态或任务。

重建时登记三个变化层次：

1. **生成规则恢复**：使用历史随机工厂与种子，从源码重新生成 step0 和35个性质头；不读取任何历史模型张量。
2. **确定性执行**：在固定软件／硬件与显式数值策略下训练，不把 DET 与原 DEFAULT 数值实现视为严格相同实验。
3. **唯一稳定化因素**：同一条 FT 轨迹同时产生 BEST 与固定 SWA。两者差别只在 checkpoint 估计器；没有额外训练轨迹或改动 LR。

与历史 BASE 或旧 C_FRESH 的差异含初始化生成路径、PT 数值执行、FT 轨迹等多因素，不能把总差值单独归因于某一项。SWA 对 BEST 的配对比较才是本周期的受控估计器比较。

这里的跨种子指 **同一新 PT 参数初态、同一随机头初态下的 FT shuffle／dropout 随机性**。本轮PT使用一个seed=42、独立串行重复两次，检验该种子的完整PT可重复性；没有多个不同PT seed，不能据此宣称PT种子分布、全初始化分布或任意平台的稳定性。

## 3. 初始化恢复：全新 PT，不复用旧权重

先在 R/recipe_audit.json 记录35份 B/finetune/F_PD_N1_F0/{task}/fold{fold}/config.json 的实际路径、数据角色、split、scaler、架构、dropout、优化器、LR、头生成规则和运行差异。已有指标可读；不重新生成旧预测，不读取逐行 TEST 作为调参输入。

静态核对发现的差异必须显式处理：

- P3 的 FreshPT 在原 modal_modules 构造后又 reset_parameters，破坏了原模态随机工厂的种子隔离；本轮不采用该覆盖逻辑。
- P3 头种子为 2026110400+10×task_index+fold；原 BASE 的头生成规则为 20261025+100×task_index+fold。本轮恢复后者。
- 原 BASE FT DataLoader 为0 worker，条件谱为4 worker。本轮 FT／validation／TEST 统一恢复0 worker；不能直接声称 worker 数就是降分原因。
- 原公共 step0 的 native／mask 头来自 UA2DGalformerPretrainer 的随机构造和显式模块抽取。直接构造 PaperFuse 的路径不能自动当作等价工厂；有无数值差异必须由生成流程与新产物的正常初始化断言界定。

新工厂固定如下，禁止加载旧 common_step0、PT deploy、下游 BEST、P3 PT 或旧随机头：

1. 在受控 CPU RNG 上设置 seed=42，构造 UA2DGalformerPretrainer(PretrainConfig(strategy='mask_align'))；按历史 initialize 的顺序，从本次新生成对象抽取 o8、encoder3d、norm2、norm3、atom_head、line_head。proj2/proj3 只承担原工厂中的 RNG 消费，不导入 BASE。
2. 以 native.modal_modules('TMOL3M') 的隔离 seed=20261025 新生成 A2/A3/fp_encoder/Q2/Q3；保留 Q2 在2D/FP间共享投影的原语义，不执行第二次 reset_parameters。
3. 用显式、严格的键集／shape／dtype 转换组装本周期 step0；写本周期 plan/schema/source_code/seed 事实，不伪装为旧周期 checkpoint。不得 strict=False。
4. 按任务固定顺序 Eea、Egb、Egc、Ei、EPS、Nc、Xc，使用 20261025+100×task_index+fold，在隔离 RNG 中新构造 property_head(0.1) 并保存35份初态。FT 实际 dropout 仍为0.5，参数不因 dropout 概率改变而重置。
5. 四组 FT 对同一 task/fold 使用逐张量一致的新 PT 初态与同一新头；两次完整PT比较通过后才允许此断言。推理用到的 gate 保持历史工厂 seed=20261025 的构造规则；新包必须记录实际组装来源。

初始化为唯一一次正常 prepare 的组成部分，允许检查本次新对象键集、参数属性与复制后的直接张量一致性，不运行 synthetic forward／smoke。历史权重读取数必须为0；旧来源 schema 不能通过改写 plan 标签来骗过旧 loader。

生成规则无法从现有代码和元数据解释，或存在未登记的缺键、形状、任务顺序或 scaler 矛盾时，停止，不猜测补齐。

## 4. 数据、采样与确定性策略

七任务，每 task fold0—4；data/splits/mips_paper5_outer/{task}.json，原数据、原 row/sample_key、缺3D名单及 train-only target scaler 不变。5,875原行全部保留；35单元合计 train18,786、validation4,714、TEST5,875。2D／3D／FP 缺失屏蔽与合法负哨兵保持原义。

可只读使用 B 配置指定的原输入／scaler、PT raw template 与 logical mapping；不重构数据、FP、3D构象、表面、谱或任何新训练统计。PT 模板911,391项、有效 logical slots910,756；原逻辑位点→key→source_row 解析必须一致。

固定运行环境沿用当前兼容栈并记录实际版本（旧稳定性证据为 PyTorch2.8.0+cu128／CUDA12.8／PyG2.7.0）；不升级依赖或换实现来追求分数。若关键版本与现有输入不兼容则停止。

所有 GPU worker 在 Python/CUDA 初始化前固定：

- CUDA_VISIBLE_DEVICES=2、CUDA_DEVICE_ORDER=PCI_BUS_ID、WORLD_SIZE=1。
- CUBLAS_WORKSPACE_CONFIG=:4096:8；PYTHONHASHSEED=20261028；Python／NumPy／torch／CUDA 的实际训练 seed 另按 PT／FT 规则设置。
- torch.use_deterministic_algorithms(True, warn_only=False)；cuDNN benchmark=false、deterministic=true；matmul/cuDNN TF32=false；matmul precision=highest；fill_uninitialized_memory=true。
- 主训练进程 CPU threads=2、interop_threads=1。不得切换 fused optimizer、torch.compile、AMP 精度、批量或 SDPA 后端来绕过门禁。
- 不支持确定性算子时报错即停止，不能设 warn_only、关闭 DET 或在线修改算子重试。旧 FT 的 DET 成功不能保证新 BF16 PT 一定支持所有路径。

PT 保持原 ordered stream、位置决定的 mask／corruption、三路独立 dropout stream 和计数；worker ID／预取时机不得参与样本语义。PT 使用已完成 P3 的 bounded reader 方案：4 spawn worker、prefetch1、每 worker LRU128且 storage估算≤256MiB、readahead=false、每64个源 microbatch安全重开只读 LMDB。它是读取工程变化，不能改变 ordered logical slots。无需重跑旧 prefix 诊断；在正常 PT 消费中记录与核对已有身份定义。

FT loader 保持历史构造位置与 generator 生命周期：seed_base+fold，batch24、shuffle=true、drop_last=false，连续迭代70个 epoch；不是每个 epoch 擅自重设相同 seed。先正确组装 PT/头/模型，再按原位置重设训练 RNG。记录实际 epoch 排序与行身份，避免记录动作额外消耗训练 RNG。

资料依据：[PyTorch2.8 重复性说明](https://docs.pytorch.org/docs/2.8/notes/randomness.html)。同种子和固定平台只是受控条件；DET 不代表不同种子会收敛到相同性能，也不保证跨版本逐位复现。

## 5. 两次同种子新 PT 与四组 FT

PT固定两个独立进程 PT_R0/PT_R1，在GPU2串行执行；均从本周期同一新随机step0开始，各完整训练10,000更新。PT_R1不得从PT_R0终点或step5000接续。以下为每次PT的完整配方：

- P_CURRENT_SHARED：native dropout0.1、PT FP dropout0。
- mask2D＋mask3D；三方向 2D→FP、3D→2D、FP→3D，每方向 ALIGN=0.08；共享 Q2/Q3，fp_prediction=false。
- microbatch168×accumulation6=1008；BF16 forward、FP32 loss／optimizer；seed42。
- 两个连续5K LR cycle，分别 warmup500、peak2e-4、min1e-6；ALIGN 全局 warmup1000只执行一次。
- 完成时10,000更新／60,000 microbatch／10,080,000 consumed presentations，三路 stream counters=[60000,60000,60000]；step5000与step10000及deploy保留。预取未消费尾部单列，不当作完成。
- 新包使用本周期 schema，显式转换到 FT，严格剔除登记的 atom_head/line_head；不能修改旧 validate_package 来接受新包。

两次PT全部正常完成后，对正常产物直接比较：同一step0、全部10K更新的loss与项/计数、已消费ordered位置与mask/corruption定义、step5000/step10000模型参数、终点optimizer/scheduler/RNG/stream状态；若原PT手工设置LR，记录其实际step/phase/LR，不新增scheduler改变算法。保存两份正常终点状态；时间/PID/路径/预取尾部不作为模型数值差异，消费顺序和真实完整更新数必须一致。此比较不启动额外前向或CPU suite；任一数学状态差异即PT_REPRODUCIBILITY_FAILED，在FT前停止，保留第一差异，不重训第三次。

通过后，S42_A/S142/S242只从PT_R0完整deploy起步，S42_B只从逐张量相等的PT_R1完整deploy起步；35个新随机头共同使用。由此检验的是同种子PT+FT链条，而三个不同FT seed仍只对应一个PT seed。

PT比较完成后才运行四组 FT，每组35单元：

| 组 | FT seed | 用途 | 正式 TEST |
|---|---|---|---|
| S42_A | 42+fold | 预登记 primary run | BEST与SWA各35 |
| S42_B | 42+fold | 同种子独立进程重复 | 0；仅重复性比较 |
| S142 | 142+fold | FT随机性检查 | BEST与SWA各35 |
| S242 | 242+fold | FT随机性检查 | BEST与SWA各35 |

每套完整70epochs=head10＋joint60，不早停、不使用中断BEST、不改变头或 PT 初态。FT FP32，native dropout0、FP0.2、head0.5；loss=MSE+0.08×ALIGN−0.08×gate_entropy。

AdamW weight_decay0.02、clip_grad_norm5；head阶段非native lr3e-4，CosineAnnealingLR(T_max10,eta_min3e-5)；joint native lr1e-4、其余1e-3，CosineAnnealingWarmRestarts(T_0=10,T_mult=1,eta_min1e-6)。优化器阶段切换与 scheduler.step 位置保持历史实现。原 native mask_token 不参加 FT 更新。

先串行完成 S42_A/EPS/fold0 与 S42_B/EPS/fold0，作为140套内的两个完整正式训练单元；直接比较正常产物，不额外诊断、不重训。通过后可按2→4并发派发其余单元，优先 EPS/Nc/Xc，全部组／task／fold均必须完成。不得按早期 validation 改预算、LR或删掉弱任务。

## 6. BEST 与固定四点 SWA：同一轨迹的配对对照

BEST 完全沿用“每个正常 epoch 的原单位 validation RMSE 严格下降才更新；相同值保留首次”的规则。

SWA 仅在 joint epoch30/40/50/60 完整训练结束时采集参数，即总 epoch40/50/60/70；四个时点和等权规则在此固定，不根据曲线或 TEST 改变。采集发生在该 epoch 的优化更新完成后，不改变后续训练模型／优化器／scheduler／RNG。

$$
\theta_{\mathrm{SWA}}=\tfrac14\left(\theta_{40}+\theta_{50}+\theta_{60}+\theta_{70}\right).
$$

CPU float64 running sum，最后除4并转为原参数 dtype；只平均同一单元同一轨迹的模型参数，不平均 optimizer、不同 seed、不同 fold、不同 task 或预测。冻结参数也按原状态一致保存。非参数 buffer 在四个时点必须保持一致，直接复制；发现会变化的 buffer／BatchNorm 必须停止说明，不能额外运行 update_bn 或引入新前向预算。

每单元只保留 BEST、最终 SWA、正常 history／validation 预测与采样记录，不要求保存四份全模型快照；记录采集 epoch、n_averaged=4、参数名和dtype等元数据，避免磁盘膨胀。SWA 前70epoch的验证只评估正常训练模型；第70epoch之后仅增加一次 SWA validation，通过同一评估接口产生独立 swa_validation_predictions 与指标，不覆盖 BEST 文件。SWA 验证是训练计划内的验证，不是额外 smoke。

主要依据：[Izmailov等，Averaging Weights Leads to Wider Optima and Better Generalization，UAI2018](https://arxiv.org/abs/1803.05407)；实现参考：[PyTorch2.8 Weight Averaging](https://docs.pytorch.org/docs/2.8/optim.html#weight-averaging-swa-and-ema)。论文给出轨迹权重平均及宽极小值的依据；本项目保留 AdamW／原周期 LR，是聚合物回归适配，不是其 SGD 图像实验的完整复现，更不能承诺一定恢复0.805。

## 7. 重复性门禁、一次冻结与正式评价

35对 S42_A/S42_B 的初态、实际训练顺序、逐更新训练 loss、完整 history 数值、逐 epoch validation 预测、BEST epoch/stage/参数、SWA参数与预测必须直接比较。日志时间／PID／路径不作为数学差异；比较数值不得用哈希或相近分数替代。相同种子比较应严格相等，并报告第一处实际差异和最大绝对差。保存正常计算的结果，不为比较追加模型前向。

首对或任意后续对有差异即 REPRODUCIBILITY_FAILED，停止、不冻结、不正式TEST、不运行最终CPU、不在线修复重试。直接比较CPU张量属于已登记的产物核对，不是单独 CPU suite。

140套及35对全部完成后，只使用三个 canonical 组 S42_A/S142/S242 的 validation，执行一次预登记质量门禁：

- SWA三组的七任务macro平均 ≥ BEST对应平均−0.005；
- 每任务的三组五折均值平均，SWA ≥ BEST−0.020；
- 全部预测有限、身份和目标尺度正确。

不通过则 VALIDATION_QUALITY_GUARD_FAILED；新正式性能未完成/不可比较。不得回退到 BEST 当作新 FINAL继续测试，也不声称 validation通过代表 TEST质量恢复。

通过后一次全局冻结：**FINAL_ESTIMATOR=BASE_SWA_LAST4_CYCLES，PRIMARY_RUN=S42_A**。冻结同时列出两个预登记估计器、三个 canonical组、全部210个正式来源、scaler、BEST epoch或SWA采集点、实际配置、原row/key和来源。BEST为受控比较估计器，不是事后候选池。S42_B不进入 TEST或跨种子统计。

随后一次正式阶段执行 BEST/SWA×3组×7任务×5fold=210单元／35,250行。每个来源只执行一次；不新增旧 BASE TEST、不重复推理、不生成教师或预冻结TEST神经特征。旧 B/T 的已有数字只读对照，读取旧模型张量0。

所有正式模型使用 eval＋no_grad，检查所有子模块 training=false；batch24、worker0、FP32、原scaler逆变换、同一原行身份join。每折先算原单位R²，再五折等权平均与std(ddof=1)，再七任务等权macro；不能把全部行拼在一起计算所谓macro。

看到TEST后禁止改选BEST、改SWA窗口／权重、选最优seed、混合任务／fold赢家、追加训练或反复评估。固定S42_A的单模型集结果与三组均值／std分别报告，不将三组预测平均包装为本轮模型。

## 8. “稳定成功”的预登记标准

本周期主目标是 BASE 稳定就绪；所有布尔门禁按未舍入值计算。不是只要重复结果相同就通过。

1. execution_complete：两次完整PT、140FT、PT对比及35对FT比较、一次冻结、210正式单元和唯一最终CPU均完整，真实exit均0。
2. same_seed_repro_pass：同种子PT完整比较及本周期35/35对FT比较全部严格通过；PT与FT分别报告布尔值和首个差异。
3. ft_seed_stability_pass（仅SWA三个canonical组）：macro std≤0.003且range≤0.006；每个任务的三组五折均值range≤0.020。
4. quality_retained：SWA三组macro平均及primary macro均 ≥ BASE−0.005=0.799675462276949；每组macro均 ≥ BASE−0.010=0.794675462276949；每任务三组均值平均 ≥ 原BASE相应任务−0.020。
5. stable_ready：以上全部true。只在同一新PT、同一头初态、固定平台的FT随机性范围内成立；不是PT seed稳健性或新数据集泛化保证。

同时报告 BEST 的相同稳定性／质量统计和SWA−BEST逐seed／task／fold配对差异，说明哪些任务的波动降低、是否只是整体分数相互抵消。三组seed只给有限描述证据，不宣称已证明普遍方差下降。

保留原用户目标与项目性能口径：primary macro≥0.810 的 target_met；primary ΔBASE≥0.010 的 performance_pass。这两个字段与 stable_ready 分开；恢复到接近原BASE不自动叫性能提升通过。新BASE别名替换仍须用户明确指定。

## 9. 固定预算、资源与速度

这是新的科学周期，**不是旧条件谱续跑**。用户将执行提示词交给DSH后，以开始实施的首次相关动作登记新t0；全周期硬上限48小时，包括实现、等待、准备、训练、正式评价和交付。唯一prepare含实现至多4小时；PT两次总阶段从首次PT派发起至多18小时、单次至多12小时，且不越总截止。最终CPU／交付合计预留2小时。不延长旧条件谱时钟，也不把旧失败成本归零。

| 预算项 | 本周期上限 |
|---|---:|
| prepare启动／新公共step0／新随机性质头 | 1／1／35 |
| PT启动／总更新／消费microbatch／presentations | 2／20,000／120,000／20,160,000 |
| FT组／完整单元 | 4／140 |
| FT epochs／更新／训练行 | 9,800／224,000／5,260,080 |
| 正常epoch验证／验证行 | 9,800／1,319,920 |
| 末尾SWA验证／验证行 | 140／18,856 |
| 新训练总更新 | 244,000 |
| 同种子产物比较 | PT1对＋FT35对；无额外前向 |
| 冻结 | 1 |
| 正式TEST单元／行 | 210／35,250 |
| 最终CPU suite | 至多1；仅完整正式TEST之后 |
| 前置测试／smoke／benchmark／额外诊断／额外前向 | 0 |
| 不同PT seed增加／新几何／表面／谱／FP生成／输出集成 | 0（两个PT均seed42） |
| retry／resume／自动修复／换卡 | 0 |

物理卡只允许 **GPU2，UUID GPU-31456dd5-a421-142b-644b-7a80f482dfb9**；容器逻辑 cuda:0，world_size1。PT单路；FT含validation最多4并发；TEST最多4并发，阶段互斥。先正常单路，再按80%当前可用显存预算准入2→4；不为吞吐更改batch、精度或数值策略。未知NVML PID只等待，不猜测身份或发信号；等待计时。

当前可见112逻辑CPU。纯元数据／角色资格解析可用min(96,实际affinity)个单线程CPU worker，避免重复构造数据或抢占训练进程。PT4单线程worker、prefetch1，训练主进程2线程；FT／TEST每作业0 DataLoader worker、主线程2。这里优先保持配方和RNG语义；不能为了CPU占满而重新引入数据读取变量。复用只读缓存与操作系统共享页，及时填补已准入GPU槽，避免重复反序列化全矩阵账本。

已有P3非DET PT10K实际耗时11,424.682秒，仅作先验，不能承诺新DET速度。每次PT使用前100次正常更新（剔除首20）及之后每500次的实测吞吐估计，首次PT还须估计第二次完整PT成本；FT使用首对正常完整单元和各任务既有耗时，按实际并发吞吐刷新。保守ETA=1.5×剩余工作实测估计＋2小时收尾，包含验证／I/O／保存／进程归并／正式TEST。无法容纳截止即停止，不删弱seed或慢任务凑交付。

内存按本周期已登记树的 Pss+SwapPss≤160GiB；RSS只诊断；MemAvailable≥32GiB、有限cgroup余量≥8GiB；可用磁盘低于60GiB时停止新派发。实现前只读记录剩余空间，输出采用流式SWA不保存四份大快照；不清理旧结果腾空间。沿用r6已交付的PSS固定对象集、pending及有组合证据的NO_USER_MM/exit_wait分类；不得放宽覆盖、把不可读写0或改变原证据条件。无mm分支在真实r6未触发，不宣称生产覆盖已验证。

GPU、worker或预计超过一分钟的命令必须在tmux session Uni-Poly独立window运行。控制器CPU-only，保留cwd/argv/环境、进程身份、日志、消费、controller/outer/bootstrap/tmux四层真实exit。

预算由R/budget.json记录本周期reservation、attempt、logged_complete与不确定区间；失败和预取不退款。P6/cumulative_budget.json 的354,471尝试／[354470,354471]完成区间、D的旧预算及全部旧时钟只读挂接为历史证据，不能把多个独立科学周期和alias混算成新的完成更新。

## 10. 允许修改与保护边界

仅允许新增以下9个隔离实现文件；本次Codex没有创建这些实现：

- configs/ua3d_base_stabilize_r1_gpu2.json
- src/modules/ua3d_base_stabilize_r1.py
- src/dataset/ua3d_base_stabilize_r1.py
- src/training/ua3d_base_stabilize_r1.py
- scripts/prepare_ua3d_base_stabilize_r1.py
- scripts/pretrain_ua3d_base_stabilize_r1.py
- scripts/run_ua3d_base_stabilize_r1.py
- scripts/bootstrap_ua3d_base_stabilize_r1.py
- tests/test_ua3d_base_stabilize_r1.py

可只读导入旧native模型、工厂、采样器、纯数值工具与已完成监控辅助函数；不得调用会写入旧OUT或改CURRENT_PLAN旧marker的控制器／prepare／unit入口。新包校验、训练、正式输出和状态更新必须局限本周期。

可按真实事实更新CURRENT_PLAN与PROJECT_HISTORY、写R和对应新日志根；保留本计划／提示词在authorization下的原始文本与相关source_text／actual_diff。已有实现、AGENTS、研究三路线原文、TODO、PH、BASE／旧PT／checkpoint、全部条件谱父根及旧稳定性／TEST复测根只读保护。禁止新增哈希／校验和／文件指纹合同、远程同步、覆盖／清理或另选备用结果根。

若同名实现文件或R/log根在实施前已存在，或出现活动writer／授权版本冲突，即停止核对，不直接采用本提示词覆盖。

## 11. 停止条件、最终集中测试与交付

prepare/schema/shape/角色/row-key/scaler/随机工厂/权限矛盾、非有限值、writer冲突、GPU/资源/时间/ETA/预算门禁、确定性算子不支持、worker非零或重复性门禁失败，均先持久化首错／遥测／消费／真实exit后停止。保留完整和中断来源；不在线修复、重试、恢复、改seed／卡／batch／worker／精度／SWA窗口。质量门禁失败与工程异常分别分类，不写成OOM。

前置pytest、contract/smoke、合成模型检查、额外benchmark、独立probe和额外模型forward均0。正常prepare静态接口检查、初始化断言和正常训练内的数值断言允许。只有210正式单元完整结束后，至多一次最终CPU suite，覆盖：初始化工厂与新schema转换、无历史权重输入、dropout/冻结/优化器分组、SWA固定采集点与平均/buffer语义、35对比较、身份join／scaler／指标汇总、预算／阶段时序和监控分支。用真实执行产物与小型数学fixture验证，不照抄旧条件谱210新FT的断言，不为CPU测试追加GPU前向。

交付至少包含：

- authorization、recipe_audit.json、initialization/source_factory.json、identity.json、registry/canonical.json、实际diff／source_text；
- 两次PT的配置／history／consumption／同一新step0来源／各自step5000与deploy／终点optimizer-scheduler-RNG／stream及预取账目、PT重复性比较；
- 140套完整FT的BEST／SWA／history／两类validation产物／config／scaler／消费，35对reproducibility结果；
- selection/final_frozen.json；210份formal预测／metrics／身份对齐；
- budget.json、jobs.json、environment／resources／clock／PSS与退出审计、四层真实exit；
- final_cpu的唯一started/command/log/exit及结果；
- final_review/results.json／report.md／delivery_summary.md，包含完整精度各组／估计器／七任务／macro、seed std与range、ΔBASE及SWA−BEST配对差；
- 按SOTA_RESULT逐任务对照最佳均值及模型；说明论文包络与数据/split/统计协议差异。两种估计器均报告，但不得根据TEST变更FINAL。

完整交付状态写COMPLETE_PENDING_REVIEW，稳定性／质量／性能分开；independent_review=PENDING，不把DSH自检写成Codex独立审查通过。任何完整正式阶段未完成时写“未完成/不可比较”，对应新macro／Δ／stable_ready保留null或NOT_EVALUATED；不得用validation补全正式结论。BASE保持原指定身份。


### 2026-11-05 用户追加授权：历史 results 磁盘清理

用户明确授权删除不必要乃至全部历史 results，并将分析保留于 PROJECT_HISTORY.md。此追加授权覆盖旧周期产物只读保留限制，不改变本轮科学合同、初始化、预算、时钟、数据或训练数学。已在删除前归档49份数值汇总与57份报告，实际删除124637个旧checkpoint/预测/特征/缓存/大批量JSONL文件，results约676→123GiB；保留小型指标/config/scaler/预算/时钟/exit记录。旧BASE模型权重已删除，数字reference0.804675462276949保持，历史证据路径不再保证存在模型/预测；不可宣称仍可原样重放或完成独立复核。

当前BASE-STABILIZE/r1完整现场、105GiB原始PT模板、logical mapping、下游输入与35源config/35 scaler均保留；清理后PT_R1仍RUNNING且无first_failure。清理于tmux Uni-Poly/results-cleanup执行，真实exit0；明细logs/results_cleanup_20261105_authorized/summary.json、manifest.jsonl、deleted.jsonl，分析快照与限制见PROJECT_HISTORY.md。训练九文件未改、未新增forward/CPU suite/重试。


### 2026-11-05 用户追加授权：results/logs 按保留范围整理

按用户明确要求，仅保留BASE reference现存资料、当前BASE稳定性及其输入/历史重复性证据、最近条件谱/HMR相关记录与清理审计；图形学仍暂停，实际活动运行只有BASE-STABILIZE/r1。删除385个无关目录/文件条目，results净减1.868GiB、logs净减0.950GiB；保留35源config与35 scaler内容相等、锁定九文件未改、当前训练RUNNING且无first_failure。原BASE权重上一轮已删除，本次未恢复；reference数字与身份不变，无法据元数据宣称原权重仍在。

目录白名单、排除依赖、writer检查、逐条删除及统计见logs/results_logs_scope_cleanup_20261105/plan.json、deleted.jsonl、summary.json，tmux真实exit0。PROJECT_HISTORY已于删除前登记范围和历史分析快照位置。data/与所有源码未改；不启动/重试实验，不增加forward或CPU suite。
