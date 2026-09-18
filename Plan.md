# GLT-V2 预测性能优化：融合预训练、Trimer–RU 对齐与 XC 适应

## 0. 交接头、授权与完成口径

| 字段 | 内容 |
| --- | --- |
| 计划 ID | GLT-PRED-20260918-01 / r2 |
| 状态 | 待授权；仅方案编制完成，代码、验证、训练和预测比较均未执行 |
| 用户要求 | 全面优化预训练、2D/3D、Trimer 与单 RU 融合、XC 微调及参数；考虑替换 7-RU FP；r2另加入独立2D–3D对齐预训练实验；供其他模型执行 |
| 本轮授权 | 编写本计划及必要文档归档、提交同步；不授权启动科学实验 |
| 角色 | Codex 规划与后续独立审查；接手模型执行并记录实际执行者；不默认启动子代理 |
| 代码基线 | r1为dev@6d44ed9；r2文档修订前为dev@cd3af56，pull --ff-only 为 Already up to date |
| 用户已有改动 | r1按请求填充用户清空的Plan.md；r2开始时工作树干净，保留r1全部未执行计划 |
| 上一周期 | GLT-ENGINEERING-20260918-01/r4 已归档；工程正确性关闭，不代表完整提速或预测提升 |
| 当前范围 | 科学设计与分阶段执行合同；只有另获授权的阶段才可实施、运行 |

**执行端先读 AGENTS.md 和本计划，再核对实时仓库、配置、产物和进程。不要把本文件中的拟新增接口当作已有 CLI。**

授权分层：用户后续说“实现并验证本计划”，默认指 S0–S2 的实施与必要有界验证；S3/S4/S5 的研究训练预算和 outer-test 必须明确包含在用户授权中。用户明确授权全部阶段时按本合同顺序执行，不重复申请已授权步骤；条件不满足时停止晋级，不用剩余预算机械补跑。历史科学实验授权不自动迁移至本计划。

科学成功不预设 R² 必须达到 0.87。候选均无效也可完成一轮有边界研究；不得将负结果写成提升，或为达到目标自动加组、加 seed、加构象。

**r2变更：**增加§5.6图级ALIGN，与FGR并列而非默认叠加；S3b从3组增为4组，拟议研究预训练总上限30k→35k，开发神经微调1620→1800epochs（另加原2epoch smoke），correctness总上限28→40updates。S4/S5名额不增加。本次授权仅为计划修订，新增实验尚未启动、尚未获运行授权。

## 1. 事实基线与证据索引

### 1.1 当前主线

当前对象为 `O8-BondPath-GalformerTrimer-Hop2`、Concat、geometry_head_norm、5k；不是旧 N+1/N+2 蒸馏，也不是退役融合模型。无 MD200、教师、InfoNCE。

| 项目 | 核实的当前实现 |
| --- | --- |
| 2D | 周期 canonical RU；137 原子特征＋1 backbone；138→512；6 层、8 heads、FFN 2048；SPD/path-node/bond-path bias |
| 3D | 冻结开放 Trimer 的每条物理键独立 state；端点元素、键长、路径键角；6 层、512、8 heads；没有显式扭转或非键距离 |
| 读出 | O8 真实 canonical 原子 mean；GLT 中心内部键 mean；分别 LN 后拼成 1024 |
| 下游 | 1024→512→1；train-only standard scaler、MSE、validation R² 选模、outer5_inner20 |
| 预训练 | motif 原子 mask 0.30，元素 CE；坐标噪声 0.03 Å 后中心键长/角度去噪；0.1×7-RU rooted FP BCE |
| 参数 | O8 18,998,040；GLT 20,021,854；两个 LN 共 2,048；deploy 39,021,942；不含下游性质头 |
| 优化器 | 预训练 AdamW lr 2e-4、wd 0、BF16、global batch 1008；5k updates、warmup 2000、cosine horizon 20k |
| 微调 | encoder lr 1e-5、fusion/head 1e-4、wd 0.02；batch 32/eval 64；100 epochs、warmup 5、patience 10 |

事实源：`src/modules/glt_dual.py`、`glt_dual_pretrain.py`、`mips_local_graph.py`、`src/dataset/glt_dual*.py`、`scripts/finetune_glt_dual.py`、`configs/mts/glt_dual_three_task_concat_geonorm.json`。源码签名以执行时为准。

### 1.2 必须保留的参考产物

- fixed deploy：`results/glt_dual_static_pretrain_5k_concat_geonorm/deploy_05000.pt`。
- 原始配置/来源：同目录 `run.json`，及 `results/glt_v2_fixed_concat_5k_20260916/deploy_05000_validation.json`。
- 七任务参考：`results/glt_v2_fixed_concat_5k_20260916/aggregation_review_7task/summary.json`。
- O8 归因：`results/glt_sci_o8ctrl_20260917/final_comparison/summary.json` 及 formal 下各 arm 的 run/summary/predictions。
- splits：`data/splits/mips_outer5_inner20/`；原始性质表：`data/raw/smi_<task>.csv`。
- 预训练 cohort：`data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1`；静态缓存同级 `dual_static_v1`；实际 parent bundle 由 run/store 解析，不猜测或切换。
- 下游 cohort：`data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1`；静态缓存同级 `dual_static_v1`；cache root 从原 run 核对。

| 既有描述性结果 | macro7 R² | XC R² |
| --- | ---: | ---: |
| fixed Concat | 0.777210 | 0.308668 |
| A：双路包提取 O8，独立微调 | 0.776328 | 0.319058 |
| B：独立 O8-only 预训练/微调 | 0.777674 | 0.336078 |

当前没有证据支持完整 GLT 在 XC 上优于 O8。以上是开发 folds 的历史结果，不是新的独立盲测；不据此逐折挑最优模型。旧七任务不含 egc，禁止与八任务 macro 混排。

### 1.3 XC 基线

本地 432 条，Xc 范围 0.13–98.81；各折 train=276，validation=69/70，test=87/86；432 个不同原始 SMILES 不等于完成 polymer identity 去重。单 heavy 原子记录 `*C*`、47.8，仅一条，不能解释整个任务低分。XC 的原始标签定义、实验/计算来源及条件应追溯，不能先假设全是实验结晶度或全部由 DFT 得到。

## 2. 论文依据与项目新增部分

主要窗口为 2024-09 至 2026-09。执行端引用时核对正式版本；预印本如实标记。

| 原文 | 采用的依据 | 本项目新增，不宣称原文已验证 |
| --- | --- | --- |
| [GRIN, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/7fe3921147c968d0b57a224c0d07e21d-Abstract-Conference.html) | RU 重复表示一致性与拓扑增强 | 中心物理 Trimer 与 canonical RU 对齐；三 RU 的理论条件不是实际构象充分性证明 |
| [FlexMol, CIKM 2025](https://arxiv.org/html/2510.07035v1) | 跨模态交互和重建、缺失模态处理 | 本文融合条件距离任务、原子—物理键桥，不是完整 FlexMol 复现 |
| [Masking Design, TMLR 2025](https://arxiv.org/abs/2512.07064) | 语义目标与编码器匹配比复杂 mask 更值得优先验证 | 周期原子环境目标与保留现有 mask |
| [SCAGE, Nature Communications 2025](https://www.nature.com/articles/s41467-025-59634-0) | 化学/几何多任务和功能团知识 | 不照搬其任务数量或声称聚合物最优 |
| [Token-Mol, Nature Communications 2025](https://www.nature.com/articles/s41467-025-59628-y) | 显式扭转承载构象信息 | 标量 GLT 的局部扭转增量，不引入 SMILES LLM |
| [DenoiseVAE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html) | 噪声机制与分子结构应匹配 | 初轮仍固定 0.03 Å，不直接实现 Noise Generator |
| [PolyConFM, 2025 预印本](https://arxiv.org/abs/2510.16023) | RU 局部构象与 RU 间相对构型分开建模 | 不以单冻结 Trimer 冒充长链 MD 或构象分布 |
| [ELoRA, ICML 2025](https://proceedings.mlr.press/v267/wang25al.html) | 低数据下受限参数更新 | 当前标量网络采用普通 LoRA，不声称实现 SO(3) ELoRA |
| [TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6) | 小样本表格读出参照 | 仅后续候选；初轮用 Ridge，避免新增依赖和模型数量 |

## 3. 核心问题与总体顺序

Q1：当前表示已有信息是否被全量微调破坏？→ 冻结读出/LoRA。

Q2：7-RU FP 能否被更贴合双路预测的任务替换？→ FP / 无 FP / 融合条件几何重建 / 图级2D–3D对齐四组；区分重建几何和对齐表示。

Q3：中心内部键 GAP 是否丢失跨 RU 的局部对应？→ 原子锚定融合，固定目标做对照。

Q4：键长/键角之外的构象信息是否有用？→ 单独添加扭转，不同时添加非键邻域。

Q5：化学监督语义是否不足？→ 用原子环境替换元素目标，固定架构。

执行顺序为 S0 审计 → S1 XC 适应模块 → S2 FP 替换及边界实现 → S3 适应/目标开发比较 → S4 条件架构与语义增量 → S5 锁定后正式确认。S4 各分支有独立开关，不要求全部晋级。

## 4. 不变项、数据划分与选择纪律

### 4.1 数据与资产

- 不改 active topology/Trimer/static/targets、manifest 或 `.frozen`；不覆盖既有 checkpoint/results。
- 复用实际冻结单构象，不生成新构象、RU±2 几何、晶胞或长链。7-RU 原任务只使用拓扑，不曾意味着 7-RU 几何。
- 移除 FP 训练消费不等于删除旧 target cache；BRICS 等公共字段照常读取。
- 中心 canonical↔O8↔实际 Trimer 原子/键映射必须可证明；不可依赖数组位置或任意 atom order。
- 初轮比较使用相同预训练 cohort、全部同序样本；不得按某新任务有效性过滤样本。化学、局部几何、融合几何各有独立 mask/分母。
- 不引入 MD200；不重跑旧蒸馏。新结构有独立 architecture/task 配置身份，沿用现有严格加载能力，禁止伪装成旧 deploy。

### 4.2 预训练验证与来源

S0 建立按 polymer identity 的 95%/5% P_train/P_val 划分，seed=42，四舍五入方式写入实际数量；同 identity 的重复表示不得跨集合。禁止以正式任务标签构造此划分。

查 PI1M 与八任务的身份重叠。主科学比较优先排除全部 benchmark 身份后建立公共 P；不修改旧 cohort，使用独立只读索引列表。所有新参考/候选使用同一 P。若 identity 无法证明，相关严格泛化声明阻断；历史 checkpoint 可用于适应诊断，但保留来源限制，不称独立盲测。

这会使新参考不同于历史训练；必须重训新 B_FP，历史 5k 只作参考，不用它替代新 matched baseline。过滤后样本数及预算对应暴露数由审计确认，不预填 PASS。

目标统计、环境词表只拟合 P_train；P_val 用固定 mask/噪声/目标集合。训练样本顺序和随机流在实验组间匹配；新增 head 的初始化、采样使用独立 generator，不扰动公共 dropout/mask/噪声序列。

### 4.3 下游选择与评估

- 保留现有 outer5_inner20 manifest；scaler 仅拟合最终 train，validation 早停和选配置；不 refit train+validation。
- S1/S3/S4 开发只用 folds 0/1 的 train/validation，outer-test loader 不实例化、不读取预测文件来调参；开发最大30 epochs、warmup5、patience10，所有组同 schedule 长度。
- 开发任务固定 xc、eps、eat，分别覆盖主问题、低数据相关风险和高分保护任务。历史 test 结果可描述，但不能反复据此选候选。
- folds 0/1 的 train 会包含其他 folds 的 outer-test 身份：该开发方式不能声称最终五折是严格嵌套独立测试。最终五折标记 development-CV；若需独立泛化结论，另立外部测试或真正嵌套计划。
- checkpoint 选择用各自预训练验证检查和锁定开发 probe；不同损失数值不可直接排名。主比较统一用第5000步部署，不能每个任务用 outer-test 挑预训练步数。
- 不跨任务注入同一 held-out polymer 的监督标签；多任务监督训练不纳入初轮。

## 5. FP 替换：Fusion-Conditioned Geometry Reconstruction（FGR）

### 5.1 为什么不是“把 FP 改成 fusion”

当前 FP 已经从融合表示预测，但标签由拓扑决定，融合可能不需要独有的几何信息。Fusion 是运算，不是监督目标。本计划选择可实现的替代：**同一预测融合表示参与中心 RU 原子对的干净距离重建**。

FGR 是 FlexMol-inspired 的项目条件几何重建，不是完整跨模态生成、3D教师蒸馏或证明两路必不可少的目标。仅有 loss 下降不能证明跨模态利用；必须做分支消融和下游比较。T_FGR本身不加入InfoNCE；r2另设T_ALIGN，二者不混用。初轮不加入EMA teacher或matching classifier。

### 5.2 固定目标身份与采样

1. 只取中心 q=0 的两个不同实际重原子；在真实 Trimer 化学图上 shortest-path distance 为2或3的无序物理原子对。禁止以 canonical 索引“捷径”构造不存在的物理路径。
2. 目标为两中心原子的干净欧氏距离；SPD=3 可提供超越两个独立键角的构象约束。不使用 q=±1 的同名多值目标，避免 decoder 无法辨识左右目标。
3. 每图最多32对：SPD2/SPD3 各最多16，某层不足将剩余额度给另一层；超过时无放回均匀采样。采样由 sample identity、epoch/absolute-position 和独立任务 seed 决定；验证固定。记录每层覆盖，不能只挑短距离或成功样本。
4. 候选由拓扑决定，不按 clean 距离排序选择；重编号 fixture 使用同步映射后的固定候选检验不变性，不要求两个独立随机抽样集合逐位一致。
5. 无候选、无效几何、中心内部键数0的样本不进入 FGR 分母，保留原子任务；明确记录 FGR 在小 RU 上的覆盖限制，不扩大目标到侧 RU 来隐藏低覆盖。
6. `t_ij=(log1p(d_ij / 1Å)-mu)/sigma`，mu/sigma 从 P_train 全部上述候选统计一次，sigma 有有限正下界；P_val和下游不参与拟合。

### 5.3 输入、decoder 与损失

继续原有 O8 motif mask 和整 Trimer 坐标副本0.03 Å噪声。所有 encoder 几何输入从 noisy 坐标计算；clean 坐标仅在监督构建器中。FGR 初版不新增几何 mask、不使用 clean pair distance 作输入，称“条件去噪重建”，不称“无几何线索的遮蔽恢复”。

定义 O8 canonical atom state 为 H，模型共用融合输出为 g_f；当前 Concat 下 g_f 为1024维，预训练/微调必须调用同一个 `fuse()`。

$$
\hat t_{ij}=D_{\rm FGR}([\operatorname{LN}(H_i+H_j),\operatorname{LN}(|H_i-H_j|),g_f]).
$$

decoder 为 `2048→256→1` GELU MLP，两个局部 LN 无 affine；g_f 已使用现有分支 LN。不输入 raw坐标、原子ID、clean目标、距离rank。对称端点组合确保交换 i/j 不改变结果。

$$
L_{\rm FGR}=\frac{1}{|V_{\rm FGR}|}\sum_{b\in V_{\rm FGR}}\frac{1}{|P_b|}\sum_{\{i,j\}\in P_b}\operatorname{Huber}_{0.5}(\hat t_{ij}-t_{ij}).
$$

主替代目标：`L = L_chem + L_geo + 0.1 L_FGR`。保留 geometry_head_norm 和原 geometry 分母语义。0.1 是固定起点，不代表与 FP 有相同梯度规模；记录各分支梯度贡献，不能仅比 loss 标量。空集合返回与图连接的有限零；各任务按全局有效图数归一化，DDP梯度累积不得平均microbatch均值。

### 5.4 必跑目标矩阵

| ID | 架构 | 目标 | 控制问题 |
| --- | --- | --- | --- |
| B_FP | 原 Concat | chem＋geo＋0.1 FP | 新数据协议下 matched baseline |
| B_NONE | 原 Concat | chem＋geo | 去掉 FP 本身的影响 |
| T_FGR | 原 Concat | chem＋geo＋0.1 FGR | 替换 FP 的总效果；相对 B_NONE 是 FGR 增量 |
| T_ALIGN | 原 Concat | chem＋geo＋lambda_align(k) ALIGN | 替换FP为图级对齐；相对B_NONE是对齐增量，与T_FGR比较两类第三任务 |

这些组共享 encoder 初始化、chem/geo head 初始化、样本/公共随机流、训练与下游预算。FP/FGR/ALIGN head参数量不同，如实报告，不称参数完全相同；最终deploy结构相同。B_NONE的未受预训练监督的融合LN明确记录为初始化状态，不与已受训练者混称同一预训练路径。ALIGN使用现有norm2/norm3后接独立投影头，使这两个部署LN接受对齐梯度。

T_FGR 稳定有效才允许补一项 `FP+FGR`，回答替换还是互补；该项在本轮核心预算外，不能自动执行。

`FGR+ALIGN`和`FP+ALIGN`同样不在核心四组内。若两个单项均有下游信号，可另修组合实验及预算；不因论文同时使用多个目标便直接叠加。

### 5.5 防泄漏和有效性检查

- 对选中 pair 的 clean距离做独立扰动，只能改变 label，不能改变 encoder输入；对 noisy坐标改变，应一致改变全部相应输入，label保持固定。
- 当前输入无非键 pair距离。未来若加入，必须屏蔽/扰动目标距离及其反向、所有直接副本和派生输入，另修任务版本。
- 仅反传 FGR，O8、GLT、实际 fusion参数在非退化fixture上须有有效梯度；不要求每个参数每batch非零。
- 固定同一 checkpoint，分别置零/置乱图级3D与局部2D decoder通路，记录验证FGR响应；这只查依赖，不能作为泛化或跨模态协同证明。
- 重建改善但下游无益时停止晋级；不自动增大权重直到 test变好。
- 若以后新增融合化学预测，需同步遮蔽所有物理副本的端点元素及元素条件几何类型。该任务不在初版，以免3D直接泄漏被遮蔽元素。

### 5.6 独立实验：Graph-Level 2D–3D Alignment（ALIGN）

#### 5.6.1 适用性与不适用边界

适合回答“同一聚合物的化学表示与局部几何表示是否能学到更可迁移的共享语义”。FGR预测具体几何量，ALIGN拉近两路表示，二者不是同一任务。[FlexMol §3.1.3（CIKM 2025）](https://arxiv.org/html/2510.07035v1)使用跨模态InfoNCE提供直接依据；本文的中心RU范围、多正例身份处理和分布式合同是项目适配，不是完整复现。

首版只做图级，不强迫O8原子与GLT键逐token一一相等，也不对齐整个3RU无差别GAP。旧N+1/N+2采用教师蒸馏，本实验是两路联合更新；不能把旧蒸馏失败或论文成功直接当成本实验结论。

两路共享化学身份，但2D不唯一决定某个构象。用独立小投影空间对齐，保留原512维表示及chem/geo目标，不直接最小化原始H/U全维MSE。即使如此仍可能损害几何特有信息，须通过XC/eps/eat开发结果裁定。

#### 5.6.2 表示、视图与参数

- `g2 = norm2(mean canonical O8 atoms)`，`g3 = norm3(mean center internal GLT bonds)`，均为512维、均在跨模态融合之前；norm2/norm3是现有Concat部署参数。
- 分别使用不共享的 `P2/P3: Linear(512,256)→GELU→Linear(256,128)`，无BatchNorm、无dropout，输出以FP32 L2 normalize（eps=1e-8）。两路encoder、LN和投影头均接收梯度，不stop-gradient、不冻结教师。
- 复用本次forward的原O8 30% motif mask与GLT 0.03 Å noisy输入，不额外运行clean encoder，不为ALIGN改变公共视图或随机流。把不同mask/噪声的差异视为扰动，不宣称为多构象。
- cosine temperature固定tau=0.1，不学习、不扫参；独立初始化seed保存。第k个optimizer update（k从1计）`lambda_align(k)=0.1*min(k/1000,1)`，与LR warmup分开记录。该ramp是目标政策的一部分，不宣称与固定FP/FGR系数梯度强度完全匹配。
- 下游移除两个投影头，保留原Concat/norm/predictor；初轮没有新增推理容量。ALIGN直接训练两路表示与分支LN，不是学会了原子级融合。

#### 5.6.3 有效集合与多正例

每个分布式microstep收集全部rank的有效配对记录V，条件为2D图有效、真实几何有效、中心内部键数>0、polymer identity已验证。invalid/N=0不作为anchor、positive或negative，仍参加其适用的其他任务，不能从训练数据集删除。

anchor i的正例集合`P(i)`是V中相同polymer identity的对侧记录，**包含其自身配对行**；不同身份为负例，不能用`j!=i`删除本应存在的跨模态正例。重复key、等价RU写法若已证明同一identity则为多正例，不当负例。无法证明等价关系时先做S0身份审计，不用相同元素组成/相同指纹替代身份。

仅有0条、1条配对或全为同一identity时，没有有效负例，整个microstep ALIGN返回连接两路投影计算图的零；不把多正例自身的softmax竞争作为学习信号。正负数量只依赖身份/有效性，不能按当前embedding相似度动态丢样本。结构相似但不同身份仍可能是语义上的假负例；记录此限制，初轮不新增hard-negative挖掘、队列或teacher。

#### 5.6.4 精确损失

令`z2_i=normalize(P2(g2_i))`、`z3_j=normalize(P3(g3_j))`，`s_ij = dot(z2_i,z3_j)/tau`。有至少两个不同identity的microstep使用正例log概率的平均，而非含义不明的“多正例InfoNCE”：

$$
\ell_i^{2\to3}=-\frac{1}{|P(i)|}\sum_{j\in P(i)}\left(s_{ij}-\log\sum_{k\in V}\exp(s_{ik})\right),\qquad
L_{\rm ALIGN}=\frac{1}{2|V|}\sum_{i\in V}\left(\ell_i^{2\to3}+\ell_i^{3\to2}\right).
$$

`3→2`方向交换两路并重建同identity正例mask；用FP32 logsumexp。正例包含在分母中。每条有效样本作为anchor等权，重复身份出现多次会增加其采样权重，如实记录，不宣称identity均匀采样。

`L_total=L_chem+L_geo+lambda_align(k)*L_ALIGN`。对齐只替代FP，不新增FGR。没有ALIGN样本时仅此项为零，其他任务与optimizer/scheduler正常执行。

#### 5.6.5 DDP和梯度累积：必须实现同一个目标

3rank×84，每个microstep对比池最多252条，而不是optimizer global batch1008。4次累积的对比池彼此独立，不保留跨microstep图、不引入队列；必须在config/report写明`contrastive_pool=distributed_microbatch`，不可声称1008个负例。实际负例数扣除invalid与多正例。

按rank固定槽位padding收集embedding、valid mask和identity。浮点embedding的all-gather必须支持跨rank梯度，并在backward把远端key贡献SUM回源rank；identity/mask不需要梯度。不能detach远端key后仍声称完整对称InfoNCE。

每rank只算本地有效anchor的`0.5*(两个方向loss之和)`，再对本地anchor求sum；跨rank、跨本次4个microstep的有效anchor总数A为统一分母（无负例microstep的anchor不计入A），rank局部sum乘`world_size/A`以抵消DDP参数梯度平均。对齐A可通过预备的identity/valid元数据在累积窗口前求得，不保留4份模型计算图；A=0安全返回零。不得先平均各microstep或各rank的均值，亦不得对已归一化ALIGN再额外乘world_size。

全部rank无论本地是否有有效几何，均按相同顺序执行collective与backward；零rank使用连接投影参数的安全占位，padding绝不成为负例。DDP no_sync的参数同步策略不能抑制loss所需的embedding gather/backward通信。

用单进程拼接相同数据（投影无dropout）作数学参考，先比浮点loss和两个投影/输入embedding梯度，再验证真实3rank；FP32参考容差起点atol/rtol=1e-5，记录硬件/实际容差。不得用“有限”代替正确的梯度比例验证。

#### 5.6.6 验收、诊断与晋级

- 独立覆盖跨rank重复identity、多正例、零rank、all-zero、全同identity、仅一有效pair、padding、两方向、累积不等有效数；no-negative场景有限零，不发生空均值。
- 验证正例相似度、负例相似度、跨样本方差/有效秩、投影前后范数、梯度范数、有效anchor/negative数和无负例microstep比例；崩塌不靠加大lambda掩盖。
- P_val每次固定同一组至多1024条记录和排序，eval后在此固定池上计算双向多正例检索与loss；明确其池大小与训练microbatch不同，不直接比较数值绝对高低。集合小于2个identity则不报告检索分数。
- 在同一P_val记录上做一次几何特征固定化诊断：按P_train统计的键类型距离/角度均值代替数值，保留化学类型和拓扑，观察对齐检索变化。只能称输入响应检查，不能视为训练消融或独立性能证据；现有几何模块不接受这种替换时记录未执行，不静默改拓扑。
- 因GLT自带元素/键类型，对齐可以靠共同化学身份完成；高检索率不是“3D独有信息有效”的证据。开发XC/eps/eat和geometry健康仍是晋级依据。
- T_ALIGN必须相对B_NONE和B_FP报告，并与T_FGR同预算比较。仅对齐loss变好但下游无增量时不晋级，不默认叠加FGR或追加seed。
- ANCHOR位于本实验对齐读出之后，ALIGN不会训练该新桥。若S3选择ALIGN为parent，S4不得直接运行“仅ALIGN＋ANCHOR”并称桥已预训练；可研究环境目标/扭转或matched O8，或另行修订有桥监督的组合实验与预算。

## 6. Trimer–RU 原子锚定融合（条件阶段）

### 6.1 身份和语义

继续维护 E_trimer 个独立物理键 state，不平均三份坐标、不提前按 canonical ID 合并物理键。给每个中心实际原子建立与其相连的真实键 incidence，包含跨 RU 键的中心端点；同原子双连接保留两个实际键及多重性。

中心原子i的几何摘要：`U_i = mean{Z_e : e incident to actual atom (i,0)}`。空集合为零。物理键通过既有GLT从整个Trimer收集上下文，不直接将侧RU末端原子的state无差别平均给中心。

不使用原始 q 正负作为可学习方向标签；只用可证明的中心身份与真实连接。中心锚定不保证任意 RU 重新切分或无限链不变性，另行测试、如实限制。

### 6.2 固定首版数学接口

$$
H'_i=H_i+m_i\,\sigma(G[\operatorname{LN}(H_i),\operatorname{LN}(U_i)])\odot W\operatorname{LN}(U_i).
$$

`G:1024→512`、`W:512→512`，门bias初始化使 sigmoid=0.05；不把W也置零。m_i要求原几何有效、有中心内部键、且该中心原子有incidence。整个增量（含bias）在m_i=0时精确为零。

保持最后输出1024：`g_f=[LN(mean_i H'_i), LN(mean_center Z)]`，第二半沿原规则mask；性质head和FGR decoder维度不变。此首版保留中心键分支，避免同时更换全部readout。旧 Concat 是新增桥关闭的回退。

化学头仍读取融合前H，防止未遮蔽3D元素造成shortcut；FGR decoder局部端点仍读取融合前H，但共享g_f依赖H'，保证融合桥有监督梯度。

**N=0保持原科学定义**：中心readout为空、geo/FGR均不监督，首版关闭桥，仅原子任务/2D预测参与。将N=0跨键信息接入下游是单独后续研究，不能在此暗中改变规则。

### 6.3 对照与推广

在选定目标不变的前提下比较同预算旧Concat与ANCHOR；共享参数显式复制初始化，新增参数单独seed。报告新增参数和训练成本；可归因为“加入原子锚定模块的整体变化”，不能声称纯对齐而排除容量影响。

两个模块都有效后才组合；不直接升级为多层cross-attention。新增架构从匹配初始化重新预训练，热启动作为另一类实验不混入主对照。

## 7. 构象信息、2D目标与其他结构候选（逐项选择）

### 7.1 扭转增量：首个3D输入候选

- 从同一冻结Trimer的连续四个不同重原子、三条真实化学键构建二面角，要求中间键接触中心RU；不添加坐标、不从canonical索引拼接假四元组。
- 首版用 `[cos(phi), cos(2phi)]`，避免任意有向符号及反射问题；明确损失了手性符号辨识力，不宣称完整立体/立构序列表示。
- 退化/近共线四元组mask，不以epsilon伪造有效角；同一物理路径与反向只计一次，关系多重性有明确记录。
- 使用 `MLP(2→64→512)`，按中间物理键归一化聚合，在初始token上加小门控残差，tanh门初值0.02；encoder用noisy扭转，不能预缓存clean扭转当noise输入。
- 第一轮仅改变输入，沿用选定损失，不新增扭转监督。对照保持模块/路径/mask相同，仅将cos特征换成固定零；这是角度值增量对照，不是无几何对照。
- 若多数样本无有效覆盖，记录而不是转为非键距离模块继续跑。非键中心query路线排在后续，需独立设计半径/邻域和FGR目标屏蔽。

### 7.2 语义化学目标候选

保留30% motif mask，先用周期 rooted radius-1 原子环境类别替换单元素类别：中心Z、中心化学状态、无序邻接的(Z,bond type)多重集合；沿显式periodic edge识别真实邻接，不靠开放端帽。P_train建词表，频次<20归UNK，评估UNK比例；超过20%则停止该版本，不擅自扩词表。

头结构沿原graph decoder，仅输出类别数改变；target不作为输入。保留独立元素准确率作诊断但不额外加入元素loss，避免把“替换”变成“增加”。先在锁定架构、锁定第三任务下比较，不同时修改mask或训练时长。

### 7.3 全面路线图中暂不自动执行的项目

1. O8全局attention或motif token：先证明长程化学瓶颈；不同时增hop/层数/readout。
2. 拓扑RU重复一致性：先审计合法等价表示；不得强迫不同有限链坐标相同，更不能假设真实分子量无影响。
3. GLT降至256维/4层、独立2D/3D LR、attention温度：各为独立后续候选，不作为本轮顺手调参。
4. 5k scheduler horizon对齐或延长20k：当前5k含2000 warmup、20k horizon；本轮保留以隔离目标/架构收益，调度研究另列。
5. 多构象/长链/条件变量/多任务/ensemble：需要数据来源、预算和跨任务身份划分的新计划。
6. 不默认恢复MD200、旧KD或KFuse；已有单knowledge KFuse退化softmax不能称有效跨模态选择。

## 8. XC 适应研究的精确合同

### 8.1 S0 数据诊断

追溯Xc标签来源、范围、重复身份、stereo声明、未知条件、有效几何/N=0和各train/validation分布。不得按test残差定制子群或删异常值。文献432条的最大98.41与本地98.81不一致，仅记来源差异，不能自动修改CSV。

已有outer-test只作冻结历史描述。本轮不新增test残差挖掘；未能溯源时保留代理标签解释，不声称预测真实加工条件下结晶度。

### 8.2 固定候选，不全面扫参

S3适应比较先只用同一个fixed Concat 5k checkpoint：

| 方式 | 可训练参数 | 超参数 |
| --- | --- | --- |
| FULL | 全模型，现行对照 | encoder1e-5，head/norm1e-4，wd0.02 |
| HEAD | 仅性质head；encoder及norm冻结eval | head1e-4，wd0.02，原512隐藏head |
| LORA | 两路每层Q/V低秩增量＋性质head；base与norm冻结 | rank8，alpha8，adapter dropout0，adapter/head1e-4，wd0.02 |
| RIDGE | 冻结1024 graph features＋线性读出 | alpha={0.1,1,10,100}，仅validation选 |

QKV为合并Linear时，仅对Q/V切片加低秩更新，K与base不变，不能顺手改变source-Q/target-K或缩放。LoRA A按独立seed初始化，B=0；首步A梯度可合法为0，不能据此判失败。冻结base dropout关闭，与FULL的正则状态差异明确报告为适应策略的一部分。HEAD/RIDGE encoder.eval；公共masked训练不进入下游。

RIDGE feature scaler只fit train，标签标准化同理；不得把全部432条先做PCA或特征筛选。初版不需PCA/TabPFN。

适应结果只决定一个全局下游策略用于后续目标比较；不可每个实验组用不同最优策略。若XC最佳策略显著损害eps/eat，则保留FULL为通用策略，XC-specific策略另报，不混成单模型总收益。

不变项：现有manifest、train样本、MSE、batch、early-stop规则、无train+val refit。禁止同轮改损失/标签变换/采样来掩盖负结果。

## 9. 实施文件、接口和部署

优先复用现有runner、DDP归约、static reader、严格deploy和恢复逻辑；不建立新通用训练框架。建议按职责增补，实际文件位置由执行端核实：

| 职责 | 现有或拟修改位置 |
| --- | --- |
| FGR targets/incidence/torsion | `src/dataset/glt_dual_pretrain.py`；必要时新增局部helper，不改旧缓存 |
| FGR/ALIGN头和分任务sum/count | `src/modules/glt_dual_pretrain.py`；ALIGN使用有梯度gather的小型loss helper |
| 原子锚定及扭转模块 | `src/modules/glt_dual.py` 或独立小模块，旧factory默认不变 |
| 目标/架构配置与恢复 | `scripts/pretrain_glt_dual.py`、现有runtime |
| HEAD/LoRA/RIDGE | `scripts/finetune_glt_dual.py`及必要的小型适应helper |
| 选择与outer-test隔离 | 现有grid/evaluate，新增development模式而非滥用旧smoke的2epoch语义 |
| 测试 | 复用dual_glt/pretrain/speed tests，新增针对FGR/anchor/PEFT的少量用例 |

拟新增配置字段：`third_task=fp|none|fgr|align`、`fusion_variant=concat|anchor`、`geometry_features=length_angle|length_angle_torsion`、`chem_target=element|environment`、`adaptation=full|head|lora|ridge`、`evaluation_mode=smoke|development|formal`。ALIGN另记`projection_dim=128`、`temperature=0.1`、`alignment_ramp_updates=1000`、`contrastive_pool=distributed_microbatch`和identity来源。这些现在不是已有CLI。

配置需拒绝不支持的组合、遗漏真实路径、错误step/task身份；沿用必要的现有metadata检查，不另造全套hash系统。保存完整resolved config。移除FP时不得在dataset无条件构建7-RU指纹，测试monkeypatch该构建器以验证没有调用；BRICS读取不受影响。

新FGR label在CPU准备阶段产生，batch只传必要target/index；不把clean坐标传encoder。来源路径只读，若现有static缺少incidence，可从同一冻结物理拓扑派生，先小样本验证；未经授权不全量写新的派生缓存。

deploy仅包含推理encoder、融合、适配器和必要norm；FGR/ALIGN/chem/geo训练头与目标均不依赖。ALIGN两个投影头只存在于训练resume包，推理不需要负例、identity对齐标签或其他batch样本。LoRA部署必须保存base身份和adapter，并验证合并前后数值一致；不得把不兼容新结构部分加载成旧模型。

## 10. 阶段、预算与运行条件

### 10.1 全阶段通用约束

所有预算均为未来授权上限，不是已运行。失败的实际updates/epochs计入预算；到上限或STOP即交回，不自动重跑。短轨迹不代表正式预训练有效，30epoch开发不是100epoch正式结果。

GPU/worker/>1分钟任务只能在Linux本机 `tmux Uni-Poly` 独立window中执行并留日志。先检查现有进程，不停止其他路线；正常监控每10分钟一次，明显错误立即检查，避免高频轮询。

输出独立根：`results/glt_pred_20260918/`、`logs/glt_pred_20260918/`；每组/fold/seed独立目录。不覆盖历史；在阶段开始按实测吞吐与checkpoint大小估算时间/磁盘，并确认可用余量。不能为适应资源默改batch、样本或精度。

### 10.2 阶段表

| 阶段 | 内容与最大预算 | 交付/停止点 |
| --- | --- | --- |
| S0 | 只读来源/split/XC/重复身份审计；新目标覆盖只抽至多1024条P_train；全量identity匹配为元数据扫描、不跑模型 | 基线身份、P划分、有效覆盖、运行资源；合同错误先停止 |
| S1 | 实现HEAD/LoRA/RIDGE与validation-only开发路径；局部单测；fixed checkpoint xc/fold0至多2epochs | 无test访问、冻结/加载/梯度正确；不作排名 |
| S2 | 实现B_FP/B_NONE/T_FGR/T_ALIGN；至多32条不同真实smoke样本可复用；原三路径与FGR恢复/失败定位最多16updates；ALIGN额外2updates smoke＋连续4/2+resume到4共8updates＋最多2updates失败定位，总上限28；原geo/FGR partial/all-zero两case之外，ALIGN再3个3rank backward-only case（partial含重复identity、all-zero、全同identity），均0updates | 局部数学参考、全梯度归约/恢复/旧路径回退；小集合重复不是正式样本流或泛化证据 |
| S3a | 适应开发：3种神经适应×3tasks×2folds×最多30epochs=540epoch上限；RIDGE6个单元×4alpha=24次fit，无GPU训练epochs | 锁定后续统一适应方式；outer-test NOT_RUN |
| S3b | B_FP/B_NONE/T_FGR/T_ALIGN 四条新5k轨迹=20,000updates，seed42；四组×3tasks×2folds×30epochs=720epoch上限 | 对齐与重建分别比较；未通过则停止架构叠加 |
| S4 | 最多3个新增5k轨迹=15,000updates；每轨迹同3tasks×2folds×30epochs，累计540epoch上限；每新实现最多4updates correctness，总12updates；包含必要的扭转OFF控制 | 在额度内按第11节选择分支，不能执行全排列 |
| S5 | 最多3组×8tasks×5folds×100epochs=12,000epoch上限，seed42；不新增预训练；只部署锁定step5000 | 组别固定为新B_FP、最终候选、O8参照；无候选则不为填满预算做S5 |

S1的2epochs、S2的28updates、S4的12updates单独计入总账；研究预训练上限35,000updates＋40 correctness updates。开发神经微调上限1800＋2epochs；S5另计。相对r1新增一条5k、180开发epochs及12 correctness updates，不新增S4/S5组或seed。正式最多120个task/fold，仅全部相应授权后可运行。

S4默认优先ANCHOR与环境目标，各一轨迹；如果将名额用于扭转，必须占两轨迹（TOR/TOFF），与ANCHOR构成三条，不再运行环境目标。选择在S3审查后、启动S4前写明，不能看test决定。S4旧参考直接复用S3锁定轨迹，不重复预训练。

上段默认安排仅适用于FP/FGR parent。若parent为ALIGN，遵守§5.6的桥梯度限制，默认环境目标一条，或TOR/TOFF两条；余下名额可留给matched O8，不为填满名额引入新模块。若需要ALIGN＋FGR组合，须另行修订而非无监督地预训练ANCHOR。

S5 O8参照优先使用现存B的部署包和一致的微调协议，但来源P与新参考不一致时只能称历史参照；若需要严格matched O8，必须在S4三个预训练名额中预留一条O8-only，而不能临时追加第四条。初始授权若不含此项，则S5仅B_FP与候选两组80单元，O8历史结果旁列不作纯因果比较。

主预训练保持3GPU×84×accum4、1008、BF16、lr2e-4、wd0、warmup2000、horizon20000、seed42及现行clip；新增头初始化隔离。若授权改资源，另记数值/随机流差异，不能宣称bitwise matched。

新B_FP在P_train按位置顺序重新训练；所有组第5000步为主checkpoint，P_val每1000步检查健康，不以任务test选择checkpoint。P_val评估固定至多1024条身份分层样本，所有组同集合；小覆盖类别另报，不以它宣称全量重建验证。

S0的1024条目标审计用于决定FGR是否可实施；须单列无目标比例、SPD3覆盖以及各RU大小分布。若总体FGR有效覆盖低于50%，停止T_FGR并交回修订目标范围，不能自行加入侧RU多值目标；此阈值是执行可行性起点，不是论文结论。B_FP/B_NONE/T_ALIGN若各自条件满足且已获授权可以继续，不因FGR低覆盖而扩大ALIGN样本集合。ALIGN另统计按真实microbatch分组的有效pair数、不同identity数与无负例比例；身份无法证明时停止ALIGN。S3正式统计可扫描P_train全部候选以拟合mu/sigma，属于授权研究准备，须在tmux记录时间/数量；不把这次全量统计说成S0的1024条抽查。

## 11. 晋级和停止规则

### 11.1 开发晋级

各组使用相同validation样本、相同fold配对；选择规则预登记为：优先XC两fold平均validation R²增量≥0.01，且任一fold不下降超过0.03；eps/eat平均均不下降超过0.01。阈值仅为工程筛选，不是显著性保证。

- T_FGR同时报告相对B_FP和B_NONE，不能把去掉FP收益归因于FGR。若只优于B_FP、不优于B_NONE，优先保留简单B_NONE，不宣称融合目标有增量。
- T_ALIGN同样相对B_FP/B_NONE报告，并与T_FGR使用相同开发协议；不按两个不可比的预训练loss排序。对齐检索改善但XC/保护任务未达同一门槛则不晋级。四组共同使用锁定的下游适应方式，不能给ALIGN额外挑选LoRA/seed。
- S4模块相对其直接parent比较；候选不超两项组成的新增机制，避免无限堆叠。S4最多选一条作为最终候选。
- 若结果混合或接近零，记录INCONCLUSIVE并停止扩大；本计划不自动追加seed。需要确认seed时另修预算，并同时补参考。
- 对不同参数量/预训练头，报告模型、训练头、可训练参数、样本暴露、时间和峰值显存。不称同steps等于同FLOPs。

### 11.2 正式结论

S5固定全部配置后，每fold独立选validation最佳、恢复后test一次。报告R²/MAE/RMSE mean±std（ddof=0）、每fold配对差值和独立列示pooled OOF；每个样本每组恰好一次OOF。不以最佳fold、逐任务最优包络代表单模型。

主要比较candidate−B_FP；XC目标为平均R²正增量、至少3/5fold为正，整体macro8不下降超过0.005；如只能达成XC-specific收益，明确不能替换通用路线。这些是决策阈值，不是统计显著性结论。报告fold相关性与不确定性；不把5fold当5个独立实验。

### 11.3 立即停止受影响阶段

- 身份/物理路径/原子映射错误、跨样本或split泄漏；clean标签进入encoder。
- N=0被伪造中心监督，invalid几何被乘零掩盖NaN；DDP某rank不参与必要collective。
- teacher/旧checkpoint/schema误加载、出现持续NaN/Inf、writer冲突、覆盖已发布资产。
- 为维持收益需绕过现有parity/加载检查、删样本、换split或修改未授权结构。
- 超预算、资源不足或来源不清。可以继续只读定位，不自动扩大执行范围。

## 12. 必要测试与验收清单

1. 原B_FP开关默认保持旧输入/forward/loss和deploy行为；公共初始化可按名称核对，不仅“同seed”。
2. FGR中心pair身份、SPD2/3、无重复、采样覆盖、单位/归一化、对称端点、空pair、安全sum/count。
3. FP=none/fgr/align时不调用7-RU指纹构建，不读取无用FP标签；BRICS公共输入一致。
4. clean/noisy知识流测试、FGR对共享fusion的梯度、decoder无raw坐标入口。
5. N=0优先找真实冻结样本并记录key；找不到明确记录，用最小确定性fixture，禁止静默skip后声称覆盖；学生原子任务保留。
6. 普通连接、同原子双连接、左右不同键长、周期多重关系；所有物理身份分别保留。
7. 刚体平移/旋转、反射（首版cos扭转）、原子同步重编号、端点互换；固定关系浮点参考与重新构图分开测。
8. ANCHOR增量OFF复现原Concat；invalid/N=0时无bias残余，新增模块第一步存在合理梯度。
9. LoRA初始输出与冻结base一致，K/base不更新；HEAD/RIDGE冻结eval；adapter保存/恢复和合并预测一致。
10. train-only scaler/词表/统计；开发禁止test访问；resume拒绝smoke/development/formal混用和不同task身份。
11. 3rank partial/all-zero分别验证geo和FGR有效分母、finite backward；不要求所有合法参数每步非零。
12. 新head/采样恢复4步连续与2+恢复到4，包括各rank RNG、optimizer、scheduler、样本位置；复用工程日志尾部处理，不重写恢复框架。
13. 最终deploy移除训练目标与头仍可预测；读取同一冻结几何，缺失时走明示fallback；**当前方案不是纯2D部署**，不能宣称不依赖几何。
14. ALIGN跨rank多正例、仅一pair、无负例、padding、归约scale、远端key梯度、4次累积不等有效数，按§5.6单进程参考与真实DDP核对；全同identity必须有限零。关闭ALIGN后公共forward/RNG不漂移。
15. ALIGN部署前后预测一致；单样本预测不依赖推理batch内其他样本，不将对比学习温度/投影头带进性质预测。

局部测试通过只表示实现合同通过。新增目标重建变好、loss下降、attention非零均不能单独作为性质提升证据。

## 13. 产物与交接格式

不另建通用handoff框架；执行记录回填本文件。沿用现有run/summary/runtime/预测/部署格式，增加必要的任务和适应字段。

每阶段至少记录：实际命令、cwd、tmux window、日志、退出码、配置、来源与split、git commit、实际累计预算、失败/修复、是否读取outer-test。训练组另存loss/有效分母/梯度摘要、best validation、逐fold预测与参数/资源统计。

阶段报告只写已运行指标；未运行留空或NOT_RUN，不填模拟结果。S0审计产物、新目标统计和选型记录独立于active缓存。结果报告更新RESULTS，实际接口更新PIPELINE，完成审查后由Codex实质性归档PROJECT_HISTORY。

## 14. 交给执行模型的启动指令

1. 核对用户授予的阶段及预算；未授权研究训练时只完成允许实施/验证。
2. 检查git/remote/活动任务，安全pull，重新读取本计划；保护用户改动。
3. S0先确认参考、Xc来源、P划分及目标覆盖；不得直接启动四条5k。
4. S1/S2完成最小实现与表内验证，保留旧路径；输出实际`--help`和已验证的smoke/development命令。拟新增命令未经实现不得伪称已可运行。
5. 交回Codex审查；已获授权且验收满足后按S3→条件S4→S5推进，未满足条件停止并报告，不反复申请已授权的小步骤。
6. 每轮改动按AGENTS提交/推送并核对远端；只暂存本轮文件，不提交缓存、checkpoint、密钥或无关改动。
7. 执行者最后标“待审查”，不自行宣称计划关闭或模型性能提升。

## 15. 执行记录与下一步

| 阶段 | 状态 | 命令/日志/产物 | 实际预算 | 审查 |
| --- | --- | --- | --- | --- |
| 文档规划 | 已编制；Codex文档自检，非独立科学验收 | 本文件、当前Git提交 | 0训练/0实验 | 等待用户选择授权范围 |
| S0 | 未执行 | — | 0 | — |
| S1/S2 | 未执行 | — | 0 | — |
| S3 | 未授权、未执行 | — | 0 | — |
| S4 | 条件阶段、未授权 | — | 0 | — |
| S5 | 正式阶段、未授权 | — | 0 | — |

**下一步建议：先授权r2的S0–S2，完成XC适应接口、FP→FGR与独立ALIGN的正确性闭环；科学训练按四组S3预算另行明确。** 这是分阶段启动建议，不把完整路线截断为只写代码，也不将本文视为已获得全部实验授权。
