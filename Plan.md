# 借鉴 O8 的共享 RU 状态 3D 输入：实施与价值验证计划

## 0. 交接头与授权

| 字段 | 内容 |
| --- | --- |
| 计划 ID | GLT-CANON3D-20260920-01 / r2 |
| 日期／状态 | 2026-09-20；新增多 RU 方案已规划、待执行授权；r1运行状态须核实执行记录，不由本修订反推 |
| 用户要求 | 给出借鉴当前 2D 的 3D 输入优化方案，并制定完整价值验证计划 |
| 本轮行为 | Codex 按用户要求新增考虑跨两个 RU 的实现合同，仅修改文档；不启动实现或训练 |
| 角色 | Codex 规划和独立审查；接手执行模型实现、验证、回填记录；不默认子代理 |
| 基线 | r1为dev@79e3d2d；r2为dev@b61876d，修改前安全pull成功，Already up to date |
| 已有改动 | r2发现未跟踪的 `.zcodeignore`、`scripts/audit_canon3d_p0.py`；均属用户/执行者工作，不修改、不暂存；不据脚本存在宣称P0已完成 |
| 授权方式 | 可授权 P0–P1 实现和有界验证；P2 主研究、P3 条件归因、P4 多 seed 确认分别授权；P5 正式复评须单独明确授权 |

本文件中的新接口、配置和产物均为拟实施合同，不是已存在 CLI。禁止直接照抄不存在的命令运行。用户以后明确授权全部某阶段时不重复询问其内部普通步骤，但不能跨越未满足的条件或超预算。

**r2变更：**保留r1所有组和科学定义，新增§12的`L_MULTI2`多锚点物理观测折叠方案及两个匹配控制；它读取冻结Trimer内跨度为2的左右RU关系，不伪造中心到RU±2坐标。新增M0/M1阶段为独立待授权分支，不自动加入已有P0–P1执行。若原执行端正在工作，应先读取本修订，原r1任务可在原授权内继续；禁止将r2科学定义静默覆盖r1产物。

### 0.1 上一周期衔接，不抹掉未完成事项

`GLT-PRED-20260918-01` 的执行归档见 PROJECT_HISTORY.md 中 2026-09-19 条目；该条是执行者记录，不等于 Codex 完成全链验收。此前 Codex 已独立复算 S5 的 80 个预测单元，逐折 test indices 与固定 manifest 一致，指标最大差异约 2.22e-16；B_NONE 的 XC 正增量仅 2/5 折，按原 gate 保留 B_FP。历史 FGR/ALIGN/ENV/TOR 增量未建立，本计划不原样重跑。

2026-09-20 源码仍显示两项收尾缺口：S5 launcher 在外层寻找实际位于嵌套目录的完成标记；aggregator 逐折指标直接采用 metrics.json，未内置对固定 split 的逐折检查。新计划不覆盖旧产物、不重跑旧 outer-test，也不替旧周期宣告 CLOSED。P0 只读记录其状态；若复用相关调度/汇总代码，须在 P1 修复这两个现实缺口并补 CPU 回归。修复后对已有 CSV 重聚合不属于重新模型测试，输出到独立审查目录。

## 1. 科学问题与可证伪假设

**目标：以单 RU 的 M 个原子状态承载 3D 表示，保留独立物理 image 关系，替换“全 Trimer 物理键更新后读中心键”的 GLT 分支。** M 指 RU 重原子数，不是旧 N+1/N+2 的内部键数。

待回答：

1. 真实几何输入是否比同结构、相同几何监督但关闭几何输入的模型提供增量？
2. 新双路方案是否优于 matched O8-only，及当前 B_FP/GLT 参考？
3. 若有效，动态读取共享 canonical 状态是否比侧环境固定编码更有价值？
4. 相比完整物理 atom 状态，新方案能否以更低成本保持预测水平？

不预设 3D 有用，不以 loss 下降作为最终收益。负结果可以完成周期。此方案是开放 Trimer 上的状态共享近似，不是精确无限周期 3D 模型，不保证 RU 切分不变性、构象分布或体相性质可辨识性。

### 1.1 论文依据与项目新增

| 原文 | 可借鉴 | 不可移植的结论 |
| --- | --- | --- |
| [GRIN，NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/7fe3921147c968d0b57a224c0d07e21d-Abstract-Conference.html) | 聚合物重复表示一致性需要显式设计 | 不证明单 Trimer 几何充分；本方案未实现其图对齐/增强 |
| [SpaceFormer，ICML 2025](https://proceedings.mlr.press/v267/lu25e.html) | 局部键几何之外的空间信息值得研究 | 本方案不使用其空间网格，也不继承其成绩 |
| [AMS，JCP 2025](https://doi.org/10.1063/5.0258496) 及 [COSMO-RS 中心单元方法](https://www.scm.com/doc/COSMO-RS/Polymers_With_COSMO-RS.html) | 邻接 RU 提供环境、中心 RU 提供代表性表征 | 不把距离网络当作量子化学表面计算 |
| [Matformer，NeurIPS 2022](https://arxiv.org/abs/2209.11807)，较早方法先例 | 有限主体状态与多 image 关系分离 | 开放 Trimer 无真实晶格，不能继承严格周期性结论 |

本项目新增的是：共享原子状态、独立真实 Trimer 距离关系、中心输出与现有 O8/Concat 的组合及下述受控实验。没有找到完全相同且已证明适用于本项目的现成方法。

## 2. 锁定不变项与范围

* O8 输入、六层数学实现、SPD/path、atom masking 与化学任务保持原样；禁止顺手修改其 attention scaling 或最短路径策略。
* 下游 Concat、LayerNorm、属性 head 和 FULL 适应策略保持；本轮不增加原子级跨模态融合。
* 所有双路主组保留 chem + 现有中心键长/角度 reconstruction + FP，权重沿用 `[1,1,0.1]`。保留 FP 是为匹配已保留的 B_FP，不宣称其最优。
* 不增加 FGR、ALIGN、ENV、torsion、teacher/KD、MD200、多任务属性 loss、ensemble、能量/力监督。
* 只读现有冻结单构象 Trimer；不生成/优化坐标，不新增真实RU±2坐标，不覆盖旧 LMDB/static/targets。r2允许读取现有RU−1与RU+1之间的真实距离，其相对shift为±2，详见§12；这不等于生成RU±2。
* 仅当前支持的双端线性均聚物；支化、多 attachment、未编码的立构/共聚序列不补猜。
* 学生/主模型数据集合不按新几何成功率二次筛选。invalid 样本仍参与可用的化学、FP和下游任务。

## 3. 输入合同：共享身份，不平均物理观测

### 3.1 adapter

拟新增 `CanonicalGeometryBatch`（可复用现有 Data 与 collate，不新建通用框架）：

```text
canonical_z             [M]
canonical_to_o8         [M]
image_z                 [3M]
image_to_canonical      [3M]
image_ru_offset         [3M]        # -1/0/+1；仅身份审计
central_image_index     [M]
image_pos               [3M,3]      # 仅输入准备；不作为 encoder 绕过特征的旁路
physical_bonds          [2,E] + bond_type/stereo/conjugation
relation_target         [R]         # canonical i
relation_image_source   [R]         # 实际 image atom
relation_source         [R]         # canonical j
relation_features       [R,F]
center_bond_endpoints    [B,2]       # 旧目标的真实中心内部键
geometry_valid          [graphs]
graph/image/relation pointers; sample keys; 既有来源身份字段
```

先核实缓存重原子映射，再要求 3M；不能按数组排列推断身份。检查元素、内部键、seam 和 stereochemistry，复用已验证的 normalized identity。含重原子端帽/残留 dummy 导致不满足合同则阻断该实现判断，不能静默截断。侧 RU 端部化学不一致须报告，不冒充无限链内部。

### 3.2 首版使用完整中心→image 关系，不使用几何选邻居

对每个 `(i,0)`，访问全部 `(j,q)`，只删除 exact self `(j=i,q=0)`：

$$
R_i=\{(j,q):1\leq j\leq M,\ q\in\{-1,0,1\}\}\setminus\{(i,0)\}.
$$

每图 `R=M*(3M-1)`。保留 `(i,-1)`、`(i,+1)`，不与 identity self-loop 混同；左右距离不平均，不按 `(i,j)` 去重。

**选择完整关系是为了归因：GEO_OFF 与 GEO_ON 的关系成员完全相同，关闭距离后不能从 radius/TopK membership 继续读到隐藏几何。** 不宣称全连接最优；局部邻域与稀疏化排到本周期之外。

复杂度为 O(M²d)，不是 O(M)。相对后述物理 atom 对照，将持久原子状态从 3M 减至 M；当前旧 GLT 是 bond 状态，不能直接用 3M 描述其状态数。P0 统计 M/R 的 p50/p95/p99/max，P1 在大样本上检查精确 query chunk。不得悄悄截断大 RU、删样本或改 TopK；无法在预算内处理时停止交回修订。

### 3.3 关系特征

每条关系：实际 distance、真实 bond type（含 nonbonded）、stereo/conjugation（仅真实有键关系有效）。不将 RU 内键机械复制成跨 image 键。不输入 signed q、任意原子序号或额外可学习左右标签。

距离编码固定为 64 Gaussian RBF（中心 0–25 Å，宽度为中心间隔）、`log1p(d/25)` 和 `d>25` bit；这些是工程起点，不是论文最优值。P0 报告超范围比例，不根据下游分数调整范围。使用真实尺度，不逐分子缩放距离。

同名 image 身份只用于索引和审计，不另增加同名 embedding；关系保留本身已允许不同 observation 形成不同消息。远距离、非键、左右观测都保留。

平移/旋转/反射不变性来自距离输入；这会丢失单独由几何手性决定的信息。已有化学 stereo 保留但不声称补全所有立构信息。无显式角度/torsion 输入，避免和已失败增强混为一轮。

## 4. 模型合同

### 4.1 主候选 S_SHARED

* M 个 persistent atom states，hidden=512、layers=6、heads=8、head_dim=64、FFN=2048、dropout=0.1。
* `h0=Embedding(element)`；3D 不读取 O8 hidden/原子 mask，不复制 O8 全部137维化学输入。关系提供键化学。
* relation encoder 为 `Linear(F,64) → GELU → Linear(64,64)`，输出 p_e∈R^64，一次 forward 内静态，每层有独立 bias/value 投影。键化学使用现有明确分类的one-hot，nonbonded独立分类；F由固定特征表确定并记录，不复用不兼容的旧距离投影权重。
* 采用 target-query/source-key，scale=1/sqrt(64)，target-wise softmax；这是新3D模型，不改现有共用 SourceAttention512。

$$
a_{ijq}^{l,r}=\frac{(Q_{l,r}x_i)^\top(K_{l,r}x_j)}{\sqrt{64}}+b_{l,r}(p_{ijq}),\qquad x=\operatorname{LN}(H^l).
$$

$$
m_i^{l,r}=\sum_{(j,q)\in R_i}\operatorname{softmax}_{R_i}(a^{l,r})[V_{l,r}x_j+P_{l,r}p_{ijq}].
$$

拼接 heads，经 output/dropout residual，再 Pre-LN FFN residual。每层读同一份旧 H，层末统一写回。按 query chunk 必须数学等价；若 key chunk 使用全局归一化，禁止局部 softmax 后相加。

无额外 identity relation，residual 保留自身；exact self 不参与距离 attention。所有 image 读取 canonical `h_j^l`，但各自几何 conditioning 不同。**不独立更新侧 RU，不等于侧 RU 只提供固定内容；共享 h_j 会随层变化。** 不假定某个 h_j 就是每个物理 image 的精确真实状态。

### 4.2 输出与旧监督适配

```text
atom_hidden      [M,512]
graph_3d         mean(atom_hidden), [graphs,512]
bond_states      [B,512]  # 仅用于保留原预训练目标的适配
geometry_valid   [graphs]
```

对中心内部键 `(a,b)`，`u_ab=MLP([h_a+h_b, abs(h_a-h_b), h_a*h_b])`，1536→512→512；端点对称，不读取该键原始 clean 几何或标签。将其送入既有非 affine geometry LayerNorm 和相同 length/angle decoder。angle decoder 仍以两条中心键的对称组合预测原目标，不添加新的几何任务。

下游只使用 graph_3d，经原 norm3 与 O8 graph_2d Concat 后属性 head；invalid 在 norm3 后屏蔽。M=1/B=0 时可有几何 atom readout 和 FP/task 梯度，但不生成中心键/角度目标，也不进入其有效分母。此为新架构边界，不能改写旧 N=0 合同或用跨键伪造中心键。

### 4.3 严格配对控制

* **G_OFF**：与 S_SHARED 同结构、所有初始参数、关系行、sample顺序与损失。RBF/log/overflow 全部替换成一致零输入，不向 encoder 提供原坐标、rank、几何摘要或角度。`geometry_valid` 仍沿用同一集合。它保留几何监督与失败状态，不是“完全没有3D知识”的模型。
* **E_STATIC（条件）**：只有 q≠0 的 K/V 改为读取 `LN(h0_j)`；q=0 仍读 `LN(h_j^l)`。层参数、relation特征与主组一致。h0 是可学习 embedding 输出，不 detach，但不递归更新侧源。用于直接检验动态 canonical 内容相对固定环境内容的增量。
* **P_PHYSICAL（条件）**：同样模块、参数及初始化，维护真实3M原子状态；每个物理原子访问其余3M−1个原子，真实化学边与距离；最后只读中心M个原子，监督仍为相同中心键。3M初始化由同一元素embedding给出，不使用位置embedding。该组同时改变侧状态独立性与侧上下文计算，不能把差值称为纯参数共享因果效应。中心 query 的关系/decoder/readout严格与S一致。

## 5. 实验矩阵与解释

| ID | 预训练/预测路径 | 要回答的问题 |
| --- | --- | --- |
| R_GLT | 当前 B_FP 原 GLT + O8 + Concat | 现有方案参照；不修改架构 |
| R_2D | O8-only；FP保留，geo loss无；Concat几何槽恒零 | 没有3D分支的同数据、同O8训练预算参考 |
| G_OFF | 新共享分支，距离全关；geo监督保留 | 新结构/容量/几何监督控制 |
| S_SHARED | 新共享分支，真实中心→image距离 | 主候选 |
| E_STATIC | S的侧source内容固定为h0 | 有条件的动态状态归因 |
| P_PHYSICAL | 同atom模型，完整3M动态状态 | 有条件的表达/成本比较 |
| L_NEAR_GEO（r2） | M个共享状态，多物理锚点；仅跨度≤1的距离开启 | 多锚点结构及近域几何控制；跨度2关系仍在 |
| L_MULTI2（r2） | 同上，全部真实关系距离开启，含跨度2 | 两个RU边界之间真实几何的增量 |
| L_OFF（r2） | 同上，全部距离输入关闭 | 多锚点模型的整体几何输入控制 |

R_2D 使用 `[norm2(g2d), zeros(512)]`，FP和属性head与双路维度相同；推理不读几何，预训练不生成geo target；额外零列不是活跃容量匹配。沿用相同cohort，即使它历史上由几何成功筛选也不扩数据。R_2D−双路是整体路线对比，不是相同参数量控制。

必须分别报告：S−G（显式距离输入价值）；G−R_2D（容量/监督等综合增量）；S−R_2D（总价值）；S−R_GLT（替换价值）；S−E（动态侧source价值）；S−P（表达/成本权衡）。不得仅因 S−R_GLT 为正就宣称状态共享单独有效。

历史 S5 的 B_NONE 不作为本轮主参考。R_GLT 仅当第6节全部身份/训练条件匹配时复用既有5k及development结果；不匹配则在预算内重训一次。不能复用已有outer-test去筛选新方案。

## 6. 数据、初始化与预训练

### 6.1 只读事实源

核验而非假定存在：

* PI1M：`data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1` 及其绑定的 static/targets/cache bundle。
* 下游：`data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1`，`data/raw/smi_<task>.csv`。
* P split：`results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json`；历史P_train=911391仅作记录，实际以manifest和来源审计为准。
* 公共初始化：`results/glt_pred_20260918/s3b_prep/common_init_v1.pt`。
* 固定下游split：`data/splits/mips_outer5_inner20/`，不得覆盖。
* 参考配置：`configs/mts/glt_pred_s3b_b_fp.json`；参考部署：`results/glt_pred_20260918/s3b_formal/b_fp/pretrain/deploy_05000.pt`。

P_train排除下游身份的既有规则、P_val固定集合继续使用；不能为新路线扩大P_train。若发现新的identity/来源错误，停止受影响比较，不悄悄改manifest。XC标签来源、重复身份问题只读记录，缺失条件不捏造。

### 6.2 公共随机性与初始化

* 初始 O8、atom head、Concat norm和FP/属性head按名称复制同一initial-state，而非只写同seed。
* 新S/G/E/P所有同名同shape参数完全相同；P复用embedding初始化物理原子，不产生额外可学习image索引。
* 同一seed下样本顺序、30% atom mask、坐标噪声相同；几何准备使用独立随机流，不影响O8 dropout/mask。G_OFF仍消费相同噪声抽样但不把结果交给encoder。
* 额外模块初始化、几何dropout与公共2D随机流隔离；P0核实历史参考是否满足相同公共流。无法核实则不称历史参考为严格初始化匹配，应重训参考或标注并停止正式晋级。
* 多seed确认时seed43/44重新建立各自公共初始化与配对轨迹，不把改变下游seed冒充预训练稳定性验证。

### 6.3 训练目标保持，不重开失败任务

使用同一次noisy坐标（σ=0.03）重算全部距离输入；clean仅构造原中心键长、中心内部键对角度target。完整relation membership不依赖距离，无clean近邻选择旁路。数据预处理不可将未加噪摘要偷渡到encoder。

原化学masked-atom CE、原FP target及其来源保持。geo目标集合、reduction、权重、decoder架构与参考保持；新atom→bond适配是明确架构差异。G/S/E/P均使用相同适配。旧B_FP原bond encoder不强行换decoder路径。N=0/无角度/invalid的每项分母分别归一化；空目标返回图连接的有限零，不对NaN乘零。

```yaml
seed: 42
world_size: 4
microbatch_per_rank: 84
gradient_accumulation: 3
global_batch: 1008
precision: bf16
optimizer: AdamW
lr: 0.0002
weight_decay: 0
warmup_updates: 2000
schedule_total_steps: 20000
end_lr: 0.000000001
stop_update: 5000
deployment_step: 5000
third_task: fp
loss_weights: [1, 1, 0.1]
```

沿用历史20k scheduler的前5k，而非偷改5k cosine；固定step5000导出，不根据P_val或下游挑不同step。P_val仅检查训练健康。P2前若确实OOM，可统一所有新跑组调整micro/accum，保持global1008，并重新验证公共随机与reduction；改变后历史结果不能自动strict matched。调整须先交回修订并锁定，不能各组独立减batch。

## 7. 下游协议：先开发，再确认，最后可选正式复评

### 7.1 开发筛查

XC、EPS、EAT × folds0/1；每组6单元。读取原manifest的最终train与validation，**不创建outer-test loader、不在epoch日志计算test、不用S5逐样本残差指导本轮调参**。

* FULL，seed42；最多30epochs、warmup5、patience10。
* train-only label scaler；Huber及其beta、batch、weight decay、dropout等以核实的S3b runner解析配置为准，P0输出resolved参数后锁定。encoder LR1e-5，fusion/head LR1e-4，batch32/eval64，finetune wd0.02。
* 相同validation R²规则选每单元best；完整保存最佳checkpoint与有限validation预测。
* 所有组使用同一个30epoch scheduler长度；不与100epoch或不同协议结果混排。
* 6单元仅为开发筛查，不宣称八任务正式收益或独立盲测。

### 7.2 预登记晋级条件（工程阈值，不是显著性）

先检查S−G：XC两fold平均R²增量≥0.01，且两fold均>0；EPS/EAT平均各不得下降超过0.01。再检查S分别相对R_2D、R_GLT：XC mean增量≥0.01、任一XC fold不低于−0.03，EPS/EAT平均各不低于−0.01。

全部满足：进入P3；只在部分非XC任务改善：记录task-specific线索，不自动改主目标或晋级；S≈G即使两者都强于旧GLT，也不能判为真实几何增量。均未满足则STOP，保留负结果，不开启新loss/neighbor sweep。数值近零只称未建立增量，不能声称等价。

### 7.3 条件归因与多seed确认

P3只增加E/P两组，预算各5k+6开发单元；不根据其分数临时更换主要候选S。S−E判断动态侧source是否有帮助；S−P只能表述共享/展开整体差异。若P明显更优则S的压缩假设存疑，不自动启动P全量路线。

P4只确认S、G及预先锁定的最强参考R*（R_GLT/R_2D中，按seed42 XC开发均值较高者，完全相同则R_GLT）；新增seed43/44，每seed三条5k和18个开发单元。

确认要求：跨3seed/2fold的S−G和S−R* XC平均各≥0.01；每个seed的XC折均差均>0；每个task EPS/EAT总体平均退化不超过0.01；报告全部seed/fold差异，不能将6点视作独立实验。若失败，STOP。R*锁定、gate只用于开发，不访问outer-test。

### 7.4 P5正式复评：单独授权，不以已看过的test作为盲测

P4通过后才提交执行请求。候选S、G、R*三组，固定seed42、部署step5000，八任务×五fold×最多100epochs、patience10，最多120单元；不新增预训练。每fold独立validation选best后test一次，不train+val refit，不按test选择seed/epoch/架构。

当前outer-test已经被历史开发和S5接触；这次只能称**固定既有五折协议下的复评**。新留出或嵌套外层评估需要另行规划、数据与预算，不在此计划中伪造独立性。

正式价值判据（S分别对G、R*）：XC mean Δ>0、≥3/5fold为正、macro8 Δ≥−0.005；另报所有任务R²/MAE/RMSE mean±std(ddof0)和pooled OOF，不能挑任务包络。通过仍非统计显著或盲测证明。R*既有正式产物只有满足同一初始化/预训练/微调/split等全部身份时可只读复用，不重新访问test；复用减少预算，不腾额度追加组。

## 8. 实施路径与边界测试

优先复用稳定reader/runner；拟新增：

```text
src/dataset/canonical_geometry.py
src/modules/canonical_geometry.py
tests/test_canonical_geometry.py
configs/mts/canon3d_{shared,off,static,physical,o8}.json
scripts/aggregate_canon3d.py
```

在现有pretrain/finetune入口显式选择3D family，不改变旧默认。部署包保存family、state_mode、geometry_input、step和维度；这是防止M状态与旧bond/3M架构静默加载所必需的最小区分，复用原metadata机制，不另建通用schema框架。旧checkpoint严格走旧分支，新配置不能静默部分加载。

P1必须覆盖：

1. M=1、多原子、环、相同attachment原子、不同左右长度；中心↔O8↔image同步重排后身份一致。
2. S/G/E关系数M(3M−1)，P关系数3M(3M−1)；exact self删除、同名侧image保留；S的persistent atom tensor始终M行。
3. 平移/旋转/反射、物理记录左右交换、原子同步置换；不要求重新生成构象或不同RU切分严格等价。FP32起点atol/rtol1e-5，记录实际误差。
4. G_OFF坐标扰动下encoder/task输出不变（固定valid、化学、噪声/随机流）；geo loss可以变化，因为target仍来自几何。S在专门构造非退化fixture上对距离变化有响应；不能要求每个真实样本每参数都有非零响应。
5. S与E初始共享张量相同；E确实只在外侧读h0，S读当前h；P中心relation与S完全一致。全局softmax chunk前向/梯度与小型dense参考一致。
6. 噪声从同一image坐标副本生成，距离几何自洽；clean标签不进入encoder。N=0无中心键/角度分母，但原子/FP/task可有效。
7. 从冻结数据最多查1024条中的真实N=0/invalid并记录key；找不到就明确使用最小fixture，禁止无限搜索或silent skip。
8. DDP partial-empty与all-empty几何目标，collective/backward一致，chem/FP继续；0 optimizer updates的分布式fixture即可，不重复大实验。
9. 部署strict-load往返、wrong-family拒绝；R_2D不读geometry；各组invalid fallback有限。新前向与原O8单独输出parity。
10. 4步连续vs2步恢复至4步，核对位置/RNG/optimizer/scheduler；本轮累计按两条轨迹各4步计8 updates，不把断点前两步重复漏计。日志尾部异常沿用已验证恢复策略。
11. launcher正确识别实际产物布局；已完成skip、不完整拒绝；aggregate从CSV重算逐折指标并对固定manifest验证有序row_indices、finite和OOF覆盖。测试不得调用真实outer-test模型。

## 9. 阶段、预算与 STOP

所有数字均为待授权上限，不是自动执行额度。各阶段结束交回审查；未通过不消耗后续预算。

| 阶段 | 工作 | 最大运行预算 |
| --- | --- | --- |
| P0 | 身份/源码/资源/数据长度审计；冻结baseline配置 | 元数据检查；最多1024条真实记录，不跑模型 |
| P1 | adapter/模型/部署/聚合实现与局部单测 | 6条路径各2步=12 updates；S恢复验证8 updates；合计20；每条路径下游XC fold0最多2epochs，共12epochs；DDP空目标0updates |
| P2 | R_GLT/R_2D/G/S主比较 | 4×5000=20000 updates；24开发单元×30=720epochs |
| P3 | 条件E/P归因 | 2×5000=10000 updates；12开发单元×30=360epochs |
| P4 | S/G/R*的seed43/44确认 | 6×5000=30000 updates；36开发单元×30=1080epochs |
| P5 | 三组八任务五折固定协议复评 | 单独授权，120单元×100=12000epochs；无新增预训练 |

r1研究上限：12条5k轨迹=60000 updates，另20 correctness updates；开发上限72单元/2160epochs，另12 smoke epochs。r2独立新增M0/M1上限见§12.8；两分支全获授权且全部实际执行时，总上限为75000研究updates＋26 correctness updates、90开发单元/2700epochs＋18 smoke epochs。P5仍最多120单元，不因r2自动增加名额。复用/早停会减少实际量，不以剩余额度增加实验。P1后先只申请P2，不默认用满P3/P4/P5。

以下情况停止受影响阶段：身份错误、跨样本边、invalid作为有效几何、target泄漏、持续NaN、架构静默加载、样本集合/FP/初始化不匹配、预算超额、outer-test提前访问、GPU/内存不足需改变科学合同。可以继续只读定位，不用增加训练来掩盖失败。

## 10. 资源、产物与报告

运行端先检查GPU/CPU/活动进程和tmux，不停止已有用户任务。所有GPU、worker、预计超过一分钟或写训练产物的任务在`Uni-Poly`独立window，保存命令、cwd、环境、日志、退出码。默认10分钟检查一次，阶段退出/异常及时处理，不高频轮询。

新根目录：`results/glt_canon3d_20260920/` 与 `logs/glt_canon3d_20260920/`；按stage/arm/seed/task/fold隔离，禁止覆盖旧目录。源码不保存大产物。复用当前reader和只读缓存，不全量构建新派生缓存；关系先按需计算，若成为瓶颈另行提案。

每run保存resolved_config、公共初始化来源、sample/split来源、完整命令、runtime/退出状态、training metrics、deploy/最佳checkpoint、validation预测。只有P5才写outer-test预测。部署加载必须实际测试，路径相同不等于内容身份相同，复用既有checkpoint fingerprint机制。

成本报告：参数量、可训练参数量、relation count分布、峰值显存、完整窗口samples/s、预训练墙钟、下游训练/推理时间；所有性能对照同硬件/线程/batch和样本清单。最多64条固定train样本做read-only推理计时，预热5次/测20次，不用outer-test；这是smoke成本证据，不是完整服务吞吐承诺。

最终至少输出：数据与源码审计、正确性报告、逐arm/task/fold/seed表、所有配对差值、成本表、失败记录和原假设裁定。避免只交一个macro数字。

结论分类：

* S不优于G：`GEOMETRY_INCREMENT_NOT_ESTABLISHED`。
* S优于G但不优于两参考：`GEOMETRY_SIGNAL_WITHOUT_ROUTE_ADVANTAGE`。
* P2/P4通过但未做P5：`DEVELOPMENT_CANDIDATE_ONLY`。
* P5固定条件通过：`FIXED_PROTOCOL_VALUE_SUPPORTED`，仍注明非独立盲测。
* 性能无明显劣化且更省资源：可报告成本候选；若未预设等价性界限和统计检验，不能称为已证明等价。

## 11. 执行记录与下一步

| 项目 | 当前状态 | 证据/限制 |
| --- | --- | --- |
| 方案与预算 | 已编制 | Codex文档自检，不是独立实施验收 |
| P0–P1 | 待授权 | 未修改模型/runner，未运行单测或smoke |
| P2–P4 | 待授权 | 无新预训练、微调或候选指标 |
| P5 | 待单独授权且受晋级条件约束 | 不读取新outer-test；既有测试已见过 |
| 旧S5收尾 | 仍有源码缺口 | 不因本计划建立而关闭旧周期 |
| r2多RU分支M0/M1 | 待授权 | 仅制定§12；没有据未跟踪脚本推断测试或审计已完成 |

下一步建议只授权P0–P1。执行者核对最新AGENTS/本计划、保护工作树、安全pull，然后补必要实现和有界验证；提交实际接口/`--help`、真实命令及产物位置，不伪造CLI。P1结束回填本节并交Codex审查，再决定P2。

研究周期结束后，Codex将最终合同、执行、审查和失败项追加PROJECT_HISTORY，明确暂无后续或另立待授权计划；不提前宣布价值成立。

## 12. r2新增：多 RU 相对关系折叠 L_MULTI2

### 12.1 先区分三个概念

1. **拓扑跨RU**：2D用`(atom_id,q)`展开化学图，通过seam连接跨单元；拓扑上可出现任意q，不需要坐标。graph hop数与RU跨度不同，两跳不一定跨两个RU。
2. **现有跨度2几何**：冻结Trimer中`(i,-1)`与`(j,+1)`的距离是真实可计算观测，`delta_q=+2`；反向为−2。r1 S只以q_target=0为锚点，因此未直接读取这些关系。P_PHYSICAL会读取它们，但其状态独立，而非本节共享折叠。
3. **缺失的中心±2几何**：`(i,0)`到`(j,+2)`没有现成Trimer坐标。`d[(i,-1),(j,+1)]`通常不等于它，禁止重命名为中心±2距离。多层共享消息的可达性也不是新增几何观测。

本节只实现第2类；第1类提供身份/相对shift的组织思想，第3类保持缺失。若用户目标必须是实际中心到RU±2，应先核实既有更长链缓存；无合规坐标则另立数据计划，不靠本方案宣称已解决。

### 12.2 从N+1/N+2继承什么

* 先枚举物理对象，再映射canonical状态；不同物理副本不能因canonical ID相同被删除。
* 继承N+2保留左右真实观测的原则，不继承N+1先平均左右长度的处理。
* 不仅保留两条跨键：本节保留全部已观测原子对，包括实际跨键、跨一个RU的非键关系及跨两个RU的非键关系。
* N+2指N个中心内部键加两条跨键，不表示它已包含中心RU±2的几何。
* 仅重新编号时，source、target、路径/角度所依据的所有物理身份必须同步；重新编号不改变真实坐标。不同角度不能因canonical端点相同而平均；本节不新增角度输入或目标。

### 12.3 输入：从单锚点扩展为多锚点观测

保持M个persistent states。对实际存在的全部物理原子构建有向关系：

$$
e=((j,q_s)\to(i,q_t)),\quad q_s,q_t\in\{-1,0,1\},\quad (j,q_s)\ne(i,q_t).
$$

每条记录保存：

```text
source_canonical=j; target_canonical=i
source_image_id; target_image_id
source_q=q_s; target_anchor_q=q_t
relative_shift=q_s-q_t
distance=norm(pos[source_image_id]-pos[target_image_id])
bond_chemistry=该物理原子对的真实键属性，或nonbonded
anchor_group=(graph_id, i, q_t)
```

`((j,q_s),(i,q_t))`是唯一物理有向关系身份；`(i,j,relative_shift)`只是折叠后的拓扑身份，**不能作为物理去重key**。同一相对shift可能有多个不同实际观测，均保留。signed q/relative_shift只用于索引、分组、跨度控制和审计，不引入新的可学习方向embedding。关系特征、RBF和模块均沿用r1。

关系总数`3M*(3M-1)`，分类计数：跨度0为`3M*(M-1)`，跨度1为`4M²`，跨度2为`2M²`。必须验证各类之和与物理枚举一致。跨度2仅连接现有左右RU；只删同一个物理原子的exact self，保留同canonical的不同副本。

这种折叠类似2D的`(i,q_t)→(i,0)`重新定锚，但本计划保留`target_anchor_q`作为来源，**不宣称坐标也随之成为中心RU的坐标**。模型学习的是由三个实际局部环境共同估计的代表性RU状态，不再是纯粹的中心局部几何状态。这是相对r1的明确科学变化。

### 12.4 更新：共享状态，不让观测数量决定锚点权重

每条关系读取同一层旧`h_i^l,h_j^l`。用r1相同Q/K/V和relation bias/value模块，在每个`(graph,i,q_t)`内独立softmax，得到临时message `m_{i,q_t}^l`。三个锚点message不是三份persistent states，不各自执行FFN或带入下一层。

固定组合：

$$
m_i^l=\tfrac12 m_{i,0}^l+\tfrac14 m_{i,-1}^l+\tfrac14 m_{i,+1}^l.
$$

组合后仅执行一次r1的output/residual/FFN更新M个H。权重是预登记工程起点，不是物理最优权重，不在本轮扫参。中心保留较高权重；左右共享权重，使同步左右交换不改变定义。不要先把所有观测混成一个softmax，否则边数量会隐式改变锚点权重。

所有几何有效图三锚点均须可证明存在；缺失/坏坐标不能仅跳过某锚点后静默重归一化，应进入统一invalid fallback并报告。不得因L失败而单独删除下游样本。

输出仍为M行代表性RU atom states，使用相同中心canonical索引与O8对齐、mean readout和atom→bond辅助适配。训练标签仍来自中心RU，**不将侧RU或跨度2关系新增为监督目标**。表示接受侧观测不意味着标签也被复制三份。

### 12.5 三组配对，不让边数量混入跨度2几何归因

| 组别 | 关系行/锚点聚合 | 距离输入 |
| --- | --- | --- |
| L_NEAR_GEO | 完整物理关系，固定1/2、1/4、1/4 | `abs(relative_shift)<=1`开启，跨度2的RBF/log/overflow全置零 |
| L_MULTI2 | 完全相同 | 全部开启，包括实际左→右/右→左 |
| L_OFF | 完全相同 | 所有距离输入全置零 |

三组参数量、初始化、relation顺序、valid集合、noisy坐标、decoder、目标、优化器、公共随机流全部一致。L_NEAR_GEO仍可通过其他距离推断跨度2的部分信息，且仍有跨度2关系，因此只称“跨度2显式距离关闭”，不称没有跨两个RU传播。L_OFF仍有几何监督，不能称完全无3D知识。

必须报告：L_MULTI2−L_NEAR_GEO（显式跨度2距离增量）；L_MULTI2−L_OFF（全部几何输入增量）；L_MULTI2−S_SHARED（多锚点与上下文范围整体增量）；L_MULTI2−P_PHYSICAL（若P已有匹配结果，则为共享折叠/物理独立更新整体比较）。不能将L−S全归因于跨度2，也不能将L−P全归因于参数数量；新组没有额外可学习参数，但关系计算比S约多3倍，不能承诺提速。

### 12.6 多 RU 的通用接口与观测范围上限

adapter以实际`observed_ru_offsets`而非硬编码3定义物理观测集合。对连续K=2w+1个真实RU，`q∈[-w,w]`，最多可观测相对跨度`2w`；关系总数`KM*(KM-1)`。这只是接口和合成fixture能力，不是新增数据授权。

当前锁定K=3、w=1、最大跨度2。扩展真实K>3之前必须有合规已冻结坐标、完整映射/端部协议和新的公平对照；本周期不生成P5、不复制Trimer坐标填充、不用路径键长相加冒充直线距离。2D可展开到更远q的纯拓扑关系只作审计参照，不以“距离0”伪装缺失几何输入。

K>3的锚点权重/目标范围需另行锁定；不得直接沿用本节三个权重后随意丢弃其他锚点。接口测试不代表真实多RU覆盖已验证。

### 12.7 M0必要验证与停止条件

复用P0已审计真实样本，不再额外扩大真实扫描上限。增加如下定向fixture和测试：

1. M=1时应有6条物理有向关系，其中2条跨度2；q=0主体仍1个state，无中心内部键目标。真实N=0未找到仍须保留确定性fixture，不silent skip。
2. 不等左右长度及弯曲Trimer中，分别验证所有距离源于原始两端坐标；不把左→右距离写入中心→RU+2字段。不能从“重新定锚”生成新的坐标。
3. 关系反向后delta变号、距离相同；侧记录与索引同步交换、canonical重编号、刚体变换后输出符合预期。相同canonical不同物理pair不能去重。
4. 以真实Trimer化学图/2D lifted seam规则核验化学身份；区分hop数、seam穿越次数、净relative_shift，不把abs(delta)=2误记为SPD2。
5. 仅从L encoder feature边界扰动跨度2距离：L_NEAR_GEO和L_OFF前向不变；L_MULTI2在非退化fixture有响应。这是算子测试，任意单独改距离不等于一个合法新构象。合法坐标变换的测试另列。
6. L_NEAR_GEO/L_MULTI2首步公共2D随机流一致；L_OFF无坐标特征旁路；噪声在同一实际Trimer副本上生成，所有物理pair一致重算。
7. 分组softmax、固定锚点加权与小型dense参考前向/梯度一致；layer末才统一更新M行；临时3M消息不作为3M隐藏状态递归。
8. 将锚点仅设为{0}的CPU退化fixture应复现S核心计算；生产L固定三个锚点，不能将此fixture开关当成新实验组。
9. partial/all-invalid DDP零目标、B=0和缺锚点拒绝/fallback；部署保存`observation_mode=multi_anchor`、`max_observed_span=2`与几何开关，strict拒绝被当作r1 S加载。

旧输出目录不复用，新增组使用独立arm名。若因端部真实化学/几何污染导致L失败，只报告此假设不成立，不静默替换端部属性或生成新构象。

### 12.8 新阶段、预算与晋级

**M0（待授权）**：在r1公共adapter/训练基础可用后，实现本节三组及定向CPU测试。每组最多2个预训练updates，共6；每组XC fold0最多2个validation-only epochs，共6；必要DDP边界0 optimizer updates。不得借本修订重新跑已通过的r1整套测试或修改其他执行者P0脚本。

**M1（另行授权）**：M0通过后，可运行三组各5k，共15000研究updates；seed42，XC/EPS/EAT×fold0/1，18开发单元，每单元最多30epochs，合计540epochs。沿用§6–7的共同数据、初始化、FP/geo目标、FULL和validation-only协议。M1不要求S已通过原晋级gate，因为它检验不同假设；但必须有已授权P2产生的matched S/R_2D/R_GLT参考才可宣称替换价值，不新增未经计入的参考轨迹。

新增门槛：L_MULTI2分别相对L_NEAR_GEO和L_OFF，XC两fold平均Δ≥0.01且两fold均>0，EPS/EAT平均各不低于−0.01。再按§7.2相同参考保护条件检查相对S、R_2D、R_GLT的替换价值。只优于L_OFF但不优于L_NEAR_GEO时，不能宣布跨度2增量；只优于S时不能排除多锚点/容量路径影响。

M1无稳定增量则停止该分支。通过也仅为开发候选：**不能把L自动替换进r1 P4/P5**；先提交所有差值与资源成本，由Codex另行锁定主候选、匹配几何控制和确认预算，用户授权后执行。已有P4/P5未花额度不构成新增组的授权。

本节新增上限为6 correctness updates、6 smoke epochs、15000研究updates、18开发单元/540epochs；不增加seed、不访问outer-test、不生成坐标。本轮只修订计划，上述实现/测试/训练均未执行。
