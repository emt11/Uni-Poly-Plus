# MTS（MIPS-Trimer-SCAGE）TODO

## TODO

- [ ] 完成当前 `G → V → MT → FINAL` 活动消融 DAG，未完成前不将后文的
  新优化池叠加到正式模型。
- [x] G 阶段真实坐标与 shuffled 对照的条件判断已完成；真实坐标未通过门槛，
  本轮跳过 Angle-v2，不重训联合预训练。
- [ ] 当前 DAG 结束后，按“新优化池”优先完成 canonical 层次读出、
  XC 抗收缩预测头和 O8 化学键关系偏置。

## 当前消融各阶段的实际改动

以下阶段不是连续向同一个模型无条件叠加功能，而是一个带晋级门的实验 DAG。
每一阶段只改变所列主变量；数据划分、PI1M_v2 联合预训练 checkpoint、O8
主干、统一微调预算和 seed 规则保持一致。下游阶段的配置会物化上一阶段的胜者，
不会在运行时凭默认值猜测父模型。

### 固定下游基线：legacy_mts_huber_v1

F 阶段及其新版微调组合已判定为退化实验，退出活动消融 DAG。后续所有候选都
固定继承已验证的旧 MTS 基线：

```text
Huber(beta=0.5)
Graph 全部模块从 epoch 0 训练
batch=32，epochs=100，patience=10
Graph LR=1e-5，projection/head LR=1e-4
weight decay=0.02，前5 epochs warmup，单次全程 cosine
gradient clip=1.0，SWA关闭，TARGET_TRANSFORM=recommended
```

F 结果只用于审计，不参与父模型、promotion 或正式汇总：

```text
F0_mse   macro R² = 0.835098
F0_huber macro R² = 0.836137
旧 MTS   macro R² = 0.842341
```

拒绝原因：loss筛选以及同期的分层解冻、阶段scheduler、SWA和新版参数组均未
超过旧 MTS 基线。`results/mts_sota_v3/F0_mse/` 和 `F0_huber/` 保留为只读
失败证据，但活动汇总会排除它们。

### G：Trimer 几何是否有效，以及连续距离是否有增益

G 阶段直接继承 `legacy_mts_huber_v1`，只改变 Trimer 几何处理：

| 配置 | 相对共同父模型的唯一主要改动 | 用途 |
|---|---|---|
| `G0_current_mcl` | 保留当前 20%/50% 距离分位 hard-mask MCL | 几何父基线 |
| `G1_mcl_disabled` | 关闭 Trimer-MCL residual；仍保留直接 Star-edge 的对称距离 RBF bias | 测量 MCL 本身的贡献，不等同于完全无3D |
| `G2_coordinate_shuffled` | 打乱 Trimer 原子—坐标对应；同时关闭无法一致置乱的缓存 Star-distance bias | 当前 MCL 的坐标负对照 |
| `G3_mcl_rbf` | 在 G0 的 hard visibility mask 内新增 64 个、0–8 Å Gaussian RBF 的 per-head 连续距离 bias，projection 零初始化 | 判断 mask 内近远差异是否提供额外信息 |
| `G4_mcl_rbf_shuffled` | 对 G3 执行坐标负对照，并关闭缓存 Star-distance bias | 排除增益来自容量或训练扰动而非真实3D |

G1 与 G2 的含义不同：G1 是“没有 MCL，但仍有聚合连接键长”；G2 是“保留
MCL 计算图但破坏正确坐标语义”。G3 只有同时满足以下条件才晋级，否则回退
G0：

```text
相对G0宏平均ΔR² >= +0.003
相对G4真实坐标增益 >= +0.002
至少5/8任务不下降
最差单任务ΔR² >= -0.010
```

### V：SMILES 与 attachment-aware CountFP 是否提供互补信息

V 阶段继承 G geometry 胜者和固定 Huber 基线，Graph 主干不变，只增加低容量残差模态：

| 配置 | 新增输入 | 融合方式 |
|---|---|---|
| `V1_smiles` | canonical P-SMILES 与 attachment 反转 view；两 view 表示平均；PubChem10M encoder + 最后四层 LoRA | `MTS graph + tanh(gate) * SMILES adapter` |
| `V2_countfp` | 2570维 attachment-aware CountFP：Morgan r2/r3 count、attachment-rooted count 和10个全局量 | `MTS graph + tanh(gate) * FP adapter` |
| `V3_smiles_countfp` | 同时加入上述两者 | 两个独立零初始化 gate 的残差和 |

这里不恢复 parallel attention。Graph 永远是锚点且不做 modality dropout；SMILES
和 FP 的 dropout 分别为 `0.10/0.15`。零初始化 gate 保证初始输出逐元素等于
Graph 父模型。每个新增模态都要比较 `real / batch-shuffled / constant-zero`，
只有真实输入优于负对照且满足以下统一门槛才晋级：

```text
相对G胜者宏平均ΔR² >= +0.003
至少5/8任务不下降
最差单任务ΔR² >= -0.010
所有真实模态的control margin > 0
```

若没有 V 候选通过，后续直接继承 G 的 Graph-only 胜者。

### MT：防泄漏多任务 PCGrad

MT 阶段继承 G/V 的胜者和固定 Huber 基线，固定相同输入模态和融合器，只比较训练组织方式：

- `MT0_single_task`：每个属性独立训练，是同一父架构的控制项；
- `MT1_multitask_pcgrad`：共享 encoder，使用八个 task-specific regression heads、
  task-balanced sampler 和 PCGrad 处理共享参数的梯度冲突。

MT1 对目标 `(task, fold)` 会从所有辅助任务训练集中排除该 outer fold 的相同
sample key，避免目标测试 polymer 经其他属性泄漏。各任务使用自己的训练折 scaler，
缺失标签由 mask 处理。MT1 只有相对 MT0 满足下列条件才晋级：

```text
宏平均ΔR² >= +0.003
至少5/8任务不下降
最差单任务ΔR² >= -0.010
```

### FINAL：严格 nested5 与候选职责分离

FINAL 不再继续发明新模块，而是把前面选出的能力拆成三个候选，在严格
`outer 5-fold + inner validation` 下重新评估：

| 配置 | 固定内容 | 回答的问题 |
|---|---|---|
| `FINAL_geometry_nested5` | 固定 Huber 基线 + G geometry胜者；Graph-only、single-task | 单独依赖MTS几何时的可信泛化能力 |
| `FINAL_modality_nested5` | 再加入V胜出的SMILES/FP组合；single-task | 互补模态在严格划分下是否仍有效 |
| `FINAL_multitask_nested5` | 在模态胜者上采用MT胜出的single-task或PCGrad方式 | 多任务收益能否在严格划分下保留 |

inner validation 只用于 checkpoint、early stopping、SWA 和后续集成权重选择；
outer test 只在模型确定后评估一次。FINAL 的 `promote` 记录 seed-42 三个候选中
宏平均 R² 最高者，但最终报告仍保留三者，不用单个 outer-test 结果反向调参。

### 可选 Angle-v2：不是默认矩阵阶段

Angle-v2 只有在 G 阶段证明真实坐标明显优于 shuffled 后才执行。它会从冻结
Trimer 坐标生成连续 `cos(theta)` sidecar，以 SmoothL1 angle regression 替换
当前20类 angle任务，并重新进行完整 PI1M_v2 联合预训练；不会重新运行
ETKDG/MMFF。角度权重只从 `0.10/0.25/0.50` 中统一选择。若重预训练带来的
下游宏平均增益不足 `0.003`，继续使用现有联合预训练 checkpoint。

### 跨阶段始终不变的控制条件

- Stage 1预训练数据固定为完整 `PI1M_v2`，下游消融复用同一 seed-42 checkpoint；
- O8固定为137维输入、6层、512维、8 heads、严格0/1/2-hop、SPD和
  single-path-node bias；
- Trimer与O8继续通过canonical atom严格映射，无效几何精确回退O8；
- 所有候选先执行seed 42完整8任务×5折，不针对单任务调参；
- F/G/V/MT使用历史兼容`shared_validation_test_fold`，FINAL使用nested5；
- 原始fold shard保留完整精度，正式汇总使用五折样本标准差和三位小数
  `mean ± std`；
- GPU仅使用0/1/2，三张卡动态领取独立 `(task, fold)`，GPU 3不可见。

## 后续执行

正式消融不再使用 folds 0/1/2。G、V、MT 和 FINAL 的每一个配置均需
执行 seed 42 的完整 `8 tasks × 5 folds`；三折结果只能作为开发 smoke，不能
用于晋级或正式结论。

### 0. 统一环境

```bash
tmux attach -t Uni-Poly
cd /root/workspace/DeepLearning/Uni-Poly-Plus-master

export PYTHON_BIN=/root/anaconda3/envs/Uni-Poly/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2

mkdir -p logs/mts_sota_v3
set -o pipefail

$PYTHON_BIN scripts/mts.py doctor \
  2>&1 | tee logs/mts_sota_v3/doctor.log
```

### 1. G：几何与连续距离完整五折（已完成，命令仅作复现记录）

已完成 `G0 current MCL`、`G1 disabled`、`G2 coordinate shuffled`、
`G3 MCL-RBF` 和 `G4 MCL-RBF shuffled`。下面命令仅用于复核或在明确改变
代码/配置后重新生成独立结果，不应作为当前下一步重复执行：

```bash
$PYTHON_BIN scripts/run_mts_sota_campaign.py validate --phase G

$PYTHON_BIN scripts/run_mts_sota_campaign.py screen \
  --phase G --seed 42 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/G_full5_seed42.log

$PYTHON_BIN scripts/run_mts_sota_campaign.py promote \
  --phase G --seed 42 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/G_promote_full5.log
```

原计划要求真实坐标相对 shuffled 的宏平均增益至少为 `0.002` 才执行后面的
Angle-v2 全量重预训练。当前 G 已完成，但 G0/G3 均未满足该条件，因此 Angle-v2
本轮明确跳过，继续使用现有联合预训练 checkpoint。

### G 阶段实际结果（seed 42，完整 8 任务 × 5 folds）

G0–G4 均已生成 40 个有效 fold shard，结果来自
`results/mts_sota_v3/screening_summary.csv`：

本轮 G screen 已结束，`Uni-Poly` tmux 当前回到 shell，未有 G 训练进程继续运行。

| 配置 | 宏平均 R² | XC R² | 相对固定基线 0.842341 | 生产判断 |
|---|---:|---:|---:|---|
| G0 current MCL | 0.841733 | 0.425739 | -0.000609 | 保留为真实几何父模型 |
| G1 MCL disabled | 0.840487 | 0.418429 | -0.001854 | 不晋级 |
| G2 coordinate shuffled | 0.851304 | 0.491638 | +0.008963 | 仅负对照，禁止生产 |
| G3 MCL-RBF | 0.840177 | 0.399235 | -0.002164 | 不晋级 |
| G4 MCL-RBF shuffled | 0.851188 | 0.488521 | +0.008846 | 仅负对照，禁止生产 |

G 阶段的真实几何结论为：

- G0 比关闭 MCL 的 G1 高 `0.001246`，说明 MCL 的容量/结构可能有价值；
- G3 比 G0 低 `0.001556`，连续 MCL-RBF 没有带来增益；
- G2/G4 都明显高于真实几何配置，说明当前 Trimer 坐标语义没有被可靠利用，
  或坐标置乱负对照改变了正则化/容量，而不是证明置乱坐标更有物理意义；
- G0 相对 G4 的宏平均差为 `-0.009455`，没有满足真实坐标优于 shuffled 至少
  `0.002` 的条件，因此本轮不执行 Angle-v2，也不重训联合预训练；
- XC 的生产父结果仍以 G0 为准，G2/G4 只能作为诊断负对照，不能写入
  `promotion_state` 或作为 V/MT/FINAL 的生产父模型。

因此，G 阶段完成后允许继续 V 阶段，但 V 的父模型固定为 G0；在修复坐标置乱
负对照实现并重新证明真实几何有效之前，不再叠加新的 3D 距离偏置。

### 2. V：SMILES 与 CountFP 完整五折

执行 `V1 MTS+SMILES`、`V2 MTS+CountFP` 和
`V3 MTS+SMILES+CountFP`。每个 fold 同时产生 real、batch-shuffled 和
constant-zero 模态负对照：

```bash
$PYTHON_BIN scripts/run_mts_sota_campaign.py validate --phase V

$PYTHON_BIN scripts/run_mts_sota_campaign.py screen \
  --phase V --seed 42 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/V_full5_seed42.log

$PYTHON_BIN scripts/run_mts_sota_campaign.py promote \
  --phase V --seed 42 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/V_promote_full5.log
```

### 3. MT：单任务与 PCGrad 完整五折

```bash
$PYTHON_BIN scripts/run_mts_sota_campaign.py validate --phase MT

$PYTHON_BIN scripts/run_mts_sota_campaign.py screen \
  --phase MT --seed 42 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/MT_full5_seed42.log

$PYTHON_BIN scripts/run_mts_sota_campaign.py promote \
  --phase MT --seed 42 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/MT_promote_full5.log
```

### 4. FINAL：严格 nested5 完整五折

FINAL 会分别执行几何胜者、模态胜者和多任务胜者三个候选。outer test 不参与
checkpoint、SWA、模型权重或集成权重选择：

```bash
$PYTHON_BIN scripts/run_mts_sota_campaign.py validate --phase FINAL

$PYTHON_BIN scripts/run_mts_sota_campaign.py screen \
  --phase FINAL --seed 42 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/FINAL_nested5_seed42.log

$PYTHON_BIN scripts/run_mts_sota_campaign.py promote \
  --phase FINAL --seed 42 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/FINAL_promote_seed42.log
```

### 5. 最终候选的 seeds 43/44

只有 seed-42 promotion gate 通过后才补 seeds 43/44。campaign 始终继续加载
seed-42 的联合预训练 checkpoint；`--seed 43/44` 只改变下游 fold seed：

```bash
$PYTHON_BIN scripts/run_mts_sota_campaign.py screen \
  --phase FINAL --seed 43 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/FINAL_nested5_seed43.log

$PYTHON_BIN scripts/run_mts_sota_campaign.py screen \
  --phase FINAL --seed 44 --folds 0 1 2 3 4 \
  2>&1 | tee logs/mts_sota_v3/FINAL_nested5_seed44.log
```

每个 `(task, fold)` 必须先平均 seeds 42/43/44 的原始空间预测，再计算 R²、
MAE 和 RMSE，不允许直接平均三个 seed 的指标。

### 6. 最多三个 nested 候选集成

```bash
$PYTHON_BIN scripts/ensemble_mts_candidates.py \
  --candidate results/mts_sota_v3/FINAL_geometry_nested5 \
  --candidate results/mts_sota_v3/FINAL_modality_nested5 \
  --candidate results/mts_sota_v3/FINAL_multitask_nested5 \
  --seeds 42 43 44 \
  --output-root results/mts_sota_v3/final_ensemble
```

误差相关性不低于 `0.95` 的候选必须被拒绝；权重只可来自 inner-validation
RMSE，不能读取 outer-test 指标。

### 7. 条件执行 Angle-v2 与预训练 benchmark

仅当 G 阶段证明真实 Trimer 坐标有效时执行：

```bash
$PYTHON_BIN scripts/mts.py build-angle-v2

CUDA_VISIBLE_DEVICES=0,1,2 \
$PYTHON_BIN scripts/mts.py benchmark-pretrain \
  --batches 500 --loader-workers 0 \
  2>&1 | tee logs/mts_sota_v3/pretrain_benchmark.log
```

benchmark 通过 2 倍吞吐、显存、rank wait 和 resume 门后，才能用选中的 batch
配置执行完整 PI1M_v2 Angle-v2 重预训练。候选权重只允许
`0.10/0.25/0.50`，最终 export 使用预训练 validation composite 最优里程碑，
而不是最后一步。

### 8. 尚待完成的验收

- [x] F 阶段已移除；F0 结果仅作为退化实验记录，不参与活动汇总。
- [x] G 阶段五个配置均完成 seed42、8任务×5折及坐标负对照；真实几何未通过
  相对 shuffled 的晋级门，生产父模型固定为 G0，G2/G4 仅保留为负对照。
- [ ] V 阶段三个配置均完成 seed42、8任务×5折及模态负对照。
- [ ] MT 阶段两个配置均完成 seed42、8任务×5折。
- [ ] FINAL 三个候选均完成 seed42 nested5；通过门后补 seeds 43/44。
- [ ] 对比 `results/best_result.csv`，记录每任务 gap 与宏平均 R²。
- [ ] 最终报告满足三位小数 `mean ± sample std`，且完整精度 shard 可复核。

## 新优化池：G/V/MT/FINAL 之外的结构与 XC 优化

本节是 G 阶段之后的候选优化池，不改变已完成的 G 结果，也不自动加入当前
`G → V → MT → FINAL` DAG。它排除了已有计划中的 MCL 连续 RBF、SMILES、CountFP、
PCGrad、Angle-v2、多 seed 和 nested5，仅记录尚未实现的优化方向。

### 1. XC 当前问题诊断

`smi_xc.csv` 只有 432 条样本，标签范围约为 `0.13–98.81`，分布右偏。G0 的
XC 为 `0.426 ± 0.063`，best result 为 `0.579`。OOF 预测与真实标签的线性斜率
约为 `0.47`，表现为：低 XC 被高估，高 XC 被低估，预测范围明显向均值收缩。

因此，XC 的首要问题不是继续增大 O8 的层数或隐藏维度，而是：

```text
canonical copy 重复计权
+ 全局 mean pooling 丢失主链/侧链层次
+ 小样本下完整 Graph wrapper 更新过强
+ 预测头和 weight decay 抑制极端值
+ Trimer 单构象几何存在噪声
```

G2/G4 坐标负对照高于 G0/G3，进一步说明当前真实 Trimer 几何尚未被可靠利用。
在修复负对照语义前，不应继续堆叠更多距离、角度或等变模块。

### 2. N1：canonical 层次化读出与 Jumping Knowledge（最高优先级）

将当前“所有 O8 copy 直接 global mean”改为：

```text
O8 六层输出
→ 按 canonical_atom_id 聚合全部 MIPS copies
→ backbone pool、side-chain pool、global pool
→ gated hierarchical readout
```

建议：

```text
copy aggregation       scatter_mean
pool hidden            128
pool heads             4
pool transformer       1层
backbone/side-chain    对称共享参数
```

六层 O8 输出增加 Jumping Knowledge：

```text
x_jk = softmax(alpha_1...alpha_6) · [x_1...x_6]
```

该模块在 canonical 层完成，不改变 O8 的 0/1/2-hop 证明，也不改变缓存。
它优先解决重复 RU copy 对样本权重的影响，并恢复主链、侧链、环结构的层次信息。

### 3. N2：O8 真实化学键和路径关系偏置

在现有 SPD、single-path-node 和 Star bias 之外，增加只作用于已有 LGA edge 的
per-head relation bias：

```text
self、single、double、triple、aromatic、virtual-star
```

对于 2-hop edge，再使用两条真实键的无序组合编码。推荐关系 embedding 大小为
`6 × 8 heads`，不扩大注意力范围，不加入方向或 RU offset。这样可以区分“相同
SPD 但化学路径不同”的原子对，尤其有利于主链刚性、芳环和支链表示。

同时增加反转不变的节点角色：

```text
到 backbone 的图距离：0/1/2/≥3
到最近 attachment boundary 的距离桶
```

### 4. N3：Trimer-MCL 角色、消息和可信度

当前三个 Trimer copy 的初始 token 主要来自相同 canonical 表示。新增：

```text
central-RU role
adjacent-RU role
same-RU / cross-RU-real-bond / cross-RU-nonbonded relation
```

左右相邻 RU 使用同一个 embedding，保持 attachment reversal 不变；Star 虚拟边
仍不进入 Trimer 真实键关系。

在已有 hard visibility mask 内，增加 zero-init 的距离条件 value message：

```text
message_ij = V_j * (1 + W_geo(phi_RBF(d_ij)))
RBF=32，范围0–8 Å，低秩rank=16
```

它与已有 MCL-RBF logit bias 不同：不改变可见性，而是让距离调节传递内容。

当前全局 geometry gate 改为样本级 confidence gate。输入仅使用不涉及绝对跨分子
能量比较的诊断量：Trimer 原子数、重试标志、每原子有限能量、star asymmetry、
接触密度和拥挤度。无效几何仍精确回退 O8。

### 5. N4：MD200 互补残差

当前 MD200 adapter 后只有一个全局标量 gate。改为零初始化 FiLM：

```text
MD200 → LayerNorm → 200→64→128
      → graph-conditioned scale/shift (512+512)
      → g'=(1+scale)⊙g+shift
```

增加小权重的 batch-level graph/MD cross-covariance penalty，避免 MD200 重复编码
O8 已经表达的结构。绝对 MMFF energy 不作为跨分子物理标签。

### 6. N5：低维结晶倾向 sidecar 与图形学特征

在不恢复 SMILES/CountFP 的前提下，增加 16–32 维确定性结构 sidecar：

```text
backbone fraction、rotatable-bond density、branch density
ring/aromatic fraction、sp2/sp3、HBD/HBA、TPSA/heavy-atom
canonical orbit entropy、attachment distance
end-to-end/contour ratio、Rg、asphericity、principal-moment ratio
backbone planarity、inter-RU torsion sin/cos、contact density
```

使用 `32→64→512` zero-gated adapter，不把它作为第四个高维模态。

图形学候选为 Trimer 非键 contact graph 的谱/热核摘要：

```text
normalized Laplacian 前16个特征值
heat trace(t=0.1,0.3,1,3)
clustering coefficient、multi-scale component count
```

该摘要对旋转、平移和原子置换不变，用于表达局部空间紧凑性，而不是声称获得
无限聚合物体相堆积。

### 7. N6：小样本微调与 XC 抗收缩预测头

O8 继续保持 `6层/512维/8 heads/FFN 2048`，不要直接扩大模型。微调建议改为
layer-wise learning rate，而不是重新引入已被 F 阶段否定的 Phase A/B/C：

```text
O8 layers 1–3       1e-6
O8 layer 4/5/6       2e-6 / 4e-6 / 8e-6
Star-RBF/MCL         1e-5
MD/readout adapter   3e-5
regression head      1e-4
```

weight decay 只作用于矩阵权重，建议 `0.005–0.01`；bias、LayerNorm 和所有 gate
使用 `0`。head dropout 建议 `0.10–0.15`。另一条低风险路径是冻结 O8 主权重，
仅对最后三层 O8 和两层 MCL 使用 LoRA：`rank=8, alpha=16, dropout=0.05`。

回归头改为线性跳连：

```text
y = Linear(g) + MLP(LayerNorm(g))
```

并加入跨任务通用的 ranking auxiliary loss，推荐权重 `0.05–0.10`，只对标签
差异超过 `0.5σ` 的样本对计算。它用于恢复 XC 的排序和预测范围，不改变主损失定义。

对于 XC，可增加有界输出头作为独立候选：预测 `[0,100]` 范围内的均值和浓度，
使用 Beta NLL 加小权重 Huber。它不能直接恢复旧版 clipped-logit，也不能替代全任务
共享主线；若使用，必须在训练折内部拟合并验证。

### 8. N7：只在训练折内进行预测校准或核方法头

由于 XC OOF 斜率约为 `0.47`，可在 outer-train 内部用 OOF 预测拟合 ridge affine
calibration：

```text
y_calibrated = a * y_pred + b
```

严禁使用共享 validation/test fold 拟合 `a,b`。

另一条适合 432 条 XC 数据的预测头是冻结 MTS embedding 后使用：

```text
RBF Kernel Ridge / Gaussian Process / CatBoost
```

输入使用 canonical hierarchical graph embedding、MD200 bottleneck 和低维物理
sidecar，再由 inner validation 决定是否与神经 head 融合。

### 9. N8：新的预训练任务

在 masked atom 和现有 angle 之外，可增加 canonical–geometry correspondence：

```text
正样本：正确 canonical atom ↔ Trimer 坐标映射
负样本：同一 Trimer 内置换 atom-coordinate mapping
```

使用 binary matching 或 InfoNCE，建议权重 `0.05–0.10`。该任务直接检验 MCL
是否学会正确的化学—坐标对应关系，特别适合解释 G2/G4 高于 G0/G3 的异常现象。

可以增加跨 RU 连接附近的无方向 `cos(torsion)` 任务，但只有在目标关系对应输入
被屏蔽时才有防泄漏意义。

### 10. 新优化池的推荐优先级

```text
P0  canonical hierarchical readout + Jumping Knowledge
P0  residual regression head + ranking loss + inner-only calibration
P1  O8 real-bond/path-bond relation bias
P1  no-decay parameter groups + layer-wise LR 或 LoRA
P1  MCL central/adjacent role + sample geometry confidence
P2  MD200 FiLM + low-dimensional crystallinity sidecar
P2  distance-conditioned value message
P3  correspondence pretraining + contact-graph spectrum
```

所有 P0–P3 模块必须跨任务共享；只有有界 XC head、XC 校准和 kernel head 属于
XC 专用候选，不能修改其它任务的模型结构或训练数据。

### 11. 新优化池验收原则

- G0/G1/G2/G3/G4 的历史结果只读保存，不因新模块覆盖；
- 新模型初始关闭所有新增 residual 时，输出必须逐元素等于 G0；
- canonical copy、attachment reversal、atom permutation 不改变 graph embedding；
- 几何无效、坐标置乱和 MD 无效样本必须精确回退对应父路径；
- 任何新增几何模块必须同时报告真实坐标、随机置乱和常量/关闭对照；
- XC 改进必须在完整 5-fold、原始标签空间报告 `R²/MAE/RMSE mean ± sample std`；
- 不能用 shared validation/test fold 选择校准参数、模型结构或集成权重；
- 若新增模块使宏平均提升但 XC 继续收缩，优先保留全任务模型并独立评估 XC head，
  不得为 XC 单任务修改共享 encoder。

## 其他

- [x] GPU 预训练热路径的软件优化已实现；实际吞吐验收仍待三卡 benchmark。














