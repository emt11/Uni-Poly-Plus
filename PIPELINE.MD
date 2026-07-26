# Uni-Poly-Plus 当前流程

本文档只描述当前可运行实现。实验入口和参数真源为 `scripts/run.sh`；旧 SCAGE、
screw、descriptor 和调参历史保留在源码、Git 历史或 `Z-Paper/`，不再混入主流程。

## 1. 任务与路线

下游任务：

```text
eat eea egb egc ei eps nc xc
```

当前支持：

| `BASELINE` | 模态 | Graph 分支 |
|---|---|---|
| `flat4` | SMILES + FP + Graph + Geom | GIN + PaiNN 历史基线 |
| `scage_parallel` | SMILES + Graph + FP | sparse MIPS-PyG + PBC 距离，当前主线 |

`BASELINE=mips` 与 `BASELINE=scage_mips_ablation` 均已移除。底层
`src/modules/mips_graph.py`、MIPS descriptor、构图、cache 和历史 checkpoint
仍保留用于源码审计与历史结果复核，但不再构成可运行路线。

## 2. `scage_parallel` 数据流

```text
P-SMILES
├─ ChemBERTa SMILES encoder ──────────────────────────────┐
├─ ECFP / MixFP encoder ──────────────────────────────────┤
└─ PolyGen-inspired PBC optimizer                         │
   └─ primitive m-RU cell                                 │
      └─ alias-free repeated supercell                    │
         └─ sparse 5-hop LGA + PBC distance               │
            └─ MIPSPeriodicGraphEncoder ──────────────────┤
                                                         │
SMILES / Graph / FP -> availability-aware parallel attention
                    -> regression head
```

该路线不实例化 PaiNN，不使用 MIPS 的 137 维输入、MD200 或 AtomPair3D512，也不
实例化旧 `SCAGEGraphEncoder`。旧 SCAGE 类保留为源码参考。

固定入口配置：

```text
graph_encoder_type = scage
graph_input = star_linking
geom_input = polygen_periodic
fusion_type = parallel_attention
Stage 1 modalities = graph
Stage 2/3 modalities = smiles graph fp
```

`scripts/pretrain.py` 和 `scripts/train.py` 会拒绝与新架构不一致的 SCAGE 参数。

## 3. PBC primitive cell

输入 P-SMILES 必须有两个各自只连接一个邻居的 `*`。现有
PolyGen-inspired optimizer 依次尝试：

```text
m_geom = 1, 2, 3, 4, 6
R = I
T = (0, 0, L_base)
```

它联合优化分数坐标和 `L_base`，约束单胞内与跨边界键长、键角、二面角及非键
碰撞。`pbc_geometry_version=14` 保持不变。只有通过严格质量门的候选才设置
`polygen_periodic_valid=True`。

这一步的角度和二面角只属于构象优化约束，不输入 Graph encoder，也不作为当前
Stage 1 学习目标。

PBC 失败时，新 Graph 路线只保留二维周期拓扑。共享 dataloader 所需的坐标字段会
写成零占位，`periodic_geometry_valid=False`，不得读取 finite-trimer 或普通 3D
fallback 作为距离。

## 4. MIPS periodic supercell

局部图配置固定为：

```text
max_hops = 5
d_thres = max_hops + 1 = 6
required boundary distance > 2*d_thres - 1 = 11
repeat factors = 1, 2, 4, 8, 16
max_repeat_rounds = 5
max_model_atoms = 384
```

PBC 成功时，从 optimizer 输出的 `m_geom`-RU primitive cell 开始倍增；失败时从
一个 RU 的二维 motif 开始倍增。选择首个满足 boundary 条件的 factor `k`。

有效 PBC 坐标不重新执行 ETKDG、MMFF/UFF、Adam 或 LBFGS，而是精确复制：

```text
r(i,q) = r_i + q(0,0,L_base)
L_model = k * L_base
model_cell_ru = k * m_geom
```

最终原子顺序为 RU0、RU1、……。节点化学字段来自一个重复单元模板并逐 RU 复制，
避免周期切口产生 seam-specific 节点输入。`scage_backbone_role` 只区分
backbone/non-backbone，不编码 attachment seam。

末端周期边不作为普通 RDKit 环写入 molecule；它只在周期邻接表中携带 `+1/-1`
image shift。

若 5 轮或 384 atoms 内仍不满足条件：

```text
graph_available = False
```

该样本仍可使用 SMILES 和 FP，Graph 只保留 self-edge 占位且输出严格为零。

## 5. Sparse LGA cache

对每个 query atom，从 `(atom_id, image_shift)` 状态执行最多 5-hop BFS。attention
edge 方向固定为 `source/key -> target/query`，并包含 self-edge。

cache 核心字段：

```text
lga_edge_index             [2,E]
lga_spd                    [E], 0..5
lga_path_index             [E,6], padding=-1
lga_path_mask              [E,6]
lga_image_shift            [E], 仅离线距离与诊断
lga_pbc_distance_confs     [C,E]
lga_geometry_valid         [E]
graph_available            [1]
periodic_geometry_valid    [1]
geometry_period_ru
mips_repeat_factor
model_cell_ru
mips_boundary_distance
mips_condition_valid
mips_alias_free
```

构图硬检查 boundary condition、每个 query 的 image alias、路径长度及索引映射。
collate 对 edge/path 的有效节点索引加 batch offset，`-1` padding 保持不变，并从
同一个随机 conformer 同时选择坐标和 edge distance。该路线不再构建 dense
`scage_spd/scage_path` batch。

版本：

```text
feature schema = scage-mips-pbc-lga-v1
periodic_lga_schema_version = 1
pbc_geometry_version = 14
```

旧 SCAGE cache 不迁移，必须完整重建。

## 6. Sparse Graph encoder

实现：`src/modules/mips_periodic_graph.py`

节点输入：

```text
9 categorical SCAGE fields
mass
van_der_waals_radius
Gasteiger partial_charge
binary backbone role
```

categorical index 0 可学习。没有 padding category、RU index、copy index、
image-shift embedding、boundary/seam embedding、Graph Token 或 descriptor。

固定结构：

```text
6 layers
hidden dim 512
8 heads
FFN dim 2048
dropout / attention dropout / activation dropout = 0.10
pre-norm residual attention
pre-norm residual GELU FFN
readout = atom global mean
```

每条 sparse edge 的 attention logit：

```text
q_i·k_j / sqrt(head_dim)
+ SPD head embedding
+ path-node head bias
+ geometry_valid * 64-RBF(PBC Euclidean distance) projection
```

SPD 和 distance projection 零初始化。path-node bias 从初始节点表示计算一次并在
6 层复用。softmax 按 target/query 的全部入边归一化。

`lga_image_shift` 只离线计算：

```text
distance = ||r_source + nT - r_target||
```

geometry-invalid 时 distance bias 精确为零，但拓扑 forward 正常。graph-unavailable
时 node/graph 表示归零。masked atom 使用独立 mask token替换化学字段，backbone
role 保留。

## 7. Stage 1

`scage_parallel` 使用 graph-only `scage_m4p`，当前只有三个任务：

| 任务 | 先验 |
|---|---:|
| masked atom | 1.0 |
| LGA SPD（0..5） | 0.5 |
| PBC LGA edge distance | 0.75 |

保持动态 loss 标准化。距离使用 scalar SmoothL1Loss；抽为回归 target 的 edge 在
该次 geometry-pretext forward 中会屏蔽自身 distance bias，防止读取答案。

以下旧任务在新主线中停用：

```text
ECFP reconstruction
PerioGT periodic contrast
same-cell / nearest-image dual distance
image-shift classification
angle prediction
torsion prediction
screw prediction
```

geometry-invalid 不产生 distance target；graph-unavailable 不产生任何 Graph target。

## 8. Stage 2

Stage 2 使用 `smiles graph fp` parallel attention，关闭动态任务权重，也不继续
Stage 1 辅助任务。目标保留：

```text
two-view fused InfoNCE
Graph-SMILES InfoNCE
Graph-FP InfoNCE
LOMO consistency
batch pooling-prior KL
fusion-conditioned masked atom
```

intrinsic availability 为：

```text
SMILES=True
FP=True
Graph=data.graph_available
```

随机 modality dropout 与 intrinsic mask 取交集。Graph 对比和融合 masked-atom
只统计可用 Graph。DDP 先 gather 有效数，再 padding gather 表示；全局少于两个
有效样本时对应 InfoNCE 为无 target 的零损失。

Stage 1 checkpoint 只严格迁移 `encoders.graph.encoder.*` 到 Stage 2，不迁移
Stage 1 fusion/head。

## 9. Stage 3

Stage 3 使用项目各路线共用的 5-fold 协议：

- KFold `n_splits=5, shuffle=True, random_state=1`；
- 每一折的 held-out fold 同时作为 validation 和 test；
- 剩余四折全部作为训练集，不再默认切分 inner validation；
- 不提供 inner-validation 切分开关，三条路线使用完全相同的划分；
- 默认 `TARGET_TRANSFORM=recommended`：eps/nc 为 log+standard，其余 standard；
- 默认 Huber/SmoothL1，`HUBER_BETA=0.5`；
- 前 5 epochs 训练 fusion/head，随后分阶段解冻 Graph/SMILES；
- FP 按当前 `FP_UNFREEZE_EPOCH` 设置；
- 支持 unimodal auxiliary、fusion-prior KL、SWA 等显式消融入口。

forward 会把 intrinsic availability 与训练态 modality dropout 合并。Graph 不可用
时，主融合、单模态辅助和 pooling prior 都不会把零 Graph 当成真实模态。

## 10. Checkpoint

新 SCAGE schema：

```text
schema = scage-mips-pbc-pyg-v1
Stage 1 meta.stage = scage_m4p
Stage 2 meta.stage = alignment
```

Stage 2 只接受新 Stage 1；Stage 3 只接受新 alignment。旧
`scage-polygen-periodic-v4`、`mips-model-v2` 和
`mips-parallel-ablation-v1` 均不能加载到 `scage_parallel`。

## 11. 运行

首次运行必须重建 cache：

```bash
BASELINE=scage_parallel \
PRETRAIN_DATASET=PI1M_50k \
REBUILD_FEATURE_CACHE=1 \
bash scripts/run.sh
```

最小流程：

```bash
BASELINE=scage_parallel PRETRAIN_DATASET=PI1M_50k \
REBUILD_FEATURE_CACHE=1 GRAPH_GEOM_EPOCHS=1 ALIGN_EPOCHS=1 \
TRAIN_EPOCHS=1 TASKS=eat FOLD_IDS=0 EXPERIMENT_TAG=mips_pbc_smoke \
bash scripts/run.sh
```

500 条预检：

```bash
/root/anaconda3/envs/Uni-Poly/bin/python scripts/preflight_mips24h.py \
  --sample-size 500
```

预检报告 repeat factor、model RU、atom/edge count、geometry-valid、
graph-unavailable、alias failure 和单样本耗时分布。
