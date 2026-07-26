# Uni-Poly-Plus 代码代理约束

## 1. 基本规则

- 默认使用中文。
- `scripts/run.sh` 是正式实验入口和参数真源。
- 用户要求实现时应修改代码并做与风险相称的验证；未运行大规模训练必须明确说明。
- worktree 可能包含用户实验修改，不得回滚、覆盖或删除无关内容。
- 不恢复已删除的 `README.MD`，不删除 cache、checkpoint、日志或历史结果。

## 2. 当前路线

```text
flat4:
  SMILES + FP + GIN Graph + PaiNN Geom

scage_parallel:
  SMILES + sparse MIPS-PyG PBC Graph + FP
  当前默认研究路线
```

`BASELINE=mips` 与 `BASELINE=scage_mips_ablation` 均已移除，传入时应明确报错。
底层 MIPS encoder、137 维输入、descriptor、构图、cache 和历史 checkpoint 仅作为
源码与实验历史保留，不得重新暴露成训练路线。

## 3. `scage_parallel` 不变量

固定配置：

```text
graph_encoder_type=scage
graph_input=star_linking
geom_input=polygen_periodic
fusion_type=parallel_attention
Stage 1 modalities=graph
Stage 2/3 modalities=smiles graph fp
```

虽然 CLI 名称仍为 `scage`，实际 Graph encoder 必须实例化
`MIPSPeriodicGraphEncoder`。旧 `SCAGEGraphEncoder` 保留源码，但主线不得实例化。

模型固定：

```text
6 layers / 512 dim / 8 heads / FFN 2048
dropout=0.10
max_hops=5
64 Gaussian RBF, 0–12 Å
atom mean readout
```

禁止重新加入：

```text
Graph Token
RU/copy index embedding
boundary/seam embedding
image-shift embedding
MIPS 137-d input或descriptor
dense attention
Euclidean cutoff建边
dual branch或branch fusion
angle/torsion作为encoder输入
```

## 4. 周期构图

保留 `src/dataset/geom_data.py` 中 PolyGen-inspired translation-only optimizer：

```text
m_geom=1/2/3/4/6
R=I
T=(0,0,L_base)
pbc_geometry_version=14
```

只接受严格质量门通过的 PBC geometry。失败样本只使用二维周期拓扑，不能把
finite-trimer、普通 3D 或拓扑 hop 当作 Å 距离。

MIPS supercell 固定：

```text
max_hops=5
d_thres=6
boundary distance >11
repeat factor=1/2/4/8/16
max_repeat_rounds=5
max_model_atoms=384
```

PBC 成功时只按 `r(i,q)=r_i+qT` 复制 primitive 坐标，并令
`L_model=kL_base`；禁止重新优化构象。周期末端边只保存在带 `+1/-1` shift 的邻接
表中。

每个 query 必须在 `(atom_id,image_shift)` 空间执行最多 5-hop BFS，并硬检查
boundary condition、alias-free、6-node path 上限及坐标映射。失败必须设置
`graph_available=False`，而不是使用不合格短 cell。

## 5. Cache 与 collate

新主线 schema：

```text
scage_data_schema=scage-mips-pbc-lga-v1
periodic_lga_schema_version=1
pbc_geometry_version=14
```

修改 LGA 字段、atom input、PBC mapping 或构图条件时必须：

1. 升级描述性 feature schema；
2. 更新 dataset load validation；
3. 更新 partial-cache matching；
4. 提醒完整 `--rebuild_feature_cache`；
5. 不迁移旧 SCAGE cache。

collate 必须：

- 给 `lga_edge_index` 和 `lga_path_index>=0` 加 node offset；
- 保留 path padding `-1`；
- 同一 conformer 同时选择坐标和 edge distance；
- 不为新 SCAGE 路线生成 dense `scage_spd/path` batch。

## 6. Attention 与几何

唯一 logit：

```text
q·k/sqrt(d_head) + SPD bias + path-node bias
+ geometry_valid * RBF(PBC Euclidean distance)
```

- softmax 按 target/query 的全部入边归一化；
- SPD 和 distance projection 零初始化；
- path bias 从初始 node embedding 计算一次，六层复用；
- `image_shift` 只用于离线距离，不进入模型；
- geometry-invalid 的 distance bias 必须精确为零；
- graph-unavailable 的 node/graph 输出必须为零。

masked atom 只替换化学字段，保留 backbone role。

## 7. 训练阶段

SCAGE Stage 1 只允许：

```text
masked atom: 1.0
LGA SPD: 0.5
PBC edge distance: 0.75
```

保留动态 loss 归一化。distance target edge 在对应 pretext forward 中必须屏蔽自身
distance bias。不得恢复 ECFP、PerioGT、same/nearest dual distance、shift class、
angle、torsion 或 screw loss，除非用户明确开启新的消融路线。

Stage 2：

- 使用 `smiles graph fp` parallel attention；
- 不使用动态权重，不继续 Stage 1 loss；
- intrinsic mask 为 SMILES=True、FP=True、Graph=`graph_available`；
- Graph-SMILES、Graph-FP 和 fusion masked atom 只统计有效 Graph；
- DDP 对变长有效样本执行 count gather + padding gather。

Stage 3：

- 使用各路线共用的 5-fold：每折 held-out partition 同时作为 validation 和 test；
- 不保留 `VALIDATION_FRACTION` 或 inner-validation 分支，三条路线必须共用完全相同的划分；
- 标签统计只能来自训练折；
- 保持现有标签变换、loss、学习率和分阶段解冻策略；
- 单任务与 8-task 运行必须由稳定 task/fold seed 保持一致。

## 8. Checkpoint

`scage_parallel`：

```text
schema=scage-mips-pbc-pyg-v1
Stage 1 stage=scage_m4p
Stage 2 stage=alignment
```

Stage 1 到 Stage 2 只严格迁移 `encoders.graph.encoder.*`。Stage 3 只接受新
alignment checkpoint。旧 SCAGE、独立 MIPS 或错阶段 checkpoint 必须明确拒绝，
不得部分加载旧 Graph 参数。

历史 `mips-model-v2` 与 `mips-parallel-ablation-v1` checkpoint 保留但当前没有
运行入口，也不得加载到 `scage_parallel`。

## 9. 验证要求

相关修改至少运行：

```text
python -m py_compile
bash -n scripts/run.sh scripts/run_pretrain.sh scripts/run_train.sh
scripts/pretrain.py --help
scripts/train.py --help
```

构图/encoder 修改还应验证：

- boundary `>11`、1/2/4/8/16 扩展和 384 atoms 上限；
- self-edge、方向、SPD、path padding、alias-free；
- PBC edge distance 等于显式坐标计算；
- 每个 query/head attention 和为 1；
- geometry-invalid topology forward；
- graph mean readout；
- Stage 1 三项 loss forward/backward；
- Stage 2/3 mixed availability；
- 新旧 checkpoint 拒绝规则。

500 条正式预检使用 `scripts/preflight_mips24h.py --sample-size 500`。
