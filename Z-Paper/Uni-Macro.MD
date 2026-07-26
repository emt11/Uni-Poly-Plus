# Uni-Macro 代码仓库复现导向分析

分析对象：`https://github.com/PchGolden/Uni-Macro/tree/main`

仓库定位：面向聚合物性质预测的 Uni-Macro 代码发布版。该仓库提供从 CSV 到 PKL/LMDB 的聚合物数据预处理、基于 Uni-Mol/Uni-Mol2 风格的 token-pair Transformer 模型、3D 构象输入、预训练任务和下游 fine-tune 流程。

本文档按 `Z-Paper/` 中已有论文分析文档的风格整理，重点是阅读源码后的实现逻辑，而不是只根据 README 推断。

## 1. 仓库整体目标

Uni-Macro 的目标是处理聚合物性质预测中的以下输入情况：

```text
1. 单个 SMILES 表示的均聚物或小分子式 repeat unit。
2. 多个 SMILES segment 表示的共聚物、混合物或多组分体系。
3. 每个 segment 可带局部数值特征。
4. 每条样本可带全局数值特征。
5. 训练目标可以是回归或分类。
```

整体流程是：

```text
CSV
-> RDKit 解析每个 SMILES segment
-> 生成 heavy-atom graph features
-> 生成多构象 3D coordinates
-> 合并多个 segment 为一个样本级 token sequence
-> 保存 PKL 或 LMDB
-> MultiMolModel 预训练或 fine-tune
```

它不是一个简单的 `SMILES encoder + 3D encoder + fusion` 项目，而是把以下信息统一进一个 token-pair Transformer：

```text
atom token
atom discrete features
bond / shortest path pair features
3D distance Gaussian pair bias
segment-level numeric token
global numeric token
CLS graph token
```

## 2. 仓库结构

核心文件：

```text
src/preprocess/preprocessing_polymer.py
src/preprocess/molecular_features.py
src/dataloader/dataloader_polymer.py
src/models/multi_mol_model.py
src/models/encoder.py
src/models/attention.py
src/models/utils.py
src/main.py
```

文件职责：

```text
preprocessing_polymer.py
  CSV -> sample dict -> PKL/LMDB。
  负责 RDKit 构象生成、多 SMILES segment 合并、fold 分配。

molecular_features.py
  从 RDKit Mol 构造 atom features、bond features、shortest path、pair type。

dataloader_polymer.py
  读取 PKL/LMDB，做 train/val split，随机选择 conformer，预训练时加 mask/noise/drop。

multi_mol_model.py
  主模型 MultiMolModel。实现 token embedding、pair embedding、3D SE(3)-invariant bias、Transformer encoder、预训练 heads、下游 regression head。

encoder.py / attention.py / utils.py
  Uni-Mol/AlphaFold 风格 token-pair Transformer 组件，包括 pair pathway、outer product、triangle multiplication、movement prediction head。

main.py
  训练入口。支持 pretrain / finetune、DDP、AMP、checkpoint、k-fold。
```

## 3. 数据输入格式

README 和源码都假设 CSV 至少包含：

```text
SMILES0
label column，例如 Tg / Ei / Eea
```

可选多 segment：

```text
SMILES0, SMILES1, ..., SMILES{k}
```

可选局部 segment 特征：

```text
seg0_feat0, seg0_feat1
seg1_feat0, seg1_feat1
...
```

可选全局特征：

```text
glob_feat0, glob_feat1
```

可选 fold 列：

```text
fold
```

如果 CSV 有 `fold`，源码直接使用该列作为 fold id。如果没有，预处理阶段按 seed 随机 shuffle 后轮流分配 fold。

## 4. CSV 到 PKL 的预处理逻辑

入口：`src/preprocess/preprocessing_polymer.py`

推荐命令：

```bash
python src/preprocess/preprocessing_polymer.py \
  --task finetune \
  --csv datasets/your_dataset.csv \
  --labels Tg \
  --kfold 5 \
  --seed 42 \
  --outroot data/processed \
  --dataset-name your_dataset_name \
  --export-splits
```

核心函数 `_process_row()` 做以下事情：

```text
1. 读取当前行所有非空 SMILES segment。
2. 对每个 SMILES 调用 _embed_and_strip_h() 生成 RDKit Mol 和多构象坐标。
3. 调用 build_initial_graph() 生成 heavy-atom graph。
4. 调用 build_graph_features() 生成 atom_feat / edge_feat / shortest_path / degree。
5. 把多个 segment 的 atom token 和坐标拼成一个样本。
6. segment 内保留自己的 edge/shortest path；segment 间 shortest path 设为 unreachable。
7. 构造全局 token、segment token 和 base attention mask。
8. 保存 label、split、polymer_id。
```

一个样本最终包含：

```text
src_token       atomic number token
src_pos         [num_confs, total_atoms, 3]
atom_feat       discrete atom features
atom_mask       atom valid mask
edge_feat       dense [N, N, edge_dim]
shortest_path   dense [N, N]
degree          atom degree
pair_type       atom pair type
segment_id      each atom belongs to which SMILES segment
base_mask       special token attention mask
glob_feat       global numeric features
glob_mask       global feature valid mask
seg_feat        segment numeric features
seg_feat_mask   segment feature valid mask
label           target values
split           fold id string
```

## 5. 3D 构象生成逻辑

入口函数：`_embed_and_strip_h()`

源码逻辑：

```text
1. RDKit 解析 SMILES。
2. AddHs。
3. 遇到 `*` dummy atom 时，临时替换为 dummy C。
4. 使用 ETKDGv3 嵌入 3D 构象。
5. 使用 MMFFOptimizeMolecule 做局部优化。
6. 优化后恢复 dummy atom。
7. RemoveAllHs，只保留 heavy atom 坐标。
8. 重复直到收集 NUM_CONFS 个 conformer 或达到 MAX_TRIES。
9. 如果 3D 全失败，fallback 到 2D coordinates。
10. 如果构象不足，尝试追加 2D 或复制最后一个 conformer padding。
```

关键点：

```text
- 使用多构象，而不是单构象。
- 使用 ETKDGv3 + MMFF。
- 没有显式计算每个 conformer 的能量。
- 没有按最低能量排序。
- 训练时 DataLoader 会随机选择一个 conformer。
- 验证/测试时使用第一个 conformer。
```

对 `*` 的处理方式：

```text
3D embedding 前临时把 * 替换成 C。
embedding / MMFF 优化后再恢复。
最后 RemoveAllHs，保留 heavy atom graph 和 coordinates。
```

这与当前 Uni-Poly-Plus 的 star-substitution / periodic_pbc 不同：

```text
Uni-Macro：* -> 临时 C，生成普通 3D 构象后恢复。
当前项目：Geom 可用 star-substitution capped repeat unit 或 periodic_pbc。
```

## 6. Graph / Pair Features

入口：`src/preprocess/molecular_features.py`

Atom features 包括：

```text
atomic number
chirality
total degree
formal charge
total H
radical electrons
hybridization
is aromatic
is in ring
```

Bond features 包括：

```text
bond type
bond stereo
is conjugated
```

Pair features 包括：

```text
shortest_path: RDKit topological distance
edge_feat: dense [N, N, 3]
pair_type: atom_i type x atom_j type
```

注意：Uni-Macro 不使用 PyG message passing graph，而是把 graph 拓扑转成 dense pair representation，作为 Transformer attention bias / pair pathway 输入。

## 7. 多 SMILES Segment 的处理

Uni-Macro 支持多个 SMILES segment。源码中每个 segment 单独构图和生成构象，然后在样本维度拼接：

```text
segment 0 atoms
segment 1 atoms
segment 2 atoms
...
```

segment 内：

```text
edge_feat / shortest_path 正常计算。
```

segment 间：

```text
edge_feat 填默认值。
shortest_path 填 510/511 作为 unreachable。
```

模型还会加入：

```text
CLS token
GLOB token
SEG token per segment
```

因此它能表达：

```text
多组分聚合物
共聚物不同单体片段
每个片段的局部条件或比例
全局实验/体系条件
```

这一点对当前项目很有价值，因为当前 Uni-Poly-Plus 主要把每条样本视为一个 SMILES，而 Uni-Macro 提供了“多 segment + 条件 token”的清晰设计。

## 8. DataLoader 与构象采样

入口：`src/dataloader/dataloader_polymer.py`

`_BaseDataset.__getitem__()`：

```python
coords = sample["src_pos"]
if coords.dim() == 3:
    choice = random.randrange(k) if self.augment else 0
    sample["src_pos"] = coords[choice]
```

含义：

```text
train mode: 从多个 conformer 中随机抽一个。
val mode: 固定使用第一个 conformer。
```

这是一种简单但有效的 conformer augmentation。

## 9. 预训练任务

入口：`PolymerPretrainDatasetMixin` 和 `MultiMolModel.forward(main_task='pretrain')`

Uni-Macro 预训练包含三个任务：

```text
1. Masked atom token prediction
2. Coordinate reconstruction
3. Distance reconstruction
```

### 9.1 Masked Atom Prediction

DataLoader 逻辑：

```python
mask_flag = torch.rand(N) < 0.15
target_token = token.clone()
target_token[~mask_flag] = 0
token[mask_flag] = MASK_IDX
```

模型输出：

```python
logits = lm_head(atom_repr)
```

loss：

```text
cross_entropy(logits[masked_atoms], target_token[masked_atoms])
```

### 9.2 Coordinate Reconstruction

DataLoader 逻辑：

```text
clean pos
-> 加 Gaussian noise, std = 0.2
-> noised_pos = pos + noise
-> Kabsch alignment: 将 noised_pos 对齐回 clean pos 的坐标框架
-> sample["target_pos"] = clean pos
-> sample["src_pos"] = aligned noisy pos
```

Kabsch 对齐的意义：

```text
去掉随机噪声导致的整体旋转/平移偏差，避免模型把刚体运动当成要学习的几何误差。
```

模型中：

```text
输入 noisy src_pos
-> encoder
-> MovementPredictionHead 预测 delta
-> pred_pos = noisy pos + delta
```

loss：

```text
L1 / absolute difference style coordinate loss
```

这和当前 Uni-Poly-Plus 最近改成的 direct noise prediction 不完全一样：

```text
Uni-Macro: 预测坐标修正 delta，监督 pred_pos -> clean pos。
当前项目: 直接预测 noise，监督 pred_noise -> true noise。
```

但 Uni-Macro 的 Kabsch alignment 是值得借鉴的，因为它避免 denoising 任务被刚体旋转/平移污染。

### 9.3 Distance Reconstruction

模型中：

```python
pred_dist = torch.cdist(pred_pos, pred_pos)
```

loss：

```text
L_dist = |pred_dist - target_dist|
```

目标是让模型学习相对几何结构，而不只学习绝对坐标。

当前项目之前删除了 pairwise distance reconstruction。是否恢复要谨慎：

```text
优点：强化局部几何结构一致性。
风险：和 coordinate/noise denoising 信息重叠，loss 权重不当会压制其他任务。
```

如果恢复，建议作为小权重辅助，而不是 1.0 直接相加。

## 10. 预训练 loss 合并方式

源码中预训练总 loss 是简单相加：

```python
loss = L_atom + L_coord + L_dist
```

没有动态权重，也没有 uncertainty weighting。

优化器：

```text
AdamW
linear warmup + cosine decay
AMP optional
grad clip
DDP optional
```

验证逻辑：

```text
每 val_every_steps 做 quick validation。
用 Latom + Lcoord + Ldist 判断是否保存 checkpoint。
```

## 11. 模型结构

主模型：`MultiMolModel`

整体结构：

```text
src_token + atom_feat + degree
+ CLS / GLOB / SEG tokens
+ edge / shortest path pair bias
+ 3D distance Gaussian pair bias
-> token-pair Transformer Encoder
-> pretrain heads or regression head
```

### 11.1 Node Pathway

节点 token 包括：

```text
CLS token
GLOB token
SEG tokens
Atom tokens
```

`AtomFeaturePlus` 将以下内容加和：

```text
atom discrete feature embedding
atom degree embedding
atomic-number token embedding
global numeric projection
segment numeric projection
```

### 11.2 Pair Pathway

`EdgeFeaturePlus` 构造 dense pair representation：

```text
shortest path embedding
edge feature embedding
special token virtual distance bias
```

3D 几何通过 `SE3InvariantKernel` 加入 pair representation：

```text
distance matrix
-> Gaussian radial basis
-> atom-pair-dependent affine transform
-> NonLinear projection to pair_dim
-> add into atom-atom pair_repr
```

### 11.3 Encoder

`EncoderBlock` 是 token-pair joint Transformer：

```text
node self-attention with pair bias
node FFN
outer product mixing: node -> pair
triangle multiplication: pair -> pair
pair FFN
```

它比普通 Graph Transformer 更接近 Uni-Mol / AlphaFold 风格：

```text
node representation 和 pair representation 交替更新。
```

### 11.4 MovementPredictionHead

Movement head 使用：

```text
node q/k/v attention
pair bias
attn over neighbors
relative delta_pos direction
-> predict per-atom 3D movement vector
```

这比简单 `Linear(node_repr)->3` 更几何友好，因为预测位移时显式使用了相对坐标方向。

## 12. Fine-tune 逻辑

入口：`src/main.py --main_task finetune`

流程：

```text
1. 构建 MultiMolModel。
2. 尝试加载 --weight_path。
3. 跳过 reg_head 和部分 shape 不匹配的 3D kernel 参数。
4. 可选 freeze_encoder，只训练 reg_head。
5. 按 fold 构建 train/val loader。
6. AdamW + CosineAnnealingLR。
7. 以 val RMSE 或 ACC 保存 best checkpoint。
```

回归 head：

```text
CLS representation
-> Linear 768->128
-> GELU
-> Linear 128->32
-> GELU
-> Linear 32->num_tasks
```

## 13. K-fold 逻辑

预处理阶段：

```text
如果 CSV 有 fold 列：直接使用。
如果没有：seed shuffle 后 i % kfold 分配。
```

训练阶段：

```text
train: split != fold{k}
val:   split == fold{k}
```

另外 `export_periogt_style_split_fold_pkls()` 会导出：

```text
train / val / test = 72 / 8 / 20 近似划分
```

但主 `PolymerPickleDataset` 的默认 fine-tune 逻辑实际上是：

```text
当前 fold 作为 val，其余 fold 作为 train。
```

也就是说，README 中提到的 exported train/val/test split 和 main training 使用的 fold filtering 需要区分。

## 14. 对当前 Uni-Poly-Plus 的可借鉴点

### 14.1 3D 构象生成：多构象 + 训练随机采样

当前项目已经有多构象和随机采样，但 Uni-Macro 强调：

```text
train 随机 conformer
val 固定第一个 conformer
```

当前项目可继续保留这点，但建议补充：

```text
conformer energy 计算与排序
最低能量 conformer 作为 data.pos / eval conformer
top-k low-energy conformers 作为 train random pool
```

### 14.2 3D denoising 加 Kabsch 对齐

Uni-Macro 预训练加噪后做 Kabsch alignment：

```text
noisy pos -> align to clean pos frame
```

当前项目 direct noise prediction 只对 `pos3d` 加噪。如果要进一步优化，可以测试：

```text
clean pos + noise
-> Kabsch align noisy to clean
-> 模型预测 aligned noise 或 clean-frame noise
```

这可以减少刚体运动干扰。

### 14.3 Movement head 比 Linear noise head 更合理

当前项目用：

```text
node_rep -> Linear -> 3D noise
```

Uni-Macro 用：

```text
node_repr + pair_repr + attention + delta_pos direction -> movement vector
```

对 PaiNN 来说，后续更合理的改法不是简单 Linear head，而是使用：

```text
PaiNN vector representation
或类似 MovementPredictionHead 的 equivariant/vector-aware head
```

### 14.4 Pair pathway 可替代简单 late fusion

当前 Uni-Poly-Plus 是：

```text
SMILES encoder
Graph encoder
FP MLP
Geom encoder
-> fusion
```

Uni-Macro 是：

```text
atom / segment / global tokens
+ pair graph/3D bias
-> 一个统一 token-pair Transformer
```

如果当前项目后续想做更强模型，可以考虑一个分支：

```text
Graph star-linking topology + periodic_pbc distance
-> dense pair bias
-> Graph/Geom joint Transformer
-> 再与 SMILES/FP 融合
```

### 14.5 Segment / Global tokens 适合聚合物实验条件

Uni-Macro 对共聚物和条件信息的设计比当前项目更自然：

```text
SEG token: 每个 monomer/repeat-unit segment 的局部特征
GLOB token: 温度、压力、实验条件、组成条件
```

当前项目如果未来加入：

```text
copolymer composition
monomer ratio
polymerization condition
temperature / pressure
```

可以借鉴 Uni-Macro 的 special-token 设计。

### 14.6 预训练任务组合

Uni-Macro 的预训练任务是：

```text
masked atom + coordinate reconstruction + distance reconstruction
```

当前项目是：

```text
Graph masked atom
PerioGT periodic augmentation contrastive
Geom direct noise prediction
Multi-modal alignment
```

可以借鉴的不是直接照搬，而是做 ablation：

```text
A. 当前 direct noise prediction
B. direct noise + Kabsch align
C. coordinate delta reconstruction + Kabsch align
D. small-weight distance reconstruction
E. vector-aware movement head
```

## 15. 不建议直接照搬的地方

### 15.1 对 `*` 的处理不完全适合 P-SMILES

Uni-Macro 临时 `* -> C` 是为了 RDKit embedding 稳定，但它没有显式模拟聚合物周期边界。

当前项目已经有：

```text
Graph star-linking
Geom periodic_pbc
```

因此不建议退回简单 `* -> C` 作为主线，只可作为 fallback 或 ablation。

### 15.2 没有最低能量 conformer 排序

Uni-Macro 生成多个 conformer，但没有 energy ranking。当前项目若要提高 3D 输入质量，应该补上：

```text
MMFF/UFF energy
sort conformers by energy
eval 使用 lowest-energy conformer
train 从 top-k low-energy conformers 随机采样
```

### 15.3 Pairwise distance loss 不应大权重直接恢复

Uni-Macro 直接 `L_atom + L_coord + L_dist`。当前项目已经引入动态 loss 和多模态 alignment，若直接把 `L_dist` 加回 1.0，可能压制 Graph/Alignment。建议只做小权重 ablation。

### 15.4 模型很重

默认参数：

```text
encoder_embed_dim = 768
encoder_layers = 12
encoder_attention_heads = 48
pair_embed_dim = 512
```

这对当前 8-task 聚合物数据可能过大，容易过拟合或训练成本过高。若迁移，应先做小模型版本：

```text
embed_dim 256/384
layers 4/6
heads 8/12
pair_dim 128/256
```

## 16. 建议当前项目的落地路径

### Phase A：3D 输入质量改进

```text
1. 计算 conformer energy。
2. pos_confs 按 energy 排序。
3. eval 使用最低能量 conformer。
4. train 从 top-k low-energy conformers 随机采样。
5. diagnostics 记录 conformer energy 分布。
```

### Phase B：Geom denoising 改进

```text
1. 加 Kabsch alignment 选项。
2. 保留 direct noise prediction 作为 baseline。
3. 增加 vector-aware denoising head ablation。
4. 对 periodic_pbc 保证只扰动 pos3d，不扰动 T/cell/pbc。
```

### Phase C：Graph/Geom joint pair branch

新建一个可选分支：

```text
star-linking graph features
periodic_pbc 3D distances
shortest path / bond features
-> dense pair Transformer
-> structure representation
```

不要直接替换当前 GINE + PaiNN 主线，先作为 ablation。

### Phase D：条件 token 支持

如果数据中出现实验条件或共聚物组成：

```text
seg token: monomer/repeat-unit 局部比例/DP
global token: temperature/pressure/source condition
```

这比把所有条件简单 concat 到 MLP 更合理。

## 17. 与当前项目的关键差异总结

```text
Uni-Macro:
  单一 token-pair Transformer 内部融合 2D/3D/条件。
  使用 dense pair features 和 3D distance Gaussian bias。
  多构象训练随机采样。
  预训练包含 masked atom、coordinate reconstruction、distance reconstruction。
  3D denoising 加 Kabsch alignment。

Uni-Poly-Plus 当前主线:
  多 encoder：SMILES / Graph / FP / Geom。
  Graph 使用 star-linking GINE。
  Geom 使用 PaiNN periodic_pbc。
  Fusion 使用 self-attention pooling 或 graph_geom_gate。
  预训练包含 Graph mask、PerioGT periodic augmentation、Geom direct noise prediction、alignment。
```

最值得当前项目立即借鉴的是：

```text
1. conformer energy ranking。
2. Kabsch-aligned 3D denoising ablation。
3. vector-aware movement head。
4. segment/global condition token 设计。
```

不建议马上照搬的是：

```text
1. 直接 * -> C 作为主线 Geom。
2. 直接恢复大权重 pairwise distance loss。
3. 直接使用 12-layer / 768-dim 大模型替换当前模型。
```

## 18. 对当前 TODO 的具体建议

当前如果要继续优化预测效果，优先级建议：

```text
Priority 1:
  加 conformer energy ranking，先保证 Geom 输入质量。

Priority 2:
  为当前 direct noise prediction 加 Kabsch alignment ablation。

Priority 3:
  将 geom denoising head 从 Linear 改为 vector-aware head。

Priority 4:
  小权重恢复 distance reconstruction 做 ablation。

Priority 5:
  长期分支实现 token-pair Graph/Geom joint Transformer。
```

这样比直接重写整个模型风险更低，也更容易判断每个改动是否真的提升。 
