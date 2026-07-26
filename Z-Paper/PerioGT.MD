# PerioGT 源码分析与对当前项目的借鉴

分析对象：`https://github.com/wuyuhui-zju/PerioGT/tree/main`

本分析基于源码阅读，不只基于 README。重点查看了：

```text
PerioGT_common/data/smiles2g_light.py
PerioGT_common/data/collator_light.py
PerioGT_common/data/pretrain_dataset.py
PerioGT_common/data/get_loader_pretrain.py
PerioGT_common/models/light.py
PerioGT_common/models/graphgps.py
PerioGT_common/trainer/pretrain_trainer.py
PerioGT_common/utils/aug.py
PerioGT_common/utils/features.py
PerioGT_common/scripts/prepare_pt_dataset.py
PerioGT_common/scripts/prepare_ft_dataset.py
PerioGT_copolym/*
PerioGT_with_cond/*
```

## 1. 仓库整体结构

PerioGT 仓库主要有三套实现：

```text
PerioGT_common      普通单聚合物属性预测主线
PerioGT_copolym    共聚物任务，输入两个 monomer / repeat-unit 和比例信息
PerioGT_with_cond  带实验条件或全局状态的任务
```

主线是 `PerioGT_common`。

它支持两类 backbone：

```text
light     LiGhT / KPGT 风格的 triplet transformer
GraphGPS  local GNN + global Transformer attention
```

关键结论：PerioGT 不是简单的“把 dimer/n-mer 放进 GNN”。它的主线更接近：

```text
P-SMILES 周期增强
-> 构造 triplet graph / line-graph-like graph
-> 加入 fingerprint 和 descriptor 虚拟节点
-> 多任务预训练
-> 下游 fine-tune 时使用 prompt / feature injection
```

## 2. P-SMILES 周期增强逻辑

代码位置：

```text
PerioGT_common/utils/aug.py
```

### 2.1 generate_multimer_smiles

`generate_multimer_smiles(num_repeat_units, smiles, replace_dummy_atoms=True)` 用于把 P-SMILES 拼成有限 oligomer。

逻辑：

1. 找到两个 `*` 原子。
2. 找到两个 `*` 的 neighbor atoms。
3. 检查两个 attachment bond type 是否一致。
4. 复制 repeat unit 多次。
5. 在相邻 repeat units 的边界 neighbor atoms 之间加键。
6. 删除内部连接处的 `*`。
7. 如果 `replace_dummy_atoms=True`，两端 `*` 用 H 替换。
8. 返回 RDKit canonical SMILES。

这说明 PerioGT 的 fingerprint / descriptor 不是直接对原始 repeat unit 计算，而是经常对 `n=3/6/9` oligomer 计算。

### 2.2 periodicity_augment

`periodicity_augment(smiles, max_mrus=3)` 是 PerioGT 里最重要的周期增强。

逻辑：

1. 找到两个 `*` 原子及其 neighbor atoms。
2. 计算两个 `*` 之间的 shortest path。
3. 从该路径中选择一个非 ring bond。
4. 切断这个 bond。
5. 在切断位置两端加新的 `*`。
6. 把原始两个 `*` 的 neighbor atoms 连接起来。
7. 删除原始两个 `*`。
8. 随机生成 1 到 `max_mrus` 个 repeat unit 的 multimer。

含义：

```text
它不是简单 shift SMILES 字符串。
它是在分子图上改变 repeat-unit 边界位置，让同一个无限聚合物有不同 P-SMILES 表示。
```

这就是 PerioGT 的周期等价增强，目的类似 RSIT：让模型对 repeat / shift 等价表示保持鲁棒。

### 2.3 periodicity_augment_traverse

`periodicity_augment_traverse(smiles)` 会遍历 backbone path 上的所有可切断非环 bond，生成多个等价 P-SMILES。

下游 prompt 构造中会用它生成多个增强图。

## 3. Graph 构建逻辑：不是普通原子图，而是 Triplet Graph

代码位置：

```text
PerioGT_common/data/smiles2g_light.py
```

PerioGT 的 graph 不是当前项目里的普通 atom graph。

当前项目：

```text
node = atom
edge = bond
```

PerioGT light：

```text
node = atom pair + bond，也就是 bonded triplet
edge = triplet 之间的 path / line-graph relation
```

### 3.1 Triplet node

对每条化学键 `(u, v)`，构造一个 graph node：

```text
begin_end = [atom_feature(u), atom_feature(v)]
edge      = bond_feature(u, v)
label     = vocab(atom_type_u, bond_type, atom_type_v)
```

这非常接近 line graph：原始分子图的 bond 被提升为模型图中的 node。

此外，如果有孤立 atom，也会构造一个带 virtual atom 的 triplet node。

### 3.2 Triplet graph edge

PerioGT 构造两类 triplet node 之间的连接：

```text
line graph path：两个 bond 共享同一个 atom
molecule graph path：两个 bond 在原分子图中存在 shortest path，且 path length 在限制范围内
```

每条 edge 上保存：

```text
path: triplet path indices
lgp: 是否 line graph path
mgp: 是否 molecule graph path
vp:  是否 virtual path
sl:  是否 self loop
```

模型里会根据这些 path 信息生成 attention bias。

### 3.3 虚拟节点：fingerprint / descriptor node

PerioGT 会额外加入 virtual nodes：

```text
vavn == 1  fingerprint virtual node
vavn == 2  molecular descriptor virtual node
```

这两个节点和所有 triplet nodes 连接。

模型输入时：

```text
普通 triplet node: atom pair embedding + bond embedding
fp virtual node:   fingerprint projection
md virtual node:   descriptor projection
```

这点非常重要：PerioGT 不是把 FP / descriptor 作为外部模态最后拼接，而是把它们作为 graph 中的虚拟节点，直接参与 message passing / attention。

## 4. LiGhT 模型逻辑

代码位置：

```text
PerioGT_common/models/light.py
```

核心模块：

```text
AtomEmbedding
BondEmbedding
TripletEmbedding
LiGhT / TripletTransformer
LiGhTPredictor
```

### 4.1 TripletEmbedding

输入：

```text
node_h = atom pair embedding
edge_h = bond embedding
fp     = fingerprint vector
md     = descriptor vector
```

普通 triplet node：

```text
triplet_h = MLP([node_h, edge_h])
```

虚拟节点：

```text
vavn == 1 -> fp_proj(fp)
vavn == 2 -> md_proj(md)
```

因此 FP / descriptor 是作为节点表示注入模型，而不是简单拼接在最后。

### 4.2 TripletTransformer

TripletTransformer 的 attention 包含三部分：

```text
node attention = QK attention
距离 attention = path length embedding
路径 attention = path 上 triplet features 聚合后生成 bias
```

源码中：

```python
g.edata['a'] = node_attn + dist_attn + path_attn
```

含义：

```text
attention 不只看 node embedding，还显式看 path distance 和 path content。
```

这对聚合物很有价值，因为聚合物性质往往受 backbone path、侧链距离、功能基团间路径影响。

### 4.3 Readout

模型输出 graph representation 时，会取：

```text
fp virtual node embedding
md virtual node embedding
普通 triplet nodes readout
```

然后拼接：

```text
g_feats = [fp_vn, md_vn, graph_readout]
```

这比当前项目的：

```text
Graph embedding + FP embedding + Geom embedding 分开融合
```

更早地融合了结构图和 descriptor/fingerprint。

## 5. GraphGPS 版本

代码位置：

```text
PerioGT_common/models/graphgps.py
```

GraphGPS 版本仍然沿用 triplet node 和 virtual nodes，但模型层换成：

```text
local GNN = GIN 或 PNA
local message passing
+ global Transformer attention
+ FFN
```

GraphGPSLayer 中：

```text
h_local = local_gnn(g, h)
h_attn  = MultiheadAttention(dense_batch(h))
h       = h_local + h_attn + FFN
```

它比 `light` 更接近普通 Graph Transformer，但依然不是原子图，而是在 triplet graph 上做。

对当前项目的意义：

```text
如果后续要引入 Graph Transformer，不能只把普通 atom graph 丢进 Transformer。
更合理的是加入 shortest-path / path-content bias，或者转成 triplet/line graph。
```

## 6. 预训练数据和特征生成

代码位置：

```text
PerioGT_common/data/pretrain_dataset.py
PerioGT_common/scripts/prepare_pt_dataset.py
PerioGT_common/utils/features.py
```

预训练数据读取：

```text
product_smiles.csv
maccs_ecfp_n3.npz
maccs_ecfp_n6.npz
maccs_ecfp_n9.npz
molecular_descriptors_n3_norm.npz
molecular_descriptors_n6_norm.npz
molecular_descriptors_n9_norm.npz
```

### 6.1 fingerprint

PerioGT 使用：

```text
MACCS fingerprint
Morgan fingerprint radius=4, nBits=1024
```

拼接后作为 binary fingerprint。

### 6.2 descriptor

PerioGT 使用：

```text
Mordred descriptors, ignore_3D=True
```

也就是说它主要是 2D descriptor，不是 3D descriptor。

### 6.3 n=3/6/9 oligomer features

预训练时会为同一个 polymer 生成不同长度 oligomer 的 FP / descriptor：

```text
n3
n6
n9
```

Collator 中周期增强生成的样本会根据 repeat unit 数 `n` 选择对应的 feature：

```text
n=1 -> 使用 n3 特征
n=2 -> 使用 n6 特征
n=3 -> 使用 n9 特征
```

注意：这里的 n3/n6/n9 不是直接作为图输入主干，而主要用于虚拟节点的 FP / descriptor 表征。

## 7. 预训练任务

代码位置：

```text
PerioGT_common/data/collator_light.py
PerioGT_common/trainer/pretrain_trainer.py
PerioGT_common/models/light.py
```

PerioGT 预训练包含四个任务。

### 7.1 Masked triplet prediction

Collator 中 `bert_mask_nodes()` 随机选择 triplet nodes：

```text
candi_rate
mask_rate
replace_rate
keep_rate
```

类似 BERT：

```text
mask 一部分
replace 一部分
keep 一部分
```

模型预测被 mask / replace / keep 的 triplet label。

loss：

```text
sl_loss = CrossEntropy(sl_predictions, sl_labels)
```

### 7.2 Fingerprint reconstruction

输入给模型的是 disturbed fingerprint。

模型通过 fingerprint virtual node 预测原始 fingerprint：

```text
fp_loss = BCEWithLogits(fp_predictions, fps)
```

### 7.3 Descriptor reconstruction

输入给模型的是 disturbed descriptor。

模型通过 descriptor virtual node 预测原始 descriptor：

```text
md_loss = MSE(md_predictions, mds)
```

### 7.4 Periodicity augmentation contrastive learning

Collator 每个 batch 会包含：

```text
原始 smiles graph
周期增强 smiles graph
```

模型输出 `cl_projector(g_feats)`。

NCE loss 将原始表示和增强表示作为正样本，不同样本作为负样本：

```text
cl_loss = DistributedNCELoss(cl_projections, temperature=0.2)
```

总 loss：

```text
loss = ((sl_loss + fp_loss + md_loss) / 3) + cl_loss
```

这点对当前项目非常重要：PerioGT 的预训练不是单纯 contrastive，也不是单纯 masked atom，而是结构重构 + descriptor/fingerprint 重构 + 周期增强对比。

## 8. 下游 Fine-tune 逻辑

代码位置：

```text
PerioGT_common/scripts/prepare_ft_dataset.py
PerioGT_common/data/finetune_dataset.py
PerioGT_common/trainer/finetune_trainer.py
```

### 8.1 下游图预处理

`prepare_ft_dataset.py` 会：

1. 读取任务 CSV。
2. 预计算 FP / Mordred descriptor。
3. 构建 DGL graphs。
4. 保存：

```text
graphs.pkl
maccs_ecfp.npz
polymer_descriptors.npz
```

### 8.2 Prompt 注入机制

如果开启 `--use_prompt`，下游构图会：

1. 对同一个 base SMILES 生成多个 periodicity augmented graphs。
2. 用预训练模型 `generate_node_emb()` 生成增强图中的 triplet node embedding。
3. 根据 bond id mapping 将增强图 embedding 聚合回 base graph 的对应 triplet node。
4. 保存到：

```text
g.ndata['prompt']
```

fine-tune 时：

```text
triplet_h = node_attn(g.ndata['prompt'], triplet_h)
```

也就是说，PerioGT 不是简单加载预训练权重，而是把周期增强视角生成的节点级 prompt embedding 注入下游图。

这比当前项目的 `load pretrained encoder -> supervised fine-tune` 更复杂。

## 9. Copolymer 与 Condition 变体

### 9.1 PerioGT_copolym

输入：

```text
smiles_A
smiles_B
ratio / architecture 信息
```

数据准备中会：

```text
分别计算 monomer A / B 的 FP 和 descriptor
构造 copolymer graph
保存 ratio.npz
```

ratio 信息包括：

```text
比例，例如 0.25 / 0.5 / 0.75
结构类型，例如 alternating / block / random
```

对当前项目帮助：如果未来要处理共聚物，这部分比 common 更有用。

### 9.2 PerioGT_with_cond

输入带 global state / condition。

数据准备会保存：

```text
global_state.npz
```

对当前项目帮助：如果后续引入实验条件、温度、溶剂、制备条件，这个分支提供了模板。

## 10. 和当前 Uni-Poly 项目的核心差异

当前项目：

```text
SMILES encoder
Graph encoder = 原子图 GIN，当前支持 star_linking
FP encoder
Geom encoder = SchNet / PaiNN
Fusion = modality-level attention
Pretrain = 多模态 embedding contrastive
```

PerioGT：

```text
Graph = triplet graph / line-graph-like graph
FP / descriptor = virtual nodes 注入 graph
Periodicity augmentation = 改变 P-SMILES 边界位置
Pretrain = masked triplet + fp reconstruction + descriptor reconstruction + periodic contrast
Fine-tune = 可用 prompt 注入增强视角节点表示
```

最主要差异：

```text
当前项目是多模态后融合。
PerioGT 是把结构、指纹、描述符更早注入同一个 graph transformer。
```

## 11. 对当前项目最有帮助的点

### Priority 1：保留并强化 star-linking / 周期等价增强

当前项目已经有 `graph_input=star_linking`。PerioGT 进一步说明，仅仅 star-linking 还不够，应该测试模型对周期等价 P-SMILES 是否鲁棒。

建议加入：

```text
periodicity_augment
periodicity_augment_traverse
RSIT evaluation
```

用途：

```text
同一聚合物的不同 P-SMILES 表示，预测应该接近。
```

### Priority 2：把 Graph 从 atom graph 升级为 bond/triplet graph 或 line graph

PerioGT 的 triplet graph 比普通 atom graph 更适合表达：

```text
bond type
atom-pair structure
path relation
backbone path
局部拓扑距离
```

当前项目短期不一定要完整迁移 DGL LiGhT，但可以借鉴：

```text
在 PyG 中构造 line graph / edge graph
或者在 Graph Transformer 中加入 shortest-path bias
```

### Priority 3：将 FP / descriptor 作为虚拟节点，而不是只在最终融合

当前项目 FP 是独立模态，最后和 graph/geom 融合。

PerioGT 的做法：

```text
FP virtual node
MD virtual node
参与 graph attention / message passing
```

可借鉴方案：

```text
在 Graph 分支增加 virtual descriptor node
把 FP / descriptor 投影成 virtual node feature
让 Graph encoder 提前融合结构和描述符
```

这比简单 `fp_embedding + graph_embedding` 后融合更强。

### Priority 4：预训练任务应更具体

当前项目的通用多模态 contrastive 太泛。

PerioGT 给出更具体的预训练组合：

```text
Graph masked triplet / atom prediction
FP reconstruction
Descriptor reconstruction
Periodicity contrastive learning
```

当前项目可逐步实现：

```text
第一步：Graph-Geom contrastive pretraining
第二步：Graph masked atom / bond prediction
第三步：FP / descriptor reconstruction
第四步：periodicity contrastive learning
```

### Priority 5：Prompt 注入机制可作为后续增强

PerioGT 下游 fine-tune 会把增强图生成的节点 embedding 注入 base graph 的 `prompt`。

这对当前项目有启发：

```text
预训练模型不只是加载权重。
也可以离线生成增强视角 embedding，作为下游 graph node 的 prompt feature。
```

但该机制工程量较大，不建议短期做。

## 12. 不建议直接照搬的点

### 12.1 不建议直接迁移 DGL 架构

当前项目基于 PyTorch Geometric。PerioGT 使用 DGL。

直接迁移会导致：

```text
Data object 全部重写
collate 全部重写
模型接口不兼容
缓存格式不兼容
```

建议只迁移思想，不直接复制 DGL 实现。

### 12.2 不建议恢复 n-mer 作为 Graph 主输入

PerioGT 中 n=3/6/9 主要用于 FP / descriptor 计算和周期增强特征，不等于它把 n-mer 全部作为主图输入。

当前项目之前 n-mer / dimer 训练很慢，不建议作为主线恢复。

### 12.3 不建议马上实现 prompt injection

Prompt injection 依赖：

```text
bond id mapping
periodicity augmented graph
预训练模型 node embedding
base graph 对齐
```

实现复杂，且当前项目 graph 是 atom graph，不是 triplet graph。应放到后续阶段。

### 12.4 不建议用 PerioGT 替代 Geom

PerioGT 的 descriptor 是 `ignore_3D=True` 的 Mordred 2D descriptor。它不是 SchNet / PaiNN 式 3D 几何模型。

当前项目已有 Geom 分支，应该保留：

```text
Geom = SchNet / PaiNN
```

PerioGT 更适合增强 Graph / FP / descriptor，而不是替代 Geom。

## 13. 推荐迁移路线

### Phase 1：周期鲁棒性与 Star-Linking 稳定化

实现：

```text
periodicity_augment / periodicity_augment_traverse
RSIT evaluation
star_linking fallback audit
```

目标：

```text
确认 star_linking 是否确实提升 P-SMILES 等价表示鲁棒性。
```

### Phase 2：Graph 结构增强

实现：

```text
backbone atom feature
attachment distance feature
shortest-path distance feature
```

保持 Graph encoder 仍为 GIN，避免一次性重写。

### Phase 3：Graph 预训练增强

实现：

```text
masked atom / bond prediction
periodicity contrastive learning
```

目标：

```text
让 Graph 分支专门学习聚合物周期等价性。
```

### Phase 4：FP / Descriptor 虚拟节点

实现：

```text
Graph 分支添加 virtual fp node / descriptor node
```

可选 descriptor：

```text
当前 RDKit/Morgan/MACCS
Mordred 2D descriptor
主链长度 / attachment distance / side-chain statistics
```

### Phase 5：Line Graph / Triplet Graph

如果 Phase 1-4 有收益，再考虑把 Graph 从 atom graph 升级为：

```text
line graph
triplet graph
Graph Transformer with path bias
```

## 14. 对当前 TODO 路线的修正建议

当前 TODO 中的推荐主线：

```text
Graph = star-linking topology graph
Geom  = original repeat-unit 3D geometry
Fusion = graph-geom gated fusion + modality attention
Pretrain = graph-geom contrastive pretraining
```

阅读 PerioGT 源码后，建议补充：

```text
1. Graph star-linking 后必须做 periodicity robustness evaluation。
2. Graph 分支应优先加入 masked atom/bond 或 masked triplet pretraining。
3. FP/descriptor 可以考虑作为 Graph virtual nodes，而不是只做独立 fp 模态。
4. n=3/6/9 更适合作为 descriptor/fingerprint 计算尺度，不适合作为当前 Graph 主输入。
5. PerioGT 的 3D 部分不是 SchNet/PaiNN，不应替代当前 Geom。
```

## 15. 最终结论

PerioGT 对当前项目最有价值的不是完整模型复刻，而是以下四个思想：

```text
1. 周期等价增强：periodicity_augment / RSIT。
2. 边/原子对级图：triplet graph / line graph。
3. 结构 + FP + descriptor 早期融合：virtual nodes。
4. 预训练任务具体化：masked structure + FP/descriptor reconstruction + periodic contrastive。
```

当前项目最合理的落地顺序：

```text
先做 star_linking + RSIT
再做 backbone / shortest-path feature
再做 graph-specific pretraining
最后考虑 line graph / virtual descriptor nodes
```

不要直接把 PerioGT 的 DGL 代码整体搬进当前项目；应该按 PyG 当前数据流增量吸收其关键机制。
