# Uni-Poly-Plus KG 模块技术说明与运行文档

本文档基于当前仓库实际代码实现编写，不基于最初设计方案推测。

已检查的核心文件：

- `src/kg/kg_loader.py`
- `src/kg/kg_mapping.py`
- `src/kg/kg_utils.py`
- `src/modules/kg.py`
- `src/dataset/dataset.py`
- `src/dataset/dataloader.py`
- `src/modules/uni_encoder.py`
- `scripts/pretrain.py`
- `scripts/train.py`
- `src/utils.py`

## 1. 知识图谱整体流程

当前 KG 数据流如下：

```text
SMILES
↓
RDKit Chem.MolFromSmiles
↓
元素提取
↓
官能团 SMARTS 匹配
↓
KG entity id 映射
↓
Dataset 写入 data.kg_entity_ids
↓
Dataloader padding 得到 batch.kg_entity_ids / batch.kg_mask
↓
KGEncoder
↓
kg_embedding [B,256]
↓
UniEncoderAttention modal_embeddings
↓
FusionModule
↓
MLP Prediction
```

完整调用链：

```text
scripts/train.py 或 scripts/pretrain.py
↓
UniDataset(..., use_kg='kg' in args.modalities)
↓
UniDataset._attach_kg_fields()
↓
src/kg/kg_mapping.py::smiles_to_kg_entity_ids()
↓
src/kg/kg_loader.py::load_kg_embedding_store()
↓
src/dataset/dataloader.py::custom_collate()
↓
src/kg/kg_utils.py::pad_kg_entity_ids()
↓
src/modules/uni_encoder.py::EncoderModule.forward()
↓
src/modules/kg.py::KGEncoder.forward()
↓
src/modules/uni_encoder.py::FusionModule.forward()
↓
prediction
```

每一步输入输出：

```text
输入:
SMILES string

RDKit解析输出:
mol: RDKit Mol

元素提取输出:
element entity ids，例如 C/O -> [6,8]

官能团检测输出:
functional group entity ids，例如 Alkyl/Phenyl/Carbonyl 等 -> [113,160,130...]

Dataset输出:
data.kg_entity_ids: [K]
data.kg_mask: [K]，当前 Dataset 中全 True

Dataloader输出:
batch.kg_entity_ids: [B,Kmax]
batch.kg_mask: [B,Kmax]

KGEncoder输出:
kg_embedding: [B,256]

UniEncoderAttention输出:
modal_embeddings: [B,num_modalities,256]
五模态时为 [B,5,256]

FusionModule输出:
fused_output: [B,256]
modality_attention: [B,num_modalities]

最终预测:
output: [B,1]
```

## 2. KG资源加载流程

当前运行时实际使用以下 KG 文件。

### `knowledge_graph/KANO_initial/ele2emb.pkl`

作用：KANO 元素 embedding。

加载位置：`src/kg/kg_loader.py::load_kg_embedding_store()`。

加载时机：当 `use_kg=True` 时，`UniDataset.__init__()` 会调用 `load_kg_embedding_store()`；此外 `KGEncoder.__init__()` 也会调用一次。函数内部有 `_KG_STORE_CACHE`，相同路径会复用缓存。

输出内容：

```text
ele2emb: dict
key: KANO element index，即 atomic_number - 1
value: 133维 embedding
```

### `knowledge_graph/KANO_initial/fg2emb.pkl`

作用：KANO 官能团 embedding。

加载位置：`src/kg/kg_loader.py::load_kg_embedding_store()`。

加载时机：同 `ele2emb.pkl`。

输出内容：

```text
fg2emb: dict
key: functional group name
value: 133维 embedding
```

### `knowledge_graph/mappings/functional_group_smarts.json`

作用：官能团名称到 SMARTS 规则的映射。

加载位置：`src/kg/kg_mapping.py::load_functional_group_smarts()`。

加载时机：调用 `smiles_to_kg_entity_ids()` 时加载；内部有 `_SMARTS_CACHE`。

输出内容：

```text
[(functional_group_name, RDKit SMARTS Mol), ...]
```

### `knowledge_graph/mappings/entity_name_alias.json`

作用：将官能团名称规范化到 `fg2emb.pkl` 使用的 key。

加载位置：`src/kg/kg_mapping.py::load_entity_alias()`。

加载时机：调用 `smiles_to_kg_entity_ids()` 时加载；内部有 `_ALIAS_CACHE`。

输出内容：

```text
alias dict: raw_name -> canonical_name
```

### `knowledge_graph/mappings/element_symbol_mapping.json`

作用：元素符号到 KANO element index 的映射。

加载位置：`src/kg/kg_mapping.py::load_element_symbol_mapping()`。

加载时机：调用 `smiles_to_kg_entity_ids()` 时加载；内部有 `_ELEMENT_SYMBOL_CACHE`。

输出内容：

```text
"C" -> 5
"N" -> 6
"O" -> 7
"Si" -> 13
...
```

注意：当前 `rel2emb.pkl`、`elementkg.owl`、`elementkgontology.embeddings.txt` 不在模型前向中直接使用。当前 KG 模块只使用离线 `ele2emb.pkl` 和 `fg2emb.pkl`。

## 3. KG实体构建逻辑

示例：

```text
CC(=O)OC1=CC=CC=C1C(=O)O
```

使用当前代码实际运行得到：

```text
kg_entity_ids = [6, 8, 113, 160, 144, 134, 126, 139, 130]
```

### 元素提取逻辑

文件：`src/kg/kg_mapping.py`

函数：`_extract_element_entity_ids()`

流程：

```text
mol.GetAtoms()
↓
atom.GetAtomicNum()
↓
如果 atomic_num <= 0，则跳过
↓
atom.GetSymbol()
↓
element_symbol_mapping.get(symbol, atomic_num - 1)
↓
kg_store.element_to_entity_id[element_index]
↓
去重后加入 entity_ids
```

示例中实际元素：

```text
C -> element_index 5 -> entity_id 6
O -> element_index 7 -> entity_id 8
```

注意这里的 `entity_id` 不是 KANO 原始 element index。当前 `kg_loader.py` 会构造一个统一 embedding matrix：

```text
0: padding
1-108: elements
109-190: functional groups
```

所以 `C` 的 KANO index 是 `5`，但统一 `entity_id` 是 `6`。

### SMARTS匹配逻辑

文件：`src/kg/kg_mapping.py`

函数：`_extract_functional_group_entity_ids()`

流程：

```text
读取 functional_group_smarts.json
↓
Chem.MolFromSmarts(smarts)
↓
mol.HasSubstructMatch(pattern)
↓
命中则进入 alias 映射
↓
kg_store.functional_group_to_entity_id[canonical_name]
↓
去重后加入 entity_ids
```

示例实际匹配到的官能团：

```text
Alkyl -> entity_id 113
Phenyl -> entity_id 160
Hydroxyl -> entity_id 144
Carboxyl -> entity_id 134
Carboalkoxy -> entity_id 126
Ether -> entity_id 139
Carbonyl -> entity_id 130
```

### alias映射逻辑

```python
_canonical_name(name, alias)
```

实际行为：

```text
alias.get(name, alias.get(name.lower(), name))
```

也就是说先查原始名称，再查小写名称，如果都没有，就直接用原始名称。

### padding逻辑

文件：`src/kg/kg_utils.py`

函数：`pad_kg_entity_ids()`

输入：

```text
entity_id_list: List[Tensor[K_i]]
```

输出：

```text
padded: [B,Kmax]
mask: [B,Kmax]
```

规则：

```text
padding_idx = 0
有效实体位置 mask=True
padding位置 mask=False
```

如果某个样本没有任何 KG 实体：

```text
ids长度为0
该样本整行 mask=False
```

如果整个 batch 都没有 KG 实体：

```text
Kmax 会被设置为 1
kg_entity_ids: [B,1]，全 0
kg_mask: [B,1]，全 False
```

## 4. KGEncoder结构

文件：`src/modules/kg.py`

类：`KGEncoder`

初始化：

```python
KGEncoder(
    joint_embedding_dim=256,
    kg_root='.',
    freeze_kg_embeddings=False
)
```

实际模块结构：

```text
kg_entity_ids [B,K]
↓
nn.Embedding.from_pretrained(...)
↓
embeddings [B,K,133]
↓
Linear(133,1)
↓
scores [B,K]
↓
mask-aware softmax
↓
weights [B,K]
↓
weighted sum
↓
pooled [B,133]
↓
Linear(133,256)
↓
LayerNorm(256)
↓
ReLU
↓
kg_embedding [B,256]
```

当前实际 `embedding_weight` 验证结果：

```text
[191,133]
```

含义：

```text
191 = 1 padding + 108 elements + 82 functional groups
133 = KANO embedding dim
```

KGEncoder 参数：

```text
embedding.weight: [191,133]
attention.weight: [1,133]
attention.bias: [1]
projection.0.weight: [256,133]
projection.0.bias: [256]
projection.1.weight: [256]
projection.1.bias: [256]
```

当前默认：

```python
freeze_kg_embeddings=False
```

所以：

```text
KG embedding.weight 可训练
attention 可训练
projection 可训练
```

但是 `EncoderModule.__init__()` 有：

```python
if encoder and freeze_encoder:
    for param in self.encoder.parameters():
        param.requires_grad = False
```

因此如果运行时传入：

```bash
--freeze_encoder
```

那么 `kg` encoder 的所有参数也会被冻结，包括：

```text
embedding
attention
projection
```

## 5. 与多模态融合的关系

模型文件：`src/modules/uni_encoder.py`

`UniEncoderAttention.forward()` 实际逻辑：

```python
embeddings = [self.encoders[modality](data) for modality in self.modality_list]
embeddings = torch.stack(embeddings, dim=1)
fused_output, modality_attention = self.fusion_module(embeddings)
output = self.mlp(fused_output)
return output, embeddings
```

如果运行：

```bash
--modalities smiles graph fp geom kg
```

模态顺序就是命令行顺序：

```text
smiles -> graph -> fp -> geom -> kg
```

每个模态输出：

```text
smiles: [B,256]
graph:  [B,256]
fp:     [B,256]
geom:   [B,256]
kg:     [B,256]
```

堆叠后：

```text
modal_embeddings: [B,5,256]
```

FusionModule 输入：

```text
embeddings: [B,M,256]
```

其中 `M = len(modality_list)`。

FusionModule 内部：

```text
[B,M,256]
↓ permute
[M,B,256]
↓ MultiheadAttention
[M,B,256]
↓ permute back
[B,M,256]
↓ residual + LayerNorm
[B,M,256]
↓ FeedForward + residual + LayerNorm
[B,M,256]
↓ AttentionPooling
fused_output: [B,256]
attention_weights: [B,M]
```

最后 MLP：

```text
[B,256]
↓ Linear(256,128)
↓ ReLU
↓ Linear(128,64)
↓ ReLU
↓ Linear(64,1)
↓
prediction [B,1]
```

## 6. 训练流程

### 预训练流程

入口：`scripts/pretrain.py`

KG 生效判断位置：

```python
use_kg='kg' in args.modalities
```

完整流程：

```text
python scripts/pretrain.py --modalities smiles graph fp geom kg
↓
parse_arguments()
↓
UniDataset(..., use_kg=True)
↓
Dataset 对每个 SMILES 生成 kg_entity_ids
↓
get_data_loader()
↓
custom_collate() padding KG ids
↓
UniEncoderAttention(modality_list=args.modalities)
↓
EncoderModule('kg') 创建 KGEncoder
↓
model(data)
↓
返回 embeddings [B,M,256]
↓
compute_contrastive_loss(embeddings)
↓
loss.backward()
↓
clip_grad_norm_
↓
optimizer.step()
↓
保存 pretrained model
```

当前预训练 loss：

文件：`src/utils.py`

函数：`compute_contrastive_loss()`

它会遍历所有模态对：

```python
for i in range(num_modalities):
    for j in range(num_modalities):
        if i != j:
```

因此加入 `kg` 后，KG 会自动参与和其他所有模态的对比学习。不新增 KG loss。

### 微调流程

入口：`scripts/train.py`

KG 生效判断位置：

```python
model_modality_list = args.modalities
use_kg = 'kg' in model_modality_list
```

完整流程：

```text
python scripts/train.py --modalities smiles graph fp geom kg
↓
UniDataset(..., use_kg=True)
↓
5-fold KFold
↓
get_data_loader()
↓
custom_collate()
↓
UniEncoderAttention(modality_list=model_modality_list)
↓
如果 pretrained_model_path 非空，strict=False 加载预训练权重
↓
train_and_evaluate()
↓
train_epoch()
↓
outputs, embeddings = model(batch)
↓
FusionModule 融合 kg 和其他模态
↓
criterion = MSELoss()
↓
loss.backward()
↓
clip_grad_norm_
↓
optimizer.step()
↓
evaluate/test
↓
collect_attention_pooling_weights()
↓
results.csv 写入 attention
```

微调 loss：

```text
MSELoss(outputs, batch.y)
```

KG 不单独计算 loss，只通过最终预测 loss 反向传播。

## 7. 当前项目运行方法

### 环境准备

代码没有写死 conda 环境名，但本机已有 `Uni-Poly` 环境。实际运行可用：

```bash
conda activate Uni-Poly
```

### KG资源检查

必须存在：

```text
knowledge_graph/KANO_initial/ele2emb.pkl
knowledge_graph/KANO_initial/fg2emb.pkl
knowledge_graph/mappings/functional_group_smarts.json
knowledge_graph/mappings/entity_name_alias.json
knowledge_graph/mappings/element_symbol_mapping.json
```

当前实现中不直接使用但已存在的资源：

```text
knowledge_graph/KANO_initial/rel2emb.pkl
knowledge_graph/KANO_initial/elementkgontology.embeddings.txt
knowledge_graph/KANO_initial/objectproperty.txt
knowledge_graph/KGembedding/elementkg.owl
```

### 数据准备

预训练默认读取：

```text
data/raw/smi_all.csv
```

因为 `scripts/pretrain.py` 默认：

```python
--dataset_name smi_all
--root ./data
```

微调默认读取这些任务 CSV：

```text
data/raw/smi_tg.csv
data/raw/smi_eea.csv
data/raw/smi_egb.csv
data/raw/smi_egc.csv
data/raw/smi_ei.csv
data/raw/smi_eps.csv
data/raw/smi_nc.csv
data/raw/smi_xc.csv
data/raw/smi_eat.csv
```

还需要：

```text
data/smiles_text_dict.json
```

因为 `UniDataset.__init__()` 固定读取它：

```python
self.text_dict = json.load(open('./data/smiles_text_dict.json', 'r'))
```

### 预训练命令

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --root ./data \
  --modalities smiles graph fp geom kg \
  --geometry_encoder schnet \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-4 \
  --temperature 0.07 \
  --max_grad_norm 1.0 \
  --save_path ./pretrained_models/saved_pretrained_model.pth
```

参数说明：

```text
--dataset_name: 读取 data/raw/{dataset_name}.csv
--root: 数据根目录
--modalities: 模态列表；包含 kg 才启用 KG
--geometry_encoder: painn 或 schnet
--batch_size: batch size
--epochs: 预训练 epoch 数
--lr: Adam 学习率
--temperature: contrastive loss 温度
--max_grad_norm: 梯度裁剪阈值
--save_path: 保存预训练模型权重
```

### 微调命令

```bash
python scripts/train.py \
  --tasks tg \
  --modalities smiles graph fp geom kg \
  --geometry_encoder schnet \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth \
  --epochs 100 \
  --patience 10 \
  --batch_size 32 \
  --max_grad_norm 1.0 \
  --results_dir ./results/results.csv \
  --models_dir ./saved_models
```

参数说明：

```text
--tasks: 训练任务；读取 data/raw/smi_{task}.csv
--modalities: 模态列表；包含 kg 才启用 KG
--geometry_encoder: painn 或 schnet
--pretrained_model_path: 加载预训练模型；strict=False
--epochs: 最大训练 epoch
--patience: early stopping patience
--batch_size: batch size
--max_grad_norm: 梯度裁剪阈值
--results_dir: results.csv 输出路径
--models_dir: best model 保存目录
```

### 如何确保 KG 生效

必须在命令中包含：

```bash
--modalities smiles graph fp geom kg
```

代码判断位置：

预训练：

```python
# scripts/pretrain.py
use_kg='kg' in args.modalities
```

微调：

```python
# scripts/train.py
use_kg = 'kg' in model_modality_list
```

模型接入位置：

```python
# src/modules/uni_encoder.py
elif modality == 'kg':
    encoder = KGEncoder(joint_embedding_dim=joint_embedding_dim)
```

前向位置：

```python
if self.modality == 'kg':
    return self.encoder(data.kg_entity_ids, data.kg_mask)
```

## 8. 结果验证

### 检查 Dataset 是否有 KG 字段

```python
from src.dataset import UniDataset

dataset = UniDataset(
    root='./data',
    dataset='smi_tg',
    smiles_model_name='./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k',
    text_model_name='./pretrained_models/encoders/multitask-text-and-chemistry-t5-base-augm',
    geometry_encoder='schnet',
    use_kg=True
)

data = dataset[0]
print(data.kg_entity_ids)
print(data.kg_mask)
```

如果 KG 生效，应该存在：

```text
data.kg_entity_ids
data.kg_mask
```

### 检查 batch

```python
from src.utils import get_data_loader

loader = get_data_loader(dataset, batch_size=4)
batch = next(iter(loader))

print(batch.kg_entity_ids.shape)
print(batch.kg_mask.shape)
```

期望：

```text
kg_entity_ids: [B,K]
kg_mask: [B,K]
```

### 检查模型结构

```python
from src.modules import UniEncoderAttention

model = UniEncoderAttention(
    joint_embedding_dim=256,
    smiles_model_name='./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k',
    text_model_name='./pretrained_models/encoders/multitask-text-and-chemistry-t5-base-augm',
    gnn_model_name='',
    geom_model_name='',
    modality_list=['smiles', 'graph', 'fp', 'geom', 'kg'],
    geometry_encoder='schnet'
)

print(model.encoders.keys())
print(model.encoders['kg'])
```

应该看到：

```text
kg
KGEncoder
```

### 检查 forward 输出

```python
batch = batch.to(device)
outputs, embeddings = model(batch)

print(outputs.shape)
print(embeddings.shape)
```

五模态时：

```text
outputs: [B,1]
embeddings: [B,5,256]
```

### 检查 attention

训练后 `scripts/train.py` 会打印：

```text
Fold X attention shape: (...)
Fold X mean attention: smiles:...;graph:...;fp:...;geom:...;kg:...
5-fold mean attention: smiles:...;graph:...;fp:...;geom:...;kg:...
```

如果 `attention` 中出现 `kg:数值`，说明 KG 已经进入 FusionModule 的 attention pooling。

### 检查 results.csv

训练完成后检查：

```text
results/results.csv
```

其中：

```text
model_modality_list
attention
```

应包含 `kg`，例如：

```text
attention = smiles:...;graph:...;fp:...;geom:...;kg:...
```

### 检查保存权重

预训练或微调保存的 state_dict 中应该包含：

```text
encoders.kg.encoder.embedding.weight
encoders.kg.encoder.attention.weight
encoders.kg.encoder.projection.0.weight
```

如果是 DataParallel 预训练，`scripts/pretrain.py` 保存时会自动去掉 `module.` 前缀。

## 9. 当前实现的限制

### 已实现

```text
1. KG 作为第五模态 kg 接入。
2. 使用 KANO 离线 element embedding。
3. 使用 KANO 离线 functional group embedding。
4. 从 SMILES 中提取元素。
5. 使用 SMARTS 检测官能团。
6. 使用 alias 做官能团名称规范化。
7. KGEncoder 使用 attention pooling。
8. KG 输出 [B,256] 后进入原 FusionModule。
9. 预训练 contrastive loss 自动包含 kg。
10. 微调 MSE loss 可反向更新 kg encoder。
11. results.csv attention 可记录 kg 模态权重。
```

### 未实现

```text
1. 没有使用 Neo4j。
2. 没有在线查询 KG。
3. 没有直接解析 OWL 参与训练。
4. 没有使用 rel2emb.pkl。
5. 没有动态图谱传播。
6. 没有 R-GCN。
7. 没有 GAT。
8. 没有 Graph Prompt。
9. 没有 KG-Augmented GNN。
10. 没有 KG 结构 loss。
11. 没有单独训练 KG embedding。
12. 没有把 KG 与 graph atom-level message passing 结合。
```

### 具体限制

```text
1. KG 实体是 set-like 去重结果，不统计元素出现次数或官能团出现次数。
2. 官能团只判断是否存在，不记录匹配次数。
3. 官能团 SMARTS 匹配质量完全依赖 functional_group_smarts.json。
4. KG embedding 默认可训练，训练后可能偏离原 KANO embedding。
5. 如果 --freeze_encoder 被使用，KGEncoder 也会被冻结。
6. Dataset 启用 KG 时，旧缓存样本会在内存中补 kg_entity_ids，但不会自动写回 .pt 缓存。
7. Dataloader 只使用 data.kg_entity_ids 重新生成 mask，没有使用 Dataset 中已有的 data.kg_mask。
8. 如果某个样本没有 KG 实体，KGEncoder 会收到全 padding mask，最后通过 projection 产生一个输出；这不是样本真实 KG 信息。
9. 当前 KGEncoder 没有暴露命令行参数控制 freeze_kg_embeddings。
10. 当前 KG 模态路径固定为 knowledge_graph/...，没有通过命令行配置。
```

### 后续可扩展方向

```text
1. 加入元素/官能团出现次数作为权重。
2. 使用 rel2emb.pkl 建模元素-元素关系。
3. 为 KGEncoder 增加 freeze_kg_embeddings 参数。
4. 将 KG 文件路径加入 argparse。
5. 让 Dataset 补完 KG 字段后可选择保存回缓存。
6. 加入 KG-only ablation 或 no-KG 对照实验。
7. 后续再考虑 OWL 图结构传播、R-GCN/GAT，但当前代码没有实现这些。
```
