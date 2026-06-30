# AGENT.MD

本文档是当前 Uni-Poly-Plus 项目的代码代理约束。所有后续代码修改、实验脚本修改和文档更新都应遵守这里的规则。

## 1. 沟通规则

- 默认使用中文回复。
- 回答要直接说明结论、原因和可执行命令。
- 不要只给计划；如果用户要求实现，应直接修改代码并运行可行验证。
- 如果无法运行完整训练，必须明确说明未运行原因。

## 2. 当前项目主线

当前项目是聚合物性质预测的多模态模型，正式主线为：

```text
SMILES + Graph + FP + Geom
```

当前重点实现：

- `smiles`：SMILES tokenizer / encoder。
- `graph`：二维拓扑图，当前推荐 `star_linking`。
- `fp`：Morgan fingerprint。
- `geom`：3D 几何模态，当前推荐 `PaiNN`，也支持 `SchNet`。

当前推荐运行路线：

```text
Stage 1: Graph + Geom 单模态预训练
Stage 2: 多模态 alignment 预训练
Stage 3: 下游 5-fold supervised training
```

入口脚本：

```bash
bash scripts/run.sh
```

只跑预训练：

```bash
bash scripts/run_pretrain.sh
```

## 3. Graph 模块约束

Graph 输入目前只应支持：

```text
repeat_unit
star_linking
```

`star_linking` 语义：

1. 删除两个 `*` attachment atoms。
2. 连接两个 attachment neighbor atoms。
3. 只用于 `graph` 模态。
4. 如果 `*` 数量不是 2、连接键不一致或 RDKit sanitize 失败，应 fallback 到 `repeat_unit` graph。

Graph 当前还包含 backbone / attachment 特征：

- `is_backbone_atom`
- `is_attachment_neighbor`
- `is_side_chain_atom`
- `is_star_linking_edge`

修改 Graph 特征维度时必须同步：

- `src/dataset/graph_data.py`
- `src/modules/graph.py`
- feature cache tag
- 需要提醒用户使用 `--rebuild_feature_cache`

## 4. Geom 模块约束

Geom 不默认使用 star-linked closed graph。

原因：

```text
star-linking 是拓扑处理，不等价于真实 3D 闭环构象。
SchNet / PaiNN 使用距离和方向信息，错误闭环会污染几何表征。
```

Geom 当前推荐：

- 使用 star-substitution capped repeat unit。
- 多构象生成。
- MMFF 优先，UFF fallback。
- 3D 失败时 fallback 到 2D。
- 训练时可随机选择 conformer。

修改 Geom 时要关注：

- `src/dataset/geom_data.py`
- `src/dataset/dataloader.py`
- `src/modules/geom.py`
- PaiNN 是否出现 `nan`
- 坐标、距离、batch index 是否一致

## 5. 多阶段预训练约束

`scripts/pretrain.py` 当前支持：

```bash
--pretrain_stage graph_geom
--pretrain_stage alignment
--pretrain_stage joint
```

语义：

- `graph_geom`：只训练 Graph / Geom 辅助任务，不做 contrastive alignment。
- `alignment`：加载上一阶段权重，做多模态 contrastive alignment，并保留小权重辅助 loss。
- `joint`：兼容旧逻辑，直接把所有启用 loss 加起来。

推荐不要把 `joint` 作为最终主线，只作为 baseline。

修改 loss 时必须明确：

- 该 loss 属于 Graph、Geom、Alignment 还是 downstream supervised。
- 是否会和其他 loss 共享 encoder。
- loss 数值尺度是否可能主导训练。
- 是否需要在 `run.sh` 中单独暴露权重。

## 6. 数据和缓存约束

原始数据：

```text
data/raw/smi_all.csv
data/raw/smi_<task>.csv
```

处理后缓存默认位于：

```text
data/processed/Schnet/
```

SMILES 级别 feature cache 是当前重要优化，不要随意删除。

如果改动以下内容，必须重建缓存：

- Graph 输入类型。
- Graph 节点/边特征维度。
- Geom 构象生成逻辑。
- SMILES token 最大长度。
- feature cache tag。

需要提醒用户加：

```bash
--rebuild_feature_cache
```

## 7. 训练和验证要求

任何涉及整体流程的改动，都必须同步更新 `PIPELINE.MD`，包括：

- 数据输入 / 输出路径。
- feature cache。
- Graph 构造。
- Geom 构象。
- 预训练阶段和 loss。
- 下游训练流程。
- `scripts/run.sh` / `scripts/run_pretrain.sh` / `scripts/run_train.sh`。

修改 Python 代码后至少运行：

```bash
python -m py_compile scripts/pretrain.py scripts/train.py src/dataset/*.py src/modules/*.py src/utils.py
```

修改 CLI 后检查：

```bash
python scripts/pretrain.py --help
python scripts/train.py --help
```

修改 shell 脚本后检查：

```bash
bash -n scripts/run.sh
bash -n scripts/run_pretrain.sh
bash -n scripts/run_train.sh
```

如环境缺少 RDKit / PyG / CUDA，不能伪造测试结果，应明确说明。

## 8. 文件修改边界

优先修改已有文件，不要新增复杂目录或框架。

常见职责：

- `src/dataset/graph_data.py`：Graph 构造、star-linking、节点/边特征。
- `src/dataset/geom_data.py`：3D 构象、star-substitution、fallback。
- `src/dataset/dataset.py`：数据集、缓存、cache tag。
- `src/dataset/dataloader.py`：batch collate、多构象选择。
- `src/modules/graph.py`：GINE / Graph encoder。
- `src/modules/geom.py`：SchNet / PaiNN。
- `src/modules/uni_encoder.py`：多模态 encoder 和融合。
- `scripts/pretrain.py`：预训练阶段和 loss。
- `scripts/train.py`：下游训练和 5-fold。
- `scripts/run.sh`：完整串行运行入口。

不要恢复旧 KG pipeline、旧 TEXT 模态、旧 dimer/n-mer 逻辑，除非用户明确要求。

## 9. Git 和产物约束

不要提交运行产物：

```text
logs/
plots/
results/
saved_models*/
pretrained_models/*.pth
data/processed/
kg_work/
```

不要提交 API key、token、私有路径中的 secret。

如果工作区已有用户改动：

- 不要 `git reset --hard`。
- 不要随意 revert 用户改动。
- 只改和当前任务直接相关的文件。

## 10. 文档约束

当前正式文档：

- `README.MD`：项目入口和运行说明。
- `PIPELINE.MD`：整体数据流、模型流、预训练和训练流程的唯一详细维护文档。
- `TODO.MD`：后续开发计划和研究方案。
- `PerioGT.MD`：PerioGT 代码/论文分析。
- `IDEAS.MD`：临时研究想法。
- `AGENT.MD`：当前代码代理约束。

`PIPELINE.MD` 是当前项目流程的唯一详细版本。以下改动必须同步更新 `PIPELINE.MD`：

- 新增或删除模态。
- 修改 Graph / Geom / FP / SMILES 处理逻辑。
- 修改缓存路径、缓存内容或 cache tag。
- 修改预训练阶段、loss、checkpoint 加载或保存路径。
- 修改下游训练 fold、validation、任务列表、结果输出路径。
- 修改任何 `scripts/run*.sh` 运行脚本。

如果修改运行命令，应优先同步 `README.MD`。

如果修改未来开发计划，应同步 `TODO.MD`。

不要让 `README.MD`、`PIPELINE.MD`、`TODO.MD`、`AGENT.MD` 出现互相冲突的运行命令或流程描述。

