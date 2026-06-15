# AGENT.md

本文件为参与本项目的代码代理和协作者提供约定。除非用户明确要求，所有改动都应遵循这里的项目边界、运行方式和验证策略。

## 项目定位

Uni-Poly 是用于聚合物性质预测的多模态深度学习项目，核心流程包括：

- 多模态数据读取与特征构建：`src/dataset/`
- 模型结构与模态融合：`src/modules/`
- 多模态对比预训练：`scripts/pretrain.py`
- 下游性质预测训练：`scripts/train.py`
- 注意力权重可视化：`scripts/plot_attention_heatmap.py`

项目包含较多数据、模型权重、日志和结果文件。修改时优先保持代码、数据格式和训练入口兼容。

## 关键目录

- `src/dataset/`：数据集、图数据、几何数据、DataLoader。改动这里通常会影响预训练和下游训练。
- `src/modules/`：Uni-Poly 模型、图编码和几何编码。改动这里需要重点检查 checkpoint 加载和模态维度。
- `scripts/`：命令行入口。新增参数时同步更新 README 和本文件。
- `data/raw/`：原始性质数据，文件名约定为 `smi_<task>.csv`。
- `pretrained_models/`、`saved_models/`：模型权重目录。默认不要重写、删除或提交大规模权重变更。
- `logs/`、`plots/`、`results/`：运行产物。除非任务明确要求，不要把历史结果当作源码重构对象。
- `caption_generation/` 与 `knowledge_graph/`：保留的历史资产，不参与当前训练运行时。新的 Polymer KG 会在后续版本重新接入。

## 环境与依赖

依赖声明在 `requirements.txt`。主要依赖包括 PyTorch、PyTorch Geometric、RDKit、Transformers、scikit-learn、matplotlib、seaborn 和 jupyter。

安装示例：

```bash
pip install -r requirements.txt
```

注意：`torch-geometric`、`torch-scatter`、`torch-sparse`、`torch-cluster` 和 `rdkit` 对 Python、CUDA、PyTorch 版本敏感。不要在未确认环境的情况下随意升级这些依赖。

## 常用命令

预训练示例：

```bash
bash scripts/run_pretrain.sh
```

等价核心命令：

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet
```

下游训练示例：

```bash
bash scripts/run_train.sh
```

等价核心命令：

```bash
python scripts/train.py \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --tasks tg eat eea egb egc ei eps nc xc \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth
```

注意力热力图：

```bash
python scripts/plot_attention_heatmap.py \
  --results_csv ./results/results.csv \
  --output ./results/attention_heatmap.png
```

`scripts/run_pretrain.sh` 和 `scripts/run_train.sh` 默认包含 `CUDA_VISIBLE_DEVICES=1`，并使用 `nohup ... &` 后台运行。自动化代理不要在没有用户许可的情况下启动长时间 GPU 训练任务。

## 开发约定

- 优先保持现有模块结构，不引入新的训练框架或配置系统，除非用户明确要求。
- 入口脚本参数使用 `argparse`，新增训练参数时保持 CLI 向后兼容。
- 数据集命名遵循 `smi_<task>`，对应原始 CSV 位于 `data/raw/smi_<task>.csv`。
- 当前支持的模态只有：`smiles`、`graph`、`fp`、`geom`。传入 `text` 或旧 `kg` 必须在 CLI 和模型入口报错。
- 几何编码器当前使用 `painn` 或 `schnet`，新增后端时需要同步检查数据处理和模型初始化。
- checkpoint 加载当前允许 `strict=False`。旧 checkpoint 中的文本或旧 KG encoder 参数会被忽略；修改模型 state dict key 时要说明兼容性。
- 保持 README 与脚本默认行为一致。当前根目录说明文件名是 `README.MD`，如需平台渲染更稳定，可改名为 `README.md`，但应确认不会影响用户已有引用。
- 不要提交或重写 `__pycache__/`、大型 `.pth`、`.pt`、`.bin`、`.safetensors` 文件，除非任务目标就是更新模型或数据资产。

## 验证策略

根据改动范围选择尽量小但有效的验证：

- 文档改动：检查 Markdown 是否能打开，必要时确认命令与脚本参数一致。
- CLI 参数或脚本改动：运行 `python scripts/<name>.py --help`。
- 数据处理改动：优先用小任务或少量样本验证 `UniDataset` 能实例化并产出 batch。
- 模型结构改动：至少验证一次前向传播，检查各模态张量维度和 attention 权重形状。
- 训练循环改动：优先用较小 epoch、单任务、较小 batch 做 smoke test，避免直接跑完整 5 折训练。
- 可视化改动：使用已有 `results/results.csv` 生成一张测试图，并确认输出文件存在。

如果环境缺少 GPU、PyG、RDKit 或本地权重，应在最终说明中明确指出未能运行的验证项和原因。

## 高风险区域

- `src/modules/uni_encoder.py`：多模态融合和 attention 权重通常集中在这里，维度错误会影响所有任务。
- `src/dataset/dataset.py`：数据字段变化会传播到预训练、训练、可视化和 checkpoint 兼容性。
- `src/dataset/geom_data.py`、`src/modules/geom.py`：与 `painn`/`schnet` 后端和三维数据格式强相关。
- 未来重新接入 Polymer KG 时，不得复用已移除的旧 KG 运行时假设；需要重新定义 embedding 文件和 RepeatUnit 映射接口。
- `scripts/train.py`：负责 5 折交叉验证、结果 CSV、模型保存和 attention 汇总，改动后要检查输出列是否仍被可视化脚本支持。

## 输出与产物

常见运行产物包括：

- `pretrained_models/saved_pretrained_model.pth`
- `saved_models/<task>/UniEncoderAttention_best.pth`
- `results/results.csv`
- `results/attention_heatmap.png`
- `plots/pretrain/loss_data.json`
- `plots/pretrain/loss_curve.png`
- `logs/pretrain.log`
- `logs/train.log`

除非用户要求复现实验或更新结果，不要因为代码修改而顺手覆盖这些产物。

## 协作提示

- 在编辑前先确认是否存在用户未说明的本地改动。
- 若需要安装依赖、下载模型、联网访问或启动长时间训练，先征得用户许可。
- 遇到数据或权重缺失时，优先给出明确路径和缺失文件名，不要伪造运行结果。
- 回复用户时优先说明实际改了什么、验证了什么、还有什么没有验证。
