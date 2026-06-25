# AGENT.MD

本文档约束代码协作者和自动化代码代理。运行教程只看 [RUN.MD](RUN.MD)。

## 语言规范

- 与用户沟通、项目说明和新增 Markdown 文档默认使用中文。
- 代码标识符、CLI 参数、文件名、JSON 字段和必要的英文技术术语保持原文。
- 如果引用外部英文资料，可以保留英文原文，但需要用中文解释其含义。

## 项目边界

- 当前支持模态：`smiles graph fp geom`。
- Graph 后端固定为 GIN，不再存在 `pyg_periogt` 或 n-mer 多视图逻辑。
- 不要恢复 LLM 文献抽取、KG pipeline、KG embedding、PerioGT/n-mer 多视图或 `kg` 训练模态，除非用户明确提出新的实现需求。
- 默认四模态行为必须保持可用。
- `data/raw/*.csv` 中的监督标签只用于训练和评估，不要作为模型输入特征。
- 不要提交 API key、authorization header 或任何 secret。
- 不要在未获用户许可时启动长时间 GPU 训练、下载大模型或批量联网任务。
- 不要无故重命名 `pretrained_models/` 和 `pretrain_models/`；当前代码已有默认路径引用。

## 高风险文件

- `src/dataset/`：数据字段变化会影响 pretrain/train/dataloader。
- `src/modules/uni_encoder.py`：多模态融合、projection、attention 维度。
- `scripts/train.py`、`scripts/pretrain.py`：训练入口和 checkpoint 兼容。

## 验证要求

按改动范围选择最小有效验证：

```bash
python -m py_compile src/dataset/*.py src/modules/*.py scripts/train.py scripts/pretrain.py
python scripts/pretrain.py --help
python scripts/train.py --help
bash -n scripts/run.sh scripts/run_pretrain.sh scripts/run_train.sh
```

## Git 与产物

- 不要回滚用户未说明的本地改动。
- 不要删除 `results/`、`saved_models/`、`pretrained_models/` 等产物，除非用户明确要求。
- 不要提交 `__pycache__/`、`.pyc`、大权重、日志和临时下载文件。
- 修改文档时保持 [README.MD](README.MD)、[RUN.MD](RUN.MD)、[AGENT.MD](AGENT.MD) 职责清晰，避免重复和冲突。
