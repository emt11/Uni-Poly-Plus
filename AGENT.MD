# AGENT.MD

本文档约束代码协作者和自动化代码代理。运行教程只看 [RUN.MD](RUN.MD)，KG 语义设计只看 [LLM_KG.MD](LLM_KG.MD)。

## 项目边界

- 当前支持模态：`smiles graph fp geom kg`。
- `kg` 是新的 Polymer KG embedding 模态，不是旧 KG，也不是 TEXT 模态。
- 默认四模态行为必须保持可用；只有显式启用 `kg` 时才读取 KG 文件。
- `data/raw/smi_all.csv` 中的 `val` 是监督标签，不得进入 LLM prompt、retrieval query、mapping、KG triples 或 embedding。
- `kg_work/` 是运行产物目录，不是源码目录；不要把大文件、日志、下载文献或 embedding 当作源码提交。

## KG 模块边界

核心实现集中在 7 个中等粒度模块：

```text
src/kg_pipeline/io_utils.py             通用 IO、hash/id、leakage 检查、数值归一化
src/kg_pipeline/dataset.py              records.csv / repeat_units.csv 与 RU canonicalize
src/kg_pipeline/literature.py           online retrieval、本地文献解析、chunk recall
src/kg_pipeline/llm_extract.py          Qwen/DeepSeek、RU alias mapping、文献抽取
src/kg_pipeline/article_validation.py   schema v2、aggregation、validation、review queue
src/kg_pipeline/kg_build.py             dataset linking、KG nodes/edges/triples/manifest
src/kg_pipeline/kg_embedding.py         TransE baseline 与 kg_embedding 导出
```

`scripts/kg/*.py` 只做 CLI 参数解析和流程编排。旧的细粒度 `src/kg_pipeline/*.py` wrapper 已删除，不要重新创建这些旧入口。

## 代码修改规则

- 优先保持 CLI 参数、输出文件名和数据格式兼容。
- 不要恢复旧 mapping、旧 `entity_aliases.csv` 作为核心路径、旧 KG 或 TEXT 模态。
- 当前 `polymer_class_candidates.jsonl` 是检索候选，不是 verified registry。
- 不要提交 API key、authorization header 或任何 secret。
- 不要在未获用户许可时启动长时间 GPU 训练、下载大模型或批量联网任务。
- 不要无故重命名 `pretrained_models/` 和 `pretrain_models/`；当前代码已有默认路径引用。

## 高风险文件

- `src/dataset/`：数据字段变化会影响 pretrain/train/dataloader。
- `src/modules/uni_encoder.py`：多模态融合、KG projection、attention 维度。
- `scripts/train.py`、`scripts/pretrain.py`：训练入口和 checkpoint 兼容。
- `src/kg_pipeline/llm_extract.py`：LLM provider、prompt、mapping、文献抽取。
- `src/kg_pipeline/literature.py`：在线检索、下载、metadata-only 文档、chunk recall。
- `src/kg_pipeline/kg_build.py`、`src/kg_pipeline/kg_embedding.py`：KG consistency 和 embedding 输出。

## 验证要求

按改动范围选择最小有效验证：

```bash
python -m compileall src scripts
python scripts/kg/run_pilot.py --help
python scripts/kg/generate_llm_mapping.py --help
python scripts/kg/extract_articles.py --help
python scripts/kg/train_kg_embedding.py --help
python scripts/pretrain.py --help
python scripts/train.py --help
bash -n scripts/run_kg_online.sh
```

KG smoke test 使用 `--mock_llm --max_repeat_units 5`，不要直接跑大规模真实 API。

## Git 与产物

- 不要回滚用户未说明的本地改动。
- 不要删除 `kg_work/`、`results/`、`saved_models/` 等产物，除非用户明确要求。
- 不要提交 `__pycache__/`、`.pyc`、大权重、日志和临时下载文件。
- 修改文档时保持 [README.MD](README.MD)、[RUN.MD](RUN.MD)、[LLM_KG.MD](LLM_KG.MD) 职责清晰，避免重复和冲突。
