# RUN.MD

本文是项目唯一运行手册。所有实际运行命令集中维护在这里。

默认工作目录：

```bash
cd /root/workspace/DeepLearning/Uni-Poly-Plus-master
```

## 1. API Key

Qwen：

```bash
export DASHSCOPE_API_KEY=你的百炼_API_KEY
export QWEN_MODEL=qwen-plus
```

DeepSeek：

```bash
export DEEPSEEK_API_KEY=你的_DEEPSEEK_API_KEY
export DEEPSEEK_MODEL=deepseek-chat
```

`--model` 可以覆盖环境变量中的模型名。API key 只从环境变量读取。

## 2. KG Pipeline 总览

目标输出：

```text
kg_work/features/kg_entity_mapping.csv
kg_work/features/kg_embedding.npy
```

端到端阶段：

```text
records -> mapping -> retrieval -> documents -> recall -> extraction -> validation -> links -> kg -> embedding
```

当前 `polymer_class_candidates.jsonl` 是 RepeatUnit 的文献检索候选，不是 verified polymer registry。当前字段为：

```json
{
  "repeat_unit_id": "...",
  "raw_smiles": "...",
  "canonical_smiles": "...",
  "aliases": ["..."],
  "confidence": 0.0
}
```

`aliases` 用于检索和召回候选文献，不应直接视为准确聚合物名称事实。

## 2.1 产物目录

`kg_work/` 是 KG pipeline 的运行产物目录，不是源码目录。常见内容包括：

```text
records.csv
repeat_units.csv
polymer_class_candidates.jsonl
retrieval/
chunks/source_chunks.jsonl
chunks/candidate_chunks.jsonl
extractions/raw/
extractions/validated/
links/dataset_links.jsonl
kg/nodes.csv
kg/edges.csv
kg/triples.tsv
kg/build_manifest.json
features/kg_entity_mapping.csv
features/kg_embedding.npy
features/embedding_manifest.json
logs/
state/
```

## 2.2 Linux 包装脚本

项目提供一个 Bash 包装脚本，位置为：

```bash
bash scripts/run_kg_online.sh
```

脚本会自动切到项目根目录，读取环境变量配置，并调用 `python scripts/kg/run_pilot.py`。常用覆盖方式：

```bash
INPUT=data/raw/smi_all.csv OUTPUT_DIR=kg_work MAX_REPEAT_UNITS=50 bash scripts/run_kg_online.sh
```

## 3. 一键运行 KG Pipeline

### 3.1 本地文献模式

本地文献放在：

```text
kg_work/documents/
```

支持 `.txt .md .html .xml .pdf`。

```bash
python scripts/kg/run_pilot.py   --input data/raw/smi_all.csv   --documents_dir kg_work/documents   --output_dir kg_work   --max_repeat_units 50   --provider qwen   --mapping_mode llm   --literature_mode local   --stop_after embedding
```

### 3.2 在线文献模式

在线模式使用 metadata-first retrieval：先查询公共 metadata API，再按 title/abstract/OA/alias 命中做轻量筛选，只下载 selected 且允许下载的 OA full text；无法下载时写 metadata-only `.txt`。

```bash
python scripts/kg/run_pilot.py   --input data/raw/smi_all.csv   --output_dir kg_work   --max_repeat_units 50   --provider qwen   --mapping_mode llm   --literature_mode online   --sources crossref,openalex,semantic_scholar,europe_pmc,arxiv   --max_queries_per_repeat_unit 5   --max_results_per_query 10   --max_articles_per_repeat_unit 5   --max_downloads_per_repeat_unit 3   --max_total_downloads 100   --min_article_score 0.4   --require_oa_for_download true   --stop_after embedding
```

### 3.3 本地 + 在线混合模式

```bash
python scripts/kg/run_pilot.py   --input data/raw/smi_all.csv   --documents_dir kg_work/documents   --output_dir kg_work   --max_repeat_units 50   --provider qwen   --mapping_mode llm   --literature_mode both   --stop_after embedding
```

### 3.4 Mock 测试

不调用真实 LLM API：

```bash
python scripts/kg/run_pilot.py   --input data/raw/smi_all.csv   --documents_dir kg_work/documents   --output_dir kg_work_mock   --max_repeat_units 5   --provider qwen   --mapping_mode llm   --mock_llm   --literature_mode local   --stop_after embedding
```

mock 输出只用于 smoke test，不应作为正式实验结果。

## 4. run_pilot.py 参数

当前命令可用参数以 `python scripts/kg/run_pilot.py --help` 为准。常用参数：

| 参数 | 作用 |
|---|---|
| `--input` | 原始 CSV，通常是 `data/raw/smi_all.csv`。 |
| `--documents_dir` | 本地和在线下载文献目录；不填时默认 `<output_dir>/documents`。 |
| `--output_dir` | KG pipeline 输出目录。 |
| `--max_repeat_units` | 限制处理的去重 RU 数量。 |
| `--provider` | `qwen` 或 `deepseek`。 |
| `--model` | 覆盖 provider 模型名。 |
| `--mapping_mode` | 当前只支持 `llm`。 |
| `--mock_llm` | mock mapping 和 mock extraction，不需要 API key。 |
| `--literature_mode` | `local` / `online` / `both`。 |
| `--sources` | 在线 metadata 来源列表。 |
| `--max_queries_per_repeat_unit` | 每个 RU 最多 query 数。 |
| `--max_results_per_query` | 每个 query 每个 source 返回结果数。 |
| `--max_articles_per_repeat_unit` | 每个 RU 最多保留 selected articles。 |
| `--max_downloads_per_repeat_unit` | 每个 RU 最多下载全文数。 |
| `--max_total_downloads` | 全局最多下载全文数。 |
| `--min_article_score` | article 轻量筛选分数阈值。 |
| `--require_oa_for_download` | `true/false`，是否只下载 OA full text。 |
| `--metadata_only` | 只保存 metadata-only txt，不下载全文。 |
| `--stop_after` | 停在指定阶段：`records/mapping/retrieval/documents/recall/extraction/validation/links/kg/embedding`。 |
| `--resume` | 已有阶段产物存在时跳过。 |
| `--max_candidates` | LLM extraction 最多处理多少 candidate chunks。 |
| `--embedding_epochs` | TransE 训练轮数。 |
| `--embedding_dim` | KG embedding 维度。 |
| `--overwrite_mapping` | 覆盖已有 mapping 输出。 |
| `--timeout` | API / 网络请求超时秒数。 |
| `--max_retries` | LLM API 重试次数。 |
| `--sleep_seconds` | 请求间隔秒数。 |

## 5. 分步运行

### 5.1 records / repeat units

```bash
python scripts/kg/prepare_records.py   --input data/raw/smi_all.csv   --output_dir kg_work   --max_repeat_units 50
```

### 5.2 LLM mapping

```bash
python scripts/kg/generate_llm_mapping.py   --repeat_units kg_work/repeat_units.csv   --output_dir kg_work   --provider qwen   --max_repeat_units 50   --overwrite
```

mock：

```bash
python scripts/kg/generate_llm_mapping.py   --repeat_units kg_work/repeat_units.csv   --output_dir kg_work_mock   --provider qwen   --max_repeat_units 5   --mock_response   --overwrite
```

### 5.3 online retrieval

```bash
python scripts/kg/retrieve_documents.py   --repeat_units kg_work/repeat_units.csv   --mapping_candidates kg_work/polymer_class_candidates.jsonl   --output_dir kg_work/retrieval   --documents_dir kg_work/documents   --sources crossref,openalex,semantic_scholar,europe_pmc,arxiv   --max_repeat_units 50   --max_queries_per_repeat_unit 5   --max_results_per_query 10   --max_articles_per_repeat_unit 5   --max_downloads_per_repeat_unit 3   --max_total_downloads 100   --min_article_score 0.4   --require_oa_for_download true
```

### 5.4 parse documents

```bash
python scripts/kg/parse_documents.py   --input_dir kg_work/documents   --output kg_work/chunks/source_chunks.jsonl
```

### 5.5 recall chunks

```bash
python scripts/kg/recall_chunks.py   --source_chunks kg_work/chunks/source_chunks.jsonl   --polymer_class_candidates kg_work/polymer_class_candidates.jsonl   --retrieval_dir kg_work/retrieval   --output kg_work/chunks/candidate_chunks.jsonl
```

### 5.6 LLM extraction

```bash
python scripts/kg/extract_articles.py   --candidate_chunks kg_work/chunks/candidate_chunks.jsonl   --source_chunks kg_work/chunks/source_chunks.jsonl   --output_dir kg_work/extractions/raw   --provider qwen   --max_candidates 20
```

### 5.7 validation

```bash
python scripts/kg/validate_articles.py   --raw_dir kg_work/extractions/raw   --source_chunks kg_work/chunks/source_chunks.jsonl   --aggregated_dir kg_work/extractions/aggregated   --output_dir kg_work/extractions/validated
```

### 5.8 dataset links

```bash
python scripts/kg/build_dataset_links.py   --validated_dir kg_work/extractions/validated   --repeat_units kg_work/repeat_units.csv   --polymer_class_candidates kg_work/polymer_class_candidates.jsonl   --output kg_work/links/dataset_links.jsonl
```

### 5.9 build KG

```bash
python scripts/kg/build_kg.py   --records kg_work/records.csv   --repeat_units kg_work/repeat_units.csv   --validated_dir kg_work/extractions/validated   --dataset_links kg_work/links/dataset_links.jsonl   --polymer_class_candidates kg_work/polymer_class_candidates.jsonl   --source_chunks kg_work/chunks/source_chunks.jsonl   --output_dir kg_work/kg   --graph strict
```

### 5.10 train KG embedding

```bash
python scripts/kg/train_kg_embedding.py   --triples kg_work/kg/triples.tsv   --nodes kg_work/kg/nodes.csv   --edges kg_work/kg/edges.csv   --repeat_units kg_work/repeat_units.csv   --output_dir kg_work/features   --embedding_dim 128   --model transe   --graph strict   --epochs 5
```

## 6. 检查 KG embedding

```bash
python - <<'PY'
import json
import numpy as np
from pathlib import Path

emb = np.load('kg_work/features/kg_embedding.npy')
manifest = json.loads(Path('kg_work/features/embedding_manifest.json').read_text())
print('embedding shape:', emb.shape)
print('dtype:', emb.dtype)
print('test_only:', manifest.get('test_only'))
print('has_literature_links:', manifest.get('has_literature_links'))
PY
```

`test_only=true` 表示当前 embedding 可用于代码 smoke test，但缺少 validated literature links，不建议作为正式实验结果。

## 7. 启用 KG 模态训练

### 7.1 pretrain with kg

```bash
python scripts/pretrain.py   --dataset_name smi_all   --modalities smiles graph fp geom kg   --geometry_encoder schnet   --kg_mapping_path kg_work/features/kg_entity_mapping.csv   --kg_embedding_path kg_work/features/kg_embedding.npy   --kg_embedding_dim 128   --kg_projection_dim 256   --kg_freeze_embedding true
```

### 7.2 train with kg

```bash
python scripts/train.py   --modalities smiles graph fp geom kg   --geometry_encoder schnet   --tasks tg eat eea egb egc ei eps nc xc   --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth   --kg_mapping_path kg_work/features/kg_entity_mapping.csv   --kg_embedding_path kg_work/features/kg_embedding.npy   --kg_embedding_dim 128   --kg_projection_dim 256   --kg_freeze_embedding true
```

`--kg_freeze_embedding true` 冻结 KG embedding table；`false` 允许微调。

## 8. 常见问题

缺少 API key：

```text
ERROR: Missing API key: set DASHSCOPE_API_KEY in the environment
```

已有 mapping 文件：

```text
FileExistsError: kg_work/polymer_class_candidates.jsonl exists; pass --overwrite to replace it
```

解决：使用 `--overwrite` 或 `run_pilot.py --overwrite_mapping`。

没有 candidate chunks：检查 `kg_work/chunks/source_chunks.jsonl`、`kg_work/chunks/candidate_chunks.jsonl`、mapping aliases 是否为空。

在线 retrieval 卡住：通常是外网、代理、API rate limit 或 source 请求超时。可降低 `--max_repeat_units`，或使用 `--metadata_only` / `--stop_after retrieval` 调试。

KG embedding 维度不一致：`--kg_embedding_dim` 必须等于 `kg_embedding.npy.shape[1]`。


## 9. 模型目录说明

当前不重命名模型目录，原因是 `scripts/train.py`、`scripts/pretrain.py`、`src/modules/geom.py` 和已有 shell 脚本已经引用默认路径。保守约定如下：

```text
pretrained_models/   外部或已训练权重，例如 encoders 和 saved_pretrained_model.pth
pretrain_models/     预训练过程输出、缓存或中间产物
```
