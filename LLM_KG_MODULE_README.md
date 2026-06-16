# LLM + Polymer KG 模块说明

本文档说明当前项目中新增加的 LLM + Polymer KG 模块如何工作、如何运行、输入输出是什么，以及如何把 KG embedding 作为 Uni-Poly 的第五模态 `kg` 接入。

本文面向项目协作者和后续代码代理。内容以当前代码为准，覆盖 `src/kg_pipeline/`、`scripts/kg/`，以及 Uni-Poly 中可选 `kg` 模态的接入代码。

状态标记含义：

- `implemented`：已有代码实现，并接入 CLI 或运行路径。
- `partially implemented`：已有 MVP，但没有完整覆盖计划文档中的目标。
- `not implemented yet`：计划中有，但当前代码没有实现。
- `temporary`：临时实现，不是最终科学映射或正式产物。
- `test-only`：可用于 smoke test，不应当视为生产级 KG 结果。

## 1. 模块定位

目标流程：

```text
data/raw/smi_all.csv
-> DatasetRecord / RepeatUnit
-> 本地文献文档与 candidate chunks
-> LLM 按 JSON schema v2.0 抽取
-> LiteratureSample / Evidence / Article / Assertion / Measurement / Event
-> dataset_links
-> Polymer KG nodes / edges / triples
-> TransE KG embedding
-> kg_entity_mapping.csv / kg_embedding.npy
-> Uni-Poly 第五模态 kg
```

核心边界：

- `LiteratureSample` 不等于 `DatasetRecord`。
- 文献事实不能直接写成当前数据行的真值。
- `val` 不能进入 KG 构建、LLM prompt、检索 query、PolymerClass 映射、dataset linking、triples 或 embedding。
- 当前不生成 `enriched_records.csv`。
- 当前不生成独立的 `numeric_descriptor_matrix`。
- KG embedding 文件是新增 `kg` 模态的输入。
- 新 `kg` 模态不是已移除的旧 KG 实现，也没有恢复 TEXT 模态。

## 2. 当前实现状态

| 阶段 | 状态 | 脚本 | 主要模块 | 输入 | 输出 | 当前限制 |
|---|---|---|---|---|---|---|
| Stage 0 records / repeat units | `implemented` | `scripts/kg/prepare_records.py` | `records.py`, `repeat_units.py` | `data/raw/smi_all.csv` | `kg_work/records.csv`, `kg_work/repeat_units.csv` | RDKit canonicalization 是 best-effort。RDKit 无法解析时保留 raw SMILES，并标记 `valid_rdkit_parse=false`。 |
| Stage 1 PolymerClass / alias mapping | `implemented`, `temporary` | `map_polymer_classes.py`, `generate_llm_mapping.py` | `polymer_classes.py`, `extraction.py` | `repeat_units.csv` | `polymer_class_candidates.jsonl`, `entity_aliases.csv` | `rule_stub` 是启发式。LLM mapping 是临时测试映射，不是正式 PolymerClass registry。 |
| Stage 2 document parsing | `partially implemented` | `parse_documents.py` | `document_parser.py` | 本地 `kg_work/documents/` | `chunks/source_chunks.jsonl` | 只支持本地文件解析；没有在线文献检索。PDF 依赖 `pypdf`；HTML/XML 解析是简单 BeautifulSoup 抽取。 |
| Stage 3 candidate chunk recall | `partially implemented` | `recall_chunks.py` | `chunk_recall.py` | `source_chunks.jsonl` | `candidate_chunks.jsonl` | 已实现 keyword、BM25-like score、section prior、table keyword score、邻域上下文。dense retrieval、reranker、真正的 sample-conditioned second recall 尚未实现。 |
| Stage 4 LLM extraction | `implemented`，provider 调用未在本文档任务中测试 | `extract_articles.py` | `extraction.py` | `candidate_chunks.jsonl`, `source_chunks.jsonl` | per-candidate `.json`, `calls.jsonl`, `extractions.jsonl` | 需要 Qwen 或 DeepSeek API key。本文档任务没有运行 LLM extraction。 |
| Stage 5 aggregation / validation | `partially implemented` | `validate_articles.py` | `aggregation.py`, `validation.py`, `schemas/v2.py` | raw extraction JSON，source chunks | `extractions/aggregated/`, `extractions/validated/`, `review_queue.json` | 聚合按 sample label、polymer name、polymer class 或首个 alias 分组。校验覆盖顶层 schema、证据句定位、枚举、正数、PDI、DP，并拒绝无效事实进入 KG。`schema_v2.json` 存在，但当前 `validate_schema_v2` 没有真正调用 JSON Schema 引擎。 |
| Stage 6 dataset links | `partially implemented` | `build_dataset_links.py` | `linking.py` | validated JSON、repeat units、mapping candidates | `links/dataset_links.jsonl` | 支持 explicit repeat-unit SMILES、PolymerClass candidate、alias 匹配。functional-group similarity 和 family-level matching 尚未实现。 |
| Stage 7 KG builder | `implemented` for MVP | `build_kg.py` | `graph_builder.py`, `numeric_features.py` | records、repeat units、validated JSON、dataset links、source chunks | `kg/nodes.csv`, `kg/edges.csv`, `kg/triples.tsv`, `kg/entity_aliases.csv`, `kg/build_manifest.json` | 输入存在时构建 dataset、mapping、literature、fact、enum、numeric-bin、provenance 节点。SourceChunk 保留在 KG 文件中，但默认不进入 TransE triples。 |
| Stage 8 TransE embedding | `implemented`，baseline | `train_kg_embedding.py` | `embedding.py` | triples、nodes、edges、repeat units | `features/kg_entity_mapping.csv`, `features/kg_embedding.npy`, `features/embedding_manifest.json` | 只实现 TransE。没有 DistMult、ComplEx、RotatE、R-GCN、HGT。没有文献链接时 manifest 标记 `test_only=true`。 |
| Stage 9 Uni-Poly `kg` 模态接入 | `implemented` | `scripts/train.py`, `scripts/pretrain.py` | `dataset.py`, `dataloader.py`, `uni_encoder.py` | 显式启用 `kg` 时读取 `kg_entity_mapping.csv`, `kg_embedding.npy` | batch KG indices / masks，模型内 KG projection | `kg` 是 opt-in。默认仍是 `smiles graph fp geom`。CLI 中 `kg_freeze_embedding` 当前只接受 `true`。 |
| Stage 10 pilot runner | `implemented` for local MVP | `run_pilot.py` | 多模块编排 | 输入 CSV、本地 documents dir | pipeline 产物和 `pilot_report.json` | 使用 `rule_stub` mapping。只有存在 candidate chunks 且 API key 可用时才会调用 extraction。 |

## 3. 目录结构

### `kg_work/`

`kg_work/` 是运行产物目录。它可能包含大文件和中间缓存，不应当作为源码目录处理，也不应默认提交大文件或长期缓存。

重要文件：

- `kg_work/records.csv`：不含标签的 DatasetRecord 表。字段包括 `record_id`、`row_index`、`smiles`、`prop`、`repeat_unit_id`。不得包含 `val`。
- `kg_work/repeat_units.csv`：去重 RepeatUnit 表，包含 raw/canonical SMILES、structure hash、RDKit parse flag、formula、repeat-unit 分子量估计、记录数、props、表示类型和 origin hint。
- `kg_work/polymer_class_candidates.jsonl`：RepeatUnit 到 PolymerClass 的候选映射。当前 `rule_stub` 或 LLM 输出都是临时/测试用途。
- `kg_work/entity_aliases.csv`：由 mapping candidates 生成的 alias 表。
- `kg_work/documents/`：本地文献输入目录。可放 `.txt`、`.md`、`.html`、`.xml`、`.pdf`。
- `kg_work/chunks/source_chunks.jsonl`：从本地文献解析出的 SourceChunk。
- `kg_work/chunks/candidate_chunks.jsonl`：按事实类型召回的候选 chunks。
- `kg_work/extractions/raw/`：LLM 原始抽取结果目录。包含 per-candidate JSON、`calls.jsonl`、`extractions.jsonl`。
- `kg_work/extractions/aggregated/`：文档级聚合结果。
- `kg_work/extractions/validated/`：validated article JSON 和 `review_queue.json`。
- `kg_work/links/dataset_links.jsonl`：LiteratureSample 到 RepeatUnit 的 typed links。
- `kg_work/kg/nodes.csv`：KG 节点表。
- `kg_work/kg/edges.csv`：KG 边表。
- `kg_work/kg/triples.tsv`：KGE 使用的三元组，格式为 `head_id<TAB>relation_type<TAB>tail_id`。
- `kg_work/kg/entity_aliases.csv`：KG 层面的 PolymerClass / LiteratureSample aliases。
- `kg_work/kg/build_manifest.json`：构图 manifest，记录 schema、graph variant、计数和输入 hash。
- `kg_work/features/kg_entity_mapping.csv`：RepeatUnit 到 embedding 行号的映射。
- `kg_work/features/kg_embedding.npy`：`float32` embedding 矩阵。
- `kg_work/features/embedding_manifest.json`：embedding manifest，记录模型、维度、graph、seed、shape、`test_only` 等信息。

部分文件只有运行到对应阶段后才生成。例如：没有运行 LLM extraction 时，`extractions/raw/` 不会有真实抽取结果；没有 validated samples 时，`dataset_links.jsonl` 可能为空。

### `src/kg_pipeline/`

离线 KG 构建代码：

- `records.py`, `repeat_units.py`：DatasetRecord 和 RepeatUnit 生成。
- `polymer_classes.py`：rule-stub mapping 和 alias 输出。
- `document_parser.py`：本地文档解析和 chunking。
- `chunk_recall.py`：candidate chunk recall。
- `extraction.py`：Qwen / DeepSeek API 调用、prompt contract、raw extraction、临时 LLM mapping。
- `aggregation.py`, `validation.py`：article-level 聚合、证据和枚举校验。
- `linking.py`：dataset links 生成。
- `graph_builder.py`, `numeric_features.py`：KG nodes / edges / triples 构建和数值处理。
- `embedding.py`：TransE baseline 和 embedding 导出。
- `schemas/`：schema helper 和 `schema_v2.json`。
- `retrieval.py`：占位接口。在线文献检索尚未实现。

### `scripts/kg/`

CLI wrappers。它们只做参数解析和流程编排，核心逻辑在 `src/kg_pipeline/`。

## 4. JSON schema v2.0 简要说明

固定 v2.0 文档结构定义在 `LLM_KG.md`，当前代码中的对应文件是 `src/kg_pipeline/schemas/v2.py` 和 `src/kg_pipeline/schemas/schema_v2.json`。

对象说明：

- `article`：来源文章元数据和稳定 `article_id`。
- `evidence_records`：证据句，绑定 `chunk_id`、section/page/paragraph 元数据和原文句子。
- `literature_samples`：文章内样品或材料体系，不是 dataset row。
- `composition_assertions`：样品级组成事实，包括 composition type、ratio basis、components 和 evidence。
- `sequence_distribution_assertions`：明确表达的序列分布事实，如 random、block、alternating。
- `chain_architecture_assertions`：明确表达的链架构事实，如 linear、branched、star、network。
- `molecular_weight_measurements`：样品级 Mn、Mw、PDI/dispersity、DP。
- `polymerization_events`：聚合、合成或制备方法和条件。
- `dataset_links`：LiteratureSample 到 RepeatUnit 的弱链接。
- `warnings`：ambiguous、invalid、rejected 或 unsupported facts 的审计信息。

证据链：

```text
Fact -> Evidence -> SourceChunk -> Article
```

当前 validation 要求事实引用 evidence，并拒绝证据句无法在 SourceChunk 中定位的事实进入 KG。当前 `validate_schema_v2` 是轻量校验，只检查顶层必需字段、schema version 和数组类型；`schema_v2.json` 文件存在，但还没有被完整执行。

## 5. LLM provider 配置

当前支持的 provider 名称：

- `qwen`
- `deepseek`

provider 通过 CLI 参数传入：

```bash
python scripts/kg/extract_articles.py --provider qwen ...
python scripts/kg/extract_articles.py --provider deepseek ...
python scripts/kg/generate_llm_mapping.py --provider qwen ...
python scripts/kg/generate_llm_mapping.py --provider deepseek ...
```

环境变量：

- Qwen API key：`DASHSCOPE_API_KEY`
- Qwen model override：`QWEN_MODEL`，默认 `qwen-plus`
- DeepSeek API key：`DEEPSEEK_API_KEY`
- DeepSeek model override：`DEEPSEEK_MODEL`，默认 `deepseek-chat`

API key 只从环境变量读取，不写入代码或日志。`calls.jsonl` 保存 call ID、provider、candidate ID、status、error、usage 和 elapsed time，不保存 authorization header。

raw extraction 输出：

- 成功候选：`kg_work/extractions/raw/<candidate_id>.json`
- 汇总成功结果：`kg_work/extractions/raw/extractions.jsonl`
- 调用审计：`kg_work/extractions/raw/calls.jsonl`

临时 LLM mapping：

- `scripts/kg/generate_llm_mapping.py` 会请求 LLM 给出保守 PolymerClass candidate。
- 输出标记 `source_type = "llm_temporary_mapping"`。
- confidence 上限为 `0.70`。
- warnings 会明确说明该映射不是 verified registry mapping。
- 该映射仅用于测试 KG pipeline，不是正式 RepeatUnit / PolymerClass 科学映射。

## 6. 运行命令

help 检查：

```bash
python scripts/kg/prepare_records.py --help
python scripts/kg/map_polymer_classes.py --help
python scripts/kg/generate_llm_mapping.py --help
python scripts/kg/parse_documents.py --help
python scripts/kg/recall_chunks.py --help
python scripts/kg/extract_articles.py --help
python scripts/kg/validate_articles.py --help
python scripts/kg/build_dataset_links.py --help
python scripts/kg/build_kg.py --help
python scripts/kg/train_kg_embedding.py --help
python scripts/kg/run_pilot.py --help
python scripts/train.py --help
python scripts/pretrain.py --help
```

最小本地流程：

```bash
python scripts/kg/prepare_records.py \
  --input data/raw/smi_all.csv \
  --output_dir kg_work

python scripts/kg/map_polymer_classes.py \
  --repeat_units kg_work/repeat_units.csv \
  --output_dir kg_work \
  --mode rule_stub

python scripts/kg/parse_documents.py \
  --input_dir kg_work/documents \
  --output kg_work/chunks/source_chunks.jsonl

python scripts/kg/recall_chunks.py \
  --source_chunks kg_work/chunks/source_chunks.jsonl \
  --output kg_work/chunks/candidate_chunks.jsonl

python scripts/kg/extract_articles.py \
  --candidate_chunks kg_work/chunks/candidate_chunks.jsonl \
  --source_chunks kg_work/chunks/source_chunks.jsonl \
  --output_dir kg_work/extractions/raw \
  --provider qwen

python scripts/kg/validate_articles.py \
  --raw_dir kg_work/extractions/raw \
  --source_chunks kg_work/chunks/source_chunks.jsonl \
  --output_dir kg_work/extractions/validated

python scripts/kg/build_dataset_links.py \
  --validated_dir kg_work/extractions/validated \
  --repeat_units kg_work/repeat_units.csv \
  --polymer_class_candidates kg_work/polymer_class_candidates.jsonl \
  --output kg_work/links/dataset_links.jsonl

python scripts/kg/build_kg.py \
  --records kg_work/records.csv \
  --repeat_units kg_work/repeat_units.csv \
  --validated_dir kg_work/extractions/validated \
  --dataset_links kg_work/links/dataset_links.jsonl \
  --source_chunks kg_work/chunks/source_chunks.jsonl \
  --output_dir kg_work/kg \
  --graph strict

python scripts/kg/train_kg_embedding.py \
  --triples kg_work/kg/triples.tsv \
  --nodes kg_work/kg/nodes.csv \
  --edges kg_work/kg/edges.csv \
  --repeat_units kg_work/repeat_units.csv \
  --output_dir kg_work/features \
  --embedding_dim 128 \
  --model transe \
  --graph strict
```

启用 Uni-Poly `kg` 模态：

```bash
python scripts/train.py \
  --modalities smiles graph fp geom kg \
  --kg_mapping_path kg_work/features/kg_entity_mapping.csv \
  --kg_embedding_path kg_work/features/kg_embedding.npy \
  --kg_embedding_dim 128 \
  --kg_projection_dim 256
```

Pilot runner：

```bash
python scripts/kg/run_pilot.py \
  --input data/raw/smi_all.csv \
  --documents_dir kg_work/documents \
  --output_dir kg_work \
  --max_repeat_units 50 \
  --provider qwen \
  --stop_after embedding
```

注意：

- `extract_articles.py` 需要 API key；缺少 key 时会明确失败。
- 没有本地文献时，document parsing 和 recall 会产生 0 个 chunks/candidates。
- 没有 validated literature samples 时，`dataset_links.jsonl` 可能为空，embedding 可能标记为 `test_only`。

## 7. KG 构建说明

`build_kg.py` 将 dataset records、repeat units、临时 mapping candidates、validated literature JSON、source chunks 和 dataset links 转成 KG 文件。

节点类型：

- Dataset 节点：`DatasetRecord`、`RepeatUnit`、临时 `PolymerClass` candidate。
- Literature 节点：`Article`、`SourceChunk`、`Evidence`、`LiteratureSample`。
- Fact 节点：`CompositionAssertion`、`SequenceDistributionAssertion`、`ChainArchitectureAssertion`、`MolecularWeightMeasurement`、`PolymerizationEvent`。
- Enum/value 节点：`CompositionType`、`SequenceDistributionType`、`ChainArchitectureType`、`PolymerizationMethod`、`Component`、`NumericBin`。

边类型示例：

- Dataset 结构：`DatasetRecord --has_repeat_unit--> RepeatUnit`。
- 映射边：`RepeatUnit --maps_to--> PolymerClass`。
- provenance：`Evidence --located_in--> SourceChunk --part_of--> Article`。
- fact 边：`LiteratureSample --has_molecular_weight--> MolecularWeightMeasurement`。
- 证据支持：`Fact --supported_by--> Evidence`。
- dataset links：`LiteratureSample --canonical_smiles_match--> RepeatUnit`。

Graph views：

- `strict`：默认 graph variant。三元组生成时包含 strict edges 和 provenance edges，但排除 SourceChunk endpoint。
- `broad`：包含 strict、broad 和 provenance edges，同样排除 SourceChunk endpoint。

当前 strict 规则：

- `exact_repeat_unit_match` 和 `canonical_smiles_match` 是 strict。
- `polymer_class_match` 只有 confidence 至少 `0.8` 时才是 strict。
- 低置信 mapping edges 是 broad。
- SourceChunk 和 Evidence 保留在 `nodes.csv` / `edges.csv` 中用于 provenance。SourceChunk 不进入 TransE triples。
- warnings 是 audit 信息，不作为普通传播事实。

连续数值：

- Mn 和 Mw 以 raw value/unit 保存到 `MolecularWeightMeasurement.properties_json`，支持时额外保存 g/mol 标准化字段。
- Mn、Mw、DP 会生成 logarithmic `NumericBin` 节点。
- PDI/dispersity 会被校验并作为属性保存；当前不生成 numeric bin。
- temperature、time、pressure、pH 保存在 `PolymerizationEvent.properties_json` 中；当前没有单位标准化或 numeric bin。

## 8. TransE KG embedding

脚本：

```bash
python scripts/kg/train_kg_embedding.py ...
```

输入：

- `kg_work/kg/triples.tsv`
- `kg_work/kg/nodes.csv`
- `kg_work/kg/edges.csv`
- `kg_work/repeat_units.csv`

输出：

- `kg_work/features/kg_entity_mapping.csv`
- `kg_work/features/kg_embedding.npy`
- `kg_work/features/embedding_manifest.json`

默认：

- model：`TransE`
- graph：`strict`
- `embedding_dim = 128`
- seed：`13`
- epochs：默认 `100`，可通过 CLI 覆盖。

mapping 与矩阵关系：

- `kg_entity_mapping.csv.embedding_index` 是 `kg_embedding.npy` 的行号。
- 当前每个 RepeatUnit 输出一行。
- `kg_embedding.npy` 是二维 `float32` 矩阵，shape 为 `[num_repeat_units, embedding_dim]`。
- mapping index 必须从 0 开始且连续。

fallback 行为：

- 如果 RepeatUnit KG node 有训练出的 entity embedding，使用该向量。
- 否则，如果当前 graph scope 中有 PolymerClass node，使用 class vector。
- 否则，使用 seeded learned unknown fallback vector。
- fallback 不是零向量。

provenance 与 embedding：

- SourceChunk 节点不进入 TransE triples。
- Evidence 节点可通过 provenance/support edges 出现在 triples 中。
- `has_literature_link` 反映 RepeatUnit 是否由文献关系类型链接，不只是 DatasetRecord 结构边。

`embedding_manifest.json` 记录 model、dimension、epochs、seed、graph variant、triple count、entity count、mapped entity count、shape、SourceChunk exclusion、fallback 和 `test_only`。

当没有 literature links 时，`test_only` 会是 `true`。当前项目中已有的 `kg_work/features/embedding_manifest.json` 可能是 `test_only=true`，因为可用生成 KG 中只有 dataset 结构和临时 mapping，没有 validated literature links。

## 9. Uni-Poly 第五模态 `kg` 接入

基础模态：

- `smiles`
- `graph`
- `fp`
- `geom`

新增可选模态：

- `kg`

CLI 启用：

```bash
python scripts/train.py \
  --modalities smiles graph fp geom kg \
  --kg_mapping_path kg_work/features/kg_entity_mapping.csv \
  --kg_embedding_path kg_work/features/kg_embedding.npy \
  --kg_embedding_dim 128 \
  --kg_projection_dim 256
```

`scripts/pretrain.py` 也暴露相同 KG 路径和维度参数。

Dataset 行为：

- `UniDataset(..., enable_kg=True, kg_mapping_path=..., kg_embedding_path=...)` 会加载 `kg_entity_mapping.csv`，并 memory-map `kg_embedding.npy` 以校验 shape 和行数。
- Dataset 会在可行时用 RDKit canonicalize 每条样本 SMILES。
- Dataset 用 `canonical_smiles` 查找 `kg_entity_mapping.csv`。
- 每个 data object 新增：
  - `kg_embedding_index`：embedding 行号；未匹配时为 `-1`。
  - `kg_mask`：是否存在有效 embedding index。
  - `has_literature_link`：该 RepeatUnit 是否有文献链接。

DataLoader 行为：

- `custom_collate` 只有在样本中存在 KG 字段时，才 batch `kg_embedding_index`、`kg_mask`、`has_literature_link`。
- 不启用 `kg` 时，原四模态 batch 不变。

模型行为：

- `UniEncoderAttention` 支持 `SUPPORTED_MODALITIES = ('smiles', 'graph', 'fp', 'geom', 'kg')`。
- `kg` 分支用 `nn.Embedding.from_pretrained` 加载 `kg_embedding.npy`。
- 训练入口中 `kg_freeze_embedding` 当前只接受 `true`，因此 KG embedding table 在 train/pretrain 中被冻结。
- 缺失 KG index 使用可训练 `unknown_embedding`。
- KG projection 默认是 `128 -> 256`，结构为 `Linear`、`LayerNorm`、`ReLU`。
- projection 后的 KG 向量与其他模态 embedding 一起进入原有 fusion module 和 attention pooling。

兼容性：

- 不启用 `kg` 时不需要 KG 文件。
- CLI 默认模态仍是 `smiles graph fp geom`。
- `scripts/train.py` 仍使用 `strict=False` 加载 checkpoint；启用 `kg` 时旧四模态 checkpoint 可以以 missing KG keys 的形式兼容加载。
- 没有修改 PaiNN / SchNet。
- 没有恢复 TEXT 模态。
- 新 `kg` 不是已删除的旧 KG 实现。

## 10. Label leakage 控制

规则：

- `val` 不进入 LLM prompt。
- `val` 不进入 retrieval query。
- `val` 不进入 PolymerClass mapping。
- `val` 不进入 `dataset_links`。
- `val` 不进入 KG nodes、edges、triples 或 embedding。

已实现控制：

- `prepare_records.py` 写出的 `records.csv` 只有 `record_id`、`row_index`、`smiles`、`prop`、`repeat_unit_id`，不写 `val`。
- `generate_llm_mapping()` 的 prompt 只使用 `repeat_unit_id`、`raw_smiles`、`canonical_smiles`。
- `extract_articles()` 的 prompt 只使用 candidate chunks 和元数据，不使用 dataset labels。
- `assert_no_val_fields()` 会递归拒绝 prompt payload 中名为 `val` 的字段。

建议审计命令：

```bash
head -1 kg_work/records.csv

rg -n '(^|,)val(,|$)|"val"\\s*:' kg_work \
  --glob '*.csv' \
  --glob '*.json' \
  --glob '*.jsonl' \
  --glob '*.tsv'
```

当前 TODO：

- 还没有把通用 label-leakage scanner 接入每个 CLI stage。
- 如果未来抽取到与当前 `prop` 相同的属性值，计划行为是 warning 或从 graph input 中屏蔽；当前尚未实现完整 property-target leakage filter。

## 11. 测试与验证

可运行的基础检查：

```bash
python -m py_compile src/kg_pipeline/*.py src/kg_pipeline/schemas/*.py scripts/kg/*.py
python scripts/kg/prepare_records.py --help
python scripts/kg/map_polymer_classes.py --help
python scripts/kg/generate_llm_mapping.py --help
python scripts/kg/parse_documents.py --help
python scripts/kg/recall_chunks.py --help
python scripts/kg/extract_articles.py --help
python scripts/kg/validate_articles.py --help
python scripts/kg/build_dataset_links.py --help
python scripts/kg/build_kg.py --help
python scripts/kg/train_kg_embedding.py --help
python scripts/kg/run_pilot.py --help
python scripts/train.py --help
python scripts/pretrain.py --help
```

数据和 KG 完整性检查：

- `records.csv` 不包含 `val`。
- 相同输入下 `record_id` 和 `repeat_unit_id` 稳定。
- evidence sentence 能在对应 SourceChunk text 中定位。
- `review_queue.json` 记录 validation warnings。
- `nodes.csv` 中 `node_id` 唯一。
- `edges.csv` 中每个 endpoint 都存在于 `nodes.csv`。
- 真实 KGE 训练前，`triples.tsv` 应非空。
- `triples.tsv` 中不应出现 SourceChunk endpoint。
- `kg_entity_mapping.csv.embedding_index` 连续，并对应 `kg_embedding.npy` 行顺序。
- `kg_embedding.npy.shape[0] == len(kg_entity_mapping.csv)`。
- 默认情况下 `kg_embedding.npy.shape[1] == 128`。
- `kg` disabled 时原四模态不需要 KG 文件。
- `kg` enabled 时应在 RDKit、PyG、Transformers、本地 tokenizer/model、geometry cache 可用时做小 batch forward。

依赖说明：

- RDKit 用于 canonical SMILES 和 dataset processing。
- PyTorch / PyG 用于模型和图数据运行时。
- Transformers 和本地 SMILES encoder 路径用于完整 SMILES forward。
- Qwen / DeepSeek extraction 需要 provider API key。
- PDF 解析需要 `pypdf`。
- CUDA 对 smoke test 不是必需，但训练会更快。

## 12. 当前限制与后续 TODO

当前限制：

- 临时 LLM mapping 不是正式 PolymerClass / RepeatUnit registry。
- `rule_stub` mapping 非常保守，通常输出 unknown 或 heuristic condensation candidate。
- 大规模在线文献检索未实现；`src/kg_pipeline/retrieval.py` 是 placeholder。
- PDF 解析依赖 `pypdf`；表格解析只是基础文本抽取，不是可靠的科学表格解析器。
- candidate recall 缺少 dense retrieval、真正的 sample-conditioned second recall 和 reranker。
- LLM provider 代码存在，但需要 API key 和外部 API 访问。
- `schema_v2.json` 存在，但当前 validation 使用轻量程序检查，没有完整执行 JSON Schema。
- dataset linking 未实现 functional-group similarity 或 family-level matching。
- numeric bins 只覆盖 Mn、Mw、DP。
- temperature、time、pressure、pH 只作为 event properties 保存，未标准化或分箱。
- TransE 是 baseline，不是最终最强 KG 模型。
- 当前不实现 R-GCN / HGT。
- 当前生成的 `kg_work/features/embedding_manifest.json` 在没有 literature links 时会是 `test_only=true`。
- 当前不启动长时间训练。
- 不恢复旧 KG 或 TEXT 模态。

后续 TODO：

- 建立 curated RepeatUnit / PolymerClass / alias registry。
- 增加可靠的文献检索和 Article 去重。
- 改进 HTML/XML/PDF/table parsing 和 row-level table evidence。
- 实现 dense retrieval 和 sample-conditioned second recall。
- 让 validation 真正执行 `schema_v2.json` 或等价严格 schema。
- 增加针对 downstream prediction target 的 property leakage filter。
- 增加 functional-group / family-level linking，并校准 confidence。
- 扩展 event condition 的数值标准化和分箱。
- 增加 KG evaluation、nearest-neighbor sanity check 和 ablation report。
- 只在 API key、prompt 和数据合规确认后运行真实 LLM extraction。
- 只在明确请求后运行下游 smoke test 或训练。

## 13. 与计划文档的已知差异

当前实现是 MVP，与 `KG_IMPLEMENTATION_PLAN.md` 有以下差异：

- 文献检索未实现；当前只有本地文档解析。
- dense retrieval 和 reranking 未实现；当前 recall 是 keyword / BM25-style。
- sample-conditioned second recall 只体现在 CLI 参数中，没有真实实现。
- 虽然存在 `schema_v2.json`，但当前没有完整 JSON Schema validation。
- LLM mapping 是临时测试映射，并限制 confidence；正式 registry 不存在。
- dataset linking 仅覆盖 canonical SMILES、PolymerClass candidate 和 alias 匹配。
- numeric binning 只覆盖 Mn、Mw、DP，不覆盖所有连续 event conditions。
- 没有 validated literature links 时，当前 embedding 会标记为 `test_only`。
