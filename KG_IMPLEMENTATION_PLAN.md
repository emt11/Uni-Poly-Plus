# Polymer KG 实现级项目计划

## 0. 文档定位

本文档将 `LLM_KG.md` 中的设计转化为可实施、可验证、可分阶段交付的工程计划。当前阶段只定义数据契约、目录、Pipeline、KG 建模、embedding 方案、Uni-Poly 接入方案和实施 TODO，不实现代码、不运行 LLM 抽取、不生成 KG embedding，也不修改现有模型或训练流程。

本计划遵守以下不可变约束：

1. `data/raw/smi_all.csv` 只有 `smiles`、`val`、`prop`。
2. `val` 是预测标签，不能进入文献检索、LLM prompt、PolymerClass 判断、dataset linking 或 KG 构建。
3. `LiteratureSample` 是文献中的具体材料体系，不等于 `DatasetRecord`。
4. 文献中的组成、序列、链架构、分子量和制备条件只能作为 `LiteratureSample` 事实。
5. 文献事实只能通过 `dataset_links` 作为邻域知识连接到 `RepeatUnit` 或 `PolymerClass`，不能回填为当前数据行真值。
6. 不生成 `enriched_records.csv`，不生成独立 `numeric_descriptor_matrix`。
7. 所有有效事实统一进入 Polymer KG。
8. 最终供下游读取的稳定文件只有 `kg_entity_mapping.csv` 和 `kg_embedding.npy`。
9. 所有事实必须能够追踪到 `Evidence -> SourceChunk -> Article`。
10. LLM 只能抽取 chunk 中明确表达的事实，不能依赖常识补全。
11. PA66、PET 等固定缩聚重复单元不能仅因涉及多个单体而标为 random、block 或 graft copolymer。

---

## 1. 项目目标与总体 Pipeline

### 1.1 目标

从 `smi_all.csv` 中的 repeat-unit-like Polymer SMILES 建立稳定的 `DatasetRecord -> RepeatUnit` 入口；从文献中抽取带证据的样品级知识；通过可审计的弱链接连接两个数据域；构建包含结构、类别、组成、序列、架构、分子量、制备条件和来源证据的 Polymer KG；最后为 RepeatUnit/PolymerClass 生成可被 Uni-Poly 读取的 KG embedding。

### 1.2 端到端流程

```text
data/raw/smi_all.csv
  -> DatasetRecord
  -> RepeatUnit normalization
  -> PolymerClass / alias mapping
  -> literature retrieval
  -> document parsing
  -> candidate chunk recall
  -> LLM extraction with JSON schema v2.0
  -> article-level LiteratureSample aggregation
  -> schema / evidence / enum / unit validation
  -> dataset_links
  -> KG nodes / edges / triples
  -> KG embedding
  -> Uni-Poly integration
```

### 1.3 Pipeline 总览

| 步骤 | 输入 | 输出 | 主要作用 | Pilot 必需 |
|---|---|---|---|---|
| DatasetRecord 建立 | `smi_all.csv` | `records.csv` | 给原始行分配稳定 ID，并物理隔离 `val` | 是 |
| RepeatUnit 规范化 | `records.csv` | `repeat_units.csv` | 去重、规范化 SMILES、建立 record 到 RepeatUnit 映射 | 是 |
| PolymerClass/alias 映射 | RepeatUnit | class candidates、alias 表 | 为检索和图连接建立类别身份候选 | 是 |
| 文献检索 | class/alias/repeat-unit query | article metadata | 获取候选文献并去重 | 是 |
| 文档解析 | HTML/XML/PDF | SourceChunk | 保留章节、段落、页码、表格和顺序 | 是 |
| candidate chunk 召回 | SourceChunk + schema 字段 | candidate chunks | 降低 LLM 成本并提高相关性 | 是 |
| LLM 局部抽取 | candidate chunk | raw extraction JSONL | 只抽取明确表达的局部事实和证据句 | 是 |
| 文档级聚合 | raw extraction + article context | article JSON v2.0 | 跨 chunk 对齐同一 LiteratureSample | 是 |
| 校验与标准化 | article JSON | validated JSON + warnings | 校验证据、枚举、单位、ID 和科学规则 | 是 |
| dataset linking | LiteratureSample + RepeatUnit/class | `dataset_links` | 建立有类型、有置信度的候选弱链接 | 是 |
| KG 构建 | validated JSON + links | nodes/edges/triples | 将对象、事实、证据和链接统一图化 | 是 |
| KG embedding | strict/broad KG | mapping + `.npy` | 生成 RepeatUnit/PolymerClass 等节点向量 | 是 |
| Uni-Poly 接入 | mapping + embedding | 新 KG 输入分支 | 将文献邻域知识用于属性预测 | Pilot 只验证读取；训练接入需另行选择 |

---

## 2. 数据边界与 ID 体系

### 2.1 数据域隔离

系统必须维护三个不同的数据域：

1. **Dataset domain**：`DatasetRecord`、`RepeatUnit`，来源于 `smi_all.csv`。
2. **Literature domain**：`Article`、`SourceChunk`、`Evidence`、`LiteratureSample` 及其事实对象。
3. **Linking domain**：`dataset_links`，只表达候选关联，不表达样品同一性。

禁止将 Literature domain 的 Mn、Mw、PDI、DP、比例或条件复制到 DatasetRecord 属性中。

### 2.2 推荐稳定 ID

| 对象 | ID 示例 | 生成原则 |
|---|---|---|
| DatasetRecord | `dr_000001` | 按源文件、原始行号和内容 hash 稳定生成 |
| RepeatUnit | `ru_<hash>` | canonical repeat-unit SMILES 的稳定 hash |
| PolymerClass | `pc_pa66` | 受控名称 slug 或注册表 ID |
| Article | `art_<doi_hash>` | 优先 DOI，缺 DOI 时由 title/year/source hash 生成 |
| SourceChunk | `chunk_<article>_<order>` | Article ID + 文档顺序 |
| Evidence | `ev_<hash>` | chunk ID + 原文句子位置 hash |
| LiteratureSample | `ls_<article>_<local>` | Article 范围内稳定编号 |
| Assertion/Measurement/Event | 类型前缀 + article/local | 由后处理程序生成，不能依赖 LLM 自由命名 |
| dataset link | `link_<hash>` | LiteratureSample + target + relation_type hash |
| KG node | `kg_<type>_<source_id>` | 保留来源对象 ID，便于反查 |

ID 生成必须幂等。断点续跑时，同一输入不得产生不同 ID。

### 2.3 `val` 隔离策略

- Stage 0 可以读取 `val` 以保留原始训练记录，但输出面向 KG 的工作文件时默认不包含 `val`。
- KG Pipeline 的函数接口只接收 `record_id`、`smiles`、`prop` 或更少字段；检索、分类、链接模块不得接受 `val` 参数。
- 日志、缓存文件名、prompt trace 和检索 query 中不得出现 `val`。
- 若抽取事实的 property 与当前 `prop` 相同，默认将该 property value 标记为 `possible_label_leakage`，不进入图输入。
- 建议增加自动审计：扫描所有 prompt、query、KG node/edge 属性，确认不存在原始标签值来源字段。

---

## 3. 推荐目录与中间产物

保持当前项目结构，最小新增以下目录。目录中的文件是后续实现规划，不在本阶段创建。

```text
kg_work/
  records.csv
  repeat_units.csv
  polymer_class_candidates.jsonl
  entity_aliases.csv
  literature_index.jsonl
  documents/
  chunks/
    source_chunks.jsonl
    candidate_chunks.jsonl
  extractions/
    raw/
    aggregated/
    validated/
  links/
    dataset_links.jsonl
  kg/
    nodes.csv
    edges.csv
    triples.tsv
    entity_aliases.csv
    build_manifest.json
  features/
    kg_entity_mapping.csv
    kg_embedding.npy
    embedding_manifest.json
  logs/
  state/

src/kg_pipeline/
  records.py
  repeat_units.py
  polymer_classes.py
  retrieval.py
  document_parser.py
  chunk_recall.py
  extraction.py
  aggregation.py
  validation.py
  linking.py
  graph_builder.py
  numeric_features.py
  embedding.py
  schemas/

scripts/kg/
  prepare_records.py
  map_polymer_classes.py
  retrieve_literature.py
  parse_documents.py
  recall_chunks.py
  extract_articles.py
  validate_articles.py
  build_dataset_links.py
  build_kg.py
  train_kg_embedding.py
  run_pilot.py
```

`src/kg_pipeline/` 是离线数据与图构建代码，不应直接依赖训练模块。`scripts/kg/` 只做参数解析和流程编排。Uni-Poly 接入在选定方案后再最小修改 `src/dataset/`、`src/modules/` 和训练入口。

---

## 4. JSON schema v2.0 到 KG 的映射

### 4.1 映射原则

- 有独立身份、需要多条关系或需要证据引用的对象建为节点。
- 枚举值优先建为受控实体节点，避免把语义埋在自由字符串中。
- 连续值默认保留为 Measurement/Event 节点属性；是否增加 numeric bin 节点由数值方案决定。
- `evidence_refs` 不作为普通字符串列表留在最终图中，而转为 `Fact --supported_by--> Evidence` 边。
- provenance 路径必须完整：`Fact -> Evidence -> SourceChunk -> Article`。
- `warnings` 默认作为审计信息，不进入消息传播图。

### 4.2 对象映射表

| JSON 对象/字段 | KG 节点 | KG 边 | 节点属性 | 边属性 |
|---|---|---|---|---|
| `article` | `Article` | 被 Sample/Chunk 引用 | title、doi、year、journal | 无 |
| `evidence_records` | `Evidence`、对应 `SourceChunk` | Evidence `located_in` SourceChunk；SourceChunk `part_of` Article | sentence、section、page、paragraph_id、chunk order | 无 |
| `literature_samples` | `LiteratureSample` | `reported_in` Article；`belongs_to` PolymerClass | sample_label、polymer_name、原始别名 | identity confidence 可选 |
| `aliases` | Alias 节点或 alias registry | Alias `alias_of` LiteratureSample/PolymerClass | alias text、normalized text、source | confidence 可选 |
| `composition_assertions` | `CompositionAssertion` | Sample `has_composition` Assertion | ratio_basis、原始比例表达 | 无 |
| `composition_type` | `CompositionType` 枚举节点 | Assertion `has_composition_type` Type | enum code、label | 无 |
| `components` | `Component` 或规范化 `ChemicalEntity` | Assertion `has_component` Component | name、role、value、normalized value/unit | component order、ratio basis 可选 |
| `sequence_distribution_assertions` | `SequenceDistributionAssertion` | Sample `has_sequence_distribution` Assertion；Assertion `has_distribution_type` Enum | 原文枚举、规范化枚举 | 无 |
| `chain_architecture_assertions` | `ChainArchitectureAssertion` | Sample `has_chain_architecture` Assertion；Assertion `has_architecture_type` Enum | 原文枚举、规范化枚举 | 无 |
| `molecular_weight_measurements` | `MolecularWeightMeasurement` | Sample `has_molecular_weight` Measurement | Mn/Mw/PDI/DP 原值、单位、标准化值 | 无 |
| `polymerization_events` | `PolymerizationEvent` | Sample `has_polymerization_event` Event；Event `has_method` Method | temperature/time/pressure/pH 原值和标准化值 | 无 |
| solvent/atmosphere | `Solvent`/`Atmosphere` 枚举或 ChemicalEntity | Event `uses_solvent`/`under_atmosphere` | normalized name | 无 |
| `dataset_links` | 默认不单独建节点；需要 reification 时可建 `DatasetLink` | Sample 到 RepeatUnit/PolymerClass 的 typed edge | 无 | relation_type、confidence、matched_on、link_id |
| `warnings` | 默认不建节点 | 默认无 | warning code、object ID、message 存 audit 文件 | 无 |

### 4.3 Article、SourceChunk 与 Evidence

JSON v2.0 顶层只有 `article` 和 `evidence_records`，但 KG 中必须显式建立 `SourceChunk`：

```text
Evidence --located_in--> SourceChunk --part_of--> Article
LiteratureSample --reported_in--> Article
Fact --supported_by--> Evidence
```

`SourceChunk` 的文本可存外部文档存储路径和 hash，不建议把整段全文复制到 `nodes.csv`。`Evidence.sentence` 保留原句，必须可以在对应 chunk 中精确或归一化匹配。

### 4.4 LiteratureSample 身份

`LiteratureSample` 节点只代表文章语境中的具体样品或材料体系。其 `polymer_name`、`sample_label`、aliases 和明确给出的 repeat-unit structure 用于身份解析。身份事实由 `identity_evidence_refs` 建立：

```text
LiteratureSample --identity_supported_by--> Evidence
LiteratureSample --belongs_to--> PolymerClass
```

`belongs_to` 可以有 `mapping_method` 和 `confidence`，但只有证据明确或确定性映射可靠时进入 strict graph。

### 4.5 CompositionAssertion

推荐将组成类型与组件拆开：

```text
LiteratureSample --has_composition--> CompositionAssertion
CompositionAssertion --has_composition_type--> CompositionType
CompositionAssertion --has_component--> Component
CompositionAssertion --supported_by--> Evidence
```

`composition_type` 受控枚举：

```text
homopolymer
condensation_multi_monomer
copolymer
terpolymer
blend
composite
mixture
unknown
```

`condensation_multi_monomer` 与 sequence distribution 分离。PA66/PET 可属于该组成类别，但在文献未明确说明时不创建 random/block/graft sequence 节点。

组件比例必须保留 `ratio_basis`。`feed_ratio`、`actual_ratio`、`weight_fraction`、`mol_fraction` 和 `stoichiometric_ratio` 不能混为同一数值。

### 4.6 SequenceDistributionAssertion

```text
LiteratureSample --has_sequence_distribution--> SequenceDistributionAssertion
SequenceDistributionAssertion --has_distribution_type--> SequenceDistributionType
SequenceDistributionAssertion --supported_by--> Evidence
```

distribution 枚举为 random、statistical、alternating、block、blocky、multiblock、gradient、graft、periodic、not_applicable、unknown。没有明确证据时使用空数组，而不是创建 `unknown` 占位事实；`unknown` 只用于原文明确表述未知或无法归一化但确有该事实的情况。

### 4.7 ChainArchitectureAssertion

```text
LiteratureSample --has_chain_architecture--> ChainArchitectureAssertion
ChainArchitectureAssertion --has_architecture_type--> ChainArchitectureType
ChainArchitectureAssertion --supported_by--> Evidence
```

architecture 枚举包括 linear、branched、star、graft、comb、brush、network、crosslinked_network、hyperbranched、dendrimer、cyclic、ladder、unknown。不得从 repeat unit 或聚合方法自动推断。

### 4.8 MolecularWeightMeasurement

```text
LiteratureSample --has_molecular_weight--> MolecularWeightMeasurement
MolecularWeightMeasurement --supported_by--> Evidence
```

Mn、Mw、PDI/Đ、DP 作为同一次 measurement 的属性共同保存。同一样品在不同方法、批次或文献位置存在多个值时保留多个 Measurement 节点，不先求均值。原始值和单位必须保留；统一单位和派生检查由确定性程序完成，不改写 JSON 原值。

### 4.9 PolymerizationEvent

```text
LiteratureSample --has_polymerization_event--> PolymerizationEvent
PolymerizationEvent --has_method--> PolymerizationMethod
PolymerizationEvent --uses_solvent--> Solvent
PolymerizationEvent --under_atmosphere--> Atmosphere
PolymerizationEvent --supported_by--> Evidence
```

temperature、time、pressure、pH 保存为 Event 节点属性或 ConditionValue 节点。v2.0 中 broad preparation information 暂存于此；构图时可以用 `event_category=polymerization|processing|preparation|unknown` 作为确定性后处理属性，但不改变固定 JSON schema。多阶段条件拆为多个 Event。

### 4.10 dataset_links

默认直接转换为有类型的边：

```text
LiteratureSample --exact_repeat_unit_match--> RepeatUnit
LiteratureSample --canonical_smiles_match--> RepeatUnit
LiteratureSample --polymer_class_match--> PolymerClass
LiteratureSample --alias_match--> RepeatUnit/PolymerClass
LiteratureSample --family_level_match--> PolymerClass
```

边属性至少包含：

```text
link_id
confidence
matched_on
evidence_ids
linking_version
```

`uncertain` 链接保留在 audit 输出中，默认不进入用于 embedding 的 strict/broad graph。若选用不支持边属性的 KGE，需要把 `relation_type` 作为关系类型，并通过阈值、分桶关系或过滤策略处理 confidence。

---

## 5. KG 文件设计

### 5.1 `kg_work/kg/nodes.csv`

每行一个节点。推荐字段：

| 字段 | 含义 |
|---|---|
| `node_id` | 全局唯一 KG ID |
| `node_type` | RepeatUnit、LiteratureSample、Measurement 等 |
| `source_id` | JSON/数据源中的原始对象 ID |
| `canonical_name` | 规范化名称 |
| `display_name` | 可读名称 |
| `properties_json` | 非关系属性；使用稳定 JSON 序列化 |
| `source_scope` | dataset、literature、enum、derived |
| `active` | 是否进入当前图版本 |
| `created_by` | deterministic、LLM_extracted、registry 等 |
| `schema_version` | `2.0` |

数值属性存入 `properties_json` 时必须同时保存 raw value、raw unit、normalized value、normalized unit 和 normalization status。不要把 Python 对象字符串写入 CSV。

### 5.2 `kg_work/kg/edges.csv`

每行一条可带属性的有向边：

| 字段 | 含义 |
|---|---|
| `edge_id` | 全局唯一边 ID |
| `head_id` | 起点 node_id |
| `relation_type` | 受控关系类型 |
| `tail_id` | 终点 node_id |
| `confidence` | 链接或映射置信度；普通事实边可为 1.0 |
| `evidence_ids` | 证据 ID 列表的稳定序列化 |
| `properties_json` | matched_on、mapping_method 等边属性 |
| `graph_scope` | strict、broad、provenance、audit |
| `active` | 是否进入当前图版本 |

### 5.3 `kg_work/kg/triples.tsv`

供传统 KGE 使用的无属性三元组文件：

```text
head_id<TAB>relation_type<TAB>tail_id
```

生成规则必须记录在 manifest 中。边属性无法直接进入三元组时：

- relation type 保留为不同关系；
- confidence 通过过滤阈值、relation bucket 或 link reification 处理；
- 原始 edge 属性仍保留在 `edges.csv`，避免信息丢失。

### 5.4 `kg_work/kg/entity_aliases.csv`

用于检索、归一化和实体链接，不等同于 KG embedding mapping：

| 字段 | 含义 |
|---|---|
| `alias_id` | alias 记录 ID |
| `alias_text` | 原始别名 |
| `normalized_alias` | 规范化文本 |
| `target_node_id` | 对应实体 |
| `target_type` | PolymerClass、LiteratureSample、ChemicalEntity 等 |
| `source_article_id` | 文献来源，可空 |
| `evidence_id` | 支持证据，可空 |
| `mapping_method` | exact、registry、LLM_candidate、manual |
| `confidence` | 校准分数 |
| `status` | accepted、review、rejected |

### 5.5 `kg_work/features/kg_entity_mapping.csv`

该文件只负责将模型入口实体映射到 embedding 行号：

| 字段 | 含义 |
|---|---|
| `embedding_index` | `kg_embedding.npy` 行号，0-based |
| `kg_node_id` | KG 节点 ID |
| `entity_type` | RepeatUnit、PolymerClass 或其他输出类型 |
| `repeat_unit_id` | 若为 RepeatUnit 则填写 |
| `canonical_smiles` | RepeatUnit 的规范化结构，可空 |
| `polymer_class_id` | 对应 PolymerClass，可空 |
| `graph_variant` | strict 或 broad |
| `has_literature_link` | 是否存在可传播的文献链接 |
| `embedding_version` | embedding 版本 |

不能在该文件中重复保存 Mn、Mw 或其他文献事实。

### 5.6 `kg_work/features/kg_embedding.npy`

- 二维 `float32` 数组，shape 为 `[num_mapped_entities, embedding_dim]`。
- 行顺序严格由 `kg_entity_mapping.csv.embedding_index` 定义。
- 至少输出所有 RepeatUnit；可同时输出 PolymerClass，但必须由 `entity_type` 区分。
- 不存在文献链接的 RepeatUnit 不能简单使用全零向量；具体回退策略由 embedding 方案决定。

### 5.7 Manifest

虽然最终下游只依赖 mapping 和 `.npy`，构建阶段应保存：

- `build_manifest.json`：输入 hash、schema、枚举、过滤阈值、strict/broad 配置、节点边数量。
- `embedding_manifest.json`：算法、超参数、随机种子、图版本、维度、训练集范围、数值编码方案。

Manifest 是复现实验所需的审计文件，不作为模型输入。

---

## 6. KG embedding 可选方案

本节只列出决策选项，不替项目负责人做最终选择。

### 6.1 方案 A：传统 KGE baseline

候选模型：TransE、DistMult、ComplEx、RotatE。

**输入**：`triples.tsv`，可选 train/validation/test triple split。  
**输出**：每个实体的离线 embedding，再筛选 RepeatUnit/PolymerClass 写入最终文件。

优点：

- 工程成熟，训练和复现实验简单。
- 适合快速验证 KG 是否提供额外信号。
- RotatE/ComplEx 能表示比 TransE 更复杂的关系模式。

缺点：

- 原生难以使用连续节点属性和 edge confidence。
- provenance 节点过多时可能稀释结构信号。
- 新实体通常需要重新训练或额外归纳机制。

实现复杂度：低。  
Baseline 适用性：高。  
论文实验适用性：适合作为强制 baseline，但不宜单独作为最终主模型。

### 6.2 方案 B：R-GCN / Relational GNN

**输入**：typed edges、节点类型、初始特征、可选 edge confidence。  
**输出**：所有节点的关系感知 embedding。

优点：

- 能按关系类型进行消息传递。
- 更容易融合 Measurement/Event 的连续特征。
- 可以通过 edge weight/gate 使用 dataset link confidence。

缺点：

- 关系多时参数量和训练成本增加。
- provenance 节点、弱链接和高阶邻域需要采样与正则化。
- transductive 实现对新增节点支持有限。

实现复杂度：中。  
Baseline 适用性：中。  
论文实验适用性：高，适合 strict/broad、数值特征和关系消融。

### 6.3 方案 C：Heterogeneous Graph Transformer / HGT

**输入**：明确的 node type、edge type、节点特征和异构图结构。  
**输出**：类型感知、注意力聚合后的节点 embedding。

优点：

- 与 Article、Sample、Assertion、Measurement、Event 等异构结构自然匹配。
- 可观察不同节点类型和关系的注意力。
- 便于未来增加 property measurement、processing event 或数据库实体。

缺点：

- 实现、调参和显存成本最高。
- 数据规模较小时容易过拟合。
- 需要严格的节点类型特征设计和邻居采样。

实现复杂度：高。  
Baseline 适用性：低。  
论文实验适用性：高，适合数据规模和 gold validation 足够后的主模型。

### 6.4 方案 D：两阶段方案

可选流程：

1. 用 TransE/DistMult/RotatE 训练离线 KG embedding。
2. 导出 RepeatUnit embedding。
3. 在 Uni-Poly 中通过 projection MLP 对齐维度。
4. 冻结 embedding，或只微调 projection；后续再比较端到端微调。

**输入**：三元组 + Uni-Poly 训练数据。  
**输出**：离线 KG embedding 和下游适配后的表示。

优点：

- KG 构建与下游模型解耦，便于缓存和复现。
- 可以比较 frozen、projection-only、fine-tuned 三种设置。
- 对现有 Uni-Poly 改动较小。

缺点：

- KGE 阶段可能未充分利用数值属性。
- 下游标签可能使 embedding projection 任务化，需严格控制训练/测试流程。
- 若直接微调整个 embedding 表，未出现实体和 CV 泄漏需要额外处理。

实现复杂度：中。  
Baseline 适用性：高。  
论文实验适用性：高，适合形成清晰消融链。

### 6.5 方案 E：strict graph 与 broad graph 对比

这不是独立编码器，而是所有 embedding 算法都应支持的图视图实验。

**strict graph**：

- 只保留 `exact_repeat_unit_match`、`canonical_smiles_match`。
- PolymerClass 映射只保留受控 registry 或人工确认的高置信映射。
- 目标是高精度、低覆盖。

**broad graph**：

- 加入 `polymer_class_match`、`alias_match`、`family_level_match`。
- 通过 threshold、relation type 或 edge confidence 降低弱链接影响。
- 目标是提高覆盖率，并量化噪声收益/损害。

优点：直接回答弱链接是否有效。  
缺点：必须固定检索语料、节点集和随机种子，否则图差异不可归因。  
实现复杂度：低到中。  
Baseline 适用性：strict graph 高。  
论文实验适用性：非常高。

### 6.6 建议定位，但不替代最终选择

- **最小可行方案候选**：方案 A（DistMult 或 RotatE）+ strict graph + 固定离线 embedding。
- **最适合论文实验的方案候选**：方案 A baseline 对比方案 B，并同时比较 strict/broad graph 和 frozen/trainable projection。
- **最适合长期扩展的方案候选**：方案 C，或以方案 B 为稳定中间形态后升级 HGT。

开始实现前需要选择：KGE 算法、embedding 维度、strict/broad 默认图、是否使用 provenance 节点训练、是否允许下游微调。

---

## 7. 连续数值进入 KG 的可选方案

适用字段：Mn、Mw、PDI/Đ、DP、temperature、time、pressure、pH，以及 component ratio。

### 7.1 通用预处理

无论选择哪种表示，先执行：

1. 保存原始 value 和 unit。
2. 解析范围、约数、上下界和科学计数法；无法安全解析时只保留原文并 warning。
3. 统一 Mn/Mw 为 g/mol、temperature 为 K 或 degC 的固定内部单位、time 为 s、pressure 为 Pa。
4. 保留 normalization status，禁止静默猜单位。
5. PDI 通常检查 `>= 1`；DP、Mn、Mw 检查正值；pH 做合理范围提示。
6. 派生 `Mw/Mn` 只用于一致性检查或派生图事实，不能覆盖文献报告的 PDI。

### 7.2 方案 A：只作为节点属性

**表示**：数值写入 MolecularWeightMeasurement 或 PolymerizationEvent 的属性。  
**优点**：语义清晰、图规模小、保留精确数值。  
**缺点**：TransE 等三元组 KGE 通常不会使用节点属性。  
**复杂度**：低。  
**适用**：使用 R-GCN/HGT 且支持节点数值特征，或先只保证数据完整性。

### 7.3 方案 B：节点属性 + numeric bin 节点

**表示**：保留精确属性，同时建立如 `Mn_10k_50k`、`Temperature_300K_350K` 的枚举节点。

```text
Measurement --has_Mn_bin--> Mn_10k_50k
Event --has_temperature_bin--> Temperature_300K_350K
```

**优点**：传统 KGE 可以学习数值区间共现；精确值仍可审计。  
**缺点**：边界选择影响结果；离散化损失精度；长尾分布需要对数分箱。  
**复杂度**：中。  
**适用**：传统 KGE baseline 和论文中的 numeric representation 消融。

分箱规则必须由确定性配置生成，建议 Mn/Mw/DP 使用 log-space bins，temperature/time/pressure 使用领域范围或训练语料 quantile bins。分箱边界在全实验中冻结，不能由 LLM 决定。

### 7.4 方案 C：数值编码器生成 value embedding

**表示**：为每类数值使用归一化标量、Fourier features、RBF 或小型 MLP 生成初始向量，输入 R-GCN/HGT。

**优点**：保留连续性，邻近数值具有相近表示；适合端到端异构 GNN。  
**缺点**：实现与调参复杂；缺失值、范围值和单位置信度需额外建模；传统 KGE 不直接适用。  
**复杂度**：高。  
**适用**：长期主模型或数值信息是核心贡献时。

### 7.5 v1 推荐候选

v1 可选择方案 B：精确数值保留为属性，同时增加固定 numeric bin 节点。原因是它同时兼容传统 KGE baseline 和后续关系 GNN。若 v1 明确使用 R-GCN 且时间有限，也可先采用方案 A。最终选择需在实现前确认。

---

## 8. KG embedding 接入 Uni-Poly 的可选方案

当前基础模态固定为 `smiles`、`graph`、`fp`、`geom`。旧 KG/TEXT 实现已经移除。本节描述未来新 Polymer KG 的接入选项，不代表当前已经实现。

### 8.1 方案 A：以新实现替代旧 KG 模态概念

含义是重新定义 `kg` 模态接口，而不是恢复旧 loader/encoder。

需要修改：

- `src/dataset/dataset.py`：根据 RepeatUnit 查 embedding index。
- `src/dataset/dataloader.py`：batch 中加入 index、mask 或直接向量。
- `src/modules/uni_encoder.py`：新增新的 KG projection/encoder 分支。
- `scripts/pretrain.py`、`scripts/train.py`：重新允许 `kg`，但必须绑定新文件格式。

Dataset/DataLoader 影响：中。  
Checkpoint 兼容：旧四模态 checkpoint 用 `strict=False` 加载；新 KG 参数为 missing keys。旧历史 KG checkpoint 不应视为兼容。  
复杂度：中。  
优点：模态语义明确，可直接进入现有 fusion。  
缺点：容易被误解为恢复旧 KG；必须严格版本化输入。

### 8.2 方案 B：作为第五模态接入

```text
smiles + graph + fp + geom + kg
```

KG embedding 经 LayerNorm/MLP projection 到 joint embedding dimension，与其他模态一起进入 Fusion Module。

需要修改模块与方案 A 相同，但强调保留四模态 baseline，并允许任意组合中显式启用 `kg`。

Dataset/DataLoader 影响：中。  
Checkpoint 兼容：四模态权重可部分加载；fusion 参数结构若不依赖固定模态数，兼容性较好。  
复杂度：中。  
优点：最容易进行 `4-modal vs 5-modal` 消融，attention 权重可解释。  
缺点：单个 RepeatUnit 向量可能压缩过多文献邻域知识；弱链接噪声作为整模态进入。

推荐场景：最小接入、消融实验、离线 KG embedding。

### 8.3 方案 C：RepeatUnit embedding 初始化或增强

不新增独立模态，将 KG embedding 用于：

- 与 SMILES CLS 表征拼接/相加；或
- 与 graph readout 拼接；或
- 作为 repeat-unit graph 的全局节点初始特征。

需要修改：对应 encoder 或其 projection 层、Dataset/DataLoader embedding lookup。  
Checkpoint 兼容：相关 projection 输入维度变化，兼容性差于独立第五模态。  
复杂度：中到高。  
优点：KG 知识更早影响结构表征。  
缺点：难以区分结构模态和 KG 模态贡献；不同注入位置需要大量消融。

推荐场景：确认第五模态有效后研究更深融合。

### 8.4 方案 D：Late fusion

```text
Uni-Poly four-modal representation
KG embedding -> MLP
concat/gated concat -> predictor
```

需要修改：Dataset/DataLoader、预测头附近新增 KG MLP；原四模态 encoder/fusion 可保持不变。  
Checkpoint 兼容：最好，四模态 backbone 可完整加载，新增 head 参数随机初始化。  
复杂度：低。  
优点：最小侵入、便于冻结 backbone、易定位 KG 增益。  
缺点：KG 无法参与早期模态交互；可能只作为额外 shortcut。

推荐场景：快速验证 KG embedding 是否含有效预测信号。

### 8.5 方案 E：Cross-attention / gated fusion

将 KG embedding 扩展为一个或多个 token，使用 molecular representation query KG tokens，或通过 confidence/coverage gate 控制 KG 注入。

需要修改：Dataset/DataLoader 支持 KG token/mask；`src/modules/uni_encoder.py` 新增 cross-attention 或 gate；可能需要邻域级 embedding 而非单向量。  
Checkpoint 兼容：四模态 encoder 可加载，fusion/cross-attention 为新参数。  
复杂度：高。  
优点：能按样本和知识覆盖程度动态控制 KG 贡献，适合多邻居知识。  
缺点：需要更复杂的输入和训练稳定性控制，解释与消融成本高。

推荐场景：后续论文主模型，前提是数据链接和 KG embedding 已经过充分验证。

### 8.6 建议定位，但不替代最终选择

- **最小可行接入候选**：方案 D，或方案 B 的单向量第五模态。
- **最适合消融实验候选**：方案 B，对比 no-KG、strict-KG、broad-KG、unknown/no-literature。
- **最适合后续论文主模型候选**：方案 E；若数据规模不足，则使用带 confidence gate 的方案 B。

开始接入前必须选择：接入位置、embedding 是否冻结、missing entity 策略、是否把 coverage mask 输入模型、checkpoint 兼容要求和预训练阶段是否启用 KG。

---

## 9. Dataset / DataLoader 对接设计

### 9.1 三级映射

每条数据记录的查找链为：

```text
record original/canonical SMILES
  -> repeat_units.csv.repeat_unit_id
  -> kg_entity_mapping.csv.kg_node_id
  -> kg_entity_mapping.csv.embedding_index
  -> kg_embedding.npy[embedding_index]
```

具体要求：

1. Dataset 初始化时加载 `repeat_units.csv` 的 canonical SMILES 映射。
2. 通过 record 的规范化 SMILES找到唯一 `repeat_unit_id`。
3. 通过 `repeat_unit_id + graph_variant + embedding_version` 找到 mapping 行。
4. 检查 `.npy` 行数、维度和 mapping index 连续性。
5. Dataset 只保存 index 或向量，不读取 Article/Measurement 等原始 KG 文件。

### 9.2 无文献链接 RepeatUnit

可选策略：

- PolymerClass fallback：使用对应 class embedding。
- learned no-literature embedding：单独的可训练向量。
- structure-only KG embedding：即使无 LiteratureSample，也由 RepeatUnit、Class、结构映射关系得到 embedding。
- mask + fallback：返回 fallback embedding，同时设置 `has_literature_link=0`。

禁止使用全零向量作为唯一策略，因为零向量可能被模型解释为数值零或产生分布偏差。推荐候选是 structure-only/class fallback + coverage mask。

### 9.3 batch 字段候选

若选择独立 KG 模态，最小字段为：

```text
kg_embedding_index
kg_mask
has_literature_link
```

也可在 Dataset 直接读取 `kg_embedding`，但 index lookup 更节省缓存空间，并方便切换 embedding 版本。`kg_mask` 表示实体映射是否有效；`has_literature_link` 表示该向量是否包含文献邻域，两者语义不同。

是否将 `has_literature_link` 作为模型 gate 输入需要实验决定；无论是否输入模型，都应保留用于 coverage 分层评估。

### 9.4 对现有四模态的隔离

- KG 文件缺失时，未启用 KG 的训练必须完全不受影响。
- Dataset 仅在显式启用新 KG 时加载 mapping 和 `.npy`。
- `smiles`、`graph`、`fp`、`geom` 的字段、缓存和 collate 逻辑保持不变。
- 新 KG 不替换 geom，不修改 PaiNN/SchNet。
- CLI 必须校验 embedding version、graph variant 和文件路径，禁止静默加载不匹配版本。

---

## 10. 实施阶段

### Stage 0：records.csv 与 repeat_units.csv

**目标**：建立数据集入口、稳定 ID 和去重 RepeatUnit。  
**输入**：`data/raw/smi_all.csv`。  
**输出**：`kg_work/records.csv`、`kg_work/repeat_units.csv`。  
**关键风险**：Polymer SMILES attachment point 规范化不稳定；不同写法误合并；`val` 泄漏到工作产物。  
**验证**：行数一致、record_id 唯一、每个 record 映射一个 RepeatUnit、canonicalization 可复现、KG 工作文件无 `val`。  
**Pilot**：必须。

### Stage 1：PolymerClass / alias mapping

**目标**：为 RepeatUnit 生成受控 PolymerClass 和 alias 候选。  
**输入**：`repeat_units.csv`、受控词表/数据库映射。  
**输出**：`polymer_class_candidates.jsonl`、`entity_aliases.csv`。  
**关键风险**：仅凭 repeat unit 过度推断；缩聚物误标共聚序列；商用名歧义。  
**验证**：候选有 method/confidence/source；PA66/PET 规则测试；人工抽查 top classes。  
**Pilot**：必须。

### Stage 2：文献检索与 chunk 召回

**目标**：按 PolymerClass/alias 合并检索，解析全文并召回可能包含 schema 事实的 chunks。  
**输入**：class/alias candidates、文献 API/本地 PDF/HTML。  
**输出**：`literature_index.jsonl`、`source_chunks.jsonl`、`candidate_chunks.jsonl`。  
**关键风险**：版权/访问限制、重复文章、PDF 顺序错误、表格丢失、召回率不足。  
**验证**：DOI 去重；chunk 顺序和定位可还原；gold facts 的 chunk recall@k；表格/补充材料覆盖率。  
**Pilot**：必须。

### Stage 3：LLM schema v2.0 局部抽取

**目标**：从候选 chunk 抽取明确事实和原文证据，不要求单 chunk 填满 JSON。  
**输入**：candidate chunks、固定 prompt、受控枚举。  
**输出**：chunk-level raw extraction JSONL、调用日志和成本统计。  
**关键风险**：幻觉、证据改写、多样品错配、全 null 占位对象、枚举越界。  
**验证**：JSON parse rate、evidence exact/normalized match、null-object rejection、人工事实 precision/recall。  
**Pilot**：必须。

### Stage 4：validation 与 LiteratureSample 聚合

**目标**：跨 chunk 对齐样品并生成 article-level validated JSON v2.0。  
**输入**：raw extractions、SourceChunk、Article metadata。  
**输出**：aggregated JSON、validated JSON、warnings、review queue。  
**关键风险**：不同样品误合并、respectively 对齐错误、单位误归一、冲突值被覆盖。  
**验证**：sample identity accuracy；每个事实 evidence 可定位；冲突值保留多记录；ID/reference 完整性。  
**Pilot**：必须。

### Stage 5：dataset_links

**目标**：建立 LiteratureSample 到 RepeatUnit/PolymerClass 的候选弱链接。  
**输入**：validated samples、repeat unit registry、class/alias registry。  
**输出**：`dataset_links.jsonl`，strict/broad link views。  
**关键风险**：同类不同样品被当作同一对象；alias 歧义；confidence 未校准。  
**验证**：按 relation_type 计算 precision/coverage；人工检查低置信链接；uncertain 不进入传播图。  
**Pilot**：必须。

### Stage 6：KG nodes / edges / triples

**目标**：将 dataset、literature、facts、provenance 和 links 统一图化。  
**输入**：records/repeat units、validated JSON、dataset links、枚举 registry。  
**输出**：`nodes.csv`、`edges.csv`、`triples.tsv`、aliases、build manifest。  
**关键风险**：重复节点、dangling edge、事实证据链断裂、strict/broad 混淆。  
**验证**：ID 唯一、edge endpoint 全存在、每个事实至少一个 Evidence、Evidence 有 Chunk/Article、图统计稳定。  
**Pilot**：必须。

### Stage 7：KG embedding

**目标**：按选定算法生成 RepeatUnit/PolymerClass embedding。  
**输入**：strict/broad KG、节点特征和 numeric representation。  
**输出**：`kg_entity_mapping.csv`、`kg_embedding.npy`、embedding manifest。  
**关键风险**：孤立节点、数值信息未被算法使用、mapping 行错位、随机性过大。  
**验证**：shape/index 校验；相同 seed 可复现；link prediction 或 node retrieval 指标；nearest-neighbor sanity check。  
**Pilot**：必须。

### Stage 8：Uni-Poly 接入

**目标**：按选定接入方案让模型读取新 embedding，保持四模态 baseline。  
**输入**：mapping、`.npy`、现有 Dataset/Model。  
**输出**：支持 KG 的可选训练路径。  
**关键风险**：checkpoint 不兼容、missing entity 处理错误、KG 默认强制加载、CV 泄漏。  
**验证**：KG disabled 时四模态行为不变；index lookup 正确；单 batch forward；旧 checkpoint `strict=False` 报告可解释。  
**Pilot**：只验证读取和 forward；正式训练需确认方案。

### Stage 9：Pilot ablation

**目标**：验证 KG 信息是否有效并定位收益来源。  
**输入**：Pilot KG embedding、选定 Uni-Poly 接入。  
**输出**：no-KG/strict/broad/numeric/provenance 等对比结果。  
**关键风险**：Pilot 样本太小、coverage 偏差、同一文献跨 fold 泄漏、结果不可归因。  
**验证**：固定 split/seed；按 coverage 分层报告；记录图版本和 embedding manifest；不使用 `val` 构图。  
**Pilot**：必须，但在文档和离线构图验收之后进行。

---

## 11. Pilot 计划

### 11.1 样本选择

选择约 50 个 representative unique RepeatUnit，不按频率机械取 Top 50。建议分层覆盖：

- homopolymer；
- condensation_multi_monomer，包括 PA66、PET 类困难案例；
- 文献明确的 copolymer；
- blend/composite；
- 简单和复杂 repeat-unit SMILES；
- 文献丰富和文献稀缺 PolymerClass；
- alias 多、结构歧义或样品对齐困难案例。

Pilot manifest 必须记录选样理由，不能使用 `val` 选样。

### 11.2 文献范围

- 每个主要 PolymerClass 选择少量高相关全文文献。
- 优先包含 Experimental、Characterization、表格和 supplementary information。
- 对同一 class 保留多篇文献，以验证 Article 去重和跨文献多 Sample 共存。
- 不跨文献合并 LiteratureSample；跨文献只通过 PolymerClass/RepeatUnit 邻域汇合。

### 11.3 人工 gold set

人工标注：

- Article 和 SourceChunk 定位；
- LiteratureSample 身份、sample label 和 alias；
- composition、sequence、architecture；
- Mn/Mw/PDI/DP；
- polymerization/preparation method 和 conditions；
- evidence sentence；
- dataset link relation type、matched_on 和 accept/reject。

### 11.4 Pilot 验收标准

最低验收建议：

1. 100% 输出可解析且通过 schema v2.0 结构校验。
2. 进入 KG 的每个事实有至少一个可定位 Evidence。
3. 不生成全 null Assertion/Measurement/Event。
4. PA66/PET 不因多单体被误标 random/block/graft。
5. LiteratureSample 合并准确率和 dataset link precision 达到预先设定阈值；阈值在 Pilot 开始前冻结。
6. strict graph 中不存在 uncertain/family-only link。
7. nodes/edges 无 dangling references，provenance 链完整。
8. mapping 与 `.npy` 行号一一对应，所有 50 个 RepeatUnit 有明确 fallback 或 embedding。
9. Uni-Poly 能读取 mapping/embedding 并完成单 batch forward；不要求本阶段启动长训练。
10. 日志、prompt、query、KG 中不存在 `val`。

建议同时报告：evidence precision、fact precision/recall、sample alignment accuracy、link precision/coverage、hallucination rate、cost per accepted fact、RepeatUnit literature coverage。

---

## 12. 开始实现前需要选择的方案

以下决策必须由项目负责人确认后再进入实现：

1. **Pilot 文献来源**：开放全文、本地 PDF、出版商 API 的使用范围。
2. **LLM 提供商与模型**：API、结构化输出能力、费用和数据合规要求。
3. **文档解析工具链**：GROBID、HTML/XML/JATS parser、PDF fallback。
4. **候选召回策略**：关键词/BM25 baseline，是否增加 embedding reranker。
5. **KG embedding baseline**：TransE、DistMult、ComplEx 或 RotatE。
6. **关系 GNN 实验**：是否在 Pilot 后实现 R-GCN/HGT。
7. **默认图视图**：strict 或 broad；broad 的 confidence threshold。
8. **连续数值方案**：属性、属性+bins 或 value encoder。
9. **provenance 节点参与范围**：全部参与 embedding，或仅用于审计/辅助训练。
10. **Uni-Poly 接入位置**：第五模态、late fusion、结构增强或 cross-attention。
11. **embedding 训练策略**：冻结、只训练 projection、或允许下游微调。
12. **无文献链接回退**：class、structure-only、learned unknown 或组合。
13. **embedding dimension 与版本策略**：例如 128/256，并与模型 joint dimension 对齐。

---

## 13. 实现 TODO

* [ ] 建立 DatasetRecord 与 RepeatUnit 注册表

  * goal: 从 `smi_all.csv` 生成稳定记录 ID、规范化 RepeatUnit 和去重映射，同时隔离 `val`。
  * files likely to add/modify: `src/kg_pipeline/records.py`, `src/kg_pipeline/repeat_units.py`, `scripts/kg/prepare_records.py`
  * output: `kg_work/records.csv`, `kg_work/repeat_units.csv`
  * validation: 行数、唯一性、canonicalization、无标签泄漏测试。

* [ ] 建立 PolymerClass 与 alias registry

  * goal: 为 RepeatUnit 和文献实体提供可审计的类别/别名候选。
  * files likely to add/modify: `src/kg_pipeline/polymer_classes.py`, `scripts/kg/map_polymer_classes.py`
  * output: `polymer_class_candidates.jsonl`, `entity_aliases.csv`
  * validation: PA66/PET 规则、mapping source/confidence、人工抽查。

* [ ] 实现文献检索与 Article 去重

  * goal: 按 class/alias 聚合检索并保存稳定 Article metadata。
  * files likely to add/modify: `src/kg_pipeline/retrieval.py`, `scripts/kg/retrieve_literature.py`
  * output: `literature_index.jsonl`, document cache
  * validation: DOI/title 去重、query trace 不含 `val`、下载状态可恢复。

* [ ] 实现文档解析与 SourceChunk

  * goal: 从 HTML/XML/PDF 生成保留章节、顺序、页码和表格来源的 chunks。
  * files likely to add/modify: `src/kg_pipeline/document_parser.py`, `scripts/kg/parse_documents.py`
  * output: `source_chunks.jsonl`
  * validation: 文本顺序、定位还原、table/caption/supplement 覆盖。

* [ ] 实现 candidate chunk recall

  * goal: 为 schema 字段召回高相关 chunks，控制 LLM 成本。
  * files likely to add/modify: `src/kg_pipeline/chunk_recall.py`, `scripts/kg/recall_chunks.py`
  * output: `candidate_chunks.jsonl`
  * validation: gold fact recall@k、按事实类型统计召回率。

* [ ] 固定 JSON schema v2.0 与 LLM extraction contract

  * goal: 定义 chunk 局部抽取、证据引用、空数组和 warning 行为。
  * files likely to add/modify: `src/kg_pipeline/schemas/`, `src/kg_pipeline/extraction.py`, `scripts/kg/extract_articles.py`
  * output: raw extraction JSONL
  * validation: schema parse、evidence match、禁止全 null fact、prompt 无 `val`。

* [ ] 实现 LiteratureSample 文档级聚合

  * goal: 按 sample label、table row、明确指代和身份证据跨 chunk 合并事实。
  * files likely to add/modify: `src/kg_pipeline/aggregation.py`
  * output: article-level aggregated JSON v2.0
  * validation: sample identity gold accuracy、冲突事实不覆盖、跨文章不合并。

* [ ] 实现 schema/evidence/enum/unit validation

  * goal: 将 raw/aggregated extraction 转为 validated JSON 和 review warnings。
  * files likely to add/modify: `src/kg_pipeline/validation.py`, `scripts/kg/validate_articles.py`
  * output: validated article JSON、warnings、review queue
  * validation: evidence chain、ID refs、枚举、单位、科学范围测试。

* [ ] 实现 dataset linking

  * goal: 生成 typed、scored、auditable LiteratureSample 到 RepeatUnit/Class 链接。
  * files likely to add/modify: `src/kg_pipeline/linking.py`, `scripts/kg/build_dataset_links.py`
  * output: `dataset_links.jsonl`, strict/broad link views
  * validation: relation-specific precision/coverage、uncertain filtering、confidence calibration。

* [ ] 实现 Polymer KG builder

  * goal: 将全部有效 JSON 对象转换为节点、属性、typed edges 和 triples。
  * files likely to add/modify: `src/kg_pipeline/graph_builder.py`, `src/kg_pipeline/numeric_features.py`, `scripts/kg/build_kg.py`
  * output: nodes、edges、triples、aliases、build manifest
  * validation: endpoint、duplicate、provenance、strict/broad graph integrity。

* [ ] 选择并实现 KG embedding baseline

  * goal: 训练选定 KGE/R-GCN/HGT 并导出 RepeatUnit/PolymerClass embedding。
  * files likely to add/modify: `src/kg_pipeline/embedding.py`, `scripts/kg/train_kg_embedding.py`
  * output: `kg_entity_mapping.csv`, `kg_embedding.npy`, embedding manifest
  * validation: shape/index、seed reproducibility、link prediction/retrieval、neighbor sanity check。

* [ ] 选择并实现 Uni-Poly KG 接入

  * goal: 在不破坏四模态 baseline 的前提下读取新 Polymer KG embedding。
  * files likely to add/modify: 选定方案后确定，预计涉及 `src/dataset/`, `src/modules/uni_encoder.py`, `scripts/pretrain.py`, `scripts/train.py`
  * output: 可选 KG 输入路径和单 batch forward
  * validation: KG disabled 回归、missing mapping、checkpoint `strict=False`、mask/coverage 行为。

* [ ] 执行 50 RepeatUnit Pilot 与消融

  * goal: 端到端验证 extraction、aggregation、linking、KG、embedding 和模型读取。
  * files likely to add/modify: `scripts/kg/run_pilot.py`, Pilot manifest/config
  * output: Pilot KG、embedding、质量报告和消融结果
  * validation: 按第 11.4 节验收标准执行。

---

## 14. 完成定义

本项目实现完成不以“成功生成一个 JSON”作为标准，而以以下条件同时满足为准：

1. `smi_all.csv` 的每个 record 可稳定映射到 RepeatUnit。
2. 每个进入 KG 的文献事实可追溯到 Evidence、SourceChunk 和 Article。
3. 不同 LiteratureSample 的事实不会因 PolymerClass 相同而被直接合并。
4. dataset link 明确区分精确结构匹配、类别匹配和弱匹配。
5. 所有有效组成、序列、架构、Measurement、Event 和 provenance 信息均进入 KG 表示。
6. `val` 未进入任何 KG 构建路径。
7. strict/broad graph 可由同一 validated source 可复现生成。
8. `kg_entity_mapping.csv` 与 `kg_embedding.npy` 完整对应。
9. 未链接文献的 RepeatUnit 有明确、可解释的 fallback。
10. Uni-Poly 在不启用 KG 时维持现有四模态行为；启用 KG 时只读取新版本化 Polymer KG embedding。

