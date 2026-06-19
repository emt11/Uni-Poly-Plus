# LLM_KG.MD

本文是 Polymer KG 的语义与开发设计文档，不是运行手册。实际命令请看 [RUN.MD](RUN.MD)。

## 0. 实现模块边界

KG 语义实现主要分布在：

```text
src/kg_pipeline/dataset.py
src/kg_pipeline/literature.py
src/kg_pipeline/llm_extract.py
src/kg_pipeline/article_validation.py
src/kg_pipeline/kg_build.py
src/kg_pipeline/kg_embedding.py
```

旧细粒度 wrapper 已删除。语义设计变更应优先落到这些新模块中。

## 1. 目标

从 `data/raw/smi_all.csv` 中的 repeat-unit-like SMILES 建立 `DatasetRecord -> RepeatUnit` 入口；从文献中抽取带证据的样品级事实；通过弱链接把文献知识接入 Polymer KG；最终导出 KG embedding 作为 Uni-Poly 的可选第五模态 `kg`。

端到端语义流程：

```text
DatasetRecord
-> RepeatUnit
-> retrieval candidates
-> Article / SourceChunk / Evidence
-> LiteratureSample facts
-> dataset_links
-> KG nodes / edges / triples
-> KG embedding
```

## 2. 不可变边界

- `LiteratureSample` 不等于 `DatasetRecord`。
- 文献中的 Mn、Mw、PDI、DP、组成比例和条件不能回填为当前 dataset row 的真值。
- `val` 是监督标签，不得进入：
  - LLM prompt
  - retrieval query
  - mapping
  - dataset linking
  - KG nodes / edges / triples
  - KG embedding
  - manifest 中的特征输入
- 当前不生成 `enriched_records.csv`。
- 当前不生成独立 `numeric_descriptor_matrix`。
- SourceChunk 可用于 provenance，但默认不进入 TransE triples。
- KG embedding 是新 `kg` 模态输入，不是旧 KG，也不是 TEXT 模态。

## 3. 当前 mapping 语义

`kg_work/polymer_class_candidates.jsonl` 当前是 RepeatUnit 的检索候选文件，不是 curated polymer registry。字段为：

```json
{
  "repeat_unit_id": "...",
  "raw_smiles": "...",
  "canonical_smiles": "...",
  "aliases": ["..."],
  "confidence": 0.0
}
```

设计要求：

- `aliases` 只用于文献检索和 candidate recall。
- `aliases` 不能直接视为 verified polymer name。
- 只靠 RU SMILES 不能保证唯一命名聚合物；后续开发应将 mapping 从“准确命名”转向“检索候选 + 证据验证”。
- 真正进入 KG 的 verified identity 应依赖文献证据、结构/单体/连接键一致性和 validation。

## 4. JSON Schema v2.0

LLM 文献抽取统一使用 schema v2.0。顶层结构：

```json
{
  "schema_version": "2.0",
  "article": {},
  "evidence_records": [],
  "literature_samples": [],
  "dataset_links": [],
  "warnings": []
}
```

### article

来源文章元数据。代表文章，不代表样品。

关键字段：

```text
article_id, title, doi, year, journal
```

### evidence_records

事实证据。每条事实必须能追踪到原文句子。

关键字段：

```text
evidence_id, chunk_id, section, page, paragraph_id, sentence
```

证据句应尽量保持原文，不要让 LLM 改写。

### literature_samples

文章中的具体样品、film、blend、composite 或材料体系。一篇文章可有多个 LiteratureSample。

关键身份字段：

```text
sample_id, sample_label, polymer_name, polymer_class, aliases, repeat_unit_smiles, identity_evidence_refs
```

`repeat_unit_smiles` 只有在原文明确给出或可靠结构工具得到时才填写，禁止 LLM 凭名称生成。

### composition_assertions

样品级组成事实。推荐枚举：

```text
composition_type:
homopolymer, condensation_multi_monomer, copolymer, terpolymer,
blend, composite, mixture, unknown

ratio_basis:
mol_fraction, weight_fraction, feed_ratio, actual_ratio, stoichiometric_ratio
```

components 的 role 推荐：

```text
monomer, comonomer, polymer_component, filler, additive, solvent, unknown
```

### sequence_distribution_assertions

共聚单元或链段排列方式。未明确表达时不推断。

推荐枚举：

```text
random, statistical, alternating, block, blocky, multiblock,
gradient, graft, periodic, not_applicable, unknown
```

### chain_architecture_assertions

链架构事实。未明确表达时不推断。

推荐枚举：

```text
linear, branched, star, comb, brush, graft, network, crosslinked, unknown
```

### molecular_weight_measurements

样品级 Mn、Mw、dispersity/PDI、DP。必须有 evidence_refs。Mn/Mw/DP 应为正值，dispersity 应合理。

### polymerization_events

聚合或制备事件，包括 method 和 conditions。

conditions 可包含：

```text
temperature, time, pressure, solvent, atmosphere, pH
```

### dataset_links

LiteratureSample 到 RepeatUnit 的候选弱链接。它不是样品同一性声明。

关键字段：

```text
link_id, literature_sample_id, target_repeat_unit, relation_type,
matched_on, confidence, evidence_refs
```

## 5. 证据链

推荐证据链：

```text
Fact -> Evidence -> SourceChunk -> Article
```

要求：

- 每个 assertion、measurement、event、identity 都应引用 evidence。
- evidence sentence 必须能在 SourceChunk text 中定位。
- 缺证据或证据定位失败的事实不得进入 KG。
- ambiguous / unsupported / conflicting 信息进入 `warnings` 或 review queue。

## 6. Label Leakage 原则

禁止路径：

```text
val -> prompt
val -> retrieval query
val -> mapping
val -> dataset link
val -> KG triples
val -> KG embedding
```

如果文献抽取到的 property 与当前 downstream `prop` 相同，应至少标记 `possible_label_leakage`；更保守的实现应默认屏蔽进入 triples。

## 7. KG 节点设计

核心节点类型：

```text
DatasetRecord
RepeatUnit
Article
SourceChunk
Evidence
LiteratureSample
CompositionAssertion
SequenceDistributionAssertion
ChainArchitectureAssertion
MolecularWeightMeasurement
PolymerizationEvent
PolymerClass / PolymerFamily / Alias（可选，取决于后续实现）
NumericBin（可选）
```

当前 mapping 文件不是 verified class registry，因此由 mapping aliases 生成的节点/边应保持 broad/review 语义，不应作为强事实。

## 8. KG 边设计

推荐关系：

```text
DatasetRecord --has_repeat_unit--> RepeatUnit
LiteratureSample --reported_in--> Article
LiteratureSample --supported_by--> Evidence
Evidence --from_chunk--> SourceChunk
LiteratureSample --has_composition_assertion--> CompositionAssertion
LiteratureSample --has_sequence_distribution_assertion--> SequenceDistributionAssertion
LiteratureSample --has_chain_architecture_assertion--> ChainArchitectureAssertion
LiteratureSample --has_molecular_weight_measurement--> MolecularWeightMeasurement
LiteratureSample --has_polymerization_event--> PolymerizationEvent
LiteratureSample --candidate_link_to--> RepeatUnit
```

SourceChunk 可保留在 nodes/edges 中用于 provenance，但默认不作为 TransE triples endpoint。

## 9. Linking 设计

链接强度分级：

```text
verified:
  文献明确给出 repeat unit / 结构证据，并能与当前 RU 对齐。

probable:
  monomer + linkage + polymerization route 与当前 RU 一致。

broad:
  只有 polymer family / class / alias 命中。

rejected:
  名称命中但结构、单体或证据不匹配。
```

聚合物领域文献通常不会给 SMILES，因此后续开发不应依赖 SMILES exact match。更现实的验证证据包括：

- polymer name / abbreviation
- monomer names
- linkage type
- polymerization route
- composition / ratio
- NMR / FTIR / GPC / DSC / TGA 等表征证据
- figure / scheme 中的结构信息（当前未完整实现）

## 10. KG Embedding 设计

输入：

```text
kg/nodes.csv
kg/edges.csv
kg/triples.tsv
repeat_units.csv
```

输出：

```text
features/kg_entity_mapping.csv
features/kg_embedding.npy
features/embedding_manifest.json
```

要求：

- `kg_embedding.npy` dtype 为 `float32`。
- `kg_entity_mapping.csv.embedding_index` 从 0 连续。
- `kg_embedding.npy.shape[0] == len(kg_entity_mapping.csv)`。
- `kg_embedding.npy.shape[1] == --embedding_dim`。
- 没有 validated literature links 时可以导出 test-only embedding，但 manifest 应标记 `test_only=true`。

## 11. TODO

优先级较高的后续开发：

- 将 RU mapping 从“名称生成”改为“检索候选 + structural keywords + risk 标记”。
- 增加结构特征提取：ester、amide、urethane、carbonate、ether、siloxane、fluorinated、aromatic 等。
- retrieval query 使用 alias + structural keywords 组合，减少错误 alias 扩散。
- 文献抽取增加 monomer、linkage、route、structure evidence 字段。
- dataset linking 增加强制证据门控：只有名称命中不足以 verified。
- validation 加强 JSON schema、证据定位和结构一致性检查。
- KG builder 明确区分 strict/broad/review/rejected 边。
- embedding manifest 继续记录 test_only、link count、triple count、fallback count。
- 保持 `val` leakage scanner。
