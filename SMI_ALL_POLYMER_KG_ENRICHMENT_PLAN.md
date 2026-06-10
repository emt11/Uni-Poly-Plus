# smi_all 全量补充 Polymer KG 信息实施方案

本文档说明如何在数据处理阶段，对 `data/raw/smi_all.csv` 中的聚合物样本进行批量知识补充，并基于补充结果构建 Polymer KG。当前阶段只做技术流程设计，不直接实现代码。

## 1. 核心思路

不要对 `smi_all.csv` 每一行直接调用 LLM 补全所有字段。推荐流程是：

```text
smi_all.csv
→ 去重 RepeatUnit / Polymer SMILES
→ RepeatUnit 识别 PolymerClass / aliases
→ 按 PolymerClass 批量检索文献
→ LLM 从文献证据中抽取 JSON
→ 校验 / 标准化 / 合并
→ 回填到每条样本
→ 构建 Polymer KG
```

核心原则：

```text
LLM 只从文献或明确来源中抽取。
未找到证据时写 not mentioned。
不要让 LLM 直接编 Mn / Mw / PDI / processing / condition。
不要把待预测属性值作为输入特征，避免 label leakage。
```

## 2. 当前输入数据

输入文件：

```text
data/raw/smi_all.csv
```

当前字段：

```text
smiles,val,prop
```

一行样本表示：

```text
repeat-unit-like SMILES + property value + property type
```

示例：

```text
smiles,val,prop
*CC(*)C,0.4343,eea
*CC(*)F,0.874,eea
```

注意：`val` 是当前任务标签，不应该被 LLM enrichment 过程使用。

## 3. 总体阶段

```text
Stage 0: 输入数据整理
Stage 1: RepeatUnit 去重与规范化
Stage 2: RepeatUnit → PolymerClass 候选识别
Stage 3: PolymerClass alias 扩展
Stage 4: 定向文献检索
Stage 5: LLM 文献 JSON 抽取
Stage 6: 校验、标准化、冲突处理
Stage 7: 样本级回填
Stage 8: JSON → KG triples
```

## 4. Stage 0：输入数据整理

目标：把 `smi_all.csv` 转成稳定的 record 表。

输出建议：

```text
kg_work/records.csv
```

字段建议：

```text
record_id
smiles
prop
label_value
```

示例：

| record_id | smiles | prop | label_value |
|---|---|---|---:|
| rec_000001 | `*CC(*)C` | eea | 0.4343 |
| rec_000002 | `*CC(*)F` | eea | 0.874 |

## 5. Stage 1：RepeatUnit 去重与规范化

目标：不要按样本行查文献，而是先对唯一重复单元去重。

处理对象：

```text
unique_smiles = unique(smi_all.smiles)
```

每个重复单元生成：

```text
repeat_unit_id
raw_smiles
canonical_smiles
structure_hash
valid_rdkit_parse
formula
molecular_weight_M0
functional_groups
```

输出建议：

```text
kg_work/repeat_units.csv
```

示例：

| repeat_unit_id | raw_smiles | canonical_smiles | M0 | status |
|---|---|---|---:|---|
| ru_000001 | `*CC(*)C` | ... | ... | parsed |
| ru_000002 | `*CO*` | ... | ... | parsed |

这一阶段主要依赖规则和 RDKit，不需要 LLM。

## 6. Stage 2：RepeatUnit 到 PolymerClass 候选识别

目标：把重复单元映射到候选聚合物类。

示例 PolymerClass：

```text
PA6
PET
PS
PMMA
PEO
PVDF
Polyamide
Polyester
Polyether
Polyimide
```

推荐方法：

```text
规则识别 + 数据库匹配 + LLM 辅助候选排序
```

输出建议：

```text
kg_work/polymer_class_candidates.jsonl
```

示例字段：

| repeat_unit_id | candidate_polymer_class | aliases | confidence | method |
|---|---|---|---:|---|
| ru_x | Polyamide_6 | PA6; Nylon 6; polycaprolactam | 0.92 | rule+LLM |
| ru_y | Poly(vinylidene fluoride) | PVDF; PVF2 | 0.88 | rule |

无法确定时标记：

```text
Unknown_PolymerClass
```

## 7. Stage 3：PolymerClass Alias 扩展

目标：为文献检索准备多种名称。

示例：

```text
Polyamide 6
PA6
Nylon 6
polycaprolactam
poly(ε-caprolactam)
poly(epsilon-caprolactam)
```

输出建议：

```text
kg_work/polymer_aliases.json
```

该阶段可以使用 LLM，但高频 PolymerClass 应尽量用数据库或人工校验。

## 8. Stage 4：定向文献检索

目标：对每个 PolymerClass 检索，而不是对每条 record 检索。

Query 模板：

```text
"{alias}" polymer Mn Mw PDI
"{alias}" glass transition temperature measurement condition
"{alias}" processing method
"{alias}" tacticity chain architecture
"{alias}" dielectric constant frequency
"{alias}" refractive index density
```

如果当前样本包含特定 `prop`，可以增加属性相关 query，例如：

```text
"{alias}" Tg DSC heating rate
```

注意：不要把数据集中的 `val` 给检索或 LLM。

每个 PolymerClass 建议先保留：

```text
Top K = 3~10 篇文献
```

输出建议：

```text
kg_work/literature_index.jsonl
```

字段建议：

```text
polymer_class_id
query
title
doi
url
abstract
pdf_path
html_path
retrieval_score
```

## 9. Stage 5：LLM 文献 JSON 抽取

LLM 输入不应是孤立 SMILES，而应是：

```text
PolymerClass + alias + 文献 chunk
```

LLM 输出严格遵循 `POLYMER_KG_PLAN.md` 中的 JSON schema。

Phase 1 优先抽取：

```text
polymer_name
polymer_abbreviation
repeat_unit
monomer
monomer_ratio
copolymer_type
Mn
Mw
PDI
DP
property_measurement
measurement_condition
evidence_sentence
DOI
```

Phase 2 再抽取：

```text
tacticity
processing_method
chain_architecture
```

Prompt 必须要求：

```text
只从原文抽取。
不能推测。
没提到写 not mentioned。
每个字段必须有 evidence_sentence。
数值保留单位。
copolymer type 只能从固定枚举选择。
Mn / Mw / PDI / DP 必须区分。
```

输出建议：

```text
kg_work/extractions/raw/{polymer_class_id}/{doi_or_hash}.json
kg_work/extractions/extraction_log.csv
```

日志字段建议：

```text
polymer_class_id
doi
chunk_id
status
llm_model
timestamp
token_count
error_message
```

## 10. Stage 6：校验、标准化、冲突处理

LLM 原始结果不能直接进入 KG。

必须执行：

| 校验项 | 规则 |
|---|---|
| JSON schema | 字段完整，缺失用 `not mentioned` |
| evidence 检查 | 抽取值必须能在 evidence sentence 附近找到 |
| 单位标准化 | kg/mol → g/mol，C/K，Hz/kHz |
| PDI 检查 | 如果 Mn、Mw、PDI 都存在，验证 `PDI ≈ Mw / Mn` |
| 属性范围 | Tg、density、RI 等合理范围 |
| 枚举检查 | copolymer type、tacticity、architecture 必须归一 |
| DOI/source | 每个 fact 必须保留来源 |
| 冲突处理 | 不覆盖，保留多来源 facts |

输出建议：

```text
kg_work/facts/polymer_class_facts.csv
kg_work/facts/polymer_class_facts.parquet
```

事实表字段建议：

```text
fact_id
polymer_class_id
field
value
unit
source_doi
evidence
confidence
```

## 11. Stage 7：样本级回填

目标：把 PolymerClass facts 回填到 `smi_all.csv` 的每条 record。

### 11.1 可直接回填的结构信息

```text
polymer_class
polymer_family
aliases
functional_groups
M0
candidate_monomer
copolymer_type if verified
```

### 11.2 只能作为文献先验的信息

```text
literature_Mn_values
literature_Mw_values
literature_PDI_values
typical_processing_methods
reported_measurement_conditions
```

不要直接写成：

```text
record.Mn = 52000
```

更合理的是：

```text
record.has_literature_prior_Mn = true
record.Mn_prior_mean = ...
record.Mn_prior_std = ...
record.Mn_prior_source_count = ...
```

输出建议：

```text
kg_work/enriched_records.csv
```

字段建议：

```text
record_id
smiles
prop
label_value
repeat_unit_id
polymer_class_id
polymer_class_confidence
polymer_family
M0
kg_entity_ids_source
chain_desc_available
Mn_prior_mean
Mw_prior_mean
PDI_prior_mean
monomer_ratio_prior
copolymer_type
tacticity
chain_architecture
evidence_count
source_doi_count
```

## 12. Stage 8：构建 Polymer KG

KG 应保留 record、polymer class、事实、来源和证据。

推荐节点：

```text
DatasetRecord
RepeatUnit
PolymerClass
Monomer
CopolymerType
MolecularWeightDescriptor
PropertyMeasurement
MeasurementCondition
Article
Evidence
```

推荐关系：

```text
DatasetRecord --has_repeat_unit--> RepeatUnit
DatasetRecord --belongs_to_polymer_class--> PolymerClass
RepeatUnit --maps_to_polymer_class--> PolymerClass
PolymerClass --has_alias--> Alias
PolymerClass --has_monomer--> Monomer
PolymerClass --has_copolymer_type--> CopolymerType
PolymerClass --has_molecular_weight_descriptor--> MWD
MWD --reported_in--> Article
MWD --has_evidence--> Evidence
PropertyMeasurement --measured_under--> MeasurementCondition
PropertyMeasurement --reported_in--> Article
```

输出建议：

```text
kg_work/kg/nodes.csv
kg_work/kg/edges.csv
kg_work/kg/triples.tsv
kg_work/kg/entity_aliases.csv
kg_work/kg/evidence.csv
```

## 13. 批处理策略

不要一次性无断点全量跑 LLM。

### Pilot

```text
50 个 unique repeat units
每个 PolymerClass Top 3 文献
```

验证目标：

```text
polymer class 映射准确率
文献检索有效率
LLM JSON 合法率
字段覆盖率
```

### Batch 1

```text
Top 200 高频 repeat units
```

重点检查常见类别：

```text
PA / PE / PP / PS / PMMA / PET / PI / PEO / PVDF
```

### Full Run

```text
全部 unique smiles
```

每个阶段保存断点状态：

```text
repeat_unit_identified
polymer_class_linked
literature_retrieved
llm_extracted
validated
triples_generated
```

失败样本输出到：

```text
kg_work/errors/
```

## 14. 风险控制

### 14.1 不要让 LLM 直接补事实

错误方式：

```text
Given *CC(*)C, tell me Mn/Mw/PDI/processing method.
```

正确方式：

```text
Given this paper text, extract explicitly mentioned Mn/Mw/PDI/processing method for polymer X.
```

### 14.2 防止 Label Leakage

如果当前 record 的 `prop = tg`，不要把文献抽取到的 `Tg value` 作为该 record 输入特征。

可以保存到 KG 作为事实，但训练时要区分：

```text
target property value: 不进入输入
measurement condition / source / polymer class: 可进入输入
```

### 14.3 PolymerClass Prior 不等于 Sample Fact

文献中的 PA6 Mn/Mw/PDI 是文献样品的事实，不一定是当前数据集中该 PA6 样本的事实。

模型输入建议使用：

```text
prior statistics
source count
confidence
```

不要把某个文献值当作当前样本的确定值。

## 15. 最小可行版本

第一版建议只做：

```text
unique smiles
→ repeat unit normalization
→ polymer class candidates
→ alias expansion
→ 每类检索 Top 3 文献
→ LLM 抽取 PolymerClass / Mn / Mw / PDI / condition / monomer / copolymer type
→ 规则校验
→ 回填 enriched_records.csv
→ 生成 nodes.csv / edges.csv / triples.tsv
```

优先字段：

```text
PolymerClass
Alias
Monomer
CopolymerType
Mn
Mw
PDI
DP
MeasurementCondition
Article
Evidence
```

第一版不要急着抽：

```text
complex processing details
structure-property rules
sequence distribution
```

## 16. 最终产物清单

建议最终生成：

```text
kg_work/records.csv
kg_work/repeat_units.csv
kg_work/polymer_class_candidates.jsonl
kg_work/polymer_aliases.json
kg_work/literature_index.jsonl
kg_work/extractions/raw/*.json
kg_work/facts/polymer_class_facts.csv
kg_work/enriched_records.csv
kg_work/kg/nodes.csv
kg_work/kg/edges.csv
kg_work/kg/triples.tsv
kg_work/kg/evidence.csv
```

进入模型的主要文件：

```text
enriched_records.csv
kg entity mapping
numeric descriptor matrix
```

进入 KG 的主要文件：

```text
nodes.csv
edges.csv
triples.tsv
evidence.csv
```

## 17. 总结

该操作可以实现，但推荐路线是：

```text
unique repeat unit
→ polymer class
→ 文献
→ JSON 抽取
→ 校验
→ 回填样本
→ KG
```

不要采用：

```text
每一行 SMILES
→ LLM 直接补全部字段
```

后者成本高、噪声大，也容易产生不可控伪知识。
