# LLM + Polymer KG 完整需求与实现方案

本文档合并并重构以下三份文档的核心内容：

```text
POLYMER_KG_PLAN.md
SMI_ALL_POLYMER_KG_ENRICHMENT_PLAN.md
LLM + KG.md
```

目标是为 Uni-Poly-Plus 项目设计一套可落地的 **LLM 文献抽取 + 数据库补充 + Polymer KG + 数值 descriptor** 方案，用于增强当前聚合物属性预测模型。

本文档不是只讨论 KG 文件格式，而是覆盖完整闭环：

```text
需求定义
→ 数据集现状
→ KG 建模原则
→ 目标字段与粗粒度策略
→ 数据来源
→ LLM 文献抽取
→ 校验与合并
→ 样本级回填
→ KG 构建
→ 模型接入
→ 实施计划
```

---

## 1. 项目目标

当前 Uni-Poly-Plus 的核心输入数据是：

```text
data/raw/smi_all.csv
```

字段为：

```text
smiles,val,prop
```

其中：

| 字段 | 含义 |
|---|---|
| `smiles` | repeat-unit-like Polymer SMILES |
| `val` | 当前任务标签值 |
| `prop` | 任务名 |

当前数据规模：

| 项目 | 数量 |
|---|---:|
| 样本行数 | 13,408 |
| unique SMILES | 9,702 |
| 属性任务数 | 9 |
| 任务分布 | `tg` 7143, `egc` 3380, `egb` 561, `xc` 432, `eat` 390, `nc` 382, `eps` 382, `ei` 370, `eea` 368 |

目标是基于当前数据集中的 repeat-unit-like SMILES，补充以下链级、样品级和文献级知识：

```text
copolymer type
monomer ratio
sequence distribution
chain architecture
Mn / Mw / PDI / DP
processing method
```

这些信息最终用于：

1. 构建可追溯 Polymer KG。
2. 生成 record-level enriched table。
3. 生成 KG embedding。
4. 生成 numeric descriptor matrix。
5. 接入 Uni-Poly 多模态模型并做 ablation。

---

## 2. 核心判断

### 2.1 当前数据不是简单“全均聚物”数据

`smi_all.csv` 中绝大多数 SMILES 是单一 repeat-unit-like 表示，且大多数含两个 `*` 连接位点。这说明数据格式主要是：

```text
一个重复单元 + 两个聚合连接位点
```

但这不等价于全部都是均聚物。

例如 PA66、聚酯、聚酰胺、聚氨酯、聚酰亚胺等都可以被表示为一个重复单元，但它们可能来自两个或多个单体。

因此需要区分：

```text
single repeat-unit representation
!=
homopolymer
```

建议新增两个概念：

```text
repeat_unit_representation_type
polymer_composition_type
```

示例：

| 字段 | 示例值 |
|---|---|
| `repeat_unit_representation_type` | `single_cru`, `multi_connection_cru`, `unknown` |
| `polymer_composition_type` | `homopolymer`, `condensation_multi_monomer`, `copolymer`, `blend`, `composite`, `unknown` |

### 2.2 RepeatUnit 不能承载全部链级事实

同一个 repeat unit 在不同文献、不同合成条件、不同加工条件下可以对应不同样品。不同样品可能有不同：

```text
monomer ratio
sequence distribution
chain architecture
Mn / Mw / PDI / DP
processing method
```

因此不能这样建：

```text
RepeatUnit_001 --Mn--> 52000
RepeatUnit_001 --processing_method--> melt_extrusion
RepeatUnit_001 --monomer_ratio--> 1:1
```

正确方式是引入：

```text
LiteratureSample
```

即：

```text
RepeatUnit
→ PolymerClass
→ LiteratureSample
→ Composition / ChainStructure / MolecularWeight / Processing
→ Article / Evidence
```

### 2.3 文献事实是 prior，不是当前样本真实值

`smi_all.csv` 没有样品编号、分子量、加工条件、测量条件。文献抽取到的事实只能视为：

```text
literature-derived prior
```

不能直接写成：

```text
record.Mn = 52000
record.processing_method = melt_extrusion
```

应回填为统计先验：

```text
Mn_prior_mean
Mn_prior_std
Mn_prior_source_count
processing_method_distribution
chain_architecture_mode
evidence_count
confidence_mean
```

---

## 3. 总体路线

优先采用目标驱动路线，而不是先构建全领域大 KG：

```text
data/raw/smi_all.csv
→ unique repeat-unit-like SMILES
→ RepeatUnit 规范化
→ RepeatUnit 到 PolymerClass 候选映射
→ PolymerClass alias 扩展
→ 数据库补充：PoLyInfo / Polymer Genome / Polymer Scholar
→ 定向文献检索
→ 文献解析与 chunk 过滤
→ LLM event-level JSON 抽取
→ 校验 / 标准化 / 冲突保留
→ LiteratureSample 与 Event 建模
→ 样本级 prior 回填
→ KG nodes / edges / triples
→ KG embedding + numeric descriptor matrix
→ Uni-Poly ablation
```

不采用：

```text
每一行 SMILES
→ LLM 直接补 copolymer type / Mn / Mw / PDI / processing
```

原因：

1. 成本高。
2. 噪声大。
3. 容易幻觉。
4. 容易把文献样品事实误当当前样本事实。
5. 容易造成 label leakage。

---

## 4. 目标 KG 字段与粗粒度策略

用户指定的 KG 结构字段如下：

```text
copolymer type
monomer ratio
sequence distribution
chain architecture
Mn / Mw / PDI / DP
processing method
```

本文档建议将这些字段分成四类：

| 类型 | 字段 | 推荐承载方式 |
|---|---|---|
| 离散结构类 | `copolymer type`, `sequence distribution`, `chain architecture` | KG 枚举实体 + relation |
| 数值组成类 | `monomer ratio` | KG CompositionEvent + numeric descriptor |
| 数值链级类 | `Mn / Mw / PDI / DP` | KG MolecularWeightDescriptor + numeric descriptor |
| 过程事件类 | `processing method` | KG ProcessingEvent + 条件属性 |

## 4.1 copolymer type

### 4.1.1 建议字段拆分

不要把所有内容都塞进一个 `copolymer_type`。建议拆成：

```text
polymer_composition_type
copolymer_sequence_type
```

`polymer_composition_type` 表示材料体系层面的组成类型：

```text
homopolymer
condensation_multi_monomer
copolymer
terpolymer
blend
composite
mixture
unknown
not_mentioned
```

`copolymer_sequence_type` 表示共聚物链上不同单元的排列方式：

```text
random_copolymer
alternating_copolymer
block_copolymer
gradient_copolymer
graft_copolymer
statistical_copolymer
multiblock_copolymer
not_applicable
unknown
not_mentioned
```

### 4.1.2 为什么要拆分

例如：

| 文献描述 | `polymer_composition_type` | `copolymer_sequence_type` |
|---|---|---|
| PA6 | `homopolymer` 或 `condensation_multi_monomer`，取决于本体定义 | `not_applicable` |
| PA66 | `condensation_multi_monomer` | `not_applicable` |
| poly(styrene-random-methyl methacrylate) | `copolymer` | `random_copolymer` |
| PS-b-PMMA | `copolymer` | `block_copolymer` |
| polymer blend | `blend` | `not_applicable` |
| silica/PA6 nanocomposite | `composite` | `not_applicable` |

如果只用一个 `copolymer_type`，会把 `blend`、`composite`、`block copolymer`、`condensation polymer` 混在一起。

### 4.1.3 LLM 输出约束

LLM 只能从固定枚举中选：

```json
{
  "polymer_composition_type": {
    "value": "homopolymer|condensation_multi_monomer|copolymer|terpolymer|blend|composite|mixture|unknown|not_mentioned",
    "evidence": "..."
  },
  "copolymer_sequence_type": {
    "value": "random_copolymer|alternating_copolymer|block_copolymer|gradient_copolymer|graft_copolymer|statistical_copolymer|multiblock_copolymer|not_applicable|unknown|not_mentioned",
    "evidence": "..."
  }
}
```

规则：

```text
如果原文没有明确说 random/block/graft/alternating/gradient，不允许 LLM 推测。
如果来自数据库或 ontology rule，source_type 必须标记为 database 或 ontology_rule。
如果只是由 repeat unit 结构推断，应标记为 inferred_by_rule，并降低 confidence。
```

## 4.2 monomer ratio

### 4.2.1 是否适合 KG

适合，但同时应进入 numeric descriptor。

KG 中建议建：

```text
CompositionEvent
```

不要只用一条边：

```text
Polymer --has_monomer_ratio--> 0.7
```

因为 ratio 必须和 monomer、ratio type、单位、来源绑定。

### 4.2.2 建议 schema

```json
{
  "composition_event": {
    "event_id": "comp_001",
    "materials": ["sample_001"],
    "components": [
      {
        "monomer_name": "styrene",
        "monomer_abbreviation": "St",
        "ratio_value": 0.7,
        "ratio_unit": "mol_fraction",
        "ratio_type": "feed_ratio|actual_ratio|weight_fraction|molar_ratio|not_mentioned"
      },
      {
        "monomer_name": "methyl methacrylate",
        "monomer_abbreviation": "MMA",
        "ratio_value": 0.3,
        "ratio_unit": "mol_fraction",
        "ratio_type": "feed_ratio|actual_ratio|weight_fraction|molar_ratio|not_mentioned"
      }
    ],
    "evidence": "...",
    "source": "..."
  }
}
```

### 4.2.3 ratio_type 约束

必须区分：

```text
feed_ratio
actual_ratio
molar_ratio
weight_fraction
volume_fraction
not_mentioned
unknown
```

不要把 feed ratio 直接当作实际链组成。

## 4.3 sequence distribution

### 4.3.1 是否适合 v1

价值高，但不建议作为 v1 的强制字段。

原因：

1. 文献不一定明确给出。
2. 表达复杂。
3. 很多情况下只有粗粒度词，如 random、block、alternating。
4. 完整序列分布需要 reactivity ratio、NMR、sequence statistics 或聚合模型。

### 4.3.2 v1 粗粒度策略

v1 只做粗粒度枚举：

```text
random
alternating
blocky
block
gradient
statistical
periodic
not_mentioned
unknown
```

`sequence_distribution` 不进入强数值模型特征，只作为 KG relation 或 metadata。

### 4.3.3 v2 扩展

v2 可增加：

```text
dyad_fraction
triad_fraction
reactivity_ratio_r1
reactivity_ratio_r2
block_length_distribution
sequence_evidence_method
```

但这需要专门文献抽取和更强校验，不建议在第一版做。

## 4.4 chain architecture

### 4.4.1 是否适合 KG

适合进 KG。

建议枚举：

```text
linear
branched
star
graft
comb
brush
network
crosslinked_network
hyperbranched
dendrimer
cyclic
ladder
unknown
not_mentioned
```

### 4.4.2 注意事项

`chain architecture` 不应默认从 repeat unit 推断。

例如普通 PA6 通常可视为 linear，但如果文献没有说明，LLM 抽取应写：

```text
not_mentioned
```

如果来自规则/数据库，可写：

```text
source_type = ontology_rule
confidence = medium
```

## 4.5 Mn / Mw / PDI / DP

### 4.5.1 是否适合 KG

适合进 KG，但不能只用 KG embedding。它们是连续数值，必须同时进入 numeric descriptor branch。

KG 中建议建：

```text
MolecularWeightDescriptor
```

示例：

```text
LiteratureSample_001 --has_molecular_weight_descriptor--> MWD_001
MWD_001 --Mn--> 52000 g/mol
MWD_001 --Mw--> 104000 g/mol
MWD_001 --PDI--> 2.0
MWD_001 --DP--> not_mentioned
MWD_001 --reported_in--> Article_001
MWD_001 --has_evidence--> Evidence_001
```

### 4.5.2 字段规范

```json
{
  "molecular_weight_descriptor": {
    "Mn": { "value": "not_mentioned", "unit": "g/mol", "normalized_value": "not_mentioned" },
    "Mw": { "value": "not_mentioned", "unit": "g/mol", "normalized_value": "not_mentioned" },
    "PDI": { "value": "not_mentioned", "unit": "dimensionless", "derived": false },
    "DP": { "value": "not_mentioned", "unit": "dimensionless", "normalized_value": "not_mentioned" },
    "method": "GPC|SEC|MALDI|viscometry|not_mentioned",
    "calibration": "polystyrene standard|not_mentioned",
    "evidence": "..."
  }
}
```

### 4.5.3 PDI 派生规则

如果文献或数据库只有 `Mn` 和 `Mw`，没有 `PDI`，可以派生：

```text
PDI = Mw / Mn
```

但必须标记：

```text
PDI.derived = true
PDI.derivation = "Mw/Mn"
source_type = "computed"
```

不要把派生值伪装成文献原始报告值。

## 4.6 processing method

### 4.6.1 是否适合 KG

适合，而且建议建成事件节点：

```text
ProcessingEvent
```

因为加工方法往往包含：

```text
method
temperature
time
pressure
solvent
atmosphere
equipment
annealing
stretching
curing
```

### 4.6.2 建议枚举

`processing_method` 粗粒度枚举：

```text
solution_casting
spin_coating
drop_casting
melt_extrusion
injection_molding
compression_molding
hot_pressing
annealing
thermal_treatment
solvent_annealing
electrospinning
3d_printing
film_drawing
stretching
curing
crosslinking
not_mentioned
unknown
```

### 4.6.3 schema

```json
{
  "processing_event": {
    "method": "solution_casting",
    "temperature": { "value": 80, "unit": "C" },
    "time": { "value": 12, "unit": "h" },
    "pressure": "not_mentioned",
    "solvent": "DMF",
    "atmosphere": "nitrogen",
    "equipment": "not_mentioned",
    "evidence": "..."
  }
}
```

---

## 5. KG 层级设计

### 5.1 为什么必须引入 LiteratureSample

同一个 PolymerClass 或 RepeatUnit 可能对应多个文献样品：

```text
PA6 sample A: Mn = 20000, melt pressed
PA6 sample B: Mn = 80000, solution cast
PA6 composite C: PA6 + clay, annealed
```

如果把它们都挂到 `PolymerClass_PA6`，会混淆样品事实。

因此 KG 中必须有：

```text
LiteratureSample
```

该节点表示：

```text
某篇文献中被明确描述的一个聚合物样品、材料体系、film、blend、composite 或 sample label。
```

### 5.2 核心节点类型

v1 节点类型：

```text
DatasetRecord
RepeatUnit
PolymerClass
Alias
LiteratureSample
Monomer
CompositionEvent
CopolymerType
SequenceDistribution
ChainArchitecture
MolecularWeightDescriptor
ProcessingEvent
Article
Evidence
SourceChunk
DatabaseSource
```

可选节点类型：

```text
Property
PropertyMeasurementEvent
MeasurementCondition
PolymerFamily
FunctionalGroup
```

虽然用户本次指定的 KG 结构不包含 measurement condition，但保留 `PropertyMeasurementEvent` 和 `MeasurementCondition` 对防止 label leakage、后续属性预测解释很有用。

### 5.3 核心关系类型

```text
DatasetRecord --has_repeat_unit--> RepeatUnit
RepeatUnit --maps_to_polymer_class--> PolymerClass
PolymerClass --has_alias--> Alias
PolymerClass --has_literature_sample--> LiteratureSample
LiteratureSample --belongs_to_polymer_class--> PolymerClass
LiteratureSample --has_composition--> CompositionEvent
CompositionEvent --has_monomer--> Monomer
CompositionEvent --has_ratio--> RatioValue
LiteratureSample --has_copolymer_type--> CopolymerType
LiteratureSample --has_sequence_distribution--> SequenceDistribution
LiteratureSample --has_chain_architecture--> ChainArchitecture
LiteratureSample --has_molecular_weight_descriptor--> MolecularWeightDescriptor
LiteratureSample --processed_by--> ProcessingEvent
MolecularWeightDescriptor --reported_in--> Article
ProcessingEvent --reported_in--> Article
CompositionEvent --reported_in--> Article
LiteratureSample --reported_in--> Article
Evidence --from_chunk--> SourceChunk
AnyFact --has_evidence--> Evidence
AnyFact --derived_from_database--> DatabaseSource
```

### 5.4 示例

```text
DatasetRecord_rec_000001 --has_repeat_unit--> RepeatUnit_ru_000001
RepeatUnit_ru_000001 --maps_to_polymer_class--> PolymerClass_PA66
PolymerClass_PA66 --has_alias--> Alias_Nylon_66

PolymerClass_PA66 --has_literature_sample--> LiteratureSample_art1_sampleA
LiteratureSample_art1_sampleA --has_composition--> Composition_art1_sampleA
Composition_art1_sampleA --has_monomer--> Monomer_hexamethylenediamine
Composition_art1_sampleA --has_monomer--> Monomer_adipic_acid
Composition_art1_sampleA --ratio_type--> stoichiometric_ratio
Composition_art1_sampleA --ratio_value--> 1:1

LiteratureSample_art1_sampleA --has_molecular_weight_descriptor--> MWD_art1_sampleA
MWD_art1_sampleA --Mn--> 52000 g/mol
MWD_art1_sampleA --Mw--> 104000 g/mol
MWD_art1_sampleA --PDI--> 2.0

LiteratureSample_art1_sampleA --processed_by--> Processing_art1_sampleA
Processing_art1_sampleA --method--> melt_extrusion
Processing_art1_sampleA --temperature--> 260 C

MWD_art1_sampleA --reported_in--> Article_001
MWD_art1_sampleA --has_evidence--> Evidence_001
```

---

## 6. 数据来源设计

### 6.1 PoLyInfo

PoLyInfo 适合用于：

```text
PolymerClass
Alias / IUPAC name
Monomer
Component composition
Homopolymer / Copolymer / Blend / Composite
Mn / Mw / DP
Processing information
Measurement condition
Reference metadata
```

需要注意：

```text
PDI 不一定是独立字段，可由 Mw/Mn 派生。
sequence distribution 和 chain architecture 可能只是部分覆盖。
copolymer type 的细粒度枚举需要具体样本页面确认。
PoLyInfo 通常提供 reference，不等于 evidence sentence。
```

PoLyInfo 不能替代文献抽取，因为：

1. 不一定提供原文证据句。
2. 细粒度字段覆盖不一定完整。
3. 需要遵守其使用条款，不能无授权大规模抓取。

### 6.2 Polymer Scholar

适合用于：

```text
polymer-property-value-unit records
property prior
文献来源辅助检索
```

但 Polymer Scholar 更偏 property records，不一定提供完整链级结构。

### 6.3 文献

文献是 evidence sentence 的主要来源。

优先级：

| 优先级 | 来源 | 理由 |
|---:|---|---|
| 1 | publisher HTML/XML / JATS | 结构清晰，段落/表格好解析 |
| 2 | PMC / arXiv / open access HTML | 可自动化 |
| 3 | PDF | 可用但表格和符号解析不稳定 |
| 4 | Abstract | 只能作为低置信来源 |

### 6.4 LLM 的作用

LLM 不负责“凭常识补全”。LLM 只负责：

```text
从给定文献 chunk 中抽取显式事实
输出严格 JSON
给出 evidence sentence
标记不确定关系
```

---

## 7. 三篇论文对方案的作用

### 7.1 Gupta 2024

Gupta 等提出大规模 polymer-property 抽取流程：

```text
全文文章
→ paragraph
→ property-specific heuristic filter
→ NER filter
→ GPT-3.5 / MaterialsBERT extraction
→ validation
→ relational database
```

对本方案的启发：

1. 不要把整篇文章直接输入 LLM。
2. 先用关键词/NER 过滤 candidate chunks。
3. LLM 只处理高信息密度 chunk。
4. value/unit/range/evidence 必须校验。
5. 成本控制是 pipeline 设计的一等目标。

但 Gupta 的局限是：

```text
主要抽 paragraph-level property records。
没有完整解决跨段落 sample reconstruction。
```

因此本方案在 Gupta 的基础上增加：

```text
article-level entity memory
LiteratureSample
event-level extraction
cross-chunk event alignment
```

### 7.2 PolyIE 2023

PolyIE 的核心贡献是：

```text
Material / Property / Value / Condition
N-ary relation extraction
```

对本方案的启发：

1. 不要把关系拆成孤立字段。
2. 对 Composition、MWD、Processing、Measurement 都应建 event。
3. 每个 event 必须绑定材料、数值、条件、来源和证据。
4. 多材料、多数值、respectively、跨句关系需要专门校验。

### 7.3 Li 2026

Li 等强调：

```text
knowledge reconstruction
JSON schema
human feedback
fine-tuning
RDF / KG
```

对本方案的启发：

1. LLM 输出 schema 应天然映射到 KG。
2. v1 先做 prompt + 人工 pilot，不急于微调。
3. 积累高质量 JSON 后再考虑 fine-tune 小模型。
4. KG 应支持查询，不只是生成 embedding。

---

## 8. 文档级抽取策略

### 8.1 为什么不能只按段落补全全部字段

一篇文献中，信息可能分布为：

```text
Introduction: polymer class / alias
Experimental: monomer, synthesis, processing
Characterization: Mn / Mw / PDI
Results: property measurement
Tables: composition / molecular weight / processing
Supplementary: detailed sample conditions
```

因此不能要求一个段落包含全部字段。

应采用：

```text
chunk-level event extraction
→ article-level fact pool
→ LiteratureSample alignment
→ PolymerClass prior aggregation
```

### 8.2 chunk 设计

文献解析后生成：

```text
paragraphs
tables
captions
section metadata
page metadata
```

候选 chunk 不只是一个段落，而是：

```text
anchor paragraph + 前后 1-2 个 paragraph
```

字段：

```json
{
  "chunk_id": "chunk_000001",
  "article_id": "art_000001",
  "doi": "...",
  "section": "Experimental",
  "page": 5,
  "anchor_paragraph_id": 12,
  "context_paragraph_ids": [10, 11, 12, 13, 14],
  "text": "...",
  "candidate_event_types": ["molecular_weight", "processing"]
}
```

### 8.3 article-level memory

每篇文章先建立实体记忆：

```json
{
  "article_id": "art_001",
  "polymer_entities": [
    {
      "canonical_name": "polyamide 6",
      "aliases": ["PA6", "nylon 6"],
      "first_defined_in_chunk": "chunk_003"
    }
  ],
  "sample_entities": [
    {
      "sample_label": "PA6-1",
      "description": "PA6 film annealed at 120 C",
      "defined_in_chunk": "chunk_006"
    }
  ]
}
```

用途：

1. 处理缩写。
2. 处理 `the sample`、`this copolymer`、`the resulting film`。
3. 辅助跨 chunk 合并。

### 8.4 候选 chunk 分类

按目标字段分类：

```text
identity_chunk
composition_chunk
sequence_chunk
architecture_chunk
molecular_weight_chunk
processing_chunk
```

关键词示例：

```text
composition:
  monomer, feed ratio, mol%, wt%, composition, copolymerized

sequence:
  random, alternating, block, gradient, statistical, sequence distribution

architecture:
  linear, branched, star, graft, comb, network, crosslinked

molecular_weight:
  Mn, Mw, M_n, M_w, PDI, dispersity, Đ, DP, GPC, SEC

processing:
  spin coating, casting, extrusion, molding, annealing, hot pressing,
  solvent, temperature, pressure, curing, stretching
```

---

## 9. LLM JSON 抽取 schema

LLM 输入：

```text
PolymerClass candidate
aliases
article metadata
article-level memory
chunk text
target event types
strict JSON schema
```

LLM 不允许输入：

```text
record label_value
当前样本 val
要求模型猜测当前样品真实 Mn/Mw/PDI 的提示
```

### 9.1 输出总 schema

```json
{
  "article": {
    "article_id": "not_mentioned",
    "title": "not_mentioned",
    "doi": "not_mentioned",
    "year": "not_mentioned",
    "journal": "not_mentioned"
  },
  "chunk": {
    "chunk_id": "not_mentioned",
    "section": "not_mentioned",
    "page": "not_mentioned"
  },
  "literature_samples": [
    {
      "sample_label": "not_mentioned",
      "polymer_name": "not_mentioned",
      "polymer_alias": "not_mentioned",
      "polymer_class": "not_mentioned",
      "sample_description": "not_mentioned",
      "evidence": "not_mentioned"
    }
  ],
  "composition_events": [],
  "sequence_distribution_events": [],
  "chain_architecture_events": [],
  "molecular_weight_events": [],
  "processing_events": [],
  "warnings": []
}
```

### 9.2 composition_events

```json
{
  "event_type": "composition_event",
  "event_id": "not_mentioned",
  "sample_label": "not_mentioned",
  "polymer_name": "not_mentioned",
  "polymer_composition_type": "homopolymer|condensation_multi_monomer|copolymer|terpolymer|blend|composite|mixture|unknown|not_mentioned",
  "copolymer_sequence_type": "random_copolymer|alternating_copolymer|block_copolymer|gradient_copolymer|graft_copolymer|statistical_copolymer|multiblock_copolymer|not_applicable|unknown|not_mentioned",
  "components": [
    {
      "component_name": "not_mentioned",
      "component_role": "monomer|comonomer|polymer_component|filler|additive|not_mentioned",
      "ratio_value": "not_mentioned",
      "ratio_unit": "mol_fraction|weight_fraction|molar_ratio|wt_percent|mol_percent|not_mentioned",
      "ratio_type": "feed_ratio|actual_ratio|stoichiometric_ratio|not_mentioned|unknown"
    }
  ],
  "evidence": {
    "sentence": "not_mentioned",
    "confidence": 0.0
  }
}
```

### 9.3 sequence_distribution_events

```json
{
  "event_type": "sequence_distribution_event",
  "sample_label": "not_mentioned",
  "polymer_name": "not_mentioned",
  "sequence_distribution_type": "random|alternating|block|blocky|gradient|statistical|periodic|unknown|not_mentioned",
  "quantitative_details": {
    "dyad_fraction": "not_mentioned",
    "triad_fraction": "not_mentioned",
    "reactivity_ratio_r1": "not_mentioned",
    "reactivity_ratio_r2": "not_mentioned"
  },
  "evidence": {
    "sentence": "not_mentioned",
    "confidence": 0.0
  }
}
```

v1 只要求 `sequence_distribution_type`，定量字段可全部为 `not_mentioned`。

### 9.4 chain_architecture_events

```json
{
  "event_type": "chain_architecture_event",
  "sample_label": "not_mentioned",
  "polymer_name": "not_mentioned",
  "chain_architecture": "linear|branched|star|graft|comb|brush|network|crosslinked_network|hyperbranched|dendrimer|cyclic|ladder|unknown|not_mentioned",
  "evidence": {
    "sentence": "not_mentioned",
    "confidence": 0.0
  }
}
```

### 9.5 molecular_weight_events

```json
{
  "event_type": "molecular_weight_event",
  "sample_label": "not_mentioned",
  "polymer_name": "not_mentioned",
  "Mn": { "value": "not_mentioned", "unit": "not_mentioned" },
  "Mw": { "value": "not_mentioned", "unit": "not_mentioned" },
  "PDI": { "value": "not_mentioned", "unit": "dimensionless", "derived": false },
  "DP": { "value": "not_mentioned", "unit": "dimensionless" },
  "method": "GPC|SEC|MALDI|viscometry|not_mentioned|unknown",
  "calibration": "not_mentioned",
  "evidence": {
    "sentence": "not_mentioned",
    "confidence": 0.0
  }
}
```

### 9.6 processing_events

```json
{
  "event_type": "processing_event",
  "sample_label": "not_mentioned",
  "polymer_name": "not_mentioned",
  "method": "solution_casting|spin_coating|drop_casting|melt_extrusion|injection_molding|compression_molding|hot_pressing|annealing|thermal_treatment|solvent_annealing|electrospinning|3d_printing|film_drawing|stretching|curing|crosslinking|not_mentioned|unknown",
  "temperature": { "value": "not_mentioned", "unit": "not_mentioned" },
  "time": { "value": "not_mentioned", "unit": "not_mentioned" },
  "pressure": { "value": "not_mentioned", "unit": "not_mentioned" },
  "solvent": "not_mentioned",
  "atmosphere": "not_mentioned",
  "equipment": "not_mentioned",
  "evidence": {
    "sentence": "not_mentioned",
    "confidence": 0.0
  }
}
```

### 9.7 warnings

```json
{
  "type": "ambiguous_relation|missing_unit|cross_sentence|table_fragment|multiple_materials|not_mentioned",
  "message": "not_mentioned"
}
```

---

## 10. Prompt 约束

LLM prompt 必须包含以下规则：

```text
Only extract facts explicitly stated in the provided text.
Do not infer missing values from prior knowledge.
If a field is absent, write "not_mentioned".
Every extracted event must include an evidence sentence copied from the text.
Use only the allowed enum values.
Do not invent sample labels.
If multiple materials or values are present and relation is ambiguous, output a warning.
Return valid JSON only.
Do not use dataset label values.
```

建议设置：

```text
temperature = 0
no conversation history
one chunk per request
schema-constrained output if available
retry only for invalid JSON
```

Few-shot 示例应按 event 类型选择：

```text
composition shot
molecular weight shot
processing shot
sequence/architecture shot
```

不要放太多 shot，避免成本和示例污染。

---

## 11. 校验与标准化

LLM 输出不能直接入 KG。必须经过四层校验。

### 11.1 JSON schema 校验

检查：

```text
字段完整
枚举合法
数组字段不是 null
数值字段格式合法
缺失值统一为 not_mentioned
```

### 11.2 evidence 校验

检查：

```text
evidence sentence 是否来自 chunk
抽取值是否出现在 evidence 附近
单位是否出现在 evidence 附近
polymer/sample mention 是否能在 chunk 或 article memory 中找到
```

### 11.3 科学规则校验

| 字段 | 校验规则 |
|---|---|
| Mn / Mw | 正数，单位归一为 g/mol |
| PDI | 通常 >= 1；如果 Mn/Mw 存在，检查 `PDI ≈ Mw/Mn` |
| DP | 正数或整数，视文献表达 |
| monomer ratio | 比例非负；mol fraction 总和应接近 1 |
| processing temperature | 单位归一，范围异常标记 |
| sequence distribution | 必须来自枚举或明确证据 |
| chain architecture | 必须来自枚举或明确证据 |

### 11.4 relation-level 校验

重点处理：

```text
多个材料 + 多个数值
respectively 句式
多个 sample label
同一段中多个 processing method
table row 关系
跨句缩写
```

不确定关系：

```text
validation_status = needs_review
confidence 降低
不进入强模型特征
仍可保存在 audit 表
```

---

## 12. 跨 chunk 合并

### 12.1 合并对象

局部 LLM 抽取得到的是 event：

```text
CompositionEvent
SequenceDistributionEvent
ChainArchitectureEvent
MolecularWeightEvent
ProcessingEvent
```

跨 chunk 合并的目标是把 event 对齐到：

```text
LiteratureSample
```

### 12.2 对齐置信度

| 对齐依据 | alignment_scope | 置信度 |
|---|---|---|
| 同一个 sample label | `sample_level` | high |
| 同一 table row | `sample_level` | high |
| 明确短语 `same sample` / `this film` 且 article memory 可解析 | `sample_level` | high |
| 同一 polymer name + 同一 section + 无竞争材料 | `article_polymer_level` | medium |
| 只有同一 PolymerClass | `polymer_class_level` | low |

只有 high / medium 置信度事实进入 numeric prior 的默认统计。low 置信度只进入 KG metadata 或单独 prior。

### 12.3 冲突处理

不覆盖冲突事实。

例如同一 PolymerClass 有多个 Mn：

```text
Mn = 20000 from Article A
Mn = 80000 from Article B
```

应保留两个 MWD event，并在回填时生成：

```text
Mn_prior_mean
Mn_prior_std
Mn_prior_min
Mn_prior_max
Mn_prior_count
Mn_prior_source_count
```

---

## 13. 输入输出目录

建议目录：

```text
kg_work/
  records.csv
  repeat_units.csv
  polymer_class_candidates.jsonl
  polymer_aliases.json
  database/
    polyinfo_records.jsonl
    database_facts.csv
  literature_index.jsonl
  articles/
    metadata.jsonl
    pdf/
    html/
    xml/
    text/
  chunks/
    chunks.jsonl
    candidate_chunks.jsonl
  extractions/
    raw/
    validated/
    rejected/
    extraction_log.csv
  facts/
    literature_samples.csv
    composition_facts.csv
    sequence_distribution_facts.csv
    chain_architecture_facts.csv
    molecular_weight_facts.csv
    processing_facts.csv
    conflict_groups.csv
  kg/
    nodes.csv
    edges.csv
    triples.tsv
    evidence.csv
    entity_aliases.csv
  features/
    enriched_records.csv
    numeric_descriptor_matrix.csv
    kg_entity_mapping.csv
    kg_embedding.npy
  evaluation/
    pilot_gold.jsonl
    hard_negative_chunks.jsonl
    extraction_metrics.csv
```

---

## 14. 分阶段实现

### Stage 0：输入数据整理

输入：

```text
data/raw/smi_all.csv
```

输出：

```text
kg_work/records.csv
```

字段：

```text
record_id
smiles
prop
label_value
```

`label_value` 不进入 LLM、不进入检索 query、不进入 KG enrichment。

### Stage 1：RepeatUnit 去重与规范化

输出：

```text
kg_work/repeat_units.csv
```

字段：

```text
repeat_unit_id
raw_smiles
canonical_smiles
structure_hash
valid_rdkit_parse
formula
molecular_weight_M0
functional_groups
repeat_unit_representation_type
polymer_origin_hint
```

`polymer_origin_hint` 可粗略标记：

```text
addition_homopolymer_like
condensation_multi_monomer_like
siloxane_like
phosphazene_like
unknown
```

这只是结构启发，不是最终 copolymer type。

### Stage 2：RepeatUnit 到 PolymerClass 候选

输出：

```text
kg_work/polymer_class_candidates.jsonl
```

方法：

```text
规则识别
数据库匹配
alias dictionary
LLM rerank
人工审核高频类
```

### Stage 3：Alias 扩展

输出：

```text
kg_work/polymer_aliases.json
```

来源：

```text
PoLyInfo
PubChem
Wikidata
Polymer handbook
人工审核
LLM suggestion with validation
```

### Stage 4：数据库补充

优先对 Top PolymerClass 做 PoLyInfo 补充：

```text
PolymerClass
Alias
Monomer
Composition
Polymer type
Mn / Mw / DP
Processing
Reference
```

注意：

```text
遵守数据库使用条款。
不得无授权大规模抓取。
来源标记为 database，不标记为 article evidence。
```

### Stage 5：定向文献检索

以 PolymerClass 为单位检索。

Query 模板：

```text
"{alias}" Mn Mw PDI DP GPC SEC
"{alias}" molecular weight dispersity
"{alias}" monomer ratio composition copolymer
"{alias}" random copolymer block copolymer graft copolymer
"{alias}" chain architecture branched star graft linear
"{alias}" processing method extrusion casting annealing
```

每个 PolymerClass：

```text
Pilot: Top 3
Batch 1: Top 5
Full run: Top 5-10
```

### Stage 6：文献解析与 chunk 过滤

解析优先级：

```text
HTML/XML > JATS > PDF > abstract
```

候选 chunk 过滤：

```text
polymer alias hit
AND
target keyword hit
AND
value/unit or event keyword present
```

输出：

```text
kg_work/chunks/candidate_chunks.jsonl
```

### Stage 7：LLM event 抽取

对每个候选 chunk 调用 LLM，输出 event JSON。

输出：

```text
kg_work/extractions/raw/{polymer_class_id}/{article_id}_{chunk_id}.json
```

### Stage 8：校验、标准化、合并

输出：

```text
kg_work/extractions/validated/
kg_work/facts/*.csv
```

### Stage 9：样本级回填

输出：

```text
kg_work/features/enriched_records.csv
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
polymer_origin_hint
copolymer_type_distribution
monomer_ratio_prior
sequence_distribution_mode
chain_architecture_mode
Mn_prior_mean
Mn_prior_std
Mw_prior_mean
Mw_prior_std
PDI_prior_mean
PDI_prior_std
DP_prior_mean
processing_method_distribution
source_doi_count
evidence_count
confidence_mean
```

### Stage 10：KG 构建

输出：

```text
kg_work/kg/nodes.csv
kg_work/kg/edges.csv
kg_work/kg/triples.tsv
kg_work/kg/evidence.csv
kg_work/kg/entity_aliases.csv
```

### Stage 11：模型接入

生成：

```text
kg_work/features/kg_entity_mapping.csv
kg_work/features/kg_embedding.npy
kg_work/features/numeric_descriptor_matrix.csv
```

---

## 15. 模型接入策略

不要只用 KG embedding。采用三路输入：

### 15.1 KG embedding branch

适合：

```text
PolymerClass
copolymer type
sequence distribution type
chain architecture
processing method category
monomer identity
```

### 15.2 numeric descriptor branch

适合：

```text
monomer ratio
Mn_prior_mean/std/count
Mw_prior_mean/std/count
PDI_prior_mean/std/count
DP_prior_mean/std/count
processing temperature/time
source_count
evidence_count
confidence_mean
```

### 15.3 metadata / mask

不直接作为数值特征，主要用于审计和防泄漏：

```text
DOI
evidence sentence
extraction method
source type
alignment_scope
validation_status
```

---

## 16. Label Leakage 控制

`val` 是当前任务标签，不能进入 enrichment。

若后续扩展 property measurement，也必须控制：

```text
如果 record.prop = tg，
文献中抽到的 Tg value 默认不进入模型输入。
```

但是以下内容可以进入：

```text
Mn / Mw / PDI / DP prior
composition prior
processing method prior
measurement condition prior
source_count
confidence
```

---

## 17. Pilot 计划

### 17.1 Pilot 范围

```text
50 个 unique repeat units
覆盖 tg / egc 高频结构
优先选择 PA / PE / PP / PS / PMMA / PET / PI / PEO / PVDF
每个 PolymerClass Top 3 文献
每篇最多 5-10 个 candidate chunks
```

### 17.2 Pilot 指标

```text
PolymerClass mapping accuracy
candidate chunk hit rate
JSON valid rate
evidence valid rate
enum valid rate
relation-level correctness
fact accept rate
cost per valid fact
record coverage
```

### 17.3 人工审核

人工检查：

```text
每类 event 至少 20 条
所有 enum 错误
所有 high confidence facts
所有冲突组样例
```

---

## 18. Batch 与 Full Run

### Batch 1

```text
Top 200 高频 repeat units
每个 PolymerClass Top 5 文献
生成第一版 KG 和 enriched_records.csv
```

目标：

```text
稳定 schema
稳定 prompt
稳定校验规则
验证模型接入收益
```

### Full Run

```text
全部 9,702 unique SMILES
按 PolymerClass 合并检索
每个 PolymerClass Top 5-10 文献
```

断点状态：

```text
repeat_unit_identified
polymer_class_linked
aliases_expanded
database_checked
literature_retrieved
articles_parsed
chunks_filtered
llm_extracted
validated
facts_merged
records_enriched
triples_generated
```

---

## 19. 质量指标

建议保存到：

```text
kg_work/evaluation/extraction_metrics.csv
```

指标：

| 指标 | 含义 |
|---|---|
| `json_valid_rate` | LLM 输出可解析 JSON 比例 |
| `evidence_found_rate` | 抽取值能在 evidence 中找到的比例 |
| `enum_valid_rate` | 枚举字段合法比例 |
| `unit_valid_rate` | 单位合法比例 |
| `relation_valid_rate` | event 中实体关系正确比例 |
| `hallucination_rate` | evidence 不支持的 fact 比例 |
| `fact_accept_rate` | 通过校验进入 facts 的比例 |
| `cost_per_valid_fact` | 每条有效 fact 成本 |
| `polymer_class_coverage` | 有补充知识的 PolymerClass 比例 |
| `record_coverage` | 有补充知识的 records 比例 |

---

## 20. 风险与处理

| 风险 | 处理 |
|---|---|
| LLM 幻觉 | evidence validation + not_mentioned + strict schema |
| 共聚类型乱输出 | 固定枚举 + enum validation |
| 把 PA66 误判成普通 homopolymer | 拆分 `polymer_composition_type` 与 `copolymer_sequence_type` |
| 把文献样品事实当当前样本事实 | 引入 `LiteratureSample`，回填 prior statistics |
| Mn/Mw/PDI 混淆 | 单独字段 + PDI 派生标记 |
| sequence distribution 难标准化 | v1 粗粒度，v2 扩展 |
| 跨段落错配 | article memory + alignment_scope |
| 数据库证据不足 | source_type 标记 database，不冒充 evidence sentence |
| 成本过高 | PolymerClass 检索 + chunk filter + pilot |
| label leakage | 不给 LLM `val`，目标属性值默认 mask |

---

## 21. 最小可行版本

v1 必须完成：

```text
records.csv
repeat_units.csv
polymer_class_candidates.jsonl
polymer_aliases.json
candidate_chunks.jsonl
raw extraction JSON
validated facts csv
enriched_records.csv
nodes.csv / edges.csv / triples.tsv / evidence.csv
numeric_descriptor_matrix.csv
```

v1 字段优先级：

| 字段 | v1 策略 |
|---|---|
| `copolymer type` | 做，粗粒度 + 枚举约束 |
| `monomer ratio` | 做，CompositionEvent + numeric |
| `sequence distribution` | 做粗粒度；缺失多则保留 `not_mentioned` |
| `chain architecture` | 做，枚举约束 |
| `Mn / Mw / PDI / DP` | 做，MolecularWeightDescriptor + numeric |
| `processing method` | 做，ProcessingEvent |

v1 不追求：

```text
完整序列统计
图中结构解析
补充材料表格全量解析
复杂跨文章 sample identity resolution
LLM 微调
```

---

## 22. 推荐下一步

1. 新建 `kg_work/` 目录结构。
2. 实现 `records.csv` 和 `repeat_units.csv` 生成脚本。
3. 为 Top 200 unique SMILES 做 PolymerClass 候选映射。
4. 建立 alias / enum / unit 字典。
5. 选 50 个 repeat units 做 pilot。
6. 对每个 PolymerClass 检索 Top 3 文献。
7. 实现 chunk 过滤。
8. 用 LLM 抽取 event JSON。
9. 做 evidence 和 enum 校验。
10. 生成第一版 `LiteratureSample` 和 event facts。
11. 回填 `enriched_records.csv`。
12. 生成 KG triples 和 numeric descriptor matrix。
13. 接入 Uni-Poly 做 ablation。

---

## 23. 总结

用户指定的 KG 结构是合理的，但必须按事实层级处理：

```text
RepeatUnit 负责结构入口。
PolymerClass 负责类别和 alias。
LiteratureSample 负责文献中的具体样品。
CompositionEvent 负责 monomer ratio 和 copolymer type。
SequenceDistributionEvent 负责粗粒度序列分布。
ChainArchitectureEvent 负责链架构。
MolecularWeightDescriptor 负责 Mn / Mw / PDI / DP。
ProcessingEvent 负责加工/制备方法。
Article / Evidence 负责来源和可追溯性。
```

进入模型时不要只依赖 KG embedding，而应使用：

```text
KG embedding
+ numeric descriptor branch
+ metadata / mask
```

这样既能保留 KG 的关系结构和证据链，也能避免连续数值在 KG embedding 中被弱化，同时降低 label leakage 和文献 prior 误用风险。

