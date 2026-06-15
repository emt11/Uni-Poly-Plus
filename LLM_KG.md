# LLM 文献抽取与 Polymer KG 构建方案

本文档定义 Uni-Poly-Plus 的 LLM 文献抽取、文档级事实聚合、Polymer KG 构建和 KG embedding 生成流程。本文固定使用 JSON schema v2.0，不再设计其他 schema。

## 1. 项目目标与数据现状

基础数据 data/raw/smi_all.csv 只有：

| 字段 | 含义 |
|---|---|
| smiles | repeat-unit-like Polymer SMILES |
| val | 属性预测标签 |
| prop | 任务名 |

数据没有 DOI、样品编号、组成、分子量或实验条件，无法证明某行与某个文献样品相同。本项目的目标是：

1. 抽取带证据的 LiteratureSample 事实。
2. 连接 RepeatUnit、PolymerClass 和 LiteratureSample。
3. 构建可追溯 Polymer KG。
4. 将多篇文献事实通过 dataset_links 接入同一 Polymer KG。
5. 使用全部结构、类别、数值、事件、来源和证据信息训练 KG embedding。

~~~text
DatasetRecord → RepeatUnit → PolymerClass → LiteratureSample
                                           ├ CompositionAssertion
                                           ├ SequenceDistributionAssertion
                                           ├ ChainArchitectureAssertion
                                           ├ MolecularWeightMeasurement
                                           ├ PolymerizationEvent
                                           └ Evidence → Article
~~~

## 2. 核心原则

### 2.1 LiteratureSample 不等于 DatasetRecord

LiteratureSample 是文章中的具体 sample、film、blend、composite 或材料体系；DatasetRecord 是 smi_all.csv 的一行。文献中的 Mn、组成和条件必须先挂在 LiteratureSample 下，不能声明为当前 record 的真实值。

### 2.2 文献事实作为 KG 邻域知识

同一 repeat unit 可对应不同组成、序列、架构、分子量和制备条件。正确做法是将这些事实保留在各自的 LiteratureSample、Measurement 和 Event 节点中，再通过 dataset_links 与 RepeatUnit/PolymerClass 相连，而不是直接写入 record.Mn。

文献知识通过图中的多来源邻居和关系结构影响 RepeatUnit/PolymerClass embedding，不再单独生成 prior 表格或统计特征文件。

### 2.3 科学语义约束

- 多个单体不必然意味着 random/block/graft copolymer。
- PA66、PET 等固定缩聚重复单元宜用 condensation_multi_monomer。
- composition type、sequence distribution、chain architecture 是不同概念。
- chain architecture 不得默认从 repeat unit 推断。
- LLM 只能抽取 chunk 中明确表达的事实，不能用常识补全。

### 2.4 Label leakage

val 不得输入 LLM、进入检索 query、参与 PolymerClass 判断、dataset linking 或 KG 构建。若未来抽取到与 prop 相同的属性值，也必须从图输入中屏蔽。

## 3. 为什么采用文档级抽取

信息通常跨章节分布：摘要提供名称，实验部分提供样品和条件，表征部分提供 Mn/Mw/PDI/DP，结果和表格提供组成、序列与架构。因此不要求一个段落填满 JSON。

~~~text
文档解析 → 语义切块 → 候选召回 → chunk 局部抽取
→ 文档内实体对齐 → LiteratureSample 聚合
→ 校验/标准化 → dataset_links → KG 构建 → KG embedding
~~~

## 4. 固定 JSON Schema v2.0

~~~json
{
  "article": {
    "article_id": "art_001",
    "title": null,
    "doi": null,
    "year": null,
    "journal": null
  },
  "evidence_records": [
    {
      "evidence_id": "ev_001",
      "chunk_id": "chunk_001",
      "section": null,
      "page": null,
      "paragraph_id": null,
      "sentence": null
    }
  ],
  "literature_samples": [
    {
      "sample_id": "sample_001",
      "sample_label": null,
      "polymer_name": null,
      "polymer_class": null,
      "aliases": [],
      "repeat_unit_smiles": null,
      "identity_evidence_refs": [],
      "composition_assertions": [
        {
          "assertion_id": "comp_001",
          "composition_type": null,
          "ratio_basis": null,
          "components": [
            {"name": null, "role": null, "value": null}
          ],
          "evidence_refs": []
        }
      ],
      "sequence_distribution_assertions": [
        {
          "assertion_id": "seq_001",
          "distribution_type": null,
          "evidence_refs": []
        }
      ],
      "chain_architecture_assertions": [
        {
          "assertion_id": "arch_001",
          "architecture_type": null,
          "evidence_refs": []
        }
      ],
      "molecular_weight_measurements": [
        {
          "measurement_id": "mw_001",
          "Mn": {"value": null, "unit": null},
          "Mw": {"value": null, "unit": null},
          "dispersity": {"value": null},
          "degree_of_polymerization": {
            "value": null,
            "dp_type": null
          },
          "evidence_refs": []
        }
      ],
      "polymerization_events": [
        {
          "event_id": "poly_001",
          "method": null,
          "conditions": {
            "temperature": {"value": null, "unit": null},
            "time": {"value": null, "unit": null},
            "pressure": {"value": null, "unit": null},
            "solvent": null,
            "atmosphere": null,
            "pH": null
          },
          "evidence_refs": []
        }
      ]
    }
  ],
  "dataset_links": [
    {
      "link_id": "link_001",
      "literature_sample_id": "sample_001",
      "target_repeat_unit": {
        "repeat_unit_id": "ru_001",
        "original_smiles": null,
        "canonical_smiles": null
      },
      "relation_type": null,
      "matched_on": [],
      "confidence": null,
      "evidence_refs": []
    }
  ],
  "warnings": []
}
~~~

未发现某类事实时使用空数组，不创建全为 null 的占位记录。

## 5. 顶层字段

### 5.1 article

表示来源文章，不表示样品。article_id 是内部 ID；title、doi、year、journal 用于追踪、去重、引用和 Article 节点属性。KG 关系为 LiteratureSample --reported_in--> Article。

### 5.2 evidence_records

保存事实的原文证据。evidence_id 供其他对象引用；chunk_id、section、page、paragraph_id 定位原文；sentence 保存原文句子而非 LLM 改写。它是防止幻觉、人工审核、跨段落合并和 provenance 的核心。

### 5.3 literature_samples

表示文章中的具体聚合物样品或材料体系。一篇文章可有多个样品，一个 PolymerClass 可对应多个 LiteratureSample。所有组成、结构、分子量和事件事实均挂在此层。

### 5.4 dataset_links

表示 LiteratureSample 与数据集 RepeatUnit 的候选弱链接，不代表二者是同一实验样品。它用于将文献知识接入以 RepeatUnit 为入口的 KG，并通过关系类型和置信度控制消息传播。

### 5.5 warnings

记录 ambiguous_relation、missing_unit、multiple_materials、sample_alignment_unclear、evidence_missing、unsupported_inference、conflicting_values、unknown_enum_value、possible_label_leakage 等问题。

## 6. LiteratureSample 身份字段

- sample_id：内部样品 ID；没有原文标签时也由程序生成。
- sample_label：作者给出的 P1、PA6-1、Sample A；未出现则为 null，禁止编造。
- polymer_name：原文名称，用于 NER、alias 和类别映射。
- polymer_class：归一化类别，如 PA66、PET、PMMA、PS、PEO。
- aliases：原文别名、缩写、商品名和同义名。
- repeat_unit_smiles：仅在原文明确给出或可信结构工具可转换时填写，禁止 LLM 凭名称生成。
- identity_evidence_refs：支持样品身份、名称和类别判断的 evidence ID。

## 7. composition_assertions

表示 LiteratureSample 的组成，承载 copolymer type、component identity 和 ratio。

| 字段 | 含义 |
|---|---|
| assertion_id | 组成断言 ID |
| composition_type | 材料组成类型 |
| ratio_basis | 比例基准 |
| components | 成分列表 |
| evidence_refs | 支持证据 |

推荐枚举：

~~~text
composition_type:
homopolymer, condensation_multi_monomer, copolymer, terpolymer,
blend, composite, mixture, unknown

ratio_basis:
mol_fraction, weight_fraction, feed_ratio, actual_ratio, stoichiometric_ratio
~~~

components.name 是 styrene、MMA、adipic acid、silica 等；role 为 monomer、comonomer、polymer_component、filler、additive、solvent 或 unknown；value 可为 0.7、70 mol% 或 1:1。Feed ratio、实际组成和固定化学计量比不能混用。

composition_assertions 嵌套在 LiteratureSample 下，因此默认表示该文献样品的组成事实，不再使用额外字段重复标记层级。事实来源通过 evidence_refs、Evidence 和 Article 关系表达；PolymerClass 级知识通过 KG 关系连接，不写入该 assertion 的层级属性。

## 8. sequence_distribution_assertions

表示共聚单元或链段沿链排列方式，不等于 composition type。

- assertion_id：断言 ID。
- distribution_type：random、statistical、alternating、block、blocky、multiblock、gradient、graft、periodic、not_applicable、unknown。
- evidence_refs：支持证据。

文献未明确说明时不允许推测。Homopolymer、blend、composite 和固定缩聚重复单元未提及时优先使用空数组。

## 9. chain_architecture_assertions

architecture_type 可为 linear、branched、star、graft、comb、brush、network、crosslinked_network、hyperbranched、dendrimer、cyclic、ladder、unknown。assertion_id 标识断言，evidence_refs 保存证据。不得根据连接位点、官能团或聚合方法默认推断架构。

## 10. molecular_weight_measurements

表示 LiteratureSample 级 Mn、Mw、PDI/Đ 和 DP。

- measurement_id：记录 ID。
- Mn.value/unit、Mw.value/unit：数值和原始单位。
- dispersity.value：文献明确报告的 PDI/Đ。
- degree_of_polymerization.value：文献明确报告的 DP；dp_type 为 number_average、weight_average 或 unknown。
- evidence_refs：支持证据。

LLM 只提取原始值、单位和证据；程序负责统一 Mn/Mw 到 g/mol。dispersity.value 和 degree_of_polymerization.value 只保存文献明确报告的值，未报告时保持 null。后处理程序可以根据 Mw/Mn 或分子量与重复单元摩尔质量计算派生结果，用于一致性校验或在 KG 中建立派生关系，但不把计算值写回这两个字段。同一样品多个值应保留多条 measurement。

这些连续值全部进入 KG。原始数值作为 MolecularWeightMeasurement 节点的属性或数值节点保存；构图时可由确定性程序增加标准化数值和区间类别，用于 KG embedding 学习数值尺度。JSON schema 本身不新增字段。

## 11. polymerization_events

保存聚合、合成或样品制备方法和条件：

- event_id：事件 ID。
- method：聚合、合成或制备方法。
- conditions：temperature、time、pressure、solvent、atmosphere、pH。
- evidence_refs：支持证据。

聚合方法如 free-radical、condensation、ring-opening polymerization；processing/preparation 如 solution casting、melt extrusion、hot pressing、annealing。二者科学概念不同，但 v2.0 没有 processing_events，因此 broad preparation information 暂存于此，并依靠 evidence 保留语境。

Schema 没有 steps。多阶段条件应拆成多个 event；无法确定归属时增加 warning。

## 12. dataset_links

- link_id：链接 ID。
- literature_sample_id：目标 LiteratureSample。
- target_repeat_unit：RepeatUnit ID、原始 SMILES 和规范化 SMILES。
- relation_type：链接类型。
- matched_on：匹配依据。
- confidence：链接程序的校准分数，不是 LLM 主观置信度。
- evidence_refs：支持身份和链接的证据。

relation_type：

~~~text
exact_repeat_unit_match, canonical_smiles_match, polymer_class_match,
alias_match, functional_group_similarity, family_level_match, uncertain
~~~

matched_on：

~~~text
canonical_smiles, polymer_name, alias, repeat_unit,
monomer_names, functional_groups, database_id
~~~

精确结构链接可使用高关系权重；class/alias 为中低权重；family/similarity 为低权重；uncertain 默认不用于向 RepeatUnit 传播知识。禁止直接把文献值写成 record 真值。

## 13. LLM 抽取原则

1. 只抽取输入文本明确表达的事实。
2. 每条 Assertion、Measurement、Event 必须引用 evidence。
3. 不编造标签、结构、比例、序列、架构或条件。
4. 缺失对象字段用 null，没有事实的数组用 []。
5. 不创建全为 null 的占位事实。
6. 多材料、多数值、respectively、表格列头缺失或代词归属不清时增加 warning。
7. Prompt 不包含 val。

## 14. 文档级抽取与职责划分

优先解析 HTML/XML/JATS，其次开放 HTML/PDF；保留 section、page、paragraph order、table/caption 和 supplementary 来源。按语义切块，并用关键词、正则、BM25、embedding 或分类器召回候选。

每个 chunk 只抽局部事实。文档级 entity memory 对齐 polymer name、alias、sample label、表格行和代词。优先级为：相同 sample label > 相同表格行 > 明确代词回指 > 无竞争材料的同名对象 > 仅同 PolymerClass。只有可靠对齐的事实才能合并到同一 sample。

| 内容 | 负责方 |
|---|---|
| Article/文本定位 | 文档解析器 |
| 局部样品事实和证据句 | LLM |
| 跨 chunk 样品合并 | 文档级聚合程序，LLM 可辅助 |
| 各类 ID | 后处理程序 |
| 枚举映射、单位转换、派生值 | 确定性程序 |
| canonical SMILES | 化学信息学工具 |
| dataset_links/confidence | 实体链接程序 |
| warnings | 校验和聚合程序 |

最终 JSON 由多个阶段共同生成，不是单次 LLM 调用的直接输出。应保存 raw extraction、validated article JSON 和 final v2.0 JSON。

## 15. 校验与标准化

### 15.1 结构和证据

- schema_version 必须为 2.0，顶层字段不缺失、不重命名。
- 数组类型和 ID 引用合法。
- evidence sentence 可在 chunk 原文定位。
- 数值、单位、材料和 sample 关系受 evidence 支持。
- evidence 失败时进入 warning/review，不静默修补。

### 15.2 枚举和科学规则

校验 composition type、ratio basis、component role、sequence、architecture、dp_type、relation type 和 matched_on。未映射值标记 warning。

Mn/Mw 必须为正数并在统计时统一 g/mol；PDI 通常不小于 1并检查 Mw/Mn；DP 为正数；比例非负；温度、时间、压力和 pH 做单位与范围校验。

## 16. Polymer KG

节点：

~~~text
DatasetRecord, RepeatUnit, PolymerClass, LiteratureSample, Article, Evidence,
CompositionAssertion, Component, SequenceDistribution, ChainArchitecture,
MolecularWeightMeasurement, PolymerizationEvent
~~~

关系：

~~~text
DatasetRecord --has_repeat_unit--> RepeatUnit
RepeatUnit --maps_to--> PolymerClass
LiteratureSample --belongs_to--> PolymerClass
LiteratureSample --reported_in--> Article
LiteratureSample --has_composition--> CompositionAssertion
CompositionAssertion --has_component--> Component
LiteratureSample --has_sequence_distribution--> SequenceDistribution
LiteratureSample --has_chain_architecture--> ChainArchitecture
LiteratureSample --has_molecular_weight--> MolecularWeightMeasurement
LiteratureSample --has_polymerization_event--> PolymerizationEvent
AnyFact --supported_by--> Evidence
LiteratureSample --linked_to_repeat_unit--> RepeatUnit
~~~

事实证据链必须为 Fact → Evidence → SourceChunk → Article。

## 17. 全部数据进入 Polymer KG

本方案不再生成 enriched_records.csv 或独立的 numeric descriptor matrix。v2.0 JSON 中的所有有效数据统一转换为 KG 节点、关系或属性：

| JSON 信息 | KG 表示 |
|---|---|
| PolymerClass、composition type、sequence、architecture、method | 枚举实体节点及关系 |
| component name | ChemicalEntity/Component 节点 |
| component ratio | CompositionAssertion 的数值属性或 RatioValue 节点 |
| Mn、Mw、PDI、DP | MolecularWeightMeasurement 的数值属性或 Value 节点 |
| temperature、time、pressure、pH | PolymerizationEvent 的数值属性或 ConditionValue 节点 |
| solvent、atmosphere | 条件实体节点及关系 |
| Article、Evidence | provenance 节点及 supported_by/reported_in 关系 |
| relation_type、confidence | dataset link 的关系类型和边属性 |
| warnings | 审核属性；默认不作为普通事实关系传播 |

同一个 RepeatUnit 关联的不同 LiteratureSample 不需要先压缩成均值或众数。它们作为不同邻居保留，KG 模型通过多跳关系和聚合机制学习其分布与共现模式。

为了比较弱链接的影响，可以构建两种图视图，但最终产物仍是 KG embedding：

- strict graph：只保留 exact_repeat_unit_match 和 canonical_smiles_match。
- broad graph：加入 polymer_class_match、alias_match 等弱链接，并使用关系类型或 confidence 控制权重。

### 17.1 连续数值的 KG 表示

全部数据进入 KG 不等于把每个浮点数当作彼此无关的字符串实体。连续值建议同时使用：

1. 原始数值属性：保留精确 value 和 unit。
2. 标准化数值属性：由后处理程序统一单位，仅用于构图和训练，不修改 JSON schema。
3. 区间节点：按训练集或领域规则离散为 Mn_bin、temperature_bin 等类别节点。
4. 关系限定：区分 has_Mn、has_Mw、has_PDI、has_temperature。

例如：

~~~text
MolecularWeightMeasurement_mw1 --has_Mn_value--> 52000
MolecularWeightMeasurement_mw1 --has_Mn_bin--> Mn_50k_100k
MolecularWeightMeasurement_mw1 --supported_by--> Evidence_ev14
LiteratureSample_s1 --has_molecular_weight--> MolecularWeightMeasurement_mw1
~~~

区间边界必须由确定性程序生成，并在训练、验证和测试间使用同一规则，不能让 LLM 决定。

## 18. KG Embedding 设计

最终只生成 KG embedding。Embedding 覆盖：

~~~text
RepeatUnit
PolymerClass
LiteratureSample
CompositionAssertion / Component
SequenceDistribution
ChainArchitecture
MolecularWeightMeasurement / numeric bins
PolymerizationEvent / condition entities / numeric bins
Article / Evidence
~~~

### 18.1 关系与数值初始化

- 枚举实体和普通关系使用可训练 embedding 或预训练 KG embedding。
- 原始连续值作为对应 Measurement/Event 节点的初始数值特征。
- 数值区间作为普通 KG 实体参与关系学习。
- relation_type 建成不同边类型。
- confidence 作为边权重或消息传递门控值。
- Evidence 和 Article 可参与 provenance-aware embedding；若图过大，可只参与 KG 训练，不输出其最终向量。

### 18.2 RepeatUnit 的最终表示

最终模型使用 RepeatUnit 节点或其 PolymerClass 节点的 KG embedding。图编码器通过多跳消息传递，将 LiteratureSample 的 Assertion、Measurement、Event 和 Evidence 信息汇总到 RepeatUnit。

不存在文献链接的 RepeatUnit 使用 unknown/no-literature embedding，并保留 coverage mask，不能用零向量暗示数值为零。

### 18.3 输出

下游只需要：

~~~text
kg_entity_mapping.csv
kg_embedding.npy
~~~

kg_entity_mapping.csv 只负责将 repeat_unit_id、PolymerClass ID 和 embedding 行号对应起来，不重复承载文献事实。

## 19. 实施路线

### 19.1 Pilot：验证正确性

选择 50 个具有代表性的 unique RepeatUnit，而不是简单选择出现频率最高的结构。样本应覆盖：

~~~text
主要 PolymerClass
均聚物、固定缩聚多单体体系和明确共聚物
简单与复杂 repeat-unit 表示
文献丰富与文献稀缺类别
容易发生样品对齐或枚举混淆的困难案例
~~~

人工阅读对应文献并建立 gold set，标注样品身份、组成、序列、架构、分子量、聚合条件、evidence 和 dataset link。Pilot 的目标是验证抽取、跨 chunk 聚合、实体链接和 KG 映射是否正确。

### 19.2 Batch：验证工程稳定性

选择约 200 个 unique RepeatUnit，重点覆盖主要 PolymerClass、不同结构复杂度及 Pilot 发现的困难案例。频率可以作为覆盖数据行数的参考，但不能作为唯一选择依据。

Batch 阶段用于验证批量文档解析、LLM 成本、失败重试、样品聚合、KG 规模和 embedding 训练稳定性。完成后冻结：

~~~text
JSON schema v2.0
受控枚举
Prompt 模板
evidence 校验规则
dataset linking 规则和阈值
数值单位标准化与区间规则
JSON 到 KG triples/attributes 的映射
~~~

### 19.3 Full run：扩展覆盖率

将全部 unique RepeatUnit 先映射到 PolymerClass，再按 PolymerClass 合并检索和去重文献，避免对多个相关 RepeatUnit 重复下载、解析和抽取同一文章。

处理状态至少按 RepeatUnit、Article 和 Chunk 保存：

~~~text
repeat_unit_normalized
polymer_class_mapped
literature_retrieved
article_parsed
chunks_recalled
llm_extracted
sample_aggregated
validated
dataset_linked
kg_generated
embedding_generated
~~~

失败后从对应阶段继续，不重新运行已经完成的文献和对象。

~~~text
数据整理 → RepeatUnit 规范化 → PolymerClass/alias
→ 文献检索 → 文档解析/chunk 召回 → LLM 抽取
→ LiteratureSample 聚合 → 校验/标准化 → dataset linking
→ KG triples/attributes → KG embedding
~~~

评估 JSON valid rate、evidence precision、sample identity accuracy、事实准确率、跨 chunk 对齐准确率、dataset link precision、hallucination rate 和 cost per accepted fact。

## 20. 风险与验收

| 风险 | 控制 |
|---|---|
| LLM 幻觉 | 明确事实 + evidence 原文校验 |
| 跨样品错配 | sample/table/entity 对齐，不确定则拆分 |
| 缩聚物误分类 | condensation_multi_monomer，禁止推断序列 |
| ratio 混淆 | ratio_basis 区分 feed/actual/stoichiometric |
| 文献值成为 record 真值 | LiteratureSample 节点隔离，通过 KG 关系间接传播 |
| 连续尺度丢失 | 原始数值属性 + 标准化属性 + 数值区间节点 |
| 弱链接噪声 | relation 分层、confidence、strict/broad graph |
| 文献量偏差 | Article 级去重、source count、最小支持度 |
| label leakage | val 全流程隔离、同目标属性屏蔽 |

验收要求：每条事实可追溯；多样品不默认合并；未提及字段不补全；PA66/PET 不误标为 random/block/graft；样品级事实通过 LiteratureSample 和 dataset link 接入 KG；dataset link 有 relation、依据和 confidence；val 不进入检索、prompt 或链接；全部有效 JSON 数据能一致转换为 KG；最终能够为 RepeatUnit 生成 KG embedding。

## 21. 总结

本方案构建的是以证据为基础的文献知识层，而不是为 smi_all.csv 每行猜测缺失字段：

~~~text
RepeatUnit → PolymerClass → LiteratureSample
                         → Assertion/Measurement/Event
                         → Evidence/Article
~~~

最终不再拆分 enriched table、numeric branch 和 KG branch。枚举、关系、连续值、实验条件、来源及证据统一进入 Polymer KG；通过节点属性、数值区间节点、边类型和边权重共同训练 KG embedding。

该分层既保留 Polymer KG 的可追溯性，也避免把文献具体样品事实误当成当前 DatasetRecord 的真实值。
