# Polymer KG 构建路线与 v1 方案摘要

本文档总结当前 Uni-Poly 项目扩展 Polymer-level KG 的推荐路线。当前数据集只有重复单元 / Polymer SMILES 与属性值，因此 KG 构建应优先服务于现有数据集的属性预测，而不是一开始构建全领域大规模 Polymer KG。

## 1. 路线选择

推荐优先采用路线 B：

```text
当前数据集重复单元 / Polymer SMILES
→ 识别候选 Polymer Class
→ 扩展 alias
→ 定向检索相关文献
→ LLM schema 抽取
→ JSON 转 KG triples
→ 对齐回 Uni-Poly 样本
→ 做 ablation 验证
```

路线 A，即先大批量抽取文献并构建大 KG，再匹配当前数据集，适合作为长期扩展，不适合作为第一阶段主路线。

| 维度 | 路线 A：先建大 KG | 路线 B：目标驱动 KG |
|---|---|---|
| 3~6 个月落地 | 难 | 可行 |
| 抽取成本 | 高 | 可控 |
| 与当前数据集覆盖 | 不确定 | 高 |
| 实体对齐难度 | 高 | 中 |
| 噪声风险 | 高 | 中 |
| 对 Uni-Poly 提升验证 | 难 | 更直接 |

核心原因：当前目标是提升已有 Uni-Poly 数据集的属性预测，因此应先为数据集中出现的重复单元补充知识。

## 2. 重复单元到 Polymer Class

通过重复单元可以识别 polymer structural class，但通常不能唯一确定具体 polymer sample。推荐建模为：

```text
DatasetRecord_001 --has_repeat_unit--> RepeatUnit_PA6
RepeatUnit_PA6 --maps_to_polymer_class--> Polyamide_6
Polyamide_6 --has_alias--> PA6
Polyamide_6 --has_alias--> Nylon_6
Polyamide_6 --has_alias--> Polycaprolactam
```

文献中抽取到的知识应视为 `literature-derived prior`，不要直接当作当前数据样本的真实实验条件。

## 3. 当前优先加入 KG 的信息

| 信息类型 | 优先级 | 推荐形式 | 阶段 |
|---|---:|---|---|
| Measurement condition | ★★★★★ | Condition feature + Measurement node | Phase 1 |
| Mn / Mw / PDI / DP | ★★★★★ | Attribute node + Numeric feature | Phase 1 |
| Copolymer type | ★★★★★ | Entity + Relation | Phase 1 |
| Monomer ratio | ★★★★☆ | Attribute node + Numeric feature | Phase 1 |
| Tacticity | ★★★★☆ | Entity / Attribute | Phase 1-2 |
| Processing method | ★★★★☆ | Entity + metadata | Phase 2 |
| Chain architecture | ★★★☆☆ | Entity + Relation | Phase 2 |
| Structure-property rules | ★★★☆☆ | Rule triple | Phase 3 |

暂不优先做 `sequence distribution`：价值高，但文献表达复杂、标准化难，适合后续阶段。

## 4. Polymer KG v1 Schema

Phase 1 Entity Types：

```text
Polymer
PolymerClass
RepeatUnit
Monomer
CopolymerType
MolecularWeightDescriptor
Property
PropertyMeasurement
MeasurementCondition
Article
Evidence
```

Phase 2 扩展：

```text
Tacticity
ProcessingMethod
ChainArchitecture
Dataset
```

Phase 3 扩展：

```text
StructurePropertyRule
RuleEvidence
SequenceDistribution
```

Phase 1 Relation Types：

```text
has_repeat_unit
maps_to_polymer_class
has_alias
has_monomer
has_monomer_ratio
has_copolymer_type
has_molecular_weight_descriptor
has_property_measurement
measures_property
measured_under
reported_in
has_evidence
```

Phase 2 Relation Types：

```text
has_tacticity
processed_by
has_chain_architecture
derived_from
```

Phase 3 Relation Types：

```text
follows_rule
affects_property
increases_property
decreases_property
supported_by
```

核心属性字段：

```text
value
unit
normalized_value
confidence
source
DOI
article_title
year
evidence_sentence
extraction_method
page
section
```

## 5. 典型三元组示例

```text
Polymer_001 --has_copolymer_type--> Random_Copolymer

Polymer_001 --has_monomer_ratio--> MonomerRatio_001
MonomerRatio_001 --monomer--> Styrene
MonomerRatio_001 --ratio_value--> 0.70
MonomerRatio_001 --ratio_unit--> mol_fraction
MonomerRatio_001 --ratio_type--> actual_ratio

Polymer_001 --has_molecular_weight_descriptor--> MWD_001
MWD_001 --Mn--> 52000 g/mol
MWD_001 --Mw--> 104000 g/mol
MWD_001 --PDI--> 2.0

Polymer_001 --has_property_measurement--> Measurement_001
Measurement_001 --measures_property--> Tg
Measurement_001 --value--> 105
Measurement_001 --unit--> C
Measurement_001 --measured_under--> Condition_001
Condition_001 --heating_rate--> 10 C/min
Measurement_001 --reported_in--> Article_001
Measurement_001 --has_evidence--> Evidence_001
```

## 6. 文献抽取 Pipeline

```text
Literature
↓
Text extraction
↓
Section filtering
↓
LLM JSON extraction
↓
Validation
↓
Normalization
↓
Entity alignment
↓
Triple generation
↓
Polymer KG
```

建议工具与策略：

| 步骤 | 作用 | 推荐工具 / 方法 |
|---|---|---|
| Literature | 获取 PDF/XML/HTML/DOI | Crossref、publisher XML、手动文献集 |
| Text extraction | 抽取正文、表格、章节 | GROBID、PyMuPDF、Science Parse |
| Section filtering | 保留 Experimental、Results、Tables | 关键词规则 + 章节过滤 |
| LLM JSON extraction | 按固定 schema 抽取 | 不微调，使用 schema-constrained prompt |
| Validation | 检查单位、范围、证据 | JSON schema + 规则校验 |
| Normalization | 单位换算与枚举归一 | 自定义字典和规则 |
| Entity alignment | 对齐 polymer / monomer | alias、CAS、SMILES、结构指纹 |
| Triple generation | JSON 转 triples | 固定映射规则 |

## 7. LLM 抽取 JSON 模板

未提及字段必须写 `not mentioned`，不要让 LLM 推测。

```json
{
  "article": {
    "title": "not mentioned",
    "doi": "not mentioned",
    "year": "not mentioned",
    "journal": "not mentioned"
  },
  "polymers": [
    {
      "polymer_name": "not mentioned",
      "polymer_abbreviation": "not mentioned",
      "repeat_unit": "not mentioned",
      "repeat_unit_smiles": "not mentioned",
      "copolymer_type": "Unknown",
      "chain_architecture": "not mentioned",
      "tacticity": "not mentioned",
      "monomers": [
        {
          "monomer_name": "not mentioned",
          "monomer_abbreviation": "not mentioned",
          "monomer_smiles": "not mentioned",
          "ratio_value": "not mentioned",
          "ratio_unit": "not mentioned",
          "ratio_type": "not mentioned"
        }
      ],
      "molecular_weight": {
        "Mn": { "value": "not mentioned", "unit": "not mentioned" },
        "Mw": { "value": "not mentioned", "unit": "not mentioned" },
        "PDI": { "value": "not mentioned", "unit": "not mentioned" },
        "DP": { "value": "not mentioned", "unit": "not mentioned" }
      },
      "processing_method": {
        "method": "not mentioned",
        "temperature": "not mentioned",
        "time": "not mentioned",
        "solvent": "not mentioned",
        "atmosphere": "not mentioned"
      },
      "property_measurements": [
        {
          "property_name": "not mentioned",
          "property_value": "not mentioned",
          "property_unit": "not mentioned",
          "measurement_method": "not mentioned",
          "measurement_condition": {
            "temperature": "not mentioned",
            "frequency": "not mentioned",
            "humidity": "not mentioned",
            "heating_rate": "not mentioned",
            "atmosphere": "not mentioned"
          },
          "evidence_sentence": "not mentioned"
        }
      ],
      "evidence_sentence": "not mentioned",
      "confidence": "not mentioned"
    }
  ]
}
```

## 8. LLM Prompt 要点

System prompt 应包含：

```text
Extract only facts explicitly stated in the provided text.
Do not infer, guess, or complete missing fields.
If a field is not mentioned, write exactly "not mentioned".
Keep numeric values with units.
Every extracted value must include an evidence sentence.
Property values must be linked to measurement conditions.
Distinguish Mn, Mw, PDI, and DP.
For monomer ratio, specify mol%, wt%, feed ratio, actual ratio, or not mentioned.
Copolymer type must be one of:
Homopolymer, Random Copolymer, Alternating Copolymer, Block Copolymer, Gradient Copolymer, Graft Copolymer, Unknown.
Return only valid JSON matching the schema.
```

## 9. JSON 到 Triples 的转换规则

| JSON Field | Head | Relation | Tail / Attribute |
|---|---|---|---|
| polymer_name | Polymer | label | literal |
| polymer_abbreviation | Polymer | abbreviation | literal |
| repeat_unit | Polymer | has_repeat_unit | RepeatUnit |
| monomers[] | Polymer | has_monomer | Monomer |
| monomer ratio | Polymer | has_monomer_ratio | MonomerRatio node |
| copolymer_type | Polymer | has_copolymer_type | CopolymerType |
| tacticity | Polymer | has_tacticity | Tacticity |
| Mn/Mw/PDI/DP | Polymer | has_molecular_weight_descriptor | MWD node |
| processing_method | Polymer | processed_by | ProcessingMethod |
| property_measurements[] | Polymer | has_property_measurement | PropertyMeasurement |
| property_name | PropertyMeasurement | measures_property | Property |
| measurement_condition | PropertyMeasurement | measured_under | Condition node |
| article.doi | Measurement / Evidence | reported_in | Article |
| evidence_sentence | Extracted node | has_evidence | Evidence |

## 10. 质量控制

| 规则 | 目的 | 优先级 |
|---|---|---|
| 单位标准化 | kg/mol 到 g/mol，C/K，Hz/kHz 等统一 | 高 |
| 属性范围检查 | 过滤明显错误，例如 Tg、density、RI 合理范围 | 高 |
| PDI 一致性 | 验证 `PDI ≈ Mw / Mn` | 高 |
| DOI 保存 | 保证来源可追溯 | 高 |
| evidence sentence 保存 | 抑制 LLM 幻觉 | 高 |
| polymer name / abbreviation 对齐 | 合并重复实体 | 高 |
| monomer 标准化 | 合并同义词、缩写、CAS、SMILES | 高 |
| 重复实体合并 | 控制 KG 膨胀 | 中高 |
| 冲突值处理 | 不覆盖，保留 source、condition、confidence | 高 |
| 人工抽样校验 | 每批抽样 5~10%，统计 precision / recall | 高 |

## 11. 与 Uni-Poly 的接入

当前阶段推荐同时使用：

```text
Polymer KG embedding + KG-derived numeric descriptor branch
```

进入 KG embedding：

```text
CopolymerType
Tacticity
ChainArchitecture
ProcessingMethod
Monomer identity
PolymerClass
```

进入 numeric descriptor branch：

```text
Monomer ratio
Mn
Mw
PDI
DP
Measurement temperature
Measurement frequency
Heating rate
Crosslink density
```

暂时只作为 metadata：

```text
DOI
evidence_sentence
page
section
confidence
extraction_method
raw processing details
```

建议 ablation：

```text
Baseline: smiles + graph + fp + geom
+ current KANO KG
+ targeted Polymer KG embedding
+ numeric descriptor branch
+ Polymer KG embedding + numeric descriptor branch
+ condition-aware measurement input
```

## 12. 3~6 个月路线

Phase 1：最小可行版本

```text
目标：围绕当前数据集构建 Targeted Polymer KG v1。
字段：PolymerClass、alias、Copolymer type、Monomer ratio、Mn、Mw、PDI、DP、Measurement condition、PropertyMeasurement。
来源：当前数据集、数据库、少量人工整理文献。
接入：KG embedding + numeric descriptor branch。
验证：覆盖率、单位标准化、PDI 检查、下游 ablation。
风险：repeat unit 到 polymer class 的实体对齐。
```

Phase 2：文献抽取与 evidence

```text
目标：建立 LLM JSON extraction pipeline。
字段：Phase 1 + Tacticity、Processing method、Chain architecture。
来源：PDF/XML/HTML 文献、表格、实验部分、结果部分。
接入：Tacticity / architecture 进 KG；processing 先做 metadata 或 condition feature。
验证：人工抽样、evidence 检查、实体对齐准确率。
风险：表格和跨句信息抽取不稳定。
```

Phase 3：规则与复杂结构

```text
目标：加入 structure-property rules、sequence distribution、crosslink density 和冲突检测。
来源：综述、教材、专家规则、专利、更多文献。
接入：规则先做辅助监督或解释，不直接替代数据驱动预测。
验证：规则一致性、KG relation prediction、跨数据集泛化。
风险：规则适用边界复杂，错误规则可能伤害模型。
```

## 13. 最终建议

当前阶段不要先做全领域大 KG。应以路线 B 为主：从当前数据集的重复单元 / Polymer SMILES 出发，识别 PolymerClass，定向检索文献，抽取结构、分子量、条件和属性测量信息，构建 Targeted Polymer KG。

这条路线能更快形成闭环：

```text
当前数据集 → 定向 KG → Uni-Poly 接入 → ablation 验证 → 再扩大 KG
```
