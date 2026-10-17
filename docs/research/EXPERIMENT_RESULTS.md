# 现存正式汇总：全部政策与七任务数值对照

从保存的JSON汇总摘录；未访问预测数组、未重算模型指标。完整精度见[evidence/formal_results.json](evidence/formal_results.json)与[CSV](evidence/formal_results.csv)。

**含复用参照、旧五/八任务方案及退役路线，不等于同口径排行榜，也不是新增实验计数。** 当前多数来源std ddof=0，旧formal_grid为ddof=1；每份来源的声明与审查状态单列。早期r13存在已知实现限制；没有新G2结果。

差值为本项目减SOTA_RESULT对应论文最佳R²；协议等价性未核实，正差不宣称公平超越。宏7仅七任务完整时计算，缺失项不补齐。所有Markdown展示三位小数，JSON/CSV保留完整精度。

## fce_param_20261014/r2/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_param_20261014/r2/final_review/seven_task_results.json`。

来源声明：shared TEST used in development; paper split equivalence unverified; numerical comparison only；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B_LR|0.921±0.012|0.929±0.012|0.911±0.008|0.798±0.031|0.764±0.044|0.846±0.044|0.234±0.140|0.772|
|R0|0.927±0.018|0.940±0.014|0.911±0.017|0.790±0.031|0.766±0.048|0.820±0.054|0.305±0.144|0.780|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|B_LR|-0.013|+0.006|-0.000|-0.041|-0.023|-0.010|-0.189|
|R0|-0.007|+0.017|+0.000|-0.049|-0.021|-0.036|-0.118|

## fce_tune_20261015/t1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261015/t1/final_review/seven_task_results.json`。

来源声明：shared validation/TEST used in development; paper split equivalence unverified; numerical comparison only；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|T1_R0|0.927±0.018|0.940±0.014|0.911±0.017|0.790±0.031|0.766±0.048|0.820±0.054|0.305±0.144|0.780|
|T1_TUNED|0.927±0.018|0.940±0.003|0.912±0.017|0.794±0.046|0.771±0.047|0.830±0.059|0.270±0.183|0.778|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|T1_R0|-0.007|+0.017|+0.000|-0.049|-0.021|-0.036|-0.118|
|T1_TUNED|-0.007|+0.017|+0.001|-0.045|-0.016|-0.026|-0.153|

## fce_tune_20261015/t2/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261015/t2/final_review/seven_task_results.json`。

来源声明：shared validation/TEST used in development; paper split equivalence unverified; numerical comparison only；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|T2_G_LR|0.918±0.020|0.937±0.007|0.911±0.012|0.793±0.036|0.764±0.046|0.825±0.058|0.354±0.106|0.786|
|T2_R0|0.927±0.018|0.940±0.014|0.911±0.017|0.790±0.031|0.766±0.048|0.820±0.054|0.305±0.144|0.780|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|T2_G_LR|-0.016|+0.014|+0.000|-0.046|-0.023|-0.031|-0.069|
|T2_R0|-0.007|+0.017|+0.000|-0.049|-0.021|-0.036|-0.118|

## fce_tune_20261015/t3/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261015/t3/final_review/seven_task_results.json`。

来源声明：shared validation/TEST used in development; paper split equivalence unverified; numerical comparison only；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|T3_COS5K|0.922±0.015|0.927±0.015|0.907±0.014|0.794±0.020|0.716±0.066|0.773±0.067|0.307±0.152|0.764|
|T3_REF|0.927±0.014|0.931±0.012|0.913±0.015|0.790±0.012|0.670±0.108|0.770±0.121|0.314±0.148|0.759|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|T3_COS5K|-0.012|+0.004|-0.004|-0.045|-0.071|-0.083|-0.116|
|T3_REF|-0.007|+0.008|+0.002|-0.049|-0.117|-0.086|-0.109|

## fce_tune_20261015/t3b/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261015/t3b/final_review/seven_task_results.json`。

来源声明：shared validation/TEST used in development; paper split equivalence unverified; numerical comparison only；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|T3B_G05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|T3_REF|0.927±0.014|0.931±0.012|0.913±0.015|0.790±0.012|0.670±0.108|0.770±0.121|0.314±0.148|0.759|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|T3B_G05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|T3_REF|-0.007|+0.008|+0.002|-0.049|-0.117|-0.086|-0.109|

## fce_tune_20261016/aligned_alpha/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/aligned_alpha/final_review/seven_task_results.json`。

来源声明：shared validation/TEST development exposed; paper equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|g05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|r10|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|
|r11|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.776±0.050|0.814±0.049|0.411±0.122|0.796|
|r9|0.916±0.012|0.912±0.013|0.914±0.012|0.801±0.019|0.768±0.026|0.817±0.044|0.430±0.088|0.794|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|g05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|r10|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|
|r11|-0.010|-0.000|+0.006|-0.032|-0.011|-0.042|-0.012|
|r9|-0.018|-0.011|+0.003|-0.038|-0.019|-0.039|+0.007|

## fce_tune_20261016/all_task_kernel/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/all_task_kernel/final_review/seven_task_results.json`。

来源声明：shared TEST development exposed; same policy reused units predeclared; paper equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|branch|0.864±0.007|0.860±0.026|0.884±0.014|0.741±0.028|0.683±0.087|0.762±0.052|0.392±0.067|0.741|
|fused|0.916±0.012|0.912±0.013|0.914±0.012|0.801±0.019|0.768±0.026|0.817±0.044|0.430±0.088|0.794|
|reference|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|branch|-0.070|-0.063|-0.027|-0.098|-0.104|-0.094|-0.031|
|fused|-0.018|-0.011|+0.003|-0.038|-0.019|-0.039|+0.007|
|reference|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|

## fce_tune_20261016/ei_kernel/recovery2/final_review/ei_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/ei_kernel/recovery2/final_review/ei_results.json`。

来源声明：shared TEST development exposed; paper protocol equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|branch|未完成／不可比较|未完成／不可比较|未完成／不可比较|0.741±0.028|未完成／不可比较|未完成／不可比较|未完成／不可比较|不可计算|
|fused|未完成／不可比较|未完成／不可比较|未完成／不可比较|0.801±0.019|未完成／不可比较|未完成／不可比较|未完成／不可比较|不可计算|
|reference|未完成／不可比较|未完成／不可比较|未完成／不可比较|0.796±0.015|未完成／不可比较|未完成／不可比较|未完成／不可比较|不可计算|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|branch|不可比较|不可比较|不可比较|-0.098|不可比较|不可比较|不可比较|
|fused|不可比较|不可比较|不可比较|-0.038|不可比较|不可比较|不可比较|
|reference|不可比较|不可比较|不可比较|-0.043|不可比较|不可比较|不可比较|

## fce_tune_20261016/g05_glr/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/g05_glr/final_review/seven_task_results.json`。

来源声明：shared validation/TEST used in development; paper split equivalence unverified; numerical comparison only；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|G05_GLR|0.925±0.012|0.928±0.011|0.909±0.012|0.779±0.047|0.759±0.037|0.834±0.054|0.359±0.114|0.785|
|T3B_G05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|G05_GLR|-0.009|+0.005|-0.002|-0.060|-0.028|-0.022|-0.064|
|T3B_G05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|

## fce_tune_20261016/global_blend/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/global_blend/final_review/seven_task_results.json`。

来源声明：shared TEST development exposed; paper equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|g05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|global|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|
|r9|0.916±0.012|0.912±0.013|0.914±0.012|0.801±0.019|0.768±0.026|0.817±0.044|0.430±0.088|0.794|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|g05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|global|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|
|r9|-0.018|-0.011|+0.003|-0.038|-0.019|-0.039|+0.007|

## fce_tune_20261016/graph_distill/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/graph_distill/final_review/seven_task_results.json`。

来源声明：development-exposed validation/TEST; paper protocol equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|CONT_REF_fused|0.926±0.008|0.927±0.009|0.917±0.013|0.810±0.018|0.764±0.054|0.821±0.049|0.368±0.099|0.790|
|CONT_REF_neural|0.924±0.009|0.927±0.013|0.913±0.014|0.800±0.029|0.751±0.062|0.810±0.060|0.320±0.124|0.778|
|GRAPH_DISTILL_fused|0.923±0.011|0.931±0.005|0.915±0.009|0.797±0.007|0.723±0.063|0.827±0.045|0.410±0.064|0.790|
|GRAPH_DISTILL_neural|0.923±0.012|0.934±0.006|0.909±0.011|0.787±0.008|0.697±0.066|0.816±0.067|0.371±0.067|0.777|
|g05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|r10|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|CONT_REF_fused|-0.008|+0.004|+0.006|-0.029|-0.023|-0.035|-0.055|
|CONT_REF_neural|-0.010|+0.004|+0.002|-0.039|-0.036|-0.046|-0.103|
|GRAPH_DISTILL_fused|-0.011|+0.008|+0.004|-0.042|-0.064|-0.029|-0.013|
|GRAPH_DISTILL_neural|-0.011|+0.011|-0.002|-0.052|-0.090|-0.040|-0.052|
|g05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|r10|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|

## fce_tune_20261016/graph_distill/repair_device_fixture/evidence_before/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/graph_distill/repair_device_fixture/evidence_before/seven_task_results.json`。

来源声明：development-exposed validation/TEST; paper protocol equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|CONT_REF_fused|0.926±0.008|0.927±0.009|0.917±0.013|0.810±0.018|0.764±0.054|0.821±0.049|0.368±0.099|0.790|
|CONT_REF_neural|0.924±0.009|0.927±0.013|0.913±0.014|0.800±0.029|0.751±0.062|0.810±0.060|0.320±0.124|0.778|
|GRAPH_DISTILL_fused|0.923±0.011|0.931±0.005|0.915±0.009|0.797±0.007|0.723±0.063|0.827±0.045|0.410±0.064|0.790|
|GRAPH_DISTILL_neural|0.923±0.012|0.934±0.006|0.909±0.011|0.787±0.008|0.697±0.066|0.816±0.067|0.371±0.067|0.777|
|g05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|r10|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|CONT_REF_fused|-0.008|+0.004|+0.006|-0.029|-0.023|-0.035|-0.055|
|CONT_REF_neural|-0.010|+0.004|+0.002|-0.039|-0.036|-0.046|-0.103|
|GRAPH_DISTILL_fused|-0.011|+0.008|+0.004|-0.042|-0.064|-0.029|-0.013|
|GRAPH_DISTILL_neural|-0.011|+0.011|-0.002|-0.052|-0.090|-0.040|-0.052|
|g05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|r10|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|

## fce_tune_20261016/huber_ft/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/huber_ft/final_review/seven_task_results.json`。

来源声明：shared validation/TEST development exposed; paper equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|fused|0.922±0.012|0.931±0.005|0.918±0.012|0.791±0.018|0.745±0.049|0.825±0.044|0.378±0.059|0.787|
|g05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|neural|0.920±0.010|0.933±0.005|0.914±0.011|0.779±0.024|0.727±0.052|0.815±0.055|0.316±0.063|0.772|
|r10|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|fused|-0.012|+0.008|+0.007|-0.048|-0.042|-0.031|-0.045|
|g05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|neural|-0.014|+0.010|+0.003|-0.060|-0.060|-0.041|-0.107|
|r10|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|

## fce_tune_20261016/norm_ft/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/norm_ft/final_review/seven_task_results.json`。

来源声明：shared validation/TEST development exposed; paper equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|fused|0.902±0.014|0.915±0.014|0.897±0.011|0.781±0.049|0.716±0.050|0.794±0.066|0.346±0.123|0.764|
|g05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|neural|0.890±0.013|0.906±0.019|0.881±0.011|0.763±0.057|0.688±0.058|0.777±0.084|0.291±0.165|0.742|
|r10|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|fused|-0.032|-0.008|-0.014|-0.058|-0.071|-0.062|-0.077|
|g05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|neural|-0.044|-0.017|-0.030|-0.076|-0.099|-0.079|-0.132|
|r10|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|

## fce_tune_20261016/radius3/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/radius3/final_review/seven_task_results.json`。

来源声明：shared validation/TEST development exposed; paper equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|r10|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|
|r11|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.776±0.050|0.814±0.049|0.411±0.122|0.796|
|r12|0.923±0.017|0.922±0.013|0.916±0.012|0.808±0.017|0.775±0.056|0.814±0.053|0.408±0.126|0.795|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|r10|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|
|r11|-0.010|-0.000|+0.006|-0.032|-0.011|-0.042|-0.012|
|r12|-0.011|-0.001|+0.005|-0.031|-0.012|-0.042|-0.015|

## fce_tune_20261016/xc_kernel/final_review/xc_results.json

本地来源（原文件不上传）：`results/fce_tune_20261016/xc_kernel/final_review/xc_results.json`。

来源声明：shared TEST development exposed; paper protocol equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|branch|未完成／不可比较|未完成／不可比较|未完成／不可比较|未完成／不可比较|未完成／不可比较|未完成／不可比较|0.392±0.067|不可计算|
|fused|未完成／不可比较|未完成／不可比较|未完成／不可比较|未完成／不可比较|未完成／不可比较|未完成／不可比较|0.430±0.088|不可计算|
|reference|未完成／不可比较|未完成／不可比较|未完成／不可比较|未完成／不可比较|未完成／不可比较|未完成／不可比较|0.371±0.157|不可计算|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|branch|不可比较|不可比较|不可比较|不可比较|不可比较|不可比较|-0.031|
|fused|不可比较|不可比较|不可比较|不可比较|不可比较|不可比较|+0.007|
|reference|不可比较|不可比较|不可比较|不可比较|不可比较|不可比较|-0.052|

## fce_tune_20261017/nonbonded_ft/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fce_tune_20261017/nonbonded_ft/final_review/seven_task_results.json`。

来源声明：development-exposed validation/TEST; paper protocol equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|NB_CONST_fused|0.924±0.016|0.928±0.004|0.917±0.013|0.805±0.022|0.711±0.078|0.812±0.056|0.398±0.109|0.785|
|NB_CONST_neural|0.923±0.017|0.929±0.008|0.912±0.012|0.796±0.023|0.669±0.101|0.797±0.072|0.351±0.130|0.768|
|NB_REAL_fused|0.927±0.010|0.932±0.005|0.917±0.013|0.794±0.025|0.735±0.061|0.824±0.049|0.336±0.134|0.781|
|NB_REAL_neural|0.925±0.012|0.933±0.004|0.912±0.014|0.783±0.023|0.715±0.061|0.819±0.053|0.273±0.170|0.766|
|g05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|r10|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|NB_CONST_fused|-0.010|+0.005|+0.006|-0.034|-0.076|-0.044|-0.025|
|NB_CONST_neural|-0.011|+0.006|+0.001|-0.043|-0.118|-0.059|-0.072|
|NB_REAL_fused|-0.007|+0.009|+0.006|-0.045|-0.052|-0.032|-0.087|
|NB_REAL_neural|-0.009|+0.010|+0.001|-0.056|-0.072|-0.037|-0.150|
|g05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|r10|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|

## fp_graph_fusion_20261017/f1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/fp_graph_fusion_20261017/f1/final_review/seven_task_results.json`。

来源声明：development-exposed validation/TEST; paper protocol equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|FP_NULL_KFUSE|0.922±0.022|0.928±0.011|0.910±0.016|0.802±0.024|0.735±0.047|0.809±0.087|0.341±0.121|0.778|
|FP_REAL_KFUSE|0.882±0.025|0.899±0.011|0.883±0.014|0.735±0.029|0.683±0.090|0.788±0.068|0.302±0.093|0.739|
|FP_REAL_MEAN|0.920±0.019|0.917±0.011|0.893±0.011|0.782±0.048|0.750±0.063|0.806±0.072|0.366±0.098|0.776|
|g05|0.919±0.023|0.924±0.014|0.911±0.014|0.796±0.015|0.758±0.078|0.797±0.062|0.371±0.157|0.782|
|r10|0.924±0.017|0.923±0.012|0.917±0.013|0.807±0.016|0.775±0.049|0.816±0.049|0.412±0.120|0.796|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|FP_NULL_KFUSE|-0.012|+0.005|-0.001|-0.037|-0.052|-0.047|-0.082|
|FP_REAL_KFUSE|-0.052|-0.024|-0.028|-0.104|-0.104|-0.068|-0.121|
|FP_REAL_MEAN|-0.014|-0.006|-0.018|-0.057|-0.037|-0.050|-0.057|
|g05|-0.015|+0.001|+0.000|-0.043|-0.029|-0.059|-0.052|
|r10|-0.010|-0.000|+0.006|-0.032|-0.012|-0.040|-0.011|

## glt_dual_opt_20261006/r1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_dual_opt_20261006/r1/final_review/seven_task_results.json`。

来源声明：paper5_outer five-fold arithmetic mean; seed42+fold; ddof0; paper equivalence unverified; not blind test；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A_ALIGN|0.928±0.015|0.926±0.013|0.915±0.016|0.787±0.029|0.747±0.042|0.807±0.076|0.357±0.179|0.781|
|B_FP|0.917±0.012|0.918±0.014|0.901±0.014|0.803±0.027|0.717±0.053|0.800±0.046|0.319±0.192|0.768|
|F_TOKEN|0.924±0.018|0.923±0.006|0.908±0.014|0.807±0.015|0.739±0.049|0.800±0.064|0.354±0.174|0.779|
|G_SEAM_GEO|0.926±0.016|0.923±0.015|0.906±0.013|0.800±0.014|0.736±0.039|0.812±0.059|0.310±0.123|0.773|
|M_TOKEN_CE|0.920±0.016|0.938±0.007|0.912±0.014|0.789±0.021|0.738±0.051|0.811±0.056|0.299±0.130|0.772|
|R_INCIDENT|0.917±0.011|0.921±0.013|0.902±0.014|0.803±0.020|0.707±0.061|0.814±0.054|0.344±0.106|0.773|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|A_ALIGN|-0.006|+0.003|+0.004|-0.052|-0.040|-0.049|-0.066|
|B_FP|-0.017|-0.005|-0.010|-0.036|-0.070|-0.056|-0.104|
|F_TOKEN|-0.010|-0.000|-0.003|-0.032|-0.048|-0.056|-0.069|
|G_SEAM_GEO|-0.008|+0.000|-0.005|-0.039|-0.051|-0.044|-0.113|
|M_TOKEN_CE|-0.014|+0.015|+0.001|-0.050|-0.049|-0.045|-0.124|
|R_INCIDENT|-0.017|-0.002|-0.009|-0.036|-0.080|-0.042|-0.079|

## glt_dual_static_finetune_formal_grid/summary.json

本地来源（原文件不上传）：`results/glt_dual_static_finetune_formal_grid/summary.json`。

来源声明：None；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

Legacy eight-task grid: source std ddof=1; macro7 is computed here from saved seven means, not the source macro8.

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|concat|0.917±0.027|0.897±0.024|0.897±0.004|0.767±0.055|0.749±0.081|0.815±0.080|0.279±0.097|0.760|
|kfuse|0.900±0.022|0.893±0.027|0.882±0.011|0.750±0.077|0.721±0.082|0.807±0.069|0.226±0.095|0.740|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|concat|-0.017|-0.026|-0.014|-0.072|-0.038|-0.041|-0.144|
|kfuse|-0.034|-0.030|-0.029|-0.089|-0.066|-0.049|-0.197|

## glt_ft_generalization_20261008/r1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_ft_generalization_20261008/r1/final_review/seven_task_results.json`。

来源声明：five shared folds; seed42+fold; ddof0; old test exposed during development; controlled project comparison only; paper Egc/Xc cohort and Xc scaling differ; not blind/SOTA/multiseed；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|FT_LOW_LR|0.921±0.012|0.915±0.013|0.905±0.014|0.808±0.022|0.745±0.058|0.811±0.077|0.358±0.157|0.780|
|FT_PARTIAL|0.917±0.017|0.913±0.018|0.897±0.010|0.784±0.038|0.719±0.100|0.818±0.055|0.338±0.137|0.769|
|FT_REG|0.917±0.012|0.926±0.013|0.903±0.017|0.803±0.020|0.743±0.078|0.802±0.040|0.351±0.193|0.778|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|FT_LOW_LR|-0.013|-0.008|-0.006|-0.031|-0.042|-0.045|-0.065|
|FT_PARTIAL|-0.017|-0.010|-0.014|-0.055|-0.068|-0.038|-0.085|
|FT_REG|-0.017|+0.003|-0.008|-0.036|-0.044|-0.054|-0.072|

## glt_ft_robustness_20261011/r1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_ft_robustness_20261011/r1/final_review/seven_task_results.json`。

来源声明：seven task five-fold mean, seed42+fold, std ddof0; shared TEST used for development selection, numeric paper comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|D|0.925±0.015|0.941±0.006|0.909±0.014|0.799±0.023|0.762±0.055|0.846±0.061|0.343±0.105|0.789|
|DG|0.926±0.017|0.938±0.009|0.910±0.014|0.781±0.020|0.747±0.064|0.808±0.081|0.316±0.079|0.775|
|G|0.916±0.011|0.937±0.010|0.915±0.011|0.799±0.027|0.747±0.053|0.803±0.053|0.319±0.134|0.777|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|D|-0.009|+0.018|-0.002|-0.040|-0.025|-0.010|-0.080|
|DG|-0.008|+0.015|-0.001|-0.058|-0.040|-0.048|-0.107|
|G|-0.018|+0.014|+0.004|-0.040|-0.040|-0.053|-0.104|

## glt_fusion_20261007/r1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_fusion_20261007/r1/final_review/seven_task_results.json`。

来源声明：paper5_outer; seed42+fold; shared split; ddof0; not independent blind test; paper protocol equivalence unverified; no multi-seed stability claim；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B0|0.916±0.012|0.926±0.013|0.900±0.018|0.803±0.020|0.737±0.073|0.802±0.040|0.292±0.198|0.768|
|G_FGR|0.923±0.020|0.921±0.014|0.905±0.016|0.790±0.040|0.718±0.062|0.794±0.080|0.334±0.199|0.769|
|G_NORM|0.923±0.013|0.919±0.010|0.903±0.018|0.818±0.029|0.704±0.074|0.769±0.073|0.334±0.145|0.767|
|K0|0.927±0.020|0.925±0.024|0.910±0.013|0.796±0.024|0.719±0.063|0.820±0.058|0.324±0.109|0.775|
|K_MD|0.925±0.014|0.929±0.015|0.912±0.015|0.800±0.021|0.701±0.057|0.790±0.068|0.289±0.172|0.764|
|K_MD_A|0.918±0.018|0.937±0.012|0.914±0.015|0.791±0.015|0.710±0.075|0.846±0.055|0.338±0.179|0.779|
|T0|0.921±0.019|0.931±0.009|0.913±0.013|0.791±0.008|0.722±0.071|0.830±0.062|0.391±0.109|0.785|
|T0_A|0.925±0.015|0.924±0.011|0.912±0.015|0.784±0.032|0.709±0.070|0.802±0.079|0.294±0.130|0.764|
|TK_MD|0.928±0.018|0.926±0.007|0.917±0.011|0.790±0.024|0.736±0.070|0.813±0.046|0.273±0.191|0.769|
|TK_MD_A|0.914±0.012|0.930±0.011|0.918±0.012|0.792±0.043|0.705±0.057|0.775±0.083|0.344±0.144|0.768|
|TK_MD_L|0.921±0.021|0.934±0.011|0.911±0.013|0.803±0.019|0.748±0.082|0.799±0.048|0.301±0.144|0.774|
|T_RAW|0.920±0.012|0.924±0.011|0.905±0.014|0.800±0.013|0.706±0.063|0.792±0.058|0.306±0.148|0.765|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|B0|-0.018|+0.003|-0.011|-0.036|-0.050|-0.054|-0.131|
|G_FGR|-0.011|-0.002|-0.006|-0.049|-0.069|-0.062|-0.089|
|G_NORM|-0.011|-0.004|-0.008|-0.021|-0.083|-0.087|-0.089|
|K0|-0.007|+0.002|-0.001|-0.043|-0.068|-0.036|-0.099|
|K_MD|-0.009|+0.006|+0.001|-0.039|-0.086|-0.066|-0.134|
|K_MD_A|-0.016|+0.014|+0.003|-0.048|-0.077|-0.010|-0.085|
|T0|-0.013|+0.008|+0.002|-0.048|-0.065|-0.026|-0.032|
|T0_A|-0.009|+0.001|+0.001|-0.055|-0.078|-0.054|-0.129|
|TK_MD|-0.006|+0.003|+0.006|-0.049|-0.051|-0.043|-0.150|
|TK_MD_A|-0.020|+0.007|+0.007|-0.047|-0.082|-0.081|-0.079|
|TK_MD_L|-0.013|+0.011|+0.000|-0.036|-0.039|-0.057|-0.122|
|T_RAW|-0.014|+0.001|-0.006|-0.039|-0.081|-0.064|-0.117|

## glt_optimization_20261011/r1/S1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_optimization_20261011/r1/S1/final_review/seven_task_results.json`。

来源声明：seven task five-fold mean, seed42+fold, std ddof0; shared TEST used for development selection, numeric paper comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|F|0.932±0.021|0.935±0.010|0.908±0.019|0.802±0.009|0.765±0.063|0.808±0.052|0.299±0.127|0.779|
|R|0.928±0.016|0.941±0.009|0.913±0.014|0.802±0.034|0.765±0.051|0.812±0.059|0.321±0.121|0.783|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|F|-0.002|+0.012|-0.003|-0.037|-0.022|-0.048|-0.124|
|R|-0.006|+0.018|+0.002|-0.037|-0.022|-0.044|-0.102|

## glt_optimization_20261011/r2/S2/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_optimization_20261011/r2/S2/final_review/seven_task_results.json`。

来源声明：seven task five-fold mean, seed42+fold, std ddof0; shared TEST used for development selection, numeric paper comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B_FT|0.924±0.010|0.934±0.007|0.913±0.015|0.794±0.039|0.769±0.040|0.798±0.066|0.309±0.112|0.777|
|L_FT|0.921±0.015|0.933±0.015|0.906±0.017|0.805±0.018|0.723±0.065|0.813±0.088|0.282±0.118|0.769|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|B_FT|-0.010|+0.011|+0.002|-0.045|-0.018|-0.058|-0.114|
|L_FT|-0.013|+0.010|-0.005|-0.034|-0.064|-0.043|-0.141|

## glt_optimization_20261011/r3/S3/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_optimization_20261011/r3/S3/final_review/seven_task_results.json`。

来源声明：seven task five-fold mean, seed42+fold, std ddof0; shared TEST used for development selection, numeric paper comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|M_A|0.922±0.011|0.929±0.007|0.908±0.014|0.792±0.040|0.751±0.052|0.824±0.046|0.321±0.147|0.778|
|M_G|0.924±0.020|0.935±0.006|0.912±0.013|0.808±0.016|0.715±0.067|0.796±0.057|0.302±0.106|0.770|
|M_L|0.923±0.017|0.925±0.008|0.901±0.016|0.783±0.031|0.730±0.065|0.830±0.042|0.327±0.141|0.774|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|M_A|-0.012|+0.006|-0.003|-0.047|-0.036|-0.032|-0.102|
|M_G|-0.010|+0.012|+0.001|-0.031|-0.072|-0.060|-0.121|
|M_L|-0.011|+0.002|-0.010|-0.056|-0.057|-0.026|-0.096|

## glt_optimization_20261011/r4/U1/final_review/recovery1/seven_task_results.json

本地来源（原文件不上传）：`results/glt_optimization_20261011/r4/U1/final_review/recovery1/seven_task_results.json`。

来源声明：seven task five-fold mean, seed42+fold, std ddof0; shared TEST used for development selection, numeric paper comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|U0_2D|0.925±0.020|0.931±0.009|0.904±0.016|0.809±0.024|0.732±0.062|0.815±0.045|0.331±0.108|0.778|
|U0_FUSED|0.924±0.018|0.941±0.012|0.912±0.020|0.799±0.023|0.776±0.046|0.821±0.059|0.336±0.119|0.787|
|UA_2D|0.927±0.016|0.926±0.010|0.905±0.016|0.803±0.029|0.768±0.061|0.813±0.065|0.373±0.107|0.788|
|UA_FUSED|0.924±0.017|0.933±0.005|0.909±0.017|0.805±0.030|0.697±0.091|0.803±0.079|0.326±0.087|0.771|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|U0_2D|-0.009|+0.008|-0.007|-0.030|-0.055|-0.041|-0.092|
|U0_FUSED|-0.010|+0.018|+0.001|-0.040|-0.011|-0.035|-0.087|
|UA_2D|-0.007|+0.003|-0.006|-0.036|-0.019|-0.043|-0.050|
|UA_FUSED|-0.010|+0.010|-0.002|-0.034|-0.090|-0.053|-0.097|

## glt_optimization_20261011/r5/F1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_optimization_20261011/r5/F1/final_review/seven_task_results.json`。

来源声明：seven tasks, five folds seed42+fold, std ddof0; shared TEST development results, paper numeric comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|F1_MUL|0.921±0.016|0.937±0.008|0.914±0.013|0.804±0.034|0.793±0.033|0.792±0.060|0.294±0.127|0.779|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|F1_MUL|-0.013|+0.014|+0.003|-0.035|+0.006|-0.064|-0.129|

## glt_optimization_20261011/r6/S4/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_optimization_20261011/r6/S4/final_review/seven_task_results.json`。

来源声明：seven tasks, five folds seed42+fold, std ddof0; shared TEST development results, paper numeric comparison only; historical four-rank references, not matched torsion ablation；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|T_FCE|0.924±0.020|0.932±0.016|0.917±0.011|0.797±0.022|0.737±0.018|0.791±0.067|0.370±0.107|0.781|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|T_FCE|-0.010|+0.009|+0.006|-0.042|-0.050|-0.065|-0.053|

## glt_optimization_20261011/r7/N1/final_review/recovery1/seven_task_results.json

本地来源（原文件不上传）：`results/glt_optimization_20261011/r7/N1/final_review/recovery1/seven_task_results.json`。

来源声明：seven tasks five folds seed42+fold, ddof=0; shared TEST development; paper numeric comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|F_CE|0.927±0.016|0.936±0.012|0.913±0.019|0.763±0.077|0.755±0.055|0.828±0.046|0.231±0.163|0.765|
|UA_2D|0.933±0.023|0.928±0.009|0.904±0.013|0.799±0.026|0.716±0.067|0.808±0.062|0.362±0.131|0.779|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|F_CE|-0.007|+0.013|+0.002|-0.076|-0.032|-0.028|-0.192|
|UA_2D|-0.001|+0.005|-0.007|-0.040|-0.071|-0.048|-0.061|

## glt_optimization_20261011/r9/S5/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_optimization_20261011/r9/S5/final_review/seven_task_results.json`。

来源声明：seven tasks, five folds seed42+fold, std ddof0; matched two-rank PT and uniform_u1 FT; added branch capacity remains; shared TEST development results, paper numeric comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|F_CE_2R|0.928±0.014|0.931±0.015|0.912±0.013|0.799±0.018|0.685±0.103|0.787±0.082|0.326±0.131|0.767|
|T_FCE_reused|0.924±0.020|0.932±0.016|0.917±0.011|0.797±0.022|0.737±0.018|0.791±0.067|0.370±0.107|0.781|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|F_CE_2R|-0.006|+0.008|+0.001|-0.040|-0.102|-0.069|-0.097|
|T_FCE_reused|-0.010|+0.009|+0.006|-0.042|-0.050|-0.065|-0.053|

## glt_periodic_seam_20261009/r1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_periodic_seam_20261009/r1/final_review/seven_task_results.json`。

来源声明：shared development folds; Egc3380 vs paper4124 Xc432 vs430; split/scaling equivalence unverified; numeric comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|F_CE|0.924±0.018|0.941±0.012|0.914±0.012|0.799±0.023|0.776±0.046|0.821±0.059|0.324±0.160|0.786|
|F_RAW|0.917±0.010|0.924±0.016|0.904±0.018|0.803±0.023|0.723±0.054|0.781±0.057|0.317±0.197|0.767|
|P_CE|0.928±0.019|0.936±0.006|0.910±0.012|0.808±0.018|0.707±0.064|0.815±0.064|0.309±0.125|0.773|
|P_RAW|0.923±0.013|0.929±0.013|0.902±0.016|0.789±0.013|0.717±0.073|0.819±0.048|0.277±0.124|0.765|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|F_CE|-0.010|+0.018|+0.003|-0.040|-0.011|-0.035|-0.099|
|F_RAW|-0.017|+0.001|-0.007|-0.036|-0.064|-0.075|-0.106|
|P_CE|-0.006|+0.013|-0.001|-0.031|-0.080|-0.041|-0.114|
|P_RAW|-0.011|+0.006|-0.009|-0.050|-0.070|-0.037|-0.146|

## glt_pred_20260918/s5_formal_review/summary.json

本地来源（原文件不上传）：`results/glt_pred_20260918/s5_formal_review/summary.json`。

来源声明：None；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

Retired route; source contains macro8; no protocol-equivalence claim.

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|b_fp|0.906±0.020|0.912±0.015|0.904±0.003|0.773±0.044|0.746±0.045|0.832±0.048|0.281±0.120|0.765|
|b_none|0.913±0.022|0.913±0.017|0.896±0.012|0.761±0.053|0.765±0.053|0.839±0.041|0.283±0.088|0.767|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|b_fp|-0.028|-0.011|-0.007|-0.066|-0.041|-0.024|-0.142|
|b_none|-0.021|-0.010|-0.015|-0.078|-0.022|-0.017|-0.140|

## glt_quality_fusion_20261010/r1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/glt_quality_fusion_20261010/r1/final_review/seven_task_results.json`。

来源声明：five folds seed42+fold ddof0; shared TEST used for development selection; numeric paper comparison only；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Q|0.898±0.017|0.918±0.007|0.904±0.016|0.753±0.043|0.729±0.048|0.835±0.075|0.337±0.138|0.768|
|Q_A|0.903±0.016|0.912±0.012|0.907±0.015|0.761±0.027|0.720±0.056|0.816±0.068|0.281±0.180|0.757|
|Q_G|0.904±0.019|0.917±0.013|0.901±0.019|0.783±0.024|0.737±0.058|0.821±0.072|0.272±0.176|0.762|
|Q_GA|0.915±0.018|0.914±0.012|0.903±0.017|0.777±0.030|0.717±0.060|0.778±0.097|0.333±0.114|0.762|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|Q|-0.036|-0.005|-0.007|-0.086|-0.058|-0.021|-0.086|
|Q_A|-0.031|-0.011|-0.004|-0.078|-0.067|-0.040|-0.142|
|Q_G|-0.030|-0.006|-0.010|-0.056|-0.050|-0.035|-0.151|
|Q_GA|-0.019|-0.009|-0.008|-0.062|-0.070|-0.078|-0.090|

## mcl_ph_20260921/p2/paper5_official_r13_1/paper5_test.json

本地来源（原文件不上传）：`results/mcl_ph_20260921/p2/paper5_official_r13_1/paper5_test.json`。

来源声明：population standard deviation over five fold R2 values；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

Five-task historical run with known angle/PH implementation limitations; macro7 is unavailable.

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|glt_ref|0.921±0.010|0.930±0.016|未完成／不可比较|0.797±0.011|0.735±0.050|0.801±0.054|未完成／不可比较|不可计算|
|m_cat|0.922±0.015|0.916±0.019|未完成／不可比较|0.780±0.017|0.726±0.048|0.802±0.031|未完成／不可比较|不可计算|
|m_gate|0.919±0.014|0.927±0.011|未完成／不可比较|0.791±0.024|0.720±0.071|0.791±0.045|未完成／不可比较|不可计算|
|m_xattn|0.924±0.008|0.929±0.007|未完成／不可比较|0.792±0.019|0.733±0.056|0.806±0.061|未完成／不可比较|不可计算|
|o8_only|0.917±0.011|0.918±0.012|未完成／不可比较|0.796±0.024|0.730±0.055|0.818±0.049|未完成／不可比较|不可计算|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|glt_ref|-0.013|+0.007|不可比较|-0.042|-0.052|-0.055|不可比较|
|m_cat|-0.012|-0.007|不可比较|-0.059|-0.061|-0.054|不可比较|
|m_gate|-0.015|+0.004|不可比较|-0.048|-0.067|-0.065|不可比较|
|m_xattn|-0.010|+0.006|不可比较|-0.047|-0.054|-0.050|不可比较|
|o8_only|-0.017|-0.005|不可比较|-0.043|-0.057|-0.038|不可比较|

## mcl_ph_opt_20261006/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/mcl_ph_opt_20261006/final_review/seven_task_results.json`。

来源声明：None；独立审查字段：`非独立人员审查`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

Retained historical predictions; no new candidate formal inference.

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|0.928（std未记录）|0.918（std未记录）|0.901（std未记录）|0.803（std未记录）|0.717（std未记录）|0.800（std未记录）|0.319（std未记录）|0.770|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|-0.006|-0.005|-0.010|-0.036|-0.070|-0.056|-0.104|

## mcl_ph_opt_20261006/r3/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/mcl_ph_opt_20261006/r3/final_review/seven_task_results.json`。

来源声明：None；独立审查字段：`非独立人员审查`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

Retained historical predictions; no new candidate formal inference.

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|0.928（std未记录）|0.918（std未记录）|0.901（std未记录）|0.803（std未记录）|0.717（std未记录）|0.800（std未记录）|0.319（std未记录）|0.770|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|-0.006|-0.005|-0.010|-0.036|-0.070|-0.056|-0.104|

## mcl_ph_r4_20261002/fast_campaign_04/seven_task_results.json

本地来源（原文件不上传）：`results/mcl_ph_r4_20261002/fast_campaign_04/seven_task_results.json`。

来源声明：None；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|0.928±0.016|0.918±0.014|0.901±0.014|0.803±0.027|0.717±0.053|0.800±0.046|0.226±0.219|0.756|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|-0.006|-0.005|-0.010|-0.036|-0.070|-0.056|-0.197|

## mcl_ph_r5_20261004/seven_task_results.json

本地来源（原文件不上传）：`results/mcl_ph_r5_20261004/seven_task_results.json`。

来源声明：paper5_outer; five-fold arithmetic mean R2, seed42+fold, std ddof0；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|0.928±0.016|0.918±0.014|0.901±0.014|0.803±0.027|0.717±0.053|0.800±0.046|0.319±0.192|0.770|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|-0.006|-0.005|-0.010|-0.036|-0.070|-0.056|-0.104|

## mcl_ph_r6_20261004/seven_task_results.json

本地来源（原文件不上传）：`results/mcl_ph_r6_20261004/seven_task_results.json`。

来源声明：paper5_outer; five-fold arithmetic mean R2, seed42+fold, std ddof0；独立审查字段：`not present; consult history/review artifacts`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|0.928±0.016|0.918±0.014|0.901±0.014|0.803±0.027|0.717±0.053|0.800±0.046|0.319±0.192|0.770|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|-0.006|-0.005|-0.010|-0.036|-0.070|-0.056|-0.104|

## sota_ft_20261005/seven_task_retained_results.json

本地来源（原文件不上传）：`results/sota_ft_20261005/seven_task_retained_results.json`。

来源声明：None；独立审查字段：`本新周期尚未执行`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

|政策|Eea|Egb|Egc|Ei|EPS|Nc|Xc|macro7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|0.928（std未记录）|0.918（std未记录）|0.901（std未记录）|0.803（std未记录）|0.717（std未记录）|0.800（std未记录）|0.319（std未记录）|0.770|

|政策−论文|Eea|Egb|Egc|Ei|EPS|Nc|Xc|
|---|---:|---:|---:|---:|---:|---:|---:|
|RETAINED_OR_SELECTED_REUSED|-0.006|-0.005|-0.010|-0.036|-0.070|-0.056|-0.104|

## ua2d_fusion_20261017/g1_fp_gate/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/ua2d_fusion_20261017/g1_fp_gate/final_review/seven_task_results.json`。

来源声明：existing shared validation/TEST development exposed; exploratory only; paper data/split equivalence unverified；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

Failed G1 placeholder; not a completed formal experiment.

## ua2d_fusion_20261017/g1_fp_gate_recovery1/final_review/seven_task_results.json

本地来源（原文件不上传）：`results/ua2d_fusion_20261017/g1_fp_gate_recovery1/final_review/seven_task_results.json`。

来源声明：shared validation/TEST exposed; exploratory only; no fair paper comparison；独立审查字段：`NOT_PERFORMED`。该字段可能早于后续审查，审查沿革另见PROJECT_HISTORY，不自动补签。

Failed G1 placeholder; not a completed formal experiment.
