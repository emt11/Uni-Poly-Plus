# Uni-Poly-Plus 基线结果索引

本文件索引保留路线 `GLT-V2 revision-2`（含旧 N+1/N+2 蒸馏与使用独立 validation 的 C0/C1/C2 对照）和 `Atomic-PC W-CAMR-v2`，并保留已退役路线 `MTS-GLT-v2-Base-5k` 的历史结果记录。完整合同与路线范围见 [`PIPELINE.md`](PIPELINE.md)。

## 已退役基线：MTS-GLT-v2-Base-5k（历史记录）

该路线的模型、预训练、下游代码及 `results/`、`logs/`、`pretrained_models/` 产物已于 2026-09-10 删除，下表为删除前的正式结果，不再可复现或重新评估。唯一保留的产物是 checkpoint，因为 Atomic-PC W-CAMR-v2 运行时读取它。

```text
name        MTS-GLT-v2-Base-5k
version     mts_glt_v2_base_5k_v1
checkpoint  results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth
schedule    direct_joint (LR warmup 5 epochs)
seed        42
protocol    historical_shared5 (非独立盲测)
```

| Task | Fusion R² mean |
|---|---:|
| eat | 0.984101 |
| eea | 0.922432 |
| egb | 0.941382 |
| egc | 0.919324 |
| ei | 0.828284 |
| eps | 0.818812 |
| nc | 0.871385 |
| xc | 0.463366 |

```text
macro8 R²                  0.8436358322
O8-only matched comparator 0.8407410869
descriptive delta          +0.0028947453
positive tasks             7/8
independent blind test     false
```

这里的 `Fusion − O8-only` 是同一 checkpoint、同一 split、同一 seed 下的描述性 matched 对照，不单独构成新的生产路线，也不宣称因果 interaction。

## 证据文件

- 成对下游汇总：[`results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json`](results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json)（保留）。
- checkpoint：[`results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth`](results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth)（保留，W-CAMR 依赖）。
- 已删除：基线与预训练配置、DDP smoke 配置、`baseline_manifest.json`、预训练 `resolved_input.json` 与 `final_report.{json,md}`。

## 运行边界

- 保留路线为 graph-only `GLT-V2 revision-2` 与 `Atomic-PC W-CAMR-v2`；已退役基线的数字只作历史对照。
- validation 与 test 共用 fold，结果不能当作 independent blind test。
- 本次仓库整理没有重新训练、重跑消融或扩大评估范围。
- 后续默认只做必要的局部测试和 smoke；新实验须使用独立配置与输出目录，并由用户明确授权。

## 保留实验路线：Atomic-PC W-CAMR-v2

```text
name             Atomic-PC W-CAMR-v2
status           retained_experimental_route
checkpoint       results/original_mips_atomic_pc_w_camr_v2/pretraining/w_camr_checkpoint.pt
optimizer steps  1504
seed             42
protocol         historical_shared5 (非独立盲测)
loaded downstream component  atomic_point_encoder only
macro8 R²        0.8432478932
```

证据为 `results/original_mips_atomic_pc_w_camr_v2/summary.json` 和 `results/original_mips_atomic_pc_w_camr_v2/downstream/aggregate_summary.json`。该路线是保留的实验路线，不是生产基线；`MTS-GLT-v2-Base-5k` 已于 2026-09-10 退役，只保留历史数字。

## N+1 / N+2 两阶段蒸馏正式结果

两个版本均完成教师 5k、学生 20k，并使用各自学生 20k 的 O8＋MD200 部署包完成 8-task × 5-fold。下表为五折样本标准差；MAE/RMSE 越低越好。

| Task | N+2 R² | N+2 MAE | N+2 RMSE | N+1 R² | N+1 MAE | N+1 RMSE | N+2−N+1 R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| eat | 0.980551 ± 0.006772 | 0.031994 ± 0.004637 | 0.049466 ± 0.009808 | 0.982898 ± 0.006419 | 0.030710 ± 0.005042 | 0.046214 ± 0.009233 | -0.002347 |
| eea | 0.924534 ± 0.025370 | 0.218389 ± 0.032413 | 0.289464 ± 0.037620 | 0.919766 ± 0.031853 | 0.218457 ± 0.034029 | 0.297870 ± 0.052695 | +0.004768 |
| egb | 0.940773 ± 0.013317 | 0.350389 ± 0.051277 | 0.471013 ± 0.065306 | 0.936586 ± 0.011389 | 0.354724 ± 0.052818 | 0.489212 ± 0.067657 | +0.004187 |
| egc | 0.920638 ± 0.003937 | 0.289986 ± 0.008505 | 0.439960 ± 0.009344 | 0.917494 ± 0.014444 | 0.289057 ± 0.015652 | 0.447407 ± 0.038800 | +0.003144 |
| ei | 0.828404 ± 0.084029 | 0.261364 ± 0.028445 | 0.396272 ± 0.094262 | 0.841059 ± 0.060719 | 0.259691 ± 0.027333 | 0.384127 ± 0.070301 | -0.012655 |
| eps | 0.805177 ± 0.071640 | 0.327127 ± 0.033110 | 0.475201 ± 0.054925 | 0.809629 ± 0.065464 | 0.328661 ± 0.028453 | 0.470600 ± 0.045567 | -0.004452 |
| nc | 0.861787 ± 0.028201 | 0.057967 ± 0.008759 | 0.087708 ± 0.010735 | 0.872521 ± 0.032651 | 0.054191 ± 0.006329 | 0.083850 ± 0.010210 | -0.010734 |
| xc | 0.393972 ± 0.061916 | 13.480219 ± 1.021704 | 18.324177 ± 1.202078 | 0.374429 ± 0.068675 | 13.709919 ± 0.921413 | 18.600464 ± 1.011366 | +0.019543 |

```text
N+2 macro8 R²             0.8319794666
N+1 macro8 R²             0.8317977899
paired N+2 − N+1          +0.0001816768
positive tasks / folds    4/8, 20/40
completed units           80/80
independent blind test    false
```

旧 GLT-v2 Fusion 的历史 macro8 R² 为 `0.8436358322`，描述性地高于 N+2 `0.0116563656`、高于 N+1 `0.0118380424`。由于新旧路线同时改变 O8、预训练、MD 融合和下游 3D 使用方式，这不是蒸馏机制的独立因果比较。N+2 相对 N+1 只有很小的 macro 正差，且正增量覆盖仅 4/8 tasks、20/40 folds，不能解释为广泛稳定优势。

证据文件：

- 汇总 JSON：[`results/mts_glt_v2_distill/comparison/summary.json`](results/mts_glt_v2_distill/comparison/summary.json)。
- 80-fold 明细：[`results/mts_glt_v2_distill/comparison/all_fold_metrics.csv`](results/mts_glt_v2_distill/comparison/all_fold_metrics.csv)。
- Markdown 报告：[`results/mts_glt_v2_distill/comparison/final_report.md`](results/mts_glt_v2_distill/comparison/final_report.md)。
- 执行 manifest：[`results/mts_glt_v2_distill/comparison/execution_manifest.json`](results/mts_glt_v2_distill/comparison/execution_manifest.json)。
- N+2 部署包：[`results/mts_glt_v2_distill/n_plus_2/student/student_deploy_020k.pt`](results/mts_glt_v2_distill/n_plus_2/student/student_deploy_020k.pt)。
- N+1 部署包：[`results/mts_glt_v2_distill/n_plus_1/student/student_deploy_020k.pt`](results/mts_glt_v2_distill/n_plus_1/student/student_deploy_020k.pt)。
- N+2 教师/学生完整状态：[`teacher_005k.pt`](results/mts_glt_v2_distill/n_plus_2/teacher/teacher_005k.pt)、[`student_020k.pt`](results/mts_glt_v2_distill/n_plus_2/student/student_020k.pt)。
- N+1 教师/学生完整状态：[`teacher_005k.pt`](results/mts_glt_v2_distill/n_plus_1/teacher/teacher_005k.pt)、[`student_020k.pt`](results/mts_glt_v2_distill/n_plus_1/student/student_020k.pt)。
- 正式链日志：[`logs/mts_glt_v2_distill/remaining_chain.log`](logs/mts_glt_v2_distill/remaining_chain.log)。

两个版本的 `historical_shared5` 都让 validation 与 test 使用同一 outer fold；这是用户明确指定且与既有协议一致的共享开发折评估，不是独立盲测。N+1/N+2 是整体方案比较；本周期没有无蒸馏 control，因此不能单独估计蒸馏的因果收益。

## GLT revision-2 C0/C1/C2 正式对照

本实验使用新协议 `outer5_inner20`：保留原 outer-test 五折，对每折 outer-train 固定划出 20% validation。三组共享学生初始化、样本顺序、训练预算和下游协议；C0 无教师，C1/C2 分别使用修复后的 N+1/N+2 GLT 教师。下游只迁移 O8＋MD200，不读取 GLT 或坐标。三组各完成 40/40，总计 120/120 个 task/fold。下表为五折 mean ± population std；MAE/RMSE 越低越好。

| Task | C0 R² | C0 MAE | C0 RMSE | C1 R² | C1 MAE | C1 RMSE | C2 R² | C2 MAE | C2 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eat | 0.972401 ± 0.009008 | 0.036400 ± 0.004410 | 0.058572 ± 0.009657 | 0.975402 ± 0.010655 | 0.034924 ± 0.004426 | 0.054741 ± 0.009862 | 0.966449 ± 0.020529 | 0.040331 ± 0.011659 | 0.062458 ± 0.016901 |
| eea | 0.908299 ± 0.021807 | 0.239956 ± 0.020367 | 0.320671 ± 0.030216 | 0.899036 ± 0.023961 | 0.249230 ± 0.021319 | 0.336367 ± 0.026041 | 0.895270 ± 0.024329 | 0.252882 ± 0.024498 | 0.343519 ± 0.028751 |
| egb | 0.922540 ± 0.011604 | 0.401159 ± 0.032095 | 0.538110 ± 0.039795 | 0.925446 ± 0.014003 | 0.384617 ± 0.037256 | 0.528033 ± 0.059495 | 0.926759 ± 0.013571 | 0.389609 ± 0.046097 | 0.524968 ± 0.067795 |
| egc | 0.899035 ± 0.010696 | 0.318731 ± 0.012676 | 0.495649 ± 0.024284 | 0.902146 ± 0.009464 | 0.314275 ± 0.009393 | 0.488120 ± 0.022561 | 0.900234 ± 0.007634 | 0.317621 ± 0.008331 | 0.492908 ± 0.014720 |
| ei | 0.789809 ± 0.058519 | 0.305873 ± 0.028501 | 0.444533 ± 0.057597 | 0.804903 ± 0.059197 | 0.298505 ± 0.032501 | 0.428934 ± 0.064312 | 0.793066 ± 0.051650 | 0.303517 ± 0.035805 | 0.441594 ± 0.051598 |
| eps | 0.774609 ± 0.044624 | 0.353823 ± 0.034902 | 0.518770 ± 0.035683 | 0.765674 ± 0.050371 | 0.350759 ± 0.033505 | 0.528104 ± 0.038384 | 0.767894 ± 0.051061 | 0.357827 ± 0.021160 | 0.524030 ± 0.029303 |
| nc | 0.843265 ± 0.058911 | 0.058995 ± 0.011835 | 0.092008 ± 0.016361 | 0.846487 ± 0.051919 | 0.058362 ± 0.010283 | 0.091452 ± 0.016999 | 0.850562 ± 0.050044 | 0.057302 ± 0.007844 | 0.090023 ± 0.014446 |
| xc | 0.338937 ± 0.118666 | 14.096393 ± 1.231748 | 19.049778 ± 1.440842 | 0.263701 ± 0.060711 | 15.181088 ± 0.834342 | 20.210593 ± 1.258645 | 0.243918 ± 0.153272 | 15.162622 ± 1.404960 | 20.331843 ± 1.642400 |

```text
macro8 R² C0                  0.8061119181
macro8 R² C1                  0.7978494262
macro8 R² C2                  0.7930190200
C1 − C0                      -0.0082624919   (5/8 tasks, 18/40 folds positive)
C2 − C0                      -0.0130928981   (4/8 tasks, 21/40 folds positive)
C2 − C1                      -0.0048304062   (3/8 tasks, 20/40 folds positive)
completed units               120/120
independent blind test        false
```

两条蒸馏路线的 macro R² 均低于 matched C0，因此本轮正式证据不支持 revision-2 GLT 教师蒸馏优于 O8＋MD200 无蒸馏对照。C1 虽有 5/8 个任务均值为正，但 macro 为负且仅 18/40 folds 为正；C2 也未满足预设候选门。按预设决策，本轮停止追加实验，后续如另行授权应优先诊断教师监督与学生 readout 对齐。

### 训练成本

|轨迹|Updates|阶段 wall|峰值单卡显存|
|-|---:|---:|---:|
|C0 student|20,000|1.774 h|1.400 GiB|
|C1 teacher|5,000|0.519 h|1.232 GiB|
|C1 student|20,000|2.167 h|2.500 GiB|
|C2 teacher|5,000|0.508 h|1.241 GiB|
|C2 student|20,000|2.305 h|2.505 GiB|

下游 C0/C1/C2 分别累计 28,235、26,904、26,527 个 optimizer updates，对应 0.727、0.692、0.691 one-GPU hours。所有 fold 均由 validation 选择最佳 checkpoint，再执行一次 outer-test 预测；没有跳过或以其他 fold 替代失败单元。

### 正式产物

- 汇总 JSON：[`results/mts_glt_distill_repair_control/comparison/summary.json`](results/mts_glt_distill_repair_control/comparison/summary.json)。
- 120-fold 指标：[`results/mts_glt_distill_repair_control/comparison/all_fold_metrics.csv`](results/mts_glt_distill_repair_control/comparison/all_fold_metrics.csv)。
- 完整 Markdown 报告：[`results/mts_glt_distill_repair_control/comparison/final_report.md`](results/mts_glt_distill_repair_control/comparison/final_report.md)。
- 初始化审计：[`results/mts_glt_distill_repair_control/comparison/initialization_audit.json`](results/mts_glt_distill_repair_control/comparison/initialization_audit.json)。
- C1/C2 教师：[`C1 teacher_005k.pt`](results/mts_glt_distill_repair_control/c1/teacher/teacher_005k.pt)、[`C2 teacher_005k.pt`](results/mts_glt_distill_repair_control/c2/teacher/teacher_005k.pt)。
- C0/C1/C2 可恢复学生状态：[`C0 student_020k.pt`](results/mts_glt_distill_repair_control/c0/student/student_020k.pt)、[`C1 student_020k.pt`](results/mts_glt_distill_repair_control/c1/student/student_020k.pt)、[`C2 student_020k.pt`](results/mts_glt_distill_repair_control/c2/student/student_020k.pt)。
- O8＋MD200 部署包：[`C0`](results/mts_glt_distill_repair_control/c0/student/student_deploy_020k.pt)、[`C1`](results/mts_glt_distill_repair_control/c1/student/student_deploy_020k.pt)、[`C2`](results/mts_glt_distill_repair_control/c2/student/student_deploy_020k.pt)。
- 三组正式下游根：[`C0`](results/mts_glt_distill_repair_control/c0/downstream/outer5_inner20)、[`C1`](results/mts_glt_distill_repair_control/c1/downstream/outer5_inner20)、[`C2`](results/mts_glt_distill_repair_control/c2/downstream/outer5_inner20)。
- 固定 split：[`data/splits/mips_outer5_inner20`](data/splits/mips_outer5_inner20)。
- revision-2 sidecar：[`N+1`](data/processed/mips_trimer_scage/periodic_line_glt_distill_v2/n_plus_1)、[`N+2`](data/processed/mips_trimer_scage/periodic_line_glt_distill_v2/n_plus_2)。
- 主执行日志：[`all_chain.log`](logs/mts_glt_distill_repair_control/all_chain.log)、[`reboot_direct_chain.log`](logs/mts_glt_distill_repair_control/reboot_direct_chain.log)。
- 最终报告日志：[`final_report_v3.log`](logs/mts_glt_distill_repair_control/final_report_v3.log)。
- 验收日志：[`tests_final_20260909.log`](logs/mts_glt_distill_repair_control/tests_final_20260909.log)、[`n0_rank_smoke_20260909.log`](logs/mts_glt_distill_repair_control/n0_rank_smoke_20260909.log)、[`resume_current_v5_20260909.log`](logs/mts_glt_distill_repair_control/resume_current_v5_20260909.log)。

执行期间主机发生两次重启，存活进程被终止；C2 学生从合法 checkpoint 恢复，放弃的 checkpoint 之后日志尾部被保留。C0 10k 中间状态曾使用旧 schema，继续前已保留旧文件、迁移为 repair schema 并严格加载。重启后的冷缓存造成间歇性 I/O 延迟，但没有改变训练步数、global batch、样本顺序或损失定义，最终所有 checkpoint、预测及指标均有限。

`outer5_inner20` 将 validation 与 outer test 分离，但样本仍属于项目已参与开发的数据，因此不是全新独立盲测。旧 GLT-v2 与旧 N+1/N+2 使用 `historical_shared5`，不能把约 `0.844`/`0.832` 与本节数字直接解释为性能升降；新旧路线也同时改变几何修复、O8、MD 融合及下游输入，不能作单机制归因。

## C0 迁移优化诊断与正式结果

本轮只读取 C0/C1/C2 的正式 `student_deploy_020k.pt`，没有重新预训练。冻结表示探针为 30/30，C0 分阶段微调为 40/40；三组均使用同一 `outer5_inner20` manifest。探针结果为五折 mean ± population std：

|组别|任务|R²|MAE|RMSE|相对 C0 R²|正配对 fold|既有全微调 R²|
|-|-|---:|---:|---:|---:|---:|---:|
|C0|xc|0.277892 ± 0.146566|15.108926 ± 1.234074|19.877955 ± 1.649107|+0.000000|0/5|0.338937|
|C0|eps|0.714566 ± 0.024595|0.403318 ± 0.038598|0.590661 ± 0.070592|+0.000000|0/5|0.774609|
|C1|xc|0.309335 ± 0.080669|15.100900 ± 1.087939|19.555496 ± 1.408404|+0.031444|2/5|0.263701|
|C1|eps|0.722274 ± 0.040104|0.385852 ± 0.057017|0.583429 ± 0.090673|+0.007708|3/5|0.765674|
|C2|xc|0.328509 ± 0.109164|14.615452 ± 1.651581|19.266855 ± 1.884372|+0.050617|4/5|0.243918|
|C2|eps|0.737073 ± 0.039626|0.377830 ± 0.057106|0.568118 ± 0.092739|+0.022507|3/5|0.767894|

30 个单元中 alpha=`100` 被选择 25 次、alpha=`10` 被选择 5 次。C1/C2 在两个任务的冻结线性探针均高于 C0，而对应既有全量微调并未保持这一优势；这更支持“微调适应存在问题”，而不是“C1/C2 冻结表示已弱于 C0”。该判断只覆盖 xc/eps，并且 Ridge 线性可读性不等于表示包含的全部任务信息。

C0 分阶段微调结果：

|Task|StageFT R²|StageFT MAE|StageFT RMSE|既有 C0 R²|R² 增量|
|-|---:|---:|---:|---:|---:|
|eat|0.958793 ± 0.016590|0.046044 ± 0.005331|0.071113 ± 0.012056|0.972401|-0.013608|
|eea|0.904372 ± 0.023816|0.245360 ± 0.018625|0.326952 ± 0.025995|0.908299|-0.003927|
|egb|0.916708 ± 0.019559|0.406902 ± 0.057912|0.557648 ± 0.081377|0.922540|-0.005832|
|egc|0.905070 ± 0.005275|0.321150 ± 0.016536|0.481178 ± 0.014898|0.899035|+0.006036|
|ei|0.777400 ± 0.045132|0.314611 ± 0.023179|0.460359 ± 0.038876|0.789809|-0.012409|
|eps|0.762523 ± 0.045449|0.360037 ± 0.030683|0.533109 ± 0.038950|0.774609|-0.012086|
|nc|0.833722 ± 0.049528|0.061150 ± 0.009301|0.095429 ± 0.015987|0.843265|-0.009544|
|xc|0.349718 ± 0.124205|14.247180 ± 1.633686|18.899933 ± 1.729051|0.338937|+0.010781|

```text
stageft macro8 R²          0.8010382929
existing C0 macro8 R²      0.8061119181
paired macro delta         -0.0050736252
positive tasks / folds     2/8, 15/40
best stage                 stage2 for 40/40 folds
optimizer updates          31,371
completed units            probes 30/30, staged FT 40/40
independent blind test     false
```

分阶段方案改善了 xc 和 egc，但另外六个任务下降，Macro8 为负；因此没有达到 `macro delta ≥ 0.005` 且至少 `6/8` 任务为正的多 seed 复核建议门槛，也不支持继续 sweep 冻结长度或学习率。40/40 最佳 checkpoint 都来自第二阶段；第一阶段模型确实参与全流程 validation 选择，但没有胜出。本比较是 head-first、optimizer 重置与随后联合微调的整体方案差异，不能把结果单独归因于某一个动作。

成本与执行：探针累计约 `0.168` one-GPU h；staged 纯训练累计 `0.641` one-GPU h，包含数据加载、validation、checkpoint 和 test 的 fold wall 累计 `2.136` one-GPU h；四卡正式调度墙钟约 34 分钟。正式命令分别为 `python scripts/run_mts_c0_transfer_probes.py --mode all --gpu-ids 0,1,2` 和 `python scripts/run_mts_c0_staged_finetune.py --gpu-ids 0,1,2,3`，均运行于 `tmux` session `Uni-Poly` 的独立 window。

证据文件：

- 完整报告：[`results/mts_c0_transfer_optimization/comparison/final_report.md`](results/mts_c0_transfer_optimization/comparison/final_report.md)。
- 机器汇总：[`summary.json`](results/mts_c0_transfer_optimization/comparison/summary.json)。
- 探针逐 fold 与逐任务：[`probe_fold_metrics.csv`](results/mts_c0_transfer_optimization/comparison/probe_fold_metrics.csv)、[`probe_task_summary.csv`](results/mts_c0_transfer_optimization/comparison/probe_task_summary.csv)。
- staged 逐 fold 与逐任务：[`stageft_fold_metrics.csv`](results/mts_c0_transfer_optimization/comparison/stageft_fold_metrics.csv)、[`stageft_task_summary.csv`](results/mts_c0_transfer_optimization/comparison/stageft_task_summary.csv)。
- 40 个正式 checkpoint/指标：[`results/mts_c0_transfer_optimization/staged_finetune/shards/42`](results/mts_c0_transfer_optimization/staged_finetune/shards/42)。
- 40 个正式预测：[`results/mts_c0_transfer_optimization/staged_finetune/predictions/42`](results/mts_c0_transfer_optimization/staged_finetune/predictions/42)。
- 正式调度日志：[`probes`](logs/mts_c0_transfer_optimization/probes/formal_scheduler.log)、[`staged`](logs/mts_c0_transfer_optimization/staged_finetune/formal_scheduler.log)、[`report`](logs/mts_c0_transfer_optimization/comparison/final_audit_v4.log)、[`tests`](logs/mts_c0_transfer_optimization/comparison/tests_final.log)。
- smoke：`results/mts_c0_transfer_optimization/smoke/`；风险相关测试为 `20 passed`。

执行中曾发现第一版 staged 正式启动的第一阶段未复刻 bias/LayerNorm no-decay 分组。受影响调度被停止，三个已完成单元和中断状态整体保留在 `results/mts_c0_transfer_optimization/staged_finetune_invalid_stage1_decay_all_20260909/`，没有进入正式汇总。修正并重新通过 2+2 epoch smoke 后，正式 40 单元从 C0 部署包在全新目录重新启动。

## C0 使用旧版 historical_shared5 的重新微调

本次从正式 C0 `student_deploy_020k.pt` 重新初始化每个 task/fold，仅使用
O8＋MD200，并按旧版 `historical_shared5` 协议完成八任务五折。该协议中每折
validation 与 test 是同一组 outer-fold 样本，因而以下结果不是独立测试，且不能
与 `outer5_inner20` 的 C0 `0.8061119181` 直接解释为同协议性能变化。

|Task|R2 mean +/- std|MAE mean +/- std|RMSE mean +/- std|
|-|-:|-:|-:|
|eat|0.982292 +/- 0.006456|0.029880 +/- 0.004694|0.046578 +/- 0.007307|
|eea|0.919152 +/- 0.020987|0.224236 +/- 0.021016|0.300432 +/- 0.024715|
|egb|0.936540 +/- 0.012528|0.363927 +/- 0.038554|0.487573 +/- 0.060265|
|egc|0.918656 +/- 0.003604|0.289830 +/- 0.009620|0.445470 +/- 0.010831|
|ei|0.817430 +/- 0.056626|0.278960 +/- 0.029788|0.412935 +/- 0.060620|
|eps|0.794402 +/- 0.053264|0.330770 +/- 0.017911|0.492142 +/- 0.036304|
|nc|0.871977 +/- 0.041372|0.054416 +/- 0.009306|0.083448 +/- 0.012097|
|xc|0.436482 +/- 0.063172|13.294133 +/- 0.556824|17.631919 +/- 0.651147|

```text
macro8 R2                    0.8346162015
completed units              40/40
optimizer updates            41,001
validation is test           true
independent blind test       false
```

逐项审计确认 40 个 checkpoint、40 个预测与 40 个指标全部存在且有限，split
identity 与 `data/splits/mips_shared5` 完全一致；每个任务的五个 test folds 恰好
覆盖全部样本一次。完整报告见
[`results/mts_c0_historical_shared5_rerun/comparison/final_report.md`](results/mts_c0_historical_shared5_rerun/comparison/final_report.md)，
正式产物位于
[`results/mts_c0_historical_shared5_rerun/formal`](results/mts_c0_historical_shared5_rerun/formal)，
调度日志位于
[`logs/mts_c0_historical_shared5_rerun/formal`](logs/mts_c0_historical_shared5_rerun/formal)。

### C1/C2 使用相同旧版五折的重新微调

C1、C2 随后使用与上述 C0 完全相同的 `historical_shared5` 配置各完成
40/40 个正式单元。每个 fold 分别从 C1 `n_plus_1` 或 C2 `n_plus_2` 的正式
20k O8＋MD200 部署包重新初始化；下游均不加载 GLT。

|组别|Macro8 R2|相对C0|正增量任务|正增量fold|
|-|-:|-:|-:|-:|
|C0|0.8346162015|--|--|--|
|C1|0.8326805138|-0.0019356877|4/8|21/40|
|C2|0.8289977937|-0.0056184078|5/8|22/40|

C2-C1 的 Macro8 R2 差值为 `-0.0036827201`，正增量任务 `4/8`、正增量
fold `22/40`。C1/C2 相对 C0 的最大负项仍是 xc，分别为 `-0.0192581` 和
`-0.0538902`。因此旧协议配对结果也没有显示蒸馏组整体超过无蒸馏 C0；由于
validation 与 test 完全相同，这些数字不能称独立盲测结果。

联合审计确认 C0/C1/C2 共 120 个 checkpoint、预测和指标均有效，sample
indices 与 `mips_shared5` manifest 一致，每任务五折 test 样本恰好覆盖一次。
C1/C2 分别执行 32,501/39,136 optimizer updates，四卡调度墙钟约
13分4秒/14分59秒。完整逐任务结果及对照见
[`results/mts_c1_c2_historical_shared5_rerun/comparison/final_report.md`](results/mts_c1_c2_historical_shared5_rerun/comparison/final_report.md)，
正式产物见 [`C1`](results/mts_c1_c2_historical_shared5_rerun/c1/formal) 和
[`C2`](results/mts_c1_c2_historical_shared5_rerun/c2/formal)。

## C0/C1 使用 5k 学生预训练模型的重新微调

前一轮 C1/C2 的旧协议重跑使用的是 20k 部署包；本节是本次明确授权的 **5k
预训练模型** 实验，产物与20k运行完全隔离。C0/C1分别从
`student_deploy_005k.pt` 初始化，在 `historical_shared5` 上完成八任务五折，
下游只使用 O8＋MD200，不加载 GLT。

|组别|预训练 step|version|Macro8 R²|C1−C0|正增量任务|正增量fold|
|-:|-:|-|-:|-:|-:|-:|
|C0|5,000|none|0.8360926133|--|--|--|
|C1|5,000|n_plus_1|0.8359388559|-0.0001537575|5/8|21/40|

逐任务五折 mean ± std（std 使用 `ddof=0`）及 MAE/RMSE 见
[`results/mts_c0_c1_historical_shared5_005k_rerun/comparison/final_report.md`](results/mts_c0_c1_historical_shared5_005k_rerun/comparison/final_report.md)。
C1 在 eat、egb、egc、ei、eps 上有小幅正均值，但 nc/xc 下降，整体 Macro8 略低于
C0，不能称为整体提升。

完整性审计确认 C0/C1 共 80/80 个单元、80/80 个 checkpoint、80/80 个预测有效，
均严格对应 5,000-step bundle，split identity 与 `data/splits/mips_shared5`
一致，每个任务的五个 test folds 恰好覆盖全部样本一次。正式产物见
[`C0`](results/mts_c0_c1_historical_shared5_005k_rerun/c0/formal) 和
[`C1`](results/mts_c0_c1_historical_shared5_005k_rerun/c1/formal)，日志见
[`logs/mts_c0_c1_historical_shared5_005k_rerun`](logs/mts_c0_c1_historical_shared5_005k_rerun)。

旧 `historical_shared5` 中 validation 与 test 相同，因此本节结果是共享开发折
评估，不是独立盲测；与此前 20k 初始化的结果属于不同 checkpoint 条件，不能混合
为同一实验组。

## GLT-V2 revision-2：无 MD＋MIPS 损失基线（New-C0，5k）

本轮新增独立实验 `glt_v2_r2_o8_nomd_mipsloss_005k`。与旧 New-C0 的整体方案不同之处
是：不实例化、不读取、不扰动 MD200；预训练只计算 30% canonical atom 的单路 masked-atom
CE（138-D 整行清零，global sum/count）；下游统一 `target_transform=standard` 与
`regression_loss=mse`。O8 仍为 6-layer Pre-LN、GELU、source-Q/target-K、SPD/path bias，
canonical atom mean pooling，predictor 为 `512→512→1`、dropout 0.1。新包仅含 O8，未读取
教师、GLT line sidecar 或坐标。

### 预训练

正式轨迹在 `Uni-Poly` 的独立 3-GPU window 中完成 `5,000/5,000` updates（PI1M_v2 全量、
local batch 84、accumulation 4、global batch 1008、BF16、AdamW `2e-4`、seed 42）。warmup
为 2,000 updates，衰减使用原 20k 曲线前缀，末步学习率为 `1.66666833e-4`。部署包
[`student_deploy_005k.pt`](glt_v2_r2_o8_nomd_mipsloss_005k/student/student_deploy_005k.pt)
的 schema 为 `mts-glt-v2-r2-o8-nomd-student-deploy-v1`，`use_md200=false`，80 个 state
tensors 且无 MD key；训练记录 5,000 行，loss `4.565→0.208`，峰值显存约
`1.372 GiB/card`。

### 新 no-MD 下游（`outer5_inner20`，40/40）

|Task|R² mean ± std|MAE mean ± std|RMSE mean ± std|
|-|-:|-:|-:|
|eat|0.969 ± 0.013|0.044 ± 0.008|0.062 ± 0.012|
|eea|0.901 ± 0.021|0.246 ± 0.008|0.332 ± 0.012|
|egb|0.914 ± 0.019|0.419 ± 0.037|0.565 ± 0.043|
|egc|0.890 ± 0.005|0.356 ± 0.018|0.518 ± 0.019|
|ei|0.758 ± 0.065|0.335 ± 0.013|0.477 ± 0.051|
|eps|0.746 ± 0.067|0.373 ± 0.031|0.549 ± 0.060|
|nc|0.816 ± 0.065|0.066 ± 0.009|0.099 ± 0.016|
|xc|0.300 ± 0.069|15.275 ± 0.955|19.688 ± 1.075|

`Macro8 R² = 0.787`。每 task 的五个 outer-test fold 均产生一次预测；40 个
checkpoint、CSV 和预测全部有限，split identity 与 `data/splits/mips_outer5_inner20`
一致。旧 New-C0 的 5k/10k/20k Macro8 参考值为 `0.782/0.779/0.774`，
但旧训练包含 MD200 且使用历史下游损失/标签协议，因此这里只作描述性比较，不能解释为
单独去掉 MD 的因果增益。

### 冻结 Ridge 探针（80/80）

对旧 New-C0 5k/10k/20k 和新 no-MD 5k，任务 `ei、xc、eps、nc` 各执行五折 Ridge
`alpha={0.1,1,10,100}`（float64、`solver=svd`）。特征和标签 scaler 只拟合 train，
alpha 只由 validation R² 选择，不做 train+validation refit。新 no-MD 的 probe R² 均值为：
`ei 0.761`、`xc 0.286`、`eps 0.722`、`nc 0.806`；完整 80 行及旧三档
对照见 [`comparison/final_report.md`](glt_v2_r2_o8_nomd_mipsloss_005k/comparison/final_report.md)。
探针仅反映冻结读出的线性可读性，不等于完整任务信息或微调因果证明。

### 审计、日志和边界

- 相关测试为 `21 passed, 1 warning`；三卡两步 no-MD smoke 的有限 CE、梯度和 O8-only 导出通过。
- 首轮预训练因错误将 5k 曲线压缩而在约 1,460 步停止，产物保留于
  `results/glt_v2_r2_o8_nomd_mipsloss_005k/failed_pretrain_schedule_student_001460/`；修复后重跑才计入上述结果。
- 下游参数解析和探针入口各有一次即时失败，均未启动对应正式单元；修复后的调度最终为
  `40/40` 和 `80/80`，失败日志均保留。
- 预训练日志：`logs/glt_v2_r2_o8_nomd_mipsloss_005k/{pretrain_005k.log,tmux_pretrain_retry01.log}`；
  下游承载日志：`tmux_finetune_retry01.log`；探针承载日志：`tmux_probes_retry01.log`。
- 本轮未启动新 10k/20k、其他 seed、3D/教师路线或额外 sweep；旧缓存、旧 New-C0 和旧 C0/C1/C2 产物未覆盖。

## GLT-V2 双路正式结果与几何失稳诊断（2026-09-16）

本轮只读核验与固定小批量诊断；未重跑下游 fold、未追加正式预训练、未重建缓存、未覆盖历史产物。

### 双路正式结果（既有运行，本轮仅校验式重聚合）

静态复用缓存管线（cohort 959,588）上的 Concat/KFuse 各 5000 update，产物
`results/glt_dual_static_pretrain_5k_{concat,kfuse}/`（resume/deploy 1000–5000，`step 5000`、
`EXIT_CODE=0`）；8 任务 × 5 折网格 `results/glt_dual_static_finetune_formal_grid/`（80/80 `exit_code=0`）。
校验式重聚合（`scripts/aggregate_glt_dual_finetune.py`，输出
`<mode>/comparison_review_20260916T001828Z/`）精确复现验收值：宏观平均 test R²
Concat `0.7877379364475444`、KFuse `0.7695629052761048`（pooled-OOF 口径分别为
`0.7931866117573143`、`0.7758626655507175`，两者是不同估计量）。逐任务 test R² 均值（concat）：
eat 0.9806、eea 0.9169、egb 0.8966、egc 0.8970、ei 0.7672、eps 0.7487、nc 0.8154、xc 0.2795；
kfuse 对应为 0.9777、0.9001、0.8927、0.8817、0.7504、0.7210、0.8073、0.2256。XC 显著偏低，
需在后续按残差/预测方差/标签分布单独检查，不用已看过的 test 反复选参数。

### Concat 几何失稳：B.3 有限回放（800/800 update，逐位复现）

从 `resume_02000.pt` 恢复、四卡／microbatch 84／global batch 1008／BF16，`--diagnostics
--stop-after-step 2800`，输出 `results/glt_v2_diag_b3_concat_replay_20260916`，日志
`logs/glt_v2_diag_b3_concat_replay_20260916/replay.log`。回放与正式运行在 800 个共同 step 上
chem/geo/FP 三项 `max|replay-ref| = 0`，`66.9968@2676` 的峰值精确重现。

窗口按 step 定义、同 step 取首次出现的全局值（各 rank 打印相同全局量）：基线 `2401–2600` 中位
`0.00092`；首个 `geo>0.1` 在 `2662`；峰值 `66.9968@2676`；`4001–5000` 中位 `0.3139`（约 342×），
末值 `0.3108`，**截至 5000 未恢复**。chem 由 `0.1854` 到 `0.1617`（无持久退化）；FP 在 `2676`
瞬态升至 `0.1987`（约 4.6×）后回到 `0.0418`。KFuse 同类事件为瞬态：首个 `geo>0.1` 在 `3274`、
峰值 `6.0736@3406`、末值 `0.00106`，完全恢复。

机制证据（dense 窗口 `2600–2720` 的 rank0 局部统计）：angle head pre-tanh 均值由 `-0.51` 漂移到
`+1.01`（2666）、`-4.07`（2668），tanh 导数由 `0.78` 塌到 `0.0012`，`2680` 起 exact ±1 比例达
`1.0000`、导数恒 `0.0000`；同期 3D 表示 `graph_3d_rms` 由 `0.82` 单调增到 `9.17`，`length_head`
梯度由 `0.037` 升到 `0.96–1.00`。尖峰以 length 为主（`2676` 分项 sum：length `5441.5`、angle
`176.4`，length 占 96.9%），angle 饱和后留下约 `26.7`（基线 0.21）的常数残差。距离 Gaussian
`σ_min` 全程恒为 `0.0172`，排除 Gaussian 宽度分支；BF16 已在固定批量诊断中排除（FP32≈BF16）。
`_module_grad_norms` 将 `p.grad is None` 记为 `0.0`，故 angle head 梯度 `0.0000` 不区分
“梯度为零”与“未进入反向图”，属当前口径限制。

上述为观测与支持性证据：表示尺度增长先于饱和，但 `~2655–2660` 的初始触发事件、以及 LayerNorm
能否阻断该链条均未验证。首要改动建议（本轮未实施）为仅在几何头输入加 LayerNorm
（`length_head`/`angle_head` 的 `nn.Sequential` 首部），并以同源 800-update 回放作最小验证预算。

### 局部验证

`tests/test_glt_dual_diagnostics.py` 新增 7 项（开关不改变 loss/梯度/state_dict/RNG、分项还原、
`component_tensors` 梯度、缺失梯度按零对齐、真实 Gaussian hook、angle 统计有限性）；
与 `tests/test_dual_glt_pretrain.py`、`tests/test_aggregate_glt_dual_finetune.py` 合计
**27 passed**。诊断入口：`scripts/pretrain_glt_dual.py --diagnostics --stop-after-step`
（诊断模式只写 resume/诊断状态、不生成 deploy）、`scripts/diagnose_glt_dual_pretrain.py`
（六 checkpoint 固定批量前向，无 optimizer update）。

## GLT-V2 固定 Concat（geometry_head_norm）5k：训练健康、deploy 与 7 任务正式微调

本轮唯一改动：在进入 geometry heads 的 3D bond state 上加 `nn.LayerNorm(512, elementwise_affine=False)`
（零新增参数），config `configs/mts/glt_dual_three_task_concat_geonorm.json`。训练身份与旧正式
Concat 5k 一致：同 cohort `30f17b59…`、同 bundle、同 dual_static、microbatch 84、global batch 1008、
lr 2e-4、warmup 2000、schedule total 20000、BF16、loss 权重 [1,1,0.1]、noise 0.03、mask 0.3、
5000 optimizer steps。差异：world_size 4→3（accumulation 3→4），global batch 均为 1008，因此每个
optimizer step 覆盖的 absolute-position 样本集合保持同一连续区间；world_size 4→3 改变的是
rank/microbatch partition、dropout RNG 与样本的对应以及数值 reduction 路径，因此该完整训练仍不是
bitwise matched single-variable run。本轮对照不是样本级单变量证明；样本级单变量证据仍是 P2 replay。

### 训练健康（`results/glt_v2_fixed_concat_5k_20260916/pretrain_health_report.json`）

`FORMAL_FIXED_5K_HEALTHY = YES`（7 项阻断检查，阈值与实测参考量均写入报告）。step 5000 三项目标：

| step 5000 | chem | geometry | fingerprint |
|-|-|-|-|
|旧 Concat（失稳）|0.1553|**0.310754**|0.0400|
|KFuse|0.1538|0.001063|0.0376|
|固定 Concat（geonorm）|0.1552|**0.000518**|0.0341|

旧 Concat 的 geometry 自 step 3000 起再未恢复；固定路线收敛至 5.18e-4。其余稳态量：length 逐图
稳态最大 0.01103（初始瞬态仅 step 1–16，峰值 2.617；失稳路线峰值 64.78@2676）、angle exact ±1
饱和率恒 0.0（失稳路线末期 1.0）、tanh 导数最小 0.0031（失稳路线 0.0）、angle 头梯度 last-500
最小 0.00206、`graph_3d_rms` 最大 1.972（末值 1.439；失稳路线最大 9.17）、clip 前梯度 last-500
最大 0.617（失稳路线峰值 1063.1、末期仍 16–37）。残留关注（非阻断）：原始 `bond_states_rms`
末值 2.068，高于失稳前健康带 1.07–1.22，低于失稳带 5.77–9.17。

### deploy（`deploy_05000_validation.json`）

`FIXED_CONCAT_DEPLOY_VALID = YES`：187 张量／39,021,942 参数、仅 O8+GLT+Concat fusion、无
chemistry/geometry/fingerprint head、无 geometry_norm 依赖、与 `resume_05000.pt` 的 encoder 子集
逐张量 bitwise 相同、strict load 与真实样本前向通过、冻结缓存零写入。eat/fold0 smoke（2 epochs）
best validation R² 0.7737，`outer_test = NOT_RUN`。

### 7 任务正式微调（egc 按用户指示跳过）

输出 `results/glt_v2_fixed_concat_5k_20260916/finetune_grid_fixed_concat`，聚合
`aggregation_review_7task/summary.json`（status PASS、task_count 7、fold_count 35、
predictions_recomputed、OOF 恰好覆盖一次）。**FIXED_CONCAT_MACRO7_R2 = 0.777210**，
pooled OOF macro7 = 0.782525。

| task | fixed test R² | best fold R² | gap | 旧 Concat | KFuse | delta vs 旧 Concat |
|-|-|-|-|-|-|-|
|eat|0.987535|0.992401|−0.004865|0.980600|0.977742|+0.006935|
|eea|0.902478|0.947349|−0.044872|0.916868|0.900062|−0.014391|
|egb|0.909553|0.928526|−0.018974|0.896611|0.892708|+0.012941|
|ei|0.780538|0.845439|−0.064901|0.767227|0.750365|+0.013312|
|eps|0.735881|0.788369|−0.052488|0.748720|0.721013|−0.012839|
|nc|0.815819|0.886824|−0.071005|0.815419|0.807303|+0.000400|
|xc|0.308668|0.400945|−0.092277|0.279492|0.225616|+0.029176|
|**macro7**|**0.777210**|||0.772134|0.753544|**+0.005077**|

口径说明：通用评测只有 5 个 outer test fold，属 development folds，不是独立盲测；macro 提升幅度小
且 2/7 任务退化，**不构成“已超过 baseline”的结论**。egc 未参与，故本轮 7 任务 macro **不可**与
8 任务 macro8（旧 Concat `0.7877379364`、KFuse `0.7695629053`）比较；表中旧 Concat/KFuse 数值已
按同样 7 任务重算。聚合脚本对非 8 任务刻意不输出 macro 字段，该 macro 由已验证的逐任务均值派生。
egc 新增 fold0/1（0.8927/0.8854）与旧 Concat 同 fold（0.8979/0.9010）仅供参考，2/5 fold 不足以判定。
XC 在三条路线中均为最低（0.279/0.226/0.309），与既有记录一致，仍需单独诊断。

## GLT-GALPH-PHRETENTION-20260920-01｜PH retention 三组开发比较（2026-09-20，限定负结果）

**科学问题**：同一个修复版 C1 主干（`C1_REPAIR_5K`，`deploy_05000.pt`，sha256 `3063cf8bf…`）下，下游**保留样本特异 PH**（F_REAL）是否优于**关闭 PH**（F_OFF）与**同容量固定 PH 分支**（F_CONST，P_train 平均 profile）。三组共享同一部署包、公共与新投影/门控初始张量、数据顺序与公共随机流；结构 `r3_new = r3 + tanh(gamma)·W_ph(冻结 PH encoder(profile))`，`gamma` 从 0 起步、不复用预训练 `alpha_ph`、无开门正则、PH encoder 冻结且 eval、无效 PH 为零残差且不删样本；**F_OFF 为本轮重训控制**，未用旧 C1 指标或任何 smoke 结果替代。

**协议**：xc/eps/eat × fold0/1（`outer5_inner20`）、seed 42、FULL adaptation、≤30 epochs、patience 10、train-only scaler、validation 选优、无 refit、`outer_test=NOT_RUN`、`validation_only=true`。18/18 单元完成，实际 **411/540 epochs**，失败 0 次。汇总 `results/glt_galph_ph_retention_20260920/p4/development_aggregate.json`。

| 任务 | F_OFF | F_CONST | F_REAL |
|-|-|-|-|
|xc（fold0／fold1）|0.327180 / 0.385436|0.327186 / 0.385435|0.327186 / 0.385435|
|eps（fold0／fold1）|0.773621 / 0.876573|0.773622 / 0.876596|0.773620 / 0.876598|
|eat（fold0／fold1）|0.991276 / 0.989242|0.991276 / 0.989242|0.991276 / 0.989242|
|**三任务均值**|**0.7238879309**|**0.7238928537**|**0.7238929676**|

差值（validation R²，越高越好）：**F_REAL − F_OFF = 5.036685268300367e-6**、**F_REAL − F_CONST = 1.1396120269679955e-7**、F_CONST − F_OFF = 4.92272406571459e-6。预登记工程阈值为三任务均值相对两个控制均 ≥ **+0.005**（XC 另有 ≥ +0.01 且两折不退化的条件），**实际差低约三个数量级、未达到**；无任务均值相对任一控制退化超过 0.01（`risk_flags` 为空）。汇总判定 `VERDICT = STOP: F_REAL does not reach the three-task gain threshold against both controls`；`EPS_ONLY_CANDIDATE=false`；无单元 `best_epoch` 触及 30 上限。

**接线有效性（排除"三组其实相同"）**：三组输入源分别为 `zero_placeholder` / `p_train_mean_fixed`（跨样本 spread 恰为 0）/ `sample_own_frozen`（spread 0.71–0.96），encoder summary 与 const 相差 0.29–0.35；`best.pt` 中约 203–206/243 个张量逐位不同（`ph_proj.weight` 相差 0.015–0.042）；γ 在 F_CONST/F_REAL 被训练到非零（`|tanh γ|` 1e-5–3.8e-4）而 F_OFF 恒为 0。成本：三组 544/550/555 s，合计 27.5 min；覆盖率完整（xc 345、eps 305、eat 312 个 train+validation 样本，invalid/missing 均为 0）。

**解释边界（必须与数值同时引用）**

* 这是**两个开发折上的 validation-R² 比较**：**不是 outer-test**、**不是 OOF**、**不是独立盲测**，也**不是显著性检验**；两折单 seed 的均值只用于对照预登记工程阈值。
* **不作统计等价声明**：差异幅度小（~1e-5）且无重复 seed，**不能**据此断言三组"统计上等价"；只能说本项目按预登记阈值**未检出** F_REAL 的增量。
* **不与旧 C1 作受控性能归因**：本轮主干为新初始化的修复配方 C1_REPAIR_5K，与旧 C1（`b7093898…`，PH 路径已退化）在主初始化与训练路径上均不同，两者差值不是受控比较。
* **不宣布 PH 或 3D 整体无效**：本轮实际比较的是**几乎关闭的 PH 残差**（实测 `|tanh γ| ≤ 3.8e-4`，残差约占参考 1.2e-4）。这一门控限制与结果同时成立：结论仅覆盖"按当前配方训练后门控极小"的情形，未检验更强条件化、非零门控初始化或解冻 encoder 的配置（均未运行、未获授权）。
* eat 上三组几乎完全相同（0.991276 / 0.989242），该任务对本轮 PH 条件化不敏感，可能接近该数据上的可分上限；不据此推断其他任务或其他 3D 表示。
* 与 F_OFF 的一致性检查：F_OFF 的 PH 残差精确为 0（显式分支），其属性路径与基础 CLS+读出路径一致，已由逐位 parity 测试与 0-update 接线诊断分别验证。

**上游背景（同周期）**：旧 C1 的 PH encoder 在下游失去可观测样本区分性（step 1000 仍可分辨、step 2000 低于容差、step ≥3000 逐位一致），机制为耦合 weight decay 压过被门控抑制的任务梯度（r2）；修复配方（PH 路径 `weight_decay=0` + `tanh(alpha)` 初值 0.02）在 5k 内不再退化，但门控自身收敛到 ≈2e-4（r4）。该背景不构成本轮下游结果的因果解释。

**产物与提交**：`results/glt_galph_ph_retention_20260920/p4/{smoke_r6,development,development_aggregate.json,smoke_r6_verification.json,diagnostics_path_check.json}`；日志 `logs/glt_galph_ph_retention_20260920/{p3,p4}/`；提交链 `886623a`/`9c0bc7e`（r6），周期全链见 `PROJECT_HISTORY.md`。**本周期以该限定负结果结束，不再追加实验。**

## GLT-3D-GAIN-20260921-01｜D2 四臂开发比较（2026-09-21，限定负结果）

同一个 S5 `B_FP` step5000 部署包（`results/glt_pred_20260918/s3b_formal/b_fp/pretrain/deploy_05000.pt`）抽取的四臂，在 `mips_outer5_inner20` 上做单 seed、三任务两折的 30-epoch 开发比较（`configs/mts/glt_pred_s3b_b_fp.json`；seed 42、patience 10、batch 32、eval 64、wd 0.02、train-only scaler、validation 选最佳 checkpoint）。四臂唯一差别是学习率：`F2D` 只跑 O8（head 输入 `[norm2(z2), 0]`）、`FBASE` 双路原配置（o8/glt 1e-5、norm2/norm3 1e-4、head 1e-4）、`FNORM` 把 norm 降到 1e-5、`FSTABLE` 再把 GLT 降到 3e-6。

| arm | xc | eps | eat | macro3 |
| --- | --- | --- | --- | --- |
| F2D | 0.41646 | 0.80583 | 0.98107 | **0.73445** |
| FBASE | 0.38152 | 0.82123 | 0.98701 | 0.72992 |
| FNORM | 0.38139 | 0.82122 | 0.98751 | 0.73004 |
| FSTABLE | 0.38990 | 0.82256 | 0.98664 | 0.73303 |

配对差值（逐任务两折均值 / macro3）：`FBASE − F2D` 为 xc **−0.03495**、eps +0.01541、eat +0.00594，macro3 **−0.00453**；`FNORM − FBASE` 为 xc −0.00013、eps −0.00001、eat +0.00050，macro3 **+0.00012**；`FSTABLE − FNORM` 为 xc **+0.00852**（两折 +0.00525／+0.01179）、eps +0.00134、eat −0.00087，macro3 **+0.00299**。

预登记门槛（候选相对 FBASE **和** F2D 的 macro3 均 ≥ +0.005；XC 均值 ≥ +0.01 且两折为正；任一任务两折均值不降超 0.01）：**FNORM 与 FSTABLE 均未达到**——相对 FBASE 的 macro3 分别为 **+0.00012** 与 **+0.00311**，相对 F2D 分别为 **−0.00441** 与 **−0.00142**；FSTABLE 的 XC 均值 **+0.00838** 也低于 +0.01。聚合器 selection 为空；`fstable/xc/fold1` 的 best_epoch 触及 30 上限（按要求未延长训练）。预算：24/24 单元 PASS、0 失败 0 重试、**668/720 epochs**、**5544 optimizer updates**、网格墙钟 833 s。

**解释边界（必须与数值同时引用）**

* 这是**单 seed、三任务两折的开发比较**：**不是 outer-test**、**不是 OOF／refit**、**不是独立盲测**，也**不是显著性检验**；阈值只是工程分支条件，小数位级别的差值不能读作稳定收益。
* **`FBASE − F2D` 为负不等于“3D 普遍无效”**：本轮只覆盖当前**单冻结 Trimer**、heavy-only token、无显式立构字段、30-epoch 开发预算的协议；xc 的差距集中在单折（fold0 −0.06650，fold1 −0.00339），且 eps／eat 上双路反而更高（+0.01541／+0.00594），属任务间收益抵消。
* **FNORM 在本轮等价于无效改动**：与 FBASE 的差异落在第 4–5 位小数（最大单折 +0.00099），不能据此判定 norm 分组机制不存在。
* **FSTABLE 是唯一有动向的杠杆但仍不足**：XC 均值 +0.00838 未达 +0.01，macro3 +0.00311 未达 +0.005，且与 F2D 相比 XC 仍为 −0.02656；eat 出现 −0.00038 的小幅回退。
* 公共随机流的末态摘要因各臂早停长度不同而不同，**不作跨臂相等要求**；`best.pt` 只是 validation 最佳权重，**不是完整训练 resume 状态**。

**产物与提交**：`results/glt_3d_gain_20260921/d2_development/{schedule.json,grid_runtime.json,aggregate.json,report.md,<arm>/<task>/fold<k>/}`；日志 `logs/glt_3d_gain_20260921/d2_development*/`；tmux window `glt3d_r6_dev`。提交链 `95e3185`（r1 D0/D1）→ `563668f`（r2）→ `eae7b81`（r3）→ `4d9f302`（r4）→ `751a35b`（r5）→ `6aba976`／`1981466`（r6 与记录补记），全部推送至 `origin/dev`。**本周期以该限定负结果结束，不追加 sweep、D3、PH 或额外 seed。**
