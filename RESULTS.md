# Uni-Poly-Plus 基线结果索引

本文件索引当前正式基线 `MTS-GLT-v2-Base-5k`、保留实验路线 `Atomic-PC W-CAMR-v2`、旧 N+1/N+2 两阶段蒸馏实验，以及使用独立 validation 的 GLT revision-2 C0/C1/C2 正式对照。完整合同见 [`PIPELINE.md`](PIPELINE.md)。

## 当前正式基线

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

- 基线配置：[`configs/mts/glt_v2_base_5k_v1.json`](configs/mts/glt_v2_base_5k_v1.json)。
- 预训练配置：[`configs/mts/glt_v2_formal_a6_h_w1_20k.json`](configs/mts/glt_v2_formal_a6_h_w1_20k.json)。
- DDP smoke 配置：[`configs/mts/glt_v2_ddp_smoke.json`](configs/mts/glt_v2_ddp_smoke.json)。
- 基线 manifest：[`results/mts_glt_v2/base_5k_v1/baseline_manifest.json`](results/mts_glt_v2/base_5k_v1/baseline_manifest.json)。
- 预训练 resolved input：[`results/mts_glt_v2/formal/a6_h_w1_20k/resolved_input.json`](results/mts_glt_v2/formal/a6_h_w1_20k/resolved_input.json)。
- 成对下游汇总：[`results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json`](results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json)。
- 最终机器可读报告：[`results/mts_glt_v2/final_report.json`](results/mts_glt_v2/final_report.json)。
- 最终 Markdown 报告：[`results/mts_glt_v2/final_report.md`](results/mts_glt_v2/final_report.md)。

## 运行边界

- 当前生产模型固定为 graph-only MTS-GLT-v2；O8-only 只作为已存在的 matched comparator 读取。
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

证据为 `results/original_mips_atomic_pc_w_camr_v2/summary.json` 和 `results/original_mips_atomic_pc_w_camr_v2/downstream/aggregate_summary.json`。该路线仍是实验结果，不改变 `MTS-GLT-v2-Base-5k` 的正式生产基线身份。

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
