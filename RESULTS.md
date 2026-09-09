# Uni-Poly-Plus 基线结果索引

本文件索引当前正式基线 `MTS-GLT-v2-Base-5k`、保留实验路线 `Atomic-PC W-CAMR-v2`，以及已完成的 N+1/N+2 两阶段蒸馏实验。完整合同见 [`PIPELINE.md`](PIPELINE.md)。

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
