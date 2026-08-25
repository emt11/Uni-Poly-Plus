# Uni-Poly-Plus 结果索引

本文件只索引当前仓库中实际存在的正式报告与诊断结果。逐 fold、配置和产物以链接文件为准；历史 shard 和 prediction 不迁移。

## 当前正式基线

```text
MTS-GLT-v2-Base-5k
version     mts_glt_v2_base_5k_v1
checkpoint results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth
schedule   Warm0 (encoder freeze warm epochs 0; LR warmup epochs 5)
```

| Task | O8-only R2 | O8+GLT R2 | Delta |
|---|---:|---:|---:|
| eat | 0.984069 | 0.984101 | +0.000033 |
| eea | 0.925549 | 0.922432 | -0.003118 |
| egb | 0.939633 | 0.941382 | +0.001749 |
| egc | 0.915657 | 0.919324 | +0.003667 |
| ei | 0.827700 | 0.828284 | +0.000584 |
| eps | 0.815260 | 0.818812 | +0.003552 |
| nc | 0.868243 | 0.871385 | +0.003142 |
| xc | 0.449817 | 0.463366 | +0.013549 |

```text
O8-only macro8                0.8407410869
O8+GLT macro8                 0.8436358322
descriptive fused increment  +0.0028947453
positive tasks                7/8
```

来源：[`final_report.md`](results/mts_glt_v2/final_report.md) / [`json`](results/mts_glt_v2/final_report.json) / [`paired_summary.json`](results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json)。基线快照见 [`config`](configs/mts/glt_v2_base_5k_v1.json)、[`manifest`](results/mts_glt_v2/base_5k_v1/baseline_manifest.json) 与 [`baseline doc`](docs/MTS_GLT_V2_BASELINE.md)。

## 正式模型与历史候选

| 路线 | O8-only macro8 | 融合 macro8 | Delta | 状态 | 报告 |
|---|---:|---:|---:|---|---|
| MTS-GLT-v2-Base-5k Warm0 | 0.840741 | 0.843636 | +0.002895 | **current formal baseline** | [`report`](results/mts_glt_v2/final_report.md) |
| MTS-GLT-v2-Base-5k Warm5 | 0.840741 | 0.841362 | +0.000621 | tested; rejected as default (`Warm5-Warm0=-0.002274`) | [`summary`](results/mts_glt_v2/fusionwarm_vs_nowarm_formal5k/eight_task_fold01234/glt_v2_warm5_vs_warm0_summary.json) |
| MTS-GLT-v1 | 0.837227 | 0.835722 | -0.001505 | historical; not promoted | [`report`](results/mts_glt_v1/final_report.md) |
| GraphGate-v1 20k legacy fusion | 0.834300 | 0.835656 | +0.001355 | historical/mechanism branch | [`report`](results/mts_glt_graphgate_v1/formal_20k_report.md) |
| GraphGate-v1 FusionWarm 20k | 0.834300 | 0.837601 | +0.003300 | historical mechanism result | [`report`](results/mts_glt_graphgate_v1/fusion_realization_v1/fusion_realization_report.md) |
| B0-v2 | — | 0.837442 | — | historical reference | [`PIPELINE`](PIPELINE.md) |

## 机制与诊断索引

| 实验 | 核心结果 | 用途 | 报告 |
|---|---|---|---|
| GLT-v2 alignment audit | graph-level alignment 与 element-level 错配审计 | GLT-v2 解释，不改变正式结果 | [`per_element_metrics.csv`](results/mts_glt_v2/alignment_audit_v1/per_element_metrics.csv) |
| GraphGate Warm5 vs Warm0 | Warm5 `0.837601` vs Warm0 `0.833362` | GraphGate schedule mechanism | [`summary`](results/mts_glt_graphgate_v1/fusionwarm_vs_nowarm_20k/eight_task_fold01234/fusionwarm_vs_nowarm_20k_8task_fold01234_summary.json) |
| Matched 5k Full/Geometry-Off × FusionWarm | interaction macro8 `+0.016212`, 7/8 positive | matched geometry mechanism | [`summary`](results/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/fusionwarm_causal_v1/eight_task_fold01234/summary/matched_fusionwarm_causal_8task_fold01234_summary.json) |
| Trimer 3D validation | frozen sensitivity、matched 5k 与误差关联 | Trimer geometry diagnostic | [`report`](results/mts_glt_graphgate_v1/trimer_validation_v1/trimer_validation_report.md) |
| 5k-to-20k representation trajectory | P1-P4、Query 与表示漂移 | trajectory diagnostic | [`report`](results/mts_glt_graphgate_v1/trajectory_selection_v1/trajectory_selection_report.md) |

## 评价口径与边界

- 正式结果覆盖 `eat eea egb egc ei eps nc xc × 5 folds × seed 42`。
- 当前正式协议为 `historical_shared5`；validation 与 test 使用同一 fold，不是独立盲测。
- `+0.002895` 是相同 GLT-v2 checkpoint 下 fused 与 O8-only 的描述性增量，不称为 geometry causal interaction。
- Screening、frozen probe、matched mechanism experiment 和正式 8×5 不混用。
- `results/best_result.csv` 保留为跨模型逐任务最优包络，不是当前 baseline 的任务表，因此本次不改写。
