# Uni-Poly-Plus 结果索引

本文件只索引当前仓库中实际存在的正式报告与诊断结果。逐 fold、配置和产物以链接文件为准；历史 shard 和 prediction 不迁移。

## 当前正式基线

```text
MTS-GLT-v2-Base-5k
version     mts_glt_v2_base_5k_v1
checkpoint results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth
schedule   Warm0 (encoder freeze warm epochs 0; LR warmup epochs 5)
```

| Task | Formal Fusion R² |
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
formal macro8 R²              0.8436358322
positive GLT-vs-O8 historical tasks  7/8
```

来源：[`final_report.md`](results/mts_glt_v2/final_report.md) / [`json`](results/mts_glt_v2/final_report.json) / [`paired_summary.json`](results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json)。基线快照见 [`config`](configs/mts/glt_v2_base_5k_v1.json)、[`manifest`](results/mts_glt_v2/base_5k_v1/baseline_manifest.json) 与 [`baseline doc`](docs/MTS_GLT_V2_BASELINE.md)。

### 历史消融参考

O8-only 仅保留为历史消融 comparator，不属于当前生产路线：

- O8-only macro8 R²：`0.8407410869`
- Fusion macro8 R²：`0.8436358322`
- descriptive Fusion − O8 increment：`+0.0028947453`
- positive tasks：`7/8`

该差值仅用于描述 GLT 融合相对 O8-only 的历史增量，不称为 causal interaction。

## 正式模型与历史候选

| 路线 | Fusion macro8 | 状态 | 报告 |
|---|---:|---|---|
| MTS-GLT-v2-Base-5k Warm0 | **0.843636** | **current formal baseline** | [`report`](results/mts_glt_v2/final_report.md) |
| MTS-GLT-v2-Base-5k Warm5 | 0.841362 | rejected (`Warm5-Warm0=-0.002274`) | [`summary`](results/mts_glt_v2/fusionwarm_vs_nowarm_formal5k/eight_task_fold01234/glt_v2_warm5_vs_warm0_summary.json) |
| MTS-GLT-v1 | 0.835722 | historical; not promoted | [`report`](results/mts_glt_v1/final_report.md) |
| GraphGate-v1 20k legacy fusion | 0.835656 | historical mechanism branch | [`report`](results/mts_glt_graphgate_v1/formal_20k_report.md) |
| GraphGate-v1 FusionWarm 20k | 0.837601 | historical mechanism result | [`report`](results/mts_glt_graphgate_v1/fusion_realization_v1/fusion_realization_report.md) |
| B0-v2 | 0.837442 | historical reference | [`PIPELINE`](PIPELINE.md) |

## 机制与诊断索引

| 实验 | 核心结果 | 用途 | 报告 |
|---|---|---|---|
| GLT-v2 alignment audit | graph-level alignment 与 element-level 错配审计 | GLT-v2 解释，不改变正式结果 | [`per_element_metrics.csv`](results/mts_glt_v2/alignment_audit_v1/per_element_metrics.csv) |
| GraphGate Warm5 vs Warm0 | Warm5 `0.837601` vs Warm0 `0.833362` | GraphGate schedule mechanism | [`summary`](results/mts_glt_graphgate_v1/fusionwarm_vs_nowarm_20k/eight_task_fold01234/fusionwarm_vs_nowarm_20k_8task_fold01234_summary.json) |
| Matched 5k Full/Geometry-Off × FusionWarm | interaction macro8 `+0.016212`, 7/8 positive | matched geometry mechanism | [`summary`](results/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/fusionwarm_causal_v1/eight_task_fold01234/summary/matched_fusionwarm_causal_8task_fold01234_summary.json) |
| Trimer 3D validation | frozen sensitivity、matched 5k 与误差关联 | Trimer geometry diagnostic | [`report`](results/mts_glt_graphgate_v1/trimer_validation_v1/trimer_validation_report.md) |
| 5k-to-20k representation trajectory | P1-P4、Query 与表示漂移 | trajectory diagnostic | [`report`](results/mts_glt_graphgate_v1/trajectory_selection_v1/trajectory_selection_report.md) |

## Validated Screening / STOP Experiments

以下项目均已完成对应的 `historical_shared5` screening 或历史诊断合同；它们不是 independent blind test。保留实现与产物用于 provenance，不重复启动相同实现。

| Experiment | Primary mechanism contrast | Total contrast | Decision | Protocol | Notes |
|---|---:|---:|---|---|---|
| X23 post-GLT conditioning | cross `+0.000057` | `-0.003388` | STOP | 3 tasks × folds 0–2 | update norm ratio仅约`0.000186` |
| X2L line conditioning | cross `+0.001983` | `-0.002917` | mechanism weak-positive / implementation STOP | 3 tasks × folds 0–2 | 3/3 tasks cross为正，但capacity为`-0.004900` |
| X2A attention routing | cross `+0.000342` | `-0.002833` | STOP | 3 tasks × folds 0–2 | cross仅1/3 tasks为正 |
| Reflection-invariant torsion | geometry `+0.000185` | `-0.006736` | STOP | 3 tasks × folds 0–2 | count control为`-0.006921` |
| Explicit O8 bond type | chemistry `-0.000727` | `-0.006782` | STOP | 3 tasks × folds 0–2 | 不支持显式bond-type bias |
| Joint radial-angular | radial context `+0.000055` | `-0.006150` | STOP | 3 tasks × folds 0–2 | additive SBF未建立增量 |
| Compact19 | frozen concat `+0.004318` | neural `-0.004923` | STOP | frozen 8×5 + downstream | 不接入正式模型 |
| GLT-v2 Warm5 vs Warm0 | — | Warm5-Warm0 `-0.002274` | STOP | formal 8 tasks × 5 folds | 保留Warm0 |

## 评价口径与边界

- 正式结果覆盖 `eat eea egb egc ei eps nc xc × 5 folds × seed 42`。
- 当前正式协议为 `historical_shared5`；validation 与 test 使用同一 fold，不是独立盲测。
- 当前生产、筛选和后续模型优化默认仅针对 Fusion 路线；O8-only 仅作为已有历史消融或明确设计的 matched control 使用，不作为独立生产候选重复训练或优化。
- O8-only 仅为历史消融 comparator；`+0.002895` 是同一 GLT-v2 checkpoint 下 Fusion − O8-only 的描述性增量，不代表独立生产路线，也不称为 geometry causal interaction。
- Screening、frozen probe、matched mechanism experiment 和正式 8×5 不混用。
- `results/best_result.csv` 保留为跨模型逐任务最优包络，不是当前 baseline 的任务表，因此本次不改写。

## 后续 Metadata-Dedup 比较口径

Metadata-Dedup 仅比较 Fusion 路线：

```text
FULL Fusion
vs
DEDUP Fusion
```

O8 主干不改变，不重复计算 FULL/DEDUP 的 O8-only 路线。主比较定义为：

$$
\Delta_{\mathrm{dedup}}
=
R^2_{\mathrm{DEDUP\ Fusion}}
-
R^2_{\mathrm{FULL\ Fusion}}
$$

该项是后续实验口径，不代表该实验已经启动或完成。
