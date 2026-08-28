# Uni-Poly-Plus 基线结果索引

本文件只索引当前保留的 `MTS-GLT-v2-Base-5k` 及其必要证据。完整合同见 [`PIPELINE.md`](PIPELINE.md)。

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
