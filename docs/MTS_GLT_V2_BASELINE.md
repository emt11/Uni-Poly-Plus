# MTS-GLT-v2-Base-5k 正式基线

## 1. 身份与来源

```text
name     = MTS-GLT-v2-Base-5k
version  = mts_glt_v2_base_5k_v1
status   = current_formal_baseline
```

正式预训练权重为：

```text
results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth
```

预训练轨迹运行到 20,000 optimizer steps，并在 5k、10k、20k 生成 probe。正式 trajectory screening 选择 5k，随后该 checkpoint 完成匹配 8 tasks × 5 folds 正式评价。因此正确表述是：

> GLT-v2 trajectory trained to 20k; formal downstream-selected checkpoint is step 5k.

它不是“20k checkpoint”。选择依据见 [`trajectory_selection.json`](../results/mts_glt_v2/formal/trajectory_selection.json)，正式结果见 [`paired_summary.json`](../results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json)。

## 2. O8：纯 2D 拓扑分支

O8 固定为 6 层、hidden 512、8 heads、`max_hops=2` 的 MIPS local graph encoder。输入为 MIPS137 原子特征、backbone embedding、SPD bias 和 single-path-node bias。

正式基线关闭 Star-RBF、MCL 和 O8 的任何 3D attention bias。Trimer 距离与角度不会进入 O8；O8 是纯 2D topology branch。

## 3. GLT：周期真实键 3D 分支

GLT 固定为 6 层、hidden 512、8 heads、head dimension 64，采用 target-Q/source-K 与 `1/sqrt(64)` 缩放。line token 只对应真实化学键，包括 RU 内部键和跨 RU 聚合连接键；SPD2、Star relation 和 virtual relation 均不是 line token。更新邻域只包含共享真实原子的 1-hop line relations。

Bond type 不进入 GLT token input，仅用于 Masked Line label 和 QC。

每个周期距离 observation 先独立经过 256 维、端点原子类型条件化的 learned Gaussian basis，再计算编码向量的 mean 与 population variance。token 同时保留 observation-count embedding 和 absolute-shift embedding。禁止先平均标量距离再编码。

共享原子的真实 1-hop line relation 使用 128 维 learned Gaussian basis。各角度 observation 编码后计算 mean 与 population variance，并结合 observation count 和 relation multiplicity 投影为 per-head attention bias。动态 identity self relation 使用独立 per-head self bias，不伪造角度。

## 4. Line 到 canonical atom

最后一层 line states 经过 incidence projection，并加入 absolute-shift incidence embedding，然后分别 scatter 到化学键的两个 canonical endpoints。每个 canonical atom 对入射 line message 求均值并执行 atom output norm，得到 512 维 3D atom geometry state。

当前基线保留 absolute shift，不引入 signed shift，也不修改 incidence semantics。

## 5. Atom-level fusion 与 graph readout

融合发生在 canonical atom pooling 之前：

```text
h_i_fused = h_i_O8
            + valid_i
            * tanh(channel_gate[512])
            * W_f(LN(h_i_3D))
```

channel gate 初始 alpha 为 `0.05`。geometry-invalid atom 的 residual 精确为零；geometry-invalid graph 精确退回 O8 pathway。

融合后的 canonical atom states 做均值池化，随后依次经过 MD200 residual、graph output adapter 和 regression head。MD200 开启，Compact19 关闭。GraphGate learned Query 不属于此基线。

## 6. 预训练合同

预训练数据为完整 PI1M_v2，seed 42，global batch 1008，BF16，learning rate `2e-4`，前 2,000 optimizer steps warmup。三个目标权重均为 1：

```text
Masked Atom   ratio 0.30
Masked Line   ratio 0.40
Bidirectional O8-GLT InfoNCE, temperature 0.10
```

正式 checkpoint step 为 5,000；完整轨迹结束于 step 20,000。

## 7. Warm0 下游合同

Warm0 表示从 downstream epoch 1 开始，全部正式可训练的 O8、GLT、atom fusion、MD200、graph adapter 和 regression head 立即联合训练。它只表示没有 encoder-frozen warm stage，并不表示没有 learning-rate warmup。

```text
encoder freeze warm epochs  0
LR warmup epochs            5
total epochs                100
early stopping patience     10
O8 / GLT / fusion / MD200 / adapter LR  1e-5
regression head LR          1e-4
Huber beta                  0.5
gradient clip               1.0
head dropout                0.25
precision                   FP32
train/eval batch            32 / 64
workers                     2
seed                        42
```

## 8. 正式结果

在 `historical_shared5`、seed 42、8 tasks × 5 folds 下：

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

该增量是描述性 fused increment，不是 geometry causal interaction。评价共享 validation/test fold，不是独立盲测。

## 9. Warm5 与历史 GraphGate

同一 5k checkpoint 的 GLT-v2 Warm5 完整 8×5 为 `0.8413621651`，相对 Warm0 为 `-0.0022736672`，正式决定为 `FAIL`。因此 Warm5 不是默认。

MTS-GLT-v1、GraphGate-v1、GraphGate Warm5、GraphGate GM/AT readout、GLT-v2 Warm5 和 Compact19 均保留为 historical/diagnostic evidence。GraphGate 使用图级 Query readout，其结构不属于当前 atom-aligned GLT-v2 基线。早期报告中的 `geometry_encoded_but_not_realized` 判断已被后续 matched geometry experiments supersede，但历史报告不改写、不删除。

## 10. 新实验默认规则

新的 MTS-GLT 主实验默认 parent 为 `MTS-GLT-v2-Base-5k`，并以：

```text
Base + exactly one primary scientific change
```

为默认设计。torsion、signed shift、non-bonded contacts、InfoNCE、fusion、incidence 或 geometry encoding 等变化每次只指定一个 primary scientific variable。不得无说明地同时改变 checkpoint step、Warm schedule、fusion、geometry 和 pretraining objective；明确设计的 factorial 实验除外。

机器可读快照见 [`config`](../configs/mts/glt_v2_base_5k_v1.json) 与 [`manifest`](../results/mts_glt_v2/base_5k_v1/baseline_manifest.json)。
