# Uni-Poly-Plus 当前流程

本文描述当前正式数据流和模型身份。结果索引见 [`RESULTS.md`](RESULTS.md)，完整基线合同见 [`docs/MTS_GLT_V2_BASELINE.md`](docs/MTS_GLT_V2_BASELINE.md)。

## 1. 当前正式基线

```text
MTS-GLT-v2-Base-5k
├─ pure-topology MIPS O8，6 layers / hidden 512 / 8 heads / SPD<=2
├─ Periodic Line GLT-v2，6 layers / hidden 512 / 8 heads / real-bond 1-hop
├─ line-to-canonical-atom incidence
├─ atom-level 512-channel gated residual
├─ canonical atom mean pooling
├─ MD200 residual + graph adapter + regression head
└─ downstream Warm0
```

正式 checkpoint：

```text
results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth
```

预训练轨迹完成到 20k，但下游正式选择 step 5k。当前 baseline 不是 20k checkpoint。

## 2. 数据流

```text
P-SMILES CSV
  -> normalized sample key / ordered cohort
  ├─ RU base LMDB
  ├─ canonical lifted topology LMDB
  ├─ open Trimer coordinates LMDB
  ├─ periodic_line_glt_v1 sidecar
  └─ MD200 mmap
        ↓
pure-topology O8 + periodic GLT-v2
        ↓
line-to-atom incidence + atom-level fusion
        ↓
canonical atom mean + MD200 + graph adapter + regression head
```

Topology、Trimer、periodic line sidecar 和 MD200 按现有冻结产物只读使用。GLT-v2 在运行时从旧 line sidecar 派生 observation moments、identity self relations 和 atom incidence，不重建 Trimer cache。

## 3. Canonical topology 与 O8

模型只维护一个 RU 的 canonical atom states。跨 RU 关系由 lifted relation 的 shift/path/SPD 字段表达，不建立 RU-1/RU+1 的独立可训练节点。

O8 输入为 MIPS137 atom features、backbone embedding、SPD bias 和 single-path-node bias，固定 `max_hops=2`。当前 baseline 中 Star-RBF、MCL 和 O8 3D attention bias 全部关闭，因此 O8 是纯 2D topology branch。

## 4. Periodic GLT-v2

GLT token 只对应 RU 内部真实化学键或跨 RU 真实聚合连接键；SPD2、Star 和 virtual relation 不进入 line graph。Bond type 只用于 Masked Line label/QC，不进入 token input。

距离 observation 逐个经过端点原子类型条件化的 learned Gaussian basis，再汇总编码 mean 与 population variance，并加入 observation-count 和 absolute-shift embeddings。共享原子的真实 1-hop line relation使用 angle encoded mean/variance、count 和 multiplicity 形成 per-head bias。动态 identity self relation使用独立 self bias。

6 层 target-Q/source-K attention 使用 `1/sqrt(64)`。最终 line state 经 incidence projection 和 absolute-shift incidence embedding scatter 到两个 canonical endpoints，再按 atom 求 incident-line mean 和 output norm。

## 5. Atom-level fusion 与 readout

```text
h_i = h_i_O8
      + valid_i * tanh(channel_gate[512]) * W(LN(h_i_3D))
```

channel gate 初始 alpha 为 0.05；geometry-invalid atom/graph 精确退回 O8。融合后执行 canonical atom mean pooling、MD200 residual、graph output adapter 和 regression head。Compact19 和 GraphGate learned Query 均关闭。

## 6. 预训练与下游

预训练：

```text
PI1M_v2 / seed 42
Masked Atom 0.30 + Masked Line 0.40 + bidirectional InfoNCE(T=0.10)
loss weights 1 / 1 / 1
global batch 1008 / BF16 / LR 2e-4 / warmup 2000 optimizer steps
trajectory 20k / selected checkpoint 5k
```

下游 Warm0：

```text
encoder-frozen warm epochs  0
LR warmup epochs            5
epochs / patience           100 / 10
O8, GLT, atom fusion, MD200, adapter LR  1e-5
head LR                     1e-4
Huber beta / clip / dropout 0.5 / 1.0 / 0.25
FP32 / train batch 32 / eval batch 64 / workers 2
```

Warm0 表示 epoch 1 起所有正式可训练模块联合优化；不表示取消 5-epoch LR warmup。

## 7. 正式结果与产物

正式 seed-42、`historical_shared5` 8×5：

```text
O8-only macro8      0.840741
O8+GLT macro8       0.843636
descriptive delta  +0.002895
positive tasks      7/8
```

评价共享 validation/test fold，不是独立盲测。正式入口：

```text
configs/mts/glt_v2_base_5k_v1.json
results/mts_glt_v2/base_5k_v1/baseline_manifest.json
results/mts_glt_v2/final_report.json
results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json
```

## 8. 路线演化与历史状态

```text
MTS / O8
  -> MTS-GLT-v1
  -> GraphGate-v1
  -> MTS-GLT-v2
  -> MTS-GLT-v2-Base-5k (current formal baseline)
```

- MTS-GLT-v1：historical candidate，图级融合未晋级。
- GraphGate-v1：historical/mechanism-validation branch；其 learned Query 不属于当前 baseline。
- GraphGate Warm5、GM/AT readout：historical diagnostics。
- GLT-v2 Warm5：完整 8×5 为 `0.841362`，相对 Warm0 `-0.002274`，已测试并拒绝作为默认。
- Compact19：未准入。
- B0-v2：历史参考，不再是 current baseline。

早期报告中的 `geometry_encoded_but_not_realized` 结论保留在历史报告中，但已被后续 matched geometry experiments supersede；不据此改变当前 baseline。

## 9. 新实验规则

新的 MTS-GLT 主实验默认 parent 为 `MTS-GLT-v2-Base-5k`，默认采用 `Base + exactly one primary scientific change`。除非明确设计 factorial，不同时改变 checkpoint step、Warm schedule、fusion、geometry 和 pretraining objective。

历史 checkpoint、报告、sidecar 和配置继续保留。新实验必须使用显式配置和独立产物路径，不能覆盖当前 baseline。
