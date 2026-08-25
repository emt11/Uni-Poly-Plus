# Uni-Poly-Plus 当前流程

本文只描述当前 B0-v2 基线及已完成的 MTS-GLT-v1 候选实验。历史 T/G/R/A、MSTA、Topo-MCL、显式 k-RU、full-Trimer MCL 与 Angle-20 路线均已退出活动代码。

## 1. 当前基线

```text
B0-v2
├─ canonical lifted 单 RU 拓扑
├─ O8 Graph Transformer，6 层，hidden 512，8 heads，max hop 2
├─ Star-RBF v2 relation attention bias
├─ MD200 图级残差（下游）
├─ masked atom + periodic coordinate denoising（预训练）
└─ MSTA OFF / MCL OFF
```

活动配置：

- `configs/mts/b0_v2.json`：正式预训练轨迹。
- `configs/mts/b0_v2_probe.json`：短筛选。
- `configs/mts/b0_v2_ddp_smoke.json`：三卡 smoke。
- `configs/mts/b0_v2_resume_probe.json`：同一三卡配置的恢复检查。

## 2. 数据流

```text
P-SMILES CSV
  ↓ normalize_polymer_smiles
sample key / ordered cohort
  ├─ RU base LMDB
  ├─ canonical lifted topology LMDB
  ├─ open Trimer coordinates LMDB
  ├─ Star-RBF v2 relation sidecar
  └─ MD200 mmap
  ↓ mips_trimer_collate
B0-v2 model
```

冻结 cache、Trimer 和 Star-RBF v2 sidecar 只读使用；当前训练不读取旧 Angle 或 MCL-threshold sidecar。

## 3. Canonical lifted 单 RU

模型只维护一个 RU 的 canonical 原子节点。跨 RU 关系不复制节点，而由 relation 字段表示：

```text
lga_edge_index
lga_spd
lga_source_image_shift
lga_path_index / lga_path_shift / lga_path_mask
polymer_link_mask
```

关系定义为：

```text
target atom a@RU0 receives from source atom b@RU(shift)
```

因此 `shift != 0` 的原子不是独立可训练节点；它们与 canonical 原子共享状态。多层 Transformer 可沿 lifted relations 传播跨 RU 信息，但不会建立或更新独立的 RU-1/RU+1 节点副本。

## 4. Star-RBF v2

Star-RBF v2 覆盖 canonical lifted `SPD <= 2` relations，并以 inverse-shared periodic pair 为单位编码：

- `shift=0`：中央 RU 内真实距离；
- `|shift|=1`：左右两个真实 observation 分别做 RBF 后平均；
- `|shift|=2`：Trimer 外侧 RU 之间的真实距离；
- true self、invalid geometry 与合同外 relation：零 bias。

每个 unique pair 的 RBF 经 `32 -> 8` 投影，再 gather 回 directed relations，与 SPD/path bias 相加，供六层 O8 attention 复用。当前 B0-v2 的 `star_rbf_upper=3.75`。

## 5. 预训练

入口：

```bash
scripts/run_mips_trimer_scage.sh configs/mts/b0_v2.json
```

固定科学路线：

```text
PI1M_v2
GPU 1,2,3 / 3-rank DDP
global batch 1008
BF16
20,000 optimizer steps
Masked Atom + periodic coordinate denoising
O8 + Star-RBF v2
MSTA OFF / MCL OFF
```

Coordinate denoising 只扰动 Trimer 坐标，并从 noisy observations 动态重算 Star-RBF v2。坐标 decoder 使用 relation endpoint 向量恢复中央 RU 原子位移。

训练中 `.last.pt` 是滚动 resume 状态；轨迹 probe 位于 5k/10k/20k。正式下游模型由选择后的 probe 通过 `publish_mts_b0_v2_final.py` 发布。

## 6. 微调

入口：

```bash
python scripts/run_mts_b0_v2_finetune.py --help
```

当前统一下游身份：

```text
O8
Star-RBF v2 ON
MSTA OFF
MCL OFF
MD200 ON
FP32
```

任务为 `eat eea egb egc ei eps nc xc`，每任务 5 folds。调度器只负责任务展开、GPU slot 和失败清理；单 fold 训练由 finetune engine 执行。

## 7. 产物职责

```text
*.last.pt                 训练中断恢复
B0-v2 trajectory probes  下游选择候选
final.pth                 已发布的正式下游权重
final.pth.complete.json   阶段完成信号
```

历史实验结果、checkpoint、cache、sidecar 和日志不属于本次代码清理范围，除非用户另行明确授权删除。

## 8. GLT 数据边界

MTS-GLT-v1 直接复用：

- canonical lifted periodic relations；
- normalized Trimer atom mapping 与坐标；
- relation shift/path/SPD；
- Star-RBF v2 periodic observation 语义。

GLT 的 bond state、距离与角度更新、inverse sharing 和预训练目标由独立候选模块实现，不在 B0-v2 中预留历史 MCL/MSTA 分支。

## 9. MTS-GLT-v1 候选路线

MTS-GLT-v1 作为独立候选实现，不替换或覆盖 B0-v2：

```text
canonical lifted Topology ─→ pure-topology O8 ─┐
                                               ├─→ bidirectional InfoNCE
open Trimer + true bonds ─→ local 1-hop GLT ──┘
```

- O8 只使用 SPD 与 single-path-node bias，Star-RBF、MSTA、MCL 均关闭。
- GLT token 只表示真实周期化学键；距离进入 line token，直接共享原子的键对角度进入 attention bias。
- 每个物理 observation 先独立做 128 维 Gaussian basis 编码，再在周期等价 observation 间平均。
- bond type 不进入 GLT 输入，只作为 Masked Line Node 标签和离线 QC 字段。
- 预训练目标为 Masked Atom、Masked Line Node 和 O8–GLT 双向 InfoNCE。
- DDP InfoNCE 先 gather 固定 shape embedding 与 valid mask，再统一过滤；negative pool 是当前 global microbatch，不等同于 accumulation 后的 optimizer batch。

活动实现与入口：

```text
src/dataset/periodic_line_glt.py
src/modules/periodic_line_glt.py
src/modules/mts_glt.py
configs/mts/glt_v1.json
scripts/build_periodic_line_glt_sidecar.py
scripts/run_mips_trimer_scage.sh
scripts/run_mts_glt_finetune.py
```

下游使用同一预训练 checkpoint 比较 `o8_only` 与 `o8_glt`。融合门零初始化，geometry-invalid 样本在 `o8_glt` 模式精确退回 O8；MD200 仍只在下游 readout 使用。

正式 seed-42、historical_shared5 结果：

```text
O8-only macro8 = 0.837227
O8+GLT  macro8 = 0.835722
delta           = -0.001505
positive tasks  = 2/8
```

因此 MTS-GLT-v1 已完成为候选实验，但当前融合模式不晋级、不替换 B0-v2。逐任务结果和限定见 `results/mts_glt_v1/final_report.md`；该协议共享 validation/test fold，非独立盲测。
