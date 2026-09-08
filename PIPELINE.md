# Uni-Poly-Plus 当前基线流程

本文描述正式生产基线 `MTS-GLT-v2-Base-5k`、保留实验路线 `Atomic-PC W-CAMR-v2`，以及已实现但尚未执行正式训练的候选路线 `MTS-GLT-v3-Galformer-20k`。结果索引见 [`RESULTS.md`](RESULTS.md)。

## 1. 基线身份

```text
MTS-GLT-v2-Base-5k
├─ MIPS O8 topology branch: 6 layers / hidden 512 / 8 heads / SPD≤2
├─ periodic line GLT-v2: 6 layers / hidden 512 / 8 heads
├─ strict real-bond 1-hop line neighborhood
├─ line-to-canonical-atom incidence projection
├─ 512-channel gated additive atom residual (initial α=0.05)
├─ canonical atom mean pooling
└─ MD200 residual + graph adapter + regression head
```

唯一正式预训练 checkpoint：

```text
results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth
```

轨迹运行至 20,000 optimizer steps；下游固定使用 step 5,000 checkpoint。

## 2. 数据与缓存

```text
P-SMILES CSV
  -> ordered sample/cohort manifest
  ├─ RU-base LMDB
  ├─ canonical lifted topology LMDB
  ├─ open Trimer coordinate LMDB
  ├─ periodic_line_glt_v1 sidecar
  └─ MD200 feature cache
        ↓
MIPS O8 + periodic GLT-v2
        ↓
line-to-atom incidence + atom residual
        ↓
canonical atom mean + MD200 + graph adapter + regression head
```

上述缓存均为只读输入。GLT-v2 在运行时从保留的 line sidecar 计算 observation moments、identity self relations 和 canonical atom incidence，不重写缓存。缓存的 `.done`、manifest、LMDB 元数据和固定 split 必须存在且匹配；发现不匹配时停止启动。

## 3. O8 输入

O8 只接收 MIPS137 atom features、backbone embedding、SPD bias 和 single-path-node bias；`max_hops=2`，6 层、512 hidden、8 heads。基线关闭 Star-RBF、MCL 和 O8 的 3D attention bias，因此 O8 是纯拓扑分支。

canonical lifted topology 只维护一个 RU 的 canonical atom state。跨 RU 关系由 relation shift、path 和 SPD 字段表达，不创建 RU−1/RU+1 的独立可训练节点。

## 4. GLT-v2 输入与对齐

GLT token 对应 RU 内部真实化学键或跨 RU 的真实聚合连接键；非真实 line relation 不进入 line neighborhood。Bond type 仅作为 masked-line label/QC，不进入 token embedding。

每个 distance observation 先经过端点原子类型条件化的 learned Gaussian basis，再计算 mean 与 population variance，并加入 observation-count 和 absolute-shift embedding。共享原子的真实 1-hop line relation使用 angle mean/variance、count 和 relation multiplicity 形成 per-head bias；每个 token 追加动态 identity self relation与独立 self bias。

attention 使用 target-Q/source-K 和 `1/sqrt(64)`。最终 line state 经 incidence projection 与 absolute-shift incidence embedding scatter 到两个 canonical endpoints，再按 incident line 求 atom mean，并进行 atom output normalization。

## 5. 3D→2D 融合与 readout

对 canonical atom `i`，基线融合为：

$$
h_i = h^{\mathrm{O8}}_i + v_i\,\tanh(g)\odot W\!\left(\operatorname{LN}(h^{\mathrm{GLT}}_i)\right),
$$

其中 `g` 为 512 通道 gate、初始 `tanh(g)=0.05`，`v_i` 是 GLT geometry-valid mask。无效 geometry 的 atom/graph 直接保留 O8 pathway；随后执行 canonical atom mean、MD200 residual、graph output adapter 和 regression head。

Compact19、额外模态和其他融合路径不属于基线运行路径。

## 6. 预训练合同

```text
dataset                 PI1M_v2
seed                    42
masked atom             ratio 0.30, weight 1.0
masked line             ratio 0.40, weight 1.0
bidirectional InfoNCE   temperature 0.10, weight 1.0
global batch            1008
precision               BF16
learning rate           2e-4
optimizer warmup        2000 optimizer steps
trajectory              20000 optimizer steps
selected checkpoint     step 5000
```

入口配置为 [`configs/mts/glt_v2_formal_a6_h_w1_20k.json`](configs/mts/glt_v2_formal_a6_h_w1_20k.json)，短验证使用 [`configs/mts/glt_v2_ddp_smoke.json`](configs/mts/glt_v2_ddp_smoke.json)。

## 7. 下游合同

```text
route                   graph-only MTS
schedule                direct_joint
LR warmup               5 epochs
epochs / patience       100 / 10
O8 / GLT / fusion / MD  1e-5
adapter                 1e-5
regression head         1e-4
loss                    Huber (β=0.5)
gradient clip           1.0
precision               FP32
train / eval batch      32 / 64
workers                 2
protocol                historical_shared5
tasks                   eat, eea, egb, egc, ei, eps, nc, xc
folds                   0, 1, 2, 3, 4
seed                    42
```

`historical_shared5` 使用同一 held-out fold 作为 validation 和 test，不是独立盲测。后续代码验证默认只运行必要的 unit test 和 smoke；扩大任务、fold、epoch 或样本范围须另行授权。

## 8. 保留入口与产物

- 预训练：`scripts/pretrain.py` → `src.training.pretrain`。
- 下游单 fold：`scripts/train.py` → `src.training.finetune.engine`。
- 调度：`scripts/run_mts_finetune_scheduler.py`。
- line sidecar：`scripts/build_periodic_line_glt_sidecar.py`、`scripts/audit_periodic_line_glt_sidecar.py`。
- cache 合同：`scripts/resolve_mips_trimer_scage.py`、`scripts/audit_mips_trimer_cache.py`、`scripts/validate_mts_cache.py`。

正式结果、resolved input、训练日志和 checkpoint 的索引集中在 [`RESULTS.md`](RESULTS.md)。所有新产物必须使用独立目录，不覆盖基线文件。

## 9. 保留实验路线：Atomic-PC W-CAMR-v2

`Atomic-PC W-CAMR-v2` 是唯一保留的非生产实验路线。其下游结构为当前 O8、Original-MIPS MD200/KFuse 和 Center-RU Atomic-PC；完整 Trimer 参加 `kNN=24` 的四层消息传递，最终只池化 `ru_offset == 0` 的中心 RU 原子。

W-CAMR 预训练只更新 Atomic-PC encoder、learned mask embedding 和临时 atom head。它在固定 50K cohort 的 48,101 个合格样本上执行 1,504 optimizer updates，对中心 RU 重原子做 15% weighted masking；权重只控制 mask 抽样，loss 是普通 masked-position mean CE。下游只迁移 `atomic_point_encoder`，不加载临时 mask/head。

入口为 `scripts/run_original_mips_atomic_pc_w_camr_v2.py`，结果位于 `results/original_mips_atomic_pc_w_camr_v2/`。Center-RU 数据构造、50K cohort 解析和历史 matched-reference 读取代码集中在 `src/training/w_camr_v2_support/`；参考产物集中在该结果目录的 `references/`。它们仅作为 W-CAMR-v2 的内部实现与 provenance 依赖保留，不是独立路线。该实验同样使用 `historical_shared5`，不是独立盲测，也不替代正式生产基线。

## 10. 已实现候选：MTS-GLT-v3-Galformer-20k

v3 使用独立的 `mts-periodic-line-glt-image-v1` sidecar。每个周期 line token 只读取一个中心锚点 image 的真实键长；每个 source-image→center-target 关系只读取对应物理实例的一个键角，不计算 Trimer 平移副本的 distance/angle mean、variance、count 或 multiplicity。token 化学输入为端点元素、BondType、BondStereo、IsConjugated，几何输入为原子对条件化的 256 维 Gaussian distance basis；关系角度通过 128 维 Gaussian basis形成 8-head bias。line 邻域仍严格是共享真实原子的 chemical-bond 1-hop。

联合预训练固定为 `L_mask2D + L_mask3D + L_cl`。O8 的 30% canonical atom masking 在 pool 前经过可学习 MD200 scalar-gated node residual；GLT 对 40% line token 使用 80/10/10 corruption与四个 factorized heads；双向 InfoNCE 使用 O8 canonical-atom mean 与 GLT atom/line readout。正式配置为 [`configs/mts/glt_v3_galformer_20k.json`](configs/mts/glt_v3_galformer_20k.json)，只保存 5k/10k/20k probes且没有 resume 路径。

下游公开 `glt_readout_mode=galformer|mips_concat`：默认 `galformer` 只加载 O8 与 MD residual，完全不实例化或读取 GLT/geometry；`mips_concat` 将 atom-aligned O8/GLT states拼成 1024 维并投影回 512 维后再执行 MD residual。sidecar 构建入口为 `scripts/build_mts_glt_v3_sidecars.py`，短 smoke 为 `scripts/smoke_mts_glt_v3.py`，下游调度入口为 `scripts/run_mts_glt_v3_finetune.py`。

当前状态仅为实现、单元测试和两步 smoke；未构建全量 v3 sidecar，未启动 20k 预训练或正式微调，因此 v3 不是新的生产基线，也没有性能结论。
