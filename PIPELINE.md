# Uni-Poly-Plus 当前基线流程

本文描述当前保留的两条路线：`GLT-V2 revision-2`（2D O8 学生及其 N+1/N+2 蒸馏与 C0/C1/C2 对照）和 `Atomic-PC W-CAMR-v2`；已退役的 `MTS-GLT-v2-Base-5k` 与 `MTS-GLT-v3-Galformer-20k` 只保留历史记录和下列明确列出的依赖产物。结果索引见 [`RESULTS.md`](RESULTS.md)。

**路线范围（2026-09-10 清理后）**

保留：

- `GLT-V2 revision-2`：`src/modules/{mips_local_graph,mts_glt_distill,periodic_line_glt_v3,uni_encoder}.py`、`src/training/finetune/`、`src/training/pretrain/glt_distill_engine.py`、`src/training/c0_transfer.py`，配置 `configs/mts/glt_distill_{n_plus_1,n_plus_2,repair_c0,repair_c1,repair_c2}.json`。
- `Atomic-PC W-CAMR-v2`：`src/modules/{original_mips_atomic_pc,original_mips_knowledge_fusion,original_mips_md200,atomic_point_encoder}.py`、`src/training/w_camr_v2_support/`、`src/training/pretrain/original_mips_atomic_pc_joint.py`，配置 `configs/atomic_point_*.json`。

已删除：`MTS-GLT-v2-Base-5k` 与 `MTS-GLT-v3-Galformer-20k` 的模型、预训练、下游、配置、测试、`results/`、`logs/` 与 `pretrained_models/` 产物。

保留的跨路线依赖（不得删除）：

- `results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth`：`src/training/w_camr_v2_support/cohort_runtime.py` 与 `center_runtime.py` 运行时读取，缺失即报错。
- `results/mts_glt_v2/downstream/formal/a6_h_w1_20k_probe_005k/paired_summary.json`：`scripts/report_mts_glt_distill.py` 读取的历史对照。
- `src/modules/periodic_line_glt_v3.py`、`scripts/build_mts_glt_v3_sidecars.py`、`data/processed/mips_trimer_scage/periodic_line_glt_image_v1`：`mts_glt_distill.py`、`scripts/build_mts_glt_distill_repair_sidecars.py` 与 `tests/test_mts_glt_distill.py` 的依赖，文件名带 v3 但服务保留路线。
- `data/processed/mips_trimer_scage/periodic_line_glt_v1`：`tests/test_mts_glt_distill.py` 的对齐 fixture。

## 1. 已退役基线：MTS-GLT-v2-Base-5k

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

该路线的训练与下游代码已于 2026-09-10 退役删除（`src/modules/mts_glt_v2.py`、
`src/modules/periodic_line_glt_v2.py`、`src/training/pretrain/glt_v2_engine.py`、
`glt_v2_objectives.py`、`configs/mts/glt_v2_*.json`、`scripts/run_mts_glt_v2_finetune.py`、
`scripts/report_mts_glt_v2_*.py`、`scripts/build_glt_v2_label_counts.py`、
`tests/test_mts_finetune_v2.py`、`tests/test_mts_periodic_line_glt.py`，以及其
`results/`、`logs/`、`pretrained_models/` 产物）。仅保留上面那个 checkpoint，因为
保留路线 Atomic-PC W-CAMR-v2 在运行时读取它。因此本节描述的架构与下游合同自此
只是历史记录：该基线已无法重新预训练、重新微调或重新评估。

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

数据缓存继续保存 `mips_x=[N,137]` 与 `mips_backbone_mask=[N]`，其中前者的
137 维化学属性语义不变。模型输入时先将 backbone 标记转为一列并拼接为
`[N,138]`，再通过单个 `Linear(138,512)`；不再实例化独立的
`Embedding(2,512)`。因此 O8 接收的是 138 维原子输入、SPD bias 和
single-path-node bias；`max_hops=2`，6 层、512 hidden、8 heads。基线关闭
Star-RBF、MCL 和 O8 的 3D attention bias，因此 O8 是纯拓扑分支。

原子 masking 的顺序固定为“拼接 backbone 列 → 对选中原子的整行 138 维
全部置零 → 线性投影”。未选中原子保留全部 137 个属性和 backbone 标记，
选中原子的初始 token 只等于线性层 bias；path bias 也从这份 masking 后的
初始 token 计算。输入缓存张量不被原地修改，mask 比例和 canonical mask
策略保持不变。

该输入布局变更会使旧的 `Linear(137,512)` 与独立 backbone 参数无法严格
加载到新模型；本路线不提供旧 checkpoint 转换或放宽 strict-loading 的路径。

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

该合同的入口配置 `configs/mts/glt_v2_formal_a6_h_w1_20k.json` 与 `configs/mts/glt_v2_ddp_smoke.json` 已随基线路线一并删除；本节作为历史记录保留。

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

`adapter 1e-5` 一行描述已退役的 v2 基线；保留路线的学习率、dropout 与 head 结构见 §11–§14（新版 DistillStudent 无 adapter 参数组，predictor 使用 `1e-4`，`head_dropout` 固定 `0.1`）。

`historical_shared5` 使用同一 held-out fold 作为 validation 和 test，不是独立盲测。后续代码验证默认只运行必要的 unit test 和 smoke；扩大任务、fold、epoch 或样本范围须另行授权。

## 8. 保留入口与产物

- GLT-V2 revision-2 预训练：`scripts/pretrain_mts_glt_distill.py --stage teacher|student` → `src.training.pretrain.glt_distill_engine.run_stage`；配置 `configs/mts/glt_distill_repair_c{0,1,2}.json`。
- GLT-V2 revision-2 分词蒸馏侧车：`scripts/build_mts_glt_distill_repair_sidecars_parallel.py`。
- 下游单 fold：`scripts/train.py` → `src.training.finetune.engine`；调度：`scripts/run_mts_finetune_scheduler.py`。
- Atomic-PC W-CAMR-v2：`scripts/run_original_mips_atomic_pc_w_camr_v2.py`。
- line sidecar：`scripts/build_periodic_line_glt_sidecar.py`、`scripts/audit_periodic_line_glt_sidecar.py`、`scripts/build_mts_glt_v3_sidecars.py`（同时服务保留路线的 image sidecar）。
- cache 合同：`scripts/audit_mips_trimer_cache.py`、`scripts/validate_mts_cache.py`。
- 已随 v2 基线删除：`scripts/run.sh`、`scripts/run_mts.sh`、`scripts/run_mips_trimer_scage.sh`、`scripts/run_train.sh`、`scripts/run_pretrain.sh`、`scripts/resolve_mips_trimer_scage.py`、`scripts/resolve_mts.py`（它们只解析 `schema=mts-glt-v2` 配置，已无有效输入）。

`scripts/pretrain.py` → `run_pretrain` 只保留入口壳：已退役的 `mts-glt-v2` 与 `mts-glt-v3-galformer-20k` schema 会直接报错，不再分发到任何模型。

正式结果、resolved input、训练日志和 checkpoint 的索引集中在 [`RESULTS.md`](RESULTS.md)。所有新产物必须使用独立目录，不覆盖保留产物。

## 9. 保留实验路线：Atomic-PC W-CAMR-v2

`Atomic-PC W-CAMR-v2` 是唯一保留的非生产实验路线。其下游结构为当前 O8、Original-MIPS MD200/KFuse 和 Center-RU Atomic-PC；完整 Trimer 参加 `kNN=24` 的四层消息传递，最终只池化 `ru_offset == 0` 的中心 RU 原子。

W-CAMR 预训练只更新 Atomic-PC encoder、learned mask embedding 和临时 atom head。它在固定 50K cohort 的 48,101 个合格样本上执行 1,504 optimizer updates，对中心 RU 重原子做 15% weighted masking；权重只控制 mask 抽样，loss 是普通 masked-position mean CE。下游只迁移 `atomic_point_encoder`，不加载临时 mask/head。

入口为 `scripts/run_original_mips_atomic_pc_w_camr_v2.py`，结果位于 `results/original_mips_atomic_pc_w_camr_v2/`。Center-RU 数据构造、50K cohort 解析和历史 matched-reference 读取代码集中在 `src/training/w_camr_v2_support/`；参考产物集中在该结果目录的 `references/`。它们仅作为 W-CAMR-v2 的内部实现与 provenance 依赖保留，不是独立路线。该实验同样使用 `historical_shared5`，不是独立盲测，也不替代正式生产基线。

## 10. 已退役路线：MTS-GLT-v3-Galformer-20k

该路线于 2026-09-10 删除：`src/modules/mts_glt_v3.py`、`src/training/pretrain/glt_v3_{engine,objectives}.py`、`configs/mts/glt_v3_*.json`、`scripts/{check_mts_glt_v3_ddp_parity,run_mts_glt_v3_finetune,run_mts_glt_v3_full_pipeline,smoke_mts_glt_v3}.py`、`tests/test_mts_glt_v3.py`、`results/mts_glt_v3_galformer_20k/`、`logs/mts_glt_v3_galformer_20k/` 与 `pretrained_models/mts_glt_v3_galformer_20k.pth`。

它只到过实现、单元测试和两步 smoke，从未构建全量 v3 sidecar，也没有启动 20k 预训练或正式微调，因此没有性能结论随删除丢失。

仍在仓库中的 v3 命名组件属于保留路线的依赖，不是该路线残留：`src/modules/periodic_line_glt_v3.py`（`mts_glt_distill.py` 导入的 mask/RBF 与 line transformer）、`scripts/build_mts_glt_v3_sidecars.py`（保留路线的 image sidecar 构建入口）与 `data/processed/mips_trimer_scage/periodic_line_glt_image_v1`（化学输入来源）。

## 11. 已完成实验：N+1 / N+2 两阶段蒸馏

该实验新增两个几何语义不同、参数结构一致的 GLT 教师。二者都使用六层严格 chemical-bond 1-hop line graph，token 输入包含无向端点元素、BondType、BondStereo、IsConjugated 与唯一键长，角度只进入 8-head attention bias；最终教师监督和蒸馏只读取中心 N 条内部键。

- `N+2`：中心 N 条内部键加左右两个跨 RU states，分别保留左右真实键长。外围关系按共享原子位于中心 RU 的物理实例重定位，并同时更新 source、target 与角度身份。
- `N+1`：中心 N 条内部键加一个共享跨 RU state；先对左右真实跨键长度取算术平均，再做距离 RBF。左右周期关系保持独立消息次数。

全量 sidecar 由既有只读 `periodic_line_glt_v1` 与 `periodic_line_glt_image_v1` 派生，没有重建或改写 Trimer 坐标：

```text
sample_count  995799
valid_count   959587
N+2 tokens    28556617
N+1 tokens    27597030
relations     73857508（每个版本）
```

教师阶段各训练 5,000 optimizer steps，目标为 40% 中心内部键的 chemistry、length-RBF 与 angle-RBF 重建。学生阶段各训练 20,000 steps；教师全冻结，学生 O8 使用 Pre-LN、source-Q/target-K、incoming softmax。学生保留 30% masked-atom CE，并在 O8 六层后执行原子条件 MD200 sigmoid-gated residual；融合前 O8 通过中心键 local cosine 与 graph-level multi-positive InfoNCE 接收教师监督。三卡 global batch 为 1008（microbatch 84、accumulation 4），BF16，AdamW LR `2e-4`，教师/学生 warmup 分别为 500/2000 steps。

下游严格只迁移各自学生 20k 的 O8＋MD200 部署包，不实例化 GLT，不读取坐标或 line sidecar。八任务五折使用 `historical_shared5`、seed 42、最多 100 epochs、patience 10；80/80 单元均已正常完成。完整入口为 [`scripts/run_mts_glt_distill_pipeline.py`](scripts/run_mts_glt_distill_pipeline.py)，配置为 [`configs/mts/glt_distill_n_plus_2.json`](configs/mts/glt_distill_n_plus_2.json) 与 [`configs/mts/glt_distill_n_plus_1.json`](configs/mts/glt_distill_n_plus_1.json)，正式产物位于 `results/mts_glt_v2_distill/`。

该路线是已完成的实验，不替换生产基线。N+1/N+2 的比较同时改变边界 state 共享和跨 RU 长度处理；与旧 GLT-v2 的比较还混合 O8 更新、预训练任务、MD 融合及下游输入变化，不能把差值单独归因于蒸馏。

## 12. 已完成实验：GLT revision 2 与无蒸馏对照

本轮修复实验使用独立根目录 `results/mts_glt_distill_repair_control/`，不覆盖第 11 节的旧 N+1/N+2 产物，也不重跑生产 GLT-v2。三组共享学生 O8、MD200 模块和 atom head 的 seed-42 初始张量、PI1M_v2 样本顺序、20k 预算及下游协议：

|组别|教师|学生目标|下游初始化|
|-|-|-|-|
|C0|无|两路 masked-atom CE|O8＋MD200 20k|
|C1|revision-2 N+1，5k|masked-atom CE＋冻结教师 local/global distillation|O8＋MD200 20k|
|C2|revision-2 N+2，5k|masked-atom CE＋冻结教师 local/global distillation|O8＋MD200 20k|

revision-2 line graph 直接从现有冻结 Trimer 坐标枚举共享中心 RU 原子的不同物理键，不生成构象，也不从旧 sidecar 推测缺失角度。N+1 把左右跨键映射到一个共享 state，并先对两条真实长度取算术平均；N+2 保留左右两个独立 state 与各自真实长度。关系去重使用物理键身份，不使用 canonical state 身份，因此同原子双连接位点仍保留 N+2 的两个有向跨键关系，或 N+1 中映射后的真实周期自关系及多重性。两个版本都只以中心内部键作为教师重建与蒸馏 target，跨 RU state 只提供上下文。

当前 revision-2 sidecar 合同为：

```text
root          data/processed/mips_trimer_scage/periodic_line_glt_distill_v2/
sample_count  995799
valid_count   959594
N+1 tokens    27597037
N+2 tokens    28556631
relations     73871338（每个版本）
revision      2
source        frozen_trimer_coordinates_direct
```

教师和学生仍使用三卡、microbatch 84、accumulation 4、global batch 1008、BF16、AdamW LR `2e-4`。教师训练 5,000 steps、warmup 500；学生训练 20,000 steps、warmup 2,000。C1/C2 的 local cosine 与 multi-positive InfoNCE 权重均为 `0.1`，在前 2,000 steps 线性 ramp；C0 不加载教师或 line sidecar。MD200 对无效行执行严格零更新，有效行若包含 NaN/Inf 则报错。部署包仅含 O8 与 MD200，不含教师、GLT 或临时 atom/line heads。

下游使用独立协议 `outer5_inner20`：保留既有 `KFold(5, shuffle=True, random_state=1)` 的 outer-test indices；对每折 outer-train 使用 `train_test_split(test_size=0.20, random_state=42+fold)` 生成 validation。标签变换只拟合最终 train，validation 只负责早停和最佳 checkpoint 选择，恢复该折最佳 checkpoint 后才进行一次 outer-test 推理。三个集合的固定 manifest 位于 `data/splits/mips_outer5_inner20/`；该协议与 `historical_shared5` 分开，结果不能互相识别为已完成。

下游三组均只实例化 O8、MD200 与 property head，不读取坐标或 line sidecar。当前新版 DistillStudent 的 graph wrapper 是 Identity，property head 为 `Linear(512,512)→GELU→Dropout(0.1)→Linear(512,1)`（§14）；本节记录的 C0/C1/C2 正式结果由更早的 `LayerNorm→Linear→LayerNorm→ReLU` adapter 加 regression MLP head 产生，不能视为新版 predictor 的复评。固定八任务、五折、seed 42、最多 100 epochs、patience 10；总计 120 个 task/fold。完整阶段入口为 [`scripts/run_mts_glt_distill_repair_pipeline.py`](scripts/run_mts_glt_distill_repair_pipeline.py)，revision-2 构建入口为 [`scripts/build_mts_glt_distill_repair_sidecars_parallel.py`](scripts/build_mts_glt_distill_repair_sidecars_parallel.py)，正式汇总入口为 [`scripts/report_mts_glt_distill_repair.py`](scripts/report_mts_glt_distill_repair.py)。

两份教师、三份学生及 120/120 个下游单元均已完成并通过 checkpoint、预测、split 身份和 OOF 覆盖核验。macro8 R² 为 C0 `0.8061119181`、C1 `0.7978494262`、C2 `0.7930190200`；两条蒸馏路线均未超过 C0。完整指标、成本、异常和产物位置见 [`RESULTS.md`](RESULTS.md) 及 [`results/mts_glt_distill_repair_control/comparison/final_report.md`](results/mts_glt_distill_repair_control/comparison/final_report.md)。

## 13. 已完成诊断：冻结探针与 C0 分阶段微调

本轮不重新预训练，也不修改 GLT、蒸馏目标、MD200、pooling 或 `outer5_inner20`。输入固定为 revision-2 正式部署包：

- C0：`results/mts_glt_distill_repair_control/c0/student/student_deploy_020k.pt`
- C1：`results/mts_glt_distill_repair_control/c1/student/student_deploy_020k.pt`
- C2：`results/mts_glt_distill_repair_control/c2/student/student_deploy_020k.pt`

冻结探针直接读取 `DistillStudent.pool(md_residual(canonical_o8))` 的 512 维 graph state，即部署包实际提供、位于下游 wrapper（新版为 Identity，不含 adapter 参数）之前的 pooled O8＋MD200 表示。C0/C1/C2 在 xc、eps 的每个 outer fold 上分别执行 Ridge `alpha={0.1,1,10,100}`；特征和标签 scaler 只拟合 train，alpha 只由原始标签尺度 validation R² 选择，test 不参与选择，也不进行 train＋validation refit。

C0 分阶段微调把任务 head 定义为下游 predictor：新版 DistillStudent 的 wrapper 是 Identity，因此 head 只有 `Linear(512,512)→GELU→Dropout(0.1)→Linear(512,1)`；本节更早记录的正式结果对应 `LayerNorm→Linear→LayerNorm→ReLU` adapter 与 regression MLP 组成的 head。第一阶段固定 10 epochs，只训练该任务 head，O8＋MD200 参数和 buffer 不变且保持 `eval()`；第二阶段从第一阶段 validation 最优 head 状态开始，使用新 optimizer 解冻联合训练最多 90 epochs。第二阶段采用 O8/MD `1e-5`、predictor `1e-4`、5 epochs warmup、cosine、patience 10；全流程最佳模型可来自任一阶段，并且只由 validation R² 更新。

正式入口与产物：

- 探针：`scripts/run_mts_c0_transfer_probes.py`；30/30 单元位于 `results/mts_c0_transfer_optimization/probes/`。
- 分阶段微调：`scripts/run_mts_c0_staged_finetune.py`；40/40 单元位于 `results/mts_c0_transfer_optimization/staged_finetune/`。
- 强审计与汇总：`scripts/report_mts_c0_transfer_optimization.py`；报告位于 `results/mts_c0_transfer_optimization/comparison/`。

所有正式预测均按原 manifest 的 test indices 保存，八个任务的五折 outer-test 各覆盖每个样本一次。正式 Macro8 R² 为 `0.8010382929`，相对既有 C0 为 `-0.0050736252`；因此分阶段方案没有改善整体八任务泛化。冻结探针中 C1/C2 在 xc、eps 均不弱于 C0，诊断证据更偏向微调适应问题，但只覆盖两个任务且不能单独证明因果机制。详细结果见 [`RESULTS.md`](RESULTS.md)。

## 14. DistillStudent O8 与下游 predictor 的新版实现

`mts_glt_version=distill|distill_repair` 使用独立的 `PreLNO8Encoder`。其六层
`SourceQPreLNLayer` 保持 hidden=512、FFN `512→2048→512`、Pre-LN、
source-Q/target-K、incoming softmax、SPD/path bias 和 dropout=0.1；FFN
激活固定为 `nn.GELU(approximate="none")`。这项激活替换只适用于新版
DistillStudent，不会全局替换 MTS-GLT-v2 或其他路线的 ReLU。

新版学生下游不再使用 graph norm/projection adapter：wrapper 直接输出
512 维 pooled O8＋MD200 表示，任务 predictor 固定为
`Linear(512,512)→GELU(approximate="none")→Dropout(0.1)→Linear(512,1)`。
Identity wrapper 不包含可训练参数；optimizer 仍覆盖 O8、MD200 和 predictor，
且不会重复或遗漏参数。运行配置会将新版学生的实际 `head_dropout` 固定记录为
`0.1`，即使旧 CLI 默认值为 `0.25`。

预训练 deploy metadata 记录 `o8_ffn_activation=GELU(approximate='none')`、
`o8_ffn_hidden=512->2048->512` 及新版下游 predictor/adapter 合同。已有
ReLU 训练产物不会因为参数形状相同或 strict loading 成功而被视为 GELU
预训练；本次不执行 checkpoint 转换、resume、预训练或微调。

因此，`RESULTS.md` 中 C0/C1/C2 的 120 个 `outer5_inner20` 正式单元、
C0 分阶段微调 40 个单元及各组 `historical_shared5` 重新微调结果，均由
137 维输入＋独立 backbone embedding、ReLU FFN、adapter head 的旧学生产生。
现有 `results/mts_glt_distill_repair_control/*/student/student_deploy_*.pt`
也不含新版架构 metadata，会被下游部署校验拒绝；新版评估必须先重新预训练
并导出学生包。

## 15. 已完成：New-C0（新版 GLT-V2 revision-2 无 MD＋MIPS-loss baseline）

New-C0 是修订后 GLT-V2 路线的纯 O8 基线。本轮明确移除 MD200，预训练只保留
原 MIPS masked-atom CE，下游使用 standard-label MSE；这是整体方案比较，不把差值
单独归因于某一个组件：

```text
GLT-V2 revision-2
├─ 138-D 原子输入（137 维 mips_x ＋ is_backbone 列）
├─ full-row canonical atom masking（30%）
├─ 6 层 Pre-LN O8（hidden 512 / 8 heads / head_dim 64 / 1/sqrt(64)）
├─ GELU（approximate="none"）
├─ 无 MD200 参数、输入或 corruption
├─ 单路 masked-atom CE（30%，138-D 整行清零，global sum/count）
├─ 无 3D teacher、line sidecar、local/global 蒸馏
├─ canonical atom mean pooling
├─ Identity graph wrapper
└─ 512→512→1 predictor，dropout 0.1
```

无 MD 分支的 `DistillStudent` 不实例化 `AtomicConditionedMD200`，canonical pooled graph
state 直接来自 O8；部署包只保存 O8 state，state key 共 80 个且不含 `md`。

实验身份与产物位置（与历史 C0/C1/C2 隔离）：

```text
config            configs/mts/glt_v2_r2_o8_nomd_mipsloss_005k.json
experiment_id     glt_v2_r2_o8_nomd_mipsloss_005k
result_root       results/glt_v2_r2_o8_nomd_mipsloss_005k
log_root          logs/glt_v2_r2_o8_nomd_mipsloss_005k
入口              scripts/run_glt_v2_r2_o8_nomd_pipeline.py {preflight,validate,smoke,pretrain,finetune}
```

该 config 通过 `protect_result_roots` 声明旧正式目录为受保护目录；新的预训练/下游/探针
产物均写入独立根目录，未修改 geometry、topology 或 MD200 cache。

训练协议：PI1M_v2 全量同序 cohort，3 GPU、local batch 84、accumulation 4、global batch
1008、BF16、AdamW `lr=2e-4`、`betas=(0.9,0.98)`、`weight_decay=0`、seed 42；实际
5,000 updates，warmup 2,000，并使用原 20k 调度曲线前缀（`schedule_total_steps=20000`）。
正式部署为 `student_deploy_005k.pt`；训练记录 5,000 行，首/末 loss 为
`4.565/0.208`，峰值显存约 `1.372 GiB/card`。

下游使用 `outer5_inner20`、8 tasks×5 folds、seed 42、batch 32/64、FP32、
`target_transform=standard`、`regression_loss=mse`、O8-only predictor
`512→512→1`（dropout 0.1），已完成 40/40。冻结表示 Ridge 探针已完成 80/80：
旧 New-C0 5k/10k/20k 各 20 个，加新 no-MD 5k 20 个；特征/标签 scaler 只拟合 train，
alpha 只由 validation R² 选择。

新 no-MD 下游 Macro8 R² 为 `0.787`；旧 New-C0 5k/10k/20k 参考值分别为
`0.782/0.779/0.774`。这些是描述性对照，因为旧包仍含 MD200、
历史下游 loss/标签协议不同，不能当作只移除 MD 的因果消融，也不是独立盲测。
逐任务、逐折指标和 80 个 probe 结果见
[`results/glt_v2_r2_o8_nomd_mipsloss_005k/comparison/final_report.md`](results/glt_v2_r2_o8_nomd_mipsloss_005k/comparison/final_report.md)。

执行中三类入口问题均已停止、修复并保留证据：首轮预训练错误压缩学习率曲线（约 1,460
updates），下游参数曾使用非法 `mts_glt_mode=none`，探针入口曾缺少 `src` 路径。修复后
正式结果均从独立合法目录完成，失败尝试不计入指标。

## 16. 新增：完整 Trimer GLT（端点元素＋14 维键特征）

本节只记录已实现的独立代码接口和局部验证，不代表已启动新的预训练或微调。旧
`periodic_line_glt_image_v1`、旧 C1/C2 checkpoint 和正式结果均未修改。

新 sidecar schema 为 `mts-periodic-line-glt-complete-v1`，由冻结的三 RU Trimer
真实物理键图直接生成内存记录：每条唯一物理键各一个 token，通常为 `3N+2`；每个
token 持有端点 `(canonical_atom_id, RU_offset)`、真实欧氏键长、端点原子序数、
`token_bond_features=[BondType(5), Conjugation(1), Ring(1), BondStereo(7)]`，即
`glt3_token_bond_features: float32[M,14]`。Ring 来自开放 Trimer 的真实 RDKit 环，
不依据周期闭合拓扑推断；Stereo 的 unknown 与 NONE 分开。所有共享真实物理原子的
有向一跳 line-graph 关系保留对应真实角度；不存在距离/角度均值或重复物理三元组。

完整分支的 token 编码为：

```text
one_hot(Z_a, 101) → Linear(101,256) ┐
one_hot(Z_b, 101) → Linear(101,256) ├→ sum + Linear(14,256)
RBF(distance; unordered Z pair) → Linear(256,256) ┘
                                   concat → LayerNorm → 512
```

两个端点共享 `Linear(101,256)` 并求和；键化学向量使用一个
`Linear(14,256)`，不再叠加旧的 BondType/Stereo/Conjugation embedding。可选
`token_mask` 在编码后用独立可学习 512 维 mask token 替换整行；`angle_mask` 只清零
对应关系的角度 bias。新 `CompleteTrimerGLTEncoder` 保持当前 8-head、Pre-LN、
GELU、6 层和 source-Q/target-K attention，全部物理键参与消息传递，最终只对
中心 RU 内部键做 GAP。无中心键时输出零图向量并标记无有效 3D readout。

下游接口 `CompleteTrimerGLTFusionRegressor` 接收已完成 GAP 的 O8 `[B,512]` 与中心
GLT `[B,512]`：分别 LayerNorm 后拼接为 `[B,1024]`，经过
`Linear(1024,512)→GELU→Dropout(0.1)→Linear(512,1)`。无有效 GLT 时在 GLT
LayerNorm 后置零，2D O8 路径仍可预测。`scripts/build_mts_glt_v3_sidecars.py`
新增 `line-complete` 入口，但本轮没有调用它、没有写新 sidecar、没有生成构象。

相关单元测试覆盖物理键数量、14 维顺序、Ring/unknown Stereo、端点交换、整 token
mask、批处理 `[M,14]` 检查、融合形状、新旧 sidecar 混用拒绝、缺失 Ring 字段拒绝、
sidecar 写入/读取往返以及刚体旋转/平移不变性；测试结果为 `8 passed`。另在 `tmux` 的
`complete_trimer_validation` window
中对冻结缓存的真实样本 1（50 tokens/16 个中心键/132 relations）和样本 10（2
个跨 RU tokens/无中心键/2 relations）完成只读构建、批处理及 CPU forward/backward；
两者均有限，样本 1 的 endpoint、bond-feature、distance、angle 和 GLT block 梯度均
为非零。验证日志保存在 `logs/complete_trimer_glt_validation.log`；window 已正常结束。

## 17. 独立新接口：O8 Bond-Path＋Galformer Trimer Hop2

本节记录当前 GLT-V2 双路接口及其规范化身份修复。代码回归已在本轮执行；真实冻结记录
审计已执行但因坐标化学异常阻断后续模型 smoke，不能把合成测试结果外推为真实数据覆盖。
旧 O8、CompleteTrimerGLTEncoder、C0/C1/C2 和历史 checkpoint 行为保持原样。

新模型工厂为 `src.modules.build_dual_glt_model(fusion_mode="concat" | "kfuse")`。
两路都是 6 层、512 hidden、8 heads、source-Q/target-K、incoming softmax，
QK 乘 `512**-0.5` 后再加 bias。没有 MD200、CLS、教师或新预训练 head/loss。

### 数据接口

`src/dataset/glt_dual.py` 提供 `build_dual_sample(topology, trimer, smiles)`、
`DualGLTDataset` 和 `dual_glt_collate`。输入已有 canonical 拓扑与冻结坐标；
输出仅模型所需的原子、物理键、路径及有效性字段，不向 batch 传递 MD 或坐标。
`FrozenDualLayerSource(topology_root, trimer_root, samples)` 使用现有只读
`LmdbLayerStore`，仅打开 topology/trimer 两层，检查完成标记、当前 schema 和 key，
缺少记录时报错，不调用缓存生成器。`samples` 为 `(32字节 key, P-SMILES)` 序列。
通用 `DualGLTDataset` 包装其他 source 时，调用者也必须关闭 MD 和构象生成。

2D 沿既有 `lga_path_index/lga_path_shift` 生成 `[R,2,14]` 键特征与 `[R,2]`
mask，不改 O8 的关系集合。化学 helper 与旧完整 Trimer 共用：BondType(5)、
Conjugation(1)、Ring(1)、Stereo(7)。用平移不变的物理键身份查询跨 RU 属性，
可处理 RU±2 拓扑路径；不平均或合并不同 shift 的关系。不同副本的键化学若不一致则
明确报错，不任意挑选。Ring 沿用开放 Trimer 的真实环语义。

3D 每条物理键一个 state。先保留完整一跳物理 line graph，再为每个无向物理
token 对保留全部等长最短路径，反序生成反向路径。逐路径生成 head bias 后，
按有向 token 对取算术平均；每个有向对仍只发送一条消息，不按路径数重复计权。
`line_path_group` 将路径行映射到关系行，collate 分别处理 token 和关系偏移。
不再按 token ID 选择几何路径。最多两个 line hops，
即三个 bond tokens、两个真实角度。self 单独标记，使用 `[token,token]` 与合成零角。
无效必需角度使整个样本 3D 无效，附带原因，不静默删除关系；空中心键读出为零。

Trimer 构建在完整连接后从 Kekulize 前的原始分子复制 BondDir/stereo，并重映射
StereoAtoms。内部 dummy 参照只映射到相邻 RU 的真实连接原子；有限链末端没有真实参照
时显式记为 `terminal_stereo_unset`/`STEREONONE`，不任意选取另一取代基或翻转 E/Z
(CIS/TRANS)。修复不重建或修改冻结坐标。已有坐标是否满足原始 E/Z 必须独立验证。
规范化 P-SMILES、缓存显式映射和调用方原始字符串通过严格图同构关联；bond-path、BRICS、
物理键和监督索引均从规范化 base atom 身份生成，原始字符串只保留为来源和独立 Stereo 审计。
真实验证仍最多两条冻结记录：普通记录须含明确的中心 E/Z，另一条为 N=0；同时检查
原坐标立体一致性和重编号预测不变性。

### 两路编码器

2D 保留 138 维输入及整行 mask、SPD/path-node bias、Pre-LN/GELU。
新增 bond bias：四项类别 embedding 各 8 维并求和，两个路径位置各一个 8×8
矩阵，按有效路径键数平均。SPD＋path-node＋bond bias 每次 forward 一次生成，
六层复用，不 detach。self bond contribution 为零。这是 MolGT 路径运算加
Galformer-style 跨层共享的项目适配，**不是 MolGT 的逐层 bias 实现**。

3D 使用 Galformer 的端点元素 `101→512` 共享投影求和、元素条件的 256 维
归一化 Gaussian 距离编码投影到 512，再以 `1024→512→512` GELU MLP 生成
token；不直接输入 14 维键化学。BondType 仅参与 `Vocab(Za,Zb,BondType)` 的
path-angle 类型条件。角度使用 128 维 Gaussian、两个位置 MLP、有效平均、
`128→128→8` head MLP，跨层共享。padding 在位置 MLP 后再次清零。
移除独立 self bias、输入与最终 LayerNorm；层内遵循 Galformer residual，
FFN 为 `512→2048→512`，feature/attention dropout=0.1。无虚拟 CLS 节点。
物理 line-hop 构图和中心池化是聚合物适配，不等于官方原子路径/虚拟节点构图。

参考：[MolGT modeling.py](https://github.com/robbenplus/MolGT/blob/master/src/models/modeling.py)、
[Galformer module_utils.py](https://github.com/peizhenbai/Galformer/blob/main/model/module_utils.py)、
[Galformer model_3d.py](https://github.com/peizhenbai/Galformer/blob/main/model/model_3d.py)。

### 融合与输出

`encode(batch)` 返回 `atom_states`、`bond_states`、`center_bond_states`、
`graph_2d`、`graph_3d`、`geometry_valid` 与两路 bias。`forward(batch)` 返回 `[B,1]`。
3D 的有效性统一为 `geometry_valid AND center_count>0`，仅中心内部键 GAP。

* `concat`：两路 GAP 分别 LayerNorm，3D 在 LN 后 mask，再拼接；
  predictor 为 `1024→512→GELU→Dropout(0.1)→1`。
* `kfuse`：原版 `OriginalMIPSAttentiveFusion`，仅 `glt3d` 一项 knowledge，
  Query/Key 为 128 维、Value 为 512 维、缩放 √512、残差系数 0.5。
  融合发生在 O8 原子 GAP 前，无额外 graph norm/adapter；
  predictor 为 `512→512→GELU→Dropout(0.1)→1`。
  单 knowledge 的 softmax 恒为 1，所以这是各原子接收相同 3D 投影残差的特例，
  Query/Key 梯度为零是预期行为。无效样本屏蔽整个投影残差，包括 Value bias。

新模型不接入旧正式 runner，不加载旧 checkpoint 作为完整新模型权重。

### 验证入口与本轮状态

`tests/test_dual_glt.py` 覆盖共享/scaling、独立参考公式、Vocab 顺序、周期路径、
物理两跳与反序、无效几何、N=0、刚体不变性、两种融合和梯度。
其人工坐标样本不宣称为真实 fixture。

本轮在 `tmux` 的 `glt_local_tests3` window 中执行了相关回归：

```text
PYTHONPATH=tests pytest -q tests/test_dual_glt.py tests/test_dual_glt_audit.py \
  tests/test_complete_trimer_glt.py tests/test_mts_canonical_periodic.py
```

结果为 `57 passed`，日志为 `logs/glt_v2_local_tests3_20260911.log`。辅助的三任务目标、
LMDB 坏样本、缓存完整性和 no-MD 路线回归在 `glt_aux_tests2` window 中为 `37 passed`，
日志为 `logs/glt_v2_aux_tests2_20260911.log`。这些用例包含合成模型前后向和梯度检查，
不构成真实构象或性能证据。
随后在补强“身份/化学错误不得降级为空几何行”及审计完整 JSON 状态后，重新执行包含
`test_dual_glt_pretrain.py` 的目标集合，结果为 `71 passed`，日志为
`logs/glt_v2_regression_postfix2_20260911.log`；仍只有合成/契约证据。
再执行覆盖 LMDB、checkpoint 生命周期、no-MD、规范化 Trimer 映射及完整性校验的扩展
目标集合，结果为 `104 passed`，日志为 `logs/glt_v2_regression_final3_20260911.log`。
补充非对称 Z 构型的规范化端点反向、映射字段异常、审计收尾退出码及“全部明确副本均检查”后，
审计目标集合为 `65 passed`（`logs/glt_v2_audit_final20_20260911.log`）；随后重跑完整
扩展目标集合为 `107 passed, 6 warnings`（`logs/glt_v2_regression_final21_20260911.log`）。

真实验证入口只接受两条现有缓存记录（一条普通、一条真实 N=0），CPU、单线程、
`num_workers=0`，两种融合各一次 forward/backward，无 optimizer、无缓存写入：

```text
python scripts/validate_dual_glt.py --topology-root TOPOLOGY_LAYER --trimer-root TRIMER_LAYER --sample ORDINARY_KEY "ORDINARY_PSMILES" --sample N0_KEY "N0_PSMILES"
```

以上路径与 key 需对应执行环境实际冻结产物，不使用历史样本编号猜测新环境身份。
本轮实际只读审计了 PI1M_v2 的两条记录（普通中心 E/Z 与真实 N=0），最终代码命令在
`glt_real_audit_final22` window 中执行；stdout JSON 和 `--report-json` 均保存为
`logs/glt_v2_real_audit_final22_20260911.json`，日志为 `logs/glt_v2_real_audit_final22_20260911.log`。
两条记录的规范化身份、原始 Stereo、2D/连接审计和 N=0 角色均有明确状态；普通记录三个
明确中心副本中有两个冻结坐标投影与 E/Z 不一致，退出码为 `1`、`outcome=DATA_ANOMALY`。
普通样本 Star-Linking 为 `REVIEW`（实际边为 aromatic/ring/conjugated），N=0 的
Star-Linking 为 `REVIEW`（两侧落在同一 boundary atom）；模型状态为 `NOT_RUN`，因此
没有执行真实记录 forward/backward。

## 18. GLT 双路三任务预训练与 outer5_inner20 微调（实现；本轮未执行）

本节是独立新入口；第 17 节模型及冻结输入不再需要借用旧 C0/C1/C2 runner。
两种模式分别使用 `configs/mts/glt_dual_three_task_concat.json` 与
`configs/mts/glt_dual_three_task_kfuse.json`。无 MD200、教师、蒸馏或 InfoNCE。
本轮未启动预训练、微调、DDP 或缓存构建；相关合成回归已在第 17 节记录，以下正式
命令仍仅供通过真实数据阻断复核后的环境使用。

### 数据、目标与共享融合

`src/dataset/glt_dual_pretrain.py` 从只读 topology/Trimer 记录按需构建输入和独立
targets。开放 Trimer 按 BRICS 切分后的连通分量与中心 RU 求交，再映射为 canonical
原子组；整组选取约 30%，至少保留一个原子。只有一个组时使用连通子集退化并计数，
单原子跳过元素 loss。138 维整体遮蔽，path-node 使用遮蔽后的 embedding。
元素头是两层 incoming-mean 图解码器，仅读取 O8 hidden；不读取 3D 或融合结果。

完整 Trimer 坐标副本添加独立 `sigma=0.03 Å` Gaussian 噪声，从同一受扰动坐标
重算全部距离/角度/path bias。干净标签不进入 encoder；原本有效的几何被噪声破坏
时明确报错。长度监督仅取中心内部键，角度监督仅取两条中心内部键的真实无向夹角，
每个只计一次；不监督多路径重复和 self 合成角。`N=0`/无效几何跳过几何任务；
无中心角度只跳过角度项。长度误差单位为 Å，角度目标为 cosine。

指纹标签来自无坐标的开放 7-RU 拓扑，以中心 RU 为 roots，radius=2、2048 bit、
`includeChirality=False`。验证 roots 到两个开放端点至少 3 hop；不满足时报错。
标签不作为输入，不引入额外 3D RU。确定性化学目标仅有进程内 512 项 LRU 缓存，
不会写全量 sidecar。`DualGLTModel.fuse(encoded)` 为预训练和微调共享的融合接口。

三项任务分别先按每个样本的有效目标平均，再对有效样本平均：

$$
L=L_{chem}+L_{geo}+0.1L_{FP},\quad
L_{geo}=\operatorname{mean}_{E_0}(\hat d-d)^2+
\operatorname{mean}_{A_0}(\widehat{\cos\theta}-\cos\theta)^2.
$$

几何任务以有中心键的有效样本为分母，没有角度的样本角度项为零。指纹采用逐 bit
BCE；数据适配拒绝无效 2D 拓扑，但保留无效 3D 的合法 2D 样本。跨 rank 按全局
sum/count 归约；日志包含任务均值、有效样本数、目标数量、各 rank 的退化数及跳过原因。
两种融合各自训练；单 knowledge KFuse 的 Query/Key 零梯度仍是预期行为。

论文依据为 [Motif-Aware Attribute Masking (2025)](https://proceedings.mlr.press/v269/inae25a.html)、
[SCAGE (2025)](https://www.nature.com/articles/s41467-025-59634-0)、
[FlexMol (2025)](https://arxiv.org/html/2510.07035v1)。BRICS 周期分组、中心标量去噪、
7-RU rooted 指纹和损失权重均为项目适配，非逐代码复刻或已验证最优设置。

### 预训练调用与恢复

`scripts/pretrain_glt_dual.py` 支持单进程/DDP；默认 microbatch=84，有效 batch=1008，
累积次数自动取 `1008/(84*world_size)`，不整除即报错。逐 optimizer update 按绝对
抽样位置分配样本；每 epoch 确定性打乱，跨 epoch 连续填满有效 batch。样本 key、seed、
绝对位置决定 mask/噪声。数据在主进程按 microbatch 构建，不启动 DataLoader workers。
BF16 仅用于模型，几何和 loss 使用 FP32。AdamW 默认 betas=(0.9,0.999)，weight decay=0。

5000 updates，LR=2e-4，warmup=2000，cosine horizon=20000，end LR=1e-9，每 1000
步保存完整恢复包和部署包。注意旧 no-MD runner 源码实际使用线性衰减；本入口按新计划
使用 cosine，因此不是旧 runner 学习率轨迹的逐步复刻，不宣称匹配旧训练。
恢复包保存任务头、优化器、调度位置、各 rank RNG、顺序样本身份及下一数据位置；
要求同配置、world size、输入路径和样本顺序。恢复不得覆盖已有的更晚 checkpoint。
部署包仅含 O8/GLT/fusion/norm，不含任务头或性质头，严格核查 fusion/step 与张量集合。

下面 shell 模板在工作目录的 `tmux` session `Uni-Poly` 独立 window 内使用。
先检查已有 session/任务，避免重复启动；`MODE` 分别设为 concat/kfuse；所有数据路径
必须指向真实冻结层，输出目录必须全新。模板不自动串行启动另一模式或微调。

```bash
MODE=concat
mkdir -p logs/glt_dual_three_task
set -o pipefail
torchrun --standalone --nproc_per_node=3 scripts/pretrain_glt_dual.py \
  --config configs/mts/glt_dual_three_task_${MODE}.json \
  --samples-csv PI1M_CSV --topology-root TOPOLOGY_LAYER --trimer-root TRIMER_LAYER \
  --output results/glt_dual_three_task/${MODE}/pretrain \
  2>&1 | tee logs/glt_dual_three_task/${MODE}_pretrain.log
```

同一命令增加 `--resume results/.../resume_01000.pt` 恢复；必须使用原输出目录。
这些命令需要用户单独授权执行，不属于当前已运行产物。

### 下游五折与指标

`scripts/finetune_glt_dual.py` 复用既有 train_epoch/evaluate/test_model、standard target
scaler 和 split 生成函数，隔离旧模型工厂与默认参数。固定八任务×五折，优先读取
`data/splits/mips_outer5_inner20/{task}.json`；缺失时按 outer seed=1、inner seed=42+fold
生成并保留。存在的 split 若不匹配则报错，不覆盖。300 条为 192/48/60。

每 fold 独立初始化性质头，加载对应模式的 5000-step 部署包；clean Trimer，无 mask、
无噪声、无预训练标签。scaler 只拟合 train，standard-label MSE。两路 LR=1e-5，
fusion/head LR=1e-4，AdamW weight decay=.02，batch=32/eval=64，FP32，worker=0；
最多 100 epochs、warmup=5、patience=10、clip=1。非有限 loss/gradient/指标报错。
按 validation R² 选权重，恢复后 test 只评估一次。每行一次 OOF 预测；五折统计 std
采用 ddof=1，macro8 为八个任务的 fold-mean R² 均值，pooled OOF 指标另列。
这是既有开发折评估，不是独立盲测。微调不提供自动断点续跑，不覆盖已有输出目录。

```bash
python scripts/finetune_glt_dual.py \
  --config configs/mts/glt_dual_three_task_${MODE}.json \
  --checkpoint results/glt_dual_three_task/${MODE}/pretrain/deploy_05000.pt \
  --raw-root data/raw --topology-root DOWNSTREAM_TOPOLOGY_LAYER --trimer-root DOWNSTREAM_TRIMER_LAYER \
  --split-root data/splits/mips_outer5_inner20 \
  --output results/glt_dual_three_task/${MODE}/downstream \
  2>&1 | tee logs/glt_dual_three_task/${MODE}_finetune.log
```

### 局部验证与阻断状态

三任务目标、DDP 归约代数、优化器/RNG 恢复、部署张量集合、固定分折和 clean 下游
helper 的合成回归已执行（`37 passed`，见上节日志）；身份/审计补强后的目标集合为
`71 passed`，最新审计相关集合为 `65 passed`，最新扩展目标集合为 `107 passed, 6 warnings`
（`logs/glt_v2_regression_final21_20260911.log`）。`finetune_glt_dual.py` 新增
`--smoke --task eat --fold 0`，只构建 train/validation loader、只用 train 拟合 scaler、
最多两 epoch，并明确把 outer-test/OOF 标为 `NOT_RUN`；默认不带 `--smoke` 仍是完整
八任务五折行为。该 smoke 入口尚未执行。

真实双记录入口使用：

```text
python scripts/validate_dual_glt.py --audit-only --topology-root TOPOLOGY_LAYER \
  --trimer-root TRIMER_LAYER --sample ORDINARY_KEY "ORDINARY_PSMILES" \
  --sample N0_KEY "N0_PSMILES" --report-json NEW_REPORT.json
```

真实入口最多两条冻结记录、CPU 单线程、无 workers/optimizer/训练循环；三任务前后向、
内存中 step=0 包的严格加载、clean 下游 forward。step=0 仅为验证包，不冒充已训练模型。
测试涵盖 mask 泄漏、目标几何、fingerprint reference、DDP 梯度归约的代数参考、优化器/RNG
恢复、绝对位置抽样、部署加载与 300 条分折；以隔离 mock 检查 validation 选模后 test
只运行一次。人工测试坐标不是实际构象。当前没有实际 DDP、真实记录模型前后向或预训练
smoke 验证。

### 数据保真审查后的局部修复（2026-09-11，本轮状态）

本次明确了 2D 周期键化学的代表规则：内部键只取开放 Trimer 中心 RU；有限链末端
内部键的 Stereo/Conjugation 不参与周期特征一致性判定。两条真实跨 RU 键仍须具有
相同的 14 维化学特征，否则显式报错，暂不任意选择副本或扩展拓扑。3D 继续保留
全部物理副本及独立几何；没有改变 star 连接类型策略、模型架构或冻结缓存。

完整 Trimer 读取新增全部 `(base_atom_id, RU_offset)` 身份、逐物理原子元素、canonical
覆盖以及预期/冻结物理键集合相等检查；缺边、额外边和键类型冲突使 3D 分支无效并
给出原因。允许有向或无向边表，正常双向存储仅在物理键层归并。缺少程序必需字段或
身份/化学解析错误现在显式拒绝，不再由双路适配器包装为普通无效角度。

复制 Stereo/BondDir 的来源改为 Kekulize 前的原始分子。`validate_dual_glt.py` 独立从
原始 P-SMILES 记录 Stereo、StereoAtoms、BondDir，并核对全部内部键副本的参照关系；
末端没有真实物理参照时只允许明确变成未指定，不能用另一显式取代基自证翻转。该检查
验证 metadata 传递，不等同于完整 CIP 重新赋值验证。化学传递检查通过后再检查冻结坐标；
轴长、投影退化有独立错误原因。
原有坐标一致性阈值 0.5 保持不变，它是几何容差判据，不是纯 E/Z 符号定义。

审计报告区分 `ANOMALY`（已发现异常）与 `REVIEW`（策略或有限链差异待复核），
提前返回的检查显式为 `NOT_RUN`。真实普通样本必须自身包含中心 E/Z；真实 N=0
须同时满足原始结构内部键数为零、模型读出键数为零。修复 LMDB 初始化失败清理、
双资源关闭与模型失败状态；正常审计 stdout 只输出最终 JSON，进度写 stderr。
`REVIEW` 不被写成 `PASS`，但不阻止审计继续收集其他检查；真实记录的明确坐标异常会
使审计以 `DATA_ANOMALY` 退出并阻止后续模型/训练 smoke。它不授权正式实验。
CLI 参数格式错误仍使用 argparse 的 stderr/非零退出，不生成样本审计报告。
资源关闭、fixture 收尾或 JSON 序列化失败均标记 `SCRIPT_ERROR` 并返回退出码 `2`；
非有限报告值不会以 NaN 写出。三任务真实验证入口复用相同数据审计，避免只凭
geometry_valid 进入模型验证。

新增 `tests/test_dual_glt_audit.py`，并扩展 `tests/test_dual_glt.py`：覆盖末端 Stereo
到 bond bias、缺边与真实身份契约、无向存储、重编号、原始参照审计的故障注入、
fixture 角色绑定、资源释放、JSON 输出、多路径独立平均公式和混合 batch 偏移；本轮
相关合成回归记录为 `57 passed`、`37 passed`，补强后目标集合 `71 passed`，审计子集
`29 passed`，最新审计相关集合 `65 passed`，最新扩展目标集合 `107 passed, 6 warnings`
（`logs/glt_v2_regression_final21_20260911.log`）。
没有生成真实 fixture、没有重建缓存、没有修改历史
结果，也没有启动预训练或微调。两条真实记录仅代表这两条记录，不外推全量覆盖率。
