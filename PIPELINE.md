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
