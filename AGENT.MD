# Uni-Poly 实现约束

## 文档规范

- 本仓库所有 `.md` 文件中的数学公式必须使用 `$$` 作为行间公式分隔符
  （如 `$$E=mc^2$$`），不得使用 `\[` `\]` 或其他 LaTeX 分隔符。

## 当前活动路线

- 路线一 `flat4` 已从生产入口移除；历史代码、结果和通用组件不再作为可运行路线。
- 当前唯一生产路线正式命名为 `MIPS-Trimer-SCAGE`，简称 `MTS`，入口为
  `BASELINE=MIPS-Trimer-SCAGE`。
- `flat4`、`scage_parallel`、独立 `mips`、M5/M6/M7、O/S/G 实验矩阵以及所有
  MIPS-PBC 路线均已退出活动入口，禁止将这些旧名称静默映射到新路线。

## MIPS-Trimer-SCAGE（MTS）

生产模型的 Graph 路径固定为：

```text
O8 两跳 Star-Linking 拓扑
+ per-head SPD bias / single-path-node bias
+ 对称的直接 Star-edge Trimer RBF bias
→ canonical atom scatter-mean
→ 两层“中央 RU query / 完整 Trimer key-value”的 SCAGE-MCL
→ zero-gated 几何残差广播到全部 O8 copies
→ 原子 mean pooling
→ zero-gated 低容量 graph-level MD200 残差
→ 回归头
```

默认生产配置 `mts-config-v2` 不得包含：

```text
AtomPair3D512
SMILES encoder
FP encoder
多模态 Stage 2
PBC / cell / image shift
带方向的 RU embedding
PaiNN
完整 SCAGE encoder
```

## 原子身份与几何约束

- O8 节点身份为 `(canonical_atom_id, mips_copy_id)`。
- Trimer 原子身份为
  `(canonical_atom_id, ru_offset=-1|0|+1)`。
- 每个 O8 copy 只能映射到 Trimer 中央 RU 内相同 canonical atom。
- 禁止将 `mips_copy_id` 解释为 Trimer 的 `ru_offset`。
- O8 的 137 维原子特征从 topology-only Trimer 的中央 RU 计算，然后按
  canonical atom 广播。这样虚拟 Star 边两端具有聚合物化后的
  degree、隐式 H 和 hybridization 语义。
- Star 虚拟边不得写入 Trimer 的真实键表。
- `d_star` 是 Trimer 两条真实 inter-RU 键长的对称平均值。
- 当两条键长的不对称度超过 `0.15 Å` 时，只关闭 Star RBF bias，不关闭
  Trimer-MCL。
- Trimer 几何无效、Star 距离无效和 MD200 无效必须分别执行独立的精确
  零回退，不得删除样本。
- 默认 ETKDG 快速路径失败时，只对该样本执行 2 候选、
  `useRandomCoords=True`、最多 200 次迭代的确定性重试；禁止全数据增加
  候选数。每个候选统一执行 MMFF94 固定 200 步松弛；不要求收敛状态，
  仅接受坐标和最终能量均有限的候选，并选择最低有限能量。
- 超过 384 个重原子的 Trimer 可保留 RDKit 2D 坐标和映射用于诊断，但
  必须设置 `trimer_geometry_valid=False`，不得进入 MCL 或 Star RBF。
- 默认生产配置中，MCL 距离只用于 20% 和 50% 分位数 hard visibility
  mask。`mts-experiment-v3` 可在不改变冻结缓存的前提下启用零初始化的
  64-RBF 连续距离 bias，但不得加入方向、RU-offset 或 PBC 信息。

## 训练与 Schema

任务命名契约固定为：

```text
路线：MIPS-Trimer-SCAGE（MTS）
Stage 1：MTS Joint Pretraining     / mts_joint_pretraining
Stage 2：MTS Property Fine-tuning  / mts_property_finetune
```

缓存 schema 保留历史 `mips-trimer-scage-*` 前缀，因为它们标识已冻结的
特征内容；路线改名不得触发缓存重建。

- Stage 1 `MTS Joint Pretraining`（内部标识 `mts_joint_pretraining`）：
  - 完整 PI1M_v2（995,799 条），20,000 optimizer steps；
  - 一次 O8 + Star RBF + 两层 Trimer-MCL forward；
  - 只训练 `1.0 masked-atom + 0.25 Trimer bond-angle`；
  - SPD/path bias保留为结构输入，但不再训练SPD/path prediction head；
  - MD200、回归头不参与预训练。
- Stage 2 `MTS Property Fine-tuning`（内部标识 `mts_property_finetune`）：
  - 直接加载联合预训练 checkpoint；
  - 只迁移 O8、Star-RBF、Trimer-MCL 和 geometry gate；MD200、Graph
    projection/norm 与回归头按 fold seed 重新初始化；
  - 固定使用 `legacy_mts_huber_v1`：Graph 全部模块从 epoch 0 训练，Huber
    beta=0.5，batch 32，100 epochs，patience 10；
  - Graph LR=1e-5，Graph projection/head LR=1e-4，weight decay=0.02，
    前 5 epochs warmup，单次全程 cosine decay，梯度裁剪 1.0，SWA 关闭；
  - 使用 `TARGET_TRANSFORM=recommended`，不执行分阶段冻结、阶段 scheduler
    重启或 validation-selected SWA；先完成 seed 42 的完整 40 folds。
- 正式汇总必须先在原始标签空间按 `(task, fold)` 平均三个 seed 的预测，
  再计算指标；5-fold 标准差使用 `ddof=1`，结果显示为三位小数的
  `mean ± std`。原始 shard 和 prediction NPZ 保留完整精度。
- 只允许暴露 GPU `0,1,2`。
- 联合预训练采用 MIPS-like `Adam(beta1=0.9,beta2=0.98,eps=1e-8,
  weight_decay=0)`、峰值学习率 `2e-4`、2000-step warmup、polynomial
  power=1 衰减到 `1e-9`，不做梯度裁剪；目标全局 batch 为 `1008`。
- 正式运行前可用 `CUDA_VISIBLE_DEVICES=0,1,2 python scripts/mts.py
  benchmark-pretrain --batches 500` 比较
  `(42,8)/(84,4)/(168,2)/(336,1)`，结果写入
  `results/mips_trimer_scage/pretrain_benchmark.json`；通过 300-step resume gate
  后才固定所选组合。

```text
配置 schema       mts-config-v2
特征 schema       mips-trimer-scage-feature-v4
Trimer cache      mips-trimer-scage-trimer-v5
Trimer LMDB       mips-trimer-scage-trimer-lmdb-v3
Angle cache       mts-trimer-bond-angle-cache-v1
Angle-v2 sidecar  mts-angle-continuous-cache-v1
checkpoint schema mts-model-v2
```

## SOTA 实验约束

- `configs/mts/default.json` 始终保持 graph-only 生产配置；实验只能使用
  `mts-experiment-v3`，结果写入 `results/mts_sota_v3`。
- 允许的几何实验为：当前 MCL、关闭 MCL、coordinate-shuffled、MCL-RBF
  及其 shuffled 负对照。MCL-RBF 必须零初始化，初始输出与当前 MCL 一致。
- 可选下游模态仅为 PubChem10M SMILES 双 attachment view 和
  attachment-aware CountFP；只能通过以 MTS Graph 为锚点的零门控残差融合。
  不得恢复 parallel attention。
- SMILES 主干和 CountFP 只在 V 阶段作为低容量、零门控残差消融；V 的新增
  adapter/LoRA 不得改变 Graph 基线的 `legacy_mts_huber_v1` 优化策略。
- 多任务实验必须按 normalized sample key 从全部辅助任务中排除目标 outer
  test polymer，并使用 task-balanced sampling；PCGrad 只作用于共享参数。
- 实验顺序固定为 G（几何）→ V（模态）→ MT（多任务）→ FINAL
  （nested 5-fold）。F 已因退化移除；后续配置必须完整物化上一步胜者，
  不得假定某个候选已经晋级。
- 兼容轨只能标记为 `shared_validation_test_fold`、非独立盲测；最终胜者还
  必须执行 outer 5-fold + hash 固定 10% inner validation 的严格轨。
- Angle-v2 只有在真实 Trimer 坐标显著优于 shuffled 后才允许构建和重训。
  它从冻结 Trimer 派生 `cos(angle)`，不得重新执行 ETKDG/MMFF。

## 缓存约束

- 路线二使用 `mips-trimer-scage-lmdb-layout-v2`，不得恢复 2048 条
  PyTorch shard 作为生产缓存。
- attachment 必须是两个不同 dummy site，但允许二者指向同一个 canonical
  boundary atom。共享 boundary 时以实际展开的开放链节点计算距离，不得将
  单 RU 内的同一原子误判为 Star self-loop。
- attachment 键型一致时保留原键型（芳香 attachment 转单键）；不一致时
  统一使用单键，并保存左右键型、mismatch 和 connection policy 诊断。
- `topology_hash` 必须包含 `ru_base_hash`；`trimer_hash` 必须同时包含
  `ru_base_hash` 和 `topology_hash`。禁止下游层复用错误上游协议。
- 缓存按 canonical P-SMILES 的 SHA256 内容键跨
  PI1M_v2 是唯一预训练 cohort；下游 union 只用于提前构建 Stage 3 特征，
  不进入预训练采样。数据集名称、标签、seed 和 tokenizer 不得进入结构
  layer hash。
- `ru_base` 是 Topology 与 Trimer 的共同计算依赖。只缺 Trimer 时禁止
  重建 O8，只缺 MD200 时禁止执行 O8 或 Trimer。
- Graph-only 路线不得实例化 tokenizer、扫描 token 长度或产生
  SMILES/FP batch 字段。
- 训练读取使用 `mips_trimer_collate`，只组织 O8 sparse edge、canonical
  mapping、Trimer 和 MD200；legacy `custom_collate` 仅供历史诊断。
- O8/Trimer 使用逐样本 LMDB，MD200 使用 cohort 顺序的只读 NumPy mmap。
  缓存只能由父进程单写，48 个 worker 只负责生成。
- Trimer 失败必须写入带 `valid=False` 的正式记录，不能删除样本或写入
  2D 坐标供 MCL 使用。
- 预期的样本级失败必须写入 unavailable record；只有 schema、内部映射
  不变量、IPC、序列化或 LMDB 损坏可以终止整个缓存任务。
- PI1M_v2 必须先通过 500 correctness 与 10k performance preflight。
  当前 builder 10 使用 `relax200 + lowest finite`；旧 builder 9 的严格
  收敛统计不能用于判断当前协议，必须在新 Trimer writer 完成后重新审计。
  全量质量门只检查 mapping failure=0、2D进入MCL=0 和
  mcl-valid/graph-available≥90%。

旧 cache 和 checkpoint 必须提前拒绝并要求重建。除非用户明确要求，否则
不得删除历史结果、checkpoint、基础 encoder 或共享数据划分。

正式训练前使用 `scripts/mts.py doctor` 检查三张 GPU、冻结 cache bundle、
共享五折清单和模型契约。`scripts/mts.py pretrain` 只启动唯一的
`mts_joint_pretraining` 阶段；不再接受 topology/geometry 分流变量。
