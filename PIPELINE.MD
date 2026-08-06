# Uni-Poly 当前流程

## 活动路线

路线一 `flat4` 已删除，不再作为可运行入口。当前唯一生产命令为：

```bash
BASELINE=MIPS-Trimer-SCAGE bash scripts/run.sh
```

- `MIPS-Trimer-SCAGE`（简称 `MTS`）：默认生产配置为 Graph-only；SOTA
  候选通过独立 `mts-experiment-v3` 配置增加受控模块。
- 旧 `scage_parallel`、独立 MIPS、M5/M6/M7 命令会直接返回迁移错误。

## 路线二：MIPS-Trimer-SCAGE（MTS）

```text
P-SMILES
  ├─ O8 Star-Linking Graph ─ SPD/path/Star-RBF attention ─┐
  ├─ 开放 finite Trimer ─ 两层 SCAGE-MCL residual ────────┤
  └─ source_star_sub ─ RDKit2DNormalized MD200 residual ──┘
                                                         ↓
                                                       回归头
```

### 1. O8 拓扑

对输入 RU 执行最小整数倍重复，直到开放链两端的 boundary graph distance
严格大于 5。设最小合法重复数为 `k`，O8 中每个节点记录：

```text
(canonical_atom_id, mips_copy_id)
```

满足条件后，在开放重复图两端加入一条虚拟 Star 边。Graph Transformer
固定为：

```text
137 维 MIPS atom input
独立 backbone embedding
6 layers
hidden dim 512
8 heads
FFN 2048
0/1/2-hop localized attention
head-dim scaling
post-norm + ReLU
atom mean pooling
```

第 `h` 个 head 上每条 attention edge 的分数为：

$$
\ell_e^h=
\frac{Q_{\rm target}^hK_{\rm source}^h}{\sqrt{64}}
+b_{\rm SPD}^h
+b_{\rm path}^h
+b_{\rm star}^h.
$$

softmax 按 target 的全部 incoming edges 归一化。SPD bias 和确定性的
single-shortest-path-node bias 均为 per-head bias。

O8 的 137 维原子特征不是直接读取有限重复图端点的隐式 H 状态，而是在
topology-only 开放 Trimer 的中央 RU 上计算，再按 canonical atom 广播到
全部 O8 copies。因此虚拟 Star 边端点具有正确的聚合物化 degree、隐式 H
和 hybridization 语义。

### 2. Trimer 身份映射

独立的几何分子固定为：

```text
RU(-1) – RU(0) – RU(+1)
```

它是开放有限链，包含两条真实 inter-RU 键，不包含首尾闭合边或 PBC。

Trimer 原子身份为：

```text
(canonical_atom_id, ru_offset=-1|0|+1)
```

唯一映射规则为：

```text
(canonical atom a, 任意 O8 copy)
→ (canonical atom a, Trimer 中央 RU)
```

禁止将 O8 的 `mips_copy_id` 解释成 Trimer 的 `-1/0/+1`。外侧两个 RU
只提供中央 RU 的局部三维环境。Star 虚拟边不会成为 Trimer 的真实键。

### 3. Trimer 构象

固定构象流程：

```text
canonical P-SMILES
→ 构造开放 RU(-1)-RU(0)-RU(+1)
→ 重原子数超过 384 时直接生成带明确标记的 RDKit 2D 坐标
→ 其余样本使用 ETKDGv3 生成 4 个候选
→ sample hash 派生确定性 seed
→ useRandomCoords=False、最大 42 次 ETKDG 迭代、单次限时 60 秒
→ 仅在快速 embedding 全部失败时，以独立确定性 seed 使用
  2 个候选、useRandomCoords=True、最大 200 次迭代重试
→ 对每个候选执行 MMFF94 固定 200 步松弛
→ 每个候选执行固定 200 步 MMFF94 松弛，不读取收敛状态
→ 仅保留坐标和最终 MMFF 能量均有限的候选，选择最低有限能量
→ RemoveHs，保留重原子坐标
```

不使用 UFF fallback。Trimer 超过 384 个重原子时不执行 ETKDG/MMFF，
其 2D 坐标和映射只作为诊断信息保留，并设置
`trimer_2d_fallback=True`、`trimer_geometry_is_3d=False` 和
`trimer_geometry_valid=False`；不得进入 MCL，Star 3D RBF 也必须关闭。
MMFF 参数不足、定向重试后仍无有效候选或映射校验失败时：

```text
trimer_geometry_valid=False
```

样本继续保留，并精确退回 O8 路径。

### 4. 对称 Star-edge 三维距离

从开放 Trimer 的两条真实 inter-RU 键计算：

$$
d_{\rm left}=\|r_{R,-1}-r_{L,0}\|,
\qquad
d_{\rm right}=\|r_{R,0}-r_{L,+1}\|,
$$

$$
d_{\rm star}=
\frac{d_{\rm left}+d_{\rm right}}{2}.
$$

Star 距离质量门：

```text
Trimer 几何有效
两条 inter-RU 键存在
两条键的 bond type 正确且一致
距离 finite
|d_left-d_right| ≤ 0.15 Å
```

通过质量门后，两条相反方向的直接 Star attention edges 共用同一个
`d_star`。模型不接收左右方向、signed shift 或 image 信息。

`d_star` 使用 32 个、范围 0–3 Å 的 Gaussian RBF，再投影为 8-head bias。
投影层零初始化。非直接 Star edges 的该项严格为零。Star 距离无效时只关闭
`b_star`，不影响 Trimer-MCL 的独立有效性。

### 5. 两层 Trimer SCAGE-MCL

O8 六层结束后，先将重复节点聚合为 canonical atom 表示：

$$
\bar{x}_a=
\operatorname{Mean}_{c}x_{a,c}^{\rm O8}.
$$

随后按 canonical atom 将表示 lift 到三个 Trimer copies。每层 MCL 固定为：

- query：中央 RU 原子；
- key/value：完整 Trimer 的全部重原子；
- 使用完整 Trimer 重原子距离分布的 20% 和 50% 分位数作为 hard mask；
- 两个尺度共享 Q/K/V；
- 每个尺度独立执行 8-head masked attention；
- 两尺度结果拼接并投影回 512 维；
- 使用 residual、LayerNorm 和 GELU FFN。

两层具有独立参数。第二层开始前，将第一层更新后的中央 RU token 写回中央
memory；外侧 RU 继续使用 canonical O8 token。只保留第二层的中央 RU
更新，通过零初始化的 512 维 geometry gate 注入：

$$
x_{a,c}^{\rm final}
=x_{a,c}^{\rm O8}
+m_{\rm geo}\tanh(\gamma_{\rm geo})
\odot\Delta_a^{\rm geo}.
$$

同一 canonical atom 的全部 O8 copies 接收完全相同的几何残差。坐标只影响
距离阈值 mask，不进入可学习 embedding，因此刚体平移和旋转不会改变输出。

### 6. Graph-level MD200

对原始 RU 执行 `source_star_sub`：

```text
两个 * 分别替换为对侧 attachment 邻接原子的元素
→ sanitize
→ RDKit2DNormalized
→ MD200
```

MD200 是纯二维描述符，不执行 EmbedMolecule、MMFF 或 UFF。原子 mean
pooling 后加入低容量图级残差：

```text
LayerNorm(200)
→ Linear(200,64)
→ GELU
→ Dropout(0.10)
→ Linear(64,512,bias=False)
→ zero-gated graph residual
```

AtomPair3D512 及旧 descriptor node cross-attention 已从生产路线移除。
MD200 无效时精确退回 O8 + Trimer。

## 训练流程

### Stage 1：MTS Joint Pretraining（`mts_joint_pretraining`）

使用 GPU `0,1,2` 的三卡 DDP，在一次共享 forward 中训练：

$$
L=
1.0L_{\rm masked\ atom}
+0.25L_{\rm Trimer\ bond\ angle}.
$$

同一 canonical atom 的全部 O8 copies 同时 mask。Trimer角度只由中央 RU
中心原子和真实化学键构造，Star虚拟边不产生目标；geometry-invalid样本
不进入角度损失但仍参与 masked atom。SPD/path bias仍作为O8结构输入，
不再设置SPD/path预测头。MD200和回归头不参与预训练。

### Stage 2：MTS Property Fine-tuning（`mts_property_finetune`）

直接加载Stage 1联合预训练checkpoint：

- 只迁移 O8、Star-RBF、Trimer-MCL 和 geometry gate；MD200、Graph
  projection/norm 和回归头按 fold seed 重新初始化；
- 固定使用 `legacy_mts_huber_v1`，Graph 全部模块从 epoch 0 训练；
- 8 个任务使用统一超参数和共享的 5-fold manifests；
- Huber(beta=0.5)、batch size 32、100 epochs、patience 10；
- `TARGET_TRANSFORM=recommended`；
- Graph LR=1e-5，Graph projection/head LR=1e-4，weight decay 0.02；
- 前 5 epochs warmup，单次全程 cosine decay，gradient clipping 1.0，
  head dropout 0.25，SWA 关闭；不执行 Phase A/B/C 冻结或阶段 scheduler
  重启。

预训练只暴露 GPU `0,1,2`。下游将独立 task 作业分配到三个
单 GPU 动态队列。先运行 seed 42 的 40 folds；宏平均相对当前基线至少
提高 0.003、至少 5/8 任务不下降且最差任务下降不超过 0.015 后，才运行
seeds 43/44。

正式结果写入 `results/mts_finetune_v2`。三个 seed 必须先在每个
`(task, fold)` 的原始标签预测上取平均，再计算 R²、MAE、RMSE；任务统计
使用五折样本标准差 `ddof=1`，统一输出三位小数的 `mean ± std`。宏 R²
先按 fold 对八个任务求平均，再对五个 fold 求 `mean ± std`。由于任务单位
不同，不汇总 macro MAE/RMSE。共享 validation/test fold 结果明确标记为
非独立盲测。

## SOTA v3 实验流程

冻结的 PI1M_v2 RU/Topology/Trimer/MD200 不变。新增内容全部作为只读
sidecar 或模型参数存在：

```text
G：当前 MCL / disabled / shuffled / continuous-RBF
→ V：Graph + SMILES / CountFP / 两者
→ MT：单任务与 leakage-safe PCGrad 多任务
→ FINAL：几何/模态/多任务三个候选分别执行 outer 5-fold + inner 10% validation
```

F 阶段及其 MSE/Huber 筛选已判定为退化实验并从活动矩阵删除；历史 F
结果只保留在独立目录中用于审计，不得参与 promotion、父模型选择或正式汇总。

MCL-v2 在原 20%/50% hard mask 内加入 64 个、0–8 Å Gaussian RBF 的
per-head bias。projection 为零初始化，因而加载当前联合预训练 checkpoint
时与原 MCL 逐元素一致。坐标负对照会破坏 MCL atom-coordinate 对应，并
关闭无法同步置换的缓存 Star scalar，避免残留真实 3D 信息。

SMILES 使用 canonical P-SMILES 的两个 attachment-rooted view，token cap
为 256，两个 CLS 表示取平均；V 阶段只训练 rank 8、alpha 16、dropout
0.05 的最后四层 attention LoRA。CountFP 为 2570 维 attachment-aware
count fingerprint。两者都通过零初始化 scalar gate 加到 MTS Graph anchor，
Graph 永不 modality-dropout，SMILES/FP dropout 分别为 0.10/0.15。

实验 sidecar 使用文件锁和原子写入，三张 GPU 同时启动同任务的 fold 时只
允许一个构建者。SMILES/FP 各自保存 validity mask；无效输入的残差权重严格
为零。所有实验仍复用同一联合预训练 Graph checkpoint。

多任务 PCGrad 对每个目标 `(task, fold)` 先从其余七个任务删除所有目标
outer-test sample key，再分别拟合训练折 TargetScaler。采样器使八个任务具有
相同期望样本数，task-specific head 计算各自损失，PCGrad 仅投影共享参数的
冲突梯度。

执行器：

```bash
# 后续从 G 开始依次使用 G、V、MT；FINAL 必须在前三阶段完成晋级后执行
/root/anaconda3/envs/Uni-Poly/bin/python scripts/run_mts_sota_campaign.py screen --phase G
/root/anaconda3/envs/Uni-Poly/bin/python scripts/run_mts_sota_campaign.py promote --phase G
```

`promotion_state.json` 保存实际胜者；V、MT、FINAL 配置会在运行前物化
前序胜出的 loss、geometry、modalities 和 finetune mode，避免运行时隐式
overlay。FINAL 分别生成 `FINAL_geometry_nested5`、
`FINAL_modality_nested5` 和 `FINAL_multitask_nested5`；最多三个候选进入
inner-validation 加权集成。

Angle-v2 是条件分支：

```bash
python scripts/mts.py build-angle-v2
MTS_ANGLE_OBJECTIVE=cosine MTS_ANGLE_WEIGHT=0.25 \
BASELINE=MIPS-Trimer-SCAGE PRETRAIN_ONLY=1 bash scripts/run.sh
```

它使用固定 sample-hash 1% validation，并每 2000 optimizer steps保存里程碑；
最终 export 选择 normalized masked-atom CE 与 angle cosine MAE 之和最低的
checkpoint，而不是无条件使用最后一步。

兼容轨输出 `historical_summary.csv`，严格轨输出 `nested_summary.csv`；原始
prediction shard 保存完整精度，正式表格显示五折 `mean ± sample std` 三位
小数。最终最多三个候选可用 `scripts/ensemble_mts_candidates.py` 按 inner
validation RMSE 加权，outer-test 指标不得参与选权。

## 配置与缓存

唯一生产配置：

[configs/mts/default.json](configs/mts/default.json)

```text
config schema     = mts-config-v2
feature schema    = mips-trimer-scage-feature-v4
Trimer cache      = mips-trimer-scage-trimer-v5
Trimer LMDB       = mips-trimer-scage-trimer-lmdb-v3
cache layout      = mips-trimer-scage-lmdb-layout-v2
checkpoint schema = mts-model-v2
```

缓存采用：

```text
canonical P-SMILES
→ SHA256 sample key
→ ru_base LMDB
   ├─ O8 topology LMDB
   └─ finite-Trimer LMDB
→ source_star_sub MD200内容缓存
→ 按cohort顺序物化为md200.npy mmap
```

`ru_base`保存Topology与Trimer共享的RU解析、attachment和canonical
identity。构建器逐样本计算缺失layer：只缺Trimer时复用现有O8映射，只缺
MD200时不运行O8或Trimer。PI1M_v2 是唯一预训练 cohort；下游 union 仅提前
稳定 Stage 3 的特征层，不参与 Stage 1/2 采样。内容键仍支持已有样本复用。

两个不同 dummy attachment site 可以共享同一个 canonical boundary atom。
构图器复制开放 RU 链，并在每个候选重复数上用两端实际 copy 节点重新计算
最短路；以 `1,2,4,8...` 找上界后搜索满足 `boundary distance > 5` 的
最小整数。Star 虚拟边只连接首尾两个不同 copy，不写入 RDKit 有限环。
attachment 键型一致时保留原键型，芳香 attachment 转为单键；两端键型不
一致时统一使用单键并保留完整诊断。

缓存依赖哈希为：

```text
ru_base_hash
→ topology_hash(ru_base_hash)
→ trimer_hash(ru_base_hash, topology_hash)
```

预期的单样本解析、attachment、原子上限、ETKDG/MMFF 或超时失败只写
unavailable record，不终止 cohort。非交互日志每5秒汇总吞吐、ETA、
Graph/Trimer有效数、失败原因与worker重启；Ctrl-C会先提交父进程已收到的
完整记录，再由resume只补缺失key。

联合预训练阶段读取冻结的`ru_base,topology,trimer`和角度派生缓存，目标为
`1.0*masked_atom + 0.25*trimer_bond_angle`；MD200仅在 Stage 2 属性微调时
加载。Graph-only初始化在tokenizer之前分流，不加载PubChem tokenizer，不扫描
最大token长度，也不生成SMILES/FP字段。

48个CPU worker只生成特征，父进程是唯一LMDB writer。每128条或30秒提交
一次事务；中断恢复最多重算一个未提交事务。Trimer失败写入正式无效记录，
样本继续使用O8。

当前缓存完成后的收尾分为两个命令：

```bash
python scripts/mts.py prepare-downstream
python scripts/mts.py finalize-cache --validate-full --audit-10k --freeze
```

该命令通过三项简化质量门后才写入 `validation/store.json` 和各层 `.frozen`；
它不会自动启动预训练。预训练只能在冻结后的完整 PI1M_v2 缓存上启动。

生产入口还提供只读准备检查和阶段分发：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/mts.py doctor
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/mts.py benchmark-pretrain --batches 500
python scripts/mts.py prepare-downstream
python scripts/mts.py finalize-cache --validate-full --audit-10k --freeze
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/mts.py pretrain
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/mts.py finetune
python scripts/mts.py summarize
```

`doctor` 不构建缓存；它只检查三张 GPU、固定模型契约、5-fold manifest
和冻结的 cache bundle。缓存 writer 活动时会报告未就绪并退出。

旧MIPS、旧SCAGE、PyTorch shard、多模态或PBC cache/checkpoint均不兼容，
正式训练前必须重建feature cache。

旧 builder 9 的分层 500 条 correctness 结果仅作历史参考；当前生产协议为
builder 10 的 `relax200 + lowest finite`，必须以新 writer 完成后的 validation
为准。以下数字保留作协议迁移前的历史记录：

```text
Graph valid                 474 / 500
shared-boundary Graph valid  74 / 75
（剩余1条展开为392原子，超过384上限）
bond-mismatch Graph valid    78 / 78
Trimer geometry valid       102 / 500
MMFF-eligible valid         102 / 436 = 23.39%
mapping/internal invariant failure = 0
```

该历史结果采用了已废弃的“必须完全收敛”判据，不能与当前有限坐标/能量
判据混用。新 Trimer writer 完成后执行一次只读 validation 和固定 10k
性能审计，再决定是否进入训练。
