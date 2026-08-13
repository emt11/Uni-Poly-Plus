# MIPS-Trimer-SCAGE（MTS）当前模型与数据流程

## 1. 路线定位

当前唯一生产路线名称为：

```text
MIPS-Trimer-SCAGE
简称：MTS
内部标识：mips_trimer_scage
```

默认生产模型是单 Graph 模态：

```text
P-SMILES
→ canonical 单 RU 周期拓扑
→ T1 MSTA（前四层 O8、最后两层 MSTA）MIPS Graph Transformer
→ Trimer Star-RBF + SCAGE-MCL
→ canonical atom mean pooling
→ MD200低容量图级残差
→ 512→256 Graph projection
→ regression head
→ 聚合物属性
```

默认生产模型不包含 SMILES encoder、FP encoder、AP3D512、PBC、PaiNN、FLAT4 或多模态 attention fusion。`mts-experiment-v3` 可以在后续消融中加入 SMILES/CountFP 的零门控残差，但它们不是本文件描述的默认生产主干。

---

## 2. 原 MIPS 为什么显式构造多 RU

原 MIPS 的局部注意力只允许 0/1/2-hop。对于较短的重复单元，如果直接在单 RU 上首尾 Star-Linking，边界附近的两跳邻域可能绕过周期切口后过早相遇，使有限图不能无歧义地模拟无限重复链。

旧实现因此执行：

```text
原始RU
→ 构造开放的 k-RU 链
→ 找到满足 boundary distance > 2*d_threshold-1 的最小k
→ d_threshold=3，因此要求 boundary distance >5
→ 在开放链两端加入虚拟Star edge
→ 对全部k个RU的原子分别执行0/1/2-hop注意力
```

例如短 RU 可能需要3个或更多副本。旧图中的节点身份为：

```text
(canonical_atom_id, ru_copy_id)
```

这种实现可以显式展开局部周期邻域，但也带来四个问题：

1. 同一化学原子被存储和更新多次；
2. 节点数、边数和显存随 `k` 增长；
3. 图表示可能受到重复次数和周期切口影响；
4. 可变数量的 O8 copies 与固定3-RU构象之间需要复杂的多对一映射。

---

## 3. 当前单 RU 替换多 RU 的核心方案

当前方案不是简单地“只保留一个 RU 的内部键”。它将无限链中的周期副本转化为 relation row，只保留一个 canonical RU 的节点状态。

### 3.1 节点空间

设规范化 RU 含 `N` 个重原子。生产图只保存：

```text
x_a,  a=0,...,N-1
```

每个原子只存在一个可学习状态，不再保存 `ru_copy_index`，也不再物化 `kN` 个 O8 节点。

原子输入仍是 MIPS 的137维特征，来自开放3-RU中中央RU的 polymerized chemical environment，从而正确反映聚合连接后的 degree、H 数、hybridization 等信息。Backbone role 不覆盖137维输入，而是通过独立的二值 embedding 加到节点表示：

```text
h_a^0 = Linear_137→512(MIPS137_a)
        + Embedding_backbone(backbone_a)
```

### 3.2 无限 lifted 状态空间

逻辑上仍考虑无限链状态：

```text
(a,q)

a：canonical RU原子编号
q：相对RU平移编号，..., -1, 0, +1, ...
```

无限链邻接定义为：

```text
RU内部键：      (a,q) ↔ (b,q)
聚合连接键：    (right,q) ↔ (left,q+1)
```

对于每个目标原子 `(t,0)`，在 lifted 状态空间中执行最大2-hop BFS。每个可见源状态 `(s,q)`生成一条 relation row：

```text
source canonical atom = s
target canonical atom = t
relative_ru_shift     = q
SPD                   = 0/1/2
single path atoms
single path shifts
direct polymer-link mask
```

`relative_ru_shift` 和 path shift 只用于构图、校验和诊断，不进入 embedding 表，也不向模型提供方向信息。

### 3.3 为什么必须保留 relation multiplicity

两个 lifted source 可能具有相同 canonical atom ID，但来自不同相对RU：

```text
(a,-1) 和 (a,+1)
```

它们在 `lga_edge_index` 中可能表现为相同的 `a→t`，但仍必须保留为两条独立 relation row。注意力的 incoming softmax 对目标 `t` 的全部 relation rows 归一化，不能先按 canonical source/target 去重。

因此当前表示为：

```text
一个canonical节点状态
+ 多条携带周期来源的relation row
```

而不是：

```text
一个RU内部图
+ 一条普通首尾边
```

共享 attachment boundary 时，左右边界可以是同一个 canonical 原子；此时 `shift=±1` 的非零平移 self relation 必须保留，不能误删成普通 self-loop。

### 3.4 单 RU 如何执行原多 RU 的消息更新

对一条 lifted relation `e=((s,q)→(t,0))`，模型读取同一个 canonical source state `h_s` 作为 key/value，读取 `h_t` 作为 query：

$$
\ell_{e,h}
=\frac{Q_h(h_t)^\top K_h(h_s)}{\sqrt{64}}
+b^h_{\rm SPD}(d_e)
+b^h_{\rm path}(p_e)
+b^h_{\rm star}(e).
$$

然后对同一 target 的所有入边执行 softmax：

$$
\alpha_{e,h}
=\operatorname{softmax}_{e:\,target(e)=t}(\ell_{e,h}),
$$

并聚合 `V_h(h_s)`。

不同 `q` 的 relation 可以复用同一个 `h_s`，但仍分别参与 softmax 和消息求和。这相当于强制所有平移等价RU副本共享状态，而不是为每个副本维护独立张量。

### 3.5 与修正后显式多 RU 对照的关系

在以下条件下，canonical 表示与足够长的显式重复链具有相同的确定性局部计算语义：

- 所有RU副本初始化学特征相同；
- 不使用 `ru_copy_id`、方向或绝对位置 embedding；
- 局部关系和 multiplicity 完整保留；
- readout按一个周期单元定义；
- dropout关闭或使用平移绑定的随机掩码。

因此等价性测试在 `eval()`、dropout关闭时比较1层/6层节点状态、Star-RBF、MCL、MD200、pooling和最终预测。训练态若旧显式copies各自采样独立dropout，它与canonical共享状态不会具有逐随机数完全相同的轨迹；这是移除冗余copies后的预期差异，不是构图错误。

当前项目同时保留可运行的 `explicit_k_ru` 对照：它物化满足
`boundary distance >5` 的最小合法 `k`，但不使用 copy/方向/shift
embedding，并在 readout 前按 `canonical_atom_id` 聚合全部 copies。
它使用独立 feature schema、Topology LMDB、model hash 和 checkpoint
身份，不能与 canonical checkpoint 交叉 resume。canonical 仍是默认且唯一
正式 20k 预训练表示；explicit 只用于等价性验证和后续受控对照。

---

## 4. O8 MIPS Graph Transformer

固定结构：

```text
节点输入       MIPS137 + independent backbone embedding
隐藏维度       512
层数           6
heads          8
head dimension 64
FFN            512→2048→512
locality       0/1/2-hop
dropout        0.10
norm           post-norm
activation     ReLU
readout        canonical atom mean
```

每层使用：

```text
K_source · Q_target / sqrt(head_dim)
+ per-head SPD bias
+ per-head single-path-node bias
+ direct Star-edge RBF bias
→ target incoming-edge softmax
→ V_source aggregation
→ residual + LayerNorm
→ ReLU FFN
→ residual + LayerNorm
```

SPD 只有3类 `0/1/2`。Path bias 从 forward 开始时的初始节点表示计算，使用单条确定性最短路径的节点表示；它不是全最短路径DAG，也不是path-bond预测分支。

---

## 5. Trimer 3D 信息

### 5.1 Trimer 定义

3D 几何使用开放的 finite Trimer：

```text
RU(-1) — RU(0) — RU(+1)
```

它包含两条真实 inter-RU 化学键，不包含首尾闭合边，也不包含 O8 的虚拟 Star edge。Trimer 是均聚物局部内部链段的3D代理，不宣称为无限聚合物的唯一平衡构象或周期晶胞。

### 5.2 构象协议

默认协议：

```text
ETKDGv3生成4个候选
→ 默认useRandomCoords=false，maxIterations=42
→ 首次失败时定向重试2个候选：useRandomCoords=true，maxIterations=200
→ MMFF94固定松弛200步
→ 保留坐标和MMFF能量均finite的候选
→ 选择松弛后最低finite能量候选
```

不要求 MMFF 状态完全收敛。Trimer 超过384个重原子时只允许生成2D诊断字段，MCL和Star-3D必须关闭；2D坐标绝不能进入模型几何分支。

### 5.3 Canonical RU 与 Trimer 的严格映射

三个原子空间保持独立：

```text
源explicit Topology原子空间
normalized Trimer原子空间
新canonical RU原子空间
```

不能依赖 canonical SMILES 的原子顺序，也不能把旧 Topology 的映射直接复用于 normalized Trimer。迁移使用带标签图同构，核对：

```text
atomic number
formal charge
aromatic flag
chiral tag
attachment role
internal degree
真实键与bond type
```

canonical RU含 `N` 个原子时，Trimer固定为：

```text
Trimer atom count = 3N
base atom ids      = [0..N-1]重复3次
RU offsets         = -1, 0, +1
central RU indices = [N,2N)
canonical atom a   → central Trimer atom N+a
```

当前 canonical O8 本身只有 `N` 个节点，所以不再执行旧流程中的：

```text
多copy scatter-mean
→ 几何residual广播回全部copies
```

而是直接以 canonical state 初始化三个 Trimer copy的token，并把中央RU更新直接写回同一个 canonical atom。

### 5.4 对称 Star-edge 距离

从Trimer的两条真实inter-RU键计算：

$$
d_{left}=\|r_{R,-1}-r_{L,0}\|,
\qquad
d_{right}=\|r_{R,0}-r_{L,+1}\|,
$$

$$
d_{star}=\frac{d_{left}+d_{right}}{2}.
$$

当坐标finite且：

$$
|d_{left}-d_{right}|\le 0.15\ \text{Å}
$$

时，`star_3d_valid=True`。`d_star`经过32个、范围0–3 Å的Gaussian RBF，再投影成8个head的bias。投影零初始化，且只加到两条直接polymer-link relation；两个方向使用同一个对称标量。

Star-distance是否有效与MCL是否有效相互独立：Star失败只关闭Star-RBF，不应关闭仍然合法的MCL。

### 5.5 两层 SCAGE-MCL

O8六层结束后得到 canonical topology states：

$$
H^{topo}\in\mathbb{R}^{N\times512}.
$$

根据 `trimer_base_ru_atom_id`，同一个 `h_a` lift到：

```text
(a,-1), (a,0), (a,+1)
```

MCL固定为两层：

- query：中央RU的 `N` 个原子；
- key/value：完整Trimer的 `3N` 个原子；
- 根据完整Trimer距离分布的20%和50%分位阈值构造两种hard visibility mask；
- 同一层的两个尺度共享Q/K/V；
- 两尺度输出拼接后投影回512维；
- 每层执行attention residual、LayerNorm、GELU FFN和LayerNorm；
- 第二层使用第一层更新后的中央token，外侧RU继续提供canonical topology memory。

默认生产 `current_mcl` 中，欧氏距离只决定“哪些原子可见”，不作为连续attention bias。`mcl_rbf` 是独立几何消融，不属于默认生产模型。

中央RU的几何增量为：

$$
\Delta h_a^{geo}=h_{a,0}^{MCL}-h_a^{topo}.
$$

通过零初始化的512维gate注入：

$$
h_a^{final}=h_a^{topo}
+\tanh(\gamma_{geo})\odot\Delta h_a^{geo}.
$$

### 5.6 几何有效性与精确回退

MCL只接受同时满足以下条件的样本：

```text
graph_available
trimer_geometry_valid
trimer_geometry_is_3d
not trimer_2d_fallback
坐标全部finite
mapping长度等于canonical节点数
mapping索引合法且全部指向central RU
```

任一条件失败时：

```text
geometry residual = 精确零
模型退回纯O8 topology
样本仍保留，不改变fold或预训练cohort
```

---

## 6. 图级表示、MD200与回归头

MCL更新后，对一个canonical RU的 `N` 个节点做平均：

$$
g_{topo+3D}=\frac{1}{N}\sum_{a=1}^{N}h_a^{final}.
$$

MD200通过 `source_star_sub` 生成：

```text
原始P-SMILES RU
→ 两个*分别替换为对侧attachment邻接原子的元素
→ canonical SMILES
→ RDKit2DNormalized 200维
```

MD200是纯2D图级描述符，不执行ETKDG、MMFF，也不包含AtomPair3D512。其低容量残差为：

```text
LayerNorm(200)
→ Linear 200→64
→ GELU
→ Dropout 0.10
→ Linear 64→512，无bias
→ zero-init scalar gate
```

$$
g_{MTS}=g_{topo+3D}
+m_{MD}\tanh(\gamma_{MD})P_{MD}(MD200).
$$

MD200无效时残差精确为零。

### 5.7 T0/T1 拓扑注意力身份

新训练生产默认已晋级为 T1（`topology_attention_variant=msta_last2`）：前四层保持
O8，最后两层
使用两个独立 incoming-edge softmax 分支：

```text
Z1: SPD ∈ {0, 1}
Z2: SPD ∈ {0, 1, 2}
output = old_output(Z2) + local_output(Z1)
```

两分支共享 Q/K/V、拓扑/path/Star bias 和 relation/head dropout mask；canonical
relation row 不按 `(source, target)` 去重，SPD=0 自关系同时保留在两支。每个
T1 层的显式关系支持仍是 `SPD ≤ 2`；六层堆叠后的有效传播范围可能超过两跳。
`local_output` 是无 bias 的 `512→512` 线性层并零初始化，因此只能通过显式
初始化器从 T0 warm start，不能把 T0 checkpoint 当作普通 T1 resume。

T1 配置和入口：

```text
configs/mts/experiments/T1_msta_readiness.json
scripts/initialize_mts_t1.py
```

初始化产物必须写入新的 `pretrained_models/mts_multiscale_topology/t1_init/`
目录，并记录 `parent_checkpoint`、`initialization=function_preserving`、
`source_model_identity=T0` 和 `model_identity=T1`。T1 smoke/benchmark 证据位于
`results/mts_multiscale_topology/t1_readiness/`，均标记为 screening-only；它们
不改变冻结 cache、`best_result.csv`、历史正式 checkpoint，也不构成 T1 优于 T0
的科学结论。

历史 T0（`topology_attention_variant=o8`）仍保留为显式对照配置；新训练不能隐式
回退到 T0。现有正式 T1 checkpoint 可作为普通 T1 下游微调来源，但 G-family
必须从独立 shared step-0 开始，不能从 T1 20k checkpoint 分叉。

T1 初始化产物用于生产微调时必须显式 opt-in，且仍按架构 warm start 处理：

```bash
MTS_ALLOW_T1_FUNCTION_PRESERVING_INIT=1 \
JOINT_CKPT=pretrained_models/mts_multiscale_topology/t1_init/mts_t1_function_preserving_init.pth \
```

launcher 将该环境变量转换为
`--allow_mts_t1_function_preserving_init`，仅接受 `init_artifact=true`、
`initialization=function_preserving`、T0 parent `source_optimizer_steps=20000`、
T1 `optimizer_steps=0`、parent SHA/source contract、当前 T1 graph hash 和零
`local_output` 权重全部一致的 checkpoint。未显式 opt-in、普通 T0/T1 resume 或
将该 init 传给 `pretrain.py --resume_state` 均会拒绝；该特例不会继承 optimizer、
scheduler 或 sampler 状态。

### 5.8 G-family readiness identity

G0/G1/G2/G3 统一使用 T1 拓扑、MD200，关闭旧 Star-RBF 与 full-Trimer MCL；G1
只读冻结 relation-geometry sidecar 的 path cosine，G2/G3 额外读取 endpoint distance，
G3 使用独立的 seed-42 条件分层置乱 artifact。四臂共享同一 T1 common step-0，
G0 geometry residual 恒为零且不进 optimizer；invalid relation/path 精确回退为零。
本周期仅做 sidecar/collate、forward/backward、DDP 和 2-epoch load/train smoke，
不启动正式 20k 或 8×5。

本修复周期的生产入口 smoke 证据独立写入
`results/mts_multiscale_topology/t1_repair/`，不得覆盖上一周期
`t1_readiness/` 证据。

随后：

```text
Graph LayerNorm
→ Linear 512→256
→ LayerNorm
→ ReLU
→ regression head:
   256→128→64→1
   GELU + dropout 0.25
```

默认 `fusion_type=none`，因此256维Graph embedding直接进入回归头，不存在额外模态权重。

---

## 7. 完整前向数据流

```text
P-SMILES
│
├─ normalized canonical RU
│  ├─ N个MIPS137原子特征
│  ├─ backbone embedding
│  └─ infinite lifted 0/1/2-hop relation rows
│        ├─ SPD bias
│        ├─ single-path-node bias
│        └─ direct Star relation mask
│                    │
│                    ▼
│             6-layer O8 Transformer
│                    │
│                    ├───────────────┐
│                    │               │
├─ open finite Trimer                │
│  ├─ canonical a ↔ central N+a      │
│  ├─ symmetric d_star → Star-RBF ───┘
│  └─ 3D distance masks
│         → two-layer central-query/full-Trimer MCL
│         → zero-gated canonical atom residual
│                    │
│                    ▼
│          canonical atom mean pooling
│                    │
├─ source_star_sub MD200
│         → zero-gated graph residual
│                    │
│                    ▼
│              Graph 512→256
│                    │
│                    ▼
│            256→128→64→1 head
│                    │
└────────────────────┴→ property prediction
```

---

## 8. 预训练与微调

### 8.1 MTS Joint Pretraining

预训练只使用完整 `PI1M_v2`，单阶段、单次共享 O8+Star+MCL forward：

$$
L_{pretrain}=1.0L_{masked\ atom}
+0.25L_{Trimer\ bond\ angle}.
$$

Masked atom：

- mask ratio 30%；
- 遮蔽完整137维化学特征，保留backbone role；
- canonical图只有一个atom state，不存在copy泄漏；
- 预测101类MIPS原子类别。

Trimer bond-angle：

- 中心原子必须属于中央RU；
- 邻居来自Trimer真实键，可位于相邻RU；
- Star虚拟边不产生角度；
- 角度分为20类，每类9°；
- 三原子表示为 `h_i+h_j+h_k`；
- 头为 `512→256→20`，LayerNorm、ReLU、dropout 0.10；
- 使用 `gamma=2` 的类别加权focal loss；
- 每个polymer先对自身角度求平均，再跨polymer平均。

MD200不参与预训练。固定训练参数：

```text
optimizer steps  20,000
global batch     1008
per-rank batch   336
accumulation     1
DataLoader       workers=6, prefetch=2（每个rank）
GPU              1,2,3
Adam betas       (0.9,0.98)
peak LR          2e-4
warmup           2,000 steps
scheduler        polynomial decay，power=1
end LR           1e-9
weight decay     0
AMP              BF16
seed             42
```

### 8.2 MTS Property Fine-tuning

正式8任务：

```text
eat eea egb egc ei eps nc xc
```

当前固定 `legacy_mts_huber_v1`：

```text
全部Graph wrapper从epoch 0训练
Graph wrapper LR  1e-5
regression head LR 1e-4
AdamW weight decay 0.02
Huber beta         0.5
batch              32
epochs             100
patience           10
warmup             5 epochs
scheduler          单次全程cosine
gradient clip      1.0
head dropout       0.25
SWA                disabled
target transform   recommended
```

本机有限速度 smoke（4 GPU、四任务、fold0、2 epochs）已验证微调默认：

```text
train batch        32
DataLoader workers 2 / slot
prefetch           2
eval batch         64
AMP                FP32
GPU slots          0,1,2,3
```

该选择来自 `results/mts_speed_optimization/finetune/worker_eval_amp_20260811/benchmark.json`：
workers=2 在“最高吞吐 2% 内优先较少 worker”规则下胜出；eval batch 64/128/256
均在同一 fold-best 模型状态上通过 `y_true` 精确一致和预测
`allclose(rtol=0, atol=1e-5)`，四任务 eval 总时间分别为
`3.024543/3.773683/3.871604s`，因此按最快安全候选选择 64。这是有限速度 smoke，不是正式模型质量或 8×5
训练结论。修复 `mips_md` residual 的 BF16/Float indexed-assignment 后，BF16
finite/loss parity gate 已通过，但聚合吞吐仅为 FP32 的 `1.019x`，低于 10%
晋级门，因此仍保留 FP32。

`recommended` 对 eps/nc 使用 `log + 训练折标准化`，其余任务使用训练折标准化；指标在逆变换后的原始标签空间计算。

历史比较继续使用共享 validation/test 的固定5折，必须标记：

```text
fold_validation_protocol=shared_validation_test_fold
independent_blind_test=false
```

---

## 9. Cache、schema与生产状态

缓存按层独立：

```text
RU base
Topology LMDB
Trimer LMDB
MD200 LMDB + cohort mmap
Angle sidecar
MCL-threshold sidecar
Cohort manifest
```

当前统一契约：

```text
config                  mts-config-v3
feature                 mts-canonical-periodic-feature-v3
Topology LMDB           mts-canonical-periodic-topology-lmdb-v3
Trimer content          mips-trimer-scage-trimer-v8
Trimer LMDB             mips-trimer-scage-trimer-lmdb-v6
canonical LGA           version 2
Angle sidecar           mts-trimer-bond-angle-cache-v2
MCL threshold           mts-mcl-threshold-array-v2
cache bundle            mts-canonical-cache-bundle-v3
checkpoint              mts-model-v3
fresh pretrain checkpoint mts-model-v4
builder                 version 12
explicit feature        mts-explicit-kru-feature-v1
explicit Topology LMDB  mts-explicit-kru-topology-lmdb-v1
explicit LGA            version 3
```

生产目标是：

```text
PI1M_v2 ∪ downstream_union
= 999,224个unique sample keys
```

错误旧 explicit cache 永久拒载；修正后的 `explicit_k_ru` 使用全新独立
root。任何没有 `.done + .frozen + store.json` 且未通过
`final_acceptance.json` 全部硬门的 canonical root 都不能用于正式训练。

截至2026-08-09，canonical exact-union、Angle/MCL sidecar、freeze、Doctor
和20项 hard gates均已通过。当前新建的是独立 explicit Topology 对照缓存，
它不能修改 canonical 冻结产物。实时状态以以下证据为准：

```text
results/mts_canonical_migration/final_acceptance.json
scripts/mts.py doctor
```

当前双拓扑实施、缓存和正式预训练执行记录见
[CODEX_CLAUDE_HANDOFF.md](CODEX_CLAUDE_HANDOFF.md)。

---

## 10. 方案的保证与边界

### 10.1 能保证什么

- 一个化学RU原子只维护一个O8节点状态；
- 完整保留无限链2-hop局部关系、relative shift与multiplicity；
- 消除短RU重复次数对节点数和mean pooling的影响；
- canonical atom与Trimer中央RU建立一一映射；
- 邻接RU只提供局部跨RU空间环境；
- 无效3D不删除样本并精确退回O8；
- MD200、Star-RBF和MCL分别具有独立有效掩码和零门控回退。

### 10.2 不能宣称什么

- canonical topology是周期商图/无限拓扑的两跳代理，不是无限链3D坐标；
- finite Trimer是局部链段构象代理，不是唯一热力学平衡构象；
- 当前3D不表示晶胞、结晶度、自由体积或长程链缠结；
- 单构象不能覆盖完整构象系综；
- eval下的显式/canonical确定性等价不意味着copy-wise独立dropout的训练随机轨迹完全相同；
- canonical bundle 已通过生产验收；explicit 对照在其独立全量 cache、验证和
  freeze 完成前不得称为可运行全量对照。
