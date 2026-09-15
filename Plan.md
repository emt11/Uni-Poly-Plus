# GLT-V2 Training-Ready Static Cache 完整构建计划

## 1. 目标与当前判断

本计划只解决一个问题：将当前已经冻结的 GLT-V2 基础缓存补齐为训练时可直接读取的
静态派生缓存，尽量不在每次 sample visit 中重复执行 RDKit 解析、物理键分词、line graph、
Hop2 路径、BRICS 和 Morgan fingerprint 构建。

当前基础数据在科学输入意义上已经完整：

```text
RU identity
├─ Topology：2D canonical topology / lifted paths
└─ Trimer：3D 坐标 / 物理键 / 原子身份
```

目前缺少的是：

```text
Topology + Trimer
        ↓ 一次性只读派生
GLT-Dual Training Static Cache
        ↓
训练时只做轻量 tensor 索引、mask/noise 和动态距离/角度计算
```

现有 `RUNTIME_STATIC_REUSE` 已经让一个 sample visit 内的 clean/noisy 分支共享
identity 与 2D bond-path 结果，但同一个 polymer 在后续 epoch 再次出现时，仍会重新执行
大部分静态构建。5,000 updates、global batch 1008 共访问：

$$
5000 \times 1008 = 5{,}040{,}000
$$

相当于 959,588 条 PI1M cohort 约 5.25 次完整访问。持久 static cache 的目标是消除这些
跨 epoch 的重复 CPU 工作，不改变模型、任务、噪声、mask、坐标或数据划分。

## 2. 执行边界

### 2.1 本计划包含

- 新增一个 GLT-Dual 专用 static cache record、builder 和只读 reader。
- 将当前 runtime 静态构建拆成“离线静态部分”和“训练时动态几何部分”。
- 为 PI1M 959,588 条正式 cohort 构建 static cache。
- 为 downstream 3,655 个唯一结构构建同一格式的 static cache。
- 接入预训练和微调入口，并保留现有 runtime 路径作为对照与回退。
- 执行相关单元测试、1k pilot、10k pilot、全量构建后的有限抽查和短 benchmark。

### 2.2 本计划不包含

- 不重新生成 RU、Topology、Trimer 或 ETKDG/MMFF 坐标。
- 不修改现有主 bundle、accepted cohort、property row 或 outer5_inner20 split。
- 不缓存 noisy distance、noisy angle、随机 atom mask 或随机噪声。
- 不直接复用包含 clean distance/angle 的旧 `CompleteTrimerGLTSidecar`。
- 不改变 Concat/KFuse、三项预训练任务、损失权重、模型架构或数据划分。
- 不因为 static cache 建成而自动启动新的正式预训练或微调。
- 不建设通用 schema registry、迁移器、逐字段 hash 系统或多层发布 gate。

### 2.3 当前运行任务

当前正在执行的 Concat 5k 使用的是 `RUNTIME_STATIC_REUSE` 路径。它继续作为现有正式
baseline，不中途切换 reader，也不把 static cache 构建与其混在同一输出目录中。
构建阶段应避开当前 5k 的 CPU 高负载时段；任何已安排的 KFuse 5k 也遵循同一原则。
本计划本身不授权启动或停止这些训练。

## 3. 两套输入 provenance

两套数据使用同一个 static cache schema，但产物必须独立：

|用途|基础 bundle|cohort|static cache 单位|
|-|-|-|-|
|PI1M 预训练|`30f17b59bc5862a1ddae7eaee03b2767df26561d9bfecb690ec8eea3ddd09ed2`|959,588 条 accepted unique structures|每个 cohort key 一条|
|8-task 微调|`1545eda5a8f6a1a7868ce01464ce7c6dc714b4685ae10dc7213b90adfbcc23b2`|6,265 property rows / 3,655 unique structures|每个 unique key 一条，dataset row 复用|

PI1M 与 downstream 不共用 static cache 目录、manifest 或 row index。downstream 的
property row、task、label 和 fold provenance 继续保留在现有 cohort 中，不复制进结构缓存。

## 4. Static cache 内容

只缓存当前 consumer 会重复计算、且与坐标扰动无关的内容。已经存在于 Topology/Trimer
中的大数组不重复保存。

### 4.1 2D bond-path

缓存：

```text
bond_path_features   float32 [R2D, 2, 14]
bond_path_mask       bool    [R2D, 2]
bond_path_offsets    int64   [N+1]
```

`lga_edge_index`、`lga_spd`、`lga_path_index`、`lga_path_shift`、
`lga_source_image_shift` 和 `mips_x` 已在 Topology 中，继续直接读取，不在 sidecar 中复制。

### 4.2 预训练化学目标

缓存 BRICS 的两级 ragged partition：

```text
brics_sample_group_ptr   int64 [N+1]
brics_group_atom_ptr     int64 [G+1]
brics_atom_index         int32 [A]
```

每个 group 中保存 canonical base atom id。运行时仍使用现有 seed 和抽样逻辑选择 motif，
因此缓存 groups 不会固定随机 mask。

Morgan fingerprint 使用 bit packing：

```text
morgan_2048_packed       uint8 [N, 256]
```

读取后再按需 unpack 为 2048 bit target，避免把固定二值标签保存为 float32。

### 4.3 3D 物理 bond tokens

缓存：

```text
token_offsets            int64 [N+1]
token_pos_index_a        int32 [M]
token_pos_index_b        int32 [M]
token_z_a                uint8 [M]
token_z_b                uint8 [M]
token_bond_type          uint8 [M]
token_center_mask        bool  [M]
```

`token_pos_index_a/b` 直接索引冻结 Trimer 的 `trimer_pos`，不在训练时重新完成 heavy-atom
投影、物理键去重和 token ordering。只保存当前 3D encoder 实际使用的化学字段；Stereo、
Conjugation、Ring 等仍在离线构建时参与化学一致性检查，但若当前模型不消费，就不为其
重复增加 per-token 数组。

### 4.4 line graph、全部等长最短路径和角度索引

attention relation 与逐路径记录数量不同，因此分别保存 offset：

```text
line_relation_offsets    int64 [N+1]
line_source              int32 [Q]
line_target              int32 [Q]

line_path_offsets        int64 [N+1]
line_path                int32 [P, 3]
line_path_mask           bool  [P, 2]
line_path_group          int32 [P]
line_is_self             bool  [P]
angle_pos_triplet        int32 [P, 2, 3]
```

`line_path_group` 在单样本内从 0 开始，collate 时仍按 relation 数量偏移；`line_path` 按
token 数量偏移。每个 relation 只保存一次 `line_source/line_target`，等长多最短路径保留
为多个 path rows，并通过 `line_path_group` 归入同一 attention relation。

`angle_pos_triplet[path, hop] = (outer_a, center, outer_b)`，索引冻结 Trimer 坐标。
padding/self 使用 `-1` 和 mask，不缓存角度数值。

### 4.5 invalid geometry

PI1M 正式 accepted cohort 预期都有有效 Trimer geometry。downstream 的合法 geometry
fallback 仍保留完整 2D bond-path、BRICS 和 fingerprint；其 3D token/relation/path 区间
为空，并保存：

```text
geometry_valid           bool  [N]
geometry_failure_code    small integer/string table
```

不得删除对应 property row，也不得用零坐标伪造 3D 输入。

## 5. 不缓存的动态内容

以下内容每次 sample visit 必须重新生成：

```text
atom_mask
coordinate_noise
noisy_coordinates
clean bond distance target
clean angle target
noisy bond distance input
noisy angle input
```

训练时的数据流固定为：

```text
static row + frozen Topology + frozen Trimer coordinates
        │
        ├─ cached BRICS groups + deterministic RNG → atom mask
        ├─ cached token endpoint indices + clean coordinates
        │       → clean distance/angle → supervision only
        └─ clean coordinates clone + N(0, 0.03 Å)
                + cached token/angle indices
                → noisy distance/angle → 3D encoder
```

这样既消除 RDKit/图算法热路径，又不会把 clean geometry 泄漏给 noisy encoder。

## 6. 文件布局与最小元数据

建议目录：

```text
data/processed/glt_dual_v2/static/
├─ pi1m/<static-id>/
└─ downstream/<static-id>/
```

每个最终目录包含上述 `.npy` 数组、`manifest.json` 和 `.done`。manifest 只保留实际需要的：

```text
format = glt-dual-training-static-v1
parent_bundle_hash
cohort_manifest_hash
ordered_sample_key_hash
sample_count
code_commit
build_parameters
array shapes/dtypes
```

不要求每个数组重复计算独立内容 hash；父 bundle、cohort 顺序、总数、shape/dtype 和最终
marker 足以支持当前工程。reader 必须通过显式路径打开，不扫描目录自动选择“最新版本”。

## 7. 构建器设计

### 7.1 新模块和入口

建议新增：

```text
src/dataset/glt_dual_static.py
scripts/build_glt_dual_static_cache.py
scripts/benchmark_glt_dual_static_cache.py
tests/test_glt_dual_static.py
```

同时最小修改：

```text
src/dataset/glt_dual.py
src/dataset/glt_dual_pretrain.py
src/training/glt_dual_runtime.py
scripts/pretrain_glt_dual.py
scripts/finetune_glt_dual.py
PIPELINE.md
```

### 7.2 分块、并行和恢复

959,588 条不能先全部保存在 Python list 中。使用固定 source-order chunk，例如每块
8,192 或 10,000 条：

1. worker 只读打开 frozen Topology/Trimer，按 key 生成 static record；
2. 主 writer 按原始 cohort index 排序后写入 chunk 临时目录；
3. 每个完整 chunk 写一个简短完成标记；
4. 中断恢复时只跳过已完成且 count/边界 key 匹配的 chunk；
5. 所有 chunk 完成后顺序拼接为最终 mmap arrays；
6. 写 manifest 和 `.done`，最后将 staging 目录改名为正式目录。

只需要一个 writer，worker 数由 pilot 的吞吐决定。优先测试 8、16、24 中的少量组合，
不做大规模 worker sweep。所有超过一分钟或启动 worker 的运行放在 tmux session
`Uni-Poly` 的独立 window，并保存命令和日志。

### 7.3 失败处理

static derivation 是确定性操作，不需要无限重试。单条失败时记录：

```text
cohort_index
sample_key
exception_type
message
```

PI1M 或 downstream 任一应有 key 无法派生时，本次 static cache 不发布；先定位实现或
身份问题。这里不需要复杂 failure taxonomy，也不把错误样本静默删除。

## 8. Runtime 接入

### 8.1 显式入口

预训练和微调新增：

```text
--dual-static-root PATH
```

不提供该参数时保留当前 `RUNTIME_STATIC_REUSE` 路径，便于完成正在运行的 baseline 和
做 A/B 对照。提供参数后，如果 sidecar 的 parent/cohort/count 不匹配则直接报出清晰错误，
不静默退回旧 runtime。

### 8.2 Reader 行为

reader 按 cohort row hint 读取，PI1M 是一一对应；downstream 通过 sample key 映射到
3,655 个 unique rows。读取后：

- Topology 仍提供 `mips_x`、lifted relations 和 atom labels；
- Trimer 只提供坐标与基础 identity carrier；
- static row 提供 bond-path、BRICS、fingerprint、tokens、line paths 和坐标索引；
- 轻量 tensor 函数计算 clean/noisy distance 与 angle；
- collate 的 atom/token/relation/path offset 规则保持不变。

checkpoint 的 run identity 增加 static manifest identity，确保断点恢复继续使用同一输入
语义。部署包不需要携带 static cache 内容。

## 9. 分阶段执行

### Phase A：代码拆分与单元测试

目标：先在不构建持久缓存时，把旧 runtime 结果拆成 static record + dynamic geometry，
证明接口正确。

测试重点：

- 2D 14 维 bond-path 与现实现逐元素相同；
- physical token 顺序、endpoint、BondType、center mask 相同；
- 多最短路径、反向 relation、self 和 mixed batch offset 相同；
- BRICS groups 分区与 Morgan 2048 bits 相同；
- N=0 和 downstream geometry fallback 能正常表达；
- 同一 seed/key/position 的 mask、noise、clean targets、noisy inputs 不变；
- sidecar roundtrip 后不携带 clean/noisy distance 或 angle。

浮点比较只对重新向量化的距离/角度使用正常数值容差（建议 `atol=1e-6`、
`rtol=1e-6`）；离散 identity、mask、path、fingerprint 必须精确相同。

### Phase B：1k PI1M pilot

从正式 959,588 cohort 取确定性的均匀 1,000 条，构建并读取。记录：

- 1,000/1,000 coverage；
- build samples/s、单样本字节数、总量磁盘外推；
- static reader samples/s；
- 原 runtime 与 static runtime 的分项耗时；
- 40 条固定真实样本的完整 tensor 等价；
- 主 frozen bundle 未被 builder 修改。

1k 的目标是尽快发现 layout 或动态几何问题，不做模型训练。

### Phase C：10k pilot 与 worker 选择

用 10,000 条连续/均匀混合样本验证真实 ragged 分布和构建吞吐。只比较少量 worker 设置，
选择吞吐稳定且内存不过高的一档。执行一个旧 runtime 与 static runtime 的相同样本
benchmark。

建议将“数据准备耗时至少下降约 25%”作为进入全量构建的最低实用标准；期望目标是约
2 倍或更高。若不足 25%，先 profile 一次剩余热点，不直接扩大 worker 或开始全量。

### Phase D：PI1M 959,588 production static cache

输入：

```text
bundle  = 30f17b59bc5862a1ddae7eaee03b2767df26561d9bfecb690ec8eea3ddd09ed2
cohort  = data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1
count   = 959588
```

执行前依据 10k pilot 外推确认磁盘足够。完成后仅做必要验收：

- sample count 和 ordered key coverage 完整；
- offsets 单调且末值等于对应数组长度；
- index bounds、shape、dtype 正确；
- 无构建失败；
- 确定性抽取 1,000 条与旧 runtime 做静态/动态等价比较；
- reader 只读打开成功。

不在这里重复全量 Stereo、ETKDG/MMFF 或基础缓存化学审计。

### Phase E：downstream 3,655 production static cache

使用 downstream unique key 的首次出现顺序构建，同 schema、独立 manifest。完成后证明：

- 3,655 个 unique structures 全部有 static row；
- 6,265 个 property rows 全部能通过 key 解析；
- 9 个已知 geometry fallback 保留 2D/BRICS/fingerprint，3D arrays 为空；
- task、label、fold 和 split 文件未改变；
- `DROPPED_DOWNSTREAM_ROWS = 0`。

downstream 较小，可做全量 shape/index/resolve 检查，不需要额外 1k→10k 阶段。

### Phase F：真实 runtime smoke 与性能对照

用同一组真实 key、相同 seed、相同 absolute positions 比较：

```text
旧 RUNTIME_STATIC_REUSE
vs.
新 PERSISTENT_STATIC_CACHE
```

检查：

- 2 real samples 和一个 small batch；
- Concat/KFuse 各一次 BF16 forward/backward；
- 三项 loss 和必要梯度有限；
- old/new mask、noise、targets 一致；
- old/new 模型输出在 BF16 合理容差内一致；
- 0、4、8 prep workers 中选少量设置测吞吐和 GPU duty cycle；
- cache reader 不产生写入。

这里只是正确性/性能 smoke，不输出模型性能结论，也不要求重新跑一个完整 5k 来证明
缓存正确。

### Phase G：启用与记录

当 PI1M 和 downstream static cache 均完成，并且 Phase F 通过后：

- 在未来正式配置中显式写入 `dual_static_root`；
- 在 `PIPELINE.md` 记录 schema、parent/cohort、路径、count、benchmark 和验证范围；
- 保留当前 5k 的 `RUNTIME_STATIC_REUSE` provenance，不改写其 run metadata；
- 为未来 20k 或新正式实验提供命令模板，但等待用户授权启动。

## 10. 最小完成条件

只维护三个直观状态：

```text
PI1M_STATIC_CACHE_READY
DOWNSTREAM_STATIC_CACHE_READY
STATIC_RUNTIME_READY
```

对应条件：

|状态|必要条件|
|-|-|
|`PI1M_STATIC_CACHE_READY=YES`|959,588 key 完整、无派生失败、抽样等价、reader 可读|
|`DOWNSTREAM_STATIC_CACHE_READY=YES`|3,655 unique / 6,265 rows 全解析、fallback 保留、零丢行|
|`STATIC_RUNTIME_READY=YES`|两套 cache ready，Concat/KFuse 小 batch 正确，存在明确准备吞吐收益|

这三个状态只说明 training-ready cache 已就绪，不等同于预训练完成、性能提升或正式微调
验收通过。

## 11. 建议日志与产物

建议 tmux windows：

```text
dual_static_tests
dual_static_pilot1k
dual_static_pilot10k
dual_static_pi1m
dual_static_downstream
dual_static_smoke
```

建议日志：

```text
logs/glt_dual_static/<phase>.command.log
logs/glt_dual_static/<phase>.log
```

建议报告：

```text
results/glt_dual_static/pilot1k_report.json
results/glt_dual_static/pilot10k_report.json
results/glt_dual_static/pi1m_report.json
results/glt_dual_static/downstream_report.json
results/glt_dual_static/runtime_smoke.json
```

## 12. 下一轮执行顺序

1. 等当前已授权 5k 任务结束或释放 CPU，不干扰其 baseline。
2. 完成 Phase A 的静态/动态拆分与相关单元测试。
3. 执行 PI1M 1k pilot。
4. 执行 PI1M 10k pilot，确定 worker 数、空间和实际收益。
5. 达到实用收益后构建 PI1M 959,588 static cache。
6. 构建 downstream 3,655 static cache并核对 6,265 rows。
7. 执行 old/new real-data runtime smoke 与吞吐对照。
8. 更新 `PIPELINE.md`，给出未来正式预训练命令模板；停止并等待实验授权。

以上顺序不会要求为了“更安全”额外重建主缓存、全量复审已冻结化学数据或增加复杂
迁移框架。必要检查只围绕三件事：key 不错位、clean/noisy 不泄漏、训练确实变快。
