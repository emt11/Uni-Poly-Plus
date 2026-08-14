# R6：旧 MTS 配置与 T/G/R/A 实验体系彻底退役计划

## 状态

```text
completed
```

R6.1–R6.4 的旧配置、旧实验编排和授权产物清理已经完成；2026-08-14 独立审查发现
的 T 专用 MSTA diagnostics、过期 R2 执行指令及少量稳定文档措辞已由第 12 节 R6.5
收尾修复并通过定向验证。当前 R6.1–R6.5 全部完成，生产仍按计划保持 fail-closed，
等待未来新配置周期。

本周期退役当前旧配置体系，并删除 T、G、R、A 四组历史实验在活动源码中的全部实验
逻辑，不设计下一版配置。这里的“全部”不只指 JSON、launcher 和报告脚本，还包括
训练参数、arm dispatch、专用初始化、Dataset/collate 字段、模型分支、negative control、
artifact reader 和实验 metadata。用户后续会单独制定新配置；在新配置和新 resolver
完成前，MTS 生产入口必须明确拒绝启动，不能通过 CLI 默认值或历史配置悄悄恢复旧
行为。

## 1. 结论与边界

### 1.1 已确定的处理方式

- 不再等待、恢复或完成 R2。R2 的旧配置随全部旧配置一起删除。
- 删除 `configs/mts/` 当前全部 34 个 JSON 配置，不保留旧 schema、旧 profile、旧
  experiment descriptor 或兼容解析分支。
- 删除 T0/T1 matched-pretrain、G0–G4/G-family、R2 和 A0–A4 的活动实验逻辑，不保留
  arm 别名、旧 CLI 参数、环境变量、metadata 字段或隐藏 fallback。
- 不为旧配置编写迁移器、别名、fallback、translation layer 或 archived-config loader。
- 不在本周期猜测下一版字段、默认值、schema 名称或目录布局。
- 新配置出现前，正式 launcher 处于 fail-closed 状态；失败必须发生在 GPU、worker、
  Dataset、cache audit 和输出目录创建之前。

### 1.2 必须保留

- 已按本轮明确授权删除经逐项核对、能够唯一归属于 T/G/R/A（含 R2、A0–A4）的
  checkpoint、results、logs、sidecar、prediction、shard 和空目录；精确清单见第 11.1
  节。不得恢复、重新生成或重新标记这些已退役产物。
- 共享原始数据、通用 cache、保留机制所需 sidecar、维护验证产物及其他非 T/G/R/A
  实验文件继续保留。剩余历史 metadata 中的旧 config/hash/profile 字段不迁移、不重写，
  仅作为历史记录存在。
- 保留 Star-RBF v2、MSTA、Attention、Backbone、非实验主干模型结构、数据定义、
  训练目标和当前通用科学实现；本周期不增加 3D 构象、不重建 cache/sidecar。保留的
  是不带实验 arm 身份的机制本身，不保留 T/G/R/A 的选择、配对或比较逻辑。
- MSTA 层和 `topology_attention_variant` 作为通用模型能力保留，但删除 T0/T1 名称、
  `model_identity`、function-preserving T1 转换和 paired step-0 合同。
- Star-RBF v2 的周期 pair 几何、sidecar reader/builder 和 attention bias 保留，但删除
  R2、`legacy_g1_frozen`、G1-parent conversion、R2 bundle 和共享 step-0 身份。
- 普通 Star-RBF/MCL 开关可作为底层显式模型参数保留，但删除 A0–A4 descriptor、A4
  random-mask 机制以及按 experiment ID 切换分支的代码。
- 本轮产物删除授权仅覆盖第 11.1 节已经核对并删除的目标；R6.5 不再删除任何产物，
  也不扩大到共享数据、通用 cache、保留机制产物或其他实验文件。
- 保留仍被活动数据算法使用的 hash，例如 deterministic sampling、sample key、排序、
  partition、row mapping 和 sidecar locator。只有随已退役调用链一起变成死代码的
  locator hash 才可删除。
- 保留 `torch.load`、state-dict key/shape/finite 检查和 `strict=True` 等直接保证模型
  可加载性的检查。
- 不再保证清理后仍存在于其他位置的历史 T/G/R/A checkpoint 可由当前代码严格加载；
  删除 G bias 等专用参数布局后不兼容是已接受结果。未来新 checkpoint 仍必须
  `strict=True`，禁止为旧 checkpoint 增加兼容 loader。

### 1.3 本周期不做

- 不创建任何新 MTS JSON 配置。
- 不决定新配置是否继承、分层、使用何种 schema 或包含哪些科学参数。
- 不运行正式 20k 预训练、完整 8×5 微调或 GPU 训练 smoke；没有活动生产配置时，
  GPU smoke 没有有效配置语义。
- 不修改与旧配置退役无关的 checkpoint 生命周期、模型数学、Dataset 字段或科学
  超参数。
- 不清理 dirty worktree 中的其他用户改动。

### 1.4 “实验逻辑全部删除”的判定口径

| 实验线 | 必须删除 | 可以保留的通用机制 |
|---|---|---|
| T | T0/T1 名称、matched/fresh-paired 初始化、step-0 转换、readiness/diagnostics、比较与 checkpoint identity | O8 attention、MSTA layer 类及其纯数学测试 |
| G | `g_family_arm`、`g0/g1/g2/g3/g4` mode、path-cosine/endpoint/permutation arm、relation-geometry bias 和双 cohort bundle | 无 G arm 身份的 canonical topology 与 Trimer 基础几何工具 |
| R | R2 experiment、G1→R2 initializer、legacy-backbone binding、R2 step-0/bundle/晋级比较 | Star-RBF v2 周期 relation 几何和 RBF 核心实现 |
| A | A0–A4 ID、ablation config/env、A4 count-matched random mask、shared-ablation checkpoint 与调度 | 普通模型构造中独立的 Star-RBF/MCL 显式布尔参数 |

若一个符号同时服务通用机制和历史实验，先把通用部分移到已有的中性模块，再删除带
T/G/R/A 语义的 wrapper；不得仅改名后完整保留旧实验状态机。

## 2. 当前真实依赖清单

### 2.1 待删除的 34 个配置

按当前仓库实际文件分组删除：

```text
configs/mts/default.json                                      1
configs/mts/explicit_k_ru.json                                1
configs/mts/pretraining/canonical_ru_angle20_v1.json          1
configs/mts/experiments/*.json                               26
configs/mts/geometry_injection_ablation/*.json                5
```

配置删除后允许空目录一并消失，不新增 `.gitkeep`、README 占位配置或旧配置归档目录。
Git 历史已经提供恢复能力，不再复制一套仓库内 archive。

### 2.2 仍绑定旧配置的核心入口

- `scripts/resolve_mips_trimer_scage.py::main()` 默认读取
  `configs/mts/default.json`，并实现旧 production、experiment、pretrain-experiment、
  G-family、R2 和 A0–A4 解析。
- 同文件 `_is_legacy_identity_field()` 会按名称宽泛忽略包含 `hash`、`sha`、
  `artifact` 或 `semantic` 的未知字段；该兼容行为必须删除。
- `scripts/run_mips_trimer_scage.sh` 默认使用 `configs/mts/default.json` 和
  `canonical_ru_angle20_v1`，还会生成 `resolved_input.json`。
- `scripts/run_mts.sh` 与 `scripts/run_train.sh` 仍把 `EXPERIMENT_CONFIG` 回退到旧
  `default.json`。
- `src/training/pretrain/config.py::load_pretrain_profile()`、
  `validate_mts_pretrain_execution()` 和 `parse_arguments()` 固定旧 profile ID 与路径。
- `src/dataset/mips_trimer_contract.py` 仍导出配置/profile 常量；只能删除已无消费者的
  配置常量，cache、layout、feature、sidecar 和 checkpoint schema 常量不得连带删除。
- `scripts/mts.py`、`scripts/run_mts_sota_campaign.py`、旧 benchmark、T/G/R/A
  initializer、monitor、watcher、audit 和 comparison 脚本仍内置旧配置路径或 profile。
- `src/modules/mips_local_graph.py` 仍包含 `topology_attention_identity()`、
  `add_function_preserving_t1_parameters()`、T0/T1 `model_identity`、
  `MTSRelationGeometryBias`、`g_family_arm`、G0–G3 geometry dispatch、G2/G3 endpoint
  distance 和 coordinate-shuffle negative-control 分支。
- `src/modules/uni_encoder.py` 仍沿三层构造链传递 `g_family_arm`、
  `relation_geometry_sidecar` 和 `g3_permutation_sidecar`。
- `src/dataset/dataset.py` 与 `src/dataset/dataloader.py` 仍加载、附加并 collate
  `mts_relation_geometry_*`、G3 permutation 和 A4 `mcl_random_mask_*` 字段。
- `src/dataset/mts_relation_geometry.py`、
  `scripts/build_mts_relation_geometry_sidecar.py` 和
  `scripts/build_mts_g3_permutation.py` 是 G-family 专用数据链；其中仅
  `prepare_topology()`/`prepare_trimer()` 被 Star-RBF v2 复用。
- `src/dataset/mts_ablation_random_mask.py` 和
  `src/modules/trimer_mcl.py` 的 `count_matched_random` 分支是 A4 专用逻辑。
- Pretrain/Finetune config 与 engine 仍接受 `initialization_state`、`paired_init_id`、
  `shared_step0_id`、`g_family_arm`、relation/permutation sidecar、
  `backbone_definition` 和 ablation 环境变量，并把它们写入 checkpoint/shard metadata。
- 多个测试和 `PIPELINE.md`、`TODO_优化.md`、`TODO_预训练.md` 仍把旧配置文件当作
  当前事实源。

### 2.3 同时收尾的已确认启动问题

`scripts/run_mips_trimer_scage.sh` 当前仍使用：

```bash
CACHE_VALIDATE=${CACHE_VALIDATE:-full}
```

未来生产入口恢复时默认应为 `sample`；完整扫描只允许由显式离线 QC 命令执行。
本周期只改默认值和调用边界，不修改冻结 sidecar 内容。

## 3. Stage R6.1：删除旧配置与专用实验编排

### 3.1 删除配置

删除第 2.1 节列出的全部 JSON。不得保留 `default.json` 作为隐式默认，也不得把旧
配置改名后继续使用。

删除后立即执行：

```bash
test -z "$(find configs/mts -type f -name '*.json' -print 2>/dev/null)"
```

### 3.2 删除只服务旧实验的脚本

下列脚本的职责建立在已删除的 T/G/R/A 配置或 profile 上，应直接删除，不改造成
兼容 wrapper：

```text
scripts/audit_mts_g0_g1_formal.py
scripts/audit_mts_g1_g2_formal.py
scripts/audit_mts_t0_t1_formal.py
scripts/audit_mts_t_pretrain0.py
scripts/compare_mts_g0_g1_formal.py
scripts/compare_mts_g1_g2_formal.py
scripts/compare_mts_g1_r2_formal.py
scripts/compare_mts_t0_t1_formal.py
scripts/initialize_mts_g_family_pretrain.py
scripts/initialize_mts_g_family.py
scripts/initialize_mts_star_rbf_v2.py
scripts/initialize_mts_t1.py
scripts/initialize_mts_t_pretrain0.py
scripts/record_mts_t1_readiness.py
scripts/record_mts_t1_repair.py
scripts/monitor_mts_t_pretrain0_v1.sh
scripts/auto_trigger_g1_after_g0.sh
scripts/auto_trigger_g2_downstream.sh
scripts/auto_trigger_g_family_downstream.sh
scripts/run_mts_geometry_injection_ablation.py
scripts/run_mts_sota_campaign.py
scripts/finalize_mts_g_pretrain_milestone.py
scripts/analyze_mts_g_precheck.py
scripts/build_mts_g3_permutation.py
scripts/mts_g_family_ddp_smoke.py
scripts/mts_g_family_readiness_smoke.py
scripts/mts_g_family_sidecar_alignment_smoke.py
scripts/mts_g_family_single_fold_smoke.py
scripts/mts_t1_ddp_smoke.py
scripts/mts_t1_readiness_smoke.py
scripts/recheck_mts_t1_diagnostics.py
scripts/mts.py
```

`scripts/build_mts_star_rbf_v2_sidecar.py`、Star-RBF v2 reader、checkpoint I/O、
Pretrain/Finetune engine 和 MSTA 模型类不是旧实验的替代品，不得整体删除；只清除
其中 T/G/R/A 专用分支。

### 3.3 删除 G-family 数据与模型执行链

删除：

```text
scripts/build_mts_relation_geometry_sidecar.py
src/dataset/mts_relation_geometry.py
```

在删除 `src/dataset/mts_relation_geometry.py` 前，将 Star-RBF v2 唯一复用的
`prepare_topology()`、`prepare_trimer()` 及其必要私有 helper 直接移入
`src/dataset/mts_star_rbf_v2.py`。不得把 G relation-sidecar schema、path-cosine、
endpoint-distance、permutation 或 artifact identity 一起搬入。

随后清理：

- `src/modules/mips_local_graph.py`：删除 `MTSRelationGeometryBias`、`g_family_arm`、
  `relation_geometry_sidecar`、`g3_permutation_sidecar`、`g0/g1/g2/g3` geometry alias、
  `g_geometry` forward 分支以及相关参数冻结逻辑。
- `src/modules/uni_encoder.py`：从所有构造层删除上述三个 G 参数的透传。
- `src/dataset/dataset.py`：删除 G arm 解析、`_init_relation_geometry_sidecar()`、
  `_attach_relation_geometry()`、G3 permutation 和 `mts_relation_geometry_*` 写入。
- `src/dataset/dataloader.py`：删除 geometry-arm 一致性、G0 空占位和所有
  `mts_relation_geometry_*` collate/rebase 逻辑。
- `src/dataset/mips_trimer_contract.py`：删除
  `CACHE_RELATION_GEOMETRY_SCHEMA` 与 `RELATION_GEOMETRY_BUILDER_VERSION`。
- `scripts/qc_mts_sidecars.py`：删除 relation-geometry 与 G3-permutation QC 子命令，
  保留 Star-RBF v2 及其他非 G 专用 QC。

磁盘上已有 `data/processed/.../relation_geometry*` 目录不删除；它们变成无人读取的历史
产物，后续若要清理必须另行授权。

### 3.4 删除 A0–A4 执行链

删除 `src/dataset/mts_ablation_random_mask.py`，并清理：

- `src/dataset/mips_trimer_contract.py` 中 `ABLATION_IDS`、
  `ABLATION_RANDOM_MASK_*` 常量；
- `src/dataset/dataset.py` 中 `ablation_config`、`ablation_id`、random-mask sidecar
  初始化/lookup，以及写入 `mts_use_*`、`mcl_random_mask_*` 的代码；
- `src/dataset/dataloader.py` 中 A4 compact-mask 打包、offset 校验和 batch 字段；
- `src/modules/trimer_mcl.py` 中 `precomputed_visible`/`count_matched_random` 分支；
- `src/modules/mips_local_graph.py` 中 `mcl_mask_mode=count_matched_random`；
- Finetune engine 的 `_ablation_switches_from_environment()`、
  `MTS_USE_STAR_RBF`、`MTS_USE_MCL`、`MTS_MCL_RANDOM_MASK` 与 ablation smoke 判断。

模型构造器若仍需要 `use_star_rbf`/`use_mcl` 来直接测试独立模块，可保留两个普通布尔
参数；Dataset 不再接收 per-sample ablation descriptor，launcher 不再通过环境变量
切换它们。

### 3.5 删除 T 与 R 身份/初始化链

清理 `src/modules/mips_local_graph.py`：

- 删除 `topology_attention_identity()`；
- 删除 `add_function_preserving_t1_parameters()`；
- 删除 `model_identity="T0"/"T1"` 及所有 T0/T1 文案；
- 保留 `MSTA_LAYER_INDICES`、MSTA layer 类、`o8`/`msta_last2` 的纯架构选择和数学
  约束，但用机制名称描述，不能再生成 T 身份 metadata。

清理 Pretrain：

- 从 `src/training/pretrain/config.py` 删除 `--initialization_state`、
  `--paired_init_id`、`--shared_step0_id`、`--g_family_arm`、
  `--relation_geometry_sidecar`、`--g3_permutation_sidecar` 和
  `--backbone_definition`；`--star_rbf_v2_sidecar` 作为通用 Star-RBF v2 输入保留。
- 从 `src/training/pretrain/engine.py` 删除
  `_load_fresh_paired_initialization()`、`_is_t1_function_preserving_init_payload()`、
  T1 local-output 初始化审计、G formal 例外、shared-step0 判断以及 T/G/R identity
  metadata。
- `.last.pt` resume 和普通显式 model-weight 加载不得因删除 fresh-paired 实验路径而
  受影响；本周期不新增另一种初始化协议替代它。

Finetune 同步从 config、engine 和 shard metadata 删除相同的 T/G/R 参数和分支。
删除 `legacy_g1_frozen` 与 R2 专用判断，但保留 `star_rbf_v2_sidecar` 到 Dataset/模型的
通用传递能力；同时删除仅为旧 T1 checkpoint 标记保留的 `t1_init_artifact` 等死变量。

### 3.6 删除历史 negative-control mode

从 Dataset、Finetune config、model constructor 和 `TrimerSCAGEMCLResidual` 删除仅为旧
G/A 实验存在的 mode：

```text
coordinate_shuffled
mcl_rbf_coordinate_shuffled
g0
g1
g2
g3
count_matched_random
```

保留当前非实验命名的正常 geometry 实现；不在本周期发明替代 mode 名称。

### 3.7 可复用 benchmark

保留 `scripts/benchmark_mts_pretrain.py` 与 `scripts/benchmark_mts_finetune.py` 的通用
测量逻辑，但删除旧 default/profile/A3 配置路径。没有显式新配置时，两者应给出清晰
错误并退出；不得物化旧配置或用旧 CLI 默认值拼出等价配置。

## 4. Stage R6.2：入口停用与兼容代码删除

### 4.1 Resolver

修改 `scripts/resolve_mips_trimer_scage.py`：

1. 删除 `DEFAULT`、`PRETRAIN_EXPERIMENT_CONFIG_SCHEMA`、旧字段集合、parent
   inheritance、G-family/R2/A0–A4 分支以及旧配置默认值。
2. 删除 `_is_legacy_identity_field()`；以后不得因未知字段名含 `hash` 或 `sha` 而
   静默接受。
3. 删除只被旧 resolver 使用的 `_resolve_g_family_artifact_bundle()`、
   `_resolve_star_rbf_v2_bundle()`、`_random_mask_sidecar_path()` 及其专用 import。
4. 暂时保留同一 CLI 文件名，但将其收缩为明确的“当前无活动 MTS 配置 schema”入口：
   配置路径必须显式提供；即使文件存在，也在解析/启动训练前提示需要先实施新配置
   方案并返回非零退出码。
5. 不生成 `resolved_input.json`，不输出可被 shell `eval` 的半成品参数。

这不是新 resolver 的设计。后续新配置周期会重新定义解析规则，并一次性严格拒绝所有
未知字段，不增加旧 schema 兼容层。

### 4.2 Shell launcher

修改：

```text
scripts/run_mips_trimer_scage.sh
scripts/run_mts.sh
scripts/run_train.sh
```

实施方式：

- 删除三个入口对 `configs/mts/default.json` 的 fallback。
- `EXPERIMENT_CONFIG` 缺失时立即报错退出；路径不存在时立即报错退出。
- 随后调用停用状态 resolver，并传播其非零退出码。
- 以上检查必须位于创建结果目录、读取 cache、调用 `nvidia-smi`、启动
  `torch.distributed.run` 或 worker 之前。
- 删除 `canonical_ru_angle20_v1` 的默认值和专用 shell 条件。
- 删除 `PRETRAIN_G_FAMILY_ARGS`、`TRAIN_G_FAMILY_ARGS`、`G_FAMILY_FORMAL_PRETRAIN`、
  `MTS_INITIALIZATION_STATE`、`MTS_PAIRED_INIT_ID`、`SHARED_STEP0_ID`、
  `BACKBONE_DEFINITION`、G3 permutation、A0–A4 和 R2 专用环境变量/参数数组。
- 删除旧 `resolved_input.json` 写入调用；磁盘上已有历史 resolved 文件不删除。
- 将未来运行时的 cache 默认值从 `full` 改为 `sample`；`CACHE_VALIDATE=full` 仅保留
  为人工显式离线诊断，不得成为普通训练启动默认。

### 4.3 Pretrain/Finetune 内部入口

修改 `src/training/pretrain/config.py` 与直接消费者：

- 删除旧 profile 文件查找、`canonical_ru_angle20_v1` 唯一性判断及旧 profile 默认值。
- `--pretrain_profile` 不再默认指向旧 ID；在新配置设计完成前，生产 launcher 不得
  调用 Pretrain。
- 保留通用 `PretrainRuntimeConfig`、Dataset 参数转换和内部单元测试所需的显式 CLI
  能力，不重新发明临时 profile。
- 删除第 3.5 节列出的 T/G/R 参数；Finetune config 同步删除 G arm、relation
  geometry/permutation、shared step-0、legacy backbone 和 A ablation 参数。

检查 `src/training/pretrain/engine.py`、`src/training/finetune/engine.py` 中对旧
`CONFIG_SCHEMA`、`EXPERIMENT_CONFIG_SCHEMA` 和 profile ID 的使用：

- 删除仅用于接受旧配置/旧 checkpoint metadata 的兼容分支。
- 删除 T/G/R/A 的运行时 dispatch、metadata 输出和 smoke 特例；不能把它们改成
  `deprecated_*` 参数继续接受。
- 保留直接控制当前 forward、loss、数据字段和严格 state-dict 加载的代码。
- 不让本阶段演变成模型或训练循环重构。

最后对 `src/dataset/mips_trimer_contract.py` 做消费者检查：仅删除已经零引用的
`EXPERIMENT_CONFIG_SCHEMA`、`PRETRAIN_PROFILE_SCHEMA`、`PRETRAIN_PROFILE_ID` 等
配置层常量；数据/cache/sidecar/checkpoint contract 常量继续保留。

## 5. Stage R6.3：测试与文档去旧配置化

### 5.1 测试处理

直接删除只证明旧实验配置内容或旧正式比较身份的测试：

```text
tests/test_mts_g0_g1_formal.py
tests/test_mts_g1_g2_formal.py
tests/test_mts_t0_t1_formal_comparison.py
tests/test_mts_geometry_injection_ablation.py
tests/test_mts_g_family_finalization_recovery.py
tests/test_mts_naming.py
tests/test_mts_g_family_dual_cohort.py
tests/test_mts_g_family_readiness.py
tests/test_mts_g_precheck.py
tests/test_mts_relation_geometry_sidecar.py
tests/test_mts_t1_readiness_repair.py
tests/test_mts_watcher_lifecycle.py
```

下列混合测试文件不能整文件删除，只删除/改写其中读取旧 JSON 的用例，保留模型、
Dataset、checkpoint 和数值行为测试：

```text
tests/test_mts_multiscale_topology.py
tests/test_mips_lmdb_cache.py
tests/test_mts_checkpoint_contract.py
tests/test_mts_training_modules.py
tests/test_mts_finetune_v2.py
tests/test_mts_star_rbf_v2.py
```

其中：

- `test_mts_training_modules.py` 删除 fresh-paired initialization fixture/测试，保留
  objective、runtime config 和独立 Finetune scheduler 测试。
- `test_mts_finetune_v2.py` 删除 A ablation 与 G arm 专用用例，保留单 fold engine、
  AMP、指标和调度行为。
- `test_mts_star_rbf_v2.py` 删除构造样本时伪造的 `g1/g0` relation-geometry 字段，改为
  仅使用 Star-RBF v2 自己的 pair/relation 字段；继续验证反演对称、shift、RBF tail
  和 finite gradient。
- `test_mts_multiscale_topology.py` 删除 `topology_attention_identity()`、
  `add_function_preserving_t1_parameters()` 和 T0/T1 identity 断言，但保留 O8/MSTA
  forward 数学测试。

新增一个小型 `tests/test_mts_config_retirement.py`，只验证现实行为：

1. `configs/mts` 下不存在 JSON。
2. 三个 shell launcher 在缺少 `EXPERIMENT_CONFIG` 时非零退出，并且没有创建产物、
   启动 Python 训练或访问 GPU。
3. 显式传入任意临时 JSON 时，resolver 以“当前无活动 schema”拒绝，不接受旧
   `schema_version`，也不静默忽略任意 `*_hash` 字段。
4. benchmark 缺少新配置时明确失败，不物化旧配置。
5. 普通训练入口的 cache 默认不是 full audit。

测试使用临时目录和 subprocess，不新增 baseline、manifest、receipt 或持久 fixture。

### 5.2 文档处理

实施时最小更新：

- `PIPELINE.md`：删除旧配置文件、profile、T/G/R/A launcher 的当前运行说明；明确
  “配置层已退役，生产启动暂停，等待下一版配置方案”。模型和数据流的稳定说明保留。
- `TODO_优化.md`、`TODO_预训练.md`：移除指向已删除 R2 配置的链接；历史结论可保留
  为文字，但不能再称旧配置为活动事实源。
- `CODEX_CLAUDE_HANDOFF.md` 与 `CODEX_LUNA_HANDOFF.md` 不在本周期顺手改写；只有
  它们被用于实际委派时才按各自流程维护。
- 历史 results/logs 中的 Markdown、JSON、CSV 不批量重写链接。

## 6. Stage R6.4：Sidecar 启动默认收尾

本阶段不重做此前已经完成的 sidecar 轻量化，只验证生产调用边界：

- `scripts/run_mips_trimer_scage.sh` 不再默认传 `cache_validate=full`。
- `src/dataset/dataset.py` 的普通 `sample` 路径只做存在性、metadata、shape、dtype、
  offset 和有限抽样检查，不因本次配置清理重新加入全文件 SHA 或全数组扫描。
- 完整 QC 继续通过 `scripts/qc_mts_sidecars.py` 显式运行，不接回 launcher。
- DDP 各 rank 不重复完整 QC。
- `src/dataset/mts_star_rbf_v2.py` 的 ordered-row hint、relation/pair index 和
  `model_row()` 行为保持不变。

如果检查发现 full audit 仍由其他普通生产入口隐式触发，只删除该调用；不扩大为新的
sidecar 重构周期。

## 7. 实施顺序

严格按以下顺序执行：

1. 删除 34 个旧配置，并立即确认无 JSON 残留。
2. 删除 T/G/R/A 专用 initializer、builder、watcher、monitor、audit、comparison、
   campaign 和 smoke 脚本。
3. 将 Star-RBF v2 需要的中性 topology/Trimer helper 移入其模块，然后删除 G
   relation-geometry reader/builder/permutation 数据链。
4. 删除 A4 random-mask Dataset/collate/MCL 数据链。
5. 删除 model/UniEncoder 中 T identity、G arm/bias、R2 binding 和 A negative-control
   dispatch，立即运行保留机制的纯数学测试。
6. 收缩 resolver 为 fail-closed 占位入口，删除全部旧 schema 与 T/G/R/A 解析分支。
7. 修改三个 launcher，确保无配置时在任何副作用前退出。
8. 清理 Pretrain/Finetune config、engine、checkpoint/shard metadata 中的 T/G/R/A
   参数和初始化/运行分支。
9. 清理两个 benchmark 的旧默认，调整测试并增加配置退役行为测试。
10. 更新稳定文档中的活动路径描述。
11. 运行静态检查和 CPU 定向回归，检查无训练、GPU、worker 或新产物。
12. 在 `PLAN.md` 末尾追加实际删除清单、测试结果与遗留项；完成后把状态改为
   `completed`。

每一步只处理当前出现的直接引用。发现未列出的旧配置消费者时，将它归入上述三类：
删除旧实验专用代码、让通用工具显式等待新配置，或把科学/数据算法与配置依赖解耦；
不得为其增加兼容层。

## 8. 最小验证

### 8.1 残留引用

```bash
test -z "$(find configs/mts -type f -name '*.json' -print 2>/dev/null)"

rg -n \
  'configs/mts/(default|explicit_k_ru|experiments|geometry_injection_ablation|pretraining)|canonical_ru_angle20_v1|mts-pretrain-experiment-v1' \
  scripts src tests PIPELINE.md TODO_优化.md TODO_预训练.md

rg -n \
  'g_family_arm|relation_geometry_sidecar|mts_relation_geometry_|g3_permutation|paired_init_id|shared_step0_id|legacy_g1_frozen|topology_attention_identity|add_function_preserving_t1_parameters|A[0-4]_(no3d|star|mcl)|MTS_MCL_RANDOM_MASK|count_matched_random|coordinate_shuffled' \
  scripts src tests PIPELINE.md TODO_优化.md TODO_预训练.md
```

两条 `rg` 均允许命中明确标记为历史的说明或本计划，但活动代码、测试 fixture 和运行
文档不得命中。普通回归指标名 `R2`、矩阵变量 `g1` 或化学字符串中的 `A1` 不是实验
身份，不得按单字母粗暴全仓删除。

### 8.2 Python、Shell 与测试

根据实际保留文件执行：

```bash
/opt/conda/envs/MTS/bin/python -m py_compile \
  scripts/resolve_mips_trimer_scage.py \
  scripts/benchmark_mts_pretrain.py \
  scripts/benchmark_mts_finetune.py \
  scripts/pretrain.py scripts/train.py \
  src/training/pretrain/*.py src/training/finetune/*.py

bash -n scripts/run_mips_trimer_scage.sh
bash -n scripts/run_mts.sh
bash -n scripts/run_train.sh

/opt/conda/envs/MTS/bin/python -m pytest -q \
  tests/test_mts_config_retirement.py \
  tests/test_mts_checkpoint_lifecycle.py \
  tests/test_mts_training_modules.py \
  tests/test_mts_finetune_v2.py \
  tests/test_mts_multiscale_topology.py \
  tests/test_mts_star_rbf_v2.py \
  tests/test_mts_speed_optimization.py

git diff --check
```

若某个保留测试仍导入被删的旧配置 fixture，应改为直接构造该测试真正需要的最小
运行参数；不得重新创建旧 JSON。

### 8.3 真实行为检查

- 无 `EXPERIMENT_CONFIG` 调用三个 launcher，均应在一秒内非零退出。
- 退出前后比较目标临时目录，确认没有生成 resolved input、checkpoint、marker 或日志。
- 检查没有新增 `torchrun`、训练 Python、DataLoader worker 或 GPU 进程。
- 不执行 DDP、单 fold smoke、20k 或 8×5；本周期验收对象是“旧配置已退役且入口不会
  误启动”，不是训练性能。

## 9. 风险点与处理

- **生产入口暂时不可用是预期结果。** 不得为了让 smoke 通过而恢复旧 default。
- **脚本删除可能连带删除有价值的模型测试。** 配置内容测试可以删，模型数学和数据
  行为测试必须保留或移到合适的现有测试文件。
- **旧结果无法从工作树直接复跑是已接受的取舍。** 代码和配置仍可从 Git 历史恢复，
  不在仓库内维护第二套兼容路径。
- **旧 T/G/R/A checkpoint 可能不再严格加载是预期结果。** 已授权目标已按第 11.1 节
  删除；若其他位置仍有历史文件，活动代码也不保留旧模型参数布局。需要复现时使用
  对应 Git revision，不给新代码添加 `strict=False` 或兼容转换。
- **不得继续扩大产物清理。** 本轮授权已经落实为第 11.1 节的精确删除清单；R6.5
  只处理源码和稳定文档残留，不再删除 checkpoint、results、logs、cache 或 sidecar。
- **不要误删算法性 hash。** 仅当其唯一消费者随旧实验代码一同删除时才删除；活动
  sampling、sample-key、partition、row mapping 和 locator 逻辑继续保留。

## 10. 完成标准

只有全部满足时才把状态改为 `completed`：

- `configs/mts` 下 34 个旧 JSON 全部删除，没有改名归档或隐式副本。
- R2 不再是阻塞项，其配置及第 11.1 节列出的专用产物已删除；共享数据、通用 cache、
  维护验证产物和非目标实验产物未动。
- 活动代码和运行文档不再引用旧 config/profile/schema 名称。
- resolver 不含旧 schema 解析、parent inheritance 或 `_is_legacy_identity_field()`。
- 三个 launcher 没有旧配置 fallback，并在缺少新配置时于任何副作用前失败。
- benchmark 不再物化或默认使用旧配置。
- 旧 T/G/R/A 专用 initializer、watcher、monitor、audit、comparison 和 campaign 脚本
  已删除。
- T0/T1 identity、fresh-paired/function-preserving initialization、shared step-0 与专用
  diagnostics 已从活动源码删除；O8/MSTA 核心 forward 仍可独立测试。
- G arm、relation-geometry bias/sidecar/permutation、dual-cohort bundle 和 G-specific
  Dataset/collate 字段已删除；Star-RBF v2 所需的中性几何 helper 已脱离 G 模块。
- R2 initializer、legacy-G1 binding、R2 bundle/metadata/比较逻辑已删除；Star-RBF v2
  周期 relation 几何、reader/builder 和模型 bias 测试仍保留。
- A0–A4 descriptor、A4 random-mask reader/collate/MCL、ablation env 和 negative-control
  mode 已删除；不存在默认成 A3 的隐式 fallback。
- 活动代码不再接受或输出 `g_family_arm`、`paired_init_id`、`shared_step0_id`、
  `legacy_g1_frozen`、`mts_relation_geometry_*` 或 `mcl_random_mask_*`。
- Pretrain/Finetune 不再为旧配置/profile/checkpoint metadata 保留兼容分支；未来模型
  仍使用 key/shape/finite 检查和 `strict=True`，但不要求历史 T/G/R/A checkpoint
  兼容清理后的参数布局。
- 普通生产路径不默认执行 full cache/sidecar audit；完整 QC 仅由显式离线命令执行。
- 配置退役测试、保留的定向 pytest、`py_compile`、`bash -n` 和
  `git diff --check` 全部通过。
- 没有新增新配置、兼容层、迁移器、Identity、manifest、hash gate 或统一 Trainer。
- 没有运行训练、启动 GPU/worker 或重建 cache/sidecar；除第 11.1 节明确授权并记录的
  T/G/R/A 专用产物外，没有修改其他历史产物。

## 11. 执行记录

### 11.1 完成状态与删除前核对

执行日期：2026-08-14。执行根目录为
`/root/workspace/Uni-Poly-Plus-master`。执行前确认工作树存在用户及其他会话的
dirty 修改，未回滚、覆盖或清理无关改动；确认 `Uni-Poly` 中没有活动训练、迁移或
`torchrun` 进程，所有窗口均为空闲 shell。

按本轮用户授权先列出并核对删除目标，再执行删除：

- `configs/mts/` 下的 34 个 JSON 全部属于旧 T/G/R/A 配置，已删除；未改名归档，
  删除后 `find configs/mts -type f -name '*.json'` 为空。
- 旧实验专用脚本 33 个、旧实验测试 11 个及两个旧 Dataset/sidecar 模块已删除。
  精确文件清单以 `git diff --name-status --diff-filter=D` 为准，按目录计数为：
  `configs/mts/` 34、`scripts/` 33、`src/` 2、`tests/` 11。
- 逐项核对并删除的实验产物目录为：

  ```text
  pretrained_models/mts_multiscale_topology/g_family_dual_cohort_repair_v1
  pretrained_models/mts_multiscale_topology/g_family_matched_v1
  pretrained_models/mts_multiscale_topology/g_family_step0_v1
  pretrained_models/mts_multiscale_topology/g_family_step0_full_v1
  pretrained_models/mts_multiscale_topology/g_family_step0_full_v2
  pretrained_models/mts_multiscale_topology/t1_formal_v1
  pretrained_models/mts_multiscale_topology/t1_init
  pretrained_models/mts_multiscale_topology/t_pretrain0_v1
  pretrained_models/mts_star_rbf_v2/legacy_backbone_formal_v1/R2
  results/mts_multiscale_topology/g_family_dual_cohort_repair_v1
  results/mts_multiscale_topology/g_family_matched_v1
  results/mts_multiscale_topology/g_family_readiness_v1
  results/mts_multiscale_topology/g_precheck_v1
  results/mts_multiscale_topology/g_prep_relation_geometry_v1
  results/mts_multiscale_topology/t0_t1_formal_v1
  results/mts_multiscale_topology/t1_readiness
  results/mts_multiscale_topology/t1_repair
  results/mts_multiscale_topology/t_pretrain0_v1
  results/mts_sota_v3/R2_g1_periodic_relation_rbf_v2_legacy_backbone_formal_v1
  results/mts_speed_optimization/r2_worker_sweep_20260813
  results/mts_star_rbf_v2/legacy_backbone_formal_v1/R2
  logs/mts_multiscale_topology/g_family_matched_v1
  logs/mts_multiscale_topology/g_precheck_v1
  logs/mts_multiscale_topology/g_prep_relation_geometry_v1
  logs/mts_multiscale_topology/t0_t1_formal_v1
  logs/mts_multiscale_topology/t1_repair
  logs/mts_multiscale_topology/t_pretrain0_v1
  logs/mts_speed_optimization/r2_worker_sweep_20260813
  logs/mts_star_rbf_v2/legacy_backbone_formal_v1/R2
  data/processed/mips_trimer_scage/relation_geometry
  data/processed/mips_trimer_scage/relation_geometry_permutation
  data/processed/mips_trimer_scage/ablation_random_mask
  ```

  上述目录均已确认不存在；`results/mts_maintenance_repair_v1/`、
  `results/mts_maintenance_smoke_v1/` 及对应 maintenance logs 保留。共享原始数据、
  通用 cache、canonical 冻结 cache、Star-RBF v2/MSTA 机制文件和其他非 T/G/R/A
  实验文件未列入删除目标。

### 11.2 R6 实际修改

阶段记录：R6.1（34 配置、专用编排和授权产物核对/删除）完成；R6.2（resolver、
launcher、Dataset/collate、模型、Pretrain/Finetune 的旧链删除）完成；R6.3（benchmark、
测试与稳定文档去旧配置化）完成；R6.4（sidecar 普通入口边界与 fail-closed 行为）完成。
每个阶段均在下一阶段前完成对应静态或 CPU 定向验证，没有跨阶段启动训练。

- `src/dataset/mts_star_rbf_v2.py` 接管中性 topology/Trimer helper；删除
  `src/dataset/mts_relation_geometry.py` 和 `src/dataset/mts_ablation_random_mask.py`。
- `src/dataset/dataset.py`、`src/dataset/dataloader.py` 删除 G/A sidecar、permutation、
  random-mask 和旧实验字段；拓扑-only 路径仅在 Trimer `.done` 存在时绑定其身份，
  不把可选 Trimer 层变成隐式构建要求。
- `src/modules/mips_local_graph.py`、`src/modules/uni_encoder.py`、
  `src/modules/trimer_mcl.py` 删除 T/G/R/A identity、bias、negative-control 和
  permutation 分支，保留通用 MSTA、Attention、Star-RBF v2 和 Trimer MCL 数学能力。
- `src/training/pretrain/{config,engine}.py`、`src/training/finetune/{config,engine}.py`、
  `src/utils.py`、`src/dataset/mips_trimer_contract.py` 删除旧 profile、arm、paired
  step-0、R2 binding 和旧 metadata；未来 checkpoint 仍走 key/shape/finite 检查和
  `strict=True`。
- `scripts/resolve_mips_trimer_scage.py` 收缩为无活动 schema 的 fail-closed resolver；
  `scripts/run_mips_trimer_scage.sh`、`scripts/run_mts.sh`、`scripts/run_train.sh` 在
  `EXPERIMENT_CONFIG` 缺失或路径不存在时，于 Python、GPU、worker、cache 和输出目录
  初始化前退出。两个 benchmark 改为必须显式提供新配置，不再物化旧默认。
- `PIPELINE.md`、`TODO_优化.md`、`TODO_预训练.md` 清理已删除活动路径；历史结论只作
  文字记录，不再作为活动配置或结果来源。
- 新增/保留 `tests/test_mts_config_retirement.py` 等定向测试；未创建新配置、迁移器、
  Identity、manifest、hash gate 或统一 Trainer。

### 11.3 验证与问题修复证据

- Python 静态编译：`py_compile`（resolver、两个 benchmark、pretrain/train、
  `src/training/pretrain/*.py`、`src/training/finetune/*.py`）退出码 0。
- Shell 语法：三个 launcher `bash -n` 退出码 0；`git diff --check` 退出码 0。
- R6 主回归：
  `tests/test_mts_config_retirement.py`、checkpoint lifecycle、training modules、
  finetune、multiscale topology、Star-RBF v2、speed optimization 共 `46 passed, 1 warning`
  （9.29s）。
- 额外定向回归：`tests/test_mts_checkpoint_contract.py` 与
  `tests/test_mips_lmdb_cache.py` 共 `31 passed, 1 warning`（5.49s）。首次运行发现
  checkpoint 测试参数遗漏当前 canonical topology 字段，以及 topology-only 临时
  cache 缺失 Trimer `.done` 时被错误强制绑定；已分别补齐测试参数并将 Dataset 改为
  “存在则绑定、缺失则保持 topology-only 合法”，修复后全通过。
- 残留引用检查：配置路径/旧 schema 检查和旧 T/G/R/A identity 正则检查均无命中
  （两个 `rg` 返回 1）；仅保留 MorganCount/RDKit 的 `R2`、回归指标 `R2` 及测试中
  的 checkpoint/cache 字符串等非实验身份命中。
- 三个 launcher 在无 `EXPERIMENT_CONFIG` 时均返回退出码 2，并输出 retired-config
  fail-closed 消息；独立临时证据目录中未生成 output/checkpoint，未访问 Python 训练、
  GPU、worker 或生产 cache。
- `pgrep` 未发现活动 `torchrun`、pretrain/train、迁移或 scheduler 进程；未启动 GPU
  训练、正式 20k、8×5 微调、cache/sidecar 重建或生产输出。维护验证产物保持原位。

### 11.4 科学语义与遗留项

Star-RBF v2 周期 relation 几何、MSTA/Attention、Backbone、Trimer 基础 helper、
数据定义和训练目标未因 R6 改写；变更仅删除旧实验选择/身份/调度链并暂停生产 resolver。
保留的 `CONFIG_SCHEMA`、cache/layout/feature/sidecar/checkpoint 常量和通用 hash 用于
当前活动数据与模型消费者，不能删除；通用缓存/RNG 中的 `legacy` 命名、RDKit/Morgan
指标的 `R2` 以及历史文档文字不是旧实验运行逻辑。旧 T/G/R/A checkpoint 即使仍有
其他位置也不再由清理后代码提供兼容加载，这是本周期接受的退役边界。

R6.1–R6.4 的修改与验证已完成。独立审查发现的 T 专用 MSTA milestone diagnostics、
过期 R2 执行指令和旧 explicit/canonical 可运行措辞已在 R6.5 收尾中删除或改为明确
历史说明；本次审查没有启动下一版 MTS 配置或任何训练任务。

## 12. Stage R6.5：独立审查后的最小收尾

### 12.1 目标与边界

本阶段只修复 R6 已声明但尚未完全落实的三项残留：

1. 删除活动 Pretrain 中 T 专用 MSTA milestone diagnostics 链。
2. 将 `TODO_优化.md`、`TODO_预训练.md` 中仍可被解释为活动任务的 R2 指令改为明确的
   历史记录，不能再要求 profiler、继续 R2 或派生 R2 新实验。
3. 将 `PIPELINE.md` 中 `explicit_k_ru`/canonical 的“当前可运行、默认正式生产”措辞
   改为“底层能力保留但当前无活动配置，生产暂停”。

不修改 MSTA forward 数学、Star-RBF v2、Attention、Backbone、Dataset、训练目标、
checkpoint 生命周期或科学超参数；不创建新配置，不启动训练/GPU/worker，不删除任何
新增产物，也不顺手重构 Pretrain engine。

### 12.2 删除 T 专用 diagnostics 活动链

修改以下真实文件：

- `src/training/pretrain/config.py::parse_arguments()`：删除 `--diagnostics_dir` 和
  `--diagnostic_steps`。
- `src/training/pretrain/engine.py`：删除 `_msta_attention_modules()`、
  `_set_msta_diagnostic_mode()`、`_make_fixed_probe()`、`_fixed_probe_losses()`、
  `_diagnostic_record()`，以及训练初始化、step 捕获、`resume_smoke` 和 finalization 中
  对 `diagnostics_enabled`、`diagnostic_steps`、`diagnostic_rows`、`diagnostic_probe`、
  `diagnostic_jsonl`、`contract.json`、`milestones.jsonl`、`report.json` 的全部调用与写入。

删除时保留：

- `msta_last2` 对应的通用 MSTA layer 和正常 forward；
- `local_output`、共享 Q/K/V、local/context relation 计算及其纯数学测试；
- 正常训练、loss、checkpoint、resume smoke 和 final checkpoint 路径；
- 与 diagnostics 无关的 RNG 保存/恢复工具。

不得把 diagnostics 改名为通用 profiler 后继续保留，也不得为删除它引入新 observer、
hook、manifest 或诊断框架。若某个 helper 仍有非 diagnostics 消费者，只删除专用调用，
以 `rg` 和直接消费者为依据决定是否保留。

### 12.3 清理过期活动文档指令

- `TODO_优化.md`：保留已经标明为历史的 R2 配置与结果说明；删除或改写“更新 handoff、
  对现有 R2 profiler、决定继续跑完 R2、基于 R2 新建 matched 实验、使用 R2 milestone”
  等活动步骤。改写后必须明确“R2 已由 R6 退役，不得执行；新实验等待未来新配置”。
- `TODO_预训练.md`：将 `M0：当前 R2` 至后续基于 R2 的推荐链标为历史研究草案，或删除
  其活动执行语气；不能把已删除 R2 当作下一轮 baseline/config。
- `PIPELINE.md`：保留 canonical 与 `explicit_k_ru` 的科学语义说明，但把
  “当前可运行对照”“canonical 是默认且唯一正式 20k 表示”等措辞改为历史/能力描述；
  与已有“下一版配置前生产启动暂停”结论统一。

不批量改写历史 results/logs，不修改 handoff，不设计下一版实验名称、JSON schema、
resolver 或生产默认。

### 12.4 最小验证

先执行残留检查：

```bash
rg -n \
  'diagnostics_dir|diagnostic_steps|mts-msta-branch-diagnostics-v1|_diagnostic_record|_set_msta_diagnostic_mode' \
  src/training/pretrain scripts tests

rg -n \
  '对现有 R2|跑完 R2|M0：当前 R2|当前项目同时保留可运行的 `explicit_k_ru`|canonical 仍是默认' \
  TODO_优化.md TODO_预训练.md PIPELINE.md
```

两条命令应无活动命中；历史说明如果必须保留，应在同一段明确标记“已退役、不得执行”。

随后运行：

```bash
/opt/conda/envs/MTS/bin/python -m py_compile \
  src/training/pretrain/config.py \
  src/training/pretrain/engine.py \
  scripts/pretrain.py

/opt/conda/envs/MTS/bin/python -m pytest -q \
  tests/test_mts_config_retirement.py \
  tests/test_mts_checkpoint_lifecycle.py \
  tests/test_mts_training_modules.py \
  tests/test_mts_multiscale_topology.py \
  tests/test_mts_star_rbf_v2.py

git diff --check
```

不运行 DDP、单 fold、20k 或 8×5。本阶段没有修改 forward 数学和数据路径，CPU 定向
回归足以验证删除 diagnostics 未破坏正常 Pretrain/MSTA 调用。

### 12.5 完成标准

仅当以下条件全部满足，才把顶部状态改回 `completed`，并在本节末尾追加实际修改与
测试结果：

- 活动 CLI 和 Pretrain engine 不再包含 T 专用 MSTA milestone diagnostics。
- MSTA 正常 forward、Pretrain 配置解析、checkpoint/resume smoke 相关测试继续通过。
- TODO 不再包含继续、profiling 或派生 R2 实验的活动指令。
- PIPELINE 明确底层机制可保留，但在新配置产生前没有可启动的生产默认或 explicit
  对照配置。
- 第 1、9、10、11 节关于授权产物删除的描述互相一致。
- 未修改其他文件、科学语义或第 11.1 节之外的历史产物。
- `py_compile`、定向 pytest 和 `git diff --check` 通过。

### 12.6 R6.5 执行记录

执行日期：2026-08-14；根目录为 `/root/workspace/Uni-Poly-Plus-master`。本轮只在
R6.5 范围内修改了以下文件，保留工作树中其他 dirty 改动：

- `src/training/pretrain/config.py`：删除 `--diagnostics_dir` 与
  `--diagnostic_steps` 两个 T 专用 milestone diagnostics CLI 参数。
- `src/training/pretrain/engine.py`：删除 MSTA milestone diagnostics helper、固定
  probe/local-off 捕获、初始化/step/resume/finalization 的 diagnostics 状态和
  `contract.json`、`milestones.jsonl`、`report.json` 写入；正常 forward、loss、
  checkpoint、resume smoke 和 RNG 保存恢复路径保留。
- `TODO_优化.md`、`TODO_预训练.md`：保留历史数值和研究分析，但把 R2 继续、恢复、
  profiler、milestone、20k/8×5 与派生 matched 实验指令改为历史不可执行说明。
- `PIPELINE.md`：将 MTS、canonical、explicit_k_ru、current_mcl 等内容改为保留能力/历史
  语义，明确当前无活动配置、没有可启动生产默认，且删除对已退役 `scripts/mts.py`
  doctor 入口的活动引用。
- 未删除或修改任何新增产物、checkpoint、results、logs、cache、sidecar、handoff 或
  其他历史文件；未创建新配置、resolver、observer、manifest、hash gate 或兼容层。

第 12.4 节验证结果：

- diagnostics 残留 `rg` 返回 1（无命中）。
- 过期 R2/旧生产措辞 `rg` 返回 1（无未标记活动命中）。
- `/opt/conda/envs/MTS/bin/python -m py_compile src/training/pretrain/config.py
  src/training/pretrain/engine.py scripts/pretrain.py` 退出码 0。
- 定向 pytest（config retirement、checkpoint lifecycle、training modules、MSTA
  multiscale、Star-RBF v2）结果为 `25 passed, 1 warning in 6.23s`。
- `git diff --check` 退出码 0；未启动 GPU、训练、worker、DDP、单 fold、20k 或 8×5，
  未重建 cache/sidecar，未产生生产输出。

R6.5 的遗留项仅包括未来新配置周期重新定义生产 resolver/default 和是否启用 canonical/
explicit 能力；本轮不设计、不实现。R6.1–R6.5 全部验收条件满足，顶部状态更新为
`completed`。

### 12.7 MSTA diagnostics 最小收尾记录

根据后续复核意见，在不改变 MSTA 数学语义的前提下继续完成最小清理：

- `src/modules/mips_local_graph.py` 删除 `MSTAMIPSLocalAttention` 的
  `diagnostic_capture`、`diagnostic_local_off`、`last_diagnostic` 三个状态字段，
  删除统计/entropy 分支，并将 forward 恢复为始终直接执行
  `context_projected + local_projected`。
- `tests/test_mts_multiscale_topology.py` 删除三个 diagnostics 专用检查，仅保留
  MSTA layer 选择、relation mask、branch softmax、forward 数学和 local-output 梯度
  测试；同步移除不再使用的测试导入。
- 扩大残留检查覆盖
  `diagnostics_dir|diagnostic_steps|mts-msta-branch-diagnostics-v1|_diagnostic_record|
  _set_msta_diagnostic_mode|diagnostic_capture|diagnostic_local_off|last_diagnostic`，
  在 `src/training/pretrain`、`src/modules/mips_local_graph.py` 和 `tests` 中均无命中。

验证结果：定向 pytest（config retirement、checkpoint lifecycle、training modules、
MSTA multiscale、Star-RBF v2）为 `23 passed, 1 warning in 6.76s`；相关 Python
`py_compile` 退出码 0；扩大残留 `rg` 返回 1；`git diff --check` 退出码 0。未修改
缓存、sidecar、checkpoint、results、logs 或其他历史产物，未启动 GPU、训练、worker、
DDP、20k 或 8×5。该后续最小收尾完成后，PLAN 顶部状态保持/恢复为 `completed`。
