# Uni-Poly-Plus 工程维护执行计划

> 状态：计划已冻结，可直接进入实施；不再增加 Identity、Manifest、Contract、Gate、Registry、BaseTrainer 或统一 Trainer。
>
> 核心原则：**删掉重复防御逻辑，只保留会直接影响正确运行的检查；用最小重构和真实运行验证替代复杂工程机制。**

## 0. 当前仓库事实与实施起点

本计划以 `/root/workspace/Uni-Poly-Plus-master` 当前工作树为唯一依据。实施前已确认的真实情况如下：

- `scripts/pretrain.py` 同时承担参数解析、数据集构造、DDP、采样器、全部预训练 objective、训练循环、resume、milestone、final checkpoint 和 completion marker。
- `scripts/train.py` 同时承担参数解析、数据集/折划分、checkpoint 迁移、单折训练、预测与 shard 写入；epoch 级训练函数主要位于 `src/utils.py`。
- 8×5 展开、四 GPU slot、LPT、失败停止派发及 shard 跳过目前实际写在 `scripts/run_mips_trimer_scage.sh` 的 Bash 函数中，而不在 `scripts/train.py`。
- launcher 当前调用 `scripts/resolve_mips_trimer_scage.py --shell`，但随后只是把原 experiment JSON 复制为 `results/.../configs/resolved_input.json`；它并不是实际 resolved 参数。
- launcher 的 `PRETRAIN_CHECKPOINT_INTERVAL_STEPS` 当前默认是 `0`，而直接运行 `scripts/pretrain.py` 时 `--checkpoint_interval_steps` 默认是 `250`；两者不一致。
- `scripts/pretrain.py::save_train_state()` 已通过 `_atomic_torch_save()` 使用临时文件和 `os.replace()`；但 formal profile 还会每 2000 optimizer steps调用 `save_categorical_milestone()` 生成 `step_XXXXX.pth`。
- final checkpoint 当前是 `state_dict + 大量 meta`，completion marker 当前复制 checkpoint/profile/code hash 等信息。
- `src/dataset/mts_star_rbf_v2.py::StarRBFV2Sidecar` 启动时会重算 metadata/hash、扫描每个 NPY SHA，并对整列数组执行 `all/any/min/max`。
- `src/dataset/dataset.py::_init_relation_geometry_sidecar()`、`RelationGeometrySidecar` 和 `RelationGeometryPermutation` 仍按 artifact/cohort/source hash 绑定；`mips_trimer_collate()` 还把这些身份字符串带进 batch。
- `scripts/run_mips_trimer_scage.sh::validate_shard()` 会重算 checkpoint、cache、prediction 和 split SHA；多个 watcher 在 completion marker 后还会调用独立 audit。

用户已明确放弃当前 R2 实验。因此实施无需等待 R2 final，也不得尝试从 R2 milestone 恢复或续训。第一项执行动作是确认并停止所有仍指向 R2 输出路径的训练、watcher 和下游触发进程；现有 R2 checkpoint、`.last.pt`、milestone、日志和结果均保持原状，不在本维护周期删除。任何历史产物清理必须另行授权。

## 1. 固定范围与不变量

### 1.1 科学语义完全冻结

本周期不得改变：

- 3D 构象数量及 Trimer 构象生成规则；
- Star-RBF v2 的 relation/pair 几何、RBF upper、projection 和无效回退；
- MSTA 层、local/context SPD、Attention 方向及缩放；
- Backbone annotation；
- 模型结构、hidden size、层数、head 数、norm、激活和 readout；
- PI1M_v2/downstream 数据定义、sample 顺序、fold 定义和任务集合；
- masked-atom/angle 等训练目标、loss 权重和科学超参数；
- 现有冻结 cache、cohort、relation-geometry sidecar 和 Star-RBF v2 sidecar 内容。

工程重构不得以重建 cache/sidecar 的方式绕过兼容问题，也不得运行正式 20k 预训练或完整 8×5 微调来证明重构正确。

### 1.2 允许修改的工程范围

- 删除活动训练/恢复/调度/监控路径中的 integrity、identity、compatibility hash 计算与比对；
- 简化 checkpoint、completion marker 和 watcher 生命周期；
- 把 sidecar 启动检查从全量审计降为轻量结构检查；
- 按职责渐进提取 Pretrain/Finetune 模块；
- 简化 resolver、launcher、结果 metadata 和相关文档；
- 添加直接数值和短运行测试。

### 1.3 明确不做

- 不新增四类 Identity、`run_manifest.json`、audit receipt 或持久 baseline；
- 不新增统一 Trainer、BaseTrainer、Registry、插件式 objective 框架；
- 不迁移、不重写历史 checkpoint/metadata；旧 hash 字段由新代码忽略；
- 不重命名历史 hash 目录；目录名只作为现有路径片段使用；
- 不删除与本计划无关的原子写、只读打开、shape/dtype、索引边界、finite、进程清理或防覆盖措施；
- 不修改当前 R2 产物；只停止仍在运行的 R2 writer/watcher。

## 2. 哈希分类：删除检测链，保留算法行为

### 2.1 必须删除的哈希

活动生产路径不再生成、传递或比较以下身份字段：

- `checkpoint_sha256`、`pretrain_code_sha256`、dirty diff/code tree SHA；
- `config_hash`、`resolved_config_hash`、`training_config_hash`、`finetune_config_hash`、`finetune_profile_hash`；
- `feature_config_hash`、`graph_model_config_hash`、`geometry_model_config_hash`、`source_geometry_model_config_hash` 作为运行兼容判定；
- cache bundle、Topology、Trimer、angle、relation geometry、Star-RBF v2、G3 permutation 的 `artifact_hash` 启动比对；
- `.done/.frozen` 内容与 metadata/artifact hash 的交叉绑定；
- NPY、LMDB/feature shard、CSV、prediction、result shard 的全文件 SHA；
- checkpoint/source/target/final cache contract 的 digest 与逐字段身份检查；
- launcher、scheduler、watcher、monitor、doctor 和 summarizer 中为上述字段服务的参数和审计。

历史 JSON/CSV/checkpoint 中已有字段不迁移、不改写。读取旧产物时允许字段存在，但不得据此拒绝加载。

### 2.2 必须保留的算法性哈希

以下哈希会直接决定数据或算法行为，不属于本轮删除范围：

- `src/dataset/lmdb_cache.py::sample_key_from_smiles()` 与 `sample_key_from_normalized()`：内容寻址和 LMDB key；
- cohort 的 sample-key/row-key 顺序及其现有 hash 目录定位：不能改变 key 映射和数据顺序；
- `scripts/pretrain.py::_graph_periodic_aug_loss()` 内由 seed/epoch/SMILES 决定选择的 hash；
- `src/dataset/dataset.py` 写入 `mts_sample_hash64` 的确定性样本标识；
- `scripts/train.py` nested5 内层验证集的 sample-key SHA 排序；
- `src/dataset/mts_star_rbf_v2.py::_signature()` 生成的 path-signature bytes；它只属于既有 QC 数据语义，不进入 forward；
- deterministic sampling、partition、排序、随机 mask 或 key 映射中实际参与选择的哈希；
- `LmdbFeatureStore.replace_raw_verified()` 的 compare-and-replace 检查：这是显式数据修复写操作的并发保护，不在训练读取热路径，且不得因本计划移除。

`sample_order_hash` 和 `split_manifest_hash` 要拆开处理：固定 fold 仍必须按 CSV 行顺序和显式 indices 使用；运行时不再计算/比较整份 manifest SHA，但必须检查 schema、sample count、5 个 fold、索引范围、重复/遗漏以及当前 CSV 行数。nested5 的 hash 排序算法保持不变。

## 3. Checkpoint、Milestone 与完成标记

### 3.1 停止生成周期 milestone

修改 `scripts/pretrain.py`：

- 删除 `save_categorical_milestone()`；
- 删除 formal profile 每 2000 step 调用该函数的分支；
- 删除仅为该 milestone 构造的 profile/code/resume metadata；
- cosine-angle 分支若仍会写 `*.step_XXXXX.pth`，也改为只保留内存中的 best state 或显式诊断 JSON，不再写周期模型 checkpoint；不得改变其 objective 计算。

修改 `scripts/run_mips_trimer_scage.sh` 和相关配置说明，使生产预训练不再暗含 milestone 产物。

`scripts/finalize_mts_g_pretrain_milestone.py` 及 `tests/test_mts_g_family_finalization_recovery.py` 当前只服务于历史 milestone 恢复。实施时先用 `rg` 确认它们不再被活动 launcher/watcher 引用，然后：

- 从生产调用链和文档中移除；
- 脚本可作为历史只读恢复工具暂存，不要求本周期删除；
- 不再为它增加新功能或继续维护成正式 finalization 路径。

现有 R2 milestone 因实验已放弃而不再具有恢复职责，但本周期仍不删除它们。

### 3.2 `.last.pt` 是唯一 resume 状态

统一两个入口的默认值：

- `scripts/run_mips_trimer_scage.sh`：`PRETRAIN_CHECKPOINT_INTERVAL_STEPS=2000`；
- `scripts/pretrain.py::parse_arguments()`：`--checkpoint_interval_steps=2000`。

保留 `scripts/pretrain.py::save_train_state()` 的 rank-state gather 和 `_atomic_torch_save()`，将其提取到 `src/training/common/checkpoint.py` 后仍保持：

```text
每 2000 optimizer steps
→ 收集完整 resume state
→ 写同目录临时文件
→ os.replace 原子覆盖 <final>.last.pt
→ DDP 同步
→ 恢复 checkpoint 操作前各 rank RNG/loader generator 状态
→ 继续训练
```

磁盘始终只有一个 `.last.pt`，不产生时间戳副本或历史版本。保存失败时旧文件仍可用，最多回退 2000 optimizer steps。

`.last.pt` 只保存恢复运行真正需要的内容：

- `train_module`/模型参数和当前存在的 auxiliary modules；
- optimizer、scheduler；
- AMP scaler（仅在运行实际创建 scaler 时保存；当前 BF16 autocast 没有 scaler，不伪造空 scaler）；
- epoch、next micro-batch index、global/micro step、optimizer step；
- 每 rank RNG、DataLoader generator 和 sampler position；
- 解析该 payload 所需的最小 layout version，以及恢复 DDP position 所必需的 world size/rank state 数量。

删除 `.last.pt` 中的 `resume_contract` 及 profile/code/config/cache/sidecar/hash identity。恢复时只执行：

1. `torch.load(..., weights_only=False)`；
2. 必需 key、基本类型和 rank-state 数量检查；
3. 模型/aux `load_state_dict(..., strict=True)`；
4. optimizer/scheduler/scaler state 加载；
5. step、RNG、loader generator、sampler position 恢复。

保留 world-size 与 sampler position 检查，因为它们直接决定 resume 是否能从正确位置继续；它们不是身份哈希。

watcher 和 Finetune 永远不得读取 `.last.pt`。

### 3.3 同路径重新运行时清除旧 marker

保留 launcher 当前“默认拒绝覆盖已有输出”的安全行为。只有用户明确授权同一路径重新运行时，使用一个清楚的 restart 入口（例如 `RESTART_SAME_PATH=1`，实现时只选定这一种）执行：

```text
确认目标是精确的 final 文件路径且没有同路径 writer
→ 删除该 final 对应的旧 .complete.json
→ 启动新训练
```

不得用递归删除、目录通配符或自动清理历史产物。restart 模式不把旧 `.last.pt` 当作 resume；只有显式 `RESUME=1` 才加载 `.last.pt`。旧 marker 删除后，即使旧 final 暂时仍存在，watcher 也不会启动下游。

### 3.4 `final.pth` 只服务下游

训练达到最后 optimizer step 后的顺序固定为：

```text
完成最后 optimizer step
→ 从内存中的 base model 提取下游需要的 state_dict
→ 写 final 临时文件
→ os.replace 原子生成 final.pth
→ torch.load 验证可读
→ 按 resolved 参数构造对应预训练模型并 strict=True 加载一次
→ 原子生成 final.pth.complete.json
```

建议最小 payload 固定为：

```python
{"state_dict": model_state_dict}
```

这样兼容 `scripts/train.py` 当前从 `checkpoint['state_dict']` 取权重的方式。`final.pth` 不保存 optimizer、scheduler、scaler、RNG、sampler、resume 计数器、预训练 heads、hash 或重复身份 metadata。

Finetune 仍保留当前必要的结构迁移逻辑：

- `_scage_checkpoint_key_compatibility()` 的 key/shape 判断；
- `select_mts_checkpoint_transfer_keys()` 只迁移 topology encoder/MSTA/G-family geometry 参数；
- 将迁移参数合并进按 fold seed 初始化的 downstream 模型后，调用 `model.load_state_dict(merged_state, strict=True)`。

删除的是 meta/hash 身份 gate，不是 key/shape/strict load。final 自检使用同一模型构造函数，避免出现“保存成功但下游无法读”的假完成。

### 3.5 completion marker

`final.pth.complete.json` 内容固定为：

```json
{
  "status": "complete"
}
```

在 `src/training/common/checkpoint.py` 中以临时 JSON + `os.replace()` 原子写入。不得保存 checkpoint 路径、step、profile、schema、hash 或 checkpoint metadata 副本。

marker 只能在 final 已完成一次 `torch.load + strict=True` 验证后生成。验证失败时清理临时文件、不生成 marker，保留原 `.last.pt` 供恢复。

### 3.6 watcher

活动 watcher 只检查：

1. `final.pth` 是普通文件；
2. `final.pth.complete.json` 是普通文件且 JSON 可解析；
3. `status == "complete"`。

满足后启动 Finetune。watcher 不再 `torch.load`，不计算 hash，不检查 milestone，不交叉比较 checkpoint metadata，也不调用独立 checkpoint audit。

需要处理的真实 watcher/monitor 包括：

- `scripts/auto_trigger_g_family_downstream.sh`：移除 `audit_mts_g0_g1_formal.py --phase checkpoint` 前置调用；
- `scripts/auto_trigger_g1_after_g0.sh`：移除内嵌 `audit_checkpoint/audit_configs` gate；
- `scripts/auto_trigger_g2_downstream.sh`：移除 `audit_mts_g1_g2_formal.py --phase checkpoint` 前置调用；
- `scripts/monitor_mts_t_pretrain0_v1.sh::validate_t1()`：若仍保留为可执行历史 monitor，改为极简 marker 检查，不再计算 SHA 或加载 checkpoint；
- 当前/未来 Star-RBF v2 watcher：使用同一三项规则，不从已放弃 R2 的旧 watcher 继续运行。

任务/折数、输出目录防覆盖、tmux 窗口去重、子进程退出码和失败后停止派发继续保留。

最终职责只有：

```text
<final>.last.pt              唯一 crash/resume 状态
<final>.pth                  唯一正式下游模型
<final>.pth.complete.json    唯一阶段完成信号
<final>.step_XXXXX.pth       停止生成
```

## 4. 删除活动运行时哈希检测链

本阶段在模块化之前完成，先缩短调用链，再移动代码。每一小组修改后立即跑定向测试，不一次性全仓机械替换 `hash` 字样。

### 4.1 Resolver 与 experiment 配置入口

修改 `scripts/resolve_mips_trimer_scage.py`：

- 删除 `digest()`、`_sha256_file()` 和 `g_family_bundle_identity_hash()` 的身份用途；
- 将 `_resolve_g_family_artifact_bundle()` 改为解析 cohort→root，检查 root/metadata/必要 array 文件、JSON/schema/cohort 和可读性，不再要求 config 中存在 `artifact_hash`，不重算 metadata digest，不绑定 `.done/.frozen` 内容；
- 将 `_resolve_star_rbf_v2_bundle()` 改为相同的轻量结构解析；保留两个 cohort、record count、RBF upper 和直接 scientific 字段一致性；
- 删除 resolved payload 中 config/model/feature/geometry/artifact/semantic SHA 字段；
- 保留 `runtime_contract` 中直接可读的 schema/version/枚举/数值字段，因为它们用于解释数据布局和构造模型；
- 对历史 experiment JSON 中的 `artifact_hash` 字段采取“允许存在但忽略”，不批量重写历史配置；新配置只写 root 和实际运行参数。

给 resolver 增加一个简单的原子输出选项，使同一次 resolve 可以：

- 向 launcher 输出 shell 变量；
- 将同一 payload 原子写到 `results/.../configs/resolved_input.json`。

`resolved_input.json` 只包含实际 resolved 参数，不包含 hash、运行日志、硬件快照或派生 provenance。

### 4.2 Launcher 与 scheduler 参数

修改 `scripts/run_mips_trimer_scage.sh`：

- 删除 `hash_training_spec()`、`stage3_training_hash()`、`JOINT_TRAINING_HASH`、`FINETUNE_CONFIG_HASH`、`FINETUNE_PROFILE_HASH`；
- 删除 `COMMON`、`PRETRAIN_G_FAMILY_ARGS`、`TRAIN_G_FAMILY_ARGS`、`STAR_RBF_V2_ARGS_*`、`PRETRAIN_IDENTITY_ARGS`、`TRAIN_IDENTITY_ARGS` 中所有 `*_hash`/`*_sha256` 参数；
- 删除 `CHECKPOINT_SHA256`、`CACHE_STORE_SHA256`、Topology/Trimer `.done` 内容读取；
- 删除从 `scripts.audit_mips_trimer_cache::_specs` 仅为算 SHA 而发生的依赖；cache/sidecar root 由 resolved 参数直接提供；
- 将原来的 `cp -f "$CONFIG" .../resolved_input.json` 替换为 resolver 的真实原子 resolved 输出；
- `VALIDATE_ONLY` 只验证参数枚举、GPU 列表、batch/world size、路径和直接 schema，不再验证 hash 格式；
- 保留严格 GPU 列表解析、输出防覆盖、失败清理、LPT 顺序和 tmux 约束。

把 `validate_shard()` 从 hash 审计改成结构验证：CSV 可读且一行、task/fold/seed/experiment_id 匹配、per-fold metrics 可解析且 finite、prediction 文件可读、`y_true/y_pred/sample_indices` 长度一致、prediction metadata 的 task/fold/seed 匹配。删除 prediction SHA、split manifest SHA、checkpoint/cache/config SHA 比对。

### 4.3 Pretrain checkpoint/config 身份链

修改 `scripts/pretrain.py`：

- 删除 `_file_sha256()`、`_path_tree_sha256()`、`_pretrain_code_identity()`、`_build_final_cache_binding()`；
- 删除所有 hash CLI 参数；
- 删除 `cache_bundle_binding_hash()`、`build_pretrain_target_contract()` 和 `_canonical_json_hash()` 的运行调用；
- 删除 fresh paired initialization 中仅用于 parent/file SHA 绑定的字段和检查，但保留 `torch.load`、模型 key/shape 和 `strict=True`；
- 删除 cost file、CSV、tokenizer tree、git diff 和 final metadata SHA；
- 将 resume 的巨大 `resume_contract` 替换为第 3.2 节的必要恢复 state；
- final 保存改为第 3.4 节的最小 payload；
- 历史 checkpoint 的 `meta` 可被读取但不参与拒载。

必须保留 `_joint_canonical_mask()`、periodic augmentation 和 diagnostics 中实际用于选择样本/mask 的确定性哈希。仅作为日志显示而不影响行为的 `sample_hash/mask_hash` 可删除。

### 4.4 Finetune checkpoint、prediction 和 shard

修改 `scripts/train.py`：

- 删除 `_target_contract_mismatch()`、`_source_contract_digest_mismatch()`、`validate_g_family_checkpoint_binding()`、`_validate_mts_t1_function_preserving_init()` 中的身份/parent SHA 路径；历史 init 专用逻辑若无活动调用则留作历史工具，不进入正式 finetune；
- 删除 `_sha256_file()` 和 hash CLI 参数；
- checkpoint 加载简化为 `torch.load → state_dict 提取 → key/shape 选择 → strict=True merge load`；
- prediction 继续临时文件 + `os.replace()`，但不再计算 `prediction_sha256`；
- prediction metadata 保留 task/fold/seed/fold_seed、评估协议、AMP、batch、GPU 和 sample indices；删除 checkpoint/cache/config/profile/split SHA；
- result shard 保留 experiment_id、task、fold、seed、显式科学/运行参数、metrics、prediction path；删除所有 hash 和复制的 checkpoint identity metadata；
- 固定 shared5 manifest 改为直接结构/索引检查，不比较 `sample_order_hash`/`split_manifest_hash`；nested5 的确定性 hash 排序不动。

`_scage_checkpoint_key_compatibility()` 和 `select_mts_checkpoint_transfer_keys()` 必须保留，它们检查真实 tensor key/shape 和允许迁移范围，不是哈希身份。

### 4.5 Cache、doctor、audit、summarizer

活动 MIPS 读取路径还涉及以下真实模块：

- `src/dataset/mips_cache_validation.py::verify_frozen_cache_bundle()`；
- `src/dataset/dataset.py::ShardedFeatureStore`、`_scage_cache_metadata_compatible()`、`_init_mts_sidecar()`；
- `src/dataset/lmdb_cache.py::build_or_load_cohort()`、`load_cohort()` 及 angle/threshold reader；
- `scripts/doctor_mips_trimer_scage.py`；
- `scripts/audit_mips_trimer_cache.py`；
- `scripts/summarize_mips_trimer_scage.py` 及仍用于生产报告的 compare 脚本。

处理方式：

- 训练启动不再调用 `verify_frozen_cache_bundle()` 的全量 hash 审计；只读打开失败、schema、record count、key/index 边界仍直接报错；
- `ShardedFeatureStore` 保留 manifest/`.done` 存在、SQLite 索引和 `torch.load`，删除 manifest digest 和首次访问 shard 时的全文件 SHA；保留 `stored_smiles == requested_smiles`；
- cohort 训练读取显式使用轻量模式：mmap、shape/dtype、长度和 key lookup；不把全数组转为 bytes 重算 SHA；cohort hash 路径和 sample keys 仍保持原样；
- `_init_mts_sidecar()` 现有 hash 目录作为 locator 保留，避免重建；删除对 required NPY 的逐文件 SHA，改为 metadata/spec、shape、dtype 和行数检查；不把 `mts_sidecar_hash` 传进模型/result；
- doctor 改成路径/JSON/schema、轻量数据读取、checkpoint strict load 和两样本 forward/backward；删除 `_assert_checkpoint_binding()`、`_verify_target_contract()` 和 checkpoint SHA 输出；
- audit/validate 脚本只作为显式离线 QC，不再被 launcher/watcher import 或自动执行；QC 检查内容语义、shape、offset、finite 和 record 数，不创建 receipt、不作为训练前置；
- summarizer/compare 直接读取 finite shard/prediction 和 task/fold/seed，不重算文件 SHA。

`src/dataset/mts_cache_integrity.py` 中 topology/trimer record 的科学内容验证函数可继续供离线 QC 使用；删除/忽略报告中的 file SHA 和 artifact binding。`src/dataset/mips_trimer_contract.py` 保留 schema/version 和 `validate_runtime_args()`；等调用方清理完成后，只删除已经无引用的 `_canonical_json_hash()`、`cache_bundle_binding_hash()`、target-contract builder，不碰科学常量。

## 5. Sidecar 启动轻量化

### 5.1 Star-RBF v2 reader

修改 `src/dataset/mts_star_rbf_v2.py::StarRBFV2Sidecar`：

- 构造参数删除 `expected_artifact_hash` 和 `verify_hashes`；
- 删除 `_sha256_file()`、metadata digest、`.done/.frozen` 内容绑定、NPY SHA；
- 可以保留 `metadata.json`、`.done`、`.frozen` 的存在检查，作为“writer 已结束”的轻量信号，但不读取其 hash；
- 保留 sidecar schema、builder/layout version、array 名称、mmap `allow_pickle=False`、metadata 声明的 shape/dtype、sample/relation/pair count、offset 数组首尾边界；
- 删除构造阶段全数组 monotonic、pair index `min/max`、count/source/valid `any` 扫描；
- 在 `model_row(index)` 中对当前样本做 O(该样本) 的局部检查：index 范围、`rs <= re`、`ps <= pe`、relation pair index 落在本样本 pair 范围、observation count 取值和所需 tensor shape；错误在首次访问该坏样本时直接报告；
- 新增 `qc_row(index)` 返回完整 QC 字段；保留 `row()` 为过渡兼容别名并在调用迁移完成后移除；
- `model_row()` 只返回 forward 实际需要的 `relation_row/relation_pair_index/relation_spd`、pair distances/count/valid/source；不读取 key、multiplicity、path signature、asymmetry、reason code。

`index_for_key()` 的 PI1M ordered `row_hint` 快路径必须保留；key→row fallback 也保留，不能改变数据访问语义。

### 5.2 Dataset 与 collate

修改 `src/dataset/dataset.py`：

- `UniDataset` 构造参数移除 relation/star/permutation expected artifact hash；
- sidecar 初始化后直接检查 `len(sidecar) == len(dataset)`；该记录数检查只读 shape，不扫描数组内容；
- Star-RBF v2 attach 改用 `model_row()`；
- 不再向 `Data` 写 `mts_star_v2_sidecar_artifact`、`mts_star_v2_model_semantic_hash`、relation geometry artifact/cohort hash；
- 保留 relation row 与当前 `lga_spd` 的直接一致性检查、local row/index 边界和 RBF upper 的直接数值配置；
- `_init_relation_geometry_sidecar()` 只传 root/cohort 名和必要结构信息，不传 source/artifact hash；G0 bypass 与 G3 permutation 的实际 correspondence/index 检查保留。

修改 `src/dataset/dataloader.py::mips_trimer_collate()`：

- 删除 `star_v2_artifacts/star_v2_semantics`、geometry artifact/cohort set 的收集和 batch 字段；
- 保留 tensor 拼接、pair offset、relation SPD 对齐、单 batch RBF upper 一致和 geometry arm 一致；
- 不修改 padding、relation multiplicity、A4 mask 或 invalid geometry 语义。

### 5.3 相关 relation reader

对 `src/dataset/mts_relation_geometry.py::RelationGeometrySidecar` 和 `RelationGeometryPermutation` 做同样处理：

- 删除 `_sha256_file()`、expected hash/cohort/source artifact 绑定和 metadata digest；
- 构造阶段只检查文件、schema、shape/dtype、offset 首尾和 record count；
- 全数组 monotonic、valid/reason、permutation min/max 移到离线 QC；
- `row()`/`permutation_for()` 对当前 slice 做局部边界检查；
- path cosine、endpoint distance、valid mask 和 G3 permutation 的真实 tensor 行为保持不变。

### 5.4 显式离线 QC

在不改写 sidecar 的前提下，给现有 sidecar 工具增加一个只读 QC 入口，优先采用小型 `scripts/qc_mts_sidecars.py`，避免让 builder 的默认行为兼任训练启动：

- 参数是明确的 sidecar root 和类型；
- 扫描 offsets monotonic、全局 index、multiplicity、valid/reason、distance finite、Star-RBF source/count、G3 permutation；
- 只打印摘要并以退出码表示成功/失败；
- 不写 audit receipt，不修改 `.done/.frozen/metadata`，不被 launcher 自动调用。

## 6. Pretrain / Finetune 渐进式职责拆分

新目录只做职责归位：

```text
src/training/
├── common/
│   ├── checkpoint.py
│   ├── distributed.py
│   ├── rng.py
│   └── runtime.py
├── pretrain/
│   ├── config.py
│   ├── objectives.py
│   └── engine.py
└── finetune/
    ├── config.py
    ├── engine.py
    └── scheduler.py
```

不得增加 `trainer.py`、基类、registry、hook 系统或跨 Pretrain/Finetune 的训练循环。

### 6.1 第一步：公共小工具

从 `scripts/pretrain.py` 原样提取：

- `distributed.py`：distributed 初始化/均值求和、rank state gather、必要 barrier/cleanup；
- `rng.py`：`_capture_rng_state()`、`_restore_rng_state()`；
- `checkpoint.py`：原子 torch save、`.last.pt` save/load、final save/strict validation、completion marker 读写；
- `runtime.py`：device/autocast/AMP 的小型公共选择函数。

只提取已有逻辑，不改变调用顺序；Pretrain 和 Finetune 可以共同调用工具，但各自保留 optimizer step、loss 和 epoch 循环。

### 6.2 第二步：Pretrain config

将以下职责移到 `src/training/pretrain/config.py`：

- `parse_arguments()`；
- `_load_pretrain_profile()` 的非 hash 参数解析；
- `_dataset_kwargs_from_args()`；
- `validate_mts_pretrain_execution()` 的 world size/global batch/枚举检查。

`scripts/pretrain.py` 暂时仍调用这些函数；先只移动定义和 import。固定 batch 对比通过后再继续。

### 6.3 第三步：Pretrain objectives

按依赖从叶子函数开始，将 objective 代码移到 `src/training/pretrain/objectives.py`：

- loss 权重与 `DynamicPretrainLossWeighter`；
- masked atom、periodic augmentation、LGA relation/SPD/path、Trimer distance；
- angle/focal/circular/geometry、repeat-cut、alignment、shortest-path 和 denoise loss；
- `TrimerAngleHead` 与 `MIPSPretrainContainer` 中只负责 objective/head 的部分。

模型 forward 和现有 loss 数学表达式逐行保留。每移动一组，旧 `scripts/pretrain.py` 调用新函数，在同一固定 batch 上比较所有分项 loss、total loss、梯度和一次 optimizer step；通过后才删除旧定义。

### 6.4 第四步：Pretrain engine

将 `main()` 中下列运行职责移到 `src/training/pretrain/engine.py::run_pretrain(args)`：

- distributed/device/seed；
- dataset、sampler、DataLoader；
- model、heads、optimizer、scheduler；
- resume state 加载；
- batch/accumulation/DDP `no_sync` 训练循环；
- `.last.pt`、final 和 marker 生命周期；
- benchmark/smoke 分支。

`scripts/pretrain.py` 最终只负责 CLI→config→`run_pretrain()`→退出码。diagnostics 仍是显式可选功能，不扩展成 gate。

### 6.5 第五步：Finetune config 与单折 engine

`src/training/finetune/config.py` 接管 `scripts/train.py::parse_arguments()` 和直接参数归一化。

`src/training/finetune/engine.py::run_finetune_job(config, task, seed, fold)` 的边界固定为一个 task×seed×fold，只负责：

- 读取一个任务数据集和固定 fold indices；
- 按 fold seed 构造模型；
- 加载 final checkpoint 并迁移/strict load；
- 构造 train/val/test loader；
- 调用/逐步接管 `src/utils.py` 中 `train_epoch()`、`evaluate()`、`test_model()`、`train_and_evaluate()`；
- 原子写一个 prediction 和一个 shard；
- 返回该 fold 的结构化结果/退出码。

engine 不接收 GPU 列表，不遍历 8 个任务或 5 个 folds，不实现 LPT，不启动子进程。当前 `multitask_pcgrad` 路径属于不同科学运行语义，本周期不删除、不改写；先保持为兼容入口，不能为追求“统一”塞入单折 engine。

### 6.6 第六步：Finetune scheduler

把 `scripts/run_mips_trimer_scage.sh` 中实际存在的以下 Bash 职责移到 `src/training/finetune/scheduler.py`：

- `cleanup_stage3_children`；
- 结构化后的 `validate_shard`；
- `launch_stage3_unit`；
- `run_finetune_seeds`；
- `LPT_V1_TASK_ORDER/LPT_V1_FOLD_ORDER`；
- queue、GPU slot 立即补位、verified shard 跳过、失败停止新派发、在途子进程清理。

scheduler 只以独立子进程调用单折 engine；GPU physical id 只写运行 metadata，不进入科学参数。`scripts/run_mips_trimer_scage.sh` 保留解析环境、准备路径和调用 Python scheduler 的薄壳。

迁移期间保留现有 `MTS_FAKE_TRAIN_CMD` 测试注入能力，先让现有 LPT/失败清理测试改为调用 Python scheduler；新 scheduler 行为通过后再删除 Bash 实现，禁止双重调度。

## 7. 配置和文档简化

### 7.1 单一简单链路

最终链路固定为：

```text
experiment JSON
→ scripts/resolve_mips_trimer_scage.py
→ results/.../configs/resolved_input.json
→ Pretrain 或 Finetune
→ .last.pt / final.pth / prediction / shard
```

`resolved_input.json` 是 resolved 参数快照，不是 Manifest：

- 不生成新身份 ID；
- 不记录源码、Git、GPU、文件 SHA 或 artifact tree；
- 不复制 checkpoint metadata；
- 不被 watcher 当作 compatibility gate；
- 训练和结果报告可以读取其中的显式参数用于展示。

### 7.2 文档职责

工程代码完成后最后更新文档，避免文档先于行为：

- `PIPELINE.md`：只写稳定数据流、入口、`.last/final/marker` 生命周期、sidecar runtime/QC 区别和运行方式；
- `CODEX_CLAUDE_HANDOFF.md` / `CODEX_LUNA_HANDOFF.md`：只保留当前状态、已完成证据和下一步，不复制长期合同；
- `TODO.md`、`TODO_预训练.md`、`TODO_优化.md`：只保留未完成事项，删除已完成的执行记录和重复 hash 身份说明；
- 删除文档中“必须 SHA/hash 才能 resume/finetune/watcher”的旧描述；
- 不改写历史结果报告中已经记录的 hash，它们是历史文本。

## 8. 分阶段实施与最小验证

每个阶段使用“定位→最小修改→立即验证→继续”的方式；任何阶段失败只修该阶段，不新增 gate 或抽象。GPU/多进程/超过一分钟的验证在 `tmux Uni-Poly` 独立窗口运行，输出写入独立工程 smoke 日志，不覆盖正式结果。

### Phase 0：停止已放弃 R2，记录只读起点

操作：

- 用 `tmux list-windows`、`ps`、GPU 进程和打开文件确认所有 R2 writer/watcher/downstream；
- 停止这些 R2 进程，确认不再写 R2 checkpoint、sidecar staging、log 或 result；
- 不恢复、不 finalize、不删除 R2 产物；
- 记录当前 dirty worktree，后续不回退用户修改。

完成：没有 R2 活动 writer，工程文件可安全重构。

### Phase 1：Checkpoint 生命周期

修改：`src/training/common/{checkpoint,rng}.py`、`scripts/pretrain.py`、`scripts/run_mips_trimer_scage.sh`、相关 watcher 和 checkpoint 测试。

验证：

- 临时目录内连续写两次 `.last.pt`，确认只有一个目标文件且第二次可读；
- 模拟保存异常，确认旧 `.last.pt` 未损坏；
- final temp→replace→load→strict load→marker，确认 marker 只有 `status`；
- final strict load 失败时 marker 不存在；
- watcher 在无 marker、错误 status 时不触发，在 final+complete 时只触发一次 stub 命令；
- `rg 'step_[0-9].*\.pth|save_categorical_milestone'` 确认活动生产写路径为零。

### Phase 2：Resolver/launcher/hash 参数清理

修改：resolver、launcher、pretrain/train CLI、active watcher/monitor、doctor/summarizer 和直接相关测试。

验证：

- resolver 对当前 experiment JSON 输出可解析的 resolved JSON；
- `resolved_input.json` 与 resolver payload 相同且不含 identity/integrity hash；
- launcher `VALIDATE_ONLY=1` 通过；
- 非法 GPU、缺失路径、非法枚举仍失败；
- `rg` 审计活动入口不再传 `*_hash/*_sha256`；
- 历史 JSON 多余 hash 字段存在时 resolver 能忽略并运行。

### Phase 3：Sidecar 轻量启动

修改：`mts_star_rbf_v2.py`、`mts_relation_geometry.py`、`dataset.py`、`dataloader.py`、离线 QC 入口及 sidecar 测试。

验证：

- worker 0 与多 worker 取相同 sample key 时 model tensor 字段逐项相等；
- `model_row()` 不访问 QC-only arrays，可通过 monkeypatch/spy 证明；
- shape/dtype/offset 首尾/当前 row index 错误仍精确失败；
- 离线 QC 能发现构造的 monotonic、index、valid/reason 错误，但训练 reader 不做全量扫描；
- 单进程与 3-rank 初始化分别记录墙钟，结果只打印/写临时日志，不建立长期 baseline 或门槛；
- Star-RBF v2 两样本 forward 结果与改前直接 `assert_close`。

### Phase 4：Pretrain 渐进提取

按 config→objective 小组→common→engine 顺序逐次修改。每次仅移动一组符号并删除旧定义。

验证：

- 固定 seed、固定两个样本、克隆模型/optimizer state；
- 同时调用尚未删除的旧路径与新提取路径，比较各项 loss、total loss、prediction/diagnostic tensor、梯度和一次 optimizer step 后参数；
- 使用 `torch.testing.assert_close`，容差按当前 dtype 设置，不写 hash baseline；
- 全部小组通过后才让 CLI 只调用 `run_pretrain()`。

### Phase 5：Finetune 单折 engine 与 scheduler

按 config→单折 engine→Python scheduler 顺序修改。

验证：

- 单 task/seed/fold 旧路径与 engine 使用同一初始 state 和 batch，比较 prediction、loss、一次 step 参数；
- shard/prediction 的 task/fold/seed、shape、finite 和 sample indices 正确；
- 用 fake trainer 验证 40 个 unit 唯一、LPT 顺序、四 slot 立即补位、已验证 shard 跳过、任一 unit 失败后停止派发并清理全部子进程；
- engine 的 import/参数中不存在 GPU 列表、任务列表或 fold 列表。

### Phase 6：短 resume 与真实 smoke

短 resume 测试：

```text
A：同一初始状态连续训练 8 step
B：同一初始状态训练 3 step → 原子写 .last.pt → 新进程 resume → 训练至 8 step
```

直接比较：每步 loss、最终参数、optimizer state tensor、scheduler step/LR、后续 sampler sample-key 顺序、optimizer/micro step。比较用 `torch.testing.assert_close` 和显式序列相等，不用 hash。

真实 smoke：

- GPU 1/2/3 三卡 DDP 预训练 2 optimizer steps；
- 一个代表性 task×seed42×fold0 的短 Finetune；
- loss、gradient、parameter、prediction finite；
- completion watcher 只用 stub 下游验证，不启动正式 8×5；
- 不创建正式 checkpoint/result 路径，不复用已放弃 R2 路径。

### Phase 7：静态回归和文档收尾

至少执行：

```bash
/opt/conda/envs/MTS/bin/python -m py_compile \
  scripts/pretrain.py scripts/train.py scripts/resolve_mips_trimer_scage.py \
  src/training/common/*.py src/training/pretrain/*.py \
  src/training/finetune/*.py \
  src/dataset/mts_star_rbf_v2.py src/dataset/mts_relation_geometry.py

bash -n scripts/run_mips_trimer_scage.sh
bash -n scripts/auto_trigger_g_family_downstream.sh
bash -n scripts/auto_trigger_g1_after_g0.sh
bash -n scripts/auto_trigger_g2_downstream.sh

/opt/conda/envs/MTS/bin/python -m pytest -q \
  tests/test_mts_checkpoint_contract.py \
  tests/test_mts_star_rbf_v2.py \
  tests/test_mts_relation_geometry_sidecar.py \
  tests/test_mts_finetune_v2.py \
  tests/test_mts_speed_optimization.py

git diff --check
```

测试文件要同步改名或重写语义：删除“hash mismatch 必须拒载”的断言，替换为 strict key/shape、轻量结构、原子 lifecycle 和真实行为对比。若拆出新的小型测试文件，应按职责命名，不创建 baseline/gate 框架。

最后更新 `PIPELINE.md`、handoff 和 TODO；不启动正式 20k 或 8×5。

## 9. 风险点与最小处理

- **误删算法性 hash**：每删除一处先确认它的返回值是否参与采样、排序、key、partition 或 cache locator；参与则保留。
- **final 过度精简导致 Finetune 缺 key**：用当前 `select_mts_checkpoint_transfer_keys()` 列出所需 tensors，并以真实模型 strict merge load 验证，不重新加入身份 metadata。
- **resume 缺 rank state**：保留每 rank RNG/loader/sampler 和 world size 数量检查；不以删除 hash 为理由删掉恢复状态。
- **轻量 sidecar 延迟暴露坏行**：当前 row 做局部边界检查；需要全局结论时显式跑离线 QC，不把全扫放回训练启动。
- **旧 marker 误触发**：默认拒绝覆盖；显式同路径 restart 第一动作只删除精确 marker。
- **scheduler 行为漂移**：先用现有 fake trainer 测试 Python scheduler，再删除 Bash 版本；不可让两套 scheduler 同时派发。
- **多任务语义受损**：`multitask_pcgrad` 不纳入单折 engine，不因工程整理改变或删除。
- **dirty worktree 冲突**：只修改本阶段目标文件，逐文件审查 diff，不回退其他用户更改。

## 10. 完成标准

全部满足才结束维护周期：

- 已放弃 R2 没有活动 writer/watcher；其历史产物未被本周期删除或改写；
- 活动生产路径不再执行 integrity/identity/compatibility hash 计算、传参或拒载；
- 剩余 hash 均能指向明确的采样、排序、sample key、partition、数据映射或写操作并发保护用途；
- resolver 真实生成简洁 `resolved_input.json`，没有 Identity/Manifest 系统；
- 预训练不再生成任何 `step_XXXXX.pth`；
- `.last.pt` 每 2000 optimizer steps原子滚动覆盖，并能完成 3→8 step resume；
- `final.pth` 原子保存、只含下游模型参数、并在 marker 前完成一次 strict load；
- `.complete.json` 只有 `{"status": "complete"}`；
- watcher 只检查 final、marker 和 status，不加载 checkpoint、不调用 hash audit；
- sidecar 训练启动没有全文件 SHA、metadata digest 或全数组扫描；
- `model_row()` 不读取 QC-only 字段，离线 QC 不进入启动路径；
- Pretrain 和 Finetune 保持独立训练循环，没有统一 Trainer；
- `finetune/engine.py` 只感知单 task×seed×fold，不感知 8×5、GPU slot 或 LPT；
- `finetune/scheduler.py` 保持四 slot、LPT、跳过、失败停止和子进程清理行为；
- 固定 batch、短 resume、三卡 2-step DDP、单折 Finetune、watcher lifecycle 和定向测试通过；
- 没有运行正式 20k 或完整 8×5，没有重建冻结 cache/sidecar，没有改变科学语义；
- 没有新增 BaseTrainer、Registry、Manifest、Identity、Contract、audit receipt、持久 baseline 或额外 Gate。

本文件是本维护周期的唯一执行计划。实施时按 Phase 0→7 顺序推进；遇到问题只针对可复现失败做最小修复，不再扩展计划机制。
