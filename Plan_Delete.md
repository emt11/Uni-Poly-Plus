# GLT-V2 项目精简与缓存清理计划

## 1. 状态、目标与范围

* 计划 ID：CLEANUP-20260916-01，r1。
* 状态：规划完成，删除执行待用户授权。当前仅授权审查并编写本文件，没有删除任何文件、缓存或产物。
* Codex 负责规划／审查，ZCode 负责后续执行。本文件是用户明确指定的清理专项计划，暂不替换仍在进行的 Plan.md 科学任务。
* 保留路线：当前 O8 Bond-Path + 完整 Trimer Galformer 3D，Concat／KFuse、三任务预训练、新 outer5_inner20 微调、当前 geonorm 变体与诊断／缓存生产／审计能力。不是只保留名称含 `glt_v2` 的文件。
* 目标：移除退役路线的入口、专用实现、专用测试和冗余派生缓存；保留当前运行、恢复、再生成、审计和结果解释所需的依赖。
* 本次基于本地与远端代码、store/manifest/run.json、目录尺寸和进程的只读核对。不运行模型或测试，不读取全部 LMDB payload，不将候选目录宣称为已经批准删除。

## 2. 当前重要事实与执行前提

远端根目录为 `/root/workspace/Uni-Poly-Plus-master`；本地为 `D:\CODE\Paper\Uni-Poly-Plus`，本地没有远端的 data/results 完整资产。两端必须分别盘点，Git push 不会同步或备份被 ignore 的缓存。

本次发现 `tmux Uni-Poly:geonorm_5k` 中存在三卡预训练及读取 worker，使用：

```text
configs/mts/glt_dual_three_task_concat_geonorm.json
data/processed/mips_trimer_scage
data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1
data/processed/glt_dual_v2/pi1m/dual_static_v1
data/processed/glt_dual_v2/pi1m/pretrain_targets_v1
results/glt_dual_static_pretrain_5k_concat_geonorm
```

**当前不可执行代码删除、缓存搬移或删除。** 等相关任务及后续自动任务自然结束，取得执行者交接后再动手；不终止训练来制造清理窗口。执行前重新核对进程、tmux、cwd、打开文件与 mmap；只看 GPU 空闲或 lmdb lock 文件存在与否都不够。

## 3. 缓存详细审查与保留集合

### 3.1 必须保留：实际读取链

读取链为 `store.json → active bundle → ru_base/topology/trimer + source → cohort → static/targets → checkpoint identity`。保留整条链；派生 static 不是原始 Trimer 的替代品，训练仍从原坐标生成噪声几何。

|路径（相对远端项目根）|核对结果|处理|
|-|-|-|
|data/processed/mips_trimer_scage/store.json|生产 PI1M store|保留|
|data/processed/mips_trimer_scage/builds/30f17b59bc5862a1ddae7eaee03b2767df26561d9bfecb690ec8eea3ddd09ed2|约 75 GiB，当前正式 bundle|完整保留 source、三层数据、metadata、manifest、key arrays、拒绝／运行记录及 .frozen|
|data/processed/mips_trimer_scage_downstream|约 236 MiB|完整保留|
|其 builds/1545eda5a8f6a1a7868ce01464ce7c6dc714b4685ae10dc7213b90adfbcc23b2|下游 active bundle|完整保留，9 条 fallback 不删除|
|data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1|959,588 条，约 247 MiB|保留；两种原融合及 geonorm 的 run.json 都指向它|
|data/processed/glt_dual_v2/pi1m/dual_static_v1|约 45 GiB，959,588 条|保留；hash `9ff122cc16df5c869582ee2fa07b6f42fa44f2a228f554d25412eb2ae6d6006d`|
|data/processed/glt_dual_v2/pi1m/pretrain_targets_v1|约 426 MiB|保留；hash `5e7b5ec8e5f96bf695494436cd471c9d44a85d99571db88adf34c7d5f5777482`|
|data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1|6,265 性质行，约 2.6 MiB|保留；不能按唯一结构数删除重复性质行|
|data/processed/glt_dual_v2/downstream/union_outer5_inner20|约 3.6 MiB|保留：下游 union 来源／身份依据|
|data/processed/glt_dual_v2/downstream/dual_static_v1|3,655 个结构，约 140 MiB|保留，绑定当前 downstream cohort|
|data/raw、data/splits/mips_outer5_inner20|原始数据、固定 split|保留全部原始数据与来源文件；本轮不清理原始数据|

保护 cache 路径不重命名、不修改旧 hash/schema、不压缩或重写 LMDB，不为节省空间重建构象。不因当前训练不直接读 RU 层而删除生产／溯源依赖。

### 3.2 高收益候选：需完成引用审计后才能删除

`data/processed/mips_trimer_scage` 总计约 193 GiB，是新旧混合根目录，禁止整体删除。下表前缀均为该目录：

|候选相对路径|当前占用约|初步判断与删除条件|
|-|-|-|
|topology|66 GiB|旧根级布局；与 active builds/.../topology 不同，确认无当前读取、硬链接共享或保留恢复依赖后清理|
|trimer|20 GiB|旧根级几何；必须确认不是当前审计 fixture 的唯一坐标来源|
|ru_base|7.7 GiB|旧根级 RU；同上，禁止混淆 active bundle 层|
|periodic_line_glt_v1|3.0 GiB|旧 line sidecar 候选|
|periodic_line_glt_image_v1|2.0 GiB|GLT-v3 image 路线候选|
|periodic_line_glt_distill_v1|4.1 GiB|旧蒸馏 sidecar 候选|
|periodic_line_glt_distill_v2|4.1 GiB|旧蒸馏 revision-2 候选|
|periodic_line_glt_distill_v2.parts|4.1 GiB|旧构建分片；确认无 writer／续建需求后清理|
|md200、md200_pi1m_v1|15 MiB、792 MiB|当前无 MD200；确认没有保留对照的运行依赖后清理|
|builds/0e7c97850147a972c5774d401731a33a05c0e1262a1e841150f87ed8a9134be7.blocked-old-failure-policy|6.3 GiB|失败旧 build；先保留失败报告及 manifest，再清理大 payload|
|builds/ece3a6d6cf6f73f76260afea10ef44f62f38c7b065bdfd672b26dcea9104fb11.contract-blocked-ru-build-boundary|769 MiB|失败旧 build；同上|

以上候选的显示尺寸相加约 119 GiB，但**不是保证可回收空间**；尚需核对硬链接、稀疏文件、打开的文件描述符及磁盘实际 blocks。执行前使用逐路径独立计量和 inode/link count 复核，清理后按文件系统可用空间差额报告。

其他候选：

* `data/processed/glt_dual_v2/pi1m/{dual_static_v1_pilot1k,dual_static_v1_pilot10k,pretrain_targets_v1_pilot1k,pretrain_targets_v1_pilot10k}`：小规模试产；若不再是现有验证入口的必需 fixture，保存验证报告后清理。不能只因包含 pilot 自动删。
* `.../cohort_30f17b59bc5862a1_v2`：与主 cohort 行数及 bundle 相同不等于可互换。核对 ordered keys、records、manifest 差异及所有 run.json 引用，确认无人使用后删；不得替换正式 cohort 来“统一版本”。
* `data/processed/{mts_bench_24,mts_bench_32,mts_bench_48,mts_cache_pilot_20260913,mts_full_dryrun_20260913,mips_trimer_scage_downstream_pilot100,trimer_ensemble_stage_a_20260912,trimer_pilot,trimer_stage_a2_20260912,trimer_v10_pilot_r2_20260913,scage}`：各约 KB 至 127 MiB；查清 fixture、构建恢复和审计引用后逐项列入候选。
* `mips_trimer_scage/{cohorts,validation}`、根级 diagnostics JSON：暂缓。validation 内还有 store.json、expected_keys.sqlite、cohorts 和失败记录，不能将它当纯临时输出。保留当前 Stereo 审计和历史故障定位证据。
* `.staging`、`.parts`、lock、临时文件：检查存活 writer、运行记录和恢复需求，不能按年龄或后缀递归清理。

### 3.3 实际删除清单必须具备的证据

ZCode 在删除前把精确 allowlist 表追加到本文件：绝对路径、所属主机、realpath、大小、store/bundle 身份、被哪些入口／run.json／审计使用、是否有打开 fd/mmap、备份或报告保留位置、拟处理动作。分类仅用 KEEP／DELETE_CANDIDATE／HOLD，缺证据即 HOLD，不默认删除。

引用审计从当前模型、生产缓存入口、所有保留 run.json、Plan.md、PIPELINE.md／RESULTS.md 的证据链出发，递归跟随 source／parents／manifest 绑定。文本搜索只能提供线索；同时检查软链接、硬链接、checkpoint 中记录的身份和数据类序列化依赖。优先复用已有 identity 读取器，不调用具有自动构建／修复行为的 Dataset。

当前大缓存不全量反序列化，不重新 hash 全部 LMDB，不全量 Stereo 扫描；必要时只对保留真实 fixture 做 readonly 读取验证。未完成上述检查前本表仍是候选，不发布“一键 rm”命令。

## 4. 代码精简：先解除依赖，再删除退役实现

### 保留入口及能力

保留 `pretrain_glt_dual.py`、`finetune_glt_dual.py`、正式 grid／aggregate、六 checkpoint／失稳诊断入口、`glt_dual_*` 当前配置（包括 geonorm）、对应 tests。保留完整 Trimer 构建、缓存 store/lifecycle、cohort/union/static/targets 生产与核验、固定 split 生成和 Stereo 审计入口。

核心保留模块包括 `glt_dual*`、`glt_bond_chemistry`、`mips_local_graph`、`original_mips_knowledge_fusion`、`canonical_periodic`、`graph_data`、`periodic_line_glt_complete`、`trimer_mcl` 和当前缓存契约。名称含 mips/scage/v3 的文件仍可能是当前依赖，禁止名称式删除。

### 初始退役候选

* 配置：`atomic_point*.json`、`glt_distill_n_plus_*.json`、`glt_distill_repair_c*.json`、旧 C0/no-MD 独立路线配置。
* 独立入口：`pretrain_mts_glt_distill.py`、`run_mts_glt_distill*`、`build_mts_glt_distill*`、`smoke_mts_glt_distill*`、旧 C0 pipeline/probes/staged-finetune/report、`run_original_mips_atomic_pc_w_camr_v2.py`、旧 GLT-v3 sidecar 构建。
* 专用实现候选：`mts_glt_distill`、`periodic_line_distill*`、`periodic_line_glt_image`、atomic-point／original_mips_atomic_pc 系列、`w_camr_v2_support`，以及确无共享用途的 MD200 组件。
* 专用 tests：旧蒸馏／atomic_pc／C0-only 测试可随被删行为退役；identity、stereo、138维输入、mask、周期关系、缓存生命周期和恢复测试即使叫 mips 仍保留。

逐个候选检查 Python import（包含 package `__init__`）、动态 import、CLI/config 路由、字符串类名、tests 与序列化对象依赖；输出具体文件 allowlist，不以模糊通配符执行删除。

已确认的耦合必须先处理：

1. `src/dataset/__init__.py` 当前 eager import 旧 sidecar／蒸馏；`src/modules/__init__.py` 当前 eager import 蒸馏、旧 encoder 和 point-cloud。先删除退役导出及引用，再删除模块。
2. `scripts/build_mts_cache.py` 使用 `dataset.py` 的生产函数，而 `dataset.py` 又导入旧 line/distill 模块；不能整个删除 dataset.py。最小拆分或解除旧路线导入，保留现有序列化可读性，不随清理改变字段／科学定义。
3. 当前微调使用 `src.utils`，后者依赖 dataloader。先抽离或保留实际使用的训练／标签标准化／collate 路径，不能删除“旧通用模块”导致新入口导入失败。
4. `original_mips_knowledge_fusion.py` 是当前 KFuse 必需文件，保留。`periodic_line_glt_v3.py` 及 `uni_encoder.py` 只有在保留路径、package 导出和 checkpoint 类依赖全部解除后才可删。
5. requirements 仅删除确认不再被保留代码使用的依赖，不在此次清理中升级包版本。

清理采用小批次 commit：退役入口／配置 → 最小依赖解耦 → 专用模块／测试。不得为通过测试同时删除共用验收标准；解耦前后用同一输入与 state_dict 比较结果。

## 5. 结果、日志与历史资产

* 完整保留当前 Concat／KFuse 5k、geonorm、正式 80-fold grid、修复后 summary/OOF、B.2/B.3/C 和 geo-LN 诊断、Stereo 全量审计及其日志；当前模型改进仍依赖它们。
* `results/glt_v2_r2_*`、`mts_glt*`、`mts_c*`、`original_mips_atomic_pc*` 不按前缀直接删除。先保留报告、配置、split 来源、逐折指标／预测、必要参考 checkpoint 和失败证据；大 checkpoint／旧缓存仅在用户接受“不再原地复现该退役实验”后进入实际删除 allowlist。
* 删除 tracked 代码有 Git 恢复路径；ignored 缓存、checkpoint、日志没有。对不可再生或仍有追溯价值的资产，先核实外部备份；仅移动到同盘目录不会释放空间，也不算备份。无备份且保留价值不明则 HOLD，不自动上传或压缩百万记录。
* `.git`、AGENTS.md、Plan.md、Plan_Delete.md、PROJECT_HISTORY.md、PIPELINE.md、RESULTS.md、TODO.md、原始数据与固定 split 都保留。不清理仓库外 Conda/pip/CUDA 缓存、用户 IDE 配置或其他项目。

## 6. 分阶段执行与验收

### 阶段 0：冻结范围和删除清单

用户授权实际清理后，先 pull 并核对当前代码／Plan.md／活动进程。在无运行者使用相关文件的窗口中完成依赖闭包和逐路径 allowlist，Codex 审查具体清单后才进入破坏性删除；不把本次计划编写当作删除授权。若科学路线已经变化，更新本计划，不能沿用过期 keep 集合。

### 阶段 1：精简 tracked 代码

保护未提交改动；只对已核准路径执行 Git 删除与必要的最小依赖解耦。每批做相关导入／CLI help、单元测试、缺引用检查，成功后单独 commit。保留可回退基线 commit，不 rewrite history。

### 阶段 2：删除核准的旧缓存

在 tmux 独立窗口记录执行清单及结果，按精确路径逐项删除，禁止对 `data/processed` 或混合根目录做递归通配删除。每项删除前重新验证 realpath 留在批准根内、不等于保护目录及其祖先、不跨符号链接／挂载点、不被进程占用、身份未变化。Windows 使用原生 PowerShell `-LiteralPath`，先核实最终绝对路径，禁止跨 shell 拼接删除命令。发现新引用则该项 HOLD，不擅自修改消费者以让删除通过。

旧 LMDB、旧分片可以在这些条件全部满足后直接释放，不强制复制百 GiB 数据到同盘。被保留的失败报告／manifest 不放在即将整删的目录里，先存入稳定审计产物目录并记录来源。

### 阶段 3：最小验收与交付

* 生产构建、双路预训练／微调／聚合／诊断入口可导入与解析参数；缓存 builder 能在独立临时目录运行相关合成契约测试，不触碰生产缓存。
* 当前 store、bundle、cohort、static/targets 的身份、记录数与文件存在性保持不变；不改变 `.frozen`、source key 顺序和 split。
* 用既有最多两条真实 fixture（普通与 N=0，如不存在如实说明），两融合 eval 前后向及部署加载验证；与删除前同 checkpoint／同输入输出比较。测试固定 RNG，预算不扩大到 optimizer 更新或正式训练。
* 聚合复算仍得到原 macro8；若只改无关文件且没有影响汇总依赖，可复用已通过证据，不机械跑全仓或 80 folds。
* 报告逐项删／留／HOLD、真实释放磁盘量、剩余依赖、运行命令及验证结果。更新 PIPELINE.md 当前入口与退役说明，不擦除 RESULTS.md 历史结论；完成后归档本轮完整清理周期。
* Git 提交／推送代码与文档，核对远端；缓存删除是远端文件系统动作，与 Git push 分别报告。本地／远端若仅一端清理不得称两端均完成。

## 7. 本轮交付与下一步

本轮仅生成清理计划、盘点当前路径与引用，不执行删除、模型或训练验证。最优先的潜在空间收益来自旧根级 topology/trimer/ru_base 与旧 sidecar，当前 75 GiB bundle 和 45 GiB static 必须保留。

下一步由 ZCode 在相关科学任务结束后，根据本计划准备精确删除 allowlist 与剩余引用证据，交 Codex 审查。未核实缓存和历史资产保持 HOLD；不为追求“只剩 GLT-V2”而删除可复现性和当前共享依赖。
