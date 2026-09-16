# GLT-V2 项目精简与缓存清理计划（当前远程训练机）

## 1. 状态、目标与范围

* 计划 ID：CLEANUP-20260916-01，r3（2026-09-16：审查后明确首批清理完成范围，补入证据收口与下一步）。
* 状态：已完成（首批 12 个精确目标的缓存清理及第 7 节证据补记已完成；原代码精简阶段未实施，其余候选继续 HOLD）。不得将本状态理解为缓存损坏、全量内容审计通过或整个项目精简已完成。
* 授权：用户要求继续当前计划；本次仅执行第 7 节 A 的只读证据补记及必要文档修订，不新增删除、代码修改、模型测试、训练或缓存重建。首批删除的历史授权与执行记录保留在第 6 节。
* Codex 负责本次只读补记、规划与审查记录并自检；本次没有独立执行者复核，因此不将同一主体的自检称为独立审查。本文件仍是用户指定专项计划，不替换 Plan.md 的科学任务。
* r3 文档基线（证据补记前）：`4c3416e3499a91acdff7853dea6801ab851024e9`；本次补记前 `dev` 分支工作树干净，HEAD 为 `248371d`，修改前 `git pull --ff-only origin dev` 返回 Already up to date。r2 环境适配基线 `43f83f7` 与首批实际删除基线 `5a8fe0f` 均为历史阶段，不混作本轮基线。
* 保留路线：当前 O8 Bond-Path + 完整 Trimer Galformer 3D，Concat／KFuse、三任务预训练、新 outer5_inner20 微调、当前 geonorm 变体与诊断／缓存生产／审计能力。不是只保留名称含 `glt_v2` 的文件。
* 目标：移除退役路线的入口、专用实现、专用测试和冗余派生缓存；保留当前运行、恢复、再生成、审计和结果解释所需的依赖。
* r1 候选盘点、r2 环境适配及首批删除分别保留其历史范围；未逐项复核的旧引用与容量不得升级为当前删除依据。r3 已完成只读证据收口；没有重新执行清理、测试、模型、训练或缓存重建。

## 2. 当前重要事实与执行前提

### 2.1 唯一执行环境：当前会话所在的 Linux 训练机

**当前计算机已经是远程训练机，不需要再登录另一台机器执行本计划。** 本文“当前机／本机”均指下面的主机；用户个人电脑上的旧 checkout 不在本轮清理范围。

|项目|r2 环境适配时直接核验值|
|-|-|
|主机名|`dzw2`|
|唯一项目根目录|`/root/workspace/Uni-Poly-Plus-master`（`pwd -P`）|
|Shell／路径语义|Bash／Linux，路径区分大小写|
|Python／torchrun|`/opt/conda/bin/python`／`/opt/conda/bin/torchrun`；不新建或升级环境|
|Git 工作分支／origin|`dev`／`https://github.com/emt11/Uni-Poly-Plus.git`|
|项目所在文件系统|`/root/workspace`，ext4；当前 `findmnt` 显示 `/dev/nvme1n1p1[/docker_home/dzw2]`|
|任务承载|现有 `tmux` session `Uni-Poly`；清理获准后另建独立 window，不能复用训练 pane|

Git 的 `origin` 是代码仓库，不是第二台待清理训练机。`pull/commit/push` 在上述项目目录执行，只同步受版本控制的变更，不能同步、备份或证明已删除被 ignore 的 data/results/logs。本计划不包含 SSH 嵌套执行、个人电脑路径、跨机复制或两端清理验收。若重新连接后主机或项目根变化，停止套用此清单并重新核对。

### 2.2 历史活动任务快照与长期保护边界

本小节记录 r2 环境适配时的 09:11 UTC 快照，不代表当前仍在训练。13:31 UTC 的首批清理预检及后续执行见 3.4 和第 6 节；不得据任一旧快照直接启动下一批删除。

2026-09-16 09:11 UTC 直接检查时，`Uni-Poly:geonorm_5k`（window 27）中仍存在三卡预训练及读取 worker：torchrun PID `1588833`，rank PID `1588950/1588951/1588952`，命令设置 `CUDA_VISIBLE_DEVICES=1,2,3`。PID／window 编号仅是本次快照，不得作为之后自动操作的固定目标。实际命令使用：

```text
configs/mts/glt_dual_three_task_concat_geonorm.json
data/processed/mips_trimer_scage
data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1
data/processed/glt_dual_v2/pi1m/dual_static_v1
data/processed/glt_dual_v2/pi1m/pretrain_targets_v1
results/glt_dual_static_pretrain_5k_concat_geonorm
logs/glt_dual_static_pretrain_concat_geonorm5k.log
```

**09:11 UTC 当时不可执行代码删除、缓存搬移或删除。** 长期约束仍为：等相关任务及后续自动任务自然结束，取得执行者交接后再动手，不终止训练来制造清理窗口；每次获准执行前重新核对进程、tmux、cwd、打开文件与 mmap，只看 GPU 空闲或 lmdb lock 文件存在与否都不够。

当时核实的是训练进程仍存活，不是训练完成、loss 正常或清理目标无人占用；当时未穷尽所有进程的 fd/mmap，后续任务须由执行者确认。不得用旧 Plan.md 中的本地／远端快照推断当前机器状态，也不在本次文档修订中改写其科学任务进度。

### 2.3 执行前只读复核入口

下列命令直接在当前训练机运行，仅用于定位，不执行清理，也不代表已完成引用／占用审计：

```bash
cd /root/workspace/Uni-Poly-Plus-master
hostname
pwd -P
date -u
git status --short --branch
git remote -v
tmux list-windows -t Uni-Poly
tmux list-panes -a -F '#{session_name}:#{window_name}.#{pane_index} pid=#{pane_pid} cwd=#{pane_current_path} cmd=#{pane_current_command}'
ps -eo pid,ppid,etime,args
findmnt -T /root/workspace/Uni-Poly-Plus-master
df -h /root/workspace/Uni-Poly-Plus-master
```

进入获准的清理阶段后，对精确候选路径另查 symlink／inode／fd／mmap 与恢复依赖，权限不足或证据不全则 HOLD。目录容量审计若预计超过一分钟，也必须在 `Uni-Poly` 的独立 window 中运行并留日志；不在训练期间反复全盘扫描，不新增高频轮询。

## 3. 缓存详细审查与保留集合

### 3.1 必须保留：实际读取链

读取链为 `store.json → active bundle → ru_base/topology/trimer + source → cohort → static/targets → checkpoint identity`。保留整条链；派生 static 不是原始 Trimer 的替代品，训练仍从原坐标生成噪声几何。

|路径（相对当前机项目根）|盘点依据（除 active bundle 外，本轮未逐项重验）|处理|
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

按 r1 盘点，`data/processed/mips_trimer_scage` 总计约 193 GiB，是新旧混合根目录，禁止整体删除。下表前缀均为该目录；尺寸不是 r2 实时测量值：

|候选相对路径|r1 记录占用约|初步判断与删除条件|
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

ZCode 在删除前把精确 allowlist 表追加到本文件：`dzw2` 上的绝对路径、核验时间、realpath、大小、store/bundle 身份、被哪些入口／run.json／审计使用、是否有打开 fd/mmap、备份或报告保留位置、拟处理动作。分类仅用 KEEP／DELETE_CANDIDATE／HOLD，缺证据即 HOLD，不默认删除。主机、真实路径或消费者变化后，旧核验失效；批准必须对应具体清单，不能只批准一个目录前缀。

引用审计从当前模型、生产缓存入口、所有保留 run.json、Plan.md、PIPELINE.md／RESULTS.md 的证据链出发，递归跟随 source／parents／manifest 绑定。文本搜索只能提供线索；同时检查软链接、硬链接、checkpoint 中记录的身份和数据类序列化依赖。优先复用已有 identity 读取器，不调用具有自动构建／修复行为的 Dataset。

当前大缓存不全量反序列化，不重新 hash 全部 LMDB，不全量 Stereo 扫描；必要时只对保留真实 fixture 做 readonly 读取验证。未完成上述检查前本表仍是候选，不发布“一键 rm”命令。

### 3.4 本次执行的精确 allowlist（2026-09-16 13:31 UTC 复核）

复核主机为 `dzw2`，项目根为 `/root/workspace/Uni-Poly-Plus-master`，预检时 HEAD 为
`5a8fe0fa2d24696a36ae7bcb311710fa12cbf9bb`，当时工作树干净且已执行
`git pull --ff-only origin dev`（Already up to date）。预检日志为
`logs/cleanup_20260916_01/preflight_audit.log`；其中没有候选路径的打开 fd、硬链接或软链接。
正式 grid 与训练子进程均已退出；仅有旧监视 shell，不读写下列候选路径。

**DELETE_CANDIDATE（本次实际处理）**

1. `/root/workspace/Uni-Poly-Plus-master/data/processed/glt_dual_v2/pi1m/dual_static_v1_pilot1k`（约 41 MiB）：仅被已保存的 `results/glt_dual_static/pilot1k_build.json` provenance 引用，无代码、配置或现行 run 入口引用。
2. `/root/workspace/Uni-Poly-Plus-master/data/processed/glt_dual_v2/pi1m/dual_static_v1_pilot10k`（约 499 MiB）：仅被已保存的 `results/glt_dual_static/pilot10k_build.json` provenance 引用。
3. `/root/workspace/Uni-Poly-Plus-master/data/processed/glt_dual_v2/pi1m/pretrain_targets_v1_pilot1k`（约 516 KiB）：仅被 pilot provenance 引用。
4. `/root/workspace/Uni-Poly-Plus-master/data/processed/glt_dual_v2/pi1m/pretrain_targets_v1_pilot10k`（约 4.9 MiB）：仅被 pilot provenance 引用。
5. `/root/workspace/Uni-Poly-Plus-master/data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1_v2`（约 247 MiB）：与正式 cohort 同计数但无当前代码、配置或保留 run.json 消费者；删除前 manifest 的 SHA256（`9eba93d60781361b005800f98d68ff8eba04a888fadf8b328915e147f1d23945`）及 `results/glt_dual_readiness_20260915/phase2_cohort_v2.json` provenance 仍保留，原 manifest 随 cohort 目录删除。
6. `/root/workspace/Uni-Poly-Plus-master/data/processed/mips_trimer_scage/periodic_line_glt_distill_v2.parts`（约 4.1 GiB）：最终 `periodic_line_glt_distill_v2` 已存在，当前源码与配置无 `.parts` 读取入口，预检无 writer/fd；删除前复制其中 `part_*.log` 到清理日志目录。
7. 两个失败 build 仅删除大 payload，保留失败 provenance：
   * `/root/workspace/Uni-Poly-Plus-master/data/processed/mips_trimer_scage/builds/0e7c97850147a972c5774d401731a33a05c0e1262a1e841150f87ed8a9134be7.blocked-old-failure-policy/ru_base/data.lmdb`
   * `/root/workspace/Uni-Poly-Plus-master/data/processed/mips_trimer_scage/builds/0e7c97850147a972c5774d401731a33a05c0e1262a1e841150f87ed8a9134be7.blocked-old-failure-policy/topology/data.lmdb`
   * `/root/workspace/Uni-Poly-Plus-master/data/processed/mips_trimer_scage/builds/0e7c97850147a972c5774d401731a33a05c0e1262a1e841150f87ed8a9134be7.blocked-old-failure-policy/source/records.jsonl`
   * `/root/workspace/Uni-Poly-Plus-master/data/processed/mips_trimer_scage/builds/ece3a6d6cf6f73f76260afea10ef44f62f38c7b065bdfd672b26dcea9104fb11.contract-blocked-ru-build-boundary/ru_base/data.lmdb`
   * `/root/workspace/Uni-Poly-Plus-master/data/processed/mips_trimer_scage/builds/ece3a6d6cf6f73f76260afea10ef44f62f38c7b065bdfd672b26dcea9104fb11.contract-blocked-ru-build-boundary/topology/data.lmdb`
   * `/root/workspace/Uni-Poly-Plus-master/data/processed/mips_trimer_scage/builds/ece3a6d6cf6f73f76260afea10ef44f62f38c7b065bdfd672b26dcea9104fb11.contract-blocked-ru-build-boundary/source/records.jsonl`
   这些目录没有 `.done`/`.frozen`；各自的 `source/manifest.json`、`ru_base/metadata.json`、`topology/metadata.json`、`rejections.jsonl`（存在时）不删除。

**KEEP（本次不处理）**：active `store.json` 及其 bundle、root-level `topology`/`trimer`/`ru_base`（W-CAMR 与旧读取链仍引用）、当前 PI1M/downstream cohort、`dual_static_v1`、`pretrain_targets_v1`、`periodic_line_glt_v1`、`periodic_line_glt_image_v1`、`periodic_line_glt_distill_v1/v2`、MD200、所有当前/保留路线代码、配置、测试、checkpoint、results 与 logs。

**HOLD（不因名称删除）**：`src/modules/mts_glt_distill.py`、`src/modules/periodic_line_glt_v3.py`、`src/modules/atomic_point_encoder.py`、`src/training/w_camr_v2_support/` 及其入口；`PIPELINE.md` 和 import/测试审查证明它们仍属于保留路线。所有未列出的历史缓存、结果和临时目录也保持 HOLD。

## 4. 代码精简：先解除依赖，再删除退役实现

**当前状态：未实施，HOLD。** 以下为原候选方案，不是已完成工作或待自动执行的删除指令。第 3.4 节中的保留结论仅说明首批不能删除；存在历史 import 不等于永久生产依赖。若继续精简，先按第 7.3 节重新分类，不能直接照本节名称清单删除。

### 保留入口及能力

保留 `pretrain_glt_dual.py`、`finetune_glt_dual.py`、正式 grid／aggregate、六 checkpoint／失稳诊断入口、`glt_dual_*` 当前配置（包括 geonorm）、对应 tests。保留完整 Trimer 构建、缓存 store/lifecycle、cohort/union/static/targets 生产与核验、固定 split 生成和 Stereo 审计入口。

核心保留模块包括 `glt_dual*`、`glt_bond_chemistry`、`mips_local_graph`、`original_mips_knowledge_fusion`、`canonical_periodic`、`graph_data`、`periodic_line_glt_complete`、`trimer_mcl` 和当前缓存契约。名称含 mips/scage/v3 的文件仍可能是当前依赖，禁止名称式删除。

### 初始退役候选

* 配置：`atomic_point*.json`、`glt_distill_n_plus_*.json`、`glt_distill_repair_c*.json`、旧 C0/no-MD 独立路线配置。
* 独立入口：`pretrain_mts_glt_distill.py`、`run_mts_glt_distill*`、`build_mts_glt_distill*`、`smoke_mts_glt_distill*`、旧 C0 pipeline/probes/staged-finetune/report、`run_original_mips_atomic_pc_w_camr_v2.py`、旧 GLT-v3 sidecar 构建。
* 专用实现候选：`mts_glt_distill`、`periodic_line_distill*`、`periodic_line_glt_image`、atomic-point／original_mips_atomic_pc 系列、`w_camr_v2_support`，以及确无共享用途的 MD200 组件。
* 专用 tests：旧蒸馏／atomic_pc／C0-only 测试可随被删行为退役；identity、stereo、138维输入、mask、周期关系、缓存生命周期和恢复测试即使叫 mips 仍保留。

逐个候选检查 Python import（包含 package `__init__`）、动态 import、CLI/config 路由、字符串类名、tests 与序列化对象依赖；输出具体文件 allowlist，不以模糊通配符执行删除。

r1 静态审查记录的耦合须在执行前按当前源码复核，并先处理：

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

用户授权实际清理后，先在当前机项目目录检查工作树、分支及 origin，按 AGENTS.md 执行对应分支的 `git pull --ff-only`（当前分支为 `dev`），成功后重读代码／Plan.md／本文件并核对活动进程。不得为同步清空用户改动或覆盖当前执行者记录。清理准备好接管时，由 Codex 将获准阶段写入 Plan.md，明确其与原科学任务的交接，不让两个执行者同时修改同一文件。

在无运行者使用相关文件的窗口中完成依赖闭包和逐路径 allowlist，Codex 审查具体清单后才进入破坏性删除；不把本次计划编写当作删除授权。若科学路线已经变化，更新本计划，不能沿用过期 keep 集合。

### 阶段 1：精简 tracked 代码

保护未提交改动；只对已核准路径执行 Git 删除与必要的最小依赖解耦。每批做相关导入／CLI help、单元测试、缺引用检查，成功后单独 commit。保留可回退基线 commit，不 rewrite history。

### 阶段 2：删除核准的旧缓存

原执行要求是在当前机 `tmux` session `Uni-Poly` 新建唯一命名窗口，工作目录固定为 `/root/workspace/Uni-Poly-Plus-master`；首批实际窗口及日志已列在下方执行记录中。不得重启或重放这些删除命令。以后获准的新批次使用新名称，日志记录主机、基线 commit、实际命令、清单、逐项退出码及删除前后磁盘计量，不仅记录“命令已提交”。

仅用当前 Bash／Linux 工具按精确路径逐项处理，禁止对 `data/processed` 或混合根目录做递归通配删除。每项删除前重新验证 realpath 留在批准根内、不等于项目根或保护目录及其祖先、不跨符号链接／挂载点、不被进程占用、身份未变化。发现新引用则该项 HOLD，不擅自修改消费者以让删除通过。不得根据 `/dev/nvme1n1p1[/docker_home/dzw2]` 推导容器外删除路径；只能操作本计划核准的项目内路径。

旧 LMDB、旧分片可以在这些条件全部满足后直接释放，不强制复制百 GiB 数据到同盘。被保留的失败报告／manifest 不放在即将整删的目录里，先存入稳定审计产物目录并记录来源。

### 阶段 3：最小验收与交付

* 生产构建、双路预训练／微调／聚合／诊断入口可导入与解析参数；缓存 builder 能在独立临时目录运行相关合成契约测试，不触碰生产缓存。
* 当前 store、bundle、cohort、static/targets 的身份、记录数与文件存在性保持不变；不改变 `.frozen`、source key 顺序和 split。
* 用既有最多两条真实 fixture（普通与 N=0，如不存在如实说明），两融合 eval 前后向及部署加载验证；与删除前同 checkpoint／同输入输出比较。测试固定 RNG，预算不扩大到 optimizer 更新或正式训练。
* 聚合复算仍得到原 macro8；若只改无关文件且没有影响汇总依赖，可复用已通过证据，不机械跑全仓或 80 folds。
* 报告当前机逐项删／留／HOLD、删除前后文件系统可用空间差额、剩余依赖、运行命令及验证结果。若有并发写入、共享存储变化或已 unlink 但未关闭的 fd，说明差额不能精确归因于本轮删除，不把候选尺寸直接当成释放量。更新 PIPELINE.md 当前入口与退役说明，不擦除 RESULTS.md 历史结论；完成后归档本轮完整清理周期。
* 按 AGENTS.md 显式提交本轮代码与文档，fetch 后检查待推送提交，推送当前同名分支并核对 GitHub origin 包含该 commit；不上传缓存、checkpoint 或大型日志。分别报告“当前训练机文件系统清理结果”和“Git 代码／文档同步结果”，不声称清理了用户个人电脑或任何其他机器。

### 阶段 2/3 实际执行记录（2026-09-16，Codex）

* 用户于本轮明确授权直接执行；执行前已在 `dzw2`、`/root/workspace/Uni-Poly-Plus-master` 核对 `dev`、origin 和活动任务，并执行 `git pull --ff-only origin dev`（`Already up to date`）。预检时 HEAD 为 `5a8fe0fa2d24696a36ae7bcb311710fa12cbf9bb`；仅旧监视 shell 存活，未发现预训练、微调、缓存 builder 或候选路径的 writer/fd/mmap。预检完整日志：`logs/cleanup_20260916_01/preflight_audit.log`。
* 按 3.4 的绝对路径 allowlist，在 `tmux` `Uni-Poly:cleanup_20260916_01_delete` 中执行删除，命令和逐项校验日志为 `logs/cleanup_20260916_01/delete.log`。实际删除 4 个 PI1M pilot、`cohort_30f17b59bc5862a1_v2`、`periodic_line_glt_distill_v2.parts`，以及两个 blocked build 的 6 个大 payload；失败 build 的 manifest、metadata、rejections 和 writer-lock provenance 保留。未删除 active bundle、root-level 读取链、active cohort/static/targets、代码、配置、测试、checkpoint、results 或历史 logs。
* 初次删除脚本的保护路径断言错误地假定 bundle 根有 `.frozen`，因此末尾返回 `1`；删除和逐项 absent 校验已完成。随后在 `tmux` `Uni-Poly:cleanup_20260916_01_delete` 的 postverify 命令按实际三层 artifact 路径重跑，日志 `logs/cleanup_20260916_01/postverify.log`，所有删除目标均 `ABSENT`，active PI1M/downstream 链、六个 `.frozen`、store 和失败 provenance 均 `RETAIN_PRESENT`。这不是数据删除失败，但应在后续脚本修订中改正路径断言。
* `periodic_line_glt_distill_v2.parts` 中 16 个 `part_*.log` 已复制到 `logs/cleanup_20260916_01/preserved_distill_v2_parts/`；日志与 ignored 数据不纳入 Git。文件系统从 `Used=708445024256`、`Available=3028318564352` 变为 `Used=695758565376`、`Available=3041005023232`，可用空间增加 `12,686,458,880` bytes（约 11.81 GiB）。该值是文件系统差额，不把候选 `du` 之和当作释放量。
* 最小只读验收先在 `tmux` `Uni-Poly:cleanup_20260916_01_accept` 执行，完整日志 `logs/cleanup_20260916_01/acceptance.log`：保留代码 import 与 7 个 CLI `--help` 均成功，路径／冻结标记检查成功；首次聚焦测试为 `53 passed, 1 failed, 1 warning`。唯一失败为 `tests/test_cache_lifecycle.py::test_downstream_geometry_fallback_keeps_complete_identity_carrier`：测试调用 `select_record_fields("trimer", data)` 漏传可选 `trimer_failure_code`，而生产 fallback 写入路径已显式传递该字段。
* 在修改前再次 `git pull --ff-only origin dev`（`Already up to date`），仅对该测试调用补上 `optional_fields=("trimer_failure_code",)`；没有修改生产生成器、缓存 schema 或科学定义。随后在 `tmux` `Uni-Poly:cleanup_20260916_01_recheck` 重跑同一 6 个目标文件，日志 `logs/cleanup_20260916_01/acceptance_recheck.log`，结果 `54 passed, 1 warning`，exit code `0`。
* 两个正式 deploy 包的只读检查在 `tmux` `Uni-Poly:cleanup_20260916_01_modelcheck` 完成，日志 `logs/cleanup_20260916_01/modelcheck.log`，新报告为 `results/cleanup_20260916_01/deploy_validation/{concat,kfuse}.json`。使用 active PI1M cohort 的真实索引 `[20, 9]`（普通明确 E/Z 与真实 `*O*` N=0），Concat/KFuse 均通过检查；脚本顶层键沿用历史名称 `FIXED_CONCAT_DEPLOY_VALID=YES`（KFuse 报告的 `fusion_mode` 明确为 `kfuse`），strict load、187 个 encoder tensor 的 resume bitwise identity、全有限、预测形状 `[2,1]`、cache zero-write 均通过；本轮没有 optimizer 更新或写入生产缓存。
* 未执行全量数据读取、Stereo 全量扫描、预训练、微调、聚合重算或缓存重建；既有正式 checkpoint／聚合和历史 backward smoke 证据未被删除，active 输入未改变，按阶段 3 的复用条款保留。此次新增的真实记录验证只读 forward，不把它称作新的训练性能结果。

## 7. r3 审查结论与证据收口

### 7.1 审查结论：首批清理有限范围通过，证据补记已完成

此前只读复核确认 12 个精确目标均不存在，active bundle／六个层级 `.frozen` 仍在，PI1M static／targets manifest hash 与原记录一致，16 个分片日志副本保留。既有日志记录 `54 passed, 1 warning`，Concat/KFuse 两份部署报告记录 strict load、187 个 encoder tensor 与 resume 逐位一致及真实双记录有限 forward。11.81 GiB 是首批日志中的文件系统差额，不是 r3 新释放空间。

这支持“首批清理未发现破坏当前读取链”，不证明整个项目精简完成，也不证明所有缓存 payload 或科学质量均通过全量审计。首次删除 wrapper 的 `.frozen` 路径断言错误保留在记录中；postverify 已覆盖真实路径，不为纠正日志重新执行删除。

|验收项|已存在证据／当前边界|本次执行／复用证据／未执行或未核实|
|-|-|-|
|删除目标与保留路径|`logs/cleanup_20260916_01/postverify.log`，且此前只读复核目标状态一致|本次只读确认删除目标仍为 `ABSENT`、active 六层 `.frozen`／store／cohort／static／targets 仍为 `RETAIN_PRESENT`；不重复删除。该证据是存在性检查，不是全量内容一致性证明。|
|局部回归|`logs/cleanup_20260916_01/acceptance_recheck.log`：54 passed，exit 0|日志没有保存完整 argv／环境。执行记录可复述的命令为 `PYTHONPATH=.:tests pytest -q tests/test_glt_dual_cache.py tests/test_glt_dual_static.py tests/test_glt_dual_diagnostics.py tests/test_frozen_cache_store.py tests/test_cache_lifecycle.py tests/test_aggregate_glt_dual_finetune.py`，cwd 为项目根、tmux 为 `Uni-Poly:cleanup_20260916_01_recheck`；六文件列表和 `PYTHONPATH` 属于执行上下文补记，不能当作日志中已持久化的独立命令证据。结果为 54 passed、1 DeprecationWarning、exit 0；本次不重跑。|
|部署加载与真实 forward|`results/cleanup_20260916_01/deploy_validation/concat.json`、`kfuse.json`；普通明确 E/Z 与真实 `*O*` N=0，输出 `[2,1]` 有限|`modelcheck.log` 未保存完整 argv；执行记录可复述为分别调用 `scripts/validate_glt_dual_deploy.py`，输入对应 `deploy_05000.pt`／`resume_05000.pt`、`--expected-step 5000`、`--fusion-mode concat|kfuse`、`--cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1`、`--cache-root data/processed/mips_trimer_scage`、`--index 20 --index 9 --device cpu` 及对应 `--report-json`，cwd 为项目根、tmux 为 `Uni-Poly:cleanup_20260916_01_modelcheck`。报告确认 strict load、187 tensors、resume bitwise identity、finite、`[2,1]` forward、cache zero-write；脚本源码在 `scripts/validate_glt_dual_deploy.py:130-145` 明确用 `build_dual_sample` 且 `torch.no_grad()`，因此这是既有只读 forward，不是新增 backward、训练或性能结果。|
|static／targets 消费路径|部署脚本没有 `--dual-static-root`／`--pretrain-target-root`，不能据此声称清理后重新消费该路径|复用既有 PI1M `run.json`：`results/glt_dual_static_pretrain_5k_{concat,kfuse}/run.json` 绑定 sample_count `959588`、cohort `b03f96a14c2...`、main bundle `30f17b59bc...`、dual static `9ff122cc16...`、targets `5e7b5ec8...`；`results/glt_dual_static_smoke4_{concat,kfuse}/run.json` 亦绑定同一身份。下游 `results/glt_dual_static_finetune_formal_grid/{concat,kfuse}/eat_fold0/run.json` 与 `results/glt_dual_static_finetune_smoke_auto_retry/{concat,kfuse}/run.json` 绑定 downstream dual-static 与 split。当前 active `.frozen` 与这些身份一致；清理后没有再次运行 static/targets consumer，故只能复用未受清理目标影响的历史身份，不能称为本次重新验证。|
|backward 与删除前后预测 parity|本次部署脚本使用 `torch.no_grad()`；resume tensor 一致不等于预测 parity|本次未执行 backward，也没有同一输入的删除前／删除后预测差分报告。`resume_identity.bitwise_identical=true` 仅证明 deploy 与 resume encoder tensor 一致；不外推为输出 parity。历史 backward／静态路径证据未被删除，但未在本清理周期复核，当前项标为未执行／未独立核实。|
|聚合结果|本轮未重算，生产代码与正式结果未删除|复用 `results/glt_dual_static_finetune_formal_grid/{concat,kfuse}/comparison_review_20260916T001828Z/summary.json`（各 `status=PASS`、8 tasks、40 folds，macro8 分别 `0.7877379364475444`／`0.7695629052761048`）及 `results/glt_dual_static_finetune_formal_grid/summary.json`（80 shards、0 failures）。另有旧 fixed Concat 7-task 报告 `results/glt_v2_fixed_concat_5k_20260916/aggregation_review_7task/summary.json`，其 scope 仍为 7 tasks；没有因本次清理新增 folds 或训练。|

### 7.2 下一步 A：补齐现有证据与文档收口（已完成）

本次由 Codex 在 `dzw2` 只读完成；没有独立执行者复核，也没有新增运行授权。

1. 读取 `acceptance_recheck.log`、`modelcheck.log`、`postverify.log`、active manifests／`.frozen`、相关 pretrain／finetune `run.json` 与聚合 summary；日志未保存的完整 argv／env 保留为缺失，不以新运行补造。
2. 已按 7.1 表逐项记录本次存在性证据、可复用身份和未执行项。static/targets、backward、parity 没有由有限 forward 代替。
3. 已记录 `cohort_30f17b59bc5862a1_v2` 原 manifest 的已知溯源损失：仅保留删除前 SHA256 `9eba93d60781361b005800f98d68ff8eba04a888fadf8b328915e147f1d23945` 与 `results/glt_dual_readiness_20260915/phase2_cohort_v2.json`；原文不可还原，不影响正式 cohort 身份，也未生成缓存补档。
4. 已同步 `Plan.md`、`PROJECT_HISTORY.md`，并修正 `PIPELINE.md` 对清理命令可追溯性的表述；`RESULTS.md` 未新增结果。

预算：只读代码／元数据／现有日志与必要文档修订；**零新增删除、零模型测试、零 GPU／worker、零训练、零构象生成／缓存重建**。不扫描全量 LMDB、不修改生产 schema、不新增通用清理框架。预计超过一分钟的只读检查仍遵守 tmux 与日志规则。

停止条件：发现产物身份不匹配、相关材料正在被另一执行者改写，或需新实验才能补证据时，停止对应补记并报告。缺失材料保留为缺失，不扩大预算。

完成标准：命令来源和验收边界可追溯，缺失项显式列出，Codex 明确接受有限范围结论或给出具体返修项；不得仅因文档写完就宣称全部验证已覆盖。

### 7.3 下一步 B：是否继续项目精简（待授权，不自动执行）

只有用户仍希望推进原代码精简目标时，才开始下一批只读依赖审查。先将剩余路径分为“当前生产必需／历史复现保留／仅旧入口引用／尚不明确”，记录实际消费者、保留理由和退出条件；这些是用途分类，不改变现有 KEEP／HOLD 状态。

存在 import 或历史 PIPELINE.md 描述只能作为依赖线索，不能单独证明旧路线必须永久保留；反之，名称旧也不能证明可删。产出精确候选清单交 Codex 审查，涉及退役哪些路线或放弃原地复现时交用户决定。代码解耦和实际删除须有对应授权，不由本节自行启动。

缓存构建恢复机制、静态存储优化和化学审计问题另立范围，不混入本轮清理。未获新增授权前，首批之外的缓存、代码、结果与临时目录继续 HOLD。

### 7.4 r3 证据补记执行记录与边界（2026-09-16，Codex，只读）

* 基线核对：`dzw2`、`/root/workspace/Uni-Poly-Plus-master`、`dev`，HEAD `248371d`，工作树干净；修改前 `git pull --ff-only origin dev` 返回 `Already up to date`。tmux `Uni-Poly` 没有运行中的 pretrain／finetune／builder／pytest；仅有一个旧监视 shell 的轮询命令，未发现其目标进程，本次未接管或终止。
* 只读来源：上述清理日志、现有部署报告、PI1M/downstream active manifest 与 `.frozen`、pretrain／finetune `run.json` 及 aggregation summary；另静态查看 `scripts/validate_glt_dual_deploy.py` 的 static 参数缺失与 `no_grad` 调用。没有全量 LMDB 读取、Stereo 扫描、缓存重建或代码调查。
* 命令追溯边界：`acceptance_recheck.log` 与 `modelcheck.log` 只保存结果和退出码，没有完整 argv／环境；7.1 所列命令是本次执行上下文可复述的调用，已明确标注为非独立日志证据，不将缺失包装成“日志完整”。
* 可复述的实际调用（不是日志原文，因原日志未保存 argv／环境）：

  ```text
  PYTHONPATH=.:tests pytest -q tests/test_glt_dual_cache.py tests/test_glt_dual_static.py tests/test_glt_dual_diagnostics.py tests/test_frozen_cache_store.py tests/test_cache_lifecycle.py tests/test_aggregate_glt_dual_finetune.py

  python scripts/validate_glt_dual_deploy.py --checkpoint results/glt_dual_static_pretrain_5k_concat/deploy_05000.pt --resume results/glt_dual_static_pretrain_5k_concat/resume_05000.pt --expected-step 5000 --fusion-mode concat --cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1 --cache-root data/processed/mips_trimer_scage --index 20 --index 9 --device cpu --report-json results/cleanup_20260916_01/deploy_validation/concat.json

  python scripts/validate_glt_dual_deploy.py --checkpoint results/glt_dual_static_pretrain_5k_kfuse/deploy_05000.pt --resume results/glt_dual_static_pretrain_5k_kfuse/resume_05000.pt --expected-step 5000 --fusion-mode kfuse --cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1 --cache-root data/processed/mips_trimer_scage --index 20 --index 9 --device cpu --report-json results/cleanup_20260916_01/deploy_validation/kfuse.json
  ```

  第一条在 `Uni-Poly:cleanup_20260916_01_recheck`、后两条在 `Uni-Poly:cleanup_20260916_01_modelcheck`，cwd 均为项目根；上述调用由执行记录复述，不能替代缺失的持久化命令日志。
* 身份复用边界：active PI1M cohort `.frozen`=`b03f96a14c2beb1d987743886cfeb2b738d94b27897f9107400d573f048d476b`、dual static=`9ff122cc16df5c869582ee2fa07b6f42fa44f2a228f554d25412eb2ae6d6006d`、targets=`5e7b5ec8e5f96bf695494436cd471c9d44a85d99571db88adf34c7d5f5777482`；downstream cohort=`0cf28f5ee54886c4bc17a3f2d15fb824d4a82dbe801aae2095920a0cfaed4b30`、dual static=`0ae172f88574c3beec0406ade84c96755681e32158b49bd385ffe862a29d3df7`。这些与保留 `run.json` 一致；deploy 只读检查本身未消费 static/targets。
* 结论：首批删除目标仍不存在，active 读取链及历史聚合产物未被清理破坏；54 项局部回归和两份 deploy 结果属于已有证据。backward、删除前后预测 parity、清理后 static/targets 重新消费、聚合重算、全量读取和科学质量审计均未执行／未独立核实。该结论不等于整个项目精简完成。

### 7.5 r3 文档交付与后续边界

本次已完成第 7 节 A 的只读证据补记、`Plan.md` 清理交接同步及同一周期 `PROJECT_HISTORY.md` 更正，并修正 `PIPELINE.md` 中对完整命令日志的错误暗示；未修改生产代码、缓存、checkpoint 或历史结果。第 7.3 节的代码精简和其余候选仍为待授权／HOLD；若用户继续处理，必须另行做当前依赖闭包和精确 allowlist，不由本计划自动删除。
