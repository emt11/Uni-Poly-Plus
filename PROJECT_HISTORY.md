# Uni-Poly 项目变更周期归档

## DOCS-20260916-04｜GLT-V2 专项清理计划（仅规划）

用户授权编写 Plan_Delete.md，未授权实际删除。Codex 先 pull，保留用户已有空白未跟踪文件，再只读核对本地依赖、远端 store/manifest/run.json、du 尺寸和活动进程，形成清理专项计划。

实际交付：明确当前双路与 geonorm 保留范围；固定正式 PI1M/downstream bundle、cohort、static/targets 保留链；将旧根级缓存和 sidecar 列为条件候选，显示尺寸约 119 GiB、实际释放量待核；规定先解除 package／dataset／utils 共用依赖，再精简专用模块。发现 geonorm_5k 正在运行，执行期禁止删除／搬移其依赖。

验证为文档静态自检及 git diff --check，无模型测试、训练、缓存生成或删除；详细清单和预算保存在 Plan_Delete.md。未覆盖当前 Plan.md，未给清理执行授予新权限。下一步是在科学任务结束后形成精确删除 allowlist、由 Codex 审查并取得实际清理授权；本记录完成的是规划文档周期，不代表清理周期完成。

本文件保存每轮“规划 → 执行 → 审查”的实际记录。当前任务在 [Plan.md](Plan.md)，协作规则在 [AGENTS.md](AGENTS.md)。按周期关闭时间追加；不以归档内容授权新任务，不追溯编造旧记录。

每轮保存最终实施计划和重要修订、执行摘要、验证证据、审查结论与下一步。仅链接当前 Plan.md 不足以归档；大段日志／diff 使用稳定路径或 commit 引用。未完成但被取消／替换的周期注明实际状态；历史更正以带日期的补充记录追加。

---

## DOCS-20260916-01｜协作分工与周期归档机制

* 日期：2026-09-16（Asia/Shanghai）。
* 状态：文档修改完成；Codex 自检，未进行独立审查。
* 规划／执行／自检：Codex。用户明确要求本次由 Codex 修改文档，属于默认分工的明确例外。
* 基线：`ff00919`；开始时 `AGENTS.md` 已有用户将 Claude Code 改为 ZCode 的未提交修改，已保留。`Plan.md` 为零字节。本周期未创建 commit。

### 规划与授权

用户要求优化 AGENTS.md：Codex 负责规划与审查、ZCode 负责具体执行；使用 Plan.md 交互；每轮完整周期存入根目录一个文档；审查后提供下一步规划。用户同时说明 GLT-V2 诊断计划正在执行，不要求重新启动或改变预算。

最终实施计划：

1. 保留已有角色替换，明确职责、交接内容、修订、授权延续和审查后下一步。
2. 以 Plan.md 为唯一当前计划入口，以 PROJECT_HISTORY.md 为周期归档；归档后才替换当前计划。
3. 将已交付的 GLT-V2 计划录入空白 Plan.md，保留原验证与一次最多 800 updates 的回放预算，状态仅按用户报告记录。
4. 只检查文档与 diff，不启动模型、测试、缓存构建或远端任务。

### 实际执行

* `AGENTS.md`：增加 Codex／ZCode 职责、Plan.md 四部分内容与状态、先归档再写下一步、文档例外与单一执行者规则；同步授权和交付条款。
* `Plan.md`：录入 `GLTV2-20260916-01/r1` 的结果汇总、数据来源核对、checkpoint 诊断、有限回放及优化决策计划；执行记录与审查明确待补记。
* `PROJECT_HISTORY.md`：建立单文档追加归档机制，保存本周期的完整摘要。
* 未修改模型、训练代码、配置、PIPELINE.md 或 RESULTS.md；未接管当前 ZCode 运行。

### 验证与审查

* 只读检查：开始时执行 `git status --short --branch`、`git diff -- AGENTS.md`，阅读当前 AGENTS.md、Plan.md 并确认 Plan.md 长度为 0。
* 静态自检：核对文档相对链接、角色分工、授权边界、执行中状态、归档顺序和回放预算；检查最终 diff 与 `git diff --check`。
* 模型／测试／训练／缓存任务：未执行。远端当前进度：未核实。
* 结论：协作与归档文档交付；不代表 GLT-V2 诊断完成或科学验收通过。实际执行者尚需补齐本轮进度。

### 下一步

当前科学计划 `GLTV2-20260916-01` 保持执行中，不归档为完成。ZCode 先在 Plan.md 补记已执行步骤、活动任务与证据，继续原授权范围；Codex 收到交付后审查、归档该周期，再制定下一步计划。此文档周期与正在运行的科学周期相互独立。

## DOCS-20260916-02｜每轮 Git 提交与远程同步

* 授权与规划：用户要求每次对话产生 Git 修改后，提交本地并同步至 `https://github.com/emt11/Uni-Poly-Plus`。本轮由 Codex 按明确文档授权实施并自检，未作独立审查。
* 实施：AGENTS.md 新增第 8 节，规定本轮修改显式暂存、提交和推送当前对应分支；保护无关修改，核对远端进展，不强推，并区分本地提交、远程同步和科学验收。
* 验证：开始时工作树干净，当前分支 dev，origin 指向用户指定仓库；文档变更进行 diff 与空白检查。未运行模型、测试或训练。
* 交付：本条归档与规则一同提交；实际提交哈希与远程同步结果以 Git 历史及本轮最终回复为准，不在提交前预称推送成功。
* 下一步：继续现有 GLT-V2 计划，新规则适用于后续执行与文档交接；不改变当前科学计划或预算。

## DOCS-20260916-03｜修改前拉取远程更新

* 授权与规划：用户要求每轮修改开始前必须 pull 更新本地；Codex 负责本次文档修改与自检。
* 执行：先核对工作树干净、dev 分支及 origin 地址，执行 `git pull --ff-only origin dev`，结果为 Already up to date；随后修改 AGENTS.md 第 8 节，规定先拉取、重读文件再修改，以及本地改动保护和同步失败处理。
* 验证与审查：检查文档 diff 和 `git diff --check`；本轮仅文档自检，未执行模型、训练或测试，不影响当前科学计划。
* 交付与下一步：本条与规则一起提交推送，实际哈希及同步结果见 Git 历史和最终回复。后续执行统一遵循“pull → 修改与验证 → commit → push → 核对远端”。

## DOCS-20260916-05｜清理计划适配当前远程训练机

* 日期：2026-09-16（UTC）。规划／文档执行／自检均为 Codex；无独立审查。用户明确要求优化当前机上的 `Plan_Delete.md`，未授权实际清理。
* 基线：`43f83f7a001fd46cd5383f2ca711e81c6ff595db`，`dev` 分支；开始时工作树干净，origin 为 `emt11/Uni-Poly-Plus`。修改前执行 `git pull --ff-only origin dev`，返回 Already up to date，随后重读文件。
* 最终实施计划与修订：`CLEANUP-20260916-01/r1 → r2`，只做环境适配。唯一清理执行环境为当前 Linux 主机 `dzw2`、项目根 `/root/workspace/Uni-Poly-Plus-master`；去掉 Windows 和两端分别清理假设，明确 GitHub origin 不备份 ignored 数据。保留当前生产 bundle／cohort／static／targets 与科学证据链，原约 119 GiB 候选容量及未重新审计的引用标为 r1 记录，不当成实时验收结果。
* 实际变化：`Plan_Delete.md` 增加主机／解释器／文件系统核验、活动任务快照、只读复核命令、当前机独立 tmux window／日志约定、逐路径授权与单机验收口径；`Plan.md` 仅增加本次文档交接注记，不替换科学计划或改变实验预算；本文件追加本周期记录。未修改模型、配置、PIPELINE.md 或 RESULTS.md。
* 只读证据：`hostname`、`pwd -P`、Git 状态／分支／remote、`tmux list-windows -t Uni-Poly`、`ps -eo pid,ppid,etime,args`、store 元数据、`findmnt -T`／`df -h`。09:11 UTC 的 `geonorm_5k` 训练与 worker 仍存活，实际命令读取受保护的 PI1M 数据链，日志为 `logs/glt_dual_static_pretrain_concat_geonorm5k.log`；未核实训练完成或所有候选 fd/mmap，不把进程存活称为训练健康证明。
* 验证预算与结果：文档 diff／`git diff --check`、旧平台假设与关键路径静态检查；没有模型测试、GPU 任务、全量 LMDB 扫描、缓存生成／搬移／删除，没有创建清理窗口。文档自检不等于清理执行或科学验收。
* 交付与下一步：本轮完成的是适配文档周期，实际提交及 push 结果见本条对应 Git 历史和最终回复。清理周期仍待授权；后续由 ZCode 在相关任务自然结束并交接后补齐精确 allowlist，Codex 审查后才可进入获准的清理阶段。未核实项 HOLD，不追加实验或终止现有任务。

## CLEANUP-20260916-01/r2｜当前训练机精确缓存清理执行

* 日期：2026-09-16（UTC）；主机 `dzw2`；项目根 `/root/workspace/Uni-Poly-Plus-master`；分支 `dev`；规划、执行和本轮审查均由 Codex 完成。用户明确授权执行 `Plan_Delete.md`，并说明现有任务已结束。
* 基线与同步：执行前核对 `git status`、branch、origin、tmux、进程和文件系统，并运行 `git pull --ff-only origin dev`，结果 `Already up to date`。删除前 HEAD 为 `5a8fe0fa2d24696a36ae7bcb311710fa12cbf9bb`；没有回滚或覆盖用户改动。
* 实施范围：依照 `Plan_Delete.md` 3.4 精确 allowlist 删除 4 个 PI1M pilot（dual_static／pretrain_targets）、`cohort_30f17b59bc5862a1_v2`、`periodic_line_glt_distill_v2.parts`，以及两个未冻结 blocked build 的 6 个大 payload。保留失败 manifest／metadata／rejections／writer-lock provenance、16 个分片日志副本、active bundle 与 root-level 读取链、active cohort/static/targets、代码／配置／测试／checkpoint／results／logs。没有删除 tracked 代码或科学产物。
* 日志与窗口：预检 `Uni-Poly:cleanup_20260916_01`／`logs/cleanup_20260916_01/preflight_audit.log`；删除 `Uni-Poly:cleanup_20260916_01_delete`／`delete.log`；修正后的 postverify／`postverify.log`；验收 `Uni-Poly:cleanup_20260916_01_accept`／`acceptance.log`。所有窗口均独立于训练 pane；未启动 GPU、worker、模型、训练或缓存构建。
* 结果：删除目标 postverify 全部 `ABSENT`，active 六个 `.frozen`、store、PI1M/downstream cohort/static/targets 和失败 provenance 全部存在。文件系统 `Available` 从 `3028318564352` 增至 `3041005023232` bytes，增加 `12,686,458,880` bytes（约 11.81 GiB）；该数值按文件系统差额记录，不冒称为候选目录大小总和。
* 验收：保留代码 import、7 个 CLI `--help` 和 active 路径检查成功。首次聚焦测试为 `53 passed, 1 failed, 1 warning`；唯一失败是测试调用漏传可选 `trimer_failure_code`。在再次 `git pull --ff-only origin dev` 后，仅对 `tests/test_cache_lifecycle.py` 的该调用补上 `optional_fields=("trimer_failure_code",)`，没有修改生产生成器、缓存 schema 或科学定义；同一 6 个目标文件在 `cleanup_20260916_01_recheck` 中为 `54 passed, 1 warning`，exit code 0。
* 模型只读验收：`cleanup_20260916_01_modelcheck` 对 Concat/KFuse 两个 step-5000 deploy 使用 active PI1M cohort 的真实索引 `[20, 9]`（普通明确 E/Z 与真实 `*O*` N=0）。两者均 strict-load、187 个 encoder tensor 与 resume bitwise identical、全有限、预测 `[2,1]`、cache zero-write，报告分别为 `results/cleanup_20260916_01/deploy_validation/concat.json` 与 `kfuse.json`；脚本顶层键沿用历史 `FIXED_CONCAT_DEPLOY_VALID` 名称，KFuse 报告内 `fusion_mode` 明确为 `kfuse`；不执行 optimizer 更新或写生产缓存。
* 过程偏差：删除脚本首次错误地把 bundle 根当作 `.frozen` 路径并返回 1；实际删除与逐项校验已完成，随后按三层 artifact 的真实 `.frozen` 路径复核通过。该脚本断言属于后续可修复的工具问题，不影响本次已核准路径的删除结果。
* 审查结论：清理动作、删除目标 postverify、修正后的局部回归和两融合真实双记录 deploy 只读检查均完成；未发现 active cache identity、读取链或 writer 冲突损坏。未执行全量读取、Stereo 全量审计、预训练、微调、聚合重算或缓存重建；既有正式与历史 backward smoke 证据未被删除，不能把本周期只读 forward 当作性能结果。
* 下一步：本清理周期无继续执行项。若要处理其他 HOLD 候选或重新生成已删除的 pilot/staging，须另行授权并重新做依赖闭包；被删除 ignored 缓存不由 Git 恢复，`cohort_30f17b59bc5862a1_v2` 原 manifest 内容未保留，仅保留删除前 SHA256 与 phase2 provenance。

### 2026-09-16 更正｜CLEANUP-20260916-01/r3 审查后续计划

* 用户明确要求将下一步写入 Plan_Delete.md；本次规划／文档执行／自检均为 Codex，不构成独立审查。文档基线 `4c3416e3499a91acdff7853dea6801ab851024e9`，dev 工作树干净，修改前 pull 为 Already up to date。实际 commit／push 结果见对应 Git 历史及交付回复。
* 更正上条“无继续执行项”的范围：首批缓存删除已完成，原代码精简未实施；仍需补记现有测试准确命令与六文件列表，以及 static／targets、backward、预测 parity、聚合的适用旧证据或未核实状态。部署报告证明 strict load、resume tensor 一致及有限 forward，不替代上述全部验证。原首次失败及 11.81 GiB 历史空间差额保留。
* 最终计划：同一 ID 升 r3，ZCode 后续只读追溯现有日志／命令并补齐验收对应表，Codex 复核后收口；找不到材料就保留缺失，不靠新测试补造历史。已删除 cohort_v2 原 manifest 仅留 hash／摘要，记录溯源损失，不自动重建。其余代码和缓存继续 HOLD，若用户仍需精简，另行授权用途分类和精确依赖审查，实际删除不自动启动。
* 实际修改：Plan_Delete.md 修订计划头、历史快照／阶段边界并新增第 7 节交接任务、预算、停止条件和完成标准；Plan.md 仅更新清理注记；本文件追加同周期更正，不重复归档首批删除，不改写原执行结论。未修改代码、缓存、PIPELINE.md 或 RESULTS.md。
* 验证与状态：仅文档 diff、引用及 `git diff --check` 自检；没有运行模型、测试、训练或清理，没有执行新依赖调查。本轮完成的是下一步计划落盘，证据补记仍待执行，不能把计划交付写成整个精简项目完成。

### 2026-09-16 更正｜CLEANUP-20260916-01/r3 证据补记完成

* 授权与角色：用户要求继续当前 `Plan_Delete.md`；本次由 Codex 在 `dzw2:/root/workspace/Uni-Poly-Plus-master` 只读追溯第 7 节 A 并修订交接文档，不新增删除、代码、模型测试、训练或缓存重建。没有独立执行者复核，故不将本次自检称为独立审查。
* 基线与活动任务：修改前工作树干净，分支 `dev`，HEAD `248371d`，`git pull --ff-only origin dev` 为 `Already up to date`。tmux `Uni-Poly` 没有运行中的 pretrain／finetune／builder／pytest；仅旧监视 shell 的轮询仍在，本次未接管或终止。
* 证据补记：读取 `logs/cleanup_20260916_01/{postverify,acceptance_recheck,modelcheck}.log`、两份 deploy JSON、active `.frozen`／manifest、PI1M/downstream pretrain／finetune `run.json` 和 aggregation summary。删除目标仍为 `ABSENT`，active 读取链仍在。清理后的局部回归已有 `54 passed, 1 warning`、exit 0；两份 deploy 报告已有 strict load、187 encoder tensors、resume bitwise identity、有限 `[2,1]` forward 和 cache zero-write。没有重跑这些检查。
* 命令和适用性边界：验收与 modelcheck 日志没有保存完整 argv／环境；Plan_Delete 第 7 节列出执行上下文可复述的命令并明确这不是独立日志证据。`scripts/validate_glt_dual_deploy.py` 未接收 static/targets 路径且在 `torch.no_grad()` 中 forward，因此 static/targets 清理后重新消费、backward 和删除前后预测 parity 均未执行／未独立核实。历史 PI1M `run.json` 仍绑定 sample_count `959588`、cohort `b03f96...`、bundle `30f17...`、dual static `9ff122...`、targets `5e7b...`；下游历史 `run.json` 只记录 downstream dual-static 与 `outer5_inner20` 的缓存／split 路径，没有缓存 hash 字段，当前 downstream manifest 与 `.frozen` 自洽，但这些 run.json 不能单独证明历史缓存内容身份一致。
* 聚合与溯源：复用 Concat/KFuse 8-task×40-fold summary（status PASS，macro8 `0.7877379364475444`／`0.7695629052761048`）和 80-shard summary，没有聚合重算。`cohort_30f17b59bc5862a1_v2` 原 manifest 仍是已知溯源损失，仅保留删除前 SHA256 `9eba93d60781361b005800f98d68ff8eba04a888fadf8b328915e147f1d23945` 与 `phase2_cohort_v2.json`，未生成补档。
* 交付与结论：已更新 `Plan_Delete.md` 第 7 节、`Plan.md` 清理注记和 `PIPELINE.md` 命令可追溯性表述；`RESULTS.md`、生产代码、缓存、checkpoint 和历史结果未改动。首批清理及有限只读证据收口完成，但代码精简仍未实施；全量读取、Stereo 质量、static/targets consumer、backward、parity 和新实验不因本条记录而通过。后续若继续处理 HOLD 候选，须重新依赖审查并取得明确授权。

## DOCS-20260916-06｜缓存优化计划交付

* 日期：2026-09-16（UTC）。用户要求 Codex 制定缓存优化计划并写入 Plan_Cache.md；规划、文档修改与自检均为 Codex，无独立审查者。只授权文档，未授权代码、缓存、benchmark 或训练执行。
* 基线与工作树：dev，HEAD 18572e373ae270469affa8ecb805db32f3c6489c；修改前核对 branch／origin 并 pull，返回 Already up to date。用户已有 Plan_Delete.md 删除及未跟踪 Plan_Cache.md（原内容为清理计划）；本轮按用户要求重写后者，不恢复、不提交前者的删除。
* 最终计划 CACHE-20260916-01/r1：保留三层缓存。A 最多 48 个接受集代表案例、4 条合同错误 source／拒绝证据及 9 个下游 fallback 的有界核对，分别裁定参照差异、审计器错误、真实化学错误和缺证。B 先以两个小 chunk 的 fixture 复现 staging 身份、半成品 chunk、单边发布和冻结后 finalize 的缺口，再做最小恢复修复；最多 32 个真实样本生成独立临时派生数据（不生成坐标），输出不超过 1 GiB。C 固定 2,048 个索引、worker=0 与至多 3 workers、baseline 和至多两个独立候选、每个配对最多 3 次、总计最多 30 分钟，记录吞吐／尾延迟／FD／RSS／映射开销，不能据此宣称 GPU 或模型收益。紧凑表示 D 单列待授权，未来也仅可先做 256 条／1 GiB 单候选，不自动全量迁移。
* 重要边界：不改变 first_valid、端基、映射、cohort、fallback、目标和随机流；不重写 active manifest／.frozen，不引入全量 hash 或通用迁移框架。两次 rename 不称为整体原子发布；优先用一致身份和幂等恢复处理单边发布。化学抽样不证明全部 FAIL 已解释，metadata hash 不证明 payload 历史未变。193 GiB 明确为清理前快照，历史 pilot benchmark 不被当成当前随机多 worker 证据。
* 实际文件变化：Plan_Cache.md 从旧清理内容改为缓存优化细则；Plan.md 仅增加专项交接注记，保留下方未收口科学计划；本文件归档文档周期。没有修改 PIPELINE.md、RESULTS.md、模型、构建器、缓存或历史结果。
* 验证与审查：仅对相关实现／已有报告做静态核对，检查文档引用、预算与授权一致性及 git diff --check；未运行单元测试、故障注入、真实 batch、benchmark、训练、构象生成、缓存构建或清理。文档自检不等于工程验收；实际 commit 和远端核对结果见本条对应 Git 历史及交付回复。
* 完成与下一步：本轮仅完成计划落盘。缓存实施状态待授权；建议先 A＋B，通过审查再 C，D 暂缓。已有科学计划和清理 HOLD 不因本轮扩大或结束，执行者应在 Plan.md 补记真实进度后交 Codex 审查。

## DOCS-20260916-07｜缓存审查后的 r2 返修计划落盘

* 日期：2026-09-16（UTC）；用户明确要求将下一步写入 Plan_Cache.md。Codex 负责文档规划、修改和自检；对 ZCode 的 7f476b4 交付已有静态审查，本次不执行代码返修、测试或 benchmark。
* 同步与基线：dev，7f476b481ceff4fa4a0561845230b329f6fa872b；修改前工作树干净，核对 origin 后执行 git pull --ff-only origin dev，返回 Already up to date，并重读相关文件。
* 计划修订：CACHE-20260916-01 保持同一 ID，r1 → r2，状态需返修。保留 r1 原始目标与报告，将当前缺口和优先级写入第 9–10 节；Plan.md 更新专项入口并追加审查／下一步，其他科学计划和 r1 执行记录不删除、不重新判定。
* 实质性返修合同：R1 修复完整 builder 幂等计时、未知 staging 身份接纳、临时完成块恢复、复用／冻结前 payload 校验、writer 互斥与释放、最终汇总后冻结六项；用两个小 chunk 的 fixture 测实际 build() 控制流和真实受控双进程竞争。R2 复用原 32 条 sample keys，在总量不超过 1 GiB 的独立临时产物中比较实际新 targets、静态字段、固定 mask/noise 的 clean/noisy 准备结果；缺失边界用 fixture，不扩大真实集合；差异／超限非零退出，不删除已有候选目录。R3 收窄化学源特征分桶的因果措辞和 8 条未解 Stereo 的状态，修正 benchmark 顺序／计数／进程资源口径及中心 angle_pairs 尚未复用的事实；只做工具局部合成测试，不安排新的长 benchmark、角度优化、阶段 D 或生产切换。R4 保存精确命令／退出码，交 Codex 复审。
* 修改文件：Plan_Cache.md、Plan.md、PROJECT_HISTORY.md；没有修改生产代码、PIPELINE.md、RESULTS.md 或旧报告／缓存。保留容量 64 不采用、默认 2 和全部科学／数据边界；已有授权不因交接撤销，计划不授权扩大预算。
* 验证：文档 diff、相关引用与逻辑检查、git diff --check；未运行单元测试、故障注入、数据构建、benchmark、模型或训练。提交／推送及远端核对结果见本条对应 Git 历史和交付回复。
* 周期边界：本条仅归档本次文档交付，不将未完成的 CACHE 实施周期归档为验收通过。r2 返修尚未执行，后续执行者按 Plan.md 记录真实进度，再交 Codex 审查。

## CACHE-20260916-01｜r1→r2→r3 缓存优化周期最终归档（2026-09-17）

* 授权与角色：本周期由用户授权缓存专项返修；Codex 负责规划与最终独立审查，ZCode 负责 r3 实现与局部验证。归档依据为 `Plan_Cache.md` 的 r1、r2、r3 记录及本次关闭核对。
* 修订链：r1 建立缓存合同、恢复、parity 与读取审查范围；r2 根据 Codex 审查结果收窄为六项恢复／冻结修复、固定 32-key parity 与工具口径返修；r3 完成 published payload contract、summary／冻结边界、固定 key provenance、parity fail-closed gate、benchmark 状态语义及对应 focused tests 的最小收口。
* 最终实现：r3 代码与测试的最终提交为 `632caa1`（`cache-opt: close r3 contract gaps`）。本周期最终文档状态为 **Codex review PASS / CLOSED**；该 PASS 仅覆盖计划约定的缓存合同、恢复与审计工具收口。
* 证据：既有 r3 focused tests 日志记录 `36 passed, 1 warning`（`logs/cache_opt_r3_tests2.log`），r2 固定 32-key parity 报告为 PASS（`results/cache_optimization_repair_20260916T234930Z/parity.json`），active frozen cache zero-write 与临时预算门控已在报告／实现中保留。上述证据在本次归档中只读复核，未重新运行测试或实验。
* 重要边界：本周期**没有全量缓存重建、格式迁移或生产迁移**，没有切换或写入 active cache，没有修改模型、配置、checkpoint 或历史结果数字；未执行的 parity 重跑、长 benchmark、payload 全字节历史完整性、断电耐久、模型／训练验证仍为未验证，不因 CLOSED 状态而改变。
* 交付与后续：`Plan_Cache.md` 与 `Plan.md` 已将缓存专项标记为 CLOSED，专项不再占据下一执行入口。若需生产切换、全量重建或新的优化，应另行制定计划并取得授权；本归档不产生这些操作的授权。

### 2026-09-17 更正与最终收口｜CACHE-20260916-01/r4 已完成

* 授权与角色：用户明确要求“将当前缓存处理设为完毕”。Codex 根据此前审查及本轮最终修正／既有日志核对完成状态更新；r4 代码和测试由执行者交付，Codex 本轮仅修改文档，不重新运行测试或实验。
* 历史更正：上条 r3 CLOSED 后又发现发布期锁路径、finalize 诊断路径及 snapshot 异常释放缺口，因此 r3 关闭不是最终状态。保留当时记录和失败日志，以本条 r4 最终收口为准。
* 最终实施与提交：`6877648` 将 flock 移至稳定的 artifact 同级路径、按 static→targets 加锁并在持锁后重新检查发布状态，增加诊断路径的词法／符号链接解析保护及对应合成回归；`84f91323cc274d04dc178beeb4e73a512ecfed1e` 将取得双锁后的 zero_write_snapshot 放入 try/finally，补充 snapshot 抛 OSError 后同进程可重新获取两把锁的测试。科学定义、数据集合、默认容量 2 和不采用容量 64 的决定不变。
* 验收证据：本轮只读核对最终 diff、生产调用位置、回归测试源码及 `logs/cache_opt_r4_snapshot_final2.log`。实际命令为 `PYTHONPATH=.:tests pytest -q tests/test_glt_dual_static_recovery.py`，既有结果 `13 passed, 1 warning`、`EXIT_CODE=0`，窗口 `Uni-Poly:cache_opt_r4_snapshot_final2`；此前 r4 12 项、r3 focused tests 和 r2 固定 32-key parity 证据保留，不称为本轮新跑结果。
* 文档操作：修改前 dev 工作树干净、HEAD 为 `84f9132`，pull 为 Already up to date；同步 `Plan_Cache.md` 完成状态与关闭说明、`Plan.md` 当前交接和本条归档；文档 diff／`git diff --check` 自检，提交与远端同步结果见对应 Git 历史和交付回复。没有改代码、缓存、PIPELINE.md、RESULTS.md 或历史指标。
* 最终结论：**已完成／CLOSED**，仅覆盖本缓存专项约定的审计解释、构建恢复、冻结／互斥、验证工具与证据口径收口。不等于全量化学认证、payload 全字节历史完整性、断电耐久、生产迁移、吞吐或模型收益验收。
* 下一步：**暂无后续缓存执行任务**。不补跑真实 parity、长 benchmark、训练，不启动全量重建、迁移、紧凑存储或清理；以后新增缓存工作另行规划并获得授权。其他科学计划不因本条自动完成。

## DOCS-20260917-01｜训练提速实施计划交付

* 授权与角色：用户要求制定当前预训练与微调提速计划；Codex完成静态核查、文档编写与自检，无独立审查者。本轮不执行代码修改、模型或benchmark。
* 基线：dev，`64d10bcd379a99e82dc5ad827d521ef08d39a342`；工作树干净，pull为Already up to date。该提交已删除旧Plan.md/Plan_Cache.md，本次新建Plan.md作为SPEED-20260917-01/r1入口，不恢复已关闭的缓存专项。
* 最终计划：保留当前GLT双通道Concat科学定义和训练工作量；先对实际static/targets路径做有界计时，再实现微调无标签clean输入的有限进程内复用和grid空闲槽位及时补位；预训练根据profile至多选择两个准备/预取/诊断开销候选。保留mask/noise、物理关系与中心目标、DDP全局分母、split、train-only scaler及validation选择，不新增全量缓存或改精度/batch。
* 证据与限制：核对当前源码、geonorm预训练run.json及fixed Concat代表微调run.json；预训练实际已有3workers/rank、BF16、no_sync和详细diagnostics；微调确有逐次clean重建，grid存在批次barrier。机器资源为瞬时快照，不把静态瓶颈假设写成已测提速。
* 拟议预算：输入parity每类最多32条，边界用fixture；预训练正确性12updates及一次必要2-step定位，性能最多160updates+40组合确认；xc/fold0微调最多8epochs且outer-test=NOT_RUN；墙钟总计2小时/产物10GiB先到即停。预算待授权，不是本轮已执行内容；GPU/worker任务仅在Uni-Poly独立窗口，低频监控。
* 实际文件：新增Plan.md，追加本条归档；不修改代码、缓存、PIPELINE.md、RESULTS.md或历史产物。文档引用/逻辑及git diff --check自检，无测试、训练、重建、清理。提交/推送结果见本条对应Git历史及交付回复。
* 结论与下一步：完成的是文档计划周期；SPEED实施仍待授权，未归档为完成。下一步按A→B→C→按证据D实施、记录后交Codex审查；不自动开启正式预训练或全任务微调。

## SPEED-20260917-01｜r1→r4 提速周期最终归档（2026-09-17）

* 授权与角色：用户授权在不改变 GLT-V2 科学定义的前提下执行有界提速返修，并明确不得自动启动正式 5k、20k 或完整微调。Codex 负责规划与最终审查，执行者完成 r1→r4 的代码、测试和 bounded smoke；本次归档由 Codex 只读核对后完成，不新增实验。
* 修订链：r1 建立计时、clean 输入复用、动态 grid 和候选准备路径的范围；r2/r3 实现有限进程内 clean cache、static/target 读取、中心 one-hop angle 向量化、可选 timing/worker 参数及 free-slot grid；r4 补齐 `prep_workers>0` 的恢复 RNG 顺序、重复 GPU 拒绝、`--clean-cache-gib` 显式转发、重复 key 不同标签和 bounded eviction 边界。
* 最终实现与记录：代码及测试最终实现提交为 `eda9f5c`（`fix: close speed plan execution contracts`）；r4 执行记录补记提交为 `f19ef02`（`docs: record speed plan r4 results`）。相关先前实现提交 `8cbe78e`、执行记录 `ff4e899` 保留在 Git 历史中。未修改 active cache、模型科学定义、配置、数据划分或历史结果数字。
* 验证证据（本次未重跑）：`logs/speed_r4_scoped_tests.log` 记录授权选择测试 `5 passed, 5 deselected, 1 warning`、退出码 0；`results/speed_20260917/r4_gpu_resume_report.json` 与 `logs/speed_r4_gpu_resume.log` 记录 4-GPU、`prep-workers=3`、总 8 logical updates 的连续／恢复比较为 PASS，模型、optimizer、scheduler、ordered keys、next position、各类 RNG、loss 与 target count 均符合既有 exact／容差口径。既有 parity、256 profile 与 `xc/fold0` A-B-B-A smoke 证据仍按原记录适用。
* 结论边界：**Codex review PASS / CLOSED** 仅覆盖 r4 约定的执行合同和有界提速证据。微调提速有 bounded evidence；预训练正式 throughput 未测。没有正式 5k/20k 预训练、Concat/KFuse 完整训练、正式 grid、8×5 微调、独立模型性能比较或全量缓存重建，因此不能把本周期写成正式性能提升或完整科学实验完成。
* 未执行与限制：没有重跑 PI1M/downstream parity、CPU 长 benchmark 或 baseline；没有启动 speed r5、正式预训练、完整微调、outer-test、构象生成、缓存迁移或新科学路线。合成测试与 bounded GPU smoke 不外推到全量训练吞吐；历史 5000-sample profile 未完成，不作为正式 throughput 证据。
* 归档操作：本轮仅修改 `Plan.md` 与本文件，未改 `.py`、config、cache、results 数字，未运行测试、benchmark、模型或训练。`Plan.md` 已切换为 `GLT-SCI-20260917-01 / r1`，状态为“待授权”；后续科学周期须由用户明确给出数据、模型包、task/fold、epoch/step 和比较预算后再执行。本归档不产生正式实验授权。

## DOCS-20260918-01｜GLT 工程提速最终执行方案交付

* 授权与角色：用户要求提供其他模型可直接执行的最终方案，明确先完成整个工程阶段，暂不涉及预测性能优化。Codex负责本轮规划、文档修改与自检，没有独立审查者；实施及有界测速仍待用户交付执行授权，不把文档当成已运行任务。
* 基线与同步：dev，`f1dfd4d`，工作树干净，修改前pull为Already up to date；当前Plan.md为空。读取现行AGENTS、实际GLT CLI、已有SPEED归档/报告并确认关键部署包和split存在；不恢复旧缓存计划，不推断其他路线的执行状态或授权。
* 最终计划：新建GLT-ENGINEERING-20260918-01/r1完整交接。S0锁定基线/资源，S1补source/static/targets/准备及进程全流程计时，S2真正按步采集详细诊断，S3同static条件下做clean cache独立对照并验证现有动态grid，S4按profile至多选择一个clean目标准备或pin传输候选，否则有据暂缓，S5做同协议ABBA及最终组合确认，S6交付采用/否决结论、启动/回退命令和独立审查材料。
* 科学边界：GLT双通道Concat、3GPU×84×4、global1008、原初始化/样本流/噪声/精度/loss/schedule及微调split不变；不改模型、融合、readout、目标权重或数据。不做预测性能调参，不读outer-test，不启动5k/20k/8×5、EQ3D或其他科学实验，不写active缓存或生成构象。
* 固定验证预算：CPU profile256条；PI1M/downstream parity各至多32；预训练总计至多316个实际updates（含短轨迹/恢复、诊断、两个单因素矩阵及组合确认），失败重跑占原预算；微调xc/fold0 cache对照8epochs，加eat/xc folds0/1两种调度smoke16epochs，总24epochs；长任务墙钟3小时、产物20GiB先到即停，GPU/worker任务仅Uni-Poly独立窗口且低频监控。没有收益可否决，不追加sweep；必要验证缺失不得宣称整个方案完成。
* 证据口径：旧约60→10.5秒是static与clean-cache组合且计时不含完整启动/保存；旧CPU准备计时漏static/target读取；旧4GPU恢复不等于3GPU正式吞吐。这些作为新方案必须解决的测量问题，不改写旧报告。原有LRU、动态grid、角度向量化、BF16/no_sync和恢复修复继续复用。
* 实际修改：Plan.md写入可独立执行的阶段、固定路径、已有/拟新增CLI、命令、预算、停止与验收条件及空白执行记录；本文件归档本次文档周期。未改生产代码、PIPELINE、RESULTS、缓存或产物；未运行模型、测试、benchmark。文档检查和git diff --check自检，提交/推送信息见对应Git历史与交付回复。
* 下一步：用户授权“执行本计划”后接手模型按S0–S6执行，Codex独立审查。工程验收完成后仅标暂无后续执行，预测性能优化必须另立计划并授权，不自动衔接启动。

## GLT-ENGINEERING-20260918-01｜r1→r4 工程周期最终归档（2026-09-18）

* 授权与角色：用户授权执行 GLT 双通道工程提速计划及后续最小返修；Codex 负责规划与本次独立审查归档，执行者完成 r1–r4 的实现、局部测试和有界 smoke。归档范围仅为工程接口、正确性、恢复与调度合同，不延伸为预测性能实验。
* 修订链：r1 建立计时、static/target 读取、clean cache、动态 grid 和有限候选范围；r2 修正 grid resume 身份隔离、预训练窗口计时、CPU profile 边界并补 worker=3 证据；r3 修正 launch-to-exit 计时的起点和退出观察边界，增加并行 batched 收割及真实 3-rank partial/all-zero geometry DDP；r4 修正 batched 完成结果因 active 列表压缩而可能错序的问题。
* 最终实现与文档提交：r3 实现提交 `4826c90`，r3 执行记录链为 `ed7a782`、`46e9e9c`、`7542369`；r4 最小修复提交 `54798fd`，最终交接文档提交 `4690d3d`。当前 `dev` 已推送并核对至 `4690d3de3f69cbfad97e2b48e61db313037bd202`，工作树干净。
* r4 独立审查证据：审查确认 `run_glt_dual_finetune_grid.py` 为每个 shard 保留稳定的原始任务序号，batched 仍并行 poll、先观察退出再 wait、保持 batch barrier；确定性假进程测试模拟 B→A→C，返回 A/B/C 且各任务一次。授权调度测试日志 `logs/glt_engineering_20260918/r4_grid_tests.log` 记录 `17 passed, 1 warning`、退出码 0；`py_compile` 和 `git diff --check` 通过。
* r3 及既有证据范围：r3 局部测试为 `27 passed, 1 warning`；真实 3-rank NCCL 报告 `results/glt_engineering_20260918/r3_no_geometry_ddp.json` 显示 partial-zero 全局计数 `[3,1,3]`、all-zero `[3,0,3]`，两 case 的 loss、backward 和 chemistry/fingerprint gradients finite，且 optimizer updates=0。r2 的 worker=3 恢复、窗口 schema 和 profile 证据按 `Plan.md` 与对应报告保留；不重复运行历史证据。
* Codex 审查结论：**Codex review PASS / CLOSED（有界工程合同）**。已确认 r4 没有改变 dynamic 补位、失败收口、模型科学定义、训练配置、数据划分、cache、checkpoint 或已有实验数字；没有发现会阻断本周期工程接口收口的剩余问题。
* 明确边界：`FULL_PRETRAIN_SPEEDUP=NOT_ESTABLISHED`、`FULL_FINETUNE_SPEEDUP=NOT_ESTABLISHED`。本周期没有正式 5k/20k、完整 Concat/KFuse 训练、8×5 微调、outer-test/OOF、全量 parity、ABBA 重跑或 cache rebuild；微调 bounded evidence 和历史预训练观察不外推为完整端到端收益，也没有比较预测性能。
* 本次归档操作：仅追加本条 `PROJECT_HISTORY.md` 记录并更新 `Plan.md` 状态；不运行测试、模型、训练、benchmark、DDP、profile，不修改 `.py`、config、cache、results 数字或 checkpoint。
* 后续：本工程周期已完成，暂无自动后续执行。任何预测性能优化必须另立科学计划，明确 reference、controlled change、task/fold、预算和停止条件，并取得用户授权；本归档不产生启动授权。

## GLT-PRED-20260918-01｜r1 规划文档交付（2026-09-18；科学周期未启动）

* 授权与角色：用户要求根据当前GLT分析制定完整预测优化计划，特别考虑将7-RU FP改为3D–2D融合监督，并写入Plan.md。Codex编制与文档自检，无独立审查者；本条只归档文档交付，不关闭或宣称完成科学周期。
* 基线与既有改动：dev@6d44ed9，用户已清空Plan.md；修改前检查remote/status并pull --ff-only成功（Already up to date）。按请求写入新计划，不恢复旧正文；工程r4完整归档已存在，未重复归档或重启。
* 最终规划：GLT-PRED-20260918-01/r1，状态待授权。S0核对Xc/来源/身份/预训练划分；S1实现FULL/HEAD/LoRA/Ridge适应接口；S2建立B_FP/B_NONE/T_FGR，FGR以共享融合表示和对称O8端点预测中心RU内SPD2/3原子对干净log距离，保留局部几何去噪与原子任务。S3先适应开发再三组matched预训练；S4条件推进中心原子—真实物理键桥、扭转或原子环境目标；S5锁定配置后正式开发五折确认。
* 重要边界：FGR为FlexMol-inspired项目条件重建，不是已证明的双模态协同或完整论文复现。不新增构象、修改active缓存、恢复MD200/旧KD；N=0仍无中心geo/FGR监督、保留原子任务。源P_train/P_val及benchmark重叠独立审计；旧checkpoint不能冒充新划分matched baseline。开发不读取outer-test，最终既有folds仍不能称全新盲测。
* 拟议上限而非授权：S1两epochs；S2最多16updates及两case零update DDP；S3三条5k；S4最多三条5k与12 correctness updates。开发神经微调最多1620+2epochs，S5最多三组8×5×100epochs（120单元），同组完整预算仍须明确授权。无收益不填满预算，不自动加seed/组。所有GPU/worker/长任务只在Uni-Poly独立window留日志，默认10分钟监控。
* 论文依据：GRIN、FlexMol、TMLR masking design、SCAGE、Token-Mol、DenoiseVAE、PolyConFM（预印本）、ELoRA及TabPFN；原文机制与项目改造分开，链接保存在本轮Plan及Git版本中。
* 实际修改与检查：只编写Plan.md并追加本条；核对现行代码/配置与参考路径，复查相关论文原始页面；文档格式、引用路径、预算逻辑和git diff --check自检。不运行单测、模型、预训练、微调、benchmark或缓存构建，不修改PIPELINE/RESULTS的既有科学结论。提交与推送信息见本轮Git历史和交付回复。
* 下一步：建议先授权S0–S2实施和有界验证；S3–S5研究训练/正式评估另行明确。已有授权若明确覆盖全部阶段则按合同条件推进，不重复申请小步骤。科学周期保持待授权，后续由执行者回填记录、Codex独立审查。

## GLT-PRED-20260918-01｜r2 对齐实验规划修订（2026-09-18；科学周期未启动）

* 授权与角色：用户要求判断2D–3D对齐是否合适并加入计划；Codex仅修订Plan.md与本条文档记录并自检，非独立科学验收。修订前dev@cd3af56工作树干净，安全pull成功；不改变r1已记录的历史状态。
* 最终增量：新增T_ALIGN，保留chem/geo、以独立图级双向多正例InfoNCE替换FP；同身份为跨模态正例、非同身份为负例，invalid/N=0剔除对齐集合但保留其他任务。O8原子GAP与GLT中心内部键GAP经现有LN及独立512→256→128投影对齐，tau0.1，lambda前1000updates线性升至0.1；保留原512维推理表示，不默认叠加FGR/FP或引入teacher。直接依据FlexMol（CIKM2025）的跨模态对齐，聚合物多正例/中心读出为项目适配。
* 分布式与边界：对比池为3×84的distributed microbatch（最多252），不是累积global1008；远端embedding gather保留梯度，按全局累积有效anchor数归一化。补多正例/全零/全同身份/单pair/padding/远端key梯度与数学参考要求；原子锚定桥不受prefusion ALIGN监督，禁止无监督地晋级该组合。高检索率不证明独有几何知识或XC提升。
* 拟议预算增量：S3b三组改四组，新增5000研究updates和180开发epochs；S2新增最多12 correctness updates及3个0update DDP case。研究总上限35,000updates＋40 correctness，开发1800＋2epochs；S4/S5名额及seed不增加。全部仍待运行授权，未启动任何实验。
* 交付与检查：更新矩阵、配置、部署、覆盖/晋级、S4桥依赖及预算总账，保留r1目标。文档格式与git diff --check自检；复核论文原始正文。未运行代码测试、模型、预训练、微调、缓存构建或清理；commit/push见本轮Git历史与交付回复。
* 下一步：用户可授权r2 S0–S2实现与有界验证；四组科学比较需明确授权S3预算。计划保持待授权，不把本条文档修订归档写成科学周期已完成。

## GLT-PRED-20260918-01｜r2 S0–S5 科学周期最终归档（2026-09-19；执行完成，待 Codex 最终审查）

* 授权与角色：r2 授权 S0–S2 实现与有界验证后，用户在本会话中按阶段逐次明确授权并执行了 S3a/S3b（含 S3b-Prep 数据准备、4GPU 资源协议冻结、四条正式 5k 与 24 个 development 单元）、S4（B_NONE 之上的 ENV/TORSION 候选）与 S5（B_NONE vs B_FP 正式 outer-test）。Codex 负责规划（本 Plan.md）与后续独立审查；ZCode 为实际执行者并完成逐阶段自检。本条由 ZCode 按用户明确要求归档，**不是 Codex 独立审查**；S0–S2 执行者自检记录见 Plan.md §15（保持原样），S3a–S5 的执行记录此前未回填 Plan.md（用户明确禁止修改），以本条为准。
* 基线与同步：周期起点 dev@ffce46d（r2 前）；本会话各阶段起点均核对 `HEAD == origin/dev` 并安全 pull。最终 `dev@018f8a0` 已推送并 fetch 核对，工作树干净（仅一个非本轮未跟踪 `.zcodeignore`，保留）。
* 执行链与提交：
  - S0–S2（前会话，执行者自检完成）：来源/split/XC 审计、FULL/HEAD/LoRA/Ridge 适应接口、`third_task=fp|none|fgr|align` 四分支与 DDP 验证；记录见 Plan.md §15 与 `results/glt_pred_20260918/s0|s1|s2/`。
  - S3a：三种神经适应开发比较后锁定 FULL 为统一下游策略（`results/glt_pred_20260918/s3a/`、`logs/glt_pred_20260918/s3a/`），development 协议（XC/EPS/EAT × fold0/1，30 epochs，outer_test=NOT_RUN）自此固定。
  - S3b-Prep（`d105ee2`）：P_train 911,391 经 `IndexedFrozenDualSource` 绑定 runner 并强制 split 身份；FGR 全 P_train FP64 统计 μ=1.3571292437646856/σ=0.1360748073854848（64,973,694 候选）；ALIGN 252 池全 P_train 诊断 PASS（3,617 microsteps，no_negative_fraction=0，identity 双射）；四 matched config 冻结；no-update smoke PASS（0 updates）。
  - 4GPU 协议（`ebed2c6`）：CPU 实测 112 逻辑=56 物理×2 SMT，T_FGR worker sweep（4/8/12 per rank → 199.8/369.7/524.5 samples/s）定 prep_workers=12/rank；四 config `expected_world_size: 3→4`；ALIGN 336 池（4×84）全 P_train 诊断重跑 PASS（2,713 microsteps，三门槛全 0/双射成立）；validator/tests 更新；no-update smoke 复跑 PASS。此前一次 3GPU↔4GPU 速度对照（用户中止）仅留 benchmark 产物，3GPU 数据不作依据。
  - S3b-END-TO-END（`187a0e7`）：B_FP/B_NONE/T_FGR/T_ALIGN 四条正式 5k 串行（world=4/micro84/accum3/pw12/seed42），全部 runtime PASS + deploy_05000 strict-load；24/24 development units；聚合脚本按固定 gate 输出 `SELECTED_PARENT=B_NONE`（B_NONE gate PASS：XC mean Δ=+0.077；T_FGR/T_ALIGN 对 B_FP 与 B_NONE 全 FAIL → FGR_INCREMENT/ALIGN_INCREMENT=NOT_ESTABLISHED）。
  - S4（`a89267e`+`ec005d2`）：S3b provenance hardening 后重跑判定不变；ENV（周期 radius-1 环境目标，P_train-only 词表 1118，P_train UNK 0.00039/fixed P_val UNK 0.00054 ≪20% 停止阈值）与 TOR_ON/TOR_OFF（中心 RU 重原子真实三键路径 [cosφ,cos2φ]，noisy 坐标驱动，零初始化门控残差 MLP(2→64→512)，OFF 为等角值控制；coverage 1024/1024 有效、退化 0）；correctness 13 tests + 3×2=6 updates（≤12）；三轨迹串行 5k 全 PASS + strict-load；18/18 development units 全部 outer_test=NOT_RUN；聚合 ENV/TORSION_INCREMENT 均 NOT_ESTABLISHED → `SELECTED_S4_PARENT=B_NONE`。ANCHOR 按源码事实跳过（chem 消费 fusion 前 2D atom_states、geo 消费 fusion 前 3D bond_states、third_task=none 时 fused 仅入图零，无学习信号）。
  - S5（`8fe5c58`+`018f8a0`）：S4 hardening 复跑不变；formal-shard provenance（protocol=outer5_inner20_formal_shard、formal_shard/adaptation=full/deployment_step=5000/outer_test=RUN_ONCE）与 exactly-once `outer_test_access.json`（STARTED→COMPLETED，已有 marker 拒绝重跑）；B_FP/B_NONE 两 checkpoint strict-load PASS；80/80 formal units（8 tasks × 5 folds × 2 arms）一次通过、0 失败、80 markers 全 COMPLETED、无重复访问；OOF union/目标对齐/双臂 row_index 一致性校验通过（float32 往返按 rtol=1e-5/atol=1e-4）。
* S5 正式结果（test R²，5-fold mean±std，ddof=0）：macro8 B_FP=0.792319、B_NONE=0.794343（Δ=+0.002024，保护条件满足）；XC Δmean=+0.002179、正 fold 2/5（+0.031/−0.107/−0.003/+0.128/−0.038）→ 预注册 XC 条件（mean>0 且 ≥3/5 正）不满足 → **B_NONE_FORMAL_GATE=FAIL，FINAL_SELECTED_MODEL=B_FP**。B_NONE 在 eat/eea/egb/eps/nc 五任务 fold-mean 更优、egc/ei 更差；XC 两臂方差均大（B_FP 0.28±0.12）。产物 `results/glt_pred_20260918/s5_formal/{summary.json,report.md}`。
* 预算总账：研究预训练 35,000 updates 上限实际用满（S3b 20,000 + S4 15,000）；S4 correctness 上限 12 实际用 6（加 S2 的 28 上限内合计 ≤34/40）；开发神经微调 1,800 epochs 上限用满（S3a 540 + S3b 720 + S4 540）；S5 正式上限 3 组 120 单元，实际 2 组 80 单元。outer-test 首次正式访问共 80 次，RUN_ONCE，无重复。
* 结论边界：**本周期至此为“执行完成、待 Codex 最终审查”，不是 CLOSED**。S3b/S4 的 development gate 结论（B_NONE 候选）在正式 outer-test 上未复现（XC fold 方向不一致），最终按预注册规则保留 B_FP 为通用路线；该判定是决策阈值结果，不宣称统计显著。FGR/ALIGN/ENV/TOR 四个目标扩展在 matched 对照下均 NOT_ESTABLISHED，按失败证据原则应 STOP，除非另立新假设与预算。历史 fixed-Concat/O8 参照与本次 paired fold difference 不构成严格因果比较（来源 population 不 matched），只作 HISTORICAL_REFERENCE_ONLY。
* 未执行与限制：ANCHOR（无信号，跳过）、matched O8 预训练（S5 仅两组 80 单元）、FP+FGR/组合目标、多构象/多任务/ensemble、预训练吞吐正式 benchmark 之外的调度研究、cache 生产切换；S3b 之前的 3GPU 数据仅作历史记录。无正式 5k 之外的超额训练，无 test 结果驱动的重跑/加 seed/换 checkpoint。
* 归档操作：仅追加本条到 `PROJECT_HISTORY.md`；不修改 `Plan.md`（用户明确禁止）、代码、config、cache、results 或 checkpoint；`git diff --check` 自检。commit/push 见本轮 Git 历史与交付回复。
* 下一步：交 Codex 对 S0–S5 全链路（源码、configs、tests、正式产物与 commit 链 `d105ee2→ebed2c6→187a0e7→a89267e→ec005d2→8fe5c58→018f8a0`）做最终独立审查并归档审查结论；审查后的生产路线整理、RESULTS/PIPELINE 更新或任何新实验均须另行授权，本归档不产生新授权。

## 2026-09-20｜旧 GLT-PRED 审查补记与 CANON3D r1 规划交付

* 旧周期状态更正：此前 Codex 对 S5 的80个已有预测单元进行了只读独立复算，R²/MAE/RMSE与保存值最大差异约2.22e-16，逐折test行序与固定manifest一致，OOF各样本一次，B_NONE晋级gate失败、保留B_FP的数值判断成立。此为有限范围验收，不是对S0–S5所有训练过程的重放或完整独立审查。
* 旧周期仍需收尾：2026-09-20源码复核确认launcher完成判定路径仍与实际嵌套产物不一致，aggregator仍直接采用保存的逐折指标；上文执行者所述COMPLETED/OOF唯一性不独立证明所有历史运行绝无重复test访问。此前Plan与执行脱节的情况现由执行者归档解释为用户阶段性禁止修改Plan；不据旧Plan文字反推未授权。保留历史记录，旧周期不因新计划而标为CLOSED。
* 本轮用户要求与角色：用户要求给出借鉴2D的3D输入优化方案及完整价值验证计划。Codex编写`GLT-CANON3D-20260920-01/r1`并做文档自检；没有执行模型实现或实验，也不是该方案的独立科学验收。
* 基线与已有改动：dev@79e3d2d；Plan.md已由用户清空，为唯一已跟踪修改。检查远端仓库后`git pull --ff-only origin dev`成功且Already up to date，在空Plan填入新合同，不恢复旧活动计划；历史科学周期未完成项保留在新Plan的衔接节。
* 最终方案：M个canonical重原子状态，中心query访问全部真实Trimer images，删exact self、保留独立左右物理关系，不平均距离；固定完整M(3M−1)关系避免geometry-OFF的近邻membership泄漏。512维/6层/8头，新3D独立target-query/source-key模块，relation-conditioned bias和value；原O8、Concat与chem/geo/FP目标不变。新atom经对称atom→bond适配沿用中心键监督，N=0不伪造目标。开放Trimer状态共享为近似，不宣称严格周期性。
* 验证计划实质：R_GLT、R_2D、G_OFF、S_SHARED四组先做matched 5k和XC/EPS/EAT×fold0/1开发筛查；只有几何和替换双重gate通过，才增加固定侧source的E_STATIC及3M动态状态P_PHYSICAL；再用S/G/锁定参考做seed43/44确认。相同数据、公共初始张量、随机流和原20k scheduler前5k；不复活FGR/ALIGN/ENV/TOR，不生成构象，不改变旧缓存。
* 拟议预算：P0最多1024条数据审计；P1六路径两步共12updates加恢复8updates=20，另12微调smoke epochs；P2–P4最多12条5k=60000研究updates、72开发单元/2160epochs，按条件逐段授权、不为用满预算补跑。P5另行授权且须通过开发确认，最多三组八任务五折120单元/12000epochs，不新增预训练；既有outer-test已见过，只能称固定协议复评。
* 交付变化：本轮仅Plan.md及本条历史衔接/规划记录；查阅当前源码、配置、报告和GRIN/SpaceFormer/Matformer原始页面，检查预算、引用与Markdown格式。未运行测试、GPU、模型、缓存构建、预训练、微调或清理；未更新PIPELINE/RESULTS以免将拟议路线写成生产事实。
* 下一步：新科学周期待授权，建议只先授权P0–P1；旧S5工程缺口在复用前做最小修复与CPU测试，不重跑旧outer-test。负结果可以结束新周期，未满足gate不得机械扩展。Git提交/推送结果见本轮交付，不以Git同步代替方案价值验收。
