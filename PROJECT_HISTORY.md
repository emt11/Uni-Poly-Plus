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
