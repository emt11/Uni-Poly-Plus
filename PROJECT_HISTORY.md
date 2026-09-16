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
* 验收：保留代码 import、7 个 CLI `--help` 和 active 路径检查成功；聚焦测试 `53 passed, 1 failed, 1 warning`。唯一失败 `tests/test_cache_lifecycle.py::test_downstream_geometry_fallback_keeps_complete_identity_carrier` 源于 `select_record_fields("trimer", ...)` 未携带可选 `trimer_failure_code`、而 fallback `_check_fields` 强制访问它；代码与测试未因本轮清理而改动，现有问题未降级为清理成功。
* 过程偏差：删除脚本首次错误地把 bundle 根当作 `.frozen` 路径并返回 1；实际删除与逐项校验已完成，随后按三层 artifact 的真实 `.frozen` 路径复核通过。该脚本断言属于后续可修复的工具问题，不影响本次已核准路径的删除结果。
* 审查结论：清理动作本身完成，未发现 active cache identity、读取链或 writer 冲突损坏；计划状态为“需返修”而非全绿完成，阻断项是独立 fallback 字段契约测试失败。未执行全量读取、Stereo 全量审计、模型前后向、预训练、微调、聚合重算或缓存重建。
* 下一步（需新授权）：先修复／明确 `trimer_failure_code` 序列化字段契约，并只重跑相关回归与 active artifact 只读验收；不扩大删除 allowlist，不恢复已删除 pilot／staging，不启动训练。
