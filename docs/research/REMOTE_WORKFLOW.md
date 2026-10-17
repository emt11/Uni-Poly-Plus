# ChatGPT规划／审查与Codex执行的闭环

长期用户要求以[AGENTS.md](../../AGENTS.md)为准。工作仓库为 https://github.com/emt11/Uni-Poly-Plus，当前研究分支dev。首条消息见[复制提示词](../../CHATGPT_START_PROMPT.md)。

## 每次规划前

ChatGPT使用可访问的GitHub连接器、浏览器或Codex只读导出，读取实际最新branch HEAD、最近五条提交及相关diff，然后读AGENTS/CURRENT_PLAN/SOTA和当前问题相关代码、证据快照与历史。记录检查时间、提交ID与变化；网页缓存与旧对话不能作为新的审查。第一次还要读CHATGPT_HANDOFF及完整架构/实验索引。

远程网页/API和连接器可能不包含本地新日志。若快照过时，只给Codex一个零科学消费的导出/同步任务：读取已有运行状态与实际产物，更新精简快照，commit/push，再回来规划；不重新训练、推理、核fit或运行旧测试。不能访问仓库时，明确缺失内容，不声称完成远程审查。

## 计划与执行提示词

ChatGPT先审查上一轮实际diff、测试/正式结果和证据局限，再给最多一个最小受控下一步。说明scientific question、reference、controlled change、comparison budget；未获授权的额外科学预算只作为提案。CURRENT_PLAN记录用户授权，不产生授权。

执行提示词必须能独立交给Codex，包含：

- 计划ID/修订、所依据远程提交、用户已授权范围及当前运行的保留条件。
- 精确源文件/配置清单、parent/head/scaler/split/seed、任务/fold/step/epoch、推理/数组/核fit/FP的预算。
- GPU2/3，PT最多2路，FT两卡总并发4；tmux Uni-Poly独立window与完整命令/日志/真实exit。
- 既有运行不重复启动；异常保留现场，明确消费known/unknown，禁止扩预算掩盖问题。
- 正常validation与正式评价顺序、冻结点、全七任务论文对照、唯一最终CPU测试范围及未运行时的如实说明。
- CURRENT_PLAN更新、关闭周期在独立审查后归档PROJECT_HISTORY，刷新远程可读证据，然后commit/push交付。用户要求起始正常后停止监控时保留运行队列，不把启动称为完成。

## Codex提交与上传范围

每个完成的代码批次提交并push当前分支；测试或训练仍待结束时明确标记。独立审查不是push前人为审批门槛；源码可以同步为“运行中／待审查”，推送不能充当测试通过证明。

|上传|保留本地|
|---|---|
|src/scripts/tests、必要configs、依赖文件|原始/processed数据、LMDB/PH/static缓存|
|AGENTS/CURRENT_PLAN/PROJECT_HISTORY/SOTA及研究说明|step0/resume/deploy/best等checkpoint|
|docs/research/evidence中的精简统计与有时间的执行快照|逐样本预测NPZ、核矩阵/核模型、FP缓存|
|失败预算与来源位置的必要摘要|完整logs/results、运行源码镜像、大型debug/benchmark产物|

.gitignore已加入二进制/数据库/本地凭据规则，保留docs Markdown/JSON/CSV。已有跟踪文件不会因.gitignore自动从历史消失；TODO是个人文档，本轮未读取、修改或提交其变化，也不删除历史文件。绝不上传本地凭据。不做git历史清理或force-push。

远程有新提交先审查差异，再安全整合；不擅自reset用户工作树。本地提交后正常push，失败则保留本地提交并报告，不能说“已上传”。交付给出分支、提交ID/链接与实际push状态。Git提交ID用于版本定位，不建立文件校验和合同。

## 本轮远程审查记录

首次读取时间为2026-10-17晚（北京时间）：默认分支及本地工作分支均为dev；远程HEAD为`059d084aa6bf89ded67f9123e8c7991da0027163`，提交“Tuning point 1”（2026-10-15 08:43:03北京时间）。其前四条为af78563、d93c8a0、1a29e1d、70579ff。原远程未包含后续本地FCE/GATE等未提交文件；本轮fetch仅更新引用，没有pull或覆盖本地文件。

本轮将补齐这些科学源码与测试、文档和精简证据，并加入20k恢复参数的一处静态修正。没有为发布重跑历史实验/pytest，也没有重启现有P2。运行后的G2/最终suite未发生，待审查标记保留；推送后的提交由Codex交付消息和实际dev HEAD查询确认，不在同一提交的文档里伪造自身提交ID。
