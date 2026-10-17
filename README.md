# Uni-Poly-Plus

聚合物性质预测科研项目。当前研究为ALIGN预训练的UA_2D与轻量FP GATE融合；已有主要完整七任务预测参考是G05＋KRR的r10全局融合（正式macro R² 0.796）。目标为[SOTA_RESULT.md](SOTA_RESULT.md)的七任务逐项参考，协议等价性尚未核实，不据数值宣称公平超越论文。

- **[科研交接入口](CHATGPT_HANDOFF.md)**：当前状态、结果差距、全部实验与架构导航。
- **[复制给ChatGPT网页版的提示词](CHATGPT_START_PROMPT.md)**：每次先审查远程最新更新，再规划并给Codex执行提示词。
- [所有已执行实验与未完成事项](docs/research/EXPERIMENT_CATALOG.md)、[全部现存正式政策表](docs/research/EXPERIMENT_RESULTS.md)。
- [当前完整模型流程](docs/research/MODEL_ARCHITECTURE.md)、[精简结果证据](docs/research/evidence/README.md)。
- [AGENTS.md](AGENTS.md)：ChatGPT规划/独立审查，Codex执行；每次代码修改完成必须commit/push。
- [CURRENT_PLAN.md](CURRENT_PLAN.md)：唯一当前授权与执行交接，r7及后续更新优先。
- [PROJECT_HISTORY.md](PROJECT_HISTORY.md)：历史已关闭或明确标注的待审查记录；[PIPELINE.md](PIPELINE.md)保留更早实现说明，当前主线以新架构说明与实际源码为准。

当前P2 r7正在训练20k步，每5k保存，随后自动接续70个GATE微调单元。GitHub中结果为带时间的快照，不能视为实时监控。完整数据、checkpoint、预测和日志保留在训练机，仓库包含代码、说明和精简统计；新clone不会自带这些本地实验依赖。
