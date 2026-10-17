# Uni-Poly：ChatGPT科研规划交接入口

先把[首条交互提示词](CHATGPT_START_PROMPT.md)复制到ChatGPT网页版。项目远程为 [emt11/Uni-Poly-Plus](https://github.com/emt11/Uni-Poly-Plus)，研究分支`dev`；每次规划先审查实际最新提交和最近更新，不能只依赖本文或缓存网页。

## 1. 当前事实与下一步

目标是[SOTA_RESULT.md](SOTA_RESULT.md)中共有七任务的逐项最佳R²。已有完整七任务的主要参考：旧UA_2D macro R² **0.788**；r10全局预测融合正式macro **0.796**，validation **0.816**；论文七任务最佳值包络 **0.810**。这几个数字不能混用，也不能把论文逐任务包络当单模型。

证据采集：**2026-10-17T23:14:43.002730+08:00（北京时间）**。当时P2 r7已完成 **5,069/20,000** 次更新、starts=1，下一次更新在途；G2尚无完成记录/正式结果，最终CPU suite未运行。本段是有时间的快照，不是实时状态。

当前队列在`Uni-Poly:ua2d_p2_align20k_r7`：从原U1/P_ALIGN step0新训练20k，每5k保存resume/deploy；固定20k parent自动接续GATE_REAL/NULL两臂×7任务×5fold＝70FT、正式预测/固定0.75融合和唯一最终CPU suite。仅GPU2/3，PT最多2路（一个周期两rank），FT两卡总并发4。按用户要求，起始正常后已停止Codex主动监控，不重复启动。

旧G1三次共43启动、19个旧step5000 parent完整FT，没有正式G1结果，保留但不拼入新parent矩阵；初次unknown消费仍保留。r5 P2因GPU2故障在PT启动前停止；GPU3-only草案未跑。最新r7两卡已实际完成起始训练，当前模型/源码仍待整个G2与最终CPU验证，不能预告提升。

本次静态资料整理发现并修正一处未来FT依赖：G2调用`restore_pure_2d(..., expected_step=20000)`，否则默认5000会拒绝新parent。PT未重启，科学预算未变，没有提前测试。该辅助模块在当前PT控制器中仅用于环境/观察；未来G2会由新的子进程加载此修正并保存实际源码。

下一步首先核对远程快照是否已过期：已有正式结果则逐项审查；仍在运行则规划结束后的评估，不另开候选。需要本地原产物时让Codex只读导出并push，不为补证重放训练。

## 2. 所有已执行实验与架构资料

|材料|用途|
|---|---|
|[实验目录](docs/research/EXPERIMENT_CATALOG.md)|从旧工程/GLT/MCL/GP到FCE调优、FP、G1事故和P2/G2的完整周期索引|
|[全部现存正式政策表](docs/research/EXPERIMENT_RESULTS.md)|43份源、126条含复用参照政策的七任务R²/std及论文差，按各自来源分组|
|[模型完整流程](docs/research/MODEL_ARCHITECTURE.md)|数据→O8/Galformer→预训练→纯二维GATE→微调→KRR/正式评价，含形状/公式/参数/训练预算|
|[精简证据](docs/research/evidence/README.md)|原精度JSON/CSV、运行消费快照与31个现存实验root索引|
|[PROJECT_HISTORY](PROJECT_HISTORY.md)|历史架构、实际预算、完整结果与当时审查；未关闭者保留原状|
|[CURRENT_PLAN](CURRENT_PLAN.md)|唯一当前授权/执行交接；当前r7与后续记录优先于旧快照|
|[AGENTS](AGENTS.md)|ChatGPT规划/审查，Codex执行；每次代码修改完成commit/push|
|[仓库同步规则](docs/research/REMOTE_WORKFLOW.md)|远程审查、计划/执行提示词与上传边界|

## 3. 当前七任务差距

|任务|旧UA_2D R²|r10融合R²|论文最佳模型|论文R²|UA−论文|r10−论文|
|---|---:|---:|---|---:|---:|---:|
|Eea|0.927±0.016|0.924±0.017|Periodic-TDL|0.934|-0.007|-0.010|
|Egb|0.926±0.010|0.923±0.012|Periodic-TDL (+Chem, +RC)|0.923|+0.003|-0.000|
|Egc|0.905±0.016|0.917±0.013|Periodic-TDL (+Chem, +RC)|0.911|-0.006|+0.006|
|Ei|0.803±0.029|0.807±0.016|Periodic-TDL|0.839|-0.036|-0.032|
|EPS|0.768±0.061|0.775±0.049|Periodic-TDL|0.787|-0.019|-0.012|
|Nc|0.813±0.065|0.816±0.049|Periodic-TDL|0.856|-0.043|-0.040|
|Xc|0.373±0.107|0.412±0.120|Periodic-TDL|0.423|-0.050|-0.011|

均为每任务五折算术均值、seed42+fold、std ddof=0，原精度计算后显示三位小数。Egb的r10差显示−0.000，原值为负，不能判断已达标。数据/split/论文统计口径等价性未核实，已有validation/TEST用于开发；只做探索性数值对照。新G2七任务全部**未完成／不可比较**，不拿旧结果补齐。

## 4. 一句话理解各模型

- **F_CE/G05/r10**：二维O8与三维Galformer做token融合；两路512维concat到1024维head。G05是geometry权重0.5的两rankparent。r10只在预测层加已选KRR，并固定一个全局0.75神经权重，不是新的神经模型。
- **UA_2D**：使用ALIGN预训练得到的O8/norm2，forward只有二维图与512→512→1 head，19,262,233参数；预训练用三维监督不等于下游输入三维。
- **G2 FP GATE**：在UA表示a上加`g·Δ`；FP 2048→32→512，门concat(a512,q32)→1，新增83,009参数，末层零使初始函数不变。REAL与常量NULL容量一致；G2的20k parent不是旧G05或旧UA step5000。
- **FP-GRAPH f1**：是旧G05三候选图级注意力，总增2,033,152参数，完整负结果；与当前轻量标量门是两个实验。旧设计附件不自动授权masked-FP预训练。

## 5. 审查与延续时最容易出错的地方

1. 宏5/宏8、validation/TEST、论文包络和当前macro7混用；显示同分不等于原精度相等。
2. “r10”至少指旧GLT架构周期与后来的FCE全局预测融合两个命名空间；先绑定完整plan/路径。
3. 只见metrics、PASS或RUNNING就认定完整实验已完成；远程未保存checkpoint与完整预测，不能伪称验收了它们。
4. 当前P2的两rankInfoNCE池168，旧四rank336；改变训练长度还改变rank/RNG，不能叫单因素PT长度消融。
5. REAL−NULL只检验同GATE容量下FP信息，不能证明GATE优于固定残差，fused−baseline也不自动是interaction。
6. r13有已知角度/PH限制；E0/E1/A1有未关闭或冲突证据；源快照未抹去这些失败。
7. 更复杂FP、Huber、LayerNorm-only、非键、蒸馏等已有负结果；仅否定已执行配方，不外推机制永远无效。

## 6. 本地与远程的区别

源码、配置模板、测试、文档与精简统计上传；数据、weights、完整logs/results、FP缓存、逐样本预测留本地。远程路径`source_local`只用于定位请求原证据，不是GitHub可打开的文件。当前代码/配置仍带训练机路径，新clone不能直接复现训练，也不应因此改写原训练目录。

每个完成的代码修改批次必须提交/push，无需重复申请；不force-push、不自动pull、不为推送运行额外模型测试。本次推送包含此前本地未同步的实验源码与测试，**不是一次对全部历史代码的动态验证**。计划审查、实际执行、最终CPU suite、独立审查与远程发布分开报告。
