# MCL-PH-20260921-01 / r11-PTDL5：五折 Periodic-TDL 训练协议适配

**状态：待执行（本轮用户已明确授权五臂全量重跑）。** 实现基线 `dev@dd23be2`；本轮开始工作区干净，`git pull --ff-only origin dev` 无更新。用户先要求调整微调策略和对应的五折处理，本轮再明确要求“重跑全量微调”；执行者 Codex。授权范围为下面的五臂 × 8 任务 × 5 折、仅 train/inner-validation，最多 200 starts / 13,750 epochs；此前禁止 outer-test 的边界保持。旧 r10R3 轨迹已归档到 `PROJECT_HISTORY.md`，原结果目录不覆盖。以下为当前唯一活动计划；下方 r10R3 内容只作历史记录，不授权重复执行。

## 科学问题与协议边界

- 问题：在同一冻结预训练包、同一项目 cohort 和固定五个 `outer5_inner20` outer fold 上，五臂若统一使用 Periodic-TDL 论文的两阶段训练策略，内部验证结果是否改变？参考为已完成的旧协议 200/200 内部验证；同时改变多项微调超参数，不能把差值归因于单项机制。
- 论文 Methods 的训练协议：head-only 10 epochs；Egc 再 joint 50 epochs，其余对应任务再 joint 60；MSE、batch 24、global grad clip 5、第一阶段 head LR 3e-4 与 cosine 降至 3e-5；第二阶段 encoder LR 1e-4（Xc 2e-4）、head LR 1e-3、10-epoch cosine warm restarts、最低 LR 1e-6；全程最低验证 RMSE 选 checkpoint。按任务设置 weight decay、head dropout；Xc 保留原目标尺度，其余目标用 train-only 标准化。依据：[论文 Methods](https://arxiv.org/pdf/2605.26833)。
- 项目适配：五臂保留现有 GLT/O8/MCL 架构和预训练包，head 的宽度及结构仍与论文 HSMP head 不同。项目 `eat` 不是论文九任务之一，暂明确借用 Eea 的正则参数；这不构成论文原值。复用现有项目 outer 五折；论文官方 release 提供自己的 `<dataset>_folds.pkl`，本项目 cohort、样本顺序及 inner seed 与之不能宣称逐样本相同。五折训练入口会对每臂每任务运行 fold0–4。当前沿用用户此前限定的 **仅训练和内部验证**，`outer_test=NOT_RUN`，不把结果当论文五折测试分数。若本轮的“五 fold”意指读取五个 outer-test，须先修订该数据/评价边界。

## 已实施与自检

- `scripts/finetune_mcl_ph.py` 增加显式 `--finetune-strategy periodic_tdl`：两阶段 trainability/optimizer/scheduler、任务超参数、原尺度 RMSE 选模、Xc 不标准化、论文 batch/dropout/clip、阶段身份与选模指标记录。原 `legacy` 入口保留，用于解释已有产物。
- 五折串行及四 GPU launcher 接受该策略并为 Egc 指定 60、其余任务指定 70 总 epochs；禁止混合策略复用旧 unit。验收器要求新协议完整阶段序列、RMSE 最优选择及两阶段参数组；旧协议产物仍通过原验收。全量聚合拒绝混合新旧策略。
- 局部无模型测试 `tests/test_mcl_ph_periodic_tdl_strategy.py` 覆盖冻结/解冻、任务参数、10+50 阶段、跨阶段 RMSE 选模、学习率周期及新协议单元验收；与既有 8×5/聚合测试合计 **21 passed**。`py_compile` 与 `git diff --check` 通过。无新 GPU 运行、无训练更新、无新 checkpoint。
- 旧轨迹 `full8x5_r10r3_3` 的 launcher 真实退出 0；本轮只读重验旧聚合 **200/200 PASS**、0 rejected、`finetune_strategy=legacy`、`outer_test=NOT_RUN`。旧权重不能后处理为新两阶段轨迹，若要得到新协议的五臂完整比较，须从四个原预训练包重新训练全部 **200 units**。

## 本轮已授权的正式运行

1. 保留旧产物，使用全新结果/日志根；先核对四包 SHA、split/cohort 索引身份、四 GPU/进程和无覆盖风险。原 5k 预训练不重跑。
2. 按五臂 × 8 任务 × fold0–4 运行 200 units；每 task/fold/arm 均从对应原始 5k 包重新开始。每 unit Egc 最多 60、其他任务最多 70 epochs，共最多 **13,750 epochs / 200 starts**；四张 GPU 可各承载独立 unit。失败即停新调度，保留现场，不自动重试、恢复或增加预算。
3. 每 unit 核对真实退出码、阶段 history、最佳 RMSE、checkpoint/预测/split/package 身份。全部 200/200 通过才汇总内部验证 R²；保持 `outer_test=NOT_RUN`。禁止将旧协议已完成 unit 混入新根。P3、OOF、refit、outer-test 仍未纳入此待授权预算。

**本轮启动前核对**：远端 `dev` 已同步；四个部署包与修订统计文件 SHA 均与旧冻结计划匹配。可信 cohort index 所绑定的 `records.jsonl` device/inode/size/mtime_ns 与现场相同；GPU 0–3 均可见且无同任务计算进程。新结果根 `results/mcl_ph_20260921/p2/full8x5_ptdl_r11_1/` 与日志根 `logs/mcl_ph_20260921/full8x5_ptdl_r11_1/` 均不存在。运行使用 `Uni-Poly` 独立 window 和四个独立 GPU worker，不设 wall-clock timeout。真实启动/运行状态须另据 launcher 命令、日志与退出码记录，不能从本段预检推断已启动或完成。

## 已归档的 r10R3 历史记录（不再是活动计划）

### MCL-PH-20260921-01 / r10R3-FULL8X5-3：五臂 8×5 内部验证微调

## FULL8X5-3 四 GPU 并行续跑（Codex；2026-09-24 UTC）

用户明确要求停止单 GPU 串行微调并使用四 GPU 重启。当前调整为 **GPU 0–3 各自串行运行 26 个互不重叠的 unit，四个 unit 同时训练**；单个 unit 仍保持原 batch、seed、30-epoch 上限、预训练包、split 与内部验证规则，不改成数据并行或改变 global batch。已完成 unit 只读复用；只在全新结果目录写入剩余 unit。

- 串行轨迹 `full8x5_r10r3_2` 停止前有 6 个新 unit 真实退出 0，加上复用的 90 个，共 **96/200** 已验收。停止时 `m_cat/egc/fold1` 正在运行；对 `Uni-Poly:96` 发送一次 Ctrl-C 后两个相关 Python 进程均已退出。该 unit 仅留 `runtime.json` 和截断日志、没有 `.exit`；launcher 的 `_launcher.exit` 也未写出，因此不得把它记成正常退出或完成。所有旧现场保留。
- 本轮新增 `scripts/run_mcl_ph_8x5_4gpu.py`，从 `full8x5_r10r3_2` 逐 unit 复核真实退出码与 `check_unit`，明确排除被中断的 `m_cat/egc/fold1`，把 104 个剩余 unit 以 round-robin 分至四张物理 GPU。每个子进程以专属 `CUDA_VISIBLE_DEVICES` 启动，unit 输出和日志路径互斥；某个 unit 失败即不再调度新 unit，其余已启动 unit 完成后停止并保留现场。全部完成才聚合 `outer_test=NOT_RUN`。
- 首轮 91 次启动，第二轮 7 次启动（6 通过、1 被用户要求中断），累计已 **98 次启动**。第三轮最多 104 次新启动；累计上限修订为 **202 次启动**、目标仍为 200 个通过 unit。被中断的部分 epoch 照实计账，不恢复其模型状态，不覆盖旧产物。新目录为 `results/mcl_ph_20260921/p2/full8x5_r10r3_3/` 和 `logs/mcl_ph_20260921/full8x5_r10r3_3/`。
- 局部验证：`pytest -q tests/test_mcl_ph_8x5_scope.py` 为 3 passed；对真实旧根只读预检为 96 accepted、104 remaining，分配 {0:26, 1:26, 2:26, 3:26}；`py_compile` 和 `git diff --check` 通过。启动前仍须核对四卡、无旧进程、目录不存在及远端同步。只做 train/inner-validation，禁止 outer-test、OOF、P3、refit、新预训练及额外 seed。完成条件仍为 launcher 真实退出 0、200/200 unit 验收及最终聚合 PASS。
- 实际启动：代码与修订计划提交 `db8c06d` 已推送 `dev`；命令完整记录于 `logs/mcl_ph_20260921/full8x5_r10r3_3_launch.sh`，运行窗口 `Uni-Poly:96:mcl_ph_full8x5_4gpu`，launcher 日志及最终退出码为相同前缀 `_launcher.log`、`_launcher.exit`。启动后 `launch.json` 记录复用 96、待运行 104、四 worker；GPU 0–3 各有一个微调计算进程，分别先运行 `m_cat/egc/fold1`、`fold2`、`fold3`、`fold4`。此时 launcher 仍在运行，最终退出码与 200/200 聚合尚未产生。

状态：**执行中**（五臂 8×5 内部验证微调，四 GPU 并行续跑）。日期：2026-09-24 UTC。用户已明确授权五臂全部、8 任务×5 折、仅微调与内部验证，并在本轮要求停止串行任务、改用四张 GPU 重启；执行者：Codex。目标为 200 个通过的 unit、每个最多 30 epochs；两条先前轨迹累计 98 次启动、96 个 unit 通过、1 个失败和 1 个按用户要求中断，本轮最多再启动 104 个 unit，故累计上限为 **202 次启动**。原 P2 development 30-unit 计划未启动，不另加其预算。既往记录保留在下文，不追改当时结论。

## FULL8X5-2 阻断修复与续跑（Codex；2026-09-23 UTC）

- 首轮现场：`results/mcl_ph_20260921/p2/full8x5_r10r3_1/`、`logs/mcl_ph_20260921/full8x5_r10r3_1/`；launcher 真实退出码 1。前 90 个 unit 的五件产物与退出码通过启动器验收；第 91 个 `m_cat/egb/fold0` 在首 epoch 报 `frozen Trimer atomic_number length disagrees with coordinates`，之后 109 个没有启动。失败产物保留，不作原地续训。
- 根因：真实 `egb` train 行 557 的冻结 Trimer 标记 `trimer_geometry_valid=False`，坐标为 `[0,3]`，但结构原子表有 107 行。`build_trimer_view()` 和 `centre_mapping()` 错把坐标行数当作结构原子数，先于已存在的 geometry fallback 发生错误。修复限定为无有效几何样本：先验证结构 carrier，再按原子表构造索引及仅供张量形状使用的零坐标；`mcl_geometry_valid=False` 继续禁止空间边、拓扑描述符及几何目标，模型读出回退到 O8。有效几何样本维持原严格坐标长度检查。
- 局部验证：新增无几何合成 fixture，PASS；对真实 `egb` 行 557 的选择性加载、MCL sample 和 collate 检查 PASS，空间边为空、geometry/readout 标志均为 false。旧 campaign 的 90 个通过 unit 经真实退出码和 `check_unit` 逐个复核，候选复用数 90，明确失败单元 1；四个包 SHA 与首轮启动身份一致。
- 续跑使用全新输出根 `results/mcl_ph_20260921/p2/full8x5_r10r3_2/`、全新日志根 `logs/mcl_ph_20260921/full8x5_r10r3_2/`。启动器要求显式 prior 输出/日志根及 `--retry-unit m_cat_egb_fold0`，逐一验收通过单元后将它们只读链接到新根；失败单元不得复用，旧根保持不变。新启动上限 110 次，首轮加本轮总上限 201 次。每个新 unit 真实退出码和 `check_unit` 通过后才继续；全部 200 个目标通过后才聚合内部验证。失败即停止，不自动恢复、重试其他 unit、读 outer-test、启动 P3/OOF/refit。
- 实际代码和计划 commit `dcdfbb2` 已推送 `dev`。启动前确认 `Uni-Poly` 可用、无同任务进程，四卡无计算进程，新产物根不存在。实际命令为 `logs/mcl_ph_20260921/full8x5_r10r3_2_launch.sh`，运行于 `Uni-Poly:96:mcl_ph_full8x5_resume`；launcher 日志为同前缀 `_launcher.log`，最终真实退出码写入 `_launcher.exit`，逐 unit 日志与退出码在新日志根。当前 launcher 已验收并链接旧轨迹的 90 个通过 unit；`m_cat/egb/fold0` 已开始训练，尚未完成。此时状态只能称执行中，不能称全量通过。
- 修复验收更新：`m_cat/egb/fold0` 真实 `.exit=0`，30 epochs、360 optimizer updates，`best_epoch=25`，`check_unit(..., stage='full8x5', expected_step=5000)` 为 PASS，部署包 SHA 为 `eed62765…e0e49`；launcher 已记录该 unit PASS 并开始 `m_cat/egb/fold1`。当前合计 **91/200 个目标 unit 通过**（旧 90 + 新 1）；首轮 91 次启动加续跑 1 次，实际累计 **92/201 次启动**。新根剩余 109 个目标 unit 待执行，最终 launcher 退出码和 200/200 聚合尚未产生。用户要求在修复确认后交接监控；本轮不发送信号、不停止训练进程，后续由其他模型只读监控。

## 当前授权的启动范围

- 五臂固定顺序 `glt_ref, o8_only, m_cat, m_gate, m_xattn`；任务固定 `eat, eea, egb, egc, ei, eps, nc, xc`，每任务 fold0–4，共 **200 units**。每 unit 最多 30 epochs，warmup 5、patience 10、seed 42 加 fold、train-only scaler、原超参数与各自 5k 包；不额外启动 30-unit P2 development。
- 使用 `scripts/run_mcl_ph_8x5.py`，全新输出根 `results/mcl_ph_20260921/p2/full8x5_r10r3_1/`，全新日志根 `logs/mcl_ph_20260921/full8x5_r10r3_1/`，`Uni-Poly` 独立 window，串行、无 wall-clock timeout。每 unit 留 `.log`、真实 `.exit` 和五件产物；退出非零或 `check_unit` 失败即停止，保留现场，不自动恢复、重训或扩大预算。
- 冻结 cohort 索引为 `results/mcl_ph_20260921/p2/cohort_record_index.json`；训练路径只读取当前折 train/validation 行。索引准备阶段曾扫描整份原始 JSONL 字节，这一事实继续如实记录。四个部署包及统计文件须匹配本计划下表 SHA，索引 source 身份及 split 行数一致；输出/日志根必须不存在，无冲突进程。
- 完成 200/200 后只聚合内部验证 R² 与 Macro8，`outer_test=NOT_RUN`。**不读取或评估 outer-test，不运行 OOF、P3、refit、新预训练或额外 seed；内部验证汇总不宣称独立盲测增益。**
- 启动前不重跑无关旧预训练 launcher fixture 或全仓测试。前轮 20 项针对性测试及真实 indexed loader 检查已通过；本轮只做动态身份与进程核对。代码已具备该 8×5 运行接口，无必要的代码改动不为形式重复修改。

## 科学问题与比较边界

- 问题：在固定的 `outer5_inner20` development 划分和训练合同下，MCL-PH 的 CAT/GATE/XATTN 是否达到预登记的双 baseline 晋级门槛？
- 参考：本轮 GLT_REF 和从同一 GLT_REF 包提取的 O8_ONLY。MCL 三臂分别使用各自的正式 5k 包。GLT_REF 与 MCL 的预训练任务、遮蔽和架构不完全相同；整体差值不能归因于 PH 或单一融合机制。
- 控制：五臂使用同一 XC/EPS/EAT × fold0/1 划分、seed 42、train-only scaler、最多 30 epochs、相同下游调度与选择规则。MCL 三臂之间仅 `fusion_mode` 不同。
- 结果层级：本阶段为 **development 筛查**，不是独立盲测或正式性能确认；不读取 outer-test，不自动进入 P3、OOF、refit 或更多 seed。

## 前置核对（2026-09-23 的只读审查）

| 下游臂 | 预训练部署包 | SHA256 |
| --- | --- | --- |
| GLT_REF、O8_ONLY | `results/mcl_ph_20260921/p2/pretrain/glt_ref_r10r1/deploy_05000.pt` | `7dc016dc422c58a6bba705a7a69c9f21051231350393a93d621b33e973883c47` |
| M_CAT | `results/mcl_ph_20260921/p2/pretrain/cat_r10r3/deploy_05000.pt` | `eed6276565f0bc827fc1f56056a25cd8469db184b0e334d04c3dd0f3cfea0e49` |
| M_GATE | `results/mcl_ph_20260921/p2/pretrain/gate_r10r3/deploy_05000.pt` | `312ea6708ee1b930001c1963714dfbdee44d8de711c5293bc0378be7a73b24e8` |
| M_XATTN | `results/mcl_ph_20260921/p2/pretrain/xattn_r10r3/deploy_05000.pt` | `9f2bea298b7341eaaf02e85a84f95974334e64b5436310127d30d82ae49f206e` |

- 四条正式预训练轨迹均有真实退出码 0、`runtime PASS/5000`；三条 MCL 轨迹各有五组千步 resume/deploy，500 为 dense、501 为 Top-2，统计 SHA `9dc8160f1de5152f6c04a963c40569bac8facd21e68a700f40186363798cf8b1`、共同新初值 SHA `499309392d578daf7a7baf1d102d7a7f5c652e5a9b42ca2904ac9d83d1fab85a`、common-init SHA `1c9f97cf5547dcd2f44846e83586dfb4ec593dd4a4f56df510be372f26f1951d` 一致。CAT/GATE/XATTN 的最终包均通过对应 `build_mcl_arm` strict-load；GLT_REF 通过 strict-load，O8_ONLY 复制 83/83 个 O8 张量，512 head 初值一致。
- XATTN 全日志中四 rank 各有 1–5000 的 5000 个 step 前缀、无非有限值标记；多 rank stdout 有行交错，故不把逐行 JSON 成功解析率当作运行步数。runner 自身在 loss 和梯度处设置非有限值硬失败，最终 runtime PASS。P2 aggregator 的针对性测试为 15 passed。
- 三个 split manifest 均标记 `outer5_inner20`；`results/mcl_ph_20260921/p2/development_r10r3/` 不存在，未有相关训练进程。开发执行前重新核对这些易变事实和 GPU 占用。
- 历史正式预训练累计 **25,082 updates / 6 次启动**（含失败 154 与 4928），本计划预训练新增预算为零；development 历史消耗为零。

## 授权后的执行方案

1. 安全同步远端并重读最新 `AGENTS.md`、本计划与 `MCL-PH.md`，检查工作区、tmux、GPU、四个包 SHA/strict-load、split、统计身份及新输出根。只读检查不得改写旧包、旧统计、cache 或失败现场。缺任一包或身份不符即阻断。
2. 在全新根 `results/mcl_ph_20260921/p2/development_r10r3/`，按 `glt_ref, o8_only, m_cat, m_gate, m_xattn` × `xc, eps, eat` × `fold0, fold1` 的固定顺序串行执行 **30 units**。每 unit `--stage development --expected-pretrain-step 5000 --epochs 30 --cohort-index results/mcl_ph_20260921/p2/cohort_record_index.json`；共同配置 `configs/mts/mcl_ph_gate.json` 只用于相同下游超参数，MCL 模式由 arm 与包严格决定。train batch 32、eval 64、seed 42、warmup 5、patience 10、MSE、train-only scaler、full adaptation；backbone 1e-5、head 1e-4、AdamW decay 0.02（bias/LN/router 免 decay），只用 validation R² 选 epoch。
3. 每 unit 在 `Uni-Poly` 的独立 window 保留完整命令、日志和真实退出码；先验收再启动下一个。要求 runtime PASS 且真实退出码 0，`run.json`/`metrics.json`/`best.pt`/`validation_predictions.npz` 齐全；`scripts.aggregate_mcl_ph.check_unit(..., stage='development', expected_step=5000)` 无问题；包 SHA、split、参数组、公共 512 head 初值、train-only scaler、epoch/更新数、选中 epoch 预测及 `outer_test=NOT_RUN` 一致。`best.pt` 仅是选中权重，不作为精确 resume 文件。
4. **仅在 30/30 unit 全部通过后**运行 `scripts/aggregate_mcl_ph_p2.py --root results/mcl_ph_20260921/p2/development_r10r3 --expected-pretrain-step 5000`。独立核对 30 个 unit 的实际包 SHA、`best.pt` 身份与预测文件。聚合必须为 `status=PASS`、30 accepted/0 rejected、`outer_test=NOT_RUN`，报告五臂每折与 Macro3、相对 O8_ONLY 和 GLT_REF 的差值、全部预登记门槛与 parent。

## 预算、门槛与停止

- development **最多 30 次 unit 启动 / 900 epochs**；patience 可以缩短单元，不把省下的 epoch 转为额外重跑。失败启动和已执行 epoch 照实计账，不自动重训、续训或增加 fold/seed。
- 每个 MCL 候选必须同时对两个 baseline 达到 Macro3 差值 ≥ +0.005、XC 均值差值 ≥ +0.01、XC 两折差值各 > 0、任一任务均值退化不超过 0.01（含边界）。合格集为空即 STOP。若 CAT 合格，GATE/XATTN 只有在合格且相对 CAT 的 Macro3 ≥ +0.002、XC 两折均正时才能取代；距最高 Macro3 < 0.002 时按合同优先 GATE。`best_epoch=30` 标注边界风险，不自动延长。
- 任一身份/包 SHA/split/训练数值/真实退出码/产物冲突或 `check_unit` 失败，停止后续 unit 和聚合、保留现场。不得读取 outer-test 特征、标签或预测；不得启动 P3、OOF、refit 或正式确认。
- 阶段完成后由执行者记录真实命令、窗口、日志、产物、预算与偏差；审查者独立核对并给出通过／需返修／阻断结论。development 聚合结果只能称筛查结论，不能称盲测性能增益。

## 本轮执行记录与阻断（Codex；2026-09-23 UTC）

- 用户明确授权本计划中的 P2 development 阶段，预算保持 30 次 unit 启动 / 最多 900 epochs；没有授权新增预训练、重跑、P3、OOF、refit 或 outer-test。执行前运行 `git pull --ff-only origin dev`，结果为 `Already up to date`；现场 `HEAD=origin/dev=2e8a5aa98921a836936d344b7cd8c8eac135e131`，工作区干净。
- 只读预检：四个部署包 SHA 均匹配本计划表。CPU strict-load：GLT_REF PASS；O8_ONLY 从 GLT_REF strict-loaded 包复制 83/83 张量；M_CAT、M_GATE、M_XATTN 均 PASS。四个 512-wide 新 head 的共同初值 SHA256 为 `1541ef7e59e5a5857275d9e473ecf62d4ca39ea2459ba7cb8baff60143be6437`。统计文件 SHA256 匹配 `9dc8160f1de5152f6c04a963c40569bac8facd21e68a700f40186363798cf8b1`。
- 三个 split manifest 均为 `outer5_inner20`，fold0/1 均可通过 resolver 的内部互斥与覆盖检查；resolver 输入的 cohort 行数取自 manifest 的 `sample_count`，真实 live cohort 行数本次未独立核对。development 输出根及专用日志根均不存在。`Uni-Poly` session 存在；未发现 MCL-PH 训练进程，四卡 GPU 利用率均为 0%，GPU 3 的 506 MiB 占用没有对应 compute process。未启动任何 unit；没有创建训练输出、checkpoint、unit 日志或退出码文件。
- **阻断原因**：`scripts/finetune_mcl_ph.py:354-363` 调用 `open_source()` 构造整个 task cohort 的 `frame`，随后在建立 train/validation `Subset` 之前，对完整 `frame['label']` 调用 `to_numpy(dtype=np.float64)` 并传入 dataset。因而 outer-test 标签也被读入并存入 dataset 的 target 数组，违反本计划“不得读取 outer-test 特征、标签或预测”的约束；虽然之后的 scaler、DataLoader 和预测只选 train/validation 索引，不能消除该读取事实。当前执行授权仅覆盖运行计划，未覆盖 finetune 数据加载实现的修改，故在首个 unit 前停止，未重试、未自动修复，也未运行聚合器。
- **下一步**：等待用户决定是否授权最小代码修改，使下游加载只接触 train/validation 标签，并在同一计划内重新审核后继续；在授权与修订前不启动任何 unit。P3、OOF、refit、outer-test 仍未授权。

## DEV2 修复、验证及 8×5 执行能力（2026-09-23 UTC）

本轮用户明确要求 Codex 直接解决字节读取阻断，并使微调器具备完整 8 任务×5 折能力。实际修改：

1. 新增 `scripts/build_mcl_ph_record_index.py`：对旧冻结 cohort 的 `records.jsonl` 作**一次性原始字节扫描**，同时计算完整 SHA256 与 6,265 个行偏移/长度，绑定原 manifest、文件 device/inode/size/mtime；索引写入新路径 `results/mcl_ph_20260921/p2/cohort_record_index.json`，旧 cohort 不变。此准备阶段确实扫描了包含 outer-test 标签的原始字节，不解析其值；不能宣称全流程从未读取这些字节。
2. `development` 和新增 `full8x5` 微调阶段强制传入 `--cohort-index`。训练读取路径在校验索引与冻结身份后用无缓冲 seek/read 只请求当前 fold 的 train/validation JSONL 行；不再顺序扫描或计算整份 `records.jsonl` 的 SHA。selected-row key、任务、原始行号、标签有限性仍严格核对；train-only scaler 与 validation 预测路径不变。索引源文件身份改变即拒绝执行，须在审查变化后重建，不能静默重建或绕过。
3. `scripts/finetune_mcl_ph.py --stage full8x5` 支持 `eat/eea/egb/egc/ei/eps/nc/xc` × fold0–4，所有臂均可使用，每 unit 最多 30 epochs；`development` 仍只允许原 XC/EPS/EAT × fold0/1。新增 `scripts/run_mcl_ph_8x5.py`，仅在调用者显式给出臂、各包和全新输出/日志根后，串行运行、逐 unit 留退出码与日志并立即验收，失败即停止；新增 `scripts/aggregate_mcl_ph_8x5.py`，对显式指定的每臂要求 40/40 unit 通过共享 `check_unit`，仅汇总 inner-validation R² 与 Macro8，保持 `outer_test=NOT_RUN`。它们不提供 P2 parent 重选、OOF、外层测试或独立盲测结论。单臂为 40 units，五臂为 200 units；这些是执行能力和潜在成本，**不是本轮运行授权**。
4. 针对性测试：`pytest -q tests/test_mcl_ph_label_isolation.py tests/test_mcl_ph_8x5_scope.py tests/test_mcl_ph_p2_aggregate.py` → **20 passed**。真实 cohort 索引构建在 `Uni-Poly:mcl_ph_record_index` 完成，exit 0；日志 `logs/mcl_ph_20260921/cohort_record_index.log`，索引记录原始文件 SHA `d9ed6fa4…4d0daa2`。真实 `egc/fold4` 选择性加载及静态 cache 绑定 PASS（2,163 train、541 validation）；全部 8×5 split 与 live manifest 的任务行数匹配，40/40 split 检查 PASS。未做模型 forward/backward、smoke 或微调；development 消耗仍 0/30 starts、0/900 epochs。
5. 扩大执行 `tests/test_mcl_ph_protocol.py` 时，两个未改动的旧预训练 launcher fixture 以 exit 6 失败（原期望 7/0）；该轮在 70 秒后停止，记录为 **2 failed、8 passed、未跑完**，不将其称为全套通过。此问题不在本次下游读取路径上，正式开发阶段启动前不以它掩盖已通过的定向测试。
6. 后续仅选择该文件的 aggregator/split/unit-directory 检查，约 60 秒后停止，**6 passed、10 deselected、未跑完**；没有新增失败。此前 20 项针对性检查仍为本轮已完成的局部验证，不把中断的扩大回归写作通过。

**恢复原 P2 development 的条件**：使用上述索引路径、原四个包和现有配置，在首个 unit 前重新核对索引/源身份、无冲突进程及输出目录；原 30 次 unit / 900 epochs 授权仍有效，失败即 STOP，不自动重跑。**8×5 运行仍待独立方案及明确预算授权**：必须指定臂集合、输出根、40/臂次 unit 与最多 1,200/臂 epochs、是否需要外层测试及其读取时机。若要外层测试结果，须另行实现锁定 epoch 后的测试评价协议；当前 `full8x5` 仅完成 train/inner-validation 微调，不读取 outer-test 标签或预测。

8×5 运行接口为 `scripts/run_mcl_ph_8x5.py --arms <明确臂列表> --package <arm=deploy_05000.pt> ... --config configs/mts/mcl_ph_gate.json --cohort-root data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1 --cache-root data/processed/mips_trimer_scage_downstream --dual-static-root data/processed/glt_dual_v2/downstream/dual_static_v1 --split-root data/splits/mips_outer5_inner20 --statistics results/mcl_ph_20260921/p0_r10r3/statistics.npz --cohort-index results/mcl_ph_20260921/p2/cohort_record_index.json --output <全新目录> --log-root <全新日志目录>`。此处是可执行接口说明，不是启动命令或预算授权；运行时须在 `Uni-Poly` 独立 window 内。

## FULL8X5-1 启动记录（Codex；2026-09-23 UTC）

- 用户本轮明确选择五臂全部、只做内部验证；不要求 outer-test。修改文档前 `git pull --ff-only origin dev` 返回 Already up to date，现场 `dev@ca9d0f9` 工作区干净。四包及统计 SHA 与本计划相符，cohort 行索引绑定 6,265 行当前源文件，无同任务进程；四卡在启动前空闲。
- 不删除模型/标签/包/产物身份和防覆盖检查；跳过无关旧预训练 launcher fixture 与全仓测试。现有 `run_mcl_ph_8x5.py` 已支持本轮 200 units，无必要代码修改，直接使用已验证代码。授权计划提交 `0d79714` 已推送 `dev`。
- 实际启动 window：`Uni-Poly:96:mcl_ph_full8x5`；命令完整保存在 `logs/mcl_ph_20260921/full8x5_r10r3_1_launch.sh`，launcher 日志和最终退出码分别为 `logs/mcl_ph_20260921/full8x5_r10r3_1_launcher.log`、`.exit`。输出根 `results/mcl_ph_20260921/p2/full8x5_r10r3_1/`，逐 unit 日志/退出码根 `logs/mcl_ph_20260921/full8x5_r10r3_1/`。无 wall-clock timeout，按显式五臂顺序串行。
- 首个 `glt_ref/eat/fold0` 已完成真实退出码 0、`runtime PASS`、30 epochs、best_epoch 23，五件必需产物齐全，`outer_test=NOT_RUN`；启动器已通过其 unit 检查并开始下一 unit。**当前只是执行中，不是 200/200 完成、P2 合格或性能增益结论。** 后续每 unit 的实际结果以各 `.exit`/`runtime.json`/聚合产物为准；失败即停止，保留现场。

## 标签隔离修订审查与阻断（Codex；2026-09-23 UTC）

- 本轮沿用计划 `MCL-PH-20260921-01 / r10R3-DEV1`。同步后基线为 `dev@8fe58759e08ea71dc20e823314a31ea1491e4060`，`origin/dev` 与本地一致，修订前工作区无既有改动。用户授权限于下游数据加载及必要局部测试；development 执行授权仅在标签隔离合同和既有前置检查均通过后继续生效。
- 候选修订改动 `scripts/finetune_mcl_ph.py`、`src/training/glt_dual_runtime.py`、`src/dataset/glt_dual_cache.py` 和 `tests/test_mcl_ph_label_isolation.py`：先由 split resolver 处理索引，再只加载 train+validation 行；dataset 仅接收这些标签；scaler 仅用 compact train indices 拟合；预测只保存 validation 行。`pytest -q tests/test_mcl_ph_label_isolation.py`：**2 passed**。测试用未选中行的无效 JSON/UTF-8 验证公开 `open_source()` 不解析该行，并验证 train-only scaler。未运行训练、模型 smoke 或全仓测试。
- live 清单核对只读取冻结 manifest、split 索引及 `keys.npy`：cohort manifest、keys、union 各为 6,265 行；XC/EPS/EAT 为 432/382/390 行，分别与 split 一致；fold0/1 的 train/validation/test 数为 XC 276/69/87、EPS 244/61/77、EAT 249/63/78；split SHA 与 cohort/union 绑定一致。没有读取真实 `records.jsonl` 或 outer-test 标签。
- 四个预训练部署包 SHA 均匹配本计划表；CPU strict-load：GLT_REF、M_CAT、M_GATE、M_XATTN 全部 PASS；O8_ONLY strict-load 后复制 83/83 个张量。按现有 `check_shared_head()` 对 O8_ONLY 与三个 MCL arm 的 512-wide head 作逐 tensor 比较，均与 `build_head()` 初值完全相同。统计 SHA 为 `9dc8160f1de5152f6c04a963c40569bac8facd21e68a700f40186363798cf8b1`；config、cache 与 static manifest 存在。development 输出根不存在；在 `logs/mcl_ph_20260921/` 下预期的 30 组 unit `.log`/`.exit` 文件共 60 个，当前全部不存在。
- **阻断仍未解除，候选修订不作为启动许可**：冻结输入把每行标签与其他记录字段放在同一个 `records.jsonl`，现有 manifest 没有 row-byte offsets 或标签隔离索引。候选 reader 虽不对未选中行做 UTF-8/JSON 解码、标签抽取或数组物化，但为按行索引定位及校验冻结 records 文件 SHA/行数，会遍历所有原始行字节；整文件 SHA 同样覆盖 outer-test 标签字节。按“完全不读取 outer-test 标签”的字面要求，当前格式无法证明未选中标签字节从未被读取；不把“不解析、不进入训练数组”自行解释成满足该要求。
- 现场没有 MCL-PH 训练进程；四卡 GPU 利用率均为 0%，GPU 3 的 444 MiB 占用无 compute process。没有启动 unit 或聚合；development 消耗保持 **0/30 starts、0/900 epochs**。
- 后续解除阻断需要可直接定位所选行且不扫描 outer-test 标签字节的既有可信索引与完整性验证路径，或用户明确修订 opaque checksum/行定位扫描的边界。本轮不生成 sidecar、不改旧 cohort、不启动 development。P3、OOF、refit 和 outer-test 仍未授权。
