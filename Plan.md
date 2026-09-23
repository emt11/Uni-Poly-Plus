# MCL-PH-20260921-01 / r10R3-DEV1：P2 development 筛查计划

状态：**阻断**。日期：2026-09-23 UTC。用户于 2026-09-23 UTC 明确授权执行本计划的 P2 development 阶段；执行者：Codex。计划创建基准为 `dev@d4f31ee`；本次执行前安全同步 `origin/dev` 后的现场基线为 `dev@2e8a5aa`，工作区干净。前一 `r10R3-GX1` 预训练周期摘要已归档到 `PROJECT_HISTORY.md`；历史失败与预算仍保留在 `MCL-PH.md`。执行在首个 unit 启动前因代码会读取 outer-test 标签而阻断，详见本文件末尾执行记录。

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
2. 在全新根 `results/mcl_ph_20260921/p2/development_r10r3/`，按 `glt_ref, o8_only, m_cat, m_gate, m_xattn` × `xc, eps, eat` × `fold0, fold1` 的固定顺序串行执行 **30 units**。每 unit `--stage development --expected-pretrain-step 5000 --epochs 30`；共同配置 `configs/mts/mcl_ph_gate.json` 只用于相同下游超参数，MCL 模式由 arm 与包严格决定。train batch 32、eval 64、seed 42、warmup 5、patience 10、MSE、train-only scaler、full adaptation；backbone 1e-5、head 1e-4、AdamW decay 0.02（bias/LN/router 免 decay），只用 validation R² 选 epoch。
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
