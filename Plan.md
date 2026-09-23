# MCL-PH-20260921-01 / r10R3-DEV2：P2 development 与 8×5 能力修订

状态：**待执行**（原 P2 development 30-unit 授权）；**待授权**（新增 8×5 正式运行）。日期：2026-09-23 UTC。用户已授权 P2 development、最小标签隔离修复及 8×5 执行能力实现；未授权增加正式训练预算。执行者：Codex。原始计划基准 `dev@d4f31ee`；本次修复基准 `dev@98c0e08`，修改前 `git pull --ff-only origin dev` 成功、工作区干净。前一 `r10R3-GX1` 预训练周期摘要已归档到 `PROJECT_HISTORY.md`；历史失败与预算保留在 `MCL-PH.md`。DEV1 的阻断和候选修复记录保留在文末；本次修订不追改当时结论。

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

## 标签隔离修订审查与阻断（Codex；2026-09-23 UTC）

- 本轮沿用计划 `MCL-PH-20260921-01 / r10R3-DEV1`。同步后基线为 `dev@8fe58759e08ea71dc20e823314a31ea1491e4060`，`origin/dev` 与本地一致，修订前工作区无既有改动。用户授权限于下游数据加载及必要局部测试；development 执行授权仅在标签隔离合同和既有前置检查均通过后继续生效。
- 候选修订改动 `scripts/finetune_mcl_ph.py`、`src/training/glt_dual_runtime.py`、`src/dataset/glt_dual_cache.py` 和 `tests/test_mcl_ph_label_isolation.py`：先由 split resolver 处理索引，再只加载 train+validation 行；dataset 仅接收这些标签；scaler 仅用 compact train indices 拟合；预测只保存 validation 行。`pytest -q tests/test_mcl_ph_label_isolation.py`：**2 passed**。测试用未选中行的无效 JSON/UTF-8 验证公开 `open_source()` 不解析该行，并验证 train-only scaler。未运行训练、模型 smoke 或全仓测试。
- live 清单核对只读取冻结 manifest、split 索引及 `keys.npy`：cohort manifest、keys、union 各为 6,265 行；XC/EPS/EAT 为 432/382/390 行，分别与 split 一致；fold0/1 的 train/validation/test 数为 XC 276/69/87、EPS 244/61/77、EAT 249/63/78；split SHA 与 cohort/union 绑定一致。没有读取真实 `records.jsonl` 或 outer-test 标签。
- 四个预训练部署包 SHA 均匹配本计划表；CPU strict-load：GLT_REF、M_CAT、M_GATE、M_XATTN 全部 PASS；O8_ONLY strict-load 后复制 83/83 个张量。按现有 `check_shared_head()` 对 O8_ONLY 与三个 MCL arm 的 512-wide head 作逐 tensor 比较，均与 `build_head()` 初值完全相同。统计 SHA 为 `9dc8160f1de5152f6c04a963c40569bac8facd21e68a700f40186363798cf8b1`；config、cache 与 static manifest 存在。development 输出根不存在；在 `logs/mcl_ph_20260921/` 下预期的 30 组 unit `.log`/`.exit` 文件共 60 个，当前全部不存在。
- **阻断仍未解除，候选修订不作为启动许可**：冻结输入把每行标签与其他记录字段放在同一个 `records.jsonl`，现有 manifest 没有 row-byte offsets 或标签隔离索引。候选 reader 虽不对未选中行做 UTF-8/JSON 解码、标签抽取或数组物化，但为按行索引定位及校验冻结 records 文件 SHA/行数，会遍历所有原始行字节；整文件 SHA 同样覆盖 outer-test 标签字节。按“完全不读取 outer-test 标签”的字面要求，当前格式无法证明未选中标签字节从未被读取；不把“不解析、不进入训练数组”自行解释成满足该要求。
- 现场没有 MCL-PH 训练进程；四卡 GPU 利用率均为 0%，GPU 3 的 444 MiB 占用无 compute process。没有启动 unit 或聚合；development 消耗保持 **0/30 starts、0/900 epochs**。
- 后续解除阻断需要可直接定位所选行且不扫描 outer-test 标签字节的既有可信索引与完整性验证路径，或用户明确修订 opaque checksum/行定位扫描的边界。本轮不生成 sidecar、不改旧 cohort、不启动 development。P3、OOF、refit 和 outer-test 仍未授权。
