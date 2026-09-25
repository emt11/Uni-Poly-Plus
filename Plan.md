# MCL-PH-20260921-01 / r13-PAPER5：论文官方五折可比评估

**状态：正式运行已获授权，代码收敛与启动中；尚未声称训练完成。** 用户本轮要求把当前五折改为与 Periodic-TDL 论文一致，并停止当前任务。执行者 Codex；修改前基线 `dev@25b081d`，工作区干净，`git pull --ff-only origin dev` 无更新。原 r12 四 GPU launcher 已退出 1，没有相关微调进程；r12 已取消并归档至 `PROJECT_HISTORY.md`，其 4 starts / 280 epochs 与失败现场均保留，尚无 outer-test R²。本轮用户随后明确要求以 `paper5_outer` 重跑，并删除旧微调逻辑；正式范围为五臂 × 五个官方可比任务 × 五折，共 125 个全新 unit，最多 8,750 epochs。

## 科学问题与可比边界

- 问题：在项目五臂架构和原四个 5k 预训练包固定的条件下，若下游使用论文发布的**相同样本、相同 outer-test 折和相同 inner-validation 划分**，项目模型在重合任务上的五折测试 R² 如何？参考为论文 Periodic-TDL 单模型的五折 test R²；控制变化是使用项目编码器/几何和预训练，架构与训练源仍有多项差异，不能把差值归因于单一组件，也不是 HSMP 复现。
- 论文 Methods：每折 80% outer train、20% held-out test；outer train 中 20% validation，最低 validation RMSE 选 checkpoint，之后一次评价 test。官方发布的 `*_folds.pkl` 给出外层 train/test；官方 `code/5_downstream.py` 对每个 outer train 使用 `train_test_split(test_size=0.2, shuffle=True, random_state=42)`。论文来源：https://arxiv.org/pdf/2605.26833；官方代码与数据：https://github.com/yasharthy/Periodic-TDL ，核对 commit `f3ba6dff6f0d065accdd235dfba160324714f30b`。
- **仅五个可证实任务**：Eea、Egb、Ei、EPS、Nc。官方清洗 CSV 与项目 raw CSV 的行序、SMILES、标签逐行相同，也与冻结 cohort 的 `original_row/source_smiles/label` 一致；每任务官方外层五折覆盖全部样本一次，内部划分逐索引重建且相同。对应 `configs/mts/periodic_tdl_official5/` 新清单，官方 CSV/pkl 快照在 `source/`，清单记录来源 commit 与 SHA。项目旧 `outer5_inner20` 清单保留用于冻结 cohort 的身份绑定，**不得再作为 r13 的评估折**。
- EAT 不在论文九任务；项目 Egc 的 3,380 个 SMILES 与发布 Egc 4,125 行无共同 SMILES；Xc 虽与发布 CSV 的 432 行逐行相同，论文表 S1 报告 430 个有效样本，缺失/过滤两行身份尚未核实。Eib/Tg 缺项目下游输入。上述任务不进入 `paper5_outer`、不参与论文同折比较；不得把 5 任务结果称为 8 或 9 任务复现。
- 指标为每个 task 每 fold 的独立 outer-test R²，报告五折均值与总体标准差。项目训练和数据曾参与开发，因此这些外层折也不是独立盲测；只可与论文报告作**同样本同折的描述性比较**。五臂之间是项目内部匹配比较，不能将论文架构差异说成 PH 单机制因果效应。

## 实施与验证状态

- `scripts/finetune_glt_3d_gain_d2.py` 的 `resolve_fold` 增加可选预期协议，原入口默认 `outer5_inner20` 不变。`scripts/finetune_mcl_ph.py` 新增 `paper5_outer`，只允许五个已核实任务和 Periodic-TDL 微调策略；选模后才读官方 outer-test。新清单 SHA 与旧 cohort split SHA 分别记录，读取冻结数据仍用其原 split 身份绑定；样本顺序不符即失败。
- 四 GPU launcher 现在只提供 `paper5_outer` 五任务 × 五折，禁止旧 unit 复用；验收和聚合要求官方协议、来源、split SHA、测试预测与 R²，输出 `paper5_test.json`。原 `full8x5`、`full8x5_outer` 的活动微调分支及旧串行 launcher 已移除；历史产物、文档及只读验收兼容保留。
- 局部验证：`pytest -q tests/test_mcl_ph_official_folds.py tests/test_mcl_ph_periodic_tdl_strategy.py tests/test_mcl_ph_8x5_scope.py` 上一轮为 11 passed；本轮收敛后为 **10 passed**。真实冻结 source 对五任务 fold0 的 train/validation/test 单行选择性加载为 `OFFICIAL_FOLD_SOURCE_PASS`；`py_compile`、`git diff --check` 通过。截至启动前，尚未运行任何 r13 模型 forward、训练、outer-test 或聚合。本轮真实 manifest 来源 SHA、旧 cohort 行身份、125 unit 分配、四个部署包与统计 SHA 均已复核通过。具体提交与推送结果见本轮交付。

## 本轮正式运行范围与停止条件

- 用户已明确授权：四 GPU、五臂 × Eea/Egb/Ei/EPS/Nc × 五个官方 outer fold，**125 个全新 unit**；每 unit 10 个 head-only + 60 个 joint epochs，上限 **8,750 epochs / 125 starts**。使用全新 `results/mcl_ph_20260921/p2/paper5_official_r13_1/` 与 `logs/mcl_ph_20260921/paper5_official_r13_1/`。沿用四个原始 5k 预训练包，不从 r12 失败轨迹恢复。r12 的 4 starts / 280 epochs 单独记账。
- 启动前核对五个官方清单和来源 SHA、cohort 绑定、预训练包身份、GPU/进程及新目录。每 unit 退出 0 且 `check_unit` 无问题；125/125 后 `paper5_test.json` PASS 才称完整。任何身份差异、非有限值、测试失败、writer 冲突或预算耗尽立即停止并保留现场；不自动加试或启动 P3/OOF/refit。
- 本轮执行证据以实际 tmux window、日志、退出码、runtime 和聚合产物为准；启动后更新记录。旧策略代码路径移除不改变历史实验结论。
