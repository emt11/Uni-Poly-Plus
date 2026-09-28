# MCL-PH-20260921-01 / r13-PAPER5：官方五折五臂评估收口

**状态：已完成。** 用户授权 `paper5_outer` 五臂 × 五个可比任务 × 五折全新运行；Codex 实施、运行与本轮只读复核。基线 `322881b`，代码提交 `4dfb05c`，启动记录 `89c1b2e`；本轮归档与结果文档提交见 Git 历史。原 r11/r12 失败产物保留，未参与本次 125-unit 结果。

## 最终实施与证据

- 官方来源 commit `f3ba6dff6f0d065accdd235dfba160324714f30b`；本项目仅 Eea/Egb/Ei/EPS/Nc 五任务的样本、标签、官方 outer fold 与内部划分可核实匹配。每臂五折；10 个 head-only + 60 个 joint epochs，最低 validation RMSE 选 checkpoint，随后一次性评价 outer-test。四 GPU、125 units / 最多 8,750 epochs。GLT_REF/O8_ONLY 共用 GLT 5k 包，CAT/GATE/XATTN 各用本臂 5k 包。
- 活动 `scripts/finetune_mcl_ph.py`、`scripts/run_mcl_ph_8x5_4gpu.py` 与 `scripts/aggregate_mcl_ph_8x5.py` 仅支持 `paper5_outer`；旧串行 launcher 已删除。局部测试 10 passed；正式运行命令 `logs/mcl_ph_20260921/paper5_official_r13_1_launch.sh`，日志与退出码在同名日志根，产物 `results/mcl_ph_20260921/p2/paper5_official_r13_1/`。
- launcher 真实退出码 0；五臂各 25 个 unit 退出 0；`paper5_test.json` 为 `PASS`、125/125 accepted、0 rejected、`outer_test=RUN`。2026-09-28 只读复算聚合也为 125/125 PASS、身份唯一、8,750 executed epochs、best_epoch 3–69；无活动 MCL 微调进程。未逐个重新 strict-load 125 个 `best.pt`，本轮复核不是外部独立审查。

## 审查结论与下一步

- 正式项目五臂比较完成。macro5 outer-test R²：GLT_REF 0.836666、O8_ONLY 0.835767、CAT 0.829390、GATE 0.829456、XATTN 0.836682。XATTN − GLT_REF = +0.000016，25 个配对折仅 12 个为正；当前证据不支持明确增益。CAT/GATE 的宏平均均低于两个匹配对照。完整逐任务数值与解释边界见 `RESULTS.md`，周期实录见 `PROJECT_HISTORY.md`。
- **下一步：暂无后续执行。** 保存现有原始日志、checkpoint、预测与聚合；如需发表级横向表格，先只读核对论文原表的任务定义、指标和标准差口径，再作描述性比较，预算 0 GPU / 0 starts / 0 epochs。当前路线 STOP 追加正式训练、P3、OOF、refit、额外 seed 或在已用 outer-test 上调参。任何新实验须先提出新假设、独立评估设计、匹配控制及明确预算，再由用户授权。
- 科学边界：官方同折只覆盖五个重合任务；项目架构与论文不同，项目曾利用这些折开发路线，因此不是独立盲测，也不能将五臂差值归因为 PH 单机制。历史早期阶段记载保留在 `MCL-PH.md` 与归档中，不覆盖当时失败和超预算事实。
