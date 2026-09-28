# MCL-PH-POSTR13-ENG-01 / r1：MCL-PH 最终工程收口

**状态：已完成。** 授权来源：用户 2026-09-28 要求制定下一步计划并执行，彻底结束 `MCL-PH.md`。规划、执行、自检与审查：Codex（同一主体自检，不称外部独立审查）。基线：`dev@ae227fd`，工作树干净；修改前已核对 remote 为 `emt11/Uni-Poly-Plus`，`git pull --ff-only origin dev` 成功（Already up to date）。上轮 `REPO-CLEANUP-20260928-01` 已归档于 `PROJECT_HISTORY.md`。

## 目标与范围

1. 补齐 MCL-PH 实际使用的 `gudhi` 依赖声明。
2. 移除不能调用当前 `paper5_outer` 微调入口的旧 downstream smoke launcher 及其专属过时断言；历史执行事实保留在 Git 与 `MCL-PH.md`。
3. 在今后的 MCL-PH unit metadata 中明确实际 arm、模型 fusion mode 和所传配置路径；现有 r13 正式产物保持逐字节不变，历史 `config` 字段仍按当时实际传入值解释。
4. 将结果、工程限制和 STOP 决定写入 `MCL-PH.md`，归档本轮后关闭当前计划。

本轮仅为工程与文档收口，不改变模型、科学定义、样本、split、训练策略、评估指标或预算。`RESULTS.md` 与 `PIPELINE.md` 的现有正式结论无需改写。

## 允许与禁止

- 允许修改 `requirements.txt`、`scripts/finetune_mcl_ph.py`、`MCL-PH.md`、`Plan.md`、`PROJECT_HISTORY.md`，并删除已退役的 `scripts/run_mcl_ph_finetune_smoke.sh` 与其旧契约测试 `tests/test_mcl_ph_protocol.py`。
- 不改写 `paper5_official_r13_1`、正式 5k 部署包、官方 split、训练/预测数据或历史事故现场；不恢复已清理的旧 checkpoint。
- 0 GPU、0 worker、0 starts、0 epochs；不启动训练、模型 smoke、P3、OOF、refit、额外 seed、新 outer-test 或缓存构建。

## 最小验证与停止条件

1. `git diff --check`，Python AST/导入与少量受影响的无模型测试。
2. 只读核对 r13 聚合 JSON SHA、125/125 PASS、125 个正式 checkpoint/测试预测和四个 5k 部署包 SHA 未变化。
3. 若发现元数据改动触及训练数值路径，或正式产物身份与既有记录不符，停止受影响修改，保留现场并报告。

## 执行记录

- 2026-09-28 UTC：修改前 `git pull --ff-only origin dev` 成功，基线 `dev@ae227fd`，工作树干净；未发现 MCL-PH 训练进程。
- `requirements.txt` 新增 `gudhi>=3.8,<4`；当前环境版本为 `gudhi 3.8.0`。
- `scripts/finetune_mcl_ph.py` 为未来 unit metadata 增加 `model_variant`、`config_path`、`config_sha256`，并同步写入未来 `best.pt`；不回写 r13 产物。
- 删除已退役的 `scripts/run_mcl_ph_finetune_smoke.sh` 与其旧契约测试 `tests/test_mcl_ph_protocol.py`。该测试初次执行暴露 7 项旧 `smoke/development` 或旧 launcher 断言，未视为新代码回归；删除后由现行 paper5 定向测试覆盖。
- 未修改正式结果、部署包、split、数据、缓存或日志产物；未启动 GPU、worker、训练、模型 smoke、缓存构建或 outer-test。

实际验证：

```text
python3 -m py_compile scripts/finetune_mcl_ph.py scripts/aggregate_mcl_ph_8x5.py scripts/run_mcl_ph_8x5_4gpu.py  PASS
bash -n scripts/run_mcl_ph_pretrain_smoke.sh                                      PASS
git diff --check                                                               PASS
pytest -q tests/test_mcl_ph_official_folds.py tests/test_mcl_ph_periodic_tdl_strategy.py  7 passed, 1 warning
```

## 审查与下一步

- `MCL-PH.md` 已标记最终关闭；r13 的 `paper5_test.json` 仍为 `PASS`、125/125、0 rejected，正式 5k 包和 125 个 unit 预测/权重未被修改。
- 旧 smoke/development 入口及对应过时测试已退役；当前 paper5 入口和定向测试保持可用。
- 本计划审查结论：**通过，已完成，暂无后续执行**。任何未来 MCL-PH 科学实验必须新建计划并重新授权。
