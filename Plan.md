# REPO-CLEANUP-20260928-02 / r1：Atomic-PC 退役与结果合同收口

**状态：已完成。** 授权来源：用户明确要求直接执行本轮清理，并确认 `Atomic-PC W-CAMR-v2` 可直接删除。执行者：Codex。基线：已在修改前执行 `git pull --ff-only origin dev`，基线 `dev@df2d000`，工作树此前干净。

## 目标与范围

1. 完整退役并删除 Atomic-PC W-CAMR-v2 专属代码、配置、测试和结果目录。
2. 删除日志中 SHA256 完全重复的副本，保留每组一个证据文件。
3. 删除不再被当前路线使用的 Atomic-PC 专属脚本和测试。
4. 将当前保留路线收敛为 GLT-V2 revision-2；共享的 MD200、knowledge fusion 和 sidecar 依赖继续保留。
5. 为新 GLT-V2 实验建立 manifest/unit/aggregate/runtime 结果合同。
6. 在 AGENTS.md 写入完整周期结束后的依赖审查和清理规则。

## 已执行修改

- 删除 Atomic-PC 专属 tracked 文件：
  - `configs/atomic_point_center_ru_v1.json`
  - `configs/atomic_point_v1.json`
  - `scripts/run_original_mips_atomic_pc_w_camr_v2.py`
  - `src/dataset/original_mips_atomic_pc.py`
  - `src/dataset/original_mips_atomic_pc_joint.py`
  - `src/modules/atomic_point_encoder.py`
  - `src/modules/original_mips_atomic_pc.py`
  - `src/training/pretrain/original_mips_atomic_pc_joint.py`
  - `src/training/w_camr_v2_support/`
  - `tests/test_original_mips_atomic_pc_collator.py`
- 删除物理结果目录：`results/original_mips_atomic_pc_v1/`、`results/original_mips_atomic_pc_w_camr_v2/`。
- 删除 16 个 SHA256 完全重复的历史日志副本；每组保留一个副本。
- 从 `src/modules/__init__.py` 移除 Atomic-PC 导出。
- 从预训练 dispatcher 文档中移除已退役 Atomic-PC 入口描述。
- 新增 `src/result_contract.py`，提供 manifest、unit、runtime 和 aggregate 的原子写入接口。
- `scripts/run_mts_finetune_scheduler.py` 在新结果根创建 manifest，并在完成后写入 runtime 状态。
- 更新 `PIPELINE.md`、`RESULTS.md`、`AGENTS.md`。

## 保留边界

- `src/modules/original_mips_knowledge_fusion.py`、`src/modules/original_mips_md200.py`、`src/dataset/md200_sidecar.py` 仍被 GLT-V2 使用，不能删除。
- MTS-GLT-v2 历史 checkpoint 作为正式历史证据保留，不再作为 Atomic-PC 运行时依赖。
- MCL-PH 正式结果和历史事故证据未修改。
- 未启动训练、GPU、worker、模型 smoke、缓存构建或结果重算。

## 最小校验

- `git diff --check`
- 变更 Python 文件 `py_compile`
- 搜索已删除 Atomic-PC 路径的残留引用
- 新结果合同临时目录的 manifest/unit/runtime/aggregate 小 smoke
- 检查当前 GLT-V2 共享 import 不再导入已删除 Atomic-PC 模块

## 停止条件与验收标准

- 若当前 GLT-V2 入口仍依赖被删除文件，停止删除并恢复该共享文件；本轮已确认 knowledge fusion、MD200 和 md200 sidecar 仍有 GLT-V2 引用，因此保留。
- `aggregate.json` 不得在 unit 未全部 PASS 时生成。
- 删除后无当前入口、配置、测试或结果文件引用已删除路径。
- `git diff --check` 和最小校验通过后停止，不追加全仓测试。

## 执行记录

- 已完成 Atomic-PC 专属代码、配置、测试和两个结果根的删除。
- 已删除 16 个 SHA256 完全重复日志副本，每组保留一个文件。
- `git diff --check`：PASS。
- `python -m py_compile src/result_contract.py scripts/run_mts_finetune_scheduler.py src/modules/__init__.py src/training/pretrain/engine.py`：PASS。
- 结果合同临时 manifest/unit/runtime/aggregate smoke：PASS。
- Atomic-PC 代码引用扫描：PASS（历史文档中的删除路径说明保留）。
- GLT-V2 共享 import（`src.modules.glt_dual`、`src.dataset.md200_sidecar`）：PASS。
- 未运行训练、GPU、worker、模型 smoke、缓存构建、结果重算或全仓测试。


## 审查与下一步

最小校验通过，当前周期可归档并推送 `dev`。暂无新的科学实验计划。
