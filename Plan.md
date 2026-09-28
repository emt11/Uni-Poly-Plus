# REPO-CLEANUP-20260928-01：MCL-PH 产物与日志最小清理

**状态：执行中。** 授权来源：用户 2026-09-28 明确要求制定并执行清理计划，删除当前无用文件并精简日志。执行者：Codex。基线：`cb9e150`，修改前工作树干净，已执行 `git pull --ff-only origin dev`（Already up to date）。

## 目标与边界

- 保留当前正式 `paper5_outer` 的 125 个 unit、五个 `deploy_05000.pt` 使用包、官方 split/config、聚合 JSON、结果文档、代码和失败/事故的可读归档。
- 只清理已被 r13 替换、当前入口不读取的旧 MCL-PH 下游 checkpoint/prediction，以及 p2r2 临时 benchmark/smoke 的大 checkpoint。
- 将 MCL-PH 运行日志压缩为可读摘要：保留原文件名、退出码、launcher 脚本和机器可读 JSON；对大于 100 KiB 的原始日志写入状态、路径、退出码、关键产物和清理说明，不再保留逐 step 输出。
- 不删除源码、配置、数据、当前正式结果、正式 5k 部署包、事故说明或 `PROJECT_HISTORY.md` 中仍有意义的结论。

## 允许清理清单

1. `results/mcl_ph_20260921/p2/full8x5_ptdl_r11_1/`、`full8x5_ptdl_outer_r12_1/`、`full8x5_r10r3_1/`、`full8x5_r10r3_2/`、`full8x5_r10r3_3/` 中的 `best.pt`、`*.npz` 和仅供旧下游恢复的 checkpoint；保留 JSON 指标/聚合与目录说明。
2. `results/mcl_ph_20260921/p2r2_perf_bench/`、`p2r2_perf_bench_rev/`、`p2r2_smoke/` 中的 `*.pt`；保留 README、分析 JSON、step 记录和轻量日志。
3. `logs/mcl_ph_20260921/` 下大于 100 KiB 的日志改写为摘要，保留文件名；不删除 `.exit`、`launch.sh`、正式 paper5 的聚合 JSON 或 unit metadata。

## 禁止操作

- 不删除 `results/mcl_ph_20260921/p2/paper5_official_r13_1/`。
- 不删除 `results/mcl_ph_20260921/p2/pretrain/` 下当前五个正式部署包及其身份/运行记录。
- 不删除 `MCL-PH-INCIDENT-r3-cat-export-hang.md`、`MCL-PH.md`、`RESULTS.md`、`PROJECT_HISTORY.md`、代码、配置或数据。
- 不启动训练、GPU、worker、模型测试、缓存构建或全仓测试；不自动恢复、重跑或扩大实验。

## 最小验证与停止条件

1. 清理前记录候选文件数、字节数和保留路径。
2. 清理后检查正式 paper5 聚合 JSON 仍为 `PASS`、125/125、0 rejected，五个 `deploy_05000.pt` 仍存在，工作树无意外 tracked diff。
3. 只做路径/JSON/hash 存在性核对和 `git diff --check`；不补充防御性测试。
4. 若发现当前入口仍依赖候选文件，立即停止该候选，不删除并记录原因。

## 执行记录

- 2026-09-28 UTC：清理前确认无 MCL-PH 进程、工作树无改动；候选旧 checkpoint/prediction 676 个、37,434,629,606 bytes；大于 100 KiB 的 MCL-PH 日志 228 个、517,538,960 bytes。
- 删除旧 `full8x5` 五个结果根中的 `best.pt`/`*.npz`，以及 `p2r2_perf_bench`、`p2r2_perf_bench_rev`、`p2r2_smoke` 中的 `*.pt`。保留这些目录的 JSON、README、step 记录和轻量结果说明。未触碰正式 paper5 结果与正式 `p2/pretrain` 部署包。
- 将 228 个大日志原地替换为路径稳定的紧凑摘要，保留 `.exit`、`launch.sh`、JSON metadata 和正式结果目录。日志目录从约 530 MiB 降为约 17 MiB。
- 清理后正式结果 SHA 保持 `f9a5aba2a734eff6430a58c8db6ced215259b0e29fa2512ee1a7d40ef25c1894`；paper5 仍为 `PASS`、125/125 accepted、0 rejected；125 个 `best.pt` 与 125 个 `test_predictions.npz` 仍存在；四个正式 5k `deploy_05000.pt` SHA 未变。
- 本轮只做路径、JSON、SHA 和 `git diff --check` 核对；未运行 pytest、模型、训练、GPU、worker 或缓存构建。

## 审查与下一步

- **审查结论：通过，限定为产物清理。** 当前正式 paper5 结果未被修改，当前入口所需的正式包和聚合物仍完整；被删除文件均属于已替换的旧下游 checkpoint/prediction 或临时 benchmark checkpoint。
- 旧历史脚本仍可能引用被清理的实验路径；这些脚本不属于当前 `paper5_outer` 入口，未在本轮恢复或重跑。若要复活历史复现，需另行恢复对应产物或新建授权计划。
- 下一步：暂无 GPU 或模型执行。可选的后续仅是把 `gudhi` 依赖和旧 smoke 入口漂移作为单独工程修复计划，仍需用户授权。
