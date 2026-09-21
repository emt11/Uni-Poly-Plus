# 事件说明：r3 阶段二 cat 臂导出阶段挂起与人工终止

- 日期：2026-09-21 UTC
- 范围：计划 `MCL-PH-20260921-01` / r3 第二阶段，`M_CAT` 预训练 smoke（cat 臂）
- 性质：**独立事件说明（由执行者 ZCode 事后外部书写）**。本文件不是 runner 产出的状态文件，不替代 `runtime.json`，也不修改它。
- 相关记录：[MCL-PH.md](MCL-PH.md) §13.3「第二阶段执行记录」。

## 1. 事件经过

| 时刻 (UTC) | 事件 | 证据 |
| --- | --- | --- |
| 12:56:02 | launcher 启动 cat 臂（`ARMS="cat gate xattn" UPDATES=2 NPROC=4 PREP_WORKERS=12 OUTPUT=results/mcl_ph_20260921/p1/pretrain_r3b`） | `logs/mcl_ph_20260921/p1_pretrain_smoke_r3b.log` 第 1–2 行 |
| 12:56:1x | 四个 rank 各自写出 step 1 记录（step 用时 14.67 s，其中数据准备 11.4 s） | 同上（rank 0–3 的 step 1 记录） |
| 12:56:42 | 四个 rank 各自写出 step 2 记录（step 用时 0.36 s）；rank 0 写出 `resume_00002.pt`（315,570,711 B） | 同上；`results/mcl_ph_20260921/p1/pretrain_r3b/cat/resume_00002.pt` |
| 12:56:42 → 13:02:0x | **挂起**：未生成 `deploy_00002.pt`；`runtime.json` 保持 `RUNNING`；rank 0 无 IO、无 deploy 文件描述符，rank 1–3 处于集体通信等待 | `logs/mcl_ph_20260921/p1_pretrain_r3b_hang_evidence.txt` |
| 13:01:4x | 执行者对 rank 0 发送 `SIGINT`：20 s 内无响应（阻塞在 C 调用，未回到 Python 循环） | 同上（SIGINT 后进程状态快照） |
| 13:02:08 | 执行者发送 `SIGTERM`，torchrun 与四个 rank 全部退出 | `logs/mcl_ph_20260921/p1_pretrain_smoke_r3b.log`：`=== ARM=cat EXIT=1 ===`、`=== ABORT after cat (exit 1); no rerun within this budget ===` |

## 2. 为什么是人工终止，而不是 runner 自己写失败记录

- runner 的失败处理（`scripts/pretrain_mcl_ph.py` 末尾的 `except BaseException`）会在**异常/信号被 Python 捕获**时写 `runtime_failure_rank{rank}.json` 与 rank 0 的 `runtime.json`（状态 `FAILED`）。
- 本次 rank 0 阻塞在 C 层等待（`futex_wait_queue`），`SIGINT` 未被处理，随后 `SIGTERM` 直接结束进程 → 失败处理**没有机会运行**，因此磁盘上不存在 runner 写的终态记录。
- 证据获取手段受限：`py-spy`、`/proc/<pid>/task/<tid>/stack`、`gdb -p` 均因 `ptrace` 限制不可用，**没有取得 Python 调用栈**。

## 3. 状态如何解读（避免误读）

| 字段 | 现状 | 正确解读 |
| --- | --- | --- |
| `cat/runtime.json` | `status: RUNNING` | **不是**「仍在运行」，也**不是** runner 记录的 `FAILED` 或 `PASS`；是进程被外部终止后遗留的未终态记录。本事件说明即为该歧义的唯一补充证据 |
| `cat/run.json` | 存在 | 该 run 已启动并建立身份记录（在 world-size 守卫与输出目录创建之后写入），**不代表训练完成** |
| `cat/deploy_00002.pt` | **不存在** | 导出未完成；任何后续流程不得假定存在导出包 |
| `cat/steps.jsonl` | 2 条（step 1、2，rank 0） | 真实完成了 2 次优化器更新；四个 rank 的控制台记录一致 |
| `cat/resume_00002.pt` | 存在 | 中断点现场；其可加载性与内容核验见 r4 记录（§13.4） |

**没有伪造任何 PASS 或 FAILED 记录**：本事件说明、`incident_operator_stop.json` 与 MCL-PH.md 的记述均为外部书写，与 runner 的产物在文件上互相区分。

## 4. 未决问题

- 挂起**未根因化**：无 Python 栈，只有线程状态、IO、GPU 利用率与阶段边界证据。
- r1 的 `cat_failed_rng_collective` 与本事件处于**同一阶段**（step 记录写入后、导出完成前），但**同因未经证实**，不得写成同一根因。
- r4（`MCL-PH-20260921-01/r4`）在不恢复训练的前提下补最小取证、CPU 离线导出与有界四 rank 复现，结论见 MCL-PH.md §13.4。

## 5. 证据路径

| 内容 | 路径 |
| --- | --- |
| launcher 日志（含真实退出码与 `ALL_ARMS_OK` 缺失） | `logs/mcl_ph_20260921/p1_pretrain_smoke_r3b.log` |
| 挂起现场快照（进程树、线程状态、GPU、目录、`runtime.json`、日志尾） | `logs/mcl_ph_20260921/p1_pretrain_r3b_hang_evidence.txt` |
| 现场产物（保持原样，未补写终态） | `results/mcl_ph_20260921/p1/pretrain_r3b/cat/` |
| 机器可读标记（放置在产物目录旁） | `results/mcl_ph_20260921/p1/pretrain_r3b/cat/incident_operator_stop.json` |
