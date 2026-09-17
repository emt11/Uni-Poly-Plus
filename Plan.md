# 当前计划：GLT-V2 下一科学周期——固定条件下的融合比较（待授权）

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `GLT-SCI-20260917-01 / r1` |
| 状态 | **待授权**；未启动本周期实验 |
| 规划角色 | Codex 负责科学问题、边界与后续审查；执行者待用户授权后按本计划实施 |
| 当前基线 | `dev`；SPEED-20260917-01/r4 已由 Codex review PASS / CLOSED |
| 归档入口 | `PROJECT_HISTORY.md` 中的 `SPEED-20260917-01` 条目 |

## 1. 上一周期关闭状态

`SPEED-20260917-01 / r4` 已完成 **Codex review PASS / CLOSED**。其范围仅覆盖提速执行合同与有界证据：worker-resume RNG、GPU slot 合同、clean-cache 转发与边界、以及 4-GPU bounded smoke。最终实现提交为 `eda9f5c`，执行记录补记提交为 `f19ef02`。

该关闭不等于正式实验完成：微调提速有 bounded evidence，预训练正式 throughput 未测；没有正式 5k/20k 预训练、完整 8×5 微调或模型性能结论。本轮不启动 speed r5。

## 2. 科学问题与参考

本周期拟回答：在相同数据、切分、训练预算和 encoder 初始化条件下，GLT-V2 的 Concat 与 KFuse 融合方式是否造成可重复的下游性质预测差异？

| 比较项 | 约束 |
| --- | --- |
| reference | 以当前 checkout 中可核实的 GLT-V2 frozen cache、固定 `outer5_inner20` split 及可用 deploy provenance 为准；正式执行前重新核对路径、hash 与 checkpoint metadata |
| controlled change | 仅改变 fusion mode（Concat 或 KFuse）；不同时改变 encoder、任务头、损失、标签变换、数据划分或模型宽度 |
| 保持一致 | 相同样本 key 顺序、缓存版本、随机种子策略、train-only scaler、optimizer/LR 规则、epoch/patience、validation 选择和 outer-test 评估口径 |
| 结果口径 | partial smoke 只能称 smoke；完整五折结果才可用于固定 OOF/macro8 比较，不以预训练 loss 或单 fold 结论替代 |

## 3. 授权后实施顺序

1. **Preflight**：读取最新 `AGENTS.md`、本计划、实际配置、frozen cache manifest、deploy metadata 和 split；确认无身份不一致、覆盖风险或活动任务冲突。
2. **小范围核验**：若用户授权 smoke，先固定 task/fold/epoch/seed 和输出目录，验证两种 fusion 的加载、标签形状、有限 loss、validation 选择及 `outer-test=NOT_RUN`。
3. **正式比较（另需明确授权）**：只有用户明确给出正式预训练包、训练预算与 8 task × 5 fold 范围后，才可分别执行 Concat/KFuse，并逐 fold 保存独立 checkpoint、预测和 provenance。
4. **汇总与审查**：检查每个 property row 恰好一条 OOF prediction、无 NaN、任务数和 fold 数完整，再计算 per-task fold mean/std、pooled OOF 与 macro8；Codex 依据固定比较条件审查，不自动解释因果或性能优越性。

## 4. 当前禁止事项

在用户提供明确授权前，不启动正式预训练、完整微调、正式 grid、outer-test 推理、缓存重建/迁移、构象生成、speed r5、MD200、教师蒸馏、O8-only 路线或任何新模型架构/数据划分/损失定义。不得覆盖历史 checkpoint、results、cache 或报告，也不得把历史 smoke 当作本周期新证据。

## 5. 验收与停止条件

授权后必须记录实际命令、分支/commit、tmux window、日志和产物路径。发现 cache identity/schema 不一致、标签或样本静默丢失、NaN/Inf、DDP 锁死、checkpoint 无法恢复或 outer-test 越权访问时，停止受影响步骤并报告；不通过放宽检查或扩大预算掩盖问题。

本计划没有为正式实验预先授权具体 GPU 数量、step/epoch、task/fold 或比较预算；这些参数须在执行前由用户明确确认。完成 smoke 不得写成正式性能结果，完成一组 fusion 也不得自动启动另一组或完整 OOF。

## 6. 当前执行记录与下一步

- 本次仅归档上一周期并切换计划；未修改 `.py`、config、cache、results 数字，未运行测试、benchmark、模型或训练。
- 当前状态为 **待授权**，没有活动的本周期任务。下一步是用户明确科学预算与数据/模型包后，由 Codex 补充可执行修订并交执行者；在此之前不启动任何命令。
