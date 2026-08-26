# Uni-Poly 项目级协作准则

## 1. 基本规则

- 默认使用中文沟通和编写项目文档；代码、CLI、日志和论文术语可保留英文。
- Markdown 块级公式使用 `$$...$$`。
- 用户当前明确要求优先于项目文档和历史计划。
- `TODO.md` 仅供用户个人记录，不作为任务授权、计划或验收依据。
- 明确区分：静态审查、计划、实现、测试、smoke、screening、消融和正式实验。

发生冲突时按以下优先级判断：

1. 用户当前要求；
2. `AGENTS.md`；
3. Codex 当前活动计划或 Claude Code 当前 handoff；
4. 真实代码、配置、checkpoint metadata 和实际产物；
5. `PIPELINE.md`、`RESULTS.md`；
6. 历史文档和旧对话。

不得根据模型名、目录名或历史描述猜测当前实现。

---

## 2. 工程与执行原则

默认采用：

**定位问题 → 最小修改 → 立即验证 → 根据结果继续。**

- 不顺手重构、不扩大范围、不提前建设复杂框架。
- 默认不新增 hash、schema、gate、迁移器等防御机制；只有现实故障证明现有机制不足时才增加。
- 修改前检查相关代码和 `git status`，不得回滚、覆盖或清理无关改动。
- 解释、审查、计划默认只读；用户授权实现后再修改代码或运行任务。
- 遇到 artifact/checkpoint/schema 不匹配、NaN/Inf、writer 冲突或可能覆盖正式产物时停止并报告，不绕过检查继续。

预计超过一分钟、使用 GPU、启动 worker 或修改训练产物的任务在 `tmux` Session `Uni-Poly` 中运行，并保留日志。

---

## 3. 协作方式

同一活动范围同一时间只能有一个生产执行者。

### Codex

负责：

- 检查真实状态；
- 规划任务；
- 控制范围；
- 科学判断；
- 审查代码、测试、日志和产物；
- 决定下一步。

### `luna_worker`

`luna_worker` 是 Codex 的直接执行子代理，不使用 handoff 文件。

负责：

- 执行 Codex 指定的代码修改、命令、测试、smoke、audit 或已授权实验；
- 收集日志和产物；
- 返回真实执行结果。

不得：

- 自行扩大范围；
- 自行设计或启动后续实验；
- 自行决定正式模型或最终科学结论；
- 回滚或清理无关修改。

流程：

**Codex 规划 → Luna 执行 → Codex 审查 → 再决定下一步。**

Codex 与 Luna 不重复执行同一范围。

### Claude Code

Claude Code 作为独立执行者时使用 `CODEX_CLAUDE_HANDOFF.md`。

Codex 制定计划，Claude Code 执行，Codex 最终审查；Claude Code 不自行改写计划或宣布最终验收通过。

---

## 4. 科学实验与正式路线

当前生产模型、baseline、数据语义和运行参数以：

- `PIPELINE.md`
- 当前配置
- 真实代码
- checkpoint metadata
- 正式结果产物

为准。

已退役实验不得自动重新进入生产路线。

科学比较必须明确：

- scientific question
- baseline
- matched control
- treatment
- mechanism increment
- total effect

若 `B = baseline`、`C = matched control`、`T = treatment`：

- `C - B`：generic architecture / capacity / adaptation effect
- `T - C`：mechanism increment
- `T - B`：total effect

不得混淆。

此外：

- `fused - baseline` 不能自动称为 interaction。
- 改变 encoder 输入语义时，优先使用 matched pretraining / matched training。
- screening 只用于决定是否值得继续，不自动替换正式 baseline。
- 部分 task/fold/epoch 或小样本结果只能称为 smoke、screening 或消融。
- 已参与模型开发的共享 validation/test fold 不是独立盲测。
- `results/best_result.csv` 的宏平均只是逐任务最优包络，不代表任何单模型。
- 候选未完成正式复评前，不宣称已超过 baseline。
- 已有充分失败证据的方向优先 STOP，不机械 sweep。

---

## 5. 验证与汇报

只验证本次修改可能破坏的行为：

- 局部代码：相关单元测试；
- Dataset/collate/forward：小样本 forward/backward；
- periodic/geometry：验证本次涉及的 identity、shift、multiplicity、不变性或 fallback；
- training/GPU/scheduler：相关短 smoke；
- 性能优化：先短 benchmark，再决定是否继续。

不主动运行无关全仓测试或昂贵实验。

交付时说明：

1. 做了什么；
2. 修改了哪些文件；
3. 运行了什么测试/实验及结果；
4. 尚未执行什么；
5. 长任务的 tmux/日志（如有）；
6. 结论属于测试、smoke、screening、消融还是正式实验。

不得用 `py_compile`、单元测试或短 smoke 宣称模型性能提升。