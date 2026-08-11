# Uni-Poly 项目级 Codex 准则

## 1. 适用范围与沟通语言

- 本文件适用于整个 Uni-Poly 仓库，是 Codex/Claude Code 在本项目中的长期执行准则。
- 与用户沟通、进度更新、结果解释和项目文档默认使用中文。
- 代码标识符、CLI 参数、文件名、日志原文及论文术语可保留英文；首次出现的关键英文术语应尽量附中文说明。
- Markdown 文档中的块级 LaTeX 公式必须使用 `$$...$$` 包裹，不使用反斜杠方括号形式的分隔符。
- 回答应先给结论，再给必要证据。必须区分“静态审查”“已实现”“测试通过”和“实验验证”，不得把计划或推断写成已完成结果。

## 2. 每次任务开始前

1. 明确用户要求属于审查、解释、制定计划、修改代码、执行任务还是监控任务。
2. 按需读取：
   - `PIPELINE.md`：当前生产模型、数据流和运行方式；
   - `CODEX_CLAUDE_HANDOFF.md`：Codex 规划、Claude Code 执行、Codex 审查的唯一活动交接文档。
3. 修改前检查 `git status` 和相关文件。工作区可能包含用户及其他会话的改动，不得回滚、覆盖或清理无关内容。
4. 启动缓存 writer、预训练或下游训练前，先检查活动进程、目标缓存状态、GPU 占用和断点状态，避免重复任务。
5. 优先定位真实代码路径和当前产物，不依据旧对话、历史文档或文件名猜测实现状态。
6. 制定缓存、迁移、预训练、微调或大规模消融计划时，必须先核对当前机器实际可用的 CPU 核数、内存、GPU 数量与型号、空闲显存、磁盘空间、共享内存和已有任务占用；据此设计 worker 数、batch size、梯度累积、DDP 和任务并发。应在安全与可复现的前提下充分利用可用资源，并给出必要的小规模吞吐基准、时间/内存估算和资源上限，不得脱离当前硬件照搬参数或让资源长期无故闲置。
7. 用户授权执行任务后，实际执行必须位于名为 `Uni-Poly` 的 tmux Session 窗口中：
   - 启动前先执行 `tmux has-session -t Uni-Poly`；Session 不存在时创建 `tmux new-session -d -s Uni-Poly`，不得改用其他 Session、裸后台进程或 `nohup` 规避该要求；
   - 每项任务使用含义明确且唯一的窗口名，启动前检查同名窗口、活动进程、锁和目标产物，避免重复执行；
   - 长任务必须把 stdout/stderr 同步写入明确日志文件，并在进度汇报中记录 Session、窗口名、命令、PID、日志和恢复方式；
   - 静态只读检查和不足一分钟的前台验证可以在当前 shell 执行；会修改产物、使用 GPU、启动 worker，或预计持续一分钟以上的命令必须进入 `Uni-Poly` 窗口；
   - 用户明确要求停止时，应在对应窗口中进行可恢复的正常终止并核验子进程，不得只关闭窗口而遗留孤儿进程。

## 3. Codex 与 Claude Code 协作流程

`CODEX_CLAUDE_HANDOFF.md` 每次只允许存在一个活动执行周期，并按以下职责流转：

1. **Codex 制定计划**
   - 先审查当前代码、产物、活动进程和硬件资源，再写入自包含计划。
   - 计划必须明确目标、现状、修改文件、实施步骤、禁止事项、测试命令、资源配置、停止条件、回滚方式和验收标准。
   - 每次 Codex 新建或更新活动计划时，必须同步清理 `CODEX_CLAUDE_HANDOFF.md` 中与当前执行周期无关、已经失效或可能误导执行的旧计划内容；该文档只保留一个活动计划，以及理解和审计该计划所必需的最小历史证据，不得持续追加并混杂多个执行周期。
   - 将文档状态设为 `ready_for_claude`。此阶段只规划，不执行计划中的生产修改或长任务。
2. **Claude Code 执行计划**
   - 只执行活动周期中已授权的内容，不擅自扩大范围或改变核心语义。
   - 将实际修改文件、命令、测试结果、运行产物、资源使用、错误、偏离计划之处和未完成项写入“Claude Code 执行记录”。
   - 不得改写 Codex 的原始计划来掩盖偏差；遇到停止条件时必须停止并如实记录。
   - 执行结束后将状态设为 `ready_for_codex_review`，不得自行宣布整个方案最终验收通过。
3. **Codex 审查执行结果**
   - 重新检查真实 diff、代码、日志、测试和产物，不只依赖 Claude Code 的文字总结。
   - 将审查结论、发现的问题、验收判断和下一步写入“Codex 审查记录”。
   - 通过时将状态设为 `completed`；需要继续处理时，由 Codex 写入新的活动周期后再交给 Claude Code。

协作边界：

- 用户的最新明确要求始终优先于交接文档。
- Claude Code 只能填写执行记录和执行状态；计划与审查结论由 Codex 负责。
- 已完成周期必须明确标记为历史记录，不能被新的执行会话重复运行。
- 新周期开始前应归档或压缩上一周期，只保留必要证据路径，避免活动指令与历史内容混杂。

## 4. 当前生产路线

当前唯一生产路线为：

```text
MIPS-Trimer-SCAGE（MTS）
内部标识：mips_trimer_scage
```

默认主干：

```text
P-SMILES
→ canonical 单 RU 原子状态
→ lifted periodic relation rows
→ O8 两跳局部 Graph Transformer
→ Trimer Star-RBF 与两层 MCL 几何残差
→ canonical atom mean pooling
→ MD200 低容量图级残差
→ regression head
```

拓扑表示固定支持两种、但不得混用：

```text
默认生产主线：canonical_lifted（单RU节点 + lifted periodic relations）
可运行对照：explicit_k_ru（修正后的最小合法显式k-RU）
```

- 默认生产模型不包含 PBC、FLAT4、GIN、PaiNN、AP3D512 或错误旧显式多 RU 实现；`explicit_k_ru`仅作为具有独立schema/cache/checkpoint身份的修正对照。
- SMILES 和 CountFP 只能作为明确的下游消融模块加入，不能在未声明的情况下改变 graph-only 基线。
- 预训练数据固定使用完整 `PI1M_v2`；下游任务固定为 `eat eea egb egc ei eps nc xc`。
- 模型细节和运行命令以 `PIPELINE.md` 及活动配置为准，本文件只规定不可破坏的长期约束。

## 5. Canonical MTS 图的不变量

- 可学习节点身份只有 `canonical_atom_id`。`relative_ru_shift` 只用于构图、校验和诊断，不进入可学习 embedding。
- 每个 target 必须保留最大两跳内全部 lifted incoming relation rows。
- canonical source/target 相同但 RU shift 不同的 relation row 不得去重；其 multiplicity 必须分别参与 incoming softmax。
- RU 内部键为 `(a,q) ↔ (b,q)`；聚合连接为 `(right,q) ↔ (left,q+1)`。
- 共享 attachment boundary 可以形成带非零 shift 的 canonical self relation；不得把它误删为普通 self-loop。
- SPD 固定表示 O8 的 `0/1/2-hop` 图距离；不得把 hop 当作欧氏距离。
- single-path-node bias 与 Star-RBF 的语义必须独立。Star mask 仅标记直接聚合连接关系。
- Star 虚拟关系不得写入 Trimer 的真实化学键表；`d_star` 只能由 Trimer 中两条真实 inter-RU 键计算。
- Readout 始终按一个 canonical RU 的原子均值定义；显式模式必须先按`canonical_atom_id`聚合全部copy，再做图级均值，不得因repeat factor改变样本权重。
- `canonical_lifted`必须保持默认。`explicit_k_ru`可以在生产模块中构造和运行，但必须使用独立feature schema、LMDB root、model hash和checkpoint；禁止加载错误旧cache，禁止跨representation resume。
- 显式模式中同一canonical atom的全部copy必须共享mask目标和Trimer几何残差；copy只作为有限图节点身份，不得形成可学习copy/方向/shift embedding。

## 6. Trimer 与原子身份

- Trimer 是开放的 `RU(-1)-RU(0)-RU(+1)` 局部 3D 代理，不是周期晶胞，也不宣称为无限聚合物的唯一平衡构象。
- canonical O8 原子只能映射到 Trimer 中央 RU 对应原子；外侧两个 RU 只提供空间环境。
- 禁止把历史 `mips_copy_id` 解释为 Trimer 的 `ru_offset`。
- 禁止依赖 canonical SMILES 或 RDKit 输出的原子顺序进行映射。
- 原子映射必须基于显式身份或经过验证的图同构，至少核验原子序数、形式电荷、芳香性、真实键及键型、attachment site、backbone 和中央 RU 一一对应。
- 存在多个 automorphism 时使用确定性的选择规则；验证失败必须记录并回退，禁止猜测映射。
- `geometry-invalid`、`Star-invalid` 和 `MD200-invalid` 必须分别精确回退；不得删除样本。
- 2D fallback 坐标绝不能进入 MCL 或任何 3D 距离分支。

## 7. Contract、缓存与 Checkpoint

- 所有 schema、builder version 和缓存 contract 的唯一代码来源是：

  ```text
  src/dataset/mips_trimer_contract.py
  ```

- 不得在脚本或文档中维护第二套版本常量；需要报告版本时应从 contract 模块或产物 metadata 读取。
- 旧 schema、错误 hash 或不匹配 artifact 必须在生产入口提前拒绝。
- 禁止使用宽泛的 `strict=False`、忽略 missing keys 或修改 metadata 来掩盖不兼容。
- 冻结缓存是不可变产物。不得修改、追加或覆盖 `.frozen` root；需要新协议时创建新 hash root。
- 同一 LMDB root 只允许一个 writer。训练进程只能只读打开经过验证和冻结的缓存。
- 缓存迁移必须支持断点恢复、单 writer、明确 manifest 和最终一致性验证；不得直接覆盖旧缓存。
- 不得删除历史 cache、checkpoint、结果或日志，除非用户明确指定准确目标并授权删除。
- Checkpoint 必须绑定其实际使用的模型配置、cohort 和 cache artifact；不得仅凭文件名判断兼容性。

## 8. 实验与评价口径

- 正式结果必须覆盖 8 个任务和固定 5 folds；部分 fold 只能称为 smoke 或 screening。
- 项目当前共享 validation/test fold 仅用于历史同口径比较，不是独立盲测；报告中必须明确这一限制。
- 多 seed 实验必须先在预测层聚合，再计算指标；不得直接平均各 seed 的 R² 代替预测集成。
- 最终任务结果使用五折 `mean ± sample std`，展示保留 3 位小数；原始 shard 保留完整精度。
- `best_result.csv` 是比较目标，不得把其中数值当作当前模型已达到的结果。
- coordinate-shuffled、zero、disabled 等配置是负对照。即使负对照分数最高，也不得直接标记为生产模型或物理 3D 改进。
- canonical 单 RU 模型在未完成正式 8 任务五折复评前，不得宣称其性能保持或超过旧多 RU MTS。
- 所有架构和超参数优化必须跨任务统一，不得为单个任务单独调参以追逐结果。

## 9. 修改与执行边界

- 用户要求“审查、解释或分析”时，只进行只读检查，不主动实现修复。
- 用户要求“制定计划”时，只输出或按要求写入计划，不修改生产代码、不启动任务。
- 用户要求“实现”时，完成必要代码修改和与风险匹配的验证，但不自动启动长时间训练。
- 用户明确要求执行训练、迁移或监控时，才允许启动相应长任务；启动后应说明进程位置、日志、断点和停止条件。
- 任何会覆盖、删除、冻结、迁移生产产物或启动多小时任务的操作，都必须确认目标和授权范围。
- 除非用户明确要求，不修改历史结果文件或研究结论。`CODEX_CLAUDE_HANDOFF.md` 只能由当前协作角色按照第3节规定的职责修改。
- 不得为了“顺手清理”进行无关重构、批量格式化或恢复已经退出的路线。

## 10. 验证策略

验证成本与改动风险匹配，使用最小但充分的证据：

- 仅文档修改：检查目标文件内容和一次 diff。
- 局部纯函数修改：运行直接相关的单元测试。
- Dataset、collate 或模型 forward 修改：增加相关测试，并完成至少两样本 forward/backward。
- 原子映射、periodic relation 或几何修改：验证原子身份、边/shift multiplicity、不变性和无效几何精确回退。
- Cache/schema/checkpoint 修改：验证契约匹配、旧产物拒载、resume 和 artifact 完整性。
- DDP、随机掩码或训练恢复修改：执行短程多 rank 一致性与中断恢复测试。
- 只有跨模块或高风险修改才扩大到全量测试；相关代码未再次变化时，不重复已通过的昂贵验证。
- 不主动运行无关 lint、benchmark 或全仓库检查。必要测试通过并检查一次 diff 后应停止。

## 11. 结果汇报

交付时至少说明：

1. 实际修改或检查了什么；
2. 哪些文件受到影响；
3. 执行了哪些验证及其结果；
4. 哪些内容尚未执行；
5. 是否启动了后台任务，以及日志和恢复位置；
6. 结果属于静态判断、smoke、消融还是正式五折实验。

不得用 `py_compile`、单元测试或两样本 smoke 证明模型性能提升；性能结论只能来自对应口径的实际实验结果。
