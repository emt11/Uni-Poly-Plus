# Uni-Poly 项目级 Codex 准则

## 1. 适用范围与沟通语言

- 本文件适用于整个 Uni-Poly 仓库，是 Codex、Claude Code 与 `luna_worker` 在本项目中的长期执行准则。
- 与用户沟通、进度更新、结果解释和项目文档默认使用中文。
- 代码标识符、CLI 参数、文件名、日志原文及论文术语可保留英文；首次出现的关键英文术语应尽量附中文说明。
- Markdown 文档中的块级 LaTeX 公式必须使用 `$$...$$` 包裹，不使用反斜杠方括号形式的分隔符。
- 回答应先给结论，再给必要证据。必须区分“静态审查”“已实现”“测试通过”和“实验验证”，不得把计划或推断写成已完成结果。

## 2. 每次任务开始前

1. 明确用户要求属于审查、解释、制定计划、修改代码、执行任务还是监控任务。
2. 按当前任务与协作路径读取：
   - `PIPELINE.md`：当前生产模型、数据流和运行方式；
   - 进入第 3.1 节流程时，只读取 `CODEX_CLAUDE_HANDOFF.md`；
   - 进入第 3.2 节流程时，只读取 `CODEX_LUNA_HANDOFF.md`。该文件由 Codex 保存交给 `luna_worker` 的计划，并在执行结束、Codex 审查后记录结论和下一步；`luna_worker` 只读。文件名区分大小写，统一使用 `.md`；
   - 除非需要检查重复委派或冲突，不得依赖另一执行者的 handoff 制定当前计划。
3. 修改前检查 `git status` 和相关文件。工作区可能包含用户及其他会话的改动，不得回滚、覆盖或清理无关内容。
4. 启动缓存 writer、预训练或下游训练前，先检查活动进程、目标缓存状态和 GPU 占用，避免重复任务。
5. 优先定位真实代码路径和当前产物，不依据旧对话、历史文档或文件名猜测实现状态。
6. 仅当计划实际涉及缓存构建、训练、迁移或大规模消融时，才核对与本次任务直接相关的硬件资源；不要求为普通代码修改罗列全部 CPU、内存、磁盘和共享内存信息。资源配置应以一次轻量检查或短 benchmark 为依据，避免脱离当前硬件照搬参数。
7. 用户授权执行任务后，凡符合以下长任务或产物修改条件的执行命令，必须位于名为 `Uni-Poly` 的 tmux Session 中：
   - 启动前先执行 `tmux has-session -t Uni-Poly`；Session 不存在时创建 `tmux new-session -d -s Uni-Poly`，不得改用其他 Session、裸后台进程或 `nohup` 规避该要求；
   - 每项任务使用含义明确且唯一的窗口名，启动前检查同名窗口、活动进程、锁和目标产物，避免重复执行；
   - 长任务必须把 stdout/stderr 同步写入明确日志文件，并在进度汇报中记录 Session、窗口名、命令、PID 和日志；
   - 静态只读检查和不足一分钟的前台验证可以在当前 shell 执行；会修改产物、使用 GPU、启动 worker，或预计持续一分钟以上的命令必须进入 `Uni-Poly` 窗口；
   - 用户明确要求停止时，应在对应窗口中正常终止并核验子进程，不得只关闭窗口而遗留孤儿进程。

### 2.1 计划制定原则

- 计划以“最短可执行路径”为默认：只包含实现目标所必需的修改、验证和产物，不为尚未出现的假设风险预先增加复杂机制。
- 先区分核心验收、可选增强和诊断信息。只有会导致数据损坏、科学语义错误、产物混用或任务无法运行的问题才能设为阻塞门；性能观测、完整性增强和边缘场景验证通常不得阻止主要目标交付。
- 一个执行周期原则上最多设置 1 至 3 个核心阻塞门。超过时必须说明每个门对应的真实故障风险；不能仅以“更严格”“更稳妥”作为理由。
- 不默认要求精确恢复、逐 bit 一致、完整 loss/gradient parity、全量测试或正式全任务复评。只有用户明确要求，或本次改动直接修改对应能力且失败会使当前产物不可用时，才把它们列为验收条件。
- 性能优化优先使用短 benchmark、有限 smoke 和 finite 检查证明“速度就绪”；除非优化改变精度、数据或模型语义，否则不附加科学等价性实验。smoke 仍不得冒充正式性能结论。
- 不因 GPU 编号、worker 数、日志目录、调度 slot 等运行元数据变化自动要求重新训练、创建新科学身份或执行恢复门。只有实际改变模型、数据、优化器轨迹或产物兼容性时才升级身份约束。
- schema、hash、sidecar、迁移器、回滚脚本和新 checkpoint 身份只在真实存在兼容性边界时引入，不为普通局部修改建立新的协议层。
- 用户可以明确豁免非破坏性的验收门。此时记录“未验证/由用户豁免”并继续其余任务，不得把豁免伪报为通过，也不得反复加入等价替代门。
- 计划应允许候选失败后保留当前默认并交付已完成实现；除非失败会污染生产产物，否则不得把一个候选未晋级扩大为整个计划失败。
- 对可从 git 或隔离输出目录自然恢复的普通改动，不要求额外设计回滚流程；只有不可逆写入、冻结、覆盖、删除或正式迁移才需要明确回滚方案。

## 3. Codex 制定、执行者实施、Codex 审查的协作流程

项目支持以下两种独立流程：Claude Code 流程使用 `CODEX_CLAUDE_HANDOFF.md`，`luna_worker` 流程使用 `CODEX_LUNA_HANDOFF.md`。两个文件各自只保留一个活动执行周期；不得让 Claude Code 与 `luna_worker` 同时执行同一计划，也不得交叉填写对方的交接文件。

一旦当前任务进入第 3 节的委派流程，生产修改由该流程指定的唯一执行者负责，Codex 不重复执行同一计划。第 2、4–11 节仍是所有参与者共同遵守的约束。

### 3.1 Codex 与 Claude Code 协作流程

1. **Codex 制定计划**
   - 先审查与任务直接相关的代码和产物；只有任务涉及运行资源或长任务时才检查活动进程和硬件，再写入自包含计划。
   - 计划只需明确目标、必要修改、实施步骤、与风险匹配的验证及完成标准。禁止事项、资源上限、停止条件和回滚方式仅在本次任务确实需要时加入。
   - 优先给出可在当前周期完成的主路径；可选增强和未来工作应单独列出，不得混入本周期验收条件。
   - 每次 Codex 新建或更新活动计划时，必须同步清理 `CODEX_CLAUDE_HANDOFF.md` 中与当前执行周期无关、已经失效或可能误导执行的旧计划内容；该文档只保留一个活动计划，以及理解和审计该计划所必需的最小历史证据，不得持续追加并混杂多个执行周期。
   - 将文档状态设为 `ready_for_claude`。此阶段只规划，不执行计划中的生产修改或长任务。
2. **Claude Code 执行计划**
   - 只执行活动周期中已授权的内容，不擅自扩大范围或改变核心语义。
   - 将实际修改文件、命令、测试结果、运行产物、资源使用、错误、偏离计划之处和未完成项写入“Claude Code 执行记录”。
   - 不得改写 Codex 的原始计划来掩盖偏差；遇到核心阻塞门时停止并如实记录。非阻塞候选失败时应保留默认、记录结果并继续其他独立步骤。
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

### 3.2 Codex 与 `luna_worker` 协作流程

当用户明确要求使用 `luna_worker` 执行，或 Codex 已获得用户对该执行方式的明确授权时，采用以下流程：

1. **Codex 制定计划并委派**
   - Codex 先检查真实代码、现有改动和相关产物，制定执行计划并写入 `CODEX_LUNA_HANDOFF.md`。
   - 文档应明确本轮目标、允许修改的范围、不得覆盖的用户或其他会话改动、验证要求，以及必要的运行或日志要求。
   - Codex 随后将该计划交给 `luna_worker` 执行。
2. **`luna_worker` 执行**
   - `luna_worker` 完整读取 `AGENTS.md` 和 `CODEX_LUNA_HANDOFF.md`，按照其中当前计划完成实现、测试、运行和必要的监控。
   - `luna_worker` 不扩大计划中的目标，不回滚或覆盖未授权的现有改动。
   - 执行完成或遇到阻塞后，将实际修改、测试结果、产物、日志、错误及未完成项报告给 Codex。
   - `luna_worker` 不负责最终验收，也不修改 `CODEX_LUNA_HANDOFF.md`。
3. **Codex 审查并更新**
   - Codex 在 `luna_worker` 执行结束后，独立检查真实 diff、代码、测试、日志和产物，而不是仅依赖执行报告。
   - 审查完成后更新 `CODEX_LUNA_HANDOFF.md`，记录本轮实际结果、验收结论、遗留问题和下一步。
   - 若任务完成，则标记为 `completed`；若需要继续，则直接写入下一轮计划；若存在无法继续的阻塞，则记录阻塞原因和解除条件。

原则上，`CODEX_LUNA_HANDOFF.md` 只由 Codex 维护：执行前写计划，执行并审查后更新结果或下一计划。

状态约定仅使用 `ready_for_luna_worker`、`completed` 和 `blocked`：Codex 写好新计划后设为 `ready_for_luna_worker`；审查成功且无需后续工作时设为 `completed`；无法继续且暂时不能制定下一轮计划时才设为 `blocked`。`luna_worker` 通过子代理返回后由 Codex 立即审查，不增加 `ready_for_codex_review` 状态。

两种流程的共同边界：

- 同一计划只能有一个执行者；需要更换执行者时，必须由 Codex 先停止或结束原委派并明确新的任务边界，避免重复执行。
- Claude Code 与 `luna_worker` 都无权修改计划目标或 Codex 审查结论；可以记录实际情况和提出建议。
- 执行者完成的是“计划执行”，不是“最终验收”；最终验收责任始终属于 Codex。
- 用户的最新明确要求、保护用户改动、避免重复执行和如实区分 smoke/正式结果等规则，对两种流程同样适用。
- `CODEX_CLAUDE_HANDOFF.md` 与 `CODEX_LUNA_HANDOFF.md` 不能互相替代。Luna 流程中，计划以及 Codex 审查后的最终执行结论和下一步保存在 `CODEX_LUNA_HANDOFF.md`；worker 的中间过程无需持续写入该文件。

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
→ 前四层 O8、最后两层 MSTA 的 Graph Transformer
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
- `canonical_lifted`必须保持默认。`explicit_k_ru`可以在生产模块中构造和运行，但必须使用独立feature schema、LMDB root、model hash和checkpoint；禁止加载错误旧cache或混用不同 representation。
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
- 用于模型比较和科学结论的架构与科学超参数必须跨任务统一，不得针对单个下游任务单独调优以追逐结果；纯运行参数可根据资源和任务规模调整，但不得改变比较语义。

## 9. 修改与执行边界

- 用户要求“审查、解释或分析”时，只进行只读检查，不主动实现修复。
- 用户要求“制定计划”时，只输出或按要求写入计划，不修改生产代码、不启动任务。
- 用户明确授权“实现”后，可以直接运行完成该实现及其必要验收所需的测试和训练，无需再次请求同义授权；启动长任务后应说明进程位置、日志和停止条件。
- 用户明确要求执行训练、迁移或监控时，按其已授权范围执行，不得擅自扩大为无关实验。
- 若计划进一步启动超出当前实现验收范围的多小时正式训练、批量实验或生产迁移，必须确认其目标已包含在用户授权范围内。
- 任何会覆盖、删除或冻结生产产物的操作，都必须确认准确目标和授权范围。
- 除非用户明确要求，不修改历史结果文件或研究结论。`CODEX_CLAUDE_HANDOFF.md` 只由第 3.1 节流程中的 Codex 与 Claude Code 修改；`CODEX_LUNA_HANDOFF.md` 只由第 3.2 节流程中的 Codex 修改，`luna_worker` 对其只读。
- 不得为了“顺手清理”进行无关重构或批量格式化。

## 10. 验证策略

验证成本与改动风险匹配，使用最小但充分的证据：

- 仅文档修改：检查目标文件内容和一次 diff。
- 局部纯函数修改：运行直接相关的单元测试。
- Dataset、collate 或模型 forward 修改：增加相关测试，并完成至少两样本 forward/backward。
- 原子映射、periodic relation 或几何修改：按本次实际涉及的语义验证原子身份、边/shift multiplicity、不变性及无效几何精确回退。
- Cache/schema/checkpoint 修改：验证契约匹配、旧产物拒载和 artifact 完整性。
- 普通训练循环、GPU 映射、worker 或调度修改：运行直接相关的短 smoke 即可；不自动要求中断恢复或逐 bit 一致性测试。
- AMP 或数值精度修改：使用代表性小批次检查 finite 和任务实际需要的容差；只有用户明确要求时才扩大为完整 loss/gradient parity gate。
- 只有跨模块或高风险修改才扩大到全量测试；相关代码未再次变化时，不重复已通过的昂贵验证。
- 不主动运行无关 lint、benchmark 或全仓库检查。必要测试通过并检查一次 diff 后应停止。
- 测试失败应先判断是否由本次修改引起。无关的迁移 fixture、历史缺失产物或已知环境问题应如实记录，但不得默认阻塞当前局部任务，也不得为了让测试变绿而放宽生产契约。

## 11. 结果汇报

交付时至少说明：

1. 实际修改或检查了什么；
2. 哪些文件受到影响；
3. 执行了哪些验证及其结果；
4. 哪些内容尚未执行；
5. 是否启动了后台任务，以及日志位置；
6. 结果属于静态判断、smoke、消融还是正式五折实验。

不得用 `py_compile`、单元测试或两样本 smoke 证明模型性能提升；性能结论只能来自对应口径的实际实验结果。
