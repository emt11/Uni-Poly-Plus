# Uni-Poly 项目级协作准则

## 1. 适用范围与沟通

- 本文件适用于整个 Uni-Poly 仓库，是 Codex、Claude Code 与 `luna_worker` 的长期协作准则。
- 默认使用中文沟通和编写项目文档；代码标识符、CLI 参数、日志原文和论文术语可以保留英文。
- Markdown 块级公式使用 `$$...$$`。
- 回答先给结论，再给必要证据；明确区分静态审查、计划、已实现、测试通过、smoke 和正式实验结果。
- 用户最新的明确要求优先于项目文档和历史计划。

## 2. 工程原则

默认采用**敏捷开发、最小改动、动态迭代**，优先执行、模拟、测量和针对性测试，不默认套用完整软件开发流程。

- 默认不新增 hash、冻结 contract、baseline 或 gate。
- 只有能明确说明**具体失败场景**，并解释 Git、版本号、主键、事务、唯一约束、类型/schema 和普通测试为何不足时，才允许新增上述机制。
- Gate 只用于**不可逆、跨系统、安全敏感或正式发布**边界；普通开发问题使用直接测试和观测解决。
- 前置检查不得挤占真正的代码执行、模拟、实验或性能测量。
- 优先采用：**定位问题 → 最小修改 → 立即验证 → 根据结果继续迭代**。
- 不为未来可能的需求提前建立复杂框架、抽象层、迁移器、回滚脚本或防御机制。
- 现有且正在使用的科学 contract、冻结产物和 checkpoint 身份继续遵守；本节限制的是没有现实故障依据的新机制。
- 一个实施周期只处理用户当前目标；可选增强和未来工作单独记录，不混入当前验收。

> 核心原则：**先证明风险，再增加保护；先运行和测量，再增加抽象。**

## 3. 开始任务与执行边界

### 3.1 最小上下文检查

1. 先判断任务属于解释、审查、计划、实现、执行训练还是监控。
2. 只读取当前任务需要的材料：
   - 涉及生产模型、数据流或运行方式时读取 `PIPELINE.md`；
   - Claude Code 流程只读取 `CODEX_CLAUDE_HANDOFF.md`；
   - `luna_worker` 流程只读取 `CODEX_LUNA_HANDOFF.md`；
   - 只有检查重复委派或冲突时才读取另一条流程的 handoff。
3. 修改前检查相关文件和 `git status`。工作区可能包含用户或其他会话的改动，不得回滚、覆盖或清理无关内容。
4. 优先检查真实代码、当前配置和实际产物，不根据旧对话、历史文档或文件名猜测状态。
5. 只有任务涉及 GPU、worker、缓存 writer、长训练或大规模产物时，才检查相应进程和资源；普通代码修改不做全套硬件审计。

### 3.2 用户授权边界

- 用户要求解释、审查或分析时，只做只读检查。
- 用户要求制定计划时，只输出计划或写入指定计划文件，不修改生产代码、不启动任务。
- 用户明确要求实现后，可以完成必要代码修改、针对性测试和验收 smoke，无需再次请求同义授权。
- 用户授权的实现如果本身包含必要的长时间训练，可以直接启动；超出当前目标的正式训练、批量实验、迁移、删除或覆盖仍需有明确授权。
- 不进行顺手重构、无关格式化、历史结果改写或范围扩张。

### 3.3 长任务

- 会修改训练产物、使用 GPU、启动 worker 或预计超过一分钟的命令，在 `tmux` Session `Uni-Poly` 中运行；Session 不存在时创建，静态检查和短前台测试可直接执行。
- 每项长任务使用唯一且有含义的窗口名，日志同步落盘。启动前只检查与该任务直接相关的同名窗口、进程、锁、GPU 和目标产物。
- 汇报 Session、窗口、命令、PID、日志路径和停止条件。
- 用户要求停止时，应终止相应任务并确认子进程已退出，不留下孤儿进程。

## 4. 委派协作流程

同一计划只能有一个生产执行者。进入委派流程后，Codex 负责计划和最终审查，指定执行者负责实现；Codex 不重复修改同一范围。第 2、3、5–9 节对所有参与者均有效。

### 4.1 Codex 与 Claude Code

1. Codex 检查相关代码和现状，将一个自包含的活动计划写入 `CODEX_CLAUDE_HANDOFF.md`，清理已失效的活动指令，状态设为 `ready_for_claude`。
2. Claude Code 只执行该计划，在执行记录中如实填写修改、命令、测试、产物、错误和偏差；结束后设为 `ready_for_codex_review`。
3. Codex 独立检查真实 diff、代码、测试、日志和产物。通过时设为 `completed`；需要继续时替换为下一活动周期。

Claude Code 不得改写 Codex 的计划或审查结论，也不得自行宣布最终验收通过。

### 4.2 Codex 与 `luna_worker`

1. 用户明确授权使用 `luna_worker` 后，Codex 将本轮计划写入 `CODEX_LUNA_HANDOFF.md`，状态设为 `ready_for_luna_worker`，再委派执行。
2. `luna_worker` 读取 `AGENTS.md` 和该 handoff，完成实现和验证，将真实结果返回 Codex；不得修改 handoff、扩大范围或回滚其他改动。
3. Codex 独立审查后更新 `CODEX_LUNA_HANDOFF.md`：完成设为 `completed`，需要继续则直接写入下一轮计划，确实无法继续时设为 `blocked`。

`CODEX_LUNA_HANDOFF.md` 只由 Codex 维护，不使用 `ready_for_codex_review` 状态。两份 handoff 不得互相替代，也不得让两个执行者同时运行同一计划。

每份 handoff 只保留一个活动周期；已完成周期不得被新执行者重复运行，必要历史只保留结论和证据路径。

## 5. 当前生产路线与科学不变量

当前生产路线为：

```text
MIPS-Trimer-SCAGE（MTS）
内部标识：mips_trimer_scage
```

当前模型、实验候选和运行参数以 `PIPELINE.md`、活动配置及实际 checkpoint metadata 为准。以下不变量仅在相关代码被修改时检查。

### 5.1 Canonical 图

- 默认拓扑是 `canonical_lifted`；`explicit_k_ru` 是具有独立缓存和 checkpoint 身份的对照，不得与默认表示混用。
- 可学习节点身份只有 `canonical_atom_id`；`relative_ru_shift` 只用于构图、校验和诊断，不进入可学习 embedding。
- 每个 target 保留最大两跳内全部 lifted incoming relation rows。canonical source/target 相同但 shift 不同的 relation 不得去重，其 multiplicity 分别进入 incoming softmax。
- RU 内部键为 `(a,q) ↔ (b,q)`；聚合连接为 `(right,q) ↔ (left,q+1)`。带非零 shift 的 canonical self relation 不得误删为普通 self-loop。
- SPD 表示 O8 的 `0/1/2-hop` 图距离，不是欧氏距离。single-path-node bias 与 Star-RBF 语义相互独立；Star mask 只表示直接聚合连接。
- Star 虚拟关系不得写入 Trimer 真实化学键表，`d_star` 只能来自 Trimer 中两条真实 inter-RU 键。
- readout 按一个 canonical RU 的原子均值定义；显式模式先按 `canonical_atom_id` 聚合 copies，再做图级均值，不得因 repeat factor 改变样本权重。
- 显式模式中同一 canonical atom 的 copies 共享 mask 目标和 Trimer 几何残差；copy、方向和 shift 不形成新的可学习 embedding。

### 5.2 Trimer 与原子身份

- Trimer 是开放的 `RU(-1)-RU(0)-RU(+1)` 局部 3D 代理，不是周期晶胞或无限聚合物平衡构象。
- canonical O8 原子只映射到中央 RU；外侧 RU 只提供空间环境。历史 `mips_copy_id` 不能解释为 `ru_offset`。
- 原子映射不得依赖 canonical SMILES 或 RDKit 输出顺序，必须使用显式身份或经过验证的图同构，并核验原子、键、attachment、backbone 和中央 RU 一一对应；多个 automorphism 使用确定性选择规则。
- Star 虚拟关系不得写入 Trimer 真实化学键表。2D fallback 坐标不得进入任何 3D 距离分支。
- geometry、Star 和 MD200 无效状态分别执行其现有精确回退，不删除样本。

### 5.3 数据与实验范围

- 预训练数据使用完整 `PI1M_v2`；下游任务为 `eat eea egb egc ei eps nc xc`。
- 默认生产模型不隐式加入 PBC、FLAT4、GIN、PaiNN、AP3D512 或历史错误的显式多 RU 实现。
- SMILES 和 CountFP 只能作为明确消融加入，不能隐式改变 graph-only 基线。
- 架构与用于科学比较的超参数跨任务统一；GPU、worker、日志路径等运行参数可按资源调整，但不得改变比较语义。

## 6. 现有 Contract、缓存与 Checkpoint

- 当前 schema、builder version 和缓存 contract 的唯一代码来源是 `src/dataset/mips_trimer_contract.py`，不得在脚本或文档复制第二套常量。
- 已冻结缓存保持只读；同一 LMDB root 只允许一个 writer，训练只读打开已验证缓存。
- 现有生产入口继续拒绝实际不兼容的 schema、artifact 和 checkpoint；不得用宽泛的 `strict=False` 或篡改 metadata 绕过不兼容。
- checkpoint 必须反映实际模型配置、cohort 和缓存 artifact，不能只凭文件名判断兼容性。
- 除非用户明确指定目标并授权，不删除、覆盖或重新标记历史 cache、checkpoint、结果和日志。
- 普通局部修改不自动新增 schema、hash、sidecar、迁移器或 checkpoint 身份。只有确认现有版本、类型、测试或 Git 无法阻止具体混用/损坏时才增加相应机制。

## 7. 实验与评价口径

- 正式结果覆盖 8 个任务和固定 5 folds；部分任务、fold 或 epoch 只能称为 smoke/screening。
- 当前共享 validation/test fold 用于历史同口径比较，不是独立盲测，报告中必须说明。
- 多 seed 先聚合预测，再计算指标；不得用各 seed 的 R² 直接平均代替预测集成。
- 任务结果使用五折 `mean ± sample std`，展示保留 3 位小数，原始 shard 保留完整精度。
- `results/best_result.csv` 记录当前逐任务最优比较目标：

  | Task | Best model | Best R² |
  |---|---|---:|
  | eat | MIPS | 0.990 |
  | eea | Mol-TDL | 0.944 |
  | egb | MIPS | 0.945 |
  | egc | MIPS | 0.926 |
  | ei | Mol-TDL | 0.867 |
  | eps | MIPS | 0.814 |
  | nc | Mol-TDL | 0.882 |
  | xc | Mol-TDL | 0.579 |

- 上述 8 个逐任务最优值的算术宏平均为 `0.868375`，按三位小数报告为 **0.868**。这是跨模型的逐任务最优包络，不是某一个单独实验的 8 任务正式结果，也不代表当前模型已经达到这些数值。
- negative control 即使得分更高，也不能直接称为生产模型或物理 3D 改进。
- 未完成正式复评时，不宣称候选模型性能保持、超过基线或已经晋级。
- 候选失败时保留当前生产默认；非破坏性候选未晋级不等于整个实现周期失败。

## 8. 最小验证策略

验证只覆盖本次改动可能破坏的行为：

- 文档修改：检查目标内容和 `git diff --check`。
- 局部纯函数：运行直接单元测试。
- Dataset、collate 或 forward：相关测试加两样本 forward/backward。
- 原子映射、periodic relation 或几何：按实际改动验证对应身份、multiplicity、shift、不变性或回退，不默认重测全部几何协议。
- 缓存/schema/checkpoint：只在本次触及相应兼容边界时验证读写、拒载和完整性。
- 训练循环、GPU、worker 或调度：运行相关短 smoke；不默认要求精确恢复、逐 bit 一致、完整 parity 或正式复评。
- AMP/数值精度：用代表性小批次检查 finite 和实际所需容差。
- 性能优化先做短 benchmark，再按结果迭代；不以静态推断代替测量。
- 只有跨模块或高风险修改才扩大测试范围；不主动运行无关 lint、全仓测试或昂贵实验。
- 测试失败先判断是否由本次修改引起，不通过放宽生产约束让测试变绿。

Gate 不用于普通代码修改。涉及不可逆覆盖/删除、跨系统迁移、安全敏感操作或正式生产发布时，才设置 1 至 3 个与具体风险对应的阻塞门。

## 9. 结果汇报

交付时简要说明：

1. 修改或检查了什么；
2. 影响哪些文件；
3. 运行了哪些测试、模拟或实验及其结果；
4. 尚未执行的内容；
5. 后台任务的 tmux 窗口和日志（如有）；
6. 结论属于静态判断、测试、smoke、消融还是真实正式实验。

不得用 `py_compile`、单元测试或短 smoke 证明模型性能提升；性能结论只能来自对应口径的实际实验。
