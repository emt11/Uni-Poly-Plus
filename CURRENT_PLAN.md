# MCL-PH-REPAIR-AND-FINETUNE-20260929-01 / r2

状态：**阻断**（2026-09-29 新的直接样本检查发现，现存 `paper5_outer` cohort 的折角色与活动 split 不一致；阶段 A 整体未完成，阶段 B–E 未授权。用户最新指令取消复现旧实验结果的目标，并要求后续不使用 SHA；旧来源恢复路径不再作为后续执行目标）。

授权来源：前次用户仅授权写计划；2026-09-29 用户本条明确授权本计划 r1 阶段 A 的最小代码修复、可信 artifact 恢复及有界 CPU 验证，并明确排除阶段 B–E。计划记录授权，不产生授权。
执行者：Codex 按本条授权实施阶段 A；本次自检不代替后续独立审查。

分工：Codex 规划与独立审查；ZCode 按用户实际授权执行。用户指定 Codex 实现时，以该指令为准。
不安排并行代理写文件或并行 writer。

## 1. 基线与目标

- 基线为 `dev` 当前工作树及已有暂存修改；HEAD 短号 `d93c8a0`。执行前重新记录完整 SHA、`git status`、相关 staged/unstaged diff、入口和运行进程。
- 既有大量暂存修改属于前轮路线收敛，不得 reset、覆盖、顺手清理或提交。下方保留上一计划原始交接，作为历史记录；新审查发现优先于其“工程通过”结论。
- 现有活动 split 为 `paper5_outer`，位于 `data/splits/mips_paper5_outer/`；五个已对齐官方任务为 `eea, egb, ei, eps, nc`。活动八任务不等于论文九任务。
- 目标顺序：修复监督与身份合同 → 建立可信预训练基线 → 筛选微调强度 → 验证 3D／PH 增量 → 对齐完整论文任务并正式复评。
- 最终目标参考 [SOTA_RESULT.md](SOTA_RESULT.md) 的逐任务最佳均值包络，不以不同任务集合的 macro 值判定达标，不承诺一定达到 SOTA。
- 已被用于开发的测试折不称独立盲测；后续开发只使用 train/inner-validation，不反复用 outer-test 选择配置。

## 2. 已核实证据与尚未核实内容

上一轮只读审查核实：

1. `src/dataset/mcl_ph_view.py::mcl_ph_collate` 对 `mcl_angle_pair` 加 `heavy_offset`，但该目标索引的是 `mcl_bond_index` 的键行。两个各含 3 原子／2 键的合成样本，第二个角度应为 `[2,3]`，实际为 `[3,4]`，读取四行键表时越界。该代码早于正式三臂预训练。真实训练错配比例未核实。
2. `persistence_pairs` 未指定 `homology_coeff_field=2`；本机 GUDHI 3.8.0 默认域为 11，而文档及部署 metadata 写 F2。真实数据中 F2/F11 的差异范围未核实，不宣称修正一定提升性能。
3. 冻结 downstream cohort 仍绑定旧 split SHA；新八任务 manifest 的 SHA 均不匹配，旧 `data/splits/mips_outer5_inner20/` 已删除。`load_dual_cohort_task_rows` 仍严格检查该身份。新 manifest 的五个官方任务索引与保留官方 manifest 一致，不代表文件 SHA 或冻结绑定相同。
4. GLT_REF 在全局种子设置前构建新 predictor，且部署包不包含 predictor。其余四臂使用固定 `HEAD_INIT_SEED`。旧 GLT_REF 运行的新 head 初始身份未保存，不能追溯宣称完全可复现。
5. `results/mcl_ph_20260921/shared_pretrain/pretrain_split_v1.json` 指向的 `pretrain_split_v1.npz` 缺失。声明 SHA 为 `31c3599e4da0abb60cb7af27ea0bf7d1ef180b357cf4c714febee86d84735cf2`。
6. 当前聚合器拒绝 r13 的 `periodic_tdl_official5` metadata，且完整范围已改成八任务。需保留历史五任务的只读审查能力，不能修改旧结果去适配新协议。
7. 当前 Xc 同样拟合 StandardScaler，与文档声明的原尺度不符；摘要中的 backbone 学习率固定为 `1e-4`，而 Xc 实际 optimizer 为 `2e-4`。r13 没有 Xc，故此项不改变 r13 五任务分数。

稳定证据入口：

- [r13 聚合](results/mcl_ph_20260921/p2/paper5_official_r13_1/paper5_test.json)：五臂×五任务×五折，125 units、8,750 epochs，历史聚合 PASS。
- [launcher 退出码](logs/mcl_ph_20260921/paper5_official_r13_1_launcher.exit)：0；125 个 unit 退出码均为 0。已重算保存的验证／测试预测，R² 与记录最大差异为 0；选优 epoch、预测索引、历史 split SHA 和臂间样本对应关系一致。
- [预训练产物根](results/mcl_ph_20260921/p2/pretrain/)：`glt_ref_r10r1`、`cat_r10r3`、`gate_r10r3`、`xattn_r10r3` 的 runtime 均记录 5k，部署包 SHA 与 r13 引用一致。
- 本轮审查未重新 strict-load 全部 125 个 best.pt，未重新生成预测，未运行全量模型／DDP 测试。历史分数保留；三条 MCL 轨迹不能称为“已验证正确角度监督”的实现结果。

| Task | GLT_REF R² | O8_ONLY R² | M_XATTN R² | 论文目标均值 | XATTN−目标 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eea | 0.920842 | 0.916881 | 0.924071 | 0.934 | -0.009929 |
| Egb | 0.930224 | 0.918312 | 0.928568 | 0.923 | +0.005568 |
| Ei | 0.796972 | 0.795711 | 0.791783 | 0.839 | -0.047217 |
| EPS | 0.734678 | 0.730104 | 0.733418 | 0.787 | -0.053582 |
| Nc | 0.800615 | 0.817827 | 0.805572 | 0.856 | -0.050428 |

XATTN macro5=0.836682，GLT_REF=0.836666，配对正差 12/25，尚无稳定整体优势。EPS／Nc 在 XATTN 的最佳验证 R² 到第 70 轮平均下降约 0.1546／0.1174；这是优化强度的调查线索，不是归因结论。

## 3. 阶段 A：最小正确性修复与可复现性恢复（审查后阻断）

### A1. 允许修改范围

仅允许为下表风险修改所列实现、相应局部测试及本文件执行记录。可新增一个职责单一的历史只读验收入口；不得顺手重构。若需要表外模型／数据语义变更，记录阻断和具体必要性。

| 工作项 | 主要文件／对象 | 必须满足的验收条件 |
| --- | --- | --- |
| 角度索引 | `src/dataset/mcl_ph_view.py`、相关 view/pretrain tests | 独立维护累计键数；原子端点用原子偏移、角度键行用键偏移。至少两图且原子数≠键数，检查角度涉及两键均属于目标图；覆盖空角度／空键。不只检查“不越界”。检查同一 collate 的负 sentinel 是否被偏移成合法跨图索引，发现则同范围最小修复。 |
| PH 定义与缓存身份 | `mcl_ph_view.py`、`mcl_ph_trajectory_cache.py`、`src/modules/mcl_ph.py`、必要的缓存／预训练入口 | 显式 F2；测试直接验证传参及独立 F2 参考。缓存／包绑定系数域、半径、描述符定义及实现版本；拒绝语义不兼容的旧 artifact。为旧包保留显式历史审查能力，不将它们标成新合同兼容。 |
| 冻结 cohort 来源与评价 split | `src/dataset/glt_dual_cache.py`、`src/training/glt_dual_runtime.py`、`src/training/split_protocol.py`、必要的微调／launcher 入口 | 优先从 Git 对象或现存可信备份恢复与冻结 SHA 完全一致的旧 manifest 到独立 provenance 目录。旧 manifest 仅证明冻结来源，活动评价只用 `paper5_outer`；保留样本顺序／行数／选中行身份检查。不得关闭 SHA 检查、重写旧 cohort manifest 或把新 split SHA 填成旧值。找不到可信来源则该项阻断。 |
| 初始化与实际超参数 | `scripts/finetune_mcl_ph.py`、必要的 `src/utils.py` 局部接口 | 构建模型前固定初始化 RNG；四个 512-wide head 仍保持公共初态；GLT_REF 的独立 head 也可复现。记录初始化 hash、seed、真实 optimizer 学习率及 scaler 方式。Xc 使用明确的 identity target transform，其他任务保持 train-only 标准化。 |
| 缺失预训练索引 | `shared_pretrain/` 的声明与可信备份 | 只恢复 SHA 完全相同的原 `.npz`，验证 train/validation 身份合同及关联文件。不重新抽样、不根据长度猜索引；没有可核验源则标“阻断”。 |
| 历史结果可验收性 | `scripts/aggregate_mcl_ph*.py` 或最小历史只读入口及测试 | 明确区分 r13 五任务历史协议和当前八任务协议；由显式合同决定期望集合，不靠目录内容推断完整性。核验保存预测与官方索引、样本身份、选优和包 SHA，不回写旧结果。 |

恢复 provenance／索引仅在用户授权 A 后允许；新文件放到新目录，不覆盖原 artifact。若恢复源字节无法满足声明 SHA，不得扩大为数据／缓存重建。

### A2. 验证预算与边界

- CPU 定向验证总墙钟上限 30 分钟；不运行全套 pytest。
- 合成 fixture、纯数学／schema／身份／聚合测试允许；真实模型 forward 最多 16 次、backward 最多 8 次，optimizer updates=0。不同测试入口合并计数；不为凑齐额度重复测试。
- 真实数据只读检查最多 8 个已有 P_train 样本及 1 个 `eea/fold0` train batch（最多 24 样本），仅用于映射、collate 和来源绑定。P_train 身份未恢复时不得猜样本或绕过守卫。
- 历史 125 units 的已有 JSON／NPZ／退出码及 manifest 可只读验收，不打开原始 outer-test 数据、不运行模型测试推理、不生成新 outer-test 预测。预训练部署包可 CPU strict-load 四个包；不重载全部 125 个微调模型作为默认门槛。
- GPU=0、分布式 worker=0、训练启动=0、正式 epochs=0；不运行 fake smoke 冒充真实训练。
- 超过一分钟的 CPU 验证放在 `tmux` session `Uni-Poly` 的独立 window `mcl_ph_repair_a`，保存精确命令、cwd、日志、退出码及实际计数；session 不可用则报告阻断，不改用无日志后台任务。
- 建议新日志根 `logs/mcl_ph_repair_20260929/a_01/`、新审查／provenance 根 `results/mcl_ph_repair_20260929/a_01/`。启动前确认不存在；若冲突，不覆盖，记录新的唯一编号。
- 禁止全量 PH cache 重建、全 cohort 标签扫描、新构象、标签改写、下载新数据、改 split、删除旧文件，以及 pull/commit/push。阶段 A 不调整半径、专家数、损失权重或下游搜索范围。

### A3. 停止与交付

- 发现 artifact／schema／身份不匹配、NaN/Inf、跨图目标、writer 冲突或预算将耗尽，停止依赖该问题的验证；保留失败输出。允许在本授权内定位及最小修复，修复后只重跑相关检查，消耗仍计总预算；同一未定位失败不得自动循环重试。
- 缺失索引或来源证据不妨碍完成独立的代码修复及合成验证，但 A 整体不能标完成，更不能启动 B。
- 完成后标“待审查”；报告每项 diff、实际测试、模型调用账目、日志、恢复文件 SHA、未执行项和阻断。ZCode 自检不等于 Codex 独立通过。
- Codex 审查实际 diff 与产物后，决定“已完成／需返修／阻断”。历史性能不重写为修复后的成绩；本文件记录审查结论，必要的模型实现历史在审查通过后再归档。

## 4. 后续路线：B–E 均待另行授权，不包含在阶段 A 提示词中

| 阶段 | scientific question | reference | controlled change | comparison budget 与门槛 |
| --- | --- | --- | --- | --- |
| B：可信预训练基线 | 正确目标与身份合同下能否得到可用 MCL 表示？ | 保存旧 r10R3 为描述性参照；新基线从原始共同初始化开始 | 优先一个 XATTN，修复组合明确为正确性变更，不作单因素归因 | 拟先有界 smoke，再另行确认一次 5k、world4/micro84/accum3/global1008；必须先解决 F2 cache／统计一致性及缺失索引。smoke、缓存与统计准备、DDP、正式训练的精确预算须在 B 修订中单列；本表不授权任何启动。 |
| C：微调强度 | 小样本任务是否受过强联合更新影响？ | 同一新可信 XATTN 包、固定新 head 和样本顺序 | 首轮只把 joint backbone lr 从 1e-4 改为 1e-5；head、调度、epoch、batch、损失、scaler 不变 | 拟 Ei/EPS/Nc×fold0/1×2 设置=12 starts、最多840 epochs，seed42+fold，仅 inner-validation；只作开发 smoke，不称正式性能提升。 |
| D：机制归因 | 3D、样本路由和 PH 各有何增量？ | 同一新 MCL 包抽取的 O8-only 与完整模型；保留原 GLT_REF/O8_ONLY 的不同来源标签 | 先控制微调阶段是否使用3D；若建立增量，再对 PH 两列固定均值、全描述符固定均值、同4Å专家作分别匹配的预训练消融 | 不把推理时关分支称重新训练消融；不把同包 O8 抽取称独立单模态预训练。精确预算待 C 审查后制定；无增量则停止复杂化。 |
| E：正式复评与九任务对齐 | 冻结候选能否达到论文目标并稳定泛化？ | 可信新 reference 与固定论文表，逐任务比较 | 候选／初始化／评价方式先锁定，再执行五折确认；任务扩展单独处理数据合同 | 初步五任务单候选25 units／1750 epochs仅为估算，不授权启动；reference、新seed和额外任务须另列预算。先解决 Egc/Xc cohort 差异并补 Eib/Tg，EAT 不代替论文任务。 |

阶段 C 的前置实现需另行授权：增加明确的 validation-only 阶段及对应聚合，训练器和 launcher 均不能自动读取 outer-test；活动 split 仍为 `paper5_outer`，不恢复旧实验协议。当前入口只支持训练后测试，不能直接拿来做超参数筛选。

拟定 C 晋级规则（在执行授权时冻结，不事后改门槛）：对每个 task/fold 计算
`(RMSE_reference - RMSE_candidate) / RMSE_reference`，六单元均值至少 1%，至少四单元为正，任一任务两折平均相对 RMSE 不得恶化超过 2%。该门槛只是开发筛选规则，不是显著性或正式性能结论。未满足则报告全部观测值并停止该候选，不扩大预算。

后续按单因素顺序考虑：胜出学习率 → joint 单次余弦衰减对比周期重启 → 局部冻结／LoRA → head 正则；每轮分别授权。若仍无法缩小 Ei/EPS/Nc 缺口，再为化学描述符、预训练目标或周期高阶结构另立科学问题。stacking／残差校正只能用内部交叉拟合及验证数据，不根据 outer-test 调权重。

达到某个均值不自动称 SOTA：需相同任务／样本／五折、相同指标口径、披露全部候选与种子，以及配对不确定性。额外 seed 不能把已开发测试折变成独立盲测。若持续无收益，允许负结果关闭路线，不无限追加训练。

## 5. 本轮执行记录与交接

- 前次计划编写（2026-09-29）：仅更新本文、检查改动及进程；当时阶段 A 消耗为 0、状态为待授权。以下为本次独立的阶段 A 执行记录。
- 开工基线：`HEAD=d93c8a0d782b112fd5e1ca587d2a066459d5b70b`；已检查 `git status`、staged/unstaged diff、相关入口和进程。既有约 294 文件的暂存变更及 `AGENTS.md`／本文未暂存差异均保留，未 reset、覆盖、清理、提交；未见 MCL 训练进程。`tmux` 起初无 server，已建立 `Uni-Poly` session，并在独立 `mcl_ph_repair_a` window 运行有日志的验证。
- 角度与 sentinel：`src/dataset/mcl_ph_view.py` 改用独立累计键行偏移处理 `mcl_angle_pair`；原子端点仍用原子偏移，`-1` 映射原样保留；非法负数和越图局部索引显式拒绝。`tests/test_mcl_ph_view.py` 两图 3 原子／2 键、跨图键归属、空键／空角度和负索引合成检查通过。
- PH 身份：`mcl_ph_view.py` 显式向 GUDHI 传 `homology_coeff_field=2`；`mcl_ph_trajectory_cache.py` 将域、半径、描述符列及实现版本纳入 cache identity 并拒绝旧 manifest；`src/modules/mcl_ph.py` 将相同语义写入部署包并在新合同加载时拒绝缺失／不匹配。旧 r13 包只用于显式历史只读验收，不声明为新 F2 合同兼容。
- 初始化及 Xc：`scripts/finetune_mcl_ph.py` 在构建模型前设置 seed，记录 head 初始 SHA、seed、实际 optimizer group 学习率及 target transform；四个 512-wide head 保持公共初态。`src/utils.py` 增加 identity target scaler，Xc 保持原尺度；其他任务仍使用 train-only standard。相关合成测试通过；未启动微调来核验真实运行输出。
- 冻结来源：`scripts/run_mcl_ph_8x5_4gpu.py` 在创建输出前核对 cohort/union provenance 及旧来源 manifest 的声明 SHA。活动 `paper5_outer` 仍只作为评价 split；`scripts/finetune_mcl_ph.py` 原有 selected-row 和 cohort SHA 守卫保留。只读预检实际以 `frozen cohort/union provenance mismatch` 拒绝当前产物。冻结 cohort 中 `eea` 旧 split SHA 为 `696d1accead74c6db7d0bbf427a33915fd5d32dddefb9ac5c8d715e2be6dfdf4`，现存 union 为 `8d1efa0cb2be3a646f7133ab4c299b103dbdb56c4368af48b8db10d5698ed6b2`；旧 manifest 源及与 cohort hash 一致的 union 来源未找到。未关闭 SHA 检查、未回填新 SHA。
- 原始预训练索引：声明 `pretrain_split_v1.npz` SHA 为 `31c3599e4da0abb60cb7af27ea0bf7d1ef180b357cf4c714febee86d84735cf2`。在 `/root`、`/mnt`、`/media`、`/tmp`、`/data`、`/opt` 等现存路径未找到同名文件；257 个 reflog commit 中未找到旧 split 路径；`git fsck` 的 87 个不可达 blob 无一匹配八个旧 manifest SHA 或该 NPZ SHA。恢复文件：**无**；恢复 SHA：**无**。未重新抽样或生成替代索引，身份合同验证因此未执行。
- r13 历史只读入口：新增 `scripts/audit_mcl_ph_r13_readonly.py`，并在 `scripts/aggregate_mcl_ph.py` 增加显式历史 manifest 参数；固定五任务×五臂×五折，检查旧官方 manifest SHA／索引顺序、冻结 cohort seal／key 文件 SHA／样本身份、验证／测试保存预测 R²、选优 epoch、unit／launcher 退出码和四个实际预训练包 SHA。最终历史验收 125/125 PASS，报告在 `logs/mcl_ph_repair_20260929/a_01/historical_audit_sealed.log`；仅打开旧 JSON／NPZ／退出码及包字节作 SHA，未打开原始 outer-test 数据或重新推理。首次审查错误地要求历史结果匹配已替换的活动 union split，得 5 项误报；定位后只改历史审查合同并重跑一次，首次输出保留在 `verify.log`。补充 cohort seal／key SHA 守卫后再次通过，前次通过输出保留在 `historical_audit.log`。
- 验证日志：`logs/mcl_ph_repair_20260929/a_01/verify.log` 为首轮 14 passed／1 warning 及首次历史审查输出；`historical_audit_sealed.log` 为最终历史 125/125 PASS；`targeted_followup.log` 为 11 passed／1 warning；`final_local.log` 为 3 passed／1 warning 及预期失败的活动身份预检；`cache_entry.log` 为缓存来源／训练入口 2 passed／1 warning；`search_git_objects.log` 为对象搜索。命令、cwd、环境和退出码记录在 `commands.txt`、`verify.sh`、`search_git_objects.sh`、对应日志／`.exit` 文件中。`python -m compileall -q` 对修改的 Python 文件通过，`git diff --check` 通过。未运行全套 pytest。
- 阶段 A 预算：定向 pytest 共 30 passed；CPU 验证远低于 30 分钟上限；真实模型 forward=0、backward=0、optimizer updates=0；GPU=0、worker=0、训练=0、正式 epoch=0；真实 P_train 样本=0、`eea/fold0` train batch=0；只读核验历史保存预测 125 units。未新建缓存、未恢复 artifact、未写旧预测或旧包、未读取原始 outer-test、未删除旧文件。新日志根为 `logs/mcl_ph_repair_20260929/a_01/`，新审查根 `results/mcl_ph_repair_20260929/a_01/`（空，无虚构产物）。
- **阻断与未执行**：可信旧 manifest／union 和原始预训练 NPZ 缺失；活动真实数据来源绑定、P_train 身份、单 batch collate、四包新合同 strict-load 和任何训练均未执行。旧 r13 MCL 包在新 PH 合同下预期拒绝，历史分数不作为修复后性能。阶段 A 自检完成，状态待 Codex 的后续独立 diff／日志／产物审查；不得自行认定已完成或进入 B–E，不执行 pull、commit、push。

### 5.1 Codex 独立审查与最小返修（2026-09-29）

- 授权与基线：用户要求给出并执行本计划下一步，随后明确允许写入本文；此次执行为阶段 A 的独立审查和授权内最小返修。复查了当前 `git status`、staged／unstaged diff、入口与进程；保留所有既有修改和产物，无训练进程。
- 介于阶段 A 自检与本次审查之间，用户单独授权清理可去除的 SHA 逻辑：删除 `benchmark_glt_dual_read.py`、`validate_glt_dual_deploy.py`、`audit_mcl_ph_optimizer_groups.py`、`verify_glt_dual_static_parity.py`、`create_mips_cache_preflight.py`、`materialize_cache_derived.py`、`src/dataset/lmdb_cache.py` 中无人读取的报告摘要，并合并 split、配置、历史审查的重复哈希计算；冻结来源、索引、缓存及模型包的拒绝式身份校验保留。该独立清理的定向测试为 21 passed、历史只读审计 125/125 PASS，命令、cwd、日志及三个 0 退出码在 `logs/mcl_ph_repair_20260929/sha_cleanup_01/`。未改旧报告或产物。
- 审查实际代码、定向测试及退出码：角度键偏移和负 sentinel、防跨图索引、显式 F2、PH cache／包语义合同、初始化 seed／初态证据、Xc identity target transform、历史五任务只读验收与活动八任务边界均有相应代码及合成验证。核对旧 `a_01` 日志为 30 个定向 pytest 通过；SHA 清理后只读历史审计仍为 125/125 PASS、`problems=[]`，见 `logs/mcl_ph_repair_20260929/sha_cleanup_01/historical_audit.log`。这些证据不替代缺失 artifact 上的真实绑定和模型运行验证。
- 发现并返修：先前 reader 在 `src/dataset/glt_dual_cache.py` 固定从 `cohort_root.parent/union_paper5_outer/manifest.json` 读取 union。即使将字节匹配的旧 manifest 恢复到独立 provenance 目录，`cohort_root.resolve()` 仍会使其读取现存不匹配 union，导致阶段 A 的合法恢复路径不可用。现于 selected-row reader、`src/training/glt_dual_runtime.py::open_source`、MCL 单元及 launcher 之间传递显式 `--cohort-union-manifest`；launcher 在创建输出前同时核对 cohort `.frozen` seal、union manifest hash 和八任务旧来源 split SHA。单元训练及测试读取同一显式 union，并在记录中保存路径。未更改冻结 cohort、现存 union、活动评价 split 或旧结果；未绕过任何身份检查。
- 新增的合成测试覆盖外置可信 union 可被绑定、错误同级 union、被篡改的外置 union、来源 split 及 cohort seal 均被拒绝。`logs/mcl_ph_repair_20260929/a_review_01/` 中的 `verify.sh`、`commands.txt`、`cwd.txt`、`compile.log`、`pytest.log`、两个 `--help` 日志及 `exit_codes.txt` 保存精确命令和结果：**8 passed, 1 warning**；compileall、pytest、两个入口的 help 退出码均为 0。补充 seal／split 拒绝断言后仅重跑受影响的单测，**1 passed, 1 warning**，命令、日志及 0 退出码在同目录的 `followup_command.txt`、`followup.log`、`followup.exit`。`git diff --check` 通过。两次均在独立 tmux window `mcl_ph_repair_a` 运行并退出，首轮墙钟约 22 秒，补充单测约 5 秒。此次审查真实模型 forward=0、backward=0、optimizer updates=0，GPU／worker／训练／正式 epoch=0；未读取原始 outer-test 或真实 P_train 样本，未运行全套 pytest。
- 现存证据与结论：冻结 cohort 声明 union hash 为 `5ea86a69923cb1eeb1a450c9ff2778e97c9d3973155dd385bb338159c12162b3`，现存 union 实算为 `474be5ae38b18922f47aca2379e504362c88e97bb71d06b02b1c984e18035147`；旧 split 根不存在，声明 SHA 为 `31c3599e4da0abb60cb7af27ea0bf7d1ef180b357cf4c714febee86d84735cf2` 的 `pretrain_split_v1.npz` 仍不存在。恢复文件及恢复 SHA：**无**。因此阶段 A 审查结论为**阻断**：代码及合成验证可验收，可信来源绑定、原始预训练索引合同、真实 batch 和四包新合同 strict-load 未执行。下一步只从可信备份取得字节匹配的旧 union／八个 split manifest 和原 `.npz`，存入独立 provenance 目录并复核声明 SHA；来源未出现前不重建缓存、不重新抽样、不启动 B–E。未执行 pull、commit、push。

### 5.2 最新用户指令下的只读检查（2026-09-29）

- 用户明确表示不需要复现以前的实验结果，并要求后续不使用 SHA。该指令优先于上文旧产物恢复步骤；历史 r13 仅保留为已归档观察值，不再追找原始旧索引或旧 manifest 以复跑。
- 已核对 HEAD、工作树、相关入口和运行进程；既有大量暂存及未暂存改动均保留。`Uni-Poly` session 可用，未见训练进程。本次没有修改代码、配置、数据、checkpoint 或实验产物。
- 对现存 `cohort_1545eda5a8f6a1_paper5_outer`、`union_paper5_outer` 和活动八任务 split 直接比较任务行数及五折各角色数量：八任务均一致。这只是数量检查，不证明逐行折身份一致。
- 在阶段 A 的真实数据上限内，只读检查 `eea/fold0` 的 24 条已有训练行：任务、原始行号、SMILES、标签均与原始任务表一致；五折 `train/validation/test` 角色与活动 split 不一致者为 **24/24**。例如任务行 10 在 cohort 中的五折角色依次为 `train, train, validation, test, train`，活动 split 为 `train, test, train, validation, validation`。本次只检查这 24 条记录，未扫描全任务标签，也未读取 outer-test 原始数据。
- 现有 launcher 的来源预检拒绝把活动 split 作为该 cohort 的来源；结合直接角色不一致，不能将这个 cohort 直接用于新实验。发现身份不匹配后，已停止依赖它的验证。未运行模型 forward/backward、优化器、GPU、worker、训练或测试预测；未新建日志和结果目录。
- 后续若要建立新基线，需要先明确授权独立的新来源和预训练索引合同：在新目录准备与活动 split 逐行一致的来源，直接比对任务行、标签、样本顺序与每折索引；预训练集合须明确排除下游身份并固定验证集，记录规则、计数及列表。新入口应以这些直接检查放行，不依赖旧文件摘要。全量来源读取、生成索引、缓存准备及任何训练均超出当前阶段 A 的允许范围；具体预算、产物目录和停止条件须在执行前修订授权。阶段 B–E 保持未启动。

---

## 上一计划原始交接（历史保留，后续结论已被本轮审查补充）

下文原样保留上一轮实施与验证记录；测试数字为上一轮记录，本轮未重跑。
其中“仅剩 npz 缺失”“工程通过”“暂无后续执行”等判断不再代表当前完整风险或后续路线。

# ROUTE-CONSOLIDATION-20260929-01 / r2

状态：已完成（依据：代码、配置、manifest、依赖闭包和定向测试已独立核对；未启动训练或正式实验）。

授权来源：用户当前计划。执行者：Codex；本轮未委托其他执行者。基线包含用户已修改的
`AGENTS.md`、`PROJECT_HISTORY.md`、`TODO.md`、删除 `Plan.md`／`RESULTS.md` 的工作树。

## 最终目标与合同

- 活动路线仅为 GLT dual（参考）和 MCL-PH（正式）；MCL-PH 保留五个 arm：
  `glt_ref`、`o8_only`、`m_cat`、`m_gate`、`m_xattn`。
- 唯一 split 为 `paper5_outer`，根目录 `data/splits/mips_paper5_outer/`，任务为
  `eat, eea, egb, egc, ei, eps, nc, xc`。
- 外层 `KFold(n_splits=5, shuffle=True, random_state=100)`；对排序后的
  outer-train 使用 `train_test_split(test_size=0.2, shuffle=True, random_state=42)`。
- MCL-PH 维持 Periodic-TDL head-only 10 epoch + joint 阶段、validation RMSE 选优和
  恢复后 outer-test 读取；EGC 为 60 总 epoch，其余任务为 70。EAT 标记为项目适配任务。
- 本轮没有训练、GPU、worker、缓存重建、outer-test 或正式实验授权和执行。

## 实际实施

### Split 与活动入口

- `scripts/create_mips_split_manifests.py` 只接受 `paper5_outer`，生成八个 manifest；每折
  记录五折索引、样本顺序 SHA-256、`sets_disjoint` 和 `union_equals_full_cohort`。
- 新增 `src/training/split_protocol.py`，统一检查协议、整数／范围／重复、互斥和完整覆盖。
- GLT dual 微调、grid launcher、聚合器，以及 MCL-PH 微调、四 GPU launcher 和 Macro8
  聚合器均固定到 `paper5_outer`。
- `src/utils.py` 收敛为两条活动路线共用的训练与指标工具，移除旧 MTS trainer 分支。

### 删除清单

已删除只属于关闭路线的代码、配置和测试：

- MTS/Distill：`scripts/build_mts*`、`scripts/run_mts*`、`scripts/pretrain_mts*`、
  `scripts/report_mts*`、`scripts/run_glt_v2*`、`scripts/report_glt_v2*`、
  `src/training/{finetune,pretrain,c0_transfer}`、对应 MTS dataset/module、配置和测试；
- GLT-Galformer、GLT-PRED：`src/{dataset,modules}` 中对应实现、配置、旧 PRED/Galph
  审计入口和结果；
- Atomic-PC/EQ3D/DND、旧 PH fusion：对应配置、模块、入口和测试；
- 旧 MCL-PH：`run_mcl_ph_r10r3.py`、旧 P2 聚合器／测试、P0 退役审计测试；
- 旧 split 目录 `data/splits/mips_outer5_inner20/`、`data/splits/mips_shared5/`；
- 对应的 ignored `logs/`、`results/` 和 `data/processed/` 目录（MTS/Distill、Galformer、
  PRED、旧 GLT v2 和旧缓存产物）。

### 保留清单与依赖审查

- 保留 GLT dual/MCL-PH 活动脚本、`src/dataset/{glt_dual,glt_dual_cache,mcl_ph_view,mcl_ph_trajectory_cache}`、
  `src/modules/{glt_dual,glt_dual_pretrain,mcl_ph,mcl_ph_pretrain,mips_local_graph}`，共享
  `cache_lifecycle/cache_spec/frozen_store/lmdb_cache`、`data/processed/glt_dual_v2`、
  `data/processed/mcl_ph_cache` 和必要 checkpoint。
- 保留 `configs/mts/periodic_tdl_official5/` 及论文来源／历史审计文档，仅供索引一致性和归档，
  不被活动运行入口作为 split 输入。
- MCL-PH 所需的旧初始化状态复制到 `results/mcl_ph_20260921/shared_pretrain/`，活动配置和
  测试已改用该路径；原 `results/glt_pred_20260918/` 暂未删除，因为工作树缺少其声明的
  `pretrain_split_v1.npz`，删除会丢失当前 artifact 依赖证据。该 artifact 不完整项仍未解决。

## 验证结果

- `python scripts/create_mips_split_manifests.py --raw-root data/raw --output data/splits/mips_paper5_outer --protocol paper5_outer`：八个 manifest 生成成功。
- 五个官方任务 Eea/Egb/Ei/EPS/Nc 的五折索引与 `configs/mts/periodic_tdl_official5/` 完全一致；八任务每折互斥且完整覆盖检查通过。
- `python -m compileall -q src scripts tests`：通过；活动入口导入检查通过。
- 定向测试通过：GLT dual／MCL-PH 合同与聚合测试 `93 passed, 1 warning`；MCL-PH view/pretrain `37 passed, 1 skipped, 2 warnings`；缓存与读写检查 `82 passed, 3 skipped, 1 warning`；MCL-PH PH 独立定义测试 `63 passed, 1 warning`。
- `git diff --check`：通过；未运行模型训练或性能实验。

## 审查结论与后续

结论：路线收敛、唯一 split 和关闭入口实现通过静态依赖审查及局部 CPU 测试；没有性能结论。剩余风险是 `shared_pretrain/pretrain_split_v1.npz` 缺失，不能把预训练恢复合同描述为完整可用。

暂无后续执行。若要恢复预训练或删除原始 `glt_pred` artifact，需新计划先补齐并核验样本索引 artifact，再取得用户授权。
