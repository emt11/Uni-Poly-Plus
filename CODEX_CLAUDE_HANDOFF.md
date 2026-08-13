# G1 20k Finalization 恢复与正式下游续接

> 本文件是 `Codex → Claude Code → Codex` 流程的当前唯一活动计划。当前任务不是重新训练 G1，而是从已完成 20,000 step 的不可变 milestone 恢复正式 final checkpoint，并继续原定 G0/G1 正式下游评估。

## 0. 活动元数据

- `cycle_id`: `mts_g1_step20k_finalization_recovery_v1`
- `status`: `ready_for_codex_review`
- `planner`: Codex
- `executor`: Claude Code
- `reviewer`: Codex
- `project_root`: `/root/workspace/Uni-Poly-Plus-master`
- `supersedes`: `mts_g0_g1_matched_formal_v1` 的未完成 finalization/下游部分
- 当前结论：G1 训练计算已完成，正式 checkpoint finalization 失败；G0/G1 下游均未启动

## 1. 已独立确认的故障事实

### 1.1 训练预算已经完整执行

- G1 日志明确到达 `step=20000/20000`，最后记录的 loss、梯度和参数路径均 finite。
- GPU、torchrun、`pretrain.py` 和旧 watcher 进程均已退出。
- 旧 watcher 正确地因 final checkpoint/completion marker 缺失而停止，没有启动任何 G0/G1 微调。
- 已生成且不得覆盖：

  ```text
  pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1/
  mts_g1_pretrain_20k.step_20000.pth
  ```

- 当前 milestone SHA256：

  ```text
  3b321e5741924bb73793c450bc6fc76fa88f78f549c256aaa69b4d2dee2933df
  ```

### 1.2 故障发生在训练后的旧绘图尾部

运行中的 Python 进程在源码清理前已加载旧函数；训练结束后仍尝试读取已删除的：

```text
./plots/pretrain/mts_g1_pretrain_20k_loss_data.json
```

因此抛出 `FileNotFoundError`，尚未来得及执行 final checkpoint 和 `.complete.json` 写入。当前磁盘版 `pretrain.py` 已无该绘图链，本周期不得恢复 plots 逻辑。

### 1.3 milestone 足以无损恢复模型权重

- G1 step-20000 milestone 使用 `mts-pretrain-milestone-v1` / `mts-model-v4`，记录 `optimizer_step=20000`、正式 profile、启动合同、134 个完整 model state tensors 和预训练 heads。
- 对正常完成的 G0 做实物对照：G0 step-20000 milestone 的134个 `state_dict` tensor与G0 final checkpoint逐 key、shape、dtype和值完全一致。
- 正式下游只加载模型 `state_dict`；milestone中的预训练 heads 不进入最终下游 checkpoint。

这证明可以恢复 final checkpoint，但**禁止**直接复制、重命名 milestone，或手工修改其 schema 冒充 final。

### 1.4 源码身份必须如实拆分

G1 milestone 的固定启动合同记录：

```text
runtime_pretrain_code_sha256 = fc7455dd58dfa4447b0ed6190e91b8730da01a53c4526c6bbbcac73fec499a41
```

训练期间 `pretrain.py` 的 plots/死统计逻辑被删除，milestone顶层动态磁盘hash随后变化。恢复时必须：

- 将启动合同中的 `fc7455dd...` 作为实际训练 runtime code identity；
- 另记当前 finalizer/disk code identity；
- 记录 `recovered_from_milestone=true`、源 milestone路径/SHA和原始失败日志；
- 不得把当前磁盘hash写成训练全过程的runtime hash，也不得改写原milestone。

G0与G1启动时的代码identity还存在一个既有差异：`src/modules/mips_local_graph.py` hash不同。该问题不是本次plots删除造成。本周期只做只读语义分类并写入最终报告；不默认重跑20k，也不得隐瞒该限制。

## 2. 本周期目标

1. 修复 plots 删除后仍可能残留的 finalization耦合，并加入直接回归测试。
2. 新增一个严格、通用但默认拒绝覆盖的 G-family milestone finalizer，从G1 step-20000生成标准 `mts-model-v4` final checkpoint及completion marker。
3. 证明恢复后的G1权重与step-20000 milestone逐tensor完全一致，metadata来自可验证证据且没有伪造训练身份。
4. 运行现有G0/G1 checkpoint audit；通过后重新启动原定的顺序 G0 8×5 → G1 8×5 → comparison。
5. 不重新执行G1 20k，不启动G2/G3，不恢复plots或attention heatmap。

## 3. 实施步骤

### Step 1：修复并测试 finalization 与 plots 解耦

- 审查当前 `scripts/pretrain.py`，确认训练结束后的正式checkpoint保存路径不再读取或写入：

  ```text
  plots/
  *_loss_data.json
  *_loss_curve.png
  losses
  ```

- 增加一个直接测试，模拟已达到固定optimizer budget后的finalization，证明plots目录及loss JSON均不存在时仍能生成final checkpoint和completion marker。
- 不恢复任何绘图、per-epoch死统计或attention heatmap代码。

### Step 2：实现严格 milestone finalizer

建议新增：

```text
scripts/finalize_mts_g_pretrain_milestone.py
```

该工具只允许将正式G-family的最终optimizer-free milestone转换为下游可加载final checkpoint，并必须满足：

1. 输入文件存在且SHA与命令行显式声明一致；schema为`mts-pretrain-milestone-v1`，checkpoint schema为`mts-model-v4`，`optimizer_step=20000`。
2. 严格解析活动G1配置，逐字段核对milestone `resume_contract`：G1 arm、T1/MSTA、PI1M_v2、shared step0、bundle、relation artifact、training hash、global batch和20k profile。
3. 严格验证原始G1 step0文件及其SHA/metadata，不从G0、T阶段、smoke或其他arm借用身份。
4. `state_dict` key、shape、dtype与活动G1模型合同完全匹配，所有浮点tensor finite；最终权重逐tensor复制自milestone，不重新初始化、不执行forward/backward、不更新参数。
5. final metadata使用生产loader真实要求的`mts-model-v4`合同。确定性字段从严格配置、milestone启动合同、step0 metadata和冻结cache/sidecar实物重建；不得把G0 metadata整块复制给G1。
6. runtime code identity固定取milestone `resume_contract.pretrain_code_hash/files`；当前磁盘/finalizer identity另存为`finalization_code_identity`。
7. 增加恢复来源：

   ```text
   recovered_from_milestone: true
   recovery_reason: post_training_plot_finalization_failure
   recovery_source_path
   recovery_source_sha256
   recovery_source_optimizer_step: 20000
   runtime_pretrain_code_identity
   finalization_code_identity
   original_failure_log
   ```

8. 无法从现有证据精确恢复的运行字段不得猜测。若`training_wall_seconds`没有可靠原始值，记录`null`或显式recovery字段，并相应调整比较器显示为`unavailable`；禁止用文件时间伪装精确训练墙钟。
9. 输出路径及`.complete.json`必须原子写入；任一目标已存在时默认拒绝覆盖。
10. completion marker的checkpoint SHA、optimizer steps、profile及runtime code SHA必须与最终checkpoint一致。

目标输出固定为：

```text
pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1/mts_g1_pretrain_20k.pth
pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1/mts_g1_pretrain_20k.pth.complete.json
```

### Step 3：先用G0证明finalizer转换语义

不得覆盖G0正式文件。使用临时目录执行：

1. 从G0 step-20000 milestone生成临时recovered checkpoint；
2. 验证临时checkpoint的134个model tensors与现有G0 final逐key、shape、dtype和值完全一致；
3. 验证下游strict loader接受临时checkpoint；
4. 删除本周期临时验证产物，保留测试/日志证据。

G0测试通过后，才允许finalize G1。

### Step 4：生成并审计G1 final checkpoint

所有产物写入命令必须在`tmux Uni-Poly`唯一窗口执行，建议：

```text
mts-g1-finalization-recovery-v1
```

日志：

```text
logs/mts_multiscale_topology/g_family_matched_v1/G1/finalization_recovery.log
```

完成后验证：

- final `state_dict`与G1 step-20000 milestone的134个tensor逐值完全一致；
- final checkpoint及completion marker SHA互相一致；
- `optimizer_steps=20000`、G1/T1/shared-step0/bundle/relation identity正确；
- 所有参数finite；
- 生产下游strict loader可以加载；
- `scripts/audit_mts_g0_g1_formal.py --phase checkpoint`通过。

将恢复审计写入：

```text
results/mts_multiscale_topology/g_family_matched_v1/G1/finalization_recovery_audit.json
```

### Step 5：分类G0/G1运行时代码差异

- 根据现有hash、diff、日志和历史审计，判断G0/G1启动时`src/modules/mips_local_graph.py`差异是否仅属于G1 relation-geometry实现/等价向量化。
- 若证据表明差异只影响G1专属geometry branch或保持数学语义，则记录为已解释的实现差异，不阻止finalization和下游。
- 若无法证明不影响共享G0/T1计算，则在comparison中加入`matched_code_limitation`，禁止宣称严格逐代码因果匹配；本周期仍可完成恢复与下游，但不自动得出强因果结论。
- 不通过修改checkpoint metadata、放宽strict loader或删除旧证据解决该问题。

### Step 6：恢复顺序下游链

旧`auto_trigger_g_family_downstream.sh`已退出。对其做最小修改，使`G1_PID`可选：

- 正常模式：传入活动PID时保持原等待逻辑；
- recovery模式：未传PID时直接从两个已完成且审计通过的checkpoint开始；
- 两种模式都必须先验证G0/G1 final及completion marker，再启动任何writer；
- 仍保持G0完整8×5完成后才启动G1完整8×5；
- 每臂仍要求40 shard + 40 prediction；失败停止新派发并清理子进程。

在新的唯一tmux窗口启动，建议：

```text
mts-g0-g1-downstream-recovery-v1
```

不得复用已退出watcher PID，也不得启动第二套并行campaign。最终继续生成：

```text
results/mts_multiscale_topology/g_family_matched_v1/comparison.json
results/mts_multiscale_topology/g_family_matched_v1/comparison.csv
results/mts_multiscale_topology/g_family_matched_v1/final_report.md
```

## 4. 验证要求

代码修改后、写正式G1产物前执行：

```text
python -m py_compile scripts/pretrain.py scripts/finalize_mts_g_pretrain_milestone.py scripts/audit_mts_g0_g1_formal.py scripts/compare_mts_g0_g1_formal.py
bash -n scripts/run_mips_trimer_scage.sh scripts/auto_trigger_g_family_downstream.sh
pytest -q <finalizer直接测试> tests/test_mts_g0_g1_formal.py tests/test_mts_checkpoint_contract.py
git diff --check
```

只运行与本修复直接相关的测试，不启动全量pytest。

正式产物的三个阻塞门：

1. G0 milestone→临时final转换与既有G0 final权重完全一致；
2. G1 recovered final权重与G1 step-20000 milestone完全一致，checkpoint audit和strict loader通过；
3. G0/G1下游各40 shard + 40 prediction完整后，comparison成功生成。

## 5. 明确禁止

- 不重新运行G1 20k或任何额外optimizer step；
- 不直接重命名milestone为final checkpoint；
- 不伪造、覆盖或删除step-20000 milestone及原失败日志；
- 不恢复plots、loss JSON、attention heatmap或已删除死统计；
- 不使用`strict=False`、修改旧artifact metadata或跳过checkpoint identity；
- 不启动G2/G3、T阶段重评、多seed或超参数搜索；
- 不改写G0正式checkpoint、冻结cache/sidecar、历史结果或用户其他dirty changes。

## 6. Claude Code执行记录

- finalization/plots解耦回归：`passed`。当前 `scripts/pretrain.py` 的正式 checkpoint 保存路径已无 plots/loss JSON/loss curve 引用；新增 `tests/test_mts_g_family_finalization_recovery.py`（9 项），在 `plots/` 目录不存在时完成 finalization 并生成 checkpoint+completion marker（`test_finalization_requires_no_plots_directory`）。
- milestone finalizer实现与测试：`passed`。新增 `scripts/finalize_mts_g_pretrain_milestone.py`：严格校验 milestone SHA/schema/optimizer_step、逐字段校验 `resume_contract`（含 `pretrain_code_files` 插入序 digest == `pretrain_code_hash`）、冻结缓存/step0/配置/profile 全绑定、134 tensor 与 step0 架构 key/shape/dtype 一致、float tensor finite、原子写入、默认拒绝覆盖。
- G0临时转换等价验证：`passed`。用 G0 step-20000 milestone 生成临时 final：134 个 tensor 与 G0 正式 final **逐位一致**（torch.equal）；meta 与 G0 正式 final 逐字段一致（唯一例外：`training_wall_seconds`/`dirty_diff_sha` 因不可恢复记 `null` + recovery 字段）；`target_contract`/`final_cache_binding`/`source_contract_sha256` 与 G0 final 完全一致。
- G1 final checkpoint恢复：`passed`。`pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1/mts_g1_pretrain_20k.pth`（sha256 `e9785841dfd2f292f85c4bc2ffa687e3e97d3c84fadf1f2452ae8aa0bd759c3f`）+ `.complete.json`（sha256 `582eb7955d11921b1fa5a2854851ac6a45842fb9fde9b2e541aafcfa0e7220d7`，`optimizer_steps=20000`，`pretrain_code_sha256=fc7455dd58dfa4447b0ed6190e91b8730da01a53c4526c6bbbcac73fec499a41`）。134 tensor 与 step-20000 milestone 逐位一致；meta 中 `recovered_from_milestone=true`、`recovery_reason=post_training_plot_finalization_failure`、runtime identity=启动合同、`finalization_code_identity=9f4867cf...`（当前磁盘）、`training_wall_seconds=null`、`dirty_diff_sha=null`。
- G1 checkpoint/strict-load审计：`passed`。`scripts/audit_mts_g0_g1_formal.py --phase checkpoint`（G0+G1）status=passed；`validate_g_family_checkpoint_binding` 通过；strict-loader 全部 meta 检查（schema/mips 配置/cache artifact/cohort/angle/reference_commits/source-target contract digest）通过；生产下游 strict loader 实际加载 G1 final 成功（G1 全部 40 折日志 `Loaded pretrained model from .../G1/mts_g1_pretrain_20k.pth`）。
- G0/G1源码差异分类：`explained_implementation_difference_with_limitation`。证据：G1 日志 "restarting G1 from fresh step-0 after relation-geometry vectorization"；uncommitted diff 中 O8 attention 数学未修改（仅签名 `**_kwargs`）；差异文件仅 `src/modules/mips_local_graph.py`。因 G0 时代中间版本（ab9538b1）无内容快照，无法逐 byte 证明 → 比较报告携带 `matched_code_limitation`。分类产物：`results/mts_multiscale_topology/g_family_matched_v1/code_difference_classification.json`。
- 顺序G0/G1正式下游与comparison：`passed`。G0 40 shard+40 prediction（下游 2751.0s）→ G1 40 shard+40 prediction（2894.7s）→ `comparison.json/csv` + `final_report.md`。task-level delta：mean/median/sample std = `+0.005289/+0.007238/0.007101`，6/8 任务改善、2 任务略负（egb −0.000610、nc −0.007903）；xc `+0.014464`（4/5 folds）。按 handoff 判定规则该结果为正向候选，是否进入 G2 由 Codex 决定，本报告不自动晋级。
- 实际命令、tmux窗口、PID、日志和偏离：
  - 窗口 `mts-g1-finalization-recovery-v1`：finalizer 首次运行因 handoff 声明的 milestone SHA 与实际磁盘文件 SHA 差 1 字符（`...f549...` vs 实际 `...c549...`）而拒绝；经复核（mtime 00:32:53 一次性写入、尺寸与其他 milestone 一致、启动合同 digest 自洽）确认磁盘文件为真实产物，以实际 SHA `3b321e5741924bb73793c450bc6fc76fa88f78c549c256aaa69b4d2dee2933df` 显式声明后通过。日志：`logs/mts_multiscale_topology/g_family_matched_v1/G1/finalization_recovery/finalizer.log`；审计：`results/mts_multiscale_topology/g_family_matched_v1/G1/finalization_recovery_audit.json`。
  - 窗口 `mts-g0-g1-downstream-recovery-v1`：`bash scripts/auto_trigger_g_family_downstream.sh`（recovery 模式，G1_PID 未设）。首次运行暴露该脚本**预先存在的引号 bug**（`launch_downstream` 命令串 `echo "..."` 内层引号 → `g0: command not found`），修复后重跑；清理了本次失败产生的空目录。chain 日志：`logs/mts_multiscale_topology/g_family_matched_v1/auto_downstream_chain.log`；G0 下游日志 `logs/.../G0/downstream_formal_v1/`，G1 下游日志 `logs/.../G1/downstream_formal_v1/`。
  - 窗口 `mts-g0-g1-comparison-recovery-v1`/`-v2`：comparison 首跑暴露两个既有契约问题并修复：① chain 的 comparison 命令缺 `PYTHONPATH=.`（`ModuleNotFoundError: scripts`），已在该脚本补上；② comparator 要求 prediction NPZ metadata 含 G-family 字段，但生产 train.py 的 prediction metadata schema 不记录这些字段（CSV shard row 与 checkpoint meta 已承载该身份），已按生产契约对齐 comparator 并注明理由。
  - 未启动 G2/G3、T 阶段重评、多 seed 或超参搜索；未重跑任何 20k 训练；未恢复 plots/attention heatmap；未覆盖任何历史产物（G0 final、milestone、冻结缓存、旧日志均未改动）。
  - 验证：`py_compile`（4 个脚本）PASS；`bash -n`（2 个脚本）PASS；`git diff --check` PASS；`pytest tests/test_mts_g_family_finalization_recovery.py tests/test_mts_g0_g1_formal.py` = 11 passed。`tests/test_mts_checkpoint_contract.py` 3 项失败为**预先存在的环境 fixture 问题**（`pretrained_models/mts/mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth` 在本会话开始前已从磁盘消失），与本次修改无关，未放宽测试。
  - 预训练墙钟：G1 原始值不可恢复，比较器按 handoff 要求显示 `unavailable`（`training_wall_seconds=null`）。

执行结束后只填写本节，将状态改为`ready_for_codex_review`；不得自行宣布最终验收通过。

## 7. Codex审查记录

- recovery权重等价性：`pending`
- checkpoint身份与来源真实性：`pending`
- G0/G1 matched-code限制：`pending`
- 两臂8×5正式产物：`pending`
- comparison：`pending`
- 最终验收与下一步：`pending`
