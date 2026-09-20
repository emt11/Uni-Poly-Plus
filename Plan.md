# PH 下游保留与匹配融合验证计划

## 0. 交接头与授权

| 字段 | 内容 |
| --- | --- |
| 计划 ID | GLT-GALPH-PHRETENTION-20260920-01 / r4 |
| 日期／状态 | 2026-09-20；**待审查**：r4 的**单条 C1_REPAIR_5K 长程确认**已执行完毕（5000/5000 updates，exit=0，证据见 §9.6、结论与判读见 §13），**待 Codex 审查**。18 units 仍未执行，研究目标未完成、未获验收 |
| 用户要求 | ① 原始：同一 C1 主干下，下游**保留样本特异 PH**是否优于**关闭 PH**与**同容量固定 PH 分支**；② r4：从原始 common initialization 出发、沿用已验证的 R_REPAIR 配方，跑**一条** ≤5000 optimizer updates 的 C1 轨迹，在指定监测点记录 PH 可训练性、固定 probe 输入区分性与残差占比，并做 strict-load 一致性检查 |
| 授权范围 | 本轮**仅一条轨迹**，累计 ≤5000 optimizer updates（含失败前已执行的更新与任何重复步骤）；两个尾项修复（gate 条件、重复三次的可观测差异）与相关局部 CPU 测试；沿用 r2/r3 已交付的代码 |
| 明确禁止 | 18 development units；重跑 N0/N1/C0；outer-test；额外 seed；XATTN；multiscale；覆盖旧 checkpoint/sidecar/日志/证据；以"最终轨迹仍是 5k"为理由超出累计预算 |
| 角色 | Codex 规划与独立审查；ZCode 执行运行、监测、一致性检查与文档并回填 §9；执行者不宣布 Codex 最终验收通过 |
| 基线 | dev：r3 基线 0e22534（pull 后 0/0）→ r4 tail 修复 commit `dbb1225` → 本轮交付 commit（见 §9.6 末行）；执行前后各 fetch 一次，远端无新进展、无分叉 |
| 产物根目录 | `results/glt_galph_ph_retention_20260920/{p0,p1}`、`logs/glt_galph_ph_retention_20260920/` |

本文件是拟实施合同，不是既有 CLI。执行端在阶段结束时回填 §9；状态只在有证据时推进。

### 0.1 上一周期衔接与交接更正

* CANON3D（`GLT-CANON3D-20260920-01` r1–r2）已按用户要求完整归档至 `PROJECT_HISTORY.md`，标记为**被本任务替换、未完成事项未获验收**（P0 审计无独立审查；P1 代码在用户指示下回滚并确认删除；P2–P5 与 r2 的 M0/M1 从未运行）。
* GALPH 四臂结果（`results/glt_galph_20260920/summary.json`）作为**已有 development 证据**保留，本轮不重跑、不改写。交接表述更正：
  1. **C1/eat/fold0 的 `best_epoch=30` 触及了 30 epoch 预算上限**，不能声称"没有单元触及上限"；该单元最佳值出现在最后一轮，是否被预算截断**未核实**。
  2. C1/xc/fold1 的 `best_epoch=20`（补齐该单元记录）。
  3. `C0−N0` 是 **CLS 编码 + 下游读出组合**的变化，不是纯 CLS 机制的归因。
  4. commit 时间与运行启动时间只提供**版本追溯线索**，不构成某 arm 实际运行代码的证明（`results/glt_galph_20260920/commit_attribution.json` 已按此口径写明）。
  5. 上一轮报告的"56 passed"**当时未落盘最终日志**：本轮已按要求重跑并保存完整输出到 `logs/glt_galph_ph_retention_20260920/preexisting_tests.log`，以该日志为准（见 §9）。
* 上一周期遗留（与本轮无关、仍未关闭）：S5 launcher 完成判定路径与 aggregator 逐折校验缺口。

### 0.2 r2 修订说明（范围与保留项）

* **r1 保留**：`p0/ph_sidecar_downstream/`（809 结构、809 valid、0 invalid，**本轮未重建**）、`p1/` 的 updates/smoke 有界验证产物、`p1/blocker_evidence.json`、`logs/.../preexisting_tests.log` 与 `p0_sidecar_build.log`。
* **r1 未执行、仍无结论**：18 个 development units 与 §7 汇总门控；任何 F_REAL/F_CONST 性能对比都不存在。
* **r2 不改变** §1–§8 登记的科学问题、三个对照、§7 判据与预算；§2–§8 继续作为**暂停中的登记合同**，本轮只做证据修正、最小返修与可训练性预检。
* r2 的证据修正与机制结论集中在 **§11**；执行记录与证据溯源见 **§9**。
* 两臂预检是"衰减策略＋门控初始化"的**组合**修复预检，**不得把两臂差值解释为单独某一项的因果效应**。

### 0.3 r3 修订说明（审查返修）

Codex 对 r2 交付提出两项诊断缺陷；r3 的修复与替代范围见 **§12**（逐项回应）。要点：

* `scripts/diagnose_glt_galph_ph_degeneration.py` 的 model 段被**重算**：`results/glt_galph_ph_retention_20260920/p1/ph_degeneration_diagnostics_r3.json` 取代 `ph_degeneration_diagnostics.json` 的 **model 段**；旧文件保留不删，取代关系记录在新文件的 `supersedes` 字段（含旧文件 sha256）。
* encoder 段与 step0 段**未受影响**，重算后逐字段相同（§12.1 证据），因此这两段以任一文件为准。
* r3 **没有**任何 optimizer update，未重跑 256 步预检，未触碰 §11.4 的预检证据（其 `residual_relative_norm` 的口径说明见 §12.3）。

### 0.4 r4 修订说明（长程确认；预算独立记录）

* **r2 的 1026 updates 与超预算 514 的历史记录保留**（§9.2 "预算偏差"行），**不因本轮而抵销或改写**；r4 的 5000-update 预算**独立记录**（§9.6）。
* **r3 的修复与替代范围保留**：`ph_degeneration_diagnostics_r3.json` 取代旧文件 **model 段**（encoder/step0/probe_set/fixed_window 段相同），旧文件保留；相关 CPU 测试保留。
* **原 18-unit 研究问题仍未回答**：r4 只回答"修复配方在 5k 上是否仍可训练、输入区分性是否保持"，**不回答** PH 保留是否提升属性预测。
* 本轮 tail 修复（记录见 §9.6）：诊断的 checkpoint 条件直接使用原始 `alpha_ph` 张量（不再 `round` 后经 `atanh` 重建）；同输入重复**三次**并报告**最大观察差异**（不再称为严格噪声上界）；§11.6/§12.1 中"step ≥1000 输入无关"的表述已改为"step 1000 仍有可观测区分性，step ≥3000 在已检查输入与精度下逐位一致"。修正在 `dbb1225` 落地，**早于 5k 轨迹启动**（时间线见 §9.6）。
* **执行结果**：`results/glt_galph_ph_retention_20260920/p1/pretrain_C1_REPAIR_5K/`（5000/5000 updates、exit=0、81.3 min）与只读诊断 `p1/ph_diagnostics_C1_REPAIR_5K.json`；判读与限制见 **§13**。
* **不做**：不因 tail 修复重跑旧 256 步预检或整套历史诊断（按 r4 合同第二节）。

## 1. 科学问题与可证伪假设

**问题：** 同一个已核实的 C1 `deploy_05000.pt` 主干，在下游**保留样本特异 PH**，是否优于**关闭 PH**，以及优于**同容量的固定 PH 条件分支**？

不验证 PH-XATTN，不改变预训练目标，不宣称聚合物 3D 已经有效。

**可证伪形式：** 若 F_REAL 相对 F_OFF 与 F_CONST 在三个任务两折均值上都达不到 §7 的工程阈值，则"下游保留样本特异 PH 有增益"不成立，本轮停止，不扩展。

## 2. 锁定实验组

三组**均从同一个已核实的 C1 `deploy_05000.pt` 加载**：

| 组 | 下游 PH 输入 |
| --- | --- |
| F_OFF | 实例化全部新增 PH 融合模块，但通过**明确分支**把其贡献置零 |
| F_CONST | 固定的 **P_train 平均 PH profile**（同一向量给所有样本） |
| F_REAL | 该样本**自己的真实 PH profile** |

三组共享：C1 checkpoint；下游公共参数与新增模块的初始张量；样本顺序、模型 seed、DataLoader 随机流；split、目标标准化、优化器、学习率、batch、损失、早停规则。公共参数一致性**用张量相等检查**，不以"相同 seed"代替。

**F_OFF 必须重新运行**，不得拿旧 C1 指标充当本轮控制组。

**设计决定（执行端记录，供审查）：** 预训练主干内部的 PH 残差路径（`ph_encoder → ph_to_summary → tanh(alpha_ph)` 加到 `summary3/g3`）在三组中**一致关闭**。理由：(a) CLS 读出使用 `out['cls3']`（层后、加 PH 残差之前的状态），该内部残差**不进入本轮属性预测**，若只在 F_REAL 打开会引入第二个未被控制的变化；(b) 保证三组差异只有 §3 的新残差项。若审查要求改为"F_REAL 同时保留预训练内部 PH 条件"，须作为新修订重新授权。

## 3. 最小模型修改

保留现有 CLS＋mean 读出：

$$r_3 = W_{readout}([\mathrm{CLS}_3, \mathrm{mean}(\mathrm{bond\_states})])$$

新增：

$$p_{ph} = \mathrm{PH\_encoder}_{frozen}(\mathrm{profile}),\qquad r_3^{new} = r_3 + \tanh(\gamma)\, W_{ph}\, p_{ph}$$

再接现有 2D/3D gate、融合与属性 head（数学定义不变）。

要求：

1. PH encoder 从 C1 的 `scale-interaction-v2` 加载，参数**冻结且保持 eval**；调用整个模型 `.train()` 后仍必须满足。
2. `W_ph` 为独立下游投影，`gamma` 初始化为零；三组初始张量完全相同。
3. 新残差**明确加到实际参与属性预测的 r3**，不能只放进 batch 或 `encoder.g3` 后被 CLS readout 绕过。
4. 下游使用**完整 8 个 PH patches**，不做随机 PH masking。
5. 不增加 PH reconstruction loss，不改变属性损失。
6. 不改 O8、六层 GLT、CLS 更新、原 gate 或 pooling 的数学定义。
7. 冻结、不优化无效的预训练专用参数（`head_2d/head_3d/cl_proj2/cl_proj3/ph_head/ph_to_summary/alpha_ph` 与 `ph_encoder`），避免把未参与预测的参数计为有效新增容量。
8. F_OFF 的预测、公共梯度与一次更新须与同初始化的原 C1 下游路径一致；新增模块初始化不得扰动公共随机流。
9. 零门控首步允许 PH 投影梯度为零；须验证门控能获得有效梯度、后续步骤可训练。

## 4. PH 数据

1. 复用冻结下游 Trimer 坐标，使用现有 `glt-ph-betti-v2` 算法；不生成构象、不改坐标、不写旧缓存。
2. 只为本轮各 task/fold 的 **train 与 validation** 所需结构构建独立小型 PH sidecar，**按结构 key 去重**。
3. 不为本轮评估读取 outer-test；同一结构因参与其他 fold 的合法训练集而被缓存，不改变逐 fold 的 train/validation/test 使用边界。
4. 始终按**完整 32-byte sample key** 查找与校验；训练位置只作快路径。
5. F_CONST 的 profile 由已有 P_train sidecar 中**有效 profile 的均值**计算，不使用下游 validation/test 统计。
6. 三组相同的几何与 PH 可用性处理：invalid 样本保留属性训练；PH 不可用时残差为零；F_CONST 与 F_REAL 使用相同有效性 mask；非有限值不得简单乘零后继续传播。
7. 记录覆盖率、失败原因与实际样本数；不得通过过滤样本改善指标。

## 5. 必要验证

仅运行与本次改动相关的检查：

* 完整 PH profile 的 shape、尺度顺序、key 对齐与 collate；
* 含 NUL 的 key、跨 epoch/乱序位置查找；
* PH encoder 冻结与 eval 状态（`.train()` 之后仍成立）；
* invalid PH 零残差且属性任务正常；
* F_OFF 回退 parity（预测 / 公共梯度 / 一次更新）；
* 三组公共初始化、样本顺序与公共随机流一致；
* 非零门控 fixture 中，更换有效 PH 可影响 F_REAL 的实际属性预测；
* F_CONST 不随样本 key 改变其固定 profile；
* checkpoint 保存与 strict-load 往返；
* scaler 只拟合最终 train，checkpoint 只按 validation R² 选择；
* 禁止 outer-test 的运行路径检查。

**有界模型验证：** 三组各 ≤2 次 optimizer updates；三组各运行 XC/fold0 且 ≤2 epochs 的独立 smoke；smoke 产物**不计入**正式 development 结果。验证失败即停止受影响阶段，局部修复后只重跑相关验证。

## 6. development 预算

验证通过后执行 **F_OFF / F_CONST / F_REAL × XC/EPS/EAT × fold0/1 = 18 units**：

* outer5_inner20 固定 split；seed 42，沿用既有 fold seed 派生规则（`seed + fold`）；
* FULL adaptation，唯 PH encoder 与预训练专用参数按 §3 冻结；
* 最多 30 epochs、patience 10、warmup 5、encoder LR 1e-5 / fusion+head LR 1e-4、wd 0.02、train batch 32 / eval 64、train-only 标准化；
* 不做 train+validation refit；每单元独立保存 validation 最佳 checkpoint；
* 新增预训练预算 **0**；不为 `best_epoch=30` 的组单独延长；正式运行失败保留现场，不换 seed/超参/预算绕过。

GPU、worker 或超过一分钟的任务在 tmux `Uni-Poly` 独立 window 执行并记录完整命令与日志。

## 7. 汇总与晋级口径

报告：三组各 task/fold 的 `best_validation_r2` 与 `best_epoch`；逐任务两折均值与三任务等权均值；`F_REAL−F_OFF`、`F_REAL−F_CONST`、`F_CONST−F_OFF`；PH 覆盖率、门控值与残差相对范数；时间、显存、实际训练 epochs；所有失败、重试、预算末端最佳值与未核实事项。

**工程筛选标准**（阈值不是统计显著性，不称盲测或正式优胜）：

1. F_REAL 相对 F_OFF 与 F_CONST 的三任务均值均至少提升 0.005；
2. 声称改善 XC 还须：相对两个控制组，XC 两折均不退化、XC 均值均至少提升 0.01；
3. 任一任务均值相对任一控制组下降超过 0.01 → 标记风险并交回审查；
4. 只有 EPS 收益时仅标为 EPS 候选；
5. 不达标准即停止扩展，禁止自动启动 XATTN、multiscale、20k continuation、额外 seed 或完整正式评测。

## 8. 产物、日志与停止规则

`results/glt_galph_ph_retention_20260920/{p0,p1,p2}/`、`logs/glt_galph_ph_retention_20260920/`；不提交缓存、checkpoint 或大型实验产物。交付后停止，不自动进入下一阶段。发现实际实现与上述科学定义矛盾时，先只读定位并报告，不静默改变实验问题、对照或预算。

## 9. 执行记录（ZCode）

以实际执行的命令、日志、退出码和产物为准。**历史报告、本轮重跑与独立审查在 §9.3 中区分。**

### 9.1 r1 记录（保留，不改写）

| 项目 | 状态 | 证据／限制 |
| --- | --- | --- |
| 执行前核对与 pull | 完成 | `git status` 干净；`pull --ff-only origin dev` 经代理 + HTTP/1.1 成功（Already up to date，0/0）；直连 HTTP/2 曾报 framing 错误 |
| CANON3D 归档 | 完成 | `PROJECT_HISTORY.md` 新增条目：被替换、未验收、未完成事项逐项列出，r1/r2 文本定位到 `b61876d`/`a231c05` |
| r1 合同 | 完成 | 本文件 §0–§8（r1 版） |
| 交接更正 | 完成 | §0.1 五项；`logs/glt_galph_ph_retention_20260920/preexisting_tests.log` = 56 passed, 3 warnings in 66.43s（2026-09-20 12:23 UTC 重跑） |
| 下游 PH sidecar (P0) | 完成 | `results/glt_galph_ph_retention_20260920/p0/ph_sidecar_downstream/`：809 结构、809 valid / 0 invalid、按 32-byte key 去重（xc 432 新 / eps 15 复用 + 367 新 / eat 380 复用 + 10 新）、构建 12.9 s；F_CONST 均值来自 P_train sidecar 的 911,391 条 valid profile |
| 实现 | 完成 | `src/dataset/glt_ph_downstream.py`、`src/modules/glt_galformer_ph_retention.py`、`src/modules/glt_galformer_ph_downstream.py`（`adjust_r3` 钩子）、`scripts/build_glt_ph_downstream_sidecar.py`、`scripts/finetune_glt_galformer_ph_retention.py`、`scripts/aggregate_glt_galph_ph_retention.py`、`scripts/diagnose_glt_galph_ph_retention_blocker.py` |
| 最小验证 (11 项) | 完成 | `tests/test_glt_galformer_ph_retention.py` 12 passed（r1 版本） |
| 有界模型验证 | 完成 | 三组各 2 updates + 三组各 XC/fold0 2-epoch smoke，全部 exit=0；`p1/{updates_*,smoke_*}` |
| 18 development units | **未执行（阻断）** | 见 §11 |
| 汇总与门控 | 未执行 | 依赖上一行 |
| Git（r1） | 完成 | commit `41d33b0` 已推送 `origin/dev`，远端核验一致 |

### 9.2 r2 记录（保留）

| 项目 | 状态 | 证据／限制 |
| --- | --- | --- |
| 执行前核对与 pull | 完成 | `git status` 仅一个未跟踪临时脚本（本轮返修前删除）；`pull --ff-only origin dev` 经代理 + HTTP/1.1 成功（Already up to date，0/0）；直连报 `GnuTLS recv error`；无活动训练进程、4 张 GPU 空闲 |
| §3.1 梯度 parity 返修 | 完成 | `tests/test_glt_galformer_ph_retention.py`：改用 `dict(named_parameters())` 逐参数比较；`None` 只在两侧同为 None 且该参数已冻结或属 `INACTIVE_DOWNSTREAM`（无 2D/3D mask 行时天然无梯度）时接受；新增"至少一个主干参数 + 一个属性 head 参数有非零有限梯度"断言；保留预测、一次更新后公共参数一致性检查 |
| §3.2 runner 返修 | 完成 | `scripts/finetune_glt_galformer_ph_retention.py`：`resolve_protocol` 强制 updates/smoke/development 三选一、`--updates` 必须为正整数；`resolve_selection` 在所有模式限定 xc/eps/eat × fold0/1；`epoch_budget` 限定 smoke ≤2、development ≤30；`runtime_status` 使"实际步数少于请求步数"时只能 FAIL 并抛出；未新增 outer-test 路径 |
| §3.3 sidecar builder 返修 | 完成 | `scripts/build_glt_ph_downstream_sidecar.py`：目标目录非空即 `FileExistsError`（拒绝覆盖）；key 非 32 字节、Trimer 取不到时报错，退化的合法几何继续走 invalid 合同；**本轮未重新构建 sidecar** |
| §3.4 aggregator 返修 | 完成 | `scripts/aggregate_glt_galph_ph_retention.py`：校验目录 group/task/fold 与文件内身份一致、protocol、冻结 C1 `checkpoint_sha256`、`best_validation_r2` 有限、`best_epoch ≤ epochs_configured ≤ 30`；`risk_flags` 非空时 `VERDICT=NEEDS_REVIEW`（附 `verdict_note`），不再自动晋级 |
| 局部测试 | 完成 | `pytest tests/test_glt_galformer_ph_retention.py` = **15 passed, 2 warnings in 16.78s，EXIT=0**（r1 的 12 项 + r2 新增 3 项，日志 `logs/glt_galph_ph_retention_20260920/retention_tests_r2.log`）；相关既有套件 = **71 passed, 3 warnings in 78.85s，EXIT=0**（`preexisting_tests_r2.log`）。两套日志均在**最终提交代码**上重跑，首行记录命令、末行记录 EXIT=0 |
| runner 无扰动 parity | 完成 | `parity_two_updates`（2 updates，监测开启）：4 个 rank 的 `stream_digest`/`global_counts`/`losses`/`sums`/`grad_norm_preclip`/`position_*`/`lr` 与旧 C1 记录 step 1–2 **逐位相同**；R_LEGACY 前 256 步与旧 C1 记录**全部字段逐位相同** → 返修不改变原训练路径，R_LEGACY 即"原配方只缩短停止位置" |
| §4 只读诊断 | 完成 | `scripts/diagnose_glt_galph_ph_degeneration.py` → `results/glt_galph_ph_retention_20260920/p1/ph_degeneration_diagnostics.json`（含 `ph_probe_profiles.npy` / `ph_probe_keys.json`：P_train sidecar 行 0–63，全部 valid）；step 0 由 common-init 重建并与 R_LEGACY 实测 step-0 逐项相等（`all_equal=true`）；5 个 resume checkpoint × {fp32, bf16} 结果见 §11.3 |
| §5/§6 两臂预检 | 完成 | R_LEGACY / R_REPAIR 各 256 updates，exit=0，见 §11.4；监测行步号按"完成的 optimizer updates"对齐到 0/1/2/16/64/128/256（v2 修正后重跑） |
| **预算偏差（据实报告）** | 超出授权 512 | 合同允许两臂合计 ≤512 updates。实际：`parity_two_updates` 2 + v1 两臂 512 + **v2 两臂 512 = 1026**。原因：v1 监测行步号与 §6 要求的 16/64/128 错位（落成 17/65/129），为满足合同重跑 v2。v1 未删除，保留为 parity 与确定性证据（v1 与 v2 训练记录须逐位一致）。未扩大任何训练视野（仍 ≤256）、未新增 arm、未改配置 |
| **失败记录（已修复）** | 无科学影响 | 诊断脚本两次启动失败：① mask 张量建在 CPU 导致 device mismatch；② 误从 `identity` 顶层取 `sample_index_artifact`（实际在 `identity['config']`）。两次都在写出任何 JSON 之前报错退出，未产生半成品证据、未影响预检；修复后一次成功（`DIAG_EXIT=0`）。另：v1 监测行步号错位（§9.2 预算偏差行）是**合同符合性**问题而非运行失败 |
| 18 development units | **仍未执行** | 本轮未授权，未启动 |
| 汇总与门控 | 仍未执行 | 依赖上一行；且即使执行，也受 §11 的机制修正约束 |
| Git（r2） | 完成 | commit `d372b08`，分支 `dev`，已推送 `origin/dev`；fetch 后 `HEAD...origin/dev` = 0/0，`origin/dev` 指向 `d372b08` 且与本地树一致；推送经代理 + HTTP/1.1（直连报 `GnuTLS recv error`）。未 force push、未改写历史 |

### 9.3 证据溯源与限制

以实际执行的命令、日志、退出码和产物为准；**历史报告、本轮重跑与独立审查在此区分**。

* **历史报告**（非本轮执行）：`p0_sidecar_build.log`、`p1/{updates_*,smoke_*}`、`blocker_evidence.json`、`preexisting_tests.log`、四臂 5k 产物 `results/glt_galph_20260920/**`。
* **本轮重跑**：§9.2 的测试与预检——命令、日志路径、退出码齐全（`preexisting_tests_r2.log` 首行记录命令、末行记录 EXIT=0；预检日志 `logs/glt_galph_ph_retention_20260920/p2/{chain_status.log,pretrain_R_*.log,diagnose_degeneration.log}`）。
* **独立审查**：本轮**没有**。以上全部为执行端自检；Codex 未检查的部分一律标为未核实。
* 数值口径：诊断同时记录 fp32 与生产 bf16 路径、各自的容差与"重复前向噪声地板"（fp32 ≈2.7e-7–7.6e-7 相对 |summary3|，bf16 ≈4.4e-3–1.1e-2）。**bf16 下任何小于该地板的差异都不作为信号**；本文件出现的所有"可观测/不可观测"判断都以此为界。
* 参数统计保留原始精度（科学计数法），未提前 round 为 0；`ph_encoder.norm.weight` 等个别张量仍为 O(1)，不因分支整体塌陷而一并归零。

### 9.4 运行环境与命令

* tmux session `Uni-Poly`，窗口 `galph_r2_pretrain`；预检脚本 `/tmp/r2_prechecks.sh`（内容与下列命令一致），启动方式与既有 smoke 相同：`python3 -m torch.distributed.run --standalone --nproc_per_node=4 scripts/pretrain_glt_galformer_ph.py --config configs/mts/glt_galph_c1.json --cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1 --cache-root data/processed/mips_trimer_scage --dual-static-root data/processed/glt_dual_v2/pi1m/dual_static_v1 --output <dir> --prep-workers 12 --log-every 100 --updates <N>`（R_REPAIR 另加 `--ph-no-weight-decay --alpha-ph-tanh-init 0.02`；两臂均加 `--ph-monitor <dir>/ph_monitor.jsonl --ph-monitor-steps 0 1 2 16 64 128 256 --ph-probe ... --ph-const-profile ...`）。
* 预检**未**保存 resume/deploy checkpoint（`save_every=1000`、`deployment_step=5000` 均未触发），未触碰 `results/glt_galph_20260920/**` 与 `p0/ph_sidecar_downstream/**`。
* 未执行诊断 backward（监测所需的 PH 梯度直接取自训练自身的 backward），因此不存在"额外 backward 污染下一步梯度/RNG/样本位置/scheduler"的风险来源。

### 9.5 r3 记录（审查返修，本轮；optimizer updates = 0）

| 项目 | 状态 | 证据／限制 |
| --- | --- | --- |
| 诊断修复 | 完成 | `scripts/diagnose_glt_galph_ph_degeneration.py`：① 每个条件使用各自的浅拷贝 batch 视图（`_with_profile`），不再原地改 `batch.ph_profile`；② 残差与两个 head 全部在同一 autocast 上下文内取得（残差由模型自身张量 `g3 − where(valid3, cls3, 0)` 反推，并另留同上下文重算的"直接视图"避免相消限制）；③ `alpha_ph` 用 `copy_` 逐位还原、训练模式在 `finally` 中还原 |
| 新增局部测试 | 完成 | `tests/test_glt_galph_ph_diagnostic.py` **5 passed**（输入不变性／调用顺序独立性／状态恢复／残差与 head 同精度／mask 检查不污染输入），CPU + stub，无需 GPU、数据与 checkpoint；日志 `retention_tests_r3.log` 含命令与 EXIT=0 |
| 只重算前向诊断 | 完成 | `p1/ph_degeneration_diagnostics_r3.json`（`DIAG_R3_EXIT=0`，日志 `p2/diagnose_degeneration_r3.log`）；**0 次 optimizer update**，未重跑 256 步预检，未启动 5k/18 units |
| 旧证据与替代范围 | 完成 | 旧 `p1/ph_degeneration_diagnostics.json` **保留未删**；新文件 `supersedes` 记录其路径与 sha256 `3e762dc4…`；重算比对：encoder 段 12/12 逐字段相同、`step0`/`probe_set`/`fixed_window` 段相同 → **替代范围仅限 model 段** |
| 预算 | 未超出 | 本轮 optimizer updates = 0；总新增算力为 2 次前向诊断（各约 3 min、单卡）与 CPU 测试 |
| Git（r3） | 完成 | commit `0e22534`（诊断两处缺陷修复），其后 `4e48f47` 为文档补充；推送与远端核验同 r2 流程（代理 + HTTP/1.1） |

### 9.6 r4 记录（单条 C1_REPAIR_5K 长程确认；实际累计 5000 updates，未超预算）

| 项目 | 状态 | 证据／限制 |
| --- | --- | --- |
| 执行前核对与 pull | 完成 | 起点 `git status` 干净、`HEAD=0e22534`；`fetch/pull origin dev` 经代理 + HTTP/1.1 成功，0/0；4 张 GPU 空闲、无活动训练进程；交付前再次 fetch，远端仍 0/0（无分叉、无他人新提交） |
| 两条 tail 修复 | 完成 | commit `dbb1225`（2026-09-20 13:37:09 UTC）：① 诊断的 checkpoint 条件直接用原始 `alpha_ph` 张量，强制门控作为**单独标注的反事实**并同时记录请求值与经 fp32 `atanh` 实现值；② 同一输入前向 3 次，报告 `repeat_max_observed_difference`（实测最大差异），10× 裕量明示为**约定**（`resolution_threshold`）而非已证上界；③ §11.6/§12.1 措辞修正（见 §0.4） |
| 代码版本（运行态） | 完成 | runner `scripts/pretrain_glt_galformer_ph.py` sha256 `6849d648…`，与 `dbb1225` 内 blob **逐字节相同**；该文件 mtime 13:27:56 UTC、轨迹启动 13:29:57 UTC、commit 13:37:09 UTC 且此后未再改动；运行自身写出 `optimizer_groups.json`（该字段只存在于 r4 修订）→ 轨迹所用 runner = `dbb1225` 版本 |
| 训练命令 | 完成 | 完整命令见 `pretrain_C1_REPAIR_5K/runtime.json`；要点：`--config configs/mts/glt_galph_c1.json --updates 5000 --ph-no-weight-decay --alpha-ph-tanh-init 0.02`，`--ph-monitor …/ph_monitor.jsonl --ph-monitor-steps 0 1 2 16 64 128 256 500 1000 2000 3000 4000 5000 --ph-probe ph_probe_profiles.npy --ph-const-profile …/p_train_mean_profile.npy`，`--prep-workers 12`；tmux `Uni-Poly:galph_r4_repair5k`，日志 `logs/glt_galph_ph_retention_20260920/p1_C1_REPAIR_5K.log` |
| 合同参数核对 | 完成 | `identity`：arm C1 / summary_mode cls / ph_mode global / encoder `scale-interaction-v2` / seed 42 / world 4 / micro 84 × accum 3 = 1008 / bf16 / lr 2e-4 / warmup 2000 / `schedule_total_steps` 20000 / grad_clip 1.0 / loss_weights [1,1,1,1] / `max_optimizer_steps` 5000 / 原 `main_bundle_hash 30f17b59…`、`cohort_hash b03f96a1…`、数据与样本顺序与原 C1 相同 |
| 参数组成员 | 完成 | `optimizer_groups.json`：Adam，组 0 = 210 个参数、`weight_decay=1e-6`；组 1 = **PH 路径 14 个参数**（`alpha_ph`、`ph_encoder.*` 12 个、`ph_to_summary.weight`）、`weight_decay=0.0`（`weight_decay_zero=true`、`ph_path_only=true`）。无 keep-positive / 远离 0 正则，α 可自由学正负 |
| 初始化来源 | 完成 | common init `results/glt_galph_20260920/p1/common_init_galph_v1.pt`，sha256 `637827e6…`（与 R_LEGACY / R_REPAIR_v2 相同）；`alpha_ph_after_init=0.020002666860818863`（= `atanh(0.02)` 的 fp32 实现值），其余参数不额外改动 |
| 实际累计 updates | 完成，**5000/5000** | `runtime.json`: `status=PASS`、`completed_steps=5000`；`records_rank{0..3}.jsonl` 各 5000 行，末行 `step=5000`、lr `1.8661e-4`；本轨迹**无失败、无重试、无重启**（`--updates 5000` 一次跑完，13:29:57 → 14:51:13 UTC，81.3 min）。r4 另行消耗的算力只有两次**前向**诊断（0 optimizer updates）与 CPU 测试 |
| 13 个监测点 | 完成 | `ph_monitor.jsonl` 13 行 = 0/1/2/16/64/128/256/500/1000/2000/3000/4000/5000（"行标 k = 完成 k 次 optimizer update 后的状态"，step 0 为**首次更新前**的初始态）；每行含 raw `alpha_ph`、`tanh_alpha`、14 个 PH 参数范数、裁剪前/后逐张量任务梯度与衰减项、本步更新量、probe 敏感性、残差占比、4 项损失、`all_finite`。表格见 §13.1 |
| 监测的不可扰动性 | 完成 | 监测前向走 `eval()` + `no_grad()`（不消耗 dropout RNG、不产生梯度、不动样本位置与 scheduler）；实测：前 256 步 4 个 rank 的 `losses/sums/global_counts/grad_norm_preclip/ph_grad_norms/stream_digest/position_first/position_last/lr/step` 与 `pretrain_R_REPAIR_v2` **逐位相同**，且两轮共享监测点 0/1/2/16/64/128/256 的**整行 JSON 完全相同**。**限制**：500 及以后的监测点没有对照双胞胎，只能依据代码路径与前述一致性推断不可扰动 |
| 停止条件检查 | 未触发 | 13 个监测点 `all_finite=true`；无 NaN/Inf（损失非有限即抛 `FloatingPointError`，未发生）；无身份/样本对齐错误（`identity` 与记录逐位一致）；参数组正确（见上）；未覆盖任何旧产物（输出目录为本轮新建）；预算未超（5000/5000） |
| checkpoint 与部署包 | 完成 | `resume_{01000,02000,03000,04000,05000}.pt`（各 620.1 MiB）+ 固定 `deploy_05000.pt`（152.2 MiB，sha256 `3063cf8b…`）；均在本轮独立目录，未触碰 `results/glt_galph_20260920/**` 与旧证据 |
| 与 256 步证据的一致性 | 完成 | 5k 轨迹的前 256 步与 `pretrain_R_REPAIR_v2`（256 步预检）内容字段逐位相同（12 个字段 × 4 rank），共享监测点整行相同 → 5k 是预检轨迹的**直接延长**，不是新配方/新初始化 |
| 只读前向诊断 | 完成 | `p1/ph_diagnostics_C1_REPAIR_5K.json`（`DIAG5K_EXIT=0`，日志 `p2/diagnose_C1_REPAIR_5K.log`）：6 个点（step 0 由 common-init 重建 + 5 个 resume）× {fp32, bf16} × {checkpoint 原生门控, 强制 0.02 反事实}，64 条固定 probe，**0 optimizer updates** |
| strict-load 一致性 | 完成 | `deploy_05000.pt` 经 `load_galformer_deployment`（strict）加载：`keys_match=true`、204/204 张量**逐位相同**（`state_all_identical=true`，差异集为空，训练专用头 `head_2d/head_3d/cl_proj2/cl_proj3/ph_head` 明确排除）；encoder 输出在 fp32 与 bf16 下**逐位相同**；模型级诊断量（`residual_relative_norm_direct`、`g3_relative_change_const`、`g3_relative_change_shuffled`）两侧最大绝对差：fp32 **4.29e-9**、bf16 **2.13e-5**，均远低于行内 `resolution_threshold`（fp32 ≈2.75e-6、bf16 ≈4.6e-2）→ `model_within_resolution = {fp32: true, bf16: true}`；诊断的 step-0 构造与旧 R_LEGACY 实测起点一致（`matches_observed_start_state=true`） |
| 失败记录（据实报告） | 1 次判据返修 | 只读诊断的 **strict-load 判据**首次实现要求两侧诊断量"逐位/near-bitwise 一致"，把本诊断分辨力以下的浮点末位差判成"不一致"；已改为**比值单位上的绝对差**与行内阈值比较（阈值 = max(tolerance, 10×三重复观察差异)），重跑后两个精度都在分辨力内。这是**报告判据**缺陷，不涉及权重、不涉及训练，**0 optimizer updates**；轨迹本身无失败、无重试 |
| 预算 | 未超出 | 本轮授权：单条 ≤5000 optimizer updates。实际：**5000**（一次跑完，无重复步骤）。r3 的 0 updates 与 r2 的 1026 updates（超 514）**分别记录、互不抵销**（§9.2、§9.5） |
| Git（r4） | 完成 | `dbb1225`（tail 修复与 runner 记录）+ 本轮交付 commit（诊断 strict-load 判据、Plan.md r4 记录；见 §13 末行），推送 `origin/dev` 并核验远端包含本轮 commit、`HEAD...origin/dev = 0/0` |

## 10. 下一步

本轮交付后交 Codex 审查（§3 四处返修的实际 diff、§4 诊断与 §5/§6 预检证据、§9.2 的预算偏差）。审查通过前不启动任何扩展：

* 18 development units **继续暂停**；本轮没有产生任何 F_REAL/F_CONST 性能结论；
* 不自动重跑 N1/C1 的 5k，不自动进入 XATTN、multiscale、额外 seed 或正式评测，不自动重跑旧四臂；
* 若审查接受 R_REPAIR 作为修复配方，缺的是**新的单条 C1 长程确认授权**（5k 约 82 min）与随后的 18 units 授权。**256 步预检不构成"5k 不会再次退化"的证据，更不构成 PH 提升属性预测的证据**；其唯一结论是"该配方在 256 步内可训练、无衰减、输入敏感性不降"；
* 若审查选择选项 B（改科学问题：下游 PH encoder 新初始化并训练）或选项 C（归档为负结果），§11.3–§11.4 的证据可直接支撑，无需新增运行。

**r3/r4 补充（2026-09-20，执行端注记，不改写上文）**：上段"缺新的单条 C1 长程确认授权"已在 r4 获批并执行完毕——单条 `C1_REPAIR_5K`（5000/5000 updates）与只读诊断的结果见 **§9.6 / §13**；r3 的诊断返修见 **§12**。18 development units 仍**未授权、未执行**；本文件 §1–§8 仍为暂停中的登记合同，是否恢复由 Codex 规划。

## 11. 方法学阻断与 r2 机制证据

### 11.1 结论（r2 修正后的表述）

1. **不再称"encoder 是严格常量函数"**。准确表述：**在已检查的 64 条固定 P_train 输入、以及 fp32 与生产 bf16 两种执行精度下，C1 `deploy_05000.pt` 的 PH encoder 输出失去可观测的样本区分性**。该性质是**逐步**形成的：step 1000 仍可分辨（跨样本 spread 6.24e-3），step 2000 已低于容差（fp32 2.38e-7 < 1e-6；bf16 恰为 0），step ≥3000 两种精度下都逐位相同（§11.3）。
2. **不再称"PH 全程从未进入"**。准确表述：**最终 checkpoint 的 PH 输入路径已退化**（PH 路径权重被衰减、门控 `tanh(alpha_ph)≈-7.5e-8`、残差相对 3D summary 仅 6.8e-13），因此 F_CONST 与 F_REAL 无法区分；**早期（<1000 步）PH 路径的实际贡献尚未充分核实**。
3. **不把旧 N1/C1 差值重新标为"纯 PH 辅助任务效应"**，见 §11.5。

### 11.2 已确认事实（来自真实训练记录，非模拟）

`results/glt_galph_20260920/p2/c1/pretrain/records_rank0.jsonl` 每步记录了 `ph_grad_norms`；与 `resume_*.pt` 的权重范数结合，可直接量化"任务梯度 vs 耦合 weight decay"：

| step | lr | `alpha_ph` 梯度 | `ph_encoder` 梯度 | `ph_encoder.patch.2.weight` 范数 |
| --- | --- | --- | --- | --- |
| 1 | 1.0e-7 | 3.634e-3 | 0.0 | 13.0952 |
| 2 | 2.0e-7 | 3.224e-3 | 1.353e-8 | 13.0952 |
| 100 | 1.0e-5 | 2.238e-3 | 2.032e-5 | — |
| 500 | 5.0e-5 | 4.308e-4 | 1.711e-5 | — |
| 1000 | 1.0e-4 | 6.216e-4 | 2.242e-6 | 6.2554 |
| 2000 | 2.0e-4 | 6.911e-4 | 2.774e-7 | 0.3774 |
| 3000 | 1.99e-4 | 5.335e-4 | 2.862e-7 | 1.846e-3 |
| 3500 | 1.97e-4 | 9.695e-6 | 6.182e-9 | — |
| 4000 | 1.94e-4 | 2.198e-6 | 1.240e-9 | 1.009e-6 |
| 4500 | 1.91e-4 | 4.232e-7 | 7.839e-12 | — |
| 5000 | 1.87e-4 | 4.767e-8 | 7.343e-13 | 4.780e-11 |

关键读法（r2 新增的定量部分）：

* **衰减项不被梯度裁剪**：`clip_grad_norm_(1.0)` 在总梯度范数约 47–63 时把任务梯度等比缩小约 50 倍，而 Adam 内部的 `grad += wd·p` 在裁剪之后加入，`wd·|p|` 对 `patch.2` 恒为 **1.25e-5–1.31e-5**。因此**裁剪后**任务梯度（step 256 实测 2.166e-6）与衰减之比约 **1:5.8**（step 16 约 1:83），而衰减方向恒定、任务方向抖动 → Adam 的净步长由衰减主导。
* **门控零初始化是起点**：`alpha_ph` 由 `torch.zeros(1)` 初始化（`src/modules/glt_galformer_ph.py:136`），而 PH encoder 通往损失的唯一路径经过 `tanh(alpha_ph)`；第 0 步该分支梯度恰为 0，之后只按 tanh(α)（全程 ≤1.5e-4）成比例增长。
* **三方互相抑制**：`alpha_ph` 自身梯度最初健康（3.6e-3），但其因子 `ph_to_summary` 同样被衰减（4.89 → 4.3e-4），于是 α 的梯度在 3500 步后也塌到 4.8e-8；反过来 encoder 的梯度被 tanh(α) 压制。分支里只有 `ph_encoder.norm.weight` 存活（7.99），其余张量全部 →0，与"输出为常量但范数非零"的实测自洽。
* `loss_ph` 0.0149 → 0.0029 由 `ph_head` 读主干 `g3` 完成（§11.5）。

**机制支持证据（**不是**完整训练轨迹的复现）**：`results/glt_galph_ph_retention_20260920/p1/blocker_evidence.json` 的 `adam_decay_check` 记录了一个受控小实验——`torch.optim.Adam(lr=2e-4, weight_decay=1e-6)` 在 4096 维参数上、**损失梯度恒为 0** 的情况下，5000 步把范数 1.2714 压到 0.0（`max|p| ≈ 4.6e-24`）；同等条件下 decoupled 的 AdamW 保持 1.2714。它说明"耦合衰减在不被任务梯度抵住时会吃掉参数"，只与上表的真实轨迹**结论一致**，不能被当作轨迹复现，也不能单独用来归因真实运行；真实的梯度—衰减竞争以上表为准。

### 11.3 只读诊断（§4，`scripts/diagnose_glt_galph_ph_degeneration.py`）

* **probe 集**：P_train sidecar 的**行 0–63**（冻结文件里固定行序、全部 valid），键值写入 `p1/ph_probe_keys.json`，张量写入 `p1/ph_probe_profiles.npy`；**不涉及任何下游 validation/test 指标**，样本选择只看行序。
* **step 0 溯源**：旧 C1 每 1000 步存一次，没有 step-0 checkpoint。step 0 由 common-init artifact（sha256 `637827e6…`）按同一构造重建，并与 **R_LEGACY 实运行的 step-0 监测行逐项相等**（`all_equal=true`：14 个 PH 张量范数 + probe spread + real-vs-const 全部相等）；该构造又被证明可复现旧 C1 前 256 步（§9.2）。因此 step 0 视为**可证明来源**，不再标"未核实"。
* **噪声地板（决定"能否算信号"）**：同一输入重复前向的相对差地板 fp32 ≈ **2.7e-7–7.7e-7**、bf16 ≈ **4.5e-3–1.1e-2**（相对 |summary3|）。判定用 `resolution_floor = max(容差, 10×地板)`。
* **每行记录自身的精度口径**（r3 新增）：如 step 5000 的 bf16 行显示 `g3` 与残差实际是 **fp32**（`tanh(alpha_ph)` 是 fp32 参数，与 bf16 的 `ph_to_summary` 输出相乘时按类型提升为 fp32，再提升整条 summary），而 `ph_head`/`cl_proj3` 在 autocast 下确实是 **bf16**。因此"bf16 行"并不等于"每个张量都是 bf16"，这正是 r2 旧表中的口径混用所在（§12.2）。
* **残差的两个视图**：`residual_relative_norm_direct`（在模型自身的表达式与精度上下文中重算，无相消）是判读用的量；`residual_relative_norm`（由 `g3 − summary` 反推，口径与模型完全一致但受相消限制）在两个数接近时作交叉校验，退化 checkpoint 上会明显失准（step 5000 fp32：直接视图 6.78e-13、相减视图 6.01e-15；bf16：直接 6.78e-13、相减 0）。

下表为 **r3 重算后**的值（`ph_degeneration_diagnostics_r3.json`，model 段口径修正；spread 与 real-vs-const 取自未受影响的 encoder 段）：

| step | 精度 | 跨样本 p_ph spread | real vs const | 残差/参考（直接视图） | Δg3（PH→const, 强制门控 0.02） | Δ`ph_head` | Δ`cl_proj3` | 可观测 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | fp32 | 1.812e-1 | 1.117e-1 | 0.0（α=0） | **1.938e-4**（地板 7.7e-6） | 1.942e-4 | 1.871e-4 | ✅ 可观测 |
| 0 | bf16 | 1.812e-1 | 1.117e-1 | 0.0 | 1.094e-2（地板 1.07e-1） | 1.265e-2 | 1.236e-2 | ❌ 地板之下 |
| 1000 | fp32 | 6.243e-3 | 4.146e-3 | 1.494e-5 / 3.220e-3（强制门控） | 1.324e-6（地板 3.2e-6） | 1.025e-6 | 8.216e-7 | ❌ |
| 1000 | bf16 | 1.056e-2 | 1.021e-2 | 3.225e-3（强制门控） | 5.062e-3（地板 5.1e-2） | 6.243e-3 | 5.772e-3 | ❌ |
| 2000 | fp32 | 2.384e-7 | 2.384e-7 | 4.213e-5 / 5.770e-3（强制门控） | 2.746e-7 | 1.370e-7 | 1.970e-7 | ❌ |
| 2000 | bf16 | 0.0（逐位相同） | 0.0 | 5.773e-3（强制门控） | 4.576e-3 | 3.355e-3 | 4.703e-3 | ❌ |
| 3000 | fp32 | 0.0 | 0.0 | 6.262e-6 / 1.387e-3（强制门控） | 2.736e-7 | 1.006e-7 | 1.862e-7 | ❌ |
| 3000 | bf16 | 0.0 | 0.0 | 1.387e-3（强制门控） | 4.611e-3 | 2.926e-3 | 4.390e-3 | ❌ |
| 4000 | fp32 | 0.0 | 0.0 | 1.895e-8 / 2.935e-5（强制门控） | 2.665e-7 | 8.978e-8 | 1.857e-7 | ❌ |
| 5000 | fp32 | 0.0 | 0.0 | 6.784e-13 / 1.811e-7（强制门控） | 2.735e-7 | 8.372e-8 | 1.849e-7 | ❌ |
| 5000 | bf16 | 0.0 | 0.0 | 6.780e-13 / 1.810e-7（强制门控） | 4.511e-3 | 2.374e-3 | 4.630e-3 | ❌ |

* **架构可承载、训练结果不可承载**：step 0 在强制门控 0.02 下 fp32 可观测（残差/参考 1.484e-2；把 PH 换成 const 后 Δg3 = 1.938e-4、Δ`ph_head` = 1.942e-4、Δ`cl_proj3` = 1.871e-4，均远高于地板 7.67e-6），说明该残差通路本身能把 PH 输入传进 g3 与读 g3 的两个头；而 step 1000–5000 的同类 fp32 差值（Δg3 ≤1.32e-6、Δ`ph_head` ≤1.03e-6、Δ`cl_proj3` ≤8.2e-7）全部落在地板之内。（`head_2d`/`head_3d` 读 atom/bond states，不消费 g3，本就不受 PH 输入影响。）
* **bf16 的可分辨力（r3 修正后的表述）**：bf16 行里 Δg3/Δhead 的量级（2.4e-3–6.2e-3）与其**重复前向地板**（4.5e-3–1.1e-2）同量级，因此这些 bf16 差值**不能被判读为信号**；这说的是**本诊断在 bf16 下的分辨力不足**（本机未启用确定性算法、重复前向本身就有该量级抖动），**不是**"bf16 不适合 PH 比较"这类一般性结论。同一结论在 fp32 下独立成立（step 2000 起跨样本 spread < 1e-6，远低于 fp32 地板），不依赖 bf16 行**。
* **mask 无目标泄漏**：把被 mask 的 patch 内容替换为任意值，encoder 输出逐位不变 —— 在 5 个 checkpoint × 2 种精度下全部 `no_observable_leakage=true`（encoder 段 r3 重算后逐字段相同）。

### 11.4 两臂可训练性预检（§5/§6，`pretrain_R_*_v2`，各 256 updates）

两臂同架构、同 common-init、同 seed 42 / world 4 / micro 84 × accum 3 / bf16 / 原数据与样本顺序 / 原损失权重 / 原 LR 与 warmup 2000；**只缩短停止位置**（5000 → 256），**不改 warmup**。R_REPAIR 仅两处不同：PH 路径参数单独成组 `weight_decay=0`，且 `tanh(alpha_ph)` 初值 0.02。

| step | 臂 | `tanh(α)` | \|patch.2\| | \|scale_fuse.0\| | \|ph_to_summary\| | 裁剪前 PH 梯度 | **裁剪后** PH 梯度 | 衰减项 | 每步更新量(patch.2) | probe spread | 残差/参考 | 有限 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | LEGACY | 0.0 | 13.0952 | 13.0533 | 13.0399 | 0.0 | 0.0 | 1.31e-5 | — | 1.812e-1 | 0.0 | ✅ |
| 0 | REPAIR | 2.000e-2 | 13.0952 | 13.0533 | 13.0399 | 1.567e-2 | 3.302e-4 | 0.0 | — | 1.812e-1 | 1.488e-2 | ✅ |
| 16 | LEGACY | -1.343e-5 | 13.0926 | 13.0490 | 13.0356 | 3.865e-6 | 1.578e-7 | 1.31e-5 | 3.134e-4 | 1.811e-1 | 1.032e-5 | ✅ |
| 16 | REPAIR | 1.9987e-2 | 13.0951 | 13.0533 | 13.0399 | 6.562e-3 | 2.678e-4 | 0.0 | 3.010e-4 | 1.814e-1 | 1.535e-2 | ✅ |
| 64 | LEGACY | -1.522e-4 | 13.0568 | 12.9871 | 12.9748 | 1.989e-5 | 1.126e-6 | 1.31e-5 | 1.252e-3 | 1.794e-1 | 6.309e-5 | ✅ |
| 64 | REPAIR | 1.9932e-2 | 13.0954 | 13.0534 | 13.0400 | 2.733e-3 | 1.540e-4 | 0.0 | 8.544e-4 | 1.840e-1 | 8.228e-3 | ✅ |
| 128 | LEGACY | -5.070e-4 | 12.9439 | 12.7964 | 12.8026 | 5.514e-5 | 2.833e-6 | 1.29e-5 | 2.460e-3 | 1.729e-1 | 1.095e-4 | ✅ |
| 128 | REPAIR | 2.0408e-2 | 13.0960 | 13.0542 | 13.0411 | 2.072e-3 | 1.681e-4 | 0.0 | 7.461e-4 | 1.883e-1 | 4.537e-3 | ✅ |
| 256 | LEGACY | -7.563e-4 | 12.5036 | 12.0711 | 12.1869 | 1.360e-4 | 2.166e-6 | 1.25e-5 | 4.745e-3 | 1.513e-1 | 1.753e-4 | ✅ |
| 256 | REPAIR | 2.0803e-2 | 13.0965 | 13.0550 | 13.0420 | 5.802e-3 | 9.129e-5 | 0.0 | 3.940e-4 | 1.920e-1 | 5.421e-3 | ✅ |

**可训练性预检结论（仅此范围）**：

* R_REPAIR：PH 参数范数在 256 步内**不降反微升**（patch.2 13.0952 → 13.0965），衰减项恒为 0；`tanh(α)` 在 0.01993–0.02081 之间**双向**移动、未被推向 0（α 可自由学正负与归零，无正则）；PH encoder 从第 0 步就获得实梯度（1.567e-2），256 步时为 5.802e-3（裁剪后 9.1e-5，仍比同刻 LEGACY 的 2.2e-6 高约 42 倍）；有真实参数更新（每步 3e-4–8e-4）；probe 输入敏感性**上升**（1.812e-1 → 1.920e-1），real-vs-const 由 1.117e-1 升至 1.187e-1；残差/参考稳定在 5.4e-3（约 0.5%），远高于 fp32 地板。
* R_LEGACY：同样的 256 步内已经在退化——三个权重范数单调下降（patch.2 −4.5%、scale_fuse.0 −7.5%、ph_to_summary −6.5%），probe 敏感性 −16%（1.812e-1 → 1.513e-1），残差/参考 1.75e-4；裁剪后任务梯度始终 ≤ 衰减项（step 256：2.2e-6 vs 1.25e-5）。
* 两臂的损失几乎相同（step 256：2D 0.72369/0.72161、3D 1.63752/1.64227、CL 3.42904/3.39922、PH 0.00836/0.00839），可见 256 步内差异只体现在 PH 分支自身；**这不是性能比较**。
* **限制（必须与结论一起引用）**：256 步仍处 warmup（step 2000 才达到 2e-4 的峰值 LR），LEGACY 的衰减速率此后还会加快，REPAIR 的梯度也会变化；**"256 步不退化"不能外推到 5k**。两臂差异是"衰减策略＋门控初始化"的**组合**效果，本轮未做单项消融，**不得归因于其中任何单独一项**。

### 11.5 对上一轮四臂结论的约束（不改数值）

`N1−N0 = −0.0063`、`C1−C0 = +0.0051` 等数值保持不变，但其可读法受 §11.2–§11.4 约束：N1/C1 的 PH **输入**路径在预训练中退化，`loss_ph` 的下降由 `ph_head` 从主干 `g3` 预测被 mask 的 Betti patch 完成（该头不在 PH 残差通路上，其梯度不被 tanh(α) 门控）。因此这些差值**既不能读作"PH 条件化输入的价值"，本轮也不把它们统一改标为"纯 PH 辅助任务效应"**——后者同样需要单独的证据（例如关掉 PH 输入、只保留 ph_head 的对照臂），本轮未做。

### 11.6 事实 / 机制推断 / 未验证事项

**已确认事实（本轮实测，见 §11.2–§11.4）**

1. C1/N1 部署中 PH 路径参数被衰减到 1e-11 以下，`tanh(alpha_ph)` ≈ ±1e-7；`ph_encoder.norm.weight` 等个别张量仍为 O(1)。
2. 真实训练记录显示 PH encoder 的任务梯度自第 2 步起就比衰减项小 2–5 个数量级，且**衰减项不被梯度裁剪**。
3. 固定 64 条 P_train probe 上：**step 1000 仍有可观测的 encoder 区分性**（跨样本 spread 6.24e-3）、step 2000 已低于容差、**step ≥3000 在已检查输入与两种执行精度下逐位一致**（不写成"输入无关"这一更强的一般性表述）。
4. step 0（重建并与实运行 step-0 逐项相等）在强制门控 0.02、fp32 下可观测（r3 重算：Δg3 = 1.938e-4、Δ`ph_head` = 1.942e-4、Δ`cl_proj3` = 1.871e-4，地板 7.67e-6）——通路本身能用，训练后的 checkpoint 不能用。
5. bf16 的重复前向噪声地板约 4.4e-3，大于全部被测 PH 效应。
6. mask 无目标泄漏（5 checkpoint × 2 精度全部通过）。
7. 本轮返修未改变原训练路径：parity smoke 与 256 步 R_LEGACY 均与旧 C1 逐位一致。
8. R_REPAIR 在 256 步内可训练：无衰减、有实梯度与实更新、输入敏感性与残差占比不降、无 NaN/Inf。

**机制推断（有证据支持，但不是直接测量）**

1. 塌陷由**耦合 weight decay 主导**（零梯度模拟为极端情形，梯度/衰减比值与逐 checkpoint 权重轨迹为支持证据）；未做"只关衰减"的单项干预实验。
2. **三方互相抑制的死锁**（encoder ↔ `ph_to_summary` ↔ α）由轨迹一致性支持，属推断。
3. 预训练目标可能在 α=0 附近存在回拉（α 长期停在 ±1.5e-4，而同期 LR 累计可走约 0.8）。**本轮未测**损失–门控曲线，此条**未验证**。

**未验证事项**

1. 修复配方在 5k 上是否仍保持健康（256 步不能证明）。
2. PH 保留是否提升属性预测——18 units 未执行，问题**仍未回答**。
3. 早期（<1000 步）PH 路径的实际贡献。
4. 机制推断第 3 条（门控是否存在回拉）。
5. N1 是否与 C1 同构退化（本轮只复核其部署权重与旧记录，未对 N1 跑诊断）。

**r4 更新（2026-09-20，不改写上述 r2 条目）**：第 1 项已由单条 `C1_REPAIR_5K` 回答——在该修复配方下 PH 路径 5k 不再退化（判读与限制见 §13.2–§13.3）；第 4 项拿到新的实测支持（本配方下门控自发收敛到 |tanh α| ≤ 2.3e-4，但**仍未测**损失–门控曲线，故仍是推断）；第 2、3、5 项**仍未回答**。

### 11.7 修复选项现状（仍待 Codex/用户决定）

* **A（保持"同一 C1 主干 + 冻结 encoder"合同）**：本轮两臂预检给出了 A 的**最小充分集合证据**——PH 路径必须排除耦合 weight decay，且门控不能零初始化。但 A 仍需要**新的 C1 5k 预训练授权**（§10）。
* **B（改变科学问题）**、**C（归档为负结果）**：本轮证据同样可用；执行端不自行选择。

## 12. r3 审查问题逐项回应（问题—修改—验证—证据—剩余限制）

本轮**没有**任何 optimizer update；以下修改只涉及诊断脚本、其局部测试与文档；预检（§11.4）、5k 与 18 units 均未触碰。

### 12.1 审查问题 1：诊断原地修改 `batch.ph_profile` 造成跨条件输入污染

* **问题**：`_forward` 通过 `data.ph_profile = profiles` 直接改写共享 batch；`model_stats` 在同一 batch 上依次跑 own/const/shuffled，于是**后一个门控迭代的 "own" 基线读到的其实是上一次替换后的输入**。
* **影响面（定量）**：只有每个 checkpoint×精度的**第二个门控行**（强制 0.02）受影响，第一行（checkpoint 自身门控）不受影响。step 0 强制门控行：Δg3 1.912e-4 → **1.938e-4**（+1.4%）、Δ`ph_head` 1.920e-4 → 1.942e-4、Δ`cl_proj3` 1.862e-4 → 1.871e-4；step ≥3000 的 encoder 在已检查输入下逐位一致（step 1000 仍有可观测区分性、step 2000 低于容差），故差值只在小数末位（如 fp32 Δg3 2.739e-7 → 2.746e-7）。因此**结论未变，但 step 0 的数值已更正**。
* **修改**：新增 `_with_profile()`，每个条件拿到自己的浅拷贝视图（`copy.copy`，张量共享、属性独立），原 batch 永不被写；每行记录 `reference_profile: 'own'`。
* **验证**：`test_forward_never_mutates_the_batch_and_is_input_pure`（调用前后 batch 的 `ph_profile` 逐位不变；穿插 const 调用后 own 结果不变）与 `test_model_stats_row_pairing_is_call_order_independent`（用测试内独立复现的公式断言强制门控行的基线确实是**样本自身**输入，逐位相等）。
* **证据**：`p1/ph_degeneration_diagnostics_r3.json`；新旧比对见 §12.5。
* **剩余限制**：encoder 段从来不经过 batch 替换路径（直接以显式张量调用），故未受影响——重算后 12/12 逐字段相同，可作交叉验证。

### 12.2 审查问题 2：bf16 前向后以 FP32 重算 residual/head 的精度口径混用

* **问题**：`_forward` 先做（可能 bf16 的）前向，再在 **autocast 之外**用 `.float()` 重算残差与两个 head，导致 bf16 行把"bf16 前向产出的 `g3`"与"fp32 重算的残差/head"混在一起。
* **事实澄清（r3 新测量）**：在真实 C1 模型里，bf16 autocast 下 **`g3` 本身仍是 fp32**（`alpha_ph` 是 fp32 参数，与 bf16 的 `ph_to_summary` 输出相乘时按类型提升为 fp32，再提升整条 summary），而 `ph_head`/`cl_proj3` 在 autocast 下确实是 **bf16**（step 5000 bf16 行已记录：g3/residual/direct = fp32、head = bf16）。所以混用的实际影响集中在 **head 项**与"无条件转 fp32 再算"的口径表达，而不是残差本身的数值。
* **修改**：残差改为**从模型自身张量反推**（`g3 − where(valid3, cls3, 0)`，与模型内部同精度），并额外给出**直接视图**（在同一 autocast 上下文内重算模型自身的表达式，避免相减的相消限制）；两个 head 移到 autocast 内求值；每行记录 `g3/residual/residual_direct/head` 四个 dtype。
* **验证**：`test_residual_and_heads_share_the_forward_precision`（fp32 与 bf16 两种模式下，残差与其直接视图的 dtype 必须等于 `g3` 的 dtype；head 必须运行在对应精度语境——bf16 行为 `torch.bfloat16`）；`test_model_stats_restores_gate_and_mode_exactly` 另行覆盖状态还原。
* **证据**：fp32 直接视图与旧值最大相对差 **3.2e-8**（仅 float32 末位差），说明**旧 fp32 数值本身没错**；bf16 行最大相对差 **6.9e-4**（口径修正的幅度）；bf16 的 `ph_head` 项由 3.6e-3/9.0e-4/4.6e-4/3.2e-4/2.7e-4 变为 6.6e-3/3.4e-3/2.9e-3/2.6e-3/2.8e-3——修正后它们与 bf16 地板同量级，**明确不可判读**。
* **剩余限制**：相减视图在退化 checkpoint 上被相消限制（step 5000 fp32：直接 6.78e-13 vs 相减 6.01e-15，比值 0.99；bf16：直接 6.78e-13 vs 相减 0）。因此判读一律以**直接视图**与 Δg3（无相消）为准，相减视图只作交叉校验，两者比值逐行记录在 `residual_views_ratio`。

### 12.3 预检监测行的口径说明（未被要求重跑，故只说明、不替换）

§11.4 的监测行来自 256 步预检，**本轮无权重跑**（optimizer updates = 0）。这些行的 `residual_relative_norm` 由监测函数在 **fp32 探针前向**（无 autocast）内计算，前后精度自洽、**不存在 r2 诊断那种混用**；其与训练时 bf16 路径对应量的差异量级为 `bf16 eps × 残差占比`，即在该量（约为参考的 5.4e-3）上约 0.4% 相对差，不改变 §11.4 的任何结论。此说明为口径澄清，不构成对既有证据的替换。

### 12.4 过度结论的修正

§11.3 原句"bf16 不适合作这类比较"已改为：bf16 行里 Δg3/Δhead（2.4e-3–6.2e-3）与其**重复前向地板**（4.5e-3–1.1e-2）同量级，故这些 bf16 差值**不能判读为信号**；这是**本诊断在 bf16 下的分辨力不足**（本机未启用确定性算法，重复前向本身就有该量级抖动），不是关于 bf16 的一般性结论；核心结论在 fp32 下独立成立。

### 12.5 重算范围、旧证据保留与替代边界

* 旧文件 `p1/ph_degeneration_diagnostics.json` **保留未删**（sha256 `3e762dc4…`，记录在新文件 `supersedes` 中）。
* 新文件：`p1/ph_degeneration_diagnostics_r3.json`（`DIAG_R3_EXIT=0`）。
* 逐字段比对结论：`step0`／`probe_set`／`fixed_window` 段**完全相同**；encoder 段 **12/12 完全相同**；差异全部在 **model 段** → **替代范围 = model 段**。
* 旧 model 段与新 model 段的差异（定量）：fp32 checkpoint 门控行 ≤3.2e-8 相对差；fp32 强制门控行 ≤2.0e-5（含 step 0 的污染修正）；bf16 行 ≤6.9e-4（口径修正）。**§11.3 的表格与结论已按 r3 数值更新**；§11.6 的"已确认事实"中与 step 0 数值有关的条目同样更新（1.938e-4 / 1.942e-4 / 1.871e-4）。

### 12.6 本轮测试与提交

* `tests/test_glt_galph_ph_diagnostic.py`：**5 passed**（输入不变性、调用顺序独立性、状态还原、残差/head 精度一致、mask 检查不污染输入）；日志 `logs/glt_galph_ph_retention_20260920/retention_tests_r3.log`（含命令与 EXIT=0）。
* 受影响的既有测试复跑：`tests/test_glt_galformer_ph_retention.py` **15 passed / EXIT=0**（r3 未改其代码，作回归确认），日志同一文件。
* Git：commit 与远端核验见 §9.5；推送后停止，等待 Codex 审查。

## 13. r4 结果与验收（单条 C1_REPAIR_5K 长程确认）

本轮只回答 r4 合同第八节限定的五件事：**5k 是否跑完且数值正确、PH 路径是否仍可训练、固定 probe 是否仍可分辨、在原门控下 PH 条件化是否可区分、strict-load 诊断是否一致**。**不回答** PH 保留是否提升属性预测（18 units 未执行，问题仍未回答）。

### 13.1 长程轨迹（5000 updates，`pretrain_C1_REPAIR_5K`）

行标 = 完成后 k 次 optimizer updates；step 0 行为首次更新前的初始态（其梯度为首步裁剪前/后梯度）。损失列为 rank0 局部窗口的 2D／3D／CL／PH。

| step | `tanh(α)` | \|patch.2.W\| | \|scale_fuse.0.W\| | \|ph_to_summary.W\| | 衰减项(前) | 任务梯度(前) | 任务梯度(后) | 本步更新量(ph_to_summary) | 残差/参考 | probe spread | real−const | 损失 2D / 3D / CL / PH | 有限 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.0000e-2 | 13.0952 | 13.0533 | 13.0399 | 0.0 | 2.622e-1 | 5.525e-3 | — | 1.488e-2 | 1.812e-1 | 1.117e-1 | 1.134 / 2.505 / 1.683 / 3.733e-3 | ✅ |
| 1 | 2.0000e-2 | 13.0952 | 13.0533 | 13.0399 | 0.0 | 2.622e-1 | 5.525e-3 | 5.019e-5 | 1.489e-2 | 1.812e-1 | 1.117e-1 | 4.608 / 10.159 / 6.723 / 1.483e-2 | ✅ |
| 2 | 2.0000e-2 | 13.0952 | 13.0533 | 13.0399 | 0.0 | 2.419e-1 | 5.077e-3 | 9.088e-5 | 1.485e-2 | 1.812e-1 | 1.117e-1 | 4.608 / 10.158 / 6.711 / 1.500e-2 | ✅ |
| 16 | 1.9989e-2 | 13.0951 | 13.0533 | 13.0399 | 0.0 | 1.010e-1 | 4.122e-3 | 6.237e-4 | 1.535e-2 | 1.814e-1 | 1.118e-1 | 4.399 / 10.050 / 6.276 / 1.482e-2 | ✅ |
| 64 | 1.9935e-2 | 13.0954 | 13.0534 | 13.0400 | 0.0 | 4.675e-2 | 2.634e-3 | 1.741e-3 | 8.228e-3 | 1.840e-1 | 1.134e-1 | 1.326 / 7.309 / 5.637 / 1.870e-2 | ✅ |
| 128 | 2.0410e-2 | 13.0960 | 13.0542 | 13.0411 | 0.0 | 4.867e-2 | 3.949e-3 | 1.576e-3 | 4.537e-3 | 1.883e-1 | 1.164e-1 | 0.888 / 2.530 / 4.601 / 1.256e-2 | ✅ |
| 256 | 2.0806e-2 | 13.0965 | 13.0550 | 13.0420 | 0.0 | 7.532e-2 | 1.185e-3 | 8.373e-4 | 5.421e-3 | 1.920e-1 | 1.187e-1 | 0.722 / 1.642 / 3.399 / 8.385e-3 | ✅ |
| 500 | 2.1203e-2 | 13.0973 | 13.0563 | 13.0429 | 0.0 | 1.494e-1 | 1.517e-3 | 1.786e-3 | 6.723e-3 | 2.008e-1 | 1.239e-1 | 0.608 / 1.250 / 2.098 / 7.060e-3 | ✅ |
| 1000 | 1.6740e-2 | 13.0997 | 13.0604 | 13.0403 | 0.0 | 5.502e-2 | 1.599e-3 | 5.476e-3 | 8.334e-3 | 2.162e-1 | 1.312e-1 | 0.407 / 0.854 / 0.763 / 3.902e-3 | ✅ |
| 2000 | 2.9385e-3 | 13.1119 | 13.0794 | 13.0361 | 0.0 | 2.405e-1 | 5.783e-3 | 3.805e-3 | 2.130e-3 | 2.982e-1 | 1.803e-1 | 0.237 / 0.174 / 0.292 / 3.322e-3 | ✅ |
| 3000 | 1.1493e-3 | 13.1215 | 13.0941 | 13.0380 | 0.0 | 1.919e-2 | 3.319e-3 | 3.715e-3 | 8.415e-4 | 3.750e-1 | 2.328e-1 | 0.156 / 0.080 / 0.073 / 2.919e-3 | ✅ |
| 4000 | 9.6756e-5 | 13.1277 | 13.1023 | 13.0344 | 0.0 | 3.315e-3 | 2.181e-3 | 6.763e-4 | 5.904e-5 | 4.801e-1 | 2.794e-1 | 0.1384 / 0.04871 / 0.01594 / 3.218e-3 | ✅ |
| 5000 | −2.2453e-4 | 13.1289 | 13.1044 | 13.0322 | 0.0 | 7.188e-4 | 5.212e-4 | 2.088e-3 | 1.262e-4 | 5.264e-1 | 2.919e-1 | 0.129 / 0.044 / 0.012 / 2.928e-3 | ✅ |

### 13.2 五项限定判读

1. **5k 完成与数值正确（通过）**：5000/5000 updates 一次跑完（81.3 min），`status=PASS`、`completed_steps=5000`；13 个监测点 `all_finite=true`，无 NaN/Inf、无异常抛出、无重试；损失单调下降（2D 1.134→0.129、3D 2.505→0.044、CL 1.683→0.012、PH 3.73e-3→2.93e-3，其间存在正常波动）。
2. **PH 可训练性（通过，仅指"未退化"）**：`weight_decay=0` 生效——**衰减项在全部 13 个点上恒为精确 0.0**；每个 PH 张量的裁剪前/后任务梯度在每一点都**非零**；每个监测区间都有**非零参数更新**（如 `ph_to_summary.weight` 5.0e-5 → 5.5e-3 → 2.1e-3）；范数不再衰减：`patch.2.W` 13.0952 → 13.1289（+0.26%）、`scale_fuse.0.W` 13.0533 → 13.1044（+0.39%）、`ph_to_summary.W` 13.0399 → 13.0322（−0.06%，属正常学习而非量级衰减）。对照旧 C1（§11.2）：同类范数在 4000 步内即从 13.0952 衰减到 1.009e-6（约 7 个数量级）、有效门控 ≈ −7.5e-8，本配方**没有重现该退化**。
3. **probe 输入区分性（通过）**：64 条固定 P_train probe 上，encoder 输出的跨样本 spread 0.1811 → **0.5264**（2.9× 上升），real−const 0.1117 → **0.2919**；只读诊断在 fp32 与 bf16 下都给出 `ph_summary_spread_resolvable=true`、`real_vs_const_resolvable=true`（5 个 resume 点全部成立），且 `real_vs_const_bit_identical=false`。**即：PH encoder 在 5k 后仍然对样本特异 PH 输入敏感，没有塌陷成常量。**
4. **原门控下的 PH 条件化（部分可区分；据实报告"模型自己把门关小"）**：`tanh(α)` 在前 500 步先升到 2.12e-2，此后**单调下降**，step 5000 为 **−2.25e-4**（α 可自由移动，本轮**未**加任何正则、**未**强制开门、**未**为开门加损失）；残差/参考相应从 1.488e-2 降到 **1.262e-4**。诊断在原生门控下：fp32 的 `g3_relative_change_const` 在 step 1000/2000/3000/5000 为 1.05e-4/3.30e-5/1.70e-5/4.02e-6，与行内阈值 3.2e-6/2.9e-6/2.8e-6/2.8e-6 相比 → `observable_at_tolerance = true/true/true/true`，而 step 0 与 step 4000 为 **false**（4.0e-6 与 1.73e-6，后者恰在阈值 2.71e-6 之下）。**判读**：到 5k 时 PH 对 `g3` 的影响已处在**本诊断 fp32 分辨率边缘**（残差仅占 summary 的 1.3e-4），能否判读为"条件化"取决于精度与容差；例外是 step 0（重建的 common-init，门控 `tanh(α)` 恰为 0，通路按构造恒等，7.70e-7 < 阈值 7.74e-6）与 step 4000（1.73e-6 < 阈值 2.71e-6）判为 `false`。bf16 下同量（4.5e-3）与地板（≈4.6e-2）不可分辨。**这属于模型自身学习到的门控行为，不是运行故障**；本轮按要求如实报告、不做干预。
5. **strict-load 一致性（通过）**：`deploy_05000.pt` 与 `resume_05000.pt` 共享 204 个张量**逐位相同**，strict 加载无缺失/多余键；encoder 输出在两种精度下逐位相同；模型级诊断量最大差 fp32 4.29e-9、bf16 2.13e-5，均远低于行内阈值 → 两个精度都 `model_within_resolution=true`；诊断的 step-0 构造与旧 R_LEGACY 实测起点一致。

### 13.3 与 256 步证据的关系、未验证事项与限制

* **一致性**：前 256 步与 `R_REPAIR_v2` 逐位相同、共享监测点整行相同（§9.6）→ 5k 结论是 256 步预检的**直接延长**，不构成新配方的独立重复。
* **r2 未验证事项 1（"5k 是否仍健康"）已由本轮回答**：在**该修复配方**（PH 路径 `wd=0` + `tanh(α)` 初值 0.02）与**新初始化长程训练**下，PH 路径 5k 不再退化（§13.2 第 2、3 条）。这只覆盖"可训练性与区分性"，**不等于** PH 保留提升下游属性预测。
* **r2 未验证事项 4（门控是否被回拉）**得到新的实测支持但**仍未构成机制结论**：本配方下门控确实自发收敛到 ≈0（|tanh α| ≤ 2.3e-4），残差占比同步降到 1.3e-4；本轮**未测**损失–门控曲线，也未做"冻结 α"或"加正则"的干预，因此**不把它归因于任何单一机制**（初始值、优化器归一化、任务对 PH 的无需求等均未被区分）。
* **仍未回答（不得据本轮声称）**：① PH 保留是否提升 2D/3D/CL 属性预测（18 units 未执行）；② 若下游用到 PH，门控自发关闭到 1e-4 量级时下游还能否获益；③ 本配方与旧 C1 的性能差异（两者是不同的初始化与训练路径，**不是**受控性能比较）；④ 早期（<1000 步）PH 路径的实际贡献。
* **限制**：本轨迹是**单条**轨迹、单一 seed（42），无重复，不构成统计证据；监测点 500+ 无对照双胞胎，其不可扰动性由代码路径推断；诊断分辨力（`resolution_threshold`）是**约定**而非已证的数值上界。
* **交付状态**：`单条C1修复配方长程确认完成，待Codex审查；PH retention 的 18-unit 性能问题仍未回答。` 执行端不宣布最终验收通过；下一步（是否解除阻断/是否授权 18 units）由 Codex 规划。

