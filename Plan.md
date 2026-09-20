# PH 下游保留与匹配融合验证计划

## 0. 交接头与授权

| 字段 | 内容 |
| --- | --- |
| 计划 ID | GLT-GALPH-PHRETENTION-20260920-01 / r2 |
| 日期／状态 | 2026-09-20；**阻断/返修中**：r2 的证据收口、最小工程返修、只读诊断与两条可训练性预检已完成；18 units 仍未执行，研究目标未完成、未获验收 |
| 用户要求 | ① 原始：同一 C1 主干下，下游**保留样本特异 PH**是否优于**关闭 PH**与**同容量固定 PH 分支**；② r2：收口 r1 的证据表述、做最小工程返修、在固定 64 条 P_train probe 上补只读诊断、执行两条匹配的 C1 可训练性预检（各 ≤256 updates） |
| 授权范围 | 本轮允许：§3.1–§3.4 四处最小返修与相关局部测试；复用已有 checkpoint 的固定小样本只读诊断；R_LEGACY / R_REPAIR 各 ≤256 optimizer updates（合计 ≤512）。原 18 units 继续暂停 |
| 明确禁止 | 18 development units；任何完整 5k 轨迹；N1 重跑；outer-test；XATTN；multiscale；额外 seed；正式性能评估；改科学定义、对照或预算；覆盖旧 checkpoint、sidecar、日志或失败现场 |
| 角色 | Codex 规划与独立审查；ZCode 执行实现、验证、诊断与预检并回填 §9；执行者不宣布 Codex 最终验收通过 |
| 基线 | dev@f6af067（pull 后 Already up to date，0/0；工作树仅有执行端上一轮遗留的未跟踪临时脚本，本轮返修前已删除）+ 本轮 commit（见 §9） |
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

### 9.2 r2 记录（本轮）

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
| 18 development units | **仍未执行** | 本轮未授权，未启动 |
| 汇总与门控 | 仍未执行 | 依赖上一行；且即使执行，也受 §11 的机制修正约束 |
| Git（r2） | 完成 | commit `d372b08`，分支 `dev`，已推送 `origin/dev`；fetch 后 `HEAD...origin/dev` = 0/0，`origin/dev` 指向 `d372b08` 且与本地树一致；推送经代理 + HTTP/1.1（直连报 `GnuTLS recv error`）。未 force push、未改写历史 |

### 9.3 证据溯源与限制

* **历史报告**（非本轮执行）：`p0_sidecar_build.log`、`p1/{updates_*,smoke_*}`、`blocker_evidence.json`、`preexisting_tests.log`、四臂 5k 产物 `results/glt_galph_20260920/**`。
* **本轮重跑**：§9.2 的测试与预检——命令、日志路径、退出码齐全（`preexisting_tests_r2.log` 首行记录命令、末行记录 EXIT=0；预检日志 `logs/glt_galph_ph_retention_20260920/p2/{chain_status.log,pretrain_R_*.log,diagnose_degeneration.log}`）。
* **独立审查**：本轮**没有**。以上全部为执行端自检；Codex 未检查的部分一律标为未核实。
* 数值口径：诊断同时记录 fp32 与生产 bf16 路径、各自的容差与"重复前向噪声地板"（fp32 ≈2.7e-7–7.6e-7 相对 |summary3|，bf16 ≈4.4e-3–1.1e-2）。**bf16 下任何小于该地板的差异都不作为信号**；本文件出现的所有"可观测/不可观测"判断都以此为界。
* 参数统计保留原始精度（科学计数法），未提前 round 为 0；`ph_encoder.norm.weight` 等个别张量仍为 O(1)，不因分支整体塌陷而一并归零。

### 9.4 运行环境与命令

* tmux session `Uni-Poly`，窗口 `galph_r2_pretrain`；预检脚本 `/tmp/r2_prechecks.sh`（内容与下列命令一致），启动方式与既有 smoke 相同：`python3 -m torch.distributed.run --standalone --nproc_per_node=4 scripts/pretrain_glt_galformer_ph.py --config configs/mts/glt_galph_c1.json --cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1 --cache-root data/processed/mips_trimer_scage --dual-static-root data/processed/glt_dual_v2/pi1m/dual_static_v1 --output <dir> --prep-workers 12 --log-every 100 --updates <N>`（R_REPAIR 另加 `--ph-no-weight-decay --alpha-ph-tanh-init 0.02`；两臂均加 `--ph-monitor <dir>/ph_monitor.jsonl --ph-monitor-steps 0 1 2 16 64 128 256 --ph-probe ... --ph-const-profile ...`）。
* 预检**未**保存 resume/deploy checkpoint（`save_every=1000`、`deployment_step=5000` 均未触发），未触碰 `results/glt_galph_20260920/**` 与 `p0/ph_sidecar_downstream/**`。
* 未执行诊断 backward（监测所需的 PH 梯度直接取自训练自身的 backward），因此不存在"额外 backward 污染下一步梯度/RNG/样本位置/scheduler"的风险来源。

## 10. 下一步

本轮交付后交 Codex 审查（§3 四处返修的实际 diff、§4 诊断与 §5/§6 预检证据、§9.2 的预算偏差）。审查通过前不启动任何扩展：

* 18 development units **继续暂停**；本轮没有产生任何 F_REAL/F_CONST 性能结论；
* 不自动重跑 N1/C1 的 5k，不自动进入 XATTN、multiscale、额外 seed 或正式评测，不自动重跑旧四臂；
* 若审查接受 R_REPAIR 作为修复配方，缺的是**新的单条 C1 长程确认授权**（5k 约 82 min）与随后的 18 units 授权。**256 步预检不构成"5k 不会再次退化"的证据，更不构成 PH 提升属性预测的证据**；其唯一结论是"该配方在 256 步内可训练、无衰减、输入敏感性不降"；
* 若审查选择选项 B（改科学问题：下游 PH encoder 新初始化并训练）或选项 C（归档为负结果），§11.3–§11.4 的证据可直接支撑，无需新增运行。

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

### 11.3 只读诊断（§4，`scripts/diagnose_glt_galph_ph_degeneration.py`）

* **probe 集**：P_train sidecar 的**行 0–63**（冻结文件里固定行序、全部 valid），键值写入 `p1/ph_probe_keys.json`，张量写入 `p1/ph_probe_profiles.npy`；**不涉及任何下游 validation/test 指标**，样本选择只看行序。
* **step 0 溯源**：旧 C1 每 1000 步存一次，没有 step-0 checkpoint。step 0 由 common-init artifact（sha256 `637827e6…`）按同一构造重建，并与 **R_LEGACY 实运行的 step-0 监测行逐项相等**（`all_equal=true`：14 个 PH 张量范数 + probe spread + real-vs-const 全部相等）；该构造又被证明可复现旧 C1 前 256 步（§9.2）。因此 step 0 视为**可证明来源**，不再标"未核实"。
* **噪声地板（决定"能否算信号"）**：同一输入重复前向的相对差地板 fp32 ≈ **2.7e-7–7.6e-7**、bf16 ≈ **4.4e-3–1.1e-2**（相对 |summary3|）。判定用 `resolution_floor = max(容差, 10×地板)`。

| step | 精度 | 跨样本 p_ph spread | real vs const | 残差/参考（本 checkpoint 门控） | Δg3（PH→const, 强制门控 0.02） | 可观测 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | fp32 | 1.812e-1 | 1.117e-1 | 0.0（α=0） | 1.912e-4（地板 7.7e-6） | ✅ 可观测 |
| 0 | bf16 | 1.812e-1 | 1.117e-1 | 0.0 | 1.090e-2（地板 1.08e-1） | ❌ 地板之下 |
| 1000 | fp32 | 6.243e-3 | 4.146e-3 | 1.494e-5 | 1.334e-6（地板 3.2e-6） | ❌ |
| 1000 | bf16 | 1.056e-2 | 1.021e-2 | 1.496e-5 | 5.121e-3（地板 5.1e-2） | ❌ |
| 2000 | fp32 | 2.384e-7 | 2.384e-7 | 4.213e-5 | 2.739e-7 | ❌ |
| 2000 | bf16 | 0.0（逐位相同） | 0.0 | 4.215e-5 | 4.561e-3 | ❌ |
| 3000 | fp32 | 0.0 | 0.0 | 6.262e-6 | 2.762e-7 | ❌ |
| 3000 | bf16 | 0.0 | 0.0 | 6.263e-6 | 4.578e-3 | ❌ |
| 4000 | fp32 | 0.0 | 0.0 | 1.895e-8 | 2.678e-7 | ❌ |
| 5000 | fp32 | 0.0 | 0.0 | 6.784e-13 | 2.725e-7 | ❌ |
| 5000 | bf16 | 0.0 | 0.0 | 6.775e-13 | 4.541e-3 | ❌ |

* **架构可承载、训练结果不可承载**：step 0 在强制门控 0.02 下 fp32 可观测（res/ref 1.48e-2；把 PH 换成 const 后 Δg3 = 1.912e-4、Δ`ph_head` = 1.920e-4、Δ`cl_proj3` = 1.862e-4，均远高于地板 7.7e-6），说明该残差通路本身能把 PH 输入传进 g3 与读 g3 的两个头；而 step 1000–5000 的同类差值（Δg3 ≤1.33e-6、Δ`ph_head` ≤1.0e-6、Δ`cl_proj3` ≤8.1e-7）全部落在地板之内。（`head_2d`/`head_3d` 读 atom/bond states，不消费 g3，本就不受 PH 输入影响。）
* **bf16 不适合作这类比较**：其重复前向地板（4.4e-3）比全部被测效应都大；本轮的"不可观测"结论在 fp32 下同样成立（step 2000 起 spread < 1e-6），不依赖 bf16 的结论。
* **mask 无目标泄漏**：把被 mask 的 patch 内容替换为任意值，encoder 输出逐位不变 —— 在 5 个 checkpoint × 2 种精度下全部 `no_observable_leakage=true`。

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
3. 固定 64 条 P_train probe 上：step 1000 仍可分辨（6.24e-3）、step 2000 低于容差、step ≥3000 在 fp32/bf16 下逐位输入无关。
4. step 0（重建并与实运行 step-0 逐项相等）在强制门控 0.02、fp32 下可观测（Δg3 1.9e-4 > 地板 7.7e-6）——通路本身能用，训练后的 checkpoint 不能用。
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

### 11.7 修复选项现状（仍待 Codex/用户决定）

* **A（保持"同一 C1 主干 + 冻结 encoder"合同）**：本轮两臂预检给出了 A 的**最小充分集合证据**——PH 路径必须排除耦合 weight decay，且门控不能零初始化。但 A 仍需要**新的 C1 5k 预训练授权**（§10）。
* **B（改变科学问题）**、**C（归档为负结果）**：本轮证据同样可用；执行端不自行选择。

