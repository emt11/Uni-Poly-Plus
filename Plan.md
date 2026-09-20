# PH 下游保留与匹配融合验证计划

## 0. 交接头与授权

| 字段 | 内容 |
| --- | --- |
| 计划 ID | GLT-GALPH-PHRETENTION-20260920-01 / r1 |
| 日期／状态 | 2026-09-20；实现与有界验证完成，**18 units 因方法学阻断未执行**（见 §11），待 Codex 审查 |
| 用户要求 | 检验"同一个 C1 预训练主干，在下游保留样本特异 PH，是否优于关闭 PH，以及同容量的固定 PH 条件分支" |
| 授权范围 | 必要实现、局部测试、有界 smoke、验收通过后的 18 个 development units |
| 明确禁止 | 新增预训练、outer-test、完整八任务五折、PH-XATTN、多尺度结构实验、额外 seed、train+validation refit、为 best_epoch=30 的组单独延长预算 |
| 角色 | Codex 规划与独立审查；ZCode 执行实现、验证、18 units 与回填记录；执行者不宣布 Codex 最终验收通过 |
| 基线 | dev@f991f84（`git status` 干净；pull 后 Already up to date，0/0 分叉） |
| 已有改动 | 无未提交改动；用户未跟踪文件保留不动、不暂存 |
| 产物根目录 | `results/glt_galph_ph_retention_20260920/`、`logs/glt_galph_ph_retention_20260920/` |

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

| 项目 | 状态 | 证据／限制 |
| --- | --- | --- |
| 执行前核对与 pull | 完成 | `git status` 干净；`pull --ff-only origin dev` 经代理 + HTTP/1.1 成功（Already up to date，0/0）；直连 HTTP/2 曾报 framing 错误 |
| CANON3D 归档 | 完成 | `PROJECT_HISTORY.md` 新增条目：被替换、未验收、未完成事项逐项列出，r1/r2 文本定位到 `b61876d`/`a231c05` |
| 本轮合同 | 完成 | 本文件 §0–§8 |
| 交接更正 | 完成 | §0.1 五项；`logs/glt_galph_ph_retention_20260920/preexisting_tests.log` = **56 passed, 3 warnings in 66.43s**（2026-09-20 12:23 UTC 重跑，替代此前未落盘的报告） |
| 下游 PH sidecar (P0) | 完成 | `results/glt_galph_ph_retention_20260920/p0/ph_sidecar_downstream/`：809 结构、809 valid / 0 invalid、跨任务按 32-byte key 去重复用 395 行（xc 432 新 / eps 15 复用 + 367 新 / eat 380 复用 + 10 新）、构建 12.9 s；F_CONST 均值来自 P_train sidecar 的 911,391 条 valid profile |
| 实现 | 完成 | `src/dataset/glt_ph_downstream.py`（sidecar 读取/数据集/collate）、`src/modules/glt_galformer_ph_retention.py`（`GalformerPHDownstream`）、`src/modules/glt_galformer_ph_downstream.py` 新增 `adjust_r3` 钩子（默认恒等，不改原路径）、`scripts/build_glt_ph_downstream_sidecar.py`、`scripts/finetune_glt_galformer_ph_retention.py`、`scripts/aggregate_glt_galph_ph_retention.py`、`scripts/diagnose_glt_galph_ph_retention_blocker.py` |
| 最小验证 (11 项) | 完成 | `tests/test_glt_galformer_ph_retention.py` **12 passed**（shape/collate、NUL key + 越界位置、冻结与 eval、invalid 零残差、F_OFF parity（预测/公共梯度/一次更新逐位一致）、三组初始张量一致、非零门控下换 profile 改变预测、F_CONST 与 key 无关且与 F_REAL 同 mask、checkpoint 往返、scaler 只拟合 train、runner 无 outer-test 路径） |
| 有界模型验证 | 完成 | 三组各 2 updates + 三组各 XC/fold0 2-epoch smoke，全部 exit=0；证据 `results/glt_galph_ph_retention_20260920/p1/{updates_*,smoke_*}`。结果：F_OFF 的 `gamma_grad=0`；F_CONST/F_REAL step1 `gamma_grad=6.56e-4`、`ph_proj_grad=0`（零门控），step2 `ph_proj_grad=8.94e-8`、`tanh(gamma)=5.88e-6`（门控可训练）；三组 trainable=40,992,145、冻结集合一致、coverage 345/345 valid |
| 18 development units | **未执行（阻断）** | F_CONST 与 F_REAL 在冻结 C1 PH encoder 下逐位相同（见 §11），运行无法回答预登记问题 |
| 汇总与门控 | 未执行 | 依赖上一行 |
| Git | 待回填 | 本轮 commit 与推送结果见交付说明 |

## 10. 下一步

本轮交付后交 Codex 审查（§7 判据、实现边界、产物与 commit 链，以及 §11 阻断的处置）。审查通过前不启动任何扩展；18 units 未执行，本轮不产生任何 F_REAL/F_CONST 结论。

## 11. 方法学阻断（执行端只读定位，2026-09-20；不得由执行端自行处置）

**结论：本轮预登记的 F_REAL vs F_CONST 对比无法用 C1 的冻结 PH encoder 测量，因为该 encoder 在预训练结束时已是输入无关的常量函数。**

只读证据（可复现：`scripts/diagnose_glt_galph_ph_retention_blocker.py`，输出 `results/glt_galph_ph_retention_20260920/p1/blocker_evidence.json`）：

1. **权重被摧毁**：C1/N1 `deploy_05000.pt` 中 `ph_encoder.patch.2.weight` norm = 4.8e-11、`ph_encoder.scale_fuse.0.weight` = 1.1e-16、`ph_encoder.scale_embedding` = 2.3e-14（初始化分别约 13、3.6、1.27）；`alpha_ph ≈ -7.5e-08`（`tanh(alpha_ph) ≈ 0`）。
2. **机制复现**：`torch.optim.Adam(lr=2e-4, weight_decay=1e-6)` 在**零损失梯度**参数上 5000 步把 norm 1.2714 压到 0.0（max|p| ≈ 4.6e-24）；同条件 AdamW（decoupled）保持 1.2714。PH encoder 的唯一损失路径经过 `tanh(alpha_ph)`，而 `alpha_ph` 全程停留在 ±1.5e-4 内 → 梯度≈0，衰减项主导。
3. **时间线**（C1 resume 检查点）：step 1000 尚健康（patch.2 = 6.26、scale_fuse = 3.56、scale_embedding = 0.35、ph_to_summary = 4.89），step 2000 → 0.377/0.018/0.0047/0.548，step 3000 → 1.8e-3/1.8e-6/7.0e-6/0.187，step 4000 → 1.0e-6/0/1.5e-9/0.0117，step 5000 → 0/0/0/4.3e-4。
4. **函数级**：把 64 个真实下游 profile 与 P_train 均值 profile 送入冻结 encoder，`p_ph` 逐元素位级相同（跨样本 spread = 0.0；`real_vs_const_bit_identical = true`）。
5. **训练级**：P1 有界验证中 F_CONST 与 F_REAL 的 loss、`gamma_grad`、`ph_proj` 梯度、`residual_relative_norm`、`tanh(gamma)` 全部位级相同。
6. **分支级**：两臂 `alpha_ph ≈ 0`，PH 残差在预训练中从未有效进入 graph summary。

**对上一轮四臂结论的更正（不改数值，只改机制标签）**：N1/C1 的 PH *输入*注入路径在预训练中失效；`LPH` 从 0.0196 降到 0.0029 反映的是 `ph_head` 从 3D summary 预测 masked Betti patch（trunk 侧辅助任务），不是 PH encoder 承载信息。`N1−N0 = −0.0063`、`C1−C0 = +0.0051` 应读作"masked-PH 预测辅助任务（C 臂另含 CLS 组合）对 trunk 的影响"，不能读作"PH 条件化输入的价值"。四臂数值本身不变。

**需要授权的修复选项（执行端不得自行选择）**：
* **A（保持"同一 C1 主干 + 冻结 encoder"合同）**：修预训练使 PH 分支真正可训练——PH 参数排除 weight decay（或改 AdamW decoupled）、并让 `alpha_ph` 以非零值/正向正则参与，随后重跑 N1/C1 5k，再执行本轮 18 units。属于**新增预训练**，须新授权。
* **B（改变科学问题）**：retention 的下游 PH encoder 改为新初始化并参与下游训练，此时 F_CONST/F_REAL 才有信号，但问题变为"下游自训练 PH 编码的价值"，与预登记问题不同，须重新登记判据与预算。
* **C（归档为负结果）**：停止 PH 输入方向，保留证据 1–6。

执行端建议：不重启预训练则本轮问题不可判定；若要保留"冻结 encoder"这一合同，A 是唯一可行路径。

