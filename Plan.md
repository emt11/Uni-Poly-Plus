# GLT-V2：结果闭环、几何失稳诊断与优化决策

> 2026-09-17 当前状态核对（CACHE-20260916-01/r3 收口后）：GLT-V2 B.3 replay 已完成；几何
> collapse 已复现并完成机制诊断；geonorm P2 validation、fixed geonorm 5k 与 deploy validation
> 已完成。正式 downstream 当前可用口径为 7 个 task／35 folds；egc 完整 5-fold 尚未完成。缓存
> r2 经 Codex 审查为 `NEEDS_REPAIR`，r3 已经 Codex 独立验收通过并关闭（`CLOSED`）。缓存专项不再
> 占据下一执行入口。以下历史段落保留原始执行语义，后续状态以本注记和对应周期执行记录为准。

> 2026-09-16 缓存专项交接更新（Codex，`CACHE-20260916-01/r2`）：状态“需返修”。ZCode 已交付 `7f476b4`，Codex 静态审查未通过；当前下一步见 [Plan_Cache.md 第 10 节](Plan_Cache.md#10-r2-下一步仅返修可靠性与证据缺口)。按“六项构建恢复／冻结修复 → 原 32 条目标及 clean/noisy parity → 化学和 benchmark 口径修正 → 复审”推进，不重新运行完整 A–C。容量 64 继续不采用，不启动长 benchmark、角度优化、阶段 D、全量重建或训练。原已获授权范围不重复申请，本轮用户只要求计划落盘，Codex 未接管执行。r1 原交接与执行材料保留在 Git／历史记录及下方；其他科学周期状态不由本次更新裁定。

> 2026-09-16 清理交接更新（Codex，`CLEANUP-20260916-01/r3`）：首批 12 个精确目标的缓存清理及第 7 节 A 的只读证据补记已完成；代码精简未实施，其余候选 HOLD。现有验收日志未保存完整 argv／env；deploy 检查未传 static/targets 且使用 `no_grad`，因此 backward、删除前后预测 parity 和清理后 static/targets 重新消费均未核实。54 passed 与两份有限 forward 报告均为既有证据，不是本轮新增模型结果。原 09:11 UTC 环境适配及“删除待授权”注记保存在 `PROJECT_HISTORY.md`；本次只同步清理交接，不改变下方科学计划的授权、预算或科学完成状态。

## 计划头

|项目|当前记录|
|-|-|
|计划 ID|GLTV2-20260916-01|
|修订|r2：阶段性审查后续执行；修复诊断入口，再执行 B.2／B.3，不增加原回放预算|
|状态|待审查（A、B.2、B.3 已完成；缓存专项 CACHE-20260916-01/r3 已 CLOSED，不占据下一执行入口）|
|授权来源|用户已选择“诊断并有限回放”，并说明将计划交给 ZCode／其他模型执行；随后明确报告计划正在执行|
|规划／审查|Codex|
|执行|ZCode／用户指定执行者；实际执行者在下方补记|
|基线|本地 HEAD `ff00919`；2026-09-16 本次只读核对远端已为 `8f625e9`、工作树干净。保留远端新增实现，不用本地旧代码覆盖|
|本次交接录入|2026-09-16；此前本地 `Plan.md` 为空。未查询远端运行进度，未启动或重启任何任务|

**续接说明：本计划已在执行，不因文件初始化而从头启动。** ZCode 先填入已执行阶段、活动进程、输出目录和证据，再继续剩余工作。如执行端已有更新后的计划或代码，先核对差异，不用本地录入内容覆盖实际记录。

**r2 交接优先级：** 下方原实施定义继续有效；本次按第四节的返修与续执行顺序工作。A 不重跑，B.2 不能运行未经修复的现有脚本。B.2 与 B.3 已获授权，不再等待重复确认。本地本文件为当前交接版本，远端 Plan.md 经本次检查仍为空；交接时仅同步该计划内容，不以同步为由覆盖代码或启动第二个执行者。

## 一、目标与固定边界

回答以下问题：现有 Concat／KFuse 结果能否完整复算；Concat 约 2660 步的几何失稳发生于哪里；下一次性能实验应优先改变哪一个因素。

已知参考：Concat macro8 R² `0.7877379364`，KFuse `0.7695629053`；Concat 几何 loss 在约 2676 步达到 `66.996841`，随后停在约 `0.31`。这些属于既有运行证据，不代表本轮已经复现。

* 不重跑原有 80 个 fold，不启动新的完整预训练、微调、seed sweep 或缓存重建。
* 不覆盖历史模型、预测、缓存或 metadata。
* 允许必要代码修改、相关局部测试、固定小批量前后向诊断，以及一次最多 800 optimizer updates 的 Concat 回放。
* 诊断期间不改架构、训练目标、LR、损失权重、数据顺序或采样定义。
* GPU、worker 和长任务使用远端 `tmux Uni-Poly` 独立 window，执行前检查已有任务；保存实际命令、环境、日志和输出路径。

## 二、实施步骤与验收

### A. 汇总接口与来源核对

1. 正式微调写出的 metrics 增加 `protocol="outer5_inner20"`。聚合器新增必需的 `--split-root`、`--raw-root`，以及默认关闭的 `--allow-legacy-missing-protocol`。
2. 本次旧产物缺 protocol 时，只在 run.json 确认正式 shard、task/fold 一致且预测身份匹配 manifest 后兼容，并记录来源；已有错误 protocol 不得兼容。
3. 从预测重算 R²／MAE／RMSE。指标比较 `rtol=1e-6, atol=1e-8`；原始标签比较 `rtol=1e-6, atol=1e-5`。核对整数唯一 row index、每折 test 集合、train/validation/test 互斥及五折一次完整覆盖。
4. 写出逐折指标、五折 mean/std（`ddof=1`）、排序 OOF 和 pooled OOF；明确两类 R² 不同。全部输入校验后写入新 `comparison_review_<UTC时间戳>` 目录，不改旧指标。宏平均须复现上述参考。
5. 只读核对 cohort、bundle、static 与 targets 绑定。解释实际 959,588 条接受记录的筛选链、`first_valid` 单构象／最多 8 候选语义、下游 3,655 个结构中 9 个 fallback。
6. 定位四条 `trimer_ru_internal_bond_contract` 的来源与分类；不据此推断整个接受集无效。核对构建 dirty 代码是否可还原，不修改历史 provenance。
7. 关联当前 bundle 的 Stereo 验收证据；若缺失，最多核对已知异常身份及一条明确 E/Z 记录，不生成坐标、不自动扩为全量审计。几何候选耗尽与身份／内部键错误分开报告。

### B. 最小诊断接口

预训练包装器增加默认关闭的可选诊断输出，不改变原 loss、梯度或 checkpoint 参数结构：

* 分开记录 chem、length、angle、FP 及各自有效计数。length + angle 必须按原逐样本归约口径还原 geo loss，另记无角度样本数。
* 记录目标与预测的均值、标准差、分位数和极值，tanh 前输出、tanh 导数和精确输出 ±1 的比例。
* 记录 3D 各层 hidden RMS、中心 pooled RMS、Gaussian 有效标准差最小值／affine 参数／输出尺度，以及 clipping 前总梯度和各模块梯度范数。
* 记录 step、rank、抽样位置；异常 batch 保留样本 key 与逐样本误差。
* 统计从已有 forward detach，不额外消耗训练随机数；不将预测头之后 `.float()` 描述为整个头按 FP32 计算。

### C. 固定小批量 checkpoint 诊断

新增 `scripts/diagnose_glt_dual_pretrain.py`，读取完整 resume checkpoint 与任务头：

* Concat／KFuse 各比较 2k、3k、5k，共六份 checkpoint。
* 从原抽样流 2k 后的位置确定同一组 16 条真实记录，每批 8 条；保存 key、position 和 seed，共用相同 mask、扰动与 target。
* 每份分别作 FP32／BF16 eval-mode 前向，不执行 optimizer update。
* 每份对第一批 8 条，以 FP32 分别计算 length、angle、FP 对 3D encoder 的梯度范数及两两 cosine；无依赖参数按零贡献处理。
* 计算直接用受扰动键长／角度预测干净值的误差参考。顺序处理模型，不为吞吐扩大样本或常驻多个 checkpoint。

### D. 一次有限恢复回放

在现有预训练入口增加运行参数 `--diagnostics`、`--stop-after-step 2800`、`--diagnostic-save-steps 2600 2660 2700 2800`。参数只控制观测／保存／停止，不进入科学配置或放宽恢复检查。

* 从 Concat `resume_02000.pt` 恢复，最多到 2800，总预算最多 800 updates；使用独立输出目录。
* 原四卡、microbatch=84、accumulation=3、global batch=1008、BF16；恢复 model、optimizer、scheduler step、各 rank RNG 与抽样位置。
* 原配置预算保持 5000，调度保持 20k cosine 前缀；不能把 2800 当作新调度终点。不生成正式部署包。
* 每步记录分项 loss 和梯度；每 20 步记录表示／Gaussian 统计，2600–2720 每步记录。
* 初始 10 步比较原日志 LR、目标数和 loss 轨迹。身份／目标数不一致立即停止；CUDA 小数值差异可记录，明显偏离先定位，不冒称精确复现。
* 保存诊断 checkpoint 不改变随机序列。NaN/Inf、writer 冲突或缓存身份错误时停止受影响运行。
* 到 2800 未复现则报告未复现，不自动延长或开启第二次回放。峰值超过角度误差上界，不能预设全部责任在 tanh。

### E. 优化决策与局部验证

根据证据只选择一个首要后续改动，形成建议，**本轮不自动启动优化训练**：

|证据|优先候选|
|-|-|
|同输入 BF16 异常、FP32 正常|仅受影响几何头／loss 改 FP32|
|hidden／预测头尺度增长|仅几何头输入增加 LayerNorm|
|Gaussian 宽度／输出尖峰|只调整相关参数化或优化设置|
|FP 与几何持续梯度冲突且 FP 主导|单独降低 FP 对共享 3D 的训练影响|
|数据／索引错误|修复数据链，不靠降低 LR 掩盖|
|根因未定位|报告剩余假设和最小补充诊断，不宣布已修复|

后续优先级：稳定训练 → matched 2D-only 对照 → 必要时增加二面角／非键距离信息 → 有证据再改融合。matched 对照保留当前 O8、bond-path、cohort、mask、化学／指纹任务、步数、调度和五折，保留对应拼接接口宽度并报告参数／计算量差异。暂停机械追加单输入 KFuse；XC 先检查 train/validation 残差与标签分布，不用已看过的 test 反复选参数。

只执行聚合协议／身份／公式、诊断关闭与开启一致性、分项还原、恢复位置／RNG／LR 和诊断保存不改变轨迹等相关局部测试；不扩大为全仓或完整正式实验。

## 三、执行记录（ZCode 维护）

### r1 阶段性交付（2026-09-16）

* 执行者报告：A.1 汇总修复与 9 项测试通过；A.2 完成筛选链、四条芳香感知契约失败、dirty 来源和已有 Stereo 审计核对；B.1 已实现可选诊断并做临时等价性检查。B.2 未跑，B.3 参数未实现、回放未启动，C 未完成。
* Codex 只读确认：远端 `8f625e9` 含相关实现，工作树干净，已不是报告中的“未提交”；两份新 summary 存在，macro8 分别为 `0.7877379364475444`／`0.7695629052761048`，pooled macro 分别为 `0.7931866117573143`／`0.7758626655507175`，旧协议兼容被明确记录。
* 汇总产物：`results/glt_dual_static_finetune_formal_grid/{concat,kfuse}/comparison_review_20260916T001828Z/`。不覆盖、不重新聚合，除非后续发现与汇总相关的新错误。
* 执行者报告的 9 passed、B.1 逐参数梯度一致、全量 Stereo 审计和四条契约错误根因，本次未重新运行或完整独立核验；ZCode 补充已有日志／命令／审计文件与对应 bundle 身份即可，不要求重复全量执行。
* 有关 dirty 的限定：限定六个 modified 文件的 diff hash 一致，可证明这些 diff 一致；要声称完整工作树／三阶段代码相同，还须说明当时新增文件的来源及内容证据。缺失则限定结论，不阻断与之无关的诊断，不重新构建缓存。
* 当前回放已消耗预算：执行者报告为 0 updates；启动前从任务／输出再次确认，累计上限仍为 800。

### ZCode 续执行记录

在此追加实际修订、测试与命令、tmux window、日志／输出路径、回放开始／停止 step 和累计 updates。不要删除上面的阶段性交付；每阶段结束或阻断时更新。

### 2026-09-16 B.2／B.3 续执行（Codex，执行中）

* 基线核对：当前 HEAD 为 `ea2fa38a3572012303ba1b06ad8eb02248aeb510`，工作树在启动前干净；未重跑 A、未启动新的完整预训练／微调或缓存构建。`Uni-Poly:full_rebuild2` 为已有的非本计划缓存任务，当前 pane 已回到 shell，未接管或重启。
* 局部验证：首次 `PYTHONPATH=.` 收集阶段因测试辅助模块未在路径中而导入失败；按仓库实际约定改为 `PYTHONPATH=.:tests` 后，`tests/test_dual_glt_pretrain.py tests/test_aggregate_glt_dual_finetune.py` 为 `20 passed, 1 warning`，日志为 `logs/glt_v2_dual_diagnostic_tests_20260916_retry.log`。本次导入失败未进入用例，未作为代码回归失败。
* B.2 已完成：在 `Uni-Poly:glt_v2_diag_b2`、GPU0 运行六份真实 `resume_{2000,3000,5000}.pt`（Concat/KFuse），固定原始抽样流 step=2000 后 16 条 PI1M 记录、两批各 8 条，FP32/BF16 eval，无 optimizer update。报告 `results/glt_v2_diag_b2_20260916/fixed_batch.json`，日志 `logs/glt_v2_diag_b2_20260916/diagnose.log`，命令记录 `logs/glt_v2_diag_b2_20260916/command.txt`，exit code 0、`status=PASS`、六份均 `OK`、无非有限值。观测到 Concat step=3000 角度头 near-saturation 约 `0.7636`、exact ±1 约 `0.1576`，geometry 分项约 `2.7`；这支持执行原计划 B.3，但不单独证明训练根因。
* B.3 已启动：`Uni-Poly:glt_v2_diag_b3`，4 GPU、从 `results/glt_dual_static_pretrain_5k_concat/resume_02000.pt` 恢复，原配置／world=4／microbatch=84／global batch=1008／BF16，`--diagnostics --stop-after-step 2800 --diagnostic-save-steps 2600 2660 2700 2800`，未生成 deploy。输出 `results/glt_v2_diag_b3_concat_replay_20260916`，日志 `logs/glt_v2_diag_b3_concat_replay_20260916/replay.log`，命令记录 `logs/glt_v2_diag_b3_concat_replay_20260916/command.txt`；截至 02:59:50 UTC 已正常运行至 step 2109，前 10 步未见 reference mismatch，累计 optimizer updates=109/最多 800。本回放未完成前不启动任何第二次回放或优化训练。

### 2026-09-16 ZCode 续执行：B.3 结果核验、失稳机制分析与 C 决策草案

**预检与边界**

* 交接时 HEAD 已从交接文档记录的 `ea2fa38` 前进到 `d417496`；两者之间的 `736285f`、`d417496` 只改 `Plan.md`。工作树除 3 个未跟踪文件外干净，其中 `scripts/diag_initial_trigger.py`、`scripts/diag_spike_attribution.py` 属另一执行者，本执行者未修改、未运行、未提交；`tests/test_glt_dual_diagnostics.py` 为本执行者新增。
* 核对 §3 记录后发现 **B.3 已由另一执行者执行完毕**（与交接文档的 “B3_REPLAY = NOT_STARTED” 不符）：输出 `results/glt_v2_diag_b3_concat_replay_20260916`，日志 `logs/glt_v2_diag_b3_concat_replay_20260916/replay.log`（末行 `EXIT_CODE=0`），命令记录为同目录 `command.txt`，四个诊断 checkpoint（2600/2660/2700/2800）与 `diagnostics_steps.jsonl`（800 行，step 2001–2800）齐全。**据此未启动任何第二次回放。**
* 累计预算：2000→2800 = **800/800 updates，已耗尽**。本轮未新增训练、优化运行、缓存重建或 fold 重跑。

**B.3 核验：回放逐位复现原轨迹**

* 与 `logs/dual_static_pretrain_concat5k.log` 按 step 对齐。该日志存在记录拼接，改用全文正则提取并按 step 去重（同一 step 各 rank 打印相同的全局值）：800/800 个共同 step 的 chem/geo/FP 三项 `max|replay-ref| = 0`。
* 失稳峰值**精确重现**：`step 2676` 的全局 geometry loss `66.996841`。说明恢复身份、RNG、数据顺序与调度状态一致，且 `--diagnostics` 观测不扰动训练轨迹（满足 §4.2 的“开关不改变轨迹”要求，B.3 层面已实证）。

**失稳轨迹（窗口按 step 定义；rank 去重取同 step 首次出现的全局值）**

* concat：基线 `2401–2600` 中位 `0.00092`；首个 `geo>0.1` 在 `2662`；峰值 `66.9968@2676`；`2700–2800` 中位 `0.4156`；`4001–5000` 中位 `0.3139`（基线约 342×），末值 `0.3108` —— **截至 5000 未恢复**（`2700` 之后 `geo>0.1` 的步数为 2301/2301）。
* chem / FP：chem `0.1854 → 0.1907@2676 → 0.1617`（无持久退化）；FP `0.0434 → 0.1987@2676（瞬态 4.6×）→ 0.0418`（瞬态抬升，无持久偏移）。
* kfuse 对照：基线 `0.00107`，首个 `geo>0.1` 在 `3274`，峰值 `6.0736@3406`，末值 `0.00106` —— 完全恢复。

**机制（dense 窗口 2600–2720 的 rank0 局部统计）**

| 量 | 2650 | 2666 | 2668 | 2676 | 2680 → 2720 |
|-|-|-|-|-|-|
| angle head pre-tanh 均值 | -0.51 | +1.01 | -4.07 | +3.73 | -13.7 → -29.2 |
| tanh 导数 均值/最小 | 0.78/0.007 | 0.42/0.30 | **0.0012/0.0006** | 0.0025 | **0.0000/0.0000** |
| 预测 exact ±1 比例 | 0 | 0 | 0 | 0 | **1.0000** |
| graph_3d RMS | 0.82 | 2.07 | 2.25 | 3.05 | 4.12 → 9.17 |
| length_head 梯度范数 | 0.037 | 0.67 | 0.95 | 0.76 | 0.96–1.00 |
| angle_head 梯度范数 | 0.039 | 0.70 | 0.0003 | 0.0001 | 0.0000 |

* 尖峰以 length 为主：`2676` 分项 sum 为 `length 5441.5 / angle 176.4`（length 占 96.9%）；`2669` 92.3%；`2710` 93.5%。**length 尖峰与 angle 平台同时存在**：angle 饱和后成为约 `26.7`（基线 0.21）的常数残差。
* 距离 Gaussian `σ_min` 全程恒为 `0.0172`（六 checkpoint 一致）→ 排除 Gaussian 宽度分支；BF16 已由 B.2 排除。
* 判读链条：3D 表示尺度增长（0.85→9.2）→ angle head pre-tanh 失控 → tanh 完全饱和、梯度恒 0 → angle 项无法自我修正 → length 项独吞梯度并产生尖峰；尖峰步 clip 前总梯度达 `1063`（基线约 0.48），存在经由 `clip_grad_norm_(1.0)` 的反馈放大。
* **口径限制**：`_module_grad_norms` 将 `p.grad is None` 记为 `0.0`，因此上表 angle_head 的 `0.0000` 无法区分“梯度为零”与“该步未进入反向图”。

**测试**

* 新增 `tests/test_glt_dual_diagnostics.py`（7 项，`PYTHONPATH=.:tests`）：诊断开关不改变 loss/sums/counts/targets、逐参数梯度、`state_dict` 与 **RNG 消耗**；`length+angle` 还原 geometry 且无角度样本计数一致；`component_tensors` 携带梯度；`_module_grad_norms` 对缺失梯度按零对齐且不丢键；Gaussian 统计取自真实 forward 并与模块参数一致；angle head 统计有限、tanh 导数落在 [0,1]。
* `tests/test_dual_glt_pretrain.py` + `tests/test_aggregate_glt_dual_finetune.py` + 新文件 = **27 passed**。
* 交接文档 §6 所述 2 个既有失败在当前 HEAD 上均已通过（`test_empty_geometry_and_batch_target_offsets` 1 passed；`tests/test_dual_glt_audit.py` 47 passed），该条已过期。

**未完成与限制**

* 诊断未记录异常 batch 的样本 key 与逐样本误差（计划 §B 该项未实现），**无法从现有证据判断尖峰是否集中在同类样本**；另一执行者的在途脚本 `scripts/diag_spike_attribution.py`（无 optimizer step）正处理该问题，本执行者未运行、结论不依赖它。
* 未执行：第二次回放、matched 2D-only、任何优化训练、缓存重建、80 fold 重跑。

**C 决策草案（三层次）**

* **已确认事实**（B.3 逐 step 可复现）：失稳自 `2662` 起、峰值 `66.9968@2676`、`5000` 未恢复；尖峰以 length 项为主；angle head 在 `2668` 后 tanh 饱和、exact ±1 比例达 `1.0`、其梯度归零；3D 表示 RMS 单调增长约 11×；距离 Gaussian σ_min 恒定；FP 仅瞬态抬升。
* **支持性证据**（相关，非因果证明）：表示尺度增长先于饱和（`2650` 0.82 → `2660` 1.01 → `2666` 2.07 → `2668` 饱和）；经裁剪的反馈放大；kfuse 同类事件可自行恢复提示与 concat 的差异在“是否进入饱和不动点”。
* **未证实假设**：`~2655–2660` 触发尺度增长的初始事件（特定 batch、累积漂移或裁剪反馈）未定位；LayerNorm 能否阻止该链条未验证；angle 饱和是主因还是尺度增长的伴生结果未分离。
* **首要改动建议（本轮不实施）**：仅在**几何头输入**加 LayerNorm——`src/modules/glt_dual_pretrain.py` 中 `length_head` 的输入 `encoded['center_bond_states']`（512 维，行 84）与 `angle_head` 的输入 `cat([left+right, |left-right|])`（1024 维，行 89）。原条件：两个 `nn.Sequential` 直接吃未归一化的 3D 表示，其 RMS 随训练增长约 11×。拟改条件：分别在两个 head 的 `nn.Sequential` 首部插入 `nn.LayerNorm(512)` / `nn.LayerNorm(1024)`，不动 3D backbone、O8 分支、融合与损失权重。最小验证预算：同一 `resume_02000.pt`、同四卡/同配置的 **800-update 回放（2000→2800）+ `--diagnostics`**，判据为 exact ±1 比例保持 ≈0、表示 RMS 有界、`geo` 无 >0.1 偏离、chem/FP 不退化；配套最小局部测试（head 输入归一化后前向形状/值域）。该预算需新一轮授权（本周期 800/800 已耗尽），且按 §2 E 优先级，matched 2D-only 对照排在稳定化之后。

## 四、Codex 审查与下一步

### 4.1 当前审查结论

A 的新汇总可作为后续参考，原 80 folds 无需重跑。B 尚不满足完成条件；现有诊断入口包含确定的调用／参考计算问题，不能直接用其输出判断模型根因。

报告措辞修正：失稳是“截至 5000 未恢复”，不是证明永久不可恢复；chem／FP 可以没有同样的持续退化，但原日志在 step 2676 的 FP 约为 0.1987，不能写成完全不受影响。失稳前均值必须给出窗口和 rank 去重规则，不能混用稳定段与已含尖峰的区间。比较同 step 的原始目标数和 loss，不因 onset 算法不同重新训练。

### 4.2 第一步：修复 B.2 入口与 B.1 观测口径

只修诊断和记录，不改模型科学定义。对远端现有实现逐项处理：

1. 删除 `pretrain_collate([row[0] for row in chunk])`，保留一次对完整 `(data, labels)` 列表的 collate；删除 `geometry_head_gradients` 空实现及未使用变量。
2. 制备 16 条固定记录时调用 `static=source.static_for(index)` 和 `target=source.target_for(index)`，与正式路径一致。两模式、六 checkpoint、两精度复用这些 CPU 样本；不重新采样、不生成构象。
3. 扰动几何参考必须来自真正的 noisy data。删除把 `deepcopy(clean0)` 当作 noisy、单样本几何与八样本 labels 混用、按 `min(count)` 截断对齐等逻辑。
4. 键长按 noisy data 的中心 bond 索引逐项对应干净 distance；角度按同一批的物理 bond pair 匹配有效、非 self、中心一跳关系，不能靠所有角度列表的前缀对齐。参考 loss 使用与训练相同的逐样本平均和有效样本分母；无中心角度样本角度项为零，并单独计数。两批参考均写入报告。
5. 为诊断暴露带梯度的 length、angle、FP 分项标量（仅诊断需要时返回，不登记参数／buffer）；复用训练原公式与有效 mask，不从已经 float/detach 的统计恢复梯度，也不额外重跑随机 forward。
6. 梯度按同一顺序遍历所有 3D encoder 参数；`None` 以同 shape 零张量对齐，不能仅拼接非 None 参数。报告 length／angle／FP 的未加权范数、按原权重加权的范数及两两 cosine。任一范数为零时 cosine 为 null 并注明，不用长度不等来代替对齐。
7. Gaussian 输出尺度通过真实 forward 的只读 hook 采集距离／角度 Gaussian 输出；angle 统计屏蔽 padding，区分有效真实角与合成 self。删除零距离、零元素类型的假输入 probe；有效 sigma 的均值与最小值统一采用 `abs(std)+0.01`。
8. tanh 导数用捕获的 pre-tanh 转 FP32 后计算，分别记录实际输出精确 ±1 比例及可选近饱和比例（明确阈值只是统计，不作为有效性 gate）。空集合返回 count=0、统计 null，不能产生 NaN 或伪造零均值。
9. 检查六 checkpoint 的 step／fusion／模型结构、原 world size、cohort／bundle／static／target 身份。固定样本来自共同原始身份；缺任一 checkpoint 则非零退出并记录缺项，不能写 MISSING 后整体成功。
10. 将错误的单一 `--checkpoint-root/<mode>` 假设改为显式 `--concat-checkpoint-root` 和 `--kfuse-checkpoint-root`，分别指向现有两份 `results/glt_dual_static_pretrain_5k_<mode>`。不为满足路径复制大 checkpoint。GPU 选择以实际 device 为准，不对 CPU 调用 set_device。

最小相关验证：补正式测试覆盖开关不改变 loss／梯度／state_dict／RNG；分项还原；物理角度配对与混合 batch offset、无角度；None 梯度对齐；真实 Gaussian hook；六 checkpoint 缺项和身份不一致。已有聚合 9 项无需因交接重复执行。确定性单元测试中可使用合成样本，但正式 16 条诊断不得用合成构象替代。

### 4.3 第二步：执行 B.2 固定批量诊断

前述相关测试通过后，在 tmux 独立 window 运行一次六 checkpoint 诊断，预算仍为每份 16 条（两批八条）、FP32／BF16 eval 前向及第一批的 FP32 分项梯度；不执行 optimizer update。

接口模板（执行者填入空闲 GPU 与新的日志／报告目录，运行前记录完整展开命令）：

```text
python scripts/diagnose_glt_dual_pretrain.py
  --cache-root data/processed/mips_trimer_scage
  --cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1
  --dual-static-root data/processed/glt_dual_v2/pi1m/dual_static_v1
  --pretrain-target-root data/processed/glt_dual_v2/pi1m/pretrain_targets_v1
  --concat-checkpoint-root results/glt_dual_static_pretrain_5k_concat
  --kfuse-checkpoint-root results/glt_dual_static_pretrain_5k_kfuse
  --report-json <新诊断目录>/fixed_batch.json
  --device cuda:<空闲设备>
```

输出六 checkpoint 对照表，至少分别列 length／angle／FP、扰动参考、精度差异、表示尺度和梯度结果。eval 精度比较只能定位当前数值敏感性，不能证明训练中 BF16 是根因；梯度冲突只称支持性证据。若全部输入身份正确且数值有限，不因观测到较差性能而等待额外批准，继续 B.3。

### 4.4 第三步：补齐并执行 B.3

遵守第二节 D 的全部原预算和恢复定义。先补参数与相关恢复测试，再从原 2k checkpoint 运行一次四卡回放至最多 2800；不先试跑一轮 optimizer smoke 再把预算清零。

* `--stop-after-step` 必须满足 `start < stop <= config.max_optimizer_steps`；用于主循环及 prefetch 的停止位置，但不得改原 config／scheduler identity。诊断保存点只允许落在本次区间。
* 每步 length／angle／FP 使用全局 sum/count 归约后仅由 rank0 写入全局日志；非线性分布统计若只来自单 rank 必须标明，不能伪称全局分位数。
* 模块梯度范数在 DDP 同步后的 clipping 前采集；全局剪裁保持原值。hook 不重复调用 encoder，不额外抽样。
* 只写新诊断目录的 resume 状态和观察报告，不生成或覆盖 deploy；保存操作保持 rank RNG 和下一抽样位置一致。
* 前 10 步若恢复身份／目标数不一致立即停止，先定位；已发生 updates 计入本周期总预算。明显轨迹偏离时交付证据，不自动重置再跑 800 步。
* 继续到 2800 或既定故障停止条件；即使不复现也不追加第二模式回放、不延期。原始 loss 有限的尖峰本身是观测对象，不另设随意阈值提前过滤。

输出 step 2001–2800 的分项轨迹及异常附近逐样本／表示／梯度证据，明确 length 尖峰与 angle 平台是否同时存在，以及损失变化是否发生在同类样本或同一优化阶段。

### 4.5 第四步：决策、交付与下一轮

完成 B.2／B.3 后，ZCode 提交事实、支持性证据、未证实假设三个层次的报告；从第二节 E 选择一个首要改动建议，给出具体改动位置、原条件、拟改条件和最小验证预算。不能把所有稳定化措施一起加入，不因低 loss 自动宣称提升性质预测。

若证据不足以选定根因，允许本轮结论为“未定位”；写出已排除内容和一项最小补充诊断，不自动执行。matched 2D-only、二面角／非键任务、融合门控、LR／损失权重实验均只列为后续建议，本轮不启动。

交付时补齐 Plan.md 执行记录以及实际受影响的 PIPELINE.md／RESULTS.md 段落（不覆盖历史结果），列明代码版本、相关测试、诊断报告、tmux／命令／日志、累计 updates 和未完成项。由 Codex 审查后，才把该完整周期存入 PROJECT_HISTORY.md 并制定下一轮计划；当前周期暂不归档为已完成。

---

## 新周期（用户直接授权）：固定 Concat 5k（geonorm）→ 7 任务正式微调 → 三方对比

### 计划头

|项|内容|
|-|-|
|计划 ID|GLTV2-FIX-20260916-02|
|修订|r1（用户 2026-09-16 直接下达 Phases 0–15）；r2 执行中修订：用户指示「可以不测试 egc」「微调执行之后使用 4 个 GPU」|
|状态|待审查（执行已完成，等待 Codex 审查；本周期未归档）|
|授权来源|用户消息直接授权（正式固定 5k + 正式微调 + 聚合 + 三方对比）|
|规划／执行／审查|规划：用户直接指令；执行：ZCode；审查：待 Codex|
|代码基线|Phase 1 记录 HEAD `70f936e`（含 P2 修复的工作树，patch sha256 `b2421c8d62d7d4a499b6d216bc6c8ea13ca082e5d5c4b13c0ce07b46dc42a50a`；该改动已由 emt11 提交为 `31c1b71`）；本轮新增脚本未提交|

### 实施计划（用户指令要点）

从 step 0 用 `geometry_head_norm=true` 正式固定 Concat 5k（world_size 3、microbatch 84、accumulation 4、global batch 1008、BF16、其余科学参数不变），随后健康审查 → deploy 校验 → eat/fold0 smoke → 8×5 正式微调（原计划；实际按 r2 指示跳过 egc，为 7×5，GPU1/2/3，最大并发 3）→ 验证式聚合 → 与旧 Concat／KFuse 三方对比。禁止项：10k/20k、KFuse 重训、O8-only、3D-only、KFuse-v2、multi-seed、其他 LayerNorm 位置、残差长度预测、robust loss、LR/loss weight/noise 改动。

### 执行记录（ZCode）

**Phase 0–2（冻结、pre-flight、3 卡 smoke）**：新增 config `configs/mts/glt_dual_three_task_concat_geonorm.json`（`geometry_head_norm: true`，零新增参数）；`scripts/pretrain_glt_dual.py` 读该字段；30-update smoke（accum=4、world=3、loss 有限、写 resume、不产生 deploy）；51 项测试通过。

**Phase 3 正式固定 5k**：tmux `Uni-Poly:geonorm_5k`，日志 `logs/glt_dual_static_pretrain_concat_geonorm5k.log`，输出 `results/glt_dual_static_pretrain_5k_concat_geonorm`，6 个 checkpoint（1000…5000）含 deploy/resume，12:06:52 UTC 到 step 5000，正常退出。

**Phase 7 健康审查**：新增 `scripts/pretrain_health_report.py`（只读）。产物 `results/glt_v2_fixed_concat_5k_20260916/pretrain_health_report.json`，**FORMAL_FIXED_5K_HEALTHY = YES**（7 项阻断检查全通过）。step 5000：chem 0.1552／geometry 0.000518／fp 0.0341（旧 Concat 0.1553／0.310754／0.0400；KFuse 0.1538／0.001063／0.0376）。稳态最长量：length 逐图 ≤0.01103、angle exact ±1 饱和率 0.0、tanh 导数最小 0.0031、angle 梯度 last-500 最小 0.00206、graph_3d_rms 最大 1.972（末值 1.439）、clip 前梯度 last-500 最大 0.617。残留关注：原始 `bond_states_rms` 末值 2.068 仍高于失稳前健康带 1.07–1.22（失稳带 5.77–9.17），已记入报告，不阻断本周期。

**Phase 8 deploy 校验**：新增 `scripts/validate_glt_dual_deploy.py`。产物 `deploy_05000_validation.json`，**FIXED_CONCAT_DEPLOY_VALID = YES**：187 张量／39,021,942 参数、全 fp32 有限、仅 O8+GLT+Concat、无 head 与 geometry_norm 依赖、与 `resume_05000.pt` encoder 子集逐张量 bitwise 相同、strict load 通过、真实样本前向 [2,1] 有限、缓存零写入。

**Phase 9 smoke**：tmux `Uni-Poly:fixed_eat0_smoke`，输出 `results/glt_v2_fixed_concat_5k_20260916/eat_fold0_smoke`，`outer_test = NOT_RUN`，2 epochs best validation R² 0.7737。

**Phase 10 正式微调**：tmux `Uni-Poly:fixed_grid40`（3 卡）→ `Uni-Poly:fixed_grid_rest`（4 卡）；输出 `results/glt_v2_fixed_concat_5k_20260916/finetune_grid_fixed_concat`，日志同名 log-root + `.console.log`。偏离：按用户 r2 指示跳过 egc（已完成 egc_fold0/1 保留、egc_fold2 半成品保留），剩余 ei/eps/nc/xc 四任务改用物理 GPU0/1/2/3 并发；为此给 `scripts/run_glt_dual_finetune_grid.py` 增加 `--resume`（续跑已有根目录、跳过已完成单元、遇半成品报错不覆盖）。最终 37 个 shard 全部 exit 0，0 失败。

**Phase 11/12 聚合**：`scripts/aggregate_glt_dual_finetune.py --task <7 任务>` → `results/glt_v2_fixed_concat_5k_20260916/aggregation_review_7task/summary.json`，status PASS、task_count 7、fold_count 35、predictions_recomputed、OOF 恰好覆盖一次、未使用 legacy 兼容。**FIXED_CONCAT_MACRO7_R2 = 0.777210**（pooled OOF macro7 = 0.782525）。聚合脚本对非 8 任务刻意不输出 macro 字段，该 macro 由已验证的逐任务均值派生并在报告中标明。

**Phase 13 三方对比**：新增 `scripts/compare_glt_dual_finetune_results.py`。产物 `three_way_comparison_7task.json/.md`。7 任务 macro：固定 Concat 0.777210、旧 Concat 0.772134（+0.005077）、KFuse 0.753544（+0.023666）。逐任务相对旧 Concat：eat +0.006935、egb +0.012941、ei +0.013312、nc +0.000400、xc +0.029176 改善；eea −0.014391、eps −0.012839 退化。

**口径与限制（待审查问题）**：(1) 跳过 egc 后只能给出 7 任务 macro，**不可**与 8 任务 macro8（0.7877379364／0.7695629053）混用；参考运行已按同样 7 任务重算。(2) 预训练对照中 world_size 由 4 变 3（accumulation 3→4），global batch 均为 1008，因此每个 optimizer step 覆盖的 absolute-position 样本集合保持同一连续区间；world_size 4→3 改变的是 rank/microbatch partition、dropout RNG 与样本的对应以及数值 reduction 路径，因此该完整训练仍不是 bitwise matched single-variable run，样本级单变量证据是 P2 replay 而非本次对比。(3) 共享 development folds 非独立盲测，macro 提升幅度小且 2/5 任务退化，不宣称已超 baseline。(4) egc 新增 fold0/1（0.8927／0.8854）与旧 Concat 同 fold（0.8979／0.9010）仅供参考，2/5 fold 不足以判定。

**未执行**：egc 其余 3 fold、8 任务 macro8、10k/20k、KFuse 重训、O8-only、3D-only、KFuse-v2、multi-seed、其他 LayerNorm 位置；本轮 0 次额外 replay、0 次缓存重建。

### 审查与下一步

待 Codex 审查。建议下一步（需新授权）：补 egc 5 fold 恢复 8 任务口径，或按计划优先级进入 MATCHED O8-ONLY 对照。

---

## 缓存优化周期 CACHE-20260916-01 / r2 执行记录（ZCode，2026-09-16）

依据 `Plan_Cache.md` §10 及用户“请你作为 ZCode 执行”的直接指令。本轮开始前在
`dev` 执行 `git pull --ff-only origin dev`（Already up to date），核对工作树、`Uni-Poly`
tmux 窗口与进程；未发现构建、训练、GPU smoke 或其他 parity 进程。active PI1M／下游
bundle、cohort、`dual_static_v1`、`pretrain_targets_v1` 全部只读。

### 实施变更

* `scripts/build_glt_dual_static_cache.py`：将 staging 身份写入移到可靠 flock 之后；未知的
  非空 staging（缺 `build_context.json`）拒绝；完整 chunk 恢复调用载荷校验；最终 manifest
  在冻结前校验连续 chunk、keys、必需数组和 target 标志；修正 `started` 幂等分支，并支持
  static 已发布、targets 缺失时的单边恢复。
* `src/dataset/glt_dual_static.py`：允许同一已确认 staging 中“`.complete` 已写、rename
  尚未发生”的临时 chunk 先验证后提升；损坏／未知临时目录只在显式 quarantine 根下隔离；
  `load_chunk_payload` 还要求完整字段集合和 ragged offset／载荷边界。
* `scripts/finalize_glt_dual_static_artifact.py`：冻结前逐 chunk 验证格式、target 标志、连续
  覆盖及载荷；已冻结 artifact 仍只读幂等或拒绝。
* `scripts/verify_glt_dual_static_parity.py`：新建独立 static 与 target 临时 artifact，比较
  固定真实 keys 的 BRICS／原子索引／packed 与解包指纹、online 与 static clean/noisy 输入、
  mask／label／中心 distance／angle／skip reasons；拒绝既有 temp 路径，记录分类状态、预算、
  zero-write 和未执行的模型状态，避免 `shutil.rmtree`。
* `scripts/benchmark_glt_dual_read.py`：计数改为真正 chunk-cache miss／array open，worker 计数
  明确为不可得；worker=0 采用 AB／BA 交替与一致预热，记录实际 schedule、重复数和资源字段
  的 self／tree 范围。生产默认容量仍为 2。
* 测试：扩展 `tests/test_glt_dual_static_recovery.py`，新增
  `tests/test_cache_optimization_tools.py`；未修改 active 缓存、模型、训练配置或数据划分。

### 实际验证

1. 初次命令 `PYTHONPATH=.` 收集阶段因测试 fixture import 路径缺失退出码 2，日志
   `logs/cache_opt_r2_tests.log`；未据此形成代码结论。按项目约定改为 `PYTHONPATH=.:tests`。
2. 相关恢复／reader／pretrain 回归最终命令：
   `PYTHONPATH=.:tests pytest -q tests/test_cache_optimization_tools.py tests/test_glt_dual_static_recovery.py tests/test_glt_dual_static.py tests/test_glt_dual_cache.py tests/test_dual_glt.py tests/test_dual_glt_pretrain.py`
   结果 `55 passed, 1 warning`，退出码 0；完整日志
   `logs/cache_opt_r2_postdoc_tests.log`。其中覆盖实际 `build()` 两小 chunk、重复幂等、
   static-only→targets 单边恢复、跨进程 flock、临时 complete chunk、payload 缺失／截断／
   offset 错位、冻结前 finalize 和报告失败分类。
   3. 首次真实 parity 在比较器修复前得到 `DATA_MISMATCH`（仅 target tuple／integer container
   比较器误报），退出码 1，保留于
   `results/cache_optimization_repair_20260916T234254Z/parity.json`。修复比较器后使用新目录
   重新执行最终命令（`Uni-Poly:cache_opt_r2_parity3`）：
   `logs/cache_optimization_repair_20260916T234930Z/parity.log`，退出码 0，报告
   `results/cache_optimization_repair_20260916T234930Z/parity.json`。PI1M 20 条（普通 12、
   无中心角 8）和下游 12 条（geometry fallback 9、普通 3）均无差异；独立临时输出
   478,748 bytes（≤1 GiB），PI1M／下游 active cache zero-write 均为 true。8 条“无中心角”
   的 angle target 为空且无 NaN；原 32 条中没有可证明真实 N=0 的记录，报告明确为
   `real_n_zero_proven=false`，未用人工样本替代。
   新报告中的两组 key 列表与 r1 `parity.json` 逐项相同（未扩充或替换真实记录）。

### R1–R4 状态与边界

| 项 | 实际状态 |
| --- | --- |
| R1.1–R1.6 | 已实现并由上述 55 项局部测试覆盖；未做断电耐久承诺 |
| R2 | 固定 32 条真实 parity PASS；未新增化学审计或构象生成 |
| R3 | benchmark 口径与工具已修正；未重新运行长 benchmark，不产生新的提速结论；化学结论保持“特征分桶＋有限逐例证据”，不改历史 JSON |
| R4 | 本执行记录与报告待 Codex 独立审查；当前状态“待审查”，不是最终验收通过 |

未执行：阶段 D 紧凑存储、全量缓存重建／格式迁移／生产切换、任何预训练／微调／GPU
任务，以及新的长 benchmark。没有修改 `PROJECT_HISTORY.md` 或 `RESULTS.md`；后续由 Codex
核对 diff、日志、临时 artifact 和 active zero-write 后再决定是否归档。

---

## 缓存优化周期 CACHE-20260916-01 / r3 执行记录（ZCode，2026-09-17，Codex 独立验收通过 / CLOSED）

依据用户提供的 r3 返修计划。执行前已读取 `AGENTS.md`、`Plan.md`、`Plan_Cache.md`、
`PIPELINE.md`、`RESULTS.md`，在 `dev` 执行 `git pull --ff-only origin dev`（Already up to date），
基线 `31e137c`；未发现 active cache build、parity、benchmark、pretrain、finetune 或 `torchrun`。

### 实际修改

* `scripts/build_glt_dual_static_cache.py`：冻结前汇总 static 的
  `geometry_valid_count`／`geometry_invalid_reason_counts`；新增 published-side payload contract
  校验，覆盖 keys、chunk 连续性、`.complete`、target/static 标志和 `load_chunk_payload`，用于
  idempotent 与单边恢复。target manifest 不增加几何汇总。
* `scripts/verify_glt_dual_static_parity.py`：成功必须同时满足 comparison PASS、active frozen
  cache zero-write 全真、临时输出未超预算；加入 `UNRESOLVED_PROVENANCE` 和固定 key 列表核对。
  `tests/fixtures/glt_dual_parity_expected_keys.json` 保存原 20 PI1M／12 downstream keys 及 digest。
* `scripts/benchmark_glt_dual_read.py`：`accepted` 改为 `performance_gate_passed`，分开记录
  zero-write、resource observation 和 overall recommendation；不把 peak FD/RSS 当作无泄漏证明。
* `tests/test_glt_dual_static_recovery.py`、`tests/test_cache_optimization_tools.py`：补齐
  builder→finalize summary、target 无伪 summary、published static/target payload refusal、parity
  zero-write／budget／fixed-key fail-closed 的 synthetic matrix A–I。
* `RESULTS.md` 修正文案为 7 tasks，并修正 world-size 改变的样本／RNG 描述；本文件和
  `PIPELINE.md` 追加当前状态与 r3 执行记录。未修改实验数字。

### 实际验证

首轮 focused 测试在 `Uni-Poly:cache_opt_r3_tests`、`logs/cache_opt_r3_tests.log` 中因故障注入
shape 与 count 相同而未触发预期拒绝，退出码 1；该测试 fixture 错误已保留。修正为真正错形后，
在 `Uni-Poly:cache_opt_r3_tests2` 执行：

`PYTHONPATH=.:tests pytest -q tests/test_glt_dual_static_recovery.py tests/test_cache_optimization_tools.py tests/test_glt_dual_static.py tests/test_glt_dual_cache.py tests/test_dual_glt_pretrain.py`

结果 `36 passed, 1 warning`，退出码 0，日志 `logs/cache_opt_r3_tests2.log`；另有 `py_compile`、
`git diff --check` 和固定 key digest 核对通过。未重跑 r2 的 32-key 真实 parity，未运行任何模型。

### 范围边界

未重建 PI1M／下游 active cache，未重新生成 conformer，未运行 2048 benchmark、Stage D、GPU、
预训练或微调，未删除／迁移／切换产物；执行阶段未修改 `PROJECT_HISTORY.md`，`RESULTS.md` 仅改文案而未改
历史实验数字。r3 随后经 Codex 独立审查通过，当前状态为 `CLOSED`；本关闭不改变上述未执行项的状态。

---

## 缓存优化周期 CACHE-20260916-01 / r1 执行记录（ZCode，2026-09-16）

依据 `Plan_Cache.md` r1（用户指示"执行该计划"）。执行前核对：`git pull --ff-only` 为 Already up to
date、分支 `dev`、工作树干净、无训练/构建进程；active 身份与计划基线一致（PI1M bundle
`30f17b59…`、cohort `b03f96a1…`、dual_static_v1 `9ff122cc…`、pretrain_targets_v1 `5e7b5ec8…`、
下游 bundle `1545eda5…`，各 235／8 chunks）。本轮唯一产物目录
`results/cache_optimization_20260916T145328Z/`，日志 `logs/cache_optimization_20260916T145328Z/`，
tmux window `cache_opt_a`／`a2`／`a3`／`a4`／`parity`／`bench`。

### 阶段 A（有界只读化学审计复核）

* 全量分解现有逐样本审计记录（`audit_reason_decomposition.json`）：11,338 FAIL 中 9,725 为非对称／
  芳香连接对、1,571 为源中显式 `[2H]`、41 两者兼具、1 条残余；20,000 PASS 对照中 19,988 无这两类特征。
* 逐例复核 44 个真实样本（预算 48）：10 条判为"审计期望 H 未计入显式源氢"、19 条判为"审计参照未对齐
  已声明 `mismatch_single` 连接策略"、1 条残余判为"审计自身重建分子导致"（payload 与源逐项一致，
  P=O 键级 2、电荷 0，重建后 `SanitizeMol` 出现 -1/SP3 氧），8 条 stereo 参照未解对照、6 条 PASS 对照。
  **0 条未解释**；未把任何状态改写为 PASS，未估计化学错误率。
* 4 条 `trimer_ru_internal_bond_contract` 在图层面复现（未调用 ETKDG／MMFF）：每个案例最外侧 RU 副本
  对芳香环的感知不同（如 `codes_per_ru [4,4,1]`），检查器据此拒绝而未发布不一致副本；这 4 条不在接受集，
  记为 manifest 的 `unclassified_failure_count: 4`。
* 9 个下游几何 fallback 独立核实：全部 `geometry_invalid`、全部仍在下游 cohort records 中、全部属 egc；
  3,646 + 9 = 3,655 口径自洽。

### 阶段 B（最小修复、故障注入与真实样本 parity）

修改文件：`src/dataset/glt_dual_static.py`、`scripts/build_glt_dual_static_cache.py`、
`scripts/finalize_glt_dual_static_artifact.py`。修复 5 处缺口（staging 构建身份、chunk 原子写与隔离
恢复、单边发布幂等、冻结后只读、读取端载荷校验），细节见 `results/cache_optimization_<RUN>/decision.md`
与 `PIPELINE.md` 附录。新增 `tests/test_glt_dual_static_recovery.py`（10 项，全部通过）；
`tests/test_glt_dual_static.py` 3 项、相关既有套件 68 项保持通过。

真实样本 parity（`parity.json`）：32 个固定真实样本（PI1M 20：普通 12＋无中心角 8；下游 12：几何
fallback 9＋普通 3）用当前构建块生成新的临时派生缓存，与已发布参考 artifact 逐数组比较，
**0 个样本存在差异**（整数／搬运字段完全一致）；临时输出 469,563 bytes（预算 1 GiB）；
冻结缓存零写入。

### 阶段 C（公平 benchmark 与候选判定）

固定 2,048 索引（`OrderedSampleStream(seed=42)`，清单与 sha256 存于 `benchmark.json`）、同一位置
布局以保证样本级 mask／noise／targets 相同、仅 CPU、无模型／GPU；worker=0 配对各 2 次重复，
另加 3 worker 预取各 1 次。分项（worker=0）：源 LMDB 读 17.8 ms、static 读 4.95 ms、target 读 1.04 ms、
完整准备 11.5 ms、合计 35.4 ms。

* 候选（有界 chunk 缓存容量 64）中位吞吐 29.68 samples/s vs 基线 28.45，比值 **1.043 < 1.10 ⇒ 不采纳**；
  p95 无退化（0.983），FD 峰值由 92 升至 1890（两次重复相同，非泄漏），且候选与基线产出**逐字节相同**的
  准备结果（`benchmark_candidate_identity.json`）。
* 第二候选（复用静态角度索引）已是当前生产路径：`materialize_dual_geometry` 直接使用 static 的
  `angle_pos_triplet`／`token_pos_index_*`，故 static 读仅占约 14%，无新的单因素改动依据。
* 生产默认容量保持 2；本轮不切换配置、不删缓存、不重建、不训练。

### 审查待办与未执行项

未执行：阶段 D（紧凑存储）、全量重建、格式迁移、生产配置切换、缓存删除、任何训练或模型评估。
停止条件均未触发（身份一致、无 writer 冲突、无非有限数据、中心监督与样本集合语义未变）；相对计划的
唯一偏差是 worker=0 配对重复次数由最多 3 次降为 2 次，以遵守 30 分钟总预算。交付结论待 Codex 审查
并按 `Plan_Cache.md` §8 分层完成标准归档；`Plan_Cache.md` 的执行记录与 `PROJECT_HISTORY.md` 归档由 Codex 维护。

### Codex 审查与 r2 下一步（2026-09-16）

* 依据：代码／测试源码及既有 results/cache_optimization_20260916T145328Z/ 报告的静态审查；没有重新运行测试。结论为需返修，不能接受上方 r1 自检的“A／B／C 全部完成”。
* 阻断项：完整 build() 幂等分支使用未初始化 started；未知非空 staging 可被补当前 context；临时 .complete 到 rename 的中断不能恢复；chunk 载荷校验未接入 builder 恢复／发布；锁初始化和持锁前写入有竞争窗口；冻结前最终汇总未接通。
* 证据缺口：32 条 parity 仅覆盖静态字段，未实际比较新目标缓存和固定 mask/noise 的完整准备路径；性能脚本不是交替配对、部分多 worker 指标仅来自主进程；中心 angle_pairs 监督仍遍历 line_path，不能称该优化已实现。
* 结论更正：11,337 条是源特征分桶数量，不是逐条化学因果验证；44 条中 8 条 Stereo 参照仍未解。“两次 FD 峰值相同”不证明无泄漏。旧测量保留，容量 64 继续不采用，默认保持 2。
* 完整返修合同见 Plan_Cache.md r2 第 10 节。先以两个小 chunk 的合成数据覆盖实际 builder 和受控双进程锁竞争，再复用原 32 条样本补适用目标／clean-noisy parity，临时输出不超过 1 GiB；不得把 reference targets 重开为 candidate。修正差异／超限退出码与临时目录保护。
* 只对现有 benchmark 工具做局部口径修复与合成测试；不新增真实长 benchmark、角度优化、阶段 D 或正式实验。保存准确 argv／环境／退出码；找不到历史测试日志则标未核实，不补造“110 passed”证据。
* 本轮仅写计划，返修未执行；由原执行者在既有授权内继续，扩大范围另行决定。修复后更新此处并置待审查；CACHE 周期仍未关闭，不归档为成功完成。
