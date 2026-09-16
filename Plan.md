# GLT-V2：结果闭环、几何失稳诊断与优化决策

## 计划头

|项目|当前记录|
|-|-|
|计划 ID|GLTV2-20260916-01|
|修订|r2：阶段性审查后续执行；修复诊断入口，再执行 B.2／B.3，不增加原回放预算|
|状态|需返修（诊断代码）；A 已交付，剩余诊断／有限回放仍在原授权范围内|
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
