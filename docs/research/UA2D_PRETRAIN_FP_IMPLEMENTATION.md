# UA2D-PRETRAIN-FP-20261019-01 / r2 执行实现

基于 dev / 3503fbe780a5399410d6db6055305925765c3ff8 及既有未提交 v2 工作区；用户授权实现和有限完整周期。旧工作区改动保留，无删除、远程同步、文件哈希验证或 TODO 读取。本文是执行方实现说明，不是独立审查结论。

## 模型与受控条件

- 新 schema `ua2d_galformer_objectives_v3`；两版均无3D→2D融合。V1为二维／三维掩码＋ALIGN；V2为二维掩码＋坐标／键长／cos角度＋ALIGN。
- ALIGN保持双向多正例之和，权重按完成updates从0渐增至0.1。坐标prediction和target差值共同除以σ=0.03Å；末层零初始化。
- FP仅为同一次masked O8 mean＋norm2的2048位Morgan监督目标。独立seed20261019无dropout头，逐分子正负位balanced BCE，权重同ALIGN渐增；FP不作为输入或下游特征。
- 共同step0固定seed42，公共模块一次保存；两版任务头保存对应初态。B_FP加载B的任务头初态及公共step0，FP头独立初始化；训练RNG重置42+rank。
- GPU2/3，world2，microbatch84，accumulation6，global1008；每微批最大168对齐候选。500更新warmup，5000更新余弦端点1e-6，AdamW、bf16、clip1。
- 新source reader使用可信direct train indices及行身份语义，不调用旧哈希路径。数据增强按seed42＋全局呈现位置复现；正常训练逐rank记录source rows和绝对位置。

## 流程与预算

专用入口 `scripts/run_ua2d_pretrain_fp.py controller`。控制器为budget.json唯一writer，worker各写独立consumption，计入失败／部分starts、attempted updates、已开始epochs／validation和预测图数，无自动retry/resume。FT每卡以至多两路一波派发，波次之间等待该卡已有计算全部释放，合计最多四路，PT与FT分阶段。

V1 PT→35 FT→V2 PT→35 FT→validation冻结B→一次固定train FP缓存→B_FP PT→35 FT→全部冻结→三臂各35正式TEST→最终一次CPU suite。预算上限来自PRETRAIN.MD r2；不运行旧controller。FT复用uniform_u1 optimizer/dropout和基础train/evaluate函数，70epochs中完整精度validation R²严格更高才换best，相等保留更早epoch。保存best完整O8/norm2/head、scaler和预测；正式恢复绑定task/fold/seed/head/parent/scaler及冻结TEST indices。

所有长任务在tmux Uni-Poly独立window。日志含command.txt、cwd.txt、start/end、stdout/stderr、真实exit.txt；控制器记录window ID和实际命令。PT保存每rank RNG及optimizer／schedule／采样位置，记录不授权自动恢复。

数据、schema、checkpoint、OOM、非有限值、覆盖或writer异常停止受影响单元并保留现场。一个FT失败后不再派发，已在途的其他单元正常收尾后交付失败证据；不替换失败折，不据部分结果冻结。

## 验证与交付

仅在完整周期末执行一次 `timeout 300s python -m pytest -q tests/test_ua2d_pretrain_fp.py`。该suite为新增风险的合成CPU测试，涵盖单位、零初态、非零等变、FP参数／位序／dummy、balanced BCE和梯度、初始化／RNG、两rank ALIGN远端梯度／无效样本、严格schema／pure2D、预算和冻结。没有逐阶段smoke、GPU预跑或旧suite重跑。

结果和日志固定在 `results/ua2d_pretrain_fp_20261019/r1/`、`logs/ua2d_pretrain_fp_20261019/r1/`；目录名沿用r1但执行revision是r2。产物已存在则拒绝重新开始。CURRENT_PLAN仅在旧writer暂停且没有观察到并发编辑时追加登记，不覆盖旧周期。

实际阶段、退出码和消耗以status.json、budget.json与各worker日志为准。全部完成后状态待审查；执行成功、0.010门槛、独立审查分开记录，PROJECT_HISTORY本轮不归档。

## 2026-10-19 11:46（北京时间）首次运行失败及修正

共同step0／两版任务头初始化完成（exit0）。V1 PT首次启动（exit1）在第一微批collate时触发NameError：新增FP标签单样本处理误重复写入collate函数，引用了不存在的config／fp_target。两个rank均记录84次样本呈现，attempted window1、completed optimizer updates0；总PT starts1、样本呈现168。没有进入模型forward、FT、FP缓存或正式预测。

已删除collate中错误重复块，并显式拒绝混合FP标签的batch；未覆盖失败产物、未重启训练。失败计入3 starts上限，完整三臂周期现需另行处理授权；本轮不会消耗剩余两次starts补跑或跳过失败臂。最终一次CPU suite仅验证交付代码，结果另见final_review和日志，不等于实验完成。

## 最终一次CPU证据（11:48，北京时间）

唯一一次suite：7 passed、1 failed、exit1，耗时12.41s。失败项为test_cpu_two_rank_collectives_and_remote_gradients中的远端候选梯度非零断言；首轮两rank不等长、重复身份和无有效几何条件没有触发先前断言失败。其他七项通过只支持其具体合成覆盖，不能证明真实PT／FT正确。

静态定位：失败断言输入为未归一化随机向量，而正式AlignmentProjection返回L2归一化向量。未归一化时除以temperature0.1会产生大幅logits，存在softmax饱和／梯度消失的数值风险；这是失败原因的推断，未复测确认，不据此断言共享ALIGN helper有错或已修好。已将该CPU测试的输入按真实接口L2归一化；此修改未再次执行，唯一suite预算已用完，最终测试仍记失败。

最终状态：阻断；不是待审查的完整实验周期。V1一次启动失败，V2／B_FP／105个FT及正式TEST均未执行，FP标签缓存与两份selection freeze均未生成，七任务无本计划正式结果。共同step0、两版任务头、失败rank消费／采样记录、真实退出码和七任务不可比较报告完整保留。旧作业保持暂停，GPU0/1未使用。后续训练重试或再执行suite均需用户明确处理预算／授权，不由本轮自动派发。

## r3：用户授权修复并恢复（2026-10-19）

用户在审查后明确要求“请你给我修复并运行！”。本轮由 Codex 实施，不能将实施自检标为独立审查通过。累计上限调整为 PT starts4、尝试窗口15001、样本呈现15120168、CPU suites2；仅覆盖已发生失败加原定三臂，FT／TEST 及 GPU2/3 限制不变。

新增显式 `scripts/run_ua2d_pretrain_fp.py recover-r2`：在既有根加锁，严格继承旧三条job和已耗预算，保存旧manifest／budget／失败报告快照，复用现有common_step0及任务头。V1第二次尝试使用`pretrain/V1_attempt2/`与新日志，通过`parent_bindings.json`供FT／freeze／正式评估读取。原`pretrain/V1/`及日志保持不动；新manifest为`execution_manifest_r3.json`。没有自动恢复失败训练状态或重建初态。

审查纠正：此前声称正式汇总读取正式目录的scaler不准确，原`aggregate()`已正确读取FT的train-only scaler；此逻辑不修改。本轮增加合成用例覆盖正式目录没有scaler副本时的汇总，同时验证恢复不重置旧账和新parent绑定。远端梯度测试维持L2归一化输入，保持非零梯度断言。

所有修复结束后仅追加一次CPU suite，通过后才恢复PT；结束时复用该证据，不再次运行suite。测试或新训练失败则停止，无进一步自动重试。

r3实际证据：修复后的CPU suite **10 passed、1 warning、12.21s、exit0**，日志`logs/ua2d_pretrain_fp_20261019/r1/pretrain_fp_repair_cpu_r3/`，通过记录`recovery/r3/test_gate.json`。本次测试后仅更新文档，不修改执行源码或重复测试。控制器窗口`Uni-Poly:pretrain_fp_recovery_r3_controller`，日志`logs/ua2d_pretrain_fp_20261019/r1/controller_r3/`；V1窗口`Uni-Poly:pretrain_fp_pt_V1_attempt2`。2026-10-19 04:26 UTC（12:26 Asia/Shanghai）检查时两rank均已完成6 updates，loss与梯度范数有限，已越过原拼批故障；这是正式PT进行中的进度，不称为smoke或完整周期完成。FT／正式TEST尚未启动，旧作业未恢复。


## r4：用户要求每版约5小时，固定5000步的工程加速重启

旧r3由用户要求停止；两rank各完成1390步，无最终checkpoint，现场与全部消费保留。新的完整三臂从原common step0重新开始，仅GPU2/3；不改变模型／loss／batch／增强／FT条件。计划细节与累计预算以PRETRAIN.MD顶部r4及CURRENT_PLAN.md为准。

新增scripts/prepare_ua2d_fast_cache.py：16个CPU worker对固定train唯一源key验证、解码并构造clean模板，一次缓存干净监督、周期组、canonical对应及动态几何索引，单LMDB writer。FP标签提前确定性生成并打包，只有B_FP训练时使用，V1/V2不读取目标。原source和旧失败／停止记录保留。

新增src/dataset/ua2d_fast_pretrain.py与scripts/pretrain_ua2d_fast.py：每rank4worker有序预取、bounded LRU、NumPy bytes IPC；独立DataLoader随机流；clean模板只读，逐次mask/noise保留原seed及位置语义。CPU计算window监督计数，保留每微批独立对比候选语义；计数／日志all-reduce合并为向量；pinned CPU batch每微批一次H2D。5000updates、global1008/micro84/acc6/候选168不变；实际step/data/compute时间和ETA记录在每臂timing.json及stdout。

新增scripts/run_ua2d_fast_campaign.py：结果／日志独立fast_r4根，持有旧控制器锁以防双writer，继承已结束jobs和预算，复用原初始化。通过UA2D_PRETRAIN_FP_CAMPAIGN=fast_r4显式上下文复用既有FT／冻结／报告代码；tmux子window的run.sh显式export该上下文，默认旧入口不改变根。每版PT命令18000秒超时；未达5000步或未保存完整包必须阻断，不以减步数满足时间目标。共享缓存准备时间与FT/TEST另计。

唯一新增工程suite tests/test_ua2d_fast_pretrain.py：**6 passed、1 warning、9.94s、exit0**。V1/V2×FP开关，普通／单原子／芳香／无几何样本，在多个位置上逐字段、逐标签、拼批索引及序列化后严格相等；验证spawn预取的绝对位置、源行序与实际准备计数；验证新根及累计旧消费。科学模型与DDP梯度等复用r3的10 passed证据，不重复执行。日志logs/ua2d_pretrain_fp_20261019/fast_r4/ua2d_fast_final_cpu/。收尾复用测试证据，不追加suite。

实现自检不等于独立审查，缓存等价性测试不能证明吞吐达到5小时；时间目标以真实完整训练和保存的wall time为准。新完整结果未产生前七任务SOTA均不可比较。
