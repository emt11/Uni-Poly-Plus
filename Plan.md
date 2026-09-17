# 当前计划：GLT 双通道预训练与微调提速

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | SPEED-20260917-01 / r3：有界执行记录收口；补记恢复 RNG 与 grid 异常路径更正 |
| 日期 | 2026-09-17（UTC） |
| 状态 | 待审查；代码实现、局部测试、32+32 parity 与 4-GPU 有界 smoke 已执行，恢复 RNG 更正后的 GPU 集成尚未复跑 |
| 授权来源 | 用户明确要求执行本计划，并补充允许使用 4 张 GPU；仍受本计划的样本、步数、时间和“不启动正式实验”边界约束 |
| 角色 | Codex 规划与后续审查；ZCode 按获授权范围实施；本轮文档由 Codex 修改、自检，无独立审查者 |
| 基线 | dev，执行前 HEAD `b185941`；工作树干净，`git pull --ff-only origin dev` 为 Already up to date |
| 交接边界 | 本文件为新提速计划；不恢复已删除的 Plan_Cache.md，不重开已 CLOSED 的缓存返修周期 |

## 1. 目标与不变项

目标是降低**相同工作量**的预训练 optimizer-update 耗时、微调 train/validation epoch 耗时和多 fold 调度总工期。先优化重复 CPU 工作与空闲等待，再根据实测决定是否优化传输；不预先承诺提速倍数。

| 比较要素 | 本轮合同 |
| --- | --- |
| 问题 | 输入复用、准备流水线和及时派发能否减少端到端时间，而不改变学习任务？ |
| reference | 当前 checkout 的 GLT 双通道 Concat 路径，以第 2 节实际配置、历史 run.json 为基准，重新做有界配对测速 |
| controlled change | 每次只改一项执行机制；入选后最多一次组合确认，不同时改模型或优化超参数 |
| 保持一致 | 相同初始化、输入 key/顺序、mask/noise、batch 划分、world size、累积、精度、loss sum/count、LR 与 scheduler、split、标签变换及验证选择规则 |

禁止以减少预训练步数、缩小 cohort、减少 validation 频率、缩短正式 epochs/patience、跳过大图/退化样本或更换精度来制造提速。保留中心 readout、物理关系多重性、N=0、无角度、无效几何的现有语义。当前不是旧 N+1/N+2 教师蒸馏路线，不新增 MD200。

不生成构象、不改 active 缓存、manifest 或 `.frozen`，不新建全量缓存、不启动正式 5k/20k 或 8×5，不改科学指标口径。第一轮不做 torch.compile、CUDA Graph、attention 替换、BF16 微调、microbatch 调整、DDP unused-parameter 策略修改或跨 fold 复用模型进程。

## 2. 已核实事实与待测假设

### 2.1 实际基线

预训练来源：[run.json](results/glt_dual_static_pretrain_5k_concat_geonorm/run.json)、[配置](configs/mts/glt_dual_three_task_concat_geonorm.json)、[runner](scripts/pretrain_glt_dual.py)。

- Concat、geometry_head_norm=true、无 MD200；PI1M 固定 cohort 959,588 行。
- 3 ranks × microbatch 84 × accumulation 4 = global batch 1008；BF16；seed=42。
- AdamW，LR=2e-4，warmup=2000，cosine horizon=20000，原训练上限5000；这些值不随短测时长改变。
- 该历史 run 已有 `--prep-workers 3` 和 `--diagnostics`，不是 worker=0 基线。
- 代码已有 `no_sync()`；DataLoader 为 prefetch_factor=4、pin_memory=False、persistent_workers=False，迭代器在训练开始建立一次。不能声称仅开启 persistent_workers 就会消除每个 step 的 worker 重启。

微调来源：[代表 run.json](results/glt_v2_fixed_concat_5k_20260916/finetune_grid_fixed_concat/eat_fold0/run.json)、[runner](scripts/finetune_glt_dual.py)、[grid](scripts/run_glt_dual_finetune_grid.py)。

- 实际微调使用 `glt_dual_three_task_concat.json`，加载 `results/glt_dual_static_pretrain_5k_concat_geonorm/deploy_05000.pt`；不可误把两个配置文件视作同一个历史命令。
- outer5_inner20；train-only standard scaler；FP32；batch=32/eval=64；encoder LR=1e-5、fusion/head LR=1e-4、weight decay=.02；最多100 epochs、warmup5、patience10，以 validation R² 选权重。
- `CleanLabeledDataset.__getitem__` 每次调用 `build_dual_sample`，每轮会重复构建无随机扰动的 clean 输入；loader 为 worker=0，未开启 pin_memory。
- grid 按 GPU 数量分批启动，等整批结束才启动下一批，确有可避免的批次等待结构；等待占总工期多少尚未测定。

机器快照：4 张驱动报告约48 GiB显存的 RTX 4090、112逻辑CPU、251 GiB RAM、`/dev/shm`16 GiB。检查时GPU利用率均0%，可用RAM约181 GiB，但swap已用满约2 GiB；这不是长期负载或可任意加 worker 的证明。执行前重新核查活动进程与资源，不停止别人的任务。

### 2.2 提速假设与证据边界

| 假设 | 静态依据 | 尚缺什么 |
| --- | --- | --- |
| 微调 CPU 输入复用收益较大 | 同一结构每次访问都重建；clean 输入不随 epoch 改变 | 构建占比、clone/collate 代价、RAM峰值、端到端收益 |
| 动态补位减少尾部空闲 | grid 以 batch barrier 派发 | 真实 shard 耗时分布；合成调度验证不能代替真实训练吞吐 |
| 预训练重复准备可减少 | clean/noisy 各构建一次，中心角度逐行 Python 扫描 | 当前 static consumer 上的实际 profile、优化后完整 target parity |
| 详细诊断/同步可能耗时 | 每步各 rank 日志，诊断计算与张量回CPU | 诊断ON/OFF成本；不可把诊断关闭收益归为模型优化 |
| pinned/nonblocking 或预取深度可改善流水线 | 当前预训练未pin，微调调用nonblocking但loader未pin | H2D/等待占比，实际张量是否pinned，RSS/共享内存/FD |

旧缓存小样本读取 benchmark 不是当前模型端到端基线；不据此采用 chunk capacity=64，默认继续2。`profile_glt_dual_runtime.py` 当前未接入 static/targets 参数，直接运行会测另一条准备路径，应先补齐最小参数或复用已有 static benchmark 中的正确读取段。

## 3. 执行顺序与允许修改

### A. 先建立可比较的计时，不搭建通用框架

1. 核对上述 run/config 和实际缓存路径、checkpoint、split 存在；复用已有身份检查，不新增 hash/schema 体系。只在独立临时输出写入本轮执行配置及命令。
2. 在现有 runner/profile 增加可选、默认关闭的简短计时：启动/打开数据、准备等待、collate、H2D、forward/backward、optimizer、日志/诊断、保存；DDP报告最慢rank的update时间和各rank分布。
3. GPU分段用CUDA events且集中读取；只在计时窗口边界同步，不在每个算子插入同步。端到端墙钟是主指标；CPU/GPU重叠段不可简单相加。计时器开销需单独记录。
4. 复用预训练停止位置/诊断保存接口，但不能被迫开启所有重型诊断才能做轻量测速；可局部解耦短测控制与诊断采集。保留原正式保存行为和checkpoint格式。
5. 历史诊断ON运行、当前轻量诊断基线、候选分别列出，不能混算。若诊断降频，非有限检查、异常日志、数据有效计数和恢复信息仍保留；所有rank的collective分支一致。

### B. 优先实现微调 clean 输入的进程内复用

主要修改 `CleanLabeledDataset`，在微调入口提供可关闭的有限 CPU cache；不写磁盘、不缓存模型hidden或GPU tensor。

- 以同一 source 实例内的结构 key 复用**无标签、无变换统计**的 clean 图，按需构建；默认仅访问 train/validation，smoke 不预热 outer-test。
- 每次读取返回独立安全副本，再从当前 `targets[index]` 附加 y。禁止将 `.to(device)`、collate 原地改动或某折标准化标签回写到缓存对象。
- 相同结构的不同性质行仍为独立行；`set_target_override` 后立即读取新标签，不复用上一折 scaler/标签。
- 设置单进程内存上限（起点4 GiB，按实际tensor载荷和RSS同时观察），超限有明确回退和命中率记录；不按超限删样本。warm-cache速度和含首次构建总时间都报告。
- 第一版微调仍用worker=0，不同时叠加多worker和cache复制。通过后再决定是否需要pinning；其收益独立比较。

### C. 修复 grid 批次等待，保持 shard 隔离

- 用小型 pending/running 队列替换分批 barrier：任一GPU槽位进程成功退出，即派发同一槽位下一个任务。
- 默认每个唯一GPU一条shard，不新增同GPU并发；保留每task/fold独立进程、seed、模型、optimizer、日志和产物目录。
- 保留任务集合及待执行顺序，改变的只是启动时间；不得因本轮计划默认运行全部TASKS。
- 某shard失败即停止新增派发，记录失败及仍运行任务；不把其他任务算作成功、不遗留无人记录的进程。退出策略明确实现与测试，不终止本计划之外的进程。
- 保留现有不覆盖/部分产物拒绝策略；使用现有完成检查，不顺便扩建新的resume框架。
- 初次验收用可控子进程模拟不同耗时、失败、已有完成及部分产物；不为证明调度器而重跑正式fold。真实总工期收益待后续获授权任务记录，不用合成百分比冒充。

### D. 根据 A 选择至多两个预训练候选

先选有明确占比的一项，确认有效再选第二项，不全量穷举：

1. **准备路径**：向量化中心one-hop角度行筛选，或复用已有静态索引。`angle_pairs`只有端点，不能未经证明把它当作line行索引；输出必须保留原顺序、方向筛选及重复物理关系，不用canonical pair去重。clean target仍由冻结坐标得到，noisy输入仍重新计算；不能复用clean距离/角度当noisy输入。先做精确筛选/gather改动，不同时重构整个几何构建。
2. **预取/传输**：基线为3 workers/rank、prefetch4；在等待证据支持时仅试一个候选，例如同worker数prefetch2（降低驻留），或仅启用正确的pinned输入与nonblocking传输。前者与后者不能一开始捆绑。pin data和labels实际tensor，确认生命周期，不无界手工pin。

若诊断成本为首要问题，第二候选可替换为降低非必要诊断频率，而不是额外增加第三项。不删finite检查、不降低validation频率。调worker时控制父进程及worker的CPU线程预算并记录实际值；避免9个worker各自开满112线程。不清系统page cache、不执行swapoff。

## 4. 必要正确性验收

仅覆盖修改风险，复用现有fixture/测试；不重开缓存化学审计。

- clean cache ON/OFF：全部输入tensor、标签、key、collate顺序一致；同key不同标签、连续两次target override、重复访问及CPU→GPU不污染缓存；有限内存回退保持相同数据。
- 预训练准备ON/OFF：同(seed,key,position)的mask、噪声、clean/noisy特征、距离/角度/fingerprint、skip reasons及有效分母一致。整数/布尔/顺序要求exact；仅浮点计算次序改变时说明原因并采用预先固定容差，不按结果放宽。
- 包含普通样本、不同大小、无角度、N=0和无效几何；已有真实边界key可复用，缺失则明确用fixture，不无限寻找或扩大样本。
- 固定初始state及各rank RNG，检查短训练loss、梯度和更新权重；不只比较初始forward。DDP覆盖单rank/全部rank无有效几何的既有合法行为。
- 若改动数据预取、随机流或checkpoint控制：连续4步 vs 2步保存恢复到4步，验证样本位置、LR、mask/noise、RNG、optimizer及模型状态；DataLoader自己的generator应与模型dropout RNG隔离，恢复不得额外消耗公共随机流。不能以同seed代替该检查。
- grid：短任务完成后不等待长任务才派发；不重复task/fold、不会占用额外GPU槽位；失败/部分产物不被复用为成功。仅用mock完成文件测试不能代替实际子进程退出测试。
- 微调smoke保持train-only scaler、每epoch validation、相同best选择、outer-test=NOT_RUN；不为测速读test标签或生成OOF指标。

## 5. 有界验证预算（获授权后使用）

所有GPU、worker或超过一分钟任务在 `Uni-Poly` 独立window，禁止在当前交互窗口执行。建议窗口前缀 `speed_r1_`，日志 `logs/speed_20260917/`，独立产物 `results/speed_20260917/<时间戳>/`；不要复用历史训练输出。

| 层级 | 固定上限与方法 |
| --- | --- |
| 局部单元/子进程测试 | 仅新增或受影响测试；调度合成最多6个短任务，不启动模型 |
| 真实输入parity | 最多32个固定预训练key、32个固定下游行；覆盖不足用fixture，不生成构象 |
| GPU正确性短测 | 最终预训练候选包baseline连续4步、candidate连续4步、candidate2+恢复2步；最多12个逻辑update，最多4 ranks；必要2-step故障定位最多一次 |
| 预训练测速 | baseline与至多2个单因素候选；每个比较A-B-B-A，每次5步warmup+15步计时；最多160个optimizer updates；不缩放原LR horizon，不让候选接续baseline权重训练 |
| 微调测速 | 固定xc、fold0；baseline与clean-cache候选A-B-B-A，各最多2epochs，既有smoke模式，只用train/validation；最多8个训练epochs |
| 可选组合确认 | 仅单因素已有效：预训练baseline/组合各1次20updates；不新增候选，微调pinning若要新增实测则留下一轮 |
| 总资源边界 | 上述上限和总墙钟2小时先到者停止；GPU最多4张并只跑一个测速组，数据CPU测量最多15分钟；临时产物最多10GiB，不删除旧产物凑预算 |

A-B-B-A每次独立进程、相同初始化/恢复点、相同绝对样本位置。报告冷启动和稳定窗口；不声称OS缓存被清空。baseline与candidate启用相同计时和日志设置，若测试诊断降频则单列为受控改变。

预训练主指标：最慢rank端到端seconds/update、1008/seconds/update的真实samples/s、p50/p95及等待占比。微调主指标：含首次准备、scaler、train、validation、best权重复制/保存的两epoch总时间；另报warm输入epoch时间，不能只取cache命中段。

每次同时记录GPU峰值、进程树RSS/PSS（可用时）、FD、共享内存、CPU线程、GPU利用率、实际样本/步数和命令。低频资源采样10–30秒，人工状态检查5–10分钟或阶段结束；不恢复高频监控。

## 6. 晋级、停止与交付

- 正确性通过是前提。工程筛选起点：配对端到端中位耗时降低至少10%，两次配对方向一致；p95无明显恶化（超过5%需解释/复查），且无持续资源增长。该阈值不是统计显著性或模型性能保证。
- 低于阈值或方向不稳：记录“收益不足/证据不足”，保持baseline，不为了达到阈值追加sweep。调度修复可按无barrier且任务隔离验收，但实际训练总工期收益仍标未验证。
- OOM、共享内存/FD不足、持续换页、NaN/Inf、样本/target不一致、不同rank collective分支、覆盖正式产物风险：停止受影响步骤，做授权范围内最小定位；不通过改变batch、过滤样本或关闭合同检查绕过。
- 默认开关是否采用由Codex依据报告审查；尚未测出的候选保持关闭。只读计时不足以宣布提速完成，单元通过也不能宣称模型预测性能提升。

交付内容：最小代码diff、相关测试及命令/退出码、每个比较的原始时间/资源记录、parity与resume结论、优化开关/回退用法、未执行项。直接复用JSON/日志，不增加通用benchmark平台、数据schema或迁移器。

建议修改范围：`src/training/glt_dual_runtime.py`、`scripts/finetune_glt_dual.py`、`scripts/run_glt_dual_finetune_grid.py`、`scripts/pretrain_glt_dual.py`；按profile必要性涉及 `src/dataset/glt_dual_pretrain.py`、`scripts/profile_glt_dual_runtime.py` 或诊断代码。共享 `src/utils.py` 若需修改，必须默认保持其他路线行为不变；模型数学实现不在范围内。

实施影响CLI/流程时再更新 `PIPELINE.md` 的对应段落（其中旧预训练模板及“无workers”描述不能覆盖当前实际CLI）。不改历史 `RESULTS.md` 数字。ZCode将执行记录写入本文件，Codex审查后在 `PROJECT_HISTORY.md` 归档实际周期；通过后也不自动启动正式预训练/全任务微调。

## 7. 当前执行记录与下一步

- 本轮完成：检查 Git/当前源码、两类历史 run.json、配置、机器资源与相关进程；确认 4 张 RTX 4090 可作为本轮有界 GPU 测速资源；此前未实施 A–D。
- 当前阶段：A→B→C→D 已执行；结果与未核实项写入下方执行记录，不能把计划目标冒充为结果。
- 文档自检：相关路径、授权/预算/科学不变项及 `git diff --check`；实际提交和推送信息见本次交付及Git历史。
- 下一步：等待 Codex 独立审查；4 GPU 仅用于本计划允许的短测，不启动正式重训或完整微调。

### 7.1 本轮执行启动记录（2026-09-17 UTC）

- 执行前核对：`git status --short --branch` 为 `dev...origin/dev` 且工作树干净；`git pull --ff-only origin dev` 返回 `Already up to date`。
- 当前无匹配的预训练、微调、cache builder、`torchrun` 或 pytest 活动进程；历史 `Uni-Poly` windows 保留，不复用其输出或覆盖产物。
- 资源核对：4 张 RTX 4090、112 logical CPUs、约 179 GiB available RAM；GPU 短测最多使用 4 张，仍遵守本计划的短测与总墙钟边界。
- 当前状态：启动记录完成；尚未宣称任何候选通过、正式提速或模型性能改善。

### 7.2 有界执行结果（2026-09-17 UTC）

- **代码修改**：`CleanLabeledDataset` 增加默认关闭、按 tensor payload 计量的进程内 LRU（smoke 使用 4 GiB）；`pretrain_glt_dual.py` 增加可选 `--timing`，并支持 `--prep-workers` 恢复时在 DataLoader iterator 建立后恢复 RNG；`profile_glt_dual_runtime.py` 支持 static/target sidecar；中心 one-hop angle 行筛选改为保持顺序与物理关系多重性的向量化 gather；finetune grid 改为 free-slot 动态派发，并对 shard 退出与 `launch()` 异常统一收口；新增 `tests/test_glt_dual_speed.py`。
- **局部验证**：`Uni-Poly:speed_r3_final_tests` 执行 py_compile 与
  `PYTHONPATH=.:tests pytest -q tests/test_glt_dual_speed.py tests/test_glt_dual_static.py tests/test_dual_glt_pretrain.py`，`18 passed, 1 warning`，退出码 0，日志 `logs/speed_r3_final_tests.log`。此前同一集合为 `18 passed`（`logs/speed_r2_grid_cleanup_tests.log`）；一次误用不存在测试文件名的命令退出码 4，未作为通过证据。
- **真实输入 parity**：PI1M 固定 32 条真实记录的 old runtime 与 static/target runtime 准备结果通过（数据字段、mask/noise、距离/角度、fingerprint、skip reasons；整数 exact，浮点 `1e-6`），报告 `results/speed_20260917/parity_32.json`，日志 `logs/speed_r2_parity32_retry.log`，退出码 0。下游 `xc` 固定 32 行 clean 输入通过同样比较，报告 `results/speed_20260917/downstream_parity_32.json`，日志 `logs/speed_r2_downstream_parity32.log`，退出码 0。第一次同进程双 LMDB 打开尝试被 LMDB 正确拒绝（`logs/speed_r2_parity32.log`，退出码 1），重排为先关闭旧 source 后复跑。
- **CPU profile**：256 条固定间隔 PI1M 记录 old/static 均为 read-only zero-write PASS。旧路径 clean `median=53.288 ms`、prepare `median=91.173 ms`；static/target 路径分别为 `0.792 ms`、`1.738 ms`，报告 `results/speed_20260917/profile_{baseline,static}_256.json`，日志 `logs/speed_r1_cpu_profile256.log`，退出码 0。5000 条 profile 在 15 分钟 CPU 预算到达时停止，未产生报告，不能外推完整吞吐。
- **微调 A-B-B-A smoke**：`Uni-Poly:speed_r1_finetune_xc` 在 `xc/fold0`、2 epoch、train/validation-only 下四次均退出码 0，`outer-test=NOT_RUN`；baseline 两次 epoch 总时长约 `59.84/59.75 s`，cache 两次约 `10.52/10.76 s`，两次验证 R² 轨迹均为 `0.0590856084 → 0.1038316778`。日志 `logs/speed_r1_finetune_xc.log`，产物 `results/speed_20260917/finetune_xc/`。这只是 smoke/timing，不是性能或下游结论。
- **4-GPU预训练 bounded smoke**：`Uni-Poly:speed_r1_pretrain_4gpu` 使用 `torchrun --nproc_per_node=4`，baseline 4 步、static/target candidate 4 步、candidate 2 步保存后恢复 2 步，均退出码 0；四 rank `valid_graphs=1008`，loss、target counts、梯度范数均有限，candidate 与 baseline 四步 loss/target 完全一致，candidate 连续与恢复的 model/optimizer/scheduler/step/ordered keys 完全一致。产物 `results/speed_20260917/pretrain/{baseline4,candidate4,candidate_resume}/`，日志 `logs/speed_r1_pretrain_4gpu.log`。初始恢复比较发现带 3 workers 时 checkpoint Torch RNG 不一致；已将恢复点移到 DataLoader iterator 建立之后并通过局部测试，但受 12-update 预算限制**未复跑修正后的 GPU 恢复集成**，因此 RNG exact 仍标为未核实。
- **调度失败路径**：18 项局部测试包含 free-slot 先派发、子进程失败后停止新派发，以及 `launch()` 异常时 drain 已持有子进程；未启动正式 fold。
- **未执行**：正式 5000/20000-step 预训练、Concat/KFuse 完整训练、完整 8×5 微调、正式 grid、5k+15-step 长 benchmark、第二预训练候选、缓存重建/迁移、post-fix GPU resume rerun。当前不把 smoke 或静态 sidecar parity 写成模型性能提升或正式实验完成。
