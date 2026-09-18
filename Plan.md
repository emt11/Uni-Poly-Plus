# GLT-V2 工程提速最终执行方案

## 0. 交接头与执行指令

| 字段 | 内容 |
| --- | --- |
| 计划 | GLT-ENGINEERING-20260918-01 / r1 |
| 状态 | 待审查；S0–S5 已执行，S6 交接材料已生成，等待 Codex 独立审查 |
| 用户要求 | 给出其他模型可直接执行的完整工程方案；完成并验收本方案后，才另行考虑预测性能优化 |
| 范围 | GLT 双通道的数据准备、诊断、计时、输入复用、传输与调度；不优化预测准确率 |
| 角色 | Codex 规划与独立验收；接手模型负责实现、局部测试和规定的有界测速，并记录实际执行者 |
| 核查基线 | dev，执行前为 `6f7c877`；2026-09-18 pull 为 Already up to date；执行期间仅修改本计划与本轮列出的源码/测试文件；此前计划已写入本文件 |
| 既有工作 | CACHE-20260916-01/r4、SPEED-20260917-01/r4 均已归档；不重做已完成工作，不复活旧科学计划 |

**给执行模型的指令：**先完整读取 AGENTS.md 和本文。收到用户“执行本计划”的授权后，依次完成 S0→S1→S2→S3→S4→S5→S6；普通实现细节自行处理，不重复询问已授权的小步骤。不要只输出另一份计划。受阻则停止受影响阶段、保留证据，继续不依赖阻断项的代码/测试工作；不得偷偷更换协议、缩小验收集合或扩大预算。最终状态只能先标“待审查”，由 Codex 审查后关闭。

本文中的新增接口均明确标为“拟新增”，实施前不能直接运行。已有 CLI 模板使用已核实参数。文档本身不替代执行授权。

## 1. 本轮只做什么、绝不做什么

### 1.1 唯一目标

在相同模型、数据和训练工作量下，降低预训练每个 optimizer update 的墙钟时间、微调完整短任务耗时和多任务调度等待，给出可复现的启动命令、保守推荐值和关闭优化的回退命令。

吞吐、时间、内存、I/O 和正确性是本轮验收指标。loss、validation 输出只作数值一致性与异常检查，不作为选模型、调参或科学结论依据。

### 1.2 冻结不变项

- 当前架构 `O8-BondPath-GalformerTrimer-Hop2`；主比较固定 Concat、无 MD200，无教师/蒸馏。
- O8/GLT 均6层、512维、8 heads；2D路径、3D物理键与1/2-hop关系、中心readout、attention方向/缩放均不改。
- 不改三任务目标、权重1/1/0.1、30%mask、0.03 Å噪声、geonorm科学定义、参数初始化、optimizer、LR、scheduler、dropout、样本顺序及各rank分配。
- 预训练固定3 ranks×84×accumulation4=1008、BF16、seed42、原5000-step配置及20000-step scheduler horizon。短测提前停止，不改原配置以伪造更短schedule。
- 微调固定FP32、train32/eval64、原LR/weight decay、train-only scaler、outer5_inner20；验证频率和选择规则不改。短测沿用已有 `--smoke` 的两epoch语义，不宣称复现100epoch完整轨迹。
- 无效几何、单原子、N=0、无角度、重复物理关系、有效图分母全部保持；不会因提速丢弃大图/异常样本或改变cohort。
- 不修改active缓存/manifest/`.frozen`，不生成构象、不全量重建/迁移、不清理历史产物、不改变mmap默认容量2。

### 1.3 明确排除

不做预测性能优化、模型结构/融合/readout修改、损失调权、扭转/非键输入、多构象、任务采样、精度或batch改动、torch.compile/CUDA Graph/attention替换、DDP `find_unused_parameters` 修改。不得运行正式5k/20k、8×5、outer-test/OOF，不新增seed，不启动或干预EQ3D及其他路线。

**本计划全部完成且审查通过，仅表示工程阶段结束；不会自动授权下一阶段预测性能研究。**

## 2. 已存在的功能和证据，不重复开发

| 项目 | 当前事实 | 本轮处理 |
| --- | --- | --- |
| static/target读取 | 已有，历史fixed-geonorm正式run已使用 | 所有主对照均开启，不拿无static慢路径冒充当前baseline |
| clean CPU Data LRU | 已有，`--clean-cache-gib`默认0，返回独立副本并附当前标签 | 做相同static条件下的0 vs 4 GiB配对比较 |
| 动态grid | 已有空闲槽位补位、重复GPU拒绝和失败收口 | 复用，补有界真实smoke调度观察，不重写 |
| 中心角度筛选 | 已向量化 | 不再次作为新优化计数 |
| BF16/no_sync/预取 | 已有，正式run为每rank3workers | 保持，不把从0workers改3算成本轮收益 |
| 恢复RNG | 已有worker恢复修复；旧4GPU恢复报告PASS | 本轮若改诊断/准备/运行控制，在锁定3GPU上局部确认 |

旧证据路径：`results/speed_20260917/`、`logs/speed_r4_scoped_tests.log`、`PROJECT_HISTORY.md` 的 SPEED归档。

必须保留的解释限制：旧xc两epoch约60秒→10.5秒，同时改变static和clean cache；不是cache单因素收益。旧 `epoch_seconds` 不含启动及best权重复制/最终保存。256条CPU profile没有把static/target读取纳入准备计时。旧4GPU正确性smoke不是3GPU正式吞吐基线。本轮报告只追加新结论，不覆盖旧JSON/日志。

## 3. 环境、固定输入与输出

工作目录 `/root/workspace/Uni-Poly-Plus-master`；当前Python `/opt/conda/bin/python`，执行前确认其环境可用，不安装/升级依赖。GPU需要3张空闲且没有已登记即将占用的设备；不足时等待用户协调，不擅自换world size、挤占或kill其他任务。

| 用途 | 固定路径 |
| --- | --- |
| 预训练配置 | `configs/mts/glt_dual_three_task_concat_geonorm.json` |
| 预训练cohort | `data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1` |
| 预训练基础缓存 | `data/processed/mips_trimer_scage` |
| static / target | `data/processed/glt_dual_v2/pi1m/dual_static_v1` / `pretrain_targets_v1`（同级） |
| 微调配置 | `configs/mts/glt_dual_three_task_concat.json` |
| 微调部署包 | `results/glt_dual_static_pretrain_5k_concat_geonorm/deploy_05000.pt` |
| 微调raw / split | `data/raw` / `data/splits/mips_outer5_inner20` |
| 微调cohort | `data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1` |
| 微调基础缓存 | `data/processed/mips_trimer_scage_downstream` |
| 微调static | `data/processed/glt_dual_v2/downstream/dual_static_v1` |

复用现有身份/strict-load检查，缺失或不兼容立即报告，不猜替代路径、不新增全量hash机制。

每轮使用新的 `results/glt_engineering_20260918/<UTC时间戳>/` 和对应 `logs/glt_engineering_20260918/<UTC时间戳>/`，不覆盖历史目录。允许用shell新建输出目录；脚本和文档用apply_patch编辑。

GPU、worker、超过一分钟及写训练产物的命令只能放在 `tmux` session `Uni-Poly` 独立window（前缀 `glt_eng_`）。先查session/进程/自动后续任务，记录cwd、window、设备、完整命令、日志、输出和退出码。stdout/stderr保留到日志；使用 `tee` 时检查真正任务的退出码，不能把tee成功当作训练成功。

资源监控每30秒一次，人工状态检查5–10分钟或阶段结束；失败即时汇报。不清系统page cache，不执行swapoff，不让监控本身占主要时间。

## 4. 分阶段实施合同

### S0 — 同步与基线固定

1. 检查git status、branch、remote，按AGENTS pull，再读最新代码与本文件。保护用户改动，不stash/reset/恢复空Plan来覆盖别人工作。
2. 核对第3节路径及上述现有功能；执行提交相对 `f1dfd4d` 有变化时记录相关diff。不要求HEAD永远等于规划基线，但科学接口有变须暂停确认。
3. 检查现有进程、GPU、RAM、共享内存、磁盘与fd限制。其他自动训练循环可能重新启动任务，不能只看某一秒GPU为0%。确认专用测速窗口后再占设备。
4. 写入本轮 `execution.json`：实际baseline commit、解释器/库版本、设备、原配置、输入路径和既有身份、输出目录、预算计数器。复用现有字段，不新建schema或运行框架。

**完成条件：**基线可加载、设备可独占测速、输出隔离明确。无资源时CPU代码/测试可继续，GPU阶段标阻断，不声称完成。

### S1 — 修正计时口径（必做）

主要文件：`scripts/profile_glt_dual_runtime.py`、`scripts/pretrain_glt_dual.py`、`scripts/finetune_glt_dual.py`。共享utils只在确有必要时改，默认行为不能影响其他路线。

1. CPU profile逐项计入source读取、static读取、target读取、clean/noisy准备、collate及完整样本时间。使用固定seed的随机256个不同索引；保存实际key/顺序，所有路径使用同一集合。记录打开数据耗时、p50/p95、图大小/关系数、资源；不遗漏移到计时边界外的工作。
2. 预训练增加**拟新增** `--benchmark-steps N --benchmark-warmup W`：N为本次实际执行总updates，W为其中不计入稳定窗口的前W步；0<=W<N，不能超过原配置剩余步数。允许不打开重型diagnostics，不改变原配置、LR horizon、有效batch或样本流。默认关闭；与旧stop接口同时指定时报错；benchmark正常结束不写正式deploy，不触发5000-step完成标志。
3. benchmark稳定窗口在所有rank就绪后计时，窗口末同步并报告最慢rank墙钟、实际图数/updates及samples/s。每rank的逐步耗时用于分布诊断，最终聚合一次，不为了每步计时新增一串all_reduce。逐步异步CPU计时必须注明，不能冒充GPU独立kernel时间。
4. 保留旧 `--timing` 为有同步扰动的诊断工具，正式ABBA测速不用它。若需GPU分段，用少量CUDA events集中读回；profile段与正常吞吐段分开报告。step/窗口时间包含实际日志开销；保存耗时另计，不偷偷移出全流程总时间。
5. 微调增加从入口至全部产物保存结束的内部总时间；外层记录子进程启动到exit的总墙钟。保留train/validation段，并单列source打开、首批准备、best权重CPU复制和最后保存。clean-cache冷启动成本必须在总时间中。
6. 输出 `runtime.json`（运行参数和资源）与 `benchmark.json`（计时和计数），允许复用已有JSON扩展字段；诊断日志不能依靠从交错DDP stdout提取JSON，写每rank独立机器可读记录，最终rank0汇总。

**验收：**计数/计时相关合成测试通过；旧CLI无新开关时保持行为；profile不写冻结缓存；配置max_steps/scheduler未被短测参数改写。

### S2 — 非必要诊断按步采集（必做，第一预训练候选）

主要文件：`scripts/pretrain_glt_dual.py`、`src/modules/glt_dual_pretrain.py`、相关测试。

1. 增加**拟新增** `--diagnostics-every N`（正整数，默认1），只在 `--diagnostics` 存在时有效；采样绝对update为首步、N的倍数及明确诊断保存步。所有rank、同一update内所有accumulation microbatch使用一致开关。
2. 未采样步必须真正跳过hook、quantile/RMS等详细统计及模块梯度统计，不只是少写JSON。保留所有更新、全局loss/有效计数、finite/gradient检查和错误输出；本轮不同时降低普通loss日志频率。
3. 不改变模型参数/buffer集合、forward返回的训练sums/counts、RNG和optimizer。清除或标记旧 `last_diagnostics`，禁止把上次统计写成本step结果。rank-local最后microbatch统计与全局loss分开标记，不伪称全局统计。
4. 候选固定every20，baseline every1；同样的3workers、static/targets、计时和保存规则。收益明确称为诊断开销节省，不归因为模型计算加速。

**验收：**诊断ON/OFF及1/20的固定输入forward、梯度和短轨迹满足第6节；未采样步确实没有详细统计调用；所有rank同步策略不变。

### S3 — 微调独立对照与调度收口（必做）

不重写CleanLabeledDataset或动态grid。只补必要计时、发现真实缺陷时局部修复。

1. 固定geonorm部署包、static开启、xc/fold0、两epoch；A=`--clean-cache-gib 0`，B=`--clean-cache-gib 4`。按A-B-B-A共4个独立进程运行，同设备和环境，不接续权重，不改变数据顺序。
2. 比较完整子进程时间、两epoch内段、cache命中/淘汰、RSS/PSS。baseline和candidate的validation输出、best epoch及最终权重仅用于一致性核对；不得按validation高低选开关。测试不读取outer-test。
3. 调度验证固定4个smoke单位：`[(xc,0),(eat,0),(xc,1),(eat,1)]`，每单位2epochs，固定两张GPU，单GPU一进程。建立最多一个简单辅助脚本，调用现有 `_run_dynamic_jobs`；child始终使用 `--smoke`，不要用生产grid的 `--formal-shard` 冒充smoke。每单位static开启、cache4 GiB；A为仅测试用的旧式两槽分批等待，B为现有动态补位，各运行一次，输出隔离。
4. A/B单位集合、命令、seed、预算一致；记录实际启动/结束及GPU空闲时间，失败停止新派发并收口自己创建的进程。此小样本只验证真实smoke能调度，不外推正式8×5收益，也不要求为了获得更好百分比反复重跑。

**验收：**S3.1无static混杂、计时完整；S3.3八个子进程全部合法smoke退出且outer-test=NOT_RUN，无重复单位/漏跑。真实grid若没有可观察尾部空闲，记录收益不足，不扩任务。

### S4 — 至多一个额外预训练候选（条件执行，不穷举）

先完成S1实测。选中项、占比和理由写入Plan执行记录，再实现。按以下决策，不让接手模型自由扩成多项实验：

| 证据 | 唯一允许的候选 | 限制 |
| --- | --- | --- |
| CPU准备在消费等待中显著，clean完整物化/重复字段组装是其可定位热点 | 将clean target提取与noisy输入组装分离；复用已验证静态索引，减少clean整图复制 | 不新增磁盘缓存；旧分支保留为默认reference，拟新增 `--prepare-mode legacy\|targets_only` 默认legacy |
| 等待主要与H2D相关，准备不是主要热点 | 对已有Data/labels正确pin，并nonblocking传输 | 拟新增 `--pin-input-memory` 默认false；worker仍3、prefetch仍4，不同时调参 |
| 主要在source/static随机读取、GPU算子，或证据不足 | 本轮不新增候选，记录暂缓原因 | 不改缓存格式/容量、不换attention、不乱加worker |

若两项都满足，选占完整update比例更高且预计可节省时间更大的一个；只可择一。优化的受影响段不足总时间10%时原则上不实施，避免用很小局部提升换取复杂度。

`targets_only`必须保留clean distance/cosine target、顺序和多重性；不能把 `angle_pairs` 直接当成line row indices，不能缓存带可学习参数的Gaussian embedding，不能将clean几何送入noisy输入。严格比较全部tensor与分母。

候选无收益或验证失败则关闭，保留baseline并记录否决；不替换另一个候选继续试，不追加第三条路径。

### S5 — 预训练公平测速与最终组合确认（必做）

主矩阵：

| 比较 | A | B | 顺序和预算 |
| --- | --- | --- | --- |
| 诊断候选 | static/targets＋3workers＋every1 | 完全相同，只every20 | A-B-B-A，4×30updates，每次前10warmup、后20计时 |
| S4候选（仅实施时） | 原准备/传输路径、固定every20 | 仅S4单项改变，every20 | A-B-B-A，4×30updates；不能把诊断收益重复计算 |
| 最终确认 | 本轮reference：every1＋legacy＋不pin | 通过正确性及单项收益检查的组合 | A/B各30updates；即使只诊断入选也记录一次最终确认 |

每次从相同随机初始化开始，固定seed/key/position、同一3张GPU、相同参数更新顺序与线程环境；绝不在candidate上接着baseline训练。样本来自完整固定cohort的原采样流，不为测速挑小分子。最终确认没有候选入选时明确跳过，保留baseline。

测速运行不写大checkpoint、不启用旧同步式 `--timing`；correctness/resume阶段另存必要checkpoint。baseline和candidate普通日志频率一致。窗口起止同步和整体进程时间都报告；不把有初始化成本的进程墙钟与steady-state samples/s混算。

若多GPU其他任务导致负载变化，本次配对作无效记录；预算允许时最多重做一次受污染单run，仍占总updates上限，否则判证据不足。不得拿历史3GPU和本次4GPU比较，也不以CPU-only profile推算正式吞吐倍数。

### S6 — 推荐配置、文档和独立验收

1. 汇总 `report.md` 与机器可读 `summary.json`：基线、精确改变项、每次原始时间、配对差值、资源、正确性结果、收益成立/不成立/未验证、未执行及原因。保留失败原始日志。
2. 推荐仅写成显式启动参数，不修改现有科学JSON或悄悄切全仓默认：clean cache4、诊断every20或S4候选仅在本轮证据支持时列为建议；未入选项保持默认。
3. 给出一份完整已验证的预训练短测命令、一份微调smoke命令、一份未来正式启动模板（明确“未执行，需另行授权”），及关闭优化的baseline命令。候选旧checkpoint能否恢复及部署包兼容结论写清楚。
4. 更新PIPELINE对应CLI/计时/推荐参数，并纠正已过期的samples-csv/topology-root示例；不改历史RESULTS数值。执行记录写回本文件。按AGENTS显式暂存负责文件、commit/push并验证远端。
5. Codex独立审查diff、原始日志和产物；审查通过后归档PROJECT_HISTORY，Plan标“已完成；暂无后续执行”。预测性能研究留待用户另行授权，不在本报告列可自动执行的科学改造任务。

## 5. 命令模板与输出合同

以下命令在第3节规定的tmux/log承载下执行。`RUN_ROOT`、`RUN_OUT`、`GPU_SET`、`FT_GPU`必须由执行者设置为本次新输出绝对路径、具体三GPU列表和单GPU编号，且写入日志；不得用HOME/CODEX_HOME作临时变量。每个run使用全新RUN_OUT。

已有CPU profile入口（先完成S1计时修正）：

```bash
/opt/conda/bin/python scripts/profile_glt_dual_runtime.py \
  --cache-root data/processed/mips_trimer_scage \
  --cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1 \
  --dual-static-root data/processed/glt_dual_v2/pi1m/dual_static_v1 \
  --pretrain-target-root data/processed/glt_dual_v2/pi1m/pretrain_targets_v1 \
  --samples 256 --seed 42 --report-json "$RUN_ROOT/profile.json"
```

**S1/S2实现并验证help后才可执行**的预训练模板（A every1，B改20）：

```bash
CUDA_VISIBLE_DEVICES="$GPU_SET" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/opt/conda/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/pretrain_glt_dual.py \
  --config configs/mts/glt_dual_three_task_concat_geonorm.json \
  --cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1 \
  --cache-root data/processed/mips_trimer_scage \
  --dual-static-root data/processed/glt_dual_v2/pi1m/dual_static_v1 \
  --pretrain-target-root data/processed/glt_dual_v2/pi1m/pretrain_targets_v1 \
  --prep-workers 3 --diagnostics --diagnostics-every 1 \
  --benchmark-steps 30 --benchmark-warmup 10 --no-deploy --output "$RUN_OUT"
```

已有微调模板（A cache0，B只改4；所有组保留dual-static-root）：

```bash
CUDA_VISIBLE_DEVICES="$FT_GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/opt/conda/bin/python scripts/finetune_glt_dual.py \
  --config configs/mts/glt_dual_three_task_concat.json \
  --checkpoint results/glt_dual_static_pretrain_5k_concat_geonorm/deploy_05000.pt \
  --raw-root data/raw --split-root data/splits/mips_outer5_inner20 \
  --cohort-root data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1 \
  --cache-root data/processed/mips_trimer_scage_downstream \
  --dual-static-root data/processed/glt_dual_v2/downstream/dual_static_v1 \
  --task xc --fold 0 --smoke --timing --clean-cache-gib 0 --output "$RUN_OUT"
```

必要局部测试基础入口为 `PYTHONPATH=.:tests /opt/conda/bin/python -m pytest -q tests/test_glt_dual_speed.py tests/test_glt_dual_diagnostics.py`；如S4修改准备路径，再选择test_glt_dual_static中直接相关的测试，不默认跑全仓或全缓存恢复测试。新增测试放现有文件或至多一个针对本轮的测试文件；允许一个薄的执行/汇总辅助脚本，不建设通用任务平台。

每次运行至少保存完整命令、exit code、配置/输入引用、真实执行步数/样本数、计时、资源及状态；报告必须能定位每个数字的run，不只有截图或平均值。产生checkpoint的correctness/smoke目录与无checkpoint测速目录分开。

## 6. 正确性验证与数值规则

1. 从旧parity manifest复用最多32个PI1M key和32个下游行，不新增随机大审计；在当前改动路径上比较输入、collate、target和有效分母。缺N=0/无角度/invalid等真实边界时明确使用已有fixture，不能静默skip。
2. 诊断和cache等不改变tensor算术的改动，CPU输入/整数/mask/顺序/RNG要求exact。GPU受既有非确定性影响的浮点项使用原报告有据可依的容差；若旧报告未定义，预登记FP32 atol=rtol=1e-5、BF16路径浮点loss/gradient/state atol=rtol=1e-4。对同配置A-A都超过阈值的情况先定位非确定性，不自动放宽阈值，不将其算作candidate通过。
3. 同一初始化baseline连续4步、最终candidate连续4步，逐步核对sample position、mask/noise、loss有效计数、LR、loss、gradients及更新权重。candidate再2步保存并恢复至4步，与candidate连续轨迹比较模型、optimizer、scheduler和每rank RNG。baseline/candidate可能有不同运行时开关，但检查点核心参数和数据身份不应静默失配。
4. 比较诊断采样前后不使用不同模型seed；resume在worker迭代器建立后恢复模型公共RNG的现有正确顺序保持不变。任何新的loader generator不得重新改变历史连续baseline轨迹。
5. DDP synthetic fixture覆盖部分rank及全部rank无几何目标，最多各一次forward/backward；保留原任务有效分母与collective，不改unused-parameter设置躲错误。
6. clean cache重复key不同label、target override、clone/move不污染、容量淘汰；pin候选则验证data和labels确实pinned且没有异步生命周期问题。新候选必须经过实际消费路径，不能只测helper函数。
7. static字段、clean/noisy目标、N=0和loss分母的差异视为正确性失败。允许候选被否决、关闭；不允许用速度收益抵消正确性失败。

顺序说明：每个候选进入S5前先完成对应局部测试、fixture的forward/gradient比较及受影响真实输入parity；第3项的12-update完整轨迹/恢复检查在最终候选组合确定后完成。首次发现数值异常即停，不以“后面还有恢复测试”为由继续测速。若需额外A-A GPU轨迹来定位非确定性，只能从第7节未使用的updates额度内调配并记录；没有剩余额度则报告阻断，不自行追加。

## 7. 总预算与停止条件

| 项目 | 上限 |
| --- | --- |
| CPU完整profile | 固定256条、1次；失败修复后最多重跑1次，总CPU profile墙钟不超过20分钟 |
| 真实parity | PI1M最多32key、下游最多32行；fixture补边界，不构建新缓存 |
| GPU分段诊断 | 最多4个预训练optimizer updates，独立于吞吐结果 |
| 预训练单因素测速 | 诊断120updates；S4如实施另120updates |
| 最终组合确认 | 最多60updates |
| 训练正确性/恢复 | baseline4＋candidate4＋candidate2/恢复2，共12updates |
| 预训练总上限 | 316个实际optimizer updates（上述4+120+120+60+12）；失败/污染重跑也占此上限 |
| 微调cache比较 | 4runs×2epochs=8epochs，仅xc/fold0，NOT_RUN outer-test |
| 真实调度smoke | 4单位×2策略×2epochs=16epochs，仅eat/xc folds0/1，NOT_RUN outer-test |
| 微调总上限 | 24epochs；不追加seed/task/fold |
| 资源 | 最多3GPU同时使用；测速组彼此不并行，其他模型任务不与测速共享设备 |
| 总限额 | 长任务累计墙钟3小时、独立产物20GiB；任一先到即停，不能自动扩预算；等待资源不计训练预算但须报告 |

RAM可用低于32GiB、共享内存持续超过75%、出现持续换页或FD耗尽风险，停止新增任务并记录；不以释放别人的内存或删除别人的文件解决。GPU OOM、NaN/Inf、身份不匹配、通信挂起、预计覆盖正式目录时立即停止受影响run。不要用改batch、删样本、清缓存重建或跳过检查恢复运行。

必要测试失败先局部定位；无法在预算内完成则列出已执行、未执行和阻断。不因“最终方案”四个字强行把不完整证据写成通过。

## 8. 晋级、完成定义与下一阶段边界

### 8.1 优化项判定

- 必须先通过正确性。性能工程采用参考：两个配对方向一致且完整相关耗时中位数下降至少10%，资源无持续增长，p95无明显恶化（超过5%需解释）。小样本阈值不当作统计显著性。
- 未达到阈值：标“收益不足/证据不足”，默认不采用；不追加sweep。诊断采样若收益不足仍可作为可选调试功能，但不能宣传提速。
- 报告区分steady-state预训练速度、微调完整子进程时间、动态调度makespan；不能相乘推算一个未经测量的总加速倍数。
- 参数启用推荐由本轮证据决定。baseline回退必须可运行，历史checkpoint/schema兼容结论清晰。

### 8.2 整个方案完成须同时满足

1. S0–S3全部有实际交付；S4执行或依决策表给出有证据的暂缓/否决；S5的实际适用比较和最终确认齐全。
2. 相关局部测试、真实parity、短轨迹及恢复通过；调度真实smoke无漏跑/重复，outer-test均NOT_RUN。
3. 有不混淆冷启动/稳定期、static/cache、诊断/计算的时间与资源报告，且每项有采用/不采用结论。
4. 最终启动/回退命令、PIPELINE、执行记录、commit/push完成，并由Codex审查通过。
5. 所有失败与未验证边界如实保留。若全体新候选均无收益但检查齐全，可称“工程评估完成，保持baseline”，不能称“已实现提速”。必要验证受阻则不能关闭整个计划。

完成后明确写：**工程阶段已验收；本轮未优化、未比较模型预测性能；暂无后续执行。** 即使validation偶然更好，也不产生预测性能结论。后续科学阶段必须另立计划并取得授权，不从本轮自动启动。

## 9. 执行记录（接手者填写，不能提前填PASS）

| 阶段 | 状态 | 实际commit/命令/window | 日志/报告/退出码 | 偏差/预算累计 |
| --- | --- | --- | --- | --- |
| S0 基线与环境 | PASS（执行者自检，待 Codex 审查） | `6f7c877`；`git pull --ff-only origin dev`；环境核对命令；无独立长任务 | `results/glt_engineering_20260918/20260918T000000Z/execution.json`；资源与输入路径均存在 | GPU 0–3 空闲（仅查询）；未占用设备；预算未消耗 |
| S1 完整计时 | PASS（执行者自检，待 Codex 审查） | `scripts/profile_glt_dual_runtime.py`；256条 CPU profile；`glt_eng_profile_003509` | `results/glt_engineering_20260918/20260918T003509Z/profile.json`；`logs/glt_engineering_20260918/20260918T003509Z/profile.log`；退出码0 | 首次字段名错误留存于 `003412Z`，修正后仅重跑1次；冻结主缓存零写入；profile预算1/1 |
| S2 诊断采样 | PASS（执行者自检，待 Codex 审查） | 33项局部测试；3-GPU baseline4/candidate4/resume2→4，共12 updates；`glt_eng_corr_*` | `results/glt_engineering_20260918/20260918T003654Z/correctness.json`；对应4份日志；退出码均0 | model/optimizer/scheduler/keys/position/rank RNG exact；候选诊断步为1/2/4；无正式checkpoint/deploy |
| S3 微调/调度 | PASS（执行者自检，待 Codex 审查） | cache A-B-B-A 4×2 epochs；两GPU batched/dynamic 各4单位×2epochs；`glt_eng_ft_*`、`glt_eng_grid_*` | `results/glt_engineering_20260918/20260918T005342Z/`、`20260918T005721Z/grid_smoke.json`；所有子进程退出码0 | cache4两配对约40% wall-clock收益且验证 exact；调度 makespan差约0.9%，不宣称收益；outer-test全NOT_RUN |
| S4 单候选或暂缓 | 暂缓（执行者自检，待 Codex 审查） | 依据256 profile与S5 ABBA结果，不新增 `targets_only`/pin 候选 | 结论写入 `results/glt_engineering_20260918/summary.json` | source读取/随机长尾主导；诊断every20配对方向不一致且未达10% gate；保持legacy/default |
| S5 公平测速 | PASS（执行者自检，待 Codex 审查） | 3-GPU、3 workers、ABBA；every1/every20各2次，每次30 updates（10 warmup+20计时） | `results/glt_engineering_20260918/20260918T004314Z/pretrain_*/benchmark.json`；4份日志；退出码均0 | 预训练本轮累计132/316 updates；未执行最终组合确认（无候选入选） |
| S6 报告/交接 | 待审查 | 汇总 `summary.json`、本计划与 PIPELINE 待更新；当前工作树待提交 | `results/glt_engineering_20260918/summary.json`；日志/产物路径见各行 | 未执行正式5k/20k、完整微调、OOF/outer-test、4-GPU正式吞吐；不写性能提升结论 |
| Codex独立审查 | 未执行 | — | — | 需审查本轮源码、测试、日志和报告后再归档 |

### 本轮执行摘要（执行者记录，待 Codex 审查）

- 实际修改：`scripts/pretrain_glt_dual.py`、`src/modules/glt_dual_pretrain.py`、`scripts/profile_glt_dual_runtime.py`、`scripts/finetune_glt_dual.py`、`src/utils.py`、`scripts/run_glt_dual_finetune_grid.py` 及两份相关测试文件；未修改模型科学配置、active cache、manifest、checkpoint 或样本集合。
- 计时口径：CPU profile 使用 seed=42 的256个不同随机索引，记录 source/static/target/clean/noisy/collate/完整样本及图大小；微调记录 source打开、首批、train/validation、best CPU copy、保存和进程总耗时；预训练记录每rank窗口及最慢rank。
- S5 预训练 only：every1 samples/s 为 325.917、331.940；every20 为 331.669、319.285，配对方向相反，未通过10%性能门槛。S4不实施额外候选；不执行最终确认组。
- S3 cache smoke：`clean-cache-gib=0` 进程墙钟 20.149/19.983 s，`=4` 为 11.921/11.872 s；四个调度单位两策略均完整、验证结果逐单位 exact，动态 makespan仅较批处理快约0.244 s。
- 资源与边界：预训练使用3张GPU（0,1,2），微调/调度使用GPU3或2、3；预训练累计132个实际 optimizer updates，微调累计24 epochs；所有真实 smoke 均未访问 outer-test。历史/正式5k、20k、完整OOF和4-GPU正式吞吐均未执行。
- 失败证据保留：`003412Z/profile.json`/日志记录首次 profile 字段错误及零写入；修正后 profile 单次重跑通过。所有长任务在 `tmux` session `Uni-Poly` 独立 `glt_eng_*` window 中执行。
