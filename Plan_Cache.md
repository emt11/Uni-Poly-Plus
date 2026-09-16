# 缓存优化计划：先补验收与恢复，再优化读取

## 0. 计划头与授权

| 项目 | 记录 |
| --- | --- |
| 计划 ID／修订 | CACHE-20260916-01 / r1 |
| 日期／环境 | 2026-09-16；当前远程 Linux 训练机，项目根 /root/workspace/Uni-Poly-Plus-master |
| 状态 | 计划文档已交付；下述实施、测试和 benchmark 均待授权、未执行 |
| 当前授权 | 用户要求 Codex 制定缓存优化计划并写入本文件；不包括代码修改、缓存构建、训练、清理或生产切换 |
| 角色 | Codex 规划／后续审查；ZCode 或用户指定执行者在获授权后实施；本轮文档修改与自检均为 Codex，不是独立审查 |
| 基准 commit | 18572e373ae270469affa8ecb805db32f3c6489c，dev；修改前 pull 返回 Already up to date |
| 已有用户改动 | Plan_Delete.md 已删除，Plan_Cache.md 为未跟踪文件且原内容为清理计划；按本次要求重写本文件，不恢复／提交该删除 |
| 交接入口 | [Plan.md](Plan.md)；本文件是缓存专项实施细则，不替换正在进行或未收口的科学计划 |
| 总结论 | 保留基础缓存＋静态派生缓存＋预训练目标缓存；先解决少量具体证据与工程缺口，不直接全量重建 |

以下命令、代码修改、产物和验收要求均是后续计划，不是已完成事实。旧清理周期保存在 [PROJECT_HISTORY.md](PROJECT_HISTORY.md) 及基准 commit 的历史文件中；其余 HOLD 不因本计划解除。

## 1. 目标与不可改变的边界

本次拟回答三个独立问题：

1. 审计中的 FAIL／UNRESOLVED 究竟是参照定义问题、审计器问题，还是冻结产物的实际化学错误？
2. 静态缓存构建遭遇进程中断时，能否拒绝错误来源、恢复半成品并完成一致发布，而不修改已冻结产物？
3. 保持样本、目标、随机流和数值语义不变，随机读取的实际瓶颈是否值得做局部优化？

正确性、物理代表性和模型预测收益分别报告。读入成功／有限输入不证明化学正确；Stereo 未见翻转不证明全部化学正确；读取提速不证明模型性能提升。

固定边界：

- 不调用 ETKDG／MMFF，不生成新构象，不改变 first_valid、候选数量、超时、端基、RU 映射或科学失败政策。
- 不改模型、loss、中心键监督定义、噪声／mask、样本排列、训练 cohort、数据 split、seed 或正式实验预算；不启动预训练、微调、GPU smoke。
- 不修改旧 LMDB、active bundle、cohort、static、targets、manifest、.frozen、checkpoint 或历史运行记录；不覆盖旧报告，不自动切换消费者。
- 下游保留全部性质行和 9 个几何 fallback；结构去重不等于性质记录去重。预训练 matched 对照仍使用同一固定 cohort。
- 不把缓存优化与历史清理合并；不重新生成已经清理的 pilot／staging，也不把历史存在过的目录写成当前可用路径。
- 优先复用现有字段、构建器、reader 和验证脚本。缺陷先用局部 fixture 复现；只增加修复该缺陷所需的检查，不预建通用迁移器、事务框架、版本体系或全量内容 hash。

## 2. 已核对的基线与证据边界

### 2.1 数据链

| 层 | 当前记录 | 本计划约束 |
| --- | ---: | --- |
| PI1M source | 988,775 | 冻结来源 |
| RU／Topology | 988,769 | 与 source 差 6 条 |
| Trimer／固定预训练 cohort | 959,588 | 相对 source 少 29,187，约 2.95%；不得静默改变接受集合 |
| dual_static_v1 | 959,588，235 chunks | 静态连接、路径和坐标索引；不是冻结 noisy 几何 |
| pretrain_targets_v1 | 959,588，235 chunks | BRICS 分组和压缩指纹目标 |
| downstream | 3,655 个结构／6,265 条性质记录 | 9 个几何 fallback 保留 |

当前源入口为 data/processed/mips_trimer_scage/store.json；PI1M bundle 为 30f17b59bc5862a1ddae7eaee03b2767df26561d9bfecb690ec8eea3ddd09ed2，下游 bundle 为 1545eda5a8f6a1a7868ce01464ce7c6dc714b4685ae10dc7213b90adfbcc23b2。

PI1M 派生缓存位于 data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1 下：

| 对象 | 当前 manifest 身份 |
| --- | --- |
| cohort | b03f96a14c2beb1d987743886cfeb2b738d94b27897f9107400d573f048d476b |
| dual_static_v1 | 9ff122cc16df5c869582ee2fa07b6f42fa44f2a228f554d25412eb2ae6d6006d |
| pretrain_targets_v1 | 5e7b5ec8e5f96bf695494436cd471c9d44a85d99571db88adf34c7d5f5777482 |

这些身份是执行前比对基线，不是对全部数组内容的完整性证明。执行者从 live store 和 manifest 解析实际路径，不凭短 hash 拼出不存在的目录。若身份变化，先核对合法更新，不能继续套用本基线。

现有 PI1M run.json 可比对相关 hash；下游 run.json 仅提供缓存／split 路径。当前 manifest 与 .frozen 自洽不能单独补证历史下游运行时的缓存内容身份。

### 2.2 几何与审计

实际几何构建为每轮最多 4 个独立候选、最多 2 轮、共享 60 秒预算、MMFF94 最多 200 steps；第一个通过检查的候选立即接受，不做最低能量排名或收敛优先。冻结结果应称“通过既定检查的单构象”。限时接受集合可能受运行负载影响。

Trimer 拒绝 29,181 条：MMFF 不支持 24,886、超时 4,207、无有效构象／嵌入失败 84、内部键合同错误 4。4 条错误单列调查，不能与普通几何失败混为一谈，也不能据此断言接受集已损坏。

既有证据：

- results/glt_dual_readiness_20260915/pi1m_stereo_audit_v4/summary.json：已声明立体化学子集 170,639 条，PASS 114,701、FAIL 11,338、UNRESOLVED 44,600；已检查 Stereo mismatch 样本数为 0。
- 同目录 phase3_stereo_sanity.json：对所查非 Stereo 失败中的 32 条成功构造有限输入；这不是独立化学正确性证明。
- results/glt_dual_static/pi1m_verify1000.json：前 1,000 条一致性 PASS，绑定当前 manifest；不是随机／全量验收。
- results/glt_dual_static/pilot10k_benchmark.json：历史顺序读取／构建基线，不覆盖真实随机多 worker 训练；相关 pilot 数据已经清理，不能直接复用其输入路径。

### 2.3 容量口径

此前记录 active bundle 约 75 GiB、static 约 45 GiB；整个 mips_trimer_scage 约 193 GiB 是清理前快照，不能作为当前占用。active bundle 属于该根目录，不能重复相加。后续若需容量结论，重新记录时间、路径、逻辑字节与实际磁盘占用。

按当前静态 manifest 的 shape／dtype 核算，数组载荷约 44.67 GiB：bond_path_features 16.18 GiB、angle_pos_triplet 12.44 GiB、line_path 6.22 GiB，其余约 9.83 GiB。这是优化线索，不是已验证压缩收益；空失败原因的固定宽度字符串约占 0.46 GiB。

## 3. 阶段 A：小预算闭合化学审计解释

### 3.1 选择与预算

优先复用已有审计逐样本记录，固定抽样清单后再看细节，避免只挑能解释成功的例子：

- 最多 48 个接受集真实样本，覆盖主要 FAIL 原因、UNRESOLVED 的端点参照缺失、PASS 对照及已有明确 Stereo 样本；记录每类数量、选择方法和 sample key，不作总体错误率估计。
- 另核对全部 4 条内部键合同错误的 source／拒绝日志；它们未被接受，不能假定存在冻结 Trimer 坐标，不为补证据重新生成构象。
- 只读核对下游现有 9 个 fallback 的身份／原因与保留状态，不重新跑几何。
- 稀有边界先查现有记录；每类定位最多 5 分钟，找不到就记录缺证。N=0 等工程边界可用最小确定性 fixture，不把 fixture 当成真实化学质量证据。
- 不新跑全量 Stereo／LMDB 扫描；必要的现有索引或拒绝日志查询可以进行。

### 3.2 逐例核对

基于 source、RU／Topology 和冻结 Trimer 的实际字段核对 atom mapping、中心内部键、跨 RU 键、端基、价态／氢数、芳香性表示及 Stereo 参照。允许从 source 做只读化学解析，不做坐标生成／优化。

明确区分中心内部化学与末端封端导致的预期差异；比较芳香／Kekulé 表示前说明等价准则；不能仅比较模型 token 或有限性。对重复关系保留物理身份与多重性；N=0 的跨键 state 不能当作中心内部键目标。

每例裁定为以下之一，并保留证据：

1. 审计参照与已声明端基规则不一致；
2. 审计器实现错误；
3. 冻结产物真实违反已声明化学合同；
4. 缺少参照，仍 UNRESOLVED。

不得仅根据旧 sanity 的分类直接复制结论。审计器修正需要局部反例测试；改变端基／科学合同或接受策略不在本计划修复范围。

### 3.3 交付与门槛

输出带 sample key、原因、字段差异和裁定依据的小样本表，以及未覆盖类别。若代表样本支持“审计器不适配”，只修正该原因的结论，不把 11,338／44,600 全部改成 PASS。

发现真实内部化学／映射错误：停止相关数据的推广和基于其正确性的结论，交回 Codex 制定数据处置方案；不自动修改缓存、扩大排除名单或终止既有任务。与真实数据无关的阶段 B 合成恢复测试仍可单独进行。

## 4. 阶段 B：最小修复静态构建、恢复与冻结

### 4.1 需要先复现的缺口

| 位置 | 当前静态审查发现 | 拟修复目标 |
| --- | --- | --- |
| scripts/build_glt_dual_static_cache.py | staging 主要比 keys；完整 chunk 主要比起点／数量／target 标志 | 相同 keys、不同 parent／配置不能错误恢复 |
| src/dataset/glt_dual_static.py 的 write_chunk | 半成品目录存在时拒绝重写 | 恢复仅处理当前构建拥有的未完成 chunk |
| 构建器的两次 os.replace | static／targets 分别发布，存在单边发布窗口 | 可识别并幂等恢复单边发布，不混用两次构建 |
| scripts/finalize_glt_dual_static_artifact.py | 可改写已发布 manifest／.frozen | 汇总和必要验证在冻结前完成；旧冻结产物只读 |
| chunk manifest／reader | shape／dtype 元数据不能证明实际 payload 完整 | 最小验证缺失、截断、shape／dtype／offset 不一致，不虚称内容 hash 已覆盖 |

### 4.2 实施合同

1. 使用现有 format、parent bundle、cohort、sample order、build parameters 等身份字段，把本次构建上下文在写首个 chunk 前保存至 staging。resume 逐项一致才可继续；不同来源不得因 keys 相同而通过。
2. 写 chunk 使用当前 staging 内独立临时目录，数组写完并关闭、必要校验完成后再形成完成标记和最终 chunk。中断残留先确认归属，再移入该次构建的隔离位置或按明确策略重建；不删除未知 staging、旧失败证据或 active 数据。
3. 已完成 chunk 恢复时检查必需文件、实际可读 header、shape／dtype、样本范围、ragged offsets 与载荷边界。缺失／截断明确报错；不默认全量扫描数组或新增全量 checksum。
4. 在 staging 中生成最终汇总和 manifest，完成验证后写 .frozen 并发布。禁止“先冻结发布、再 finalize 改身份”。对旧 frozen 调用 finalize 时应明确拒绝修改，或只输出独立诊断报告。
5. static 与 targets 两个独立根目录不可能靠两次 rename 自动获得整体原子性。首选最小幂等恢复：启动时区分未发布／单边发布／双边发布；只有来源、keys、参数和对应构建完全一致，才复用已发布一侧并补齐另一侧，绝不覆盖已冻结一侧。
6. 需要两者的 consumer 在打开前验证双边完整及绑定一致；仅使用 static 的合法旧路径不被强行改成必须提供 targets。已有格式的正常读取保持兼容，旧 metadata 不回填／篡改。
7. 不允许并发 writer 接手同一 staging；复用已有锁或明确单 writer 检查。若当前机制不足，先用冲突 fixture 证明，再补最小互斥，不另建调度系统。
8. 本轮只承诺通过测试的进程中断恢复；不把文件 rename 或 .complete 等同于断电耐久性保证。不默认增加新 schema／全局事务标记；若现有字段无法消除真实歧义，提出最小变更给 Codex 审查。

### 4.3 验证预算与接受标准

在新建独立临时目录中使用至少两个小 chunk 的确定性 fixture，覆盖：

- 连续完成与中断后恢复的 keys、数组值、shape／dtype 和语义 metadata 一致；时间戳／运行来源等非语义字段单独比较。
- 第一块 payload 写一半、写完但无完成标记、chunk 完成后、static 发布但 targets 未发布等中断点；恢复不重写完整冻结侧。
- 同 keys 换 parent／配置／cohort 拒绝；双边错配拒绝；缺文件、截断、错误 offset 拒绝；已完成重复调用不覆盖产物。
- 两个 writer 指向同一测试 staging 时不能同时写入；冻结后 finalize 不改 manifest／.frozen。
- 保留旧格式 reader 的相关局部测试，不跑无关全仓回归。

随后最多取 32 个固定真实样本做新临时派生缓存与在线构建的一致性检查，不生成坐标；覆盖普通、明确 Stereo、N=0／无角度及下游 fallback 中存在的类别。缺失类别写明并用 fixture 补工程边界。临时派生输出总上限 1 GiB，达到上限即停，不放宽预算。

必须验证 sample identity／顺序、离散特征、物理键／角度索引、multiplicity、中心 readout、BRICS／指纹目标；固定噪声与 mask 后比较 clean/noisy 输入。整数和纯搬运字段要求完全一致；浮点重算使用与现有验证一致的明确容差并报告最大误差，不能为通过测试放宽容差。

## 5. 阶段 C：先量瓶颈，再做一个最小读取优化

### 5.1 公平 benchmark 合同

| 项目 | 约定 |
| --- | --- |
| 工程问题 | 随机读取是否主要消耗在反复 mmap/open，或预训练重复遍历中心角度？ |
| reference | 同机、同一冻结缓存、未修改 reader 和数据准备路径的短 baseline；历史顺序 benchmark 仅作背景 |
| controlled change | 首先只试一个有界 chunk 映射缓存配置；如另做 angle_pairs 复用，单独比较，不混为单因素 |
| 样本与随机性 | 固定 seed=42 的 2,048 个索引及顺序，保存清单；保持样本级 mask／noise／targets 相同 |
| 规模 | worker=0 与一个最多 3 workers 的配置；每个 baseline／candidate 配对最多 3 次；不启动 GPU／模型 |
| 总预算 | baseline、至多两个单因素候选合计最多 30 分钟；超时记录局限，不扩样本／worker／配置网格 |
| 输出 | samples/s、延迟 p50/p95、打开／映射次数、进程树 RSS、FD 峰值、page faults、运行负载 |

计时分别报告 source／static／targets 读取和完整数据准备（包括 clean/noisy、mask、目标）的开销，不能只计 materialize 排除 I/O。多 worker 报告进程树资源，不把主进程 RSS 当作总占用，也不把共享映射简单求和声称为唯一物理内存。

baseline 与 candidate 同样预热，交替顺序，记录 OS page cache 和其他任务干扰；不得 drop_caches、改系统设置或停止其他任务来制造冷缓存。只说明观测环境，不称为严格冷盘或正式训练吞吐。

### 5.2 候选与晋级

- mmap 候选：当前 LRU=2 面对 235 chunks 是待验证风险。先测 open/mmap 次数、FD 和 RSS，再选一个较大的有界容量；所有 workers 的总资源需受控。不要默认映射所有 chunks。
- 角度候选：若 profile 显示重复路径遍历显著，再复用已有 angle_pairs 等静态索引。保持中心筛选、端点排序／去重、多重性与 target 顺序完全一致；固定随机流比较结果。没有足够开销证据则不实施。
- 不采用 chunk-shuffle、删样本、缓存噪声、改变 batch 或 mask 来换取速度。缓存淘汰不得使尚被 batch 引用的 numpy／tensor 失效；增加对应生命周期 fixture。
- 首轮仅评估 CPU 读取／数据准备，不推断 GPU 利用率或端到端训练收益。
- 候选须通过阶段 B 对应一致性检查，并且配对吞吐中位数提升至少 10%、p95 不退化超过 5%、无 FD／RSS 持续增长，才建议后续采用。阈值是工程筛选，不是统计显著性；若代价显著增加或结果受负载干扰，记为不确定／不采用，不继续 sweep。
- 本轮即使通过，也只交付代码候选和报告；生产配置切换、正式训练验证仍另行授权。

## 6. 阶段 D：紧凑存储只保留为后续选项

本阶段不属于 A–C 的默认执行预算，状态为待授权；先有前述结果，再决定是否值得启动。

候选按优先级评估：可精确恢复的离散路径特征紧凑 dtype；空 invalid_reason 的紧凑表达；“物理键特征一次存储＋路径索引”的表示。不能预先承诺压缩率或总体提速。

如后续明确授权，仅选一个表示候选，最多 256 条既有冻结样本、独立输出不超过 1 GiB；检查值域／索引上界并验证恢复后的特征一致，测磁盘、CPU 解码和随机读取。不得量化坐标、截断距离／角度、改变索引含义或直接迁移全量 static。

候选与旧 reader 显式区分，旧数据只读兼容；确需新格式时再提出最小版本变更。全量派生重建、上线切换、旧缓存删除必须另列容量、时长、回退与授权方案；不以“静态不生成构象”为理由自动开跑。

## 7. 文件范围、运行环境与产物

拟修改只限已复现问题相关文件：

- scripts/build_glt_dual_static_cache.py、scripts/finalize_glt_dual_static_artifact.py；
- src/dataset/glt_dual_static.py；必要时 src/dataset/glt_dual_pretrain.py；
- 现有相关测试、verify／benchmark 入口；化学审计脚本只在证实审计器缺陷后局部改动。
- scripts/build_mts_cache.py、src/dataset/trimer_mcl.py 是来源核对对象，默认不改生成器和失败政策。

执行前重新 pull、读 AGENTS.md／Plan.md／本文件，核对 active 身份、工作树、进程和 tmux；不得覆盖用户改动或重复启动执行者。预计超过一分钟、启动 worker 或写实验产物的工作均在 tmux session Uni-Poly 独立 window 中进行，保留完整 argv、解释器、环境、cwd、退出码和日志；当前交互窗口不承担长任务。不自动终止既有训练或高频轮询。

获授权后才创建 results/cache_optimization_<UTC>/ 与对应 logs 目录；使用本次唯一目录，不覆盖历史报告。建议保存：

~~~text
baseline.json                  # 来源、配置、实际环境；复用现有身份字段
audit_cases.jsonl               # 有界真实样本证据与未覆盖类别
recovery_tests.json             # 各中断点、预期／实际行为、退出码
parity.json                     # 真实样本与 fixture 分开
benchmark.json                 # 原始配对计时和资源，不只记录最佳值
decision.md                    # 采用／不采用／缺证、剩余风险
~~~

这是产物职责约定，不要求新增通用 runner 或统一 schema。小表也可并入现有报告；不要为填充清单创建空 PASS。临时构建目录用 mktemp 独立创建，记录路径；只处理本次拥有的临时数据，保留失败证据，不执行广泛清理。

## 8. 停止条件与完成定义

立即停止受影响阶段并报告：身份错配、writer 冲突、可能改写 frozen／正式产物、非有限有效数据、索引或中心监督语义变化、丢失样本／fallback、需要改变科学合同或超过预算。只读定位可以继续；授权内的局部缺陷修好并验证后才能恢复。

分层完成标准：

1. A：代表案例有可复核裁定、4 条合同错误单列、缺证类别明确；不是全量化学正确性认证。
2. B：已复现缺口得到最小修复，相关故障注入／真实小样本 parity 通过；旧 active 冻结身份不变。没有旧 payload 全量内容摘要时，仍明确不能据此证明所有字节历史未变。
3. C：固定预算内有真实配对性能和资源报告；未达门槛也可完成调查，结论为不采用，不强迫产生优化收益。
4. Codex 根据 diff、准确命令、退出码和产物审查；ZCode 自检不代替最终审查。真实执行记录更新至 Plan.md，再按规则归档 PROJECT_HISTORY.md。
5. 仅当流程代码确实改变后更新 PIPELINE.md；没有训练结果就不新增 RESULTS.md 性能结论。全部项目文件改动按 AGENTS.md 提交、推送并核对远端。

整个缓存体系的物理代表性、全量字节完整性和下游预测收益不属于本轮可宣布完成的结论。全量构建、格式迁移、生产切换和清理均未执行，不能混入“优化已完成”。

## 9. 当前执行记录与下一步

| 项目 | 本轮状态 |
| --- | --- |
| 相关代码／既有 metadata 与报告的静态核对 | 已用于制定本计划；不是新增测试 |
| 文档修改 | 本文件重写为缓存优化计划；Plan.md 增加专项交接；PROJECT_HISTORY.md 归档文档周期 |
| 化学小样本复核、缺陷复现、代码修复 | 未执行，待授权 |
| 单元测试、故障注入、真实样本 parity、benchmark | 未执行，待授权 |
| 构象生成、全量重建、训练、生产切换、清理 | 未执行，不在当前授权范围 |

建议首个执行批次：阶段 A 的有界只读复核＋阶段 B 的合成缺陷复现和必要局部修复；通过审查后再安排阶段 C。阶段 D 暂不启动。用户若授权 A–C，可在上述预算和门槛内顺序推进，不为已明确授权的步骤重复申请；未授权的扩大事项仍需另行决定。
