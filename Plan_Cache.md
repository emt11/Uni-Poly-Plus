# 缓存优化计划：先补验收与恢复，再优化读取

## 0. 计划头与授权

| 项目 | 记录 |
| --- | --- |
| 计划 ID／修订 | CACHE-20260916-01 / r2：审查后最小返修 |
| 日期／环境 | 2026-09-16；当前远程 Linux 训练机，项目根 /root/workspace/Uni-Poly-Plus-master |
| 状态 | 待审查；r2 返修已由 ZCode 执行，尚未由 Codex 独立验收 |
| 当前授权 | 用户直接要求按本计划由 ZCode 执行。范围仍限于 R1–R4 的局部代码、相关测试、32 条固定 parity 与文档／报告；不授权生产切换、全量重建、训练或 GPU smoke |
| 角色 | Codex 规划与后续审查；ZCode 执行。本节执行结果是执行记录，不代表独立验收通过 |
| 基准 commit | r2 审查／返修基线 7f476b481ceff4fa4a0561845230b329f6fa872b，dev；r1 规划基线 18572e3 保留作为历史记录 |
| 修改前状态 | dev 工作树干净，pull 返回 Already up to date；r1 文档创建时的用户改动已记入 PROJECT_HISTORY.md，不沿用为当前状态 |
| 交接入口 | [Plan.md](Plan.md)；本文件是缓存专项实施细则，不替换正在进行或未收口的科学计划 |
| 总结论 | 保留基础缓存＋静态派生缓存＋预训练目标缓存；先解决少量具体证据与工程缺口，不直接全量重建 |

r2 执行以第 9 节的审查状态和第 10 节的返修合同为准。第 1–8 节保留 r1 的目标、基线与原始验收要求，不表示它们已全部完成，也不构成重新运行 A–D 的指令；其中构建缺口表为 r1 修改前的静态发现，当前缺口以第 10 节为准。旧清理周期见 [PROJECT_HISTORY.md](PROJECT_HISTORY.md)，其余 HOLD 不解除。

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

## 9. r1 执行记录与 Codex 审查状态

审查对象：commit 7f476b4；既有产物 results/cache_optimization_20260916T145328Z/，日志 logs/cache_optimization_20260916T145328Z/。ZCode 原执行记录保留在 Plan.md；以下为 Codex 对代码、测试源码及既有报告的静态审查，不冒称本轮重跑测试。

| 项目 | 已有工作／证据 | 审查结论 |
| --- | --- | --- |
| A 化学解释 | 44 条逐例记录、源特征分桶、4 条合同拒绝及 9 个 fallback 核对材料 | 有价值的局部证据；须收窄“99.99% 已解释／0 条未解释”，不接受全量化学正确性结论 |
| B 实现 | staging context、临时 chunk、发布状态、锁、finalize 和 reader 校验已有代码 | 需返修：幂等报错、未知 staging 接纳、中断窗口、恢复校验未接通、锁竞争、冻结前汇总缺失 |
| B 测试 | 执行者报告相关测试通过 | 当前辅助函数测试不足以验证完整 build()；“110 passed”不能替代未覆盖路径，需要准确命令／日志／退出码 |
| B 真实 parity | 报告 32 条静态字段一致（20 PI1M＋12 下游，含 9 fallback） | 仅接受报告覆盖的静态字段范围；新目标缓存及固定 mask/noise 的准备路径未充分验证 |
| C 性能 | 2,048 索引的既有测量、容量 64 观测增益约 4.3%；另有 22 条准备结果 digest 对照 | 保留“不采纳 64、默认 2”；交替顺序／多进程资源／计数口径有缺陷，不是严格配对提速证据 |
| C 第二候选 | 几何 materialize 已读取静态索引 | 不等于中心监督 angle_pairs 已复用；prepare_pretrain_sample 仍遍历 line_path，候选暂缓而非已完成 |
| D 与正式运行 | 紧凑存储、全量重建、生产切换、清理和新训练未执行 | 继续 HOLD，不进入 r2 |

r1 报告中“停止条件均未触发／A–C 完成”的自检不能覆盖上述审查发现。当前无证据要求重建 active 缓存，但新版构建恢复流程不能据此宣布可靠。r1 的实现、原始失败记录和测量值均保留，不回写或抹除历史。

## 10. r2 下一步：仅返修可靠性与证据缺口

### 10.1 范围、顺序与执行前核对

顺序固定为 R1 → R2 → R3 → R4：构建恢复修复 → 补齐原样本 parity → 证据／工具口径修正 → Codex 复审。不重做阶段 A 全量分解，不重跑阶段 C 长 benchmark，不实施角度优化或阶段 D。

执行前按 AGENTS.md 核对 Git／pull、当前实现和活动任务。复用第 2 节冻结身份作为比对基线；身份变化先解释，不修改产物强行匹配。r2 只在独立临时目录与新的 results/cache_optimization_repair_<UTC>/、logs/cache_optimization_repair_<UTC>/ 写本轮数据；不得覆盖 r1 报告、active 数据或未知 staging。

原范围内返修由 ZCode／用户指定执行者继续；本轮 Codex 不接管代码或测试。任何 worker、预计超过一分钟或写验证产物的命令均使用 tmux Uni-Poly 独立 window，保存完整 argv、cwd、解释器、环境、退出码；不另建通用运行框架，不启动其他代理。

### 10.2 R1：构建／恢复／冻结的六项最小修复

以下位置以 7f476b4 为基线，行号会随修复变化。首先加反例，再最小修复，不顺手重构。

| ID | 缺陷与入口 | 必须达到的行为 |
| --- | --- | --- |
| R1.1 | build_glt_dual_static_cache.py 的 IDEMPOTENT 分支先使用 started | 初始化顺序正确；完整 build() 重复调用返回成功，不改任何已发布文件，不遗留新的 staging 内容 |
| R1.2 | _prepare_artifact 为无 context 的已有 staging 写入当前 context | 区分新空目录与有旧 keys／chunks 的未知目录；后者无法证明来源时拒绝，不能补身份后复用、移动或删除 |
| R1.3 | write_chunk 在临时目录写 .complete 后、rename 前中断 | 对同一已确认构建的完整临时 chunk 先校验，再完成发布／安全恢复；不能永久拒绝，也不能把未知或损坏临时目录当成完成块 |
| R1.4 | _existing_chunk 未调用新增载荷校验 | 所有恢复复用和最终发布之前验证必需文件、shape／dtype、范围、offset 与载荷对应；缺失／截断／错位必须在冻结前失败 |
| R1.5 | 锁先创建文件后写 PID，空锁可被另一 writer 删除；准备写入早于持锁 | 任何 keys/context/chunk 写入前已有可靠互斥；空锁或未知持有者不能自动当成死进程；异常和部分加锁失败均释放本次持有的锁 |
| R1.6 | _artifact_manifest 先冻结，finalize 汇总未接入 builder | 验证并汇总几何数量／失败原因后形成最终 manifest，再冻结发布；正常 builder 产物再次 finalize 为只读幂等，旧 frozen 不改写 |

实现细则：

- R1.5 优先复用已有可靠锁；当前 Linux 可采用稳定路径上的进程级文件锁等最小方案，不引入分布式锁系统。锁须覆盖身份检查到发布，不能因 staging rename 失去互斥；不能把其他活进程的锁误删。测试第二把锁获取失败、异常退出、成功发布后的释放与可重新进入。
- R1.4 将现有校验用于 builder，不仅用于消费者。只有计数／标记通过不能冻结；不要求新增全量内容 hash，也不许把 metadata 验证称为 payload 全字节审计。
- static 与 targets 仍是两个独立发布对象；已发布一侧只读校验并复用，未发布一侧可恢复。保留仅 static 的合法入口，不新增强制 targets 依赖。
- R1.6 不通过允许 finalize 修改 frozen 绕过问题；诊断报告写独立目录，不允许指向被保护的 manifest／.frozen。
- 正常新建、恢复、幂等都必须走实际 build() 控制流；可以用确定性 worker 结果替代昂贵化学运算，但不能把 build() 自身 mock 掉。

必要合成验收：固定两个小 chunk，同时覆盖 static-only 与 static＋targets。调用完整 builder，比较连续构建和各中断恢复的 keys、数组、最终语义 metadata；时间戳／代码来源等非语义差异单列，不要求伪造同一运行身份。

故障矩阵至少覆盖：

1. 新建完成后重复调用；不启动不必要的构建 worker、不新增 staging、不重写 frozen。
2. payload 半写、payload 写完但无 .complete、临时 .complete 已写但未 rename、chunk 完成后、static 已发布而 targets 未发布。
3. 同 keys 不同 parent／cohort／参数；旧非空 staging 缺 context；已发布一侧与缺失侧来源不一致。
4. 缺文件、真实截断、错误 shape／dtype、非单调或终点不符的 offsets；恢复与发布均拒绝且不产生新 frozen 成功产物。
5. 两个实际进程受控竞争同一构建，用同步点稳定触发锁窗口，不靠偶然 sleep；恰一个 writer 可以进入写路径，另一个等待或明确失败。测试持锁异常退出和锁释放。
6. 汇总字段与 fixture 实际 valid／invalid 数量一致；正常新产物 finalize 幂等；旧 frozen 缺汇总或绑定不一致时拒绝修改、字节不变。

仅跑本次相关单元／集成测试，不以凑齐旧“110”计数为目标。测试模块、case 数、命令、退出码和失败／修复顺序据实记录；不得只测 _publish_plan() 就宣称单边发布恢复通过。这些测试仅证明进程中断行为，不承诺断电耐久性。

### 10.3 R2：补齐原 32 条样本的实际目标与 clean/noisy parity

复用 r1 parity.json 中已固定的 20 PI1M＋12 下游 sample keys，不扩充真实样本集合，不重新生成构象；全部临时派生输出总量仍不超过 1 GiB。已有静态字段一致结果保留，不把它改名为全路径验收。

改 scripts/verify_glt_dual_static_parity.py，补充以下实际比较：

- PI1M 新建并读取本轮独立 pretrain_targets 临时产物，比较 BRICS 分组、原子索引、packed／解包指纹；不能把 reference targets 再打开作为 candidate，不能丢弃新生成 target。
- 对相同 key、position、seed、sigma 和 ratio，比较原参考／在线路径与新临时缓存路径的 clean 输入、noisy 输入、atom mask／label、中心 distance／angle_pairs／angle_cos、fingerprint 与 skip reasons，核对物理多重性、顺序和身份。两个分支各自恢复相同样本级随机条件。
- 下游无已发布 targets 时不声称做了 target artifact 对照；比较适用的在线数据准备与新 static 路径，明确来源。9 个 fallback 保留，不能只挑有效样本。
- N=0 不等于“无中心角”：仅在原 32 条中有可证明记录时标真实覆盖，否则使用最小 fixture 并注明。验证中心目标为空、无 NaN、跨键 state 不进入中心监督；不为补此项扩大真实样本搜索或启动训练。
- 整数、索引、搬运字段完全一致；浮点使用原已有明确容差并报告最大误差，不为通过放宽。显式检查有限性。
- 在写下一块前检查输出预算；不在超限后仅写 within_size_budget=false 然后成功退出。差异、身份错配、非有限或超限均返回非零并保留诊断。
- 输出目录须为本次新建且归属明确；禁止现有脚本对传入 --temp-root 下已存在候选目录直接 shutil.rmtree。重复路径应拒绝，不清除历史证据。

用极小 fixture 验证“故意修改一个 target／造成超限 → 非零退出”。真实 parity 只补上述缺失路径及修复必要的回归，不重新跑整个化学审计、模型 forward/backward 或训练。

### 10.4 R3：纠正化学结论、benchmark 工具与文档

**化学解释仅收窄，不新增数据审计：**

- 11,337／11,338 指具有相关源特征的 FAIL 数量，表述为“特征分桶＋有限逐例机制证据”，不能写成全部已获因果解释或冻结化学认证。
- 44 条保留原分类：10 条显式氢、19 条连接参照、1 条重建路径、6 条 PASS 对照、8 条 Stereo 参照仍未解。若使用“0 条未解释”，须限定被复核失败机制范围；8 条 UNRESOLVED 不能因此变 PASS。
- 4 条合同拒绝、9 个 fallback 材料保留；不改 source、生成器政策或接受集合。旧 JSON 不覆盖，用修订说明或新报告解释旧字段 fail_explained_by_source_features 的实际含义。

**benchmark 工具局部修正，不新跑长 benchmark：**

- scripts/benchmark_glt_dual_read.py 若保留为可用工具，改为明确 AB／BA 等交替计划、相同预热，并记录实际顺序；实际重复数写入报告，去除固定“三次”措辞。
- chunk_map_events 只计真正 miss／映射，或明确改名为调用次数；多 worker 的计数不能拿主进程变量冒充汇总。FD／RSS／faults 字段必须标明 self／workers／tree、当前值／采样峰值；未采集字段标缺失。
- 用 mock／小合成 reader 验证命中计数、顺序和报告口径；可以在既有 R1 多进程测试环境检查资源字段，不新增真实数据性能长测。
- 旧容量 64 的 +4.3% 保留为有顺序混杂的观测；两次 FD 峰值相同不证明无泄漏；22 条 digest 一致不代表 2,048 条完整 parity。默认容量维持 2，不改生产训练配置。
- 中心 angle_pairs 监督仍在 prepare_pretrain_sample 中遍历 line_path；与 materialize 消费几何索引分开描述。该候选标“暂缓／尚未实现”，本次不修改算法、不另开 profile 或角度优化。
- C 以“修正文档与工具、保守不采用、性能验收受限”收口，不强行补成提速证明。后续若确需新的长 benchmark，另列精确剩余／新增预算并按授权处理。

同步更新实际受影响的 PIPELINE.md 附录和 Plan.md；保留 r1 原执行记录并追加更正，不把过去写成当时已知问题。RESULTS.md 不新增性能／科学结论。

### 10.5 R4：交付与再次审查

交付包含逐项 R1.1–R1.6 证据表、实际 builder 故障注入日志、R2 key 清单与差异报告、R3 口径更正，以及 Git commit。日志保存完整 argv、环境、退出码；“历史测试命令未找到”如实注明，不能用新测试补造历史。

执行记录先写 Plan.md，状态置“待审查”。Codex 按以下标准复核后才可关闭周期：

- 六项恢复／冻结问题均有对应失败反例和修复后通过证据；完整 build() 路径覆盖，发布与锁行为无绕过。
- 原 32 条适用目标／clean-noisy 路径已比较，fixture 与真实样本清晰分开；失败时不会成功退出，不扩大样本或预算。
- active 缓存仍只读、未切换；报告前后核对的实际字段与范围，不宣称 metadata 快照能证明所有字节未改。
- A／C 的过强结论已纠正；仍未解和未采集项保留，容量 64 不采用，角度优化／阶段 D 未启动。
- 不生成构象、不全量重建、不删除缓存、不运行任何正式实验；未知身份／writer 冲突／NaN／超预算按第 8 节停止。

本轮文档交付不等于独立验收通过。r2 周期当前状态为“待审查”；不将 ZCode 自检或本轮局部测试写成最终验收结论。
