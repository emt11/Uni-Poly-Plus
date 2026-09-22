# MCL-PH：多尺度距离专家与 PH 路由替换 GLT 3D 通道

计划 ID：`MCL-PH-20260921-01` ｜ 修订：r2 ｜ 日期：2026-09-21 UTC

## 0. 状态、角色与授权

- **状态：r10R1 / P2-E2E 阻断（P2 未完成）**：预注册阈值边界已按审查返修（`no_task_sacrifice` 改为 `>= -0.01`，含界限回归，**15 passed**）；**GLT_REF 第二次正式启动成功**（`glt_ref_r10r1`：step0→5000、`EXIT=0`、`runtime PASS/5000`、cadence 1000–5000 全在、verifier `VERIFY=0`、CPU strict-load 通过、O8 83/83）；但 **M_CAT 在第 4928 个 update 被本轮驱动的 4 h `timeout` 上限停止**（`EXIT=124`、SIGTERM 15、无 checkpoint、supervisor 未介入、milestones 500 `dense` / 501 `top2` 正常）。按 r10R1 §19「MCL arm retry 禁止」与 r10 §21/§41 立即停止整个 P2 执行并交回：**未重试 CAT、未启动 GATE/XATTN、未运行 deployment check 与 30 个 development unit、未运行 P2 aggregate**；现场 `p2/pretrain/cat/` 与 r10 的 `glt_ref/` 均原样保留。预算已消耗：预训练 **10,082** updates（154 + 5000 + 4928），development 0 unit。详见 §13.13。此前 r10 / P2-E2E 阻断（P2 未完成）：Phase I（P2 development aggregator `scripts/aggregate_mcl_ph_p2.py` + model-free regression，13 passed）已完成、已提交并推送 `fbd78ff`；Phase II 的**第一个正式启动 GLT_REF** 在 step 154 被我自己挂到的外部 stall supervisor 以 SIGTERM 停止（`STOPPED_BY_SUPERVISOR`；该 supervisor 的进度判据只认 MCL runner 才写的 `stages_rank*.log`，GLT dual runner 不写该文件，180 s 静默阈值必然触发）。按 §12/§21/§41 停止整个 P2 执行并交回：**未重试、未启动其它 arm、未改代码、失败现场保留**；**不写 `P2 execution evidence complete`**。P2 预算已消耗：GLT_REF 1 次启动 / 154 updates（无 checkpoint），其余 3 个 arm 0 启动，development 0 unit。详见 §13.12。此前 r9 执行完成（`optimizer_groups` 重复关键字缺陷已最小修复并无模型回归通过；五个 XC/fold0/1-epoch downstream smoke 全部 `exit 0`/`PASS`；五臂 aggregate `status=PASS`、`acceptance=PASS`、5/5/0，`outer_test=NOT_RUN`）——**P1 执行证据完整，待 ChatGPT 审查**；ZCode 不写 `P1 PASS`、不写「正式验收通过」，也不做任何性能/排序结论。此前 r8 部分完成并已停止（GATE/XATTN 当前代码生产 smoke 通过、三臂 deployment checker PASS；downstream 首个 unit `glt_ref` 因**既有代码缺陷**在写 `run.json` 时崩溃，按指令立即停止，未修代码、未重试、未继续其它臂；该轮未写 `P1 PASS`、未写 `CAT smoke fixed`）。r3 第二阶段在 cat 臂导出阶段挂起后按停止条件中止（详见 §13.3 与 [MCL-PH-INCIDENT-r3-cat-export-hang.md](MCL-PH-INCIDENT-r3-cat-export-hang.md)）。r4 在不恢复训练的前提下完成：checkpoint 核验（PASS）、CPU 离线导出（PASS，产物仅为「恢复导出候选」）、四 rank GPU 收尾复现（**未复现**）。r5 完成四项收尾状态修正与针对性验证（14 passed），但唯一一次真实运行因预建输出目录触发防覆盖守卫而中止（0 update）。r6 的唯一一次 cat 两步运行**完成了 2 个 update**（`resume_00002.pt` 与 r3b 逐字节相同），随后在导出窗口抛出 `IndexError` 并进入收尾死锁——**r3 挂起由此复现并定位到具体缺陷**（`shared` 变量遮蔽），另有看门狗 180 s 自动停止不可达的缺陷；按指令未改代码、未重试，详见 §13.5/§13.6。r7A 按授权**只**修这两个已定位缺陷（导出身份变量改名 + 停滞进度只认 `stages_rank*.log` 真实 mark）并做无模型回归（8 passed；GPU/forward/backward/update/训练启动均为 0），**未运行 CAT**，详见 §13.7。r7B 在 r7A 之后完成唯一一次 CAT 两步真实运行（world 4 / microbatch 84 / accumulation 3 / BF16）：2/2 updates、`resume_00002.pt` 与 r3b/r6 **逐字节相同**、**首次产出 `deploy_00002.pt`**、launcher exit 0 + verifier `PASS`（`strict_cleanup`）+ `ALL_ARMS_OK`、`runtime.json` `PASS/cleanup complete/main_returned true`、supervisor `ROOT_EXITED` 未介入、resume↔deploy encoder **170/170 逐张量相同**；但 §8 的 `_mcl_ph_r3_deployment_check.py` **首次运行**暴露其自身名称约定缺陷（`changed_shared_tensors` 恒为 0），该检查未取得 PASS，详见 §13.8。r7C 只修该 verifier（encoder 命名空间映射 + 缺失键报错 + `encoder.fusion.` 前缀）并新增 model-free regression（6 passed），随后**用现有 r7B 产物**（零训练、CPU only）复验：`status=PASS`、`problems=[]`、`shared_encoder_tensors=79`、`changed_shared_encoder_tensors=79`、`shared_encoder_missing_keys=[]`，详见 §13.9。r8 补齐 GATE 与 XATTN 的当前代码两步 smoke（各 1 次启动、2 个 update、launcher/verifier/supervisor/runtime 全 PASS，denominators 1008/1008、dense 路由、无 stall），三臂 deployment checker **PASS**（三臂 `shared_encoder_tensors=79`、`initial_value_difference_count=0`、init SHA 相同）；downstream 五臂在第一个 unit `glt_ref` 处失败（训练与评估正常完成，写 `run.json` 时报 `TypeError: dict() got multiple values for keyword argument 'optimizer_groups'`，由 `3d66198` 引入、自引入以来首次被执行），按指令 STOP，详见 §13.10。r9 按授权**只**修该缺陷（抽成 `build_summary` 纯函数，diff +14/−4），无模型回归 **4 passed**（GPU/forward/backward/update/训练启动均为 0），随后在全新输出根 `results/mcl_ph_20260921/p1/finetune_r9` 完成五个 XC/fold0/1-epoch downstream smoke（全部 `exit_code=0`、`status=PASS`、`executed_epochs=1`、`optimizer_updates=9`、`best_epoch=1`、`pretrain_step=2`、`outer_test=NOT_RUN`）并通过固定五臂 aggregate（`status=PASS`、`acceptance=PASS`、`units_expected=5`、`units_accepted=5`、`units_rejected=0`）；**本轮未运行任何预训练**，r8 失败现场保持原状，详见 §13.11。
- 当前完成度：P0 已完成（r1 定版审计 + r2 受限修订，判定缺陷见 §13.3）；P1 **执行证据完整、待审查**（CAT/GATE/XATTN 三条 MCL 路线两步 smoke、三臂 deployment 验收、五个 downstream smoke unit 与五臂 aggregate 均已通过；接口可用性已闭合，性能结论待 P2/授权，最终验收由 ChatGPT 审查远端 commit 后给出）；P2 已按 r10 授权启动但**在第一个正式预训练 arm 处阻断**（Phase I 已完成并推送；GLT_REF 单次启动被外部 supervisor 误停，其余 arm、30 个 development unit 与 P2 aggregation **均未执行**，等待规划方对第二次启动与 supervisor 策略的决定）；r10R1 已在授权范围内完成阈值边界返修并使 **GLT_REF 重试成功（step0→5000 全量通过验收）**，但 **CAT 因执行侧 4 h 超时上限在 step 4928 处失败**（额度已消耗），P2 再次阻断待授权；P3 未授权。r3 的范围、预算、停止条件与实际基准见 §13.3，r4 见 §13.4，r5 见 §13.5，r6 见 §13.6，r7A 见 §13.7，r7B 见 §13.8，r7C 见 §13.9，r8 见 §13.10，r9 见 §13.11，r10/P2 见 §13.12，r10R1 见 §13.13。
- r1 状态（历史，已被 r2 取代）：**待授权执行**。
- Codex 规划和审查；ZCode 在用户授权后执行。推荐第一次仅授权 P0＋P1，后续阶段必须分别交回审查。
- 文档基准：`dev@0d633d8`，已安全 pull、无远端更新。原 `MCL-PH.md` 为空；用户未跟踪 `.zcodeignore` 不修改、不提交。
- 本文件是用户指定的新方案合同。此前 `3D.md` 的空间 adapter 和 `PH.md` 的候选不是本轮执行任务，不能叠加其预算或自动启动。旧 GLT/PH 产物保留，只作来源明确的参考。
- 当前 `Plan.md` 不存在。本轮不恢复用户此前删除的文件。执行端读取 AGENTS、本文件及届时存在的 Plan；若有其他有效计划，先核对冲突，不凭文档自行产生授权。
- 本计划研究的是冻结开放 Trimer 的有限几何，**不是**多构象、无限链、链间堆积或加工条件建模。允许负结果结束，不承诺 XC 或任何指标提升。

## 1. 问题、证据与论文依据

旧 D2 的开发比较未建立原 GLT 相对 2D-only 的整体收益；旧 PH late-residual 收益接近零。本计划不重启那条残差注入路线，而检验：**空间原子专家是否比化学键主干有效，以及包含 PH 的拓扑信息能否选择有效尺度。** 历史结果不能代替新的匹配 reference。

| 文献 | 可借鉴机制 | 本项目边界 |
| --- | --- | --- |
| [MI-MoE，2026 预印本](https://arxiv.org/html/2601.12637v1)，§3、Appendix A | 距离专家、五描述符拓扑路由、Top-k混合；SchNet专家3层128维 | 本文缩为2/3/4 Å；修订边界归一化与均衡项；非完整复现 |
| [SchNet，2017](https://arxiv.org/abs/1706.08566) | 距离连续滤波与原子残差更新 | 本文增加化学边类型，称SchNet-inspired，不称近两年新模型 |
| [GMU，2017](https://arxiv.org/abs/1702.01992) | 乘性通道门控融合 | 下述原子对齐残差是项目适配；原文非聚合物任务 |
| [FlexMol，CIKM 2025](https://arxiv.org/html/2510.07035v1)，§3.1 | 2D/3D cross-attention与重建训练 | 仅借鉴跨模态交互，不复现共享主干或缺失模态生成 |
| [MMPolymer，CIKM 2024](https://arxiv.org/html/2406.04727v2)，§3.3 | 化学遮蔽与几何去噪预训练 | 同步RU遮蔽、分尺度非键监督是新增实验设计 |

上述机制不构成本项目性能保证。半径子集、特征、损失权重、融合残差0.1、采样上限和晋级门槛均在本合同中预先固定，不宣称是论文最优值。

MI-MoE 的路由是 **Randić、Wiener、efficiency、Betti-0、Betti-1 五类描述符共同决定**，不是PH-only。原文两个均衡项按书面定义重复，本项目只保留一个明确可微的均衡项。聚合物原文划分与本项目不同，不横向比较其绝对分数。

## 2. 锁定架构与来源

```text
单RU原图 → 原O8 → N×512 ──────────────────────────────┐
                                                     │
冻结Trimer重原子 → E2 / E3 / E4 → 各 A×128            │
       └→ 同view拓扑轨迹[31,5] → Router → alpha[3]     │
                                ↓                    │
                    按alpha混合 → A×128              │
                                ↓ 中心原子显式对齐   │
                              N×128 → Gate 或 XAttn ──┘
                                          ↓ N×512
                                   中心原子mean → 属性head
```

不保留旧GLT作为并行隐藏支路，不加载其权重到新专家；O8架构、SPD/path数学保持。预训练遮蔽输入的收紧见§6，必须披露为变化。新模型默认无MD200、无FP、无ALIGN/FGR/ENV/扭转任务；旧baseline例外按原合同保留。

| 来源 | 指定位置 |
| --- | --- |
| 旧baseline配置 | `configs/mts/glt_pred_s3b_b_fp.json` |
| 共同原始初始化 | `results/glt_pred_20260918/s3b_prep/common_init_v1.pt` |
| 预训练split | `results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json` |
| 下游cohort | `data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1` |
| 下游基础几何 | `data/processed/mips_trimer_scage_downstream` |
| 下游static | `data/processed/glt_dual_v2/downstream/dual_static_v1` |
| 下游split | `data/splits/mips_outer5_inner20` |

预训练缓存从既有cohort/配置实际解析，不能按相似目录名猜。P0核验source、样本顺序、映射、split、初始化及已知benchmark重叠声明；不擅自改cohort、不声称独立盲测。复用现有身份检查，不新建通用版本框架。

## 3. 输入与三尺度空间图

### 3.1 Packed batch

```text
原O8字段                     不变；预训练使用遮蔽后的副本
sample_key                   原始bytes，不截断NUL
z, formal_charge, aromatic   [A]，对应同一真实Trimer重原子
pos_view                     [A,3]，本次noisy或微调clean坐标，Å
atom_batch                   [A]
central_atom_index           [N]，按O8真实canonical原子顺序
image_to_canonical           [A]，无对应者=-1
physical_bond_ends           [B,2]，匹配既有物理键行顺序
physical_bond_type           [B]
bond_center                 [B]
geometry_input_valid         [G]
center_atom_readout_valid    [G]
topology_trajectory          [G,31,5]
supervision masks/indices    长度、角度、非键pair的独立目标
```

`A` 不假定为3N；不加入氢，不猜端帽身份。侧RU同名原子保持独立物理状态。中心映射必须覆盖每个真实O8原子恰一次，核对元素、连接与适用立体化学。缺失映射是阻断，不默默降为普通invalid。预登记几何失败仍保留2D训练样本。

形式电荷/芳香性从匹配坐标的真实Trimer化学身份读取或确定性恢复；恢复化学对象不允许重新生成坐标。若不能证明来源，停止P0，不由执行端偷偷改为element-only。

### 3.2 特征编码

节点：三张独立128维embedding相加：Z、formal_charge、aromatic。Z支持1..118并有UNKNOWN/MASK；电荷为-3..3、OTHER/MASK；芳香性false/true/MASK。未知类别映射OTHER并记录，MASK与未知严格区分。每个专家独立embedding。

边：64维固定Gaussian RBF＋8维bond-type embedding。bond type复用既有真实化学类别，另有NONBONDED和UNKNOWN；不能将每条空间边都当化学键。RBF中心 `4k/63 Å`、宽度 `4/63 Å`，公式 `exp(-0.5*((d-c)/s)^2)`。距离类型不附带元素对标签，避免重复泄漏被遮蔽原子身份。

无绝对坐标MLP、无原子序号embedding、无q左右符号、无中心标志embedding。坐标只供距离/PH；中心标志仅供对齐/监督/池化。额外杂化、H数、手性特征不进入r1。

### 3.3 专家关系

三个cutoff为 `c=(2,3,4) Å`，包含所有同图 `i!=j, d<=c_k` 的关系；两方向存储，物理无向边只计一次。是嵌套邻域，不是互斥环带。无TopK、无跨图连接、无canonical去重、不强制加入阈值外化学键。不同实际原子重合/非有限坐标为审计失败，不以self过滤掩盖。

截断包络 `f_k(d)=0.5*(1+cos(pi*d/c_k))`，范围内使用、外部为零。在边界上的零权重与拓扑边存在并不矛盾，fixture必须覆盖。孤立原子保留自身表示，空邻居消息为零。

## 4. 三个独立 SchNet-inspired 专家

每专家3个interaction blocks、hidden=128；不共享参数，无dropout，无坐标更新或向量状态。真实三RU原子每层全部更新，最终读中心原子。参数格式：

$$
h_i^{0,k}=E_Z^k+E_q^k+E_a^k,
\quad e_{ij}=[\operatorname{RBF}_{64}(d_{ij}),E_b^k(type_{ij})].
$$

每层从同一旧H计算：

$$
m_i^{l,k}=\sum_{j\in\mathcal N_k(i)}
V_{l,k}h_j^{l,k}\odot F_{l,k}(e_{ij})f_k(d_{ij}),
\quad h_i^{l+1,k}=h_i^{l,k}+U_{l,k}(m_i^{l,k}).
$$

`V:128→128`无bias；`F:72→128→128`、`U:128→128→128`，中间SiLU。U两层无bias，以使空消息产生零增量；F bias=0初始化。所有Linear Xavier uniform，embedding Normal(0,0.02)，无额外LayerNorm进入expert blocks。聚合FP32累加，回写转为状态dtype。输出为各自 `H_k[A,128]`。

三个专家初始参数不相同，但同一编号专家在不同实验组逐张量同初值。独立初始化RNG，不移动O8或训练公共流。小半径不是“纯共价”，反射不变不等于识别手性。

## 5. PH＋图统计路由：先选尺度，后做模态融合

### 5.1 过滤与五描述符

专家只有三个；路由在 `r_t=1.5+0.1t Å, t=0..30` 上取样。用同一实际重原子点云，从0到4.5 Å构造欧氏VR过滤；无向图无self、无额外化学边。下列为固定列顺序：

1. `R_norm=2/n * sum_{undirected edges} 1/sqrt(deg(u)*deg(v))`，无边为0。
2. 修订 `W_norm`：各连通分量大小m≥3时，`w_C=(sum_{u<v in C} shortest_path(u,v)-m(m-1)/2)/((m^3-m)/6-m(m-1)/2)`；m≤2时w_C=0。按各分量 `m(m-1)/2` 加权平均，分母零则0。此为明确项目修订，不照搬断连图不适用的全图归一化。
3. `efficiency= sum_{ordered u!=v} 1/shortest_path(u,v) / (n*(n-1))`；断连贡献0，n≤1取0。
4. `beta0_norm(r)=活跃H0区间数/n`，含essential H0。
5. `beta1_norm(r)=活跃H1区间数/max(1,B1)`，B1为截至4.5 Å出生的全部正长度H1区间数，含右删失区间。

PH用系数域F2、最大同调维度1，必须处理填充三角形（VR的2-skeleton）；不能用图的`E−V+C`冒充H1。活跃定义 `birth<=r<death`；到4.5仍未死亡保留右删失/∞语义，不能把death设成4.5导致最后点错误消失。零长度区间不计；空集合明确为0。坐标/距离计算float64，descriptor送网络前float32。矩阵为[31,5]。

P0在环境中核验PH实现并冻结版本；优先已有可靠实现。不得以近似图环计数、PH失败统一置零、随机丢样本替代。n=0是无效几何而非正常拓扑输入。

### 5.2 Router与Top-2

固定 `Flatten155→Linear128→GELU→Linear3`。描述符已归一化，不再做可学习输入LN或z-score；常量控制使用P_train固定均值[31,5]。独立Normal(0,0.02)权重、零bias，最后层不能全零，以免初始所有样本严格tie。

updates1..500用dense softmax；从update501开始按原始logits保留最大的两个，再softmax。并列按cutoff升序稳定选择，记录tie率，不加入未登记routing noise。微调/推理按部署metadata使用Top-2，不重新warmup。Top-1禁止。P1的step2 smoke导出也显式设置inference routing=Top-2，并另记训练实际只经历dense；strict往返比较的原模型同样设为Top-2，不混用两种模式冒充parity。

$$
\widetilde H_i^{3D}=\sum_{k=1}^3\alpha_k H_{i,k},\quad
p_b=\operatorname{softmax}(a_b),\quad
L_{bal}=3\sum_k(\operatorname{mean}_{b\in V}p_{b,k})^2-1.
$$

同图所有原子共享alpha。均衡用Top-2之前soft概率；硬选择频率仅诊断。有效集V为当前全局有效几何图；无有效图返回有限、可反传零值。所有rank执行同序collective，使用有autograd语义的全局求和，禁止把detach的全局均值当作有正确梯度的实现。梯度累积时均衡按每个分布式microstep有效图计算并平均；**不声称等价于对1008图一次计算**。主任务按其各自有效分母精确累积。

第一版计算全部三个专家，再按alpha混合：属于稀疏混合、密集执行，不声称稀疏加速。独立几何监督仍训练未被选专家。均衡不强迫单样本三尺度等权，不用注意力图替代预测验证。

### 5.3 坐标噪声与成本约束

预训练所有expert edges、RBF、PH、router descriptor及非键target候选来自**同一份本次noisy坐标**；clean坐标只用作监督。复用既有一次高斯扰动，不额外抽噪声。化学feature mask与几何view不是一回事。

在线PH必须先在P0计时；不能为提速换成clean PH。微调clean descriptor允许写独立小sidecar，key必须是sample key，不按流位置猜身份。noisy缓存若存在，必须绑定具体view，不允许sample-key-only复用。旧PH[3,32] sidecar不等价于本合同[31,5]，禁止静默复用。

## 6. 化学遮蔽：跨RU与跨模态同步

复用30% motif采样及既有小图策略，掩码先在canonical RU产生。对每个被mask原子：O8的直接原子输入按mask处理；Trimer中所有可证明对应的副本同时将Z、电荷、芳香性替换MASK。没有对应关系的端帽不能随意映射，必须确认其特征没有直接复制目标身份。

审计O8 atom embedding、token_z、bond_path_features及邻接/路径中的显式端点身份副本。凡直接编码被mask原子目标身份的字段，其对应位置替换原有中性/mask表示；只处理身份编码，不删化学连接或SPD。如原缓存无足够索引定位，P1补最小只读adapter定位；无法证明覆盖则停止，不能仅调用atom_mask就宣称无泄漏。

此项改变预训练输入遮蔽，不修改O8 attention方程；新路线各臂保持一致。原GLT reference保留原合同，因此整体差值包含遮蔽差异，不能单因素归因。目标通过剩余连接/几何推断属于条件学习，不承诺没有任何间接线索。

随机流：保留公共样本顺序、motif mask、坐标噪声流。新pair采样使用独立稳定子流，由seed42、原始key、流位置和固定子流标签派生；不使用Python随机hash。新fusion及3D网络的初始化/随机计算隔离，沿用D2异常恢复的私有流机制。

## 7. 三项预训练任务与精确归一化

### 7.1 两个原子CE

O8输出 `H2[N,512]` 与融合输出 `HF[N,512]` 使用同一个 `Linear512→101` 分类head；标签沿用当前element_index的1..100＋UNKNOWN类别，不增加ENV词表。按真实mask位置先图内平均。

几何有效且有mask的图：`L_atom,b=0.5*CE(H2)+0.5*CE(HF)`；几何无效图：只用`CE(H2)`、权重1。再按有mask的图平均。无mask小图不进入此分母，但不从数据集中删除。训练不是要求两个模态表示相同，不增加InfoNCE。

### 7.2 局部几何

每个专家共享同一套decoder权重（跨三个专家共享，不跨实验组动态共享）：

`bond_repr = MLP384→256→128([hi+hj, abs(hi-hj), hi*hj])`，中间GELU。

`length_head:128→1` 预测标准化 `log1p(clean length)`；`angle_head:256→128→1→tanh`，输入两条原目标键repr的和、绝对差，预测clean cos(angle)。角度/中心键target集合与旧数据定义相同，不新增侧RU监督。decoder不读clean坐标、原始目标距离或距离rank。

长度与角度各用Huber beta=0.5，分别先图内均值；每图在存在的两项之间等权，再对三个专家等权，形成 `local_b`。没有中心内部键不参与局部几何目标；有键无角时只算长度。

### 7.3 分尺度非键距离

每图全部物理无序pair中，要求至少一端为中心重原子、非直接共价键、noisy distance在(0,4]；物理身份去重，不canonical去重。按noisy距离划为(0,2]、(2,3]、(3,4]，分别只由slot E2/E3/E4预测；每bin最多32对均匀无放回采样。

共享decoder `MLP384→256→1`，输入对应expert原子states的和/绝对差/乘积；预测标准化log1p(clean distance)，Huber beta=0.5。先pair均值，再对图内非空bin等权，形成 `nonbond_b`。空间专家的输入图是嵌套图，监督bin是互斥集合，两者不混淆。

`geo_b` 在可用的local/nonbond项之间等权，两项都有时各0.5；只有一项时权重1。最后只在至少有一项目标的图上均值。采样位置及目标集合在新模型各臂一致。same-cutoff对照仍保留原slot监督分工，不能随其4Å输入改变target集合。

归一化统计固定从P_train按key字节序前4096个样本计算：每样本用固定参考view产生候选，统计真实clean目标；长度和非键距离分别一组mu/std，std下限1e-6。参考view使用独立seed42子流，不消费训练随机流。不得用下游或P_val拟合。统计产生一次共用，不能各arm单独计算。

报告直接复制noisy距离的同集合误差、训练集常量预测误差、模型误差；不能仅凭低重建loss声称学习有效。

### 7.4 总损失与梯度路线

$$
L=L_{atom}+L_{geo}+10^{-3}L_{bal}.
$$

融合CE训练O8、fusion、router及选中expert；独立几何loss训练全部expert和decoder，不经过alpha，不假装它能训练router。decoder只存在预训练，fusion与router必须进入部署。FP从新主方案删除，不把其效果解释为已由本轮单独验证；任何FP开关消融另需修订预算。

历史所称“N=0”是中心内部键数为0；为避免与本文N个中心原子的记号冲突，实现中写作`num_center_bonds=0`。此时仍可能有有效中心原子读出与非键监督；无target项不伪造分母。`geometry_input_valid`、`center_atom_readout_valid`、`local_target_valid`、`nonbond_target_valid`各自维护。与旧GLT的N=0行为不同属已声明架构变化。

## 8. 两个融合候选与一个简单控制

所有fusion只接收 `H2[N,512]` 和按PH权重混合、按中心身份对齐的 `T[N,128]`，不看三个独立expert输出、PH原始曲线或alpha，不做第二次尺度选择。标量距离主干保持旋转/平移/反射不变，不能声称区分镜像。

### F_GATE：主候选

$$
u_i=W_2\operatorname{LN}(h_i),\quad v_i=W_v\operatorname{LN}(t_i),
\quad g_i=\sigma(W_g[u_i,v_i,u_i\odot v_i,|u_i-v_i|]+b_g),
\quad h_i^F=h_i+0.1W_o(g_i\odot v_i).
$$

`W2:512→128`、`Wv:128→128`、`Wg:512→128`、`Wo:128→512`。前两者无bias；门bias0、门weight Normal(0,0.001)，初始g约0.5但非严格常量；Wo无bias、Xavier非零初始化。两处LN独立、affine正常初始化。无fusion dropout，无额外FFN，无可学习全局scalar gate。

### F_XATTN：主要竞争者

`Q=Linear512→128(LN(H2))`，`K,V=Linear128→128(LN(T))`，同一个3D LN供K/V；4heads×32，一层，输出 `H2+0.1*Wo(ConcatHeads(softmax(QK^T/sqrt32+mask)V))`。所有projection无bias、Xavier非零初始化。keys是该图全部中心原子，不是单个图向量或三个scale token。仅padding/graph mask，无新位置bias、无FFN、dropout0。

复杂度N²，精确query分块可用，不能chunk内softmax替代全局softmax。只有一个中心原子时合法退化为单key投影，不要求此时Q/K有非零选择梯度。

### F_CAT：简单控制

`u=LN(H2), v=Linear128→512(LN(T))`，`HF=Linear1024→512([u;v])`。用于完整token级融合CE，不能把一个图级向量广播当作对应原子。是原子concat控制，不冒充原GLT的图级concat。非零Xavier、bias0。

所有候选几何invalid时在分支入口跳过并原样返回H2，不依赖zero tensor经过bias来实现fallback。HF mean池化到512，之后统一 `LayerNorm512→Linear512→256→GELU→Dropout0.1→Linear256→1`；head仅在微调新建，同初始张量。**无额外独立z3图向量旁路**，否则无法判断fusion作用。

原KFuse不是本轮候选：单知识槽位softmax恒1，退化为图向量广播；三个expert分别做槽位又改变PH选尺度合同。不删除旧实现。

## 9. 训练、初始化、部署

### 9.1 新路线预训练

```yaml
seed: 42
world_size: 4
microbatch_per_rank: 84
accumulation: 3
global_batch: 1008
optimizer: AdamW
lr: 0.0002
weight_decay: 0
precision: bf16
max_updates: 5000
warmup_updates: 2000
schedule_total_updates: 20000
end_lr: 0.000000001
save_every: 1000
atom_mask_ratio: 0.30
noise_sigma: 0.03
router_dense_updates: 500
router_top_k_afterwards: 2
```

继承既有scheduler公式，不把20k schedule改为5k。PH在CPU对同noisy view计算，网络不对PH算法求导；参数仍端到端从损失训练。CPU开销未验证前不得启动5k。批大小不足不能静默更改，先交回成本报告。

O8从同一个common-init按名称加载；新专家/融合/decoder独立确定性初始化，共享部分复制到各arm。旧common-init未覆盖的共有head按上述初始化新建一次共用初态；不得strict加载旧GLT键权重到原子expert。不同融合器不宣称全参数相同，只保证公共部分。

计算过新任务的训练器不能静默复用旧GLT loss hooks（例如distance_basis/angle_bias不存在）；新统计直接采集本模块已计算张量，不额外forward/backward。

### 9.2 微调

`outer5_inner20`，XC/EPS/EAT×fold0/1，seed42，FULL adaptation；train batch32、eval64；最多30epochs、warmup5、patience10，MSE、train-only scaler。O8/专家/router/fusion lr均1e-5，新head lr1e-4；AdamW decay0.02，但bias、LN和router参数无decay。继承D2的scheduler算法，记录参数分组。

仅validation R²选epoch，保留被选epoch已算出的预测；禁止outer-test特征/标签/预测、OOF、refit。只允许读取test索引验证split完整性。每unit独立best.pt，best.pt不是完整resume。无已验证的全状态恢复就不从best.pt静默继续。

运行时保存公共初值，沿用公共O8/head RNG与3D私有流隔离；pair采样/PH不得移动公共dropout流。data loader顺序一致。训练失败消费与重跑均计预算；报告崩溃不允许自动重训。

### 9.3 部署

保存O8、三expert、router、fusion及其LN；几何decoder/原子CE head不进推理包。metadata固定cutoffs、node/edge类别、PH定义、dense半径、Top-2、fusion类型和源split/初始化。复用现有身份检查并增加新架构必需字段，不新增通用schema体系。

strict-load必须拒绝旧GLT/GALPH部署包与错误fusion模式；预训练融合参数在微调保留，只有属性head新建。输出同时提供 `atom_2d/atom_3d_center/atom_fused/graph_fused/alpha/valid_flags`，训练decoder按需访问全物理原子各expert输出，推理可不返回诊断张量。

## 10. 阶段、预算与对照矩阵

所有阶段当前均未获执行授权。用户仅授权某阶段时，不能启动后续；本计划中的总量不是一次运行许可。

> 历史说明（r3 登记）：本节以下的 P0/P1/P2/P3 段落是 r1 规划时的合同文本。P0/P1 已先后获用户授权并在 r1/r2/r3 执行，P2/P3 仍未授权。段落中的预算是**上限**，不是已执行量；实际消耗见 §13.1–§13.3。

### P0：数据/泄漏/拓扑成本审计

CPU最多60分钟、0模型调用、0 optimizer updates。超过一分钟在tmux。按key固定前512个P_train样本审计映射与五描述符、边数/PH计算时间/拓扑差异；4096个固定train样本计算§7统计与路由常量（同参考noisy view），不扫描outer-test。若时限内未完成，交回部分报告，不静默减样本后填完整PASS。

必须输出：输入电荷/芳香性来源、中心映射、身份mask依赖表、真实N=0 key或缺失声明、PH小图定义审计、三尺度非空率、PH通道方差、单样本prep p50/p95/max与内存。预测在线PH成本若不能支持训练，报告阻断，不提前全量派生缓存。不得因为某个描述符恒定就自动换PH定义或加半径。

### P1：实现与有限验证

先做无模型的失败落盘/launcher/aggregator fixture，再做模型调用。必要测试：

- 中心与物理原子映射、同名副本、边界距离、无跨图边、化学bond-type真实来源；同步mask覆盖显式身份副本。
- 解析PH：孤立点、树、三角形填充、正方形环及对角填充、断连、小n、右删失；与独立小规模reference一致。
- 平移/旋转/置换不变及反射不变边界；noisy input各路径一致，无clean PH旁路。
- 新融合fallback是H2 identity；中心内部键0不造目标；真实N=0优先，找不到明确fixture，不能skip后称覆盖。
- Top-2权重和为1、恰两项选中、dense→Top-2切换、各slot确有独立参数；router存在任务梯度、被拒expert仍有几何loss梯度（有目标的非退化fixture）。
- Gate各路径/XAttn多key梯度、共用CE head、所有部署参数strict往返；不要求所有参数在任何输入上非零。
- DDP部分rank/全部rank无几何，CE仍可backward；全局mask分母、均衡梯度和collective正确。
- split用生产严格校验：协议outer5_inner20、validation_is_test严格false、整数且非bool、范围、无重复、互斥和全覆盖。

硬预算：

| 项 | 上限 |
| --- | --- |
| CPU模型局部测试（toy和真实都计数） | 96 forwards、48 backwards、0steps，CPU30分钟 |
| 四预训练路径GLT_REF/CAT/GATE/XATTN smoke | 各2updates，总8，正式world/batch/schedule，仅提前停止 |
| 额外DDP检查 | 2次分布式forward/backward，0steps：partial-zero与all-zero；4rank总8次rank调用，并在fixture设router step501 |
| 微调smoke | 5臂各XC/fold0一epoch，总5epochs，使用相应smoke包，O8_ONLY取GLT_REF包；30epoch scheduler提前停止 |

缺陷重跑计入同类预算，耗尽停止；不得挪用P2额度。每个pytest中真实模型调用也要登记，不称“零模型验证”。报告路径先验证，结果先保存、诊断后保存；失败明确非PASS、保留checkpoint/history，launcher立即捕获真实退出码。

P1只验实现可用，不排名；输出PH在线prep及端到端smoke成本。若无法满足显存或新路径每步耗时超过GLT_REF的3倍，交回成本审查，不自行截断邻域、切clean PH或修改microbatch。通过后停止等审查。

### P2：架构/融合开发筛查，另行授权

四条新匹配5k轨迹：

| arm | 预训练/3D | 下游 |
| --- | --- | --- |
| GLT_REF | 原配置原GLT＋FP，common-init，重新5k | 原concat1024 head与D2 FBASE协议 |
| O8_ONLY | 不另训，取本轮GLT_REF的O8 | 新候选同款512 head，无GLT实例 |
| M_CAT | 本合同三expert/五描述符/新任务 | §8 F_CAT |
| M_GATE | 同上 | §8 F_GATE |
| M_XATTN | 同上 | §8 F_XATTN |

预算：4×5000=20000updates；5臂×3tasks×2folds=30units，最多900epochs。各轨迹从原始初始化开始，不继续P1 smoke。共同参数初值相同不是训练后权重相同。GLT_REF是整体旧路线reference，架构/任务/遮蔽/head多项不同；O8_ONLY是双路预训练抽取，非独立单模态预训练。历史结果不能直接充当本轮matched reference。

先建立合格集：CAT/GATE/XATTN各自相对O8_ONLY及GLT_REF，macro3均≥+0.005、XC均值均≥+0.01且XC两折分别为正，任一任务均值退化不超过0.01，并无协议风险。随后选parent：若CAT合格，仅当一个合格高级融合相对CAT的macro3≥+0.002且XC两折均正，才允许替代CAT；没有则选CAT。有可替代候选时在这些候选中选择。若CAT不合格，则在合格的GATE/XATTN中选择。候选选择先取最高macro3；距最高值不足0.002的候选视为工程近似持平，优先GATE。合格集为空即STOP。选择记录所有结果，不能仅报告赢家；高级融合未胜CAT时不声称融合创新成功。

门槛是开发工程筛选，不是显著性。无臂满足整体条件则STOP，不启动P3；不得追加radius/lr/gate sweep。best_epoch撞30标风险，不自动延长。

### P3：PH与多尺度归因，另行授权

冻结P2选出的融合结构和全部训练合同，以其R_FULL为reference，新增三条从原始初值开始的5k：

| 对照 | 唯一规定变化 | 解释 |
| --- | --- | --- |
| R_GRAPH | Router的Betti两列替换为P_train固定均值 | R_FULL−R_GRAPH检验PH输入增量 |
| R_CONST | Router全部五列替换为同一均值轨迹 | R_FULL−R_CONST检验样本相关拓扑输入 |
| R_SAME4 | 三expert输入cutoff均4Å，拓扑轨迹/路由/slot监督保持 | R_FULL−R_SAME4检验不同距离邻域，而非仅参数容量 |

所有替换在descriptor产生后执行；常量不得读取validation统计。参数量、初值、mask/noise、sample顺序、部署步数一致。R_CONST可学全局固定偏好，不等于均匀平均。对照不再额外实例化新loss。预算3×5k=15000updates、18units≤540epochs；不重复训练可证明一致的R_FULL。

分别报告每个配对差。每个机制要标“开发增量已建立”，要求对应macro3≥+0.005、XC两折均正、任一任务均值退化不超过0.01；否则写未建立，不扩大解释。若R_GRAPH不劣于FULL，不能以整体架构优秀声称PH有效，可建议去PH但不自动再训。无外层测试，不宣称正式SOTA。

P2＋P3拟议总上限：35000正式预训练updates，48个开发units、1440epochs；P1预算另计。**不是默认一次全部执行。** 更长预训练、全部八任务五折、额外seed、outer-test、FP/遮蔽/输入特征单项消融与确认预算均不在r1，需审查后的新修订。

## 11. 工程实施与产物

拟新增最小组件（实际存在同职责组件时复用）：

```text
src/dataset/mcl_ph_view.py          # 同view、映射、同步mask、目标、PH
src/modules/mcl_ph.py             # 三expert、router、三种fusion
src/modules/mcl_ph_pretrain.py    # 两个CE＋独立几何decoder
scripts/pretrain_mcl_ph.py
scripts/finetune_mcl_ph.py
scripts/aggregate_mcl_ph.py
configs/mts/mcl_ph_*.json
tests/test_mcl_ph_*.py
```

这些不是现有CLI。实现后交付--help和实际smoke命令；先核对函数签名，不根据文档猜接口。避免重写D2已验证的split/RNG/失败记录；默认旧runner与生产模型不变。仅在新adapter副本中mask，禁止原地改共享batch或冻结缓存。

```text
results/mcl_ph_20260921/
  p0/                              # source/mask/topology/statistics/cost
  p1/                              # fixture/smoke，失败尝试不覆盖
  p2/<arm>/pretrain/
  p2/<arm>/downstream/<task>/fold<k>/
  p3/<arm>/...
logs/mcl_ph_20260921/
```

每unit需resolved config、代码/数据/split/初始化与checkpoint身份、运行期公共初值证据、参数组、epochs_run、optimizer_updates、history、best.pt、validation predictions/sample keys、metrics、runtime/真实退出码和资源消耗。仅完整通过写PASS；aggregator拒绝部分文件、NaN、错误mode/step/protocol、smoke复用及outer_test不为NOT_RUN。

监测在既有forward/backward中采集：task分项/有效分母、每expert梯度和update、router概率/硬选择频率/熵、融合增量相对范数、门控分布、PH五列变化、距离直接复制基线、prep/GPU/端到端时间。不为监测新增backward。保存step1/2/500/501/1000/2000/3000/4000/5000统计；step0仅初值记录。权重图和loss下降不是最终性能证据。

## 12. STOP、运行规范与完成定义

立即停止受影响阶段：映射不可证、身份泄漏、clean PH旁路、错误归一化/空均值、PH失败被统一置零、跨图/划分泄漏、不兼容部署静默加载、持续非有限、公共RNG失配、产物覆盖、预算耗尽。允许只读定位；科学定义或预算变化交Codex，不由执行端自行更换。

GPU/worker/超过一分钟的命令只能在tmux session `Uni-Poly` 独立window并保留日志；先查现有任务。不频繁监控：长训练默认每10分钟检查或事件触发，异常不等待下一轮。无授权不停止其他进程。Git按AGENTS先安全pull，显式暂存本轮文件，提交推送并核对远端，不提交缓存/权重/大日志。

阶段完成必须同时有代码/测试或运行证据、预算真实账目、完整产物、失败记录与边界；ZCode只标待审查，Codex才验收。性能问题只由匹配development和后续授权确认回答，不能用smoke或strict-load替代。

当前完成的是**计划文档**。P0/P1/P2/P3均未执行。第一建议授权范围仅P0＋P1；通过后由Codex根据成本和正确性决定下一步。旧PH retention、旧D2及3D.md预算不续用，历史超额不追认。

> 历史说明（r3 登记，2026-09-21）：本段是 r1 规划时状态。事实更新——P0 已执行（r1 定版审计 PASS 499.8 s，r2 受限 Randić 修订 PASS 315.4 s）；P1 已执行但**未完成**（4 臂中 3 臂预训练可用、微调 4/5 unit、五臂验收 PARTIAL）；P2/P3 未执行也未授权。原文保留，不改写当时结论。

## 13. 执行记录与审查区

ZCode在每阶段后追加：实际基准/变更、命令和tmux window、结果与失败、预算消耗、产物/日志、未完成项、待审查问题。不得擦除失败或以新跑覆盖旧跑。Codex追加逐项验收与下一步；当前均为空，尚无新模型验证或预测结果。

### 13.1 r1 执行记录（执行者 ZCode，2026-09-21 UTC，基准 `dev@cca9bd3`）

**授权与范围**：用户仅授权 P0＋P1。预训练 smoke 的 optimizer updates 与微调 smoke 的 epoch 在计划 P1 硬预算表内。

**设备更正（r3，2026-09-21）**：上文 r1 曾记为“GPU 未使用”，该表述**不正确**。证据：`results/mcl_ph_20260921/p1/pretrain/*/runtime.json` 记录 `device="cuda:0"`、`world_size=4`，且 `cat_failed_diagnostics_device` 的失败信息为 `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`——即 r1 的预训练/微调 smoke 实际在 GPU 上运行。P0 审计本身为 CPU（`devices: cpu`、`model_calls: 0`）。更正后 r1 的实际 GPU 用量：预训练 smoke 4 臂（含失败重跑）与微调 smoke 5 次运行，均为 world_size=4 的短程 smoke，未做正式训练轨迹。

**P0：数据/泄漏/拓扑成本审计**

- 命令：`python scripts/audit_mcl_ph_p0.py --mapping-samples 512 --statistics-samples 4096 --output results/mcl_ph_20260921/p0`；tmux session `Uni-Poly` window `mcl_ph_p0`。
- 前 4 次运行部分完成或已被取代，全部保留：`logs/mcl_ph_20260921/p0_audit.log`、`p0_audit_r2.log`、`p0_audit_r3.log`、`p0_audit_r4_superseded.log`、`p0_audit_final.log`，以及 `results/mcl_ph_20260921/p0/audit_superseded_prepatch_conncheck.json`。
- 最终结果：`logs/mcl_ph_20260921/p0_audit_definitive.log`，`status=PASS`，499.8 s（上限 60 min，未超），产物 `results/mcl_ph_20260921/p0/audit.json`、`statistics.npz`。
- 关键量：`connectivity_mismatch=0`、`geometry_invalid=0`、`canonical_atoms=13484`、`canonical_without_heavy=5`（5 个轻氢同位素副本全数计入，未静默丢弃）、`stereo_atoms=0`、`readout_invalid=4`、`bond_category` 无缺失；`length/nonbond` 归一化统计与内存投影见 audit.json。真实 N=0（"无中心内部键"）在审计集合内未出现，声明保留在 audit.json。
- 已知缺陷（r2 修复，见 §13.2）：`statistics.router_mean/router_std` 由未归一化 Randić 求出，列 0 均值 63.6–80.1，违反 §5.1 声明的 [0,1] 区间。
- 未核实项：`p0_audit_final.log` 之前各次的失败原因未逐条归档（仅保留日志原文）。

**P1：实现与有限验证**

- 预训练 smoke：launcher `scripts/run_mcl_ph_pretrain_smoke.sh`，world_size=4，`--stop-after-step 2`，tmux window `mcl_ph_p1_pretrain-`，日志 `logs/mcl_ph_20260921/p1_pretrain_smoke*.log`；driver 日志 `p1_pretrain_driver*.log`。
  - `glt_ref`、`gate`、`xattn`：`EXIT=0`，各 2 updates，`run.json`/`runtime.json`/`deploy_00002.pt`/`resume_00002.pt` 齐全（`results/mcl_ph_20260921/p1/pretrain/<arm>`）。
  - `cat`：3 次失败，均保留失败现场。`cat_failed_common_init`（`ValueError: common initialization state is incompatible: angle_head...`，真实 exit 1）、`cat_failed_diagnostics_device`（`RuntimeError: Expected all tensors to be on the same device... cuda:0 and cpu`，真实 exit 1）、`cat_failed_rng_collective`（rank0 在 checkpoint collective 处等待，被用户停止请求中止；`runtime.json` 无终态，仍为 `RUNNING`，`steps.jsonl` 记录 2 steps）。
  - 结论：**4 臂中 3 臂可用，CAT 不可用**，因此没有 `m_cat` 的微调包。
- 微调 smoke：launcher `scripts/run_mcl_ph_finetune_smoke.sh`，XC/fold0，各 1 epoch（9 optimizer updates），tmux window `mcl_ph_p1_finetune`，日志 `p1_finetune_smoke*.log`。
  - `glt_ref`、`o8_only`、`m_gate`、`m_xattn`：`status=PASS`，`exit_code=0`，`metrics.json` 含 history/val R²。
  - 第 2 次尝试的 `m_gate_failed_collate_interface` 失败现场保留；`m_cat` **未运行**（无预训练包）。计划 5 臂，实际 4 臂。
- `results/mcl_ph_20260921/p1/finetune/aggregate.json` 为 r1 语义产物：因 `metrics.json` 缺 `optimizer_groups` 判 `units_accepted=0`、`status=INCOMPLETE`。**该文件不追溯修改**；r2 以只读补充审计替代（§13.2 修复 4）。
- 无模型/模型测试脚本：`tests/test_mcl_ph_ph.py`、`tests/test_mcl_ph_protocol.py`、`tests/test_mcl_ph_pretrain.py`（含 4-rank DDP 检查）、`tests/test_mcl_ph_view.py`。

**r1 实际预算账目（含超支，未自我追认）**

| 项目 | 计划上限 | 实际（r1） | 结论 |
| --- | --- | --- | --- |
| CPU 模型 forwards | 96 | ≈110 | **超支 ≈14** |
| CPU backward | 48 | ≈70 | **超支 ≈22** |
| optimizer steps（预训练 smoke） | 8 updates | 8（glt_ref/gate/xattn 各 2，cat 2） | 未超（失败重试计入后正好用满） |
| 微调 smoke | 5 epochs | 4（m_cat 未运行） | 未超 |
| 额外 DDP 检查 | 4 rank × 8 次调用 | 4-rank DDP 测试 + 多次失败重跑 | **超支（未逐次计数）** |
| P0 墙钟 | 60 min | 499.8 s | 未超 |
| GPU / 正式实验 | 0 正式实验 | **r1 smoke 使用了 GPU（cuda:0，world_size=4）**，无正式训练轨迹 | 更正（见上） |

超支主因：CAT 3 次失败后的定位与重跑、以及 4-rank DDP 检查的调试；数字来自本轮日志统计，其中"额外 DDP 检查"未逐次计数，标为**未核实**。

**平台与同步状态**

- 提交 `3d66198`（r1 改动）**仅存在于本地**：`git push` 经 HTTPS 连续失败（`GnuTLS recv error (-110): The TLS connection was non-properly terminated.`，含 `-c http.proxy= -c https.proxy=` 与 HTTP/1.1），`origin/dev` 仍为 `cca9bd3`。未使用 force push，未重写历史。
- 用户未跟踪文件 `.zcodeignore` 全程未修改、未提交。

**未完成项**：CAT 预训练；`m_cat` 微调；五臂完整验收；由审查判定修复后是否补跑。

### 13.2 r2 执行记录（执行者 ZCode，2026-09-21 UTC，基准 `dev@3d66198`）

**授权与范围**：用户对 r2 仅授权"最小返修"——5 项修复与针对性验证（A 无模型、B 初始化终态、C 数学一致性）以及受限的 P0 修订。硬约束 GPU=0、`optimizer.step=0`、预训练 updates=0、微调 epochs=0；未启动 CAT，未启动任何正式轨迹。**本轮不宣称 P1 完成。**

**问题 → 修改（逐项文件级）**

| # | r1 审查问题 | r2 修改 | 文件 |
| --- | --- | --- | --- |
| 1 | Randić 少 `2/n` 因子，列 0 可达 ~185，违反已声明 [0,1] | `_randic` 恢复 `2/n * Σ 1/sqrt(deg(u)deg(v))`，docstring 指向 §5.1 | `src/dataset/mcl_ph_view.py` |
| 2 | Pretrainer 级 `self.apply` 递归覆盖 Router/Gate 专用初值 | 只对 `atom_head`/`local_decoder`/`nonbond_decoder` 施加 `_initialise_new_heads`；共享初始状态 schema 升 `mcl-ph-shared-new-init-v2`，v1 直接拒绝并注明原因 | `src/modules/mcl_ph_pretrain.py`、`scripts/pretrain_mcl_ph.py` |
| 3 | 主任务用局部有效图数、balance 未按 accumulation 平均、数学 loss／反向缩放／日志混在一起 | 新增 `effective_graph_counts`（整个 update 的有效图掩码与计数）与 `effective_term`（分离 `math_loss`／`backward_scale`／`loss`）；`objective(report, weights, world_size, denominators, accumulation)` 使用 update 级全局分母；`balance_term` 只返回数学量，`world/accumulation` 缩放改由 objective 施加；空分母为可反传的有限零 | `src/modules/mcl_ph_pretrain.py`、`src/modules/mcl_ph.py`、`scripts/pretrain_mcl_ph.py`（update 前用 AllReduce 汇总各 rank 有效图数） |
| 4 | `metrics.json` 未写 `optimizer_groups`，旧 4 个微调 unit 被判 INCOMPLETE | 后续运行写入 `optimizer_groups=group_evidence`；旧 unit 由新增**只读**审计核验 | `scripts/finetune_mcl_ph.py`、`scripts/audit_mcl_ph_optimizer_groups.py`（新增） |
| 5 | 验收未固定五臂，子集可能被当成通过 | `ACCEPTANCE_ARMS` 固定 5 臂（`xc`/fold0 → 5 units）；全范围无拒绝才 PASS，子集 PARTIAL(exit 5)，缺臂或字段冲突 INCOMPLETE(exit 4)；payload 增加 `acceptance`/`requested_*`/`partial_reason` | `scripts/aggregate_mcl_ph.py` |

**A. 无模型验证**：`tests/test_mcl_ph_ph.py`（新增 `randic_reference` 手算参考：单边 1.0、三角形 1.0、三角形+孤立点 0.75、四点路径、两条不相交边、四星、无边；五列区间与列序检查；断言 Randić 不再是未归一化度数和）与 `tests/test_mcl_ph_protocol.py`（五臂接受、子集 PARTIAL exit 5、缺臂/字段冲突拒绝）。结果：`test_mcl_ph_ph.py` 63 passed；`test_mcl_ph_protocol.py` 24 passed / 2 skipped。

**B. 初始化终态**：`tests/test_mcl_ph_pretrain.py` 新增两项——完整 `MCLPHPretrainer` 构造后 Router/Gate 仍保持声明初值（Router `Normal(0,0.02)` 相对偏差 ≤0.15、bias 全零、末层非零；Gate 权重 `Normal(0,0.001)`、bias 零、`centre.mean()≈0.5`、左右输入可区分），以及 v2 共享初始状态确实由正确初始化的模型生成、v1 产物被拒绝。仅实例化 Router 的做法无法通过（构造顺序覆盖会被该检查捕获）。结果：通过。

**C. 数学一致性**：`tests/test_mcl_ph_objective_math.py`（新增 pytest 入口）以 torchrun 4 rank 驱动 `tests/_mcl_ph_objective_check.py`。**7 passed**，覆盖并给出证据：

- update 级分母 = 各 rank×microstep 有效图数之和（`atom=30`、`geometry=17`，accumulation=3），四 rank 一致；旧"局部分母"公式给出 0.6/0.214/0.5/0.3，与闭式真值 0.4333 可区分。
- 主任务梯度 vs 解析闭式：偏差 3.4e-8；`math_loss`×`world`=`loss`、`backward_scale`=`world/有效图数`（分离检查）。
- 全局无几何：分母 0，梯度恰为 0.0（可反传的有限零），统计 `effective_graphs=0`、`numerator=0`、`loss=0`。
- balance（生产 `balance_term`，accumulation=3）：对照全局闭式梯度偏差 1.5e-8。
- 真实模型（rank 0 两原子/两几何图、rank 1 一原子/无几何、rank≥2 空监督）：update 级分母 `{atom:3, geometry:2}` ≠ 任何 rank 的局部计数；**atom_head 的 rank 平均梯度与单进程全局计算逐元素相等（偏差 0.0，范数 15.935277 相同）**。该阶段关闭随机干扰（`eval()`）——`BondPathO8` 的 dropout 在 train 模式下会使两次前向不可比。
  **证据边界（r3 更正）**：该比较只覆盖 `atom_head` 一个参数块，且是在 `eval()`、`accumulation=1`、balance 权重 0、rank 平均由脚本内一次显式 AllReduce 手工模拟（**未使用 `DistributedDataParallel`**）的条件下得到的。因此它**不能**称为“全模型梯度一致”，也**不是**真实 DDP 验收；O8、专家、几何 decoder、fusion、router 的梯度与真实 DDP 的运行正确性在 r2 均为未检验（r3 已补，见 §13.3）。
- 说明（r3 更正）：真实模型阶段 accumulation=1 以控制本轮 CPU 预算；accumulation=3 的累积语义由驱动生产 `objective` 的合成阶段覆盖。
  r2 原文曾由“不同 rank 的参数使用不同”推断“真实 DDP 不合法”，该结论**已删除**：DDP 的 unused-parameter 处理（`find_unused_parameters=True` 会把未被使用的参数标记为就绪）与**集合通信顺序**是两个独立问题，前者不决定后者。r2 观测到的挂起由**测试参考路径自身**造成——单进程 reference 在 rank 0 上调用带隐式 collective 的生产 forward（`balance_term` 内的两次 AllReduce），而其余 rank 已进入 `all_gather_object`，集合通信次序错配，gloo 在 90 s 超时后中止。真实 DDP 在按 rank 同序调用时不存在该问题；r3 因此新增了真实 DDP 的 partial-zero / all-zero 检查。

**P0 修订（仅 Randić 影响面）**：`scripts/audit_mcl_ph_p0_randic.py`（新增，模型无关，CPU）→ `results/mcl_ph_20260921/p0/statistics_randic_revision.json` 与 `.npz`。4096 样本、315.4 s（≤30 min）、`matches_frozen_sample_set=true`（与冻结审计同一 `ordered_key_sha256=c0402dca…`）。修正后五列全在 [0,1]：

| 列 | min | max | mean | std |
| --- | --- | --- | --- | --- |
| randic | 0.0 | 0.998914 | 0.968945 | 0.055405 |
| wiener | 0.0 | 1.0 | 0.345343 | 0.233511 |
| efficiency | 0.0 | 0.837691 | 0.252635 | 0.130668 |
| betti0_per_atom | 0.004444 | 1.0 | 0.030754 | 0.073456 |
| betti1_per_edge | 0.0 | 1.0 | 0.217646 | 0.281827 |

新报告替换的字段（写在 JSON 的 `supersedes` 中）：`statistics.router_mean`、`statistics.router_std`（旧值同时抄录在报告中：列 0 均值 63.6–80.1），以及 `statistics.npz` 的同名字段；**不替换** `length`/`nonbond` 归一化、`target_counts`、`nonbond_bin_histogram`、`mapping`、`memory`、`ph_definition`、`identity_source`、`cost_projection`（这些由原始距离与图结构导出，不受 Randić 影响）。冻结产物未被修改（`audit.json`、`statistics.npz` 的 mtime 仍为 10:27）。模型与数据路径都不读取 `router_mean/router_std`（`load_geometric_statistics` 只把它们放进统计记录），因此旧的列 0 未污染任何训练归一化；它影响的是 Router 的输入特征与报告统计。

事实观察（供审查，不构成结论）：修正后 Randić 列接近饱和（mean 0.969、std 0.055），Router 直接消费未归一化的 155 维描述符，其余四列承担主要方差。

**补充审计（修复 4 的只读证据）**：`results/mcl_ph_20260921/p1/finetune/optimizer_groups_audit.json`，4/4 `COMPLETE_FROM_OWN_RECORDS`（`run.json` 与 `best.pt` 的 `optimizer_groups` 一致、组名合法、共享字段一致），`acceptance=PARTIAL`（4/5 unit，缺 `m_cat`），`metrics_json_modified=false`、`training_rerun=false`；未重训、未改写旧预测。r1 的 `p1/finetune/aggregate.json` 保持原样。

**r2 实际预算账目（未自我追认超支）**

| 项目 | r2 上限 | 实际 | 结论 |
| --- | --- | --- | --- |
| CPU 模型 forwards | 48 | 已核实 ≥64，另有未核实额外消耗（见下） | **超支 ≥16** |
| CPU 模型 backwards | 32 | 已核实 ≥64，另有未核实额外消耗 | **超支 ≥32** |
| CPU 模型验证墙钟 | 30 min | ≈12.9 min | 未超 |
| P0 修订墙钟（模型无关） | 30 min | 316.7 s（smoke 1.3 s + 正式 315.4 s） | 未超 |
| `optimizer.step` / 预训练 updates / 微调 epochs / GPU | 0 | 0 / 0 / 0 / 0 | 符合 |

Test C 逐次账目（r3 更正；逐条按当时脚本结构与日志阶段推进推算，**不通过重跑恢复账目，也不给伪造的精确总数**）：

| 运行 | 日志 | 已核实 forwards / backwards | 依据 |
| --- | --- | --- | --- |
| check.log（DDP 版，被终止） | `logs/mcl_ph_20260921/r2_objective_check.log` | 8 / 8 | rank 1–3 打印 `model_partial done`，rank 0 已完成主阶段后卡在 reference；脚本结构为 4 rank × 2 microsteps |
| check2 | `r2_objective_check2.log` | 12 / 12 | 脚本结构：主阶段 8 + rank0 reference 4 |
| check3 | `r2_objective_check3.log` | 12 / 12 | 同上（在比较处失败，reference 已跑完） |
| check4 | `r2_objective_check4.log` | 6 / 6 | accumulation=1：主 4 + reference 2 |
| pytest 入口 ×3 | `r2_objective_math_pytest{,2}.log` 与联合运行 | 18 / 18 | 每次 6 / 6 |
| **已核实合计** | | **≥64 / ≥64** | 相对 r2 上限 48/32：forwards 超支 ≥16、backwards 超支 ≥32 |

**未核实的额外消耗**（只列出，不重跑补齐）：① `r2_objective_check.log` 中 rank 0 的 reference 首个前向卡死在集合通信内部，是否计入无法核实；② r2 早期运行的 `tests/test_mcl_ph_pretrain.py`（含 4-rank DDP 测试）与初始化测试的模型调用没有留下日志。因此 r2 原文的“48–56”**不是包含所有运行的可靠总量上界**。超支集中在**测试工具本身**的调试（DDP 集合通信配置、dropout 随机性、fixture 计数一致性），生产代码的修复未因此改动。r2 的消耗不冲抵 r1 的 P1 预算（r1 自身已超支，见 §13.1）；r2 早期运行的 `tests/test_mcl_ph_pretrain.py`（含 4-rank DDP）与初始化测试的逐次模型调用未登记，标为**未核实**。

**未完成 / 待审查**

- P1 仍未完成：CAT 预训练无成功 unit，`m_cat` 微调 unit 缺失，五臂验收只有 4/5（PARTIAL）。
- `results/mcl_ph_20260921/p1/pretrain/cat_failed_rng_collective/runtime.json` 仍为 `RUNNING`（进程被用户停止请求中止，未写终态），保留现场不追溯修改。
- 修复后是否补跑三条既有预训练 arm 与相应微调、以及新增预算，**待本轮审查后由用户与 Codex 决定**；本轮未启动任何训练。
- 同步：r1 提交 `3d66198` 与 r2 提交 `91626cb` 已推送 `origin/dev`（`cca9bd3..91626cb`，非 force），并以 `git ls-remote origin dev` 核对远端确为 `91626cb`。r1 期间 HTTPS 推送因 `GnuTLS recv error (-110)` 失败属暂时性网络故障，r2 复核时同一凭据与代理配置下 fetch/push 均恢复；r1 的未推送事实保留在 §13.1，不追改。

### 13.3 r3 执行记录（执行者 ZCode，2026-09-21 UTC，基准 `dev@88c3f76`）

**计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r3「证据收口＋有界恢复 P1 smoke」** |
| 状态 | **受阻（第一阶段完成；第二阶段部分执行后按停止条件中止，待 Codex 审查）** |
| 授权来源 | 用户 2026-09-21 的 r3 指令（本轮执行范围、预算与停止条件均由该指令给定） |
| 角色 | Codex 规划与审查；ZCode 执行。ZCode 只标「待 Codex 审查」，不宣布验收通过 |
| 基准 commit | `dev@88c3f76`（r2 末尾提交）。开工前 `git ls-remote origin dev` 核对远端 = `88c3f76`，与本地 HEAD 一致 |
| 开工前本地改动 | `MCL-PH.md`、`scripts/audit_mcl_ph_p0_randic.py`、`tests/_mcl_ph_objective_check.py` 为 r2 已改未提交内容；用户未跟踪的 `.zcodeignore` **未修改、未提交**；r1/r2 的初始化、checkpoint、日志与产物一律保留 |
| 预算与停止条件 | 见下表 |

**授权范围、预算与停止条件（登记，防止事后追认）**

| 阶段 | 范围 | 预算 | 停止条件 |
| --- | --- | --- | --- |
| 第一阶段 A | 证据表述更正（r2 的 atom_head 局限、删除 DDP 结论、重列账目、r1 GPU 更正、历史状态标注） | 仅文档 | — |
| 第一阶段 B | P0 审计报告判定修复 + H1 列更名 + 合成记录测试 | CPU、模型无关 | — |
| 第一阶段 C | 针对性数学与真实 DDP 验证（广参数块参考比较、r2 既有证据沿用、一次真实 DDP partial/all-zero） | **CPU 模型验证 ≤24 forwards、≤20 backwards，累计墙钟 ≤20 min；GPU=0；`optimizer.step`=0；预训练 updates=0；微调 epochs=0；不跑整份 pytest** | **任何模型验证失败或超时 → 停止模型调用，只允许只读定位，不自动重跑，交回 Codex，不进入第二阶段** |
| 第二阶段 | M_CAT/M_GATE/M_XATTN 各 2 预训练 updates（共 6）＋ 三臂 XC/fold0 各 1 微调 epoch（共 3） | 新增**不可转移**训练预算 6 updates + 3 epochs；任一训练失败即停止后续训练、不自动重试 | 部署导出/严格加载失败则该臂不启动微调 |
| 第三阶段 | GLT_REF / O8_ONLY / M_CAT / M_GATE / M_XATTN 五臂汇总与验收材料 | 只读 | 缺证据或冲突即 INCOMPLETE，不得以子集充当完整 |
| 全轮禁止 | P2/P3、正式 5k、development、outer-test、OOF、新 seed、新 probe、conformer 生成、改冻结缓存、清理历史 | — | — |

**第一阶段 A：证据表述更正**

| 编号 | 问题 | 修改 | 位置 |
| --- | --- | --- | --- |
| A1 | r2 把「真实模型梯度参考」写成整模梯度一致性 | 明确该参考只覆盖 `atom_head`（eval、accumulation=1、balance 权重 0、手写 rank 平均），不得称整模梯度一致性或真实 DDP 验收 | §13.2 A1 |
| A2 | r2 曾写「不同 rank 使用不同参数 → 真实 DDP 非法」 | **删除该结论**；区分「unused 参数处理」与「集合通信次序」两个独立问题；记录 hang 的真实原因（该阶段的单进程 reference 在其它 rank 处于 `all_gather_object` 时触发了 `balance_term` 的两次集合通信） | §13.2 A2、`tests/_mcl_ph_objective_check.py` docstring |
| A3 | r2 的「48–56」不是全部运行的可靠上界 | 改为逐次可核实账目（≥64/≥64）＋单列未核实项；**不重跑补齐、不给伪造精确总数** | §13.2 表 |
| A4 | r1「GPU未使用」与旧状态表述 | 按 `runtime.json` 与 `cuda:0 and cpu` 报错更正为 r1 smoke 使用 `cuda:0`、world_size 4；旧「P0/P1均未执行」段标注为历史 | §13.1、§0、§12 历史说明 |

**第一阶段 B：P0 修订报告判定修复**

- `scripts/audit_mcl_ph_p0_randic.py` 的 `judge()` 重写：样本身份缺失（自己的或冻结审计的）、身份不一致、样本不完整（`partial` 或 `used != requested`）、任一列 `min/max/mean/std` 非有限、任一列超出声明区间（容差 `1e-6`）或 `within_declared_range` 标记为假，**均不得 PASS**；缺失的冻结身份记为问题而非静默匹配。
- 列更名（仅命名，公式/数据/输入语义不变）：`betti1_per_edge` → **`beta1_norm`**（第 4 列 = 活跃 H1 区间数 / `max(1, B1)`，不是逐边量）。`COLUMNS`、`COLUMN_DEFINITIONS`、`DECLARED_RANGES` 与脚本输出字段统一使用新名；`COLUMN_DEFINITIONS['beta1_norm']` 内保留到 r2 名称的桥接说明。
- 覆盖测试：新增 `tests/test_mcl_ph_p0_audit_judgement.py`（模型无关）——合成记录覆盖 PASS / 缺自身身份 / 缺冻结身份 / 身份不一致 / 非有限 / 超区间（1.001）／区间内舍入仍 PASS / 未完成样本 / 缺列 / 命名桥接 / `_column_summary` 越界标记。**14 passed in 4.90 s**。
- 旧报告处理（**只读**）：**不重算 4096、不修改 `statistics_randic_revision.json`**。以 r3 判定只读复核该报告的结果如实记录：冻结样本身份 `c0402dca…341a` 与旧报告记录的 `ordered_key_sha256` **完全一致**（真实 hash-match 证据成立），五列范围证据保留；唯一「问题」是旧报告沿用 r2 时期列名 `betti1_per_edge` 而 r3 判定按 `beta1_norm` 查找，故报「该列缺失」——**属命名不一致，不是数据缺陷，也不构成「旧报告被判定失败」的实质结论**。不改旧产物、不改旧结论。

**第一阶段 C：针对性数学与真实 DDP 验证**

工具：`tests/_mcl_ph_r3_reference_ddp_check.py`（2 rank gloo，真实 `DistributedDataParallel(find_unused_parameters=True)`，生产 forward/objective）、`tests/test_mcl_ph_r3_reference_ddp.py`（启动入口）、`tests/_mcl_ph_r3_unused_probe.py`（单进程探针）、`tests/test_mcl_ph_r3_payload.py`（模型无关的判定与载荷测试）。

1. **广参数块参考比较**（不再只比 `atom_head`）：比较块为 `encoder.o8`、`encoder.branch.experts`、`encoder.branch.router`、`encoder.fusion`、`local_decoder`、`nonbond_decoder`、`atom_head`；两侧同为构造后 `eval()`、同一初始化（`load_state_dict` 后逐张量 `torch.equal` 断言）、同一输入与精度、关闭随机性；被比较损失为 `atom + geometry`（balance 权重 0，平衡项不计入本验证，由 r2 闭式覆盖）。逐参数区分 `None`／恒零／非零。`partial` 配置实测（`logs/mcl_ph_20260921/r3_reference_ddp_partial.json`）：

| 块 | verified_on | 非零比较 | 两侧恒零 | 两侧皆 None | 本 rank 无梯度 | 最大偏差 |
| --- | --- | --- | --- | --- | --- | --- |
| encoder.o8 | [0, 1] | 74 | 6 | 3 | 0 | 0 |
| encoder.branch.experts | [0, 1] | 53 | 0 | 22 | 0 | 8.97e-44 |
| encoder.branch.router | [0, 1] | 4 | 0 | 0 | 0 | 0 |
| encoder.fusion | [0, 1] | 9 | 0 | 0 | 0 | 0 |
| local_decoder | [0, 1] | 10 | 0 | 0 | 0 | 0 |
| nonbond_decoder | [0, 1] | 4 | 0 | 0 | 0 | 0 |
| atom_head | [0, 1] | 2 | 0 | 0 | 0 | 0 |

   参考梯度为零的块记为**未验证**（`all_zero` 配置下两个 decoder 的 `compared_nonzero = 0`、`verified_on = []`），不作为通过。
2. **r2 已通过的证据沿用、不重跑**：`accumulation=3` 的更新级分母、累加平均与平衡项闭式（`logs/mcl_ph_20260921/r2_objective_check4.log`）保持原样，仅在本轮载荷中标注 `CARRIED_OVER_FROM_R2`、`re_run: false`，并明确规定它**不替代**另外两项判定。
3. **一次真实 DDP partial + all-zero**（生产 forward/objective、正确的 unused-parameter 配置、有限 loss/backward、**不使用手写 AllReduce**）：更新级分母实测 `partial = {atom: 3.0, geometry: 2.0}`（rank 0 贡献 2 原子/2 几何，geometry-free rank 贡献 1 原子/0 几何）、`all_zero = {atom: 4.0, geometry: 0.0}`；`denominators_source = update_level`；loss 有限（partial rank 0 = 5.5373）。
4. **通信次序与超时退出路径（静态核对 + 运行约束）**：所有 rank 以相同顺序执行 forward / objective / backward / reference；被比较路径中除 DDP reducer 外只有 `balance_term` 的两次集合通信（`effective_term` 使用调用方给的更新级分母，不引入集合通信）；进程组超时 90 s、`faulthandler` 300 s 打印栈并退出，故次序不匹配会显式失败而不是挂死。r2 hang 的原因按 A2 记录。
5. **三项判定分开记录**：`analytic_formula_consistency`（沿用 r2，未重跑）、`model_gradient_reference`（本轮广参数块比较）、`real_ddp_runtime`（本轮真实 DDP 运行），载荷中三者并列且互不替代。

**第一阶段的关键发现（含对既有结论的更正）**

- **单进程探针**（`logs/mcl_ph_20260921/r3_unused_probe.log`）：对 `partial` 第二个 rank 的同一 fixture（`geometry=False`、`atom_mask=(True, False, False)`）做一次普通前向/反向，loss 有限 = 4.1697，`report` 计数 `atom 1 / geometry 0 / local 0 / nonbond 0`；参数梯度状态为——`local_decoder` 10/10 与 `nonbond_decoder` 4/4 全为 `None`；`encoder.o8` 83 个中 3 个 `None`、6 个恒零、74 个非零；`encoder.branch.experts` 75 个中 22 个 `None`、22 个恒零、31 个非零；`router` 4、`fusion` 9、`atom_head` 2 全非零。这满足「正确区分 None、零和非零」的要求，也是「哪些块在该 rank 真的没被前向触达」的直接证据。
- **更正一条 r2 遗留推论**：真实 DDP 运行时，本 rank 未使用（`grad is None`）的参数**不会被留成无梯度**。生产配置只设 `find_unused_parameters=True`，`skip_all_reduce_unused_params` 保持默认 `False`，DDP 会跨 rank 归约 locally-used map，仅跳过**所有 rank 都未使用**的桶（`reducer.hpp`：`all_reduce_local_used_map`／`is_unused_bucket`／`should_skip_all_reduce_bucket`）。因此 geometry-free rank 的两个 decoder 拿到的是 **rank 平均梯度**，与「全体 rank fixture 的单进程参考」一致——这正是 `partial` 实测中两个 decoder 在 rank 0/1 都被比较且验证、`unused_on_this_rank = 0` 的原因。据此**撤回**「production update 会在该 rank 跳过该参数」的旧说法；`_verdict` 中相应提示文本改为中性描述（若真出现无梯度则标为「需另行定位」）。
- **`all_zero` 载荷未持久化**：首次启动（含两个配置）的载荷在 pytest 失败输出里被截断，`all_zero` 分支的证据是当次 pytest 断言（`logs/mcl_ph_20260921/r3_reference_ddp.log`：`1 failed, 5 passed`，失败项即本轮已修正的错误断言，`all_zero` 相关断言通过）。按预算与「不自动重跑」要求未再启动，标为**证据仅存于日志**。
- **载荷结构补强**：`partial` 重跑时新增逐 rank 证据（`local_counts`、更新级分母、`numerator/effective_graphs`、`same_initial_state`）与 `--output` 持久化，避免再次出现「结论在日志、逐 rank 证据丢失」；判定逻辑与载荷结构由 `tests/test_mcl_ph_r3_payload.py`（模型无关，10 passed）覆盖，并以合成载荷**重放**启动入口的全部断言，替代无法负担的第二次真实启动。

**第一阶段预算账目（r3，含失败与探针，未冲抵 r1/r2）**

| 项目 | 第一阶段上限 | 实际 | 结论 |
| --- | --- | --- | --- |
| CPU 模型 forwards | 24 | 19（首次启动 12 + `partial` 重跑 6 + 探针 1） | 未超 |
| CPU 模型 backwards | 20 | 19（同上） | 未超 |
| CPU 模型验证墙钟 | 20 min | ≈1 min（启动均为十秒量级；pytest 自报 10.68 s） | 未超 |
| GPU / `optimizer.step` / 预训练 updates / 微调 epochs | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 符合 |
| 整份 pytest | 不运行 | 只运行模型无关测试文件（`tests/test_mcl_ph_r3_payload.py`、`tests/test_mcl_ph_p0_audit_judgement.py`） | 符合 |
| 重算 4096 样本 / 改旧报告 | 禁止 | 未重算、未修改旧报告 | 符合 |

命令 / tmux window / 日志：

| 内容 | 命令（要点） | tmux window | 日志与产物 |
| --- | --- | --- | --- |
| 首次真实 DDP 启动（两配置） | `python -m pytest tests/test_mcl_ph_r3_reference_ddp.py -q` | `Uni-Poly: mclph_r3_ddp` | `logs/mcl_ph_20260921/r3_reference_ddp.log` |
| `partial` 证据重跑 | `python -m torch.distributed.run --standalone --nproc_per_node=2 tests/_mcl_ph_r3_reference_ddp_check.py --configurations partial --output …` | `Uni-Poly: mcl_ph_r3_refddp` | `logs/mcl_ph_20260921/r3_reference_ddp_partial.{log,json}` |
| 单进程 unused 探针 | `python tests/_mcl_ph_r3_unused_probe.py` | `Uni-Poly: mcl_ph_r3_probe` | `logs/mcl_ph_20260921/r3_unused_probe.log` |
| 模型无关测试 | `python -m pytest tests/test_mcl_ph_r3_payload.py tests/test_mcl_ph_p0_audit_judgement.py -q` | 前台（秒级，无 GPU/worker） | 10 passed / 14 passed |

**第一阶段未完成 / 待审查**

- 三层验证必须分别审查：解析式一致性（沿用 r2 证据）、模型梯度参考（本轮扩块）、真实 DDP 运行（本轮 partial + all_zero），不得互相替代。
- `all_zero` 的逐 rank 载荷未持久化（仅日志断言）；如需正式证据，建议下一轮以明确的 forward/backward 预算执行一次 `--configurations all_zero --output …`。
- 启动入口改动后未再真实启动（预算所限），其断言以模型无关的合成载荷重放验证；如实标为「未二次真实运行」。
- 不是性能结论：本阶段无任何预测评估，也未比较 R²，不宣称任何提升。

**第二阶段执行记录（部分执行；cat 臂在导出/收尾阶段挂起，按停止条件中止）**

前置：不训练 fixture（要求 3，先验证再训练）

- 命令 `python -m pytest tests/test_mcl_ph_protocol.py tests/test_mcl_ph_r3_payload.py tests/test_mcl_ph_p0_audit_judgement.py -q`（tmux `Uni-Poly:mcl_ph_r3_fixture2`，日志 `logs/mcl_ph_20260921/p1_protocol_fixture_r3b_final.log`）：**51 passed in 150.88 s**。
- 其中 `tests/test_mcl_ph_protocol.py` 含两个真实 torchrun world-4 启动（缺数据、不训练），覆盖：输出目录已存在时 rank 0 写 `runtime.json` 状态 `FAILED`、`exit_code` 与进程退出码一致、错误为 `FileNotFoundError`，四个 rank 各写 `runtime_failure_rank{0..3}.json`，stdout 末行 `FAILED`，**不写 `run.json`**，日志中不出现 `ALL_ARMS_OK`；输出目录尚未建立时只在 stdout 报 FAILED 且不留下目录。即「结果先写、诊断失败不丢结果、失败不写 PASS/ALL_DONE」在无训练 fixture 下成立。
- 修复前那次运行（`logs/mcl_ph_20260921/p1_protocol_fixture_r3b.log`）为 `1 failed, 25 passed`：失败项是该测试自身在**非 torchrun** 下启动 runner（runner 在世界大小守卫处退出，早于建立输出目录，故契约的 `is_dir()` 守卫无法记录），属测试脚手架缺陷，不是 runner 契约缺陷；改为 torchrun world-4 fixture 并补一条「目录未建立」用例后全绿。该失败与修复如实保留。

共同初始化（要求 1；构造级检查，0 forward / 0 backward / 0 `optimizer.step`）

- 工具 `tests/_mcl_ph_r3_init_check.py`：只构造三臂模型并调用生产 `apply_shared_init`，不调用前向；tmux `Uni-Poly:mcl_ph_r3_initcheck2`，载荷 `logs/mcl_ph_20260921/r3_init_check.json`，**status PASS、problems 为空**。
- 产物 `results/mcl_ph_20260921/p1/pretrain_r3b/shared_new_init.pt`：schema `mcl-ph-shared-new-init-v2`、seed 20260921、`source MCLPHPretrainer(fusion=gate)`、95 张量、sha256 `499309392d578daf…b85a`；与 cat 臂 `step_0000.json` 记录的 `shared_new_initialization` 哈希与张量数**逐项一致**——运行所用即该文件。
- 三臂接受同一份 v2：共同块（`encoder.branch.*`、`atom_head.*`、`local_decoder.*`、`nonbond_decoder.*`）实测各 95 张量，逐张量 `torch.equal` 应用 95/95；跨三臂比较 95 个共同张量**零差异**。
- 专属初始化保留：融合参数不在共同初始化内（三臂融合张量集合为 cat 8、gate 9、xattn 8 个，互不相同），应用共同初始化后融合张量**零变化**。声明初始化保留：Router `net.0.weight` std 实测 0.019960（声明 0.02）、`net.0.bias` 最大绝对值 0；gate 臂 `fusion.gate.weight` std 实测 0.0010002（声明 0.001）。
- v1 被拒绝且量化了缺陷：`results/mcl_ph_20260921/p1/pretrain/shared_new_init.pt`（schema `mcl-ph-shared-new-init-v1`、sha `fc4e236ca24b14ce…`）的 Router `net.0.weight` std 实测 **0.083650**（≈ 声明值的 4 倍，即被递归 `apply` 覆写后的状态）；`apply_shared_init` 明确拒绝它并给出「v1 predates the r2 initialization fix」信息。**未用任何 v1 或 r1 checkpoint 恢复。**
- 受控变化（AGENTS §5 对照）：三份 arm 配置逐键比较，**唯一差异键为 `fusion_mode`**；其余 37 个键（world size 4、microbatch 84、accumulation 3、global batch 1008、bf16、AdamW lr 2e-4 / wd 0、warmup 2000、schedule 20000、seed 42、cutoffs [2,3,4]、dropout 0.1、balance 1e-3、router dense 500 → Top-2、common/statistics 产物路径）完全一致。

启动、消费与产物

- 命令（tmux `Uni-Poly:mcl_ph_r3_pretrain`，日志 `logs/mcl_ph_20260921/p1_pretrain_smoke_r3b.log`）：`ARMS="cat gate xattn" UPDATES=2 NPROC=4 PREP_WORKERS=12 OUTPUT=results/mcl_ph_20260921/p1/pretrain_r3b LOG=logs/mcl_ph_20260921/p1_pretrain_smoke_r3b.log PYTHON=$(command -v python) bash scripts/run_mcl_ph_pretrain_smoke.sh`。
- 时间线（launcher 日志）：`12:56:02` 启动 cat；四个 rank 均写出 step 1 与 step 2 记录（step 1 用时 14.67 s，其中数据准备 11.4 s；step 2 用时 0.36 s）；`12:56:42` 写出 `resume_00002.pt`；此后**在导出/收尾阶段挂起**；`13:02:08` 中止。
- 已消耗 **2/6 预训练 updates（仅 cat）**；gate、xattn 与三条微调臂 0。产物保留现场、未补写终态：`results/mcl_ph_20260921/p1/pretrain_r3b/{shared_new_init.pt,cat/{run.json,runtime.json,step_0000.json,steps.jsonl,resume_00002.pt}}`，**无 `deploy_00002.pt`**，`runtime.json` 仍为 `RUNNING`（进程被中止，未写终态）。
- 运行期实测（全部取自既有 forward/backward，未为监控额外加一次 backward）：`update_denominators = {atom: 1008, geometry: 1008}`（= 84×4×3，与锁定的 world/microbatch/accumulation 一致）、`denominators_source = update_level`、`accumulation = 3`、`weights [1, 1, 0.001]`、`router_mode = dense`、loss 有限、`grad_total_preclip ≈ 63.6`；`grad_norms` 覆盖 `router 0.1267`、`fusion 13.04`、`expert_2a/3a/4a ≈ 7.83/8.03/7.99`、`atom_head 10.76`、`local_geometry_decoder 0.200`、`nonbond_decoder 0.067`、`o8_2d_encoder 59.78`——Router 在两次生产更新中确实取得非零梯度。

cat 挂起证据（快照 `logs/mcl_ph_20260921/p1_pretrain_r3b_hang_evidence.txt`）

- rank 0 主线程处于 `futex_wait_queue`，8 s 采样内 0 字节 IO、无 deploy 文件描述符、GPU 0 利用率 0%；rank 1–3 状态 R、CPU 94–97%、GPU 1–3 利用率 100%（NCCL busy-wait）。
- `py-spy`、`/proc/<tid>/stack`、`gdb -p` 均因权限（`ptrace_scope`）不可用，**未能取得 Python 栈，未定位到具体代码行**：挂起**未根因化**。
- 与 r1 的 `cat_failed_rng_collective` 处于同一阶段（step 记录写入后、导出完成前）；本轮**不声称**两次挂起同因。
- 停止手段：先 `SIGINT`（20 s 内无响应，说明阻塞在 C 调用而非 Python 循环）→ `SIGTERM` 关闭 torchrun 与四个 rank。launcher 如实记录 `=== ARM=cat EXIT=1 ===` 与 `=== ABORT after cat (exit 1); no rerun within this budget ===`，日志中**无 `ALL_ARMS_OK`**（先写结果、再写 ALL_DONE 的退出码契约在真实失败中成立）。

要求 (5) 的收集结果（分项）

- 已满足：路由模式、更新级全局分母（atom/geometry = 1008）、专家/路由/融合梯度、平衡项与累加平均、实际更新与 `resume_00002.pt`、真实退出码、时间（step / forward_backward / preparation seconds）、每 rank `valid_graphs`/`target_counts`、PH 五列轨迹统计（`trajectory.per_column_mean/std`）、专家更新范数、融合增量范数与 `distance_copy_baseline`。
- **未满足**：Router logits、soft 概率、熵与硬选择**未收集**——四个 rank 一致地 `diagnostics.router = {}`、`monitoring.router = null`。静态定位根因：`MCLPHBranch.collect_diagnostics` 从未被设置，`MCLPHEncoder._set_fusion_diagnostics`（`src/modules/mcl_ph.py:431-433`）只把开关传给 `self.fusion`，而 `MCLPHBranch.forward`（`src/modules/mcl_ph.py:285-287`）读取的是分支自身的 `collect_diagnostics`（`:241`）；r1 产物同样为空。最小修复（下一轮候选，**本轮未改模型**）：在 encoder 中同时设置分支开关。
- **未满足**：记录中没有内存字段（`runtime.json`、`step_*.json` 均无显存/RSS 指标），只能靠外部 `nvidia-smi` 采样；本轮未做外部采样，故不给出内存数字。

第二阶段预算账目

| 项目 | 第二阶段授权 | 实际 | 说明 |
| --- | --- | --- | --- |
| 预训练 optimizer updates | 6（三臂各 2） | **2**（仅 cat） | gate/xattn 未启动 |
| 微调 epochs | 3（三臂 XC/fold0 各 1） | **0** | 未启动 |
| 其他训练 / 重试 | 禁止 | 0 | 无重试、无额 run |
| 监控用额外 forward/backward | 不允许 | 0 | 监控数据全部取自既有 forward/backward |
| 构造级初始化检查 | 未单列（CPU、无前向） | 3 次模型构造，0 forward / 0 backward / 0 step | `tests/_mcl_ph_r3_init_check.py` |
| 不训练 fixture | 要求 3 前置 | 51 passed（含 2 次 torchrun world-4 失败记录 fixture） | 无训练 |

不冲抵 r1/r2：上述 2 次 update 与 r1/r2 的消耗分别计账。

第三阶段：未执行

- 五臂汇总未启动：M_CAT 缺 `deploy_00002.pt`（导出未完成），M_GATE/M_XATTN 未运行；按指令不得以子集充当完整，故本轮**没有生成任何验收材料**，状态只能是 PARTIAL/INCOMPLETE。
- `scripts/aggregate_mcl_ph.py` 未运行、未改动；本轮**未**添加来源映射（无新臂产物可映射），也未复制任何产物、未改动旧 metrics。
- `tests/_mcl_ph_r3_deployment_check.py` 已写好并通过 `py_compile`，但**未运行**：不存在任何 `deploy_*.pt`，要求 6 的严格加载验证没有对象。

第二阶段未完成 / 待审查

- cat 臂导出/收尾阶段挂起**未根因化**（无 Python 栈）；与 r1 同阶段、同因未知。是否授予新的定位预算（例如导出阶段的分阶段轻量日志，或把 `faulthandler` 覆盖到导出路径）由 Codex 决定。
- 因 cat 无 deploy 包，三条 MCL 微调臂与五臂验收**均不具备启动条件**；本轮已在停止条件处停止，未自动重试、未进入 P2/P3。
- 要求 (5) 的 Router logits/概率/熵与内存指标两项未满足，属实现缺口而非本次运行偶发；需下一轮以最小改动补齐。
- 不是性能结论：两次 update 的 loss 只是训练期观测量，本轮未做任何预测评估、未比较 R²、不宣称提升。

**同步（r3）**

- 第一阶段提交 `56f6071`、第二阶段提交 `89789cb`，均推送 `origin/dev`（`cca9bd3..89789cb`，非 force）；以 `git ls-remote origin dev` 核对远端确为 `89789cb3a5435dc278d2ef1555749cf3700dc820`。
- 本次推送第 1 次因 `Failed to connect to github.com port 443 … Connection timed out` 失败，第 2 次成功；`git pull --ff-only origin dev` 同期遇到 `GnuTLS recv error (-110)`，故本轮提交基于开工时已核验的本地 `88c3f76` 基线、未含远端新提交（拉取失败已如实记录，不用 force、不重写历史）。用户未跟踪的 `.zcodeignore` 未修改、未提交。


### 13.4 r4 执行记录（执行者 ZCode，2026-09-21 UTC，基准 `dev@08531dd`）

**计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r4「导出挂起零更新定位＋监控缺口最小修复」** |
| 状态 | **执行完成（待 Codex 审查）**：A 通过、B 未复现、根因未定位，见本节「三～八」 |
| 授权来源 | 用户 2026-09-21 的 r4 指令（本轮范围、预算与禁止项均由该指令给定） |
| 角色 | Codex 规划与审查；ZCode 执行。ZCode 只标「待 Codex 审查」，不宣布验收通过 |
| 基准 commit | `dev@08531dd`（r3 末尾提交）。开工前 `git ls-remote origin dev` 核对远端 = `08531dd`，与本地 HEAD 一致 |
| 开工前本地改动 | 仅用户未跟踪的 `.zcodeignore`（**不修改、不提交**）；r1/r2/r3 的初始化、checkpoint、失败目录、日志、统计与 `runtime.json=RUNNING` 一律保留 |
| 同步 | `git pull --ff-only origin dev` 首次直连失败（`GnuTLS recv error (-110)`／443 超时），改用已配置代理后成功：**Already up to date**（HEAD 仍为 `08531dd`），故本轮无未合并的远端工作 |

**r4 范围、预算与停止条件（登记，防止事后追认）**

| 项目 | 内容 |
| --- | --- |
| 允许 | 只读检查 r3 现场/checkpoint/日志；导出与收尾路径的最小诊断与针对性修复；Router 监控开关与内存记录修复；下述有界零更新验证 |
| 禁止 | `optimizer.step`、训练 backward、预训练 update、微调 epoch；重跑 CAT 两步；启动 GATE/XATTN 或任何微调；P2/P3、outer-test、构象生成、冻结缓存修改；改模型数学/初始化/描述符/损失/batch/schedule；把独立导出成功追记为原训练 PASS |
| A. CPU 离线导出 | **最多 1 次，墙钟 ≤5 min**；从已核验 resume 提取 encoder 状态经生产 deployment 逻辑写入新的 r4 候选路径；不覆盖原目录、不调用训练循环、不消费数据流；strict-load 并逐张量核对。**只验证离线导出可行性，不证明分布式挂起已解决**；超时或失败即停止实际模型定位、交回、不自动重试 |
| B. 四 rank GPU 收尾复现 | **最多 1 次，墙钟 ≤3 min**；仅当 A 成功且静态检查不能排除分布式收尾问题时执行；同 rank/device/backend 布置；从已有 checkpoint 构造状态，**不做 forward/backward/update**；尽量复用生产收尾函数；明确未复现的部分；预置有限超时、栈输出与外部终止。不能复现时只报「本次零更新条件下未复现」，**不得写已修复**；失败或超时后不再进行第二次 GPU 复现 |
| 本轮总计 | optimizer updates=0；backward=0；微调 epochs=0；GPU 实际定位 ≤1 次且 ≤3 min；**CPU 模型 forward ≤8 次**（仅用于监控/部署的局部验证）；CPU 模型验证累计墙钟 ≤10 min；所有失败、参考计算与子模块 forward 均计入 |
| 预算核算 | 启动前核算，不足即停止、不自动追加 |
| 交付限制 | r4 独立生成的 deploy 只能标**「恢复导出候选」**：不能说明 r3 完整 PASS、不能启动微调、不能作为最终 P1 验收材料 |

**三、静态定位（只读）**

1. **真实收尾顺序**（`scripts/pretrain_mcl_ph.py` 的 stop 分支与 `finally`，逐行核对，未按 mtime 推断）：

| 序 | 步骤 | 执行者 | 是否集合通信 | 是否隐式同步 |
| --- | --- | --- | --- | --- |
| 1 | 末步记录构造/打印；`steps.jsonl` 仅 rank 0 | 所有 rank / rank 0 | 否 | 否 |
| 2 | `rng_state()`（含 `torch.cuda.get_rng_state_all()`） | 所有 rank | 否 | 读 CUDA RNG 状态需与设备一致；**是否内部同步未在随包头文件核实**（`.cpp` 未随包发布） |
| 3 | `dist.all_gather_object(states, rng_state())` | 所有 rank | **是** | NCCL 输出需取回，隐含设备同步 |
| 4 | `save_checkpoint(resume_00002.pt)`（`.tmp` + `os.replace`） | **仅 rank 0** | 否 | 否（本地 IO） |
| 5 | `deployment_package`：170 个 encoder 张量 `detach().cpu().clone()` | **仅 rank 0** | 否 | **是**（D2H 复制的同步语义） |
| 6 | `save_checkpoint(deploy_00002.pt)` | **仅 rank 0** | 否 | 否（本地 IO） |
| 7 | `dist.barrier()` | 所有 rank | **是** | 是 |
| 8 | `runtime.json` 置 PASS（现增记 memory） | **仅 rank 0** | 否 | 否 |
| 9 | `finally:` `source.close()` → `destroy_process_group()` | 所有 rank | destroy 为集合收尾 | source.close 为本地 LMDB env/注册表关闭（`frozen_store.py`：`_READ_ENV_REGISTRY_LOCK` + `env.close()`；`cache_lifecycle.py`：`flock(LOCK_UN)`） |
| 10 | `main()` 返回 → DataLoader 迭代器析构 → 12 个 prep worker 收尾 | 所有 rank | 否 | torch 2.8.0：`w.join(timeout=MP_STATUS_CHECK_INTERVAL=5.0)` 后对残留进程 `w.terminate()`，**有界** |

   核对结论：**收尾路径不存在 rank 条件集合通信**（集合操作只有第 3、7 步与第 9 步的 destroy，三者都要求全 rank 参与）；生产未设置进程组超时（用 NCCL 默认值），这与 r1/r3 观察到的「长时间无进展」一致但本身不构成故障。

2. **`save_checkpoint` / `deployment_package` / DataLoader 生命周期**：`save_checkpoint` 为临时文件 + `os.replace`，无锁、无 IPC；`deployment_package` 是纯字典构造（本轮只加可选 `progress` 观察钩子）；DataLoader 用 `persistent_workers=False`、`pin_memory=False`，worker 收尾在**进程组销毁之后**且如上所述有界。未发现关闭等待构成无界阻塞。

3. **checkpoint 核验**（`tests/_mcl_ph_r4_checkpoint_audit.py`，CPU 只读，**PASS / problems 空**）：`resume_00002.pt`（315,570,711 B，sha256 `e91bda10bfb0db5a…`）记录 `step=2`、`next_position=2016=2×1008`、`scheduler={step:2, lr:2.0000e-07}`；`ordered_keys` 911,391 条（sha256 `4e934987…`）；身份块与 `run.json` 完全一致，且 `statistics_sha256`、`shared_new_init_sha256` 与磁盘文件哈希一致；模型 186 张量 / 20,651,691 参数，键集合与 shape 与新建 `MCLPHPretrainer('cat')` 相同，**全部有限**；优化器 1 组（lr 2e-7、wd 0）、186 条 state 全为 `step=2` 且无 NaN/Inf；`rng` 4 条（每 rank 一条），每条含 python/numpy/torch/cuda，**cuda 每条约 4 个设备张量**（即每个 rank 都读了全部 4 个可见设备的 RNG 状态）。
   限制：**文件可加载不等于精确 resume**；采样位置与冻结数据源的一致性未验证（本审计刻意不打开数据源），已在载荷的 `not_verified` 中标明。

4. r1 与 r3 的「同阶段挂起」只作阶段一致记录，**本轮未认定同因**，也未据此两例推断任何共同机制。

**四、最小取证（新增观测，不含机制修复）**

- `StageLogger`：每 rank 独立文件 `stages_rank{rank}.log`，每条 `monotonic/rank/pid/stage/event` 立即 flush；分别在 `loop`、每步 `step`、`rng_gather`、`resume_save`、`deployment_package`、`deploy_save`、`barrier`、`cleanup` 落点。
- **停滞看门狗**：`faulthandler.dump_traceback_later(stall, repeat=True, file=本rank文件)` 在每次 mark 时重新武装（默认 120 s，可用 `MCL_PH_STALL_SECONDS` 覆盖），只 dump 不杀进程，各 rank 写 `stall_stack_rank{rank}.txt`；不依赖 ptrace/gdb，也不改系统安全配置。
- **CPU 复制定位**：`deployment_package(..., progress=...)` 在每次 `.cpu()` 复制前后记录 `tensor_start/tensor_complete` + 张量名，只记录名字，不打印内容、不逐元素。
- **失败落盘顺序**：训练体内的异常现在在 `finally` **之前**写 `failure_rank{rank}.log`（append + flush）与 `runtime_failure_rank{rank}.json`，`phase='before_cleanup'`；最外层处理器保留为兜底，写 `phase='at_exit'`。失败路径不引入任何集合操作。
- **构建与验证的对应关系**：上述 fixture 与 A/B 运行时，`finally:` 中 `source.close()` 与 `destroy_process_group()` 的顺序曾被编辑临时调换；随后已**改回原顺序**（source 先关闭，再销毁进程组，与本轮之前的代码一致），该处处只影响清理次序、不涉及任何被采集字段。fixture 与 A/B 结果按当时构建记录，未据此声称最终构建已复验该顺序。
- **新增同步的自我登记（r5 更正）**：上述观测不新增任何**显式集合通信**（文件写入、看门狗计时线程，均不发起 collective）。但本节先前「不新增任何同步」的表述过宽，r5 予以更正：**把 CUDA 标量取回主机本身可能产生等待**——`_diagnostics` / `_step_diagnostics` / `_gradient_groups` 中的 `.item()`、`float(tensor)`（router 统计、tie rate、`readout_valid_graphs`、loss 与 grad 标量）都会在该设备上隐式等待已入队的计算完成。这与本轮之前就存在的逐步标量记录属同一类取值模式，是既有行为的延续而非新机制；r5 明确**不**为消除此类等待改写训练路径（见 §13.5）。`deployment_package` 的 `.cpu()` 与 `rng_state()` 的既有同步同理，未被本轮引入。**没有把「加 sync/sleep/延长 timeout」当作修复**（外部 `timeout` 与 90 s 进程组超时仅出现在 B 的取证脚本中，不在生产路径）。

**五、零更新验证（预算核算见下）**

A. **CPU 离线导出**（`tests/_mcl_ph_r4_offline_export.py`，1 次，**PASS**，墙钟 **1.63 s**）：从核验过的 resume 装载 encoder 状态，经生产 `deployment_package` + `save_checkpoint` 写入新路径 `results/mcl_ph_20260921/p1/r4_recovery/deploy_00002_recovery_candidate.pt`（81,404,731 B，sha256 `b031880e748bb6ce…`）；以生产 `load_deployment` 严格加载通过（`expected_step=2`、`expected_fusion=cat`，加载后 `inference_mode='top2'`、`router_mode='top2'`）；**170/170 张量与 resume 中对应 encoder 权重逐张量 `torch.equal` 相同**。原目录未被覆盖。
   限制：只证明离线导出可行；未覆盖 CUDA→host 复制、进程组、worker，**不证明 r3 分布式挂起已解决**，输出仅为「恢复导出候选」。

B. **四 rank GPU 收尾复现**（`tests/_mcl_ph_r4_epilogue_replay.py`，`torchrun --nproc_per_node=4`，1 次，**未复现挂起**）：复现阶段全部完成，四 rank 均 PASS，外部 `timeout 175` 未触发（真实退出码 0），看门狗未触发（四个 `stall_stack_rank*.txt` 均为 0 字节）。rank 0 各阶段耗时：进程组 0.01 s、状态装载 0.70 s、DDP 包装 0.40 s、RNG gather 0.01 s、**deployment 包（CUDA→CPU，170 张量）0.07 s**、保存 0.06 s、barrier <0.01 s、销毁 0.09 s；全程 ≈1.9 s。
   复现内容：生产设备/进程组选择、按 checkpoint 声明的设置构造模型、`DistributedDataParallel(find_unused_parameters=True)`、`rng_state()`+`all_gather_object`、rank 0 的 `deployment_package`（真实 D2H）+`save_checkpoint`、收尾 `barrier`。
   **未复现内容（因此未被本轮排除）**：两次训练步本身、优化器与 DDP reducer 的状态、训练消耗过的 RNG 流、每 rank 12 个 prep worker 与 DataLoader、autocast/AMP 状态。
   交叉校验：B 产生的 `deploy_00002_epilogue_replay.pt` 与 A 的候选包**170/170 张量相同**，两包仅在 `source` 溯源字段上不同（文件大小差 720 B 即来自该字段）。
   结论措辞：**「本次零更新条件下未复现」**——不写「已修复」，不写「与 r3 无关」。

**六、根因判定**

- **未定位**：r3 挂起仍无 Python 栈（ptrace 受限），本轮无法给出证实的位置或机制。
- **已由证据排除/削弱的候选**：① 导出代码本身的逻辑缺陷——A（CPU）1.63 s 完成、B（GPU，含真实 D2H）0.07 s 完成；② 「每 rank 读取全部 4 个可见设备 RNG 导致跨设备同步死锁」——B 在同样 4 设备可见下 `rng_gather` 仅 0.01 s 通过；③ DataLoader/worker 收尾等待——torch 2.8.0 中该路径 `join(timeout=5 s)` + `terminate()` 有界，且发生在进程组销毁之后；④ rank 条件集合通信——静态核对不存在。
- **仍候选（均未证实）**：(i) 训练期内存/页回收或主机分配器在高占用下造成的 rank 0 主机侧停顿（本轮零更新环境的内存占用远小于训练时：无优化器状态、无预取批次、无 12×4 个 worker）；(ii) 与训练期并存的 prep worker / 预取队列的交互；(iii) 训练步遗留的设备侧工作与导出复制之间的次序问题；(iv) 一次性驱动/IO 抖动。
   **重要限制**：r3 现场没有内存记录（本轮才补上），因此上述资源类候选**无法用已有数据检验**。
- **本轮没有对导出路径做机制性修复**：因为证据不支持任何具体故障位置。实际改动的只有观测（阶段日志、看门狗、张量进度）与失败记录顺序（真实的记录缺陷：原实现只有在 `finally` 清理成功返回后才会写错误记录）。若后续仍要改导出语义（如 CPU 快照或调整保存顺序），**必须明确标注是「故障修复」还是「未确证机制的规避方案」**，本轮两者都没有做。

**七、监控缺口修复**

1. **Router 诊断开关接通**（`src/modules/mcl_ph.py`）：`MCLPHEncoder._set_fusion_diagnostics` 现在同时把开关传给 `self.branch`（此前只传 fusion，这解释了 r1/r3 每个 run 的 `diagnostics.router={}` 与 `monitoring.router=null`）；`MCLPHBranch._diagnostics` 从已有 forward 收集 **logits（每专家 mean/std/min/max）、soft 概率（mean/std）、熵（mean/std/min/max）、routing mode**，并新增 `router_hard_selection_note`：dense 阶段 `router_hard_selection` 保持 `null` 且注明 `not_applicable: dense routing uses the soft mixture for every graph`，**不伪造硬选择**；Top-2 阶段为每图两个选择的计数（2 图 → 4 个选择）。
2. **内存记录**（`scripts/pretrain_mcl_ph.py::memory_record`）：每步写入 `record['memory']`，字段为 `cuda_peak_allocated_bytes`、`cuda_peak_reserved_bytes`（`window='step'`、`units='bytes'`，窗口在读取后 `reset_peak_memory_stats` 重开），以及 `cpu_peak_rss_bytes`（`cpu_scope='rank_process_peak_rss_dataloader_workers_excluded'`，明确不含 worker）；无 CUDA 时写 `'NOT_MEASURED'` 而非 0。`runtime.json` 的 PASS 记录也带一份。
3. 监控不额外 backward、不改变 RNG/路由/loss/训练状态（下条验证）。
4. **小 fixture 验证**（`tests/test_mcl_ph_r4_monitoring.py`，全部 CPU、无 backward）：开关接通且 `diagnostics.router` 非空、logits/概率/熵字段有限、soft 概率为归一化单纯形（和 ≈1）、dense 阶段硬选择为 `null` 且带 not-applicable 说明、Top-2 阶段给出选择计数；**同状态同输入下监控开/关：`fused`/`atom_states`/`mixed`/`alpha` 逐张量 `torch.equal`，`torch` RNG 状态逐位相同，模型参数相同**（dense 与 Top-2 两种模式各验一次）。首轮 7 次 forward（其中 1 项因**测试自身断言写错**——把 `report['monitoring']` 当作 pretrainer 的键，实际应为 `last_diagnostics['router']`——失败），修正断言后只重跑该项 1 次，**累计 8 次 forward，正好用满上限**；该失败与修正如实记录，模型代码未因该失败改动。

**八、无模型 fixture（先于实际定位执行）**

- `tests/test_mcl_ph_r4_forensics.py`：**5 passed in 31.28 s**——阶段标记立即落盘（模拟外部杀进程后仍在）、张量进度记录到具体张量名、停滞看门狗确实把**本 rank** 的栈写进独立文件、`memory_record` 的窗口/单位/范围字段正确、以及 **torchrun world-4 下训练体内失败在清理前落盘**（`failure_rank{rank}.log` 首条的 `phase` 必须是 `before_cleanup`）。
- 回归：`tests/test_mcl_ph_protocol.py` **27 passed in 149.73 s**（runner 改动后失败记录契约、launcher 退出码契约未破坏）。

**r4 预算账目**

| 项目 | r4 上限 | 实际 | 结论 |
| --- | --- | --- | --- |
| optimizer updates / backward / 微调 epochs | 0 / 0 / 0 | 0 / 0 / 0 | 符合 |
| GPU 实际定位 | ≤1 次且 ≤3 min | **1 次**（B，≈1.9 s，外部 `timeout 175` 未触发，退出码 0） | 未超 |
| CPU 模型 forward | ≤8 | **8**（监控 fixture 7 + 修正断言后重跑 1） | 用满 |
| CPU 模型验证累计墙钟 | ≤10 min | ≈3 min（checkpoint 审计 ~40 s、A 1.63 s、监控 fixture 11 s + 重跑 5 s、构造/装载等） | 未超 |
| A. CPU 离线导出 | ≤1 次、≤5 min | **1 次**，1.63 s，PASS | 未超 |
| B. 四 rank GPU 复现 | ≤1 次、≤3 min | **1 次**，未复现 | 未超 |
| 其他训练 / 重试 / P2·P3 / outer-test | 禁止 | 0 | 符合 |

**命令 / tmux window / 日志与产物**

| 内容 | 命令（要点） | tmux window | 日志与产物 |
| --- | --- | --- | --- |
| checkpoint 核验 | `python tests/_mcl_ph_r4_checkpoint_audit.py` | `Uni-Poly: mcl_ph_r4_ckpt` | `logs/mcl_ph_20260921/r4_checkpoint_audit.{log,json}` |
| 无模型 fixture | `python -m pytest tests/test_mcl_ph_r4_forensics.py -q` | `Uni-Poly: mcl_ph_r4_fixture` | `logs/mcl_ph_20260921/r4_forensics_fixture.log` |
| 协议回归 | `python -m pytest tests/test_mcl_ph_protocol.py -q` | `Uni-Poly: mcl_ph_r4_protocol` | `logs/mcl_ph_20260921/r4_protocol_regression.log` |
| A. CPU 离线导出 | `timeout 300 python tests/_mcl_ph_r4_offline_export.py` | `Uni-Poly: mcl_ph_r4_exportA` | `logs/mcl_ph_20260921/r4_offline_export.{log,json}`；产物 `results/mcl_ph_20260921/p1/r4_recovery/deploy_00002_recovery_candidate.pt` |
| B. 四 rank 复现 | `timeout 175 python -m torch.distributed.run --standalone --nproc_per_node=4 tests/_mcl_ph_r4_epilogue_replay.py` | `Uni-Poly: mcl_ph_r4_replayB` | `logs/mcl_ph_20260921/r4_epilogue_replay.log`；产物 `results/mcl_ph_20260921/p1/r4_recovery/{deploy_00002_epilogue_replay.pt,epilogue_replay_rank*.json,stages_rank*.log,stall_stack_rank*.txt}` |
| 监控 fixture | `python -m pytest tests/test_mcl_ph_r4_monitoring.py -q`（首轮）与单项重跑 | `Uni-Poly: mcl_ph_r4_monitor{,2}` | `logs/mcl_ph_20260921/r4_monitoring_fixture{,_retry}.log` |

**r4 交付状态**

- 产物定性：`results/mcl_ph_20260921/p1/r4_recovery/` 下的两个 deploy 均为**「恢复导出候选」**：**不表示 r3 完整 PASS**，**不能启动微调**，**不能作为最终 P1 验收材料**。
- 未修改 r3 现场：`pretrain_r3b/cat/runtime.json` 仍为 `RUNNING`；新增独立事件说明 `MCL-PH-INCIDENT-r3-cat-export-hang.md` 与 `results/mcl_ph_20260921/p1/pretrain_r3b/cat/incident_operator_stop.json` 记录 launcher 退出、人工终止与证据，未伪造 runner 写出的 FAILED/PASS。
- 未改动模型数学、初始化、描述符、损失、batch 或 schedule；未改冻结缓存；未生成构象；未跑 P2/P3、outer-test、正式 5k；未自动重试任何失败。
- 未执行：任何训练恢复（CAT 是否复用恢复导出、是否追加验证、剩余训练是否恢复，均待 Codex 审查后由用户决定）。

### 13.5 r5 执行记录（执行者 ZCode，2026-09-21 UTC，基准 `dev@cf2199c`）

**计划头（执行前登记）**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r5「收尾状态修正＋一次真实 CAT 两步复现」** |
| 状态 | **受阻（部分交付）**：修正与针对性验证完成；唯一一次真实运行在预备守卫处中止（0 update，未重试），见本节「七～十二」 |
| 授权来源 | 用户 2026-09-21 的 r5 指令（本轮范围、预算、停止条件与禁止项均由该指令给定） |
| 角色 | Codex 规划与审查；ZCode 执行。ZCode 只写「待 Codex 审查」，不宣布验收通过 |
| 基准 commit | `dev@cf2199c`（r4 末尾提交）。开工前 `git ls-remote origin dev` 核对远端 = `cf2199c9fe16aa36d87c82ed45521715cecc4fb3`，与本地 HEAD 一致 |
| 同步 | `git pull --ff-only origin dev` → **Already up to date**（首次直连即成功，未使用代理绕行） |
| 开工前状态 | `git status` 仅 `?? .zcodeignore`（用户未跟踪文件，不修改、不提交）；无 `pretrain_mcl_ph`/`finetune_mcl_ph`/torch.distributed 进程；4 张 GPU 空闲（GPU3 有 490 MiB 常驻，非本轮进程） |

**一、范围（用户指令给定，不自行扩大）**

- 只做两项：① 最小代码修正（本节「二」四项）；② **一次** cat 两步真实运行（本节「四」）。
- 不启动 GATE/XATTN、任何微调、P2/P3、正式 5k 或 outer-test；不改模型数学、初始化、数据语义、超参数、batch、worker 数量；不使用恢复导出候选启动微调。
- 新产物写入**独立目录** `results/mcl_ph_20260921/p1/pretrain_r5/`；不修改 r3 现场（`pretrain_r3b/cat/runtime.json` 保持 `RUNNING` 原样）与 r4 恢复候选。

**二、最小代码修正（四项）**

| 序 | 缺陷 | 修正方向 |
| --- | --- | --- |
| 1 | 完成状态未区分：训练/导出完成、清理完成、真实进程退出成功混为一个 `PASS` | runner 分三级写状态（`TRAINING_COMPLETE` → `cleanup='complete'` → `PASS` 只在 `main()` 返回后）；launcher 在子进程退出码 0 **且** 必要产物齐全 **且** 记录显示清理完成时才允许写 `ALL_ARMS_OK`；异常路径不新增可能无界等待的 collective |
| 2 | Router 诊断 `std` 使用默认无偏估计 | 改 `correction=0`；单图与空集合给出有限统计或 `NOT_APPLICABLE` 显式语义；不改 forward/路由/loss |
| 3 | 内存记录用同一个含糊「step」窗口覆盖 CUDA 与 CPU | 拆成两项各自标注：CUDA = 自上次 reset 以来的 step 窗口峰值；CPU = rank 进程生命周期峰值（不含 workers） |
| 4 | §13.4 的「监控不新增任何同步」表述过宽 | 区分「不新增显式 collective」与「CUDA 标量取回主机可能产生等待」；不为消除等待改写训练路径 |

**三、先做针对性验证的预算与停止条件（执行前登记）**

- 用假进程/合成记录验证四项：清理失败或未完成不得最终 `PASS`；只有真实退出成功才允许完成标记；单图与空集合诊断不产生 NaN；内存窗口标签正确。
- 预算：CPU 诊断函数/子模块 forward **≤4**、backward **0**、optimizer update **0**、累计墙钟 **≤5 min**；**不得**运行整套模型测试文件。
- 停止条件：验证失败即停止，**不自动重跑**，**不进入**真实运行。

**四、CAT 真实运行的预算与停止条件（执行前登记）**

- 预算：CAT optimizer updates **≤2**，启动次数 **≤1**；从与 r3 相同的已核验 v2 初态**重新开始**（不 resume、不续旧轨迹）。
- 保持不变：world=4、microbatch、accumulation、BF16、schedule、数据顺序、噪声、prep workers（12）与训练配置；保留 r4 新增的每 rank 阶段日志与即时 flush、faulthandler 停滞栈、逐张量导出开始/完成、Router 与内存记录、清理前失败落盘。
- 运行前静态核对：命令、目录、权限、初始化与输出路径、报告写入路径。
- 外部总超时 **≤15 min**；收尾阶段连续 **120 s** 无进展 → 产生栈；连续 **180 s** 无进展 → 停止本次进程组并保存现场；**只终止本轮明确的 PID/进程组**。
- 不通过延长超时、减少 worker 或改变保存方式规避复现。
- 任何失败、超时或人工终止：**立即停止**，本轮不重试、不启动其他臂；记录最后完成阶段、每 rank 最后日志、栈、内存、真实退出码与已完成 update 数；无法取得栈也如实报告。

**五、成功后的有限核验（执行前登记）**

- 仅做权重/metadata 核验：两个 updates 与分母、loss、梯度、Router 统计有限；resume 与 deploy 可加载；deploy 与 resume 对应 encoder 张量逐一一致；strict-load 为规定的 Top-2 推理模式；清理记录与 launcher 退出码一致。
- **不**额外运行预测 forward 或微调。即使成功，也只写「本次完整训练条件下两步 smoke 成功，旧挂起根因仍未确定」，不写「挂起已彻底修复」，不自动恢复剩余训练。

**六、本轮硬上限**

| 项目 | 上限 |
| --- | --- |
| CPU 诊断/子模块 forward | ≤4 |
| 训练启动 | ≤1 |
| CAT optimizer updates | ≤2 |
| 额外 backward | 0 |
| 微调 epochs | 0 |
| 其他臂训练 | 0 |

（结果、证据与预算核算见下。）

**七、实际修改（对照「二」；ZCode 实现，未改模型数学/初始化/数据语义/超参）**

| 序 | 文件 | 实际改动 |
| --- | --- | --- |
| 1 | `scripts/pretrain_mcl_ph.py` | `runtime.json` 改为三级状态：训练体与导出完成 → `status='TRAINING_COMPLETE'`, `cleanup='pending'`, `main_returned=false`；`finally` 清理跑到末尾（`source.close()` 后 `destroy_process_group()`，原顺序不变）→ `cleanup='complete'`；只有 `main()` 正常返回后由新增 `finalize_runtime_record()` 写 `status='PASS'`, `main_returned=true`, `process_exit='observed_by_launcher'`。清理未完成时该函数**拒绝**升级为 PASS。失败时 `training_complete` 为假，`finally` 不改写记录；最外层处理器仍写 FAILED。异常路径**未新增**任何 collective（新增的只是文件读写） |
| 1 | `scripts/run_mcl_ph_pretrain_smoke.sh` | 每个臂退出码 0 后追加独立校验 `scripts/verify_mcl_ph_arm.py`；校验不通过 → `ABORT after <arm> (verification N)`、`exit 6`，**不写** `ALL_ARMS_OK`。glt_ref 臂用基础级校验（其 runner 无 cleanup 字段、且本轮不改它），MCL 臂用 `--strict-cleanup`（要求 `cleanup='complete'`、`main_returned=true`、`run.json`/`steps.jsonl`、`resume_00002.pt` 与 `deploy_00002.pt` 齐全、`completed_steps==UPDATES`） |
| 1 | `scripts/verify_mcl_ph_arm.py`（新增） | 无模型、只读产物目录的判定器，输出一行 JSON 证据并在不通过时返回 1；供 launcher 调用，也可用合成记录单独测试 |
| 2 | `src/modules/mcl_ph.py` | `MCLPHBranch._diagnostics`：新增 `graphs = logits.size(0)`；`graphs == 0` 时返回 `router_statistics='NOT_APPLICABLE_EMPTY_BATCH'` 且全部统计字段为 `None`（不再出现空张量最小值报错或 NaN）；`graphs >= 1` 时改用**总体标准差** `std(correction=0)`，单图给出有限 0 并在 `router_statistics_note` 注明「单图总体标准差按定义为 0」；`router_tie_rate` 在非空集合才有值。**同类兄弟修复**（同一缺陷类别、同属仅观测字段）：`FusionGate` 诊断同样加 `gate_statistics` 语义与 `correction=0`，`scripts/pretrain_mcl_ph.py` 的 `_step_diagnostics.trajectory.per_column_std` 同样处理。forward/路由/loss **未改动** |
| 3 | `scripts/pretrain_mcl_ph.py` | `memory_record()` 改为两个独立子记录：`cuda.window='step_since_last_reset'`（自上次 `reset_peak_memory_stats` 起的 step 窗口峰值）与 `cpu.window='rank_process_lifetime'`（rank 进程生命周期峰值 RSS），`cpu.scope` 保留「不含 dataloader workers」；不再出现共用的 `window='step'` |
| 4 | `MCL-PH.md` §13.4 | 「不新增任何同步」更正为：不新增**显式集合通信**；但 CUDA 标量取回主机（`.item()`/`float(tensor)`）可能等待设备上已入队的计算，属既有取值模式；不为此改写训练路径 |

**八、针对性验证（假进程 / 合成记录）**

- 新增 `tests/test_mcl_ph_r5_completion.py`：**14 passed**（28.92 s）。覆盖：完成形态被接受；`TRAINING_COMPLETE`（清理未完成、无 deploy）被拒并列出 `cleanup is 'pending'`/`missing file: deploy_00002.pt`；缺产物被拒；update 数不符被拒；参照臂基础级契约；**真实 launcher + 假 runner** 三种失败形态（记录停在 TRAINING_COMPLETE → `exit 6` 且无 `ALL_ARMS_OK`；子进程失败 → 无标记；外部 `timeout` 杀进程 → `exit 124` 且无标记）与成功形态（`VERIFY=0` 后才出现 `ALL_ARMS_OK`）；看门狗在 180 s 无进展时**只停自己那棵树**并保留现场、在运行正常结束时不动手；Router 单图/空集合与门控单图/空集合诊断的有限性与 `NOT_APPLICABLE` 语义；内存窗口标签。
- 更新 `tests/test_mcl_ph_r4_forensics.py` 的内存断言到新标签：**1 passed**（4.65 s）。
- 本次验证**发现并修复了两个仪器缺陷**：① 监督器把僵尸进程当作存活（`/proc` 仍存在），导致「运行已结束」时不下线；改为按 `/proc/<pid>/stat` 的 `Z` 状态判定。② 停树报告未记录根进程被 `SIGTERM`，与「只终止本次 PID/进程组」的取证要求不符；现将根进程一并记入 `signalled_terminate`。另修正一处测试预期：torchrun 会把子进程退出码 3 掩码为 1，launcher 记录的是 1。
- 预算：CPU 诊断函数/子模块 forward **4/4**（2 次 `_diagnostics` 直调 + 2 次 `FusionGate` 前向，尺寸为 1 行与 0 行）、backward **0**、optimizer update **0**、累计墙钟 **≈34 s**（≤5 min）；未运行整套模型测试文件。

**九、一次 CAT 真实运行：启动 1 次，在预备守卫处中止（0 optimizer update）**

| 项目 | 内容 |
| --- | --- |
| tmux | `Uni-Poly: mcl_ph_r5_cat`（命令结束后 window 自动关闭） |
| 命令 | `ARMS='cat' UPDATES=2 NPROC=4 PREP_WORKERS=12 OUTPUT=results/mcl_ph_20260921/p1/pretrain_r5 LOG=logs/mcl_ph_20260921/r5_pretrain_cat.log MCL_PH_STALL_SECONDS=120 timeout -k 30 900 bash scripts/run_mcl_ph_pretrain_smoke.sh`，同 window 内以 `tests/_mcl_ph_r5_stall_supervisor.py --root-pid <launcher> --stack-seconds 120 --stop-seconds 180` 监督 |
| 开始 / 结束 | 14:22:26Z → 14:22:35Z（≈9 s，远低于 900 s 外部超时） |
| 中止原因 | `FileExistsError: new training requires a new output directory`（runner 预备阶段防覆盖守卫，rank0 判定后广播到四 rank）。根因是**我在执行前静态核对时用 `mkdir -p .../pretrain_r5/cat` 预建了 arm 输出目录**，而该守卫要求新训练的输出目录**不存在** |
| 真实退出码 | torchrun `ChildFailedError` → launcher `=== ARM=cat EXIT=1 ===` → `=== ABORT after cat (exit 1) ===`，**无** `ALL_ARMS_OK`，未进入 `verify_mcl_ph_arm.py`（日志无 `VERIFY=`） |
| 已完成 update 数 | **0**（未读数据、未建模前向/反向、无阶段日志、无 checkpoint、未消耗 CUDA 训练） |
| 失败落盘证据 | `results/.../pretrain_r5/cat/`：`failure_rank{0..3}.log`（`phase='at_exit'`，append+flush，monotonic）、`runtime_failure_rank{0..3}.json`、rank0 `runtime.json` = `FAILED/exit_code=1/phase=at_exit` |
| 收尾监督 | `stall_supervisor.json`：`status='ROOT_EXITED'`、`intervened=false`、`silent_seconds=5.0` —— 正常/快速失败路径下**未误杀**，也未触发 120 s 栈 |
| 本轮修正的验证价值 | 修正 1 的**失败分支在真实 4-rank torchrun 下被验证**：四 rank 各自落 FAILED 记录、rank0 写 `runtime.json=FAILED`、launcher 不写完成标记 |
| 未取得 | 任何关于「两步运行是否再次在导出/收尾挂起」的证据；旧挂起根因与本轮之前一样**未定位** |

**十、预算核算（本轮硬上限）**

| 项目 | 上限 | 实际 | 结论 |
| --- | --- | --- | --- |
| CPU 诊断/子模块 forward | ≤4 | **4** | 用满 |
| 训练启动 | ≤1 | **1**（14:22:26Z，在预备守卫处中止，0 update） | 用满，**不再启动** |
| CAT optimizer updates | ≤2 | **0/2** | 未消耗 |
| 额外 backward | 0 | **0** | 符合 |
| 微调 epochs | 0 | **0** | 符合 |
| 其他臂训练（gate/xattn/glt_ref 等） | 0 | **0** | 符合 |
| P2/P3、正式 5k、outer-test、构象生成、恢复导出候选微调 | 禁止 | **0** | 符合 |
| 外部总超时 | ≤15 min | 9 s（未触发） | 符合 |

**十一、限制与下一步（待用户/Codex 授权，不在本轮执行）**

- 我的操作错误（预建 arm 目录）触发了防覆盖守卫；这不是代码缺陷，也不是 r3 挂起复现。**静态核对应为只读检查**：用父目录探测可写性、不创建 arm 输出目录；下一轮请使用**尚不存在**的新目录（如 `results/mcl_ph_20260921/p1/pretrain_r5b/cat`）。
- 依 r5 指令「任何失败、超时或人工终止：立即停止，本轮不重试」与硬上限「训练启动 ≤1」，本轮**未重试**、未启动其他臂。故「一次真实两步运行」这一目标**未完成**。
- 请求的下一步（待授权）：在相同预算与停止条件下再授权**一次** cat 两步运行（输出目录预先不存在；其余命令、初态、配置、preworker、BF16、schedule、数据顺序均不变），以取得悬挂相关证据；CLEANUP/退出状态的三级语义与监控输出已可用。
- 仍未完成/未定位：r3 导出挂起的根因；`resume`/`deploy` 的第三步核验（本轮未产生新 checkpoint，无法执行）；gate/xattn 预训练与三条微调；五臂验收。

**十二、交付状态**

- 状态：**受阻（部分交付）**——修正 1–4 与针对性验证完成；唯一一次真实运行在预备守卫处中止，0 update，未重试。
- 修改文件：`scripts/pretrain_mcl_ph.py`、`scripts/run_mcl_ph_pretrain_smoke.sh`、`scripts/verify_mcl_ph_arm.py`（新增）、`src/modules/mcl_ph.py`、`tests/test_mcl_ph_r5_completion.py`（新增）、`tests/_mcl_ph_r5_stall_supervisor.py`（新增）、`tests/_mcl_ph_r5_stub_runner.py`（新增）、`tests/test_mcl_ph_r4_forensics.py`（内存断言）、`MCL-PH.md`。
- 未改动：模型数学/初始化/数据语义/超参/batch/worker 数；冻结缓存；r3 现场（`pretrain_r3b/cat/runtime.json` 仍 `RUNNING`，文件清单与 sha 未变）；r4 恢复导出候选；`.zcodeignore`（未跟踪、未提交）。
- 未取得任何性能或收敛结论；本轮不写「挂起已修复」，也不宣称任何 PASS。
- 提交与同步：本轮改动提交为 `c9fcb50`，推送 `cf2199c..c9fcb50 dev -> dev`（非 force），`git ls-remote origin dev` 核对远端 = `c9fcb50c9708e9aaad525fa6246c5ef36bb1190f`，与本地 HEAD 一致；`.zcodeignore`（用户未跟踪文件）未修改、未提交；`logs/`、`results/` 下的运行产物按仓库 `.gitignore` 不提交，只按路径引用。远端在本轮开始与提交前均为 `cf2199c`，无他人未合并的提交。

### 13.6 r6 执行记录（执行者 ZCode，2026-09-21 UTC，基准 `dev@7fc7779`）

**计划头（执行前登记）**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r6「修正操作流程后，一次 CAT 两步运行」** |
| 状态 | **受阻（执行后交回审查）**：唯一一次运行完成 2 个 update 后在导出窗口因 `shared` 遮蔽缺陷抛 `IndexError` 并进入收尾死锁（挂起被复现且定位），按指令未重试、未改代码 |
| 授权来源 | 用户 2026-09-21 的 r6 指令：新增授权「最多 1 次训练启动、CAT 最多 2 optimizer updates」；旧预算不抵扣、不追认；失败即停止、不自动重试 |
| 角色 | Codex 规划与审查；ZCode 执行 |
| 基准 commit | `dev@7fc7779`（含 r5 提交 `8e67680`；`git merge-base --is-ancestor 8e67680 HEAD` = 是） |
| 同步 | `git pull --ff-only origin dev` → **Already up to date**；`git ls-remote origin dev` = `7fc7779c6fa6575125b721d7c02dc81c867b3b7b`，与本地 HEAD 一致 |
| 远端新增 | `7fc7779`（作者 emt11，内容仅把 `.zcodeignore` 纳入版本管理，81 行；由用户本人提交，本轮不修改、不碰该文件） |
| 工作区 | `git status` 干净（无未提交改动）；无 `pretrain_mcl_ph`/`finetune_mcl_ph`/`torch.distributed.run`/看门狗进程；无遗留的 `mcl_ph_r*` tmux window；4 张 GPU 空闲（GPU3 490 MiB 常驻非本轮进程） |

**一、r6 范围**

- 仅做一次 cat 两步真实运行（不修改模型或训练代码、不重跑测试套件、不新增框架）。
- 新增授权：训练启动 **≤1**、CAT optimizer updates **≤2**；额外 forward/backward **0**、微调 epochs **0**。
- 不启动 GATE/XATTN、GLT_REF、O8_ONLY、P2/P3、正式 5k、outer-test 或缓存构建；不使用恢复导出候选或旧 checkpoint 续训；不重新随机生成初始化。
- 本轮只允许必要的计划/执行记录文档修改；若发现需要改代码，**停止并交回**，不边修边重试。
- 旧预算不抵扣、不追认：r5 的 1 次启动（0 update，预备守卫中止）不用于抵扣本轮。

**二、执行前静态核对（只读；结果如下）**

| 检查 | 方法（只读） | 结果 |
| --- | --- | --- |
| 基准含 8e67680 | `git merge-base --is-ancestor` | 是；HEAD = `7fc7779` |
| arm 路径不存在 | `test -e results/mcl_ph_20260921/p1/pretrain_r6/cat` | `EXISTS=no`（创建完输出根目录后复检仍为 `no`） |
| arm 路径非符号链接 | `test -L` | `SYMLINK=no` |
| 仅创建输出根与日志目录 | `mkdir -p results/mcl_ph_20260921/p1/pretrain_r6 logs/mcl_ph_20260921` | 已建；**未创建** `.../pretrain_r6/cat`（留给生产 runner） |
| v2 初态逐字节身份 | `cmp` + `sha256sum`（r3b / r5 / r6 三份副本） | 三份**逐字节相同**，sha256 均 `499309392d578daf7a7baf1d102d7a7f5c652e5a9b42ca2904ac9d83d1fab85a`（schema `mcl-ph-shared-new-init-v2`） |
| 写权限 | 仅在**父目录**探测（`touch`/`rm` 临时文件），未对 arm 路径做 mkdir 或试写 | `PARENT_WRITABLE` |
| 配置与 r3 一致 | 当前 `configs/mts/mcl_ph_cat.json` 与 r3 `runtime.json` 内嵌 `config` 逐键对比（38 键） | **零差异**；`fusion_mode=cat`、microbatch 84、accumulation 3、global_batch 1008、lr 2e-4、wd 0、warmup 2000、schedule_total_steps 20000、end_lr 1e-9、seed 42、cutoffs [2,3,4]、dropout 0.1、balance_weight 1e-3、router_dense_updates 500、top_k 2、noise_sigma 0.03、atom_mask_ratio 0.3、max_optimizer_steps 5000；配置文件自 `3d66198` 起未改动 |
| 数据/缓存路径 | 与 r3 相同的 launcher 默认值 | cohort `cohort_30f17b59bc5862a1`、cache `mips_trimer_scage`、static `dual_static_v1`、statistics `results/mcl_ph_20260921/p0/statistics.npz`（sha256 `67a77db02822e24b600daaf2a5e141cc69eebb04a354ae5af38f3e9ad873b9c7`） |
| 命令 | 显式 `ARMS='cat'`（不使用默认 `glt_ref cat gate xattn`）、`UPDATES=2`、`NPROC=4`、`PREP_WORKERS=12` | 见「三」 |
| 看门狗 | 沿用已提交的 `tests/_mcl_ph_r5_stall_supervisor.py`（r5 已验证；对尚不存在的 stages 目录 glob 返回空、不报错） | 复用，不新增脚本 |

**三、最终命令（只读核对后原样执行）**

```
ARMS='cat' UPDATES=2 NPROC=4 PREP_WORKERS=12 \
OUTPUT=results/mcl_ph_20260921/p1/pretrain_r6 \
LOG=logs/mcl_ph_20260921/r6_pretrain_cat.log \
MCL_PH_STALL_SECONDS=120 \
timeout -k 30 900 bash scripts/run_mcl_ph_pretrain_smoke.sh
# 同 window 内另起（已验证的）看门狗：
python tests/_mcl_ph_r5_stall_supervisor.py --root-pid <launcher pid> \
  --stages-dir results/mcl_ph_20260921/p1/pretrain_r6/cat \
  --log logs/mcl_ph_20260921/r6_pretrain_cat.log \
  --report results/mcl_ph_20260921/p1/pretrain_r6/stall_supervisor.json
```

- tmux：`Uni-Poly: mcl_ph_r6_cat`（独立 window）。
- 外部总时限 ≤15 min；收尾 120 s 无进展出栈（进程内 faulthandler 立即 flush）、180 s 无进展由看门狗终止**本次进程树**并保存现场；只终止本轮明确 PID 及其后代。
- 保持 r4/r5 既有机制：每 rank 阶段日志与逐张量导出进度、faulthandler 停滞栈、Router/全局分母/梯度/内存记录、真实退出码、清理状态与 launcher 的 `verify_mcl_ph_arm.py` 验收。

**四、停止条件（执行前登记）**

- 任何失败、超时或人工终止：**本轮立即停止**，不自动重试、不启动其他臂或微调；即使失败发生在 0 update，也消耗本轮唯一启动额度。
- 如实保存：实际 updates、最后完成阶段、各 rank 栈、真实退出码、看门狗介入记录。
- 成功结论只能是：「本次完整训练条件下 CAT 两步 smoke 成功；旧 r3 挂起根因仍未确定，未证明问题永久消失。」
- 若失败，明确区分**预备守卫/训练错误**与**导出挂起**，不混为同一故障。

（实际运行、验收与预算核算见下。）

**五、实际运行（唯一一次启动，14:29:0xZ → 14:33:04Z，约 3.5 min）**

| 时间（UTC） | 事件 |
| --- | --- |
| 14:29:0x | `torchrun --nproc_per_node=4` 启动（完整训练条件：world 4、microbatch 84、accumulation 3、BF16、dense 路由、effective_graphs 1008、分母 `update_level`） |
| — | **step 1、step 2 两个 optimizer update 均真实完成**（`steps.jsonl` 2 行；loss atom 5.193→5.163、geometry 0.252、balance 1.18e-4、`grad_total_preclip` 63.7/63.6，全部有限；`router_mode=dense`；新监控字段在真实记录中出现：`router_statistics='population_std_correction_0'`、`memory.cuda.window='step_since_last_reset'`、`memory.cpu.window='rank_process_lifetime'`） |
| 14:28:53 | `resume_00002.pt` 落盘完成，rank1–3 到达 `barrier` 并**阻塞**（stage log 最后一条：`barrier: enter`） |
| 14:29:04.27 | rank0：`deployment_package: enter` → `IndexError` → `failure: enter` → `cleanup: enter` → `cleanup: source_closed`（14:29:04.42）后**无** `process_group_destroyed`；rank0 阻塞在进程组销毁 |
| 14:31:0x | 进程内看门狗（120 s 无进展）输出四份**真实 Python 栈**（见「六」） |
| 14:32:5x | 看门狗的 180 s 自动停止**未生效**（原因见「七.2」），由操作者调用模块内既有 `stop_tree` 逻辑定向终止本次进程树：54 个后代全部 SIGTERM、**0 个需要 SIGKILL**、无幸存进程、GPU 显存释放；记录写入 `results/mcl_ph_20260921/p1/pretrain_r6/stall_supervisor_manual_stop.json` |
| 14:33:04 | launcher 退出码 **143**（SIGTERM，来自窗口日志 `R6_CAT_LAUNCHER_EXIT=143 SUPERVISOR_EXIT=0`）；supervisor 报告 `status='ROOT_EXITED'`、`intervened=false`、`silent_seconds=90.0` |

产物（`results/mcl_ph_20260921/p1/pretrain_r6/cat/`）：

- `resume_00002.pt` 315,570,711 B，sha256 `e91bda10bfb0db5a6da5f0a05c6bd0c22458ef48bb8a1053c2ef9e13f8818af2` —— 与 r3b 的 `resume_00002.pt` **逐字节相同**（`cmp` 通过），即两轮两步计算完全一致。
- **无 `deploy_00002.pt`**；无 `PASS`、无 `ALL_ARMS_OK`；`runtime.json` = `FAILED`/`exit_code=1`/`phase='before_cleanup'`；`failure_rank0.log`、`runtime_failure_rank0.json` 各一份（rank1–3 未抛异常，故无失败记录）。
- 阶段日志 `stages_rank{0..3}.log`、停滞栈 `stall_stack_rank{0..3}.txt`（各 3.1–3.3 KB）。

**六、根因：r3 导出挂起本轮被复现并定位**

1. **触发**：`shared` 变量遮蔽。`scripts/pretrain_mcl_ph.py:592`（denominator 循环内，`91626cb` r2 引入）把 `shared` 从 `apply_shared_init()` 的 dict（第 466 行）重绑为 `global_sum` 返回的 1-D tensor；导出窗口的 `source={'shared_new_init_sha256': shared['sha256']}`（`3d66198` 引入，现第 692 行）于是对 1-D tensor 做字符串索引 → `IndexError: too many indices for tensor of dimension 1`。**无模型最小复现**（CPU、1 元素张量）：`torch.tensor([1008.0])['sha256']` → 同一 UserWarning + 同一 IndexError。
2. **为什么表现为静默挂起且无 traceback**：rank0 抛错后进入 `finally` → `destroy_process_group()`，而 rank1–3 已在 `dist.barrier()` 等待 rank0 永不抵达的 barrier，形成死锁；异常被 `finally` 拖住，stderr 的 traceback 永不打印。看门狗 120 s 输出的真实栈：
   - rank0：`torch/distributed/distributed_c10d.py:2186 destroy_process_group` ← `scripts/pretrain_mcl_ph.py:727 main` ← `:774 <module>`
   - rank1–3：`torch/distributed/distributed_c10d.py:4811 barrier` ← `scripts/pretrain_mcl_ph.py:700 main` ← `:774 <module>`
3. **与 r3 的关系（不混为同一故障，但同源）**：r3b 的 launcher 日志在**同一表达式**（当时的 `pretrain_mcl_ph.py:535`）输出了**同一个 UserWarning**，随后静默 5.4 分钟直至人工 SIGTERM；两轮 `resume_00002.pt` 逐字节相同。故 r3 phase-2 的导出挂起**由本轮复现并定位到该缺陷**（此前状态为「未定位」）。r3 phase-1（10:53 那次，`exitcode: -15`、`balance≈96`）是另一次失败，与本缺陷无关，不并入。
4. **分类**：本次失败**不是**预备守卫失败，**不是**训练错误，而是**导出窗口缺陷 + 收尾死锁**；训练部分（两个 update）在完整条件下完成且与 r3 逐字节一致。

**七、本轮新发现的两个待修缺陷（按指令本轮不修改代码，交回 Codex）**

1. **`shared` 遮蔽导致导出必失败**（第 592 行重绑 / 第 692 行使用）：任何跑到 deploy 的 cat/gate/xattn 两步运行都会在导出元数据处抛 `IndexError`，随后进入收尾死锁。修法方向：把循环内 `shared` 改名为 `denominator_total`（或把初始化 dict 改名为 `shared_init`），保持数值与文件语义不变。
2. **看门狗的 180 s 自动停止不可达**：进程内看门狗每 120 s（`repeat=True`）向 `stall_stack_rank*.txt` 追加栈，这些写入被 `signature()` 计入 progress，静默时间永远不会超过 120 s（实测手工停止时 `silent_seconds=90.0`，`intervened=false`，报告文件只有在 root 退出后才生成）。修法方向：`signature()` 排除 `stall_stack_*`（只认 `stages_rank*` 的 mark）或改为只读最后一条 mark 的时间；r5 的 fixture 用静态目录，未能暴露该交互，需补一条「栈转储不得重置停滞计时」的 fixture。

**八、验收结论**

- **不满足「成功」**：launcher 未以 0 退出（143，操作者定向终止），`verify_mcl_ph_arm.py` 未进入；无部署包。故 §三 的只读验收**不适用**；仅存在训练侧记录（见「五」），**不宣称本轮 smoke 成功**。
- 训练侧事实（仅作记录）：2/2 updates、分母 1008/1008、loss/梯度/Router 统计有限、`resume` 与 r3 逐字节一致。
- 结论表述（按 §四 要求）：本次**复现了导出/收尾挂起**，且**定位到具体缺陷**；但该缺陷的修复与「挂起是否永久消失」需要下一轮在修好后重新验证。

**九、预算核算**

| 项目 | 上限 | 实际 |
| --- | --- | --- |
| 训练启动 | ≤1 | **1**（唯一一次，用满） |
| CAT optimizer updates | ≤2 | **2**（两个 update 均真实完成） |
| 额外模型 forward/backward | 0 | **0**（未运行任何预测/训练前向或反向） |
| 微调 epochs | 0 | 0 |
| 重跑测试套件 | 0 | 0 |
| GATE/XATTN/GLT_REF/O8_ONLY/P2/P3/5k/outer-test/缓存构建 | 0 | 0 |
| 代码修改 | 仅文档 | 仅 `MCL-PH.md`（本轮未改代码） |
| 外部总时限 | ≤15 min | 未触发（运行 ≈3.5 min，其中收尾挂起 ≈3.8 min 由定向终止结束） |

**十、交付状态**

- 状态：**受阻（执行后交回）**——唯一一次运行完成 2 个 update 后在导出窗口失败并挂起，根因定位；按指令未重试、未改代码、未启动其他臂。
- 产物与证据：`results/mcl_ph_20260921/p1/pretrain_r6/`（`cat/` 阶段日志、栈、FAILED 记录、resume；`stall_supervisor.json`、`stall_supervisor_manual_stop.json`、`shared_new_init.pt`）；日志 `logs/mcl_ph_20260921/r6_pretrain_cat.log`、`logs/mcl_ph_20260921/r6_tmux_window.log`。
- 未改动：模型数学/初始化/数据语义/超参/batch/worker 数；r3 现场（`pretrain_r3b/cat/runtime.json` 仍 `RUNNING`）；r4 恢复导出候选；`.zcodeignore`（`7fc7779` 由用户提交，本轮未碰）。
- 未取得：部署包与 §三 的 CPU/strict-load 核验；gate/xattn 与微调线索；任何性能结论。
- 提交与同步：本轮改动（仅 `MCL-PH.md`）提交为 `f7f42f2`；首次 `git push` 因 `gnutls_handshake() failed: The TLS connection was non-properly terminated.` 失败（本地提交已完成），随即重试成功 `7fc7779..f7f42f2 dev -> dev`（非 force），`git ls-remote origin dev` = `f7f42f279a794e2b3adab4e86863004e8bbec84d`，与本地 HEAD 一致。运行产物位于 `results/`、`logs/`（按 `.gitignore` 不提交，只按路径引用）；`.zcodeignore` 未修改。

---

### 13.7 r7A 执行记录（ZCode 执行；2026-09-21 UTC；基准 `dev@8108dc3`）

**计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r7A「已定位导出故障与 stall supervisor 最小返修」** |
| 状态 | **r7A 工程返修完成，待 Codex 审查**（本轮无任何真实训练；不代表 CAT 两步成功，也不代表 P1 完成） |
| 授权来源 | 用户 2026-09-21 的 r7A 指令：只修两个已定位缺陷 + 无模型针对性回归；**禁止启动 CAT/GATE/XATTN/微调，禁止模型 forward/backward** |
| 角色 | Codex/ChatGPT 规划与审查；ZCode 执行 |
| 开始时 HEAD / pull | `dev@8108dc3`（= 本轮审查基线）；`git pull --ff-only origin dev` → **Already up to date**；`git ls-remote origin dev` = `8108dc36d9bc8ed40fbf65d803b073774c406685`，远端无更新提交，无冲突实现，未触发停止条件 |

**一、修改文件（仅 3 个）**

- `scripts/pretrain_mcl_ph.py` —— 修复 A（`shared` 遮蔽）。
- `tests/_mcl_ph_r5_stall_supervisor.py` —— 修复 B（停滞计时把假进展当真实进展）。
- `tests/test_mcl_ph_r7a_repair.py` —— 新增，无模型回归（8 项）。

**二、修复 A：初始化身份的遮蔽**

- 身份变量改名：`shared = apply_shared_init(...)` → **`shared_init`**，并同步全部引用点——训练启动 `run.json` 的 `identity.shared_new_init_sha256`、`step_0000.json` 的 `shared_new_initialization`、deploy `source.shared_new_init_sha256`，共 3 处，全部读取同一个 `shared_init`；导出窗口不再出现 `shared[...]`。
- denominator 归约抽成具名纯函数 `reduce_update_denominators(update_counts, global_sum, device)`（原循环逐行搬移），训练入口改为一行调用；循环内临时量改名 `denominator_total`。**本文件中不再存在名为 `shared` 的变量绑定**（含 helper 内部，AST 断言）。
- **未改**：`effective_graph_counts` 的语义与取值、global reduction 的调用与输入张量、denominator 数值、objective（`L_atom + L_geo + 1e-3·L_bal`）、accumulation 缩放、loss 权重、DDP 行为、checkpoint identity、deployment metadata 字段名。
- **未采用**任何掩盖式修法：没有重新打开 `shared_new_init.pt`、没有重算 SHA、没有在 deploy 时新加 fallback。
- 训练入口的其余部分未顺手重构。

**三、修复 B：supervisor 的进度定义**

- 进度判定由 `signature()`（stages 日志 + `stall_stack_rank*.txt` + 普通 log 的 size/mtime）改为 `progress_marker()` + `_stage_log_marker()`：**只读 `stages_rank*.log`**，且只认**真实追加**（同时记录字节长度与最后一行内容，故单纯的 `mtime` 刷新不算）。
- 明确不计入进度：`stall_stack_rank*.txt` 增长、普通 runtime 日志增长、supervisor 自身输出与报告、对任意文件的 `os.utime`。
- `--log` 仍保留在 CLI 与报告中（取证用途），但**不再参与计时**；报告字段由 `signature_at_stop` 改为 `progress_at_stop`。
- `StageLogger.tensor_progress()` 的写入本就落在 `stages_rank*.log`，因此**真实导出进度仍被识别**（Test C 验证）。
- **未改**：默认阈值 120 s / 180 s、`EXIT_*` 返回码、`stop_tree`/`alive`/`descendants`/`stage_tail` 交互面、`StageLogger` 语义；**未新增通用 watchdog 框架**。

**四、实际运行的精确命令与结果（无模型回归）**

```
python -m pytest tests/test_mcl_ph_r7a_repair.py -q                   # 8 passed, 10.24s → exit 0
python -m pytest tests/test_mcl_ph_r5_completion.py -q -k supervisor  # 2 passed, 12 deselected, 10.95s → exit 0
```

日志：`logs/mcl_ph_20260921/r7a_regression_final.log`（首次尝试 `r7a_regression.log`；重跑 `r7a_regression_retry.log`）。

| 测试 | 覆盖内容 | 结果 |
| --- | --- | --- |
| Test A-1 `test_update_denominators_are_the_globally_summed_effective_graph_counts` | 走生产 helper：world=4 的假 collective 下得 `{'atom': 1008.0, 'geometry': 1008.0}`，collective 收到的正是本 rank 的 252；不等计数（1008/1004）第二组同样正确 | PASS |
| Test A-2 `test_the_initialization_identity_cannot_be_shadowed_again` | AST：模块内不存在名为 `shared` 的绑定；`shared_init` 恰好绑定一次且来自 `apply_shared_init(...)` | PASS |
| Test A-3 `test_the_deploy_metadata_reads_that_same_identity` | AST：deploy `source` 的 `shared_new_init_sha256` 读的是 `shared_init['sha256']`（同一身份变量） | PASS |
| Test A-4 `test_the_denominator_reduction_touches_no_identity` | helper 内部绑定集合恰为 `{update_counts, global_sum, device, denominators, name, value, denominator_total}`，不读写身份名；且 `main()` 确实调用该 helper | PASS |
| Test A-5 `test_progress_marker_counts_only_appended_stage_marks` | 追加 `stall_stack_rank0.txt` 或 `os.utime` 均不改变 marker，真实追加 `stages_rank*.log` 才改变 | PASS |
| Test B `test_growing_stack_dumps_and_logs_cannot_postpone_the_stop` | 隔离 fixture：stages 日志不再产生真实 mark，同时线程持续（数十次）向 `stall_stack_rank*.txt` 与普通 log 追加；supervisor 仍按缩短阈值介入（`stop_seconds=0.8`，`silent_seconds_at_stop < 1.5`），退出码 9，`intervened=true`，`progress_at_stop` 仍等于停止前的真实 mark 长度；**旁观进程未被误伤** | PASS |
| Test C `test_a_real_stage_mark_resets_the_silence_timer` | 以 0.4 s 间隔写入 5 条真实 stage mark（阈值 0.8 s）：最后一条 mark 之后进程仍存活（`alive_after_last_mark=[True]`），停止发生在最后一条 mark 之后一个阈值（`silent_seconds_at_stop ≥ stop_seconds−0.05`），无提前终止 | PASS |
| Test D `test_a_normal_exit_is_not_reported_as_a_stall` | 子进程自行退出（rc 0）→ supervisor 退出码 0、`status='ROOT_EXITED'`、`intervened=false`、报告无 `tree` 键、stdout 无 `STOPPED_BY_SUPERVISOR`、子进程自身 returncode 0 | PASS |
| 既有契约回归 | r5 的两条 supervisor 测试（静态 stages 日志下按阈值停止；正常运行不动手）在新进度定义下仍通过 | PASS |

**失败与修复（如实记录）**：首轮 7 passed / 1 failed——Test B 的 `assert bystander.poll() is None` 得到 `-15`，原因是**我的夹具**在 `finally` 中先终止了旁观进程，才执行该断言；把存活判定移到 `run_supervisor` 返回后、清理之前后通过。这是**测试夹具缺陷**，不是产品缺陷；产品代码的「只终止本次进程树」行为未改。

**五、预算核算（无模型回归）**

| 项目 | 上限 | 实际 |
| --- | --- | --- |
| 训练启动（CAT / GATE / XATTN / 任何 arm） | 0 | **0** |
| 模型 forward / backward | 0 / 0 | **0 / 0** |
| optimizer update | 0 | **0** |
| 微调 epoch | 0 | **0** |
| GPU | 0 | **0**（`nvidia-smi --query-compute-apps` 为空；fixture 全为 CPU 子进程） |
| 全仓 pytest 套件 | 0 | **0**（仅上述两个文件/选择器） |
| 回归墙钟 | ≤5 min | **≈42 s**（首轮 10.40 s + 10.85 s；最终确认 10.24 s + 10.95 s） |
| 修改范围 | 仅必要代码 | 2 个源文件 + 1 个新测试文件 + 本执行记录 |
| 遗留进程 | 0 | **0**（`pgrep` 无残留） |

**六、明确未声称 / 未验证**

- 本轮**未运行 CAT**，因此**不声称**「导出故障已在真实训练中修复」、**不声称**「CAT 两步 smoke 成功」、**不声称**「所有 DDP 异常退出已修复」。`rank0` 在 collective 中抛异常后 `destroy_process_group()` 与其余 rank 的 `barrier` 仍可能互相等待——该**收尾死锁本轮未处理**，`destroy_process_group`/barrier 异常协议未改；外部 supervisor 只保证「真实阶段停滞超过阈值时定向终止本次进程树」。
- 修复 A 的证据是**静态契约 + 纯函数/身份变量回归**；它在真实两步运行中能否写出 `deploy_00002.pt` 并通过 launcher 验收，**仍未验证**（需下一轮单独授权）。
- 修复 B 的证据是**缩短阈值的隔离 fixture**；120 s/180 s 默认值**在真实运行中未再验证**。
- P1 **仍未完成**（cat 仍停在 2/6 update、无 deploy 包；M_GATE/M_XATTN 与三条微调未运行）；**无任何 MCL-PH 性能、收敛或泛化结论**。
- 未改动：模型数学/Router/PH 与拓扑定义/Randić-Wiener-Betti/fusion/loss/balance/shared-init schema/optimizer/scheduler/config/dataset/collate/DDP objective/launcher 训练预算/P2/P3/outer-test/checkpoint 与历史结果。
- 未读取或改写 r3–r6 运行产物作为「修复」材料；r3 现场保持原状（`pretrain_r3b/cat/runtime.json` 仍 `RUNNING`）；未恢复根目录 `Plan.md`；未修改 `.zcodeignore`、`PH.md`、`3D.md`。
- 结论层级：本记录属于**实现 + 无模型测试**，不是 smoke、不是消融、不是正式实验。

**七、提交与同步**：本轮 4 个文件（`scripts/pretrain_mcl_ph.py`、`tests/_mcl_ph_r5_stall_supervisor.py`、`tests/test_mcl_ph_r7a_repair.py`、`MCL-PH.md`）创建**单独 commit `8b44361`**，非 force push 成功 `8108dc3..8b44361 dev -> dev`；`git ls-remote origin dev` = `8b443610c7a459d09c60e4b404d002c22e2af189`，与本地 HEAD 一致。运行产物位于 `results/`、`logs/`（按 `.gitignore` 不提交，只按路径引用）；未提交 checkpoint、缓存或临时目录。下一步（不在本轮执行）：在用户授权下做修好后的真实 CAT 两步运行，以验证导出写出与 launcher 验收，并在真实运行中确认修正后的定向停止生效。

---

### 13.8 r7B 执行记录（ZCode 执行；2026-09-21 UTC；基准 `dev@bd9fd63`）

**计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r7B「r7A 修复后的真实 CAT 两步生产闭环验证」** |
| 状态 | **r7B CAT 两步生产闭环通过（launcher/verifier/runtime/supervisor 全 PASS，resume↔deploy 170/170 一致）；§8 的 `deployment_check.py` 首次运行暴露其自身名称约定缺陷（非本轮引入），该检查未取得 PASS；P1 仍未完成**（不等于 `P1 PASS`） |
| 授权来源 | 用户 2026-09-21 的 r7B 指令：**最多 1 次 CAT 启动、最多 2 个 optimizer updates**，不启动其它 arm、不做性能实验 |
| 角色 | Codex/ChatGPT 规划与审查；ZCode 执行 |
| 开始时 HEAD / pull | `dev@bd9fd63`（= 审查基线）；`git pull --ff-only origin dev` → **Already up to date**，`git ls-remote origin dev` = `bd9fd633bfef433bbdf389f01b5b116410bf8505`；`8b44361` 经 `merge-base --is-ancestor` 确认为 HEAD 祖先；`shared_init`/`reduce_update_denominators`/supervisor 进度判定经 grep 与 r7A 回归（8 passed）复核，远端未改动相关代码，未触发停止条件 |

**一、唯一一次启动**

| 项目 | 数值 |
| --- | --- |
| tmux | session `Uni-Poly`，新 window `mcl_ph_r7b_cat`（window 78） |
| 驱动脚本 | `/tmp/mcl_ph_r7b_driver.sh`（非仓库文件） |
| launcher PID | `2271111`（`timeout -k 30 900 bash scripts/run_mcl_ph_pretrain_smoke.sh`） |
| 命令 | `ARMS='cat' UPDATES=2 NPROC=4 PREP_WORKERS=12 OUTPUT=results/mcl_ph_20260921/p1/pretrain_r7b LOG=logs/mcl_ph_20260921/r7b_pretrain_cat.log MCL_PH_STALL_SECONDS=120 timeout -k 30 900 bash scripts/run_mcl_ph_pretrain_smoke.sh &`，随后 `python tests/_mcl_ph_r5_stall_supervisor.py --root-pid 2271111 --stages-dir …/pretrain_r7b/cat --log …/r7b_pretrain_cat.log --report …/pretrain_r7b/stall_supervisor.json` |
| 时间 | 启动 15:09:09Z → launcher 退出 15:09:51Z，window 汇总 15:09:54Z；**墙钟 ≈45 s**（外部时限 ≤15 min，未触发） |
| 启动次数 | **1 / 1（用满，无重试）** |
| 输出目录 | `results/mcl_ph_20260921/p1/pretrain_r7b/`（父目录预先创建，`cat/` 由生产 runner 自建；启动前 `test ! -e` 与 `test ! -L` 均通过） |

**二、训练：2/2 updates 真实完成**（`cat/steps.jsonl`，rank0 记录）

| step | lr | loss atom | loss geometry | loss balance | grad_total_preclip | update_denominators | router_mode | finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0e-07 | 5.193426 | 0.251861 | 1.182954e-04 | 63.7255 | `{atom: 1008.0, geometry: 1008.0}` | dense | 全部有限 |
| 2 | 2.0e-07 | 5.162579 | 0.252755 | 1.176198e-04 | 63.6309 | `{atom: 1008.0, geometry: 1008.0}` | dense | 全部有限 |

两步均 < `router_dense_updates=500`，故训练用 dense 路由（与部署包固定的 Top-2 **推理**模式不是同一件事，不混写）。

**三、resume / deploy 产物**

| 文件 | 大小 (B) | SHA256 |
| --- | --- | --- |
| `cat/resume_00002.pt` | 315,570,711 | `e91bda10bfb0db5a6da5f0a05c6bd0c22458ef48bb8a1053c2ef9e13f8818af2` |
| `cat/deploy_00002.pt` | 81,401,259 | `c0f20f0643f2a08ec9a168f459ccb093d6a110c931d53ef4b993fcbab15e53cc` |

resume 字段实测：`step = 2`、`next_position = 2016`、`scheduler.step = 2`、`identity` 与本次 `run.json identity` **逐键相等**。deploy 由**本轮生产运行**生成，未使用 r4 recovery candidate。

**四、闭环阶段证据**（`cat/stages_rank*.log`，rank0 共 360 条 mark）

rank0：`loop:enter → step:enter(1) → barrier:enter(1) → step:complete(1) → step:enter(2) → rng_gather(2) → resume_save(2) → deployment_package:enter(2) → [95 个 tensor 的 start/complete] → deployment_package:complete → deploy_save:enter → deploy_save:complete → barrier:enter(2) → step:complete(2) → loop:complete → cleanup:enter → cleanup:source_closed → cleanup:process_group_destroyed → cleanup:complete`。
rank1–3：无 deploy 段（导出仅 rank0），均经过 `barrier:enter(2) → step:complete(2) → loop:complete → cleanup:* → complete`。**没有任何 rank 停在 barrier 或 destroy_process_group。**
`stall_stack_rank*.txt` 四个文件均为 **0 字节**：进程内 120 s 看门狗从未触发。

**五、runtime / launcher / verifier / supervisor**

- `cat/runtime.json`：`status='PASS'`、`completed_steps=2`、`cleanup='complete'`、`main_returned=true`、`export_complete=true`；memory 窗口为 `step_since_last_reset`（CUDA）/`rank_process_lifetime`（CPU）。
- launcher：`ARM=cat EXIT=0`（15:09:51Z）→ `verify_mcl_ph_arm.py --strict-cleanup` 输出 `{"verdict": "PASS", "problems": [], "strict_cleanup": true, "evidence": {"cleanup": "complete", "completed_steps": 2, "deploy_00002.pt": 81401259, "main_returned": true, "resume_00002.pt": 315570711, "status": "PASS"}}` → `ARM=cat VERIFY=0` → `=== RUNNER DONE ALL_ARMS_OK ===`。
- window 汇总：`R7B_CAT_LAUNCHER_EXIT=0 SUPERVISOR_EXIT=0`；supervisor 报告 `status='ROOT_EXITED'`、`intervened=false`、`silent_seconds=5.0`、`stack_seconds=120`、`stop_seconds=180`。无 `STOPPED_BY_SUPERVISOR`。
- 日志：`logs/mcl_ph_20260921/r7b_pretrain_cat.log`、`logs/mcl_ph_20260921/r7b_tmux_window.log`。

**六、v2 初始化身份核验（§7）——通过**

`pretrain_r7b/shared_new_init.pt`：schema `mcl-ph-shared-new-init-v2`、95 张量、4278237 B、SHA256 `499309392d578daf7a7baf1d102d7a7f5c652e5a9b42ca2904ac9d83d1fab85a`（与历史期望值一致，且与 r3b/r6 副本 `cmp` 逐字节相同）。三处引用实测全部等于该 SHA：`run.json identity.shared_new_init_sha256`、`step_0000.json shared_new_initialization.sha256`、`deploy source.shared_new_init_sha256`。

**七、§8 CPU 只读 deployment 验证——FAILED（单一条款），根因已只读定位，非本轮引入**

命令：`python tests/_mcl_ph_r3_deployment_check.py --pretrain-root results/mcl_ph_20260921/p1/pretrain_r7b --step 2 --arms cat --output results/mcl_ph_20260921/p1/pretrain_r7b/deployment_check.json` → 退出码 4，`status = FAILED`，`problems = ["cat: no shared tensor differs from the initial snapshot, so no update happened"]`。

通过的条款：`strict_load=true`、`architecture=MCL-PH-O8-MultiscaleDistanceRouter`、`training_route=mcl_ph`、`fusion_mode=cat`、`step=2`、`inference_mode=top2`、`inference_top_k=2`、`dense_updates=500`、`tensor_count=170`、`parameter_count=20651691`、`shared_new_init_sha256` 与文件一致。未通过的只有「update 是否真的发生」这一条，其判据是 `changed_shared_tensors == 0`，而根因是**名称约定不匹配**：

- `shared_new_state(model)` 取的是**完整 pretrainer** 的名字（前缀 `encoder.branch.`、`atom_head.`、`local_decoder.`、`nonbond_decoder.`）；`deployment_package(encoder, …)` 序列化的是 **encoder 自身** 的 `state_dict()`（`o8.*` / `branch.*` / `fusion.*`）。
- 检查器直接做 `name in package['state_dict']`，两类名字永不相等（实测直接交集 = 0，加 `encoder.` 前缀仍是 0），因此 `shared_tensors = 0` → 无论训练是否发生都会报「no update happened」。
- 该脚本自 r3 起未修改（`git log` 仅 `89789cb`），r7A（`8b44361`）只改了 4 个文件、未触碰它或 `src/modules/mcl_ph.py`；且 §13.3 记录（`MCL-PH.md:675`）明确写着该脚本「已写好并通过 `py_compile`，但**未运行**：不存在任何 `deploy_*.pt`」——r7B 是它的**首次执行**，不是回归。

**等价事实的独立只读证据**（零 forward）：把初始化快照去掉 `encoder.` 前缀后与部署包比较，79/79 个共有张量**全部**不同于初始值，最大绝对差 `3.0174851417541504e-07`（与 lr 1e-7/2e-7 两步一致）；训练侧 loss 由 5.193426 降到 5.162579。故「没有发生更新」的结论**不成立**。

按本轮规则：**不修代码**，该检查器缺陷连同上面证据交回 ChatGPT 决定修法（修法方向：比较前任一侧统一去掉 `encoder.` 前缀，并排除 16 个不属于 encoder 的预训练头张量）。

**八、§9 resume ↔ deploy encoder 逐张量一致性——通过**

纯 CPU、只读、零 forward：`resume['model']['encoder.<name>']` ↔ `deploy['state_dict']['<name>']`。

| 指标 | 结果 |
| --- | --- |
| encoder 张量数（resume / deploy） | 170 / 170 |
| 缺失 / 多余 | 0 / 0 |
| shape 或 dtype 不一致 | 0 |
| 数值不一致 | 0 |
| `torch.equal == True` | **170 / 170** |
| mismatch count | **0** |

**九、§10 resume SHA 复现诊断**：本轮 `resume_00002.pt` 与 r6（以及 r3b）的 `resume_00002.pt` **SHA256 完全相同**（`e91bda10…`），在 r7A 只做数学等价改造的前提下，这是「改造未改变数值」的额外复现证据；未据此重跑任何训练。

**十、预算核算**

| 项目 | 上限 | 实际 |
| --- | --- | --- |
| CAT 训练启动 | ≤1 | **1（用满，无重试）** |
| optimizer updates | ≤2 | **2（两步均真实完成）** |
| 额外模型 forward / backward | 0 / 0 | **0 / 0**（验收仅 `torch.load` + CPU `load_deployment` 严格加载，无前向/反向） |
| 微调 epoch | 0 | **0** |
| GATE / XATTN / GLT_REF / O8_ONLY / P2 / P3 / outer-test / 新缓存 | 0 | **0** |
| 其它 arm | 0 | **0**（`ARMS='cat'`） |
| 外部总时限 | ≤15 min | **≈45 s** |
| 修改范围 | 只允许文档 | **仅 `MCL-PH.md`**（本轮未改任何代码） |
| 遗留进程 / GPU | 0 | **0**（无残留进程；`nvidia-smi --query-compute-apps` 为空） |

**十一、未执行 / 未声称**

- **未**运行 GATE / XATTN / GLT_REF / O8_ONLY / 微调 / P2 / P3 / outer-test / 5k；未增加 update；未新建缓存；未启动其它 arm。
- **不声称** `P1 PASS`（本轮只得到「r7A 修复后的 CAT 两步真实 smoke 生产闭环通过」）；五臂验收、三条微调、M_GATE/M_XATTN 仍未完成。
- **不声称**性能、收敛或泛化结论：本记录属**真实两步 smoke**，不是消融或正式实验；两步 loss 变化不能作为性能证据。
- §8 的 `deployment_check.py` **未取得 PASS**（原因见「七」，已定位为检查器自身缺陷）；该缺陷**本轮未修**。
- 未改写 r3–r7A 历史记录；未恢复根目录 `Plan.md`；未修改 `.zcodeignore`、`PH.md`、`3D.md`；r3 现场保持原状。

**十二、提交与同步**：本轮只改 `MCL-PH.md`，提交为 **`2d8d589`**，非 force push 成功 `bd9fd63..2d8d589 dev -> dev`；`git ls-remote origin dev` = `2d8d589b743f50d696416fea23b22bf48b9ffd7d`，与本地 HEAD 一致。运行产物位于 `results/mcl_ph_20260921/p1/pretrain_r7b/` 与 `logs/mcl_ph_20260921/`（按 `.gitignore` 不提交，只按路径引用）；未提交 checkpoint、缓存或临时目录。**未自动启动 GATE/XATTN 或微调**；交回 ChatGPT/Codex 审查。

---

### 13.9 r7C 执行记录（ZCode 执行；2026-09-21 UTC；基准 `dev@8c1454f`）

**计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r7C「修复 deployment checker 的 shared-init 命名空间错误并用现有 r7B 产物复验」** |
| 状态 | **r7C deployment verifier repaired；r7B 现有 CAT 产物通过 deployment 复验；P1 仍未完成**（不写 `P1 PASS`） |
| 授权来源 | 用户 2026-09-21 的 r7C 指令：只修 `tests/_mcl_ph_r3_deployment_check.py` 的命名空间错误 + 新增 model-free regression + 用现有产物复验；**禁止重新训练 CAT** |
| 角色 | Codex/ChatGPT 规划与审查；ZCode 执行 |
| 开始时 HEAD / pull | `dev@8c1454f`（= 审查基线）；`git pull --ff-only origin dev` → **Already up to date**；`git ls-remote origin dev` 同为 `8c1454f…`；该 checker 自 `89789cb`（r3）以来无人改动（`git log` 与 `git diff HEAD` 均确认），未触发停止条件 |

**一、verifier bug 的准确原因（r7B 已只读定位，本轮确认）**

- `shared_new_state()`（生产代码）以**完整 pretrainer** 的 `state_dict()` 命名，故 **shared-init artifact**（`shared_new_init.pt`，实测 **95** 张）里的名字带前缀：`encoder.branch.*` 79 张 + 16 个预训练头（`atom_head.*` 2、`local_decoder.*` 10、`nonbond_decoder.*` 4）。**`encoder.o8.*` 与 `encoder.fusion.*` 不属于该 artifact**（它们属于共享主干与融合部分，不在 shared-init 快照内）。
- `step_0000.json` 的 `parameters` 是**完整 pretrainer 参数快照**（实测 186 项 = 上列 95 项 + `encoder.o8.*` 83 + `encoder.fusion.*` 8）；§13.9 原先把这一步的 186 项写成了 shared-init artifact 的成员，r8 在下方更正，结论不变。
- `deployment_package(encoder, …)`（生产代码）序列化的是 `encoder.state_dict()`，名字为 encoder 相对形式：`o8.*` / `branch.*` / `fusion.*`，共 170。
- 旧 checker 直接做 `name in package['state_dict']`，两类名字永不相等，于是 `shared_tensors = 0` → 无论是否训练都会报「no update happened」。**这是 verifier 不理解 bundle，而不是 bundle 有错**（本轮未改 bundle）。
- 附带缺陷：`FUSION_PREFIX = 'fusion.'` 无法匹配 `step_0000.json` 中真实的 `encoder.fusion.*`，多臂同时检查时会把 fusion 参数当成 shared 公共参数。

**二、修复方式（只改 checker，最小改动）**

新增 4 个纯函数，替换 `main()` 中原地展开的比较与判据：

| 函数 | 作用 |
| --- | --- |
| `shared_encoder_state(shared_state)` | 只取 `encoder.` 前缀项并去掉该前缀，得到 encoder 相对命名空间；`atom_head.*` / `local_decoder.*` / `nonbond_decoder.*` 按构造排除（它们不属于部署 bundle） |
| `shared_encoder_deltas(shared_encoder, package_state)` | 返回逐张量 max-abs delta **与缺失键列表**；缺失键报错而不是静默取交集 |
| `shared_initial_names(common_names)` | 多臂共享初始化的公共名过滤（排除 `encoder.fusion.*`） |
| `arm_problems(arm, record, expected_sha)` | 单臂契约判据（Top-2、MCL-PH 路由、映射非空、缺失键、是否真的更新、init SHA），纯函数以便测试 |

其余改动：`FUSION_PREFIX` → `encoder.fusion.`；记录字段 `shared_tensors`／`changed_shared_tensors` 替换为语义准确的 `shared_encoder_tensors`／`changed_shared_encoder_tensors`，新增 `shared_encoder_missing_keys`，保留 `max_abs_delta_from_shared_init`（现已指 encoder 相对快照差值）；模块 docstring 第三条同步改写。旧字段名在整个仓库中**只有该 checker 使用**（`grep -rn` 确认，`_mcl_ph_r3_init_check.py` 的 `shared_tensors_applied_verbatim` 是无关字段），故更名不影响任何其它使用者。

**未改**：`src/modules/mcl_ph.py`、`deployment_package()`、`load_deployment()`、`scripts/pretrain_mcl_ph.py`、checkpoint / deploy schema、任何训练或数据路径。

**三、新增 model-free regression：`tests/test_mcl_ph_r7c_deployment_check.py`**

```
python -m pytest tests/test_mcl_ph_r7c_deployment_check.py -q   # 6 passed in 1.32s → exit 0
```

| 用例 | 内容 | 结果 |
| --- | --- | --- |
| Case 1 | 合成快照 `encoder.branch.a/b`、`encoder.fusion.expand.weight`、`atom_head.x`、`nonbond_decoder.y` 对 `branch.*`、`fusion.*`、`o8.y`：映射恰为 3 个 encoder 相对键，两个预训练头被排除，缺失键为空 | PASS |
| Case 2 | 全部 mapped 张量相同 → `changed_shared_encoder_tensors = 0`，checker 报「no update happened」 | PASS |
| Case 3 | 至少一个张量改变 → 计为更新、`max_abs_delta > 0`、无问题 | PASS |
| Case 4 | mapped 键 `branch.b` 在部署中缺失 → 明确报 key mismatch（列出缺失键），且**不因另一个张量已改变而被掩盖** | PASS |
| Case 5 | `FUSION_PREFIX == 'encoder.fusion.'`，`encoder.fusion.*` 不进入多臂 shared 公共参数比较 | PASS |
| 源码契约 | 旧字段名 `'shared_tensors'` / `'changed_shared_tensors'` 不得再现 | PASS |

测试构造过程如实记录：首轮 5 passed / 1 failed，失败原因是**我的合成张量改了 shape**（`branch.a` 由 2 元素变 3 元素）导致相减 `RuntimeError`——产品路径上 `load_deployment` 的严格加载已先行校验 shape，该情形不可达；把用例改为只改数值后通过。这是**夹具缺陷**，不是 checker 缺陷。

**四、用现有 r7B 产物复验（零训练、CPU only）**

```
python tests/_mcl_ph_r3_deployment_check.py \
  --pretrain-root results/mcl_ph_20260921/p1/pretrain_r7b --step 2 --arms cat \
  --output results/mcl_ph_20260921/p1/pretrain_r7b/deployment_check_r7c.json
# exit code 0
```

| 字段 | 值 |
| --- | --- |
| `status` | **PASS** |
| `problems` | **[]** |
| `strict_load` | true |
| `training_route` / `fusion_mode` / `step` | `mcl_ph` / `cat` / 2 |
| `inference_mode` / `inference_top_k` | `top2` / 2 |
| `shared_encoder_tensors` | **79** |
| `changed_shared_encoder_tensors` | **79** |
| `shared_encoder_missing_keys` | `[]` |
| `max_abs_delta_from_shared_init` | `3.0174851417541504e-07` |
| `shared_new_init_sha256` | `499309392d578daf7a7baf1d102d7a7f5c652e5a9b42ca2904ac9d83d1fab85a`（与 artifact 相同） |
| `tensor_count` / `parameter_count` | 170 / 20651691 |

79 = 快照中 `encoder.branch.*` 的张量数，与 r7B 记录一致；`max_abs_delta` 与 r7B 的独立只读比较结果相同。**未改 bundle、未重新导出、未重跑训练**：`deploy_00002.pt` 仍为 15:09:48、SHA256 `c0f20f06…` 与 r7B 记录一致；`resume_00002.pt`、`runtime.json` 均未触碰；本轮仅新增 `deployment_check_r7c.json`。

**五、预算核算**

| 项目 | 上限 | 实际 |
| --- | --- | --- |
| GPU | 0 | **0**（`nvidia-smi --query-compute-apps` 为空；checker 与测试全为 CPU，仅构造/加载张量） |
| 训练启动 / optimizer update | 0 / 0 | **0 / 0**（**未重跑 CAT**） |
| 模型 forward / backward | 0 / 0 | **0 / 0**（`load_deployment` 只做严格加载） |
| 微调 epoch | 0 | **0** |
| 其它 arm（GATE/XATTN/GLT_REF/O8_ONLY）/ P2 / P3 / outer-test | 0 | **0** |
| CPU regression 墙钟 | ≤5 min | **≈1.3 s**（回归）+ 首轮失败重跑 1.41 s + checker 数秒 |
| 修改范围 | 仅 checker、其 regression、文档 | `tests/_mcl_ph_r3_deployment_check.py`、`tests/test_mcl_ph_r7c_deployment_check.py`、`MCL-PH.md` |
| 未执行的重复验收 | —— | 未重复比较 170 个 resume/deploy 张量、未重算训练 loss、未重新生成 deploy |

**六、未执行 / 未声称**

- **未重跑 CAT**，未启动 GATE/XATTN/GLT_REF/O8_ONLY/微调/P2/P3/outer-test，未增加 update，未新建缓存。
- **不写 `P1 PASS`**：本轮只完成「deployment verifier 修复 + 现有 r7B 产物通过复验」，五臂验收、M_GATE/M_XATTN 与三条微调仍未完成。
- **无性能、收敛或泛化结论**；本记录属**实现 + 无模型测试 + 对既有产物的只读复验**，不是新的 smoke、消融或正式实验。
- 未改写 r3–r7B 历史记录；未恢复根目录 `Plan.md`；未修改 `.zcodeignore`、`PH.md`、`3D.md`。

**七、提交与同步**：本轮 3 个文件（`tests/_mcl_ph_r3_deployment_check.py`、`tests/test_mcl_ph_r7c_deployment_check.py`、`MCL-PH.md`）提交为 **`2a519ee`**，非 force push 成功 `8c1454f..2a519ee dev -> dev`；`git ls-remote origin dev` = `2a519ee605e303c2455b6f15b40278aa466cabd1`，与本地 HEAD 一致。产物与日志位于 `results/mcl_ph_20260921/p1/pretrain_r7b/`（`deployment_check_r7c.json`）与 `logs/mcl_ph_20260921/`（`r7c_regression.log`、`r7c_final.log`），按 `.gitignore` 不提交，只按路径引用。**未自动启动 GATE/XATTN 或微调**；交回 ChatGPT/Codex 审查。

---

### 13.10 r8 执行记录（ZCode 执行；2026-09-21 UTC；基准 `dev@a47a910`；**在 downstream 第一臂处停止**）

**计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r8「补齐 GATE/XATTN 两步 smoke + 五臂 downstream 一 epoch smoke + 固定 aggregator」** |
| 状态 | **部分完成并已停止：GATE/XATTN 当前代码生产 smoke 通过、三臂 deployment checker PASS；downstream 在第一个 unit（`glt_ref`）处因**既有代码缺陷**失败，按指令立即停止（不修代码、不重试、不继续后续臂）；**P1 未完成** |
| 授权来源 | 用户 2026-09-21 的 r8 指令：GATE/XATTN 各 ≤1 次启动、各 ≤2 updates；downstream 5 arms × 1 epoch，每 arm 最多启动一次，任一失败即停止；CAT 0；GLT_REF 0 |
| 角色 | Codex/ChatGPT 规划与审查；ZCode 执行 |
| 开始时 HEAD / pull | `dev@a47a910`（= 审查基线）；`git pull --ff-only origin dev` → **Already up to date**；远端同为 `a47a910…`；无其它执行者改动目标代码；未触发停止条件 |
| tmux windows | `mcl_ph_r8_gate`、`mcl_ph_r8_xattn`、`mcl_ph_r8_finetune`（session `Uni-Poly`） |

**一、GLT_REF 只读前置核验（§2）——通过**

`results/mcl_ph_20260921/p1/pretrain/glt_ref/deploy_00002.pt`（156,146,129 B，**未重训**）CPU-only、零 forward 检查：

| 项目 | 结果 |
| --- | --- |
| 文件存在 / `step` | 存在；`step = 2` |
| 历史配置 | `run.json` 的 `third_task = fp`，命令为 `scripts/pretrain_glt_dual.py --config configs/mts/glt_pred_s3b_b_fp.json --third-task fp`；package `architecture = O8-BondPath-GalformerTrimer-Hop2`、`fusion_mode = concat`、`torsion_modules = false`、187 张量（`o8.*` 83 张） |
| 当前 loader strict-load | `GLTReferenceArm` + `load_dual_deployment(..., 2)` → **通过** |
| `o8_only` 可用性 | `build_o8_only(...)` → **83/83 个 O8 张量完整复制**，无缺键/形状错误 |
| 与当前 downstream contract 冲突 | 无（package 未携带 `route`/`third_task` 字段，二者为 `None`；路线身份由 `run.json` 的 `third_task=fp` 与 architecture 佐证） |

**二、GATE 两步 smoke（§3）——通过**（window `mcl_ph_r8_gate`，15:26:14Z→15:26:59Z，墙钟 ≈45 s）

```bash
ARMS='gate' UPDATES=2 NPROC=4 PREP_WORKERS=12 \
OUTPUT=results/mcl_ph_20260921/p1/pretrain_r7b LOG=logs/mcl_ph_20260921/r8_pretrain_gate.log \
MCL_PH_STALL_SECONDS=120 timeout -k 30 900 bash scripts/run_mcl_ph_pretrain_smoke.sh
# + tests/_mcl_ph_r5_stall_supervisor.py --root-pid <launcher> --stages-dir .../gate --report .../stall_supervisor_gate.json
```

| 项目 | 结果 |
| --- | --- |
| launcher / verifier | `ARM=gate EXIT=0`；`VERIFY=0`（`verdict PASS`、`strict_cleanup`）；`ALL_ARMS_OK` |
| supervisor | `ROOT_EXITED`、`intervened=false`、exit 0；`stall_stack_rank*.txt` 全 0 字节 |
| runtime | `status=PASS`、`completed_steps=2`、`cleanup=complete`、`main_returned=true`、`export_complete=true` |
| step 1 / 2 | atom 5.045283 → 5.007334、geometry 0.251861 / 0.252755，loss 与梯度**全部有限**；`update_denominators = {atom: 1008.0, geometry: 1008.0}`；`router_mode=dense` |
| 产物 | `gate/resume_00002.pt`（311,039,279 B）、`gate/deploy_00002.pt`（79,890,673 B） |
| identity | `identity.shared_new_init_sha256 = 499309…fab85a`（与 artifact 相同） |

**三、XATTN 两步 smoke（§4）——通过**（window `mcl_ph_r8_xattn`，15:27:48Z→15:28:33Z，墙钟 ≈45 s）

同上合同；`ARM=xattn EXIT=0`、`VERIFY=0`、`ALL_ARMS_OK`；supervisor `ROOT_EXITED`/`intervened=false`；runtime `PASS/2/complete/true`；step 1/2 atom 5.015443 → 4.978117、geometry 0.251861 / 0.252755，有限；denominators 1008/1008；`router_mode=dense`；`stall_stack_rank*.txt` 全 0 字节。

**四、三臂 deployment 验收（§5）——通过**

```bash
python tests/_mcl_ph_r3_deployment_check.py --pretrain-root results/mcl_ph_20260921/p1/pretrain_r7b \
  --step 2 --arms cat gate xattn --output results/mcl_ph_20260921/p1/pretrain_r7b/deployment_check_r8.json
# exit 0
```

| 字段 | 结果 |
| --- | --- |
| `status` / `problems` | **PASS** / `[]` |
| 每臂（cat、gate、xattn） | `strict_load=true`、`step=2`、`training_route=mcl_ph`、`inference_mode=top2`、`inference_top_k=2`、`shared_encoder_tensors=79`、`changed_shared_encoder_tensors=79`、`shared_encoder_missing_keys=[]`、`max_abs_delta_from_shared_init=3.017e-07` |
| 联合初态 | `shared_initial_parameters=178`、**`initial_value_difference_count=0`**；三臂 `shared_new_init_sha256` 完全相同（`499309…fab85a`），schema 均为 `mcl-ph-shared-new-init-v2` |

**五、五臂 downstream smoke（§6）——在第一个 unit 处失败并停止**

新输出根 `results/mcl_ph_20260921/p1/finetune_r8`，五臂依序直接调用 `scripts/finetune_mcl_ph.py`（`--stage smoke --task xc --fold 0 --epochs 1 --expected-pretrain-step 2`，cohort/cache/static/split/statistics 用当前 P1 合同路径，config `configs/mts/mcl_ph_gate.json`，与历史四个 unit 相同）。**未复制、改名或伪造任何 checkpoint**：`glt_ref` 与 `o8_only` 用旧根的 `pretrain/glt_ref/deploy_00002.pt`，三个 MCL 臂各用 `pretrain_r7b/<arm>/deploy_00002.pt`。

| unit | 结果 |
| --- | --- |
| `glt_ref` | **训练本身成功**（9/9 steps、1 个 epoch、train_loss 1.0288、validation_loss 0.5686、R² 0.0127，均有限），但在**写 `run.json` 时崩溃**：`TypeError: dict() got multiple values for keyword argument 'optimizer_groups'`，退出码 **1** |
| `o8_only` / `m_cat` / `m_gate` / `m_xattn` | **未启动**（按「任一 unit 非零退出即 STOP」的指令，未凑表） |

失败 unit 现场（`finetune_r8/glt_ref/xc/fold0/`）：`runtime.json` = `status FAILED`、`exit_code=1`、`process_wall_seconds≈11.77`；**`metrics.json`、`best.pt`、`validation_predictions.npz` 均已写出**（metrics 内含 `optimizer_updates=9`、`best_epoch=1`、`outer_test=NOT_RUN`、split 完整/互斥/全覆盖）；**缺 `run.json`**。session 内无残留进程，GPU 已释放。

**六、根因（只读定位，本轮未修）**

`scripts/finetune_mcl_ph.py:415-418`：

```python
summary = dict(status='PASS', command=sys.argv, config=config,
               optimizer_groups=group_evidence, device=str(device), **{
                   key: value for key, value in common.items()
                   if key not in ('history', 'load_state_dict_result')})
```

而 `common = dict(...)`（第 383 行起）**已经**在 `optimizer_groups=group_evidence`（第 411 行）写入同一个键，`**{...}` 展开时与之重复 → `TypeError`。该行由 **`3d66198`（2026-09-21 11:04:19，P0 审计与 P1 实现）**引入（`git blame` 确认），目的是让 `run.json` 带上 `optimizer_groups`。

- **影响面**：`summary` 的构造位于 `run_unit` 的公共路径，**五个臂全部会崩**，与臂、任务、checkpoint 无关；失败发生在训练与评估**之后**，即模型训练本身是正常的。
- **为何 r1/r2 的四个旧 unit 能通过**：它们在 10:59–11:02 运行，**早于** `3d66198`（11:04:19）；`find results -name run.json -path "*finetune*" -newermt "2026-09-21 11:05"` 为空，即该缺陷自引入以来**从未被执行过**，r8 是首次暴露。
- **修法方向（交回，不在本轮执行）**：删除 `summary = dict(...)` 显式参数中重复的 `optimizer_groups=group_evidence`（`common` 已提供该键），或改由 `common` 统一提供；不改变任何科学定义或记录字段。

**七、预算核算**

| 项目 | 上限 | 实际 |
| --- | --- | --- |
| GATE 启动 / updates | ≤1 / ≤2 | **1 / 2**（用满） |
| XATTN 启动 / updates | ≤1 / ≤2 | **1 / 2**（用满） |
| 新增 MCL pretrain updates 合计 | ≤4 | **4** |
| CAT 启动 / updates | 0 / 0 | **0 / 0**（只读复用 r7B 产物） |
| GLT_REF 启动 | 0 | **0**（只做 CPU 只读 strict-load 核验） |
| downstream smoke | 5 arms × 1 epoch | **1 个 arm 启动、执行 1 epoch（9 updates）后失败**；其余 4 臂 **0** |
| P2 / P3 / outer-test / 5k / 新 seed / 新 cache | 0 | **0** |
| 修改范围 | 只允许文档 | **仅 `MCL-PH.md`**（本轮未改任何代码） |

**八、未执行 / 未声称**

- 未启动 `o8_only` / `m_cat` / `m_gate` / `m_xattn` 四个 downstream unit，未运行固定五臂 aggregator（`scripts/aggregate_mcl_ph.py` 未执行）。
- **不写 `P1 PASS`**，也不写 r8 的十项完成清单：`GATE`、`XATTN`、三臂 checker 已过，但五个 downstream unit、aggregate 未完成。
- **无任何性能或架构排名结论**：本轮不比较、不解释任何 R²；`m_cat`/`m_gate`/`m_xattn` 的 downstream 接口尚未验证。GLT_REF 的 R² 0.0127 只是**一次接口 smoke 的观测值**，不构成性能证据。
- 未访问 outer-test（各 unit `outer_test = NOT_RUN`）；未启动任何 5k 正式训练或 P2 内容。
- 未改写 r3–r7C 历史记录（仅按指令更正 §13.9 中 shared-init artifact 与 `step_0000.json` 完整快照的混写，结论不变）；未恢复 `Plan.md`；未修改 `.zcodeignore`、`PH.md`、`3D.md`。

**九、提交与同步**：本轮只改 `MCL-PH.md`，提交为 **`d07cd92`**，非 force push 成功 `a47a910..d07cd92 dev -> dev`；`git ls-remote origin dev` = `d07cd92c8a1b9c0b2fa1c3042f77ebd8d0d8aca3`，与本地 HEAD 一致。产物与日志：`results/mcl_ph_20260921/p1/pretrain_r7b/{gate,xattn,deployment_check_r8.json,stall_supervisor_gate.json,stall_supervisor_xattn.json}`、`results/mcl_ph_20260921/p1/finetune_r8/glt_ref/xc/fold0/`（失败现场）、`logs/mcl_ph_20260921/{r8_pretrain_gate.log,r8_pretrain_xattn.log,r8_finetune_smoke.log,r8_tmux_*.log}`，按 `.gitignore` 不提交，只按路径引用。**未自动启动其它 downstream 臂、aggregator、GATE/XATTN 微调或任何 5k 训练**；交回 ChatGPT/Codex 审查。

### 13.11 r9 执行记录（ZCode 执行；2026-09-21 UTC；基准 `dev@73bd4cc`；**五臂 downstream smoke 与 aggregate 全部通过**）

**计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r9「修复 `optimizer_groups` 重复关键字缺陷 + 全新根重跑五臂 downstream smoke + 固定五臂 aggregator」** |
| 状态 | **执行完成**：最小代码修复 + 无模型回归 4 passed；五个 XC/fold0/1-epoch unit 全部 `exit_code=0`、`status=PASS`；五臂 aggregate `status=PASS`、`acceptance=PASS`、5/5/0。**P1 执行证据完整，待 ChatGPT 审查**（未写 `P1 PASS`，也未写「正式验收通过」） |
| 授权来源 | 用户 2026-09-21 的 r9 指令：只修该缺陷（不得顺手改其它内容）、CPU-only 无模型回归、全新输出根、五臂各最多启动一次、任一失败即停止、**本轮不运行任何预训练** |
| 角色 | Codex/ChatGPT 规划与审查；ZCode 执行 |
| 开始时 HEAD / pull | `dev@73bd4cc`（= 审查基线）；`git pull --ff-only origin dev`；无其它执行者改动目标代码；未触发停止条件 |
| tmux window | `mcl_ph_r9_finetune`（session `Uni-Poly`，window 82）；15:36:51Z→15:39:13Z，墙钟 ≈142 s |

**一、缺陷与最小代码修复**

r8 已只读定位（§13.10 六）：`run_unit()` 的成功路径在构造 `summary` 时把 `optimizer_groups` **同时**作为显式关键字与 `common` 展开键传入，而 `common` 已含该键 → `TypeError: dict() got multiple values for keyword argument 'optimizer_groups'`。本轮**只**删除重复的显式关键字，并把它抽成同文件内 5–10 行纯函数：

```python
def build_summary(common, *, config, device, command):
    return dict(status='PASS', command=command, config=config, device=str(device), **{
        key: value for key, value in common.items()
        if key not in ('history', 'load_state_dict_result')})

# run_unit() 内（原 summary = dict(status='PASS', ..., optimizer_groups=group_evidence, ...)）
summary = build_summary(common, config=config, device=device, command=sys.argv)
write_json(folder / 'run.json', dict(summary, history=common['history']))
```

| 项目 | 结果 |
| --- | --- |
| diff 规模 | `scripts/finetune_mcl_ph.py \| 18 ++++++++++++++----`（**+14 / −4**：新增 13 行纯函数 + 1 行调用；删除 4 行旧构造）；无其它文件改动 |
| `run.json` 字段 | 不变：`optimizer_groups` 仍来自 `common`（实测非空，四组 `backbone`/`backbone_no_decay`/`head`/`head_no_decay`）；`history` 仍由 `dict(summary, history=common['history'])` 写入 |
| 未改动 | `common['optimizer_groups']`、optimizer group 内容、JSON 字段名、metrics/`best.pt` schema、训练循环/optimizer/scheduler/split/model、PH/Router/fusion 数学、aggregator 规则；未启动预训练 |

**二、无模型回归（§3）**

新增 `tests/test_mcl_ph_r9_summary.py`（4 个测试；纯 CPU、无模型、无 forward/backward）：A. `common` 已含 `optimizer_groups` 且 summary 可构造；B. `summary['optimizer_groups']` 与 `common['optimizer_groups']` **同一对象**且只出现一次；C. `summary` 本身不含 `history`/`load_state_dict_result`，而生产 payload `dict(summary, history=...)` 含 history；D. AST 断言 `build_summary` 内不再出现显式 `optimizer_groups` 关键字、`dict(x, **{...})` 复现 `TypeError: multiple values`、`run_unit()` 调用 `build_summary`。

```bash
python -m pytest tests/test_mcl_ph_r9_summary.py -q    # 4 passed, 1 warning in 5.19s; exit 0
```

日志 `logs/mcl_ph_20260921/r9_regression.log`。回归预算：GPU = 0、forward = 0、backward = 0、optimizer step = 0、训练启动 = 0；未运行全量 pytest。

**三、五个 downstream smoke unit（§6/§7/§8）——全部通过**

新输出根 `results/mcl_ph_20260921/p1/finetune_r9`（**启动前五个 `<arm>/xc/fold0` 均不存在**，由 runner 自建；未覆盖 r1/r8 旧目录）。五个 unit 在同一 tmux window 内**顺序**直接调用同一脚本，任一非零退出即中止；**每个 arm 只启动一次，无重试**：

```bash
python scripts/finetune_mcl_ph.py --arm <ARM> --stage smoke --config configs/mts/mcl_ph_gate.json \
  --expected-pretrain-step 2 \
  --cohort-root data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1 \
  --cache-root data/processed/mips_trimer_scage_downstream \
  --dual-static-root data/processed/glt_dual_v2/downstream/dual_static_v1 \
  --split-root data/splits/mips_outer5_inner20 \
  --statistics results/mcl_ph_20260921/p0/statistics.npz \
  --task xc --fold 0 --epochs 1 --output results/mcl_ph_20260921/p1/finetune_r9 \
  --checkpoint <每臂自己的 deploy_00002.pt>
```

| unit | checkpoint（未复制/改名） | exit | status | executed_epochs | optimizer_updates | best_epoch | pretrain_step | outer_test | run.json 墙钟 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `glt_ref` | `pretrain/glt_ref/deploy_00002.pt` | 0 | PASS | 1 | 9 | 1 | 2 | NOT_RUN | 11.19 s |
| `o8_only` | `pretrain/glt_ref/deploy_00002.pt`（同包，runner 抽 O8） | 0 | PASS | 1 | 9 | 1 | 2 | NOT_RUN | 11.21 s |
| `m_cat` | `pretrain_r7b/cat/deploy_00002.pt` | 0 | PASS | 1 | 9 | 1 | 2 | NOT_RUN | 30.81 s |
| `m_gate` | `pretrain_r7b/gate/deploy_00002.pt` | 0 | PASS | 1 | 9 | 1 | 2 | NOT_RUN | 31.24 s |
| `m_xattn` | `pretrain_r7b/xattn/deploy_00002.pt` | 0 | PASS | 1 | 9 | 1 | 2 | NOT_RUN | 30.73 s |

每个 unit 的五个必需产物（`run.json`、`runtime.json`、`metrics.json`、`best.pt`、`validation_predictions.npz`）**全部存在**；逐项核验通过：`stage=smoke`、`protocol=mcl_ph_smoke`、`requested_epochs=executed_epochs=1`、`optimizer_updates=9 > 0`、`history` 长度 1 且 `train_loss`/`validation_loss`/`validation_r2` 全部有限、`best_epoch=1`、`pretrain_step=2`、`optimizer_groups` 非空、`outer_test=NOT_RUN`、`split.protocol=outer5_inner20`、`validation_is_test=false`、`sets_disjoint=true`、`union_equals_full_cohort=true`（每臂 276 训练 / 69 验证 / 87 未触及的 outer-test 行）。`run.json` 的 `pretrain_package_sha256`：`glt_ref`/`o8_only` = `ae37744b…`、`m_cat` = `c0f20f06…`、`m_gate` = `2d91fce7…`、`m_xattn` = `c1590e44…`（身份互不相同，与各自 checkpoint 对应）。**未比较、未解释任何 R² 高低。**

**四、五臂 aggregate（§12）——通过**

```bash
python scripts/aggregate_mcl_ph.py --root results/mcl_ph_20260921/p1/finetune_r9 --stage smoke \
  --expected-pretrain-step 2 --arms glt_ref o8_only m_cat m_gate m_xattn --tasks xc --folds 0 \
  --output results/mcl_ph_20260921/p1/finetune_r9/aggregate.json    # exit 0
```

`status=PASS`、`acceptance=PASS`、`units_expected=5`、`units_accepted=5`、`units_rejected=0`、`accepted[]` 五臂、`rejected=[]`、`outer_test=NOT_RUN`；aggregator 规则未做任何修改。

**五、r8 失败现场保持原状（§5）**

`results/mcl_ph_20260921/p1/finetune_r8/glt_ref/xc/fold0/` 未被写入：仍为 `runtime.json` `status=FAILED`/`exit_code=1`/`process_wall_seconds≈11.77`，**无 `run.json`**，`metrics.json`/`best.pt`/`validation_predictions.npz` 保持 r8 当时状态（mtime 15:29Z 未变）；未从 `metrics.json` 手工复活成功 unit，其已消耗的 1 个 epoch 永久保留在账上。

**六、预算核算**

| 项目 | 上限 | 实际 |
| --- | --- | --- |
| 预训练启动 / pretrain updates | 0 / 0 | **0 / 0**（本轮不运行任何预训练） |
| downstream smoke | 5 arms × 1 epoch | **5 次启动、5 个 epoch**（每臂一次，无重试） |
| 修改范围 | 缺陷修复 + 回归 + 文档 | `scripts/finetune_mcl_ph.py`（+14/−4）、`tests/test_mcl_ph_r9_summary.py`（新增）、`MCL-PH.md` |
| P2 / P3 / 5k / EPS / EAT / fold1 / outer-test / 新 seed / 新 cache | 0 | **0** |

**七、未执行 / 未声称**

- 未运行任何预训练；未新跑 CAT/GATE/XATTN/GLT_REF 的 pretrain smoke（只读复用现有 `deploy_00002.pt`）。
- **无任何性能结论**：五个 XC R² 只是一次接口 smoke 的观测值，本轮不做臂间排序、不做融合优劣判断、不写「PH 改善 XC」；接口可用性（P1）与性能（P2）严格分开。
- 未访问 outer-test（各 unit `outer_test=NOT_RUN`）；未启动 5k 正式训练、P2/P3、EPS/EAT、fold1；aggregator 规则未被改动以「凑通过」。
- **不写 `P1 PASS`/「正式验收通过」**；ZCode 只报告执行证据完整，最终验收由 ChatGPT 审查远端 commit 后给出。
- 未改写 r3–r8 历史记录（含 r8 的 FAILED 记录）；未恢复 `Plan.md`；未修改 `.zcodeignore`、`PH.md`、`3D.md`。未回滚其它执行者的改动。

**八、提交与同步**：本轮 3 个文件（`scripts/finetune_mcl_ph.py`、`tests/test_mcl_ph_r9_summary.py`、`MCL-PH.md`）提交为 **`eee5583`**，非 force push 成功 `73bd4cc..eee5583 dev -> dev`；`git ls-remote origin dev` = `eee55832ec160519e8625965d258bed5ad278511`，与本地 HEAD 一致。产物与日志：`results/mcl_ph_20260921/p1/finetune_r9/{<arm>/xc/fold0/,aggregate.json}`、`logs/mcl_ph_20260921/{r9_regression.log,r9_finetune_smoke.log,r9_tmux_finetune.log}`，按 `.gitignore` 不提交，只按路径引用。**未自动启动 P2 或任何后续实验**；交回 ChatGPT/Codex 审查。

### 13.12 r10 / P2-E2E 执行记录（ZCode 执行；2026-09-21 UTC；基准 `dev@be9fee6`；**Phase I 完成并已提交；Phase II 在 GLT_REF 单次启动被外部 supervisor 误停后按 §21/§41 阻断**）

**计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r10「P2-E2E：P2 aggregator → 4×5000 正式预训练 → 5 arms × 3 tasks × 2 folds FULL 微调 → 30-unit 验收 → P2 aggregation + parent selection」** |
| 状态 | **阻断（计划未完成）**：Phase I（P2 development aggregator + model-free regression）已完成、已提交并推送（`fbd78ff`）；Phase II 的**第一个正式启动 GLT_REF** 在 step 154 被我自己挂上的外部 stall supervisor 以信号 15 停止（`STOPPED_BY_SUPERVISOR`）。按 §12「任一失败 STOP」与 §21「每个 arm 最多 1 次正式启动、失败已执行 updates 永久计入预算、不自动 resume/retry」、§41「supervisor intervention → 立即停止、只允许只读定位、不边修边继续」**停止整个 P2 执行并交回**。**不写 `P2 execution evidence complete`，不写 `P2 accepted`。** |
| 授权来源 | 用户 2026-09-21 的 r10 指令（完整 P2）：4 starts / ≤20,000 pretrain updates，30 development units / ≤900 epochs，P3 与 outer-test 为 0 |
| 角色 | Codex/ChatGPT 规划与审查；ZCode 执行 |
| 开始时 HEAD / pull | `dev@be9fee6`（= 计划 §1 写的 HEAD）；`git pull --ff-only origin dev` → **Already up to date**；`git ls-remote origin dev` = `be9fee6…`；无其它执行者改动目标代码；未触发停止条件 |
| tmux | session `Uni-Poly`，window **83 `mcl_ph_p2_pretrain`**（`mcl_ph_p2_downstream` 窗口未使用，Phase III 未启动） |

**一、Phase I（§4–§11）——完成并已提交**

新增 `scripts/aggregate_mcl_ph_p2.py`：复用 `scripts.aggregate_mcl_ph.check_unit`（**未另写一套 unit 校验**），固定 scope = 5 arms × 3 tasks × 2 folds = 30 units、`stage=development`、`expected_pretrain_step=5000`；输出 `status/units_expected=30/units_accepted/units_rejected/outer_test=NOT_RUN`、每臂 task×fold×mean 与 `macro3`、三个 MCL 臂对 `O8_ONLY`/`GLT_REF` 的 matched deltas、§9 预注册 gate（**同时**对两个 baseline）与 §10 parent selection（CAT 默认、GATE/XATTN 替代条件、Macro3 差 <0.002 视为工程持平并优先 GATE、无合格臂 → `NO_QUALIFIED_ARM`）。任何 unit 缺失/失败/NaN/错误 step/protocol/split/`outer_test` → `INCOMPLETE` 且**不计算 winner**（`r2/deltas/qualification` 置 null，退出码 4）。**未改动 P1 aggregator 与任何生产代码。**

新增 `tests/test_mcl_ph_p2_aggregate.py`（纯 CPU、无模型、无 forward/backward）：覆盖 §11 要求的全部 10 个用例（完整 30 units→PASS；缺 unit / NaN / outer_test 错误→INCOMPLETE；CAT qualified→CAT；GATE 满足替代条件→GATE；XATTN 不满足 XC fold 一致性→不可替代；CAT 失败 GATE 合格→GATE；无合格臂→`NO_QUALIFIED_ARM`；Macro3 近似持平→GATE tie preference）另加 3 个纯函数/单 baseline 用例，**13 passed in 5.18s**（`logs/mcl_ph_20260921/p2_aggregator_regression.log`）。CLI 端到端复核：合成 30 units → `PASS/30/30/0/parent=m_cat` exit 0；对真实 r9 根（smoke 产物）→ `INCOMPLETE`、30 rejected、首条理由 `run stage is 'smoke', not 'development'`（不会把 smoke 当 development 接受）。

提交：**`fbd78ff`**（`scripts/aggregate_mcl_ph_p2.py`、`tests/test_mcl_ph_p2_aggregate.py`），非 force push `be9fee6..fbd78ff dev -> dev`，`git ls-remote origin dev` 与本地 HEAD 一致。

**二、Phase II 正式预训练——GLT_REF 单次启动被误停（阻断点）**

启动前核验（全部通过）：4 张 GPU 空闲、无其它训练进程；§14 的数据路径（cohort `…/pi1m/cohort_30f17b59bc5862a1`、cache `data/processed/mips_trimer_scage`、dual static `…/pi1m/dual_static_v1`、split `pretrain_split_v1.json`、MCL `statistics.npz`）与 §27 的四个 config 均存在；config 实测与 §16/§17 完全一致（seed 42、microbatch 84、global_batch 1008、max_steps 5000、save_every 1000、lr 2e-4、warmup 2000、schedule 20000、bf16、router dense 500 → top2、balance 1e-3、cutoffs 2/3/4）；输出根 `results/mcl_ph_20260921/p2/pretrain` 不存在（四个 arm 目录与 `shared_new_init.pt` 均 `test ! -e` 通过）。

执行方式：`/tmp/p2_pretrain_driver.sh`（执行侧驱动，**未改任何生产代码**）按 §12 顺序 `glt_ref → cat → gate → xattn` 串行启动，逐 arm 复刻 `scripts/run_mcl_ph_pretrain_smoke.sh` 的 arm 命令与 `verify_mcl_ph_arm.py` 校验（MCL 臂 `--strict-cleanup`），任一非零退出/校验失败即中止；GLT_REF 额外把 §18 要求的 `--diagnostic-save-steps 1000 2000 3000 4000 5000` 显式写出（生产 launcher 只能传单一值，故该臂直调同一 runner）。实际命令（window `mcl_ph_p2_pretrain`，16:09:53Z 启动）：

```bash
timeout -k 60 14400 python3 -m torch.distributed.run --nproc_per_node=4 --standalone \
  scripts/pretrain_glt_dual.py --config configs/mts/glt_pred_s3b_b_fp.json \
  --cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1 \
  --cache-root data/processed/mips_trimer_scage \
  --dual-static-root data/processed/glt_dual_v2/pi1m/dual_static_v1 \
  --pretrain-target-root data/processed/glt_dual_v2/pi1m/pretrain_targets_v1 \
  --third-task fp --diagnostics --diagnostic-save-steps 1000 2000 3000 4000 5000 \
  --stop-after-step 5000 --prep-workers 12 --output results/mcl_ph_20260921/p2/pretrain/glt_ref
# 同一 shell 内并行启动：tests/_mcl_ph_r5_stall_supervisor.py --root-pid <runner>
#   --stages-dir results/mcl_ph_20260921/p2/pretrain/glt_ref --report .../stall_supervisor_glt_ref.json
```

**训练本身是健康的**：4 rank 正常前向/反向，`losses` 有限（step 8 时 `[4.5606, 2.2497, 0.6944]`），lr 随 warmup 正常上升，实测 **0.857 s/step**（step 35→105 / 60 s），到停止前已执行 **step 1–154（154 个 update，全部有 diagnostics 记录）**。

**停止事实**（`stall_supervisor_glt_ref.json`、`runtime.json`、arm 日志）：

| 证据 | 值 |
| --- | --- |
| supervisor | `status=STOPPED_BY_SUPERVISOR`、`intervened=true`、`silent_seconds_at_stop=180.016`、`stack_requested_after_silent_seconds=120.012` |
| 进度信号 | **`progress_at_stop=[]`、`stage_tail={}`**（supervisor 的进度定义只有 `stages_rank*.log` 的新增 mark） |
| 终止方式 | SIGTERM(15) → `torchrun` 报 `SignalException: Process … got signal: 15`；`tree.survivors_after_kill=[]`，现场无残留进程 |
| 时间 | 启动 16:09:53Z → 停止 16:12:55Z，墙钟 ≈182 s（≈ 180 s 静默阈值） |
| 产物 | `glt_ref/` 只有 `run.json`、`runtime.json`（仍为 `status: RUNNING`）、`records_rank{0..3}.jsonl`、`diagnostics_steps.jsonl`；**无 `resume_*.pt`、无 `deploy_*.pt`**；`cat/`、`gate/`、`xattn/`、`shared_new_init.pt` 均未创建 |

**三、根因（只读定位；执行侧驱动缺陷，非模型/生产代码缺陷）**

- supervisor 的进度判据**只**认 `stages_rank*.log` 的真实 append（r7A 授权的最小机制）。该文件由 **MCL runner**（`scripts/pretrain_mcl_ph.py`）写出（r7B `pretrain_r7b/cat/` 内实测有 `stages_rank0-3.log` 与 `stall_stack_rank*.txt`）。
- **GLT dual runner（`scripts/pretrain_glt_dual.py`）根本不写 stage log**：`find results/mcl_ph_20260921/p2/pretrain/glt_ref -name 'stages_rank*.log' -o -name 'stall_stack*'` 为空（supervisor 的 `progress_at_stop=[]` 是同一事实的另一侧证据）。因此把该 supervisor 挂到 glt_ref 上，**必然**在 180 s 后误判为“停滞”并杀掉一个正常推进的训练。
- 该 supervisor 在 r7B/r8 只被挂到过 MCL runner（cat/gate/xattn），**从未**用于 GLT_REF；本轮执行侧驱动为了让四个 arm 的停止契约一致而统一挂载，这就是本次误停的直接原因。
- 结论：**不是**数据/identity/NaN/NaN-grad/checkpoint/strict-load 失败，也**不是** §41 所列的模型或生产实现缺陷；被停止的是一个健康的正式训练启动。按 §41 本轮**只做只读定位，未改任何代码，未 resume/retry，未启动后续 arms**。

**四、预算核算（失败计入，不写 0）**

| 项目 | 上限 | 实际 |
| --- | --- | --- |
| 正式预训练启动 | 4（每 arm 1） | **1**（GLT_REF；被 supervisor 误停，其余 3 个 arm **未启动**） |
| 正式预训练 updates | ≤20,000 | **154**（GLT_REF step 1–154；CAT/GATE/XATTN = 0） |
| 产出 checkpoint | — | **0**（无 `resume_05000.pt`/`deploy_05000.pt`，GLT_REF 全部 cadence 点均未到达） |
| development units | ≤30 starts / ≤900 epochs | **0 / 0** |
| P2 aggregate / qualification / parent selection | — | **未运行** |
| P3 / outer-test / 额外 seed / sweep | 0 | **0**（各 unit `outer_test` 从未被访问；本轮无 downstream unit） |
| 代码修改 | 仅 §43 允许的两处新增 | `scripts/aggregate_mcl_ph_p2.py`、`tests/test_mcl_ph_p2_aggregate.py`（生产代码与 P1 aggregator 未改） |

**五、未执行 / 未声称**

- 未启动 `cat`/`gate`/`xattn` 三个 MCL 正式预训练；未创建 P2 的 `shared_new_init.pt`，故 §19 的 schema/SHA 校验尚未发生。
- 未做 §22 三臂 deployment 验收、未做 §23 GLT_REF 5k strict-load 验收（无 5k 产物）。
- 未运行任何 development unit、未运行 P2 aggregator、未做 qualification 与 parent selection；**不写 `P2 execution evidence complete`、不写任何合格臂/selected parent、不做任何性能结论**。
- 未重试 GLT_REF、未 resume、未改 batch/worker/lr/radius；未删除或覆盖失败现场；未恢复 `Plan.md`；未修改 `.zcodeignore`、`PH.md`、`3D.md`。

**六、交回 ChatGPT 的决策点（执行者不自行决定）**

1. GLT_REF 的**唯一一次正式启动已消耗**（154 updates，无 checkpoint）。是否授权第二次启动、以及是否仍按 §13 的 `p2/pretrain/glt_ref` 路径（该目录已存在，runner 的防覆盖守卫会拒绝新建训练），需由规划方决定（例如换新根目录或新修订号）。
2. 是否把 GLT_REF（`pretrain_glt_dual.py`）**排除在外部 stall supervisor 之外**，或改为不对 dual runner 设静默停止（MCL runner 的 `stages_rank*.log` 机制不适用于它）。
3. 其余三个 MCL arm 与 Phase III 的 30 个 unit 是否仍按原预算执行（本轮的启动计数与 update 数需按上表计入）。

**七、提交与同步**：本轮共两个 commit，均已非 force push 到 `origin/dev`：**`fbd78ff`**（Phase I：`scripts/aggregate_mcl_ph_p2.py`、`tests/test_mcl_ph_p2_aggregate.py`）与 **`e86d795`**（本记录，仅 `MCL-PH.md`）；`git ls-remote origin dev` = `e86d7953816809abd0bd2f3f7d0d6ca6e950ee4e`，与本地 HEAD 一致。产物与日志：`results/mcl_ph_20260921/p2/pretrain/glt_ref/`（失败现场，未删除未覆盖）、`results/mcl_ph_20260921/p2/pretrain/stall_supervisor_glt_ref.json`、`logs/mcl_ph_20260921/{p2_pretrain_driver.log,p2_pretrain_glt_ref.log,p2_supervisor_glt_ref.log,p2_tmux_pretrain.log,p2_aggregator_regression.log}`，按 `.gitignore` 不提交，只按路径引用。**未自动重试 GLT_REF、未启动任何后续 arm 或下游实验**；交回 ChatGPT/Codex 决定后继续。

### 13.13 r10R1 执行记录（ZCode 执行；2026-09-21 UTC；基准 `dev@a5d996e`；**P2-E2E 续行：阈值边界返修 + GLT_REF 第二次正式启动授权**）

**计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r10R1「P2 aggregator 预注册阈值边界修复 + GLT_REF 第二次正式启动 + 继续原 P2-E2E」** |
| 状态 | **执行中**：返修与回归已完成并提交（见一、三）；GLT_REF 第二次正式启动与后续 P2 阶段在本记录末尾追加（见四） |
| 授权来源 | 用户 r10R1 指令：新增 **GLT_REF extra start = 1**（≤5000 updates、**from step0**、禁止从 step154 resume）；返修只允许改 aggregator 阈值与 regression |
| 角色 | Codex/ChatGPT 规划与审查（含 r10 的 aggregate gate 边界审查）；ZCode 执行 |
| 开始时 HEAD / pull | `dev@a5d996e`（= 审查基线）；`git pull --ff-only origin dev` → **Already up to date**；`git ls-remote origin dev` = `a5d996e…`；工作区干净 |
| 被保留的失败预算 | GLT_REF formal starts = **1**、updates = **154**（r10，`STOPPED_BY_SUPERVISOR`，执行侧 supervisor 挂载错误）；现场 `results/mcl_ph_20260921/p2/pretrain/glt_ref/` **永久保留，不删不移不覆盖** |

**一、预注册阈值边界修复（§2/§3）**

锁定合同是「任一任务均值退化**不超过** 0.01」，即 ΔR²_task **≥ −0.01**（含等号）。原实现用了严格大于：

```python
'no_task_sacrifice': all(delta[f'{task}_mean'] > TASK_SACRIFICE_FLOOR for task in TASKS)   # 旧
'no_task_sacrifice': all(delta[f'{task}_mean'] >= TASK_SACRIFICE_FLOOR for task in TASKS)  # 新
```

| 项目 | 结果 |
| --- | --- |
| 修改范围 | **只改这一处比较符**（+1 行注释）；`MACRO3_MIN_DELTA`、`XC_MEAN_MIN_DELTA`、`xc_fold0/1_positive`、替代条件、tie band、selection 规则**均未改动** |
| 边界行为实测 | `delta = 0.0 - 0.01`（正好是 float(−0.01)）：旧 `>` → `False`（**误判为 sacrifice**），新 `>=` → `True`；`delta = −0.0101` 两者皆 `False` |
| 新增回归 | `test_n_a_task_mean_at_exactly_the_floor_still_qualifies`（恰好 −0.01 → `no_task_sacrifice=True` 且 `qualified=True`）、`test_o_a_task_mean_below_the_floor_is_a_sacrifice`（−0.0101 → 均为 `False`）；两者为纯函数、无模型 |
| 回归结果 | `python -m pytest tests/test_mcl_ph_p2_aggregate.py -q` → **15 passed in 5.22s**，exit 0；CPU-only（GPU/forward/backward/optimizer update = 0） |
| 未改 | `src/modules/mcl_ph.py`、`src/dataset/mcl_ph_view.py`、`scripts/pretrain_mcl_ph.py`、`scripts/pretrain_glt_dual.py`、`scripts/finetune_mcl_ph.py`、`configs/mts/*.json`；**未给 dual runner 添加 StageLogger**（该问题不是 production runner bug） |

**二、预算与运行策略修订（§5/§6/§7/§12）**

| 项目 | 内容 |
| --- | --- |
| 已消费（永久计入） | GLT_REF formal starts = **1**、optimizer updates = **154**（不得改写为 0） |
| 新增授权 | GLT_REF extra start = **1**、retry updates ≤ **5000**、**from step0**（正式 matched trajectory 必须从原始初始化开始，禁止从 step154 resume） |
| 修订后历史上限 | **154 + 4×5000 = 20,154** optimizer updates；formal starts = **5**（1 次失败 GLT_REF + 1 次重试 + CAT + GATE + XATTN） |
| GLT_REF 新输出根 | `results/mcl_ph_20260921/p2/pretrain/glt_ref_r10r1/`（不得写入旧失败目录）；日志 `logs/mcl_ph_20260921/p2_glt_ref_r10r1_pretrain.log` |
| GLT_REF 训练语义 | 与失败启动保持一致：world 4、seed 42、`third_task=fp`、5000 updates、同一 P_train/common-init/batch/schedule、`--diagnostics --diagnostic-save-steps 1000 2000 3000 4000 5000 --stop-after-step 5000`（仅去掉错误挂载的外部 supervisor） |
| supervisor 策略（§7） | `tests/_mcl_ph_r5_stall_supervisor.py` **禁止**用于 GLT_REF（dual runner 不产生 `stages_rank*.log`，必然误判）；CAT/GATE/XATTN **继续启用**（120 s stack / 180 s stop，MCL runner 确有 StageLogger marks） |
| downstream 映射（§14） | `glt_ref` 与 `o8_only` **必须**使用 `p2/pretrain/glt_ref_r10r1/deploy_05000.pt`；不得误用失败目录 |

**三、Phase II 执行结果（GLT_REF 重试成功；M_CAT 在 step 4928 被本轮的 timeout 上限停止）**

驱动 `/tmp/p2r1_driver.sh`（执行侧脚本，未改生产代码）于 window `Uni-Poly:mcl_ph_p2_glt_ref_r10r1` 顺序启动：GLT_REF 重试（**无 supervisor**）→ CAT/GATE/XATTN（带 MCL supervisor）→ 三臂 deployment check；任一失败即中止。

**GLT_REF 第二次正式启动——成功（§10 全部满足）**

| 项目 | 结果 |
| --- | --- |
| 启动 / updates | `results/mcl_ph_20260921/p2/pretrain/glt_ref_r10r1`，**step0 → 5000**（未从 step154 resume） |
| 命令 | 与失败启动同一语义（world 4、seed 42、`third_task=fp`、microbatch 84、global_batch 1008、bf16、lr 2e-4、warmup 2000、schedule 20000、`--stop-after-step 5000`、`--diagnostics --diagnostic-save-steps 1000 2000 3000 4000 5000`），**仅去掉了错误挂载的外部 supervisor** |
| 进程 / runtime | `EXIT=0`（21:43:37Z→23:27:54Z，墙钟 ≈1 h 44 min）；`runtime.status=PASS`、`completed_steps=5000` |
| checkpoint cadence（§18） | `resume_01000/02000/03000/04000/05000.pt` 与 `deploy_01000/02000/03000/04000/05000.pt` **全部存在** |
| verifier | `scripts/verify_mcl_ph_arm.py --label glt_ref --arm-dir … --updates 5000` → **VERIFY=0** |
| CPU strict-load（§10/§23） | `load_dual_deployment(..., expected_step=5000)` **成功**；`run.json third_task=fp`、`step=5000`；`o8_only_copied_tensors=83`、`o8_only_copy_complete=true`、`problems=[]`、`status=PASS`（`p2_glt_ref_precheck.py`） |

**M_CAT——在第 4928 个 update 处被本轮驱动的 `timeout` 上限停止（失败，启动额度已消耗）**

| 项目 | 结果 |
| --- | --- |
| 启动 | 23:28:00Z；命令与 r10 的 MCL 臂一致（`configs/mts/mcl_ph_cat.json`、`--shared-new-init p2/pretrain/shared_new_init.pt`、world 4、prep 12、`--stop-after-step 5000`、MCL supervisor 启用） |
| 停止 | **03:28:02Z，`EXIT=124`（`timeout -k 60 14400` 的 4 h 上限到期）**；日志尾部为 `SignalException: … got signal: 15` |
| 实际进度 | **step 4928 / 5000**（差约 7 分钟）；末步 `losses={atom 0.1028, geometry 0.0266, balance 0.0527}`、`grad_total_preclip=0.2435` 全部有限，`router_mode=top2` |
| milestones（§20/§13） | `steps.jsonl` 记录 `1,2,500,501,1000,2000,3000,4000`；**`500=dense`、`501=top2`**、其后均 `top2`（5000 未到达，故无该点） |
| supervisor | `status=ROOT_EXITED`、**`intervened=false`**、`silent_seconds=5.0`（未介入；`stall_stack_rank*.txt` 全 0 字节）——即**不是**停滞，而是被我的超时上限杀掉 |
| 产物 | 目录内只有 `run.json`、`runtime.json`（仍为 `RUNNING`）、`step_0000.json`、`steps.jsonl`、`stages_rank*.log`；**无 `resume_*.pt`、无 `deploy_*.pt`**（MCL runner 只在最终步保存） |
| 现场 | `results/mcl_ph_20260921/p2/pretrain/cat/` **原样保留**，未删除未覆盖；`shared_new_init.pt` 已由该运行创建，其 schema/SHA 检查因该臂未成功而**未执行** |

**GATE / XATTN 未启动**；三臂 deployment check（§13）与 30 个 development unit（Phase III）、P2 aggregate（Phase IV）**均未执行**。

**四、预算核算（失败计入，不写 0）**

| 项目 | 授权 | 实际 |
| --- | --- | --- |
| GLT_REF 失败启动（r10） | 1 | **1**（154 updates，现场保留） |
| GLT_REF 重试（r10R1） | 1 | **1**（**5000 updates，成功**） |
| CAT / GATE / XATTN | 各 1 | **1 / 0 / 0**（CAT 执行 **4928** updates 后失败；GATE、XATTN 未启动） |
| 预训练 updates 合计 | ≤ 20,154 | **10,082** = 154 + 5000 + 4928 |
| development units | ≤ 30 starts / ≤ 900 epochs | **0 / 0** |
| P2 aggregate / qualification / parent selection | — | **未运行** |
| P3 / outer-test / 额外 seed / sweep / 第三次 GLT_REF | 0 | **0** |

**五、根因（只读定位；执行侧超时设置，非模型/生产代码缺陷）**

- 本轮驱动的每个 arm 都套了 `timeout -k 60 14400`（4 h）。该值是按 GLT_REF 的实测速率（≈1.25 s/step 含启动）定的。
- 但 **MCL 三个 arm 的实际速率 ≈2.9 s/step**：`preparation_seconds` 虽小、`forward_backward_seconds ≈0.45 s`，然而在线 PH/topology 与 2/3/4 Å 邻域的 12×4 个 `pt_data_worker` 把 CPU 打满（load ≈50/112），使每步墙钟远高于 fb 时间。5000 updates 需要 ≈4.2–4.5 h（含启动与最终 311 MB `resume` + 80 MB `deploy` 导出），4 h 上限因此在 **step 4928** 处提前触发。
- 这与 r10 的 GLT_REF 事件同属**执行侧参数错误**（r10 是 supervisor 挂错，本轮是超时上限太短）；训练本身健康（无 NaN/Inf、无停滞、supervisor 未介入、milestones 正常）。生产 launcher/runner 与 §43 允许的改动范围**未被修改**。

**六、未执行 / 未声称**

- 未重试 CAT（r10R1 §19 明确把「MCL arm retry」列为禁止项，只能由规划方/用户另行授权）；未启动 GATE/XATTN；未运行三臂 deployment check、30 个 development unit、P2 aggregate。
- **不写 `P2 execution evidence complete`、不写任何合格臂/selected parent、不做任何性能结论**。
- GLT_REF（`glt_ref_r10r1`）的 5k 产物本身是完整且已验证的（PASS/5000/verifier 0/strict-load PASS/O8 83/83），可供后续 downstream 映射使用；CAT 现场保留，未改写 r10 记录，未删除旧失败目录。
- 未修改 `src/`、`scripts/pretrain_*.py`、`scripts/finetune_mcl_ph.py`、`configs/mts/*.json`；未给 dual runner 加 StageLogger。

**七、交回 ChatGPT 的决策点**

1. 是否授权 **CAT 第二次正式启动**（以及是否换新输出根，例如 `cat_r10r2`；现有 `p2/pretrain/cat/` 已存在，runner 的防覆盖守卫会拒绝在该路径新建训练），并把 MCL 臂的超时上限改为 ≥5 h（实测 ≈2.9 s/step）。
2. 是否把「每臂超时」纳入计划固定参数（本轮 4 h 由执行侧自定，属计划未明确项）。
3. GATE/XATTN 与 Phase III/IV 是否在本轮授权内继续（目前均未启动、预算未消耗）。

**八、提交与同步**：本轮共两个 commit，均已非 force push 到 `origin/dev`：**`c34fbc2`**（阈值边界返修 + 边界 regression + r10R1 计划头/返修记录）与 **`65820e4`**（本节执行结果与阻断记录）；`git ls-remote origin dev` 与本地 HEAD 一致（`65820e44bed9a1f847e47258a903fd79c5898ed1`）。产物与日志：`results/mcl_ph_20260921/p2/pretrain/glt_ref_r10r1/`（5k 成功产物）、`results/mcl_ph_20260921/p2/pretrain/{cat,glt_ref}/`（两处失败现场，均原样保留）、`logs/mcl_ph_20260921/{p2r1_driver.log,p2r1_tmux.log,p2_glt_ref_r10r1_pretrain.log,p2_pretrain_cat.log,p2_supervisor_cat.log}`，按 `.gitignore` 不提交，只按路径引用。**未自动重试 CAT、未启动 GATE/XATTN 或任何下游实验**；交回 ChatGPT/Codex 决定后继续。





### 13.14 r10R2 执行记录（ZCode 执行；2026-09-22 UTC；基准 `dev@9bad4c8` → 实现提交 `ff32085`；**离线 topology trajectory cache 实现、通过 hard gate，full 5.04M 构建已完成并经独立校验**）

**一、计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r10R2-PERF**（用户当轮消息 §0–§35） |
| 授权来源 | 用户当轮消息本身：§33 明确预算「CPU 单测 + 20k worker benchmark + 1 global batch 等价性 + full 5.04M 构建；GPU 正式训练 0，最多 2-step cached smoke；ChatGPT 审查 cache 实现与 benchmark 前不得启动正式 CAT retry」 |
| 角色 | 执行：ZCode；规划/审查：ChatGPT（Codex 角色） |
| 基准 commit | r10R1 结束点 `dev@9bad4c8`；本轮代码实现提交 **`ff32085`**（已 push） |
| 允许改动（§32） | `src/dataset/mcl_ph_view.py`、新增 `src/dataset/mcl_ph_trajectory_cache.py`、新增 `scripts/build_mcl_ph_trajectory_cache.py`、`scripts/pretrain_mcl_ph.py`、新增 `tests/test_mcl_ph_trajectory_cache.py`、`MCL-PH.md` |
| 禁止项 | 未修改 `src/modules/mcl_ph*.py`、fusion/router/loss/configs/finetune/aggregators/downstream；未启动 CAT/GATE/XATTN；未跑正式预训练或微调 |

**二、实现（§4、§10–§19）**

| 文件 | 变化 |
| --- | --- |
| `src/dataset/mcl_ph_view.py` | 抽出**唯一**的 noisy-view helper `build_mcl_ph_view(topology, trimer, smiles, *, seed, key, position, sigma, ratio, identity=None, view='noisy')`：内部按 `sample_generator(seed, key, position)` → `motif_mask(...)`（消耗 RNG）→ `_perturb(...)` 的同一顺序取噪声，因此在线路径与 cache builder **不存在两套 RNG 实现**。`prepare_mcl_ph_sample(..., trajectory_override=None)`：override 必须满足 `(31,5)`/float32/全有限，缺失或元数据不符则 **FAIL CLOSED**（不回落在线 PH）；`geometry_valid=False` 与 override 非零互相矛盾时直接报错；`MCLPHMicrobatchStream(..., trajectory_cache=None)` 用**已算出的 position**取行 |
| `src/dataset/mcl_ph_trajectory_cache.py`（新增） | `MCLPHTrajectoryCache` 只读 reader（`np.load(mmap_mode='r')`、惰性开 shard、fork 后重开映射、首次打开校验 manifest sha256、identity/coverage 失败即报错）；`position_trajectory(...)`（position → `OrderedSampleStream.index_at` → sample key → 共享 noisy view → `materialize_dual_geometry` 校验 → `five_descriptors`）；`TrajectoryCacheBuilder`（每段写盘后落 **segment marker**） |
| `scripts/build_mcl_ph_trajectory_cache.py`（新增） | 分片并行 builder：**不**计算 2/3/4 Å 图、`edge_distance`、nonbond 候选、模型前反向与优化器；完成证据由文件系统（marker）而非 driver 内存承担；`--adopt-written-segments` 用于接手一份 driver 记录丢失的已写数据 |
| `scripts/pretrain_mcl_ph.py` | 新增 `--trajectory-cache`；缺省 `trajectory_cache={'mode':'online'}`，提供时先按 run identity（seed/σ/mask ratio/global_batch/split/split-SHA/cohort hash/static hash）**精确校验**，并在 `run.json` / `runtime.json` 写入 `mode/path/schema/dtype/total_positions/shard_size/manifest_identity` |
| `tests/test_mcl_ph_trajectory_cache.py`（新增） | 40 项：布局、reader 值/dtype/可写/惰性/pickle、8 个 identity 字段逐项拒绝、覆盖不足/缺 shard/损坏 sha/截断/shape-dtype/布局不符/无 manifest 全部拒绝、该 helper 的严格性、真实数据 online≡cached（`torch.equal`）、worker 数不变性（0/1/4/12）、以及本轮新增的 builder 驱动层（marker/续建/恢复/失败上报/采纳） |

**三、验证结果（全部为真实执行）**

| 验证 | 命令 / 产物 | 结果 |
| --- | --- | --- |
| §23 hard gate（真实 1 个 global batch） | `tests/_mcl_ph_r10r2_batch_gate.py`（positions 0–1007，DataLoader worker + inline 两条路径） | **PASS**：1008 samples，17 个 batch 字段 + 19 个 label 字段逐项相等，`mcl_fallback_count`、`mcl_statistics_applied` 相等；cache shard sha256 `84250fb2…122ed4` |
| 单测 | `pytest tests/test_mcl_ph_trajectory_cache.py` | **40 passed**（含真实 cohort 等价性、worker 不变性、以及 trainer 侧 identity 拒绝） |
| 既有测试回归 | `test_mcl_ph_pretrain / protocol / modules` | 12 项失败**与本轮无关**：10 项为 `src/modules/mcl_ph_pretrain.py:77` 的 `global_sum` NameError（仅在不传 `global_count` 的调用下触发，生产调用点都传），2 项 launcher 协议测试早于本轮；已在 `git worktree`（`dev@9bad4c8`）复现同样失败 |
| §13 worker benchmark（positions 0–19999） | `logs/mcl_ph_20260921/trajectory_cache_bench.log` | 12 workers **89.3/s**（224.0 s）、24 workers **117.0/s**（170.9 s，load1 66.9）、48 workers **109.3/s**（183.0 s，load1 182.7）→ 选择 **24 workers**（不按核数直接开 112） |
| §24 数据管线 benchmark（4 rank × 12 workers，19 steps = 19,152 positions） | `results/mcl_ph_20260921/p2r2_gate/pipeline_{online,cached}.json` | online **220.32 samples/s**（86.93 s）vs cached **336.94 samples/s**（56.84 s）= **+52.9%**；max-over-ranks 每步 prep 等待 median 4.6725 → 1.9512 s（**−58.2%**）、p90 9.5332 → 6.4945 s（**−31.9%**）、p99 14.1065 → 7.3540 s、max 14.1065 → 7.3540 s；**无新增长尾** → §25 的「samples/s +30%」分支达成（p90 −50% 分支未达成，如实记录） |
| §33 2-step smoke（online vs cached） | `results/mcl_ph_20260921/p2r2_smoke/{online,cached}`、`logs/mcl_ph_20260921/p2r2_cached_smoke.log` | 两臂 `EXIT=0`；**step1 / step2 的 `losses` 与 `grad_total_preclip` 完全相同**（如 step2：atom 5.16257905960083 / geometry 0.25275495648384094 / balance 0.00011761983478209004 / grad 63.630943298339844）；provenance 正确（online=`{"mode":"online"}`，cached 含 path/schema/dtype/total_positions/shard_size/manifest_identity 8 字段）；`shared_new_init.pt` SHA 运行前后一致 `499309392d578daf7a7baf1d102d7a7f5c652e5a9b42ca2904ac9d83d1fab85a`（§30 未被重新生成） |

**四、full build 事故：driver 记录中断（已定位、已修复，数据无损）**

- **现象（04:39:47Z）**：driver 的进度日志停在 `04:39:45Z rows=93000`、manifest 的 `updated_at` 停在 `04:28:18Z` 且 `shards=[]`，但 24 个 worker 继续正常产出（05:19 仍 ≈140 positions/s）。
- **根因（已由 driver 退出时的 traceback 确证）**：`ff32085` 版 driver 第 151 行是
  `remaining = {index: sum(task[2] for task in tasks if task[0] == index) for index, _ in bounds}`。
  `bounds = shard_bounds(...)` 是 `(first_position, rows)` 的列表，`for index, _ in bounds` 取到的是 **first position**（0, 100000, …）而不是 shard 序号；shard 0 因 first=0 恰好命中，driver 因此撑过了前 100,000 行，第一个 shard 1 的结果到达时父进程在 `remaining[shard] -= rows` 处抛 **`KeyError: 1`**。这与日志最后一行 `04:39:45Z rows=93000`（即第 400 个 250 行块 = 100,000 行附近）完全吻合。
- **为什么“卡死”而不是立刻退出**：CPython 3.11 `multiprocessing.Pool` 的 `join()` 在 `pool._cache` 非空（结果迭代器未取完）时不会给 worker 发哨兵，因此 `finally: pool.close(); pool.join()` 会**阻塞到整个任务队列被算完**（本例 9.5 h）。该语义已用最小复现证明：200 个 0.3 s 任务、消费 20 条结果后 `close()+join()` → 15.1 s 才返回（`/tmp/mp_deadlock_repro3.py`）。
- **结论修正**：此处先前记录的假设（把异常解释为“worker 抛出的异常经结果流在父线程重抛”）**已被 traceback 取代**——真实异常是**父进程自身的索引 bug**；`join()` 的排空语义（即“卡死”的原因）仍成立。**这是 driver 缺陷，不是数据、模型或 checkpoint 缺陷**，与下方只读证据一致。
- **排除数据缺陷（只读证据）**：(i) 全量 zero-scan（`logs/mcl_ph_20260921/trajectory_cache_zeroscan.log`）显示 **shard 0–3 共 400,000 rows 零 zero row**（1600/1600 segment 可采纳）；(ii) 用同一 `position_trajectory` 单进程复算 positions **88,000–100,000**（`logs/mcl_ph_20260921/p2r2_failure_probe.log`）→ **0 failure**。
- **数据无损**：原进程在被异常打断后仍把 **全部 5,040,000 行写完**（13:58:41Z 退出，`EXIT=1`），未产生任何缺口。
- **修复（§32 允许文件内，最小机制）**：① 每段写盘 flush 后由**写盘的进程**落 `build_state/segments/segment_<shard>_<first>.json`；② worker 不再向上抛异常，而是回传 `SEGMENT FAILED shard=… first_position=… <异常>`（定位到 position）；③ driver 以 **marker 为准**判定进度/完成（结果流只作失败报告），**不再有任何 `bounds[shard]`/`remaining` 式下标算术**；④ 结束/异常一律 `pool.terminate()`（不再 `close()+join()`），⑤ `--stall-seconds` 看门狗 + 有界重派（`--max-requeues`），⑥ 断点续建：已标记段不重算、manifest 丢失可由 marker 重建（`RECOVERED shard …`）、`--adopt-written-segments` 按「整段非零」规则接手已写数据（该规则只会把已写行判成需重建，不会反向）。
- **回归**：修复后单测 39→40 passed；§23 hard gate 重跑 **PASS**，且 cache shard sha256 与修复前**逐字节一致**（`84250fb2…122ed4`）。

**五、full build 最终状态（已完成）**

| 项目 | 值 |
| --- | --- |
| 启动 | 2026-09-22T04:28:18Z，tmux `Uni-Poly:mcl_ph_cache`，日志 `logs/mcl_ph_20260921/trajectory_cache_build.log` |
| 参数 | 24 workers、shard 100 000、chunk 250、positions 0–5,039,999（51 分片）、CPU-only（无 GPU）；构建期 load1 ≈108–140（112 核） |
| 数据完成 | 2026-09-22T13:58:41Z（末分片 `trajectory_5000000_5039999.npy` 写满 40 000 行）；原 driver 于同一时刻以 `EXIT=1` 退出并打印上述 traceback |
| 墙钟 / 速率 | **9.51 h**（34 223 s）→ 5 040 000 rows，**平均 147.3 positions/s** |
| 分片 | **51**（50 × 100 000 + 1 × 40 000） |
| 体积 | **3 124 806 528 B = 2.91 GiB**（= 5 040 000 × 31 × 5 × 4 B，零额外开销；与 §6 预估一致） |
| 收尾（采纳+补建） | `--adopt-written-segments`：**20 160/20 160 段全部采纳、0 段需要重算、0 zero row**；14:00:06Z 写出 `complete=true` 的 manifest（`completed_shards=51`、`adopted_segments=20160`、逐 shard sha256、`builder_commit=d711514`） |
| reader 校验 | `MCLPHTrajectoryCache(root, required_positions=5 040 000)` 0.348 s 打开；51 shard 逐分片全量扫描 **0 zero row**、首次打开校验 **0 checksum 失配**；另用独立 `hashlib` 复算 51 个文件的 sha256，**0 失配** |
| 端到端等价（采纳数据） | positions **0 / 1007 / 1 234 567 / 2 500 000 / 3 777 777 / 5 000 000 / 5 039 999**：`np.array_equal(cached, online trajectory)` 全部成立；带 `trajectory_override` 时 `mcl_trajectory / mcl_pos / mcl_masked / mcl_edge_index / mcl_edge_distance / mcl_edge_type` 与全部 19 个 label 字段 `torch.equal` 成立（含末位 5 039 999，即被采纳而非重算的数据） |
| 现场 | `data/processed/mcl_ph_cache/p2_noisy_seed42_sigma003_step5000_v1/`（3 GB，按 `.gitignore` 不提交）；校验日志 `logs/mcl_ph_20260921/trajectory_cache_verify.log` |

**六、预算核算**

| 项目 | 授权 | 实际 |
| --- | --- | --- |
| GPU 正式预训练 / 微调 | 0 | **0** |
| cached smoke | ≤ 2 steps | **2 steps**（另加同 smoke 的 online 对照 2 steps，非正式训练） |
| 正式 CAT retry（`cat_r10r2`） | 0（须先经 ChatGPT 审查 cache 与 benchmark） | **0（未启动）** |
| GATE / XATTN / Phase III 30 units / aggregate / downstream | 0 | **0** |
| full cache 构建 | 授权（§14） | **完成 1 次**（9.51 h CPU；含一次 driver 事故的收尾采纳，未追加第二次全量构建） |

**七、未执行 / 未声称**

- cache 已完成并独立校验（`complete=true`、51 shard、0 缺口、0 校验失配、7 个跨全量位置逐位等价），但**未**据此声称任何训练/科学结论：§24 的 samples/s +52.9% 仍是 4-rank **数据管线**代理 benchmark，不是训练吞吐，也未验证 CAT/GATE/XATTN 的收益。
- 未启动 CAT/GATE/XATTN，未跑 downstream/finetune/Phase III/aggregate；`results/mcl_ph_20260921/p2/pretrain/cat/`（旧 CAT 现场）与 `shared_new_init.pt`（SHA 未变）未被触碰。
- 未修改 `src/modules/mcl_ph*.py`（含 `mcl_ph_pretrain.py:77` 的既有 `global_sum` NameError，仅上报不修）、fusion/router/loss/configs/finetune/aggregators/downstream。
- 3 GB cache 不提交；事故期间使用的探针脚本在 `/tmp`，未纳入仓库。

**八、提交与同步**：实现提交 `ff32085`、修复提交 `e2c89af`、trainer identity 测试提交 `d711514` 已非 force push 到 `origin/dev`；本轮收尾（§五最终数据、§四根因确证、§六/§七更新）见随后 commit。

### 13.15 r10R2-perf 执行记录（ZCode 执行；2026-09-22 UTC；基准 `dev@5e29c4f` → 代码提交 `a4ec194`；**hot path 去掉 per-run shard HASH + 三臂 paired 真实训练 benchmark**）

**一、计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r10R2-perf**（用户当轮消息） |
| 授权来源 | 用户当轮消息：① 删除正式训练 cache reader hot path 中重复的 shard SHA256 扫描（保留 `verify_checksums=True` 作为显式 forensic/debug 能力）；② 用真实训练循环确认 CAT/GATE/XATTN 三臂 cached wall 低于 online |
| 角色 | 执行：ZCode；规划/审查：ChatGPT（Codex 角色） |
| 基准 commit | `dev@5e29c4f`（= remote HEAD，本轮开始前 `git pull --ff-only` 为 up-to-date）；本轮代码提交 **`a4ec194`** |
| 允许 / 禁止 | 只改 production 打开路径 + 针对性测试 + 本节；**未**修改或重建 cache builder、未改 `src/dataset/mcl_ph_trajectory_cache.py` 的 reader 能力；未启动 5000-step retry、downstream、outer-test、P3；未覆盖任何正式 pretrain 目录 |
| 预算 | 6 个 scratch start × 每个 ≤30 updates = **180 updates**（性能验证，不计入 P2 pretraining 预算，不用于任何模型/科学排名） |

**二、代码改动（最小）**

| 文件 | 变化 |
| --- | --- |
| `scripts/pretrain_mcl_ph.py` | `open_trajectory_cache()` 改为 `MCLPHTrajectoryCache(path, expected=expected, **verify_checksums=False**)`，并加注释说明：2.9 GiB × 每 rank 一次全量 HASH 的成本高于它省下的在线 trajectory；完整性属于显式 forensic 通道。reader 默认仍为 `verify_checksums=True`，本轮**未**改动 `src/dataset/mcl_ph_trajectory_cache.py` |
| 仍然 fail-closed（未因关闭 SHA 而放宽） | schema、dtype=float32、shape `[31,5]`、seed、noise_sigma、mask_ratio、global_batch、sample-index split + split SHA256、cohort manifest hash、dual-static manifest hash、requested position coverage、shard 文件存在、`.npy` shape/dtype |

**三、测试（只跑相关 cache tests）**

`python3 -m pytest tests/test_mcl_ph_trajectory_cache.py -q` → **44 passed**（既有 40 + 新增 4），未跑全仓测试。

| 新增测试 | 证明内容 |
| --- | --- |
| `test_the_production_reader_does_not_hash_shards` | 以计数 monkeypatch 替换 `sha256_file`：经 production 入口打开并读完 10 个 position（跨 3 个 shard）→ **HASH 调用 0 次**；同一个 cache 用 `MCLPHTrajectoryCache(path)`（默认）读同样位置 → 每个 shard 恰好 1 次（默认能力保留） |
| `test_the_production_reader_trades_the_hash_for_the_field_checks` | 就地篡改 shard 字节（manifest 哈希过期）：production 路径按原值读出，forensic 默认路径报 `does not match its manifest hash` |
| `test_the_production_reader_still_fails_closed_without_checksums` | seed / noise_sigma / atom_mask_ratio / global_batch / split-SHA / cohort-hash 逐项仍报 `identity mismatch`；删除 shard 文件后读取仍报 `FileNotFoundError: shard is missing` |
| `test_the_production_reader_refuses_incomplete_coverage` | manifest 缺 shard 时 `require_positions` 仍报 `incomplete over the requested range` |

**四、paired benchmark 设置（全部为真实训练入口）**

6 次 `python3 -m torch.distributed.run --nproc_per_node=4 --standalone scripts/pretrain_mcl_ph.py`，`arm × mode`，scratch root `results/mcl_ph_20260921/p2r2_perf_bench/`（含 `README.md` 标注 **PERF BENCH**、非 P2 arm、不计预算）；`results/mcl_ph_20260921/p2/pretrain/` 只读未动。

| matched 项 | 值 |
| --- | --- |
| world size / seed | 4 / 42（config） |
| config | `configs/mts/mcl_ph_{cat,gate,xattn}.json`；三份除 `fusion_mode` 外逐键相同（microbatch 84、global_batch 1008、bf16、router_dense_updates 500、noise_sigma 0.03、atom_mask_ratio 0.3、cutoffs 2/3/4、`sample_index_artifact`=pretrain_split_v1 train） |
| accumulation / global batch | 3（=1008/(84×4)）/ 1008 |
| `shared_new_init.pt` | 同一个文件，运行前后 SHA 均为 `499309392d578daf7a7baf1d102d7a7f5c652e5a9b42ca2904ac9d83d1fab85a` |
| sample-index | `results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json` / train |
| 其他 | `--diagnostics`、`--prep-workers 12`、`--stop-after-step 30`、BF16 |
| **唯一变化** | 是否 `--trajectory-cache`（cached 指向完整 5.04M cache） |

脚本与日志：`tests/_mcl_ph_r10r2perf_bench.sh`（launcher）、`tests/_mcl_ph_r10r2perf_analyse.py`（等价性 + 计时分析，可复算）；`logs/mcl_ph_20260921/p2r2perf_bench.log`、每臂 `results/mcl_ph_20260921/p2r2_perf_bench/{arm}_{mode}.stdout.log`；tmux `Uni-Poly:mcl_ph_perf_bench`。**6/6 `EXIT=0`**。

**五、数值等价（steps 1 与 2）——三臂全部 PASS**

| arm | losses（三项分项逐位相同） | update_denominators | grad_total_preclip | router_mode | 其他 |
| --- | --- | --- | --- | --- | --- |
| cat | step2 atom 5.16257905960083 / geometry 0.25275495648384094 / balance 0.00011761983478209004 | 相同 | 63.630943298339844（online=cached） | dense/dense | valid_graphs、target_counts 相同 |
| gate | step1 atom 5.045283317565918（grad 94.16641998291016） | 相同 | 相同 | dense/dense | 同上 |
| xattn | step2 atom 4.9781174659729 / geometry 0.2527552545070648 / balance 0.00011750062549253926 | 相同 | 相同 | dense/dense | 同上 |

- sample ordering：两模式 `resume_00030.pt` 的 `ordered_keys`（911 391 个样本键）**逐位相同**，`next_position` 相同（30×1008=30 240）。
- provenance：online = `{"mode":"online"}`；cached = `{"mode":"cached", path, schema, dtype, total_positions 5040000, shard_size 100000, manifest_identity(8 字段)}`。

**六、性能（steady = steps 6–30；全程 = steps 1–30）**

| arm | mode | 30-step wall (s) | 训练 30 步 (s) | steady 总和 (s) | mean | median | p90 | p99 | max | prep median | prep p90 | fb median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cat | online | 125.3 | 97.17 | 69.28 | 2.771 | 0.555 | 6.522 | 8.566 | 8.566 | 0.139 | 3.578 | 0.398 |
| cat | cached | **81.4** | **53.77** | **38.24** | 1.529 | **0.521** | 3.424 | 7.539 | 7.539 | 0.014 | 1.784 | 0.387 |
| gate | online | 125.4 | 97.01 | 68.72 | 2.749 | 0.541 | 8.497 | 11.086 | 11.086 | 0.012 | 2.898 | 0.458 |
| gate | cached | **81.2** | **53.05** | **36.64** | 1.466 | 0.571 | 3.430 | 3.475 | 3.475 | 0.002 | 2.422 | 0.383 |
| xattn | online | 124.9 | 97.67 | 71.23 | 2.849 | 0.535 | 8.376 | 11.267 | 11.267 | 0.012 | 2.438 | 0.447 |
| xattn | cached | **80.7** | **52.88** | **36.07** | 1.443 | 0.551 | 3.246 | 4.477 | 4.477 | 0.001 | 1.125 | 0.411 |

speedup（online/cached）：30-step wall **1.54 / 1.54 / 1.55×**；训练 30 步 **1.81 / 1.83 / 1.85×**；steady 总和 **1.81 / 1.88 / 1.98×**；steady mean 同量级；**steady median step 1.065 / 0.947 / 0.972×**（GATE、XATTN 的 cached median 反而略高 0.02–0.03 s）。

steady 时间分解（prep / forward-backward / 其余）：cat 28.83/9.92/30.53 → **9.83/10.16/18.25**；gate 26.63/10.79/31.30 → **12.11/10.42/14.11**；xattn 23.72/11.34/36.17 → **7.77/11.10/17.20**。即 forward-backward 总和两模式相同（差异 ≤0.4 s），省下的是 prep 与**其余步内时间**（CPU 争用）。

paired 逐步差（online − cached，steps 1–30）：总和 **+43.41 / +43.96 / +44.80 s**，中位数 +0.179 / +0.027 / +0.115 s，online 更慢的步数 **24 / 20 / 23（共 30）**；prep > 0.5 s 的步数 online **12** vs cached **8 / 8 / 7**。

**七、对本轮验收条件的逐条判定**

| 条件 | cat | gate | xattn |
| --- | --- | --- | --- |
| cached steady median `step_seconds` < online | **PASS**（0.521 < 0.555） | **FAIL**（0.571 > 0.541） | **FAIL**（0.551 > 0.535） |
| cached 30-step total wall < online | PASS（81.4 < 125.3） | PASS（81.2 < 125.4） | PASS（80.7 < 124.9） |
| numerical equivalence | PASS | PASS | PASS |
| 无新的严重 tail regression | PASS（p90/p99/max 全面更小） | PASS | PASS |

- **`all_three_cached_faster`（按字面合并条件，含 median 项）= false**；若以「30-step 总时间 + 数值等价 + 无 tail regression」为准 = **true**。本条如实并列，不选择有利版本。
- 机制（有数据支持）：该管线用 DataLoader 预取，约 2/3 的步在两种模式下 `preparation_seconds ≈ 0.001 s`，因此 **median 对在线 PH 成本不敏感**；在线成本体现在队列排空的少数步（prep p90 2.4–3.6 s，online 12 步 prep >0.5 s）与 CPU 争用导致的其余步变慢（steady「其余」30.5–36.2 s → 14.1–18.3 s）。故 median 判据在 GATE/XATTN 上是 ±5% 量级的统计噪声，而总量差 1.5–1.9× 是一致的。
- 局限（如实记录）：每对按用户指定顺序 online 先跑，cached 因此受益于 page cache / GPU 状态，但 prep 差值本身是 worker 的 CPU 工作（24.1/24.5/27.9 s），无法由 page cache 解释；steady 窗口仅 25 步，median 统计不稳定。**本轮不追加 GPU 运行**，是否更换判据（总时间/稳态总和/p90/p99/max）由规划方决定。

**八、预算核算**：scratch benchmark **6 starts × 30 updates = 180 optimizer updates**（性能验证，不计入 P2 预算）；正式预训练/微调 **0**；5000-step retry、downstream、outer-test、P3 **均 0**。

**九、未执行 / 未声称**

- 未重跑 CAT/GATE/XATTN 5000-step，未跑 downstream/outer-test/P3，未用本 benchmark 做任何模型优劣或科学结论。
- benchmark 产物仅在 scratch root（`results/` 按 `.gitignore` 不提交）；`results/mcl_ph_20260921/p2/pretrain/{cat,glt_ref,glt_ref_r10r1,shared_new_init.pt}` 未被写入或覆盖。
- 未修改 cache builder、未重建 cache；`src/dataset/mcl_ph_trajectory_cache.py` 的 `verify_checksums=True` 默认未被删除。

**十、提交与同步**：代码提交 `a4ec194`（hot path + 4 项针对性测试）已 push 到 `origin/dev`；本节与两个证据脚本见随后 commit。

### 13.16 r10R2-perf-rev 执行记录（ZCode 执行；2026-09-22 UTC；基准 `dev@954310c`；**反序 cross-over 消除运行顺序混杂 + 修订性能 acceptance**）

**一、计划头**

| 项目 | 内容 |
| --- | --- |
| 计划 ID / 修订 | `MCL-PH-20260921-01` / **r10R2-perf-rev**（用户当轮消息） |
| 授权来源 | 用户当轮消息：执行 reverse-order cross-over（cached 先跑）以消除上一轮固定 `online → cached` 顺序造成的热缓存/运行顺序混杂；**并明确自本轮起把性能 acceptance 改为**「`cached training_seconds_full < online` 且 `cached training_seconds_steady < online` 且 `cached steady p90 step_seconds <= online` 且 numerical equivalence PASS」，`median step_seconds` 仅报告、不再作为 PASS/FAIL gate |
| 时间顺序（**不追溯改写**） | §13.15 的判定（按旧 median-step 合并 gate，`all_three_cached_faster = false`）**保持原样，本轮不改写、不覆盖、不重新解释**；修订 gate 自本轮（§13.16）起生效，依据即上述用户消息 |
| 角色 | 执行：ZCode；规划/审查：ChatGPT（Codex 角色） |
| 基线 commit | `dev@954310c17090151cb59dffc082cba41ac01d4af5`（= remote HEAD；`git pull --ff-only` up-to-date；开始前工作区干净、无正式训练进程、4 GPU 空闲） |
| 本轮禁止项（已遵守） | **未修改**模型、数据、cache 内容、PH 定义、fusion、router、loss、训练配置；未重建 cache；未启动 CAT/GATE/XATTN 5000-step、downstream、P3、outer-test；未覆盖上一轮 benchmark 或任何正式 P2 目录 |
| 预算 | 6 starts × 30 updates = **180 optimizer updates**（performance validation，不计入 P2 pretraining budget，不用于模型排名） |

**二、代码改动**

生产代码**零改动**（仍为 `a4ec194` 的 production reader）。本轮仅新增两个证据脚本并更新本节：`tests/_mcl_ph_r10r2perfrev_bench.sh`（反序 launcher）、`tests/_mcl_ph_r10r2perf_crossover.py`（两种顺序并列分析 + 修订 gate + cross-over 判定，可复算）。两次运行都不修改 cache 内容：`data/processed/mcl_ph_cache/p2_noisy_seed42_sigma003_step5000_v1` 只读挂载（manifest 仍为 `complete=true`、51 shards、5 040 000 positions、builder_commit `d711514`）。

**三、设置（与 A 轮逐项 matched，仅顺序不同）**

新 scratch root `results/mcl_ph_20260921/p2r2_perf_bench_rev/`（含 `README.md` 标注 **PERF BENCH (reverse order)**）；顺序 CAT cached→online、GATE cached→online、XATTN cached→online；world 4、seed 42、各臂同 config、同一 `shared_new_init.pt`（SHA 前后 `499309392d578daf7a7baf1d102d7a7f5c652e5a9b42ca2904ac9d83d1fab85a`）、同 sample-index（pretrain_split_v1/train）、microbatch 84、accumulation 3、global_batch 1008、BF16、`--diagnostics`、`--prep-workers 12`、`--stop-after-step 30`；唯一数据路径差异 = cached 传 `--trajectory-cache data/processed/mcl_ph_cache/p2_noisy_seed42_sigma003_step5000_v1`。**6/6 `EXIT=0`**，日志无 NaN/Inf。上一轮 root 未被触碰：`find … | sha256sum` 指纹前后同为 `0f3e48a1…b36fa`。

**四、数值等价（steps 1/2）与 sample ordering——两种顺序、三臂全部 PASS**

| arm | A 轮（online 先） | B 轮（cached 先） | sample ordering |
| --- | --- | --- | --- |
| cat | PASS | PASS | `ordered_keys` 911 391 键逐位相同、`next_position` 相同 |
| gate | PASS | PASS | 同上 |
| xattn | PASS | PASS | 同上 |

核对字段：losses（三项分项逐位）、update_denominators、grad_total_preclip、router_mode（dense）、valid_graphs、target_counts。额外确定性证据：**跨顺序**同 arm 的 steps 1/2 losses 与 grad_total_preclip 亦逐位相同（例：cat step2 atom `5.16257905960083`；gate `5.007333755493164`；xattn `4.9781174659729`，两种顺序一致）。未重复 full cache checksum，也未重复 1008-position equality gate。

**五、两种顺序并列统计（steps 1–30 full；steps 6–30 steady；单位 s）**

| order | arm | mode | wall | full | steady | step med | p90 | p99 | max | prep med | prep p90 | prep max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A online→cached | cat | online | 125.3 | 97.17 | 69.28 | 0.555 | 6.522 | 8.566 | 8.566 | 0.139 | 3.578 | 4.614 |
| A | cat | cached | 81.4 | 53.77 | 38.24 | 0.521 | 3.424 | 7.539 | 7.539 | 0.014 | 1.784 | 2.137 |
| A | gate | online | 125.4 | 97.01 | 68.72 | 0.541 | 8.497 | 11.086 | 11.086 | 0.012 | 2.898 | 5.582 |
| A | gate | cached | 81.2 | 53.05 | 36.64 | 0.571 | 3.430 | 3.475 | 3.475 | 0.002 | 2.422 | 2.905 |
| A | xattn | online | 124.9 | 97.67 | 71.23 | 0.535 | 8.376 | 11.267 | 11.267 | 0.012 | 2.438 | 5.569 |
| A | xattn | cached | 80.7 | 52.88 | 36.07 | 0.551 | 3.246 | 4.477 | 4.477 | 0.001 | 1.125 | 2.041 |
| **B cached→online** | cat | cached | 82.0 | **54.04** | **37.71** | 0.469 | **3.775** | 6.031 | 6.031 | 0.012 | 1.862 | 4.328 |
| B | cat | online | 125.7 | 97.93 | 70.97 | 1.513 | 5.954 | 8.869 | 8.869 | 0.092 | 4.151 | 5.475 |
| B | gate | cached | 81.2 | **53.96** | **36.21** | 1.053 | **3.264** | 4.443 | 4.443 | 0.001 | 2.922 | 3.942 |
| B | gate | online | 127.1 | 99.01 | 72.45 | 0.460 | 9.032 | 12.931 | 12.931 | 0.012 | 2.745 | 5.524 |
| B | xattn | cached | 81.0 | **52.63** | **36.74** | 0.543 | **3.350** | 5.193 | 5.193 | 0.002 | 1.103 | 2.291 |
| B | xattn | online | 124.0 | 96.53 | 68.71 | 0.565 | 8.225 | 10.811 | 10.811 | 0.010 | 4.118 | 5.437 |

speedup（online/cached）：A 轮 wall 1.541/1.544/1.547、full 1.807/1.829/1.847、steady 1.812/1.876/1.975、p90 1.905/2.477/2.580；B 轮 wall 1.533/1.565/1.531、full **1.812/1.835/1.834**、steady **1.882/2.001/1.870**、p90 1.577/2.767/2.455。

**六、修订 gate 逐条判定（两种顺序）**

| arm | order | full cached<online | steady cached<online | steady p90 cached≤online | equivalence | 该顺序成立 |
| --- | --- | --- | --- | --- | --- | --- |
| cat | A | ✔ (53.77<97.17) | ✔ (38.24<69.28) | ✔ (3.424≤6.522) | PASS | **是** |
| cat | B | ✔ (54.04<97.93) | ✔ (37.71<70.97) | ✔ (3.775≤5.954) | PASS | **是** |
| gate | A | ✔ (53.05<97.01) | ✔ (36.64<68.72) | ✔ (3.430≤8.497) | PASS | **是** |
| gate | B | ✔ (53.96<99.01) | ✔ (36.21<72.45) | ✔ (3.264≤9.032) | PASS | **是** |
| xattn | A | ✔ (52.88<97.67) | ✔ (36.07<71.23) | ✔ (3.246≤8.376) | PASS | **是** |
| xattn | B | ✔ (52.63<96.53) | ✔ (36.74<68.71) | ✔ (3.350≤8.225) | PASS | **是** |

**判定：`cached_pretraining_path = QUALIFIED`**（三个 arm、两种运行顺序均满足修订 gate 的四项条件；完整 5.04M cache 可作为 CAT/GATE/XATTN 正式预训练 production path）。

median 仅报告、不作 gate，本轮数据也给出直接理由：同 arm 同 mode 的 steady median 在两种顺序间最多漂移 **+173%**（cat online 0.555→1.513）、**+84%**（gate cached 0.571→1.053），而同一对照下 full 变化 ≤2.1%、steady ≤5.4%、p90 ≤10.2%；xattn 的 median 条件在 A 轮 false、B 轮 true。旧 gate 判定（§13.15）与此不冲突且**保持不变**。

**七、预算与未执行**

180 updates（6×30，performance validation，不计入 P2 预算）；正式预训练/微调 **0**；CAT/GATE/XATTN 5000-step、downstream、outer-test、P3 **均未启动**。benchmark 产物全部位于 `results/mcl_ph_20260921/p2r2_perf_bench{,_rev}/`（`results/` 按 `.gitignore` 不提交）；`results/mcl_ph_20260921/p2/pretrain/` 与 cache 未被写入。未做任何模型优劣或科学结论：本轮只回答「cache 是否降低真实训练 wall/累计训练时间且不恶化 p90」。

**八、提交与同步**：本轮证据脚本与本节见随后 commit（基线 `954310c`）；不自动启动 CAT/GATE/XATTN 5000-step、downstream、P3、outer-test。
