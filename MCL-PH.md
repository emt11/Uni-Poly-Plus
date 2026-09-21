# MCL-PH：多尺度距离专家与 PH 路由替换 GLT 3D 通道

计划 ID：`MCL-PH-20260921-01` ｜ 修订：r2 ｜ 日期：2026-09-21 UTC

## 0. 状态、角色与授权

- **状态：受阻 / 待审查（r5 部分交付）**。r3 第二阶段在 cat 臂导出阶段挂起后按停止条件中止（详见 §13.3 与 [MCL-PH-INCIDENT-r3-cat-export-hang.md](MCL-PH-INCIDENT-r3-cat-export-hang.md)）。r4 在不恢复训练的前提下完成：checkpoint 核验（PASS）、CPU 离线导出（PASS，产物仅为「恢复导出候选」）、四 rank GPU 收尾复现（**未复现**），**根因仍未定位**；同时补齐 Router 诊断开关与内存记录（§13.4）。r5 完成四项收尾状态修正与针对性验证（14 passed），但**唯一一次** cat 两步真实运行因我方预建输出目录触发防覆盖守卫而在预备阶段中止（0 update、未重试），故「真实两步复现」目标未完成，详见 §13.5。
- 当前完成度：P0 已完成（r1 定版审计 + r2 受限修订，判定缺陷见 §13.3）；P1 **仍未完成**（cat 完成 2/6 授权 update 但无 r3 自己的 `deploy_00002.pt`，M_GATE/M_XATTN 与三条微调未运行，五臂验收仅 4/5 PARTIAL）；P2/P3 未授权。r3 的范围、预算、停止条件与实际基准见 §13.3，r4 见 §13.4，r5 见 §13.5。
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

