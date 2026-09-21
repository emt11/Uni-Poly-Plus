# MCL-PH：多尺度距离专家与 PH 路由替换 GLT 3D 通道

计划 ID：`MCL-PH-20260921-01` ｜ 修订：r1 ｜ 日期：2026-09-21 UTC

## 0. 状态、角色与授权

- **状态：待授权执行**。本轮只获授权制定本文件；实现、模型验证、数据派生和训练尚未启动。
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

## 13. 执行记录与审查区

ZCode在每阶段后追加：实际基准/变更、命令和tmux window、结果与失败、预算消耗、产物/日志、未完成项、待审查问题。不得擦除失败或以新跑覆盖旧跑。Codex追加逐项验收与下一步；当前均为空，尚无新模型验证或预测结果。
