# PH 融合与 GLT 微调提升：端到端候选计划

## 0. 交接头、状态与授权

| 项目 | 锁定内容 |
| --- | --- |
| 计划 ID | `GLT-PH-END2END-20260920-01 / r1` |
| 规划日期／角色 | 2026-09-20；Codex 规划与审查，ZCode 后续执行 |
| 状态 | **计划已编写，实施与实验待授权；尚无候选性能结果** |
| 本轮授权 | 用户要求在 `PH.md` 制定完整方案；仅文档修改、检查、提交与推送 |
| 目标 | 提升真实下游微调泛化表现，重点观察 XC；不以 PH loss、门控幅度、重建精度代替属性评估 |
| 编写基线 | `dev@fd6378f`；已 `git pull --ff-only origin dev`，Already up to date |
| 已有用户改动 | `Plan.md` 删除、`.zcodeignore` 未跟踪、`PH.md` 空文件未跟踪；本轮只编辑／提交 `PH.md`，不恢复或提交用户删除 |
| 历史边界 | PHRETENTION r6 已以限定负结果结束；不重跑其零初始化 late-residual 实验，不覆盖旧缓存、checkpoint 和报告 |

所有路径中的拟新增文件、stage 和参数均为**待实现接口合同**，不能把本文命令名当作现有 CLI。本文给出三条可独立实施的端到端路线，不要求一次全部运行。默认推荐顺序 **B → A → C**；每个阶段结束交回审查，不自动消耗后续预算。

## 1. 文献依据与科学问题

检索窗口约为 2024-09 至 2026-09。以下只借鉴明确机制，所有项目改造均需本项目验证。

| 依据 | 原论文机制 | 本项目借鉴／不继承的结论 |
| --- | --- | --- |
| [PiPE，ICML 2025](https://proceedings.mlr.press/v267/verma25b.html)，[方法正文](https://arxiv.org/html/2506.05814v1) | 位置状态、学习过滤得到的 PH、节点／边拓扑特征参与逐层更新 | A/C 借鉴中间层拓扑条件化；本文不实现动态可微 PH，不称完整 PiPE，不继承其表达能力定理 |
| [MI-MoE，2026 预印本](https://arxiv.org/html/2601.12637v1) | 不同距离尺度的专家与包含 PH 的拓扑路由；有分子及聚合物任务 | B 借鉴空间尺度路由；采用共享小模块而非复制完整专家；论文整体收益不是 PH 单因素证据 |
| [MCP，Briefings in Bioinformatics 2024](https://academic.oup.com/bib/article/25/6/bbae465/7774896) | 元素选择、多覆盖持久性、barcode 统计与树模型 | C 借鉴化学选择与拓扑粒度；本文仍为普通 VR PH，不恢复完整 MCP，不把多半径叫多覆盖阶数 |
| [Mol-TDL，2024 预印本](https://arxiv.org/html/2410.04765v1)，[2026 ACS Nano 版本](https://doi.org/10.1021/acsnano.5c11744) | 多尺度单纯复形消息传递与预训练 | B/C 借鉴非键合空间交互；不把单纯复形消息传递的收益全部归因于 PH，不照搬绝对坐标特征 |
| [DPD＋PH，Digital Discovery 2025](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00376d) | 从聚合物微相分离模拟提取 PH 表示并分析性质关联 | 说明输入尺度重要；单 Trimer 没有介观形态、链间堆积与加工历史，不能据此许诺 XC 提升 |

### 1.1 已有结果如何约束新计划

本轮静态核对 `results/glt_galph_ph_retention_20260920/p4/development_aggregate.json`：REAL−OFF 约 `5.0367e-6`，REAL−CONST 约 `1.1396e-7`，旧门控残差实验应停止。这个结果没有证明 PH 在中间层或尺度路由中无效。

当前代码事实：

- `src/modules/glt_galformer_ph.py`：O8 与 GLT 均六层；GALPH CLS 路线在各层更新全 Trimer 物理键与 CLS，PH 在编码后加入 summary。**不能把基类的 center-only readout 当成当前 GALPH 实际读出。**
- `src/modules/glt_galformer_ph_pretrain.py`：2D 原子类别 CE、3D 键类别 CE、全局 InfoNCE、PH patch 重建；3D CE 不是坐标去噪。
- `src/modules/glt_galformer_ph_downstream.py`：各路 `[CLS; mean]→512` 后门控融合；微调 readout/head 是新参数。
- `src/dataset/glt_ph.py`：冻结 open Trimer 的 `[3,32]` Betti profile，不是逐原子拓扑编码。

**三条新假设**：A 检验 PH 提前进入 token 更新；B 检验 PH 能否选择有效空间尺度；C 检验原子空间主干与局部化学敏感 PH 是否比键主干更适合目标。不同路线的整体差值不是 PH 的纯因果贡献；每条路线内部必须有匹配对照。

## 2. 共享输入、数据与身份合同

### 2.1 冻结来源

1. 只读现有 RU／Topology／Trimer、dual-static 和 PH 缓存，不调用构象生成、MMFF、坐标优化、MD，不写旧 LMDB 或已发布 sidecar。
2. 预训练参考划分：`results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json`，train 为 P_train、validation 为 P_val；执行端读取实际字段，核对 source/cohort/sample identity。缺失或无法解释的身份冲突停止，不自行生成替代划分。
3. 核对 benchmark 身份与 P_train 的排除情况、common-init 来源。若仍有已知重叠，记录局限；若发现与既有声明不符，先交回，不能静默改 cohort 后仍声称历史 matched。
4. 三路线学生均消费相同预训练样本，不按新增 PH 成功与否筛选学生。2D-only 对照也用这一 cohort，承认它有历史几何成功选择偏差。
5. 新 PH／邻域产物独立于旧缓存；复用现有 key 读取及版本检查，不另建通用缓存框架。所有 32-byte key 按原始 bytes 比较，不能用会截断 NUL 的字符串语义。
6. PH 常量和标准化统计只拟合 P_train，不拟合 P_val 或下游全集。下游标签 scaler 只拟合对应最终 train。

### 2.2 统一 batch 合同（单图示意）

```text
原 O8 全部字段                    原样保留；N 个真实 RU 原子，512 维编码
sample_key                       原始 32-byte key
polymer_identity                 既有稳定身份
pos                              float [A,3]，冻结 Trimer 的实际全原子坐标，Å
z                                int64 [A]，与 pos 对齐，包含显式氢
atom_ru_offset                   int64 [A]，已证实身份 -1/0/+1；端帽另行标记
atom_to_canonical                int64 [A]，无映射者 -1，不猜测氢／端帽身份
physical_bond_index              int64 [2,B]，无向实际键，一条键仅一行
physical_bond_type               int64 [B]
physical_bond_to_state           int64 [B]，与既有 GLT bond rows 对齐
bond_center                     bool [B]，只供审计，不改变本轮全体读出
geometry_valid                  bool
global_profile                  float [3,32]；A/B 使用
global_profile_valid            bool
local_profile                   float [A,4,32]；仅 C 使用
local_channel_present           bool [A,4]；仅 C 使用
spatial_edge_index[s]            int64 [2,Rs]，无 self、双向、逐图
graph_ptr / atom_ptr / bond_ptr  packed batch 分界
```

执行端必须从真实 adapter 获取原子与坐标的对应关系（包括已发生过的 `_topology_z` 问题），不能凭相同长度推断。A 不需要新增空间边，但共用身份检查。A/B/C 均不假定 `A=3N`，因为全原子、显式氢、端帽可能存在。

### 2.3 Trimer 与单 RU 的分工

- O8：保持现有单 RU 化学表示和周期连接实现，不改输入、SPD/path、参数名。
- 3D：始终读取实际 open Trimer。A/B 维护 B 个实际键状态；C 维护 A 个实际原子状态。左右 RU **都更新**，不复制中心坐标、不强制共享状态。
- 下游：两路各压缩成 512 维图表示，再按 §7 融合，不要求 Trimer 与 RU token 数相等。
- 本轮不同时验证中心-only、N+1/N+2 或 canonical state sharing；那属于另一个数据语义对照，不自动重启已取消的 CANON3D。
- 几何无效：在 encoder 之前分组跳过，不对 NaN 乘零；`r3=0`，2D 属性任务照常。PH 单独无效：使用对应常量输入并记录有效率，不能改变三臂的任务样本集合。
- 中心内部键 N=0 与整个 Trimer 无键不是同一概念。A/B 沿用真实物理键；无真实键则 3D 无效。C 有原子即可编码；没有中心键不产生中心键监督。本轮 3D CE 明确监督实际可用物理键，与旧中心-only 蒸馏合同不同。

## 3. PH 与非 PH 对照输入

### 3.1 A/B：复用现有 global profile

使用 `glt-ph-betti-v2` 的三个通道、32 个已定义采样半径和归一化实现，不重新发明采样位置。实际半径从代码读取并固化配置，不用 `linspace` 的另一种端点约定代替。输入为 `P:[G,3,32]`。

每个 family 固定三臂，三臂都实例化同样的网络：

| 后缀 | 输入 | 归因 |
| --- | --- | --- |
| `CONST` | P_train PH profile 的逐元素均值，所有样本相同 | 新容量＋新架构，但无样本特异 PH |
| `PH` | 当前样本真实 PH | 相对 CONST 测 PH 条件化 |
| `STAT` | 下述普通空间统计，同样 `[3,32]` | 检查收益是否只是一般距离／密度摘要 |

STAT 在同一半径 r 计算三个通道：①重原子无序对的距离 CDF；②全原子无序对的距离 CDF；③重原子 cutoff graph 的度方差除以 `max(1,n_heavy-1)^2`。不足两个点时相应通道为零；度方差使用总体方差。CDF 分母为实际无序对数，不包含 self。

各输入在进入网络前使用同一套 **P_train PH** 每元素均值和总体标准差（std 下限 `1e-3`），三臂同变换；STAT 不裁剪，记录范围。这使 CONST 为标准化零输入，保留非零网络 bias／尺度编码。STAT 与 PH 分布不同必须在报告中披露，不能仅靠 STAT 失败声称数学上“PH 不可替代”。只对有效PH记录拟合统计；缺失PH仍保留训练样本并用CONST回退。P1可使用256条P_train的临时统计做smoke，必须标记SMOKE_ONLY，不能流入P2；P2从共同初始状态开始，不能续训P1模型。

### 3.2 A/B 共用 profile encoder

```text
按半径连续四点分 patch：[G,3,32] → [G,8,12]
content = MLP(12→128→128, GELU)
token = LayerNorm(content + learned_scale_embedding[8,128])
两层 Pre-LN Transformer：hidden128 / heads4 / FFN512 / dropout0.1
输出 T=[G,8,128]，p=mean(T,scale)=[G,128]
```

投影按末维执行，尺度位置保留。所有 PH 网络从共同的新初始化训练；不冻结旧 C1 PH encoder，不加载旧 `alpha_ph/ph_to_summary/ph_head`。MLP bias 初始化 0，Linear weight Xavier uniform，尺度 embedding normal std0.02；随机流独立于公共 O8/GLT。

### 3.3 C：局部元素选择 PH（独立方案，不与 A/B 混用）

对每个实际原子 a，以其坐标为中心选取 `distance(a,u)≤6Å` 的实际点集，不限制 canonical/RU；包括中心点。计算四条曲线：局部 heavy H0、heavy H1、`{C,N,O}` H0、`{N,O}` H0，32 半径与 §3.1 一致。使用完整 VR 2-skeleton 计算 H1，不能把普通图 cycle rank 当 VR H1；系数域固定 F2。计数除以对应点数；未死亡条目按 right-censored 持續到 cutoff，不能伪造有限 death；空集合曲线0且 presence=false，单点按 H0 定义处理。

CONST 为 P_train 所有实际原子等权平均的 `[4,32]` 和四维 presence 均值，广播给每个原子；PH 为真实曲线和 presence；STAT 为四个对应点集的无序对距离 CDF、presence 不变。presence 本身提供化学组成线索，PH−STAT 更能控制这一因素。输入标准化方式与 §3.1 同义，统计对象改为 local PH。

局部邻域的点集选择截断与半径 PH 是两个不同操作；不得称相对持久同调或无限链 PH。不通过某个 cycle representative 将 PH 归属给任意编号原子，局部特征归属由球心确定。C 的局部 PH 成本必须先在 256 条 P_train 样本上实测，报告外推时间与空间，再申请全量构建；不允许边训练边调用 GUDHI。C仅曲线做上述标准化，presence保持0/1或CONST的均值，不重复标准化。旧缓存采样半径必须核对数值，local VR和STAT统一使用同一半径数组。

## 4. 方案 A：Layer-PH Cross-Attention GLT

**问题：不改变现有 GLT 关系图，仅将 PH 提前到 token 更新，是否改善微调？**

### 4.1 架构与输入输出

- 以 `GLTGalPH(summary_mode='cls', ph_mode=None)` 的 O8、GLT、CLS 数学为骨架，保留六层、hidden512、heads8。
- 完成 GLT 第2层及第4层后，各插入一个独立 cross-attention block。query 为同图所有物理 bond states 与 CLS3，key/value 为该图的八个 PH tokens，禁止跨 packed graph。
- Q：`LN(H)→Linear512→512`；K/V：`LN(T)→Linear128→512`；8 heads，head_dim64，缩放 `1/sqrt64`，输出投影512→512。
- 更新 `H'=H+0.1*Dropout(CrossAttention)`，然后 `H_next=H'+Dropout(FFN(LN(H')))`，FFN512→1024→512。0.1 为固定系数，不可学习，不加开门正则。
- CONST/PH/STAT 全部执行相同 cross-attention 和 FFN，无 bypass；CONST 仍含可学习尺度位置，不是原 GLT parity。
- 输出 O8 atom states `[N,512]`、实际 bond states `[B,512]`、CLS2/CLS3 `[G,512]`。使用共同 masked-token heads、共同下游读出。

### 4.2 预训练、微调与限制

严格使用 §6 的三项主损失；PH 不再作为默认重建目标。PH 投影与 GLT 一起训练，微调继续更新，不新增 PH readout 残差。A 对 R0 的差值包含额外层与容量；只有 A_PH−A_CONST 是相同结构下的信息对照。即使有效，也不能称完整 PiPE。

## 5. 方案 B：PH-Routed Multiscale GLT（首先执行）

**问题：给 GLT 增加真实空间消息后，PH 能否提供超出固定尺度组合与普通空间统计的收益？**

### 5.1 空间关系

取三个嵌套半径 `2.5 / 4.0 / 6.0 Å`，包括所有非 self 的真实全原子对，双向存储、不过滤化学键、不加 TopK、不把三者误做互斥壳层。坐标不变；边几何为 32 Gaussian RBF：中心0..6Å等距、宽度中心间距，另加真实 bonded bit。半径终点采用 `d≤r`；运行时浮点容差测试与临界关系重建测试分开。

所有尺度按相同距离规范构建；原有化学 line-path 完全保留。显存超限允许按 query 精确分块，不能静默删边或改半径。

### 5.2 插入 GLT 第3层与第5层之后

每个插入模块有独立参数；同一模块三个尺度共享 message MLP，另有三个尺度 embedding128：

1. 从旧 bond states 对 incident bonds 做原子均值，空 incident 返回零512；Linear512→128、LN，得到临时 `U:[A,128]`。**不另外输入 clean 元素类别**，避免绕过 masked bond token。
2. 每条空间边 j→i，`MLP([U_i,U_j,RBF(d_ij),bonded_bit,scale_emb_s])`：输入417→256→128，GELU。按接收原子均值聚合，空邻域为零；三个尺度消息分别经过同一无 affine LN。
3. `router_l(p)=Linear128→128→3(GELU)`，softmax 得每图三权重，不使用 TopK。最后一层 weight/bias 初始化0，使初始权重1/3；前层第一步梯度可能为0，属于合法初始化，不要求所有参数首步非零。
4. 原子更新消息 `M_i=sum_s π_s M_i,s`。对物理键(a,b)用 `M_a+M_b`，Linear128→512，乘固定0.1后加到原 bond states；不区分端点顺序。该模块不改 CLS，后续原 GLT 层让 CLS 消费更新后的键状态。
5. 不增加第二个 FFN；使用后续 GLT block。第5层后仍有第6层，因此两次空间更新均进入 CLS。

### 5.3 路由与输出合同

CONST 为样本无关但可学习的尺度组合；PH 是真实拓扑条件路由；STAT 是一般空间描述符路由。三者共同初始化、相同边和参数量。各层记录平均路由、跨样本方差、三个消息范数及其余弦相似度；路由不变化不是实现错误的充分条件，也不是自动调整正则的理由。

输出与 A/R0 完全相同。预训练 §6、微调 §7；不在 readout 再加入 PH。该方案不模拟真实势能或链间相互作用，不声称所有小于6Å的边都有同等物理作用。

## 6. 方案 C 与共享预训练合同

### 6.1 C：Local-PH Atom Spatial Transformer，替换 3D 键主干

**问题：让几何主干直接维护原子空间状态，并把局部 PH 作为逐层结构条件，是否更适合下游？**

此路线不同于 A/B，仅在 B/A 无明确候选或用户选择替代主干时进入；不能将多个改动的总体收益归因于某一因素。保留旧GLT的初始bond embedding及mask操作作为输入适配，不实例化其六层line Transformer或角度bias；不能把废弃的完整GLT留在optimizer里虚增参数。

输入全 Trimer 原子，初始 atom256 只由 **masked/replaced 的既有 bond embedding512 的 incident mean →Linear512→256** 得到；无 incident 使用 learned empty256。避免直接读取被遮蔽的 clean z。化学身份通过已有键 embedding 进入，局部 PH 描述的可见化学组成属于已声明条件输入。

空间边为 `d≤6Å` 的实际全原子双向图，另加 CLS3 与所有原子双向边及 CLS self；原子自环由残差承担。六层 Pre-LN Transformer，hidden256、heads8、head_dim32、FFN1024、dropout0.1。距离32 RBF 和 bonded bit 经MLP33→64→8产生每头 bias；CLS边用三类 learned bias。普通节点内容 logits 固定 `Q_target·K_source/sqrt32`，按 target 全邻域归一化；不改 O8 的旧 attention。

局部 PH `[4,32]` 每四个半径为一个patch，得到8×16；拼接该原子四维 presence 后每patch20→64→64，加入尺度embedding，均值→LN 得 `p_i64`。两层共享全图的局部编码器参数，不运行动态PH。第2、4层之前，逐原子 `Linear320→256([LN(H_i),p_i])`，固定0.1残差到 H_i；CLS不直接读局部PH，靠后续attention汇总。所有层使用同一 p_i，两个融合Linear独立。此为 PiPE-inspired 局部结构条件化，**不是可学习过滤版 PiPE**。

输出：atom3 `[A,512]` 与 CLS3 `[G,512]` 由共同 Linear256→512得到；对每个真实键(a,b)，用 `[u_a+u_b,abs(u_a-u_b),u_a*u_b]`（1536维）经MLP1536→512→512形成 `bond_states`，供与 A/B 相同的键 CE。下游3D mean 使用所有真实 atom3，不用生成的bond_states；此读出差异必须记录。O8仍输出 `[N,512]`。

### 6.2 主实验预训练任务（所有候选统一）

```text
L = L_atom2D + L_bond3D + L_CL
L_PH = 0；L_FP = 0；无额外坐标去噪、torsion、ENV、FGR
```

- 原 O8 masked atom CE101 与物理 bond类别 CE25755。直接复用当前 GALPH 的 40% 选择、80% MASK／10% REPLACE／10% KEEP 及 donor、关系mask实现，不重写另一套采样。
- A/B 的键 CE 消费融合后的实际 bond states；C 消费 §6.1 对称键 decoder。
- CL 复用当前跨rank multi-positive InfoNCE，temperature0.1，输入为 CLS2/CLS3 各经512→256→128。不把 PH 当第三路对比正例。
- 每项按当前累积窗口的全局有效计数归一化；token CE 与图级 CL 分母不同，不平均 microbatch 均值。空目标 rank 仍参加 collectives 和同步 backward；全局无目标时返回有限图零。
- PH 输入使用完整 clean frozen profile，不做 PH patch 遮蔽。它是**有意开放的几何／局部组成条件**，不宣称 chemical-token prediction 无捷径。A/B/C 的 STAT/CONST 只改变对应输入，不改变 mask 或目标。
- 对 C 新路径逐一检查 masked bond 的 clean endpoints 没有从 atom initialization、bias 或额外 z embedding 重新注入；PH 局部组成及空间关系本身属于显式条件，不把该条件掩饰为无泄漏去噪。
- 当前独立物理副本 mask 可能允许跨 RU 化学复制；主轮保持历史定义以控制变量，报告这一局限，不在本轮静默改 orbit mask。

### 6.3 初始化与训练数值

参考 `configs/mts/glt_galph_c1.json` 与现有 common init，但新的主实验不加载旧已训练 C1：

| 项目 | 主轮锁定值 |
| --- | --- |
| 公共初始张量 | `results/glt_galph_20260920/p1/common_init_galph_v1.pt` 中 O8、原 GLT、CLS、兼容heads；按名称复制，列出使用／不用的键，不能 strict=False 静默跳过 |
| A/B/R0 | 共享全部兼容公共张量；不实例化旧 PH summary residual |
| C | 只加载相同 O8、CLS2及兼容2D head/CL投影；新3D用独立 seed20260922；三臂新张量逐位相同 |
| 新模块初始化 | §3定义；family新模块使用隔离seed：A20260923、B20260924、C20260922 |
| 预训练数据随机流 | seed42；样本、mask、donor、公共dropout流一致；额外模块dropout用独立generator／局部RNG状态，不能扰动公共流 |
| 优化器 | **AdamW**，lr2e-4；矩阵wd1e-6，bias／normalization／所有PH编码及融合router参数wd0；参数集合显式记录 |
| 调度 | warmup2000，cosine总长20000，end_lr1e-9；5k只是同一20k schedule的截断，不重缩调度 |
| batch | world4、每卡micro84、accum3，global1008；BF16；clip1.0，unscale后clip |
| OOM政策 | 四卡资源不足先停；若需micro42×accum6，必须整组三臂统一登记后重新开始，不静默只改一臂 |
| 保存 | 256及1000/2000/3000/4000/5000；5k固定导出，不按下游挑预训练步数 |

使用 AdamW 是为新实验建立清晰衰减合同，并非声称旧 Adam 是全部失败原因。因此 R0 也要按同一新配方重训；不能拿旧 C1 的分数作为严格 matched baseline。预训练选型不访问 outer-test，P_val 只用于诊断，主轮固定5k。

### 6.4 两个训练参考与当前部署锚点

- `R0`：当前 CLS双路 GLT，ph_mode=None，使用同一三损失与新优化器。是“新配方下原架构”参考，不是旧 C1 权重重命名。
- `R2D`：同一O8／CLS2、masked atom CE；不实例化GLT、PH或CL，推理使用 §7 的2D_ONLY分支。与双路不同训练任务是其定义的一部分，不能把差值只归因于3D推理输入。
- 另允许对同一候选部署做零训练 `DUAL vs 2D_ONLY` 诊断，但这是推理分支干预，不等价于从头训练的2D模型，不能用其下降证明3D一定有泛化价值。
- **当前部署锚点 `CURRENT`**：`configs/mts/glt_galph_c1_repair_5k_identity.json` 指定的修复C1 5k（sha256 `3063cf8bef341841b665c93f118a1eb251308d5d67dfa41b4179465570f302e2`），使用原r6 F_OFF架构与本计划相同的微调配置重新微调。其旧PH模块不进入属性路径且冻结，不新增预训练。CURRENT不属于matched-pretraining机制对照，而是回答“是否超过当前模型”的部署锚点。旧r6分数只作历史描述，不能替代这一同微调协议比较。

## 7. 共享微调：最终要优化的目标

### 7.1 固定读出与训练

A/B/R0：`r2=Linear1024→512([CLS2,mean O8 atoms])`，`r3=Linear1024→512([CLS3,mean physical bonds])`。C 的3D mean改为atom3均值；其他不变。R2D 只构造r2及属性head。

```text
g = sigmoid(MLP1024→256→1([r2,r3]))
z = LN(r2 + geometry_valid * g * Linear512→512(r3))
y = MLP512→256→1(z)，GELU、dropout0.1
R2D：z = LN(r2)
```

PH encoder、空间模块、融合模块在预训练和微调均存在且继续训练；不另加 `gamma·PH`，不冻结 PH，不更换CONST/PH/STAT身份。训练only CE/CL heads从属性优化器剔除。共享新readout/head从同一fold初始张量复制，不只依赖“seed相同”。

| 项目 | 锁定值 |
| --- | --- |
| 划分 | 已固定 `outer5_inner20`，读取三个真实索引；禁止 val=test；P0从r6各unit run.json解析实际split_root并校验，唯一匹配后写入新config，不能自行重建或选另一个同名目录 |
| 探索集合 | xc、eps、eat × folds0/1；仅development，outer-test不加载进预测循环 |
| 标签 | final train-only StandardScaler；不按全任务拟合、不删几何失败行 |
| 损失 | 标准化标签上的 Huber，delta0.5，每任务独立训练 |
| optimizer | AdamW；所有预训练encoder含PH／spatial lr1e-5；新readout／fusion／属性head lr1e-4 |
| weight decay | 矩阵0.02，bias／normalization／PH encoder及其融合router参数0；同family全臂一致 |
| batch／eval | 32／64；dropout0.1；clip1.0；训练FP32，禁止只给某组开启AMP |
| seed | 基础42；模型及DataLoader seed=基础seed+fold_id；确认seed见§9 |
| 探索 | 最多30epochs、warmup5、patience10；schedule总长30，validation R²最大选best，平局取更早epoch |
| 确认／正式 | 最多100epochs、warmup5、patience10，schedule总长100；从部署checkpoint重新微调，不能续接30epoch模型混排；八任务固定eat/eea/egb/egc/ei/eps/nc/xc |

每个task/fold独立best，先恢复best再保存validation预测。探索及确认只访问validation。正式阶段才恢复best后执行一次test，不做train+val refit。best在最后epoch只记预算敏感性，不自动追加epoch。

### 7.2 XC 的预登记判读

1. XC作为单独主要任务，同时报告EPS/EAT保护指标；三任务宏平均不能掩盖XC退化。
2. 不根据XC表现重新挑PH半径、原子子集、cache内容或预训练checkpoint；这些修改需新修订。
3. 已有数据与split曾参与开发，不能称独立盲测。确认seed仅衡量训练随机性，不把重复seed当新增独立样本。
4. 收集train/validation loss、每折R²/MAE/RMSE、best_epoch、预测方差、3D融合比例；小门控仅为诊断，不强迫其变大。
5. 若标签近常数造成R²不定义，应保留原因、以非有限指标停止该单元验收，不能截断成0再算宏平均。

## 8. 科学比较、晋级与停止

每个family首先只跑 `CONST / STAT / PH`；R0、R2D、CURRENT同一已验证run可跨family复用。A/B/C之间参数量和读出不完全匹配，只比较完整路线，不做单机制归因。CURRENT不进入新PH组的初始化，其冻结旧PH参数与原r6保持一致。

定义 `macro3` 为三个任务各自两折R²均值再平均；gain均为候选减参考，MAE/RMSE方向相反。

**探索推进条件（只允许称探索候选）**：

- PH 相对本family CONST 与 R0：macro3分别至少 +0.005；XC两折均值分别至少 +0.01，且XC两折差值对这两个参考都为正；
- PH相对STAT：macro3>0，XC均值>0；
- 相对CONST与R0，任一任务两折均值下降不得超过0.01；相对CURRENT，macro3和XC均须为正，不能只超过新建弱参考；
- 无身份／finite／预算违规。阈值是工程筛选，不是显著性检验。

不满足则该family不自动长训；如CONST已改善而PH没有，记录“结构候选、PH未建立”，允许将CONST列作最终非PH候选建议，不能强行保留PH。微小正值或混合方向为INCONCLUSIVE，不通过自动换seed反复寻找正值。

多个family通过时，先按 `min(XC_PH−XC_CONST, XC_PH−XC_R0)` 排名；差小于0.002时按macro3同类最小差，再按推理耗时较低者，最后固定B>A>C。最多晋级一个family，不能合并三种机制另造未经验证模型。

**确认条件**：§9确认集合上，对CONST、R0和CURRENT同时满足macro3≥+0.005、XC≥+0.01，XC两个fold分别跨seed均值为正；对STAT的增量必须已在探索报告中明确，正式阶段不再做“PH独有信息”更强主张。报告所有seed及不确定性，不宣称统计独立。正式报告必须同时给R2D比较；若PH不优于R2D，不宣称复杂度值得。候选20k与CURRENT5k的比较只能说明完整方案增量，不是等预训练成本的PH归因；等成本归因使用R0和CONST。

## 9. 分阶段执行与预算（全部实施预算待授权）

本轮只有文档预算。用户后续可只授权某一阶段；未用额度不自动转为额外组／seed／epoch，失败与重试实际消耗都计入。

| 阶段 | 明确范围 | 上限与停止点 |
| --- | --- | --- |
| P0 来源／接口审计 | 不训练；核对真实代码、数据、split、common-init、活动进程；A/B描述符256样本，C仅在选中时另256样本局部PH测算 | 每family最多256个P_train样本＋下游每任务16条只读几何；不读outer-test标签；产出resolved路径、统计拟合来源、预计缓存成本后交审查 |
| P1 实现与最小验证 | 只实现获选family和R0/R2D公共路径；fixture测试、真实smoke、保存／恢复 | 每个新增arm：2updates forward/backward＋连续4与2后恢复2的比较共8updates，总10updates；微调xc/fold0每arm1epoch；报告路径先0update验证。DDP partial/all-empty各一次backward、0update |
| P2-B 首批预训练 | R0、R2D、B_CONST、B_STAT、B_PH，共5条 | 每条5000updates，总25000；先到256审查可训练性，256计入5000；不通过不继续 |
| P3-B 探索微调 | 上述5臂＋CURRENT，共6臂×3任务×2fold | 36units，每unit≤30epochs，总≤1080；全为development smoke级筛查，不能作正式性能结论 |
| P2/P3-A 条件分支 | 只有用户选择A且明确授权时：A三臂独立同配方训练与探索 | 15000updates＋18units/≤540epochs；R0/R2D复用，不重训；不得由B失败自动启动 |
| P2/P3-C 条件分支 | 只有局部PH成本审查及用户授权通过后：C三臂 | 15000updates＋18units/≤540epochs；禁止自动全量构建局部PH |
| P4 长程预训练 | 最优PH、其CONST、R0、R2D共4条，从各自5000完整resume继续到20000 | 新增≤60000updates；全部共用原20k schedule，不能只延长PH；如果无family通过则不启动 |
| P5 稳定性确认 | 四条20k部署＋CURRENT5k，共5臂×xc/eps/eat×fold0/1×finetune基础seed42/43/44 | 90units，每unit≤100epochs，总≤9000；预训练仍seed42，不谎称三次预训练重复；只validation |
| P6 正式复评 | P5通过且用户单独授权；上述5条冻结部署×8任务×5fold×finetune seed42 | 200units，每unit≤100epochs，总≤20000；每样本每臂一次outer-test，OOF与fold mean±std(ddof0)分开 |

P1连续／恢复预算说明：连续4＋分段2＋恢复2＝8，再加独立smoke2＝10；若复用同一跑出的前2步，仍不得改写实际账目。全部11条新训练臂（两参考＋三family各三臂）与一个CURRENT的理论上限为55000预训练updates、72探索units/2160epochs；P1新臂至多110updates/11epochs，CURRENT只加一次1epoch微调smoke（P1微调总上限12epochs），不重训CURRENT；**这不是一键执行全部分支的授权**。

生产特征构建与统计另记CPU预算：P0的256条仅用于验收和测算，不充当P_train总体统计。P2开始前，A/B需要对固定P_train拟合PH统计、生成STAT，并准备所需下游无标签输入；C还需全量local PH。先提交P0外推的时间／空间、样本清单和已有可复用产物，由用户明确授权这一离线构建阶段；未授权不能把全量构建隐藏为训练前处理。phase授权记录必须同时写明“训练”与“所需离线特征构建”是否包含。局部PH单样本失败保留原因，PH臂退回CONST且同臂valid率报告；出现映射错误则不是可回退的普通PH失败。

256步检查固定使用16条P_train probe，在step0/1/16/64/128/256记录有限性、真实／常量输入差、输出差、新模块梯度与更新量、空间边数和资源。全程使用固定eval FP32探针，检查前后还原RNG／模式／输入；若真实与常量在所有已检查中间张量上都低于1e-6绝对差，先定位再交审查，不自行放大门控或重跑。可训练性通过不能代替P3属性结果。

P6不允许按test选择方案、seed或损失。由于历史test可能已被旧研究使用，报告为固定协议复评，不冒称全新盲测。若用户要求真正独立泛化结论，需另立数据／外层评估计划，不能在这里临时改split。

## 10. 预训练任务与微调策略的后续优化（不混入主矩阵）

三方案主轮均有完整的预训练与微调定义。以下仅在结构候选成立后，由Codex另行修订并申请预算，**本r1不授权执行**：

1. **局部空间监督**：比较现有键CE与加入非键合距离任务。监督对固定为全图非键合无序对均匀抽至多128个；预训练目标为P_train标准化log1p距离，Huber0.5，权重0.1。输入边统一使用化学图而非target cutoff图，屏蔽对应距离及所有可直接派生特征，PH条件支路在这次辅助forward禁用，才能避免由cutoff成员关系与cleanPH读取答案。额外forward成本必须配对记录，不能只给treatment更多训练预算。不是重启旧FGR。
2. **PH辅助监督而非输入**：PH-free encoder预测完整profile，控制组同架构但权重0；必须与PH输入路线分开命名，不让输入PH重建自身作为主要证据。
3. **对齐权重**：CL=1 vs CL=0，匹配重训；不默认更强对齐更好。
4. **XC适应性**：FULL vs“前5epochs只训练新readout/head，之后全部解冻”，同100epoch总预算、同schedule，R0／CONST／PH全配对；不只为PH额外调参，不使用test挑策略。
5. **更丰富全局PH**：barcode集合编码或元素通道一次只改一个；训练集固定统计，空图和censored bars定义先锁定。完整MCP、动态PiPE、H2、多构象、MD均不在本轮。
6. **attention缩放**：当前旧实现head64但scale=1/sqrt512。若测试标准缩放，作为独立3D版本，不静默改共享O8，不与PH试验同时修改。

这一节不是把未定义细节交给执行者临场选：进入任一项前必须更新活动合同、完整预算及接口，r1执行端看到此节应停止而非自行扩展。

## 11. 工程实施与必要验证

### 11.1 最小新增范围

优先复用现有样本读取、mask、DDP归一化、split、checkpoint、报告失败保存逻辑；不要修改旧GALPH生产类以改变历史数学。拟新增路径：

```text
src/dataset/glt_ph_fusion_inputs.py          # spatial/local PH adapter，纯构建与读取分离
src/modules/glt_ph_fusion_candidates.py      # A/B/C独立类与共同输出
scripts/pretrain_glt_ph_fusion.py            # 复用训练组件，family/arm显式
scripts/finetune_glt_ph_fusion.py            # development/formal明确互斥
scripts/aggregate_glt_ph_fusion.py
configs/mts/ph_fusion/{common,a,b,c}.json
tests/test_glt_ph_fusion_inputs.py
tests/test_glt_ph_fusion_models.py
tests/test_glt_ph_fusion_protocol.py
```

若拆分更小文件不会改变合同，可由执行者自行决定；不得借此重构全项目。局部PH全量构建命令必须独立，不能藏在Dataset构造函数里。

runner拟支持 `audit / preflight / pretrain / finetune / aggregate`，明确 `family={reference,A,B,C}`、`arm={R0,R2D,CONST,STAT,PH}`、`mode={smoke,development,formal}`、config、真实source/split/output路径。实现后先交付 `--help`、resolved config及实际命令，不凭本文想象现有参数签名。不允许缺失路径时回退猜测。

### 11.2 只测本轮风险

- PH数值与现有缓存抽样一致；新local PH对合成连通／环／填充三角形参考正确，明确graph cycle与VR H1的区别。
- 同步重排原子、键、mapping：atom/bond输出等变、graph输出不变；平移／旋转／反射不变。FP32 atol/rtol起点1e-5，极端近邻边界单测。
- 三个空间图嵌套、无self、无跨图、真键与空间接触分开；参数相同三臂只替换声明输入。
- 同原子双attachment、跨RU两端不等长、中心N=0、空incident、无几何、空PH子集、含NUL key；真实样本找不到时fixture且明确证据层级。
- C masked输入不经原始z绕过；A/B公共mask与原模型一致；常量输入跨样本相同，真实输入有变化，STAT独立于PH函数。
- 固定初始模型、eval、非退化fixture：真实输入和常量应可观察到中间消息／输出差异。B初始uniform路由例外：先验证router参数微小非零的反事实并逐位恢复；正常训练256步再检查。
- 各新模块至少存在有限非零任务梯度；不要求每参数每batch都有梯度。新PH参数不得重复进optimizer，不用“范数没降”代替更新证据。
- DDP某rank无几何、全部rank无几何时collective/backward正常；无几何仍训练2D。
- 保存恢复：真实named_parameters、optimizer、scheduler、数据位置与各rank RNG；额外diagnostic不修改batch、gate或train/eval状态，不扰动公共随机流。
- 输入不同≠模型有用；预测不同≠性能提升；32／256条检查不能声称全量化学审计。

### 11.3 报告与失败路径

沿用r6修复：先保存best与核心metrics，再执行附加diagnostics；任一步必需步骤失败，写runtime FAIL、`complete=false`、非零退出，保留已完成训练。launcher立即保存真实exit code，失败不写ALL_DONE，不用Bash内建变量GROUPS。

在授权smoke前先用零update真实小batch检验报告路径，防止重复消耗训练后才发现设备错误。日志超过checkpoint时保留被放弃尾部，不覆盖失败证据。resume身份检查沿用现有必要机制，不默认新建一套hash框架。

## 12. 运行、交付与完成定义

### 12.1 运行约束

所有GPU、worker或预计超过一分钟的命令在 `tmux Uni-Poly` 独立window；启动前查重，不停止其他工作。预训练并行资源不足时等待／报告，不自行降world。下游最多4个unit并行各占一卡，必须记录实际GPU与CPU线程；每个unit自己保存完整产物。

结果根建议 `results/glt_ph_end2end_20260920/`，日志 `logs/glt_ph_end2end_20260920/`；family/arm/stage/task/fold/seed分别隔离。新局部PH sidecar单独目录，旧active cache不修改。监控默认15分钟一次，只在失败或阶段结束时立即检查，禁止高频轮询。

### 12.2 必需交付

每条预训练：resolved config、初始化来源、数据划分身份、optimizer组、实际步数、有效分母、P_val诊断、resume、固定step deployment、运行成本。每个微调unit：run.json、完整history、best.pt、metrics.json、runtime.json、按indices对齐预测，明确mode/protocol/arm/checkpoint/epochs_run/outer_test状态。

所有报告同时列出：

1. 修改、测试、smoke、开发比较、正式评估各自状态；
2. R0、R2D、CONST、STAT、PH的逐task/fold/seed差值与聚合公式；
3. PH输入、参数梯度、路由／注意力只是诊断；属性结果才决定模型选择；
4. 总参数、训练参数、PH与空间缓存开销、预训练时间、推理延迟／显存；
5. 全样本主结果及几何valid-only补充，不能只报告有利子集；
6. 失败、未执行、超预算、最后epoch最佳、既有benchmark暴露等限制。

### 12.3 验收与终止

- P0/P1通过仅为实现／接口合格；P3是探索筛查，不宣布预测性能提升。
- P5通过仅为有限开发稳定性；P6之后才能说固定协议正式复评是否改善，并披露非盲测历史。
- 数据身份错、mask直接偷读、NaN/Inf、跨图泄漏、非匹配预算、test参与选择立即停止受影响阶段。
- 全部候选未通过则形成有价值的限定负结论，保留较优无PH架构建议；不能以“目标是提升”作为无限追加实验理由。
- 用户未授权新阶段时交回规划，不自动从P3走到P6。ZCode报告执行完成，Codex依据diff与产物最终审查。

## 13. 本轮文档交付记录与下一步

**已完成（文档层）**：核对AGENTS与工作区、成功ff-only pull，读取当前GALPH模型／训练／微调配置与checkpoint身份记录，核对上述论文链接，制定三条端到端候选与分阶段对照。未修改代码、缓存、split、checkpoint；未执行单元测试、模型、GPU、预训练、微调或构象生成。

**文档自检，不是独立审查**：检查三臂输入语义、维度、参数组、损失与预算闭合；检查Markdown与git diff。按AGENTS在PROJECT_HISTORY.md追加本轮文档交付归档，不关闭尚未授权的新实验周期。后续执行记录应进入经用户指定的活动入口；本文件不伪造实验结果。

**建议下一次用户授权仅覆盖 P0＋P1-B（含R0/R2D接口），暂不覆盖5k训练。** 执行端先提供空间关系成本、公共mask／初始化一致性、真实forward/backward与报告保存结果。通过审查后，再单独授权P2-B/P3-B。A/C作为完整备选保留，不因文档齐全自动实施。

## 14. r1 P0＋P1-B 执行记录（ZCode，2026-09-21，待 Codex 审查）

**状态**：`P0/P1-B执行完成，待Codex审查`。本记录只覆盖用户授权的 P0＋P1-B（含 R0/R2D/CURRENT 接口）；**未启动 P2-B/P3-B**，未训练 5k，未做开发比较，未访问 outer-test，未实现 A/C。所有数值为自己的执行结果，不是独立审查；性能结论仍是空白。

### 14.1 执行前核对

* `git status` 显示用户改动 `M PH.md`（删除"与 Plan.md 的关系"一行）、`D Plan.md`、未跟踪 `.zcodeignore`；`git pull --ff-only origin dev` 成功且 Already up to date（0/0，基线 `bc21d05`）。tmux `Uni-Poly` 无活动训练进程，4 张 4090 空闲。用户改动全部保护，未暂存未提交。

### 14.2 P0 来源／身份／接口／成本审计（`results/glt_ph_end2end_20260920/p0/p0_audit.json`，`P0_EXIT=0`）

* **状态 PASS**（首轮两次因真实接口问题报 BLOCKED，见 14.5）。抽样 256 条 P_train（split 工件 `results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json`，sha256 `c51345e6…`，train 911391 / validation 47968 / fixed validation 1024）。
* **冻结来源**：cohort `data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1`，`manifest_hash b03f96a1…`（与 split 工件一致）、records 959588、`main_bundle_hash 30f17b59…`、dual-static manifest `9ff122cc…`；PH sidecar `results/glt_galph_20260920/p0/ph_sidecar_betti_v2`，`glt-ph-betti-v2`、3 通道 × 32 bins、rows 911391 = P_train，`sidecar_covers_p_train=true`。
* **半径数组**：直接读取 `src/dataset/glt_ph.radius_grid()`（`linspace(0.8, 6.0, 32)`，sha256 `9e1d3a1b…` 记录在报告里），未另用端点约定重算。
* **32-byte key**：256 条全部 32 字节，其中 **33 条含 NUL 字节** → 任何字符串语义比较都会截断，本实现一律按 bytes 处理（reader 与审计均已覆盖）。
* **CURRENT 身份**：`configs/mts/glt_galph_c1_repair_5k_identity.json` 的 sha256 `3063cf8b…` 实测一致；common-init `637827e6…` 一致（201 张量；**不含 cls/ph 块**，与 r1 源码一致）。
* **原子／键／映射**：抽样样本 geometry_valid **256/256**；重原子投影与 GLT bond 行**逐行**核对（元素 + 坐标距离，容差 1e-3 Å）**0 失败**；`bond_row_count` 全部一致。真实键表按 GLT 自己的排序规则重建：`(not center, atom_a, atom_b, q_a, q_b, local_a, local_b)`，端点按 `(atom_a,q_a,local_a)` 定向。
* **三空间图规模**（全原子、双向、含所有非 self 对）：2.5Å 均值 **744**（min 14 / max 2644）、4Å **1954**（20 / 8066）、6Å **4376**（20 / 22304）；样本原子数均值 121.3（min 5 / max 395），重原子 59.3，Trimer 键 61.2。嵌套包含、无 self、双向对称、无跨图在 256 条上 0 违例；`spatial_edge_index` 与逐图重建的集合逐条相同。
* **成本**：每样本边张量（index/distance/bonded/scale，22 B/边）在 6Å 下约 96 KB → 投影到完整 P_train 约 **86.0 GiB**（2.5/4Å 分别为 14.6/38.4 GiB）；RBF 特征（33×fp32）只在消息 MLP 内按 16384 条边分块计算，未整体物化。数据构建 25.9 ms/样本、STAT 2.4 ms/样本 → 单进程为整个 P_train 生成 STAT 约 **0.63 h**（P2 前须单独授权离线构建，本轮**未**构建全量 STAT 缓存）。
* **下游无标签几何**：xc/eps/eat 各 16 条，全部取自对应 fold0 的 train∪validation 索引，**未读 test 索引、未读标签**；PH 有效率 16/16，6Å 边数均值 2283 / 1092 / 1167。限制：下游审计只统计了最大半径的边数（2.5/4Å 未逐条记录）。
* **SMOKE_ONLY 统计**：用这 256 条 P_train 的有效 PH 记录拟合 `[3,32]` 均值/标准差（std 下限 1e-3）→ `results/glt_ph_end2end_20260920/p0/ph_stats_smoke_256.npz`（sha256 `f4868b1e…`，文件内写死 `scope=SMOKE_ONLY_256_P_TRAIN_NOT_FOR_P2`）；P1 只在这份统计上训练，P2 必须另行拟合并新起初始化。

### 14.3 P1-B 实现

新增（全部只读既有缓存，未写旧缓存、未改历史 checkpoint）：

| 文件 | 作用 |
| --- | --- |
| `src/dataset/glt_ph_fusion_inputs.py` | 全原子 Trimer 读数、重原子投影、物理键表（GLT 行序）、三嵌套空间图、32-RBF+bonded、STAT 三通道、条件输入（CONST/STAT/PH）、标准化、融合样本与 collate、下游 adapter |
| `src/modules/glt_ph_fusion_candidates.py` | 条件 profile encoder（§3.2）、空间消息块（§5.2）、`GLTFusionB`（三臂）、`R2DModel`、`FusionPretrainer`、`FusionDownstream`（§7.1 读出）、部署包与 strict 加载 |
| `scripts/pretrain_glt_ph_fusion.py` | 预训练 runner：`--arm R0/R2D/CONST/STAT/PH`、`--mode smoke/pretrain`、AdamW 分组（矩阵 wd 1e-6，bias/norm/PH侧 0）、20k schedule 截断、resume、结果先落盘、FAIL runtime |
| `scripts/finetune_glt_ph_fusion.py` | 微调 runner：6 臂、Huber(0.5)、trunk 1e-5 / readout-head 1e-4、train-only scaler、validation-only、`complete/stage_completed` 语义 |
| `scripts/aggregate_glt_ph_fusion.py` | 单元校验（complete/outer_test/validation_only/有限性/预算）＋ §8 预登记阈值判定 |
| `scripts/audit_glt_ph_fusion_p0.py`、`scripts/check_glt_ph_fusion_paths.py`、`scripts/check_glt_ph_fusion_ddp.py`、`scripts/verify_glt_ph_fusion_smoke.py`、`scripts/run_glt_ph_fusion.sh` | P0 审计、零 update 报告路径检查、DDP 空几何检查、smoke 核验、launcher |
| `configs/mts/ph_fusion/{common,b,smoke_b,smoke_reference,downstream,downstream_smoke_step4,downstream_smoke_step16}.json` | §6.3/§7.1 锁定数值；smoke 配置显式标 `smoke_only` 并缩小 batch |
| `src/dataset/glt_ph.py` | 只加 `radius_grid()` 只读访问器（+5 行），不改历史数学 |

**合同落点**：O8 与 GLT 原类原样复用（`BondPathO8`/`GalformerTrimer3D`，GLTFusionB 继承 `GLTGalPH` 并重写 forward 以在第 3、5 层后插入块）；三尺度共享同一 message MLP 与无 affine LN；端点对称回写 `M_a+M_b → Linear128→512 ×0.1`；router 最后一层零初始化 → 初始 softmax **恰为 1/3**；**无** PH 总开关、无开门正则、不加载旧 C1 PH encoder（新模块在家族种子 20260924 下构建）；R0 = 原 `GLTGalPH('cls', None)` + `galformer_collate` 原数据路径；R2D = O8/CLS2 单路、无 GLT/CL，微调用 `2D_ONLY` 分支；CURRENT = 原 r6 F_OFF 架构（`GLTGalPH('cls','global')` + 无效 PH 占位 → 零残差），checkpoint 经身份记录 sha256 与 `load_galformer_deployment` 双重校验。

### 14.4 验证（零训练部分）

* **fixture 测试 34 项全通过**（CPU，无 GPU/数据依赖）：`tests/test_glt_ph_fusion_inputs.py` 12、`tests/test_glt_ph_fusion_models.py` 13、`tests/test_glt_ph_fusion_protocol.py` 9。覆盖：RBF 网格与 bonded bit、三图嵌套/无 self/双向、**临界半径 d=r 属于内图**、刚体不变（旋转+反射+平移）、STAT 与暴力参考一致及退化点集、重原子投影的双索引空间、GLT 行序复现、条件输入三臂与 PH 失效回退 CONST、router 零初始化与结构化反事实（逐位还原）、置换等变、空邻域恒等、**bf16 autocast 不混 dtype**、三臂共同初始化、部署 strict 加载（错 step/架构/缺键均拒绝）、优化器 wd 合同、launcher 真实退出码与失败即停、失败现场保留、汇总器拒绝部分产物。
* **既有套件 44 项仍全通过**（retention r5/r6 与诊断套件），确认 `radius_grid()` 未影响历史行为。
* **零 update 真实报告路径检查**（`p1/path_check.json`，`PATHCHK_EXIT=0`）：真实 4 样本 batch，GPU，bf16；预训练与下游两条路径各跑 forward/backward，**0 optimizer updates**；预训练 loss 17.573 有限、`spatial.*` 30 个张量梯度非零有限、`conditional.*` 首步为 0（零初始化 router 的声明行为）、22756 条空间边；下游 xc/fold0 4 样本 DUAL 预测有限，`metrics.json`+`best.pt` 写盘并复读成功。
* **DDP 空几何检查**（`ddp_control.json` / `ddp_partial.json` / `ddp_all-empty.json`，各 `EXIT=0`）：真实 4 rank backward，**0 optimizer updates**。control：4 rank 全有限、梯度非零；partial：rank0 无几何（3D 目标 0、空间边 0），rank1-3 正常（3D 目标 21/50/76）；all-empty：4 rank 全无几何，2D 目标 23/7/16/24 仍参与、loss/梯度全有限非零 → 通信正常、无 NaN、2D 任务不被空几何带走。
* **一个反例说明**：P_train 抽样 256 条中**没有** geometry-invalid 结构，因此 DDP 的"空几何"是**按声明空分支合成**（清空 3D 键/线/掩码/空间关系与 3D 目标，保留 O8/2D 字段与原子表），不是真实无效样本；该限制写在检查脚本与结果 JSON 里。

### 14.5 缺陷、修复与失败现场（据实报告）

| # | 现象 | 根因 | 修复与验证 |
| --- | --- | --- | --- |
| 1 | P0 首轮 BLOCKED：`bond_row_distance`（元素正确） | 我最初假设 GLT bond 行按 `(local_a, local_b)` 排序，实际是 `(not center, atom_a, atom_b, q_a, q_b, local_a, local_b)` 且端点按 canonical 序定向 | 按真实规则重建键表；256 条 0 失败（`physical_bond_table` 内注释记录该事实） |
| 2 | P0 次轮 BLOCKED：2/256 仍失败 | `bond_atom_index` 用了重原子枚举下标，却被拿去索引**全原子**坐标/空间表；H 交错时 off-by-one | 键表同时返回 `index`（重原子空间，与 GLT 行对齐）与 `atom_index`（全原子空间，供消息回写）；两个原失败样本与 13 个其他样本全部通过 |
| 3 | 零 update 路径检查 EXIT=1 | `edge_rbf` 的 RBF 中心建在 CPU，边距离在 CUDA | 基函数跟随输入 device；新增 CUDA 条件回归测试 |
| 4 | 零 update 路径检查第二次 EXIT=1 | bf16 autocast 下消息输出 bf16、累加器 fp32 → `index_add_` 报错 | 显式 `.to(pooled.dtype)`／回写 `.to(root.dtype)`；新增 CPU bf16-autocast 回归测试 |
| 5 | DDP 首次运行"卡住"（rank0 空闲、其余 3 rank 满负荷） | **不是** DDP 语义问题：rank0 在前向崩溃，其余 rank 停在 NCCL 集合通信 | 单进程复现拿到真实栈；根因是我 DDP 检查脚本的"空几何"合成错误清掉了属于 2D/拓扑的 `bond_path_fields` 且未同步清空 `label_3d`；改为**按样本**应用声明空分支后再 collate |
| 6 | 预训练 smoke 第一步失败（launcher 正确报 code=1、立即停止、未写 ALL_DONE） | runner 把参考臂名（R0/R2D）传进只声明 CONST/STAT/PH 的 family-B 数据层 | 参考臂改走原 `galformer_collate` 数据路径，B 臂走融合路径；失败现场保留为 `p1/smoke_R0_2_failed_arm_wiring/`（含 FAIL `runtime.json` 与 `FAILURE_NOTE.md`，**消耗 0 updates**：无 records、无 checkpoint、`metrics_written=false`） |
| 7 | 汇总器在单折输入下写出 NaN 被拒 | 缺失折被当作 NaN 参与比较 | 缺失折显式跳过并记入 `xc_folds`，`xc_ok` 要求两折齐全 |

### 14.6 运行、日志与预算

* 全部 GPU/长任务在 tmux `Uni-Poly` 独立 window；日志根 `logs/glt_ph_end2end_20260920/`，产物根 `results/glt_ph_end2end_20260920/`（p0/p1），未覆盖任何历史产物。
* **预训练 smoke：5 路径 × 10 updates = 50/50 updates 用满**（每条 = 独立 2 + 连续 4 + 分段 2 + 恢复 2）。launcher `chain_status.log`：**20/20 步 `code=0`、以 `ALL_DONE` 结束、`LAUNCHER_EXIT=0`**；外加 1 次 0-update 失败尝试（14.5 #6）。
* **CURRENT 新增预训练 updates = 0**（复用 r6 部署包，仅做 1 epoch 接口 smoke）。
* **微调 smoke：6 臂 × xc/fold0 × 1 epoch = 6/6 epochs**；`finetune_status.log` **6/6 `code=0`、`ALL_DONE`、`FT_LAUNCHER_EXIT=0`**；6 个单元 `complete=true`、`stage_completed=diagnostics`、`validation_only=true`、`outer_test=NOT_RUN`、`best.pt` 与 `metrics.json` 齐全，root `runtime.json` 全部 PASS。
* **smoke 核验**（`p1/smoke_verification.json`，`status=PASS`）：预训练 **64/64** 项通过，含**恢复逐位一致**（连续第 3-4 步与"2 步后恢复再 2 步"在 4 个 rank 上的 losses/grad_norm/stream_digest/position/lr 完全相同）、独立 2 步与前两步一致、部署 strict 加载、优化器分组合同；微调 6/6 单元通过。
* 零训练检查不计入训练预算：P0 审计、路径检查、DDP 三项合计 **0 optimizer updates**。
* **微调 smoke 的 R² 仅证明流程可跑，不是性能证据**（1 epoch、预训练 checkpoint 只有 4 updates）：R0 0.042678、R2D 0.080607、CURRENT 0.097700、CONST 0.014634、STAT 0.047213、PH −0.114820（xc/fold0）。**不得**据此比较路线优劣，也不得与 r6 的历史分数对照。

### 14.7 未执行项与限制

* **未执行**：P2-B 5k 预训练、P3-B 开发比较、方案 A/C 的任何实现或运行、全量 STAT 缓存构建、局部 PH 全量构建、正式微调、outer-test、OOF、train+validation refit、构象生成；CURRENT 未重训。
* **限制**：P1 的 smoke 用缩减 batch（micro 4 / global 16，配置显式标 `smoke_only`）与 SMOKE_ONLY 统计；P2 必须用 `configs/mts/ph_fusion/{common,b}.json` 与另行拟合的 P_train 统计并从共同初始状态开始，**不能续训 P1 模型**。三臂在初始 uniform 路由下输出相同（已声明并测试），因此"输入不同"不等于"当前已产生功能差异"；可训练性/接线通过不构成属性预测增益。P0 的下游边数只记录了 6Å；DDP 空几何为合成案例；R0/R2D 的 P1 验证基于 4-update 部署。
* **待 Codex 审查的问题**：(1) 三空间图的边规模（6Å 均值 4376/样本，投影 86 GiB 输入）是否要求 P2 前先做按 query 分块以外的结构削减；(2) 我自行决定的两项工程选择——参考臂走原 `galformer_collate` 数据路径、smoke 使用缩减 batch——是否符合计划意图；(3) P2 的离线特征构建（全量 STAT + P_train 统计）需单独授权与预算，本轮未启动；(4) 零初始化 router 使 conditional encoder 首步无梯度（已声明），是否需要在 P2 首 256 步检查中单独记录其解冻时点。
