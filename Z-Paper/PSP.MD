# PSP 论文与源码分析：从 P-SMILES 构建有限与无限聚合物结构

分析对象：

- 论文：[Polymer Structure Predictor (PSP): A Python Toolkit for Predicting Atomic-Level Structural Models for a Range of Polymer Geometries](https://doi.org/10.1021/acs.jctc.2c00022)
- 期刊：Journal of Chemical Theory and Computation, 2022
- 代码：[Ramprasad-Group/PSP](https://github.com/Ramprasad-Group/PSP)
- 本次源码审计提交：`fa846bdd07b45461d5d747e5bd60b5ee80f13938`

本文重点分析 `ChainBuilder`，因为它与当前项目的 strict periodic geometry 最相关。

## 1. PSP 的目标

PSP 是一个聚合物原子结构构建工具，不是神经网络。

输入：

```text
重复单元 P-SMILES
```

输出可覆盖：

```text
monomer
finite linear/loop oligomer
infinite periodic chain
polymer crystal candidates
amorphous box
GAFF2 / OPLS-AA parameter files
```

四个主要模块：

```text
MoleculeBuilder
ChainBuilder
CrystalBuilder
AmorphousBuilder
```

## 2. PSP 中的 P-SMILES

PSP 使用两个 `*` 表示聚合连接点。

例如：

```text
*CC*
```

两个 `*` 的相邻真实原子是 linking atoms。

PSP 要求环聚合单体先被人工打开并写成线性 P-SMILES。第一版只支持 homopolymer，不支持 copolymer、ladder polymer 和 network polymer。

## 3. 有限链与无限链是两条不同流程

### MoleculeBuilder

```text
连接多个 P-SMILES
-> 端部 * 用 H/end cap 替换，或首尾闭环
-> RDKit conformer search
-> UFF optimize
-> 可选短程 GAFF2 MD
```

### ChainBuilder

```text
先寻找适合施加周期边界的 building unit
-> 构造无限一维周期链
```

ChainBuilder 不是简单复制 MoleculeBuilder 得到的单体坐标。

## 4. Dummy Atom 替换

为了让 RDKit/Open Babel 能生成连接端附近的合理局部几何，PSP 将 `*` 替换为真实 dummy 元素：

```text
single attachment bond -> Cl
double attachment bond -> O
```

这些元素不是最终聚合物原子，只是构象生成的几何支架。

选 Cl/O 的作用是：

```text
保留 attachment bond direction
让力场有可识别的元素和参数
辅助定义两端延伸向量
```

当前项目将 `*` 替换为 H 的 repeat-unit conformer 无法完全保留相同的连接方向约束，这是 PSP 可提供的一个具体改进点。

## 5. Boundary Vector 与展开角

每端定义：

```text
dummy atom
linking/boundary atom
```

形成两个向量。论文将两向量夹角记为 `alpha`。

目标是：

```text
alpha -> 180 degrees
```

其物理直觉是让左右聚合方向尽可能相反，从而便于沿一个轴构建周期链。

## 6. Rotatable Bond 搜索

PSP 找出 building unit 中可旋转单键，随机选择：

```text
一个 rotatable bond
一个候选 torsion angle
```

通过旋转这些单键探索构象空间。

这比当前流程：

```text
ETKDG 生成完整三聚体
-> MMFF/UFF 局部优化
-> 后拟合 `(R,T)`
```

更直接地优化“是否适合周期延伸”。

## 7. Simulated Annealing 目标

论文给出：

```text
E = E1 + E2 + E3
```

其中：

```text
E1 = UFF energy
E2 = E_UFF * (1 - alpha / 180)
E3 = 0 if connectivity valid else 10 * E_UFF
```

公开源码当前实现与论文文字略有差异：

```text
E_cost = E_UFF
       + E_UFF * (1 - alpha / 180)
       + E_UFF * penalty * 10
```

论文公式排版看起来像包含平方项，但复现应以源码中的实际 objective 为准。

## 8. Simulated Annealing 调度

源码默认：

```text
Steps = 20
Substeps = 10
```

每个外循环逐步降温，每个内循环尝试新的 bond/torsion。

源码接受更差解的概率从约：

```text
start p = 0.3
end p = 0.001
```

逐步下降。

增加 Steps/Substeps 通常提高成功率，但增加 CPU 成本。

论文性能测试中的高质量配置使用更大的搜索预算，例如 SA2 报告：

```text
steps = 100
substeps = 20
```

## 9. Local Optimization

SA 后，PSP 使用 Open Babel UFF 和 steepest descent 做局部优化。

优化时可固定：

```text
dummy atoms
linking atoms
```

以防周期连接方向在局部最小化中被破坏。

这与当前项目的重要区别是：

```text
PSP 在优化阶段就约束 boundary geometry。
当前项目主要在优化后用 quality gate 检查 boundary/screw consistency。
```

## 10. Dimer Fallback

如果单 RU 无法找到合适周期构象，PSP 不直接放弃。

它会：

```text
构建 flexible dimer
把 dimer 当作新的 building unit
重新执行构象搜索、SA、局部优化和周期构建
```

其理由是：

```text
dimer 增加 backbone rotatable bonds，
允许最小周期单元大于一个化学 RU。
```

这直接支持当前项目采用 adaptive m-RU periodic cell。

## 11. Intrachain Correction / Stretching

即使 `alpha` 较大，随机 torsion 搜索仍可能得到扭曲或螺旋 backbone。

PSP 的第一版反而主动避免 helical geometry，因为作者担心它引入额外 intrachain van der Waals interaction。

`MakePolymerStraight` 的逻辑为：

```text
1. 将两端 dummy/linking atoms 朝 z 轴相反方向移动。
2. 每步约移动 0.5 Å。
3. 固定这些边界原子并优化其他原子。
4. 监控 UFF energy increase。
5. 遇到超过阈值的能量突增时停止。
```

用户参数：

```text
IntraChainCorr = 1
Tol_ChainCorr = 50 kJ/mol by default
```

论文测试了 50 和 150 kJ/mol 阈值。

## 12. 无限周期链构造

当 building unit 合格后：

```text
1. 对齐两端 dummy atoms 定义的轴到 z。
2. 删除 dummy atoms。
3. 以 attachment-to-attachment 的轴向跨度定义周期长度。
4. 建立只在 z 方向周期的 unit cell。
5. x/y 方向留约 12 Å 真空隔离周期像。
```

因此 PSP 的无限链主要是：

```text
translation-only 1D PBC
```

它不是当前项目的一般 screw `(R,T)`。

## 13. Oligomer 输出

ChainBuilder 也可输出长度为 `n` 的 oligomer：

```text
复制 building unit 沿 z 排列
移除内部 dummy atoms
端部用 H 饱和
```

如果 attachment 是双键，直接用 H 封端可能不满足价态，因此论文建议重新选择 P-SMILES 切断位置，使 attachment bond 尽量是单键。

## 14. 1000 P-SMILES 性能结果

MoleculeBuilder dimer：

```text
single core success = 97.2%
multicore success = 98.5%
```

但这只是有限 dimer 保持 connectivity 的成功率，不是无限链成功率。

ChainBuilder 无限链：

```text
simple dimerization = 71.8%
SA + stretching, threshold 150 kJ/mol = 94.6%
```

时间：

```text
81.4% samples < 5 min
57.9% samples < 1 min
single core per model
```

## 15. 94.6% 的正确含义

这个数字不能直接解释成：

```text
94.6% polymer 获得真实最低能无限链
```

它表示 PSP 定义下：

```text
成功构造出满足连接和周期构建条件的 chain model
```

其质量依赖：

```text
UFF energy
alpha 展开程度
connectivity
stretching threshold
```

并未使用当前项目严格的：

```text
left/right common `(R,T)` RMSD
five-cell collision
force gradient gate
explicit screw symmetry consistency
```

因此 PSP 的 94.6% 与当前 strict screw 25%-50% 不是同一指标。

## 16. 实验结构验证

论文展示 PE、PVC、PVDF：

```text
PE 与实验链结构一致。
PVC 可生成 isotactic/syndiotactic examples。
PVDF 总是偏向 beta phase。
```

PVDF 的例子暴露 UFF 的限制：

```text
UFF 认为 beta phase 能量低于 alpha phase，
但实验中 alpha phase 也容易出现。
```

作者明确指出真实构象可能是 bulk property，单链能量不足以决定体相构象。

## 17. CrystalBuilder

PSP 从单链进一步生成 crystal candidates：

```text
复制链
绕自身轴旋转
绕另一条链旋转
沿 z 平移
移动到最近原子距离约不小于 2 Å
UFF optimize and rank
```

这属于链间 packing，不应混入当前单链 SCAGE geometry bias。

## 18. AmorphousBuilder

流程为：

```text
生成有限 oligomers
-> 根据密度确定 simulation box
-> PACKMOL 随机平移旋转并装箱
-> 生成 LAMMPS + OPLS-AA/GAFF2 参数
```

这说明材料性质可能需要多链无定形环境，但成本远高于重复单元级属性模型。

## 19. PSP 局限

论文明确列出：

```text
高度支化大分子构象搜索困难
不能保证覆盖全部重要 conformers
只支持 homopolymers
不支持 copolymer / ladder / network polymers
UFF 准确性有限
孤立链不等价于 bulk polymer
```

额外工程问题：

```text
源码依赖较旧 RDKit/Open Babel/PySIMM/LAMMPS 生态
大量 CSV/XYZ 文件中间操作
部分 pandas API 已过时
原生流程不适合直接放入 GPU DataLoader online 执行
```

## 20. 与当前项目的对照

| 项目 | PSP ChainBuilder | 当前 screw_periodic |
|---|---|---|
| 周期形式 | z-axis translation | general screw `(R,T)` |
| 搜索变量 | rotatable-bond torsions | ETKDG conformers + fitted rigid transform |
| 边界处理 | dummy/linking atoms constrained during search | trimer connection + post-fit quality gate |
| 优化器 | SA + UFF + steepest descent | ETKDG + MMFF/UFF |
| 多 RU | monomer failure -> dimer building unit | 当前严格主线通常 one-RU screw |
| 拉直 | explicit stretching | 无 PSP-style axial stretching |
| 成功率 | up to 94.6% under PSP criteria | strict screw 约 25%-50%，定义更严格 |
| 输出用途 | DFT/MD initial structure | SCAGE distance bias / geometry targets |

## 21. 对当前项目最有价值的帮助

### 21.1 周期约束应进入构象搜索，而非只做后验拟合

当前项目的高失败率部分来自：

```text
自由三聚体局部最低能结构
不一定满足同一个 exact screw operation
```

PSP 给出的方向是：

```text
在搜索 torsion 时直接优化 boundary extension 和 connectivity。
```

### 21.2 attachment cap 应表达连接方向

只用 H cap 容易让端部几何收缩成普通 C-H/O-H 键。

可借鉴 dummy cap，但不必机械使用 Cl/O：

```text
用虚拟 cap 坐标和 boundary direction constraint
或使用可参数化 dummy atom
```

避免 cap 的元素化学影响污染最终能量。

### 21.3 从 one-RU 自动升级到 multi-RU

PSP 的 dimer fallback 是 adaptive periodic unit 的直接工程依据。

当前项目应推广为：

```text
m = 1 -> 2 -> 3 -> 4 -> 6
```

每一级都要求 strict periodic closure，不能把普通有限 oligomer 称作 PBC。

### 21.4 分层 candidate search

推荐：

```text
快速 torsion/axis heuristic
-> force-field constrained optimization
-> strict geometry quality gate
-> 保存 top-k candidates
```

这比一开始就对每个样本执行高预算三聚体多构象搜索更适合 PI1M 50k。

### 21.5 明确区分“构建成功”和“物理可信”

建议 diagnostics 同时报告：

```text
topology_build_success
periodic_closure_success
forcefield_convergence
collision_pass
strict_symmetry_pass
final_periodic_coverage
```

不要只给一个 success rate。

## 22. 不建议直接照搬的部分

### 22.1 不建议强制拉直所有聚合物

PSP 主动避免 helix，但当前项目的创新恰恰包含 screw/helical periodicity。

正确策略是：

```text
保留 helical/screw candidate
并列 translation candidate
由质量和能量选择
```

### 22.2 不建议只用 UFF 排序

当前已有 MMFF 优先、UFF fallback，应保留，并考虑更可靠的 ML potential/force-field refinement 作为离线高质量路线。

### 22.3 不建议将 150 kJ/mol 阈值直接复制

这个阈值是 PSP stretching 能量突增门，不等于当前每原子能量、梯度或 symmetry RMSD 阈值。

### 22.4 不建议在线调用完整 PSP

PSP 单样本可达分钟级，5 万样本会超出当前 24 h 预算。应重写其核心思想为批量、缓存友好的 torsion candidate generator。

## 23. 推荐的混合实现

### Step 1：建立 boundary-aware capped RU

```text
识别 left/right boundary atoms
构建虚拟 cap directions
保留 attachment bond type
```

### Step 2：PSP-style torsion proposals

```text
枚举/随机采样 backbone rotatable bonds
目标 = force-field energy
     + boundary-axis penalty
     + connectivity/collision penalty
```

不要强制 `alpha=180°`，而是同时产生：

```text
translation candidate
screw candidate
```

### Step 3：直接周期优化

```text
translation: optimize coordinates + L
screw: optimize coordinates + axis + theta + rise
```

周期像在优化过程中解析生成，不再依赖自由三聚体后拟合。

### Step 4：adaptive m-RU

```text
one-RU fail
-> dimer periodic unit
-> trimer/4mer/6mer periodic unit
```

### Step 5：统一 SCAGE 输入

```text
periodic operation list
periodic unit size
explicit-image distances
periodic type embedding
```

## 24. 对 95% 目标的判断

PSP 说明接近 95% 的工程构建覆盖率具有可行性，但前提是：

```text
允许 SA 搜索
允许 stretching
允许 dimer building unit
采用较宽松的“可用于后续计算的初始结构”定义
```

若当前项目坚持：

```text
one-RU exact screw
+ strict RMSD/gradient/collision gates
```

则 PSP 不支持其达到 95%。

可行目标应改为：

```text
strict periodic coverage >= 95%
= valid one-RU screw
  + valid m-RU screw/translation cells

nonperiodic fallback <= 5%
```

## 25. 总结

PSP 对当前项目最有价值的不是某个固定参数，而是三个工程原则：

```text
1. 周期边界适配应参与构象搜索目标。
2. 单 RU 失败后应扩大最小周期单元。
3. 构建结构只是高层计算的初始结构，必须继续做质量诊断。
```

它是当前 strict screw 高失败率最直接的可执行改进来源，但不能将其 94.6% 不加区分地当作当前目标已经被论文证明。

## 26. 参考链接

- [JCTC 论文页面](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00022)
- [PSP 官方代码](https://github.com/Ramprasad-Group/PSP)
- [PSP Colab](https://colab.research.google.com/github/Ramprasad-Group/PSP/blob/master/Colab_notebook/psp_Colab_notebook.ipynb)
- [polyGen 论文](https://doi.org/10.1021/acs.chemmater.5c01644)
