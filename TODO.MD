## Graph
1. 模型选用
   + Graph Transformer + Uni-Mol
   + GIN + PaiNN
   + Line Graph Transformer
2. 原 star-linking 是否需要优化？
3. FGNN：Fractal Graph Neural Network 是什么？
    FGNN 是这篇论文的核心预测网络。
    作者说 fractal 的思想是从不同层次、不同尺度、不同路径来处理问题。具体到网络结构，FGNN block 包含多个不同长度的 sub-path：
    浅路径：响应快，层数少
    深路径：表达能力强，层数多
    join operation：融合不同路径输出
    drop-path：随机关闭部分路径，防止过拟合
    论文 Figure 4 展示了 FGNN 的分形扩展结构：一个输入经过不同深度的图神经网络路径，最后通过 join layer 融合。作者使用 mean operator 作为 join 操作。
    你可以把它理解成：
    FGNN 不是单一路径堆叠 GNN，而是让分子图信息经过不同深度的路径，捕获不同尺度的结构特征。
    这有点像模型内部的 multi-scale ensemble。

## Geom
1. Geom 模块需不需要 star-linking ，如果加上了 star-linking 怎么保证和聚合物链一样合理？Graph 加上 star-linking 在 Graph-Transfomrer 和 GNN 的消息传递中都能和 聚合物链效果一致！   

## 预训练
1. 预训练权重问题
2. 预训练逻辑优化

## 模型
1. 每个模型的优化以及参数的优化

