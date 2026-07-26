# 问题
1. PBC不合理
2. 融合模块
3. 当前 FP dropout 较大，以后需要调整
4. GT 中 Local Graph Attention
   当前默认：
   hop < 5 才允许 atom-atom topology attention
   这是一个可以调整的参数
5. 双分支如何加到 GT 中 concat/bias
6. 百分比距离还没有加上
7.  可以 dropout 加原子注意力
8.  descriptor是否开启
9.  作者证明，对于由以下组件构成的模型：
   message-passing layers；
   nodewise transformations；
   mean pooling；
   在无限周期聚合物链上进行消息传递，与在这个 star-linking 单元图上进行消息传递是等价的。
   因此它并没有真正建立成百上千个重复单元，而是用一个闭合的重复单元图来模拟无限周期边界。
10. 让 预训练数据 先存储好
11. 加入几何置信度

## 预训练任务
+ masked atom
+ ECFP reconstruction
+ Periodic shortest path
+ Screw geometry
+ PerioGT contrast

## Graph
1. 图表征：
   + star-linking
   + 周期增强 Triper
   + pbc_star-linking
2. 模型选用
   + GIN
   + Line Graph Transformer
   + FGNN

## Geom
1. 计算 T 的方案：
   + backbone boundary T 
   + Backbone Projection-T自适应 cutoff
2. 模型选用
   + PaiNN
   + 作为掩码辅助 Graph Transformer
3. 预训练优化：Augmenting Molecular Graphs with Geometries via Machine Learning Interatomic Potential

## FP

## SMILES



