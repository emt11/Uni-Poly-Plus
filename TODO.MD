# IDEAS
1. Infinite Polymer

## Graph
1. 模型选用
   + Graph Transformer + Uni-Mol
   + GIN + PaiNN
   + Line Graph Transformer
2. 原 star-linking 是否需要优化？
3. FGNN

## Geom
1. Geom 模块需不需要 star-linking ，如果加上了 star-linking 怎么保证和聚合物链一样合理？Graph 加上 star-linking 在 Graph-Transfomrer 和 GNN 的消息传递中都能和 聚合物链效果一致！   

## FP

## SMILES

## 预训练
> 预训练数据可以使用 PL1M 
1. 预训练权重问题
2. 预训练逻辑优化
   + 周期性增强 PerioGT 中
   + 预训练任务调整
     + Repeat-unit vs star-linking consistency 调整
     + 删除 Pairwise distance reconstruction 逻辑
     + 周期性预测
     + 3D 噪音使用 EDCL 论文优化

## 模型
1. 每个模型的优化以及参数的优化
   + FP 模型的优化
   + SMILES 模型的优化


