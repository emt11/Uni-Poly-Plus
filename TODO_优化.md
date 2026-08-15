# MTS 优化待办（R6 后）

## 当前状态

- [x] R6 工程退役周期完成：旧 T/G/R/A 配置、编排、专用 diagnostics 和授权产物已退役。
- [x] Pretrain/Finetune 职责拆分、checkpoint 生命周期和 sidecar 轻量启动维护已完成定向验证。
- [x] 生产 resolver/launcher 在没有活动配置时 fail-closed，不会意外启动 GPU、worker 或训练。
- [ ] 下一版 MTS 配置尚未定义；当前没有可启动的生产默认或正式实验。

> 本文从现在起只保存未完成的优化任务和必要的完成记录。已退役的 R2、T1、
> G-family 和 A0–A4 不是 baseline、配置或可执行入口，不得恢复、续训或派生新实验。

## P0：建立下一版可测量基线

- [ ] 由独立配置周期定义新的 experiment JSON 与 resolver 字段，不继承旧 arm、identity
  或 profile 兼容层。
- [ ] 在配置中显式固定数据、拓扑、Trimer/Star-RBF/MSTA 开关、预训练目标、
  batch、精度、optimizer step 和下游协议。
- [ ] 新配置首先通过 CPU 解析、两样本 forward/backward 和三卡 2-step DDP smoke；
  在此之前不做正式 20k 或 8×5。
- [ ] 用新配置生成一份短程基线测量，记录 data wait、H2D、forward、backward、
  optimizer、rank wait、显存和 samples/s。旧 R2 worker 数值只是历史参考，不得直接
  写入新生产默认。

## P1：预训练前向性能

- [ ] 在新基线上运行短 `torch.profiler`，先确认真实热点，不根据旧日志直接改代码。
- [ ] 若 profiler 确认 CUDA 同步开销显著，优先检查
  `src/modules/mips_local_graph.py::MSTAMIPSLocalAttention._branch_mask()` 中的
  `target.max().item()` 和 GPU `any()`；将可以一次验证的结构不变量移到 collate/首批
  检查，保留直接影响正确性的边界检查。
- [ ] 若 Star-RBF v2 的 `min/max/any` 和 relation→pair gather 位于热路径，对比“每批重复
  验证”与“collate 后结构已验证”的实测差异，不改 Star-RBF v2 几何数学。
- [ ] 检查 `src/training/pretrain/engine.py` 的 per-batch `.item()`/小张量 collective；只对
  profiler 命中的同步点做 GPU 累加和日志间隔汇总。
- [ ] 每个优化只做一个最小修改，立即重跑同一短 benchmark；吞吐未提升或数值
  行为改变时回退该候选。

## P1：DataLoader 与 Sidecar 启动

- [ ] 只有新基线的 data wait 占比足以影响吞吐时，才重测 worker/prefetch；不再默认
  扩大 workers。
- [ ] 对预训练比较 `workers=0` 与少量多 worker 候选，同时记录 CPU、RAM、`/dev/shm`、
  worker crash 和 samples/s；不用旧 R2 结果代替当前测量。
- [ ] 测量 `StarRBFV2Sidecar` 单进程和 DDP 多 rank 启动耗时，确认训练只读
  `model_row()`，完整 QC 仅由 `scripts/qc_mts_sidecars.py` 显式调用。
- [ ] 若启动 I/O 仍是主要瓶颈，再定位 mmap/page-fault 和多 rank 重复读取；不重建
  冻结 cache/sidecar，不重新引入全文件 SHA 或完整 audit 到启动路径。

## P1：让 3D 信息获得直接预训练信号

科学候选必须在新 baseline 稳定后单独制定 matched 实验计划。本待办不授权直接
实施或训练。

- [ ] 第一候选仅加入 periodic relation distance/RBF 不变量去噪，不增加 3D 构象数量。
- [ ] 只监督合法、非 self、`SPD≤2` 的 relations，分别记录 `|shift|=0/1/2` 损失，
  避免目标被同 RU relation 数量主导。
- [ ] 首轮只比较“新 baseline objective”与“baseline + 几何去噪”，不同时改
  Attention、MSTA、Backbone、RBF upper 或下游超参。
- [ ] 若不变量去噪有稳定收益，再比较 shift-conditioned geometry message/value gate；只使用
  `|shift|`，保持 relation inversion symmetry。
- [ ] 只在距离/RBF 去噪不足且有清晰必要时，考虑由归一化相对方向构造的等变
  坐标去噪头；不使用普通 `Linear(hidden, 3)` 直接回归绝对 XYZ。
- [ ] topology-only/Trimer-3D 双视图对齐放在上述候选之后，先短测两次 forward 的
  真实成本，不一次引入多个新 objective。

## P2：Finetune 速度与下游质量

- [ ] 新 checkpoint 可严格加载后，在一个代表性 task×fold 上重测 workers、eval batch 和
  FP32/BF16；旧的 workers=2、eval batch=64、FP32 只作历史参考。
- [ ] 微调性能候选必须保持 prediction/loss 在明确容差内一致，不为吞吐改变 fold、
  task 或训练语义。
- [ ] 预训练几何候选完成后，再单独制定多任务微调或 Graph+SMILES matched 对照；
  XC、EI、EEA 仅作重点观察任务，不针对单任务调网络追分。
- [ ] 正式评价仍使用 8 tasks×5 folds 的 task-level 聚合，不把 40 folds 当作独立样本。

## 统一执行顺序

```text
定义新配置
→ CPU 解析与两样本正确性
→ 3 卡 2-step DDP smoke
→ 短基线测量
→ profiler 定位真实热点
→ 单点工程优化与同口径复测
→ matched 3D 预训练目标计划
→ screening 短训练
→ 证据足够时才授权正式 20k
→ final checkpoint 后再做微调速度和 8×5 评价
```

## 完成口径

- 工程优化必须有同配置、同 batch 的前后测量，不用静态推断宣称加速。
- 科学候选必须与新 baseline 做 matched 比较，不把已退役实验伪装成当前对照。
- smoke/screening 只证明可运行，不代表模型性能。
- 不增加 3D 构象数量，不重建冻结 cache/sidecar，不在无活动配置时启动训练。
