# Uni-Poly-Plus 当前活动流程

截至 2026-09-29，项目只保留两条下游路线：

| 路线 | 定位 | 活动入口 |
| --- | --- | --- |
| GLT dual | 参考路线，保留 O8、3D、`concat`/`kfuse` 结构和既有训练逻辑 | `scripts/finetune_glt_dual.py`、`scripts/run_glt_dual_finetune_grid.py` |
| MCL-PH | 当前正式路线，保留 `glt_ref`、`o8_only`、`m_cat`、`m_gate`、`m_xattn` 五个 arm | `scripts/finetune_mcl_ph.py`、`scripts/run_mcl_ph_8x5_4gpu.py` |

## 唯一划分合同

八个任务 `eat / eea / egb / egc / ei / eps / nc / xc` 共用
`data/splits/mips_paper5_outer/`。生成器
`scripts/create_mips_split_manifests.py` 只接受 `paper5_outer`：

```text
outer: KFold(n_splits=5, shuffle=True, random_state=100)
inner: train_test_split(outer_train_sorted, test_size=0.2,
                        shuffle=True, random_state=42)
```

每个 manifest 固定记录任务、CSV 样本顺序 SHA-256、五折索引，以及每折
`sets_disjoint=true`、`union_equals_full_cohort=true`。活动入口和聚合器拒绝旧的
`outer5_inner20`、`mips_shared5` 输入；`periodic_tdl_official5` 文件仅作为五个论文任务的历史索引证据，不是运行输入。

## MCL-PH 微调

MCL-PH 沿用 Periodic-TDL 策略：head-only 10 epoch，随后按任务配置进行 joint
阶段；以 validation RMSE 选择 checkpoint，恢复后才读取 outer-test。EGC 使用 60
个总 epoch（10 + 50），其余任务使用 70 个总 epoch（10 + 60）。EAT 是项目适配
任务，不能描述为论文官方任务。聚合范围是 Macro8；正式结果仍由实际产物和
`paper5_test.json` 证明，本次路线收敛未启动训练。

## 共享输入与缓存

两条路线共用只读 GLT dual cohort、静态缓存和必要的 MCL-PH 统计／轨迹缓存。
预训练初始化与样本索引存放在 `results/mcl_ph_20260921/shared_pretrain/`；缺失的
`pretrain_split_v1.npz` 是当前已有 artifact 不完整项，不能据此宣称可恢复预训练。

## 已关闭路线

MTS/Distill、GLT-Galformer、GLT-PRED、Atomic-PC/EQ3D/DND、旧 PH fusion 及退役
MCL-PH 执行／开发聚合入口、配置、测试和对应日志／结果已删除。原始 GLT-PRED
初始化目录因缺少其索引 `.npz` 而暂作为依赖证据保留；它不再被活动配置读取。
论文来源快照、`PROJECT_HISTORY.md` 和历史审计文档保留为归档，不被活动入口读取。

## 当前验证边界

本周期只做了 manifest、静态导入、`py_compile`、`pytest` 和 `git diff --check`；
没有启动训练、GPU worker、缓存重建、outer-test 或正式实验。smoke／单元测试只能
说明实现合同和局部行为通过，不能说明性能提升。
