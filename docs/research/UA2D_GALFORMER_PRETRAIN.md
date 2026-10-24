# UA_2D：Galformer式双视图预训练的两个实现版本

实现 ID：`UA2D-GALFORMER-PT-20261019-01 / r2`。授权来源：用户要求实现两种预训练策略，并进一步指定V2使用二维掩码恢复、坐标去噪、键长恢复、键角恢复及2D–3D对齐，V1/V2均不使用3D→2D融合。当前授权是代码、配置、必要测试与说明；新正式PT/FT/TEST预算为0，不启动训练、不修改旧r1_retest1。

本次静态检查时间为2026-10-19 02:31 UTC（10:31 Asia/Shanghai），基于本地`dev`分支、HEAD `3503fbe780a5399410d6db6055305925765c3ff8`及工作区中的r1新增实现。工作区已有其他修改/删除/未跟踪文件，本次仅修改本文及对应的2份源码、1个入口、2份配置、1份专项测试，保留其他改动。

## 两个版本及论文边界

依据[Galformer预印本](https://arxiv.org/html/2309.00483v1)的掩码预测、双视图对比目标；本实现保留项目512维、6层O8/三维编码器及周期Trimer载体，不复现论文12层768维架构或其GEOM训练结果。

| 实现 | 配置 | 损失 |
|---|---|---|
| V1 | `configs/ua2d_galformer_mask_align.json` | 二维掩码原子CE＋三维掩码线节点类型CE＋双向InfoNCE |
| V2 | `configs/ua2d_galformer_mask_align_geometry.json` | 二维掩码原子CE＋坐标去噪MSE＋键长MSE＋cos键角MSE＋双向InfoNCE |

O8以原子为节点，两个版本的二维任务均预测101类元素；选40%的节点，用80% MASK、10%随机同分子节点、10%保持处理，损失只在选中位置计算。O8遮蔽整份原子特征及backbone标记，路径节点编码也读取遮蔽后表示。

V1另外对三维物理键线节点执行独立的40%、80/10/10掩码，预测25755类无序端点元素/键类型组合；三维mask遮蔽整份线节点输入及其类型相关注意力bias，周期等价物理键副本同步腐蚀。V2不生成三维掩码标签，不创建三维类型分类头，也不对三维线节点执行MASK或随机替换；其三维输入保持元素/键身份，只加入坐标噪声。两种版本均不将二维掩码同步到三维元素身份，因为两路没有特征融合。

图表示使用项目原有mean/incident pooling，无论文virtual node；两路独立LayerNorm及512→256→128 projection后作对比。对比始终使用融合前表示；重复分子身份不作为假负例，无有效几何者不参加三维和对比损失。双向对比相加，显式将已有方向平均helper乘2，对应论文Eq.14的方向和。各主损失默认权重1。

两个版本默认使用同样的0.03 Å坐标噪声、相同二维mask及公共模块初始化。按本次用户指定，V2同时去掉三维掩码并加入三种几何恢复任务，因此V1/V2是两种整体策略的比较，不能将差值单独归因于几何恢复。V1是Galformer目标的O8适配版，不是所有数据增强细节的原样复刻；将`noise_sigma=0`可得到无坐标加噪的V1配置，但与默认V2比较时必须说明噪声变化。

## 几何恢复及坐标对称性

V2使用同一份已有干净Trimer作为目标，从带噪构象计算三维输入。只读取现存坐标，不生成构象或重建历史缓存。

- 键长恢复：对中心重复单元incident物理键回归干净长度，使用现有权重去除跨副本重复贡献。
- 键角恢复：对中心重原子真实one-hop键对预测干净cos θ；合并两个方向，索引使用bond offset，不使用atom offset。
- 坐标恢复：预测去质心后“干净−带噪”位移。由不变的三维键表示预测标量，再乘输入的相对坐标向量，聚合到物理原子；输出随旋转同步旋转，对整体平移不变。监督中心RU原子，不用普通MLP直接回归任意世界坐标。

损失先按有效分子平均，再作全局有效分子归一化；gradient accumulation使用整个optimizer window的全局分母，避免不同rank有效几何数量不同造成错误加权。V1不创建几何恢复头，也不读取这些恢复目标；V2不创建三维掩码分类头，不读取三维掩码目标。V1共3项损失，V2共5项损失，活动项的默认权重均为1。

## 是否需要3D进入2D

原始UA_2D在预训练的元素恢复支路使用cross-attention，属性微调/预测保留O8而舍弃该融合；UA_FUSED是另一路径。图中的融合不能解释为所有UA_2D下游都使用3D。

**依本次用户要求，两个版本都强制`fusion=none`，不是仅将其设为默认值。** 模型不构造BondTokenFusion，不执行cross-attention，不计算融合后原子CE；配置、命令行与下游恢复入口均拒绝启用3D→2D融合。

二维掩码头直接使用O8原子表示；三维任务头只使用Galformer三维表示；双向InfoNCE在两路各自池化、归一化和投影后计算。对齐损失的梯度可同时更新两个编码器，这是跨模态监督，不是把三维特征输入二维forward。V2中三种几何损失直接监督三维编码器，O8通过对齐目标学习与三维表示一致的信息。下游恢复仅保留O8、norm2及属性head，预测不要求三维输入。原有UA_FUSED或旧controller中的融合路径不属于本次修改范围。

## 入口、初始化及恢复

- 模块：`src/modules/ua2d_galformer_pretrain.py`。
- 数据准备：`src/dataset/ua2d_galformer_pretrain.py`。
- 显式预算入口：`scripts/pretrain_ua2d_galformer.py`。不在旧controller中自动注册新实验。
- `--ua-initialization`可严格复制旧UA deployment包的O8、三维backbone、norm2/norm3；不复制旧融合器。新掩码头、projection、mask token与几何头保持各自初态。缺失或不兼容张量直接报错，不使用strict=False。
- 本次目标语义变更后包schema更新为`ua2d_galformer_objectives_v2`，旧r1包不会被静默解释为新策略。`restore_downstream(package, head_state)`只构造O8/norm2/head，显式传`fused=True`直接报错；包中`prediction_paths`固定为`['pure_2d']`。
- 下游输出标准化属性值，调用方仍须使用该task/fold的train-only scaler反变换。属性head仍由调用方明确提供初态或已训练权重。

以下仅是未来授权执行的命令模板，`STEPS`、`MICROBATCH`等预算尚未授权，不可当前启动：

```bash
torchrun --standalone --nproc_per_node=2 scripts/pretrain_ua2d_galformer.py \
  --config configs/ua2d_galformer_mask_align.json \
  --source-root EXISTING_DIRECT_SOURCE --index-artifact EXISTING_TRAIN_INDEX \
  --output NEW_OUTPUT_V1 --max-steps STEPS --microbatch MICROBATCH --accumulation ACCUMULATION
```

V2只更换config和新output。入口仅允许`--fusion none`或省略该参数。运行必须在tmux `Uni-Poly`独立window，保留命令、cwd、日志和真实exit；禁止覆盖旧目录。显式steps决定新预算，不默认沿用旧5k或25k额度。本轮不增加task/fold/seed，不更新SOTA结果或声称已训练有效。

## 本轮验证与限制

本次r2最终一次CPU专项测试只使用合成分子与坐标，验证独立二维mask、V1/V2损失集合、V2不读取或应用三维mask、跨图索引、各活动头梯度、版本共同初态、坐标等变性、纯二维恢复及禁止融合的约束、optimizer-window分母。不运行训练smoke、旧主线suite或真实checkpoint重放。多GPU训练、吞吐与正式下游收益不在本轮验证范围。

历史r1执行记录（2026-10-19，Asia/Shanghai）：当时专项CPU测试为 **9 passed、1 warning、7.84s、exit 0**；warning为torch_geometric依赖弃用提示。tmux窗口`Uni-Poly:ua2d_galformer_final_cpu`，日志`logs/ua2d_galformer_implementation_20261019/final_cpu/`。该记录对应修改前目标，不能代替本次r2验证。其后静态收尾补充了输出目录归属保护，未重复pytest。

r2执行记录（2026-10-19 02:31 UTC／10:31 Asia/Shanghai）：实现调整完成，最终唯一一次专项CPU测试 **11 passed、1 warning、8.03s、exit 0**；warning仍为torch_geometric依赖弃用提示。tmux窗口`Uni-Poly:ua2d_galformer_r2_final_cpu`，日志`logs/ua2d_galformer_implementation_20261019/final_cpu_r2/pytest.log`，命令及环境记录`command.json`，退出码`exit_code.txt`，起止时间`execution.json`，均位于同目录。验证覆盖合成分子单元级正确性，包括二维掩码损失对三维参数无直接梯度、几何损失对O8参数无直接梯度；不代表真实预训练、分布式训练或下游收益已验证。未启动新预训练、微调或正式TEST。

本轮只调整上述新源码/配置/专项测试/说明，未修改旧模型或controller；CURRENT_PLAN.md由原实验writer维护，避免同时修改，其当前运行状态不以本文判断。本次修订暂在本文记录，CURRENT_PLAN.md未写回本次变更。完成层级为实现及专项单元测试通过，状态为“待审查”，尚未独立审查；本轮新PT/FT/正式TEST/GPU训练均0。后续需按用户明确授权制定实际预算并执行，不自动续跑旧实验。
