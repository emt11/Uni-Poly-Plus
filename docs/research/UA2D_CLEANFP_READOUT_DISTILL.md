# UA2D-CLEANFP-READOUT-DISTILL-20261019-01 / r1 实现记录

授权来源：用户“请你自己执行！”。基于dev／3503fbe780a5399410d6db6055305925765c3ff8和现有未提交工作区，执行PRETRAIN.MD的A–M；旧计划不恢复。

- 新模型 `src/modules/ua2d_cleanfp_pretrain.py`：复用O8和MaskedGalformer3D实现；同一源样本具有clean／masked两视图，两侧编码器各调用两次。原子／线节点CE使用masked；FP与InfoNCE使用clean。Q读出为4×32维query注意力、512维输出及零初始化残差门。QL对clean三维键状态按原incidence聚合，单向停止目标梯度；目标不是独立冻结教师。
- 新数据入口 `src/dataset/ua2d_cleanfp_pretrain.py`：继承原有序模板读取和预取，仅新增双视图collate及bytes传输；旧模板只读。沿用原source row、增强位置、周期分组和Morgan标签，不重建缓存。
- 新训练上下文 `src/training/ua2d_readout_distill.py`：显式新根、新六臂、新预算；原FT数据／优化／预测保存／聚合函数采用静态复用，绑定新常量和新的pure-2D恢复入口，不通过修改旧模块全局状态切换计划。对缓存只排除旧的优化weight／head／loss描述字段，完整化学参数仍须相符。
- 动态权重：原始loss按全局optimizer-window有效分母归约；rank0使用前200有效窗口尺度、20/20历史比、fp64 softmax、[0.25,4]有界比例投影及0.9平滑；总和为K。每个窗口6个微批共用一份系数，rank0广播下一份系数；ALL1始终为1。resume保存完整控制历史，但不自动重启。
- PT入口 `scripts/pretrain_ua2d_cleanfp.py`：world2/micro84/acc6、5000updates、bf16、原LR；每臂外层18000秒期限。分别记录源样本、消费位置与两次编码视图。生成最终deploy／resume，训练过程不进行额外验证forward。
- 控制器 `scripts/run_ua2d_cleanfp_campaign.py`：仅物理GPU2／3；六臂依次PT→35FT，冻结后各正式TEST一次；新目录单writer。长任务全部派发到Uni-Poly独立window，保留命令／cwd／stdout／stderr／exit。最终或异常收尾只运行一次新CPU suite，不执行旧suite或benchmark。
- 下游pure-2D恢复保留Q读出，10epochs仅训练属性head，之后60epochs联合训练O8／readout／norm2／head。逐折加载同一原始属性head初态，正式加载完整best.pt和train-only scaler。

新增专项测试 `tests/test_ua2d_cleanfp_readout_distill.py` 覆盖query图隔离／初始等价、clean数据流、local索引／归约／梯度、动态窗口及状态恢复、pure-2D checkpoint、两rank合成CPU候选与累积梯度。启动前只进行静态语法和源码检查；测试结果须以最终实际日志为准。

结果根：`results/ua2d_cleanfp_readout_distill_20261019/r1/`；日志根：`logs/ua2d_cleanfp_readout_distill_20261019/r1/`。预算、吞吐和当前状态以机器记录为准。本实现说明不代表测试通过、性能改善或独立审查通过。

## 实际执行收尾（2026-10-20 08:05北京时间）

六臂PT各5000步、210个FT／14700epochs／336000updates及210个正式TEST均完成，所有作业exit0，控制器exit0；所有本轮预算实际消耗与既定上限一致，新增FP生成0。PT单臂90–93分钟，完整周期约11小时05分钟。最终唯一专项CPU suite为7 passed、1 warning、11.70秒、exit0，日志位于 `logs/ua2d_cleanfp_readout_distill_20261019/r1/cleanfp_final_cpu_r1/`；未重复运行测试或实验。

最终validation／TEST宏平均分别为：G_ALL1 0.805／0.778，G_DYN 0.810／0.774，Q_ALL1 0.804／0.783，Q_DYN 0.811／0.783，QL_ALL1 0.808／0.774，QL_DYN 0.812／0.790。QL_DYN相对G_ALL1的TEST增量+0.012达到数值门槛，validation增量+0.007未过开发门槛；按冻结规则仍保留本轮reference G_ALL1，不根据TEST更换候选。数值展示3位小数，原始精度及全部折级路径见final_review/results.json；七任务SOTA对照已登记CURRENT_PLAN。

状态为待审查，实施自检不能代替独立审查，未归档PROJECT_HISTORY，也未把本轮reference自动替换为项目历史基线。
