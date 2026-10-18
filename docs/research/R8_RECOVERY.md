# r8：受限P2恢复与G2接续

本次仅修复运行恢复和计数，不改变科学模型或候选。基线dev d053b726cec65e5bf3ea92356b904f92cde7e083；用户明确授权的一次恢复从旧resume_15000完整状态继续到逻辑20000，保留旧r7配置身份和旧失败现场。

恢复接口同时限制来源、config和输出；逐rank核对model、optimizer、scheduler位置、RNG和数据位置。段内completed=5000与逻辑end=20000分别记录，旧18348/18347和旧在途unknown分列；完成判定要求继承记录、真实exit0、四组checkpoint路径索引及既有过程validation。未重新运行1k/2.5k/5k验证或搜索checkpoint。

PT计数使用torchrun进程出生身份、实际rank生命周期和正常CUDA启动组；NVML在两卡按PID集合去重。不能归属的当前PID保守保留，无法解释的身份／超限保存完整快照后停止。没有固定PID偏移、忽略362769或放宽两rank上限；历史362769仍未核实。G2同一外部PID跨卡仅占一个总槽位，自有预留和worker按既有CUDA出生证据去重，总并发仍4。

新G2绑定恢复所得20k parent，独立70FT，不混入旧19完整FT；复用3414 FP，新生成最多2461，旧43次启动／消费／unknown保留。全70个validation best冻结后才开放TEST；固定0.75神经+0.25既有KRR，70正式预测／140神经融合数组／至多35 UA_FIXED75数组。模型、head、split、种子、scaler、配方不变。

最终CPU suite仍由G2控制器在全部科学完成后只运行一次（空CUDA、timeout300s），人工夹具补充恢复／并发边界；本次尚未执行，提交不等于通过。异常不重试。

r8恢复启动交接（2026-10-18T10:07:29.454289+08:00）。窗口Uni-Poly:ua2d_p2_r8_recovery1，controller PID5409、torchrun PID5566。一次新增PT恢复、累计2次启动；本次已完成45次、attempted46次（可有一次在途），逻辑completed step15045。两rank完整model／optimizer／scheduler／RNG／数据继承记录已落盘，rank CUDA group依据正常启动快照绑定；旧18348边界仍unknown。当前实际状态RUNNING、exit=null，不能写成实验完成。

最近日志中位步耗时2.455秒；PT粗估2026-10-18 13:09—2026-10-18 14:25（北京时间），含G2／正式评估／最终CPU的全周期粗估2026-10-18 13:39—2026-10-18 16:05。估算使用正常日志，未跑benchmark；G2仅参考旧小任务55—78秒，Egc没有匹配完整耗时，区间不含外部资源等待与异常。

G2／正式预测／CPU均未启动；全部七任务仍未完成／不可比较，沿用第26.4论文参考。科学条件不变，共享validation／TEST参与开发，仅探索性比较。Codex已停止主动监控，不等待整周期；控制器仍持续正常限流、故障处理和成功后自动接续，不能自动第三次PT启动。

日志logs/ua2d_fusion_20261017/p2_align20k_r7_recovery1/，本地一次快照results/ua2d_fusion_20261017/p2_align20k_r7_recovery1/startup_status_r8.json；远程精简快照docs/research/evidence/r8_recovery_status.json，旧网页快照过期。未来20k产物／checkpoint索引／消费／真实exit位于新P2 final_review，G2输出到g2_fp_gate_pt20k_r8新根。最终CPU suite只在全科学完成后运行一次；当前未运行pytest，不把静态交付检查或commit/push称测试通过；独立审查未进行、不提前归档。
