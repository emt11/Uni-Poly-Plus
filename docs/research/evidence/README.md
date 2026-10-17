# 已上传证据与本地原产物的边界

这些文件是从保存的JSON/目录一次性只读摘录的快照；采集时间在各JSON的`captured_at`中。不是实时GPU状态，也不是重新评估或独立审查。

|文件|内容|精度／限制|
|---|---|---|
|formal_results.json|43份源汇总、126条含复用参照的政策摘要；每任务R²/RMSE、std、论文差与macro7|保留原精度；不同旧协议分开，缺失项不补齐；含原报告审查声明|
|formal_results.csv|上述逐来源/政策/任务的可导入表|数值保持完整精度；不等于126个新增独立实验|
|execution_status.json|当前P2/G2及旧G1失败预算快照，另附E1/A1/SOTA-FT旧消费摘录|旧RUNNING字段和远程快照不能证明当前进程活跃；有冲突保留待核实|
|local_inventory.json|全部31个现存results顶层root、run/subrun及metadata记录计数|文件数不是完成实验数，含工程/备份/失败/开发与正式，非文件指纹或验收合同|

原始数据、split存量文件、checkpoint、完整日志、预测数组、kernel模型和FP缓存仍在训练机；JSON中的`source_local`路径只用于请求原证据，不能在GitHub直接打开。目录清单是科研运行索引，不计算或验证任何文件hash。

后续由Codex从实际产物刷新这些快照，并将更新提交/push；若只需要更新状态，不重跑训练或预测。独立审查需要的最小缺失材料另行只读导出，不能仅凭本目录宣称checkpoint和逐样本预测已经验收。
