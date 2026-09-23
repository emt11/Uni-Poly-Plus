# MCL-PH-20260921-01 / r10R3 执行记录

状态：**执行中（CAT 正式预训练已启动）**。基准：`dev@29e359e`；2026-09-23 UTC 开始前 `git pull --ff-only origin dev` 成功。授权来源：用户本轮明确要求实施 r10R3 方案。规划、执行与自检：Codex；本轮没有第二执行者的独立审查。

## 目标与边界

修复非键 clean target 的重原子索引、千步 checkpoint、备用分母导入和 O8_ONLY 重复 LayerNorm；重算固定 P_train 4096 样本统计。前置检查通过后，顺序完成 CAT、GATE、XATTN 各一次 5000-update 正式预训练，随后完成五臂 × XC/EPS/EAT × fold0/1 的 30 个 development units，最后运行 P2 聚合。GLT_REF 使用已验收的 `glt_ref_r10r1/deploy_05000.pt`，O8_ONLY 从同一包提取 O8。P3、OOF、outer-test 均不在授权内。

## 预算与停止条件

- 历史正式预训练：10,082 updates / 3 次启动；本轮新增上限：15,000 updates / 3 次启动；累计上限：25,082 updates / 6 次启动。
- Development 上限：30 units / 900 epochs。每 unit 最多 30 epochs，warmup 5、patience 10、seed 42，train-only scaler，仅 validation R² 选 epoch。
- 任一身份、统计、目标数值、cache 覆盖、checkpoint、非有限值、writer 冲突、退出码或预算检查失败，停止依赖的后续阶段。不中断后自动恢复，不重启同一臂，不覆盖旧目录。

## 执行顺序与验收

1. 修复代码并运行针对性无模型及局部测试；核对三臂仅 `fusion_mode` 不同，配置仍为 world 4、microbatch 84、accumulation 3、global batch 1008、BF16、5000 updates、500→501 切换。
2. 在 `results/mcl_ph_20260921/p0_r10r3/` 重算并核对 4096 样本的统计、key SHA、有限性和目标修订；核对完整 trajectory cache 的身份、51 shards 和 5,040,000 位置覆盖。
3. 在新目录 `p2/pretrain/{cat,gate,xattn}_r10r3/` 逐臂启动。每臂核对退出码、runtime、五个千步 resume/deploy、step/fusion 身份、strict-load、有限损失与梯度、路由 500/501 及实际预算。
4. 五臂部署包齐全且验证后，在 `p2/development_r10r3/` 串行运行 30 units。逐 unit 核对包 SHA、split、head 初值、参数组、`best.pt`、选中 epoch 预测与退出码。
5. 30/30 通过后运行 `scripts/aggregate_mcl_ph_p2.py`，核对双 baseline 门槛和 parent。只给 development 筛查结论。

## 当前执行证据

- 定向测试：`results/mcl_ph_20260921/r10r3_targeted_tests.log`，23 passed / exit 0。
- P0 重算：`Uni-Poly:mcl_ph_r10r3_p0`，命令 `python scripts/audit_mcl_ph_p0.py --output results/mcl_ph_20260921/p0_r10r3 --statistics-output results/mcl_ph_20260921/p0_r10r3/statistics.npz`；日志 `p0_r10r3/audit.log`，exit 0、`PASS`、4096/4096、key SHA `c0402dca…341a`、各数组有限；新统计 SHA `9dc8160f…cf8b1`。修订后的 nonbond `(mu,sigma)=(1.3945351,0.1372659)`，旧统计为 `(1.3962990,0.1445435)`；旧文件保留。
- Cache 校验：`Uni-Poly:mcl_ph_r10r3_cache`，生产 reader 对 51 shard 均校验 SHA256，完整覆盖 5,040,000 位置，exit 0；cohort、static、sample-index SHA 与 P0 来源一致。
- 配置比对：CAT/GATE/XATTN JSON 除 `fusion_mode` 外相同，world 4、84×3×4、BF16、5000 steps、save_every 1000、warmup 2000、scheduler 20000、dense 500→Top-2 501 均未变。
- 代码与计划预检提交：`31ff03f`，已推送并核实 `origin/dev` 同哈希。
- CAT：`Uni-Poly:mcl_ph_r10r3_cat`，`python3 -m torch.distributed.run --standalone --nproc_per_node=4 scripts/pretrain_mcl_ph.py` 加 CAT 配置、新统计、共享初值、完整 trajectory cache、`--diagnostics --prep-workers 12 --stop-after-step 5000`；输出 `p2/pretrain/cat_r10r3/`，日志 `logs/mcl_ph_20260921/p2_cat_r10r3_pretrain.log`，退出码文件同路径 `.exit`；**已启动，结果待核验**。
- 后续串行执行由 `scripts/run_mcl_ph_r10r3.py` 在 `Uni-Poly` 独立 window 等待 CAT 真正退出。它对每臂做退出码、runtime、五组千步文件、统计/共同初值身份、500/501 路由、有限损失/梯度及部署 strict-load 检查；失败立即记录 `STOPPED`，不启动下一臂。三臂齐全后严格读取 GLT_REF/O8_ONLY 包，30 个 unit 逐个核对退出码、unit checker、包 SHA 与 `best.pt` 身份，再运行 P2 聚合。运行状态和停止原因写入 `logs/mcl_ph_20260921/r10r3_driver_status.json`，全程不设墙钟 timeout、不自动重试。
- 初始工作区干净；修改过程中出现非本轮的 `.zcodeignore` 删除，未恢复、未暂存、未纳入本任务。

本节只记录真实进度；后续结果、提交和远端同步核实后更新。历史失败与超预算记录保留在 `MCL-PH.md`。
