# MCL-PH-20260921-01 / r10R3 GATE→XATTN 执行记录

计划修订：r10R3-GX1。状态：**执行中（GATE 已完整验收通过；XATTN 尚未启动，是下一步）**。原计划基线：`dev@73a1980`；本次记录更新基线：`dev@3dd751e`，`origin/dev` 同步，修改前 `git pull --ff-only origin dev` 返回 Already up to date、工作区干净。授权来源：用户明确要求仅执行 GATE、XATTN 两次正式预训练。实际执行与自检：Codex；独立审查待后续。

## 本轮问题、对照与范围

- Scientific question：在相同数据、统计、共同初始化、路由和训练预算下，验证 GATE 与 XATTN 融合配置的正式预训练产物能否完整生成并通过身份、清理和部署严格加载验收。
- Reference：已验收的 CAT 5,000-update arm，部署包 SHA256 `eed6276565f0bc827fc1f56056a25cd8469db184b0e334d04c3dd0f3cfea0e49`。
- Controlled change：每臂相对 CAT 仅改变配置中的 `fusion_mode`；GATE 与 XATTN 串行启动，各仅一次、最多 5,000 updates。
- 固定预算：world=4、microbatch=84、accumulation=3、global batch=1008、BF16、AdamW lr=2e-4、warmup=2,000、scheduler=20,000、500 dense updates 后 Top-2、save_every=1,000。
- 已消耗正式预训练预算：15,082 updates / 4 次启动；本轮余额 10,000 updates / 2 次启动。不得重试、续训、添加 pilot 或 wall-clock timeout。
- 本轮仅执行 GATE → 独立验收 → XATTN → 独立验收。不得启动 development、P3、OOF、outer-test；不读取 outer-test，不报告预测性能排名。

## 固定输入与前置证据

- 统计：`results/mcl_ph_20260921/p0_r10r3/statistics.npz`，SHA256 `9dc8160f1de5152f6c04a963c40569bac8facd21e68a700f40186363798cf8b1`。
- 共同新参数初值：`results/mcl_ph_20260921/p2/pretrain/shared_new_init.pt`，SHA256 `499309392d578daf7a7baf1d102d7a7f5c652e5a9b42ca2904ac9d83d1fab85a`。
- 初始 common artifact SHA256 `1c9f97cf5547dcd2f44846e83586dfb4ec593dd4a4f56df510be372f26f1951d`；CAT runtime identity 与两个源文件 SHA 一致。
- Trajectory cache：`data/processed/mcl_ph_cache/p2_noisy_seed42_sigma003_step5000_v1`。当前 manifest 为 complete=true、51 shards、5,040,000 positions；manifest 的 seed/noise/mask/global batch/sample-index/cohort/static identity 与已验收 CAT runtime 一致。既有 `logs/mcl_ph_20260921/trajectory_cache_verify.log` 末尾记录 `FULL CACHE VERIFY OK`。
- CAT：真实退出码文件 `logs/mcl_ph_20260921/p2_cat_r10r3_pretrain.exit=0`；runtime PASS/5000、cleanup/export/main-return 全完成；五组 resume/deploy 齐全；`scripts/verify_mcl_ph_arm.py --strict-cleanup` PASS；`deploy_05000.pt` step=5000/fusion=cat，统计和共同新初值 SHA 匹配，`build_mcl_arm('cat', ...)` strict-load PASS。
- CAT 配置、GATE、XATTN 三配置逐字段比较仅 `fusion_mode` 不同。当前 GATE/XATTN 输出目录均不存在。
- 当前 GPU：4 张 RTX 4090，预检时无活动训练进程、GPU 利用率 0%。CAT 旧 tmux window 保留；新臂须各用独立 window。

## 实施顺序与验收门

1. 在 `Uni-Poly:mcl_ph_r10r3_gate` 使用 `configs/mts/mcl_ph_gate.json` 启动一次正式 5,000-update 训练。输出 `results/mcl_ph_20260921/p2/pretrain/gate_r10r3/`，日志 `logs/mcl_ph_20260921/p2_gate_r10r3_pretrain.log`，真实退出码写入 `logs/mcl_ph_20260921/p2_gate_r10r3_pretrain.exit`。
2. GATE 退出后、XATTN 启动前独立核验：退出码 0；runtime PASS、completed_steps=5000、cleanup=complete、export_complete=true、main_returned=true；1000/2000/3000/4000/5000 的 resume/deploy 全部存在；step 500=dense、501=top2；四 rank loss/gradient 有限；统计/初值身份一致；deploy step/fusion/source 身份正确；`build_mcl_arm('gate', ...)` strict-load PASS；运行 `scripts/verify_mcl_ph_arm.py --label gate_r10r3 --arm-dir results/mcl_ph_20260921/p2/pretrain/gate_r10r3 --updates 5000 --strict-cleanup` 并记录部署包 SHA。
3. 只有 GATE 每项均通过后，才在 `Uni-Poly:mcl_ph_r10r3_xattn` 使用 `configs/mts/mcl_ph_xattn.json` 启动一次正式 5,000-update 训练。输出 `results/mcl_ph_20260921/p2/pretrain/xattn_r10r3/`，日志 `logs/mcl_ph_20260921/p2_xattn_r10r3_pretrain.log`，真实退出码写入 `logs/mcl_ph_20260921/p2_xattn_r10r3_pretrain.exit`。
4. XATTN 使用与 GATE 相同的逐项验收门，build arm 使用 `build_mcl_arm('xattn', ...)`。任一项失败即保留现场并停止，不重试、不续训、不开始其他阶段。

每个正式命令均由 `Uni-Poly` 独立 tmux window 承载，工作目录为仓库根目录；stdout/stderr 完整写入指定日志，命令真实退出码写入同名前缀 `.exit`。不重启 `scripts/run_mcl_ph_r10r3.py` 串行执行器。

## 执行记录

- CAT 前置验收：通过，证据见上。
- GATE：于 2026-09-23 03:43:56 UTC 在 `Uni-Poly:mcl_ph_r10r3_gate` 启动；训练退出码文件时间戳为 05:59:27.964 UTC，内容 `0`。命令使用本计划第 1 步所列 GATE 配置和固定输入，工作目录为仓库根目录；stdout/stderr 位于 `logs/mcl_ph_20260921/p2_gate_r10r3_pretrain.log`，产物位于 `results/mcl_ph_20260921/p2/pretrain/gate_r10r3/`。runtime 为 PASS / 5000，cleanup complete、export_complete=true、main_returned=true；五组 resume/deploy 齐全。解析完整日志得到四 rank 各 5000 条记录，所有 loss、总梯度及分组梯度有限，step 500 全 rank=dense、step 501 全 rank=top2；统计 SHA、shared-new-init SHA 和 common-init SHA 均与固定输入匹配。`scripts/verify_mcl_ph_arm.py --label gate_r10r3 --arm-dir results/mcl_ph_20260921/p2/pretrain/gate_r10r3 --updates 5000 --strict-cleanup` 返回 0 / PASS。CPU `build_mcl_arm('gate', package, expected_step=5000)` strict-load PASS；deploy 元数据为 step=5000、fusion=gate、training_route=mcl_ph，deploy SHA256 `312ea6708ee1b930001c1963714dfbdee44d8de711c5293bc0378be7a73b24e8`，参数数 19,958,723。逐项验收日志：`logs/mcl_ph_20260921/p2_gate_r10r3_strictcheck.log`，真实检查退出码记录为 `.exit=0`。阶段正式预算消耗 5000 updates / 1 次启动。
- 用户要求停止周期监控后，没有向 GATE 发送信号；2026-09-23 08:10 UTC 的单次现场核对发现 GATE 已完成。按该要求，此后不恢复 5 分钟周期监控。
- XATTN：尚未启动；GATE 独立验收现已全部通过。下一步仍按原授权，在独立 window `Uni-Poly:mcl_ph_r10r3_xattn` 启动一次，最多 5000 updates；完成后执行同一验收门。当前阶段总预算为此前 15,082 + GATE 5,000 = 20,082 updates / 5 次正式启动；剩余上限 5,000 updates / 1 次启动。
- 文件修改前同步：`git pull --ff-only origin dev` 成功；本次记录更新基线 `3dd751e4c7eebd4d3c745a82a3ec455dfc6e4402`。`.zcodeignore` 删除已包含在既有基线中，不属于本轮改动，本轮不恢复、不暂存。

## 停止条件

身份/统计/cache 不符、目标目录非空、writer 冲突、任何 rank 停滞或异常、loss/gradient 非有限、路由边界错误、checkpoint 缺失、严格加载失败、runtime/cleanup 不通过、真实退出码非零或预算超限：立即停止后续阶段并保留现场。GATE 未完整 PASS 时不启动 XATTN。整个本轮不启动 development、P3、OOF 或 outer-test。
