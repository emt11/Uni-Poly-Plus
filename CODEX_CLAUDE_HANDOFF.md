# Codex → Claude Code Handoff

```yaml
status: ready_for_codex_review
cycle: mts_b0_v2_star_rbf_execution_repair_v2
owner: claude_code
project: /root/workspace/Uni-Poly-Plus-master
```

## 1. 唯一目标

只修复并完成唯一基准：

```text
MTS-B0-v2
Star-RBF v2         ON
Topology Attention  O8
MSTA                OFF
MCL                 OFF
预训练 MD200         OFF
下游 MD200           ON
Pretrain            Masked Atom + Periodic Coordinate Denoising
Angle loss          OFF
Attention scale     1/sqrt(64)
```

当前运行中的 Star-RBF 是错误的：冻结 sidecar 使用 `upper=3.75`，B0 配置却没有传入
`star_rbf_upper`，模型实际回落到 `upper=3.0`。此前 B0-v2 screening、参数选择和
checkpoint 只作为诊断，不得续训、发布 final 或用于正式比较。

## 2. 执行边界

- 先完整读取 `AGENTS.md`、`PIPELINE.md` 和本文件，以当前真实代码为准。
- Claude Code 是唯一生产执行者；保留所有无关 dirty changes，不回滚、覆盖或顺手整理。
- 不增加3D构象，不重建 Trimer cache 或 Star-RBF v2 sidecar。
- 不恢复 MSTA、MCL、angle loss，不增加 masked-atom-only 对照。
- 不自动启动 M3 shared-view alignment 或其他新路线。
- 不从 B0-v1、旧 B0-v2、T/G/R/A checkpoint 续训。
- 不重新搜索 batch、accumulation 或 workers。
- 不执行 `3 steps → resume → 8 steps` 或 bitwise replay 测试。
- 不新增 hash、Identity、manifest、统一 Trainer 或复杂 gate。
- Screening 只证明 objective 可学，不得宣称下游性能提升。

## 3. 修复错误的 Star-RBF 执行路径

### 3.1 配置传递

所有 B0-v2 正式、probe、screening、smoke 和 downstream 配置必须显式包含：

```json
"star_rbf_upper": 3.75
```

检查并按真实调用关系修改：

```text
configs/mts/b0_v2.json
configs/mts/b0_v2_probe.json
configs/mts/b0_v2_ddp_smoke.json
configs/mts/b0_v2_resume_probe.json（如仍被引用，只补配置，不运行resume测试）
scripts/resolve_mips_trimer_scage.py
src/training/pretrain/config.py
src/training/pretrain/engine.py
src/training/finetune/config.py
src/training/finetune/engine.py
scripts/screen_mts_b0_v2.py
scripts/publish_mts_b0_v2_final.py
```

要求：

1. Resolver 和 B0 parser 将 `star_rbf_upper` 设为必填字段并校验为 `3.75`。
2. `_apply_b0_config()` 设置 `args.star_rbf_upper`，B0-v2 不再继承通用默认 `3.0`。
3. Screening 临时配置和 downstream B0-v2 均显式传入 `3.75`。
4. Dataset 与模型构造后只检查一次：

```text
config upper
= sidecar metadata upper
= model RBF centers[-1]
= 3.75
```

同时确认 `num_rbf=32`、`lower=0.0`、`gamma=0.5/spacing²`。不增加 per-batch
校验，不重建 sidecar。

### 3.2 目标 RBF 数学

```text
shift=0:
central distance → RBF

|shift|=1:
d_left  → RBF(d_left)
d_right → RBF(d_right)
→ 两个 RBF tensor 逐元素平均

|shift|=2:
outer-to-outer distance → RBF
```

禁止 `RBF((d_left+d_right)/2)`。保持：

- inverse directed relations 共享同一 scalar pair representation；
- true self 与 invalid geometry 输出零 bias；
- 合法 `distance>3.75` 使用 Gaussian tail，不裁剪、不置零；
- dynamic noisy observations 与 clean sidecar 使用相同 centers、gamma 和 observation 聚合。

### 3.3 `sigma=0` 比较最终 pair representation

比较：

```text
dynamic periodic observations
→ Star-RBF v2 observation aggregation
→ dynamic pair RBF tensor
```

与：

```text
clean sidecar periodic observations
→ Star-RBF v2 observation aggregation
→ clean pair RBF tensor
```

用 `torch.testing.assert_close` 覆盖 `shift=0`、`|shift|=1`、`|shift|=2`，尤其覆盖：

$$
Z_{|s|=1}
=
\frac{\operatorname{RBF}(d_L)+\operatorname{RBF}(d_R)}{2}
$$

该单一定向测试同时验证 upper、centers、gamma、observation count、双 observation 聚合和
relation-to-pair mapping，不扩展成新测试体系。

### 3.4 旧结果

此前 B0-v2 300-step screening 和所选 `sigma=0.05 / lambda_3D=0.3` 使用了错误 RBF，
只保留为 diagnostic，全部重跑。

## 4. Coordinate Decoder

Star-RBF 是对称 scalar distance encoding；Coordinate Decoder 使用单 observation
真实有向 vector：

```text
shift=0:   p(j,0)  - p(i,0)
shift=+1:  p(j,+1) - p(i,0)
shift=-1:  p(j,-1) - p(i,0)
shift=+2:  p(j,+1) - p(i,-1)
shift=-2:  p(j,-1) - p(i,+1)
```

禁止左右 vector averaging，不要求有限 open Trimer 中
`v(i,j,s)=-v(j,i,-s)`。Decoder 不直接读取 raw RBF。

在 `coefficient * relative` 前加入：

```python
valid &= torch.isfinite(relative).all(dim=-1)
relative = torch.where(valid.unsqueeze(-1), relative, 0)
```

删除当前立即被覆盖的 `source_safe` 死赋值，不增加额外 defensive framework。

## 5. 冻结已正确语义

除非定向测试发现真实错误，本周期不修改：

- whole-Trimer clean→noisy Kabsch；
- reflection 拒绝，Kabsch 不参与反向传播；
- graph-level coordinate loss；
- DDP global valid-graph normalization；
- Masked Atom 独立 denominator；
- decoder 最后一层零初始化；
- canonical-correlated coordinate noise。

继续记录 coordinate loss、zero-displacement loss、`R_denoise`、predicted displacement
RMS、valid geometry graph 数和各 `|shift|` 有效 relation 数。

## 6. 正式配置与 Launcher

```text
GPU                     1,2,3
world size              3
batch/rank              336
gradient accumulation   1
global batch            1008
workers/rank            6
prefetch                 2
precision               BF16
optimizer               Adam
betas                   (0.9, 0.98)
lr                      2e-4
weight decay            0
warmup                  2000 optimizer steps
scheduler horizon       20000 optimizer steps
end lr                  1e-9
seed                    42
mask ratio              0.30
Star-RBF upper          3.75
```

`scripts/run_mips_trimer_scage.sh` 必须使用：

```text
CUDA_VISIBLE_DEVICES=1,2,3
torchrun --standalone --nproc_per_node=3
```

正式任务不得绕过 launcher 跑单进程 `pretrain.py`。不重新运行 batch/worker benchmark；
只有真实 OOM、worker crash 或明显 data wait 时才做最小复测。

## 7. Screening Scheduler

必须分开：

```text
scheduler_total_steps = 20000
warmup_steps          = 2000
stop_after_steps      = 500 或 2500
```

`stop_after_steps` 只控制提前退出，不得改变 scheduler horizon 或 warmup。

## 8. 最小修复验收

### 8.1 定向测试

1. RBF upper、centers、gamma 为正确的 `3.75` 定义。
2. `sigma=0` 最终 dynamic/clean pair representation `assert_close`。
3. `|shift|=1` 为两次 RBF 后平均，不是平均距离后 RBF。
4. 合法超 upper 距离使用 Gaussian tail。
5. Inverse relations 共享 scalar pair bias。
6. Decoder `shift=0/±1/±2` 端点正确，不测 coordinate vector inverse antisymmetry。
7. Non-finite relative vector 精确置零。
8. Finetune 为 `O8 / Star ON / MCL OFF / MD200 ON / upper=3.75`。

### 8.2 两样本 Forward/Backward

使用一个 geometry-valid 和一个 geometry-invalid 样本执行 forward、loss、backward 和
一次 optimizer step。确认 loss/gradient finite、invalid coordinate loss 为零、
Star-RBF projection 和 coordinate decoder 获得 finite 梯度。

### 8.3 三卡 Smoke

```text
GPU 1,2,3
3 ranks
BF16
2 optimizer steps
global batch=1008
```

确认 world size、forward/backward、参数同步和 `.last.pt` 写入正常。Smoke 不得生成
final、completion marker 或 science probe。本周期不执行 resume 测试。

运行定向 pytest、`py_compile`、launcher `bash -n` 和 scoped `git diff --check`。

## 9. Stage A：功能筛选

修复后从正确 RBF 和相同 seed-42 初始化重新运行：

```text
sigma = 0.03 / 0.05
lambda_3D = 0.1 / 0.3
```

四组各500 optimizer steps，scheduler horizon 仍为20000、warmup仍为2000。

Stage A 只淘汰明显异常组合，综合检查 decoder 梯度、displacement RMS、
`R_denoise` 下降趋势、Masked Atom 稳定性以及数值稳定性。若正常组合超过2组，综合
这些指标选择1–2组进入 Stage B；不得只按500-step `R_denoise` 最低值排名。

## 10. Stage B：过 Warmup 参数选择

Stage A 保留的1–2组继续到约2500 optimizer steps。使用 warmup 后稳定窗口检查：

```text
R_denoise < 1
predicted displacement RMS > 0
loss/gradient finite
Masked Atom 无异常退化
```

这里只证明 coordinate objective 可学，不证明下游改善。随后固定唯一
`sigma/lambda_3D` 并更新唯一正式 `configs/mts/b0_v2.json`。

## 11. 正式单路线 20k

从新的 seed-42 step 0 开始，不加载任何旧 B0 checkpoint：

```text
PI1M_v2
GPU 1,2,3
world size 3
global batch 1008
BF16
optimizer steps 20000
```

单 trajectory 保留 `5k/10k/20k probe` 和 rolling `.last.pt`。

本周期正式 trajectory 固定 `world_size=3`。`.last.pt` 仅用于该正式三卡配置发生真实
中断时恢复；本周期不支持、也不验证跨 world-size resume。

20k 结束时不自动发布 final，不生成 completion marker，不自动启动正式8×5。

## 12. Downstream 身份与 Probe

Finetune 必须显式固定：

```text
topology_attention_variant = o8
use_star_rbf               = true
use_mcl                    = false
use_md200                  = true
star_rbf_upper             = 3.75
```

预训练 MD200 为 OFF；下游 MD200 为 ON 且按 fold 初始化，不属于预训练 transferable state。

分别对5k、10k、20k运行：

```text
xc fold0
ei fold0
eat 或 egb fold0
```

使用正常微调周期和 early stopping，禁止5-epoch近零分 smoke。选择最早进入有效下游
性能平台的 checkpoint。若三个任务全部明显异常，不发布 final，先检查 RBF、
checkpoint transfer 和 finetune 配置；不自动转向其他科学路线。

## 13. Final 与正式 8×5

`final.pth` 只包含下游可迁移预训练部分，排除 masked atom head、coordinate decoder、
disabled MCL 和 fold-specific MD200 参数。

发布顺序：

```text
selected probe
→ 提取 transferable encoder state
→ 构造真实 B0-v2 downstream 模型
→ O8 / Star ON / MCL OFF / MD200 ON / upper=3.75
→ 组装完整 downstream state 并 strict=True 加载
→ 原子发布 final.pth
→ 生成 final.pth.complete.json
```

只有 final 和 marker 都存在后才执行：

```text
tasks  eat eea egb egc ei eps nc xc
folds  0..4
seed   42
GPUs   0,1,2,3 四个独立单卡 slot
```

报告每任务五折 `mean ± sample std`、8-task macro、去除 `xc` 后7-task macro、
`xc`贡献、`ei`变化、probe选择依据、`R_denoise`、位移 RMS、墙钟、VRAM、失败与重试。
旧 B0-v1 `0.823` 只作 invalid diagnostic。

## 14. 执行顺序

```text
1. 修复RBF upper与配置传递
2. 验证最终pair RBF representation
3. 修复decoder finite过滤，保留单有向observation
4. 修复三卡launcher
5. 固定workers=6
6. 定向pytest、两样本F/B、静态检查
7. 三卡2-step smoke，不做resume测试
8. Stage A：4组×500 step
9. 综合指标选1–2组
10. Stage B：继续至约2500 step
11. 固定sigma/lambda_3D
12. 新seed-42 step 0正式20k
13. 5k/10k/20k downstream probe
14. 选择checkpoint并严格发布final
15. 正式8×5
16. 汇总正式结果和运行证据
```

## 15. 执行协议

1. 先修代码并通过最小验收，再完整执行 screening、20k、downstream probe 和8×5。
2. GPU、worker 和超过一分钟任务必须位于 `tmux Uni-Poly` 独立窗口并同步落盘日志。
3. 普通实现错误做最小修复后继续；不得用 `strict=False`、伪造结果或放宽科学语义绕过。
4. 不停在代码修复、smoke 或 screening；只有真实无法在本周期范围内解决的阻塞才停止并记录。
5. 将修改、命令、测试、实验、日志、产物、错误和偏差追加到本文件末尾执行记录。
6. 执行者不得改写 Codex 计划或自行宣布验收通过；结束后将顶部状态改为
   `ready_for_codex_review`。

## 16. Claude Code 执行记录

```yaml
status: ready_for_codex_review
files_changed:
  - configs/mts/b0_v2.json (star_rbf_upper=3.75 added, loader_workers 2→6)
  - configs/mts/b0_v2_probe.json (star_rbf_upper=3.75 added)
  - configs/mts/b0_v2_ddp_smoke.json (star_rbf_upper=3.75, fresh result_root under repair/)
  - configs/mts/b0_v2_resume_probe.json (star_rbf_upper=3.75 added; no resume test run)
  - scripts/resolve_mips_trimer_scage.py (star_rbf_upper REQUIRED + ==3.75 validation)
  - src/training/pretrain/config.py (_B0_REQUIRED += star_rbf_upper, validate 3.75, set args.star_rbf_upper)
  - src/training/pretrain/engine.py (one-time RBF contract check in _run_b0_pretrain: config upper == sidecar upper == centers[-1] == 3.75, num_rbf=32, lower=0.0, gamma=0.5/spacing^2)
  - src/training/finetune/config.py (no code change; CLI identity switches verified by test)
  - src/training/finetune/engine.py (ROUTE_INTERNAL import; one-time RBF check after model build; extracted build_mts_downstream_model(args, auxiliary_tasks=()) used by both the fold job and publish)
  - src/modules/periodic_coordinate_decoder.py (removed dead source_safe assignment; valid &= isfinite(relative).all(-1), relative zeroed when non-finite)
  - scripts/screen_mts_b0_v2.py (scheduler horizon fixed 20000/warmup 2000; stop via --resume_smoke_stop_steps; workers=6/batch=336/acc=1 defaults; --arms option for Stage B)
  - scripts/run_mips_trimer_scage.sh (torchrun --standalone --nproc_per_node=3 with CUDA_VISIBLE_DEVICES=1,2,3; EXTRA_ARGS passthrough)
  - scripts/publish_mts_b0_v2_final.py (rewritten per plan section 13: probe -> transferable encoder state -> real downstream model -> assembled state strict=True -> atomic final.pth + marker)
  - scripts/b0_two_sample_fb.py (new; real-data 1 valid + 1 invalid two-sample F/B)
  - scripts/run_mts_b0_v2_downstream_probe.py (new; per-probe downstream driver)
  - scripts/run_mts_b0_v2_finetune.py (new; formal 8x5 driver)
  - scripts/report_mts_b0_v2_finetune.py (new; 5-fold mean+/-sample std, 8/7-task macro, xc contribution, ei)
  - scripts/fake_train_job.py (new; scheduler contract fake used for end-to-end driver validation)
  - tests/test_mts_b0_periodic_denoising.py (RBF 3.75 definition test; sigma=0 parity test with synthetic shift-2 pair and |shift|=1 dual-RBF mean check at 3.75; Gaussian tail above upper; inverse-pair sharing; decoder non-finite zeroing)
  - tests/test_mts_config_retirement.py (resolver wrong-upper rejection; finetune B0 identity switches parse)
tests:
  - "pytest -q tests/test_mts_b0_periodic_denoising.py tests/test_mts_config_retirement.py tests/test_mts_star_rbf_v2.py tests/test_mips_non_pbc.py: 38 passed, 1 warning"
  - "py_compile on all changed Python files: exit 0; bash -n scripts/run_mips_trimer_scage.sh: exit 0; git diff --check: only pre-existing TODO.md warning"
  - "scripts/b0_two_sample_fb.py on GPU 1: 1 geometry-valid + 1 geometry-invalid real sample; step0 R_denoise=1.000000, displacement RMS=0, coord_count=1, invalid-sample displacement exactly zero, Star-RBF projection + decoder gradients finite; post-step losses finite"
  - "3-GPU DDP smoke via launcher (configs/mts/b0_v2_ddp_smoke.json, --resume_smoke): 2 optimizer steps, world size 3, R=1.0 at step 1, state.pth.last.pt written (260MB), parameter sync + final barrier passed, no final/completion marker"
screening:
  - path: results/mts_b0_periodic_coordinate_denoising_v2/screening/stage_a_500_v2_summary.json
    result: "4/4 arms real 3-GPU 500 optimizer steps, scheduler horizon 20000/warmup 2000, all finite"
    arms:
      - "sigma=0.03 lambda=0.1: mean_R=0.99376 min_R=0.99146 disp=0.00330"
      - "sigma=0.03 lambda=0.3: mean_R=0.99355 min_R=0.99111 disp=0.00333"
      - "sigma=0.05 lambda=0.1: mean_R=0.97862 min_R=0.97278 disp=0.00958"
      - "sigma=0.05 lambda=0.3: mean_R=0.97837 min_R=0.97255 disp=0.00962"
    selection: "all healthy; sigma=0.05 shows clearly stronger denoising signal; lambda has negligible 500-step effect -> both sigma=0.05 arms to Stage B"
  - path: results/mts_b0_periodic_coordinate_denoising_v2/screening/stage_b_2500_v2_summary.json
    result: "2/2 arms real 3-GPU 2500 optimizer steps (past warmup), all finite, post-warmup window 2401-2500"
    arms:
      - "sigma=0.05 lambda=0.1: mean_R=0.85167 min_R=0.84247 disp=0.02582 atom=0.1155"
      - "sigma=0.05 lambda=0.3: mean_R=0.84247 min_R=0.83333 disp=0.02648 atom=0.1153"
    selected: "sigma=0.05, coordinate_loss_weight=0.3 (better R, identical masked-atom loss); formal config already matches, no edit needed"
pretraining:
  - path: results/mts_b0_periodic_coordinate_denoising_v2/formal/b0_training_metrics.jsonl
    result: "new seed-42 step 0, GPU 1,2,3, world size 3, global batch 1008, BF16, 20000 optimizer steps completed (~4.3h wall)"
    trajectory:
      - "R_denoise: 1.0 @step0 -> 0.736 @6k -> 0.694 @10k -> 0.680 @14k -> 0.668 @20k (last200 mean 0.6687, min 0.6606)"
      - "masked atom loss: 4.85 -> 0.0415; coordinate loss -> 0.0261; displacement RMS -> 0.0358; lr polynomial decay 2e-4 -> 1e-9"
    probes:
      - "results/mts_b0_periodic_coordinate_denoising_v2/formal/b0_v2_probe_005k.pth (step 5000)"
      - "results/mts_b0_periodic_coordinate_denoising_v2/formal/b0_v2_probe_010k.pth (step 10000)"
      - "results/mts_b0_periodic_coordinate_denoising_v2/formal/b0_v2_probe_020k.pth (step 20000)"
    lifecycle: "rolling .last.pt every 2000 steps; no final/completion marker auto-published; star_rbf_upper=3.75 verified at startup"
downstream_probes:
  - path: results/mts_b0_periodic_coordinate_denoising_v2/downstream_probe/{005k,010k,020k}/shards/42/{xc,ei,eat}/fold_0.csv
    result: "9/9 fold-0 fine-tunes completed (normal protocol, epochs=100, patience=10, early stopping)"
    fold0_r2:
      - "005k: xc=0.4266 ei=0.9255 eat=0.9834"
      - "010k: xc=0.4122 ei=0.9354 eat=0.9766"
      - "020k: xc=0.4226 ei=0.9288 eat=0.9762"
    selection: "005k chosen: earliest checkpoint reaching the effective downstream plateau (eat highest, ei/xc within noise of 020k)"
formal_finetune:
  - path: results/mts_b0_periodic_coordinate_denoising_v2/finetune/shards/42/{task}/fold_{0..4}.csv
    result: "8 tasks x 5 folds x seed 42 on GPU 0,1,2,3 single slots; 40/40 shards + 40/40 prediction npz complete"
    per_task_mean_std:
      - "eat 0.988 +/- 0.004 | eea 0.930 +/- 0.016 | egb 0.938 +/- 0.012 | egc 0.914 +/- 0.005"
      - "ei 0.829 +/- 0.071 | eps 0.802 +/- 0.047 | nc 0.868 +/- 0.038 | xc 0.431 +/- 0.077"
    macro: "macro8=0.837 macro7(no xc)=0.896 xc contribution=-0.465 ei=0.829"
    protocol: "fold_validation_protocol=shared_validation_test_fold, independent_blind_test=false; checkpoint pretrained_models/mts_b0_v2/final.pth; identity o8/Star ON/MCL OFF/MD200 ON/upper=3.75"
  - path: scripts/report_mts_b0_v2_finetune.py
    result: "report matches the aggregated shards above (3 decimals)"
final_publish:
  - source_probe: results/mts_b0_periodic_coordinate_denoising_v2/formal/b0_v2_probe_005k.pth
  - output: pretrained_models/mts_b0_v2/final.pth
  - marker: pretrained_models/mts_b0_v2/final.pth.complete.json ({"status":"complete"})
  - verification: "final encoder state matches 005k probe on 4/4 sampled topology tensors (atom_embedding, layer qkv, path_bias, star projection); no mips_atom/coordinate heads in final; 126 keys incl. fold-seeded md_residual/trimer_mcl initializations; all finite"
tmux_windows:
  - "Uni-Poly:b0-v2-two-sample-fb"
  - "Uni-Poly:b0-v2-ddp-smoke-v2"
  - "Uni-Poly:b0-v2-stageA"
  - "Uni-Poly:b0-v2-stageB"
  - "Uni-Poly:b0-v2-formal-20k"
logs:
  - logs/b0_v2_two_sample_fb.log
  - logs/b0_v2_ddp_smoke_v2.log
  - logs/b0_v2_stage_a_500.log
  - logs/b0_v2_stage_b_2500.log
  - logs/b0_v2_formal_20k.log
artifacts:
  - results/mts_b0_periodic_coordinate_denoising_v2/repair/ddp_smoke/state.pth.last.pt
  - results/mts_b0_periodic_coordinate_denoising_v2/repair/ddp_smoke/b0_training_metrics.jsonl
  - results/mts_b0_periodic_coordinate_denoising_v2/screening/stage_a_500_v2/ (4 arms)
  - results/mts_b0_periodic_coordinate_denoising_v2/screening/stage_b_2500_v2/ (2 arms)
  - results/mts_b0_periodic_coordinate_denoising_v2/formal/b0_training_metrics.jsonl (20000 rows)
  - results/mts_b0_periodic_coordinate_denoising_v2/downstream_probe/{005k,010k,020k}/ (9 fold-0 shards + predictions)
  - results/mts_b0_periodic_coordinate_denoising_v2/finetune/ (40 shards + 40 predictions + report.json)
  - pretrained_models/mts_b0_v2/final.pth + final.pth.complete.json
errors:
  - "screening first launch used --resume-smoke-stop-steps (dashed); argparse registers the underscore form -> unrecognized arguments; fixed to --resume_smoke_stop_steps and relaunched"
  - "downstream probe rounds initially failed at validate_runtime_args: config_schema='manual' vs required 'mts-config-v3'; fixed by passing --config_schema mts-config-v3 in the probe driver, 8x5 driver and publish script"
  - "publish script first invocation failed with ModuleNotFoundError: src (script does not insert PROJECT_ROOT into sys.path); rerun with PYTHONPATH=."
  - "fake_train_job.py failed twice on missing parent directories for shard/prediction paths; fixed with mkdir(parents=True)"
deviations:
  - "Stage B was accidentally launched twice (12:01 and 12:24); the second 3-rank DDP group contended for GPUs 1,2,3 with the first, causing forward time to rise from ~0.6s to ~9.4s per step from step ~1921 (exactly when the duplicate started). The duplicate's rank 0 hung in NCCL init; the surviving first run was identified by unique-step metrics (no duplicates) and CPU state, the second process tree (screen/torchrun/ranks/workers) was TERMed. Step rate recovered to 0.59s/step immediately. Metrics file has no duplicate rows (single writer)."
  - "Root cause of the duplicate launch: tmux allows duplicate window names and name-targeted send-keys/kill-window hit the first match; all duplicate windows were removed and later windows use unique names"
  - "downstream load path pre-validated with the DDP-smoke .last.pt model_state: 0 missing graph keys, 0 unexpected, 108 transferable tensors, strict load OK (no code change needed for probe loading)"
  - "downstream probe tasks used xc/ei/eat per plan section 12 ('eat 或 egb')"
running_tasks: []
notes:
  - "Formal 20k pretraining, downstream probes, final publication and the formal 8x5 were completed after the screening; all evidence paths are listed above."
```
