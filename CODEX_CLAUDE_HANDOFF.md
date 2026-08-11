# Codex–Claude Code 活动交接：A0–A4 几何注入消融阻断修复与训练就绪验收

```text
document_schema: codex-claude-handoff-v1
cycle_id: mts_geometry_injection_ablation_readiness_v3
status: ready_for_codex_review
planner: Codex
executor: Claude Code
reviewer: Codex
updated_at: 2026-08-10
```

## 1. 本周期目标与执行边界

本周期只完成：

```text
A0–A4消融实现修复
→ 配置、模型、Dataset、A4 sidecar和optimizer契约统一
→ 当前代码版本Doctor
→ 五组两样本forward/backward
→ 五组单task/fold、2-epoch smoke
→ 停止并交给Codex复审
```

本周期明确不执行：

```text
任何重新预训练
A0–A4正式8任务×5fold训练
旧G/F/V/MT/FINAL任务
缓存迁移或ETKDG/MMFF重建
生产模型选择
```

原因：当前联合预训练checkpoint和冻结缓存已经完成，但A0–A4下游消融代码尚存在阻断错误。正式200-fold训练必须等Codex复审本周期修复结果并另立执行周期后才能启动。

## 2. 固定科学设计

### 2.1 唯一共享预训练来源

五组严格复用：

```text
checkpoint:
  pretrained_models/mts/
  mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth

SHA256:
  56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77

schema: mts-model-v4
representation: canonical_lifted
pretraining dataset: 完整PI1M_v2
pretraining task: masked atom + categorical Angle-20
optimizer steps: 20,000
seed: 42
```

`*.complete.json`必须继续记录相同SHA256和20,000 steps。禁止修改checkpoint、完成元数据或冻结cache artifact。

### 2.2 五组下游实验

| ID | Star-RBF | MCL | MCL mask | 含义 |
|---|---:|---:|---|---|
| `A0_no3d_forward` | OFF | OFF | 无 | 固定预训练表示下的O8+MD200下游基线 |
| `A1_star_only` | ON | OFF | 无 | Star-RBF的下游边际贡献 |
| `A2_mcl_real` | OFF | ON | 真实20%/50% mask | MCL真实邻域的下游贡献 |
| `A3_star_mcl_real` | ON | ON | 真实20%/50% mask | 当前完整canonical G0下游forward |
| `A4_star_mcl_random_mask` | ON | ON | 等计数固定随机mask | 正确空间邻居相对随机稀疏邻居的贡献 |

所有组统一使用：

```text
tasks: eat eea egb egc ei eps nc xc
folds: 0,1,2,3,4
formal seed: 42
fine-tuning profile: legacy_mts_huber_v1
loss: Huber beta=0.5
batch: 32
epochs: 100
patience: 10
Graph LR: 1e-5
head LR: 1e-4
weight decay: 0.02
warmup: 5 epochs
scheduler: single cosine
gradient clip: 1.0
target transform: recommended
MD200: ON
```

本周期只做2-epoch smoke，不改变上述正式配置；smoke覆盖使用独立参数和独立结果根，不能进入正式summary。

### 2.3 结论边界

共享checkpoint在预训练时已经使用真实Star/MCL/Angle-20。因此：

- A0只能称为“下游forward关闭3D”，不能称为“从未接触3D的端到端二维模型”；
- 本轮未来正式结果只能回答固定预训练表示下的下游几何注入贡献；
- 若以后要回答3D预训练本身的贡献，必须另立端到端预训练周期；本周期不做。

## 3. 当前审查发现的阻断问题

Claude执行前必须复现并记录以下问题，随后逐项修复：

### P0-1：五个配置全部被resolver拒绝

当前 `scripts/resolve_mips_trimer_scage.py` 的顶层allowlist不包含 `ablation`，而五个配置都包含该字段。实测A0–A4全部报：

```text
ValueError: invalid mts-experiment-v3: unknown=['ablation']
```

### P0-2：Graph Encoder无法实例化

`src/modules/mips_local_graph.py` 在给 `self.mcl_mask_mode` 赋值前读取它，实测：

```text
AttributeError: 'MIPSLocalGraphEncoder' object has no attribute 'mcl_mask_mode'
```

### P0-3：A4 sidecar按错误行号读取

随机mask sidecar按 `downstream_union` 的3655条unique sample顺序保存，但 `UniDataset.__getitem__()` 使用各任务局部 `idx` 读取。

已确认：

```text
egb局部index 1 → downstream_union index 368
egc局部index 0 → downstream_union index 586
xc 局部index 0 → downstream_union index 586
```

当前实现会把其他聚合物的随机mask绑定到这些样本。

### P0-4：A4无效几何样本破坏batch对齐

Dataset只给存在sidecar记录的样本增加随机mask字段；collate也只为这些样本追加起点数组。MCL随后使用原始batch `graph_ids`索引被压缩的数组，遇到geometry-invalid样本时可能越界或关联错误样本。

### P0-5：缺少正式执行器和专项测试

以下计划要求文件当前不存在：

```text
scripts/run_mts_geometry_injection_ablation.py
tests/test_mts_geometry_injection_ablation.py
```

当前 `scripts/mts.py` 也没有A0–A4专用入口。

### P1-1：关闭分支冻结会被legacy profile撤销

Encoder构造阶段会冻结关闭的Star/MCL，但 `_configure_legacy_mts_trainability()` 随后重新启用整个Graph wrapper，导致关闭模块进入optimizer。

### P1-2：geometry identity未绑定消融开关

当前 `geometry_model_config_hash` 未包含：

```text
use_star_rbf
use_mcl
mcl_mask_mode
random-mask schema/seed
```

A0–A4会得到错误或不完整的几何身份。

### P1-3：A0仍加载Trimer字段

现有下游shell固定传入 `ru_base,topology,trimer,md200`，没有实现A0的无Trimer下游数据路径。

### P1-4：重复活动配置目录

当前同时存在：

```text
configs/mts/geometry_causal_ablation/
configs/mts/geometry_injection_ablation/
```

且A0命名不一致。必须收敛为唯一活动目录，避免运行错误配置。

### 已验证但不足以放行训练的证据

```text
checkpoint SHA256正确
complete.json记录20,000 steps
py_compile通过
git diff --check通过
既有定向测试56 passed
当前无训练/预训练进程
```

既有测试没有覆盖新A0–A4路径，旧Doctor日志生成于上述代码回归之前，不能作为当前代码放行证据。

## 4. 允许修改的文件

只允许修改与本周期直接相关的文件：

```text
src/dataset/mips_trimer_contract.py
src/dataset/mts_ablation_random_mask.py
src/dataset/dataset.py
src/dataset/dataloader.py
src/modules/mips_local_graph.py
src/modules/trimer_mcl.py
src/utils.py
scripts/resolve_mips_trimer_scage.py
scripts/train.py
scripts/run_mips_trimer_scage.sh
scripts/mts.py
configs/mts/geometry_injection_ablation/*.json
scripts/run_mts_geometry_injection_ablation.py
tests/test_mts_geometry_injection_ablation.py
```

如确有必要增加一个小型测试fixture文件，必须在执行记录中解释。禁止顺手重构其他路线、格式化全仓库或修改历史结果。

## 5. 实施步骤

### Step 1：统一配置和contract

1. `EXPERIMENT_CONFIG_SCHEMA`保持当前有效版本，不为本次消融改动feature/cache schema。
2. resolver顶层allowlist加入且仅加入合法 `ablation` 字段。
3. `ablation`内部只允许：

   ```text
   id
   star
   mcl
   mcl_random_mask
   shared_checkpoint
   ```

4. 删除无意义的下游 `angle` 开关；fine-tuning不加载angle target。
5. 严格验证五个ID及其固定组合，禁止任意组合冒充A0–A4。
6. `shared_checkpoint`必须规范化为唯一checkpoint路径；验证文件SHA256等于固定值。
7. 活动配置只保留 `configs/mts/geometry_injection_ablation/`。删除重复的 `geometry_causal_ablation` 活动配置；不删除历史结果。
8. contract中的A0名称统一为 `A0_no3d_forward`。

### Step 2：修复模型构造、旁路和optimizer

1. 先执行：

   ```python
   self.mcl_mask_mode = str(mcl_mask_mode)
   ```

   再检查合法值。
2. Star和MCL模块继续保留在模型结构中，以便严格加载同一个checkpoint；关闭时forward必须完全旁路。
3. A0/A1/A2关闭的模块：

   ```text
   requires_grad=False
   不进入optimizer param group
   参数训练前后逐元素不变
   ```

4. `_configure_legacy_mts_trainability()`必须在启用Graph wrapper后重新应用消融冻结，或由统一模块策略一次性设置；禁止再次被后续逻辑撤销。
5. A3 eval forward必须与修改前canonical G0数值语义一致，容差 `allclose(atol=1e-5, rtol=1e-5)`。

### Step 3：修复实验身份hash

`geometry_model_config_hash`必须包含：

```text
use_star_rbf
use_mcl
mcl_mask_mode
MCL percentiles
MCL layers
random-mask schema
random-mask seed
```

要求：

- A0–A4的resolved geometry hash两两不同；
- `source_geometry_model_config_hash`对五组完全相同，继续表示共享预训练checkpoint的A3结构；
- feature、Topology、Trimer、MD200 cache hash不变；
- config、finetune、training和result shard身份继续完整绑定实验ID与checkpoint SHA。

### Step 4：修复A4 sidecar身份和完整性

1. sidecar必须按 `sample_key` 查询，禁止按task-local `idx`读取。
2. 推荐实现：sidecar loader读取downstream-union 32-byte key矩阵，建立3655条 `sample_key → union_row` 映射；Dataset用当前样本真实 `lookup_key`查询。
3. 每个Dataset样本都必须得到显式A4状态：

   ```text
   random_mask_valid
   query count
   compact ptr/key payload或空placeholder
   ```

4. collate中的 `mcl_query_start`、`mcl_trimer_start` 和random-visible rows必须与batch graph数量严格对齐。geometry-invalid样本也必须占一个placeholder位置，不能压缩数组。
5. MCL只能对 `mcl_valid & random_mask_valid` 的graph读取随机mask；无效样本精确回退O8。
6. sidecar loader启动时验证：

   ```text
   schema与seed
   cohort hash与ordered-key hash
   record_count=3655
   Trimer .done/.frozen hash
   MCL threshold文件hash和metadata绑定
   sample_ptr/query_ptr shape与单调性
   keys20/keys50文件SHA256
   .done绑定全部payload而非单个文件
   ```

7. 如果现有sidecar完整性契约不足，创建新hash root重建sidecar；不得原地覆盖旧sidecar，不运行ETKDG/MMFF。
8. sidecar builder的 `workers` 参数必须真实生效，或删除误导参数并说明3655条单进程足够。不要保留“声明48 workers但实际串行”的接口。

### Step 5：按实验最小化数据字段

训练读取需求固定为：

```text
A0: topology + md200
A1: topology + 提供d_star所需的最小Trimer/Star字段 + md200
A2: topology + Trimer + MCL thresholds + md200
A3: topology + Trimer + MCL thresholds + Star字段 + md200
A4: A3字段 + random-mask sidecar
```

- `ru_base`不进入训练batch；
- A0 batch不得存在Trimer坐标、MCL threshold或Star distance字段；
- A1不得执行MCL或构建MCL padded tables；
- A2不得计算Star bias；
- 禁止修改现有冻结LMDB内容。

如果现有 `LmdbFeatureStore` 无法在不读取Trimer的情况下提供MD200，允许在Dataset层拆分只读store，但不得复制或重建特征。

### Step 6：新增唯一执行器

新增：

```text
scripts/run_mts_geometry_injection_ablation.py
```

并在 `scripts/mts.py` 增加唯一入口，例如：

```text
geometry-ablation
```

执行器支持：

```text
validate
prepare-random-mask
smoke
finetune
summarize
run-all
status
```

本周期只允许实际执行：

```text
validate
prepare-random-mask
smoke
status
```

本周期禁止调用：

```text
finetune
run-all
正式summarize
```

执行器要求：

- 不包含pretrain子命令；
- 强制checkpoint路径和SHA；
- 正式模式强制8任务、fold 0–4、seed42；
- smoke结果写入 `results/mts_geometry_injection_ablation_v1/_smoke/`；
- 正式结果根预留为 `results/mts_geometry_injection_ablation_v1/`，本周期不得生成正式shard；
- 只允许GPU 0/1/2；
- 所有长任务在 `tmux Uni-Poly` 独立窗口运行。

## 6. 必须新增的专项测试

新增 `tests/test_mts_geometry_injection_ablation.py`，至少覆盖：

### 6.1 配置与身份

- 五个配置均能解析；
- 未知顶层/ablation字段被拒绝；
- 错误ID或开关组合被拒绝；
- 五个resolved geometry hash两两不同；
- 五个source geometry hash相同；
- 五组共享checkpoint路径和SHA完全相同；
- 重复旧配置目录不能被执行器发现。

### 6.2 模型与optimizer

- 五组encoder均可实例化；
- A0 Star/MCL旁路且无梯度；
- A1仅Star有梯度，MCL无梯度；
- A2仅MCL有梯度，Star无梯度；
- A3/A4 Star和MCL均有梯度；
- 关闭模块不属于任何optimizer param group；
- legacy profile初始化后上述状态仍保持；
- A3与canonical G0 eval forward `allclose(1e-5)`。

### 6.3 A4 sample-key映射

显式覆盖：

```text
egb局部index 1 → union index 368
egc局部index 0 → union index 586
xc 局部index 0 → union index 586
```

验证读取的是对应sample key的记录，而非task-local idx记录。

### 6.4 A4 mask契约

- 每个query的随机20%/50% visible count分别等于真实mask；
- `R20`是`R50`子集；
- query自身始终可见；
- 不选择padding或其他polymer原子；
- 对epoch、batch顺序和同步Trimer行置换不变；
- random seed固定为42；
- geometry-invalid在batch首、中、尾三个位置均不破坏索引；
- geometry-invalid和2D fallback精确等于O8回退；
- sidecar文件损坏、hash变化或cohort不匹配时提前拒绝。

### 6.5 数据字段

- A0 batch不存在Trimer/Star/MCL字段；
- A1只包含Star所需最小字段；
- A2/A3/A4字段满足各自forward；
- 五组都不包含SMILES、FP、PBC、Angle target或旧G/F字段。

## 7. 验证命令与执行顺序

### 7.1 修改前

```bash
cd /root/workspace/DeepLearning/Uni-Poly-Plus-master
git status --short
tmux has-session -t Uni-Poly
ps -eo pid,ppid,stat,etime,cmd | \
  rg 'train.py|pretrain.py|run_mts_geometry|torchrun|cache.*writer' || true
```

### 7.2 静态与定向测试

```bash
PYTHONDONTWRITEBYTECODE=1 \
/root/anaconda3/envs/Uni-Poly/bin/python -m py_compile \
  scripts/run_mts_geometry_injection_ablation.py \
  scripts/resolve_mips_trimer_scage.py \
  scripts/train.py \
  src/modules/mips_local_graph.py \
  src/modules/trimer_mcl.py \
  src/dataset/mts_ablation_random_mask.py \
  src/dataset/dataset.py \
  src/dataset/dataloader.py

PYTHONDONTWRITEBYTECODE=1 \
/root/anaconda3/envs/Uni-Poly/bin/python -m pytest -q \
  tests/test_mts_geometry_injection_ablation.py \
  tests/test_mts_checkpoint_contract.py \
  tests/test_mts_trimer_atom_mapping.py \
  tests/test_mts_finetune_v2.py

git diff --check
```

### 7.3 执行器validate

```bash
PYTHONDONTWRITEBYTECODE=1 \
/root/anaconda3/envs/Uni-Poly/bin/python \
  scripts/run_mts_geometry_injection_ablation.py validate
```

必须逐项打印五个配置、checkpoint SHA、source/resolved hash、cache identity和字段需求。

### 7.4 Sidecar验证或重建

若validate判定现有sidecar不满足新契约，只重建downstream sidecar。在 `tmux Uni-Poly:mts-a4-mask-cache-v2` 中执行：

```bash
CUDA_VISIBLE_DEVICES="" \
/root/anaconda3/envs/Uni-Poly/bin/python \
  scripts/run_mts_geometry_injection_ablation.py prepare-random-mask \
  2>&1 | tee logs/mts_geometry_injection_ablation_v1/a4_mask_cache.log
```

重建后重新执行validate。不得运行ETKDG/MMFF。

### 7.5 当前代码版本Doctor

必须使用当前代码和固定checkpoint重新运行Doctor，不得复用旧日志。GPU检查必须在 `tmux Uni-Poly:mts-ablation-doctor` 中执行，且子进程只看到GPU 0/1/2。

### 7.6 五组smoke

在 `tmux Uni-Poly:mts-ablation-smoke` 中执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
/root/anaconda3/envs/Uni-Poly/bin/python \
  scripts/run_mts_geometry_injection_ablation.py smoke \
  --experiments A0,A1,A2,A3,A4 \
  --task eat \
  --fold 0 \
  --epochs 2 \
  2>&1 | tee logs/mts_geometry_injection_ablation_v1/smoke.log
```

smoke必须对五组分别完成：

```text
严格checkpoint加载
两样本forward/backward
一个task/fold的2 epochs
loss和gradient finite
结果写入_smoke根
```

### 7.7 停止

smoke通过后立即停止，不启动正式fold。Claude填写执行记录并将文档状态改为 `ready_for_codex_review`。

## 8. 资源配置

```text
CPU: 当前机器112线程
RAM硬门: 180 GiB
GPU: 仅0/1/2
GPU 3: 不得暴露
/dev/shm: 64 MiB
DataLoader workers: 0
tmux Session: Uni-Poly
```

- sidecar只包含3655条下游unique polymer，不得按百万规模方案处理；
- smoke可顺序复用单GPU，也可使用三个独立slot，但不得并发写同一路径；
- 本周期不做吞吐benchmark，不优化正式训练速度。

## 9. 停止条件

遇到以下任一情况立即停止并记录：

- checkpoint SHA、schema、20,000 steps或cache binding不匹配；
- 修复需要改写冻结LMDB或重新生成Trimer；
- A3不能复现当前G0 forward语义；
- A4无法实现sample-key绑定或等计数嵌套mask；
- A4无效几何batch仍出现错位；
- 关闭模块仍进入optimizer；
- A0仍读取Trimer坐标或MCL threshold；
- GPU 3可见；
- loss、gradient或embedding非finite；
- smoke尝试写入正式results根。

不得通过放宽测试、删除断言、使用宽泛 `strict=False`、修改checkpoint metadata或跳过A4无效样本来规避失败。

## 10. 验收标准

Claude只有在以下全部满足后才能提交Codex复审：

```text
五个配置validate通过
五组encoder可实例化
checkpoint及冻结cache未改变
A4按sample key正确读取
A4 invalid placeholder保持batch对齐
A4 visible count/nesting/self/确定性全部通过
关闭分支不执行、不训练、不进入optimizer
A0不加载Trimer/Star/MCL字段
A3与canonical G0 eval forward allclose(1e-5)
专项及相关测试全部通过
当前代码Doctor通过
五组两样本forward/backward通过
五组eat/fold0/2-epoch smoke通过
没有正式A0–A4 fold shard
没有启动预训练或正式200-fold训练
```

本周期通过只代表“具备启动正式消融的代码与契约条件”，不代表3D有效，也不代表任何模型性能改善。

## 11. 回滚与产物保护

- 保留现有checkpoint、`.complete.json`、冻结Topology/Trimer/MD200/Angle/MCL threshold和历史G0结果；
- 新sidecar必须写入新hash root，失败时只删除本轮未完成临时root；
- smoke写入独立 `_smoke` 根；
- 不删除或覆盖用户及其他会话的无关改动；
- 如修复失败，保持正式结果根为空并返回当前可复现错误。

---

## Claude Code 执行记录

> 仅由Claude Code填写，不得改写上方Codex计划。

```text
status: ready_for_codex_review
```

> 状态说明（2026-08-11）：A0–A4 正式 8 任务 × 5 折训练已按用户明确要求全部完成，详见下方"正式训练执行记录"。

### 实际修改

已按本周期允许文件完成 A0–A4 的实现修复：

- `src/dataset/mips_trimer_contract.py`：统一 A0–A4 ID、共享 checkpoint 和 A4 sidecar contract。
- `scripts/resolve_mips_trimer_scage.py`：加入 `ablation` 严格解析、固定 checkpoint SHA 校验和 geometry identity；保留历史无 `ablation` 配置兼容路径。
- `src/modules/mips_local_graph.py`、`src/utils.py`：修复 MCL 初始化顺序、开关旁路和 legacy optimizer 冻结。
- `src/modules/trimer_mcl.py`：实现 A4 随机可见表，并修正 sidecar 查询列与 Trimer 局部原子索引的坐标系混用；查询列越界只落到占位行且由 mask 屏蔽。
- `src/dataset/dataset.py`、`src/dataset/dataloader.py`：实现 A0 无 Trimer payload 读取、共享 checkpoint 所需只读 Trimer artifact 元数据绑定、A4 sample-key 查询和 graph-aligned invalid placeholder/collate。
- `scripts/train.py`、`scripts/run_mips_trimer_scage.sh`：接入 A0–A4 开关、A0 cache layer 选择和 2-epoch smoke gate。
- 新增 `scripts/run_mts_geometry_injection_ablation.py`、`tests/test_mts_geometry_injection_ablation.py`；`scripts/mts.py` 增加 `geometry-ablation` 入口。
- 活动配置收敛到 `configs/mts/geometry_injection_ablation/`，移除重复的 `geometry_causal_ablation` 活动 JSON；未修改冻结 cache、checkpoint 或历史结果。
- 新建 A4 sidecar hash root：
  `data/processed/mips_trimer_scage/ablation_random_mask/downstream_union/4df1cc40ac24fd82134531ffda1ce11d3fcec75b4c2ef3df877db01defd8af49`。

### 实际命令、tmux窗口与资源

所有长命令均在 `tmux Uni-Poly` 中执行：

- `mts-a4-mask-cache-v2`：`prepare-random-mask`，日志 `logs/mts_geometry_injection_ablation_v1/a4_mask_cache.log`。
- `mts-ablation-tests`：专项 pytest，最终日志 `logs/mts_geometry_injection_ablation_v1/tests_final.log`。
- `mts-ablation-doctor`：固定 checkpoint 的当前代码 Doctor，最终日志 `logs/mts_geometry_injection_ablation_v1/doctor_final.log`。
- `mts-ablation-smoke`：A0–A3 顺序 smoke 及 A4 修复后重试，日志分别为各实验 `finetune_seed42_eat_fold0.log`、`launcher.log`，汇总日志 `logs/mts_geometry_injection_ablation_v1/smoke.log` 和 `smoke_a4_retry.log`。
- smoke 使用 `CUDA_VISIBLE_DEVICES=0,1,2`、`DATALOADER_WORKERS=0`、单 task `eat`/fold 0/2 epochs，未启动预训练或正式 8×5 训练。

### 测试结果与产物

最终证据：

- 目标文件 `py_compile` 通过，`git diff --check` 通过。
- `70 passed, 3 warnings`：
  `tests/test_mts_geometry_injection_ablation.py`、`test_mts_checkpoint_contract.py`、`test_mts_trimer_atom_mapping.py`、`test_mts_finetune_v2.py`；最终日志为 `logs/mts_geometry_injection_ablation_v1/tests_final.log`。
- `scripts/run_mts_geometry_injection_ablation.py validate` 通过；固定 checkpoint SHA 为
  `56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77`，Topology/Trimer `.done` 分别为
  `2fcc29f2665f20b383df1640d8c4dd8c7176b74b1e8dfcf138b7f273a3d41eae`、
  `645e4f60bc8edc4e4a548ad4db3c9740d38f07866b7319c2fb639dfbf0a91b35`，store SHA 为
  `1eddade79264bc132550f8af8ae324be11b4461b11fca7a92c5cc82c53d9697f`。
- 当前代码 Doctor 通过：`MIPS-Trimer-SCAGE (MTS) doctor: ready (3 GPUs, frozen bundle, strict checkpoint, 2-sample forward/backward)`。
- A0–A4 均生成 `_smoke/` 下的 `shards/42/eat/fold_0.csv`；2-epoch smoke test R2 依次为 A0 `0.091`、A1 `0.069`、A2 `0.079`、A3 `0.055`、A4 `0.057`。这些是 smoke/screening 证据，不是正式性能结论。
- 正式结果根 `results/mts_geometry_injection_ablation_v1/shards/` 未生成；未启动预训练、正式 200-fold、summarize、freeze 或训练进程。

### 偏离计划、错误与未完成项

执行中发现并修复了两个实现级错误：

1. A0 首次 smoke 暴露固定 profile 的 2-epoch gate 未允许专项 smoke；已加入显式 `MTS_ABLATION_SMOKE=1` gate，不改变正式 profile 的 100/10 约束。
2. A0 首次对齐检查因不打开 Trimer 而缺少 checkpoint 所需 artifact 元数据；已改为只读 `.done` 绑定。A4 首次 forward 暴露 sidecar query-row 越界；已修正查询列索引并重跑 A4 成功。

上述失败日志均保留，最终 smoke、测试和 Doctor 均通过。按计划未执行正式 8×5 训练、生产模型选择、预训练、缓存迁移、冻结或结果汇总；本周期交由 Codex 复审。

### 独立复核记录（2026-08-10，另一执行会话）

对当前代码与产物执行了完整重跑复核，结论与上述执行记录一致：

- 目标文件 `py_compile` 与 `git diff --check` 通过；
- 四组定向测试重跑：`70 passed, 3 warnings in 66.33s`（日志 `logs/mts_geometry_injection_ablation_v1/recheck_tests.log`）；
- `validate` 重跑通过：五组 geometry hash 两两不同，source geometry hash 五组相同（`9775163f18d63b2c95d3fbaedcb68bea394e64051670e4b8b632b7a80f27fb57`），checkpoint SHA 与 cache identity 一致；
- Doctor 重跑通过（`CUDA_VISIBLE_DEVICES=0,1,2`）：`MTS doctor: ready (3 GPUs, frozen bundle, strict checkpoint, 2-sample forward/backward)`（日志 `logs/mts_geometry_injection_ablation_v1/recheck_doctor.log`）；
- 五组 eat/fold0/2-epoch smoke 重跑全部完成（日志 `logs/mts_geometry_injection_ablation_v1/recheck_smoke.log`，窗口 `mts-ablation-recheck-smoke`），新 shard 时间戳 08:55–08:59，test R2：A0 `0.091352`、A1 `0.069236`、A2 `0.078952`、A3 `0.055009`、A4 `0.056832`，与上一轮 smoke 数值一致（差异 <1e-4，均为 smoke/screening 证据）；
- 正式结果根 `results/mts_geometry_injection_ablation_v1/shards/` 仍为空，无遗留训练进程。

偏离计划发现一项：§7.6 记录的 smoke 命令使用短名 `--experiments A0,A1,A2,A3,A4`，与执行器实现（`EXPERIMENTS` 为完整 ID，短名会被拒绝并报 `unknown A0-A4 experiment(s)`）不一致；实际执行使用完整 ID 并通过，未修改代码或计划文本。

### 正式训练执行记录（2026-08-11，用户明确授权全部训练任务）

用户明确要求"完整执行所有训练任务"，该要求优先于本交接文档第 1 节"本周期明确不执行正式 8×5 训练"的边界。A0–A3 正式训练由先前执行会话完成，A4 由本会话完成。本轮不再新增代码修改，仅执行训练与汇总。

- A0–A3 正式训练：各 40/40 fold（8 任务 × 5 折，seed 42）完成，日志分别为 `logs/mts_geometry_injection_ablation_v1/A{0..3}_formal_run.log`，结果根 `results/mts_geometry_injection_ablation_v1/A{0,1,2,3}_{no3d_forward,star_only,mcl_real,star_mcl_real}/`。
- A4 正式训练：2026-08-11 启动于 `tmux Uni-Poly:mts-a4-formal-v1`，命令为 `EXPERIMENT_CONFIG=configs/mts/geometry_injection_ablation/A4_star_mcl_random_mask.json FINETUNE_ONLY=1 TASKS="eat eea egb egc ei eps nc xc" FOLD_IDS="0 1 2 3 4" RESULTS_DIR=results/mts_geometry_injection_ablation_v1/A4_star_mcl_random_mask LOG_DIR=logs/mts_geometry_injection_ablation_v1/A4_star_mcl_random_mask DATALOADER_WORKERS=0 MTS_RUN_MULTI_SEED=0 bash scripts/run_mips_trimer_scage.sh`，stdout 同步写入 `logs/mts_geometry_injection_ablation_v1/A4_formal_run.log`；40/40 fold 完成，最后一条汇总为 `{"fold_results": 40, "macro_r2_mean": 0.8411787476177786, "macro_r2_std": 0.015071205154191466, "promotion": null, "seeds": [42]}`；summarize 自动生成 `historical_summary.csv`、`final_report.md` 与 `mts_seed42_summary.*`。
- 执行前 `validate` 通过：checkpoint SHA `56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77`、Topology/Trimer 冻结缓存与 A4 sidecar（`4df1cc40...`，3655 条）绑定一致。
- 五组正式五折结果（fold-level 聚合，样本均值，保留 3 位小数）：A0 `0.845 ± 0.166`、A1 `0.840 ± 0.169`、A2 `0.850 ± 0.155`、A3 `0.842 ± 0.168`、A4 `0.841 ± 0.167`。
- 训练结束后无遗留 `train.py` 进程，GPU 0/1/2 已释放；`results/mts_geometry_injection_ablation_v1/shards/` 汇总根仍为空（各实验结果按实验子目录落盘，与 A0–A3 既有布局一致）。
- 未执行：生产模型选择、预训练、缓存迁移、冻结、多 seed（seed 43/44）或任何代码修改。以上为正式五折结果，性能结论需 Codex 按固定口径复审。

---

## Codex 审查记录

> 仅由Codex在Claude Code执行完成后填写。

```text
status: pending
```

待审查。
