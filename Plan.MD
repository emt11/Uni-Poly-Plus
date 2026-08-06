# MTS 单 Canonical RU 生产化与无重算缓存迁移计划

## 1. 目标、现状与执行边界

### 1.1 最终目标

将当前显式 `k-RU` O8 图正式替换为：

```text
一个 canonical RU 的 N 个原子状态
+ lifted periodic relation rows
→ O8 两跳注意力
→ canonical atom直接映射至Trimer中央RU
→ Trimer-MCL residual直接更新canonical节点
→ canonical atom mean pooling
→ MD200 residual
→ property
```

必须保持：

- lifted relation 中不同 `relative_ru_shift` 的重复边；
- SPD、single-path-node bias和对称Star-RBF；
- Trimer真实坐标、真实键、`d_star`和MD200；
- canonical模型与corrected explicit `k-RU`参考在关闭dropout时数值等价。

### 1.2 当前状态

已存在的只读历史产物：

```text
RU base:
f9b2d126bf25be3efe7ef86286366c05a23131a2b6c5a7758eb4ee32f752e0ee

旧显式Topology:
d1da740b029e7957fb13e61c451f36e4f351a6ab2afd3c5ee3c94a3b2c2097a6

旧Trimer:
6b63dd15a035d551fe4d3c5821f46fc333fdc96136df67ce6d176cc720907ff4

MD200:
faa0430c49682633ee38c4e6fc5f6c21ec0926766051475e0813fecf45851442

PI1M_v2 cohort:
0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049

downstream smi_all cohort:
ede5f8bc4ba277708c57787249ec85733b1a30a9c149ed9703ec55c80a843bb2
```

记录数量：

```text
PI1M_v2                 995,799
PI1M_v2 + downstream并集 999,224
MD200 downstream          3,655
```

当前canonical核心数值抽查误差约为 `10^-6`，但生产仍被以下问题阻断：

- 新Topology hash连带生成新Trimer hash，当前代码会重新执行百万次ETKDG/MMFF；
- 单记录迁移对非canonical P-SMILES原子顺序不安全；
- shared-boundary旧placeholder不能迁移；
- 完整测试仍有2项失败；
- 文档仍描述显式多RU模型；
- `doctor`因新cache bundle不存在而失败。

### 1.3 本次执行终点

独立Luna Max必须完成：

```text
代码整改
→ 500条preflight
→ 固定10k preflight
→ 999,224条Topology/Trimer迁移
→ 仅对确有必要的旧placeholder执行定向Trimer重建
→ Angle/MCL sidecar重绑定
→ full validation
→ freeze
→ doctor通过
```

本次禁止启动：

```text
20,000-step联合预训练
任何下游fold训练
G/V/MT/FINAL消融
```

执行代理开始修改前，必须先将本计划完整覆盖写入 `Plan.MD`。

---

## 2. 修改文件与生产契约

### 2.1 必须修改

核心与Schema：

- `src/dataset/mips_trimer_contract.py`
- `src/dataset/canonical_periodic.py`
- `src/dataset/graph_data.py`
- `src/dataset/__init__.py`
- `configs/mts/default.json`

缓存与迁移：

- `src/dataset/dataset.py`
- `src/dataset/lmdb_cache.py`
- `src/dataset/mips_cache_validation.py`
- `src/dataset/dataloader.py`
- `scripts/finalize_mips_trimer_cache.py`
- `scripts/doctor_mips_trimer_scage.py`
- `scripts/resolve_mips_trimer_scage.py`
- `scripts/mts.py`

模型与训练契约：

- `src/modules/mips_local_graph.py`
- `src/modules/trimer_mcl.py`
- `scripts/pretrain.py`
- `scripts/train.py`

测试与文档：

- `tests/test_mts_canonical_periodic.py`
- `tests/test_mips_lmdb_cache.py`
- `tests/test_mts_naming.py`
- `AGENT.MD`
- `PIPELINE.MD`
- `TODO.MD`

### 2.2 必须新增

```text
scripts/migrate_mts_canonical_cache.py
tests/reference_mts_explicit.py
tests/test_mts_canonical_cache_migration.py
```

### 2.3 固定Schema

```text
config                 mts-config-v3
feature                mts-canonical-periodic-feature-v1
Topology LMDB          mts-canonical-periodic-topology-lmdb-v1
LGA                     2
checkpoint              mts-model-v3
cache bundle            mips-trimer-scage-cache-bundle-v2
Trimer content          mips-trimer-scage-trimer-v6
Trimer LMDB             mips-trimer-scage-trimer-lmdb-v4
Angle cache             mts-trimer-bond-angle-cache-v1（新artifact hash）
RU base                 保持mips-trimer-scage-ru-base-v2
MD200                   保持mips-trimer-scage-md200-lmdb-v1
```

所有常量只能从 `mips_trimer_contract.py` 导出。删除 `canonical_periodic.py` 中重复定义的Schema常量。

### 2.4 旧实现处理

- 历史LMDB、checkpoint、结果文件不得删除、改写或迁移metadata。
- corrected explicit `k-RU`只保留于 `tests/reference_mts_explicit.py`。
- 从生产模块导出中移除explicit reference和迁移测试helper。
- `build_mips_local_structure`等旧构图函数可暂时保留源码，但生产入口、Dataset和模型不得调用。
- 旧feature/checkpoint schema必须在生产入口提前拒绝，不允许`strict=False`。

---

## 3. 实施步骤

### 3.1 Canonical topology与模型收敛

统一生产语义：

```text
node = canonical_atom_id
relation source = (canonical_atom_id, relative_ru_shift)
relation target = (canonical_atom_id, 0)
```

构图要求：

- 对每个target执行最大2-hop lifted BFS；
- internal bond连接 `(a,q) ↔ (b,q)`；
- polymer link连接 `(right,q) ↔ (left,q+1)`；
- 相同canonical source/target但shift不同的关系必须保留；
- `image_shift/path_shift`只作为结构诊断，不进入任何embedding；
- shared-boundary允许出现相同canonical节点的 `shift=-1/0/+1` 关系；
- 所有直接跨RU关系使用同一个对称polymer-link mask和同一个`d_star` RBF；
- `mips_repeat_factor`和boundary distance只保留为诊断字段。

模型要求：

- O8直接更新N个canonical节点；
- incoming softmax包含全部lifted relation行及其multiplicity；
- MCL不再scatter-mean显式copies；
- Trimer三个RU token均由canonical topology state按照`trimer_base_ru_atom_id`生成；
- 中央RU residual直接加回canonical节点；
- geometry-invalid逐元素精确回退O8；
- graph pooling只平均N个canonical节点。

### 3.2 原子身份安全迁移

禁止依赖canonical SMILES前后的RDKit原子顺序。

为每个样本建立：

```text
old_ru_atom_id
→ normalized_canonical_atom_id
→ trimer_base_ru_atom_id
→ central_trimer_atom_index
```

映射方法：

1. 解析原始RU和normalized RU；
2. 使用原子序数、形式电荷、芳香性、真实键型及两个attachment site建立图同构；
3. 多个合法automorphism时选择字典序最小映射；
4. 映射后验证：
   - 原子数一致；
   - atomic number逐原子一致；
   - RU内部真实键和键型一致；
   - attachment连接语义一致；
   - backbone mask一致；
   - Trimer中央RU原子与canonical RU一一对应；
5. 任一验证失败不得猜测映射。

Topology迁移：

- 新Topology始终从normalized RU构建lifted relations；
- 旧Graph有效时，将每个canonical atom的显式copy特征先验证完全一致，再按原子置换复制一行MIPS137/backbone；
- 同时用新原生MIPS137构造结果校验复制值；
- 旧Graph-unavailable/shared-boundary placeholder不得复制两原子占位特征，直接从RU base构建新的canonical二维Topology；
- Topology构建只使用2D/RDKit图操作。

### 3.3 无重算Trimer迁移

新增全量流式迁移器：

```bash
python scripts/migrate_mts_canonical_cache.py
```

CLI固定支持：

```text
preflight
migrate
validate
regenerate-obsolete
```

迁移器必须：

- 只读打开旧 `d1da...`、`6b63...` 和 `f9b2...`；
- 父进程单写新LMDB；
- 使用`uint8[N]`状态数组和流式任务，不建立百万条Python jobs list；
- 每128条或30秒提交事务；
- 支持`--resume`；
- 输出`.done`前验证记录数和metadata hash；
- 中断最多重算127条；
- 不修改旧LMDB。

对旧Trimer有效记录：

- 位级复制以下字段：
  ```text
  trimer_pos
  trimer_atomic_number
  trimer_edge_index
  trimer_bond_type
  trimer_ru_offset
  trimer_central_ru_mask
  energy/method/status
  star_3d_distance/asymmetry/valid
  geometry flags
  ```
- 按原子置换重写：
  ```text
  trimer_base_ru_atom_id
  trimer_central_atom_index
  mips_to_trimer_central_index
  ```
- 新mapping长度必须等于canonical节点数N；
- 每个mapping必须指向central RU中的同canonical atom。

对旧Trimer无效记录：

- 新Graph仍不可用：保留Trimer unavailable；
- 旧失败属于真实ETKDG/MMFF/超大分子失败：保留O8回退，不重新生成；
- 旧失败仅因过时Topology/shared-boundary placeholder，而新Graph有效：写入
  `needs_geometry_regeneration.jsonl`；
- identity mapping无法验证：写入同一manifest，禁止盲目迁移。

`regenerate-obsolete`只能处理上述manifest中的样本。禁止扫描全库重新运行ETKDG/MMFF。最终报告必须给出：

```text
reused_geometry_count
preserved_invalid_count
selective_regeneration_count
mapping_failure_count
ETKDG_call_count
```

### 3.4 Sidecar、冻结与切换

- 对迁移后的PI1M_v2 Trimer顺序扫描重新物化Angle sidecar；只计算真实键角，不运行ETKDG/MMFF。
- 重新物化`mcl_thresholds.npy`，绑定新Trimer artifact hash。
- MD200继续使用`faa0430c...`，不得重算。
- 新Topology和Trimer必须覆盖PI1M_v2与`smi_all`并集，共999,224条。
- full validation和freeze完成前，不得将新bundle标记为生产可用。
- 任一步失败时保留可恢复的新LMDB，但删除本轮临时`.frozen`和`store.json`。
- freeze成功后再次调用共享`verify_frozen_cache_bundle()`，最后才原子写入`store.json`。

### 3.5 Joint pretraining与checkpoint接口

只完成接口验证，不启动正式训练：

- canonical mask长度为N；
- 每个canonical atom只产生一个masked target；
- Trimer三个copy全部来自同一个masked canonical state；
- Angle objective保持现有真实Trimer键角语义；
- 旧`mts-model-v2`只允许在测试helper中作为固定权重；
- 生产resume和fine-tune只接受`mts-model-v3`；
- checkpoint绑定新Topology、Trimer、Angle artifact和cache bundle hash。

### 3.6 文档同步

`AGENT.MD`和`PIPELINE.MD`必须改为：

```text
canonical N-node O8
→ lifted periodic relations
→ direct canonical Trimer-MCL residual
```

删除：

```text
O8 node=(canonical_atom_id,mips_copy_id)
canonical scatter-mean
geometry residual广播到全部O8 copies
mts-config-v2
mips-trimer-scage-feature-v4生产说明
```

`TODO.MD`将“单RU模拟多RU”拆为：

```text
代码完成
500 preflight
10k preflight
全量迁移
freeze
doctor
```

只在对应产物真实完成后勾选。

---

## 4. 执行命令与测试

### 4.1 静态与单元测试

```bash
cd /root/workspace/DeepLearning/Uni-Poly-Plus-master

PY=/root/anaconda3/envs/Uni-Poly/bin/python

find src scripts tests -name '*.py' -print0 \
  | xargs -0 "$PY" -m py_compile

bash -n scripts/run.sh
bash -n scripts/run_pretrain.sh
bash -n scripts/run_train.sh
bash -n scripts/run_mts.sh
bash -n scripts/run_mips_trimer_scage.sh

git diff --check

PYTHONPATH=. "$PY" -m pytest -q \
  tests/test_mts_canonical_periodic.py \
  tests/test_mts_canonical_cache_migration.py \
  tests/test_mips_lmdb_cache.py \
  tests/test_mts_naming.py

PYTHONPATH=. "$PY" -m pytest -q
```

要求：

```text
0 failed
仅允许已明确标记原因的skip
git diff --check无输出
```

### 4.2 等价性测试

自动覆盖：

```text
k=1,2,3,4,7
*CC*
*CCO*
*C(*)C(=O)OCC(C)(C)C
*c1ccccc1*
支链及非canonical/attachment反转输入
```

必须验证：

- lifted BFS关系集合完整；
- 不同shift的重复canonical edge保留；
- shared-boundary存在非零shift self relation；
- 每个target/head incoming attention和为1；
- shift字段改变不改变模型输出；
- dropout关闭时，canonical与corrected explicit参考的：
  ```text
  attention logits/weights
  每层node state
  final node state
  Star-RBF
  MCL residual
  MD200后graph embedding
  regression prediction
  ```
  全部满足：
  ```text
  atol=1e-5
  rtol=1e-5
  ```

### 4.3 500条preflight

```bash
PYTHONPATH=. "$PY" scripts/migrate_mts_canonical_cache.py preflight \
  --cohort PI1M_preflight500 \
  --source-topology-hash d1da740b029e7957fb13e61c451f36e4f351a6ab2afd3c5ee3c94a3b2c2097a6 \
  --source-trimer-hash 6b63dd15a035d551fe4d3c5821f46fc333fdc96136df67ce6d176cc720907ff4 \
  --workers 8 \
  --resume
```

要求：

```text
500/500处理完成
mapping failure=0
正常有效Trimer的ETKDG调用=0
旧shared-boundary placeholder得到有效canonical Topology
迁移后valid Trimer坐标逐元素相同
```

### 4.4 固定10k preflight

```bash
PYTHONPATH=. "$PY" scripts/migrate_mts_canonical_cache.py preflight \
  --cohort PI1M_preflight10k \
  --source-topology-hash d1da740b029e7957fb13e61c451f36e4f351a6ab2afd3c5ee3c94a3b2c2097a6 \
  --source-trimer-hash 6b63dd15a035d551fe4d3c5821f46fc333fdc96136df67ce6d176cc720907ff4 \
  --workers 16 \
  --resume
```

要求：

```text
mapping failure=0
2D进入MCL=0
migrated valid geometry保持率=100%
预测全量迁移时间和空间写入报告
```

### 4.5 全量迁移

```bash
PYTHONPATH=. "$PY" scripts/migrate_mts_canonical_cache.py migrate \
  --cohorts PI1M_v2,smi_all \
  --source-topology-hash d1da740b029e7957fb13e61c451f36e4f351a6ab2afd3c5ee3c94a3b2c2097a6 \
  --source-trimer-hash 6b63dd15a035d551fe4d3c5821f46fc333fdc96136df67ce6d176cc720907ff4 \
  --workers 32 \
  --commit-size 128 \
  --resume
```

只对manifest执行定向几何重建：

```bash
PYTHONPATH=. "$PY" scripts/migrate_mts_canonical_cache.py regenerate-obsolete \
  --manifest results/mts_canonical_migration/needs_geometry_regeneration.jsonl \
  --workers 48 \
  --resume
```

完整验证：

```bash
PYTHONPATH=. "$PY" scripts/migrate_mts_canonical_cache.py validate \
  --cohorts PI1M_v2,smi_all \
  --full
```

冻结：

```bash
PYTHONPATH=. "$PY" scripts/mts.py finalize-cache \
  --validate-full \
  --audit-10k \
  --freeze
```

最终检查：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTHONPATH=. "$PY" scripts/mts.py doctor
```

不得运行任何pretrain或finetune命令。

---

## 5. 禁止事项与验收标准

### 5.1 禁止事项

执行代理不得：

- 修改、删除或重写 `f9b2...`、`d1da...`、`6b63...`、`faa043...`；
- 为全部PI1M_v2重新运行ETKDG/MMFF；
- 依赖canonical SMILES前后的原子顺序；
- 将`relative_ru_shift`变成可学习embedding；
- 将Star虚拟边写入Trimer真实键；
- 使用`strict=False`绕过checkpoint/cache不兼容；
- 将旧checkpoint metadata直接改成v3；
- 删除历史checkpoint、结果或cache；
- 恢复PBC、SMILES、FP、AP3D、FLAT4或其他路线；
- 顺手清理与本任务无关的脏worktree；
- 启动20k预训练或任何下游训练；
- 在测试失败时把失败标记为“预期失败”而继续freeze。

### 5.2 最终硬验收

代码：

```text
全部pytest通过
py_compile通过
bash -n通过
git diff --check通过
生产入口不调用显式k-RU builder
旧schema生产加载明确拒绝
```

缓存：

```text
RU base记录数          999,224
新Topology记录数       999,224
新Trimer记录数         999,224
MD200记录数              3,655
PI1M_v2 cohort记录数    995,799
mapping failure               0
2D进入MCL                     0
MCL valid / Graph available ≥90%
```

复用：

```text
所有可迁移valid Trimer坐标、真实键、能量和d_star位级相同
正常valid记录ETKDG/MMFF调用数=0
只有needs_geometry_regeneration manifest中的样本允许重新生成
旧缓存manifest、.done和内容SHA256保持不变
```

模型：

```text
canonical与corrected explicit误差≤1e-5
attachment反转输出不变
geometry-invalid严格等于canonical O8回退
有效MD200、Star-RBF、MCL均包含在等价测试中
两样本joint forward/backward loss和gradient finite
```

生产：

```text
新Topology/Trimer/Angle均存在.done、.frozen和validation
store.json通过二次验证
doctor输出ready
不存在活动cache writer或训练进程
```

输出报告：

```text
results/mts_canonical_migration/migration_summary.json
results/mts_canonical_migration/migration_summary.md
results/mts_canonical_migration/needs_geometry_regeneration.jsonl
```

报告必须包含源/目标hash、记录数、复用数、定向重建数、失败分布、耗时、磁盘增量、ETKDG调用数及全部验收项。

### 5.3 完成定义

只有同时满足以下条件，Luna Max才能声明完成：

```text
代码与文档同步
全部测试零失败
500/10k preflight通过
999,224条全量迁移完成
定向重建完成
full validation通过
新bundle冻结
doctor ready
未启动长期训练
```
