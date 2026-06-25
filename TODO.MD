# ✅ 已完成：实现 SMILES 级 Feature Cache

## 完成日期

2026-06-25

## 实现摘要

新增了 SMILES 级 feature cache，使得每个 unique SMILES 只处理一次（graph、fingerprint、geometry、token），任务 CSV 只读取 label 并从 cache 中取出预计算 features。

## 改动文件

### src/dataset/dataset.py

新增参数：
- `use_feature_cache=True` — 启用/禁用 feature cache
- `feature_source_dataset=None` — 构建 feature cache 的源 dataset（默认等于当前 dataset）
- `rebuild_feature_cache=False` — 强制重建 cache
- `max_smiles_length=None` — 手动指定 token 长度（None 时自动计算）
- `max_smiles_length_cap=256` — 自动计算时的长度上限

新增方法：
- `_init_with_feature_cache()` — feature cache 模式的初始化入口
- `_init_legacy()` — 旧 dataset-level cache 路径（`--disable_feature_cache` 时使用）
- `_compute_max_token_length_for_file()` — 内部 token 长度计算
- `_build_feature_cache()` — 从 source CSV 构建 SMILES 级 cache dict
- `_compute_smiles_features()` — 计算单个 SMILES 的所有结构特征（不含 y）
- `_build_labeled_data_list()` — 从 task CSV + feature cache 构建带标签的 data_list

关键行为：
- feature cache 以 raw SMILES 字符串为 key
- `_build_labeled_data_list()` 中通过 `copy.deepcopy()` 复制 feature，避免不同任务 y 相互覆盖
- cache miss 时现场计算并写回 in-memory cache，打印 warning 但不中断
- token 截断时打印 `[token] truncated N / M SMILES at max_length=...`
- 兼容 `--disable_feature_cache` 回退到旧逻辑

### scripts/pretrain.py

新增 CLI 参数：
- `--feature_source_dataset` (默认 = dataset_name)
- `--disable_feature_cache`
- `--rebuild_feature_cache`
- `--max_smiles_length`
- `--max_smiles_length_cap` (默认 256)

### scripts/train.py

新增同样 CLI 参数，其中 `feature_source_dataset` 默认为 `smi_all`。

### 缓存文件命名

```text
data/Schnet/feature_cache_smi_all_gin-dimer_tok256.pt
```

## 验证结果

- ✅ Python 编译检查通过
- ✅ CLI --help 显示新参数
- ✅ `--disable_feature_cache` 回退旧路径正常
- ✅ `--rebuild_feature_cache` 强制重建正常
