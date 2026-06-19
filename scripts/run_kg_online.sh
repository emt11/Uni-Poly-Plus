#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 使用在线文献检索和 Qwen 运行 Polymer KG pipeline。
# 使用方法：
#   export DASHSCOPE_API_KEY=your_bailian_api_key
#   export QWEN_MODEL=qwen-plus
#   bash scripts/run_kg_online.sh
#
# 可选：用环境变量临时覆盖默认参数：
#   INPUT=data/raw/smi_all.csv OUTPUT_DIR=kg_work MAX_REPEAT_UNITS=50 bash scripts/run_kg_online.sh

# 原始聚合物数据集。至少需要包含 smiles 列。
INPUT="${INPUT:-data/raw/smi_all.csv}"

# KG pipeline 的总输出目录：records、repeat_units、retrieval 文件、
# 文献抽取结果、KG triples，以及最终 KG embedding 都会写到这里。
OUTPUT_DIR="${OUTPUT_DIR:-kg_work}"

# 文献目录。在线检索下载的 OA 全文或 metadata-only TXT 会写到这里；
# local 模式也会从这里读取已有文献。
DOCUMENTS_DIR="${DOCUMENTS_DIR:-${OUTPUT_DIR}/documents}"

# 最多处理多少个去重后的 RepeatUnit。测试时用小值；
# 大规模运行时再调大。
MAX_REPEAT_UNITS="${MAX_REPEAT_UNITS:-50}"

# LLM 服务商。当前支持 qwen 和 deepseek。
PROVIDER="${PROVIDER:-qwen}"

# PolymerClass / alias 映射模式。正式流程只使用 llm。
MAPPING_MODE="${MAPPING_MODE:-llm}"

# 文献模式：
#   local  = 只解析 DOCUMENTS_DIR 中已有的本地文献
#   online = 在线检索文献，并解析下载文件或 metadata-only 文件
#   both   = 先在线检索，再解析 DOCUMENTS_DIR 中的全部文献
LITERATURE_MODE="${LITERATURE_MODE:-online}"

# 在线检索使用的公共 metadata 来源。
SOURCES="${SOURCES:-crossref,openalex,semantic_scholar,europe_pmc,arxiv}"

# 每个 query 在每个 source 中最多取多少条 metadata 结果。
MAX_RESULTS_PER_QUERY="${MAX_RESULTS_PER_QUERY:-10}"

# 每个 RepeatUnit 打分筛选后最多保留多少篇候选文章。
MAX_ARTICLES_PER_REPEAT_UNIT="${MAX_ARTICLES_PER_REPEAT_UNIT:-5}"

# 每个 RepeatUnit 最多下载多少篇全文。
MAX_DOWNLOADS_PER_REPEAT_UNIT="${MAX_DOWNLOADS_PER_REPEAT_UNIT:-3}"

# 整个运行过程的全文下载总上限。
MAX_TOTAL_DOWNLOADS="${MAX_TOTAL_DOWNLOADS:-100}"

# 只在 metadata 显示 open access 时下载全文。
REQUIRE_OA_FOR_DOWNLOAD="${REQUIRE_OA_FOR_DOWNLOAD:-true}"

# pipeline 停止阶段。embedding 表示跑完整个 KG pipeline。
STOP_AFTER="${STOP_AFTER:-embedding}"

# 送入 LLM extraction 的 candidate chunks 数量上限。大规模运行时可调大。
MAX_CANDIDATES="${MAX_CANDIDATES:-20}"

# TransE embedding 维度。训练 Uni-Poly 时必须和 --kg_embedding_dim 一致。
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"

# TransE 训练轮数。smoke test 可以设小一点。
EMBEDDING_EPOCHS="${EMBEDDING_EPOCHS:-20}"

# API / 网络请求超时时间，单位秒。
TIMEOUT="${TIMEOUT:-120}"

# LLM API 失败后的重试次数。
MAX_RETRIES="${MAX_RETRIES:-2}"

# API / 网络请求之间的暂停时间，用于降低限流压力。
SLEEP_SECONDS="${SLEEP_SECONDS:-1.0}"

# 是否覆盖已有 PolymerClass / alias mapping。
# false：如果 kg_work/polymer_class_candidates.jsonl 已存在，则停止，避免误覆盖真实 LLM 结果。
# true ：重新生成 mapping，适合你更换 provider/model 或想重跑时使用。
OVERWRITE_MAPPING="${OVERWRITE_MAPPING:-false}"

# 是否启用断点续跑。true 表示已有阶段产物存在时跳过对应阶段。
RESUME="${RESUME:-false}"

# 日志目录和日志文件。默认写到 OUTPUT_DIR/logs，并同时显示在终端。
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/kg_online_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "日志文件: ${LOG_FILE}"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"

if [[ "${PROVIDER}" == "qwen" && -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "ERROR: DASHSCOPE_API_KEY is not set. Run: export DASHSCOPE_API_KEY=your_key" >&2
  exit 1
fi

if [[ "${PROVIDER}" == "deepseek" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set. Run: export DEEPSEEK_API_KEY=your_key" >&2
  exit 1
fi

ARGS=(
  --input "${INPUT}"
  --documents_dir "${DOCUMENTS_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --max_repeat_units "${MAX_REPEAT_UNITS}"
  --provider "${PROVIDER}"
  --mapping_mode "${MAPPING_MODE}"
  --literature_mode "${LITERATURE_MODE}"
  --sources "${SOURCES}"
  --max_results_per_query "${MAX_RESULTS_PER_QUERY}"
  --max_articles_per_repeat_unit "${MAX_ARTICLES_PER_REPEAT_UNIT}"
  --max_downloads_per_repeat_unit "${MAX_DOWNLOADS_PER_REPEAT_UNIT}"
  --max_total_downloads "${MAX_TOTAL_DOWNLOADS}"
  --require_oa_for_download "${REQUIRE_OA_FOR_DOWNLOAD}"
  --stop_after "${STOP_AFTER}"
  --max_candidates "${MAX_CANDIDATES}"
  --embedding_dim "${EMBEDDING_DIM}"
  --embedding_epochs "${EMBEDDING_EPOCHS}"
  --timeout "${TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --sleep_seconds "${SLEEP_SECONDS}"
)

if [[ "${OVERWRITE_MAPPING}" == "true" ]]; then
  ARGS+=(--overwrite_mapping)
fi

if [[ "${RESUME}" == "true" ]]; then
  ARGS+=(--resume)
fi

echo "运行配置:"
echo "  INPUT=${INPUT}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  DOCUMENTS_DIR=${DOCUMENTS_DIR}"
echo "  MAX_REPEAT_UNITS=${MAX_REPEAT_UNITS}"
echo "  PROVIDER=${PROVIDER}"
echo "  LITERATURE_MODE=${LITERATURE_MODE}"
echo "  SOURCES=${SOURCES}"
echo "  MAX_TOTAL_DOWNLOADS=${MAX_TOTAL_DOWNLOADS}"
echo "  OVERWRITE_MAPPING=${OVERWRITE_MAPPING}"
echo "  RESUME=${RESUME}"
echo "执行命令:"
printf '  python scripts/kg/run_pilot.py'
printf ' %q' "${ARGS[@]}"
printf '\n'

python scripts/kg/run_pilot.py "${ARGS[@]}"

echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
