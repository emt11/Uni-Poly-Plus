#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/Uni-Poly-Plus-master

gate_path="results/mts_glt_v2/mscontact_v1/c5_mixed_v1/pretrain_gate.json"
result_root="results/mts_glt_v2/mscontact_v1/c5_mixed"
last_state="pretrained_models/mts_glt_v2/mscontact_v1/c5_mixed.pth.last.pt"
published="pretrained_models/mts_glt_v2/mscontact_v1/c5_mixed_005k.pth"

if [[ -e "${result_root}/training_metrics.jsonl" || -e "${last_state}" || -e "${published}" ]]; then
    echo "refusing to overwrite an existing C5-Mixed trajectory" >&2
    exit 1
fi

python scripts/check_mts_glt_v2_c5_mixed_gate.py --output "${gate_path}"
python - "${gate_path}" <<'PY'
import json
from pathlib import Path
import sys
gate = json.loads(Path(sys.argv[1]).read_text())
if not gate.get("pass"):
    raise SystemExit("C5-Mixed pretraining gate failed")
if gate["parameter_count_delta"] != 0:
    raise SystemExit("C5-Mixed/MS45 parameter count differs")
if any(
    value["relation_key_mismatch"] != 0
    for value in gate["relation_universe"].values()
):
    raise SystemExit("C5-Mixed/MS45 relation universe differs")
PY

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 scripts/pretrain.py \
    --experiment_config configs/mts/mscontact_v1/c5_mixed_500.json

python - <<'PY'
import json
import math
from pathlib import Path
path = Path("results/mts_glt_v2/mscontact_v1/c5_mixed/training_metrics.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if len(rows) != 500 or int(rows[-1].get("step", -1)) != 500:
    raise SystemExit("C5-Mixed functional screen did not reach step 500")
required = ("masked_atom_loss", "masked_line_loss", "infonce_loss")
if any(not math.isfinite(float(row[key])) for row in rows for key in required):
    raise SystemExit("C5-Mixed functional screen has non-finite loss")
Path("results/mts_glt_v2/mscontact_v1/c5_mixed_v1/functional_screen_500.json").write_text(
    json.dumps({"pass": True, "step": 500, "final": rows[-1]}, indent=2) + "\n"
)
PY

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 scripts/pretrain.py \
    --experiment_config configs/mts/mscontact_v1/c5_mixed_5k.json \
    --resume_state "${last_state}"

python - "${published}" <<'PY'
from pathlib import Path
import shutil
import sys
source = Path("results/mts_glt_v2/mscontact_v1/c5_mixed/mts_glt_v2_probe_005k.pth")
target = Path(sys.argv[1])
if not source.is_file():
    raise SystemExit(f"missing C5-Mixed 5k probe: {source}")
if target.exists():
    raise SystemExit(f"refusing to overwrite published probe: {target}")
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(target.suffix + ".tmp")
shutil.copy2(source, temporary)
temporary.replace(target)
PY
