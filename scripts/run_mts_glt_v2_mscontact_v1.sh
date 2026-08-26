#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/Uni-Poly-Plus-master

python - <<'PY'
import json
from pathlib import Path
qc = json.loads(Path('results/mts_glt_v2/mscontact_v1/sidecar_qc.json').read_text())
for cohort, result in qc.items():
    failures = {key: value for key, value in result['violations'].items() if value}
    if failures:
        raise SystemExit(f'{cohort} spatial sidecar QC failed: {failures}')
matched = json.loads(Path('results/mts_glt_v2/mscontact_v1/step0_matched_init.json').read_text())
if not matched.get('pass'):
    raise SystemExit('S4/MS45 matched initialization failed')
PY

run_500() {
    local arm="$1"
    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 scripts/pretrain.py \
        --experiment_config "configs/mts/mscontact_v1/${arm}_500.json"
}

run_5k() {
    local arm="$1"
    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 scripts/pretrain.py \
        --experiment_config "configs/mts/mscontact_v1/${arm}_5k.json" \
        --resume_state "pretrained_models/mts_glt_v2/mscontact_v1/${arm}.pth.last.pt"
    python - "$arm" <<'PY'
from pathlib import Path
import shutil
import sys
arm = sys.argv[1]
source = Path(f'results/mts_glt_v2/mscontact_v1/{arm}/mts_glt_v2_probe_005k.pth')
target = Path(f'pretrained_models/mts_glt_v2/mscontact_v1/{arm}_005k.pth')
if not source.is_file():
    raise SystemExit(f'missing 5k probe: {source}')
if target.exists():
    raise SystemExit(f'refusing to overwrite published probe: {target}')
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(target.suffix + '.tmp')
shutil.copy2(source, temporary)
temporary.replace(target)
PY
}

run_500 s4
run_500 ms45

python - <<'PY'
import json
import math
from pathlib import Path

required = {
    "masked_atom_loss", "masked_line_loss", "infonce_loss",
    "mean_infonce_pool_size", "spatial_residual_norm", "spatial_to_o8_ratio",
    "spatial_core_norm", "spatial_outer_norm",
}
summary = {}
for arm in ("s4", "ms45"):
    path = Path(f"results/mts_glt_v2/mscontact_v1/{arm}/training_metrics.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or int(rows[-1].get("step", -1)) != 500:
        raise SystemExit(f"{arm} functional screen did not reach step 500")
    bad = [
        (row.get("step"), key) for row in rows for key in required
        if key not in row or not math.isfinite(float(row[key]))
    ]
    if bad:
        raise SystemExit(f"{arm} functional screen has non-finite/missing metrics: {bad[:5]}")
    summary[arm] = {key: float(rows[-1][key]) for key in sorted(required)}
    summary[arm]["total_loss"] = sum(
        float(rows[-1][key])
        for key in ("masked_atom_loss", "masked_line_loss", "infonce_loss")
    )
Path("results/mts_glt_v2/mscontact_v1/functional_screen_500.json").write_text(
    json.dumps({"pass": True, "arms": summary}, indent=2) + "\n"
)
PY

run_5k s4
run_5k ms45
