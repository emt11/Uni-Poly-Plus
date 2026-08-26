#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/Uni-Poly-Plus-master

config="configs/mts/mscontact_v1/formal_8x5_s4_c5.json"
results_root="results/mts_glt_v2/mscontact_v1/formal_8x5_s4_c5_v1"
logs_root="logs/mts_glt_v2/mscontact_v1/formal_8x5_s4_c5_v1"

python scripts/prepare_mts_glt_v2_mscontact_formal.py

python scripts/run_mts_glt_v2_mscontact_downstream.py \
    --arm s4 --config "${config}" --gpu-ids 0,1,2,3 \
    --results-root "${results_root}" --logs-root "${logs_root}"

python scripts/run_mts_glt_v2_mscontact_downstream.py \
    --arm c5_mixed --config "${config}" --gpu-ids 0,1,2,3 \
    --results-root "${results_root}" --logs-root "${logs_root}"
