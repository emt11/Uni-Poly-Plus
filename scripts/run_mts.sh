#!/bin/bash
set -euo pipefail

# Canonical MTS launcher.  The implementation remains in the reviewed
# launcher during the naming migration so frozen cache paths and behavior do
# not change; all public identity and output naming is supplied below.
export BASELINE=MIPS-Trimer-SCAGE
export EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-configs/mts/default.json}
export MTS_ROUTE_NAME=MIPS-Trimer-SCAGE
export MTS_ROUTE_SHORT_NAME=MTS

exec bash scripts/run_mips_trimer_scage.sh
