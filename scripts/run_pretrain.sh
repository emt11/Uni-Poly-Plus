#!/bin/bash
set -euo pipefail

# Keep one source of truth for MIPS-Trimer-SCAGE (MTS), full PI1M_v2
# pretraining, torchrun, BF16 parity gating, Stage 1/2 parameters.
PRETRAIN_ONLY=1 bash scripts/run.sh
