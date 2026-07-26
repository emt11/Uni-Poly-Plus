#!/bin/bash
set -euo pipefail

# Keep one source of truth for SCAGE-MIPS architecture, PI1M_50k migration,
# torchrun, BF16 parity gating, Stage-1 and Stage-2 parameters.
PRETRAIN_ONLY=1 bash scripts/run.sh
