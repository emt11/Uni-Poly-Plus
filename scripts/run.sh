#!/bin/bash
set -euo pipefail

# Strict production dispatcher.  MIPS-Trimer-SCAGE (MTS) is the only active
# route.  Retired route names are deliberately not enumerated here: they must
# fail through the generic unsupported-route branch and cannot be mistaken for
# a supported compatibility mode.
BASELINE=${BASELINE:-MIPS-Trimer-SCAGE}
case "$BASELINE" in
  MIPS-Trimer-SCAGE|MTS)
    exec bash scripts/run_mts.sh
    ;;
  *)
    echo "Unsupported BASELINE=$BASELINE (the only supported route is MIPS-Trimer-SCAGE/MTS)." >&2
    exit 2
    ;;
esac
