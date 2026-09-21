#!/usr/bin/env bash
# Sequential step runner for the GLT-PH end-to-end cycle.
#
# Discipline inherited from the r6 repair: the real exit code is captured
# immediately after a step, the first failure stops the sequence (later steps
# are never started), the error path never writes ALL_DONE, and the status log
# records the exact command, tag, exit code and log path of every step.
#
# Usage:  STEPS_FILE=path/to/steps.tsv LOG_DIR=path/to/logs ./run_glt_ph_fusion.sh
# The steps file holds one step per line: "<tag><TAB><shell command>"; empty
# lines and lines starting with '#' are ignored.
set -u

STEPS_FILE=${STEPS_FILE:?STEPS_FILE must point at the tab-separated step list}
LOG_DIR=${LOG_DIR:?LOG_DIR must point at the log directory}
STATUS_NAME=${STATUS_NAME:-chain_status.log}

mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/$STATUS_NAME"
: > "$STATUS"
echo "START $(date -Iseconds) steps_file=$STEPS_FILE" >> "$STATUS"

while IFS=$'\t' read -r tag command; do
    [ -z "${tag:-}" ] && continue
    case "$tag" in \#*) continue ;; esac
    echo "BEGIN $(date -Iseconds) tag=$tag command=$command" >> "$STATUS"
    log="$LOG_DIR/$tag.log"
    bash -lc "$command" > "$log" 2>&1
    code=$?
    echo "[$(date -Iseconds)] EXIT tag=$tag code=$code log=$log" >> "$STATUS"
    if [ "$code" -ne 0 ]; then
        echo "STOPPED_AFTER_FAILURE $(date -Iseconds) tag=$tag code=$code" >> "$STATUS"
        exit "$code"
    fi
done < "$STEPS_FILE"

echo "ALL_DONE $(date -Iseconds)" >> "$STATUS"
