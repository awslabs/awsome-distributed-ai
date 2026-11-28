#!/usr/bin/env bash
# Run stages of the training job, in order, as one batch job on both GPU nodes, and follow its log.
# Usage: bash tools/run.sh <stage> [<stage> ...]   (see tools/README.md)
# Ctrl-C or a closed browser tab stops only the following; the batch job keeps running.
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
python3 "$lab/lib/workshop.py" run --list "$@" > /dev/null
mkdir -p "$LLM_RESULTS_DIR/logs"
log=$LLM_RESULTS_DIR/logs/$(date -u +%Y%m%dT%H%M%SZ)-$1.log
job=$(sbatch --parsable --job-name=aim347 "${two_nodes[@]}" --output="$log" \
    --wrap="exec python3 -u $lab/lib/workshop.py run $*")
echo "Batch job $job, log $log"
echo "Ctrl-C stops only this following. Follow it again with: tail -f $log"
# tail ends when the job has left the queue; the EXIT trap ends the queue watcher after Ctrl-C.
while [[ -n $(squeue -h -j "$job" -t PENDING,RUNNING,COMPLETING 2> /dev/null) ]]; do sleep 2; done &
watcher=$!
trap 'kill "$watcher" 2> /dev/null || true' EXIT
tail -F --pid="$watcher" "$log" 2> /dev/null
