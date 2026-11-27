#!/usr/bin/env bash
# Check the nodes, GPUs, EFA devices, FSx, image, data, monitoring and Grafana sign-in.
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 0 ]] || { echo 'Usage: bash tools/check.sh' >&2; exit 2; }
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
sinfo -p "$PARTITION" -N -h -o '%N %t' > "$work/sinfo" 2>&1 || true
srun "${two_nodes[@]}" --time=00:02:00 --immediate=120 python3 "$lab/lib/workshop.py" node-probe \
    > "$work/nodes" 2> "$work/nodes.err" || true
status=0
python3 "$lab/lib/workshop.py" check-report --sinfo "$work/sinfo" --nodes "$work/nodes" || status=$?
if [[ -s $work/nodes.err && $status != 0 ]]; then
    echo 'The node check printed:'
    sed 's/^/  /' "$work/nodes.err"
fi
exit "$status"
