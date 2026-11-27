#!/usr/bin/env bash
# Show that the lab is prepared: both GPU nodes with their GPUs and EFA devices, FSx for Lustre,
# the training image, the data and the model, the node counters, and the Grafana sign-in.
# It stops at the first check that fails. Usage: bash tools/check.sh
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
echo "AIM347 check ($AIM347_ENV)"
echo "== both GPU nodes: $GPUS_PER_NODE GPUs and 2 EFA devices on each"
# shellcheck disable=SC2016
srun "${two_nodes[@]}" --time=00:02:00 bash -c '
    gpus=$(nvidia-smi -L | wc -l) efa=$(ls /sys/class/infiniband | wc -l)
    echo "$(hostname): $gpus GPUs, $efa EFA devices"
    [[ $gpus == "$1" && $efa -ge 2 ]]' bash "$GPUS_PER_NODE"
echo '== FSx for Lustre: Module 6 writes about 205 GB of checkpoints'
df -h --output=target,fstype,size,avail "$AIM347_HOME"
echo '== the training image, the data and the model'
ls -l "$LAB_IMAGE" "$LLM_DATA_DIR"/tokens-*/manifest.json "$LLM_DATA_DIR/model/model.safetensors"
echo '== node counters in Prometheus on the login node'
nodes_with() {
    curl -fsk --get --data-urlencode "query=count($1)" https://localhost/prometheus/api/v1/query |
        python3 -c 'import json, sys; result = json.load(sys.stdin)["data"]["result"]; print(result[0]["value"][1] if result else 0)'
}
lustre=$(nodes_with 'aim347_lustre_collection_success == 1')
efa=$(nodes_with 'count by (instance_id) (node_amazonefa_tx_bytes)')
echo "Lustre counters from $lustre nodes, EFA counters from $efa nodes"
[[ $lustre == 2 && $efa == 2 ]]
echo '== Grafana'
curl -fsk -o /dev/null "${AIM347_GRAFANA_HEALTH_URL:-https://localhost/grafana/api/health}"
echo "$AIM347_GRAFANA_URL  user admin  password $(< "$AIM347_GRAFANA_PASSWORD_FILE")"
echo 'All checks passed.'
