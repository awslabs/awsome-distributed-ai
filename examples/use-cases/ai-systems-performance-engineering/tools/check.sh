#!/usr/bin/env bash
# Show that the lab is prepared: both GPU nodes with their GPUs and EFA devices, FSx for Lustre,
# the training image, the data and the model, the node counters, and the Grafana sign-in.
# It stops at the first command that fails. Usage: bash tools/check.sh
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
echo "AIM347 check ($AIM347_ENV)"
echo '== both GPU nodes: expect 8 GPUs and 2 EFA devices on each'
# shellcheck disable=SC2016
srun "${two_nodes[@]}" --time=00:02:00 bash -c \
    'echo "$(hostname): $(nvidia-smi -L | wc -l) GPUs, $(ls /sys/class/infiniband | wc -l) EFA devices"'
echo '== FSx for Lustre: Module 6 writes about 205 GB of checkpoints'
df -h --output=target,fstype,size,avail "$AIM347_HOME"
echo '== the training image, the data and the model'
ls -l "$LAB_IMAGE" "$LLM_DATA_DIR"/tokens-*/manifest.json "$LLM_DATA_DIR/model/model.safetensors"
echo '== node counters in Prometheus: expect 2 and 2'
nodes_with() {
    curl -fsk --get --data-urlencode "query=count($1)" https://localhost/prometheus/api/v1/query |
        python3 -c 'import json, sys; result = json.load(sys.stdin)["data"]["result"]; print(result[0]["value"][1] if result else 0)'
}
echo "Lustre counters from $(nodes_with 'aim347_lustre_collection_success == 1') nodes," \
    "EFA counters from $(nodes_with 'count by (instance_id) (node_amazonefa_tx_bytes)') nodes"
echo '== Grafana'
curl -fsk -o /dev/null "${AIM347_GRAFANA_HEALTH_URL:-https://localhost/grafana/api/health}"
echo "$AIM347_GRAFANA_URL  user admin  password $(< "$AIM347_GRAFANA_PASSWORD_FILE")"
echo 'All checks passed.'
