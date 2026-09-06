#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/lib/common.sh"
cd "$LAB_DIR/observability"
mode=${1:-login}
case "$mode" in
  login)
    : "${LOGIN_BIND_IP:?Set LOGIN_BIND_IP to the private login-node address}"
    : "${COMPUTE_NODES:?Set COMPUTE_NODES to two comma-separated resolvable node names}"
    export LOGIN_BIND_IP COMPUTE_NODES
    mkdir -p runtime
    if [[ ! -f runtime/grafana-password ]]; then
      python3 -c 'import secrets; print(secrets.token_urlsafe(24))' > runtime/grafana-password
      chmod 644 runtime/grafana-password
      chmod 700 runtime
    fi
    python3 - <<'PY'
import json,os
from pathlib import Path
nodes=os.environ['COMPUTE_NODES'].split(',')
if len(nodes)!=2: raise ValueError('exactly two node hostnames are required')
scrapes=[dict(job_name='pushgateway',honor_labels=True,static_configs=[dict(targets=['pushgateway:9091'])])]
for name,port in [('dcgm',9400),('node',9100),('efa',9109),('vllm',8000)]:
 scrapes.append(dict(job_name=name,static_configs=[dict(targets=[f'{node}:{port}'],labels=dict(node=node)) for node in nodes]))
Path('runtime/prometheus.yml').write_text(json.dumps({'global':{'scrape_interval':'5s'},'scrape_configs':scrapes},indent=2))
PY
    docker compose -p aim347-observability -f compose.yaml up -d
    ;;
  compute)
    export TEXTFILE_DIR=/var/lib/aim347/textfile
    sudo mkdir -p "$TEXTFILE_DIR"
    docker compose -p aim347-compute -f compute.yaml up -d
    # lctl must run on the host with its installed Lustre libraries.
    sudo systemctl is-active --quiet aim347-lustre || sudo systemd-run --unit=aim347-lustre --collect --property=Restart=on-failure \
      /usr/bin/python3 "$LAB_DIR/lib/counters.py" lustre --output "$TEXTFILE_DIR/lustre.prom"
    ;;
  down-login) docker compose -p aim347-observability -f compose.yaml down ;;
  down-compute)
    sudo systemctl stop aim347-lustre
    docker compose -p aim347-compute -f compute.yaml down
    ;;
  *) echo 'Usage: 0.deploy-observability.sh login|compute|down-login|down-compute' >&2; exit 2 ;;
esac
