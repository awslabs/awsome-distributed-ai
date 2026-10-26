#!/usr/bin/env bash
# Run router_client.py inside the model container of an endpoint, so the walkthrough needs
# no load balancer, DNS, or certificates.
#
#   scripts/pod.sh <endpoint-name> drive /tmp/tickets_traffic.jsonl --seconds 900             # captured (port 8081)
#   scripts/pod.sh <endpoint-name> eval  /tmp/tickets_eval.jsonl  --port 8000 --labels v2     # not captured
#   scripts/pod.sh <endpoint-name> bench /tmp/tickets_traffic.jsonl --port 8000|8081          # latency
#
# The first call copies router_client.py and the ticket files into the pod's /tmp.
set -euo pipefail
EP="$1"; shift
HERE="$(cd "$(dirname "$0")/.." && pwd)"
POD=$(kubectl get pod -l app="$EP" -o jsonpath='{.items[0].metadata.name}')
if ! kubectl exec "$POD" -c "$EP" -- test -f /tmp/router_client.py 2>/dev/null; then
  kubectl cp "$HERE/scripts/router_client.py" "$POD:/tmp/router_client.py" -c "$EP"
  for f in "$HERE"/data/*.jsonl; do kubectl cp "$f" "$POD:/tmp/$(basename "$f")" -c "$EP"; done
fi
MODE="$1"; shift
exec kubectl exec "$POD" -c "$EP" -- python3 /tmp/router_client.py "$MODE" "$@" --model "$EP"
