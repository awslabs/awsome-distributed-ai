#!/usr/bin/env bash
# Run as root on one GPU node: a 5 s interval for the PCS monitoring stack's
# EFA collector, and the companion's Lustre client counter in node-exporter's
# textfile directory. Safe to run again; it changes nothing the second time.
# Usage: bash facilitator/install-node-collectors.sh [--verify] [--timeout SECONDS]
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
systemd_dir=${AIM347_SYSTEMD_DIR:-/etc/systemd/system}
textfile_dir=${AIM347_TEXTFILE_DIR:-/var/lib/prometheus/node-exporter}
verify=0 timeout=300
usage() { echo 'Usage: bash facilitator/install-node-collectors.sh [--verify] [--timeout SECONDS]' >&2; exit 2; }
while (($#)); do
    case $1 in
        --verify) verify=1 ;;
        --timeout) [[ ${2:-} =~ ^[0-9]+$ ]] || usage; timeout=$2; shift ;;
        *) usage ;;
    esac
    shift
done
[[ $(id -u) == 0 ]] || { echo 'install-node-collectors: run as root' >&2; exit 2; }
host=$(hostname)

# 1. The upstream timer runs every 30 s; clear its interval, then set 5 s.
install -d -m 0755 "$systemd_dir/efa-metrics.timer.d"
printf '[Timer]\nOnUnitActiveSec=\nOnUnitActiveSec=5s\nAccuracySec=1s\n' > "$systemd_dir/efa-metrics.timer.d/aim347-5s.conf"
systemctl daemon-reload
systemctl try-restart efa-metrics.timer

# 2. The Lustre client counter writes lustre.prom every 5 s (lib/counters.py).
install -d -m 0755 "$textfile_dir"
if ! systemctl is-active --quiet aim347-lustre; then
    systemctl reset-failed aim347-lustre.service >/dev/null 2>&1 || true
    systemd-run --unit=aim347-lustre --collect --property=Restart=on-failure \
        /usr/bin/python3 "$lab/lib/counters.py" lustre --output "$textfile_dir/lustre.prom"
fi
if ((!verify)); then
    echo "install-node-collectors $host: efa-metrics.timer drop-in written; lustre counter started"
    exit 0
fi

# 3. Wait until the timer, the counter file and node-exporter all show them.
# The collection-success line is written on every pass; write counters appear
# only after this client has written to /fsx, so they are not waited for.
timer_ok() {
    systemctl is-active --quiet efa-metrics.timer &&
        systemctl show efa-metrics.timer -p TimersMonotonic | grep -Eq 'OnUnitActiveUSec=5s( |;|}|$)'
}
lustre_ok() { grep -qx 'aim347_lustre_collection_success 1' "$textfile_dir/lustre.prom" 2>/dev/null; }
exporter_ok() {
    local metrics
    metrics=$(curl -s http://localhost:9100/metrics) || return 1
    grep -qx 'aim347_lustre_collection_success 1' <<<"$metrics" && grep -Eq '^node_amazonefa_tx_bytes[{ ]' <<<"$metrics"
}
deadline=$((SECONDS + timeout))
while :; do
    waiting=
    timer_ok || waiting='efa-metrics.timer at 5 s'
    [[ -n $waiting ]] || lustre_ok || waiting="aim347_lustre_collection_success 1 in $textfile_dir/lustre.prom"
    [[ -n $waiting ]] || exporter_ok || waiting='node-exporter serving the Lustre success line and node_amazonefa_tx_bytes'
    if [[ -z $waiting ]]; then
        echo "install-node-collectors $host: efa-metrics.timer 5s ok; lustre counter ok; node-exporter serves both"
        exit 0
    fi
    if ((SECONDS >= deadline)); then
        echo "install-node-collectors $host: timed out after $timeout s waiting for $waiting" >&2
        exit 1
    fi
    sleep 1
done
