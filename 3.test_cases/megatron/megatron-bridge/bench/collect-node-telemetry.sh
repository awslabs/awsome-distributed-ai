#!/usr/bin/env bash
set -euo pipefail
destination="${1:?destination directory}"
interval_seconds="${TELEMETRY_INTERVAL_SECONDS:-5}"
mkdir -p "${destination}"
find /sys/class/infiniband -type f \( -path '*/ports/*/counters/*' -o -path '*/ports/*/hw_counters/*' \) \
  -print0 | sort -z | xargs -0 -r awk '{print FILENAME "=" $0}' > "${destination}/efa-counters-start.txt" || true
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,nounits -l "${interval_seconds}" > "${destination}/nvidia-smi.csv" &
monitor_pid=$!
finish() {
  kill "${monitor_pid}" >/dev/null 2>&1 || true
  wait "${monitor_pid}" >/dev/null 2>&1 || true
  find /sys/class/infiniband -type f \( -path '*/ports/*/counters/*' -o -path '*/ports/*/hw_counters/*' \) \
    -print0 | sort -z | xargs -0 -r awk '{print FILENAME "=" $0}' > "${destination}/efa-counters-end.txt" || true
}
trap finish EXIT
while true; do
  date -u +%FT%TZ
  ps -eo pid,pcpu,pmem,comm --sort=-pcpu | head -20
  sleep "${interval_seconds}"
done > "${destination}/cpu-proxy.txt"
