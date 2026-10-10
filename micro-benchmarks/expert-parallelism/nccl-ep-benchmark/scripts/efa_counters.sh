#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# efa_counters.sh <label> - snapshot the EFA hardware byte counters of every rdmap device on
# this node as one JSON line. total_bytes sums rdma_read_bytes, rdma_read_resp_bytes,
# rdma_write_bytes and rdma_write_recv_bytes; send/recv/tx/rx bytes are recorded, not summed.
# Exits 2 when no EFA device or no counter is readable. EFA_SYSFS overrides the sysfs root.
set -uo pipefail
LABEL="${1:?usage: efa_counters.sh <label>}"
ROOT="${EFA_SYSFS:-/sys/class/infiniband}"
# Some container /sys views omit hw_counters (the host has them). If a read-only hostPath /sys
# is mounted at /hostsys, use it instead (README troubleshooting).
if ! compgen -G "$ROOT/rdmap*/ports/1/hw_counters" > /dev/null && [ -d /hostsys/class/infiniband ]; then
  ROOT=/hostsys/class/infiniband
fi
SUMMED=(rdma_read_bytes rdma_read_resp_bytes rdma_write_bytes rdma_write_recv_bytes)
RECORDED=(send_bytes recv_bytes tx_bytes rx_bytes)
total=0
n=0
devs=""
for d in "$ROOT"/rdmap*; do
  hw="$d/ports/1/hw_counters"
  [ -d "$hw" ] || continue
  entries=""
  for c in "${SUMMED[@]}" "${RECORDED[@]}"; do
    [ -r "$hw/$c" ] || continue
    v=$(<"$hw/$c")
    [[ "$v" =~ ^[0-9]+$ ]] || continue
    entries+="${entries:+,}\"$c\":$v"
    case " ${SUMMED[*]} " in *" $c "*) total=$((total + v)) ;; esac
  done
  [ -n "$entries" ] || continue
  devs+="${devs:+,}\"$(basename "$d")\":{$entries}"
  n=$((n + 1))
done
printf '{"label":"%s","host":"%s","at_utc":"%s","devices":{%s},"n_devices":%d,"total_bytes":%d}\n' \
  "$LABEL" "$(hostname)" "$(date -u +%FT%T.%6NZ)" "$devs" "$n" "$total"
[ "$n" -gt 0 ] || exit 2
