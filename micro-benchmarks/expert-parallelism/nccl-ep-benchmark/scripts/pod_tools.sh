#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# pod_tools.sh - in-pod helpers for run_nccl_ep_efa.sh. RUN_DIR names the per-run directory.
#
#   pod_tools.sh preflight                      key=value facts about this pod
#   pod_tools.sh prepare <tag>                  create $RUN_DIR/out/<tag> (refuses to reuse it)
#   pod_tools.sh mpirun <tag> <np> <hosts> <cidr> <ssh-port> <timeout-s> <exe> [args...]
#   pod_tools.sh reap                           stop leftover ranks/daemons by exact PID
#   pod_tools.sh gpu                            "<compute apps> <MiB used> <GPU count>"
set -uo pipefail
: "${RUN_DIR:?RUN_DIR must name the per-run directory}"
OFI_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so

cmd_preflight() {
  echo "hostname=$(hostname)"
  echo "gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
  echo "gpu_compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
  echo "efa_devices=$(find /sys/class/infiniband -maxdepth 1 -name 'rdmap*' 2>/dev/null | wc -l)"
  echo "fi_info_efa_providers=$(/opt/amazon/efa/bin/fi_info -p efa -t FI_EP_RDM 2>/dev/null | grep -c 'provider: efa')"
  echo "zombies=$(ps -eo stat= | awk '/^Z/ {n++} END {print n+0}')"
  echo "libnccl_files=$(find / -xdev \( -path /proc -o -path /sys \) -prune -o -name 'libnccl.so*' -type f -print 2>/dev/null | sort | tr '\n' ' ')"
  echo "ofi_plugin_present=$([ -f "$OFI_PLUGIN" ] && echo 1 || echo 0)"
  echo "sshd_present=$([ -x /usr/sbin/sshd ] && echo 1 || echo 0)"
  if [ -f /opt/nccl-ep/BUILD-RECORD.txt ]; then sed 's/^/record./' /opt/nccl-ep/BUILD-RECORD.txt; else echo "record.verdict=MISSING"; fi
}

cmd_prepare() {
  local o="$RUN_DIR/out/${1:?tag}"
  [ -e "$o" ] && { echo "$o exists; never reused" >&2; exit 2; }
  mkdir -p "$o"
}

cmd_mpirun() {
  local tag="$1" np="$2" hosts="$3" cidr="$4" port="$5" tmo="$6"
  shift 6
  local o="$RUN_DIR/out/$tag"
  [ -d "$o" ] || { echo "$o missing (run prepare first)" >&2; exit 2; }
  [ -e "$o/mpirun.exit" ] && { echo "$o already ran" >&2; exit 2; }
  # Ranks started over ssh do not inherit the image ENV, so every setting travels with -x.
  local envs=(
    "PATH=/opt/nccl-ep/test/nccl_ep:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    "LD_LIBRARY_PATH=/opt/nccl/lib:/opt/nccl-ep/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/gdrcopy/lib:/usr/local/cuda/lib64"
    "NCCL_HOME=/opt/nccl" "NCCL_EP_HOME=/opt/nccl-ep" "CUDA_HOME=/usr/local/cuda"
    "NCCL_GIN_TYPE=2"
    "NCCL_NET_PLUGIN=$OFI_PLUGIN" "NCCL_GIN_PLUGIN=$OFI_PLUGIN"
    "FI_PROVIDER=efa" "FI_EFA_USE_DEVICE_RDMA=1"
    "NCCL_NVLS_ENABLE=0" "NCCL_SOCKET_IFNAME=^lo,docker"
    "NCCL_DEBUG=INFO" "NCCL_DEBUG_SUBSYS=INIT,ENV,NET"
    "NCCL_EP_DEBUG=${NCCL_EP_DEBUG:-0}" "NCCL_EP_JIT_LOG=1"
    "NCCL_EP_JIT_CACHE_DIR=${RUN_DIR%/*}/jit-cache"
    "PMIX_MCA_gds=hash"
  )
  local x=() kv
  for kv in "${envs[@]}"; do x+=(-x "$kv"); done
  mkdir -p "${RUN_DIR%/*}/jit-cache"
  printf '%s\n' "$@" > "$o/argv.txt"
  printf '%s\n' "${envs[@]}" > "$o/env.txt"
  printf 'np=%s\nhosts=%s\ncidr=%s\nstarted_utc=%s\n' "$np" "$hosts" "$cidr" "$(date -u +%FT%TZ)" > "$o/spec.txt"
  # MPI itself uses TCP only (pml ob1, btl tcp): the cm/ofi path would put MPI traffic on EFA
  # and into the hardware counters this run uses as transport evidence.
  env "${envs[@]}" timeout "$tmo" /opt/amazon/openmpi/bin/mpirun --allow-run-as-root --prefix /opt/amazon/openmpi \
    -np "$np" -H "$hosts" --bind-to none --map-by slot \
    --mca plm_rsh_agent ssh \
    --mca plm_rsh_args "-p $port -i $RUN_DIR/ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    --mca pml ob1 --mca btl self,vader,tcp --mca btl_tcp_if_include "$cidr" --mca oob_tcp_if_include "$cidr" \
    --output-filename "$o/mpi" "${x[@]}" "$@" > "$o/mpirun.stdout" 2> "$o/mpirun.stderr"
  local rc=$?
  echo "$rc" > "$o/mpirun.exit"
  date -u +%FT%TZ > "$o/finished.utc"
  echo "mpirun exit=$rc -> $o"
  return "$rc"
}

leftovers() {
  ps -eo pid=,stat=,comm= | awk '$2 !~ /^Z/ && ($3=="ep_test" || $3=="ep_bench" || $3=="orted" || $3=="mpirun" || $3=="prted") {print $1}' | tr '\n' ' '
}

cmd_reap() {
  local left killed
  left=$(leftovers)
  echo "leftover: ${left:-none}"
  [ -z "$left" ] && return 0
  # shellcheck disable=SC2086  # a list of PIDs
  kill $left 2>/dev/null
  sleep 5
  killed=$(leftovers)
  # shellcheck disable=SC2086
  [ -n "$killed" ] && kill -9 $killed 2>/dev/null
  echo "killed: ${killed:-none}"
}

cmd_gpu() {
  local apps mem
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || { echo NVIDIA-SMI-ERROR; return 0; }
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) || { echo NVIDIA-SMI-ERROR; return 0; }
  echo "$(printf '%s' "$apps" | grep -c .) $(printf '%s\n' "$mem" | awk '{s+=$1} END {print s+0}') $(printf '%s\n' "$mem" | grep -c .)"
}

case "${1:-}" in
  preflight) cmd_preflight ;;
  prepare) shift; cmd_prepare "$@" ;;
  mpirun) shift; cmd_mpirun "$@" ;;
  reap) cmd_reap ;;
  gpu) cmd_gpu ;;
  *) echo "usage: $0 preflight|prepare|mpirun|reap|gpu" >&2; exit 2 ;;
esac
