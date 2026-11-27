#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Fault, evidence and recovery commands for the target GPU node, run from the
# login node with your own sudo. The target is AIM344_TARGET_NODE in lab.env.
#
#   bash lab.sh status            the target's state in Slurm and the current incident
#   bash lab.sh collect <n>       run check n of the health suite on the target as root
#   bash lab.sh collect kernel    save the target's kernel log of this boot
#   bash lab.sh collect recovery  save each GPU's recovery action
#   bash lab.sh reset-efa         drain the target and reset one of its EFA devices
#                                 (job.sh incident 6 runs this beside the job)
#   bash lab.sh recover           reboot the target through Slurm and wait until it is idle
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$LAB_DIR/lab.env"
target=$AIM344_TARGET_NODE
collected=$HOME/aim344-results/collected
reason=aim344-device-recovery

# Runs a command on the target: as a step of your running allocation when you
# hold one there, otherwise as a job of its own with the node's GPUs. A drained
# node starts no new job, so that srun gives up after 30 seconds instead of
# waiting in the queue.
on_target() {
    local job
    local -a where=(--partition="$PARTITION" --gres="gpu:$GPUS_PER_NODE" --immediate=30)
    job=$(squeue -h -u "$USER" -w "$target" -t RUNNING -o %i | head -n 1)
    [[ -z $job ]] || where=(--jobid="$job" --overlap)
    srun "${where[@]}" --nodelist="$target" --nodes=1 --ntasks=1 --mpi=none "$@" < /dev/null
}

# Saves a command's output under ~/aim344-results/collected and prints the path.
save() {
    local record
    record=$collected/$1-$(date -u +%Y%m%dT%H%M%SZ).txt
    shift
    mkdir -p -- "$collected"
    "$@" 2>&1 | tee "$record" || true
    printf 'Saved: %s\n' "$record"
}

case ${1:-} in
    status)
        scontrol show node "$target" | grep -E 'NodeName|State=|Reason=|LastBusyTime|BootTime'
        printf 'Current incident: %s\n' "$(cat "$HOME/aim344-results/current-incident" 2>/dev/null || echo none)"
        ;;
    collect)
        case ${2:-} in
            kernel) save kernel on_target sudo journalctl -k -b --utc -o short-iso-precise --no-pager ;;
            recovery) save recovery on_target nvidia-smi --query-gpu=index,uuid,pci.bus_id,gpu_recovery_action --format=csv ;;
            [0-9])
                save "check-$2" on_target sudo env PATH="/opt/amazon/efa/bin:$PATH" \
                    HEALTHCHECK_EXPECTED_EFA_COUNT="$HEALTHCHECK_EXPECTED_EFA_COUNT" \
                    bash "$AIM344_SUITE_ENTRY" --check "$2" --verbose
                ;;
            *) echo 'Usage: bash lab.sh collect <check number>|kernel|recovery' >&2; exit 2 ;;
        esac
        ;;
    reset-efa)
        # Drain first, so that Slurm starts no new job on the node, then issue a
        # PCIe function-level reset of the EFA device that the training stand-in uses.
        sudo "$AIM344_SLURM_BIN/scontrol" update nodename="$target" state=drain reason="$reason"
        # shellcheck disable=SC2016
        on_target sudo sh -c 'cd "/sys/class/infiniband/$1/device" && echo flr > reset_method && echo 1 > reset' \
            sh "$AIM344_EFA_DEVICE"
        printf 'EFA device %s on %s was reset at %s.\n' "$AIM344_EFA_DEVICE" "$target" "$(date -u +%FT%TZ)"
        ;;
    recover)
        # The reboot clears the device state; nextstate=RESUME returns the node to
        # service when slurmd registers again. The checkpoint tmpfs is a systemd
        # mount unit (setup-node.sh), so it is mounted again at boot.
        sudo "$AIM344_SLURM_BIN/scontrol" reboot ASAP nextstate=RESUME reason="$reason" "$target"
        printf 'Reboot requested at %s; waiting for %s to return.\n' "$(date -u +%FT%TZ)" "$target"
        sleep 60
        until [[ $(sinfo -h -N -n "$target" -o %t) == idle ]]; do sleep 10; done
        printf '%s is idle again at %s.\n' "$target" "$(date -u +%FT%TZ)"
        ;;
    *) sed -n '4,13p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
