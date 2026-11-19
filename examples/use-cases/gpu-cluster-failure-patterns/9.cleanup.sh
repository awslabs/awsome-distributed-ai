#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Run only after all workload/result collection steps, before leaving the
# dedicated allocation. Retains images, measured results, and every historical
# root except the torch roots of this account's finished allocations.
set -euo pipefail
[[ $# -eq 0 ]] || { echo "Usage: bash ${0##*/} (it takes no arguments)" >&2; exit 2; }
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$LAB_DIR/common.sh"
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_name
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_env SLURM_OVERLAP
: "${SLURM_JOB_ID:?Run before exiting salloc}"
[[ $SLURM_JOB_ID =~ ^[1-9][0-9]*$ && $EUID -ne 0 ]] || {
    echo 'Cleanup requires an unprivileged current numeric job ID.' >&2; exit 1;
}
set_allocation_cpus
# Ask Slurm, not an inherited user variable, whose live allocation this is.
job=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ " $job " == *" JobId=$SLURM_JOB_ID "* &&
   " $job " == *" UserId=$(id -un)($EUID) "* &&
   " $job " == *" JobState=RUNNING "* && " $job " == *" NumNodes=2 "* ]] || {
    echo 'Cleanup refused: allocation owner/state mismatch.' >&2; exit 1;
}
# Occupy the whole allocation without --overlap: a still-running preparation,
# collective or storage step prevents cleanup, including after a client timeout.
# Do not wait indefinitely for a failed step; retain roots and report failure.
# shellcheck disable=SC2016
SLURM_MPI_TYPE=none srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --cpus-per-task="$AIM344_CPUS_PER_NODE" --exclusive --immediate=10 \
    --mpi=none --cpu-bind=none bash -c '
set -euo pipefail
[[ $SLURM_JOB_ID =~ ^[1-9][0-9]*$ && $EUID -ne 0 ]]
# This deployed-site cleanup supports the current data path only. A future
# site path change must update this exact path in the same approved rollout.
# Never scan or fall back to historical trees when the site mapping changes.
data=${ENROOT_DATA_PATH:-/tmp/enroot/data/user-$EUID}
[[ $data == /tmp/enroot/data/user-$EUID ]] || {
    echo "Cleanup refused: unsupported data path" >&2; exit 1;
}
[[ -e $data || -L $data ]] || exit 0
[[ -d $data && ! -L $data && $(realpath -e -- "$data") == "$data" &&
   $(stat -c %u -- "$data") == "$EUID" &&
   $(stat -c %a -- "$data") == 700 ]] || {
    echo "Cleanup refused: untrusted data directory" >&2; exit 1;
}
export ENROOT_DATA_PATH=$data
for label in "aim344_$SLURM_JOB_ID" "aim344_torch_$SLURM_JOB_ID" \
             "aim344_check5_$SLURM_JOB_ID" "aim344_verify_$SLURM_JOB_ID"; do
    # Pyxis supports either user or job container scope; exact names only.
    for name in "pyxis_$label" "pyxis_${SLURM_JOB_ID}_$label"; do
        root=$data/$name
        [[ -e $root || -L $root ]] || continue
        [[ -d $root && ! -L $root && $(stat -c %u -- "$root") == "$EUID" ]] || {
            echo "Cleanup refused: untrusted root $name" >&2; exit 1;
        }
        enroot remove -f "$name"
        [[ ! -e $root && ! -L $root ]] || {
            echo "Cleanup failed: retained root $name" >&2; exit 1;
        }
        printf "current-job container removed: %s\n" "$name"
    done
done
# The torch roots of this account'"'"'s earlier allocations on this node. The
# DataLoader and device incidents end without this cleanup (the participant
# leaves the allocation, or the job ends in the fault), so a later cleanup on the
# node removes their roots once Slurm reports that job finished, when its
# processes are gone. A root whose job is still active, or about which Slurm
# gives no answer, stays for a later cleanup. squeue prints the three fields
# asked for and nothing else, so text in a job name cannot pass for its state.
# A root that could be removed and was not makes the cleanup fail, after the
# other roots have been handled.
user=$(id -un)
failed=0
for root in "$data"/pyxis_aim344_torch_* "$data"/pyxis_*_aim344_torch_*; do
    [[ -e $root || -L $root ]] || continue
    name=${root##*/}
    id=${name##*_}
    [[ $id =~ ^[1-9][0-9]*$ && $id != "$SLURM_JOB_ID" ]] || continue
    [[ $name == "pyxis_aim344_torch_$id" || $name == "pyxis_${id}_aim344_torch_$id" ]] || continue
    if answer=$(squeue -h -t all -j "$id" -o "%i|%u|%T" 2>&1); then
        IFS="|" read -r job owner state rest <<< "$answer"
        [[ $answer != *$'"'"'\n'"'"'* && $job == "$id" && $owner == "$user" && -z $rest ]] || continue
        case $state in
            COMPLETED|CANCELLED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE) ;;
            *) continue ;;
        esac
    else
        # Slurm forgets a finished job after MinJobAge.
        [[ $answer == *"Invalid job id specified"* ]] || continue
    fi
    [[ -d $root && ! -L $root && $(stat -c %u -- "$root") == "$EUID" ]] || {
        echo "Cleanup skipped: untrusted earlier root $name" >&2; continue;
    }
    enroot remove -f "$name" || true
    if [[ -e $root || -L $root ]]; then
        echo "Cleanup failed: retained earlier root $name" >&2
        failed=1
    else
        printf "earlier-job container removed: %s\n" "$name"
    fi
done
exit "$failed"
'
