#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# The participant's job launcher for the six incidents.
#
#   bash job.sh notes        create your notes file
#   bash job.sh run          run the training stand-in, the epoch job and the all-reduce
#                            sweep on your pair
#   bash job.sh incident [N] start incident N and run its job (without N, the one after
#                            the incident started last)
#   bash job.sh rerun        run the current incident's job again, as the pair is now
#                            (with the options a page gives for that incident)
#   bash job.sh resume       undo what the incident changed and resume the job
#
# Every command except notes runs inside your two-node allocation (terminal A).
# Each incident is a fixed recipe in incident_job below. The number of the incident
# started last is kept in ~/aim344-results/current-incident, which rerun and resume read.
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

RESULTS_HOME=$HOME/aim344-results
CURRENT=$RESULTS_HOME/current-incident
# What the launcher's own steps print (plugin moves, checkpoint clean-up, the EFA reset).
lab_log=$RESULTS_HOME/lab.log
# The training stand-in runs 60 seconds. Incident 6 runs 120, which leaves time for
# the EFA reset after two progress records. A resumed job ends shortly after its
# first checkpoint past the step it resumed from (workload.py --end-after-save).
RUN_SECONDS=60
EFA_RESET_RUN_SECONDS=120
# The job whose answer differs needs only a few seconds of steps.
CORRUPTION_SECONDS=20
# Every job writes the stacks of a process still waiting 60 seconds after its
# timer starts. The launcher ends the DataLoader job 120 seconds after srun.
STACK_DUMP_SECONDS=60
DATALOADER_LIMIT_SECONDS=120
CHECKPOINT_EVERY=200
# The training stand-in: a broadcast, a cross-rank checksum comparison and a
# checked all-reduce per step, with the checkpoint writer.
STAND_IN=(device --checkpoint-every="$CHECKPOINT_EVERY" --replica-check
          --stack-dump-seconds="$STACK_DUMP_SECONDS")
# The epoch job: two epochs of four batches from a DataLoader whose workers are
# created anew for each epoch, one checked all-reduce and one progress record per batch.
EPOCH_JOB=(dataloader --start-method fork --epochs 2 --stack-dump-seconds="$STACK_DUMP_SECONDS")

usage() {
    cat >&2 <<'USAGE'
Usage: bash job.sh <command>

  notes        Create your notes file, ~/aim344-results/notes.md
  run          Run the training stand-in, the epoch job and the all-reduce sweep on your pair
  incident [N] Start incident N (1 to 6) and run its job; without N, the incident
               after the one you started last
  rerun        Run the current incident's job again, as your pair is now;
               add an option only where your page gives one
  resume       Undo what the current incident changed and resume the job

Run every command except notes inside your allocation, in terminal A.
USAGE
    exit 2
}

# ---------------------------------------------------------------- notes
# ---------------------------------------------------------------- notes
make_notes() {
    local notes=$RESULTS_HOME/notes.md
    mkdir -p -- "$RESULTS_HOME"
    if [[ -e $notes || -L $notes ]]; then
        printf 'Your notes file already exists and was left as it is: %s\n' "$notes"
        return 0
    fi
    local incident_fields case_fields
    incident_fields=$'Row of the triage table:\n\nPredictions:\n\nEarliest error, its line number and its rank:\n\nLater symptoms, and the ranks that printed them:\n\nHypothesis:\n\nAlternative, the records that make it less likely, and the records that leave it open:\n\nRecovery and why:\n\nTransfer question:\n'
    # Every incident page, and the cost page, end with a case of three questions.
    case_fields=$'1.\n\n2.\n\n3.\n'
    (
        set -o noclobber
        cat > "$notes" <<NOTES
# Your cluster

GPU count:

Stops per week (estimate):

# Triage table

| What you see | Who reported | First record to read | Check that separates the causes | Kind of recovery | Your healthy value |
|---|---|---|---|---|---|
| Nobody times out, but the job is slow | | | | | |
| One rank does not report | | | | | |
| Every rank reports at once | | | | | |
| Progress stops and nothing prints | | | | | |
| Everything passes, but an answer differs | | | | | |
| A rank's CUDA error comes first | | | | | |

# Healthy values

Transport:

Ranks that finished NCCL setup:

Checksums:

Checkpoint line:

Longest gap between progress lines:

End of epoch 1:

Sweep, transport:

Sweep, 2 GiB line:

Sweep, EFA byte changes:

Bus bandwidth at 128 MiB:

Bus bandwidth at the smaller size:

Wrong elements:

EFA devices that moved bytes:

Storage:

Warnings:

Pre-session Check 6 record:

# Incident 1

$incident_fields
## Case

$case_fields
# Incident 2

$incident_fields
## Case

$case_fields
# Incident 3

$incident_fields
## Case

$case_fields
# Incident 4

$incident_fields
## Case

$case_fields
# Incident 5

$incident_fields

Rank, node, GPU UUID and PCI address:

Kernel Xid and current GPU recovery action:

New process and checkpoint resume position:

Recovered checksums, progress and new errors:

## Card C

$case_fields
# Incident 6

Row of the triage table:

Predictions:

| Record | What it shows | Which part it names |
|---|---|---|
| Last progress line | | |
| Watchdog lines | | |
| NCCL WARN lines | | |
| Kernel log | | |
| Check 0 | | |
| Check 2 | | |
| Check 3 | | |
| Check 6 | | |

Hypothesis:

Alternative that remains:

Recovery and why:

Checks that must pass after the reboot:

What the verification job shows, and what it does not:

## Case

$case_fields
## Card A

$case_fields
## Card B

$case_fields
# Loss table

| Time | Event | UTC |
|---|---|---|
| t0 | Last progress line | |
| t1 | First watchdog line | |
| t2 | End of the job | |
| t3 | recover typed | |
| t4 | recover returned | |
| t5 | Verification job passed | |
| t6 | Training job running again | |

| Interval | From | To | Minutes |
|---|---|---|---|
| Detection | t0 | t1 | |
| Clean-up | t1 | t2 | |
| Decision and request | t2 | t3 | |
| Recovery | t3 | t4 | |
| Confirmation | t4 | t5 | |
| Resume | t5 | t6 | |
| Rework | The last checkpoint | The stop | |
| Incident 3 | | | |
| Incident 4 | | | |
| Incident 5 | | | |

## Case

$case_fields
NOTES
    )
    printf 'Created your notes file: %s\n' "$notes"
}

# ---------------------------------------------------------------- jobs
# One torch run. Prints the job log's path before and after the run, the time the
# job ended in UTC and, for an incident, which incident the job was.
torch_job() {
    local name=$1 phase=$2 rc=0
    shift 2
    AIM344_RUN_ID=$name-$(date -u +%Y%m%dT%H%M%SZ)
    printf 'Job log: %s\n' "$RESULTS_DIR/$AIM344_RUN_ID.log"
    run_torch "$phase" "$@" || rc=$?
    printf 'job_end_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'Job log: %s\n' "$RESULTS_DIR/$AIM344_RUN_ID.log"
    [[ -z ${incident:-} ]] || printf 'This job was incident %s.\n' "$incident"
    return "$rc"
}

sweep_job() {
    local name=$1 rc=0
    AIM344_RUN_ID=$name-$(date -u +%Y%m%dT%H%M%SZ)
    printf 'Job log: %s\n' "$RESULTS_DIR/$AIM344_RUN_ID.nccl.log"
    # The sweep runs in the NCCL container and uses every EFA device of each node.
    use_nccl_container
    (unset AIM344_EFA_IFACE; run_sweep job) || rc=$?
    printf 'job_end_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'Job log: %s\n' "$RESULTS_DIR/$AIM344_RUN_ID.nccl.log"
    [[ -z ${incident:-} ]] || printf 'This job was incident %s.\n' "$incident"
    return "$rc"
}

# A step of the launcher itself, whose output goes to the lab log.
lab_step() {
    printf '%s lab step: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$lab_log"
    "$@" >> "$lab_log" 2>&1 || { printf 'A lab step failed; read %s\n' "$lab_log" >&2; exit 1; }
}

# Runs one job, then removes the lab's containers from both GPU nodes, so that the
# hour's allocations do not fill the nodes' container storage. Returns the job's
# status. Only run and resume end this way: incident and rerun keep the container
# in which the incident made its change.
ending_job() {
    local rc=0
    "$@" || rc=$?
    # shellcheck disable=SC2016
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --mpi=none --cpu-bind=none bash -c '
        export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-/tmp/enroot/data/user-$EUID}
        enroot list | grep aim344_ | xargs -r -n 1 enroot remove -f' < /dev/null >> "$lab_log" 2>&1 || true
    return "$rc"
}

# The healthy run: the training stand-in, the epoch job and the all-reduce sweep,
# so that each incident's output differs from a healthy one only in its fault.
healthy_job() {
    torch_job job job "${STAND_IN[@]}" --writer-rank=0 --duration-seconds="$RUN_SECONDS" \
        || return
    local AIM344_JOB_LIMIT_SECONDS=$DATALOADER_LIMIT_SECONDS
    torch_job job-epochs job "${EPOCH_JOB[@]}" || return
    sweep_job job
}

# Removes the filler and the partial write that Incident 2 leaves in the checkpoint
# directories, which are on the nodes and outlive the allocation. Without it a job
# that starts at step 0 fails at its first save.
clear_checkpoint_leftovers() {
    printf 'Clearing the checkpoint directories on both nodes.\n'
    prepare_torch
    lab_step node_command python3 /opt/aim344/recover-checkpoint.py
}

# Incident 6: once the job has printed two progress records, reset one EFA device
# of the target node (lab.sh reset-efa). Runs beside the job and ends with it.
reset_efa_after_progress() {
    local log=$1 count
    while :; do
        count=$(grep -c '^AIM344 collective_progress=' "$log" 2>/dev/null || true)
        (( ${count:-0} < 2 )) || break
        sleep 1
    done
    lab_step bash "$LAB_DIR/lab.sh" reset-efa
}

# The current incident's job: $1 is incident, rerun or resume.
incident_job() {
    local what=$1 name=incident-$incident
    [[ $what == incident ]] || name=$name-$what
    local -a steps=("${STAND_IN[@]}")
    if ((${#rerun_options[@]})) && [[ $incident != 3 ]]; then
        echo "Incident $incident takes no options. Run it again with: bash job.sh rerun" >&2
        exit 2
    fi
    prepare_allocation
    # Every job with a checkpoint writer starts its incident on clean directories.
    [[ $what != incident || $incident =~ ^[13]$ ]] || clear_checkpoint_leftovers
    case $incident in
        1)  # A slow job: the aws-ofi-nccl plugin is moved out of NCCL's search path
            # inside the allocation's container, so NCCL falls back to sockets.
            case $what in
                incident)
                    lab_step node_command bash /opt/aim344/plugin-state.sh restore
                    lab_step node_command bash /opt/aim344/plugin-state.sh hide
                    sweep_job "$name"
                    ;;
                rerun) sweep_job "$name" ;;
                resume)
                    lab_step node_command bash /opt/aim344/plugin-state.sh restore
                    ending_job sweep_job "$name"
                    ;;
            esac
            ;;
        2)  # A checkpoint write fails: rank 9's node writes the checkpoint, and
            # workload.py --inject fills its 32 MiB checkpoint directory first.
            steps+=(--writer-rank=9 --duration-seconds="$RUN_SECONDS")
            case $what in
                incident) torch_job "$name" job "${steps[@]}" --inject ;;
                rerun) torch_job "$name" job "${steps[@]}" --resume ;;
                resume)
                    clear_checkpoint_leftovers
                    ending_job torch_job "$name" job "${steps[@]}" --resume --end-after-save
                    ;;
            esac
            ;;
        3)  # A DataLoader that stops: the rank process holds the dataset's lock in a
            # thread when the second epoch forks its workers.
            local -a loader=("${EPOCH_JOB[@]}")
            AIM344_JOB_LIMIT_SECONDS=$DATALOADER_LIMIT_SECONDS
            case $what in
                incident) torch_job "$name" job "${loader[@]}" --dataset-cache=rank-thread ;;
                # Options from the command line come last, so they win.
                rerun) torch_job "$name" job "${loader[@]}" --dataset-cache=rank-thread "${rerun_options[@]}" ;;
                resume) ending_job torch_job "$name" job "${loader[@]}" ;;
            esac
            ;;
        4)  # A wrong result on one GPU: rank 13 flips bit 22 of one element at step 37.
            steps+=(--writer-rank=0)
            case $what in
                incident)
                    torch_job "$name" job "${steps[@]}" --duration-seconds="$CORRUPTION_SECONDS" \
                        --corrupt-rank=13 --corrupt-step=37 --corrupt-bit=22
                    ;;
                # The flip happens once, so a second run is clean.
                rerun) torch_job "$name" job "${steps[@]}" --duration-seconds="$CORRUPTION_SECONDS" ;;
                # Roll back to the last checkpoint saved before the changed step.
                resume)
                    ending_job torch_job "$name" job "${steps[@]}" --duration-seconds="$RUN_SECONDS" \
                        --resume --resume-before=37 --end-after-save
                    ;;
            esac
            ;;
        5)  # A device-side assertion (Xid 43 class): rank 0 indexes out of range on its GPU at step 201.
            steps+=(--writer-rank=0 --duration-seconds="$RUN_SECONDS")
            case $what in
                incident|rerun) torch_job "$name" job "${steps[@]}" --assert-rank=0 --assert-step=201 ;;
                resume) ending_job torch_job "$name" job "${steps[@]}" --resume --end-after-save ;;
            esac
            ;;
        6)  # An EFA device reset under a running job.
            steps+=(--writer-rank=0 --duration-seconds="$EFA_RESET_RUN_SECONDS")
            case $what in
                incident)
                    local progress=$RESULTS_HOME/device-$SLURM_JOB_ID.log rc=0
                    # An earlier run in this allocation must not count as this run's progress.
                    : > "$progress"
                    reset_efa_after_progress "$progress" &
                    local helper=$!
                    torch_job "$name" device "${steps[@]}" || rc=$?
                    kill "$helper" 2>/dev/null || true
                    wait "$helper" 2>/dev/null || true
                    return "$rc"
                    ;;
                rerun) torch_job "$name" device "${steps[@]}" ;;
                resume)
                    echo 'Incident 6 is recovered from terminal B, outside the allocation: bash lab.sh recover' >&2
                    exit 2
                    ;;
            esac
            ;;
    esac
}

# The options rerun passes on to the DataLoader job, in the forms its page gives.
rerun_options=()
parse_rerun_options() {
    while (($#)); do
        case $1 in
            --dataset-cache=*|--dataloader-timeout=*|--reinit-in-worker) rerun_options+=("$1") ;;
            --dataset-cache|--dataloader-timeout)
                (($# >= 2)) || usage
                rerun_options+=("$1=$2")
                shift ;;
            *) usage ;;
        esac
        shift
    done
}

# ---------------------------------------------------------------- commands
(($# >= 1)) || usage
verb=$1
shift
case $verb in
    notes) (($# == 0)) || usage; make_notes; exit 0 ;;
    run|resume) (($# == 0)) || usage ;;
    incident) (($# <= 1)) || usage ;;
    rerun) parse_rerun_options "$@" ;;
    *) usage ;;
esac

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run this inside your allocation: take it in terminal A with the salloc line on your page.' >&2
    exit 2
fi
: "${AIM344_EFA_DEVICE:?Open a new shell, whose login profile sets AIM344_EFA_DEVICE}"
# The training stand-in uses this one EFA device on each node.
export AIM344_EFA_IFACE=$AIM344_EFA_DEVICE
# Node-local results, as the login profile sets them in an allocation shell.
export RESULTS_DIR=${RESULTS_DIR:-/var/tmp/aim344-$(id -un)-$SLURM_JOB_ID}
mkdir -p -- "$RESULTS_HOME"
# shellcheck source=common.sh
source "$LAB_DIR/common.sh"
# Every torch job copies its output to ~/aim344-results/device-<job>.log, and its
# processes write the fault injector's own lines to files on the nodes
# (workload.py --lab-dir), not to the job's output.
AIM344_PROGRESS_COPY=1
AIM344_LAB_LOG=$lab_log

case $verb in
    run)
        prepare_allocation
        ending_job healthy_job
        ;;
    incident)
        incident=${1:-$(( $(cat "$CURRENT" 2>/dev/null || echo 0) + 1 ))}
        [[ $incident =~ ^[1-6]$ ]] || usage
        printf '%s\n' "$incident" > "$CURRENT"
        printf 'Starting incident %s.\n' "$incident"
        incident_job incident
        ;;
    rerun|resume)
        incident=$(cat "$CURRENT" 2>/dev/null) || {
            echo 'No incident has started yet. Start the first with: bash job.sh incident 1' >&2; exit 2;
        }
        incident_job "$verb"
        ;;
esac
