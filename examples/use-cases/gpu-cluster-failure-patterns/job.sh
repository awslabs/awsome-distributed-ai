#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# The participant's job launcher for the hour's incidents.
#
#   bash job.sh notes      create your notes file
#   bash job.sh run        run the training stand-in on your pair
#   bash job.sh incident   start your next incident and run its job
#   bash job.sh rerun      run the current incident's job again, as the pair is now
#   bash job.sh resume     undo what the lab changed and resume the job
#
# Every command except notes runs inside your two-node allocation (terminal A).
# incident asks the lab controls for the incident's profile
# (12.device-exercise.sh next) and runs its job from that profile; rerun and
# resume reuse the same profile and never ask for a new incident. The launcher
# prints none of the incident's settings. The lab's own lines go to a lab log
# beside the profile, which the facilitator reads.
set -euo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

RESULTS_HOME=$HOME/aim344-results
INCIDENTS=$RESULTS_HOME/incidents
# The training stand-in runs 60 seconds; the incident in which the lab controls
# act on the target runs 120, which leaves them a window after two progress
# records. A resumed job runs long enough to pass one more checkpoint.
RUN_SECONDS=60
CONTROLLER_RUN_SECONDS=120
RESUME_SECONDS=10
# Checkpoint interval in steps; measurement 1 on g7.48xlarge may change it.
CHECKPOINT_EVERY=200
# The controller's own checks decide when it acts. These only bound how long the
# launcher keeps asking during one run.
NEXT_ATTEMPTS=10
NEXT_INTERVAL=3

usage() {
    cat >&2 <<'USAGE'
Usage: bash job.sh <command>

  notes      Create your notes file, ~/aim344-results/notes.md
  run        Run the training stand-in on your pair
  incident   Start your next incident and run its job
  rerun      Run the current incident's job again, as your pair is now
  resume     Undo what the lab changed in the current incident and resume the job

Run every command except notes inside your allocation, in terminal A.
USAGE
    exit 2
}

refuse() {
    printf '%s\n' "$*" >&2
    exit 3
}

lab_note() {
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$lab_log"
}

# ---------------------------------------------------------------- notes
make_notes() {
    local notes=$RESULTS_HOME/notes.md
    mkdir -p -- "$RESULTS_HOME"
    if [[ -e $notes || -L $notes ]]; then
        printf 'Your notes file already exists and was left as it is: %s\n' "$notes"
        return 0
    fi
    local incident_fields
    incident_fields=$'Row of the triage table:\n\nPredictions:\n\nEarliest error, its line number and its rank:\n\nLater symptoms, and the ranks that printed them:\n\nHypothesis:\n\nAlternative, the records that make it less likely, and the records that leave it open:\n\nRecovery and why:\n\nTransfer question:\n'
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

# Healthy values

Transport:

Ranks that finished NCCL setup:

Checksums:

Checkpoint line:

Rate:

Longest gap between progress lines:

End of epoch 1:

Bus bandwidth at 128 MiB:

Bus bandwidth at the smaller size:

Wrong elements:

EFA devices that moved bytes:

Storage:

Warnings:

Pre-session Check 6 record:

# Incident 1

$incident_fields
# Incident 2

$incident_fields
# Incident 3

$incident_fields
# Incident 4

Row of the triage table:

Predictions:

| Record | What it shows | Which part it names |
|---|---|---|
| Last progress line | | |
| Watchdog lines | | |
| Correctness error after the watchdog lines | | |
| Kernel log | | |
| Check 0 | | |
| Check 2 | | |
| Check 3 | | |
| Check 6 | | |

Hypothesis:

Alternative that remains:

Recovery and why:

Will the lab's recovery accept this node?

What the verification job shows, and what it does not:

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
| Incident 2 | | | |
| Incident 3 | | | |

# Card answers

## Card A

1.

2.

3.

## Card B

1.

2.

3.

## Card C

1.

2.

3.
NOTES
    )
    printf 'Created your notes file: %s\n' "$notes"
}

# ---------------------------------------------------------------- checks
require_allocation() {
    [[ -n ${SLURM_JOB_ID:-} ]] || refuse 'Run this inside your allocation: take it in terminal A with the salloc line on your page, then run job.sh there.'
    local job
    job=$(scontrol show job "$SLURM_JOB_ID" -o) || refuse "Slurm did not describe your allocation $SLURM_JOB_ID; tell your table expert."
    # The lab controls act only on a job of this account whose name starts with
    # aim344- and whose walltime is at most 10 minutes. Every run is held to the
    # same rule, so that no incident is singled out by a refusal.
    [[ " $job " == *" JobName=aim344-"* ]] || refuse 'Take your allocation with the salloc line on your page: its job name starts with aim344-.'
    [[ $job =~ (^|[[:space:]])TimeLimit=(00:0[1-9]:[0-5][0-9]|00:10:00)([[:space:]]|$) ]] \
        || refuse 'Take your allocation with the salloc line on your page: job.sh runs in an allocation of 1 to 10 minutes.'
    [[ ${AIM344_EFA_DEVICE:-} =~ ^[A-Za-z0-9_]+$ ]] \
        || refuse 'AIM344_EFA_DEVICE is not set. Open a new terminal, whose login profile sets it, or tell your table expert.'
    # The training stand-in uses this one EFA device on each node.
    export AIM344_EFA_IFACE=$AIM344_EFA_DEVICE
    # Node-local results, as the login profile sets them in an allocation shell.
    export RESULTS_DIR=${RESULTS_DIR:-/var/tmp/aim344-$(id -un)-$SLURM_JOB_ID}
}

# ---------------------------------------------------------------- profiles
# Sets incident, exercise and writer_rank from one profile, after checking that it
# is this account's own regular file and that this launcher applies every setting.
read_profile() {
    local fields
    fields=$(python3 - "$1" 2>/dev/null <<'PY'
import json, os, stat, sys
path = sys.argv[1]
handle = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
info = os.fstat(handle)
if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
    raise SystemExit(1)
with os.fdopen(handle) as stream:
    profile = json.load(stream)
number, exercise = profile['incident'], profile['exercise']
settings = profile.get('settings') or {}
# The settings this launcher applies, for each exercise it runs.
applied = {'storage': {'writer_rank'}, 'slow-job': set(), 'efa-reset': set()}
if (not isinstance(number, int) or isinstance(number, bool) or number < 1
        or not isinstance(exercise, str) or not exercise.replace('-', '').isalpha()
        or not isinstance(settings, dict)
        or set(settings) - applied.get(exercise, set(settings))):
    raise SystemExit(1)
writer = settings.get('writer_rank', 0)
if not isinstance(writer, int) or isinstance(writer, bool) or not 0 <= writer < 4096:
    raise SystemExit(1)
print(number, exercise, writer)
PY
) || refuse 'The incident profile cannot be used by this launcher; tell your table expert.'
    read -r incident exercise writer_rank <<< "$fields"
    profile=$1
    lab_log=${profile%.json}-lab.log
}

# The profile of the incident issued last.
read_current_profile() {
    local path
    path=$(python3 - "$INCIDENTS" <<'PY'
import glob, json, os, sys
newest = None
for path in glob.glob(os.path.join(sys.argv[1], 'incident-*.json')):
    try:
        with open(path) as stream:
            issued = json.load(stream)['issued_at']
    except (OSError, ValueError, KeyError, TypeError):
        continue
    if isinstance(issued, (int, float)) and (newest is None or issued > newest[0]):
        newest = (issued, path)
if newest:
    print(newest[1])
PY
)
    [[ -n $path ]] || refuse 'No incident has started on your seat yet. Start one with: bash job.sh incident'
    read_profile "$path"
}

# ---------------------------------------------------------------- jobs
# One torch run with a neutral run id. Prints the job log's path before and
# after the run, and the time the job ended in UTC.
torch_job() {
    local name=$1 phase=$2 rc=0
    shift 2
    AIM344_RUN_ID=$name-$(date -u +%Y%m%dT%H%M%SZ)
    AIM344_PROGRESS_COPY=1
    printf 'Job log: %s\n' "$RESULTS_DIR/$AIM344_RUN_ID.log"
    run_torch "$phase" "$@" || rc=$?
    printf 'job_end_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'Job log: %s\n' "$RESULTS_DIR/$AIM344_RUN_ID.log"
    return "$rc"
}

sweep_job() {
    local name=$1 rc=0
    AIM344_RUN_ID=$name-$(date -u +%Y%m%dT%H%M%SZ)
    printf 'Job log: %s\n' "$RESULTS_DIR/$AIM344_RUN_ID.nccl.log"
    # The sweep uses every EFA device of each node, as 0.baseline.sh does.
    (unset AIM344_EFA_IFACE; run_sweep job) || rc=$?
    printf 'job_end_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'Job log: %s\n' "$RESULTS_DIR/$AIM344_RUN_ID.nccl.log"
    return "$rc"
}

# Prepare the allocation once, just before the first step that uses it, so that a
# refusal costs no container start.
prepared=0
ensure_prepared() {
    if ((prepared == 0)); then
        prepare_slurm
        prepared=1
    fi
}

# A lab step whose output belongs in the lab log, not on the participant's terminal.
lab_step() {
    lab_note "lab step: $*"
    "$@" >> "$lab_log" 2>&1 || refuse 'The lab could not prepare this step; tell your table expert.'
}

progress_records() {
    local count
    count=$(grep -cE "^AIM344 collective_progress=[0-9]+ collectives; .*; job_id=$SLURM_JOB_ID; " "$1" 2>/dev/null) || true
    printf '%s\n' "${count:-0}"
}

# Runs beside the job: once the job has printed two progress records, ask the lab
# controls to act, until they reply that the incident is under way. A refusal is
# final, even one that comes after they acted, so it is never asked again.
controller_step() {
    local copy=$1 stop=$2 attempt status reply
    until (( $(progress_records "$copy") >= 2 )); do
        if [[ -e $stop ]]; then
            lab_note 'controller step: the job ended before two progress records'
            printf 'The job ended before the lab controls could act. Tell your table expert.\n' >&2
            return 0
        fi
        sleep 1
    done
    for ((attempt = 1; attempt <= NEXT_ATTEMPTS; attempt++)); do
        [[ -e $stop ]] && break
        status=0
        reply=$(bash "$LAB_DIR/12.device-exercise.sh" next 2>&1) || status=$?
        lab_note "next, attempt $attempt, exit $status: $reply"
        if ((status != 0)); then
            printf '%s\n' "$reply" >&2
            return 0
        fi
        [[ $reply == *' is under way.'* ]] && return 0
        sleep "$NEXT_INTERVAL"
    done
    lab_note 'controller step: no reply said the incident is under way'
    printf 'The lab controls did not act during this run. Tell your table expert.\n' >&2
}

controller_job() {
    local name=$1 copy=$RESULTS_HOME/device-$SLURM_JOB_ID.log stop rc=0 helper
    shift
    stop=$INCIDENTS/.job-ended-$SLURM_JOB_ID-$$
    rm -f -- "$stop"
    mkdir -p -- "$RESULTS_HOME"
    # An earlier run in this allocation must not count as this run's progress.
    : > "$copy"
    controller_step "$copy" "$stop" &
    helper=$!
    torch_job "$name" device "$@" || rc=$?
    # Let a request that is under way finish; never cut it off.
    : > "$stop"
    wait "$helper" || true
    rm -f -- "$stop"
    return "$rc"
}

# The current incident's job: $1 is incident, rerun or resume.
incident_job() {
    local what=$1 name=incident-$incident
    [[ $what == incident ]] || name=$name-$what
    local -a steps=(device --checkpoint-every="$CHECKPOINT_EVERY" --mark-lab-lines)
    case $exercise in
        storage)
            steps+=(--writer-rank="$writer_rank")
            ensure_prepared
            case $what in
                incident) torch_job "$name" job "${steps[@]}" --duration-seconds="$RUN_SECONDS" --inject ;;
                rerun) torch_job "$name" job "${steps[@]}" --duration-seconds="$RUN_SECONDS" --resume ;;
                resume)
                    prepare_torch
                    lab_step node_command python3 /opt/aim344/recover-checkpoint.py
                    torch_job "$name" job "${steps[@]}" --duration-seconds="$RESUME_SECONDS" --resume
                    ;;
            esac
            ;;
        slow-job)
            ensure_prepared
            case $what in
                incident) lab_step node_command bash /opt/aim344/plugin-state.sh hide ;;
                resume) lab_step node_command bash /opt/aim344/plugin-state.sh restore ;;
            esac
            sweep_job "$name"
            ;;
        efa-reset)
            steps+=(--writer-rank=0 --duration-seconds="$CONTROLLER_RUN_SECONDS")
            case $what in
                incident) ensure_prepared; controller_job "$name" "${steps[@]}" ;;
                rerun) ensure_prepared; torch_job "$name" device "${steps[@]}" ;;
                resume) refuse "Incident $incident is recovered with the command on your page, not with job.sh resume." ;;
            esac
            ;;
        *)
            refuse "Incident $incident cannot run live with this version of the lab. Answer its questions from the expected output on your page."
            ;;
    esac
}

# ---------------------------------------------------------------- commands
[[ $# -eq 1 ]] || usage
verb=$1
case $verb in
    notes) make_notes; exit 0 ;;
    run|incident|rerun|resume) ;;
    *) usage ;;
esac

require_allocation
# shellcheck source=common.sh
source "$LAB_DIR/common.sh"
AIM344_RUN_ID=
AIM344_LAB_LOG=
AIM344_PROGRESS_COPY=0

case $verb in
    run)
        ensure_prepared
        torch_job job job device --checkpoint-every="$CHECKPOINT_EVERY" --writer-rank=0 \
            --duration-seconds="$RUN_SECONDS"
        ;;
    incident)
        # Prepared before the request, so that a failure here leaves the incident
        # unissued. Exactly one request for the incident itself; the incident in
        # which the lab controls act asks again from controller_step, after progress.
        ensure_prepared
        status=0
        reply=$(bash "$LAB_DIR/12.device-exercise.sh" next 2>&1) || status=$?
        ((status == 0)) || { printf '%s\n' "$reply" >&2; exit "$status"; }
        handed=$(sed -n 's/^Profile: //p' <<< "$reply" | tail -n 1)
        [[ -n $handed ]] || { printf '%s\n' "$reply" >&2; refuse 'The lab controls handed over no incident profile; tell your table expert.'; }
        read_profile "$handed"
        AIM344_LAB_LOG=$lab_log
        lab_note "next: $reply"
        printf 'Starting incident %s.\n' "$incident"
        incident_job incident
        ;;
    rerun|resume)
        read_current_profile
        AIM344_LAB_LOG=$lab_log
        incident_job "$verb"
        ;;
esac
