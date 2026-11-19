#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Install the lab controls for the participant of one cluster. The operator runs
# this once per account before the session, on the login node, as root.
#
#   sudo AIM344_SCHEDULE_FILE=/root/seat-1.json AIM344_PARTITION=<queue> \
#        AIM344_SUITE_ENTRY=<suite entry on the target> bash install-participant-control.sh \
#        <assignment-id> ubuntu <target-node> <target-instance-id> <target-host>
#
# The participant is the account that owns the cluster, ubuntu on the event's
# image, and keeps its own sudo. What this creates:
#   the controller at /usr/local/sbin/aim344-device-session, which
#     12.device-exercise.sh runs with sudo -n
#   /etc/aim344-device-session.json describing this one assignment
#   the participant's ~/aim344-results
#   a root key pair for the controller's maintenance route to the target
# It prints the prepare-target.py command to run on the target, with this key,
# and the login profile line for the participant's ~/.bashrc.
#
# AIM344_SCHEDULE_FILE (optional) is this seat's incident schedule: a JSON list
# in the order the participant's `next` starts them, each entry an exercise and
# the settings the launcher passes to the job, for example
#
#   [{"exercise": "slow-job"},
#    {"exercise": "storage", "settings": {"writer_rank": 9}},
#    {"exercise": "dataloader"},
#    {"exercise": "silent-corruption",
#     "settings": {"corrupt_rank": 13, "corrupt_step": 37, "corrupt_bit": 22}},
#    {"exercise": "cuda-assert", "settings": {"assert_rank": 0, "assert_step": 201}},
#    {"exercise": "efa-reset"}]
#
# The participant pages number the incidents in this order. Each exercise takes
# only the settings its launcher applies, and a rank must be a rank of the job:
# below AIM344_WORLD_SIZE, which defaults to 16 (2 nodes of 8 GPUs). The
# installer refuses a corrupt_step outside 1 to 4095 but cannot know the step
# rate: choose a corrupt_step that the 20-second run reaches (about 480 steps on
# g7.48xlarge), or the incident shows no differing line.
#
# Exercises: slow-job, storage, dataloader, silent-corruption, cuda-assert (applied by the
# launcher in the participant's job) and efa-reset (applied by the controller
# once that job reports progress). The schedule is copied into the session
# configuration, and this seat's record of issued incidents is reset. The file is
# checked before anything is changed, and a seat whose round still holds its
# exercise node is refused. A finished round of the earlier pass is kept as
# <assignment>.round-<n>.before-reset.json in the state directory, and the seat
# continues from a fresh record. Without the variable a rerun keeps the schedule
# already configured, and its record.
set -euo pipefail

[[ $(id -u) == 0 ]] || { echo 'Run this with sudo on the login node.' >&2; exit 1; }

assignment=${1:?assignment id, for example seat-1}
participant=${2:?participant account, ubuntu on the event image}
target_node=${3:?Slurm node name of the exercise target}
target_instance=${4:?EC2 instance id of the exercise target}
target_host=${5:?address of the exercise target}

[[ $assignment =~ ^[a-z0-9-]+$ ]] || { echo 'Bad assignment id.' >&2; exit 2; }
[[ $participant =~ ^[a-z][a-z0-9-]*$ ]] || { echo 'Bad participant user name.' >&2; exit 2; }
[[ $target_node =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Bad Slurm node name.' >&2; exit 2; }
[[ $target_instance =~ ^i-[0-9a-f]+$ ]] || { echo 'Bad instance id.' >&2; exit 2; }
id "$participant"

lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# The session helper may already be installed by the staging step. Prefer an
# explicit source, then the companion layout, then the installed copy.
if [[ -n ${AIM344_SESSION_SOURCE:-} ]]; then
    session_source=$AIM344_SESSION_SOURCE
elif [[ -f $lab/facilitator/device-session.py ]]; then
    session_source=$lab/facilitator/device-session.py
elif [[ -f /usr/local/sbin/aim344-device-session ]]; then
    session_source=/usr/local/sbin/aim344-device-session
else
    echo 'Cannot find device-session.py; set AIM344_SESSION_SOURCE.' >&2
    exit 2
fi
[[ -f $session_source ]] || { echo "Not a file: $session_source" >&2; exit 2; }
slurm_bin=${AIM344_SLURM_BIN:-/opt/aws/pcs/scheduler/slurm-25.05/bin}
state_dir=/var/lib/aim344-device-session
session_config=/etc/aim344-device-session.json
suite_entry=${AIM344_SUITE_ENTRY:?Set AIM344_SUITE_ENTRY to the pinned suite entry point on the target}
export AIM344_PARTITION=${AIM344_PARTITION:?Set AIM344_PARTITION to the assigned queue}
participant_home=$(getent passwd "$participant" | cut -d: -f6)

# The schedule file is checked before this installer changes anything, so a refused
# file leaves the seat as it was. The helper that will run the schedule checks it,
# and a seat whose round still holds its exercise node keeps its schedule.
AIM344_SCHEDULE_FILE=${AIM344_SCHEDULE_FILE:-}
checked_schedule=$(python3 - "$AIM344_SCHEDULE_FILE" "$session_source" \
        "$session_config" "$state_dir" "$assignment" <<'PY'
import importlib.machinery, importlib.util, json, os, sys
schedule_file, helper_path, config_path, state_dir, assignment = sys.argv[1:6]
loader = importlib.machinery.SourceFileLoader('aim344_device_session', helper_path)
helper = importlib.util.module_from_spec(
    importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(helper)
previous = {}
if os.path.exists(config_path):
    with open(config_path) as handle:
        config = json.load(handle)
    state_dir = config.get('state_dir', state_dir)
    previous = config.get('assignments', {}).get(assignment, {})
world_size = os.environ.get('AIM344_WORLD_SIZE',
                            str(previous.get('world_size', helper.DEFAULT_WORLD_SIZE)))
if not world_size.isdigit() or int(world_size) < 2 or int(world_size) % 2:
    raise SystemExit('AIM344_WORLD_SIZE must divide over the two assigned nodes')
entries = previous.get('exercise_schedule')
if schedule_file:
    try:
        with open(schedule_file) as handle:
            entries = json.load(handle)
    except OSError as error:
        raise SystemExit(f'refusing {schedule_file}: {error.strerror}')
    except ValueError as error:
        raise SystemExit(f'refusing {schedule_file}: it is not JSON ({error})')
try:
    schedule = (helper.validate_schedule(entries, world_size=int(world_size))
                if schedule_file or entries is not None else None)
    if schedule_file:
        state = helper.StateStore(state_dir, assignment).read()
        if state.phase in helper.HOLDING_PHASES:
            raise SystemExit(f'refusing to reset the schedule of {assignment}: its round '
                             f'is at {state.phase} and still holds the exercise node. '
                             f'Finish it with recover before resetting the schedule.')
except helper.Refusal as refusal:
    raise SystemExit(f'refusing the schedule of {assignment}: {refusal}')
if schedule_file:
    print(json.dumps(schedule))
PY
)

echo "== installing the session helper"
installed=/usr/local/sbin/aim344-device-session
if [[ $(readlink -f "$session_source") == "$(readlink -f "$installed")" ]]; then
    echo "The session helper is already staged at $installed; keeping it."
    chown root:root "$installed"
    chmod 0755 "$installed"
else
    install -o root -g root -m 0755 "$session_source" "$installed"
fi
sha256sum "$installed"

echo "== maintenance key for the controller's route to the target"
install -d -o root -g root -m 0700 /root/.ssh
maintenance_key=/root/.ssh/aim344-maintenance
if [[ ! -s $maintenance_key ]]; then
    ssh-keygen -t ed25519 -N '' -q -f "$maintenance_key" -C 'aim344-maintenance'
fi
echo 'On the TARGET, as root, run prepare-target.py with this key:'
printf "  python3 %s %s %s '%s'\n" "$lab/facilitator/prepare-target.py" \
    "$target_node" "$participant" "$(cat "$maintenance_key.pub")"

echo "== session configuration"
install -d -o root -g root -m 0700 "$state_dir"
AIM344_CHECKED_SCHEDULE=$checked_schedule \
python3 - "$session_config" "$assignment" "$participant" "$target_node" \
        "$target_instance" "$target_host" "$state_dir" "$slurm_bin" "$suite_entry" \
        "$installed" <<'PY'
import importlib.machinery, importlib.util, json, os, pwd, sys
(path, assignment, participant, node, instance, host, state_dir, slurm_bin,
 suite_entry, helper_path) = sys.argv[1:11]
previous = {}
if os.path.exists(path):
    with open(path) as handle:
        config = json.load(handle)
    state_dir = config.get('state_dir', state_dir)
    previous = config.get('assignments', {}).get(assignment, {})

# The seat's incident schedule, checked before any change by the helper that will
# run it, so the installer and the controller cannot disagree about which
# exercises exist. The installed helper checks it once more here.
loader = importlib.machinery.SourceFileLoader('aim344_device_session', helper_path)
helper = importlib.util.module_from_spec(
    importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(helper)
world_size = os.environ.get('AIM344_WORLD_SIZE',
                            str(previous.get('world_size', helper.DEFAULT_WORLD_SIZE)))
if not world_size.isdigit() or int(world_size) < 2 or int(world_size) % 2:
    raise SystemExit('AIM344_WORLD_SIZE must divide over the two assigned nodes')
account = pwd.getpwnam(participant)
checked = os.environ.get('AIM344_CHECKED_SCHEDULE')
schedule = json.loads(checked) if checked else previous.get('exercise_schedule')
try:
    if checked or schedule is not None:
        schedule = helper.validate_schedule(schedule, world_size=int(world_size))
except helper.Refusal as refusal:
    raise SystemExit(f'refusing the schedule of {assignment}: {refusal}')
if checked:
    try:
        # A new schedule starts this seat from its first incident. Under the seat's
        # own lock, so no request of the controller runs while its records change.
        store = helper.StateStore(state_dir, assignment)
        with store.exclusive():
            state = store.read()
            if state.phase in helper.HOLDING_PHASES:
                raise SystemExit(f'refusing to reset the schedule of {assignment}: its '
                                 f'round is at {state.phase} and still holds the exercise '
                                 f'node. Finish it with recover, or remove {store.path} '
                                 f'once the node is replaced, then rerun.')
            progress = os.path.join(state_dir, f'schedule-{assignment}.json')
            if os.path.lexists(progress):
                os.unlink(progress)
                print(f'Reset the issued-incident record {progress}')
            if state.phase != 'ready':
                # A finished round of the earlier pass would otherwise stay in
                # status, with its check results and its incident's answer. Keep it
                # for the operator, then continue from a fresh record and the next
                # round number, as the controller does when a new round starts. The
                # dots keep the name from ever being another assignment's record.
                kept = os.path.join(state_dir,
                                    f'{assignment}.round-{state.round}.before-reset.json')
                helper._write_private_json(kept, state.to_dict())
                store.write(helper.State(round=state.round + 1))
                print(f'Kept round {state.round} of the earlier pass in {kept}; the '
                      f'seat continues from round {state.round + 1}')
    except helper.Refusal as refusal:
        raise SystemExit(f'refusing to reset the schedule of {assignment}: {refusal}')

seat = {
    'world_size': int(world_size),
    'participant_user': participant,
    'participant_uid': account.pw_uid,
    # The participant's own primary group. Results are written by a process that
    # has dropped to this uid and gid, so both are needed; without the gid the
    # helper would have to guess one.
    'participant_gid': account.pw_gid,
    'answers_dir': f'{account.pw_dir}/aim344-results',
    'target_node': node,
    'target_instance_id': instance,
    'target_host': host,
    'coordinator_node': os.uname().nodename,
    'partition': os.environ['AIM344_PARTITION'],
    'drain_reason': 'aim344-device-recovery',
    'job_name_prefix': 'aim344-',
    'slurm_bin': slurm_bin,
    'suite_entry': suite_entry,
    # What a participant may run with collect after any exercise: the suite's
    # read-only checks the target's maintenance allowlist holds (prepare-target.py
    # writes "checks": [0, 2, 3, 6] there). The times below are the
    # intervals between completion times in one sequential run as root on a G7e
    # node (two EFA) on 2026-11-01; that run did not record when check 0 started.
    # The suite headers estimate 5 s, 3 s, 10 s and 5-15 min.
    #   0  nvidia-smi query, GPU count, Xid/SXid scan of the kernel log, ECC and
    #      retired pages; not recorded; reads only.
    #   2  lspci, ibv_devices, fi_info -p efa, /dev/infiniband, lsmod, ulimit -l and
    #      the gdrcopy sanity test (30 s limit), which pins and copies GPU memory;
    #      about 8 s.
    #   3  nvidia-smi topo -m and topo -p2p, and NVLink status and error queries
    #      where NVLink is expected; about 1 s; reads only.
    #   6  fi_pingpong self-loopback per EFA domain (180 s limit each) and the EFA
    #      statistics; about 19 s; real EFA traffic and counter growth.
    # Checks 1 and 4 (DCGM levels 2 and 4: minutes to hours, they start or stop
    # GPU telemetry) and 5 (a two-node MPI test) stay outside this list.
    'collect_checks': [0, 2, 3, 6],
    # What recovery must see PASS, per kind, before the node returns to service.
    # Widening collect does not change it; any WARN still keeps the node out.
    'qualify_checks': {'gpu': [0, 3], 'efa': [2, 6]},
    'reboot_wait_seconds': 900,
    'reboot_poll_seconds': 10,
    'maintenance_identity_file': '/root/.ssh/aim344-maintenance',
}
if schedule is not None:
    seat['exercise_schedule'] = schedule
# One cluster, one participant: the configuration holds this assignment only.
config = {'state_dir': state_dir, 'assignments': {assignment: seat}}
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(descriptor, 'w') as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write('\n')
os.chmod(path, 0o600)
print(f'Wrote {assignment} into {path}'
      + (f' with a {len(schedule)}-incident schedule' if schedule else ' with no incident schedule'))
PY

install -d -o "$participant" -g "$(id -g "$participant")" -m 0700 "$participant_home/aim344-results"

# The login profile (facilitator/participant-profile.sh) lets a new shell run the
# lab commands with no setup. It reads lab.env, which carries the seat's target,
# peer and EFA device. The installer prints its ~/.bashrc line rather than writing
# it, so a host whose account other work also uses can leave the line out.
profile=$lab/facilitator/participant-profile.sh
profile_line=$(printf '[ -r %q ] && . %q  # AIM344 login profile' "$profile" "$profile")

echo "== participant-facing summary"
cat <<SUMMARY
Assignment:        $assignment
Participant:       $participant
Exercise node:     $target_node ($target_instance)
Results directory: $participant_home/aim344-results

As $participant, add the login profile to ~/.bashrc on this login node:
  $profile_line
Then, in a new shell as $participant:
  ./12.device-exercise.sh status

Operator view of this seat (exercise kinds and schedule included):
  sudo /usr/local/sbin/aim344-device-session --operator status
A node recovery cannot qualify stays out of service until an operator replaces
it after the session.
SUMMARY
