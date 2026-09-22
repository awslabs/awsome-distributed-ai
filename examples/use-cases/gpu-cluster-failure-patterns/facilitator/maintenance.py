#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""AIM344 maintenance forced command, installed on the fault target.

Install a root-owned copy at /usr/local/sbin/aim344-maintenance and reach it
only through the target's root authorized_keys forced command:

    command="/usr/local/sbin/aim344-maintenance",no-port-forwarding,
    no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 ...

This is the narrow maintenance route the coordinator's session helper calls. It
accepts a fixed set of actions and no free-form command, so possession of the
coordinator's maintenance key does not grant a root shell on the target.

Actions:
  inspect                    existing aim344-device-fault inspect (device inventory)
  boot-id                    read /proc/sys/kernel/random/boot_id
  gpu-remove --confirm T     existing helper, unchanged conditions
  efa-unbind [--job N] --confirm T
  efa-rebind --confirm T
  collect <check>            read-only health check on the drained node
  gpu-prepare --operation T [--fresh-capture]
                             record/pause on a new start, or reuse intact originals
  gpu-restore --operation T  put that recorded state back, or fail
  efa-activity               two timed samples of the SELECTED EFA's byte
                             counters and this node's queue, for the active-EFA
                             pre-injection condition; observation only
  restore-runtime [--participant U]   existing restore-runtime.sh

`--operation T` is the coordinator's per-round operation identifier. Both the
recording and the restoration require the same one, so original-state records left
behind by another round -- or by another table sharing this exercise node -- cannot
satisfy this round's restore.

The participant that owns the restored checkpoint fixture is named per request,
because the fixture belongs to whichever table is authenticated on the coordinator
rather than to a fixed account. The name is accepted only if it appears in this
node's root-owned `participants` allowlist, so the coordinator cannot ask for an
arbitrary user; omitting it keeps the configured default.

Device mutations are delegated to /usr/local/sbin/aim344-device-fault, which
keeps its own identity, drain, job-ownership and idle-GPU conditions. This
wrapper does not relax them and cannot reach any other device.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

DEVICE_FAULT = '/usr/local/sbin/aim344-device-fault'
RESTORE_RUNTIME = '/var/lib/aim344-device-recovery/restore-runtime.sh'
CONFIG = Path('/etc/aim344-maintenance.json')
STAGING_UID = 0
SAFE_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
# The health suite calls fi_info to check the libfabric EFA provider, and it
# lives outside the default PATH. Omitting it made check 2 report
# 'fi_info not found -- libfabric may not be installed' and FAIL on a node whose
# EFA counts were all correct (expected=2, driver-bound=2, RDMA=2, uverbs=2),
# which is a false failure introduced by the caller's environment rather than by
# the device. The prior qualification wrapper set the same directories.
TOOL_PATH = '/opt/amazon/efa/bin:/opt/amazon/openmpi/bin'
# A confirmation token from the existing helper: <instance>/<action>/<target>
TOKEN = re.compile(r'^i-[0-9a-f]+/(?:gpu-remove|efa-unbind|efa-rebind)/'
                   r'(?:GPU-[0-9a-f-]+|[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])$')


class Refusal(Exception):
    pass


def environment(config):
    return {'PATH': f'{TOOL_PATH}:{config["slurm_bin"]}:{SAFE_PATH}',
            'LC_ALL': 'C', 'HOME': '/root', 'SHELL': '/bin/false'}


def load_config(require_root_owned=True):
    info = CONFIG.lstat()
    import stat as stat_module
    if not stat_module.S_ISREG(info.st_mode):
        raise Refusal('The maintenance configuration must be a regular file.')
    if stat_module.S_IMODE(info.st_mode) & 0o077:
        raise Refusal('The maintenance configuration must be mode 0600.')
    if require_root_owned and info.st_uid != 0:
        raise Refusal('The maintenance configuration must be owned by root.')
    config = json.loads(CONFIG.read_text())
    for key in ('slurm_bin', 'participant_user', 'stage_dir', 'checks', 'suite_entry'):
        if key not in config:
            raise Refusal(f'The maintenance configuration is missing {key}.')
    if not str(config['slurm_bin']).startswith('/opt/aws/pcs/scheduler/slurm-'):
        raise Refusal('Unexpected Slurm installation.')
    return config


def parse(argv):
    """The whole accepted grammar. Anything else is refused before it runs."""
    if not argv:
        raise Refusal('No maintenance action given.')
    for item in argv:
        if not re.fullmatch(r'[A-Za-z0-9:_./=-]+', item):
            raise Refusal('Unsupported characters in the maintenance request.')
    action, rest = argv[0], argv[1:]
    if action == 'restore-runtime':
        options = {}
        remaining = list(rest)
        while remaining:
            name = remaining.pop(0)
            if name == '--participant':
                if not remaining:
                    raise Refusal('--participant needs an account name.')
                candidate = remaining.pop(0)
                if not re.fullmatch(r'[a-z][a-z0-9-]{0,31}', candidate):
                    raise Refusal('Unusable participant account name.')
                options['participant'] = candidate
            else:
                raise Refusal(f'Unsupported option for {action}: {name}')
        return action, options
    if action in ('inspect', 'boot-id', 'instance-id', 'remount-staging',
                  'efa-activity', 'admit-replacement', 'replacement-evidence'):
        if rest:
            raise Refusal(f'{action} takes no argument.')
        return action, {}
    if action in ('gpu-prepare', 'gpu-restore'):
        # `--operation TOKEN` names which round's original-state records these refer
        # to, so a restore cannot be satisfied by a record another round -- or
        # another table on this same node -- left behind. It is an argument, not an
        # environment variable, because the maintenance route is an ssh forced
        # command and sshd passes only SSH_ORIGINAL_COMMAND.
        options = {}
        remaining = list(rest)
        while remaining:
            name = remaining.pop(0)
            if name == '--operation':
                if 'operation' in options:
                    # A repeated option is a malformed request, not a request whose
                    # last value wins. The rejected parser assigned each occurrence
                    # in turn, so `--operation A --operation B` silently became B.
                    raise Refusal('--operation was given more than once.')
                if not remaining:
                    raise Refusal('--operation needs its identifier.')
                options['operation'] = remaining.pop(0)
            elif name == '--fresh-capture' and action == 'gpu-prepare':
                if 'fresh_capture' in options:
                    raise Refusal('--fresh-capture was given more than once.')
                options['fresh_capture'] = True
            else:
                raise Refusal(f'Unsupported option for {action}: {name}')
        if 'operation' not in options:
            raise Refusal(f'{action} needs the operation identifier of the round it '
                          f'belongs to.')
        return action, options
    if action == 'collect':
        if len(rest) != 1 or not re.fullmatch(r'[0-9]{1,2}', rest[0]):
            raise Refusal('collect takes one check identifier.')
        return action, {'check': rest[0]}
    if action in ('gpu-remove', 'efa-unbind', 'efa-rebind'):
        options = {}
        remaining = list(rest)
        while remaining:
            name = remaining.pop(0)
            if name == '--confirm':
                if not remaining:
                    raise Refusal('--confirm needs its token.')
                options['confirm'] = remaining.pop(0)
            elif name == '--job' and action == 'efa-unbind':
                if not remaining or not re.fullmatch(r'[0-9]+', remaining[0]):
                    raise Refusal('--job needs a numeric job identifier.')
                options['job'] = remaining.pop(0)
            else:
                raise Refusal(f'Unsupported option for {action}: {name}')
        if 'confirm' not in options:
            raise Refusal(f'{action} needs its confirmation token.')
        if not TOKEN.match(options['confirm']):
            raise Refusal('Malformed confirmation token.')
        if not options['confirm'].split('/')[1] == action:
            raise Refusal('The confirmation token names a different operation.')
        return action, options
    raise Refusal(f'Unknown maintenance action {action!r}.')


def run(argv, config, timeout):
    finished = subprocess.run([str(a) for a in argv], capture_output=True, text=True,
                              timeout=timeout, check=False, env=environment(config),
                              stdin=subprocess.DEVNULL)
    sys.stdout.write(finished.stdout)
    sys.stderr.write(finished.stderr)
    return finished.returncode


def collect(check, config):
    """Run one read-only health check on this node through the maintenance route.

    A drained node cannot accept an ordinary batch job, so this runs the pinned
    suite directly here. It is the suite's own entry point; no command shadows
    nvidia-smi or sinfo, and no verdict is rewritten.
    """
    allowed = [str(c) for c in config['checks']]
    if str(check) not in allowed:
        raise Refusal(f'Check {check} is not enabled on this node.')
    entry = Path(config['suite_entry'])
    if not entry.is_absolute() or entry.is_symlink() or not entry.is_file():
        raise Refusal(f'The pinned health suite entry point is not a regular file: {entry}')
    revision = entry.parents[2] / 'REVISION'
    if revision.is_file():
        sys.stdout.write(f'# suite_revision={revision.read_text().strip()}\n')
    sys.stdout.write(f'# suite_entry_sha256={_sha256(entry)}\n')
    results = Path('/var/lib/aim344-device-recovery/maintenance-collect') / f'check-{check}'
    results.mkdir(parents=True, exist_ok=True)
    return run(['/bin/bash', str(entry), '--check', str(check), '--verbose',
                '--timeout', str(config.get('check_timeout', 900)),
                '--results-dir', str(results)], config,
               timeout=int(config.get('check_timeout', 900)) + 120)


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


RECORD_DIR = Path('/var/lib/aim344-device-recovery')
# The node's telemetry owner, named once. It holds the GPU, which is why
# device-fault.sh refuses an idle removal while it runs.
TELEMETRY_UNIT = 'nvidia-dcgm.service'
# The active-state words systemd reports. Read off the deployed systemd rather than
# from documentation: on the target, `is-active nvidia-dcgm.service` prints `active`
# with rc 0, and a unit that does not exist at all prints `inactive` with rc 4
# (participant-revision/runs/rework7-read-telemetry-semantics/output.log:3-5). So a
# nonzero return code is not by itself a usable signal, and the word alone is not
# either: the same word `inactive` is what a NONEXISTENT unit reports.
TELEMETRY_STATES = ('active', 'inactive', 'failed', 'activating', 'deactivating',
                    'reloading')
# The only LoadState this helper accepts as 'the expected unit is present and its
# active state means something'. Everything else -- `not-found` for a unit that does
# not exist, `masked`, `bad-setting`, `error` -- is refused, because for all of them
# systemd still reports ActiveState=inactive and `is-active` still prints the word
# `inactive`.
#
# Measured rather than assumed, on systemd 255.4-1ubuntu8.17:
#   `systemctl show -p LoadState -p ActiveState ssh.service`      -> loaded/active
#   `... apt-daily.service`                                      -> loaded/inactive
#   `... aim344-nonexistent-unit.service`                        -> not-found/inactive
#   `... cryptdisks-early.service` (masked)                      -> masked/inactive
# and `systemctl is-active` prints `inactive` for the last two while exiting 4 and 3
# respectively. So a missing or masked unit is indistinguishable from a genuinely
# stopped one on the word, and distinguishable on LoadState.
# (participant-revision/runs/rework9-read-loadstate/output.log)
LOADED_STATE = 'loaded'
# The only two this helper will record as an original state, and therefore the only
# two it will ever put back. `activating`, `deactivating` and `reloading` are
# transitions, not states to restore to; `failed` is a state this helper has no
# defined restoration for, and starting or stopping a unit that was failed before
# the round would be a change rather than a restoration.
#
# The consequence is deliberate and it is the fail-closed direction: a round is
# refused BEFORE anything is mutated when the telemetry owner is in a state this
# helper could not put back, rather than being allowed to start and then having its
# restoration invented afterwards.
RESTORABLE_TELEMETRY_STATES = ('active', 'inactive')
# Words that mean the unit is mid-transition, so one observation is not yet an
# answer. systemd distinguishes these from settled states
# (systemd/src/basic/unit-def.c, v255: UNIT_ACTIVATING = "activating",
# UNIT_DEACTIVATING = "deactivating", UNIT_RELOADING = "reloading"), and
# `systemctl is-active` counts UNIT_RELOADING among its good states
# (src/systemctl/systemctl-is-active.c), which is precisely why 'not the word
# active' cannot be read as 'inactive'.
TRANSIENT_TELEMETRY_STATES = ('activating', 'deactivating', 'reloading')
# An operation token, as the coordinator's session helper generates it:
# <assignment>/<round>/<32 hex>. The random part is what makes it unique per
# accepted start; the assignment and round are there so a facilitator reading the
# record can tell which round it belongs to. This helper does not parse meaning out
# of it -- it requires the record and the request to carry the SAME token, which is
# what an assignment-local round number could not establish.
OPERATION_TOKEN = re.compile(r'^[a-z0-9][a-z0-9._-]{0,63}/[0-9]{1,6}/[0-9a-f]{32}$')


def _operation_token(options=None):
    """The operation this request belongs to, or a Refusal.

    Both `gpu-prepare` and `gpu-restore` require it. The rejected label was
    `str(state.round)`, and the round number is local to one assignment: two tables
    sharing an exercise node both start at round 1 (device-session.py State
    defaults, StateStore files named per assignment), and the target writes every
    assignment's originals to the same two files. So table 2's round-1 restore was
    satisfied by table 1's round-1 record -- including in the interrupted-preparation
    case the intent marker exists to make safe, where table 2's prepare never wrote
    a record at all.

    A token is generated by the coordinator's root helper once per accepted start,
    persisted in the round's state record before any side effect, and reused
    unchanged by every retry of that round. It never comes from participant input:
    the participant drives one fixed verb per connection and the forced command
    passes them no arguments.
    """
    value = str((options or {}).get('operation') or '').strip()
    if not value:
        raise Refusal('This maintenance action needs the operation identifier of '
                      'the round it belongs to (--operation).')
    if not OPERATION_TOKEN.match(value):
        raise Refusal(f'Unusable operation identifier {value!r}.')
    return value


def _sync_directory(directory):
    """Put a directory ENTRY on stable storage, or refuse.

    Each record below is created by a rename, and fsync(2) states that an fsync of
    the file "does not necessarily ensure that the entry in the directory containing
    the file has also reached disk. For that an explicit fsync() on a file descriptor
    for the directory is also needed." gpu_prepare's records are read later as the
    only observation of the node's original state, so a record whose directory entry
    did not survive is an original this round can no longer put back.

    EVERY failure refuses, EINVAL included: EINVAL means this filesystem does not
    support synchronizing the entry at all, so the durability the mutation is about
    to rely on was not established. 'The platform cannot establish it' is not 'it is
    established', and the caller has changed nothing at that point.
    """
    handle = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(handle)
    except OSError as error:
        raise Refusal(
            f'{directory} could not be synchronized to storage ({error}), so a '
            f'record written in it is not established to survive a restart; '
            f'nothing was changed on this node.') from error
    finally:
        os.close(handle)


def _write_record(name, value, operation):
    """One original-state record: the value, its operation, and when it was taken.

    Written to a temporary name and renamed, so a reader never sees a half-written
    record -- an empty or truncated file is exactly what the rejected code read as
    'restoration unnecessary'.

    Both the file's data and the directory entry are synchronized before this
    returns, because the caller's next step is the mutation this record is supposed
    to make undoable. A rename is atomic for a reader of this filesystem; it is not
    evidence that either version reached storage, and the restoration side has no
    other source for the original state. A synchronization failure therefore raises
    rather than returning, and `gpu_prepare` performs no mutation after it.

    The write itself is still the same `write_text` call, and the synchronization is
    a separate step on the same file, so a failure of either one stops the caller
    before the mutation. Opening the written file again to synchronize it is what
    keeps this ordering explicit: the data is on the filesystem, then it is on
    storage, then the entry naming it is on storage, and only then does the caller
    change the node.
    """
    body = json.dumps({'value': value, 'operation': operation,
                       'recorded_at': time.time()}, sort_keys=True) + '\n'
    target = RECORD_DIR / name
    temporary = RECORD_DIR / f'{name}.new-{os.getpid()}'
    temporary.write_text(body)
    handle = os.open(temporary, os.O_WRONLY)
    try:
        os.fsync(handle)
    except OSError as error:
        raise Refusal(
            f'The original-state record {target} could not be synchronized to '
            f'storage ({error}), so it is not established to survive a restart; '
            f'nothing was changed on this node.') from error
    finally:
        os.close(handle)
    os.replace(temporary, target)
    _sync_directory(RECORD_DIR)


def _read_record(path, operation, allowed, description):
    """The recorded value for THIS operation, or a Refusal naming what is wrong.

    Missing, empty, unparseable, belonging to another operation, carrying no
    operation at all, or holding a value this helper cannot put back are all
    failures. None of them is evidence that restoration is unnecessary, which is the
    S4 defect.

    A record written before this change carries either a bare word or a numeric
    `round` label. Neither identifies the round that wrote it -- that is S4-B -- so
    both are refused. An in-flight round with unknown originals stays isolated;
    this route does not invent a restoration or authorize reuse of that instance.
    """
    if not path.is_file():
        raise Refusal(f'No recorded {description} to restore ({path} is absent); '
                      f'this node keeps its drain.')
    raw = path.read_text().strip()
    if not raw:
        raise Refusal(f'The recorded {description} is empty ({path}); this node '
                      f'keeps its drain.')
    try:
        record = json.loads(raw)
    except ValueError:
        raise Refusal(f'The recorded {description} is not readable ({path}: '
                      f'{raw[:80]!r}); this node keeps its drain.')
    if not isinstance(record, dict):
        raise Refusal(f'The recorded {description} has an unexpected shape '
                      f'({path}); this node keeps its drain.')
    recorded_operation = record.get('operation')
    if not recorded_operation:
        raise Refusal(f'The recorded {description} names no operation ({path}), so '
                      f'nothing establishes that it belongs to this round; this '
                      f'node keeps its drain; use participant replace.')
    if str(recorded_operation) != operation:
        raise Refusal(f'The recorded {description} belongs to operation '
                      f'{recorded_operation}, not {operation}; this node keeps its '
                      f'drain; use participant replace.')
    value = record.get('value')
    if value not in allowed:
        raise Refusal(f'Unusable recorded {description}: {value!r}; this node keeps '
                      f'its drain.')
    return value


def efa_activity(config):
    """Sample the SELECTED EFA's byte counters and the node's queue, twice.

    This exists for the plan's active-EFA condition, which requires observing
    recent collective progress and increasing traffic on the device about to be
    unbound, immediately before the mutation. Neither the participant nor the
    coordinator can be trusted to assert that: the participant's own job name is
    not evidence that anything is moving, and the coordinator does not see this
    node's counters. So the observation is made here, on the target, by root,
    reading the same root-owned allowlist that names which device may be touched.

    The device is the provisioned `efa_rdma_device`, never a caller-supplied name,
    so this cannot be pointed at another device. Two samples separated by a real
    interval are printed as one JSON record; the caller decides what counts as
    increasing. Nothing is mutated and no verdict is assigned here.
    """
    fault_config = json.loads(Path('/etc/aim344-device-fault.json').read_text())
    device = fault_config['efa_rdma_device']
    if not re.fullmatch(r'[a-z0-9]+', device):
        raise Refusal('The provisioned RDMA device name is unusable.')
    node = fault_config['slurm_node']
    interval = float(config.get('activity_interval_seconds', 6))
    if not 1 <= interval <= 60:
        raise Refusal('The configured activity interval is outside 1-60 s.')
    counters = ('tx_bytes', 'rx_bytes', 'rdma_write_bytes', 'tx_pkts', 'rx_pkts')

    def sample():
        port_root = Path('/sys/class/infiniband') / device / 'ports'
        values = {}
        for port in sorted(port_root.glob('*')):
            for name in counters:
                path = port / 'hw_counters' / name
                if path.is_file():
                    values[f'{port.name}/{name}'] = int(path.read_text().strip())
        if not values:
            # An absent counter set is not zero traffic. It means the device is
            # not exposing counters, which must be reported rather than read as
            # "no activity".
            raise Refusal(f'No byte counters are exposed for {device}; activity '
                          'cannot be observed and must not be assumed.')
        return values

    def jobs():
        listing = subprocess.run(
            [str(Path(config['slurm_bin']) / 'squeue'), '-h', '-w', node,
             '-o', '%i|%u|%j|%T|%M'],
            capture_output=True, text=True, check=False, env=environment(config))
        rows = []
        for line in listing.stdout.splitlines():
            parts = line.strip().split('|')
            if len(parts) >= 5 and parts[0]:
                rows.append({'id': parts[0], 'user': parts[1], 'name': parts[2],
                             'state': parts[3], 'runtime': parts[4]})
        return rows, listing.returncode

    first = sample()
    started = time.monotonic()
    jobs_before, queue_rc_before = jobs()
    time.sleep(interval)
    second = sample()
    elapsed = time.monotonic() - started
    jobs_after, queue_rc_after = jobs()
    deltas = {name: second[name] - first[name] for name in second
              if name in first}
    sys.stdout.write(json.dumps({
        'action': 'efa-activity',
        'device': device,
        'slurm_node': node,
        'interval_s': elapsed,
        'counters_before': first,
        'counters_after': second,
        'deltas_bytes': deltas,
        'jobs_before': jobs_before,
        'jobs_after': jobs_after,
        'queue_readable': queue_rc_before == 0 and queue_rc_after == 0,
    }, sort_keys=True) + '\n')
    return 0


def _read_unit(config):
    """One reading of the expected unit: (load_state, active_state).

    `systemctl show -p LoadState -p ActiveState` is used rather than `is-active`
    because `is-active` cannot answer the question this helper has to ask. It prints
    exactly one word, and the word `inactive` is what systemd reports for a unit that
    is genuinely stopped, for a unit that DOES NOT EXIST, and for a masked unit
    alike; the exit status differs between those cases but a status is not a state
    and it is not the same across builds. Measured on systemd 255.4-1ubuntu8.17:
    `apt-daily.service` -> loaded/inactive with `is-active` rc 3,
    `aim344-nonexistent-unit.service` -> not-found/inactive with rc 4, a masked unit
    -> masked/inactive with rc 3
    (participant-revision/runs/rework9-read-loadstate/output.log). The target's own
    reading agrees on the not-found case: rc 4 and the word `inactive`
    (participant-revision/runs/rework7-read-telemetry-semantics/output.log:5).

    Both properties come from one invocation, so the presence of the unit and its
    active state are the same observation rather than two that could disagree.
    Returns ('', '') for a reading that produced neither property, which is what a
    failed invocation leaves behind; the caller refuses it rather than reading it as
    a stopped service.
    """
    finished = subprocess.run(['/usr/bin/systemctl', 'show',
                               '-p', 'LoadState', '-p', 'ActiveState',
                               TELEMETRY_UNIT],
                              capture_output=True, text=True, check=False,
                              env=environment(config))
    properties = {}
    for line in (finished.stdout or '').splitlines():
        if '=' in line:
            name, value = line.split('=', 1)
            properties[name.strip()] = value.strip()
    return properties.get('LoadState', ''), properties.get('ActiveState', '')


def _observe_telemetry(config, wanted, attempts=None, pause=None):
    """Read the unit until it is settled, and return (load_state, active_state).

    Three of the active-state words systemd reports are transitions rather than
    states (TRANSIENT_TELEMETRY_STATES). A single observation of `activating` is not
    evidence of anything: it is the unit on its way somewhere. So the observation is
    repeated for a bounded interval, and what is returned is the last reading -- which
    the caller compares against the state it was restoring to and fails closed on
    anything else.

    A LoadState that is not `loaded` returns immediately. Waiting cannot make an
    absent unit appear or unmask a masked one, and the caller must refuse it rather
    than accept its `inactive` as a stopped service.

    An empty active state is what a failed invocation leaves behind, and it is
    returned as the empty string rather than being read as a state.

    The wait stops as soon as `wanted` is observed, so the ordinary case costs one
    call. The bound comes from the configuration so the target can be given a longer
    one without editing this helper; it is a wait, never a retry of the operation.
    """
    if attempts is None:
        attempts = int(config.get('telemetry_settle_attempts', 6))
    if pause is None:
        pause = float(config.get('telemetry_settle_pause_seconds', 1.0))
    attempts = max(1, int(attempts))
    load, state = '', ''
    for attempt in range(attempts):
        load, state = _read_unit(config)
        if load != LOADED_STATE:
            # Not the expected unit, or not a unit whose state can be restored.
            return load, state
        if state == wanted:
            return load, state
        if state not in TRANSIENT_TELEMETRY_STATES:
            # A settled word that is not the one wanted. Waiting longer would not
            # make it the right one, and the caller must see it.
            return load, state
        if attempt + 1 < attempts:
            time.sleep(float(pause))
    return load, state


def _existing_record(path, operation, allowed):
    """This operation's recorded value here: ('usable', value), or a state word.

    Returns one of:

      ('usable', value)   a record for THIS operation holding a value in `allowed`
      ('absent', None)    no record file at all
      ('untagged', None)  a record naming NO operation at all
      ('foreign', None)   a record naming a DIFFERENT operation
      ('damaged', None)   a record for this operation that cannot be used: not
                          readable as JSON, not a dict, or holding a value outside
                          `allowed`

    Absence alone does not authorize capture. Only an explicitly authorized new
    start can record over absent or foreign records. Untagged and damaged records
    are ambiguous even on that path: neither establishes whose original was lost.

    Never raises. Restoration reads the same files through `_read_record`, which
    refuses instead, because there the absence of a usable record is a failure.
    """
    if not path.is_file():
        return 'absent', None
    try:
        record = json.loads(path.read_text().strip() or 'null')
    except (OSError, ValueError):
        return 'damaged', None
    if not isinstance(record, dict):
        return 'damaged', None
    recorded_operation = str(record.get('operation') or '')
    if not recorded_operation:
        return 'untagged', None
    if recorded_operation != operation:
        return 'foreign', None
    value = record.get('value')
    if value not in allowed:
        return 'damaged', None
    return 'usable', value


# Durable evidence that ONE operation has already reached the point of capturing the
# node's original state. Written before the first original-state record and never
# rewritten for the same operation, so `gpu_prepare` can tell 'this operation has
# captured, and a record is missing' from 'this operation has not captured yet'.
#
# This is the minimum needed to answer the review's fourth finding. The records
# themselves cannot answer it: a record whose `operation` field is gone names no round,
# so a repeat cannot tell whether the missing token was its own -- and if it was, the
# surviving half is the only observation of the node's original state, and overwriting
# it from the already-changed node destroys it permanently while reporting success.
# There is nothing to reconstruct it from afterwards, which is why the evidence has to
# be written before the side effects rather than derived after them.
#
# One file PER OPERATION, named by a digest of the token, rather than one file naming
# the latest. A single file would answer 'has any operation captured here', and a later
# operation's capture would overwrite it, so an earlier operation's repeat would read
# 'not captured' and be free to recapture from a node it had already changed. Two
# rounds cannot in fact hold this node at once through the shipped controller, but this
# helper is reached by a forced command and must not depend on the caller's exclusion
# for the property it is here to guarantee. The digest keeps the name filesystem-safe:
# the token contains '/'.
CAPTURE_MARKER_PREFIX = 'capture-started'


def _capture_marker_name(operation):
    import hashlib
    digest = hashlib.sha256(operation.encode()).hexdigest()[:32]
    return f'{CAPTURE_MARKER_PREFIX}-{digest}.txt'


def _has_captured_before(operation, records=()):
    """Has THIS operation already begun capturing the node's original state?

    Two independent kinds of evidence, and either one answers yes:

      its own marker is present, or exists and cannot be read. The marker is written
      before the first record, so the harm being prevented -- silently recapturing
      after side effects -- means an unreadable marker counts as 'a capture may have
      happened'. Another operation's marker is irrelevant: each has its own file.

      a surviving record already names this operation. That record IS a capture by this
      operation; nothing else could have written its token. This half of the test is
      what makes the answer correct for a pair written before the marker existed: such
      a record is intact and markerless, and reading only the marker classified it as
      'never captured', so a later loss of the other record's token let both be
      overwritten from the already-changed node. A record cannot be evidence for its
      own survival, but it is evidence that the capture happened.

    `records` is the (state word, value) result of `_existing_record` for each file, in
    any order; only the state words are used. It defaults to empty, which asks the
    narrower question 'is there a marker for this operation', and that is what a caller
    comparing two operations' markers wants.
    """
    if any(state == 'usable' for state, _ in records):
        return True
    path = RECORD_DIR / _capture_marker_name(operation)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text().strip() or 'null')
    except (OSError, ValueError):
        return True
    if not isinstance(record, dict):
        return True
    return str(record.get('operation') or '') == operation


def gpu_prepare(config, options=None):
    """Record and pause exactly what the existing helper requires to be paused.

    device-fault.sh:113-119 refuses idle GPU removal unless persistence is
    disabled and no compute process holds the selected GPU. It does not change
    those itself, so this records the prior state on the root volume and then
    disables persistence and pauses the node's telemetry owner. gpu_restore puts
    both back. Nothing here decides whether the node is healthy.

    Both records name the operation they belong to, so a restore cannot be satisfied
    by a record another round -- or another table sharing this node -- left behind.
    The operation identifier is generated once per accepted start by the
    coordinator's root helper and arrives as `--operation` on the request.

    The telemetry state is recorded only when it is one this helper can put back:
    `active` or `inactive`, observed on a unit systemd reports as `loaded`. A
    transitional or failed unit refuses here, before anything is mutated, rather than
    being recorded as an original state whose restoration would then have to be
    invented. So does a unit that is absent or masked: systemd reports the word
    `inactive` for those too, and `inactive` recorded from a unit that does not exist
    is not an original state, it is the absence of one.

    A REPEAT of this action for the same operation does not re-record. The first
    accepted call is what observed the node before this round changed it; by the
    second, telemetry is already paused and persistence already disabled, so
    re-recording would replace the round's original state with the state this round
    itself produced -- and the restoration would then put back `inactive`/`Disabled`
    as though that were how the node was found. A repeat therefore keeps the existing
    records and reports them, and re-applies the pause and the persistence change from
    FRESH observations of the node, because what still needs doing is a fact about the
    node now rather than about what was recorded. That makes the action idempotent per
    operation, which is a property of THIS helper. What it is not is evidence that a
    caller retries: the shipped coordinator does not re-send `gpu-prepare` for a round
    in progress -- `DeviceSession.start` refuses or returns `already_started`, and
    recovery calls `gpu-restore`. Idempotence is required here because a helper that
    could be called twice must not lose the original either way, not because something
    currently calls it twice.

    Whether a repeat may proceed at all is decided from durable evidence, not from a
    single file. Two things establish that this operation has already captured: its
    own capture marker, written before the first original-state record, and a surviving
    record that names this operation, which nothing but this operation could have
    written. Either one means a record that is now missing or unattributable is a LOST
    original, and no fresh reading can stand in for it, because the node has already
    been changed by this round. Such a repeat refuses with the drain in place. The
    second kind of evidence is what covers a pair written before the marker existed: on
    the marker alone, an intact markerless pair read as 'never captured', so a later
    loss of one token let both originals be overwritten. A record naming no operation
    is refused for the same reason when this operation has captured -- it may be this
    round's own record with its token gone, and overwriting it would destroy the only
    observation of the original state while reporting success.
    """
    fault_config = json.loads(Path('/etc/aim344-device-fault.json').read_text())
    gpu = fault_config['gpu_uuid']
    operation = _operation_token(options)
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    mode_file = RECORD_DIR / 'selected-persistence-mode.txt'
    telemetry_file = RECORD_DIR / 'native-dcgm-state.txt'
    # What THIS operation already recorded, if it has run before.
    mode_state, recorded_mode = _existing_record(
        mode_file, operation, ('Enabled', 'Disabled'))
    telemetry_state, recorded_telemetry = _existing_record(
        telemetry_file, operation, RESTORABLE_TELEMETRY_STATES)
    resuming = mode_state == 'usable' and telemetry_state == 'usable'
    # Has THIS operation already captured the node's original state? Its own marker is
    # written before the first record, so a marker present means a capture happened here
    # even when no usable record survives -- which is exactly the case a repeat must not
    # paper over.
    captured_before = _has_captured_before(
        operation, ((mode_state, recorded_mode), (telemetry_state, recorded_telemetry)))
    # A record this operation cannot use, after this operation has already captured.
    # Not "record again": by the time a repeat runs, this round has paused telemetry and
    # disabled persistence, so a fresh reading is the state the ROUND produced, not the
    # state it found. Writing that as the original state would make the loss permanent
    # and then report a successful restoration to it. So the round stops with its drain
    # in place and says which record is unusable.
    #
    # Unknown originals cannot become fresh ones merely because both tokens, both
    # files, or a legacy marker are missing. The trusted controller supplies the
    # explicit flag ONLY for a newly generated operation under its pair lock. It
    # never resends it for an in-flight round. The flag does not override surviving
    # evidence of capture or make an untagged/damaged record attributable.
    unusable = (('damaged', 'untagged', 'foreign', 'absent') if captured_before
                else ('damaged', 'untagged'))
    lost = [name for name, condition in
            (('the telemetry state', telemetry_state),
             ('the persistence mode', mode_state))
            if condition in unusable
            or (condition == 'absent' and not resuming
                and 'usable' in (mode_state, telemetry_state))]
    if lost:
        raise Refusal(
            f'The exercise node\'s original state is uncertain: '
            f'{" and ".join(lost)} can no longer be read from its records. The node '
            f'may already have been changed by this round, so a fresh reading could '
            f'record what this round did rather than what it found; nothing was '
            f'changed and this node keeps its drain. It cannot be reused through '
            f'this capture route.')
    if not resuming and not (options or {}).get('fresh_capture', False):
        raise Refusal(
            'No intact originals belong to this operation, and this request does '
            'not authorize a first capture. Missing legacy records are not evidence '
            'of a new round; nothing was changed and this node keeps its drain.')

    mode = subprocess.run(['/usr/bin/nvidia-smi', '-i', gpu,
                           '--query-gpu=persistence_mode',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True,
                          env=environment(config)).stdout.strip()
    if mode not in ('Enabled', 'Disabled'):
        raise Refusal(f'Unexpected persistence mode: {mode!r}')
    # One settled reading of the telemetry owner. A unit caught mid-transition is
    # waited on, because 'activating' is not a state and recording it would make the
    # round's original state unrestorable.
    load, telemetry = _observe_telemetry(config, 'active')
    if not load and not telemetry:
        # Neither property came back: the invocation itself failed. That is an
        # unreadable observation, not a unit in a state, and not an absent unit
        # either -- naming it as one would claim more than was observed.
        raise Refusal(f'Unexpected {TELEMETRY_UNIT} observation: neither its load '
                      f'state nor its active state could be read; nothing was '
                      f'changed.')
    if load != LOADED_STATE:
        # The expected unit is not present as a unit this helper can restore. A
        # `not-found` or `masked` LoadState carries ActiveState=inactive, so without
        # this the round would record 'inactive' as its original state and later
        # report a restoration of a unit that was never there.
        raise Refusal(f'{TELEMETRY_UNIT} reports LoadState={load or "unreadable"!r}, '
                      f'not {LOADED_STATE!r}, so this round cannot establish that '
                      f'the expected telemetry unit is present; its state word '
                      f'({telemetry or "unreadable"}) is not an original state. '
                      f'Nothing was changed. Use participant recover, then replace if required.')
    # Whether the observation is usable at all is a fact about THIS reading, so it is
    # tested on a first call and on a repeat alike. Confining these to the first-capture
    # branch was the defect: a repeat with valid saved originals skipped them, so a unit
    # that was `activating`, `failed` or unreadable fell through to the actions below,
    # matched neither `active` nor a persistence change, and returned 0 with the node
    # not prepared and success reported.
    if telemetry in TRANSIENT_TELEMETRY_STATES:
        raise Refusal(f'{TELEMETRY_UNIT} is {telemetry} and did not settle, so this '
                      f'round cannot establish what state it is in; nothing was '
                      f'changed.')
    if telemetry not in TELEMETRY_STATES:
        # An empty string from a failed invocation, or a truncated read. Recording it
        # unchecked is what let a missing or unusable record read as "restoration
        # unnecessary" later; acting on it unchecked is what let a repeat report a
        # preparation it had not observed.
        raise Refusal(f'Unexpected {TELEMETRY_UNIT} state: {telemetry!r}; '
                      f'nothing was changed.')
    if resuming:
        # This operation already captured the node's original state. The records are
        # kept exactly as they are -- they are the only observation of the node before
        # this round changed it.
        #
        # The capture evidence is preserved as well as the records. An intact pair
        # written before the capture marker existed carries its evidence only in the
        # records' own `operation` fields, and that was the remaining gap: accepting
        # such a repeat left it markerless, so a LATER loss of both tokens left
        # `_has_captured_before` with neither kind of evidence, and the next call read
        # 'never captured' and recorded the already-paused, already-disabled node as
        # its original state. Writing the marker here on an accepted repeat makes the
        # capture durable independently of the records that can lose their tokens.
        # This is the same marker the first-capture branch writes, and it is written
        # for the operation whose records were just accepted, so it asserts nothing
        # the pair does not already establish.
        if not _has_captured_before(operation):
            _write_record(_capture_marker_name(operation), 'started', operation)
        # A previous rename may have succeeded but its directory fsync failed.
        # Visible records therefore do not establish durability on retry. Sync
        # their existing bytes (including the marker), then their directory,
        # without rewriting originals from the node's current state.
        for path in (telemetry_file, mode_file,
                     RECORD_DIR / _capture_marker_name(operation)):
            try:
                handle = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(handle)
                finally:
                    os.close(handle)
            except OSError as error:
                raise Refusal(
                    f'The retained original-state record {path} could not be '
                    f'synchronized to storage ({error}); nothing was changed '
                    f'and this node keeps its drain.') from error
        _sync_directory(RECORD_DIR)
        # What is NOT taken from them is the decision about what still has to be done.
        # The rejected code assigned `mode, telemetry = recorded_mode,
        # recorded_telemetry` here and then tested those values below, so a repeat of a
        # round whose original was `inactive`/`Disabled` skipped both actions -- and if
        # something had started the service or re-enabled persistence in between, the
        # repeat returned 0 with the node NOT prepared. The observations above describe
        # the node as it is now, which is what the pause and the persistence change are
        # a function of, so they are used for that and the records are used for the
        # report.
        sys.stdout.write(f'recorded persistence_mode={recorded_mode} '
                         f'{TELEMETRY_UNIT}={recorded_telemetry} operation={operation} '
                         f'(kept from this operation\'s earlier preparation; '
                         f'now {mode}/{telemetry})\n')
    else:
        if telemetry not in RESTORABLE_TELEMETRY_STATES:
            # Only reachable on a FIRST capture: this is about what can be recorded as
            # an original state, not about what can be observed. `failed` is a state
            # this route has no defined restoration for, so no round starts on it.
            raise Refusal(f'{TELEMETRY_UNIT} is {telemetry}, which this route has no '
                          f'defined restoration for, so it will not start a round it '
                          f'cannot undo; nothing was changed. Use participant recover, then replace if required.')
        # The capture marker is written BEFORE the first record, so a later repeat can
        # tell 'this operation captured and a record is missing' from 'this operation
        # has not captured yet'. A record naming this operation carries the same
        # evidence once it exists, which is what covers a pair written before this
        # marker did (see `_has_captured_before`).
        _write_record(_capture_marker_name(operation), 'started', operation)
        # The telemetry record is written FIRST, and the pair is written before either
        # state is changed, so a crash between the two writes leaves at most one
        # record and no mutation. That case is refused above rather than re-recorded,
        # because a repeat cannot tell whether the missing half was ever written.
        _write_record('native-dcgm-state.txt', telemetry, operation)
        _write_record('selected-persistence-mode.txt', mode, operation)
        sys.stdout.write(f'recorded persistence_mode={mode} '
                         f'{TELEMETRY_UNIT}={telemetry} operation={operation}\n')
    # Both actions are decided from the observations of the node made above, on a first
    # call and on a repeat alike, and each is confirmed. A preparation that reports
    # success has established that the node IS prepared, rather than that it once was.
    if telemetry == 'active':
        subprocess.run(['/usr/bin/systemctl', 'stop', TELEMETRY_UNIT],
                       check=True, env=environment(config))
        sys.stdout.write(f'paused {TELEMETRY_UNIT}\n')
        settled_load, settled = _observe_telemetry(config, 'inactive')
        if settled_load != LOADED_STATE or settled != 'inactive':
            raise Refusal(
                f'{TELEMETRY_UNIT} did not stop: it reports '
                f'LoadState={settled_load or "unreadable"} '
                f'ActiveState={settled or "unreadable"}, so this round cannot '
                f'establish that the telemetry owner has released the GPU; the '
                f'recorded original state is unchanged and this node keeps its drain.')
    elif telemetry != 'inactive':
        # Observed, settled, loaded, and neither `active` nor `inactive`: `failed` is
        # the word this reaches, on a repeat whose saved originals are valid. There is
        # no action here that makes the GPU free -- stopping a failed unit does not
        # establish that it released the device -- so the preparation does not report a
        # success it has not observed. The originals are untouched.
        raise Refusal(
            f'{TELEMETRY_UNIT} is {telemetry}, so this round cannot establish that the '
            f'telemetry owner has released the GPU. The original state this operation '
            f'recorded is unchanged and nothing was changed here; this node keeps its '
            f'drain. Use participant recover, then replace if required.')
    if mode != 'Disabled':
        subprocess.run(['/usr/bin/nvidia-smi', '-i', gpu, '--persistence-mode=0'],
                       capture_output=True, text=True, check=True,
                       env=environment(config))
        now = subprocess.run(['/usr/bin/nvidia-smi', '-i', gpu,
                              '--query-gpu=persistence_mode',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True,
                             env=environment(config)).stdout.strip()
        if now != 'Disabled':
            raise Refusal(f'Persistence is still {now} on the selected GPU.')
        sys.stdout.write('disabled persistence on the selected GPU\n')
    return 0


def gpu_restore(config, options=None):
    """Put the recorded persistence mode and telemetry owner back, or fail.

    Every branch that returns 0 has actually observed what it restored. The rejected
    version treated a missing, empty or unrecognised telemetry record exactly like
    evidence that restoration was unnecessary: it restored DCGM only under
    `if telemetry_file.is_file() and read_text().strip() == 'active'` and otherwise
    returned 0, so a round whose record was never written -- or was written by a
    different round -- resumed a node whose telemetry owner might still be paused.
    Its replacement then still returned 0 for observations that established nothing:
    an original `inactive` whose unit read `activating`, or a recorded `failed` unit
    now `active`, were both reported as restored.

    So the only two original states this helper records are the two it can put back,
    and each is confirmed by observation:

      recorded 'active'    -> start the unit, then observe loaded/`active`, else nonzero
      recorded 'inactive'  -> stop the unit if it is running, then observe
                              loaded/`inactive`, else nonzero
      unit not loaded      -> refuse; the round keeps its drain
      no usable record     -> refuse; the round keeps its drain

    A transitional observation is waited on for a bounded interval and then fails
    closed. `_observe_telemetry` carries that wait; nothing here reads 'not the word
    active' as inactive, and nothing here reads the `inactive` of an absent or masked
    unit as a restored stopped service -- both are refused on LoadState, because the
    round recorded the state of a unit that was present and a missing unit is not
    that unit in a different state.
    """
    operation = _operation_token(options)
    mode_file = RECORD_DIR / 'selected-persistence-mode.txt'
    telemetry_file = RECORD_DIR / 'native-dcgm-state.txt'
    wanted = _read_record(mode_file, operation, ('Enabled', 'Disabled'),
                          'persistence mode')
    telemetry_before = _read_record(telemetry_file, operation,
                                    RESTORABLE_TELEMETRY_STATES,
                                    f'{TELEMETRY_UNIT} state')
    fault_config = json.loads(Path('/etc/aim344-device-fault.json').read_text())
    gpu = fault_config['gpu_uuid']
    subprocess.run(['/usr/bin/nvidia-smi', '-i', gpu,
                    f'--persistence-mode={1 if wanted == "Enabled" else 0}'],
                   capture_output=True, text=True, check=True, env=environment(config))
    now = subprocess.run(['/usr/bin/nvidia-smi', '-i', gpu,
                          '--query-gpu=persistence_mode',
                          '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True,
                         env=environment(config)).stdout.strip()
    if now != wanted:
        raise Refusal(f'Persistence is {now}, expected the recorded {wanted}.')
    sys.stdout.write(f'restored persistence_mode={now}\n')

    if telemetry_before == 'active':
        subprocess.run(['/usr/bin/systemctl', 'start', TELEMETRY_UNIT],
                       check=False, env=environment(config))
        load, state = _observe_telemetry(config, 'active')
        sys.stdout.write(f'restored {TELEMETRY_UNIT}={state or "unreadable"} '
                         f'(LoadState={load or "unreadable"}, recorded '
                         f'{telemetry_before})\n')
        if load != LOADED_STATE:
            sys.stdout.write(f'{TELEMETRY_UNIT} is not a loaded unit '
                             f'(LoadState={load or "unreadable"}), so this round '
                             'cannot establish that the unit it recorded was '
                             'restored; preserve this state and use participant replace.\n')
            return 4
        if state != 'active':
            # This is an incomplete restoration, not an advisory. Returning 0 here
            # let the caller resume a node whose telemetry owner was still paused.
            sys.stdout.write(f'{TELEMETRY_UNIT} did not return to active; preserve '
                             'this state and use participant replace.\n')
            return 4
        return 0
    # Recorded 'inactive': the round deliberately stopped the service, or it was
    # already stopped. A reboot can start it, and leaving it running would be a
    # change this exercise did not ask for.
    load, state = _observe_telemetry(config, 'inactive')
    if load == LOADED_STATE and state != 'inactive':
        subprocess.run(['/usr/bin/systemctl', 'stop', TELEMETRY_UNIT],
                       check=False, env=environment(config))
        load, state = _observe_telemetry(config, 'inactive')
    sys.stdout.write(f'restored {TELEMETRY_UNIT}={state or "unreadable"} '
                     f'(LoadState={load or "unreadable"}, recorded '
                     f'{telemetry_before})\n')
    if load != LOADED_STATE:
        sys.stdout.write(f'{TELEMETRY_UNIT} is not a loaded unit '
                         f'(LoadState={load or "unreadable"}), so its state word '
                         'does not establish that the unit this round recorded is '
                         'back as it was; preserve this state and use '
                         'participant replace.\n')
        return 4
    if state != 'inactive':
        sys.stdout.write(f'{TELEMETRY_UNIT} does not read inactive but this round '
                         'recorded it inactive; preserve this state and use '
                         'participant replace.\n')
        return 4
    return 0


def _image_digest(fd, size):
    """Hash one validated descriptor in order, with bounded read-ahead.

    Cold root EBS reads on the qualified PCS image are latency-bound at queue
    depth one. Schedule at most 32 one-MiB chunks per batch, never reopen a path,
    and hash every byte in file order (not a digest of chunk digests).
    """
    chunk = 1024 * 1024
    digest = hashlib.sha256()

    def read_at(offset):
        wanted = min(chunk, size - offset)
        data = bytearray()
        while len(data) < wanted:
            part = os.pread(fd, wanted - len(data), offset + len(data))
            if not part:
                raise Refusal('Root staging image changed size while hashing.')
            data.extend(part)
        return data

    with ThreadPoolExecutor(max_workers=32) as readers:
        for start in range(0, size, chunk * 32):
            for block in readers.map(read_at, range(start, min(size, start + chunk * 32), chunk)):
                digest.update(block)
    return digest.hexdigest()


def verify_root_staging(config):
    """Successor contract: pinned root-directory images, never an NVMe bind."""
    stage = Path(config['stage_dir'])
    try:
        info = stage.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != STAGING_UID
                or info.st_mode & 0o022 or stage.resolve() != stage):
            raise Refusal('Untrusted root staging directory.')
        info = stage.parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != STAGING_UID
                or info.st_mode & 0o022):
            raise Refusal('Untrusted root staging parent.')
        mount = subprocess.run(['/usr/bin/findmnt', '-n', '-o', 'TARGET', '-T', str(stage)],
                               capture_output=True, text=True, check=False,
                               env=environment(config))
        if mount.returncode or mount.stdout.strip() != '/':
            raise Refusal('Root staging is not on the root filesystem; refusing an overlaid source.')
        pins = config.get('staging_image_sha256', {})
        for name in ('aim344.sqsh', 'nccl-baseline.sqsh'):
            expected = pins.get(name)
            if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
                raise Refusal('Root staging requires both provisioned image hashes.')
            fd = os.open(stage / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as image:
                info = os.fstat(image.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != STAGING_UID
                        or info.st_mode & 0o022 or info.st_nlink != 1):
                    raise Refusal('Untrusted root staging image.')
                observed = _image_digest(image.fileno(), info.st_size)
                after = os.fstat(image.fileno())
                if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                        info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                    raise Refusal('Root staging image changed while hashing.')
                if observed != expected:
                    raise Refusal(f'Root staging image {name} differs from its provisioned source.')
    except OSError as error:
        raise Refusal(f'Root staging source is unavailable: {error}') from error
    sys.stdout.write(f'verified pinned root staging at {stage}; no bind mount needed\n')
    return 0


def remount_staging(config):
    """Re-establish the staging bind mount that a reboot removes.

    Measured on the target: after `scontrol reboot`, /opt/dlami/nvme is mounted
    again but /opt/aim344 is not a mountpoint, so the staged images disappear
    from the path restore-runtime.sh checks and it fails with 'Missing staged
    image'. The staged tree itself survives on the ephemeral filesystem. This
    never formats storage and refuses if the source is absent, so an instance
    whose instance store really was lost is reported rather than papered over.
    """
    stage = Path(config['stage_dir'])
    source_root = Path(config.get('staging_source_root', '/opt/dlami/nvme'))
    policy = config.get('staging_policy', 'nvme-bind')
    if policy == 'root-directory':
        return verify_root_staging(config)
    if policy != 'nvme-bind':
        raise Refusal('Unsupported staging policy; no mount attempted.')
    source = source_root / stage.name
    if not source_root.is_mount():
        raise Refusal(f'{source_root} is not mounted; the staged images cannot be '
                      'restored from here. Preserve this state.')
    expected_uuid = config.get('staging_fs_uuid')
    if expected_uuid:
        observed = subprocess.run(['/usr/bin/findmnt', '-n', '-o', 'UUID',
                                   str(source_root)], capture_output=True,
                                  text=True, check=False,
                                  env=environment(config)).stdout.strip()
        if observed != expected_uuid:
            raise Refusal(f'{source_root} has filesystem UUID {observed!r}, not the '
                          f'provisioned {expected_uuid!r}. Refusing to mount.')
    if not source.is_dir():
        raise Refusal(f'The staged tree {source} is absent; the instance store may '
                      'have been replaced. Preserve this state and use participant replace.')
    for image in ('aim344.sqsh', 'nccl-baseline.sqsh'):
        if not (source / image).is_file():
            raise Refusal(f'The staged image {source / image} is absent.')
    stage.mkdir(parents=True, exist_ok=True)
    if stage.is_mount():
        sys.stdout.write(f'{stage} is already mounted\n')
    else:
        subprocess.run(['/usr/bin/mount', '--bind', str(source), str(stage)],
                       check=True, capture_output=True, text=True,
                       env=environment(config))
        sys.stdout.write(f'bind mounted {source} at {stage}\n')
    # Confirm the mount actually exposes the same directory, not just that the
    # mount command returned success.
    if os.stat(stage).st_ino != os.stat(source).st_ino:
        raise Refusal(f'{stage} does not expose {source} after mounting.')
    sys.stdout.write(f'verified {stage} exposes {source} '
                     f'(inode {os.stat(stage).st_ino})\n')
    return 0


def instance_identity():
    """This node's EC2 instance id, independent of any device's state.

    Separate from `inspect` on purpose. `inspect` calls the device helper, which
    requires the provisioned GPU PCI function to exist (device-fault.sh:64,
    'Provisioned gpu PCI function is absent'), and a GPU round removes exactly
    that function. So during a GPU fault `inspect` cannot succeed, and any caller
    that used it merely to confirm identity was refused for the wrong reason.
    Measured on the target with the fault applied: `inspect` returns rc=1 with
    that message while `board_asset_tag` still reads i-0123456789abcdef0
    (participant-revision/runs/rework2-probe-gpu-inspect/output.log).

    The device helper itself reads identity from the same DMI field before it
    looks at any device (device-fault.sh:45-46), so this is the same source, not
    a weaker one.
    """
    value = Path('/sys/devices/virtual/dmi/id/board_asset_tag').read_text().strip()
    if not re.fullmatch(r'i-[0-9a-f]+', value):
        raise Refusal('Invalid instance identity.')
    return value


def resolve_participant(config, requested):
    """Which account owns the restored checkpoint fixture on this node.

    The fixture is 0700 and owned by one account, so naming the wrong one makes
    it unreadable to the participant who is actually running. Observed on both
    nodes: the fixture was `ubuntu 700` while the authenticated participant was
    aim344-t1, and aim344-t1 could neither read nor write it
    (participant-revision/runs/rework2-inspect-{target,coordinator}/output.log).

    The requested name must appear in this node's root-owned `participants`
    allowlist. The coordinator therefore selects among accounts the facilitator
    provisioned here; it cannot introduce one, so a compromised coordinator
    cannot chown the fixture to an account of its choosing.
    """
    allowed = config.get('participants')
    if allowed is None:
        # An older configuration without the allowlist keeps exactly its previous
        # behaviour rather than silently accepting any requested name.
        allowed = [config['participant_user']]
    if not isinstance(allowed, list) or not allowed:
        raise Refusal('The maintenance configuration has an unusable participant '
                      'allowlist.')
    if requested is None:
        return config['participant_user']
    if requested not in allowed:
        raise Refusal(f'{requested} is not a participant account provisioned on '
                      'this node.')
    return requested


def main():
    if os.geteuid() != 0:
        sys.stderr.write('The maintenance route runs as root.\n')
        return 1
    original = os.environ.get('SSH_ORIGINAL_COMMAND')
    argv = sys.argv[1:] if len(sys.argv) > 1 else (original.split() if original else [])
    try:
        config = load_config()
        action, options = parse(argv)
        if action == 'boot-id':
            sys.stdout.write(Path('/proc/sys/kernel/random/boot_id').read_text())
            return 0
        if action == 'instance-id':
            sys.stdout.write(instance_identity() + '\n')
            return 0
        if action == 'replacement-evidence':
            evidence = {'instance_id': instance_identity(), 'records': {}}
            for name in ('selected-persistence-mode.txt', 'native-dcgm-state.txt'):
                path = RECORD_DIR / name
                if path.is_file() and not path.is_symlink() and path.stat().st_size <= 16384:
                    evidence['records'][name] = path.read_text()
                else:
                    evidence['records'][name] = None
            sys.stdout.write(json.dumps(evidence) + '\n')
            return 0
        if action == 'admit-replacement':
            receipt = Path('/var/lib/aim344-replacement/ready.json')
            report = json.loads(receipt.read_text())
            if report.get('instance_id') != instance_identity() or report.get('ready') is not True:
                raise Refusal('Replacement bootstrap receipt does not match this instance.')
            _write_record('replacement-admitted.json', instance_identity(), 'replacement-admission')
            return 0
        if action == 'inspect':
            return run([DEVICE_FAULT, 'inspect'], config, timeout=60)
        if action == 'collect':
            return collect(options['check'], config)
        if action == 'gpu-prepare':
            return gpu_prepare(config, options)
        if action == 'gpu-restore':
            return gpu_restore(config, options)
        if action == 'remount-staging':
            return remount_staging(config)
        if action == 'efa-activity':
            return efa_activity(config)
        if action == 'restore-runtime':
            if config.get('staging_policy') == 'root-directory':
                verify_root_staging(config)
            if not Path(RESTORE_RUNTIME).is_file():
                raise Refusal(f'{RESTORE_RUNTIME} is not installed.')
            participant = resolve_participant(config, options.get('participant'))
            sys.stdout.write(f'# fixture_owner={participant}\n')
            return run(['/bin/bash', RESTORE_RUNTIME, participant,
                        config['stage_dir']], config, timeout=600)
        command = [DEVICE_FAULT, action]
        if 'job' in options:
            command += ['--job', options['job']]
        command += ['--confirm', options['confirm']]
        return run(command, config, timeout=int(config.get('mutation_timeout', 300)))
    except Refusal as refusal:
        sys.stderr.write(str(refusal) + '\n')
        return 3
    except subprocess.TimeoutExpired:
        sys.stderr.write('External observation deadline reached; the command may '
                         'still be running. Preserve this state.\n')
        return 124


if __name__ == '__main__':
    sys.exit(main())
