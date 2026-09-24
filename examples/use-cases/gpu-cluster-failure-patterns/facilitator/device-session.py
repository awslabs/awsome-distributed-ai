#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""AIM344 participant device session helper.

Install a root-owned copy at /usr/local/sbin/aim344-device-session on the
coordinator node and reach it through an SSH forced command:

    command="/usr/local/sbin/aim344-device-session --assignment table-1",
    no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 ...

The participant supplies only a verb in SSH_ORIGINAL_COMMAND. The assignment
comes from the forced command, so the authenticated key, not participant input,
selects the target. This helper decides nothing about device health: verdicts
come from validation/gpu-cluster-healthcheck, and every device mutation goes
through the existing root-owned aim344-device-fault helper on the target.

Scope of what a participant can cause, by construction:
  start gpu | start efa | status | collect <allowlisted check> | recover | replace
No node name, PCI address, job identifier, shell command or check outside the
per-kind allowlist is accepted.
"""
import argparse
import datetime
import errno
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time

SAFE_PATH = ('/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
DEFAULT_CONFIG = Path('/etc/aim344-device-session.json')
MAINTENANCE_COMMAND = '/usr/local/sbin/aim344-maintenance'
# 'preparing' is recorded before the first mutation, so a failure part way
# through preparation still leaves a recoverable record. 'recovery-failed' keeps
# a node drained and out of service after a required check or restoration step
# did not pass; it is retryable but it is not a return to service.
PHASES = ('ready', 'preparing', 'fault-applied', 'investigating', 'recovering',
          'recovery-failed', 'replacement-required', 'runtime-ready', 'verified')
# Phases in which this assignment still holds its exercise node.
HOLDING_PHASES = ('preparing', 'fault-applied', 'investigating', 'recovering',
                  'recovery-failed', 'replacement-required')
USAGE = ('Commands: start gpu | start efa | active gpu | active efa | status | collect <check> | recover | replace')


def node_reason(node):
    """Return (body, raw), preserving untrusted Reason text for exact checks.

    Live queries use Slurm's multiline format: fields start with three spaces,
    reason continuations with ten (node_info.c). Never split on field-shaped
    words inside Reason. One-line compatibility accepts only the captured
    annotated format with a unique, terminal EC2 identity pair; other ambiguous
    one-line suffixes stay in raw and cannot establish ownership.
    """
    annotation = (r' \[[^\s@\[\]]+@[0-9]{4}-(?:0[1-9]|1[0-2])-'
                  r'(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):'
                  r'[0-5][0-9]:[0-5][0-9]\]')
    text = node.rstrip('\n')
    if '\n' in text:
        reasons = re.findall(r'^   Reason=(.*(?:\n          [^\n]*)*)', text, re.M)
    else:
        reasons = re.findall(r'(?:^|\s)Reason=(.*)', text)
    if len(reasons) != 1:
        return None, text if re.search(r'(?:^|\s)Reason=', text) else ''
    raw = reasons[0]
    if '\n' not in text:
        # A field token alone is not a boundary. Require the final annotation
        # and the exact recorded serializer suffix, not arbitrary known fields.
        suffix = re.fullmatch(r'(.*' + annotation + r') InstanceId=i-[0-9a-f]{17}'
                              r' InstanceType=[A-Za-z0-9.-]+ *', raw)
        if suffix and len(re.findall(r'(?:^|\s)InstanceId=', text)) == 1:
            raw = suffix[1]
    body = re.sub(annotation + r'\Z', '', raw, count=1)
    return body, raw


class Refusal(Exception):
    """A refusal that is safe to show the participant."""


def _open_directory(parent_fd, name, owner_uid):
    """Open one directory component relative to parent_fd, creating it if absent.

    Every step is descriptor relative and O_NOFOLLOW, so no component of the path
    can be swapped for a symlink between the check and the use: the descriptor
    refers to the inode that was opened, not to a name that may be re-resolved.
    The opened directory must belong to the expected owner, which stops a
    pre-planted directory owned by somebody else from being written through.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        handle = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        handle = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EMLINK):
            raise Refusal(f'Refusing a symlinked results component: {name}') from error
        if error.errno == errno.ENOTDIR:
            raise Refusal(f'A results component is not a directory: {name}') from error
        raise
    info = os.stat(handle)
    if not stat.S_ISDIR(info.st_mode):
        os.close(handle)
        raise Refusal(f'A results component is not a directory: {name}')
    if owner_uid is not None and info.st_uid != int(owner_uid):
        os.close(handle)
        raise Refusal(f'A results component is not owned by your account: {name}')
    return handle


def _write_under(root, relative, name_for_attempt, text, owner_uid,
                 max_attempts=64):
    """Write text into root/relative/<name>, never overwriting an existing file.

    O_EXCL is what protects fault-time evidence: a later capture cannot truncate
    an earlier one, so the caller gets a fresh -attempt-N name instead. O_NOFOLLOW
    with O_EXCL also refuses a pre-planted symlink or FIFO at the leaf, because
    O_EXCL requires that the final component not exist at all.
    """
    root = Path(root)
    descriptors = []
    try:
        try:
            top = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError as error:
            raise Refusal(f'Your results directory is missing: {root}') from error
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise Refusal('Your results directory is a symlink or not a '
                              'directory; nothing was written.') from error
            raise
        descriptors.append(top)
        info = os.stat(top)
        if owner_uid is not None and info.st_uid != int(owner_uid):
            raise Refusal('Your results directory is not owned by your account; '
                          'nothing was written.')
        for component in relative:
            descriptors.append(_open_directory(descriptors[-1], component, owner_uid))
        leaf_fd = descriptors[-1]
        for attempt in range(1, max_attempts + 1):
            name = name_for_attempt(attempt)
            try:
                handle = os.open(name,
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=leaf_fd)
            except FileExistsError:
                continue
            except FileNotFoundError as error:
                # The directory this descriptor refers to was removed while the
                # request was in flight. Nothing was written, which is the point;
                # report it as a refusal rather than a traceback.
                raise Refusal('Your results directory changed while the check was '
                              'running, so nothing was written. Run status, then '
                              'collect again.') from error
            except OSError as error:
                if error.errno in (errno.ELOOP, errno.ENXIO, errno.ESTALE):
                    raise Refusal('Refusing to write through a symlinked or '
                                  'special results file.') from error
                raise
            with os.fdopen(handle, 'w') as output:
                output.write(text)
            directory = root.joinpath(*relative)
            return directory / name
        raise Refusal('Too many captures already exist for this check in this '
                      'round; nothing was written.')
    finally:
        for handle in reversed(descriptors):
            try:
                os.close(handle)
            except OSError:
                pass


class Completed:
    def __init__(self, returncode, stdout, stderr, dispatched=None):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.dispatched = dispatched


def safe_environment(slurm_bin):
    """A fixed environment for privileged calls.

    Nothing from the participant's connection is inherited: no LD_PRELOAD, no
    PYTHONPATH, no BASH_ENV, no SSH_ORIGINAL_COMMAND.
    """
    return {'PATH': f'{slurm_bin}:{SAFE_PATH}',
            'LC_ALL': 'C',
            'TZ': 'UTC',
            'SLURM_TIME_FORMAT': 'standard',
            'HOME': '/root',
            'SHELL': '/bin/false'}


class Executor:
    """Runs the two privileged routes. No shell, fixed argv[0]."""

    def __init__(self, slurm_bin, target_host, maintenance_command=MAINTENANCE_COMMAND,
                 identity_file=None):
        self.slurm_bin = slurm_bin
        self.target_host = target_host
        self.maintenance_command = maintenance_command
        self.identity_file = identity_file
        self.known_hosts = ''

    def _run(self, argv, timeout):
        try:
            finished = subprocess.run(argv, capture_output=True, text=True,
                                      timeout=timeout, check=False,
                                      env=safe_environment(self.slurm_bin),
                                      stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as expired:
            # An external deadline is not proof the remote command ended.
            def text_of(value):
                if value is None:
                    return ''
                return value.decode('utf-8', 'replace') if isinstance(value, bytes) else value

            return Completed(124, text_of(expired.stdout),
                             text_of(expired.stderr)
                             + '\nExternal observation deadline reached; '
                               'remote command termination is not established.\n')
        return Completed(finished.returncode, finished.stdout, finished.stderr)

    def run_maintenance(self, argv, timeout=None):
        """Reach the target's root-owned maintenance forced command."""
        return self._run(self._maintenance_argv(argv), timeout or 300)

    def _maintenance_argv(self, argv, interactive=False):
        ssh = ['/usr/bin/ssh', '-n', '-o', 'BatchMode=yes',
               '-o', 'StrictHostKeyChecking=yes',
               '-o', 'ConnectTimeout=10']
        if self.identity_file:
            ssh += ['-i', self.identity_file]
        if self.known_hosts:
            ssh += ['-o', f'UserKnownHostsFile={self.known_hosts}',
                    '-o', 'GlobalKnownHostsFile=/dev/null']
        ssh += [f'root@{self.target_host}', '--'] + [str(a) for a in argv]
        if interactive:
            ssh.remove('-n')
        return ssh

    def run_active_fault(self, argv, record, timeout=90):
        """Stream existing maintenance SSH; ACK only durable off-target evidence."""
        import selectors
        try:
            process = subprocess.Popen(self._maintenance_argv(argv, interactive=True),
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       env=safe_environment(self.slurm_bin))
        except OSError as error:
            return Completed(127, '', f'FLR transport could not start; no request was sent: {error}\n',
                             dispatched=False)
        assert process.stdin is not None and process.stdout is not None
        output, pending = bytearray(), bytearray()
        deadline = time.monotonic() + timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        process.kill()  # ONLY our local SSH client, never workload PIDs
                        process.wait()
                        return Completed(124, output.decode('utf-8', 'replace'),
                                         'FLR observation timed out; remote outcome is unknown.\n')
                    if not selector.select(remaining):
                        continue
                    block = os.read(process.stdout.fileno(), 65536)
                    if not block:
                        break
                    output.extend(block)
                    pending.extend(block)
                    while b'\n' in pending:
                        raw, _, rest = pending.partition(b'\n')
                        pending[:] = rest
                        acknowledgment = record(raw.decode('utf-8', 'replace'))
                        if acknowledgment:
                            try:
                                process.stdin.write((acknowledgment + '\n').encode())
                                process.stdin.flush()
                            except OSError as error:
                                return Completed(255, output.decode('utf-8', 'replace'),
                                                 f'FLR ACK transport failed; remote outcome is unknown: {error}\n')
                return Completed(process.wait(timeout=max(1, deadline - time.monotonic())),
                                 output.decode('utf-8', 'replace'), '')
        except subprocess.TimeoutExpired:
            return Completed(124, output.decode('utf-8', 'replace'),
                             'FLR observation timed out; remote outcome is unknown.\n')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            # A failed ACK flush may fail again on close. Do not replace the
            # observation result or a record/fsync exception with that error.
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass

    def run_slurm(self, argv, timeout=None):
        return self._run([str(a) for a in argv], timeout or 60)

    def run_cloud(self, region, service, action, payload):
        # No participant environment, credential profile, endpoint or argv is accepted.
        return self._run(['/usr/local/bin/aws', service, action, '--region', region,
                          '--cli-input-json', json.dumps(payload), '--output', 'json',
                          '--no-cli-pager'], 90)


class State:
    """The small record the transitions need. Root-owned, outside the target."""

    FIELDS = ('phase', 'kind', 'started_at', 'target_node', 'target_instance_id',
              'job_id', 'boot_id', 'recovered_by', 'recovered_at', 'notes',
              'collected', 'deadline_at', 'round', 'operation', 'prepared',
              'original_node_state',
              'original_node_reason', 'reboot_completed', 'boot_id_after_reboot',
              'boot_id_before_reboot', 'reboot_requested_at', 'reboot_requests',
              'check_results', 'baseline_check_results', 'failure',
              'deadline_attempts', 'active_efa_evidence', 'active_fault')

    def __init__(self, phase='ready', kind=None, started_at=None, target_node=None,
                 target_instance_id=None, job_id=None, boot_id=None,
                 recovered_by=None, recovered_at=None, notes='', collected=None,
                 deadline_at=None, round=1, operation=None, prepared=None,
                 original_node_state=None,
                 original_node_reason=None, reboot_completed=False,
                 boot_id_after_reboot=None, boot_id_before_reboot=None,
                 reboot_requested_at=None, reboot_requests=0,
                 check_results=None, baseline_check_results=None, failure=None,
                 deadline_attempts=0, active_efa_evidence=None, active_fault=None):
        if phase not in PHASES:
            raise Refusal(f'Unusable state record: unknown phase {phase!r}.')
        self.phase = phase
        self.kind = kind
        self.started_at = started_at
        self.target_node = target_node
        self.target_instance_id = target_instance_id
        self.job_id = job_id
        self.boot_id = boot_id
        self.recovered_by = recovered_by
        self.recovered_at = recovered_at
        self.notes = notes
        self.collected = list(collected or [])
        self.deadline_at = deadline_at
        # Which round of this assignment produced the evidence. Round numbers
        # keep each round's fault-time observations from being overwritten.
        self.round = int(round or 1)
        # The operation identifier this round's original-state records on the target
        # are written under, generated once per accepted start by this root helper.
        # A round number cannot do this job: it is local to an assignment, so two
        # tables sharing an exercise node both hold a round 1, and the target keeps
        # every assignment's originals in the same two files.
        self.operation = operation
        # Preparation steps already applied, so recovery knows what to undo even
        # if the request that applied them failed afterwards.
        self.prepared = list(prepared or [])
        # The node's state and drain reason before this exercise touched it.
        self.original_node_state = original_node_state
        self.original_node_reason = original_node_reason
        # A completed reboot is recorded so a later failure does not reboot again.
        self.reboot_completed = bool(reboot_completed)
        self.boot_id_after_reboot = boot_id_after_reboot
        self.boot_id_before_reboot = boot_id_before_reboot
        # When a reboot was requested, and how many have been requested for this
        # round. Persisted before the request is issued, so a retry after a lost
        # observation reconciles the outstanding request instead of queueing
        # another one.
        self.reboot_requested_at = reboot_requested_at
        self.reboot_requests = int(reboot_requests or 0)
        # check identifier and phase -> verdict, kept distinctly per round.
        self.check_results = dict(check_results or {})
        # The same shape, captured before the fault was applied. This is the only
        # thing a post-recovery WARN may be compared against: without a recorded
        # pre-fault observation of that check there is no baseline, and a WARN
        # cannot be called pre-existing.
        self.baseline_check_results = dict(baseline_check_results or {})
        # Why recovery stopped, when it did. Cleared on a qualified recovery.
        self.failure = failure
        # How many times the deadline sweep has already tried this round. Bounds
        # unattended retries so a stuck table is reported rather than mutated on
        # every scheduled tick.
        self.deadline_attempts = int(deadline_attempts or 0)
        # What was observed on the selected EFA immediately before an active
        # injection, and whether it qualified. None for an idle round, which is
        # the distinction the review asked to stop being silent: a round with no
        # record here did not test the active path.
        self.active_efa_evidence = (dict(active_efa_evidence)
                                    if active_efa_evidence else None)
        self.active_fault = dict(active_fault) if active_fault else None

    def to_dict(self):
        return {name: getattr(self, name) for name in self.FIELDS}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise Refusal('Unusable state record.')
        unknown = set(data) - set(cls.FIELDS)
        if unknown:
            raise Refusal(f'Unusable state record: unexpected fields {sorted(unknown)}.')
        kept = {k: v for k, v in data.items()}
        try:
            state = cls(**kept)
        except TypeError as error:
            raise Refusal(f'Unusable state record: {error}') from error
        # A record that claims a fault without naming the fault is not a state
        # the helper produced; refuse it rather than acting on it.
        if state.phase != 'ready':
            if state.kind not in ('gpu', 'efa'):
                raise Refusal('Unusable state record: phase without a fault kind.')
            if not state.target_node or state.started_at is None:
                raise Refusal('Unusable state record: incomplete transition.')
        return state


def _sync_directory(directory):
    """Put a directory ENTRY on stable storage, or refuse.

    A renamed file is only durable once the directory that names it is synchronized
    too: fsync(2) says an fsync on the file "does not necessarily ensure that the
    entry in the directory containing the file has also reached disk. For that an
    explicit fsync() on a file descriptor for the directory is also needed." Every
    record this controller treats as durable evidence is created by a rename, so
    without this the evidence can vanish while the operation it authorized has
    already happened.

    O_DIRECTORY|O_RDONLY, because a directory cannot be opened for writing and this
    only needs a descriptor to synchronize. EVERY failure refuses, EINVAL included:
    EINVAL means this filesystem does not support synchronizing the entry at all, so
    the durability the caller is about to rely on was not established, and treating
    'the platform cannot do it' as 'it is done' is exactly the inference this repair
    removes. The caller has made no side effect at that point, so the round keeps its
    reservation and reports an incomplete operation; no dependent mutation runs.
    """
    handle = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(handle)
    except OSError as error:
        raise Refusal(
            f'The record directory {directory} could not be synchronized to '
            f'storage ({error}), so a record written in it is not established '
            'to survive a restart. Nothing was done that would depend on it.'
        ) from error
    finally:
        os.close(handle)


class StateStore:
    def __init__(self, directory, assignment_id, target_node=None):
        self.directory = Path(directory)
        self.assignment_id = assignment_id
        self.path = self.directory / f'{assignment_id}.json'
        self.lock_path = self.directory / f'{assignment_id}.lock'
        # Exclusion keyed by the physical exercise node, not by the assignment.
        # Two assignments pointing at the same node take different assignment
        # locks, so an assignment lock alone lets both scan the other's state,
        # find it free, and enter preparation. The pair lock below is the one
        # that makes ownership of the hardware exclusive.
        self.target_node = target_node
        self.pair_lock_path = (self.directory / f'pair-{target_node}.lock'
                               if target_node else None)
        self._lock_handle = None
        self._pair_handle = None

    def read(self):
        if self.path.is_symlink():
            raise Refusal('Unusable state record: refusing a symlinked state file.')
        if not self.path.exists():
            return State()
        info = self.path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise Refusal('Unusable state record: it must be a private regular file.')
        try:
            data = json.loads(self.path.read_text())
        except ValueError as error:
            raise Refusal(f'Unusable state record: {error}') from error
        return State.from_dict(data)

    def write(self, state):
        """Persist this round's record, durably, before its dependent side effect.

        Every caller writes here BEFORE the operation the record is supposed to make
        recoverable -- the drain, the reboot request, the device call -- and recovery
        then reads the ABSENCE of a fact as evidence that its operation was never
        issued. That reading is only sound if the write survives the host, so the
        file's data and the directory entry that names it are both synchronized here
        rather than merely renamed into place. `os.replace` gives readers an atomic
        switch between two versions; it does not put either version on stable
        storage. fsync(2) is explicit that the two are separate: "Calling fsync()
        does not necessarily ensure that the entry in the directory containing the
        file has also reached disk. For that an explicit fsync() on a file descriptor
        for the directory is also needed."

        A synchronization that fails is a refusal, not a warning. The caller's next
        step is the side effect this record exists to explain, so continuing would
        issue an operation whose intent record may not outlive the machine -- exactly
        the case that makes a mutated node look untouched after a restart.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.json.new')
        if temporary.is_symlink():
            raise Refusal('Refusing a symlinked state file.')
        handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(handle, 'w') as output:
                json.dump(state.to_dict(), output, indent=2, sort_keys=True)
                output.write('\n')
                output.flush()
                os.fsync(output.fileno())
        except OSError as error:
            raise Refusal(
                'This round\'s record could not be written to storage in a form '
                f'that survives a restart ({error}), so nothing was done that would '
                'depend on it. Your node keeps its reservation exactly as it is.'
            ) from error
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)
        _sync_directory(self.directory)
        return state

    class _Lock:
        def __init__(self, store, pair=False):
            self.store = store
            self.pair = pair

        def __enter__(self):
            self.store._acquire()
            if self.pair:
                try:
                    self.store._acquire_pair()
                except BaseException:
                    self.store._release()
                    raise
            return self.store

        def __exit__(self, *exception):
            if self.pair:
                self.store._release_pair()
            self.store._release()
            return False

    def exclusive(self):
        """Per-assignment exclusion. A second connection is refused, not queued."""
        return self._Lock(self)

    def exclusive_pair(self):
        """Per-assignment plus per-node exclusion, for anything that mutates.

        Ordering is always assignment lock first, then pair lock, so two
        requests cannot deadlock against each other.
        """
        return self._Lock(self, pair=True)

    @staticmethod
    def _flock(path, message):
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(handle)
            if error.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise Refusal(message) from error
            raise
        return handle

    def _acquire(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock_handle = self._flock(
            self.lock_path,
            'Another request for this assignment is in progress. '
            'Wait for it to finish, then run status.')

    def _acquire_pair(self):
        if self.pair_lock_path is None:
            raise Refusal('This request needs an exercise node before it can '
                          'take the node lock.')
        self.directory.mkdir(parents=True, exist_ok=True)
        self._pair_handle = self._flock(
            self.pair_lock_path,
            'Another table is using this exercise node right now. '
            'Wait until they finish recovering, then start again.')

    def _release(self):
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
            os.close(self._lock_handle)
            self._lock_handle = None

    def _release_pair(self):
        if self._pair_handle is not None:
            fcntl.flock(self._pair_handle, fcntl.LOCK_UN)
            os.close(self._pair_handle)
            self._pair_handle = None


class Request:
    def __init__(self, verb=None, argument=None, error=None):
        self.verb = verb
        self.argument = argument
        self.error = error


def parse_participant_request(text):
    """The whole participant input grammar.

    Accepts only: start gpu, start efa, status, collect <digits>, recover, replace.
    Anything else, including any option, separator or newline, is an error.
    """
    if text is None:
        return Request(error=f'No command given. {USAGE}')
    if '\n' in text or '\r' in text or '\0' in text:
        return Request(error=f'One command per connection. {USAGE}')
    # Report a typed option as an option, before the character-class check, so
    # the participant is told why rather than only that the input was rejected.
    for word in text.split():
        if word.startswith('-'):
            return Request(error=f'Options are not accepted. {USAGE}')
    if not re.fullmatch(r'[A-Za-z0-9 ]*', text):
        return Request(error=f'Unsupported characters in the command. {USAGE}')
    words = text.split()
    if not words:
        return Request(error=f'No command given. {USAGE}')
    verb, arguments = words[0], words[1:]
    if verb in ('start', 'active'):
        if len(arguments) != 1:
            return Request(error=f'Use start gpu or start efa. {USAGE}')
        if arguments[0] not in ('gpu', 'efa'):
            return Request(error='The exercise provides start gpu and start efa only.')
        return Request(verb=verb, argument=arguments[0])
    if verb in ('status', 'recover', 'replace'):
        if arguments:
            return Request(error=f'{verb} takes no argument. {USAGE}')
        return Request(verb=verb)
    if verb == 'collect':
        if len(arguments) != 1:
            return Request(error=f'Use collect <check>. {USAGE}')
        if arguments[0] != 'kernel' and not re.fullmatch(r'[0-9]{1,2}', arguments[0]):
            return Request(error='A check is one of the identifiers listed by status.')
        return Request(verb='collect', argument=arguments[0])
    return Request(error=f'Unknown command {verb!r}. {USAGE}')


def load_config(path=DEFAULT_CONFIG, require_root_owned=True):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise Refusal('The session configuration must be a regular file.')
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise Refusal('The session configuration must be mode 0600.')
    if require_root_owned and info.st_uid != 0:
        raise Refusal('The session configuration must be owned by root.')
    data = json.loads(path.read_text())
    if not isinstance(data.get('assignments'), dict) or not data['assignments']:
        raise Refusal('The session configuration has no assignments.')
    if not data.get('state_dir'):
        raise Refusal('The session configuration has no state directory.')
    return data


class Result:
    def __init__(self, state, output=None, saved_path=None, node_state=None,
                 node_reason=None, already_started=False, already_recovered=False,
                 expired=False, next_step=''):
        self.state = state
        self.output = output
        self.saved_path = saved_path
        self.node_state = node_state
        self.node_reason = node_reason
        self.already_started = already_started
        self.already_recovered = already_recovered
        self.expired = expired
        self.next_step = next_step


class DeviceSession:
    def __init__(self, config, assignment_id, peer_uid, executor=None, clock=time.time):
        assignments = config['assignments']
        if assignment_id not in assignments:
            # Do not disclose which assignments exist.
            raise Refusal('This key is not bound to a usable assignment.')
        assignment = assignments[assignment_id]
        # Which table this connection belongs to is decided by the SSH key that
        # authenticated it: each key carries its own forced command naming one
        # assignment, and a participant cannot read another table's private key.
        # This check confirms the caller is the expected control account, so a
        # different local account cannot drive the helper even if it can run it.
        expected = assignment.get('caller_uid', assignment.get('participant_uid'))
        if expected is None or int(expected) != int(peer_uid):
            raise Refusal('This key is not bound to the assignment it requested. '
                          'The pre-session key/assignment binding is invalid; no node action is authorized.')
        self.config = config
        self.assignment_id = assignment_id
        self.assignment = dict(assignment)
        self.clock = clock
        self.store = StateStore(config['state_dir'], assignment_id,
                                target_node=assignment.get('target_node'))
        self.slurm_bin = assignment['slurm_bin']
        self.executor = executor or Executor(
            slurm_bin=self.slurm_bin,
            target_host=assignment.get('target_host', assignment['target_node']),
            identity_file=assignment.get('maintenance_identity_file'))

    # Replacement records are per logical node, not per alias or IP. The existing
    # pair lock serializes writers. Committed bindings and in-flight predecessor
    # routes are loaded AFTER locking, including by previously constructed handlers.
    def _replacement_path(self):
        node = self.assignment['target_node']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', node):
            raise Refusal('Unusable assigned node name.')
        return self.store.directory / f'replacement-{node}.json'

    def _replacement_record(self):
        path = self._replacement_path()
        if not path.exists() and not path.is_symlink():
            return {}
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise Refusal('Unusable replacement record.')
        try:
            record = json.loads(path.read_text())
            if (record['node'] != self.assignment['target_node']
                    or record['phase'] not in ('prepared', 'dispatching', 'waiting',
                                               'bootstrap', 'qualifying', 'complete')
                    or not isinstance(record['retired'], list)):
                raise ValueError('invalid binding')
            if (not self._valid_instance(record['initial_id'])
                    or not self._valid_instance(record['old_id'])
                    or record['old_id'] != record['evidence']['target_instance_id']
                    or record['old_id'] not in record['retired']
                    or not all(self._valid_instance(i) for i in record['retired'])
                    or record['owner'] not in self.config['assignments']):
                raise ValueError('invalid replacement provenance')
            if record['phase'] in ('bootstrap', 'qualifying', 'complete'):
                candidate = record['candidate']
                if (not self._valid_instance(candidate['instance_id'])
                        or candidate['instance_id'] in record['retired']):
                    raise ValueError('invalid candidate')
                ipaddress.ip_address(candidate['host'])
            if record['phase'] == 'complete' and record['binding'] != record['candidate']:
                raise ValueError('invalid committed binding')
            if record['phase'] != 'complete' and 'binding' in record:
                raise ValueError('uncommitted successor binding')
            if 'predecessor' in record:
                predecessor = record['predecessor']
                if (predecessor['instance_id'] != record['old_id']
                        or predecessor['instance_id'] != record['old_snapshot']['InstanceId']
                        or predecessor['host'] != record['old_snapshot']['PrivateIpAddress']):
                    raise ValueError('invalid predecessor provenance')
                ipaddress.ip_address(predecessor['host'])
            return record
        except (ValueError, KeyError, TypeError) as error:
            raise Refusal('Unusable replacement record; nothing changed.') from error

    def _write_replacement(self, record):
        path = self._replacement_path()
        self.store.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.new')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as output:
            json.dump(record, output, indent=2, sort_keys=True)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)

    def _refresh_binding(self, replacing=False):
        record = self._replacement_record()
        if not record:
            return
        configured = self.config['assignments'][self.assignment_id]['target_instance_id']
        if configured != record['initial_id']:
            raise Refusal('This assignment does not belong to the replacement binding.')
        if record.get('admission_release_pending') and (
                not replacing or record['owner'] != self.assignment_id):
            raise Refusal('Replacement admission release is pending; owner must repeat replace.')
        if record['phase'] == 'complete':
            binding = record['binding']
            if binding['instance_id'] in record['retired']:
                raise Refusal('A retired identity cannot be reused.')
        else:
            if not replacing:
                raise Refusal('The assigned node is being replaced; run replace to continue.')
            if record['owner'] != self.assignment_id:
                raise Refusal('Another table owns this replacement.')
            # This is a route to the already-authorized failed predecessor, NOT
            # a committed successor. old_id is reserved in retired at preparation
            # even before dispatch, so it must never be loaded as a binding.
            # Never derive this route from the newly discovered scheduler node.
            binding = record.get('predecessor')
            if not binding:
                raise Refusal('Replacement predecessor provenance is missing; nothing changed.')
        self.assignment['target_instance_id'] = binding['instance_id']
        self.assignment['target_host'] = binding['host']
        self.executor.target_host = binding['host']
        # Initial provisioning uses the configured SSH trust. Every successor
        # uses the per-ID authenticated SSM host file retained across operations.
        self.executor.known_hosts = (str(self.store.directory / f'host-{binding["instance_id"]}')
                                     if binding['instance_id'] != record['initial_id'] else '')

    def _cloud(self, service, action, payload):
        settings = self.assignment.get('replacement')
        if not settings or settings.get('enabled') is not True:
            raise Refusal('Replacement provisioning is not enabled for this assignment.')
        result = self.executor.run_cloud(settings['region'], service, action, payload)
        if result.returncode:
            raise Refusal(f'Replacement {service}:{action} did not confirm completion; '
                          'evidence is kept. Run replace again.')
        # RebootInstances has no CLI output on success. Other callers need a
        # JSON response; never turn their missing/malformed output into success.
        if (service, action) == ('ec2', 'reboot-instances') and not result.stdout.strip():
            return {}
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise Refusal('Replacement received an unreadable API response.') from error

    @staticmethod
    def _valid_instance(instance):
        return isinstance(instance, str) and re.fullmatch(r'i-(?:[0-9a-f]{8}|[0-9a-f]{17})', instance)

    def _replacement_node(self):
        line = self._node_line()
        fields = {}
        for key in ('NodeName', 'InstanceId', 'NodeAddr', 'State'):
            values = re.findall(rf'\b{key}=(\S+)', line)
            if len(values) != 1:
                raise Refusal(f'The scheduler did not return one {key}; replacement waits.')
            fields[key] = values[0]
        if (fields['NodeName'] != self.assignment['target_node']
                or not self._valid_instance(fields['InstanceId'])):
            raise Refusal('Scheduler identity does not match the assigned target.')
        try:
            ipaddress.ip_address(fields['NodeAddr'])
        except ValueError as error:
            raise Refusal('Scheduler address is not a literal IP.') from error
        fields['Reason'] = self._node_field(line, 'Reason')
        fields['raw_observation'] = line
        return fields

    def _owns_recovery_reason(self, line, state=None):
        """Exact ownership, with one Slurm-produced reboot transition.

        Do not normalize this suffix in node_reason: fault injection, admission,
        upgrades and a successor have no authority to inherit an old reboot.
        The suffix alone (including its annotation) is never ownership evidence.
        A durable request for this round and the fresh scheduler identity and
        isolation are required; reboot completion/queue/boot checks remain with
        the dependent action's existing guards.
        """
        reason = self._node_field(line, 'Reason')
        ours = self.assignment['drain_reason']
        if reason == ours:
            return True
        if reason != ours + ' : reboot issued' or state is None:
            return False
        if (state.target_node != self.assignment['target_node']
                or state.target_instance_id != self.assignment['target_instance_id']
                or state.reboot_requests < 1
                or not isinstance(state.reboot_requested_at, (int, float))
                or not 0 < state.reboot_requested_at <= self.clock()
                or not isinstance(state.operation, str)
                or not self.OPERATION_TOKEN.fullmatch(state.operation)
                or state.operation.split('/')[:2] != [self.assignment_id, str(state.round)]
                or 'drain' not in state.prepared):
            return False
        for key, expected in (('NodeName', state.target_node),
                              ('InstanceId', state.target_instance_id)):
            if re.findall(rf'\b{key}=(\S+)', line) != [expected]:
                return False
        flags = re.findall(r'\bState=(\S+)', line)
        return (len(flags) == 1
                and bool(set(flags[0].split('+')).intersection({'DRAIN', 'DOWN'})))

    def _replacement_instance(self, instance, host=None, retired=False):
        if not self._valid_instance(instance):
            raise Refusal('Invalid replacement instance identity.')
        cfg = self.assignment['replacement']
        if instance in cfg['protected_instance_ids']:
            raise Refusal('The coordinator or peer cannot be replaced.')
        response = self._cloud('ec2', 'describe-instances', {'InstanceIds': [instance]})
        rows = [row for reservation in response.get('Reservations', [])
                for row in reservation.get('Instances', [])]
        if len(rows) != 1 or rows[0].get('InstanceId') != instance:
            raise Refusal('EC2 did not confirm the exact replacement identity.')
        row = rows[0]
        if retired and row.get('State', {}).get('Name') in ('shutting-down', 'terminated'):
            return row  # Placement may disappear; pre-dispatch snapshot is durable.
        tags = {tag['Key']: tag['Value'] for tag in row.get('Tags', [])}
        if (tags.get('aws:pcs:compute-node-group-id') != cfg['group_id']
                or row.get('CapacityReservationId') != cfg['capacity_reservation_id']
                or row.get('InstanceType') != cfg['instance_type']
                or row.get('SubnetId') not in cfg['subnet_ids']
                or row.get('ImageId') != cfg['image_id']
                or (host is not None and row.get('PrivateIpAddress') != host)):
            raise Refusal('EC2 membership, capacity, image or address does not match provisioning.')
        return row

    def _replacement_capacity(self):
        cfg = self.assignment['replacement']
        cluster = self._cloud('pcs', 'get-cluster', {
            'clusterIdentifier': cfg['cluster_id']}).get('cluster', {})
        prologs = [item.get('parameterValue') for item in
                   cluster.get('slurmConfiguration', {}).get('slurmCustomSettings', [])
                   if item.get('parameterName') == 'Prolog']
        expected = ('/usr/local/sbin/aim344-replacement-prolog'
                    if cfg.get('provisioning_mode') == 'ssm-install'
                    else '/opt/aim344/.prolog/dispatch.sh')
        if cluster.get('status') != 'ACTIVE' or prologs != [expected]:
            raise Refusal('The pre-session replacement admission Prolog is not configured.')
        group = self._cloud('pcs', 'get-compute-node-group', {
            'clusterIdentifier': cfg['cluster_id'],
            'computeNodeGroupIdentifier': cfg['group_id']}).get('computeNodeGroup', {})
        if (group.get('status') != 'ACTIVE' or group.get('id') != cfg['group_id']
                or group.get('scalingConfiguration') != cfg['scaling_configuration']
                or group.get('customLaunchTemplate') != cfg['launch_template']):
            raise Refusal('PCS provisioning changed; replacement does not update group capacity.')

    def _active_cleanup_authority(self, state):
        # Retirement is a separate participant remedy, not a reward for a
        # successful SSH/reset/reboot. An unreachable predecessor cannot supply
        # those receipts. Scheduler + exact cloud membership are checked by
        # replace immediately before dispatch; this only recognizes the round.
        return bool(state.active_fault
                    and state.active_fault.get('mutation') == state.kind + '-flr'
                    and state.kind + '-flr-attempted' in state.prepared
                    and state.job_id
                    and state.target_instance_id == self.assignment['target_instance_id']
                    and state.target_node == self.assignment['target_node']
                    and state.phase in ('replacement-required', 'recovery-failed'))

    def _replacement_idle(self, allow_cleanup=False):
        # Unlike the permissive legacy queue parser, destructive retirement must
        # reject a malformed row as well as a foreign or unfinished allocation.
        result = self.executor.run_slurm([
            str(Path(self.slurm_bin) / 'squeue'), '-h', '-w',
            self.assignment['target_node'], '-o', '%i|%u|%j|%T'])
        if result.returncode:
            raise Refusal('Replacement requires a readable queue.')
        if result.stdout.strip():
            state = self.store.read()
            # Active GPU cleanup may remain CG even after walltime and reboot.
            # Only this round's exact allocation may be retired with its target;
            # no scancel/pkill, and never a new job by the same participant.
            own = self._own_job(self._jobs_on_target())
            if (allow_cleanup and self._active_cleanup_authority(state)
                    and own and own['id'] == state.job_id and own['state'] == 'COMPLETING'):
                return True
            raise Refusal('Replacement requires an empty readable queue. End your allocation first.')
        return False

    def _replacement_reservation(self, record, release=False):
        """Keep a whole logical node excluded across no-roll initialization.

        STATIC_ALLOC cannot substitute the peer when the old guest disappears.
        Only our durable intent may adopt a lost create/delete acknowledgement.
        This is a Slurm reservation, never an EC2 capacity-reservation mutation.
        """
        if self.assignment['replacement'].get('provisioning_mode') != 'ssm-install':
            return
        if release and record.get('admission_release_pending') is False:
            return
        name = 'aim344-replacement-' + record['old_id']
        if record.get('admission_reservation_intent') and (
                not record.get('admission_topology')
                or (record.get('admission_reservation_confirmed')
                    and not record.get('admission_lifetime'))):
            raise Refusal('Admission reservation provenance is incomplete; nothing released.')

        def field(line, key):
            values = re.findall(rf'(?:^|\s){key}=(\S+)', line)
            if len(values) != 1:
                raise Refusal('Incomplete or duplicate admission reservation field: ' + key)
            return values[0]

        def topology():
            line = self._node_line()
            if field(line, 'NodeName') != record['node']:
                raise Refusal('Admission node topology does not match assignment.')
            values = {}
            for key in ('Sockets', 'CoresPerSocket', 'ThreadsPerCore', 'CPUTot'):
                value = field(line, key)
                if not re.fullmatch(r'[1-9][0-9]*', value):
                    raise Refusal('Admission node topology is not complete.')
                values[key] = int(value)
            cores = values['Sockets'] * values['CoresPerSocket']
            if cores * values['ThreadsPerCore'] != values['CPUTot']:
                raise Refusal('Admission node CPU topology is inconsistent.')
            return {'cores': cores, 'cpus': values['CPUTot']}

        def observe():
            result = self._scontrol('show', 'reservation', '-o')
            if result.returncode:
                raise Refusal('Admission reservation census is unreadable; nothing released.')
            matches = []
            for line in result.stdout.splitlines():
                if not line.strip() or line.strip() == 'No reservations in the system':
                    continue
                names = re.findall(r'\bReservationName=(\S+)', line)
                if len(names) != 1:
                    raise Refusal('Malformed admission reservation census.')
                if names == [name]:
                    matches.append(line)
            if len(matches) > 1:
                raise Refusal('Ambiguous admission reservation.')
            if not matches:
                return False
            line = matches[0]
            for key, expected in (('Nodes', record['node']), ('NodeCnt', '1'),
                                  ('Users', 'root'), ('Accounts', '(null)'),
                                  ('Groups', '(null)'), ('State', 'ACTIVE'),
                                  ('Features', '(null)'), ('PartitionName', '(null)'),
                                  ('Licenses', '(null)'), ('BurstBuffer', '(null)'),
                                  ('MaxStartDelay', '(null)'), ('Duration', '365-00:00:00')):
                if field(line, key) != expected:
                    raise Refusal('Admission reservation scope or lifetime changed.')
            # 25.05.9 accepts STATIC_ALLOC, but prints STATIC. UNLIMITED input
            # becomes a finite start + YEAR_SECONDS end, not an eternal barrier.
            flags = field(line, 'Flags').split(',')
            if sorted(flags) != ['SPEC_NODES', 'STATIC']:
                raise Refusal('Admission reservation does not pin only the assigned node.')
            try:
                times = [datetime.datetime.strptime(field(line, key), '%Y-%m-%dT%H:%M:%S')
                         .replace(tzinfo=datetime.timezone.utc).timestamp()
                         for key in ('StartTime', 'EndTime')]
            except ValueError as error:
                raise Refusal('Admission reservation timestamps are unreadable.') from error
            start, end = times
            now = self.clock()
            if end - start != 365 * 24 * 3600 or not start <= now < end - 3600:
                raise Refusal('Admission reservation has expired or insufficient remaining lifetime.')
            if record.get('admission_reservation_intent'):
                intent_at = record.get('admission_intent_at')
                if (not isinstance(intent_at, (int, float))
                        or not intent_at - 5 <= start <= intent_at + 300):
                    raise Refusal('Admission reservation start does not match durable intent.')
                lifetime = record.get('admission_lifetime')
                if lifetime is not None and lifetime != times:
                    raise Refusal('Admission reservation lifetime changed since confirmation.')
            shape = topology()
            # Partial-core reservations serialize NodeName/CoreIDs; whole-node
            # reservations have no such section (25.05.9 reservation.c _pack_resv).
            if (re.search(r'(?:^|\s)(?:NodeName|CoreIDs)=', line)
                    or field(line, 'CoreCnt') != str(shape['cores'])
                    or field(line, 'TRES') != 'cpu=' + str(shape['cpus'])
                    or (record.get('admission_topology') is not None
                        and record['admission_topology'] != shape)):
                raise Refusal('Admission reservation does not cover the complete node resources.')
            if record.get('admission_reservation_intent'):
                record['admission_lifetime'] = times
            return True

        present = observe()
        if release:
            if not record.get('admission_reservation_intent'):
                raise Refusal('Missing admission reservation ownership; nothing released.')
            if (record['phase'] != 'complete' or record.get('binding') != record.get('candidate')
                    or not record.get('admission_reservation_confirmed')
                    or not record.get('admission_lifetime') or not record.get('admission_topology')):
                raise Refusal('Qualified admission binding is incomplete; nothing released.')
            # Every retry re-establishes BOTH file and directory durability. A
            # visible complete record may be a rename whose directory sync failed.
            # This must precede deletion, including reconciliation of a lost ack.
            self._write_replacement(record)
            self._replacement_capacity()
            self._replacement_idle()
            node = self._replacement_node()
            candidate = record['candidate']
            if (node['InstanceId'] != candidate['instance_id'] or node['NodeAddr'] != candidate['host']
                    or self._current_boot_id() != record['qualified_boot']
                    or any(flag in node['State'] for flag in
                           ('DRAIN', 'DOWN', 'REBOOT_', 'POWERING', 'NOT_RESPONDING'))):
                raise Refusal('Qualified identity changed before admission release; barrier retained.')
            if present:
                self._scontrol('delete', 'ReservationName=' + name)
                if observe():
                    raise Refusal('Qualified replacement reservation release is pending; repeat replace.')
            record['admission_release_pending'] = False
            self._write_replacement(record)
            return
        if not record.get('admission_reservation_intent'):
            if present:
                raise Refusal('Preexisting reservation is not owned by this replacement.')
            record['admission_reservation_intent'] = True
            record['admission_intent_at'] = self.clock()
            record['admission_topology'] = topology()
            self._write_replacement(record)
        if not present:
            if record.get('admission_reservation_confirmed') or record['phase'] not in ('prepared', 'dispatching'):
                raise Refusal('Admission barrier disappeared; replacement remains incomplete.')
            # No existing barrier: a new create attempt may have a later start.
            # Persist the new window before dispatch; never change it when adopting.
            record['admission_intent_at'] = self.clock()
            record.pop('admission_lifetime', None)
            self._write_replacement(record)
            self._scontrol('create', 'reservation', 'ReservationName=' + name,
                           'StartTime=now', 'Duration=UNLIMITED', 'Users=root',
                           'Nodes=' + record['node'], 'Flags=STATIC_ALLOC')
            if not observe():
                raise Refusal('Admission reservation was not confirmed; no retirement.')
        record['admission_reservation_confirmed'] = True
        self._write_replacement(record)

    def _replacement_bootstrap(self, record):
        cfg = self.assignment['replacement']
        instance = record['candidate']['instance_id']
        if not record.get('command_id'):
            # The document has no shell/target parameters. Duplicate unknown-send
            # bootstrap calls are serialized and idempotent on the same new node.
            response = self._cloud('ssm', 'send-command', {
                'DocumentName': cfg['bootstrap_document'],
                'DocumentVersion': cfg['bootstrap_document_version'],
                'DocumentHash': cfg['bootstrap_document_sha256'],
                'DocumentHashType': 'Sha256', 'InstanceIds': [instance],
                'Parameters': {}, 'MaxConcurrency': '1', 'MaxErrors': '0',
                'TimeoutSeconds': 600, 'Comment': 'AIM344 assigned replacement bootstrap'})
            command = response.get('Command', {})
            if command.get('InstanceIds') != [instance] or not self.BOOT_ID.fullmatch(command.get('CommandId', '')):
                raise Refusal('Bootstrap acknowledgement did not identify the requested node.')
            record['command_id'] = command['CommandId']
            self._write_replacement(record)
        result = self._cloud('ssm', 'get-command-invocation', {
            'CommandId': record['command_id'], 'InstanceId': instance})
        if (result.get('CommandId') != record['command_id'] or result.get('InstanceId') != instance):
            raise Refusal('Bootstrap result belongs to a different request.')
        if result.get('Status') != 'Success' or result.get('ResponseCode') != 0:
            record['bootstrap_result'] = result
            if result.get('Status') in ('Failed', 'Cancelled', 'TimedOut'):
                record.pop('command_id', None)
            self._write_replacement(record)
            raise Refusal('Replacement bootstrap is incomplete; run replace again. No new retirement.')
        try:
            report = json.loads(result['StandardOutputContent'])
        except (ValueError, KeyError) as error:
            raise Refusal('Bootstrap returned no authenticated report.') from error
        if (report.get('instance_id') != instance or report.get('node') != record['node']
                or report.get('release') != cfg['release_sha256']
                or report.get('ready') is not True):
            raise Refusal('Bootstrap identity or pinned materials did not qualify.')
        key = report.get('host_key', '')
        if not re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,2}', key):
            raise Refusal('Bootstrap did not return an authenticated SSH host key.')
        # Authenticated SSM response, never ssh-keyscan or accept-new. Isolated
        # known_hosts avoids overwriting unrelated administrator trust entries.
        known = self.store.directory / f'host-{instance}'
        fd = os.open(known, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as output:
            output.write(f'{record["candidate"]["host"]} {key}\n')
            output.flush()
            os.fsync(output.fileno())
        _sync_directory(known.parent)
        self.executor.known_hosts = str(known)
        record['bootstrap_report'] = report
        record['phase'] = 'qualifying'
        self._write_replacement(record)

    def replace(self):
        """Retire one failed identity; repeat requests never retire its successor."""
        with self.store.exclusive_pair():
            self._refresh_binding(replacing=True)
            state = self.store.read()
            record = self._replacement_record()
            if record and record['owner'] != self.assignment_id and record['phase'] != 'complete':
                raise Refusal('Another table owns this replacement.')
            if record and record['phase'] == 'complete' and record['round'] == state.round and record['owner'] == self.assignment_id:
                # Crash after binding commit but before assignment-state write.
                self._replacement_reservation(record, release=True)
                self._write_replacement(record)
                state.phase = 'runtime-ready'
                state.target_instance_id = record['binding']['instance_id']
                state.boot_id = record['qualified_boot']
                state.recovered_by = 'participant-replacement'
                state.recovered_at = record['completed_at']
                state.failure = None
                self.store.write(state)
                return Result(state, already_recovered=True, next_step=self._next_step(state))
            if self._other_assignment_holding_target():
                raise Refusal('Another table is using this exercise node.')
            cfg = self.assignment.get('replacement', {})
            if not cfg.get('enabled'):
                raise Refusal('Replacement provisioning is not enabled for this assignment.')
            if self.assignment['target_node'] == self.assignment['coordinator_node']:
                raise Refusal('The coordinator cannot be a replacement target.')
            self._replacement_capacity()
            self._replacement_idle(allow_cleanup=True)
            node = self._replacement_node()
            if not record or record['phase'] == 'complete':
                if state.phase not in ('replacement-required', 'recovery-failed'):
                    raise Refusal('Run supported recover first; replacement requires a failed recovery.')
                if (state.target_instance_id != self.assignment['target_instance_id']
                        or state.target_node != self.assignment['target_node']):
                    raise Refusal('The failed round does not match this assignment.')
                old = state.target_instance_id
                # Automatic PCS replacement may already have happened. The old
                # EC2 record still has to establish our configured membership.
                old_snapshot = self._replacement_instance(old)
                predecessor = {'instance_id': old, 'host': self.assignment['target_host']}
                if predecessor['host'] != old_snapshot['PrivateIpAddress']:
                    raise Refusal('Assigned predecessor address does not match EC2; nothing changed.')
                try:
                    originals = self.executor.run_maintenance(['replacement-evidence'], timeout=30)
                except (OSError, subprocess.TimeoutExpired, Refusal) as error:
                    originals = Completed(124, '', f'Target evidence unavailable: {error}')
                old_record = record
                if old_record:
                    archive = self.store.directory / f'replacement-history-{old_record["old_id"]}.json'
                    if archive.is_symlink():
                        raise Refusal('Refusing symlinked replacement archive.')
                    if archive.exists() and json.loads(archive.read_text()) != old_record:
                        raise Refusal('Replacement archive differs; preserving both operation histories.')
                    temporary = archive.with_suffix('.new')
                    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
                    with os.fdopen(fd, 'w') as output:
                        json.dump(old_record, output)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, archive)
                    _sync_directory(archive.parent)
                record = {'node': state.target_node, 'initial_id': old_record.get('initial_id', old),
                          'owner': self.assignment_id, 'round': state.round,
                          'old_id': old, 'phase': 'prepared', 'retired': old_record.get('retired', []) + [old],
                          'evidence': state.to_dict(), 'prepared_at': self.clock(),
                          'old_snapshot': old_snapshot,
                          'predecessor': predecessor,
                          'originals_capture': {'returncode': originals.returncode,
                                                'stdout': originals.stdout, 'stderr': originals.stderr},
                          'target_only_originals': 'Not reconstructed; unavailable after target loss.'}
                # Rewrite even a previously visible record before dispatch: a
                # preceding directory-fsync failure is not a durability receipt.
                self._write_replacement(record)
            old = record['old_id']
            self._replacement_reservation(record)
            if record['phase'] in ('prepared', 'dispatching', 'waiting'):
                if record.get('old_terminated'):
                    old_status = 'terminated'
                else:
                    old_row = self._replacement_instance(old, retired=True)
                    old_status = old_row['State']['Name']
                if old_status not in ('shutting-down', 'terminated'):
                    record['phase'] = 'dispatching'
                    self._write_replacement(record)
                    # Evidence capture, cloud reads and durable intent can take
                    # time. Authorize the dependent dispatch with fresh reads,
                    # not the observations from before those operations. This
                    # is not an atomic transaction with the scheduler or EC2.
                    self._replacement_reservation(record)
                    cleanup = self._replacement_idle(allow_cleanup=True)
                    node = self._replacement_node()
                    if (node['InstanceId'] != old
                            or node['NodeAddr'] != self.assignment['target_host']
                            or node['NodeAddr'] != record['old_snapshot']['PrivateIpAddress']):
                        raise Refusal('Scheduler identity/address changed before retirement.')
                    flags = set(node['State'].split('+'))
                    forbidden = set() if cleanup else {'ALLOCATED', 'MIXED', 'COMPLETING'}
                    if not flags.intersection({'DRAIN', 'DOWN'}) or flags.intersection(forbidden):
                        raise Refusal('The failed node is not idle and isolated; no retirement was requested.')
                    # Preserve the fresh raw scheduler audit, including Slurm's suffix.
                    record['retirement_observation'] = node['raw_observation']
                    self._write_replacement(record)
                    evidence = State.from_dict(record['evidence'])
                    if (record['round'] != state.round
                            or evidence.operation != state.operation
                            or not self._owns_recovery_reason(node['raw_observation'], evidence)):
                        raise Refusal('The failed node has a foreign drain reason: '
                                      + repr(node_reason(node['raw_observation'])[1]))
                    self._replacement_instance(old, host=self.assignment['target_host'])
                    self._cloud('ec2', 'terminate-instances', {'InstanceIds': [old], 'SkipOsShutdown': True})
                    # Read back exact target before claiming retirement, including
                    # on a retry whose earlier acknowledgement was lost.
                    old_row = self._replacement_instance(old, retired=True)
                    old_status = old_row['State']['Name']
                record['phase'] = 'waiting'
                record['old_terminated'] = old_status == 'terminated'
                self._write_replacement(record)
                if old_status != 'terminated':
                    raise Refusal('Old instance retirement is pending. Run replace again; only the old ID is addressed.')
                node = self._replacement_node()
                new = node['InstanceId']
                if new in record['retired']:
                    raise Refusal('PCS has not registered a new assigned identity yet. Run replace again.')
                self._replacement_instance(new, node['NodeAddr'])
                record['candidate'] = {'instance_id': new, 'host': node['NodeAddr']}
                record['phase'] = 'bootstrap'
                self._write_replacement(record)
            candidate = record['candidate']
            if (node['InstanceId'] != candidate['instance_id'] or node['NodeAddr'] != candidate['host']):
                raise Refusal('Replacement identity changed during initialization; no further retirement authorized.')
            current_instance = self._replacement_instance(candidate['instance_id'], candidate['host'])
            if current_instance.get('State', {}).get('Name') != 'running':
                raise Refusal('Replacement instance is not running yet; run replace again.')
            if any(flag in node['State'] for flag in (*self.REBOOT_PENDING_FLAGS, 'POWERING_UP', 'NOT_RESPONDING')):
                raise Refusal('Replacement is not scheduler-ready yet. Run replace again.')
            reason = self._node_field(self._node_line(), 'Reason')
            admission_reason = 'AIM344-admission-' + candidate['instance_id']
            if ('DRAIN' in node['State'] or 'DOWN' in node['State']) and reason not in (
                    '', self.assignment['drain_reason'], admission_reason):
                raise Refusal('Replacement has a foreign isolation reason; not overwriting it.')
            drained = self._scontrol('update', f'NodeName={record["node"]}', 'State=DRAIN',
                                     f'Reason={self.assignment["drain_reason"]}')
            if drained.returncode:
                raise Refusal('Replacement could not be isolated for qualification.')
            self._replacement_bootstrap(record)
            self.executor.target_host = candidate['host']
            if self._instance_id() != candidate['instance_id']:
                raise Refusal('Authenticated maintenance identity does not match replacement.')
            boot = self._boot_id()
            self._maintenance(['restore-runtime', '--participant', self.assignment['participant_user']], timeout=900)
            clean = State()  # Never inherit warning allowances from retired hardware.
            for check in ('0', '2', '3', '6'):
                outcome = self.executor.run_maintenance(['collect', check], timeout=900)
                text = (outcome.stdout or '') + (outcome.stderr or '')
                record.setdefault('check_attempts', []).append(
                    {'check': check, 'output': text, 'returncode': outcome.returncode, 'boot': boot})
                record.setdefault('checks', {})[check] = {'output': text, 'returncode': outcome.returncode}
                self._write_replacement(record)
                producers = re.findall(r'\[(?:PASS|WARN|FAIL|SKIP)\]\s+([^\s:]+):', text)
                if (outcome.returncode or self.check_verdict(text) != 'PASS'
                        or producers != [self.CHECK_NAMES[check]]
                        or not self._observation_is_complete(clean, check, text)[0]):
                    raise Refusal(f'Replacement check {check} did not qualify; retry replace without retiring it.')
            self._replacement_capacity()
            self._replacement_reservation(record)
            self._replacement_idle()
            node = self._replacement_node()
            if node['InstanceId'] != candidate['instance_id'] or node['NodeAddr'] != candidate['host']:
                raise Refusal('Replacement binding changed during qualification.')
            self._require_current_recovery_observations(boot, 'replacement admission')
            self._maintenance(['admit-replacement'])
            self._replacement_idle()
            final_node = self._replacement_node()
            if (final_node['InstanceId'] != candidate['instance_id']
                    or final_node['NodeAddr'] != candidate['host']):
                raise Refusal('Replacement identity changed before RESUME.')
            self._require_current_recovery_observations(boot, 'RESUME')
            resumed = self._scontrol('update', f'NodeName={record["node"]}', 'State=RESUME')
            if resumed.returncode:
                raise Refusal('Replacement qualified but scheduler RESUME failed; retry replace.')
            # Slurm RESUME first marks the node NOT_RESPONDING while requesting
            # registration. An immediate read therefore refuses a healthy node,
            # and a retry's new DRAIN/RESUME repeats that transition forever.
            # Observe this one request; never accept the transient as readiness
            # or send another RESUME inside the observation window.
            for attempt in range(30):
                observed = self._replacement_node()
                if (observed['InstanceId'] != candidate['instance_id']
                        or observed['NodeAddr'] != candidate['host']
                        or any(flag in observed['State'] for flag in
                               ('DRAIN', 'DOWN', 'REBOOT_', 'POWERING'))
                        or observed['Reason'] not in ('', self.assignment['drain_reason'])):
                    raise Refusal('Scheduler binding or isolation changed after RESUME; run replace again.')
                if 'NOT_RESPONDING' not in observed['State']:
                    break
                if attempt == 29:
                    raise Refusal('Scheduler has not confirmed replacement in service; run replace again.')
                time.sleep(1)
            # The wait does not extend the lifetime of the earlier observations.
            # Keep the scheduling barrier until identity, boot, queue and its
            # exact reservation have been checked again before binding commit.
            if self._current_boot_id() != boot:
                raise Refusal('Replacement boot changed after RESUME; run replace again.')
            self._replacement_idle()
            self._replacement_reservation(record)
            # The binding is the single authoritative commit point. Other aliases
            # stay excluded until it is durable. Old round evidence remains here.
            record['binding'] = candidate
            record['qualified_boot'] = boot
            record['phase'] = 'complete'
            record['completed_at'] = self.clock()
            if self.assignment['replacement'].get('provisioning_mode') == 'ssm-install':
                record['admission_release_pending'] = True
            self._write_replacement(record)
            self._replacement_reservation(record, release=True)
            self._refresh_binding()
            state.phase = 'runtime-ready'
            state.target_instance_id = candidate['instance_id']
            state.boot_id = boot
            state.recovered_by = 'participant-replacement'
            state.recovered_at = self.clock()
            state.failure = None
            state.notes = 'Fresh replacement initialized; old target evidence retained. Take a fresh allocation.'
            self.store.write(state)
            return Result(state, next_step=self._next_step(state))

    # ---------------- scheduler reads, on the coordinator ----------------
    def _scontrol(self, *arguments):
        return self.executor.run_slurm([str(Path(self.slurm_bin) / 'scontrol'), *arguments])

    def _node_line(self):
        result = self._scontrol('show', 'node', self.assignment['target_node'])
        if result.returncode != 0:
            raise Refusal(f'Could not read the exercise node state: {result.stderr.strip()}')
        return result.stdout

    @staticmethod
    def _node_field(line, name):
        if name == 'Reason':
            body, raw = node_reason(line)
            if body is None and raw:
                raise Refusal('Ambiguous scheduler Reason: ' + repr(raw))
            return body if body is not None else ''
        match = re.search(rf'\b{name}=(\S*)', line)
        return match.group(1) if match else ''

    def _jobs_on_target(self):
        result = self.executor.run_slurm([
            str(Path(self.slurm_bin) / 'squeue'), '-h',
            '-w', self.assignment['target_node'], '-o', '%i|%u|%j|%T'])
        if result.returncode != 0:
            raise Refusal(f'Could not read the exercise queue: {result.stderr.strip()}')
        jobs = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split('|')
            if len(parts) != 4 or not re.fullmatch(r'[1-9][0-9]*', parts[0]):
                raise Refusal('The exercise queue contains a malformed job row; nothing was changed.')
            jobs.append({'id': parts[0].strip(), 'user': parts[1].strip(),
                         'name': parts[2].strip(),
                         'state': parts[3].strip() if len(parts) > 3 else ''})
        return jobs

    def _own_job(self, jobs):
        """The one allowed job, or None. Any other job stops the request."""
        if len(jobs) > 1:
            raise Refusal('More than one job is using the exercise node; nothing was changed.')
        allowed = None
        for job in jobs:
            if job['user'] != self.assignment['participant_user']:
                raise Refusal('An unapproved job is using your exercise node. '
                              'It was left running; nothing was changed.')
            if not job['name'].startswith(self.assignment['job_name_prefix']):
                raise Refusal('A job outside this exercise is using your exercise '
                              'node. It was left running; nothing was changed.')
            allowed = job
        return allowed

    # ---------------- maintenance route ----------------
    def _maintenance(self, argv, timeout=None):
        result = self.executor.run_maintenance(argv, timeout=timeout)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip().splitlines()
            message = detail[-1] if detail else f'The maintenance route refused {argv[0]}.'
            # While the exercise node reboots, the maintenance route is simply
            # unreachable. Report that as a stage of recovery rather than handing
            # the participant a raw transport error.
            if result.returncode == 255 or 'Connection' in message:
                raise Refusal('Your exercise node is not reachable right now, which '
                              'is expected while it restarts. It stays reserved for '
                              'you; run status, then recover again in a few minutes.')
            raise Refusal(message)
        return result

    # What `boot-id` returns when it returns an observation: the contents of
    # /proc/sys/kernel/random/boot_id, which the kernel formats as a lowercase
    # hyphenated UUID (maintenance.main writes that file verbatim). Anything else --
    # empty, whitespace, a diagnostic line, a truncated read -- is not an observation of
    # the boot, and a comparison against it would be comparing something else. The
    # pattern is deliberately about SHAPE only: this helper never interprets the value,
    # it only asks whether two readings are the same reading.
    BOOT_ID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'
                         r'[0-9a-f]{4}-[0-9a-f]{12}')

    def _inspect(self):
        result = self._maintenance(['inspect'])
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get('action') == 'inspect':
                return record
        raise Refusal('The maintenance route did not return a device inventory.')

    def _boot_id(self):
        """The node's boot id, or a Refusal. Never an empty string.

        `start` records this before it reserves the node and before any device call, and
        every later completion decision is relative to it: `_await_boot` reports a
        reboot as complete when the CURRENT boot differs from this one. So an unusable
        answer here is not a value to store -- it is a comparison baseline that no
        observation established, and stored as `''` it makes any nonempty later reading
        look like a change, including the boot the node never left. Refusing here keeps
        that decision honest at the only point where nothing has been changed yet.

        rc 0 with empty stdout is a real shape, not a hypothetical one: `boot-id` is
        `maintenance.main` writing /proc/sys/kernel/random/boot_id, and a truncated
        transfer or an ssh channel that closes after the exit status leaves exactly
        that. The value is also format-checked, because a boot id that is not a UUID is
        not this file's contents and comparing it would compare something else.
        """
        answer = self._maintenance(['boot-id']).stdout.strip()
        if not self.BOOT_ID.fullmatch(answer):
            raise Refusal(
                'Your exercise node did not report a usable current boot '
                f'({answer[:60]!r} is not a boot id), and every later decision about '
                'whether it restarted is made against that value. Nothing was changed '
                'and your node was not reserved; run status, then try again in a few '
                'minutes. No fault was applied.')
        return answer

    def _instance_id(self):
        """The node's instance identity, without requiring any device to exist.

        `_inspect` goes through the device helper, which refuses when the
        provisioned GPU PCI function is absent, and a GPU round removes exactly
        that function. Using `_inspect` for an identity check therefore made GPU
        recovery impossible: measured on the target with the fault applied,
        recovery stopped at 'Provisioned gpu PCI function is absent:
        0000:ba:00.0' before it could request the reboot that would restore the
        device, leaving the node drained and the round stuck at investigating
        (participant-revision/runs/rework2-accept-gpu-c1/output.log and
        rework2-probe-gpu-inspect/output.log).

        Identity comes from the same DMI field the device helper itself reads
        before it looks at any device, so this is not a weaker check. It is the
        identity check separated from a device-inventory check that a fault is
        expected to break.
        """
        result = self._maintenance(['instance-id'])
        return result.stdout.strip()

    def _other_assignment_holding_target(self):
        """Another table must not be mid-round on the same exercise node.

        Observed on the pair: when a second assignment pointed at a node already
        faulted by the first, its start failed only incidentally, with the device
        helper's 'Provisioned gpu PCI function is absent: 0000:ba:00.0'. That
        leaks an internal device identity and does not tell the participant what
        happened. Ownership is now checked explicitly, before any mutation.
        """
        target = self.assignment['target_node']
        for name, other in self.config['assignments'].items():
            if name == self.assignment_id:
                continue
            if other.get('target_node') != target:
                continue
            try:
                state = StateStore(self.config['state_dir'], name,
                                   target_node=target).read()
            except Refusal:
                # An unreadable record for another table still means this node is
                # not provably free, so treat it as held.
                return name
            # A table that has finished recovering no longer holds the node: its
            # device is back and the node is in service. Only a round between
            # start and completed recovery is exclusive, and a round whose
            # recovery failed still holds it, because the node is still drained.
            if state.phase in HOLDING_PHASES:
                return name
        return None

    # ---------------- answers directory ----------------
    @staticmethod
    def write_strategy(euid, target_uid):
        """Who performs a participant-output filesystem operation.

        Root must never create, write or chown a file through a directory the
        participant controls: the participant can replace a checked component
        between the check and the syscall, and O_NOFOLLOW protects only the final
        component (open(2): "Symbolic links in earlier components of the pathname
        will still be followed"). So the privileged helper drops to the
        participant's own uid and gid and does the whole operation there, where a
        swapped component grants nothing the participant did not already have.
        """
        if target_uid is None:
            return 'refuse'
        target_uid = int(target_uid)
        if euid == target_uid:
            return 'direct'
        if euid == 0:
            return 'drop'
        return 'refuse'

    def _participant_ids(self):
        uid = self.assignment.get('participant_uid')
        gid = self.assignment.get('participant_gid', uid)
        return (None if uid is None else int(uid),
                None if gid is None else int(gid))

    def _answers_root(self):
        """Path only. Existence and type are established through descriptors."""
        return Path(self.assignment['answers_dir'])

    def collect_output_name(self, check, phase, attempt=1):
        suffix = '' if attempt == 1 else f'-attempt-{attempt}'
        return f'check-{check}-{phase}{suffix}.log'

    def collect_output_path(self, check, phase='fault', round=1, attempt=1):
        """Where a capture goes. One file per round and phase, never reused."""
        return (self._answers_root() / 'collected' / f'round-{round}'
                / self.collect_output_name(check, phase, attempt))

    def _save_output(self, check, text, phase='fault', round=1):
        """Write one capture as the participant, under their own uid and gid."""
        uid, gid = self._participant_ids()
        strategy = self.write_strategy(os.geteuid(), uid)
        if strategy == 'refuse':
            raise Refusal('This assignment has no participant account to own its '
                          'results; nothing was written. Pre-session account provisioning is incomplete.')
        relative = ('collected', f'round-{round}')
        if strategy == 'direct':
            return _write_under(self._answers_root(), relative,
                                lambda attempt: self.collect_output_name(
                                    check, phase, attempt), text, uid)
        return self._write_in_dropped_child(uid, gid, relative, check, phase,
                                            round, text)

    def _write_in_dropped_child(self, uid, gid, relative, check, phase, round, text):
        read_end, write_end = os.pipe()
        child = os.fork()
        if child == 0:                                    # pragma: no cover
            # The child never returns to the caller's code path.
            status = 1
            try:
                os.close(read_end)
                os.setgroups([])
                os.setgid(gid)
                os.setuid(uid)
                if os.geteuid() != uid or os.getuid() != uid:
                    os._exit(3)
                written = _write_under(
                    self._answers_root(), relative,
                    lambda attempt: self.collect_output_name(check, phase, attempt),
                    text, uid)
                os.write(write_end, str(written).encode())
                status = 0
            except BaseException as error:
                try:
                    os.write(write_end, f'!{error}'.encode()[:3000])
                except OSError:
                    pass
                status = 2
            finally:
                try:
                    os.close(write_end)
                except OSError:
                    pass
                os._exit(status)
        os.close(write_end)
        message = b''
        while True:
            block = os.read(read_end, 4096)
            if not block:
                break
            message += block
        os.close(read_end)
        _, wait_status = os.waitpid(child, 0)
        code = (os.WEXITSTATUS(wait_status) if os.WIFEXITED(wait_status)
                else -os.WTERMSIG(wait_status))
        if code != 0:
            detail = message.decode('utf-8', 'replace').lstrip('!').strip()
            raise Refusal('Your results file could not be written as your own '
                          'account, so nothing was written'
                          + (f': {detail}' if detail else '.'))
        return Path(message.decode())

    # A warning line as the pinned suite emits it. Two shapes reach the controller,
    # both read out of the deployed suite rather than assumed:
    #
    #   lib/common.sh:186-191  check_warn -> '[WARN] <check-name>: <details>'
    #   lib/common.sh:41-43    log_warn   -> '[WARN] <details>'
    #
    # Both are printed with no colour when stdout is not a terminal, confirmed on
    # the target: the check-6 capture produced through the maintenance route holds
    # zero ESC bytes (participant-revision/runs/rework7-read-warn-format/output.log).
    # checks/6-efa-loopback.sh emits the per-condition log_warn lines
    # ('EFA rx_drops detected (N)', 'EFA retransmission timeouts detected (N)') and
    # then one aggregate check_warn that names neither counter, which is why the
    # aggregate label alone cannot distinguish a drop from a retransmission.
    #
    # The check name is captured separately from the details, so a comparison can
    # tell WHICH check warned and so the two counter sentences below, which are
    # `log_warn` lines with no check name, are distinguishable from the aggregate.
    WARNING_LINE = re.compile(r'\[WARN\]\s+(?:(\d+-[a-z0-9-]+):\s*)?(.*\S)\s*$')
    # A '[WARN]' this parser could not read as a condition at all. Recorded rather
    # than skipped: a warning line nobody can compare must keep the node drained,
    # and silently dropping it was how an unreadable line became no line.
    WARNING_MARKER = re.compile(r'\[WARN\]')
    # The ONLY numbers in a warning that this helper interprets: the two counters
    # check 6 documents as cumulative, each matched as its own whole sentence.
    #
    # checks/6-efa-loopback.sh@a4ba07eb reads them out of `rdma -p statistic show`
    # and warns per condition:
    #
    #   :166-171  rx_drops        summed over links, then
    #             log_warn "EFA rx_drops detected (${rx_drops}) -- possible network issues"
    #   :167-175  retrans_timeout_events summed over links, then
    #             log_warn "EFA retransmission timeouts detected (${retrans_timeouts})"
    #
    # Both are `log_warn` AT THE PINNED COMMIT, so at that revision they arrive with
    # no check name in front of them. That is NOT relied on below, because it is not
    # stable across revisions of the suite: on riv-aim344 `main` the very same two
    # sentences are emitted by `check_warn "${CHECK_NAME}"`
    # (checks/6-efa-loopback.sh:166,169 at 39848da1), which prints
    # `[WARN] <name>: <details>` (lib/common.sh:186-189) instead of log_warn's bare
    # `[WARN] <details>` (lib/common.sh:41-43).
    #
    # An earlier version of this helper only read a counter when the check-name
    # prefix was ABSENT. Against the pinned suite that was right, and against the
    # upstream suite it silently disabled the whole S2 refusal: the prefixed
    # `[WARN] 6-efa-loopback: EFA retransmission timeouts detected (2)` parsed with
    # counter=None, so an unchanged counter qualified as an ordinary exact-match
    # condition and the node was auto-resumed -- the exact defect S2 exists to
    # prevent, reachable by upgrading the pin. The counter sentence is therefore
    # recognised by its OWN text, with the check-name prefix optional, so the
    # interpretation does not depend on which emitter the deployed suite happens to
    # use. The name is still captured separately for the aggregate distinction.
    #
    # Nothing else in this suite's warnings is treated as a counter BY THIS HELPER,
    # and the rejected code treating every digit run as one is the defect this
    # replaces. That is a statement about this helper's own interpretation, not a
    # claim that no other check reports an accumulation: checks/3-topology-check.sh
    # sums NVLink replay, recovery and CRC errors across links and reports them in one
    # sentence. It gets no special treatment here, so its numbers are part of a
    # condition this route refuses to qualify at all rather than compares. Read out of
    # the pinned sources rather than inferred from the sentences:
    #
    #   checks/0-nvidia-smi-check.sh:275,239  '... Xid ${code} (${severity}/${group})'
    #       is a CONDITION IDENTIFIER, and 13|31|94|126 share MONITOR/RESTART_APP
    #       (:219-220), so digit normalisation made Xid 94 and Xid 13 the same
    #       sentence with a 'decreased counter'.
    #   checks/0-nvidia-smi-check.sh:373   'GPU ${gpu_idx}: ${retire_sbe} SBE
    #       retired pages (threshold: ${MAX_RETIRED_PAGES_SBE})' carries a device
    #       index and a THRESHOLD alongside the count.
    #   checks/2-efa-enumeration.sh:189-191 'Memory lock limit ${memlock} KB is
    #       below 16 GiB' fires on `memlock -lt 16777216`: a LOWER number is worse.
    #   checks/3-topology-check.sh:85   'Found ${inactive_links} inactive NVLink(s)'
    #       is a point-in-time count, not an accumulation.
    CHECK6_CUMULATIVE_COUNTERS = (
        ('rx_drops',
         re.compile(r'^EFA rx_drops detected \((\d+)\) -- possible network '
                    r'issues$')),
        ('retrans_timeout_events',
         re.compile(r'^EFA retransmission timeouts detected \((\d+)\)$')),
    )

    @classmethod
    def warning_details(cls, text):
        """The warning conditions in one check's output, as comparable records.

        Returns a list of {'condition': str|None, 'counter': dict|None} in the
        order they were printed.

        `condition` is the warning as the suite printed it, normalised only in
        whitespace and in the '[WARN] <name>: ' prefix: '[WARN] <name>: <details>'
        or '[WARN] <details>'. Every number it contains is KEPT, because in this
        suite's warnings numbers are Xid codes, GPU indices, device counts and
        limits at least as often as they are counters. Two conditions are the same
        condition only when their text is identical.

        `counter` is set only for a line that is one of the two documented
        cumulative counters of check 6, as {'name': str, 'value': int}. It is the
        one place a number is read as a quantity, and it is the caller that decides
        whether a lower value may be accepted.

        `condition` is None for a line that holds '[WARN]' but could not be read as
        a condition. Such a line is uninterpretable, not absent.
        """
        details = []
        for line in (text or '').splitlines():
            match = cls.WARNING_LINE.search(line)
            if not match:
                if cls.WARNING_MARKER.search(line):
                    details.append({'condition': None, 'counter': None})
                continue
            name, body = match.group(1), match.group(2)
            prefix = f'[WARN] {name}: ' if name else '[WARN] '
            counter = None
            # The counter sentences are matched on their own text whether or not the
            # emitter prefixed them with the check name (see
            # CHECK6_CUMULATIVE_COUNTERS: the pinned suite uses log_warn, upstream
            # uses check_warn for the same two sentences). The aggregate is still
            # distinguishable, because it is a DIFFERENT sentence and matches neither
            # pattern.
            for counter_name, pattern in cls.CHECK6_CUMULATIVE_COUNTERS:
                found = pattern.match(body)
                if found:
                    counter = {'name': counter_name,
                               'value': int(found.group(1))}
                    break
            details.append({'condition': prefix + body, 'counter': counter})
        return details

    # The check-6 aggregate `check_warn` sentence, and the part of it that names a
    # cumulative-statistics condition without saying WHICH. Read out of the pinned
    # suite rather than paraphrased: checks/6-efa-loopback.sh@a4ba07eb prints
    #
    #   check_warn "${CHECK_NAME}" "EFA loopback completed for ${device_count}
    #       domain(s); cumulative EFA statistics contain drops or retransmission
    #       timeouts (see logs)"
    #
    # only when `stats_warning` was set, and `stats_warning` is set only by the two
    # per-counter `log_warn` lines above it. So this sentence ASSERTS that at least
    # one of those counters is nonzero while naming neither, and its own '(see logs)'
    # says where the detail is. An observation carrying this sentence and no counter
    # detail is therefore an INCOMPLETE observation of a condition it says exists,
    # not an observation of a different condition. `_baseline_allows_warning` refuses
    # it rather than comparing two incomplete captures with each other.
    #
    # Matched on the invariant tail, so the domain count varies without this becoming
    # a silent mismatch.
    CHECK6_STATISTICS_AGGREGATE = re.compile(
        r'cumulative EFA statistics contain drops or retransmission timeouts')

    @classmethod
    def _claims_cumulative_statistics(cls, details):
        """Does this observation carry the check-6 aggregate that asserts a counter?

        True when any condition line matches the aggregate above. The counter map is
        not consulted: the point is exactly that the aggregate can appear with no
        counter detail behind it.
        """
        return any(item.get('condition')
                   and cls.CHECK6_STATISTICS_AGGREGATE.search(item['condition'])
                   for item in details)

    # Former candidate conditions retained for specific refusal diagnostics, matched
    # on its own whole sentence from the PINNED suite (pins.env HEALTHCHECK_COMMIT =
    # a4ba07eb15e6f277063b4000346f9109c98de843). The rejected implementation compared
    # any two warnings by text and resumed on equality, which is unsound for most of
    # this suite's warnings and was the review's fatal finding: equal text is not equal
    # evidence.
    #
    # A condition may appear here only if all three hold, read out of the check's own
    # source rather than inferred from its wording:
    #
    #   point in time   the sentence reports what the node is like NOW, so two readings
    #                   of the same value are two observations of the same condition.
    #                   An event history and an accumulating counter are not: the same
    #                   number can be a different set of events, and the accumulation
    #                   epoch is not this controller's to observe.
    #   complete        the sentence carries the whole condition -- the quantity and
    #                   whatever identifies it -- so nothing has to be looked up
    #                   elsewhere to know what was observed.
    #   verifiable here the check reads it from a live interface each run, so the
    #                   post-recovery reading is a fresh observation rather than a
    #                   replay of a stored one.
    #
    # Everything else is refused, and the refusal says which of those properties is
    # missing. Refusing is not a finding that the hardware is faulty: it says this route
    # cannot establish that the post-recovery observation is the pre-fault one, so the
    # node keeps its drain. No WARN is currently auto-qualified: even the memlock
    # candidate lacks a supported producer capture/provenance. The final gate below
    # refuses it regardless of matching names or printed revision strings.
    QUALIFIABLE_CONDITIONS = (
        # checks/2-efa-enumeration.sh, step 7: `memlock=$(ulimit -l 2>/dev/null || echo
        # "0")` and `if [[ "${memlock}" -lt 16777216 ]]`. A live limit, carrying its own
        # value and the threshold it is compared against. This is the condition the
        # earlier review considered as a candidate, not a supported acceptance.
        #
        # The value must be NONZERO, and that is not a tightening for its own sake: the
        # pin's own fallback prints `0` when `ulimit -l` fails, so `Memory lock limit 0
        # KB` is the one memlock sentence that can be a failed observation rather than
        # an observed limit. The check does not record which it was, and nothing else
        # this controller holds distinguishes them, so it is refused with everything
        # else that cannot be established.
        ('the memory-lock limit',
         re.compile(r'^Memory lock limit (?!0 KB)\d+ KB is below 16 GiB'
                    r' -- EFA performance may be degraded$')),
    )
    # Warnings the pinned suite emits from a producer whose own success it does not
    # test. They are NOT qualifiable, and the reason is a property of the check rather
    # than of the sentence: step 5 is
    #
    #   for mod in "${required_modules[@]}"; do
    #       if ! lsmod | grep -qw "${mod}"; then warn_efa ... "EFA kernel module ${mod}
    #       not loaded"
    #
    # (checks/2-efa-enumeration.sh@a4ba07eb, step 5, and the gdrdrv branch below it).
    # A negated pipeline takes the warning branch when `lsmod` SUCCEEDS and the module
    # is absent, and equally when `lsmod` fails or is not installed -- kmod's own
    # `do_lsmod` has a `could not get list of modules` path that returns before printing
    # anything (kmod v34, tools/lsmod.c). The check then completes with rc 0 and a
    # consistent advisory aggregate either way, so two captures can agree on this
    # sentence while both are failed observations. Nothing this controller records
    # distinguishes the two, and inferring the producer's success from its own advisory
    # would be the same error as inferring an accumulation epoch.
    #
    # They are listed rather than merely omitted so the refusal can say which property
    # is missing, and so a future pin that adds a producer-success signal has one place
    # to change.
    UNTRUSTED_PRODUCER_CONDITIONS = (
        ('a kernel module reported not loaded',
         re.compile(r'^EFA kernel module [a-z0-9_]+ not loaded$')),
        ('the gdrdrv module reported not loaded',
         re.compile(r'^gdrdrv module not loaded'
                    r' -- GPUDirect RDMA may fall back to slower paths$')),
    )
    # The check-2 aggregate, and the count it reports. `warn_efa` increments
    # `advisory_count` and then calls `check_warn` for each advisory, and the final
    # branch prints this sentence with that count
    # (checks/2-efa-enumeration.sh@a4ba07eb, `warn_efa` and the `advisory_count -gt 0`
    # branch). A WARN verdict from that check therefore implies both the advisories and
    # this line: the two check_fail branches above it return before it is reached, and
    # they make the verdict FAIL. So a check-2 WARN capture carrying this sentence must
    # carry exactly that many detail conditions, and one carrying it with none at all is
    # the summary-only observation the review named.
    ADVISORY_AGGREGATE = re.compile(
        r'EFA devices enumerated with (\d+) advisories; inspect raw output')

    @classmethod
    def _condition_body(cls, condition):
        """One condition line reduced to the sentence the check's own source prints.

        The stored form is '[WARN] <name>: <details>' or '[WARN] <details>', and the
        patterns above are written against `<details>` as it appears in the pinned
        source, so the prefix is removed before matching. Both emitter shapes reach
        here.
        """
        body = condition.split(': ', 1)[1] if ': ' in condition else condition
        return body[len('[WARN] '):] if body.startswith('[WARN] ') else body

    @classmethod
    def _untrusted_producer(cls, details):
        """The conditions here whose own observation the pinned check cannot vouch for.

        See UNTRUSTED_PRODUCER_CONDITIONS. Reported separately from `_unqualifiable`
        because the missing property is different, and the refusal says which it is.
        """
        found = []
        for item in details:
            condition = item.get('condition')
            if not condition:
                continue
            body = cls._condition_body(condition)
            if any(pattern.match(body)
                   for _, pattern in cls.UNTRUSTED_PRODUCER_CONDITIONS):
                found.append(condition)
        return found

    @classmethod
    def _unqualifiable(cls, details):
        """The conditions in this observation that this route will not qualify.

        Returns the condition texts that match no entry of QUALIFIABLE_CONDITIONS, in
        the order they were printed. The check-2 aggregate is not reported here: it is
        judged separately, because its presence is required rather than forbidden.
        """
        rejected = []
        for item in details:
            condition = item.get('condition')
            if not condition:
                continue
            if cls.ADVISORY_AGGREGATE.search(condition):
                continue
            body = cls._condition_body(condition)
            if not any(pattern.match(body)
                       for _, pattern in cls.QUALIFIABLE_CONDITIONS):
                rejected.append(condition)
        return rejected

    @classmethod
    def _advisory_count_is_consistent(cls, details):
        """Does a check-2 aggregate agree with the detail lines beside it?

        Returns (present, consistent). `present` is False when no aggregate is in the
        capture at all, and for a check-2 WARN that is itself a defect rather than the
        ordinary case: `warn_efa` increments `advisory_count` for every advisory it
        prints, and the final branch prints the aggregate whenever that count is
        nonzero, so a check-2 WARN implies both. A capture holding details without it is
        missing part of the observation the check made. `_baseline_allows_warning`
        requires presence for a qualifiable check-2 condition and refuses an
        inconsistent count wherever one appears.

        When an aggregate IS present, the count it reports must equal the number of
        non-aggregate conditions in the same capture: the aggregate counts exactly the
        `warn_efa` calls that printed them, so a mismatch means the capture is missing
        detail the check says it produced.
        """
        aggregate = None
        others = 0
        for item in details:
            condition = item.get('condition')
            if not condition:
                continue
            found = cls.ADVISORY_AGGREGATE.search(condition)
            if found:
                aggregate = int(found.group(1))
            else:
                others += 1
        if aggregate is None:
            return False, True
        return True, aggregate == others

    # The three sentences the pinned check's statistics step can print, quoted from
    # checks/6-efa-loopback.sh@a4ba07eb:179,182,185 rather than paraphrased:
    #
    #   if [[ "${rx_drops}" -eq 0 && "${retrans_timeouts}" -eq 0 ]]; then
    #       log_verbose "EFA statistics clean -- no drops or retransmissions"
    #   ...
    #   else
    #       log_verbose "rdma statistic show failed -- EFA statistics skipped"
    #   ...
    #       log_verbose "rdma tool not found -- EFA statistics skipped"
    #
    # The first is printed only after `rdma -p statistic show` SUCCEEDED and both
    # counters read zero. The other two are printed when the collection step never ran
    # at all, and they leave `stats_warning=0`, so both fall through to `check_pass` at
    # :197 whenever the loopback test itself passed. The resulting PASS sentence is
    # byte-identical in all three cases, which is why the verdict cannot answer
    # whether the counters were observed and these lines can.
    #
    # They arrive in the capture because `maintenance.collect` runs the suite with
    # `--verbose`, which sets VERBOSE=1 (gpu-healthcheck.sh:124-125,181), and
    # log_verbose then prints to stderr (lib/common.sh:49-53), which this controller
    # concatenates with stdout. Confirmed in a stored capture: a real check-2 run over
    # the maintenance route carries '[DEBUG] Memory lock limit OK'
    # (participant-revision/runs/rework7-read-warn-format/output.log:107), so the
    # DEBUG stream of the pinned suite does reach this process.
    CHECK6_STATISTICS_OBSERVED = re.compile(
        r'EFA statistics clean -- no drops or retransmissions')
    CHECK6_STATISTICS_SKIPPED = re.compile(r'EFA statistics skipped')

    @classmethod
    def counters_were_observed(cls, text):
        """Did this check-6 capture actually read the cumulative counters?

        True only on POSITIVE evidence: the clean-statistics sentence above, or a
        per-counter warning line, both of which the check prints only after
        `rdma -p statistic show` returned successfully. Absence is not evidence here --
        a capture that says nothing about the statistics step establishes nothing about
        it, exactly like the two explicit 'skipped' sentences.
        """
        text = text or ''
        if cls.CHECK6_STATISTICS_OBSERVED.search(text):
            return True
        return any(item.get('counter') for item in cls.warning_details(text))

    def _observation_is_complete(self, state, check, recovered_text):
        """Reject explicit incomplete PASS independently of the baseline.

        Check 6's skipped-statistics sentences leave stats_warning=0 and still
        reach PASS at the pin's lines 182/185/197. They take precedence even over
        a clean sentence in the same capture. Preserve the existing additional
        requirement for positive counter evidence after a counter-WARN baseline.
        """
        if self.CHECK6_STATISTICS_SKIPPED.search(recovered_text or ''):
            return False, (
                'this run reports that its statistics collection was skipped. '
                'Its loopback PASS does not establish a complete observation, '
                'regardless of the pre-fault result. This is not a finding that '
                'the hardware is faulty; the node keeps its drain and evidence.')
        recorded = state.baseline_check_results.get(
            f'round-{state.round}/baseline/check-{check}')
        if not recorded or recorded.get('verdict') != 'WARN':
            return True, ''
        before = recorded.get('warnings') or []
        named = sorted(
            item['condition'] for item in before
            if item.get('condition')
            and (item.get('counter')
                 or self.CHECK6_STATISTICS_AGGREGATE.search(item['condition'])))
        if not named:
            return True, ''
        if self.counters_were_observed(recovered_text):
            return True, ''
        how = 'records no observation of those counters at all'
        return False, (
            'this round\'s pre-fault run of that check reported a cumulative EFA '
            f'statistics condition ({"; ".join(named)}), and this run {how}, so it '
            'did not re-read the counters that condition is about. The check\'s own '
            'PASS says its loopback test succeeded; it does not say those counters '
            'were observed, because the pinned check prints the same PASS sentence '
            'whether it read them, found them clean, or never read them at all. '
            'This is not a finding that the hardware is faulty: your node keeps its '
            'drain and your evidence. Retry with a complete observation; if that '
            'cannot be obtained, replacement is required.')

    @classmethod
    def _condition_check_name(cls, condition):
        """The check name the emitter embedded in this condition, or None.

        `check_warn` prints '[WARN] <CHECK_NAME>: <details>' (lib/common.sh:186-191),
        so a condition carrying a name says which check produced it. `log_warn` prints
        '[WARN] <details>' with no name (:41-43), and that is not a mismatch -- it is
        the absence of a claim, which the caller treats separately from a name that
        disagrees.
        """
        if not condition or not condition.startswith('[WARN] '):
            return None
        body = condition[len('[WARN] '):]
        if ': ' not in body:
            return None
        name = body.split(': ', 1)[0]
        return name or None

    # The check names the pinned suite uses for the checks this exercise allows, read
    # from each check's own CHECK_NAME at a4ba07eb rather than constructed from the
    # identifier: the identifier the participant types is '2', and the name the emitter
    # prints is '2-efa-enumeration'. A check whose name is not listed here cannot have
    # its producer identity tested, and `_baseline_allows_warning` refuses to
    # auto-qualify a warning it cannot attribute rather than accepting it unchecked.
    CHECK_NAMES = {
        '0': '0-nvidia-smi',
        '2': '2-efa-enumeration',
        '3': '3-topology-check',
        '6': '6-efa-loopback',
    }

    def _producer_identity_disagrees(self, check, details):
        """Conditions whose own embedded check name is not the requested check's.

        This is the producer-identity test the qualification decision was missing: it
        was called with the check it asked for and never compared that against the name
        the emitter printed, so a capture carrying another check's WARN text qualified
        as long as both sides carried the same wrong text. That is what a suite-identity
        mismatch, a pin drift, or a maintenance route that returned the wrong check's
        output looks like on the wire.

        A condition with no embedded name is not reported here: `log_warn` prints the
        two check-6 counter sentences without one, and those are refused on their own
        cumulative-counter grounds. Only a name that DISAGREES is an attribution
        failure.
        """
        expected = self.CHECK_NAMES.get(str(check))
        found = []
        for item in details:
            condition = item.get('condition')
            if not condition:
                continue
            name = self._condition_check_name(condition)
            if name is None:
                continue
            if expected is None or name != expected:
                found.append(condition)
        return found

    def _baseline_allows_warning(self, state, check, recovered_text):
        """Is this post-recovery WARN the same condition, unchanged, as before?

        Returns (allowed, sentence); currently every WARN is refused, including
        the former memlock candidate. The existing comparisons are retained to
        explain specific missing evidence before the unconditional final refusal.
        The sentence is what the participant and the
        record are told, so it must describe the evidence rather than assert a
        conclusion. Matching pre-fault conditions are historical diagnostics,
        not current authorization to accept a WARN.

        Equality is on the conditions as printed, identifiers and limits included.
        The rejected implementation replaced every digit run with '#' and then
        accepted any number that had not risen, which is wrong twice over in this
        suite's own warnings:

          checks/0-nvidia-smi-check.sh:275   'Found 1 Xid message(s): Xid 94
              (MONITOR/RESTART_APP)' and the same sentence with 'Xid 13' normalise
              to one shape, because 13|31|94|126 share that severity/group
              (:219-220). A DIFFERENT fault condition read as a decreased counter.
          checks/2-efa-enumeration.sh:189-191 'Memory lock limit 8192 KB is below
              16 GiB' fires on `memlock -lt 16777216`. A LOWER limit is a worse
              condition, and 'has not risen' accepted it.

        The two counters check 6 documents as cumulative (CHECK6_CUMULATIVE_COUNTERS)
        are the only numbers read as quantities, and a warning that reports one of
        them is NEVER qualified here. Their accumulation epoch is the instance launch
        or the last driver reset, not the boot: AWS's EFA metric documentation says
        the values are cumulative "since instance launch or the last driver reset"
        (EC2 User Guide, "Monitor an Elastic Fabric Adapter on Amazon EC2", available
        EFA driver metrics), and the kernel driver really does reset on its own
        lifecycle -- efa_probe reaches efa_com_dev_reset(EFA_REGS_RESET_NORMAL) and
        efa_remove_device reaches efa_com_dev_reset (drivers/infiniband/hw/efa/
        efa_main.c). An EFA round performs exactly that lifecycle: this exercise's own
        helper writes the driver's unbind and bind files (device-fault.sh:143-146).
        So no reboot is needed to restart the accumulation, and this controller has no
        observation of the epoch at all -- nothing it records distinguishes "the same
        counter still at 2" from "reset, then two new events". Equality rejects a
        visible fall; it cannot detect a reset followed by a return to the same value.
        Nor is the epoch inferrable: a counter value, an uptime and an interface name
        are all consistent with either history, and inventing an epoch from them would
        be another inference in place of an observation.

        The consequence is deliberate and it is the fail-closed direction: a residual
        counter warning cannot be auto-resumed by this route at all. The round says so
        and keeps its drain for participant replacement. What is NOT claimed is that
        the hardware is unhealthy -- only that this route cannot establish that the
        post-recovery counter is the pre-fault one.

        The check-6 aggregate is refused on its own for a related but distinct reason
        (CHECK6_STATISTICS_AGGREGATE): that sentence asserts a cumulative-statistics
        condition without naming which one, so an observation carrying it without the
        per-counter detail is an incomplete observation of a condition it says exists.
        Two such captures cannot qualify each other, however identical they are.

        A '[WARN]' whose condition could not be read is a refusal, not an absence.

        The comparison is on the text the maintenance route returned to this
        process, recorded in the root-owned state file under
        /var/lib/aim344-device-session (0700 root, install-participant-control.sh
        writes it). The participant's own copy of the log is not consulted: they
        can rewrite it, and it must not be able to authorize returning a node to
        service.
        """
        key = f'round-{state.round}/baseline/check-{check}'
        recorded = state.baseline_check_results.get(key)
        if not recorded:
            return False, ('this round recorded no pre-fault result for that '
                           'check, so there is nothing establishing the warning '
                           'was already there.')
        verdict = recorded.get('verdict')
        if verdict != 'WARN':
            return False, (f'this round\'s pre-fault run of that check reported '
                           f'{verdict}, not WARN, so the warning is new.')
        if recorded.get('returncode') != 0:
            return False, ('this round\'s pre-fault run of that check did not '
                           'complete, so it does not establish a baseline.')
        before = recorded.get('warnings')
        if not before:
            # A WARN verdict with no recorded warning detail is an incomplete
            # baseline, not a permissive one. It happens when the baseline was
            # captured by an older helper, or when the text did not survive.
            return False, ('this round\'s pre-fault run of that check warned but '
                           'its warning details were not recorded, so there is '
                           'nothing to compare this warning against.')
        if any('condition' not in item for item in before):
            # A record written by the helper that normalised every number away.
            # Its shapes cannot be compared with conditions, and reading them as
            # conditions would compare a normalised sentence with a real one.
            return False, ('this round\'s pre-fault warning details were recorded '
                           'in a format this helper cannot compare, so the '
                           'warning cannot be qualified.')
        after = self.warning_details(recovered_text)
        if not after:
            return False, ('this check reported WARN but no warning line could be '
                           'read from its output, so the condition cannot be '
                           'compared with the pre-fault one.')
        if any(item['condition'] is None for item in after):
            return False, ('this check printed a warning line this helper could '
                           'not read as a condition, so it cannot be compared '
                           'with the pre-fault one.')
        if any(item['condition'] is None for item in before):
            return False, ('this round\'s pre-fault run printed a warning line '
                           'this helper could not read as a condition, so there '
                           'is nothing comparable to qualify against.')

        # Conditions as a multiset of texts, and the documented counters keyed by
        # name. A repeated identical condition is not more than one condition.
        def conditions(details):
            return {item['condition'] for item in details
                    if not item.get('counter')}

        def counters(details):
            table = {}
            for item in details:
                counter = item.get('counter')
                if counter:
                    # The worst the run reported for that counter, so a repeated
                    # line is compared against the highest rather than the last.
                    name, value = counter['name'], int(counter['value'])
                    table[name] = max(value, table.get(name, value))
            return table

        baseline_conditions, recovered_conditions = conditions(before), conditions(after)
        appeared = sorted(recovered_conditions - baseline_conditions)
        if appeared:
            return False, ('this warning is not the one the pre-fault run '
                           f'produced. New after recovery: {"; ".join(appeared)}.')
        vanished = sorted(baseline_conditions - recovered_conditions)
        if vanished:
            # The conditions changed even though nothing new appeared. Whatever
            # made a pre-fault condition stop being reported also makes this
            # observation something other than the recorded one.
            return False, ('this warning does not report what the pre-fault run '
                           'reported; these conditions are no longer named: '
                           f'{"; ".join(vanished)}.')

        baseline_counters, recovered_counters = counters(before), counters(after)
        if baseline_counters or recovered_counters:
            # A cumulative counter cannot be qualified by this route at all. The
            # accumulation epoch is the instance launch or the last driver reset, and
            # an EFA round performs a driver unbind/rebind, so nothing this controller
            # observed establishes that the two counts accumulated from the same
            # start. The refusal names the counters, says what could not be
            # established, and says what it does NOT mean, because a participant
            # reading it must not conclude either that the hardware is broken or that
            # the warning is nothing.
            named_counters = ', '.join(
                f'{name}={value}' for name, value in
                sorted((recovered_counters or baseline_counters).items()))
            return False, (
                'this warning reports a cumulative EFA counter '
                f'({named_counters}), and this route cannot establish that the '
                'count it reports accumulated from the same start as the pre-fault '
                'one: those counters restart at the instance launch or at the last '
                'EFA driver reset, and this round rebound that driver. An equal '
                'count is therefore not evidence that this is the pre-existing '
                'condition, and nothing here observed the reset. This warning '
                'cannot be resumed automatically, and this is not a finding that '
                'the hardware is faulty; your node keeps its drain and your '
                'evidence; replacement is required.')

        if (self._claims_cumulative_statistics(after)
                or self._claims_cumulative_statistics(before)):
            # The aggregate says a cumulative-statistics condition exists and does not
            # say which; with no counter detail in either observation, the two
            # captures are incomplete observations of that condition rather than
            # observations of the same one. Reached only when neither map held a
            # counter, since the branch above returns first.
            return False, (
                'this check reports that cumulative EFA statistics contain drops or '
                'retransmission timeouts without reporting either counter, so the '
                'condition it names was not observed in a form this round can '
                'compare with its pre-fault run. An aggregate that says "see logs" '
                'is not the condition; your node keeps its drain and your evidence, '
                'and replacement is required.')

        # Equal text is not equal evidence, which is why equality is necessary here and
        # not sufficient. The former point-in-time candidate is also disabled at the
        # final gate; every other warning in this suite either reports an event
        # history, an accumulation, a summary of detail that is not in the sentence,
        # or a condition whose own observation the check does not establish, and for
        # those two identical captures can be two different situations. The refusal
        # names the conditions it would not accept, so a participant can see WHICH
        # warning kept the node drained, and says which property was missing.
        for observation, side in ((after, 'this check'),
                                  (before, 'this round\'s pre-fault run')):
            mismatched = self._producer_identity_disagrees(check, observation)
            if mismatched:
                expected = (self.CHECK_NAMES.get(str(check))
                            or 'not a check this route can attribute')
                return False, (
                    f'{side} reported a warning whose own text names a different '
                    f'check than the one this round asked for '
                    f'({"; ".join(sorted(set(mismatched)))}; check {check} is '
                    f'{expected}). The emitter prints the check\'s own name beside '
                    f'each warning, so a name that disagrees means this capture is '
                    f'not an observation of the check it was requested for, and '
                    f'nothing here establishes which run produced it. This is not a '
                    f'finding that the hardware is faulty: your node keeps its drain '
                    f'and your evidence; replacement is required.')
            untrusted = self._untrusted_producer(observation)
            if untrusted:
                return False, (
                    f'{side} reported a warning whose own observation this route '
                    f'cannot confirm succeeded ({"; ".join(sorted(set(untrusted)))}). '
                    f'The pinned check emits that sentence both when it read the '
                    f'system successfully and found the condition, and when the '
                    f'reading itself failed, and it records nothing that separates '
                    f'them; two runs agreeing on it are therefore not two '
                    f'observations of the same condition. This is not a finding that '
                    f'the hardware is faulty: your node keeps its drain and your '
                    f'evidence; replacement is required.')
            rejected = self._unqualifiable(observation)
            if rejected:
                return False, (
                    f'{side} reported a warning this route will not qualify '
                    f'automatically ({"; ".join(sorted(set(rejected)))}). Only a '
                    f'condition that describes the node as it is now, carries its own '
                    f'detail, and is re-read from the system on every run can be '
                    f'established as unchanged by comparing two runs; an event count, '
                    f'an accumulated total or a summary that points at a log cannot, '
                    f'because the same text can describe a different situation. This '
                    f'is not a finding that the hardware is faulty: your node keeps '
                    f'its drain and your evidence; replacement is required.')

        # A check-2 warning implies its own advisory lines AND its aggregate, so a
        # capture missing either is missing detail the check says it produced. Both
        # sides are checked, because an incomplete baseline cannot qualify anything
        # either. Requiring the aggregate's PRESENCE is what the previous version left
        # out: it refused a count that disagreed and accepted a capture with no count at
        # all, so detail-only output qualified on exactly the evidence the aggregate
        # exists to complete.
        for observation, side in ((after, 'this check'),
                                  (before, 'this round\'s pre-fault run')):
            present, consistent = self._advisory_count_is_consistent(observation)
            if present and not consistent:
                return False, (
                    f'{side} reported a count of EFA advisories that does not match '
                    f'the advisories in the same output, so the conditions behind that '
                    f'count were not all observed here. An incomplete observation '
                    f'cannot be compared with another one; your node keeps its drain '
                    f'and your evidence; replacement is required.')
            if not present:
                return False, (
                    f'{side} reported an EFA advisory without the count the pinned '
                    f'check prints beside it, so this capture does not establish how '
                    f'many advisories that run produced. What is missing may be the '
                    f'condition that matters; your node keeps its drain and your '
                    f'evidence; replacement is required.')

        named = '; '.join(sorted(recovered_conditions))
        return False, (
            'memlock advisory WARN auto-qualification is disabled: no supported '
            'producer capture establishes this acceptance. Matching warning text, '
            'check names or printed suite revisions cannot enable it '
            f'({named}). This is not a finding that the hardware is faulty.')

    # The active EFA path's pre-injection conditions, from the approved plan and
    # DEVICE-RECOVERY.md:53: "Preserve two advancing progress records and that
    # EFA's increasing counters before injection. Immediately before mutation,
    # confirm that the same job is running and that its latest progress record is
    # recent. Abort injection if progress stopped during preparation."
    #
    # Those conditions are implemented below and qualified on the pair. They are
    # NOT, however, enough to make the active path a participant-operated round on
    # this stack, and that was established by measurement rather than by reading:
    # the sysfs unbind write does not return while the workload holds the device.
    # Measured twice on gpu-g7-1. Round 13 issued the unbind at 15:24:55, the
    # kernel logged 'Unregister ib device' immediately, and the write completed at
    # 15:29:55 -- the same second the workload was cancelled, 300 s later, having
    # already exceeded the maintenance route's mutation timeout. A dedicated probe
    # then held the write open deliberately: it was still outstanding at t+159 s
    # with the job RUNNING, and returned at t+171 s, 11 s after the job ended
    # (participant-revision/runs/rework5-unbind-blocking-probe/output.log). So the
    # duration is not a property of the device; it is the holding process.
    #
    # A participant drives one command per connection through a forced command.
    # An operation that cannot return until they cancel the job it is faulting
    # cannot be completed from that route: they would have to end their own
    # allocation from a second terminal while `start efa` is still blocked, and
    # the round's own state record would stay at 'preparing' throughout. This is
    # the same class as the active GPU removal this exercise already excludes for
    # a measured reason (DEVICE-RECOVERY.md:40).
    #
    # The participant route therefore refuses an active EFA start and says why,
    # which is the second option the review left open: keep the active path
    # unavailable pending an authorised adoption decision rather than ship a round
    # that silently tests the idle path. The qualification below is retained and
    # exercised, because a facilitator-driven active trial needs exactly it, and
    # because adopting the path later must not mean writing these conditions from
    # scratch.
    ACTIVE_EFA_PARTICIPANT_PATH_ADOPTED = False
    # Minimum bytes the selected device must have moved across the observation
    # window for it to count as carrying the workload. A few hundred kilobytes of
    # scheduler and health traffic can appear on an otherwise idle device, so the
    # threshold is well above that and far below what the device workload moves:
    # its tensor alone is 32 MiB per collective (workload.py:129), and a measured
    # active window moved 304 GB in 6 s while an idle one moved 0 B.
    ACTIVE_EFA_MIN_BYTES = 8 * 1024 * 1024
    # How long the job must already have been running. A job that started a second
    # ago has not established advancing collectives.
    ACTIVE_EFA_MIN_RUNTIME_SECONDS = 30

    @staticmethod
    def _parse_slurm_runtime(text):
        """Seconds from Slurm's %M elapsed field, or None if it is unreadable.

        Slurm prints elapsed time in five shapes: MM:SS, HH:MM:SS, D-HH:MM:SS,
        and for a job that has not started, 0:00 or INVALID. Returning None for
        anything unrecognised matters, because the caller must treat an
        unreadable runtime as "not established" rather than as zero or as large.
        """
        value = (text or '').strip()
        if not re.fullmatch(r'(?:\d+-)?[\d:]+', value):
            return None
        days = 0
        if '-' in value:
            day_part, value = value.split('-', 1)
            days = int(day_part)
        parts = value.split(':')
        if not all(part.isdigit() for part in parts) or not 1 <= len(parts) <= 3:
            return None
        numbers = [int(part) for part in parts]
        while len(numbers) < 3:
            numbers.insert(0, 0)
        hours, minutes, seconds = numbers
        return days * 86400 + hours * 3600 + minutes * 60 + seconds

    def _active_progress(self, state):
        """Associate existing coordinator workload output with this exact job.

        Participant output is activity evidence, NOT identity authority. The
        target independently checks scheduler owner, GPU PID/cgroup and traffic.
        Read as that participant, never root through a participant-owned path.
        """
        uid, gid = self._participant_ids()
        if uid is None or not re.fullmatch(r'[1-9][0-9]*', str(state.job_id)):
            raise Refusal('No participant/job for collective progress.')
        path = self._answers_root() / f'device-{state.job_id}.log'
        reader = ('import os,stat,sys; '
                  'f=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK); '
                  's=os.fstat(f); '
                  'assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid(); '
                  'os.lseek(f,max(0,s.st_size-262144),os.SEEK_SET); '
                  'sys.stdout.buffer.write(os.read(f,262144))')
        identity = {'user': uid, 'group': gid, 'extra_groups': []} if os.geteuid() == 0 else {}
        if os.geteuid() not in (0, uid):
            raise Refusal('Cannot read participant progress under its owner.')
        try:
            result = subprocess.run([sys.executable, '-c', reader, str(path)],
                                    capture_output=True, text=True, timeout=5, **identity)
        except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
            raise Refusal('Coordinator collective progress could not be read; no ACK.') from error
        return self._parse_active_progress(state, result.stdout if result.returncode == 0 else '')

    def _parse_active_progress(self, state, text):
        pattern = (r'AIM344 collective_progress=(\d+) collectives; tensor_bytes=(\d+) B; '
                   r'timestamp=([0-9.]+) s since epoch; 0 mismatches; '
                   r'job_id=([1-9][0-9]*); world_size=([1-9][0-9]*); rank=0\.')
        records = [m.groups() for m in re.finditer(pattern, text)]
        if len(records) < 2:
            raise Refusal('Need two advancing collective records in the coordinator job log.')
        before, after = records[-2:]
        try:
            valid = (before[3] == after[3] == str(state.job_id)
                     and before[4] == after[4] and int(after[4]) >= 2
                     and int(after[0]) > int(before[0]) > 0
                     and int(before[1]) == int(after[1]) > 0
                     and 0 < float(after[2]) - float(before[2]) <= 15
                     and 0 <= self.clock() - float(after[2]) <= 15)
        except ValueError:
            valid = False
        if not valid:
            raise Refusal('Collective progress is stale, stopped or belongs to a different job.')
        return {'job_id': state.job_id, 'records': [list(before), list(after)],
                'source': 'participant coordinator workload log; not identity authority'}

    def _require_active_efa_workload(self, state, own_job):
        """Qualify the active EFA path, or refuse to inject on it.

        Three conditions, all observed rather than asserted:

        the participant's own job is still RUNNING on the exercise node and has
        been for long enough to have completed collectives;
        the SELECTED EFA's byte counters advanced during a fresh observation
        window on the target itself, read by root from the provisioned device
        rather than from anything the caller names;
        the same job is still the one running when that window closes, so a
        workload that stopped while preparation was in flight aborts the
        injection instead of being unbound as though it were live.

        The rejected implementation had none of this: it selected an own-name
        prefixed job (:800-801 @0fdf6204) and forwarded its id to efa-unbind, and
        device-fault.sh:116-140 checked ownership and identity but never progress
        or traffic. So a queued or idle job named aim344-anything qualified the
        active path, and the recorded EFA rounds all started from an empty queue,
        which means they exercised the idle path only.

        A refusal here leaves the round at 'preparing' with its drain in place and
        no device touched; recover restores it.
        """
        outcome = self.executor.run_maintenance(['efa-activity'], timeout=180)
        if outcome.returncode != 0:
            detail = (outcome.stderr or outcome.stdout or '').strip().splitlines()
            message = detail[-1] if detail else 'no detail'
            raise Refusal(
                'The active EFA round needs the selected device observed carrying '
                f'your workload, and that observation did not complete ({message}). '
                'Nothing was changed. End your allocation and use the idle EFA '
                'round instead.')
        record = None
        for line in (outcome.stdout or '').splitlines():
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if candidate.get('action') == 'efa-activity':
                record = candidate
                break
        if record is None:
            raise Refusal('The maintenance route returned no activity observation '
                          'for the selected EFA; nothing was changed.')

        failures = []
        deltas = record.get('deltas_bytes') or {}
        moved = sum(value for name, value in deltas.items()
                    if name.endswith(('/tx_bytes', '/rx_bytes')))
        if not deltas:
            failures.append('no byte counters were read for the selected device, '
                            'so its traffic is unestablished')
        elif moved < self.ACTIVE_EFA_MIN_BYTES:
            failures.append(
                f'the selected device {record.get("device")} moved {moved} B '
                f'during a {record.get("interval_s", 0):.1f} s observation, below '
                f'the {self.ACTIVE_EFA_MIN_BYTES} B this round requires as '
                f'evidence that your collectives are using it')
        if not record.get('queue_readable'):
            failures.append('the exercise node queue could not be read on both '
                            'sides of the observation')

        # The same job, still running, on both sides of the window. A different
        # job or a vanished one means the workload this round was about to fault
        # is not the workload that is running now.
        for label in ('jobs_before', 'jobs_after'):
            rows = record.get(label) or []
            match = [row for row in rows if row.get('id') == own_job['id']]
            if not match:
                failures.append(f'your job {own_job["id"]} was not on the node '
                                f'when the observation {label.split("_")[1]}')
                continue
            job = match[0]
            if job.get('state') != 'RUNNING':
                failures.append(f'your job {own_job["id"]} was {job.get("state")}, '
                                f'not RUNNING, when the observation '
                                f'{label.split("_")[1]}')
            runtime = self._parse_slurm_runtime(job.get('runtime'))
            if runtime is None:
                failures.append(f'your job {own_job["id"]} reported an unreadable '
                                f'elapsed time ({job.get("runtime")!r})')
            elif runtime < self.ACTIVE_EFA_MIN_RUNTIME_SECONDS:
                failures.append(
                    f'your job {own_job["id"]} had run for {runtime} s, less than '
                    f'the {self.ACTIVE_EFA_MIN_RUNTIME_SECONDS} s this round '
                    f'requires before an active injection')

        saved = None
        try:
            saved = self._save_output(
                'efa-activity', json.dumps(record, indent=2, sort_keys=True) + '\n',
                phase='active-precondition', round=state.round)
        except Refusal:
            # The observation itself is what matters; failing to file a copy of it
            # is recorded rather than allowed to abort a qualified round.
            pass
        state.active_efa_evidence = {
            'device': record.get('device'),
            'interval_s': record.get('interval_s'),
            'moved_bytes': moved,
            'job_id': own_job['id'],
            'qualified': not failures,
            'refusals': failures,
            'log': None if saved is None else str(saved),
            'observed_at': self.clock(),
        }
        self.store.write(state)
        if failures:
            state.failure = ('The active EFA preconditions were not met: '
                             + '; '.join(failures))
            self.store.write(state)
            raise Refusal(
                'The active EFA round was not started, because the conditions it '
                'requires were not observed: ' + '; '.join(failures)
                + '. Your node keeps its reservation and no device was changed. '
                'Start the device workload and let it report progress, then start '
                'again; or end your allocation and run the idle EFA round.')

    # Same grammar as maintenance.OPERATION_TOKEN; persisted tokens are not normalized.
    OPERATION_TOKEN = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}/[0-9]{1,6}/[0-9a-f]{32}')

    def _require_operation(self, state):
        """This round's operation identifier, or a refusal.

        A round started before this change has no identifier, and the target now
        requires one on both calls. Rather than inventing one -- which would let a
        restore match records it never wrote -- the round stops and keeps its drain.
        """
        operation = state.operation
        if not isinstance(operation, str) or not self.OPERATION_TOKEN.fullmatch(operation):
            raise Refusal(
                'This round has no usable recorded operation identifier, so the exercise '
                'node\'s original-state records cannot be shown to belong to it. '
                'Your node stays out of service; replacement is required, not '
                'instructor restoration.')
        return operation

    # The line maintenance.gpu_prepare prints once BOTH original-state records are on
    # the target and before it changes either of them (maintenance.py:589-592: the
    # two `_write_record` calls, then this line, then the `systemctl stop` at :593-596
    # and the persistence write at :597-600). Quoted from that source rather than
    # paraphrased, because this controller reads it to decide whether a failed
    # preparation left a state that can be put back:
    #
    #   sys.stdout.write(f'recorded persistence_mode={mode} {TELEMETRY_UNIT}=...
    #
    # Only the invariant prefix is matched, so the mode, the unit and the operation
    # can change without this becoming a silent mismatch.
    ORIGINALS_RECORDED = 'recorded persistence_mode='
    # The exit status the maintenance route uses for its OWN refusals, and the only
    # one that establishes "the target reached a decision and declined". Read out of
    # maintenance.main rather than inferred from the number: `except Refusal as
    # refusal: sys.stderr.write(...); return 3`, while its `except
    # subprocess.TimeoutExpired` returns 124 and a non-root invocation returns 1.
    #
    # This matters because the absence of ORIGINALS_RECORDED is only meaningful when
    # the helper's stdout is complete. A refusal exits main normally, so the
    # interpreter flushes; a run killed at the observation deadline does not, and
    # stdout is a block-buffered pipe under ssh, so the marker is lost even when both
    # records WERE written. Measured: a child that writes the marker and then works
    # past the deadline yields b'' of captured stdout without an explicit flush.
    # Treating that as "refused before capture" would decline to restore a node this
    # round really did change, which is the wrong direction.
    MAINTENANCE_REFUSED = 3

    def _new_operation(self, round_number):
        """A unique identifier for one accepted start of one round.

        `<assignment>/<round>/<32 hex>`. The random half is what makes it unique; the
        assignment and round are there so a facilitator reading the target's record
        can tell whose it is. Generated by this root helper, never supplied by the
        participant: the forced command gives them one fixed verb per connection and
        passes them no arguments at all.

        It is written into the round's state record before any side effect, and every
        retry of that round reuses it unchanged. That is what makes an interrupted
        preparation safe: the target's records either carry this operation, in which
        case they are this round's, or they do not, in which case the restore refuses
        instead of putting back a state some other round observed.
        """
        return (f'{self.assignment_id}/{int(round_number)}/'
                f'{secrets.token_hex(16)}')

    def _capture_baseline(self, state):
        """Run each required check once before the fault, and record the verdicts.

        This is what makes a later 'the baseline also warns' statement checkable.
        It runs while the node is drained and before any device mutation, through
        the same maintenance route and the same suite entry point the recovery
        qualification uses, so the two observations are comparable.

        A baseline that cannot be collected does not itself stop the round.
        Capture supports diagnosis and comparison only; missing capture leaves
        less diagnostic evidence. All WARN acceptance remains disabled, whether
        or not a matching baseline was recorded.
        """
        for check in self.allowed_checks(state):
            try:
                outcome = self.executor.run_maintenance(['collect', check],
                                                        timeout=900)
                text = (outcome.stdout or '') + (outcome.stderr or '')
                saved = self._save_output(check, text, phase='baseline',
                                         round=state.round)
                verdict = self.check_verdict(text) or 'UNREADABLE'
                record = {
                    'verdict': verdict,
                    'returncode': outcome.returncode,
                    'log': str(saved),
                    # The warning conditions themselves, not just the label. This
                    # is what makes a later 'the same warning' statement checkable,
                    # and it is kept in the root-owned state file rather than read
                    # back from the participant's copy of the log.
                    'warnings': self.warning_details(text),
                    # Which boot this observation was made on. The two check-6
                    # counters are cumulative, so a count recorded before a reboot
                    # and a count read after one did not accumulate from the same
                    # start; `_baseline_allows_warning` refuses to compare them.
                    'boot_id': state.boot_id,
                }
            except Refusal as refusal:
                record = {'verdict': 'NOT-COLLECTED', 'returncode': None,
                          'log': None, 'warnings': [], 'boot_id': state.boot_id,
                          'detail': str(refusal)}
            state.baseline_check_results[
                f'round-{state.round}/baseline/check-{check}'] = record
        self.store.write(state)

    # ---------------- verbs ----------------
    def start(self, kind, active=False):
        if kind not in ('gpu', 'efa'):
            raise Refusal('The exercise provides start gpu and start efa only.')
        # The pair lock is taken before the state is even read: a scan that
        # decides the node is free is only meaningful while nobody else can be
        # deciding the same thing. Assignment exclusion alone does not do that,
        # because two tables on one node hold two different assignment locks.
        with self.store.exclusive_pair():
            self._refresh_binding()
            state = self.store.read()
            previous_round = state.round
            if state.phase in ('runtime-ready', 'verified'):
                # The previous round finished recovering, so the participant may
                # begin the next one. This is a new round, not a repeat request.
                state = State(round=previous_round + 1)
            if state.phase == 'recovery-failed':
                raise Refusal(
                    'Your previous round did not finish recovering, so your node '
                    'is still out of service. Run recover again, or use replace '
                    'if recovery cannot qualify it; a new round cannot start on a node that has '
                    'not been restored.')
            if state.phase != 'ready':
                if state.kind != kind:
                    raise Refusal(
                        f'Your {state.kind} exercise is already at {state.phase}. '
                        f'Finish it with recover before starting {kind}.')
                return Result(state, already_started=True,
                              next_step='Use status, then collect a check.')

            # A display label cannot turn an existing operation into a new first
            # capture. Completed rounds were reset above; a genuinely unused
            # record has none of these operation/preparation facts.
            if (state.operation or state.prepared or state.kind
                    or state.started_at is not None or state.target_node
                    or state.target_instance_id or state.reboot_requested_at is not None
                    or state.reboot_requests or state.reboot_completed):
                raise Refusal(
                    'This ready-labelled record still contains an existing '
                    'operation. A new original-state capture is not authorized; '
                    'the node stays out of service.')

            round_number = state.round
            line = self._node_line()
            node_state = self._node_field(line, 'State')
            reason = self._node_field(line, 'Reason')
            ours = self.assignment['drain_reason']
            if 'DRAIN' in node_state and reason and reason != ours:
                raise Refusal(
                    f'Your exercise node is already drained for a different reason '
                    f'({reason}). It was left as it is; this assignment cannot clear unrelated isolation.')

            # Per-pair exclusive ownership, before anything is changed. The pair
            # lock above already serialises this, so the recorded-state scan is
            # now a check on a stable view rather than a race.
            holder = self._other_assignment_holding_target()
            if holder is not None:
                raise Refusal('Another table is using this exercise node right now. '
                              'Wait until they finish recovering, then start again.')

            jobs = self._jobs_on_target()
            own = self._own_job(jobs)
            if active and (not own or own['state'] != 'RUNNING'):
                raise Refusal('Active FLR needs your one RUNNING device workload allocation.')
            if kind == 'gpu' and jobs and not active:
                raise Refusal('The GPU exercise runs while your node is idle. '
                              'End your allocation, then start again.')
            if (kind == 'efa' and own and not active
                    and not self.ACTIVE_EFA_PARTICIPANT_PATH_ADOPTED):
                # Refused BEFORE the drain, so a participant who happens to have a
                # job running is not left holding a reserved node for a round that
                # cannot proceed. The measurement behind this is in the class
                # comment above: the unbind write does not return until the holding
                # workload ends, so this round cannot be completed from a
                # one-command-per-connection route.
                raise Refusal(
                    'The EFA exercise runs while your node is idle. A job of '
                    'yours is running on it, and the active variant is not '
                    'available from this route: the unbind does not return until '
                    'the workload holding the device ends, which was measured on '
                    'this pair at 171 s outstanding, returning 11 s after the job '
                    'was cancelled. End your allocation, then start again. '
                    'Nothing was changed and your node was not reserved.')

            inventory = self._inspect()
            instance_id = inventory.get('instance_id')
            if instance_id != self.assignment['target_instance_id']:
                raise Refusal('Your exercise node no longer reports the expected '
                              'instance identity; nothing was changed.')
            operation = kind + '-flr' if active else ('gpu-remove' if kind == 'gpu' else 'efa-unbind')
            token = (inventory.get('confirmation') or {}).get(operation)
            if not token:
                raise Refusal('The maintenance route returned no confirmation for '
                              'this operation.')
            boot_id = self._boot_id()

            # Ownership and intent are recorded before the first mutation, along
            # with the node's original state, so a failure part way through
            # preparation still leaves a record recovery can act on. Without this
            # a failed gpu-prepare left the node drained and half modified while
            # the stored state still said 'ready', and recover then refused with
            # 'There is no exercise to recover yet'.
            state = State(phase='preparing', kind=kind, started_at=self.clock(),
                          target_node=self.assignment['target_node'],
                          target_instance_id=instance_id,
                          job_id=own['id'] if own else None, boot_id=boot_id,
                          round=round_number,
                          operation=self._new_operation(round_number),
                          original_node_state=node_state,
                          original_node_reason=node_reason(line)[1],
                          deadline_at=self.clock()
                          + float(self.assignment['recovery_deadline_seconds']))
            if active:
                state.active_fault = {'mutation': operation, 'bdf': inventory[kind + '_bdf'],
                                      'boot_id': boot_id, 'events': [], 'outcome': 'not-issued'}
            self.store.write(state)

            if 'DRAIN' not in node_state:
                drained = self._scontrol(
                    'update', f'NodeName={self.assignment["target_node"]}',
                    'State=DRAIN', f'Reason={ours}')
                if drained.returncode != 0:
                    state.failure = 'The exercise node could not be reserved.'
                    self.store.write(state)
                    raise Refusal(f'Could not reserve your exercise node: '
                                  f'{drained.stderr.strip()}')
                state.prepared.append('drain')
                self.store.write(state)

            # The baseline for this round, collected while the node is drained and
            # before any device is touched. Retain it for comparison and diagnosis;
            # a matching baseline does not authorize WARN acceptance. Run after drain
            # so no foreign job competes for the device while it measures.
            if not active:
                self._capture_baseline(state)

            if kind == 'efa' and own and not active:
                # The active EFA path. The plan requires recent advancing
                # collectives AND increasing traffic on the SELECTED device,
                # observed immediately before the mutation; a job whose name
                # starts with the exercise prefix establishes neither. Both are
                # observed here through the target's own root route, and a round
                # that cannot establish them does not proceed on the active path.
                self._require_active_efa_workload(state, own)
            if kind == 'gpu' or active:
                # The existing helper requires persistence disabled and telemetry
                # not holding the GPU; it does not change them itself. Record and
                # pause both, so recovery can put them back.
                #
                # The intent is persisted BEFORE the call, not after it and not
                # only in the exception handler. gpu_prepare stops nvidia-dcgm
                # first and disables persistence second (maintenance.py:214-229),
                # so a coordinator process that dies between those two steps left
                # a durable record with no marker at all, and recovery skipped
                # gpu-restore because it only tests these markers. Telemetry then
                # stayed paused while the round still reached runtime-ready.
                state.prepared.append('gpu-prepare-attempted')
                self.store.write(state)
                # The operation identifier is on the request, so recovery cannot be
                # satisfied by original-state records another round -- or another
                # table sharing this node -- left behind. It has to be an argument:
                # the maintenance route is an ssh forced command, and sshd carries
                # only SSH_ORIGINAL_COMMAND, so an exported variable never arrives.
                #
                # The call is made through the executor rather than `_maintenance`
                # because the REFUSAL's own output is evidence this round needs. The
                # target writes both original-state records and then prints
                # `recorded persistence_mode=... operation=...` BEFORE it pauses
                # telemetry or disables persistence (maintenance.gpu_prepare). So a
                # refusal carrying that line is a round whose originals exist and
                # whose telemetry may already have been changed, and a refusal
                # without it is a round that was stopped before either. Recovery
                # needs that distinction: it is the difference between a state that
                # can be put back and one that was never taken.
                prepare = self.executor.run_maintenance(
                    # Reached only for the new operation generated above, under
                    # the pair lock. Repeated start returns before that point;
                    # recovery reuses the token but never authorizes recapture.
                    ['gpu-prepare', '--operation', self._require_operation(state),
                     '--fresh-capture'] + (['--capture-only'] if active else []),
                    timeout=120)
                prepared_output = (prepare.stdout or '') + (prepare.stderr or '')
                if prepare.returncode == 0 or self.ORIGINALS_RECORDED in prepared_output:
                    # rc 0 means gpu-prepare completed, so both records are on the
                    # target. A nonzero result carrying the recorded line means they
                    # were written and something after that failed.
                    state.prepared.append('gpu-prepare-recorded')
                    self.store.write(state)
                elif (prepare.returncode == self.MAINTENANCE_REFUSED
                      and prepared_output.strip()):
                    # The target REACHED A DECISION and refused, and its answer does
                    # not carry the recorded line: it stopped before writing either
                    # original-state record, so nothing on that node was changed.
                    # That is a different round from one whose outcome is unknown,
                    # and recovery treats it differently -- see
                    # `_require_recoverable_preparation`.
                    #
                    # ONLY the helper's own refusal status counts here. Any other
                    # nonzero answer -- a transport failure (255), an observation
                    # deadline (124), a non-root invocation (1), a crash -- leaves the
                    # outcome unknown, and one of them actively DESTROYS the evidence
                    # this branch reads: a run killed at the deadline never flushes
                    # its block-buffered stdout, so the recorded line is absent even
                    # when both records exist and telemetry is already paused.
                    # Accepting a bare `not in (0, 255)` here therefore refused to
                    # restore rounds that had really changed the node.
                    state.prepared.append('gpu-prepare-refused-before-capture')
                    self.store.write(state)
                if prepare.returncode != 0:
                    detail = prepared_output.strip().splitlines()
                    message = (detail[-1] if detail
                               else 'The maintenance route refused gpu-prepare.')
                    # The markers are already durable; record why it stopped.
                    state.failure = message
                    state.notes = 'Preparation did not complete.'
                    self.store.write(state)
                    if prepare.returncode == 255 or 'Connection' in message:
                        raise Refusal(
                            'Your exercise node is not reachable right now, which '
                            'is expected while it restarts. It stays reserved for '
                            'you; run status, then recover again in a few minutes.')
                    raise Refusal(message)
                state.prepared.append('gpu-prepare')
                self.store.write(state)
            argv = [operation]
            if active:
                assert own is not None
                argv += ['--job', own['id'], '--participant', self.assignment['participant_user'],
                         '--operation', state.operation, '--boot', boot_id]
            elif kind == 'efa' and own:
                argv += ['--job', own['id']]
            argv += ['--confirm', token]
            # The device mutation is journalled BEFORE it is issued, for the same
            # reason the preparation is: a marker written afterwards cannot record a
            # call that never returned, and its absence is then indistinguishable
            # from a call that was never made. Recovery reads this to tell a round
            # that faulted a device from one that stopped before it.
            state.prepared.append(f'{operation}-attempted')
            if active:
                assert state.active_fault is not None
                # Persist uncertainty before dispatch, including interruption
                # between a durable mutation marker and its ACK/result.
                state.active_fault['outcome'] = 'unknown-or-refused'
            self.store.write(state)
            try:
                if active:
                    assert state.active_fault is not None
                    def record(line):
                        assert state.active_fault is not None
                        try:
                            event = json.loads(line)
                        except ValueError:
                            return None
                        if not isinstance(event, dict):
                            return None
                        if event.get('action') == 'active-flr-authorized':
                            expected = {'operation': state.operation, 'mutation': operation,
                                        'instance_id': state.target_instance_id, 'boot_id': boot_id,
                                        'job_id': state.job_id,
                                        'participant': self.assignment['participant_user'],
                                        'bdf': state.active_fault['bdf']}
                            if any(event.get(key) != value for key, value in expected.items()):
                                raise Refusal('Active fault authorization differs from the assigned round.')
                            if state.active_fault['events']:
                                raise Refusal('Repeated active fault authorization; no second ACK.')
                            state.active_fault['collective_progress'] = self._active_progress(state)
                            state.active_fault['events'].append(event)
                            self.store.write(state)  # file + directory fsync BEFORE ACK
                            return 'ACK ' + hashlib.sha256(line.encode()).hexdigest()
                        if event.get('action') in ('mutation-start', 'mutation-returned'):
                            expected_actions = (['active-flr-authorized'] if event['action'] == 'mutation-start'
                                                else ['active-flr-authorized', 'mutation-start'])
                            if [item.get('action') for item in state.active_fault['events']] != expected_actions:
                                raise Refusal('Active mutation markers are out of order; no ACK.')
                            expected = {'operation': operation, 'instance_id': state.target_instance_id,
                                        'boot_id': boot_id, 'bdf': state.active_fault['bdf']}
                            if any(event.get(key) != value for key, value in expected.items()):
                                raise Refusal('Active mutation marker identity changed.')
                            if event['action'] == 'mutation-start':
                                state.active_fault['collective_progress'] = self._active_progress(state)
                            state.active_fault['events'].append(event)
                            self.store.write(state)
                            if event['action'] == 'mutation-start':
                                return 'ACK ' + hashlib.sha256(line.encode()).hexdigest()
                        return None
                    try:
                        result = self.executor.run_active_fault(argv, record)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        # Observation or evidence persistence can fail after the
                        # request/ACK reached the target. Keep the same round and
                        # markers; never infer that the device was not touched.
                        result = Completed(124 if isinstance(error, subprocess.TimeoutExpired) else 255,
                                           '', f'FLR observation failed; remote outcome is unknown: {error}\n')
                    actions = [event.get('action') for event in state.active_fault['events']]
                    confirmed = (result.returncode == 0 and actions ==
                                 ['active-flr-authorized', 'mutation-start', 'mutation-returned'])
                    state.active_fault['outcome'] = ('returned' if confirmed else 'unknown-or-refused')
                    if result.dispatched is False and not actions:
                        state.active_fault['outcome'] = 'not-issued'
                    self.store.write(state)
                    self._save_output('active-flr', (result.stdout or '') + (result.stderr or ''),
                                      phase='fault', round=state.round)
                    if not confirmed:
                        raise Refusal('Active FLR did not confirm completion; evidence retained. '
                                      'Do not repeat the fault. Run status, collect evidence, then recover.')
                else:
                    self._maintenance(argv)
            except Refusal as refusal:
                # Leave the drain in place: the node's device state is unknown
                # until someone looks. Recovery is a separate participant step.
                state.phase = 'fault-applied'
                state.notes = 'The device operation did not report success.'
                state.failure = str(refusal)
                self.store.write(state)
                raise

            state.phase = 'fault-applied'
            state.failure = None
            self.store.write(state)
            return Result(state, next_step='Use status, then collect a check.')

    def status(self):
        with self.store.exclusive_pair():
            state = self.store.read()
            record = self._replacement_record()
            line = self._node_line()
            next_step = (f'Replacement stage: {record["phase"]}. Run replace to continue.'
                         if record and record['phase'] != 'complete' else self._next_step(state))
            return Result(state,
                          node_state=self._node_field(line, 'State'),
                          node_reason=node_reason(line)[1],
                          next_step=next_step)

    def _next_step(self, state):
        return {
            'ready': 'Run start gpu or start efa when you are ready.',
            'preparing': 'Preparation did not finish; run recover to restore your node.',
            'fault-applied': 'Collect an allowed check and compare it with your baseline.',
            'investigating': 'Record your evidence, then run recover.',
            'recovering': 'Recovery is running; run status again.',
            'recovery-failed': 'Your node is still out of service; run recover again.',
            'replacement-required': 'Your node stays isolated. Run replace to initialize a fresh assigned instance.',
            'runtime-ready': 'Take a fresh allocation and verify Check 5 and storage.',
            'verified': 'This round is complete.',
        }[state.phase]

    def allowed_checks(self, state):
        table = self.assignment['allowed_checks']
        return [str(c) for c in table.get(state.kind or '', [])]

    @staticmethod
    def check_verdict(text):
        """The suite's own verdict for one check, read from its output.

        The suite prints exactly one [PASS]/[FAIL]/[WARN]/[SKIP] line per check
        through lib/common.sh check_pass/check_fail/check_warn/check_skip, and it
        emits no colour escapes when stdout is not a terminal (lib/common.sh
        `if [[ -t 1 ]]`), which is the case over the maintenance route. Verified
        against the collected logs on the target: check-6.log carries a bare
        '[FAIL] 6-efa-loopback: ...' and zero ESC bytes.
        """
        found = None
        for line in text.splitlines():
            match = re.search(r'\[(PASS|FAIL|WARN|SKIP)\]\s+(\S+)', line)
            if not match:
                continue
            verdict = match.group(1)
            # A FAIL anywhere in a check's output is the check's verdict.
            if verdict == 'FAIL':
                return 'FAIL'
            if found is None or (found == 'PASS' and verdict == 'WARN'):
                found = verdict
        return found

    def _collect_kernel(self, state):
        # Missing target/journal is evidence unavailability, never a recovery gate.
        try:
            boot = (state.active_fault or {}).get('boot_id') or state.boot_id_before_reboot or state.boot_id
            if not self.BOOT_ID.fullmatch(str(boot or '')):
                raise Refusal('No recorded fault boot UUID; kernel window unavailable.')
            result = self.executor.run_maintenance(['kernel-evidence', '--boot', boot], timeout=15)
            text = (result.stdout or '') + (result.stderr or '')
            text = f'# bounded kernel attempt returncode={result.returncode}; not full capture\n' + text
        except (OSError, subprocess.TimeoutExpired, Refusal) as error:
            text = f'# kernel evidence unavailable: {type(error).__name__}: {error}\n'
        saved = self._save_output('kernel', text, phase='fault', round=state.round)
        return text, saved

    def collect(self, check):
        with self.store.exclusive_pair():
            self._refresh_binding()
            state = self.store.read()
            if self._replacement_record().get('phase') not in (None, 'complete'):
                raise Refusal('Replacement is in progress; run replace to continue.')
            if state.phase == 'ready':
                raise Refusal('There is no exercise running yet. Start one first.')
            if check == 'kernel':
                text, saved = self._collect_kernel(state)
                return Result(state, output=text, saved_path=saved, next_step=self._next_step(state))
            if not re.fullmatch(r'[0-9]{1,2}', str(check)):
                raise Refusal('A check is one of the identifiers listed by status.')
            allowed = self.allowed_checks(state)
            if str(check) not in allowed:
                raise Refusal(f'Check {check} is not available for this exercise. '
                              f'Available: {", ".join(allowed)}.')
            phase = 'recovered' if state.phase in ('runtime-ready', 'verified') else 'fault'
            result = self.executor.run_maintenance(['collect', str(check)], timeout=900)
            text = (result.stdout or '') + (result.stderr or '')
            # The write is descriptor relative, as the participant, and never
            # truncates: an earlier capture keeps its name and this one becomes
            # -attempt-N. Fault-time evidence therefore survives a later re-check.
            saved = self._save_output(str(check), text, phase=phase, round=state.round)
            if state.phase == 'fault-applied':
                state.phase = 'investigating'
            if str(check) not in state.collected:
                state.collected.append(str(check))
            self.store.write(state)
            return Result(state, output=text, saved_path=saved,
                          next_step=self._next_step(state))

    def _reconcile_outstanding_reboot(self, state, notes):
        """Settle any reboot this round already requested, before anything else.

        Returns True when a reboot has completed (now or earlier), so the caller
        knows the staging bind mount must be re-established. Raises Refusal when the
        scheduler still holds a request, because waiting is the whole remedy, and
        also when the current boot could not be established at all: an unread or
        empty boot id is not evidence that the boot is unchanged, and reading it as
        unchanged let a completed reboot go unreconciled.

        This runs on EVERY attempt, whatever the fault kind and whatever the latest
        rebind did. The rejected control flow nested it inside `if needs_reboot:`,
        and `needs_reboot` came from the latest rebind outcome, so an EFA round that
        failed a rebind, requested a reboot, timed out observing it, and then
        succeeded on the next attempt's rebind skipped reconciliation entirely: it
        resumed with a scheduler reboot still outstanding, or skipped
        remount-staging after a boot it never recorded. That branch is reachable
        because the device helper's rebind succeeds when the device is already bound
        (device-fault.sh:145-147), which is exactly the state a completed reboot
        leaves behind.

        Ordering matters as much as the check: reconciliation happens BEFORE the
        rebind decision, so a round that already rebooted does not issue a device
        mutation on the strength of a stale `needs_reboot`.

        A completion already on the record is not permission to stop looking. It says
        the node came back once, which is why no second reboot is ever requested for
        it; it does not say the node is still on that boot or that the scheduler is
        no longer holding a request. A restoration-only retry after a failed
        `gpu-restore` or a failed check therefore re-reads both, because between
        attempts the node can restart again -- a facilitator's own `scontrol reboot`,
        or a request this exercise did not make -- and restoring against a changed
        boot would run against a staging mount that reboot removed. Refusing is the
        remedy there, not another reboot.
        """
        if state.reboot_completed:
            # Recorded as completed in an earlier attempt. Never repeated: the node
            # came back once, and a failure in the restoration that follows is not a
            # reason to cycle the hardware again. What the record does NOT settle is
            # the node's state right now, so the pending request and the current boot
            # are read again before this attempt restores anything.
            if self._reboot_is_pending():
                state.notes = '; '.join(
                    notes + ['A reboot is pending with the scheduler for a round '
                             'whose earlier reboot was already reconciled.'])
                self.store.write(state)
                raise Refusal(
                    'Your exercise node has another reboot pending with the scheduler, '
                    'after the restart this round already recorded. This attempt '
                    'cannot restore a node that is due to restart under it, so it '
                    'stays reserved for you, no reboot was requested and nothing was '
                    'changed; run recover again in a few minutes, or use '
                    'replace to continue on a fresh assigned instance.')
            current = self._current_boot_id()
            recorded = str(state.boot_id or '').strip()
            if not current or not self.BOOT_ID.fullmatch(recorded):
                missing = ('your exercise node did not report its current boot'
                           if not current else
                           'this round\'s recorded boot is not a boot id')
                state.notes = '; '.join(
                    notes + [f'A reconciled reboot could not be re-checked: '
                             f'{missing}.'])
                self.store.write(state)
                raise Refusal(
                    'Your exercise node restarted earlier in this round, and this '
                    f'attempt could not establish its current boot: {missing}. It '
                    'stays reserved for you, no reboot was requested and nothing was '
                    'restored; run recover again in a few minutes, or use '
                    'replace to continue on a fresh assigned instance.')
            if current != recorded:
                # It restarted AGAIN since the completion was recorded. Whatever did
                # that, this round did not, and its restoration would run against a
                # node in a state it has not observed -- including a staging bind
                # mount the new boot removed. No second reboot is requested.
                state.notes = '; '.join(
                    notes + ['The node restarted again after this round reconciled '
                             'its reboot.'])
                self.store.write(state)
                raise Refusal(
                    'Your exercise node has restarted again since this round recorded '
                    'its restart, so what this attempt would restore is not the state '
                    'the round observed. It stays reserved for you, no reboot was '
                    'requested and nothing was restored; use replace to continue '
                    'on a fresh assigned instance.')
            return True
        if not state.reboot_requests:
            # This round has no reboot of its own to settle, and that is where the
            # remaining gap was: returning here read 'we did not request one' as 'no
            # reboot concerns this node'. They are different facts. Between two
            # attempts the node can acquire a reboot nothing in this round asked for --
            # a facilitator's own `scontrol reboot`, a health-check remediation -- and
            # restoring under it would run against a node due to restart, including a
            # staging bind mount the new boot removes. A first attempt can equally
            # find a request already outstanding before it issues its own.
            #
            # So the same two observations the recorded-request branches make are made
            # here, before any mutation: is a request pending with the scheduler, and
            # is the node still on the boot this round recorded. Neither is read as a
            # reason to reboot -- this branch never requests one -- and an unreadable
            # answer refuses rather than being treated as 'unchanged', which is the
            # same direction `_reconcile_outstanding_reboot` already takes above.
            if self._reboot_is_pending():
                state.notes = '; '.join(
                    notes + ['A reboot is pending with the scheduler for a round '
                             'that never requested one.'])
                self.store.write(state)
                raise Refusal(
                    'Your exercise node has a reboot pending with the scheduler that '
                    'this round did not request. This attempt cannot restore a node '
                    'that is due to restart under it, so it stays reserved for you, '
                    'no reboot was requested and nothing was changed; run recover '
                    'again in a few minutes, or use replace.')
            current = self._current_boot_id()
            recorded = str(state.boot_id or '').strip()
            if not current or not self.BOOT_ID.fullmatch(recorded):
                missing = ('your exercise node did not report its current boot'
                           if not current else
                           'this round\'s recorded boot is not a boot id')
                state.notes = '; '.join(
                    notes + [f'A round with no reboot request could not establish '
                             f'the node\'s boot: {missing}.'])
                self.store.write(state)
                raise Refusal(
                    'This attempt could not establish which boot your exercise node '
                    f'is on: {missing}. What it would restore is therefore not '
                    'established to be the state this round observed, so your node '
                    'stays reserved, no reboot was requested and nothing was '
                    'restored; run recover again in a few minutes, or use '
                    'replace to continue on a fresh assigned instance.')
            if current != recorded:
                # The node restarted, and this round did not ask for that. This is an
                # OBSERVATION, not an unknown, so it is reconciled rather than refused:
                # what a restart invalidates for the steps below is the staging bind
                # mount, which a boot removes, so the caller is told a boot happened and
                # re-establishes it before restoring. Nothing here records
                # `reboot_completed`: this round requested no reboot, and claiming that
                # restart as its own completion would let a GPU round skip the reboot
                # its own recovery needs on the strength of somebody else's.
                notes.append('The node restarted for a reason this round did not '
                             'request; the staging mount is re-established before '
                             'restoring, and no reboot was requested here.')
                self.store.write(state)
                return True
            return False
        current = self._current_boot_id()
        recorded = str(state.boot_id or '').strip()
        usable_baseline = bool(self.BOOT_ID.fullmatch(recorded))
        if current and usable_baseline and current == recorded:
            # Read, valid, and the same boot: this attempt continues as if no reboot
            # had happened, which the pending check below confirms or refuses.
            pass
        elif current and usable_baseline:
            # A different boot, both sides usable. That is a completed restart, but a
            # completion is not RECORDED until the scheduler has also retired the
            # request: the flag it still holds means another restart is due, and a
            # round that recorded completion here could restore, qualify and resume
            # under it. The first wait applies the same condition (see the
            # `_reboot_is_pending` check after `_await_boot`); this branch used to
            # write the completion and return before reaching it, so the refusal there
            # was undone by the very next attempt.
            if self._reboot_is_pending():
                state.notes = '; '.join(
                    notes + ['The reboot requested earlier completed, but the '
                             'scheduler still holds the request.'])
                self.store.write(state)
                raise Refusal(
                    'Your exercise node has restarted, but the scheduler still holds '
                    'the reboot request for it, so this attempt cannot establish that '
                    'the restart it made is finished. It stays reserved for you, no '
                    'second reboot was requested and nothing was restored; run '
                    'recover again in a few minutes.')
            notes.append('The reboot requested earlier completed while this round '
                         'was not watching.')
            state.boot_id_before_reboot = state.boot_id
            state.boot_id = current
            state.reboot_completed = True
            state.boot_id_after_reboot = current
            if 'reboot' not in state.prepared:
                state.prepared.append('reboot')
            self.store.write(state)
            return True
        else:
            # The boot could not be established: the target did not answer the
            # boot-id request, it answered with nothing, this round never recorded
            # the boot it started on, or what it recorded is not a boot id. None of
            # those is "the boot is unchanged", and treating them as unchanged is what
            # let a reboot this round requested go unreconciled -- the node may have
            # come back already, in which case the staging bind mount is gone and
            # restoration would run against a tree that is not there.
            #
            # A recorded value that is not a UUID is refused rather than compared, for
            # the same reason `_await_boot` refuses it: it differs from every real
            # reading, so comparing it manufactures 'the boot changed' out of a value
            # no observation established. `start` will not record anything else, so
            # such a value came from a helper that did not check.
            #
            # Fail closed: the request stays on the record, the node keeps its
            # reservation, and the next attempt reads the boot again. This is
            # retryable when the target is reachable; an unusable recorded baseline
            # requires replacement, never reconstruction from the failed node.
            if not recorded:
                missing = 'this round has no recorded boot to compare against'
            elif not usable_baseline:
                missing = ('this round\'s recorded boot is not a boot id, so nothing '
                           'can be compared with it')
            else:
                missing = 'your exercise node did not report its current boot'
            state.notes = '; '.join(
                notes + [f'A reboot requested earlier could not be reconciled: '
                         f'{missing}.'])
            self.store.write(state)
            raise Refusal(
                'Your exercise node has a reboot already requested and this attempt '
                f'could not establish whether it completed: {missing}. It stays '
                'reserved for you, no second reboot was requested and nothing was '
                'changed; run recover again in a few minutes, or use '
                'replace to continue on a fresh assigned instance.')
        if self._reboot_is_pending():
            # The scheduler still holds the request. A second request would queue
            # another cycle. `_reboot_is_pending` answers 'yes' when the node
            # cannot be read, which is the conservative direction here.
            state.notes = '; '.join(
                notes + ['The reboot requested earlier is still pending with the '
                         'scheduler.'])
            self.store.write(state)
            raise Refusal(
                'Your exercise node has a reboot already requested and has not '
                'come back yet. It stays reserved for you and no second reboot '
                'was requested; run recover again in a few minutes.')
        # A request was recorded, the node is readable, its boot id has not changed
        # and the scheduler no longer holds the request. Nothing here establishes
        # that a reboot happened, so this attempt proceeds as if none had -- but the
        # request stays on the record, so the count still reflects it.
        notes.append('A reboot requested earlier is no longer pending and this '
                     'node did not change boot id; continuing recovery.')
        return False

    # The mutation markers `start` journals before it issues each device operation.
    # Their presence means the request was made and its outcome is unknown; their
    # absence means it was never made, because they are written before the call.
    #
    # INTENT and COMPLETION are separate facts and neither stands for the other. These
    # markers are intent: `<operation>-attempted` says the request was issued, not that
    # it returned or that the device changed. That is what makes them safe to act on --
    # an issued request with an unknown outcome needs recovery -- and it is also why
    # none of them may be read as evidence that a restoration already happened.
    MUTATION_MARKERS = ('gpu-remove-attempted', 'efa-unbind-attempted',
                        'gpu-flr-attempted', 'efa-flr-attempted')
    # The markers that mean this round called `gpu-prepare` on the target, whatever
    # that call then answered. `gpu-prepare-attempted` is journalled before the call and
    # records intent only; `gpu-prepare-recorded`, `gpu-prepare` and
    # `gpu-prepare-refused-before-capture` are written once an answer arrived and record
    # what that answer established. Any of them means the target may have paused
    # telemetry or disabled persistence, so the round is recoverable and only
    # `gpu-restore` can decide what to put back.
    PREPARATION_MARKERS = ('gpu-prepare-attempted', 'gpu-prepare-recorded',
                           'gpu-prepare', 'gpu-prepare-refused-before-capture')

    def _mutation_was_attempted(self, state):
        """Did this round ever issue a device mutation?

        Read from the round's own journal, and from nothing else. Phase cannot answer
        it: a round is left at 'preparing' both by a preparation that stopped before
        the mutation and by one that reached it, 'fault-applied' is written after the
        mutation returns so it is absent exactly when the answer matters most, and
        'investigating' is written by `collect` AND by both recovery entry points'
        refusal handlers, so it does not identify its own author.

        A record whose journal carries no mutation marker is therefore a round with no
        established device call, whatever phase it holds. For a record written before
        the markers existed that is an ambiguous state rather than a proven one, and it
        refuses: see `_require_recoverable_preparation`.
        """
        return any(marker in state.prepared for marker in self.MUTATION_MARKERS)

    def _call_was_issued(self, state):
        """Did this round call ANYTHING on the exercise node that could change it?

        The union of the two marker sets, each written before its call. The exercise's
        own drain is deliberately not counted: it is scheduler isolation this round
        reinstates rather than a change to the node that needs undoing, and counting it
        would make every round that got as far as reserving the node look mutated.
        """
        return (self._mutation_was_attempted(state)
                or any(marker in state.prepared
                       for marker in self.PREPARATION_MARKERS))

    def _require_recoverable_preparation(self, state):
        """Refuse to cycle hardware for a round that provably changed nothing.

        The case the review named first: `gpu-prepare` refuses an unrestorable telemetry
        original BEFORE writing either original-state record, `start` keeps that
        failed preparation and directs the participant to recover, and recovery then
        set needs_reboot unconditionally for a GPU round and issued `scontrol reboot`.
        The reboot changed a service whose state had just been judged unrestorable,
        and the `gpu-restore` that follows had no valid originals to undo it with. A
        refusal at the helper is not a refusal of the path.

        The second case is a round that never reached the helper at all. `start`
        persists `preparing` before it reserves the node, so a drain the scheduler
        refused (`start`, the drain branch) or a baseline collection interrupted by a
        dying coordinator (`_capture_baseline`, called before the preparation marker)
        both leave a round whose journal holds no preparation marker and no mutation
        marker. Nothing on that node was asked to change and no original state was
        captured, so a reboot, a remount or a runtime restoration would be this
        exercise modifying a node it had not touched -- and `gpu-restore` is skipped in
        exactly that state, so nothing would put the reboot's own effects back either.

        The distinction is what the round's own journal establishes, and it has five
        values rather than two:

          nothing was called
              neither marker set is present, and both are written BEFORE their call, so
              this round issued no preparation and no device operation. Keep the
              isolation and report it. This is the case that must not reboot, and it
              must stay refused across retries, which it does because the journal does
              not change.
          a device mutation was issued
              recovery is required whatever else failed, because the device is in an
              unknown state. `<operation>-attempted` is journalled before the call, so
              this marker records the REQUEST rather than its completion, and that is
              the point: an issued request whose answer never arrived is exactly the
              round that needs restoring.
          the target ANSWERED with a refusal that predates its own recording
              `gpu-prepare-refused-before-capture`, written only when the target
              replied and its reply does not carry the line it prints once both
              records exist. Nothing on that node was changed, so there is nothing to
              put back: this is the other case that must not reboot.
          a record this version did not write, holding no journal
              `fault-applied` and `investigating` are not attributable to `start` or
              `collect`: both recovery entry points write `investigating` from their
              own refusal handlers, so a phase does not identify its author, and
              recovery overwrites it with 'recovering' before this gate is even
              reached. Such a record is recognised by what it durably lacks -- an
              operation token, which `start` has written before any side effect since
              the records became operation-bound -- rather than by a display value. It
              establishes neither that a device call happened nor that none did, and an
              ambiguous state is refused rather than upgraded. The cost is real and it
              is the fail-closed direction: a node whose device is genuinely faulted
              stays isolated for participant replacement. The alternative was
              rebooting on a phase, and such a round captured no original state either,
              so nothing could put the reboot's own effects back.
          anything else -- rc 0, a refusal after recording, a transport failure, a
          dead coordinator, an unanswered call
              the outcome is unknown or the originals exist. Recovery proceeds, which
              is what keeps the interrupted-preparation window (R5) safe: telemetry
              may already be paused and only `gpu-restore` can put it back.

        Fail-closed here means proceeding when a call was made and its outcome is
        unknown, and stopping when the journal establishes that no call was made, that
        the target refused before recording, or that the record cannot establish either.
        The harm being prevented is cycling a node this round did not modify; the harm
        on the other side is leaving a modified one unrestored, and the JOURNAL is what
        tells them apart. A phase never does.
        """
        if not self._call_was_issued(state):
            if not (state.operation or '').strip():
                # No journalled call AND no operation token: a record this version did
                # not write, so the absence of markers does not establish the absence
                # of calls. Ambiguous, and therefore refused with its isolation intact.
                # The discriminator is the missing token rather than the phase, which
                # this method's caller has already replaced with 'recovering'.
                self._fail_recovery(state, [
                    'This round\'s record carries no operation identifier and no '
                    'journalled preparation or device operation, so it does not '
                    'establish what was done on the exercise node. A phase cannot '
                    'establish it either: both recovery entry points write one after a '
                    'refusal. Nothing was changed and the node keeps its isolation.'])
                raise Refusal(
                    'Your round\'s record does not establish what was done on the '
                    'exercise node: it names no preparation and no device operation, '
                    'and it carries no operation identifier that the node\'s '
                    'original-state records could be matched against. Recovery will '
                    'not reboot or restore a node on that basis, and no original state '
                    'was recorded that could be put back afterwards, so the node keeps '
                    'its reservation exactly as it is. Use replace to continue on '
                    'a fresh assigned instance.')
            self._fail_recovery(state, [
                'This round issued no preparation and no device operation on the '
                'exercise node -- both are journalled before they are called -- so it '
                'captured no original state and there is nothing to put back. A '
                'reboot, a remount or a runtime restoration would change the node '
                'rather than restore it.'])
            raise Refusal(
                'Your round stopped before anything on the exercise node was changed: '
                'it never asked the node to pause telemetry and never issued a device '
                'operation, so no original state was recorded. Recovery will not '
                'reboot or restore a node it did not modify, so the node keeps its '
                'reservation exactly as it is. Use replace to continue on a fresh assigned instance.')
        restore_originals = self._require_coherent_restoration_obligations(state)
        if self._mutation_was_attempted(state):
            return restore_originals
        if 'gpu-prepare-refused-before-capture' not in state.prepared:
            return restore_originals
        if 'gpu-prepare-recorded' in state.prepared:
            # A later attempt of the same round did record originals. They exist, so
            # they are restorable and the round is recoverable.
            return restore_originals
        self._fail_recovery(state, [
            'The exercise node refused preparation before this round recorded its '
            'original state, and no device operation was issued, so there is nothing '
            'to put back and a reboot would change the node rather than restore it.'])
        raise Refusal(
            'Your round stopped before anything on the exercise node was changed: '
            'the node refused preparation before its original state was recorded, '
            'and no device operation was issued. Recovery will not reboot or restore '
            'a node it did not modify, so the node keeps its reservation exactly as '
            'it is. Use replace to continue on a fresh assigned instance.')

    def _require_coherent_restoration_obligations(self, state):
        """Validate journal/token before changes; return the restoration obligation.

        Preparation intent alone needs restoration: an unanswered helper may have
        paused the node. A recorded answer and a successful preparation carry the
        same obligation. A refusal before capture cannot authorize later removal
        unless a subsequent capture was recorded. Display phase proves none of these.
        """
        markers = set(state.prepared)
        preparation = bool(markers.intersection(self.PREPARATION_MARKERS))
        if state.active_fault:
            self._require_operation(state)
            if (state.active_fault.get('mutation') != state.kind + '-flr'
                    or not state.job_id or not preparation):
                raise Refusal('Active FLR restoration provenance is incomplete; use replace.')
            if (markers.intersection({'gpu-remove-attempted', 'efa-unbind-attempted'})
                    or (('efa' if state.kind == 'gpu' else 'gpu') + '-flr-attempted' in markers)
                    or (self._mutation_was_attempted(state)
                        and (state.kind + '-flr-attempted' not in markers
                             or 'gpu-prepare-recorded' not in markers))):
                raise Refusal('Active FLR mutation provenance is inconsistent; use replace.')
            return True
        gpu_mutation = 'gpu-remove-attempted' in markers
        efa_mutation = 'efa-unbind-attempted' in markers
        problem = None
        if ((state.kind == 'gpu' and efa_mutation)
                or (state.kind == 'efa' and (preparation or gpu_mutation))):
            problem = 'The journal names preparation or mutation for a different fault kind.'
        elif state.kind == 'gpu':
            try:
                self._require_operation(state)
            except Refusal as error:
                problem = str(error)
            if gpu_mutation and not preparation:
                problem = 'The GPU mutation has no journalled preparation to restore.'
            if ('gpu-prepare-refused-before-capture' in markers
                    and 'gpu-prepare-recorded' not in markers and gpu_mutation):
                problem = 'The GPU mutation follows a refused capture with no recorded originals.'
        if problem:
            self._fail_recovery(state, [problem])
            raise Refusal(problem + ' Recovery will not reboot or restore on this '
                          'record. The node stays isolated; replacement is required, '
                          'not instructor restoration. Use replace to continue.')
        return state.kind == 'gpu' and preparation

    def _recover(self, state, recovered_by):
        if self._other_assignment_holding_target():
            raise Refusal('Another table holds this node; recovery cannot change it.')
        state.phase = 'recovering'
        state.failure = None
        self.store.write(state)
        notes = []
        # Every recovery attempt revalidates that this node is still the one this
        # round owns and is still ours to change, before it changes anything. The
        # earlier code guarded this behind `if not state.reboot_completed`, so a
        # restoration-only retry went from an instance-id comparison straight to
        # remount-staging and restore-runtime. A node returned to service between
        # attempts, or now carrying a foreign job or a foreign drain, was modified
        # before anything checked. Same instance is not the same permission.
        self._require_recoverable_target(state)
        # And that this round actually has something to recover. A round refused
        # before it recorded an original and before it touched a device must not be
        # rebooted: see _require_recoverable_preparation. This is checked before the
        # reboot decision below and before any restoration call, and after the target
        # validation above, so the node's own drain is reinstated either way.
        restore_originals = self._require_recoverable_preparation(state)
        if (state.active_fault and state.reboot_requests and not state.reboot_completed
                and self._current_boot_id() == state.boot_id):
            self._active_reboot_escalation(state)
        if 'efa-rebind-failed-no-originals' in state.prepared:
            self._require_replacement(state, state.notes or
                                      'EFA rebind failed without restorable originals.')
        # Any outstanding reboot is settled FIRST, for both fault kinds, before the
        # rebind decision and independent of what the last rebind returned.
        rebooted = self._reconcile_outstanding_reboot(state, notes)
        # Completed requests already recorded the reconciled boot. Only an external
        # restart (no local request) needs a separate current baseline here.
        recovery_boot = (self._current_boot_id()
                         if rebooted and not state.reboot_completed else state.boot_id)
        if not recovery_boot:
            raise Refusal('The recovery boot is unreadable; nothing will be restored.')
        needs_reboot = False

        if not state.reboot_completed:
            if state.kind == 'efa' and not state.active_fault:
                # The rebind's own outcome is judged separately from the checks
                # above. The device helper refuses a rebind while a job is
                # present, and that refusal means 'end the allocation', not
                # 'reboot this node'.
                inventory = self._inspect()
                token = (inventory.get('confirmation') or {}).get('efa-rebind')
                if not token:
                    raise Refusal('The maintenance route returned no confirmation '
                                  'for the rebind; nothing was changed.')
                rebind = self.executor.run_maintenance(['efa-rebind', '--confirm', token])
                if rebind.returncode != 0:
                    detail = (rebind.stderr or rebind.stdout or '').strip().splitlines()
                    message = detail[-1] if detail else 'The rebind did not succeed.'
                    if self._is_safety_refusal(message):
                        # A safety or identity refusal is not a qualified rebind
                        # failure, so it must not become a reboot request.
                        self._fail_recovery(state, notes + [message])
                        raise Refusal(
                            'Recovery stopped before changing anything: '
                            f'{message} Your node stays out of service until this '
                            'is resolved.')
                    # EFA start captured no reboot-affected originals. A failed
                    # rebind cannot authorize rebooting and leaving those changes.
                    state.prepared.append('efa-rebind-failed-no-originals')
                    self._require_replacement(state, message)
            else:
                needs_reboot = True

        if needs_reboot:
            # The target is revalidated at the top of every attempt; re-read the
            # drain reason here because the rebind above can have taken minutes.
            self._require_recoverable_target(state)
            line = self._node_line()
            reason = self._node_field(line, 'Reason')
            if reason and not self._owns_recovery_reason(line, state):
                raise Refusal(f'Your exercise node now carries a different drain '
                              f'reason ({reason}); recovery stopped.')
            # Any earlier request was already reconciled at the top of this attempt,
            # unconditionally, so reaching here means no reboot of this round's is
            # outstanding. The reconciliation used to live inside this branch, which
            # made it conditional on the latest rebind having failed.

        if needs_reboot:
            # Basic scontrol reboot only. PCS documents replacement after reboot
            # with nextstate=DOWN; the historical incident used EC2 reboot, not a
            # trial of that option. Early RESUME clears the drain before checks.
            # The request is persisted BEFORE it is issued, so a process that dies
            # between the two still leaves a record that reconciles above rather
            # than a clean-looking state that reboots again.
            self._require_current_recovery_observations(recovery_boot, 'reboot', state)
            state.reboot_requests = int(state.reboot_requests or 0) + 1
            state.reboot_requested_at = self.clock()
            self.store.write(state)
            request = self._scontrol('reboot', f'reason={self.assignment["drain_reason"]}',
                                     self.assignment['target_node'])
            if request.returncode != 0:
                raise Refusal(f'Could not request the exercise reboot: '
                              f'{request.stderr.strip()}')
            new_boot = self._await_boot(state.boot_id)
            if new_boot is None:
                if state.active_fault and self._current_boot_id() == state.boot_id:
                    self._active_reboot_escalation(state)
                state.notes = '; '.join(notes + ['The node has not returned yet.'])
                self.store.write(state)
                raise Refusal('Your exercise node has not returned yet. It stays '
                              'reserved; run recover again in a few minutes.')
            if self._reboot_is_pending():
                # A changed boot id and an outstanding request together are not a
                # completed reboot. The scheduler's own flags say the request it holds
                # has not been retired, so this attempt cannot record a completion:
                # doing so would let the round restore, qualify and resume while the
                # node is still due to restart under it. `_reboot_is_pending` answers
                # 'yes' when the node cannot be read, which is the conservative
                # direction. The next attempt reconciles through
                # `_reconcile_outstanding_reboot`, which reads the current boot against
                # the recorded one; the request stays on the record either way.
                state.notes = '; '.join(
                    notes + ['The reboot completed but the scheduler still holds the '
                             'request.'])
                self.store.write(state)
                raise Refusal(
                    'Your exercise node reported a restart, but the scheduler still '
                    'holds the reboot request for it, so this attempt cannot establish '
                    'that the restart it made is finished. It stays reserved for you, '
                    'no second reboot was requested and nothing was restored; run '
                    'recover again in a few minutes.')
            # Record the boot the round STARTED on before overwriting it, so the
            # evidence still shows that a reboot happened. Assigning both
            # state.boot_id and state.boot_id_after_reboot the same new value made
            # the two fields identical in the state file, which reads as "no
            # reboot occurred" to anyone auditing the round afterwards -- exactly
            # the wrong conclusion, since the reboot is the recovery mechanism.
            # Observed on gpu-g7-1 round 7: both fields held
            # 00000000-0000-4000-8000-000000000001 even though the node's
            # BootTime and `last reboot` both confirmed it really rebooted
            # (participant-revision/runs/rework2-verify-target-recovered/output.log).
            state.boot_id_before_reboot = state.boot_id
            state.boot_id = new_boot
            recovery_boot = new_boot
            # Persist the completed boot before anything fallible follows it, so
            # a retry resumes restoration instead of rebooting again.
            state.reboot_completed = True
            state.boot_id_after_reboot = new_boot
            state.prepared.append('reboot')
            self.store.write(state)
            rebooted = True

        # Identity again after the reboot, from the DMI field rather than the
        # device inventory. The device may legitimately still be absent at this
        # point on an EFA round that fell back to a reboot, and a device-inventory
        # refusal here would stop the restoration that fixes it.
        if self._instance_id() != state.target_instance_id:
            raise Refusal('The recovered node reports a different instance identity; '
                          'use replace to validate and initialize the assigned successor.')
        if rebooted:
            # Validate the provisioned staging contract before runtime restore:
            # legacy NVMe needs its bind re-established; root-resident successors
            # retain their pinned images and must never bind NVMe over them.
            # Root-directory successors verify both large pinned images here.
            # A cold EBS read is not a 120-second mount operation.
            self._maintenance(['remount-staging'], timeout=900)
        self._maintenance(['restore-runtime', '--participant',
                           self.assignment['participant_user']], timeout=900)
        if restore_originals:
            # Put the recorded persistence mode and telemetry owner back before
            # the node returns to service. A failure here is a required-step
            # failure: the node keeps its drain. `gpu-prepare-attempted` is now
            # written before gpu-prepare is called, so the only window in which
            # neither marker exists is before any side effect has been applied.
            #
            # maintenance.gpu_restore now refuses when there is no usable record for
            # THIS operation -- absent, empty, unparseable, naming another operation
            # or none at all, or holding a value it cannot put back -- and returns
            # nonzero, which arrives here as a required-step failure. A missing
            # original-state record therefore stops the round instead of passing for
            # a restore that never happened. It also puts a deliberately inactive
            # telemetry service back to inactive and confirms the state it observed,
            # so a reboot that started the service does not leave the node changed
            # and an unsettled observation does not pass as a restoration.
            restored = self.executor.run_maintenance(
                ['gpu-restore', '--operation', self._require_operation(state)],
                timeout=180)
            if restored.returncode != 0:
                detail = (restored.stderr or restored.stdout or '').strip().splitlines()
                message = detail[-1] if detail else \
                    'The recorded GPU state was not fully restored.'
                self._fail_recovery(state, notes + [message])
                raise Refusal(
                    f'Your node was not fully restored: {message} It stays out of '
                    'service, with your evidence kept. Run recover again, or use '
                    'replace to continue on a fresh assigned instance.')
            if self._telemetry_not_restored(restored):
                message = ('Telemetry did not return to active after restoration.')
                self._fail_recovery(state, notes + [message])
                raise Refusal(
                    'Your node was restored except for its telemetry service, so '
                    'it stays out of service with your evidence kept. Use '
                    'replace to continue on a fresh assigned instance.')

        failures = []
        warnings = []
        unsupported_warning = False
        for check in self.allowed_checks(state):
            outcome = self.executor.run_maintenance(['collect', check], timeout=900)
            text = (outcome.stdout or '') + (outcome.stderr or '')
            saved = self._save_output(check, text, phase='recovered',
                                      round=state.round)
            verdict = self.check_verdict(text)
            # The verdict recorded per round and phase, so the fault-time
            # observation is never replaced by this one.
            state.check_results[f'round-{state.round}/recovered/check-{check}'] = {
                'verdict': verdict or 'UNREADABLE',
                'returncode': outcome.returncode,
                'log': str(saved),
                # Recorded for the same reason as the baseline's: the round's
                # record should show which condition was accepted or refused, not
                # only that a WARN happened.
                'warnings': self.warning_details(text),
            }
            # Qualification is fail closed in the strict sense: a check counts as
            # passed only if it produced a complete PASS AND the transport that
            # carried it returned 0. Baseline-matched WARN acceptance is disabled.
            # Everything else -- FAIL, SKIP, an unreadable verdict, a nonzero
            # return code under any verdict -- keeps the node drained.
            #
            # The suite's own verdict is read first, because it is the more
            # specific answer: gpu-healthcheck.sh propagates a check's nonzero
            # status for a FAIL (run_single_check returns check_exit,
            # participant-revision/runs/rework-read-suite-exitcodes3/output.log:33),
            # so a FAIL arrives with rc=1 and must be reported as a failed check
            # rather than as a transport problem.
            if verdict == 'FAIL':
                failures.append(f'Check {check} did not pass after recovery '
                                f'(FAIL).')
            elif verdict == 'SKIP':
                # A skipped check is an absent qualification, not a lenient pass.
                # checks/6-efa-loopback.sh skips when fi_pingpong is unavailable,
                # which is exactly the case where the loopback evidence this
                # round needs was never produced.
                failures.append(
                    f'Check {check} was skipped after recovery, so this round '
                    f'has no result for it.')
            elif verdict is None:
                failures.append(f'Check {check} produced no readable verdict; '
                                f'treating it as not passed.')
            elif outcome.returncode != 0:
                # A PASS or WARN carried by a nonzero return code did not complete
                # on the far side, whatever text arrived. 124 is this helper's own
                # external observation deadline (Executor._run), which is not
                # evidence the remote check ended at all.
                failures.append(
                    f'Check {check} reported {verdict} but did not complete: the '
                    f'maintenance route returned {outcome.returncode}.')
            elif verdict == 'WARN':
                allowed, detail = self._baseline_allows_warning(state, check, text)
                if allowed:
                    warnings.append(f'Check {check} returned WARN after recovery; '
                                    f'{detail}')
                else:
                    unsupported_warning = True
                    failures.append(
                        f'Check {check} returned WARN after recovery and '
                        f'{detail}')
            else:
                # A PASS. The label is the suite's own and it is not being
                # second-guessed here; what is checked is whether the run behind it
                # observed the condition this round's baseline recorded. The pinned
                # check 6 can print PASS after its statistics collection was skipped
                # entirely, so 'PASS' is not by itself evidence that a recorded
                # counter condition was re-read. Explicit skipped observations
                # refuse regardless of baseline; complete clean PASS is unaffected.
                complete, detail = self._observation_is_complete(state, check, text)
                if not complete:
                    failures.append(
                        f'Check {check} reported PASS after recovery but '
                        f'{detail}')

        if failures:
            if unsupported_warning:
                self._require_replacement(state, '; '.join(notes + failures + warnings))
            self._fail_recovery(state, notes + failures + warnings)
            raise Refusal(
                'Your node was not returned to service because a required check '
                'did not pass after recovery: ' + ' '.join(failures)
                + ' Your evidence is kept and the node stays drained. Run recover '
                'again for a complete observation. If recovery cannot qualify '
                'the node, replacement is required, not instructor restoration.')

        self._require_current_recovery_observations(recovery_boot, 'RESUME', state)
        resumed = self._scontrol('update', f'NodeName={self.assignment["target_node"]}',
                                 'State=RESUME')
        if resumed.returncode != 0:
            raise Refusal(f'Could not return your node to service: '
                          f'{resumed.stderr.strip()}')

        state.phase = 'runtime-ready'
        state.recovered_by = recovered_by
        state.recovered_at = self.clock()
        state.failure = None
        state.notes = '; '.join(notes + warnings)
        self.store.write(state)
        return state, notes + warnings

    # Refusals from the device helper and the maintenance route that mean the
    # request must stop, not escalate to a reboot. Each phrase is quoted from the
    # shipped helper: device-fault.sh requires a drain and refuses foreign jobs
    # and changed identities before it touches a device.
    SAFETY_REFUSALS = (
        'unapproved job',
        'no longer present on the target node',
        'end the faulted allocation',
        'end the allocation',
        'drain the target node',
        'drain reason is not this aim344 exercise',
        'not the provisioned fault target',
        'uuid/bdf mapping changed',
        'pci vendor changed',
        'pci device id changed',
        'management pci identity changed',
        'invalid instance identity',
        'unexpected driver binding',
        'confirmation does not match',
        'still has a compute process',
    )

    @classmethod
    def _is_safety_refusal(cls, message):
        lowered = (message or '').lower()
        return any(phrase in lowered for phrase in cls.SAFETY_REFUSALS)

    @staticmethod
    def _telemetry_not_restored(result):
        """gpu_restore can report an incomplete telemetry restoration on rc 0.

        Three shapes, all from maintenance.gpu_restore: a recorded-active service
        that did not come back, a recorded-inactive service that does not read
        inactive, and a unit systemd no longer reports as loaded at all -- the last
        being the case where the state word `inactive` is the same word an absent or
        masked unit reports. All three print `preserve this state`, and all three
        matter to the caller even when the return code is 0, so the text is read as
        well as the code.
        """
        text = ((result.stdout or '') + (result.stderr or '')).lower()
        return ('did not return to active' in text
                or 'recorded it inactive' in text
                or 'is not a loaded unit' in text)

    def _active_reboot_escalation(self, state):
        """One exact-instance EC2 fallback, then participant-owned replacement.

        An API acknowledgement is only a request, never a successful recovery.
        The ordinary recovery path still requires a fresh boot and suite PASS.
        """
        assert state.active_fault is not None
        wait = float(self.assignment.get('active_reboot_grace_seconds', 120))
        if not 30 <= wait <= 900:
            raise Refusal('Active reboot grace must be between 30 and 900 seconds.')
        since = state.active_fault.get('external_reboot_at', state.reboot_requested_at)
        if since is None or self.clock() - since < wait:
            raise Refusal('The requested reboot is still within its observation window; '
                          'keep evidence and run recover again.')
        if state.active_fault.get('external_reboot_attempted'):
            self._require_replacement(state, 'The active-fault reboot did not return within its window.')
        self._require_recoverable_target(state)
        node = self._replacement_node()
        if (node['InstanceId'] != state.target_instance_id
                or node['NodeAddr'] != self.assignment['target_host']):
            raise Refusal('Active recovery scheduler binding changed; no external reboot.')
        self._replacement_capacity()
        instance = self._replacement_instance(state.target_instance_id,
                                               host=self.assignment['target_host'])
        if instance.get('State', {}).get('Name') != 'running':
            raise Refusal('The active recovery instance is not running; use replace.')
        state.active_fault['external_reboot_attempted'] = True
        state.active_fault['external_reboot_at'] = self.clock()
        self.store.write(state)  # persist request intent before API; never blind retry
        self._cloud('ec2', 'reboot-instances', {'InstanceIds': [state.target_instance_id]})
        # Acknowledged dispatch is distinct from pre-call intent and observed boot
        # completion. Persist before the fallible readback; a lost ACK leaves only
        # intent, which already prevents a duplicate request on later attempts.
        state.active_fault['external_reboot_requested'] = True
        state.active_fault['external_reboot_requested_at'] = self.clock()
        self.store.write(state)
        self._replacement_instance(state.target_instance_id, host=self.assignment['target_host'])
        raise Refusal('The exact assigned instance was requested to reboot through EC2; '
                      'completion is not established. Run recover again, then replace if required.')

    def _require_recoverable_target(self, state):
        """Identity, drain and job ownership, before any recovery mutation.

        A stale assignment or a foreign job must stop recovery rather than queue
        a reboot. The previous code path reached scontrol reboot without ever
        consulting the queue or the current instance identity.

        Identity is read with instance-id, not inspect: a GPU round has removed
        the provisioned PCI function, so inspect necessarily refuses and using it
        here blocked the very reboot that restores the device.
        """
        if self._instance_id() != state.target_instance_id:
            raise Refusal('Your exercise node no longer reports the instance this '
                          'round started on; nothing was changed. Use replace '
                          'to validate and initialize the assigned successor.')
        line = self._node_line()
        node_state = self._node_field(line, 'State')
        reason = self._node_field(line, 'Reason')
        ours = self.assignment['drain_reason']
        if not node_state:
            raise Refusal('The scheduler returned no State; recovery will not change the node.')
        if 'DRAIN' not in node_state and 'DOWN' not in node_state:
            # The exercise's own drain is what keeps other work off the node while
            # it is recovered. Reinstate it rather than mutating an in-service node.
            drained = self._scontrol(
                'update', f'NodeName={self.assignment["target_node"]}',
                'State=DRAIN', f'Reason={ours}')
            if drained.returncode != 0:
                raise Refusal('Your exercise node is in service and could not be '
                              'reserved for recovery; nothing was changed.')
            if 'drain' not in state.prepared:
                state.prepared.append('drain')
                self.store.write(state)
        elif reason and not self._owns_recovery_reason(line, state):
            raise Refusal(f'Your exercise node is drained for a different reason '
                          f'({reason}); recovery stopped and nothing was changed.')
        jobs = self._jobs_on_target()
        # _own_job refuses any job that is not this participant's exercise job.
        own = self._own_job(jobs)
        if state.active_fault and own:
            if own['id'] != state.job_id:
                raise Refusal('A different job is now using the fault target; nothing was changed.')
            if own['state'] != 'COMPLETING':
                raise Refusal('The faulted allocation is still active. Keep collecting evidence; '
                              'wait for its bounded walltime/exit, then run recover. No job was killed.')

    def _require_replacement(self, state, reason):
        """Persist isolation's outcome, not a claim that replacement was performed."""
        self._fail_recovery(state, [reason])
        state.phase = 'replacement-required'
        self.store.write(state)
        raise Refusal(reason + ' The node stays isolated. Replacement is required; '
                      'no fallback reboot or instructor restoration is allowed. '
                      'Run replace to continue on a fresh assigned instance.')

    def _fail_recovery(self, state, notes):
        """Keep the node drained and the round out of service, retryably."""
        state.phase = 'recovery-failed'
        state.notes = '; '.join(n for n in notes if n)
        state.failure = state.notes
        state.recovered_by = None
        state.recovered_at = None
        self.store.write(state)

    def _await_boot(self, old_boot):
        """Wait until the node reports a boot DIFFERENT from `old_boot`, or give up.

        The comparison is only meaningful when both sides are observations, so both
        sides are required to look like one.

        The recorded side is what the defect turned on: against `''` every nonempty
        reading differs, so the boot the node never left was read as a change, the
        caller recorded a completed reboot, and the round could restore, qualify and
        resume with its request still outstanding. Emptiness is not the only unusable
        baseline, though. Any value that is not a boot id differs from every real
        reading, so it produces the same false 'the boot changed' on the first
        comparison -- which is why the shape is required here and not merely
        non-emptiness. `start` refuses to record anything else, so a stored value of
        another shape came from a helper that did not check, and it is refused rather
        than compared.

        What that costs is stated rather than hidden: a round whose record holds an
        unusable boot cannot have its reboot confirmed by this route at all, so it keeps
        its drain and its evidence for participant replacement. That is the fail-closed
        direction, and it is not a claim about the hardware.
        """
        if not self.BOOT_ID.fullmatch(str(old_boot or '').strip()):
            # Fail closed: the reboot has been requested and stays on the record, and
            # the next attempt reconciles it through `_reconcile_outstanding_reboot`,
            # which refuses an unusable comparison too.
            raise Refusal(
                'Your exercise node was asked to restart, but this round has no usable '
                'record of the boot it started on, so nothing here can establish that '
                'the restart completed. The request stays on the record, your node '
                'stays reserved and nothing was restored; use replace to continue '
                'on a fresh assigned instance.')
        deadline = self.clock() + float(self.assignment.get('reboot_wait_seconds', 900))
        while self.clock() < deadline:
            result = self.executor.run_maintenance(['boot-id'], timeout=30)
            current = (result.stdout or '').strip()
            if (result.returncode == 0 and self.BOOT_ID.fullmatch(current)
                    and current != old_boot):
                return current
            if self.clock() >= deadline:
                break
            time.sleep(float(self.assignment.get('reboot_poll_seconds', 10)))
        return None

    def _current_boot_id(self):
        """The node's boot id if it answers with one, or None.

        Unlike `_boot_id` this never raises: it is used to ask 'did the reboot we
        requested already finish?', where an unreachable node is one of the expected
        answers rather than an error. An answer that is not a boot id is None for the
        same reason it is a refusal there -- it is not an observation of the boot -- and
        the caller distinguishes None from 'the boot is unchanged'.
        """
        result = self.executor.run_maintenance(['boot-id'], timeout=30)
        if result.returncode != 0:
            return None
        current = (result.stdout or '').strip()
        return current if self.BOOT_ID.fullmatch(current) else None

    # The reboot flags this Slurm build emits in a node's State field. Read out of
    # the deployed libraries rather than assumed from the documentation: 25.05.9
    # ships REBOOT_REQUESTED and REBOOT_ISSUED in libslurm.so.43.0.0 and
    # libslurmfull.so, and no other REBOOT_* token
    # (participant-revision/runs/rework4-read-slurm-reboot-tokens/output.log).
    # scontrol's own help names `cancel_reboot`, which is the operation these flags
    # describe.
    REBOOT_PENDING_FLAGS = ('REBOOT_REQUESTED', 'REBOOT_ISSUED')

    def _require_current_recovery_observations(self, expected_boot, action, state=None):
        """Fresh observations at the dependent decision, not an external-event lock."""
        current = self._current_boot_id()
        if (not current or not isinstance(expected_boot, str)
                or not self.BOOT_ID.fullmatch(expected_boot) or current != expected_boot):
            raise Refusal(f'The current boot is unreadable or changed during recovery; '
                          f'not issuing {action}. The node stays isolated.')
        line = self._node_line()
        node_state = self._node_field(line, 'State')
        reason = self._node_field(line, 'Reason')
        if not node_state or any(flag in node_state for flag in self.REBOOT_PENDING_FLAGS):
            raise Refusal(f'The scheduler State is missing or a reboot is pending; '
                          f'not issuing {action}. The node stays isolated.')
        if ('DRAIN' not in node_state and 'DOWN' not in node_state):
            raise Refusal(f'The node lost its recovery isolation; not issuing {action}.')
        if reason and not self._owns_recovery_reason(line, state):
            raise Refusal(f'The drain reason changed to {reason}; not issuing {action}.')
        if action == 'RESUME' and self._jobs_on_target():
            raise Refusal('Job cleanup is not complete; the node cannot be resumed.')

    def _reboot_is_pending(self):
        """Does the scheduler still hold a reboot request for the exercise node?

        Conservative on purpose. If the node cannot be read, the answer is 'yes,
        assume a request is outstanding', because the failure mode this protects
        against is issuing a second reboot.
        """
        try:
            line = self._node_line()
        except Refusal:
            return True
        state = self._node_field(line, 'State')
        return not state or any(flag in state for flag in self.REBOOT_PENDING_FLAGS)

    def recover(self):
        # Recovery mutates the node, so it takes the pair lock too.
        with self.store.exclusive_pair():
            self._refresh_binding()
            if self._replacement_record().get('phase') not in (None, 'complete'):
                raise Refusal('Replacement is in progress; run replace to continue.')
            state = self.store.read()
            if state.phase == 'ready':
                raise Refusal('There is no exercise to recover yet.')
            if state.phase in ('runtime-ready', 'verified'):
                return Result(state, already_recovered=True,
                              next_step='Take a fresh allocation and verify '
                                        'Check 5 and storage.')
            try:
                if state.active_fault:
                    try:
                        self._collect_kernel(state)
                    except (OSError, Refusal):
                        pass  # participant output failure cannot require instructor rescue
                state, notes = self._recover(state, 'participant')
            except Refusal:
                # A recovery that stopped part way must stay retryable. Leaving
                # the record at 'recovering' would refuse the participant's next
                # attempt and require a facilitator, which is what this exercise
                # exists to avoid. The node keeps its drain either way, and a
                # required-step failure keeps the distinct 'recovery-failed'
                # phase rather than being downgraded to 'investigating'.
                stuck = self.store.read()
                if stuck.phase == 'recovering':
                    stuck.phase = ('replacement-required'
                                   if self.assignment.get('replacement', {}).get('enabled')
                                   else 'investigating')
                    self.store.write(stuck)
                raise
            return Result(state, next_step='Take a fresh allocation and verify '
                                           'Check 5 and storage.')

    def expire(self):
        """Deadline safeguard. Recorded separately from a participant recovery."""
        with self.store.exclusive_pair():
            self._refresh_binding()
            if self._replacement_record().get('phase') not in (None, 'complete'):
                raise Refusal('Replacement is in progress; run replace to continue.')
            state = self.store.read()
            if state.phase in ('ready', 'runtime-ready', 'verified'):
                return Result(state, expired=False, next_step=self._next_step(state))
            if state.deadline_at is None or self.clock() <= float(state.deadline_at):
                return Result(state, expired=False, next_step=self._next_step(state))
            # Count the attempt before trying it, not after. A recovery that dies
            # part way through must still leave evidence that this round was
            # already attempted, otherwise an unattended sweep would retry it
            # without bound.
            state.deadline_attempts = int(state.deadline_attempts or 0) + 1
            self.store.write(state)
            try:
                state, notes = self._recover(state, 'deadline')
            except Refusal:
                stuck = self.store.read()
                if stuck.phase == 'recovering':
                    stuck.phase = ('replacement-required'
                                   if self.assignment.get('replacement', {}).get('enabled')
                                   else 'investigating')
                    self.store.write(stuck)
                raise
            return Result(state, expired=True,
                          next_step='This recovery ran on the exercise deadline, '
                                    'not from your diagnosis.')


def sweep(config, clock=time.time, session_factory=None):
    """Run the deadline safeguard across every configured assignment.

    This exists because the per-assignment `--expire` hook was never reachable
    without somebody typing it: no timer, cron entry or other scheduled caller
    invoked it on either node (confirmed on both:
    participant-revision/runs/rework2-inspect-{target,coordinator}/output.log,
    'no aim344 systemd timer', 'no aim344 cron.d entry'). A participant who walks
    away therefore had no recovery path at all, which is what plan task 3.8 asks
    for.

    Failures are bounded in three ways, because an unattended sweep must not turn
    one stuck table into a repeating mutation attempt:
      one table's refusal never stops the others, so a single stuck round cannot
      starve the rest;
      an assignment whose recovery has already failed the allowed number of times
      is left alone and reported, rather than retried forever;
      nothing is retried within one sweep. The next scheduled run is the retry.

    session_factory builds the per-assignment handler. It defaults to the real
    one, which reaches the real privileged routes; the tests pass a recording
    executor instead so the bounding behaviour can be exercised without hardware.
    """
    factory = session_factory or (
        lambda assignment_id, peer_uid: DeviceSession(
            config, assignment_id, peer_uid=peer_uid, clock=clock))
    attempts_allowed = int(config.get('deadline_attempts', 3))
    summary = []
    for assignment_id in sorted(config['assignments']):
        assignment = config['assignments'][assignment_id]
        expected = assignment.get('caller_uid', assignment.get('participant_uid'))
        try:
            handler = factory(assignment_id, expected)
        except Refusal as refusal:
            summary.append(f'{assignment_id}: not usable ({refusal})')
            continue
        try:
            state = handler.store.read()
        except Refusal as refusal:
            summary.append(f'{assignment_id}: unreadable state ({refusal})')
            continue
        if state.phase in ('ready', 'runtime-ready', 'verified'):
            summary.append(f'{assignment_id}: nothing due (phase {state.phase})')
            continue
        if state.deadline_at is None or clock() <= float(state.deadline_at):
            summary.append(f'{assignment_id}: within its deadline '
                           f'(phase {state.phase})')
            continue
        if int(state.deadline_attempts or 0) >= attempts_allowed:
            # Stop rather than mutate the same node on every timer tick. The
            # round stays recorded and drained for participant replace.
            summary.append(f'{assignment_id}: deadline recovery already attempted '
                           f'{state.deadline_attempts} times and did not succeed; '
                           f'participant replacement required (phase {state.phase})')
            continue
        try:
            result = handler.expire()
        except Refusal as refusal:
            summary.append(f'{assignment_id}: deadline recovery did not complete '
                           f'({refusal})')
            continue
        if result.expired:
            summary.append(f'{assignment_id}: recovered on its deadline, '
                           f'phase {result.state.phase}, '
                           f'recovered_by {result.state.recovered_by}')
        else:
            summary.append(f'{assignment_id}: no deadline action taken '
                           f'(phase {result.state.phase})')
    return summary


def render(result, verb):
    lines = []
    state = result.state
    if verb == 'status':
        lines.append(f'Exercise phase: {state.phase}')
        if state.kind:
            lines.append(f'Exercise: {state.kind}')
        lines.append(f'Round: {state.round}')
        lines.append(f'Node state: {result.node_state}')
        if result.node_reason:
            lines.append(f'Node reason: {result.node_reason}')
        if state.collected:
            lines.append(f'Checks collected: {", ".join(state.collected)}')
        if state.check_results:
            lines.append('Recorded check results:')
            for key in sorted(state.check_results):
                record = state.check_results[key]
                lines.append(f'  {key}: {record.get("verdict")} ({record.get("log")})')
        if state.recovered_by:
            lines.append(f'Recovered by: {state.recovered_by}')
        if state.active_fault:
            lines.append(f'Active FLR: {state.active_fault.get("mutation")} '
                         f'job={state.job_id} outcome={state.active_fault.get("outcome")}')
            lines.append('FLR return is not workload recovery; retain watchdog and cleanup logs.')
        elif state.kind == 'efa':
            # Which EFA path this round took, stated rather than left to be
            # inferred from whether a job happened to be present. An idle round is
            # a legitimate round; it is just not evidence about the active one.
            evidence = state.active_efa_evidence
            if evidence is None:
                lines.append('EFA path: idle (no active-workload observation was '
                             'recorded for this round)')
            else:
                lines.append(
                    f'EFA path: active, qualified={evidence.get("qualified")} '
                    f'(device {evidence.get("device")} moved '
                    f'{evidence.get("moved_bytes")} B over '
                    f'{float(evidence.get("interval_s") or 0):.1f} s under job '
                    f'{evidence.get("job_id")})')
                if evidence.get('log'):
                    lines.append(f'  observation: {evidence["log"]}')
                for refusal in evidence.get('refusals') or []:
                    lines.append(f'  unmet: {refusal}')
        if state.phase == 'recovery-failed':
            lines.append('Your node is still out of service: recovery did not '
                         'complete, so it was not returned to the scheduler.')
        if state.notes:
            lines.append(f'Notes: {state.notes}')
    elif verb == 'collect':
        lines.append(result.output.rstrip('\n'))
        lines.append(f'Saved: {result.saved_path}')
    elif verb in ('start', 'active'):
        if result.already_started:
            lines.append(f'This exercise is already running at {state.phase}.')
        else:
            lines.append(f'Started the {state.kind} exercise. Phase: {state.phase}.')
    elif verb in ('recover', 'replace'):
        if result.already_recovered:
            lines.append('Recovery has already completed for this round.')
        else:
            lines.append('Runtime is restored and your node is back in service.')
            if state.notes:
                lines.append(f'Notes: {state.notes}')
            lines.append('Runtime ready is not the same as a verified workload.')
    if result.next_step:
        lines.append(f'Next: {result.next_step}')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--assignment')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--expire', action='store_true',
                        help='Facilitator deadline sweep for one assignment; '
                             'not a participant verb')
    parser.add_argument('--sweep', action='store_true',
                        help='Facilitator deadline sweep across every assignment; '
                             'the scheduled entry point, not a participant verb')
    options = parser.parse_args(argv)
    try:
        # Argument shape first, before anything touches the filesystem. A bad
        # invocation should say so rather than report whatever the configuration
        # check happened to notice first.
        if options.sweep and options.expire:
            raise Refusal('Use either --sweep or --expire, not both.')
        if not options.sweep and not options.assignment:
            raise Refusal('An assignment is required.')
        config = load_config(Path(options.config))
        if options.sweep:
            if os.geteuid() != 0:
                raise Refusal('The deadline sweep runs as root.')
            for line in sweep(config):
                sys.stdout.write(line + '\n')
            return 0
        if options.expire:
            if os.geteuid() != 0:
                raise Refusal('The deadline sweep runs as root.')
            assignment = config['assignments'].get(options.assignment)
            if assignment is None:
                raise Refusal('Unknown assignment.')
            expected = assignment.get('caller_uid', assignment.get('participant_uid'))
            handler = DeviceSession(config, options.assignment, peer_uid=expected)
            result = handler.expire()
            sys.stdout.write(render(result, 'status'))
            return 0
        request = parse_participant_request(os.environ.get('SSH_ORIGINAL_COMMAND', ''))
        if request.error:
            sys.stderr.write(request.error + '\n')
            return 2
        # Under the forced command the helper runs as root through one sudo rule,
        # so os.getuid() is 0 and cannot identify the caller. sudo sets SUDO_UID
        # itself and overwrites any value the caller tries to supply (verified on
        # the target: an attempted SUDO_UID=1234567 still arrived as the control
        # account's uid), so it is the trustworthy caller identity here.
        peer = os.environ.get('SUDO_UID')
        peer = int(peer) if peer and peer.isdigit() else os.getuid()
        handler = DeviceSession(config, options.assignment, peer_uid=peer)
        verb = request.verb
        if verb in ('start', 'active'):
            result = handler.start(request.argument, active=verb == 'active')
        elif verb == 'status':
            result = handler.status()
        elif verb == 'collect':
            result = handler.collect(request.argument)
        elif verb == 'replace':
            result = handler.replace()
        else:
            result = handler.recover()
        sys.stdout.write(render(result, verb))
        return 0
    except Refusal as refusal:
        sys.stderr.write(str(refusal) + '\n')
        return 3


def _peer_uid_from_connection():
    """The uid of the account that authenticated this SSH connection.

    sshd runs the forced command as that account, so the effective uid of this
    process is the authenticated identity. There is no participant-supplied
    value in this path.
    """
    return os.getuid()


if __name__ == '__main__':
    sys.exit(main())
