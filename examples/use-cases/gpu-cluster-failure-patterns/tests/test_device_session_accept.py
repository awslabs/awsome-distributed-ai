# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Repairs for the participant-acceptance findings F3, F5, F6 and F17.

Each test was written to fail against the tree the review rejected and to pass
only once the named defect is repaired. They reuse the recorded fake executor
from test_device_session.py and the config fixture from
test_device_session_rework.py rather than introducing another framework.

Two of these findings are about artifacts outside this Python helper, so the
tests read those artifacts:

  F3  the shipped verification entry point must not need a permission the
      participant does not hold. Asserted against 13.verify-after-recovery.sbatch
      itself, because that file is what a participant runs.
  F17 the restored checkpoint fixture must belong to the authenticated table's
      account. Asserted through the maintenance helper's own argument handling
      and through what the session helper actually asks it for.

F4 (a full participant acceptance cycle) is not a unit test: it needs the pair.
Its evidence is the recorded hardware cycles in REWORK-ACCEPTANCE.md.

Run: python3 -m unittest discover -s <lab>/tests -p 'test_*.py'
"""
import contextlib
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import unittest

LAB = Path(__file__).resolve().parents[1]
HELPER = LAB / 'facilitator' / 'device-session.py'
MAINTENANCE = LAB / 'facilitator' / 'maintenance.py'
VERIFY_SBATCH = LAB / '13.verify-after-recovery.sbatch'
RESTORE_RUNTIME = LAB / 'facilitator' / 'restore-runtime.sh'
DEVICE_FAULT = LAB / 'facilitator' / 'device-fault.sh'
INSTALLER = LAB / 'facilitator' / 'install-participant-control.sh'

_maint_spec = importlib.util.spec_from_file_location('aim344_maintenance_accept',
                                                     MAINTENANCE)
if _maint_spec is None or _maint_spec.loader is None:         # pragma: no cover
    raise unittest.SkipTest(f'Maintenance helper not importable: {MAINTENANCE}')
maintenance = importlib.util.module_from_spec(_maint_spec)
_maint_spec.loader.exec_module(maintenance)

_rework_spec = importlib.util.spec_from_file_location(
    'aim344_rework_accept',
    Path(__file__).resolve().parent / 'test_device_session_rework.py')
if _rework_spec is None or _rework_spec.loader is None:       # pragma: no cover
    raise unittest.SkipTest('test_device_session_rework.py not importable')
_rework = importlib.util.module_from_spec(_rework_spec)
_rework_spec.loader.exec_module(_rework)
Base = _rework.Base
ReworkExecutor = _rework.ReworkExecutor
# Reuse the module object that the shared Base and its executor were built
# against. Loading device-session.py a second time here would create a second,
# distinct Refusal class, so assertRaises(session.Refusal) would not catch the
# Refusal the handler actually raises and would report an error instead of a
# pass or a failure. Same file, same module object.
session = _rework.session

# The boot a hand-built round is recorded as having started on. It has to be a real
# boot id, because that is what the record holds: `boot-id` returns the contents of
# /proc/sys/kernel/random/boot_id, and the controller compares two readings of that
# file to decide whether the node restarted. A label like 'boot-before' is not a value
# any record can hold, and a fixture carrying one tests a comparison the deployed
# helper never makes -- it also differs from every real reading, so it would read as a
# completed reboot on the first comparison. This value differs from the fake
# executor's own boot ids, which is what these tests need.
BOOT_BEFORE = 'dddddddd-0000-0000-0000-00000000000b'


def flattened(path):
    """One whitespace-normalised string, so a wrapped line still matches.

    A substring check against a file that wraps its own long lines silently finds
    nothing, which reads exactly like a passing assertion. Flatten first.
    """
    return re.sub(r'\s+', ' ', path.read_text())


def sudo_invocations(text):
    """Lines in a shell script that actually invoke sudo as a command.

    The word appearing inside a quoted message is not an invocation, so a plain
    substring search over-reports. This detector is exercised on both a script
    that must be clean and the rejected line that must be caught, so it is not
    validated only on the input that passes.
    """
    found = []
    pattern = re.compile(r'(?:^|[;&|(]\s*|\s\|\|\s|\s&&\s)\s*sudo\s')
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for match in pattern.finditer(line):
            prefix = line[:match.start()]
            if prefix.count("'") % 2 == 0 and prefix.count('"') % 2 == 0:
                found.append(stripped)
                break
    return found


def _hold_lock(state_dir, started_event):
    """Take the assignment lock in a separate process, then wait to be killed.

    Module level so it survives being pickled for the spawn start method.
    """
    store = session.StateStore(state_dir, 'table-1', target_node='gpu-g7-1')
    with store.exclusive():
        started_event.set()
        time.sleep(30)


# --------------------------------------------------------------------------
# F3: the shipped verification entry point must be runnable as authorized.
# --------------------------------------------------------------------------
class F3ParticipantCanRunTheVerification(unittest.TestCase):
    def test_the_verification_entry_point_does_not_invoke_sudo(self):
        """The participant holds no sudo rule, so a sudo here is unusable.

        The one scoped rule belongs to the control account and permits only the
        session helper with a fixed assignment. Confirmed on both nodes:
        `sudo -n -l -U aim344-t1` reports 'not allowed to run sudo'
        (participant-revision/runs/rework2-inspect-coordinator/output.log:35).
        """
        offenders = sudo_invocations(VERIFY_SBATCH.read_text())
        self.assertEqual([], offenders,
                         f'{VERIFY_SBATCH.name} still invokes sudo: {offenders}')

    def test_the_sudo_detector_would_catch_the_rejected_invocation(self):
        """Negative control for the test above, which must be able to fail.

        The rejected line is fed through the same detector. A checker exercised
        only on the input that passes is half verified: it could accept anything.
        The second case is the printed message in the repaired script, which must
        NOT be reported, so the detector is not simply matching the word.
        """
        rejected = ('NCCL_CONTAINER=/opt/aim344/nccl-baseline.sqsh NCCL_TIMEOUT=600 \\\n'
                    '  sudo --preserve-env=NCCL_CONTAINER,NCCL_TIMEOUT,SLURM_JOB_ID,PATH \\\n'
                    '  bash "$suite" --check 5\n')
        self.assertEqual(1, len(sudo_invocations(rejected)),
                         sudo_invocations(rejected))
        message = ("printf '   invoked_as=%s (uid %s), sudo is not used here\\n' "
                   '"$(id -un)" "$(id -u)"\n')
        self.assertEqual([], sudo_invocations(message))

    def test_the_named_sudo_free_command_is_the_one_that_runs_check_5(self):
        """A negative control for the test above.

        Deleting the sudo is only a repair if the suite is still invoked. This
        asserts the check-5 invocation is present, so an empty file would fail
        rather than pass the sudo test vacuously.
        """
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('run_check5 "$suite" "$out/check5"', flat)
        common = flattened(LAB / 'common.sh')
        self.assertIn('--check 5', common)
        self.assertIn('bash "$suite"', common)
        self.assertEqual([], sudo_invocations((LAB / 'common.sh').read_text()))

    def test_the_entry_point_records_the_identity_it_actually_ran_as(self):
        """Who ran it is a fact to record, not an assumption to state."""
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('invoked_as', flat)
        self.assertIn('id -un', flat)

    def test_the_entry_point_prints_raw_correctness_rows_not_only_a_verdict(self):
        """F4's evidence requirement, enforced in the artifact that produces it.

        The suite writes its raw capture to nccl-allreduce-raw.txt in the results
        directory (read in the suite that runs:
        checks/5-nccl-allreduce.sh:230). A verdict line alone does not show the
        measurement, so the entry point must print the rows.
        """
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('nccl-allreduce-raw.txt', flat)
        self.assertIn('Out of bounds values', flat)

    def test_a_per_device_zero_byte_delta_fails_the_verification(self):
        """A green log with no traffic is not a pass.

        An earlier run recorded all-zero deltas on both devices and the script
        did not notice (participant-revision/runs/run-verification-2/output.log).
        """
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('ZERO BYTES', flat)
        self.assertIn('tx_bytes', flat)

    def test_the_fixture_is_checked_on_every_node_before_the_storage_workload(self):
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('fixture_accessible_on_all_nodes_exit_code', flat)
        self.assertIn('/run/aim344-checkpoints', flat)

    def test_the_rendezvous_address_is_an_address_not_a_scheduler_node_name(self):
        """A Slurm node name does not resolve inside the container.

        The scheduler name may be unresolved even when its registered address
        is reachable. Select NodeAddr rather than deriving an address from the
        logical node name. Collective checks alone do not validate rendezvous.
        """
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('NodeAddr=', flat)
        self.assertIn('MASTER_ADDR=$(scontrol show node', flat)
        # And it must refuse to start the workload rather than hang if no address
        # could be resolved.
        body = VERIFY_SBATCH.read_text()
        addr = body.index('could not resolve a rendezvous address')
        workload = body.index('torch-node.sh storage')
        self.assertLess(addr, workload)
        # The unresolved-address path must still report the storage step as not
        # attempted rather than leaving the reader to infer it. That state now
        # travels through the shared retrieval function, so the assertion is on
        # the call that carries it: `retrieve_results <overall> not-attempted`.
        self.assertIn('retrieve_results 1 not-attempted', flat)

    def test_an_unusable_fixture_stops_before_the_workload_rather_than_hanging(self):
        """Fail fast, because the slow path costs the whole allocation.

        A rank that cannot read /checkpoints does not exit: it blocks in
        rendezvous until the job's time limit. Measured on job 86, where only one
        of the two nodes had been repaired: the fixture check reported
        first-host readable=yes and second-host readable=NO, the workload
        was started anyway, and the job was CANCELLED DUE TO TIME LIMIT about ten
        minutes later with the summary line never printed. The permission cause
        was buried under a TCPStore timeout.
        """
        body = VERIFY_SBATCH.read_text()
        check = body.index('fixture_accessible_on_all_nodes_exit_code')
        workload = body.index('torch-node.sh storage')
        exits = [m.start() for m in re.finditer(r'^\s*exit 1$', body, re.M)]
        self.assertTrue(any(check < position < workload for position in exits),
                        'nothing exits between the fixture check and the workload, '
                        'so an unusable fixture would still start it')
        self.assertIn('retrieve_results 1 not-attempted', flattened(VERIFY_SBATCH))


class F3PerDeviceDeltaJudgement(unittest.TestCase):
    """Exercise the embedded delta checker on both outcomes, not only failure.

    A checker validated only against the input that fails is half verified: it
    could reject everything and look correct. These run the real block extracted
    from the sbatch, against a passing input and several failing ones.

    R6 changed what the checker reads. It used to glob `*/efa-deltas.json` on the
    batch host, so a node whose record existed only on its own filesystem was
    never judged: measured on job 89, where both hosts printed their records and
    only the target's two devices received verdicts
    (participant-revision/runs/rework2-read-job89/output.log:55-59). It now reads
    the collected stream of every rank's output, keyed by the Slurm node name each
    rank printed, and judges that against the node list Slurm allocated.
    """

    def setUp(self):
        body = VERIFY_SBATCH.read_text()
        # The heredoc opener carries a trailing `|| status=1`, so the delimiter is
        # not the last thing on its line. Anchoring on the newline after the
        # opener rather than immediately after the delimiter is what makes this
        # match; the earlier version matched nothing and every test in this class
        # failed loudly rather than passing vacuously, which is the point of the
        # size assertion below.
        match = re.search(r"<<'PYDELTA'[^\n]*\n(.*?)\nPYDELTA\n", body, re.S)
        self.assertIsNotNone(match, 'the per-device delta checker block is missing')
        assert match is not None
        self.program = match.group(1)
        # Refuse to run a vacuous extraction: the first version of a harness like
        # this one silently extracted nothing and reported a pass either way.
        self.assertGreater(len(self.program.splitlines()), 15,
                           'extracted too few lines to be the real checker')
        self.assertIn('ZERO BYTES', self.program)

    def run_checker(self, records_by_node, expected_nodes=None, expected_efa=2,
                    interleave_markers=False):
        """Run the shipped checker over a stream shaped like the real one.

        `records_by_node` maps a Slurm node name to the deltas_bytes mapping that
        node's rank printed, or to None for a node that printed its marker and
        then failed to produce a record.

        `interleave_markers` reproduces what srun actually does with concurrent
        ranks' stdout: on job 104 both AIM344_DELTA_NODE markers arrived before
        both records. The checker must be indifferent to that, which is why this
        option exists and why one test sets it.
        """
        if expected_nodes is None:
            expected_nodes = list(records_by_node)
        hosts = {'gpu-g7-1': 'host-a.example', 'gpu-g7-2': 'host-b.example',
                 'gpu-g7-9': 'host-other.example'}
        markers, bodies = [], []
        for node, deltas in records_by_node.items():
            host = hosts.get(node, node)
            markers.append(f'AIM344_DELTA_NODE={node} host={host}')
            if deltas is None:
                bodies.append('EFA byte counters unavailable; traffic attribution '
                              'is UNVALIDATED.')
                continue
            bodies.append(json.dumps({
                'host': host, 'instance_id': 'i-0123456789abcdef0',
                'instance_type': 'g7.48xlarge', 'interval_s': 12.0,
                'deltas_bytes': deltas}))
        if interleave_markers:
            lines = markers + bodies
        else:
            lines = [item for pair in zip(markers, bodies) for item in pair]
        node_map = ' '.join(f'{node}={hosts.get(node, node)}'
                            for node in expected_nodes)
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / 'efa-deltas-all-nodes.txt'
            stream.write_text('\n'.join(lines) + '\n')
            program = Path(tmp) / 'delta.py'
            program.write_text(self.program)
            finished = subprocess.run(
                ['python3', str(program), str(stream)],
                capture_output=True, text=True, check=False,
                env={**os.environ,
                     'AIM344_EXPECTED_NODES': ' '.join(expected_nodes),
                     'AIM344_EXPECTED_EFA': str(expected_efa),
                     'AIM344_NODE_MAP': node_map})
            return finished.returncode, finished.stdout + finished.stderr

    @staticmethod
    def counters(devices, tx, rx):
        values = {}
        for device in devices:
            base = f'/sys/class/infiniband/{device}/ports/1/hw_counters'
            values[f'{base}/tx_bytes'] = tx
            values[f'{base}/rx_bytes'] = rx
            values[f'{base}/rdma_write_bytes'] = tx
        return values

    def test_real_traffic_on_every_device_of_every_node_passes(self):
        """The success path, so the checker is not merely a rejector."""
        code, output = self.run_checker({
            'gpu-g7-1': self.counters(['rdmap176s0', 'rdmap83s0'],
                                      4_000_000, 3_500_000),
            'gpu-g7-2': self.counters(['rdmap176s0', 'rdmap83s0'],
                                      3_900_000, 3_600_000),
        })
        self.assertEqual(0, code, output)
        self.assertIn('moved bytes', output)
        self.assertNotIn('ZERO BYTES', output)
        # Both allocated nodes must actually appear in the verdict, not just one.
        self.assertIn('gpu-g7-1', output)
        self.assertIn('gpu-g7-2', output)

    def test_one_silent_device_fails_even_when_the_other_moved_bytes(self):
        code, output = self.run_checker({
            'gpu-g7-1': {
                **self.counters(['rdmap176s0'], 4_000_000, 3_500_000),
                **self.counters(['rdmap83s0'], 0, 0),
            },
            'gpu-g7-2': self.counters(['rdmap176s0', 'rdmap83s0'],
                                      3_900_000, 3_600_000),
        })
        self.assertEqual(1, code, output)
        self.assertIn('ZERO BYTES', output)

    def test_a_stream_with_no_records_at_all_is_unvalidated_not_a_pass(self):
        code, output = self.run_checker({}, expected_nodes=['gpu-g7-1', 'gpu-g7-2'])
        self.assertEqual(1, code, output)
        self.assertIn('UNVALIDATED', output)

    def test_a_node_that_produced_no_record_fails_the_run(self):
        """R6's central case: the other allocated node is silent.

        Under the rejected implementation this passed, because the judgement
        globbed the batch host's own directory and a node that wrote nothing there
        simply did not appear.
        """
        code, output = self.run_checker({
            'gpu-g7-1': self.counters(['rdmap176s0', 'rdmap83s0'],
                                      4_000_000, 3_500_000),
            'gpu-g7-2': None,
        })
        self.assertEqual(1, code, output)
        self.assertIn('gpu-g7-2', output)
        self.assertIn('UNVALIDATED', output)

    def test_a_node_missing_from_the_stream_entirely_fails_the_run(self):
        """Not even a marker: the rank never ran. Still a failure, not a pass."""
        code, output = self.run_checker(
            {'gpu-g7-1': self.counters(['rdmap176s0', 'rdmap83s0'],
                                       4_000_000, 3_500_000)},
            expected_nodes=['gpu-g7-1', 'gpu-g7-2'])
        self.assertEqual(1, code, output)
        self.assertIn('gpu-g7-2', output)

    def test_an_empty_delta_mapping_is_not_evidence_of_traffic(self):
        """A record existing is not a device having moved bytes.

        The rejected checker iterated `record['deltas_bytes'].items()` and an
        empty mapping produced no device, no failure and no output, so the run
        passed on a record that measured nothing.
        """
        code, output = self.run_checker({
            'gpu-g7-1': {},
            'gpu-g7-2': self.counters(['rdmap176s0', 'rdmap83s0'],
                                      3_900_000, 3_600_000),
        })
        self.assertEqual(1, code, output)
        self.assertIn('gpu-g7-1', output)

    def test_fewer_devices_than_expected_fails_even_when_each_moved_bytes(self):
        """One EFA back instead of two is the exact fault being recovered from."""
        code, output = self.run_checker({
            'gpu-g7-1': self.counters(['rdmap176s0'], 4_000_000, 3_500_000),
            'gpu-g7-2': self.counters(['rdmap176s0', 'rdmap83s0'],
                                      3_900_000, 3_600_000),
        })
        self.assertEqual(1, code, output)
        self.assertIn('expected 2', output)

    def test_a_record_from_a_node_outside_the_allocation_fails_the_run(self):
        code, output = self.run_checker(
            {'gpu-g7-1': self.counters(['rdmap176s0', 'rdmap83s0'], 4_000_000,
                                       3_500_000),
             'gpu-g7-2': self.counters(['rdmap176s0', 'rdmap83s0'], 3_900_000,
                                       3_600_000),
             'gpu-g7-9': self.counters(['rdmap176s0', 'rdmap83s0'], 1, 1)},
            expected_nodes=['gpu-g7-1', 'gpu-g7-2'])
        self.assertEqual(1, code, output)
        self.assertIn('host-other.example', output)

    def test_interleaved_rank_output_is_still_attributed_correctly(self):
        """Regression for a defect this checker actually had, found by running it.

        srun multiplexes concurrent ranks' stdout. On job 104 both node markers
        arrived before both records, so a checker that claimed each record for the
        marker above it assigned both to one node and failed the run for the other
        having 'produced NO counter record' -- while that node had in fact moved
        about 11 GB on each of its two devices
        (participant-revision/runs/rework5-read-cyc-efa-c1/output.log). The
        attribution now uses the host each producer wrote into its own record, so
        the ordering does not matter.
        """
        good = {
            'gpu-g7-1': self.counters(['rdmap176s0', 'rdmap83s0'],
                                      11_014_607_592, 11_013_213_956),
            'gpu-g7-2': self.counters(['rdmap176s0', 'rdmap83s0'],
                                      11_016_585_120, 11_016_190_384),
        }
        grouped_code, grouped_output = self.run_checker(good)
        self.assertEqual(0, grouped_code, grouped_output)
        interleaved_code, interleaved_output = self.run_checker(
            good, interleave_markers=True)
        self.assertEqual(0, interleaved_code, interleaved_output)
        self.assertIn('every expected node', interleaved_output)
        for node in ('gpu-g7-1', 'gpu-g7-2'):
            self.assertIn(node, interleaved_output)

    def test_interleaving_does_not_hide_a_genuinely_silent_node(self):
        """The other direction, so the fix did not become 'accept anything'."""
        code, output = self.run_checker(
            {'gpu-g7-1': self.counters(['rdmap176s0', 'rdmap83s0'],
                                       4_000_000, 3_500_000),
             'gpu-g7-2': None},
            interleave_markers=True)
        self.assertEqual(1, code, output)
        self.assertIn('gpu-g7-2', output)
        self.assertIn('UNVALIDATED', output)


class R7ParticipantResultRetrieval(unittest.TestCase):
    """The participant must be able to read their own verification result.

    The job's log and results live on whichever node Slurm chose as the batch
    host, and that can be the fault target while the participant's terminal is on
    the coordinator. Every cited acceptance cycle hit this: the cycle output says
    the log is on the other node and was 'collected separately by SSM'
    (participant-revision/runs/rework2-accept-efa-c3/output.log:70-76). An
    administrative readback is reviewer evidence, not the participant reading
    their own result, so the shipped script has to carry the transfer itself.
    """

    def test_the_results_are_distributed_to_every_allocated_node(self):
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('retrieve_results', flat)
        self.assertIn('result_bundle=', flat)
        self.assertIn('result_distribution_exit_code', flat)
        self.assertIn('nodes_with_matching_bundle', flat)

    def test_retrieval_uses_no_privilege_and_no_administrative_route(self):
        """It runs inside the participant's own job, as themselves.

        A retrieval that needed sudo, ssh to the target, or the maintenance key
        would not be a participant action at all, which is the whole finding.
        """
        body = VERIFY_SBATCH.read_text()
        self.assertEqual([], sudo_invocations(body))
        retrieval = body[body.index('retrieve_results() {'):]
        for forbidden in ('ssh ', 'scp ', 'ssm', 'aim344-maintenance',
                          'aws '):
            self.assertNotIn(forbidden, retrieval,
                             f'the retrieval path reaches for {forbidden!r}')
        self.assertIn('srun', retrieval)

    def test_retrieval_does_not_use_sbcast_which_this_deployment_refuses(self):
        """Measured, not assumed from the tool's name.

        sbcast is the obvious primitive for this and it does not work on the
        deployed PCS build: as aim344-t1 inside their own allocation it returned
        'sbcast: error: Slurm job 98 lookup error: Unspecified error' with rc=255,
        and no file landed on either node
        (participant-revision/runs/rework5-r7-read-probe-log-target/output.log).
        A plain stdin redirect through srun landed byte-identical
        participant-owned copies on both
        (participant-revision/runs/rework5-r7-read-transfer-log-target/output.log).
        """
        body = VERIFY_SBATCH.read_text()
        retrieval = body[body.index('retrieve_results() {'):]
        invocations = [line.strip() for line in retrieval.splitlines()
                       if re.match(r'^\s*sbcast\s', line)]
        self.assertEqual([], invocations,
                         f'retrieval calls sbcast, which this build refuses: '
                         f'{invocations}')

    def test_the_distributed_copy_is_verified_by_digest_on_each_node(self):
        """A truncated or stale copy at the destination reads as a result."""
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('sha256sum', flat)
        self.assertIn('RESULT RETRIEVAL FAILED', flat)

    def test_a_failed_retrieval_fails_the_run(self):
        """Otherwise a run nobody can inspect still reports success."""
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('if retrieve_results "$status" "$storage"; then', flat)
        self.assertIn('else status=1 fi exit "$status"', flat)

    def test_every_exit_path_retrieves_what_it_produced(self):
        """Including the early ones, whose evidence matters most.

        An `exit` that skipped retrieval would reproduce R7 for exactly the
        failing rounds a participant most needs to read.
        """
        body = VERIFY_SBATCH.read_text()
        after_definition = body[body.index('retrieve_results() {'):]
        exits = [m.start() for m in re.finditer(r'^\s*exit 1$', after_definition,
                                                re.M)]
        self.assertTrue(exits, 'no early exit found after the retrieval function')
        for position in exits:
            preceding = after_definition[:position]
            self.assertIn('retrieve_results', preceding.rsplit('printf', 1)[0],
                          'an exit path does not retrieve its results first')

    def test_the_instructions_name_the_participant_side_command(self):
        """The transfer is only useful if the participant is told how to read it."""
        flat = flattened(VERIFY_SBATCH)
        self.assertIn('tar -xf', flat)
        self.assertIn('summary.txt', flat)


# --------------------------------------------------------------------------
# F17: the checkpoint fixture belongs to the authenticated table's account.
# --------------------------------------------------------------------------
class F17FixtureFollowsTheAssignment(Base):
    def test_the_session_helper_names_its_own_participant_when_restoring(self):
        """Not the target node's fixed default.

        The target's configuration said participant_user=ubuntu while the
        authenticated participant was aim344-t1, so the fixture came back as
        `ubuntu 0700` and aim344-t1 could neither read nor write it, on both
        nodes (participant-revision/runs/rework2-inspect-target/output.log:17
        and rework2-inspect-coordinator/output.log:17-18).
        """
        handler = self.make()
        handler.store.write(self.baseline_for(session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'efa-unbind-attempted'])))
        handler.recover()
        restores = [call for call in self.executor.maintenance_calls
                    if call and call[0] == 'restore-runtime']
        self.assertEqual(1, len(restores), self.executor.maintenance_calls)
        self.assertEqual(['restore-runtime', '--participant', 'aim344-t1'],
                         [str(a) for a in restores[0]])

    def test_each_table_restores_its_own_account(self):
        """A negative control: the name must track the assignment, not be fixed."""
        handler = self.make(assignment='table-2')
        handler.store.write(self.baseline_for(session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'efa-unbind-attempted'])))
        handler.recover()
        restores = [call for call in self.executor.maintenance_calls
                    if call and call[0] == 'restore-runtime']
        self.assertEqual(['restore-runtime', '--participant', 'aim344-t2'],
                         [str(a) for a in restores[0]])


class F17MaintenanceParticipantAllowlist(unittest.TestCase):
    def test_restore_runtime_accepts_a_named_participant(self):
        action, options = maintenance.parse(['restore-runtime', '--participant',
                                            'aim344-t1'])
        self.assertEqual('restore-runtime', action)
        self.assertEqual('aim344-t1', options['participant'])

    def test_restore_runtime_still_accepts_no_argument(self):
        action, options = maintenance.parse(['restore-runtime'])
        self.assertEqual('restore-runtime', action)
        self.assertEqual({}, options)

    def test_an_account_outside_this_node_allowlist_is_refused(self):
        config = {'participant_user': 'aim344-t1',
                  'participants': ['aim344-t1', 'aim344-t2']}
        with self.assertRaises(maintenance.Refusal):
            maintenance.resolve_participant(config, 'root')
        with self.assertRaises(maintenance.Refusal):
            maintenance.resolve_participant(config, 'ubuntu')

    def test_an_allowlisted_account_is_accepted(self):
        config = {'participant_user': 'aim344-t1',
                  'participants': ['aim344-t1', 'aim344-t2']}
        self.assertEqual('aim344-t2',
                         maintenance.resolve_participant(config, 'aim344-t2'))

    def test_no_request_keeps_the_configured_default(self):
        config = {'participant_user': 'aim344-t1',
                  'participants': ['aim344-t1', 'aim344-t2']}
        self.assertEqual('aim344-t1',
                         maintenance.resolve_participant(config, None))

    def test_a_configuration_without_an_allowlist_accepts_only_its_default(self):
        """An older config must not become permissive by omission."""
        config = {'participant_user': 'ubuntu'}
        self.assertEqual('ubuntu', maintenance.resolve_participant(config, None))
        self.assertEqual('ubuntu', maintenance.resolve_participant(config, 'ubuntu'))
        with self.assertRaises(maintenance.Refusal):
            maintenance.resolve_participant(config, 'aim344-t1')

    def test_a_shell_metacharacter_in_the_account_name_is_refused(self):
        for bad in ('aim344-t1;id', 'aim344 t1', '../root', 'AIM344-T1', ''):
            with self.subTest(bad=bad):
                with self.assertRaises(maintenance.Refusal):
                    maintenance.parse(['restore-runtime', '--participant', bad])

    def test_an_unknown_option_is_still_refused(self):
        with self.assertRaises(maintenance.Refusal):
            maintenance.parse(['restore-runtime', '--stage-dir', '/tmp'])


class F17FixtureArtifacts(unittest.TestCase):
    def test_restore_runtime_hands_over_existing_fixture_contents(self):
        """The directory alone is not enough.

        A marker or checkpoint left by the previous table stays unwritable for
        the next one, and the workload then fails inside the container with no
        indication that ownership is the cause. Observed on the coordinator: the
        fixture marker was root-owned inside an ubuntu-owned directory
        (participant-revision/runs/rework2-inspect-coordinator/output.log:12).

        The handover mechanism changed for R1: the entries are enumerated and
        their ownership changed through descriptors, never by handing a
        participant-created name to `chown`. What must hold is that the contents
        are handed over at all, so this asserts the enumeration and the
        descriptor-level change rather than the old command shape. R1's own tests
        in test_device_session_rework.py execute the handover and check where the
        ownership actually lands.
        """
        flat = flattened(RESTORE_RUNTIME)
        self.assertIn('os.listdir(fixture)', flat)
        self.assertIn('os.fchown(handle, uid, gid)', flat)
        self.assertIn('handed_over=', flat)
        # And the construction the review rejected is gone from the code, not
        # merely fenced. Comments are excluded: the script explains why that
        # construction is unsafe, and a substring search over the whole file would
        # match its own explanation.
        code = ' '.join(line for line in RESTORE_RUNTIME.read_text().splitlines()
                        if not line.lstrip().startswith('#'))
        self.assertNotIn('-exec chown', code)

    def test_the_device_helper_accepts_the_provisioned_table_accounts(self):
        """The job-owner condition is kept, not widened to any user."""
        flat = flattened(DEVICE_FAULT)
        self.assertIn('participant_users', flat)
        self.assertIn('user in allowed_users', flat)
        self.assertIn('An unapproved job is using the target node.', flat)

    def test_the_installer_records_this_node_participant_allowlist(self):
        flat = flattened(INSTALLER)
        self.assertIn('participants', flat)


# --------------------------------------------------------------------------
# F6: the deadline safeguard is scheduled, bounded, and recorded distinctly.
# --------------------------------------------------------------------------
class F6ScheduledDeadlineRecovery(Base):
    def run_sweep(self, now):
        """One sweep pass using the recording executor, not the real routes.

        The default factory builds a DeviceSession that would reach real ssh and
        real scontrol. The tests need the same decisions with the calls recorded,
        so they inject a factory that shares this test's executor.
        """
        config = session.load_config(self.config_path, require_root_owned=False)
        return session.sweep(
            config, clock=lambda: now,
            session_factory=lambda assignment_id, peer_uid: session.DeviceSession(
                config, assignment_id, peer_uid=peer_uid,
                executor=self.executor, clock=lambda: now))

    def test_a_sweep_entry_point_exists_and_covers_every_assignment(self):
        """The per-assignment hook needed somebody to type it.

        Nothing invoked it: no systemd timer and no cron entry on either node
        (participant-revision/runs/rework2-inspect-target/output.log:28-29 and
        rework2-inspect-coordinator/output.log:30-31).
        """
        self.assertTrue(hasattr(session, 'sweep'))
        summary = self.run_sweep(1000.0)
        self.assertEqual(2, len(summary), summary)
        self.assertTrue(any('table-1' in line for line in summary))
        self.assertTrue(any('table-2' in line for line in summary))

    def test_an_abandoned_round_past_its_deadline_is_recovered_unattended(self):
        handler = self.make(now=9999.0)
        handler.store.write(self.baseline_for(session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'efa-unbind-attempted'],
            deadline_at=2000.0)))
        summary = self.run_sweep(9999.0)
        line = next(entry for entry in summary if entry.startswith('table-1'))
        self.assertIn('recovered on its deadline', line)
        self.assertIn('recovered_by deadline', line)
        after = handler.store.read()
        self.assertEqual('runtime-ready', after.phase)
        # Recorded as a deadline recovery, never as a diagnosis the participant
        # earned: the plan requires those to stay distinguishable.
        self.assertEqual('deadline', after.recovered_by)
        self.assertEqual(1, after.deadline_attempts)

    def test_a_round_within_its_deadline_is_not_touched(self):
        """The positive control for the sweep: it must not recover everything."""
        handler = self.make()
        handler.store.write(session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'efa-unbind-attempted'],
            deadline_at=9_000_000.0))
        summary = self.run_sweep(1000.0)
        line = next(entry for entry in summary if entry.startswith('table-1'))
        self.assertIn('within its deadline', line)
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())
        self.assertEqual('investigating', handler.store.read().phase)

    def test_a_completed_round_is_not_recovered_again(self):
        handler = self.make(now=9999.0)
        handler.store.write(session.State(
            phase='runtime-ready', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            recovered_by='participant', deadline_at=2000.0))
        summary = self.run_sweep(9999.0)
        line = next(entry for entry in summary if entry.startswith('table-1'))
        self.assertIn('nothing due', line)
        self.assertEqual([], self.executor.reboots())
        self.assertEqual('participant', handler.store.read().recovered_by)

    def test_repeated_sweeps_stop_after_the_allowed_attempts(self):
        """Bounded failure: a stuck table must not be mutated on every tick."""
        handler = self.make(now=9999.0)
        handler.store.write(session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'efa-unbind-attempted'],
            deadline_at=2000.0))
        # Final finding 2 supersedes the old fallback expectation: this failure
        # retires the node without reboot/runtime changes; sweeps remain bounded.
        self.executor.maintenance_failures['efa-rebind'] = (
            1, 'The rebind did not succeed.\n')
        self.executor.maintenance_failures['restore-runtime'] = (
            1, 'restore-runtime failed.\n')
        lines = []
        for _ in range(5):
            lines.append(next(entry for entry in self.run_sweep(9999.0)
                              if entry.startswith('table-1')))
        attempted = [line for line in lines if 'did not complete' in line]
        abandoned = [line for line in lines if 'participant replacement required' in line]
        self.assertEqual(3, len(attempted), lines)
        self.assertEqual(2, len(abandoned), lines)
        self.assertEqual(3, handler.store.read().deadline_attempts)
        self.assertEqual([], self.executor.reboots())
        self.assertEqual([], self.executor.resumes())
        self.assertNotIn('restore-runtime', self.executor.actions())
        self.assertEqual('replacement-required', handler.store.read().phase)

    def test_one_stuck_table_does_not_stop_the_other(self):
        # Fairness applies to different physical nodes. The old fixture put
        # both rounds on gpu-g7-1, incorrectly requiring bypass of its owner.
        self.config_body['assignments']['table-2']['target_node'] = 'gpu-g7-9'
        self.write_config()
        stuck = self.make(assignment='table-1', now=9999.0)
        stuck.store.write(session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'efa-unbind-attempted'],
            deadline_at=2000.0,
            deadline_attempts=99))
        healthy = self.make(assignment='table-2', now=9999.0)
        healthy.store.write(self.baseline_for(session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-9', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'efa-unbind-attempted'],
            deadline_at=2000.0)))
        summary = self.run_sweep(9999.0)
        first = next(line for line in summary if line.startswith('table-1'))
        second = next(line for line in summary if line.startswith('table-2'))
        self.assertIn('participant replacement required', first)
        self.assertIn('recovered on its deadline', second)

    def test_an_unreadable_state_record_is_reported_not_raised(self):
        """An unattended sweep must survive one corrupt record."""
        (self.state_dir / 'table-1.json').write_text('{not json')
        os.chmod(self.state_dir / 'table-1.json', 0o600)
        summary = self.run_sweep(9999.0)
        line = next(entry for entry in summary if entry.startswith('table-1'))
        self.assertIn('unreadable state', line)
        self.assertTrue(any('table-2' in entry for entry in summary))

    def test_the_sweep_records_a_deadline_attempt_even_when_it_fails(self):
        handler = self.make(now=9999.0)
        handler.store.write(session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'efa-unbind-attempted'],
            deadline_at=2000.0))
        # A safety refusal from the device helper: recovery must stop rather than
        # reboot, and the attempt must still be counted.
        self.executor.maintenance_failures['efa-rebind'] = (
            1, 'An unapproved job is using the target node.\n')
        summary = self.run_sweep(9999.0)
        line = next(entry for entry in summary if entry.startswith('table-1'))
        self.assertIn('did not complete', line)
        self.assertEqual(1, handler.store.read().deadline_attempts)
        self.assertEqual([], self.executor.reboots())

    def test_the_sweep_is_installed_as_a_schedule_not_left_to_a_person(self):
        """F6's actual complaint: the hook was never reachable unattended."""
        flat = flattened(INSTALLER)
        self.assertIn('aim344-deadline-sweep.timer', flat)
        self.assertIn('aim344-device-session --sweep', flat)
        self.assertIn('systemctl enable --now aim344-deadline-sweep.timer', flat)
        # oneshot with a start timeout: a hung pass is killed rather than left
        # holding the per-node lock, and the next tick is the retry.
        self.assertIn('Type=oneshot', flat)
        self.assertIn('TimeoutStartSec=', flat)

    def test_the_sweep_command_line_is_accepted_by_the_helper(self):
        """--sweep must not need --assignment, and must refuse both together.

        Before the repair, argparse itself rejected the invocation because
        --assignment was required, so a scheduled `--sweep` could never even
        parse. stderr is captured rather than left to interleave with the test
        runner's own output.
        """

        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            code = session.main(['--sweep', '--config', str(self.config_path)])
        # Not root in the test environment, so this refuses on privilege or on
        # the config's ownership check rather than on argument parsing. Either
        # way it is a Refusal the helper produced, not an argparse exit.
        self.assertEqual(3, code, captured.getvalue())
        self.assertNotIn('the following arguments are required',
                         captured.getvalue())

    def test_sweep_and_expire_together_are_refused(self):
        """A negative control: the two modes must not silently both apply."""

        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            code = session.main(['--sweep', '--expire', '--assignment', 'table-1',
                                 '--config', str(self.config_path)])
        self.assertEqual(3, code, captured.getvalue())

    def test_a_participant_verb_still_requires_an_assignment(self):
        """Making --assignment optional must not make it omittable in practice."""

        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            code = session.main(['--config', str(self.config_path)])
        self.assertEqual(3, code, captured.getvalue())
        self.assertIn('assignment', captured.getvalue().lower())


# --------------------------------------------------------------------------
# F5: an interrupted client, and re-entry without repeated mutation.
# --------------------------------------------------------------------------
class F5InterruptionAndReEntry(Base):
    def test_a_second_request_during_a_live_one_is_refused_not_queued(self):
        """The lock is what makes an interrupted client safe to reconnect after."""
        handler = self.make()
        handler.store.write(session.State(
            phase='fault-applied', kind='gpu', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0'))
        with handler.store.exclusive():
            other = self.make(executor=ReworkExecutor())
            with self.assertRaises(session.Refusal) as caught:
                other.collect('0')
            self.assertIn('in progress', str(caught.exception))

    def test_recover_re_entry_after_a_completed_round_repeats_no_mutation(self):
        """Re-entry must be idempotent, not a second injection.

        A participant whose terminal died reconnects and runs recover again. The
        round already completed, so nothing may be reinjected and no reboot may
        be requested.
        """
        handler = self.make()
        handler.store.write(self.baseline_for(session.State(
            phase='investigating', kind='efa', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'efa-unbind-attempted'])))
        handler.recover()
        first_actions = list(self.executor.actions())
        first_reboots = len(self.executor.reboots())
        first_resumes = len(self.executor.resumes())

        again = self.make(executor=self.executor)
        result = again.recover()
        self.assertTrue(result.already_recovered)
        self.assertEqual(first_actions, self.executor.actions(),
                         'a re-entered recover repeated a privileged call')
        self.assertEqual(first_reboots, len(self.executor.reboots()))
        self.assertEqual(first_resumes, len(self.executor.resumes()))

    def test_start_re_entry_mid_round_does_not_reinject_the_fault(self):
        handler = self.make()
        handler.store.write(session.State(
            phase='fault-applied', kind='gpu', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            prepared=['drain', 'gpu-prepare']))
        before = list(self.executor.actions())
        result = handler.start('gpu')
        self.assertTrue(result.already_started)
        self.assertEqual(before, self.executor.actions(),
                         'a re-entered start reinjected the fault')

    def test_status_after_an_interrupted_request_still_reports_the_round(self):
        """State lives on the coordinator, so a dead client loses nothing."""
        handler = self.make()
        handler.store.write(session.State(
            phase='investigating', kind='gpu', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            collected=['0'], round=2, prepared=['drain', 'gpu-prepare']))
        reconnected = self.make(executor=ReworkExecutor())
        result = reconnected.status()
        self.assertEqual('investigating', result.state.phase)
        self.assertEqual(2, result.state.round)
        self.assertEqual(['0'], result.state.collected)

    def test_a_lock_released_by_a_killed_process_does_not_stay_held(self):
        """A killed client must not strand the assignment.

        flock is released by the kernel when the holding process dies, which is
        why an interrupted request leaves no stuck lock. This asserts that with a
        real process rather than by reasoning about it.
        """
        started = multiprocessing.Event()
        child = multiprocessing.Process(target=_hold_lock,
                                        args=(str(self.state_dir), started))
        child.start()
        try:
            self.assertTrue(started.wait(10), 'child never took the lock')
            store = session.StateStore(self.state_dir, 'table-1',
                                       target_node='gpu-g7-1')
            with self.assertRaises(session.Refusal):
                with store.exclusive():
                    pass
            assert child.pid is not None
            os.kill(child.pid, signal.SIGKILL)
            child.join(10)
            # The very next attempt succeeds: nothing needs cleaning up by hand.
            with store.exclusive():
                pass
        finally:
            if child.is_alive():                            # pragma: no cover
                child.kill()
                child.join(5)


# --------------------------------------------------------------------------
# GPU recovery must not depend on the device the fault removed.
# Found by running a GPU cycle on the pair, not by reading the code.
# --------------------------------------------------------------------------
class GpuRecoveryDoesNotRequireTheRemovedDevice(Base):
    """The identity check must not be a device-inventory check.

    Measured on the target: with the GPU fault applied, the provisioned PCI
    function 0000:ba:00.0 is absent, and `aim344-maintenance inspect` returns
    rc=1 'Provisioned gpu PCI function is absent' while `board_asset_tag` still
    reads i-0123456789abcdef0 (runs/rework2-probe-gpu-inspect/output.log). The
    recovery path used inspect purely to confirm identity, so every GPU recovery
    refused before it could request the reboot that restores the device, and the
    round stayed at investigating with the node drained
    (runs/rework2-accept-gpu-c1/output.log).
    """

    class InspectFailsExecutor(ReworkExecutor):
        """A target whose device inventory refuses, as a GPU fault makes it.

        instance-id still answers, because the DMI field does not depend on the
        removed device. This is the real behaviour, mirrored.
        """

        def run_maintenance(self, argv, timeout=None):
            if argv and argv[0] == 'inspect':
                self.maintenance_calls.append(list(argv))
                return session.Completed(
                    1, '', 'Provisioned gpu PCI function is absent: 0000:ba:00.0\n')
            return super().run_maintenance(argv, timeout=timeout)

    def test_gpu_recovery_completes_while_the_device_inventory_refuses(self):
        executor = self.InspectFailsExecutor()
        handler = self.make(executor=executor)
        handler.store.write(session.State(
            phase='investigating', kind='gpu', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, operation=self.OPERATION,
            prepared=['drain', 'gpu-prepare']))
        result = handler.recover()
        self.assertEqual('runtime-ready', result.state.phase)
        self.assertEqual(1, len(executor.reboots()), executor.reboots())
        self.assertEqual(1, len(executor.resumes()), executor.resumes())
        # Identity was still checked, through the route that works during a fault.
        self.assertIn(['instance-id'],
                      [list(c) for c in executor.maintenance_calls])

    def test_a_changed_instance_still_stops_recovery_through_the_new_route(self):
        """The negative control: separating the checks must not remove one.

        If instance-id reports a different instance, recovery must still refuse
        and must not reboot. Otherwise this repair would have traded a false
        refusal for a missing check.
        """
        executor = self.InspectFailsExecutor()
        executor.inspect['instance_id'] = 'i-0999999999999999f'
        handler = self.make(executor=executor)
        handler.store.write(session.State(
            phase='investigating', kind='gpu', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, prepared=['drain', 'gpu-prepare']))
        with self.assertRaises(session.Refusal) as caught:
            handler.recover()
        self.assertIn('instance', str(caught.exception).lower())
        self.assertEqual([], executor.reboots())
        self.assertEqual([], executor.resumes())

    def test_the_maintenance_route_exposes_a_device_free_identity_action(self):
        action, options = maintenance.parse(['instance-id'])
        self.assertEqual('instance-id', action)
        self.assertEqual({}, options)
        with self.assertRaises(maintenance.Refusal):
            maintenance.parse(['instance-id', 'extra'])

    def test_the_identity_source_is_the_one_the_device_helper_itself_reads(self):
        """Not a weaker check: the same DMI field, read before any device.

        device-fault.sh reads board_asset_tag and validates it before it looks at
        any PCI function, so reading the same field is the same authority.
        """
        self.assertIn('board_asset_tag', flattened(MAINTENANCE))
        self.assertIn('board_asset_tag', flattened(DEVICE_FAULT))
        helper = flattened(HELPER)
        self.assertIn("self._maintenance(['instance-id'])", helper)

    def test_the_state_file_still_shows_that_a_reboot_happened(self):
        """The reboot evidence must survive in the record.

        state.boot_id was overwritten with the new boot and boot_id_after_reboot
        was set to the same value, so both fields ended up identical and the
        record read as "no reboot occurred" -- the opposite of what happened, and
        the reboot IS the recovery mechanism for a GPU round. Seen on gpu-g7-1
        round 7, where the node's own BootTime and `last reboot` confirmed the
        reboot while the state file's two boot ids matched
        (runs/rework2-verify-target-recovered/output.log).
        """
        executor = self.InspectFailsExecutor()
        handler = self.make(executor=executor)
        handler.store.write(session.State(
            phase='investigating', kind='gpu', started_at=1.0,
            target_node='gpu-g7-1', target_instance_id='i-0123456789abcdef0',
            boot_id=BOOT_BEFORE, operation=self.OPERATION,
            prepared=['drain', 'gpu-prepare']))
        result = handler.recover()
        state = result.state
        self.assertTrue(state.reboot_completed)
        self.assertEqual(BOOT_BEFORE, state.boot_id_before_reboot)
        self.assertNotEqual(state.boot_id_before_reboot, state.boot_id_after_reboot)
        # And it must survive the round trip through the state file.
        reloaded = handler.store.read()
        self.assertEqual(BOOT_BEFORE, reloaded.boot_id_before_reboot)
        self.assertNotEqual(reloaded.boot_id_before_reboot,
                            reloaded.boot_id_after_reboot)

    def test_start_still_uses_the_full_inventory_for_its_confirmation_token(self):
        """start must NOT be weakened: it needs the device to exist.

        Before a fault, the device is present and the token comes from the
        inventory. Separating identity from inventory must not turn start into a
        check that would inject into a node whose device is already missing.
        """
        helper = HELPER.read_text()
        start = helper.index('    def start(self, kind, active=False):')
        end = helper.index('    def status(self):')
        body = helper[start:end]
        self.assertIn('self._inspect()', body)
        self.assertIn("'confirmation'", body)


if __name__ == '__main__':
    unittest.main()
