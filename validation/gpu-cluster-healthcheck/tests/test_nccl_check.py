"""Check 5 launch and Check 6 counter regressions; mocks are not hardware qualification.

Adapted from the existing test_nccl_check.py at public revision ffed0df6102e68b3e94806373006b045e432298f.
Launch/profile and main-result parsing regressions use mocked commands, not GPUs.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SUITE = Path(__file__).resolve().parents[1]
ROW = (Path(__file__).parent / 'nccl-valid-rows.txt').read_text()
PROVIDER = 'NET/OFI Selected Provider is efa\n'
HEADER = '# size count type redop root time algbw busbw #wrong time algbw busbw #wrong\n'


class NcclResultTests(unittest.TestCase):
    def run_check(self, *, instance='g7.48xlarge', imds='g7.48xlarge',
                  dmi='', binary=None, local=False, isolation=False, exit_code=0,
                  orchestrator=False, nodes='2', mpi='pmix', long_help=False,
                  wrapper=False, container=None, output=None, efa_stats=None,
                  rdma_exit=0, pin_ignored=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, body in {
                'cat': 'if [ "$1" = /sys/devices/virtual/dmi/id/product_name ]; then printf "%s\\n" "$TEST_DMI"; else exec /bin/cat "$@"; fi',
                'curl': 'echo curl >> "$TEST_DETECTION"; printf "%s\\n" "$TEST_IMDS"',
                'ec2-metadata': 'exit 1',
                'nvidia-smi': 'echo fixture-driver',
                'srun': '''if [ "$1" = --help ]; then
    [ "$TEST_LOCAL" = 1 ] || echo container-image
    if [ "$TEST_LONG_HELP" = 1 ]; then
        python3 -c 'print("Slurm help fixture\\n" * 40000)'
        exit $?
    fi
    exit 0
fi
# Do not execute the wrapper's per-node DCGM/EFA phase.
if [ "$1" = --ntasks-per-node=1 ] && [ "$2" = bash ]; then exit 0; fi
python3 -c 'import json,os,sys; open(os.environ["TEST_ARGS"],"a").write(json.dumps(sys.argv[1:])+"\\n")' "$@"
cat "$TEST_OUTPUT"
exit "$TEST_EXIT"''',
            }.items():
                p = root / name
                p.write_text('#!/bin/sh\n' + body + '\n')
                p.chmod(0o755)
            (root / 'output').write_text(
                PROVIDER + HEADER + ROW + '# Out of bounds values : 0 OK\n'
                if output is None else output)
            # No inherited scheduler, NCCL or profile settings may influence fixtures.
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(('SLURM_', 'NCCL_', 'EXPECTED_', 'INSTANCE_',
                                        'HEALTHCHECK_', 'DRY_RUN', 'NVLINK_', 'EFA_PROVIDER'))}
            env.update(PATH=f'{root}:/usr/bin:/bin', SLURM_JOB_NUM_NODES=nodes,
                       SLURM_NTASKS='16', INSTANCE_TYPE=instance, TEST_IMDS=imds,
                       TEST_DMI=dmi, TEST_DETECTION=str(root / 'detection'),
                       TEST_OUTPUT=str(root / 'output'), TEST_ARGS=str(root / 'args'),
                       TEST_LOCAL=str(int(local)), TEST_EXIT=str(exit_code),
                       RESULTS_DIR=str(root / 'results'), NCCL_ISOLATION_TESTS=str(int(isolation)),
                       NCCL_MPI=mpi, TEST_LONG_HELP=str(int(long_help)))
            if efa_stats is not None:
                # Extend this existing command-mock bench; never invoke EFA hardware.
                (root / 'statistics').write_text(efa_stats)
                for name, body in {
                    'fi_info': 'printf "domain: rdma0-rdm\\ndomain: rdma1-rdm\\n"',
                    'fi_pingpong': '''printf "%s|%s|%s\\n" "$FI_EFA_IFACE" "$FI_EFA_ENABLE_SHM_TRANSFER" "$*" >> "$TEST_PIN_CALLS"
if [ "$TEST_PIN_IGNORED" != 1 ]; then
    case "$FI_EFA_IFACE" in rdma0|rdma1) ;; *) exit 61;; esac
fi
case "$*" in *localhost) exit 0;; *) exec /bin/sleep 30;; esac''',
                    'sleep': 'exit 0',
                    'rdma': 'printf "%s\\n" "$*" >> "$TEST_ARGS"; cat "$TEST_STATS"; exit "$TEST_RDMA_EXIT"',
                }.items():
                    p = root / name
                    p.write_text('#!/bin/sh\n' + body + '\n')
                    p.chmod(0o755)
                env.update(TEST_STATS=str(root / 'statistics'),
                           TEST_PIN_CALLS=str(root / 'pin-calls'),
                           TEST_PIN_IGNORED=str(int(pin_ignored)),
                           TEST_RDMA_EXIT=str(rdma_exit), VERBOSE='1')
                p = subprocess.run(['bash', str(SUITE / 'checks/6-efa-loopback.sh')],
                                   env=env, text=True, capture_output=True, timeout=20)
                self.last_output = p.stdout + p.stderr
                raw = root / 'results/efa-statistics.txt'
                self.last_raw = raw.read_text() if raw.exists() else None
                self.last_rdma_calls = ((root / 'args').read_text().splitlines()
                                        if (root / 'args').exists() else [])
                self.last_pin_calls = (root / 'pin-calls').read_text().splitlines()
                report = json.loads((root / 'results/check-6-efa-loopback.json').read_text())
                return p.returncode, report, [], False
            if container is not None:
                env['NCCL_CONTAINER'] = container
            if wrapper:
                env.update(HEALTHCHECK_DIR=str(SUITE), RESULTS_BASE=str(root / 'wrapper'),
                           SLURM_JOB_ID='fixture', SLURM_JOB_NODELIST='fixture-[1-2]')
            if binary is not None:
                env['NCCL_TESTS_BIN'] = binary
            if local:
                # A real executable fixture on PATH exercises local binary selection.
                p = root / (binary or 'all_reduce_perf')
                p.write_text('#!/bin/sh\nexit 0\n')
                p.chmod(0o755)
            entry = ['gpu-healthcheck.sh', '--check', '5'] if orchestrator else ['checks/5-nccl-allreduce.sh']
            if wrapper:
                entry = ['slurm/sbatch-intensive.sh']
            p = subprocess.run(['bash', str(SUITE / entry[0]), *entry[1:]],
                               env=env, text=True, capture_output=True, timeout=20)
            result_dir = root / 'wrapper/job-fixture-intensive/nccl' if wrapper else root / 'results'
            report = json.loads((result_dir / 'check-5-nccl-allreduce.json').read_text())
            calls = [json.loads(line) for line in (root / 'args').read_text().splitlines()] if (root / 'args').exists() else []
            self.last_output = p.stdout + p.stderr
            raw = result_dir / 'nccl-allreduce-raw.txt'
            self.last_raw = raw.read_text() if raw.exists() else None
            return p.returncode, report, calls, (root / 'detection').exists()

    def test_check6_absolute_all_links_not_domain_or_delta(self):
        # Synthetic rdma text; not an incident capture or pinning proof.
        stats = ('link rdma0/1 rx_drops 0 retrans_timeout_events 1\n'
                 'link rdma1/1\n    rx_drops 2\n    retrans_timeout_events 3\n'
                 'link other0/2 rx_drops 0 retrans_timeout_events 5\n')
        rc, report, _, _ = self.run_check(efa_stats=stats)
        self.assertEqual(rc, 0)
        self.assertEqual(report['status'], 'WARN')
        self.assertEqual(self.last_rdma_calls, ['-p statistic show'])
        self.assertEqual(self.last_raw, stats)
        self.assertIn('cumulative absolute totals over all RDMA links returned', self.last_output)
        self.assertIn('not per-domain or loopback/fault deltas', self.last_output)
        self.assertIn('EFA retransmission timeouts detected (9)', self.last_output)
        self.assertIn('EFA rx_drops detected (2)', self.last_output)
        zero = ('\n\tlink rdma0/1\r\n'
                '  tx_pkts 18446744073709551615 rx_drops 0\r\n'
                '\n  retrans_timeout_events 0 retrans_bytes 99\n'
                ' link other0/2 rx_drops 0 retrans_timeout_events 0\n')
        _, report, _, _ = self.run_check(efa_stats=zero)
        self.assertEqual(report['status'], 'PASS')
        self.assertIn('statistics clean', self.last_output)
        self.assertNotIn('absolute totals unknown', self.last_output)
        self.assertEqual(self.last_rdma_calls, ['-p statistic show'])

    def test_check6_missing_malformed_counters_are_unknown(self):
        good = 'link rdma0/1 rx_drops 0 retrans_timeout_events 0\n'
        cases = ['', 'link rdma0/1\n', good.replace('retrans_timeout_events 0', ''),
                 good + 'link rdma1/1 rx_drops 0\n', good + good,
                 good.replace('link rdma0/1 ', ''),
                 good.replace('rdma0/1', 'rdma0'),
                 good.replace('rdma0/1', 'rdma0/0'),
                 good.replace('rx_drops 0', 'rx_drops 0 rx_drops 0')]
        # Independent review: an incomplete next header must not be absorbed
        # into a preceding valid zero block, even with whitespace or no final LF.
        cases += [good + 'link\n', good + 'link  \n', good + 'link',
                  good + '\t link\t\r\n', good + '\n  link\n',
                  good.rstrip('\n') + ' link\n',
                  good + 'link\nrdma1/1 rx_drops 0 retrans_timeout_events 0\n']
        malformed_sections = (
            'link\n',
            'link rx_drops 0 retrans_timeout_events 0\n',
            'link /1 rx_drops 0 retrans_timeout_events 0\n',
            'link rdma1/ rx_drops 0 retrans_timeout_events 0\n',
            'link rdma1/1\n',
            'link rdma1/1 rx_drops\n',
            'link rdma1/1 rx_drops 0 retrans_timeout_events\n',
            'rdma1/1\n',
            'lin\n',
            'link: rdma1/1\n',
            'damaged-section\n',
            'damaged-section rdma1/1\n',
            'link rdma1/1 rx_drops\n 0 retrans_timeout_events 0\n',
            'link rdma1/1 rx_drops 0\n retrans_timeout_events\n 0\n',
            'link rdma1/1 rx_drops 0 retrans_timeout_events 0 tx_pkts\n',
            'link rdma1/1 rx_drops 0 retrans_timeout_events 0 tx_pkts N/A\n',
        )
        other = good.replace('rdma0/1', 'other0/2')
        for malformed in malformed_sections:
            cases.extend((malformed + good, good + malformed,
                          good + malformed + other))
        # No partial nonzero sum may positively mark an incomplete snapshot
        # as observed to the controller either.
        cases.append(good.replace('rx_drops 0', 'rx_drops 4') +
                     'link rdma1/1 rx_drops 0\n')
        cases.append(good.rstrip('\n') + ' link rdma1/1\n')
        for value in ('N/A', '-1', '1.0', '2x', '18446744073709551616'):
            cases.append(good.replace('retrans_timeout_events 0',
                                      'retrans_timeout_events ' + value))
        for stats in cases:
            with self.subTest(stats=stats):
                _, report, _, _ = self.run_check(efa_stats=stats)
                self.assertEqual(report['status'], 'WARN')
                self.assertIn('[WARN]', self.last_output)
                self.assertIn('rx_drops=unknown, retrans_timeout_events=unknown',
                              self.last_output)
                self.assertIn('absolute totals unknown, not zero', self.last_output)
                self.assertNotIn('statistics clean', self.last_output)
                self.assertNotIn('EFA rx_drops detected', self.last_output)
                self.assertNotIn('EFA retransmission timeouts detected', self.last_output)
                self.assertEqual(self.last_rdma_calls, ['-p statistic show'])

    def test_check6_u64_absolute_and_reset_wrap_do_not_create_delta(self):
        # Separate executions, not a measured pair. A drop could be reset/wrap;
        # equal values likewise cannot establish a stable driver epoch.
        for before, after in ((2, 9), (2, 1), (4, 4), (18446744073709551615, 0)):
            with self.subTest(before=before, after=after):
                for value in (before, after):
                    stats = f'link rdma0/1 rx_drops 0 retrans_timeout_events {value}\n'
                    self.run_check(efa_stats=stats)
                    self.assertIn('not per-domain or loopback/fault deltas', self.last_output)
                    self.assertEqual(self.last_rdma_calls, ['-p statistic show'])
                    if value:
                        self.assertIn(f'EFA retransmission timeouts detected ({value})', self.last_output)
                    else:
                        self.assertIn('no delta or epoch continuity established', self.last_output)
        maximum = 18446744073709551615
        self.run_check(efa_stats=''.join(
            f'link rdma{i}/1 rx_drops 0 retrans_timeout_events {maximum}\n' for i in (0, 1)))
        self.assertIn(f'EFA retransmission timeouts detected ({maximum * 2})', self.last_output)

    def test_check6_rdma_failure_stays_skipped_not_zero(self):
        self.run_check(efa_stats='', rdma_exit=1)
        self.assertIn('rdma statistic show failed -- EFA statistics skipped', self.last_output)
        self.assertNotIn('rx_drops=0', self.last_output)

    def test_check6_owner_pinning_and_negative_control(self):
        stats = 'link rdma0/1 rx_drops 0 retrans_timeout_events 0\n'
        rc, report, _, _ = self.run_check(efa_stats=stats)
        self.assertEqual((rc, report['status']), (0, 'PASS'))
        calls = [line.split('|', 2) for line in self.last_pin_calls]
        self.assertEqual({c[0] for c in calls},
                         {'rdma0', 'rdma1', 'definitely-nonexistent-device'})
        self.assertTrue(all(c[1] == '0' for c in calls))
        for iface in ('rdma0', 'rdma1'):
            selected = [c[2] for c in calls if c[0] == iface]
            self.assertEqual(len(selected), 2)
            self.assertEqual(sum(c.endswith('localhost') for c in selected), 1)
        rc, report, _, _ = self.run_check(efa_stats=stats, pin_ignored=True)
        self.assertNotEqual(rc, 0)
        self.assertEqual((report['status'], report['severity']), ('FAIL', 'RESET'))
        self.assertIn('Negative control failed', self.last_output)
        self.assertEqual(self.last_rdma_calls, [])
        self.assertIsNone(self.last_raw)

    def test_owner_severity_maximum_and_existing_consumers(self):
        # Reproduce #1270's published local test plan; all commands are local.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = {'status': 'PASS', 'severity': '', 'overall_status': 'PASS',
                    'overall_severity': '', 'tests': [{'gpu': 0, 'result': 'seed'}]}
            (root / 'check-0-nvidia-smi.json').write_text(json.dumps(seed))
            command = '''source "$1"
log_result "0-nvidia-smi" "FAIL" "Xid 79: GPU fell off the bus" "ISOLATE"
log_result "0-nvidia-smi" "WARN" "Persistence mode disabled" "MONITOR"
log_result "5-nccl-allreduce" "WARN" "Bus bandwidth below threshold"
log_result "5-nccl-allreduce" "PASS" "NCCL all_reduce completed"
log_result "6-efa-loopback" "WARN" "timeout warning" "MONITOR"
log_result "6-efa-loopback" "PASS" "loopback completed"
log_result "6-efa-loopback" "PASS" "loopback completed"
'''
            p = subprocess.run(['bash', '-c', command, 'test', str(SUITE / 'lib/common.sh')],
                               env=dict(os.environ, RESULTS_DIR=tmp, JSON_OUTPUT='1'),
                               text=True, capture_output=True, timeout=10)
            self.assertEqual(p.returncode, 0, p.stderr)
            events = [json.loads(line) for line in p.stdout.splitlines()]
            self.assertEqual(events[-1]['status'], 'PASS')
            records = {p.name: json.loads(p.read_text()) for p in root.glob('check-*.json')}
            r = records['check-0-nvidia-smi.json']
            self.assertEqual((r['status'], r['severity']), ('FAIL', 'ISOLATE'))
            self.assertEqual(r['tests'], seed['tests'])
            self.assertNotIn('overall_status', r)
            self.assertNotIn('overall_severity', r)
            self.assertEqual(records['check-5-nccl-allreduce.json']['status'], 'WARN')
            r = records['check-6-efa-loopback.json']
            self.assertEqual((r['status'], r['severity']), ('WARN', 'MONITOR'))
            self.assertEqual(r['details'], 'timeout warning | loopback completed')
            for remove_fault, status, severity in ((False, 'FAIL', 'ISOLATE'),
                                                   (True, 'WARN', 'MONITOR')):
                if remove_fault:
                    (root / 'check-0-nvidia-smi.json').unlink()
                a = subprocess.run(['python3', str(SUITE / 'lib/aggregate-results.py'),
                                    '--results-dir', tmp], text=True, capture_output=True,
                                   check=True, timeout=10)
                summary = json.loads(a.stdout)
                self.assertEqual((summary['overall_status'], summary['overall_severity']),
                                 (status, severity))
                k = subprocess.run(['python3', str(SUITE / 'kubernetes/determine-severity.py'),
                                    '--results-dir', tmp], text=True, capture_output=True,
                                   check=True, timeout=10)
                self.assertEqual(k.stdout.strip(), f'{status.lower()}:{severity}')

    def test_owner_severity_accumulation_resets_per_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run(['bash', '-c', '''source "$1"
log_result "6-efa-loopback" "FAIL" "old fault" "ISOLATE"
init_check "6-efa-loopback"
log_result "6-efa-loopback" "PASS" "new execution"
''', 'test', str(SUITE / 'lib/common.sh')],
                env=dict(os.environ, RESULTS_DIR=tmp, EXPECTED_GPU_COUNT='8', DRY_RUN='0'),
                text=True, capture_output=True, timeout=10)
            self.assertEqual(p.returncode, 0, p.stderr)
            r = json.loads((Path(tmp) / 'check-6-efa-loopback.json').read_text())
            self.assertEqual((r['status'], r['severity'], r['details']),
                             ('PASS', '', 'new execution'))

    def captured_output(self, job=252):
        return (Path(__file__).parent / f'nccl-captured-{job}.txt').read_text()

    def test_captured_correctness_and_provider(self):
        for job in (252, 294):
            with self.subTest(job=job):
                output = self.captured_output(job)
                # Independent reference: both #wrong columns, not the implementation parser.
                rows = [line.split() for line in output.splitlines()
                        if line.split() and line.split()[0].isdigit()]
                self.assertEqual(len(rows), 25)
                self.assertTrue(all(len(row) == 13 and row[8] == row[12] == '0'
                                    for row in rows))
                self.assertIn('# Out of bounds values : 0 OK', output)
                for route in ({}, {'orchestrator': True}, {'wrapper': True}):
                    with self.subTest(route=route):
                        rc, report, _, _ = self.run_check(
                            output=output, orchestrator=route.get('orchestrator', False),
                            wrapper=route.get('wrapper', False))
                        self.assertEqual((rc, report['status']), (0, 'PASS'))
                        self.assertNotIn('EFA provider not confirmed', self.last_output)
                        self.assertIn(f"max busbw={max(float(row[11]) for row in rows):g}",
                                      report['details'])
                        self.assertEqual(self.last_raw, output)

    def test_provider_detection_consumes_large_output(self):
        output = self.captured_output() + ('NCCL INFO fixture padding\n' * 10000)
        rc, report, _, _ = self.run_check(output=output)
        self.assertEqual((rc, report['status']), (0, 'PASS'))
        self.assertNotIn('EFA provider not confirmed', self.last_output)

    def test_missing_or_malformed_correctness_never_passes(self):
        good = self.captured_output()
        lines = good.splitlines()
        row_index = next(i for i, line in enumerate(lines)
                         if line.split() and line.split()[0].isdigit())
        cases = {'empty': '', 'provider_only': PROVIDER,
                 'summary_only': '# Out of bounds values : 0 OK\n',
                 'missing_summary': good.replace('# Out of bounds values : 0 OK\n', ''),
                 'bad_summary': good.replace('values : 0 OK', 'values : N/A OK'),
                 'nonzero_summary': good.replace('values : 0 OK', 'values : 1 FAILED')}
        for column in (8, 12):
            for value in ('1', 'N/A', '-1', '0.0'):
                changed = lines.copy()
                fields = changed[row_index].split()
                fields[column] = value
                changed[row_index] = ' '.join(fields)
                cases[f'wrong_{column}_{value}'] = '\n'.join(changed) + '\n'
            changed = lines.copy()
            fields = changed[row_index].split()
            del fields[column]
            changed[row_index] = ' '.join(fields)
            cases[f'missing_wrong_{column}'] = '\n'.join(changed) + '\n'
        for column, value in ((7, 'nan'), (11, 'inf'), (11, '-1')):
            changed = lines.copy()
            fields = changed[row_index].split()
            fields[column] = value
            changed[row_index] = ' '.join(fields)
            cases[f'bandwidth_{column}_{value}'] = '\n'.join(changed) + '\n'
        for name, output in cases.items():
            for orchestrator in (False, True):
                with self.subTest(case=name, orchestrator=orchestrator):
                    rc, report, _, _ = self.run_check(output=output, orchestrator=orchestrator)
                    self.assertEqual((rc, report['status']), (1, 'FAIL'))
                    self.assertNotIn('[PASS]', self.last_output)
                    self.assertEqual(self.last_raw, output.rstrip('\n') + '\n')

    def test_malformed_size_cannot_hide_result_rows(self):
        lines = self.captured_output().splitlines()
        row_index = next(i for i, line in enumerate(lines)
                         if line.split() and line.split()[0].isdigit())
        # Keep the other 24 rows and completion intact: this is not a row-count test.
        for size in ('8x', 'N/A', '-8', '8.0', '1e1'):
            for wrong in ('0', '1'):
                changed = lines.copy()
                fields = changed[row_index].split()
                fields[0], fields[8] = size, wrong
                changed[row_index] = ' '.join(fields)
                output = '\n'.join(changed) + '\n'
                for route in ({}, {'orchestrator': True}, {'wrapper': True}):
                    with self.subTest(size=size, wrong=wrong, route=route):
                        rc, report, _, _ = self.run_check(
                            output=output, orchestrator=route.get('orchestrator', False),
                            wrapper=route.get('wrapper', False))
                        self.assertEqual((rc, report['status']), (1, 'FAIL'))
                        self.assertNotIn('[PASS]', self.last_output)
                        self.assertEqual(self.last_raw, output)

    def test_missing_and_truncated_columns_cannot_hide_result_rows(self):
        lines = self.captured_output().splitlines()
        row_index = next(i for i, line in enumerate(lines)
                         if line.split() and line.split()[0].isdigit())
        fields = lines[row_index].split()
        cases = {
            'review_invalid_size_missing_count':
                '8x float sum -1 54.23 0.00 0.00 1 53.44 0.00 0.00 0',
            'review_missing_size_count':
                'float sum -1 54.23 0.00 0.00 1 53.44 0.00 0.00 0',
            'review_invalid_size_double':
                '8x 2 double sum -1 54.23 0.00 0.00 1 53.44 0.00 0.00 0',
            'unknown_fragment': 'damaged-result',
            'embedded_log_token': '8x NCCL INFO ' + ' '.join(fields[3:]),
        }
        for column in range(len(fields)):
            cases[f'missing_column_{column}'] = ' '.join(fields[:column] + fields[column + 1:])
        for cut in range(1, len(fields)):
            cases[f'missing_prefix_{cut}'] = ' '.join(fields[cut:])
            cases[f'missing_suffix_{cut}'] = ' '.join(fields[:-cut])
        # Keep all other measured rows and the normal completion. No row-count rule.
        for name, row in cases.items():
            changed = lines.copy()
            changed[row_index] = row
            output = '\n'.join(changed) + '\n'
            for route in ({}, {'orchestrator': True}, {'wrapper': True}):
                with self.subTest(case=name, route=route):
                    rc, report, _, _ = self.run_check(
                        output=output, orchestrator=route.get('orchestrator', False),
                        wrapper=route.get('wrapper', False))
                    self.assertEqual((rc, report['status']), (1, 'FAIL'))
                    self.assertNotIn('[PASS]', self.last_output)
                    self.assertEqual(self.last_raw, output)

    def test_table_context_is_required_and_comments_do_not_end_it(self):
        good = self.captured_output()
        header = next(line for line in good.splitlines() if line.startswith('#') and 'size' in line)
        first_row = next(line for line in good.splitlines()
                         if line.split() and line.split()[0].isdigit())
        cases = {
            'missing_header': good.replace(header + '\n', ''),
            'malformed_header': good.replace(header, '# size count type redop'),
            'comment_then_bad_row': good.replace(first_row, '# interleaved comment\n\ndamaged-result'),
            'repeated_header_then_bad_row': good.replace(first_row, HEADER + 'damaged-result'),
        }
        for name, output in cases.items():
            for route in ({}, {'orchestrator': True}, {'wrapper': True}):
                with self.subTest(case=name, route=route):
                    rc, report, _, _ = self.run_check(
                        output=output, orchestrator=route.get('orchestrator', False),
                        wrapper=route.get('wrapper', False))
                    self.assertEqual((rc, report['status']), (1, 'FAIL'))
                    self.assertEqual(self.last_raw, output)

    def test_nonfinite_numeric_fields_never_pass(self):
        lines = self.captured_output().splitlines()
        row_index = next(i for i, line in enumerate(lines)
                         if line.split() and line.split()[0].isdigit())
        cases = [(column, value) for column in (5, 6, 7, 9, 10, 11)
                 for value in ('1e9999', '9' * 400, 'inf', 'NaN')]
        cases += [(column, '9' * 400) for column in (0, 1, 8, 12)]
        for column, value in cases:
            changed = lines.copy()
            fields = changed[row_index].split()
            fields[column] = value
            changed[row_index] = ' '.join(fields)
            output = '\n'.join(changed) + '\n'
            for route in ({}, {'orchestrator': True}, {'wrapper': True}):
                with self.subTest(column=column, value=value, route=route):
                    rc, report, _, _ = self.run_check(
                        output=output, orchestrator=route.get('orchestrator', False),
                        wrapper=route.get('wrapper', False))
                    self.assertEqual((rc, report['status']), (1, 'FAIL'))
                    self.assertNotIn('[PASS]', self.last_output)
                    self.assertEqual(self.last_raw, output)

    def test_finite_scientific_notation_and_diagnostics_remain_valid(self):
        lines = self.captured_output().splitlines()
        row_index = next(i for i, line in enumerate(lines)
                         if line.split() and line.split()[0].isdigit())
        fields = lines[row_index].split()
        # nccl-tests getFloatStr emits both integer and decimal mantissas.
        for column, value in zip((5, 6, 7, 9, 10, 11),
                                 ('5.423e+01', '0e0', '0E+00', '5E1', '1e-03', '2E-03')):
            fields[column] = value
        lines[row_index] = ' '.join(fields)
        lines.insert(row_index + 1, 'node:123:456 [0] NCCL INFO interleaved diagnostic')
        lines.insert(row_index + 2, 'NCCL INFO diagnostic a b c d e f g h i j')
        lines.insert(row_index + 3, '# (B) (elements) (us) (GB/s) (GB/s)')
        lines.insert(row_index + 4, '')
        lines.insert(row_index + 5, HEADER.rstrip('\n'))
        # Non-table prologue/epilogue is not parsed as a result by column count.
        lines.insert(0, '1 launcher diagnostic outside the result table')
        lines.append('2 launcher diagnostic after completion')
        output = '\n'.join(lines) + '\n'
        for route in ({}, {'orchestrator': True}, {'wrapper': True}):
            with self.subTest(route=route):
                rc, report, _, _ = self.run_check(
                    output=output, orchestrator=route.get('orchestrator', False),
                    wrapper=route.get('wrapper', False))
                self.assertEqual((rc, report['status']), (0, 'PASS'))
                self.assertIn('max busbw=39.53', report['details'])
                self.assertNotIn('EFA provider not confirmed', self.last_output)
                self.assertEqual(self.last_raw, output)

    def test_provider_absence_still_warns(self):
        output = '\n'.join(line for line in self.captured_output().splitlines()
                           if 'Selected provider' not in line) + '\n'
        for provider in ('', 'Selected Provider is efa_fake\n',
                         'Selected Provider is efabric\n'):
            with self.subTest(provider=provider):
                rc, _, _, _ = self.run_check(output=provider + output)
                self.assertEqual(rc, 0)
                self.assertIn('EFA provider not confirmed', self.last_output)

    def test_nonzero_execution_and_timeout_remain_failures(self):
        for exit_code in (1, 124, 127, 137):
            with self.subTest(exit_code=exit_code):
                rc, report, _, _ = self.run_check(output=self.captured_output(),
                                                 exit_code=exit_code)
                self.assertEqual((rc, report['status']), (1, 'FAIL'))
                self.assertEqual(self.last_raw, self.captured_output())

    def test_container_binary_and_g7_rank_mapping(self):
        rc, report, calls, _ = self.run_check()
        self.assertEqual((rc, report['status']), (0, 'PASS'))
        args, = calls
        self.assertIn('/opt/nccl-tests/build/all_reduce_perf', args)
        self.assertIn('--ntasks=2', args)
        self.assertIn('--nodes=2', args)
        self.assertIn('--mpi=pmix', args)
        self.assertEqual(args[args.index('-g') + 1], '8')

    def test_explicit_type_survives_orchestrator_and_child(self):
        rc, report, calls, detection = self.run_check(imds='', orchestrator=True)
        self.assertEqual((rc, report['instance_type']), (0, 'g7.48xlarge'))
        self.assertEqual(calls[0][calls[0].index('-g') + 1], '8')
        self.assertFalse(detection)

    def test_dmi_without_imds_or_override(self):
        rc, report, calls, detection = self.run_check(instance='', imds='', dmi='g7.48xlarge', orchestrator=True)
        self.assertEqual((rc, report['instance_type']), (0, 'g7.48xlarge'))
        self.assertEqual(calls[0][calls[0].index('-g') + 1], '8')
        self.assertFalse(detection)

    def test_imds_fallback(self):
        rc, report, calls, detection = self.run_check(instance='', dmi='Not an EC2 type')
        self.assertEqual((rc, report['instance_type']), (0, 'g7.48xlarge'))
        self.assertEqual(calls[0][calls[0].index('-g') + 1], '8')
        self.assertTrue(detection)

    def test_unknown_profile_never_launches_zero_gpus(self):
        rc, report, calls, _ = self.run_check(instance='unknown.test', imds='unknown.test')
        self.assertNotEqual(rc, 0)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(calls, [])

    def test_unavailable_metadata_never_guesses_eight_gpus(self):
        rc, report, calls, _ = self.run_check(instance='', imds='')
        self.assertNotEqual(rc, 0)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(calls, [])

    def test_isolation_launches_share_container_binary(self):
        rc, _, calls, _ = self.run_check(instance='p4d.24xlarge', isolation=True)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        for args in calls:
            self.assertIn('/opt/nccl-tests/build/all_reduce_perf', args)
            self.assertEqual(args[args.index('-g') + 1], '8')
            self.assertIn('--ntasks=2', args)

    def test_container_override_is_one_quoted_argument(self):
        binary = '/custom nccl/build/all_reduce_perf'
        rc, _, calls, _ = self.run_check(instance='p4d.24xlarge', binary=binary, isolation=True)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        for args in calls:
            self.assertIn(binary, args)

    def test_long_help_keeps_container_detection_and_mpi_override(self):
        rc, _, calls, _ = self.run_check(long_help=True, mpi='pmix_v4')
        self.assertEqual(rc, 0)
        self.assertIn('/opt/nccl-tests/build/all_reduce_perf', calls[0])
        self.assertIn('--mpi=pmix_v4', calls[0])

    def test_local_path_fallback_is_retained(self):
        rc, _, calls, _ = self.run_check(instance='p4d.24xlarge', local=True, isolation=True)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        for args in calls:
            self.assertIn('all_reduce_perf', args)
            self.assertFalse(any(a.startswith('--container-image') for a in args))

    def test_local_override(self):
        rc, _, calls, _ = self.run_check(instance='p4d.24xlarge', local=True, binary='custom_all_reduce_perf', isolation=True)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        for args in calls:
            self.assertIn('custom_all_reduce_perf', args)

    def test_wrapper_uses_check5_default_and_preserves_image_override(self):
        default = ('docker://public.ecr.aws#hpc-cloud/nccl-tests:'
                   'cuda13.0.2-efa1.48.0-ofiv1.19.0-ncclv2.30.4-1-testsv2.18.3')
        for container in (None, '', '/staged image/native-sm120.sqsh'):
            with self.subTest(container=container):
                rc, report, calls, _ = self.run_check(wrapper=True, container=container)
                self.assertEqual((rc, report['status']), (0, 'PASS'))
                args, = calls
                self.assertIn('--container-image=' + (container or default), args)
                self.assertIn('/opt/nccl-tests/build/all_reduce_perf', args)
                self.assertIn('--ntasks=2', args)
                self.assertEqual(args[args.index('-g') + 1], '8')

    def test_nvlink_split_is_node_local_and_scoped_to_subtest(self):
        for local in (False, True):
            with self.subTest(local=local):
                rc, _, calls, _ = self.run_check(instance='p4d.24xlarge', local=local,
                                                  isolation=True, nodes='4')
                self.assertEqual(rc, 0)
                self.assertEqual(len(calls), 3)
                nvlink, efa, main = calls
                binary = 'all_reduce_perf' if local else '/opt/nccl-tests/build/all_reduce_perf'
                self.assertEqual(nvlink[nvlink.index('env'):nvlink.index(binary)],
                                 ['env', 'NCCL_TESTS_SPLIT_MASK=0xffffffff',
                                  'NCCL_P2P_LEVEL=NVL', 'NCCL_NET=Socket'])
                for args in calls:
                    self.assertIn('--nodes=4', args)
                    self.assertIn('--ntasks=4', args)
                    self.assertIn('--ntasks-per-node=1', args)
                    self.assertIn('--mpi=pmix', args)
                    self.assertEqual(args[args.index('-g') + 1], '8')
                for args in (efa, main):
                    self.assertFalse(any('NCCL_TESTS_SPLIT' in arg for arg in args))

    def test_non_nvlink_profiles_skip_only_nvlink_subtest(self):
        for instance in ('g7.48xlarge', 'g6.48xlarge'):
            with self.subTest(instance=instance):
                rc, _, calls, _ = self.run_check(instance=instance, isolation=True)
                self.assertEqual(rc, 0)
                efa, main = calls
                self.assertIn('256M', efa)
                self.assertIn('128M', main)
                for args in calls:
                    self.assertFalse(any('NCCL_TESTS_SPLIT' in arg for arg in args))
                    self.assertIn('/opt/nccl-tests/build/all_reduce_perf', args)

    def test_missing_binary_is_not_a_pass(self):
        rc, report, _, _ = self.run_check(exit_code=127)
        self.assertNotEqual(rc, 0)
        self.assertEqual(report['status'], 'FAIL')

    def test_one_node_still_skips(self):
        rc, report, calls, _ = self.run_check(nodes='1')
        self.assertEqual((rc, report['status'], calls), (0, 'SKIP', []))

    def test_existing_profiles_and_exact_matching(self):
        for instance, expected in [('p4d.24xlarge', '8|4|true|efa'),
                                   ('p5en.48xlarge', '8|16|true|efa'),
                                   ('p6-b200.48xlarge', '8|8|true|efa'),
                                   ('g7.48xlarge', '8|2|false|efa'),
                                   ('g7x48xlarge', '0|0|false|efa')]:
            with self.subTest(instance=instance):
                p = subprocess.run(
                    ['bash', '-c', 'curl() { return 1; }; ec2-metadata() { return 1; }; '
                     'source "$1"; load_instance_profile; '
                     'printf "%s|%s|%s|%s" "$EXPECTED_GPU_COUNT" "$EXPECTED_EFA_COUNT" '
                     '"$NVLINK_EXPECTED" "$EFA_PROVIDER"', 'profile-test',
                     str(SUITE / 'lib/common.sh')],
                    env=dict(os.environ, INSTANCE_TYPE=instance),
                    text=True, capture_output=True, timeout=10)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertEqual(p.stdout, expected)

    def test_sibling_gpu_check_rejects_missing_g7_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Stop before kernel-log reads: seven fixture GPUs must fail count validation.
            for name, body in {
                'nvidia-smi': 'printf "fixture GPU\\n%.0s" {1..7}',
                'curl': 'exit 1',
                'ec2-metadata': 'exit 1',
                'journalctl': 'exit 0',
                'dmesg': 'exit 0',
                'cat': 'if [[ "$1" = /sys/devices/virtual/dmi/id/product_name ]]; then exit 1; else exec /bin/cat "$@"; fi',
            }.items():
                p = root / name
                p.write_text('#!/bin/bash\n' + body + '\n')
                p.chmod(0o755)
            p = subprocess.run(['bash', str(SUITE / 'checks/0-nvidia-smi-check.sh')],
                               env=dict(os.environ, PATH=f'{root}:/usr/bin:/bin',
                                        INSTANCE_TYPE='g7.48xlarge', DRY_RUN='0',
                                        RESULTS_DIR=str(root / 'results')),
                               text=True, capture_output=True, timeout=10)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('GPU count mismatch: expected=8, detected=7', p.stdout)

    def test_g7_profile_and_kubernetes_copy(self):
        expected = 'g7.48xlarge|8|2|false|efa'
        self.assertIn(expected, (SUITE / 'instance-profiles.conf').read_text().splitlines())
        self.assertIn(expected, (SUITE / 'kubernetes/manifests/01-configmap.yaml').read_text())


if __name__ == '__main__':
    unittest.main()
