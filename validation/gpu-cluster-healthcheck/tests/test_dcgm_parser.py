# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Parser regression fixtures; these tests are not hardware qualification.

Ported from fix/aim344-g7-device-qualification (5ae1845, Keita Watanabe).
Field names follow NVIDIA/DCGM nvvs/include/NvvsJsonStrings.h and
dcgmi/Diag.cpp (status strings Pass/Skip/Fail/Not Run); error_severity values
follow dcgmlib/dcgm_errors.h dcgmErrorSeverity_t.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SUITE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('parser', SUITE / 'lib/parse-dcgm-results.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


def data(status='Pass', **extra):
    return {'DCGM Diagnostic': {'test_categories': [{'category': 'Hardware', 'tests': [
        {'name': 'memory', 'results': [dict(entity_id=7, entity_group='GPU', status=status, **extra)],
         'test_summary': {'status': status}}]}]}}


class Classification(unittest.TestCase):
    def test_modern_entities_are_parsed(self):
        result = parser.classify_results(data(), 2)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['test_summary'][0]['gpu_details'][0]['gpu_id'], 7)

    def test_empty_and_runtime_error_fail(self):
        for item in ({}, {'DCGM Diagnostic': {}}, {'runtime_error': 'connection failed'}, []):
            with self.subTest(item=item):
                self.assertEqual(parser.classify_results(item, 4)['severity'], 'RESET')

    def test_skipped_is_not_pass(self):
        self.assertEqual(parser.classify_results(data('Skip'), 4)['status'], 'WARN')

    def test_unknown_status_fails(self):
        self.assertEqual(parser.classify_results(data('Unexpected'), 4)['status'], 'FAIL')

    def test_dcgm_severity_is_not_legacy_warning_level(self):
        for value, expected in ((1, 'MONITOR'), (2, 'ISOLATE'), (6, 'RESET'), (5, 'RESET')):
            with self.subTest(value=value):
                result = parser.classify_results(
                    data('Fail', warnings=[{'error_severity': value, 'warning': 'fixture'}]), 4)
                self.assertEqual(result['severity'], expected)

    def test_global_error_severity_is_preserved(self):
        item = data()
        item['global_errors'] = [{'error_severity': 2, 'warning': 'fixture'}]
        self.assertEqual(parser.classify_results(item, 4)['severity'], 'ISOLATE')

    def test_summary_only_cannot_pass(self):
        item = data()
        item['DCGM Diagnostic']['test_categories'][0]['tests'][0]['results'] = []
        self.assertEqual(parser.classify_results(item, 4)['status'], 'FAIL')

    def test_legacy_shape_still_passes(self):
        item = data()
        item['DCGM GPU Diagnostic'] = item.pop('DCGM Diagnostic')
        self.assertEqual(parser.classify_results(item, 2)['status'], 'PASS')

    def test_summary_failure_is_retained(self):
        item = data()
        item['DCGM Diagnostic']['test_categories'][0]['tests'][0]['test_summary']['status'] = 'Fail'
        self.assertEqual(parser.classify_results(item, 4)['severity'], 'ISOLATE')

    # Integration-necessary additions beyond the ported fixtures.
    def test_legacy_string_warnings_do_not_raise(self):
        result = parser.classify_results(data('Pass', warnings=['legacy text']), 2)
        self.assertEqual(result['status'], 'WARN')
        self.assertEqual(result['warnings'][0]['raw'], 'legacy text')

    def test_not_run_test_without_records_is_warn_not_pass(self):
        item = data()
        item['DCGM Diagnostic']['test_categories'][0]['tests'].append({'name': 'eud'})
        result = parser.classify_results(item, 4)
        self.assertEqual(result['status'], 'WARN')
        self.assertEqual(result['test_summary'][1]['status'], 'WARN')

    def test_trailing_text_after_json_is_tolerated(self):
        value = parser.parse_dcgm_json('note\n{"DCGM Diagnostic": {}}\ntrailing')
        self.assertEqual(value, {'DCGM Diagnostic': {}})


class CheckScripts(unittest.TestCase):
    """Check 1 / Check 4 result handling with dcgmi and the host mocked.

    Ported behavior from 5ae1845: a nonzero dcgmi exit never passes on
    passing JSON, the parsed report is a separate file (check-*.json stays
    the log_result record), and Check 4 refuses to kill an externally
    managed dcgm-exporter.
    """

    def run_check(self, script, dcgmi_exit, exporter_pids=''):
        payload = json.dumps(data())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = {
                'dcgmi': f"printf '%s\\n' '{payload}'; exit {dcgmi_exit}",
                'pgrep': f'''case "$*" in
*nv-hostengine*) echo 4242;;
*dcgm-exporter*) [ -n "{exporter_pids}" ] && echo "{exporter_pids}" || exit 1;;
*) exit 1;;
esac''',
                'nvidia-smi': '''case "$*" in
*mig.mode.current*) printf '0, Disabled\\n';;
*) exit 0;;
esac''',
                'systemctl': 'exit 1',
                'curl': 'exit 92',
                'ec2-metadata': 'exit 92',
            }
            for name, body in commands.items():
                executable = root / name
                executable.write_text('#!/bin/sh\n' + body + '\n')
                executable.chmod(0o755)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(('SLURM_', 'EXPECTED_', 'INSTANCE_',
                                        'HEALTHCHECK_', 'DRY_RUN', 'DCGM_'))}
            env.update(PATH=f'{root}:/usr/bin:/bin', INSTANCE_TYPE='g7.48xlarge',
                       RESULTS_DIR=str(root / 'results'))
            result = subprocess.run(['bash', str(SUITE / 'checks' / script)],
                                    env=env, text=True, capture_output=True, timeout=30)
            results = root / 'results'
            record = json.loads((results / f'check-{script[:-3]}.json').read_text())
            files = sorted(p.name for p in results.iterdir())
            return result.returncode, record, files

    def test_passing_json_with_zero_exit_passes(self):
        code, record, files = self.run_check('1-dcgm-diag-l2.sh', 0)
        self.assertEqual((code, record['status']), (0, 'PASS'))
        self.assertIn('dcgm-l2-parsed.json', files)
        self.assertIn('dcgm-l2-raw.json', files)
        # The parser verdict lives in its own file; the check record only
        # holds what log_result wrote.
        self.assertNotIn('test_summary', record)

    def test_nonzero_dcgmi_exit_with_passing_json_fails(self):
        for script in ('1-dcgm-diag-l2.sh', '4-dcgm-diag-l4.sh'):
            with self.subTest(script=script):
                code, record, files = self.run_check(script, 3)
                self.assertEqual((code, record['status'], record['severity']),
                                 (1, 'FAIL', 'RESET'))
                self.assertIn('exit code 3', record['details'])
                level = script[len('X-dcgm-diag-'):-3]
                self.assertIn(f'dcgm-{level}-parsed.json', files)

    def test_l4_refuses_to_kill_external_exporter(self):
        code, record, files = self.run_check('4-dcgm-diag-l4.sh', 0, exporter_pids='7777')
        self.assertEqual((code, record['status'], record['severity']), (1, 'FAIL', 'RESET'))
        self.assertIn('7777', record['details'])
        self.assertNotIn('dcgm-l4-raw.json', files)


if __name__ == '__main__':
    unittest.main()
