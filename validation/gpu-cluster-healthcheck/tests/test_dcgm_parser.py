"""Parser regression fixtures; these tests are not hardware qualification."""
import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('parser', Path(__file__).parents[1] / 'lib/parse-dcgm-results.py')
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
                result = parser.classify_results(data('Fail', warnings=[{'error_severity': value, 'warning': 'fixture'}]), 4)
                self.assertEqual(result['severity'], expected)
    def test_global_error_severity_is_preserved(self):
        item = data(); item['global_errors'] = [{'error_severity': 2, 'warning': 'fixture'}]
        self.assertEqual(parser.classify_results(item, 4)['severity'], 'ISOLATE')

    def test_summary_only_cannot_pass(self):
        item = data(); item['DCGM Diagnostic']['test_categories'][0]['tests'][0]['results'] = []
        self.assertEqual(parser.classify_results(item, 4)['status'], 'FAIL')
    def test_legacy_shape_still_passes(self):
        item = data(); item['DCGM GPU Diagnostic'] = item.pop('DCGM Diagnostic')
        self.assertEqual(parser.classify_results(item, 2)['status'], 'PASS')
    def test_summary_failure_is_retained(self):
        item = data(); item['DCGM Diagnostic']['test_categories'][0]['tests'][0]['test_summary']['status'] = 'Fail'
        self.assertEqual(parser.classify_results(item, 4)['severity'], 'ISOLATE')

if __name__ == '__main__': unittest.main()
