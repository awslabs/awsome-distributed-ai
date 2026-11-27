"""tools/progression.py prints the recorded table for the result files of the run of 2026-11-03.

After an intended change to the table, write the expected output again from the example directory:
LLM_RESULTS_DIR=tests/fixtures/spain-20261103 GPUS_PER_NODE=8 python3 tools/progression.py --markdown \\
    > tests/fixtures/spain-20261103/progression.md
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest

LAB = Path(__file__).resolve().parents[1]
FIXTURE = LAB / 'tests/fixtures/spain-20261103'


class ProgressionGoldenTest(unittest.TestCase):
    def test_the_table_of_the_recorded_runs(self):
        env = {'PATH': os.environ['PATH'], 'LLM_RESULTS_DIR': str(FIXTURE), 'GPUS_PER_NODE': '8'}
        result = subprocess.run([sys.executable, str(LAB / 'tools/progression.py'), '--markdown'], env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, (FIXTURE / 'progression.md').read_text())


if __name__ == '__main__':
    unittest.main()
