"""Local instance-detection controls; not hardware qualification."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SUITE = Path(__file__).resolve().parents[1]
COMMON = SUITE / 'lib/common.sh'


class InstanceDetection(unittest.TestCase):
    def test_dmi_and_metadata_fallback_ignore_environment(self):
        for dmi, expected, network in [('g7.48xlarge', 'g7.48xlarge', False),
                                       ('not an instance', 'p5.48xlarge', True),
                                       ('', 'p5.48xlarge', True)]:
            with self.subTest(dmi=dmi), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'cat').write_text('#!/bin/bash\nprintf "%s\\n" "$LOCAL_DMI"\n')
                (root / 'curl').write_text('#!/bin/bash\nprintf "call\\n" >> "$LOCAL_CALLS"\n'
                                          'case "$*" in *api/token*) printf token;; '
                                          '*) printf p5.48xlarge;; esac\n')
                for file in root.iterdir():
                    file.chmod(0o755)
                calls = root / 'calls'
                result = subprocess.run(['bash', '-c',
                    'source "$1"; INSTANCE_TYPE=forged.8xlarge; detect_instance_type', 'bash', str(COMMON)],
                    env=dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'],
                             LOCAL_DMI=dmi, LOCAL_CALLS=str(calls)), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)
                self.assertEqual(calls.exists(), network)

    def test_actual_host_dmi_with_metadata_unavailable(self):
        dmi = Path('/sys/devices/virtual/dmi/id/product_name')
        if not dmi.exists():
            self.skipTest('No live DMI on this local executor')
        result = subprocess.run(['bash', '-c',
            'source "$1"; curl() { return 7; }; ec2-metadata() { return 1; }; '
            'INSTANCE_TYPE=forged.8xlarge; detect_instance_type', 'bash', str(COMMON)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), dmi.read_text().strip())
