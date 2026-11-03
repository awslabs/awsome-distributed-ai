# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The G7 protocol settings of instance-type.sh and the image page-cache helper
facilitator/image-cache.py, run on the CPU host. instance-type.sh runs with its
instance-type detection replaced; image-cache.py reads and checks real temporary
files with mincore(2)."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

LAB = Path(__file__).resolve().parents[1]
IMAGE_CACHE = LAB / 'facilitator' / 'image-cache.py'


def protocol_environment(instance_type):
    """Run apply_g7_protocol as if on `instance_type`; return its stderr and the two variables."""
    script = (f'source {LAB / "instance-type.sh"}\n'
              f'detect_instance_type() {{ printf "%s\\n" {instance_type}; }}\n'
              'apply_g7_protocol\n'
              'printf "OFI_NCCL_PROTOCOL=%s NCCL_PROTO=%s\\n" '
              '"${OFI_NCCL_PROTOCOL-unset}" "${NCCL_PROTO-unset}"\n')
    # stdin from /dev/null: bash reads the bashrc files when its input is a socket.
    result = subprocess.run(['bash', '-euc', script], capture_output=True, text=True,
                            stdin=subprocess.DEVNULL,
                            env={'PATH': os.environ['PATH'], 'NCCL_PROTO': '^LL'}, check=True)
    return result.stderr, result.stdout.strip()


class G7Protocol(unittest.TestCase):
    def test_g7_selects_rdma_and_the_simple_protocol_and_says_so(self):
        stderr, values = protocol_environment('g7.48xlarge')
        self.assertEqual('OFI_NCCL_PROTOCOL=RDMA NCCL_PROTO=Simple', values)
        self.assertRegex(stderr, r'^host=\S+ instance_type=g7\.48xlarge '
                                 r'OFI_NCCL_PROTOCOL=RDMA NCCL_PROTO=Simple\n$')

    def test_other_instance_types_keep_their_environment(self):
        stderr, values = protocol_environment('p5.48xlarge')
        self.assertEqual('OFI_NCCL_PROTOCOL=unset NCCL_PROTO=^LL', values)
        self.assertEqual('', stderr)

    def test_both_torch_and_sweep_launchers_apply_it(self):
        for name in ('torch-node.sh', 'sweep-rank.sh'):
            self.assertIn('\napply_g7_protocol\n', (LAB / name).read_text(), name)

    def test_check5_passes_nccl_proto_into_the_container(self):
        source = (LAB / 'common.sh').read_text()
        line = next(l for l in source.splitlines() if 'pyxis_container_env=' in l)
        names = line.split('=', 1)[1].split(',')
        self.assertIn('NCCL_PROTO', names)
        self.assertIn('OFI_NCCL_PROTOCOL', names)


def evict(path):
    """Ask the kernel to drop the file's clean pages from the page cache."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


class ImageCache(unittest.TestCase):
    def run_helper(self, *argv):
        return subprocess.run([sys.executable, '-B', str(IMAGE_CACHE), *argv], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=60)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ.get('AIM344_IMAGE_CACHE_TEST_DIR'))
        self.images = []
        for name, size in (('aim344.sqsh', 9 * 1024 * 1024 + 123), ('nccl-baseline.sqsh', 5 * 1024 * 1024)):
            path = Path(self.directory.name) / name
            path.write_bytes(os.urandom(size))
            self.images.append(str(path))

    def tearDown(self):
        self.directory.cleanup()

    def test_check_fails_on_an_evicted_image_and_read_brings_every_page_back(self):
        for path in self.images:
            evict(path)
        result = self.run_helper('check', *self.images)
        if 'resident 0 of' not in result.stdout:
            self.skipTest('this filesystem keeps the pages after POSIX_FADV_DONTNEED: '
                          + result.stdout)
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertIn('not every page is in the page cache', result.stdout)
        result = self.run_helper('read', *self.images)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn(f'image-cache: {self.images[0]} read {9 * 1024 * 1024 + 123} B in ', result.stdout)
        pages = (9 * 1024 * 1024 + 123 + os.sysconf('SC_PAGE_SIZE') - 1) // os.sysconf('SC_PAGE_SIZE')
        self.assertIn(f'image-cache: {self.images[0]} resident {pages} of {pages} pages (100.0 %)',
                      result.stdout)
        self.assertTrue(result.stdout.endswith('every page of every image is in the page cache\n'))
        self.assertEqual(0, self.run_helper('check', *self.images).returncode)

    def test_a_partly_read_image_fails_the_check(self):
        # The first 1 MiB only, with read-ahead off: the rest of the image stays out.
        self.run_helper('read', self.images[0])
        evict(self.images[1])
        fd = os.open(self.images[1], os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            os.pread(fd, 1024 * 1024, 0)
        finally:
            os.close(fd)
        result = self.run_helper('check', *self.images)
        pages = 5 * 1024 * 1024 // os.sysconf('SC_PAGE_SIZE')
        match = re.search(rf'{re.escape(self.images[1])} resident (\d+) of {pages} pages \(([0-9.]+) %\)',
                          result.stdout)
        self.assertIsNotNone(match, result.stdout)
        if int(match.group(1)) in (0, pages):
            self.skipTest('this filesystem did not keep a partial read in the page cache: ' + result.stdout)
        # Short of every page, the percentage never rounds up to 100.0.
        self.assertLess(float(match.group(2)), 100.0)
        self.assertEqual(1, result.returncode, result.stdout)

    def test_a_missing_or_empty_image_is_an_error(self):
        empty = Path(self.directory.name) / 'empty.sqsh'
        empty.write_bytes(b'')
        for path in (str(empty), str(Path(self.directory.name) / 'absent.sqsh')):
            result = self.run_helper('check', path)
            self.assertNotEqual(0, result.returncode)
            self.assertIn('Error', result.stderr)

    def test_the_defaults_are_the_two_lab_images(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('image_cache', IMAGE_CACHE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(('/opt/aim344/aim344.sqsh', '/opt/aim344/nccl-baseline.sqsh'), module.IMAGES)

    def test_node_preparation_reads_both_images(self):
        source = (LAB / 'facilitator' / 'prepare-compute.sh').read_text()
        self.assertIn('python3 "$lab/facilitator/image-cache.py" read '
                      '"$stage/aim344.sqsh" "$stage/nccl-baseline.sqsh"', source)


if __name__ == '__main__':
    unittest.main()
