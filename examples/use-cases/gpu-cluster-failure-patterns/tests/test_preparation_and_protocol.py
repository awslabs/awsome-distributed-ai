# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The G7 protocol settings of instance-type.sh and the image page-cache helper
facilitator/image-cache.py, run on the CPU host. instance-type.sh runs with its
instance-type detection replaced; image-cache.py reads and checks real temporary
files with mincore(2)."""
import contextlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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


def cached_at(path, offset):
    """Whether the page at `offset` is in the page cache, by a read that may not wait for the disk.

    preadv2 with RWF_NOWAIT returns the data only when it is already cached and
    fails with EAGAIN otherwise, so this answers without mincore(2), the call
    that image-cache.py itself uses. Returns None where the filesystem does not
    support RWF_NOWAIT.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        return bool(os.preadv(fd, [bytearray(1)], offset, os.RWF_NOWAIT))
    except BlockingIOError:
        return False
    except OSError:
        return None
    finally:
        os.close(fd)


def load_image_cache():
    import importlib.util
    spec = importlib.util.spec_from_file_location('image_cache', IMAGE_CACHE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ImageCache(unittest.TestCase):
    def run_helper(self, *argv):
        return subprocess.run([sys.executable, '-B', str(IMAGE_CACHE), *argv], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=60)

    def assert_ran(self, result):
        """The helper ran to its verdict: exit 0 or 1, no traceback, one line per image."""
        self.assertIn(result.returncode, (0, 1), result.stdout + result.stderr)
        self.assertEqual('', result.stderr)
        self.assertEqual(len(self.images), result.stdout.count(' resident '), result.stdout)

    def require_eviction(self):
        """Skip unless this filesystem drops an evicted file's pages, as seen without mincore."""
        probe = Path(self.directory.name) / 'probe'
        probe.write_bytes(os.urandom(1024 * 1024))
        evict(str(probe))
        if cached_at(str(probe), 0) is not False:
            self.skipTest('this filesystem keeps the pages after POSIX_FADV_DONTNEED, '
                          'or cannot tell without waiting')

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
        self.require_eviction()
        for path in self.images:
            evict(path)
        result = self.run_helper('check', *self.images)
        self.assert_ran(result)
        self.assertEqual(1, result.returncode, result.stdout)
        self.assertIn(f'image-cache: {self.images[0]} resident 0 of', result.stdout)
        self.assertIn('not every page is in the page cache', result.stdout)
        result = self.run_helper('read', *self.images)
        self.assert_ran(result)
        self.assertEqual(0, result.returncode, result.stdout)
        self.assertIn(f'image-cache: {self.images[0]} read {9 * 1024 * 1024 + 123} B in ', result.stdout)
        pages = (9 * 1024 * 1024 + 123 + os.sysconf('SC_PAGE_SIZE') - 1) // os.sysconf('SC_PAGE_SIZE')
        self.assertIn(f'image-cache: {self.images[0]} resident {pages} of {pages} pages (100.0 %)',
                      result.stdout)
        self.assertTrue(result.stdout.endswith('every page of every image is in the page cache\n'))
        self.assertEqual(0, self.run_helper('check', *self.images).returncode)

    def test_a_partly_read_image_fails_the_check(self):
        self.require_eviction()
        # The first 1 MiB only, with read-ahead off: the rest of the image stays out.
        self.assertEqual(0, self.run_helper('read', self.images[0]).returncode)
        evict(self.images[1])
        fd = os.open(self.images[1], os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            os.pread(fd, 1024 * 1024, 0)
        finally:
            os.close(fd)
        last = 5 * 1024 * 1024 - 1
        if not (cached_at(self.images[1], 0) is True and cached_at(self.images[1], last) is False):
            self.skipTest('this filesystem did not keep exactly the partial read in the page cache')
        result = self.run_helper('check', *self.images)
        self.assert_ran(result)
        pages = 5 * 1024 * 1024 // os.sysconf('SC_PAGE_SIZE')
        match = re.search(rf'{re.escape(self.images[1])} resident (\d+) of {pages} pages \(([0-9.]+) %\)',
                          result.stdout)
        self.assertIsNotNone(match, result.stdout)
        self.assertGreater(int(match.group(1)), 0, result.stdout)
        self.assertLess(int(match.group(1)), pages, result.stdout)
        # Short of every page, the percentage never rounds up to 100.0.
        self.assertLess(float(match.group(2)), 100.0)
        self.assertEqual(1, result.returncode, result.stdout)

    def test_an_account_that_cannot_write_the_image_is_refused(self):
        if os.geteuid() == 0:
            self.skipTest('root may read the residency of any file')
        # Root-owned and not writable by this account, like /opt/aim344/*.sqsh.
        path = '/etc/passwd'
        self.assertNotEqual(os.geteuid(), os.stat(path).st_uid)
        self.assertFalse(os.access(path, os.W_OK))
        for action in ('check', 'read'):
            result = self.run_helper(action, path)
            self.assertNotEqual(0, result.returncode, result.stdout)
            self.assertIn('run image-cache.py as root', result.stderr)
            self.assertNotIn(' resident ', result.stdout)
            self.assertNotIn(' read ', result.stdout)

    def test_the_caller_rule_follows_owner_write_access_and_root(self):
        module = load_image_cache()
        path = self.images[0]
        owned = os.stat(path)
        other = os.stat_result((owned.st_mode, owned.st_ino, owned.st_dev, owned.st_nlink,
                                owned.st_uid + 1, owned.st_gid, owned.st_size,
                                owned.st_atime, owned.st_mtime, owned.st_ctime))
        cases = ((owned.st_uid, False, owned, False), (owned.st_uid, False, other, True),
                 (owned.st_uid, True, other, False), (0, False, other, False))
        for euid, writable, info, refused in cases:
            with self.subTest(euid=euid, writable=writable, owner=info.st_uid):
                with mock.patch.object(module.os, 'geteuid', return_value=euid), \
                        mock.patch.object(module.os, 'access', return_value=writable):
                    if refused:
                        with self.assertRaisesRegex(OSError, 'run image-cache.py as root'):
                            module.require_trusted_caller(path, info)
                    else:
                        module.require_trusted_caller(path, info)

    def run_main_with(self, module, action, residency):
        """Run main() with read_image and resident_pages replaced; `residency` maps state to vectors."""
        events = []
        state = {'read': []}

        def read_image(path):
            events.append(('read', path))
            state['read'].append(path)
            return 100

        def resident_pages(path):
            events.append(('check', path))
            return residency(path, state['read'])

        output = io.StringIO()
        with mock.patch.object(module, 'read_image', read_image), \
                mock.patch.object(module, 'resident_pages', resident_pages), \
                mock.patch.object(module, 'require_trusted_caller', lambda path, info: None), \
                mock.patch.object(module.os, 'stat', lambda path: None), \
                contextlib.redirect_stdout(output):
            status = module.main([action, 'a.sqsh', 'b.sqsh'])
        return status, events, output.getvalue()

    def test_read_reads_every_image_before_it_checks_any(self):
        module = load_image_cache()
        status, events, _ = self.run_main_with(module, 'read', lambda path, read: (10, 10))
        self.assertEqual(0, status)
        self.assertEqual([('read', 'a.sqsh'), ('read', 'b.sqsh'),
                          ('check', 'a.sqsh'), ('check', 'b.sqsh')], events)

    def test_an_image_evicted_while_the_next_is_read_fails_the_read(self):
        # Reading b evicts half of a, as under memory pressure.
        def residency(path, read):
            if path == 'a.sqsh':
                return 10, 5 if 'b.sqsh' in read else 10
            return 10, 10
        module = load_image_cache()
        status, _, output = self.run_main_with(module, 'read', residency)
        self.assertEqual(1, status, output)
        self.assertIn('image-cache: a.sqsh resident 5 of 10 pages (50.0 %)', output)
        self.assertIn('not every page is in the page cache', output)

    def test_check_verdicts_from_supplied_residency(self):
        module = load_image_cache()
        cases = (((2000, 2000), (10, 10), 0, '(100.0 %)'),
                 ((2000, 1999), (10, 10), 1, '(99.9 %)'),
                 ((2000, 0), (10, 10), 1, '(0.0 %)'),
                 ((2000, 2000), (10, 9), 1, '(90.0 %)'))
        for a, b, expected, shown in cases:
            with self.subTest(a=a, b=b):
                status, events, output = self.run_main_with(
                    module, 'check', lambda path, read: a if path == 'a.sqsh' else b)
                self.assertEqual(expected, status, output)
                self.assertIn(shown, output)
                self.assertFalse(any(kind == 'read' for kind, _ in events))

    def test_a_missing_or_empty_image_is_an_error(self):
        empty = Path(self.directory.name) / 'empty.sqsh'
        empty.write_bytes(b'')
        for path in (str(empty), str(Path(self.directory.name) / 'absent.sqsh')):
            result = self.run_helper('check', path)
            self.assertNotEqual(0, result.returncode)
            self.assertIn('Error', result.stderr)

    def test_the_defaults_are_the_two_lab_images(self):
        module = load_image_cache()
        self.assertEqual(('/opt/aim344/aim344.sqsh', '/opt/aim344/nccl-baseline.sqsh'), module.IMAGES)

    def test_node_preparation_reads_both_images(self):
        source = (LAB / 'facilitator' / 'prepare-compute.sh').read_text()
        self.assertIn('python3 "$lab/facilitator/image-cache.py" read '
                      '"$stage/aim344.sqsh" "$stage/nccl-baseline.sqsh"', source)


if __name__ == '__main__':
    unittest.main()
