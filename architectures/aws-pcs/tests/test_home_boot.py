#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Offline home lifecycle regression: python3 tests/test_home_boot.py -v.

No mounts, network, system manager calls or root are needed. Execute the real
shell logic with sandbox paths and command doubles, real rsync, and the host's
fstab generator. Model the synchronous NFSv3 rpc-statd start edge explicitly:
this reproduces a wait dependency, not a kernel NFS mount or PCS registration.
Requires Python 3, PyYAML, rsync, systemd-fstab-generator and systemd-analyze
(Ubuntu 24.04).
"""
import configparser
import graphlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = Path("/usr/lib/systemd/system-generators/systemd-fstab-generator")
OPTIONS = "noatime,nfsvers=3,sync,nconnect=16,rsize=1048576,wsize=1048576,defaults"


class CfnLoader(yaml.SafeLoader):
    pass


def cfn_tag(loader, tag, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    else:
        value = loader.construct_mapping(node)
    return {tag: value}


CfnLoader.add_multi_constructor("!", cfn_tag)


def home_action(template):
    actions = template["Resources"]["PCSNodeGroupCompute"]["Properties"]["NodeLifecycleActions"]
    return next(a for a in actions["Stages"]["NodeBootstrapped"]
                if a.get("Name") == "mount-openzfs-home")


# Selected start dependencies from a native PCS peer's systemctl readback.
# Edges point from a job to the jobs it must wait for. cloud-init's synchronous
# lifecycle child and mount.nfs's on-demand rpc-statd start are NOT visible to
# systemd's ordering-cycle check. These are modeled edges, not a target trace.
BOOT_WAITS = {
    "cloud-init.service": {"lifecycle"},
    "network-online.target": {"cloud-init.service"},
    "sysinit.target": {"cloud-init.service"},
    "rpc-statd.service": {"network-online.target"},
    "home.mount": {"network-online.target", "rpc-statd.service"},
    "remote-fs.target": {"home.mount"},
    "slurmd.service": {"remote-fs.target", "sysinit.target"},
}


def boot_order(policy, synchronous_mount):
    waits = {job: set(deps) for job, deps in BOOT_WAITS.items()}
    waits["lifecycle"] = {"mount.nfs"} if policy == "EVERY_BOOT" else set()
    waits["mount.nfs"] = {"rpc-statd.service"} if synchronous_mount else set()
    return list(graphlib.TopologicalSorter(waits).static_order())


# Deliberately synthetic command doubles. Fail on unexpected calls rather than
# falling through to the host's mount/systemctl/curl. All files are in TMPDIR.
DOUBLE = r'''#!/usr/bin/env python3
import json, os, pathlib, shutil, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
p = pathlib.Path(os.environ["CASE_ROOT"])
with (p / "calls").open("a") as f:
    f.write(json.dumps([name, *args]) + "\n")
if name == "curl":
    print("test-token" if "PUT" in args else "us-west-2")
elif name == "mountpoint":
    sys.exit(0 if (p / "mounted").exists() else 1)
elif name == "mount":
    assert args == [str(p / "home")], args
    if os.environ.get("MOUNT_FAIL") == "1":
        sys.exit(32)
    shutil.rmtree(p / "home")
    (p / "home").symlink_to(p / "shared", target_is_directory=True)
    (p / "mounted").touch()
elif name == "systemctl":
    assert args == ["daemon-reload"], args
elif name == "sestatus":
    print("SELinux status: disabled")
elif name != "sleep":
    raise AssertionError(name)
'''


class HomeBootTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pcs-home-boot-")
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)
        self.home = self.p / "home"
        self.shared = self.p / "shared"
        for directory in (self.home / "ubuntu", self.shared / "ubuntu", self.p / "bin",
                          self.p / "etc", self.p / "stash"):
            directory.mkdir(parents=True)
        (self.home / "ubuntu/local-only").write_text("new local file")
        (self.home / "ubuntu/existing").write_text("local must lose")
        (self.shared / "ubuntu/existing").write_text("shared must win")
        (self.shared / "ubuntu").chmod(0o700)
        os.utime(self.shared / "ubuntu", (1234567890, 1234567890))
        (self.p / "etc/fstab").write_text("# unrelated entries are retained\n")
        for name in ("curl", "mount", "mountpoint", "systemctl", "sleep", "sestatus"):
            path = self.p / "bin" / name
            path.write_text(DOUBLE)
            path.chmod(0o755)
        self.env = {"PATH": str(self.p / "bin") + ":/usr/bin:/bin",
                    "CASE_ROOT": str(self.p), "TMPDIR": str(self.p)}
        # Translate only filesystem paths; leave all shell control flow intact.
        text = (ROOT / "assets/scripts/mount-openzfs-home.sh").read_text()
        replacements = {
            "/var/log/pcs-mount-openzfs-home.log": str(self.p / "mount.log"),
            "/etc/": str(self.p / "etc") + "/",
            "/tmp/home": str(self.p / "stash"),
            "/home": str(self.home),
        }
        text = re.sub(r"/var/log/pcs-mount-openzfs-home\.log|/etc/|/tmp/home|/home(?=[/\s\";\[]|$)",
                      lambda match: replacements[match.group()], text)
        self.script = self.p / "script.sh"
        self.script.write_text(text)

    def run_home(self, fs_id="fs-01234567890abcdef", fail=False):
        env = dict(self.env, MOUNT_FAIL="1" if fail else "0")
        return subprocess.run(["bash", str(self.script), fs_id], env=env,
                              capture_output=True, text=True, timeout=15)

    def calls(self, name):
        return [c for c in map(json.loads, (self.p / "calls").read_text().splitlines())
                if c[0] == name]

    def guard(self):
        path = self.p / "etc/systemd/system/slurmd.service.d/10-home-mount.conf"
        self.assertTrue(path.exists(), "missing native slurmd home requirement")
        cfg = configparser.ConfigParser()
        cfg.read(path)
        return cfg["Unit"]

    def test_first_boot_preserves_shared_home_and_mount_contract(self):
        before = (self.shared / "ubuntu").stat()
        result = self.run_home()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.home / "ubuntu/existing").read_text(), "shared must win")
        self.assertEqual((self.home / "ubuntu/local-only").read_text(), "new local file")
        after = (self.home / "ubuntu").stat()
        self.assertEqual((after.st_uid, after.st_gid, after.st_mode),
                         (before.st_uid, before.st_gid, before.st_mode))
        self.assertEqual(len(self.calls("mount")), 1)
        self.assertIn(f"fs-01234567890abcdef.fsx.us-west-2.amazonaws.com:/fsx/ {self.home} nfs {OPTIONS} 0 0",
                      (self.p / "etc/fstab").read_text())
        self.assertFalse((self.p / "stash").exists())
        self.assertEqual(self.guard()["Requires"], "home.mount")
        self.assertEqual(self.guard()["After"], "home.mount")
        self.assertEqual(self.calls("systemctl"), [["systemctl", "daemon-reload"]])

    def test_rerun_updates_fstab_without_recopying_or_remounting(self):
        self.assertEqual(self.run_home().returncode, 0)
        before = (self.home / "ubuntu").stat()
        result = self.run_home("fs-fedcba09876543210")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = (self.p / "etc/fstab").read_text().splitlines()
        self.assertEqual(sum(str(self.home) in line for line in lines), 1)
        self.assertIn("fs-fedcba09876543210.fsx.us-west-2.amazonaws.com", lines[-1])
        self.assertEqual(len(self.calls("mount")), 1)
        self.assertEqual((self.home / "ubuntu").stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(self.guard()["Requires"], "home.mount")
        # An already-correct fstab must not skip installing the requirement.
        (self.p / "etc/systemd/system/slurmd.service.d/10-home-mount.conf").unlink()
        result = self.run_home("fs-fedcba09876543210")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.guard()["Requires"], "home.mount")
        self.assertEqual(len(self.calls("mount")), 1)
        self.assertEqual((self.home / "ubuntu").stat().st_mtime_ns, before.st_mtime_ns)

    def test_first_boot_mount_failure_stays_fatal(self):
        result = self.run_home(fail=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(len(self.calls("mount")), 6)
        self.assertEqual(len(self.calls("sleep")), 5)
        self.assertFalse((self.p / "mounted").exists())
        self.assertTrue((self.p / "stash").exists())
        self.assertEqual((self.home / "ubuntu/existing").read_text(), "local must lose")

    def test_reboot_action_selection_breaks_native_wait_cycle(self):
        # Positive control: the old config and real shell mount call create the
        # modeled cycle; deleting the synchronous rpc-statd edge removes it.
        result = self.run_home()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        synchronous_mount = bool(self.calls("mount"))
        with self.assertRaises(graphlib.CycleError):
            boot_order("EVERY_BOOT", synchronous_mount)
        boot_order("EVERY_BOOT", False)
        for path in sorted((ROOT / "assets").glob("add-cng*.yaml")):
            with self.subTest(template=path.name):
                template = yaml.load(path.read_text(), Loader=CfnLoader)
                action = home_action(template)
                self.assertEqual(action["OnError"], "TERMINATE")
                self.assertEqual(action["Arguments"], [{"Ref": "FSxOpenZFSFilesystemId"}])
                self.assertIn(action["ExecutionPolicy"], ("FIRST_BOOT_ONLY", "EVERY_BOOT"))
                order = boot_order(action["ExecutionPolicy"], synchronous_mount)
                self.assertLess(order.index("cloud-init.service"), order.index("home.mount"))
                self.assertLess(order.index("home.mount"), order.index("slurmd.service"))
                actions = template["Resources"]["PCSNodeGroupCompute"]["Properties"]["NodeLifecycleActions"]
                self.assertEqual(actions["ScriptCachingPolicy"], "CACHE_ONCE")
                lustre = next(a["If"][1] for a in actions["Stages"]["NodeBootstrapped"]
                              if "If" in a and a["If"][0] == "HasLustre")
                self.assertEqual((lustre["ExecutionPolicy"], lustre["OnError"]),
                                 ("EVERY_BOOT", "TERMINATE"))

    def test_fstab_generates_required_reboot_mount_and_slurmd_guard(self):
        self.assertTrue(GENERATOR.exists(), "requires systemd-fstab-generator")
        self.assertEqual(self.run_home().returncode, 0)
        fstab = self.p / "reboot.fstab"
        fstab.write_text((self.p / "etc/fstab").read_text().replace(str(self.home), "/home"))
        output = self.p / "generated"
        output.mkdir()
        env = dict(self.env, SYSTEMD_FSTAB=str(fstab), SYSTEMD_IN_INITRD="0")
        result = subprocess.run([str(GENERATOR), str(output), str(output), str(output)],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        mount = configparser.ConfigParser()
        mount.read(output / "home.mount")
        self.assertEqual(mount["Mount"]["Where"], "/home")
        self.assertEqual(mount["Mount"]["Type"], "nfs")
        self.assertEqual(mount["Mount"]["Options"], OPTIONS)
        self.assertTrue((output / "remote-fs.target.requires/home.mount").is_symlink())
        guard = self.guard()
        # Requires plus After propagates mount activation failure to slurmd.
        # Offline verification checks the missing-unit transaction below; it
        # cannot execute a failed NFS mount or observe runtime failure handling.
        self.assertIn("home.mount", guard["After"].split())
        self.assertIn("home.mount", guard["Requires"].split())

        # Let the host's real systemd loader expand mount dependencies and load
        # the installed drop-in. Minimal offline units model the peer ordering;
        # no service is started and no running system manager is contacted.
        units = self.p / "units"
        units.mkdir()
        fixtures = {
            "cloud-init.service": "[Unit]\nDefaultDependencies=no\nBefore=network-online.target sysinit.target\n[Service]\nType=oneshot\nExecStart=/usr/bin/true\n",
            "rpc-statd.service": "[Unit]\nDefaultDependencies=no\nAfter=network-online.target\n[Service]\nType=oneshot\nExecStart=/usr/bin/true\n",
            "slurmd.service": "[Unit]\nDefaultDependencies=no\nAfter=remote-fs.target sysinit.target\n[Service]\nExecStart=/usr/bin/true\n",
        }
        for target in ("network", "network-online", "remote-fs-pre", "remote-fs",
                       "sysinit", "shutdown", "umount", "local-fs-pre"):
            fixtures[target + ".target"] = "[Unit]\nDefaultDependencies=no\n"
        for name, text in fixtures.items():
            (units / name).write_text(text)
        (units / "slurmd.service.d").symlink_to(
            self.p / "etc/systemd/system/slurmd.service.d", target_is_directory=True)
        env = dict(self.env, SYSTEMD_UNIT_PATH=f"{units}:{output}", SYSTEMD_LOG_LEVEL="debug",
                   SYSTEMD_ENVIRONMENT_GENERATOR_PATH=str(self.p / "empty-generators"))
        result = subprocess.run(["systemd-analyze", "verify", "--man=no", "--generators=no",
                                 str(units / "slurmd.service"), str(output / "home.mount"),
                                 str(units / "cloud-init.service"), str(units / "rpc-statd.service")],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        # systemd's mount loader, not the model, supplies these NFS dependencies.
        dump = result.stdout + result.stderr
        self.assertIn("Requires: home.mount", dump)
        self.assertIn("After: home.mount", dump)
        self.assertIn("After: network-online.target", dump)
        (output / "home.mount").unlink()
        result = subprocess.run(["systemd-analyze", "verify", "--man=no", "--generators=no",
                                 str(units / "slurmd.service")], env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0, "missing fstab mount must prevent slurmd start")
        self.assertIn("home.mount", result.stderr)

        # Negative control: ordering alone does NOT require the mount. The same
        # missing-unit transaction succeeds when Requires is removed.
        dropin = units / "slurmd.service.d/10-home-mount.conf"
        dropin.write_text(dropin.read_text().replace("Requires=home.mount\n", ""))
        result = subprocess.run(["systemd-analyze", "verify", "--man=no", "--generators=no",
                                 str(units / "slurmd.service")], env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
