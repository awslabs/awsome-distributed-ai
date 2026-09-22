#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Fixed-document replacement initialization; root only, no participant inputs.

The reviewed SSM document supplies a root-owned manifest. All fetches are hash
checked before execution/extraction. Runs only for the manifest's logical node.
No coordinator credential, device original or old host identity is copied.
"""
import fcntl
import grp
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path('/var/lib/aim344-replacement')
ROOT_UID = 0


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(argv):
    result = subprocess.run(argv, check=True, capture_output=True, text=True,
                            timeout=900, env={'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                                             'HOME': '/root', 'LC_ALL': 'C'})
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.stdout.strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def durable(path, text, mode=0o600):
    path = Path(path)
    trusted_directory(path.parent)
    tmp = path.with_suffix('.new')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, 'w') as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
        os.fchmod(output.fileno(), mode)
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def extract(archive, destination):
    """No links, devices, absolute names or traversal in a reviewed release."""
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        for member in members:
            name = Path(member.name)
            require(not name.is_absolute() and '..' not in name.parts
                    and (member.isfile() or member.isdir()), 'Unsafe release archive member')
        source.extractall(destination, members=members, filter='data')


def account(entry):
    name, uid, gid = entry['name'], entry['uid'], entry['gid']
    require(re.fullmatch(r'aim344-[a-z0-9-]+', name) and type(uid) is int
            and type(gid) is int and uid >= 1000 and gid >= 1000,
            'Invalid participant identity')
    try:
        group = grp.getgrgid(gid)
        require(group.gr_name == name, 'Participant gid collision')
    except KeyError:
        run(['/usr/sbin/groupadd', '-g', str(gid), name])
    try:
        user = pwd.getpwnam(name)
    except KeyError:
        try:
            pwd.getpwuid(uid)
        except KeyError:
            run(['/usr/sbin/useradd', '-m', '-u', str(uid), '-g', str(gid), '-s', '/bin/bash', name])
        else:
            raise RuntimeError('Participant uid collision')
        user = pwd.getpwnam(name)
    require(user.pw_uid == uid and user.pw_gid == gid, 'Participant account mismatch')
    require(not any(g.gr_name in ('sudo', 'wheel', 'docker', 'lxd') and name in g.gr_mem
                    for g in grp.getgrall()), 'Participant has privileged group membership')


def trusted_directory(path):
    if path == path.parent:
        return
    if not path.exists() and not path.is_symlink():
        trusted_directory(path.parent)
        path.mkdir(mode=0o755)
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == ROOT_UID
            and info.st_mode & 0o022 == 0, 'Untrusted material directory')


def install_tree(source, destination):
    """Retry a partial root-only copy without deleting or following existing names."""
    trusted_directory(destination)
    require(not destination.is_symlink() and destination.stat().st_uid == ROOT_UID
            and destination.stat().st_mode & 0o022 == 0, 'Untrusted material destination')
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            install_tree(child, target)
        else:
            require(not target.is_symlink(), 'Material destination symlink')
            if target.exists():
                info = target.lstat()
                require(stat.S_ISREG(info.st_mode) and info.st_uid == ROOT_UID
                        and info.st_mode & 0o022 == 0 and info.st_nlink == 1,
                        'Untrusted existing material')
            temporary = target.with_name(target.name + '.replacement-new')
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as output, child.open('rb') as original:
                shutil.copyfileobj(original, output)
                output.flush()
                os.fsync(output.fileno())
                os.fchmod(output.fileno(), 0o755 if child.stat().st_mode & 0o111 else 0o644)
            os.replace(temporary, target)
    directory = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def prepare_enroot():
    # Same root-owned sticky-parent boundary as the participant installer.
    parents = set()
    for line in Path('/etc/enroot/enroot.conf').read_text().splitlines():
        match = re.match(r'^ENROOT_(?:RUNTIME|CACHE|DATA)_PATH\s+(\S+)', line.strip())
        if not match:
            continue
        parts = []
        for component in match[1].split('/'):
            if component.startswith(('user-', 'group-')) or component in ('cache', 'data'):
                break
            parts.append(component)
        parents.add('/'.join(parts))
    require(parents, 'Missing enroot runtime path configuration')
    for path in sorted(parents):
        require(str(Path(path).parent) in ('/tmp', '/var/tmp'), 'Unsupported shared enroot parent')
        parent = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(parent)
            require(info.st_uid == 0 and info.st_mode & stat.S_ISVTX, 'Untrusted /tmp')
            name = Path(path).name
            try:
                os.mkdir(name, 0o1777, dir_fd=parent)
            except FileExistsError:
                pass
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                require(os.fstat(fd).st_uid == 0, 'Untrusted container cache')
                os.fchmod(fd, 0o1777)
                # Enroot creates these shared intermediates before user-N.
                # A prior root run can leave them 0700 even when the outer
                # parent is shared. Never follow or relabel a foreign entry.
                for name in ('cache', 'data'):
                    try:
                        os.mkdir(name, 0o1777, dir_fd=fd)
                    except FileExistsError:
                        pass
                    nested = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                     dir_fd=fd)
                    try:
                        require(os.fstat(nested).st_uid == 0, 'Untrusted container cache')
                        os.fchmod(nested, 0o1777)
                    finally:
                        os.close(nested)
            finally:
                os.close(fd)
        finally:
            os.close(parent)


def inventory(manifest, instance):
    routes = json.loads(run(['/usr/sbin/ip', '-j', 'route', 'show', 'default']))
    require(routes and len({r.get('dev') for r in routes}) == 1, 'Ambiguous management route')
    management = routes[0]['dev']
    device = Path('/sys/class/net') / management / 'device'
    require((device / 'driver').resolve().name == 'ena', 'Management is not ENA')
    gpu_rows = run(['/usr/bin/nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader']).splitlines()
    require(gpu_rows, 'No GPU inventory')
    uuid, bus = [x.strip() for x in gpu_rows[0].split(',')]
    gpu = bus.lower()[-12:]
    efas = [p for p in sorted(Path('/sys/bus/pci/devices').iterdir())
            if (p / 'driver').resolve().name == 'efa' and not list((p / 'net').glob('*'))]
    require(efas, 'No separate EFA device')
    efa = efas[0]
    rdma = list((efa / 'infiniband').iterdir())
    require(len(rdma) == 1 and len({gpu, efa.name, device.resolve().name}) == 3,
            'Device inventory is ambiguous or includes management')
    cfg = {'instance_id': instance, 'slurm_node': manifest['node'],
           'slurm_bin': manifest['slurm_bin'], 'participant_user': manifest['participants'][0]['name'],
           'participant_users': [p['name'] for p in manifest['participants']],
           'gpu_uuid': uuid, 'gpu_bdf': gpu, 'efa_bdf': efa.name,
           'efa_rdma_device': rdma[0].name, 'management_interface': management,
           'management_bdf': device.resolve().name}
    for kind, path in (('gpu', Path('/sys/bus/pci/devices') / gpu), ('efa', efa)):
        for field in ('vendor', 'device'):
            cfg[f'{kind}_{field}'] = (path / field).read_text().strip()
    return cfg


def bootstrap(manifest):
    instance = Path('/sys/devices/virtual/dmi/id/board_asset_tag').read_text().strip()
    require(re.fullmatch(r'i-[0-9a-f]{17}', instance), 'Invalid instance')
    require(instance not in manifest['protected_instance_ids'], 'Protected coordinator/peer')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', manifest['node']), 'Invalid logical node')
    node = run([str(Path(manifest['slurm_bin']) / 'scontrol'), 'show', 'node', manifest['node'], '-o'])
    require(re.findall(r'\bInstanceId=(\S+)', node) == [instance]
            and re.findall(r'\bNodeName=(\S+)', node) == [manifest['node']], 'Wrong scheduler binding')
    # A repeated document after qualification cannot reset fixtures/allowlists or
    # revoke admission. Return the previous receipt only for these exact bytes.
    stage = Path('/opt/aim344')
    stage.mkdir(mode=0o755, parents=True, exist_ok=True)
    require(stage.lstat().st_uid == ROOT_UID and not stage.is_symlink()
            and stage.stat().st_mode & 0o022 == 0 and stage.resolve() == stage,
            'Untrusted staging directory')
    require(run(['/usr/bin/findmnt', '-n', '-o', 'TARGET', '-T', str(stage)]) == '/',
            'Replacement staging requires the root filesystem, not a mounted source')
    receipt = ROOT / 'ready.json'
    if receipt.exists():
        report = json.loads(receipt.read_text())
        require(report['instance_id'] == instance and report['release'] == manifest['release_sha256'],
                'Existing bootstrap belongs to different identity/release')
        # Receipt is not reinitialization authority after materials disappear.
        for name, expected in report['installed_hashes'].items():
            require(digest(name) == expected, 'Installed release changed after bootstrap')
        return report
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        temp = Path(temporary)
        for name in ('companion.tgz', 'healthcheck-pinned.tgz', 'aim344.sqsh', 'nccl-baseline.sqsh'):
            item = manifest['objects'][name]
            require(re.fullmatch(r'[0-9a-f]{64}', item['sha256']), 'Missing object pin')
            # Streaming CLI operations do not support --cli-input-json.
            # Keep manifest values in individual argv elements, never a shell.
            command = ['/usr/local/bin/aws', 's3api', 'get-object',
                       '--region', manifest['region'], '--bucket', item['bucket'],
                       '--key', item['key'], '--no-cli-pager']
            if item.get('version_id'):
                command.extend(['--version-id', item['version_id']])
            run(command + [str(temp / name)])
            require(digest(temp / name) == item['sha256'], 'Downloaded material hash mismatch')
        require(manifest['objects']['companion.tgz']['sha256'] == manifest['release_sha256'],
                'Release does not match companion pin')
        extract(temp / 'companion.tgz', temp / 'companion')
        extract(temp / 'healthcheck-pinned.tgz', temp / 'healthcheck')
        # Never extract archives through a pre-existing participant-controlled tree.
        for source, destination in ((temp / 'companion', stage / 'device-recovery/companion'),
                                    (temp / 'healthcheck', Path('/opt/aim344-healthcheck')),
                                    (temp / 'healthcheck', stage / 'device-recovery/candidate-v2')):
            install_tree(source, destination)
        images = temp / 'images'
        images.mkdir()
        for name in ('aim344.sqsh', 'nccl-baseline.sqsh', 'healthcheck-pinned.tgz'):
            shutil.move(temp / name, images / name)
        install_tree(images, stage)
    lab = stage / 'device-recovery/companion'
    for entry in manifest['participants']:
        account(entry)
    prepare_enroot()
    # /tmp can be cleared on reboot. Prepare shared parents before user access,
    # never let the first participant recreate them as private 0700 directories.
    durable('/etc/systemd/system/aim344-enroot-parents.service', '''[Unit]
Description=AIM344 shared enroot parents before user access
DefaultDependencies=no
After=systemd-remount-fs.service systemd-tmpfiles-setup.service
Before=sysinit.target shutdown.target
Conflicts=shutdown.target
[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /usr/local/sbin/aim344-replacement-bootstrap --prepare-enroot
RemainAfterExit=yes
[Install]
RequiredBy=sysinit.target
''', 0o644)
    run(['/usr/bin/systemctl', 'daemon-reload'])
    run(['/usr/bin/systemctl', 'enable', 'aim344-enroot-parents.service'])
    durable(lab / 'lab.env', 'export LAB_DIR=/opt/aim344/device-recovery/companion\n'
            'export PARTITION=gpu-g7 GPUS_PER_NODE=8 CPUS_PER_RANK=24 MEMORY_PER_NODE=0\n'
            'export NCCL_ENROOT_IMAGE=/opt/aim344/nccl-baseline.sqsh\n'
            'export TORCH_IMAGE=/opt/aim344/aim344.sqsh\n'
            'export CHECKPOINT_DIR=/run/aim344-checkpoints\nexport FI_EFA_USE_HUGE_PAGE=0\n', 0o644)
    durable(stage / 'device-recovery/13.verify-after-recovery.sbatch',
            (lab / '13.verify-after-recovery.sbatch').read_text(), 0o755)
    fault = inventory(manifest, instance)
    durable('/etc/aim344-device-fault.json', json.dumps(fault))
    suite = '/opt/aim344-healthcheck/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh'
    maintenance = {'slurm_bin': manifest['slurm_bin'], 'participant_user': fault['participant_user'],
                   'participants': fault['participant_users'], 'stage_dir': str(stage),
                   'staging_policy': 'root-directory',
                   'staging_image_sha256': {name: manifest['objects'][name]['sha256']
                                            for name in ('aim344.sqsh', 'nccl-baseline.sqsh')},
                   'checks': [0, 2, 3, 6], 'suite_entry': suite}
    durable('/etc/aim344-maintenance.json', json.dumps(maintenance))
    for source, destination in (('facilitator/maintenance.py', '/usr/local/sbin/aim344-maintenance'),
                                ('facilitator/device-fault.sh', '/usr/local/sbin/aim344-device-fault'),
                                ('facilitator/restore-runtime.sh', '/var/lib/aim344-device-recovery/restore-runtime.sh'),
                                ('prejob-prolog.sh', '/opt/aim344/prejob-prolog.sh')):
        durable(destination, (lab / source).read_text(), 0o755)
    public = manifest['maintenance_public_key']
    require(re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,2}', public), 'Invalid maintenance public key')
    ssh = Path('/root/.ssh')
    ssh.mkdir(mode=0o700, exist_ok=True)
    require(not ssh.is_symlink(), 'Symlink SSH directory')
    auth = ssh / 'authorized_keys'
    require(not auth.is_symlink(), 'Symlink authorized_keys')
    existing = auth.read_text() if auth.exists() else ''
    line = 'restrict,command="/usr/local/sbin/aim344-maintenance" ' + public
    if line not in existing.splitlines():
        durable(auth, existing.rstrip('\n') + '\n' + line + '\n')
    run(['/usr/local/sbin/aim344-device-fault', 'inspect'])
    run(['/bin/bash', '/var/lib/aim344-device-recovery/restore-runtime.sh', fault['participant_user'], str(stage)])
    host_key = ' '.join(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text().split()[:2])
    hashes = {str(p): digest(p) for root in (lab, Path('/opt/aim344-healthcheck'))
              for p in root.rglob('*') if p.is_file()}
    hashes.update({str(stage / name): manifest['objects'][name]['sha256']
                   for name in ('aim344.sqsh', 'nccl-baseline.sqsh')})
    report = {'instance_id': instance, 'node': manifest['node'], 'release': manifest['release_sha256'],
              'installed_hashes': hashes,
              'host_key': host_key, 'ready': True}
    durable(receipt, json.dumps(report))
    return report


def main():
    require(os.geteuid() == 0 and len(sys.argv) == 2, 'Root fixed-document invocation required')
    if sys.argv[1] == '--prepare-enroot':
        # Fixed root-only boot/runtime action; no manifest, target or caller path.
        # Existing descriptor/ownership checks refuse participant-owned parents.
        prepare_enroot()
        return
    path = Path(sys.argv[1])
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == 0 and info.st_mode & 0o077 == 0,
            'Manifest must be a private root file')
    manifest = json.loads(path.read_text())
    ROOT.mkdir(mode=0o700, exist_ok=True)
    require(not ROOT.is_symlink() and ROOT.stat().st_uid == 0, 'Untrusted replacement directory')
    fd = os.open(ROOT / 'bootstrap.lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        report = bootstrap(manifest)
        print(json.dumps({k: v for k, v in report.items() if k != 'installed_hashes'}))
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
