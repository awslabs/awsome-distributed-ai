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
import shlex
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
    with os.fdopen(fd, 'wb' if isinstance(text, bytes) else 'w') as output:
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


def verification_environment(manifest):
    """Public launch defaults from the reviewed site manifest, never root env."""
    values = dict(LAB_DIR='/opt/aim344/device-recovery/companion',
                  GPUS_PER_NODE='8', CPUS_PER_RANK='24', MEMORY_PER_NODE='0',
                  NCCL_ENROOT_IMAGE='/opt/aim344/nccl-baseline.sqsh',
                  TORCH_IMAGE='/opt/aim344/aim344.sqsh',
                  CHECKPOINT_DIR='/run/aim344-checkpoints', FI_EFA_USE_HUGE_PAGE='0')
    allowed = {'PARTITION', 'NCCL_CONTAINER', 'NCCL_TESTS_BIN', 'NCCL_MPI',
               'NCCL_TIMEOUT', 'NCCL_ISOLATION_TESTS', 'NCCL_ISOLATION_TIMEOUT',
               'AIM344_CHECK5_TIMEOUT', 'AIM344_CONTAINER_PREP_TIMEOUT',
               'NCCL_SOCKET_IFNAME', 'OMPI_MCA_pml', 'OMPI_MCA_btl',
               'OMPI_MCA_btl_tcp_if_exclude', 'PMIX_MCA_gds', 'OFI_NCCL_PROTOCOL',
               'FI_EFA_USE_HUGE_PAGE', 'FI_PROVIDER', 'NCCL_DEBUG', 'TORCH_IMAGE',
               'NCCL_ENROOT_IMAGE', 'AIM344_EXPECTED_EFA_DEVICES'}
    overrides = manifest.get('verification_environment', {})
    require(isinstance(overrides, dict) and set(overrides) <= allowed,
            'Unsupported verification environment')
    require(all(isinstance(v, str) and v and '\n' not in v for v in overrides.values()),
            'Invalid verification environment value')
    values.update(overrides)
    for key in ('NCCL_CONTAINER', 'NCCL_ENROOT_IMAGE', 'TORCH_IMAGE'):
        if key in values:
            require(values[key] in ('/opt/aim344/aim344.sqsh', '/opt/aim344/nccl-baseline.sqsh'),
                    'Verification image must be a provisioned pinned object')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', values.get('PARTITION', '')),
            'Supply assigned PARTITION in verification_environment')
    suite_root = manifest.get('check5_suite_root', '/opt/aim344-healthcheck')
    if suite_root != '/opt/aim344-healthcheck':
        require(re.fullmatch(r'/opt/aim344/healthcheck-candidate-[0-9a-f]{64}', suite_root)
                and 'check5-candidate.tgz' in manifest['objects'],
                'Candidate requires an immutable site path and pinned archive')
    values['AIM344_SUITE_ENTRY'] = suite_root + '/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh'
    # Assignment defaults survive clean sudo/login environments when sourced by
    # the participant; existing explicit caller values retain precedence.
    return ''.join(f'export {key}=${{{key}:-{shlex.quote(value)}}}\n'
                   for key, value in values.items())


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def private_json(path):
    path = Path(path)
    require(path.resolve() == path, 'Private record source path changed')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == ROOT_UID
            and info.st_mode & 0o077 == 0 and info.st_nlink == 1,
            'Untrusted private upgrade record')
    return json.loads(path.read_text())


def verify_installed(report):
    require(report.get('installed_hashes'), 'Missing installed hashes')
    for name, expected in report['installed_hashes'].items():
        path = Path(name)
        require(path.is_absolute() and path.resolve() == path, 'Installed source path changed')
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == ROOT_UID
                and info.st_mode & 0o022 == 0 and info.st_nlink == 1,
                'Untrusted installed material')
        require(digest(path) == expected, 'Installed release changed after bootstrap')


def validate_destination(path):
    """Check all existing ancestors without creating or changing any of them."""
    path = Path(path)
    require(path.is_absolute(), 'Absolute installation path required')
    for candidate in (path, *path.parents):
        if candidate.exists() or candidate.is_symlink():
            info = candidate.lstat()
            require(info.st_uid == ROOT_UID and info.st_mode & 0o022 == 0
                    and (stat.S_ISDIR(info.st_mode) or
                         (candidate == path and stat.S_ISREG(info.st_mode) and info.st_nlink == 1)),
                    'Untrusted existing installation destination')
        if candidate == Path('/'):
            break


def node_reason(node):
    """Return (body, raw), preserving untrusted Reason text for exact checks.

    Live queries use Slurm's multiline format: fields start with three spaces,
    reason continuations with ten (node_info.c). Never split on field-shaped
    words inside Reason. One-line compatibility accepts only the captured
    annotated format with a unique, terminal EC2 identity pair; other ambiguous
    one-line suffixes stay in raw and cannot establish ownership.
    """
    annotation = (r' \[[^\s@\[\]]+@[0-9]{4}-(?:0[1-9]|1[0-2])-'
                  r'(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):'
                  r'[0-5][0-9]:[0-5][0-9]\]')
    text = node.rstrip('\n')
    if '\n' in text:
        reasons = re.findall(r'^   Reason=(.*(?:\n          [^\n]*)*)', text, re.M)
    else:
        reasons = re.findall(r'(?:^|\s)Reason=(.*)', text)
    if len(reasons) != 1:
        return None, text if re.search(r'(?:^|\s)Reason=', text) else ''
    raw = reasons[0]
    if '\n' not in text:
        # A field token alone is not a boundary. Require the final annotation
        # and the exact recorded serializer suffix, not arbitrary known fields.
        suffix = re.fullmatch(r'(.*' + annotation + r') InstanceId=i-[0-9a-f]{17}'
                              r' InstanceType=[A-Za-z0-9.-]+ *', raw)
        if suffix and len(re.findall(r'(?:^|\s)InstanceId=', text)) == 1:
            raw = suffix[1]
    body = re.sub(annotation + r'\Z', '', raw, count=1)
    return body, raw


def isolated(manifest, instance):
    control = str(Path(manifest['slurm_bin']) / 'scontrol')
    node = run([control, 'show', 'node', manifest['node']])
    reason, raw_reason = node_reason(node)
    states = re.findall(r'\bState=(\S+)', node)
    require(re.findall(r'\bInstanceId=(\S+)', node) == [instance]
            and re.findall(r'\bNodeName=(\S+)', node) == [manifest['node']]
            and len(states) == 1 and set(states[0].split('+')) in
            ({'IDLE', 'DRAIN'}, {'IDLE', 'DRAIN', 'CLOUD'})
            and reason == 'AIM344-admission-' + instance,
            'Upgrade requires idle identity-bound admission isolation; Reason=' + repr(raw_reason))
    require(not run([str(Path(manifest['slurm_bin']) / 'squeue'), '-h', '-w', manifest['node']]),
            'Upgrade refuses queued or active jobs')
    uids = {str(p['uid']) for p in manifest['participants']}
    require(not uids.intersection(run(['/bin/ps', '-eo', 'uid=']).split()),
            'Upgrade refuses participant processes')


def existing_preflight(manifest, instance, upgrade=None):
    """Read-only authorization; caller holds bootstrap.lock through publication.

    Explicit local root upgrade pins BOTH original files. A root-owned backup
    is the retry authority only for this exact target and original identity.
    """
    receipt = ROOT / 'ready.json'
    if not upgrade:
        require(not (ROOT / 'upgrade-pending.json').exists(), 'Explicit upgrade retry required')
        if not receipt.exists():
            return None
        report = private_json(receipt)
        require(report.get('ready') is True and report['instance_id'] == instance
                and report['release'] == manifest['release_sha256'],
                'Existing bootstrap belongs to different identity/release')
        require(report.get('manifest_sha256') == manifest_digest(manifest),
                'Bootstrap manifest changed; explicit reprovisioning required')
        verify_installed(report)
        return report
    require(os.geteuid() == 0 and len(upgrade) == 2
            and all(re.fullmatch(r'[0-9a-f]{64}', pin) for pin in upgrade),
            'Root upgrade requires original receipt and manifest SHA256')
    backup = ROOT / ('upgrade-' + manifest_digest(manifest))
    old_receipt = backup / 'ready.json' if backup.exists() else receipt
    old_manifest = backup / 'manifest.json' if backup.exists() else Path('/etc/aim344-replacement.json')
    old, previous = private_json(old_receipt), private_json(old_manifest)
    require(digest(old_receipt) == upgrade[0] and digest(old_manifest) == upgrade[1],
            'Original upgrade receipt/manifest pin mismatch')
    require(old.get('ready') is True and old.get('instance_id') == instance
            and old.get('node') == manifest['node']
            and old.get('release') == previous['release_sha256']
            and previous['objects']['companion.tgz']['sha256'] == old['release'],
            'Original upgrade identity/release mismatch')
    require(old.get('host_key') == ' '.join(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text().split()[:2]),
            'Original host key changed')
    require(old.get('manifest_sha256', manifest_digest(previous)) == manifest_digest(previous),
            'Original receipt manifest mismatch')
    for key in ('node', 'region', 'slurm_bin', 'participants', 'maintenance_public_key'):
        require(previous[key] == manifest[key], 'Upgrade cannot change identity/site: ' + key)
    pending = ROOT / 'upgrade-pending.json'
    if backup.exists() and not pending.exists():
        current = private_json(receipt)
        if (current.get('ready') is True and current.get('instance_id') == instance
                and current.get('manifest_sha256') == manifest_digest(manifest)):
            verify_installed(current)
            return {'complete': current}
    isolated(manifest, instance)
    if pending.exists():
        require(private_json(pending) == {'target': manifest_digest(manifest),
                'original_receipt': upgrade[0], 'original_manifest': upgrade[1],
                'instance_id': instance}, 'Different upgrade already pending')
    else:
        verify_installed(old)
        require(digest(receipt) == upgrade[0], 'Current receipt differs from original')
        # Legacy receipts did not cover installed helper/config copies. Bind
        # those to the validated source and same local identity before writing.
        lab = Path('/opt/aim344/device-recovery/companion')
        for source, target in (('facilitator/maintenance.py', '/usr/local/sbin/aim344-maintenance'),
                               ('facilitator/device-fault.sh', '/usr/local/sbin/aim344-device-fault'),
                               ('facilitator/restore-runtime.sh', '/var/lib/aim344-device-recovery/restore-runtime.sh'),
                               ('prejob-prolog.sh', '/opt/aim344/prejob-prolog.sh')):
            verify_installed({'installed_hashes': {
                target: old['installed_hashes'].get(str(lab / source))}})
        require(private_json('/etc/aim344-device-fault.json') == inventory(previous, instance),
                'Current device inventory/config changed')
        config = private_json('/etc/aim344-maintenance.json')
        require(config.get('slurm_bin') == previous['slurm_bin']
                and config.get('participants') == [p['name'] for p in previous['participants']]
                and config.get('staging_policy') == 'root-directory'
                and config.get('staging_image_sha256') == {n: previous['objects'][n]['sha256']
                    for n in ('aim344.sqsh', 'nccl-baseline.sqsh')},
                'Current maintenance site/image configuration changed')
    # Never recreate accounts, credentials or unrelated runtime originals.
    for entry in manifest['participants']:
        user, group = pwd.getpwnam(entry['name']), grp.getgrgid(entry['gid'])
        require(user.pw_uid == entry['uid'] and user.pw_gid == entry['gid']
                and group.gr_name == entry['name'], 'Existing participant account mismatch')
        require(not any(g.gr_name in ('sudo', 'wheel', 'docker', 'lxd') and entry['name'] in g.gr_mem
                        for g in grp.getgrall()), 'Participant has privileged group membership')
    auth = Path('/root/.ssh/authorized_keys')
    require(not auth.is_symlink() and auth.resolve() == auth and
            'restrict,command="/usr/local/sbin/aim344-maintenance" ' + manifest['maintenance_public_key']
            in auth.read_text().splitlines(), 'Existing maintenance credential missing or changed')
    return {'old': old, 'previous': previous, 'backup': backup}


def begin_upgrade(manifest, instance, upgrade, context):
    isolated(manifest, instance)
    backup = context['backup']
    if not backup.exists():
        # Publish the complete backup directory in one rename; a killed copy
        # leaves only a disposable temporary directory, never retry authority.
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temp = Path(temporary)
            durable(temp / 'ready.json', (ROOT / 'ready.json').read_bytes())
            durable(temp / 'manifest.json', Path('/etc/aim344-replacement.json').read_bytes())
            os.rename(temp, backup)
        directory = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    durable(ROOT / 'upgrade-pending.json', json.dumps({'target': manifest_digest(manifest),
        'original_receipt': upgrade[0], 'original_manifest': upgrade[1], 'instance_id': instance}))
    # Old admission gates also see ready=False. Preserve the original receipt,
    # never remove it to pretend this is fresh initialization.
    durable(ROOT / 'ready.json', json.dumps({'instance_id': instance, 'ready': False,
            'manifest_sha256': manifest_digest(manifest), 'upgrade_from': upgrade[0]}))
    durable('/var/lib/aim344-device-recovery/replacement-admitted.json',
            json.dumps({'value': None, 'operation': 'upgrade-not-admitted'}))


def bootstrap(manifest, upgrade=None, install_metadata=None):
    launch_environment = verification_environment(manifest)
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
    if not stage.exists():
        require(not upgrade, 'Existing upgrade staging missing')
        stage.mkdir(mode=0o755, parents=True, exist_ok=True)
    require(stage.lstat().st_uid == ROOT_UID and not stage.is_symlink()
            and stage.stat().st_mode & 0o022 == 0 and stage.resolve() == stage,
            'Untrusted staging directory')
    require(run(['/usr/bin/findmnt', '-n', '-o', 'TARGET', '-T', str(stage)]) == '/',
            'Replacement staging requires the root filesystem, not a mounted source')
    receipt = ROOT / 'ready.json'
    context = existing_preflight(manifest, instance, upgrade)
    if upgrade and context and 'complete' in context:
        return context['complete']
    if context and not upgrade:
        if install_metadata:
            install_metadata()
        return context
    require(re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,2}', manifest['maintenance_public_key']),
            'Invalid maintenance public key')
    require(manifest['objects']['companion.tgz']['sha256'] == manifest['release_sha256'],
            'Release does not match companion pin')
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        temp = Path(temporary)
        downloads = ['companion.tgz', 'healthcheck-pinned.tgz', 'aim344.sqsh', 'nccl-baseline.sqsh']
        if manifest.get('check5_suite_root', '/opt/aim344-healthcheck') != '/opt/aim344-healthcheck':
            downloads.append('check5-candidate.tgz')
        reused = set()
        for name in downloads:
            item = manifest['objects'][name]
            require(re.fullmatch(r'[0-9a-f]{64}', item['sha256']), 'Missing object pin')
            if upgrade and name.endswith('.sqsh'):
                require(context['previous']['objects'][name] == item,
                        'Existing-node upgrade requires unchanged image source/pin')
                require(context['old']['installed_hashes'].get(str(stage / name)) == item['sha256'],
                        'Original receipt does not bind image')
                verify_installed({'installed_hashes': {str(stage / name): item['sha256']}})
                reused.add(name)
                continue
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
        # The published archive has a companion/ prefix; install its contents,
        # not the extraction parent. Keep the suite's repository-relative layout.
        extract(temp / 'companion.tgz', temp / 'companion-release')
        companion = temp / 'companion-release/companion'
        require(companion.is_dir(), 'Missing companion/ release archive prefix')
        require(all((companion / name).is_file() for name in (
            'facilitator/maintenance.py', 'facilitator/device-fault.sh', 'facilitator/restore-runtime.sh',
            'prejob-prolog.sh', '13.verify-after-recovery.sbatch', 'common.sh', 'pins.env')),
            'Incomplete companion release')
        extract(temp / 'healthcheck-pinned.tgz', temp / 'healthcheck')
        suite_entry = 'validation/gpu-cluster-healthcheck/gpu-healthcheck.sh'
        require((temp / 'healthcheck' / suite_entry).is_file(), 'Incomplete pinned healthcheck')
        if 'check5-candidate.tgz' in downloads:
            extract(temp / 'check5-candidate.tgz', temp / 'check5-candidate')
            require((temp / 'check5-candidate' / suite_entry).is_file(), 'Incomplete candidate healthcheck')
        sources = [(companion, stage / 'device-recovery/companion'),
                   (temp / 'healthcheck', Path('/opt/aim344-healthcheck'))]
        if 'check5-candidate.tgz' in downloads:
            sources.append((temp / 'check5-candidate', Path(manifest['check5_suite_root'])))
        expected_files = {str(destination / p.relative_to(source)): digest(p)
                          for source, destination in sources for p in source.rglob('*') if p.is_file()}
        for path in [*expected_files, '/etc/aim344-device-fault.json', '/etc/aim344-maintenance.json',
                     '/etc/systemd/system/aim344-enroot-parents.service',
                     '/usr/local/sbin/aim344-maintenance', '/usr/local/sbin/aim344-device-fault',
                     '/var/lib/aim344-device-recovery/restore-runtime.sh',
                     '/var/lib/aim344-device-recovery/replacement-admitted.json',
                     '/opt/aim344/prejob-prolog.sh', '/opt/aim344/device-recovery/13.verify-after-recovery.sbatch']:
            validate_destination(path)
        if upgrade:
            begin_upgrade(manifest, instance, upgrade, context)
        if install_metadata:
            install_metadata()
        # Never extract archives through a pre-existing participant-controlled tree.
        for source, destination in ((companion, stage / 'device-recovery/companion'),
                                    (temp / 'healthcheck', Path('/opt/aim344-healthcheck'))):
            install_tree(source, destination)
        if 'check5-candidate.tgz' in downloads:
            install_tree(temp / 'check5-candidate', Path(manifest['check5_suite_root']))
        images = temp / 'images'
        images.mkdir()
        for name in ('aim344.sqsh', 'nccl-baseline.sqsh', 'healthcheck-pinned.tgz'):
            if name not in reused:
                shutil.move(temp / name, images / name)
        install_tree(images, stage)
    lab = stage / 'device-recovery/companion'
    if not upgrade:
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
    durable(lab / 'lab.env', launch_environment, 0o644)
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
        expected_files[str(Path(destination))] = digest(lab / source)
    public = manifest['maintenance_public_key']
    require(re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,2}', public), 'Invalid maintenance public key')
    ssh = Path('/root/.ssh')
    ssh.mkdir(mode=0o700, exist_ok=True)
    require(not ssh.is_symlink(), 'Symlink SSH directory')
    auth = ssh / 'authorized_keys'
    require(not auth.is_symlink(), 'Symlink authorized_keys')
    existing = auth.read_text() if auth.exists() else ''
    line = 'restrict,command="/usr/local/sbin/aim344-maintenance" ' + public
    if not upgrade and line not in existing.splitlines():
        durable(auth, existing.rstrip('\n') + '\n' + line + '\n')
    run(['/usr/local/sbin/aim344-device-fault', 'inspect'])
    if not upgrade:
        run(['/bin/bash', '/var/lib/aim344-device-recovery/restore-runtime.sh', fault['participant_user'], str(stage)])
    host_key = ' '.join(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text().split()[:2])
    # Compare to the verified downloaded bytes, not hashes invented from the
    # destination after a partial/corrupt installation. lab.env is site-generated.
    expected_files[str(lab / 'lab.env')] = hashlib.sha256(launch_environment.encode()).hexdigest()
    verify_installed({'installed_hashes': expected_files})
    roots = {lab, Path('/opt/aim344-healthcheck'),
             Path(manifest.get('check5_suite_root', '/opt/aim344-healthcheck'))}
    hashes = {str(p): digest(p) for root in roots for p in root.rglob('*') if p.is_file()}
    verify = stage / 'device-recovery/13.verify-after-recovery.sbatch'
    hashes[str(verify)] = digest(verify)
    hashes.update(expected_files)
    hashes.update({str(stage / name): manifest['objects'][name]['sha256']
                   for name in ('aim344.sqsh', 'nccl-baseline.sqsh')})
    report = {'instance_id': instance, 'node': manifest['node'], 'release': manifest['release_sha256'],
              'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
              'installed_hashes': hashes,
              'host_key': host_key, 'ready': True}
    verify_installed(report)
    if upgrade:
        isolated(manifest, instance)
    durable(receipt, json.dumps(report))
    if upgrade:
        (ROOT / 'upgrade-pending.json').unlink()
        directory = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
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
