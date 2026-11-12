#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Write the exercise target's configuration. Run as root on the target node,
after prepare-compute.sh:

    sudo python3 facilitator/prepare-target.py <slurm-node> ubuntu '<maintenance public key>'

The maintenance public key is the contents of /root/.ssh/aim344-maintenance.pub
on the login node. install-participant-control.sh prints this command with the
key filled in. This script reads the node's GPU, EFA and management
devices, writes /etc/aim344-device-fault.json and /etc/aim344-maintenance.json
for the participant, installs facilitator/maintenance.py as
/usr/local/sbin/aim344-maintenance and adds the key to root's authorized_keys
with that forced command. It prints the device helper's inventory, whose
efa_rdma_device is the AIM344_EFA_DEVICE value for lab.env, and the node's SSH
host key, for root's known_hosts on the login node.

AIM344_SLURM_BIN names the Slurm client directory (default
/opt/aws/pcs/scheduler/slurm-25.05/bin), AIM344_STAGE_DIR the staged images
(default /opt/aim344), and AIM344_EXPECTED_EFA_DEVICES, when set, the number of
EFA devices the launch configuration assigns to the node.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

LAB = Path(__file__).resolve().parents[1]
SUITE_ENTRY = '/opt/aim344-healthcheck/validation/gpu-cluster-healthcheck/gpu-healthcheck.sh'


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


def write(path, text, mode):
    path = Path(path)
    temporary = path.with_name(path.name + '.new')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, 'w') as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def inventory(node, slurm_bin, participant, instance):
    # Ask the kernel which path reaches EC2 metadata; multiple ENA defaults
    # with different metrics are valid and do not make that selected path ambiguous.
    routes = json.loads(run(['/usr/sbin/ip', '-j', 'route', 'get', '169.254.169.254']))
    require(isinstance(routes, list) and len(routes) == 1
            and isinstance(routes[0], dict)
            and isinstance(routes[0].get('dev'), str)
            and re.fullmatch(r'[A-Za-z0-9_-]+', routes[0]['dev']),
            'Ambiguous management route')
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
    cfg = {'instance_id': instance, 'slurm_node': node, 'slurm_bin': slurm_bin,
           'participant_user': participant, 'participant_users': [participant],
           'gpu_uuid': uuid, 'gpu_bdf': gpu, 'efa_bdf': efa.name,
           'efa_rdma_device': rdma[0].name, 'management_interface': management,
           'management_bdf': device.resolve().name}
    for kind, path in (('gpu', Path('/sys/bus/pci/devices') / gpu), ('efa', efa)):
        for field in ('vendor', 'device'):
            cfg[f'{kind}_{field}'] = (path / field).read_text().strip()
    return cfg


def maintenance_config(slurm_bin, participant, stage, expected_efa):
    config = {'slurm_bin': slurm_bin, 'participant_user': participant,
              'participants': [participant], 'stage_dir': str(stage),
              'staging_policy': 'root-directory',
              'staging_image_sha256': {name: digest(stage / name)
                                       for name in ('aim344.sqsh', 'nccl-baseline.sqsh')},
              'checks': [0, 2, 3, 6], 'suite_entry': SUITE_ENTRY}
    if expected_efa is not None:
        require(re.fullmatch(r'[1-9][0-9]?', expected_efa) and int(expected_efa) <= 32,
                'Invalid assigned EFA count')
        config['expected_efa_devices'] = int(expected_efa)
    return config


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    require(os.geteuid() == 0, 'Run this as root on the target node.')
    require(len(arguments) == 3, 'Usage: prepare-target.py <slurm-node> <participant> '
                                 "'<maintenance public key>'")
    node, participant, public = arguments
    require(re.fullmatch(r'[A-Za-z0-9_-]+', node), 'Bad Slurm node name.')
    require(re.fullmatch(r'[a-z][a-z0-9-]{0,31}', participant), 'Bad participant user name.')
    public = ' '.join(public.split()[:2])
    require(re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,2}', public), 'Invalid maintenance public key')
    slurm_bin = os.environ.get('AIM344_SLURM_BIN', '/opt/aws/pcs/scheduler/slurm-25.05/bin')
    stage = Path(os.environ.get('AIM344_STAGE_DIR', '/opt/aim344'))
    instance = Path('/sys/devices/virtual/dmi/id/board_asset_tag').read_text().strip()
    require(re.fullmatch(r'i-[0-9a-f]{17}', instance), 'Invalid instance')
    scheduler = run([str(Path(slurm_bin) / 'scontrol'), 'show', 'node', node, '-o'])
    require(re.findall(r'\bInstanceId=(\S+)', scheduler) == [instance]
            and re.findall(r'\bNodeName=(\S+)', scheduler) == [node],
            f'Slurm does not name this instance {node}')
    run(['/usr/bin/id', participant])
    write('/etc/aim344-device-fault.json',
          json.dumps(inventory(node, slurm_bin, participant, instance)), 0o600)
    write('/etc/aim344-maintenance.json',
          json.dumps(maintenance_config(slurm_bin, participant, stage,
                                        os.environ.get('AIM344_EXPECTED_EFA_DEVICES'))), 0o600)
    write('/usr/local/sbin/aim344-maintenance',
          (LAB / 'facilitator/maintenance.py').read_text(), 0o755)
    ssh = Path('/root/.ssh')
    ssh.mkdir(mode=0o700, exist_ok=True)
    keys = ssh / 'authorized_keys'
    existing = keys.read_text() if keys.exists() else ''
    line = 'restrict,command="/usr/local/sbin/aim344-maintenance" ' + public
    if line not in existing.splitlines():
        write(keys, existing.rstrip('\n') + ('\n' if existing.strip() else '') + line + '\n', 0o600)
    print(run(['/usr/local/sbin/aim344-device-fault', 'inspect']))
    host_key = ' '.join(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text().split()[:2])
    print(f'ssh host key: {host_key}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        sys.stderr.write(f'prepare-target.py: {error}\n')
        sys.exit(2)
