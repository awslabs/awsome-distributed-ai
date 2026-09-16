#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Install a root-owned copy at /usr/local/sbin/aim344-device-fault.
set -euo pipefail
exec /usr/bin/python3 - "$@" <<'PY'
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def command(*args):
    # The external controller owns recovery if a kernel task is uninterruptible.
    # A userspace timeout alone cannot recover a D-state GPU operation.
    return subprocess.check_output(args, text=True, timeout=10).strip()


def event(action, **details):
    print(json.dumps(dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         action=action, **details)), flush=True)


parser = argparse.ArgumentParser(description='Operate only the provisioned AIM344 GPU or EFA device')
parser.add_argument('action', choices=['inspect', 'gpu-remove', 'efa-unbind', 'efa-rebind'])
parser.add_argument('--job', type=int, help='The one dedicated AIM344 job allowed on this node')
parser.add_argument('--confirm', help='Exact token printed by inspect for this action')
args = parser.parse_args()
require(os.geteuid() == 0, 'Use the root-owned installed helper through the facilitator.')
config_path = Path('/etc/aim344-device-fault.json')
st = config_path.lstat()
require(stat.S_ISREG(st.st_mode) and st.st_uid == 0 and st.st_mode & 0o077 == 0,
        'The allowlist must be a regular root-owned file with mode 0600.')
cfg = json.loads(config_path.read_text())
instance = Path('/sys/devices/virtual/dmi/id/board_asset_tag').read_text().strip()
require(instance == cfg['instance_id'], 'This instance is not the provisioned fault target.')
require(re.fullmatch(r'i-[0-9a-f]+', instance), 'Invalid instance identity.')
require(re.fullmatch(r'[A-Za-z0-9_-]+', cfg['slurm_node']), 'Invalid Slurm node name.')
for key in ['gpu_bdf', 'efa_bdf', 'management_bdf']:
    require(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', cfg[key]),
            'Invalid provisioned PCI identity.')
require(len({cfg['gpu_bdf'], cfg['efa_bdf'], cfg['management_bdf']}) == 3,
        'Fault targets must differ from the management device.')
management = Path('/sys/class/net') / cfg['management_interface'] / 'device'
require(management.resolve().name == cfg['management_bdf'], 'Management PCI identity changed.')
require((management / 'driver').resolve().name == 'ena', 'Management driver is not ENA.')
routes = json.loads(command('/usr/sbin/ip', '-j', 'route', 'show', 'default'))
require(routes and all(r.get('dev') == cfg['management_interface'] for r in routes),
        'Default management route differs from the provisioned ENA interface.')

def identity(kind, allow_unbound=False):
    bdf = cfg[kind + '_bdf']
    device = Path('/sys/bus/pci/devices') / bdf
    require(device.exists(), f'Provisioned {kind} PCI function is absent: {bdf}')
    require((device / 'vendor').read_text().strip() == cfg[kind + '_vendor'], 'PCI vendor changed.')
    require((device / 'device').read_text().strip() == cfg[kind + '_device'], 'PCI device ID changed.')
    driver = (device / 'driver').resolve().name if (device / 'driver').exists() else None
    expected = 'nvidia' if kind == 'gpu' else 'efa'
    require(driver == expected or (allow_unbound and driver is None), 'Unexpected driver binding.')
    if kind == 'efa':
        require(not list((device / 'net').glob('*')), 'Refusing an EFA function with a network interface.')
        if driver:
            rdma = sorted(p.name for p in (device / 'infiniband').glob('*'))
            require(rdma == [cfg['efa_rdma_device']], 'EFA RDMA identity changed.')
    else:
        raw = command('/usr/bin/nvidia-smi', '-i', cfg['gpu_uuid'],
                      '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader')
        uuid, bus = [v.strip() for v in raw.split(',')]
        require(uuid == cfg['gpu_uuid'] and bus.lower()[-12:] == bdf, 'GPU UUID/BDF mapping changed.')
    return device, driver


def token(action):
    target = cfg['gpu_uuid'] if action == 'gpu-remove' else cfg['efa_bdf']
    return f'{instance}/{action}/{target}'

if args.action == 'inspect':
    identity('gpu')
    identity('efa', allow_unbound=True)
    event('inspect', instance_id=instance, gpu_uuid=cfg['gpu_uuid'], gpu_bdf=cfg['gpu_bdf'],
          efa_bdf=cfg['efa_bdf'], efa_rdma_device=cfg['efa_rdma_device'],
          management_interface=cfg['management_interface'], management_bdf=cfg['management_bdf'],
          confirmation={a: token(a) for a in ['gpu-remove', 'efa-unbind', 'efa-rebind']})
    sys.exit(0)

require(args.confirm == token(args.action), 'Confirmation does not match the provisioned action and target.')
slurm = Path(cfg['slurm_bin'])
require(str(slurm).startswith('/opt/aws/pcs/scheduler/slurm-'), 'Unexpected Slurm installation.')
node = command(str(slurm / 'scontrol'), 'show', 'node', cfg['slurm_node'], '-o')
require(re.search(r'\bState=\S*DRAIN', node), 'Drain the target node before any device operation.')
require('Reason=aim344-device-recovery' in node, 'The drain reason is not this AIM344 exercise.')
jobs = command(str(slurm / 'squeue'), '-h', '-w', cfg['slurm_node'], '-o', '%i|%u|%j').splitlines()
for job in jobs:
    job_id, user, name = job.split('|')
    require(args.job is not None and job_id == str(args.job) and user == cfg['participant_user']
            and name.startswith('aim344-'), 'An unapproved job is using the target node.')
if args.job is not None:
    require(any(job.split('|')[0] == str(args.job) for job in jobs),
            'The specified dedicated job is no longer present on the target node.')
if args.action == 'gpu-remove':
    require(not jobs and args.job is None,
            'GPU removal is qualified only while idle; end the allocation first.')
    persistence = command('/usr/bin/nvidia-smi', '-i', cfg['gpu_uuid'],
                          '--query-gpu=persistence_mode', '--format=csv,noheader,nounits')
    require(persistence == 'Disabled',
            'Record and disable persistence on the selected GPU before idle removal.')
    compute = command('/usr/bin/nvidia-smi', '-i', cfg['gpu_uuid'],
                      '--query-compute-apps=pid', '--format=csv,noheader,nounits')
    require(not compute, 'The selected GPU still has a compute process.')
if args.action == 'efa-rebind':
    require(not jobs, 'End the faulted allocation before EFA rebind; old communicators are not reusable.')

kind = 'gpu' if args.action == 'gpu-remove' else 'efa'
device, driver = identity(kind, allow_unbound=args.action == 'efa-rebind')
boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
event('mutation-start', operation=args.action, instance_id=instance, bdf=device.name,
      boot_id=boot_id, allowed_job_id=args.job)
if args.action == 'gpu-remove':
    (device / 'remove').write_text('1\n')
elif args.action == 'efa-unbind':
    (device / 'driver' / 'unbind').write_text(device.name + '\n')
elif driver is None:
    Path('/sys/bus/pci/drivers/efa/bind').write_text(device.name + '\n')
event('mutation-returned', operation=args.action, instance_id=instance, bdf=device.name, boot_id=boot_id)
PY
