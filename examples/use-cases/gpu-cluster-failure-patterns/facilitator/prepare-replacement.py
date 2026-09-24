#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Generate local pre-session replacement artifacts. Never deploys or calls AWS.

Inputs: a reviewed manifest plus coordinator/role/account provisioning details.
Output: MIME user data part (merge with existing LT, never replace PCS data),
fixed SSM document, narrow IAM policy and assignment configuration fragment.
"""
import argparse
import base64
import gzip
import hashlib
import inspect
import json
import runpy
from pathlib import Path
import re


def generate(manifest, site, bootstrap):
    # Emit the same reviewed launch defaults for the independent login host.
    # Participants source this file themselves, not through the root controller.
    helpers = runpy.run_path(str(Path(__file__).with_name('replacement-bootstrap.py')))
    participant_environment = helpers['verification_environment'](manifest)
    mode = site.get('provisioning_mode', 'boothook')
    if mode not in ('boothook', 'ssm-install'):
        raise ValueError('Unknown replacement provisioning mode')
    for key in ('node', 'region', 'slurm_bin', 'participants', 'objects',
                'release_sha256', 'maintenance_public_key', 'protected_instance_ids'):
        if key not in manifest:
            raise ValueError(f'Missing manifest field {key}')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', manifest['node']):
        raise ValueError('Invalid logical node')
    if not re.fullmatch(r'[0-9a-f]{64}', manifest['release_sha256']):
        raise ValueError('Release must be hash pinned')
    if not re.fullmatch(r'[0-9]{12}', site['account']):
        raise ValueError('Invalid account')
    if not re.fullmatch(r'[a-z0-9-]+', manifest['region']):
        raise ValueError('Invalid region')
    if not re.fullmatch(r'pcs_[a-z0-9]+', site['group_id']):
        raise ValueError('Invalid PCS group')
    if not all(re.fullmatch(r'i-[0-9a-f]{17}', i) for i in manifest['protected_instance_ids']):
        raise ValueError('Invalid protected identity')
    coordinator = site['coordinator_instance_id']
    if coordinator not in manifest['protected_instance_ids']:
        raise ValueError('Coordinator must be protected')
    if not re.fullmatch(r'AROA[A-Z0-9]{17}', site['coordinator_role_id']):
        raise ValueError('Supply the instance profile role unique RoleId, not its name')
    region, account = manifest['region'], site['account']
    prefix = f'arn:aws:ec2:{region}:{account}:instance/'
    source = {'ArnEquals': {'ec2:SourceInstanceARN': prefix + coordinator}}
    # FAS preserves the caller identity, not necessarily origin context. Match
    # the EC2-issued role session, NOT merely the role shared by all PCS nodes.
    forwarded = {'Bool': {'aws:ViaAWSService': 'true'},
                 'StringEquals': {'aws:userid': site['coordinator_role_id'] + ':' + coordinator}}
    group = {'StringEquals': {'ec2:ResourceTag/aws:pcs:compute-node-group-id': site['group_id']}, **source}
    document_name = 'AIM344Replacement-' + hashlib.sha256(
        (mode + json.dumps(manifest, sort_keys=True) + bootstrap).encode()).hexdigest()[:24]
    document_arn = f'arn:aws:ssm:{region}:{account}:document/{document_name}'
    # Immutable document has no parameters; it cannot be repurposed as a shell.
    document = {'schemaVersion': '2.2', 'description': 'Initialize reviewed AIM344 replacement only',
                'parameters': {}, 'mainSteps': [{'action': 'aws:runShellScript', 'name': 'bootstrap',
                    'inputs': {'timeoutSeconds': '1800', 'runCommand': [
                        'set -eu',
                        "printf '%s\\n' '" + hashlib.sha256(bootstrap.encode()).hexdigest()
                        + "  /usr/local/sbin/aim344-replacement-bootstrap' '"
                        + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
                        + "  /etc/aim344-replacement.json' | sha256sum -c - >&2",
                        'exec /usr/bin/python3 /usr/local/sbin/aim344-replacement-bootstrap /etc/aim344-replacement.json']}}]}
    policy = {'Version': '2012-10-17', 'Statement': [
        {'Effect': 'Allow', 'Action': ['ec2:DescribeInstances'], 'Resource': '*', 'Condition': source},
        {'Effect': 'Allow', 'Action': ['pcs:GetCluster'],
         'Resource': f'arn:aws:pcs:{region}:{account}:cluster/{site["cluster_id"]}', 'Condition': source},
        {'Effect': 'Allow', 'Action': ['pcs:GetComputeNodeGroup'],
         # PCS authorizes this read against BOTH parent cluster and exact group.
         'Resource': [f'arn:aws:pcs:{region}:{account}:cluster/{site["cluster_id"]}',
                      f'arn:aws:pcs:{region}:{account}:cluster/{site["cluster_id"]}/computenodegroup/{site["group_id"]}'],
         'Condition': source},
        {'Effect': 'Allow', 'Action': ['ec2:TerminateInstances', 'ec2:RebootInstances'], 'Resource': prefix + '*', 'Condition': group},
        {'Effect': 'Deny', 'Action': ['ec2:TerminateInstances', 'ec2:RebootInstances', 'ssm:SendCommand'],
         'Resource': [prefix + i for i in manifest['protected_instance_ids']]},
        {'Effect': 'Allow', 'Action': 'ssm:SendCommand', 'Resource': document_arn, 'Condition': source},
        {'Effect': 'Allow', 'Action': 'ssm:SendCommand', 'Resource': prefix + '*',
         'Condition': {'StringEquals': {'ssm:resourceTag/aws:pcs:compute-node-group-id': site['group_id']}, **source}},
        {'Effect': 'Allow', 'Action': 'ssm:SendCommand', 'Resource': document_arn, 'Condition': forwarded},
        {'Effect': 'Allow', 'Action': 'ssm:SendCommand', 'Resource': prefix + '*',
         'Condition': {'Bool': forwarded['Bool'], 'StringEquals': {
             **forwarded['StringEquals'], 'ssm:resourceTag/aws:pcs:compute-node-group-id': site['group_id']}}},
        {'Effect': 'Allow', 'Action': 'ssm:GetCommandInvocation', 'Resource': '*', 'Condition': source},
    ]}
    # Node profile's only new object permissions. No grant/update/pass-role or
    # group mutation permission is generated for coordinator or participants.
    objects = manifest['objects'].values()
    reads = [{'Effect': 'Allow', 'Action': ['s3:GetObject', 's3:GetObjectVersion'],
              'Resource': f'arn:aws:s3:::{item["bucket"]}/{item["key"]}'} for item in objects]
    gate = '''#!/usr/bin/env python3
import json
import hashlib
from pathlib import Path
import sys
import subprocess
import re
''' + inspect.getsource(helpers['node_reason']) + '''
cfg = json.loads(Path('/etc/aim344-replacement.json').read_text())
instance = Path('/sys/devices/virtual/dmi/id/board_asset_tag').read_text().strip()
if instance not in cfg['protected_instance_ids']:
    try:
        ready = json.loads(Path('/var/lib/aim344-replacement/ready.json').read_text())
        admitted = json.loads(Path('/var/lib/aim344-device-recovery/replacement-admitted.json').read_text())
        assert ready['instance_id'] == instance and ready['ready'] is True
        assert admitted['value'] == instance
        assert not Path('/var/lib/aim344-replacement/upgrade-pending.json').exists()
        generation = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        assert ready['manifest_sha256'] == generation
        assert admitted['manifest_sha256'] == generation
    except (OSError, ValueError, KeyError, AssertionError):
        # Slurm 25.05.9 preserves an existing nonempty reason when processing
        # Prolog failure. Set our exact identity-bound reason BEFORE failing;
        # never reinterpret the generic 'Prolog error' as our own isolation.
        control = str(Path(cfg['slurm_bin']) / 'scontrol')
        observation = subprocess.run([control, 'show', 'node', cfg['node']],
                                     check=True, capture_output=True, text=True).stdout
        reason, raw_reason = node_reason(observation)
        state = re.findall(r'\\bState=(\\S+)', observation)
        own = 'AIM344-admission-' + instance
        if (re.findall(r'\\bInstanceId=(\\S+)', observation) != [instance]
                or re.findall(r'\\bNodeName=(\\S+)', observation) != [cfg['node']]
                or len(state) != 1):
            sys.exit('AIM344 admission identity not confirmed')
        if (any(flag in state[0] for flag in ('DRAIN', 'DOWN', 'FAIL'))
                or (reason is None and raw_reason)
                or reason not in (None, '', '(null)', 'None', own)):
            # Do not rewrite an unrelated drain or claim generic Prolog failure.
            sys.exit('AIM344 admission remains isolated')
        subprocess.run([control, 'update',
                        'NodeName=' + cfg['node'], 'State=DRAIN',
                        'Reason=' + own], check=True)
        sys.exit('AIM344 replacement is not admitted')
'''
    # cloud-boothook is part of existing launch-template provisioning, before
    # PCS starts slurmd. Failure leaves the dispatcher absent/failing, not open.
    # No lifecycle-agent version assumption or group update is made at session time.
    files = {'/etc/aim344-replacement.json': (json.dumps(manifest, sort_keys=True), '0600'),
             '/usr/local/sbin/aim344-replacement-bootstrap': (bootstrap, '0700'),
             '/usr/local/sbin/aim344-replacement-admission': (gate, '0755'),
             '/opt/aim344/.prolog/dispatch.sh': ('#!/bin/bash\nset -euo pipefail\n'
                '/usr/local/sbin/aim344-replacement-admission\n'
                'exec /opt/aim344/prejob-prolog.sh\n', '0755')}
    hook = ['#cloud-boothook', '#!/bin/bash', 'set -euo pipefail',
            'install -d -m 0755 /opt/aim344/.prolog /usr/local/sbin',
            '# Root-only metadata, on host OUTPUT and forwarded/container traffic.',
            'iptables -C OUTPUT -d 169.254.169.254 -m owner ! --uid-owner 0 -j REJECT || iptables -I OUTPUT 1 -d 169.254.169.254 -m owner ! --uid-owner 0 -j REJECT',
            'iptables -C FORWARD -d 169.254.169.254 -j REJECT || iptables -I FORWARD 1 -d 169.254.169.254 -j REJECT',
            'ip6tables -C OUTPUT -d fd00:ec2::254 -m owner ! --uid-owner 0 -j REJECT || ip6tables -I OUTPUT 1 -d fd00:ec2::254 -m owner ! --uid-owner 0 -j REJECT',
            'ip6tables -C FORWARD -d fd00:ec2::254 -j REJECT || ip6tables -I FORWARD 1 -d fd00:ec2::254 -j REJECT']
    firewall = '#!/bin/bash\nset -euo pipefail\n' + '\n'.join(hook[5:9]) + '\n'
    if mode == 'boothook':
        # cloud-init.service precedes sysinit.target on the qualified PCS AMI.
        # Returning an error alone is unsafe: cloud-init can continue after a
        # failed boothook. Keep that first-boot ordering barrier closed instead.
        # SIGKILL/root administrator intervention is not a participant boundary.
        hook[2:3] = ['set -Eeuo pipefail',
                    "trap 'trap \"\" TERM INT HUP; printf \"AIM344 early metadata setup failed; boot held before user access\\n\" >&2; while :; do sleep 3600; done' ERR TERM INT HUP"]
        files['/usr/local/sbin/aim344-replacement-imds'] = (firewall, '0700')
        files['/etc/systemd/system/aim344-replacement-imds.service'] = ('''[Unit]
Description=AIM344 root-only metadata boundary before user access
DefaultDependencies=no
After=systemd-remount-fs.service
Before=sysinit.target network-pre.target shutdown.target
Conflicts=shutdown.target
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/aim344-replacement-imds
RemainAfterExit=yes
[Install]
RequiredBy=sysinit.target
''', '0644')
    for path, (text, permissions) in files.items():
        encoded = base64.b64encode(gzip.compress(text.encode(), mtime=0)).decode()
        hook += [f"test ! -L {path}", f"printf '%s' '{encoded}' | base64 -d | gzip -d > {path}", f'chown root:root {path}', f'chmod {permissions} {path}']
    if mode == 'boothook':
        hook += ['systemctl daemon-reload',
                 'systemctl enable aim344-replacement-imds.service',
                 '/bin/bash /usr/local/sbin/aim344-replacement-imds']
    hook += ['# Verify required pre-session PCS Prolog setting separately; do not modify it here.']
    prolog = '/usr/local/sbin/aim344-replacement-prolog'
    if mode == 'ssm-install':
        # A fresh LT-v1 node has none of these files. The immutable, parameterless
        # document carries the reviewed bytes, rather than executing a mutable
        # download or relying on files which only existed on the retired node.
        files[prolog] = files.pop('/opt/aim344/.prolog/dispatch.sh')
        files['/usr/local/sbin/aim344-replacement-imds'] = (firewall, '0700')
        files['/etc/systemd/system/aim344-replacement-imds.service'] = ('''[Unit]
Description=AIM344 root-only metadata boundary
Before=slurmd.service
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/aim344-replacement-imds
RemainAfterExit=yes
[Install]
RequiredBy=slurmd.service
''', '0644')
        payload = {path: {'text': text, 'mode': int(permissions, 8),
                          'sha256': hashlib.sha256(text.encode()).hexdigest()}
                   for path, (text, permissions) in files.items()}
        encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
        installer = '''import base64, gzip, hashlib, json, os, subprocess, sys
from pathlib import Path
assert os.geteuid() == 0, 'Root fixed-document installation required'
payload = json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
for item in payload.values():
    assert hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256'], 'Payload changed'
cfg = json.loads(payload['/etc/aim344-replacement.json']['text'])
# Define the pinned helper, but do not execute main or initialize anything yet.
helper = {'__name__': 'fixed_document_helper'}
exec(compile(payload['/usr/local/sbin/aim344-replacement-bootstrap']['text'], 'pinned-bootstrap', 'exec'), helper)
instance = Path('/sys/devices/virtual/dmi/id/board_asset_tag').read_text().strip()
helper['require'](helper['re'].fullmatch(r'i-[0-9a-f]{17}', instance)
                  and instance not in cfg['protected_instance_ids'], 'Protected or invalid instance')
observed = helper['run']([str(Path(cfg['slurm_bin']) / 'scontrol'), 'show', 'node', cfg['node'], '-o'])
helper['require'](helper['re'].findall(r'\\bInstanceId=(\\S+)', observed) == [instance]
                  and helper['re'].findall(r'\\bNodeName=(\\S+)', observed) == [cfg['node']],
                  'Wrong scheduler binding; no installation')
helper['trusted_directory'](helper['ROOT'])
lock = os.open(helper['ROOT'] / 'bootstrap.lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
helper['fcntl'].flock(lock, helper['fcntl'].LOCK_EX)
for path in payload:
    helper['validate_destination'](path)
def metadata():
    for path, item in payload.items():
        helper['durable'](path, item['text'], item['mode'])
    helper['run'](['/bin/bash', '/usr/local/sbin/aim344-replacement-imds'])
    helper['run'](['/usr/bin/systemctl', 'daemon-reload'])
    helper['run'](['/usr/bin/systemctl', 'enable', 'aim344-replacement-imds.service'])
    helper['run'](['/usr/bin/systemctl', 'start', 'aim344-replacement-imds.service'])
try:
    report = helper['bootstrap'](cfg, upgrade=UPGRADE_ARGUMENTS, install_metadata=metadata)
    print(json.dumps({k: v for k, v in report.items() if k != 'installed_hashes'}))
finally:
    os.close(lock)
'''.replace('PAYLOAD', repr(encoded))
        # Only the local root administrator artifact accepts original pins.
        # The participant-reachable parameterless SSM document stays fresh-only.
        upgrade_installer = installer.replace('UPGRADE_ARGUMENTS', 'sys.argv[1:]')
        upgrade_installer = "import sys\nassert len(sys.argv) == 3, 'Supply original receipt SHA256 and manifest SHA256'\n" + upgrade_installer
        installer = installer.replace('UPGRADE_ARGUMENTS', 'None')
        document['mainSteps'][0]['inputs']['runCommand'] = [
            "set -eu\n/usr/bin/python3 - <<'AIM344_FIXED_INSTALL'\n" + installer + 'AIM344_FIXED_INSTALL']
    # Include the actual generated installer/gate as well as the manifest in
    # the immutable document name. A generator-only repair needs a new name.
    document_name = 'AIM344Replacement-' + hashlib.sha256(
        json.dumps(document, sort_keys=True).encode()).hexdigest()[:24]
    for statement in policy['Statement']:
        if statement['Resource'] == document_arn:
            statement['Resource'] = f'arn:aws:ssm:{region}:{account}:document/{document_name}'
    fragment = dict(site)
    for key in ('account', 'coordinator_instance_id', 'coordinator_role_id'):
        fragment.pop(key, None)
    fragment.update(enabled=True, region=region, protected_instance_ids=manifest['protected_instance_ids'],
                    release_sha256=manifest['release_sha256'], bootstrap_document=document_name,
                    bootstrap_document_version='1', bootstrap_document_sha256='READ_BACK_DOCUMENT_SHA256_BEFORE_ENABLE')
    # The actual SSM document hash must come from GetDocument after separately
    # authorized provisioning; hashing JSON here is not a service attestation.
    fragment['enabled'] = False
    if mode == 'ssm-install':
        fragment['admission_prolog'] = prolog
    user_data = ('MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary="AIM344-REPLACEMENT"\n\n'
                 '--AIM344-REPLACEMENT\nContent-Type: text/cloud-boothook; charset="us-ascii"\n\n'
                 + '\n'.join(hook) + '\n\n--AIM344-REPLACEMENT--\n')
    if mode == 'boothook' and len(user_data.encode()) > 16384:
        raise ValueError('Replacement user-data part exceeds EC2 size; bake reviewed bootstrap into AMI')
    result = {'participant-lab.env': participant_environment,
              'bootstrap-document.json': document, 'coordinator-policy.json': policy,
            'node-object-policy.json': {'Version': '2012-10-17', 'Statement': reads},
            'assignment-replacement.json': fragment, 'replacement-boothook.sh': '\n'.join(hook) + '\n',
            'replacement-user-data.mime': user_data}
    if mode == 'ssm-install':
        # These are alternative provisioning modes, not instructions to roll LT.
        result['upgrade-existing-node.py'] = upgrade_installer
        result.pop('replacement-boothook.sh')
        result.pop('replacement-user-data.mime')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('site', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest, site = json.loads(args.manifest.read_text()), json.loads(args.site.read_text())
    artifacts = generate(manifest, site, Path(__file__).with_name('replacement-bootstrap.py').read_text())
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, value in artifacts.items():
        path = args.output / name
        path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')
        path.chmod(0o600)
        print(name, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
