"""Enable or restore the partition-guarded AIM344 Prolog using the PCS API."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('action', choices=('enable', 'restore'))
p.add_argument('--cluster', required=True)
p.add_argument('--region', required=True)
p.add_argument('--state-file', type=Path, required=True)
p.add_argument('--dispatcher', default='/fsx/aim344/.prolog/dispatch.sh')
a = p.parse_args()
if not a.dispatcher.startswith('/') or any(c.isspace() for c in a.dispatcher):
    p.error('--dispatcher must be an absolute path without whitespace')


def aws(*arguments):
    return json.loads(subprocess.check_output(
        ['aws', 'pcs', *arguments, '--region', a.region, '--output', 'json', '--no-cli-pager'], text=True))


cluster = aws('get-cluster', '--cluster-identifier', a.cluster)['cluster']
if cluster['status'] != 'ACTIVE':
    raise RuntimeError(f"PCS cluster is {cluster['status']}")
current = cluster.get('slurmConfiguration', {}).get('slurmCustomSettings', [])
if a.action == 'enable':
    if a.state_file.exists():
        raise FileExistsError(f'Preserve the saved configuration: {a.state_file}')
    if any(item['parameterName'] == 'Prolog' for item in current):
        raise RuntimeError('An existing Prolog must be preserved and explicitly chained by the cluster operator')
    a.state_file.write_text(json.dumps(dict(cluster=a.cluster, region=a.region, settings=current, dispatcher=a.dispatcher), indent=2) + '\n')
    settings = [item for item in current if item['parameterName'] != 'JobRequeue'] + [
        dict(parameterName='Prolog', parameterValue=a.dispatcher),
        dict(parameterName='JobRequeue', parameterValue='1')]
else:
    saved = json.loads(a.state_file.read_text())
    if saved['cluster'] != a.cluster or saved['region'] != a.region:
        raise ValueError('Saved state belongs to another cluster or Region')
    installed = {item['parameterName']: item['parameterValue'] for item in current}
    if installed.get('Prolog') != saved.get('dispatcher', '/fsx/aim344/.prolog/dispatch.sh'):
        raise RuntimeError('The installed Prolog changed; inspect it before restoring saved state')
    # Do not discard unrelated custom settings added while the lab was active.
    saved_map = {item['parameterName']: item['parameterValue'] for item in saved['settings']}
    expected = saved_map | {'Prolog': saved.get('dispatcher', '/fsx/aim344/.prolog/dispatch.sh'), 'JobRequeue': '1'}
    if installed != expected:
        raise RuntimeError('Cluster custom settings changed during the lab; reconcile them before restoration')
    settings = saved['settings']
print(json.dumps(dict(action=a.action, cluster=a.cluster, settings=settings)), flush=True)
aws('update-cluster', '--cluster-identifier', a.cluster,
    '--slurm-configuration', json.dumps(dict(slurmCustomSettings=settings)))
for _ in range(90):
    cluster = aws('get-cluster', '--cluster-identifier', a.cluster)['cluster']
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), cluster['status'], flush=True)
    if cluster['status'] == 'ACTIVE':
        actual = cluster.get('slurmConfiguration', {}).get('slurmCustomSettings', [])
        if {x['parameterName']: x['parameterValue'] for x in actual} != {x['parameterName']: x['parameterValue'] for x in settings}:
            raise RuntimeError('PCS returned different custom settings')
        break
    if cluster['status'].endswith('FAILED'):
        raise RuntimeError(cluster.get('errorInfo', cluster['status']))
    time.sleep(10)
else:
    raise TimeoutError('PCS did not finish the update within fifteen minutes')
