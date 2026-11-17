"""Write matched packed configurations from the two allocated EKS nodes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired import check_pair
from deployment import read_config

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--context', required=True)
p.add_argument('--nodes', required=True, help='Two comma-separated allocated Kubernetes node names')
p.add_argument('--engine-image', required=True)
p.add_argument('--router-image', required=True)
p.add_argument('--namespace-prefix', default='aim345-event')
p.add_argument('--output', type=Path, default=Path('.'))
p.add_argument('--gpus-per-stack', type=int, help='Even GPU budget per node; defaults to all allocatable GPUs')
p.add_argument('--efas-per-stack', type=int, help='EFA device budget per node; defaults to all allocatable EFA devices')
p.add_argument('--cpu-per-stack', type=int, help='Total engine and router CPU limit in vCPUs; default leaves 3 allocatable vCPUs for node agents')
p.add_argument('--memory-gib-per-stack', type=int, help='Total engine and router memory limit in GiB; default leaves 4 allocatable GiB for node agents')
p.add_argument('--model-cache-host-path', default='/mnt/k8s-disks/0/aim345-models')
p.add_argument('--inventory', type=Path, help='Saved kubectl node List for offline review')
a = p.parse_args()
names = a.nodes.split(',')
if len(set(names)) != 2 or not a.namespace_prefix.startswith('aim345-'):
    p.error('Use two distinct nodes and an aim345- namespace prefix')
if not all('@sha256:' in image for image in (a.engine_image, a.router_image)):
    p.error('Both images require immutable digests')
inventory = json.loads(a.inventory.read_text()) if a.inventory else json.loads(subprocess.check_output(
    ['kubectl', '--context', a.context, 'get', 'nodes', *names, '-o', 'json'], text=True))
nodes = {node['metadata']['name']: node for node in inventory['items']}
if set(names) - nodes.keys():
    p.error('Assigned nodes are missing from the inventory')
if len({nodes[name]['metadata']['labels']['node.kubernetes.io/instance-type'] for name in names}) != 1:
    p.error('Use matching instance types for the comparison')
def available_budget(node):
    resources = node['status']['allocatable']
    if 'cpu' not in resources or 'memory' not in resources:
        p.error('Node inventory must include allocatable CPU and memory')
    raw_cpu, memory = str(resources['cpu']), resources['memory']
    cpu = Decimal(raw_cpu[:-1]) / 1000 if raw_cpu.endswith('m') else Decimal(raw_cpu)
    suffixes = {'Ki': 2**10, 'Mi': 2**20, 'Gi': 2**30, 'Ti': 2**40}
    memory_bytes = int(memory[:-2]) * suffixes[memory[-2:]] if memory[-2:] in suffixes else int(memory)
    return cpu, memory_bytes

available = [available_budget(nodes[name]) for name in names]
if a.cpu_per_stack is None:
    a.cpu_per_stack = min(int(cpu) for cpu, _ in available) - 3
if a.memory_gib_per_stack is None:
    a.memory_gib_per_stack = min(memory // 2**30 for _, memory in available) - 4

base = json.loads((Path(__file__).resolve().parents[1] / 'config.example.json').read_text())
a.output.mkdir(parents=True, exist_ok=True)
if a.cpu_per_stack <= 4 or a.memory_gib_per_stack <= 8:
    p.error('The stack budget must exceed the router limit of 4 vCPUs and 8 GiB')
if not a.model_cache_host_path.startswith('/mnt/'):
    p.error('Use a dedicated model cache under /mnt')
configs = []
for mode, name in zip(('unified', 'disaggregated'), names):
    node = nodes[name]
    resources = node['status']['allocatable']
    cpu, memory_bytes = available_budget(node)
    if a.cpu_per_stack > cpu or a.memory_gib_per_stack * 2**30 > memory_bytes:
        p.error('CPU or memory stack budget exceeds Kubernetes allocatable resources')
    available_gpus, available_efas = int(resources['nvidia.com/gpu']), int(resources['vpc.amazonaws.com/efa'])
    gpus = available_gpus if a.gpus_per_stack is None else a.gpus_per_stack
    efas = available_efas if a.efas_per_stack is None else a.efas_per_stack
    if gpus > available_gpus or efas > available_efas:
        p.error('Requested GPU or EFA budget exceeds the assigned node inventory')
    if gpus < 2 or gpus % 2 or efas < 1:
        p.error('Packed placement needs an even GPU count and at least one EFA per node')
    config = dict(base, context=a.context, namespace=f'{a.namespace_prefix}-{mode}',
                  image=a.engine_image, router_image=a.router_image,
                  instance_type=node['metadata']['labels']['node.kubernetes.io/instance-type'],
                  nodes=[name], placement='packed', gpus_per_worker=gpus // 2, efa_per_worker=efas,
                  packed_cpu=a.cpu_per_stack - 4, packed_memory=f'{a.memory_gib_per_stack - 8}Gi',
                  model_cache_host_path=a.model_cache_host_path)
    config['tolerations'] = [dict(key=t['key'], operator='Equal', value=t.get('value', ''), effect=t['effect'])
                             for t in node['spec'].get('taints', [])]
    path = a.output / f'config.{mode}.json'
    if path.exists():
        raise FileExistsError(f'Preserve the existing configuration: {path}')
    path.write_text(json.dumps(config, indent=2) + '\n')
    configs.append(read_config(path))
print(json.dumps(dict(gpus_per_stack=check_pair(*configs), nodes=names,
                     namespaces=[c['namespace'] for c in configs]), indent=2))
