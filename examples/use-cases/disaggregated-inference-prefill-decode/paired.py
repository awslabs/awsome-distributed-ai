"""Deploy, verify, or remove simultaneous stacks on separate equal allocations."""
import argparse
import copy
import json
from pathlib import Path
import urllib.request
from deployment import OWNER, kubectl, owned_namespace, read_config, render

MATCHED = ('context', 'image', 'router_image', 'instance_type', 'gpus_per_worker', 'efa_per_worker',
           'model_id', 'model_revision', 'context_length_tokens',
           'chunked_prefill_size_tokens', 'disable_prefix_cache', 'placement', 'packed_cpu', 'packed_memory')


def check_pair(unified, split):
    for key in MATCHED:
        if unified.get(key) != split.get(key):
            raise ValueError(f'Paired configuration mismatch: {key}')
    if unified['namespace'] == split['namespace']:
        raise ValueError('Paired stacks need different namespaces')
    if set(unified['nodes']) & set(split['nodes']):
        raise ValueError('Paired stacks must use disjoint node sets')
    expected = 1 if unified.get('placement') == 'packed' else 2
    if len(unified['nodes']) != expected or len(split['nodes']) != expected:
        raise ValueError(f'This paired placement requires {expected} nodes per stack')
    return 2 * unified['gpus_per_worker']


def paired_render(c, mode):
    namespace = {'apiVersion': 'v1', 'kind': 'Namespace', 'metadata': {'name': c['namespace'], 'labels': OWNER}}
    if c.get('placement') != 'packed':
        doc = render(c, mode)
        doc['items'].insert(0, namespace)
        return doc
    # Reuse exactly the sequential engine options, then give each process its
    # own GPUs and listening port. One pod requests the EFA device once.
    expanded = copy.deepcopy(c)
    expanded['nodes'] = [c['nodes'][0], c['nodes'][0]]
    doc = render(expanded, mode)
    engines = [x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
    packed = copy.deepcopy(engines[0])
    packed['metadata']['name'] = 'engines'
    packed['metadata']['labels'] = OWNER | {'app': 'engines', 'component': 'engine'}
    packed['spec']['selector']['matchLabels'] = {'app': 'engines'}
    template = packed['spec']['template']
    template['metadata']['labels'] = packed['metadata']['labels']
    spec = template['spec']
    spec['hostNetwork'] = False
    spec['dnsPolicy'] = 'ClusterFirst'
    engine = spec['containers'][0]
    plans = []
    for index, item in enumerate(engines):
        original = item['spec']['template']['spec']['containers'][0]
        args = original['args'][:]
        args[args.index('--port') + 1] = str(30000 + index)
        env = {x['name']: x['value'] for x in original['env']}
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(index*c['gpus_per_worker'], (index+1)*c['gpus_per_worker']))
        plans.append({'role': item['metadata']['labels']['role'], 'args': args, 'env': env})
    engine['command'] = ['bash', '-c', 'ulimit -l unlimited; python3 /lab/verify_image.py && exec python3 /lab/packed.py']
    engine.pop('args')
    engine['env'].append({'name': 'AIM345_ENGINE_PLANS', 'value': json.dumps(plans)})
    for resources in engine['resources'].values():
        resources['nvidia.com/gpu'] = 2*c['gpus_per_worker']
        resources['cpu'] = str(c.get('packed_cpu', 8))
        resources['memory'] = c.get('packed_memory', '128Gi')
    health = "import urllib.request; [urllib.request.urlopen('http://127.0.0.1:%d/health'%p, timeout=8).read() for p in (30000,30001)]"
    for key in ('startupProbe', 'readinessProbe'):
        engine[key].pop('httpGet')
        engine[key]['exec'] = {'command': ['python3', '-c', health]}
        engine[key]['timeoutSeconds'] = 20
    result = [namespace]
    names = [x['metadata']['name'] for x in engines]
    for item in doc['items']:
        if item in engines:
            continue
        if item['kind'] == 'Service' and item['metadata']['name'] in names:
            index = names.index(item['metadata']['name'])
            item['spec']['selector'] = OWNER | {'app': 'engines'}
            # A headless service resolves directly to the pod, so both the
            # advertised and target ports must identify the right process.
            item['spec']['ports'][0].update(port=30000+index, targetPort=30000+index)
        if item['kind'] == 'Deployment' and item['metadata']['name'] == 'router':
            router_pod = item['spec']['template']['spec']
            router_pod['nodeSelector'] = spec['nodeSelector'].copy()
            router_pod['tolerations'] = c['tolerations']
            router = router_pod['containers'][0]
            router['args'] = [arg.replace(f"{names[1]}.{c['namespace']}.svc.cluster.local:30000", f"{names[1]}.{c['namespace']}.svc.cluster.local:30001") for arg in router['args']]
        result.append(item)
    result.append(packed)
    return {'apiVersion': 'v1', 'kind': 'List', 'items': result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['render', 'deploy', 'verify', 'cleanup'])
    p.add_argument('--unified-config', default='config.unified.json')
    p.add_argument('--disaggregated-config', default='config.disaggregated.json')
    p.add_argument('--unified-url', default='http://127.0.0.1:8000')
    p.add_argument('--disaggregated-url', default='http://127.0.0.1:8001')
    p.add_argument('--output', type=Path, default=Path('results/paired'))
    a = p.parse_args()
    configs = [read_config(a.unified_config), read_config(a.disaggregated_config)]
    gpus = check_pair(*configs)
    docs = [paired_render(c, mode) for c, mode in zip(configs, ['unified', 'disaggregated'])]
    a.output.mkdir(parents=True, exist_ok=True)
    for mode, doc in zip(['unified', 'disaggregated'], docs):
        (a.output / (mode + '.json')).write_text(json.dumps(doc, indent=2)+'\n')
    if a.action == 'render':
        print(json.dumps({'gpus_per_stack': gpus, 'namespaces': [c['namespace'] for c in configs], 'output': str(a.output)}))
        return
    if a.action == 'deploy':
        # Do not silently adopt another participant's existing namespace.
        for c in configs:
            existing = kubectl(c, 'get', 'namespace', c['namespace'], '--ignore-not-found', '-o', 'name', capture_output=True).stdout.strip()
            if existing:
                raise RuntimeError(f'Namespace already exists: {existing}; verify or explicitly clean up your own run')
        for c, doc in zip(configs, docs):
            owned_namespace(c, create=True)
            kubectl(c, 'apply', '-f', '-', input=json.dumps(doc))
        return
    if a.action == 'cleanup':
        for c in configs:
            owned_namespace(c)
            kubectl(c, 'delete', 'namespace', c['namespace'], '--wait=true')
            kubectl(c, 'delete', 'priorityclass', c['namespace']+'-nonpreempting')
        return
    records = []
    for c, url in zip(configs, [a.unified_url, a.disaggregated_url]):
        owned_namespace(c)
        pods = json.loads(kubectl(c, 'get', 'pods', '-l', 'component=engine', '-o', 'json', capture_output=True).stdout)['items']
        count = sum(int(x['resources']['requests'].get('nvidia.com/gpu', 0)) for pod in pods for x in pod['spec']['containers'])
        if count != gpus or not pods:
            raise RuntimeError('Live GPU request does not match the paired budget')
        for pod in pods:
            if pod['spec']['nodeName'] not in c['nodes'] or not pod['status'].get('containerStatuses') or not all(x['ready'] for x in pod['status']['containerStatuses']):
                raise RuntimeError('Engine placement or readiness differs from the paired configuration')
        with urllib.request.urlopen(url.rstrip('/')+'/health', timeout=30) as response:
            health = response.status
        records.append({'namespace': c['namespace'], 'url': url, 'http_status': health, 'gpus': count,
                        'pod_uids': [x['metadata']['uid'] for x in pods], 'nodes': c['nodes']})
    (a.output/'endpoints.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
