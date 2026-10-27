"""Allocation isolation and packed process addressing, without GPU claims."""
import copy
import json
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile
from paired import check_pair, paired_render


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.a = json.loads(Path('config.example.json').read_text())
        self.a.update(namespace='aim345-test-a', nodes=['node-a'], placement='packed', gpus_per_worker=1, efa_per_worker=1)
        self.b = dict(self.a, namespace='aim345-test-b', nodes=['node-b'])

    def test_pair_rejects_overlap_or_changed_budget_and_model(self):
        self.assertEqual(check_pair(self.a, self.b), 2)
        for update in [dict(nodes=['node-a']), dict(namespace=self.a['namespace']), dict(gpus_per_worker=2), dict(model_revision='a'*40)]:
            with self.assertRaises(ValueError):
                check_pair(self.a, self.b | update)

    def test_packed_processes_use_disjoint_gpus_and_distinct_service_ports(self):
        docs = [paired_render(c, mode) for c, mode in [(self.a, 'unified'), (self.b, 'disaggregated')]]
        priorities = []
        for c, doc in zip([self.a, self.b], docs):
            self.assertEqual(next(x['metadata']['name'] for x in doc['items'] if x['kind']=='Namespace'), c['namespace'])
            priorities.append(next(x['metadata']['name'] for x in doc['items'] if x['kind']=='PriorityClass'))
            deployments = {x['metadata']['name']: x for x in doc['items'] if x['kind']=='Deployment'}
            pod = deployments['engines']['spec']['template']['spec']
            container = pod['containers'][0]
            self.assertFalse(pod['hostNetwork'])
            self.assertEqual(container['resources']['requests']['nvidia.com/gpu'], 2)
            self.assertEqual(container['resources']['requests']['vpc.amazonaws.com/efa'], 1)
            plans = json.loads(next(x['value'] for x in container['env'] if x['name']=='AIM345_ENGINE_PLANS'))
            self.assertEqual([x['env']['CUDA_VISIBLE_DEVICES'] for x in plans], ['0', '1'])
            self.assertEqual([x['args'][x['args'].index('--port')+1] for x in plans], ['30000','30001'])
            services = [x for x in doc['items'] if x['kind']=='Service' and x['metadata']['name']!='router']
            self.assertEqual([x['spec']['ports'][0]['targetPort'] for x in services], [30000,30001])
            router = deployments['router']['spec']['template']['spec']
            self.assertEqual(router['nodeSelector']['kubernetes.io/hostname'],c['nodes'][0])
            self.assertTrue(any(':30001' in x for x in router['containers'][0]['args']))
            self.assertEqual(pod['preemptionPolicy'],'Never')
        self.assertNotEqual(*priorities)

    def test_preparation_enforces_reduced_node_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = root / 'nodes.json'
            inventory.write_text(json.dumps({'items': [
                {'metadata': {'name': name, 'labels': {'node.kubernetes.io/instance-type': 'g7.48xlarge'}},
                 'spec': {}, 'status': {'allocatable': {'nvidia.com/gpu': '8', 'vpc.amazonaws.com/efa': '2', 'cpu': '191450m', 'memory': '709364952Ki'}}}
                for name in ('node-a', 'node-b')]}))
            cmd = [sys.executable, 'facilitator/prepare-config.py', '--context', 'test',
                   '--nodes', 'node-a,node-b', '--engine-image', 'engine@sha256:'+'a'*64,
                   '--router-image', 'router@sha256:'+'b'*64, '--inventory', str(inventory),
                   '--gpus-per-stack', '4', '--efas-per-stack', '1', '--cpu-per-stack', '96',
                   '--memory-gib-per-stack', '384', '--model-cache-host-path', '/mnt/nvme/models']
            subprocess.run(cmd + ['--output', str(root/'valid')], check=True, capture_output=True)
            for mode in ('unified', 'disaggregated'):
                config = json.loads((root/'valid'/f'config.{mode}.json').read_text())
                pods = [x['spec']['template']['spec'] for x in paired_render(config, mode)['items'] if x['kind']=='Deployment']
                limits = [c['resources']['limits'] for pod in pods for c in pod['containers']]
                self.assertEqual(sum(int(c.get('nvidia.com/gpu', 0)) for c in limits), 4)
                self.assertEqual(sum(int(c.get('vpc.amazonaws.com/efa', 0)) for c in limits), 1)
                self.assertEqual(sum(int(c['cpu']) for c in limits), 96)
                self.assertEqual(sum(int(c['memory'].removesuffix('Gi')) for c in limits), 384)
            for flag, value in [('--gpus-per-stack', '10'), ('--gpus-per-stack', '3'), ('--efas-per-stack', '3'), ('--cpu-per-stack', '4'), ('--memory-gib-per-stack', '8'), ('--memory-gib-per-stack', '740'), ('--cpu-per-stack', '192')]:
                rejected = cmd.copy()
                rejected[rejected.index(flag)+1] = value
                result = subprocess.run(rejected + ['--output', str(root/'invalid')], capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_optional_profile_rejects_secondary_shape_and_mainline_model(self):
        from deployment import read_config, render
        c = json.loads(Path('config.v4.example.json').read_text())
        c.update(context='test', nodes=['a','b'], image='v4@sha256:'+'a'*64,
                 router_image='router@sha256:'+'b'*64)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            path.write_text(json.dumps(c))
            qualified = read_config(path)
            workers = [x for x in render(qualified, 'disaggregated')['items'] if x['kind']=='Deployment' and x['metadata']['labels'].get('component')=='engine']
            self.assertEqual({x['spec']['template']['spec']['nodeSelector']['kubernetes.io/hostname'] for x in workers}, {'a','b'})
            for update in [dict(instance_type='g7.24xlarge', gpus_per_worker=4, efa_per_worker=1),
                           dict(placement='packed'), dict(engine_profile='mainline'),
                           dict(model_id='deepseek-ai/DeepSeek-V2-Lite-Chat')]:
                path.write_text(json.dumps(c | update))
                with self.assertRaises(AssertionError):
                    read_config(path)

    def test_separate_node_path_preserves_two_worker_deployments(self):
        a = dict(self.a, placement='separate-nodes', nodes=['a','b'], gpus_per_worker=2)
        b = dict(a, namespace='aim345-test-b', nodes=['c','d'])
        self.assertEqual(check_pair(a,b),4)
        workers = [x for x in paired_render(b,'disaggregated')['items'] if x['kind']=='Deployment' and x['metadata']['labels'].get('component')=='engine']
        self.assertEqual(len(workers),2)
        self.assertEqual([x['spec']['template']['spec']['nodeSelector']['kubernetes.io/hostname'] for x in workers], ['c','d'])

    def test_page_size_and_engine_env_reach_engines_or_are_refused(self):
        from deployment import LAB_ENV, read_config, render
        c = json.loads(Path('config.example.json').read_text())
        c.update(context='test', nodes=['a','b'], instance_type='g7.48xlarge', gpus_per_worker=4, efa_per_worker=1, image='lab@sha256:'+'a'*64, router_image='router@sha256:'+'b'*64)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            def engines(update):
                path.write_text(json.dumps(c | update))
                return [x['spec']['template']['spec']['containers'][0] for x in render(read_config(path), 'disaggregated')['items'] if x['kind']=='Deployment' and x['metadata']['labels'].get('component')=='engine']
            for engine in engines({}):
                self.assertNotIn('--page-size', engine['args'])
                self.assertEqual(engine['env'], [{'name': k, 'value': v} for k, v in LAB_ENV.items()])
            for engine in engines(dict(page_size_tokens=64, engine_env={'FI_EFA_ENABLE_SHM_TRANSFER': '0'})):
                self.assertEqual(engine['args'][engine['args'].index('--page-size') + 1], '64')
                self.assertEqual(engine['env'][-1], {'name': 'FI_EFA_ENABLE_SHM_TRANSFER', 'value': '0'})
            for update in [dict(page_size_tokens=0), dict(page_size_tokens='64'), dict(page_size_tokens=True),
                           dict(engine_env=['FI_EFA_ENABLE_SHM_TRANSFER=0']), dict(engine_env={'FI_EFA_ENABLE_SHM_TRANSFER': 0}),
                           dict(engine_env={'FI_PROVIDER': 'tcp'}), dict(engine_env={'CUDA_VISIBLE_DEVICES': '0'}),
                           dict(engine_env={'AIM345_ENGINE_PLANS': 'x'}), dict(engine_env={'A': ''}), dict(engine_env={'A': '$(FI_PROVIDER)'}), dict(engine_env={'A': 'a$$b'}),
                           dict(engine_env={'A': 'a\0b'}), dict(engine_env={'A=B': '1'}), dict(engine_env={'A\nB': '1'}), dict(engine_env={'変数': '1'}), dict(engine_env={'': '1'})]:
                path.write_text(json.dumps(c | update))
                with self.assertRaises(AssertionError):
                    read_config(path)

    def test_pair_matches_page_size_but_not_engine_env(self):
        self.assertEqual(check_pair(self.a, self.b | dict(engine_env={'FI_EFA_ENABLE_SHM_TRANSFER': '0'})), 2)
        self.assertIsNone(self.a['page_size_tokens'])
        self.assertEqual(check_pair(self.a, {k: v for k, v in self.b.items() if k != 'page_size_tokens'}), 2)
        plans = next(x for x in paired_render(self.b | dict(engine_env={'FI_EFA_ENABLE_SHM_TRANSFER': '0'}), 'disaggregated')['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine')['spec']['template']['spec']['containers'][0]['env']
        plans = json.loads(next(x['value'] for x in plans if x['name'] == 'AIM345_ENGINE_PLANS'))
        self.assertEqual({p['env']['FI_EFA_ENABLE_SHM_TRANSFER'] for p in plans}, {'0'})
        with self.assertRaises(ValueError):
            check_pair(self.a, self.b | dict(page_size_tokens=64))


if __name__ == '__main__': unittest.main()
