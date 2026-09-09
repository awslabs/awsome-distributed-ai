"""Allocation isolation and packed process addressing, without GPU claims."""
import copy
import json
from pathlib import Path
import unittest
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

    def test_separate_node_path_preserves_two_worker_deployments(self):
        a = dict(self.a, placement='separate-nodes', nodes=['a','b'], gpus_per_worker=2)
        b = dict(a, namespace='aim345-test-b', nodes=['c','d'])
        self.assertEqual(check_pair(a,b),4)
        workers = [x for x in paired_render(b,'disaggregated')['items'] if x['kind']=='Deployment' and x['metadata']['labels'].get('component')=='engine']
        self.assertEqual(len(workers),2)
        self.assertEqual([x['spec']['template']['spec']['nodeSelector']['kubernetes.io/hostname'] for x in workers], ['c','d'])


if __name__ == '__main__': unittest.main()
