# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The exercise target's configuration: device inventory and maintenance pins.

No device or scheduler is touched: sysfs is a temporary tree and the commands
the inventory runs are recorded stand-ins.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

LAB = Path(__file__).resolve().parents[1]
INSTANCE = 'i-1123456789abcdef0'


def load():
    spec = importlib.util.spec_from_file_location('aim344_prepare_target',
                                                  LAB / 'facilitator' / 'prepare-target.py')
    if spec is None or spec.loader is None:
        raise unittest.SkipTest('prepare-target.py not importable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


target = load()


class Inventory(unittest.TestCase):
    def test_the_selected_management_route_and_a_separate_efa(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def mapped(value):
                return root / str(value).lstrip('/')
            pci = mapped('/sys/bus/pci/devices')
            for bdf, driver in (('0000:00:01.0', 'ena'), ('0000:00:02.0', 'nvidia'),
                                ('0000:00:03.0', 'efa')):
                device = pci / bdf
                device.mkdir(parents=True)
                (device / 'driver').symlink_to('/fixture/drivers/' + driver)
                (device / 'vendor').write_text('0x1234')
                (device / 'device').write_text('0x5678')
            interface = mapped('/sys/class/net/eth0')
            interface.mkdir(parents=True)
            (interface / 'device').symlink_to(pci / '0000:00:01.0')
            efa = pci / '0000:00:03.0'
            (efa / 'infiniband/rdma-new').mkdir(parents=True)
            selected_routes = [{'dev': 'eth0', 'prefsrc': '10.1.2.51'}]
            calls = []

            def command(argv):
                calls.append(argv)
                if argv == ['/usr/sbin/ip', '-j', 'route', 'get', '169.254.169.254']:
                    return json.dumps(selected_routes)
                return 'GPU-new-identity, 00000000:00:02.0'
            with mock.patch.object(target, 'Path', side_effect=mapped), \
                    mock.patch.object(target, 'run', side_effect=command):
                cfg = target.inventory('gpu-g7-1', '/slurm', 'ubuntu', INSTANCE)
                self.assertEqual((INSTANCE, 'gpu-g7-1', 'ubuntu', ['ubuntu']),
                                 (cfg['instance_id'], cfg['slurm_node'], cfg['participant_user'],
                                  cfg['participant_users']))
                self.assertEqual(cfg['gpu_bdf'], '0000:00:02.0')
                self.assertEqual(cfg['efa_rdma_device'], 'rdma-new')
                self.assertEqual(cfg['management_interface'], 'eth0')
                self.assertEqual(cfg['management_bdf'], '0000:00:01.0')
                self.assertIn(['/usr/sbin/ip', '-j', 'route', 'get', '169.254.169.254'], calls)
                for bad in ([], [{}], [{'dev': ''}], [{'dev': '../eth0'}],
                            [{'dev': 'eth0'}, {'dev': 'eth1'}]):
                    selected_routes = bad
                    with self.subTest(routes=bad), \
                            self.assertRaisesRegex(RuntimeError, 'Ambiguous management route'):
                        target.inventory('gpu-g7-1', '/slurm', 'ubuntu', INSTANCE)
                selected_routes = [{'dev': 'eth0'}]
                (efa / 'net/eth1').mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError, 'No separate EFA'):
                    target.inventory('gpu-g7-1', '/slurm', 'ubuntu', INSTANCE)


class MaintenanceConfig(unittest.TestCase):
    def test_the_staged_images_are_pinned_for_the_participant(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            pins = {}
            for name in ('aim344.sqsh', 'nccl-baseline.sqsh'):
                (stage / name).write_bytes(name.encode())
                pins[name] = hashlib.sha256(name.encode()).hexdigest()
            config = target.maintenance_config('/slurm', 'ubuntu', stage, '2')
            self.assertEqual(pins, config['staging_image_sha256'])
            self.assertEqual(('ubuntu', ['ubuntu'], 'root-directory', [0, 2, 3, 6], 2),
                             (config['participant_user'], config['participants'],
                              config['staging_policy'], config['checks'],
                              config['expected_efa_devices']))
            self.assertNotIn('expected_efa_devices',
                             target.maintenance_config('/slurm', 'ubuntu', stage, None))
            for count in ('0', '33', '02', '2;id'):
                with self.subTest(count=count), self.assertRaises(RuntimeError):
                    target.maintenance_config('/slurm', 'ubuntu', stage, count)

    def test_the_maintenance_helper_reads_the_written_configuration(self):
        spec = importlib.util.spec_from_file_location('aim344_maintenance_target',
                                                      LAB / 'facilitator' / 'maintenance.py')
        maintenance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(maintenance)
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            for name in ('aim344.sqsh', 'nccl-baseline.sqsh'):
                (stage / name).write_bytes(name.encode())
            path = stage / 'aim344-maintenance.json'
            target.write(path, json.dumps(target.maintenance_config(
                '/opt/aws/pcs/scheduler/slurm-25.05/bin', 'ubuntu', stage, '2')), 0o600)
            with mock.patch.object(maintenance, 'CONFIG', path):
                config = maintenance.load_config()
            self.assertEqual('2', maintenance.environment(config)['HEALTHCHECK_EXPECTED_EFA_COUNT'])
            self.assertEqual('ubuntu', maintenance.resolve_participant(config, 'ubuntu'))
            with self.assertRaises(maintenance.Refusal):
                maintenance.resolve_participant(config, 'aim344-participant')


if __name__ == '__main__':
    unittest.main()
