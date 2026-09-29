"""Exercise shell detection and the actual launch script with scheduler stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResourceTests(unittest.TestCase):
    def test_detects_visible_gpus_and_slurm_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp)/'nvidia-smi'
            stub.write_text('#!/bin/bash\nfor ((i=0;i<FAKE_GPUS;i++)); do echo "GPU $i: synthetic GPU"; done\n')
            stub.chmod(0o755)
            for count in (2,4,8):
                for source in ('nvidia-smi','SLURM_GPUS_ON_NODE'):
                    env = os.environ | {'PATH':tmp+':'+os.environ['PATH'], 'FAKE_GPUS':str(count), 'OMP_NUM_THREADS':'1'}
                    env.pop('SLURM_GPUS_ON_NODE',None)
                    if source=='SLURM_GPUS_ON_NODE': env[source]=str(count)
                    row = json.loads(subprocess.check_output(['bash',str(ROOT/'lib/resources.sh')],env=env,text=True))
                    self.assertEqual(row['gpus'],count)
                    self.assertEqual(row['gpu_count_source'],source)
                    self.assertGreater(row['available_processors'],0)

if __name__ == '__main__': unittest.main()
