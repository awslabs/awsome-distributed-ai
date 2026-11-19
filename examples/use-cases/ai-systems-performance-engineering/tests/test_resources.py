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

    def test_actual_job_constructs_rank_counts_and_records_detected_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp)/'lab'; shutil.copytree(ROOT,lab,ignore=shutil.ignore_patterns('results','.env','__pycache__'))
            bin_dir = Path(tmp)/'bin'; bin_dir.mkdir()
            srun = bin_dir/'srun'
            srun.write_text('''#!/usr/bin/env python3
import json,os,sys,subprocess
args=sys.argv[1:]
with open(os.environ['COMMAND_LOG'],'a') as f: f.write(json.dumps(args)+'\\n')
if any(x.endswith('/lib/resources.sh') for x in args):
 for host in ('node-a','node-b'): print(json.dumps(dict(host=host,gpus=int(os.environ['FAKE_GPUS']),available_processors=48)))
elif any('torchrun' in x for x in args):
 sys.exit(subprocess.run(args[args.index('bash'):],env=os.environ|{'SLURM_PROCID':'0'}).returncode)
''')
            srun.chmod(0o755)
            (bin_dir/'scontrol').write_text('#!/bin/bash\nif [[ $2 == hostnames ]]; then printf "node-a\\nnode-b\\n"; else echo "NodeName=node-a NodeAddr=10.0.0.1"; fi\n'); (bin_dir/'scontrol').chmod(0o755)
            (bin_dir/'torchrun').write_text('#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["COMMAND_LOG"],"a") as f: f.write(json.dumps(["torchrun",*sys.argv[1:]])+"\\n")\nsys.exit(7)\n')
            (bin_dir/'torchrun').chmod(0o755)
            for count in (2,4,8):
                run = 'test-'+str(count)
                log = Path(tmp)/(run+'.jsonl')
                env = os.environ | dict(PATH=str(bin_dir)+':'+os.environ['PATH'],LAB_DIR=str(lab),RUN_ID=run,SLURM_JOB_NUM_NODES='2',SLURM_JOB_ID='123',SLURM_CPUS_ON_NODE='96',SLURM_JOB_NODELIST='node-a,node-b',NCCL_SOCKET_IFNAME='=test',LAB_IMAGE='/test.sqsh',DATA_DIR='/test-data',INSTANCE_TYPE='g7e.12xlarge',DENSE_TFLOPS='123',FAKE_GPUS=str(count),COMMAND_LOG=str(log))
                result = subprocess.run(['bash',str(lab/'lib/job.sh'),'v0'],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,7,result.stderr)
                calls = [json.loads(x) for x in log.read_text().splitlines()]
                mpi = next(x for x in calls if '--mpi=pmix' in x)
                self.assertIn('--ntasks='+str(2*count),mpi)
                self.assertIn('--ntasks-per-node='+str(count),mpi)
                launch = next(x for x in calls if x[0]=='torchrun')
                self.assertIn('--nproc-per-node='+str(count),launch)
                self.assertIn('--nnodes=2',launch)
                node_launch = next(x for x in calls if any('torchrun' in arg for arg in x) and x[0] != 'torchrun')
                self.assertIn('--cpus-per-task=96', node_launch)
                self.assertIn('--master-addr=10.0.0.1',launch)
                record = json.loads((lab/'results'/run/'v0-resources.json').read_text())
                self.assertEqual(record['gpus_per_node'],count)
                ledger = json.loads((lab/'results'/run/'v0/allocation.jsonl').read_text())
                self.assertEqual(ledger['gpu_count'],2*count)
                self.assertEqual(ledger['useful_tokens'],0)


if __name__ == '__main__': unittest.main()
