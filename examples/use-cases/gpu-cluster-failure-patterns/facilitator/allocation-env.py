"""Print the participant allocation settings from registered Slurm nodes."""
import argparse
import os
import re
import shlex
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--partition', default=os.environ.get('PARTITION'), required='PARTITION' not in os.environ)
a = p.parse_args()
names = sorted(set(subprocess.check_output(['sinfo', '-p', a.partition, '-N', '-h', '-o', '%N'], text=True).split()))
if len(names) != 2:
    p.error('The workshop partition must contain exactly two assigned nodes')
shapes = set()
for name in names:
    fields = dict(word.split('=', 1) for word in subprocess.check_output(['scontrol', 'show', 'node', name, '-o'], text=True).split() if '=' in word)
    match = re.search(r'(?:^|,)gpu(?::[^:,()]+)?:(\d+)(?:\(|,|$)', fields.get('Gres', ''))
    if not match:
        p.error('Each assigned node must advertise a numeric GPU GRES count')
    gpus, cpus = int(match[1]), int(fields['CPUTot'])
    if not gpus or cpus % gpus:
        p.error('The registered CPU count must divide evenly among GPU ranks')
    shapes.add((gpus, cpus // gpus))
if len(shapes) != 1:
    p.error('Use a homogeneous pair for the collective comparison')
gpus, cpus_per_rank = shapes.pop()
for key, value in dict(PARTITION=a.partition, GPUS_PER_NODE=gpus, CPUS_PER_RANK=cpus_per_rank, MEMORY_PER_NODE=0).items():
    print(f'export {key}={shlex.quote(str(value))}')
