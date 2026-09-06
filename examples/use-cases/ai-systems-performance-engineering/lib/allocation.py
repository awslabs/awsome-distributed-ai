import json
from pathlib import Path
import sys
out,start,end,status,instance=sys.argv[1:]
out=Path(out); status=int(status)
record=dict(instance_type=instance,gpu_count=4,allocated_gpu_seconds=4*(float(end)-float(start)),
            accounting_scope='training job body including preflight, nccl-tests, startup and failed attempts; excludes separate GEMM and serving jobs',
            exit_code=status,useful_tokens=0)
if status==0:
    record['useful_tokens']=json.loads((out/'summary.json').read_text())['useful_tokens']
with (out/'allocation.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
