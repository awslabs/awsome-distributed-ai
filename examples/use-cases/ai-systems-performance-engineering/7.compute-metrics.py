#!/usr/bin/env python3
"""Recompute comparable MFU and explicitly scoped useful tokens per GPU-hour."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'lib'))
from metrics import calculate, push


def main():
    p=argparse.ArgumentParser(); p.add_argument('results',type=Path)
    p.add_argument('--gemm',type=Path,help='Observed GEMM JSON from the same instance type')
    p.add_argument('--pushgateway',default='')
    a=p.parse_args()
    gemm=json.loads(a.gemm.read_text()) if a.gemm else None
    rows=[]
    comparison=None
    for config in ('v0','v1','v2','v3'):
        path=a.results/config/'summary.json'
        if not path.exists():
            rows.append(dict(config=config,status='UNVALIDATED')); continue
        s=json.loads(path.read_text()); instance=s['instance_type']
        fingerprint={k:s[k] for k in ('instance_type','gpu_count','nonembedding_parameters','sequence_tokens','tokens_per_step','measured_steps','model_config','data_sha256')}
        if comparison is not None and fingerprint!=comparison:
            raise ValueError('ladder runs must use the same hardware, model, data, token batch and measured step count')
        comparison=fingerprint
        if s['status']!='completed': raise ValueError('incomplete run')
        if gemm:
            if gemm['instance_type']!=instance or gemm['sparse'] or gemm['input_dtype']!='bf16':
                raise ValueError('GEMM denominator must match the training instance type and dense BF16 precision')
            peak=gemm['best_tflops_per_gpu']; source='current GEMM measurement'
        elif instance=='g7e.12xlarge':
            peak=429.1; source='PI Oregon g7e.12xlarge measurement, 2026-08-15'
        else:
            raise ValueError('Nonproduction hardware requires --gemm from that instance type')
        values=calculate(s['tokens_per_step']*s['measured_steps']/s['measured_seconds'],s['nonembedding_parameters'],s['gpu_count'],peak)
        if instance=='g7e.12xlarge': values['mfu_dense_theory_ratio']=values['tflops_per_gpu']/480.0
        accounting=a.results/config/'allocation.jsonl'
        if accounting.exists():
            attempts=[json.loads(line) for line in accounting.read_text().splitlines()]
            if any(x['instance_type']!=instance or x['allocated_gpu_seconds']<=0 for x in attempts):
                raise ValueError('invalid allocation accounting')
            gpu_hours=sum(x['allocated_gpu_seconds'] for x in attempts)/3600
            useful=sum(x['useful_tokens'] for x in attempts)
            values['useful_tokens_per_gpu_hour']=useful/gpu_hours
            values['allocated_gpu_hours']=gpu_hours
        else:
            print(f'{config}: no allocation ledger; goodput is UNVALIDATED',file=sys.stderr)
        row=dict(config=config,status='VALIDATED',instance_type=instance,denominator_tflops_per_gpu=peak,
                 denominator_source=source,**values)
        rows.append(row)
        push(a.pushgateway,a.results.name,config,instance,values)
    ratios=[r['mfu_dense_gemm_ratio'] for r in rows if 'mfu_dense_gemm_ratio' in r]
    print(json.dumps(dict(results=rows,monotonic_increase_observed=all(b>a for a,b in zip(ratios,ratios[1:])) if len(ratios)==4 else None),indent=2))


if __name__=='__main__': main()
