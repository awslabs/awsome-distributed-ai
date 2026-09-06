"""One FSDP training workload, controlled by cumulative configuration files."""
import argparse
import functools
import json
import os
from pathlib import Path
import time
import warnings
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from data import Tokens
from metrics import calculate, push


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--data', required=True)
    p.add_argument('--output', required=True); p.add_argument('--instance-type', required=True)
    p.add_argument('--dense-tflops', type=float, required=True)
    p.add_argument('--model-config', default='/opt/aim347/configs/model.json')
    p.add_argument('--steps', type=int, default=100); p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--microbatch', type=int, default=1); p.add_argument('--cpu-rounds', type=int, default=2000)
    p.add_argument('--pushgateway', default=''); p.add_argument('--run-id', default='manual')
    a=p.parse_args()
    if a.steps <= a.warmup or min(a.warmup,a.cpu_rounds)<0 or a.microbatch<1 or a.dense_tflops<=0:
        p.error('invalid step counts, batch size or peak')
    config=json.loads(Path(a.config).read_text())
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); dist.init_process_group('nccl')
    torch.set_num_threads(1); torch.manual_seed(347)
    model_config=Qwen2Config(**json.loads(Path(a.model_config).read_text()))
    model_config.use_cache=False; model_config._attn_implementation='sdpa'
    model=Qwen2ForCausalLM(model_config)
    nonembedding=model.num_parameters(exclude_embeddings=True)
    model=FSDP(model, device_id=local, use_orig_params=True,
               auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}),
               mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16))
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    dataset=Tokens(a.data,config['layout'],a.cpu_rounds)
    sampler=DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    loader=DataLoader(dataset, batch_size=a.microbatch, sampler=sampler, num_workers=config['workers_per_rank'],
                      pin_memory=True, persistent_workers=config['workers_per_rank']>0,
                      multiprocessing_context='spawn' if config['workers_per_rank'] else None, drop_last=True)
    if len(loader)<a.steps:
        raise ValueError('dataset is too small for this fixed-length run')
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    sequence=dataset.meta['sequence_tokens']; tokens_per_step=world*a.microbatch*sequence
    durations=[]; epoch_start=time.perf_counter(); iterator=iter(loader)
    for step in range(a.steps):
        # Barrier prevents rank skew; timing includes data wait, transfers, forward,
        # backward, optimizer and the slowest rank. Warmup steps update weights too.
        dist.barrier(); torch.cuda.synchronize(); started=time.perf_counter()
        batch=next(iterator).to(local, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(input_ids=batch[:,:-1]).logits
            loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.size(-1)),batch[:,1:].reshape(-1))
        loss.backward(); optimizer.step(); torch.cuda.synchronize()
        elapsed=torch.tensor(time.perf_counter()-started,device=local,dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        if step>=a.warmup:
            durations.append(elapsed.item())
        if rank==0 and durations and ((step+1)%10==0 or step+1==a.steps):
            values=calculate(tokens_per_step*len(durations)/sum(durations),nonembedding,world,a.dense_tflops)
            values['step_duration_ms']=1000*sum(durations)/len(durations)
            if a.instance_type=='g7e.12xlarge':
                values['mfu_dense_theory_ratio']=values['tflops_per_gpu']/480.0
            record=dict(values,step_count=step+1,loss=loss.item(),config=config['name'],instance_type=a.instance_type)
            print(json.dumps(record),flush=True)
            with (output/'steps.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            try: push(a.pushgateway,a.run_id,config['name'],a.instance_type,values)
            except Exception as error: warnings.warn(f'Pushgateway failed, local metrics retained: {error}')
    if rank==0:
        summary=dict(config=config['name'],instance_type=a.instance_type,gpu=torch.cuda.get_device_name(local),
                     gpu_count=world,nonembedding_parameters=nonembedding,sequence_tokens=sequence,
                     tokens_per_step=tokens_per_step,measured_steps=len(durations),measured_seconds=sum(durations),
                     useful_tokens=a.steps*tokens_per_step,training_wall_seconds=time.perf_counter()-epoch_start,
                     dense_tflops_per_gpu=a.dense_tflops,workers_per_rank=config['workers_per_rank'],
                     affinity_cores=len(os.sched_getaffinity(0)),model_config=model_config.to_dict(),
                     data_sha256=dataset.meta['sha256'],torch_version=torch.__version__,status='completed',**values)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__': main()
