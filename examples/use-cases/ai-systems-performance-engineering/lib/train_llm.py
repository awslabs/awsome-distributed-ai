"""Pinned Qwen3 full-parameter updates using PyTorch FSDP2 and DCP.

The CPU fixture is a tiny randomly initialized Qwen3 for correctness tests only.
Production runs load the immutable public checkpoint and pretokenized real text.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from llm_checkpoint import Checkpoints, latest_completed, close_checkpointers
from llm_data import Documents, UpdateBatches
from llm_metrics import ContinuousWindow, accumulation_steps, retained_progress


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--attention-implementation', choices=['sdpa', 'eager'], default='sdpa',
                        help='standard Transformers attention implementation; eager is an explicit reference baseline')
    parser.add_argument('--fused-optimizer', action='store_true')
    parser.add_argument('--reshard-after-forward', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--forward-prefetch', action='store_true',
                        help='issue the next FSDP block all-gather earlier from the CPU')
    parser.add_argument('--checkpoint-mode', choices=['none', 'sync', 'process'], default='none')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--interrupt-after-update', type=int)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--cpu-fixture', action='store_true')
    parser.add_argument('--allocation-label', required=True,
                        help='resource class/assignment; retain Slurm and GPU inventory separately')
    return parser.parse_args()


def mark_attention_core(model, implementation):
    """Profile only: run every attention call inside record_function('attention/core').

    tools/profile-table.py attributes forward kernels launched inside the range,
    and backward kernels through the autograd sequence numbers of its operators.
    """
    import sys
    if implementation == 'eager':
        # The decoder looks up the module-level function at call time.
        module = sys.modules[type(model.model.layers[0].self_attn).__module__]
        original = module.eager_attention_forward

        def eager_attention_forward(*args, **kwargs):
            with record_function('attention/core'):
                return original(*args, **kwargs)
        module.eager_attention_forward = eager_attention_forward
    else:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original = ALL_ATTENTION_FUNCTIONS[implementation]

        def attention_forward(*args, **kwargs):
            with record_function('attention/core'):
                return original(*args, **kwargs)
        ALL_ATTENTION_FUNCTIONS[implementation] = attention_forward


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    import transformers
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if args.workers < 0 or not 0 <= cfg['warmup_updates'] < cfg['updates']:
        raise ValueError('invalid workers or warmup/update interval')
    if (args.interrupt_after_update is not None
            and not 0 < args.interrupt_after_update < cfg['updates']):
        raise ValueError('interruption must precede the final update')
    checkpoints = cfg['checkpoint_updates']
    if (sorted(set(checkpoints)) != checkpoints or not checkpoints
            or checkpoints[0] < 1 or checkpoints[-1] != cfg['updates']):
        raise ValueError('checkpoint update indices must include the final update')
    accumulation = accumulation_steps(cfg['global_batch_samples'], args.microbatch, world)
    if args.cpu_fixture:
        device = torch.device('cpu')
        dist.init_process_group('gloo')
    else:
        device = torch.device('cuda', local)
        torch.cuda.set_device(device)
        dist.init_process_group('nccl')
    checkpoint_group = dist.new_group(backend='gloo')
    torch.set_num_threads(1)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    setup_profile = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False, profile_memory=False, with_stack=False,
    ) if args.profile else None
    if setup_profile is not None:
        setup_profile.start()
    dataset = Documents(args.data)
    if len(dataset) < cfg['global_batch_samples'] * cfg['updates']:
        raise ValueError('insufficient documents for the fixed update interval')
    if dataset.manifest['sequence_length'] != cfg['sequence_length']:
        raise ValueError('data sequence length differs from workload')
    if not args.cpu_fixture:
        for key in ('model_id', 'tokenizer_revision', 'dataset_id', 'dataset_revision',
                    'dataset_subset', 'dataset_split'):
            if dataset.manifest[key] != cfg[key]:
                raise ValueError(f'prepared data provenance mismatch: {key}')
    if args.cpu_fixture:
        model_config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=2, num_attention_heads=2,
                                  num_key_value_heads=1, head_dim=8,
                                  tie_word_embeddings=True, attention_dropout=0.0)
        model_config._attn_implementation = args.attention_implementation
        model = Qwen3ForCausalLM(model_config)
    else:
        if args.model_path:
            pin = json.loads((args.model_path / 'aim347-pin.json').read_text())
            if pin != {key: cfg[key] for key in ('model_id', 'model_revision')}:
                raise ValueError('local model snapshot pin mismatch')
        with record_function('setup/model_load'):
            source = str(args.model_path) if args.model_path else cfg['model_id']
            model = AutoModelForCausalLM.from_pretrained(
                source, revision=cfg['model_revision'], token=False, trust_remote_code=False,
                torch_dtype=torch.float32, attn_implementation=args.attention_implementation,
                local_files_only=bool(args.model_path))
    model.config.use_cache = False
    if args.profile:
        mark_attention_core(model, args.attention_implementation)
    if model.config.attention_dropout != 0:
        raise ValueError('microbatch comparisons require dropout-free chosen model')
    if not args.cpu_fixture:
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for block in model.model.layers:
            with record_function('setup/shard_block'):
                fully_shard(block, mp_policy=policy, reshard_after_forward=args.reshard_after_forward)
        with record_function('setup/shard_root'):
            fully_shard(model, mp_policy=policy, reshard_after_forward=args.reshard_after_forward)
        if args.forward_prefetch:
            blocks = list(model.model.layers)
            for block, following in zip(blocks, blocks[1:]):
                block.set_modules_to_forward_prefetch([following])
    elif world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    if args.cpu_fixture and args.fused_optimizer:
        raise ValueError('GPU optimizations are not part of CPU fixture evidence')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                  betas=tuple(cfg['adam_betas']), eps=cfg['adam_epsilon'],
                                  weight_decay=cfg['weight_decay'],
                                  **({'fused': True} if args.fused_optimizer else {}))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    workload = {key: cfg[key] for key in (
        'model_id', 'model_revision', 'tokenizer_revision', 'sequence_length',
        'global_batch_samples', 'updates', 'warmup_updates', 'seed', 'learning_rate',
        'weight_decay', 'adam_betas', 'adam_epsilon', 'precision', 'scheduler')}
    workload.update(data_sha256=dataset.manifest['sha256'], packing='none',
                    initial_state='pretrained revision' if not args.cpu_fixture else 'tiny random CPU fixture',
                    loss_weighting='sum shifted nonpadding token losses / global update useful tokens',
                    data_parallel_size=world)
    if args.cpu_fixture:
        workload.update(model_id=None, model_revision=None, tokenizer_revision=None,
                        precision='float32 CPU correctness fixture',
                        fixture_architecture=model_config.to_dict())
    if args.checkpoint_mode != 'none':
        workload['checkpoint_updates'] = checkpoints
    workload = json.loads(json.dumps(workload))
    configuration = dict(microbatch=args.microbatch, accumulation_steps=accumulation,
                         workers=args.workers, optimizer='fused' if args.fused_optimizer else 'stock',
                         reshard_after_forward=args.reshard_after_forward, attention=args.attention_implementation,
                         forward_prefetch=args.forward_prefetch,
                         nccl_protocol=os.environ.get('NCCL_PROTO', 'auto'),
                         nccl_algorithm=os.environ.get('NCCL_ALGO', 'auto'),
                         requested_nccl_network=os.environ.get('NCCL_NET', 'auto'),
                         requested_nccl_net_plugin=os.environ.get('NCCL_NET_PLUGIN', 'auto'),
                         requested_nccl_gin_plugin=os.environ.get('NCCL_GIN_PLUGIN', 'auto'),
                         requested_fabric_provider=os.environ.get('FI_PROVIDER', 'auto'),
                         requested_ofi_protocol=os.environ.get('OFI_NCCL_PROTOCOL', 'auto'),
                         checkpoint_mode=args.checkpoint_mode,
                         parallelism='FSDP2' if not args.cpu_fixture else 'CPU fixture')
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    saves = Checkpoints(args.output / 'checkpoints', model, optimizer, scheduler,
                        asynchronous=args.checkpoint_mode == 'process', group=checkpoint_group,
                        process_async=args.checkpoint_mode == 'process')
    synchronize = torch.cuda.synchronize if not args.cpu_fixture else lambda: None
    start_update = 0
    useful_tokens = 0
    segment_started = time.perf_counter()
    if args.resume:
        path = args.resume if (args.resume / 'COMPLETED.json').exists() else latest_completed(args.resume)
        if path is None:
            raise ValueError('no completed checkpoint available')
        with record_function('checkpoint/restore'):
            restored = saves.restore(path, expected_workload=workload)
        start_update = restored['completed_updates']
        useful_tokens = restored['useful_tokens']
    if start_update >= cfg['updates']:
        raise ValueError('resume must execute a new optimizer update')
    initial_tokens = useful_tokens
    sampler = UpdateBatches(start_update=start_update, stop_update=cfg['updates'],
                            global_batch=cfg['global_batch_samples'], microbatch=args.microbatch,
                            rank=rank, dp_size=world)
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=not args.cpu_fixture, generator=generator,
                        persistent_workers=args.workers > 0,
                        multiprocessing_context='spawn' if args.workers else None)
    # Iterator startup belongs to the training/recovery interval. Prefetch is
    # continuous thereafter; no per-step timing gaps exclude logging or checks.
    full_window = ContinuousWindow(dist.barrier, synchronize)
    full_window.start()
    with record_function('setup/iterator_start'):
        iterator = iter(loader)
    if setup_profile is not None:
        setup_profile.stop()
        setup_profile.export_chrome_trace(str(args.output / f'setup-trace-rank-{rank}.json'))
    steady_window = ContinuousWindow(dist.barrier, synchronize)
    measured_tokens = 0
    measured_started = False
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU] + (
            [] if args.cpu_fixture else [torch.profiler.ProfilerActivity.CUDA]),
        # Record every update, as without a schedule. A schedule also makes the
        # profiler mark each update as a ProfilerStep#<n> span in the trace.
        schedule=lambda step: torch.profiler.ProfilerAction.RECORD,
        record_shapes=True, profile_memory=True, with_stack=False,
        on_trace_ready=lambda trace: trace.export_chrome_trace(
            str(args.output / f'trace-rank-{rank}.json')),
    ) if args.profile else nullcontext()
    model.train()
    with (args.output / f'rank-{rank}-updates.jsonl').open('w', buffering=1) as log, profiler as profile:
        for update in range(start_update, cfg['updates']):
            # State dict aliases include FSDP storages whose next-forward lifecycle
            # can free/reuse buffers. Fence before zero_grad AND forward, not step.
            saves.before_source_reuse()
            if update >= cfg['warmup_updates'] and not measured_started:
                steady_window.start()
                measured_started = True
            total_tokens = dataset.update_tokens(update, cfg['global_batch_samples'])
            normalizer = torch.tensor(total_tokens, device=device)
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=device)
            local_tokens = 0
            for micro in range(accumulation):
                with record_function('input/wait'):
                    batch = next(iterator)
                    local_tokens += int((batch['labels'][:, 1:] != -100).sum())
                with record_function('input/H2D'):
                    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with record_function('forward_and_loss'):
                    # Transformers' standard loss shifts labels, ignores -100
                    # and sums before dividing by num_items_in_batch. FSDP/DDP
                    # averages gradients, so compensate by the actual DP size.
                    loss = model(**batch, num_items_in_batch=normalizer).loss
                    local_loss += loss.detach()
                with record_function('backward/collectives'):
                    (loss * world).backward()
            with record_function('validation/finite'):
                if not torch.isfinite(local_loss):
                    raise RuntimeError('nonfinite loss')
            with record_function('optimizer'):
                optimizer.step()
                scheduler.step()
            with record_function('validation/token_and_loss_reduction'):
                accounting = torch.tensor([local_tokens, local_loss.item()], device=device, dtype=torch.float64)
                dist.all_reduce(accounting)
                if int(accounting[0].item()) != total_tokens:
                    raise RuntimeError('actual shifted nonpadding token count differs from fixed update work')
            useful_tokens += int(accounting[0].item())
            if measured_started:
                measured_tokens += local_tokens
            progress = {'completed_updates': update + 1, 'useful_tokens': useful_tokens,
                        'next_sample': (update + 1) * cfg['global_batch_samples'], 'workload': workload}
            if args.checkpoint_mode != 'none' and update + 1 in checkpoints:
                with record_function('checkpoint/save_and_stage'):
                    saves.save(progress)
            with record_function('checkpoint/completion_poll'):
                saves.poll()
            row = dict(update=update + 1, local_useful_tokens=local_tokens,
                       monotonic_seconds=time.perf_counter(),
                       checkpoint_upload_pending=saves.pending is not None,
                       global_useful_tokens=int(accounting[0].item()), global_mean_loss=accounting[1].item(),
                       local_normalized_loss=local_loss.item(),
                       durable_update=(saves.durable_progress or {}).get('completed_updates', 0))
            if profile is not None:
                profile.step()
            if args.interrupt_after_update == update + 1:
                dist.barrier()
                row['controlled_interruption'] = True
            with record_function('logging'):
                log.write(json.dumps(row) + '\n')
            if args.interrupt_after_update == update + 1:
                log.flush()
                os.fsync(log.fileno())
                dist.barrier()
                # Explicit process interruption: deliberately do not drain or
                # publish the in-flight checkpoint. Only this job's ranks exit.
                if profile is not None:
                    profile.stop()
                    dist.barrier()
                os._exit(75)
        with record_function('checkpoint/final_flush'):
            saves.finish()
        measured_seconds = steady_window.stop()
        training_seconds = full_window.stop()
    # Both reductions happen after the measured interval.
    timing = torch.tensor([measured_seconds, training_seconds], dtype=torch.float64, device=device)
    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
    token_count = torch.tensor(measured_tokens, dtype=torch.int64, device=device)
    dist.all_reduce(token_count)
    segment_seconds = time.perf_counter() - segment_started
    allocation = dict(label=args.allocation_label, gpu_count=0 if args.cpu_fixture else world,
                      dp_size=world)
    rank_result = dict(rank=rank, host=socket.gethostname(), local_rank=local,
                       local_measured_seconds=measured_seconds, local_measured_useful_tokens=measured_tokens,
                       peak_cuda_allocated_bytes=0 if args.cpu_fixture else torch.cuda.max_memory_allocated(),
                       gpu_name=None if args.cpu_fixture else torch.cuda.get_device_name(),
                       gpu_uuid=None if args.cpu_fixture else str(torch.cuda.get_device_properties(device).uuid),
                       measurement_started_utc_seconds=steady_window.started_utc_seconds,
                       measurement_finished_utc_seconds=steady_window.finished_utc_seconds,
                       cpu_affinity_cores=len(os.sched_getaffinity(0)), status='completed')
    (args.output / f'rank-{rank}.json').write_text(json.dumps(rank_result, indent=2) + '\n')
    ranks = [None] * world
    dist.all_gather_object(ranks, rank_result)
    if rank == 0:
        allocation['cpu_affinity_cores_per_rank'] = [row['cpu_affinity_cores'] for row in ranks]
        allocation['gpu_uuids'] = [row['gpu_uuid'] for row in ranks]
        scope = ('training/save/final durable completion' if args.checkpoint_mode != 'none'
                 else 'continuous post-warmup training')
        result = dict(status='completed', evidence_kind='CPU correctness fixture' if args.cpu_fixture else 'LLM hardware run',
                      workload=workload, configuration=configuration, allocation=allocation,
                      profiled=args.profile,
                      measurement_scope=scope, ranks=ranks,
                      measured_seconds=timing[0].item(), measured_useful_tokens=token_count.item(),
                      measurement_started_utc_seconds=steady_window.started_utc_seconds,
                      measurement_finished_utc_seconds=steady_window.finished_utc_seconds,
                      training_throughput_tokens_per_second=token_count.item() / timing[0].item(),
                      training_wall_seconds=timing[1].item(), segment_seconds=segment_seconds,
                      initial_progress_tokens=initial_tokens, final_progress_tokens=useful_tokens,
                      completed_updates=cfg['updates'], torch_version=torch.__version__,
                      transformers_version=transformers.__version__,
                      segment_source='attendee/local resume' if args.resume else 'fresh run')
        if saves.durable_progress:
            result['retained_progress'] = retained_progress(
                initial_tokens, saves.durable_progress['useful_tokens'], segment_seconds,
                initial_updates=start_update, durable_updates=saves.durable_progress['completed_updates'],
                scope='measured on rank 0 from after model loading, FSDP wrapping, optimizer and checkpoint-manager construction to after the final flush, the closing barriers of the timing windows and the two all-reduces; includes any restore; excludes earlier processes and queue time')
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    finally:
        close_checkpointers()
