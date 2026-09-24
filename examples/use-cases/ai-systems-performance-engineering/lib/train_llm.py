"""Pinned Qwen3 full-parameter updates using PyTorch FSDP2 and DCP.

The CPU fixture is a tiny randomly initialized Qwen3 for correctness tests only.
Production runs load the immutable public checkpoint and pretokenized real text.
"""
import argparse
import hashlib
from functools import partial
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
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_utils import no_init_weights

from llm_checkpoint import Checkpoints, latest_completed, close_checkpointers
from llm_data import Documents, UpdateBatches, trim_padding
from llm_metrics import ContinuousWindow, accumulation_steps, training_goodput


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--parallel-model-loading', action='store_true',
                        help='load independent pretrained weight shards with the Transformers thread pool')
    parser.add_argument('--resume-from-config', action='store_true',
                        help='on restore only, construct the model without loading redundant pretrained weights')
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--trim-padding', action='store_true')
    parser.add_argument('--causal-right-padding', action='store_true',
                        help='omit redundant attention mask for validated right-padded independent documents')
    parser.add_argument('--fused-optimizer', action='store_true')
    parser.add_argument('--attention-implementation', choices=['sdpa', 'eager'], default='sdpa',
                        help='standard Transformers attention implementation; eager is an explicit reference baseline')
    parser.add_argument('--liger-linear-ce', action='store_true',
                        help='Qwen2 only: Liger 0.6.4 fused linear cross entropy, other kernels unchanged')
    parser.add_argument('--liger-fp32-accumulation', action='store_true')
    parser.add_argument('--compile-blocks', action='store_true')
    parser.add_argument('--compile-loss', action='store_true')
    parser.add_argument('--compile-backend', choices=['inductor', 'aot_eager'], default='inductor',
                        help='aot_eager isolates graph/autograd changes without generated kernels; diagnostic only')
    parser.add_argument('--compile-pointwise', action='store_true',
                        help='compile Qwen3 normalization and activation modules without compiling matrix products')
    parser.add_argument('--compile-pointwise-scope', choices=['all', 'norms', 'activations'], default='all',
                        help='isolate normalization from activation compilation; requires --compile-pointwise')
    parser.add_argument('--compile-preserve-casts', action='store_true',
                        help='preserve eager low-precision cast boundaries in compiled operators')
    parser.add_argument('--reshard-after-forward', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--deterministic', action='store_true',
                        help='require deterministic PyTorch algorithms for repeatable correctness and performance comparisons')
    parser.add_argument('--skip-uninitialized-fill', action='store_true',
                        help='omit deterministic allocator poison fills; deterministic algorithms remain required')
    parser.add_argument('--verify-numerics', action='store_true',
                        help='save full local gradients/parameter deltas after measurement; correctness run only')
    parser.add_argument('--delay-gradient-sync', action='store_true')
    parser.add_argument('--retain-between-microbatches', action='store_true',
                        help='retain FSDP parameters until the final backward of each update; gradient reductions stay unchanged')
    parser.add_argument('--forward-prefetch', action='store_true',
                        help='issue the next FSDP block all-gather earlier from the CPU')
    parser.add_argument('--fsdp-blocks-per-group', type=int, default=1,
                        help='adjacent decoder blocks per standard FSDP communication group')
    parser.add_argument('--hybrid-shard-size', type=int, default=0,
                        help='HSDP shard dimension; zero keeps the existing full-world FSDP mesh')
    parser.add_argument('--defer-replica-reduction', action='store_true',
                        help='HSDP only: reduce-scatter each microbatch, all-reduce only at the update boundary')
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--drain-before-planned-interruption', action='store_true',
                        help='planned restart only: finish earlier saves after useful intervening updates')
    parser.add_argument('--checkpoint-skip-every', type=int, default=0,
                        help='retain activations in every Nth decoder layer instead of checkpointing it')
    parser.add_argument('--save-alternate-mlp-projections', action='store_true',
                        help='standard selective checkpoint contexts save gate/up GEMMs in alternate layers')
    parser.add_argument('--checkpoint-mode', choices=['none', 'sync', 'async', 'process'], default='none')
    parser.add_argument('--checkpoint-writer-threads', type=int, default=1)
    parser.add_argument('--checkpoint-copy-ahead-bytes', type=int, default=10_000_000,
                        help='DCP synchronous writer GPU-to-CPU copy-ahead bytes per writer thread')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--interrupt-after-update', type=int)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--node-local-output', action='store_true',
                        help='per-node results; checkpoint mode requires a single node')
    parser.add_argument('--cpu-fixture', action='store_true')
    parser.add_argument('--allocation-label', required=True,
                        help='resource class/assignment; retain Slurm and GPU inventory separately')
    return parser.parse_args()


def omit_right_padding_attention_mask(batch):
    """Keep sequence shape and loss masking; valid causal queries precede padding."""
    mask = batch['attention_mask']
    if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask[:, 1:] > mask[:, :-1]):
        raise ValueError('causal mask omission requires binary right padding')
    if not torch.all(mask[:, 0] == 1) or torch.any(batch['labels'][mask == 0] != -100):
        raise ValueError('padding must follow valid tokens and have ignored labels')
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    if not torch.equal(batch['position_ids'], positions):
        raise ValueError('causal mask omission requires independent zero-based document positions')
    return {key: value for key, value in batch.items() if key != 'attention_mask'}


def compile_pointwise_modules(model, scope, backend, options):
    """Compile only the selected Qwen3 norms/activations, never projections."""
    for block in model.model.layers:
        norms = (block.input_layernorm, block.post_attention_layernorm,
                 block.self_attn.q_norm, block.self_attn.k_norm)
        modules = (() if scope == 'activations' else norms)
        if scope != 'norms':
            modules += (block.mlp.act_fn,)
        for module in modules:
            module.compile(dynamic=True, backend=backend, options=options)
    if scope != 'activations':
        model.model.norm.compile(dynamic=True, backend=backend, options=options)


def main():
    args = parse_args()
    os.environ['HF_ENABLE_PARALLEL_LOADING'] = 'true' if args.parallel_model_loading else 'false'
    os.environ['HF_PARALLEL_LOADING_WORKERS'] = '3'
    cfg = json.loads(args.config.read_text())
    if torch.__version__.split('+')[0] != '2.9.1':
        raise ValueError('this recipe requires torch==2.9.1')
    import transformers
    if transformers.__version__ != '4.57.6':
        raise ValueError('this recipe requires transformers==4.57.6')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if args.hybrid_shard_size and (args.cpu_fixture or args.hybrid_shard_size < 2
                                  or world % args.hybrid_shard_size or world <= args.hybrid_shard_size):
        raise ValueError('HSDP requires GPU execution and a nontrivial shard divisor of world size')
    if args.fsdp_blocks_per_group < 1:
        raise ValueError('FSDP blocks per group must be positive')
    if args.save_alternate_mlp_projections and (
            not args.activation_checkpointing or args.checkpoint_skip_every
            or args.compile_blocks or args.compile_pointwise or args.compile_loss):
        raise ValueError('operator checkpoint policy requires eager execution and full layer checkpointing')
    if args.checkpoint_skip_every and (args.checkpoint_skip_every < 2 or not args.activation_checkpointing):
        raise ValueError('selective layer checkpointing requires activation checkpointing and interval >= 2')
    if args.liger_fp32_accumulation and not args.liger_linear_ce:
        raise ValueError('Liger FP32 accumulation requires fused linear CE')
    if args.liger_linear_ce and (args.compile_loss or args.cpu_fixture or cfg['model_id'] != 'Qwen/Qwen2.5-7B'):
        raise ValueError('linear CE candidate requires Qwen2.5-7B GPU execution without compile-loss')
    if args.fsdp_blocks_per_group != 1 and (args.cpu_fixture or args.forward_prefetch
                                          or args.compile_blocks or args.compile_pointwise or args.compile_loss):
        raise ValueError('grouped FSDP requires GPU execution without compile or explicit prefetch')
    if args.workers < 0 or not 0 <= cfg['warmup_updates'] < cfg['updates']:
        raise ValueError('invalid workers or warmup/update interval')
    if args.compile_preserve_casts and not (args.compile_blocks or args.compile_loss or args.compile_pointwise):
        raise ValueError('cast preservation requires compiled operators')
    if args.compile_pointwise_scope != 'all' and (not args.compile_pointwise or args.compile_blocks):
        raise ValueError('pointwise scope requires pointwise compilation without block compilation')
    if args.compile_backend == 'aot_eager' and (
            not args.verify_numerics or args.compile_preserve_casts or args.cpu_fixture
            or not (args.compile_blocks or args.compile_loss or args.compile_pointwise)):
        raise ValueError('aot_eager requires compiled GPU operators in a correctness run without Inductor cast options')
    if (args.interrupt_after_update is not None
            and not 0 < args.interrupt_after_update < cfg['updates']):
        raise ValueError('interruption must precede the final update')
    checkpoints = cfg['checkpoint_updates']
    if (sorted(set(checkpoints)) != checkpoints or not checkpoints
            or checkpoints[0] < 1 or checkpoints[-1] != cfg['updates']):
        raise ValueError('checkpoint update indices must include the final update')
    accumulation = accumulation_steps(cfg['global_batch_samples'], args.microbatch, world)
    if args.defer_replica_reduction and (not args.hybrid_shard_size or args.delay_gradient_sync or accumulation < 2):
        raise ValueError('replica deferral requires HSDP accumulation with local gradient synchronization enabled')
    if args.retain_between_microbatches and (args.cpu_fixture or args.reshard_after_forward or accumulation < 2):
        raise ValueError('microbatch retention requires GPU accumulation and --no-reshard-after-forward')
    if args.deterministic:
        # Set before CUDA initialization. Apply the same policy to every run in
        # a comparison, including performance runs, rather than only fixtures.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    if args.skip_uninitialized_fill:
        if not args.deterministic:
            raise ValueError('skipping allocator fills requires deterministic algorithms')
    torch.utils.deterministic.fill_uninitialized_memory = not args.skip_uninitialized_fill
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
            if args.resume and args.resume_from_config:
                model_config = AutoConfig.from_pretrained(
                    source, revision=cfg['model_revision'], token=False,
                    trust_remote_code=False, local_files_only=bool(args.model_path))
                with no_init_weights():
                    model = AutoModelForCausalLM.from_config(
                        model_config, torch_dtype=torch.float32, attn_implementation=args.attention_implementation,
                        trust_remote_code=False)
                model.tie_weights()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    source, revision=cfg['model_revision'], token=False, trust_remote_code=False,
                    torch_dtype=torch.float32, attn_implementation=args.attention_implementation,
                    local_files_only=bool(args.model_path))
    model.config.use_cache = False
    if args.liger_linear_ce:
        from importlib.metadata import version
        if args.cpu_fixture or model.config.model_type != 'qwen2' or version('liger-kernel') != '0.6.4':
            raise ValueError('linear CE candidate requires Qwen2 and liger-kernel 0.6.4')
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2(model=model, rope=False, rms_norm=False,
                                   swiglu=False, cross_entropy=False, fused_linear_cross_entropy=True)
    if model.config.attention_dropout != 0:
        raise ValueError('microbatch comparisons require dropout-free chosen model')
    if args.activation_checkpointing:
        if args.checkpoint_skip_every > len(model.model.layers):
            raise ValueError('checkpoint skip interval exceeds decoder layer count')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if args.save_alternate_mlp_projections:
            from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy
            def projection_policy(ctx, op, *operands, **kwargs):
                if (op == torch.ops.aten.mm.default and len(operands) == 2
                        and operands[0].shape[-1] == model.config.hidden_size
                        and operands[1].shape[-1] == model.config.intermediate_size):
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE
            for index, layer in enumerate(model.model.layers):
                if index % 2 == 0:
                    layer._gradient_checkpointing_func = partial(
                        checkpoint, use_reentrant=False,
                        context_fn=partial(create_selective_checkpoint_contexts, projection_policy))
        if args.checkpoint_skip_every:
            for index, layer in enumerate(model.model.layers):
                if (index + 1) % args.checkpoint_skip_every == 0:
                    layer.gradient_checkpointing = False
    if not args.cpu_fixture:
        compile_options = {'emulate_precision_casts': True} if args.compile_preserve_casts else {}
        if args.compile_pointwise:
            compile_pointwise_modules(model, args.compile_pointwise_scope, args.compile_backend, compile_options)
        if args.compile_loss:
            model.loss_function = torch.compile(model.loss_function, dynamic=True,
                                                backend=args.compile_backend, options=compile_options)
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        mesh = (init_device_mesh('cuda', (world // args.hybrid_shard_size, args.hybrid_shard_size),
                                 mesh_dim_names=('replicate', 'shard'))
                if args.hybrid_shard_size else None)
        blocks = list(model.model.layers)
        for start in range(0, len(blocks), args.fsdp_blocks_per_group):
            group = blocks[start:start + args.fsdp_blocks_per_group]
            with record_function('setup/shard_block'):
                fully_shard(group[0] if len(group) == 1 else group,
                            mesh=mesh, mp_policy=policy, reshard_after_forward=args.reshard_after_forward)
        with record_function('setup/shard_root'):
            fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=args.reshard_after_forward)
        if args.forward_prefetch:
            blocks = list(model.model.layers)
            for block, following in zip(blocks, blocks[1:]):
                block.set_modules_to_forward_prefetch([following])
        if args.compile_blocks:
            for block in model.model.layers:
                block.compile(dynamic=True, backend=args.compile_backend, options=compile_options)
    elif world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    if args.cpu_fixture and (args.compile_blocks or args.compile_loss or args.compile_pointwise or args.fused_optimizer):
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
                         parallel_model_loading=args.parallel_model_loading,
                         resume_from_config=args.resume_from_config,
                         trim_padding=args.trim_padding, deterministic=args.deterministic,
                         causal_right_padding=args.causal_right_padding,
                         fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
                         workers=args.workers, optimizer='fused' if args.fused_optimizer else 'stock',
                         compile_blocks=args.compile_blocks, compile_loss=args.compile_loss,
                         compile_pointwise=args.compile_pointwise,
                         compile_pointwise_scope=args.compile_pointwise_scope,
                         compile_backend=args.compile_backend,
                         compile_preserve_casts=args.compile_preserve_casts,
                         compile_dynamic=True, compiler_threads=os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'),
                         reshard_after_forward=args.reshard_after_forward, attention=args.attention_implementation,
                         delay_gradient_sync=args.delay_gradient_sync,
                         retain_between_microbatches=args.retain_between_microbatches,
                         nccl_protocol=os.environ.get('NCCL_PROTO', 'auto'),
                         nccl_algorithm=os.environ.get('NCCL_ALGO', 'auto'),
                         requested_nccl_network=os.environ.get('NCCL_NET', 'auto'),
                         requested_nccl_net_plugin=os.environ.get('NCCL_NET_PLUGIN', 'auto'),
                         requested_nccl_gin_plugin=os.environ.get('NCCL_GIN_PLUGIN', 'auto'),
                         requested_fabric_provider=os.environ.get('FI_PROVIDER', 'auto'),
                         requested_ofi_protocol=os.environ.get('OFI_NCCL_PROTOCOL', 'auto'),
                         forward_prefetch=args.forward_prefetch,
                         fsdp_blocks_per_group=args.fsdp_blocks_per_group,
                         hybrid_shard_size=args.hybrid_shard_size,
                         liger_linear_ce=args.liger_linear_ce,
                         liger_fp32_accumulation=args.liger_fp32_accumulation,
                         defer_replica_reduction=args.defer_replica_reduction,
                         activation_checkpointing=args.activation_checkpointing,
                         checkpoint_skip_every=args.checkpoint_skip_every,
                         save_alternate_mlp_projections=args.save_alternate_mlp_projections,
                         checkpoint_mode=args.checkpoint_mode,
                         drain_before_planned_interruption=args.drain_before_planned_interruption,
                         checkpoint_writer_threads=args.checkpoint_writer_threads,
                         checkpoint_copy_ahead_bytes=args.checkpoint_copy_ahead_bytes,
                         parallelism='FSDP2' if not args.cpu_fixture else 'CPU fixture')
    if args.node_local_output and args.checkpoint_mode != 'none' and world != int(os.environ.get('LOCAL_WORLD_SIZE', world)):
        raise ValueError('node-local checkpoint tests require a single node; multi-node DCP needs shared storage')
    if (local == 0 if args.node_local_output else rank == 0):
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    saves = Checkpoints(args.output / 'checkpoints', model, optimizer, scheduler,
                        asynchronous=args.checkpoint_mode in ('async', 'process'), group=checkpoint_group,
                        writer_threads=args.checkpoint_writer_threads,
                        copy_ahead_bytes=args.checkpoint_copy_ahead_bytes,
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
        record_shapes=True, profile_memory=True, with_stack=False,
        on_trace_ready=lambda trace: trace.export_chrome_trace(
            str(args.output / f'trace-rank-{rank}.json')),
    ) if args.profile else nullcontext()
    def local_tensor(tensor):
        if args.verify_numerics and args.hybrid_shard_size:
            # Validate replicas before gathering; count each global slice once
            # in the existing same-rank numerical comparator.
            shard = tensor.to_local().detach().cpu().contiguous()
            digest = hashlib.sha256(shard.numpy().tobytes()).hexdigest()
            replicas = [None] * (world // args.hybrid_shard_size)
            dist.all_gather_object(replicas, digest, group=mesh.get_group('replicate'))
            if len(set(replicas)) != 1:
                raise ValueError('HSDP replicated numerical state differs')
            full = tensor.detach().full_tensor()
            chunk = (full.shape[0] + world - 1) // world
            start = min(rank * chunk, full.shape[0])
            return full.narrow(0, start, min(chunk, full.shape[0] - start)).clone()
        return tensor.to_local() if hasattr(tensor, 'to_local') else tensor

    initial_parameters = ({name: local_tensor(parameter).detach().cpu().clone()
                           for name, parameter in model.named_parameters()} if args.verify_numerics else None)
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
                    if args.trim_padding:
                        batch = trim_padding(batch)
                    if args.causal_right_padding:
                        batch = omit_right_padding_attention_mask(batch)
                with record_function('input/H2D'):
                    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                sync = not args.delay_gradient_sync or micro == accumulation - 1
                if not args.cpu_fixture:
                    model.set_requires_gradient_sync(sync)
                    if args.defer_replica_reduction:
                        model.set_requires_all_reduce(micro == accumulation - 1)
                    if args.retain_between_microbatches:
                        # Reshard before optimizer.step so it updates the same
                        # FP32 shards; do not change gradient reduction order.
                        model.set_reshard_after_backward(micro == accumulation - 1)
                context = model.no_sync() if args.cpu_fixture and world > 1 and not sync else nullcontext()
                with context:
                    with record_function('forward_and_loss'):
                        # Transformers' standard loss shifts labels, ignores -100
                        # and sums before dividing by num_items_in_batch. FSDP/DDP
                        # averages gradients, so compensate by the actual DP size.
                        loss_options = {'accum_dtype': torch.float32} if args.liger_fp32_accumulation else {}
                        loss = model(**batch, num_items_in_batch=normalizer, **loss_options).loss
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
                if args.drain_before_planned_interruption:
                    with record_function('checkpoint/planned_restart_flush'):
                        saves.finish()
                    row['durable_update'] = (saves.durable_progress or {}).get('completed_updates', 0)
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
                if profile is not None and not args.drain_before_planned_interruption:
                    profile.stop()
                    dist.barrier()
                if args.drain_before_planned_interruption:
                    close_checkpointers()
                    dist.destroy_process_group()
                    raise SystemExit(75)
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
    if args.verify_numerics:
        numerical = {
            'parameters': {name: local_tensor(p).detach().cpu() for name, p in model.named_parameters()},
            'gradients': {name: local_tensor(p.grad).detach().cpu() for name, p in model.named_parameters() if p.grad is not None},
        }
        numerical['parameter_deltas'] = {name: value - initial_parameters[name] for name, value in numerical['parameters'].items()}
        torch.save(numerical, args.output / f'numerics-rank-{rank}.pt')
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
                      profiled=args.profile, numerical_verification=args.verify_numerics,
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
            result['training_goodput'] = training_goodput(
                initial_tokens, saves.durable_progress['useful_tokens'], segment_seconds,
                segment_seconds * allocation['gpu_count'],
                scope='restore through training and final flush; excludes model loading, sharding, optimizer construction, historical interruption and queue time')
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    finally:
        close_checkpointers()
