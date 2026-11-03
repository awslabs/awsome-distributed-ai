#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Small real NCCL workloads for checkpoint exhaustion and late DataLoader workers."""
import argparse
import ctypes
from datetime import timedelta
import errno
from functools import partial
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

# Every rank's tensors live on its GPU. The CPU unit tests point this at the CPU.
DEVICE = 'cuda'
# A name that an interrupted save of a real job could leave, so that a listing of
# the directory does not say the lab filled it. recover-checkpoint.py removes it.
FILLER_NAME = 'ckpt-shard-00017.partial'
# The lab's own lines, such as the injector's, carry this marker when the launcher
# runs an incident without naming its cause (job.sh). That launcher moves them to
# the lab log instead of the job log. Without --mark-lab-lines they print plainly.
LAB_LINE_MARKER = 'AIM344 lab: '
# With --end-after-save, the steps that run after the first periodic save: one
# more progress record.
END_AFTER_SAVE_STEPS = 20


def lab_line(args, text):
    prefix = LAB_LINE_MARKER if getattr(args, 'mark_lab_lines', False) else ''
    print(prefix + text, flush=True)


def checkpoint_dir():
    path = Path('/checkpoints')
    if (path / '.aim344-fixture').read_text().strip() != 'AIM344 isolated checkpoint fixture':
        raise RuntimeError('The facilitator must prepare an isolated checkpoint fixture.')
    return path


def save_checkpoint(path, step):
    # Preserve the last successful checkpoint if a subsequent write fails.
    with (path / 'next.json').open('w') as output:
        json.dump({'next_step': step}, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(path / 'next.json', path / 'last.json')


def fill_checkpoint_path(path, args):
    # Never fill an unbounded shared filesystem. A larger/unconfigured quota
    # causes this bounded attempt to fail without arming the workload fault.
    written = 0
    exhausted = False
    filler = path / FILLER_NAME
    try:
        with filler.open('xb', buffering=0) as output:
            block = b'\0' * (1024 * 1024)
            while written < 64 * 1024 * 1024:
                written += output.write(block)
            os.fsync(output.fileno())
    except OSError as error:
        if error.errno not in (errno.ENOSPC, errno.EDQUOT):
            raise
        exhausted = True
        lab_line(args, f'checkpoint injection: {error}; filler_bytes={written} B')
    if not exhausted:
        filler.unlink()
        raise RuntimeError('No exhaustion within the 64 MiB fill limit; configure the isolated quota.')


def save_with_retry(path, step):
    # Healthy ranks enter the collective while the checkpoint writer retries.
    # This is an explicit retry policy, not an intrinsic property of ENOSPC.
    deadline = time.monotonic() + 60
    while True:
        try:
            save_checkpoint(path, step)
            return
        except OSError as error:
            if error.errno not in (errno.ENOSPC, errno.EDQUOT):
                raise
            print(f'checkpoint write retry: {error}', flush=True)
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)


def checkpoint_start(args, path):
    """The step every rank starts from, read or recorded by the checkpoint writer.

    Only the writer uses its node's checkpoint directory: on --resume it reads the
    last successful checkpoint, otherwise it records step 0. It then broadcasts the
    step, so every rank starts from the writer's value.
    """
    start_step = 0
    if dist.get_rank() == args.writer_rank:
        if args.resume:
            start_step = json.loads((path / 'last.json').read_text())['next_step']
            print(f'Resuming from last successful checkpoint: next_step={start_step} (step index).', flush=True)
        else:
            save_checkpoint(path, 0)
    state = torch.tensor([start_step], device=DEVICE)
    dist.broadcast(state, src=args.writer_rank)
    return int(state.item())


def collective(tensor):
    tensor.fill_(1)
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    if not torch.all(tensor == dist.get_world_size()).item():
        raise RuntimeError('Collective correctness mismatch.')


def storage(args, tensor):
    rank = dist.get_rank()
    path = checkpoint_dir()
    start_step = checkpoint_start(args, path)
    if rank == args.writer_rank:
        if args.inject:
            fill_checkpoint_path(path, args)
        save_with_retry(path, start_step + 1)
    collective(tensor)
    if rank == 0:
        print(f'Storage workload completed: next_step={start_step + 1} (step index); 0 mismatches.', flush=True)


def run_steps(args, tensor):
    """The training stand-in: one broadcast and one checked all-reduce per step.

    Rank 0 decides when to stop and prints a progress record every 20 collectives;
    the external controller reads those records before an active device fault.
    With --checkpoint-every N, the checkpoint writer saves the next step index
    before the all-reduce of every N-th step. A save that fails keeps that one rank
    out of the all-reduce that the other ranks have entered. --inject fills the
    writer's directory just before its first periodic save. --end-after-save ends
    the job END_AFTER_SAVE_STEPS steps after the first periodic save past the start
    step, so that a resumed job shows one save whatever its rate;
    --duration-seconds still bounds it.
    """
    rank = dist.get_rank()
    writer = bool(args.checkpoint_every) and rank == args.writer_rank
    path = checkpoint_dir() if writer else None
    start_step = checkpoint_start(args, path) if args.checkpoint_every else 0
    if writer and not args.resume:
        print(f'checkpoint saved: next_step={start_step} (step index)', flush=True)
    end_step = None
    if args.end_after_save:
        first_save = (start_step // args.checkpoint_every + 1) * args.checkpoint_every
        end_step = first_save + END_AFTER_SAVE_STEPS
    injected = False
    # Progress comes only after a completed, correctness-checked collective.
    # The external controller observes progress and device counters before injection.
    deadline = time.monotonic() + args.duration_seconds
    sequence = 0
    keep_running = torch.ones(1, dtype=torch.int32, device=DEVICE)
    while True:
        # Rank 0 ends the loop collectively; local deadlines can diverge.
        if rank == 0:
            keep_running.fill_(int(time.monotonic() < deadline
                                   and (end_step is None or start_step + sequence < end_step)))
        dist.broadcast(keep_running, src=0)
        if not keep_running.item():
            break
        step = start_step + sequence
        if writer and sequence and step % args.checkpoint_every == 0:
            if args.inject and not injected:
                injected = True
                fill_checkpoint_path(path, args)
            save_with_retry(path, step)
            print(f'checkpoint saved: next_step={step} (step index)', flush=True)
        collective(tensor)
        sequence += 1
        if rank == 0 and sequence % 20 == 0:
            print(f'AIM344 collective_progress={sequence} collectives; '
                  f'tensor_bytes={tensor.numel() * tensor.element_size()} B; '
                  f'timestamp={time.time():.6f} s since epoch; 0 mismatches; '
                  f'job_id={os.environ.get("SLURM_JOB_ID", "unassigned")}; '
                  f'world_size={dist.get_world_size()}; rank=0.', flush=True)
    if rank == 0:
        print(f'Device workload completed: {sequence} collectives; 0 mismatches.', flush=True)


def dataloader_event(event, rank, epoch, **fields):
    print('DataLoader lifecycle: ' + json.dumps(
        dict(event=event, rank=rank, host=socket.gethostname(), pid=os.getpid(),
             epoch=epoch, **fields), sort_keys=True), flush=True)


def dataloader_worker_init(worker_id, *, rank, epoch, parent_pid):
    # Module-level + partial is spawn-picklable; no CUDA/distributed calls here.
    dataloader_event('worker_init', rank, epoch, worker_id=worker_id,
                     parent_pid=parent_pid)


def positive_epochs(value):
    try:
        epochs = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('epochs must be a positive integer') from None
    if epochs < 1:
        raise argparse.ArgumentTypeError('epochs must be a positive integer')
    return epochs


def dataloader(args, tensor):
    # A completed collective above forces lazy NCCL initialization and memory
    # registration before workers start. Keep that GPU allocation alive.
    get_env = ctypes.CDLL(None).getenv
    get_env.argtypes = [ctypes.c_char_p]
    get_env.restype = ctypes.c_char_p
    rank = dist.get_rank()
    if rank == 0:
        value = get_env(b'FI_EFA_FORK_SAFE')
        print(f'After first collective: C environment FI_EFA_FORK_SAFE={value!r}; '
              f'DataLoader start_method={args.start_method}.', flush=True)
    loader = DataLoader(TensorDataset(torch.arange(32)), batch_size=8,
                        num_workers=2, multiprocessing_context=args.start_method,
                        persistent_workers=False)
    expected_batches = 4
    batches = 0
    for epoch in range(1, args.epochs + 1):
        loader.worker_init_fn = partial(dataloader_worker_init, rank=rank,
                                        epoch=epoch, parent_pid=os.getpid())
        log = partial(dataloader_event, rank=rank, epoch=epoch,
                      expected_batches=expected_batches)
        log('iterator_start')
        iterator = iter(loader)
        log('iterator_created')
        epoch_batches = 0
        for batch in iterator:
            log('batch_received', batch=epoch_batches + 1)
            expected = torch.arange(epoch_batches * 8, (epoch_batches + 1) * 8)
            if len(batch) != 1 or not torch.equal(batch[0], expected):
                raise RuntimeError('DataLoader batch correctness mismatch.')
            collective(tensor)
            epoch_batches += 1
            batches += 1
            log('collective_complete', batch=epoch_batches)
        # Exhaustion joins nonpersistent workers before the next iter(loader).
        if epoch_batches != expected_batches:
            raise RuntimeError(f'Expected {expected_batches} batches; got {epoch_batches}.')
        log('iterator_end', batches=epoch_batches)
        del iterator
    if rank == 0:
        print(f'DataLoader workload completed: {batches} batches; 0 mismatches.', flush=True)


def nonnegative_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a non-negative integer') from None
    if number < 0:
        raise argparse.ArgumentTypeError('must be a non-negative integer')
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['storage', 'dataloader', 'device'])
    parser.add_argument('--duration-seconds', type=int, default=600)
    parser.add_argument('--inject', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--writer-rank', type=nonnegative_int, default=None,
                        help='storage and device modes: rank that writes the checkpoint (default: 0)')
    parser.add_argument('--checkpoint-every', type=nonnegative_int, default=0,
                        help='device mode: save a checkpoint every N steps (default: 0, none)')
    parser.add_argument('--end-after-save', action='store_true',
                        help='device mode: end 20 steps after the first periodic save past the '
                             'start step, or at --duration-seconds if that comes first')
    parser.add_argument('--mark-lab-lines', action='store_true',
                        help="storage and device modes: mark the lab's own lines for the launcher's lab log")
    parser.add_argument('--start-method', choices=['fork', 'spawn'], default='fork')
    parser.add_argument('--epochs', type=positive_epochs, default=1,
                        help='DataLoader epochs (default: 1); numbered controls use 2')
    args = parser.parse_args(argv)
    if args.mode == 'dataloader':
        # The DataLoader launchers pass extra arguments through, so a checkpoint
        # option given by mistake must fail rather than run the unchanged control.
        given = [flag for flag, value in (('--writer-rank', args.writer_rank is not None),
                                          ('--inject', args.inject), ('--resume', args.resume),
                                          ('--mark-lab-lines', args.mark_lab_lines)) if value]
        if given:
            parser.error(f'{", ".join(given)}: not used in dataloader mode')
    if args.writer_rank is None:
        args.writer_rank = 0
    if args.checkpoint_every and args.mode != 'device':
        parser.error('--checkpoint-every applies to device mode only')
    if args.mode == 'device' and (args.inject or args.resume) and not args.checkpoint_every:
        parser.error('--inject and --resume in device mode need --checkpoint-every')
    if args.end_after_save and (args.mode != 'device' or not args.checkpoint_every):
        parser.error('--end-after-save applies to device mode with --checkpoint-every')
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=30))
    if args.writer_rank >= dist.get_world_size():
        raise SystemExit(f'--writer-rank {args.writer_rank} is not a rank of this '
                         f'{dist.get_world_size()}-rank job.')
    tensor = torch.ones(32 * 1024 * 1024 if args.mode == 'device' else 1024 * 1024, device=DEVICE)
    collective(tensor)
    if dist.get_rank() == 0:
        build_nccl = '.'.join(str(part) for part in torch.cuda.nccl.version())
        print(f'torch={torch.__version__}; CUDA={torch.version.cuda}; '
              f'torch_build_nccl={build_nccl}; world_size={dist.get_world_size()} ranks. '
              'Use NCCL startup logs for the loaded runtime version.', flush=True)
    if args.mode == 'storage':
        storage(args, tensor)
    elif args.mode == 'dataloader':
        dataloader(args, tensor)
    else:
        run_steps(args, tensor)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
