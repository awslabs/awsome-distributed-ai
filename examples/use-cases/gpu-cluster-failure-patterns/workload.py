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


def fill_checkpoint_path(path):
    # Never fill an unbounded shared filesystem. A larger/unconfigured quota
    # causes this bounded attempt to fail without arming the workload fault.
    written = 0
    exhausted = False
    try:
        with (path / 'aim344-filler.bin').open('xb', buffering=0) as output:
            block = b'\0' * (1024 * 1024)
            while written < 64 * 1024 * 1024:
                written += output.write(block)
            os.fsync(output.fileno())
    except OSError as error:
        if error.errno not in (errno.ENOSPC, errno.EDQUOT):
            raise
        exhausted = True
        print(f'checkpoint injection: {error}; filler_bytes={written} B', flush=True)
    if not exhausted:
        (path / 'aim344-filler.bin').unlink()
        raise RuntimeError('No exhaustion within the 64 MiB fill limit; configure the isolated quota.')


def collective(tensor):
    tensor.fill_(1)
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    if not torch.all(tensor == dist.get_world_size()).item():
        raise RuntimeError('Collective correctness mismatch.')


def storage(args, tensor):
    rank = dist.get_rank()
    path = checkpoint_dir()
    start_step = 0
    if rank == 0:
        if args.resume:
            start_step = json.loads((path / 'last.json').read_text())['next_step']
            print(f'Resuming from last successful checkpoint: next_step={start_step} (step index).', flush=True)
        else:
            save_checkpoint(path, 0)
    state = torch.tensor([start_step], device='cuda')
    dist.broadcast(state, src=0)
    start_step = int(state.item())
    if args.inject and rank == 0:
        fill_checkpoint_path(path)
    # Healthy ranks enter the collective while the checkpoint writer retries.
    # This is an explicit retry policy, not an intrinsic property of ENOSPC.
    if rank == 0:
        deadline = time.monotonic() + 60
        while True:
            try:
                save_checkpoint(path, start_step + 1)
                break
            except OSError as error:
                if error.errno not in (errno.ENOSPC, errno.EDQUOT):
                    raise
                print(f'checkpoint write retry: {error}', flush=True)
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
    collective(tensor)
    if rank == 0:
        print(f'Storage workload completed: next_step={start_step + 1} (step index); 0 mismatches.', flush=True)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['storage', 'dataloader', 'device'])
    parser.add_argument('--duration-seconds', type=int, default=600)
    parser.add_argument('--inject', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start-method', choices=['fork', 'spawn'], default='fork')
    parser.add_argument('--epochs', type=positive_epochs, default=1,
                        help='DataLoader epochs (default: 1); numbered controls use 2')
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=30))
    tensor = torch.ones(32 * 1024 * 1024 if args.mode == 'device' else 1024 * 1024, device='cuda')
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
        # Progress comes only after a completed, correctness-checked collective.
        # The external controller observes progress and device counters before injection.
        deadline = time.monotonic() + args.duration_seconds
        sequence = 0
        keep_running = torch.ones(1, dtype=torch.int32, device='cuda')
        while True:
            # Rank 0 ends the loop collectively; local deadlines can diverge.
            if dist.get_rank() == 0:
                keep_running.fill_(int(time.monotonic() < deadline))
            dist.broadcast(keep_running, src=0)
            if not keep_running.item():
                break
            collective(tensor)
            sequence += 1
            if dist.get_rank() == 0 and sequence % 20 == 0:
                print(f'AIM344 collective_progress={sequence} collectives; '
                      f'tensor_bytes={tensor.numel() * tensor.element_size()} B; '
                      f'timestamp={time.time():.6f} s since epoch; 0 mismatches; '
                      f'job_id={os.environ.get("SLURM_JOB_ID", "unassigned")}; '
                      f'world_size={dist.get_world_size()}; rank=0.', flush=True)
        if dist.get_rank() == 0:
            print(f'Device workload completed: {sequence} collectives; 0 mismatches.', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
