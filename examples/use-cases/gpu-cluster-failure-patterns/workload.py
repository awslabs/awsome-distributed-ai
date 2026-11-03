#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Small real NCCL workloads: checkpoint exhaustion, DataLoader workers, and a
cross-rank replica check of one matrix product with an optional software bit flip."""
import argparse
from collections import Counter
import ctypes
from datetime import timedelta
import errno
import faulthandler
import glob
from functools import partial
import json
import multiprocessing.util
import os
from pathlib import Path
import socket
import sys
import threading
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, TensorDataset, get_worker_info

# The lab controls hand the launcher one incident profile; the launcher exports the
# profile's settings as JSON in this variable. workload.py applies them as if they
# were options, so they appear neither on the job's command line nor in its log.
INCIDENT_SETTINGS_VARIABLE = 'AIM344_INCIDENT_SETTINGS'
# Stack dumps go to the container's mount of RESULTS_DIR, which is node-local.
STACK_DUMP_DIR = '/results'
# How long a cache refill holds the dataset's lock.
REFILL_SECONDS = 2.0
# cuBLAS gives bitwise identical GEMM results on one toolkit, one GPU model and a
# single stream; one of these workspace settings removes the multi-stream exception
# and is what torch.use_deterministic_algorithms(True) requires for torch.mm.
CUBLAS_DETERMINISTIC_WORKSPACES = (':4096:8', ':16:8')

# Every rank's tensors live on its GPU. The CPU unit tests point this at the CPU.
DEVICE = 'cuda'
# A name that an interrupted save of a real job could leave, so that a listing of
# the directory does not say the lab filled it. recover-checkpoint.py removes it.
FILLER_NAME = 'ckpt-shard-00017.partial'
# With --end-after-save, the steps that run after the first periodic save: one
# more progress record.
END_AFTER_SAVE_STEPS = 20


def emit(text):
    """Print one line of the job's output with a single write.

    torchrun starts every worker with `python -u`, and under unbuffered output
    print() writes the text and the newline in two calls, so another process's
    output can land between them. One write of a line shorter than PIPE_BUF bytes
    reaches the shared pipe whole.
    """
    sys.stdout.write(text + '\n')
    sys.stdout.flush()


def lab_line(lab_dir, text):
    """Print one of the lab's own lines, or, with a lab directory, keep it off the output.

    The lab's own lines (the injector's, the DataLoader lifecycle) name what the lab
    did. When job.sh runs an incident without naming its cause, it passes --lab-dir:
    each process then appends these lines to its own file in that directory, with
    the time, and common.sh collect_lab_records moves the files into the lab log
    after the run. No part of them passes through the job's output, where lines of
    several processes can merge. Each line is one write to a file no other process
    writes, so it is never split.
    """
    if not lab_dir:
        emit(text)
        return
    os.makedirs(lab_dir, mode=0o700, exist_ok=True)
    now = time.time()
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(now)) + f'.{int(now % 1 * 1000):03d}Z'
    path = os.path.join(lab_dir, f'{socket.gethostname()}-pid{os.getpid()}.log')
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(handle, f'{stamp} {text}\n'.encode())
    finally:
        os.close(handle)


def progress_record(count, tensor):
    """Rank 0's progress record, which the controller and the pages read."""
    emit(f'AIM344 collective_progress={count} collectives; '
         f'tensor_bytes={tensor.numel() * tensor.element_size()} B; '
         f'timestamp={time.time():.6f} s since epoch; 0 mismatches; '
         f'job_id={os.environ.get("SLURM_JOB_ID", "unassigned")}; '
         f'world_size={dist.get_world_size()}; rank=0.')


def checkpoint_dir():
    path = Path('/checkpoints')
    if (path / '.aim344-fixture').read_text().strip() != 'AIM344 isolated checkpoint fixture':
        raise RuntimeError('The facilitator must prepare an isolated checkpoint fixture.')
    return path


def save_checkpoint(path, step, saved=None):
    # Preserve the last successful checkpoint if a subsequent write fails.
    # A lab checkpoint holds only its step, so the step loop's writer keeps the
    # steps of the run's earlier checkpoints in `saved`, which a rollback reads.
    record = {'next_step': step}
    if saved is not None:
        record['saved'] = saved
    with (path / 'next.json').open('w') as output:
        json.dump(record, output)
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
        lab_line(getattr(args, 'lab_dir', None), f'checkpoint injection: {error}; filler_bytes={written} B')
    if not exhausted:
        filler.unlink()
        raise RuntimeError('No exhaustion within the 64 MiB fill limit; configure the isolated quota.')


def save_with_retry(path, step, saved=None):
    # Healthy ranks enter the collective while the checkpoint writer retries.
    # This is an explicit retry policy, not an intrinsic property of ENOSPC.
    deadline = time.monotonic() + 60
    while True:
        try:
            save_checkpoint(path, step, saved)
            return
        except OSError as error:
            if error.errno not in (errno.ENOSPC, errno.EDQUOT):
                raise
            emit(f'checkpoint write retry: {error}')
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)


def rollback_point(record, before):
    """The newest checkpoint in `record` saved before step index `before`.

    A checkpoint with next_step=N holds the steps below step index N, and it is
    saved after the replica check of step N. Every checkpoint with N below S was
    therefore saved before the replica check of step S, the step a replica line
    `step=S` names. Returns N and the saved steps up to N; the later checkpoints
    are dropped with the rollback.
    """
    saved = record.get('saved', [record['next_step']])
    earlier = [step for step in saved if step < before]
    if not earlier:
        raise SystemExit(f'No checkpoint was saved before step={before}.')
    start_step = max(earlier)
    return start_step, [step for step in saved if step <= start_step]


def checkpoint_start(args, path):
    """The step every rank starts from, read or recorded by the checkpoint writer.

    Only the writer uses its node's checkpoint directory: on --resume it reads the
    last successful checkpoint, or with --resume-before the newest one saved before
    that step, otherwise it records step 0. It then broadcasts the step, so every
    rank starts from the writer's value. Returns the step and, on the writer, the
    steps of the checkpoints saved so far (None on the other ranks).
    """
    start_step = 0
    saved = None
    if dist.get_rank() == args.writer_rank:
        if args.resume:
            record = json.loads((path / 'last.json').read_text())
            before = getattr(args, 'resume_before', None)
            if before is None:
                start_step = record['next_step']
                saved = record.get('saved', [start_step])
                emit(f'Resuming from last successful checkpoint: next_step={start_step} (step index).')
            else:
                start_step, saved = rollback_point(record, before)
                emit(f'Resuming from the last checkpoint saved before step={before}: '
                     f'next_step={start_step} (step index).')
        else:
            saved = [0]
            save_checkpoint(path, 0, saved)
    state = torch.tensor([start_step], device=DEVICE)
    dist.broadcast(state, src=args.writer_rank)
    return int(state.item()), saved


def collective(tensor):
    tensor.fill_(1)
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    if not torch.all(tensor == dist.get_world_size()).item():
        raise RuntimeError('Collective correctness mismatch.')


def storage(args, tensor):
    rank = dist.get_rank()
    path = checkpoint_dir()
    start_step, _ = checkpoint_start(args, path)
    if rank == args.writer_rank:
        if args.inject:
            fill_checkpoint_path(path, args)
        save_with_retry(path, start_step + 1)
    collective(tensor)
    if rank == 0:
        emit(f'Storage workload completed: next_step={start_step + 1} (step index); 0 mismatches.')


def run_steps(args, tensor):
    """The training stand-in: one broadcast and one checked all-reduce per step.

    Rank 0 decides when to stop and prints a progress record every 20 collectives;
    the external controller reads those records before an active device fault.
    With --replica-check, each step first all-gathers the ranks' checksums of one
    matrix product (ReplicaCheck), checked by step index, and rank 0 prints the
    agreement after each progress record. With --checkpoint-every N, the checkpoint
    writer then saves the next step index before the all-reduce of every N-th step,
    so the all-reduce ends every step. A save that fails keeps that one rank out of
    the all-reduce that the other ranks have entered. --inject fills the writer's
    directory just before its first periodic save. --end-after-save ends the job
    END_AFTER_SAVE_STEPS steps after the first periodic save past the start step,
    so that a resumed job shows one save whatever its rate; --duration-seconds still
    bounds it. With --stack-dump-seconds, each rank writes its stacks when that long
    passes without a progress record.
    """
    rank = dist.get_rank()
    writer = bool(args.checkpoint_every) and rank == args.writer_rank
    path = checkpoint_dir() if writer else None
    start_step, saved = checkpoint_start(args, path) if args.checkpoint_every else (0, None)
    if writer and not args.resume:
        emit(f'checkpoint saved: next_step={start_step} (step index)')
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
    # With --replica-check each step also compares one matrix product across ranks.
    replica = (ReplicaCheck(args, tensor.device, rank=rank, world_size=dist.get_world_size())
               if getattr(args, 'replica_check', False) else None)
    arm_stack_dump(getattr(args, 'stack_dump_seconds', 0),
                   getattr(args, 'stack_dump_dir', STACK_DUMP_DIR), 'rank', rank)
    try:
        while True:
            # Rank 0 ends the loop collectively; local deadlines can diverge.
            if rank == 0:
                keep_running.fill_(int(time.monotonic() < deadline
                                       and (end_step is None or start_step + sequence < end_step)))
            dist.broadcast(keep_running, src=0)
            if not keep_running.item():
                break
            step = start_step + sequence
            if replica is not None:
                replica.step(step)
            if writer and sequence and step % args.checkpoint_every == 0:
                if args.inject and not injected:
                    injected = True
                    fill_checkpoint_path(path, args)
                saved.append(step)
                save_with_retry(path, step, saved)
                emit(f'checkpoint saved: next_step={step} (step index)')
            collective(tensor)
            sequence += 1
            if sequence % 20 == 0:
                rearm_stack_dump()
                if rank == 0:
                    progress_record(sequence, tensor)
                    if replica is not None:
                        replica.report(step)
    finally:
        cancel_stack_dump()
    if rank == 0:
        emit(f'Device workload completed: {sequence} collectives; 0 mismatches.')


def dataloader_event(event, rank, epoch, *, lab_dir=None, **fields):
    # A lifecycle line names the DataLoader, so under --lab-dir it is a lab line.
    lab_line(lab_dir, 'DataLoader lifecycle: ' + json.dumps(
        dict(event=event, rank=rank, host=socket.gethostname(), pid=os.getpid(),
             epoch=epoch, **fields), sort_keys=True))


# ---------------- stack capture ----------------
# faulthandler.dump_traceback_later runs a watchdog thread that holds an internal
# lock until it fires or is cancelled. A process forked while the timer is armed
# copies the held lock without the thread, so arming or cancelling in that child
# blocks forever. The rank process therefore arms only after iter(loader) returned
# (the workers exist by then) and cancels before the next iter(loader); each worker
# arms its own timer in worker_init_fn.
# Device mode forks no process, so its timer can stay armed through the step loop;
# rearm_stack_dump restarts it at each progress record.
_stack_dump = None  # (open file, path, header length, seconds) of this process's armed timer
# A run moves the stack files of an earlier run in the same directory here.
PREVIOUS_STACKS = 'previous-stacks'


def arm_stack_dump(seconds, directory, role, rank):
    """Write every thread's stack of this process to its own file after `seconds`.

    The file is named by host, rank, role and pid so that the 48 dumps of a
    two-node job do not interleave. Returns the path, or None when off.
    """
    global _stack_dump
    if not seconds:
        return None
    cancel_stack_dump()
    path = Path(directory) / f'stacks-{socket.gethostname()}-rank{rank}-{role}-pid{os.getpid()}.txt'
    handle = path.open('w')
    header = (f'host={socket.gethostname()} rank={rank} role={role} pid={os.getpid()} '
              f'armed_at={time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} '
              f'dump_after={seconds} s\n')
    handle.write(header)
    handle.flush()
    faulthandler.dump_traceback_later(seconds, repeat=False, file=handle)
    _stack_dump = (handle, path, len(header.encode()), seconds)
    return path


def rearm_stack_dump():
    """Restart this process's armed timer with its file, after progress was made."""
    if _stack_dump is not None:
        handle, _, _, seconds = _stack_dump
        faulthandler.dump_traceback_later(seconds, repeat=False, file=handle)


def retire_stack_files(directory):
    """Move this host's stack files that an earlier run left in `directory` to PREVIOUS_STACKS.

    A stack read lists the directory's stacks-* files, so after a run that wrote
    none it would otherwise print the earlier run's stacks again. One process per
    node calls this. It moves only the files named with its own host, so that on a
    results directory that both nodes share, two nodes never move the same file.
    """
    host = glob.escape(socket.gethostname())
    old = sorted(Path(directory).glob(f'stacks-{host}-rank[0-9]*.txt'))
    if old:
        kept = Path(directory) / PREVIOUS_STACKS
        kept.mkdir(exist_ok=True)
        for path in old:
            os.replace(path, kept / path.name)


def cancel_stack_dump():
    """Cancel this process's timer. A file that holds only its header is removed."""
    global _stack_dump
    if _stack_dump is None:
        return
    handle, path, header_length, _ = _stack_dump
    _stack_dump = None
    faulthandler.cancel_dump_traceback_later()
    handle.close()
    try:
        if path.stat().st_size == header_length:
            path.unlink()
    except OSError:
        pass


# ---------------- DataLoader datasets ----------------
class CachedDataset(Dataset):
    """32 sequential items read from a cache behind a lock that a refill thread holds.

    `cache_owner` decides which process creates the lock, the cache and the thread:

    - `rank-thread`: the rank process creates them in `__init__`, and `refresh()`
      (called between epochs) starts the refill thread and returns while it holds
      the lock. The next `iter(loader)` forks the workers at that moment, so each
      worker's copy of the lock is held by a thread that does not exist in the
      worker: `__getitem__` waits forever, and the rank waits for the batch. Under
      spawn or forkserver the dataset is pickled for each worker and a lock cannot
      be pickled, so `iter(loader)` fails in epoch 1 instead.
    - `worker-thread`: the process that reads the items creates them on first use,
      refills there, and the dataset drops them when it is pickled. This completes
      under fork, spawn and forkserver.

    `reinit()` replaces the lock and rebuilds the cache in the calling process (the
    worker_init_fn fix, fork only, which must also rebuild the data because the fork
    copied it in the middle of a refill). With `worker_cuda`, `__getitem__` also
    touches CUDA, which a worker forked after the rank initialized CUDA cannot do.
    """

    def __init__(self, cache_owner, *, worker_cuda=False, items=32):
        if cache_owner not in ('rank-thread', 'worker-thread'):
            raise ValueError(f'unknown cache owner {cache_owner!r}')
        self.cache_owner = cache_owner
        self.worker_cuda = worker_cuda
        self.items = items
        self.generation = 0
        self.owner_pid = None
        self.lock = None
        self.cache = None
        self.refill_thread = None
        if cache_owner == 'rank-thread':
            self._create()

    def _create(self):
        self.owner_pid = os.getpid()
        self.lock = threading.Lock()
        self.cache = [torch.tensor(index) for index in range(self.items)]
        self.refill_thread = None

    def _refill(self, started):
        with self.lock:
            started.set()
            # The protected data is rebuilt item by item while the lock is held.
            self.cache = []
            for index in range(self.items):
                self.cache.append(torch.tensor(index))
                time.sleep(REFILL_SECONDS / self.items)

    def _start_refill(self):
        started = threading.Event()
        self.refill_thread = threading.Thread(target=self._refill, args=(started,),
                                              name='aim344-cache-refill', daemon=True)
        self.refill_thread.start()
        started.wait()

    def refresh(self):
        """Begin the next epoch's refill. For `rank-thread` this returns while the
        refill thread holds the lock; for `worker-thread` the reader refills itself."""
        self.generation += 1
        if self.cache_owner == 'rank-thread':
            self._start_refill()

    def reinit(self):
        """Replace the lock and rebuild the cache in this process."""
        self._create()

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.cache_owner == 'worker-thread':
            for name in ('owner_pid', 'lock', 'cache', 'refill_thread'):
                state[name] = None
        return state

    def __len__(self):
        return self.items

    def __getitem__(self, index):
        if self.cache_owner == 'worker-thread' and self.owner_pid != os.getpid():
            self._create()
            self._start_refill()
        if self.worker_cuda:
            torch.zeros(1, device='cuda')
        with self.lock:
            return (self.cache[index],)


def dataloader_worker_init(worker_id, *, rank, epoch, parent_pid, stack_dump_seconds=0,
                           stack_dump_dir=STACK_DUMP_DIR, reinit_cache=False, lab_dir=None):
    # Module-level + partial is spawn-picklable; no CUDA/distributed calls here.
    dataloader_event('worker_init', rank, epoch, worker_id=worker_id,
                     parent_pid=parent_pid, lab_dir=lab_dir)
    if reinit_cache:
        dataset = get_worker_info().dataset
        if hasattr(dataset, 'reinit'):
            dataset.reinit()
    if arm_stack_dump(stack_dump_seconds, stack_dump_dir, f'worker{worker_id}', rank):
        # A worker that finishes its epoch removes its unused stack file on exit.
        multiprocessing.util.Finalize(None, cancel_stack_dump, exitpriority=0)


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
    lab_dir = getattr(args, 'lab_dir', None)
    if rank == 0:
        value = get_env(b'FI_EFA_FORK_SAFE')
        lab_line(lab_dir, f'After first collective: C environment FI_EFA_FORK_SAFE={value!r}; '
                          f'DataLoader start_method={args.start_method}.')
    # The exercise settings default to the healthy control; tests pass a bare namespace.
    cache_owner = getattr(args, 'dataset_cache', 'none')
    affected_rank = getattr(args, 'affected_rank', None)
    affected = affected_rank is None or affected_rank == rank
    stack_dump_seconds = getattr(args, 'stack_dump_seconds', 0)
    stack_dump_dir = getattr(args, 'stack_dump_dir', STACK_DUMP_DIR)
    if cache_owner == 'none':
        dataset = TensorDataset(torch.arange(32))
    else:
        dataset = CachedDataset(cache_owner,
                                worker_cuda=affected and getattr(args, 'worker_cuda', False))
    loader = DataLoader(dataset, batch_size=8,
                        num_workers=2, multiprocessing_context=args.start_method,
                        persistent_workers=False, timeout=getattr(args, 'dataloader_timeout', 0))
    expected_batches = 4
    batches = 0
    for epoch in range(1, args.epochs + 1):
        loader.worker_init_fn = partial(dataloader_worker_init, rank=rank,
                                        epoch=epoch, parent_pid=os.getpid(),
                                        stack_dump_seconds=stack_dump_seconds,
                                        stack_dump_dir=stack_dump_dir,
                                        reinit_cache=getattr(args, 'reinit_in_worker', False),
                                        lab_dir=lab_dir)
        log = partial(dataloader_event, rank=rank, epoch=epoch,
                      expected_batches=expected_batches, lab_dir=lab_dir)
        log('iterator_start')
        if epoch > 1 and affected and cache_owner != 'none':
            # The refill for this epoch begins now; iter(loader) creates the workers
            # while it runs.
            dataset.refresh()
        iterator = iter(loader)
        log('iterator_created')
        arm_stack_dump(stack_dump_seconds, stack_dump_dir, 'rank', rank)
        epoch_batches = 0
        try:
            for batch in iterator:
                log('batch_received', batch=epoch_batches + 1)
                expected = torch.arange(epoch_batches * 8, (epoch_batches + 1) * 8)
                if len(batch) != 1 or not torch.equal(batch[0], expected):
                    raise RuntimeError('DataLoader batch correctness mismatch.')
                collective(tensor)
                epoch_batches += 1
                batches += 1
                log('collective_complete', batch=epoch_batches)
                # The same record as the step loop's, one per batch collective: it
                # counts the job's batches and says nothing about epochs.
                if rank == 0:
                    progress_record(batches, tensor)
        finally:
            # Cancelled before the next iter(loader) forks workers (see arm_stack_dump).
            cancel_stack_dump()
        # Exhaustion joins nonpersistent workers before the next iter(loader).
        if epoch_batches != expected_batches:
            raise RuntimeError(f'Expected {expected_batches} batches; got {epoch_batches}.')
        log('iterator_end', batches=epoch_batches)
        del iterator
    if rank == 0:
        emit(f'DataLoader workload completed: {batches} batches; 0 mismatches.')


def nonnegative_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a non-negative integer') from None
    if number < 0:
        raise argparse.ArgumentTypeError('must be a non-negative integer')
    return number


# ---------------- replica check (silent corruption, simulated in software) ----------------
def device_identity(device):
    """Host, LOCAL_RANK, CUDA device index, UUID and PCI address of this rank's device."""
    identity = dict(host=socket.gethostname(), local_rank=os.environ.get('LOCAL_RANK', 'unset'),
                    index=None, uuid='none', pci='none')
    if device.type != 'cuda':
        return identity
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    uuid = str(getattr(properties, 'uuid', 'unknown'))
    identity['index'] = index
    identity['uuid'] = uuid if uuid.startswith('GPU-') or uuid == 'unknown' else f'GPU-{uuid}'
    if hasattr(properties, 'pci_bus_id'):
        # Upper-case hex, as nvidia-smi prints pci.bus_id, so the two can be matched.
        identity['pci'] = (f'{properties.pci_domain_id:08X}:{properties.pci_bus_id:02X}:'
                           f'{properties.pci_device_id:02X}.0')
    return identity


def require_deterministic(device, environ=os.environ):
    """Make every GEMM of the replica check reproducible bit for bit on identical GPUs."""
    if device.type == 'cuda':
        if environ.get('CUBLAS_WORKSPACE_CONFIG') not in CUBLAS_DETERMINISTIC_WORKSPACES:
            raise SystemExit('The replica check needs CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) '
                             'in the environment before CUDA starts; torch-node.sh sets it.')
        # fp32 inputs stay fp32 in cuBLAS: no TF32 rounding of the mantissa.
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.use_deterministic_algorithms(True)


def int32_bit_mask(bit):
    """The int32 value with only `bit` set; bit 31 is the sign bit of the int32 view."""
    return (1 << bit) - (1 << 32 if bit == 31 else 0)


def nccl_error_recorded(device):
    """Whether this rank's NCCL watchdog has recorded a failed or timed-out collective.

    A collective called without async_op is queued on the current stream, and reading
    its output waits for it. A collective that cannot complete ends only when the
    watchdog, which records the error first, aborts the communicator (torchrun sets
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1). The read then returns whatever part of the
    output had arrived.
    """
    if device.type != 'cuda':
        return False
    return dist.group.WORLD._get_backend(device).get_error().name != 'SUCCESS'


class ReplicaCheck:
    """Every rank computes the same matrix product and the ranks compare checksums.

    Each step every rank builds the same fp32 inputs on the CPU from one seed and
    the step number, copies them to its device, computes `C = A @ B` on the default
    stream and reduces `C`, viewed as int32, to an int64 sum. That sum is bit-exact:
    one changed bit changes it by a power of two, while a tolerance-based comparison
    would pass a flip of the lowest mantissa bits. The sums are all-gathered and each
    rank compares them with the majority value. The job raises nothing: the rank
    whose sum differs prints one line naming its host, LOCAL_RANK, device index,
    UUID and PCI address when the disagreement starts and when it ends, and rank 0's
    report line counts the window's disagreements beside the progress record. A rank
    compares only after an all-gather that completed: once its NCCL watchdog has
    recorded a failed or timed-out collective, neither the rank's lines nor rank 0's
    report line are printed.

    The simulated fault is a software flip of one bit of one element of `C` on the
    selected rank (`--corrupt-rank`), once at step index `--corrupt-step` or, with
    `--corrupt-recurring`, on every step from it on. Its lines name steps by step
    index, as the checkpoint lines do. The GPU is not altered.
    """

    REPORT_PREFIX = 'AIM344 replica check:'

    def __init__(self, args, device, *, rank, world_size, environ=os.environ):
        require_deterministic(device, environ)
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.n = args.replica_n
        self.seed = args.replica_seed
        self.corrupt_rank = args.corrupt_rank
        self.corrupt_step = args.corrupt_step
        self.corrupt_bit = args.corrupt_bit
        self.corrupt_recurring = args.corrupt_recurring
        self.identity = device_identity(device)
        self.gathered = torch.zeros(world_size, dtype=torch.int64, device=device)
        self.differing = set()
        self.window_steps = 0
        self.window_differences = Counter()

    def inputs(self, step):
        generator = torch.Generator().manual_seed(self.seed + step)
        a = torch.randn(self.n, self.n, generator=generator)
        b = torch.randn(self.n, self.n, generator=generator)
        return a.to(self.device), b.to(self.device)

    def corrupts(self, step):
        if self.rank != self.corrupt_rank:
            return False
        return step == self.corrupt_step or (self.corrupt_recurring and step >= self.corrupt_step)

    def checksum(self, step):
        a, b = self.inputs(step)
        product = (a @ b).contiguous()
        words = product.view(-1).view(torch.int32)
        if self.corrupts(step):
            words[words.numel() // 3] ^= int32_bit_mask(self.corrupt_bit)
        return words.to(torch.int64).sum()

    def step(self, step):
        local = self.checksum(step).reshape(1)
        dist.all_gather_into_tensor(self.gathered, local)
        values = self.gathered.tolist()
        if nccl_error_recorded(self.device):
            # The all-gather did not complete: its output is partly the previous step's.
            return
        majority = Counter(values).most_common(1)[0][0]
        differing = {r for r, value in enumerate(values) if value != majority}
        who = (f'rank {self.rank} ({self.identity["host"]}, LOCAL_RANK {self.identity["local_rank"]}, '
               f'CUDA device {self.identity["index"]}, UUID {self.identity["uuid"]}, '
               f'PCI {self.identity["pci"]})')
        if self.rank in differing and self.rank not in self.differing:
            emit(f'{self.REPORT_PREFIX} step={step}; {who} checksum {values[self.rank]} differs '
                 f'from the other {self.world_size - len(differing)} ranks ({majority}).')
        elif self.rank not in differing and self.rank in self.differing:
            emit(f'{self.REPORT_PREFIX} step={step}; {who} checksum agrees again '
                 f'({majority}).')
        self.differing = differing
        self.window_steps += 1
        self.window_differences.update(differing)

    def report(self, step):
        """Rank 0's summary of the steps since the previous report."""
        if nccl_error_recorded(self.device):
            return
        agreeing = self.world_size - len(self.window_differences)
        line = (f'{self.REPORT_PREFIX} step={step}; {agreeing}/{self.world_size} checksums agree '
                f'over the last {self.window_steps} steps')
        for rank in sorted(self.window_differences):
            line += f'; rank {rank} differed on {self.window_differences[rank]} of {self.window_steps} steps'
        emit(line + '.')
        self.window_steps = 0
        self.window_differences = Counter()


# ---------------- settings from the lab controls ----------------
def apply_incident_settings(parser, args, environ=os.environ, argv=None):
    """Apply the incident profile's settings from INCIDENT_SETTINGS_VARIABLE.

    The value is a JSON object whose keys are option names in attribute form
    (`corrupt_rank` for `--corrupt-rank`). Values pass the option's own validation.
    An option given on the command line keeps its value, so a participant's fix
    flag on a rerun wins over the incident's setting. The positional mode cannot be
    set here.
    """
    raw = environ.get(INCIDENT_SETTINGS_VARIABLE)
    if not raw:
        return args
    variable = INCIDENT_SETTINGS_VARIABLE
    argv = sys.argv[1:] if argv is None else argv
    given = {word.split('=', 1)[0] for word in argv if word.startswith('--')}
    try:
        settings = json.loads(raw)
    except ValueError:
        parser.error(f'{variable} is not JSON')
    if not isinstance(settings, dict):
        parser.error(f'{variable} must be a JSON object')
    options = {action.dest: action for action in parser._actions
               if action.option_strings and action.dest != 'help'}
    for key, value in settings.items():
        action = options.get(key)
        if action is None:
            parser.error(f'{variable}: unknown setting {key!r}')
        if given.intersection(action.option_strings):
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                parser.error(f'{variable}: {key} must be true or false')
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                parser.error(f'{variable}: {key} must be a number or a string')
            try:
                value = action.type(str(value)) if action.type else str(value)
            except (argparse.ArgumentTypeError, ValueError) as error:
                parser.error(f'{variable}: {key}: {error}')
            if action.choices is not None and value not in action.choices:
                parser.error(f'{variable}: {key} must be one of {", ".join(map(str, action.choices))}')
        setattr(args, key, value)
    return args


def nonnegative_seconds(value):
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a non-negative number of seconds') from None
    if seconds < 0:
        raise argparse.ArgumentTypeError('must be a non-negative number of seconds')
    return seconds


def rank_number(value):
    try:
        rank = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a non-negative integer') from None
    if rank < 0:
        raise argparse.ArgumentTypeError('must be a non-negative integer')
    return rank


def positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a positive integer') from None
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def bit_position(value):
    try:
        bit = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a bit position from 0 to 31') from None
    if not 0 <= bit <= 31:
        raise argparse.ArgumentTypeError('must be a bit position from 0 to 31')
    return bit


def add_exercise_arguments(parser):
    """Options of the DataLoader exercise, worker CUDA and the replica check."""
    group = parser.add_argument_group('DataLoader exercise')
    group.add_argument('--dataset-cache', choices=['none', 'rank-thread', 'worker-thread'],
                       default='none',
                       help='which process creates the dataset lock, cache and refill thread '
                            '(default: none, the plain tensor dataset)')
    group.add_argument('--dataloader-timeout', type=nonnegative_seconds, default=0,
                       help='DataLoader timeout in seconds; 0 waits without limit (default: 0)')
    group.add_argument('--stack-dump-seconds', type=nonnegative_seconds, default=0,
                       help='dataloader mode: write every process stack this many seconds after '
                            'the epoch\'s workers exist; device mode: this long after the last '
                            'progress record; 0 is off (default: 0)')
    group.add_argument('--stack-dump-dir', default=STACK_DUMP_DIR,
                       help=f'directory of the stack files (default: {STACK_DUMP_DIR})')
    group.add_argument('--reinit-in-worker', action='store_true',
                       help='recreate the lock and cache in worker_init_fn (fork only)')
    group.add_argument('--worker-cuda', action='store_true',
                       help='DataLoader workers touch CUDA when they read an item')
    group.add_argument('--affected-rank', type=rank_number, default=None,
                       help='only this rank refills its cache or uses worker CUDA '
                            '(default: every rank)')
    group = parser.add_argument_group('replica check (device mode)')
    group.add_argument('--replica-check', action='store_true',
                       help='compare one matrix product across ranks, bit for bit, every step')
    group.add_argument('--replica-n', type=positive_int, default=2048,
                       help='matrix size of the replica check (default: 2048)')
    group.add_argument('--replica-seed', type=int, default=2026,
                       help='seed of the CPU-generated inputs (default: 2026)')
    group.add_argument('--corrupt-rank', type=rank_number, default=None,
                       help='flip one bit of the product on this rank, in software')
    group.add_argument('--corrupt-step', type=positive_int, default=1,
                       help='step index of the flip (default: 1)')
    group.add_argument('--corrupt-bit', type=bit_position, default=22,
                       help='bit of the int32 view of one element (default: 22)')
    group.add_argument('--corrupt-recurring', action='store_true',
                       help='flip on every step from --corrupt-step on')


def check_exercise_arguments(parser, args):
    """Consistency of the exercise options after the settings were applied."""
    if args.corrupt_rank is not None:
        args.replica_check = True
    if args.replica_check and args.mode != 'device':
        parser.error('--replica-check applies to device mode only')
    # These act on the cached dataset; the plain tensor dataset has no lock to
    # replace, no refill and no item read on the GPU, so they would change nothing.
    on_the_cache = [flag for flag, value in (('--worker-cuda', args.worker_cuda),
                                             ('--reinit-in-worker', args.reinit_in_worker),
                                             ('--affected-rank', args.affected_rank is not None))
                    if value]
    if (on_the_cache or args.dataset_cache != 'none') and args.mode != 'dataloader':
        parser.error('--dataset-cache, --worker-cuda, --reinit-in-worker and --affected-rank apply to '
                     'dataloader mode only')
    if on_the_cache and args.dataset_cache == 'none':
        parser.error(f'{", ".join(on_the_cache)}: only with --dataset-cache rank-thread or worker-thread')
    if args.reinit_in_worker and args.start_method != 'fork':
        parser.error('--reinit-in-worker is the fork-only fix')
    if args.stack_dump_seconds and args.mode == 'storage':
        parser.error('--stack-dump-seconds applies to dataloader and device modes')
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['storage', 'dataloader', 'device'])
    parser.add_argument('--duration-seconds', type=int, default=600)
    parser.add_argument('--inject', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-before', type=positive_int, default=None,
                        help='with --resume: start from the newest checkpoint saved before this '
                             'step index (a replica line step=S), not from the last one')
    parser.add_argument('--writer-rank', type=nonnegative_int, default=None,
                        help='storage and device modes: rank that writes the checkpoint (default: 0)')
    parser.add_argument('--checkpoint-every', type=nonnegative_int, default=0,
                        help='device mode: save a checkpoint every N steps (default: 0, none)')
    parser.add_argument('--end-after-save', action='store_true',
                        help='device mode: end 20 steps after the first periodic save past the '
                             'start step, or at --duration-seconds if that comes first')
    parser.add_argument('--lab-dir', default=None,
                        help="write the lab's own lines to one file per process in this directory, "
                             'not to standard output (job.sh)')
    parser.add_argument('--start-method', choices=['fork', 'spawn', 'forkserver'], default='fork')
    parser.add_argument('--epochs', type=positive_epochs, default=1,
                        help='DataLoader epochs (default: 1); numbered controls use 2')
    add_exercise_arguments(parser)
    args = parser.parse_args(argv)
    check_exercise_arguments(parser, apply_incident_settings(parser, args, argv=argv))
    if args.mode == 'dataloader':
        # The DataLoader launchers pass extra arguments through, so a checkpoint
        # option given by mistake must fail rather than run the unchanged control.
        # --lab-dir is used there: job.sh keeps the lifecycle lines off the output.
        given = [flag for flag, value in (('--writer-rank', args.writer_rank is not None),
                                          ('--inject', args.inject), ('--resume', args.resume))
                 if value]
        if given:
            parser.error(f'{", ".join(given)}: not used in dataloader mode')
    if args.resume_before is not None and not args.resume:
        parser.error('--resume-before applies with --resume')
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
    if args.stack_dump_seconds and os.environ.get('LOCAL_RANK') == '0':
        # No process arms its timer before the first collective, which needs this one.
        retire_stack_files(args.stack_dump_dir)
    tensor = torch.ones(32 * 1024 * 1024 if args.mode == 'device' else 1024 * 1024, device=DEVICE)
    collective(tensor)
    if dist.get_rank() == 0:
        build_nccl = '.'.join(str(part) for part in torch.cuda.nccl.version())
        emit(f'torch={torch.__version__}; CUDA={torch.version.cuda}; '
             f'torch_build_nccl={build_nccl}; world_size={dist.get_world_size()} ranks. '
             'Use NCCL startup logs for the loaded runtime version.')
    if args.mode == 'storage':
        storage(args, tensor)
    elif args.mode == 'dataloader':
        dataloader(args, tensor)
    else:
        run_steps(args, tensor)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
