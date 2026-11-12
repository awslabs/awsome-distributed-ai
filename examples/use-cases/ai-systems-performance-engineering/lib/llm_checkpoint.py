"""PyTorch DCP save/load plus rank-local recipe state and completion markers."""
import json
import os
from pathlib import Path
import random
import time
import gc

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.profiler import record_function
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

_OPEN_CHECKPOINTERS = []


def close_checkpointers():
    errors = []
    while _OPEN_CHECKPOINTERS:
        try:
            _OPEN_CHECKPOINTERS.pop().close()
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError('checkpoint cleanup failed: ' + '; '.join(str(e) for e in errors)) from errors[0]


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def _durable_file(path, write):
    """Publish a file only after its contents and directory entry are flushed."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def latest_completed(root):
    # Rolling archival leaves lightweight completion records but relocates tensors.
    # A source campaign with an archive index is not a local restore candidate.
    if (Path(root).parent / 'S3_ARCHIVE.json').is_file() or (Path(root).parent.parent / 'S3_ARCHIVE.json').is_file():
        return None
    completed = []
    for marker in Path(root).glob('update-*/COMPLETED.json'):
        row = json.loads(marker.read_text())
        if (marker.parent / '.metadata').is_file():
            completed.append((row['completed_updates'], marker.parent))
    return max(completed)[1] if completed else None


class Checkpoints:
    """One outstanding DCP write; both modes save exactly the same state."""

    def __init__(self, root, model, optimizer, scheduler, *, asynchronous=False, group=None,
                 writer_threads=1, copy_ahead_bytes=10_000_000, process_async=False):
        if writer_threads < 1:
            raise ValueError('checkpoint writer threads must be positive')
        if copy_ahead_bytes < 1:
            raise ValueError('checkpoint copy-ahead bytes must be positive')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.asynchronous = asynchronous
        self.writer_threads = writer_threads
        self.copy_ahead_bytes = copy_ahead_bytes
        self.group = group
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.pending = None
        self.durable_progress = None
        self.process_async = process_async
        self.staging_future = None
        self.snapshot_sources = None
        self.stager = None
        self.planner = None
        self.save_count = 0
        self.source_storage_refs = {}
        if process_async:
            if not asynchronous:
                raise ValueError('process checkpointing requires asynchronous mode')
            from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
            from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
            self.stager = DefaultStager(StagingOptions(
                use_async_staging=True, use_pinned_memory=True,
                use_shared_memory=True, use_non_blocking_copy=True))
            self.planner = DefaultSavePlanner(enable_plan_caching=True)
            _OPEN_CHECKPOINTERS.append(self)

    def event(self, phase, **fields):
        memory = {}
        if phase in ('save_start', 'staging_fence', 'globally_complete'):
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith(('VmRSS:', 'VmLck:')):
                    key, value, _ = line.split()
                    memory[key.rstrip(':') + '_bytes'] = int(value) * 1024
        with (self.root.parent / f'checkpoint-events-rank-{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(phase=phase, monotonic_seconds=time.perf_counter(),
                                         save_count=self.save_count, **memory, **fields)) + '\n')

    def before_source_reuse(self):
        """Fence before next FSDP forward/zero_grad, not merely optimizer.step."""
        if self.staging_future is not None:
            started = time.perf_counter()
            self.staging_future.result()
            # v2.9.1 async_save's already-done callback can mask a staging
            # exception. The stager's public synchronization also checks it.
            self.stager.synchronize_staging()
            wait_seconds = time.perf_counter() - started
            storages = {}
            def collect(value):
                if isinstance(value, torch.Tensor):
                    local = value.to_local() if hasattr(value, 'to_local') else value
                    storage = local.untyped_storage()
                    storages[storage.data_ptr()] = storage
                elif isinstance(value, dict):
                    for item in value.values():
                        collect(item)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        collect(item)
            collect(self.snapshot_sources)
            self.event('staging_fence', wait_seconds=wait_seconds,
                       snapshot_source_storage_bytes=sum(s.nbytes() for s in storages.values()))
            self.staging_future = None
            self.snapshot_sources = None

    def close(self):
        # No marker/barrier on exception cleanup; never credit a failed save.
        errors = []
        try:
            try:
                self.before_source_reuse()
            except Exception as error:
                errors.append(error)
            # Even a staging failure must not skip waiting for the upload consumer.
            if self.pending is not None and self.pending[2] is not None:
                try:
                    self.pending[2].result()
                except Exception as error:
                    errors.append(error)
            if self.stager is not None:
                try:
                    self.stager.close()
                except Exception as error:
                    errors.append(error)
        finally:
            self.pending = None
            self.staging_future = None
            self.stager = None
            self.snapshot_sources = None
            # Native 2.9.1 stager dispatch closures form cycles; public close()
            # only joins its executor. Collect after dropping application refs.
            self.source_storage_refs.clear()
            gc.collect()
        if errors:
            raise RuntimeError('checkpoint cleanup failed: ' + '; '.join(str(e) for e in errors)) from errors[0]

    def save(self, progress):
        self.finish()
        self.save_count += 1
        self.event('save_start', update=progress['completed_updates'])
        path = self.root / f"update-{progress['completed_updates']:06d}"
        if self.rank == 0:
            path.mkdir(exist_ok=False)
        _barrier()
        extra = {
            'progress': progress, 'world_size': self.world,
            'scheduler': self.scheduler.state_dict(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        with record_function('checkpoint/sidecar_flush'):
            _durable_file(path / f'rank-{self.rank}.pt', lambda stream: torch.save(extra, stream))
        with record_function('checkpoint/state_dict'):
            model, optimizer = get_state_dict(self.model, self.optimizer)
        state = {'model': model, 'optimizer': optimizer}
        if self.process_async:
            # Keep native weak-cache keys alive across saves. Only source storage
            # wrappers are retained; copying still fences before source reuse.
            def retain(value):
                if isinstance(value, torch.Tensor):
                    value = value.to_local() if hasattr(value, 'to_local') else value
                    storage = value.untyped_storage()
                    self.source_storage_refs[id(storage)] = storage
                elif isinstance(value, dict):
                    for item in value.values():
                        retain(item)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        retain(item)
            retain(state)
            self.event('retained_source_storages', count=len(self.source_storage_refs),
                       bytes=sum(s.nbytes() for s in self.source_storage_refs.values()))
        writer = dcp.FileSystemWriter(path, sync_files=True, thread_count=self.writer_threads,
                                      # PROCESS receives CPU-staged tensors; the GPU copy
                                      # loader otherwise initializes a needless child CUDA
                                      # context on default GPU0 (native 2.9.1).
                                      per_thread_copy_ahead=0 if self.process_async else self.copy_ahead_bytes)
        if self.process_async:
            from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
            # The staging stream does not wait on the producer CUDA stream.
            torch.cuda.synchronize()
            self.snapshot_sources = state
            response = dcp.async_save(
                state, storage_writer=writer, process_group=self.group,
                planner=self.planner, async_stager=self.stager,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS)
            self.staging_future = response.staging_completion
            future = response.upload_completion
            count = self.save_count
            response.staging_completion.add_done_callback(
                lambda f: self.event('staging_complete', checkpoint_sequence=count,
                                     failed=f.exception() is not None))
            future.add_done_callback(
                lambda f: self.event('upload_complete', checkpoint_sequence=count,
                                     failed=f.exception() is not None))
        elif self.asynchronous:
            # Default FileSystemWriter stages synchronously to an independent CPU
            # snapshot before returning; only the disk write overlaps updates.
            with record_function('checkpoint/dcp_async_stage_and_submit'):
                future = dcp.async_save(state, storage_writer=writer, process_group=self.group)
            self.event('staging_complete', checkpoint_sequence=self.save_count, failed=False)
            count = self.save_count
            future.add_done_callback(
                lambda f: self.event('upload_complete', checkpoint_sequence=count,
                                     failed=f.exception() is not None))
        else:
            with record_function('checkpoint/dcp_sync_write'):
                dcp.save(state, storage_writer=writer, process_group=self.group)
            future = None
            self.event('upload_complete', checkpoint_sequence=self.save_count, failed=False)
        self.pending = (path, dict(progress), future)
        self.event('save_return', update=progress['completed_updates'])
        if not self.asynchronous:
            self.finish()

    def finish(self):
        if self.pending is None:
            return
        path, progress, future = self.pending
        self.before_source_reuse()
        if future is not None:
            started = time.perf_counter()
            future.result()
            self.event('pending_upload_wait', wait_seconds=time.perf_counter() - started)
        _barrier()
        if self.rank == 0:
            _durable_file(path / 'COMPLETED.json',
                          lambda stream: stream.write(json.dumps(progress).encode()))
        _barrier()
        self.durable_progress = progress
        self.pending = None
        self.event('globally_complete', update=progress['completed_updates'])

    def poll(self):
        """Publish completed async saves at update boundaries.

        Readiness uses the training group, separate from DCP's background group.
        """
        if self.pending is None or self.pending[2] is None:
            return
        ready = self.pending[2].done()
        if dist.is_initialized():
            device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else 'cpu'
            flag = torch.tensor(int(ready), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            ready = bool(flag.item())
        if ready:
            self.finish()

    def restore(self, path, *, expected_workload):
        path = Path(path)
        for parent in (path.parent.parent, path.parent.parent.parent):
            index = parent / 'S3_ARCHIVE.json'
            if index.is_file():
                raise ValueError(f'checkpoint tensors were archived; restore the verified archive to a new directory using {index}')
        progress = json.loads((path / 'COMPLETED.json').read_text())
        if progress['workload'] != expected_workload:
            raise ValueError('resume workload does not match the completed checkpoint')
        extra = torch.load(path / f'rank-{self.rank}.pt', map_location='cpu', weights_only=False)
        if extra['world_size'] != self.world or extra['progress'] != progress:
            raise ValueError('rank state or data-parallel world size mismatch')
        with record_function('checkpoint/restore_state_dict'):
            model, optimizer = get_state_dict(self.model, self.optimizer)
        state = {'model': model, 'optimizer': optimizer}
        with record_function('checkpoint/read'):
            dcp.load(state, checkpoint_id=path, process_group=self.group)
        with record_function('checkpoint/apply'):
            set_state_dict(self.model, self.optimizer, model_state_dict=state['model'],
                           optim_state_dict=state['optimizer'])
        self.scheduler.load_state_dict(extra['scheduler'])
        random.setstate(extra['python_rng'])
        np.random.set_state(extra['numpy_rng'])
        torch.set_rng_state(extra['torch_rng'])
        if extra['cuda_rng'] is not None:
            torch.cuda.set_rng_state(extra['cuda_rng'])
        self.durable_progress = progress
        _barrier()
        return progress
