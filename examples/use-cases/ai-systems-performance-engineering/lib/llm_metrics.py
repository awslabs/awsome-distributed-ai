"""Continuous-window and retained-progress accounting, independent of CUDA."""
import math
import time


class ContinuousWindow:
    """Synchronize only at boundaries; include all work between them."""

    def __init__(self, barrier, synchronize, clock=time.perf_counter):
        self.barrier = barrier
        self.synchronize = synchronize
        self.clock = clock
        self.started = None

    def start(self):
        if self.started is not None:
            raise RuntimeError('window already started')
        self.synchronize()
        self.barrier()
        self.started = self.clock()
        self.started_utc_seconds = time.time()

    def stop(self):
        if self.started is None:
            raise RuntimeError('window has not started')
        self.synchronize()
        # Include the terminal boundary and rank skew, then aggregate outside.
        self.barrier()
        seconds = self.clock() - self.started
        self.finished_utc_seconds = time.time()
        self.started = None
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('window duration must be finite and positive')
        return seconds


def accumulation_steps(global_batch, microbatch, data_parallel_size):
    if min(global_batch, microbatch, data_parallel_size) < 1:
        raise ValueError('batch sizes and data-parallel size must be positive')
    divisor = microbatch * data_parallel_size
    if global_batch % divisor:
        raise ValueError('global batch must be divisible by microbatch times DP size')
    return global_batch // divisor


def retained_progress(initial_tokens, retained_tokens, elapsed_seconds, *, scope,
                      initial_updates, durable_updates):
    """Use final durable progress, never sum executed/replayed update tokens.

    initial_updates and durable_updates are the updates completed at the start of the interval
    and in its last durable checkpoint: the retained tokens are those of the updates between them.
    """
    if (initial_tokens < 0 or retained_tokens < initial_tokens
            or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0
            or initial_updates < 0 or durable_updates < initial_updates
            or not scope):
        raise ValueError('invalid retained progress, update range, interval or scope')
    useful = retained_tokens - initial_tokens
    return {
        'metric': 'Retained progress',
        'retained_progress_tokens': useful,
        'retained_tokens_per_second': useful / elapsed_seconds,
        'elapsed_seconds': elapsed_seconds,
        'initial_completed_updates': initial_updates,
        'durable_completed_updates': durable_updates,
        'scope': scope,
    }


# The workload entries that fix the work of one update, beside its mean target tokens.
UPDATE_WORK = ('model_id', 'sequence_length', 'global_batch_samples', 'data_parallel_size')


def update_work(workload, tokens, updates):
    """The work of one update: the entries of UPDATE_WORK and the mean targets of `updates` updates."""
    if updates < 1 or tokens < 0:
        raise ValueError('per-update work needs at least one update and nonnegative tokens')
    return {key: workload.get(key) for key in UPDATE_WORK} | {'targets_per_update': tokens / updates}


def training_goodput(retained_tokens_per_second, work, reference):
    """Training goodput: the retained progress rate over the fault-free rate.

    work is update_work() of the run. The fault-free rate is the throughput of the timed updates
    of reference, the summary.json of a completed, unprofiled run without saves that does the
    same work per update. The run's interval also holds the data loader's start and the warmup
    updates, which the fault-free rate leaves out, so even a run without saves stays below 1.
    """
    if not reference:
        raise ValueError('no fault-free run')
    for failed, reason in ((reference.get('status') != 'completed', 'is not completed'),
                           (reference.get('profiled'), 'was profiled'),
                           (reference.get('measurement_scope') != 'continuous post-warmup training', 'saves')):
        if failed:
            raise ValueError(f'the fault-free run {reason}')
    workload = reference['workload']
    expected = update_work(workload, reference['measured_useful_tokens'],
                           workload['updates'] - workload['warmup_updates'])
    differs = [key for key in expected if work.get(key) != expected[key]]
    if differs:
        raise ValueError('different work per update: ' + ', '.join(differs))
    fault_free = reference['training_throughput_tokens_per_second']
    if (not math.isfinite(fault_free) or fault_free <= 0
            or not math.isfinite(retained_tokens_per_second) or retained_tokens_per_second < 0):
        raise ValueError('the rates must be finite and the fault-free rate positive')
    return {
        'metric': 'Training goodput',
        'training_goodput': retained_tokens_per_second / fault_free,
        'retained_tokens_per_second': retained_tokens_per_second,
        'fault_free_tokens_per_second': fault_free,
    }


def paired_training_report(before, after):
    """Report a pair only when the same work and allocation were measured."""
    for key in ('workload', 'allocation', 'measurement_scope', 'measured_useful_tokens'):
        if before[key] != after[key]:
            raise ValueError(f'paired training mismatch: {key}')
    if before['configuration'].get('deterministic', False) != after['configuration'].get('deterministic', False):
        raise ValueError('paired training changed determinism policy')
    for row in (before, after):
        if row['status'] != 'completed' or row['profiled']:
            raise ValueError('pair requires completed, unprofiled runs')
        if row['measured_seconds'] <= 0 or not math.isfinite(row['measured_seconds']):
            raise ValueError('invalid continuous measurement duration')
    return {
        'metric': 'Training throughput',
        'before_configuration': before['configuration'],
        'after_configuration': after['configuration'],
        'workload': before['workload'],
        'allocation': before['allocation'],
        'useful_tokens': before['measured_useful_tokens'],
        'before_seconds': before['measured_seconds'],
        'after_seconds': after['measured_seconds'],
        'speedup_ratio': before['measured_seconds'] / after['measured_seconds'],
        'qualification': 'one observed pair; numerical and repeated-run review required',
    }
