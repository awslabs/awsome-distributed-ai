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


def training_goodput(initial_tokens, retained_tokens, elapsed_seconds,
                     allocated_gpu_seconds, *, scope):
    """Use final durable progress, never sum executed/replayed update tokens."""
    if (initial_tokens < 0 or retained_tokens < initial_tokens
            or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0
            or not math.isfinite(allocated_gpu_seconds) or allocated_gpu_seconds < 0
            or not scope):
        raise ValueError('invalid retained progress, interval or allocation scope')
    useful = retained_tokens - initial_tokens
    return {
        'metric': 'Training goodput',
        'retained_progress_tokens': useful,
        'training_goodput_tokens_per_second': useful / elapsed_seconds,
        'elapsed_seconds': elapsed_seconds,
        'allocated_gpu_hours': allocated_gpu_seconds / 3600,
        'tokens_per_allocated_gpu_hour': (
            useful * 3600 / allocated_gpu_seconds if allocated_gpu_seconds else None),
        'scope': scope,
    }


def paired_training_report(before, after):
    """Report a pair only when the same work and allocation were measured."""
    for key in ('workload', 'allocation', 'measurement_scope', 'measured_useful_tokens'):
        if before[key] != after[key]:
            raise ValueError(f'paired training mismatch: {key}')
    if before['configuration'].get('deterministic', False) != after['configuration'].get('deterministic', False):
        raise ValueError('paired training changed determinism policy')
    for row in (before, after):
        if row['status'] != 'completed' or row['profiled'] or row.get('numerical_verification'):
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


def observability_values(result):
    """Publish completed measurements with distinct campaign/segment names."""
    if result.get('evidence_kind') not in ('LLM hardware run', 'LLM serving hardware run'):
        raise ValueError('completed hardware provenance is required; CPU fixtures are not dashboard evidence')
    if result.get('status') != 'completed' or result.get('profiled') or result.get('numerical_verification'):
        raise ValueError('requires a completed, unprofiled hardware measurement')
    if result.get('metric') == 'Training goodput':
        if result.get('status') != 'completed' or result.get('source') != 'fresh controlled interruption and resume':
            raise ValueError('requires a completed fresh campaign')
        return {
            'llm_training_goodput_campaign_tokens_per_second': result['training_goodput_tokens_per_second'],
            'llm_campaign_retained_tokens': result['retained_progress_tokens'],
            'llm_campaign_elapsed_seconds': result['elapsed_seconds'],
            'llm_campaign_tokens_per_allocated_gpu_hour': result['tokens_per_allocated_gpu_hour'],
            'llm_campaign_allocated_gpu_hours': result['allocated_gpu_hours'],
        }
    if result.get('metrics', {}).get('metric') == 'Serving goodput':
        metrics = result['metrics']
        return {f'llm_serving_{key}': metrics[key] for key in (
            'offered_requests', 'qualified_requests',
            'failed_requests', 'timed_out_requests', 'quality_valid_requests',
            'one_token_tpot_undefined_requests', 'measurement_seconds')} | {
                'llm_serving_goodput_requests_per_second': metrics['serving_goodput_requests_per_second']}
    if (result.get('status') != 'completed' or result.get('profiled')
            or result.get('numerical_verification')):
        raise ValueError('requires completed unprofiled training measurement')
    values = {
        'llm_training_throughput_tokens_per_second': result['training_throughput_tokens_per_second'],
        'llm_training_measured_tokens': result['measured_useful_tokens'],
        'llm_training_measured_seconds': result['measured_seconds'],
    }
    if 'training_goodput' in result:
        values['llm_training_goodput_segment_tokens_per_second'] = result['training_goodput']['training_goodput_tokens_per_second']
    for key in ('measurement_started_utc_seconds', 'measurement_finished_utc_seconds'):
        if key in result:
            values['llm_training_' + key] = result[key]
    return values
