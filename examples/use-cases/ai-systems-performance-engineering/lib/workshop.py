#!/usr/bin/env python3
"""Shared code of the AIM347 workshop tools in tools/.

Standard library only: it runs on the login node, on the GPU nodes outside the
training image, and in the CPU tests. The shell tools stay thin and call the
subcommands at the end of this file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import ssl
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import charts
from llm_metrics import paired_training_report

LAB = Path(__file__).resolve().parents[1]
STATE = Path.home() / '.aim347'

# SmolLM2-1.7B (configs/smol-*.json): parameters, decoder layers, hidden size.
MODEL_PARAMETERS = 1_711_376_384
MODEL_LAYERS = 24
MODEL_HIDDEN = 2048

# Prepared datasets, in preparation order. The first preparation also
# downloads the model. A fingerprint is the manifest sha256 over tokens.bin
# and lengths.bin. tokens-2048-1280 is pinned after its first preparation.
DATASETS = {
    'tokens-2048-1280': dict(config='configs/smol-2048-40.json', records=1280),
    'tokens-2048-160': dict(config='configs/smol-2048.json', records=160),
    'tokens-1024-160': dict(config='configs/smol-1024.json', records=160),
}
EXPECTED_FINGERPRINTS = {
    'tokens-2048-1280': None,
    'tokens-2048-160': 'a8d0d10c6fa35c3abf087e4c97abde04ab0014b03c88b9f4301a26ea590f7736',
    'tokens-1024-160': 'e15aa0d7a1fce39159494ab0010d92c8f537c12a56f78810cc64d9a019a3c520',
}
MINIMUM_DOCUMENT_TOKENS = 4096
PROVENANCE_KEYS = ('model_id', 'tokenizer_revision', 'dataset_id', 'dataset_revision',
                   'dataset_subset', 'dataset_split')
EFA_DEVICES = {'g7.48xlarge': 2}
FSX_FREE_BYTES = 300 * 2**30

JOB_SETTINGS = {
    'LLM_TRANSPORT': ('socket', 'efa'),
    'ATTENTION': ('eager', 'sdpa'),
    'CHECKPOINT_MODE': ('none', 'sync', 'process'),
}
EFA_COUNTERS = ('tx_bytes', 'rx_bytes', 'rdma_read_bytes', 'rdma_write_bytes', 'rdma_write_recv_bytes')
NETWORK = {'socket': 'Socket', 'efa': 'AWS Libfabric'}
RESTORE_FROM, RESTORE_STOP, INTERRUPT_AFTER = 20, 25, 30

# Subcommand: required job.env values (None accepts any), kind and config.
SUBCOMMANDS = {
    'baseline': dict(requires=('socket', 'eager', 'none'), kind='timed', config='configs/smol-2048.json'),
    'probe': dict(requires=(None, None, 'none'), kind='timed', config='configs/smol-1024.json'),
    'profile': dict(requires=(None, None, 'none'), kind='profile', config='configs/smol-2048.json'),
    'sdpa-socket': dict(requires=('socket', 'eager', 'none'), kind='timed', config='configs/smol-2048.json'),
    'efa': dict(requires=('efa', 'eager', 'none'), kind='timed', config='configs/smol-2048.json'),
    'sdpa': dict(requires=('efa', 'sdpa', 'none'), kind='timed', config='configs/smol-2048.json'),
    'checkpoint': dict(requires=('efa', 'sdpa', ('sync', 'process')), kind='timed',
                       config='configs/smol-2048-40.json'),
    'restore': dict(requires=('efa', 'sdpa', None), kind='restore', config='configs/smol-2048-40.json'),
    'interrupt': dict(requires=('efa', 'sdpa', None), kind='interrupt', config='configs/smol-2048-40.json'),
}
MAIN_PATH = ('baseline', 'efa', 'sdpa')
PROGRESSION_ORDER = ('baseline', 'probe-socket-eager', 'sdpa-socket', 'efa', 'probe-efa-eager',
                     'sdpa', 'ckpt-sync', 'ckpt-process', 'interrupt')
PROFILE_ORDER = ('profile-socket-eager', 'profile-efa-eager', 'profile-efa-sdpa')


class UsageError(ValueError):
    """Exit 2: usage, job.env, or the state a subcommand requires."""


# ---------------------------------------------------------------- job.env

def parse_job_env(text, name='job.env'):
    """Parse, never source: three known keys, each once, with allowed values."""
    values = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.partition('=')
        if not separator or key not in JOB_SETTINGS:
            raise UsageError(f'{name} line {number}: {raw!r} is not one of '
                             + ', '.join(f'{k}=...' for k in JOB_SETTINGS))
        if key in values:
            raise UsageError(f'{name} line {number}: {key} is set a second time')
        if value not in JOB_SETTINGS[key]:
            raise UsageError(f'{name} line {number}: {key} must be one of '
                             + ', '.join(JOB_SETTINGS[key]) + f' (it says {value!r})')
        values[key] = value
    missing = [key for key in JOB_SETTINGS if key not in values]
    if missing:
        raise UsageError(f'{name}: missing ' + ', '.join(missing) + '. Start over with: git checkout job.env')
    return values


def job_env_path():
    return Path(os.environ.get('AIM347_JOB_ENV') or Path(os.environ.get('AIM347_COMPANION', LAB)) / 'job.env')


def read_job_env(path=None):
    path = Path(path or job_env_path())
    try:
        return parse_job_env(path.read_text(), path.name)
    except OSError as error:
        raise UsageError(f'cannot read {path}: {error}') from error


# ---------------------------------------------------------------- stage plans

def config_values(config, lab=LAB):
    return json.loads((Path(lab) / config).read_text())


def dataset_for(config):
    for name, dataset in DATASETS.items():
        if dataset['config'] == config:
            return name
    raise KeyError(config)


def requirement_errors(subcommand, job):
    errors = []
    for key, allowed in zip(JOB_SETTINGS, SUBCOMMANDS[subcommand]['requires']):
        if allowed is None:
            continue
        allowed = (allowed,) if isinstance(allowed, str) else allowed
        if job[key] not in allowed:
            wanted = allowed[0]
            errors.append(f'{subcommand} expects {key}={" or ".join(allowed)} in job.env '
                          f'(it says {job[key]}). Change it with: '
                          f"sed -i 's/^{key}={job[key]}$/{key}={wanted}/' job.env")
    return errors


def plan(subcommand, job, lab=LAB):
    """The stages one subcommand runs, as data. Raises UsageError on a wrong job.env."""
    if subcommand not in SUBCOMMANDS:
        raise UsageError(f'unknown subcommand {subcommand!r}; choose one of '
                         + ', '.join(list(SUBCOMMANDS) + ['attach']))
    errors = requirement_errors(subcommand, job)
    if errors:
        raise UsageError('\n'.join(errors))
    spec = SUBCOMMANDS[subcommand]
    transport, attention, saving = job['LLM_TRANSPORT'], job['ATTENTION'], job['CHECKPOINT_MODE']
    if subcommand == 'probe':
        runs = [(f'probe-{transport}-{attention}', transport, attention, 'none')]
    elif subcommand == 'profile':
        runs = [(f'profile-{transport}-{attention}', transport, attention, 'none')]
    elif subcommand == 'sdpa-socket':
        runs = [('sdpa-socket', 'socket', 'sdpa', 'none')]
    elif subcommand == 'checkpoint':
        runs = [('ckpt-sync', transport, attention, 'sync'), ('ckpt-process', transport, attention, 'process')]
    elif subcommand == 'restore':
        runs = [('restore', transport, attention, 'sync')]
    elif subcommand == 'interrupt':
        runs = [('interrupt', transport, attention, 'process')]
    else:
        runs = [(subcommand, transport, attention, saving)]
    values = config_values(spec['config'], lab)
    return [dict(name=name, subcommand=subcommand, kind=spec['kind'], transport=t, attention=a, saving=s,
                 config=spec['config'], data=dataset_for(spec['config']), sequence_length=values['sequence_length'],
                 updates=values['updates'], warmup_updates=values['warmup_updates'],
                 global_batch=values['global_batch_samples'], checkpoint_updates=values['checkpoint_updates'],
                 job_env=dict(job))
            for name, t, a, s in runs]


def trainer_arguments(stage, label, *, output=True):
    arguments = ['--config', stage['config'], '--data', f"/data/{stage['data']}", '--model-path', '/data/model',
                 '--attention-implementation', stage['attention'], '--microbatch', '1', '--workers', '2',
                 '--checkpoint-mode', stage['saving'], '--allocation-label', label]
    if stage['kind'] == 'profile':
        arguments.append('--profile')
    if stage['kind'] == 'restore':
        arguments += ['--resume', f'/results/ckpt-sync/train/checkpoints/update-{RESTORE_FROM:06d}',
                      '--interrupt-after-update', str(RESTORE_STOP)]
    if output:
        arguments += ['--output', f"/results/{stage['name']}/train"]
    return arguments


def launch_command(stage, label, results, gpus):
    if stage['kind'] == 'interrupt':
        return (['python3', '13.measure-recovery.py', '--output', str(Path(results) / stage['name'] / 'campaign'),
                 '--allocated-gpus', str(gpus), '--interrupt-after-update', str(INTERRUPT_AFTER), '--', 'bash',
                 '2.run-llm.sh'] + trainer_arguments(stage, label, output=False))
    return ['bash', '2.run-llm.sh'] + trainer_arguments(stage, label)


def timed_targets(stage):
    return ((stage['updates'] - stage['warmup_updates']) * stage['global_batch']
            * (stage['sequence_length'] - 1))


# ---------------------------------------------------------------- metrics

def flop_per_token(sequence_length):
    """6N + 12 L d T: all parameters, plus attention scores and weighted values."""
    return 6 * MODEL_PARAMETERS + 12 * MODEL_LAYERS * MODEL_HIDDEN * sequence_length


def gpu_seconds_per_million_tokens(seconds, tokens, gpus):
    return seconds * gpus / tokens * 1e6


def mfu(tokens_per_second, sequence_length, gemm_tflops_per_gpu, gpus):
    return tokens_per_second * flop_per_token(sequence_length) / (gpus * gemm_tflops_per_gpu * 1e12)


def compute_only_seconds(targets, sequence_length, gemm_tflops_per_gpu, gpus):
    """The update's arithmetic alone at the measured GEMM peak."""
    return targets * flop_per_token(sequence_length) / (gpus * gemm_tflops_per_gpu * 1e12)


def peak_memory_gib(summary):
    return max(rank['peak_cuda_allocated_bytes'] for rank in summary['ranks']) / 2**30


def read_ceilings(results):
    path = Path(results) / 'ceilings' / 'ceilings.json'
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def aggregate_ceilings(directory):
    """Median over all GPUs of the GEMM best rate and of the median read bandwidth."""
    directory = Path(directory)
    gpus = []
    for gemm_path in sorted(directory.glob('gemm-*-gpu*.json')):
        match = re.fullmatch(r'gemm-(.+)-gpu(\d+)\.json', gemm_path.name)
        host, index = match.group(1), int(match.group(2))
        gemm = json.loads(gemm_path.read_text())
        memory = json.loads((directory / f'dram-{host}-gpu{index}.json').read_text())
        gpus.append(dict(host=host, gpu=index, gpu_name=gemm['gpu'],
                         best_tflops_per_gpu=gemm['best_tflops_per_gpu'],
                         median_bytes_per_second=memory['median_bytes_per_second']))
    if not gpus:
        raise ValueError(f'no GEMM results in {directory}')
    gpus.sort(key=lambda row: (row['host'], row['gpu']))
    gemm = statistics.median(row['best_tflops_per_gpu'] for row in gpus)
    memory = statistics.median(row['median_bytes_per_second'] for row in gpus)
    return dict(gemm_tflops_per_gpu=gemm, memory_read_bytes_per_second_per_gpu=memory,
                ridge_flop_per_byte=gemm * 1e12 / memory, gpu_count=len(gpus),
                gemm='BF16 inputs, FP32 accumulation, 8192 x 8192 x 8192, best of 30 trials per GPU',
                memory_read='1 GiB streaming read, median of 30 trials per GPU', gpus=gpus)


def ceilings_lines(ceilings):
    hosts = sorted({row['host'] for row in ceilings['gpus']})
    width = max(len(host) for host in hosts)
    count = ceilings['gpu_count']

    def table(key, scale, digits):
        return [f'  {host:<{width}}  ' + ' '.join(f'{row[key] / scale:.{digits}f}' for row in ceilings['gpus']
                                                   if row['host'] == host) for host in hosts]
    return (['GEMM, BF16 inputs with FP32 accumulation, 8192 x 8192 x 8192, best of 30 trials per GPU (TFLOP/s)']
            + table('best_tflops_per_gpu', 1, 1)
            + [f"  Your GEMM peak (median of {count} GPUs): {ceilings['gemm_tflops_per_gpu']:.1f} TFLOP/s per GPU",
               'Memory read, 1 GiB streaming read, median of 30 trials per GPU (GB/s)']
            + table('median_bytes_per_second', 1e9, 0)
            + [f"  Your memory-read ceiling (median of {count} GPUs): "
               f"{ceilings['memory_read_bytes_per_second_per_gpu'] / 1e9:.0f} GB/s per GPU",
               f"Ridge point: {ceilings['ridge_flop_per_byte']:.0f} FLOP per byte"])


def ceilings_summary(ceilings):
    if not ceilings:
        return 'GEMM peak n/a (run bash tools/hold-and-measure.sh)'
    return (f"GEMM peak {ceilings['gemm_tflops_per_gpu']:.1f} TFLOP/s per GPU, memory read "
            f"{ceilings['memory_read_bytes_per_second_per_gpu'] / 1e9:.0f} GB/s per GPU, "
            f"ridge {ceilings['ridge_flop_per_byte']:.0f} FLOP per byte")


# ---------------------------------------------------------------- run output

def read_rows(path):
    """Rows of a rank log; an incomplete last line (still being written) is skipped."""
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            break
    return rows


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


NCCL_NETWORK = re.compile(r'(\S+):(\d+):\d+ \[(\d+)\] NCCL INFO Using network (.+)')
NCCL_PROVIDER = re.compile(r'NET/OFI Selected provider is (\S+), fabric is (\S+) \(found (\d+) nics\)')
NCCL_PROTOCOL = re.compile(r'NET/OFI Using transport protocol (.+)')
NCCL_CHANNEL = re.compile(r'NCCL INFO Channel \S+ : .* via NET/(.+)')
# NCCL_DEBUG_SUBSYS must include ENV for this line (2.run-llm.sh sets it).
NCCL_PROTO_ENV = re.compile(r'(\S+):(\d+):\d+ \[\d+\] NCCL INFO NCCL_PROTO set by environment to (.+)')


def parse_nccl(text):
    networks, nccl_protos = {}, {}
    providers, protocols = [], []
    channels = gdrdma = 0
    for line in text.splitlines():
        match = NCCL_NETWORK.search(line)
        if match:
            networks[(match.group(1), match.group(2))] = match.group(4).strip()
            continue
        match = NCCL_PROVIDER.search(line)
        if match:
            value = f'{match.group(1)}, fabric {match.group(2)} (found {match.group(3)} nics)'
            if value not in providers:
                providers.append(value)
            continue
        match = NCCL_PROTOCOL.search(line)
        if match:
            value = match.group(1).strip()
            if value not in protocols:
                protocols.append(value)
            continue
        match = NCCL_CHANNEL.search(line)
        if match:
            channels += 1
            gdrdma += '/GDRDMA' in match.group(1)
            continue
        match = NCCL_PROTO_ENV.search(line)
        if match:
            nccl_protos[(match.group(1), match.group(2))] = match.group(3).strip()
    return dict(networks=count_values(networks), ranks=len(networks), providers=providers, protocols=protocols,
                network_channels=channels, gdrdma_channels=gdrdma, nccl_protos=count_values(nccl_protos))


def count_values(by_process):
    """{value: number of processes} from {(host, pid): value}."""
    counts = {}
    for value in by_process.values():
        counts[value] = counts.get(value, 0) + 1
    return counts


def nccl_problem(nccl, transport, world):
    expected = NETWORK[transport]
    if nccl['ranks'] < world or set(nccl['networks']) != {expected}:
        return f'expected {expected} on {world} ranks, found {network_text(nccl, world) or "no network lines"}'
    return None


def network_text(nccl, world):
    return '; '.join(f'{network} on {count} of {world} ranks' for network, count in nccl['networks'].items())


def nccl_proto_text(nccl, world):
    return ('; '.join(f'{value} on {count} of {world} ranks' for value, count in nccl['nccl_protos'].items())
            or 'not set on any rank')


def read_efa_counters(root='/sys/class/infiniband'):
    """Counters of every EFA device (driver efa); missing counter files are skipped."""
    devices, errors = {}, []
    root = Path(root)
    for device in sorted(root.iterdir()) if root.is_dir() else []:
        if os.path.basename(os.path.realpath(device / 'device' / 'driver')) != 'efa':
            continue
        values = {}
        for counter in EFA_COUNTERS:
            path = device / 'ports' / '1' / 'hw_counters' / counter
            if not path.exists():
                continue
            try:
                values[counter] = int(path.read_text().strip())
            except (OSError, ValueError) as error:
                errors.append(f'{path}: {error}')
        devices[device.name] = values
    return devices, errors


def merge_efa_readings(lines):
    """srun output, one JSON line per node, into {host: {device: {counter: value}}}.

    A node with an unreadable counter, or with no counter at all, maps to {}.
    """
    nodes = {}
    for line in lines:
        try:
            reading = json.loads(line)
        except ValueError:
            continue
        devices = reading.get('devices', {})
        readable = not reading.get('errors') and any(devices.values())
        nodes[reading['host']] = devices if readable else {}
    return nodes


def efa_deltas(before, after):
    """{host: None (unavailable) or dict(sent, received, increased)}."""
    deltas = {}
    for host in sorted(set(before) | set(after)):
        first, last = before.get(host) or {}, after.get(host) or {}
        changes = {(device, counter): last[device][counter] - value
                   for device, counters in first.items() for counter, value in counters.items()
                   if counter in last.get(device, {})}
        if not changes:
            deltas[host] = None
            continue
        deltas[host] = dict(
            sent=sum(v for (_, counter), v in changes.items() if counter == 'tx_bytes'),
            received=sum(v for (_, counter), v in changes.items() if counter == 'rx_bytes'),
            increased=any(v > 0 for v in changes.values()))
    return deltas


def format_bytes(value):
    for unit, size in (('TB', 1e12), ('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if abs(value) >= size:
            return f'{value / size:.1f} {unit}'
    return f'{value} B'


def efa_text(deltas):
    if not deltas:
        return 'EFA counters unavailable on both nodes'
    return '; '.join(f'EFA counters unavailable on {host}' if delta is None else
                     f"{host} +{format_bytes(delta['sent'])} sent, +{format_bytes(delta['received'])} received"
                     for host, delta in deltas.items())


def segment_logs(stage, stage_dir):
    if stage['kind'] == 'interrupt':
        return [stage_dir / 'campaign' / 'interrupted.log', stage_dir / 'campaign' / 'resumed.log']
    return [stage_dir / 'launch.log']


def rank_logs(directory):
    return sorted(Path(directory).glob('rank-*-updates.jsonl'))


def post_run_checks(stage, stage_dir, launcher_exit, world):
    """Decide whether the run succeeded (exit 4 if not), then check it (exit 5 on a failed check).

    Returns (exit code, list of check results, failure reason or None).
    """
    stage_dir = Path(stage_dir)
    train = stage_dir / 'train'
    if stage['kind'] == 'restore':
        # Without drain every rank exits 75 after the stop update, so the
        # launcher always reports failure; decide from the rows instead.
        logs = rank_logs(train)
        last = [read_rows(path)[-1:] for path in logs]
        if len(logs) != world or not all(rows and rows[0]['update'] == RESTORE_STOP
                                         and rows[0].get('controlled_interruption') is True for rows in last):
            return 4, [], (f'not every one of {world} ranks stopped after update {RESTORE_STOP} '
                           f'(launcher exit status {launcher_exit})')
    elif stage['kind'] == 'interrupt':
        if launcher_exit != 0 or not (stage_dir / 'campaign' / 'campaign.json').is_file():
            return 4, [], f'the recovery controller exited with status {launcher_exit}'
    elif launcher_exit != 0:
        return 4, [], f'the launcher exited with status {launcher_exit}'
    checks = []
    for log in segment_logs(stage, stage_dir):
        try:
            nccl = parse_nccl(log.read_text(errors='replace'))
        except OSError:
            nccl = parse_nccl('')
        problem = nccl_problem(nccl, stage['transport'], world)
        checks.append(dict(name='NCCL network', ok=problem is None, detail=problem or log.name))
    deltas = efa_deltas(read_json(stage_dir / 'efa-before.json', {}), read_json(stage_dir / 'efa-after.json', {}))
    if stage['transport'] == 'efa':
        flat = [host for host, delta in deltas.items() if delta is not None and not delta['increased']]
        checks.append(dict(name='EFA counters', ok=not flat,
                           detail=('no EFA counter increased on ' + ', '.join(flat)) if flat else efa_text(deltas)))
    if stage['kind'] == 'timed':
        summary = read_json(train / 'summary.json')
        ok = bool(summary) and summary.get('status') == 'completed' and not summary.get('profiled')
        checks.append(dict(name='Summary', ok=ok, detail='summary.json' if ok else 'no completed, unprofiled summary.json'))
    if stage['kind'] == 'restore':
        rows = read_rows(train / 'rank-0-updates.jsonl')
        ok = bool(rows) and rows[0]['update'] == RESTORE_FROM + 1
        checks.append(dict(name='Restore start', ok=ok,
                           detail=f"rank 0 starts at update {rows[0]['update'] if rows else 'none'}"))
    return (0 if all(check['ok'] for check in checks) else 5), checks, None


# ---------------------------------------------------------------- display

class RowPrinter:
    """Rows of rank 0 as they appear; seconds are the gaps between consecutive rows."""

    def __init__(self, warmup_updates, save_updates=None):
        self.warmup = warmup_updates
        self.saves = save_updates
        self.previous = None

    def header(self):
        text = f"{'update':>6}{'seconds':>10}{'loss':>10}"
        return text + (f"{'save':>6}{'pending':>9}{'durable':>9}" if self.saves is not None else '')

    def line(self, row):
        if row['update'] <= self.warmup:
            seconds = 'warmup'
        elif self.previous is None:
            seconds = '-'
        else:
            seconds = f"{row['monotonic_seconds'] - self.previous:.3f}"
        self.previous = row['monotonic_seconds']
        text = f"{row['update']:>6}{seconds:>10}{row['global_mean_loss']:>10.6f}"
        if self.saves is not None:
            text += (f"{'yes' if row['update'] in self.saves else '':>6}"
                     f"{'yes' if row['checkpoint_upload_pending'] else 'no':>9}{row['durable_update']:>9}")
        return text

    @staticmethod
    def flush_line(seconds, durable):
        return f"{'flush':>6}{seconds:>10.3f}{'-':>10}{'-':>6}{'-':>9}{durable:>9}"


def flush_seconds(rows, events, final_update):
    """Time from the last row to the final save becoming durable (0 if already durable)."""
    last = rows[-1]
    if last['durable_update'] == final_update:
        return 0.0
    for event in events:
        if event.get('phase') == 'globally_complete' and event.get('update') == final_update:
            return event['monotonic_seconds'] - last['monotonic_seconds']
    return None


def pairs_lines(pairs, width=29):
    width = max([width] + [len(label) + 2 for label, _ in pairs])
    return [f'  {label:<{width}}{value}' for label, value in pairs]


def lines_for(pairs):
    """Each line at width 29, or two spaces after a longer label."""
    return [line for pair in pairs for line in pairs_lines([pair])]


def describe(stage):
    saving = {'none': 'no saving', 'sync': 'sync saves', 'process': 'process saves'}[stage['saving']]
    if stage['saving'] != 'none' and stage['kind'] != 'restore':
        saving += ' after updates ' + ', '.join(str(u) for u in stage['checkpoint_updates'])
    if stage['kind'] == 'restore':
        saving = f'sync restore from ckpt-sync update {RESTORE_FROM}'
    extra = f", {stage['sequence_length']:,} positions" if stage['sequence_length'] != 2048 else ''
    extra += ', profiled' if stage['kind'] == 'profile' else ''
    return f"{stage['name']} ({stage['transport']}, {stage['attention']} attention, {saving}{extra})"


def header_lines(stage, allocation_id, stage_dir):
    used = dict(LLM_TRANSPORT=stage['transport'], ATTENTION=stage['attention'], CHECKPOINT_MODE=stage['saving'])
    lines = [f"[aim347] {stage['name']}: " + ' '.join(f'{k}={v}' for k, v in used.items())]
    changed = [f'{key}={value}' for key, value in used.items() if stage['job_env'].get(key) != value]
    if changed:
        lines.append('[aim347] this run alone uses ' + ' '.join(changed) + '; job.env is unchanged')
    work = (f"{stage['updates']} updates of {stage['global_batch']} sequences x "
            f"{stage['sequence_length']:,} positions")
    saves = ', '.join(str(u) for u in stage['checkpoint_updates'])
    if stage['kind'] == 'restore':
        lines.append(f'[aim347] restores ckpt-sync after update {RESTORE_FROM}, runs updates '
                     f'{RESTORE_FROM + 1}-{RESTORE_STOP}, then stops')
    elif stage['kind'] == 'interrupt':
        lines.append(f'[aim347] {work} with process saves after updates {saves}; stops after update '
                     f'{INTERRUPT_AFTER} and resumes from the last completed checkpoint')
    else:
        timed = f"updates {stage['warmup_updates'] + 1}-{stage['updates']} are timed"
        if stage['saving'] != 'none':
            timed += f' with saves after updates {saves}'
        if stage['kind'] == 'profile':
            lines.append(f'[aim347] {work}; this run is profiled, so its time is not compared')
        else:
            lines.append(f'[aim347] {work}; {timed} ({timed_targets(stage):,} targets)')
    lines.append(f'[aim347] allocation {allocation_id}, results {stage_dir}')
    return lines


def nccl_pairs(stage, stage_dir, world):
    logs = segment_logs(stage, Path(stage_dir))
    texts = []
    for log in logs:
        try:
            texts.append(Path(log).read_text(errors='replace'))
        except OSError:
            texts.append('')
    nccl = parse_nccl(''.join(texts))
    if len(logs) == 1:
        pairs = [('NCCL network', network_text(nccl, world) or 'no "Using network" lines in the log')]
    else:
        # Each segment starts its own ranks; report each one.
        pairs = [(f'NCCL network ({log.stem} run)', network_text(parse_nccl(text), world)
                  or 'no "Using network" lines in the log') for log, text in zip(logs, texts)]
    if stage['transport'] == 'efa':
        # Counted per segment, because each segment starts its own ranks.
        protos = [nccl_proto_text(parse_nccl(text), world) for text in texts]
        proto = (protos[0] if len(set(protos)) == 1
                 else '; '.join(f'{log.stem} run: {value}' for log, value in zip(logs, protos)))
        pairs += [('provider', '; '.join(nccl['providers']) or 'not reported'),
                  ('protocol', '; '.join(nccl['protocols']) or 'not reported'),
                  ('NCCL_PROTO', proto),
                  ('channels sending from GPU memory (/GDRDMA)',
                   f"{nccl['gdrdma_channels']} of {nccl['network_channels']} network channel lines")]
    return pairs


def efa_pair(stage_dir):
    deltas = efa_deltas(read_json(Path(stage_dir) / 'efa-before.json', {}),
                        read_json(Path(stage_dir) / 'efa-after.json', {}))
    return ('EFA bytes during the run', efa_text(deltas))


def timed_block(stage, stage_dir, summary, ceilings, world, efa_losses=None):
    if stage['saving'] != 'none':
        return checkpoint_block(stage, stage_dir, summary, world)
    seconds = summary['measured_seconds']
    tokens = summary['measured_useful_tokens']
    rate = summary['training_throughput_tokens_per_second']
    first = stage['warmup_updates'] + 1
    if ceilings:
        peak = ceilings['gemm_tflops_per_gpu']
        mfu_text = (f"{100 * mfu(rate, stage['sequence_length'], peak, world):.1f} %    "
                    f"({flop_per_token(stage['sequence_length']) / 1e9:.2f} GFLOP per token, {peak:.1f} TFLOP/s per GPU)")
    else:
        mfu_text = 'n/a (run bash tools/hold-and-measure.sh)'
    lines = [describe(stage)] + lines_for(
        [(f"timed updates {first}-{stage['updates']}", f'{seconds:.2f} s  ({tokens:,} targets)'),
         ('tokens/s', f'{rate:,.0f}'),
         ('GPU-seconds per 1M tokens', f'{gpu_seconds_per_million_tokens(seconds, tokens, world):.0f}'),
         ('MFU vs your GEMM peak', mfu_text),
         ('peak memory (largest rank)', f'{peak_memory_gib(summary):.2f} GiB')]
        + nccl_pairs(stage, stage_dir, world) + [efa_pair(stage_dir)])
    if efa_losses is not None:
        lines += loss_comparison_lines(stage, stage_dir, efa_losses)
    return lines


def loss_comparison_lines(stage, stage_dir, reference_rows):
    rows = read_rows(Path(stage_dir) / 'train' / 'rank-0-updates.jsonl')
    last = stage['updates']
    if not reference_rows:
        return ['  stage efa has no rows to compare losses with; run bash tools/run.sh efa first']
    these = [row['global_mean_loss'] for row in rows]
    eager = [row['global_mean_loss'] for row in reference_rows]
    return pairs_lines([
        (f"losses 1-{last} (this run, {stage['attention']})", ' '.join(f'{v:.6f}' for v in these)),
        (f'losses 1-{last} (stage efa, eager)', ' '.join(f'{v:.6f}' for v in eager)),
        ('difference', ' '.join(f'{a - b:+.6f}' for a, b in zip(these, eager)))])


def checkpoint_block(stage, stage_dir, summary, world):
    goodput = summary.get('training_goodput', {})
    first = stage['warmup_updates'] + 1
    return [describe(stage)] + lines_for(
        [(f"timed updates {first}-{stage['updates']} with saves and final flush",
          f"{summary['measured_seconds']:.2f} s  ({summary['measured_useful_tokens']:,} targets)"),
         ('durable at the end', f"update {stage['updates']}, {goodput.get('retained_progress_tokens', 0):,} tokens"),
         ('goodput interval', f"{summary['segment_seconds']:.2f} s  "
                              '(from the end of model loading to the end of the final flush)'),
         ('GPUs', str(summary['allocation']['gpu_count']))]
        + nccl_pairs(stage, stage_dir, world) + [efa_pair(stage_dir)])


def restore_block(stage, stage_dir, reference_rows, world):
    rows = read_rows(Path(stage_dir) / 'train' / 'rank-0-updates.jsonl')
    reference = {row['update']: row['global_mean_loss'] for row in reference_rows}
    lines = [describe(stage), f"{'update':>8}{'loss (restore)':>17}{'loss (ckpt-sync)':>19}{'difference':>13}"]
    differences = []
    for row in rows:
        before = reference.get(row['update'])
        difference = None if before is None else row['global_mean_loss'] - before
        if difference is not None:
            differences.append(abs(difference))
        lines.append(f"{row['update']:>8}{row['global_mean_loss']:>17.6f}"
                     + (f'{before:>19.6f}{difference:>+13.6f}' if before is not None else f"{'-':>19}{'-':>13}"))
    lines.append(f"first update after restore: {rows[0]['update'] if rows else 'none'}")
    if differences:
        lines.append(f'largest loss difference over updates {rows[0]["update"]}-{rows[-1]["update"]}: '
                     f'{max(differences):.6f}')
    return lines + lines_for(nccl_pairs(stage, stage_dir, world) + [efa_pair(stage_dir)])


def upload_sentence(row, save_updates, stop):
    saved = [update for update in save_updates if update <= stop]
    newest = saved[-1]
    earlier = saved[-2] if len(saved) > 1 else 0
    if row['checkpoint_upload_pending'] and row['durable_update'] == earlier:
        return f'the update-{newest} upload was still pending (durable update {earlier})'
    return f"the update-{newest} upload had finished (durable update {row['durable_update']})"


def interrupt_block(stage, stage_dir, world):
    campaign_dir = Path(stage_dir) / 'campaign'
    campaign = read_json(campaign_dir / 'campaign.json', {})
    interrupted = read_rows(campaign_dir / 'interrupted' / 'rank-0-updates.jsonl')
    resumed = read_rows(campaign_dir / 'resumed' / 'rank-0-updates.jsonl')
    printer = RowPrinter(stage['warmup_updates'], stage['checkpoint_updates'])
    recap = [printer.line(row) for row in interrupted][-5:]
    stop = campaign.get('interrupted_update', INTERRUPT_AFTER)
    restored = campaign.get('restored_update')
    row = next((r for r in interrupted if r['update'] == stop), None)
    completed = sorted(path.parent.name for path in (campaign_dir / 'interrupted' / 'checkpoints').glob(
        'update-*/COMPLETED.json'))
    lines = [printer.header()] + recap + [describe(stage)]
    lines.append(f'  stopped after update {stop}; '
                 + (upload_sentence(row, stage['checkpoint_updates'], stop) if row else 'no row for the stop update'))
    lines.append('  completed checkpoints at the stop: ' + (', '.join(completed) or 'none'))
    lines.append(f"  resumed from update {restored}; first update after the resume: "
                 f"{resumed[0]['update'] if resumed else 'none'}")
    lines.append(f"  updates run twice: {restored + 1}-{stop} ({campaign.get('replayed_update_count')} updates)")
    lines += pairs_lines([('goodput including the stop',
                           f"{campaign['training_goodput_tokens_per_second']:,.0f} tokens/s, "
                           f"{campaign['tokens_per_allocated_gpu_hour'] / 1e6:.2f} M tokens per GPU-hour  "
                           '(first launch to the final durable save)')])
    return lines + lines_for(nccl_pairs(stage, stage_dir, world) + [efa_pair(stage_dir)])


def tail(path, count=40):
    try:
        return Path(path).read_text(errors='replace').splitlines()[-count:]
    except OSError:
        return []


def failure_lines(stage, stage_dir, reason):
    stage_dir = Path(stage_dir)
    lines = [f"{stage['name']} failed: {reason}", f"last 40 lines of {stage_dir / 'launch.log'}:"]
    lines += tail(stage_dir / 'launch.log')
    if stage['kind'] == 'interrupt':
        segments = [p for p in segment_logs(stage, stage_dir) if p.exists()]
        if segments:
            last = max(segments, key=lambda p: p.stat().st_mtime)
            lines += [f'last 40 lines of {last}:'] + tail(last)
    return lines


def result_lines(stage, stage_dir, run, results, world):
    """What run.sh prints when a stage has finished, by exit status."""
    stage_dir = Path(stage_dir)
    code = run.get('exit_status')
    if code == 4:
        return failure_lines(stage, stage_dir, run.get('failure') or 'unknown')
    if stage['kind'] == 'restore':
        lines = restore_block(stage, stage_dir, read_rows(Path(results) / 'ckpt-sync' / 'train' / 'rank-0-updates.jsonl'), world)
    elif stage['kind'] == 'interrupt':
        lines = interrupt_block(stage, stage_dir, world)
    elif stage['kind'] == 'profile':
        lines = [describe(stage)] + lines_for(
            [('traces', f"{len(list((stage_dir / 'train').glob('trace-rank-*.json')))} files in {stage_dir / 'train'}"),
             ('read the profile with', f"python3 tools/profile-table.py {stage['name']}")]
            + nccl_pairs(stage, stage_dir, world) + [efa_pair(stage_dir)])
    else:
        summary = read_json(stage_dir / 'train' / 'summary.json')
        if not summary:
            lines = [describe(stage), '  no summary.json']
        else:
            efa_rows = (read_rows(Path(results) / 'efa' / 'train' / 'rank-0-updates.jsonl')
                        if stage['name'] == 'sdpa' else None)
            lines = timed_block(stage, stage_dir, summary, read_ceilings(results), world, efa_rows)
            lines += stage_chart_lines(stage, results, world)
    for check in run.get('checks', []):
        if not check['ok']:
            lines.append(f"check failed: {check['name']}: {check['detail']}")
    if code == 5:
        lines.append('The run finished, but a check failed. Fix it before reading the result.')
    if stage['subcommand'] == 'sdpa-socket':
        lines.append('job.env is unchanged')
    return lines


# ---------------------------------------------------------------- charts
# Each chart answers one question about the participant's own runs. A stage
# without a bar prints the reason; nothing is estimated.

MAIN_PATH_LABELS = {'baseline': ('baseline (socket, eager)', 'baseline'), 'efa': ('efa (efa, eager)', 'EFA'),
                    'sdpa': ('sdpa (efa, sdpa)', 'SDPA')}
# The full-length stage that a half-length probe is read beside, by transport and attention.
FULL_LENGTH = {('socket', 'eager'): 'baseline', ('socket', 'sdpa'): 'sdpa-socket',
               ('efa', 'eager'): 'efa', ('efa', 'sdpa'): 'sdpa'}
UPDATE_COMPARED_WITH = {'sdpa-socket': 'baseline', 'efa': 'baseline', 'sdpa': 'efa'}
SAVING_STAGES = (('ckpt-sync', 'synchronous saves'), ('ckpt-process', 'process-mode saves'))


def chart_stage(results, name):
    """(run, summary, None) for a stage that has a result, else (run or None, None, the reason)."""
    run = read_json(Path(results) / name / 'run.json')
    if not run:
        return None, None, 'not run yet'
    if run.get('status') != 'finished':
        return run, None, 'still running'
    if run.get('exit_status') == 4:
        return run, None, 'failed, so it has no bar; run it again'
    summary = read_json(Path(results) / name / 'train' / 'summary.json')
    if not summary:
        return run, None, 'no summary.json'
    return run, summary, None


def failed_check(run):
    return ', a check failed' if run.get('exit_status') == 5 else ''


def cumulative_chart_lines(results):
    """One bar per main-path stage: 1.00x plus each later step's measured change."""
    _, baseline, _ = chart_stage(results, 'baseline')
    rows = []
    for name in MAIN_PATH:
        label, change = MAIN_PATH_LABELS[name]
        run, summary, problem = chart_stage(results, name)
        row = dict(label=label, change=change, total=None, note=problem)
        if not problem:
            row['label'] += failed_check(run)
            if baseline is None:
                row['note'] = 'no baseline to compare with'
            elif name == 'baseline':
                row['total'] = 1.0
            else:
                try:
                    row['total'] = paired_training_report(baseline, summary)['speedup_ratio']
                except ValueError as error:
                    row['note'] = 'not compared: ' + str(error).replace('paired training mismatch: ', 'different ')
        rows.append(row)
    return charts.cumulative_chart('How much faster than baseline? Each bar adds one step '
                                   '(your runs, same work on the same allocation)', rows)


def update_time_names(stage):
    if stage['subcommand'] == 'probe':
        return [FULL_LENGTH[(stage['transport'], stage['attention'])], stage['name']]
    if stage['name'] in UPDATE_COMPARED_WITH:
        return [UPDATE_COMPARED_WITH[stage['name']], stage['name']]
    return [stage['name']]


def update_time_chart_lines(results, names, ceilings, world):
    """One update of each stage: the model's arithmetic at the GEMM peak, and the rest."""
    rows = []
    for name in names:
        run, summary, problem = chart_stage(results, name)
        if problem:
            rows.append(dict(label=name, note=problem))
            continue
        stage = run['plan']
        timed = stage['updates'] - stage['warmup_updates']
        seconds = summary['measured_seconds'] / timed
        row = dict(label=f"{name}, {stage['sequence_length']:,} positions", text=f'{seconds:.2f} s{failed_check(run)}')
        if ceilings:
            arithmetic = compute_only_seconds(summary['measured_useful_tokens'] / timed, stage['sequence_length'],
                                              ceilings['gemm_tflops_per_gpu'], world)
            row.update(segments=[(arithmetic, charts.FILLS[0]), (seconds - arithmetic, charts.FILLS[2])],
                       detail=f'{charts.FILLS[0]} arithmetic at your GEMM peak {arithmetic:.2f} s   '
                              f'{charts.FILLS[2]} the rest {seconds - arithmetic:.2f} s')
        else:
            row.update(segments=[(seconds, charts.FILLS[2])],
                       detail='arithmetic n/a (run bash tools/hold-and-measure.sh)')
        rows.append(row)
    return charts.stacked_chart("One update: the model's arithmetic at your GEMM peak, and the rest "
                                '(mean of the timed updates)', rows, [], 's')


def memory_chart_lines(results, names):
    rows = []
    for name in names:
        run, summary, problem = chart_stage(results, name)
        if problem:
            rows.append(dict(label=name, value=None, note=problem))
            continue
        peak = peak_memory_gib(summary)
        rows.append(dict(label=f"{name} ({run['plan']['attention']} attention)", value=peak,
                         text=f'{peak:.2f} GiB{failed_check(run)}'))
    return charts.bar_chart('Peak memory of the largest rank', rows)


def goodput_chart_lines(results):
    """Training goodput of the two saving stages beside the throughput of stage sdpa, which does not save."""
    rows = []
    for name, saving in SAVING_STAGES:
        run, summary, problem = chart_stage(results, name)
        goodput = (summary or {}).get('training_goodput')
        if problem or not goodput:
            rows.append(dict(label=f'{name}, {saving}', value=None, note=problem or 'no goodput in summary.json'))
            continue
        value = goodput['tokens_per_allocated_gpu_hour'] / 1e6
        rows.append(dict(label=f'{name}, {saving}', value=value, text=f'{value:.2f}{failed_check(run)}'))
    run, summary, problem = chart_stage(results, 'sdpa')
    label = 'sdpa, no saves (throughput, reference)'
    if problem:
        rows.append(dict(label=label, value=None, note=problem))
    else:
        value = summary['training_throughput_tokens_per_second'] * 3600 / summary['allocation']['gpu_count'] / 1e6
        rows.append(dict(label=label, value=value, text=f'{value:.2f}{failed_check(run)}'))
    return charts.bar_chart('Training goodput, M tokens per GPU-hour (higher is better)', rows, notes=[
        'The reference is the throughput of the timed updates of stage sdpa, a run that does not save.'])


def stage_chart_lines(stage, results, world):
    """The charts printed under the result block of a timed stage."""
    if stage['kind'] != 'timed':
        return []
    if stage['saving'] != 'none':
        return [''] + goodput_chart_lines(results)
    lines = [''] + update_time_chart_lines(results, update_time_names(stage), read_ceilings(results), world)
    if stage['name'] == 'sdpa':
        lines += [''] + memory_chart_lines(results, ['efa', 'sdpa'])
    if stage['name'] in MAIN_PATH:
        lines += [''] + cumulative_chart_lines(results)
    return lines


def protocol_chart_lines(runs):
    """Bus bandwidth by message size, NCCL's own choice beside Simple. runs: [(label, rows, choices)]."""
    rows = []
    for size in sorted(set(runs[0][1]) | set(runs[1][1])):
        for index, (label, table, choices) in enumerate(runs):
            name = f"{size_text(size) if index == 0 else '':>7}  {label}"
            if size not in table:
                rows.append(dict(label=name, value=None, note='no row in all_reduce_perf.txt'))
                continue
            busbw = table[size][1]
            rows.append(dict(label=name, value=busbw, text=f"{busbw:.2f}  {', '.join(choices.get(size, ['-']))}"))
    return charts.bar_chart('Bus bandwidth of all_reduce_perf by message size, GB/s (higher is better)', rows)


# ---------------------------------------------------------------- allocation

GRANTED = re.compile(r'Granted job allocation (\d+)')


def parse_granted(text):
    match = GRANTED.search(text)
    return int(match.group(1)) if match else None


def parse_scontrol_record(text):
    """Fields of one `scontrol show -o` record (values without spaces)."""
    return dict(token.split('=', 1) for token in text.split() if '=' in token)


def time_limit_minutes(value):
    """Slurm [days-]hours:minutes:seconds, or minutes:seconds; other values unchanged."""
    if not re.fullmatch(r'(\d+-)?\d+(:\d+){0,2}', value):
        return value
    days, _, clock = value.rpartition('-')
    parts = [int(part) for part in clock.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 1440 + parts[0] * 60 + parts[1] + (parts[2] > 0)


def write_allocation(path, job_id, nodelist, cpus_per_node, nodes=2):
    values = [('SLURM_JOB_ID', job_id), ('SLURM_JOB_NUM_NODES', nodes), ('SLURM_JOB_NODELIST', nodelist),
              ('SLURM_JOB_CPUS_PER_NODE', f'{cpus_per_node}(x{nodes})'), ('AIM347_ALLOCATION_ID', job_id)]
    write_atomic(path, ''.join(f'export {key}={shlex.quote(str(value))}\n' for key, value in values), mode=0o600)


def read_allocation(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        words = shlex.split(line)
        if len(words) == 2 and words[0] == 'export' and '=' in words[1]:
            key, value = words[1].split('=', 1)
            values[key] = value
    return values


def write_atomic(path, text, mode=0o644):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(text)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


# ---------------------------------------------------------------- provisioning

def env_file_text(*, stage_dir, companion, slurm_version, cpus_per_node, instance_type, nic,
                  image_sha256, grafana_url, password_file):
    """aim347.env: export lines, values quoted; the PATH line expands $PATH when sourced."""
    if not re.fullmatch(r'\d+\.\d+', slurm_version):
        raise UsageError(f'invalid Slurm version {slurm_version!r}')
    home = str(stage_dir)
    values = [('AIM347_HOME', home), ('AIM347_COMPANION', str(companion)),
              ('PATH', None), ('PARTITION', 'gpu'), ('GPUS_PER_NODE', 8),
              ('AIM347_CPUS_PER_NODE', cpus_per_node), ('AIM347_INSTANCE_TYPE', instance_type),
              ('NCCL_SOCKET_IFNAME', f'={nic}'), ('LAB_IMAGE', f'{home}/images/aim347-lab.sqsh'),
              ('LAB_IMAGE_SHA256', image_sha256), ('LLM_DATA_DIR', f'{home}/data'),
              ('LLM_RESULTS_DIR', f'{home}/results'), ('LLM_CONTAINER_NAME', 'aim347-lab'),
              ('AIM347_ALLOCATION_LABEL', 'pcs-g7.48xlarge-2nodes-16gpu'), ('AIM347_GRAFANA_URL', grafana_url),
              ('AIM347_GRAFANA_PASSWORD_FILE', str(password_file)), ('AIM347_SKIP_LEGACY_ENV', 1)]
    lines = []
    for key, value in values:
        if key == 'PATH':
            lines.append(f'export PATH="/opt/aws/pcs/scheduler/slurm-{slurm_version}/bin:$PATH"\n')
        else:
            lines.append(f'export {key}={shlex.quote(str(value))}\n')
    return ''.join(lines)


def dataset_fingerprint(directory):
    digest = hashlib.sha256()
    for name in ('tokens.bin', 'lengths.bin'):
        with (Path(directory) / name).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def verify_dataset(data_dir, name, lab=LAB):
    """(ok, detail, manifest) for one prepared dataset against its config and pin."""
    directory = Path(data_dir) / name
    manifest = read_json(directory / 'manifest.json')
    if not manifest:
        return False, f'no manifest in {directory}', None
    config = config_values(DATASETS[name]['config'], lab)
    expected = dict(records=DATASETS[name]['records'], sequence_length=config['sequence_length'],
                    minimum_document_tokens=MINIMUM_DOCUMENT_TOKENS, **{k: config[k] for k in PROVENANCE_KEYS})
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False, f'manifest {key} is {manifest.get(key)!r}, expected {value!r}', manifest
    rows, positions = manifest['records'], manifest['sequence_length']
    sizes = {'tokens.bin': rows * positions * 4, 'lengths.bin': rows * 4}
    for file, size in sizes.items():
        path = directory / file
        if not path.is_file() or path.stat().st_size != size:
            return False, f'{file} is missing or not {size} bytes', manifest
    fingerprint = dataset_fingerprint(directory)
    if fingerprint != manifest['sha256']:
        return False, f'fingerprint {fingerprint[:8]} differs from its manifest ({manifest["sha256"][:8]})', manifest
    pinned = EXPECTED_FINGERPRINTS[name]
    if pinned and fingerprint != pinned:
        return False, f'fingerprint {fingerprint[:8]} (expected {pinned[:8]})', manifest
    return True, f'fingerprint {fingerprint[:8]} (expected {(pinned or manifest["sha256"])[:8]})', manifest


def verify_model(data_dir, lab=LAB):
    config = config_values('configs/smol-2048.json', lab)
    expected = {key: config[key] for key in ('model_id', 'model_revision')}
    pin = read_json(Path(data_dir) / 'model' / 'aim347-pin.json')
    if pin != expected or not (Path(data_dir) / 'model' / 'config.json').is_file():
        return False, f'model pin {pin!r} does not match {expected!r}'
    return True, f"{expected['model_id']} at {expected['model_revision'][:8]}"


# ---------------------------------------------------------------- node readers

def node_name():
    return os.environ.get('SLURMD_NODENAME') or socket.gethostname()


def efa_devices(root='/sys/class/infiniband'):
    root = Path(root)
    return [device.name for device in sorted(root.iterdir()) if root.is_dir()
            and os.path.basename(os.path.realpath(device / 'device' / 'driver')) == 'efa']


def node_probe():
    report = dict(host=node_name(), gpus=[], efa=efa_devices(), error=None)
    try:
        output = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=60, check=True).stdout
        report['gpus'] = [[part.strip() for part in line.rsplit(',', 1)] for line in output.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError) as error:
        report['error'] = f'nvidia-smi: {error}'
    return report


def http_text(url, timeout=10):
    """GET a URL on the login node; the monitoring stack uses a self-signed certificate."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
        return response.status, response.read().decode()


def prometheus_count(base, query, fetch=http_text):
    status, body = fetch(f'{base}/api/v1/query?' + urllib.parse.urlencode({'query': query}))
    result = json.loads(body)['data']['result']
    return int(float(result[0]['value'][1])) if result else 0


# ---------------------------------------------------------------- check.sh

def check_lines(env, sinfo_text, node_reports, fetch=http_text, lab=LAB):
    """(lines, failures) of tools/check.sh, from the gathered Slurm and node outputs."""
    rows = []

    def add(name, ok, detail):
        rows.append((name, ok, detail))

    add('environment', True, env.get('AIM347_ENV', '/fsx/aim347/participant/aim347.env'))
    gpus_per_node = int(env.get('GPUS_PER_NODE', 8))
    nodes = [line.split() for line in sinfo_text.splitlines() if len(line.split()) == 2]
    names = sorted({name for name, _ in nodes})
    states = sorted({state for _, state in nodes})
    usable = bool(states) and all(state in ('idle', 'alloc', 'mix', 'comp') for state in states)
    add('partition', len(names) == 2 and usable,
        f"{env.get('PARTITION', 'gpu')}: {len(names)} nodes ({', '.join(names)}), {', '.join(states) or 'no state'}"
        + ('' if len(names) == 2 and usable else ' (expected 2 nodes in idle, alloc or mix)'))
    reports = {report['host']: report for report in node_reports}
    if len(reports) != 2:
        add('gpus', False, f'{len(reports)} nodes answered the node check (expected 2)')
    expected_efa = EFA_DEVICES.get(env.get('AIM347_INSTANCE_TYPE'), 1)
    for host in sorted(reports):
        report = reports[host]
        kinds = sorted({tuple(gpu) for gpu in report['gpus']})
        text = f"{host}: {len(report['gpus'])} x " + ('; '.join(f'{name}, {memory} MiB' for name, memory in kinds)
                                                     or 'no GPU') + (' each' if len(kinds) == 1 else '')
        ok = len(report['gpus']) == gpus_per_node and len(kinds) == 1
        add('gpus', ok, text + ('' if ok else f" (expected {gpus_per_node}){' ' + report['error'] if report.get('error') else ''}"))
    for host in sorted(reports):
        devices = reports[host]['efa']
        ok = len(devices) >= expected_efa
        add('efa', ok, f"{host}: {len(devices)} EFA devices ({', '.join(devices)})"
            + ('' if ok else f' (expected {expected_efa})'))
    home = Path(env.get('AIM347_HOME', '/fsx/aim347/participant'))
    try:
        fstype = subprocess.run(['findmnt', '-T', str(home), '-n', '-o', 'FSTYPE,TARGET'], capture_output=True,
                                text=True, timeout=10).stdout.split()
        stats = os.statvfs(home)
        size, free = stats.f_blocks * stats.f_frsize, stats.f_bavail * stats.f_frsize
        ok = fstype[:1] == ['lustre'] and free >= FSX_FREE_BYTES
        add('fsx', ok, f"{fstype[1] if len(fstype) > 1 else home}: {fstype[0] if fstype else 'unknown'}, "
                       f'{size / 2**40:.1f} TiB, {free / 2**40:.1f} TiB free (needs 300 GiB)')
    except (OSError, subprocess.SubprocessError) as error:
        add('fsx', False, f'{home}: {error}')
    image = Path(env.get('LAB_IMAGE', home / 'images/aim347-lab.sqsh'))
    recorded = (Path(str(image) + '.sha256').read_text().split() or [''])[0] if Path(str(image) + '.sha256').is_file() else ''
    verified = {line.split()[0] for line in Path(str(image) + '.verified').read_text().splitlines()
                if len(line.split()) >= 2 and line.split()[1] == 'OK'} if Path(str(image) + '.verified').is_file() else set()
    ok = image.is_file() and bool(recorded) and recorded == env.get('LAB_IMAGE_SHA256') and len(verified) >= 2
    add('image', ok, f'{image.name}, SHA-256 {recorded[:12] or "missing"} '
                     + ('checked on both nodes at preparation' if ok else
                        f'(environment {env.get("LAB_IMAGE_SHA256", "")[:12] or "missing"}, verified on '
                        f'{len(verified)} nodes, expected 2)'))
    data = Path(env.get('LLM_DATA_DIR', home / 'data'))
    for name in ('tokens-2048-160', 'tokens-1024-160', 'tokens-2048-1280'):
        ok, detail, manifest = verify_dataset(data, name, lab)
        shape = (f"{manifest['records']:,} documents x {manifest['sequence_length']:,} positions, "
                 if manifest else '')
        add('data', ok, f'{name}: {shape}{detail}')
    ok, detail = verify_model(data, lab)
    add('model', ok, detail)
    base = env.get('AIM347_PROMETHEUS_URL', 'https://localhost/prometheus')
    try:
        lustre = prometheus_count(base, 'count(aim347_lustre_collection_success == 1)', fetch)
        efa = prometheus_count(base, 'count(count by (instance_id) (node_amazonefa_tx_bytes))', fetch)
        ok = lustre == 2 and efa == 2
        add('monitoring', ok, 'Prometheus scrapes both GPU nodes; EFA counters and the Lustre collector present' if ok
            else f'Prometheus sees the Lustre collector on {lustre} nodes and EFA counters on {efa} nodes (expected 2 and 2)')
    except (OSError, ValueError, KeyError, IndexError) as error:
        add('monitoring', False, f'{base}: {error}')
    url = env.get('AIM347_GRAFANA_URL', '')
    try:
        status, body = fetch(env.get('AIM347_GRAFANA_HEALTH_URL', 'https://localhost/grafana/api/health'))
        password = Path(env.get('AIM347_GRAFANA_PASSWORD_FILE', STATE / 'grafana-admin-password')).read_text().strip()
        ok = status == 200 and json.loads(body).get('database') == 'ok' and bool(password)
        add('grafana', ok, f'{url}  user admin  password {password}' if ok else f'{url}: health {status} {body[:80]}')
    except (OSError, ValueError) as error:
        add('grafana', False, f'{url}: {error}')
    lines = ['AIM347 check'] + [f"  {name:<13}{'PASS' if ok else 'FAIL'}  {detail}" for name, ok, detail in rows]
    failures = sum(not ok for _, ok, _ in rows)
    lines.append('All checks passed.' if not failures else
                 f'{failures} checks failed. Fix the first FAIL line before continuing.')
    return lines, failures


# ---------------------------------------------------------------- run.sh

def results_dir():
    value = os.environ.get('LLM_RESULTS_DIR')
    if not value:
        raise UsageError('LLM_RESULTS_DIR is not set; open a new shell so it reads aim347.env')
    return Path(value)


def world_size():
    return 2 * int(os.environ.get('GPUS_PER_NODE', 8))


def utc_stamp():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    # The foreground run.sh never waits for its worker, so a finished worker
    # stays a zombie of it until run.sh exits.
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return True


def read_running(state=None):
    path = Path(state or STATE) / 'running'
    for _ in range(5):
        running = read_json(path)
        if running is not None:
            break
        if not path.exists():
            return None
        time.sleep(0.2)
    if running is None or not pid_alive(running.get('pid')):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    return running


def stage_run(results, name, invocation):
    run = read_json(Path(results) / name / 'run.json')
    return run if run and run.get('invocation') == invocation else None


def latest_invocation(results):
    runs = [run for run in (read_json(path) for path in Path(results).glob('*/run.json')) if run]
    if not runs:
        return None
    latest = max(runs, key=lambda run: run.get('start_utc', ''))
    return dict(invocation=latest['invocation'], stages=latest['stages'], pid=None,
                allocation_id=latest.get('allocation_id'))


def run_stage(stage, invocation, results, *, runner=subprocess.run, efa_reader=None, label=None, world=None):
    """The worker's part of one stage: launch, counters, checks, run.json. Returns the exit code."""
    results = Path(results)
    world = world or world_size()
    label = label or os.environ.get('AIM347_ALLOCATION_LABEL', 'aim347')
    stage_dir = results / stage['name']
    if stage_dir.exists():
        attempts = results / 'attempts'
        attempts.mkdir(exist_ok=True)
        stage_dir.rename(attempts / f"{stage['name']}-{utc_stamp()}")
    stage_dir.mkdir(parents=True)
    command = launch_command(stage, label, results, world)
    run = dict(stage=stage['name'], subcommand=stage['subcommand'], invocation=invocation['invocation'],
               stages=invocation['stages'], plan=stage, job_env=stage['job_env'],
               settings=dict(transport=stage['transport'], attention=stage['attention'], saving=stage['saving']),
               trainer_argv=command, config=stage['config'], data=stage['data'],
               allocation_id=invocation.get('allocation_id'), start_utc=time.time(), status='running')
    write_atomic(stage_dir / 'run.json', json.dumps(run, indent=2) + '\n')
    log = stage_dir / 'launch.log'
    efa_reader = efa_reader or read_efa_on_nodes
    try:
        with log.open('a') as stream:
            write_atomic(stage_dir / 'efa-before.json', json.dumps(efa_reader(stream), indent=2) + '\n')
            stream.flush()
            env = dict(os.environ, LLM_TRANSPORT=stage['transport'])
            launcher_exit = runner(command, cwd=str(LAB), env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            write_atomic(stage_dir / 'efa-after.json', json.dumps(efa_reader(stream), indent=2) + '\n')
        code, checks, failure = post_run_checks(stage, stage_dir, launcher_exit, world)
    except BaseException as error:
        run.update(end_utc=time.time(), exit_status=4, checks=[], status='finished',
                   failure=f'run.sh stopped the stage: {error!r}')
        write_atomic(stage_dir / 'run.json', json.dumps(run, indent=2) + '\n')
        raise
    run.update(end_utc=time.time(), launcher_exit=launcher_exit, exit_status=code, checks=checks,
               failure=failure, status='finished')
    write_atomic(stage_dir / 'run.json', json.dumps(run, indent=2) + '\n')
    return code


def read_efa_on_nodes(log):
    """One srun step on both nodes, outside the image; errors go to the stage log."""
    result = subprocess.run(['srun', '--nodes=2', '--ntasks-per-node=1', '--cpus-per-task=1', 'python3',
                             str(LAB / 'lib' / 'workshop.py'), 'efa-read'],
                            stdout=subprocess.PIPE, stderr=log, text=True)
    return merge_efa_readings(result.stdout.splitlines())


def combine(code, stage_code):
    return 4 if 4 in (code, stage_code) else (5 if 5 in (code, stage_code) else 0)


def worker(invocation, state=None, **options):
    """Background part of run.sh: every stage of one subcommand, in order."""
    state = Path(state or STATE)
    code = 0
    try:
        for stage in invocation['plan']:
            code = combine(code, run_stage(stage, invocation, results_dir(), **options))
            if code == 4:
                break
    finally:
        running = read_json(state / 'running')
        if running and running.get('invocation') == invocation['invocation']:
            (state / 'running').unlink()
    return code


def rows_sources(stage, stage_dir):
    if stage['kind'] == 'interrupt':
        return [stage_dir / 'campaign' / 'interrupted' / 'rank-0-updates.jsonl',
                stage_dir / 'campaign' / 'resumed' / 'rank-0-updates.jsonl']
    return [stage_dir / 'train' / 'rank-0-updates.jsonl']


def follow_stage(name, invocation, results, *, out=print, interval=1.0, clock=time.monotonic, world=None):
    """Print one stage's header, rows as they appear, then its result. Returns its exit code."""
    world = world or world_size()
    stage_dir = Path(results) / name
    started = clock()
    run = stage_run(results, name, invocation['invocation'])
    while run is None:
        if not pid_alive(invocation.get('pid')):
            out(f'[aim347] {name} did not start; see {STATE / "worker.log"}')
            return 4
        time.sleep(interval)
        run = stage_run(results, name, invocation['invocation'])
    stage = run['plan']
    for line in header_lines(stage, run.get('allocation_id'), stage_dir):
        out(line)
    live = run.get('status') != 'finished'
    saves = stage['checkpoint_updates'] if stage['saving'] != 'none' and stage['kind'] != 'restore' else None
    sources = rows_sources(stage, stage_dir)
    printed, segment, waited = 0, 0, 0
    printer = RowPrinter(stage['warmup_updates'], saves)
    header_done = False
    while True:
        rows = read_rows(sources[segment])
        if rows and not header_done:
            out(printer.header())
            header_done = True
        for row in rows[printed:]:
            out(printer.line(row))
        printed = len(rows)
        finished = (run or {}).get('status') == 'finished'
        if segment + 1 < len(sources) and sources[segment + 1].exists():
            segment, printed = segment + 1, 0
            printer = RowPrinter(stage['warmup_updates'], saves)
            out(f'[aim347] stopped after update {INTERRUPT_AFTER}; resumed from the last completed checkpoint')
            continue
        if finished:
            break
        if not pid_alive(invocation.get('pid')):
            out(f'[aim347] the stage stopped before it finished; see {stage_dir / "launch.log"}')
            return 4
        elapsed = int(clock() - started)
        if live and not rows and elapsed >= waited + 10:
            waited = elapsed - elapsed % 10
            out(f'[aim347] {waited} s: ' + ('restarting from the last completed checkpoint' if segment
                                            else 'starting containers and loading the model'))
        time.sleep(interval)
        run = stage_run(results, name, invocation['invocation'])
    code = run['exit_status']
    if saves is not None and stage['kind'] != 'interrupt' and code != 4 and rows:
        events = read_rows(stage_dir / 'train' / 'checkpoint-events-rank-0.jsonl')
        flush = flush_seconds(rows, events, stage['updates'])
        if flush is not None:
            out(RowPrinter.flush_line(flush, stage['updates']))
    out(f"[aim347] finished in {run['end_utc'] - run['start_utc']:.0f} s")
    for line in result_lines(stage, stage_dir, run, results, world):
        out(line)
    return code


def follow(invocation, results, **options):
    code = 0
    try:
        for name in invocation['stages']:
            stage_code = follow_stage(name, invocation, results, **options)
            code = combine(code, stage_code)
            if stage_code == 4:
                break
    except KeyboardInterrupt:
        print('\nStill running. Follow it again with: bash tools/run.sh attach')
        return 0
    return code


def restore_ready(results):
    run = read_json(Path(results) / 'ckpt-sync' / 'run.json') or {}
    marker = Path(results) / 'ckpt-sync' / 'train' / 'checkpoints' / f'update-{RESTORE_FROM:06d}' / 'COMPLETED.json'
    return run.get('exit_status') in (0, 5) and marker.is_file()


def start_run(subcommand, state=None):
    """Foreground part of run.sh. Returns the exit code."""
    state = Path(state or STATE)
    results = results_dir()
    running = read_running(state)
    if running:
        current = next((name for name in running['stages']
                        if (stage_run(results, name, running['invocation']) or {}).get('status') != 'finished'),
                       running['stages'][0])
        if subcommand != 'attach':
            print(f'Stage {current} is still running; following it')
        return follow(running, results)
    if subcommand == 'attach':
        last = latest_invocation(results)
        if not last:
            print('No stage has run yet.')
            return 0
        return follow(last, results)
    if not os.environ.get('AIM347_ALLOCATION_ID'):
        print('AIM347: no session allocation. Run: bash tools/hold-and-measure.sh', file=sys.stderr)
        return 3
    stages = plan(subcommand, read_job_env())
    if subcommand == 'restore' and not restore_ready(results):
        raise UsageError(f'restore needs a finished ckpt-sync stage with a completed update-{RESTORE_FROM:06d}. '
                         'Run: bash tools/run.sh checkpoint')
    state.mkdir(mode=0o700, exist_ok=True)
    invocation = dict(invocation=utc_stamp(), stages=[stage['name'] for stage in stages], plan=stages,
                      subcommand=subcommand, allocation_id=os.environ['AIM347_ALLOCATION_ID'], pid=None)
    try:
        lock = os.open(state / 'running', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print('Another run.sh is starting a stage; following it')
        time.sleep(2)
        return start_run('attach', state)
    with os.fdopen(lock, 'w') as stream:
        stream.write(json.dumps(dict(invocation, pid=os.getpid())))
    with (state / 'worker.log').open('w') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', json.dumps(invocation)],
                                 cwd=str(LAB), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    invocation['pid'] = child.pid
    write_atomic(state / 'running', json.dumps(invocation), mode=0o600)
    return follow(invocation, results)


# ---------------------------------------------------------------- optional: NCCL protocols

# tools/nccl-protocols.sh runs all_reduce_perf over EFA with OFI_NCCL_PROTOCOL=RDMA
# twice: with NCCL's own protocol choice, then with NCCL_PROTO=Simple.
PROTOCOL_RUNS = (('nccl-choice', 'NCCL_PROTO not set'), ('simple', 'NCCL_PROTO=Simple'))
# Rank 0 prints one such line per collective when NCCL_DEBUG_SUBSYS includes TUNING.
NCCL_TUNING = re.compile(r'NCCL INFO AllReduce: (\d+) Bytes -> Algo (\S+) proto (\S+)')


def nccl_tests_rows(text):
    """{size: (out-of-place time in us, out-of-place bus bandwidth in GB/s, #wrong of both)} of an nccl-tests table."""
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 13 and fields[0].isdigit() and fields[1].isdigit():
            try:
                rows[int(fields[0])] = (float(fields[5]), float(fields[7]), int(fields[8]) + int(fields[12]))
            except ValueError:
                continue
    return rows


def tuning_choices(text):
    """{size in bytes: ['ALGO/PROTO', ...]} from the TUNING lines of rank 0."""
    choices = {}
    for match in NCCL_TUNING.finditer(text):
        choices.setdefault(int(match.group(1)), set()).add(f'{match.group(2)}/{match.group(3)}')
    return {size: sorted(values) for size, values in choices.items()}


def size_text(size):
    return f'{size // 2**20} MiB' if size % 2**20 == 0 else f'{size:,} B'


def protocol_command(variant, out, env, lab=LAB):
    """The srun step of one run: one MPI rank per GPU on both nodes, in the training image."""
    gpus = int(env.get('GPUS_PER_NODE', 8))
    command = ['srun', '--nodes=2', f'--ntasks={2 * gpus}', f'--ntasks-per-node={gpus}',
               f"--cpus-per-task={int(env['AIM347_CPUS_PER_NODE']) // gpus}", '--mpi=pmix', '--cpu-bind=none',
               '--time=10', f"--container-image={env['LAB_IMAGE']}"]
    if env.get('LLM_CONTAINER_NAME'):
        command.append(f"--container-name={env['LLM_CONTAINER_NAME']}")
    command += [f"--container-mounts={lab}:/opt/aim347,{env['LLM_RESULTS_DIR']}:{env['LLM_RESULTS_DIR']}",
                '--container-workdir=/opt/aim347', '--no-container-remap-root', '--container-env=NCCL_SOCKET_IFNAME',
                'bash', 'lib/nccl-protocols-rank.sh', variant, str(Path(out) / variant)]
    return command


def protocol_report(out, world):
    """(lines, exit code) from the two runs in `out`: 0, 4 when a table is missing, 5 on wrong values."""
    runs = []
    for variant, label in PROTOCOL_RUNS:
        directory = Path(out) / variant
        try:
            table = (directory / 'all_reduce_perf.txt').read_text(errors='replace')
        except OSError:
            table = ''
        logs = ''.join(path.read_text(errors='replace') for path in sorted(directory.glob('nccl-*.log')))
        runs.append((label, nccl_tests_rows(table), tuning_choices(logs), parse_nccl(logs)))
    missing = [label for label, rows, _, _ in runs if not rows]
    if missing:
        return [f'[aim347] no all_reduce_perf table for {", ".join(missing)} in {out}'], 4
    lines = [f"{'':12}{runs[0][0]:<40}{runs[1][0]}",
             (f"{'size':>9}" + f"{'time (us)':>12}{'busbw (GB/s)':>14}  {'NCCL ran':<12}" * 2).rstrip()]
    for size in sorted(set(runs[0][1]) | set(runs[1][1])):
        line = f'{size_text(size):>9}'
        for _, rows, choices, _ in runs:
            time_us, busbw, _ = rows.get(size, (float('nan'), float('nan'), 0))
            line += f"{time_us:>12.1f}{busbw:>14.2f}  {', '.join(choices.get(size, ['-'])):<12}"
        lines.append(line.rstrip())
    lines.append('[aim347] NCCL_PROTO in the NCCL logs: '
                 + '; '.join(f'{nccl_proto_text(nccl, world)} ({label})' for label, _, _, nccl in runs))
    lines.append('[aim347] NCCL logs, with the algorithm and protocol of every collective on rank 0: '
                 + ', '.join(str(Path(out) / variant) for variant, _ in PROTOCOL_RUNS))
    lines += [''] + protocol_chart_lines([(label, rows, choices) for label, rows, choices, _ in runs])
    wrong = [f"{label} at {', '.join(size_text(size) for size, row in sorted(rows.items()) if row[2])}"
             for label, rows, _, _ in runs if any(row[2] for row in rows.values())]
    if wrong:
        lines.append('[aim347] wrong values: ' + '; '.join(wrong))
        return lines, 5
    return lines, 0


def run_protocols(state=None, runner=subprocess.run, env=None):
    """tools/nccl-protocols.sh. Returns the exit code."""
    env = dict(os.environ if env is None else env)
    if not env.get('LLM_RESULTS_DIR'):
        raise UsageError('LLM_RESULTS_DIR is not set; open a new shell so it reads aim347.env')
    if read_running(state):
        print('AIM347: a stage is still running. Run this tool after it finishes; '
              'bash tools/run.sh attach follows it.', file=sys.stderr)
        return 2
    stamp = utc_stamp()
    out, suffix = Path(env['LLM_RESULTS_DIR']) / 'nccl-protocols' / stamp, 1
    while out.exists():
        suffix += 1
        out = out.with_name(f'{stamp}-{suffix}')
    world = 2 * int(env.get('GPUS_PER_NODE', 8))
    print(f'[aim347] all_reduce_perf, 1 MiB to 64 MiB, on {world} GPUs over EFA with OFI_NCCL_PROTOCOL=RDMA')
    print(f'[aim347] results {out}')
    for number, (variant, label) in enumerate(PROTOCOL_RUNS, 1):
        (out / variant).mkdir(parents=True)
        print(f'[aim347] run {number} of {len(PROTOCOL_RUNS)}: {label}', flush=True)
        with (out / variant / 'all_reduce_perf.txt').open('w') as stream:
            code = runner(protocol_command(variant, out, env), stdout=stream, stderr=subprocess.STDOUT).returncode
        if code != 0:
            print(f'[aim347] run {number} ({label}) failed with exit status {code}; '
                  f"last 40 lines of {out / variant / 'all_reduce_perf.txt'}:")
            print('\n'.join(tail(out / variant / 'all_reduce_perf.txt')))
            return 4
    lines, code = protocol_report(out, world)
    print('\n'.join(lines))
    return code


# ---------------------------------------------------------------- command line

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='tools/run.sh')
    run.add_argument('subcommand', nargs='?')
    commands.add_parser('worker').add_argument('invocation')
    commands.add_parser('nccl-protocols', help='tools/nccl-protocols.sh')
    commands.add_parser('efa-read', help='on a GPU node: EFA counters as one JSON line')
    commands.add_parser('node-probe', help='on a GPU node: GPUs and EFA devices as one JSON line')
    report = commands.add_parser('check-report', help='tools/check.sh output')
    report.add_argument('--sinfo', type=Path, required=True)
    report.add_argument('--nodes', type=Path, required=True)
    commands.add_parser('granted', help='job ID from salloc output on stdin')
    allocation = commands.add_parser('write-allocation')
    allocation.add_argument('--job-id', type=int, required=True)
    allocation.add_argument('--cpus-per-node', type=int, required=True)
    allocation.add_argument('--output', type=Path, required=True)
    ceilings = commands.add_parser('ceilings')
    ceilings.add_argument('--dir', type=Path, required=True)
    ceilings.add_argument('--gpus', type=int, required=True)
    verify = commands.add_parser('verify-data')
    verify.add_argument('--data', type=Path, required=True)
    verify.add_argument('--dataset', action='append', choices=list(DATASETS))
    verify.add_argument('--model', action='store_true')
    env = commands.add_parser('write-env')
    for option in ('stage-dir', 'companion', 'slurm-version', 'cpus-per-node', 'instance-type', 'nic',
                   'image-sha256', 'grafana-url', 'password-file'):
        env.add_argument('--' + option, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            if not args.subcommand:
                raise UsageError('usage: bash tools/run.sh <subcommand>; subcommands: '
                                 + ', '.join(list(SUBCOMMANDS) + ['attach']))
            if args.subcommand != 'attach' and args.subcommand not in SUBCOMMANDS:
                raise UsageError(f'unknown subcommand {args.subcommand!r}; subcommands: '
                                 + ', '.join(list(SUBCOMMANDS) + ['attach']))
            return start_run(args.subcommand)
        if args.command == 'worker':
            return worker(json.loads(args.invocation))
        if args.command == 'nccl-protocols':
            return run_protocols()
        if args.command == 'efa-read':
            devices, errors = read_efa_counters()
            for error in errors:
                print(f'efa-read {node_name()}: {error}', file=sys.stderr)
            print(json.dumps(dict(host=node_name(), devices=devices, errors=errors)))
            return 0
        if args.command == 'node-probe':
            print(json.dumps(node_probe()))
            return 0
        if args.command == 'check-report':
            reports = [json.loads(line) for line in args.nodes.read_text().splitlines() if line.startswith('{')]
            lines, failures = check_lines(dict(os.environ), args.sinfo.read_text(), reports)
            print('\n'.join(lines))
            return 1 if failures else 0
        if args.command == 'granted':
            job_id = parse_granted(sys.stdin.read())
            if job_id is None:
                return 3
            print(job_id)
            return 0
        if args.command == 'write-allocation':
            record = parse_scontrol_record(subprocess.run(['scontrol', 'show', 'job', '-o', str(args.job_id)],
                                                          capture_output=True, text=True, check=True).stdout)
            nodes = subprocess.run(['scontrol', 'show', 'hostnames', record['NodeList']], capture_output=True,
                                   text=True, check=True).stdout.split()
            write_allocation(args.output, args.job_id, record['NodeList'], args.cpus_per_node, len(nodes))
            print(f"Session allocation {args.job_id}: {', '.join(nodes)}, {len(nodes) * int(os.environ.get('GPUS_PER_NODE', 8))} "
                  f"GPUs, time limit {time_limit_minutes(record['TimeLimit'])} minutes")
            print('Saved in ~/.aim347/allocation.env. Every tool reads it, also in a new shell.')
            return 0
        if args.command == 'ceilings':
            result = aggregate_ceilings(args.dir)
            if result['gpu_count'] != args.gpus:
                print(f"found results for {result['gpu_count']} GPUs in {args.dir}, expected {args.gpus}", file=sys.stderr)
                return 4
            write_atomic(args.dir / 'ceilings.json', json.dumps(result, indent=2) + '\n')
            print('\n'.join(ceilings_lines(result)))
            return 0
        if args.command == 'verify-data':
            failed = False
            for name in args.dataset or []:
                ok, detail, _ = verify_dataset(args.data, name)
                print(f"{node_name()} {name} {'OK' if ok else 'FAIL'} {detail}")
                failed |= not ok
            if args.model:
                ok, detail = verify_model(args.data)
                print(f"{node_name()} model {'OK' if ok else 'FAIL'} {detail}")
                failed |= not ok
            return 1 if failed else 0
        if args.command == 'write-env':
            write_atomic(Path(args.stage_dir) / 'aim347.env', env_file_text(
                stage_dir=args.stage_dir, companion=args.companion, slurm_version=args.slurm_version,
                cpus_per_node=int(args.cpus_per_node), instance_type=args.instance_type, nic=args.nic,
                image_sha256=args.image_sha256, grafana_url=args.grafana_url, password_file=args.password_file))
            return 0
    except UsageError as error:
        print(error, file=sys.stderr)
        return 2
    return 2


if __name__ == '__main__':
    sys.exit(main())
