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

from llm_metrics import training_goodput, update_work

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
# EFA devices attached per node: g6e.48xlarge and g7e.48xlarge have four, and the test events attach two.
EFA_DEVICES = {'g7.48xlarge': 2, 'g6e.48xlarge': 2, 'g7e.48xlarge': 2}
# OFI_NCCL_PROTOCOL of the EFA path (2.run-llm.sh, lib/nccl-protocols-rank.sh); RDMA otherwise.
OFI_PROTOCOLS = {'g6e.48xlarge': 'SENDRECV'}
FSX_FREE_BYTES = 300 * 2**30

EFA_COUNTERS = ('tx_bytes', 'rx_bytes', 'rdma_read_bytes', 'rdma_write_bytes', 'rdma_write_recv_bytes')
NETWORK = {'socket': 'Socket', 'efa': 'AWS Libfabric'}
RESTORE_FROM, RESTORE_STOP, INTERRUPT_AFTER = 20, 25, 30

# Optional A: stage try-<change> runs the final configuration with one trainer change,
# and stage rerun-<change> the same configuration without it. `argument` is what the change does to the trainer
# arguments of stage sdpa; `key` and `value` are the summary.json configuration entry that
# shows the change took effect, and `default` the entry without it. `also` lists entries the
# change moves with it (two sequences at once leave one accumulation step per update).
TRY_CHANGES = {
    'microbatch2': dict(argument='--microbatch 2', key='microbatch', value=2, default=1,
                        also=('accumulation_steps',)),
    'fused-optimizer': dict(argument='--fused-optimizer', key='optimizer', value='fused', default='stock', also=()),
    'forward-prefetch': dict(argument='--forward-prefetch', key='forward_prefetch', value=True, default=False,
                             also=()),
    'no-reshard-after-forward': dict(argument='--no-reshard-after-forward', key='reshard_after_forward',
                                     value=False, default=True, also=()),
}
# Stage: transport, attention, saving, kind and config. tools/run.sh takes these names.
STAGES = {
    'baseline': ('socket', 'eager', 'none', 'timed', 'configs/smol-2048.json'),
    'probe-socket-eager': ('socket', 'eager', 'none', 'timed', 'configs/smol-1024.json'),
    'profile-socket-eager': ('socket', 'eager', 'none', 'profile', 'configs/smol-2048.json'),
    'sdpa-socket': ('socket', 'sdpa', 'none', 'timed', 'configs/smol-2048.json'),
    'efa': ('efa', 'eager', 'none', 'timed', 'configs/smol-2048.json'),
    'probe-efa-eager': ('efa', 'eager', 'none', 'timed', 'configs/smol-1024.json'),
    'profile-efa-eager': ('efa', 'eager', 'none', 'profile', 'configs/smol-2048.json'),
    'sdpa': ('efa', 'sdpa', 'none', 'timed', 'configs/smol-2048.json'),
    'profile-efa-sdpa': ('efa', 'sdpa', 'none', 'profile', 'configs/smol-2048.json'),
    'ckpt-sync': ('efa', 'sdpa', 'sync', 'timed', 'configs/smol-2048-40.json'),
    'ckpt-process': ('efa', 'sdpa', 'process', 'timed', 'configs/smol-2048-40.json'),
    'restore': ('efa', 'sdpa', 'sync', 'restore', 'configs/smol-2048-40.json'),
    'interrupt': ('efa', 'sdpa', 'process', 'interrupt', 'configs/smol-2048-40.json'),
}
for _change in TRY_CHANGES:
    STAGES[f'try-{_change}'] = STAGES[f'rerun-{_change}'] = STAGES['sdpa']
PROGRESSION_ORDER = ('baseline', 'probe-socket-eager', 'sdpa-socket', 'efa', 'probe-efa-eager',
                     'sdpa', 'ckpt-sync', 'ckpt-process', 'interrupt')


class UsageError(ValueError):
    """Exit 2: usage, or a setting the tool needs."""


# ---------------------------------------------------------------- stage plans

def config_values(config, lab=LAB):
    return json.loads((Path(lab) / config).read_text())


def dataset_for(config):
    for name, dataset in DATASETS.items():
        if dataset['config'] == config:
            return name
    raise KeyError(config)


def stage_plan(name, lab=LAB):
    """One stage of the STAGES table, as data. Raises UsageError on an unknown name."""
    if name not in STAGES:
        raise UsageError(f'unknown stage {name!r}; stages: ' + ', '.join(STAGES))
    transport, attention, saving, kind, config = STAGES[name]
    prefix, _, rest = name.partition('-')
    values = config_values(config, lab)
    return dict(name=name, kind=kind, transport=transport, attention=attention, saving=saving, config=config,
                data=dataset_for(config), sequence_length=values['sequence_length'], updates=values['updates'],
                warmup_updates=values['warmup_updates'], global_batch=values['global_batch_samples'],
                checkpoint_updates=values['checkpoint_updates'],
                change=rest if prefix in ('try', 'rerun') else None, with_change=prefix == 'try')


def trainer_arguments(stage, label, *, output=True):
    arguments = ['--config', stage['config'], '--data', f"/data/{stage['data']}", '--model-path', '/data/model',
                 '--attention-implementation', stage['attention'], '--microbatch', '1', '--workers', '2',
                 '--checkpoint-mode', stage['saving'], '--allocation-label', label]
    if stage.get('with_change'):
        change = TRY_CHANGES[stage['change']]
        if change['key'] == 'microbatch':
            arguments[arguments.index('--microbatch') + 1] = str(change['value'])
        else:
            arguments.append(change['argument'])
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


# Dense BF16 Tensor Core peak per GPU in TFLOP/s, and whether it is an estimate, keyed by the
# CUDA device name that the result files record (gpu_name). MFU divides by the GPU count
# times this value: PaLM (arXiv 2204.02311), section 4.1 and appendix B, uses the published
# peak matmul throughput. Dense BF16 training cannot use a 2:4 sparse rate.
DENSE_BF16_PEAKS = {
    # g7.48xlarge. NVIDIA's datasheet and product page list "FP16 | BF16 Tensor Core 406 TFLOPS"
    # without saying whether sparsity is included; the Workstation Edition datasheet footnotes the
    # same FP4 figure "with sparsity". Estimate: 1,024 dense FLOP per SM per clock x 82 SMs (10,496
    # CUDA cores / 128) x the 2,415 MHz boost clock (product brief PB-12790-001_02) = 202.8, half of
    # 406. 1,024 is the dense FP16 rate with FP16 accumulation in the RTX Blackwell GPU architecture
    # whitepaper (appendix B, table 4: RTX 5080, 225.1 TFLOPS at 84 SMs and 2,617 MHz). That GeForce
    # table runs BF16 with FP32 accumulation at half of it (112.6, 512 per SM per clock); the
    # estimate assumes that this professional part runs it at the full rate, as NVIDIA lists for the
    # L40S (362.05) against the L40 on the same chip (181, Ada whitepaper, appendix C). The measured
    # GEMM supports it: 167.6 TFLOP/s per GPU, where a half rate would cap at 101.4.
    'NVIDIA RTX PRO 4500 Blackwell Server Edition': (202.8, True),
    # g6e.48xlarge. NVIDIA's L40S page: "BFLOAT16 Tensor Core TFLOPS 362.05 I 733*", "*With Sparsity".
    'NVIDIA L40S': (362.05, False),
    # g7e.48xlarge. Estimated as for the RTX PRO 4500, with the same full-rate assumption: 1,024 x
    # 188 SMs (24,064 CUDA cores in the datasheet / 128) x the 2,430 MHz boost clock (product brief
    # SP-12355-001_v02) = 467.8. A measured BF16 GEMM of 429.1 TFLOP/s on g7e.12xlarge (README.md)
    # exceeds the 233.9 that a half rate would allow.
    # Cross-check: the product page lists "FP16 | BF16 Tensor Core 1 PFLOP" unmarked. Twice 467.8 is
    # 935.6, which that one-digit figure rounds, as its 4 PFLOPS FP4 and 2 PFLOPS FP8 round the sparse
    # 3.74 and 1.87; half of 1 PFLOP, 500, is therefore 7 % above the estimate.
    'NVIDIA RTX PRO 6000 Blackwell Server Edition': (467.8, True),
}


def dense_peak(gpu):
    """(TFLOP/s per GPU, estimated) for a GPU name; ValueError for a GPU the table does not list."""
    if gpu not in DENSE_BF16_PEAKS:
        raise ValueError(f'no dense peak for {gpu}')
    return DENSE_BF16_PEAKS[gpu]


def gpu_name(summary):
    """The one GPU name that every rank of summary.json recorded, else None."""
    names = {rank.get('gpu_name') for rank in summary.get('ranks', [])}
    return names.pop() if len(names) == 1 else None


def flop_share(tokens_per_second, sequence_length, tflops_per_gpu, gpus):
    """The model's FLOP per second over gpus x tflops_per_gpu."""
    return tokens_per_second * flop_per_token(sequence_length) / (gpus * tflops_per_gpu * 1e12)


def mfu(tokens_per_second, sequence_length, gpu, gpus):
    """Model FLOPs utilization: the model's FLOP per second over gpus x the dense BF16 peak of gpu."""
    return flop_share(tokens_per_second, sequence_length, dense_peak(gpu)[0], gpus)


def gemm_share(tokens_per_second, sequence_length, gemm_tflops_per_gpu, gpus):
    """The model's FLOP per second over gpus x your GEMM peak (measured with measure-ceilings.sh). Not MFU."""
    return flop_share(tokens_per_second, sequence_length, gemm_tflops_per_gpu, gpus)


def mfu_text(summary, sequence_length, world):
    """The value of the MFU line of a result block."""
    gpu = gpu_name(summary)
    if gpu is None:
        return 'n/a (summary.json names no single GPU)'
    try:
        peak, estimated = dense_peak(gpu)
    except ValueError as error:
        return f'n/a ({error})'
    value = mfu(summary['training_throughput_tokens_per_second'], sequence_length, gpu, world)
    return (f"{100 * value:.1f} %  ({flop_per_token(sequence_length) / 1e9:.2f} GFLOP per token, "
            f"dense BF16 peak {peak:g} TFLOP/s per GPU{', estimated' if estimated else ''})")


def gemm_share_text(summary, sequence_length, ceilings, world):
    """The value of the line that compares the model's FLOP rate with your measured GEMM peak."""
    if not ceilings:
        return 'n/a (run bash tools/measure-ceilings.sh)'
    rate = ceilings['gemm_tflops_per_gpu']
    value = gemm_share(summary['training_throughput_tokens_per_second'], sequence_length, rate, world)
    return f'{100 * value:.1f} %  (your GEMM peak {rate:.1f} TFLOP/s per GPU)'


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
        return 'GEMM peak n/a (run bash tools/measure-ceilings.sh)'
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
    """{host: None (unavailable) or dict(sent, received, increased); sent or received is None if not read."""
    deltas = {}
    for host in sorted(set(before) | set(after)):
        first, last = before.get(host) or {}, after.get(host) or {}
        changes = {(device, counter): last[device][counter] - value
                   for device, counters in first.items() for counter, value in counters.items()
                   if counter in last.get(device, {})}
        if not changes:
            deltas[host] = None
            continue
        totals = {}
        for name in ('tx_bytes', 'rx_bytes'):
            values = [v for (_, counter), v in changes.items() if counter == name]
            totals[name] = sum(values) if values else None
        deltas[host] = dict(sent=totals['tx_bytes'], received=totals['rx_bytes'],
                            increased=any(v > 0 for v in changes.values()))
    return deltas


def format_bytes(value):
    for unit, size in (('TB', 1e12), ('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if abs(value) >= size:
            return f'{value / size:.1f} {unit}'
    return f'{value} B'


def efa_text(deltas):
    def change(value, direction):
        return f'{direction} bytes not read' if value is None else f'+{format_bytes(value)} {direction}'

    if not deltas:
        return 'EFA counters unavailable on both nodes'
    return '; '.join(f'EFA counters unavailable on {host}' if delta is None else
                     f"{host} {change(delta['sent'], 'sent')}, {change(delta['received'], 'received')}"
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
        if ok and stage.get('change'):
            change = TRY_CHANGES[stage['change']]
            expected = change['value'] if stage['with_change'] else change['default']
            found = summary.get('configuration', {}).get(change['key'])
            checks.append(dict(name='Change', ok=found == expected,
                               detail=f"configuration {change['key']} is {setting_text(found)}"
                                      + ('' if found == expected else f', expected {setting_text(expected)}')))
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
    extra += f", {TRY_CHANGES[stage['change']]['argument']}" if stage.get('with_change') else ''
    return f"{stage['name']} ({stage['transport']}, {stage['attention']} attention, {saving}{extra})"


def setting_text(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def change_lines(stage):
    """The header line of a try stage or its rerun: what differs from stage sdpa."""
    if not stage.get('change'):
        return []
    change = TRY_CHANGES[stage['change']]
    if not stage['with_change']:
        return ['[aim347] the configuration of stage sdpa again, without the change, '
                'for the spread between two runs']
    if change['key'] == 'microbatch':
        what = f"{change['argument']} in place of --microbatch {change['default']}"
    else:
        what = f"{change['argument']} added"
    return [f'[aim347] the configuration of stage sdpa with one change: {what}']


def change_pair(stage, summary):
    """The configuration entries that show whether the change took effect."""
    change = TRY_CHANGES[stage['change']]
    configuration = summary.get('configuration', {})
    return ('setting in summary.json', ', '.join(f'{key} {setting_text(configuration.get(key))}'
                                                 for key in (change['key'],) + change['also']))


def header_lines(stage, job_id, stage_dir):
    used = dict(LLM_TRANSPORT=stage['transport'], ATTENTION=stage['attention'], CHECKPOINT_MODE=stage['saving'])
    lines = [f"[aim347] {stage['name']}: " + ' '.join(f'{k}={v}' for k, v in used.items())]
    lines += change_lines(stage)
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
    lines.append(f'[aim347] batch job {job_id}, results {stage_dir}')
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


def timed_block(stage, stage_dir, summary, ceilings, world, efa_losses=None, results=None):
    if stage['saving'] != 'none':
        return checkpoint_block(stage, stage_dir, summary, world, results)
    seconds = summary['measured_seconds']
    tokens = summary['measured_useful_tokens']
    rate = summary['training_throughput_tokens_per_second']
    first = stage['warmup_updates'] + 1
    lines = [describe(stage)] + lines_for(
        [(f"timed updates {first}-{stage['updates']}", f'{seconds:.2f} s  ({tokens:,} targets)'),
         ('tokens/s', f'{rate:,.0f}'),
         ('GPU-seconds per 1M tokens', f'{gpu_seconds_per_million_tokens(seconds, tokens, world):.0f}'),
         ('MFU', mfu_text(summary, stage['sequence_length'], world)),
         ('share of your GEMM peak', gemm_share_text(summary, stage['sequence_length'], ceilings, world)),
         ('peak memory (largest rank)', f'{peak_memory_gib(summary):.2f} GiB')]
        + ([change_pair(stage, summary)] if stage.get('change') else [])
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


# The intervals of the retained progress, as lib/train_llm.py (segment_seconds) and
# 13.measure-recovery.py (elapsed) measure them.
SAVING_INTERVAL = ('from the end of setup (model loading, FSDP wrapping,',
                   'optimizer, checkpoint manager) to the end of the final flush and',
                   'the closing synchronization across ranks')
CAMPAIGN_INTERVAL = ('from just before the first launch to the exit of the',
                     'resumed launch, with both launches and both model loads')


# The recorded settings in which a run may differ from its fault-free reference, stage sdpa.
REFERENCE_MAY_DIFFER = ('checkpoint_mode',)
ABOVE_ONE = 'above 1: check the reference run'


def retained_updates(record, progress):
    """(updates completed at the start, update of the last durable checkpoint) of the retained tokens.

    Result files written before the retained-progress block held these two numbers record them
    only for a fresh run or a campaign, which start at update 0 and end durable at their last update.
    """
    first, last = progress.get('initial_completed_updates'), progress.get('durable_completed_updates')
    if first is None or last is None:
        if record.get('initial_progress_tokens', 0):
            raise ValueError('no update range of the resumed segment in the result file')
        if 'completed_updates' not in record:
            raise ValueError('no completed_updates in the result file')
        first, last = 0, record['completed_updates']
    if last <= first:
        raise ValueError('no update retained')
    return first, last


def run_problem(record):
    """Why this run's own record gives no goodput, or None."""
    for failed, reason in ((record.get('status') != 'completed', 'this run is not completed'),
                           (record.get('profiled'), 'this run was profiled'),
                           (record.get('numerical_verification'), 'this run ran numerical verification'),
                           (record.get('performance_eligible') is False, 'this run is marked not performance-eligible')):
        if failed:
            return reason
    return None


def reference_differences(record, gpu, reference):
    """What the reference run of stage sdpa recorded differently from this run, beyond the checkpoint mode."""
    config, other = record.get('configuration'), reference.get('configuration')
    if not config or not other:
        return ['configuration (not recorded)']
    differs = sorted(key for key in set(config) | set(other)
                     if key not in REFERENCE_MAY_DIFFER and config.get(key) != other.get(key))
    workload, other_workload = record.get('workload', {}), reference.get('workload', {})
    differs += [key for key in ('model_revision', 'precision') if workload.get(key) != other_workload.get(key)]
    label = record.get('allocation', {}).get('label')
    if not label:
        differs.append('allocation label (not recorded)')
    elif label != reference.get('allocation', {}).get('label'):
        differs.append('allocation label')
    if gpu is None:
        differs.append('GPU (not recorded)')
    elif gpu != gpu_name(reference):
        differs.append('GPU')
    return differs


def stage_goodput(results, record, progress, gpu):
    """Training goodput of a saving stage (record: its summary.json) or of interrupt (its campaign.json).

    gpu is the GPU name of the run. The fault-free rate is the throughput of stage sdpa in the
    same results directory, which does the same work per update with the same configuration
    and allocation, without saves. Raises ValueError, with the reason, when there is none.
    """
    problem = run_problem(record)
    if problem:
        raise ValueError(problem)
    run, reference, problem = stage_record(results, 'sdpa')
    if problem:
        raise ValueError('no completed run of stage sdpa')
    if run.get('exit_status') != 0:
        raise ValueError('stage sdpa had a failed check' if run.get('exit_status') == 5
                         else f"stage sdpa ended with exit status {run.get('exit_status')}")
    differs = reference_differences(record, gpu, reference)
    if differs:
        raise ValueError('stage sdpa ran with a different ' + ', '.join(differs))
    first, last = retained_updates(record, progress)
    work = update_work(record.get('workload', {}), progress['retained_progress_tokens'], last - first)
    return training_goodput(progress['retained_tokens_per_second'], work, reference)


def goodput_pairs(record, progress, interval, results, gpu):
    """The block lines from the retained tokens to the training goodput."""
    if not progress or 'retained_tokens_per_second' not in progress:
        return [('training goodput', 'not computed: no retained progress in the result file')]
    try:
        first, last = retained_updates(record, progress)
    except ValueError as error:
        return [('training goodput', f'not computed: {error}')]
    pairs = [('retained tokens', f"{progress['retained_progress_tokens']:,} "
                                 f"(updates {first + 1}-{last}, durable at the end)"),
             ('interval', f"{progress['elapsed_seconds']:.2f} s, {interval[0]}")]
    pairs += [('', part) for part in interval[1:]]
    pairs.append(('retained progress rate',
                  f"{progress['retained_tokens_per_second']:,.0f} tokens/s (retained tokens / interval)"))
    try:
        goodput = stage_goodput(results, record, progress, gpu)
    except ValueError as error:
        return pairs + [('training goodput', f'not computed: {error}')]
    value = goodput['training_goodput']
    return pairs + [('fault-free rate', f"{goodput['fault_free_tokens_per_second']:,.0f} tokens/s "
                                        '(stage sdpa, which does not save)'),
                    ('training goodput', f'{value:.2f} (retained progress rate / fault-free rate)'
                                         + (f', {ABOVE_ONE}' if value > 1 else ''))]


def checkpoint_block(stage, stage_dir, summary, world, results):
    first = stage['warmup_updates'] + 1
    return [describe(stage)] + lines_for(
        [(f"timed updates {first}-{stage['updates']} with saves and final flush",
          f"{summary['measured_seconds']:.2f} s  ({summary['measured_useful_tokens']:,} targets)")]
        + goodput_pairs(summary, summary.get('retained_progress'), SAVING_INTERVAL, results, gpu_name(summary))
        + nccl_pairs(stage, stage_dir, world) + [efa_pair(stage_dir)])


def campaign_gpu(campaign_dir):
    """The GPU name of a campaign: campaign.json records none; its resumed run's summary.json does."""
    return gpu_name(read_json(Path(campaign_dir) / 'resumed' / 'summary.json', {}))


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


def interrupt_block(stage, stage_dir, world, results):
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
    lines += lines_for(goodput_pairs(campaign, campaign, CAMPAIGN_INTERVAL, results, campaign_gpu(campaign_dir)))
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
    """What the batch job prints when a stage has finished, by exit status."""
    stage_dir = Path(stage_dir)
    code = run.get('exit_status')
    if code == 4:
        return failure_lines(stage, stage_dir, run.get('failure') or 'unknown')
    if stage['kind'] == 'restore':
        lines = restore_block(stage, stage_dir, read_rows(Path(results) / 'ckpt-sync' / 'train' / 'rank-0-updates.jsonl'), world)
    elif stage['kind'] == 'interrupt':
        lines = interrupt_block(stage, stage_dir, world, results)
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
            lines = timed_block(stage, stage_dir, summary, read_ceilings(results), world, efa_rows, results)
            if stage.get('change') and not stage['with_change']:
                lines += [''] + try_comparison_lines(stage['change'], results)
    for check in run.get('checks', []):
        if not check['ok']:
            lines.append(f"check failed: {check['name']}: {check['detail']}")
    if code == 5:
        lines.append('The run finished, but a check failed. Fix it before reading the result.')
    return lines


# ---------------------------------------------------------------- comparisons

def stage_record(results, name, record=('train', 'summary.json')):
    """(run, record, None) for a stage that has a result, else (run or None, None, the reason).

    The record is the stage's summary.json unless another path under the stage is given."""
    run = read_json(Path(results) / name / 'run.json')
    if not run:
        return None, None, 'not run yet'
    if run.get('status') != 'finished':
        return run, None, 'still running'
    if run.get('exit_status') == 4:
        return run, None, 'failed; run it again'
    summary = read_json(Path(results, name, *record))
    if not summary:
        return run, None, f'no {record[-1]}'
    return run, summary, None


def failed_check(run):
    return ', a check failed' if run.get('exit_status') == 5 else ''


def try_comparison_lines(change, results):
    """After rerun-<change>: the spread between the two unchanged runs, and the change against their mean.

    The two unchanged runs are stage sdpa and rerun-<change>. A run that is missing, unfinished
    or failed, or whose check failed, gives no comparison.
    """
    seconds = []
    for name in ('sdpa', f'rerun-{change}', f'try-{change}'):
        run, summary, problem = stage_record(results, name)
        if problem:
            return [f'not compared: {name} {problem}']
        if failed_check(run):
            return [f'not compared: a check of {name} failed']
        seconds.append(summary['measured_seconds'])
    sdpa, rerun, tried = seconds
    mean = (sdpa + rerun) / 2
    spread = abs(sdpa - rerun)
    delta = tried - mean
    return [f'run-to-run spread: |sdpa - rerun-{change}| = {spread:.3f} s ({100 * spread / mean:.2f} %)',
            f'change: try-{change} - mean(sdpa, rerun-{change}) = {delta:+.3f} s ({100 * delta / mean:+.2f} %)']


# ---------------------------------------------------------------- allocation

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
              ('AIM347_ALLOCATION_LABEL', f'pcs-{instance_type}-2nodes-16gpu'), ('AIM347_GRAFANA_URL', grafana_url),
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


# ---------------------------------------------------------------- the batch job of run.sh

def results_dir():
    value = os.environ.get('LLM_RESULTS_DIR')
    if not value:
        raise UsageError('LLM_RESULTS_DIR is not set; open a new shell so it reads aim347.env')
    return Path(value)


def world_size():
    return 2 * int(os.environ.get('GPUS_PER_NODE', 8))


def utc_stamp():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


def rows_sources(stage, stage_dir):
    if stage['kind'] == 'interrupt':
        return [stage_dir / 'campaign' / 'interrupted' / 'rank-0-updates.jsonl',
                stage_dir / 'campaign' / 'resumed' / 'rank-0-updates.jsonl']
    return [stage_dir / 'train' / 'rank-0-updates.jsonl']


def run_stage(stage, results, *, popen=subprocess.Popen, efa_reader=None, out=print, interval=1.0):
    """One stage inside the batch job: header, launch, rows of rank 0 as they appear, checks, run.json, result.

    Returns 0, 4 when the run failed, or 5 when it finished and a check failed.
    """
    results = Path(results)
    world = world_size()
    job_id = os.environ.get('SLURM_JOB_ID')
    stage_dir = results / stage['name']
    if stage_dir.exists():
        attempts = results / 'attempts'
        attempts.mkdir(exist_ok=True)
        stage_dir.rename(attempts / f"{stage['name']}-{utc_stamp()}")
    stage_dir.mkdir(parents=True)
    command = launch_command(stage, os.environ.get('AIM347_ALLOCATION_LABEL', 'aim347'), results, world)
    run = dict(stage=stage['name'], plan=stage,
               settings=dict(transport=stage['transport'], attention=stage['attention'], saving=stage['saving']),
               trainer_argv=command, config=stage['config'], data=stage['data'], allocation_id=job_id,
               start_utc=time.time(), status='running')
    write_atomic(stage_dir / 'run.json', json.dumps(run, indent=2) + '\n')
    for line in header_lines(stage, job_id, stage_dir):
        out(line)
    saves = stage['checkpoint_updates'] if stage['saving'] != 'none' and stage['kind'] != 'restore' else None
    sources = rows_sources(stage, stage_dir)
    printer = RowPrinter(stage['warmup_updates'], saves)
    printed, segment, header_done = 0, 0, False
    efa_reader = efa_reader or read_efa_on_nodes
    with (stage_dir / 'launch.log').open('a') as log:
        write_atomic(stage_dir / 'efa-before.json', json.dumps(efa_reader(log), indent=2) + '\n')
        log.flush()
        process = popen(command, cwd=str(LAB), env=dict(os.environ, LLM_TRANSPORT=stage['transport']),
                        stdout=log, stderr=subprocess.STDOUT)
        while True:
            # Poll before reading, so that the last read sees every row of a finished launch.
            finished = process.poll() is not None
            rows = read_rows(sources[segment])
            if rows and not header_done:
                out(printer.header())
                header_done = True
            for row in rows[printed:]:
                out(printer.line(row))
            printed = len(rows)
            if segment + 1 < len(sources) and sources[segment + 1].exists():
                segment, printed = segment + 1, 0
                printer = RowPrinter(stage['warmup_updates'], saves)
                out(f'[aim347] stopped after update {INTERRUPT_AFTER}; resumed from the last completed checkpoint')
                continue
            if finished:
                break
            time.sleep(interval)
        write_atomic(stage_dir / 'efa-after.json', json.dumps(efa_reader(log), indent=2) + '\n')
    code, checks, failure = post_run_checks(stage, stage_dir, process.returncode, world)
    run.update(end_utc=time.time(), launcher_exit=process.returncode, exit_status=code, checks=checks,
               failure=failure, status='finished')
    write_atomic(stage_dir / 'run.json', json.dumps(run, indent=2) + '\n')
    if saves is not None and stage['kind'] != 'interrupt' and code != 4 and rows:
        events = read_rows(stage_dir / 'train' / 'checkpoint-events-rank-0.jsonl')
        flush = flush_seconds(rows, events, stage['updates'])
        if flush is not None:
            out(RowPrinter.flush_line(flush, stage['updates']))
    out(f"[aim347] finished in {run['end_utc'] - run['start_utc']:.0f} s")
    for line in result_lines(stage, stage_dir, run, results, world):
        out(line)
    return code


def read_efa_on_nodes(log):
    """One srun step on both nodes, outside the image; errors go to the stage log."""
    result = subprocess.run(['srun', '--nodes=2', '--ntasks-per-node=1', '--cpus-per-task=1', 'python3',
                             str(LAB / 'lib' / 'workshop.py'), 'efa-read'],
                            stdout=subprocess.PIPE, stderr=log, text=True)
    return merge_efa_readings(result.stdout.splitlines())


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
    wrong = [f"{label} at {', '.join(size_text(size) for size, row in sorted(rows.items()) if row[2])}"
             for label, rows, _, _ in runs if any(row[2] for row in rows.values())]
    if wrong:
        lines.append('[aim347] wrong values: ' + '; '.join(wrong))
        return lines, 5
    return lines, 0


def run_protocols(runner=subprocess.run, env=None):
    """tools/nccl-protocols.sh. Returns the exit code."""
    env = dict(os.environ if env is None else env)
    if not env.get('LLM_RESULTS_DIR'):
        raise UsageError('LLM_RESULTS_DIR is not set; open a new shell so it reads aim347.env')
    stamp = utc_stamp()
    out, suffix = Path(env['LLM_RESULTS_DIR']) / 'nccl-protocols' / stamp, 1
    while out.exists():
        suffix += 1
        out = out.with_name(f'{stamp}-{suffix}')
    world = 2 * int(env.get('GPUS_PER_NODE', 8))
    protocol = OFI_PROTOCOLS.get(env.get('AIM347_INSTANCE_TYPE'), 'RDMA')
    print(f'[aim347] all_reduce_perf, 1 MiB to 64 MiB, on {world} GPUs over EFA with OFI_NCCL_PROTOCOL={protocol}')
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
    run = commands.add_parser('run', help='inside the batch job of tools/run.sh: the stages, in order')
    run.add_argument('stages', nargs='*')
    run.add_argument('--list', action='store_true', help='check the names and print them, without running')
    commands.add_parser('nccl-protocols', help='tools/nccl-protocols.sh')
    commands.add_parser('efa-read', help='on a GPU node: EFA counters as one JSON line')
    commands.add_parser('node-probe', help='on a GPU node: GPUs and EFA devices as one JSON line')
    report = commands.add_parser('check-report', help='tools/check.sh output')
    report.add_argument('--sinfo', type=Path, required=True)
    report.add_argument('--nodes', type=Path, required=True)
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
            if not args.stages:
                raise UsageError('usage: bash tools/run.sh <stage> [<stage> ...]; stages: ' + ', '.join(STAGES))
            stages = [stage_plan(name) for name in args.stages]
            if args.list:
                print(' '.join(args.stages))
                return 0
            codes = []
            for stage in stages:
                codes.append(run_stage(stage, results_dir()))
                if codes[-1] == 4:
                    break
            return 4 if 4 in codes else max(codes)
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
