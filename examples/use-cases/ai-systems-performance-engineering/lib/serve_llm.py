"""Start independent vLLM replicas on disjoint GPUs of one Slurm-assigned node."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import re


def allocation_cpus(value, node_count):
    """Require a homogeneous one-task-per-node allocation."""
    counts = []
    for group in value.split(','):
        match = re.fullmatch(r'(\d+)(?:\(x(\d+)\))?', group)
        if not match:
            raise ValueError('invalid SLURM_JOB_CPUS_PER_NODE')
        counts.extend([int(match[1])] * int(match[2] or 1))
    if len(counts) != node_count or len(set(counts)) != 1 or counts[0] < 1:
        raise ValueError('requires homogeneous allocated CPUs on every node')
    return counts[0]


def collect_manifest(output, *, addresses, gpus_per_node, tensor_parallel_size,
                     max_num_seqs, max_num_batched_tokens, port_base, cpus_per_node, cfg,
                     model_path='/media/model', expected_job=None, expected_nodes=None):
    """Collect launcher records after readiness; no GPU or vLLM import needed."""
    if not addresses or len(set(addresses)) != len(addresses) or cpus_per_node < 1:
        raise ValueError('unique assigned node addresses and positive CPU count required')
    if expected_job is not None and (expected_nodes is None or len(expected_nodes) != len(addresses)):
        raise ValueError('allocation node names required with expected job identity')
    if min(gpus_per_node, tensor_parallel_size, max_num_seqs, max_num_batched_tokens) < 1 or gpus_per_node % tensor_parallel_size:
        raise ValueError('TP must divide the assigned GPU count')
    replicas_per_node = gpus_per_node // tensor_parallel_size
    expected = {output / f'node-{node}' / f'replica-{replica}.json'
                for node in range(len(addresses)) for replica in range(replicas_per_node)}
    if set(output.glob('node-*/replica-*.json')) != expected:
        raise ValueError('replica records do not match the assigned placement')
    uuids, endpoints = [], []
    for node, address in enumerate(addresses):
        for replica in range(replicas_per_node):
            record = json.loads((output / f'node-{node}' / f'replica-{replica}.json').read_text())
            if expected_job is not None and (record.get('slurm_job_id') != expected_job
                                             or expected_nodes is None
                                             or record.get('slurm_node_name') != expected_nodes[node]):
                raise ValueError('replica record belongs to a different allocation or node')
            command = record['command']
            checks = {'--tensor-parallel-size': str(tensor_parallel_size),
                      '--max-num-seqs': str(max_num_seqs),
                      '--max-num-batched-tokens': str(max_num_batched_tokens),
                      '--port': str(port_base + replica), '--dtype': 'bfloat16',
                      '--max-model-len': '4096', '--gpu-memory-utilization': '0.85',
                      '--generation-config': 'vllm', '--served-model-name': 'aim347', '--host': '0.0.0.0'}
            expected_tokens = ['vllm', 'serve', model_path, '--no-enable-prefix-caching']
            for flag, value in checks.items():
                expected_tokens.extend([flag, value])
            if command[:3] != expected_tokens[:3] or Counter(command) != Counter(expected_tokens):
                raise ValueError('unsupported replica command structure')
            for flag, value in checks.items():
                if (command.count(flag) != 1 or command.index(flag) + 1 >= len(command)
                        or command[command.index(flag) + 1] != value):
                    raise ValueError(f'replica command mismatch: {flag}')
            if '--no-enable-prefix-caching' not in command or '--enable-prefix-caching' in command:
                raise ValueError('replica prefix-cache policy mismatch')
            if record['vllm_version'] != '0.20.2' or record['model_pin'] != {
                    key: cfg[key] for key in ('model_id', 'model_revision')}:
                raise ValueError('replica model or engine pin mismatch')
            if len(record['gpu_uuids']) != tensor_parallel_size:
                raise ValueError('replica GPU count mismatch')
            uuids.extend(record['gpu_uuids'])
            endpoints.append(f'http://{address}:{port_base + replica}')
    if any(not isinstance(uuid, str) or not uuid for uuid in uuids) or len(set(uuids)) != len(uuids):
        raise ValueError('assigned GPU UUIDs must be nonempty and unique')
    manifest: dict = dict(endpoints=endpoints, gpu_budget=len(uuids),
                    placement=dict(tensor_parallel_size=tensor_parallel_size, replicas=len(endpoints)),
                    server_batch=dict(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens),
                    workload={key: cfg[key] for key in ('model_id', 'model_revision', 'tokenizer_revision')},
                    allocation=dict(gpu_uuids=sorted(uuids), nodes=len(addresses), cpus_per_node=cpus_per_node))
    manifest['workload'].update(precision='bfloat16', vllm_version='0.20.2', prefix_cache=False,
                                max_model_len=4096, gpu_memory_utilization=0.85)
    if any((output / name).exists() for name in ('manifest.json', 'endpoints.txt')):
        raise ValueError('manifest/endpoints already exist; preserve the earlier placement')
    created = []
    try:
        for name, text in [('manifest.json', json.dumps(manifest, indent=2)),
                           ('endpoints.txt', '\n'.join(endpoints) + '\n')]:
            with (output / name).open('x') as stream:
                created.append(output / name)
                stream.write(text)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return manifest


def stop_replicas(processes):
    """Stop every process group created by this launcher, even exited leaders."""
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tensor-parallel-size', type=int, required=True)
    parser.add_argument('--gpus-per-node', type=int, required=True)
    parser.add_argument('--port-base', type=int, default=8100)
    parser.add_argument('--max-num-seqs', type=int, default=256)
    parser.add_argument('--max-num-batched-tokens', type=int, default=8192)
    parser.add_argument('--model', default='/media/model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collect-manifest', action='store_true',
                        help='collect existing launcher records on the allocation shell instead of starting replicas')
    args = parser.parse_args()
    if min(args.tensor_parallel_size, args.gpus_per_node, args.max_num_seqs,
           args.max_num_batched_tokens) < 1 or args.gpus_per_node % args.tensor_parallel_size:
        parser.error('TP must divide the assigned GPU count')
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/llm.json').read_text())
    if args.collect_manifest:
        nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], text=True).splitlines()
        addresses = []
        for node in nodes:
            fields = subprocess.check_output(['scontrol', 'show', 'node', node, '-o'], text=True).split()
            addresses.append(next(field.split('=', 1)[1] for field in fields if field.startswith('NodeAddr=')))
        result = collect_manifest(args.output, addresses=addresses, gpus_per_node=args.gpus_per_node,
                                  tensor_parallel_size=args.tensor_parallel_size, max_num_seqs=args.max_num_seqs,
                                  max_num_batched_tokens=args.max_num_batched_tokens, port_base=args.port_base,
                                  cpus_per_node=allocation_cpus(os.environ['SLURM_JOB_CPUS_PER_NODE'], len(nodes)), cfg=cfg,
                                  model_path=args.model, expected_job=os.environ['SLURM_JOB_ID'], expected_nodes=nodes)
        print(json.dumps(result, indent=2))
        return
    import vllm
    import torch
    if vllm.__version__ != '0.20.2':
        raise ValueError('requires the pinned vLLM 0.20.2 image')
    if torch.cuda.device_count() != args.gpus_per_node:
        raise ValueError('visible GPUs do not match the declared Slurm allocation')
    assigned = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = assigned.split(',') if assigned else [str(i) for i in range(args.gpus_per_node)]
    if len(devices) != args.gpus_per_node:
        raise ValueError('CUDA_VISIBLE_DEVICES differs from the GPU budget')
    model_pin = json.loads((Path(args.model) / 'aim347-pin.json').read_text())
    if model_pin != {key: cfg[key] for key in ('model_id', 'model_revision')}:
        raise ValueError('serving model snapshot pin mismatch')
    gpu_uuids = [str(torch.cuda.get_device_properties(i).uuid) for i in range(args.gpus_per_node)]
    args.output.mkdir(parents=True, exist_ok=False)
    processes = []
    logs = []
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for replica in range(args.gpus_per_node // args.tensor_parallel_size):
            gpu_ids = devices[replica * args.tensor_parallel_size:(replica + 1) * args.tensor_parallel_size]
            command = ['vllm', 'serve', args.model, '--served-model-name', 'aim347',
                       '--tensor-parallel-size', str(args.tensor_parallel_size),
                       '--dtype', 'bfloat16', '--max-model-len', '4096',
                       '--max-num-seqs', str(args.max_num_seqs),
                       '--max-num-batched-tokens', str(args.max_num_batched_tokens),
                       '--gpu-memory-utilization', '0.85', '--no-enable-prefix-caching',
                       '--generation-config', 'vllm', '--host', '0.0.0.0',
                       '--port', str(args.port_base + replica)]
            log = (args.output / f'replica-{replica}.log').open('w')
            logs.append(log)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpu_ids))
            process = subprocess.Popen(command, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            (args.output / f'replica-{replica}.json').write_text(json.dumps(
                {'pid': process.pid, 'command': command, 'assigned_devices': gpu_ids,
                 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'slurm_node_name': os.environ['SLURMD_NODENAME'],
                 'gpu_uuids': gpu_uuids[replica * args.tensor_parallel_size:(replica + 1) * args.tensor_parallel_size],
                 'model_pin': model_pin, 'vllm_version': vllm.__version__, 'started_utc_seconds': time.time()}, indent=2) + '\n')
        while not stopped:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('a replica exited; inspect its retained log')
            time.sleep(0.5)
    finally:
        stop_replicas(processes)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
