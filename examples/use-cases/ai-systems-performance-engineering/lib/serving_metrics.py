"""Weights-only decode traffic model and observed vLLM metric arithmetic."""
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request

METRICS = ('vllm:generation_tokens_total', 'vllm:request_success_total',
           'vllm:inter_token_latency_seconds_count', 'vllm:inter_token_latency_seconds_sum')


def model_bytes(directory, tensor_parallel):
    directory = Path(directory)
    raw = (directory/'config.json').read_bytes()
    config = json.loads(raw)
    if config.get('model_type') != 'qwen2' or config.get('quantization_config') or not config.get('tie_word_embeddings'):
        raise ValueError('This byte model supports the pinned unquantized, tied-embedding Qwen2 model')
    heads = config['num_key_value_heads']
    if tensor_parallel < 1 or config['num_attention_heads'] % tensor_parallel or (heads >= tensor_parallel and heads % tensor_parallel) or (tensor_parallel > heads and tensor_parallel % heads):
        raise ValueError('Unsupported tensor-parallel head partition')
    tensors = {}
    for path in sorted(directory.glob('*.safetensors')):
        with path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if size > 100_000_000:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(f.read(size))
        for name, entry in header.items():
            if name == '__metadata__':
                continue
            if name in tensors or entry['dtype'] != 'BF16':
                raise ValueError('Duplicate tensor or non-BF16 checkpoint')
            count = math.prod(entry['shape'])
            if count < 1 or entry['data_offsets'][0] < 0 or entry['data_offsets'][1]-entry['data_offsets'][0] != count*2 or 8+size+entry['data_offsets'][1] > path.stat().st_size:
                raise ValueError('Checkpoint tensor size mismatch')
            # Qwen2's norm parameters are replicated across TP ranks; K/V
            # projection heads are replicated when TP exceeds the KV-head count.
            copies = tensor_parallel if 'layernorm.' in name or name == 'model.norm.weight' else 1
            if '.k_proj.' in name or '.v_proj.' in name:
                copies = max(1, tensor_parallel//heads)
            tensors[name] = dict(shape=entry['shape'], checkpoint_bytes=count*2,
                                 copies=copies, logical_read_bytes=count*2*copies, source_file=path.name)
    if not tensors or 'model.embed_tokens.weight' not in tensors or 'lm_head.weight' in tensors:
        raise ValueError('Expected the pinned tied-weight checkpoint layout')
    # The tied embedding matrix is read by the output projection each step.
    # Input lookup traffic, KV state, activations, TP communication and allocator
    # padding are outside this weights-only logical traffic model.
    return dict(model_config=config, config_sha256=hashlib.sha256(raw).hexdigest(),
                tensor_parallel_size=tensor_parallel, dtype='bfloat16', bytes_per_element=2,
                checkpoint_weight_bytes=sum(x['checkpoint_bytes'] for x in tensors.values()),
                weights_bytes_per_decode_step=sum(x['logical_read_bytes'] for x in tensors.values()),
                tensors=tensors, scope='logical weights-only reads, including TP norm and KV-head parameter replication; excludes KV-cache traffic, activations, input lookups, padding and physical cache effects')


def snapshot(endpoint):
    with urllib.request.urlopen(endpoint.rstrip('/')+'/metrics', timeout=30) as response:
        text = response.read().decode()
    values = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split()[0]
        if name in METRICS:
            value = float(line.split()[-1])
            if not math.isfinite(value):
                raise ValueError('Nonfinite serving metric')
            values[name] = values.get(name, 0)+value
    if set(values) != set(METRICS):
        raise ValueError('Required vLLM counters or ITL histogram are absent')
    return values


def decode_record(before, after, rows):
    delta = {key: after[key]-before[key] for key in METRICS}
    tokens = sum(row['output_tokens'] for row in rows)
    steps = tokens-len(rows)
    if (delta[METRICS[0]] != tokens or delta[METRICS[1]] != len(rows)
            or delta[METRICS[2]] != steps or steps < 1 or delta[METRICS[3]] <= 0):
        raise ValueError('Server counters do not match serialized single-token decoding; exclude concurrent traffic, speculative decoding and counter resets')
    return dict(decode_steps=steps, server_decode_seconds=delta[METRICS[3]],
                server_mean_itl_seconds=delta[METRICS[3]]/steps, metric_deltas=delta)


def compute_mbu(serving, weights, bandwidth):
    if serving.get('concurrency') != 1 or not serving.get('decode_measurements'):
        raise ValueError('MBU requires the serialized server-metric measurement mode')
    if not math.isfinite(weights['weights_bytes_per_decode_step']) or weights['weights_bytes_per_decode_step'] <= 0:
        raise ValueError('Weights bytes per decode step must be finite and positive')
    modeled = 0
    capacity = 0
    evidence = {}
    for endpoint, decode in serving['decode_measurements'].items():
        if any(not math.isfinite(decode[key]) or decode[key] <= 0 for key in ['decode_steps', 'server_decode_seconds']):
            raise ValueError('Decode counts and durations must be finite and positive')
        samples = bandwidth[endpoint]
        if len(samples) != weights['tensor_parallel_size'] or len({x['gpu_uuid_hex'] for x in samples}) != len(samples):
            raise ValueError('Supply one distinct measured GPU per tensor-parallel rank')
        for sample in samples:
            if (sample['correctness'] != 'passed' or sample['method'] != 'streaming_read_only_cg'
                    or sample['instance_type'] != serving['instance_type'] or not math.isfinite(sample['median_bytes_per_second']) or sample['median_bytes_per_second'] <= 0):
                raise ValueError('Bandwidth provenance does not match the serving measurement')
        rate = sum(sample['median_bytes_per_second'] for sample in samples)
        numerator = weights['weights_bytes_per_decode_step']*decode['decode_steps']
        denominator = rate*decode['server_decode_seconds']
        modeled += numerator
        capacity += denominator
        evidence[endpoint] = dict(modeled_weight_read_bytes=numerator, measured_dram_bytes_per_second=rate,
                                 server_decode_seconds=decode['server_decode_seconds'], mbu_weights_only_ratio=numerator/denominator,
                                 gpu_measurements=samples)
    return dict(serving, mbu_weights_only_ratio=modeled/capacity, modeled_weight_read_bytes=modeled,
                measured_decode_capacity_bytes=capacity, mbu_by_endpoint=evidence, byte_model=weights,
                mbu_scope='modeled weights-only traffic divided by measured read bandwidth over active server decode intervals; not a physical DRAM byte-counter utilization or whole-session utilization')
