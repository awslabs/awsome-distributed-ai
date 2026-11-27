"""The counters of a vLLM server that lib/serving_decode.py reads, and the decode time between two readings."""
import math
import urllib.request

METRICS = ('vllm:generation_tokens_total', 'vllm:request_success_total',
           'vllm:inter_token_latency_seconds_count', 'vllm:inter_token_latency_seconds_sum')


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
