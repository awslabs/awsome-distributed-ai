"""Arithmetic shared by training, the CLI and unit tests. Ratios are dimensionless."""
import json
import math
import urllib.request


def calculate(tokens_per_second, nonembedding_parameters, gpus, dense_tflops):
    if not all(math.isfinite(x) and x > 0 for x in (tokens_per_second, nonembedding_parameters, gpus, dense_tflops)):
        raise ValueError('throughput, parameters, GPU count and dense peak must be positive and finite')
    achieved = 6 * nonembedding_parameters * tokens_per_second / gpus / 1e12
    return {'tokens_per_second': tokens_per_second, 'tflops_per_gpu': achieved,
            'mfu_dense_gemm_ratio': achieved / dense_tflops}


def push(url, run_id, config, instance_type, values):
    if not url:
        return
    # Job names and identifiers are constrained by the launch scripts.
    from urllib.parse import quote
    labels = f'instance_type={json.dumps(instance_type)}'
    body = ''.join(f'# TYPE aim347_{k} gauge\naim347_{k}{{{labels}}} {v}\n' for k, v in values.items())
    req = urllib.request.Request(url.rstrip('/') + '/metrics/job/aim347/run_id/' + quote(run_id, safe='') + '/config/' + quote(config, safe=''),
                                 data=body.encode(), method='PUT', headers={'Content-Type': 'text/plain; version=0.0.4'})
    with urllib.request.urlopen(req, timeout=2) as response:
        response.read()
