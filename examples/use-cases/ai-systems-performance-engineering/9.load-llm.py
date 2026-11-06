#!/usr/bin/env python3
"""Finite closed-loop multi-endpoint serving with quality and latency goodput."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from serving_goodput import serving_goodput, stream_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('endpoints', nargs='+')
    parser.add_argument('--requests-file', type=Path, required=True)
    parser.add_argument('--server-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--ttft-slo-seconds', type=float, required=True)
    parser.add_argument('--tpot-slo-seconds', type=float, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=120)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--warmup-per-endpoint', type=int, default=2)
    args = parser.parse_args()
    if min(args.concurrency, args.max_tokens, args.timeout_seconds,
           args.ttft_slo_seconds, args.tpot_slo_seconds) <= 0 or args.warmup_per_endpoint < 0:
        parser.error('positive counts and latency limits required')
    if args.output.exists():
        parser.error('output already exists; retain historical measurements')
    raw = args.requests_file.read_bytes()
    specs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not specs or any(set(spec) != {'prompt', 'expected_json'} for spec in specs):
        parser.error('each request must contain prompt and expected_json')
    manifest = json.loads(args.server_manifest.read_text())
    if manifest['placement']['replicas'] != len(args.endpoints):
        parser.error('one endpoint per declared replica required')
    if manifest['placement']['replicas'] * manifest['placement']['tensor_parallel_size'] != manifest['gpu_budget']:
        parser.error('placement does not consume the declared GPU budget')
    allocation = manifest.get('allocation')
    if allocation is not None:
        uuids = allocation.get('gpu_uuids', [])
        if (len(uuids) != manifest['gpu_budget'] or len(set(uuids)) != len(uuids)
                or any(not isinstance(value, str) or not value for value in uuids)):
            parser.error('recorded allocation must identify every assigned GPU exactly once')
    evidence_kind = 'LLM serving hardware run' if allocation else 'Serving allocation unverified'
    # Warm each endpoint symmetrically, outside measurement; record all outcomes.
    warmup = [stream_request(args.endpoints, i, specs[i % len(specs)], model='aim347',
                             max_tokens=args.max_tokens, timeout_seconds=args.timeout_seconds)
              for i in range(len(args.endpoints) * args.warmup_per_endpoint)]
    def server_metrics():
        snapshots = []
        for endpoint in args.endpoints:
            try:
                with urllib.request.urlopen(endpoint.rstrip('/') + '/metrics', timeout=10) as response:
                    metrics = response.read().decode()
                snapshots.append(dict(endpoint=endpoint, prometheus=metrics, captured_utc_seconds=time.time()))
            except (OSError, ValueError) as error:
                snapshots.append(dict(endpoint=endpoint, error=f'{type(error).__name__}: {error}'))
        return snapshots

    # Keep server ITL histograms separate from client receive-time TPOT. These
    # snapshots bracket the measured workload, after symmetric warmup.
    metrics_before = server_metrics()
    measured_start_utc = time.time()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        rows = list(pool.map(lambda item: stream_request(
            args.endpoints, item[0], item[1], model='aim347', max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds), enumerate(specs)))
    seconds = time.perf_counter() - start
    measured_finish_utc = time.time()
    metrics_after = server_metrics()
    result = dict(manifest, status='completed', evidence_kind=evidence_kind,
                  measurement_started_utc_seconds=measured_start_utc,
                  measurement_finished_utc_seconds=measured_finish_utc,
                  server_metrics_before=metrics_before, server_metrics_after=metrics_after, client_load={'kind': 'closed_loop', 'concurrency': args.concurrency},
                  workload=dict(manifest['workload'], requests_sha256=hashlib.sha256(raw).hexdigest(),
                                max_tokens=args.max_tokens, temperature=0, seed=347,
                                enable_thinking=False, warmup_per_endpoint=args.warmup_per_endpoint,
                                timeout_seconds=args.timeout_seconds,
                                ttft_slo_seconds=args.ttft_slo_seconds, tpot_slo_seconds=args.tpot_slo_seconds,
                                quality_policy='exact JSON equality'),
                  metrics=serving_goodput(rows, seconds, ttft_slo_seconds=args.ttft_slo_seconds,
                                          tpot_slo_seconds=args.tpot_slo_seconds),
                  requests=rows, warmup=warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
