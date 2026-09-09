#!/usr/bin/env python3
"""Stream completions from both replicas and retain observed request latencies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import urllib.request
from lib.serving_metrics import snapshot, decode_record


def request(endpoint, max_tokens):
    payload=dict(model='aim347',prompt='Explain why dense matrix multiplication needs a dense FLOPS denominator.',
                 max_tokens=max_tokens,temperature=0,stream=True,stream_options={'include_usage':True})
    req=urllib.request.Request(endpoint.rstrip('/')+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter(); first=None; last=None; intervals=[]; tokens=None; done=False
    with urllib.request.urlopen(req,timeout=120) as response:
        for raw in response:
            if not raw.startswith(b'data: '): continue
            data=raw[6:].strip()
            if data==b'[DONE]':
                done=True
                break
            event=json.loads(data)
            if event.get('usage'): tokens=event['usage']['completion_tokens']
            if any(choice.get('text') for choice in event.get('choices',[])):
                now=time.perf_counter()
                if first is None: first=now
                if last is not None: intervals.append(now-last)
                last=now
    if first is None or tokens is None or not done: raise RuntimeError('response lacks content, completion-token usage or stream completion')
    return dict(endpoint=endpoint,ttft_seconds=first-start,elapsed_seconds=time.perf_counter()-start,
                output_tokens=tokens,mean_chunk_interval_seconds=statistics.mean(intervals) if intervals else None)


def main():
    p=argparse.ArgumentParser(); p.add_argument('endpoints',nargs='+')
    p.add_argument('--instance-type',required=True); p.add_argument('--requests',type=int,default=16)
    p.add_argument('--concurrency',type=int,default=4); p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--output',type=Path,default=Path('serving-results.json'))
    p.add_argument('--measure-decode', action='store_true', help='Record server decode intervals for weights-only MBU; requires concurrency of one request')
    a=p.parse_args()
    if min(a.requests,a.concurrency,a.max_tokens)<1: p.error('counts must be positive')
    if a.measure_decode and (a.concurrency != 1 or a.max_tokens < 2 or a.requests < len(a.endpoints)):
        p.error('Decode measurement requires concurrency of one request, at least two output tokens and a request for each endpoint')
    before = {url: snapshot(url) for url in a.endpoints} if a.measure_decode else {}
    started=time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as pool:
        rows=list(pool.map(lambda i:request(a.endpoints[i%len(a.endpoints)],a.max_tokens),range(a.requests)))
    elapsed=time.perf_counter()-started
    result=dict(instance_type=a.instance_type,requests_count=len(rows),wall_seconds=elapsed,concurrency=a.concurrency,
                output_tokens_per_second=sum(r['output_tokens'] for r in rows)/elapsed,
                median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in rows),requests=rows)
    if a.measure_decode:
        measurements = {}
        for endpoint in a.endpoints:
            for attempt in range(50):
                after = snapshot(endpoint)
                try:
                    measurements[endpoint] = decode_record(before[endpoint], after, [r for r in rows if r['endpoint']==endpoint])
                    break
                except ValueError:
                    if attempt == 49: raise
                    time.sleep(.1)
        result['decode_measurements'] = measurements
        result['server_metrics_before'] = before
        result['mean_server_itl_seconds'] = sum(x['server_decode_seconds'] for x in measurements.values())/sum(x['decode_steps'] for x in measurements.values())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
