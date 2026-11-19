<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Test Results

Verified on 2026-11-18 with `nvmeHostPath=/mnt/k8s-disks/0`.

- Cluster: Amazon EKS 1.35 (ap-south-1), one `p6-b200.48xlarge` (8× B200), `m5.large` system nodes.
- Software: Helm 3.21.3, chart `llm-d-router-standalone` v1.0.0, GAIE v1.6.2 CRD, `lmsysorg/sglang:v0.5.22`, AIPerf 0.13.0.

There were two runs on the same node:

1. **Cold start, functional checks, scaling, monitoring, and the benchmark.** In this run the model-server objects were applied as plain manifests, before they were moved into `modelserver/`. The chart renders the same ServiceAccount, Deployment, and bench Pod (compared field by field after `helm template`), and the Deployment kept the same pod-template hash (`5dc4dfbdb4`).
2. **The [Deploy](./README.md#deploy) steps verbatim with both Helm charts.** The model servers were reinstalled as release `dsv4flash-sglang`, and the router release was upgraded in place. The functional and scaling checks were repeated (see [Helm re-run](#helm-re-run)).

**Deploy.** All 4 pods were scheduled at 12:33:16 and turned Ready together at 12:45:46. That is a 12.5-minute cold start: weights 302 s plus the DSpark draft 17 s, then the DeepGEMM/JIT warmup and CUDA graphs. Each replica reports `max_total_num_tokens=9540096`, `max_running_requests=256`, `context_len=512000`. EPP started `controller=inferencepool`, and the rendered pool was:

```json
{"appProtocol":"http","endpointPickerRef":{"failureMode":"FailOpen","kind":"Service","name":"dsv4flash-router-epp","port":{"number":9002}},
 "selector":{"matchLabels":{"llm-d.ai/guide":"dsv4flash"}},"targetPorts":[{"number":8000}]}
```

**Discovery and requests** (through `dsv4flash-router-epp.dsv4flash-llmd.svc`, from the bench pod):

```text
EndpointSlice: 10.90.19.114:true 10.90.20.206:true 10.90.20.145:true 10.90.30.189:true
llm_d_epp_ready_endpoints{name="dsv4flash-router"} 4
non-stream: "content":"391" ... "finish_reason":"stop"   HTTP 200 0.63s
stream:     6 SSE data lines, ends with "data: [DONE]"
```

**Prefix affinity.** The same 3,413-token prompt was sent 4 times. All 4 went to one replica (per-pod `sglang:num_requests_total` before → after: `10.90.30.189` 1 → 5, others unchanged), and requests 2–4 each reported `cached_tokens 3328`.

**Scaling.** One request was sent every 5–7 s, plus 20 back-to-back requests at 3 replicas.

```text
12:48:08  replicas=4 -> 3
12:48:16  http=200 llm_d_epp_ready_endpoints 3     (and every later check, 3)
          20 reqs at 3 replicas: 20 x 200
12:49:32  replicas=3 -> 4; new pod Ready after 152 s (warm restart)
          llm_d_epp_ready_endpoints 4, new pod 10.90.24.161 served 12 of the next 40 requests
```

**Monitoring.** The collector already ran the `llmd-epp` job. Rendering the stack's `adot-collector-prometheus.yaml` with `scrape-configs/llmd-epp.yaml` (the step-6 `envsubst`) yields a job identical to the one running. Amazon Managed Service for Prometheus showed `up=1` for the 4 SGLang pods (`job="sglang"`) and the EPP (`job="epp"`). All 24 metric names used in `dashboard.json` had series. The dashboard was not imported into Grafana in this run.

**Benchmark.** AgentX c=64, 300 s run with `--unsafe-override`, using the [Benchmark](./README.md#benchmark) command. The 65-request sequential warmup took 621 s before the measured phase.

| Requests | Errors | Req/s | Input tok/s | Output tok/s | TTFT p50 / p99 | ITL p50 / p99 | Avg ISL | Prefix hit (actual / theoretical) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 683 | 0 | 2.07 | 213,971 | 2,221 | 0.40 / 2.17 s | 2.8 / 6.6 ms | 103,383 | 94.5% / 95.5% |

During that window, AMP showed the following per replica:

| Pod | Requests (share) | Cache hit | Max queued | Peak KV occupied |
| --- | --- | --- | --- | --- |
| `…-8l68c` | 208 (30.5%) | 94.1% | 0 | 30% |
| `…-hd2qj` (rejoined pod) | 209 (30.6%) | 95.5% | 0 | 35% |
| `…-7g5cg` | 144 (21.1%) | 93.9% | 0 | 25% |
| `…-vm5xn` | 121 (17.7%) | 93.7% | 0 | 34% |

The actual hit rate is within 1 point of the trace's ceiling with no queueing. Uneven request shares are expected, because affinity keeps each session on the replica that holds its prefix. Speculative decoding accepted 3.9 tokens per step (mean). At c=64 the KV is far from full, so this run checks routing correctness, not capacity. See [Earlier router comparison](#earlier-router-comparison) for behavior at higher concurrency.

## Helm re-run

The Deploy steps were run as written (13:28 UTC). `helm upgrade --install dsv4flash-sglang` created revision 1, the router release went to revision 2, and the CRD apply was `configured` (unchanged). The 4 replicas were Ready at 13:31:33, a 3-minute warm restart reusing the page cache and JIT cache. The Deployment kept pod-template hash `5dc4dfbdb4` from the first run.

```text
helm upgrade ... --set bench.enabled=true    revision 2, Deployment generation stays 1 (replicas untouched); bench pod "aiperf ready"
non-stream:  "content":"391"  HTTP 200 0.22s
stream:      6 SSE data lines, ends with "data: [DONE]"
affinity:    3,413-token prompt x4 -> all on 10.90.28.204 (1 -> 5), requests 2-4 cached_tokens 3328
13:42:18  helm upgrade --reuse-values --set replicas=3   (revision 3)
13:42:23  http=200 llm_d_epp_ready_endpoints 3     (and every check after it, 3); then 20 reqs at 3 replicas: 20 x 200
13:43:40  helm upgrade --reuse-values --set replicas=4   (revision 4); new pod Ready 13:46:11 (151 s)
          llm_d_epp_ready_endpoints 4, EndpointSlice 4 x ready; 40 requests: 40 x 200, new pod 10.90.25.162 served 10
```

## Earlier router comparison

Not a run of this sample: an earlier comparison on the same 4 replicas with AgentX (round-robin, the SGLang model gateway, and several EPP configurations, 1800 s per concurrency level), with EPP configured through `file-discovery` instead of an `InferencePool`. It is why this sample uses an affinity-first profile and `maxConcurrency: 64`.

- Prefix-aware routing matters most. At c=256, round-robin reached 2.09 req/s at 49% cache hits; prefix-aware routers reached 6.2–6.5 req/s at 89–94%.
- Without an affinity filter first, EPP adds a `utilization-detector` filter that, at c=512, pushed requests off the replica holding their prefix (17% lower throughput). This profile puts the affinity filter first, which avoids that.
- Above c≈256, uncached prefill saturates the replicas whatever the router does.
- Precise prefix-cache routing (KV events) is not included. It needs extra SGLang flags and a vLLM render service, and was not tried with an `InferencePool`.
