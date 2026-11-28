<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Router values

Values for the `llm-d-router-standalone` chart v1.0.0, layered in this order (see [Deploy](../README.md#deploy)):

1. [`upstream/base.values.yaml`](./upstream/base.values.yaml): verbatim from llm-d v1.0.0 `guides/recipes/router/`.
2. [`upstream/optimized-baseline.values.yaml`](./upstream/optimized-baseline.values.yaml): verbatim from `guides/optimized-baseline/router/`.
3. [`values.yaml`](./values.yaml): this sample's overrides. Make changes here, not in `upstream/`.
4. [`flow-control.values.yaml`](./flow-control.values.yaml): optional, see [Flow control overlay](#flow-control-overlay).

The chart creates the `InferencePool` `dsv4flash-router` and one pod running EPP with an Envoy sidecar, so no Gateway API is needed. Without the overlay, EPP logs `InferenceObjective GVK does not exist ... Skipping`. That is expected.

## Overrides

| Value | Setting | Why |
| --- | --- | --- |
| `modelServers.type` | `sglang` | Default `vllm`; used by the chart's generated plugin file. With this sample's own plugin file, EPP reads SGLang metric names because the model pods carry `llm-d.ai/engine-type: sglang` |
| `modelServers.matchLabels` | `llm-d.ai/guide: dsv4flash` | Must match `guide` in [`../modelserver/values.yaml`](../modelserver/values.yaml), or EPP finds no replicas and requests fail with `503` |
| `monitoring.prometheus.auth.enabled` | `false` | EPP serves `/metrics` without auth, so the `llmd-epp` scrape job sends no token |
| `epp.flags.v` | `1` | The guide's `2` logs every request |
| `epp.tolerations` | `nvidia.com/gpu` | Lets the router pod run on the GPU node. Drop it if your system nodes have 8 CPU free |
| `peakPrefillThroughput` | `30000` | Measured for one DSV4-Flash `tp=2` replica on B200 (the default 15928 is for Qwen3-32B on H100) |

The scheduling profile is the guide's: `prefix-cache-affinity-filter` → `token-load-scorer`. The chart writes each `epp.pluginsCustomConfig` key into ConfigMap `dsv4flash-router-epp`, and EPP loads `/config/<pluginsConfigFile>`. To see the loaded config (with the overlay, the key is `flow-control-plugins\.yaml`):

```bash
kubectl -n $NS get cm dsv4flash-router-epp -o jsonpath='{.data.optimized-baseline-plugins\.yaml}'
```

## Flow control overlay

Once more than ready replicas × `maxConcurrency` requests are in flight (4 × 128 = 512), EPP queues new requests. As slots free up, it releases them by priority band, then round-robin across `x-llm-d-inference-fairness-id` values, then first come, first served.

Differences from llm-d v1.0.0 `guides/flow-control`:

| Value | Here | Guide | Why |
| --- | --- | --- | --- |
| Scheduling profile | as in `values.yaml` | queue/KV/prefix scorers | Flow control only decides when a request is released |
| `maxConcurrency` | 128 | 132 | Set for the `bench_serving` workload in [Verifying flow control](#verifying-flow-control), not for AgentX |
| `defaultRequestTTL` | 300s | 60s | Gives best-effort requests queued behind premium more time before they are dropped |
| Band `maxRequests` | 500 / 300 / 300 | 500 / 200 / 50 | Up to 256 requests wait at 768 in flight (768 − 512); the guide's limits would reject most of them |
| `inferenceObjectives` | in values | separate `objectives.yaml` | The chart binds them to this release's pool |

- If `maxConcurrency` is too high, the backlog moves back into SGLang, where priority does not apply. If it is too low, GPUs sit idle. For other workloads, see [Tune Flow Control Concurrency](https://github.com/llm-d/llm-d/blob/v1.0.0/docs/operations/traffic/flow-control-tuning.md).
- A full queue or an expired TTL returns `429` with `x-llm-d-request-dropped-reason` (`rejected-saturated` / `rejected-ttl-expired`).
- Clients can set the `x-llm-d-*` headers themselves. Behind a public endpoint, strip them and set them from the caller's identity.

### Verifying flow control

The effect only shows above 512 in flight, so both runs below start two clients together, one at c=512 and one at c=256, for up to 768 in flight (`--max-concurrency` is a ceiling). Run 1 needs flow control off, so do it before the [Flow control](../README.md#flow-control-and-priority-optional) steps.

The load is SGLang's `bench_serving` with the `generated-shared-prefix` dataset in multi-turn mode, sized from the statistics of `semianalysis_cc_traces_weka_062126_256k` instead of replaying it. Each concurrency slot is one 8-turn conversation with its own system prompt, so conversations share no prefix. With `--gsp-range-ratio 0.5`, the averages are a 43k-token first turn, 1380 new input tokens and 1260 output tokens per turn. The numbers are for comparing the two runs with each other, not with AgentX results.

The two clients must use different seeds. `bench_serving` draws the system prompts group by group from `--seed`, so with the same seed the c=256 client would send exactly the first 256 system prompts of the c=512 client, and prefix affinity would route each pair to the same replica. Each call passes its own `--seed`: 42 for the c=512 client and 43 for the c=256 client, the same in both runs so the runs stay comparable.

| Run | Load | Expected |
| --- | --- | --- |
| 1. No overlay | c=512 + c=256, no headers, together | SGLang admits every request at once (up to 256 running per replica). The KV cache fills, cached prefixes are evicted and prefilled again, and both clients see the same high latency |
| 2. Overlay | premium c=512 + best-effort c=256, together | Requests above 512 in flight wait in EPP and are released premium first. The gain has two sources. Capping each replica at 128 keeps its KV cache from filling, so prefixes stay cached and both clients can finish sooner than in run 1. Priority then gives premium the free slots first, at best-effort's expense: best-effort's queued requests wait longer, and some expire at the TTL (`EvictedTTL`, `429`) |

Run from the recipe directory, with `NS` set and the bench pod enabled (see [Benchmark](../README.md#benchmark)). This helper starts `bench_serving` from the SGLang image in the background inside the bench pod, so a dropped `kubectl` session does not stop the run. Arguments: concurrency (also the number of conversations), run name, request_rate, then any extra `bench_serving` flags.

```bash
run_bench() {
  local c=$1 name=$2 rate=$3; shift 3
  kubectl -n $NS exec dsv4flash-sglang-bench -- bash -c "
    mkdir -p /nvme/bench-results && cd /nvme
    setsid nohup python3 -m sglang.benchmark.serving --backend sglang-oai-chat \
      --base-url http://dsv4flash-router-epp.$NS.svc.cluster.local:80 \
      --model deepseek-ai/DeepSeek-V4-Flash-0731 \
      --dataset-name generated-shared-prefix \
      --gsp-num-groups $c --gsp-prompts-per-group 1 --gsp-num-turns 8 \
      --gsp-system-prompt-len 57000 --gsp-question-len 1840 --gsp-output-len 1680 \
      --gsp-range-ratio 0.5 --gsp-fast-prepare \
      --max-concurrency $c --request-rate $rate --warmup-requests 1 --ready-check-timeout-sec 0 \
      --output-file /nvme/bench-results/$name.jsonl $* > /nvme/bench-results/$name.log 2>&1 &"
}
watch_bench() { kubectl -n $NS exec dsv4flash-sglang-bench -- tail -f /nvme/bench-results/$1.log; }
```

Both runs send the same prompts, and upgrading the router between them leaves the SGLang radix caches in place while the new EPP starts with an empty prefix index. Flush every replica's cache before each run so both start from the same state. A request through the router reaches only one replica, so this calls each pod directly; SGLang refuses to flush while requests are running, so run it only after both clients have finished:

```bash
flush_cache() {
  for ip in $(kubectl -n $NS get pods -l llm-d.ai/guide=dsv4flash -o jsonpath='{.items[*].status.podIP}'); do
    kubectl -n $NS exec dsv4flash-sglang-bench -- curl -s -X POST http://$ip:8000/flush_cache; echo " ($ip)"
  done
}
```

1. Baseline, flow control off, both clients without priority headers:

   ```bash
   flush_cache
   run_bench 512 fc-run1-c512 4 --seed 42
   run_bench 256 fc-run1-c256 2 --seed 43
   watch_bench fc-run1-c512    # Ctrl-C stops watching, not the run; wait for the summary table
   watch_bench fc-run1-c256    # wait for this summary too before step 2 replaces the EPP pod
   ```

2. Turn flow control on with [Flow control](../README.md#flow-control-and-priority-optional) steps 1–3, then run the same load with two tiers:

   ```bash
   flush_cache
   run_bench 512 fc-run2-premium 4 --seed 42 --header x-llm-d-inference-objective=premium x-llm-d-inference-fairness-id=interactive
   run_bench 256 fc-run2-besteffort 2 --seed 43 --header x-llm-d-inference-objective=best-effort x-llm-d-inference-fairness-id=batch
   watch_bench fc-run2-premium
   watch_bench fc-run2-besteffort
   ```

Compare the summary tables at the end of each log, client by client: `fc-run1-c512` against `fc-run2-premium`, and `fc-run1-c256` against `fc-run2-besteffort`, including each client's completed count. For example, `kubectl -n $NS exec dsv4flash-sglang-bench -- tail -40 /nvme/bench-results/fc-run2-besteffort.log`. On the flow-control dashboard ([Monitoring](../README.md#monitoring-optional)), run 2 shows up in "Where requests wait: EPP vs SGLang", "Queue time P90 by priority" and "Request outcomes by priority". `RejectedCapacity` or `EvictedTTL` on the premium band means the band limits or `defaultRequestTTL` are too tight for the load; on the best-effort band it is the cost of giving premium priority at this load.

### Reference results

One run of the steps above on one `p6-b200.48xlarge` (4 replicas, `tp=2`), for reference. The prefix-cache hit rate is cached tokens over prefill tokens, summed over the 4 replicas' prefill log lines. KV usage is the mean `token usage` on the same lines.

| Client | Completed | Duration (s) | Req/s | Output tok/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 / p90 / p99 (ms) | E2E p50 / p90 / p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Run 1, c=512 | 4096 / 4096 | 1478.03 | 2.77 | 3,508 | 12,226 / 34,048 / 39,682 | 137.4 / 166.7 / 233.5 | 167,379 / 254,622 / 340,466 |
| Run 2, c=512 `premium` | 4096 / 4096 | 851.02 | 4.81 | 6,092 | 2,732 / 17,150 / 39,045 | 57.9 / 75.3 / 123.5 | 74,880 / 115,109 / 166,055 |
| Run 1, c=256 | 2048 / 2048 | 1467.87 | 1.40 | 1,784 | 12,220 / 33,963 / 39,134 | 137.9 / 162.9 / 232.0 | 167,038 / 256,602 / 328,355 |
| Run 2, c=256 `best-effort` | 1846 / 2048 | 872.86 | 2.11 | 2,710 | 658 / 155,733 / 302,399 | 15.3 / 86.6 / 124.3 | 19,646 / 248,706 / 412,542 |

| Run | Prefix-cache hit rate | Mean KV usage | EPP outcomes |
| --- | --- | --- | --- |
| 1. No overlay | 0.208 | 0.89 | n/a |
| 2. Overlay | 0.574 | 0.62 | premium: all `Dispatched`; best-effort: 202 `EvictedTTL` (= 2048 − 1846) |

- In run 1, both clients see the same latency: TTFT p50 about 12.2 s and TPOT p50 about 137 ms. Each replica runs about 170 requests at once (client concurrency 451 + 227, over 4), its KV cache stays near full, and only 21% of prefill tokens hit the cache.
- In run 2, both clients finish in about 860 s instead of about 1470 s. Holding each replica at 128 requests lowers KV usage to 0.62 and raises the hit rate to 0.57, which is where most of premium's TPOT gain (137 → 58 ms) comes from.
- Best-effort's median request is fast (TTFT p50 0.66 s), because once admitted it runs on an unsaturated replica. Its tail is where priority costs it: TTFT p90 is 156 s, and 202 requests expired in the EPP queue at the 300 s TTL.
- These runs do not separate the two sources of the gain. To measure the concurrency cap alone, repeat run 2 without the `x-llm-d-inference-objective` headers.
