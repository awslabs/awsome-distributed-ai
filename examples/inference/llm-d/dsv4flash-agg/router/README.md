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
| 1. No overlay | c=512 + c=256, no headers, together | SGLang admits every request at once (up to 256 running per replica), so the backlog forms inside SGLang and both clients see the same latency |
| 2. Overlay | premium c=512 + best-effort c=256, together | Requests above 512 in flight wait in EPP and are released premium first. Premium's gain comes from priority, at best-effort's expense: best-effort waits longer, and some of its requests can expire at the TTL (`EvictedTTL`, `429`) |

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
