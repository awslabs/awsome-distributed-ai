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
| `modelServers.type` | `sglang` | Default `vllm`; selects the metric names EPP reads |
| `modelServers.matchLabels` | `llm-d.ai/guide: dsv4flash` | Must match `guide` in [`../modelserver/values.yaml`](../modelserver/values.yaml), or EPP finds no replicas and Envoy forwards without routing |
| `epp.flags.v` | `1` | The guide's `2` logs every request |
| `epp.tolerations` | `nvidia.com/gpu` | Lets the router pod run on the GPU node. Drop it if your system nodes have 8 CPU free |
| `peakPrefillThroughput` | `30000` | Measured for one DSV4-Flash `tp=2` replica on B200 (the default 15928 is for Qwen3-32B on H100) |
| Envoy listener | `stream_idle_timeout: 0s`, `request_timeout: 0s` | The 5-minute default cuts requests that wait longer for the first token |

The scheduling profile is the guide's: `prefix-cache-affinity-filter` → `token-load-scorer`. The chart writes each `epp.pluginsCustomConfig` key into ConfigMap `dsv4flash-router-epp`, and EPP loads `/config/<pluginsConfigFile>`. To see the loaded config (with the overlay, the key is `flow-control-plugins\.yaml`):

```bash
kubectl -n $NS get cm dsv4flash-router-epp -o jsonpath='{.data.optimized-baseline-plugins\.yaml}'
```

## Flow control overlay

Once more than ready replicas × `maxConcurrency` requests are in flight (4 × 64 = 256), EPP queues new requests. As slots free up, it releases them by priority band, then round-robin across `x-llm-d-inference-fairness-id` values, then first come, first served.

Differences from llm-d v1.0.0 `guides/flow-control`:

| Value | Here | Guide | Why |
| --- | --- | --- | --- |
| Scheduling profile | as in `values.yaml` | queue/KV/prefix scorers | Flow control only decides when a request is released |
| `maxConcurrency` | 64 | 132 | 4 replicas held AgentX at c=256 and saturated at c=512 ([comparison](../TEST_RESULTS.md#earlier-router-comparison)) |
| `defaultRequestTTL` | 300s | 60s | AgentX requests run ~80 s, so queued ones wait minutes |
| Band `maxRequests` | 500 / 300 / 300 | 500 / 200 / 50 | About 256 requests wait at c=512; the guide's limits would reject most of them |
| `inferenceObjectives` | in values | separate `objectives.yaml` | The chart binds them to this release's pool |

- If `maxConcurrency` is too high, the backlog moves back into SGLang, where priority does not apply. If it is too low, GPUs sit idle. For other workloads, see [Tune Flow Control Concurrency](https://github.com/llm-d/llm-d/blob/v1.0.0/docs/operations/traffic/flow-control-tuning.md).
- A full queue or an expired TTL returns `429` with `x-llm-d-request-dropped-reason` (`rejected-saturated` / `rejected-ttl-expired`).
- Clients can set the `x-llm-d-*` headers themselves. Behind a public endpoint, strip them and set them from the caller's identity.

### Verifying flow control

The effect only shows above 256 in flight. Run the [Benchmark](../README.md#benchmark) command three times:

| Run | Load | Expected |
| --- | --- | --- |
| 1. No overlay | c=512 | Deep SGLang queues; TTFT rises for everyone |
| 2. Overlay | c=512 | The queue moves from SGLang into EPP, throughput about the same |
| 3. Overlay | premium c=128 + best-effort c=384, together | Premium TTFT stays near the c=256 level; best-effort waits |

For run 3, start two AIPerf clients with different `--artifact-dir` values, one with `-H 'x-llm-d-inference-objective:premium'` and one with `-H 'x-llm-d-inference-objective:best-effort'`. Watch the flow-control dashboard ([Monitoring](../README.md#monitoring-optional)): run 2 shows up in "Where requests wait: EPP vs SGLang", run 3 in "Queue time P90 by priority".
