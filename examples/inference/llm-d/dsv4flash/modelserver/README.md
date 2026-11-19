<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Model server chart

Helm chart `dsv4flash-sglang`: the SGLang replicas (`Deployment dsv4flash-sglang`) and an optional AIPerf client pod. It is derived from llm-d v1.0.0 `guides/optimized-baseline/modelserver/gpu/sglang` (base recipe plus `patch-sglang.yaml`). The router chart's `InferencePool` selects these pods by the `llm-d.ai/guide` label.

## Values

All settings are in [`values.yaml`](./values.yaml).

| Value | Default | Notes |
| --- | --- | --- |
| `nvmeHostPath` | (required) | Node NVMe holding `huggingface/hub`. `/opt/dlami/nvme` on HyperPod, `/mnt/k8s-disks/0` on self-managed EKS |
| `guide` | `dsv4flash` | Pod label `llm-d.ai/guide`. Must match `router.modelServers.matchLabels` in [`../router/values.yaml`](../router/values.yaml) |
| `image` | `docker.io/lmsysorg/sglang:v0.5.22` | |
| `model` | `deepseek-ai/DeepSeek-V4-Flash-0731` | Also the served model name |
| `replicas` | `4` | Each replica uses `tp` GPUs, so one 8-GPU node holds 4 |
| `tp` | `2` | Sets `--tp` and the `nvidia.com/gpu` request |
| `instanceTypes` | `p6-b200.48xlarge`, `ml.p6-b200.48xlarge` | Node affinity. EKS labels the bare EC2 type, HyperPod the `ml.` one |
| `engineArgs` | DSpark, `flashinfer_mxfp4`, … | SGLang flags other than model, host, port, and `tp` |
| `bench.enabled` | `false` | Adds the AIPerf client pod, see [Benchmark client](#benchmark-client) |

Any change other than `replicas` or `bench` restarts every replica together. The Deployment uses `Recreate` because the GPUs are fully allocated, and surge pods would stay Pending.

## Model loading and startup

The pods mount `<nvmeHostPath>` at `/models` and load the weights from `/models/huggingface` with `HF_HUB_OFFLINE=1`, so the weights must be staged first (see [Prerequisites](../README.md#prerequisites)). To download at startup instead, add `HF_TOKEN` to the container and remove `HF_HUB_OFFLINE` in [`templates/deployment.yaml`](./templates/deployment.yaml).

A cold start takes about 12.5 minutes: weights ~5 minutes, the DSpark draft model, then DeepGEMM/JIT warmup and CUDA graphs ~6 minutes. The JIT caches go to `<nvmeHostPath>/sglang-jit-cache`, so a restart that also hits the page cache takes 2.5–3.5 minutes.

| Probe | Path | Setting |
| --- | --- | --- |
| `startupProbe` | `/v1/models` | Every 30 s, up to 40 minutes |
| `readinessProbe` | `/v1/models` | Only Ready pods join the `InferencePool`, so this must stay an honest "model loaded" signal |
| `livenessProbe` | `/health` | 6 failures × 10 s, because `/health` can be slow under heavy prefill |

## Settings the router depends on

- `--enable-metrics`: EPP scrapes `/metrics` for queue depth, KV usage, and the KV page size. Without it, routing degrades.
- `SGLANG_TIMEOUT_KEEP_ALIVE=900`: the default of 5 seconds closes the upstream connections Envoy pools.
- `--enable-cache-report`: adds `usage.prompt_tokens_details.cached_tokens` to responses, which is how the affinity check in [Verify](../README.md#verify) reads the cache hit.
- The labels `sglang-metrics=true` and `component=unified` and the `prometheus.io/*` annotations are for the [monitoring](../monitoring/README.md) stack's `sglang` scrape job. The router does not use them.

## Scaling

`helm upgrade --reuse-values --set replicas=N` changes only the replica count. A pod leaves the pool when it stops being Ready and joins once its `readinessProbe` passes, with no router change. `kubectl scale deploy/dsv4flash-sglang` works the same way, but the next `helm upgrade` resets the count to the chart value. Replicas on more B200 nodes are picked up the same way. EPP's `llm_d_epp_ready_endpoints` metric follows the Ready replica count.

## Benchmark client

`bench.enabled=true` adds Pod `dsv4flash-sglang-bench`. It is not created by a normal install. Turning it on or off does not touch the replicas. The pod installs AIPerf 0.13.0 into a venv on the SGLang image, prints `aiperf ready`, and then idles until it is disabled, so you can `kubectl exec` several runs into it.

Why it is a pod in the cluster:

- The router Service is `ClusterIP`. A `port-forward` does not hold up under dozens of concurrent, minute-long streaming requests.
- AIPerf needs the DeepSeek tokenizer and Python dependencies, which the SGLang image already has.
- The dataset and tokenizer cache go to `<nvmeHostPath>/huggingface`, so a re-created pod does not rebuild the AgentX dataset.
- It requests 32 CPU for client-side tokenization and no GPU. The `m5.large` system nodes this was verified on cannot fit it, so it runs on the GPU node.

AIPerf notes for the [Benchmark](../README.md#benchmark) command:

- The `inferencex-agentx-mvp` scenario's minimum duration is 900 s. `--unsafe-override` allows a shorter smoke run. Use `--benchmark-duration 1800` for comparable numbers.
- The first run builds the dataset, which is slow. That is why `AIPERF_DATASET_CONFIGURATION_TIMEOUT` and `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` are raised. Each run then does a sequential 65-request warmup (~10 minutes) before the measured phase.
- AIPerf reads server metrics from `--url`, which the router balances across replicas. It therefore logs `counter reset` warnings and its server-metrics export is unreliable. Use the client-side table and the dashboard instead.
- Keep `AIPERF_HTTP_KEEPALIVE_TIMEOUT` below every idle timeout on the path, or requests fail with `ECONNRESET`.
- `-H 'name:value'` adds a request header, for example the flow-control headers in [Flow control and priority](../README.md#flow-control-and-priority-optional).
