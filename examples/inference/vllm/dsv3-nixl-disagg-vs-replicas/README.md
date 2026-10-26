<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Two instances, two ways: prefill/decode disaggregation with NIXL or two replicas

With two instances and a model that fits on one, you can run two complete replicas, or split the work: one instance
prefills every prompt and the other decodes every answer, and each prompt's KV cache moves between them. This test
case serves [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) (688 GB of FP8 weights, pinned to
a revision) both ways on the same two `p5en.48xlarge` instances, with the same vLLM configuration (TP 8 on each
instance), and compares output throughput, time to first token (TTFT), time per output token (TPOT) and the
99th-percentile gap between tokens (ITL):

| Arm | `vllm-prefill` | `vllm-decode` | How requests are split |
| --- | --- | --- | --- |
| `replicas` | one complete server | one complete server | the benchmark client sends half of the prompts, at half the concurrency, to each |
| `disagg` | prefills every prompt | decodes every answer | vLLM's example proxy on `vllm-decode` sends each request to both servers |

In the `disagg` arm, vLLM's `NixlConnector` moves the KV cache over EFA with [NIXL](https://github.com/ai-dynamo/nixl)'s
LIBFABRIC backend. `kv_load_failure_policy` is `fail`, so a failed transfer is an error instead of a silent
recomputation on the decode instance. The arms alternate within each round (ABBA), so drift during the run can't
favor one of them. The default workloads include 100,000-token prompts, the length of a long agent or coding
session, where prefill chunks interleaved with decode steps slow every answer a replica is generating.

The image is the [AWS Deep Learning Container for vLLM](https://aws.github.io/deep-learning-containers/vllm/) 0.32,
pinned to a dated tag; it ships NIXL with the LIBFABRIC plugin and the EFA software. `2.benchmark.sh` downloads
vLLM's example proxy (`toy_proxy_server.py`) at the commit the image was built from and checks its SHA-256.

## Prerequisites

- An Amazon EKS cluster with two `p5en.48xlarge` nodes, the
  [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) and the
  [EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin).
- Each pod takes a whole instance and holds the model in memory: about 1.7 TiB of the instance's 2 TiB.
- Outbound internet access from the pods to Hugging Face and `raw.githubusercontent.com`. An `HF_TOKEN` is optional
  for this public model; to use one, create it as a secret:
  `kubectl create secret generic hf-token --from-literal=HF_TOKEN=<token>`.
- `kubectl`, `envsubst` and `python3` (3.8 or later) where you run the scripts.

## Run

1. Configure the run:

   ```bash
   cp env_vars.example env_vars
   $EDITOR env_vars               # set NAMESPACE, PREFILL_NODE and DECODE_NODE
   source env_vars
   ```

2. Create the two pods (`vllm-prefill`, `vllm-decode`). They only sleep; the scripts start vLLM inside them:

   ```bash
   ./0.create-pods.sh
   ```

3. Download the model into both pods and check that the copies match:

   ```bash
   ./1.stage-model.sh
   ```

4. Serve and benchmark both arms. `WORKLOADS` lists `input:output` token counts and the concurrencies to run them at
   (`input:output@c1,c2,...`, even concurrencies only); the default is
   `2048:256@8,32,64 8192:256@8,32,64 100000:256@2,4,8` in two rounds (`ROUNDS`):

   ```bash
   ./2.benchmark.sh               # results in ./results/disagg-vs-replicas-<UTC time>
   python3 3.analyze.py results/disagg-vs-replicas-<UTC time>
   ```

5. Delete the pods:

   ```bash
   kubectl -n $NAMESPACE delete pod -l app=vllm-pd
   ```

## What the scripts check

- **KV transfer.** A disaggregated round counts only if every vLLM worker on both instances logged
  `Backend LIBFABRIC was instantiated` and every decode worker logged `Successfully created connection` to the
  prefill instance (`transport.txt`); otherwise its rows are removed from `serving.csv`. With
  `kv_load_failure_policy` set to `fail`, a transfer that fails later fails its request instead of being recomputed
  on the decode instance, which would hide the cost of the transfer.
- **Proxy.** The proxy script must match `PROXY_SHA256` before it runs.
- **Model copies.** `1.stage-model.sh` compares the pods' file listings and sizes.

## Results

One round on two `p5en.48xlarge` under different layer-3 network nodes in one Availability Zone
(`ROUNDS=1 WORKLOADS="8192:256@8 100000:256@2,4,8"`), AWS vLLM DLC 0.32.0 (vLLM 0.32.0, NIXL 1.5.0), DeepSeek-V3-0324
at TP 8 on each instance. In the disaggregated arm all 16 workers instantiated LIBFABRIC and the decode workers
connected to the prefill instance; all 176 requests of the run completed. From `3.analyze.py` (disaggregated /
replicas; medians unless marked p99):

| Prompt tokens | In flight | Output tokens/s | TTFT | TPOT | TPOT p99 | ITL p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 8,192 | 8 | 487 / 522 | 0.5 / 1.2 s | 12.6 / 10.8 ms | 12.7 / 13.9 ms | 19.7 / 11.4 ms |
| 100,000 | 2 | 27 / 41 | 14.1 / 9.3 s | 12.7 / 12.7 ms | 12.7 / 12.8 ms | 13.2 / 13.3 ms |
| 100,000 | 4 | 28 / 48 | 31.9 / 13.8 s | 12.7 / 30.1 ms | 12.7 / 47.1 ms | 13.2 / 739 ms |
| 100,000 | 8 | 29 / 52 | 66.8 / 22.6 s | 12.7 / 67.0 ms | 12.7 / 117.8 ms | 13.2 / 1,038 ms |

- **With 100,000-token prompts, the replicas' answers slow down once each replica holds more than one request**: TPOT
  2.4 times the disaggregated arm's at 4 in flight and 5.3 times at 8, with gaps between tokens of up to a second.
  Each replica's decode steps share the GPUs with the prefill chunks of the other requests' long prompts. The
  disaggregated arm's decoding stayed at 12.7 ms per token, with a p99 gap of 13.2 ms.
- **The replicas still produced more output and answered sooner** (1.5 to 1.8 times the output at 100,000 tokens,
  lower TTFT), because both instances prefill. The one prefill instance queues: at 8 in flight it took 285 seconds
  for 32 prompts, 8.9 seconds per prompt.
- **At 8,192 tokens the arms were close**: the replicas produced 7% more output, the disaggregated arm answered in
  less than half the time.

The run took 41 minutes, 6 of them staging the model. The first vLLM start took 580 seconds, a later one 200.

## Notes

- The pods run as a non-root user with a read-only root filesystem and no capabilities. vLLM's caches, the launch
  scripts and the logs go to `/tmp`, an `emptyDir`; the model goes to `/model`, a memory-backed `emptyDir` that is
  gone when the pod ends. Each pod ends itself after 4 hours (`activeDeadlineSeconds`).
- In the `replicas` arm, the percentiles are the mean of the two replicas' percentiles, and throughput is their sum.
- `kubectl exec` from outside the cluster can drop; the scripts run vLLM, the proxy and the benchmark clients
  detached inside the pods and retry their `kubectl exec` calls.
