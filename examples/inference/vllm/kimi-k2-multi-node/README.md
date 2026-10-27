<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Kimi K2 across two instances with vLLM: pipeline or tensor parallelism

[Kimi K2 Instruct-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905) (1,029 GB of FP8 weights) fits in
the 1,128 GB of GPU memory of one `p5en.48xlarge`, with almost nothing left for KV cache. This sample serves it with
vLLM on Amazon EKS across `NUM_NODES` instances, as a StatefulSet with one pod per instance: pod 0 serves the
OpenAI-compatible API, and the other pods join it headless. Three variables choose the layout:

| Layout | `NUM_NODES` | `TP` | `PP` | What crosses between the instances |
| --- | --- | --- | --- | --- |
| Pipeline parallelism across two instances | 2 | 8 | 2 | Activations between the two pipeline stages, once per step |
| Tensor parallelism across two instances | 2 | 16 | 1 | Every layer's tensor-parallel all-reduces |
| One instance | 1 | 8 | 1 | Nothing |

The image is the [AWS Deep Learning Container for vLLM](https://aws.github.io/deep-learning-containers/vllm/) 0.32,
pinned to a dated tag, and the model is pinned to a revision. NCCL reaches the other instance over EFA through the
image's aws-ofi-nccl plugin.

## Prerequisites

- An Amazon EKS cluster with `NUM_NODES` `p5en.48xlarge` nodes, the
  [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) and the
  [EFA device plugin](https://docs.aws.amazon.com/eks/latest/userguide/node-efa.html). Each pod takes a whole
  instance and holds the model in memory: it requests 1,700 GiB of the instance's 2 TiB.
- Outbound internet access from the pods to Hugging Face, and a Hugging Face token as a secret:
  `kubectl create secret generic hf-token --from-literal=HF_TOKEN=<token>`. The model is public, but Hugging Face
  rate-limits unauthenticated downloads: without a token, one 1,029 GB download took more than an hour.
- `kubectl` and `envsubst` where you run the commands.

## Deploy and benchmark

1. Configure the layout and source it:

   ```bash
   cp env_vars.example env_vars   # then edit NUM_NODES, TP and PP
   source env_vars
   ```

2. Create the API key. vLLM requires it on every `/v1` request, and `benchmark.yaml` reads it from the same secret:

   ```bash
   kubectl create secret generic vllm-api-key --from-literal=VLLM_API_KEY="$(openssl rand -hex 32)"
   ```

3. Deploy, and wait until pod 0 answers. Each pod first downloads the model into memory, then vLLM loads it and
   captures CUDA graphs; with a token, the deployments below were ready 14 to 20 minutes after `kubectl apply`:

   ```bash
   VARS='$IMAGE $INSTANCE_TYPE $GPU_PER_NODE $EFA_PER_NODE $MODEL_REPO $MODEL_REVISION $NUM_NODES $TP $PP'
   envsubst "$VARS" < serve.yaml | kubectl apply -f -
   kubectl rollout status statefulset/kimi-k2 --timeout=60m
   ```

4. On two instances, check that NCCL uses EFA. NCCL falls back to TCP sockets without failing when it can't load the
   plugin, so every pod should log `Using network Libfabric` and none `Using network Socket`:

   ```bash
   for p in $(kubectl get pods -l app=kimi-k2 -o name); do
     echo "$p: $(kubectl logs "$p" | grep -c 'Using network Libfabric') EFA, $(kubectl logs "$p" | grep -c 'Using network Socket') socket"
   done
   ```

5. Benchmark from inside the cluster. The job runs `vllm bench serve` with random prompts of 1,024 tokens and 128
   output tokens, at 1, 8 and 32 requests in flight, three rounds, and prints one CSV line per run (edit `ROUNDS`,
   `CONCURRENCY`, `INPUT_LEN` and `OUTPUT_LEN` in `benchmark.yaml` to change them):

   ```bash
   envsubst "$VARS" < benchmark.yaml | kubectl apply -f -
   kubectl wait --for=condition=complete job/kimi-k2-bench --timeout=60m
   kubectl logs job/kimi-k2-bench
   ```

6. Clean up, or change the layout in `env_vars` and redeploy from step 3:

   ```bash
   envsubst "$VARS" < benchmark.yaml | kubectl delete -f -
   envsubst "$VARS" < serve.yaml | kubectl delete -f -
   ```

## Results

Two deployments of each layout on `p5en.48xlarge` instances in one Availability Zone, each with the benchmark job's
three rounds (six runs per cell), AWS vLLM DLC 0.32.0, 1,024 input and 128 output tokens. Every two-instance
deployment logged only `Using network Libfabric` and no `Using network Socket` on both pods, and every deployment
answered `/v1/models` with 401 without the key and 200 with it. Mean of the runs' median time per output token
(TPOT, with the range of the six runs), output throughput, median time to first token (TTFT), and the KV cache vLLM
reported:

| Layout | TPOT at 1 in flight | TPOT at 8 | TPOT at 32 | Output tokens/s at 32 | TTFT at 32 | KV cache |
| --- | --- | --- | --- | --- | --- | --- |
| One instance, TP8 | 8.73 ms (8.71 to 8.74) | 11.98 ms (11.94 to 12.02) | 16.42 ms (16.31 to 16.61) | 587 | 5,000 ms | 12,640 tokens |
| PP2×TP8 | 8.90 ms (8.88 to 8.91) | 12.61 ms (12.18 to 13.42) | 17.48 ms (16.30 to 19.73) | 1,693 | 207 ms | 1,876,704 tokens |
| TP16 | 14.67 ms (14.61 to 14.73) | 17.55 ms (17.49 to 17.64) | 25.50 ms (24.40 to 27.80) | 1,171 | 258 ms | 914,016 tokens |

- **Pipeline parallelism across the instances costs little per token**: 2% more TPOT than one instance at 1 in
  flight and 5% at 8. Tensor parallelism across them costs 68% and 46%, because every layer's all-reduces cross EFA.
- **Both two-instance layouts serve more at 32 in flight**, 2.9 and 2.0 times one instance's output. Next to the
  weights, one instance has KV cache for only 12,640 tokens, so requests queue: its TTFT at 32 in flight is 5
  seconds.
- **TP16 holds half of PP2×TP8's KV cache**: this model's multi-head latent attention keeps the same compressed KV
  cache on every tensor-parallel rank, so each layer's cache is stored 16 times in TP16 and 8 times in PP2×TP8.

## Notes

- The pods run as a non-root user with a read-only root filesystem and no capabilities. The model goes to `/model`, a
  memory-backed `emptyDir` that is gone when the pod ends, so every deployment downloads it again.
- The API listens on port 8000 of pod 0, reachable as `kimi-k2-0.kimi-k2:8000` from the namespace, and refuses `/v1`
  requests without the key. The pods also listen on the rendezvous port 29500 for vLLM's startup; restrict both
  with a NetworkPolicy if your cluster enforces them.
- vLLM's group doesn't recover from one pod restarting by itself. Delete all the pods
  (`kubectl delete pod -l app=kimi-k2`) and the StatefulSet starts them together again.
- `--max-model-len 8192` and `--gpu-memory-utilization 0.92` in `serve.yaml` match the results above; the two-instance
  layouts have KV cache to spare for longer contexts.
