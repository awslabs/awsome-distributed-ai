<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Serving one model across two instances: pipeline or tensor parallelism, near or far

When a model needs more than one instance, vLLM can split it two ways. Pipeline parallelism (PP) puts half of the
layers on each instance and passes activations between them; tensor parallelism (TP) splits every layer across all
16 GPUs, so each layer's all-reduces cross between the instances. This test case serves the same model in three
layouts on Amazon EKS and measures time per output token (TPOT), output throughput and time to first token (TTFT):

| Layout | Instances | What crosses between the instances |
| --- | --- | --- |
| `tp8` | 1 | Nothing: the single-instance baseline |
| `pp2tp8` | 2 | Activations between the two pipeline stages, once per step |
| `tp16` | 2 | Every layer's tensor-parallel all-reduces |

Each two-instance layout also runs with two placements of the second instance, chosen from the network-node labels
that Amazon EKS puts on nodes of
[instance types that support EC2 instance topology](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-topology.html)
(`topology.k8s.aws/network-node-layer-1` to `-3`): *near* shares all three network-node layers with the first
instance, *far* shares layers 1 and 2 and differs at layer 3. Near and far alternate within each round, so drift
during the run can't favor one of them.

The model is [Kimi K2 Instruct-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905) (1,029 GB of FP8
weights, pinned to a revision), which just fits in the 1,128 GB of GPU memory of one `p5en.48xlarge`, so all three
layouts can serve it. The image is the [AWS Deep Learning Container for vLLM](https://aws.github.io/deep-learning-containers/vllm/)
0.32, pinned to a dated tag.

## Prerequisites

- An Amazon EKS cluster with three `p5en.48xlarge` nodes, the
  [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) and the
  [EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin). Two of the three
  nodes must share a layer-3 network node; `0.choose-nodes.sh` shows whether they do.
- Each pod takes a whole instance and holds the model in memory: about 1.7 TiB of the instance's 2 TiB.
- Outbound internet access from the pods to Hugging Face. An `HF_TOKEN` is optional for this public model; to use
  one, create it as a secret: `kubectl create secret generic hf-token --from-literal=HF_TOKEN=<token>`.
- `kubectl`, `envsubst` and `python3` (3.8 or later) where you run the scripts.

## Run

1. Configure the run and pick the nodes:

   ```bash
   cp env_vars.example env_vars
   ./0.choose-nodes.sh            # prints the layer-3 groups and suggested export lines
   $EDITOR env_vars               # paste HEAD_NODE, NEAR_NODE and FAR_NODE; set NAMESPACE
   source env_vars
   ```

2. Create the three pods (`vllm-head`, `vllm-near`, `vllm-far`). They only sleep; the scripts start vLLM inside them:

   ```bash
   ./1.create-pods.sh
   ```

3. Download the model into every pod and check that the copies match:

   ```bash
   ./2.stage-model.sh
   ```

4. Serve and benchmark each layout. Defaults: two rounds, 1,024 input and 128 output tokens, 1, 8, 32 and 64
   requests in flight. Set `LAYOUTS`, `ROUNDS`, `CONCS`, `INLEN` and `OUTLEN` to change them:

   ```bash
   ./3.benchmark-layouts.sh       # results in ./results/layouts-<UTC time>
   python3 4.analyze.py results/layouts-<UTC time>
   ```

5. Delete the pods:

   ```bash
   kubectl -n $NAMESPACE delete pod -l app=vllm-layouts
   ```

## What the scripts check

- **Placement.** `3.benchmark-layouts.sh` reads the pods' nodes and their labels and stops unless `vllm-near` shares
  three network-node layers with `vllm-head` and `vllm-far` shares two.
- **Transport.** A two-instance run counts only if every NCCL process on both instances logs
  `Using network Libfabric` (the aws-ofi-nccl plugin over EFA) and none logs `Using network Socket`. NCCL falls back
  to TCP sockets without failing when it can't load the plugin, so the script sets `NCCL_NET_PLUGIN` to the plugin's
  path in the image (`NET_PLUGIN` to override) and discards any run whose logs don't prove EFA.
- **Model copies.** `2.stage-model.sh` compares each pod's file listing and sizes with the head's.

## Results

One round on three `p5en.48xlarge` in one Availability Zone (`ROUNDS=1 CONCS="1 8 32"`; the near pair under one
layer-3 network node, the far instance under another), AWS vLLM DLC 0.32.0 (vLLM 0.32.0, PyTorch 2.13.0, NCCL 2.29.7),
1,024 input and 128 output tokens. Every two-instance run had all 16 NCCL processes on EFA. Median TPOT, output
throughput and median TTFT, from `4.analyze.py`:

| Layout | Second instance | TPOT at 1 / 8 / 32 in flight | Output tokens/s at 32 in flight | TTFT at 32 in flight |
| --- | --- | --- | --- | --- |
| `tp8` | none | 8.74 / 11.93 / 16.64 ms | 573 | 5,104 ms |
| `pp2tp8` | near | 8.94 / 12.31 / 20.21 ms | 1,272 | 614 ms |
| `pp2tp8` | far | 8.91 / 12.28 / 19.80 ms | 1,300 | 581 ms |
| `tp16` | near | 13.95 / 16.87 / 27.63 ms | 972 | 705 ms |
| `tp16` | far | 14.49 / 17.44 / 28.73 ms | 937 | 722 ms |

- **Pipeline parallelism across the instances cost little per token**: 2 to 3% more TPOT than one instance at 1 and
  8 in flight. Tensor parallelism across them cost 60% at 1 in flight and 41% at 8; in that layout every layer's
  all-reduces cross EFA.
- **Both two-instance layouts served more at 32 in flight**, 2.2 and 1.7 times one instance's output, because one
  instance has almost no memory left for KV cache after the weights: vLLM reported 12,640 tokens of KV cache for
  `tp8`, 1,876,704 for `pp2tp8` and 914,016 for `tp16`, and `tp8`'s queue shows in its 5.1-second TTFT. `tp16` holds
  half the KV cache of `pp2tp8` because this model's multi-head latent attention keeps the same compressed KV cache
  on every tensor-parallel rank, so each layer's cache is stored 16 times in `tp16` and 8 times in `pp2tp8`.
- **In this round, placement mattered for `tp16` only**: with the far second instance, TPOT was 3.3 to 4.0% higher at every
  concurrency. For `pp2tp8`, the far runs were 0.3 to 2.0% faster than the near ones.

The run took 64 minutes after the model was staged (11.6 minutes for all three pods at once). Each vLLM start took
7 to 9 minutes, so the defaults (two rounds and 64 in flight as well) take about twice as long.

## Notes

- The pods run as a non-root user with a read-only root filesystem and no capabilities. vLLM's caches, the launch
  scripts and the logs go to `/tmp`, an `emptyDir`; the model goes to `/model`, a memory-backed `emptyDir` that is
  gone when the pod ends. Each pod ends itself after 4 hours (`activeDeadlineSeconds`).
- Each vLLM start took 7 to 9 minutes (560 seconds for the first, 440 for each later one), most of it kernel warm-up
  and CUDA graph capture.
- `kubectl exec` from outside the cluster can drop; the scripts run vLLM and the benchmark client detached inside the
  pods and retry their `kubectl exec` calls.
