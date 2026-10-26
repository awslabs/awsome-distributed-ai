<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Which parallelism dimension should cross between instances? A Megatron-LM layout study on Amazon EKS

When a training job spans two instances, part of every step waits on the network, and how much depends on which
parallelism dimension crosses between the instances. This study trains the same GPT model with Megatron-LM in six
layouts on two 8-GPU instances, with the same global batch (and so the same work per step) in every one, and reports
the step time and throughput of each:

| Layout | Instances | TP | PP | DP | What crosses between the instances |
| --- | --- | --- | --- | --- | --- |
| `tp8-dp2` | 2 | 8 | 1 | 2 | Gradients, once per step |
| `tp8-pp2` | 2 | 8 | 2 | 1 | Activations between the two pipeline stages |
| `tp16` | 2 | 16 | 1 | 1 | Every layer's tensor-parallel all-reduces |
| `tp8-split` | 2 (4 GPUs each) | 8 | 1 | 1 | Every layer's all-reduces, for one TP 8 group split 4 + 4 |
| `tp8` | 1 | 8 | 1 | 1 | Nothing: the baseline for `tp8-split` |
| `tp8-half` | 1 | 8 | 1 | 1 | Nothing: `tp16`'s per-GPU work (half the attention heads, feed-forward width and vocabulary) on one instance, the baseline for `tp16` |

`tp8-half` separates the two costs of `tp16`: each GPU computes on thinner slices of every layer and makes twice as
many all-reduces, which costs time even over NVLink, and part of those all-reduces cross EFA. `tp16` minus `tp8-half`
is the crossing alone; `tp8-split` minus `tp8` is the crossing for a TP 8 group.

Two-instance layouts also run with two placements of the second instance, chosen from the network-node labels that
Amazon EKS puts on nodes of
[instance types that support EC2 instance topology](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-topology.html)
(`topology.k8s.aws/network-node-layer-1` to `-3`): *near* shares all three network-node layers with the first
instance, *far* shares layers 1 and 2 and differs at layer 3. The default arms run the placement-sensitive cases on
both.

The model is a GPT with 24 layers, hidden size 4,096 and 32 attention heads (about 5 billion parameters), trained in
BF16 on Megatron-LM's mock data for 80 iterations at a global batch of 16 sequences, so no dataset or shared file
system is needed. The model fits on one GPU; it is sized so that communication is a large share of each step and the
layouts differ clearly. Set `NUM_LAYERS`, `HIDDEN_SIZE`, `NUM_HEADS`, `FFN_HIDDEN_SIZE`, `VOCAB_SIZE`,
`GLOBAL_BATCH` and `ARMS` to price your own model.

## Prerequisites

- An Amazon EKS cluster with three 8-GPU EFA instances (tested on `p5en.48xlarge`), the
  [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) and the
  [EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin). Two of the three
  nodes must share a layer-3 network node; `0.choose-nodes.sh` shows whether they do.
- The image built from [`../../aws-megatron-lm.Dockerfile`](../../aws-megatron-lm.Dockerfile) as
  [`../README.md`](../README.md) describes, pushed under a fixed tag rather than the `latest` that README uses (for
  example `export REPO_URI=${REGISTRY}/${ECR_REPOSITORY_NAME}:core_v0.17.0`), so every pod runs the same build.
- `kubectl`, `envsubst` and `python3` (3.8 or later) where you run the scripts.

No training operator is needed: the pods only sleep, and the scripts start `torchrun` in them with its static
rendezvous and an explicit `--node-rank`, so rank 0 is always on `megatron-head` and each layout's node order is the
one the script chose. (The c10d rendezvous backend orders nodes by host name and ignores `--node-rank`.)

## Run

1. Configure the run and pick the nodes:

   ```bash
   cp env_vars.example env_vars
   ./0.choose-nodes.sh            # prints the layer-3 groups and suggested export lines
   $EDITOR env_vars               # set IMAGE and NAMESPACE; paste HEAD_NODE, NEAR_NODE and FAR_NODE
   source env_vars
   ```

2. Create the three pods (`megatron-head`, `megatron-near`, `megatron-far`):

   ```bash
   ./1.create-pods.sh
   ```

3. Train in every arm. `ARMS` lists `layout:placement:sequence-length` (placement `near`, `far`, or `one` for the
   one-instance layouts); the default list runs 10 arms in two rounds, the second in reverse order, which took about
   70 minutes on `p5en.48xlarge`:

   ```bash
   ./2.run-layouts.sh             # results in ./results/layouts-<UTC time>
   python3 3.analyze.py results/layouts-<UTC time>
   ```

4. Delete the pods:

   ```bash
   kubectl -n $NAMESPACE delete pod -l app=megatron-layouts
   ```

## What the scripts check

- **Placement.** `2.run-layouts.sh` reads the pods' nodes and their labels and stops unless `megatron-near` shares
  three network-node layers with `megatron-head` and `megatron-far` shares two.
- **Transport.** A two-instance run counts only if every rank's process on both instances logs
  `Using network Libfabric` (the aws-ofi-nccl plugin over EFA) and none logs `Using network Socket`; otherwise its
  `transport` column says `NOT_PROVEN` and `3.analyze.py` leaves it out. NCCL falls back to TCP sockets without failing
  when it can't load the plugin, so the script sets `NCCL_NET_PLUGIN` to the plugin's path in the image
  (`NET_PLUGIN` to override).
- **Step time.** Megatron-LM logs the step time from the last rank, which is on the last instance; the script reads
  every pod's log, drops the first logged interval (warm-up), and takes the median of the last eight (iterations 41
  to 80).

## Results

Two rounds of the default arms on three `p5en.48xlarge` in one Availability Zone (the near pair under one layer-3
network node, the far instance under another), with the image `aws-megatron-lm.Dockerfile` builds (PyTorch
2.11.0a0 from NGC 26.02, NCCL 2.29.2, Transformer Engine 2.12.0, Megatron-LM `core_v0.17.0`, EFA installer 1.48.0,
aws-ofi-nccl 1.19.0). All 20 runs counted; every two-instance run had all of its ranks on EFA. Medians over the two
rounds, from `3.analyze.py`:

| Layout | Second instance | Tokens per sequence | Step time | Tokens/s |
| --- | --- | --- | --- | --- |
| `tp8-dp2` | near | 4,096 | 850.7 ms | 77,040 |
| `tp8-pp2` | near | 4,096 | 881.6 ms | 74,336 |
| `tp16` | near | 4,096 | 1,567.9 ms | 41,798 |
| `tp16` | far | 4,096 | 1,586.8 ms | 41,311 |
| `tp8-half` | none | 4,096 | 1,535.1 ms | 42,691 |
| `tp8-split` | near | 4,096 | 1,594.2 ms | 41,110 |
| `tp8` | none | 4,096 | 1,564.8 ms | 41,886 |
| `tp8-split` | near | 8,192 | 2,559.1 ms | 51,219 |
| `tp8-split` | far | 8,192 | 2,557.7 ms | 51,246 |
| `tp8` | none | 8,192 | 1,971.0 ms | 66,501 |

- **Tensor parallelism across the instances cost 45.7% of the throughput** of data parallelism across them (`tp16`
  against `tp8-dp2`), and little of that is the network: `tp8-half`, the same per-GPU work on one instance, was only
  2.1% faster than `tp16`. The crossing was 5% of `tp16`'s extra step time; the thinner slices of every layer were
  the rest.
- **Pipeline parallelism across the instances cost 3.5%.**
- **Splitting one TP 8 group across the instances cost 1.9% at 4,096 tokens and 23.0% at 8,192**, where each
  all-reduce carries twice the data.
- **Placement stayed within the noise**: `tp16` was 1.2% slower on the far pair, `tp8-split` at 8,192 tokens 0.1%
  faster, while the two rounds of the same arm differed by up to 3.0%.

## Notes

- The pods run as a non-root user with a read-only root filesystem and no capabilities; the launch scripts, logs and
  caches go to `/tmp`, an `emptyDir`. Each pod ends itself after 4 hours (`activeDeadlineSeconds`).
- `EXTRA_ENV` adds environment variables to every rank, for example to try an NCCL setting:
  `EXTRA_ENV="NCCL_ALGO=allreduce:^tree" ./2.run-layouts.sh`. It overrides NCCL's choice for every all-reduce, so
  compare against a run without it.
- Placement differences between near and far pairs are small for training; compare them with the spread between
  rounds before reading anything into them.
