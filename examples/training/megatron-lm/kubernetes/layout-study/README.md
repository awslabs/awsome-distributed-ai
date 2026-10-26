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
  [`../README.md`](../README.md) describes, pushed under a fixed tag (not `latest`).
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
   RUNTIME_PLACEHOLDER on `p5en.48xlarge`:

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
- **Step time.** Megatron-LM logs the step time from the last rank, which is on the second instance; the script reads
  every pod's log, drops the first logged interval (warm-up), and takes the median of the last eight (iterations 41
  to 80).

## Results

RESULTS_PLACEHOLDER

## Notes

- The pods run as a non-root user with a read-only root filesystem and no capabilities; the launch scripts, logs and
  caches go to `/tmp`, an `emptyDir`. Each pod ends itself after 4 hours (`activeDeadlineSeconds`).
- `EXTRA_ENV` adds environment variables to every rank, for example to try an NCCL setting:
  `EXTRA_ENV="NCCL_ALGO=allreduce:^tree" ./2.run-layouts.sh`. It overrides NCCL's choice for every all-reduce, so
  compare against a run without it.
- Placement differences between near and far pairs are small for training; compare them with the spread between
  rounds before reading anything into them.
