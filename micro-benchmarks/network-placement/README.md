<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Network placement micro-benchmarks for Amazon EC2 instance topology

Amazon EC2 describes its network as [layers of network nodes](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-topology-works.html), and Amazon EKS labels each node of a supported instance type with the network nodes above it: `topology.k8s.aws/network-node-layer-1` (top) through `topology.k8s.aws/network-node-layer-3` (closest to the instance). Two instances under the same layer-3 network node are closest. These scripts measure what that placement means for your instances, on Amazon EKS, with nothing but pods and `kubectl exec`:

| Step | Script | What it measures |
|---|---|---|
| 0 | [`0.topology-groups.sh`](0.topology-groups.sh) | How your nodes, and the probe pods on them, group by network node |
| 1 | [`1.efa-latency-matrix.sh`](1.efa-latency-matrix.sh) | One-way EFA latency (`fi_pingpong`) between every pair of instances, plus a within-host floor on each, grouped by how many network-node layers the pair shares |
| 2 | [`2.rank-order.sh`](2.rank-order.sh) | NCCL collective times across instances in a topology-sorted node order and in an interleaved one, where NCCL's ring crosses between layer-3 network nodes on more hops |
| 3 | [`3.layer3-affinity.sh`](3.layer3-affinity.sh) | Whether the scheduler keeps pods under one layer-3 network node with plain pod affinity |

Every multi-instance NCCL run passes through [`check_efa_transport.sh`](check_efa_transport.sh), which discards the run unless every NCCL process reports `Using network Libfabric` and none reports the TCP socket fallback. NCCL falls back to sockets without an error when it can't load its network plugin, so an unchecked timing may not be an EFA timing at all.

## Prerequisites

- An Amazon EKS cluster with at least two GPU instances of a type that supports [EC2 instance topology](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-topology-prerequisites.html), in one Availability Zone, with the NVIDIA device plugin and the [EFA device plugin](https://docs.aws.amazon.com/eks/latest/userguide/node-efa.html). Step 2 needs at least four instances spread over at least two layer-3 network nodes, with two or more instances under each of two of them, so that the two node orders differ. Step 3 needs two instances under one layer-3 network node.
- `kubectl` access that can create pods and use `pods/exec` in one namespace. No cluster-admin rights are needed.
- `python3` (3.8 or later, standard library only) and `bash` on the workstation that runs the scripts.

The probe pods hold whole instances for as long as they run (2 hours at most, set by `activeDeadlineSeconds`). Delete them as soon as you're done, as shown in [Clean up](#clean-up).

The pods run as a non-root user (UID 1000) with a read-only root filesystem, no Linux capabilities, no privilege escalation, and the runtime's default seccomp profile; `/tmp` and `/dev/shm` are `emptyDir` volumes. EFA and NCCL need none of the dropped privileges, provided the EFA devices are readable by every user (`ls -l /dev/infiniband` shows `crw-rw-rw-` on Amazon EKS nodes with the EFA device plugin).

## Tested configuration

| Component | Version |
|---|---|
| Instances | 5 x p5en.48xlarge (8 x NVIDIA H200, 16 EFA interfaces each), one Availability Zone |
| Amazon EKS | Kubernetes 1.34 |
| Probe image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-efa`: EFA installer 1.47.0 (libfabric 2.4.0amzn1.0), aws-ofi-nccl 1.18.0, NCCL 2.28.9, PyTorch 2.11.0 |

Any image with libfabric's EFA provider (`fi_pingpong`, `fi_info`), PyTorch with NCCL, and aws-ofi-nccl works. Set `FI_BIN`, `NET_PLUGIN`, and `TUNER_PLUGIN` if your image installs them elsewhere.

## Step 0: Start the probe pods and read the topology

Clone the repository and start one probe pod per instance. Set `parallelism` and `completions` in [`kubernetes/probe-pods.yaml`](kubernetes/probe-pods.yaml) to the number of instances you want to measure, and the instance type and GPU and EFA counts for yours:

```bash
git clone https://github.com/awslabs/awsome-distributed-ai.git
cd awsome-distributed-ai/micro-benchmarks/network-placement
export NAMESPACE=default
kubectl -n $NAMESPACE apply -f kubernetes/probe-pods.yaml
kubectl -n $NAMESPACE wait --for=condition=Ready pod -l app=placement-probe --timeout=20m
INSTANCE_TYPE=p5en.48xlarge ./0.topology-groups.sh
```

To measure particular nodes, add a node affinity on `kubernetes.io/hostname` to the pod template, or label the nodes and add a `nodeSelector`.

Network node IDs are printed as short names (`L3-a`, `L3-b`, ...), assigned in a fixed order for the same set of nodes. Output on the tested cluster:

```text
== Nodes of type p5en.48xlarge
L1-a L2-a L3-a ip-10-0-1-11.compute.internal
L1-a L2-a L3-b ip-10-0-1-12.compute.internal
L1-a L2-a L3-b ip-10-0-1-13.compute.internal
L1-a L2-a L3-b ip-10-0-1-14.compute.internal
L1-a L2-a L3-c ip-10-0-1-15.compute.internal
L1-a L2-a L3-d ip-10-0-1-16.compute.internal
L1-a L2-a L3-e ip-10-0-1-17.compute.internal

7 nodes in 5 layer-3 groups (L3-a: 1, L3-b: 3, L3-c: 1, L3-d: 1, L3-e: 1)

== Running probe pods (app=placement-probe)
L1-a L2-a L3-b placement-probe-1-xc2jz ip-10-0-1-12.compute.internal
L1-a L2-a L3-b placement-probe-3-5w4bj ip-10-0-1-14.compute.internal
L1-a L2-a L3-b placement-probe-4-sqcpp ip-10-0-1-13.compute.internal
L1-a L2-a L3-c placement-probe-2-rs57p ip-10-0-1-15.compute.internal
L1-a L2-a L3-e placement-probe-0-b27fp ip-10-0-1-17.compute.internal

5 nodes in 3 layer-3 groups (L3-b: 3, L3-c: 1, L3-e: 1)
```

## Step 1: Measure what a network layer costs

```bash
./1.efa-latency-matrix.sh
python3 analyze_latency.py results/latency-<timestamp>
```

The script uses the same EFA device name on every host, runs each pair four times per message size (8 and 1,024 bytes by default) and discards the first round, and shuffles the pair order with a fixed seed so near and far pairs interleave in time. On every host it first measures a floor between two of the host's own EFA devices, with `FI_EFA_ENABLE_SHM_TRANSFER=0` so that the traffic stays on EFA, so a slow host can't pose as a far one. Output on the tested cluster:

```text
8-byte messages, one-way latency (us)
  within one host (floor): 5 pods, mean 8.30 (8.16 to 8.57 by pod)
  same layer-3 node: 3 pairs, mean 9.23 (9.09 to 9.31)
  different layer-3 node, same layer 2: 7 pairs, mean 11.36 (10.98 to 11.66)
  every same-layer-3 pair faster than every other pair: True (gap between bands +1.67 us); crossing layer 3 adds 2.12 us (23%)

1024-byte messages, one-way latency (us)
  within one host (floor): 5 pods, mean 8.91 (8.65 to 9.20 by pod)
  same layer-3 node: 3 pairs, mean 10.30 (10.27 to 10.31)
  different layer-3 node, same layer 2: 7 pairs, mean 12.37 (11.84 to 12.79)
  every same-layer-3 pair faster than every other pair: True (gap between bands +1.53 us); crossing layer 3 adds 2.07 us (20%)
```

## Step 2: Compare node orders

NCCL numbers instances in the order their ranks appear and builds its ring between instances in that order. The script builds two orders of the probe pods from the labels, a sorted one and an interleaved one, launches `torchrun` with the static rendezvous and an explicit `--node-rank` (the c10d rendezvous assigns node ranks itself and ignores `--node-rank`), and times all-reduce, all-gather, and all-to-all from 8 bytes to 4 GiB per rank. The two orders alternate within each round, and a run counts only if [`check_efa_transport.sh`](check_efa_transport.sh) confirms EFA on every process:

```bash
PODS=<pod-a>,<pod-b>,<pod-c>,<pod-d> ./2.rank-order.sh     # or leave PODS unset to use every probe pod
python3 analyze_rank_order.py results/rank-order-<timestamp>
```

`orders.txt` records how many ring hops between instances cross a layer-3 network node in each order. To check that each order was kept, `node-ranks.txt` lists the lowest NCCL rank on each pod for each run (node rank times GPUs per node), and `rings.txt` NCCL's ring in rank order. Output on the tested cluster:

```text
sorted 3 4 placement-probe-1-xc2jz placement-probe-4-sqcpp placement-probe-2-rs57p placement-probe-0-b27fp
interleaved 4 4 placement-probe-1-xc2jz placement-probe-2-rs57p placement-probe-4-sqcpp placement-probe-0-b27fp
layer3 placement-probe-1-xc2jz=L3-b placement-probe-4-sqcpp=L3-b placement-probe-2-rs57p=L3-c placement-probe-0-b27fp=L3-e

# 8 of 8 runs kept (EFA confirmed on 32 of 32 processes in each); python3 analyze_rank_order.py, selected sizes:
coll,bytes,sorted_us,interleaved_us,ratio_median,ratio_min,ratio_max,sorted_busbw_GBps,interleaved_busbw_GBps,rounds
all_gather,33554432,326.33,241.96,0.7428,0.6741,1.3607,99.79,134.35,4
all_gather,1073741824,3244.11,3246.67,1.0002,0.9961,1.0056,320.64,320.39,4
all_gather,4294967296,11812.10,11806.63,0.9993,0.9977,1.0008,352.25,352.41,4
all_reduce,33554432,448.50,452.34,1.0119,1.0056,1.0183,144.95,143.72,4
all_reduce,1073741824,5984.61,5989.89,1.0005,0.9972,1.0033,347.62,347.32,4
all_reduce,4294967296,22644.08,22634.39,0.9994,0.9987,1.0009,367.49,367.65,4
all_to_all,33554432,722.21,741.80,1.0093,0.9806,1.1063,45.01,43.82,4
all_to_all,1073741824,18529.25,18352.03,0.9904,0.9868,0.9946,56.14,56.68,4
all_to_all,4294967296,72552.34,72410.71,0.9976,0.9963,0.9987,57.35,57.46,4
```

## Step 3: Keep a group of pods under one layer-3 network node

[`kubernetes/layer3-affinity.yaml`](kubernetes/layer3-affinity.yaml) pins an anchor pod to one node and asks for a follower on a different node under the same layer-3 network node, with pod affinity on `topology.k8s.aws/network-node-layer-3` and pod anti-affinity on `kubernetes.io/hostname`. The pods request 100 millicores, so they fit beside the probe pods:

```bash
./3.layer3-affinity.sh
```

Output on the tested cluster:

```text
anchor   ip-10-0-1-12.compute.internal
follower ip-10-0-1-14.compute.internal
different host: True; same network node at layers 1, 2, 3: [True, True, True]
```

For multi-pod workloads, put the same pod affinity rule in the worker template. With [Kueue's topology-aware scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/), a cluster administrator defines a `Topology` whose levels are the three network-node labels and the host name, and a workload asks for placement with a pod-set annotation such as `kueue.x-k8s.io/podset-required-topology: topology.k8s.aws/network-node-layer-3`. This directory doesn't create Kueue objects, because that needs cluster-admin rights.

## Interpreting the results

- **Latency:** pairs under one layer-3 network node should be faster than pairs that differ at layer 3, and every pair should be slower than the floor. If a pair is slower than its label suggests, repeat the run before you trust either number. If two layers of pairs overlap, the difference between them is smaller than the run-to-run spread on your instances.
- **Rank order:** a ratio of interleaved to sorted time near 1 means the extra layer-3 crossings didn't cost anything measurable for that collective and size. Small messages have the largest round-to-round spread; compare the ratio's range to its median before you read a difference into it.
- Only compare numbers taken on the same instances, close together in time. A replaced node is a different instance, possibly under a different network node, so read the labels again.

## Known issues

- `kubectl exec` from a workstation can fail with a transient TLS handshake timeout. A failed latency measurement is recorded as `NA` and left out of the analysis; a failed rank-order run is discarded. Run the scripts from a host close to the cluster API endpoint if this happens often.
- EFA hardware counters (`/sys/class/infiniband/*/ports/1/hw_counters`) might not be visible inside pods. To monitor EFA traffic on Amazon EKS, run [efa-node-exporter](../../observability/efa-node-exporter) on the hosts.
- If `0.topology-groups.sh` shows nodes without labels, the instance type might not support instance topology. See the [prerequisites](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-topology-prerequisites.html).
- For Slurm clusters, see [topology-aware-nccl-tests](../nccl-tests/slurm/topology-aware-nccl-tests), which sorts the hostfile with `ec2:DescribeInstanceTopology`.

## Clean up

```bash
kubectl -n $NAMESPACE delete job placement-probe
```

`3.layer3-affinity.sh` deletes its own pods when it exits.
