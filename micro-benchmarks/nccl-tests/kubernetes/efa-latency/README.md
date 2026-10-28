# EFA latency between instances (Amazon EKS)

[`efa-latency-matrix.yaml`](efa-latency-matrix.yaml) measures one-way EFA latency with `fi_pingpong` between every pair of `NUM_NODES` instances, plus a floor inside each instance (two EFA devices of one instance, shared memory off). Its launcher starts both ends of each measurement over the MPI Operator's SSH, 2,000 round trips at 8 B and at 1 KiB, in four rounds with the pair order shuffled per round. [`analyze_efa_latency.py`](analyze_efa_latency.py) drops the first round as warm-up and groups the pairs by how many [network-node layers](../../README.md#placing-the-workers-by-network-topology-amazon-eks) their nodes share.

The step runs the public [nccl-tests image](../../README.md#1-prepare-the-container-image-and-other-artifacts) for the EFA installer's libfabric (`fi_pingpong`, `fi_info`) and for the `sshd` that the MPI Operator needs. It uses none of the image's CUDA or NCCL, so the workers request no GPUs.

## Prerequisites

- An Amazon EKS cluster with the [EFA device plugin](https://docs.aws.amazon.com/eks/latest/userguide/node-efa.html) and the [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator), as in [Amazon EKS](../../README.md#amazon-eks).
- `NUM_NODES` instances of a type that supports Amazon EC2 topology, in one Availability Zone. EFA traffic can't cross zones, so the workers have a pod affinity on `topology.kubernetes.io/zone`. The first worker can go to either zone; if that zone has fewer than `NUM_NODES` free instances, the rest stay `Pending`.
- Topology labels on those nodes: the `kubectl get nodes -L` command in [Placing the workers by network topology](../../README.md#placing-the-workers-by-network-topology-amazon-eks) shows a value in every layer column. The analysis stops if a node has none.
- Idle instances: each worker holds every EFA device of its instance.
- `kubectl`, `envsubst` and Python 3 where you run the commands.

## Run

From this directory:

```bash
export INSTANCE_TYPE=p5en.48xlarge NUM_NODES=8 EFA_PER_NODE=16
envsubst '$INSTANCE_TYPE $NUM_NODES $EFA_PER_NODE' < efa-latency-matrix.yaml | kubectl apply -f -
kubectl wait --for=condition=Succeeded mpijob/efa-latency --timeout=60m
kubectl logs job/efa-latency-launcher > latency.log
kubectl get nodes -o json > nodes.json
python3 analyze_efa_latency.py latency.log nodes.json
envsubst '$INSTANCE_TYPE $NUM_NODES $EFA_PER_NODE' < efa-latency-matrix.yaml | kubectl delete -f -
```

On 8 instances the run takes about 35 minutes; the time grows with the square of `NUM_NODES`. If a worker isn't reachable 15 minutes after the launcher starts, for example because it stays `Pending`, the launcher exits with an error.

## Results

The analysis prints one CSV line per pair, with its mean, minimum and maximum over the rounds, then a summary per message size. On eight `p5en.48xlarge` in one Availability Zone, five layer-3 network nodes among them (one with three instances, one with two, three with one), the run took 30 minutes, every measurement returned, and the summary was:

```text
8 B, usec/xfer (mean of rounds per pair):
  floor (within an instance): 8 instances, mean 8.18 (8.02 to 8.36)
  share 3 layers: 4 pairs, mean 9.25 (9.17 to 9.33)
  share 2 layers: 24 pairs, mean 11.29 (10.82 to 11.86)
  3 -> 2 shared layers adds +2.04 us (+22%); every 3-layer pair faster than every 2-layer pair: True

1024 B, usec/xfer (mean of rounds per pair):
  floor (within an instance): 8 instances, mean 8.81 (8.57 to 9.02)
  share 3 layers: 4 pairs, mean 10.12 (9.95 to 10.23)
  share 2 layers: 24 pairs, mean 12.31 (11.73 to 13.02)
  3 -> 2 shared layers adds +2.19 us (+22%); every 3-layer pair faster than every 2-layer pair: True
```

Pairs that share more layers should be faster, and every pair slower than the floor. Before you read a difference between two layer counts, check that their ranges don't overlap; if they do, the difference is smaller than the spread between pairs on your instances. Each range is over per-pair means; the CSV lines before the summary give each pair's minimum and maximum over its rounds. The numbers above come from one Availability Zone, instance type and hour, so measure yours.
