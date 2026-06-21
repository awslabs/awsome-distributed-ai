<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
-->

# Prerequisite — provision the B300 node (self-managed EKS only)

This is the prerequisite for deploying
[`manifests/dsv4pro-deploy.yaml`](./manifests/dsv4pro-deploy.yaml)
on a **self-managed (eksctl) EKS** cluster.

> Skip this entirely on **HyperPod-on-EKS** — HyperPod provisions the nodes for
> you. Go straight to the [README Deploy section](./README.md#deploy).

## What it does

[`manifests/nodegroup-b300-eks.yaml`](./manifests/nodegroup-b300-eks.yaml)
creates the B300 spot nodegroup: a single 8-GPU `p6-b300.48xlarge` node that is

- **labeled** `nvidia.com/gpu.product: B300` (used by the verify commands below;
  `dsv4pro-deploy.yaml` itself schedules on the `p6-b300.48xlarge`
  `node.kubernetes.io/instance-type` label, which kubelet sets automatically);
- **tainted** `nvidia.com/gpu=true:NoSchedule` — which `dsv4pro-deploy.yaml`
  tolerates (keeps non-GPU pods off the node);
- set up to **auto-mount its 8×3.7TB local NVMe** as a ~28TB RAID0 at
  `/mnt/k8s-disks/0` on every (re)launch, via
  `preBootstrapCommands: ["/usr/bin/setup-local-disks raid0"]`.

## Create

```bash
# Fill in the <PLACEHOLDERS> (cluster name, region, VPC/subnets/SG) first.
eksctl create nodegroup -f manifests/nodegroup-b300-eks.yaml
```

`eksctl create nodegroup` builds a CloudFormation stack → EKS managed nodegroup
→ spot ASG, then issues the spot request (~3-5 min). If it hangs in creation,
the AZ most likely has no B300 spot capacity right now.

## Verify

```bash
# Node Ready, and confirm it really is SPOT capacity
kubectl get nodes -l nvidia.com/gpu.product=B300 -L eks.amazonaws.com/capacityType

# NVMe RAID0 is mounted in the HOST mount namespace (not just a helper pod's)
NODE=$(kubectl get nodes -l nvidia.com/gpu.product=B300 -o jsonpath='{.items[0].metadata.name}')
kubectl debug node/$NODE -it --image=public.ecr.aws/amazonlinux/amazonlinux:2023 \
  -- grep k8s-disks /host/proc/1/mountinfo
# Expect a line: /dev/md127 ... /mnt/k8s-disks/0 ... xfs
```

## Why `preBootstrapCommands` (and not a manual mount)

`preBootstrapCommands` run in cloud-init, **in the host mount namespace, before
kubelet starts**. So the RAID0 is ready when pods land, and it is recreated
automatically when the spot node is reclaimed and replaced — the workload pod
always finds its disk. `setup-local-disks` also redirects the kubelet/containerd
data dirs onto the NVMe, keeping the 500GB root EBS free.

Mounting **after** the node is up (e.g. `nsenter`/`chroot` from a privileged
pod) is error-prone and not persistent:

- `chroot /host mount ...` only changes the root dir, **not** the mount
  namespace — the mount lives in the helper pod's namespace and vanishes when
  the pod is deleted. The host's real namespace never sees it, so the
  workload's `hostPath` silently lands on the **root disk** and can fill it.
- The correct manual form is `nsenter -t 1 -m -- mount ...` (enter the host
  PID1 mount namespace), but it still doesn't survive a node replacement.

Use `preBootstrapCommands` for anything persistent; reserve `nsenter` for
one-off fixes on an already-running node.

> Note: `preBootstrapCommands` is only injected by eksctl on the stock AL2023
> AMI (no custom launch template). With a custom AMI / launch template, bake the
> `setup-local-disks raid0` call into your own userData instead.

## Tear down

```bash
# --drain=false skips the eviction drain (a single-node cluster can deadlock on
# the coredns PodDisruptionBudget otherwise).
eksctl delete nodegroup --cluster <CLUSTER_NAME> --name b300-spot \
  --region <REGION> --drain=false --wait
```
