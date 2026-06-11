# Infrastructure: CPU instance group for the reward pool

These helpers add/remove a **CPU instance group** on an existing SageMaker
HyperPod (EKS) cluster to host the disaggregated reward service
(`kubernetes/reward-service.yaml`). The new group is placed in the **same
subnet/AZ** as the GPU pool and inherits the security group, execution role, and
lifecycle config from an existing CPU group, so it joins EKS identically.
**No EFA** is requested -- the reward RPC is low-bandwidth HTTP.

By default the group is created with **EC2 Spot capacity**
(`CapacityRequirements: { Spot: {} }`). The reward service is stateless and
fault-tolerant (SLIME's `remote_rm` client retries with backoff), so Spot is a
good fit and saves up to ~90% vs On-Demand. HyperPod handles interruptions by
tainting the node, gracefully evicting pods, and replacing capacity. Spot
requires `Continuous` node provisioning. Pass `CAPACITY_TYPE=on-demand` for an
On-Demand group instead. See
[Spot instances in HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-spot.html).

> `CapacityRequirements` is **immutable** after a group is created -- choose
> Spot vs On-Demand up front; it cannot be flipped later.

> HyperPod's `UpdateCluster` API replaces the *entire* `InstanceGroups` list.
> These scripts read the current cluster, add/remove the one group, and resubmit
> the full set so existing groups are preserved (including each group's existing
> Spot/On-Demand capacity setting).

## Prerequisites

- `awscli` configured with permissions for `sagemaker:DescribeCluster` and
  `sagemaker:UpdateCluster`
- `python3`
- An existing CPU ("general") instance group to use as the networking template
- For Spot: the cluster must use `Continuous` node provisioning, and you need
  EC2 Spot service quota for the chosen instance type

## Add a CPU instance group (Spot by default)

```bash
CLUSTER_NAME=<your-hyperpod-cluster> \
REGION=us-east-1 \
GROUP_NAME=reward-spot-c5 \
INSTANCE_TYPE=ml.c5.4xlarge \
INSTANCE_COUNT=4 \
CAPACITY_TYPE=spot \
infra/add_cpu_instance_group.sh
```

Variables (all optional except `CLUSTER_NAME`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CLUSTER_NAME` | _(required)_ | HyperPod cluster name |
| `REGION` | `us-east-1` | AWS region |
| `GROUP_NAME` | `reward-spot-c5` | Name for the new instance group |
| `INSTANCE_TYPE` | `ml.c5.4xlarge` | CPU instance type |
| `INSTANCE_COUNT` | `4` | Number of nodes |
| `CAPACITY_TYPE` | `spot` | `spot` or `on-demand` (immutable after create) |
| `TEMPLATE_GROUP` | _(auto)_ | Existing CPU group to copy subnet/SG/role/lifecycle from; auto-detected if unset |
| `EBS_GB` | `100` | Extra EBS volume size per node |

Provisioning takes ~10-15 min. Track it with:

```bash
aws sagemaker describe-cluster --region "$REGION" --cluster-name "$CLUSTER_NAME" \
  --query 'InstanceGroups[].[InstanceGroupName,CurrentCount,TargetCount,Status]' \
  --output table

# Nodes join EKS labeled sagemaker.amazonaws.com/instance-group-name=<GROUP_NAME>
kubectl get nodes -l node.kubernetes.io/instance-type=ml.c5.4xlarge
```

Then set `REWARD_NODE_GROUP=<GROUP_NAME>` (see `env_vars.disaggregated.example`)
so the reward pods schedule onto this pool.

## Remove the instance group (cleanup)

Stop the cost of the CPU pool when finished:

```bash
CLUSTER_NAME=<your-hyperpod-cluster> \
GROUP_NAME=reward-spot-c5 \
infra/remove_cpu_instance_group.sh
```
