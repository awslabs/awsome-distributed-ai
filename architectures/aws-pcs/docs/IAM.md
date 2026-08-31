# IAM Permissions Guide

The cluster distinguishes **two human roles** with very different
responsibilities. Each ships a CloudFormation template that creates the
customer-managed IAM policy and an IAM group with it attached (optionally
adding existing IAM users at deploy time).

| Role | Launch |
|---|---|
| **Cluster admin** — deploys/updates/deletes clusters. Full CRUD on CloudFormation, PCS, EC2 (VPC/SG/launch templates/placement groups/NAT/EIP), FSx, scoped IAM, SSM Parameter Store, KMS, Secrets Manager, and (optionally) Image Builder. **Broad — do not hand to every engineer.** | [![Launch](../images/launch-stack.svg)](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/aws-pcs/cluster-admin-iam.yaml&stackName=pcs-cluster-admins) |
| **Cluster user** — engineers running jobs on an existing cluster. SSM session to the **login node only**, port-forward Grafana, read the Grafana password, read PCS cluster/queue status. **No AWS-API mutations** (cannot create, modify, or delete cluster resources) and **no compute-node sessions**. The login-node SSM shell it grants runs as `ssm-user` with passwordless sudo (SSM default), so on the login node it is effectively root — meaning read access to shared `/home`, Slurm accounting, and the instance role's `/pcs/<id>/ldap/*` SSM secret; treat it as trusted-operator-scope, not arbitrary-viewer-scope. | [![Launch](../images/launch-stack.svg)](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/aws-pcs/cluster-user-iam.yaml&stackName=pcs-cluster-users) |

Both templates take an optional `AttachUsers` parameter (comma-separated
existing IAM user names) so you can wire up group membership at deploy time,
or add users later via the IAM console. The cluster-admin template's
`AttachImageBuilderPolicy` defaults to `false`; set it `true` only if the
admin will also deploy the standalone DLAMI builder
(`pcs-ready-dlami-with-enroot-pyxis.yaml`). The cluster-user template
requires `ClusterStackName` and scopes SSM session access to that one
cluster's login node — deploy one stack of it per cluster. That scoping
keys on the EC2 `Name` tag (`ssm:resourceTag/Name = <ClusterStackName>-login`),
which is operator-mutable: re-tagging a login node changes who can reach it,
and if another instance in the account happens to carry a matching `Name`
tag it would also satisfy the condition. Treat `Name` as part of this
policy's trust surface.

---

## Considerations

These are **sample, slightly-broader-than-strict-least-privilege** policies,
derived from the AWS-published
[minimum permissions for an AWS PCS service administrator](https://docs.aws.amazon.com/pcs/latest/userguide/security-min-permissions.html)
plus the extra permissions the all-in-one template needs because it provisions
VPC + FSx + IAM roles itself (the AWS reference policy assumes those already
exist). Review and tighten before production use.

**Pairing with AWS-managed policies.** For a smaller customer-managed
surface, attach AWS-managed policies for parts of the stack and trim the
matching statements: `AWSCloudFormationFullAccess`, `AmazonFSxFullAccess`,
`AWSImageBuilderFullAccess` are reasonable fits. Avoid
`AmazonEC2FullAccess` — it is materially overprivileged (e.g. EBS
public-share); prefer the customer-managed EC2 statements in the template.
There is no `AmazonPCSFullAccess`, so the PCS portion has to stay
customer-managed.

**Not covered by these policies:**
- The compute instance role (passed to EC2 by `cluster.yaml`) — use the
  AWS-managed `AWSPCSComputeNodePolicy`.
- The Image Builder build instance role — use the AWS-managed
  `EC2InstanceProfileForImageBuilder` /
  `EC2InstanceProfileForImageBuilderECRContainerBuilds`.
- Fine-grained per-cluster scoping — both policies use `Resource: "*"` for
  many EC2/VPC actions where resource-level scoping is limited. Sample-grade,
  deliberate.
