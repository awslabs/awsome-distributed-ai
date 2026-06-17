# IAM Permissions Guide

The cluster distinguishes **two human roles** with very different
responsibilities, and ships a ready-to-deploy IAM policy stack for each:

| Role | Who | What they can do | Template | Deploy |
|---|---|---|---|---|
| **Cluster admin** | The person who deploys/updates/deletes the cluster | Full CRUD on the infrastructure: CloudFormation, PCS, EC2 (VPC/SG/launch templates/placement groups/NAT/EIP), FSx, scoped IAM, SSM Parameter Store, KMS, Secrets Manager, and (optionally) Image Builder | [`cluster-admin-iam.yaml`](../assets/cluster-admin-iam.yaml) | [<kbd>🚀 Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/cluster-admin-iam.yaml&stackName=pcs-cluster-admins) |
| **Cluster user** | Engineers who just run jobs on an existing cluster | SSM session **to the login node only**, port-forward Grafana, read the Grafana password, read PCS cluster/queue status. **Cannot create, modify, or delete anything**, and cannot open shells on compute nodes | [`cluster-user-iam.yaml`](../assets/cluster-user-iam.yaml) | [<kbd>🚀 Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/cluster-user-iam.yaml&stackName=pcs-cluster-users) |

Splitting the roles means the deployer's broad permissions never have to be
handed to every engineer who just wants to `srun`, and the user role can be
given out widely — it can't accidentally delete the cluster or get a shell on a
compute node.

---

## Deploying the policies

Each template creates the customer-managed IAM policies, an IAM group with them
attached, and (optionally) adds existing IAM users to that group. Deploy both
as CloudFormation stacks:

| Stack | Creates | 1-click |
|---|---|---|
| **Cluster admin** | `<stack>-PCSClusterAdmin-core` (always) + `<stack>-PCSClusterAdmin-imagebuilder` (if opted in) managed policies + an IAM group | [<kbd>🚀 Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/cluster-admin-iam.yaml&stackName=pcs-cluster-admins) |
| **Cluster user** | `<stack>-PCSClusterUser` managed policy + an IAM group | [<kbd>🚀 Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/cluster-user-iam.yaml&stackName=pcs-cluster-users) |

Or from the CLI (use `--template-body` against a local checkout for a pre-merge
sandbox test):

```bash
# Admin: create the policies + group, attach existing users, include Image Builder perms
aws cloudformation create-stack \
  --stack-name pcs-cluster-admins \
  --template-body file://architectures/aws-pcs/assets/cluster-admin-iam.yaml \
  --parameters ParameterKey=AttachUsers,ParameterValue=alice,bob \
               ParameterKey=AttachImageBuilderPolicy,ParameterValue=true \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM

# User: create the policy + group, attach existing users
aws cloudformation create-stack \
  --stack-name pcs-cluster-users \
  --template-body file://architectures/aws-pcs/assets/cluster-user-iam.yaml \
  --parameters ParameterKey=AttachUsers,ParameterValue=carol,dave \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

Both templates take an `AttachUsers` parameter (comma-separated existing IAM
user names) so you can wire up group membership at deploy time, or leave it
empty and add users to the group later. The admin template's
`AttachImageBuilderPolicy` defaults to `false`; set it `true` only when the
admin will also deploy the standalone DLAMI builder
(`pcs-ready-dlami-with-enroot-pyxis.yaml`).

### What the cluster user can do once attached

```bash
# Find the login node and open a session
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=PCS-login" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
aws ssm start-session --target $INSTANCE_ID

# Port-forward Grafana (443 -> 8443), then open https://localhost:8443/grafana/
aws ssm start-session --target $INSTANCE_ID \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["443"],"localPortNumber":["8443"]}'

# Read the Grafana admin password
aws ssm get-parameter --name "/pcs/<cluster-id>/grafana/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text
```

---

## Considerations

These are **sample, slightly-broader-than-strict-least-privilege** policies,
derived from the AWS-published
[minimum permissions for an AWS PCS service administrator](https://docs.aws.amazon.com/pcs/latest/userguide/security-min-permissions.html)
plus the extra permissions the all-in-one template needs because it provisions
VPC + FSx + IAM roles itself (the AWS reference policy assumes those already
exist). Review and tighten before production use.

**Login-node access is scoped by the `Name` tag.** The user policy conditions
`ssm:StartSession` on `ssm:resourceTag/Name` matching `PCS-login*`. PCS does not
emit a dedicated "is this a login node" tag, so the templates set
`Name=PCS-login` on the login node and `Name=PCS-<cng-name>` on compute nodes
(`PCS-cpu1`, `PCS-hpc8a`, …) — the most stable signal available. **The `Name`
tag is operator-mutable**: if you re-tag a login node, update the policy
condition to match (or fork the templates to add a dedicated `IsLoginNode=true`
tag and key off that).

**Combined CRUD is intentional, not a mistake.** The admin policy covers
create + update + delete in one policy because (1) CFN rollback on a failed
Create requires Delete actions, (2) UpdateStack is operationally a superset of
Create (it may replace resources), and (3) drift detection during Update calls
Describe across every service. If you want a read-only variant, reduce the same
actions to `*:Describe*` / `*:Get*` / `*:List*` for an auditor role.

**The admin policy is split into core + Image Builder** because the combined
document (~7.4 KB) exceeds the IAM 6,144-character per-policy limit. The
~5.8 KB core covers a normal deploy; the ~1.7 KB Image Builder add-on is only
needed for the standalone DLAMI builder. The CFN template attaches both to the
group when `AttachImageBuilderPolicy=true`.

**There is no `AmazonPCSFullAccess` managed policy** as of January 2026 — AWS
publishes only `AWSPCSComputeNodePolicy` (for compute instances) and
`AWSPCSServiceRolePolicy` (the service-linked role). The PCS portion of the
admin policy must therefore be customer-managed.

**Pairing with AWS-managed policies.** For a smaller customer-managed surface
you can attach AWS-managed policies for parts of the stack and trim the matching
statements: `AWSCloudFormationFullAccess`, `AmazonFSxFullAccess`,
`AWSImageBuilderFullAccess` are reasonable fits. Avoid `AmazonEC2FullAccess` —
it is materially overprivileged (e.g. EBS public-share); prefer the
customer-managed EC2 statements in the template.

### Not covered by these policies

- **The compute instance role** (passed to EC2 by `cluster.yaml`) — provisioned
  by the templates themselves; use the AWS-managed `AWSPCSComputeNodePolicy`.
- **The Image Builder build instance role** — use the AWS-managed
  `EC2InstanceProfileForImageBuilder` /
  `EC2InstanceProfileForImageBuilderECRContainerBuilds`.
- **Fine-grained per-cluster scoping** — both policies use `Resource: "*"` for
  many EC2/VPC actions because resource-level scoping there is limited. This is
  a deliberate sample-grade choice.

### Refining to least-privilege via CloudTrail

To generate a tighter policy from real usage:

1. Deploy a representative cluster in a sandbox account with broad permissions on
   the deploying principal (so nothing fails for spurious IAM reasons).
2. Let the full lifecycle run — deploy, then `delete-stack` — so CloudTrail
   captures every API call.
3. Generate a policy from CloudTrail with IAM Access Analyzer
   (`aws accessanalyzer start-policy-generation` → `get-generated-policy`), then
   diff against the template's statements. Access Analyzer output is usually
   *narrower* on actions but leaves `Resource: "*"`; the template's resource ARNs
   are usually the keepers.

The same approach works for the user policy — exercise the user workflows
(start a session, port-forward Grafana, terminate it) in a sandbox, then narrow.

---

## Verifying the policies

To confirm the admin policy can deploy a cluster end-to-end and the user policy is
correctly constrained (login-only SSM, no LDAP-password access), see the reproducible
procedure in [tests/iam-test.md](../tests/iam-test.md).

> **Note on the template-source bucket.** The admin policy grants no `s3:GetObject`,
> because the production templates live in the public `awsome-distributed-ai`
> bucket (CFN fetches `--template-url` anonymously). If you host the templates in
> a **private** bucket, the deploying principal additionally needs `s3:GetObject`
> on that bucket — grant it separately; it is intentionally out of the
> cluster-admin policy.
