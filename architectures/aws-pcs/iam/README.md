# IAM Sample Policies

## Overview

The PCS reference cluster has **two distinct human principals** with very
different responsibilities:

```
                 ┌──────────────────────────┐
   admin   ───►  │  cluster-admin           │ ──► CloudFormation, PCS, EC2 (VPC,
   (deploys      │  (CRUD on infra +        │     SG, LT, PG, EIP, NAT, Endpoint),
   the cluster)  │   PCS resources)         │     FSx, IAM (scoped), SSM Param,
                 └──────────────────────────┘     KMS, Secrets Manager, optional
                                                  Image Builder
                                                  (creates everything; can also delete)

                 ┌──────────────────────────┐
   user    ───►  │  cluster-user            │ ──► SSM StartSession to LOGIN node only
   (uses an      │  (read-only + SSM        │     (Name=PCS-login* tag scope), port
   existing      │   session to login)      │     forward Grafana 443→8443, read
   cluster)      └──────────────────────────┘     /pcs/*/grafana/admin-password,
                                                  read PCS cluster / queue status
                                                  (cannot create or modify anything)
```

Splitting like this means the deployer needs broad IAM but doesn't have to
share those credentials with every engineer who just wants to `srun`, and the
user role can be handed out widely (it can't accidentally delete the cluster
or open shells on compute nodes).

## Quick start — deploy as CloudFormation

The fastest way to provision both groups + the policies attached to them is
the two CloudFormation templates here. Each creates the customer-managed
policies, an IAM group, and (optionally) attaches existing IAM users to the
group. Click below to launch:

| Stack | What it creates | Launch |
|---|---|---|
| **Cluster admin** | Two managed policies (`*-PCSClusterAdmin-core` always; `*-PCSClusterAdmin-imagebuilder` if opted in) + an IAM group with both attached. Optional: add existing IAM users to the group. | [<kbd>🚀 Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/cluster-admin-iam.yaml&stackName=pcs-cluster-admins) |
| **Cluster user** | One managed policy (`*-PCSClusterUser`) + an IAM group with it attached. Optional: add existing IAM users to the group. | [<kbd>🚀 Deploy</kbd>](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/cluster-user-iam.yaml&stackName=pcs-cluster-users) |

> The Quick-create launch buttons above point at the production S3 bucket
> (`awsome-distributed-ai`). For an in-place sandbox test (e.g. before the
> templates are merged upstream), use the local `template-body` form:
>
> ```bash
> aws cloudformation create-stack \
>   --stack-name pcs-cluster-admins \
>   --template-body file://architectures/aws-pcs/assets/cluster-admin-iam.yaml \
>   --parameters ParameterKey=AttachUsers,ParameterValue=alice,bob \
>                ParameterKey=AttachImageBuilderPolicy,ParameterValue=true \
>   --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
> ```

### Files in this directory

| File | Purpose |
|---|---|
| [`cluster-admin-iam.yaml`](../assets/cluster-admin-iam.yaml) | CFN template: creates 2 managed policies + IAM group for the admin role |
| [`cluster-user-iam.yaml`](../assets/cluster-user-iam.yaml) | CFN template: creates 1 managed policy + IAM group for the user role |
| [`cluster-admin-policy.json`](./cluster-admin-policy.json) | Raw JSON — **core** admin policy (CFN, EC2, FSx, PCS, IAM-scoped, SSM Param, KMS, Secrets, Logs); fits inside the IAM 6,144-char per-policy limit |
| [`cluster-admin-imagebuilder-policy.json`](./cluster-admin-imagebuilder-policy.json) | Raw JSON — **optional** Image Builder admin policy. Attach only when deploying `pcs-ready-dlami-with-enroot-pyxis.yaml` |
| [`cluster-user-policy.json`](./cluster-user-policy.json) | Raw JSON — end-user policy (SSM session to login node + read-only) |

The CFN templates simply embed the JSON files as `PolicyDocument`. If you
prefer not to use CFN (e.g. CLI-only environment), the JSON files are
self-sufficient — see [Manual install without CFN](#manual-install-without-cfn).

### Verification

These templates were validated end-to-end on a real account (us-east-2) on
2026-06-10:

| Check | Result |
|---|---|
| Both CFN stacks `CREATE_COMPLETE` | ✅ |
| Test admin user can `cloudformation:CreateStack` for `pcs-ml-cluster-deploy-all.yaml`, all nested stacks `CREATE_COMPLETE` (~28 min) | ✅ |
| Test admin user can `cloudformation:UpdateStack` (parameter change → `UPDATE_COMPLETE`) | ✅ |
| Test admin user can `cloudformation:DeleteStack` (full teardown) | ✅ |
| `aws iam simulate-principal-policy` on `cfn:CreateStack`, `pcs:CreateCluster`, `ec2:RunInstances`, `fsx:CreateFileSystem`, `iam:CreateRole` (with proper `arn:aws:iam::*:role/AWSPCS-*` resource ARN) all `allowed` | ✅ |
| Test user can `pcs:GetCluster`, `cloudformation:DescribeStacks`, `ssm:GetParameter` on Grafana password | ✅ |
| Test user `ssm:StartSession` on login node (Name=`PCS-login`) `allowed` | ✅ |
| Test user `ssm:StartSession` on compute node (Name=`PCS-hpc8a`, etc.) `implicitDeny` | ✅ |
| Test user `cloudformation:CreateStack`, `pcs:CreateCluster`, `ec2:RunInstances` all `implicitDeny` | ✅ |
| CloudTrail review of admin user's deploy + update + delete cycle: 0 `AccessDenied` errors across 332+ API calls (CFN, EC2, FSx, PCS, IAM, SSM, KMS, Secrets) | ✅ |

`aws accessanalyzer validate-policy` against all three JSON files: 0 errors.

## Design intent

These are **sample, slightly broader-than-strict-least-privilege** policies for
reference, derived primarily from
[the AWS-published "minimum permissions for an AWS PCS service administrator"
policy](https://docs.aws.amazon.com/pcs/latest/userguide/security-min-permissions.html)
plus the additional permissions the all-in-one template needs because it
provisions VPC + FSx + IAM roles itself (the AWS reference policy assumes those
already exist).

Choices made:

- **Combined CRUD, not phase-split.** A single policy covers create + update +
  delete because CFN rollback on a failed Create requires Delete actions, and
  Update is operationally a strict superset of Create.
- **Customer-managed, not inline; admin policy split into core + Image Builder.**
  The combined admin policy is ~7.4 KB which exceeds **both** the IAM inline
  per-policy limit and the customer-managed per-policy limit (both are 6,144
  characters as of January 2026). Split into a 5,769-char core file and a
  1,674-char optional Image Builder file. Most users only need the core; attach
  Image Builder only when deploying `pcs-ready-dlami-with-enroot-pyxis.yaml`.
- **No `AmazonPCSFullAccess` available.** As of this writing AWS publishes
  `AWSPCSComputeNodePolicy` (for compute instances, not deployers) and
  `AWSPCSServiceRolePolicy` (the SLR's managed policy) but no
  service-administrator managed policy. The PCS portion of the admin policy
  must be customer-managed.
- **User policy scopes login-node access via the `Name` tag.** `ssm:StartSession`
  is conditioned on `ssm:resourceTag/Name` matching `PCS-login*`. PCS does not
  emit a dedicated "is this a login node" tag — the all-in-one templates set
  `Name=PCS-login` on the login CNG and `Name=PCS-<cng-name>` on compute CNGs
  (e.g. `PCS-cpu1`, `PCS-hpc8a`), so this is the most stable signal available.
  **The `Name` tag is operator-mutable**: if you re-tag a login node, update
  the policy condition to match — or fork the templates to add a dedicated
  `IsLoginNode=true` tag and switch to that.
- **Production hardening points.** Tighten with resource-level conditions
  (account ID, region, stack-name prefix, tag matchers) before using in
  production. Validation by deploying with a sandbox account and running
  CloudTrail / IAM Access Analyzer is recommended — see
  [Refining via CloudTrail](#refining-via-cloudtrail) below.

## What's NOT covered here

- **The compute instance role** (passed to EC2 via the launch template by
  `cluster.yaml`). That's a separate role provisioned by the templates
  themselves, not by these policies. Use the AWS-managed
  `AWSPCSComputeNodePolicy` for it.
- **The Image Builder build instance role** (used by the standalone DLAMI
  template's Image Builder pipeline). Use AWS-managed
  `EC2InstanceProfileForImageBuilder` /
  `EC2InstanceProfileForImageBuilderECRContainerBuilds` for that.
- **Fine-grained per-cluster scoping.** Both policies use `Resource: "*"` for
  many actions because resource-level scoping for VPC / EC2 networking
  resources is limited and tag-based filtering would require the templates to
  consistently propagate a known tag. This is a deliberate sample-grade choice.

---

## `cluster-admin-iam.yaml` — what it grants

The CFN template combines the core JSON + (optionally) the Image Builder JSON.
Sids in the JSON files:

| Sid | Source | Purpose |
|---|---|---|
| `CloudFormationStackLifecycle` + `CloudFormationDescribeStackless` | core | Create/update/delete the parent and all nested CFN stacks; ChangeSets; describe and list |
| `EC2NetworkingAndComputeLifecycle` | core | VPC / subnets / IGW / NAT / EIP / security groups / route tables / VPC endpoints / launch templates / placement groups; `RunInstances` / `CreateFleet` for PCS-internal compute launches |
| `EC2DescribeForPCS` | core | All `ec2:Describe*` (PCS' control plane and CFN both call many of these) |
| `FSxLifecycle` | core | FSx for Lustre + OpenZFS create / update / delete / describe / tag |
| `PCSFullAccess` | core | `pcs:*` — clusters, compute node groups, queues |
| `IAMRoleAndInstanceProfileLifecycle` | core | Create / delete / tag / attach for IAM roles + instance profiles whose names contain `PCS` / `pcs` / `ImageBuilder` |
| `IAMPassRoleToPCSAndEC2` | core | `iam:PassRole` scoped to `*PCS*` roles passed to `ec2.amazonaws.com` / `pcs.amazonaws.com` |
| `ServiceLinkedRolesOneTime` | core | `iam:CreateServiceLinkedRole` for PCS / Spot / FSx / Image Builder (no-op once the SLR exists) |
| `SSMParameterLifecycle` | core | Put / get / delete on `parameter/pcs/*` (Grafana admin password, etc.) and `parameter/aws/service/*` (PCS-Ready DLAMI auto-resolve) |
| `KMSAccessForDefaultEncryption` | core | KMS data-plane operations for default-encrypted EBS / FSx volumes (verbatim from AWS PCS minimum-permissions policy) |
| `SecretsManagerForPCS` | core | `secretsmanager:*` scoped to `secret:pcs!*` (PCS internally creates a Slurm cluster auth secret) |
| `CloudWatchLogsForMonitoring` + `PCSVendedLogsDelivery` | core | CloudWatch Logs delivery wiring for PCS-vended Slurm logs |
| `IAMPassRoleToImageBuilder` | imagebuilder | `iam:PassRole` scoped to `*ImageBuilder*` roles passed to `imagebuilder.amazonaws.com` |
| `ImageBuilderLifecycleOptional` | imagebuilder | EC2 Image Builder pipeline / recipe / component / distribution / lifecycle policy |

**Where it came from.** The `PCSFullAccess` + `EC2DescribeForPCS` +
`IAMPassRoleToPCSAndEC2` + `ServiceLinkedRolesOneTime` +
`KMSAccessForDefaultEncryption` + `SecretsManagerForPCS` blocks are derived
directly from
[`security-min-permissions.html`](https://docs.aws.amazon.com/pcs/latest/userguide/security-min-permissions.html).
The remaining blocks (CloudFormation, EC2 networking lifecycle, FSx, IAM role
lifecycle, SSM Parameter Store, Image Builder, CloudWatch Logs) are layered on
because the all-in-one template provisions VPC + FSx + IAM roles itself.

**A combined CRUD policy is intentional, not a mistake.** Splitting per phase
(deploy-only / update-only / delete-only) breaks in three places: (1) CFN
rollback on Create requires Delete actions, (2) Update is a strict superset
of Deploy (any UpdateStack may replace resources), (3) Drift detection during
Update calls Describe across every service. The natural alternative is
**operator policy = the CRUD JSON above** vs **auditor policy = the same
actions reduced to `*:Describe*` / `*:Get*` / `*:List*`**.

### Pairing with AWS-managed policies

If you want a smaller customer-managed surface, you can attach AWS-managed
policies for parts of the stack and trim the corresponding Sids:

| AWS-managed | Replaces / overlaps with | Notes |
|---|---|---|
| `AWSCloudFormationFullAccess` | `CloudFormationStackLifecycle` + `CloudFormationDescribeStackless` | Broader (no resource-level scoping) but battle-tested |
| `AmazonFSxFullAccess` | `FSxLifecycle` | Closest fit; covers everything the templates need |
| `AWSImageBuilderFullAccess` | `ImageBuilderLifecycleOptional` | Only needed when the standalone DLAMI template is deployed |
| `AmazonEC2FullAccess` | `EC2NetworkingAndComputeLifecycle` + `EC2DescribeForPCS` | **Materially overprivileged** (e.g. EBS public-share). Prefer the customer-managed Sids |

There is **no** `AmazonPCSFullAccess` AWS-managed policy as of this writing —
the PCS portion has to be customer-managed.

---

## `cluster-user-iam.yaml` — what it grants

Sids:

| Sid | Purpose |
|---|---|
| `DiscoverClusterAndLoginNode` | `ec2:DescribeInstances` + `cloudformation:DescribeStacks` so the user can find the login node and read stack outputs (e.g. `ClusterId`) |
| `PCSReadOnly` | `pcs:Get*` / `pcs:List*` so the user can see cluster / queue / compute-node-group status |
| `GrafanaPasswordRead` | `ssm:GetParameter` scoped to `parameter/pcs/*/grafana/*` (the admin password generated by `cluster.yaml` when `DeployMonitoring=true`) |
| `SSMSessionToLoginNode` | `ssm:StartSession` scoped to instances whose `Name` tag matches `PCS-login*`, with the SSM documents needed for shell, SSH-over-SSM, and port-forwarding |
| `SSMSessionDocumentLookup` | `ssm:DescribeSessions` / `DescribeInstanceInformation` etc. (the SSM client / Session Manager UI calls these to render the connection list) |
| `SSMSessionTerminateOwn` | `ssm:TerminateSession` / `ResumeSession` scoped to sessions whose ID prefix matches `${aws:username}-*` (= sessions the user themselves opened) |

The `SSMSessionToLoginNode` Sid uses **`ssm:resourceTag/Name`** to scope sessions
to login nodes only. The all-in-one templates set `Name=PCS-login` on the
login node group and `Name=PCS-<cng-name>` on compute CNGs (e.g.
`PCS-cpu1`, `PCS-hpc8a`, `PCS-gpu-p5`), so this works out of the box. **Tag
caveat**: the `Name` tag is operator-mutable; if you rename a login node,
update the policy condition to match.

### Typical user workflows the policy enables

1. **Connect to the login node:**
   ```bash
   INSTANCE_ID=$(aws ec2 describe-instances \
     --filters "Name=tag:Name,Values=PCS-login" \
               "Name=instance-state-name,Values=running" \
     --query 'Reservations[0].Instances[0].InstanceId' --output text)
   aws ssm start-session --target $INSTANCE_ID
   ```

2. **Open Grafana via SSM port-forward (443 → 8443):**
   ```bash
   aws ssm start-session --target $INSTANCE_ID \
     --document-name AWS-StartPortForwardingSession \
     --parameters '{"portNumber":["443"],"localPortNumber":["8443"]}'
   # then open https://localhost:8443/grafana/
   ```

3. **Retrieve the Grafana admin password** (run this once, paste into the Grafana
   login):
   ```bash
   CLUSTER_ID=<from-stack-output>
   aws ssm get-parameter --name "/pcs/${CLUSTER_ID}/grafana/admin-password" \
     --with-decryption --query 'Parameter.Value' --output text
   ```

4. **SSH-over-SSM** (with `~/.ssh/config` configured):
   ```bash
   ssh hpc8a-login
   ```

### Safety properties

- **No write access to cluster resources.** No `CreateStack`, no `pcs:Create*`,
  no `ec2:RunInstances`, no `iam:*`, no `fsx:*` write actions.
- **Login nodes only.** The `Name` tag scope on `ssm:StartSession` blocks the
  user from opening shells on compute nodes (which are tagged
  `Name=PCS-<cng-name>`, e.g. `PCS-cpu1` or `PCS-gpu-p5`).
- **Own sessions only.** `ssm:TerminateSession` is scoped to sessions whose ID
  matches the user's IAM username, so users can't kill each other's sessions.

---

## Manual install without CFN

If you'd rather skip CloudFormation, attach the JSON files directly:

```bash
# Cluster admin (split into core + optional Image Builder)
aws iam create-policy \
  --policy-name PCSClusterAdminCore \
  --policy-document file://architectures/aws-pcs/iam/cluster-admin-policy.json
aws iam create-policy \
  --policy-name PCSClusterAdminImageBuilder \
  --policy-document file://architectures/aws-pcs/iam/cluster-admin-imagebuilder-policy.json
aws iam attach-user-policy \
  --user-name <deployer-user> \
  --policy-arn arn:aws:iam::<account-id>:policy/PCSClusterAdminCore
# attach the Image Builder one only when needed:
aws iam attach-user-policy \
  --user-name <deployer-user> \
  --policy-arn arn:aws:iam::<account-id>:policy/PCSClusterAdminImageBuilder

# Cluster user
aws iam create-policy \
  --policy-name PCSClusterUser \
  --policy-document file://architectures/aws-pcs/iam/cluster-user-policy.json
aws iam attach-user-policy \
  --user-name <user> \
  --policy-arn arn:aws:iam::<account-id>:policy/PCSClusterUser
```

For a CFN deployment role pattern, attach to a role instead of a user.

---

## Refining via CloudTrail

These JSON files are sample-grade. The recommended refinement path:

1. **Deploy a representative cluster in a sandbox AWS account** with admin
   permissions on the deploying principal so nothing fails for spurious IAM
   reasons.
2. **Wait for the deploy to succeed**, then for `aws cloudformation delete-stack`
   to finish, so CloudTrail captures the full lifecycle.
3. **Generate a least-privilege policy from CloudTrail history** with IAM Access
   Analyzer:

   ```bash
   aws accessanalyzer start-policy-generation \
     --policy-generation-details principalArn=<deployer-user-or-role-arn> \
     --cloud-trail-details accessRole=<analyzer-role>,startTime=<deploy-start>,trails=[{cloudTrailArn=<trail-arn>}]
   # poll until JobStatus=SUCCEEDED
   aws accessanalyzer get-generated-policy --job-id <job-id> \
     --include-resource-placeholders --query 'GeneratedPolicyResult.GeneratedPolicies[0].Policy' \
     --output text > generated-policy.json
   ```

4. **Diff against the JSON files in this directory.** The Access Analyzer
   output tends to be *narrower* than the sample (it only lists actions actually
   called) but with `Resource: "*"` even when narrower scoping is possible —
   the sample's resource ARNs are usually keepers.

The same approach works for `cluster-user-policy.json` — exercise the user
workflows (start a session, port-forward Grafana, terminate the session) in a
sandbox with broad SSM perms, then narrow to what was actually called.
