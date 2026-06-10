# IAM Sample Policies

## Overview

The PCS reference cluster has **two distinct human principals** with very
different responsibilities:

```
                 ┌──────────────────────────┐
   admin   ───►  │  cluster-admin-policy    │ ──► CloudFormation, PCS, EC2 (VPC,
   (deploys      │  (CRUD on infra +        │     SG, LT, PG, EIP, NAT, Endpoint),
   the cluster)  │   PCS resources)         │     FSx, IAM (scoped), SSM Param,
                 └──────────────────────────┘     KMS, Secrets Manager, Image Builder
                                                  (creates everything; can also delete)

                 ┌──────────────────────────┐
   user    ───►  │  cluster-user-policy     │ ──► SSM StartSession to LOGIN node only,
   (uses an      │  (read-only + SSM        │     port-forward Grafana 443→8443,
   existing      │   session to login)      │     read /pcs/*/grafana/admin-password,
   cluster)      └──────────────────────────┘     read PCS cluster / queue status
                                                  (cannot create or modify anything)
```

Splitting like this means the deployer needs broad IAM but doesn't have to
share those credentials with every engineer who just wants to `srun`, and the
user role can be handed out widely (it can't accidentally delete the cluster
or open shells on compute nodes).

| File | Principal | Lifecycle | Size | Statements |
|---|---|---|---:|---:|
| `cluster-admin-policy.json` | Deploying user / role | Create / update / delete cluster | ~7.4 KB | 16 |
| `cluster-user-policy.json` | Cluster end-users (ML engineers) | Connect / read-only on existing cluster | ~1.5 KB | 6 |

Both files are valid identity-based JSON policies that have passed
`aws accessanalyzer validate-policy` cleanly (0 findings).

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
  Update is operationally a strict superset of Create. See the
  [`cluster-admin-policy.json` section below](#cluster-admin-policyjson--deploy--update--delete)
  for the full reasoning.
- **Customer-managed, not inline.** The admin policy is ~7.4 KB which exceeds
  the 6,144-character limit for inline policies but fits comfortably under the
  17,408-character limit for customer-managed policies. Attach as a
  customer-managed policy.
- **No `AmazonPCSFullAccess` available.** As of this writing AWS publishes
  `AWSPCSComputeNodePolicy` (for compute instances, not deployers) and
  `AWSPCSServiceRolePolicy` (the SLR's managed policy) but no
  service-administrator managed policy. The PCS portion of the admin policy
  must be customer-managed.
- **User policy is tag-scoped, not name-scoped.** `ssm:StartSession` is
  conditioned on `ssm:resourceTag/aws:pcs:compute-node-group-name` matching
  `login*`, so the policy survives users renaming their CNGs.
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

## `cluster-admin-policy.json` — deploy / update / delete

What it covers (Sids in the JSON):

| Sid | Purpose |
|---|---|
| `CloudFormationStackLifecycle` + `CloudFormationDescribeStackless` | Create/update/delete the parent and all nested CFN stacks; ChangeSets; describe and list |
| `EC2NetworkingAndComputeLifecycle` | VPC / subnets / IGW / NAT / EIP / security groups / route tables / VPC endpoints / launch templates / placement groups; `RunInstances` / `CreateFleet` for PCS-internal compute launches |
| `EC2DescribeForPCS` | All `ec2:Describe*` (PCS' control plane and CFN both call many of these) |
| `FSxLifecycle` | FSx for Lustre + OpenZFS create / update / delete / describe / tag |
| `PCSFullAccess` | `pcs:*` — clusters, compute node groups, queues |
| `IAMRoleAndInstanceProfileLifecycle` | Create / delete / tag / attach for IAM roles + instance profiles whose names contain `PCS` / `pcs` / `ImageBuilder` (matches what the templates produce) |
| `IAMPassRoleToPCSAndEC2` | `iam:PassRole` scoped to `*PCS*` roles passed to `ec2.amazonaws.com` / `pcs.amazonaws.com` |
| `IAMPassRoleToImageBuilder` | `iam:PassRole` scoped to `*ImageBuilder*` roles passed to `imagebuilder.amazonaws.com` |
| `ServiceLinkedRolesOneTime` | `iam:CreateServiceLinkedRole` for PCS / Spot / FSx / Image Builder (a no-op once the SLR exists) |
| `SSMParameterLifecycle` | Put / get / delete on `parameter/pcs/*` (Grafana admin password, etc.) and `parameter/aws/service/*` (PCS-Ready DLAMI auto-resolve) |
| `ImageBuilderLifecycleOptional` | EC2 Image Builder pipeline / recipe / component / distribution / lifecycle policy — only used when deploying `pcs-ready-dlami-with-enroot-pyxis.yaml` |
| `KMSAccessForDefaultEncryption` | KMS data-plane operations for default-encrypted EBS / FSx volumes (verbatim from the AWS PCS minimum-permissions policy) |
| `SecretsManagerForPCS` | `secretsmanager:*` scoped to `secret:pcs!*` (PCS internally creates a Slurm cluster auth secret) |
| `CloudWatchLogsForMonitoring` + `PCSVendedLogsDelivery` | CloudWatch Logs delivery wiring for PCS-vended Slurm logs (used when `DeployMonitoring=true` and PCS log delivery is configured) |

**Where it came from.** The `PCSFullAccess` + `EC2DescribeForPCS` + `IAMPassRoleToPCSAndEC2`
+ `ServiceLinkedRolesOneTime` + `KMSAccessForDefaultEncryption` + `SecretsManagerForPCS`
blocks are derived directly from the AWS-published "minimum permissions for an AWS PCS
service administrator" policy at
[docs.aws.amazon.com/pcs/.../security-min-permissions.html](https://docs.aws.amazon.com/pcs/latest/userguide/security-min-permissions.html).
The remaining blocks (CloudFormation, EC2 networking lifecycle, FSx, IAM role
lifecycle, SSM Parameter Store, Image Builder, CloudWatch Logs) are layered on
because the AWS reference policy assumes the VPC, FSx filesystems, and IAM roles
already exist — whereas the all-in-one template provisions all of them.

**A combined CRUD policy is intentional, not a mistake.** Splitting per phase
(deploy-only / update-only / delete-only) breaks in three places:

1. **CFN rollback on Create requires Delete actions.** A failed `CreateStack`
   automatically rolls back, deleting any resources it managed to create. A
   "deploy-only" policy without `Delete*` leaves the principal unable to clean up
   `CREATE_FAILED` stacks.
2. **Update is a strict superset of Deploy.** Any `UpdateStack` may replace
   resources (an `add-cng*.yaml` `LaunchTemplate` version bump replaces the LT,
   which is Create + Delete + Modify in CFN's eyes).
3. **Drift detection during Update calls `Describe*` across every service.**

The natural alternative is **operator policy = the CRUD JSON above** vs
**auditor policy = the same actions reduced to `*:Describe*` / `*:Get*` / `*:List*`**.

### How to attach

For a customer-managed policy from the JSON:

```bash
aws iam create-policy \
  --policy-name PCSClusterAdmin \
  --policy-document file://architectures/aws-pcs/iam/cluster-admin-policy.json
aws iam attach-user-policy \
  --user-name <deployer-user> \
  --policy-arn arn:aws:iam::<account-id>:policy/PCSClusterAdmin
```

Or attach to an assumable role for a CFN deployment role pattern.

### Pairing with AWS-managed policies

If you want a smaller customer-managed surface, you can attach AWS-managed
policies for parts of the stack and trim the corresponding Sids out of
`cluster-admin-policy.json`:

| AWS-managed | Replaces / overlaps with | Notes |
|---|---|---|
| `AWSCloudFormationFullAccess` | `CloudFormationStackLifecycle` + `CloudFormationDescribeStackless` | Broader (no resource-level scoping) but battle-tested |
| `AmazonFSxFullAccess` | `FSxLifecycle` | Closest fit; covers everything the templates need |
| `AWSImageBuilderFullAccess` | `ImageBuilderLifecycleOptional` | Only needed when the standalone DLAMI template is deployed |
| `AmazonEC2FullAccess` | `EC2NetworkingAndComputeLifecycle` + `EC2DescribeForPCS` | **Materially overprivileged** (e.g. EBS public-share). Prefer the customer-managed Sids |

There is **no** `AmazonPCSFullAccess` AWS-managed policy as of this writing — the
PCS portion has to be customer-managed.

---

## `cluster-user-policy.json` — SSM access for end users

What it covers:

| Sid | Purpose |
|---|---|
| `DiscoverClusterAndLoginNode` | `ec2:DescribeInstances` + `cloudformation:DescribeStacks` so the user can find the login node and read stack outputs (e.g. `ClusterId`) |
| `PCSReadOnly` | `pcs:Get*` / `pcs:List*` so the user can see cluster / queue / compute-node-group status |
| `GrafanaPasswordRead` | `ssm:GetParameter` scoped to `parameter/pcs/*/grafana/*` (the admin password generated by `cluster.yaml` when `DeployMonitoring=true`) |
| `SSMSessionToLoginNode` | `ssm:StartSession` scoped to **only** EC2 instances tagged `aws:pcs:compute-node-group-name=login*`, with the SSM documents needed for shell, SSH-over-SSM, and port-forwarding |
| `SSMSessionDocumentLookup` | `ssm:DescribeSessions` / `DescribeInstanceInformation` etc. (the SSM client / Session Manager UI calls these to render the connection list) |
| `SSMSessionTerminateOwn` | `ssm:TerminateSession` / `ResumeSession` scoped to sessions whose ID prefix matches `${aws:username}-*` (= sessions the user themselves opened) |

The `SSMSessionToLoginNode` Sid uses
**`ssm:resourceTag/aws:pcs:compute-node-group-name`** to scope sessions to login
nodes only. Compute nodes are intentionally NOT reachable via this policy — users
should reach them through Slurm (`srun`, `sbatch`), not directly via SSM.

### Typical user workflows the policy enables

1. **Connect to the login node:**
   ```bash
   INSTANCE_ID=$(aws ec2 describe-instances \
     --filters "Name=tag:aws:pcs:compute-node-group-name,Values=login" \
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

4. **SSH-over-SSM** (with `~/.ssh/config` configured per the README):
   ```bash
   ssh hpc8a-login
   ```

### Safety properties

- **No write access to cluster resources.** No `CreateStack`, no `pcs:Create*`,
  no `ec2:RunInstances`, no `iam:*`, no `fsx:*` write actions.
- **Login nodes only.** The instance-tag scope on `ssm:StartSession` blocks the
  user from opening shells on compute nodes (which are tagged
  `aws:pcs:compute-node-group-name=<cng-name>`, e.g. `cpu1` or `gpu-p5`).
- **Own sessions only.** `ssm:TerminateSession` is scoped to sessions whose ID
  matches the user's IAM username, so users can't kill each other's sessions.

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

4. **Diff against `cluster-admin-policy.json`**. The Access Analyzer output
   tends to be *narrower* than the sample (it only lists actions actually
   called) but with `Resource: "*"` even when narrower scoping is possible —
   the sample's resource ARNs are usually keepers.

The same approach works for `cluster-user-policy.json` — exercise the user
workflows (start a session, port-forward Grafana, terminate the session) in a
sandbox with broad SSM perms, then narrow to what was actually called.
