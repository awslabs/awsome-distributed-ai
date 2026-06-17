# IAM Test: cluster-admin / cluster-user policies

Validates the two IAM policy stacks ([`cluster-admin-iam.yaml`](../assets/cluster-admin-iam.yaml)
and [`cluster-user-iam.yaml`](../assets/cluster-user-iam.yaml), documented in
[docs/IAM.md](../docs/IAM.md)):

1. A principal with **only** the cluster-admin policy can deploy and tear down a full
   cluster — without `iam:CreatePolicy`.
2. A principal with **only** the cluster-user policy is constrained to SSM access on the
   login node and cannot read the OpenLDAP admin password.

This is the representative "two-role" use case: an admin who owns cluster lifecycle, and a
user who only runs jobs / views dashboards.

---

## Setup

Deploy both policy stacks, then create a throwaway role attached to **only** the admin
(or only the user) policy — nothing else — so the test reflects exactly what each policy
grants.

```bash
# Deploy the policy + group stacks (see docs/IAM.md for the parameters)
aws cloudformation create-stack --stack-name pcs-iam-admin \
  --template-url https://<bucket>.s3.amazonaws.com/<prefix>cluster-admin-iam.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region <region>
aws cloudformation create-stack --stack-name pcs-iam-user \
  --template-url https://<bucket>.s3.amazonaws.com/<prefix>cluster-user-iam.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region <region>
```

Attach the resulting managed policy (admin or user) to a dedicated test role you can
`assume-role` into, or use `aws iam simulate-principal-policy` against the policy ARN for
a no-deploy check.

---

## Test A — admin policy deploys and deletes a cluster

Assume the admin-only role, then run a representative deploy (multi-user + SSH + monitoring
all on, to exercise the IAM, SSM, and KMS permissions):

```bash
aws cloudformation create-stack --stack-name pcs-iam-test \
  --template-url https://<bucket>.s3.amazonaws.com/<prefix>pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=<az> \
    ParameterKey=DirectoryService,ParameterValue=OpenLDAP-LoginNode \
    ParameterKey=SSHAccessCidr,ParameterValue=<your-cidr>/32 \
    ParameterKey=MonitoringStack,ParameterValue=Prometheus-LoginNode \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region <region>
# ... wait for CREATE_COMPLETE, then:
aws cloudformation delete-stack --stack-name pcs-iam-test --region <region>
```

**Expected:**
- Every nested stack reaches `CREATE_COMPLETE` with **no `AccessDenied`**.
- The same admin-only role tears the stack down to `DELETE_COMPLETE` with no `AccessDenied`.
- It works **without `iam:CreatePolicy`** — the instance role's permissions are attached
  inline (`PutRolePolicy`), not as a managed policy, so deploy-all needs no policy-creation
  right.

No-deploy equivalent with `simulate-principal-policy` (pass proper resource ARNs and the
`iam:PassedToService` context): `cloudformation:CreateStack`,
`ec2:RunInstances`/`CreateSubnet`/`CreateSecurityGroup`, `fsx:CreateFileSystem`,
`pcs:CreateCluster`, `iam:CreateRole`/`PutRolePolicy`/`CreateInstanceProfile`/`PassRole`
(to `ec2` and `pcs`), `ssm:PutParameter`/`GetParameter` on `/pcs/*/ldap/*`, `kms:Decrypt`
→ all `allowed`; `iam:CreatePolicy` → `implicitDeny` (expected, and not needed).

---

## Test B — user policy is constrained

With the user-only role / policy:

**Expected (allowed):** `pcs:GetCluster`/`ListComputeNodeGroups`,
`cloudformation:DescribeStacks`, `ec2:DescribeInstances`, `ssm:StartSession` **on the login
node**, and reading `/pcs/*/grafana/*` (the Grafana password).

**Expected (denied — `implicitDeny`):** `cloudformation:CreateStack`, `pcs:CreateCluster`,
`ec2:RunInstances`, `fsx:CreateFileSystem`, `iam:CreateRole`; `ssm:StartSession` on a
**compute** node; and — critically for security — reading `/pcs/*/ldap/*` (the OpenLDAP
admin password).

```bash
# Example simulate checks (repeat per action/resource):
aws iam simulate-principal-policy --policy-source-arn <user-role-arn> \
  --action-names ssm:StartSession \
  --resource-arns arn:aws:ec2:<region>:<acct>:instance/<login-instance-id>
# → allowed for the login node, implicitDeny for a compute node
```

---

## Verified

Run end-to-end on a real account (us-east-2) against the major-update templates, with a
dedicated role attached to only the admin (or user) policy:

- Both IAM stacks `CREATE_COMPLETE`.
- Admin-only role: deploy-all (DirectoryService + SSHAccessCidr + monitoring) →
  `CREATE_COMPLETE`, `delete-stack` → `DELETE_COMPLETE`, both with no `AccessDenied`, and
  with `iam:CreatePolicy` simulating `implicitDeny` (confirming it isn't needed).
- User-only role: status/describe/login-SSM allowed; create/compute-SSM denied; can read
  `/pcs/*/grafana/*` but **not** `/pcs/*/ldap/*`.
