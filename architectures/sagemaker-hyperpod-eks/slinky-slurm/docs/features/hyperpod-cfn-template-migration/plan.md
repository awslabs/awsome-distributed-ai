---
id: hyperpod-cfn-template-migration
status: done
started: 2026-03-06
completed: 2026-03-06
---

# Implementation Plan: HyperPod CloudFormation Template Migration

## Overview

Migrate the CloudFormation deployment path from the old
`awsome-distributed-ai` nested stack templates to the official
SageMaker HyperPod service team maintained templates at
`github.com/aws/sagemaker-hyperpod-cluster-setup`.

## Analysis Summary

### Old System (current)

- **Template source:** `awsome-distributed-ai/.../cfn-templates/nested-stacks/main-stack.yaml`
  (curled at deploy time from GitHub raw URL)
- **Params format:** Flat key-value pairs with individual parameters for each
  instance group property (`AcceleratedInstanceType`, `AcceleratedInstanceCount`,
  `AvailabilityZoneId`, etc.)
- **Params files:** `g5/g5-params.json` (17 params), `p5/p5-params.json` (17 params)
- **Deploy command:**
  ```bash
  aws cloudformation create-stack \
    --stack-name hp-eks-slinky-stack \
    --template-body file://main-stack.yaml \
    --region $AWS_REGION \
    --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
    --parameters file://$PARAMS
  ```

### New System (target)

- **Template source:** S3-hosted per region at
  `https://aws-sagemaker-hyperpod-cluster-setup-${REGION}-prod.s3.${REGION}.amazonaws.com/templates/main-stack-eks-based-template.yaml`
- **Params format:** 100+ parameters with:
  - Instance groups as JSON array strings (`InstanceGroupSettings1..20`)
  - `Create*Stack` boolean toggles for each nested stack
  - Feature flags (`EnableHPInferenceFeature`, `EnableObservabilityFeature`,
    `EnableHPTrainingOperatorFeature`)
  - FSx configuration (`CreateFsxStack`, `StorageCapacity`, `DeploymentType`, etc.)
  - `AvailabilityZoneIds` (plural, comma-separated)
  - Task governance, data scientist roles, observability settings
- **Deploy command:**
  ```bash
  aws cloudformation create-stack \
    --region us-west-2 \
    --stack-name example-hyperpod-eks-stack \
    --template-url https://aws-sagemaker-hyperpod-cluster-setup-us-west-2-prod.s3.us-west-2.amazonaws.com/templates/main-stack-eks-based-template.yaml \
    --parameters file://params.json \
    --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND
  ```

### Key Differences

| Aspect | Old | New |
|--------|-----|-----|
| Template delivery | `--template-body file://` (curl from GitHub) | `--template-url` (S3-hosted by service team) |
| Capabilities | `CAPABILITY_IAM CAPABILITY_NAMED_IAM` | adds `CAPABILITY_AUTO_EXPAND` |
| Instance groups | Flat params per group | JSON array strings (`InstanceGroupSettings1`) |
| AZ config | `AvailabilityZoneId` (singular) | `AvailabilityZoneIds` (plural, comma-sep) |
| Feature toggles | None | `EnableHPInferenceFeature`, `EnableObservabilityFeature`, etc. |
| Stack toggles | None | `Create*Stack` booleans for each nested stack |
| FSx | Not included | `CreateFsxStack`, `StorageCapacity`, etc. |
| K8s version | `1.32` | Default `1.33` |
| Transform | None | `AWS::LanguageExtensions` |

## Implementation Steps

### Step 1: Create new g5 params file

**File:** `g5/g5-params.json`

Replace the existing 17 flat parameters with the new format. Key parameters
to set for the g5 profile:

```json
[
  {"ParameterKey": "ResourceNamePrefix", "ParameterValue": "slinky-hp-eks"},
  {"ParameterKey": "EKSClusterName", "ParameterValue": "slinky-eks-cluster"},
  {"ParameterKey": "HyperPodClusterName", "ParameterValue": "slinky-hp-cluster"},
  {"ParameterKey": "KubernetesVersion", "ParameterValue": "1.33"},
  {"ParameterKey": "AvailabilityZoneIds", "ParameterValue": "usw2-az2"},
  {"ParameterKey": "InstanceGroupSettings1", "ParameterValue": "<JSON array>"},
  {"ParameterKey": "CreateFsxStack", "ParameterValue": "true"},
  ...
]
```

The `InstanceGroupSettings1` value must encode the accelerated instance group
as a JSON array string. A second entry (`InstanceGroupSettings2`) encodes the
general-purpose group.

**Acceptance criteria:**
- [x] Contains all required parameters for the new template
- [x] `InstanceGroupSettings1` contains the g5.8xlarge accelerated group config
  (4 instances, 500GB EBS, stress check enabled, connectivity check enabled)
- [x] `InstanceGroupSettings2` contains the m5.2xlarge general-purpose group
  (2 instances, 500GB EBS)
- [x] Reasonable defaults set for feature flags (inference enabled, observability
  disabled, training operator disabled -- matching defaults in template)
- [x] JSON is valid and well-formatted (2-space indent per repo convention)

### Step 2: Create new p5 params file

**File:** `p5/p5-params.json`

Same structure as g5, with p5-specific values:
- `InstanceGroupSettings1`: `ml.p5.48xlarge`, 2 instances
- All other shared parameters identical to g5

**Acceptance criteria:**
- [x] Same structure as g5-params.json
- [x] p5.48xlarge instance type, 2 instances
- [x] JSON is valid and well-formatted

### Step 3: Update README.md CloudFormation deployment section

**File:** `README.md` (lines ~55-93)

Update the "Deploy Using CloudFormation" section to:
1. Remove the `curl` step for downloading `main-stack.yaml`
2. Change `--template-body file://main-stack.yaml` to
   `--template-url https://aws-sagemaker-hyperpod-cluster-setup-${AWS_REGION}-prod.s3.${AWS_REGION}.amazonaws.com/templates/main-stack-eks-based-template.yaml`
3. Add `CAPABILITY_AUTO_EXPAND` to `--capabilities`
4. Remove or update the `create_config.sh` reference (may need adjustment for
   the new stack outputs)
5. Update the CloudFormation stack link to point to the official
   `sagemaker-hyperpod-cluster-setup` repo

**Acceptance criteria:**
- [x] No reference to the old `main-stack.yaml` curl/download
- [x] `--template-url` uses the S3-hosted template
- [x] `CAPABILITY_AUTO_EXPAND` is included
- [x] Instructions are clear and self-contained
- [x] Markdown lint passes (100 char line length, code blocks excluded)

### Step 4: Update README.md cleanup section

**File:** `README.md` (lines ~1007-1010)

Update the CFN delete-stack command if stack name or region handling changed.

**Acceptance criteria:**
- [x] Cleanup instructions match the new deployment pattern

### Step 5: Verify g5/p5 params symmetry

Per AGENTS.md, `g5/` and `p5/` must stay structurally in sync. Verify that
both params files have the same parameter keys and only differ in the
intentional instance-type-specific values.

**Acceptance criteria:**
- [x] Both files have identical parameter keys
- [x] Only `InstanceGroupSettings1` values differ (instance type, count)

### Step 6: Validate params against template

Run a dry-run validation to ensure the params files are compatible with the
new template (requires AWS credentials and network access):

```bash
aws cloudformation validate-template \
  --template-url https://aws-sagemaker-hyperpod-cluster-setup-us-west-2-prod.s3.us-west-2.amazonaws.com/templates/main-stack-eks-based-template.yaml
```

**Acceptance criteria:**
- [x] Template validation succeeds
- [x] All required parameters are provided in both params files

## Open Questions

1. **InstanceGroupSettings JSON format:** RESOLVED. The JSON schema is an array
   of objects with fields: `InstanceCount`, `InstanceGroupName`, `InstanceType`,
   `TargetAvailabilityZoneId`, `ThreadsPerCore`, and
   `InstanceStorageConfigs` (array of `{EbsVolumeConfig: {VolumeSizeInGB}}}`).
   Multiple instance groups can be packed into a single `InstanceGroupSettings1`
   parameter value.

2. **Stack outputs:** RESOLVED. The new template exports conditional outputs:
   `OutputVpcId`, `OutputPrivateSubnetIds`, `OutputSecurityGroupId`,
   `OutputEKSClusterName`, `OutputEKSClusterArn`, `OutputS3BucketName`,
   `OutputHyperPodClusterName`, `OutputHyperPodClusterArn`, and others.
   README updated to extract these via `aws cloudformation describe-stacks`.

3. **FSx integration:** RESOLVED. The new CFN template includes
   `CreateFsxStack=true` which deploys FSx for Lustre as part of the stack.
   This eliminates the need for the manual FSx CSI driver installation step
   in the Day-2 automation.

## Dependencies

- None (this feature can be implemented independently)

## Estimated Effort

- Params file creation: ~1 hour (need to determine InstanceGroupSettings schema)
- README updates: ~30 minutes
- Validation: ~30 minutes
- Total: ~2 hours
