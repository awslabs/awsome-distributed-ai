---
id: hyperpod-cfn-template-migration
name: HyperPod CloudFormation Template Migration
type: Enhancement
priority: P1
effort: Medium
impact: High
created: 2026-03-06
---

# HyperPod CloudFormation Template Migration

## Problem Statement

The CloudFormation deployment option in this project needs to be updated to use
the official SageMaker HyperPod service team maintained CloudFormation templates
instead of custom or outdated templates. The upstream templates are maintained at
<https://github.com/aws/sagemaker-hyperpod-cluster-setup/tree/main/eks/cloudformation>
and should be the canonical source for EKS-based HyperPod cluster deployments.

## Proposed Solution

Migrate the CloudFormation deployment path to reference the official
`sagemaker-hyperpod-cluster-setup` templates. The deployment should use a
`params.json` file with the AWS CLI:

```bash
aws cloudformation create-stack \
  --region us-west-2 \
  --stack-name example-hyperpod-eks-stack \
  --template-url https://aws-sagemaker-hyperpod-cluster-setup-us-west-2-prod.s3.us-west-2.amazonaws.com/templates/main-stack-eks-based-template.yaml \
  --parameters file://params.json \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND
```

### Key References

- **Upstream repo:**
  <https://github.com/aws/sagemaker-hyperpod-cluster-setup/tree/main/eks/cloudformation>
- **Local reference:**
  `/Users/natharno/Repos/sagemaker-hyperpod-cluster-setup/eks/cloudformation`

## Success Criteria

- [ ] TBD

## Notes

Created via feature-capture
