# IRSA Setup for Ray Pods S3 Access

This guide explains how to set up IAM Roles for Service Accounts (IRSA) to give your Ray pods access to S3 for managed tiered checkpointing.

## Prerequisites

- AWS CLI configured with appropriate permissions (for OIDC provider setup)
- kubectl configured to access your EKS cluster

## Quick Setup

Run the automated setup script:

```bash
./setup/setup-irsa.sh
```

This script will:
1. Check/create OIDC provider for your EKS cluster
2. Create an IAM policy with S3 full access
3. Create an IAM role with trust policy for the service account
4. Attach the policy to the role
5. Create a Kubernetes service account with the IAM role annotation

## What Gets Created

### IAM Policy
- **Name**: `ray-s3-access-policy`
- **Permissions**: Full S3 access (`s3:*`)

### IAM Role
- **Name**: `ray-s3-access-role`
- **Trust Policy**: Allows the Kubernetes service account to assume this role via OIDC

### Kubernetes Service Account
- **Name**: `ray-s3-sa`
- **Namespace**: `default`
- **Annotation**: Links to the IAM role ARN

## How It Works

1. **OIDC Provider**: EKS cluster has an OIDC identity provider that allows Kubernetes service accounts to authenticate with AWS IAM
2. **Service Account**: Ray pods use a Kubernetes service account annotated with an IAM role ARN
3. **IAM Role**: The IAM role has a trust policy that allows the service account to assume it
4. **Credentials**: AWS SDK automatically retrieves temporary credentials via the OIDC token

## Verification

After running the setup, verify the configuration:

```bash
# Check service account
kubectl get sa ray-s3-sa -n default -o yaml

# Check IAM role
aws iam get-role --role-name ray-s3-access-role

# Check policy attachment
aws iam list-attached-role-policies --role-name ray-s3-access-role
```

## Using with RayCluster

The RayCluster YAML has been updated to use this service account. Both head and worker pods will have:

```yaml
spec:
  serviceAccountName: ray-s3-sa
```

This gives them automatic S3 access without needing to manage credentials.

## Testing S3 Access

Once the cluster is deployed, you can test S3 access from a pod:

```bash
# Get a pod name
POD_NAME=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Test S3 access
kubectl exec -it $POD_NAME -- aws s3 ls s3://${S3_BUCKET_NAME}/
```

## Troubleshooting

### OIDC Provider Not Found

If you get an error about OIDC provider, you need to create an IAM OIDC identity provider for your cluster. This allows Kubernetes service accounts to authenticate with AWS IAM.

**Option 1: Using AWS CLI**
```bash
# Get your cluster's OIDC issuer URL
oidc_url=$(aws eks describe-cluster --name $EKS_CLUSTER_NAME \
    --query "cluster.identity.oidc.issuer" --output text)

# Create the OIDC provider
aws iam create-open-id-connect-provider \
    --url $oidc_url \
    --client-id-list sts.amazonaws.com
```

**Option 2: Using eksctl**
```bash
# Install eksctl first
# macOS: brew install eksctl
# Linux: curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp && sudo mv /tmp/eksctl /usr/local/bin

# Create OIDC provider
eksctl utils associate-iam-oidc-provider --cluster $EKS_CLUSTER_NAME --approve
```

### Credentials Not Working
1. Check service account annotation:
   ```bash
   kubectl get sa ray-s3-sa -o yaml
   ```
2. Check pod has the service account:
   ```bash
   kubectl get pod <pod-name> -o yaml | grep serviceAccountName
   ```
3. Check environment variables in pod:
   ```bash
   kubectl exec -it <pod-name> -- env | grep AWS
   ```

### Permission Denied
1. Verify IAM role has the policy attached
2. Check the trust policy allows your service account
3. Ensure the OIDC provider ARN matches in the trust policy

## Security Considerations

The current setup grants full S3 access (`s3:*`). For production, consider:

1. **Restrict to specific buckets**:
   ```json
   "Resource": [
       "arn:aws:s3:::sagemaker-mtcpt-${ACCOUNT}",
       "arn:aws:s3:::sagemaker-mtcpt-${ACCOUNT}/*"
   ]
   ```

2. **Limit actions**:
   ```json
   "Action": [
       "s3:GetObject",
       "s3:PutObject",
       "s3:DeleteObject",
       "s3:ListBucket"
   ]
   ```

3. **Add conditions** for additional security

## Cleanup

To remove the IRSA setup:

```bash
# Delete service account
kubectl delete sa ray-s3-sa -n default

# Detach policy from role
aws iam detach-role-policy \
    --role-name ray-s3-access-role \
    --policy-arn arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/ray-s3-access-policy

# Delete IAM role
aws iam delete-role --role-name ray-s3-access-role

# Delete IAM policy
aws iam delete-policy \
    --policy-arn arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/ray-s3-access-policy
```

## References

- [EKS IRSA Documentation](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html)
- [SageMaker HyperPod Managed Tiered Checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-managed-tiered-checkpointing.html)
