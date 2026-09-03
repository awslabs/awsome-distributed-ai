#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

###########################
###### Default Values #####
###########################

AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-west-2")
INFRA=""
STACK_NAME="hp-eks-slinky-stack"
CODEBUILD_STACK_NAME="slurmd-codebuild-stack"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LB_CONTROLLER_IAM_ROLE_NAME="AmazonEKS_LB_Controller_Role_slinky"
LB_CONTROLLER_IAM_POLICY_NAME="AWSLoadBalancerControllerIAMPolicy_slinky"
EBS_CSI_IAM_ROLE_NAME="AmazonEKS_EBS_CSI_DriverRole_slinky"
EBS_CSI_INLINE_POLICY_NAME="SageMakerHyperPodVolumeAccess"

###########################
###### Usage Function #####
###########################

usage() {
    cat <<EOF
Usage: $0 --infra <cfn|tf> [OPTIONS]

Tear down the Slurm cluster and HyperPod EKS infrastructure in reverse
order: Slurm cluster -> Slurm operator -> MariaDB -> CodeBuild stack ->
HyperPod infrastructure stack.

Required:
  --infra <cfn|tf>          Infrastructure method used for deployment

Optional:
  --region <region>         AWS region (default: AWS CLI configured or us-west-2)
  --stack-name <name>       HyperPod CFN stack name (default: hp-eks-slinky-stack)
  --help                    Show this help message

Examples:
  # Destroy CloudFormation-based deployment
  $0 --infra cfn

  # Destroy Terraform-based deployment with custom stack name
  $0 --infra tf --stack-name my-slinky-stack
EOF
    exit 0
}

###########################
###### Parse Arguments ####
###########################

while [[ $# -gt 0 ]]; do
    case $1 in
        --infra)
            INFRA="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

###########################
### Validate Arguments ####
###########################

if [[ -z "${INFRA}" ]]; then
    echo "Error: --infra is required (cfn or tf)"
    exit 1
fi

if [[ "${INFRA}" != "cfn" && "${INFRA}" != "tf" ]]; then
    echo "Error: --infra must be 'cfn' or 'tf' (got: ${INFRA})"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Slurm Cluster Teardown"
echo "=========================================="
echo ""
echo "  Infrastructure: ${INFRA}"
echo "  Region: ${AWS_REGION}"
echo "  Stack name: ${STACK_NAME}"
echo ""

read -r -p "This will destroy all Slurm and infrastructure resources. Continue? [y/N] " response
if [[ ! "${response}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

###########################
## Source env_vars.sh #####
###########################

if [[ -f "${SCRIPT_DIR}/env_vars.sh" ]]; then
    source "${SCRIPT_DIR}/env_vars.sh"
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || {
    echo "WARNING: Could not determine AWS account ID. Check your AWS credentials."
    echo "  IAM resources (policies, roles, Pod Identity associations) will NOT be cleaned up."
    echo "  Proceeding with Kubernetes-only teardown..."
    AWS_ACCOUNT_ID=""
}

###########################
## Uninstall Slurm ########
###########################

echo ""
echo "Uninstalling Slurm cluster..."
helm uninstall slurm -n slurm 2>/dev/null || echo "  Slurm cluster not found (already removed)."

echo "Uninstalling Slurm operator..."
helm uninstall slurm-operator -n slinky 2>/dev/null || echo "  Slurm operator not found (already removed)."

###########################
## Uninstall MariaDB ######
###########################

echo ""
echo "Deleting MariaDB instance..."
kubectl delete -f "${SCRIPT_DIR}/mariadb.yaml" 2>/dev/null || echo "  MariaDB instance not found."

echo "Uninstalling MariaDB operator..."
helm uninstall mariadb-operator -n mariadb 2>/dev/null || echo "  MariaDB operator not found (already removed)."

###########################
## Delete FSx PVC #########
###########################

echo ""
echo "Deleting FSx Lustre PV/PVC..."
kubectl delete -f "${SCRIPT_DIR}/lustre-pvc-slurm.yaml" 2>/dev/null || echo "  FSx PV/PVC not found."

###########################
## Delete Namespaces ######
###########################

echo ""
echo "Deleting namespaces..."
kubectl delete namespace slurm 2>/dev/null || echo "  Namespace slurm not found."
kubectl delete namespace slinky 2>/dev/null || echo "  Namespace slinky not found."
kubectl delete namespace mariadb 2>/dev/null || echo "  Namespace mariadb not found."

###########################
## Untag Public Subnets ###
###########################

echo ""
echo "Removing LB Controller subnet tags..."
if [[ -n "${VPC_ID:-}" ]]; then
    PUBLIC_SUBNET_IDS=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=${VPC_ID}" \
                  "Name=map-public-ip-on-launch,Values=true" \
        --query "Subnets[].SubnetId" --output text \
        --region "${AWS_REGION}" 2>/dev/null || echo "")

    if [[ -n "${PUBLIC_SUBNET_IDS}" ]]; then
        for SUBNET_ID in ${PUBLIC_SUBNET_IDS}; do
            aws ec2 delete-tags \
                --resources "${SUBNET_ID}" \
                --tags "Key=kubernetes.io/role/elb" \
                --region "${AWS_REGION}" 2>/dev/null || true
            echo "  Untagged ${SUBNET_ID}"
        done
    fi
else
    echo "  VPC_ID not set, skipping subnet untag."
fi

###########################
## Uninstall EBS CSI ######
###########################

echo ""
echo "Removing EBS CSI driver..."

# Delete EBS CSI addon
if [[ -n "${EKS_CLUSTER_NAME:-}" ]]; then
    if aws eks describe-addon \
        --cluster-name "${EKS_CLUSTER_NAME}" \
        --addon-name "aws-ebs-csi-driver" \
        --region "${AWS_REGION}" &>/dev/null; then
        echo "  Deleting EBS CSI driver addon..."
        aws eks delete-addon \
            --cluster-name "${EKS_CLUSTER_NAME}" \
            --addon-name "aws-ebs-csi-driver" \
            --region "${AWS_REGION}" \
            --no-cli-pager 2>/dev/null || true
    fi

    # Delete Pod Identity association
    EBS_ASSOC_ID=$(aws eks list-pod-identity-associations \
        --cluster-name "${EKS_CLUSTER_NAME}" \
        --namespace "kube-system" \
        --service-account "ebs-csi-controller-sa" \
        --region "${AWS_REGION}" \
        --query "associations[0].associationId" --output text 2>/dev/null || echo "None")

    if [[ "${EBS_ASSOC_ID}" != "None" && -n "${EBS_ASSOC_ID}" ]]; then
        echo "  Deleting EBS CSI Pod Identity association ${EBS_ASSOC_ID}..."
        aws eks delete-pod-identity-association \
            --cluster-name "${EKS_CLUSTER_NAME}" \
            --association-id "${EBS_ASSOC_ID}" \
            --region "${AWS_REGION}" \
            --no-cli-pager 2>/dev/null || true
    fi
else
    echo "  WARNING: EKS_CLUSTER_NAME not set — skipping EBS CSI addon and"
    echo "  Pod Identity cleanup. These resources may need manual deletion."
fi

# Delete gp3 StorageClass
kubectl delete storageclass gp3 2>/dev/null || echo "  gp3 StorageClass not found."

# Delete IAM role and policies
if aws iam get-role --role-name "${EBS_CSI_IAM_ROLE_NAME}" &>/dev/null; then
    echo "  Detaching policies from ${EBS_CSI_IAM_ROLE_NAME}..."
    aws iam detach-role-policy \
        --role-name "${EBS_CSI_IAM_ROLE_NAME}" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy" 2>/dev/null || true
    aws iam delete-role-policy \
        --role-name "${EBS_CSI_IAM_ROLE_NAME}" \
        --policy-name "${EBS_CSI_INLINE_POLICY_NAME}" 2>/dev/null || true
    echo "  Deleting IAM role ${EBS_CSI_IAM_ROLE_NAME}..."
    aws iam delete-role \
        --role-name "${EBS_CSI_IAM_ROLE_NAME}" 2>/dev/null || true
fi

###########################
## Uninstall LB Controller
###########################

echo ""
echo "Uninstalling AWS Load Balancer Controller..."
helm uninstall aws-load-balancer-controller -n kube-system 2>/dev/null || \
    echo "  LB Controller not found (already removed)."

# Delete Pod Identity association
if [[ -n "${EKS_CLUSTER_NAME:-}" ]]; then
    ASSOC_ID=$(aws eks list-pod-identity-associations \
        --cluster-name "${EKS_CLUSTER_NAME}" \
        --namespace "kube-system" \
        --service-account "aws-load-balancer-controller" \
        --region "${AWS_REGION}" \
        --query "associations[0].associationId" --output text 2>/dev/null || echo "None")

    if [[ "${ASSOC_ID}" != "None" && -n "${ASSOC_ID}" ]]; then
        echo "  Deleting Pod Identity association ${ASSOC_ID}..."
        aws eks delete-pod-identity-association \
            --cluster-name "${EKS_CLUSTER_NAME}" \
            --association-id "${ASSOC_ID}" \
            --region "${AWS_REGION}" \
            --no-cli-pager 2>/dev/null || true
    fi
else
    echo "  WARNING: EKS_CLUSTER_NAME not set — skipping LB Controller Pod"
    echo "  Identity cleanup. This resource may need manual deletion."
fi

# Delete IAM role and policy
if [[ -z "${AWS_ACCOUNT_ID}" ]]; then
    echo "  WARNING: Skipping LB Controller IAM cleanup — AWS_ACCOUNT_ID not available."
elif aws iam get-role --role-name "${LB_CONTROLLER_IAM_ROLE_NAME}" &>/dev/null; then
    echo "  Detaching policy from role ${LB_CONTROLLER_IAM_ROLE_NAME}..."
    LB_POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${LB_CONTROLLER_IAM_POLICY_NAME}"
    aws iam detach-role-policy \
        --role-name "${LB_CONTROLLER_IAM_ROLE_NAME}" \
        --policy-arn "${LB_POLICY_ARN}" 2>/dev/null || true
    echo "  Deleting IAM role ${LB_CONTROLLER_IAM_ROLE_NAME}..."
    aws iam delete-role \
        --role-name "${LB_CONTROLLER_IAM_ROLE_NAME}" 2>/dev/null || true
fi

if [[ -n "${AWS_ACCOUNT_ID}" ]]; then
    LB_POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${LB_CONTROLLER_IAM_POLICY_NAME}"
    if aws iam get-policy --policy-arn "${LB_POLICY_ARN}" &>/dev/null; then
        echo "  Deleting IAM policy ${LB_CONTROLLER_IAM_POLICY_NAME}..."
        aws iam delete-policy --policy-arn "${LB_POLICY_ARN}" 2>/dev/null || true
    fi
fi

###########################
## Uninstall cert-manager #
###########################

echo ""
echo "Uninstalling cert-manager..."
helm uninstall cert-manager -n cert-manager 2>/dev/null || \
    echo "  cert-manager not found (already removed)."
kubectl delete namespace cert-manager 2>/dev/null || echo "  Namespace cert-manager not found."

###########################
## Delete CodeBuild Stack #
###########################

echo ""
echo "Deleting CodeBuild infrastructure..."

if [[ "${INFRA}" == "cfn" ]]; then
    # The ECR repository has DeletionPolicy: Retain so CFN will not attempt
    # to delete it (avoids failure when images exist). It is preserved for
    # the customer and listed in the summary below.
    if aws cloudformation describe-stacks \
        --stack-name "${CODEBUILD_STACK_NAME}" \
        --region "${AWS_REGION}" &>/dev/null; then
        echo "  Deleting CodeBuild CloudFormation stack..."
        aws cloudformation delete-stack \
            --stack-name "${CODEBUILD_STACK_NAME}" \
            --region "${AWS_REGION}"
        aws cloudformation wait stack-delete-complete \
            --stack-name "${CODEBUILD_STACK_NAME}" \
            --region "${AWS_REGION}"
        echo "  CodeBuild stack deleted."
    else
        echo "  CodeBuild stack not found (already removed)."
    fi
else
    cb_dir="${SCRIPT_DIR}/codebuild-tf"
    if [[ -d "${cb_dir}" ]] && [[ -f "${cb_dir}/terraform.tfstate" ]]; then
        # Remove the ECR repository from Terraform state so that
        # 'terraform destroy' does not attempt to delete it (avoids failure
        # when images exist). The repo is preserved for the customer.
        # When create_ecr_repository=false these resources are not in state,
        # so the state rm commands fail silently via || true.
        terraform -chdir="${cb_dir}" state rm 'aws_ecr_repository.slurmd[0]' 2>/dev/null || true
        terraform -chdir="${cb_dir}" state rm 'aws_ecr_lifecycle_policy.slurmd[0]' 2>/dev/null || true

        echo "  Destroying CodeBuild Terraform resources..."
        terraform -chdir="${cb_dir}" destroy -auto-approve \
            -var="source_s3_bucket=unused"
        echo "  CodeBuild resources destroyed."
    else
        echo "  CodeBuild Terraform state not found (already removed)."
    fi
fi

###########################
## Delete Infrastructure ##
###########################

echo ""
echo "Deleting HyperPod EKS infrastructure..."

if [[ "${INFRA}" == "cfn" ]]; then
    if aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" &>/dev/null; then
        echo "  Deleting CloudFormation stack '${STACK_NAME}'..."
        aws cloudformation delete-stack \
            --stack-name "${STACK_NAME}" \
            --region "${AWS_REGION}"
        echo "  Stack deletion initiated."
        echo "  (This typically takes 15-20 minutes)"
        echo ""
        echo "  To monitor progress:"
        echo "    aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${AWS_REGION}"
    else
        echo "  Stack '${STACK_NAME}' not found (already removed)."
    fi
else
    tf_dir="${SCRIPT_DIR}/../terraform-modules/hyperpod-eks-tf"
    if [[ -d "${tf_dir}" ]]; then
        echo "  Destroying Terraform infrastructure..."
        terraform -chdir="${tf_dir}" plan -destroy -var-file="custom.tfvars"

        read -r -p "  Apply Terraform destroy? [y/N] " tf_response
        if [[ "${tf_response}" =~ ^[Yy]$ ]]; then
            terraform -chdir="${tf_dir}" destroy -var-file="custom.tfvars" -auto-approve
            echo "  Terraform resources destroyed."
        else
            echo "  Terraform destroy aborted."
        fi
    else
        echo "  Terraform directory not found: ${tf_dir}"
    fi
fi

###########################
## Clean Up Local Files ###
###########################

echo ""
echo "Cleaning up local generated files..."
rm -f "${SCRIPT_DIR}/slurm-values.yaml"
rm -f "${SCRIPT_DIR}/slurm-login-service-patch.yaml"
rm -f "${SCRIPT_DIR}/lustre-pvc-slurm.yaml"
rm -f "${SCRIPT_DIR}/env_vars.sh"
echo "  Cleaned up."

###########################
## Preserved Resources ####
###########################

echo ""
echo "The following resources were intentionally preserved:"
echo ""

# Check for S3 build context bucket
if [[ -n "${AWS_ACCOUNT_ID}" && -n "${AWS_REGION}" ]]; then
    PRESERVED_BUCKET="dlc-slurmd-codebuild-${AWS_ACCOUNT_ID}-${AWS_REGION}"
    if aws s3api head-bucket --bucket "${PRESERVED_BUCKET}" 2>/dev/null; then
        echo "  S3 bucket: ${PRESERVED_BUCKET}"
        echo "    To delete: aws s3 rb s3://${PRESERVED_BUCKET} --force"
        echo ""
    fi
fi

# Check for ECR repository
if aws ecr describe-repositories --repository-names dlc-slurmd \
    --region "${AWS_REGION}" &>/dev/null 2>&1; then
    echo "  ECR repository: dlc-slurmd"
    echo "    To delete: aws ecr delete-repository --repository-name dlc-slurmd --force --region ${AWS_REGION}"
    echo ""
fi

echo "=========================================="
echo "  Teardown Complete"
echo "=========================================="
echo ""
