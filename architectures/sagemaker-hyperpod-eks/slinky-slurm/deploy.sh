#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

###########################
###### Default Values #####
###########################

AWS_REGION="us-west-2"
AZ_ID="usw2-az2"
INSTANCE_TYPE=""
INSTANCE_COUNT=4
INFRA=""
STACK_NAME="hp-eks-slinky-stack"
TRAINING_PLAN=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###########################
## Source Helper Library ###
###########################

source "${SCRIPT_DIR}/lib/deploy_helpers.sh"

###########################
###### Usage Function #####
###########################

usage() {
    cat <<EOF
Usage: $0 --instance-type <ml.X.Y> --infra <cfn|tf> [OPTIONS]

Deploy HyperPod EKS infrastructure using CloudFormation or Terraform.

Required:
  --instance-type <type>    SageMaker instance type for the accelerated group
                            (e.g. ml.g5.8xlarge, ml.p5.48xlarge, ml.g6.12xlarge)
  --infra <cfn|tf>          Infrastructure deployment method

Optional:
  --instance-count <N>      Number of accelerated instances (default: 4)
  --training-plan <name>    Training plan name (auto-resolves ARN and AZ)
  --region <region>         AWS region (default: us-west-2)
  --az-id <az-id>           Availability zone ID for instance groups and FSx
                            (default: usw2-az2)
  --stack-name <name>       CloudFormation stack name (default: hp-eks-slinky-stack)
  --help                    Show this help message

Examples:
  # Deploy 4 ml.g5.8xlarge instances in us-west-2 using CloudFormation
  $0 --instance-type ml.g5.8xlarge --infra cfn

  # Deploy 2 ml.p5.48xlarge instances using Terraform
  $0 --instance-type ml.p5.48xlarge --instance-count 2 --infra tf

  # Deploy with a training plan (AZ auto-resolved from plan)
  $0 --instance-type ml.p5.48xlarge --instance-count 2 --training-plan my-p5-plan --infra cfn

  # Deploy in a different region
  $0 --instance-type ml.g5.8xlarge --infra cfn --region us-east-1 --az-id use1-az2

  # Deploy with custom stack name
  $0 --instance-type ml.g5.8xlarge --infra cfn --stack-name my-slinky-stack
EOF
    exit 0
}

###########################
###### Parse Arguments ####
###########################

while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --az-id)
            AZ_ID="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --instance-count)
            INSTANCE_COUNT="$2"
            shift 2
            ;;
        --infra)
            INFRA="$2"
            shift 2
            ;;
        --stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        --training-plan)
            TRAINING_PLAN="$2"
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

if [[ -z "${INSTANCE_TYPE}" ]]; then
    echo "Error: --instance-type is required (e.g. ml.g5.8xlarge)"
    exit 1
fi

if [[ -z "${INFRA}" ]]; then
    echo "Error: --infra is required (cfn or tf)"
    exit 1
fi

if [[ "${INFRA}" != "cfn" && "${INFRA}" != "tf" ]]; then
    echo "Error: --infra must be 'cfn' or 'tf' (got: ${INFRA})"
    exit 1
fi

# Resolve instance profile (validates type via EC2 API, sets ACCEL_INSTANCE_TYPE
# and ACCEL_INSTANCE_COUNT)
resolve_instance_profile "${INSTANCE_TYPE}" "${INSTANCE_COUNT}"

###########################
## Training Plan Resolve ##
###########################

TRAINING_PLAN_ARN=""
if [[ -n "${TRAINING_PLAN}" ]]; then
    resolve_training_plan "${TRAINING_PLAN}" "${AWS_REGION}"

    # Override AZ_ID if training plan's AZ differs
    if [[ "${TRAINING_PLAN_AZ_ID}" != "${AZ_ID}" ]]; then
        echo ""
        echo "WARNING: Overriding --az-id from '${AZ_ID}' to '${TRAINING_PLAN_AZ_ID}'"
        echo "  to match training plan '${TRAINING_PLAN}'."
        echo "  The cluster subnet must be in the training plan's AZ."
        AZ_ID="${TRAINING_PLAN_AZ_ID}"
    fi
fi

###########################
## Check Prerequisites ####
###########################

echo "Checking prerequisites..."
check_command "aws" || exit 1

if [[ "${INFRA}" == "cfn" ]]; then
    check_command "jq" || exit 1
elif [[ "${INFRA}" == "tf" ]]; then
    check_command "terraform" || exit 1
    check_command "jq" || exit 1
fi

# Validate AWS credentials
if ! aws sts get-caller-identity --region "${AWS_REGION}" &>/dev/null; then
    echo "Error: Invalid AWS credentials or unable to reach AWS."
    exit 1
fi

echo "  AWS CLI: OK"
echo "  AWS credentials: OK"
echo "  Region: ${AWS_REGION}"
echo "  Instance type: ${ACCEL_INSTANCE_TYPE} x ${ACCEL_INSTANCE_COUNT}"
echo "  Infrastructure: ${INFRA}"
echo "  AZ ID: ${AZ_ID}"
if [[ -n "${TRAINING_PLAN_ARN}" ]]; then
    echo "  Training plan: ${TRAINING_PLAN} (${TRAINING_PLAN_ARN})"
fi

###########################
## Resolve AZ IDs #########
###########################

echo ""
echo "Resolving availability zones for region ${AWS_REGION}..."

AZ_IDS=$(aws ec2 describe-availability-zones \
    --region "${AWS_REGION}" \
    --filters "Name=opt-in-status,Values=opt-in-not-required" \
    --query "AvailabilityZones[?ZoneType=='availability-zone'].ZoneId | sort(@) | [:5]" \
    --output text | tr '\t' ',')

if [[ -z "${AZ_IDS}" ]]; then
    echo "Error: No availability zones found in region ${AWS_REGION}."
    exit 1
fi

echo "  Resolved AZ IDs: ${AZ_IDS}"
echo "  Instance/FSx AZ ID: ${AZ_ID}"

# Verify the specified AZ_ID exists in the resolved list
if ! validate_az_id "${AZ_ID}" "${AZ_IDS}"; then
    echo ""
    echo "WARNING: AZ ID '${AZ_ID}' was not found in the resolved AZ list for ${AWS_REGION}."
    echo "  Available AZs: ${AZ_IDS}"
    echo "  Specify a valid AZ ID with --az-id or press Ctrl+C to abort."
    read -r -p "  Continue anyway? [y/N] " response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

###########################
## Deploy via CFN #########
###########################

deploy_cfn() {
    local params_file="${SCRIPT_DIR}/params.json"

    if [[ ! -f "${params_file}" ]]; then
        echo "Error: Parameters file not found: ${params_file}"
        exit 1
    fi

    echo ""
    echo "Preparing CloudFormation parameters..."
    echo "  Source: ${params_file}"
    echo "  Stack name: ${STACK_NAME}"

    # Resolve AZ values and instance overrides in the params file
    local resolved_params
    resolved_params=$(resolve_cfn_params \
        "${params_file}" \
        "${AZ_IDS}" \
        "${AZ_ID}" \
        "${ACCEL_INSTANCE_TYPE}" \
        "${ACCEL_INSTANCE_COUNT}" \
        "${TRAINING_PLAN_ARN}")

    # Write resolved params to a temp file
    local resolved_file
    resolved_file=$(mktemp /tmp/resolved-params-XXXXXXXX)
    mv "${resolved_file}" "${resolved_file}.json"
    resolved_file="${resolved_file}.json"
    echo "${resolved_params}" > "${resolved_file}"

    echo "  Resolved params written to: ${resolved_file}"
    echo ""

    # Construct the S3 template URL
    local template_url="https://aws-sagemaker-hyperpod-cluster-setup-${AWS_REGION}-prod.s3.${AWS_REGION}.amazonaws.com/templates/main-stack-eks-based-template.yaml"

    # Validate the template and cross-check parameter keys
    if ! validate_cfn_template "${template_url}" "${resolved_file}" "${AWS_REGION}"; then
        echo ""
        echo "Error: CloudFormation template validation failed."
        echo "  Fix the issues above and re-run."
        rm -f "${resolved_file}"
        exit 1
    fi

    echo ""
    echo "Deploying CloudFormation stack '${STACK_NAME}'..."
    echo "  Template: ${template_url}"
    echo ""

    # Check if the stack already exists to choose create vs update
    local stack_status
    stack_status=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" \
        --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "DOES_NOT_EXIST")

    if [[ "${stack_status}" == "DOES_NOT_EXIST" ]]; then
        echo "  Creating new stack..."
        aws cloudformation create-stack \
            --region "${AWS_REGION}" \
            --stack-name "${STACK_NAME}" \
            --template-url "${template_url}" \
            --parameters "file://${resolved_file}" \
            --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND

        echo ""
        echo "Stack creation initiated. Waiting for completion..."
        echo "  (This typically takes 20-30 minutes)"
        echo ""

        aws cloudformation wait stack-create-complete \
            --region "${AWS_REGION}" \
            --stack-name "${STACK_NAME}"

        echo "Stack '${STACK_NAME}' created successfully."
    elif [[ "${stack_status}" =~ ^(CREATE_COMPLETE|UPDATE_COMPLETE|UPDATE_ROLLBACK_COMPLETE)$ ]]; then
        echo "  Stack already exists (status: ${stack_status}). Updating..."
        local update_output
        if update_output=$(aws cloudformation update-stack \
            --region "${AWS_REGION}" \
            --stack-name "${STACK_NAME}" \
            --template-url "${template_url}" \
            --parameters "file://${resolved_file}" \
            --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND 2>&1); then
            # Update initiated successfully
            echo ""
            echo "Stack update initiated. Waiting for completion..."
            echo ""

            aws cloudformation wait stack-update-complete \
                --region "${AWS_REGION}" \
                --stack-name "${STACK_NAME}"

            echo "Stack '${STACK_NAME}' updated successfully."
        elif echo "${update_output}" | grep -q "No updates are to be performed"; then
            echo "  No updates needed — stack is already up to date."
        else
            echo "Error updating stack: ${update_output}" >&2
            rm -f "${resolved_file}"
            exit 1
        fi
    else
        echo "Error: Stack '${STACK_NAME}' is in state '${stack_status}'."
        echo "  Cannot create or update. Resolve the stack state manually."
        rm -f "${resolved_file}"
        exit 1
    fi

    echo ""

    # Extract and export stack outputs
    extract_cfn_outputs

    # Clean up temp file
    rm -f "${resolved_file}"
}

###########################
## Extract CFN Outputs ####
###########################

extract_cfn_outputs() {
    echo "Extracting stack outputs..."
    echo ""

    local account_id
    account_id=$(aws sts get-caller-identity --query Account --output text)

    local eks_cluster_name
    eks_cluster_name=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputEKSClusterName'].OutputValue" \
        --output text)

    local vpc_id
    vpc_id=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputVpcId'].OutputValue" \
        --output text)

    local private_subnet_id
    private_subnet_id=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputPrivateSubnetIds'].OutputValue" \
        --output text | cut -d',' -f1)

    local security_group_id
    security_group_id=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputSecurityGroupId'].OutputValue" \
        --output text)

    # Write env_vars.sh for sourcing
    local env_file="${SCRIPT_DIR}/env_vars.sh"
    cat > "${env_file}" <<EOF
export AWS_ACCOUNT_ID="${account_id}"
export AWS_REGION="${AWS_REGION}"
export EKS_CLUSTER_NAME="${eks_cluster_name}"
export VPC_ID="${vpc_id}"
export PRIVATE_SUBNET_ID="${private_subnet_id}"
export SECURITY_GROUP_ID="${security_group_id}"
export STACK_ID="${STACK_NAME}"
EOF

    echo "Environment variables written to: ${env_file}"
    echo ""
    echo "  AWS_ACCOUNT_ID=${account_id}"
    echo "  AWS_REGION=${AWS_REGION}"
    echo "  EKS_CLUSTER_NAME=${eks_cluster_name}"
    echo "  VPC_ID=${vpc_id}"
    echo "  PRIVATE_SUBNET_ID=${private_subnet_id}"
    echo "  SECURITY_GROUP_ID=${security_group_id}"
    echo ""
    echo "To load these variables into your shell, run:"
    echo "  source ${env_file}"
}

###########################
## Deploy via Terraform ###
###########################

deploy_tf() {
    local tfvars_file="${SCRIPT_DIR}/custom.tfvars"
    local tf_dir="${SCRIPT_DIR}/../terraform-modules/hyperpod-eks-tf"

    if [[ ! -f "${tfvars_file}" ]]; then
        echo "Error: Terraform variables file not found: ${tfvars_file}"
        exit 1
    fi

    if [[ ! -d "${tf_dir}" ]]; then
        echo "Error: Terraform modules directory not found: ${tf_dir}"
        echo "  Expected at: ${tf_dir}"
        echo "  Make sure you have cloned the awsome-distributed-ai repo."
        exit 1
    fi

    echo ""
    echo "Preparing Terraform deployment..."
    echo "  Source tfvars: ${tfvars_file}"
    echo "  Terraform dir: ${tf_dir}"

    # Copy the tfvars file into the TF module directory and patch it
    local target_tfvars="${tf_dir}/custom.tfvars"
    cp "${tfvars_file}" "${target_tfvars}"

    resolve_tf_vars \
        "${target_tfvars}" \
        "${AWS_REGION}" \
        "${AZ_ID}" \
        "${ACCEL_INSTANCE_TYPE}" \
        "${ACCEL_INSTANCE_COUNT}" \
        "${TRAINING_PLAN_ARN}"

    echo "  Resolved tfvars written to: ${target_tfvars}"
    echo ""

    echo "Initializing Terraform..."
    terraform -chdir="${tf_dir}" init

    echo ""
    echo "Generating Terraform plan..."
    terraform -chdir="${tf_dir}" plan -var-file="custom.tfvars"

    echo ""
    read -r -p "Apply this Terraform plan? [y/N] " response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        echo "Aborted. You can apply manually with:"
        echo "  cd ${tf_dir}"
        echo "  terraform apply -var-file=custom.tfvars"
        exit 0
    fi

    echo ""
    echo "Applying Terraform..."
    terraform -chdir="${tf_dir}" apply -var-file="custom.tfvars" -auto-approve

    echo ""
    echo "Terraform apply completed successfully."
    echo ""

    # Run terraform_outputs.sh if it exists
    local outputs_script="${SCRIPT_DIR}/../terraform-modules/terraform_outputs.sh"
    if [[ -f "${outputs_script}" ]]; then
        echo "Extracting Terraform outputs..."
        chmod +x "${outputs_script}"
        # Clear any existing env_vars.sh to avoid stale/duplicate exports
        rm -f "${SCRIPT_DIR}/../terraform-modules/env_vars.sh"
        (cd "${SCRIPT_DIR}/../terraform-modules" && ./terraform_outputs.sh)

        local env_file="${SCRIPT_DIR}/../terraform-modules/env_vars.sh"
        if [[ -f "${env_file}" ]]; then
            # Copy env_vars.sh to slinky-slurm directory for convenience
            cp "${env_file}" "${SCRIPT_DIR}/env_vars.sh"
            echo "Environment variables written to: ${SCRIPT_DIR}/env_vars.sh"
            echo ""
            echo "To load these variables into your shell, run:"
            echo "  source ${SCRIPT_DIR}/env_vars.sh"
        fi
    else
        echo "Note: terraform_outputs.sh not found at ${outputs_script}"
        echo "  You may need to manually extract outputs."
    fi
}

###########################
###### Main Execution #####
###########################

echo ""
echo "=========================================="
echo "  HyperPod EKS Infrastructure Deployment"
echo "=========================================="
echo ""

if [[ "${INFRA}" == "cfn" ]]; then
    deploy_cfn
elif [[ "${INFRA}" == "tf" ]]; then
    deploy_tf
fi

echo ""
echo "=========================================="
echo "  Deployment Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Source the environment variables:"
echo "       source ${SCRIPT_DIR}/env_vars.sh"
echo "  2. Update your kubectl context:"
echo "       aws eks update-kubeconfig --name \$EKS_CLUSTER_NAME --region ${AWS_REGION}"
echo "  3. Verify node connectivity:"
echo "       kubectl get nodes"
echo ""
