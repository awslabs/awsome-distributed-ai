#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

###########################
###### Default Values #####
###########################

RUN_SETUP=true
SETUP_ARGS=""
SKIP_CERT_MANAGER=false
SKIP_LB_CONTROLLER=false
SKIP_EBS_CSI=false
CLUSTER_NAME_OVERRIDE=""
VPC_ID_OVERRIDE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-west-2")
SLURM_OPERATOR_VERSION="1.0.1"
SLURM_CHART_VERSION="1.0.1"
MARIADB_OPERATOR_VERSION="25.10.4"
CERT_MANAGER_VERSION="1.19.2"
LB_CONTROLLER_CHART_VERSION="1.11.0"
LB_CONTROLLER_POLICY_URL="https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.11.0/docs/install/iam_policy.json"
LB_CONTROLLER_IAM_ROLE_NAME="AmazonEKS_LB_Controller_Role_slinky"
LB_CONTROLLER_IAM_POLICY_NAME="AWSLoadBalancerControllerIAMPolicy_slinky"
EBS_CSI_IAM_ROLE_NAME="AmazonEKS_EBS_CSI_DriverRole_slinky"
EBS_CSI_INLINE_POLICY_NAME="SageMakerHyperPodVolumeAccess"

###########################
###### Usage Function #####
###########################

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Install the Slurm cluster on HyperPod EKS. Runs setup.sh first (unless
--skip-setup is specified), then deploys MariaDB, the Slurm operator, and
the Slurm cluster via Helm.

Optional:
  --skip-setup              Use previously generated slurm-values.yaml
  --skip-cert-manager       Skip cert-manager installation (already installed)
  --skip-lb-controller      Skip AWS LB Controller installation (already installed)
  --skip-ebs-csi            Skip EBS CSI driver installation (already installed)
  --cluster-name <name>     EKS cluster name (overrides env_vars.sh)
  --vpc-id <id>             VPC ID (overrides env_vars.sh)
  --region <region>         AWS region (default: AWS CLI configured or us-west-2)
  --help                    Show this help message

Options passed through to setup.sh:
  --instance-type <type>    SageMaker instance type (e.g. ml.g5.8xlarge)
  --instance-count <N>      Number of accelerated instances (default: 4)
  --infra <cfn|tf>          Infrastructure method for CodeBuild stack
  --repo-name <name>        ECR repository name
  --tag <tag>               Image tag
  --local-build             Build image locally instead of CodeBuild
  --skip-build              Skip image build (use existing image in ECR)

Examples:
  # Full install: build image + deploy Slurm (ml.g5.8xlarge via CloudFormation)
  $0 --instance-type ml.g5.8xlarge --infra cfn

  # Skip setup (slurm-values.yaml already generated)
  $0 --skip-setup

  # Bring your own cluster (skip deploy.sh, provide cluster details directly)
  $0 --skip-setup --cluster-name my-cluster --vpc-id vpc-12345 \\
     --skip-cert-manager --skip-lb-controller --skip-ebs-csi

  # Build locally, then install
  $0 --instance-type ml.p5.48xlarge --instance-count 2 --infra tf --local-build
EOF
    exit 0
}

###########################
###### Parse Arguments ####
###########################

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            RUN_SETUP=false
            shift
            ;;
        --skip-cert-manager)
            SKIP_CERT_MANAGER=true
            shift
            ;;
        --skip-lb-controller)
            SKIP_LB_CONTROLLER=true
            shift
            ;;
        --skip-ebs-csi)
            SKIP_EBS_CSI=true
            shift
            ;;
        --cluster-name)
            CLUSTER_NAME_OVERRIDE="$2"
            shift 2
            ;;
        --vpc-id)
            VPC_ID_OVERRIDE="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            SETUP_ARGS="${SETUP_ARGS} --region $2"
            shift 2
            ;;
        --help)
            usage
            ;;
        --instance-type|--instance-count|--infra|--repo-name|--tag)
            SETUP_ARGS="${SETUP_ARGS} $1 $2"
            shift 2
            ;;
        --local-build|--skip-build)
            SETUP_ARGS="${SETUP_ARGS} $1"
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

echo ""
echo "=========================================="
echo "  Slurm Cluster Installation"
echo "=========================================="
echo ""

###########################
## Run Setup ##############
###########################

if [[ "${RUN_SETUP}" == "true" ]]; then
    echo "Running setup.sh..."
    echo ""
    # shellcheck disable=SC2086
    bash "${SCRIPT_DIR}/setup.sh" ${SETUP_ARGS}
else
    echo "Skipping setup (--skip-setup)..."
    if [[ ! -f "${SCRIPT_DIR}/slurm-values.yaml" ]]; then
        echo "Error: slurm-values.yaml not found."
        echo "  Run without --skip-setup, or generate it manually."
        exit 1
    fi
    echo "  Using existing slurm-values.yaml"
fi

echo ""

###########################
## Source env_vars.sh #####
###########################

if [[ -f "${SCRIPT_DIR}/env_vars.sh" ]]; then
    source "${SCRIPT_DIR}/env_vars.sh"
fi

# Apply CLI overrides (--cluster-name / --vpc-id take precedence over env_vars.sh)
if [[ -n "${CLUSTER_NAME_OVERRIDE}" ]]; then
    EKS_CLUSTER_NAME="${CLUSTER_NAME_OVERRIDE}"
fi
if [[ -n "${VPC_ID_OVERRIDE}" ]]; then
    VPC_ID="${VPC_ID_OVERRIDE}"
fi

# EKS_CLUSTER_NAME and VPC_ID are needed for the LB Controller.
# They come from env_vars.sh (generated by deploy.sh), CLI flags, or the environment.
if [[ -z "${EKS_CLUSTER_NAME:-}" ]]; then
    echo "Error: EKS_CLUSTER_NAME not set."
    echo "  Run deploy.sh first, set it in env_vars.sh, or pass --cluster-name."
    exit 1
fi
if [[ -z "${VPC_ID:-}" ]]; then
    echo "Error: VPC_ID not set."
    echo "  Run deploy.sh first, set it in env_vars.sh, or pass --vpc-id."
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

###########################
## Install cert-manager ###
###########################

if [[ "${SKIP_CERT_MANAGER}" == "true" ]]; then
    echo "Skipping cert-manager (--skip-cert-manager)..."
else
    echo "Installing cert-manager (v${CERT_MANAGER_VERSION})..."

    helm repo add jetstack https://charts.jetstack.io 2>/dev/null || true
    helm repo update jetstack

    helm upgrade --install cert-manager jetstack/cert-manager \
        --version="${CERT_MANAGER_VERSION}" \
        --namespace=cert-manager --create-namespace \
        --set crds.enabled=true \
        --wait
    echo "  cert-manager installed."

    # Wait for cert-manager webhook to be ready (prevents race conditions
    # with components that create Certificate resources at install time).
    echo "  Waiting for cert-manager webhook..."
    kubectl wait --for=condition=Available \
        deployment/cert-manager-webhook \
        -n cert-manager --timeout=120s
    echo "  cert-manager webhook ready."
fi

###########################
## Tag Public Subnets #####
###########################

if [[ "${SKIP_LB_CONTROLLER}" == "true" ]]; then
    echo "Skipping subnet tagging and LB Controller (--skip-lb-controller)..."
else
    # The AWS LB Controller requires public subnets to have the tag
    # kubernetes.io/role/elb=1 for internet-facing NLBs. The HyperPod CFN
    # template does NOT add this tag, so we add it here.
    echo "Tagging public subnets for LB Controller (kubernetes.io/role/elb=1)..."

    PUBLIC_SUBNET_IDS=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=${VPC_ID}" \
                  "Name=map-public-ip-on-launch,Values=true" \
        --query "Subnets[].SubnetId" --output text \
        --region "${AWS_REGION}" 2>/dev/null || echo "")

    if [[ -n "${PUBLIC_SUBNET_IDS}" ]]; then
        for SUBNET_ID in ${PUBLIC_SUBNET_IDS}; do
            aws ec2 create-tags \
                --resources "${SUBNET_ID}" \
                --tags "Key=kubernetes.io/role/elb,Value=1" \
                --region "${AWS_REGION}"
            echo "  Tagged ${SUBNET_ID}"
        done
    else
        echo "  WARNING: No public subnets found in VPC ${VPC_ID}."
        echo "  The LB Controller may not be able to provision internet-facing NLBs."
    fi

    ###########################
    ## Install LB Controller ##
    ###########################

    echo ""
    echo "Installing AWS Load Balancer Controller (chart v${LB_CONTROLLER_CHART_VERSION})..."

    helm repo add eks https://aws.github.io/eks-charts 2>/dev/null || true
    helm repo update eks

    # Create IAM policy for the LB Controller (idempotent -- ignores if exists)
    if ! aws iam get-policy \
        --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${LB_CONTROLLER_IAM_POLICY_NAME}" \
        &>/dev/null; then
        echo "  Creating IAM policy ${LB_CONTROLLER_IAM_POLICY_NAME}..."
        POLICY_JSON=$(curl -fsSL "${LB_CONTROLLER_POLICY_URL}")
        LB_POLICY_ARN=$(aws iam create-policy \
            --policy-name "${LB_CONTROLLER_IAM_POLICY_NAME}" \
            --policy-document "${POLICY_JSON}" \
            --query "Policy.Arn" --output text)
        echo "  Policy created: ${LB_POLICY_ARN}"
    else
        LB_POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${LB_CONTROLLER_IAM_POLICY_NAME}"
        echo "  IAM policy already exists: ${LB_POLICY_ARN}"
    fi

    # Create IAM role for Pod Identity
    if ! aws iam get-role --role-name "${LB_CONTROLLER_IAM_ROLE_NAME}" &>/dev/null; then
        echo "  Creating IAM role ${LB_CONTROLLER_IAM_ROLE_NAME}..."
        TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"pods.eks.amazonaws.com"},"Action":["sts:AssumeRole","sts:TagSession"]}]}'
        aws iam create-role \
            --role-name "${LB_CONTROLLER_IAM_ROLE_NAME}" \
            --assume-role-policy-document "${TRUST_POLICY}" \
            --no-cli-pager
        aws iam attach-role-policy \
            --role-name "${LB_CONTROLLER_IAM_ROLE_NAME}" \
            --policy-arn "${LB_POLICY_ARN}"
        echo "  Role created and policy attached."
    else
        echo "  IAM role already exists: ${LB_CONTROLLER_IAM_ROLE_NAME}"
        # Ensure policy is attached (idempotent — handles partial failure on
        # previous run where role was created but attachment failed)
        aws iam attach-role-policy \
            --role-name "${LB_CONTROLLER_IAM_ROLE_NAME}" \
            --policy-arn "${LB_POLICY_ARN}" 2>/dev/null || true
    fi

    LB_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${LB_CONTROLLER_IAM_ROLE_NAME}"

    # Create Pod Identity association (idempotent -- check first)
    EXISTING_ASSOC=$(aws eks list-pod-identity-associations \
        --cluster-name "${EKS_CLUSTER_NAME}" \
        --namespace "kube-system" \
        --service-account "aws-load-balancer-controller" \
        --region "${AWS_REGION}" \
        --query "associations[0].associationId" --output text 2>/dev/null || echo "None")

    if [[ "${EXISTING_ASSOC}" == "None" || -z "${EXISTING_ASSOC}" ]]; then
        echo "  Creating Pod Identity association..."
        aws eks create-pod-identity-association \
            --cluster-name "${EKS_CLUSTER_NAME}" \
            --namespace "kube-system" \
            --service-account "aws-load-balancer-controller" \
            --role-arn "${LB_ROLE_ARN}" \
            --region "${AWS_REGION}" \
            --no-cli-pager
        echo "  Pod Identity association created."
    else
        echo "  Pod Identity association already exists: ${EXISTING_ASSOC}"
    fi

    # Install the Helm chart
    helm upgrade --install aws-load-balancer-controller \
        eks/aws-load-balancer-controller \
        --version="${LB_CONTROLLER_CHART_VERSION}" \
        --namespace=kube-system \
        --set clusterName="${EKS_CLUSTER_NAME}" \
        --set region="${AWS_REGION}" \
        --set vpcId="${VPC_ID}" \
        --wait
    echo "  AWS Load Balancer Controller installed."

    # Wait for the webhook to be ready (prevents race conditions with
    # Service annotations that trigger LB provisioning).
    echo "  Waiting for LB Controller webhook..."
    kubectl wait --for=condition=Available \
        deployment/aws-load-balancer-controller \
        -n kube-system --timeout=120s
    echo "  LB Controller webhook ready."
fi

###########################
## Install EBS CSI Driver #
###########################

# The EBS CSI driver is required for dynamic PV provisioning (MariaDB, Slurm
# accounting). On HyperPod, it also needs sagemaker:AttachClusterNodeVolume
# permissions via an inline policy.

if [[ "${SKIP_EBS_CSI}" == "true" ]]; then
    echo ""
    echo "Skipping EBS CSI driver (--skip-ebs-csi)..."

    # Still create gp3 StorageClass if needed (cheap and required by MariaDB)
    if ! kubectl get storageclass gp3 &>/dev/null; then
        echo "  Creating gp3 StorageClass..."
        kubectl apply -f - <<SC
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
volumeBindingMode: WaitForFirstConsumer
SC
        echo "  gp3 StorageClass created."
    else
        echo "  gp3 StorageClass already exists."
    fi
else
    echo ""
    echo "Installing EBS CSI Driver..."

    # Create IAM role for EBS CSI controller (Pod Identity)
    if ! aws iam get-role --role-name "${EBS_CSI_IAM_ROLE_NAME}" &>/dev/null; then
        echo "  Creating IAM role ${EBS_CSI_IAM_ROLE_NAME}..."
        TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"pods.eks.amazonaws.com"},"Action":["sts:AssumeRole","sts:TagSession"]}]}'
        aws iam create-role \
            --role-name "${EBS_CSI_IAM_ROLE_NAME}" \
            --assume-role-policy-document "${TRUST_POLICY}" \
            --no-cli-pager

        # Attach AWS managed policy for EBS CSI
        aws iam attach-role-policy \
            --role-name "${EBS_CSI_IAM_ROLE_NAME}" \
            --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"

        # Inline policy for HyperPod-specific volume operations
        INLINE_POLICY=$(cat <<POLICY
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:AttachClusterNodeVolume",
                "sagemaker:DetachClusterNodeVolume"
            ],
            "Resource": "arn:aws:sagemaker:${AWS_REGION}:${AWS_ACCOUNT_ID}:cluster/*"
        },
        {
            "Effect": "Allow",
            "Action": "eks:DescribeCluster",
            "Resource": "arn:aws:eks:${AWS_REGION}:${AWS_ACCOUNT_ID}:cluster/${EKS_CLUSTER_NAME}"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:AttachVolume",
                "ec2:DetachVolume",
                "ec2:DescribeVolumes"
            ],
            "Resource": "arn:aws:ec2:${AWS_REGION}:${AWS_ACCOUNT_ID}:volume/*"
        }
    ]
}
POLICY
        )
        aws iam put-role-policy \
            --role-name "${EBS_CSI_IAM_ROLE_NAME}" \
            --policy-name "${EBS_CSI_INLINE_POLICY_NAME}" \
            --policy-document "${INLINE_POLICY}"
        echo "  Role created with EBS CSI and HyperPod policies."
    else
        echo "  IAM role already exists: ${EBS_CSI_IAM_ROLE_NAME}"
        # Ensure policies are attached (idempotent — handles partial failure
        # on previous run where role was created but attachment failed)
        aws iam attach-role-policy \
            --role-name "${EBS_CSI_IAM_ROLE_NAME}" \
            --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy" 2>/dev/null || true
    fi

    EBS_CSI_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${EBS_CSI_IAM_ROLE_NAME}"

    # Create Pod Identity association for EBS CSI controller
    EXISTING_EBS_ASSOC=$(aws eks list-pod-identity-associations \
        --cluster-name "${EKS_CLUSTER_NAME}" \
        --namespace "kube-system" \
        --service-account "ebs-csi-controller-sa" \
        --region "${AWS_REGION}" \
        --query "associations[0].associationId" --output text 2>/dev/null || echo "None")

    if [[ "${EXISTING_EBS_ASSOC}" == "None" || -z "${EXISTING_EBS_ASSOC}" ]]; then
        echo "  Creating Pod Identity association for ebs-csi-controller-sa..."
        aws eks create-pod-identity-association \
            --cluster-name "${EKS_CLUSTER_NAME}" \
            --namespace "kube-system" \
            --service-account "ebs-csi-controller-sa" \
            --role-arn "${EBS_CSI_ROLE_ARN}" \
            --region "${AWS_REGION}" \
            --no-cli-pager
        echo "  Pod Identity association created."
    else
        echo "  Pod Identity association already exists: ${EXISTING_EBS_ASSOC}"
    fi

    # Install the EBS CSI driver EKS addon
    EXISTING_ADDON=$(aws eks describe-addon \
        --cluster-name "${EKS_CLUSTER_NAME}" \
        --addon-name "aws-ebs-csi-driver" \
        --region "${AWS_REGION}" \
        --query "addon.status" --output text 2>/dev/null || echo "NOT_FOUND")

    if [[ "${EXISTING_ADDON}" == "NOT_FOUND" ]]; then
        echo "  Installing EBS CSI driver addon..."
        aws eks create-addon \
            --cluster-name "${EKS_CLUSTER_NAME}" \
            --addon-name "aws-ebs-csi-driver" \
            --region "${AWS_REGION}" \
            --no-cli-pager
        echo "  Waiting for addon to become active..."
        aws eks wait addon-active \
            --cluster-name "${EKS_CLUSTER_NAME}" \
            --addon-name "aws-ebs-csi-driver" \
            --region "${AWS_REGION}"
        echo "  EBS CSI driver addon installed."
    else
        echo "  EBS CSI driver addon already installed (status: ${EXISTING_ADDON})."
    fi

    # Create gp3 StorageClass (needed by MariaDB and Slurm accounting PVCs)
    if ! kubectl get storageclass gp3 &>/dev/null; then
        echo "  Creating gp3 StorageClass..."
        kubectl apply -f - <<SC
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
volumeBindingMode: WaitForFirstConsumer
SC
        echo "  gp3 StorageClass created."
    else
        echo "  gp3 StorageClass already exists."
    fi
fi

###########################
## Apply FSx PVC ##########
###########################

echo ""
echo "Applying FSx Lustre PV/PVC..."
kubectl create namespace slurm 2>/dev/null || true
kubectl apply -f "${SCRIPT_DIR}/lustre-pvc-slurm.yaml"
echo "  FSx PV/PVC applied."

###########################
## Install MariaDB ########
###########################

echo ""
echo "Installing MariaDB operator (v${MARIADB_OPERATOR_VERSION})..."

helm repo add mariadb-operator \
    https://mariadb-operator.github.io/mariadb-operator 2>/dev/null || true
helm repo update mariadb-operator

helm upgrade --install mariadb-operator mariadb-operator/mariadb-operator \
    --version="${MARIADB_OPERATOR_VERSION}" \
    --namespace=mariadb --create-namespace \
    --set crds.enabled=true \
    --wait
echo "  MariaDB operator installed."

echo ""
echo "Creating MariaDB instance..."

kubectl apply -f "${SCRIPT_DIR}/mariadb.yaml"
echo "  MariaDB instance applied."

# Wait for MariaDB to be ready
echo "  Waiting for MariaDB to be ready..."
kubectl wait --for=condition=Ready mariadb/mariadb \
    -n slurm --timeout=300s 2>/dev/null || \
    echo "  WARNING: MariaDB readiness check timed out. Proceeding anyway."

###########################
## Install Slurm Operator #
###########################

echo ""
echo "Installing Slurm Operator (v${SLURM_OPERATOR_VERSION})..."

# Delete stale CRDs only if they exist (upgrading from pre-1.0 Slinky)
if kubectl get crd clusters.slinky.slurm.net &>/dev/null; then
    echo "  Removing stale CRD: clusters.slinky.slurm.net"
    kubectl delete crd clusters.slinky.slurm.net
fi
if kubectl get crd nodesets.slinky.slurm.net &>/dev/null; then
    echo "  Removing stale CRD: nodesets.slinky.slurm.net"
    kubectl delete crd nodesets.slinky.slurm.net
fi

helm upgrade --install slurm-operator \
    oci://ghcr.io/slinkyproject/charts/slurm-operator \
    --version="${SLURM_OPERATOR_VERSION}" \
    --namespace=slinky --create-namespace \
    --set crds.enabled=true \
    --wait
echo "  Slurm operator installed."

###########################
## Install Slurm Cluster ##
###########################

echo ""
echo "Installing Slurm Cluster (v${SLURM_CHART_VERSION})..."

if ! helm status slurm -n slurm &>/dev/null; then
    helm upgrade --install slurm \
        oci://ghcr.io/slinkyproject/charts/slurm \
        --values="${SCRIPT_DIR}/slurm-values.yaml" \
        --version="${SLURM_CHART_VERSION}" \
        --namespace=slurm
    echo "  Slurm cluster installed."
else
    echo "  Slurm cluster already installed."
    echo "  To update: helm upgrade slurm oci://ghcr.io/slinkyproject/charts/slurm \\"
    echo "    --values=${SCRIPT_DIR}/slurm-values.yaml --version=${SLURM_CHART_VERSION} -n slurm"
fi

###########################
## Configure NLB ##########
###########################

echo ""
echo "Configuring login service NLB..."

# Wait for the login service to exist
echo "  Waiting for slurm-login-slinky service..."
MAX_WAIT=60
WAIT_COUNT=0
until kubectl get service slurm-login-slinky -n slurm &>/dev/null; do
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [[ ${WAIT_COUNT} -ge ${MAX_WAIT} ]]; then
        echo "Error: Timed out waiting for slurm-login-slinky service (${MAX_WAIT} attempts)." >&2
        echo "  Check Slurm cluster status: kubectl -n slurm get pods" >&2
        exit 1
    fi
    sleep 5
done
echo "  Service found."

# Get public IP for NLB source range restriction
IP_ADDRESS="$(curl -s https://checkip.amazonaws.com)"
echo "  Source IP: ${IP_ADDRESS}"

# Generate and apply service patch
sed "s|\${ip_address}|${IP_ADDRESS}|g" \
    "${SCRIPT_DIR}/slurm-login-service-patch.yaml.template" \
    > "${SCRIPT_DIR}/slurm-login-service-patch.yaml"

kubectl patch service slurm-login-slinky -n slurm \
    --patch-file "${SCRIPT_DIR}/slurm-login-service-patch.yaml"
echo "  Login service patched with NLB annotations."

###########################
## Wait for NLB ###########
###########################

echo ""
echo "Waiting for NLB endpoint..."

SLURM_LOGIN_HOSTNAME=""
for i in $(seq 1 60); do
    SLURM_LOGIN_HOSTNAME=$(kubectl get services -n slurm \
        -l app.kubernetes.io/instance=slurm,app.kubernetes.io/name=login \
        -o jsonpath="{.items[0].status.loadBalancer.ingress[0].hostname}" 2>/dev/null || true)

    if [[ -n "${SLURM_LOGIN_HOSTNAME}" ]]; then
        break
    fi
    sleep 5
done

echo ""
echo "=========================================="
echo "  Installation Complete"
echo "=========================================="
echo ""

if [[ -n "${SLURM_LOGIN_HOSTNAME}" ]]; then
    echo "  Login endpoint: ${SLURM_LOGIN_HOSTNAME}"
    echo ""
    echo "  SSH into the Slurm login node:"
    echo "    ssh -i ~/.ssh/id_ed25519_slurm root@${SLURM_LOGIN_HOSTNAME}"
else
    echo "  WARNING: NLB hostname not yet available."
    echo "  Check with: kubectl get svc slurm-login-slinky -n slurm"
fi

echo ""
echo "  Verify cluster status:"
echo "    kubectl -n slurm get pods -l app.kubernetes.io/instance=slurm"
echo ""
