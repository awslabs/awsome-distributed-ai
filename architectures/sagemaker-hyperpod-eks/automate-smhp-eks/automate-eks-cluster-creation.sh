#!/bin/bash

# Workshop Automation Script for SageMaker HyperPod with EKS
# This script automates the steps of creating a HyperPod cluster with EKS orchestration

# Exit immediately if a command exits with a non-zero status. Print commands and their arguments as executed
set -e

# HAVE JQ INSTALLED!!!

#===Global===
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
export AWS_REGION=${AWS_DEFAULT_REGION:-$(aws configure get region)}
export K8_VERSION="1.31.2"      # Update this with AMI releases
export K8_RELEASE_DATE="2024-11-15" # Update this with AMI releases
export DEVICE=$(uname)
export OS=$(uname -m)

TOTAL_STEPS=5
CURRENT_STEP=0

#===Style Definitions===
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print a yellow header
print_header() {
    echo -e "\n${BLUE}=================================================${NC}"
    echo -e "\n${YELLOW}==== $1 ====${NC}\n"
    echo -e "\n${BLUE}=================================================${NC}"
}

# UX Function for a Progress Bar :)
progress_bar() {
    local duration=$1
    local steps=$2
    local width=50
    local progress=0

    for ((i=0; i<steps; i++)); do
        progress=$(( (i * width) / steps ))
        printf "\r[%-${width}s] %d%%" "$(printf '#%.0s' $(seq 1 $progress))" "$(( (progress * 100) / width ))"
        sleep 0.1
    done
    echo
}

#===Function Definitions===

# Helper function to install AWS CLI depending on OS
install_aws_cli() {
    if [[ $DEVICE == *"Darwin"* ]]; then
        curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
        sudo installer -pkg AWSCLIV2.pkg -target /
        rm AWSCLIV2.pkg
    elif [[ $DEVICE == *"Linux"* ]]; then   
        if [[ $OS == *"x86_64"* ]]; then
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        elif [[ $OS == *"aarch64"* ]]; then
            curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
        else
            echo "Unsupported Linux architecture: $OS. Please check https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html to install manually"
            exit 1    
        fi
        unzip awscliv2.zip
        sudo ./aws/install --update
        rm -rf aws awscliv2.zip
    else
        echo "Unsupported device: $DEVICE. Please check https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html to install manually"    
    fi
}

# Function to check the AWS CLI version and install/update as required
check_and_install_aws_cli() {
    echo -e "${BLUE}=== Checking AWS CLI Installation ===${NC}"

    if ! command -v aws &> /dev/null; then
        echo -e "${YELLOW}⚠️  AWS CLI is not installed. Installing...${NC}"
        install_aws_cli
    else
        echo -e "${GREEN}✅ AWS CLI found. Checking version...${NC}"
        CLI_VERSION=$(aws --version | awk '{print $1}' | cut -d/ -f2)

        echo -e "${BLUE}Current version: ${YELLOW}$CLI_VERSION${NC}"
        echo -e "${BLUE}Min. required version: ${YELLOW}2.17.47${NC}"

        if [[ "$(printf '%s\n' "2.17.47" "$CLI_VERSION" | sort -V | head -n1)" != "2.17.47" ]]; then
            echo -e "${YELLOW}⚠️  AWS CLI version $CLI_VERSION is lower than required.${NC}"
            echo -e "${YELLOW}   Updating AWS CLI...${NC}"
            install_aws_cli
        else
            echo -e "${GREEN}✅ AWS CLI version $CLI_VERSION is up to date.${NC}"
        fi
    fi     

    echo -e "${BLUE}=== AWS CLI Check Complete ===${NC}\n"

}

# Function to check if Git is installed and configured
check_git() {
    if ! command -v git &> /dev/null; then
        echo "Git is not installed. Please install Git and try again."
        exit 1
    fi
}

# Function to check and install kubectl
install_kubectl() {
    echo -e "${BLUE}=== Checking kubectl Installation ===${NC}"

    # Check if kubectl is already installed
    if ! command -v kubectl &> /dev/null; then
        echo -e "${YELLOW}⚠️  kubectl is not installed. Installing...${NC}"
        install_kubectl_binary
    else
        echo -e "${GREEN}✅ kubectl found. Checking version...${NC}"
        CURRENT_VERSION=$(kubectl version --client -o json | jq -r '.clientVersion.gitVersion' | tr -d 'v')
        
        echo -e "${BLUE}Current version: ${YELLOW}$CURRENT_VERSION${NC}"
        echo -e "${BLUE}Required version: ${YELLOW}$KUBERNETES_VERSION${NC}"

        # Compare versions
        if [[ "$(printf '%s\n' "$KUBERNETES_VERSION" "$CURRENT_VERSION" | sort -V | head -n1)" != "$KUBERNETES_VERSION" ]]; then
            echo -e "${YELLOW}⚠️  kubectl version $CURRENT_VERSION is lower than required version $KUBERNETES_VERSION${NC}"
            echo -e "${YELLOW}Installing newer version...${NC}"
            install_kubectl_binary
        else
            echo -e "${GREEN}✅ kubectl version $CURRENT_VERSION meets requirements${NC}"
        fi
    fi

    echo -e "${BLUE}=== kubectl Check Complete ===${NC}\n"
}

# Helper function to perform the actual installation of kubectl 
install_kubectl_binary() {
    # Create temporary directory for download
    TMP_DIR=$(mktemp -d) && cd "$TMP_DIR"   

    if [[ $DEVICE == "Darwin" ]]; then
        echo -e "${BLUE}Installing kubectl for macOS...${NC}"
        curl -O "https://s3.us-west-2.amazonaws.com/amazon-eks/${KUBERNETES_VERSION}/${KUBECTL_RELEASE_DATE}/bin/darwin/amd64/kubectl"
    elif [[ $DEVICE == "Linux" ]]; then
        if [[ $OS == "x86_64" ]]; then
            echo -e "${BLUE}Installing kubectl for Linux (amd64)...${NC}"
            curl -O "https://s3.us-west-2.amazonaws.com/amazon-eks/${KUBERNETES_VERSION}/${KUBECTL_RELEASE_DATE}/bin/linux/amd64/kubectl"
        elif [[ $OS == "aarch64" ]]; then
            echo -e "${BLUE}Installing kubectl for Linux (arm64)...${NC}"
            curl -O "https://s3.us-west-2.amazonaws.com/amazon-eks/${KUBERNETES_VERSION}/${KUBECTL_RELEASE_DATE}/bin/linux/arm64/kubectl"
        else
            echo -e "${YELLOW}⚠️ Unsupported Linux architecture: $OS${NC}"
            cd - > /dev/null
            rm -rf "$TMP_DIR"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  Unsupported operating system: $DEVICE. Please check https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html to install kubectl manually and re-run this script{NC}"
        cd - > /dev/null
        rm -rf "$TMP_DIR"
        return 1
    fi

    # Make kubectl executable and move to bin directory
    chmod +x ./kubectl
    mkdir -p $HOME/bin && mv ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH

    # Clean up
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    
    echo -e "${GREEN}✅ kubectl installed successfully${NC}"
}

# Function to install eksctl
install_eksctl() {
    echo -e "${BLUE}=== Checking eksctl Installation ===${NC}"

    # Check if eksctl is already installed
    if ! command -v eksctl &> /dev/null; then
        echo -e "${YELLOW}⚠️  eksctl is not installed. Installing...${NC}"
        
        # Check OS type
        if [[ "$DEVICE" == "Darwin" ]]; then
            echo -e "${BLUE}Installing eksctl using Homebrew. If you don't want to use Homebrew, please exit the script, install manually and re-run.${NC}"
            
            # Install using Homebrew
            if ! brew tap weaveworks/tap; then
                echo -e "${RED}❌ Failed to tap weaveworks repository${NC}"
                return 1
            fi
            
            if ! brew install weaveworks/tap/eksctl; then
                echo -e "${RED}❌ Failed to install eksctl${NC}"
                return 1
            fi
        else
            # Linux installation
            if [[ "$OS" == "x86_64" ]]; then
                ARCH="amd64"
            elif [[ "$OS" == "aarch64" ]]; then
                ARCH="arm64"
            else
                echo -e "${RED}❌ Unsupported architecture: $OS. Please refer to https://eksctl.io/installation/ to install manually.${NC}"
                return 1
            fi
            
            PLATFORM="${DEVICE}_$ARCH"
            TMP_DIR=$(mktemp -d)
            cd "$TMP_DIR"

            echo -e "${BLUE}Downloading eksctl for $PLATFORM...${NC}"
            
            # Download eksctl
            if ! curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"; then
                echo -e "${RED}❌ Failed to download eksctl${NC}"
                cd - > /dev/null
                rm -rf "$TMP_DIR"
                return 1
            fi

            # Download and verify checksum
            echo -e "${BLUE}Verifying checksum...${NC}"
            if ! curl -sL "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_checksums.txt" | grep $PLATFORM | sha256sum --check; then
                echo -e "${RED}❌ Checksum verification failed${NC}"
                cd - > /dev/null
                rm -rf "$TMP_DIR"
                return 1
            fi

            # Extract and install
            echo -e "${BLUE}Installing eksctl...${NC}"
            if ! tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp; then
                echo -e "${RED}❌ Failed to extract eksctl${NC}"
                cd - > /dev/null
                rm -rf "$TMP_DIR"
                return 1
            fi

            # Clean up downloaded tar
            rm eksctl_$PLATFORM.tar.gz

            # Move to bin directory with sudo
            if ! sudo mv /tmp/eksctl /usr/local/bin; then
                echo -e "${RED}❌ Failed to install eksctl to /usr/local/bin${NC}"
                cd - > /dev/null
                rm -rf "$TMP_DIR"
                return 1
            fi

            # Clean up
            cd - > /dev/null
            rm -rf "$TMP_DIR"
        fi

        # Verify installation
        if command -v eksctl &> /dev/null; then
            echo -e "${GREEN}✅ eksctl installed successfully${NC}"
            eksctl version
        else
            echo -e "${RED}❌ eksctl installation verification failed${NC}"
            return 1
        fi
    else
        echo -e "${GREEN}✅ eksctl is already installed${NC}"
        eksctl version
    fi

    echo -e "${BLUE}=== eksctl Check Complete ===${NC}\n"
}

# Function to install Helm
install_helm() {
    echo -e "${BLUE}=== Checking eksctl Installation ===${NC}"
    if ! command -v helm &> /dev/null; then
        echo -e "${YELLOW}⚠️  Helm is not installed. Installing...${NC}"
        curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
        chmod 700 get_helm.sh
        ./get_helm.sh
        rm -f get_helm.sh
        echo -e "${GREEN}✅ Helm installed successfully${NC}"
    else
        echo -e "${GREEN}✅ Helm is already installed${NC}"
    fi
}

# Function to clone the awsome-distributed-ai repository
clone_adt() {
    REPO_NAME="awsome-distributed-ai"
    if [ -d "$REPO_NAME" ]; then
        echo -e "${YELLOW}⚠️  The directory '$REPO_NAME' already exists.${NC}"
        echo -e "${GREEN}Do you want to remove it and clone again? (yes/no): ${NC}"
        read -e REMOVE_AND_CLONE
        if [[ $REMOVE_AND_CLONE == "yes" ]]; then
            echo -e "${YELLOW}Removing existing directory...${NC}"
            rm -rf "$REPO_NAME"
            echo -e "${BLUE}Cloning repository...${NC}"
            git clone --depth=1 https://github.com/awslabs/awsome-distributed-ai/
            echo -e "${GREEN}✅ Repository cloned successfully${NC}"
        else
            echo -e "${BLUE}Using existing directory...${NC}"
        fi
    else
        echo -e "${BLUE}Cloning repository $REPO_NAME...${NC}"
        git clone --depth=1 https://github.com/awslabs/awsome-distributed-ai/
        echo -e "${GREEN}✅ Repository cloned successfully${NC}"
    fi
}

# Helper function to get user inputs with default values specified
get_input() {
    local prompt="$1"
    local default="$2"
    local input
    read -e -p "$prompt [$default]: " input
    echo "${input:-$default}"    
}

region_check() {
    echo -e "${BLUE}Please confirm that your AWS region is ${GREEN}$AWS_REGION${BLUE} (default).${NC}"
    echo -e "${BLUE}If not, enter the AWS region where you want to set up your cluster (e.g., us-west-2):${NC}"
    
    read -p "> " NEW_REGION

    if [[ -z "$NEW_REGION" ]]; then
        echo -e "${GREEN}✅ Using default region: ${YELLOW}$AWS_REGION${NC}"
    else
        export AWS_REGION="$NEW_REGION"
        echo -e "${GREEN}✅ Region updated to: ${YELLOW}$AWS_REGION${NC}"
    fi    

    echo -e "\n${BLUE}Your region is set to: ${YELLOW}$AWS_REGION${NC}"
    echo -e "${BLUE}Ensure your chosen region supports SageMaker HyperPod.${NC}"
    echo -e "${GREEN}You can check out https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html#sagemaker-hyperpod-available-regions to learn about supported regions.${NC}"
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read
}

# Helper function to get all the env_vars set, depending on deployment type
check_and_prompt_env_vars() {
    echo -e "${BLUE}=== Checking Required Environment Variables ===${NC}"

    # Define variables and their default values using simple variables
    # Removed EKS_CLUSTER_ARN from the list since we'll set it programmatically
    VARS="AWS_REGION EKS_CLUSTER_NAME BUCKET_NAME EXECUTION_ROLE VPC_ID SUBNET_ID SECURITY_GROUP NODE_RECOVERY ACCEL_INSTANCE_TYPE ACCEL_COUNT ACCEL_VOLUME_SIZE GEN_INSTANCE_TYPE GEN_COUNT GEN_VOLUME_SIZE"
    
    # Check each variable and prompt if missing
    for var in $VARS; do
        if ! grep -q "export ${var}=" env_vars 2>/dev/null; then
            echo -e "${YELLOW}${var} is not set in env_vars${NC}"
            
            # Set default value based on variable name
            default=""
            case $var in
                "AWS_REGION")
                    default="us-west-2"
                    ;;
                "NODE_RECOVERY")
                    default="Automatic"
                    ;;
                "ACCEL_INSTANCE_TYPE")
                    default="ml.g5.12xlarge"
                    ;;
                "ACCEL_COUNT")
                    default="1"
                    ;;
                "ACCEL_VOLUME_SIZE")
                    default="500"
                    ;;
                "GEN_INSTANCE_TYPE")
                    default="ml.m5.2xlarge"
                    ;;
                "GEN_COUNT")
                    default="1"
                    ;;
                "GEN_VOLUME_SIZE")
                    default="500"
                    ;;
            esac
            
            while true; do
                value=$(get_input "${var}" "${default}")
                
                # First check if the value is empty
                if [[ -z "$value" ]]; then
                    echo -e "${RED}Empty value not allowed. Please provide a value for ${var}${NC}"
                    continue
                fi

                # Validation for specific variables
                case $var in
                    "NODE_RECOVERY")
                        if [[ "$value" =~ ^(Automatic|Manual)$ ]]; then
                            break
                        else
                            echo -e "${RED}Invalid input. Please enter either 'Automatic' or 'Manual'${NC}"
                        fi
                        ;;
                    "AWS_REGION")
                        if [[ "$value" =~ ^[a-z]+-[a-z]+-[0-9]+$ ]]; then
                            break
                        else
                            echo -e "${RED}Invalid region format. Please use format like 'us-west-2'${NC}"
                        fi
                        ;;
                    "EKS_CLUSTER_NAME")
                        # Verify the cluster exists and get its ARN
                        if cluster_arn=$(aws eks describe-cluster --name "$value" --query 'cluster.arn' --output text 2>/dev/null); then
                            echo "export EKS_CLUSTER_ARN=${cluster_arn}" >> env_vars
                            echo -e "${GREEN}✅ Retrieved and set EKS_CLUSTER_ARN${NC}"
                            break
                        else
                            echo -e "${RED}Could not find EKS cluster with name '$value'. Please verify the cluster exists.${NC}"
                        fi
                        ;;
                    *"_COUNT")
                        if [[ "$value" =~ ^[0-9]+$ ]]; then
                            break
                        else
                            echo -e "${RED}Invalid input. Please enter a positive number${NC}"
                        fi
                        ;;
                    *"_VOLUME_SIZE")
                        if [[ "$value" =~ ^[0-9]+$ ]] && [ "$value" -gt 0 ]; then
                            break
                        else
                            echo -e "${RED}Invalid input. Please enter a positive number for volume size${NC}"
                        fi
                        ;;
                    *"INSTANCE_TYPE")
                        if [[ "$value" =~ ^ml\.[a-z0-9]+\.[0-9]*xlarge$ ]]; then
                            break
                        else
                            echo -e "${RED}Invalid instance type format. Please use format like 'ml.g5.12xlarge' or 'ml.m5.2xlarge'${NC}"
                        fi
                        ;;
                    *)
                        # For any other variables, just ensure they're not empty
                        if [[ -n "$value" ]]; then
                            break
                        else
                            echo -e "${RED}Empty value not allowed. Please provide a value for ${var}${NC}"
                        fi
                        ;;
                esac
            done
            
            echo "export ${var}=${value}" >> env_vars
            echo -e "${GREEN}✅ Added ${var} to env_vars${NC}"
        fi
    done

    # Source the updated env_vars file
    source env_vars
    echo -e "${GREEN}=== Environment Variables Check Complete ===${NC}"
}





# Function to setup environment variables
setup_env_vars() {
    echo -e "${BLUE}=== Setting Up Environment Variables ===${NC}"
    echo -e "${GREEN}Cloning awsome-distributed-ai${NC}"
    clone_adt

    # Clear env_vars from previous runs
    > env_vars
    unset EKS_CLUSTER_NAME EKS_CLUSTER_ARN BUCKET_NAME EXECUTION_ROLE VPC_ID SUBNET_ID SECURITY_GROUP HP_CLUSTER_NAME ACCEL_INSTANCE_TYPE ACCEL_COUNT ACCEL_VOLUME_SIZE GEN_INSTANCE_TYPE GEN_COUNT GEN_VOLUME_SIZE NODE_RECOVERY

    echo -e "${BLUE}Enter the name of the SageMaker VPC CloudFormation stack that was deployed as a prerequisite (default: hyperpod-eks-full-stack):${NC}"
    read -e STACK_ID_VPC
    export STACK_ID=${STACK_ID_VPC:-hyperpod-eks-full-stack}

    if [ "$STACK_ID" != "hyperpod-eks-full-stack" ]; then
        echo -e "${GREEN}✅ Configuration script updated with stack name: $STACK_ID${NC}"
    else
        echo -e "${GREEN}Using default stack name: hyperpod-eks-full-stack${NC}"
    fi

    echo -e "${YELLOW}Generating new environment variables...${NC}"
    
    generate_env_vars() {
        ./awsome-distributed-ai/architectures/sagemaker-hyperpod-eks/create_config.sh
        # bash create_config.sh
    }

    # Capture stdout + stderr
    if error_output=$(generate_env_vars 2>&1); then
        echo -e "${GREEN}✅ New environment variables generated and sourced${NC}"
    else
        echo -e "${YELLOW}⚠️  Error occurred while generating environment variables:${NC}"
        echo -e "${YELLOW}$error_output${NC}"
        echo -e "Options:"
        echo -e "1. Press Enter to continue with the rest of the script (Not Recommended, unless you know how to set the environment variables manually!)"
        echo -e "2. Press Ctrl+C to exit the script."

        read -e -p "Select an option (Enter/Ctrl+C): " choice

        if [[ -z "$choice" ]]; then
            echo -e "${BLUE}Continuing with the rest of the script...${NC}"
        fi
    fi    

    source env_vars

    check_and_prompt_env_vars

    echo -e "\n${BLUE}=== Environment Variables Summary ===${NC}"
    echo -e "${YELLOW}Note: You may ignore the INSTANCES parameter for now${NC}"
    echo -e "${GREEN}Current environment variables:${NC}"
    cat env_vars

    echo -e "\n${BLUE}=== Environment Setup Complete ===${NC}"
}

# Function to setup lifecycle scripts
setup_lifecycle_scripts() {
    echo -e "${BLUE}=== Setting Up Lifecycle Scripts ===${NC}"

    cd awsome-distributed-ai/architectures/sagemaker-hyperpod-eks/LifecycleScripts/base-config/


    echo -e "${BLUE}Uploading your lifecycle scripts to S3 bucket ${YELLOW}${BUCKET}${NC}"
    # upload data
    upload_to_s3() {
        aws s3 cp on_create.sh s3://${BUCKET_NAME} --output json
    }

    if error_output=$(upload_to_s3 2>&1); then
        echo -e "${GREEN}✅ Lifecycle scripts uploaded successfully${NC}"
    else
        echo -e "${YELLOW}⚠️  Error occurred while uploading lifecycle scripts to S3 bucket:${NC}"
        echo -e "${YELLOW}$error_output${NC}"
        echo -e "Options:"
        echo -e "1. Press Enter to continue with the rest of the script (Not Recommended, unless you know how to set the environment variables manually!)"
        echo -e "2. Press Ctrl+C to exit the script."

        read -e -p "Select an option (Enter/Ctrl+C): " choice

        if [[ -z "$choice" ]]; then
            echo -e "${BLUE}Continuing with the rest of the script...${NC}"
        else
            exit 1
        fi
    fi  

    # move back to env_var directory
    cd ../../../../..

    echo -e "\n${BLUE}=== Lifecycle Scripts Setup Complete ===${NC}"
}

# Function to configure EKS cluster
configure_eks_cluster() {
    echo -e "${BLUE}=== Configuring your EKS Cluster ===${NC}"
    
    # Helper function for error handling
    handle_error() {
        local error_output=$1
        local step_name=$2
        echo -e "${YELLOW}⚠️  Error occurred during ${step_name}:${NC}"
        echo -e "${YELLOW}$error_output${NC}"
        echo -e "Options:"
        echo -e "1. Press Enter to continue with the rest of the script (Not Recommended unless you understand the implications!)"
        echo -e "2. Press Ctrl+C to exit the script."

        read -e -p "Select an option (Enter/Ctrl+C): " choice

        if [[ -z "$choice" ]]; then
            echo -e "${BLUE}Continuing with the rest of the script...${NC}"
            return 0
        else
            return 1
        fi
    }

    # Get current IAM user/role information
    echo -e "${YELLOW}Checking current IAM identity...${NC}"
    if ! error_output=$(CALLER_IDENTITY=$(aws sts get-caller-identity --output json) 2>&1); then
        if ! handle_error "$error_output" "IAM identity check"; then
            return 1
        fi
    fi

    CALLER_IDENTITY=$(aws sts get-caller-identity --output json)

    ACCOUNT_ID=$(echo "$CALLER_IDENTITY" | jq -r .Account)
    USER_ARN=$(echo "$CALLER_IDENTITY" | jq -r .Arn)
    PRINCIPAL_TYPE=$(echo "$USER_ARN" | cut -d':' -f6 | cut -d'/' -f1)
    USER_NAME=$(echo "$USER_ARN" | cut -d'/' -f2)

    echo -e "${GREEN}✅ Current identity:${NC}"
    echo -e "Account ID: ${YELLOW}$ACCOUNT_ID${NC}"
    echo -e "ARN: ${YELLOW}$USER_ARN${NC}"
    echo -e "Principal Type: ${YELLOW}$PRINCIPAL_TYPE${NC}"

    echo -e "${YELLOW}Note: You are authenticated using an IAM $PRINCIPAL_TYPE $USER_NAME${NC}"

    if [[ $PRINCIPAL_TYPE == "assumed-role" ]]; then
        # Extract the role name from assumed-role ARN
        ROLE_NAME=$(echo "$USER_ARN" | cut -d'/' -f2)
        USER_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
        echo -e "${YELLOW}Converting assumed-role ARN to IAM role ARN: ${USER_ARN}${NC}"
    fi        

    echo -e "${GREEN}Note: By default, Amazon EKS will automatically create an AccessEntry with the AmazonEKSClusterAdminPolicy for the IAM principal that you use to deploy the CloudFormation stack. To allow other users entry, check out https://catalog.us-east-1.prod.workshops.aws/workshops/2433d39e-ccfe-4c00-9d3d-9917b729258e/en-US/10-tips/07-add-users${NC}"

    # Create access entry for current user
    echo -e "${YELLOW}Adding current $PRINCIPAL_TYPE $USER_NAME to $EKS_CLUSTER_NAME...${NC}"

    if ! error_output=$(aws eks create-access-entry \
        --cluster-name "$EKS_CLUSTER_NAME" \
        --principal-arn "$USER_ARN" \
        --type "STANDARD" 2>&1); then
        if ! handle_error "$error_output" "access entry creation"; then
            return 1
        fi
    else
        echo -e "${GREEN}✅ Successfully created access entry${NC}"
    fi

    # Associate admin policy
    echo -e "${YELLOW}Associating admin policy...${NC}"
    if ! error_output=$(aws eks associate-access-policy \
        --cluster-name "$EKS_CLUSTER_NAME" \
        --principal-arn "$USER_ARN" \
        --policy-arn "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy" \
        --access-scope '{"type": "cluster"}' 2>&1); then
        if ! handle_error "$error_output" "policy association"; then
            return 1
        fi
    else
        echo -e "${GREEN}✅ Successfully associated admin policy${NC}"
    fi

    # Update kubeconfig
    echo -e "${YELLOW}Updating kubeconfig for cluster ${EKS_CLUSTER_NAME}...${NC}"
    if ! error_output=$(aws eks update-kubeconfig --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" 2>&1); then
        if ! handle_error "$error_output" "kubeconfig update"; then
            return 1
        fi
    else
        echo -e "${GREEN}✅ Successfully updated kubeconfig${NC}"
    fi

    # Verify current context
    echo -e "${YELLOW}Verifying current context...${NC}"
    if ! error_output=$(current_context=$(kubectl config current-context) 2>&1); then
        if ! handle_error "$error_output" "context verification"; then
            return 1
        fi
    else
        current_context=$(kubectl config current-context)
        echo -e "${GREEN}Current context: ${current_context}${NC}"
        
        confirm_context=$(get_input "Is this the correct context? (y/n)" "y")
        
        if [[ "$confirm_context" == "n" || "$confirm_context" == "N" ]]; then
            echo -e "${YELLOW}Available contexts:${NC}"
            kubectl config get-contexts -o name | nl
            
            context_number=$(get_input "Enter the number of the context you want to use" "1")
            
            # Get the selected context name
            selected_context=$(kubectl config get-contexts -o name | sed -n "${context_number}p")
            
            if [[ -n "$selected_context" ]]; then
                if ! error_output=$(kubectl config use-context "$selected_context" 2>&1); then
                    if ! handle_error "$error_output" "context switch"; then
                        return 1
                    fi
                else
                    echo -e "${GREEN}✅ Switched to context: ${selected_context}${NC}"
                    current_context="$selected_context"
                fi
            else
                echo -e "${RED}❌ Invalid context number${NC}"
                if ! handle_error "Invalid context number selected" "context selection"; then
                    return 1
                fi
            fi
        else
            echo -e "${GREEN}✅ Continuing with current context: ${current_context}${NC}"
        fi
    fi

    # Check cluster services
    echo -e "${YELLOW}Checking cluster services to test...${NC}"
    svc_output=$(kubectl get svc 2>&1)
    if ! error_output=${svc_output}; then
        if ! handle_error "$error_output" "cluster services check"; then
            return 1
        fi
    else
        echo -e "${GREEN}✅ Successfully retrieved cluster services${NC}"
        echo -e "${GREEN}Here's the output of kubectl get svc${NC}"

        echo -e "${YELLOW}$(kubectl get svc)${NC}"
    fi

    # Offer to add additional users
    add_users=$(get_input "Would you like to add additional ADMIN users to the cluster? (y/n)" "n")

    if [[ $add_users == "y" ]]; then
        echo -e "${YELLOW}Please enter usernames (one per line). Press Ctrl+D when finished:${NC}"
        TMP_USERS_FILE=$(mktemp)
        cat > "$TMP_USERS_FILE"

        while read -r username; do
            echo -e "${YELLOW}Adding user: $username${NC}"
            
            # Create access entry
            if ! error_output=$(aws eks create-access-entry \
                --cluster-name "$EKS_CLUSTER_NAME" \
                --principal-arn "arn:aws:iam::${ACCOUNT_ID}:user/${username}" \
                --type "STANDARD" 2>&1); then
                if ! handle_error "$error_output" "access entry creation for $username"; then
                    rm -f "$TMP_USERS_FILE"
                    return 1
                fi
            fi

            # Associate admin policy
            if ! error_output=$(aws eks associate-access-policy \
                --cluster-name "$EKS_CLUSTER_NAME" \
                --principal-arn "arn:aws:iam::${ACCOUNT_ID}:user/${username}" \
                --policy-arn "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy" \
                --access-scope '{"type": "cluster"}' 2>&1); then
                if ! handle_error "$error_output" "policy association for $username"; then
                    rm -f "$TMP_USERS_FILE"
                    return 1
                fi
            fi

            echo -e "${GREEN}✅ Successfully added user $username${NC}"
        done < "$TMP_USERS_FILE"

        rm -f "$TMP_USERS_FILE"
    fi

    echo -e "\n${GREEN}=== EKS Cluster Configuration Complete ===${NC}"
    echo -e "${BLUE}Your cluster ${EKS_CLUSTER_NAME} is now configured and ready to use${NC}"
}

# Function to install EKS dependencies
# TODO: TEST
install_eks_dependencies() {
    echo -e "${BLUE}=== Installing EKS Dependencies ===${NC}"

    # Create temporary directory for cloning
    TMP_DIR=$(mktemp -d)
    echo -e "${YELLOW}Created temporary directory: ${TMP_DIR}${NC}"

    # Clone the repository
    echo -e "${YELLOW}Cloning SageMaker HyperPod CLI repository...${NC}"
    if ! error_output=$(git clone https://github.com/aws/sagemaker-hyperpod-cli.git "${TMP_DIR}/sagemaker-hyperpod-cli" 2>&1); then
        handle_error "$error_output" "repository cloning"
    else
        echo -e "${GREEN}✅ Repository cloned successfully${NC}"
    fi

    # Change to helm chart directory
    cd "${TMP_DIR}/sagemaker-hyperpod-cli/helm_chart" || {
        handle_error "Failed to change to helm chart directory" "directory navigation"
    }

    # Lint the helm chart
    echo -e "${YELLOW}Linting Helm chart...${NC}"
    if ! error_output=$(helm lint HyperPodHelmChart 2>&1); then
        handle_error "$error_output" "helm lint"
    else
        echo -e "${GREEN}✅ Helm chart lint successful${NC}"
    fi

    # Update dependencies
    echo -e "${YELLOW}Updating Helm dependencies...${NC}"
    if ! error_output=$(helm dependencies update HyperPodHelmChart 2>&1); then
        handle_error "$error_output" "helm dependencies update"
    else
        echo -e "${GREEN}✅ Helm dependencies updated successfully${NC}"
    fi

    # Dry run
    echo -e "${YELLOW}Performing Helm dry run...${NC}"
    if ! error_output=$(helm install hyperpod-dependencies HyperPodHelmChart --dry-run 2>&1); then
        handle_error "$error_output" "helm dry run"
    else
        echo -e "${GREEN}✅ Helm dry run successful${NC}"
    fi

    # Actual deployment
    echo -e "${YELLOW}Deploying Helm chart...${NC}"
    if ! error_output=$(helm install hyperpod-dependencies HyperPodHelmChart 2>&1); then
        handle_error "$error_output" "helm installation"
    else
        echo -e "${GREEN}✅ Helm chart deployed successfully${NC}"
    fi

    # Verify deployment
    echo -e "${YELLOW}Verifying deployment...${NC}"
    if ! error_output=$(helm list 2>&1); then
        handle_error "$error_output" "helm verification"
    else
        echo -e "${GREEN}✅ Helm deployment verified. Here's all of your helm deployments listed${NC}"
        echo -e "${YELLOW}$(helm list)${NC}"
    fi

    # Return to original directory and cleanup
    echo -e "${YELLOW}Cleaning up...${NC}"
    cd - > /dev/null || {
        handle_error "Failed to return to original directory" "directory navigation"
    }
    
    if ! error_output=$(rm -rf "${TMP_DIR}" 2>&1); then
        handle_error "$error_output" "temporary directory cleanup"
    else
        echo -e "${GREEN}✅ Cleanup completed${NC}"
    fi

    echo -e "\n${GREEN}=== EKS Dependencies Installation Complete ===${NC}"
}

create_cluster_config() {
    echo -e "${BLUE}=== Creating Cluster Configuration ===${NC}"
    
    # Get cluster name
    local cluster_name=$(get_input "Enter cluster name" "ml-cluster")

    # Initialize instance groups array
    local instance_groups="["
    local group_count=1
    local first_group=true

    # Configure accelerator groups
    while true; do
        if [[ $first_group == true ]]; then
            echo -e "${BLUE}=== Configuring primary accelerator worker group ===${NC}"
            first_group=false
        else
            if [[ $(get_input "Do you want to add another accelerator worker group? (yes/no)" "no") == "yes" ]]; then
                echo -e "${BLUE}=== Configuring additional accelerator worker group ===${NC}"
                instance_groups+=","
            else
                break
            fi
        fi

        # Get accelerator configuration
        local group_name=$(get_input "Enter worker group name" "worker-group-${group_count}")
        local instance_type=$(get_input "Enter instance type" "$ACCEL_INSTANCE_TYPE")
        local instance_count=$(get_input "Enter number of instances" "$ACCEL_COUNT")

        # Training plan configuration for this worker group
        local TRAINING_PLAN_ARN=""
        if [[ $(get_input "Are you using training plans for this worker group? (yes/no)" "no") == "yes" ]]; then
            while true; do
                TRAINING_PLAN=$(get_input "Enter the training plan name" "")

                # Attempt to describe the training plan
                echo -e "${YELLOW}Attempting to retrieve training plan details...${NC}"
                
                if ! TRAINING_PLAN_DESCRIPTION=$(aws sagemaker describe-training-plan --training-plan-name "$TRAINING_PLAN" --output json 2>&1); then
                    echo -e "${BLUE}❌Error: Training plan '$TRAINING_PLAN' not found. Please try again.${NC}"
                    if [[ $(get_input "Would you like to try another training plan? (yes/no)" "yes") != "yes" ]]; then
                        echo -e "${YELLOW}Exiting training plan configuration.${NC}"
                        break
                    fi
                else
                    # Extract relevant information from the description
                    TRAINING_PLAN_ARN=$(echo "$TRAINING_PLAN_DESCRIPTION" | jq -r '.TrainingPlanArn')
                    AVAILABLE_INSTANCE_COUNT=$(echo "$TRAINING_PLAN_DESCRIPTION" | jq -r '.AvailableInstanceCount')
                    TOTAL_INSTANCE_COUNT=$(echo "$TRAINING_PLAN_DESCRIPTION" | jq -r '.TotalInstanceCount')
                    TRAINING_PLAN_AZ=$(echo "$TRAINING_PLAN_DESCRIPTION" | jq -r '.ReservedCapacitySummaries[0].AvailabilityZone')
                    TP_INSTANCE_TYPE=$(echo "$TRAINING_PLAN_DESCRIPTION" | jq -r '.ReservedCapacitySummaries[0].InstanceType')

                    CF_AZ=$(aws ec2 describe-subnets --subnet-ids $SUBNET_ID --output json | jq -r '.Subnets[0].AvailabilityZone')

                    echo -e "${GREEN}Training Plan Details:${NC}"
                    echo -e "  ${YELLOW}Name:${NC} $TRAINING_PLAN"
                    echo -e "  ${YELLOW}Available Instance Count:${NC} $AVAILABLE_INSTANCE_COUNT"
                    echo -e "  ${YELLOW}Total Instance Count:${NC} $TOTAL_INSTANCE_COUNT"
                    echo -e "  ${YELLOW}Training Plan Availability Zone:${NC} $TRAINING_PLAN_AZ"
                    echo -e "  ${YELLOW}Training Plan Instance Type:${NC} $TP_INSTANCE_TYPE"

                    # Validate configuration against training plan
                    local validation_passed=true

                    if [[ $instance_count -gt $AVAILABLE_INSTANCE_COUNT ]]; then
                        echo -e "${YELLOW}Warning: The requested instance count ($instance_count) is greater than the available instances in the training plan ($AVAILABLE_INSTANCE_COUNT).${NC}"
                        if [[ $(get_input "Do you want to continue anyway? (yes/no)" "no") != "yes" ]]; then
                            echo -e "${BLUE}Would you like to update the instance count? (yes/no)${NC}"
                            if [[ $(get_input "Update instance count?" "yes") == "yes" ]]; then
                                instance_count=$(get_input "Enter the new number of instances" "1")
                                echo -e "${GREEN}Updated instance count to $instance_count${NC}"
                                validation_passed=true
                            else
                                validation_passed=false
                            fi
                        fi
                    fi

                    if [[ $instance_type != $TP_INSTANCE_TYPE ]]; then
                        echo -e "${YELLOW}Warning: The requested instance type ($instance_type) does not match the instance type in the training plan ($TP_INSTANCE_TYPE).${NC}"
                        echo -e "${BLUE}Do you want to continue anyway? If you choose \"no\", then the script will update instance type for you and proceed. (yes/no)${NC}"
                        if [[ $(get_input "Continue with mismatched instance type?" "no") != "yes" ]]; then
                            instance_type=$TP_INSTANCE_TYPE
                            echo -e "${GREEN}Updated instance type to $instance_type${NC}"
                        fi
                    fi

                    if [[ $TRAINING_PLAN_AZ != $CF_AZ ]]; then
                        echo -e "${YELLOW}Warning: The training plan availability zone ($TRAINING_PLAN_AZ) does not match the cluster availability zone ($CF_AZ).${NC}"
                        if [[ $(get_input "Do you want to continue anyway? (yes/no)" "no") != "yes" ]]; then
                            validation_passed=false
                        fi
                    fi

                    if [[ $validation_passed == true ]]; then
                        break
                    else
                        echo -e "${YELLOW}Training plan validation failed. Please try another training plan.${NC}"
                        if [[ $(get_input "Would you like to try another training plan? (yes/no)" "yes") != "yes" ]]; then
                            echo -e "${YELLOW}Exiting training plan configuration.${NC}"
                            TRAINING_PLAN_ARN=""
                            break
                        fi
                    fi
                fi
            done
        fi

        # Get health check configuration
        echo -e "${BLUE}=== Configuring health checks ===${NC}"
        local health_checks=()

        if [[ $(get_input "Would you like to enable instance stress test? (yes/no)" "yes") == "yes" ]]; then
            health_checks+=("InstanceStress")
        fi
        if [[ $(get_input "Would you like to enable instance connectivity test? (yes/no)" "yes") == "yes" ]]; then
            health_checks+=("InstanceConnectivity")
        fi


        # Add accelerator group configuration
        instance_groups+="    
        {
            \"InstanceGroupName\": \"${group_name}\",
            \"InstanceType\": \"${instance_type}\",
            \"InstanceCount\": ${instance_count},
            \"InstanceStorageConfigs\": [
                {
                    \"EbsVolumeConfig\": {
                        \"VolumeSizeInGB\": ${ACCEL_VOLUME_SIZE}
                    }
                }
            ],
            \"LifeCycleConfig\": {
                \"SourceS3Uri\": \"s3://${BUCKET_NAME}\",
                \"OnCreate\": \"on_create.sh\"
            },
            \"ExecutionRole\": \"${EXECUTION_ROLE}\",
            \"ThreadsPerCore\": 2"

        # Add health checks if any were selected
        if [ ${#health_checks[@]} -gt 0 ]; then
            local formatted_checks=""
            for check in "${health_checks[@]}"; do
                if [ -z "$formatted_checks" ]; then
                    formatted_checks="\"$check\""
                else
                    formatted_checks="$formatted_checks, \"$check\""
                fi
            done
            instance_groups+=",
            \"OnStartDeepHealthChecks\": [${formatted_checks}]"
        fi


        # Add TrainingPlanArn if it exists for this worker group
        if [[ -n "$TRAINING_PLAN_ARN" ]]; then
            instance_groups+=",
            \"TrainingPlanArn\": \"${TRAINING_PLAN_ARN}\""
        fi

        instance_groups+="
        }"
        
        group_count=$((group_count + 1))
    done

    # Configure general purpose groups
    while [[ $(get_input "Do you want to add a general purpose worker group? (yes/no)" "no") == "yes" ]]; do
        echo -e "${BLUE}=== Configuring general purpose worker group ===${NC}"
        instance_groups+=","

        # Get general purpose configuration
        local group_name=$(get_input "Enter worker group name" "worker-group-${group_count}")
        local instance_type=$(get_input "Enter instance type" "$GEN_INSTANCE_TYPE")
        local instance_count=$(get_input "Enter number of instances" "$GEN_COUNT")

        # Add general purpose group configuration
        instance_groups+="    
        {
            \"InstanceGroupName\": \"${group_name}\",
            \"InstanceType\": \"${instance_type}\",
            \"InstanceCount\": ${instance_count},
            \"InstanceStorageConfigs\": [
                {
                    \"EbsVolumeConfig\": {
                        \"VolumeSizeInGB\": ${GEN_VOLUME_SIZE}
                    }
                }
            ],
            \"LifeCycleConfig\": {
                \"SourceS3Uri\": \"s3://${BUCKET_NAME}\",
                \"OnCreate\": \"on_create.sh\"
            },
            \"ExecutionRole\": \"${EXECUTION_ROLE}\",
            \"ThreadsPerCore\": 1
        }"
        
        group_count=$((group_count + 1))

        if [[ $(get_input "Do you want to add another general purpose worker group? (yes/no)" "no") != "yes" ]]; then
            break
        fi
        instance_groups+=","
    done

    # Complete the JSON structure
    local config=$(cat << EOF
{
    "ClusterName": "${cluster_name}",
    "Orchestrator": {
        "Eks": {
            "ClusterArn": "${EKS_CLUSTER_ARN}"
        }
    },
    "InstanceGroups": ${instance_groups}],
    "VpcConfig": {
        "SecurityGroupIds": ["${SECURITY_GROUP}"],
        "Subnets": ["${SUBNET_ID}"]
    },
    "NodeRecovery": "${NODE_RECOVERY}"
}
EOF
)

    # Write to file
    echo "$config" > cluster-config.json
    echo -e "${GREEN}✅ Created cluster configuration in cluster-config.json${NC}"
}

# Function to display the prerequisites before starting this workshop
display_important_prereqs() {
    echo -e "${BLUE}Before running this script, please ensure the following:${NC}\n"

    echo -e "${GREEN}1. 🔑 IAM Credentials:${NC}"
    echo "   You have Administrator Access Credentials in IAM."
    echo "   This is crucial as we'll be using CloudFormation to create IAM roles and policies."
    echo "   Run 'aws configure' to set up your credentials."

    echo -e "\n${GREEN}2. 🌐 VPC Stack:${NC}"
    echo "   Deploy the hyperpod-eks-full-stack using:"
    echo "   https://catalog.us-east-1.prod.workshops.aws/workshops/2433d39e-ccfe-4c00-9d3d-9917b729258e/en-US/00-setup/02-own-account"
    echo -e "   Choose between ${BLUE}\"Full Deployment\"${NC}, ${BLUE}\"Integrative Deployment\"${NC}, and ${BLUE}\"Minimal Deployment\"${NC}, depending on what you'd like to provision. This script accounts for all of these deployment modes."
    echo -e "   ${YELLOW}Full Deployment: Creates all prerequisite resources$ (10 minutes),${NC}"
    echo -e "   ${YELLOW}Integrative Deployment: Uses existing VPC and EKS cluster, but creates new network resources(subnet, security group, CIDR block), S3 bucket, and IAM role for the HyperPod cluster. (3 minutes),${NC}"
    echo -e "   ${YELLOW}Minimal Deployment: Uses exising VPC, network resources and EKS cluster. Creates only S3 bucket IAM role for the HyperPod cluster. (1 minute),${NC}"

    echo -e "\n${GREEN}3. 💻 Development Environment:${NC}"
    echo "   Ensure you have a Linux-based development environment (macOS works great too)."

    echo -e "\n${GREEN}5. 🔧 Packages required for this script to run:${NC}"
    echo "   Please ensure that jq is installed prior to running this script. All other packages will be installed by the script, if not installed already."

    echo -e "\n${YELLOW}Ready to proceed? Press Enter to continue or Ctrl+C to exit...${NC}"
    read
}

# Function to create the cluster
create_cluster() {
    echo -e "${GREEN}✅ Creating cluster for you!${NC}"

    if ! output=$(aws sagemaker create-cluster \
        --cli-input-json file://cluster-config.json \
        --region $AWS_REGION \
        --output json 2>&1); then

        echo -e "${YELLOW}⚠️  Error occurred while creating the cluster:${NC}"
        echo -e "${YELLOW}$output${NC}"

        echo -e "Options:"
        echo -e "1. Press Enter to continue with the rest of the script (you will run the command below yourself!)"
        echo -e "2. Press Ctrl+C to exit the script."

        # Command to create the cluster
        echo -e "${GREEN} aws sagemaker create-cluster \\"
        echo -e "${GREEN}    --cli-input-json file://cluster-config.json \\"
        echo -e "${GREEN}    --region $AWS_REGION --output json${NC}\n"

        read -e -p "Select an option (Enter/Ctrl+C): " choice

        if [[ -z "$choice" ]]; then
            echo -e "${BLUE}Continuing with the rest of the script...${NC}"
        else
            exit 1
        fi
    else
        echo -e "${GREEN}✅ Cluster creation request submitted successfully. To monitor the progress of cluster creation, you can either check the SageMaker console, or you can run:.${NC}"    
        echo -e "${YELLOW}watch -n 1 aws sagemaker list-clusters --output table${NC}"
    fi
}

# Warning message function
warning() {
    echo -e "${BLUE}⚠️  Please note:${NC}"
    echo -e "   - Cluster creation may take some time (~15-20 min)"
    echo -e "   - This operation may incur costs on your AWS account"
    echo -e "   - Ensure you understand the implications before proceeding\n"
}

# Function to display goodbye message
goodbye() {
    # Final goodbye message
    echo -e "${GREEN}Thank you for using the SageMaker HyperPod Cluster Creation Script!${NC}"
    echo -e "${GREEN}For any issues or questions, please refer to the AWS documentation.${NC}"
    echo "https://docs.aws.amazon.com/sagemaker/latest/dg/smcluster-getting-started.html"

    # Exit message
    echo -e "\n${BLUE}Exiting script. Good luck with your SageMaker HyperPod journey! 👋${NC}\n"
}  


#===Main Script===
main() {
    print_header "🚀 Welcome to the SageMaker HyperPod EKS Cluster Creation Script! 🚀"

    # Prerequisites
    display_important_prereqs

    # Checking AWS Account ID
    echo -e "\n${BLUE}🔍 AWS Account Verification${NC}"
    echo -e "Your AWS Account ID is: ${GREEN}$AWS_ACCOUNT_ID${NC}"
    echo "Press Enter to confirm ✅ or Ctrl+C to exit❌..."
    read

    # Checking Git installation
    check_git

    # Checking AWS CLI version and installation
    echo -e "\n${BLUE}📦 AWS CLI Installation and Verification${NC}"
    check_and_install_aws_cli

    # Install other requirements
    echo -e "\n${BLUE}📦 kubectl, eksctl, helm installation verification${NC}"
    echo -e "\n${YELLOW}📦📦📦📦📦📦📦${NC}"
    install_kubectl
    echo -e "\n${YELLOW}📦📦📦📦📦📦📦${NC}"
    install_eksctl
    echo -e "\n${YELLOW}📦📦📦📦📦📦📦${NC}"
    install_helm

    # Checking Region
    echo -e "\n${BLUE}🌎 AWS Region Configuration${NC}"
    region_check

    # Lifecycle Scripts Setup
    echo -e "\n${BLUE}🔧 Setting Up Lifecycle Scripts & Environment Variables${NC}"
    echo -e "${BLUE}Configuring environment variables and lifecycle scripts...${NC}"
    setup_env_vars
    setup_lifecycle_scripts
    echo -e "${GREEN}✅ Lifecycle scripts setup completed${NC}"

    # Configuring EKS Cluster
    echo -e "\n${BLUE}⚙️ Configuring EKS Cluster${NC}"
    configure_eks_cluster
    install_eks_dependencies

    # Cluster Configuration
    echo -e "\n${BLUE}🚀 Creating the Cluster${NC}"
    echo -e "${BLUE}Generating cluster configuration...${NC}"
    create_cluster_config
    echo -e "${GREEN}✅ Cluster configuration created successfully${NC}"
    
    echo -e "${BLUE}ℹ️  For your viewing, here's the cluster configuration generated. Please make sure it looks right before proceeding. Press enter to continue, or Ctrl+C to exit and make changes${NC}"
    echo -e "${YELLOW}$(cat cluster-config.json | jq . --color-output)${NC}"
    read

    print_header "🎉 Cluster Creation Script Completed! 🎉"

    # Instructions for next steps
    echo -e "${GREEN}Congratulations! You've completed all the preparatory steps.${NC}"
    echo -e "${YELLOW}Next Steps:${NC}"

    CREATE_CLUSTER=$(get_input "Do you want the script to create the cluster for you now? (yes/no):" "yes")
    # read -e -p "Do you want the script to create the cluster for you now? (yes/no): " CREATE_CLUSTER
    if [[ "$CREATE_CLUSTER" == "yes" ]]; then
        warning
        create_cluster
        goodbye
    else
        echo -e "${YELLOW}Run the following command to create the cluster. Exiting this script!${NC}"

        # Command to create the cluster
        echo -e "${GREEN} aws sagemaker create-cluster \\"
        echo -e "${GREEN}    --cli-input-json file://cluster-config.json \\"
        echo -e "${GREEN}    --region $AWS_REGION --output json${NC}\n"

        echo -e "${YELLOW}To monitor the progress of cluster creation, you can either check the SageMaker console, or you can run:.${NC}"    
        echo -e "${GREEN}watch -n 1 aws sagemaker list-clusters --output table${NC}"

        \
        warning
        goodbye
    fi    
}

main
