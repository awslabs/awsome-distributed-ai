#!/bin/bash

# Workshop Automation Script for SageMaker HyperPod with EKS
# This script automates the steps of creating a HyperPod cluster with EKS orchestration

# Exit immediately if a command exits with a non-zero status. Print commands and their arguments as executed
set -e

# HAVE JQ INSTALLED!!!

#===Global===
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
export AWS_REGION=${AWS_DEFAULT_REGION:-$(aws configure get region)}
export K8S_VERSION="1.31"
export DEVICE=$(uname)
export OS=$(uname -m)
export BASE_URL="s3://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0/2433d39e-ccfe-4c00-9d3d-9917b729258e/"

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

# Helper function to enforce the use of Yy or Nn in binary responses
get_yes_no() {
    local prompt="$1"
    local default="$2"
    local answer
    local prompt_text
    
    # Set up prompt text based on default
    if [ "$default" = "y" ]; then
        prompt_text="$prompt (Y/n): "
    else
        prompt_text="$prompt (y/N): "
    fi
    
    while true; do
        echo -e -n "${Green}$prompt_text${NC}"
        read -r -n 1 answer
        
        # Handle the enter key (empty input)
        if [ -z "$answer" ]; then
            echo
            [ "$default" = "y" ] && return 0 || return 1
        fi
        
        echo # Move to a new line
        
        case $answer in
            [Yy]) return 0 ;;
            [Nn]) return 1 ;;
            *) echo "Please answer 'y' or 'n' or press enter for default" ;;
        esac
    done
}

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

check_jq() {
    if ! command -v jq &> /dev/null; then
        echo "jq is not installed. Please install jq and try again."
        exit 1
    fi
}

# Function to display the prerequisites before starting this workshop
display_important_prereqs() {
    echo -e "${BLUE}Before running this script, please ensure the following:${NC}\n"

    echo -e "${GREEN}1. 🔑 IAM Credentials:${NC}"
    echo "   You have Administrator Access Credentials in IAM."
    echo "   This is crucial as we'll be using CloudFormation to deploy supporting infrastructure."
    echo "   Run 'aws configure' to set up your credentials."

    echo -e "\n${GREEN}3. 💻 Development Environment:${NC}"
    echo "   Ensure you have a Linux-based development environment (macOS works great too)."

    echo -e "\n${GREEN}5. 🔧 Packages required for this script to run:${NC}"
    echo "   Please ensure that jq is installed prior to running this script. All other packages will be installed by the script, if not installed already."

    echo -e "\n${YELLOW}Ready to proceed? Press Enter to continue or Ctrl+C to exit...${NC}"
    read
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

# Helper function to get user inputs with default values specified
get_input() {
    local prompt="$1"
    local default="$2"
    local input
    read -e -p "$prompt [$default]: " input
    echo "${input:-$default}"    
}

# Function to clone the awsome-distributed-ai repository
clone_adt() {
    REPO_NAME="awsome-distributed-ai"
    if [ -d "$REPO_NAME" ]; then
        echo -e "${YELLOW}⚠️  The directory '$REPO_NAME' already exists.${NC}"
        if get_yes_no "Do you want to remove it and clone again?" "n"; then
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

unset_env_vars() {
    # Clear tmp_env_vars from previous runs
    > tmp_env_vars
    unset EKS_CLUSTER_NAME EKS_CLUSTER_ARN S3_BUCKET_NAME EXECUTION_ROLE VPC_ID PRIVATE_SUBNET_ID SECURITY_GROUP_ID HP_CLUSTER_NAME ACCEL_INSTANCE_TYPE ACCEL_INSTANCE_COUNT ACCEL_VOLUME_SIZE GEN_INSTANCE_TYPE GEN_INSTANCE_COUNT GEN_VOLUME_SIZE NODE_RECOVERY
}

# Function to setup environment variables
setup_env_vars() {
    echo -e "${BLUE}=== Setting Up Environment Variables ===${NC}"
    echo -e "${GREEN}Cloning awsome-distributed-ai${NC}"
    clone_adt

    export STACK_ID=${STACK_NAME:-hyperpod-eks-full-stack}
    # Source IDs / names of pre-existing resources set by the user
    source tmp_env_vars

    echo -e "Setting up environment variables from the ${STACK_NAME} CloudFormation stack outputs.."

    echo -e "${YELLOW}Generating new environment variables...${NC}"
    
    generate_env_vars() {
        ./awsome-distributed-ai/architectures/sagemaker-hyperpod-eks/create_config.sh
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

    # Source anything set by create_config.sh pulled from CloudFormation outputs or pre-set environment variables    
    source env_vars

    echo -e "\n${BLUE}=== Environment Variables Summary ===${NC}"
    echo -e "${YELLOW}Note: We'll walk you through updating the INSTANCE parameter defaults shortly.${NC}"
    echo -e "${GREEN}Current environment variables:${NC}"
    cat env_vars

    echo -e "\n${BLUE}=== Environment Setup Complete ===${NC}"
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
    echo -e "${GREEN}Note: By default, Amazon EKS will automatically create an AccessEntry with the AmazonEKSClusterAdminPolicy for the IAM principal that you use to create the EKS Cluster. To allow other users entry, check out https://catalog.us-east-1.prod.workshops.aws/workshops/2433d39e-ccfe-4c00-9d3d-9917b729258e/en-US/10-tips/07-add-users${NC}"

    # Skip access entry creation if the EKS Cluster Stack was deployed 
    # Using the current IAM entity (user or role) from this script 
    # Attempt to create if the EKS Cluster was previously created
    # Possibly by another IAM entity
    if [[ $CREATE_EKSCluster_STACK == "false" ]]; then 

        if [[ $PRINCIPAL_TYPE == "assumed-role" ]]; then
            # Extract the role name from assumed-role ARN
            ROLE_NAME=$(echo "$USER_ARN" | cut -d'/' -f2)
            USER_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
            echo -e "${YELLOW}Converting assumed-role ARN to IAM role ARN: ${USER_ARN}${NC}"
        fi    
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
        if ! get_yes_no "Is this the correct context?" "y"; then
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

    # Offer to add EKS access entries for other IAM principals (users or roles)
    echo -e "${BLUE}=== Adding EKS access entries ===${NC}"
    if get_yes_no "Would you like to create additional EKS access entries for IAM Users or Roles to allow ADMIN access to the EKS cluster?" "n"; then
        while true; do
            echo -e "${YELLOW}Select principal type to add:${NC}"
            echo "1) IAM User"
            echo "2) IAM Role"
            echo "3) Done adding principals"
            read -p "Enter your choice (1-3): " choice

            case $choice in
                1|2)
                    if [ "$choice" -eq 1 ]; then
                        principal_type="user"
                        read -p "Enter username: " principal_name
                    else
                        principal_type="role"
                        read -p "Enter role name: " principal_name
                    fi

                    # Skip if empty input
                    [ -z "$principal_name" ] && continue

                    # Validate principal name format
                    if ! [[ "$principal_name" =~ ^[a-zA-Z0-9+=,.@_-]+$ ]]; then
                        echo -e "${RED}Invalid $principal_type name format. Names can contain letters, numbers, and the following characters: +=,.@_-${NC}"
                        continue
                    fi

                    # Check if principal exists
                    if ! error_output=$(aws iam get-${principal_type} --${principal_type}-name "$principal_name" 2>&1); then
                        echo -e "${RED}Error: $principal_type '$principal_name' does not exist or you don't have permission to access it.${NC}"
                        continue
                    fi
                    
                    echo -e "${YELLOW}Adding $principal_type: $principal_name${NC}"
                    
                    # Create access entry
                    if ! error_output=$(aws eks create-access-entry \
                        --cluster-name "$EKS_CLUSTER_NAME" \
                        --principal-arn "arn:aws:iam::${ACCOUNT_ID}:${principal_type}/${principal_name}" \
                        --type "STANDARD" 2>&1); then
                        handle_error "$error_output" "access entry creation for $principal_name" || return 1
                    fi

                    # Associate admin policy
                    if ! error_output=$(aws eks associate-access-policy \
                        --cluster-name "$EKS_CLUSTER_NAME" \
                        --principal-arn "arn:aws:iam::${ACCOUNT_ID}:${principal_type}/${principal_name}" \
                        --policy-arn "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy" \
                        --access-scope '{"type": "cluster"}' 2>&1); then
                        handle_error "$error_output" "policy association for $principal_name" || return 1
                    fi

                    echo -e "${GREEN}✅ Successfully added $principal_type $principal_name${NC}"
                    echo -e "\n${YELLOW}Select next action:${NC}"
                    # Loop continues automatically to show menu again
                    ;;
                3)
                    echo -e "${GREEN}Finished adding principals${NC}"
                    break
                    ;;
                *)
                    echo -e "${RED}Invalid choice. Please enter 1, 2, or 3${NC}"
                    ;;
            esac
        done
    fi 

    echo -e "\n${GREEN}=== EKS Cluster Configuration Complete ===${NC}"
    echo -e "${BLUE}Your cluster ${EKS_CLUSTER_NAME} is now configured and ready to use${NC}"
}

create_hyperpod_cluster_config() {
    echo -e "${BLUE}=== Creating Cluster Configuration ===${NC}"
    
    # Get cluster name
    local cluster_name=$(get_input "Enter cluster name" "ml-cluster")

    # Initialize instance groups array
    local instance_groups="["
    local group_count=1
    local first_group="true"

    # Configure accelerator groups
    while true; do
        if [[ $first_group == "true" ]]; then
            echo -e "${BLUE}=== Configuring primary accelerator worker group ===${NC}"
            first_group="false"
        else
            if get_yes_no "Do you want to add another accelerator worker group?" "n"; then
                echo -e "${BLUE}=== Configuring additional accelerator worker group ===${NC}"
                instance_groups+=","
            else
                break
            fi
        fi

        # Get accelerator configuration
        local group_name=$(get_input "Enter worker group name" "worker-group-${group_count}")
        local instance_type=$(get_input "Enter instance type" "$ACCEL_INSTANCE_TYPE")
        local instance_count=$(get_input "Enter number of instances" "$ACCEL_INSTANCE_COUNT")

        # Training plan configuration for this worker group
        local TRAINING_PLAN_ARN=""
        if get_yes_no "Are you using training plans for this worker group?" "n"; then
            while true; do
                TRAINING_PLAN=$(get_input "Enter the training plan name" "")

                # Attempt to describe the training plan
                echo -e "${YELLOW}Attempting to retrieve training plan details...${NC}"
                
                if ! TRAINING_PLAN_DESCRIPTION=$(aws sagemaker describe-training-plan --training-plan-name "$TRAINING_PLAN" --output json 2>&1); then
                    echo -e "${BLUE}❌Error: Training plan '$TRAINING_PLAN' not found. Please try again.${NC}"
                    if ! get_yes_no "Would you like to try another training plan?" "y"; then
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

                    CF_AZ=$(aws ec2 describe-subnets --subnet-ids $PRIVATE_SUBNET_ID --output json | jq -r '.Subnets[0].AvailabilityZone')

                    echo -e "${GREEN}Training Plan Details:${NC}"
                    echo -e "  ${YELLOW}Name:${NC} $TRAINING_PLAN"
                    echo -e "  ${YELLOW}Available Instance Count:${NC} $AVAILABLE_INSTANCE_COUNT"
                    echo -e "  ${YELLOW}Total Instance Count:${NC} $TOTAL_INSTANCE_COUNT"
                    echo -e "  ${YELLOW}Training Plan Availability Zone:${NC} $TRAINING_PLAN_AZ"
                    echo -e "  ${YELLOW}Training Plan Instance Type:${NC} $TP_INSTANCE_TYPE"

                    # Validate configuration against training plan
                    local validation_passed="true"

                    if [[ $instance_count -gt $AVAILABLE_INSTANCE_COUNT ]]; then
                        echo -e "${YELLOW}Warning: The requested instance count ($instance_count) is greater than the available instances in the training plan ($AVAILABLE_INSTANCE_COUNT).${NC}"
                        if ! get_yes_no "Do you want to continue anyway?" "n"; then
                            if get_yes_no "Would you like to update the instance count?" "y"; then
                                instance_count=$(get_input "Enter the new number of instances" "1")
                                echo -e "${GREEN}Updated instance count to $instance_count${NC}"
                                validation_passed="true"
                            else
                                validation_passed="false"
                            fi
                        fi
                    fi

                    if [[ $instance_type != $TP_INSTANCE_TYPE ]]; then
                        echo -e "${YELLOW}Warning: The requested instance type ($instance_type) does not match the instance type in the training plan ($TP_INSTANCE_TYPE).${NC}"
                        echo -e "${BLUE}Do you want to continue anyway? If you choose \"n\", then the script will update instance type for you and proceed.${NC}"
                        if ! get_yes_no "Continue with mismatched instance type?" "n"; then
                            instance_type=$TP_INSTANCE_TYPE
                            echo -e "${GREEN}Updated instance type to $instance_type${NC}"
                        fi
                    fi

                    if [[ $TRAINING_PLAN_AZ != $CF_AZ ]]; then
                        echo -e "${YELLOW}Warning: The training plan availability zone ($TRAINING_PLAN_AZ) does not match the cluster availability zone ($CF_AZ).${NC}"
                        if ! get_yes_no "Do you want to continue anyway?" "n"; then
                            validation_passed="false"
                        fi
                    fi

                    if [[ $validation_passed == "true" ]]; then
                        break
                    else
                        echo -e "${YELLOW}Training plan validation failed. Please try another training plan.${NC}"
                        if ! get_yes_no "Would you like to try another training plan?" "y"; then
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
        if get_yes_no "Would you like to enable instance stress test?" "y"; then
            health_checks+=("InstanceStress")
        fi
        if get_yes_no "Would you like to enable instance connectivity test?" "y"; then
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
                \"SourceS3Uri\": \"s3://${S3_BUCKET_NAME}\",
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
    while get_yes_no "Do you want to add a general purpose worker group?" "n"; do
        echo -e "${BLUE}=== Configuring general purpose worker group ===${NC}"
        instance_groups+=","

        # Get general purpose configuration
        local group_name=$(get_input "Enter worker group name" "worker-group-${group_count}")
        local instance_type=$(get_input "Enter instance type" "$GEN_INSTANCE_TYPE")
        local instance_count=$(get_input "Enter number of instances" "$GEN_INSTANCE_COUNT")

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
                \"SourceS3Uri\": \"s3://${S3_BUCKET_NAME}\",
                \"OnCreate\": \"on_create.sh\"
            },
            \"ExecutionRole\": \"${EXECUTION_ROLE}\",
            \"ThreadsPerCore\": 1
        }"
        
        group_count=$((group_count + 1))

        if ! get_yes_no "Do you want to add another general purpose worker group?" "n"; then
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
        "SecurityGroupIds": ["${SECURITY_GROUP_ID}"],
        "Subnets": ["${PRIVATE_SUBNET_ID}"]
    },
    "NodeRecovery": "${NODE_RECOVERY}"
}
EOF
)

    # Write to file
    echo "$config" > cluster-config.json
    echo -e "${GREEN}✅ Created cluster configuration in cluster-config.json${NC}"
}

# Helper function for stack name validation
validate_stack_name() {
    local stack_name=$1
    local default_name=$2
    
    while true; do
        stack_name=$(get_input "Enter the CloudFormation stack name to use" "$default_name")
        if [[ $stack_name =~ ^[a-zA-Z][a-zA-Z0-9-]{0,127}$ ]]; then
            echo "$stack_name"
            return 0
        else
            echo -e "${RED}Error: Stack name must start with a letter, contain only letters, numbers, and hyphens, and be 128 characters or less in length.${NC}"
            echo -e "${RED}Please try again.${NC}"
        fi
    done
}

# Helper function for deploying CloudFormation stacks
deploy_stack() {
    local stack_name=$1
    local template_url=$2
    local parameter_overrides=("${@:3}")  # Get remaining arguments as array

    # Create a temporary file to store the template
    local temp_file=$(mktemp)
    
    # Download the template from S3
    if ! aws s3 cp "$template_url" "$temp_file"; then
        echo -e "${RED}Error: Failed to download template from S3${NC}"
        rm -f "$temp_file"
        exit 1
    fi
    
    # Build the parameter override string
    local parameter_override_string=""
    if [ ${#parameter_overrides[@]} -gt 0 ]; then
        parameter_override_string="--parameter-overrides ${parameter_overrides[*]}"
    fi

    # Build deployment command
    local deploy_cmd="aws cloudformation deploy \
        --template-file ${temp_file} \
        --stack-name ${stack_name} \
        --region ${AWS_REGION} \
        --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
        ${parameter_override_string}"
        
    # Add profile parameter only if AWS_PROFILE is set or default profile exists
    if [[ -n "${AWS_PROFILE}" ]]; then
        deploy_cmd+=" --profile ${AWS_PROFILE}"
    elif aws configure list-profiles 2>/dev/null | grep -q "^default$"; then
        deploy_cmd+=" --profile default"
    fi

    echo -e "${GREEN}✨ Deploying CloudFormation stack ${stack_name} to region ${AWS_REGION}...${NC}"
    echo -e "${GREEN}This will take approximately 15 minutes to complete...${NC}"
    if [ ${#parameter_overrides[@]} -gt 0 ]; then
        echo "Using parameters:"
        for override in "${parameter_overrides[@]}"; do
            # Temporary Suppression
            if [[ "${override}" != "CreateHyperPodClusterStack=false" ]]; then 
                echo "  $override"
            fi
        done
    fi 

    # Dry-run?
    if [[ "${DRY_RUN:-"false"}" == "true" ]]; then
        echo "Would execute:"
        echo "$deploy_cmd"

    # Execute deployment
    else
        # if ! eval "$deploy_cmd"; then
        #     echo -e "${RED}Error: CloudFormation stack deployment for ${stack_name} failed${NC}"
        #     exit 1
        # fi

        # echo -e "${GREEN}✅ CloudFormation stack deployment for ${stack_name} completed successfully!${NC}"

        eval "$deploy_cmd" > /dev/null 2>&1 &

        # Give CloudFormation a moment to initiate the stack creation
        sleep 10

        # Wait for stack to complete (with timeout)
        max_attempts=60 # 30 minutes
        attempt=1
        while [ $attempt -le $max_attempts ]; do
            # Add error handling for the describe-stacks command
            stack_status=$(aws cloudformation describe-stacks --stack-name "$stack_name" --query 'Stacks[0].StackStatus' --output text 2>/dev/null)
            
            if [ $? -ne 0 ]; then
                if [ $attempt -eq $max_attempts ]; then
                    echo -e "${RED}Error: Unable to get stack status after maximum attempts${NC}"
                    exit 1
                fi
                echo -e "${YELLOW}Waiting for stack creation to begin...${NC}"
                sleep 10
                ((attempt++))
                continue
            fi
            
            if [[ $stack_status == *"ROLLBACK"* || $stack_status == *"FAILED"* ]]; then
                echo -e "${RED}Error: CloudFormation stack deployment for ${stack_name} failed with status: ${stack_status}${NC}"
                exit 1
            elif [[ $stack_status == *"COMPLETE"* ]]; then
                echo -e "${GREEN}✅ CloudFormation stack deployment for ${stack_name} completed successfully with status: ${stack_status}${NC}"
                break
            else
                if [ $attempt -eq $max_attempts ]; then
                    echo -e "${YELLOW}Warning: Reached maximum attempts waiting for stack completion. Current status: ${stack_status}${NC}"
                    exit 1
                fi
                echo -e "${YELLOW}Stack ${stack_name} is still in progress (${stack_status}). Waiting...${NC}"
                sleep 30
                ((attempt++))
            fi
        done
    fi
}

# Helper function to validate basic CIDR format
validate_cidr_format() {
    local cidr=$1
    # Check for valid IPv4 CIDR format and prefix length
    if [[ $cidr =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}/([0-9]|[1-2][0-9]|3[0-2])$ ]]; then
        # Extract the IP parts
        IFS='/' read -r ip_part prefix_length <<< "$cidr" 
        IFS='.' read -r a b c d <<< "$ip_part"
        
        # Validate each octet is <= 255
        if [[ $a -le 255 && $b -le 255 && $c -le 255 && $d -le 255 ]]; then
            # Check if it's not a reserved/special use IP range
            # 0.0.0.0/8 (Current network)
            # 127.0.0.0/8 (Loopback)
            # 169.254.0.0/16 (Link-local)
            # 224.0.0.0/4 (Multicast)
            # 240.0.0.0/4 (Reserved)
            if ! { \
                [[ $a == 0 ]] || \
                [[ $a == 127 ]] || \
                [[ $a == 169 && $b == 254 ]] || \
                [[ $a == 224 ]] || \
                [[ $a == 240 ]] || \
                [[ $a == 255 && $b == 255 && $c == 255 && $d == 255 ]]; # 255.255.255.255 (Broadcast)
            }; then
                return 0  # Valid CIDR format
            fi
        fi
    fi
    return 1  # Invalid CIDR format
}

# Validate AWS VPC CIDR range
validate_vpc_cidr() {
    local cidr=$1
    
    # First check basic CIDR format
    if ! validate_cidr_format "$cidr"; then
        echo "Invalid CIDR format"
        return 1
    fi 
    
    # Extract prefix length
    local prefix_length
    prefix_length=$(echo "$cidr" | cut -d'/' -f2)
    
    # Check if prefix length is within AWS VPC limits (between /16 and /28)
    if [[ $prefix_length -lt 16 || $prefix_length -gt 28 ]]; then
        echo "VPC CIDR block size must be between /16 and /28"
        return 1
    fi 
    
    # Extract IP parts
    local ip_part
    ip_part=$(echo "$cidr" | cut -d'/' -f1)
    IFS='.' read -r a b c d <<< "$ip_part"
    
    # Check for reserved AWS ranges
    if [[ $a == 198 && $b == 19 && $c == 255 ]]; then
        echo "198.19.255.0/24 is reserved for use by AWS"
        return 1
    fi
    
    # Check for restricted ranges
    # 0.0.0.0/8
    # 127.0.0.0/8 (Loopback)
    # 169.254.0.0/16 (Link Local)
    # 224.0.0.0/4 (Multicast)
    if [[ $a == 0 ]] || \
       [[ $a == 127 ]] || \
       [[ $a == 169 && $b == 254 ]] || \
       [[ $a == 224 ]] || \
       [[ $a == 255 && $b == 255 && $c == 255 && $d == 255 ]]; then # Broadcast
        echo "Reserved or restricted IP range"
        return 1
    fi

    return 0
}


# Helper function to validate resource IDs
validate_resource_id() {
    local resource_id=$1
    local resource_type=$2

    # Define patterns and AWS CLI commands based on resource type
    local pattern
    local aws_command
    
    case $resource_type in
        "vpc")
            pattern="^vpc-([a-f0-9]{8}|[a-f0-9]{17})$"
            aws_command="aws ec2 describe-vpcs --vpc-ids"
            ;;
        "nat")
            pattern="^nat-([a-f0-9]{8}|[a-f0-9]{17})$"
            aws_command="aws ec2 describe-nat-gateways --nat-gateway-ids"
            ;;
        "subnet")
            pattern="^subnet-([a-f0-9]{8}|[a-f0-9]{17})$"
            aws_command="aws ec2 describe-subnets --subnet-ids"
            ;;
        "az")
            pattern="^[a-z]{3,4}[0-9]-az[0-9]$"
            aws_command="aws ec2 describe-availability-zones --zone-ids"
            ;;
        "rtb")
            pattern="^rtb-([a-f0-9]{8}|[a-f0-9]{17})$"
            aws_command="aws ec2 describe-route-tables --route-table-ids"
            ;;
        "eks")
            pattern="^[[:alnum:]][[:alnum:]_-]{0,99}$"
            aws_command="aws eks describe-cluster --name"
            ;;
        "s3")
            pattern="^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
            aws_command="aws s3api head-bucket --bucket"
            ;;
        "iam")
            pattern="^[a-zA-Z0-9_+=,.@-]{1,64}$"
            aws_command="aws iam get-role --role-name"
            ;;
        *)
            echo "Unsupported resource type: $resource_type"
            return 1
            ;;
    esac

    # Check format
    if [[ $resource_id =~ $pattern ]]; then
        # Check if resource exists
        if $aws_command "$resource_id" >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

config_resource_prefix() {
    while true; do 
        export RESOURCE_PREFIX=$(get_input "Enter the prefix you want to use to name CloudFormation resources" "sagemaker-hyperpod-eks")
        if [[ $RESOURCE_PREFIX =~ ^[A-Za-z][A-Za-z0-9]*([-][A-Za-z0-9]+)*$ && ${#RESOURCE_PREFIX} -le 28 ]]; then
            break
        else
            echo -e "${RED}Invalid resource prefix. Please try again.${NC}"
            echo -e "${RED}Ensure that the resource prefix is between 1-28 characters long, begins with a letter; contain only ASCII letters, digits, and hyphens; and does not end with a hyphen or contain two consecutive hyphens.${NC}"
        fi
    done
}

deploy_code_editor_stack() {
    if ! get_yes_no "Do you want to deploy the SageMaker Studio Code Editor CloudFormation Stack to create an IDE?" "y"; then
        echo -e "${YELLOW}Skipping SageMaker Studio Code Editor CloudFormation Stack deployment...${NC}"
        return 0
    fi

    SMCE_STACK_NAME=$(validate_stack_name "" "sagemaker-studio-code-editor")
    
    # Initialize parameter overrides array
    local parameter_overrides=()

    # Set ResourceNamePrefix
    parameter_overrides+=("ResourceNamePrefix=${RESOURCE_PREFIX}")
    
    # Handle VPC parameter
    if ! get_yes_no "Do you want to use the default VPC for SageMaker Studio Code Editor?" "y"; then
        echo -e "${GREEN}Creating a new VPC for Studio Code Editor.${NC}"
        parameter_overrides+=("UseDefaultVpc=false")
    fi

    # Deploy the stack
    deploy_stack "$SMCE_STACK_NAME" "${BASE_URL}sagemaker-studio-stack.yaml" "${parameter_overrides[@]}"
}

# Configure parameters for the VPC Stack
config_vpc_stack() {
    if get_yes_no "Do you want to create a new VPC to use with HyperPod?" "y"; then 
        # Using an alternate VPC CIDR ranges
        if ! get_yes_no "Do you want to use the default VPC CIDR range 10.192.0.0/16?" "y"; then

            while true; do 
                VPC_CIDR=$(get_input "Enter the VPC CIDR range" "10.192.0.0/16")
                if validate_vpc_cidr "$VPC_CIDR"; then
                    break
                else
                    echo -e "${RED}Invalid VPC CIDR range. Please try again.${NC}"
                fi
            done

            while true; do
                PUBLIC_SUBNET1_CIDR=$(get_input "Enter the CIDR range to use for public subnet 1" "10.192.10.0/24")
                if validate_cidr_format "$PUBLIC_SUBNET1_CIDR"; then
                    break
                else
                    echo -e "${RED}Invalid CIDR range for public subnet 1. Please try again.${NC}"
                fi  
            done

            while true; do
                PUBLIC_SUBNET2_CIDR=$(get_input "Enter the CIDR range to use for public subnet 2" "10.192.11.0/24")
                if validate_cidr_format "$PUBLIC_SUBNET2_CIDR"; then
                    break
                else
                    echo -e "${RED}Invalid CIDR range for public subnet 2. Please try again.${NC}"
                fi
            done

        else 
            # Using an alternate CIDR range for public subnet 1
            if ! get_yes_no "Do you want to use the default CIDR range 10.192.10.0/24 for public subnet 1?" "y"; then
                while true; do
                    PUBLIC_SUBNET1_CIDR=$(get_input "Enter the CIDR range to use for public subnet 1" "10.192.10.0/24")
                    if validate_cidr_format "$PUBLIC_SUBNET1_CIDR"; then
                        break
                    else
                        echo -e "${RED}Invalid CIDR range for public subnet 1. Please try again.${NC}"
                    fi  
                done
            fi

            # Using an alternate CIDR range for public subnet 2
            if ! get_yes_no "Do you want to use the default CIDR range 10.192.11.0/24 for public subnet 2?" "y"; then
                while true; do
                    PUBLIC_SUBNET2_CIDR=$(get_input "Enter the CIDR range to use for public subnet 2" "10.192.11.0/24")
                    if validate_cidr_format "$PUBLIC_SUBNET2_CIDR"; then
                        break
                    else
                        echo -e "${RED}Invalid CIDR range for public subnet 2. Please try again.${NC}"
                    fi
                done
            fi
        fi
    # Using an existing VPC 
    else 
        export CREATE_VPC_STACK="false"
        # Get the VPC ID
        while true; do 
            export VPC_ID=$(get_input "Enter the VPC ID you want to use" "")
            if validate_resource_id "$VPC_ID" "vpc"; then
                echo "export VPC_ID=${VPC_ID}" >> tmp_env_vars
                break
            else
                echo -e "${RED}Invalid VPC ID. Please try again.${NC}"
            fi
        done
        # Get the NAT Gateway ID
        while true; do
            NAT_GATEWAY_ID=$(get_input "Enter the NAT Gateway ID you want to use" "")
            if validate_resource_id "$NAT_GATEWAY_ID" "nat"; then
                break
            else
                echo -e "${RED}Invalid NAT Gateway ID. Please try again.${NC}"
            fi
        done
    fi
}

# Configure parameters for the Private Subnet Stack
config_private_subnet_stack() {
    if get_yes_no "Do you want to create a new private subnet to use with HyperPod?" "y"; then 
        # Get the availability zone ID
        # Get the first available AZ ID from the region
        DEFAULT_AZ_ID=$(aws ec2 describe-availability-zones \
            --region $AWS_REGION \
            --query 'AvailabilityZones[0].ZoneId' \
            --output text)

        while true; do 
            AZ_ID=$(get_input "Enter the ID of the Availability Zone where you want to create the private subnet" "$DEFAULT_AZ_ID")
            if validate_resource_id "$AZ_ID" "az"; then
                break
            else
                echo -e "${RED}Invalid Availability Zone ID. Please try again.${NC}"
            fi
        done
        # Get the private subnet cidr range
        if ! get_yes_no "Do you want to use the default CIDR range 10.1.0.0/16 for the private subnet?" "y"; then 
            while true; do
                PRIVATE_SUBNET1_CIDR=$(get_input "Enter the CIDR range you want to use for the private subnet" "10.1.0.0/16")
                if validate_cidr_format "$PRIVATE_SUBNET1_CIDR"; then
                    break
                else
                    echo -e "${RED}Invalid CIDR range for the private subnet. Please try again.${NC}"
                fi
            done
        fi
    # Using an existing Private Subnet
    else
        export CREATE_PrivateSubnet_STACK="false"
        # Get the private subnet ID
        while true; do
            export PRIVATE_SUBNET_ID=$(get_input "Enter the private subnet ID you want to use" "")
            if validate_resource_id "$PRIVATE_SUBNET_ID" "subnet"; then 
                echo "export PRIVATE_SUBNET_ID=${PRIVATE_SUBNET_ID}" >> tmp_env_vars
                break
            else
                echo -e "${RED}Invalid private subnet ID. Please try again.${NC}"
            fi 
        done
        # Get the private route table ID
        while true; do
            PRIVATE_ROUTE_TABLE_ID=$(get_input "Enter the private route table ID you want to use" "")
            if validate_resource_id "$PRIVATE_ROUTE_TABLE_ID" "rtb"; then
                break
            else
                echo -e "${RED}Invalid private route table ID. Please try again.${NC}"
            fi 
        done
    fi
}

# Configure parameters for the EKS Cluster Stack
config_eks_cluster_stack() {
    if get_yes_no "Do you want to create a new EKS cluster to use with HyperPod?" "y"; then 
        if ! get_yes_no "Do you want to use the default CIDR range 10.192.7.0/28 for EKS private subnet 1?" "y"; then
            while true; do
                EKS_PRIVATE_SUBNET1_CIDR=$(get_input "Enter the CIDR range you want to use for EKS private subnet 1" "10.192.7.0/28")
                if validate_cidr_format "$EKS_PRIVATE_SUBNET1_CIDR"; then 
                    break
                else
                    echo -e "${RED}Invalid CIDR range for EKS private subnet 1. Please try again.${NC}"
                fi 
            done
        fi
        if ! get_yes_no "Do you want to use the default CIDR range 10.192.8.0/28 for EKS private subnet 2?" "y"; then 
            while true; do
                EKS_PRIVATE_SUBNET2_CIDR=$(get_input "Enter the CIDR range you want to use for EKS private subnet 2" "10.192.8.0/28")
                if validate_cidr_format "$EKS_PRIVATE_SUBNET2_CIDR"; then 
                    break
                else
                    echo -e "${RED}Invalid CIDR range for EKS private subnet 2. Please try again.${NC}"
                fi 
            done
        fi 
        if ! get_yes_no "Do you want to use the latest supported version of Kubernetes ($K8S_VERSION)?" "y"; then
            while true; do
                K8S_VERSION=$(get_input "Enter the Kubernetes version you want to use (1.29, 1.30, 1.31)" "1.31")
                case "$K8S_VERSION" in
                    1.29|1.30|1.31) 
                        break
                        ;;
                    *)
                        echo -e "${RED}Invalid Kubernetes version. Please try again.${NC}"
                        ;;
                esac 
            done
        fi 
        if ! get_yes_no "Do you want to use the default EKS cluster name sagemaker-hyperpod-eks-cluster?" "y"; then
            while true; do
                export EKS_CLUSTER_NAME=$(get_input "Enter the EKS cluster name you want to use" "sagemaker-hyperpod-eks-cluster")
                if [[ $EKS_CLUSTER_NAME =~ ^[[:alnum:]][[:alnum:]_-]{0,99}$ ]]; then
                    break
                else
                    echo -e "${RED}Invalid EKS cluster name. Please try again.${NC}"
                fi
            done
        fi
    # Using an existing EKS cluster
    else 
        export CREATE_EKSCluster_STACK="false"
        while true; do 
            export EKS_CLUSTER_NAME=$(get_input "Enter the name of the EKS cluster you want to use" "")
            if validate_resource_id "$EKS_CLUSTER_NAME" "eks"; then
                echo "export EKS_CLUSTER_NAME=${EKS_CLUSTER_NAME}" >> tmp_env_vars
                break
            else
                echo -e "${RED}Invalid EKS cluster name. Please try again.${NC}"
            fi
        done
        # Get the EKS cluster security group ID 
        export SECURITY_GROUP_ID=$(aws eks describe-cluster \
            --name "$EKS_CLUSTER_NAME" \
            --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" \
            --output text)
        # Verify we got the security group
        if [[ -z "$SECURITY_GROUP_ID" ]]; then
            echo -e "${RED}Failed to get the EKS cluster security group ID.${NC}"
            exit 1
        else 
            echo "export SECURITY_GROUP_ID=${SECURITY_GROUP_ID}" >> tmp_env_vars
        fi
    fi 
    # Create access entry for SageMaker Code Editor? 
    # This is done automatically by the EKS cluster stack (eks-cluster-stack.yaml) 
    # If the Code Editor stack (sagemaker-studio-stack.yaml) was previously deployed in the same account and region
    # Using the SageMakerStudioExecutionRoleArn exported output
    if [[ -n "${SMCE_STACK_NAME}" ]]; then
        echo -e "${GREEN}SageMaker Studio Code Editor CloudFormation Stack ${SMCE_STACK_NAME} previously deployed."
        echo -e "${GREEN}Creating an EKS access entry for SageMaker Studio Code Editor.${NC}"

        # Create an access entry for an existing EKS cluster
        if [[ $CREATE_EKSCluster_STACK == "false" ]]; then 
            # Get role arn
            if ! SMCE_ROLE_ARN=$(aws cloudformation describe-stacks \
                --stack-name "$SMCE_STACK_NAME" \
                --query 'Stacks[].Outputs[?OutputKey==`SageMakerStudioExecutionRoleArn`][].OutputValue' \
                --output text); then
                echo "Failed to get SageMaker Studio execution role ARN from stack $SMCE_STACK_NAME"
                if get_yes_no "Skip and continue?" "y"; then
                    echo -e "${YELLOW}Skipping SageMaker Studio Code Editor access entry creation...${NC}"
                    return 0
                else 
                    exit 1
                fi
            fi
            # Validate we got a role ARN
            if [[ -z "$SMCE_ROLE_ARN" ]]; then
                echo "SageMaker Studio execution role ARN not found in stack $SMCE_STACK_NAME outputs"
                if get_yes_no "Skip and continue?" "y"; then
                    echo -e "${YELLOW}Skipping SageMaker Studio Code Editor access entry creation...${NC}"
                    return 0
                else
                    exit 1
                fi
            fi
            # Create the access entry
            if ! aws eks create-access-entry \
                --cluster-name "$EKS_CLUSTER_NAME" \
                --principal-arn "$SMCE_ROLE_ARN" \
                --type STANDARD; then
                echo "Failed to create EKS access entry"
                if get_yes_no "Skip and continue?" "y"; then
                    echo -e "${YELLOW}Skipping SageMaker Studio Code Editor access entry creation...${NC}"
                    return 0
                else 
                    exit 1
                fi
            fi
            # Associate admin access policy 
            if ! aws eks associate-access-policy \
                --cluster-name "$EKS_CLUSTER_NAME" \
                --principal-arn "$SMCE_ROLE_ARN" \
                --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
                --access-scope type=cluster; then
                echo "Failed to associate admin access policy"
                if get_yes_no "Skip and continue?" "y"; then
                    echo -e "${YELLOW}Skipping SageMaker Studio Code Editor access entry creation...${NC}"
                    return 0
                else 
                    exit 1
                fi
            fi
        # Create an access entry for the new EKS cluster
        else 
            USING_SM_CODE_EDITOR="true"
        fi
    fi 
}

# Configure parameters for the S3 Bucket Stack 
config_s3_bucket_stack() {
    if ! get_yes_no "Do you want to create a new S3 bucket to store the HyperPod lifecycle script?" "y"; then
        export CREATE_S3Bucket_STACK="false"
        while true; do
            export S3_BUCKET_NAME=$(get_input "Enter the name of the S3 bucket you want to use to store the HyperPod lifecycle script." "")
            if validate_resource_id "$S3_BUCKET_NAME" "s3"; then
                echo "export S3_BUCKET_NAME=${S3_BUCKET_NAME}" >> tmp_env_vars
                break
            else
                echo -e "${RED}Invalid S3 bucket name. Please try again.${NC}"
            fi
        done
    fi
}

# Configure parameters for the SageMaker IAM Role Stack
config_sagemaker_iam_role_stack() {
    echo -e "${GREEN}Your HyperPod cluster needs to assume an IAM execution role to run and communicate with necessary AWS resources on your behalf.${NC}"
    if ! get_yes_no "Do you want to create a new IAM execution role for your HyperPod cluster to assume?" "y"; then
        export CREATE_SageMakerIAMRole_STACK="false"
        while true; do
            export SAGEMAKER_IAM_ROLE_NAME=$(get_input "Enter the name of the IAM execution role you want to use for HyperPod." "")
            if validate_resource_id "$SAGEMAKER_IAM_ROLE_NAME" "iam"; then
                echo "export EXECUTION_ROLE=\"arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_IAM_ROLE_NAME}\"" >> tmp_env_vars
                break
            else
                echo -e "${RED}Invalid IAM role name. Please try again.${NC}"
            fi
        done
    fi
}

# Deploy the Main Stacks
deploy_main_stack() {
    echo -e "${GREEN}The Main CloudFormation Stack is now configured and ready to create the supporting workshop infrastructure.${NC}"
    
    # Print stack creation summary
    echo -e "\n${GREEN}The Main CloudFormation Stack will deploy the following resources:${NC}"
    # Print resources (excludes HyperPodCluster from display)
    for stack in VPC PrivateSubnet SecurityGroup EKSCluster S3Bucket LifeCycleScript SageMakerIAMRole HelmChart; do
        env_var="CREATE_${stack}_STACK"
        if [[ "${!env_var}" != "false" ]]; then
            # Check for adding rules to an existing security group pulled from existing EKS cluster
            if [[ "$stack" == "SecurityGroup" ]]; then 
                if [[ "$CREATE_EKSCluster_STACK" == "false" ]]; then 
                    echo -e "  ${CYAN}✓${NC} $stack ${YELLOW}(Adding rules to existing Security Group: $SECURITY_GROUP_ID)${NC}"
                else
                    echo -e "  ${CYAN}✓${NC} $stack"
                fi
    
            else 
                echo -e "  ${CYAN}✓${NC} $stack"
            fi
        else
            # Get the corresponding resource ID based on stack type
            resource_id=""
            case $stack in
                "VPC")
                    [[ -n "$VPC_ID" ]] && resource_id=" (Using existing VPC: $VPC_ID)"
                    ;;
                "PrivateSubnet")
                    [[ -n "$PRIVATE_SUBNET_ID" ]] && resource_id=" (Using existing Subnet: $PRIVATE_SUBNET_ID)"
                    ;;
                "SecurityGroup")
                    [[ -n "$SECURITY_GROUP_ID" ]] && resource_id=" (Using existing Security Group: $SECURITY_GROUP_ID)"
                    ;;
                "EKSCluster")
                    [[ -n "$EKS_CLUSTER_NAME" ]] && resource_id=" (Using existing EKS Cluster: $EKS_CLUSTER_NAME)"
                    ;;
                "S3Bucket")
                    [[ -n "$S3_BUCKET_NAME" ]] && resource_id=" (Using existing S3 Bucket: $S3_BUCKET_NAME)"
                    ;;
                "SageMakerIAMRole")
                    [[ -n "$SAGEMAKER_IAM_ROLE_NAME" ]] && resource_id=" (Using existing IAM Role: $SAGEMAKER_IAM_ROLE_NAME)"
                    ;;
            esac
            echo -e "  ${RED}✗${NC} $stack${YELLOW}${resource_id}${NC}"
        fi
    done
    echo ""

    if ! get_yes_no "Do you want to deploy the Main CloudFormation Stack now?" "y"; then
        echo -e "${YELLOW}The Main CloudFormation Stack must be deployed to create the supporting workshop infrastructure.${NC}"
        echo -e "${YELLOW}This includes networking, security, and IAM resources required by SageMaker HyperPod.${NC}"
        echo -e "${RED}Exiting script - please run again when ready to deploy the infrastructure.${NC}"
        exit 0
    fi

    local stack_name=$(validate_stack_name "" "hyperpod-eks-full-stack")
    export STACK_NAME="$stack_name"

    # Initialize parameter overrides array
    local parameter_overrides=()

    # Function to add parameter if environment variable is set
    add_parameter_if_set() {
        local param_name=$1
        local env_var_name=$2
        if [[ -n "${!env_var_name}" ]]; then
            parameter_overrides+=("${param_name}=${!env_var_name}")
        fi
    }

    # Resource Name Prefix Parameter
    add_parameter_if_set "ResourceNamePrefix" "RESOURCE_PREFIX"

    # VPC Parameters
    add_parameter_if_set "VpcCIDR" "VPC_CIDR"
    add_parameter_if_set "PublicSubnet1CIDR" "PUBLIC_SUBNET1_CIDR"
    add_parameter_if_set "PublicSubnet2CIDR" "PUBLIC_SUBNET2_CIDR"

    # Private Subnet Parameters
    add_parameter_if_set "AvailabilityZoneId" "AZ_ID"
    add_parameter_if_set "PrivateSubnet1CIDR" "PRIVATE_SUBNET1_CIDR"
    add_parameter_if_set "VpcId" "VPC_ID"
    add_parameter_if_set "NatGatewayId" "NAT_GATEWAY_ID"

    # EKS Parameters
    add_parameter_if_set "KubernetesVersion" "K8S_VERSION"
    add_parameter_if_set "EKSClusterName" "EKS_CLUSTER_NAME"
    add_parameter_if_set "EKSPrivateSubnet1CIDR" "EKS_PRIVATE_SUBNET1_CIDR"
    add_parameter_if_set "EKSPrivateSubnet2CIDR" "EKS_PRIVATE_SUBNET2_CIDR"
    add_parameter_if_set "SecurityGroupId" "SECURITY_GROUP_ID"
    add_parameter_if_set "UsingSMCodeEditor" "USING_SM_CODE_EDITOR"
    # add_parameter_if_set "ParticipantRoleArn" "PARTICIPANT_ROLE_ARN"

    # S3 Bucket Parameters
    add_parameter_if_set "PrivateRouteTableId" "PRIVATE_ROUTE_TABLE_ID"

    # Lifecycle Script Parameters
    add_parameter_if_set "S3BucketName" "S3_BUCKET_NAME"

    # Helm Installer Parameters
    add_parameter_if_set "HelmRepoUrl" "HELM_REPO_URL"
    add_parameter_if_set "HelmRepoPath" "HELM_REPO_PATH"
    add_parameter_if_set "HelmRelease" "HELM_RELEASE"
    add_parameter_if_set "CustomResourceS3Bucket" "CUSTOM_RESOURCE_S3_BUCKET"
    add_parameter_if_set "LayerS3Key" "LAYER_S3_KEY"
    add_parameter_if_set "FunctionS3Key" "FUNCTION_S3_KEY"


    # ==== Disable HyperPod CloudFormation =====
    # Until training plan properties 
    # And instance group resources are supported
    # ==========================================

    # HyperPod Parameters
    # add_parameter_if_set "HyperPodClusterName" "HYPERPOD_CLUSTER_NAME"
    # add_parameter_if_set "NodeRecovery" "NODE_RECOVERY"
    # add_parameter_if_set "SageMakerIAMRoleName" "SAGEMAKER_IAM_ROLE_NAME"
    # add_parameter_if_set "PrivateSubnetId" "PRIVATE_SUBNET_ID"

    # Accelerated Instance Group Parameters
    # add_parameter_if_set "AcceleratedInstanceGroupName" "ACCELERATED_GROUP_NAME"
    # add_parameter_if_set "AcceleratedInstanceType" "ACCELERATED_INSTANCE_TYPE"
    # add_parameter_if_set "AcceleratedInstanceCount" "ACCELERATED_INSTANCE_COUNT"
    # add_parameter_if_set "AcceleratedEBSVolumeSize" "ACCELERATED_EBS_SIZE"
    # add_parameter_if_set "AcceleratedThreadsPerCore" "ACCELERATED_THREADS_PER_CORE"
    # add_parameter_if_set "EnableInstanceStressCheck" "ENABLE_STRESS_CHECK"
    # add_parameter_if_set "EnableInstanceConnectivityCheck" "ENABLE_CONNECTIVITY_CHECK"
    # add_parameter_if_set "AcceleratedLifeCycleConfigOnCreate" "ACCELERATED_LIFECYCLE_CONFIG_ON_CREATE"

    # General Purpose Instance Group Parameters
    # add_parameter_if_set "CreateGeneralPurposeInstanceGroup" "CREATE_GENERAL_PURPOSE_GROUP"
    # add_parameter_if_set "GeneralPurposeInstanceGroupName" "GENERAL_PURPOSE_GROUP_NAME"
    # add_parameter_if_set "GeneralPurposeInstanceType" "GENERAL_PURPOSE_INSTANCE_TYPE"
    # add_parameter_if_set "GeneralPurposeInstanceCount" "GENERAL_PURPOSE_INSTANCE_COUNT"
    # add_parameter_if_set "GeneralPurposeEBSVolumeSize" "GENERAL_PURPOSE_EBS_SIZE"
    # add_parameter_if_set "GeneralPurposeThreadsPerCore" "GENERAL_PURPOSE_THREADS_PER_CORE"
    # add_parameter_if_set "GeneralPurposeLifeCycleConfigOnCreate" "GENERAL_PURPOSE_LIFECYCLE_CONFIG_ON_CREATE"

    # Disable HyperPod CloudFormation 
    export CREATE_HyperPodCluster_STACK="false"

    # Stack Creation Flags (includes HyperPodCluster to disable)
    for stack in VPC PrivateSubnet SecurityGroup EKSCluster S3Bucket LifeCycleScript SageMakerIAMRole HelmChart HyperPodCluster; do
        env_var="CREATE_${stack}_STACK"
        add_parameter_if_set "Create${stack}Stack" "$env_var"
    done
    
    # Deploy the stack
    deploy_stack "$STACK_NAME" "${BASE_URL}main-stack.yaml" "${parameter_overrides[@]}"
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

     # Check jq instillation 
    check_jq

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

    unset_env_vars

    # Configure CloudFormation stacks
    config_resource_prefix
    deploy_code_editor_stack
    config_vpc_stack
    config_private_subnet_stack
    config_eks_cluster_stack
    config_s3_bucket_stack
    config_sagemaker_iam_role_stack
    
    # Deploy CloudFormation stacks
    deploy_main_stack

    # Environment variables
    echo -e "${BLUE}Configuring environment variables...${NC}"
    setup_env_vars

    # Configuring EKS Cluster
    echo -e "\n${BLUE}⚙️ Configuring EKS Cluster${NC}"
    configure_eks_cluster

    # HyperPod Cluster Configuration
    echo -e "\n${BLUE}🚀 Creating the HyperPod Cluster${NC}"
    echo -e "${BLUE}Generating cluster configuration...${NC}"
    create_hyperpod_cluster_config
    echo -e "${GREEN}✅ HyperPod Cluster configuration created successfully${NC}"
    
    echo -e "${YELLOW}$(cat cluster-config.json | jq . --color-output)${NC}"
    echo -e "${BLUE}ℹ️ Above is the generated HyperPod Cluster configuration. Please make sure it looks correct before proceeding. Press enter to continue, or Ctrl+C to exit and make changes${NC}"
    read

    print_header "🎉 Cluster Creation Script Completed! 🎉"

    # Instructions for manual next steps
    echo -e "${GREEN}Congratulations! You've completed all the preparatory steps.${NC}"
    echo -e "${YELLOW}Next Steps:${NC}"

    if get_yes_no "Do you want the script to create the HyperPod cluster for you now?" "y"; then
        warning
        create_cluster
        goodbye
    else
        echo -e "${YELLOW}Run the following command to create the HyperPod cluster. Exiting this script!${NC}"

        # Command to create the HyperPod Cluster
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
