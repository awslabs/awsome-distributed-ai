#!/bin/bash

# Script to list ECR repositories that would be created
# Usage: ./list-ecr-repos.sh [AWS_REGION] [AWS_ACCOUNT_ID]

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/ecr-images.conf"

# Configuration
DEFAULT_REGION="us-east-2"
REGION="${1:-$DEFAULT_REGION}"
ACCOUNT_ID="${2:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "YOUR_ACCOUNT_ID")}"

echo "📋 ECR Repositories Configuration"
echo "Region: $REGION"
echo "Account ID: $ACCOUNT_ID"
echo ""

if [[ -f "$CONFIG_FILE" ]]; then
    echo "🏗️  Repositories that will be created/used:"
    
    # First, show skipped images (AWS-managed, available in all regions)
    echo "  ⏭️  aws-efa-k8s-device-plugin (SKIPPED - available in all AWS regions)"
    echo "    Will use: 602401143452.dkr.ecr.$REGION.amazonaws.com/eks/aws-efa-k8s-device-plugin:v0.5.6"
    echo ""
    
    # Get the regional account ID for health monitoring agent
    case "$REGION" in
        "us-east-1") HMA_ACCOUNT="767398015722" ;;
        "us-west-2") HMA_ACCOUNT="905418368575" ;;
        "us-east-2") HMA_ACCOUNT="851725546812" ;;
        "us-west-1") HMA_ACCOUNT="011528288828" ;;
        "eu-central-1") HMA_ACCOUNT="211125453373" ;;
        "eu-north-1") HMA_ACCOUNT="654654141839" ;;
        "eu-west-1") HMA_ACCOUNT="533267293120" ;;
        "eu-west-2") HMA_ACCOUNT="011528288831" ;;
        "ap-northeast-1") HMA_ACCOUNT="533267052152" ;;
        "ap-south-1") HMA_ACCOUNT="011528288864" ;;
        "ap-southeast-1") HMA_ACCOUNT="905418428165" ;;
        "ap-southeast-2") HMA_ACCOUNT="851725636348" ;;
        "sa-east-1") HMA_ACCOUNT="025066253954" ;;
        *) HMA_ACCOUNT="905418368575" ;;  # Default to us-west-2
    esac
    echo "  ⏭️  hyperpod-health-monitoring-agent (SKIPPED - available in all AWS regions with regional account IDs)"
    echo "    Will use: $HMA_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/hyperpod-health-monitoring-agent:1.0.819.0_1.0.267.0"
    echo ""
    
    # Now process active (non-commented) images that will be copied
    while IFS='=' read -r repo_name source_image; do
        # Skip comments and empty lines
        [[ "$repo_name" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$repo_name" ]] && continue
        
        # Remove leading/trailing whitespace
        repo_name=$(echo "$repo_name" | xargs)
        source_image=$(echo "$source_image" | xargs)
        
        echo "  • $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repo_name"
        echo "    Source: $source_image"
        echo ""
    done < "$CONFIG_FILE"
else
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "💡 To copy these images to ECR, run:"
echo "   make copy-images-to-ecr"
echo "   # or with custom region/account:"
echo "   make copy-images-to-ecr REGION=us-west-2 ACCOUNT_ID=123456789012"