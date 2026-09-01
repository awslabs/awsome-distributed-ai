#!/bin/bash

# Script to copy container images to ECR repositories
# Usage: ./copy-images-to-ecr.sh [AWS_REGION] [AWS_ACCOUNT_ID]

set -e

# Retry function for network operations
retry_command() {
    local max_attempts=3
    local delay=5
    local attempt=1
    local command="$@"
    
    while [ $attempt -le $max_attempts ]; do
        echo "  🔄 Attempt $attempt of $max_attempts..."
        if eval "$command"; then
            return 0
        else
            if [ $attempt -eq $max_attempts ]; then
                echo "  ❌ Command failed after $max_attempts attempts"
                return 1
            fi
            echo "  ⏳ Waiting ${delay}s before retry..."
            sleep $delay
            attempt=$((attempt + 1))
            delay=$((delay * 2))  # Exponential backoff
        fi
    done
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/ecr-images.conf"

# Configuration
DEFAULT_REGION="us-east-2"
REGION="${1:-$DEFAULT_REGION}"
ACCOUNT_ID="${2:-$(aws sts get-caller-identity --query Account --output text)}"

# Configure Docker for better network handling
export DOCKER_BUILDKIT=0  # Disable BuildKit for better compatibility
export DOCKER_CLI_EXPERIMENTAL=disabled

# Load images from configuration file
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "🚀 Starting ECR image copy process..."
echo "Region: $REGION"
echo "Account ID: $ACCOUNT_ID"

# Check if jq is available for JSON parsing
if ! command -v jq &> /dev/null; then
    echo "⚠️  Warning: jq not found. Platform verification will be limited."
    echo "   Install jq for full manifest inspection: brew install jq"
fi
echo ""

# Login to ECR (target account)
echo "🔐 Logging into target ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Note: NVIDIA Container Registry (nvcr.io) allows anonymous access for public images
# No authentication required for nvidia/k8s-device-plugin and other public images
# AWS-managed images (aws-efa-k8s-device-plugin, hyperpod-health-monitoring-agent) are skipped

# Process each line in the config file
while IFS='=' read -r repo_name source_image; do
    # Skip comments and empty lines
    [[ "$repo_name" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$repo_name" ]] && continue
    
    # Skip images that are available in all regions (no need to copy)
    if [[ "$repo_name" == "aws-efa-k8s-device-plugin" ]]; then
        echo "⏭️  Skipping $repo_name (available in all AWS regions)"
        continue
    fi
    
    if [[ "$repo_name" == "hyperpod-health-monitoring-agent" ]]; then
        echo "⏭️  Skipping $repo_name (available in all AWS regions with regional account IDs)"
        continue
    fi
    
    # Remove leading/trailing whitespace
    repo_name=$(echo "$repo_name" | xargs)
    source_image=$(echo "$source_image" | xargs)
    
    target_repo="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$repo_name"
    
    echo ""
    echo "📦 Processing $repo_name..."
    echo "  Source: $source_image"
    echo "  Target: $target_repo"
    
    # Check if ECR repository exists, create if not
    if ! aws ecr describe-repositories --repository-names "$repo_name" --region "$REGION" >/dev/null 2>&1; then
        echo "  📝 Creating ECR repository: $repo_name"
        aws ecr create-repository --repository-name "$repo_name" --region "$REGION" >/dev/null
        echo "  ✅ Repository created"
    else
        echo "  ✅ Repository exists"
    fi
    
    # Pull source image with explicit platform enforcement
    echo "  ⬇️  Pulling source image for linux/amd64..."
    
    # First, remove any existing local image to avoid conflicts
    docker rmi "$source_image" 2>/dev/null || true
    
    # Set Docker environment to force platform selection
    export DOCKER_DEFAULT_PLATFORM=linux/amd64
    
    # Pull with explicit platform flag
    retry_command "docker pull --platform linux/amd64 \"$source_image\""
    
    # Verify the pulled image is actually linux/amd64
    echo "  🔍 Verifying image platform..."
    image_info=$(docker inspect "$source_image" --format '{{.Os}}/{{.Architecture}}' 2>/dev/null || echo "unknown")
    
    if [[ "$image_info" == "linux/amd64" ]]; then
        echo "  ✅ Confirmed platform: $image_info"
    else
        echo "  ⚠️  Warning: Image platform is $image_info (expected linux/amd64)"
        
        # Try to get the specific AMD64 digest and pull by digest
        if command -v jq &> /dev/null; then
            echo "  🔄 Attempting to pull AMD64 variant by digest..."
            amd64_digest=$(docker manifest inspect "$source_image" 2>/dev/null | jq -r '.manifests[]? | select(.platform.os=="linux" and .platform.architecture=="amd64") | .digest' 2>/dev/null || echo "")
            
            if [[ -n "$amd64_digest" && "$amd64_digest" != "null" ]]; then
                # Extract image name without tag
                image_base=$(echo "$source_image" | cut -d':' -f1)
                amd64_image="$image_base@$amd64_digest"
                
                echo "  📥 Pulling AMD64 digest: $amd64_digest"
                docker rmi "$source_image" 2>/dev/null || true
                retry_command "docker pull \"$amd64_image\""
                
                # Tag it back to the original name
                docker tag "$amd64_image" "$source_image"
                
                # Verify again
                image_info=$(docker inspect "$source_image" --format '{{.Os}}/{{.Architecture}}' 2>/dev/null || echo "unknown")
                if [[ "$image_info" == "linux/amd64" ]]; then
                    echo "  ✅ Successfully pulled AMD64 variant: $image_info"
                else
                    echo "  ❌ Still wrong platform after digest pull: $image_info"
                    echo "  Available platforms:"
                    docker manifest inspect "$source_image" 2>/dev/null | jq -r '.manifests[]?.platform | .os + "/" + .platform.architecture' 2>/dev/null || echo "  Unable to inspect manifest"
                    exit 1
                fi
            else
                echo "  ❌ No AMD64 digest found in manifest"
                exit 1
            fi
        else
            echo "  ❌ jq required for digest-based pulling. Install with: brew install jq"
            exit 1
        fi
    fi
    
    # Tag for ECR
    echo "  🏷️  Tagging image..."
    docker tag "$source_image" "$target_repo:latest"
    
    # Extract and tag with original version if available
    if [[ "$source_image" == *":"* ]]; then
        original_tag=$(echo "$source_image" | cut -d':' -f2)
        if [[ "$original_tag" != "latest" ]]; then
            docker tag "$source_image" "$target_repo:$original_tag"
            echo "  🏷️  Tagged with version: $original_tag"
        fi
    fi
    
    # Push to ECR with retry logic
    echo "  ⬆️  Pushing to ECR..."
    retry_command "docker push \"$target_repo:latest\""
    
    # Push version tag if it exists
    if [[ "$source_image" == *":"* ]]; then
        original_tag=$(echo "$source_image" | cut -d':' -f2)
        if [[ "$original_tag" != "latest" ]]; then
            echo "  ⬆️  Pushing version tag..."
            retry_command "docker push \"$target_repo:$original_tag\""
        fi
    fi
    
    echo "  ✅ Successfully copied $repo_name"
done < "$CONFIG_FILE"

echo ""
echo "🎉 All images successfully copied to ECR!"