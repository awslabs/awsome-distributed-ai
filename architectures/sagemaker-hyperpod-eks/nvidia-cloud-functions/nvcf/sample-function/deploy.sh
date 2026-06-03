#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy.sh -- Build, push, create, and deploy the echo function on NVCF
#
# This script:
#   1. Builds the container image for linux/amd64 (supports docker or finch)
#   2. Tags and pushes it to the NGC Private Registry
#   3. Creates a function in NVCF
#   4. Deploys the function to the registered SageMaker HyperPod cluster
#
# Required environment variables:
#   NGC_API_KEY    - NGC Personal API Key (with Cloud Functions + Private Registry scopes)
#   NGC_ORG_NAME   - NGC organization name (e.g., "qdrlnbkss123")
#
# Optional environment variables:
#   CLUSTER_GROUP  - NVCF cluster group name (default: nvcf-hyperpod)
#   GPU_TYPE       - GPU type for deployment (default: A10G)
#   INSTANCE_TYPE  - NVCF instance type (default: AWS.GPU.A10G_1x)
#                    Find yours via: curl https://api.ngc.nvidia.com/v2/nvcf/clusterGroups
#   IMAGE_TAG      - Container image tag (default: 1.0.0)
#
# Usage:
#   export NGC_API_KEY="your-api-key"
#   export NGC_ORG_NAME="your-org-name"
#   chmod +x deploy.sh
#   ./deploy.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# Auto-source nvcf-config.env if it exists (centralised user configuration)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${REPO_ROOT}/nvcf-config.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/nvcf-config.env"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Configuration
IMAGE_NAME="nvcf-echo"
IMAGE_TAG="${IMAGE_TAG:-1.0.0}"
CLUSTER_GROUP="${CLUSTER_GROUP:-nvcf-hyperpod}"
GPU_TYPE="${GPU_TYPE:-A10G}"
INSTANCE_TYPE="${INSTANCE_TYPE:-AWS.GPU.A10G_1x}"
FUNCTION_NAME="echo-function-hyperpod"
INFERENCE_URL="/echo"
HEALTH_URI="/health"
INFERENCE_PORT=8000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Container runtime: prefer docker, fall back to finch
CONTAINER_RT=""

# ---------------------------------------------------------------------------
# Detect container runtime (docker or finch)
# ---------------------------------------------------------------------------
detect_runtime() {
    # Try docker first
    if command -v docker &>/dev/null; then
        if docker info &>/dev/null; then
            CONTAINER_RT="docker"
            info "Using container runtime: docker"
            return
        else
            warn "docker found but daemon is not accessible — trying alternatives."
        fi
    fi

    # Try finch
    if command -v finch &>/dev/null; then
        CONTAINER_RT="finch"
        info "Using container runtime: finch"
        return
    fi

    # Also check common install paths for finch
    if [[ -x /usr/local/bin/finch ]]; then
        CONTAINER_RT="/usr/local/bin/finch"
        info "Using container runtime: /usr/local/bin/finch"
        return
    fi

    error "No container runtime found. Install docker or finch."
    exit 1
}

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
validate() {
    if [[ -z "${NGC_API_KEY:-}" ]]; then
        error "NGC_API_KEY is not set."
        echo "  Generate one at: https://org.ngc.nvidia.com/setup/personal-keys"
        echo "  Ensure it has Cloud Functions + Private Registry scopes."
        exit 1
    fi

    if [[ -z "${NGC_ORG_NAME:-}" ]]; then
        error "NGC_ORG_NAME is not set."
        echo "  Find yours at: https://org.ngc.nvidia.com/profile"
        exit 1
    fi

    detect_runtime
}

# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------
build_image() {
    info "Building container image: ${IMAGE_NAME}:${IMAGE_TAG}..."
    ${CONTAINER_RT} build \
        -f "${SCRIPT_DIR}/Dockerfile" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        --platform linux/amd64 \
        "${SCRIPT_DIR}"

    info "Build complete."
}

# ---------------------------------------------------------------------------
# 2. Tag and Push to NGC Private Registry
# ---------------------------------------------------------------------------
push_image() {
    local full_image="nvcr.io/${NGC_ORG_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

    info "Logging in to NGC Private Registry (nvcr.io)..."
    echo "${NGC_API_KEY}" | ${CONTAINER_RT} login nvcr.io -u '$oauthtoken' --password-stdin

    info "Tagging image as ${full_image}..."
    ${CONTAINER_RT} tag "${IMAGE_NAME}:${IMAGE_TAG}" "${full_image}"

    info "Pushing to NGC Private Registry..."
    ${CONTAINER_RT} push "${full_image}"

    info "Push complete. Image available at ${full_image}"
}

# ---------------------------------------------------------------------------
# 3. Create Function in NVCF
# ---------------------------------------------------------------------------
create_function() {
    local full_image="nvcr.io/${NGC_ORG_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

    info "Creating function '${FUNCTION_NAME}' in NVCF..."

    local response
    response=$(curl --silent --location 'https://api.ngc.nvidia.com/v2/nvcf/functions' \
        --header 'Content-Type: application/json' \
        --header 'Accept: application/json' \
        --header "Authorization: Bearer ${NGC_API_KEY}" \
        --data "{
            \"name\": \"${FUNCTION_NAME}\",
            \"inferenceUrl\": \"${INFERENCE_URL}\",
            \"healthUri\": \"${HEALTH_URI}\",
            \"inferencePort\": ${INFERENCE_PORT},
            \"containerImage\": \"${full_image}\"
        }")

    FUNCTION_ID=$(echo "${response}" | python3 -c "import sys, json; print(json.load(sys.stdin)['function']['id'])" 2>/dev/null || echo "")
    FUNCTION_VERSION_ID=$(echo "${response}" | python3 -c "import sys, json; print(json.load(sys.stdin)['function']['versionId'])" 2>/dev/null || echo "")

    if [[ -n "${FUNCTION_ID}" && -n "${FUNCTION_VERSION_ID}" ]]; then
        info "Function created successfully."
        info "  Function ID:         ${FUNCTION_ID}"
        info "  Function Version ID: ${FUNCTION_VERSION_ID}"
    else
        error "Failed to create function. Response:"
        echo "${response}" | python3 -m json.tool 2>/dev/null || echo "${response}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 4. Deploy Function to SageMaker HyperPod Cluster
# ---------------------------------------------------------------------------
deploy_function() {
    info "Deploying function to cluster group '${CLUSTER_GROUP}'..."

    local response
    response=$(curl --silent --location \
        "https://api.ngc.nvidia.com/v2/nvcf/deployments/functions/${FUNCTION_ID}/versions/${FUNCTION_VERSION_ID}" \
        --header 'Content-Type: application/json' \
        --header 'Accept: application/json' \
        --header "Authorization: Bearer ${NGC_API_KEY}" \
        --data "{
            \"deploymentSpecifications\": [
                {
                    \"backend\": \"${CLUSTER_GROUP}\",
                    \"gpu\": \"${GPU_TYPE}\",
                    \"instanceType\": \"${INSTANCE_TYPE}\",
                    \"maxInstances\": 1,
                    \"minInstances\": 1,
                    \"maxRequestConcurrency\": 2
                }
            ]
        }")

    info "Deployment response:"
    echo "${response}" | python3 -m json.tool 2>/dev/null || echo "${response}"
}

# ---------------------------------------------------------------------------
# 5. Test Invocation
# ---------------------------------------------------------------------------
test_invocation() {
    echo ""
    info "To invoke the function once deployed:"
    echo ""
    echo "  curl --location \"https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/\${FUNCTION_ID}\" \\"
    echo "    --header 'Content-Type: application/json' \\"
    echo "    --header \"Authorization: Bearer \${NGC_API_KEY}\" \\"
    echo "    --data '{\"message\": \"hello from SageMaker HyperPod!\"}'"
    echo ""
    info "Replace \${FUNCTION_ID} with: ${FUNCTION_ID}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "========================================="
    echo "  NVCF Echo Function Deployment"
    echo "========================================="
    echo ""

    validate

    case "${1:-all}" in
        build)
            build_image
            ;;
        push)
            push_image
            ;;
        create)
            create_function
            ;;
        deploy)
            if [[ -z "${FUNCTION_ID:-}" || -z "${FUNCTION_VERSION_ID:-}" ]]; then
                error "FUNCTION_ID and FUNCTION_VERSION_ID must be set for deploy-only."
                exit 1
            fi
            deploy_function
            ;;
        all)
            build_image
            push_image
            create_function
            deploy_function
            test_invocation
            ;;
        *)
            echo "Usage: $0 {build|push|create|deploy|all}"
            exit 1
            ;;
    esac
}

main "$@"
