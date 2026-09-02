#!/bin/bash

# Deploy NeMo Data Processing Pod
# This script helps deploy and manage the data processing pod

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POD_NAME="nemo-data-processing"
POD_TEMPLATE="$SCRIPT_DIR/data-processing-pod-template.yaml"
LOAD_DATASET_SCRIPT="$SCRIPT_DIR/load_dataset.py"
CONFIGMAP_NAME="nemo-dataset-scripts"

# Default values
DEFAULT_PVC_NAME="fsx-claim"
DEFAULT_MOUNT_PATH="/mnt/nemo"
DEFAULT_DATASET_NAME="wikitext"
DEFAULT_DATASET_CONFIG="wikitext-103-v1"

show_help() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy      Deploy the data processing pod with scripts mounted
    delete      Delete the data processing pod and ConfigMap
    status      Check pod status
    logs        Show pod logs
    exec        Execute interactive shell in the pod
    help        Show this help message

Options:
    --pvc-name NAME         Name of the PVC to mount (default: $DEFAULT_PVC_NAME)
    --mount-path PATH       Mount path in the container (default: $DEFAULT_MOUNT_PATH)
    --dataset-name NAME     Hugging Face dataset name (default: $DEFAULT_DATASET_NAME)
    --dataset-config CONFIG Dataset configuration (default: $DEFAULT_DATASET_CONFIG)
                           Use 'none' for datasets without configs

Examples:
    $0 deploy
    $0 deploy --pvc-name my-fsx-pvc --mount-path /data
    $0 deploy --dataset-name wikitext --dataset-config wikitext-103-v1
    $0 deploy --dataset-name openwebtext --dataset-config none
    $0 deploy --dataset-name bookcorpus --dataset-config none
    $0 exec
    $0 logs
    $0 delete

Scripts will be available in the pod at:
    /scripts/load_dataset.py

To run the dataset loading script after exec:
    cd $DEFAULT_MOUNT_PATH
    python /scripts/load_dataset.py

Note: Some datasets require a config (like 'wikitext'), others don't (like 'openwebtext').
If unsure, try with the config first, the script will fallback to no config if needed.
EOF
}

parse_args() {
    PVC_NAME="$DEFAULT_PVC_NAME"
    MOUNT_PATH="$DEFAULT_MOUNT_PATH"
    DATASET_NAME="$DEFAULT_DATASET_NAME"
    DATASET_CONFIG="$DEFAULT_DATASET_CONFIG"
    COMMAND=""
    
    # Track if user explicitly set dataset name or config
    USER_SET_DATASET=false
    USER_SET_CONFIG=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --pvc-name)
                PVC_NAME="$2"
                shift 2
                ;;
            --mount-path)
                MOUNT_PATH="$2"
                shift 2
                ;;
            --dataset-name)
                DATASET_NAME="$2"
                USER_SET_DATASET=true
                shift 2
                ;;
            --dataset-config)
                DATASET_CONFIG="$2"
                USER_SET_CONFIG=true
                shift 2
                ;;
            deploy|delete|status|logs|exec|help|--help|-h)
                COMMAND="$1"
                shift
                ;;
            *)
                if [[ -z "$COMMAND" ]]; then
                    COMMAND="$1"
                else
                    echo "Unknown option: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # If user provided dataset but no config, default to "none"
    if [[ "$USER_SET_DATASET" == "true" && "$USER_SET_CONFIG" == "false" ]]; then
        DATASET_CONFIG="none"
    fi
    
    if [[ -z "$COMMAND" ]]; then
        COMMAND="help"
    fi
}

create_pod_manifest() {
    local temp_manifest=$(mktemp)
    
    # Create pod manifest with substituted values
    sed -e "s|__PVC_NAME__|$PVC_NAME|g" \
        -e "s|__MOUNT_PATH__|$MOUNT_PATH|g" \
        -e "s|__DATASET_NAME__|$DATASET_NAME|g" \
        -e "s|__DATASET_CONFIG__|$DATASET_CONFIG|g" \
        "$POD_TEMPLATE" > "$temp_manifest"
    
    echo "$temp_manifest"
}

create_configmap() {
    echo "Creating ConfigMap with dataset scripts..."
    
    # Check if script exists
    if [[ ! -f "$LOAD_DATASET_SCRIPT" ]]; then
        echo "Error: load_dataset.py not found at $LOAD_DATASET_SCRIPT"
        exit 1
    fi
    
    # Delete existing ConfigMap if it exists
    kubectl delete configmap "$CONFIGMAP_NAME" --ignore-not-found=true
    
    # Create ConfigMap from the script file
    kubectl create configmap "$CONFIGMAP_NAME" \
        --from-file=load_dataset.py="$LOAD_DATASET_SCRIPT"
    
    echo "ConfigMap '$CONFIGMAP_NAME' created successfully"
}

deploy_pod() {
    echo "Deploying NeMo data processing pod..."
    echo "PVC Name: $PVC_NAME"
    echo "Mount Path: $MOUNT_PATH"
    echo "Dataset Name: $DATASET_NAME"
    echo "Dataset Config: $DATASET_CONFIG"
    
    # Check if template exists
    if [[ ! -f "$POD_TEMPLATE" ]]; then
        echo "Error: Pod template not found at $POD_TEMPLATE"
        exit 1
    fi
    
    # Check if PVC exists
    if ! kubectl get pvc "$PVC_NAME" &>/dev/null; then
        echo "Warning: PVC '$PVC_NAME' not found. Please ensure the PVC is created."
        echo "You can check available PVCs with: kubectl get pvc"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Create ConfigMap with scripts
    create_configmap
    
    # Create temporary manifest with substituted values
    local temp_manifest=$(create_pod_manifest)
    
    kubectl apply -f "$temp_manifest"
    echo "Pod deployed. Waiting for it to be ready..."
    kubectl wait --for=condition=Ready pod/$POD_NAME --timeout=600s
    echo "Pod is ready!"
    
    # Clean up temporary manifest
    rm -f "$temp_manifest"
    
    echo ""
    echo "Setup complete! The load_dataset.py script is available at /scripts/load_dataset.py"
    echo ""
    echo "To use the script:"
    echo "  1. Access the pod: $0 exec"
    echo "  2. Navigate to your data directory: cd $MOUNT_PATH"
    echo "  3. Run the script: python /scripts/load_dataset.py"
    echo ""
    echo "Dataset configuration:"
    echo "  - Dataset: $DATASET_NAME"
    echo "  - Config: $DATASET_CONFIG"
    echo ""
    echo "Other commands:"
    echo "  - Check status: $0 status"
    echo "  - View logs: $0 logs"
    echo "  - Delete pod: $0 delete"
}

delete_pod() {
    echo "Deleting NeMo data processing pod..."
    kubectl delete pod $POD_NAME --ignore-not-found=true
    echo "Pod deleted."
    
    echo "Deleting ConfigMap..."
    kubectl delete configmap "$CONFIGMAP_NAME" --ignore-not-found=true
    echo "ConfigMap deleted."
}

check_status() {
    echo "Pod status:"
    kubectl get pod $POD_NAME 2>/dev/null || echo "Pod not found"
    echo
    echo "ConfigMap status:"
    kubectl get configmap "$CONFIGMAP_NAME" 2>/dev/null || echo "ConfigMap not found"
}

show_logs() {
    echo "Pod logs:"
    kubectl logs $POD_NAME 2>/dev/null || echo "Pod not found or no logs available"
}

exec_shell() {
    echo "Connecting to pod shell..."
    echo ""
    kubectl exec -it $POD_NAME -- /bin/bash
}

# Parse command line arguments
parse_args "$@"

case "$COMMAND" in
    deploy)
        deploy_pod
        ;;
    delete)
        delete_pod
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    exec)
        exec_shell
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac 