#!/bin/bash

# AWS and Registry Configuration
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export IMAGE=peft-optimum-neuron
export TAG=:latest
export IMAGE_URI=${REGISTRY}${IMAGE}${TAG}

# Job Configuration
export NAMESPACE=kubeflow
export INSTANCE_TYPE=ml.trn1.32xlarge  # Options: ml.trn1.32xlarge, ml.trn1n.32xlarge, ml.trn2.48xlarge

# Storage Configuration
export FSX_CLAIM=fsx-claim

# Model and Dataset Configuration
export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
export MODEL_OUTPUT_PATH=/fsx/peft_ft/model_artifacts/llama3-8B
export DATASET_NAME=databricks/databricks-dolly-15k
export HF_TOKEN=""  # Set your Hugging Face token here

# Training Configuration
export NEURON_CACHE_DIR=/fsx/neuron_cache
export CHECKPOINT_DIR=/fsx/peft_ft/model_checkpoints
export CHECKPOINT_DIR_COMPILE=/fsx/peft_ft/model_checkpoints/compile
export MAX_SEQ_LENGTH=2048
export EPOCHS=1
export LEARNING_RATE=2e-05
export TRAIN_BATCH_SIZE=1
export MAX_TRAINING_STEPS=-1

# Derive parallelism settings from instance type.
# NeuronDevices are the physical devices (used for K8s resource requests).
# NeuronCores are the logical cores (used for nproc_per_node and NEURON_RT_NUM_CORES).
# TP_SIZE is the tensor parallelism degree.
case "$INSTANCE_TYPE" in
    ml.trn1.32xlarge|ml.trn1n.32xlarge)
        export EFA_PER_NODE=8
        export NEURON_PER_NODE=16     # 16 NeuronDevices
        export NEURON_CORES=32        # 32 NeuronCores
        export TP_SIZE=8
        ;;
    ml.trn2.48xlarge)
        export EFA_PER_NODE=16
        export NEURON_PER_NODE=16     # 16 NeuronDevices
        export NEURON_CORES=64        # 64 NeuronCores (LNC=2 default)
        export TP_SIZE=4
        ;;
    *)
        echo "ERROR: Unsupported instance type: $INSTANCE_TYPE"
        echo "Supported: ml.trn1.32xlarge, ml.trn1n.32xlarge, ml.trn2.48xlarge"
        exit 1
        ;;
esac

echo "Instance: $INSTANCE_TYPE -> NeuronDevices=$NEURON_PER_NODE, NeuronCores=$NEURON_CORES, TP=$TP_SIZE, EFA=$EFA_PER_NODE"

# Generate the final yaml files from templates
for template in download_model compile_peft launch_peft_train consolidation merge_lora; do
    cat templates/${template}.yaml-template | envsubst > ${template}.yaml
done

echo "Generated all YAML files successfully."
