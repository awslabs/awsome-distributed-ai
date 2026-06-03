#!/bin/bash

export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export IMAGE=llama3_trn
export TAG=:latest
export IMAGE_URI=${REGISTRY}${IMAGE}${TAG}

export JOB_NAME=trn1-llama3-training
export NUM_NODES=1
export INSTANCE_TYPE=ml.trn1.32xlarge
export EFA_PER_NODE=8
export NEURON_PER_NODE=16
export FI_PROVIDER=efa


export FSX_CLAIM=fsx-claim # Change this according to the pvc created.

# Tokenize_data configs

export HF_ACCESS_TOKEN=hf_xxxxxx
export TOKENIZED_DATA_PATH=/fsx/tokenized_data
export DATASET_NAME=wikicorpus
export dATASET_CONFIG_NAME=raw_en
export HF_MODEL_NAME=meta-llama/Meta-Llama-3-8B # change this to meta-llama/Meta-Llama-3-8B if you want to train llama3 8B model


export NEURON_CACHE_DIR=/fsx/neuron_cache
export CHECKPOINT_DIR=/fsx/checkpoints
export NUM_KEPT_CHECKPOINTS=2
export CHECKPOINT_FREQ=100
export NUM_NODES=1
export MAX_STEPS=1000
export STEPS_THIS_RUN=100
export BATCH_SIZE=1

export MODEL_PATH=config_8b_llama3


cat tokenize_data.yaml-template | envsubst > tokenize_data.yaml

cat llama3_train.yaml-template | envsubst > llama3_train.yaml
