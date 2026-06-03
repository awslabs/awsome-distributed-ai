#!/bin/bash

#SBATCH --job-name=esm2-accelerate
#SBATCH --output=accelerate-%x.%j.out
#SBATCH --nodes=2              # number of nodes
#SBATCH --exclusive # job has exclusive use of the resource, no sharing


######################
### Set enviroment ###
######################
source /fsx/ubuntu/miniconda3/bin/activate
conda activate esm2

export GPUS_PER_NODE=4
######################

## Plenty of EFA level variables
## export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa
export NCCL_DEBUG=INFO

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank $SLURM_PROCID \
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap EsmLayer
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_cpu_ram_efficient_loading True \
    --fsdp_sync_module_states True \
    --fsdp_use_orig_params True \
    "

export TRAIN_SCRIPT="/fsx/ubuntu/train.py"
export TRAIN_SCRIPT_ARGS=" \
    --config_name "facebook/esm2_t30_150M_UR50D" \
    --dataloader_num_workers 8 \
    --bf16 True \
    --do_eval True \
    --do_preprocess False \
    --do_train True \
    --gradient_accumulation_steps 16 \
    --logging_steps 16 \
    --num_train_epochs 1 \
    --output_dir "/fsx//ubuntu/output" \
    --per_device_train_batch_size 8 \
    --max_train_samples 100000 \
    --tokenizer_name "facebook/esm2_t30_150M_UR50D" \
    --dataset_dir "/fsx/ubuntu/processed/arrow/" \
    --torch_compile False \
    --pad_to_max_length True \
    --max_seq_length 512
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $TRAIN_SCRIPT $TRAIN_SCRIPT_ARGS" 
srun $CMD