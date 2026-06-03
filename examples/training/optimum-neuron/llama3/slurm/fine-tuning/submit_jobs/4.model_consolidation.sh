#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=consolidation 
#SBATCH --output=/fsx/ubuntu/peft_ft/logs/4_model_consolidation.log

export OMP_NUM_THREADS=1

srun python3 "/fsx/ubuntu/awsome-distributed-training/examples/pytorch/optimum-neuron/llama3/src/model_consolidation.py" \
    --input_dir "/fsx/ubuntu/peft_ft/model_checkpoints/checkpoint-1251" \
    --output_dir "/fsx/ubuntu/peft_ft/model_checkpoints/adapter_shards_consolidation"\
    --save_format "safetensors"