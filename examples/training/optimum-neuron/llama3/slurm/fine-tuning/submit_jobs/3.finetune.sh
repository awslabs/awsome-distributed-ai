#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=finetuning 
#SBATCH --output=/fsx/ubuntu/peft_ft/logs/3_finetune.log

export OMP_NUM_THREADS=1
export NEURON_EXTRACT_GRAPHS_ONLY=0

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi
srun  ${AUTO_RESUME} bash /fsx/ubuntu/awsome-distributed-ai/examples/training/optimum-neuron/llama3/slurm/fine-tuning/finetune-llama3-8B.sh
