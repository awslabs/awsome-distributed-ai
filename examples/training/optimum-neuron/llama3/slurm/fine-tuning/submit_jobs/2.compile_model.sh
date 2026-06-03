#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=compile
#SBATCH --output=/fsx/ubuntu/peft_ft/logs/2_parallel_compile.out

export OMP_NUM_THREADS=1
export NEURON_EXTRACT_GRAPHS_ONLY=1

srun bash /fsx/ubuntu/awsome-distributed-training/examples/pytorch/optimum-neuron/llama3/slurm/fine-tuning/finetune-llama3-8B.sh
