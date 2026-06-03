#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=inference
#SBATCH --output=/fsx/ubuntu/peft_ft/logs/6_inference.log

export OMP_NUM_THREADS=1
export HF_TOKEN="<Your Hugging Face Token>"

srun python3 "/fsx/ubuntu/awsome-distributed-ai/examples/training/optimum-neuron/llama3/src/run_inference.py" \
    --model_path "/fsx/ubuntu/peft_ft/model_checkpoints/final_model_output" \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct"