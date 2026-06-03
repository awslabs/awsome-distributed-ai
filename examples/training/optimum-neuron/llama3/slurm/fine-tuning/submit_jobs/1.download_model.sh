#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=download 
#SBATCH -o /fsx/ubuntu/peft_ft/logs/1_download_model.out

export OMP_NUM_THREADS=1
export HF_TOKEN="<Your Hugging Face Token>"

INPUT_PATH="/fsx/ubuntu/awsome-distributed-training/examples/pytorch/optimum-neuron/llama3/src/get_model.py"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_OUTPUT_PATH="/fsx/ubuntu/peft_ft/model_artifacts/llama3-8B"
TOKENIZER_OUTPUT_PATH="/fsx/ubuntu/peft_ft/tokenizer/llama3-8B"

srun python3 $INPUT_PATH \
    --model_id $MODEL_ID \
    --model_output_path $MODEL_OUTPUT_PATH \
    --tokenizer_output_path $TOKENIZER_OUTPUT_PATH