## Running SmolLM-1.7B Training on Two Single-GPU EC2 Instances with Slurm

This guide demonstrates how to run distributed training across two GPU instances using Slurm. For simplicity and cost-effectiveness, we use instances with a single GPU each (e.g., g5.8xlarge) to showcase Tensor Parallelism (TP=2) training.

### Prerequisites

1. **Cluster Setup**
   - A Slurm cluster on AWS with at least two GPU compute nodes
   - We recommend using either AWS ParallelCluster or SageMaker HyperPod with our provided templates in the [architectures directory](../../../architectures)

2. **Node Requirements**
   - Docker
   - [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) for container orchestration
   - An FSx for Lustre filesystem mounted at `/fsx`
   - Shared home directory accessible across head and compute nodes

3. **Container Setup**
   - Build the `picotron` container image following the guidance in [here](..)
   - Convert the Docker image to a squash file:
     ```bash
     enroot import -o picotron.sqsh dockerd://picotron:latest
     ```

4. **Environment Variables**
   ```bash
   # Your Hugging Face token is needed to retrieve model and data from HF Hub
   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

### Running the Distributed Training Job

1. **Create Configuration File**
   ```bash
   # TensorParallelism on 2 GPUs
   enroot create --name picotron picotron.sqsh
   enroot start --mount ${PWD}:${PWD} \
      --env NVIDIA_VISIBLE_DEVICES=void picotron \
       python3 /picotron/create_config.py \
       --out_dir ${PWD}/conf --exp_name llama-1B-tp2 --dp 1 --tp 2 --pp 1  \
       --pp_engine 1f1b --model_name HuggingFaceTB/SmolLM-1.7B --num_hidden_layers 5 \
       --total_train_steps 5000 \
       --grad_acc_steps 2 --mbs 4 --seq_len 128 --hf_token ${HF_TOKEN} 
   enroot remove -f picotron
   ```

   This will create a config file `./conf/llama-1B-tp2/config.json` describing training configurations, including model architecture, training configuration, and dataset to use.

2. **Submit the Training Job**
   ```bash
   sbatch train.sbatch
   ```

3. **Monitor Training Progress**
   - Check the log directory `log` in the current directory
   - Look for files of the form `picotron_[job-number].out`
   - These logs will be continuously updated with your training progress

