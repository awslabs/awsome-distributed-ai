## Running SmolLM-1.7B Training with 3D Parallelism on a Single EC2 Instance

This guide demonstrates how to train the SmolLM-1.7B model using 3D parallelism on a single EC2 instance. While the model is relatively small compared to larger language models, it serves as an excellent example to understand and experiment with different types of model parallelism:

- Data Parallelism (DP): Distributes training batches across GPUs
- Tensor Parallelism (TP): Splits model layers across GPUs
- Pipeline Parallelism (PP): Divides model vertically into pipeline stages

In this example, we configure:
- DP=2: Data parallel across 2 groups
- TP=2: Each layer split across 2 GPUs
- PP=2: Model divided into 2 pipeline stages
This configuration requires 8 GPUs total (2 x 2 x 2 = 8), which can be found in instances like p5.48xlarge.

### Prerequisites

Before running this example, you need to:
1. Build the Picotron container image following the guidance in [here](..)
2. Have an EC2 instance with 8 GPUs (e.g., p5.48xlarge)

### How to Run the Distributed Training Job

First, export your Hugging Face token as an environment variable:

   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

Then create configuration file using the aforementioned container:


   ```bash
   docker run --rm -v ${PWD}:${PWD} picotron python3 /picotron/create_config.py \
       --out_dir ${PWD}/conf --exp_name llama-1B-dp2-tp2-pp2 --dp 2 --tp 2 --pp 2  \
       --pp_engine 1f1b --model_name HuggingFaceTB/SmolLM-1.7B --num_hidden_layers 5 \
       --total_train_steps 5000 \
       --grad_acc_steps 2 --mbs 4 --seq_len 128 --hf_token ${HF_TOKEN} 
    ```

Finally, submit the training job as follows:

    ```
    declare -a DOCKER_ARGS=(
        --rm                  # Remove container after it exits
        --gpus all           # Make all GPUs available to the container
        -v ${PWD}:${PWD}     # Mount current directory into container

    )
    declare -a TORCHRUN_ARGS=(
        --nproc_per_node=8 # Total number of processes = DP x TP x PP = 2 x 2 x 2 = 8
        --nnodes=1 # Running on a single EC2 instance
        --rdzv_id=$((RANDOM + 62000)) # Arbitrary ID since we only have 1 node
        --rdzv_backend=c10d # PyTorch's distributed backend
        --rdzv_endpoint=localhost # Use localhost 
    )
   docker run "${DOCKER_ARGS[@]}"  picotron torchrun "${TORCHRUN_ARGS[@]}" \
    /picotron/train.py --config ${PWD}/conf/llama-1B-dp2-tp2-pp2/config.json
   ```

After running the training job, you'll see output logs showing the training progress. The logs include:

- Model initialization details like tokenizer creation and parameter count (536.90M parameters)
- Training metrics per step including:
  - Loss value
  - Global batch size
  - Training throughput (tokens/sec total and per GPU)
  - Total tokens processed
  - Model FLOPs Utilization (MFU)
  - GPU memory usage

The example log shows the first 3 steps of training, where the loss decreases from 10.90 to 8.92 as training progresses, and the throughput increases as the training pipeline warms up from 199 tokens/sec to over 10K tokens/sec.

    ```text
      warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
    Downloading readme: 100%|██████████| 1.06k/1.06k [00:00<00:00, 8.88MB/s]
    Downloading data: 100%|██████████| 249M/249M [00:15<00:00, 16.5MB/s] 
    Downloading data: 100%|██████████| 248M/248M [00:00<00:00, 255MB/s]  
    Downloading data: 100%|██████████| 246M/246M [00:01<00:00, 183MB/s]  
    Downloading data: 100%|██████████| 248M/248M [00:00<00:00, 257MB/s]  
    Downloading data: 100%|██████████| 9.99M/9.99M [00:00<00:00, 39.6MB/s]
    Generating train split: 100%|██████████| 2119719/2119719 [00:05<00:00, 356503.03 examples/s]
    Generating validation split: 100%|██████████| 21990/21990 [00:00<00:00, 367978.65 examples/s]
    rank 0: Creating tokenizer
    rank 0: Broadcasting tokenizer to all ranks
    Grouping texts in chunks of 129:  55%|█████▍    | 218000/400000 
    ...
    SafeTensors files downloaded successfully! ✅
    rank 0: Creating model config
    init dataloader time: 265.49s
    rank 0: Broadcasting model_config to all ranks
    rank 2: Initializing model meta device
    init model parallel time: 1.57s
    Number of parameters: 536.90M
    ...
    [rank 2] Step: 1     | Loss: 10.9062 | Global batch size:   2.05K | Tokens/s:  199.02 | Tokens/s/GPU:   24.88 | Tokens:   2.05K | MFU:  0.01% | Memory usage:   1.75GB
    [rank 2] Step: 2     | Loss: 8.5469 | Global batch size:   2.05K | Tokens/s:   1.68K | Tokens/s/GPU:  209.63 | Tokens:   4.10K | MFU:  0.07% | Memory usage:   1.75GB
    [rank 2] Step: 3     | Loss: 8.9219 | Global batch size:   2.05K | Tokens/s:  10.61K | Tokens/s/GPU:   1.33K | Tokens:   6.14K | MFU:  0.43% | Memory usage:   1.75GB
    ```
        `