# NanoVLM Test Case

This test case demonstrates distributed training of [NanoVLM](https://github.com/huggingface/nanoVLM/), a repository for training/finetuning a small sized Vision-Language Model with a lightweight implementation in pure PyTorch. 


## 1. Prerequisites

This guide assumes that you have the following:

- A functional Slurm cluster on AWS. This test case also assumes that the cluster node uses Ubuntu-based OS.
- Docker, for Slurm [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) need to be installed as well.
- An FSx for Lustre filesystem mounted on `/fsx` in all Slurm nodes. Also, this test case assumes that the home directory is also a shared directory.

Make sure that your current directory is under a shared filesystem such as `/fsx`. 

## 2. Clone this repo

  ```bash
  cd ~
  git clone https://github.com/aws-samples/awsome-distributed-ai/
  cd awsome-distributed-ai/examples/use-cases/nanovlm/
  ```


## 3. Install Dependencies and Prepare Virtual Environment

Create Virtual environment and install the dependencies to download our dataset and test the generation in subsequent sections.

  ```bash
  sudo apt install python3.10-venv
  python3 -m venv nanoVLM-env
  source nanoVLM-env/bin/activate
  pip install torch numpy torchvision pillow datasets huggingface-hub transformers wandb einops accelerate loguru lmms_eval

  ```

## 4. Hugging Face token

First, create a Hugging Face account to retrieve a [token](https://huggingface.co/settings/tokens.). Log in to your account and create an access token from Hugging Face Tokens. 


### Get huggingface token

```bash
huggingface-cli login
```

You will be prompted to input the token. Paste the token and answer `n` when asked to add the token as a git credential.

```

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) n
Token is valid (permission: read).
Your token has been saved to /fsx/ubuntu/.cache/huggingface/token
Login successful
```

Then export the saved token `${HF_TOKEN}` to use in the subsequent steps

```bash
export HF_TOKEN=$(cat /path_where_the_token_is_saved_from_the_above_step)
```
for example:
```bash
export HF_TOKEN=$(cat /fsx/ubuntu/.cache/huggingface/token)
```

## 5. Clone the nanoVLM repository

```bash
git clone https://github.com/huggingface/nanoVLM.git
cd nanoVLM
git checkout 9de5e17ac2f4c578c32085131d966464cdd252b5
cd ..
```
This sample has been developed with the above commit hash. 

## 6. Download the dataset required for the training

Specify path to download dataset for example:

```bash
export DATASET_DIR=$PWD/datasets/cauldron
```

The default dataset path will be $DATASET_DIR and the datasets are ["clevr", "vqav2", "docvqa"]. 

### (Optional) You can modify this as needed to dowload the entire dataset by setting the configs to the entry below in Line 24 in slurm/download_dataset.sbatch file:

```bash
configs = get_dataset_config_names("HuggingFaceM4/the_cauldron")
```

```bash
cd slurm
sbatch download_dataset.sbatch
```

## 7. Update the dataset and checkpoint path in the NanoVLM config 

```bash
cd ..
sed -i "s|train_dataset_path: str = '[^']*'|train_dataset_path: str = '$DATASET_DIR'|" $PWD/nanoVLM/models/config.py
```

Since this demo is just to showcase the workflow, we can also reduce the number of evaluation tasks from [mmstar,mmmu,ocrbench,textvqa,docvqa,scienceqa,mme,infovqa] to just using [mmstar,mmmu] with the command below:

```bash
sed -i "s/lmms_eval_tasks: str = 'mmstar,mmmu,ocrbench,textvqa,docvqa,scienceqa,mme,infovqa'/lmms_eval_tasks: str = 'mmstar,mmmu'/" $PWD/nanoVLM/models/config.py
```

```bash
export CHECKPOINT_DIR=$PWD/nanoVLM/checkpoints
```

```bash
sed -i "s|vlm_checkpoint_path: str = '[^']*'|vlm_checkpoint_path: str = '$CHECKPOINT_DIR'|" $PWD/nanoVLM/models/config.py
```

Disable logging metrics to wandb for this sample:
```bash
sed -i "s/log_wandb: bool = True/log_wandb: bool = False/" $PWD/nanoVLM/models/config.py
```

### (Optional) If training and running evaluations on g5 instances, update the configuration as below to avoid OOM issues.
```bash

sed -i \
  -e 's/lm_max_position_embeddings: int = 8192/lm_max_position_embeddings: int = 2048/' \
  -e 's/lm_max_length: int = 8192/lm_max_length: int = 2048/' \
  -e 's/max_img_size: int = 2048/max_img_size: int = 1024/' \
  -e 's/data_cutoff_idx: int = None/data_cutoff_idx: int = 5000/' \
  -e 's/gradient_accumulation_steps: int = 8/gradient_accumulation_steps: int = 4/' \
  -e 's/eval_interval: int = 500/eval_interval: int = 50/' \
  -e 's/stats_log_interval: int = 100/stats_log_interval: int = 10/' \
  -e 's/max_training_steps: int = 80100/max_training_steps: int = 500/' \
  -e 's/max_images_per_example: int = 8/max_images_per_example: int = 2/' \
  -e 's/max_images_per_knapsack: int = 36/max_images_per_knapsack: int = 8/' \
  -e 's/max_sample_length: int = 8192/max_sample_length: int = 2048/' \
  -e 's/train_dataset_name: tuple\[str, ...\] = ("all", )/train_dataset_name: tuple[str, ...] = ("default",)/' \
  -e 's/log_wandb: bool = True/log_wandb: bool = False/' \
  -e "s/lmms_eval_tasks: str = 'mmstar,mmmu,ocrbench,textvqa,docvqa,scienceqa,mme,infovqa'/lmms_eval_tasks: str = 'mmstar,mmmu'/" \
  $PWD/nanoVLM/models/config.py
```

## 8. Build and Configure the NaNoVLM Job Container
The provided Dockerfile (`nanoVLM.Dockerfile`) will set up the environment with all required dependencies:

```bash
docker build -t nanovlm:latest -f nanovlm.Dockerfile .
enroot import -o nanovlm.sqsh  dockerd://nanovlm:latest
```
## 9. Launch Training

```bash
cd slurm
sbatch launch_training.sbatch
```
Note the path where the checkpoints will be generated from the slurm.out log file as this will be used in the subsequent sections for evaluation and generation

For example:

```
/fsx/ubuntu/nanoVLM/checkpoints/nanoVLM_siglip2-base-patch16-512_1024_mp4_SmolLM2-360M-Instruct_2xGPU_5000samples_bs8_500_lr_vision_5e-05-language_5e-05-0.00512_0923-230408/step_450
```

## 10. Run evaluation

```bash
sbatch launch_evaluation.sbatch
```

## 11. Test generation

```bash
cd ../nanoVLM
python generate.py --checkpoint $CHECKPOINT_DIR

```
