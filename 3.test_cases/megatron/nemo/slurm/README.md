# Running NVIDIA NeMo 2.0 with Nemo-Run on Slurm

This project provides a step-by-step guide to deploying NVIDIA NeMo 2.0 on a Slurm cluster. It covers preparing the environment, and running NeMo jobs for large-scale AI training.

## 1. Prerequisites

This guide assumes that you have the following:

- A functional Slurm cluster on AWS. This test case also assumes that the cluster node uses Ubuntu-based OS.
- Docker, for Slurm [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) need to be installed as well.
- An FSx for Lustre filesystem mounted on `/fsx` in all Slurm nodes. Also, this test case assumes that the home directory is also a shared directory.

It is recommended that you use the templates in the architectures [directory](../../1.architectures) for setting up Amazon SageMaker HyperPod or AWS Parallel Cluster.

Make sure that your current directory is under a shared filesystem such as `/fsx`. 


## 3. Clone this repo

  ```bash
  cd ~
  git clone https://github.com/awslabs/awsome-distributed-training/
  cd awsome-distributed-training/3.test_cases/22.nemo-run/slurm
  ```

## 4. Build and Configure the NeMo Job Container

Before running NeMo jobs, build a custom optimized container image for EFA and Convert the Docker container image to an [Enroot](https://github.com/NVIDIA/enroot) squash file that will be stored under your home dirctory.

Build Image:

  ```bash
  docker build --progress=plain -t aws-nemo:26.02 -f ../Dockerfile ..
  enroot import -o ~/aws-nemo-26-02.sqsh dockerd://aws-nemo:26.02
  ```

## 5. Install Dependencies and Prepare NeMo 2.0 Environment

### Install Python 3.10 and NeMo-Run

Update and ensure the correct python version is installed

  ```bash
  sudo apt update
  sudo apt install software-properties-common -y
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install -y python3.10 python3.10-venv python3.10-dev
  python3.10 --version
  ```

Create Virtual environment and install NeMo Run and other dependencies

  ```bash
  python3.10 -m venv nemo-env
  source nemo-env/bin/activate
  bash venv.sh
  ```

### Download Tokenizer Files

We will be running the pre-training of LLaMa8B model in an offline mode, so we will download the vocab and merges files for the tokenizer

  ```bash
  mkdir -p ~/megatron
  wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O ~/megatron/megatron-gpt-345m_vocab
  wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O ~/megatron/megatron-gpt-345m_merges
  ```

## 6. Launch Pretraining Job with NeMo-Run

In NeMo-Run, you can build and configure everything using Python, eliminating the need for multiple combinations of tools to manage your experiments. The only exception is when setting up the environment for remote execution, where we rely on Docker. You can find how to set write your Python script for Nemo-Run in [NVIDIA documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html).
In this example, we run the following script to start the LLaMa 8B pretraining job:

  ```bash
  python run.py --container_image ~/aws-nemo-26-02.sqsh --nodes 2 --partition dev --env_vars_file env_vars.json --max_steps 1000
  ```

## 7. References

- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/index.html)
- [NVIDIA NeMo Github](https://github.com/NVIDIA/NeMo)
- [NVIDIA NeMo resiliency example](https://github.com/NVIDIA/NeMo/tree/main/examples/llm/resiliency)
- [AWS SageMaker HyperPod Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)
- [AWS SageMaker HyperPoc Workshop](https://catalog.workshops.aws/sagemaker-hyperpod/en-US)

