#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=create_env
#SBATCH -o /fsx/ubuntu/peft_ft/logs/0_create_env.out

set -ex

# Install Python 3.10 (required by optimum-neuron 0.4.5).
# On Ubuntu 24.04 (SDK 2.28 DLAMI), Python 3.10 is available via deadsnakes PPA.
# On Ubuntu 22.04 (SDK 2.27 DLAMI), Python 3.10 is the system Python.
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev git

srun bash -c "sudo apt-get update && sudo apt-get install -y software-properties-common && sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv python3.10-dev git"

python3.10 -m venv /fsx/ubuntu/peft_ft/env_llama3_8B_peft
source /fsx/ubuntu/peft_ft/env_llama3_8B_peft/bin/activate
pip install -U pip

python3 -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"

# Install Neuron SDK packages first to prevent CUDA torch from being pulled
python3 -m pip install --upgrade neuronx-cc==2.23.6484.0 torch-neuronx==2.8.0.2.12.22436 torchvision
python3 -m pip install --upgrade neuronx-distributed==0.17.26814

python3 -m pip install 'optimum-neuron[training]==0.4.5'
