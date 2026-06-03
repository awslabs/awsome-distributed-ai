# Train Evolutionary Scale Models (ESM) with BioNemo

NVIDIA BioNeMo is a domain-specific machine learning framework for training and using foundation models for biology. This includes models for analyzing proteins, small molecules, and other biological molecules. NVIDIA first announced it in [September 2022](https://nvidianews.nvidia.com/news/nvidia-launches-large-language-model-cloud-services-to-advance-ai-and-digital-biology) and released a more comprehensive version on DGX cloud at [GTC 2023](https://nvidianews.nvidia.com/news/nvidia-unveils-large-language-models-and-generative-ai-services-to-advance-life-sciences-r-d). The GTC 2023 release included two main capabilities:
1. A NeMo-based training framework to enable ML teams to create training and inference jobs via Python scripts. submitted via DGX-hosted notebooks
2. A web application that enabled scientists to create inference jobs and visualize output data.

|Num|                                    BioNeMo Model Support                                     |
|:-:|:--------------------------------------------------------------------------------------------:|
| 1 |      [ESM-1nv](https://docs.nvidia.com/bionemo-framework/latest/models/esm1-nv.html)         |
| 2 |      [ESM-2nv](https://docs.nvidia.com/bionemo-framework/latest/models/esm2-nv.html)         |
| 3 |      [MegaMolBART](https://docs.nvidia.com/bionemo-framework/latest/models/megamolbart.html) |
| 4 |      [DiffDock](https://docs.nvidia.com/bionemo-framework/latest/models/diffdock.html)       |
| 5 |      [EquiDock](https://docs.nvidia.com/bionemo-framework/latest/models/equidock.html)       |
| 6 |      [ProtT5nv](https://docs.nvidia.com/bionemo-framework/latest/models/prott5nv.html)       |


This project provides a guide to run [Nvidia's BioNemo](https://docs.nvidia.com/bionemo-framework/latest/index.html) on AWS ParallelCluster and pretrain the popular [ESM models](https://github.com/facebookresearch/esm) specifically the [ESM1nv](https://docs.nvidia.com/bionemo-framework/latest/notebooks/model_training_esm1nv.html) model.


## 0. Prerequisites

0. You have access to the bionemo container. To get the access to BioNeMo, visit the [information website](https://www.nvidia.com/en-us/clara/bionemo/).

1. Have a slurm based AWS ParallelCluster created with a FSx for Lustre filesystem mounted. Below we are presenting instructions for a cluster with compute nodes instantiated with an Ubuntu based AMI.

## 1. Install Nvidia Container CLI

### 1.1 If you have created your cluster with the AWS ParallelCluster Base AMI or [DLAMI](https://aws.amazon.com/machine-learning/amis/) or your custom AMI, please make sure `libnvidia-container cli` is installed. You can follow the instructions below to install it.   

### 1.2 To install libnvidia-container cli:
We need [libnvidia-container cli](https://github.com/NVIDIA/libnvidia-container) to train models in an Nvidia container. We follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). This installation needs to be done in each compute node.

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update \
  && sudo apt-get install libnvidia-container1 \
  && sudo apt-get install libnvidia-container-tools
```
### 1.3 You can set the Nemo Multimodal version and others as environment variables:

SSH into the head node of your cluster and run:

```
export PYTHON_VERSION=3.10
# We are using Python version 3.10 in this work. For a different Python version select the right Miniconda file from https://repo.anaconda.com/miniconda/
export MINICONDA_INSTALLER=Miniconda3-py310_23.5.2-0-Linux-x86_64
export TARGET_PATH=/apps/bionemo-src   # Must be a shared filesystem. This is where Nemo launcher scripts will reside.
export DOCKER_IMAGE_NAME=bionemo
export TAG=latest
export ENROOT_IMAGE=/apps/${DOCKER_IMAGE_NAME}
export DATASET_PATH=/fsx/
```

## 1.4. Pull this github repo

```bash
cd /apps/
git clone https://github.com/awslabs/awsome-distributed-training.git
cp -r /apps/awsome-distributed-training/examples/14.bionemo/* ./apps/
```

## 2. Pull Image

```bash
cd /apps/
docker pull nvcr.io/nvidia/clara/bionemo-framework:1.2
```

## 3. Create Conda env
We need a conda environment that has the necessary dependencies for submitting multiple arrays of slurm jobs via [HYDRA](https://github.com/facebookresearch/hydra) which NeMo uses to configuring both NeMo models and the PyTorch Lightning Trainer. 
```
# Miniconda is already installed if you are using the DLAMI but needs installation with Base AMI

wget -O miniconda.sh "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}.sh" \
    && bash miniconda.sh -b -p /apps/.conda \
          &&  /apps/.conda/bin/conda init bash  

source ~/.bashrc    
conda create --name bionemo python=${PYTHON_VERSION}

source activate bionemo

pip3 install -r requirements.txt

```
All package versions in the above `requirements.txt` file is recommended from Nvidia. An older version of the package `opencv-python-headless==4.8.0.74` has to be installed to avoid this [error](https://github.com/rom1504/img2dataset/issues/355) with [img2dataset](https://github.com/rom1504/img2dataset) package.



## 4. Build customized docker image
To achieve target performance of Nemo-Multimodal with EFA on P5 and P4de instances, we provide a customized 
`examples/14.nemo-multimodal/0.Dockerfile` and we can build a image like below:

```
docker build -t ${DOCKER_IMAGE_NAME}:${TAG} -f 0.Dockerfile .
```

## 5. Convert image
Convert the Docker container image to an [Enroot](https://github.com/NVIDIA/enroot) squash file that will be stored in `/apps`. This step takes a few minutes.
```
enroot import -o ${ENROOT_IMAGE}.sqsh dockerd://${DOCKER_IMAGE_NAME}

```

## 6. Download and preprocess data
We will use the popular [UniRef50](https://www.uniprot.org/help/uniref) dataset for pretraining. We will use BioNemo's in-built functionality to download and pre-process data. To this end, we provide `prepare_uniref50.py` file to do so. You can edit the above to download and process [UniRef90]((https://www.uniprot.org/help/uniref)). To run the above python code on your slurm cluster in the BioNemo cluster execute the following:

```bash
sbatch 1.uniref50.slurm
```

This will download raw data in `/fsx/raw/` and save pre-processed `train, validation and test` csv files in `/fsx/processed/`. The log files for submitted jobs are written to the local directory. To check the status of the datasets download job, you can tail the log file:

```bash
tail -f slurm-uniref-<slurm_job_id>.out
```



## 7. Pretrain ESM models
Now we are ready to submit distributed training jobs to pretrain `ESM1nv` models. We provide the `2.esm1nv_pretrain.slurm` script to run training 4 `p4de.24xlarge` nodes with `8xA100 80 GB` GPUs. Make sure data paths and model configuration is correct if you are running on custom data. To kick off distributed training execute:

```bash
sbatch 2.esm1nv_pretrain.slurm

```

Before kicking off training, first train, validation and test datasets are indexed and dataloaders are created and then you should see an example output like below:

```bash
Epoch 0:   3%|▎         | 34103/1100000 [5:28:58<171:22:21,  1.73it/s, loss=2.52, v_num=, reduced_train_loss=2.510, global_step=3.1e+4, consumed_samples=2.54e+8, val_loss=2.510]
Epoch 0:   3%|▎         | 34106/1100000 [5:29:00<171:22:19,  1.73it/s, loss=2.52, v_num=, reduced_train_loss=2.520, global_step=3.1e+4, consumed_samples=2.54e+8, val_loss=2.510]
Epoch 0:   3%|▎         | 34109/1100000 [5:29:02<171:22:09,  1.73it/s, loss=2.52, v_num=, reduced_train_loss=2.520, global_step=3.1e+4, consumed_samples=2.54e+8, val_loss=2.510]
Epoch 0:   3%|▎         | 34112/1100000 [5:29:03<171:22:00,  1.73it/s, loss=2.52, v_num=, reduced_train_loss=2.520, global_step=3.1e+4, consumed_samples=2.54e+8, val_loss=2.510]
```

## 8. Run container on Head Node [Troubleshooting]
Once the above image is pulled, you can run the container on the head node like below. This step could be used for troubleshooting purposes. Here we are running the container just to be able to copy launcher scripts on the host machine. If you need to run the container on the compute nodes, you would need to add `--gpus all` flag to the run command. It is recommended to have the docker run flags like below, as recommended by Nvidia PyTorch containers, otherwise you may potentially run into an error like [this](https://github.com/NVIDIA/Megatron-LM/issues/516)

```
 docker run -it nvcr.io/nvidia/clara/bionemo-framework:latest bash
```

