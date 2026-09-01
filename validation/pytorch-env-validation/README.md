# PyTorch Environment Validation

This test runs a PyTorch script to screen for NCCL, MPI, OpenMP, CUDA.... on your environment. This script is executed once per instance and helps you verify your environment: The AWS [Deep Learning Container](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html) is used for that purpose.

Here you will:
- Build a container from the AWS [Deep Learning Container](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html) and convert it to a squash file using [Enroot](https://github.com/NVIDIA/enroot).
- Run a Python script to screen the PyTorch environment with [Pyxis](https://github.com/NVIDIA/pyxis) via Slurm.
- Mount a local directory in the container via Pyxis.

## 0. Preparation

This guide assumes that you have the following:

- A functional Slurm cluster on AWS.
- Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
- Enroot requires libmd to compile and squashfs-tools to execute.
- A shared directory mounted on `/apps`

It is recommended that you use the templates in the architectures [directory](../../architectures) to deploy Slurm (for example AWS ParallelCluster).


## 1. Build the container and the squash file

We use the AWS [Deep Learning Container](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html) as a base for your validation container and the EFA libraries to use the latest versions. Here, you will start by building your container image then convert it to a squash file via Enroot.

To build the container:

1. Copy the file `0.pytorch-screenl.Dockerfile` or its content to your head-node.
2. Build the container image with the command below
   ```bash
   # get the region, this assumes we run on EC2
   AWS_AZ=$(ec2-metadata --availability-zone | cut -d' ' -f2)
   AWS_REGION=${AWS_AZ::-1}

   # Authenticate with ECR to get the AWS Deep Learning Container
   aws ecr get-login-password | docker login --username AWS \
      --password-stdin 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training

   # Build the container
   docker build -t pytorch-screenl -f 0.pytorch-screenl.Dockerfile --build-arg="AWS_REGION=${AWS_AZ::-1}" .
   ```
3. Once the image is built, you can check if it is present with `docker images`. You should see an output similar to this one:
   ```
   REPOSITORY                                                           TAG                                     IMAGE ID       CREATED         SIZE
   pytorch-screen                                                       latest                                  2892fe08195a   2 minutes ago   21.6GB
   ...
   763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training   2.0.1-gpu-py310-cu118-ubuntu20.04-ec2   3d25d3d0f25e   2 months ago    20.8GB
   ...
   ```
3. Convert the container image to a squash file via Enroot
   ```bash
   enroot import -o /apps/pytorch-screen.sqsh  dockerd://pytorch-screen:latest
   ```
   The file will be stored in the `/apps` directory.

> You can set versions and the branch for NCCL and EFA by editing the variables below in the Dockerfile.

> | Variable              | Default     |
> |-----------------------|-------------|
> |`EFA_INSTALLER_VERSION`| `latest`    |
> |`AWS_OFI_NCCL_VERSION` | `aws`       |

## 2. Running the Pytorch screening

Now you copy the files `1.torch-screen.sbatch` and `pytorch-screen.py` to your cluster in the same directory then submit a test job with the command below from where the files are placed:

```bash
sbatch 1.torch-screen.sbatch
```

An output file named `slurm-XX.out`, with `XX` being the job ID, will be placed in the directory. It will report the environment variables, location of `python`, `nvidia-smi` and PyTorch environment variables for each node (instance). Please keep in mind that each process, 1 per node, will write concurrently to the output file. Each process output is prepended by their ID `:0` for process 0, `:1` for process 1. These can be interleaved. Below is an example of output:


```bash
0: torch.backends.opt_einsum.strategy=None
0: torch.distributed.is_available()=True
0: torch.distributed.is_mpi_available()=True
0: torch.distributed.is_nccl_available()=True
1: torch.cuda.is_available()=True
1: torch.backends.cuda.is_built()=True
1: torch.backends.cuda.matmul.allow_tf32=False
1: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=True
1: torch.backends.cuda.cufft_plan_cache=<torch.backends.cuda.cuFFTPlanCacheManager object at 0x7f72d0415a80>
1: torch.backends.cuda.preferred_linalg_library(backend=None)=<_LinalgBackend.Default: 0>
1: torch.backends.cuda.flash_sdp_enabled()=True
```

> **Execute on X number nodes?**: to change the number of nodes modify the line `SBATCH -N 2` and change `2` to the desired number of nodes on which you'd like to run this script.
