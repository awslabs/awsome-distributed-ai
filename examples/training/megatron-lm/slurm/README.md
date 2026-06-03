# Running Megatron-LM on Slurm

This directory contains scripts and instructions for setting up the Megatron-LM training environment on a Slurm cluster. For detailed instructions on running distributed training jobs with this environment, please refer to the subdirectories.


## 1. Preparation

This guide assumes that you have the following:

- A functional Slurm on AWS.
- Docker, for Slurm [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) need to be installed as well.
- An FSx for Lustre filesystem mounted on `/fsx` in all Slurm nodes.

It is recommended that you use the templates for [AWS Parallel Cluster](../../../architectures/aws-parallelcluster/) or [Amazon SageMaker HyperPod Slurm](../../../architectures/sagemaker-hyperpod-slurm) set up.

You will also setup the following variables in your terminal environment.

```bash
export DATA_PATH=/fsx/data # FSx for Lustre shared file-system
```

The following instructions assume you have cloned this repository under such a shared filesystem and changed your current directory to this directory.


## 2. Environment Setup 

This section of the guide how to build a Megatron-LM container then convert it into a Squash file via [Enroot](https://github.com/NVIDIA/enroot).

Below are the steps you need to follow:

1. Copy the file `0.distributed-training.Dockerfile` or its content to your head-node or any instance where you have the [Docker](https://docs.docker.com/get-docker/) cli available.
2. Build the container image with the command below

   ```bash
   docker build -t aws-megatron-lm -f ../aws-megatron-lm.Dockerfile .
   ```

3. Once the image is built, you can check if it is present with `docker images`. You should see an output similar to this one:

   ```text
   [ubuntu@ip-10-0-10-78 ~]$ docker images
   REPOSITORY               TAG         IMAGE ID       CREATED          SIZE
   megatron-training           latest      a33c9d5bcb6e   9 seconds ago    20.7GB
   ```

4. Prepare the image for your target environment.

   Create the squash file with the command below.
 
   ```bash
   enroot import -o aws-megatron-lm.sqsh  dockerd://aws-megatron-l:latest
   ```

   The file will be stored in the current directory (if left as default). The output should look as below.

    ```bash
    [ubuntu@ip-10-0-10-78 ~]$ enroot import -o ./megatron-training.sqsh  dockerd://megatron-training:latest
    [INFO] Fetching image

    e19aa13505c1710876982dc440226dc479da5177dc4770452cc79bedc8b5b41d

    [INFO] Extracting image content...
    [INFO] Creating squashfs filesystem...

    Parallel mksquashfs: Using 32 processors
    Creating 4.0 filesystem on /home/ec2-user/megatron-training.sqsh, block size 131072.
    [==========================================================/] 299550/299550 100%

    Exportable Squashfs 4.0 filesystem, gzip compressed, data block size 131072
       uncompressed data, uncompressed metadata, uncompressed fragments, uncompressed xattrs
       duplicates are not removed
    ...
    ```

## 2. Next Steps 

Now that you have the Megatron-LM container and have enabled the squash file, you can scale your training job with the container on your Slurm cluster. The subdirectories illustrate detailed end-to-end instructions for different models.



