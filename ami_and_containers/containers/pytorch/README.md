# Template of PyTorch image optimized for GPU EC2 instances

The directory provides a sample `Dockerfile` intended as a reference. It provides optional stanzas
(commented or active). Instead of building this Dockerfile directly, we strongly recommend you to
read through this `Dockerfile`, understand what it does, then create your own `Dockerfile`
cherry-picking just the necessary stanzas for your use cases.

Before running a command, _make it a habit to always review_ scripts, configurations, or whatever
files involved. Very frequently, this directory requires you to edit files, or provides
explanations, tips and tricks in the form of comments within various files.

With that said, feel free to explore the example. Happy coding, and experimenting!

## 1. Essential software

In principle, the reference `Dockerfile` does the following:

- Provide PyTorch built for NVidia CUDA devices, by using a recent NVidia PyTorch image as the
  parent image.
- Remove unneccessary networking packages that might conflict with AWS technologies.
- Install EFA user-space libraries. It's important to avoid building the kernel drivers during
  `docker build`, and skip the self-tests, as both of these steps fail are expected to fail when run
  during container build.
- **OPTIONAL** -- On rare cases when your MPI application crashes when run under Slurm as `srun
  --mpi=pmix ...`, you might need to rebuild the OpenMPI to match with the PMIX version in the host.
- **OPTIONAL** -- Install NCCL in case the parent image hasn't caught up with the NCCL version you
  want to use.
- Typical environment variables for OpenMPI, EFA, and NCCL. Best practice to enforce these in the
  image, otherwise it can be error prone to manually set these variables when starting containers.

  **NOTE**: recent version of aws-ofi-nccl simplifies a lot of environment variables (see the [the
  official EFA cheatsheet in
  aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl/blob/master/doc/efa-env-var.md)). Hence, the
  provided template `Dockerfile` has (almost) no environment variables for EFA anymore (and this is
  one major simplification for those who've been exposed to older EFA examples elsewhere).
- User-space of gdrcopy -- **NOTE**: no-op (like this example) when already built-in in the parent
  image.
- Install [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) to get NCCL to utilize EFA.
- Install [nccl-test](https://github.com/NVIDIA/nccl-tests) as a built-in diagnostic tool.
- **OPTIONAL** -- Additional packages that worth to mention due to special build requirements, e.g.,
  installing [xformers](https://github.com/facebookresearch/xformers#install-troubleshooting) from
  source may encounter OOM unless special care is taken.
- And more reference stanzas may be added in future.

## 2. Frequently-used commands

Once you've created your own Dockerfile, it's time to build an image out of it:

```bash
# Build a Docker image
docker build --progress=plain -t nvidia-pt-od:latest .

# If Dockerfile is named differently, e.g., 0.nvcr-pytorch-aws.dockerfile
docker build --progress=plain -t nvidia-pt-od:latest -f 0.nvcr-pytorch-aws.dockerfile .

# Verify the image has been built
docker images

# Convert to enroot format. Attempt to remove an existing .sqsh, otherwise enroot refuses to
# run when the output .sqsh file already exists.
rm /fsx/nvidia-pt-od__latest.sqsh ; enroot import -o /fsx/nvidia-pt-od__latest.sqsh dockerd://nvidia-pt-od:latest
```

Tips: when building on a compute node (or a build node), you save the built Docker image on a shared
filesystem such as `/fsx`, to allow other nodes (e.g., head node, or other compute nodes) to load
the image to their local Docker registry.

```bash
# Build node: save image to file
docker save nvidia-pt-od:latest > /fsx/nvidia-pt-od__latest.tar

# Load image to local docker registry -> on head node, or new compute/build node
docker load < /fsx/nvidia-pt-od__latest.tar

# Verify the image has been loaded
docker images
```
