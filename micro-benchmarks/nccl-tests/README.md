# NCCL Tests

[NCCL Tests](https://github.com/NVIDIA/nccl-tests) enable you to evaluate the performance of the network using the Nvidia Collective Communication Library. This test case contains a Docker file and scripts to submit NCCL tests on Slurm or [Amazon EKS](https://aws.amazon.com/eks/). Please refer to the relevant instructions below, depending on your environment.

## 0. Prepare the runtime environment

### Slurm 
If you are using Slurm, this guide assumes that you have the following:

- A functional Slurm cluster on AWS.
- Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
- Enroot requires libmd to compile and squashfs-tools to execute.
- A shared directory mounted on `/apps`

It is recommended that you use the templates in the architectures [directory](../../1.architectures)

### Amazon EKS
If you are using EKS, this guide assumes that you have the following:

- A functional EKS cluster on AWS. <br/>
To set up one, please refer to [aws-do-eks](https://bit.ly/do-eks), [Amazon EKS Blueprints for Terraform](https://github.com/aws-ia/terraform-aws-eks-blueprints/tree/main), [Amazon EKS Blueprints for CDK](https://aws-quickstart.github.io/cdk-eks-blueprints/), or others.
- NVIDIA device plugin deployed to your cluster. <br/>
If you need to deploy it, please refer to [deployment/nvidia-device-plugin](https://github.com/aws-samples/aws-do-eks/blob/main/Container-Root/eks/deployment/nvidia-device-plugin) or [k8s-device-plugin/deployments](https://github.com/NVIDIA/k8s-device-plugin/tree/main/deployments).
- EFA device plugin deployed to your cluster. <br/>
If you need to deploy it, please refer to [deployment/efa-device-plugin](https://github.com/aws-samples/aws-do-eks/tree/main/Container-Root/eks/deployment/efa-device-plugin) or [aws-efa-eks](https://github.com/aws-samples/aws-efa-eks).
- Kubeflow MPI operator deployed to your cluster. <br/>
If you need to deploy it, please refer to [deployment/kubeflow/mpi-operator](https://github.com/aws-samples/aws-do-eks/tree/main/Container-Root/eks/deployment/kubeflow/mpi-operator) or [kubeflow/mpi-operator](https://github.com/kubeflow/mpi-operator). 
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#cliv2-linux-install)

## 1. Prepare the container image and other artifacts

The NCCL tests are packaged in a container.

> You can set versions and the branch for NCCL and EFA by editing the variables below in the Dockerfile.

> | Variable              | Default     | Repository                                                                                  |
> |-----------------------|-------------|---------------------------------------------------------------------------------------------|
> |`CUDA_VERSION`         | `13.0.2`    |                                                                                             |
> |`GDRCOPY_VERSION`      | `v2.5.2`    | [link](https://github.com/NVIDIA/gdrcopy)                                                   |
> |`EFA_INSTALLER_VERSION`| `1.48.0`    | [link](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-enable) |
> |`AWS_OFI_NCCL_VERSION` | *(deprecated)* | AWS OFI NCCL plugin is now bundled with EFA installer                                    |
> |`NCCL_VERSION`         | `v2.30.4-1` | [link](https://github.com/NVIDIA/nccl)                                                      |
> |`NCCL_TESTS_VERSION`   | `v2.18.3`   | [link](https://github.com/NVIDIA/nccl-tests)                                                |

The image is built with `NVCC_GENCODE` covering `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, and `sm_103` — i.e. native binaries for A100, RTX 30/Ada/Lovelace, H100/H200, B200/GB200, and B300/GB300. PTX is not embedded; if you target a newer architecture, add it to the `NVCC_GENCODE` lines in `nccl-tests.Dockerfile`.

You must pick each version of the library and set them as variables before proceed:

```bash
GDRCOPY_VERSION=v2.5.2
EFA_INSTALLER_VERSION=1.48.0
NCCL_VERSION=v2.30.4-1
NCCL_TESTS_VERSION=v2.18.3
TAG="efa${EFA_INSTALLER_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}"
CONTAINER_IMAGE_NAME_TAG="nccl-tests:${TAG}"
```

### Build the container

If you wish to build the containar image by yourself, follow this section. Alternatively, you can use a prebuild image on a public ECR repository `public.ecr.aws/hpc-cloud/nccl-tests`. If you wish to do so, skip this section.

1. Build the container image with the command below:
   ```bash
   docker build -f nccl-tests.Dockerfile \
          --build-arg="GDRCOPY_VERSION=${GDRCOPY_VERSION}" \
          --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
          --build-arg="NCCL_VERSION=${NCCL_VERSION}" \
          --build-arg="NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION}" \
          -t ${CONTAINER_IMAGE_NAME_TAG} \
          .
   ```
   Note: If you are using an arm64 platform (like p6e-gb200.36xlarge, or any ultraserver comprised of that instance, for example), pass in `--platform=linux/arm64` into the `docker build` command above. Or, just build your image directly on an `arm64` based instance!
 
1. Once the container image is prepared, you can check if it is present with `docker images`. You should see an output similar to this one:
   ```
   REPOSITORY               TAG                        IMAGE ID       CREATED         SIZE
   nccl                     latest                     6e981e5cf6a5   5 hours ago     8.61GB
   ...
   nvidia/cuda              13.0.2-devel-ubuntu22.04   a86c511c87e1   2 weeks ago     6.56GB
   ```

### Slurm

To run the NCCL tests on Slurm, you will need to convert the container into a Squash file using Enroot.

Convert the container image to a squash file via Enroot. If you have the built image locally use the following command:

   ```bash
   enroot import -o /fsx/nccl-tests.sqsh dockerd://${CONTAINER_IMAGE_NAME_TAG}
   ```

If you want to pull the image from the public ECR use the following command:

   ```bash
   enroot import -o /fsx/nccl.sqsh dockerd://public.ecr.aws/hpc-cloud/${CONTAINER_IMAGE_NAME_TAG}
   ```

The file will be stored in the `/fsx` directory.

### Amazon EKS

To run the NCCL tests on EKS with the local image you built in the previous step, you will need to build the container image, then push it to a container registry, such as the private [ECR](https://aws.amazon.com/ecr/) in your AWS account.
You can skip this part if you use pre-built image on `public.ecr.aws/hpc-cloud/nccl-tests`.

1. Create the ECR repository if it does not exist
   ```bash
   ECR_REPOSITORY_NAME="nccl-tests"
   aws ecr create-repository --repository-name ${ECR_REPOSITORY_NAME}
   ```

1. Get the ECR repository URI:
   ```bash
   REPO_URI=`aws ecr describe-repositories --query repositories[].[repositoryUri] | grep "/${ECR_REPOSITORY_NAME}" | tr -d '"' | xargs`
   ECR_URI=${REPO_URI%"/${ECR_REPOSITORY_NAME}"}
   ```

1. Build the container image:
   ```bash
   docker image build -t ${REPO_URI}:${TAG} -f ./nccl-tests.Dockerfile .
   ```
1. Login to the container registry
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_URI}
   ```

1. Push the container image to the registry
   ```bash
   docker image push ${REPO_URI}:${TAG}
   ```

## 2. Running the NCCL Tests

Note: For topology aware NCCL tests, with features like export to csv, 
passing in a topologically sorted hostfile to mpirun, look in slurm/topology-aware-nccl-tests

### Slurm with container

Copy the file `slurm/nccl-tests.sbatch` or its content on your cluster then submit a preprocessing jobs with the command below:

```bash
sbatch nccl-tests.sbatch
```

### Slurm with Deep Learning AMI

In this step, you can use the NCCL tests already compiled in the deep learning AMI.

Copy the file `slurm/nccl-tests-deep-learning-ami.sbatch` or its content on your cluster then submit a preprocessing jobs with the command below:

```bash
sbatch nccl-tests.sbatch
```

### Results

All_reduce performance test will be executed from 8B to 2GB on 2x p4de.24xlarg, the output should look as below (with a lot more information).
```txt
0: #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
0: #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
0:            8             2     float     sum      -1    164.3    0.00    0.00      0    163.3    0.00    0.00      0
0:           16             4     float     sum      -1    161.4    0.00    0.00      0    160.6    0.00    0.00      0
0:           32             8     float     sum      -1    161.5    0.00    0.00      0    160.9    0.00    0.00      0
0:           64            16     float     sum      -1    161.0    0.00    0.00      0    160.6    0.00    0.00      0
0:          128            32     float     sum      -1    161.2    0.00    0.00      0    161.2    0.00    0.00      0
0:          256            64     float     sum      -1    161.8    0.00    0.00      0    161.5    0.00    0.00      0
0:          512           128     float     sum      -1    162.3    0.00    0.01      0    161.5    0.00    0.01      0
0:         1024           256     float     sum      -1    165.0    0.01    0.01      0    164.8    0.01    0.01      0
0:         2048           512     float     sum      -1    177.6    0.01    0.02      0    178.5    0.01    0.02      0
0:         4096          1024     float     sum      -1    169.4    0.02    0.05      0    170.0    0.02    0.05      0
0:         8192          2048     float     sum      -1    175.9    0.05    0.09      0    175.7    0.05    0.09      0
0:        16384          4096     float     sum      -1    193.4    0.08    0.16      0    192.5    0.09    0.16      0
0:        32768          8192     float     sum      -1    224.6    0.15    0.27      0    223.7    0.15    0.27      0
0:        65536         16384     float     sum      -1    227.4    0.29    0.54      0    225.5    0.29    0.54      0
0:       131072         32768     float     sum      -1    229.7    0.57    1.07      0    226.8    0.58    1.08      0
0:       262144         65536     float     sum      -1    238.2    1.10    2.06      0    235.5    1.11    2.09      0
0:       524288        131072     float     sum      -1    260.3    2.01    3.78      0    259.9    2.02    3.78      0
0:      1048576        262144     float     sum      -1    309.4    3.39    6.35      0    307.3    3.41    6.40      0
0:      2097152        524288     float     sum      -1    432.6    4.85    9.09      0    397.4    5.28    9.90      0
0:      4194304       1048576     float     sum      -1    533.9    7.86   14.73      0    530.3    7.91   14.83      0
0:      8388608       2097152     float     sum      -1    762.8   11.00   20.62      0    760.7   11.03   20.68      0
0:     16777216       4194304     float     sum      -1   1191.8   14.08   26.39      0   1190.4   14.09   26.42      0
0:     33554432       8388608     float     sum      -1   1540.2   21.79   40.85      0   1540.7   21.78   40.83      0
0:     67108864      16777216     float     sum      -1   2644.6   25.38   47.58      0   2647.2   25.35   47.53      0
0:    134217728      33554432     float     sum      -1   3912.9   34.30   64.31      0   3932.8   34.13   63.99      0
0:    268435456      67108864     float     sum      -1   7201.7   37.27   69.89      0   7227.9   37.14   69.64      0
0:    536870912     134217728     float     sum      -1    13552   39.62   74.28      0    13548   39.63   74.30      0
0:   1073741824     268435456     float     sum      -1    26217   40.96   76.79      0    26194   40.99   76.86      0
0:   2147483648     536870912     float     sum      -1    51406   41.78   78.33      0    51406   41.77   78.33      0
```

All_reduce performance test will be executed from 8B to 16GB on 2x p5.48xlarge, the output should look as below (with a lot more information).
```txt
0: #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
0: #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
0:            8             2     float     sum      -1    69.12    0.00    0.00      0    72.43    0.00    0.00      0
0:           16             4     float     sum      -1    72.64    0.00    0.00      0    73.91    0.00    0.00      0
0:           32             8     float     sum      -1    74.06    0.00    0.00      0    64.75    0.00    0.00      0
0:           64            16     float     sum      -1    65.48    0.00    0.00      0    74.40    0.00    0.00      0
0:          128            32     float     sum      -1    74.92    0.00    0.00      0    65.43    0.00    0.00      0
0:          256            64     float     sum      -1    74.12    0.00    0.01      0    65.70    0.00    0.01      0
0:          512           128     float     sum      -1    69.50    0.01    0.01      0    66.88    0.01    0.01      0
0:         1024           256     float     sum      -1    69.24    0.01    0.03      0    69.04    0.01    0.03      0
0:         2048           512     float     sum      -1    72.22    0.03    0.05      0    71.29    0.03    0.05      0
0:         4096          1024     float     sum      -1    78.58    0.05    0.10      0    78.55    0.05    0.10      0
0:         8192          2048     float     sum      -1    81.44    0.10    0.19      0    80.47    0.10    0.19      0
0:        16384          4096     float     sum      -1    94.36    0.17    0.33      0    82.35    0.20    0.37      0
0:        32768          8192     float     sum      -1    111.7    0.29    0.55      0    89.75    0.37    0.68      0
0:        65536         16384     float     sum      -1    135.1    0.48    0.91      0    103.8    0.63    1.18      0
0:       131072         32768     float     sum      -1    108.9    1.20    2.26      0    96.55    1.36    2.55      0
0:       262144         65536     float     sum      -1    128.0    2.05    3.84      0    104.7    2.50    4.70      0
0:       524288        131072     float     sum      -1    123.7    4.24    7.95      0    113.3    4.63    8.67      0
0:      1048576        262144     float     sum      -1    123.2    8.51   15.95      0    121.3    8.64   16.21      0
0:      2097152        524288     float     sum      -1    147.2   14.24   26.70      0    147.1   14.25   26.72      0
0:      4194304       1048576     float     sum      -1    168.6   24.87   46.64      0    167.7   25.02   46.91      0
0:      8388608       2097152     float     sum      -1    204.8   40.96   76.80      0    201.1   41.71   78.20      0
0:     16777216       4194304     float     sum      -1    298.1   56.28  105.52      0    298.3   56.24  105.45      0
0:     33554432       8388608     float     sum      -1    439.7   76.31  143.09      0    417.7   80.33  150.62      0
0:     67108864      16777216     float     sum      -1    601.5  111.57  209.19      0    604.2  111.07  208.26      0
0:    134217728      33554432     float     sum      -1    870.3  154.22  289.16      0    876.8  153.07  287.01      0
0:    268435456      67108864     float     sum      -1   1468.2  182.83  342.81      0   1452.7  184.78  346.46      0
0:    536870912     134217728     float     sum      -1   2559.2  209.78  393.34      0   2554.9  210.14  394.00      0
0:   1073741824     268435456     float     sum      -1   4607.6  233.04  436.95      0   4565.6  235.18  440.96      0
0:   2147483648     536870912     float     sum      -1   9074.5  236.65  443.72      0   9108.5  235.77  442.06      0
0:   4294967296    1073741824     float     sum      -1    17286  248.46  465.87      0    17343  247.64  464.33      0
0:   8589934592    2147483648     float     sum      -1    33605  255.62  479.28      0    33628  255.44  478.95      0
0:  17179869184    4294967296     float     sum      -1    66132  259.78  487.09      0    66109  259.87  487.26      0
```

To change the type of collective to test, modify the line with `srun` in the file `nccl-tests.sbatch` and change `all_reduce_perf` to any of: `all_gather_perf`, `alltoall_perf`, `gather_perf`, `reduce_perf`, `scatter_perf`, `all_reduce_perf`, `broadcast_perf`, `hypercube_perf`, `reduce_scatter_perf`, `sendrecv_perf`.


### Amazon EKS

1. Prepare the MPIJob manifest
   Edit file `kubernetes/nccl-tests.yaml` and adjust the following values:

   - `slotsPerWorker: 8`: set to the number of GPUs per node in your cluster
   - `<account>.dkr.ecr.<region>.amazonaws.com/<image>:<tag>`: set to your container image URI. You may specify the public ECR image instead as `image: public.ecr.aws/hpc-cloud/nccl-tests:<tag>`. Note: change both locations in the file. You may use `echo ${CONTAINER_IMAGE_NAME_TAG}` to print the image URI.
   - `-np 16`: set -np option in mpirun to (*`number_of_worker_nodes`* * *`number_of_gpus_per_node`*), other mpirun parameters if needed for your instance type, please refer to [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl/blob/master/doc/efa-env-var.md)
   - `replicas: 2`: set to number of worker pods you would like the test to run on. This must be less than or eaqual to the number of nodes in your cluster.
   - `node.kubernetes.io/instance-type: "p5.48xlarge"`: set to the instance type of the nodes in your cluster against which you would like the nccl test to be run
   - `nvidia.com/gpu: 8`: set to the number of GPUs per node in your cluster, adjust in both the limits and requests section
   - `vpc.amazonaws.com/efa: 32`: set to the number of EFA adapters per node in your cluster, adjust in both the limits and requests section

   Please note that the current default settings have been specified for instance type p5.48xlarge. Only the image URI is required to be set for running the test on this instance type.
   The current manifest executes the `all_reduce_perf` test. If you wish to execute other NCCL tests, change the section between lines 59 and 73 in this MPIJob manifest file. 

2. Apply the MPIJob manifest to the cluster
   ```bash
   kubectl apply -f ./nccl-tests.yaml
   ```

3. Wait until pods to enter the Running state
   To monitor the state of the pods, execute the following command:
   ```bash
   watch kubectl get pods -o wide
   ```
   Once the state of the launcher and worker pods becomes "Running", press `Ctrl-C` to return to the command prompt.

4. View test logs
   To follow the test logs, execute the following command:
   ```bash
   kubectl logs -f $(kubectl get pods | grep launcher | cut -d ' ' -f 1)
   ```

   The following is an example exerpt from the logs of a NCCL all_reduce_perf test, executed on a cluster with two p5.48xlarge instances (using EFA_INSTALLER_VERSION=1.28.0, NCCL_TESTS_VERSION=master, NCCL_VERSION=2.18.5):

   ```log
   [1,0]<stdout>:#                                                              out-of-place                       in-place          
   [1,0]<stdout>:#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
   [1,0]<stdout>:#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
   [1,0]<stdout>:           0             0     float     sum      -1    15.51    0.00    0.00      0    15.52    0.00    0.00      0
   ...
   [1,0]<stdout>:    67108864      16777216     float     sum      -1    996.0   67.38  126.33      0   1001.7   66.99  125.62      0
   [1,0]<stdout>:   134217728      33554432     float     sum      -1   1950.5   68.81  129.02      0   1725.6   77.78  145.83      0
   [1,0]<stdout>:   268435456      67108864     float     sum      -1   3010.8   89.16  167.17      0   3020.7   88.87  166.62      0
   [1,0]<stdout>:   536870912     134217728     float     sum      -1   3608.0  148.80  279.00      0   3599.7  149.14  279.64      0
   [1,0]<stdout>:  1073741824     268435456     float     sum      -1   6426.3  167.09  313.29      0   6426.1  167.09  313.29      0
   [1,0]<stdout>:  2147483648     536870912     float     sum      -1   9197.5  233.49  437.79      0   9195.2  233.54  437.89      0
   [1,0]<stdout>:# Out of bounds values : 0 OK
   [1,0]<stdout>:# Avg bus bandwidth    : 52.9753
   ```
   Press `Ctrl-C` to return to the command prompt if you do not wish to wait until the launcher pod enters the "Completed" state.

5. Clean up test run
   Before running a subsequent test, the current MPIJob needs to be deleted:
   ```bash
   kubectl delete -f nccl-tests.yaml
   ```

## 3. Understanding NCCL Bandwidth

The NCCL tests reports metrics for the time to execute a given communication collective operation, the Algorithmic bandwidth and the bus bandwidth.

The algorithm bandwidth is based on the following data_size / time where data_size is the size of the data being exchanged through the collective operation while time is the time taken by the operation. The bus bandwidth is generated using a formula specific to each collective operation to reflect the speed of inter-GPU communications. This metric can be used to compare to the hardware peak bandwidth “independently to the number of ranks used” (as shared here).

| API           | Algbw                                              | Busbw                                    | Theoretical Max BW    | source                              |
|---------------|----------------------------------------------------|------------------------------------------|-----------------------|-------------------------------------|
| AllReduce     | baseBw = (count * typesize) / 1.0E9 / sec          | busBw = baseBw * (2*(nranks - 1)/nranks) | B = S/t * (2*(n-1)/n) | https://tinyurl.com/all-reduce      |
| ReduceScatter | baseBw = (count * nranks * typesize) / 1.0E9 / sec | busBw = baseBw * ((nranks - 1)/nranks)   | B = S/t * (n-1)/n     | https://tinyurl.com/reduce-scatter  |
| AllGather     | baseBw = (count * typesize) / 1.0E9 / sec          | busBw = baseBw * ((nranks - 1)/nranks)   | B = S/t * (n-1)/n     | https://tinyurl.com/all-gather      |
| Broadcast     | baseBw = (count * typesize) / 1.0E9 / sec          | busBw = baseBw                           | B = S/t               | https://tinyurl.com/nccl-broadcast  |
| Gather        | baseBw = (count * nranks * typesize) / 1.0E9 / sec | busBw = baseBw * ((nranks - 1)/nranks)   | B = S/t * (n-1)/n     | https://tinyurl.com/nccl-gather     |
| Reduce        | baseBw = (count * typesize) / 1.0E9 / sec          | busBw = baseBw                           | B = S/t               | https://tinyurl.com/nccl-reduce     |
| Scatter       | baseBw = (count * nranks * typesize) / 1.0E9 / sec | busBw = baseBw * ((nranks - 1)/nranks)   | B = S/t * (n-1)/n     | https://tinyurl.com/nccl-scatter    |
| AlltoAll      | baseBw = (count * nranks * typesize) / 1.0E9 / sec | busBw = baseBw * ((nranks - 1)/nranks)   | B = S/t * (n-1)/n     | https://tinyurl.com/nccl-all-to-all |
| SendRecv      | baseBw = (count * typesize) / 1.0E9 / sec          | busBw = baseBw                           | B = S/t               | https://tinyurl.com/sendrcv         |



#### Notes for Algbw & Busbw**

* `typesize` : size of the data type transferred in bytes (2 bytes for half-precision, 4 for single precision....).
* `count` : number of elements transferred through the collective communication operation.
* `nranks` : number of ranks participating to the collective communication operation.
* `sec` : time in seconds to execute the collective communication operation.

#### Notes for the Theoretical Max BW

The formula defines the maximum theoretical bandwidth that can be achieved on different communication collectives in the ideal case.

* `n` : number of ranks participating to the operation. (similar to nranks for Algbw and Busbw)
* `t` : time to complete the operation. (similar to sec for Algbw and Busbw)
* `S` : number of elements being communicated (similar to count for Algbw and Busbw)
* `B` : theoretical peak bandwidth.