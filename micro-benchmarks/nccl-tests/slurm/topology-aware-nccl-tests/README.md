# NCCL Tests

[NCCL Tests](https://github.com/NVIDIA/nccl-tests) enable you to evaluate the performance of the network using the Nvidia Collective Communication Library. This test case contains a Docker file and scripts to submit NCCL tests on Slurm. Please refer to the relevant instructions below, depending on your environment.

**This is a newer version of slurm tests with additional features**
- Run in container mode or AMI mode 
- Batch submission of multiple test combinations
- Configurable test parameters in the script
- Conversion of nccl test result summary to csv
- Support for topology-aware scheduling

## 0. Prepare the runtime environment

### Slurm 
If you are using Slurm, this guide assumes that you have the following:

- A functional Slurm cluster on AWS.
- Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
- Enroot requires libmd to compile and squashfs-tools to execute.
- A shared directory mounted on `/fsxl`

It is recommended that you use the templates in the architectures [directory](../../../../architectures)

## 1. Prepare the container image and other artifacts

The NCCL tests are packaged in a container.

> You can set versions and the branch for NCCL and EFA by editing the variables below in the Dockerfile.

> | Variable              | Default     | Repository                                                                                  |
> |-----------------------|-------------|---------------------------------------------------------------------------------------------|
> |`CUDA_VERSION`         | `12.8.1`    |                                                                                             |
> |`GDRCOPY_VERSION`      | `v2.5.1`    | [link](https://github.com/NVIDIA/gdrcopy)                                                   |
> |`EFA_INSTALLER_VERSION`| `1.43.2`    | [link](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-enable) |
> |`AWS_OFI_NCCL_VERSION` | `v1.16.3`   | [link](https://github.com/aws/aws-ofi-nccl)                                                 |
> |`NCCL_VERSION`         | `v2.27.7-1` | [link](https://github.com/NVIDIA/nccl)                                                      |
> |`NCCL_TESTS_VERSION`   | `v2.16.9`   | [link](https://github.com/NVIDIA/nccl-tests)                                                |

You must pick each version of the library and set them as variables before proceed:

```bash
GDRCOPY_VERSION=v2.5.1
EFA_INSTALLER_VERSION=1.43.2
AWS_OFI_NCCL_VERSION=v1.16.3
NCCL_VERSION=v2.27.7-1
NCCL_TESTS_VERSION=v2.16.9
TAG="efa${EFA_INSTALLER_VERSION}-ofi${AWS_OFI_NCCL_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}"
CONTAINER_IMAGE_NAME_TAG="nccl-tests:${TAG}"
```

### Build the container

If you wish to build the container image by yourself, follow this section. Alternatively, you can use a prebuilt image on a public ECR repository `public.ecr.aws/hpc-cloud/nccl-tests`. If you wish to do so, skip this section.

1. Build the container image with the command below:
   ```bash
    #Navigate to the slurm directory:
   cd micro-benchmarks/nccl-tests/slurm/

   docker build -f nccl-tests.Dockerfile \
          --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
          --build-arg="AWS_OFI_NCCL_VERSION=${AWS_OFI_NCCL_VERSION}" \
          --build-arg="NCCL_VERSION=${NCCL_VERSION}" \
          --build-arg="NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION}" \
          -t ${CONTAINER_IMAGE_NAME_TAG} \
          .
   
   ```
 
1. Once the container image is prepared, you can check if it is present with `docker images`. You should see an output similar to this one:
   ```
   REPOSITORY               TAG                        IMAGE ID       CREATED         SIZE
   nccl                     latest                     6e981e5cf6a5   5 hours ago     8.61GB
   ...
   nvidia/cuda              12.8.1-devel-ubuntu22.04   a86c511c87e1   2 weeks ago     6.56GB
   ```

### Slurm

To run the NCCL tests on Slurm, you will need to convert the container into a Squash file using Enroot.

Convert the container image to a squash file via Enroot. If you have the built image locally use the following command:

   ```bash
   enroot import -o /fsxl/nccl-tests.sqsh dockerd://${CONTAINER_IMAGE_NAME_TAG}
   ```

If you want to pull the image from the public ECR use the following command:

   ```bash
   enroot import -o /fsxl/nccl.sqsh dockerd://public.ecr.aws/hpc-cloud/${CONTAINER_IMAGE_NAME_TAG}
   ```

The file will be stored in the `/fsxl` directory.

## 2. Running the NCCL Tests

### Slurm with container

clone the awsome-distributed-ai repo on your head node
`git clone https://github.com/awslabs/awsome-distributed-ai.git`


Navigate to the topology-aware-nccl-tests directory:
```bash
cd topology-aware-nccl-tests
```


### Supported Operations

| Operation | Description |
|-----------|-------------|
| `allreduce` | Combines values from all ranks and distributes result to all ranks |
| `allgather` | Gathers data from all ranks and distributes to all ranks |
| `reducescatter` | Combines values and scatters results across ranks |
| `alltoall` | Each rank sends different data to every other rank |
| `gather` | Gathers data from all ranks to a single root rank |
| `reduce` | Combines values from all ranks to a single root rank |
| `scatter` | Scatters data from root rank to all other ranks |
| `broadcast` | Broadcasts data from root rank to all other ranks |
| `hypercube` | Hypercube communication pattern test |
| `sendrecv` | Point-to-point send/receive operations |

### Running multiple operations in parallel

Here are the two common masks in use to partition the set of GPUs into smaller sets, each executing the same operation in parallel while measuring nccl performance.

| Mask | Description | Use Case |
|---------|-------------|----------|
| `0x0` | All zeros | This is equivalent to NCCL_TESTS_SPLIT="AND 0x0" . This disables the gpu split: all GPUs participate together in a single operation, maximizing intra-group communication and measuring full payload bandwidth for the entire set. Use 0x0 to aggregate all GPUs, focusing on overall system communication performance|
| `0x7` | Bit pattern 0111 | This is equivalent to NCCL_TESTS_SPLIT="AND 0x7" or NCCL_TESTS_SPLIT="MOD 8": On systems with 8 GPUs, run 8 parallel operations, each with 1 GPU per node (purely communicating over the inter-node network). Use this to split large clusters into many single-GPU groups for measuring individual inter-node or isolated bandwidths |

Refer to [nccl-tests] (https://github.com/NVIDIA/nccl-tests?tab=readme-ov-file#running-multiple-operations-in-parallel) for more information 

### Advanced Features

#### Topology-Aware Scheduling
Enable topology optimization by providing a sorted hostfile to mpirun:

In November 2023, AWS announced the [Instance Topology API](https://aws.amazon.com/about-aws/whats-new/2023/11/instance-topology-api-ml-hpc-workloads/).
It provides customers a unique per account hierarchical view of the relative proximity between Amazon EC2 instances.
To learn more, please visit the [EC2 User Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-topology.html).

There are two way to use the topology API to maximize nccl performance.

1. Minimize the number of switch levels that need to be crossed in a single job.  With Slurm, you can achieve this by using the slurm topology plugin. To enable topology plugin in slurm, refer to this repo [ec2-topology-aware-for-slurm](https://github.com/aws-samples/ec2-topology-aware-for-slurm?tab=readme-ov-file.) . Slurm will then attempt to allocate job resources based on topology.

2. Once a job's resources are allocated, you want to have the NCCL communicator ranks organized so that (on average) communicator ranks that are close together are physically close together.  Slurm natively does not have a way to do this.  It assumes that hostnames are generated in a way that this is true based on a pure sorting of the hostnames in the job after (1) happens.  But because hostnames are not assigned based on topology in EC2, that doesn't work.  This is why you also should also pass the sorted hostfile when launching the mpirun job running your nccl test.

Follow the steps below to generate a topologically sorted hostfile which you can pass to mpirun

```bash
# First Generate hostfile. Run this in on your cluster head node
./generate_hostfile.sh
# this will generate a file with of name hostnames.txt which contains all the nodes in your cluster

# Second sort it by passing it to hostfile_topologify.py
# Replace us-east-1 with your actual AWS region
python hostfile_topologify.py --input hostnames.txt --output topo_sorted_hostnames.txt --region us-east-1

# Edit submit script to use topology file
# Set TOPO_SORTED_FILE="topo_sorted_hostnames.txt" in submit_nccl_test_ami.sh
```

**hostfile_topologify.py Usage:**
```bash
# Basic usage with default region (us-east-1)
python hostfile_topologify.py --input hostnames.txt --output sorted_hostnames.txt

# Specify custom AWS region
python hostfile_topologify.py --input hostnames.txt --output sorted_hostnames.txt --region ap-northeast-1

# Output to stdout (default)
python hostfile_topologify.py --input hostnames.txt --region eu-west-1
```

**Parameters:**
- `--input`: Input hostfile containing node hostnames (required)
- `--output`: Output file for sorted hostnames (optional, defaults to stdout)
- `--region`: AWS region where your cluster is deployed (optional, defaults to us-east-1)

#### Container Mode 
```bash
# Single test defaults to 0x0 test split mask
sbatch nccl-tests-container.sbatch allreduce

#Run all supported collectives (allreduce, allgather,reducescatter, alltoall ) and test split mask (0x0, 0x7)
./submit_nccl_test_container.sh
```


#### AMI Mode 
```bash
# Single test defaults to 0x0 test split mask
sbatch nccl-tests-ami.sbatch allreduce

#Run all supported collectives (allreduce, allgather,reducescatter, alltoall ) and test split masks (0x0, 0x7)
./submit_nccl_test_ami.sh
```

#### Custom Parameters

**Container Mode**: Modify `submit_nccl_test_container.sh`:
```bash
# Edit configuration variables in your script
NODE_COUNTS=(8 16 32)  # Test different scales
TEST_TYPES=("allreduce")  # Focus on specific operations
SPLIT_MASKS=("0x0" "0x7")  # NCCL_TESTS_SPLIT_MASK value "Running multiple operations in parallel" for more info
APPS_PATH="/fsxl"  # Container location
```

**AMI Mode**: Modify `submit_nccl_test_ami.sh`:
```bash
# Edit configuration variables
NODE_COUNTS=(8 16 32)  # Test different scales
TEST_TYPES=("allreduce")  # Focus on specific operations
SPLIT_MASKS=("0x0" "0x7")  # NCCL_TESTS_SPLIT_MASK value "Running multiple operations in parallel" for more info
TOPO_SORTED_FILE="topo_sorted_hostnames.txt" or pass in empty string "" if you dont have a topologically sorted hostfile # see section "Topology-Aware Scheduling" for more info
```

## 3. Result Processing and Analysis

### Automated Result Processing

The `process_nccl_results.sh` script provides automated result processing:
**Features:**
- Automatic detection of Container vs AMI output formats
- CSV conversion with descriptive filenames
- Topology-aware result naming (adds "_topo" suffix when topology sorting is used)
- Comprehensive job status reporting
- Organized result storage in `nccl_results/` directory


```bash
# Process results from container tests (manual job tracking)
./process_nccl_results.sh your_job_ids_file.txt

# Process results from your submit_nccl_test_ami.sh output
./process_nccl_results.sh logs/submitted_jobs_ami_20250907_001101.txt
```

### Result File Naming Convention

Generated csv files with nccl test results are automatically organized with descriptive names:
```
nccl_results/
├── nccl_16_ami_allreduce_0x0_20250907_001101_results.csv
├── nccl_16_ami_allreduce_0x0_topo_20250907_001101_results.csv  # With topology
└── nccl_16_container_allgather_0x7_20250907_001101_results.csv
```

Format: `nccl_{nodes}_{container/ami}_{operation}_{pattern}[_topo]_{timestamp}_{type}.csv`

### Performance Output Format

NCCL tests output performance data from 8B to 17GB on p5en.48xlarge instances will be written to logs dir:

```txt
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    983.2    0.00    0.00      0    166.1    0.00    0.00      0
          16             4     float     sum      -1    167.3    0.00    0.00      0    171.2    0.00    0.00      0
        ...
  17179869184    4294967296     float     sum      -1    92173  186.39  369.86      0    92284  186.16  369.42      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 84.0569 
```

### Monitoring Jobs

```bash
# Monitor specific job output (container mode)
tail -f logs/nccl-tests-container_<job_id>.out

# View all submitted jobs 
cat logs/submitted_jobs_ami_<timestamp>.txt

```


## 4. Testing

### Test Suite for hostfile_topologify.py

A comprehensive test suite is available to validate the topology sorting functionality:

```bash
./run_unit_tests.sh
```

**Test Coverage:**
- **Topology-Based Ordering**: Validates that hosts are output in correct topology-aware order for optimal NCCL placement
- **Pagination with Ordering**: Tests pagination handling (>64 hosts) while maintaining topology-based grouping
- **Instance ID Processing**: Validates that the expected EC2 instance IDs are correctly processed from the mock topology API response
- **Empty File Handling**: Tests graceful handling of empty hostfiles
- **Hierarchical Network Validation**: Ensures hosts are grouped by network topology layers in contiguous blocks

**Mock Data Structure:**
The test suite uses realistic mock EC2 API responses that simulate a hierarchical network topology:
- **Small Test**: 4 instances (i-1example, i-2example, i-3example, i-4example)
- **Large Test**: 70 instances (i-1example through i-70example) to test pagination
- **Instance Type**: All instances use p5en.48xlarge for consistency
- **Network Hierarchy**:
  - **Level 1**: All instances share a single top-level network node (nn-1example)
  - **Level 2**: Instances alternate between two intermediate nodes (nn-2example, nn-3example)
  - **Level 3**: Each instance has a unique leaf network node (nn-4example, nn-5example, etc.)

**Topology Ordering Validation:**
Tests verify that the output maintains topology-aware ordering where:
- Hosts with the same Level 2 network node are grouped together
- Groups are contiguous (no interleaving between different topology groups)
- This ordering optimizes NCCL communication by placing nearby ranks on physically adjacent instances

**Test Files:**
- `test_hostfile_topologify.py`: Comprehensive test suite for topology-aware host ordering
  - `test_topology_based_ordering`: Validates 4-host topology sorting with order verification
  - `test_pagination_with_topology_ordering`: Tests 70-host pagination while maintaining topology groups
  - `test_empty_hostfile`: Handles empty input files gracefully
- `test_requirements.txt`: Test dependencies (pytest, boto3, botocore)
- `run_unit_tests.sh`: Convenience script to run tests with optional coverage reporting

## 5. File Reference

### Core Scripts

| File | Purpose |
|------|---------|
| `nccl-tests-ami.sbatch` | SLURM batch script for AMI-based execution |
| `nccl-tests-container.sbatch` | SLURM batch script for container-based execution |
| `submit_nccl_test_ami.sh` | Automated test suite submission script |
| `submit_nccl_test_container.sh` | Automated container test suite submission script |
| `process_nccl_results.sh` | Automated result processing and CSV conversion |
| `generate_hostfile.sh` | Generate a file with all the hosts in a cluster |
| `hostfile_topologify.py` | Generate sorted hostfile for topology optimization (supports --region parameter) |

### Test Files

| File | Purpose |
|------|---------|
| `test_hostfile_topologify.py` | Comprehensive test suite for topology-aware host ordering with order validation |
| `test_requirements.txt` | Test dependencies (pytest, boto3, botocore) |
| `run_unit_tests.sh` | Convenience script to run tests with optional coverage reporting |

### Output Directories

| Directory | Contents |
|-----------|----------|
| `logs/` | Job output files and submission tracking |
| `nccl_results/` | Processed CSV results and summaries |


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