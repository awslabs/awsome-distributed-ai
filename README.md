# ML Training Reference Architectures & Tests <!-- omit from toc -->

This repository contains reference architectures and test cases for distributed model training with [Amazon SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html), [AWS ParallelCluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/what-is-aws-parallelcluster.html), [AWS Parallel Computing Service (PCS)](https://aws.amazon.com/pcs/), [AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/what-is-batch.html), and [Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html). The test cases cover different types and sizes of models as well as different frameworks and parallel optimizations (PyTorch DDP/FSDP, Megatron-LM, NeMo...).

The major components of this directory are:

```
├── architectures/               # CloudFormation templates for reference architectures
├── ami_and_containers/          # Scripts to create AMIs and container images
├── examples/                  # Reference test cases and/or benchmark scripts
├── validation_and_observability/# Tools to measure performance or troubleshoot
└── micro-benchmarks/              # Micro-benchmarks (NCCL, NCCOM, NVSHMEM, etc.)
```

**NOTE**: The architectures are designed to work with the S3 bucket and VPC created using reference templates `architectures/common/` and `architectures/vpc_network/`. _You're strongly recommended to deploy these two templates **before** deploying any of the reference architectures._

## 0. Workshops

You can follow the workshops below to train models on AWS. Each contains examples for several test cases as well as nuggets of information on operating a cluster for LLM training.

| Name                                                                               | Comments                                                        |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [AI on SageMaker HyperPod](https://awslabs.github.io/ai-on-sagemaker-hyperpod/)   | Workshop for SageMaker HyperPod, shows how to deploy and monitor it |
| [AWS ParallelCluster](https://catalog.workshops.aws/ml-on-aws-parallelcluster)     | Similar workshop as HyperPod but on ParallelCluster             |
| [AWS Parallel Computing Service](https://catalog.workshops.aws/ml-on-pcs)     | Similar workshop as HyperPod but on AWS Parallel Computing Service             |

## Blog

Posts about distributed ML training on AWS are published at <https://awslabs.github.io/awsome-distributed/>. The Hugo source lives on the [`content`](https://github.com/awslabs/awsome-distributed/tree/content) branch.

Blog content is editorially curated by AWS authors. Code samples in this repo (`architectures/`, `examples/`, etc.) accept external contributions as usual — see [CONTRIBUTING.md](./CONTRIBUTING.md).

## 1. Architectures

Architectures are located in `architectures` and consist of utilities and service-related architectures.

| Name                                                                           | Category | Usage                                                |
| ------------------------------------------------------------------------------ | -------- | ---------------------------------------------------- |
| [`common`](./architectures/common)                                       | Storage  | Common resources (S3 bucket, event notifications)    |
| [`vpc_network`](./architectures/vpc_network)                             | Network  | Create a VPC with subnets and required resources     |
| [`aws-parallelcluster`](./architectures/aws-parallelcluster)             | Compute  | Cluster templates for GPU & custom silicon training  |
| [`aws-batch`](./architectures/aws-batch)                                 | Compute  | AWS Batch template for distributed training          |
| [`amazon-eks`](./architectures/amazon-eks)                               | Compute  | Manifest files to train with Amazon EKS              |
| [`sagemaker-hyperpod-slurm`](./architectures/sagemaker-hyperpod-slurm)               | Compute  | SageMaker HyperPod template for distributed training |
| [`ldap_server`](./architectures/ldap_server)                             | Identity | LDAP server for multi-user cluster access            |
| [`sagemaker-hyperpod-eks`](./architectures/sagemaker-hyperpod-eks)       | Compute  | SageMaker HyperPod with EKS orchestration            |
| [`accounting-database`](./architectures/accounting-database)             | Tooling  | Accounting database for job tracking                 |
| [`aws-pcs`](./architectures/aws-pcs)                                           | Compute  | AWS Parallel Computing Service templates with Slurm scheduler |

You will also find [documentation](./architectures/efa-cheatsheet.md) for EFA and the recommended environment variables.

## 2. Custom Amazon Machine Images

Custom machine images can be built using [Packer](https://www.packer.io) for AWS ParallelCluster, Amazon EKS and plain EC2. These images are based on Ansible roles and playbooks.

## 3. Examples

Examples live under `examples/` and are organized along two axes:

- **`examples/training/`** and **`examples/inference/`** — *framework-centric*. The training or inference engine is the subject, and model variants underneath illustrate it (e.g. `training/fsdp/`, `training/megatron-lm/`, `training/nemo/`). Swapping the model gives "the same example with a different model."
- **`examples/use-cases/`** — *use-case-centric*. A specific model or task is the subject and the framework is incidental (e.g. `use-cases/detr-finetune/`, `use-cases/vjepa2/`). Swapping the framework would still leave a recognizable demo.

Each example follows this general structure:

```
examples/
├── training/                   # framework-centric training/fine-tuning engines
│   └── <framework>/            # e.g. fsdp, deepspeed, megatron-lm, nemo, trl
│       └── <model>/            # e.g. llama3 (may be omitted for single-model cases)
│           ├── Dockerfile      # Container / environment setup
│           ├── README.md
│           ├── slurm/          # Slurm-specific launch scripts
│           └── kubernetes/     # Kubernetes manifests
├── inference/                  # framework-centric inference engines (vllm, …)
└── use-cases/                  # use-case-centric end-to-end demos
    └── <name>/                 # e.g. detr-finetune, esm2-hyperpod
```

The top-level directory for each example contains general introduction and environment setup (Dockerfiles, training scripts, configs), while subdirectories provide service-specific launch instructions.

Browse [`examples/`](./examples) to see the full list of frameworks, engines, and use cases.

## 4. Validation and Observability

Utility scripts and tools for validating your environment and monitoring training jobs are under `validation_and_observability/`.

| Name                                                                                            | Comments                                                        |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [`pytorch-env-validation`](./validation_and_observability/pytorch-env-validation)         | Validates your PyTorch environment                              |
| [`gpu-cluster-healthcheck`](./validation_and_observability/gpu-cluster-healthcheck)       | GPU cluster health checks                                       |
| [`efa-node-exporter`](./validation_and_observability/efa-node-exporter)                   | Node exporter with Amazon EFA monitoring modules                |
| [`prometheus-grafana`](./validation_and_observability/prometheus-grafana)                  | Monitoring for SageMaker HyperPod and EKS GPU clusters          |
| [`nsight`](./validation_and_observability/nsight)                                         | Shows how to run Nvidia Nsight Systems to profile your workload |

## 5. Micro-benchmarks

Micro-benchmarks for evaluating network and communication performance are under `micro-benchmarks/`.

| Name                                                                  | Comments                                      |
| --------------------------------------------------------------------- | --------------------------------------------- |
| [`nccl-tests`](./micro-benchmarks/nccl-tests)                         | NCCL collective communication benchmarks      |
| [`nccom-tests`](./micro-benchmarks/nccom-tests)                       | NCCOM communication benchmarks                |
| [`nvshmem`](./micro-benchmarks/nvshmem)                               | NVSHMEM benchmarks                            |
| [`expert-parallelism`](./micro-benchmarks/expert-parallelism)         | Expert parallelism (MoE) benchmarks           |

## 6. Contributors

Thanks to all the contributors for building, reviewing and testing.

[![Contributors](https://contrib.rocks/image?repo=awslabs/awsome-distributed)](https://github.com/awslabs/awsome-distributed/graphs/contributors)

## 7. Star History

[![Star History Chart](https://api.star-history.com/svg?repos=awslabs/awsome-distributed&type=Date)](https://star-history.com/#awslabs/awsome-distributed&Date)
