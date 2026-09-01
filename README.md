# Distributed AI on AWS: Reference Architectures & Examples <!-- omit from toc -->

This repository contains reference architectures and examples for distributed AI training and inference on [Amazon SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html), [AWS ParallelCluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/what-is-aws-parallelcluster.html), [AWS Parallel Computing Service (PCS)](https://aws.amazon.com/pcs/), and [Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html). The examples cover different model families and sizes, training frameworks and parallel optimizations (PyTorch DDP/FSDP, Megatron-LM, NeMo), and serving engines (vLLM, SGLang, NVIDIA Dynamo).

The major components of this repository are:

```
├── architectures/               # Cluster reference architectures (CloudFormation, Terraform)
├── ami/                         # Scripts to create Amazon Machine Images (Packer/Ansible)
├── examples/                    # Runnable training, inference, and use-case examples
├── validation/                  # Environment and cluster health validation tools
├── observability/               # Monitoring, metrics exporters, and profiling
└── micro-benchmarks/            # Micro-benchmarks (NCCL, NCCOM, NVSHMEM, etc.)
```

## Workshops

You can follow the workshops below to train models on AWS. Each walks through several examples and shares practical guidance on operating a cluster for LLM training.

| Name                                                                               | Comments                                                        |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [AI on SageMaker HyperPod](https://awslabs.github.io/ai-on-sagemaker-hyperpod/)   | Deploying, operating, and monitoring SageMaker HyperPod clusters |
| [AWS ParallelCluster](https://catalog.workshops.aws/ml-on-aws-parallelcluster)     | The same journey on AWS ParallelCluster             |
| [AWS Parallel Computing Service](https://catalog.workshops.aws/ml-on-pcs)     | The same journey on AWS Parallel Computing Service             |

## Blog

Posts about distributed AI on AWS are published at <https://awslabs.github.io/awsome-distributed-ai/>. The Hugo source lives on the [`content`](https://github.com/awslabs/awsome-distributed-ai/tree/content) branch.

Blog content is editorially curated by AWS authors. Code samples in this repo (`architectures/`, `examples/`, etc.) accept external contributions as usual; see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Architectures

Each subdirectory under `architectures/` is a deployable cluster architecture or a shared building block.

| Name                                                                           | Category | Usage                                                |
| ------------------------------------------------------------------------------ | -------- | ---------------------------------------------------- |
| [`common`](./architectures/common)                                       | Storage  | Common resources (S3 bucket, event notifications)    |
| [`vpc_network`](./architectures/vpc_network)                             | Network  | Create a VPC with subnets and required resources     |
| [`aws-parallelcluster`](./architectures/aws-parallelcluster)             | Compute  | Cluster templates for GPU & custom silicon training  |
| [`amazon-eks`](./architectures/amazon-eks)                               | Compute  | Manifest files to train with Amazon EKS              |
| [`sagemaker-hyperpod-slurm`](./architectures/sagemaker-hyperpod-slurm)               | Compute  | SageMaker HyperPod with Slurm orchestration |
| [`ldap_server`](./architectures/ldap_server)                             | Identity | LDAP server for multi-user cluster access            |
| [`sagemaker-hyperpod-eks`](./architectures/sagemaker-hyperpod-eks)       | Compute  | SageMaker HyperPod with EKS orchestration            |
| [`accounting-database`](./architectures/accounting-database)             | Tooling  | Accounting database for job tracking                 |
| [`aws-pcs`](./architectures/aws-pcs)                                           | Compute  | AWS Parallel Computing Service templates with Slurm scheduler |

See [`docs/efa-cheatsheet.md`](./docs/efa-cheatsheet.md) for EFA tuning and the recommended environment variables.

## Custom Amazon Machine Images

Custom machine images can be built using [Packer](https://www.packer.io) for AWS ParallelCluster, Amazon EKS, and plain EC2. These images are based on Ansible roles and playbooks.

## Examples

Examples live under `examples/` and are organized along two axes:

- **`examples/training/`** and **`examples/inference/`** are *framework-centric*: the training or inference engine is the subject, and model variants underneath illustrate it (e.g. `training/fsdp/`, `training/megatron-lm/`, `training/nemo/`). Swapping the model gives "the same example with a different model."
- **`examples/use-cases/`** is *use-case-centric*: a specific model or task is the subject and the framework is incidental (e.g. `use-cases/detr-finetune/`, `use-cases/vjepa2/`). Swapping the framework would still leave a recognizable demo.

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

The top-level directory for each example contains a general introduction and environment setup (Dockerfiles, training scripts, configs), while subdirectories provide service-specific launch instructions.

Browse [`examples/`](./examples) to see the full list of frameworks, engines, and use cases.

## Validation and Observability

Environment and cluster health validation tools live under `validation/`; monitoring stacks, metrics exporters, and profiling guides live under `observability/`.

| Name                                                                                            | Comments                                                        |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [`pytorch-env-validation`](./validation/pytorch-env-validation)         | Validates your PyTorch environment                              |
| [`gpu-cluster-healthcheck`](./validation/gpu-cluster-healthcheck)       | GPU cluster health checks                                       |
| [`efa-node-exporter`](./observability/efa-node-exporter)                   | Node exporter with Amazon EFA monitoring modules                |
| [`prometheus-grafana`](./observability/prometheus-grafana)                  | Monitoring for SageMaker HyperPod and EKS GPU clusters          |
| [`nsight`](./observability/nsight)                                         | Shows how to run Nvidia Nsight Systems to profile your workload |

## Micro-benchmarks

Micro-benchmarks for evaluating network and communication performance are under `micro-benchmarks/`.

| Name                                                                  | Comments                                      |
| --------------------------------------------------------------------- | --------------------------------------------- |
| [`nccl-tests`](./micro-benchmarks/nccl-tests)                         | NCCL collective communication benchmarks      |
| [`nccom-tests`](./micro-benchmarks/nccom-tests)                       | NCCOM communication benchmarks                |
| [`nvshmem`](./micro-benchmarks/nvshmem)                               | NVSHMEM benchmarks                            |
| [`expert-parallelism`](./micro-benchmarks/expert-parallelism)         | Expert parallelism (MoE) benchmarks           |

## Contributors

Thanks to all the contributors for building, reviewing and testing.

[![Contributors](https://contrib.rocks/image?repo=awslabs/awsome-distributed-ai)](https://github.com/awslabs/awsome-distributed-ai/graphs/contributors)

## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=awslabs/awsome-distributed-ai&type=Date)](https://star-history.dera.page/#awslabs/awsome-distributed-ai&Date)
