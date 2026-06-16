# AWS PCS — Test & Validation Guide

Test procedures for validating an AWS PCS cluster deployed from the templates
in [`../assets`](../assets).

For operational guidance (Slurm version trade-offs, AMI pinning, monitoring,
FSx tuning), see [`../docs/OPERATIONS.md`](../docs/OPERATIONS.md).

---

## Pre-merge test matrix

Run this **complete set** before merging template or script changes.

| # | Category | Tests | File | When to run |
|---|---|---|---|---|
| 1-3, 8 | **Infrastructure** | Monitoring stack, container runtime (first-boot + AMI build), template lint | [`infra-test.md`](./infra-test.md) | Every PR |
| 4-6 | **Compute** | CPU queue, GPU families (G/P5/P6), NCCL multi-node EFA | [`compute-test.md`](./compute-test.md) | Every PR |
| 7 | **Training** | FSDP Llama-2 7B distributed training | [`training-test.md`](./training-test.md) | GPU PRs |
| 9 | **HPC EFA** | EFA on CPU instances (hpc6a/hpc7a/hpc8a), OSU benchmarks | [`hpc-efa-test.md`](./hpc-efa-test.md) | EFA wiring changes |
| 10 | **Storage** | FSx health check + performance regression test (noatime benchmark) | [`storage-test.md`](./storage-test.md) | FSx / mount changes |
| 11-12 | **Multi-user** | OpenLDAP directory + Slurm managed accounting | [`multi-user-test.md`](./multi-user-test.md) | Directory / accounting changes |
| 13 | **GPU health** | GPU Cluster Health Check suite (DCGM, EFA, NVLink, NCCL thresholds) | [`gpu-healthcheck-test.md`](./gpu-healthcheck-test.md) | GPU CNG deploys |

---

## Quick-start: single-cluster shortcut

A single `pcs-ml-cluster-deploy-all.yaml` deploy with `DeployMonitoring=true`,
`DeployOnDemandCNG=true`, and `DeployPseriesCNG=true` exercises Tests 1–7 in
one cluster. Tests 8-13 are separate paths run only when their inputs change.

---

## Canonical assets reused (not duplicated)

| Workload | Source in this repo | PCS-specific delta |
|---|---|---|
| NCCL `all_reduce` | [`micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch`](../../../micro-benchmarks/nccl-tests) | Partition name, Enroot import on login node |
| FSDP Llama-2 7B | [`3.test_cases/pytorch/FSDP`](../../../3.test_cases/pytorch/FSDP) | Cache on `/fsx`, 2 nodes |
| GPU Health Check | [`4.validation_and_observability/2.gpu-cluster-healthcheck`](../../../4.validation_and_observability/2.gpu-cluster-healthcheck) | sbatch wrapper, partition name |

---

## Notes

- All Slurm commands run as `ubuntu` from the login node (SSM or SSH)
- Slurm binaries: `export PATH=/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH`
- Template lint: `aws cloudformation validate-template --template-body file://assets/<name>.yaml`
- Regression criteria (storage): >10% degradation blocks the change

---

## Cleanup

Delete the stack from the CloudFormation console (select → **Delete**) or:

```bash
aws cloudformation delete-stack --stack-name <stack-name>
aws cloudformation wait stack-delete-complete --stack-name <stack-name>
```

Nested stacks (and FSx) are deleted automatically — back up FSx data first.

If DELETE_FAILED on CNG stacks (PCS timing dependency), delete PCS CNGs first:
```bash
CLUSTER_ID=<id>
for cng in $(aws pcs list-compute-node-groups --cluster-identifier $CLUSTER_ID --query 'computeNodeGroups[].id' --output text); do
  aws pcs delete-compute-node-group --cluster-identifier $CLUSTER_ID --compute-node-group-identifier $cng
done
sleep 60
aws cloudformation delete-stack --stack-name <stack-name>
```
