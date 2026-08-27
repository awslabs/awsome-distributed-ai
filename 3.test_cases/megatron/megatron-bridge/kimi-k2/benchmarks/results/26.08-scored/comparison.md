# Kimi-K2 EP arm training-output comparison

Overall status: `PASS`.

The absolute loss tolerance is 0.001114845276 dimensionless and comes from the preserved NCCL BF16 self-repeat envelope.
Gradient-norm and sampled update-L2 bounds are derived without a fitted multiplier as `elementwise NCCL BF16 tolerance * sqrt(sampled elements)`.
For performance runs without full-precision optimizer samples, the rounded gradient norms from iteration logs are plotted as diagnostics and are not used as a numeric gate.

## throughput-no-overlap, repeat 1 (dimensionless)

Group status: `PASS`.

| Arm | Iterations, dimensionless | First loss, dimensionless | Last loss, dimensionless | Maximum absolute loss delta vs NCCL, dimensionless | Maximum gradient-norm delta / bound, parameter-gradient units | Maximum sampled update-L2 delta / bound, parameter units | Loss gate | Gradient gate | Update gate | Route hash gate |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `nccl-alltoall` | 40 | 12.03574 | 0.03982611 | Reference | Reference | Reference | Reference | Reference | Reference | Reference |
| `uccl` | 40 | 12.03581 | 0.03982973 | 0.000217 | 0.001 / Not gated | Not recorded | PASS | Not required | Not required | PASS |
| `deepep-v1-nvshmem` | 40 | 12.03581 | 0.03984408 | 0.000125 | 0.001 / Not gated | Not recorded | PASS | Not required | Not required | PASS |
| `deepep-v2-gin-gda` | 40 | 12.0358 | 0.03983385 | 8.2e-05 | 0.001 / Not gated | Not recorded | PASS | Not required | Not required | PASS |

## throughput-no-overlap, repeat 2 (dimensionless)

Group status: `PASS`.

| Arm | Iterations, dimensionless | First loss, dimensionless | Last loss, dimensionless | Maximum absolute loss delta vs NCCL, dimensionless | Maximum gradient-norm delta / bound, parameter-gradient units | Maximum sampled update-L2 delta / bound, parameter units | Loss gate | Gradient gate | Update gate | Route hash gate |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `nccl-alltoall` | 40 | 12.03574 | 0.03982207 | Reference | Reference | Reference | Reference | Reference | Reference | Reference |
| `uccl` | 40 | 12.03581 | 0.03984191 | 0.000223 | 0.001 / Not gated | Not recorded | PASS | Not required | Not required | PASS |
| `deepep-v1-nvshmem` | 40 | 12.03581 | 0.03983857 | 0.000106 | 0.001 / Not gated | Not recorded | PASS | Not required | Not required | PASS |
| `deepep-v2-gin-gda` | 40 | 12.0358 | 0.03983185 | 0.000153 | 0.001 / Not gated | Not recorded | PASS | Not required | Not required | PASS |

## throughput-no-overlap, repeat 3 (dimensionless)

Group status: `PASS`.

| Arm | Iterations, dimensionless | First loss, dimensionless | Last loss, dimensionless | Maximum absolute loss delta vs NCCL, dimensionless | Maximum gradient-norm delta / bound, parameter-gradient units | Maximum sampled update-L2 delta / bound, parameter units | Loss gate | Gradient gate | Update gate | Route hash gate |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `nccl-alltoall` | 40 | 12.03574 | 0.03983534 | Reference | Reference | Reference | Reference | Reference | Reference | Reference |
| `uccl` | 40 | 12.03581 | 0.03983251 | 0.000129 | 0.001 / Not gated | Not recorded | PASS | Not required | Not required | PASS |
| `deepep-v1-nvshmem` | 40 | 12.03581 | 0.03983726 | 0.000122 | 0.001 / Not gated | Not recorded | PASS | Not required | Not required | PASS |
| `deepep-v2-gin-gda` | 40 | 12.0358 | 0.03983143 | 0.000154 | 0.001 / Not gated | Not recorded | PASS | Not required | Not required | PASS |

![Loss curves for throughput-no-overlap](loss-curves-throughput-no-overlap.png)

![Training-output curves for throughput-no-overlap](training-output-curves-throughput-no-overlap.png)

![Training-output deltas for throughput-no-overlap](training-output-deltas-throughput-no-overlap.png)
