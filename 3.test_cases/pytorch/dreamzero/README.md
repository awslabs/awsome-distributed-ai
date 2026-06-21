<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DreamZero LIBERO 14B SFT (World-Action Model) on Amazon EKS

This test case is a working example of **deploying DreamZero training on Amazon
EKS**. DreamZero is a **14B-parameter World-Action Model (WAM)** — a Wan-based,
causal (autoregressive) video diffusion transformer that *jointly* denoises
future video frames and robot actions via flow matching. Here it continues
supervised fine-tuning from the released
[`GEAR-Dreams/DreamZero-DROID`](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
14B checkpoint on the **LIBERO** manipulation benchmark and evaluates it in the
LIBERO simulator — exercising multi-node **FSDP2** training over **EFA** with
**KubeRay**. The focus is the **EKS deployment mechanics** (image build,
multi-node EFA/NCCL, FSDP2 sharded checkpointing, sim eval), not a training-quality
or transfer result. Note that LIBERO is a **simulation** of the same Franka Emika
Panda arm that DROID captures in the **real world**, so warm-starting the DROID
checkpoint onto LIBERO must bridge a real→sim visual domain gap. LIBERO is used
here as a convenient public dataset + simulator to exercise the pipeline
end-to-end; in practice you would swap it for your own dataset (real or simulated)
for task-specific or cross-embodiment fine-tuning.

Upstream:

- [github.com/RLinf/RLinf](https://github.com/RLinf/RLinf) (training framework)
- [github.com/RLinf/dreamzero](https://github.com/RLinf/dreamzero) (the `groot` WAM
model code).

## Architecture

![DreamZero WAM](diagrams/dreamzero-wam.drawio.svg)

For the full breakdown — the KubeRay/FSDP2 training topology and a component-level
walkthrough of the World-Action Model — see
[**Architecture** in the walkthrough](kubernetes/libero/README.md#architecture).

## Layout

```
dreamzero/
├── Dockerfile                 # two-stage RLinf + EFA overlay image
├── dcp-save-gloo-coordinator.patch  # DCP checkpoint fix, applied to RLinf at build time
├── diagrams/                  # WAM + infra topology (draw.io + SVG)
└── kubernetes/libero/         # the EKS recipe (RayJob SFT + eval) — see its README
```

## Hardware

**2× `p5en.48xlarge`** (8× NVIDIA H200 each = **16 GPUs total**), **16 EFA NICs per
node**, and **FSx for Lustre with ≥250 GB free** (a 14B FSDP DCP checkpoint is
~140–206 GB).

## Prerequisites

An Amazon EKS cluster that can schedule **2× `p5en.48xlarge`** (8× H200 + EFA
each) — provisioned however you like (a static managed node group, a Capacity
Block reservation, or on-demand autoscaling such as Karpenter; the workload is
fixed-size, so autoscaling is a convenience, not a requirement) — plus EFA
networking, the KubeRay operator, and FSx for Lustre shared storage. See
[`Amazon EKS distributed training architecture`](../../../1.architectures/4.amazon-eks/README.md) for
cluster setup. The detailed prerequisite checklist lives in the walkthrough below.

## Full walkthrough

**Full step-by-step walkthrough: [`kubernetes/libero/README.md`](kubernetes/libero/README.md)** —
build-image → push-to-ECR → stage models/dataset → generate metadata → multi-node
SFT RayJob → DCP→`.pt` conversion → LIBERO simulator eval.

## Results / validation status

**Infrastructure & pipeline** — validated **end-to-end** on 2× `p5en.48xlarge`:
the KubeRay RayJob reached `SUCCEEDED`, a sharded FSDP2 DCP checkpoint was
written, converted to a single `.pt` on CPU, and consumed by the LIBERO
simulator eval (image build → multi-node EFA/NCCL → FSDP2 sharded checkpointing →
DCP→`.pt` conversion → in-sim eval).

**Training convergence** — a **300-step** SFT run on 2× `p5en.48xlarge` (16×
H200, FSDP2 `full_shard` over EFA, ~6.9 s/step) reduced `train/loss` from
**0.232 → 0.085** with a clean, monotonic-ish curve (`action_loss` and
`dynamics_loss` both ~0.04–0.05 at the end). It wrote a **207 GB** DCP checkpoint
(16 shards + `.metadata`) with **zero** `UnpicklingError` — exercising the
in-image `dcp-save-gloo-coordinator.patch` at the full 16.48B-parameter scale.
This demonstrates the pipeline *trains and converges*, not a converged policy:
300 steps is a short run (the released 14B checkpoint trained for 100K), so it is
**not** a task-accuracy claim.

**Accuracy reference** — a 1-step checkpoint yields `eval/success_once = 0.0`
(expected — it validates the eval machinery, not competence). For real accuracy,
train longer; RLinf's own LIBERO-Spatial SFT of the **5B** WAM
([`RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000`](https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000))
reaches **~96.7% `success_once` by step 18000** ([RLinf docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_dreamzero.html)),
confirming the recipe converges with sufficient steps.

The image is built with the **local `docker buildx`** two-stage build and pushed
to ECR (see [`kubernetes/libero/build-push.sh`](kubernetes/libero/build-push.sh)).

## References

- RLinf training framework — [github.com/RLinf/RLinf](https://github.com/RLinf/RLinf)
- DreamZero (`groot`) model code — [github.com/RLinf/dreamzero](https://github.com/RLinf/dreamzero)
- DreamZero-DROID checkpoint — [huggingface.co/GEAR-Dreams/DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
- EKS cluster architectures — [`1.architectures/4.amazon-eks`](../../../1.architectures/4.amazon-eks)

## Security

See [CONTRIBUTING](https://github.com/aws-samples/awsome-distributed-training/blob/main/CONTRIBUTING.md#security-issue-notifications)
for more information. Credentials (Hugging Face tokens, etc.) flow through
Kubernetes Secrets, never committed to rendered YAML.

## License

This project is licensed under the MIT-0 License. See the LICENSE file.
