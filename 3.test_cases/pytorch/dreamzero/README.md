<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

# DreamZero LIBERO 14B SFT (World-Action Model) on Amazon EKS

DreamZero is a **16.48B-parameter World-Action Model (WAM)** — a Wan-based
video-diffusion Diffusion Transformer that *jointly* denoises future video frames
and robot actions in a shared causal self-attention space. This test case
continues supervised fine-tuning from the released
[`GEAR-Dreams/DreamZero-DROID`](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
14B checkpoint onto a *new* embodiment (LIBERO) and evaluates it in the LIBERO
simulator on Amazon EKS — demonstrating cross-embodiment transfer and multi-node
**FSDP2** training over **EFA** with **KubeRay**. Upstream:
[github.com/RLinf/RLinf](https://github.com/RLinf/RLinf) (training framework) and
[github.com/RLinf/dreamzero](https://github.com/RLinf/dreamzero) (the `groot` WAM
model code).

## Architecture

![DreamZero WAM](diagrams/dreamzero-wam.drawio.svg)

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

An Amazon EKS cluster with GPU autoscaling (Karpenter), EFA networking, the
KubeRay operator, and FSx for Lustre shared storage. See
[`../../1.architectures/4.amazon-eks`](../../1.architectures/4.amazon-eks) for
cluster setup. The detailed prerequisite checklist lives in the walkthrough below.

## Full walkthrough

**Full step-by-step walkthrough: [`kubernetes/libero/README.md`](kubernetes/libero/README.md)** —
build-image → push-to-ECR → stage models/dataset → generate metadata → multi-node
SFT RayJob → DCP→`.pt` conversion → LIBERO simulator eval.

## Results / validation status

The pipeline was validated **end-to-end** with a **1-step SFT smoke run** on 2×
`p5en.48xlarge`: the KubeRay RayJob reached `SUCCEEDED`, a **209 GB sharded FSDP
DCP checkpoint** was written, and it was converted to a single **91.7 GB `.pt`**
on CPU and consumed by the LIBERO simulator eval. This proves the *infrastructure
and pipeline* (image build, multi-node EFA/NCCL, FSDP2 sharded checkpointing,
DCP→`.pt` conversion, and in-sim eval) — **not** task accuracy. A 1-step
checkpoint yields `eval/success_once = 0.0`, which is **expected**; real accuracy
requires a multi-step SFT run (raise `runner.max_steps`).

The image is built with the **local `docker buildx`** two-stage build and pushed
to ECR (see [`kubernetes/libero/setup/build-push.sh`](kubernetes/libero/setup/build-push.sh)).

## References

- RLinf training framework — [github.com/RLinf/RLinf](https://github.com/RLinf/RLinf)
- DreamZero (`groot`) model code — [github.com/RLinf/dreamzero](https://github.com/RLinf/dreamzero)
- DreamZero-DROID checkpoint — [huggingface.co/GEAR-Dreams/DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
- EKS cluster architectures — [`1.architectures/4.amazon-eks`](../../1.architectures/4.amazon-eks)

## Security

See [CONTRIBUTING](https://github.com/aws-samples/awsome-distributed-training/blob/main/CONTRIBUTING.md#security-issue-notifications)
for more information. Credentials (Hugging Face tokens, etc.) flow through
Kubernetes Secrets, never committed to rendered YAML.

## License

This project is licensed under the MIT-0 License. See the LICENSE file.
