# PCS-Ready DLAMI version history

AWS publishes the **PCS-Ready DLAMI Base GPU AMI (Ubuntu 24.04)** as the default
compute-node image for AWS PCS. Every deploy-all / add-cng template in this repo
resolves it through the public SSM parameter
`/aws/service/pcs/ami/dlami-base-ubuntu2404/x86_64/latest/ami-id` when `AmiId`
is left empty.

The image is a two-layer stack:

1. **Base DLAMI** — Ubuntu 24.04, kernel, NVIDIA driver, CUDA stack, EFA, DCGM,
   containerd, OFI-NCCL. Maintained on the DLAMI release cadence.
2. **PCS layer** — PCS Agent, one or more Slurm builds, EFS utils. Added on top
   by the PCS team.

The AMI `Description` embeds every PCS-layer version and the base DLAMI build
date, for example:

```
PCS-Ready DLAMI based on Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20260724.
PCS Agent: 1.5.0-1. Slurm: 24.11.7-2, 25.05.8-2, 25.11.6-2. EFS Utils: 2.4.2
```

New builds are announced through the SNS topic
`arn:aws:sns:us-west-2:265886551188:pcs-ready-dlami-release-notifications`
(see the [PCS-Ready DLAMI user guide](https://docs.aws.amazon.com/pcs/latest/userguide/working-with_ami_pcs-ready-dlami.html)).

## PCS-Ready DLAMI x86\_64

Snapshot (as of 2026-08-05) of the published x86\_64 PCS-Ready DLAMIs, newest
first. AMI IDs are region-scoped and shown for **us-east-2** — the SSM
parameter above resolves to the newest build in each region, so leaving
`AmiId` empty in the templates is the region-portable path; refer here only
when pinning to a specific build.

Base-DLAMI component versions (kernel, NVIDIA driver, CUDA, DCGM, EFA,
OFI-NCCL, containerd) come from the DLAMI release notes for the build named in
the "Base DLAMI" column. Intermediate base builds without their own release
notes inherit values from the nearest previous published build.

| PCS build | AMI ID (us-east-2) | Base DLAMI | PCS Agent | Slurm | EFS Utils | Kernel | NVIDIA driver | Default CUDA | CUDA stack | DCGM | EFA | OFI-NCCL | containerd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-27 | `ami-0631f0a4211880f98` | 20260724 | 1.5.0-1 | 24.11.7-2, 25.05.8-2, 25.11.6-2 | 2.4.2 | 6.17.0-1019 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.6.0 | 1.47.0 | 1.18.0 | v2.2.6 |
| 2026-07-22 | `ami-045669174d015d92c` | 20260721 | 1.5.0-1 | 24.11.7-2, 25.05.8-2, 25.11.6-2 | 2.4.2 | 6.17.0-1019 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.6.0 | 1.47.0 | 1.18.0 | v2.2.6 |
| 2026-07-20 | `ami-0c91bfa1d3999dfea` | 20260717 | 1.5.0-1 | 24.11.7-2, 25.05.8-2, 25.11.6-2 | 2.4.2 | 6.17.0-1019 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.6.0 | 1.47.0 | 1.18.0 | v2.2.6 |
| 2026-07-13 | `ami-0c1b2f4317de42125` | 20260710 | 1.4.0-1 | 24.11.7-1, 25.05.8-2, 25.11.6-2 | 2.4.2 | 6.17.0-1019 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.6.0 | 1.47.0 | 1.18.0 | v2.2.5 |
| 2026-07-08 | `ami-01b64e972d350b317` | 20260707 | 1.4.0-1 | 24.11.7-1, 25.05.8-2, 25.11.6-2 | 2.4.2 | 6.17.0-1019 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.6.0 | 1.47.0 | 1.18.0 | v2.2.5 |
| 2026-07-06 | `ami-04e4916942b70bf64` | 20260703 | 1.4.0-1 | 24.11.7-1, 25.05.8-2, 25.11.6-2 | 2.4.2 | 6.17.0-1019 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.6.0 | 1.47.0 | 1.18.0 | v2.2.5 |
| 2026-06-29 | `ami-03ef51e588e1864db` | 20260626 | 1.4.0-1 | 24.11.7-1, 25.05.7-1, 25.11.2-1 | 2.4.2 | 6.17.0-1019 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.6.0 | 1.47.0 | 1.18.0 | v2.2.5 |
| 2026-06-22 | `ami-039d4de58ea181e3d` | 20260619 | 1.4.0-1 | 24.11.7-1, 25.05.7-1, 25.11.2-1 | 2.4.2 | 6.17.0-1017 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.5.3 | 1.47.0 | 1.18.0 | v2.2.4 |
| 2026-06-09 | `ami-08a86ee651999836e` | 20260609 | 1.4.0-1 | 24.11.7-1, 25.05.7-1, 25.11.2-1 | 2.4.2 | 6.17.0-1017 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.5.3 | 1.47.0 | 1.18.0 | v2.2.4 |
| 2026-06-07 | `ami-08ab1cf913f168f31` | 20260605 | 1.4.0-1 | 24.11.7-1, 25.05.7-1, 25.11.2-1 | 2.4.2 | 6.17.0-1017 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.5.3 | 1.47.0 | 1.18.0 | v2.2.4 |
| 2026-06-02 | `ami-0da9c0741ad44f950` | 20260602 | 1.4.0-1 | 24.11.7-1, 25.05.7-1, 25.11.2-1 | 2.4.2 | 6.17.0-1015 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.5.3 | 1.47.0 | 1.18.0 | v2.2.4 |
| 2026-05-30 | `ami-0ddf60cecde4935de` | 20260529 | 1.4.0-1 | 24.11.7-1, 25.05.7-1, 25.11.2-1 | 2.4.2 | 6.17.0-1015 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.5.3 | 1.47.0 | 1.18.0 | v2.2.4 |
| 2026-05-23 | `ami-0b0605bcf9ffffa63` | 20260522 | 1.4.0-1 | 24.11.7-1, 25.05.7-1, 25.11.2-1 | 2.4.2 | 6.17.0-1015 | 595.71.05 | 13.2 | 12.8, 12.9, 13.0, 13.2 | 4.5.3 | 1.47.0 | 1.18.0 | v2.2.4 |

Full base DLAMI release notes:
<https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html>
