# P5 GPU Validation — Results & Fixes (2026-05-31)

Validation run of the P5 (p5.48xlarge, 8×H100) node group on PCS cluster
`pcs_d3g5oj8y3j`, Slurm 25.11, 2 nodes / 16 GPUs, EFA 32-NIC. Capacity Block
`cr-0af82458b684f2c57` (us-east-2a).

## Stage results

| Stage | Method | Result |
|-------|--------|--------|
| 1. GPU sanity | `nvidia-smi` in a Pyxis container (`docker://nvidia/cuda:12.4.1-base-ubuntu22.04`) | ✅ 8× H100 80GB detected, container ran |
| 2a. NCCL (AMI-native) | DLAMI-bundled `all_reduce_perf` via `mpirun` (`01`/`02-nccl-tests-ami.sbatch`) | ✅ peak busbw **482 GB/s**, `#wrong 0` |
| 2b. NCCL (container) | Prebuilt `public.ecr.aws/hpc-cloud/nccl-tests` via Pyxis (`02-nccl-tests.sbatch`) | ✅ peak busbw **480 GB/s**, `#wrong 0` |
| 3. Megatron Llama-2 | (pending — needs HF tokenizer) | ⏳ |

NCCL logs confirmed EFA was actually used:
`NET/OFI Selected provider is efa, fabric is efa-direct (found 32 nics)`,
`Using transport protocol RDMA`, `Plugin selected platform: AWS`.

## Two ways to get the NCCL-tests binary

- **AMI-native (fastest, no build):** the PCS DLAMI ships prebuilt
  `all_reduce_perf` under `/usr/local/cuda-13.2/efa/test-cuda-13.2/` plus
  `/opt/amazon/ofi-nccl` + `/opt/amazon/efa`. Just `mpirun` it. See
  `02-nccl-tests-ami.sbatch`.
- **Prebuilt container (no local build):** import the published image, then run
  via Pyxis. The image is pulled through the local docker daemon (`dockerd://`),
  which resolves public ECR correctly:
  ```bash
  enroot import -o /fsx/validation/nccl-tests.sqsh dockerd://public.ecr.aws/hpc-cloud/nccl-tests
  ```
  Run the import on a node with a large root disk (see disk note below). Then
  `sbatch 02-nccl-tests.sbatch` (uses `--container-image=/fsx/validation/nccl-tests.sqsh`).

We did NOT build the container from source — building locally on the DLAMI was
slow and overflowed the small root disk; the published image avoids both.

## Fixes made during this run (folded into the templates/scripts)

### 1. Enroot/Pyxis did not install on GPU nodes
`scripts/install-enroot-pyxis.sh` aborted under `set -exo pipefail` inside the
`if nvidia-smi` GPU-only block, so **Enroot/Pyxis never installed on GPU nodes**
(CPU/login nodes skip that block and were fine). Two causes, both fixed (commit
`ad9f4c9`):
- `gpg --dearmor` failed `gpg: cannot open '/dev/tty'` with no tty →
  `gpg --batch --no-tty --yes --dearmor`.
- The per-distribution nvidia repo path (`.../ubuntu24.04/...`) 404'd and an HTML
  page got written into the apt source list → use the version-agnostic
  `https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list`
  with `curl -fsSL`.

After the fix, re-created GPU nodes show `post-install ... exit 0`, Pyxis
auto-installed, and `--container-image` works in batch jobs. Note: Pyxis
plugstack changes require a `slurmd` restart to take effect.

### 2. Root disk too small for container images
The PCS DLAMI root volume defaults to ~75 GiB, which overflows when pulling /
importing large images (NCCL-tests squashfs ~10 GB; Megatron ~20 GB). Added a
`RootVolumeSize` parameter (**default 300 GiB**) + `BlockDeviceMappings` to both
`add-cng.yaml` and `add-cng-p5.yaml`. Re-created GPU nodes came up with a 300 GiB
root and the container import then succeeded.

Caveat learned: FSx Lustre cannot host the overlayfs that `enroot import` mounts
(`failed to mount overlay: Invalid argument`). Import on a node with a large
local root disk (now 300 GiB) and write only the final `.sqsh` to `/fsx`.

## Access note

Iterate over an SSM-proxied SSH session (`ssh pcs-login`) rather than repeated
`aws ssm send-command`. Slurm binaries are only on PATH in a login shell, so wrap
commands: `ssh pcs-login 'bash -lc "sinfo; squeue"'`.
