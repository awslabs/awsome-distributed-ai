# Managing Neuron SDK Versions on HyperPod Trainium Clusters

This guide covers how Neuron SDK versions are managed on HyperPod clusters
with Trainium (trn1) or Inferentia (inf2) instances, and how to pin a
specific SDK version when your workload requires it.

## How the Neuron SDK is delivered on HyperPod

HyperPod clusters launch from the [Deep Learning AMI (DLAMI)](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-release-ami-slurm.html),
which ships with the Neuron SDK pre-installed. The AMI includes:

- **Host-level packages** (managed via `apt`): `aws-neuronx-dkms`,
  `aws-neuronx-runtime-lib`, `aws-neuronx-collectives`, `aws-neuronx-tools`
- **Pre-built Python virtual environments** under `/opt/` with
  `torch-neuronx`, `neuronx-cc`, `neuronx-distributed`, and other
  userspace libraries

When you run
[`update-cluster-software`](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-operate.html#sagemaker-hyperpod-operate-cli-command-update-cluster-software),
HyperPod replaces the root volume with the latest DLAMI and re-runs
your lifecycle scripts. This updates both the host-level Neuron packages
and the pre-built environments to the version shipped with the new AMI.

## Pinning a specific SDK version

If your workload requires a specific Neuron SDK version (for example,
to match a tested configuration or to avoid a known regression), pin the
**userspace packages** in a Python virtual environment. Do not attempt
to downgrade the host-level DKMS driver.

### Host-level driver vs. userspace packages

| Layer | Examples | Managed by | Can you pin? |
|-------|----------|------------|--------------|
| **Host driver** | `aws-neuronx-dkms`, `aws-neuronx-runtime-lib`, `aws-neuronx-collectives` | AMI / `update-cluster-software` | No — use the AMI version |
| **Userspace** | `neuronx-cc`, `torch-neuronx`, `neuronx-distributed`, `transformers` | `pip` in a Python venv | **Yes** — pin in a venv |

The Neuron host driver is forward-compatible with older userspace
packages. For example, a host running DKMS 2.26.5.0 (SDK 2.28) works
correctly with `neuronx-cc==2.23.6484.0` (SDK 2.27) installed in a venv.

### Creating a pinned environment

Create a virtual environment on shared storage (e.g., FSx for Lustre)
so all compute nodes can access it:

```bash
# Create a venv on shared storage
python3.10 -m venv /fsx/envs/my-neuron-env
source /fsx/envs/my-neuron-env/bin/activate

# Install specific Neuron SDK userspace packages
pip install neuronx-cc==2.23.6484.0
pip install torch-neuronx==2.8.0.2.12.22436
pip install neuronx-distributed==0.17.26814

# Install your framework dependencies
pip install transformers accelerate
```

> [!TIP]
> Use a `requirements.txt` file to make pinned versions reproducible
> across environments and team members.

### Verifying the environment

After creating the venv, verify the installed versions match your
expectations:

```bash
source /fsx/envs/my-neuron-env/bin/activate
pip list | grep neuron
```

To check the host-level driver version on a compute node:

```bash
apt list --installed 2>/dev/null | grep neuronx-dkms
```

### Using the pinned environment in Slurm jobs

Reference the venv in your Slurm batch scripts:

```bash
#!/bin/bash
#SBATCH --job-name=my-neuron-job
#SBATCH --nodes=1

source /fsx/envs/my-neuron-env/bin/activate
python train.py
```

## Finding available SDK versions

Each Neuron SDK release maps to specific package versions. To find the
versions for a given release:

- **Release notes**: [AWS Neuron Release Notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html)
- **PyPI**: Search for `neuronx-cc`, `torch-neuronx`, etc. on
  [pypi.org](https://pypi.org) to see all published versions
- **Neuron pip repo**: `https://pip.repos.neuron.amazonaws.com`

## What changed (and why)

Previously, these lifecycle scripts included an `update_neuron_sdk.sh`
script that used `apt-get` to replace the host-level Neuron packages
with a hardcoded older version (SDK 2.21.0). This script was removed
because:

1. **It downgraded the SDK.** The AMI ships a newer SDK than the script
   installed, so running it replaced newer packages with older ones.
2. **Host-level pinning is fragile.** Replacing DKMS drivers via
   `apt-get` can break the tested AMI configuration and is undone by
   `update-cluster-software`.
3. **Userspace pinning is the correct approach.** Workloads that need a
   specific SDK version should pin userspace packages in a venv, which
   is isolated, reproducible, and forward-compatible with newer host
   drivers.

The `enable_update_neuron_sdk` configuration flag in `config.py` has
also been removed. If your `config.py` references this flag, remove the
line — it is no longer recognized.
