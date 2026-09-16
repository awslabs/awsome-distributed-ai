# GPU cluster failure patterns on AWS

AIM344 is a 60-minute Builders' Session on diagnosing distributed GPU jobs and verifying recovery. The main sequence uses a dedicated PCS G7 pair for GPU or EFA device unavailability, followed by checkpoint-write exhaustion, a DataLoader process-start comparison, and a discussion of correctness evidence. Select diagnostics from the earliest observed failure and preserve their raw output.

The diagnostic owner is [validation/gpu-cluster-healthcheck](../../../validation/gpu-cluster-healthcheck). The companion invokes that suite and supplies workload, injection, restoration and evidence-recording helpers. Administrative GPU PCI removal changes operating-system device visibility; it does not establish physical damage or NVIDIA Xid 79. EFA driver unbind is a separate intervention from the optional plugin-fallback exercise.

## Validation status

The current device-recovery qualification targets 2 physical g7.48xlarge nodes on AWS PCS Slurm in Europe (Spain), with 8 GPUs and 2 EFA devices per node. [VALIDATION.md](VALIDATION.md) records completed runs and remaining limits. Earlier g7e, p5, p6-b300 and g7.24xlarge-equivalent measurements retain their original configurations there. A constrained g7.48xlarge allocation does not qualify physical g7.24xlarge device recovery.

The health suite is pinned by `HEALTHCHECK_COMMIT` in [pins.env](pins.env). This is a development qualification revision, pending upstream integration. Fetch the exact published revision for reproduction; a final released workshop pin requires the fixes to be merged upstream. Do not substitute local edits under the same revision label.

## Prerequisites and deployment

Use the [PCS reference architecture](../../../architectures/aws-pcs/README.md) to prepare a dedicated homogeneous pair with EFA, compatible NVIDIA drivers, Python version 3, Enroot version 3.5.0 and Pyxis version 0.20.0 built against the deployed Slurm version. Keep the pair allocated during the exercise. The facilitator needs an external maintenance host, permission to drain and resume the exact nodes, and a verified PCS Slurm reboot route that retains the instance and capacity.

Build and stage the images before the session. The base image is pinned to `public.ecr.aws/hpc-cloud/nccl-tests@sha256:a5390d3f0eb50f3e5854085e2ae0fef90b9eb4f3b7486ed4cab0d47864809d4c`. Its tag resolves CUDA version 13.0.2, NCCL version 2.30.4, nccl-tests version 2.18.3, EFA installer version 1.48.0 and aws-ofi-nccl version 1.19.0. The PyTorch fixture uses version 2.9.0+cu130. Keep runtime NCCL startup logs because the wheel's build version and the loaded library can differ.

The Dockerfile rebuilds the pinned NCCL and test sources with native Blackwell `sm_120` kernels. The public base image alone lacks those kernels and a usable PTX fallback for this G7 workload. The preparation helper builds both targets, imports them and verifies the health-suite archive against its commit:

```bash
cd examples/use-cases/gpu-cluster-failure-patterns
bash facilitator/prepare-login.sh
sha256sum /fsx/aim344/aim344.sqsh /fsx/aim344/nccl-baseline.sqsh
```

The default preparation requires a Lustre staging directory. For an existing node-local staging filesystem, set `AIM344_NODE_LOCAL=1` and `AIM344_STAGE_DIR=/opt/aim344`. Copy the same companion files, `healthcheck-pinned.tgz`, and image files to the same absolute paths on both compute nodes. Verify their hashes and pre-import both images through Pyxis. The prepared Spain session uses `/opt/aim344/device-recovery/companion` for this directory and `/opt/aim344` for the images. Record the existing staging filesystem and remount it without formatting after reboot.

On each compute node, install the suite and prepare the private 32 MiB checkpoint fixture:

```bash
sudo env AIM344_NODE_LOCAL=1 AIM344_STAGE_DIR=/opt/aim344   bash facilitator/prepare-compute.sh ubuntu
sudo env AIM344_NODE_LOCAL=1 AIM344_STAGE_DIR=/opt/aim344   bash facilitator/prepare-prolog.sh gpu-g7
```

Use the actual participant user, staging directory and partition for your deployment. Shared-storage preparation can use the helpers' default stage instead. The device injector requires a separately provisioned root-owned target allowlist; follow [Device recovery](facilitator/DEVICE-RECOVERY.md) before enabling it.

Create `lab.env` in the prepared companion directory on both nodes. For the prepared Spain topology, its settings are:

```bash
export LAB_DIR=/opt/aim344/device-recovery/companion
export PARTITION=gpu-g7 GPUS_PER_NODE=8 CPUS_PER_RANK=24 MEMORY_PER_NODE=0
export NCCL_ENROOT_IMAGE=/opt/aim344/nccl-baseline.sqsh
export TORCH_IMAGE=/opt/aim344/aim344.sqsh
export CHECKPOINT_DIR=/run/aim344-checkpoints
export PATH=/opt/amazon/efa/bin:/opt/aws/pcs/scheduler/slurm-25.05/bin:$PATH
export FI_EFA_USE_HUGE_PAGE=0
export NCCL_MIN_BUS_BW=30
unset AIM344_EFA_IFACE FI_EFA_IFACE OFI_NCCL_PROTOCOL AIM_ENFORCE_STANDIN
```

The memory setting is Slurm's all-memory sentinel. Keep the assigned instance types, scheduler version and paths with the evidence; another deployment must derive its own values. Use the unaffected GPU node as the participant coordinator because the suite reads the local instance profile and GPU environment before launching its collective check.

### Facilitator preparation for the PCS health gate

The real Prolog invokes suite checks with identifiers `0` and `2`. The root-owned dispatcher applies it only to the dedicated partition and permits root maintenance jobs. Enable the dispatcher through the PCS API while preserving the cluster's other settings:

```bash
python3 facilitator/pcs-prolog.py enable --cluster "$CLUSTER_ID" --region "$AWS_REGION"   --dispatcher /opt/aim344/.prolog/dispatch.sh --state-file pcs-before-aim344.json
```

Keep the state file on the external controller. Confirm the effective Slurm Prolog configuration and its journal on both nodes before admitting participants. Use `pcs-prolog.py restore` with the same cluster, Region and state file when restoring the prior scheduler settings. The helper refuses to overwrite an existing Prolog or unrelated changes made after activation.

Run suite identifiers `0`, `2`, `3` and `6` on both healthy hosts. Run DCGM identifiers `1` and `4` with exclusive access before the session; stop containerized telemetry through its owner and restore its previous state afterward. Retain the raw tests, entities and skipped coverage. The Level 4 command alone does not prove that EUD executed.

## Healthy baseline and G7 allocation

From the prepared directory on the unaffected coordinator, load the environment and read the registered resources:

```bash
source lab.env
eval "$(python3 facilitator/allocation-env.py --partition "$PARTITION")"
salloc --exclusive --partition="$PARTITION" --nodes=2 --gres="gpu:$GPUS_PER_NODE"   --ntasks-per-node=1 --cpus-per-task="$((GPUS_PER_NODE * CPUS_PER_RANK))"   --mem="$MEMORY_PER_NODE" --time=00:10:00
bash 10.healthcheck-nccl.sh
exit
```

Invoke Check 5 once from the coordinator. It launches 1 MPI process per node with 8 GPUs per process on this pair. The sweep covers 8 B through 128 MiB with correctness enabled. Require a complete sweep and 0 mismatches in both correctness columns; keep the provider log and per-device EFA byte deltas. Compare bandwidth with repeated healthy runs using the same configuration. The helper does not assign an independent health verdict.

The prepared Spain pair uses an advisory floor of 30 GB/s for Check 5. Across 6 fresh allocations after accepted device recoveries, the maximum bus bandwidth ranged from 38.18 GB/s through 39.64 GB/s with all 25 rows correct. This floor is a review trigger for that configuration. Derive a new floor when the hardware, image or launch mapping changes.

The Torch and optional long-sweep launchers select `OFI_NCCL_PROTOCOL=RDMA` for G7. Keep the runtime transport logs and physical counters as evidence. Leave `AIM344_EFA_IFACE` unset for the whole-node baseline; the facilitator may select the qualified physical EFA for the EFA fault round.

## Device fault and recovery

Follow [the complete facilitator procedure](facilitator/DEVICE-RECOVERY.md). Keep the control session and logs outside the target. The root-owned helper checks the instance, GPU UUID/BDF or EFA BDF, driver binding, management interface, drain reason, allowed job and exact confirmation token.

For GPU removal, end the healthy allocation and keep the target idle. The facilitator records and pauses telemetry, saves and disables persistence on the selected GPU, and confirms that no compute process uses it. The qualified procedure uses basic PCS Slurm reboot after removal. Active GPU removal is retained as a recorded diagnostic case because its shutdown stalled during testing.

For the separate EFA round, take an allocation named `aim344-device` with the baseline resource flags. Set `AIM344_EFA_IFACE` to the physical RDMA device supplied by the facilitator. From the unaffected coordinator, run:

```bash
: "${AIM344_EFA_IFACE:?Set the facilitator-supplied physical RDMA device}"
export AIM344_EFA_IFACE
bash 11.device-workload.sh
```

The workload checks every completed collective and emits advancing progress records. The facilitator verifies progress and physical EFA traffic, drains only the target, and unbinds the selected EFA. Observe the application outcome and the suite's fault diagnostics. Record an incomplete command with its external deadline and process state; a timeout cannot guarantee termination of an uninterruptible kernel task.

End the faulted allocation. Rebind the selected EFA only after its old processes have ended; use the verified reboot route when required. Keep the target drained while restoring its existing staging mount, image paths, private tmpfs marker and ownership, persistence mode, telemetry, Slurm daemon and Prolog. Review the maintenance checks and any retained warnings before resuming. Take a fresh allocation for Check 5 and the storage smoke test, with the EFA selector unset for whole-node verification. Keep the instances and reservation capacity.

## Checkpoint writes and collective timeout

Use a fresh healthy allocation after device recovery. Select the private fixture and run:

```bash
export CHECKPOINT_DIR=/run/aim344-checkpoints
bash 5.inject-storage.sh
bash 6.recover-storage.sh
bash 9.cleanup.sh
exit
```

From the coordinator terminal, take a fresh allocation with the baseline resource flags and a 20-minute time limit. Run `bash 10.healthcheck-nccl.sh` once there, then keep that allocation for the DataLoader controls.

The injector writes at most 64 MiB into a private 32 MiB tmpfs and refuses to arm the fault if that limit does not exhaust it. Its writer retries for up to 60 seconds while peers wait at a collective with a 30-second deadline. Preserve the earlier `ENOSPC` or quota error, the later application outcome and the launcher status. Wait for failed workers to exit before recovery.

Recovery removes only the lab filler and partial write. It retains `last.json`, which stores a next-step index for this fixture, then resumes and checks the collective result. This file is not a complete model or optimizer checkpoint. The exercise reproduces local write exhaustion; an FSx outage or Lustre quota needs separate qualification.

## Late DataLoader workers

Run both controls in the healthy allocation after storage recovery:

```bash
bash 7.probe-dataloader-fork.sh
bash 8.use-dataloader-spawn.sh
```

Both scripts hold `FI_EFA_USE_HUGE_PAGE=0` by default and vary the worker-start method. They complete a collective before creating workers, read the C environment's fork-safety flag, and record completion or a bounded hang. Keep the kernel, PyTorch, runtime NCCL, libfabric and plugin versions with the result. If both methods complete, record non-reproduction on that workload and stack.

Forked children can inherit thread and lock state from an initialized parent. [PyTorch's multiprocessing guidance](https://docs.pytorch.org/docs/2.6/notes/multiprocessing.html) explains those deadlocks and requires `spawn` or `forkserver` for CUDA in subprocesses. The DataLoader fixture uses CPU data in its workers; its successful completion does not qualify arbitrary CUDA work in forked children.

## Optional pre-job marker and plugin fallback

The synthetic marker demonstration uses `1.inject-prejob.sh` and `2.recover-prejob.sh`. Run it with no interactive allocation occupying the pair. Match the Prolog journal to the drained node and any held requeue; the marker explains the scheduler transition and does not diagnose hardware. Preserve and restore node weights if they are changed to make the probe deterministic.

The optional plugin comparison uses `0.baseline.sh`, `3.inject-fallback.sh` and `4.recover-fallback.sh` in the same allocation. It covers 8 B through 2 GiB with 1 MPI rank per GPU. The injection temporarily hides the OFI library directory in private writable containers, including network and tuner libraries. A Socket selection, correct arithmetic and flat EFA counters together identify the observed fallback. These measurements have a different message range and launch mapping from Check 5, so retain separate baselines.

## Cleanup and known edge cases

Run `bash 9.cleanup.sh` in the healthy allocation, then exit it. Retain the evidence. The facilitator restores exercise-owned markers, mounts, telemetry and scheduler settings while retaining the dedicated pair and capacity. Remove only this exercise's private container instances; keep the staged images available for reuse.

A launcher or image error must be resolved before its result can describe GPU communication. Use `FI_EFA_IFACE` to select a physical EFA; `FI_EFA_DEVICE_NAME` is ignored by the shipped libfabric. Missing counters require investigation and must not be recorded as 0 B. `iptables` and `tc netem` do not establish an EFA device failure because the EFA data path bypasses that kernel networking path.

For a known administrative device removal, preserve the suite severity and keep the node drained through recovery and qualification. `ISOLATE` alone does not require replacing the instance. For an unexplained recurring fault, retain the raw evidence and follow the platform's repair or replacement process. Use the [failure-signature runbook](RUNBOOK.md) to record the decision.
