# GPU cluster failure patterns on AWS

AIM344 is a 60-minute Builders' Session on diagnosing distributed GPU jobs and verifying recovery. The main sequence uses a dedicated PCS G7 pair for GPU or EFA device unavailability, followed by checkpoint-write exhaustion, a DataLoader process-start comparison, and a discussion of correctness evidence. Select diagnostics from the earliest observed failure and preserve their raw output.

The diagnostic owner is [validation/gpu-cluster-healthcheck](../../../validation/gpu-cluster-healthcheck). The companion invokes that suite and supplies workload, injection, restoration and evidence-recording helpers. Administrative GPU PCI removal changes operating-system device visibility; it does not establish physical damage or NVIDIA Xid 79. EFA driver unbind is a separate intervention from the optional plugin-fallback exercise.

## Validation status

The current device-recovery qualification targets 2 physical g7.48xlarge nodes on AWS PCS Slurm in Europe (Spain), with 8 GPUs and 2 EFA devices per node. [VALIDATION.md](VALIDATION.md) records completed runs and remaining limits. Earlier g7e, p5, p6-b300 and g7.24xlarge-equivalent measurements retain their original configurations there. A constrained g7.48xlarge allocation does not qualify physical g7.24xlarge device recovery.

The health suite is pinned by `HEALTHCHECK_COMMIT` in [pins.env](pins.env) to candidate `ffed0df6102e68b3e94806373006b045e432298f`. Use the [suite acquisition procedure](facilitator/DEVICE-RECOVERY.md#suite-acquisition) before preparation. This candidate is not yet published or merged into upstream main; the public fetch route becomes usable only after publication and exact remote verification. Its runtime files and modes match the measured suite; differences are regression-test packaging and a README correction linking the separately acquired native GPU image recipe to an available public revision. Do not substitute the companion checkout's unpinned suite or local edits under the same revision label.

Independent source/evidence review accepts the measured participant recovery/replacement and fresh workload flow; retained artifacts were checked byte-for-byte. **Full hardware qualification remains withheld**: first-hook-held failure/access-denial proof is missing and IPv6 is not qualified. Candidate publication is a separate packaging decision, not closure of those limits. See the [current scoped record](VALIDATION.md#september-19-2026-scoped-participant-recovery-audit), including successor/peer differences and timing limits. The 60-minute session format is not a measured end-to-end latency guarantee.

## Prerequisites and deployment

Use the [PCS reference architecture](../../../architectures/aws-pcs/README.md) for the dedicated same-hardware pair with EFA, compatible NVIDIA drivers, Python version 3, Enroot version 3.5.0 and Pyxis version 0.20.0 built against the deployed Slurm version. The authoritative measured provisioning composition is [Device recovery](facilitator/DEVICE-RECOVERY.md#pre-session-provisioning--separate-authorization): independent login coordinator, reviewed early LT2 isolation plus unchanged native/custom MIME, per-node fixed SSM initializer, literal admission Prolog and root-only whole-node reservation through qualification. The pair is not a homogeneous final-helper rollout. Keep capacity reserved during the exercise, without promising uninterrupted two-node availability. New provisioning or fault trials require separate operational authorization.

Build and stage the images before the session. The base image is pinned to `public.ecr.aws/hpc-cloud/nccl-tests@sha256:a5390d3f0eb50f3e5854085e2ae0fef90b9eb4f3b7486ed4cab0d47864809d4c`. Its tag resolves CUDA version 13.0.2, NCCL version 2.30.4, nccl-tests version 2.18.3, EFA installer version 1.48.0 and aws-ofi-nccl version 1.19.0. The PyTorch fixture uses version 2.9.0+cu130. Keep runtime NCCL startup logs because the wheel's build version and the loaded library can differ.

The Dockerfile rebuilds the pinned NCCL and test sources with native Blackwell `sm_120` kernels. The public base image alone lacks those kernels and a usable PTX fallback for this G7 workload. The preparation helper builds both targets, imports them and verifies the health-suite archive against its commit:

```bash
cd examples/use-cases/gpu-cluster-failure-patterns
bash facilitator/prepare-login.sh
sha256sum /fsx/aim344/aim344.sqsh /fsx/aim344/nccl-baseline.sqsh
```

The asset-preparation helper defaults to Lustre. For an existing node-local staging directory, set `AIM344_NODE_LOCAL=1` and `AIM344_STAGE_DIR=/opt/aim344`. The measured successor initializer installs hash-pinned materials at `/opt/aim344/device-recovery/companion` and images at `/opt/aim344`. Its `root-directory` policy validates root-filesystem staging after reboot; this is not an NVMe remount. Historical nodes with the NVMe-bind policy retain their own source/UUID checks and are not silently migrated.

The following manual compute preparation is historical initial bring-up, not the current replacement initializer or a participant restoration step:

```bash
sudo env AIM344_NODE_LOCAL=1 AIM344_STAGE_DIR=/opt/aim344   bash facilitator/prepare-compute.sh ubuntu
sudo env AIM344_NODE_LOCAL=1 AIM344_STAGE_DIR=/opt/aim344   bash facilitator/prepare-prolog.sh gpu-g7
```

For current pre-session provisioning, use the fixed initializer and matching participant identities described in [Device recovery](facilitator/DEVICE-RECOVERY.md), not the historical `ubuntu` example above. The device injector requires a separately provisioned root-owned target allowlist before enabling it.

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

The memory setting is Slurm's all-memory sentinel. Keep the assigned instance types, scheduler version and paths with the evidence; another deployment must derive its own values. The participant coordinator is the independent login host, outside the replaceable GPU group. Submit `13.verify-after-recovery.sbatch` there; Check 5 runs once on its allocated GPU batch host, not on the login host. The local suite pin uses physical DMI detection without granting participants IMDS access. Historical unaffected-GPU coordinator examples below are not the current control topology.

### Facilitator preparation for the PCS health gate

The health Prolog invokes suite checks with identifiers `0` and `2`. In the measured composition, the root-owned literal admission wrapper chains that health Prolog. Only separately authorized pre-session setup may configure the cluster setting, after installing the reviewed wrapper and preserving other settings:

```bash
python3 facilitator/pcs-prolog.py enable --cluster "$CLUSTER_ID" --region "$AWS_REGION"   --dispatcher /usr/local/sbin/aim344-replacement-prolog --state-file pcs-before-aim344.json
```

Keep the state file on the external controller. Confirm the effective Slurm Prolog configuration and its journal on both nodes before admitting participants. Use `pcs-prolog.py restore` with the same cluster, Region and state file when restoring the prior scheduler settings. The helper refuses to overwrite an existing Prolog or unrelated changes made after activation.

Run suite identifiers `0`, `2`, `3` and `6` on both healthy hosts. Run DCGM identifiers `1` and `4` with exclusive access before the session; stop containerized telemetry through its owner and restore its previous state afterward. Retain the raw tests, entities and skipped coverage. The Level 4 command alone does not prove that EUD executed.

## Healthy baseline and G7 allocation

For the current measured path, submit `sbatch --wait /opt/aim344/device-recovery/13.verify-after-recovery.sbatch` from the independent login account and use its printed fresh-`srun` retrieval command after completion. The following interactive baseline is the historical GPU-host-shell route, not a command sequence to run Check 5 locally on the login host:

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

Follow [participant recovery and pre-session provisioning](facilitator/DEVICE-RECOVERY.md). Keep the control session and logs outside the target. The root-owned helper checks the instance, GPU UUID/BDF or EFA BDF, driver binding, management interface, drain reason, allowed job and exact confirmation token. The manual trials below are historical qualification context, not participant session remedies; the current route is the participant-operated variant.

For GPU removal, end the healthy allocation and keep the target idle. The facilitator records and pauses telemetry, saves and disables persistence on the selected GPU, and confirms that no compute process uses it. The qualified procedure uses basic PCS Slurm reboot after removal. Active GPU removal is retained as a recorded diagnostic case because its shutdown stalled during testing.

For the separate EFA round, take an allocation named `aim344-device` with the baseline resource flags. Set `AIM344_EFA_IFACE` to the physical RDMA device supplied by the facilitator. From the unaffected coordinator, run:

```bash
: "${AIM344_EFA_IFACE:?Set the facilitator-supplied physical RDMA device}"
export AIM344_EFA_IFACE
bash 11.device-workload.sh
```

The workload checks every completed collective and emits advancing progress records. The facilitator verifies progress and physical EFA traffic, drains only the target, and unbinds the selected EFA. Observe the application outcome and the suite's fault diagnostics. Record an incomplete command with its external deadline and process state; a timeout cannot guarantee termination of an uninterruptible kernel task.

End the faulted allocation. Rebind the selected EFA only after its old processes have ended; use the verified reboot route when required. Keep the target drained while restoring its existing staging mount, image paths, private tmpfs marker and ownership, persistence mode, telemetry, Slurm daemon and Prolog. Review the maintenance checks and any retained warnings before resuming. Take a fresh allocation for Check 5 and the storage smoke test, with the EFA selector unset for whole-node verification. Keep the instances and reservation capacity.

### Participant-operated variant

The procedure above is facilitator-driven. For a Builders' Session, the same rounds run from the participant's own terminal instead, with no facilitator step during normal progress. Install it once per assigned pair with [`facilitator/install-participant-control.sh`](facilitator/install-participant-control.sh), which creates an unprivileged account per table, an SSH key pinned to a forced command, and exactly one narrowly scoped sudo rule. The participant then uses:

```bash
bash 12.device-exercise.sh status
bash 12.device-exercise.sh start gpu      # or: start efa
bash 12.device-exercise.sh collect 0      # allowlisted checks only
bash 12.device-exercise.sh recover
bash 12.device-exercise.sh replace      # only after supported recovery cannot qualify
```

No node name, PCI address, job id or command reaches the privileged helper; the assignment comes from a root-owned configuration and the confirmation token is read from the target's own inspection output. `recover` ends at `runtime-ready`, which is deliberately not the same result as a verified workload: the participant then takes their own allocation and runs [`13.verify-after-recovery.sbatch`](13.verify-after-recovery.sbatch) for the Check 5 collective, per-device EFA counters and the storage fixture. That script runs entirely as the participant and needs no sudo, which is the point: an entry point that required a permission the participant is not granted would not be a participant-operated verification at all. It prints the suite's own raw correctness rows rather than only a verdict line, judges the EFA byte deltas of every device on every allocated node rather than only on the node it happens to be running on, and checks the private checkpoint fixture is usable on every allocated node before starting the distributed storage workload.

The batch host is whichever of the two assigned nodes Slurm picks, so the job's log and results can land on the node the participant does not have a terminal on. The script therefore copies its own results to every allocated node before it exits, verifies the copy's digest there, and prints the `tar -xf` command that reads them back. A run whose results could not be distributed reports itself as unverified rather than as a pass.

A round in progress on a node is exclusive to that table, and an exercise deadline recovery is recorded separately from a recovery the participant earned by diagnosing.

The runtime has scoped independent acceptance for two actual changed-address participant replacements and a fresh verification job, not full hardware or event-readiness approval. `replace` accepts no target argument. It preserves off-target evidence, retires the exact assigned failed EC2 identity, and initializes/qualifies the PCS-provisioned successor without instructor restoration. Retries do not terminate the healthy successor. Fixed SSM, pinned bootstrap, IAM, early LT isolation and the literal Prolog require separate pre-session provisioning. No PCS group or EC2 capacity-reservation update is a session action; the controller does create/release its owned Slurm scheduling reservation. WARN acceptance is disabled. Successful replacement yields fresh runtime readiness, not recovery of lost model state or proof that Check 5 has run.

Both participant rounds run while the exercise node is idle. The GPU round always did; the EFA round does too, because the active variant cannot be completed from a one-command-per-connection route on this stack: the unbind does not return until the workload holding the device ends, measured here at 171 s outstanding and returning 11 s after the job was cancelled. `start efa` with one of your own jobs on the node is refused before anything is reserved, and says so. [Device recovery](facilitator/DEVICE-RECOVERY.md) records the measurement and what adopting the active path would require.

## Log investigation cases

No case bundle is distributed with this candidate. The optional case helper requires a separately approved bundle staged read-only on the coordinator. It reports unavailable material rather than inventing cases. Once such a bundle is supplied:

```bash
bash 14.case-exercise.sh          # list the cases and their files
bash 14.case-exercise.sh start    # set up your own answers file
```

Each case gives a reported symptom and the files a responder would have been handed. The reported symptom is what somebody believed at the time, and deciding whether the evidence supports it is part of the work. Investigate with ordinary tools; nothing is aliased or wrapped, so `nvidia-smi` and `sinfo` talk about the real cluster rather than the case.

Answers are not staged with the cases, and no answer key is in this repository. Ask the workshop owner about separately approved teaching materials; anonymization alone does not confer distribution permission. Stage only an approved bundle with [`facilitator/stage-cases.sh`](facilitator/stage-cases.sh), which verifies the candidate against its build manifest while it is still private and exposes it at the participant path only if it passes. Private means unreadable, not merely unmounted: the durable master root and each candidate tree are created `0700`, and a candidate's directories are opened for traversal only after it has passed verification, so an unverified or rejected tree cannot be read at its own path either. A candidate that fails verification is never mounted: the participant path keeps serving the previously verified tree, or stays unavailable if there was none. An interrupted run leaves its tree marked incomplete and unreadable, and the next run removes it. The script then reads the exposed path back through the mount, confirms even root cannot write through it by attempting a write, checks as an unprivileged account that the bundle is readable through the mount and that the master root is not, and refuses a durable copy on instance storage.

The bundle checks in [`tests/test_case_bundle.py`](tests/test_case_bundle.py) match identifier shapes without publishing restricted literals. Optional literal checks read an owner-supplied list outside the repository:

```bash
AIM344_INTERNAL_DENYLIST=/path/to/internal-literals.txt \
  AIM344_CASES=/opt/aim344/cases python3 -m unittest discover -s tests
```

With the variable unset that one test reports a skip naming what did not run, rather than passing. Run the suite as an ordinary user, not with `sudo`: two tests assert refusals that only hold for a non-root caller.

## Checkpoint writes and collective timeout

Use a fresh healthy allocation after device recovery. Select the private fixture and run:

```bash
export CHECKPOINT_DIR=/run/aim344-checkpoints
bash 5.inject-storage.sh
bash 6.recover-storage.sh
bash 9.cleanup.sh
exit
```

This optional historical interactive storage/DataLoader sequence uses a GPU-host shell inside the fresh allocation, not a local shell on the independent login coordinator. Take the baseline resource flags and a 20-minute time limit, run `bash 10.healthcheck-nccl.sh` once there, then keep that allocation for the DataLoader controls. The current audited recovery workload path remains `13.verify-after-recovery.sbatch`; job141 does not newly qualify these additional fault modules.

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

Run `bash 9.cleanup.sh` in the healthy allocation, then exit it. Retain the evidence. Only during post-session administrative teardown does the facilitator restore exercise-owned markers, mounts, telemetry and scheduler settings while retaining the dedicated pair and capacity; this is not an in-session device-recovery remedy. Participants use `recover` or `replace` for that route. Remove only this exercise's private container instances; keep the staged images available for reuse.

A launcher or image error must be resolved before its result can describe GPU communication. Use `FI_EFA_IFACE` to select a physical EFA; `FI_EFA_DEVICE_NAME` is ignored by the shipped libfabric. Missing counters require investigation and must not be recorded as 0 B. `iptables` and `tc netem` do not establish an EFA device failure because the EFA data path bypasses that kernel networking path.

For a known administrative device removal, preserve the suite severity and keep the node drained through recovery and qualification. `ISOLATE` alone does not require replacing the instance. For an unexplained recurring fault, retain the raw evidence and follow the platform's repair or replacement process. Use the [failure-signature runbook](RUNBOOK.md) to record the decision.
