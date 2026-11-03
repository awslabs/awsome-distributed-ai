# GPU cluster failure patterns on AWS

AIM344 is a 60-minute Builders' Session on diagnosing distributed GPU jobs and verifying recovery. This unapproved schedule proposal assigns one live branch per table: GPU or EFA device fault, or checkpoint-write exhaustion with a DataLoader comparison. Tables compare the other mechanisms from recorded evidence while recovery is pending. Device tables compare recorded storage, then DataLoader outcomes without running those commands. Storage tables run storage recovery, fresh verification/retrieval and DataLoader controls and compare recorded device results. Both routes then discuss correctness coverage and complete the runbook. Full procedures remain available for separately scheduled practice. Content-owner agreement and a human-paced rehearsal are still required. Select diagnostics from the earliest observed failure and preserve their raw output.

The diagnostic owner is [validation/gpu-cluster-healthcheck](../../../validation/gpu-cluster-healthcheck). The companion invokes that suite and supplies workload, injection, restoration and evidence-recording helpers. Administrative GPU PCI removal changes operating-system device visibility; it does not establish physical damage or NVIDIA Xid 79. EFA driver unbind is a separate intervention from the optional plugin-fallback exercise.

## Validation status

The current device-recovery qualification targets 2 physical g7.48xlarge nodes on AWS PCS Slurm in Europe (Spain), with 8 GPUs and 2 EFA devices per node. [VALIDATION.md](VALIDATION.md) records completed runs and remaining limits. Earlier g7e, p5, p6-b300 and g7.24xlarge-equivalent measurements retain their original configurations there. A constrained g7.48xlarge allocation does not qualify physical g7.24xlarge device recovery.

The health dependency in [pins.env](pins.env) identifies exact review candidate `3333c9d04059c23703853bf5f89326aaace56d6e`. It adds the G7e capability profile and trusted assigned-EFA contract to historical public base `474155a63059769bcea6365d4ec2b2f34d4c8000`, while preserving profile-only Check 2 warnings when no assignment is supplied. This revision records local review candidates, not merged or released source, deployment or hardware qualification; public availability has not been verified. The [suite acquisition procedure](facilitator/DEVICE-RECOVERY.md#suite-acquisition) accepts an explicitly supplied clean local checkout, or the canonical repository only after the publication owner verifies the exact public commit. Any later accepted release requires new archive and manifest hashes.

Independent source/evidence review accepts the measured participant recovery/replacement and fresh workload flow; retained artifacts were checked byte-for-byte. **Full hardware qualification remains withheld**: first-hook-held failure/access-denial proof is missing and IPv6 is not qualified. Candidate publication is a separate packaging decision, not closure of those limits. See the [earlier scoped record](VALIDATION.md#september-19-2026-scoped-participant-recovery-audit), including successor/peer differences and timing limits. The 60-minute session format is not a measured end-to-end latency guarantee.

The [September 24 fixed-candidate record](VALIDATION.md#september-24-2026-fixed-candidate-participant-efa-recovery) confirms active EFA failure, participant recovery refusing WARN, replacement, fresh correctness/storage and participant retrieval. The measured frozen trial took 39 minutes 10 seconds; it does not prove that all failure branches fit a first-time 60-minute session. The participant GPU attempt has a missing reset-return marker and fault-time kernel evidence, despite successful later recovery and retrieval. Keep those limits separate from the accepted EFA sequence.

This integration includes the reviewed EC2 empty-response/receipt-ordering fix, Check 5 table/provider parsing, Check 6 interface pinning and counter handling, result-severity accumulation, current-job cleanup after allocated-node bundle verification, and two-epoch DataLoader controls. The integrated runtime completed normal verification, storage correctness and subsequent participant retrieval on the two-node target. Node-local Check 6 retained WARN/MONITOR in saved JSON despite successful loopback completion; cumulative timeout counts were 9 and 18, not incident deltas or evidence of harmlessness. Both fork and spawn controls completed two epochs with new nonpersistent workers, four batches per epoch, eight batch collectives per rank, zero mismatches and exit 0. This is non-reproduction, not a reproduced or repaired DataLoader failure. The EC2 acknowledgment fix remains locally regression-tested only. These results do not establish default/bootstrap/replacement rollout, full-size performance qualification, future storage headroom, or the event participant entry path. Source publication and pin/documentation changes do not repeat or extend those hardware trials.

The scheduled incidents and split check tables described under [Participant-operated variant](#participant-operated-variant) have local unit tests only; [VALIDATION.md](VALIDATION.md#november-2-2026-scheduled-incidents-and-split-check-tables) lists what they cover and what awaits hardware. The participant launcher `job.sh`, the login profile and the checkpoint-writer options have local unit tests, and the storage and slow-job incidents have one p5 run through `job.sh`; [VALIDATION.md](VALIDATION.md#november-2-2026-the-participant-launcher-and-the-cause-hidden-job-incidents) lists both and what still awaits hardware.

## Prerequisites and deployment

Use the [PCS reference architecture](../../../architectures/aws-pcs/README.md) for the dedicated same-hardware pair with EFA, compatible NVIDIA drivers, Python version 3, Enroot version 3.5.0 and Pyxis version 0.20.0 built against the deployed Slurm version. The authoritative measured provisioning composition is [Device recovery](facilitator/DEVICE-RECOVERY.md#pre-session-provisioning--separate-authorization): independent login coordinator, reviewed early LT2 isolation plus unchanged native/custom MIME, per-node fixed SSM initializer, literal admission Prolog and root-only whole-node reservation through qualification. The September 19 pair was not a homogeneous final-helper rollout; the latest fixed-candidate deployment is recorded separately in VALIDATION.md. Keep capacity reserved during the exercise, without promising uninterrupted two-node availability. New provisioning or fault trials require separate operational authorization.

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

The current initializer generates `participant-lab.env` for installation as `lab.env` on the login coordinator and compute nodes. It preserves explicit caller overrides and does not set `PATH` or `NCCL_MIN_BUS_BW`. The following hand-written environment is the historical Spain configuration, not the current generated defaults:

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

Run suite identifiers `0`, `2`, `3` and `6` on both healthy hosts. Run DCGM identifiers `1` and `4` with exclusive access before the session; stop containerized telemetry through its owner and restore its previous state afterward. Retain the raw tests, entities and skipped coverage. Provide approved participant-readable raw paths with host/device/boot and capture identity in the access handoff. This candidate does not stage such a pre-session raw bundle automatically. Participants can read the existing staged `VALIDATION.md` for historical scope, but it cannot establish a current warning baseline. The Level 4 command alone does not prove that EUD executed.

## Healthy baseline and G7 allocation

For the post-recovery path, source the site-generated `/opt/aim344/device-recovery/companion/lab.env` on the independent login coordinator, explicitly run `export PATH=/opt/amazon/efa/bin:/opt/aws/pcs/scheduler/slurm-25.05/bin:$PATH` for the measured PCS installation, and confirm `command -v sinfo salloc sbatch srun`. Then submit `sbatch --wait --export=ALL --partition="$PARTITION" --cpus-per-task="$((GPUS_PER_NODE * CPUS_PER_RANK))" /opt/aim344/device-recovery/13.verify-after-recovery.sbatch` from the independent login account and use its printed fresh-`srun` retrieval command after completion. The following interactive baseline is the historical GPU-host-shell route, not a command sequence to run Check 5 locally on the login host:

```bash
source lab.env
eval "$(python3 facilitator/allocation-env.py --partition "$PARTITION")"
salloc --exclusive --partition="$PARTITION" --nodes=2 --gres="gpu:$GPUS_PER_NODE"   --ntasks-per-node=1 --cpus-per-task="$((GPUS_PER_NODE * CPUS_PER_RANK))"   --mem="$MEMORY_PER_NODE" --time=00:10:00
bash 10.healthcheck-nccl.sh
exit
```

Invoke Check 5 once from the coordinator. It launches 1 MPI process per node with 8 GPUs per process on this pair. The sweep covers 8 B through 128 MiB with correctness enabled. Require a complete sweep and 0 mismatches in both correctness columns; keep the provider log and per-device EFA byte deltas. Compare bandwidth with repeated healthy runs using the same configuration. The helper does not assign an independent health verdict.

The historical September 16 configuration explicitly set a 30 GB/s advisory for Check 5. Across six fresh allocations after accepted device recoveries, maximum bus bandwidth ranged from 38.18 GB/s through 39.64 GB/s with all 25 rows correct. The current generated environment does not set that advisory; the fixed G7 suite defaults to zero without an explicit override. Do not apply the historical floor or claim performance acceptance for the latest candidate. Compare the actual configuration and retain its provider/performance WARN.

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

The procedure above is facilitator-driven. For a Builders' Session, participants operate fault injection and recovery from their own terminal. Install the route once per assigned pair with [`facilitator/install-participant-control.sh`](facilitator/install-participant-control.sh), which creates an unprivileged account per table, an SSH key pinned to a forced command, and exactly one narrowly scoped sudo rule. Choose an idle removal/unbind round with `start gpu` or `start efa`, or follow the [active FLR participant procedure](facilitator/DEVICE-RECOVERY.md#active-flr-participant-integration-candidate) for a fault during the running device workload:

```bash
bash 12.device-exercise.sh status
bash 12.device-exercise.sh start gpu      # idle round; or: start efa
# For an active round, replace the start command with:
# bash 12.device-exercise.sh active gpu   # or: active efa
bash 12.device-exercise.sh collect 0      # checks 0, 2, 3 or 6, or kernel
bash 12.device-exercise.sh recover
```

When recovery cannot qualify the node, the node stays out of service for the rest of the session and an operator replaces it afterwards from the coordinator with `sudo /usr/local/sbin/aim344-device-session --operator replace --assignment <assignment-id>`. The participant verb `replace` is still accepted by the grammar and is refused without changing anything. `sudo /usr/local/sbin/aim344-device-session --operator status --assignment <assignment-id>` shows the operator the exercise kind, the seat's schedule with each entry's settings and issue time, the path of the device-operation transcript and the original text of the last recovery refusal.

No node name, PCI address, job id or command reaches the privileged helper; the assignment comes from a root-owned configuration and the confirmation token is read from the target's own inspection output. `recover` ends at `runtime-ready`, which is deliberately not the same result as a verified workload: the participant then takes their own allocation and runs [`13.verify-after-recovery.sbatch`](13.verify-after-recovery.sbatch) for the Check 5 collective, per-device EFA counters and the storage fixture. That script runs entirely as the participant and needs no sudo, which is the point: an entry point that required a permission the participant is not granted would not be a participant-operated verification at all. It prints the suite's own raw correctness rows rather than only a verdict line, judges the EFA byte deltas of every device on every allocated node rather than only on the node it happens to be running on, and checks the private checkpoint fixture is usable on every allocated node before starting the distributed storage workload. Because the workload writes on its writer's node only, each node also writes a small file to its checkpoint directory, flushes it and removes it, and `storage=0` requires that write to succeed on every node.

The batch host is whichever of the two assigned nodes Slurm picks, so the job's log and results can land on the node the participant does not have a terminal on. The script therefore copies its own results to every allocated node before it exits, verifies the copy's digest there, and prints the `tar -xf` command that reads them back. A run whose results could not be distributed reports itself as unverified rather than as a pass.

A round in progress on a node is exclusive to that table, and an exercise deadline recovery is recorded separately from a recovery the participant earned by diagnosing.

The replacement route has scoped independent acceptance for two actual changed-address replacements, which participants ran in the measured trials, and a fresh verification job, not full hardware or event-readiness approval. The operator runs the same route now, and it accepts no target argument. It preserves off-target evidence, retires the exact assigned failed EC2 identity, and initializes/qualifies the PCS-provisioned successor without instructor restoration. Retries do not terminate the healthy successor. Fixed SSM, pinned bootstrap, IAM, early LT isolation and the literal Prolog require separate pre-session provisioning. No PCS group or EC2 capacity-reservation update is a session action; the controller does create/release its owned Slurm scheduling reservation. WARN acceptance is disabled. Successful replacement yields fresh runtime readiness, not recovery of lost model state or proof that Check 5 has run.

`start gpu` and `start efa` require an idle target. For `active gpu` or `active efa`, first prepare both terminals on the same independent coordinator as the same participant, including identity, environment and client-PATH checks. Then launch the existing 120-second device workload in terminal A and keep its log open. Use terminal B for the fault, `status` and `collect`. The [active procedure](facilitator/DEVICE-RECOVERY.md#active-flr-participant-integration-candidate) gives the allocation, progress-log and recovery steps. After failed recovery, the replacement route may retire the assigned predecessor while this round's exact same-participant job remains COMPLETING. Other jobs and RUNNING allocations still block retirement. Admission of the successor and RESUME require an empty queue.

#### Checks a participant can collect

While a round holds the exercise node, `collect` accepts the same suite checks whatever the exercise: `0`, `2`, `3` and `6`, plus `kernel`. Recovery still requires the kind's own checks to PASS before the node returns to service (GPU `0` and `3`, EFA `2` and `6`), and any WARN still refuses recovery. When a seat's configuration names no such checks for the kind, `start` and `recover` refuse and change nothing, so a node is never resumed after zero checks. An idle `start` records a baseline for all four checks, so the baseline file names no longer depend on the kind. An active round starts while the participant's job runs and records no baseline, because running the checks then would load that job's node, so its next step points the participant to the healthy values recorded in the baseline module. The installer writes the two lists as `collect_checks` and `qualify_checks` in the assignment. A configuration written before this change has only `allowed_checks`, which keeps its earlier meaning for collection, baseline and qualification; a seat schedule refuses to run on such a configuration, because its collect refusal would list the kind's checks.

Once recovery returns the node to service, the node can carry the participant's next job, so `collect` narrows. It runs only the round's own qualification checks, and only while no job is on the node, because Check 6 sends EFA traffic and Check 2's gdrcopy test pins GPU memory. When `next` issues the following incident, `collect` refuses every check and `kernel` as it does before any round, so root-run checks stop with the round they belong to.

| Check | What root runs on the target | Time in one G7e run | Effect on the node |
|---|---|---|---|
| `0` | `nvidia-smi` queries, GPU count, Xid/SXid scan of the kernel log, ECC and retired pages | not recorded (suite estimate about 5 s) | reads only |
| `2` | `lspci`, `ibv_devices`, `fi_info -p efa`, `/dev/infiniband`, `lsmod`, `ulimit -l`, and the gdrcopy sanity test (30 s limit) | 8 s | the gdrcopy test pins and copies GPU memory |
| `3` | `nvidia-smi topo -m` and `topo -p2p`, and NVLink status and error queries where NVLink is expected | 1 s | reads only |
| `6` | `fi_pingpong` self-loopback per EFA domain (180 s limit each) and the EFA statistics | 19 s | real EFA traffic and counter growth |

The times come from one sequential run of checks `0`, `2`, `3` and `6` as root on a healthy g7e.48xlarge node with two EFAs on November 1, 2026. Each is the interval between the completion times of consecutive checks, and the run did not record when check `0` started. Collect times on g7.48xlarge have not been measured. Checks `1`, `4` and `5` stay outside both lists, and the target's maintenance allowlist is unchanged, so no new program becomes reachable as root.

#### Scheduled incidents

A seat can instead receive its exercises as numbered incidents whose cause the controller does not name. During account preparation the facilitator writes the seat's schedule as a JSON list, one entry per incident in the order they are issued, and passes it to the installer:

```bash
sudo AIM344_SCHEDULE_FILE=/root/seat-1.json bash facilitator/install-participant-control.sh \
     <assignment-id> <participant-user> <target-node> <target-instance-id> <target-host>
```

```json
[{"exercise": "slow-job"},
 {"exercise": "storage", "settings": {"writer_rank": 9}},
 {"exercise": "dataloader"},
 {"exercise": "silent-corruption", "settings": {"corrupt_rank": 13, "corrupt_step": 37, "corrupt_bit": 22, "corrupt_recurring": false}},
 {"exercise": "efa-reset"}]
```

The known exercises are `slow-job`, `storage`, `dataloader`, `silent-corruption` and `efa-reset`. Before it changes anything on the host, the installer checks the file with the validation of the controller it installs. It then copies the schedule into the root-only session configuration and resets the seat's record of issued incidents. It refuses a file that another account can read, a symlink, a file that is not JSON and an unknown exercise, and it refuses to reset the schedule while the seat's round still holds its exercise node. When the seat's last round has finished, the installer keeps that round's record in the state directory as `<assignment>.round-<n>.before-reset.json` and continues from a fresh record with the next round number, so `status` no longer shows the earlier pass's check results or answer line. Setting names match `[a-z][a-z0-9_]{0,31}`. Values are numbers, true/false or text of at most 128 characters from `[A-Za-z0-9._:/=+-]` that does not start with `-`, so a launcher cannot turn a value into shell syntax or into an option. A rerun without the variable keeps the configured schedule and its record.

The participant runs `bash 12.device-exercise.sh next`, which takes no argument and issues at most one incident per call. For a job exercise (`slow-job`, `storage`, `dataloader`, `silent-corruption`) it records the incident in the root-only state, then writes the incident's profile into the participant's results directory as the participant and prints where it is:

```text
Incident 4 is ready.
Profile: /home/aim344-t1/aim344-results/incidents/incident-4.json
Next: Your launcher runs this incident from the profile above. To repeat its job, reuse that profile rather than running next again.
```

The profile holds the incident number and the seat's incident count, the exercise, its settings, `controller_step` and the issue time. Only the current incident's profile leaves the root-only state, so later entries and their order stay unreadable to the participant. A failed profile write is finished by the next call rather than skipping the incident.

For `efa-reset`, the first `next` hands over the profile in the same way, with `controller_step` set to `next-after-progress`. A later `next` applies an active EFA function-level reset to the assigned target through the existing `active efa` route and all of its guards, once the participant's own job is RUNNING on the target and `~/aim344-results/device-<job-id>.log` shows two advancing progress records, the latest at most 15 seconds old. It then prints `Incident <n> is under way.` Until those conditions hold, `next` repeats the ready reply and changes nothing. `11.device-workload.sh` writes that log and those records. `next` refuses while a round still holds the node and after the node is kept out of service.

Each issued incident carries a random identifier, and the round that `next` starts for the device incident records it. A round left by an earlier pass of the same schedule therefore never counts as the current incident: `next` applies the device operation again, and `status` prints neither an incident number nor an answer line for the old round.

While a scheduled round is unresolved, participant output names neither the kind nor the operation. `status` prints `Incident: <n>` instead of `Exercise: <kind>`, omits the FLR and EFA-path lines, and lists the same collectable checks for every kind. Refusals pass through the same concealment, the operation transcript is kept in the root-only state directory instead of the results directory, and `start` or `active` with either kind gets one identical refusal. Once the round reaches `runtime-ready`, `verified` or `replacement-required`, `status` adds one answer line, for example `Answer for incident 4: the lab controller reset one EFA device on your exercise node (a PCIe function-level reset of 0000:b0:00.0) while your job was running.` Recovery qualification results after `recover` still name check numbers, which point to the kind, so a page that asks for a written diagnosis asks for it before `recover`.

The participant launcher below applies the profiles of the `storage`, `slow-job` and `efa-reset` incidents. The `dataloader` and `silent-corruption` workload modes are not in this version, so those two profiles are handed out and nothing applies them.

#### The participant launcher, job.sh

`job.sh` is the command a participant types for every job of the hour. It runs the job that the current incident's profile describes and prints none of the profile's settings, so the participant meets each incident as an ordinary job that fails.

```bash
bash job.sh notes      # create ~/aim344-results/notes.md, the notes file the pages fill in
bash job.sh run        # run the training stand-in, then the all-reduce sweep, on the healthy pair
bash job.sh incident   # ask the lab controls for the next incident and run its job
bash job.sh rerun      # run the current incident's job again, as the pair is now
bash job.sh resume     # undo what the lab changed in the current incident and resume the job
```

Every command except `notes` runs inside the participant's two-node allocation. Before it starts anything, the launcher checks three things: the allocation's job name starts with `aim344-`, its walltime is 1 to 10 minutes, and the login profile has set `AIM344_EFA_DEVICE`. The lab controls act only on a job that meets the first two conditions. Every run is held to all three, so a refusal does not single out one incident. Results go to `RESULTS_DIR`, which defaults to `/var/tmp/aim344-<user>-<job-id>` on each node. Every run of the training stand-in described below also writes `~/aim344-results/device-<job-id>.log` on the login node, the copy that the controller reads before an active device fault, so the copy's presence does not point to that incident either. Run ids, and the result files named after them, are `job-<time>`, `incident-<n>-<time>`, `incident-<n>-rerun-<time>` and `incident-<n>-resume-<time>`. The launcher prints the job log's path before and after the run, then `launcher_exit_code=` and `job_end_utc=`. Arguments it does not take are refused.

The training stand-in is `workload.py device --checkpoint-every=200`. In each step rank 0 broadcasts whether the job continues, and every rank runs one checked 128 MiB all-reduce. Rank 0 prints a progress record every 20 collectives in the format the controller parses. One rank, the checkpoint writer, saves the next step index to its node's private checkpoint directory when a job starts from step 0 and before the all-reduce of every 200th step, and prints `checkpoint saved: next_step=<n> (step index)` after each save. A save that fails with `ENOSPC` is retried once a second for up to 60 seconds, so the writer stays out of an all-reduce that the other ranks have entered. `job.sh run` runs the stand-in for 60 seconds, with rank 0 as the writer and the EFA device in `AIM344_EFA_DEVICE` on each node. When the stand-in succeeds, `job.sh run` then runs the 8 B to 2 GiB sweep that the `slow-job` incident runs, over every EFA device of each node, so the healthy and the incident output differ only in what the lab changed. The sweep's run id is `job-<time>` as well, and its files are `.nccl.log`, `.status.log` and `.counters.log`.

`job.sh incident` sends one `next` request for the incident itself, reads the profile the reply names, and runs that exercise:

| Exercise | `job.sh incident` | `job.sh rerun` | `job.sh resume` |
|---|---|---|---|
| `storage` | The stand-in for 60 seconds with the profile's `writer_rank` as the writer, whose directory is filled just before its first periodic save | The stand-in from the writer's last checkpoint, with no new fill. It fails the same way while the directory is still full | Removes the filler and the partial write with `recover-checkpoint.py`, then runs the stand-in from the last checkpoint until 20 steps after its first periodic save past that step (`workload.py --end-after-save`), for at most 60 seconds. From step 0 that is 220 collectives, with one save at step 200 |
| `slow-job` | Hides the OFI plugin in the allocation's private NCCL container, then runs the 8 B to 2 GiB sweep over every EFA device of each node | The sweep again, with the plugin still hidden | Restores the plugin, then runs the sweep |
| `efa-reset` | The stand-in for 120 seconds. Once the copy holds two progress records, the launcher asks `next` again until the reply says the incident is under way | The stand-in again, with no request to the lab controls | Refused, because `12.device-exercise.sh recover` is this incident's recovery |
| `dataloader`, `silent-corruption` | Refused, with a pointer to the page's expected output | Refused | Refused |

For `efa-reset` the launcher asks at most 10 times, 3 seconds apart, while the job runs, and never cuts off a request that is under way. The first refusal ends the requests, because a refusal that comes after the reset was dispatched still means the fault may have been applied. The requests come from a helper process beside the job. It ends when the job ends, also when the launcher exits early (an unset `TORCH_IMAGE` or `CHECKPOINT_DIR`, no node address) or receives TERM, and it checks once a second that the launcher still runs, so a killed launcher leaves no helper that a later job in the same allocation could make ask. `rerun` and `resume` reuse the profile of the incident issued last and never send `next`. The hidden plugin stays hidden in that allocation's container until `job.sh resume` restores it, and a later allocation creates a new container, so the fault does not carry into the next incident.

The launcher checks the allocation before the first step and starts each container with the first step that uses it: the NCCL container with a sweep or a lab step, the torch container with a torch job. A torch-only command therefore leaves no NCCL root on the nodes. `job.sh run` and `job.sh resume` end by running [`9.cleanup.sh`](9.cleanup.sh), which removes this allocation's containers from both GPU nodes, whatever the job's exit status. `incident` and `rerun` keep them, because the `slow-job` incident's change lives in its container until `resume`. The roots of an allocation that ends without `run` or `resume`, such as the `efa-reset` incident's, stay on the nodes until an operator removes them. A cleanup failure prints one line that asks the participant to tell the table expert and does not change the job's exit status.

The lab's own lines stay out of the participant's terminal and job log. They go to a lab log beside the profile, `~/aim344-results/incidents/incident-<n>-lab.log`: the controller's replies, the plugin action and the inventory of the hidden-plugin sweep, the output of the checkpoint recovery, the writer's injection line and the container removal. The healthy run's inventory and container removal go to `~/aim344-results/incidents/job-lab.log`. `workload.py --mark-lab-lines` prefixes the injection line with `AIM344 lab: `, and [`common.sh`](common.sh) passes the job's output through [`lab-lines.py`](lab-lines.py), which appends marked lines to the lab log and passes every other line on as it arrives. The filler is named `ckpt-shard-00017.partial`, a name that an interrupted save could leave. The cause stays out of what the participant's commands print, but the profile, the lab log and `job.sh` itself remain readable by the participant, and the pages ask participants not to read them before writing a diagnosis. The operator reads the lab log in the participant's home directory, and the schedule with `--operator status`.

#### The login profile

[`facilitator/participant-profile.sh`](facilitator/participant-profile.sh) lets a new shell on the login node run the lab commands with no setup. The installer installs it when it is given the seat's handout values:

```bash
sudo AIM344_SCHEDULE_FILE=/root/seat-1.json AIM344_PEER_NODE=<peer-node> \
     AIM344_EFA_DEVICE=<target-efa-rdma-device> \
     AIM344_SLURM_BIN=/opt/aws/pcs/scheduler/slurm-25.11/bin \
     bash facilitator/install-participant-control.sh \
     <assignment-id> <participant-user> <target-node> <target-instance-id> <target-host>
```

`AIM344_EFA_DEVICE` is the target's `efa_rdma_device` from `/etc/aim344-device-fault.json` on the target, the device that the `efa-reset` incident resets. The installer checks both values before it changes anything. It also refuses to install the profile when `$AIM344_SLURM_BIN/salloc` is not an executable file (the default directory names Slurm 25.05, so a cluster that runs another version needs the variable), and when the lab directory, `facilitator/`, the profile, `facilitator/allocation-env.py` or an existing `lab.env` is not owned by root or is writable by its group or by others. Each directory on the way to the lab directory must be owned by root and not writable by others, unless it is sticky like `/tmp`. Every participant's `~/.bashrc` sources these files, so only root may be able to change them. It writes them, with the target node and the Slurm client directory, to `/etc/aim344-seats/<participant-user>.env` (root, `0644`), and adds one line to the participant's `~/.bashrc` that sources the profile. Without the two values it installs no profile and says so. The first interactive shell of a session loads `lab.env`, puts `/opt/amazon/efa/bin` and the Slurm client directory on `PATH`, sets `PARTITION`, `GPUS_PER_NODE`, `CPUS_PER_RANK` and `MEMORY_PER_NODE` with [`facilitator/allocation-env.py`](facilitator/allocation-env.py), and exports `AIM344_TARGET_NODE`, `AIM344_PEER_NODE` and `AIM344_EFA_DEVICE`. A shell that starts in the home directory moves to the lab directory. Shells started from it, such as the one `salloc` opens, inherit these values, and a shell inside an allocation also exports `RESULTS_DIR=/var/tmp/aim344-<user>-<job-id>`, the directory where the job's processes write on each node.

[VALIDATION.md](VALIDATION.md#november-2-2026-the-participant-launcher-and-the-cause-hidden-job-incidents) records what has been tested.

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

The numbered storage and DataLoader scripts launch their workers through Slurm from the login allocation shell. After storage cleanup and `exit`, submit `13.verify-after-recovery.sbatch` from outside any allocation and retrieve its results. Only after verification and retrieval finish, take a fresh allocation with the baseline resource flags and a 20-minute time limit for the DataLoader controls. Set `RESULTS_DIR=/var/tmp/aim344-$(id -un)-$SLURM_JOB_ID` in each allocation to retain writable coordinator-side logs. The historical GPU-host `10.healthcheck-nccl.sh` procedure is not the current independent-login verification route.

The injector writes at most 64 MiB into a private 32 MiB tmpfs and refuses to arm the fault if that limit does not exhaust it. Its writer retries for up to 60 seconds while peers wait at a collective with a 30-second deadline. Preserve the earlier `ENOSPC` or quota error, the later application outcome and the launcher status. Wait for failed workers to exit before recovery.

Both storage scripts take one option, `--writer-rank N`. It moves the initial save, the read of `last.json` on resume, the source of the start-step broadcast, the fill and the retried save to rank N; ranks 0 to 7 run on the first node and ranks 8 to 15 on the second. The scripts refuse any other argument before they use the allocation, and the numbered launchers that take no arguments refuse every argument, so a mistyped option fails instead of running the unchanged exercise. The verification job's storage step uses the default writer, rank 0, on the first node, and the job writes a small file on each node as well, so a directory that a writer on the second node filled fails `storage=` even though the workload completes. The injector names its filler `ckpt-shard-00017.partial`.

Recovery removes only the lab filler and partial write, under the current name and under the earlier name `aim344-filler.bin`. It retains `last.json`, which stores a next-step index for this fixture, then resumes and checks the collective result. This file is not a complete model or optimizer checkpoint. The exercise reproduces local write exhaustion; an FSx outage or Lustre quota needs separate qualification.

## Late DataLoader workers

Run both controls in the healthy allocation after storage recovery:

```bash
fork_rc=0
bash 7.probe-dataloader-fork.sh || fork_rc=$?
printf 'fork_return=%s\n' "$fork_rc"
```

Stop here and confirm the fork step has drained before running the next block
in the same allocation shell. A launcher timeout alone does not prove remote
worker exit; if draining is uncertain, collect evidence rather than overlap controls.

```bash
spawn_rc=0
bash 8.use-dataloader-spawn.sh || spawn_rc=$?
printf 'spawn_return=%s\n' "$spawn_rc"
((fork_rc == 0 && spawn_rc == 0))
```

Both scripts explicitly select `--epochs 2`; the direct `workload.py --epochs` default remains 1 (positive integers only). They reuse one CPU DataLoader, explicitly set `persistent_workers=False`, and exhaust each iterator before creating the next epoch's workers. Each rank checks four batches of eight sequential CPU values per epoch and performs eight parent collectives total, in addition to the initial collective. No worker uses CUDA. Both scripts pass further arguments to `workload.py`, which refuses `--writer-rank`, `--inject`, `--resume`, `--mark-lab-lines` and `--checkpoint-every` in dataloader mode, so a mistyped option fails instead of running the unchanged control.

Flushed `DataLoader lifecycle:` JSON lines identify rank, host, PID and epoch at iterator start/creation, batch receipt, collective completion and iterator exhaustion; worker-init lines add worker ID and parent PID. Require two complete epochs, four batches each, two new worker identities per epoch and rank, eight completed batch collectives, zero mismatches and launcher exit 0. Iterator exhaustion is the natural nonpersistent shutdown boundary; a missing final marker is not completion.

Both scripts hold `FI_EFA_USE_HUGE_PAGE=0` by default and vary the worker-start method. They retain the initialized parent CUDA allocation, complete a collective before creating workers, and read the C environment's fork-safety flag. Fork-protection settings are unchanged; the OFI plugin may set `FI_EFA_FORK_SAFE=1`. The existing 180-second TERM/30-second kill-after launcher watchdog and 30-second process-group timeout are unchanged. Keep actual exit statuses: 124 (timeout), 137 (SIGKILL, possibly kill-after), exceptions and natural exit 0 are distinct; none alone proves a second-epoch deadlock. Capture both control results even in a `set -e` wrapper, but do not start spawn until the fork step has drained.

Keep the kernel, PyTorch, runtime NCCL, libfabric and plugin versions with the result. If both methods complete, record expanded non-reproduction on that workload and stack, not a repaired/reproduced hang. Do not force a failure with worker CUDA, artificial locks or disabled fork safety. Obtain the actual application's dataset/decoder and worker/iterator settings before extending synthetic stress. Local lifecycle tests require CPU PyTorch: `python3 -m unittest discover -s tests -p 'test_dataloader_epochs.py' -v`; these mock parent NCCL/CUDA and do not qualify GPU/EFA behavior.

Forked children can inherit thread and lock state from an initialized parent. [PyTorch's multiprocessing guidance](https://docs.pytorch.org/docs/2.6/notes/multiprocessing.html) explains those deadlocks and requires `spawn` or `forkserver` for CUDA in subprocesses. The DataLoader fixture uses CPU data in its workers; its successful completion does not qualify arbitrary CUDA work in forked children.

## Optional pre-job marker and plugin fallback

The synthetic marker demonstration uses `1.inject-prejob.sh` and `2.recover-prejob.sh`. Run it with no interactive allocation occupying the pair. Match the Prolog journal to the drained node and any held requeue; the marker explains the scheduler transition and does not diagnose hardware. Preserve and restore node weights if they are changed to make the probe deterministic.

The optional plugin comparison uses `0.baseline.sh`, `3.inject-fallback.sh` and `4.recover-fallback.sh` in the same allocation. It covers 8 B through 2 GiB with 1 MPI rank per GPU. The injection temporarily hides the OFI library directory in private writable containers, including network and tuner libraries. A Socket selection, correct arithmetic and flat EFA counters together identify the observed fallback. These measurements have a different message range and launch mapping from Check 5, so retain separate baselines. The same fallback without its cause in the participant's output is the `slow-job` incident of [`job.sh`](#the-participant-launcher-jobsh).

## Cleanup and known edge cases

Run `bash 9.cleanup.sh` in the healthy allocation, then exit it. Retain the evidence. Only during post-session administrative teardown does the facilitator restore exercise-owned markers, mounts, telemetry and scheduler settings while retaining the dedicated pair and capacity; this is not an in-session device-recovery remedy. Participants use `recover`, and an operator runs the replacement route after the session. Remove only this exercise's private container instances; keep the staged images available for reuse.

A launcher or image error must be resolved before its result can describe GPU communication. Use `FI_EFA_IFACE` to select a physical EFA; `FI_EFA_DEVICE_NAME` is ignored by the shipped libfabric. Missing counters require investigation and must not be recorded as 0 B. `iptables` and `tc netem` do not establish an EFA device failure because the EFA data path bypasses that kernel networking path.

For a known administrative device removal, preserve the suite severity and keep the node drained through recovery and qualification. `ISOLATE` alone does not require replacing the instance. For an unexplained recurring fault, retain the raw evidence and follow the platform's repair or replacement process. Use the [failure-signature runbook](RUNBOOK.md) to record the decision.
