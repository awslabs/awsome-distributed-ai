# GPU and EFA device recovery

The diagnostic owner is [validation/gpu-cluster-healthcheck](https://github.com/awslabs/awsome-distributed-ai/tree/main/validation/gpu-cluster-healthcheck). The coordinator applies its verdicts; it does not substitute a hardware diagnosis. GPU PCI removal changes operating-system visibility, not physical hardware, and does not establish NVIDIA Xid 79. EFA driver unbind changes one PCI function and RDMA domain, not an NCCL plugin.

Independent source/evidence review accepts the measured participant recovery/replacement and fresh workload flow on the dedicated pair. Retained source, payloads and results were checked byte-for-byte. **Full hardware qualification remains withheld.** The lab controls now run for the cluster's own `ubuntu` account through `sudo`, and that route has local tests only. Candidate publication is separate from hardware or event-readiness approval. Local recording-executor tests are not cloud replacement evidence; see the [scoped audit record](../VALIDATION.md#september-19-2026-scoped-participant-recovery-audit). This document authorizes no new deployment, fault injection or rollout.

## Participant session

Work as the cluster's own `ubuntu` account on the login node. `12.device-exercise.sh` passes each request to the controller with `sudo -n`, and the controller's configuration names the one target. A request takes no node, instance, PCI, job or shell argument. For a fault during a running workload, follow the [active FLR procedure](#active-flr-participant-integration-candidate), using `active gpu` or `active efa`. The commands below start an idle round:

```bash
./12.device-exercise.sh start gpu
./12.device-exercise.sh status
./12.device-exercise.sh collect 0
./12.device-exercise.sh collect 3
./12.device-exercise.sh recover
```

For the separate idle EFA round, use `start efa` and checks `2` and `6`. `start gpu` and `start efa` require an idle target, so end the previous allocation before either start. Active FLR instead requires your running device allocation; keep its coordinator-side log open and preserve fault-time logs before recovery.

Supported `recover` is the first route. GPU recovery requires coherent operation-bound original records, a valid recorded reboot baseline, and fresh scheduler/boot observations. Successful EFA rebind does not reuse an old communicator. Failed EFA rebind does not escalate into a reboot without originals; it requires replacement. Missing originals are not reconstructed from an already-modified node. Known statistics-skipped PASS refuses regardless of baseline; WARN auto-qualification is disabled, including source-shaped memlock advisory matches. The one exception is a Check 6 WARN whose only warnings are the cumulative `rx_drops` and `retrans_timeout_events` totals, after a passing loopback test, which recovery records as MONITOR ([Checks a participant can collect](../README.md#checks-a-participant-can-collect)).

When recovery cannot qualify the node, the node stays drained for the rest of the session, and the controller records the round as `replacement-required`. The participant verb `replace` is refused without changing anything. After the session, an operator replaces the node in the ordinary way. Terminate the instance, let PCS launch its successor under the same Slurm name, and prepare the successor with the GPU-node steps of [Pre-session provisioning](#pre-session-provisioning--separate-authorization). Then rerun step 3 on the login node with the successor's instance ID and address and without `AIM344_SCHEDULE_FILE`, so that it keeps the seat's schedule. The controller's configuration holds the target's instance ID and address, and the installer is what writes them. Last, remove the seat's state record, `/var/lib/aim344-device-session/<assignment>.json` on the login node, so that the next round starts from `ready`. An event account is deleted after the session, so this matters for a reused rehearsal account. On the login node the operator can read the seat as root:

```bash
sudo /usr/local/sbin/aim344-device-session --operator status
```

It shows the exercise kind, the seat's schedule with each entry's settings, its issue time and, for a job incident that has ended, its end time, the path of the device-operation transcript and the original text of the last recovery refusal.

Drain ownership keeps the complete Reason body and its raw annotation for audit. The coordinator accepts exactly the configured reason, or that reason plus Slurm 25.05.9's exact ` : reboot issued` suffix only with this assignment/round's durable reboot request, operation and matching isolated scheduler identity. The suffix is not stripped by the shared parser. Fault injection and the initial start remain exact-only, so neither inherits a reboot authority. Unknown suffixes and field-shaped Reason text (including `Comment=`) remain foreign.

The controller's maintenance hop to the target uses pre-pinned per-instance host trust.

Runtime readiness is not workload verification. After recovery, submit the existing verification job from a fresh allocation:

```bash
source /opt/aim344/device-recovery/companion/lab.env
export PATH=/opt/amazon/efa/bin:/opt/aws/pcs/scheduler/slurm-25.05/bin:$PATH
command -v sinfo salloc sbatch srun
unset AIM344_EFA_IFACE FI_EFA_IFACE
sbatch --wait --export=ALL --partition="$PARTITION" --cpus-per-task="$((GPUS_PER_NODE * CPUS_PER_RANK))" /opt/aim344/device-recovery/13.verify-after-recovery.sbatch
```

Check 5 runs once from the allocation coordinator. Its per-UID, per-job named container is initialized on both nodes before the collective timers, with `AIM344_CONTAINER_PREP_TIMEOUT` (default 600 seconds). Isolation/main reuse that root, and storage reuses it only when its image matches. No prior job or participant cache is required. The default 25-minute allocation remains the outer limit. In the September 24 fresh post-replacement trial, the allocation completed in 51 seconds including 6 seconds of job-local cold preparation; this is a measured run, not a general cold-start guarantee. Check 5 uses include-unset/exclude-only MPI configuration and Pyxis container-env precedence overrides scoped to its own steps, not host counters or storage. Keep both correctness columns, every allocated node's per-device EFA deltas and the storage workload result. The verification script checks all allocated nodes, not just whichever rank's log is visible locally, and copies results to each allocated node with digest verification. A replaced node's preparation initializes the isolated 32 MiB fixture with step 0; that seed is not recovered model/optimizer state. No instructor hand-restoration or facilitator-arranged replacement is a session remedy.

After completion, retrieve the batch-host log and run its printed fresh-`srun`, digest-verification, extraction and summary commands. The main numeric rows in the terminal log omit the full rank headers and separate isolation capture. With `job` set to this verification job's identifier, inspect the extracted files:

```bash
read -r -p 'Verification job identifier: ' job
raw_dir="$HOME/aim344-verify-$job/verify-$job/check5"
less "$raw_dir/nccl-allreduce-raw.txt"
less "$raw_dir/nccl-efa-only.txt"
```

Require 25 main rows from 8 B through 128 MiB and one EFA-only 256 MiB row. In each raw file, inspect all 16 GPU rank headers across two MPI processes, both zero-mismatch columns and zero out-of-bounds values. Keep the job log, extracted summary, per-device counter results and provider/performance WARN separately. These are default verifier paths; a site override of the run name changes the inner directory printed by the job.

## Pre-session provisioning — separate authorization

Provisioning finishes before the participant receives the account. Each account holds one cluster: an independent login node, which runs the controller, and two GPU nodes, one of which is the exercise target. The participant is `ubuntu` on every host, with the same UID and GID. Run each step as root on the host it names. The lab directory is `/opt/aim344/device-recovery/companion` on all three hosts.

1. On the login node, build or reuse the two images and the pinned health-suite archive with `facilitator/prepare-login.sh` ([Suite acquisition](#suite-acquisition)). Install the companion tree, root-owned, as the lab directory on all three hosts, and copy `13.verify-after-recovery.sbatch` to `/opt/aim344/device-recovery/` on the login node.
2. On each GPU node, put `aim344.sqsh`, `nccl-baseline.sqsh` and `healthcheck-pinned.tgz` in `/opt/aim344` on the root filesystem. Then run `facilitator/prepare-compute.sh ubuntu` and `facilitator/prepare-prolog.sh <partition>`, both with `AIM344_NODE_LOCAL=1 AIM344_STAGE_DIR=/opt/aim344`. `prepare-compute.sh` installs the pinned suite, the health Prolog, the device helper and runtime restoration. It also keeps the enroot directories shared at every boot, mounts the 32 MiB checkpoint fixture for `ubuntu` and reads both images into the page cache. Configure the cluster's Prolog as the dispatcher that `prepare-prolog.sh` writes, `/opt/aim344/.prolog/dispatch.sh` ([Facilitator preparation for the PCS health gate](../README.md#facilitator-preparation-for-the-pcs-health-gate)).
3. On the login node, run `facilitator/install-participant-control.sh` for `ubuntu` with the target's Slurm name, instance ID and address, and with `AIM344_PARTITION`, `AIM344_SUITE_ENTRY`, `AIM344_SLURM_BIN` and the seat's `AIM344_SCHEDULE_FILE` ([Scheduled incidents](../README.md#scheduled-incidents)). It prints the `prepare-target.py` command for step 4, with the contents of `/root/.ssh/aim344-maintenance.pub` as its key, and the login profile line.
4. On the target, run the command that step 3 printed, `facilitator/prepare-target.py <target-node> ubuntu '<maintenance public key>'`, with `AIM344_SLURM_BIN` and, for an assignment that names its EFA count, `AIM344_EXPECTED_EFA_DEVICES`. It writes `/etc/aim344-device-fault.json` and `/etc/aim344-maintenance.json` for `ubuntu`, pins both staged images by SHA-256 under the `root-directory` staging policy, installs the maintenance helper behind root's forced command, and prints the device inventory and the node's SSH host key. Add `<target-host> <host key>` to root's `~/.ssh/known_hosts` on the login node, then check the route from there with `ssh -n -o BatchMode=yes -o StrictHostKeyChecking=yes -i /root/.ssh/aim344-maintenance root@<target-host> -- inspect`.
5. Generate `lab.env` with `facilitator/lab-env.py` and install it in the lab directory on all three hosts. Pass `PARTITION`, `CPUS_PER_RANK` ([Assigned PCS CPU budget](#assigned-pcs-cpu-budget)), `AIM344_EXPECTED_EFA_DEVICES` when the assignment names it, `NCCL_SOCKET_IFNAME` from the management route (keep a leading `=` for an exact interface) and `NCCL_PROTO=Simple` on g7.48xlarge ([Check 5 protocol on G7](#check-5-protocol-on-g7)). Pass the seat's values as well: `AIM344_TARGET_NODE`, `AIM344_PEER_NODE`, `AIM344_EFA_DEVICE` (the `efa_rdma_device` that step 4 printed) and `AIM344_SLURM_BIN`. Each line keeps a value that the caller's environment already sets. The generator sets neither `PATH` nor a Check 5 bandwidth floor.
6. As `ubuntu` on the login node, add the login profile line that step 3 printed to `~/.bashrc`. In a new shell, `command -v sinfo salloc sbatch srun` resolves the four clients, and `bash 12.device-exercise.sh status` prints `Exercise phase: ready`.

Before the session, run `sudo -n true` as `ubuntu` on the login node: the lab controls run through it. Verify the enroot directories, image paths, fixture ownership and the Prolog gate on both GPU nodes. A missing `fi_info` on PATH, a private `/tmp/enroot`, mismatched UID or an invalid staging source is an environment failure, not device-fault evidence.

`prepare-compute.sh` writes `/etc/tmpfiles.d/aim344-enroot.conf`, so systemd-tmpfiles keeps `/tmp/enroot` and its `cache` and `data` directories root-owned with mode 1777, at preparation and at every boot. A first enroot run as root therefore leaves them open to `ubuntu`.

Recovery checks the staged images before it restores the runtime. Under the `root-directory` policy that `prepare-target.py` writes, `/opt/aim344` is a root-owned directory on the root filesystem. Recovery refuses an overlaid staging mount and checks both images against their SHA-256 pins, so a missing, changed or overlaid source refuses recovery. It never binds `/opt/dlami/nvme/aim344` over them. A target configured with the older `nvme-bind` policy keeps its source and UUID checks. Size the root volume for the pinned images and the companion tree.

### Images in the page cache

A container start reads its image from the root volume unless the image is already in the page cache. On the g7.48xlarge pair, the first torch container on a node that had not read the image since its reboot took 168 s to create, and the launcher's 180 s limit ended the job before its first progress line. With the image cached, creation took about 5 s. Supported recovery leaves both images cached on the target, because its staging check hashes them in full. Nothing reads them on the other node, or on any node after a reboot or replacement that does not go through supported recovery.

After each GPU node's last boot, and after any such reboot or replacement, read both images once on every GPU node, as root and on both nodes at the same time:

```bash
python3 /opt/aim344/device-recovery/companion/facilitator/image-cache.py read
```

Each node reads its own copy of the two images (22.26 GB), which at the root volume's 125 MiB/s takes about 170 s. Then, among the checks before participant access, run `image-cache.py check` as root on both nodes. It asks `mincore` which pages are resident and reads no data, so the check itself cannot warm a cold image. For a file that the caller neither owns nor can open for writing, `mincore` reports every page as resident, so the helper refuses such a caller with one line on stderr and exits 1. It must report every page of both images resident and exit 0; otherwise run `read` on that node again.

### Assigned PCS CPU budget

Pass `CPUS_PER_RANK` to `facilitator/lab-env.py` from the assigned scheduler CPU budget. For example, eight GPUs on a node with `CPUEfctv=96 CPUTot=96` and SMT1 require the string `"12"`. The default remains `"24"` for legacy
assignments. Values must be canonical positive decimal strings (`[1-9][0-9]*`);
zero, signs, leading zeros, nonintegers and shell expressions are rejected.
There is no new hardware ceiling: the owner must match the value to the assigned
scheduler capacity. This changes CPU resources, not eight GPUs per node, all
16 GPU ranks, correctness, EFA counts or the required bandwidth qualification.

Regenerate `lab.env` on all three hosts after changing the value. The value is
a participant launch default, never privileged maintenance authority; job
environment does not override the root configuration.

Source the generated `lab.env` on the coordinator before submission and pass
`--cpus-per-task="$((GPUS_PER_NODE * CPUS_PER_RANK))"` to both `sbatch` (above)
and `salloc`. The batch directive stays at 192 for legacy compatibility, so
sourcing defaults inside the job alone cannot repair an oversized request.
For this assignment the explicit request is 96 CPUs per node, one task per node,
two nodes and `--gres=gpu:8`. Native Slurm remains the capacity validator.

Inside the allocation, the verifier and existing native entry helpers use
`SLURM_JOB_CPUS_PER_NODE` (homogeneous `96(x2)` or `96`), falling back to
`SLURM_CPUS_ON_NODE` when absent. They replace stale CPU hints with that budget,
not `GPUS_PER_NODE * 24`. Explicit per-node `srun -c` uses 96; the optional
one-MPI-rank-per-GPU sweep uses 12. `common.sh` also sets Slurm's native
`SRUN_CPUS_PER_TASK` default so unflagged preparation, Check 5 and retrieval
steps use the allocation. This is not a scheduler helper or a change to planned
foreign-G7 paths.

Check the selected suite's shared `launch_args` before qualification. An explicit `-c` takes precedence over `SRUN_CPUS_PER_TASK`; both main and isolation launches must use the allocated CPU budget while preserving all GPU ranks and correctness, network and performance checks. Keep the suite pinned and do not fork a local copy to override its resource request.

### Assigned EFA count on G7e

For an explicitly two-interface `g7e.48xlarge` assignment, pass `AIM344_EXPECTED_EFA_DEVICES=2` to `facilitator/lab-env.py`, and set the same variable for `facilitator/prepare-target.py`, which writes it into the root-owned maintenance configuration. Maintenance and the health Prolog use the root-owned count, not participant overrides. Without the assignment, the G7e profile expects four interfaces. Keep the matching suite and assignment together when preparing an accepted release.

### Check 5 protocol on G7

For a `g7.48xlarge` assignment, pass `NCCL_PROTO=Simple` to `facilitator/lab-env.py`, which writes it into `lab.env`, and `run_check5` passes it into the Check 5 container together with `OFI_NCCL_PROTOCOL`. The torch and sweep launchers set the same value from the detected instance type, but Check 5 does not run that detection, so the manifest is how the value reaches it. Without it, NCCL runs Check 5's 1 to 8 MiB rows as LL at 0.25 to 0.30 GB/s on this instance. Do not set `^LL`: an exclusion list enables LL128 over PCIe. The measurement is in [VALIDATION.md](../VALIDATION.md#november-3-2026-the-replica-check-after-an-aborted-collective-the-nccl-protocol-on-g748xlarge-and-both-images-in-the-page-cache).

### Suite acquisition

The exact `HEALTHCHECK_COMMIT` in `pins.env` identifies a review candidate based on historical public commit `474155a63059769bcea6365d4ec2b2f34d4c8000`. This revision does not establish merge, release, hardware qualification or public availability. For local source review, explicitly supply the clean healthcheck checkout as `HEALTHCHECK_SOURCE`. After separate publication approval and verification that the exact full SHA is reachable from an approved public ref, the publication owner may instead set `HEALTHCHECK_SOURCE=https://github.com/awslabs/awsome-distributed-ai.git` and use the same procedure unchanged. Do not substitute main or the historical base for the pin. Deployment still requires separate approval; any accepted release needs new archives and manifests, not historical archive digests.

The matching suite treats missing `instance-profiles.conf` as a fail-closed packaging error. That intentional change is separate from preserved profile-only warnings; an unknown profile with the file present and no assignment retains the existing defaults.

From the companion directory in the owned Git checkout:

```bash
set -euo pipefail
source pins.env
repo=$(git rev-parse --show-toplevel)
: "${HEALTHCHECK_SOURCE:?Supply a clean local checkout or the publication-verified canonical repository}"
git -C "$repo" fetch --depth=1 --no-tags "$HEALTHCHECK_SOURCE" "$HEALTHCHECK_COMMIT"
test "$(git -C "$repo" rev-parse FETCH_HEAD)" = "$HEALTHCHECK_COMMIT"
git -C "$repo" cat-file -e "$HEALTHCHECK_COMMIT^{commit}"
git -C "$repo" archive "$HEALTHCHECK_COMMIT" validation/gpu-cluster-healthcheck | gzip -n -6 > healthcheck-pinned.tgz
```

These commands populate local objects/FETCH_HEAD without switching branches and create only the named archive; they do not publish or deploy it. `prepare-login.sh` compares staged suite contents and executable modes against this exact Git archive. Build/import steps remain separate pre-session work.

Create `companion.tgz` from the separately accepted companion commit. Set `COMPANION_COMMIT` to its full accepted commit SHA before running this block; do not infer acceptance from the current branch tip. Run from any directory inside that Git checkout (including the companion directory). Git must support `archive --mtime`. The archive is written to the caller's current directory:

```bash
set -euo pipefail
: "${COMPANION_COMMIT:?Set the full accepted companion commit SHA}"
repo=$(git rev-parse --show-toplevel)
test "$(git -C "$repo" rev-parse "$COMPANION_COMMIT^{commit}")" = "$COMPANION_COMMIT"
epoch=$(git -C "$repo" show -s --format=%ct "$COMPANION_COMMIT")
git -C "$repo" -c tar.umask=0022 archive --format=tar --prefix=companion/ \
  --mtime="@$epoch" "$COMPANION_COMMIT:examples/use-cases/gpu-cluster-failure-patterns" \
  | gzip -n -6 > companion.tgz
sha256sum companion.tgz
```

The explicit repository root prevents subdirectory filtering; the commit timestamp prevents tree-object archives from using generation time. The fixed tar mode mask and gzip level/header settings define the release recipe. Record Git/gzip versions and the commit alongside both archive digests. Verify the complete companion file set and modes against the accepted tree before approval, not just command exit status. Retain the reviewed archive bytes as the release artifact; if regeneration differs, stop and obtain review of the new bytes rather than reusing a prior hash. Never use the entire original development branch as a release payload. Mainline integration must retain the ownership of KeitaW's Check 6 change from [PR #1221](https://github.com/awslabs/awsome-distributed-ai/pull/1221) and Nathan Na's result-severity changes from [PR #1270](https://github.com/awslabs/awsome-distributed-ai/pull/1270). The latter's original author metadata and commit references are preserved in the review branch. The remaining Kubernetes changes from #1221 are not included here. Neither existing PR branch was rewritten; feature-branch availability is not mainline merge.

### Deployment-specific materials and protected peer

Site values, object locations, instance and image identifiers and historical source/evidence mapping remain with the workshop owner, outside distribution. Do not reuse an old deployment's pins for the accepted candidate. The source package does not provision infrastructure or confer access to private image objects. Build images using the pinned Dockerfile or obtain explicitly approved digest-pinned images.

The September 19 measured protected peer retained older maintenance code with a deliberate DMI suite/verifier overlay. It was a workload peer, **not** a final-helper-equivalent replacement target. Do not describe the historical pair as a homogeneous final-helper deployment. Exact retained evidence remains unchanged outside this public candidate.

## Evidence, retries and qualification boundaries

For the unapproved 60-minute delivery proposal, assign one live branch per table. The fixed EFA trial consumed 39 minutes 10 seconds from frozen start through retrieval; fault start to runtime readiness alone was 32 minutes 39.111 seconds. Allocate 40 minutes after the common 12-minute entry/baseline for the chosen branch, with 8 minutes for comparison and evidence completion. Device tables compare recorded storage, then DataLoader outcomes during waits without running those pages' commands. Storage tables execute storage recovery, fresh verification/retrieval and DataLoader controls, and compare recorded GPU/EFA evidence. Both routes continue through correctness coverage and the runbook. Preserve full additional branches for separately scheduled practice. This proposed schedule needs a first-time human rehearsal and content-owner agreement; machine timings do not establish that it fits. No instructor performs per-table rescue.

The EFA faulted job had detected correctness exceptions as well as all-rank watchdogs. The recovered fresh job had zero mismatches across 25 main rows and one EFA-only isolation row, 16 GPU ranks across two MPI processes, four active EFA rails, and successful 16-rank storage. Retain provider/performance WARN and later retransmission deltas. The corrected kernel window is tied to the EFA fault boot and cannot retroactively supply the missing earlier GPU fault capture.

<a id="active-flr-participant-integration-candidate-local-verification-only"></a>

### Active FLR participant integration candidate

`12.device-exercise.sh active gpu` and `active efa` use the existing assignment, forced-command maintenance connection and recovery state machine. `start gpu/efa` retain idle removal/unbind semantics; active requests never fall back to those operations. The fixed-candidate EFA participant path has independently reviewed hardware evidence through replacement, fresh verification and retrieval. The GPU participant attempt has incomplete reset-delivery evidence despite successful recovery. See the [latest validation record](../VALIDATION.md#september-24-2026-fixed-candidate-participant-efa-recovery). Administrator FLR trials remain separate from participant-path qualification.

Prepare terminals A and B on the same independent login coordinator before starting the workload. Run this in both terminals and compare the hostname, assigned participant username and UID:

```bash
hostname
id -un
id -u
cd /opt/aim344/device-recovery/companion
source lab.env
export PATH=/opt/amazon/efa/bin:/opt/aws/pcs/scheduler/slurm-25.05/bin:$PATH
command -v sinfo salloc sbatch srun
bash 12.device-exercise.sh status
```

Keep terminal B ready at its prompt, with no pending prior round. In terminal A only, take the dedicated two-node allocation and keep the workload terminal attached:

```bash
cd /opt/aim344/device-recovery/companion
source lab.env
salloc --job-name=aim344-device --exclusive --partition="$PARTITION" --nodes=2 --gres="gpu:$GPUS_PER_NODE" --ntasks-per-node=1 --cpus-per-task="$((GPUS_PER_NODE * CPUS_PER_RANK))" --mem="$MEMORY_PER_NODE"
export RESULTS_DIR=/var/tmp/aim344-$(id -un)-$SLURM_JOB_ID
# EFA branch only: confirm this physical rail against the prepared assignment.
export AIM344_EFA_IFACE=rdmap83s0
bash 11.device-workload.sh
```

For GPU, leave `AIM344_EFA_IFACE` unset. The selected EFA name above is from the measured G7 pair, not a portable device-name assumption. The launcher saves the combined output to `~/aim344-results/device-JOBID.log` on the login coordinator. Its timed workload runs for 120 seconds. The launcher gives the workload step a Slurm time limit of six minutes, which also covers its start and cleanup, so the allocation itself needs no time limit. In the already-prepared terminal B, wait for two advancing correctness-checked progress records, then issue exactly one `bash 12.device-exercise.sh active efa` (or `active gpu`). Use `status` and the allowlisted `collect <check>` commands afterward.

The control command accepts no node, job ID or BDF. The target checks the fixture process/cgroup, exact authenticated participant/job and selected-EFA byte movement including `rdma_write_bytes`. Before each mutation acknowledgement, the coordinator requires two advancing records for the exact job/world size, with the latest no older than 15 seconds. These records establish activity; scheduler ownership and target PID/cgroup checks establish authorization independently.

If the workload ends naturally before any active request was sent, preserve the log, exit the allocation, confirm the pair is idle, and repeat the allocation/workload step with terminal B ready. Once a request has been sent, an unknown outcome requires diagnosis and supported recovery, not another fault.

The target sends authorization and mutation-start records over the same SSH connection. Each must be synchronized to root-owned coordinator state before its hash acknowledgement permits the next target write. `reset_method` must read back exactly `flr`; there is no bus fallback. Start is evidence of intent, not a claim that reset returned. Both original persistence and telemetry states are captured without pausing the active workload, and restored after reboot.

If the observation deadline expires or the ACK connection fails, the coordinator retains the mutation attempt and any saved markers with an unknown outcome. Use `status`, `collect` and then `recover`; repeating `active` does not issue another fault. Only a local failure to start the transport establishes that no request was sent.

The device launcher no longer SIGKILLs its `srun`/log pipeline at 210 seconds. Slurm ends the workload step at the step's own time limit of six minutes (`srun --time=6`) while the existing log remains attached. GPU D-state cleanup can outlive both the NCCL watchdog and that limit. After the step ends, end the allocation with `exit`, and wait until the job has left the queue or shows COMPLETING before `recover`. Recovery refuses a RUNNING job and a different job even if owned by the same participant. Same-instance Slurm reboot is tried first. If the node has not returned when the configured grace ends, it stays out of service, and an operator replaces it after the session. No broad kill or instructor restoration is introduced.

`collect kernel` exports a bounded useful kernel window for the recorded fault boot: the last 512 matching GPU/EFA/PCI/reset/lost-message records, at most 524288 raw bytes and an 8-second read window. The payload includes instance/boot, byte count, SHA256, command return code and a bound-hit indicator, and explicitly does not claim a complete journal. `recover` attempts the same export before recovery, but target/journal/output failure never blocks recovery. The deployed correction uses the supported `journalctl -k` option and rejects a null boot UUID before collection. In the EFA trial, inner and outer collection statuses were both zero and the selected fault-boot window contained 512 records. Earlier GPU captures from the old collector remain unavailable. Prior-boot journal availability and events lost to kernel floods are not guaranteed. Collected suite/kernel outputs, mutation records and coordinator workload logs remain off-target. A wholly unresponsive guest and every stalled-target variant remain distinct from that measured path. Do not interpret FLR return or driver `reset done` as CUDA recovery, physical PCIe loss, Xid79, or a guarantee across repetitions.

State and dispatch intent are private coordinator files under the existing state directory. File synchronization and directory synchronization precede dependent operations. A directory-sync failure can occur after rename; refusal is not proof that no file was renamed. Neither local fsync tests nor mock API acknowledgements establish survival of a physical crash.

Participant captures remain append-only by round and attempt under `~/aim344-results/collected`. The output writer drops to the participant UID/GID and uses descriptor-relative operations. The checkpoint fixture is mounted with the participant's UID and GID, so the participant owns everything in it.

Manual pre-session experiments must not write bare persistence words to the participant helper's `selected-persistence-mode.txt` or `native-dcgm-state.txt`: those paths contain operation-bound JSON. Store standalone experimental records in a separate evidence directory. The participant helper authorizes fresh capture only on a newly generated trusted operation; retries synchronize and reuse retained originals rather than recapturing.

Historical active-GPU removal stalled guest shutdown. Active EFA unbind remained outstanding until its workload ended: the dedicated probe returned at t+171 s, 11 s after cancellation. These measurements explain why `start gpu` and `start efa` remain idle-only; `active gpu` and `active efa` use the separate FLR procedure above. They do not establish that every supported idle fault stalls. Historical replacement after EC2 reboot was not a trial of `scontrol reboot nextstate=DOWN`.

Earlier recovery-decision/implementation and executor-first reports remain historical evidence with the workshop owner. This runbook defines the current participant-remedy and WARN-acceptance contract. Passing named reproductions supports those exercised inputs, not all subrequirements; imported TestCase executions are regression reruns, not newly authored tests.

## Primary mechanism references

- [PCS reboot FAQ](https://docs.aws.amazon.com/pcs/latest/userguide/slurm-reboot-faq.html): ordinary Slurm reboot retains the instance; `nextstate=DOWN` requests replacement after reboot.
- [Slurm 25.05.9 node manager](https://github.com/SchedMD/slurm/blob/slurm-25-05-9-1/src/slurmctld/node_mgr.c): `_drain_node` and `validate_node_specs` preserve an already-drained node on Prolog failure. This is upstream source matched to the historical binary version, not a freshly measured deployed binary.

Health and a fresh participant job must be observed on the dedicated pair before hardware readiness is claimed. There is no cloud or publication approval in this document.
