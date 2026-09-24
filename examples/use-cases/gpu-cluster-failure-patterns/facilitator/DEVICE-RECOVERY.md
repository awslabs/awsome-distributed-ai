# GPU and EFA device recovery

The diagnostic owner is [validation/gpu-cluster-healthcheck](https://github.com/awslabs/awsome-distributed-ai/tree/main/validation/gpu-cluster-healthcheck). The coordinator applies its verdicts; it does not substitute a hardware diagnosis. GPU PCI removal changes operating-system visibility, not physical hardware, and does not establish NVIDIA Xid 79. EFA driver unbind changes one PCI function and RDMA domain, not an NCCL plugin.

Independent source/evidence review accepts the measured participant recovery/replacement and fresh workload flow on the dedicated pair. Retained source, payloads and results were checked byte-for-byte. **Full hardware qualification remains withheld** for missing first-hook-held failure/access-denial proof and unqualified IPv6. Candidate publication is separate from hardware or event-readiness approval. Local recording-executor tests are not cloud replacement evidence; see the [scoped audit record](../VALIDATION.md#september-19-2026-scoped-participant-recovery-audit). This document authorizes no new deployment, fault injection or rollout.

## Participant session

Use the unprivileged coordinator account and the existing forced-command connection. The authenticated key selects the assignment. There is no participant node, instance, PCI, job, shell or IAM option. For a fault during a running workload, follow the [active FLR procedure](#active-flr-participant-integration-candidate), using `active gpu` or `active efa`. The commands below start an idle round:

```bash
./12.device-exercise.sh start gpu
./12.device-exercise.sh status
./12.device-exercise.sh collect 0
./12.device-exercise.sh collect 3
./12.device-exercise.sh recover
```

For the separate idle EFA round, use `start efa` and checks `2` and `6`. `start gpu` and `start efa` require an idle target, so end the previous allocation before either start. Active FLR instead requires your running device allocation; keep its coordinator-side log open and preserve fault-time logs before recovery.

Supported `recover` is the first route. GPU recovery requires coherent operation-bound original records, a valid recorded reboot baseline, and fresh scheduler/boot observations. Successful EFA rebind does not reuse an old communicator. Failed EFA rebind does not escalate into a reboot without originals; it requires replacement. Missing originals are not reconstructed from an already-modified node. Known statistics-skipped PASS refuses regardless of baseline; WARN auto-qualification is disabled, including source-shaped memlock advisory matches.

When recovery cannot qualify the node, retain isolation and use:

```bash
./12.device-exercise.sh replace
./12.device-exercise.sh status
```

`replace` retires only the failed instance recorded for this assignment. It preserves a coordinator-side round snapshot and attempts to capture the two original-state records without requiring the failed node to answer. Unavailable records remain unavailable. The coordinator checks the assigned scheduler identity, queue, PCS membership, protected coordinator/peer IDs, image, launch-template version and reserved-capacity binding. Retirement normally requires an empty queue. After failed active-FLR recovery, the sole exception is this round's exact same-participant job in COMPLETING on the assigned failed predecessor, with a recorded active mutation attempt. The owned isolation and root-only scheduling reservation remain required. It does not change group bounds, cancel a capacity reservation, launch an arbitrary instance or select a spare node.

Each invocation advances the same recorded operation. A pending retirement, bootstrap or check returns a specific incomplete result, normally exit code `3`. Wait for that invocation to return before issuing another control command, then repeat `replace` as instructed. Continue the same pending bootstrap request; do not cancel it or issue a separate initializer. Read terminal status for that exact request rather than treating an earlier saved InProgress snapshot as the final result. A lost termination acknowledgement is reconciled by exact old EC2 ID. Bootstrap/health retries never terminate the new candidate. Another assignment sharing the target cannot claim it during replacement. The per-logical-node binding is shared by legitimate aliases and does not rewrite the unrelated negative-control assignment merely because an IP matches.

After evidence capture, cloud reads and durable dispatch intent, the coordinator rereads the queue and scheduler identity/address, isolation and reason immediately before termination. An unreadable queue, any job outside the exact COMPLETING exception, or a changed binding refuses dispatch. These are fresh dependent-decision observations, not an atomic transaction with Slurm and EC2. A previously terminated exact old ID still reconciles without addressing its successor.

Drain ownership keeps the complete Reason body and its raw annotation for audit. The coordinator accepts exactly the configured reason, or that reason plus Slurm 25.05.9's exact ` : reboot issued` suffix only with this assignment/round's durable reboot request, operation and matching isolated scheduler identity. The suffix is not stripped by the shared parser. Fault injection, initial start, bootstrap upgrades, generated admission gates and successor qualification remain exact-only: none inherits a predecessor's reboot authority. Unknown suffixes and field-shaped Reason text (including `Comment=`) remain foreign. Replacement reuses the existing `dispatching` intent and reservation; it never clears the Reason or creates a new retirement to repair this mismatch.

The participant-to-control SSH wrapper uses `StrictHostKeyChecking=accept-new`. The separate control-to-target maintenance hop uses pre-pinned per-instance host trust; do not describe every SSH hop as strict pre-pinned trust.

The new instance must register under the same assigned Slurm name with a different EC2 identity. A changed boot alone is not replacement. The fixed, hash-pinned SSM document initializes the new instance and reports its SSH public host key over the authenticated service response; the coordinator never uses blind `ssh-keyscan` or disables host checking. Fresh device allowlists, accounts, pinned materials, runtime initialization and checks `0`, `2`, `3`, `6` precede admission. Successor admission and RESUME require an empty queue, including completion of the predecessor job's cleanup. Old device tokens and warning baselines are not reused.

Runtime readiness is not workload verification. After either recovery or replacement, submit the existing verification job from a fresh allocation:

```bash
source /opt/aim344/device-recovery/companion/lab.env
export PATH=/opt/amazon/efa/bin:/opt/aws/pcs/scheduler/slurm-25.05/bin:$PATH
command -v sinfo salloc sbatch srun
unset AIM344_EFA_IFACE FI_EFA_IFACE
sbatch --wait --export=ALL --partition="$PARTITION" /opt/aim344/device-recovery/13.verify-after-recovery.sbatch
```

Check 5 runs once from the allocation coordinator. Its per-UID, per-job named container is initialized on both nodes before the collective timers, with `AIM344_CONTAINER_PREP_TIMEOUT` (default 600 seconds). Isolation/main reuse that root, and storage reuses it only when its image matches. No prior job or participant cache is required. The default 25-minute allocation remains the outer limit. In the September 24 fresh post-replacement trial, the allocation completed in 51 seconds including 6 seconds of job-local cold preparation; this is a measured run, not a general cold-start guarantee. Check 5 uses include-unset/exclude-only MPI configuration and Pyxis container-env precedence overrides scoped to its own steps, not host counters or storage. Keep both correctness columns, every allocated node's per-device EFA deltas and the storage workload result. The verification script checks all allocated nodes, not just whichever rank's log is visible locally, and copies results to each allocated node with digest verification. A fresh replacement initializes the isolated 32 MiB fixture with step 0; that seed is not recovered model/optimizer state. No instructor hand-restoration or facilitator-arranged replacement is a session remedy.

After completion, retrieve the batch-host log and run its printed fresh-`srun`, digest-verification, extraction and summary commands. The main numeric rows in the terminal log omit the full rank headers and separate isolation capture. With `job` set to this verification job's identifier, inspect the extracted files:

```bash
read -r -p 'Verification job identifier: ' job
raw_dir="$HOME/aim344-verify-$job/verify-$job/check5"
less "$raw_dir/nccl-allreduce-raw.txt"
less "$raw_dir/nccl-efa-only.txt"
```

Require 25 main rows from 8 B through 128 MiB and one EFA-only 256 MiB row. In each raw file, inspect all 16 GPU rank headers across two MPI processes, both zero-mismatch columns and zero out-of-bounds values. Keep the job log, extracted summary, per-device counter results and provider/performance WARN separately. These are default verifier paths; a site override of the run name changes the inner directory printed by the job.

## Pre-session provisioning — separate authorization

Provisioning must finish before participants receive access. It is not something `replace` silently deploys. The one authoritative measured composition is an **independent login coordinator outside the GPU group**, the reviewed environment-independent `aim344-early-imds` LT2 part with unchanged native/custom MIME, **per-node `ssm-install` fixed initialization**, literal `/usr/local/sbin/aim344-replacement-prolog`, and a root-only whole-node Slurm reservation through qualification. The early hook supplies the pre-access boundary; the late SSM service alone does not. Keep the protected peer and coordinator excluded from retirement and fixed-document initialization.

Historical topology placed the coordinator on the other GPU node. The explicitly authorized pre-session login cutover and both-GPU LT1→LT2 roll superseded it. The earlier no-roll-only candidate did not establish first-boot access isolation and is not the current recipe. The generator's default `boothook` mode embeds node-specific initialization and is **not** the reviewed shared-LT part. Neither that default nor a late-SSM-only installation is a substitute for the composition here. Preserve/export evidence before any separately authorized rollout; a shared GPU-group LT change can replace both GPU nodes. Historical GO does not authorize another roll, and no group update is a participant recovery action.

1. Record account, Region, scheduler version, target and protected IDs, Slurm names, partition, current AMI/LT version, scaling bounds, subnets and reservation. Record expected GPU/EFA counts independently of the device being tested. Retain the existing reservation and group minimum/maximum; no launch-before-terminate guarantee exists when all reserved slots are occupied.
2. Install the existing participant control account/forced-key boundary with `install-participant-control.sh`. Its target setup text describes initial provisioning, not recovery. Confirm matching participant UID/GID on every allocated node; no sudo, docker/lxd group, root maintenance key or instance-profile credentials may be exposed to participants.
3. Acquire the exact suite pin as described below. Archive the companion candidate and suite separately; re-archiving edited documentation changes the companion release hash and requires a newly reviewed manifest/document. Record all four private S3 object keys, SHA-256 hashes and version IDs when available. Mutable candidate names are not a release pin. No object/image availability or deployed hash is certified by local generation.
4. Create a private manifest for `prepare-replacement.py`: `node`, `region`, `slurm_bin`, `participants` (list of `name`, numeric `uid`, `gid`), `protected_instance_ids`, `maintenance_public_key` (public ed25519 key only), `release_sha256` (companion archive hash), and `objects`. Object keys are `companion.tgz`, `healthcheck-pinned.tgz`, `aim344.sqsh`, `nccl-baseline.sqsh`; each value has `bucket`, `key`, `sha256`, optional `version_id`. Include only approved table accounts. No old GPU/EFA allowlist or private key goes into this manifest. Add `verification_environment` with the assigned `PARTITION`, the provisioned `NCCL_CONTAINER` image path and site `NCCL_SOCKET_IFNAME` (retain a leading `=` for an exact interface). Supported Check 5 timing/MPI/network overrides are enumerated in `replacement-bootstrap.py`. The default suite remains the pinned `/opt/aim344-healthcheck` tree. To use a separately approved, unmerged Check 5 candidate, also supply `check5_suite_root=/opt/aim344/healthcheck-candidate-<64 lowercase hex revision digest>` and a separately hash-pinned `objects["check5-candidate.tgz"]` with the repository-relative suite layout. This stages that candidate separately, never relabels it as main, and records its installed hashes. Install the generated `participant-lab.env` as readable `lab.env` on the independent login coordinator; bootstrap writes identical defaults on the replacement. Match `AIM344_PARTITION` in control-account provisioning to this assignment. Participants source `lab.env` and submit directly without sudo; explicit caller overrides survive the generated defaults. The generator does not add client directories to `PATH`. The participant setup must explicitly add `/opt/amazon/efa/bin` and the manifest's installed `slurm_bin` (the measured PCS installation used `/opt/aws/pcs/scheduler/slurm-25.05/bin`), then verify `command -v sinfo salloc sbatch srun` in both terminals before allocation. The generator also does not set a 30 GB/s Check 5 floor.
5. Supply a site JSON with **`"provisioning_mode": "ssm-install"`**, `account`, `coordinator_instance_id`, `coordinator_role_id` (the unique `RoleId` returned by IAM GetRole, not a role name), `cluster_id`, `group_id`, `capacity_reservation_id`, `instance_type`, `subnet_ids`, `image_id`, `scaling_configuration` and `launch_template`. The latter two must exactly match PCS response shapes (`minInstanceCount`/`maxInstanceCount`, `id`/`version`). Generate files locally:

   ```bash
   python3 facilitator/prepare-replacement.py manifest.json site.json generated-replacement
   ```

6. Review the generated fixed SSM document, coordinator IAM policy, object-read policy and disabled assignment fragment. This mode emits **no MIME/LT artifact**. The document has no free-form parameters. The coordinator grant is constrained to credentials issued to the protected coordinator, one PCS group and the fixed document; explicit denies protect coordinator/peer IDs. It contains no grant, PassRole, group-update or capacity-reservation-change action. The saved actual effective IAM calls passed their scoped allow/deny controls; organization SCPs were not enumerated. A new deployment still needs effective-boundary validation. Read-only DescribeInstances necessarily uses `Resource: "*"`.
7. Use the separately reviewed shared early part, [early-boothook.sh](early-boothook.sh), unchanged from the measured LT2 component (SHA-256 `87e4dfef46eec5b108e2befff862e78cba7abce31ba3a42feab462db4e9f73a6`). It installs only environment-independent host/forwarding IPv4/IPv6 rules and `aim344-early-imds.service`, not a per-node manifest or initializer. Merge it with the site's existing custom shell and PCS native init/config/finalize MIME parts; the deployment-specific merged MIME is deliberately not distributed. Check combined size and read back actual user data, image, dependencies and hashes. Do not replace this part with the default node-specific generated boothook. Only separately authorized pre-session infrastructure may configure the unique literal `/usr/local/sbin/aim344-replacement-prolog`, after installing its reviewed wrapper and preserving/chaining existing health settings. A non-root account alone does not restrict IMDS; installed IPv6 rules alone do not qualify IPv6 denial.
8. The generated assignment fragment is intentionally disabled. After separately authorized provisioning, read back the fixed document's actual version/hash and installed LT/group/Prolog/credential boundaries, then install the fragment as `replacement` in the trusted assignment and enable it. A local JSON hash is not the SSM service's document hash. No IAM grant or deployment occurs in the generator.

The measured fixed SSM document installs the pinned initializer and per-node manifest after native registration, then fetches and validates pinned objects, rejects unsafe archive entries, creates matching unprivileged accounts, discovers fresh GPU/EFA/management identities, installs narrow maintenance access and initializes runtime paths. Partial initialization is retryable on the same candidate. It does not depend on native lifecycle-agent support. No formatting of storage or arbitrary lost-original salvage is performed.

The shared early boothook must run inside a cloud-init network-stage unit ordered before `sysinit.target`; inspect the actual AMI's units and boot journal. On the inspected PCS Ubuntu 24.04 image, SSH socket/service, cron and user sessions are after that target. Its trap is intended to hold the first boot on setup failure instead of returning an error that cloud-init could ignore. It also enables `aim344-early-imds.service` as an early required dependency of `sysinit.target` for subsequent boots. A held boot requires pre-session infrastructure repair, not participant access or a claim of readiness. Successful fresh boots and reboot persistence were measured, but the actual delayed/failing service test was a **later-boot installed-service `ExecStartPre` failure**, not a first-hook-held trial. First-hook failure/access/job denial remains a missing hardware gate; local Bash trap tests do not fill it. Different image ordering needs separate qualification.

### Fixed initializer and admission in the measured composition

The explicit `ssm-install` mode generates a parameterless SSM document that installs the pinned initializer, manifest, admission gate and persistent **late** `aim344-replacement-imds.service` itself. It emits no LT/user-data artifact and does not update the compute group; that does not make the overall measured setup no-roll. The independent early LT2 service remains necessary and is not overwritten by the late payload. The fixed document validates protected identities and exact scheduler binding before installation, serializes durable file installation, applies IPv4/IPv6 host/forwarding restrictions and enables the late service before subsequent slurmd starts. Four object hashes and authenticated SSM host-key response remain requirements. Read back the content-derived document's actual service version/hash before enabling.

This composition requires the unique literal cluster Prolog `/usr/local/sbin/aim344-replacement-prolog`, not the historical dispatcher and not a wildcard. Install the reviewed wrapper on the dedicated pair before configuring that Prolog; the protected GPU peer chains its existing health Prolog without receiving target initialization. The coordinator is on the independent login host. Preserve existing configuration and export state first. A fresh successor lacks this literal file until SSM installs it. Actual installed Slurm 25.05.9 missing-literal and present-but-unadmitted denials were observed; another deployment must requalify them before retirement. Do not accept a generic Prolog error as exercise-owned isolation.

Before retiring the old ID, the coordinator also creates and reads back a whole-node, root-only Slurm reservation named `aim344-replacement-<old-id>` with `Nodes=<assigned-node>`, `Duration=UNLIMITED`, and `Flags=STATIC_ALLOC`. It does not use MAINT, OVERLAP, IGNORE_JOBS, REPLACE or a node count that could substitute the peer. This scheduling reservation is distinct from the EC2 capacity reservation. Durable intent owns create retries; changed/missing confirmed reservations fail closed. It remains through bootstrap, health and scheduler resume, and is deleted only after a durable qualified binding and fresh identity/boot/queue checks. Failed deletion keeps participant/alias entry blocked; the same participant repeats `replace`. No instructor restoration is the remedy.

Those are input spellings, not readback values. Slurm 25.05.9 prints `STATIC` and converts unlimited duration to a finite 365-day interval (`Duration=365-00:00:00`). The controller fixes CLI timestamps to UTC, checks start against the durable create intent, persists the observed start/end, and rejects changed intervals or less than one hour remaining. This is not an everlasting scheduler barrier: expiration remains a limit, and the independently verified admission Prolog is still required. Whole-node readback must have matching `CoreCnt` and CPU TRES against the trusted scheduler's socket/core/thread/CPU topology, no partial-core `NodeName`/`CoreIDs` section, and no additional resource or access scope. Incomplete, duplicate, inconsistent or changed fields refuse on creation, retry and release. Every release retry synchronizes the complete binding file and its directory before attempting deletion; a visible rename after failed directory synchronization is not a durability receipt.

The scoped audit observed actual whole-node STATIC reservation denial for ordinary and explicit unauthorized requests, missing/present-but-unadmitted Prolog failures, retention through registration/qualification and normal participant-driven release. The historical standalone reservation probe ended at a COMPLETING-state observation race, not a clean whole-script pass; later replacement records establish release. A new deployment must prove those gates afresh. Root/Slurm administrator bypass is outside the participant boundary. If a gate fails, retain isolation and stop retirement, not roll a group as a remedy. These scheduler observations do not close first-hook-held or IPv6 hardware gates.

Successors support one explicit storage layout: `/opt/aim344` is a root-owned directory on the root filesystem, with sufficient root-volume space for the pinned images and companion materials. Bootstrap refuses an overlaid staging mount and writes `staging_policy=root-directory` plus both image SHA-256 pins into the private maintenance configuration. Later GPU recovery validates the same source and ownership before runtime restoration; it does not bind `/opt/dlami/nvme/aim344` over those materials. Missing, changed or overlaid sources refuse recovery. Previously provisioned nodes without that policy retain the existing NVMe-bind contract and source/UUID checks; they are not silently migrated. Validate root-volume sizing and absence of conflicting staging mounts during pre-session provisioning.

SSM forward-access grants additionally require `aws:ViaAWSService=true` and the exact coordinator `aws:userid` (`RoleId:instance-id`). They retain document/group/protected-ID restrictions, rather than admitting every instance sharing the role. Check that the role trust permits only the intended EC2 service, not principals able to forge that session name, and that other policies cannot bypass these boundaries. Direct calls still require the coordinator's `ec2:SourceInstanceARN`; no forward-access exception permits termination.

If a job reaches the admission Prolog before initialization, the gate reads the exact local/scheduler identity and drains with `AIM344-admission-<new-instance-id>` before returning failure. Slurm 25.05.9 preserves that existing drain; replacement may reconcile this specific reason, not a generic `Prolog error`, another instance's marker or administrator isolation. Existing isolation is never rewritten by the gate. The failed job must finish leaving the queue before `replace` continues; no automatic cancellation of another job is authorized.

Verify normal participant privilege negatives before the session: `sudo -n true`, reading coordinator configuration/private key, arbitrary root SSH/scp through the maintenance connection, and another table's key must fail. IMDS qualification requires root-positive controls alongside participant host/container denial, not merely an unreachable endpoint. Actual host/enroot IPv4 coverage passed for both users on both nodes; IPv6 and arbitrary Docker-forwarded traffic are not qualified. Latest saved IPv6 endpoint flags are enabled, but the subnet has no IPv6 CIDR/root route. Verify enroot shared parents, image paths, fixture ownership and the Prolog gate on both nodes. A missing `fi_info` on PATH, a private `/tmp/enroot`, mismatched UID or an invalid staging source is an environment failure, not device-fault evidence.

The fixed initializer also installs `aim344-enroot-parents.service`, required before `sysinit.target` and after tmpfiles setup. This recreates checked root-owned shared enroot parents after reboot cleanup, before a participant can create them privately. Runtime restoration repeats the same descriptor/ownership checks. An existing participant-owned parent is refused, not relabelled or deleted; use participant replacement for that ambiguous runtime. Verify actual service ordering, reboot and both participant accounts before declaring the successor qualified.

### Explicit existing-node upgrade (local root provisioning only)

The normal parameterless SSM document remains a fresh-node/exact-repeat path;
it refuses a different or legacy receipt before changing installed metadata.
`prepare-replacement.py` with `provisioning_mode=ssm-install` also emits the
private `upgrade-existing-node.py`, using the same pinned installer/bootstrap.
It is **not** a participant maintenance command or a new controller. The full
bootstrap now exceeds the legacy per-node EC2 boothook size limit; that mode
refuses rather than truncating. Use the existing SSM provisioning composition.

During an authorized pre-session cutover, retain coordinator history/credentials
and root backups. Quiesce coordinator access and isolate only the intended node
with the existing identity-bound admission drain (`IDLE+DRAIN`, optional CLOUD,
reason exactly `AIM344-admission-<instance-id>`). The upgrade does not drain,
cancel, resume or operate any other node; it refuses queued jobs or participant
processes. Keep that isolation across all hosts until qualification is complete.
Supply the independently checked SHA256 of the **original receipt bytes** and
**original manifest bytes**, not the new manifest or merely its release hash:

```bash
sudo /usr/bin/python3 /root/reviewed/upgrade-existing-node.py \
  "$ORIGINAL_READY_SHA256" "$ORIGINAL_MANIFEST_SHA256"
```

The reviewed artifact contains the complete new per-node manifest and helper
bytes. It validates current instance/Slurm binding, old receipt/source hashes,
device/site identity, matching existing users and maintenance public key, then
downloads and validates release archives before metadata publication. Unchanged
images require identical old/new bucket/key/version/hash and a fresh local hash;
they are neither downloaded nor copied. Changing images or users is deliberately
outside this minimal upgrade path. Accounts, runtime fixtures, unrelated history,
SSH keys, and original GPU/DCGM records are not reinitialized.

Before installing, it archives the exact old receipt and manifest under the
root-private `/var/lib/aim344-replacement/upgrade-<new-manifest-digest>/`, writes
an explicit pending marker, marks the current receipt not-ready, and invalidates
admission. A failed partial installation stays isolated. Repeat the **same**
artifact with the **same original hashes** after correcting the cause; do not
remove receipts, backups or pending markers. A completed retry only verifies and
returns the current receipt. No multi-file or multi-host atomicity is claimed.

Success publishes a new ready receipt only after verified installation and a
fresh isolation check; it does not admit or resume. After owner qualification,
the existing `admit-replacement` route binds admission to the new manifest digest
under the same bootstrap lock. The existing Prolog rejects stale admission or a
pending upgrade. The deployment owner must regenerate a new immutable SSM
document from the reviewed final bytes and read back its service hash; do not
edit an existing content-addressed document in place. Local regression tests are
not deployment, GPU/EFA, IAM, or event readiness evidence.

### Suite acquisition

This recipe acquires the public review candidate, not an upstream mainline release or a deployed event configuration. The health suite commit below preserves the reviewed integrated candidate's complete suite bytes and executable modes. Generate new release archives and manifests; do not reuse an earlier trial archive digest after changing pins or documentation.

The companion review branch is `riv2026/aim344-reviewed-20260924`; its suite dependency is `riv2026/aim344-healthcheck-reviewed-20260924` at the exact `HEALTHCHECK_COMMIT` in `pins.env`. The suite commit `474155a63059769bcea6365d4ec2b2f34d4c8000` was fetched publicly and its archive files compared with the committed tree. If the exact commit cannot be fetched, stop rather than substitute main or an unlabelled patch. From the companion directory in a Git checkout:

```bash
set -euo pipefail
source pins.env
repo=$(git rev-parse --show-toplevel)
git -C "$repo" fetch --no-tags https://github.com/awslabs/awsome-distributed-ai.git "$HEALTHCHECK_COMMIT"
test "$(git -C "$repo" rev-parse FETCH_HEAD)" = "$HEALTHCHECK_COMMIT"
git -C "$repo" cat-file -e "$HEALTHCHECK_COMMIT^{commit}"
git -C "$repo" archive "$HEALTHCHECK_COMMIT" validation/gpu-cluster-healthcheck | gzip -n -6 > healthcheck-pinned.tgz
```

These commands populate objects/FETCH_HEAD without switching branches and create only the named archive. For local review before publication, replace the HTTPS source with an explicitly supplied path to the clean healthcheck-release worktree; the exact commit comparison is still required. This does not make a public-download claim. `prepare-login.sh` compares staged suite file contents and executable modes against this exact Git archive. Build/import steps remain separate pre-session work.

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

Site manifests, merged MIME, fixed document/service hashes, object locations, instance/image/template identifiers and historical source/evidence mapping remain with the workshop owner, outside distribution. Generate and review new manifests and actual service hashes for the accepted candidate rather than reusing the old deployment's pins. The source package includes the environment-independent early hook but does not provision infrastructure or confer access to private image objects. Build images using the pinned Dockerfile or obtain explicitly approved digest-pinned images.

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
salloc --job-name=aim344-device --exclusive --partition="$PARTITION" --nodes=2 --gres="gpu:$GPUS_PER_NODE" --ntasks-per-node=1 --cpus-per-task="$((GPUS_PER_NODE * CPUS_PER_RANK))" --mem="$MEMORY_PER_NODE" --time=00:06:00
export RESULTS_DIR=/var/tmp/aim344-$(id -un)-$SLURM_JOB_ID
# EFA branch only: confirm this physical rail against the prepared assignment.
export AIM344_EFA_IFACE=rdmap83s0
bash 11.device-workload.sh
```

For GPU, leave `AIM344_EFA_IFACE` unset. The selected EFA name above is from the measured G7 pair, not a portable device-name assumption. The launcher saves the combined output to `~/aim344-results/device-JOBID.log` on the login coordinator. Its timed workload runs for 120 seconds; the six-minute allocation limit additionally covers launch and cleanup. In the already-prepared terminal B, wait for two advancing correctness-checked progress records, then issue exactly one `bash 12.device-exercise.sh active efa` (or `active gpu`). Use `status` and the allowlisted `collect <check>` commands afterward.

The control command accepts no node, job ID or BDF. The target checks the fixture process/cgroup, exact authenticated participant/job, allocation limit and selected-EFA byte movement including `rdma_write_bytes`. Before each mutation acknowledgement, the coordinator requires two advancing records for the exact job/world size, with the latest no older than 15 seconds. These records establish activity; scheduler ownership and target PID/cgroup checks establish authorization independently.

If the workload ends naturally before any active request was sent, preserve the log, exit the allocation, confirm the pair is idle, and repeat the allocation/workload step with terminal B ready. Once a request has been sent, an unknown outcome requires diagnosis and supported recovery, not another fault.

The target sends authorization and mutation-start records over the same SSH connection. Each must be synchronized to root-owned coordinator state before its hash acknowledgement permits the next target write. `reset_method` must read back exactly `flr`; there is no bus fallback. Start is evidence of intent, not a claim that reset returned. Both original persistence and telemetry states are captured without pausing the active workload, and restored after reboot.

If the observation deadline expires or the ACK connection fails, the coordinator retains the mutation attempt and any saved markers with an unknown outcome. Use `status`, `collect` and then `recover`; repeating `active` does not issue another fault. Only a local failure to start the transport establishes that no request was sent.

The device launcher no longer SIGKILLs its `srun`/log pipeline at 210 seconds. Slurm walltime bounds the allocation while the existing log remains attached; GPU D-state cleanup can outlive both the NCCL watchdog and that walltime. Wait for terminal exit/COMPLETING before `recover`. Recovery refuses a RUNNING job and a different job even if owned by the same participant. Same-instance Slurm reboot is tried first; after the configured grace, one exact-instance EC2 reboot may be requested through the existing replacement authority. It is never reported as a successful recovery until a fresh boot, restored runtime and suite PASS are observed. If it stalls or SSH is unreachable, `replace` can retire only this round's exact COMPLETING allocation and assigned failed instance, after failed recovery. A successful reset, log retrieval or prior EC2 reboot is NOT a replacement prerequisite. Scheduler identity/address, owned drain, queue owner and exact cloud membership are checked before retirement. The scheduler may still display ALLOCATED/COMPLETING while that sole job cleans up; those flags do not turn its failed instance into a running foreign allocation. Replacement admission and RESUME still require an empty queue. No broad kill, instructor restoration or group/reservation capacity mutation is introduced.

The generated coordinator policy adds `ec2:RebootInstances` under the existing source-instance/group restrictions and protected-instance explicit deny. Review and provision that policy with the revised helper bytes, not an old manifest. `collect kernel` exports a bounded useful kernel window for the recorded fault boot: the last 512 matching GPU/EFA/PCI/reset/lost-message records, at most 524288 raw bytes and an 8-second read window. The payload includes instance/boot, byte count, SHA256, command return code and a bound-hit indicator, and explicitly does not claim a complete journal. `recover` attempts the same export before recovery, but target/journal/output failure never blocks recovery or replacement. The deployed correction uses the supported `journalctl -k` option and rejects a null boot UUID before collection. In the EFA trial, inner and outer collection statuses were both zero and the selected fault-boot window contained 512 records. Earlier GPU captures from the old collector remain unavailable. Prior-boot journal availability and events lost to kernel floods are not guaranteed. Collected suite/kernel outputs, mutation records and coordinator workload logs remain off-target. The EFA participant trial verified retirement through route update and fresh workload retrieval. A wholly unresponsive guest and every stalled-target variant remain distinct from that measured path. Do not interpret FLR return or driver `reset done` as CUDA recovery, physical PCIe loss, Xid79, or a guarantee across repetitions.

State, binding, dispatch intent and replacement health logs are private coordinator files under the existing state directory. File synchronization and directory synchronization precede dependent operations. A directory-sync failure can occur after rename; refusal is not proof that no file was renamed. Neither local fsync tests nor mock API acknowledgements establish survival of a physical crash.

Participant captures remain append-only by round and attempt under `~/aim344-results/collected`. The output writer drops to the participant UID/GID and uses descriptor-relative operations. Runtime fixture handover retains the existing no-follow, same-filesystem protections. A symlink/socket left by a participant is not followed by root.

The deadline sweep is the existing scheduled one-shot controller route, not a new management service. It does not perform replacements automatically; it keeps isolation after bounded ordinary-recovery attempts. The participant uses `replace`. Isolation is not a completed round.

Manual pre-session experiments must not write bare persistence words to the participant helper's `selected-persistence-mode.txt` or `native-dcgm-state.txt`: those paths contain operation-bound JSON. Store standalone experimental records in a separate evidence directory. The participant helper authorizes fresh capture only on a newly generated trusted operation; retries synchronize and reuse retained originals rather than recapturing.

Historical active-GPU removal stalled guest shutdown. Active EFA unbind remained outstanding until its workload ended: the dedicated probe returned at t+171 s, 11 s after cancellation. These measurements explain why `start gpu` and `start efa` remain idle-only; `active gpu` and `active efa` use the separate FLR procedure above. They do not establish that every supported idle fault stalls. Historical replacement after EC2 reboot was not a trial of `scontrol reboot nextstate=DOWN`.

Earlier recovery-decision/implementation and executor-first reports remain historical evidence with the workshop owner. This runbook defines the current participant-remedy, WARN-acceptance and replacement-completion contract. Passing named reproductions supports those exercised inputs, not all subrequirements; imported TestCase executions are regression reruns, not newly authored tests.

## Primary mechanism references

- [EC2 TerminateInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_TerminateInstances.html): exact-ID idempotence, data loss, and `SkipOsShutdown`.
- [PCS reboot FAQ](https://docs.aws.amazon.com/pcs/latest/userguide/slurm-reboot-faq.html): ordinary Slurm reboot retains the instance; `nextstate=DOWN` requests replacement after reboot.
- [PCS instance discovery](https://docs.aws.amazon.com/pcs/latest/userguide/working-with_compute-instances.html): service-owned compute-group membership tag.
- [PCS user data](https://docs.aws.amazon.com/pcs/latest/userguide/working-with_ec2-user-data.html) and [early boothook](https://docs.aws.amazon.com/pcs/latest/userguide/working-with_ec2-user-data_early-boot.html): MIME merging and early execution on every boot.
- [SSM SendCommand](https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_SendCommand.html) and [GetCommandInvocation](https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_GetCommandInvocation.html): exact instance/document/version/hash and eventual consistency.
- [IMDS restrictions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-metadata-limiting-access.html): host firewall boundaries, not automatic protection from account privilege separation.
- [SSM Run Command setup](https://docs.aws.amazon.com/systems-manager/latest/userguide/run-command-setting-up.html), [FAS](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_forward_access_sessions.html), and [IAM principal key values](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_variables.html): forward-service allowance and preserved exact EC2 role-session identity.
- [Slurm 25.05.9 node manager](https://github.com/SchedMD/slurm/blob/slurm-25-05-9-1/src/slurmctld/node_mgr.c): `_drain_node` and `validate_node_specs` preserve an already-drained node on Prolog failure. This is upstream source matched to the historical binary version, not a freshly measured deployed binary.

PCS owns relaunch under unchanged bounds. Reserved-slot availability, successful node-name registration, bootstrap, health and a fresh participant job must be observed on the dedicated pair before hardware readiness is claimed. There is no cloud or publication approval in this document.
