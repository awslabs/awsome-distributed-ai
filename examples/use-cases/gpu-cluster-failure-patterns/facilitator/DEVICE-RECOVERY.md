# GPU and EFA device recovery

The diagnostic owner is [validation/gpu-cluster-healthcheck](https://github.com/awslabs/awsome-distributed-ai/tree/main/validation/gpu-cluster-healthcheck). The coordinator applies its verdicts; it does not substitute a hardware diagnosis. GPU PCI removal changes operating-system visibility, not physical hardware, and does not establish NVIDIA Xid 79. EFA driver unbind changes one PCI function and RDMA domain, not an NCCL plugin.

Independent source/evidence review accepts the measured participant recovery/replacement and fresh workload flow on the dedicated pair. Retained source, payloads and results were checked byte-for-byte. **Full hardware qualification remains withheld** for missing first-hook-held failure/access-denial proof and unqualified IPv6. Candidate publication is separate from hardware or event-readiness approval. Local recording-executor tests are not cloud replacement evidence; see the [scoped audit record](../VALIDATION.md#september-19-2026-scoped-participant-recovery-audit). This document authorizes no new deployment, fault injection or rollout.

## Participant session

Use the unprivileged coordinator account and the existing forced-command connection. The authenticated key selects the assignment. There is no participant node, instance, PCI, job, shell or IAM option.

```bash
./12.device-exercise.sh start gpu
./12.device-exercise.sh status
./12.device-exercise.sh collect 0
./12.device-exercise.sh collect 3
./12.device-exercise.sh recover
```

For the separate EFA round, use `start efa` and checks `2` and `6`. Both starts require an idle target. End the previous allocation before starting or replacing a node. Preserve fault-time logs before recovery.

Supported `recover` is the first route. GPU recovery requires coherent operation-bound original records, a valid recorded reboot baseline, and fresh scheduler/boot observations. Successful EFA rebind does not reuse an old communicator. Failed EFA rebind does not escalate into a reboot without originals; it requires replacement. Missing originals are not reconstructed from an already-modified node. Known statistics-skipped PASS refuses regardless of baseline; WARN auto-qualification is disabled, including source-shaped memlock advisory matches.

When recovery cannot qualify the node, retain isolation and use:

```bash
./12.device-exercise.sh replace
./12.device-exercise.sh status
```

`replace` retires only the failed instance recorded for this assignment. It preserves a coordinator-side round snapshot and attempts to capture the two original-state records without requiring the failed node to answer. Unavailable records remain unavailable. The coordinator checks the assigned scheduler identity, empty queue, PCS membership, protected coordinator/peer IDs, image, launch-template version and reserved-capacity binding. It does not change group bounds, cancel a reservation, launch an arbitrary instance or select a spare node.

Each invocation advances the same recorded operation. A pending retirement, bootstrap or check returns a specific incomplete result; repeat `replace`. A lost termination acknowledgement is reconciled by exact old EC2 ID. Bootstrap/health retries never terminate the new candidate. Another assignment sharing the target cannot claim it during replacement. The per-logical-node binding is shared by legitimate aliases and does not rewrite the unrelated negative-control assignment merely because an IP matches.

After evidence capture, cloud reads and durable dispatch intent, the coordinator rereads the queue and scheduler identity/address, isolation and reason immediately before termination. An unreadable or nonempty queue or a changed binding refuses dispatch. These are fresh dependent-decision observations, not an atomic transaction with Slurm and EC2. A previously terminated exact old ID still reconciles without addressing its successor.

The new instance must register under the same assigned Slurm name with a different EC2 identity. A changed boot alone is not replacement. The fixed, hash-pinned SSM document initializes the new instance and reports its SSH public host key over the authenticated service response; the coordinator never uses blind `ssh-keyscan` or disables host checking. Fresh device allowlists, accounts, pinned materials, runtime initialization and checks `0`, `2`, `3`, `6` precede admission. Old device tokens and warning baselines are not reused.

Runtime readiness is not workload verification. After either recovery or replacement, submit the existing verification job from a fresh allocation:

```bash
sbatch --wait /opt/aim344/device-recovery/13.verify-after-recovery.sbatch
```

Check 5 runs once from the allocation coordinator. Keep both correctness columns, every allocated node's per-device EFA deltas and the storage workload result. The verification script checks all allocated nodes, not just whichever rank's log is visible locally, and copies results to each allocated node with digest verification. A fresh replacement initializes the isolated 32 MiB fixture with step 0; that seed is not recovered model/optimizer state. No instructor hand-restoration or facilitator-arranged replacement is a session remedy.

## Pre-session provisioning — separate authorization

Provisioning must finish before participants receive access. It is not something `replace` silently deploys. The one authoritative measured composition is an **independent login coordinator outside the GPU group**, the reviewed environment-independent `aim344-early-imds` LT2 part with unchanged native/custom MIME, **per-node `ssm-install` fixed initialization**, literal `/usr/local/sbin/aim344-replacement-prolog`, and a root-only whole-node Slurm reservation through qualification. The early hook supplies the pre-access boundary; the late SSM service alone does not. Keep the protected peer and coordinator excluded from retirement and fixed-document initialization.

Historical topology placed the coordinator on the other GPU node. The explicitly authorized pre-session login cutover and both-GPU LT1→LT2 roll superseded it. The earlier no-roll-only candidate did not establish first-boot access isolation and is not the current recipe. The generator's default `boothook` mode embeds node-specific initialization and is **not** the reviewed shared-LT part. Neither that default nor a late-SSM-only installation is a substitute for the composition here. Preserve/export evidence before any separately authorized rollout; a shared GPU-group LT change can replace both GPU nodes. Historical GO does not authorize another roll, and no group update is a participant recovery action.

1. Record account, Region, scheduler version, target and protected IDs, Slurm names, partition, current AMI/LT version, scaling bounds, subnets and reservation. Record expected GPU/EFA counts independently of the device being tested. Retain the existing reservation and group minimum/maximum; no launch-before-terminate guarantee exists when all reserved slots are occupied.
2. Install the existing participant control account/forced-key boundary with `install-participant-control.sh`. Its target setup text describes initial provisioning, not recovery. Confirm matching participant UID/GID on every allocated node; no sudo, docker/lxd group, root maintenance key or instance-profile credentials may be exposed to participants.
3. Acquire the exact suite pin as described below. Archive the companion candidate and suite separately; re-archiving edited documentation changes the companion release hash and requires a newly reviewed manifest/document. Record all four private S3 object keys, SHA-256 hashes and version IDs when available. Mutable candidate names are not a release pin. No object/image availability or deployed hash is certified by local generation.
4. Create a private manifest for `prepare-replacement.py`: `node`, `region`, `slurm_bin`, `participants` (list of `name`, numeric `uid`, `gid`), `protected_instance_ids`, `maintenance_public_key` (public ed25519 key only), `release_sha256` (companion archive hash), and `objects`. Object keys are `companion.tgz`, `healthcheck-pinned.tgz`, `aim344.sqsh`, `nccl-baseline.sqsh`; each value has `bucket`, `key`, `sha256`, optional `version_id`. Include only approved table accounts. No old GPU/EFA allowlist or private key goes into this manifest.
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

### Suite acquisition

The companion branch is `riv2026/aim344-distribution`; its suite dependency is the separate `riv2026/aim344-healthcheck-release` branch at the exact `HEALTHCHECK_COMMIT` in `pins.env`. At authoring both are local review candidates, not published or merged releases. The following public route is conditional on publication and exact remote verification; if the commit cannot be fetched, stop rather than substitute main or an unlabelled patch. From the companion directory in a Git checkout:

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

The explicit repository root prevents subdirectory filtering; the commit timestamp prevents tree-object archives from using generation time. The fixed tar mode mask and gzip level/header settings define the release recipe. Record Git/gzip versions and the commit alongside both archive digests. Verify the complete companion file set and modes against the accepted tree before approval, not just command exit status. Retain the reviewed archive bytes as the release artifact; if regeneration differs, stop and obtain review of the new bytes rather than reusing a prior hash. Never use the entire original development branch as a release payload. The suite candidate retains measured runtime bytes but has different test content and the README's corrected public image-recipe link, so its archive hash is new. Mainline integration must coordinate the overlapping [PR #1221](https://github.com/awslabs/awsome-distributed-ai/pull/1221), especially its environment-override contract versus physical DMI-derived expectations. Feature-branch availability is not mainline merge.

### Deployment-specific materials and protected peer

Site manifests, merged MIME, fixed document/service hashes, object locations, instance/image/template identifiers and historical source/evidence mapping remain with the workshop owner, outside distribution. Generate and review new manifests and actual service hashes for the accepted candidate rather than reusing the old deployment's pins. The source package includes the environment-independent early hook but does not provision infrastructure or confer access to private image objects. Build images using the pinned Dockerfile or obtain explicitly approved digest-pinned images.

The measured protected peer retained older maintenance code with a deliberate DMI suite/verifier overlay. It was a workload peer, **not** a final-helper-equivalent replacement target. Do not describe the historical pair as a homogeneous final-helper deployment. Exact retained evidence remains unchanged outside this public candidate.

## Evidence, retries and qualification boundaries

State, binding, dispatch intent and replacement health logs are private coordinator files under the existing state directory. File synchronization and directory synchronization precede dependent operations. A directory-sync failure can occur after rename; refusal is not proof that no file was renamed. Neither local fsync tests nor mock API acknowledgements establish survival of a physical crash.

Participant captures remain append-only by round and attempt under `~/aim344-results/collected`. The output writer drops to the participant UID/GID and uses descriptor-relative operations. Runtime fixture handover retains the existing no-follow, same-filesystem protections. A symlink/socket left by a participant is not followed by root.

The deadline sweep is the existing scheduled one-shot controller route, not a new management service. It does not perform replacements automatically; it keeps isolation after bounded ordinary-recovery attempts. The participant uses `replace`. Isolation is not a completed round.

Manual pre-session experiments must not write bare persistence words to the participant helper's `selected-persistence-mode.txt` or `native-dcgm-state.txt`: those paths contain operation-bound JSON. Store standalone experimental records in a separate evidence directory. The participant helper authorizes fresh capture only on a newly generated trusted operation; retries synchronize and reuse retained originals rather than recapturing.

Historical active-GPU removal stalled guest shutdown. Active EFA unbind remained outstanding until its workload ended: the dedicated probe returned at t+171 s, 11 s after cancellation. These measurements explain why both participant starts are idle-only. They do not establish that every supported idle fault stalls. Historical replacement after EC2 reboot was not a trial of `scontrol reboot nextstate=DOWN`.

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
