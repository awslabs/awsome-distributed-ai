# AWS PCS — Roadmap / TODO

Implementation items under consideration for future work on the AWS PCS templates
(`architectures/aws-pcs`). This is a living checklist — add items via PR so the history is
captured in git, and check them off or remove them when done.

Priority: 🔴 high · 🟡 medium · 🟢 low

## Templates & deployment

- [x] 🟡 **Multi-AZ support in the prerequisites stack.** `ml-cluster-prerequisites.yaml`
  now supports up to 3 private-subnet AZs via `AdditionalSubnetAZ2`/`AdditionalSubnetAZ3`
  (CIDR1 split into four /18 blocks; additional subnets share the primary AZ's single
  NAT gateway). This unblocks `OpenZFSDeploymentType=MULTI_AZ` and higher-availability
  layouts. *(Note: OpenZFS MULTI_AZ wiring of the 2nd subnet into the FSx resource is a
  follow-up; the subnets + routing are in place.)*
- [x] ✅ **Targeted ODCR support for GPU node groups.** Done: `CapacityReservationType`
  (`capacity-block` | `targeted-odcr`) on `add-cng-p5`/`add-cng-p6-b200`/`add-cng-p6-b300`
  plus a `PlacementGroupName` parameter for reusing an existing cluster placement group
  (`PseriesPlacementGroupName` on deploy-all). `targeted-odcr` sets
  `CapacityReservationTarget` **without** `MarketType=capacity-block` and **keeps** the
  placement group (On-Demand billing against the reservation);
  `capacity-block` (the default) is unchanged and backward compatible. Verified without
  GPU capacity by the static method below: a launch-template-only harness deployed in all
  three modes, reading back the generated `LaunchTemplateData` —
  On-Demand → `MarketType=null, crid=null, pg=<new>`;
  `capacity-block` → `MarketType=capacity-block, crid=<id>, pg=null`;
  `targeted-odcr` → `MarketType=null, crid=<id>, pg=<existing group>`.
  **Confirmed end-to-end on real hardware** (2026-08-01): a PCS GPU node group
  created with `CapacityReservationType=targeted-odcr` and an existing
  placement group launched 2x p5.48xlarge that EC2 attributed to the targeted
  reservation (`CapacityReservationId` set on both instances,
  `Placement.GroupName` = the passed-in group) and the reservation's
  `AvailableInstanceCount` went 2 -> 0, i.e. the capacity was actually consumed.
  A Slurm job on the resulting `gpu` partition ran on both nodes.
- [x] ✅ **P4d / P4de support.** `add-cng-p4d.yaml` (4 network cards, all EFA, card 0 on
  DeviceIndex 0, cards 1-3 on DeviceIndex 1, per `describe-instance-types`
  `NetworkInfo.NetworkCards` / `EfaInfo.MaximumEfaInterfaces = 4`; identical for
  p4d.24xlarge and p4de.24xlarge) plus a `P4DCNGStack` branch in deploy-all
  (`PseriesInstanceType` accepts `p4d.24xlarge` / `p4de.24xlarge`). Verified with the
  static method (2026-09-08, us-west-2, no GPU capacity): CNGs created with
  `Min/MaxCount=0` for both types reached ACTIVE and the generated launch template
  carried the 4-NIC EFA layout, `MarketType=null`, a new placement group. A live boot
  is pending the re:Invent dry-run ODCR (2x p4d.24xlarge, CTI P508054549).
- [ ] 🟡 **Scope down the instance role's `AmazonS3ReadOnlyAccess`.** The PCS instance
  role in `cluster.yaml` attaches `AmazonS3ReadOnlyAccess` **unconditionally** (every node
  can read every S3 bucket in the account). The upstream
  [aws-hpc-recipes `pcs-iip-minimal`](https://github.com/aws-samples/aws-hpc-recipes/blob/main/recipes/pcs/getting_started/assets/pcs-iip-minimal.yaml)
  makes it an **opt-in** (`EnableS3ReadOnly`, off by default); ml-pcs lost that gate when
  the role was brought into `cluster.yaml`. Restore the opt-in (a parameter, or scope to
  the named data/templates bucket(s)) so the default cluster doesn't grant account-wide S3
  read. **IAM-behaviour change — needs care:** training test cases (FSDP, Megatron) and any
  workload reading datasets/checkpoints from S3 rely on this today, so validate those
  before tightening. (`AmazonSSMManagedInstanceCore` is also unconditional here, but SSM is
  a core feature of this architecture — login/connect, cluster-user policy — so it stays on
  by default.)
- [ ] 🟢 **Trainium (Trn) validation.** Validate the templates on Trainium instances
  (e.g. trn1/trn2) — node group, EFA/networking, and a sample training run.
- [ ] 🟡 **Graviton (arm64) CPU CNG support — `hpc7g` / `c7gn`.** EFA-capable arm64
  HPC instances (`hpc7g.16xlarge`, `c7gn.16xlarge`) are out of scope today: the
  cluster's default `AmiId` auto-resolves the **x86_64** PCS-Ready DLAMI, so pairing
  these types with the default AMI fails to launch. An arm64 PCS DLAMI exists at
  `/aws/service/pcs/ami/dlami-base-ubuntu2404/arm64/latest/ami-id` (verified via
  `aws ssm get-parameters-by-path`), so this is well-defined as a follow-up: branch
  `AmiId` resolution by the CNG's instance architecture (or expose an `arm64` toggle),
  add an arm64 Enroot/Pyxis first-boot path (`assets/scripts/install-enroot-pyxis.sh` is x86
  only today), and validate hpc7g + c7gn end-to-end on real hardware.
- [ ] 🟡 **P6e-GB200 / P6e-GB300 (Grace-Blackwell) support.** Add node-group templates for
  the GB200/GB300 NVL instances (e.g. p6e-gb200.36xlarge). These are Grace (arm64) CPUs
  with a different NIC/EFA layout (e.g. p6e-gb200 = 17 network cards) and likely need an
  arm64 PCS-Ready DLAMI and arm64 Enroot/Pyxis builds — validate the AMI, EFA, and a
  sample run.
- [ ] 🟢 **Consolidate the per-family GPU add-cng templates (p4d / p5 / p6-b200 / p6-b300).**
  The four `add-cng-p4d`/`add-cng-p5`/`add-cng-p6*` templates are ~85% identical; the real difference
  is the `NetworkInterfaces` EFA layout (card count, whether card 0 is EFA or ENA-only, and
  the EFA DeviceIndex). They are kept separate today so each NIC list stays flat and
  hand-checkable against the EC2 docs. Investigate generating the interface list from a
  per-instance-type mapping with `Fn::ForEach` (`AWS::LanguageExtensions`) so a new GPU
  family is a one-line mapping entry — but first confirm the required `CAPABILITY_AUTO_EXPAND`
  does not break the README/workshop one-click quick-create links.
- [ ] 🟡 **Document/provision the IAM permissions deploy-all needs.** A one-click /
  deploy-all run creates IAM roles, PCS clusters, EC2/VPC/FSx, Image Builder, SSM, etc.
  Document the minimum deploying-principal permissions (and provide a ready-made policy or
  a deploy-role CloudFormation/managed policy), so users in restricted accounts can grant
  exactly what's required instead of needing broad admin.
- [ ] 🟡 **Client-side Lustre-on-EFA + GDS support (P5 / P5e / P5en / P6-B200).**
  `FSxLustreEnableEfa=true` configures the *FSx server side* (PERSISTENT_2 EfaEnabled).
  The *client side* — installing the Lustre client + EFA modules, configuring LNet over
  EFA via the AWS-provided `setup.sh --optimized-for-gds`, and (for GDS) building/loading
  `nvidia-fs.ko` with `cufile.json` — is currently out of scope and not handled by
  `install-enroot-pyxis.sh`. Add a new opt-in lifecycle-action script
  (e.g. `scripts/install-fsx-lustre-efa.sh`) that runs the
  [official FSx EFA client setup](https://docs.aws.amazon.com/fsx/latest/LustreGuide/configure-efa-clients.html)
  and the GDS driver build, surface a `OnDemandEnableFSxLustreEfaClient` /
  `PseriesEnableFSxLustreEfaClient` toggle to invoke it, and validate end-to-end with:
  - **GDSIO** — direct GPU-to-storage path, target the ~78-94 GiB/s read on a 96 TiB
    filesystem from the reference repo
  - **ior** — POSIX / MPIIO bandwidth on `/fsx`, multi-process / multi-node, ranks
    binding to local EFA NICs (validates the EFA path is actually carrying the I/O,
    not falling back to TCP)
  - **mdtest** — metadata IOPS on `/fsx` (file create/stat/remove rates), exercises
    PERSISTENT_2's metadata-configuration path that is required for EfaEnabled
  Reference design + 8x H200 throughput numbers (~78-94 GiB/s on a 96 TiB filesystem) at
  [aws-samples/sample-fsx-lustre-gds-sharded-model-loading](https://github.com/aws-samples/sample-fsx-lustre-gds-sharded-model-loading).

## Software stack

- [ ] 🟡 **Spack as a first-class install option.** Today the cluster ships Enroot/Pyxis
  (containers) + the PCS-Ready DLAMI's pre-installed CUDA/NCCL/EFA stack, but no native
  package manager for HPC software (MPI variants, BLAS/LAPACK, scientific libraries,
  source-built apps). Add an opt-in `Spack` install path — e.g. a node lifecycle action
  variant that bootstraps Spack into shared `/fsx`, configures
  [aws-pcluster-spack](https://github.com/spack/spack-configs)-style external packages
  for PCS (Slurm, EFA libfabric, FSx for Lustre client), and uses the
  `aws-pcluster-` compiler + EFA/NCCL targets so binaries are tuned for the instance
  family. Single shared install on `/fsx` works for the whole cluster, so this fits
  cleanly alongside the existing layout. Validate on at least one CPU + one GPU node.
- [ ] 🟢 **Intel oneAPI (HPC Toolkit) install option.** For users running ICC/IFX/MPI/MKL
  workloads, add an opt-in install path (apt repo or shared `/fsx` install) that places
  Intel oneAPI HPC Toolkit on the cluster, with `module load`-style discoverability that
  composes with the Spack option above. Likely a separate
  lifecycle-action script invoked by users explicitly (large download, not
  every cluster needs it).
- [ ] 🟢 **NVIDIA HPC SDK install option.** Same shape as the Intel one — opt-in
  install of the NVIDIA HPC SDK (nvhpc, nvfortran, NCCL/CUDA-aware MPI variants) for
  GPU clusters that build their own apps. Less critical than Spack since Pyxis containers
  already cover most NVIDIA-stack use cases, but useful for native-build workflows.
- [ ] 🟢 **Module system (Lmod / environment-modules).** Once Spack and/or the Intel /
  NVIDIA toolkits land, ship a working `module avail` so users can switch toolchains the
  way they would on a traditional HPC system instead of editing `PATH` by hand.

## User management

- [x] 🟡 **Integrate a user-management backend (LDAP/AD).** Done for OpenLDAP:
  `DirectoryService=OpenLDAP-LoginNode` runs slapd on the login node (DB on shared
  `/home/ldap-db`) with SSSD on all compute nodes (CPU + GPU). Users added via
  `ldap-add-user` resolve cluster-wide; home dirs auto-create; Slurm sees LDAP users
  transparently. See `docs/USER-MANAGEMENT.md`. *(Follow-up: managed-directory options
  `DirectoryService=SimpleAD`/`ManagedAD` for multi-login-node / HA — the param enum is
  already extensible.)*
- [ ] 🟡 **Stable LDAP endpoint across login-node replacement (OpenLDAP-LoginNode).**
  Compute clients bake the login node's **private IP** into SSSD's `ldap_uri`
  (`setup-directory.sh` `setup_client_internal`). The user DB on `/home/ldap-db` survives a
  login-node replacement, but the new node gets a **new private IP**, so already-running
  compute nodes keep a stale `ldap_uri` — cached users still resolve (`cache_credentials`),
  but uncached/new users and group expansion fail until each compute node is rebooted or
  SSSD is reconfigured. (Newly-booting compute nodes are fine: discovery now re-reads the
  `directory-role=server` tag, with retry.) Upstream's
  [`dir/demo_openldap`](https://github.com/aws-samples/aws-hpc-recipes/tree/main/recipes/dir/demo_openldap)
  avoids this by fronting the directory with an **NLB (stable DNS)**, so the client URI
  never changes when the backend restarts. Options for ml-pcs: (a) put the login IP behind a
  **Route 53 private hosted-zone record** the replacement updates, and point `ldap_uri` at
  the DNS name; (b) have running compute nodes periodically re-resolve the tag and rewrite
  `ldap_uri`; or (c) defer to the managed-directory path (SimpleAD/ManagedAD) which provides
  a stable endpoint by design. Until then, document the post-replacement recovery
  (`sss_cache -E` + `systemctl restart sssd` across compute, or node replacement) — see
  `docs/USER-MANAGEMENT.md`. *(The admin-password regen-on-replacement issue is already
  fixed: `setup_server` reuses the existing SSM password instead of generating a new one.)*

## Monitoring

- [ ] 🟡 **AWS-managed monitoring stack option.** Offer Amazon Managed Service for
  Prometheus + Amazon Managed Grafana as an alternative to the self-hosted stack on the
  login node (see `observability/prometheus-grafana`), so users can
  use a managed backend instead of running the containers themselves.
- [ ] 🟢 **Persist Prometheus TSDB / Grafana DB across login-node replacement.** Monitoring
  runs on the login node and is otherwise replacement-safe (compute scraping is pull-based
  via EC2 service discovery on the `aws:pcs:cluster-id` tag — no compute→login IP
  dependency; the Grafana password is reused from SSM; the stock dashboards + datasource are
  file-provisioned). The one gap: the **Prometheus TSDB and Grafana DB live in node-local
  Docker named volumes** (`/var/lib/docker/volumes/...`, deliberately node-local to avoid the
  shared-`/home` Stale-file-handle race that motivated the `/opt` install), so a replacement
  **loses historical metrics and any user-created/edited Grafana state** (custom dashboards,
  edits to the stock ones, annotations, alert rules). Options: (a) the managed AMP/AMG backend
  above (best — storage is off-node by design); (b) bind the TSDB/Grafana volumes to an EBS
  volume that re-attaches on replacement; (c) periodic `slapcat`-style export of dashboards +
  TSDB snapshots to `/fsx`. **Do NOT** simply move the volumes onto `/home` — that reintroduces
  the NFS Stale-file-handle race the `/opt` install was created to fix. Documented as a known
  limitation in [OPERATIONS.md §3.2](./OPERATIONS.md). Verified on a real login-node
  replacement (2026-06): discovery/password/dashboards recovered automatically; only history
  was lost.
- [x] 🟡 **Rename `DeployMonitoring` → `MonitoringStack` (enum).** Done in deploy-all:
  `MonitoringStack: none | Prometheus-LoginNode` (default `Prometheus-LoginNode`),
  aligning with the `DirectoryService` `<what>-<where>` pattern. `AMP-AMG`/`CloudWatch`
  remain as future AllowedValues for the managed-monitoring item above. deploy-all
  converts to the nested templates' `DeployMonitoring=true/false` internally, so
  add-cng*.yaml are unchanged. **Breaking change** at the deploy-all interface
  (bundled into the major-update PR alongside `GrafanaPublicAccessCidr`→`GrafanaAccessCidr`
  and `SSHAccessCidr`).

## Testing / docs

- [ ] 🟡 **Automate the validation matrix.** The `tests/` guide is run manually; add a
  script that deploys, runs the CPU/GPU/NCCL/FSDP checks, and asserts the expected results
  for CI-style regression testing.
