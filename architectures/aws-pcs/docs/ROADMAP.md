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
- [ ] 🟡 **Targeted ODCR support for GPU node groups.** Today `CapacityReservationId`
  on `add-cng-p5`/`add-cng-p6-b200`/`add-cng-p6-b300` is **Capacity Block for ML only** —
  setting it forces `MarketType=capacity-block` and drops the placement group, so a
  *targeted* On-Demand Capacity Reservation (ODCR) cannot be consumed (only "open" ODCRs,
  via the empty/On-Demand path, work). Add a `CapacityReservationType` enum
  (`none` | `capacity-block` | `targeted-odcr`) and branch the launch template:
  `targeted-odcr` sets `CapacityReservationTarget` **without** `MarketType=capacity-block`
  and **keeps** the placement group (On-Demand billing against the reservation).
  `none`/`capacity-block` stay equivalent to today (backward compatible). Replaces the
  current "do not put an ODCR ID here" caveat. Verification can be done **without GPU
  capacity**: (1) static — create the GPU CNGs with `Min/MaxCount=0` and assert the
  generated launch template's `CapacityReservationSpecification`/`InstanceMarketOptions`/
  `Placement` per type; (2) dynamic — the branch logic is instance-family-independent, so
  exercise actual targeted-ODCR consumption (`InstanceLifecycle` empty = On-Demand, reserved
  count decrements) on a cheap type (c6i/g5).
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
- [ ] 🟢 **Consolidate the per-family GPU add-cng templates (p5 / p6-b200 / p6-b300).**
  The three `add-cng-p6*`/`add-cng-p5` templates are ~85% identical; the real difference
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
  `install-enroot-pyxis.sh`. Add a new opt-in post-install path
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
  source-built apps). Add an opt-in `Spack` install path — e.g. a `PostInstallScriptUrl`
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
  `PostInstallScriptUrl`-style script invoked by users explicitly (large download, not
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

## Monitoring

- [ ] 🟡 **AWS-managed monitoring stack option.** Offer Amazon Managed Service for
  Prometheus + Amazon Managed Grafana as an alternative to the self-hosted stack on the
  login node (see `4.validation_and_observability/4.prometheus-grafana`), so users can
  use a managed backend instead of running the containers themselves.
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
