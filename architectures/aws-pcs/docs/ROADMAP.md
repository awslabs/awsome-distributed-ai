# AWS PCS — Roadmap / TODO

Implementation items under consideration for future work on the AWS PCS templates
(`architectures/aws-pcs`). This is a living checklist — add items via PR so the history is
captured in git, and check them off or remove them when done.

Priority: 🔴 high · 🟡 medium · 🟢 low

## Templates & deployment

- [ ] 🟡 **Multi-AZ support in the prerequisites stack.** `ml-cluster-prerequisites.yaml`
  currently creates a single private subnet, so `OpenZFSDeploymentType` excludes the
  `MULTI_AZ` types. Add a second private subnet (and the related routing) to enable
  Multi-AZ FSx and higher-availability deployments.
- [ ] 🟢 **Trainium (Trn) validation.** Validate the templates on Trainium instances
  (e.g. trn1/trn2) — node group, EFA/networking, and a sample training run.
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

- [ ] 🟡 **Integrate a user-management backend (LDAP/AD).** Provide a way to manage cluster
  users centrally instead of the single `ubuntu` user — e.g. integrate an LDAP/OpenLDAP or
  AWS Managed Microsoft AD directory (see `1.architectures/6.ldap_server`) so login/compute
  nodes authenticate against a shared directory (multi-user clusters, per-user home dirs,
  Slurm accounting per user).

## Monitoring

- [ ] 🟡 **AWS-managed monitoring stack option.** Offer Amazon Managed Service for
  Prometheus + Amazon Managed Grafana as an alternative to the self-hosted stack on the
  login node (see `4.validation_and_observability/4.prometheus-grafana`), so users can
  use a managed backend instead of running the containers themselves.

## Testing / docs

- [ ] 🟡 **Automate the validation matrix.** The `tests/` guide is run manually; add a
  script that deploys, runs the CPU/GPU/NCCL/FSDP checks, and asserts the expected results
  for CI-style regression testing.
