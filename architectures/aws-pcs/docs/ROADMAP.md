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
  arm64 PCS-ready DLAMI and arm64 Enroot/Pyxis builds — validate the AMI, EFA, and a
  sample run.
- [ ] 🟡 **Document/provision the IAM permissions deploy-all needs.** A one-click /
  deploy-all run creates IAM roles, PCS clusters, EC2/VPC/FSx, Image Builder, SSM, etc.
  Document the minimum deploying-principal permissions (and provide a ready-made policy or
  a deploy-role CloudFormation/managed policy), so users in restricted accounts can grant
  exactly what's required instead of needing broad admin.

## User management

- [ ] 🟡 **Integrate a user-management backend (LDAP/AD).** Provide a way to manage cluster
  users centrally instead of the single `ubuntu` user — e.g. integrate an LDAP/OpenLDAP or
  AWS Managed Microsoft AD directory (see `architectures/6.ldap_server`) so login/compute
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
