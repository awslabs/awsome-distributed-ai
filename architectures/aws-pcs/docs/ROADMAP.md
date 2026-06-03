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

## Monitoring

- [ ] 🟡 **AWS-managed monitoring stack option.** Offer Amazon Managed Service for
  Prometheus + Amazon Managed Grafana as an alternative to the self-hosted stack on the
  login node (see `4.validation_and_observability/4.prometheus-grafana`), so users can
  use a managed backend instead of running the containers themselves.

## Testing / docs

- [ ] 🟡 **Automate the validation matrix.** The `tests/` guide is run manually; add a
  script that deploys, runs the CPU/GPU/NCCL/FSDP checks, and asserts the expected results
  for CI-style regression testing.
