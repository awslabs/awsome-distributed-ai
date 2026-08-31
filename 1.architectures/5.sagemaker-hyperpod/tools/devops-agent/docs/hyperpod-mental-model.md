# HyperPod Mental Model (for AI assistants)

This document captures the operational mental model an AI needs to work
on HyperPod clusters without falling into common traps. It covers what
HyperPod is, who owns what, how data flows, and most importantly **what
AIs typically get wrong** because their training data treats HyperPod as
"basically EC2 + Slurm/EKS" — which is only partially true.

Keep this file open while doing any HyperPod testing or debugging. Update
it when you discover a new source of confusion.

## Audience and scope

This doc is intended for anyone — human operator or AI assistant —
working on a HyperPod cluster who needs more than what the product
docs alone cover: a shared mental model of how HMA detection,
lifecycle scripts, the cluster agent, orchestrator state, CloudTrail,
and system signals fit together.

Entries that apply to only one orchestrator (Slurm or EKS) are tagged
inline.

## Ownership boundaries (the #1 thing AIs miss)

The most consequential surprise about HyperPod is **the EC2 instances
that run your workloads are NOT in your AWS account.** They're owned by
the HyperPod service account. The customer account holds:

- The **SageMaker cluster resource** (visible via `aws sagemaker describe-cluster`)
- The **VPC, subnets, security groups** that the cluster uses
- The **ENIs** that HyperPod attaches into your VPC for each cluster node
- The **EBS / FSx storage** that you provisioned
- The **lifecycle-script S3 bucket** (a regular customer S3 bucket)
- IAM roles and execution roles used by the cluster

But the cluster's **EC2 instances themselves are in the HyperPod service
account**. The customer sees ENIs in their VPC and gets SageMaker-API
visibility into the nodes, but `aws ec2 describe-instances` from the
customer's account will not show them.

### Concrete consequences

| Question | Wrong answer (default AI assumption) | Right answer |
|---|---|---|
| "How do I SSH into a HyperPod node?" | `ssh ec2-user@<ip>` after grabbing the keypair from `aws ec2 describe-instances` | You can't SSH directly. Use `aws ssm start-session --target sagemaker-cluster:{cluster-id}_{instance-group-name}-{instance-id}` |
| "How do I get the EC2 topology of a HyperPod node?" | `aws ec2 describe-instance-topology --instance-ids ...` | That API doesn't work for HyperPod-owned instances. The topology data is delivered to the cluster agent at provisioning time and stored in `/opt/ml/config/resource_config.json` on the controller as `InstanceLabel.network-node-layer-{1,2,3}`. |
| "How do I see CPU metrics for a HyperPod node?" | `aws cloudwatch get-metric-data --namespace AWS/EC2 ...` | Metrics are published under the `ClusterAgent` and `SagemakerHealthMonitoringAgent` CloudWatch namespaces, not `AWS/EC2`. |
| "How do I terminate a faulty HyperPod node?" | `aws ec2 terminate-instances ...` | Use `aws sagemaker batch-replace-cluster-nodes` (or, in some cases, `batch-delete-cluster-nodes`). |
| "Why doesn't my IAM policy on `ec2:DescribeInstances` apply?" | Should — the nodes are EC2. | The IAM principal that calls the EC2 API is the HyperPod service account, not yours. Customer IAM policies on EC2 don't apply to HyperPod-owned instances. |

### Mental shortcut

> **Anything related to the EC2 lifecycle of cluster nodes goes through
> the SageMaker `aws sagemaker ...` APIs, not the EC2 API.** ENIs,
> security groups, and the surrounding VPC are still yours; the instances
> are not.

## Cluster types

HyperPod supports two orchestrators:

### Slurm clusters

- One or more **controller nodes** (typically `ml.m5.12xlarge` or similar)
  running `slurmctld` plus the SageMaker `ClusterAgent` and
  `SageMakerHealthMonitoringAgent` services.
- Multiple **worker instance groups**, each backed by a fleet of
  homogeneous instances (e.g. `worker-group-1: 4× ml.p5.48xlarge`,
  `worker-group-2: 2× ml.m5.xlarge`).
- A shared FSx for Lustre mount at `/fsx` (usually) for cross-node files.
- Optional login nodes.

Operator access is via SSM to the controller, then standard Slurm
commands. Job submission is via `sbatch`/`srun` from the controller or a
login node.

### EKS clusters

- A managed EKS control plane (in the customer account).
- HyperPod manages the worker node groups (still on HyperPod-owned EC2).
- Workloads run as pods; access is via `kubectl` configured against the
  cluster's EKS endpoint.
- HyperPod adds custom labels and an Operator that handles node
  health/replacement.

A single cluster is ONE type; you don't mix Slurm and EKS in the same
SageMaker cluster.

## Key services

The two orchestrators have **fundamentally different on-host agent
architectures**. This is one of the biggest sources of cross-orchestrator
confusion: don't assume a service that exists on Slurm runs the same way
on EKS, or vice versa.

### Slurm node services (systemd-based)

Slurm HyperPod nodes run systemd units, but **which units are active
depends on the node's role** (controller vs compute vs login). The
unit files for most agents are installed on every node, but several
are only *started* on the controller. Don't assume "the service is
present" means "it's running."

**Services running on every Slurm node (controller + compute + login):**

- **`sagemaker-host-agent.service`** (binary
  `/usr/bin/SageMakerHostAgent`) — per-node companion to the cluster
  agent. Listens on `*:4000` inside the `sagemaker_agent_namespace`
  netns. Reads `/opt/ml/config/host_agent.json` (which on this version
  only contains `ResourceConfigS3Uri`).
- **`sagemaker-health-monitoring-agent.service`** (HMA, binary
  `/usr/bin/SageMakerHealthMonitoringAgent`) — watches for hardware
  faults (NVIDIA Xid, EFA errors, kernel deadlocks, OOM events).
  Listens on `127.0.0.1:8888` in the netns. Triggers `Action:Reboot`
  or `Action:Replace` on the node and propagates to the HyperPod
  control plane. Only present on AMIs that ship with HMA — older AMIs
  ship without it (`systemctl status sagemaker-health-monitoring-agent`
  to confirm).
- **`sagemaker-role-proxy-agent.service`** — proxies IAM role
  credentials.
- **`sagemaker-fluent-bit.service`** — log shipping.

**Controller-only running services:**

- **`sagemaker-cluster-agent.service`** (binary
  `/usr/bin/SageMakerClusterAgent`) — owns Slurm config reconciliation.
  Listens on `*:4001` in the `sagemaker_agent_namespace` netns. Reads
  `/opt/ml/config/resource_config.json` and
  `/opt/ml/config/cluster_agent.json`. Writes `slurm.conf` and
  `topology.{conf,yaml}` on the controller. **The unit file is
  installed on workers too, but exits with status 0 immediately after
  start (`Active: inactive (dead)`)** — the binary at
  `/usr/bin/SageMakerClusterAgent` exists everywhere but only runs as
  a daemon on controllers. Don't grep for `:4001` on workers; it isn't
  bound.
- **`slurmctld.service`** — the Slurm controller daemon.
- **`slurmdbd.service`** — the Slurm accounting DB daemon. Only on
  the controller.
- **`slurm_exporter.service`** — Prometheus exporter for Slurm
  metrics. Only on the controller.

**Services running on compute and login (but not controller):**

- **`slurmd.service`** — the Slurm compute daemon. Not running on the
  controller (its local `slurmd.log` is empty). **Runs on login nodes
  too**: HyperPod registers the login instance into the Slurm cluster
  as a regular compute target, so login nodes can have jobs scheduled
  onto them via the `Nodes=ALL` default partition (e.g. `dev`). If you
  don't want that, partition definitions or node `Features`/`Weight`
  need to be set explicitly. See "Login node scheduling caveat" below.
  - **slurmd uses Slurm's configless mode**: `ExecStartPre` deletes
    any local `/opt/slurm/etc/slurm.conf`, then `ExecStart` runs
    `/opt/slurm/sbin/slurmd -D --conf-server <controller-customer-ip>`.
    `slurmd` pulls `slurm.conf` (and `cgroup.conf`, `gres.conf`,
    `plugstack.conf`) from the controller at boot and caches them in
    `/var/spool/slurmd/conf-cache/`. The controller enables this by
    setting `SlurmctldParameters=enable_configless` in its
    `slurm.conf`. Debugging implications:
    - There is **no `slurm.conf` on disk** under `/opt/slurm/etc/` on
      compute or login — that's by design, not missing-file evidence.
    - Read `/var/spool/slurmd/conf-cache/slurm.conf` to see what the
      node is actually running with.
    - "Worker can't find slurm.conf" → suspect the controller
      `--conf-server` IP is wrong/unreachable, or `slurmctld` is down,
      not a "missing shared storage" problem.

**Compute-node-only (NOT on login):**

- **`custom-health-monitor.service`** (description: "HyperPod Slurm
  Custom Health Monitor", script `/usr/local/bin/custom-health-monitor.sh`).
  Per-node health watcher that complements HMA on Slurm compute nodes.
  Verified present on `worker1` instance group, absent on `login`.
  Likely scoped by instance-group naming convention or per-IG LCS step.

**Binaries installed on every Slurm node (but not necessarily running):**

`SageMakerClusterAgent`, `SageMakerHealthMonitoringAgent`,
`SageMakerHostAgent`, `SageMakerJobMonitoringAgent` — all under
`/usr/bin/`. Presence of a binary does not mean the service is
running; check `systemctl is-active <unit>` first.

**Netns port summary (Slurm):**

| Port (in `sagemaker_agent_namespace`) | Bound by | Where |
|---|---|---|
| `*:4000` | `SageMakerHostAgent` | controller + compute + login |
| `*:4001` | `SageMakerClusterAgent` | **controller only** |
| `127.0.0.1:8888` | `SageMakerHealthMonitoringAgent` | every node with HMA |
| `127.0.0.1:9081` | java (HMA support process) | every node with HMA |

The cluster agent's listener is reachable only from inside
`sagemaker_agent_namespace` — connecting to `127.0.0.1:4001` from the
default netns returns `Connection refused`.

### EKS node services (mostly pod-based, NOT systemd)

The EKS shape differs in several material ways.

**On-host systemd services that exist on an EKS worker:**

- **`hyperpod-host-agent.service`** (binary `/usr/bin/hyperpod-host-agent`)
  — the EKS counterpart to `sagemaker-cluster-agent.service`, but
  **not the same agent**. Listens on `127.0.0.1:4000` (not 4001)
  inside the `sagemaker_agent_namespace` netns. There is no host-side
  "kick the agent to reconfigure" recipe on EKS; reconfiguration is
  driven via the K8s control plane.
- **`hyperpod-role-proxy-agent.service`** — proxies IAM role
  credentials for in-cluster workloads.
- **`sagemaker-cloudwatch-agent.service`** and
  **`sagemaker-fluent-bit.service`** — log/metric shippers; same as
  on Slurm.
- **`kubelet`** — managed by EKS, not customer. Started with
  `--hostname-override=hyperpod-i-<instance-id>` so the k8s node name
  is derived from the EC2 instance ID (see "Identity stability"
  section — this is why the EKS node name *changes* on replacement
  even though the prefix pattern looks stable).

**On-host services that do NOT exist on EKS — don't grep for these:**

- `sagemaker-cluster-agent.service` — not present.
- `sagemaker-health-monitoring-agent.service` — not present as a
  systemd unit; HMA runs as a DaemonSet (see below).

**HyperPod-managed pods (DaemonSets) on EKS:**

- **HMA DaemonSet** — `health-monitoring-agent` and
  `health-monitoring-agent-non-nvidia` in namespace `aws-hyperpod`.
  Container image
  `905418368575.dkr.ecr.<region>.amazonaws.com/hyperpod-health-monitoring-agent`.
  `hostPath`-mounts the node's `/var/log` (rw), `/dev/kmsg` (ro), and
  `/etc/localtime` (ro) so it can still read kernel Xid lines and
  syslog from inside the pod.
  - **Node-selector gotcha**: HMA only schedules on GPU/Neuron
    instance types (matchExpressions on `node.kubernetes.io/instance-type`
    against a fixed allowlist of `p4d/p4de/p5*/p6*/g5*/g6*/gr6*/g6e*/g7e*`
    variants). **CPU-only workers (m5, c5, etc.) have HMA DaemonSet
    pods with `Desired=0` — HMA is not running on them at all.** Don't
    expect HMA fault signals from non-accelerator nodes.
- **HyperPod training operator** — `hp-training-operator-...` pod in
  `aws-hyperpod` namespace. Handles training-CRD lifecycle.
- **EFA / NVIDIA / Neuron device plugins** —
  `dependencies-aws-efa-k8s-device-plugin`,
  `dependencies-nvidia-device-plugin`,
  `neuron-device-plugin-daemonset` in `kube-system`; scheduled by
  instance type the same way HMA is.

**Where "cluster agent" responsibility lives on EKS:** there is no
single host-side cluster agent. Node-label reconciliation, health-label
application, and the response to `Action:Reboot`/`Action:Replace`-style
triggers are driven by HyperPod's service-side control plane and the
in-cluster operator pods — the customer-visible surface is the
`sagemaker.amazonaws.com/*` labels on each node and the
`list-cluster-events` API.

## Key config files

### Slurm

| Path | Purpose | Controller | Compute / Login |
|---|---|---|---|
| `/opt/ml/config/resource_config.json` | The canonical view of cluster instances. On Slurm: `ClusterType: "Slurm"`, has a `SlurmConfig` block (`PrimaryControllerIp`, `SecondaryControllerIps`), instance entries with `InstanceName` (Slurm-style `<group>-<index>`), `InstanceId`, `AgentIpAddress` (agent-network IP), `CustomerIpAddress` (customer-VPC IP). **Caveat:** the on-node copy can be stale — workers may show instance groups that have since been scaled to 0. The controller's view is the most current. | yes | yes (can lag controller) |
| `/opt/ml/config/cluster_agent.json` | ClusterAgent settings (`ResourceConfigS3Uri`, `SlurmConfiguration.EnableTopology`, `TopologyLabel`, `UseInstanceNames`, `CreateMagneticReservations`). | yes | **no** |
| `/opt/ml/config/host_agent.json` | HostAgent settings — on this version, just `{"ResourceConfigS3Uri": "..."}`. | yes | yes |
| `/opt/ml/config/life_cycle_config.json` | LCS bundle reference (the `S3Uri` + script names the agent will fetch and run at provision time). | yes | yes |
| `/opt/ml/config/provisioning_parameters.json` | Cluster-creation parameters consumed by the LCS during provisioning (head-node IPs, partition layout, etc.). | yes | yes |
| `/opt/ml/metadata/resource-metadata.json` | **Often documented but not observed on disk** on either controller or compute under steady-state on the AMI versions verified — `/opt/ml/metadata/` itself is absent. Treat any LCS reference to this path as needing verification on the actual AMI in use. | (absent) | (absent) |
| `/var/log/aws/clusters/sagemaker-cluster-agent.log` | ClusterAgent's structured log. | yes | (file may exist but log empty — agent doesn't run as daemon) |

### EKS (on every node)

Only a subset of the `/opt/ml/config/*` files exist on EKS workers,
and the `resource_config.json` schema is different.

| Path | Present? | Notes |
|---|---|---|
| `/opt/ml/config/resource_config.json` | yes | EKS-shaped: `ClusterType: "Kubernetes"`, has a `KubernetesConfig` block (`EksClusterName`, `EksClusterEndpoint`, `EksClusterCertificateAuthority`, `EksServiceIpv4Cidr`). Per-instance entries expose `InstanceName` (the k8s hostname, `hyperpod-i-<id>`), `CustomerIpAddress` (customer-VPC IP), `AgentIpAddress` (HyperPod agent-network IP). `InstanceLabel` is `null` on non-accelerator nodes — `network-node-layer-N` only populated on topology-eligible SKUs. |
| `/opt/ml/config/life_cycle_config.json` | yes | Same purpose as Slurm. |
| `/opt/ml/config/cluster_agent.json` | **no** | Slurm-controller-only. |
| `/opt/ml/config/host_agent.json` | **no** | Slurm-only. |
| `/opt/ml/config/provisioning_parameters.json` | **no** (not observed) | If present, only during provisioning. |
| `/opt/ml/metadata/` (and `resource-metadata.json`) | **no** | The whole directory is absent on EKS workers. |
| `/var/log/aws/clusters/sagemaker-cluster-agent.log` | **no** | There is no cluster-agent process on EKS. |

> **LCS bundle staging**: the LCS bundle is downloaded into
> `/tmp/<s3-bucket-name>/<s3-prefix>/` on the node (the bucket and
> prefix come from the `S3Uri` in `/opt/ml/config/life_cycle_config.json`)
> and executed from there. The files remain on disk after the run
> finishes, but they live under `/tmp/` and may disappear on instance
> reboot. The durable record of an LCS run is the CloudWatch log
> stream
> `/aws/sagemaker/Clusters/<cluster-name>/<cluster-id>/LifecycleConfig/<group>/<instance-id>`,
> not a file on disk. The canonical on-disk LCS log is
> `/var/log/provision/provisioning.log` (a copy of the CloudWatch
> stream for the most recent run).

### Slurm config files by node role

For Slurm log files by node role, see "Logs (via SSM and CloudWatch)"
below — all log paths live there as the canonical source.

| Path | Purpose | Controller | Compute / Login |
|---|---|---|---|
| `/opt/slurm/etc/slurm.conf` | Standard Slurm config. Includes `TopologyPlugin`, `SchedulerParameters`, `PartitionName=...`, and per-node `NodeName=ip-N-N-N-N` lines keyed by the customer-VPC IP. Written by ClusterAgent during reconfigure. Controller-side `SlurmctldParameters=enable_configless` lets workers/login fetch this file via configless mode. | yes | **no** — `slurmd`'s `ExecStartPre` actively deletes any local `/opt/slurm/etc/slurm.conf` at boot, and `slurmd --conf-server <controller>` re-fetches into `/var/spool/slurmd/conf-cache/`. Absence is intentional. |
| `/var/spool/slurmd/conf-cache/slurm.conf` (+ `cgroup.conf`, `gres.conf`, `plugstack.conf`) | The actual config the on-node `slurmd` is running with, pulled from the controller in configless mode and cached locally. | no (controller doesn't run `slurmd`) | yes — this is where to look when debugging worker/login Slurm config |
| `/opt/slurm/etc/topology.conf` | Switch hierarchy for Slurm 24.x topology. Written only when topology applies — single-leaf or `topology/flat` clusters do not have this file. | yes (if topology) | no (compute/login pulls topology config via configless if present) |
| `/opt/slurm/etc/topology.yaml` | Multi-topology YAML for Slurm 25+. Written for heterogeneous clusters; overrides per-partition. Same caveat — not present on `topology/flat` clusters. | yes (if topology) | no |
| `/opt/slurm/etc/slurmdbd.conf` | Slurm DB daemon config. | yes | no |
| `/opt/slurm/etc/gres.conf` | GRES (generic resource) config — GPUs etc. May be empty on CPU-only clusters. | yes | no |

### Login node scheduling caveat

HyperPod login nodes are full Slurm compute members — they run
`slurmd --conf-server <controller>` and register themselves into the
cluster. The default `PartitionName=dev Nodes=ALL` line in
`slurm.conf` means **jobs submitted to `dev` can land on the login
node alongside interactive user sessions**. The login node only
appears in partitions whose `Nodes=` matches it (so SKU-named
partitions like `ml.m5.xlarge` may or may not include it depending
on how partition definitions are generated). If you want users to
have a job-free login experience, either:

- Submit jobs explicitly to a non-default partition (e.g. `--partition=worker1`).
- Add `Features=login` or `Weight` to keep `sbatch` from picking
  login nodes for `dev` jobs.
- Cordon equivalent: `scontrol update NodeName=<login-ip> State=DRAIN Reason="login-only"`.

In practice: a freshly scaled-up `login` instance joins the `dev`
partition as `idle` immediately and is eligible to run any job
submitted without `--partition=`.

### Operator edits to `slurm.conf` may not survive

*(Slurm only)*

Edits to `slurm.conf` made by an operator MAY be overwritten by
ClusterAgent on a future reconfigure event. The public AWS doc warns:
> *"manual changes may be overwritten by HyperPod during subsequent
> cluster updates, including scaling operations, node replacements, and
> other cluster lifecycle events."*

The exact set of fields the agent rewrites vs. preserves is not fully
documented. If you need a `slurm.conf` change to survive cluster
lifecycle events, verify it survives each event class you care about
(reconfigure, scaling, replacement, `UpdateClusterSoftware`) on a test
cluster before relying on it.

### EKS-only

Cluster state is in the EKS API, not a flat file. Capture via `kubectl
get ... -o yaml` and treat the YAML output as the config artifact:
node labels, taints, ConfigMaps, CRDs the HyperPod operator manages,
and any feature-specific resources.

## Resiliency model — how a fault becomes a recovery

This is the cross-source narrative most AI agents lack when an issue
spans HMA / orchestrator / lifecycle / API. It applies in both
directions: forward ("a GPU just threw Xid 79, what happens next?") and
backward ("the node disappeared, what fired and where do I look?").

### Canonical detection-to-recovery flow

1. **Hardware signal arrives on the node.** Kernel Xid line in
   `dmesg`/syslog, DCGM field violation, EFA fabric error, kernel
   deadlock, OOM killer, etc.
2. **HMA detects on its next scan cycle.** HMA is **passive and
   cycle-based** — it doesn't react instantly; "wait for next cycle" is
   normal behavior. Detection examples: GPU count mismatch via
   `nvidia-smi`, Neuron device count mismatch via `neuron-ls`, ECC UCE,
   specific Xid codes.
3. **HMA marks the node** as needing reboot or replace. Visible as:
   - **Slurm**: node state `fail`/`drain` with `Reason="Action:Reboot"`
     or `Reason="Action:Replace"` (exact case-sensitive match — see
     "Common AI-confusing details").
   - **EKS**: node labels + taint:
     `sagemaker.amazonaws.com/node-health-status=Unschedulable:NoSchedule`,
     `fault-types`, `fault-reasons`, plus a `fault-details` annotation
     (JSON array of recent faults).
4. **HyperPod decides whether to act.** Gated on cluster-level
   `NodeRecovery=Automatic | None`. With `None`, the labels appear but
   no recovery is triggered.
5. **HyperPod issues the underlying EC2 call** (reboot or
   terminate-and-launch) via the service principal — **not visible in
   customer-account CloudTrail under any customer principal**. The
   customer-visible record of this is the `list-cluster-events` API
   (EKS) and `list-cluster-nodes` status transitions (both).
6. **On replace**, lifecycle scripts re-run on the fresh instance.
   Output streams to CloudWatch and `/var/log/provision/provisioning.log`
   on the new node.
7. **On reboot**, lifecycle scripts do NOT re-run. The agent + slurmd
   (or kubelet) restart and the node re-joins.
8. **Reconciliation happens** — on **Slurm**, the cluster agent
   rewrites `slurm.conf` and topology files. On **EKS**, there is no
   host-side cluster agent; the HyperPod service-side control plane
   re-asserts node labels and the in-cluster operator reconciles
   training-CRD state.
9. **Orchestrator reflects** the node back to `idle`/`Ready`.

A common AI failure mode — drifting between interpretations like
"this is a Slurm issue", "this needs a manual API call", and
"HyperPod will replace the node automatically" — comes from stopping
at step 3 (the orchestrator side) without recognizing steps 4–6
(the HyperPod-managed actions that have no customer CloudTrail
signature).

### Reverse interpretation rules (what does this signal mean?)

Use these to reconstruct what happened from a partial set of signals:

- **Slurm `Reason="Action:Reboot"`/`Action:Replace`** appearing on a
  node = HMA, the operator, or another in-cluster trigger has asked
  HyperPod to take that action. HyperPod will act only if
  `NodeRecovery=Automatic`.
- **HMA event in CloudWatch (`HealthMonitoringAgentDetectionEvent`)
  with no corresponding `Action:` reason on the node** = HMA detected
  something but didn't escalate to a recovery class.
- **`list-cluster-nodes` shows node `Pending` with no customer
  CloudTrail event in the matching time window** = HyperPod-service
  replacement in progress. The trigger was either HMA or another
  internal signal; look in CloudWatch logs to confirm.
- **`list-cluster-nodes` shows `Failed`** = replacement attempt
  completed unsuccessfully. The most common cause is an EFA health
  check failure during the new instance's boot (often transient — a
  manual `batch-replace-cluster-nodes` retry frequently succeeds).
  HyperPod does NOT auto-retry out of `Failed`.
- **Slurm view "healthy" / EKS view "Ready" but HyperPod API says
  `Pending`** = orchestrator is stale; an EC2 replacement is mid-flight
  but the new node hasn't fully registered yet.
- **Lifecycle script re-execution log lines** = the node was replaced
  (not just rebooted). Reboot leaves the lifecycle log untouched.

### Cross-source correlation table

When asked "what happened to this node?", consult these in order:

| Order | Source | What it tells you |
|---|---|---|
| 1 | `aws sagemaker list-cluster-nodes` + `describe-cluster-node` | Current HyperPod-side status (`Running`/`Pending`/`Failed`/`ShuttingDown`); current instance ID (changes on replace) |
| 2 | `aws sagemaker list-cluster-events` *(EKS; Slurm with Continuous Provisioning)* | The 500 most recent control-plane events with timestamps and reasons; up to 5 pages of 100. The canonical record of replacement attempts (including failed ones) — survives even when individual nodes disappear from `list-cluster-nodes` between retries. |
| 3 | CloudWatch log group `/aws/sagemaker/Clusters/<NAME>/<CLUSTER-ID>` | HMA events, lifecycle output, deep-health-check results (see "Logs (via SSM)" below for stream layout) |
| 4 | Slurm: `scontrol show node <ip>` / `sinfo -R`; EKS: `kubectl describe node <name>` | Orchestrator-side state and reason strings |
| 5 | On-node `/var/log/provision/provisioning.log`, `/var/log/syslog`, `dmesg`, `journalctl` | The actual hardware/system events that triggered everything |
| 6 | Customer CloudTrail | Customer-principal API calls (e.g. operator-triggered `BatchReplaceClusterNodes`). **Absence of a customer event when a node was rebooted/replaced is itself a signal** — the action was service-initiated. |

## Identity stability across reboot vs. replace

A high-leverage AI-confusing detail: **which identifiers survive each
recovery class?** The two orchestrators differ on several rows — chiefly
hostname/IP semantics on replace.

### Slurm

| Identifier | `Action:Reboot` | `Action:Replace` |
|---|---|---|
| EC2 instance ID | preserved | **new** |
| Private IP | preserved | typically preserved (ENI persists in the customer VPC; not contractually guaranteed) |
| Slurm NodeName | preserved | **preserved** ("the node is replaced with a fresh instance using the same host name"). `NodeName=` in `slurm.conf` is `ip-N-N-N-N` (derived from the customer-VPC IP), not the SageMaker `InstanceName` (`<group>-<index>`). The two diverge: `InstanceName` is what shows up in `resource_config.json`. |
| Root EBS volume | preserved | **destroyed** |
| Secondary EBS volumes (e.g. `/opt/sagemaker`) | preserved | **destroyed** |
| Instance store / NVMe (e.g. `/opt/dlami/nvme`) | block contents survive an OS reboot, but cloud-init / lifecycle scripts may re-mkfs the mount on boot | ephemeral (new EC2, new instance store) |
| Lifecycle scripts re-run? | no | **yes** |
| Training process state | interrupted | **destroyed** |
| Topology placement (which network leaf) | preserved | usually preserved, not guaranteed (see "Topology placement on replacement" below) |

### EKS

The k8s node name and OS hostname both encode the EC2 instance ID,
which **changes** on replace, so the EKS row for "node identity" is
not symmetric with Slurm.

| Identifier | `Action:Reboot` | `Action:Replace` |
|---|---|---|
| EC2 instance ID | preserved | **new** |
| Customer-VPC IP (`CustomerIpAddress`) | preserved | **not guaranteed preserved** (no contractual ENI-persistence on EKS replace; treat as a new IP) |
| Agent-network IP (`AgentIpAddress`) | preserved | **new** |
| OS-level `hostname` | preserved (`ip-<agent-ip>.<region>.compute.internal` — derived from the agent-network IP) | **new** |
| K8s node name (`kubelet --hostname-override`) | preserved (`hyperpod-i-<instance-id>`) | **new** (instance ID changed → new k8s Node object; old node disappears, fresh node joins) |
| Root EBS volume | preserved | **destroyed** |
| Secondary EBS volume `/opt/sagemaker` (holds kubelet + container state) | preserved | **destroyed** |
| Instance store / NVMe (e.g. `/opt/dlami/nvme`, GPU SKUs only) | block contents survive an OS reboot, but cloud-init / lifecycle scripts may re-mkfs the mount on boot | ephemeral (new EC2, new instance store) |
| Lifecycle scripts re-run? | no | **yes** |
| Training pod state | the pod is evicted/rescheduled by the orchestrator when the node taints | destroyed; pods re-scheduled onto the replacement node |
| Topology placement (which network leaf) | preserved | usually preserved, not guaranteed (see "Topology placement on replacement" below) |

**Practical EKS implication**: any customer tooling that pins by
`CustomerIpAddress`, OS hostname, or k8s node name will break on
replace. Pin by SageMaker `NodeId` (which is itself tied to the EC2
instance ID and changes too, but is the canonical identifier in the
HyperPod API surface) — or, for in-cluster pinning, by labels like
`sagemaker.amazonaws.com/instance-group-name` that survive replacement.

**Preconditions for `BatchReplaceClusterNodes`**:
- Cluster must have been patched at least once via `UpdateClusterSoftware`.
- Cluster must be `InService`.
- Slurm controller nodes cannot be replaced.
- Maximum 25 NodeIds per call.

**Customer guidance:** treat replacement as if the node never existed
before. Anything customers want to survive failure must live on FSx,
S3, EFS, or other shared/external storage — never on the node's local
volumes.

> **`UpdateClusterSoftware` AMI repaint wipes the root EBS the same
> way replacement does.** When `update-cluster-software` changes a
> node's AMI (default → custom, custom → custom v2, custom → default,
> etc.), the underlying primary EBS volume is rewritten from the new
> AMI — the EC2 instance ID can be preserved but the root filesystem
> from the previous boot is gone. This is **by design**, not a bug.
>
> The practical implication for debugging: any on-node log or journal
> from the pre-update boot is lost — `/var/log/*`, `journalctl` output
> for `SageMakerHostAgent` / `sagemaker-cluster-agent`, any LCS-bundle
> staging files under `/tmp/<bucket>/`, and anything else on the root
> volume. The **only durable evidence** of a failed AMI update is
> what was exported off-node before the rewrite: the CloudWatch LCS
> log stream
> `/aws/sagemaker/Clusters/<name>/<id>/LifecycleConfig/<group>/<id>`
> (always exported), plus whatever the customer wrote to FSx / S3 /
> EFS during the failed run. Don't expect to SSM back in after a
> rollback and find a journal explaining what killed the previous
> attempt — it's gone with the old root volume.

### `UpdateClusterSoftware` operational semantics (Continuous Provisioning mode)

Even after the "root EBS gets wiped" callout above, several `UpdateClusterSoftware` behaviors surprise testers and customer-facing tooling on the first encounter. Consolidated here:

**`SoftwareUpdateStatus` field on `describe-cluster` `InstanceGroups[]`.** This is the canonical per-IG signal for "an AMI update is in flight"; distinct from the generic `Status: Updating` that both `UpdateCluster` and `UpdateClusterSoftware` produce. Observed transitions:
- `n/a` (absent) → `InProgress` → `Succeeded` on the happy path.
- `n/a` → `InProgress` → `RollbackInProgress` (transient, visible only briefly) → `Failed` on failure.

`InProgress` alone is not enough to know the update is healthy; watch for `RollbackInProgress` or `Failed` at the tail. Poll every ~60 seconds; the terminal state usually lands 20–30 minutes after `update-cluster-software` returns.

**`InstanceStatus.Message` is sticky across retries.** When `UpdateClusterSoftware` retries after a first-attempt failure, `InstanceStatus.Status` transitions back to `SystemUpdating`, but `InstanceStatus.Message` retains the failure text from the previous attempt (e.g. `"Lifecycle scripts did not run successfully..."`). During the retry window an operator sees `Status=SystemUpdating` AND a failure `Message` simultaneously. Two readings are possible: (a) it's a bug — the message should clear at retry start; (b) it's intentional — the message is a last-failure-cause preserving debugability across retries. Either way, treat `Message` as "most recent failure cause, may or may not still be current" rather than "current status." Runbooks pinning on `Status` alone are safer than combining `Status` + `Message`.

**⚠ Likely bug (behavior subject to change): `describe-cluster-node` — but NOT `describe-cluster` — surfaces the data-plane's shared AMI ID during `update-cluster-software`.** Observed behavior during an in-flight `UpdateClusterSoftware` operation:

- `describe-cluster` → `InstanceGroups[].DesiredImageId` returns the AMI ID **the customer passed** to `--image-id`. Consistent across all three code paths (`create-cluster`, `update-cluster`, `update-cluster-software`).
- `describe-cluster-node` → `DesiredImageId` behavior depends on the code path:
  - `create-cluster` and `update-cluster` paths: returns the AMI ID **the customer passed** (their owned AMI).
  - `update-cluster-software` path: returns the **data-plane's shared copy** of the customer's AMI, which is a different AMI ID that `describe-images --profile <customer>` cannot resolve (the customer doesn't own it and can't see it).

Reported to the service team as a bug — expected to be fixed so `describe-cluster-node.DesiredImageId` returns the customer's AMI ID consistently across all three code paths. Until then, a customer comparing `describe-cluster-node.DesiredImageId` against the value they passed to `--image-id` will see a mismatch even when everything is working correctly. Two safe cross-checks: (a) use `describe-cluster` — its `DesiredImageId` matches the customer's input; (b) check `CurrentImageId` after the update settles — that returns to the customer's owned AMI on both APIs.

**Retry-then-rollback semantics on failure.** `UpdateClusterSoftware` has a built-in **auto-retry loop before it gives up** — unlike node replacement, which does NOT auto-retry out of `Failed` (see "Common AI-confusing details" for the replacement case). The observed sequence when the first LCS attempt on the new AMI fails:

1. **Multiple LCS attempts on the target AMI.** Each attempt fully re-images the node from the target AMI and re-runs the LCS bundle from scratch. Multiple `[SageMaker] Downloading lifecycle scripts` markers appear in the CloudWatch LCS stream (`/aws/sagemaker/Clusters/<name>/<id>/LifecycleConfig/<group>/<id>`) without matching `succeeded` markers between them. Each attempt is a fresh root-EBS rewrite.
2. **Rollback to the pre-update AMI** once the retry budget is exhausted. Rollback target is whatever the node was on before the update was invoked (e.g. `default → custom` fails → rolls back to `default`; `custom v1 → custom v2` would roll back to `custom v1`; old `default → new default` would roll back to old `default`). Only the `default → custom` failure path has been directly observed; the rollback shape for other directions is inferred from the observed case, not directly measured.
3. **Final LCS run on the rolled-back AMI**, which typically succeeds — same bundle, same target as the original boot — and emits a normal `[SageMaker] The lifecycle scripts succeeded.` marker in CloudWatch.
4. **Terminal state**: `SoftwareUpdateStatus=Failed` on the target IG. `CurrentImageId` and `DesiredImageId` both back to the pre-update value. EC2 instance ID preserved (rollback is software-state, not EC2 replacement).

Practical implications:

- **The CloudWatch LCS stream carries the whole history in one place.** Count `[SageMaker] Downloading lifecycle scripts` markers to see how many attempts were made; count `succeeded` markers to see how many completed. `attempts − succeeded > 1` means a real retry happened. The final `succeeded` (if any) is usually the rollback, not the last update attempt.
- **`SoftwareUpdateStatus=InProgress` covers the entire retry window**, not just a single attempt. If you're polling and see `InProgress` for 20+ minutes with the node cycling through `SystemUpdating` → transient `Failed` → `SystemUpdating` again, that's the auto-retry loop at work. Don't conclude the update is stuck just because you saw `Failed` briefly.
- **A `[SageMaker] The lifecycle scripts succeeded.` marker in CloudWatch doesn't mean the AMI update succeeded** — it might be the rollback's LCS run. To distinguish, check `SoftwareUpdateStatus`: `Succeeded` vs `Failed`. On-node, whether the "new" AMI's marker / customization is present is the most reliable "did the AMI update actually stick?" test — but the specific file to check depends on which direction the update was going.
- The retry budget (how many attempts before rollback, and by what time budget) has not been precisely characterized. Observed in one failure campaign: ~4 LCS attempts spread across ~18 minutes before rollback, on an `ml.m5.xlarge` worker. Treat as approximate; different regions, instance types, or provisioning-mode variants may differ.

## Storage layout — survives reboot vs. replace

HyperPod nodes typically have multiple volumes mounted:

| Mount | Volume type | Survives reboot? | Survives replace? |
|---|---|---|---|
| `/` (root, 100 GB default) | EBS | yes | **no** |
| `/opt/sagemaker` | secondary EBS | yes | **no** |
| `/opt/dlami/nvme` | NVMe instance store | **no** (cleared on reboot) | no |
| `/fsx` (Slurm) | FSx for Lustre | yes | yes (external to node) |
| EFS / FSx mounts via PVC (EKS) | external | yes | yes (external to node) |

Root volume "is not intended to grow post-creation" — don't put
expanding state there. `/dev/shm` exhaustion is a distinct failure
mode from disk-full.

## Lifecycle scripts

- **`on_create.sh`** is the only documented entry point. There is no
  `on_replace.sh`, `on_update.sh`, or `on_reboot.sh`. Replacement just
  re-runs `on_create.sh` on the fresh instance.
- **Re-execution triggers:**
  - Initial cluster creation: yes
  - Node replacement (`Action:Replace` or `BatchReplaceClusterNodes`):
    **yes** (on the new instance)
  - Node reboot (`Action:Reboot` or `BatchRebootClusterNodes`): **no**
  - `UpdateClusterSoftware`: **yes** — the update wipes the root EBS
    volume and re-runs `on_create.sh` as part of node initialization
    on the new AMI. See "`UpdateClusterSoftware` AMI repaint wipes the
    root EBS" callout above.
  - Cluster-agent reconfigure events (Slurm `slurm.conf` /
    `topology.*` rewrites that don't replace the instance): **no**
- **Inputs the script can read on the node:**
  - `/opt/ml/config/resource_config.json` — cluster instance inventory
    and (on accelerator SKUs) topology labels. Note: the on-node copy
    can lag the cluster's current state — workers may show instance
    groups that have since scaled to 0.
  - `/opt/ml/metadata/resource-metadata.json` — referenced by some
    public guidance as "node-level metadata," but `/opt/ml/metadata/`
    is **not observed on disk** under steady-state on the AMI versions
    inspected (both Slurm and EKS workers). The file may only exist
    during `on_create.sh` execution and be cleaned up after, or may
    be specific to certain AMI versions. If your LCS depends on it,
    verify presence at runtime — don't assume.
  - `provisioning_parameters.json` — keyed by `InstanceGroupName`; a
    mismatch with the cluster's actual group name causes `KeyError`
- **Slurm provisioning ordering:** controller nodes provision first;
  compute/login nodes need the controller to be up because their
  `slurmd` uses Slurm configless mode (fetches `slurm.conf` from the
  controller at boot — see "Slurm node services" above for the full
  mechanics). **Controller failure cascades:** if `slurmctld` is down
  or the controller's customer-VPC IP is unreachable, compute and
  login nodes can't bootstrap their Slurm config.
- **Idempotency:** not required. `on_create.sh` runs exactly once per
  root-volume lifetime — at initial provisioning, after replacement,
  and after `UpdateClusterSoftware` (each starts from a fresh root
  EBS). The script never re-executes on top of its own prior state,
  so "already configured" guards are unnecessary.
- **Common failure modes** observed in field tests:
  - Missing S3 VPC endpoint when controller needs to pull artifacts
  - IAM gap on `s3:GetObject`/`s3:ListBucket`
  - CRLF line endings (must be LF; valid shebang `#!/bin/bash`)
  - Infinite loops / unbounded retries (no explicit lifecycle timeout
    is documented, but hangs cause failure)
  - Instance-group name mismatch between
    `provisioning_parameters.json` and the API call
  - S3 key/filename mismatch with the `OnCreate` parameter

> **Recipe — Custom AMI hygiene before `create-image`.** Not strictly a
> mental-model concept, but the reason it matters is: the
> [HyperPod custom AMI docs](https://docs.aws.amazon.com/sagemaker/latest/dg/hyperpod-custom-ami-how-to.html)
> say nothing about pre-snapshot cleanup, but builder-instance
> cloud-init state baked into the AMI can suppress `on_create.sh`-related
> modules on first boot — and SSH host keys + `machine-id` will be
> shared across every node. Before `stop-instances`/`create-image`, run
> (in the same SSM customization invocation, not a separate one):
> `sudo cloud-init clean --logs --seed && sudo rm -f /etc/ssh/ssh_host_*
> && sudo truncate -s 0 /etc/machine-id && sudo rm -rf
> /var/lib/cloud/instances/* /var/lib/cloud/data/*`.

## CloudTrail attribution

The principal you see in CloudTrail tells you who initiated an action:

- **Customer principal** (e.g. `arn:aws:sts::<acct>:assumed-role/...`):
  the customer's user, role, or assumed-role session called the API
  directly. Examples: an operator running `aws sagemaker batch-replace-cluster-nodes`,
  a developer running `aws sagemaker update-cluster`, an SDK call from
  customer code.
- **HyperPod service principal** (`sagemaker.amazonaws.com` or the
  `AWSServiceRoleForSageMakerHyperPod` SLR): the HyperPod control
  plane took an action internally on the customer's behalf. Examples:
  the `ec2:RebootInstances` call that underlies a `BatchRebootClusterNodes`
  invocation, EC2 instance creation/termination during a replace.
- **No event at time T** in the customer trail: HyperPod likely took an
  internal action that doesn't generate a customer-visible CloudTrail
  event. The `list-cluster-events` API (EKS) is the canonical customer
  record of these.

**Resource types tied to HyperPod that appear in CloudTrail:**
`AWS::SageMaker::Cluster`, `AWS::IAM::ServiceLinkedRole` (the
`AWSServiceRoleForSageMakerHyperPod`), `AWS::EC2::VPC`,
`AWS::CloudFormation::Stack`, `AWS::Lambda::Function`, `Custom::Resource`.

## Event surfaces — EventBridge, `list-cluster-events`, `describe-cluster-event`

Three overlapping surfaces expose HyperPod cluster events. They agree
on the identity of an event but NOT on the content — different consumers
see different fields populated. Choosing the right surface for a given
task matters.

### `FailureMessage` is only in `describe-cluster-event` today

*(Both — verified on EKS 2026-06-30/2026-07-01)*

> ⚠️ **Bug — reported to the HyperPod engineering team; subject to
> change.** The findings below describe the state as of early July 2026.
> When this is fixed, the EventBridge envelope should carry the same
> `EventMetadata` content that `describe-cluster-event` returns.

For non-Info events, HyperPod populates a `FailureMessage` field under
`EventDetails.EventMetadata.<subtree>` — where `<subtree>` is one of
`Cluster`, `Instance`, `InstanceGroup`, `InstanceGroupScaling`,
`InstanceMonitor`, or `InstanceHealth`. **The EventBridge envelope
delivers this subtree as `null` even when the API returns a populated
value.** The only way to see the actual failure message today is to
call `describe-cluster-event --event-id <eid> --cluster-name <name>`
after receiving the EventBridge event.

Concrete example — a capacity error on `worker4`:

```
$ aws sagemaker describe-cluster-event --cluster-name k8-1 \
    --event-id f17f580b-2d73-4615-8957-c09ef6991182
{
  "EventDetails": {
    "EventDetails": {
      "EventMetadata": {
        "Instance": {
          "FailureMessage": "We currently do not have sufficient
            capacity to launch new ml.g5.8xlarge instances.
            Please try again.",
          "NodeLogicalId": "50be05d5-c749-4023-9fde-4ce95ff0e6cb"
        }
      }
    },
    "Description": "Failed to provision EC2 Instance in Cluster k8-1
      and InstanceGroup worker4",
    ...
  }
}
```

The same event on EventBridge shows `EventMetadata.Instance: null` and
just the generic `Description` field. Consumers that only read the
EventBridge envelope get the "what happened" (a generic string) but
NOT the "why" (the specific capacity failure).

`describe-cluster-event` requires **`NodeProvisioningMode: Continuous`**
on the cluster. Non-Continuous clusters return
`ValidationException` — the FailureMessage is genuinely unreachable
on those clusters, so consumers must build with description-only as
a fallback path.

### EventBridge envelope has null-placeholder subtrees for the full `EventMetadata` shape

*(Both — verified on EKS)*

Even when `EventMetadata.<subtree>` is `null`, the envelope always
includes the FULL set of six subtree keys as placeholders:

```
"EventMetadata": {
  "Cluster": null,
  "Instance": null,
  "InstanceGroup": null,
  "InstanceGroupScaling": null,
  "InstanceMonitor": null,
  "InstanceHealth": null
}
```

Different fault types populate DIFFERENT subtrees when the bug above
gets fixed:

| Fault class | Populated subtree(s) observed |
|---|---|
| Per-instance provisioning failure (capacity, EFA health-check, etc.) | `EventMetadata.Instance.FailureMessage`, sometimes `.NodeLogicalId` |
| Cluster-level scaling operations, "lost orchestration-ready", etc. | `EventMetadata.Cluster.FailureMessage` |
| (Others TBD as we observe them) | `InstanceGroup`, `InstanceGroupScaling`, `InstanceMonitor`, `InstanceHealth` — schema exists, contents not yet characterized |

Consumers that want to extract fault content MUST walk all six
subtrees, not just `Instance` — the useful FailureMessage may be in
any of them. A common bug shape is code like `EventMetadata.get("Instance", {}).get("FailureMessage")`
that silently misses cluster-level failures.

### HyperPod emits paired events for the same underlying fault

*(Both — verified on EKS)*

A single provisioning fault produces TWO SageMaker EventIds within
~1 second, both `EventLevel=Error`, describing the same underlying
event from different angles:

```
EventId d906a77c-...  Description="Failed to provision EC2 Instance in Cluster k8-1 and InstanceGroup worker4"
EventId 2ed1305e-...  Description="Instance creation in Cluster k8-1 and InstanceGroup worker4 failed"
```

Both fire on EventBridge, so a naive one-event-per-investigation
pipeline creates TWO investigations for what an operator would call
one incident. Dedup logic that keys on `(IG, description)` or
`(IG, description + failure_message)` will treat them as distinct
signatures (different descriptions).

Options for dedup consumers:
- Time-window collapse: within N seconds of the same fault type on the
  same IG, treat as one.
- Prefer the "Failed to provision" event (it's typically emitted first
  and can be enriched with FailureMessage via `describe-cluster-event`);
  ignore the paired "Instance creation ... failed" as a duplicate.
- Widen the signature to strip descriptions and use only
  `<IG>:<FailureMessage>` — but that fails when FailureMessage is
  `null` on the EventBridge envelope (see previous callout).

### SageMaker `EventId` vs EventBridge `id` are different values

*(Both)*

The EventBridge envelope's top-level `id` field is EventBridge's own
UUID for the delivered event. The SageMaker `EventId` — the one that
works with `describe-cluster-event` and appears in `list-cluster-events`
— is nested at `detail.EventDetails.EventId`. Both are UUIDs; they
look identical in shape. Using the EventBridge `id` in
`describe-cluster-event` returns `ResourceNotFoundException`.

Downstream tooling should extract the SageMaker EventId from the
nested field, not the top-level envelope field.

### `ClusterStatus == InService` does NOT mean "nothing is scaling" (Continuous Provisioning mode)

*(Both — verified on EKS with `NodeProvisioningMode: Continuous`)*

In **Continuous Provisioning mode**, `describe-cluster` returns
`ClusterStatus: InService` throughout customer-initiated
`UpdateCluster` scaling operations — the status does NOT flip to
`Updating` even while instances are being added or removed. This is
the documented behavior of Continuous mode, not a bug.

To detect a scaling operation in progress on a Continuous-mode
cluster, iterate `InstanceGroups[]` and compare `CurrentCount` vs
`TargetCount`:

- `CurrentCount < TargetCount` → **scaling up** (waiting for new
  instances to provision)
- `CurrentCount > TargetCount` → **scaling down** (waiting for
  instances to terminate)
- `CurrentCount == TargetCount` on every IG → **steady state**

**Do not rely on ClusterStatus alone** for the "am I scaling?"
signal. Consumers building on Continuous-mode clusters (which
includes any cluster using `describe-cluster-event`, since that API
requires Continuous mode — see "FailureMessage" callout above) must
key their scaling detection on `CurrentCount` vs `TargetCount`.

Note: this is distinct from — but related to — the pre-existing
`describe-cluster` / `list-cluster-nodes` divergence callout below
(under "Common AI-confusing details"). That entry covers node-list
vs count-of-nodes disagreement during any transition. This entry
is specifically about ClusterStatus behavior under Continuous mode.

## Capacity options and capacity failures

*(Both)*

HyperPod's capacity model is a frequent source of customer — and
therefore AI — confusion, because it does **not** map onto the EC2
capacity concepts an AI's training data expects (ODCR, Capacity Blocks,
Spot). Getting this wrong produces two failure modes: recommending an
EC2 capacity mechanism HyperPod doesn't use, and misclassifying a
capacity `Failed` as the same transient-retry case as an EFA
health-check failure.

### The three ways to get capacity

| Option | When | How you consume it |
|---|---|---|
| **On-Demand** | Small SKUs, experiments | Just create the cluster. Not guaranteed for large GPU SKUs (p4d/p5/…), placement is not topology-optimized, **not recommended for production**. |
| **Flexible Training Plans** | Medium-to-large, predictable workloads | Query availability by type/count/schedule, self-purchase (discounted, up to 180 days), then pass the plan's **`TrainingPlanArn`** on the instance group. Guaranteed capacity + better topology. |
| **Reserved Capacity via AWS account team** | Large-scale, long-term | The account team allocates capacity to your account in a specific AZ. **No ID to pass** — see below. |

### HyperPod does NOT use EC2 ODCR or Capacity Blocks

This is the single most consequential capacity misconception.

- HyperPod capacity is **not** an EC2 On-Demand Capacity Reservation
  (ODCR) and **not** an EC2 Capacity Block. There is no
  `CapacityReservationId` / Capacity-Block ID to specify on the cluster,
  and the EC2 capacity-reservation APIs do not apply.
- Capacity that your AWS account team reserves for you (account × AZ) is
  **not visible in your EC2 console** — `aws ec2
  describe-capacity-reservations` returns nothing and the EC2 "Capacity
  Reservations" page is empty. This mirrors the ownership boundary at
  the top of this doc: the reservation lives in the HyperPod service
  account, not yours. **Absence of an EC2 capacity reservation is NOT
  evidence that no capacity was allocated.**

### How to actually consume account-team-reserved capacity

> **You do NOT specify any capacity/reservation ID.** Create the cluster
> normally — the ONLY things that must be correct are the **account** and
> the **Availability Zone** (via the instance group's subnet). As long as
> the cluster's account × AZ matches where the account team allocated the
> capacity, HyperPod picks it up **automatically**.

This is the exact point customers (and AIs) trip on: they hunt for a
field to plug a reservation ID into, don't find one, and conclude the
setup is incomplete. There is no such field for account-team
reservations — the binding is implicit through account × AZ.

Practical consequence for the instance group config:

- The instance group's **subnet must be in the AZ where capacity was
  reserved.** Wrong AZ → the reservation isn't matched → you fall back to
  on-demand and hit a capacity error even though capacity "exists."
- **Flexible Training Plans are different**: you DO pass `TrainingPlanArn`
  on the instance group, AND the AZ/subnet must match where the plan's
  capacity lives.

### Reading a capacity `Failed` correctly

A capacity shortfall surfaces as a per-instance provisioning failure. The
`FailureMessage` (via `describe-cluster-event` — see "`FailureMessage` is
only in `describe-cluster-event` today") reads like:

> *"We currently do not have sufficient capacity to launch new
> ml.g5.8xlarge instances. Please try again."*

or

> *"We currently do not have sufficient capacity in the Availability Zone
> you requested."*

**Do not treat this like an EFA-health-check `Failed`.** The resiliency
section notes EFA health-check failures are often transient and a manual
retry usually succeeds. A capacity `Failed` is a different animal:

- **On-Demand**: retrying into an exhausted pool for a large GPU SKU will
  keep failing. The fix is a capacity *option* change (Flexible Training
  Plan or account-team reservation), a different AZ, or smaller/fewer
  instances — **not** a blind retry.
- **Training Plan or account-team reservation** and still hitting a
  capacity error → suspect a **config mismatch**, not pool exhaustion:
  - Training Plan: `TrainingPlanArn` missing or wrong.
  - Account-team reservation: the cluster's AZ (instance-group subnet)
    doesn't match the reserved AZ.
  - Verify the subnet's AZ, and confirm the reservation's account × AZ
    with your account team.

**Diagnostic rule:** a capacity-worded `Failed` on a cluster that is
*supposed* to have reserved capacity is far more likely an **AZ/subnet
misalignment** (a customer-config class) than genuine pool exhaustion.

## Common AI-confusing details

Each entry below is tagged with whether it applies to Slurm, EKS, or
both. Add new entries here as they're discovered.

### Slurm "topology/flat" vs "topology/default" are the same plugin

*(Slurm only)*

`scontrol show config` reports `TopologyPlugin = topology/flat` even when
`slurm.conf` says `TopologyPlugin=topology/default`. They are the same
plugin under different display names. Don't waste time hunting for a
"flat" config line that doesn't exist.

### `describe-cluster-node` does NOT expose topology

*(Both)*

The SageMaker API `describe-cluster-node` returns instance metadata
(IP, type, AZ, etc.) but **not** the `network-node-layer-N` labels. To
see the labels on Slurm, read `/opt/ml/config/resource_config.json` on
the controller via SSM. On EKS, use `kubectl get node <name>
--show-labels` (HyperPod surfaces `topology.k8s.aws/network-node-layer-N`
as Kubernetes node labels). The Slurm gap is a known asymmetry with
the EKS surface.

### `aws ec2 describe-instance-topology` doesn't work on HyperPod nodes

*(Both)*

For the same reason — the instances are not in the customer's EC2
account, so customer-side EC2 APIs cannot describe them. The topology
data exists in `resource_config.json` (Slurm) or as node labels (EKS),
not in the customer-account EC2 API.

### Orchestrator view vs HyperPod API view diverge during replacement

*(Both — Slurm example shown; same pattern on EKS)*

| State | Slurm `sinfo` | EKS `kubectl get node` | HyperPod API |
|---|---|---|---|
| Pre-failure | `idle` or `allocated` | `Ready` | `Running` |
| Post-failure (mid-replacement) | `fail*` or `down*` | `NotReady` | `Pending` |
| Post-recovery | `idle` | `Ready` | `Running` |

The two views can be out of sync for several minutes during a
replacement. Slurm's `idle` (or EKS's `Ready`) does NOT mean
"operationally healthy" if the HyperPod API says `Pending`.

### `describe-cluster` and `list-cluster-nodes` are not consistent during transitions

*(Both)*

The two APIs have different update cadences and can disagree about
how many nodes exist and what state they're in during scale-down,
IG-delete, and node-replacement transitions. Typically
`list-cluster-nodes` filters terminating / deleting / failed
instances before `describe-cluster.InstanceGroups[].CurrentCount`
drops to match. Reconciliation usually completes within seconds for
small instance types, minutes for large GPU types.

**Don't treat agreement between the two surfaces as the steady-state
condition.** Capture both at the same UTC second and reconstruct
timelines from per-cycle pairs, not from a single snapshot. A node
disappearing from `list-cluster-nodes` while still counted in
`describe-cluster.CurrentCount` is the normal mid-transition shape.

### Scale-in-progress emits spurious `Warn` events with misleading FailureMessage

*(Both — verified on EKS)*

During customer-initiated `UpdateCluster` scaling operations
(scale-up OR scale-down), HyperPod emits a stream of `Warn`-level
`SageMaker HyperPod Cluster Event` entries with descriptions like:

> *"N node(s) lost orchestration-ready status. Current: X/Y
> orchestration-ready across M instance group(s)."*

Each event carries a different `N` as the scaling progresses (e.g.
`1 node(s) lost...`, then `3 node(s) lost...`, then `4 node(s)
lost...`). These are **progress updates during a customer-initiated
operation, not incident signals**.

`DescribeClusterEvent` on any of these events returns a
`FailureMessage` under `EventMetadata.Cluster`:

> *"Request to service failed. If failure persists after retry,
> contact customer support."*

**That FailureMessage is misleading.** No customer-visible service
failed — the message appears for scale-in-progress events regardless
of whether anything is actually broken. Consumers of the event stream
must NOT treat this as a hard-fault signal on its own.

**How to distinguish "spurious scale-in-progress" from a real fault:**

- **`describe-cluster`** — if any `InstanceGroups[].CurrentCount !=
  TargetCount`, the cluster is mid-scaling. "lost orchestration-ready"
  events during this window are progress noise.
- **CloudTrail** (secondary signal) — a customer-principal
  `UpdateCluster` API call in the last ~10 minutes is strong evidence
  the noise is customer-driven.
- **What's NOT sufficient**: cluster `ClusterStatus == InService`
  during scaling is common (HyperPod does not always flip to
  `Updating` for small scaling operations). Do not use `ClusterStatus`
  alone as the "am I scaling?" signal.

The `hyperpod-incident-triage` skill (rule 3 as of v0.3.0) uses this
detection to `SKIP` "lost orchestration-ready" events during scaling
operations. Non-scaling `Warn`/`Error` events with different
descriptions (e.g. `"Failed to provision EC2 Instance"`,
`"EFA health checks did not run successfully"`) still PROCEED as
normal incidents — those are real faults that can happen during a
scale-up.

### `Failed` instance status is not necessarily terminal

*(Both)*

When HyperPod replacement itself fails — most often due to a transient
EFA health check failure during boot — the node lands in `Failed`
status. **HyperPod may auto-retry from `Failed`**, particularly when
Continuous Provisioning is enabled on the cluster. Earlier guidance
that `Failed` is terminal is outdated; treat `Failed` as a transient
state inside a retry chain until enough time has passed without
progress to conclude HyperPod has given up.

Additionally, **a node can disappear from `list-cluster-nodes`
between retry attempts** — the customer-visible API surface is not
always populated during the gap between one failed attempt and the
next. Don't infer "gave up" from a single missing snapshot.

To distinguish "still retrying" from "stuck", consult
`list-cluster-events` (the canonical customer record of replacement
attempts, including failed ones) and apply a time budget — at least
60–90 minutes of no new attempt and no successful `Running`
transition before concluding retry is exhausted. A single replacement
attempt typically takes 20–30 minutes (see "How long things take"
below), so a window covering 2–3 attempts is the floor.

The error message in `describe-cluster-node` is:
> *"Instance replacement failed: EFA health checks did not run
> successfully. Ensure that your VPC and security groups are properly
> configured before attempting to create a new cluster."*

That message implies a customer VPC misconfiguration but is often
transient. Manual replacement frequently succeeds with no VPC changes.

**A capacity-worded `Failed` is NOT this case.** When the
`FailureMessage` is about insufficient capacity (`"We currently do not
have sufficient capacity..."`) rather than EFA health checks, the
transient-retry reasoning above does NOT apply — retrying an
on-demand large-GPU request into an exhausted pool keeps failing, and a
capacity error on a cluster that should have reserved capacity is usually
an AZ/subnet misalignment. See "Capacity options and capacity failures"
above.

### Topology placement on replacement is *usually* preserved, but not guaranteed

*(Both)*

When HyperPod replaces an instance, the new EC2 *typically* lands on
the same network switch as the old one, preserving the cluster's
topology layout (`topology.yaml` on Slurm, node labels on EKS). This
appears to be the common case in field testing. **It is not a
contractual guarantee** — leaf shuffling has been observed under
multi-node-replacement pressure (multiple replacements in flight at
once). Customer code should not depend on replacement-preserves-leaf.

### Recurring fault signature does NOT prove physical-host affinity

*(Both)*

A common AI failure mode when looking at recurring hardware-fault
signatures (e.g. the same Xid code on the same `PCI:0000:b9:00` /
`NVLink link 6` across consecutive replacements): defaulting to "every
replacement keeps landing on the same faulty physical hardware." That
explanation is almost always wrong on HyperPod, for three reasons:

1. **Instance placement is service-account, not customer-account.**
   The EC2 instance is owned by the HyperPod service account (see
   "Ownership boundaries" above). The customer surface — `NodeId`,
   `InstanceId`, ENI, K8s node name — does not expose the underlying
   physical host, so the customer literally cannot observe "same
   physical host." Any claim that two replacements landed on the same
   physical host is **[unverified]** — there is no customer-side API
   that surfaces that.
2. **EC2 placement is non-deterministic per replacement.** AWS EC2's
   placement model assigns a fresh instance from the capacity pool on
   each new launch. While *topology leaf* may be preserved (see the
   "Topology placement on replacement" entry above), the underlying
   *physical host* is not — and topology preservation isn't even a
   contract. Two replacements that hit the same fault are far more
   likely to be hitting the same fault *cause* than the same physical
   host.
3. **The Xid code names a fault class, not a physical fingerprint.**
   `Xid 74 / PCI:0000:b9:00 / NVLink link 6` is the same string on
   *every* GPU of the same SKU when that GPU has an NVLink link 6
   issue — the PCI BDF is the slot, not a unique GPU identifier. Two
   physically distinct GPUs of the same SKU with NVLink defects on
   link 6 will produce identical strings.

When a recurring signature appears, prefer these competing hypotheses
over physical-host affinity (and present them as competing, not
committed):

- **Software / workload pattern**: a particular NCCL collective,
  driver / CUDA version, or workload code path triggers the fault on
  whatever GPU it lands on. Test by changing the workload (or moving
  the instance group to a different node and seeing if the fault
  follows).
- **Infrastructure path**: an EFA fabric path, leaf switch, or shared
  network resource is flaky and surfaces as GPU-level errors on
  workloads that hit it. Test by moving the instance group to a
  different subnet / AZ.
- **Statistical hardware**: a bad batch of GPUs is over-represented
  in the capacity pool for this SKU + AZ. Test by waiting + retrying
  later, or by asking AWS Support to exclude specific hardware.

The verdict in a recurring-pattern incident should list these
hypotheses, mark each `[unverified]`, and recommend the operator
discriminate by an action (e.g. "open an AWS Support case requesting
hardware exclusion" or "move the IG to a different subnet to test
infrastructure-path hypothesis"). It should **not** commit to a
single explanation absent evidence the customer surface can produce.

#### The only reliable physical-identity check (operator-only)

The single deterministic way to know whether two replacement
instances landed on the **same physical GPU** is to compare GPU
UUIDs across them via `nvidia-smi`:

```
nvidia-smi -L
# GPU 0: NVIDIA L4 (UUID: GPU-35af0a5d-06b3-f6cc-1fab-c63687d448ff)
```

The UUID is a per-GPU identifier baked into the GPU's vBIOS — it
survives reboot and replacement of *everything around* the GPU, but
it's unique to that physical silicon. If the same UUID appears on
two different EC2 instance IDs in your cluster, you genuinely are
hitting the same GPU. If they differ, the replacements landed on
different physical hardware and "same physical hardware" is ruled
out.

**This check requires SSM** (`aws ssm start-session ...` plus
`nvidia-smi -L`). The DevOps Agent permission guardrail does not
allow `ssm:StartSession`, so DevOps Agent **cannot perform this
check itself** — it must recommend that the operator run it and
either paste the output back or open a Support case with the
collected UUIDs.

The recommended operator action is therefore:

```
# On each instance that hit the fault, capture the GPU UUID:
aws ssm start-session \
  --target sagemaker-cluster:<cluster-id>_<group>-<instance-id> \
  --document-name AWS-StartNonInteractiveCommand \
  --parameters '{"command":["nvidia-smi -L"]}'

# Repeat for each affected InstanceId in the cluster events.
# Compare the UUID strings. If they match, it's the same physical GPU.
```

This UUID-comparison evidence is the ONLY way to elevate
"physical-host affinity" from `[unverified]` to `[direct]`. Without
it, hypotheses must remain competing.

### Slurm auto-resume requeues, not shrinks-in-place, on GRES nodes

*(Slurm only)*

GRES-attached nodes (e.g. `Gres=gpu:h100:8`) — which includes every
HyperPod GPU node — cause Slurm to disallow in-place node
replacement, so the HyperPod auto-resume plugin falls back to
requeueing the job after a node failure. The fallback is the
documented path on HyperPod, not a bug; the alternative
"shrink-in-place" code path applies only on non-GRES clusters which
are rare in practice for HyperPod.

End-to-end recovery still works via the requeue path: the job is
re-queued, waits for the replacement to come up, and gets re-allocated
on the recovered nodes. Customers should expect logs to indicate
"requeue" rather than "in-place shrink."

### sbatch `--output=` truncates on requeue by default

*(Slurm only)*

Slurm's default `--open-mode` is `truncate`, which means a requeued
job (common on HyperPod after node replacement) wipes its prior
output log. Customer guidance: pass `--open-mode=append` on the
`sbatch` invocation, or have the inner script write per-attempt
timestamped files. `%j` (job ID) does NOT help since the job ID stays
the same across requeues.

### `--auto-resume=1` is an `srun` flag, not an `sbatch` flag

*(Slurm only)*

Common mistake: writing `sbatch --auto-resume=1 ...`. That doesn't work
— `sbatch: unrecognized option '--auto-resume=1'`. The correct usage is:
```
sbatch ...args... --wrap='srun --auto-resume=1 ./inner.sh'
```
The `--auto-resume=1` belongs on the `srun` command inside the batch
allocation.

### Slurm `Reason="Action:..."` matching is exact and case-sensitive

*(Slurm only)*

HyperPod auto-recovery matches the Slurm node `Reason` field
**exactly, case-sensitive**. Any mismatch is silently ignored — the
node sits drained with the operator's reason but no action is taken.

| Variant | Effect |
|---|---|
| `Action:Reboot` | accepted |
| `Action:Replace` | accepted |
| `action:replace` | ignored (wrong case) |
| `Action: Reboot` | ignored (extra space) |
| `Action:Reboot⎵` | ignored (trailing whitespace) |
| `Action:Reboot.` | ignored (trailing punctuation) |
| `Reboot` / `replace this` | ignored (wrong format) |

Canonical operator commands:
```
scontrol update node=<ip> state=fail reason="Action:Reboot"
scontrol update node=<ip> state=fail reason="Action:Replace"
```

### EKS manual-trigger labels are the parallel mechanism

*(EKS only)*

Two labels emulate the Slurm `Action:...` reasons:
- `sagemaker.amazonaws.com/node-health-status=UnschedulablePendingReboot`
- `sagemaker.amazonaws.com/node-health-status=UnschedulablePendingReplacement`

Setting one of these triggers the **same** HyperPod recovery process
that HMA invokes automatically.

### `aws ssm start-session` is rate-limited to 3 TPS per account

*(Both)*

Hitting this throttle returns `ThrottlingException`. Affects any
parallel SSM-based fan-out (issue collection, multi-node diagnostics).
Use exponential backoff or serialize.

Also: `aws ssm start-session` intermittently returns empty stdout
with `"Cannot perform start session: EOF"` even when the command ran
successfully. Mitigation: wrap with `unbuffer` (from the `expect`
package) on the client side, or use the
[`hyperpod_run_on_multi_nodes.py`](../hyperpod_run_on_multi_nodes/)
helper in this repo which handles the unbuffer wrapping and retry
semantics, plus parallel fan-out across multiple nodes with the
3-TPS-per-account SSM throttle factored in.

### `ssm:SendCommand` does NOT work against `sagemaker-cluster:` targets

*(Both)*

The SSM document for HyperPod targets only supports `start-session`,
not `send-command`. If you need to script remote commands, you have to
go through `start-session` with stdin (which is what
`hyperpod_run_on_multi_nodes.py` does). This is why HyperPod tooling
favors interactive-session-over-SSM as the only remote-execution path.

## Common AWS APIs and their HyperPod equivalents

| Standard EC2/AWS API | HyperPod equivalent |
|---|---|
| `aws ec2 describe-instances` | `aws sagemaker list-cluster-nodes --cluster-name <name>` |
| `aws ec2 terminate-instances` | `aws sagemaker batch-delete-cluster-nodes` (shrinks the group) |
| Reboot/replace a node | `aws sagemaker batch-replace-cluster-nodes` |
| `aws ec2 describe-instance-topology` | Slurm: read `/opt/ml/config/resource_config.json` on the controller. EKS: `kubectl get node <name> --show-labels` (look for `topology.k8s.aws/network-node-layer-N`). |
| `aws ec2 reboot-instances` | Inject GPU fault → HMA marks `Action:Reboot`. To drain a node without touching EC2: Slurm `scontrol update NodeName=<n> State=DOWN Reason="..."`, or EKS `kubectl cordon <n>` + `kubectl drain <n>`. |
| SSH into a node | `aws ssm start-session --target sagemaker-cluster:{cluster-id}_{group-name}-{instance-id}` |
| Patch the OS / agents | `aws sagemaker update-cluster-software` |

## Reaching nodes via SSM

The HyperPod SSM target format:

```
sagemaker-cluster:{cluster-id}_{instance-group-name}-{instance-id}
```

Where `cluster-id` is the last segment of the cluster ARN
(`arn:aws:sagemaker:<region>:<account>:cluster/<cluster-id>`).
Example:

```
aws ssm start-session --target sagemaker-cluster:<cluster-id>_controller-machine-i-0123456789abcdef0
```

Once in, you have shell access on the node and can run anything: read
config files, inspect orchestrator state (Slurm: `sinfo`/`squeue`/
`scontrol`; EKS: `systemctl status kubelet`, `journalctl -u hyperpod-host-agent`,
or `kubectl` from a separately-configured client — `kubectl` is not
typically configured on the node itself), restart services, fetch logs, etc.

**Identity gotcha on EKS**: the OS-level `hostname` on an EKS worker is
`ip-<agent-ip>.<region>.compute.internal` (uses the agent-network IP,
e.g. `172.16.x.x`), NOT the customer-VPC IP that shows up in
`kubectl get nodes -o wide` and in `resource_config.json`'s
`CustomerIpAddress`. The k8s node name is the kubelet override
`hyperpod-i-<instance-id>`. So a single EKS worker has at least three
identity strings (OS hostname, k8s hostname, EC2 instance ID) plus
two IPs (customer-VPC, agent-network), and they do NOT all change in
lockstep on reboot vs. replace — see "Identity stability" above.

## Where to find data

### Cluster identity and instance inventory
- `aws sagemaker describe-cluster --cluster-name <name>` — cluster ARN,
  status, instance group config.
- `aws sagemaker list-cluster-nodes --cluster-name <name>` — list of
  instance IDs with status.
- `aws sagemaker describe-cluster-node --cluster-name <name> --node-id <id>`
  — per-instance detail.

### Slurm state (via SSM to controller)
- `sinfo`, `sinfo -N -l`, `sinfo -R` — partition / node state, drain reasons.
- `squeue`, `squeue -a` — job queue.
- `scontrol show config` — live Slurm config (after any reconfigure).
- `scontrol show partition`, `scontrol show job <id>`, `scontrol show node <name>`.
- `scontrol show topology` — only meaningful when topology plugin is tree/block.

### EKS state (via kubectl)
- `kubectl get nodes --show-labels` — includes
  `topology.k8s.aws/network-node-layer-{1,2,3}` and
  `topology.k8s.aws/ultraserver-id` labels per node.
- `kubectl get pods -A` — workloads.
- `kubectl describe node <name>` — full node detail.

### Logs (via SSM and CloudWatch)

**Canonical CloudWatch log group:**
```
/aws/sagemaker/Clusters/<CLUSTER_NAME>/<CLUSTER_ID>
```

The `<CLUSTER_ID>` is the **ARN suffix**, not the cluster name. Derive
via:
```
aws sagemaker describe-cluster --cluster-name <name> \
    --query 'ClusterArn' --output text | cut -d/ -f2
```

**Streams within that log group:**

| Stream prefix | Source |
|---|---|
| `LifecycleConfig/<group>/<instance-id>` | `on_create.sh` and orchestrator output |
| `SagemakerHealthMonitoringAgent/<group>/<instance-id>` | HMA detection events (`HealthMonitoringAgentDetectionEvent`) |
| `DeepHealthCheckResults/<log_stream_id>` | `StartClusterHealthCheck` results |

**On-node paths (via SSM):**

| Log / file | Path | Slurm controller | Slurm compute/login | EKS |
|---|---|---|---|---|
| ClusterAgent (structured JSON) | `/var/log/aws/clusters/sagemaker-cluster-agent.log` | yes | (file may exist but agent doesn't run as daemon) | **no** |
| ClusterAgent (systemd journal) | `sudo journalctl -u sagemaker-cluster-agent` | yes | (unit exited immediately, brief journal only) | **no** |
| HMA (systemd journal) | `sudo journalctl -u sagemaker-health-monitoring-agent` | yes (also CloudWatch) | yes (also CloudWatch) | **no** — HMA is a DaemonSet; use `kubectl logs -n aws-hyperpod ds/health-monitoring-agent` |
| HostAgent (systemd journal) | `sudo journalctl -u sagemaker-host-agent` (Slurm) / `sudo journalctl -u hyperpod-host-agent` (EKS) | yes | yes | yes (different binary) |
| RoleProxyAgent (systemd journal) | `sudo journalctl -u sagemaker-role-proxy-agent` (Slurm) / `sudo journalctl -u hyperpod-role-proxy-agent` (EKS) | yes | yes | yes |
| Custom health monitor (Slurm compute only) | `sudo journalctl -u custom-health-monitor` | no | yes | n/a |
| Lifecycle script output | `/var/log/provision/provisioning.log` | yes | yes | yes |
| Cluster log dir (collected by issue-report) | `/var/log/aws/clusters/` | yes | yes | yes (subset) |
| Deep health check | `/var/log/aws/clusters/sagemaker-deep-health-check.log` | yes | yes | yes |
| DCGM | `/var/log/nvidia-dcgm/` (`nvvs*.log`) | (GPU nodes) | (GPU nodes) | (GPU nodes) |
| Cloud-init / system | `/var/log/syslog`, `/var/log/cloud-init.log`, `dmesg` | yes | yes | yes |
| EFA installer manifest | `/opt/amazon/efa_installed_packages` | yes | yes | yes |

**Slurm node-role-specific log files:**

| Log | Path | Controller | Compute | Login |
|---|---|---|---|---|
| Slurm controller daemon | `/var/log/slurm/slurmctld.log` | has content | (file present but empty) | (empty) |
| Slurm compute daemon | `/var/log/slurm/slurmd.log` | (empty) | has content | **has content** — login runs `slurmd` too (configless mode) |
| Slurm DBD | `/var/log/slurm/slurmdbd.log` | has content | no | no |
| Slurmd configless cache | `/var/spool/slurmd/conf-cache/{slurm.conf,gres.conf,cgroup.conf,plugstack.conf}` | absent | yes — pulled from controller via `slurmd --conf-server` | yes — same as compute |

**EKS-specific (on-node):**

| Log | Path |
|---|---|
| kubelet | `sudo journalctl -u kubelet` |
| Container logs | `kubectl logs <pod>` from a client, or `/var/log/pods/...` on the node |
| HyperPod EKS operator | `kubectl logs -n <operator-namespace> <operator-pod>` |

**Which signals live where:**

- **CloudWatch only**: HMA detection events, deep-health-check results.
- **CloudWatch + on-node**: lifecycle script output.
- **On-node only**: kubelet, containerd, dmesg, system journal,
  `slurmctld.log`/`slurmd.log`, DCGM findings.
- **Service API only**: `list-cluster-events` (EKS, and Slurm with
  Continuous Provisioning) and `list-cluster-nodes` state transitions
  — **not** in CloudWatch.

### CloudWatch metrics

| Namespace | What's there |
|---|---|
| `ClusterAgent` | Cluster-level instance counts (`ALLOC`, `IDLE`, `TotalInstances`), API latencies, reconfigure events |
| `SagemakerHealthMonitoringAgent` | Per-instance hardware health events |
| `AWS/Usage` | SageMaker service quotas |

## Hardware fault signals (general guide)

These are the cross-scenario signals to recognize; per-scenario triage
recipes belong in the specific troubleshooting skills, not here.

### NVIDIA Xid codes

Xid is NVIDIA's general error reporting mechanism for GPU faults.
For the authoritative catalog of Xid codes, their meanings, and
NVIDIA's own severity guidance, see NVIDIA's documentation:
<https://docs.nvidia.com/deploy/xid-errors/index.html>.

HyperPod chooses a remediation action (reboot or replace) based on
the error class it detects — different Xid codes, ECC events, and
other hardware signals map to different actions. The specific
mapping is an internal service detail and is subject to change;
don't hard-code assumptions about which Xid causes which action.

### ECC errors

- **Correctable (CE)**: background noise. A persistently growing rate
  on one GPU is worth escalating, but isolated events are normal.
- **Uncorrectable (UCE)**: failing memory. Drain and replace the node.
- DCGM fields: `ecc.errors.corrected.volatile.total`,
  `ecc.errors.uncorrected.volatile.total`
- **Row remap state** is the authoritative silent-degradation signal —
  GPUs degrading but not yet failing show up here before Xid/ECC fires.

### EFA fabric

- Security group **must** allow all inbound + outbound to itself
  (self-reference rules). Missing this is the most common cause of
  the `"EFA health checks did not run successfully"` error during
  instance bring-up.
- EFA installer manifest at `/opt/amazon/efa_installed_packages`.
- Baseline env: `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`,
  `NCCL_SOCKET_IFNAME=^lo,docker`. Fork-mem failure → set
  `FI_EFA_USE_HUGE_PAGE=0`.

### Thermal

GPU at or above 88°C is the surfaced thermal-concern
threshold. A single GPU persistently throttling while peers stay cool
→ drain and replace (suggests a per-GPU cooling fault, not a workload
issue).

### Neuron (Trainium / Inferentia)

Different toolchain:
- Counters live under Neuron sysfs, exposed via `neuron-ls`,
  `neuron-top`, `neuron-monitor`.
- `nvidia-smi` does not apply.
- Collectives use AWS Neuron Collectives, **not NCCL**.
- HMA's count-mismatch detection uses `neuron-ls` output instead of
  `nvidia-smi` for Trainium nodes.

### Common error strings to grep on

Verbatim strings worth recognizing across logs:

- `"EFA health checks did not run successfully. Ensure that your VPC and security groups are properly configured before attempting to create a new cluster."` — instance-replacement boot failure (often transient).
- `"Target is not connected"` — SSM session can't reach the node.
- `"Cannot perform start session: EOF"` — SSM PTY quirk; intermittent
  empty-stdout response. Wrap with `unbuffer` (from `expect`), or use
  the [`hyperpod_run_on_multi_nodes.py`](../hyperpod_run_on_multi_nodes/)
  helper which handles the unbuffer wrapping + retry semantics. Note
  that `sudo dcgmi test --inject ...` triggers this EOF response
  *every* time when invoked through `aws ssm start-session ... --document-name
  AWS-StartNonInteractiveCommand` even though the inject itself
  succeeds on-node — the dcgmi binary writes its output in a way that
  breaks the PTY framing. The multi-node helper handles this correctly.
- `"Embedded stack failed"` — real CloudFormation nested-stack error.
- `"We currently do not have sufficient capacity in the Availability Zone you requested"` — EC2 capacity issue, not a customer config issue.
- `"InvalidPermission.Duplicate"` — SG authorize idempotency, treat as success.
- `"Node unexpectedly rebooted"` — upstream Slurm message, **not** a
  HyperPod fault signal.

## How to trigger cluster events

Realistic ways to force state transitions on the cluster. Useful for
testing, debugging, and simulating customer failures. Treat all of
these as potentially destructive — they're operations that affect
running workloads.

| Event | How to trigger |
|---|---|
| Instance replacement / reboot (HyperPod **EKS**, AL2023) | Write a fake `NVRM: Xid` kernel-log line to **`/var/log/messages`**. Xid **74** → `Action:Replace` (label `UnschedulablePendingReplacement`); Xid **73** → `Action:Reboot` (label `UnschedulablePendingReboot`). Example: `sudo sh -c "echo \"$(date '+%b %d %H:%M:%S') $(hostname) kernel: NVRM: Xid (PCI:0000:b9:00): 74, pid=<unknown>, name=<unknown>, NVLink: fatal error detected on link 6(0x10000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0)\" >> /var/log/messages"`. **Verified end-to-end**: ~33s from log write to `UnschedulablePendingReplacement`. Canonical recipe: [HyperPod EKS resiliency guide](https://awslabs.github.io/ai-on-sagemaker-hyperpod/docs/eks-orchestration/validation-and-testing/resiliency/eks-resiliency#2emulate-instance-failure). **DCGM `dcgmi inject` does NOT trigger HMA on EKS** — the EKS HMA DaemonSet watches the kernel log only. |
| Instance replacement (HyperPod **Slurm**, Ubuntu 22.04) | `sudo dcgmi test --inject --gpuid 0 -f 230 -v 79` on the target node. Injects a replace-grade GPU fault via DCGM's field-injection API; HMA on Slurm watches DCGM counters and marks `Action:Replace`. **Do NOT just append an Xid line to `/var/log/syslog` on Slurm if you also want `srun --auto-resume=1` to retry the job** — the auto-resume plugin's diagnostic call inspects DCGM state, and a syslog-only Xid doesn't show up there, so it returns `ResumeAction=NONE` and the job stops instead of requeueing. The `dcgmi inject` path drives both signals (HMA marks the node + DCGM state is consistent so auto-resume sees the fault). See "Slurm job task failure" rows below and the `--auto-resume=1` callout in the Common AI-confusing details section. |
| Instance replacement (operator-driven) | `aws sagemaker batch-replace-cluster-nodes --cluster-name <name> --node-ids <i-...>` |
| Instance termination + shrink (operator-driven) | `aws sagemaker batch-delete-cluster-nodes` (reduces instance group count) |
| Slurm job task failure (no GPU fault) | `sudo kill -11 <task-pid>` on the node running the task. The task exits with status 139; auto-resume's diagnostic call returns `ResumeAction=NONE` (no hardware issue detected), so the job stops rather than requeueing. |
| Slurm job task failure (with GPU fault, exercising auto-resume) | `dcgmi inject` first to set up DCGM hardware-fault state, then `kill -11 <task-pid>` within ~5s. Auto-resume's diagnostic returns `ResumeAction=RETRYSTEP`; if `srun --auto-resume=1` was used, the job requeues after the node is replaced. A syslog-only Xid line does NOT produce this behavior on Slurm — DCGM state must show the fault for the diagnostic to classify it. |
| Scale up / scale down | SageMaker console "Update cluster" or `update-cluster` API. Adjust the instance group's target count. |
| Cluster patching (AMI / agent upgrade) | `aws sagemaker update-cluster-software --cluster-name <name>` |
| Drain a Slurm node without touching EC2 | `sudo scontrol update NodeName=<node> State=DRAIN Reason="..."` |
| Cordon an EKS node | `kubectl cordon <node>` followed by `kubectl drain <node>` |

Notes:
- HMA fault injection (Xid syslog line, `dcgmi inject`) only works
  on nodes where HMA is actually running:
  - **Slurm**: check `systemctl status sagemaker-health-monitoring-agent.service`.
    Older AMIs ship without HMA.
  - **EKS**: HMA is a DaemonSet (`health-monitoring-agent` /
    `health-monitoring-agent-non-nvidia` in namespace `aws-hyperpod`)
    with a node-selector restricted to GPU/Neuron SKUs.
    `kubectl get pods -n aws-hyperpod -o wide` and confirm a
    health-monitoring-agent pod is `Running` on the target node.
    CPU workers (m5, c5, etc.) have **no HMA pod scheduled** — fault
    injection will not trigger anything.
- **Use the right injection path for your orchestrator** — they're
  not interchangeable:
  - **HyperPod EKS** (default AMI: **AL2023**): inject by appending a
    fake `NVRM: Xid` line to **`/var/log/messages`**. The HMA
    DaemonSet tails the kernel log file; DCGM injection does NOT
    reach it.
  - **HyperPod Slurm** (default AMI: **Ubuntu 22.04**): inject via
    **`sudo dcgmi test --inject ...`**. HMA on Slurm reads DCGM
    counters; appending an Xid line to `/var/log/syslog` will mark
    the node via HMA but **leaves DCGM state clean**, which means the
    Slurm auto-resume plugin's diagnostic call returns
    `ResumeAction=NONE` and the running job will stop instead of
    requeueing. The `dcgmi inject` path drives both signals at once
    so auto-resume works correctly.
  - Custom AMIs follow their base distro's convention. If unsure on
    a given node, run `ls -la /var/log/{messages,syslog} 2>/dev/null`
    or check where `rsyslog`/`systemd-journald` is writing kernel
    messages.
- On EKS, the HMA DaemonSet `hostPath`-mounts the node's `/var/log`
  read-write so the syslog-write recipe works the same way from the
  node shell — but **write to whichever file matches the AMI**
  (`/var/log/messages` on AL2023). Writing Xid lines to a file HMA
  isn't tailing produces no reaction even though the syscall succeeds
  — failure mode is "node sits Schedulable for minutes after the
  inject", which is misleading.

## How long things take

Rough wall-clock budgets for HyperPod operations. Use these to plan
polling intervals, job time limits, and patience windows
(`--switches=N@T:T`, `max_switch_wait`, etc.).

| Event | Typical duration |
|---|---|
| Cluster-agent reconfigure (Slurm: rewrites `slurm.conf`/`topology.*`, runs `scontrol reconfigure`) | seconds |
| HMA fault detection → node marked `Action:Reboot`/`Action:Replace` (visible in `sinfo` on Slurm, node taint on EKS) | 5–30 seconds |
| Task failure → auto-resume diagnostic call → cluster agent response *(Slurm only)* | 10–30 seconds |
| Node reboot (HMA-driven, no EC2 replacement) | 1–3 minutes |
| Node replacement (any trigger): old EC2 terminate + new EC2 launch + lifecycle scripts + slurmd/EKS join + agent reconfigure | **20–30 minutes** |
| Topology rewrite after a replacement (Slurm: `slurm.conf` / `topology.yaml`. EKS: node labels.) | seconds, but happens at the end of the replacement window |
| `UpdateClusterSoftware` (cluster patching, full agent upgrade) | 30–60+ minutes |
| Cluster creation from scratch | 15–25 minutes |
| Scale-up adding new instances | similar to node replacement: 15–25 minutes per instance batch |
| Scale-down: orchestrator-side prune (Slurm controller / EKS API) | seconds |
| Scale-down: EC2 termination + storage cleanup | another 15–20 minutes after the orchestrator-side prune |

Implications:

- Long-running test jobs should set `--time` well above the replacement
  window (`--time=120` minutes is a safe floor when auto-resume is in
  play; the default 5–20 minute test runs will time out before
  replacements settle).
- Polling intervals: 30–60s during early reaction (the first minute
  after a trigger), 3–5 minutes during replacement wait, 10–20 minutes
  for `UpdateClusterSoftware`. Don't poll faster — the SageMaker control
  plane's view changes coarsely.
- Background sleeps inside SSM scripts should stay under the runner's
  60s timeout; for longer waits, return control to the local poller and
  re-enter via a fresh SSM session.
- The "orchestrator side updated; HyperPod API still catching up"
  window (or vice versa) is typically 1–5 minutes after a state
  change. Cross-check both views rather than trusting one.

## Things still unclear / under investigation

Open questions worth confirming. Update this section as the answers
become known.

- *(Slurm)* Does `UpdateClusterSoftware` rewrite `slurm.conf` from
  scratch (clobbering operator edits)?
- *(Both)* Is there a CloudWatch event or EventBridge signal that
  fires when a node enters `Failed` status?
- *(Both)* Why does HyperPod silently retry some replacement-boot
  failures but not the EFA-health-check failure mode?
- *(Both)* What is the documented contract for lifecycle script
  execution environment — user, working directory, UID, env vars,
  timeout values?
- *(Both)* Are `BatchRebootClusterNodes` and `BatchReplaceClusterNodes`
  available on both Slurm and EKS? Some docs imply EKS-only for some
  flows. Worth confirming before recommending to operators.
