# Self-hosted NVIDIA Cloud Functions on Amazon EKS, with SageMaker HyperPod as GPU compute

Run the **open-source** NVIDIA Cloud Functions (NVCF) control plane on Amazon EKS,
and register a SageMaker HyperPod cluster as a remote GPU compute plane.

## Relationship to the existing sample

This is an **addition**, not a replacement. The [parent sample](..) registers a
HyperPod cluster with **NVIDIA-hosted** NVCF: NVIDIA runs the control plane in NGC, and you
attach your cluster to it. That requires an NGC organization with the Cloud
Functions service entitlement and a Cloud Functions Admin role.

This directory covers the other topology. NVIDIA open-sourced NVCF under
Apache-2.0 in [`dsx-ai-factory/nvcf`](https://github.com/dsx-ai-factory/nvcf)
(moved from `NVIDIA/nvcf` on 2026-11-03), so you can run the control plane
yourself. The practical consequences:

| | NVIDIA-hosted (parent sample) | Self-hosted (this directory) |
|---|---|---|
| Control plane runs on | NGC | Your EKS cluster |
| NGC Cloud Functions entitlement | **Required** | **Not required** |
| NGC credential needed | Cloud Functions + Private Registry scopes | Registry/chart pull only |
| Compute registration | HyperPod -> NGC | HyperPod -> your control plane |
| You operate | The GPU cluster | Control plane + GPU cluster |

The entitlement difference is the main unlock. A self-hosted install needs an NGC
key only to pull charts from `helm.ngc.nvidia.com/nvidia/nvcf` and images from
`nvcr.io/nvidia/nvcf` — both **public** NGC catalogs. No image mirroring into a
private org, and no Cloud Functions entitlement.

## Architecture

Multi-cluster: the control plane and the GPU compute plane are separate EKS
clusters. This is the shape NVIDIA recommends, because one control plane can
register compute from several clusters.

```
                 VPC vpc-0123456789abcdef0  (10.0.0.0/16)
   ┌──────────────────────────────────────────────────────────────────────┐
   │                                                                      │
   │  EKS: nvcf-cp  (k8s 1.35)              EKS: my-hyperpod-eks-cluster (1.35)  │
   │  ┌────────────────────────────┐        ┌───────────────────────────┐ │
   │  │ NVCF control plane         │        │ NVCA operator             │ │
   │  │  api, nvct-api, api-keys   │        │ GPU Operator              │ │
   │  │  invocation, grpc-proxy    │◀──────▶│ SMB CSI driver            │ │
   │  │  sis, ess, notary, reval   │  NATS  │                           │ │
   │  │  Cassandra / NATS / OpenBao│  gRPC  │ HyperPod instance groups: │ │
   │  └────────────┬───────────────┘  HTTP  │  inference-group g5.8xl   │ │
   │               │                        │  training-group  g5.2xl   │ │
   │    ┌──────────▼──────────┐             └───────────────────────────┘ │
   │    │ nvcf-gateway        │                                           │
   │    │ Envoy Gateway       │   internal NLB, private to this VPC       │
   │    │ :80 :10081 :4222    │   (no internet-facing listener)           │
   │    └─────────────────────┘                                           │
   └──────────────────────────────────────────────────────────────────────┘
```

Both clusters sit in one VPC so the NVCA agent reaches the control plane over
private networking.

### The load balancer is internal, on purpose

NVIDIA's Gateway quickstart creates an **internet-facing** NLB and serves the
control-plane API over **plain HTTP on port 80**. That endpoint includes
`admin-token-issuer-proxy`, which mints the first admin token. This stack's
audience is your team, not anonymous callers, so
[`infra/gateway.yaml`](infra/gateway.yaml) sets
`service.beta.kubernetes.io/aws-load-balancer-internal: "true"`.

Operate it from outside the VPC with `kubectl port-forward`, which is
authenticated by IAM against the EKS API server. If you ever need this reachable
from the internet, put TLS and a real authenticating gate in front of it first —
do not just flip the annotation.

## Version compatibility

Two independent constraints pin the Kubernetes version:

- NVCF supports the latest Kubernetes minor and the two prior minors (N-2).
  With 1.37 current, that is **1.35 – 1.37**.
- SageMaker HyperPod supports up to **1.35**.

**1.35 is the only version both accept.** Run both clusters on it.

The release bundle also pins its own toolchain, and it is strict:

| Tool | Required | What Homebrew installs today |
|---|---|---|
| helm | **3.x** — `check-versions` hard-fails on 4.x | 4.2.0 ❌ |
| helmfile | **1.1.x** — "v1.2.0+ breaks deployment ordering" | 1.8.3 ❌ |
| helm-diff | required by helmfile apply/diff | not installed ❌ |

`brew install helm helmfile` therefore produces a toolchain this stack refuses.
[`scripts/install-toolchain.sh`](scripts/install-toolchain.sh) fetches the pinned
versions (checksum-verified) into a local `.toolchain/` and leaves your system
helm alone.

## Prerequisites

Both clusters need:

- A default-capable CSI `StorageClass` with dynamic provisioning. Both of our
  clusters shipped only the legacy in-tree `gp2`
  (`kubernetes.io/aws-ebs`), which is **not** backed by the EBS CSI driver. Install
  the `aws-ebs-csi-driver` addon and apply
  [`infra/storageclass-gp3.yaml`](infra/storageclass-gp3.yaml).

The compute cluster additionally needs:

- NVIDIA GPU Operator (HyperPod ships this).
- The [SMB CSI driver](https://github.com/kubernetes-csi/csi-driver-smb)
  (`smb.csi.k8s.io`), used by NVCA for shared model-cache storage.

And, for multi-cluster, a **resolvable wildcard domain** pointing at the Gateway.
See [You need real DNS](#you-need-real-dns-the-biggest-gap) — this is not
optional, and a raw ELB hostname will not do.

## Quickstart

```bash
# 0. Pinned toolchain (helm 3.x, helmfile 1.1.x, helm-diff)
./scripts/install-toolchain.sh
source ../.toolchain/env.sh

# 1. Control-plane cluster (k8s 1.35, CPU-only, in the existing VPC)
eksctl create cluster -f infra/cluster-controlplane.yaml

# 2. gp3 on both clusters
kubectl --context nvcf-cp      apply -f infra/storageclass-gp3.yaml
kubectl --context nvcf-compute apply -f infra/storageclass-gp3.yaml

# 3. Gateway API CRDs, Envoy Gateway, and the internal-NLB Gateway
kubectl --context nvcf-cp apply -f \
  https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/experimental-install.yaml
helm --kube-context nvcf-cp upgrade --install eg \
  oci://docker.io/envoyproxy/gateway-helm --version v1.5.4 \
  -n envoy-gateway-system --force-conflicts --wait
kubectl --context nvcf-cp apply -f infra/gateway.yaml

# 4. Private DNS: *.nvcf.internal -> the internal NLB (REQUIRED, see below)
./scripts/setup-dns.sh

# 5. Render the environment + secrets files and create the pull secrets
BUNDLE_DIR=/path/to/nvcf-1.0.4 DOMAIN=nvcf.internal ./scripts/render-config.sh

# 5. Install the control plane
make -C "$BUNDLE_DIR/deploy/stacks/self-managed" install HELMFILE_ENV=eks-multi
```

Steps 6-8 (register the HyperPod compute cluster, install the NVCA operator,
deploy and invoke a function) require the `nvcf-cli` binary — see
[Building nvcf-cli](#building-nvcf-cli).

## What tripped us up

Every item here was a real failure on a real cluster, not a hypothetical.

### You need real DNS — the biggest gap

The upstream CSP example sets `global.domain` to the Gateway's load balancer
hostname and never says you need a domain of your own. For plain HTTP routing
that reads fine, because a client can dial the bare load balancer and pass the
per-service name only as an HTTP `Host` header — which is exactly what the doc
describes for the NVCA agent.

But `global.workerEndpoints.*` are **URLs**, and clients DNS-resolve them. AWS
ELB DNS names resolve only exactly; there is no record for
`worker-api.<name>.elb.us-east-1.amazonaws.com`. So the install gets a long way
and then `grpc-proxy` fails — and it fails in a way that does not name DNS:

```
Using fixed bearer token authentication for NVCF client
  {"nvcf_host": "worker-api.<nlb>.elb.us-east-1.amazonaws.com:80", "tls_enabled": false}
...
GET /health  ->  503
Liveness probe failed: HTTP probe failed with statuscode: 503
Container helm-nvcf-grpc-proxy failed liveness probe, will be restarted
```

The pod connects to NATS, binds all its listeners, serves 503 on its own
`/health`, and gets killed in a loop. `make install` then aborts with
`Error: context deadline exceeded`, leaving the release `failed` and the
remaining releases never attempted.

The fix is a wildcard record in a Route 53 **private** hosted zone attached to
the VPC ([`scripts/setup-dns.sh`](scripts/setup-dns.sh)). Both clusters resolve
`*.nvcf.internal` to the internal NLB, and nothing is published to public DNS.

Two things to watch:

- **A wildcard does not cover the zone apex.** `*.nvcf.internal` does not match
  `nvcf.internal`, so every URL must use a subdomain. The compute-plane
  `icmsServiceURL` and `revalServiceURL` therefore point at `sis.<domain>` and
  `reval.<domain>` rather than the apex.
- **The zone is private**, so `nslookup` from your laptop fails by design.
  Verify from a pod:

  ```bash
  kubectl --context nvcf-cp run dnscheck --rm -it --restart=Never \
    --image=public.ecr.aws/docker/library/busybox:1.36 -- nslookup api.nvcf.internal
  ```

### Why that DNS problem aborted the whole install

A missing DNS record should have left one component unready, not stopped the
install. It escalated because of the probe configuration in the 1.0.4 chart.

`grpc-proxy`'s `/health` is a *readiness*-shaped check with two dependencies:

```go
health.Config{Name: "nvcf grpc api", ... client.Check(ctx, &pb.HealthCheckRequest{}) }
health.Config{Name: "nats",          ... nc.Status() != nats.CONNECTED }
```

The 1.0.4 chart wires **both** probes to that same endpoint:

```
liveness:  /health  port=client
readiness: /health  port=client
```

So an unreachable `nvcf-api` fails liveness, Kubernetes restarts the pod, and it
never converges. Helm then waits out its timeout and reports
`UPGRADE FAILED: context deadline exceeded`, `make install` aborts, and the
releases ordered after `grpc-proxy` are never attempted.

Upstream has already fixed this. The current source moves liveness to `/livez`
on the admin port, checking only NATS, with this comment:

> `livenessHandler` serves `/livez` on the admin port. It fails only when the
> NATS connection is closed: nats.go never reopens a closed connection, so a
> restart is the only recovery. nvcf-api and a reconnecting NATS connection are
> left to readiness, so their outages do not restart the pod.

The 1.0.4 chart predates that change. Two practical consequences:

- On 1.0.4, any transient `nvcf-api` unavailability can turn into a restart loop
  that fails the install. Treat a `grpc-proxy` CrashLoopBackOff as "check its
  dependencies", not "grpc-proxy is broken".
- Fixing the dependency is not enough on its own. The pod will be deep in
  CrashLoopBackOff exponential backoff, so force a fresh rollout:

  ```bash
  kubectl --context nvcf-cp rollout restart deploy/grpc-proxy-deployment -n nvcf
  ```

To confirm the dependency independently of the probe, call the gRPC health
service directly — first in-cluster, then through the Gateway. Both must return
`SERVING`:

```bash
kubectl --context nvcf-cp run grpccheck --rm -i --restart=Never \
  --image=docker.io/fullstorydev/grpcurl:v1.9.1 -- \
  -plaintext api.nvcf.svc.cluster.local:9090 grpc.health.v1.Health/Check

kubectl --context nvcf-cp run grpccheck --rm -i --restart=Never \
  --image=docker.io/fullstorydev/grpcurl:v1.9.1 -- \
  -plaintext worker-api.nvcf.internal:80 grpc.health.v1.Health/Check
```

That image is distroless, so pass `grpcurl` flags directly. Wrapping them in
`sh -c` fails with `exec: "sh": executable file not found in $PATH`.

1. **Helm 4 + Gateway API CRDs.** Envoy Gateway v1.5.4 bundles its own Gateway
   API CRDs, which collide with the v1.2.0 CRDs the NVCF doc has you apply first.
   Helm 4 defaults to server-side apply, so the install dies with
   `Apply failed with 2 conflicts ... conflicts with "kubectl-client-side-apply"`
   on `grpcroutes` and `httproutes`. Add `--force-conflicts`.

2. **The bundle rejects your Homebrew toolchain.** See the table above. The
   helmfile constraint is not cosmetic — the control plane must come up in
   dependency order, because Cassandra and OpenBao have to be ready before the
   services that run migrations against them.

3. **One `m6i.4xlarge` is not enough**, despite being the documented
   "Staging/Demo" node. The stack installs **29 Helm releases**; summed CPU
   *requests* are **19.75 vCPU** against **15.89** allocatable on one node. The
   two largest single requesters are `state-metrics` and `sis`
   (spot-instance-service) at 4 vCPU each. Use two nodes — the sizing guide does
   allow "one or two nodes" for this tier.

4. **Default replica counts are not single-node friendly.** Cassandra, NATS and
   OpenBao all render at 3 replicas, and the docs state each Cassandra replica
   must be on its own node. The rendered StatefulSets carry no pod
   anti-affinity, so on a 2-node pool they co-schedule and come up — but this is
   explicitly **not** an HA deployment. Do not describe it as one.
   `openbao.injector.replicas` is dropped to 1 here, which base.yaml itself
   recommends for single-node pools.

5. **EKS private subnets are often too small.** The existing cluster's private
   subnets were `/28` — 11 usable addresses. VPC CNI assigns a VPC IP per pod, so
   a `/28` cannot host a control-plane node. We created two `/24`s.

6. **VPC quota.** `us-east-1` was at 5/5 VPCs, so `eksctl create cluster` rolled
   back with `The maximum number of VPCs has been reached`. Putting the control
   plane in the existing HyperPod VPC is both the workaround and the better
   design — it keeps NVCA traffic private.

7. **HyperPod's small nodes can be at their pod ceiling.** The `g5.2xlarge`
   training node reported `maxPods=14` with 14 running, so the SMB CSI
   DaemonSet could never schedule there and sat `Pending` forever with
   `0/2 nodes are available: 1 Too many pods`. Scope the DaemonSet to the
   instance group that actually runs function workers:

   ```bash
   helm upgrade --install csi-driver-smb csi-driver-smb/csi-driver-smb \
     -n kube-system --version v1.17.0 \
     --set 'node.nodeSelector.sagemaker\.amazonaws\.com/instance-group-name=inference-group'
   ```

8. **Layer the environment file; don't copy it.** The doc says to copy
   `environments/base.yaml` (823 lines) and edit in place. The stack's
   `helmfile.d/*.gotmpl` already load `base.yaml` *then*
   `environments/${HELMFILE_ENV}.yaml`, so an override-only file works and keeps
   the diff reviewable. See
   [`infra/controlplane-env.yaml.template`](infra/controlplane-env.yaml.template).

9. **The docker credential is not a dockerconfigjson.** Despite the placeholder
   name `REPLACE_WITH_BASE64_DOCKER_CREDENTIAL`, each value is the raw
   `base64("$oauthtoken:<NGC_API_KEY>")`. Getting this wrong surfaces much later
   as `function create` returning
   `400 "must be base64 encoded username:password format"`.

10. **Multi-cluster needs two separate layers of endpoint config.**
    `global.workerEndpoints.*` sets the URLs the control plane advertises *into
    worker pods*; `global.nvcaOperator.selfManaged.*Override` (compute-plane file)
    sets how the *agent itself* reaches the control plane. Both are required, and
    the `nvcfApi.grpc` / `nvctApi.grpc` GRPCRoutes must be enabled.

11. **Register with the compute context active.** `register-cluster` probes the
    *current* kubectl context for its OIDC issuer and JWKS. Run it against the
    control-plane context and the agent later fails authentication with
    `Signed JWT rejected: no matching key(s) found`.

### Registry credentials for the function image

Getting a function image accepted has three separate traps.

**The control-plane profile inherits your CLI endpoints.** `control-plane profile
export` copies `base_http_url` / `base_grpc_url` from the nvcf-cli config into
`controlPlane.endpoints.computeReachable` and `controlPlane.gateway` — the
addresses the *compute cluster* will use. Operate the CLI through a
`kubectl port-forward` and `127.0.0.1:8080` is silently written into the profile
the GPU cluster is meant to dial. Keep two configs: one with the tunnel address
for local CLI use, one with the in-VPC address used only for `profile export`.
Check the result before registering:

```bash
grep -A 4 computeReachable deploy/stacks/self-managed/out/control-plane-profile.yaml
```

**NVCF requires a registered credential per registry hostname, even when the
node can already pull.** The HyperPod node's own IAM role pulls from ECR in the
same account without any imagePullSecret — verified with a plain pod — but
`function create` still fails:

```
400 "Missing CONTAINER registry credential for hostname '<acct>.dkr.ecr.us-east-1.amazonaws.com'"
```

**For ECR, NVCF wants AWS API keys, not a docker token.** The obvious move is
`aws ecr get-login-password`, which returns a bearer token for `docker login`.
Registering it as username `AWS` plus that password is rejected by NVCF's own
backend:

```
400 {"__type":"UnrecognizedClientException",
     "message":"The security token included in the request is invalid."}
```

That is an AWS API error, surfaced through NVCF: it authenticates to ECR itself
and treats the username/password as an access key ID and secret access key. A
12-hour `get-login-password` token is the wrong credential type, and it would
expire anyway. Supply an access key for a principal with ECR read on the
repository, or host the image somewhere NVCF accepts without one.

**kaniko plus a ConfigMap build context.** ConfigMap volumes project each key as
a symlink into `..data/`, and kaniko's `COPY` preserves the symlink, so it
dangles in the built image. The failure surfaces later as a missing file:

```
ERROR: Could not open requirements file:
  [Errno 2] No such file or directory: 'requirements.txt'
```

Dereference into an `emptyDir` with an initContainer (`cp -L`) and point kaniko
at that instead.

### The discovered instance type does not fit a shared HyperPod node

This is the one that stops a function from ever scheduling, and the error
message points at the wrong thing.

NVCA's dynamic GPU discovery registers an instance type derived from the node.
On an `ml.g5.8xlarge` (32 vCPU, A10G) it registers:

```json
{ "name": "NCP.GPU.A10G_1x", "value": "NCP.GPU.A10G",
  "cpuCores": 27, "systemMemory": "106Gi", "gpuMemory": "22Gi", "gpuCount": 1,
  "description": "NVIDIA-A10G (ampere family) on a g5.8xlarge machine" }
```

Note `name` carries a `_1x` suffix and `value` does not. Deploy with `value`
and the API rejects it (`Invalid InstanceType 'NCP.GPU.A10G'`); the node label
`nvca.nvcf.nvidia.io/instance-type` uses `value`. Pass `name` to
`function deploy create`.

The spec advertises **27** cpuCores, but the pod NVCF generates requests more
than that:

| Container | CPU | Memory | GPU |
|---|---|---|---|
| `inference` | 26600m | 106Gi | 1 |
| `utils` (sidecar) | 4000m | 4Gi | - |
| **total** | **30600m** | **110Gi** | 1 |

The 4-core `utils` sidecar is not counted against the advertised 27 cores, so
the pod needs 30.6 of the node's 32 vCPU. A HyperPod node is not empty: the
GPU Operator, EBS and SMB CSI drivers, `aws-node`, `kube-proxy` and HyperPod's
own `health-monitoring-agent` already request about 1595m. With 31850m
allocatable that leaves 30255m free, so the worker is short by **345m** and
stays `Pending` forever:

```
0/2 nodes are available: 1 Insufficient cpu,
  1 node(s) didn't match Pod's node affinity/selector.
```

Read that carefully. The "didn't match selector" node is the *training* group,
correctly excluded because NVCA only labels the inference group
`nvca.nvcf.nvidia.io/schedule=true`. The "Insufficient cpu" node is the
inference node. It is easy to misread this as a selector problem and go looking
at labels, which are fine.

Consequences:

- A `1x` instance type on a 32-vCPU node effectively requires a node dedicated
  to that one worker, and even then clears by only about 1.2 vCPU.
- Freeing capacity by relocating control-plane Deployments does not work here:
  the training node reports `maxPods=14` with 14 running, so nothing can move
  to it.
- Give the inference instance group a second node, or use a larger instance
  type, so one node can host the worker alongside the required DaemonSets.

`nvca.nvcf.nvidia.io/schedule=true` lands in the instance group's
`KubernetesConfig.DesiredLabels`, so HyperPod reapplies it to replacement and
new nodes. Scaling the group does not lose the label.

Two constraints make this harder to fix than it looks:

**You cannot change an instance group's type.** The obvious remedy is to move
the inference group to a larger shape, but HyperPod refuses:

```
ValidationException: Updating fields InstanceType on an InstanceGroup not supported
```

Add a new instance group with the larger type instead, and scale the old one
down once workers are landing on the new one.

**GPU quotas are per instance type.** Scaling the existing group out fails if
the type's quota is already consumed:

```
ResourceLimitExceeded: The account-level service limit
  'ml.g5.8xlarge for cluster usage' is 1 Instances,
  with current utilization of 1 Instances and a request delta of 1 Instances.
```

Check what you actually have before planning capacity, because each type is
quotaed separately and a dev account may have exactly 1 of each:

```bash
aws service-quotas list-service-quotas --service-code sagemaker --region us-east-1 \
  --query 'Quotas[?contains(QuotaName,`for cluster usage`) && Value > `0`].{Name:QuotaName,Value:Value}' \
  --output table
```

That per-type quota is also the workaround: adding a group of a *different*
GPU type draws on that type's own quota, so a second GPU node is possible even
when the first type is exhausted.

Trimming the node instead does not get there. On this cluster the reclaimable
requests (EBS CSI controller replicas, the SMB CSI controller and node
DaemonSets, the NVCA operator, kube-proxy) total about 275m against the 345m
shortfall, and cutting requests below real usage just moves the failure from
scheduling to runtime contention.

### Watch out for account cleanup automation

Partway through this build every control-plane pod went unschedulable with
`no nodes available to schedule pods`. The managed nodegroup had been set to
`desiredSize=0, minSize=0`. It was not an autoscaler: CloudTrail showed an
Amazon dev-account cleanup Lambda calling `UpdateAutoScalingGroup`.

```bash
aws cloudtrail lookup-events --region us-east-1 --start-time <iso8601> \
  --lookup-attributes AttributeKey=EventName,AttributeValue=UpdateAutoScalingGroup \
  --query 'Events[].{Time:EventTime,User:Username}' --output text
```

If you are building a long-running demo in a sandbox account, check for this
before you start and get the resources excluded. Do not disable the tooling.

Recovery after the nodes came back was not automatic, and the failure modes are
worth knowing:

- EBS volumes are AZ-pinned. Two OpenBao replicas had volumes in the same AZ
  while the replacement nodegroup put one node per AZ, so one replica could not
  schedule: `1 node(s) didn't match PersistentVolume's node affinity`. With
  quorum intact on the other two members, deleting the stranded replica's PVC
  and pod let it rebind in an AZ with capacity and re-sync.
- `nvct-api` came up `0/1` with
  `FileNotFoundException: /home/app/vault/secrets.json`. The OpenBao injector's
  default `webhook.failurePolicy: Ignore` admits pods while the injector is
  down, so they start with no vault-agent sidecar and no secrets. Deleting the
  pod once the injector is healthy fixes it; set `failurePolicy: Fail` if you
  would rather pods block than start broken.
- The NVCA agent reconnected on its own and returned to `healthy` without
  intervention.

### The compute plane needs a node that is not a GPU worker node

One function worker can claim essentially an entire GPU node, which leaves
nowhere for NVCA's own agent to run. Plan a non-GPU node into the compute
cluster from the start.

The instance type NVCA registers is derived from the node it finds, and it is
re-derived when the nodes change. With one `ml.g5.8xlarge` the catalog
advertised 27 cpuCores. After that node was replaced by an `ml.g5.16xlarge`,
the same instance type name came back as:

```json
{"name": "NCP.GPU.A10G_1x",    "cpuCores": 59,  "systemMemory": "224Gi"}
{"name": "NCP.GPU.A10G_1x.x2", "cpuCores": 123, "systemMemory": "457Gi"}
```

So the worker scaled with the node: it now requests 58520m plus the 4000m
`utils` sidecar, or 62520m of the node's 63770m allocatable. That is 98 percent
of the node, and 228Gi of 239Gi memory.

With the GPU node that full, the NVCA agent (100m for the agent plus 500m for
its webhook) cannot schedule:

```
0/2 nodes are available: 1 Insufficient cpu, 1 Too many pods.
```

The cluster then reports `HEALTH=Unknown` even though an already-running worker
keeps serving requests, so this can hide until the next deployment.

Two things made this easy to miss:

- It worked at first only because a second GPU node existed. The agent was
  living on the `ml.g5.8xlarge` while workers ran on the `ml.g5.16xlarge`.
  Scaling the smaller group to zero to save money removed the agent's home.
- `global.nodeSelector` in the compute-plane environment pins the **NVCA
  operator and agent**, not function workers, despite reading like the
  opposite. The worker pod carries no `nodeSelector` at all; NVCA places
  workers using the node label it manages itself,
  `nvca.nvcf.nvidia.io/schedule=true`, which it applies only to registered GPU
  instance groups. Training nodes never receive that label, so training and
  served inference stay separated without any selector of your own. Pinning the
  agent to a named instance group couples it to that group's lifecycle, so
  leave `nodeSelector.enabled: false`.

The fix is a small CPU-only instance group for the compute plane's control
components. It is also considerably cheaper than keeping a second GPU node
alive: an `ml.m5.2xlarge` costs a fraction of an `ml.g5.8xlarge`, and CPU
instance quotas are usually far more generous than GPU ones. NVCA will not put
workers there, because it only labels GPU groups.

See `infra/hyperpod-instance-groups.json` for the group layout used here:
training, a GPU inference group, and `cp-helpers` for NVCA and the CSI
controllers.

### The invocation callback needs its own address and its own DNS record

This is the last thing between a deployed function and a working one, and it is
not covered by `global.workerEndpoints` at all.

When a request arrives, the invocation service tells the worker where to call
back to attach to it. It builds that address by prefixing the worker's dashed
pod IP onto the host from its own
`WORKER_STREAM_PROPERTIES__SERVICE_ADDRESS`, which defaults to an in-cluster
Service name. A worker on a separate compute cluster accepts the request, tries
the callback, and fails:

```
error while making inference request
  Post "http://10-0-1-23.invocation.nvcf.svc.cluster.local:8080/v2/nvcf/worker/request-attach":
  dial tcp: lookup ... no such host
failed to handle request  error="response already sent"
```

The caller sees only `API error 504`, and the worker is `2/2 Running` and
reports `connected to NVCF` and `inference container is healthy`, so nothing
looks wrong from the outside. Confirm by reading the metering line the worker
emits, which records `"response_type":"ERRORED"`.

Setting `global.workerEndpoints.invocationServiceURL` does **not** fix this.
That value is not passed to the worker as an env var at all; the callback
address comes from the invocation service's own setting. Override it:

```yaml
invocation:
  env:
    WORKER_STREAM_PROPERTIES__SERVICE_ADDRESS: "http://invocation.<domain>"
```

Omit the port. The in-cluster default uses `:8080`, which the Gateway load
balancer does not expose; the HTTP listener is on 80.

Then add a **second, nested** DNS record. The advertised address is
`10-0-1-23.invocation.<domain>`, which is two labels below the zone, and a
wildcard matches only one label, so `*.<domain>` does not cover it. The
invocation HTTPRoute already carries the matching wildcard hostname:

```bash
kubectl --context nvcf-cp get httproute invocation-service -n envoy-gateway \
  -o jsonpath='{.spec.hostnames}'
# ["*.invocation.nvcf.internal","invocation.nvcf.internal"]
```

`scripts/setup-dns.sh` creates both `*.<domain>` and `*.invocation.<domain>`.

One caveat worth knowing: routed through the Gateway, the per-pod hostname
resolves to the Service, which load-balances across all invocation-service
pods. The address is therefore only reliably the intended pod while
`replicaCount` is 1. Raising invocation-service replicas needs a routing path
that preserves per-pod addressing.

After that change, invocation works:

```
Status: fulfilled
Response: { "result": "hello from SageMaker HyperPod" }
```

and the worker runs on the HyperPod GPU node with a GPU attached, while the
control plane tracks the allocation:

```
pod 0-sr-<request-id>  node hyperpod-i-0abcdef1234567890  (ml.g5.16xlarge)  gpu 1
gpuUsage: { "A10G": { "capacity": 2, "allocated": 1, "available": 1 } }
```

### API key scopes do not cover cluster management

`nvcf-cli init` mints an admin token with a `cluster-management` scope, but
`nvcf-cli api-key generate` issues function, task and event-ledger keys without
it. The CLI prefers the API key for "user operations", so `cluster list` fails
even though registration succeeded:

```
Using API KEY (user operation) for GET /v2/nvcf/clusterGroups
  - expects scope: list_cluster_groups
Error: API error 403 {"detail":"Access Denied", ...}
```

This is a scope mismatch, not a broken registration. Confirm the cluster from
the compute side instead, which is the authoritative signal:

```bash
kubectl --context nvcf-compute get nvcfbackend -n nvca-operator
# NAME            AGE   VERSION   HEALTH
# hyperpod-nvcf   49s   3.10.3    healthy
```

## Building nvcf-cli

The compute-plane Makefile calls `nvcf-cli` during registration, so it is
required for the multi-cluster flow. **There are no published binaries** — the
`src/clis/nvcf-cli/v1.27.0` release has zero assets, and the `nvcf/1.0.4` bundle
ships only Helm stacks. It must be built from source:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/dsx-ai-factory/nvcf.git
cd nvcf && git sparse-checkout set src/clis/nvcf-cli src/libraries/go
cd src/clis/nvcf-cli
go build -o /usr/local/bin/nvcf-cli .     # pure Go; module is self-contained
```

If your host toolchain or sandbox blocks a local build, build it in a pod on
a cluster that has egress and copy the binary out. Cross-compiling works
because the CLI is pure Go:

```bash
kubectl run nvcf-cli-build --image=golang:1.26 --restart=Never -- sleep 3600
kubectl exec nvcf-cli-build -- sh -c '\
  cd /tmp && git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/dsx-ai-factory/nvcf.git && cd nvcf && \
  git sparse-checkout set src/clis/nvcf-cli src/libraries/go && \
  cd src/clis/nvcf-cli && \
  GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -o /tmp/nvcf-cli .'
kubectl cp default/nvcf-cli-build:/tmp/nvcf-cli ./nvcf-cli && chmod +x ./nvcf-cli
```

The upstream README documents a Bazel build
(`bazel build //src/clis/nvcf-cli:nvcf-cli`), which also works and is what CI
uses. The module has no `replace` directives pointing outside its own tree, so a
plain `go build` is sufficient.

## Layout

| Path | What it is |
|---|---|
| `infra/cluster-controlplane.yaml` | eksctl config for the CPU-only control-plane cluster |
| `infra/storageclass-gp3.yaml` | CSI-backed default StorageClass for both clusters |
| `infra/gateway.yaml` | EnvoyProxy, GatewayClass and Gateway — internal NLB |
| `infra/controlplane-env.yaml.template` | Override-only helmfile environment |
| `scripts/install-toolchain.sh` | Pinned helm 3.x / helmfile 1.1.x / helm-diff |
| `scripts/render-config.sh` | Renders env + secrets, creates registry pull secrets |

## References

- [dsx-ai-factory/nvcf](https://github.com/dsx-ai-factory/nvcf) — Apache-2.0 source
- [CSP end-to-end installation example](https://github.com/dsx-ai-factory/nvcf/blob/main/docs/self-managed/csp-end-to-end-example-installation.md)
- [Infrastructure sizing](https://github.com/dsx-ai-factory/nvcf/blob/main/docs/overview/infrastructure-sizing.md)
- [Self-managed cluster registration](https://github.com/dsx-ai-factory/nvcf/blob/main/docs/compute-plane/cluster-management/self-managed.md)
