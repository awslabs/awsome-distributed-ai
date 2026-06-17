# Infrastructure Tests (Tests 1-3, 8)

Validates cluster infrastructure: monitoring stack, container runtime (Enroot/Pyxis)
first-boot install, and the pre-baked AMI build path.

---

## Test 1: Monitoring stack

With `MonitoringStack=Prometheus-LoginNode` (default), Prometheus/Grafana/exporters install on the
login node and DCGM/node exporters on the compute nodes.

```bash
# On the login node:
docker ps --format "table {{.Names}}\t{{.Status}}"          # login: prometheus, grafana, nginx, cloudwatch-exporter, node-exporter, pushgateway
tail -5 /var/log/monitoring-install.log                     # ends "...complete (exit 0)"
ls -la /opt/aws-parallelcluster-monitoring                  # installed on node-local /opt (NOT /home)
curl -s http://localhost:9090/api/v1/targets | \
  python3 -c 'import sys,json;[print(t["labels"].get("instance"),t["health"]) for t in json.load(sys.stdin)["data"]["activeTargets"]]'
curl -s http://localhost:6817/metrics | head               # Slurm OpenMetrics
```

**Expected:** the six login containers are `Up`; install log exits 0; the tree is under
`/opt`; all Prometheus targets `up`; Slurm OpenMetrics returns Prometheus-format text.
For dashboard access (SSM port-forward or public CIDR) see
[README §8 Monitoring](../README.md#8-monitoring). Use `MonitoringVersion=v2.9.1`+ on PCS.

---

## Test 2: Enroot/Pyxis container runtime (first-boot install)

This is the default path used by every cluster that doesn't override `AmiId`:
`PostInstallScriptUrl` runs `install-enroot-pyxis.sh` once on each node at first boot.
The pre-baked-AMI path is validated separately as [Test 8](#test-8-pre-baked-ami-build-standalone-dlami-template).

Deploy `pcs-ml-cluster-deploy-all.yaml` with both `AmiId` and `PostInstallScriptUrl`
left at their defaults (so SSM auto-resolves the latest PCS-Ready DLAMI and
post-install runs the Enroot/Pyxis installer), then on any node:

> **The default `PostInstallScriptUrl` is now an `s3://` URL** (empty →
> `s3://<S3BucketName>/<S3KeyPrefix>scripts/install-enroot-pyxis.sh`, fetched with
> the instance role). Verified end-to-end against a **private** test bucket: the
> node's post-install log shows `Downloading post-install script from s3://…` and
> `pyxis.conf` is installed — so no public S3 is required (works in dev accounts). An
> `http(s)://` value still works too (curl), for public/GitHub-raw scripts.

```bash
which enroot                                                       # /usr/bin/enroot
ls /opt/aws/pcs/scheduler/slurm-*/lib/slurm/spank_pyxis.so         # per-version Pyxis SPANK plugin
cat /etc/aws/pcs/scheduler/slurm-*/plugstack.conf.d/pyxis.conf     # points at the matching .so
tail -1 /var/log/pcs-post-install.log                              # "...completed (exit 0)"
```

**Expected:** `enroot` on `PATH`; a `spank_pyxis.so` under the **cluster's** Slurm version
dir, and the plugstack `pyxis.conf` referencing that exact path; post-install log exits 0.
The Test 1/6/7 container jobs are the functional proof that Pyxis works.

> **⚠️ Regression-test rule for `assets/scripts/install-enroot-pyxis.sh`.** This script has bitten
> us repeatedly in ways a single 25.11 GPU run does not catch. **Any change to it MUST be
> retested across the full matrix at the top of this guide**, specifically:
> - **All supported Slurm versions** (25.05 **and** 25.11). The Pyxis SPANK plugin is
>   ABI-locked to its Slurm version — a plugin built for the wrong version stops slurmd from
>   starting (`Incompatible Slurm plugin version`). The script builds Pyxis for the version
>   passed in `PCS_SLURM_VERSION` and installs the `.so` to a per-version path; a regression
>   here only shows on the *other* version.
> - **The pre-baked AMI path too** ([Test 8](#test-8-pre-baked-ami-build-standalone-dlami-template)).
>   `pcs-ready-dlami-with-enroot-pyxis.yaml` carries its **own copy** of the Enroot/Pyxis
>   steps in its Image Builder UserData — editing `install-enroot-pyxis.sh` does **not**
>   change the AMI path until you rebuild. Build an AMI per supported `SlurmVersion`,
>   deploy a cluster pinned to it (`AmiId=<ami-xxx>` + `PostInstallScriptUrl=' '`), and
>   run a container job.
> - **On a clean first boot**, not a hand-patched node — post-install runs before
>   slurmd/profile.d/controller exist, and several bugs only appear there.

---

## Test 3: CPU queue

Deployed by default as `cpu1` (`DeployOnDemandCNG=true`, `c6i.4xlarge`, 0–4 dynamic).

```bash
sinfo                                          # cpu1 partition present, nodes idle~
srun --partition=cpu1 --nodes=1 hostname       # a node powers up and runs
```

**Expected:** `cpu1` shows in `sinfo`; a dynamically-scaled node launches and the job
returns its hostname.

---

---

## Test 8: Pre-baked AMI build (standalone DLAMI template)

**When to run:** when `pcs-ready-dlami-with-enroot-pyxis.yaml` or any code it bakes
in (`assets/scripts/install-enroot-pyxis.sh`) changes — the cluster stack does NOT run
Image Builder, so a fix to the install script is only in the AMI after a rebuild.
Skip this test if you only touched the cluster templates.

This is an **independent flow**, not a deploy-all parameter: build an AMI with the
standalone template, then deploy a cluster pinned to that AMI ID with
`PostInstallScriptUrl=' '` so nothing else runs at boot.

The AMI is **single-Slurm-version by design** (Pyxis SPANK plugin ABI is
version-locked) — so when you run this test, run it for **every supported
`SlurmVersion`** that the install-script change could affect (typically both 25.05
and 25.11).

### Step 1 — build the AMI (~30 min one-time per Slurm version)

```bash
SLURM_VERSION=25.11   # repeat for 25.05 if relevant

aws cloudformation create-stack \
  --stack-name pcs-dlami-${SLURM_VERSION/./} \
  --template-url https://<bucket>.s3.amazonaws.com/<prefix>pcs-ready-dlami-with-enroot-pyxis.yaml \
  --parameters ParameterKey=SlurmVersion,ParameterValue=${SLURM_VERSION} \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region us-west-2

aws cloudformation wait stack-create-complete \
  --stack-name pcs-dlami-${SLURM_VERSION/./} \
  --region us-west-2

AMI_ID=$(aws cloudformation describe-stacks \
  --stack-name pcs-dlami-${SLURM_VERSION/./} \
  --query 'Stacks[0].Outputs[?OutputKey==`DLAMIforPCSAmiId`].OutputValue' \
  --output text --region us-west-2)
echo "$AMI_ID"   # ami-0xxxxxxxxxxxxxxxx
```

### Step 2 — deploy a cluster pinned to that AMI

```bash
aws cloudformation create-stack \
  --stack-name pcs-amitest-${SLURM_VERSION/./} \
  --template-url https://<bucket>.s3.amazonaws.com/<prefix>pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-west-2a \
    ParameterKey=SlurmVersion,ParameterValue=${SLURM_VERSION} \
    ParameterKey=AmiId,ParameterValue=${AMI_ID} \
    ParameterKey=PostInstallScriptUrl,ParameterValue=' ' \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region us-west-2
```

### Step 3 — verify the bake landed and a container job runs

On any node from the new cluster:

```bash
which enroot                                                       # /usr/bin/enroot (pre-baked)
ls /opt/aws/pcs/scheduler/slurm-${SLURM_VERSION}/lib/slurm/spank_pyxis.so  # built for matching Slurm
cat /etc/aws/pcs/scheduler/slurm-${SLURM_VERSION}/plugstack.conf.d/pyxis.conf  # references the .so
test ! -s /var/log/pcs-post-install.log && echo "post-install did not run (PostInstallScriptUrl=' ')"
```

Then a container job through the login node (same form as Test 2):

```bash
srun --partition=cpu1 --nodes=1 --ntasks=1 \
  --container-image=ubuntu:22.04 bash -c "echo PYXIS_FROM_AMI_OK"
```

**Expected:** `enroot` and the per-version Pyxis files exist **without** the
post-install hook running (because `PostInstallScriptUrl=' '`); the container job
prints `PYXIS_FROM_AMI_OK`. Slurmd starts cleanly (no `Incompatible Slurm plugin
version` in `journalctl -u slurmd`).

### Step 4 — clean up

```bash
aws cloudformation delete-stack --stack-name pcs-amitest-${SLURM_VERSION/./} --region us-west-2
aws cloudformation delete-stack --stack-name pcs-dlami-${SLURM_VERSION/./} --region us-west-2
```

The DLAMI stack's AMI itself is **not** automatically deregistered when the stack is
deleted — if you need to free its EBS snapshots, deregister the AMI manually
(`aws ec2 deregister-image --image-id $AMI_ID`) and delete the snapshot.

---
