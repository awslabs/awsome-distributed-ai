# Running Jupyter on a Compute Node

How to run a Jupyter (Lab/Notebook) server on a compute node as a **Slurm job**
and reach it from your local browser, with no inbound ports opened on the
cluster. Works for the single-`ubuntu`-user default cluster and for
[multi-user (OpenLDAP)](./USER-MANAGEMENT.md) clusters — the per-user notes are
called out inline.

## How it works

```
Browser (http://localhost:8888)
   │
   │  aws ssm start-session … AWS-StartPortForwardingSessionToRemoteHost
   ▼
Login node  (SSM session entry point)
   │  forwards to <compute-node-ip>:<port> inside the cluster security group
   ▼
Compute node — Jupyter server, launched as an sbatch job
```

- **Jupyter runs as a Slurm job** (not hand-started on a node): submitting the
  job wakes the queue from 0 nodes, GPU allocation (`--gres`) and accounting
  work as usual, and the `--time` limit auto-terminates a forgotten server.
  Anything installed by hand on a node would also be lost when PCS replaces
  the node — a job is re-submittable.
- **Access goes through the login node over SSM.** The
  `AWS-StartPortForwardingSessionToRemoteHost` document opens the session on
  the *login* node and forwards to the compute node inside the cluster
  security group. No `SSHAccessCidr`, no SSH keys, nothing exposed — and it is
  already permitted by the stock
  [`cluster-user-iam.yaml`](../assets/cluster-user-iam.yaml) policy (which
  allows `ssm:StartSession` on the login node only, plus the port-forwarding
  documents).
- **The Jupyter token is the user boundary.** The server binds to the node's
  private IP, so any cluster user could reach the port; the token (stored
  under `$HOME`, mode 600) is what keeps a session private to its owner.

## Prerequisites

- On your workstation: AWS CLI + the
  [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html),
  and either cluster-admin credentials or the
  [cluster-user policy](./IAM.md).
- **Multi-user clusters:** your `$HOME` must exist before the first job — log
  in to the login node (SSH or SSM) once so `pam_mkhomedir` creates it. Slurm
  jobs do not create home directories.
- **Do not run Jupyter on the login node.** It has no GPUs and is shared with
  the monitoring stack; always launch the server as a Slurm job on a compute
  node, as Step 2 does.

## Step 1 — one-time: create a Jupyter environment on `/home`

On the login node (as the user who will run Jupyter). `/home` is shared
FSx for OpenZFS, so every compute node sees the venv:

```bash
python3 -m venv $HOME/jupyter-env
$HOME/jupyter-env/bin/pip install --upgrade pip jupyterlab
```

**To run the GPU-visibility verify cell below**, also install PyTorch here so
the kernel can `import torch` without another install roundtrip. The
PCS-Ready DLAMI's NVIDIA driver (595.x) supports current CUDA wheels; the
`cu130` build matches the image's CUDA 13.2 stack (see
[PCS-READY-DLAMI.md](./PCS-READY-DLAMI.md)):

```bash
$HOME/jupyter-env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
```

> For ML work, also set `HF_HOME=/fsx/$USER/.hf-cache` in your notebooks/jobs —
> the HuggingFace cache must not live on NFS `/home` (file-locking errors under
> concurrent access), and a per-user path avoids ownership clashes on
> multi-user clusters.

## Step 2 — submit the Jupyter job

Save the following as `$HOME/jupyter.sbatch`. It has **no `--partition` line on
purpose** — you pick the queue (and, for GPUs, the GPU count) at submit time,
so the *same* script serves both CPU and GPU sessions:

```bash
#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --time=8:00:00            # auto-terminate after 8 h — adjust to taste
#SBATCH --output=%u-jupyter-%j.log

umask 077   # everything this job writes is owner-only

# Port derived from the job ID (range 8000-8999). Collisions are unlikely but
# possible if two servers share a node with job IDs differing by a multiple of 1000.
PORT=$((8000 + SLURM_JOB_ID % 1000))
NODE_IP=$(hostname -I | awk '{print $1}')

# Token kept out of the job log on purpose (the log may be readable by others
# depending on where you submit from); mode-600 file under $HOME instead.
TOKEN_FILE=$HOME/.jupyter-token-$SLURM_JOB_ID
openssl rand -hex 24 > "$TOKEN_FILE"

cat <<EOM
======================================================================
Jupyter is running on compute node $NODE_IP, port $PORT (job $SLURM_JOB_ID).
To connect, follow "Step 3" in JUPYTER.md with these values:
    NODE_IP = $NODE_IP
    PORT    = $PORT
    token   = run  cat $TOKEN_FILE  on the login node
Stop the server:  scancel $SLURM_JOB_ID
======================================================================
EOM

source $HOME/jupyter-env/bin/activate
exec jupyter lab --no-browser --ip="$NODE_IP" --port="$PORT" \
  --ServerApp.token="$(cat "$TOKEN_FILE")" \
  --notebook-dir="$HOME"
```

Then submit it with `sbatch` **from `$HOME`** — Slurm's default working
directory for a job is the submitter's CWD, and the `--output=%u-jupyter-%j.log`
line above resolves relative to that. Submitting from anywhere else (e.g. an
SSM shell that lands in `/var/…`) drops the log there instead of `$HOME`, and
the connect step in Step 3 won't find it. **`-p <queue>` is required** — Slurm
has no default partition, so a bare `sbatch jupyter.sbatch` is rejected with
*"invalid partition specified"*. Run `sinfo` to list your queues, then:

```bash
cd $HOME

# CPU session — pick a CPU queue (e.g. cpu1)
sbatch -p cpu1 jupyter.sbatch

# GPU session — pick a GPU queue and request GPUs with --gres
sbatch -p gpu-g6 --gres=gpu:1 jupyter.sbatch
```

GPU allocation behaviour is detailed in [Using GPUs](#using-gpus) below.

The first submission on an idle queue waits ~2–3 minutes for the node to
scale up (8–12 minutes if the node is also running its first-boot
Enroot/Pyxis install).

## Step 3 — connect

The job log (`~/<user>-jupyter-<jobid>.log`, printed by the running job) gives
you the compute node's `NODE_IP` and `PORT`. With those two values:

**1. On your workstation** — open the SSM tunnel (fill in the `NODE_IP` /
`PORT` from the job log). The login instance ID is resolved from the
CloudFormation stack via the PCS API — same recipe used in
[README §6](../README.md#6-accessing-the-cluster), so it works across
multi-cluster VPCs without accidentally targeting another cluster's login:

```bash
STACK_NAME=pcs-ml-cluster        # your CloudFormation stack name
AWS_REGION=us-east-1             # your region

CLUSTER_ID=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" --query 'Stacks[0].Outputs[?OutputKey==`ClusterId`].OutputValue' --output text)
[ -n "$CLUSTER_ID" ] && [ "$CLUSTER_ID" != "None" ] || { echo "No ClusterId — check STACK_NAME/AWS_REGION"; return 1; }

LOGIN_CNG_ID=$(aws pcs list-compute-node-groups --cluster-identifier "$CLUSTER_ID" --region "$AWS_REGION" --query 'computeNodeGroups[?name==`login`].id' --output text)
LOGIN_ID=$(aws ec2 describe-instances --region "$AWS_REGION" --filters "Name=tag:aws:pcs:compute-node-group-id,Values=$LOGIN_CNG_ID" "Name=instance-state-name,Values=running" --query 'Reservations[0].Instances[0].InstanceId' --output text)

aws ssm start-session --region "$AWS_REGION" --target "$LOGIN_ID" \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters "host=<NODE_IP>,portNumber=<PORT>,localPortNumber=8888"
```

Leave this running — **Ctrl-C closes the tunnel** (the Jupyter job keeps
running; reconnect any time until the job ends).

**2. On the login node** — read the token:

```bash
cat ~/.jupyter-token-<jobid>
```

**3. In your browser** — open `http://localhost:8888/?token=<token>`.

> **Required IAM.** The CLI calls above are already permitted by the stock
> [`cluster-user-iam.yaml`](../assets/cluster-user-iam.yaml) policy (and by any
> broader admin credentials). Under a restricted role, ensure it grants:
> `cloudformation:DescribeStacks`, `pcs:ListComputeNodeGroups`,
> `ec2:DescribeInstances` (find the login node); `ssm:StartSession` on the
> login instance (`ssm:resourceTag/Name = <ClusterStackName>-login`) and on
> the `AWS-StartPortForwardingSessionToRemoteHost` document; and
> `ssm:TerminateSession` / `ssm:DescribeSessions` to end the tunnel.

### Running without a token (single-user clusters only)

If you're the only user on the cluster and want to skip the token
copy-paste, swap the `--ServerApp.token=…` line in the sbatch script
with the disabled-auth form:

```bash
exec jupyter lab --no-browser --ip="$NODE_IP" --port="$PORT" \
  --ServerApp.token='' --ServerApp.password='' \
  --notebook-dir="$HOME"
```

The token file (and the `openssl rand` line above it) are no longer
needed and can be dropped; you can also drop the `token   = ...` line
from the sbatch banner heredoc. Steps 1 and 3 of Step 3 are unchanged —
skip sub-step 2 (there is no token to read). Connect to
`http://localhost:8888/lab` — no token required.

Jupyter's default XSRF protection is left **enabled** (this recipe does
not set `--ServerApp.disable_check_xsrf`); the Lab UI's own cookie+header
flow works fine with it on, and it is the one browser-side guard that
still prevents a random webpage open in your browser from POSTing to
`http://localhost:8888` and executing code on the compute node.

> **Do NOT do this on a multi-user cluster.** Jupyter binds to the compute
> node's private IP; any other cluster user's job on any other node in the
> same VPC subnet can reach that IP:port. The token is what separates one
> user's notebook (with your `$HOME` and credentials) from another user's.
> Removing it means anyone who can run a job on this cluster can attach to
> your kernel. Only use the token-less form when **every IAM principal who
> can reach the VPC or submit jobs is trusted as the same person** — the
> single-`ubuntu`-user default cluster is that case by construction;
> `DirectoryService=OpenLDAP-LoginNode` (or any shared multi-user setup)
> is not, so always keep the token there.

## Stopping

```bash
scancel <jobid>            # or let the --time limit expire
rm ~/.jupyter-token-<jobid>
```

The queue scales back to 0 after the idle timeout, so a stopped Jupyter job
costs nothing.

## Using GPUs

You launch a GPU session with the same script and the `sbatch -p <gpu-queue>
--gres=gpu:N` form shown in Step 2 — nothing else changes.

### How the GPU allocation behaves

- **`--gres=gpu:N` is the only knob.** Slurm sets `CUDA_VISIBLE_DEVICES`
  accordingly and PyTorch sees exactly those GPUs. Request only what you
  need — the remaining GPUs on a multi-GPU node stay available for other
  jobs.
- **Multi-GPU works inside one notebook.** All allocated GPUs are visible
  to the kernel, so `DataParallel` / FSDP / `accelerate` run as usual.
- **Sizing.** `--gres=gpu:1` is plenty for most interactive exploration.
  Keep `--time` tight — an idle notebook holds its GPUs until the job
  ends; prefer a batch job for multi-hour training.
- **Multi-NIC GPU nodes (P5/P6) work as-is.** `hostname -I` returns one
  address per EFA NIC; the script binds to the first, which is
  same-VPC-reachable from the login node.
- **Containerized kernels (optional).** Wrap `jupyter lab` in
  `srun --container-image=<sqsh> --container-mounts=/fsx,$HOME …` to run
  the notebook inside a Pyxis-imported image (see [README §7](../README.md#7-running-a-job)).

### Verify GPU visibility from the notebook

Paste this into a new notebook cell after the kernel comes up. It confirms
Slurm's `--gres` count, that PyTorch sees the same set of GPUs, and that a
kernel actually runs on the allocated device:

```python
import os, torch

# 1. What Slurm handed us
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_JOB_GPUS       =", os.environ.get("SLURM_JOB_GPUS"))

# 2. What PyTorch actually sees
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1024**3:.1f} GiB  SM {p.major}.{p.minor}")

# 3. Prove compute runs on the allocated GPU (raises if the kernel can't launch)
if torch.cuda.is_available():
    x = torch.randn(4096, 4096, device="cuda:0")
    print("matmul on cuda:0 -> sum =", float((x @ x).sum()))
```

Expected: `torch.cuda.device_count()` **equals the `--gres=gpu:N` you
requested**. Note that `nvidia-smi` from inside the job may still show
all physical GPUs on the node — that's expected on these clusters;
allocation is enforced per-process via `CUDA_VISIBLE_DEVICES`, which
`torch.cuda.device_count()` respects.

