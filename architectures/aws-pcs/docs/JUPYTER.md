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

## Step 1 — one-time: create a Jupyter environment on `/home`

On the login node (as the user who will run Jupyter). `/home` is shared
FSx for OpenZFS, so every compute node sees the venv:

```bash
python3 -m venv $HOME/jupyter-env
$HOME/jupyter-env/bin/pip install --upgrade pip jupyterlab
```

> For ML work, also set `HF_HOME=/fsx/$USER/.hf-cache` in your notebooks/jobs —
> the HuggingFace cache must not live on NFS `/home` (file-locking errors under
> concurrent access), and a per-user path avoids ownership clashes on
> multi-user clusters.

## Step 2 — submit the Jupyter job

Save as `$HOME/jupyter.sbatch` and submit **from your home directory** with
`sbatch jupyter.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=cpu1          # or your GPU queue; then add e.g. --gres=gpu:1
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

The first submission on an idle queue waits ~2–3 minutes for the node to
scale up (8–12 minutes if the node is also running its first-boot
Enroot/Pyxis install).

## Step 3 — connect

The job log (`~/<user>-jupyter-<jobid>.log`, printed by the running job) gives
you the compute node's `NODE_IP` and `PORT`. With those two values:

**1. On your workstation** — open the SSM tunnel (fill in your Region and the
`NODE_IP` / `PORT` from the log):

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

> **Required IAM.** The two CLI calls above are already permitted by the stock
> [`cluster-user-iam.yaml`](../assets/cluster-user-iam.yaml) policy (and by any
> broader admin credentials). Under a restricted role, ensure it grants:
> `ec2:DescribeInstances` (find the login node); `ssm:StartSession` on the
> login instance (`ssm:resourceTag/Name = PCS-login*`) and on the
> `AWS-StartPortForwardingSessionToRemoteHost` document; and
> `ssm:TerminateSession` / `ssm:DescribeSessions` to end the tunnel.

## Stopping

```bash
scancel <jobid>            # or let the --time limit expire
rm ~/.jupyter-token-<jobid>
```

The queue scales back to 0 after the idle timeout, so a stopped Jupyter job
costs nothing.

## Using GPUs

The same sbatch script works on a GPU queue — change the `#SBATCH` header, and
request GPUs with `--gres`:

```bash
#SBATCH --partition=gpu-g6        # your GPU queue name
#SBATCH --gres=gpu:1              # GPUs for this Jupyter session
#SBATCH --time=8:00:00
```

Add the ML stack to the venv from Step 1 (once):

```bash
$HOME/jupyter-env/bin/pip install torch   # + transformers, etc.
```

How the GPU allocation behaves:

- **Slurm enforces the `--gres` count.** The job gets `CUDA_VISIBLE_DEVICES`
  set to its allocated GPUs (e.g. `0,1` for `--gres=gpu:2` on a 4-GPU
  g6.12xlarge), so frameworks like PyTorch see exactly the requested GPUs —
  `torch.cuda.device_count()` matches the `--gres` count. GPU node groups
  configure gres automatically (e.g. `Gres=gpu:L4:4`; check with
  `scontrol show node <node>`).
- **Multi-GPU works inside one notebook.** All allocated GPUs are visible to
  the kernel, so `DataParallel` / FSDP / `accelerate` with
  `num_processes=<gres count>` run as usual. Leave GPUs you don't need
  unrequested — on multi-GPU nodes Slurm can schedule other jobs (another
  user's Jupyter, batch training) onto the remaining GPUs.
- **Sizing:** request only what you interactively need (`--gres=gpu:1` is
  plenty for most exploration) and keep `--time` tight — an idle notebook
  holds its GPUs until the job ends. For multi-hour *training*, prefer a
  batch job over a notebook so the GPUs free up when the run finishes.
  (Remember `HF_HOME=/fsx/$USER/.hf-cache` from Step 1 for model downloads.)
- **Multi-NIC GPU nodes (P5/P6) work as-is.** On these instances `hostname -I`
  returns dozens of addresses (one per EFA NIC). The script binds Jupyter to
  the first entry, which in practice is the primary interface's IP (verified
  on p5.48xlarge, 32 NICs) — and any entry is same-VPC-reachable from the
  login node, so the port-forward path works regardless. The kernel saw all
  8 H100s with `--gres=gpu:8`.
- **Containerized kernels (optional):** to use an NGC image as the notebook
  environment instead of a venv, wrap the server in Pyxis:
  `srun --container-image=<image> --container-mounts=/fsx,$HOME …` around the
  `jupyter lab` command inside the job. Import large images once to
  `/fsx/*.sqsh` (see the [README §7](../README.md#7-running-a-job) enroot
  note) and pass the `.sqsh` path as the image.

## Notes

- **Multi-user:** each user runs their own server job under their own UID; the
  job-ID-derived port makes same-node collisions unlikely (see the comment in
  the script). Keep tokens in `$HOME` (created mode 600 by the script) — treat
  the token like a password to your account, since the SSM/IAM layer alone
  does not distinguish cluster users.
- **Multiple clusters in one account:** the `Name=*login` tag filter in the
  connect snippet matches the login node of *every* PCS cluster that follows
  the naming convention — add a `"Name=tag:ClusterName,Values=<your-stack>"`
  filter to pin it when you run more than one.
- **Alternative when `SSHAccessCidr` is set:** a plain SSH tunnel also works —
  `ssh -L 8888:<compute-node-ip>:<port> <user>@<login-public-ip>` — useful for
  users who cannot install the Session Manager plugin.
- **Do not run Jupyter on the login node.** It has no GPUs, and it is shared
  by every user (and hosts the monitoring stack).
- **Scripts on `/fsx` can't be exec'd directly** (Lustre blocks `execve` on
  some paths) — keep the sbatch file under `$HOME`, or invoke via
  `bash /fsx/script.sh`.
