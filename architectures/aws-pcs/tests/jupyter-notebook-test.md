# Jupyter Notebook on a Compute Node — Verification Procedure

Verifies that a Jupyter (Lab) server runs on a Slurm compute node as a batch
job, is reachable from a workstation browser without opening inbound ports,
and — on a GPU queue — actually sees its allocated GPU(s) from a notebook
kernel.

> **Portability note:** this procedure is written against a generic Slurm
> cluster and is not tied to the PCS reference templates. Prerequisites are
> stated explicitly below so the test can be relocated to another test set
> (e.g. a repo-level `test_cases` collection) and run on any Slurm cluster
> that meets them.

---

## Prerequisites

- A Slurm cluster with:
  - a login node the tester can reach (SSH, or SSM if on AWS),
  - a CPU partition and a GPU partition (any CUDA GPU; verified on
    `g6.12xlarge`, 4× NVIDIA L4),
  - a **shared `$HOME`** visible from login and compute nodes (NFS or
    equivalent) — the venv, the sbatch script, and the token file live there,
  - GPU partitions configured with `gres/gpu` (check
    `scontrol show node <gpu-node> | grep Gres`).
- On the workstation (only for the browser-access step): AWS CLI + Session
  Manager plugin if using SSM port forwarding, or plain SSH `-L` if the login
  node accepts SSH.
- Internet egress from login and compute nodes (pip installs, first run).

Time: ~15 min on a warm cluster; add node scale-up time (2–3 min per queue,
more if the node runs first-boot provisioning) on a scale-to-zero cluster.

---

## Step 1 — environment setup (once, on the login node)

```bash
python3 -m venv $HOME/jupyter-env
$HOME/jupyter-env/bin/pip install --upgrade pip jupyterlab
$HOME/jupyter-env/bin/pip install torch        # GPU test only
$HOME/jupyter-env/bin/jupyter lab --version
```

**Expected:** a version number prints (verified with 4.6.1). Because `$HOME`
is shared, every compute node sees the same venv.

---

## Step 2 — CPU: Jupyter server as a Slurm job

Save as `$HOME/jupyter.sbatch` (a trimmed variant of the script in
[JUPYTER.md](../docs/JUPYTER.md) — the server-launch logic is the same; **keep
the two in sync** when changing it. Adjust `--partition` to your CPU queue):

```bash
#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --output=%u-jupyter-%j.log

umask 077

# Port range 8000-8999; collisions possible only if two servers share a node
# with job IDs differing by a multiple of 1000
PORT=$((8000 + SLURM_JOB_ID % 1000))
NODE_IP=$(hostname -I | awk '{print $1}')

TOKEN_FILE=$HOME/.jupyter-token-$SLURM_JOB_ID
openssl rand -hex 24 > "$TOKEN_FILE"

echo "Jupyter on $(hostname) ($NODE_IP) port $PORT; token in $TOKEN_FILE"

source $HOME/jupyter-env/bin/activate
exec jupyter lab --no-browser --ip="$NODE_IP" --port="$PORT" \
  --ServerApp.token="$(cat "$TOKEN_FILE")" \
  --notebook-dir="$HOME"
```

```bash
cd $HOME && sbatch jupyter.sbatch
squeue                                    # wait for state R
head -30 $(ls -t $HOME/*-jupyter-*.log | head -1)
ls -l $HOME/.jupyter-token-*
```

**Expected:**

- Job reaches `R` (on scale-to-zero: `CF` for 2–3 min first).
- The log shows the node IP and the job-derived port (e.g. job 10 → port 8010).
- The token file exists with mode `-rw-------` (600) and is **not** echoed in
  the job log.

### 2a. HTTP reachability and token auth (from the login node)

```bash
NODE_IP=<ip from log>; PORT=<port from log>; JOBID=<jobid>
curl -s -o /dev/null -w '%{http_code}\n' http://$NODE_IP:$PORT/api/status   # no token
curl -s -H "Authorization: token $(cat $HOME/.jupyter-token-$JOBID)" \
  http://$NODE_IP:$PORT/api/status
```

**Expected:** `403` without the token; with it, a JSON status body such as
`{"connections": 0, "kernels": 0, "last_activity": "...", "started": "..."}`.

### 2b. Browser path from the workstation

Via SSM (AWS; `LOGIN_ID` = login-node instance ID) — run locally:

```bash
aws ssm start-session --target $LOGIN_ID \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters host=$NODE_IP,portNumber=$PORT,localPortNumber=8888
# other terminal — TOKEN=<value from `cat ~/.jupyter-token-<jobid>` on the login node>:
curl -s -o /dev/null -w '%{http_code}\n' "http://localhost:8888/lab?token=$TOKEN"
```

(Or the SSH equivalent: `ssh -L 8888:$NODE_IP:$PORT <login-node>`.)

**Expected:** `200` from `/lab?token=…`, and the same `/api/status` JSON as 2a
through `localhost:8888`. Opening the URL in a browser shows JupyterLab.

---

## Step 3 — GPU: server on a GPU queue, GPU visible from the kernel

Copy the script to `jupyter-gpu.sbatch` and change only the header:

```bash
#SBATCH --job-name=jupyter-gpu
#SBATCH --partition=gpu-g6         # your GPU queue
#SBATCH --gres=gpu:1
```

Submit and repeat the Step 2 checks (server up, 403/JSON, tunnel). Then verify
GPU visibility **inside the job's allocation** — `--overlap` attaches to the
running Jupyter job without consuming its resources:

```bash
srun --jobid=<gpu-jobid> --overlap nvidia-smi
```

**Expected:** the full GPU table for the node (all physical GPUs listed —
e.g. 4× NVIDIA L4 on g6.12xlarge; allocation limits are enforced per-process
via `CUDA_VISIBLE_DEVICES`, not in `nvidia-smi`).

### 3a. Framework-level GPU check (the actual pass/fail gate)

Save as `$HOME/gpu_check.py`:

```python
import json, os, subprocess
import torch
res = {
    "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "<unset>"),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if res["cuda_available"]:
    res["device_name"] = torch.cuda.get_device_name(0)
    x = torch.rand(1000, 1000, device="cuda")
    res["matmul_sum_positive"] = bool((x @ x).sum().item() > 0)
print(json.dumps(res, indent=2))
```

Create a one-cell notebook that runs the same script (the cell path must
match where `gpu_check.py` was saved):

```bash
cat > $HOME/gpu_check.ipynb <<EOF
{
 "cells": [
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": ["exec(open('$HOME/gpu_check.py').read())"]}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
 "nbformat": 4, "nbformat_minor": 5
}
EOF
```

Run the check two ways — directly in the allocation, and through the notebook
executed by the Jupyter machinery (`jupyter execute` uses the same kernel the
browser would):

```bash
# direct
srun --jobid=<gpu-jobid> --overlap $HOME/jupyter-env/bin/python $HOME/gpu_check.py

# as a notebook (prints nothing on success; output lands in gpu_check_out.ipynb)
srun --jobid=<gpu-jobid> --overlap $HOME/jupyter-env/bin/jupyter execute \
  --output=gpu_check_out.ipynb $HOME/gpu_check.ipynb

# print the executed cell's output
python3 -c "import json; nb=json.load(open('$HOME/gpu_check_out.ipynb')); [print(''.join(o.get('text',[]))) for c in nb['cells'] for o in c.get('outputs',[]) if 'text' in o]"
```

**Expected (both ways, pass criteria in bold):**

```json
{
  "CUDA_VISIBLE_DEVICES": "0",
  "cuda_available": true,          ← must be true
  "device_count": 1,               ← must equal the --gres count
  "device_name": "NVIDIA L4",
  "matmul_sum_positive": true      ← must be true (a real CUDA op ran)
}
```

`CUDA_VISIBLE_DEVICES` matching the `--gres` count (here `0` for `gpu:1` on a
4-GPU node) confirms Slurm's gres enforcement reaches the kernel process; the
matmul confirms compute actually executes on the device, not just enumeration.
(Note: `device_count == --gres count` assumes the cluster constrains devices
per job — `ConstrainDevices=yes` in `cgroup.conf`, the PCS default. On a
cluster without device cgroups the kernel sees every GPU on the node.)

---

## Step 4 — cleanup

```bash
scancel <cpu-jobid> <gpu-jobid>
rm $HOME/.jupyter-token-*
```

**Expected:** `squeue` empties; on scale-to-zero clusters the nodes terminate
after the idle timeout.

---

## Verified configurations

All runs: AWS PCS, Slurm 25.11, us-east-2, scale-to-zero queues; JupyterLab
4.6.1, torch 2.12.1+cu130, driver 595.71.05 / CUDA 13.2. Access via SSM
port-forward through the login node (no inbound ports). Every run passed the
same gates — server up as a Slurm job, token file mode 600, `403` without /
JSON with token, `/lab` = `200` through the tunnel, and (GPU) the kernel
seeing exactly the `--gres` allocation from both direct `srun` and a
`jupyter execute` notebook run.

| Date | Cluster | Queue / instance | GPU check |
|---|---|---|---|
| 2026-07-07 | existing dev cluster | `cpu1` (c-family) | — |
| 2026-07-07 | existing dev cluster | `gpu-g6` g6.12xlarge (4× L4), `--gres=gpu:1` | `device_count=1`, `CUDA_VISIBLE_DEVICES=0`, matmul OK |
| 2026-07-08 | **fresh `deploy-all`** (E2E gate also passed: 6 monitoring containers, Pyxis container job) | `cpu1` c6i.4xlarge, scale-from-zero | — |
| 2026-07-08 | same fresh cluster | `gpu-g6` g6.12xlarge, first-boot node | identical to 07-07 GPU run |
| 2026-07-08 | **fresh `deploy-all` + P5 CNG on a Capacity Block** (GPU E2E also passed: Pyxis CUDA container `nvidia-smi`, DCGM exporter scraping all 8 GPUs) | `gpu-p5` p5.48xlarge (8× H100, 32 NICs), `--gres=gpu:8` | `device_count=8`, `CUDA_VISIBLE_DEVICES=0,…,7`, matmul OK; multi-NIC first-IP binding worked unchanged |
| 2026-07-09 | fresh `deploy-all` cluster — **clean-room pass**: every command block in this file and JUPYTER.md executed verbatim in its documented role (login node / workstation), including the `Name=*login` login-node discovery, the workstation `$TOKEN` curl, and the notebook-creation heredoc | `cpu1` + `gpu-g6` g6.12xlarge, `--gres=gpu:1` | identical results; notebook run output extracted with the documented print command |
