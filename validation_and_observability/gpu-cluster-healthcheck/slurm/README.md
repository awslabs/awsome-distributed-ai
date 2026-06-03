# GPU Health Check Suite -- Slurm Integration

Slurm-native integration for the GPU cluster health check suite, including prolog/epilog hooks, sbatch job wrappers, and cron-based sweep automation.

## Components

| Script | Purpose |
|--------|---------|
| `prolog-gpu-healthcheck.sh` | Slurm prolog -- runs checks 0 + 2 before every job |
| `sbatch-lightweight.sh` | sbatch wrapper for the lightweight suite (checks 0-3) |
| `sbatch-intensive.sh` | sbatch wrapper for the intensive suite (checks 4-6) |
| `sbatch-quarantine-workflow.sh` | Full quarantine decision workflow |
| `examples/cron-rolling-sweep.sh` | Periodic sweep of idle GPU nodes |
| `examples/slurm-epilog-example.sh` | Epilog with exit-code routing |

### Slurm-to-Kubernetes Concept Mapping

| Slurm | Kubernetes |
|-------|------------|
| Prolog/epilog | DaemonSet agent (continuous) |
| `scontrol drain` | Taint `gpu-healthcheck.aws-samples.io/unhealthy=true:NoSchedule` |
| Cron rolling sweep | CronJob sweeper |
| `sbatch --exclusive` quarantine | Quarantine Job on cordoned node |
| Node state | Node labels (`gpu-healthcheck.aws-samples.io/status`) |

## Prolog Setup

Add to `slurm.conf`:

```conf
Prolog=/path/to/2.gpu-cluster-healthcheck/slurm/prolog-gpu-healthcheck.sh
PrologTimeout=900   # 15 minutes
```

By default, the prolog runs only checks 0 (nvidia-smi) and 2 (EFA enumeration) -- completing in ~8 seconds. DCGM L2 is **off by default** in prolog because it adds minutes of silence before job output appears.

> **Note:** Prolog output goes to syslog / slurmd logs, not to job output files.

To enable DCGM L2 in the prolog (adds 2-10 minutes):

```conf
PrologFlags=Contain
PrologSlurmctld=
# In the prolog environment:
# GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1
```

Or set globally in `/etc/default/gpu-healthcheck`:
```bash
GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1
```

A non-zero exit code from the prolog causes Slurm to drain the node and requeue the job.

## sbatch Examples

```bash
# Lightweight suite across 4 nodes
sbatch -N 4 -p gpu slurm/sbatch-lightweight.sh

# Intensive suite on 2 nodes (exclusive)
sbatch -N 2 -p maintenance --exclusive slurm/sbatch-intensive.sh

# Quarantine workflow on a suspect node
sbatch -N 1 -w suspect-node-001 --exclusive slurm/sbatch-quarantine-workflow.sh
```

## Cron Sweep

Set up a periodic sweep of idle GPU nodes:

```bash
# Check up to 10 idle nodes every 4 hours
echo "0 */4 * * * /path/to/slurm/examples/cron-rolling-sweep.sh >> /var/log/gpu-sweep.log 2>&1" | crontab -
```

## Epilog Pattern

The `examples/slurm-epilog-example.sh` demonstrates exit-code routing after job completion:
- Normal exit (0): quick nvidia-smi check only
- Signal kills (137, 139): full prolog-level check
- Any nvidia-smi failure: immediate node drain
