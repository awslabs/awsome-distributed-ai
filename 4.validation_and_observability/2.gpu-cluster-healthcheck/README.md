# GPU Cluster Health Check Suite

A comprehensive, open-source health check suite for self-managed GPU clusters on AWS, supporting both Slurm and Kubernetes (EKS). This suite leverages publicly available tools (DCGM, NCCL tests, EFA utilities, nvidia-smi) to validate GPU, network, and interconnect health.

The suite provides two operational modes: **lightweight checks** for regular use (prolog/epilog, cron) and **intensive checks** for quarantine and post-mortem scenarios. It follows a **replace, don't repair** operational model -- on AWS, the primary remediation for confirmed hardware faults is instance replacement rather than on-node repair.

## Quick Start

### Prerequisites

| Tool | Required For | Installation |
|------|-------------|-------------|
| NVIDIA Driver | All GPU checks | Pre-installed on GPU AMIs |
| [DCGM Toolkit](https://developer.nvidia.com/dcgm) | Checks 1, 4 | `apt install datacenter-gpu-manager` or via NVIDIA repo |
| EFA Installer | Checks 2, 6 | [AWS EFA Installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html) |
| NCCL Tests Container | Check 5 | `docker://public.ecr.aws/hpc-cloud/nccl-tests:latest` |
| [Pyxis](https://github.com/NVIDIA/pyxis) + [Enroot](https://github.com/NVIDIA/enroot) | Check 5 (container) | Optional -- falls back to local `all_reduce_perf`. Pyxis < v0.20 requires the explicit `docker://` scheme on the image reference; newer Pyxis auto-detects |
| Python 3.6+ | Result parsing | Pre-installed on most Linux distributions |

### Installation

```bash
# Clone the repository
git clone https://github.com/awslabs/awsome-distributed-training.git
cd awsome-distributed-training/4.validation_and_observability/2.gpu-cluster-healthcheck

# Make all scripts executable
chmod +x gpu-healthcheck.sh checks/*.sh slurm/*.sh slurm/examples/*.sh

# Run a quick validation (dry-run mode, no GPU required)
./gpu-healthcheck.sh --suite lightweight --dry-run
```

### First Check

```bash
# Run lightweight suite on a GPU node
./gpu-healthcheck.sh --suite lightweight

# Run a single check
./gpu-healthcheck.sh --check nvidia-smi

# View results
cat /tmp/gpu-healthcheck-*/summary.json | python3 -m json.tool
```

## Architecture

### Directory Structure

```
2.gpu-cluster-healthcheck/
├── README.md                          # This documentation
├── gpu-healthcheck.sh                 # Master orchestrator
├── instance-profiles.conf             # Per-instance-type hardware expectations
├── lib/
│   ├── common.sh                      # Shared utilities (logging, detection, formatting)
│   ├── parse-dcgm-results.py          # DCGM JSON → severity classification
│   └── aggregate-results.py           # Per-node → cluster summary aggregation
├── checks/
│   ├── 0-nvidia-smi-check.sh          # Quick nvidia-smi validation (~5s)
│   ├── 1-dcgm-diag-l2.sh             # DCGM Level 2 diagnostics (2.5-10.5 min)
│   ├── 2-efa-enumeration.sh           # EFA device + provider validation (~3s)
│   ├── 3-topology-check.sh            # GPU/NVLink/PCIe topology (~10s)
│   ├── 4-dcgm-diag-l4.sh             # DCGM Level 4 diagnostics (45 min-2.25 hr)
│   ├── 5-nccl-allreduce.sh           # Multi-node NCCL all_reduce (10-20 min)
│   └── 6-efa-loopback.sh             # Per-device EFA loopback (5-15 min)
├── slurm/
│   ├── prolog-gpu-healthcheck.sh      # Slurm prolog (checks 0-2)
│   ├── sbatch-lightweight.sh          # sbatch job for lightweight suite
│   ├── sbatch-intensive.sh            # sbatch job for intensive suite
│   ├── sbatch-quarantine-workflow.sh  # Full quarantine decision workflow
│   └── examples/
│       ├── cron-rolling-sweep.sh      # Periodic sweep across idle nodes
│       └── slurm-epilog-example.sh    # Epilog with exit-code routing
├── kubernetes/
│   ├── README.md                      # Kubernetes deployment documentation
│   ├── Dockerfile                     # Container image build
│   ├── agent.sh                       # DaemonSet agent entrypoint
│   ├── sweeper.sh                     # CronJob sweeper entrypoint
│   ├── determine-severity.py          # Single-node severity aggregator
│   └── manifests/
│       ├── 00-namespace.yaml          # gpu-healthcheck namespace
│       ├── 01-configmap.yaml          # Instance profiles + agent config
│       ├── 02-rbac.yaml               # ServiceAccount, ClusterRole, Binding
│       ├── 03-daemonset-agent.yaml    # Lightweight check agent
│       ├── 04-cronjob-sweeper.yaml    # Rolling DCGM L2 sweep
│       └── 05-job-quarantine.yaml     # Intensive check template
```

### Check Levels

The suite organizes checks into two suites based on operational impact and runtime:

| Suite | Checks | Runtime | Use Case | Node Access |
|-------|--------|---------|----------|-------------|
| **Lightweight** | 0-3 | ~15 minutes | Prolog/epilog, periodic sweeps, triage | Shared OK |
| **Intensive** | 4-6 | 1-3 hours | Quarantine, post-mortem, deep diagnostics | Exclusive required |

### Severity Classification

All check results are classified into four severity levels that map directly to operational actions:

| Severity | Meaning | Action |
|----------|---------|--------|
| **ISOLATE** | Critical hardware fault confirmed | Drain node, initiate instance replacement |
| **REBOOT** | Node reboot required to clear error | Drain/cordon node, reboot, re-test |
| **RESET** | GPU reset may resolve the issue | Attempt GPU reset (`nvidia-smi --gpu-reset`); may require power-cycle |
| **MONITOR** | Minor anomaly, not service-affecting | Keep in service, flag for review |

## Instance Profiles

The file `instance-profiles.conf` defines expected hardware counts per instance type. Check scripts auto-detect the instance type via IMDS and look up the expected configuration.

| Instance Type | GPUs | EFA Devices | NVLink | Provider |
|--------------|------|-------------|--------|----------|
| p4d.24xlarge | 8 | 4 | Yes | efa |
| p4de.24xlarge | 8 | 4 | Yes | efa |
| p5.48xlarge | 8 | 32 | Yes | efa |
| p5e.48xlarge | 8 | 32 | Yes | efa |
| p5en.48xlarge | 8 | 16 | Yes | efa |
| p6-b200.48xlarge | 8 | 8 | Yes | efa |

To add a new instance type, append a line to `instance-profiles.conf`:
```
<instance-type>|<gpu-count>|<efa-count>|<nvlink-expected>|<efa-provider>
```

## Check Reference

### Check 0: nvidia-smi Validation

**Runtime:** ~5 seconds | **Suite:** Lightweight

Verifies basic GPU driver functionality:
- Confirms `nvidia-smi` executes and returns zero exit code
- Counts detected GPUs and compares against the instance profile
- Scans kernel log (`journalctl -k` with `dmesg` fallback) for Xid errors with severity-aware classification aligned with [NVIDIA XID Errors r590](https://docs.nvidia.com/deploy/pdf/XID_Errors.pdf):

| Group | Xid Codes | Severity | NVIDIA r590 Action | Rationale |
|---|---|---|---|---|
| CHECK_MECHANICALS | 54 | ISOLATE | CHECK_MECHANICALS | Hardware / mechanical fault detected |
| RESTART_BM | 79 | REBOOT | RESTART_BM | GPU fallen off bus; bare-metal restart required |
| RESTART_VM | 151 | REBOOT | RESTART_VM | VM restart required |
| BOOT_REATTEMPT | 168 | REBOOT | BOOT_REATTEMPT_OR_ENABLE_ECC | Boot reattempt or enable ECC |
| WORKFLOW_XID_48 | 48 | RESET | WORKFLOW_XID_48 | Workflow-driven; Xid 154 parsing may escalate |
| DRAM_RETIREMENT | 64 | RESET (A100→REBOOT) | RESET_GPU | DRAM retirement failure; A100 needs reboot |
| RESET_GPU | 95 | RESET (A100+MIG_off→REBOOT) | RESET_GPU | GPU reset required; A100 w/o MIG needs reboot |
| RESET_GPU | 109, 110, 119, 120, 136, 140, 143, 155, 156, 158 | RESET | RESET_GPU | GPU reset or node reboot clears the error |
| WORKFLOW_NVLINK | 74 | `NVLINK_DEFAULT` | WORKFLOW_NVLINK_ERR | NVLink error; configurable (default: RESET) |
| WORKFLOW_NVLINK5 | 144-150 | `NVLINK5_DEFAULT` | WORKFLOW_NVLINK5_ERR | Blackwell NVLink5; configurable (default: MONITOR) |
| CHECK_UVM | 159 | RESET if UVM in use, else MONITOR | CHECK_UVM | C2C/CHI UVM error; conditional |
| PSHC_INFO | 162, 163 | MONITOR | PSHC_INFO | PSHC informational |
| PSHC_LOW_LIFETIME | 164 | MONITOR | PSHC_LOW_LIFETIME | PSHC low lifetime warning |
| PSHC_ZERO_LIFETIME | 165 | MONITOR (`STRICT_PSHC=1`→ISOLATE) | PSHC_ZERO_LIFETIME | PSHC zero lifetime; strict mode isolates |
| RESTART_APP | 13, 31, 94, 126 | MONITOR | RESTART_APP | Application-level fault; restart job, not node |
| IGNORE | 43, 63, 92, 121 | MONITOR | IGNORE | Informational or normal operation |
| XID_154_INFO | 154 | MONITOR (+ derived action parsing) | Informational | Meta-Xid; derived action text may escalate to REBOOT or RESET |
| CONTACT_SUPPORT | 81, 125, 157 | MONITOR | CONTACT_SUPPORT | Deprecated/unused; escalate to humans if seen |

- **Xid 154 derived action parsing**: Xid 154 lines carry authoritative recovery action text. The check parses for "Node Reboot Required" (→REBOOT), "GPU Reset Required" / "Drain and Reset" / "Drain P2P" (→RESET). Unrecognized or "(None)" stays MONITOR.
- **Tunables**: `KERNEL_LOG_LINES` (default: 4000), `NVIDIA_LOG_TAIL` (default: 200), `STRICT_PSHC` (default: 0), `NVLINK5_DEFAULT` (default: MONITOR), `NVLINK_DEFAULT` (default: RESET). NVLink tunables accept MONITOR, RESET, or REBOOT.
- Scans kernel log for SXid (NVSwitch) errors; SXid alongside Xid 74 indicates NVSwitch root cause
- Checks GPU persistence mode status
- Captures GPU serial numbers and UUIDs to `gpu-uuids.csv` for asset tracking

```bash
./gpu-healthcheck.sh --check 0
# or
./gpu-healthcheck.sh --check nvidia-smi
```

### Check 1: DCGM Level 2 Diagnostics

**Runtime:** 2.5 - 10.5 minutes | **Suite:** Lightweight

Runs medium-depth NVIDIA Data Center GPU Manager diagnostics:
- Deployment readiness validation
- PCIe bandwidth verification
- GPU memory stress test (short)
- SM stress test (short)

Pre-flight: Verifies `nv-hostengine` is running and starts it if needed, with awareness of systemd-managed DCGM services.

```bash
./gpu-healthcheck.sh --check 1
# or
./gpu-healthcheck.sh --check dcgm-l2

# Override timeout (default: 15 min)
DCGM_TIMEOUT=600 ./gpu-healthcheck.sh --check dcgm-l2
```

### Check 2: EFA Enumeration

**Runtime:** ~3 seconds | **Suite:** Lightweight

Validates EFA network infrastructure:
- Counts EFA PCI devices via `lspci` and compares against instance profile
  - On `p5en` instances, retries up to 3 times with 5-second intervals for known late EFA initialization
- Lists RDMA devices via `ibv_devices`
- Verifies libfabric EFA provider via `fi_info -p efa`
- Checks `/dev/infiniband/uverbs*` device node presence
- Validates required kernel modules: `efa`, `ib_uverbs`, `ib_core`
  - Checks for `gdrdrv` module when GDRCopy is installed (`/opt/gdrcopy`)
- Runs GDRCopy sanity check (`/opt/gdrcopy/bin/sanity -v`) if available
- Checks memory lock limits (`ulimit -l`) — warns if below 16 GiB

```bash
./gpu-healthcheck.sh --check 2
# or
./gpu-healthcheck.sh --check efa-enumeration
```

### Check 3: Topology Validation

**Runtime:** ~10 seconds | **Suite:** Lightweight

Validates GPU interconnect topology:
- Captures `nvidia-smi topo -m` matrix and checks for disconnected links
- Validates NVLink presence and status per GPU pair
- Checks `nvidia-fabricmanager` service status (RESET severity if not running)
- Queries NVLink error counters (`nvidia-smi nvlink -e`) and warns on non-zero replay, recovery, or CRC errors
- Verifies PCIe switch groupings
- On B200 instances: additional `nvidia-smi topo -p2p rwn` check

```bash
./gpu-healthcheck.sh --check 3
# or
./gpu-healthcheck.sh --check topology
```

### Check 4: DCGM Level 4 Diagnostics

**Runtime:** 45 minutes - 2.25 hours | **Suite:** Intensive | **Requires:** Exclusive access

The most comprehensive GPU diagnostic available. Includes everything in L2 plus Extended Utility Diagnostics (EUD) and pulse power testing.

**Pre-flight requirements** (all validated automatically):
1. Node must be exclusively allocated (no other GPU processes)
2. MIG must be disabled
3. Concurrent GPU telemetry services are stopped for the duration
4. `nv-hostengine` must be running

**Operational warnings:**
- Pulse power test draws variable power -- inform facility operators
- EUD cannot run with MIG enabled
- Do not run concurrent GPU monitoring during the test
- systemd may auto-restart DCGM services mid-test

```bash
./gpu-healthcheck.sh --check 4 --exclusive
# or
./gpu-healthcheck.sh --check dcgm-l4 --exclusive
```

### Check 5: Multi-Node NCCL all_reduce

**Runtime:** 10-20 minutes | **Suite:** Intensive | **Requires:** Minimum 2 nodes

Runs `all_reduce_perf` from the NCCL tests container to validate multi-node GPU communication over EFA:
- Message sizes: 8B to 128MB (power-of-2 sweep)
- Verifies EFA provider selection via `NCCL_DEBUG=INFO`
- Compares measured bus bandwidth against per-instance-type thresholds

**Optional isolation sub-tests** (`NCCL_ISOLATION_TESTS=1`):
- **NVLink-only test:** Forces `NCCL_P2P_LEVEL=NVL NCCL_NET=Socket` to isolate intra-node NVLink performance. Thresholds: p4d=200 GB/s, p5/p5e/p5en=500 GB/s, p6-b200=600 GB/s
- **EFA-only test:** Forces `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET='AWS Libfabric'` to isolate inter-node EFA performance (requires >= 2 nodes)
- Both sub-tests produce MONITOR-level warnings only — the full-stack test remains the authoritative pass/fail
- Each sub-test has its own timeout: `NCCL_ISOLATION_TIMEOUT` (default: 600s / 10 min)

Environment variables set automatically:
```bash
FI_PROVIDER=efa
FI_EFA_USE_DEVICE_RDMA=1
NCCL_NET_GDR_LEVEL=2
NCCL_DEBUG=INFO
```

```bash
# Must be run within a multi-node Slurm allocation
salloc -N 2 --exclusive
./gpu-healthcheck.sh --check 5

# With isolation sub-tests enabled
NCCL_ISOLATION_TESTS=1 ./gpu-healthcheck.sh --check 5
```

### Check 6: EFA Loopback Bandwidth/Latency

**Runtime:** 5-15 minutes | **Suite:** Intensive

Tests each EFA device individually:
- Iterates over all RDMA devices discovered via `ibv_devices`
- Runs `fi_pingpong` or `ib_write_bw` in loopback mode per device
- Reports bandwidth and latency per device
- Compares per-device bandwidth against instance-type thresholds (default: 20 Gbps for all supported types)
  - Override with `EFA_MIN_BW` env var (in Gbps)
- Collects EFA statistics via `rdma -p statistic show` and warns on `rx_drops` or `retrans_timeout_events`
  - Statistics saved to `efa-statistics.txt`

```bash
./gpu-healthcheck.sh --check 6
# or
./gpu-healthcheck.sh --check efa-loopback

# With custom bandwidth threshold (Gbps)
EFA_MIN_BW=25 ./gpu-healthcheck.sh --check 6
```

## DCGM Operational Guide

### Level 2 vs Level 4 Positioning

| Aspect | Level 2 (`dcgmi diag -r 2`) | Level 4 (`dcgmi diag -r 4`) |
|--------|------|------|
| **Purpose** | Production gate / fast triage | Deep post-mortem / quarantine |
| **Runtime** | 2.5 - 10.5 min | 45 min - 2.25 hr |
| **When to run** | Prolog, epilog, periodic sweep | Node drained, exclusive access only |
| **EUD included** | No | Yes (~20 min, requires MIG disabled) |
| **Pulse test** | No | Yes (variable power draw) |
| **Safe for production** | Yes (with timeout guard) | No -- requires exclusive node access |
| **hostengine** | Must be running | Must be running; beware systemd auto-restart |

### Pre-flight Checklist (L4)

Before running DCGM Level 4:

- [ ] Node is drained from the scheduler (no new jobs/pods will be placed)
- [ ] No other GPU processes running (`nvidia-smi --query-compute-apps=pid --format=csv`)
- [ ] MIG is disabled (`nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader`)
- [ ] DCGM exporter stopped (`systemctl stop dcgm-exporter` or container stopped)
- [ ] `nv-hostengine` running (check 4 handles this automatically)
- [ ] Facility operators aware of variable power draw during pulse test

### Severity Classification from DCGM Results

DCGM diagnostic results include a `warning_level` field per test per GPU:

| Warning Level | Severity | Action |
|--------------|----------|--------|
| 3 | **ISOLATE** | Drain node, initiate instance replacement |
| 2 | **RESET** | Attempt GPU reset via `nvidia-smi --gpu-reset` |
| 1 | **MONITOR** | Keep in service, flag for review |
| 0 | **PASS** | No action required |

The `parse-dcgm-results.py` script converts raw DCGM JSON into this classification automatically.

### hostengine Management

The DCGM host engine (`nv-hostengine`) must be running for diagnostics. Be aware of these interactions:

- If managed by systemd (`nvidia-dcgm.service`), systemd may auto-restart monitoring services that were stopped for L4
- The check scripts handle starting `nv-hostengine` if it is not running
- For L4, consider: `systemctl stop nvidia-dcgm && nv-hostengine` to avoid systemd interference

### NVIDIA Documentation References

- [DCGM Diagnostics Overview](https://docs.nvidia.com/dcgm/latest/user-guide/dcgm-diagnostics.html)
- [DCGM Run Levels](https://docs.nvidia.com/dcgm/latest/user-guide/dcgm-diagnostics.html#run-levels)
- [DCGM JSON Output](https://docs.nvidia.com/dcgm/latest/user-guide/dcgm-diagnostics.html#json-output)
- [DCGM Policy Management](https://docs.nvidia.com/dcgm/latest/user-guide/dcgm-policy-management.html)

## Slurm Integration

The `slurm/` directory provides Slurm-native integration -- prolog/epilog hooks, sbatch job wrappers, and cron-based sweep automation. See [`slurm/README.md`](slurm/README.md) for setup instructions.

| Script | Purpose |
|--------|---------|
| `prolog-gpu-healthcheck.sh` | Prolog hook -- checks 0 + 2 before every job |
| `sbatch-lightweight.sh` | sbatch wrapper for lightweight suite |
| `sbatch-intensive.sh` | sbatch wrapper for intensive suite |
| `sbatch-quarantine-workflow.sh` | Quarantine decision workflow |
| `examples/cron-rolling-sweep.sh` | Periodic sweep of idle nodes |
| `examples/slurm-epilog-example.sh` | Epilog with exit-code routing |

## Kubernetes (EKS) Support

The `kubernetes/` directory provides a full Kubernetes-native deployment for Amazon EKS clusters. See [`kubernetes/README.md`](kubernetes/README.md) for setup instructions.

| Component | K8s Resource | Purpose |
|-----------|-------------|---------|
| **Agent** | DaemonSet | Continuous lightweight checks on every GPU node |
| **Sweeper** | CronJob | Rolling DCGM L2 sweep on idle nodes |
| **Quarantine** | Job (template) | Intensive diagnostics on suspect nodes |

## Decision Flow

The recommended operational workflow for handling suspected GPU issues:

```
               Monitoring / Job Failure
                       │
                       ▼
              ┌─────────────┐
              │  Drain Node  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Lightweight  │  gpu-healthcheck.sh --suite lightweight
              │ Suite (0-3)  │
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │             │
          PASS ▼         FAIL ▼
     ┌──────────┐    ┌──────────────┐
     │ Return to│    │Check Severity│
     │ Service  │    └──────┬───────┘
     └──────────┘           │
                 ┌──────┬───┴────┬───────┐
                 │      │        │       │
            ISOLATE  REBOOT   RESET   MONITOR
                 │      │        │       │
                 ▼      ▼        ▼       ▼
           ┌────────┐ ┌──────┐ ┌─────┐ ┌────────┐
           │Replace │ │Reboot│ │GPU  │ │Return +│
           │Instance│ │Node  │ │Reset│ │ Flag   │
           └────────┘ └──┬───┘ └──┬──┘ └────────┘
                         │        │
                         ▼        ▼
                     ┌─────────────┐
                     │ Intensive   │  gpu-healthcheck.sh --suite intensive
                     │ Suite (4-6) │
                     └──────┬──────┘
                            │
                     ┌──────┴──────┐
                     │             │
                 PASS ▼         FAIL ▼
            ┌──────────┐    ┌──────────┐
            │ Return to│    │ Replace  │
            │ Service  │    │ Instance │
            └──────────┘    └──────────┘
```

The guiding principle is **replace, don't repair**. On AWS, the primary remediation for confirmed hardware faults is instance replacement. The intensive suite exists to confirm whether a fault is real (warranting replacement) or transient (safe to return to service after reboot).

## Troubleshooting

### nvidia-smi fails to execute

- Verify NVIDIA driver is loaded: `lsmod | grep nvidia`
- Check for driver/kernel mismatch: `dmesg | grep -i nvidia`
- Verify GPU device nodes exist: `ls /dev/nvidia*`

### DCGM diagnostics timeout

- L2 can take up to 10.5 minutes on 8-GPU instances -- set `DCGM_TIMEOUT` accordingly
- L4 can take up to 2.25 hours -- set `DCGM_L4_TIMEOUT` accordingly
- Verify `nv-hostengine` did not crash: `pgrep nv-hostengine`

### nv-hostengine won't start

- Check if another instance is running: `pgrep -a nv-hostengine`
- Check systemd status: `systemctl status nvidia-dcgm`
- Verify DCGM is installed: `dcgmi --version`

### EFA devices not detected

- Verify EFA installer ran successfully: `fi_info -p efa`
- Check kernel modules: `lsmod | grep efa`
- Verify PCI devices: `lspci | grep -i EFA`
- Check for EFA-compatible instance type

### NCCL all_reduce fails

- Verify EFA provider: ensure `NCCL_DEBUG=INFO` output shows "Selected Provider is EFA"
- Check container image availability: `enroot import docker://public.ecr.aws/hpc-cloud/nccl-tests:latest`
- Verify inter-node connectivity: security groups must allow all traffic between nodes
- Check for NCCL version compatibility with driver version

#### Container image pull returns 401 Unauthorized from `registry-1.docker.io`

Symptom in stderr:
```
URL https://registry-1.docker.io/v2/public.ecr.aws/hpc-cloud/nccl-tests/manifests/latest returned error code: 401 Unauthorized
pyxis: failed to import docker image
spank: required plugin spank_pyxis.so: task_init() failed with rc=-1
```

Cause: older Pyxis (< v0.20) does not auto-detect the registry hostname in bare image references and routes the pull through Docker Hub, which returns 401 for the `public.ecr.aws/...` path. Force the registry scheme explicitly:

```bash
NCCL_CONTAINER=docker://public.ecr.aws/hpc-cloud/nccl-tests:latest \
  ./gpu-healthcheck.sh --suite intensive --exclusive
```

Or update Pyxis to v0.20 or newer. The script's default already uses the `docker://` scheme as of [PR #TBD](https://github.com/awslabs/awsome-distributed-ai/pulls); this troubleshooting note covers customers running older Pyxis with the previous default still pinned in their configuration.

### Topology shows disconnected GPUs

- Check NVSwitch status (p5/p5e): `nvidia-smi nvlink --status`
- Verify PCIe ACS is disabled (can interfere with P2P): `setpci -s '*:*' ECAP_ACS+6.w`
- Check for recent Xid errors indicating NVLink failures: `dmesg | grep -i xid`

### Xid errors classified as ISOLATE, REBOOT, or RESET

Xid severity classification is aligned with the [NVIDIA XID Errors r590](https://docs.nvidia.com/deploy/pdf/XID_Errors.pdf) catalog. The four severity tiers are:

- **ISOLATE** (Xid 54; Xid 165 with `STRICT_PSHC=1`): Critical hardware fault. Drain the node and replace the instance.
- **REBOOT** (Xid 79, 151, 168; Xid 64 on A100; Xid 95 on A100 w/o MIG): Node reboot required. Reboot and re-test; escalate to replacement if recurring.
- **RESET** (Xid 48, 109, 110, 119, 120, 136, 140, 143, 155, 156, 158; Xid 64, 95 on non-A100): GPU reset required. Attempt `nvidia-smi --gpu-reset`; may require power-cycle.
- **MONITOR** (Xid 13, 31, 43, 63, 81, 92, 94, 121, 125, 126, 154, 157, 162-164): Application-level fault or informational. No node action required.

Xid 154 carries authoritative recovery action text — the check parses it for "Node Reboot Required" (→REBOOT) or "GPU Reset Required" / "Drain and Reset" / "Drain P2P" (→RESET).

Review kernel log for the specific Xid error and affected GPU:
```bash
journalctl -k | grep -i "NVRM.*Xid"
```

### NVLink error counters show non-zero values

Non-zero replay, recovery, or CRC errors indicate NVLink degradation. Check individual link status:
```bash
nvidia-smi nvlink -e        # Error counters
nvidia-smi nvlink --status  # Link status
```
If errors are persistent across reboots, the node should be replaced.

### GDRCopy sanity check failed

GDRCopy enables GPU memory registration for RDMA. If the sanity check fails:
- Verify the `gdrdrv` kernel module is loaded: `lsmod | grep gdrdrv`
- Check GDRCopy installation: `/opt/gdrcopy/bin/sanity -v`
- NCCL may fall back to slower CPU-bounce-buffer paths without GDRCopy

### MIG prevents DCGM L4

Disable MIG before running L4 diagnostics:
```bash
sudo nvidia-smi -i 0 -mig 0    # Disable MIG on GPU 0
sudo nvidia-smi --gpu-reset     # Reset GPUs (requires no running processes)
```
