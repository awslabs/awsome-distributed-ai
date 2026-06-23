# NCCL Tests on P6e-GB200 (NVL72) — Two-Tier Benchmark

This extends the standard [nccl-tests](./README.md) suite for **GB200 / P6e-GB200 UltraServers**. It exists because GB200 collective performance on AWS has a structure that no single all-reduce number captures, and the structure is the whole point: the answer splits at the NVLink-domain boundary.

## The GB200-on-AWS collective story (why this is different)

A P6e-GB200 UltraServer federates 18 `p6e-gb200.36xlarge` instances (4 GB200 GPUs each) into **one 72-GPU multi-node NVLink domain** (the NVL36x2 wiring). That boundary is everything:

- **Inside one UltraServer (≤72 GPUs):** NCCL uses **NVLS** — NVLink SHARP reduction performed in-fabric by the NVSwitch ASIC. This is identical to a reference NVL72 system; the SHARP-style reduction is fully preserved on AWS because it lives in the NVSwitch fabric, not the EC2 network.
- **Across UltraServers (>72 GPUs):** traffic falls onto **EFA**, which has **no in-network SHARP**. `aws-ofi-nccl` implements only NCCL's point-to-point transport and exports no CollNet symbol, so `NCCL_COLLNET_ENABLE=1` and `NCCL_ALGO=Collnet*` are **silent no-ops** on EFA. The algorithm that crosses the boundary efficiently is **NVLSTree**: NVLink SHARP within each domain, then a Tree reduction between domains over EFA.

The practical recovery is architectural, not a knob: keep tensor- and expert-parallel shapes **inside** the 72-GPU domain where NVLS reduces in-fabric; for larger jobs rely on NVLSTree plus the AWS tuner — never `NCCL_COLLNET_ENABLE`.

## Canonical environment block

[`gb200-env.sh`](./gb200-env.sh) is the single source of truth for the GB200 NCCL+EFA environment — **source it from every GB200 workload** in this repo (training, inference, microbenchmarks) rather than re-deriving env vars. It deliberately sets only five variables (`FI_PROVIDER`, `FI_EFA_FORK_SAFE`, `NCCL_DEBUG`, `NCCL_NVLS_ENABLE=1`, `NCCL_MNNVL_ENABLE=1`) and leaves `NCCL_PROTO`/`NCCL_ALGO`/`NCCL_BUFFSIZE` to the bundled aws-ofi-nccl platform tuner. Hand-setting protocol/algorithm from InfiniBand-era guides turns that tuner off and surrenders per-message-size selection.

## What's here

| File | Purpose |
|---|---|
| `gb200-env.sh` | Canonical NCCL+EFA env block (sourced by all GB200 samples) |
| `kubernetes/nccl-tests-gb200.yaml` | **Intra-domain** run (existing): one ComputeDomain, `np=72`, one clique |
| `kubernetes/nccl-tests-gb200-multidomain.yaml` | **Cross-domain** run: two ComputeDomains, `np=144`, two cliques over EFA |
| `slurm/nccl-tests-gb200.sbatch` | Slurm run; `--nodes 18` intra (72 GPU) or `--nodes 36` cross (144 GPU) |
| `gb200-sweep.sh` | Two-tier sweep: intra (NVLS on/off) and cross (tuner vs NVLSTree/Ring/Tree) across AllReduce/AllGather/ReduceScatter/AllToAll |

## Running it

**Intra-domain (runnable on one UltraServer):**
```bash
# Kubernetes (EKS / HyperPod-EKS, K8s >= 1.33, NVIDIA DRA driver)
kubectl apply -f kubernetes/nccl-tests-gb200.yaml
# Slurm
sbatch slurm/nccl-tests-gb200.sbatch                 # --nodes=18
# Sweep (inside the launcher container)
SCENARIO=intra ./gb200-sweep.sh
```

**Cross-domain (requires two UltraServers — authored-to-spec):**
```bash
kubectl apply -f kubernetes/nccl-tests-gb200-multidomain.yaml
sbatch --nodes=36 slurm/nccl-tests-gb200.sbatch
SCENARIO=cross ./gb200-sweep.sh
```

The sweep prints the standard nccl-tests table; pipe the `busbw` column through [`nccl_to_csv.py`](./nccl_to_csv.py) for plotting. Every run uses `-c 1`, so the **`#wrong` column must be `0`** — a nonzero value indicates silent cross-domain NVLink corruption and must block the run (see `4.validation_and_observability` for the hard gate).

## Testability

| Path | Status |
|---|---|
| Intra-domain (≤72 GPU, NVLS) | **Runnable** on one `u-p6e-gb200x72` Capacity Block |
| Cross-domain (144 GPU over EFA, NVLSTree) | **Authored-to-spec** — needs two UltraServers / two cliques held simultaneously; AWS publishes no all-reduce busbw beyond a single domain |

## Version pins (P6e-GB200, 2026-06)

CUDA 13.0.2 · NCCL 2.30.4-1 · aws-ofi-nccl ~1.19.0 (bundled in EFA installer 1.48.0) · libfabric ≥ 1.22.0 · nccl-tests v2.18.3 · gdrcopy 2.5.2 · arm64 (Grace), device code for `sm_100` (GB200) and `sm_103` (GB300). GB200 MNNVL requires NCCL ≥ 2.25.2.
