# HPC EFA Tests (Test 9)

Validates EFA on CPU HPC instances (hpc6a/hpc7a/hpc8a) including placement
group auto-creation, multi-NIC wiring, and OSU MPI benchmark results.

---

## Test 9: EFA on CPU HPC instances (hpc6a / hpc7a / hpc8a)

**When to run:** when `add-cng.yaml`'s EFA wiring (`EfaInterfaceCount`,
`PlacementGroupName`) or the deploy-all forwarding params
(`OnDemandEfaInterfaceCount`, `OnDemandPlacementGroupName`) change. Skip if only
GPU/monitoring/AMI paths were touched.

> **EFA is enabled by the interface count.** `OnDemandEfaInterfaceCount=0`
> (default) = no EFA; `1` or `2` enables EFA with that many NICs. There is no
> separate `OnDemandEnableEfa` flag (removed — the count alone drives it).

This validates that the on-demand CPU CNG actually launches with EFA NICs in a
cluster placement group, and that MPI / libfabric over EFA works end-to-end.
Verified bandwidth numbers are recorded in
[tests/README.md](./README.md#major-update-pr--configurations-run-end-to-end-on-real-hardware);
this section documents the **how-to** so a contributor can reproduce.

### Step 1 — deploy with EFA on the CPU CNG

```bash
# hpc7a / hpc8a have 2 EFA NICs; hpc6a has 1.
INSTANCE_TYPE=hpc7a.96xlarge
EFA_NICS=2

# AZ availability is region-specific — confirm with describe-instance-type-offerings:
# hpc7a is in us-east-2b (and others); hpc8a is in us-east-2b / eu-north-1 / ap-northeast-1;
# hpc6a is in us-east-2 (b) / us-west-2 / eu-west-1 etc. AZ MUST contain the type.
AWS_AZ=us-east-2b

aws cloudformation create-stack \
  --stack-name pcs-hpc-efa \
  --region us-east-2 \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/aws-pcs/pcs-ml-cluster-deploy-all.yaml \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=$AWS_AZ \
    ParameterKey=DeployOnDemandCNG,ParameterValue=true \
    ParameterKey=OnDemandInstanceType,ParameterValue=$INSTANCE_TYPE \
    ParameterKey=OnDemandCngName,ParameterValue=hpc \
    ParameterKey=OnDemandQueueName,ParameterValue=hpc \
    ParameterKey=OnDemandMinCount,ParameterValue=0 \
    ParameterKey=OnDemandMaxCount,ParameterValue=2 \
    ParameterKey=OnDemandEfaInterfaceCount,ParameterValue=$EFA_NICS \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

The `OnDemandCNGStack` nested stack auto-creates a cluster placement group and
exposes the name as a stack output (`PlacementGroupName`). To share the PG across
multiple CNGs (heterogeneous tightly-coupled jobs), pass
`OnDemandPlacementGroupName=<existing-pg-name>` instead.

### Step 2 — verify EFA visibility on a compute node

After the stack reaches CREATE_COMPLETE and an `srun` has woken up a node:

```bash
# On a compute node (via Slurm srun from login):
srun -p hpc -N 1 -n 1 bash -c '
  /opt/amazon/efa/bin/fi_info -p efa | head -20  # provider=efa, FI_EP_RDM
  lspci | grep -iE "EFA|Elastic"                 # 2 EFA + 2 ENA on hpc7a/hpc8a
'
```

Expected (hpc7a / hpc8a):
- `fi_info -p efa`: shows `efa-direct` and `efa` fabrics on `rdmap36s0` and
  `rdmap42s0` (or matching device names).
- `lspci`: 2 lines `Elastic Fabric Adapter` (or `Device efa3` on hpc8a) + 2 lines
  `Elastic Network Adapter`.

### Step 3 — OSU MPI micro-benchmarks

Build OSU 7.4 on `/fsx` once (shared across compute nodes) using the
PCS-Ready DLAMI's `/opt/amazon/openmpi`:

```bash
mkdir -p /fsx/osu && cd /fsx/osu
curl -fL -o osu.tgz https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz
tar xf osu.tgz && cd osu-micro-benchmarks-7.4
PATH=/opt/amazon/openmpi/bin:$PATH ./configure CC=mpicc CXX=mpicxx --prefix=/fsx/osu
PATH=/opt/amazon/openmpi/bin:$PATH make -j8
```

Submit a 2-node sbatch with the AWS-tuned EFA env:

```bash
cat > /fsx/osu/osu-bench.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=osu-efa
#SBATCH --partition=hpc
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --output=/fsx/osu/logs/%x_%j.out

set -ex
OSU=/fsx/osu/osu-micro-benchmarks-7.4
export PATH=/opt/amazon/openmpi/bin:$PATH

export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1                # huge page stays at the default (=1, on)
export OMPI_MCA_pml=cm                    # AWS Open MPI 4.1.7 has no pml=ofi
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_provider_include=efa
export OMPI_MCA_btl=^openib,tcp

MPI_X="-x FI_PROVIDER -x FI_EFA_FORK_SAFE \
       -x OMPI_MCA_pml -x OMPI_MCA_mtl -x OMPI_MCA_mtl_ofi_provider_include \
       -x OMPI_MCA_btl"

mpirun -np 2  -N 1  $MPI_X $OSU/c/mpi/pt2pt/standard/osu_latency
mpirun -np 2  -N 1  $MPI_X $OSU/c/mpi/pt2pt/standard/osu_bw
mpirun -np 2  -N 1  $MPI_X $OSU/c/mpi/pt2pt/standard/osu_bibw
mpirun -np 32 -N 16 $MPI_X $OSU/c/mpi/pt2pt/standard/osu_mbw_mr
mpirun -np 32 -N 16 $MPI_X $OSU/c/mpi/collective/blocking/osu_allreduce
EOF

mkdir -p /fsx/osu/logs
sbatch -p hpc /fsx/osu/osu-bench.sbatch
```

Reference numbers (hpc8a, 2 nodes) are in
[tests/README.md](./README.md#major-update-pr--configurations-run-end-to-end-on-real-hardware).

### Step 4 — observe NIC-level traffic in Grafana (optional)

The monitoring stack's Compute Node Details dashboard has dedicated EFA panels
sourced from the `node_amazonefa_*` metrics produced by the v2.7+
`efa-metrics.sh` textfile collector:

- RDMA Read / Write Throughput
- SRD Retransmitted Packets
- Work-Request Errors

For per-NIC `tx_bytes` / `rx_bytes` rate during a benchmark, query Prometheus
directly:

```promql
rate(node_amazonefa_tx_bytes[30s]) * 8 / 1e9   # Gbps per (instance, device)
sum by (instance) (rate(node_amazonefa_tx_bytes[30s])) * 8 / 1e9  # both NICs
```

(The textfile collector cadence is 30s; rates over windows shorter than that are
zeros most of the time. OSU sub-tests are also short — 10–30s each — so wall-clock
peak in Prometheus typically reads below the OSU-reported peak.)

### Step 5 — clean up

When done with EFA testing, just delete the CNG stack (or the whole deploy-all
stack). The auto-created cluster placement group is owned by the CNG stack and
is removed automatically. Slurm puts the EFA compute nodes to sleep on idle
(`SuspendTime`), so leaving the cluster up between benchmarks does not keep the
hpc7a/hpc8a instances running.

---
