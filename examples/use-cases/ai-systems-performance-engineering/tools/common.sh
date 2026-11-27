# shellcheck shell=bash
# Sourced by every AIM347 tool: the event environment and the Slurm options for both GPU nodes.
AIM347_ENV=${AIM347_ENV:-/fsx/aim347/participant/aim347.env}
export AIM347_ENV
if [[ ! -f $AIM347_ENV ]]; then
    echo "AIM347: environment file not found: $AIM347_ENV" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$AIM347_ENV"
# Both nodes, whole: one task per node with all its CPUs and GPUs. sbatch, salloc and srun take these.
# shellcheck disable=SC2034
two_nodes=(--partition="$PARTITION" --nodes=2 --ntasks-per-node=1 --cpus-per-task="$AIM347_CPUS_PER_NODE"
           --exclusive --gres=gpu:"$GPUS_PER_NODE")
