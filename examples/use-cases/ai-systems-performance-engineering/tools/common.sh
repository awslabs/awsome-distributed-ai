# shellcheck shell=bash
# Sourced by every AIM347 tool: the event environment, then the session allocation.
AIM347_ENV=${AIM347_ENV:-/fsx/aim347/participant/aim347.env}
export AIM347_ENV
if [[ ! -f $AIM347_ENV ]]; then
    echo "AIM347: environment file not found: $AIM347_ENV" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$AIM347_ENV"
if [[ -f $HOME/.aim347/allocation.env ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.aim347/allocation.env"
fi

aim347_allocation_running() {
    [[ -n ${AIM347_ALLOCATION_ID:-} ]] || return 1
    [[ $(squeue -h -j "$AIM347_ALLOCATION_ID" -o %T 2>/dev/null) == RUNNING ]]
}

aim347_require_allocation() {
    if ! aim347_allocation_running; then
        echo 'AIM347: no session allocation. Run: bash tools/hold-and-measure.sh' >&2
        exit 3
    fi
}

# An allocation that has ended must not steer srun into it.
if ! aim347_allocation_running; then
    unset SLURM_JOB_ID SLURM_JOB_NUM_NODES SLURM_JOB_NODELIST SLURM_JOB_CPUS_PER_NODE AIM347_ALLOCATION_ID
fi
