# shellcheck shell=bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# The participant's login profile on the login node; ~/.bashrc sources it. It loads
# lab.env, puts the EFA and Slurm clients on PATH and opens the lab directory.
# Inside an allocation it sets RESULTS_DIR to node-local storage, where job.sh
# writes its results and the job's processes write their stack files.
if [[ $- == *i* ]]; then
    aim344_lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    # shellcheck source=/dev/null
    source "$aim344_lab/lab.env"
    case ":$PATH:" in
        *":$AIM344_SLURM_BIN:"*) ;;
        *) export PATH=/opt/amazon/efa/bin:$AIM344_SLURM_BIN:$PATH ;;
    esac
    if [[ $PWD == "$HOME" ]]; then cd -- "$aim344_lab" || true; fi
    if [[ -n ${SLURM_JOB_ID:-} ]]; then export RESULTS_DIR=/var/tmp/aim344-$USER-$SLURM_JOB_ID; fi
    unset aim344_lab
fi
