# shellcheck shell=bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# The login profile of the participant on the login node. One line in the
# participant's ~/.bashrc sources this file (the installer prints that line), so
# every interactive shell reads it, including the shell that salloc opens.
#
# The first shell of a session loads the site's lab.env, which also carries the
# seat's handout values (AIM344_TARGET_NODE, AIM344_PEER_NODE, AIM344_EFA_DEVICE)
# and the Slurm client directory (AIM344_SLURM_BIN). It puts the EFA and Slurm
# clients on PATH and sets the allocation values that Slurm registers for the pair
# (PARTITION, GPUS_PER_NODE, CPUS_PER_RANK, MEMORY_PER_NODE). A shell that starts
# in the home directory moves to the lab directory. Shells started from it,
# such as the one salloc opens, inherit these values. A shell inside an
# allocation also exports RESULTS_DIR on node-local storage, where job.sh
# writes its results and the job's processes write their stack files.
if [[ $- == *i* ]]; then
    aim344_lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
    if [[ -z ${AIM344_PROFILE_LOADED:-} && -r $aim344_lab/lab.env ]]; then
        # shellcheck source=/dev/null
        source "$aim344_lab/lab.env"
        case ":$PATH:" in
            *":$AIM344_SLURM_BIN:"*) ;;
            *) export PATH=/opt/amazon/efa/bin:$AIM344_SLURM_BIN:$PATH ;;
        esac
        if aim344_values=$(python3 "$aim344_lab/facilitator/allocation-env.py" --partition "${PARTITION:-}" 2>/dev/null); then
            eval "$aim344_values"
        else
            printf 'AIM344 login profile: Slurm did not return the allocation values; raise your hand.\n' >&2
        fi
        export AIM344_PROFILE_LOADED=1
        if [[ $PWD == "$HOME" ]]; then cd -- "$aim344_lab" || true; fi
        unset aim344_values
    fi
    if [[ -n ${AIM344_PROFILE_LOADED:-} && -n ${SLURM_JOB_ID:-} ]]; then
        RESULTS_DIR=/var/tmp/aim344-$(id -un)-$SLURM_JOB_ID
        export RESULTS_DIR
    fi
    unset aim344_lab
fi
