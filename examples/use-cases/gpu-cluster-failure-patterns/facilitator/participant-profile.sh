# shellcheck shell=bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# The login profile of a participant account on the login node. The installer
# adds one line to the account's ~/.bashrc that sources this file, so every
# interactive shell reads it, including the shell that salloc opens.
#
# The first shell of a session loads the site's lab.env, puts the EFA and Slurm
# clients on PATH, sets the allocation values that Slurm registers for the pair
# (PARTITION, GPUS_PER_NODE, CPUS_PER_RANK, MEMORY_PER_NODE), and exports the
# seat's handout values from the root-owned seat file that the installer writes
# (AIM344_TARGET_NODE, AIM344_PEER_NODE, AIM344_EFA_DEVICE). A shell that starts
# in the home directory moves to the lab directory. Shells started from it,
# such as the one salloc opens, inherit these values. A shell inside an
# allocation also exports RESULTS_DIR on node-local storage, where job.sh
# writes its results and the job's processes write their stack files.
if [[ $- == *i* ]]; then
    aim344_lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
    aim344_seat=${AIM344_SEAT_DIR:-/etc/aim344-seats}/$(id -un).env
    if [[ -z ${AIM344_PROFILE_LOADED:-} && -r $aim344_lab/lab.env && -r $aim344_seat ]]; then
        # shellcheck source=/dev/null
        source "$aim344_lab/lab.env"
        # shellcheck source=/dev/null
        source "$aim344_seat"
        case ":$PATH:" in
            *":$AIM344_SLURM_BIN:"*) ;;
            *) export PATH=/opt/amazon/efa/bin:$AIM344_SLURM_BIN:$PATH ;;
        esac
        if aim344_values=$(python3 "$aim344_lab/facilitator/allocation-env.py" --partition "${PARTITION:-}" 2>/dev/null); then
            eval "$aim344_values"
        else
            printf 'AIM344 login profile: Slurm did not return the allocation values; tell your table expert.\n' >&2
        fi
        export AIM344_PROFILE_LOADED=1
        if [[ $PWD == "$HOME" ]]; then cd -- "$aim344_lab" || true; fi
        unset aim344_values
    fi
    if [[ -n ${AIM344_PROFILE_LOADED:-} && -n ${SLURM_JOB_ID:-} ]]; then
        RESULTS_DIR=/var/tmp/aim344-$(id -un)-$SLURM_JOB_ID
        export RESULTS_DIR
    fi
    unset aim344_lab aim344_seat
fi
