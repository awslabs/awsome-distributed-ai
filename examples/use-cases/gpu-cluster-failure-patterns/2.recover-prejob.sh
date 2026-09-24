#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Run on the same compute node before the administrator resumes it in Slurm.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Run as root on the marked compute node.' >&2; exit 2; }
marker=/run/aim344-unhealthy
[[ -f $marker && ! -L $marker && $(stat -c %u "$marker") == 0 ]] || exit 1
read -r tag target_user < "$marker"
[[ $tag == AIM344 && -n $target_user ]] || exit 1
rm -- "$marker"
SLURM_JOB_ID=aim344-recovery SLURM_JOB_USER="$target_user" bash /opt/aim344/prejob-prolog.sh
echo 'Synthetic marker removed and lightweight checks passed. Inspect the drain reason before resuming this node.'
