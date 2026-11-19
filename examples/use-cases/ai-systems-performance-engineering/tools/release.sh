#!/usr/bin/env bash
# Release the session allocation so the GPU nodes are free.
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 0 ]] || { echo 'Usage: bash tools/release.sh' >&2; exit 2; }
# shellcheck source=tools/common.sh
source "$lab/tools/common.sh"
state=$HOME/.aim347
job_id=${AIM347_ALLOCATION_ID:-$( (squeue --me -h --name=aim347-session -o %i || true) | sed -n 1p)}
if [[ -z $job_id ]]; then
    echo 'No session allocation is held.'
    rm -f "$state/allocation.env" "$state/running"
    squeue --me
    exit 0
fi
scancel "$job_id"
for _ in $(seq 60); do
    [[ -z $(squeue --me -h --name=aim347-session) ]] && break
    sleep 1
done
if [[ -n $(squeue --me -h --name=aim347-session) ]]; then
    echo "Allocation $job_id is still listed after 60 s:"
    squeue --me
    exit 1
fi
rm -f "$state/allocation.env" "$state/running"
if [[ -z $(squeue --me -h) ]]; then
    echo "Released allocation $job_id. squeue shows no jobs of yours."
else
    echo "Released allocation $job_id. squeue still shows other jobs of yours:"
fi
squeue --me
