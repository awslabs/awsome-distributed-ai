#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Fail closed unless every NCCL process in a job log reports EFA ("Using network Libfabric") and none reports the TCP
# socket fallback. NCCL falls back to sockets silently when it can't load its network plugin, and the job still
# completes, so a timing without this check may not be an EFA timing at all.
# Run the job with NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS including INIT and NET.
#
# Usage: ./check_efa_transport.sh <job.log> [<job.log> ...] <number-of-NCCL-processes>
set -uo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <job.log> [<job.log> ...] <number-of-NCCL-processes>"; exit 2; }
WORLD=${*: -1}; LOGS=("${@:1:$#-1}")
# NCCL INFO lines start with <host>:<pid>:<tid>; count distinct host:pid pairs that logged the EFA line.
efa=$(cat "${LOGS[@]}" | grep -oE "[A-Za-z0-9.-]+:[0-9]+:[0-9]+ \[[0-9]+\] NCCL INFO Using network Libfabric" \
      | cut -d: -f1,2 | sort -u | wc -l | tr -d ' ')
tcp=$(cat "${LOGS[@]}" | grep -c "Using network Socket")
if [ "$tcp" -gt 0 ] || [ "$efa" -lt "$WORLD" ]; then
  echo "NOT EFA: $efa of $WORLD processes on Libfabric, $tcp on sockets; discard this run"
  exit 1
fi
echo "EFA confirmed on $efa of $WORLD processes"
