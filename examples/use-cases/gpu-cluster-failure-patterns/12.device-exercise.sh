#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Participant entry point for the device exercise.
#
# Run this from your coordinator terminal. It sends one command over the
# exercise's control connection. You never need root on the exercise node, and
# you cannot select another table's node: the connection itself carries your
# assignment.
#
#   ./12.device-exercise.sh start gpu    begin the GPU round
#   ./12.device-exercise.sh start efa    begin the EFA round
#   ./12.device-exercise.sh status       where your exercise stands
#   ./12.device-exercise.sh collect 0    run one allowed check and keep its output
#   ./12.device-exercise.sh recover      ask for recovery once you have your evidence
#
# start runs idle removal/unbind; active runs FLR against your running workload.
# status lists the checks available for your current round. Recovery
# leaves the runtime ready; you then take a fresh allocation and verify it
# yourself.
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage: 12.device-exercise.sh <command>

  start gpu        Begin the GPU round on your assigned exercise node
  start efa        Begin the EFA round on your assigned exercise node
  active gpu       FLR the provisioned GPU used by your running device workload
  active efa       FLR the provisioned EFA carrying your running device workload
  status           Show your exercise phase, node state and available checks
  collect <check>  Run one allowed read-only check and save its output
  collect kernel   Save a bounded kernel window (unavailable logs do not block recovery)
  recover          Request recovery after recording your evidence
  replace          Replace the assigned instance after recovery cannot qualify it

start needs an idle node. active requires your one RUNNING device workload.
Keep the workload terminal and logs on the unaffected coordinator; use a second
terminal for active/collect/recover. Do not repeat an active fault after timeout.
The exercise node, its devices and your job are resolved from your assignment.
This script accepts no node name, PCI address or job identifier.
USAGE
    exit 2
}

[[ $# -ge 1 ]] || usage
case "$1" in
    start|active)
        [[ $# -eq 2 ]] || usage
        [[ $2 == gpu || $2 == efa ]] || usage
        request="$1 $2"
        ;;
    collect)
        [[ $# -eq 2 ]] || usage
        [[ $2 == kernel || $2 =~ ^[0-9]{1,2}$ ]] || usage
        request="collect $2"
        ;;
    status|recover|replace)
        [[ $# -eq 1 ]] || usage
        request="$1"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        ;;
esac

# The control endpoint is prepared for you. It is a forced command, so the only
# thing that travels is the request above.
control_host=${AIM344_CONTROL_HOST:-localhost}
control_user=${AIM344_CONTROL_USER:-aim344-control}
identity=${AIM344_CONTROL_IDENTITY:-$HOME/.ssh/aim344-exercise}

if [[ ! -r $identity ]]; then
    printf 'Cannot read your exercise key at %s. Ask the facilitator.\n' "$identity" >&2
    exit 3
fi

set +e
ssh -n -i "$identity" \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=10 \
    -o RequestTTY=no \
    "$control_user@$control_host" "$request"
status=$?
set -e

if [[ $status -eq 255 ]]; then
    printf '\nThe control connection failed. Your exercise state is kept on the\n' >&2
    printf 'coordinator, so run status again once the connection returns.\n' >&2
fi
exit "$status"
