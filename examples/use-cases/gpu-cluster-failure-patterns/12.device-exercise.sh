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
#   ./12.device-exercise.sh status       where your incident stands
#   ./12.device-exercise.sh next         start your seat's next incident (job.sh runs this)
#   ./12.device-exercise.sh ended <issue id>  record that the incident has ended (job.sh runs this)
#   ./12.device-exercise.sh collect 0    run one allowed check and keep its output
#   ./12.device-exercise.sh recover      ask for recovery once you have your evidence
#
# next follows the order set for your seat and does not say
# what the incident is. status lists the checks you can collect. Recovery
# leaves the runtime ready; you then take a fresh allocation and verify it
# yourself. A node that recovery cannot qualify stays out of service, and an
# operator replaces it after the session. The start and active commands remain
# for a page that names its round; that page gives the command in full.
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage: 12.device-exercise.sh <command>

  status           Show where your incident stands and which checks you can collect
  next             Start your seat's next incident; job.sh runs this for you
  ended <issue id> Record that the incident has ended; job.sh runs this for you
  collect <check>  Run one allowed check on your target and save its output
  collect kernel   Save a window of your target's kernel log
  recover          Request recovery of your target once you have your evidence

A page that names its round gives that round's start or active command in full.
Send one command at a time; a second request while one runs is refused.
A node that recovery cannot qualify stays out of service; an operator replaces
it after the session.
Your target, its devices and your job come from your assignment, so this
script accepts no node name, device address or job identifier.
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
    ended)
        [[ $# -eq 2 ]] || usage
        [[ $2 =~ ^[0-9a-f]{32}$ ]] || usage
        request="ended $2"
        ;;
    next|status|recover|replace)
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
    printf 'Cannot read your exercise key at %s. Raise your hand.\n' "$identity" >&2
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
