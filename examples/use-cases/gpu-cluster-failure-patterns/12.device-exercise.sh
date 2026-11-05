#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Participant entry point for the device exercise.
#
# Run this from your coordinator terminal. It passes one command to the lab
# controller, which runs as root through sudo and acts on the target named in
# its configuration.
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
  collect recovery Read and save the current recovery actions
  resume <issue id> Check whether the current job may resume; job.sh runs this
  recover          Request recovery of your target once you have your evidence

A page that names its round gives that round's start or active command in full.
Send one command at a time; while one runs, any command other than status is refused.
A node that recovery cannot qualify stays out of service; an operator replaces
it after the session.
Your target, its devices and your job come from the lab's configuration, so this
script accepts no node name, device address or job identifier.
USAGE
    exit 2
}

[[ $# -ge 1 ]] || usage
case "$1" in
    start|active)
        [[ $# -eq 2 ]] || usage
        [[ $2 == gpu || $2 == efa ]] || usage
        request=("$1" "$2")
        ;;
    collect)
        [[ $# -eq 2 ]] || usage
        [[ $2 == kernel || $2 == recovery || $2 =~ ^[0-9]{1,2}$ ]] || usage
        request=(collect "$2")
        ;;
    ended)
        [[ $# -eq 2 ]] || usage
        [[ $2 =~ ^[0-9a-f]{32}$ ]] || usage
        request=(ended "$2")
        ;;
    resume)
        [[ $# -eq 2 ]] || usage
        [[ $2 =~ ^[0-9a-f]{32}$ ]] || usage
        request=(resume "$2")
        ;;
    next|status|recover|replace)
        [[ $# -eq 1 ]] || usage
        request=("$1")
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        ;;
esac

exec sudo -n /usr/local/sbin/aim344-device-session "${request[@]}"
