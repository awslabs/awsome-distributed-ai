#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Run as root directly on ONE dedicated, idle compute node via SSM.
set -euo pipefail
[[ $EUID == 0 && $# == 1 ]] || {
    echo 'Usage (on the dedicated compute node): sudo bash 1.inject-prejob.sh ATTENDEE_USER' >&2
    exit 2
}
id "$1" >/dev/null
[[ $1 =~ ^[a-zA-Z0-9_.-]+$ ]] || exit 2
[[ -x /opt/aim344/prejob-prolog.sh ]] || {
    echo 'Install the dedicated lab Prolog first; see README.md.' >&2; exit 1;
}
# Do not overwrite an existing marker or follow a symlink.
(umask 077; set -o noclobber; printf 'AIM344 %s\n' "$1" > /run/aim344-unhealthy)
printf 'Armed a synthetic pre-job failure for user=%s on host=%s.\n' "$1" "$(hostname)"
