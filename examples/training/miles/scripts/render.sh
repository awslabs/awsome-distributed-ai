#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Fill a manifest's ${VAR} references from the environment, refusing to render one that is unset or
# empty. Plain envsubst writes `nvidia.com/gpu: ''` and exits 0, and that surfaces later as a quantity
# parse error or an ImagePullBackOff naming nothing.
#
# The substitution is restricted to the variables checked above, passed as envsubst's SHELL-FORMAT.
# Unrestricted, envsubst also replaces bare `$VAR`, which the check does not look for, so a `$HOME` in
# a command string inside the manifest would be rewritten without ever being verified.
set -uo pipefail

f="${1:?usage: render.sh <manifest>}"
[ -f "${f}" ] || { echo "[ERROR] render.sh: ${f} not found." >&2; exit 1; }
command -v envsubst >/dev/null || {
  echo "[ERROR] render.sh needs envsubst, from GNU gettext. Install gettext-base (Debian/Ubuntu)," >&2
  echo "[ERROR] gettext (Amazon Linux/RHEL) or 'brew install gettext' (macOS)." >&2; exit 1; }

# Comment lines are stripped first so a ${VAR} mentioned only in a comment is not demanded.
vars="$(envsubst --variables "$(grep -v '^[[:space:]]*#' "${f}")" | sort -u)"
missing=""
for v in ${vars}; do
  [ -n "$(printenv "${v}")" ] || missing="${missing} ${v}"
done
[ -z "${missing}" ] || {
  echo "[ERROR] render.sh: ${f} needs these unset or empty variables:${missing}" >&2
  echo "[ERROR] Copy env_vars from its example again and re-apply your edits." >&2; exit 1; }

# The ${%s} is literal on purpose: it is the SHELL-FORMAT envsubst matches against, not an
# expansion. ${vars} is left unquoted so printf repeats the format once per variable.
# shellcheck disable=SC2016,SC2086
envsubst "$(printf '${%s} ' ${vars})" < "${f}"
