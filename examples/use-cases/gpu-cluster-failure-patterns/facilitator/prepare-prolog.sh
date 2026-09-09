#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'Run as root on the login node' >&2; exit 1; }
partition=${1:?Usage: prepare-prolog.sh LAB_PARTITION}
[[ $partition =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
stage=/fsx/aim344
findmnt -T "$stage" -n -o FSTYPE | grep -qx lustre
[[ ! -L $stage && ! -L $stage/.prolog ]] || exit 2
chown root:root "$stage"
chmod 0755 "$stage"
chown root:root "$stage/healthcheck-pinned.tgz"
chmod 0644 "$stage/healthcheck-pinned.tgz"
chown -R root:root "$stage/healthcheck-source"
chmod -R go-w "$stage/healthcheck-source"
install -d -o root -g root -m 0755 "$stage/.prolog"
dispatcher="$stage/.prolog/dispatch.sh"
[[ ! -L $dispatcher ]] || exit 2
# Expand the partition now; Slurm supplies the job partition when the Prolog runs.
# shellcheck disable=SC2016
printf '#!/usr/bin/env bash\nset -euo pipefail\n[[ ${SLURM_JOB_PARTITION:-} == %q ]] || exit 0\n# Root maintenance jobs must be able to diagnose a failed participant gate.\n[[ ${SLURM_JOB_USER:-} != root ]] || exit 0\nexec /opt/aim344/prejob-prolog.sh\n' "$partition" > "$dispatcher"
chown root:root "$dispatcher"
chmod 0755 "$dispatcher"
namei -l "$dispatcher"
cat "$dispatcher"
