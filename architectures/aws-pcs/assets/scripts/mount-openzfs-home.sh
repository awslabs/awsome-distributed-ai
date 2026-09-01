#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# mount-openzfs-home.sh — mount the shared FSx OpenZFS filesystem as /home.
#
# Usage: mount-openzfs-home.sh <fs-id>
#   <fs-id>  FSx OpenZFS filesystem ID (e.g. fs-01234567890abcdef)
#
# On first boot the DLAMI has a local /home already populated (at least
# /home/ubuntu). We rsync it aside, mount the shared FSx OpenZFS export at
# /home, then rsync the aside back in with --ignore-existing so any pre-existing
# shared state wins. The restore skips perms/dir-times (--no-perms
# --omit-dir-times) so it only fills in missing files and never rewrites the
# mode/times of a directory that already exists on the shared export (e.g. an
# admin-tightened /home/ubuntu). IMDS region is used to build the FSx DNS name.
#
# Runs as root. Idempotent — safe to re-run on reboot when the fstab entry is
# already present and /home is already mounted.

set -euo pipefail

FS_ID="${1:?filesystem-id required as $1}"
LOG=/var/log/pcs-mount-openzfs-home.log
exec > >(tee -a "$LOG") 2>&1

# AWS region from IMDSv2 (falls back to metadata-v1 read).
# -f so a 4xx/5xx body (e.g. a 401 under HttpTokens=required) becomes an empty
# string instead of being passed through as a bogus region; --retry/timeouts to
# ride out a throttled or slow IMDS; || true so a connection failure does not
# kill the script on the assignment line under `set -e` — that would skip the
# ${REGION:?} diagnostic below, and on a TERMINATE action the instance is gone
# before anyone can read the (truncated, error-less) log.
IMDS_CURL="curl -sf --retry 5 --retry-connrefused --connect-timeout 2 --max-time 5"
TOKEN=$($IMDS_CURL -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300" || true)
if [ -n "$TOKEN" ]; then
  REGION=$($IMDS_CURL -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/placement/region || true)
else
  REGION=$($IMDS_CURL http://169.254.169.254/latest/meta-data/placement/region || true)
fi
: "${REGION:?failed to resolve region from IMDS}"

DNS="${FS_ID}.fsx.${REGION}.amazonaws.com"
FSTAB_LINE="${DNS}:/fsx/ /home nfs noatime,nfsvers=3,sync,nconnect=16,rsize=1048576,wsize=1048576,defaults 0 0"

# Keep exactly one active /home entry. Match the mountpoint field, not the whole
# line, so a changed FS id/region (stack update) rewrites it instead of leaving
# a stale second entry that breaks `mount`/reboot. Commented lines are ignored.
ensure_home_fstab_entry() {
  grep -qxF "$FSTAB_LINE" /etc/fstab && return 0
  sed -i -E '\|^[^#].*[[:space:]]/home[[:space:]]|d' /etc/fstab
  echo "$FSTAB_LINE" >> /etc/fstab
}

# Tolerate rsync 23/24 so they don't trip `set -e` into a TERMINATE: 23 is a
# non-trivial ACL hitting an NFSv3 export with no ACL protocol, 24 a source file
# vanishing mid-copy. Any other non-zero code is real and propagates.
safe_rsync() {
  local rc=0
  rsync "$@" || rc=$?
  [ "$rc" -eq 0 ] || [ "$rc" -eq 23 ] || [ "$rc" -eq 24 ] || return "$rc"
}

# Already mounted — nothing to do (reboot path).
if mountpoint -q /home; then
  echo "/home already mounted; ensuring fstab entry is present"
  ensure_home_fstab_entry
  exit 0
fi

# Stash the local /home, mount over it, restore any file that is not on the
# shared filesystem.
mkdir -p /tmp/home
safe_rsync -aA /home/ /tmp/home
ensure_home_fstab_entry

# Bounded retry: the OpenZFS DNS name is commonly not yet resolvable on a fresh
# node (NFS settle race) and `mount` fails instantly. This action runs with
# OnError:TERMINATE, so a bare failure replaces the node into the same window —
# retry here so TERMINATE fires only on a persistent failure.
# Mount /home specifically (not `mount -a`) so an unrelated NFS entry a custom
# AMI may carry can't fail this action and terminate the node.
n=0; max=6; delay=10
while :; do
  if mount /home && mountpoint -q /home; then
    break
  fi
  n=$((n + 1))
  if [ "$n" -ge "$max" ]; then
    echo "ERROR: /home mount failed after $max attempts" >&2
    exit 1
  fi
  echo "mount attempt $n/$max failed (likely NFS DNS settle race); retrying in ${delay}s"
  sleep "$delay"
done
if [ "enabled" = "$(sestatus 2>/dev/null | awk '/^SELinux status:/{print $3}')" ]; then
  setsebool -P use_nfs_home_dirs 1
fi
safe_rsync -aA --ignore-existing --no-perms --omit-dir-times /tmp/home/ /home
rm -rf /tmp/home

echo "/home mounted from $DNS"
