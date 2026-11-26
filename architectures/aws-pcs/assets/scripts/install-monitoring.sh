#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# install-monitoring.sh — install the monitoring stack. Runs on every node
# group when monitoring is enabled: the Prometheus/Grafana server on the login
# node (MonitoringRole=login), the DCGM/node/Slurm exporters on compute nodes
# (MonitoringRole=compute). The upstream installer branches on role internally.
#
# Runs as a nodeReady lifecycle action (slurmd already up) rather than blocking
# node availability on the multi-minute apt + Docker install. Trade-off: on
# compute nodes a job can be dispatched while this is still finishing, since
# nodeReady runs after the node is job-eligible.
#
# Usage: install-monitoring.sh <monitoring-version> <monitoring-repo> [<dcgm-exporter-image>]
#   monitoring-version   — aws-parallelcluster-monitoring git ref (tag/branch)
#   monitoring-repo      — GitHub "owner/repo" to fetch from
#   dcgm-exporter-image  — optional dcgm-exporter image override; empty keeps
#                          the installer's default pin
#
# stdout/stderr land in the per-script lifecycle log
# (/var/log/amazon/pcs/lifecycle/actions/nodeReady/install-monitoring.log).
#
# On the login node the Prometheus TSDB and Grafana DB are kept on shared /home
# (PERSIST_DIR) so they survive a login-node replacement, and Prometheus keeps
# RETENTION_TIME of history capped at RETENTION_SIZE. The Grafana volume is
# pre-created on /home before the upstream installer runs; the Prometheus volume
# moves to /home only after the retention flags are in the node-local compose
# file, so no Prometheus with the default 15-day retention ever opens the shared
# TSDB. Both steps are best-effort: if one fails, monitoring still installs with
# that store node-local (and Prometheus with its default retention).

set -uo pipefail

MONITORING_VERSION="${1:?monitoring-version required as $1}"
MONITORING_REPO="${2:?monitoring-repo required as $2}"
# The monitoring installer reads DCGM_EXPORTER_IMAGE from its environment and
# falls back to its default pin when empty. See aws-parallelcluster-monitoring #50.
export DCGM_EXPORTER_IMAGE="${3:-}"

echo "Installing monitoring stack: ${MONITORING_REPO}@${MONITORING_VERSION}"

PERSIST_DIR=/home/monitoring-data
RETENTION_TIME=30d
RETENTION_SIZE=50GB
# The upstream installer starts the login stack as compose project
# "monitoring-head" from MONITORING_HOME, with named volumes prometheus-data and
# grafana-data, and picks the node type from the monitoring-role instance tag.
COMPOSE_PROJECT=monitoring-head
MONITORING_HOME=/opt/aws-parallelcluster-monitoring

imds_tag() {
  local token
  token=$(curl -fsS -X PUT http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60") || return 1
  curl -fsS -H "X-aws-ec2-metadata-token: ${token}" \
    "http://169.254.169.254/latest/meta-data/tags/instance/$1"
}
NODE_TYPE=$(imds_tag monitoring-role 2>/dev/null || echo compute)

# Pre-create the Grafana volume as a bind mount into PERSIST_DIR; compose reuses
# an existing volume of the same name, so the data lands on /home. Labels match
# what compose would set, so it adopts it without a warning. PERSIST_DIR is
# root-only: Docker copies the image's mode-777 /var/lib/grafana into the empty
# directory, and every user on every node can reach /home.
persist_grafana() {
  local vol="${COMPOSE_PROJECT}_grafana-data" dir="${PERSIST_DIR}/grafana-data"
  mountpoint -q /home || { echo "WARN: /home is not mounted; keeping Grafana data node-local"; return 1; }
  docker info >/dev/null 2>&1 || { echo "WARN: docker not running yet; keeping Grafana data node-local"; return 1; }
  install -d -m 0700 -o root -g root "${PERSIST_DIR}" && mkdir -p "$dir" && chown 472:0 "$dir" || return 1
  # The search index is a cache Grafana rebuilds from grafana.db. A login node
  # lost without a clean shutdown leaves its lock on the index held by the
  # FSx for OpenZFS (NFSv3) server, and Grafana then exits at start.
  rm -rf "$dir/unified-search"
  docker volume inspect "$vol" >/dev/null 2>&1 && return 0
  docker volume create --driver local \
    --opt type=none --opt o=bind --opt device="$dir" \
    --label com.docker.compose.project="${COMPOSE_PROJECT}" \
    --label com.docker.compose.volume=grafana-data \
    "$vol" >/dev/null && echo "Volume $vol -> $dir"
}

# Prometheus deletes a block by removing its files; over NFS a file that is still
# open is renamed to .nfs* instead, so the block directory stays behind without
# meta.json. Prometheus then skips it on load but fails every compaction on it.
# Called only while no Prometheus has the TSDB open.
remove_incomplete_blocks() {
  local block
  for block in "$1"/0*/; do
    [ -d "$block" ] && [ ! -e "${block}meta.json" ] || continue
    echo "Removing incomplete TSDB block ${block}"
    rm -rf "$block"
  done
  return 0
}

# The upstream compose file has no retention knob, and the installer's first
# start of Prometheus uses the default 15-day retention, which deletes older
# blocks when it opens a TSDB. So add the flags to the node-local compose file
# first, and only then point the Prometheus volume at /home and recreate it.
persist_prometheus() {
  local compose="${MONITORING_HOME}/compose/head.yml"
  local vol="${COMPOSE_PROJECT}_prometheus-data" dir="${PERSIST_DIR}/prometheus-data"
  if ! grep -q -- '--storage.tsdb.retention' "$compose"; then
    grep -q -- "- '--storage.tsdb.path=/prometheus'" "$compose" || {
      echo "WARN: --storage.tsdb.path not found in ${compose}; keeping Prometheus node-local with default retention"
      return 1
    }
    sed -i "s|^\( *\)- '--storage.tsdb.path=/prometheus'|&\n\1- '--storage.tsdb.retention.time=${RETENTION_TIME}'\n\1- '--storage.tsdb.retention.size=${RETENTION_SIZE}'|" "$compose"
  fi
  cd "${MONITORING_HOME}" || return 1
  if mountpoint -q /home && [ "$(docker volume inspect -f '{{index .Options "device"}}' "$vol" 2>/dev/null)" != "$dir" ]; then
    install -d -m 0700 -o root -g root "${PERSIST_DIR}" && mkdir -p "$dir" && chown 65534:65534 "$dir" &&
      docker compose -f "$compose" -p "${COMPOSE_PROJECT}" rm -s -f prometheus >/dev/null &&
      docker volume rm "$vol" >/dev/null &&
      # A login node lost without a clean shutdown leaves its flock on this file
      # held by the FSx for OpenZFS (NFSv3) server, and Prometheus then refuses
      # to open the TSDB. This runs once, on a new login instance (the login
      # group keeps one instance), so the previous holder is gone.
      rm -f "$dir/lock" &&
      remove_incomplete_blocks "$dir" &&
      docker volume create --driver local \
        --opt type=none --opt o=bind --opt device="$dir" \
        --label com.docker.compose.project="${COMPOSE_PROJECT}" \
        --label com.docker.compose.volume=prometheus-data \
        "$vol" >/dev/null && echo "Volume $vol -> $dir" ||
      echo "WARN: could not move Prometheus data to $dir"
  fi
  docker compose -f "$compose" -p "${COMPOSE_PROJECT}" up -d prometheus &&
    echo "Prometheus retention set to ${RETENTION_TIME} / ${RETENTION_SIZE}"
}

[ "${NODE_TYPE}" = "login" ] && persist_grafana

# Root-owned scratch dir. nodeReady runs after the node is job-eligible, so a
# dispatched job could otherwise win a race to create a fixed /tmp path as a
# directory and make the download fail (curl exit 23) — with FIRST_BOOT_ONLY +
# CONTINUE that would skip monitoring permanently. mktemp -d avoids the race.
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

# Fetch post-install.sh from MONITORING_REPO at the MONITORING_VERSION ref and
# pass both to it so the tarball is pulled from the same source — this lets a
# fork + branch be used for testing unreleased changes.
curl -fsSL --retry 3 --retry-delay 15 \
  "https://raw.githubusercontent.com/${MONITORING_REPO}/${MONITORING_VERSION}/post-install.sh" \
  -o "$WORKDIR/post-install.sh" || {
    echo "ERROR: failed to fetch monitoring post-install.sh from ${MONITORING_REPO}@${MONITORING_VERSION}" \
         "(curl retries transient/network errors but not HTTP 4xx — a 404 from a wrong repo/ref exits on the first try)"
    exit 1
  }

# Make every child `apt-get` in the upstream installer wait up to 5 min for
# the dpkg lock (default is fail-fast). The installer runs unedited, so we
# can't pass `-o DPkg::Lock::Timeout=300` per-call — an apt.conf.d drop-in is a
# persistent node-wide file, so every apt-get on the node (including
# unattended-upgrades) inherits the timeout thereafter. Guards against races
# with unattended-upgrades / apt-daily at first boot.
echo 'DPkg::Lock::Timeout "300";' > /etc/apt/apt.conf.d/99lock-timeout

# Retry up to 3 times for non-lock transient failures (network flakes, image
# pulls). The installer is idempotent (reuses the existing Grafana SSM param,
# recreates containers), so re-running self-heals a one-off hiccup instead of
# leaving the node unmonitored.
rc=1
for attempt in 1 2 3; do
  echo "Monitoring install attempt ${attempt}/3..."
  bash "$WORKDIR/post-install.sh" "${MONITORING_VERSION}" "${MONITORING_REPO}"
  rc=$?
  [ "$rc" -eq 0 ] && break
  echo "Monitoring install attempt ${attempt} failed (exit ${rc}); retrying in 15s..."
  sleep 15
done

# Upstream installs under `bash -x | tee`, tracing the generated Grafana admin
# password into this log before the upstream `unset`. The login node is
# multi-user, so restrict the log wherever the installer wrote it. Upstream fix
# (drop -x / redact) tracked separately.
[ -f /var/log/parallelcluster-monitoring-install.log ] && \
  chmod 600 /var/log/parallelcluster-monitoring-install.log

[ "${NODE_TYPE}" = "login" ] && [ "$rc" -eq 0 ] && persist_prometheus

echo "Monitoring installation complete (exit ${rc})"
exit "$rc"
