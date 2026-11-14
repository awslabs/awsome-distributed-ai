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
# RETENTION_TIME of history capped at RETENTION_SIZE. Both steps are best-effort:
# if one fails, monitoring still installs with node-local, default-retention storage.

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

# Pre-create the login stack's named volumes as bind mounts into PERSIST_DIR;
# compose reuses an existing volume of the same name, so the data lands on
# /home. Labels match what compose would set, so it adopts them without a warning.
persist_volumes() {
  mountpoint -q /home || { echo "WARN: /home is not mounted; keeping monitoring data node-local"; return 1; }
  docker info >/dev/null 2>&1 || { echo "WARN: docker not running yet; keeping monitoring data node-local"; return 1; }
  local vol owner
  for vol in prometheus-data:65534:65534 grafana-data:472:0; do
    owner="${vol#*:}"; vol="${vol%%:*}"
    mkdir -p "${PERSIST_DIR}/${vol}"
    chown "${owner}" "${PERSIST_DIR}/${vol}"
    if docker volume inspect "${COMPOSE_PROJECT}_${vol}" >/dev/null 2>&1; then
      echo "Volume ${COMPOSE_PROJECT}_${vol} already exists; leaving it as is"
      continue
    fi
    docker volume create --driver local \
      --opt type=none --opt o=bind --opt device="${PERSIST_DIR}/${vol}" \
      --label com.docker.compose.project="${COMPOSE_PROJECT}" \
      --label com.docker.compose.volume="${vol}" \
      "${COMPOSE_PROJECT}_${vol}" >/dev/null && echo "Volume ${COMPOSE_PROJECT}_${vol} -> ${PERSIST_DIR}/${vol}"
  done
}

# The upstream compose file has no retention knob, so add the flags to the
# node-local copy after install and recreate Prometheus.
set_retention() {
  local compose="${MONITORING_HOME}/compose/head.yml"
  grep -q -- '--storage.tsdb.retention' "$compose" 2>/dev/null && return 0
  grep -q -- "- '--storage.tsdb.path=/prometheus'" "$compose" 2>/dev/null || {
    echo "WARN: --storage.tsdb.path not found in ${compose}; Prometheus keeps its default retention"
    return 1
  }
  sed -i "s|^\( *\)- '--storage.tsdb.path=/prometheus'|&\n\1- '--storage.tsdb.retention.time=${RETENTION_TIME}'\n\1- '--storage.tsdb.retention.size=${RETENTION_SIZE}'|" "$compose"
  (cd "${MONITORING_HOME}" && docker compose -f "$compose" -p "${COMPOSE_PROJECT}" up -d prometheus) \
    && echo "Prometheus retention set to ${RETENTION_TIME} / ${RETENTION_SIZE}"
}

[ "${NODE_TYPE}" = "login" ] && persist_volumes

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

[ "${NODE_TYPE}" = "login" ] && [ "$rc" -eq 0 ] && set_retention

echo "Monitoring installation complete (exit ${rc})"
exit "$rc"
