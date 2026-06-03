#!/bin/bash

# must be run as sudo
# USAGE: start_slurm.sh <NODE_TYPE> [<CONTOLLER_ADDRESSES>] [<SMHP_NCCL_METRICS>] [<SMHP_NCCL_DUMP_INTERVAL_SECONDS>] [<SMHP_NCCL_PLUGIN_PATH>]
# - Where NODE_TYPE is one of follow values: controller, compute, login
# - SMHP_NCCL_METRICS is "1" to enable NCCL Inspector task prolog
# - SMHP_NCCL_DUMP_INTERVAL_SECONDS is the dump interval in seconds (default: 30)
# - SMHP_NCCL_PLUGIN_PATH is the path to the NCCL Inspector .so plugin

set -ex

LOG_FILE="/var/log/provision/provisioning.log"
CONTROLLER_IP_VALUES=($2)
SMHP_NCCL_METRICS=$3
SMHP_NCCL_DUMP_INTERVAL_SECONDS=${4:-30}
SMHP_NCCL_PLUGIN_PATH=$5

main() {
  echo "[INFO] START: Starting Slurm daemons"

  # The scripts are downloaded from the customer S3 bucket by HyperPod into
  # /tmp/<bucket-name>/, which is the working directory this script is
  # launched from.  Derive the path dynamically so it works regardless of
  # the bucket name.
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

  # /tmp may be cleared on reboot, so copy prolog/epilog to a persistent
  # location and make them executable there.
  # slurmctld (controller) validates executability at startup; slurmd
  # (compute) must also be able to exec them when a job runs.
  SLURM_SCRIPTS_DIR="/opt/slurm/etc/scripts"
  mkdir -p "$SLURM_SCRIPTS_DIR"
  for script in prolog.sh epilog.sh; do
      src="$SCRIPT_DIR/$script"
      dst="$SLURM_SCRIPTS_DIR/$script"
      if [ -f "$src" ]; then
          cp "$src" "$dst"
          chmod +x "$dst"
          echo "[INFO] Copied and made executable: $dst"
      else
          echo "[WARN] $src not found, skipping"
      fi
  done

  # Create NCCL Inspector task prolog script on all nodes before starting daemons
  if [[ "$SMHP_NCCL_METRICS" == "1" ]]; then
    echo "[INFO] Creating NCCL metrics task prolog script..."
    DUMP_INTERVAL_MICROSECONDS=$((SMHP_NCCL_DUMP_INTERVAL_SECONDS * 1000000))

    cat > /opt/slurm/etc/task_prolog.sh << EOF
#!/bin/bash
if [ ! -f ${SMHP_NCCL_PLUGIN_PATH} ]; then
  echo "[WARN] NCCL Inspector plugin not found at ${SMHP_NCCL_PLUGIN_PATH}, skipping NCCL metrics" >&2
  exit 0
fi
echo "export NCCL_PROFILER_PLUGIN=${SMHP_NCCL_PLUGIN_PATH}"
echo "export NCCL_INSPECTOR_ENABLE=1"
echo "export NCCL_INSPECTOR_PROM_DUMP=1"
echo "export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=${DUMP_INTERVAL_MICROSECONDS}"
echo "export NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/nccl_inspector/"
EOF
    chmod +x /opt/slurm/etc/task_prolog.sh
  fi

  if [[ $1 == "controller" ]]; then
    echo "[INFO] This is a Controller node. Start slurm controller daemon..."

    # Inject Prolog/Epilog paths into slurm.conf before slurmctld reads it.
    # Point at the persistent copies under /opt/slurm/etc/scripts/ rather than /tmp.
    # Remove any pre-existing lines to avoid duplicates on re-runs, then append.
    sed -i '/^Prolog=/d;/^Epilog=/d' /opt/slurm/etc/slurm.conf
    printf '\n' >> /opt/slurm/etc/slurm.conf
    echo "Prolog=${SLURM_SCRIPTS_DIR}/prolog.sh"  >> /opt/slurm/etc/slurm.conf
    echo "Epilog=${SLURM_SCRIPTS_DIR}/epilog.sh"  >> /opt/slurm/etc/slurm.conf
    echo "[INFO] Added Prolog and Epilog to /opt/slurm/etc/slurm.conf"

    # Configure NCCL Inspector slurm.conf on controller
    if [[ "$SMHP_NCCL_METRICS" == "1" ]]; then
      echo "[INFO] Adding TaskProlog to slurm.conf..."
      echo "TaskProlog=/opt/slurm/etc/task_prolog.sh" >> /opt/slurm/etc/slurm.conf
    fi

    systemctl enable --now slurmctld

    mv /etc/systemd/system/slurmd{,_DO_NOT_START_ON_CONTROLLER}.service \
        || { echo "Failed to mask slurmd, perhaps the AMI already masked it?" ; }
  elif [[ $1 == "compute" ]] || [[ $1 == "login" ]]; then
    echo "[INFO] Running on $1 node. Start slurm daemon..."

    # Login nodes must still restart slurmd to fetch slurm.conf to /var/spool/slurmd/, however
    # slurmd won't run because slurm.conf does not contain login nodes.
    SLURMD_OPTIONS="--conf-server $CONTROLLER_IP_VALUES" envsubst < /etc/systemd/system/slurmd.service > slurmd.service
    mv slurmd.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable --now slurmd

    mv /etc/systemd/system/slurmctld{,_DO_NOT_START_ON_CONTROLLER}.service \
        || { echo "Failed to mask slurmctld, perhaps the AMI already masked it?" ; }
  fi

  echo "[INFO] Start Slurm Script completed"
}

main "$@"
