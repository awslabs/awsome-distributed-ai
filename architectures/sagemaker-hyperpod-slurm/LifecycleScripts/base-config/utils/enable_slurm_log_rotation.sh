#!/usr/bin/env bash

LOGROTATE_CONF_FILEPATH="/etc/logrotate.d/sagemaker-hyperpod-slurm"

echo "[$(hostname)] Adding Slurm log rotation configuration to ${LOGROTATE_CONF_FILEPATH}"

cat <<EOF >>${LOGROTATE_CONF_FILEPATH}
"/var/log/slurm/*.log" {
    rotate 2
    size 50M
    copytruncate
    nocompress

    missingok
    nodelaycompress
    nomail
    notifempty
    noolddir
    sharedscripts
    postrotate
        pkill -x --signal SIGUSR2 slurmctld
        pkill -x --signal SIGUSR2 slurmd
        pkill -x --signal SIGUSR2 slurmdbd
        exit 0
    endscript
}
EOF
