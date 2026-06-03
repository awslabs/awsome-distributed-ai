#!/bin/bash

# 
# This script sets up slurm accounting to an RDS endpoint
# 1. Writes the accounting config to the slurm.conf file
# 2. Creates the slurmdbd.conf in /opt/slurm/etc
# 3. Creates the slurmdbd.service file in /etc/systemd/system
# 4. Restarts the slurmctld and slurmdbd daemons
# 
# It assumes that the user has created a connection to their RDS database
# The script takes the RDS endpoint as param
# 
# Usage: 
# ./setup_rds_accounting.sh <RDS_ENDPOINT>
# 

set -e
set -x
set -o pipefail  # trace ERR through pipes
set -o errtrace  # trace ERR through 'time command' and other functions
set -o nounset   ## set -u : exit the script if you try to use an uninitialised variable


RDS_ENDPOINT="$1"
DB_HOST="head-node"
DB_USER="admin"
DB_PASS="XXXXX"
echo "RDS_ENDPOINT=$RDS_ENDPOINT"


create_slurmdbd_config() {
  cat >/opt/slurm/etc/slurmdbd.conf <<EOF
#
#
ArchiveEvents=yes
ArchiveJobs=yes
ArchiveResvs=yes
ArchiveSteps=no
ArchiveSuspend=no
ArchiveTXN=no
ArchiveUsage=no
AuthType=auth/munge
DbdHost=$DB_HOST  #YOUR_MASTER_IP_ADDRESS_OR_NAME
DbdPort=6819
DebugLevel=info
PurgeEventAfter=1month
PurgeJobAfter=12month
PurgeResvAfter=1month
PurgeStepAfter=1month
PurgeSuspendAfter=1month
PurgeTXNAfter=12month
PurgeUsageAfter=24month
SlurmUser=slurm
LogFile=/var/log/slurmdbd.log
PidFile=/var/run/slurmdbd.pid
StorageType=accounting_storage/mysql
StorageUser=$DB_USER
StoragePass=$DB_PASS
StorageHost=$RDS_ENDPOINT # Endpoint from RDS console
StoragePort=3306

EOF
}


write_accounting_to_slurm_conf() {
  cat >>/opt/slurm/etc/slurm.conf <<EOF

# ACCOUNTING
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30
#
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=$DB_HOST
AccountingStorageUser=$DB_USER
AccountingStoragePort=6819
EOF
}


create_slurmdbd_service() {
  cat >/etc/systemd/system/slurmdbd.service <<EOF
[Unit]
Description=Slurm node daemon
After=munge.service network.target remote-fs.target home.mount
Wants=munge.service network.target remote-fs.target home.mount
ConditionPathExists=/opt/slurm/etc/slurmdbd.conf
Documentation=man:slurmdbd(8)

[Service]
Type=simple
ExecCondition=bash -c "cd /opt/slurm/etc/ >& /dev/null"
EnvironmentFile=-/opt/slurm/etc/default/slurmctld
ExecStart=/opt/slurm/sbin/slurmdbd -D \$SLURMD_OPTIONS
ExecReload=/bin/kill -HUP \$MAINPID
PIDFile=/run/slurmdbd.pid
KillMode=process
LimitNOFILE=131072
LimitMEMLOCK=infinity
LimitSTACK=infinity
Delegate=yes
TasksMax=infinity

[Install]
WantedBy=multi-user.target graphical.target
EOF

}

restart_slurm_daemons() {
  systemctl daemon-reload
  systemctl restart slurmctld
  systemctl restart slurmdbd 
}

main() {
  read -s -p "Enter the password to connect to the RDS database: " DB_PASS
  create_slurmdbd_config
  write_accounting_to_slurm_conf
  create_slurmdbd_service
  restart_slurm_daemons
}

main "$@"

