#!/bin/bash

set -euo pipefail

# Required with Ubuntu 22.04 AMI release
apt-get -y -o DPkg::Lock::Timeout=120 update && apt-get -y -o DPkg::Lock::Timeout=120 install apg


# https://askubuntu.com/a/1472412
# Exclude special characters that break create_slurm_database().
EXCLUDED_CHAR="'\"\`\\[]{}()*#"
SLURM_DB_PASSWORD=$(apg -a 1 -M SNCL -m 10 -x 10 -n 1 -E "${EXCLUDED_CHAR}")

# Retain adt behavior. However, verbosity will be disabled at select places to
# prevent credentials getting leaked to Cloudwatch logs.
set -x

SLURM_ACCOUNTING_CONFIG_FILE=/opt/slurm/etc/accounting.conf
SLURMDB_CONFIG_FILE=/opt/slurm/etc/slurmdbd.conf
SLURMDB_SERVICE_FILE=/etc/systemd/system/slurmdbd.service

LOG_DIR=/var/log/provision
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# Setup MariaDB using secure_installation and default password.
# Use expect to for the interactive shell.
setup_mariadb() {
  echo "Running mysql_secure_installation"
  set +x
  SECURE_MYSQL=$(expect -c "
  set timeout 10
  log_file /var/log/provision/secure_mysql.log
  spawn mysql_secure_installation
  expect \"Enter current password for root (enter for none):\"
  send \"\r\"
  expect \"Change the root password?\"
  send \"n\r\"
  expect \"Remove anonymous users?\"
  send \"y\r\"
  expect \"Disallow root login remotely?\"
  send \"y\r\"
  expect \"Remove test database and access to it?\"
  send \"y\r\"
  expect \"Reload privilege tables now?\"
  send \"y\r\"
  expect eof
  ")
  set -x
  chmod 400 /var/log/provision/secure_mysql.log
}

# Create the default database for SLURM accounting
create_slurm_database() {
  set +x
  echo "Creating accounting database"
  local ESCAPED_SLURM_DB_PASSWORD=$(printf '%q' "$SLURM_DB_PASSWORD")
  SETUP_MYSQL=$(expect -c "
  set timeout 15
  log_file /var/log/provision/setup_mysql.log
  match_max 10000
  spawn sudo mysql -u root -p
  expect \"Enter password:\"
  send \"\r\"
  sleep 1
  expect \"*]>\"
  send \"grant all on slurm_acct_db.* TO 'slurm'@'localhost' identified by '${ESCAPED_SLURM_DB_PASSWORD}' with grant option;\r\"
  sleep 1
  expect \"*]>\"
  send \"create database slurm_acct_db;\r\"
  sleep 1
  expect \"*]>\"
  send \"exit\r\"
  expect eof
  ")
  set -x
  chmod 400 /var/log/provision/setup_mysql.log
}

# Setup the configuration for slurmdbd to use MariaDB.
create_slurmdbd_config() {
  # Do not push db credentials to Cloudwatch logs
  echo 'BEGIN: create_slurmdbd_config()'
  set +x
  SLURM_DB_USER=slurm SLURM_DB_PASSWORD="$SLURM_DB_PASSWORD" envsubst < "$SLURMDB_CONFIG_FILE.template" > $SLURMDB_CONFIG_FILE
  set -x

  chown slurm:slurm $SLURMDB_CONFIG_FILE
  chmod 600 $SLURMDB_CONFIG_FILE
  echo 'END: create_slurmdbd_config()'
}

# Append the accounting settings to accounting.conf, this file is empty by default and included into
# slurm.conf. This is required for Slurm to enable accounting.
add_accounting_to_slurm_config() {
    # `hostname -i` gave us "hostname: Name or service not known". So let's parse slurm.conf.
    DBD_HOST=$(awk -F'[=(]' '/^SlurmctldHost=/ { print $NF }' /opt/slurm/etc/slurm.conf | tr -d ')')
    cat >> $SLURM_ACCOUNTING_CONFIG_FILE << EOL
# ACCOUNTING
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=$DBD_HOST
AccountingStoragePort=6819
AccountingStorageTRES=gres/gpu
GresTypes=gpu
EOL
}

main() {
  echo "[INFO]: Start configuration for SLURM accounting."

  # Start mariadb and check status
  systemctl start mariadb
  systemctl status mariadb

  setup_mariadb
  create_slurm_database

  create_slurmdbd_config
  add_accounting_to_slurm_config

  systemctl enable --now slurmdbd

  # validate_slurm_accounting
  echo "[INFO]: Completed configuration for SLURM accounting."
}

main "$@"
