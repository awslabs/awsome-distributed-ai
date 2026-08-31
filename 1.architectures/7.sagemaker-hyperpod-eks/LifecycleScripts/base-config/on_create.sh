#!/bin/bash

set -ex

LOG_FILE="/var/log/provision/provisioning.log"
mkdir -p "/var/log/provision"
touch "$LOG_FILE"

logger() {
  echo "$@" | tee -a "$LOG_FILE"
}

logger "[start] on_create.sh"

if [ -f "./on_create_main.sh" ]; then
  if ! bash ./on_create_main.sh >> "$LOG_FILE" 2>&1; then
    logger "[error] on_create_main.sh failed, waiting 60 seconds before exit, to make sure logs are uploaded"
    sync
    sleep 60
    logger "[stop] on_create.sh with error"
    exit 1
  fi
else
  logger "[warning] on_create_main.sh not found, skipping execution"
fi

logger "[stop] on_create.sh"
