#!/bin/bash

set -ex

LOG_FILE="/var/log/provision/provisioning.log"

# Function to log messages
logger() {
  echo "$@" | tee -a $LOG_FILE
}

# Function to add user associations to Slurm accounting
setup_user_associations() {
    logger "[INFO] Setting up user associations for Slurm accounting"
    
    # Wait for slurmdbd to be ready
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if systemctl is-active --quiet slurmdbd; then
            logger "[INFO] slurmdbd is active"
            sleep 5  # Give it a few more seconds to fully initialize
            break
        fi
        logger "[INFO] Waiting for slurmdbd to start... (attempt $((attempt+1))/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        logger "[ERROR] slurmdbd failed to start within timeout"
        return 1
    fi
    
    # Add associations for existing users
    logger "[INFO] Adding user associations to Slurm accounting"
    
    # First, ensure the cluster is added to the accounting database
    sacctmgr -i add cluster $(sacctmgr show cluster format=cluster --noheader | head -1) || true
    
    # Add root account if it doesn't exist
    sacctmgr -i add account root Description="Root Account" || true
    
    # Add ubuntu user to root account
    if id -u ubuntu >/dev/null 2>&1; then
        logger "[INFO] Adding ubuntu user to root account"
        sacctmgr -i add user ubuntu account=root || true
    fi
    
    # Add associations for users from shared_users.txt if it exists
    SHARED_USER_FILE="shared_users.txt"
    if [[ -f $SHARED_USER_FILE ]] && [[ -s $SHARED_USER_FILE ]]; then
        while IFS="," read -r username uid home; do
            if id -u "$username" >/dev/null 2>&1; then
                logger "[INFO] Adding $username to root account"
                sacctmgr -i add user "$username" account=root || true
            fi
        done < $SHARED_USER_FILE
    fi
    
    logger "[INFO] User associations setup completed"
}

main() {
    setup_user_associations
}

main "$@"