#!/bin/bash

# must be run as sudo

set -eux

# FSx Lustre Endpoints
FSX_DNS_NAME="$1"
FSX_MOUNTNAME="$2"
MOUNT_POINT="$3"

# Track whether EFA was fully configured (set to true by configure_efa_lustre on success)
EFA_CONFIGURED=false

# Function for error handling
handle_error()
{
    local exit_code=$?
    echo "Error occurred in command: $BASH_COMMAND"
    echo "Exit code: $exit_code"
    echo "Exit logs:"
    sudo dmesg | tail -n 20
    echo "Mount status:"
    mount | grep lustre || true
    echo "LNet status:"
    sudo lctl list_nids || true
    exit $exit_code
}

trap handle_error ERR

# DEBUG: Verify parameters are set
verify_parameters()
{
    if [ -z "$FSX_DNS_NAME" ] || [ -z "$FSX_MOUNTNAME" ] || [ -z "$MOUNT_POINT" ]; then
        echo "Usage: $0 <fsx_dns_name> <fsx_mountname> <mount_point>"
        exit 1
    fi
}

# Print Lustre client version
print_lustre_version()
{
    echo "Lustre client version:"
    modinfo lustre | grep 'version:' | head -n 1 | awk '{print $2}'
}

# Verify if FSxL is created with EFA-enabled and if the FS is in the same AZ (cross AZ is not supported)
verify_fsx_efa_compatibility()
{
    local fsx_dns_name="$1"

    echo "[INFO] Verifying FSx EFA compatibility"

    # Extract FSx filesystem ID from DNS name
    local fsx_id=$(echo "$fsx_dns_name" | cut -d'.' -f1)

    # Get instance AZ
    local instance_az
    TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s --max-time 3 2>/dev/null)
    if [[ -n "$TOKEN" ]]; then
        instance_az=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s --max-time 3 http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null)
    else
        instance_az=$(curl -s --max-time 3 http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null)
    fi

    if [[ -z "$instance_az" ]]; then
        echo "[WARN] Could not determine instance AZ - proceeding without EFA verification"
        return 1
    fi

    # Get FSx filesystem details (EFA and Subnet details)
    local fsx_info
    if ! fsx_info=$(aws fsx describe-file-systems --file-system-ids "$fsx_id" --query 'FileSystems[0].{LustreConfiguration: LustreConfiguration, SubnetIds: SubnetIds}' --output json 2>/dev/null); then
        echo "[WARN] Could not describe FSx filesystem - proceeding without EFA verification"
        return 1
    fi

    # Get FSx AZ from subnet (To match FSx and instance AZ)
    local fsx_subnet=$(echo "$fsx_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['SubnetIds'][0])" 2>/dev/null)

    if [[ -z "$fsx_subnet" ]]; then
        echo "[WARN] Could not determine FSx subnet - proceeding without EFA verification"
        return 1
    fi

    local fsx_az=$(aws ec2 describe-subnets --subnet-ids "$fsx_subnet" --query 'Subnets[0].AvailabilityZone' --output text 2>/dev/null)

    if [[ "$instance_az" != "$fsx_az" ]]; then
        echo "[INFO] FSx filesystem is in different AZ ($fsx_az vs $instance_az) - EFA not supported cross-AZ"
        return 1
    fi

    # Check if FSx has EFA enabled (checking for EfaEnabled field and value. Currently, as observed, if FSx is created without EFA, the field doesn't exist in the describe call)
    local efa_enabled=$(echo "$fsx_info" | python3 -c "import sys, json; data=json.load(sys.stdin); lustre_config=data.get('LustreConfiguration', {}); print('FieldNotPresent' if 'EfaEnabled' not in lustre_config else lustre_config['EfaEnabled'])" 2>/dev/null)

    if [[ "$efa_enabled" != "True" ]]; then
        if [[ "$efa_enabled" == "FieldNotPresent" ]]; then
            echo "[INFO] FSx filesystem was not created with EFA enabled - skipping EFA configuration"
        else
            echo "[INFO] FSx filesystem has EFA disabled (EfaEnabled: $efa_enabled) - skipping EFA configuration"
        fi
        return 1
    fi

    echo "[INFO] FSx filesystem is EFA-compatible (same AZ: $instance_az, EfaEnabled: true)"
    return 0
}

# Configure EFA for Lustre if supported
configure_efa_lustre()
{
    echo "[INFO] Configuring EFA for FSx Lustre"

    # Check if instance has EFA drivers installed and configured
    if [[ -x "/opt/amazon/efa/bin/fi_info" ]]; then
        if /opt/amazon/efa/bin/fi_info -p efa >/dev/null 2>&1; then
            echo "[INFO] EFA provider detected successfully"
        else
            echo "[INFO] EFA provider not available - skipping EFA configuration"
            return 0
        fi
    else
        echo "[INFO] EFA tools not found - skipping EFA configuration"
        return 0
    fi

    # Verify FSx EFA compatibility
    if ! verify_fsx_efa_compatibility "$FSX_DNS_NAME"; then
        echo "[INFO] FSx not EFA-compatible - skipping EFA configuration"
        return 0
    fi

    echo "[INFO] EFA requirements met - proceeding with EFA configuration"
    echo "[INFO] - EFA provider: available"
    echo "[INFO] - FSx EFA enabled: yes"
    echo "[INFO] - Same AZ: yes"

    # Download EFA configuration script
    if ! ansible localhost -m ansible.builtin.get_url -a "url=https://docs.aws.amazon.com/fsx/latest/LustreGuide/samples/configure-efa-fsx-lustre-client.zip dest=/tmp/configure-efa-fsx-lustre-client.zip mode='0644'"; then
        echo "[ERROR] Failed to download EFA configuration script"
        return 1
    fi

    # Extract the zip file
    ansible localhost -m ansible.builtin.unarchive -a "src=/tmp/configure-efa-fsx-lustre-client.zip dest=/tmp remote_src=yes"

    # Make script executable and run it
    ansible localhost -b -m ansible.builtin.file -a "path=/tmp/configure-efa-fsx-lustre-client/setup.sh mode='0755'"
    ansible localhost -b -m ansible.builtin.command -a "/tmp/configure-efa-fsx-lustre-client/setup.sh"

    # Cleanup
    ansible localhost -m ansible.builtin.file -a "path=/tmp/configure-efa-fsx-lustre-client.zip state=absent"
    ansible localhost -m ansible.builtin.file -a "path=/tmp/configure-efa-fsx-lustre-client/setup.sh state=absent"

    EFA_CONFIGURED=true
    echo "[INFO] EFA configuration for FSx Lustre completed"
}

# Load lnet modules
load_lnet_modules()
{
  ansible localhost -b -m ansible.builtin.modprobe -a "name=lnet state=present"
  ansible localhost -b -m ansible.builtin.modprobe -a "name=lustre state=present"
  lctl network up || { echo "Error: Failed to bring up LNet network"; exit 1; }     # Simplifying: Instead of using ansible.builtin.shell
}

# Mount the FSx Lustre file system using Ansible
mount_fs() {
    local max_attempts=5
    local attempt=1
    local delay=5
    local test_file="$MOUNT_POINT/test_file_$(hostname)"

    echo "[INFO] Ensuring $MOUNT_POINT directory exists..."
    ansible localhost -b -m ansible.builtin.file -a "path=$MOUNT_POINT state=directory" || true

    echo "[INFO] Mounting FSx Lustre on $MOUNT_POINT..."
    echo "[INFO] Using test file: $test_file"

    while (( attempt <= max_attempts )); do
        echo "============================"
        echo "[INFO] Attempt $attempt of $max_attempts"
        echo "============================"

        echo "[STEP] Mounting FSx..."
        if ! ansible localhost -b -m ansible.posix.mount -a \
            "path=$MOUNT_POINT src=$FSX_DNS_NAME@tcp:/$FSX_MOUNTNAME fstype=lustre opts=noatime,flock,_netdev,x-systemd.automount,x-systemd.requires=network-online.target dump=0 passno=0 state=mounted"; then
            echo "[WARN] Mount command failed — retrying in $delay seconds"
            sleep "$delay"; ((attempt++)); continue
        fi

        echo "[STEP] Verifying mountpoint..."
        if ! ansible localhost -b -m ansible.builtin.command -a "mountpoint $MOUNT_POINT"; then
            echo "[WARN] Mountpoint verification failed — retrying in $delay seconds"
            sleep "$delay"; ((attempt++)); continue
        fi
        echo "[STEP] Triggering automount..."
        ls -la "$MOUNT_POINT" >/dev/null 2>&1 || true

        echo "[STEP] Testing file access (touch)..."
        if ! ansible localhost -b -m ansible.builtin.file -a "path=$test_file state=touch"; then
            echo "[WARN] Touch failed — retrying in $delay seconds"
            sleep "$delay"; ((attempt++)); continue
        fi

        echo "[STEP] Testing file access (delete)..."
        if ! ansible localhost -b -m ansible.builtin.file -a "path=$test_file state=absent"; then
            echo "[WARN] Delete failed — retrying in $delay seconds"
            sleep "$delay"; ((attempt++)); continue
        fi

        echo "[SUCCESS] FSx mount succeeded on attempt $attempt"
        return 0
    done

    echo "[ERROR] FSx mount failed after $max_attempts attempts"
    return 1
}

# Verify that EFA transport is actually carrying data to the OSS.
# Lustre MGS/MDS traffic always goes over ENA/TCP regardless of EFA configuration.
# Only bulk data I/O to the OSS uses EFA. If EFA is misconfigured (e.g. security groups,
# subnet issues), Lustre silently falls back to ENA/TCP with no errors — the only way
# to detect this is to generate OSS traffic and check the EFA interface counters.
verify_efa_transport()
{
    if [[ "$EFA_CONFIGURED" != "true" ]]; then
        echo "[INFO] EFA was not configured — skipping EFA transport verification"
        return 0
    fi

    echo "[INFO] Verifying EFA transport for FSx Lustre..."

    # Check that the EFA LNet network is present
    if ! lnetctl net show -v --net efa > /dev/null 2>&1; then
        echo "[ERROR] EFA LNet network not found despite EFA being configured"
        return 1
    fi

    # Capture baseline EFA counters before generating I/O
    local send_before recv_before
    send_before=$(lnetctl net show -v --net efa | awk '/send_count:/ {print $2}')
    recv_before=$(lnetctl net show -v --net efa | awk '/recv_count:/ {print $2}')

    echo "[INFO] EFA counters before I/O — send: $send_before, recv: $recv_before"

    # Generate bulk I/O to force traffic through the OSS (not just MGS/MDS metadata).
    # A 64MB write + read is sufficient to move the counters if EFA is active.
    local test_file="$MOUNT_POINT/.efa_verify_$(hostname)"
    dd if=/dev/zero of="$test_file" bs=1M count=64 oflag=direct 2>/dev/null || true
    sync
    dd if="$test_file" of=/dev/null bs=1M iflag=direct 2>/dev/null || true
    rm -f "$test_file"

    # Capture EFA counters after I/O
    local send_after recv_after
    send_after=$(lnetctl net show -v --net efa | awk '/send_count:/ {print $2}')
    recv_after=$(lnetctl net show -v --net efa | awk '/recv_count:/ {print $2}')

    echo "[INFO] EFA counters after I/O — send: $send_after, recv: $recv_after"

    # Verify that both send and recv counters increased — if either is flat,
    # data is not flowing over EFA and has silently fallen back to ENA/TCP.
    if [[ "$send_after" -le "$send_before" ]] || [[ "$recv_after" -le "$recv_before" ]]; then
        echo "[ERROR] ============================================="
        echo "[ERROR] EFA TRANSPORT VERIFICATION FAILED"
        echo "[ERROR] ============================================="
        echo "[ERROR] EFA was configured but data is NOT flowing over EFA interfaces."
        echo "[ERROR] Traffic is likely falling back to ENA/TCP."
        echo "[ERROR] Send count: $send_before -> $send_after"
        echo "[ERROR] Recv count: $recv_before -> $recv_after"
        echo "[ERROR] Troubleshooting:"
        echo "[ERROR]   - Verify security groups allow EFA traffic between nodes and FSx"
        echo "[ERROR]   - Confirm FSx filesystem and instance are in the same AZ"
        echo "[ERROR]   - Check that FSx was created with EFA enabled"
        echo "[ERROR] ============================================="
        return 1
    fi

    echo "[SUCCESS] EFA transport verified — data is flowing over EFA interfaces"
    echo "[INFO] Send count: $send_before -> $send_after (delta: $((send_after - send_before)))"
    echo "[INFO] Recv count: $recv_before -> $recv_after (delta: $((recv_after - recv_before)))"
    return 0
}



restart_daemon()
{
  ansible localhost -b -m ansible.builtin.systemd -a "daemon_reload=yes"
  ansible localhost -b -m ansible.builtin.systemd -a "name=remote-fs.target state=restarted"
  # Readable status check
  echo "Check status of fsx automount service..."
  systemctl status fsx.automount
}

main()
{
    verify_parameters
    echo "Mount_fsx called with fsx_dns_name: $FSX_DNS_NAME, fsx_mountname: $FSX_MOUNTNAME"
    echo "Using mount_point: $MOUNT_POINT"
    echo "LUSTRE CLIENT CONFIGURATION $(print_lustre_version)"
    configure_efa_lustre
    load_lnet_modules
    mount_fs || exit 1
    verify_efa_transport || exit 1
    restart_daemon
    echo "FSx Lustre mounted successfully to $MOUNT_POINT"
}

main "$@"
