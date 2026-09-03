#!/bin/bash
set -e

sudo touch /tmp/env_vars

echo "=== Initial Environment ==="
echo "Running as user: $(whoami)"
echo "User groups: $(groups)"
echo "Working directory: $(pwd)"
echo "Home directory: $HOME"
echo "PATH: $PATH"

# Function to check if AWS credentials are fully available
check_aws_credentials() 
{
    aws sts get-caller-identity &>/dev/null
    return $?
}

# Function to check if we can access the cluster
check_cluster_access() 
{
    local cluster_name=$1
    aws sagemaker describe-cluster --cluster-name "$cluster_name" &>/dev/null
    return $?
}

# Function to wait for services to be ready
wait_for_services() 
{
    local cluster_name=$1
    local max_attempts=30
    local wait_time=20
    local attempt=1

    echo "Waiting for AWS services to be fully available..."
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt of $max_attempts (waiting ${wait_time}s between attempts)"
        
        if check_aws_credentials && check_cluster_access "$cluster_name"; then
            echo "AWS services are ready!"
            return 0
        fi
        
        echo "AWS services not ready yet..."
        sleep $wait_time
        attempt=$((attempt + 1))
        wait_time=$((wait_time + 10))  # Increase wait time with each attempt
    done
    
    echo "Timed out waiting for AWS services"
    return 1
}

check_and_install_dependencies() 
{
    local max_attempts=50
    local attempt=1
    local all_installed=false

    local debian_packages=(
        "libglib2.0-dev"
        "libgtk2.0-dev"
        "libgtk-3-dev"
        "liblua5.3-dev"
        "libhttp-parser-dev"
        "libjson-c-dev"
        "pkg-config"
        "glib-2.0-dev"
    )

    echo "Packages currently installed (only checking relevant ones):"
    dpkg -l | grep -E 'glib|gtk|lua|http-parser|json'

    echo "Checking and installing required dependencies..."

    while [ $attempt -le $max_attempts ] && [ "$all_installed" = false ]; do
        all_installed=true

        if ! sudo apt-get update; then
            echo "WARNING: apt-get update failed. Waiting 10 seconds before retry..."
            sleep 10
            attempt=$((attempt + 1))
            continue
        fi

        for package in "${debian_packages[@]}"; do
            echo "Checking package: $package"
            if ! dpkg -l | grep "^ii.*$package"; then
                echo "Package $package not found. Installing..."
                if ! sudo apt-get install -y "$package"; then
                    echo "WARNING: Failed to install $package. Waiting 10 seconds before retry..."
                    sleep 10
                    all_installed=false
                    break
                fi
            fi
        done

        if [ "$all_installed" = false ]; then
            echo "WARNING: Attempt $attempt of $max_attempts: Some packages failed to install properly. Waiting 10 seconds before retry..."
            sleep 10
            attempt=$((attempt + 1))
        fi
    done

    if [ "$all_installed" = true ]; then
        echo "All required dependencies are installed."
        return 0
    else
        echo "ERROR: Failed to install all required dependencies after $max_attempts attempts."
        return 1
    fi
}

CLUSTER_NAME=$1
HEAD_NODE_NAME=$2

echo "export CLUSTER_NAME=$CLUSTER_NAME" >> env_vars
echo "export HEAD_NODE_NAME=$HEAD_NODE_NAME" >> env_vars

# Validate input parameters
if [ -z "$CLUSTER_NAME" ] || [ -z "$HEAD_NODE_NAME" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: $0 CLUSTER_NAME HEAD_NODE_NAME"
    exit 1
fi

# Wait for AWS services to be ready before proceeding
if ! wait_for_services "$CLUSTER_NAME"; then
    echo "Failed to verify AWS services availability"
    exit 1
fi

# Install all required dependencies
check_and_install_dependencies
sudo apt install -y vim git jq curl build-essential wget \
    munge libmunge-dev \
    libssl-dev libpam0g-dev \
    python3-dev python3-pip \
    pkg-config xxd \
    jq libcurl4-openssl-dev libtool libtool-bin libhdf5-dev iproute2

# Get all controller IDs as an array
CONTROLLER_IDS=($(aws sagemaker list-cluster-nodes \
    --cluster-name ${CLUSTER_NAME} \
    --query "ClusterNodeSummaries[?InstanceGroupName=='${HEAD_NODE_NAME}'].InstanceId" \
    --output text))

# Print the number of controllers found
echo "Found ${#CONTROLLER_IDS[@]} controller nodes"

# Arrays to store information for each controller
declare -a CONTROLLER_IPS=()
declare -a CONTROLLER_HOSTNAMES=()

# Iterate through each controller ID
for controller in "${CONTROLLER_IDS[@]}"; do
    echo "Processing controller: ${controller}"

    # Get controller info
    CONTROLLER_INFO=$(aws sagemaker describe-cluster-node \
        --cluster-name ${CLUSTER_NAME} \
        --node-id ${controller})

    # Extract IP and hostname
    CONTROLLER_IP=$(echo ${CONTROLLER_INFO} | jq -r '.NodeDetails.PrivatePrimaryIp')
    CONTROLLER_HOSTNAME=$(echo ${CONTROLLER_INFO} | jq -r '.NodeDetails.PrivateDnsHostname' | sed 's/.ec2.internal//')

    # Use array length for the index in env_vars
    current_index=${#CONTROLLER_IPS[@]}

    # Store in arrays
    CONTROLLER_IPS+=("${CONTROLLER_IP}")
    CONTROLLER_HOSTNAMES+=("${CONTROLLER_HOSTNAME}")

    echo "Controller Number: $((current_index + 1))"
    echo "Controller ${controller}:"
    echo "  IP: ${CONTROLLER_IP}"
    echo "  Hostname: ${CONTROLLER_HOSTNAME}"

    echo "export CONTROLLER_ID_${current_index}=${controller}" >> env_vars
    echo "export CONTROLLER_IP_${current_index}=${CONTROLLER_IP}" >> env_vars
    echo "export CONTROLLER_HOSTNAME_${current_index}=${CONTROLLER_HOSTNAME}" >> env_vars
done

# Get cluster ID
CLUSTER_ID=$(aws sagemaker describe-cluster --cluster-name ${CLUSTER_NAME} --query 'ClusterArn' --output text | cut -d'/' -f2)
echo "export CLUSTER_ID=$CLUSTER_ID" >> env_vars

# TODO: MAKE THIS BETTER :) 
# sleep 240
# echo "WAITING FOR SERVICES TO BE READY BEFORE INSTALLING SLURM"

# Install SSM plugin
sudo curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" -o "/tmp/session-manager-plugin.deb"
sudo dpkg -i /tmp/session-manager-plugin.deb

# Install SLURM client
SLURM_VERSION="24.11.3"
wget https://download.schedmd.com/slurm/slurm-${SLURM_VERSION}.tar.bz2
tar xvf slurm-${SLURM_VERSION}.tar.bz2
cd slurm-${SLURM_VERSION}

# Ensure correct permissions before building
sudo chown -R $(whoami):$(whoami) .
sudo chmod 755 .

# Create a temporary staging directory
STAGE_DIR="/tmp/slurm-stage"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

./configure --prefix=/usr/local
make -j$(nproc)

make DESTDIR="${STAGE_DIR}" install

# Copy over all the contents to the right dir
sudo cp -r ${STAGE_DIR}/usr/local/* /usr/local/

# Post Installation (based on warnings received)
sudo libtool --finish /usr/local/lib
sudo mkdir -p /usr/local/lib/slurm
sudo cp src/api/.libs/libslurmfull.* /usr/local/lib/slurm/
sudo ldconfig

# Create symbolic links to actual slurm plugin directory
sudo mkdir -p /opt/slurm/lib
sudo ln -sf /usr/local/lib/slurm /opt/slurm/lib/slurm

cd ..
rm -rf slurm-${SLURM_VERSION}* "${STAGE_DIR}"

# Create config directory and slurm.conf
sudo mkdir -p /usr/local/etc
sudo tee /usr/local/etc/slurm.conf << EOF
ClusterName=${CLUSTER_NAME}
EOF

# Add controllers 
for ((i=0; i<${#CONTROLLER_HOSTNAMES[@]}; i++)); do
    echo "SlurmctldHost=${CONTROLLER_HOSTNAMES[i]}(${CONTROLLER_IPS[i]})" | sudo tee -a /usr/local/etc/slurm.conf
done

# Continue with the rest of the configuration
sudo tee -a /usr/local/etc/slurm.conf << 'EOF'

# Authentication configuration
AuthType=auth/munge
CryptoType=crypto/munge
SlurmUser=slurm

# Communication ports
SlurmdPort=6818
SlurmctldPort=6817

# For interactive jobs from login node
LaunchParameters=use_interactive_step
EOF

# Create symbolic link to the config file
sudo mkdir -p /etc/slurm
sudo mkdir -p /opt/slurm/etc
sudo ln -sf /usr/local/etc/slurm.conf /etc/slurm/slurm.conf
sudo ln -sf /usr/local/etc/slurm.conf /opt/slurm/etc/slurm.conf

# Create required directories and set permissions
sudo mkdir -p /var/spool/slurm
sudo chmod 755 /var/spool/slurm

# Create slurm user if it doesn't exist
if ! id "slurm" &>/dev/null; then
    sudo useradd -r -s /bin/bash slurm
fi

# Set up MUNGE directories with correct permissions
sudo chmod 700 /etc/munge
sudo chmod 711 /var/lib/munge

# Set up permissions for slurm spool dir
sudo chown slurm:slurm /var/spool/slurm
sudo chmod 755 /var/spool/slurm

# Create SLURM log directory
sudo mkdir -p /var/log/slurm
sudo chown slurm:slurm /var/log/slurm
sudo chmod 755 /var/log/slurm

# Create SLURM run directory
sudo mkdir -p /var/run/slurm
sudo chown slurm:slurm /var/run/slurm
sudo chmod 755 /var/run/slurm

# Create state directory (if not exists)
sudo mkdir -p /var/spool/slurm/state
sudo chown -R slurm:slurm /var/spool/slurm/state
sudo chmod 755 /var/spool/slurm/state

echo "=== Pre-MUNGE Setup State ==="
echo "MUNGE key location: $(ls -l /etc/munge/munge.key 2>/dev/null || echo 'No munge key')"
echo "MUNGE directories:"
ls -ld /var/run/munge /etc/munge /var/log/munge 2>/dev/null || echo "Munge directories not found"

# Get MUNGE key from controller using AWS SSM Session Manager
echo "Getting MUNGE key from controller..."
TEMP_FILE=$(mktemp)
TEMP_FILE_2=$(mktemp)

# This still errors out with key:
# Z+T5PlNJGukOvrQgKZL6YvRjCmkuyTk0C3jpSFOZtaczWWtHUqCZWQttU/q31kYsbKZnLj0LSpHP7Oweu/jRiwC4ikVlA5ot89SWpXOlKu5vGG05j4CyESeqbd0X+DC1PrnmWRJm6XBQZ3F8250nUKB2x+F4pNZjVvHVOkzGz8nqkz2gX3YhWZHIjInPVnCYh6Qe1c3QjG6F0C9UsGq09dxt14G7EN3fUmvSj9G3tO5oAznYL1zezphySH0KnxtXWt6gTu8iNJbcaj8Ci9ZjWcdHopPZkXZrxJkVw4OiAlsvfqyvum4HLIjFpqGDn1zsCN3Qm8bXBRSZ3IfWusxsV9ZjLbwExQNPg4trmFIlLEG+CWsF03yEiKGmvRaxP907rg/UIOhbw3JGJJ9xfteoUj9A+lii6SFXzdpgQ6jPonQFjgqrOAx1VZwAaDZx+lgxXCTaASlMSp+L3YKF9fBO2y6zJd7VzRxJ3ydIEMaAbK7bVAaN862wLeKVyHRuzoxKsdoWxn3Q4mCwLCn/XdSrbA5RD4PUP6WjKUHVVx92BRuUgnaAKUxmzdFhgq0rdctHN7tZBdBU7DjuaFy7eJdUdGNPBTtVtoZCqTHo3w4qZGBRCnT52cszM1hAzebm0FZEpL8oNbACJCr/CtF/42buP2K+qzaqICoOR8ZiSImSko5hw0vU4/ODauSJ9oaDVvOZzBmNf4W2hh0fTTDwr7jgu+1/4j4Qg/qBA7n/HkiiMEPIKV+s9K65oRRvabSpZqOQE8CXM0AEvCwz7D53oWKqAJvu3l7rARpQMKN6UOWp0QKRULMrhfQmIQtcTpS0ohgwWZF5/0ZebtYlnmSKzNzD8DXLE2yP/J9MsHGzuJdRN2UkmE66lprEJRkauJNsOcHFlVQiKtR1O4rw725YbqMSH1eiTNbePmDuejWdRBWcaaj7Vj5NRd7tHoEVDegG1E5VneuubLQQ/HwlEj82LBq4ljIEn0AF8bRVz1xESBEOBYoAQjRB1pFL/vpfspll5cVn3Xl7HcADVsVe13ani9ehuKTLjbZPsp+Toe0qF1AZzbJ2Yn74eZu/ZGxNGsd4Ixhggvcm5g3x8Hv2Fa/d5PifH+iqaylAXOLP3mWNXQYtYhF216pW90COJJp/PZWKj/vIyvbqxSsV3HWFbBYO9eRZzMP4/h4SC0i0sKsYShhGd5lONxNWwBttP1OuAVCwvw8d4LCannot perform start session: EOF
# MAY NEED TO HAVE USER RUN THIS PART MANUALLY  

sleep 5

# Same MUNGE Keys
CONTROLLER_ID=${CONTROLLER_IDS[0]}

MUNGE_KEY_SIZE=$(aws ssm start-session \
   --target "sagemaker-cluster:${CLUSTER_ID}_${HEAD_NODE_NAME}-${CONTROLLER_ID}" \
   --document-name AWS-StartInteractiveCommand \
   --parameters '{"command":["\n\n sudo stat -c%s /etc/munge/munge.key \n\n"]}' | grep -v "session" | tr -d '[:space:]')

echo "Munge Key Size: ${MUNGE_KEY_SIZE} bytes"

MAX_ATTEMPTS=10
ATTEMPT=1
SUCCESS=false

while [ $ATTEMPT -le $MAX_ATTEMPTS ] && [ "$SUCCESS" = false ]; do
    echo "Attempt $ATTEMPT of $MAX_ATTEMPTS to fetch MUNGE key..."

    # Clear both temp files
    > "${TEMP_FILE}"
    > "${TEMP_FILE_2}"

    sleep 5

    # Get hexdump of MUNGE key from controller
    aws ssm start-session \
        --target "sagemaker-cluster:${CLUSTER_ID}_${HEAD_NODE_NAME}-${CONTROLLER_ID}" \
        --document-name AWS-StartInteractiveCommand \
        --parameters '{"command":["\n\n sudo hexdump -C /etc/munge/munge.key"]}' \
        > "${TEMP_FILE}" && sleep 5

    # Convert hexdump back to binary key
    echo "Installing MUNGE key attempt ${ATTEMPT}..."
    cat "${TEMP_FILE}" | grep "^[0-9a-f].*  |" | sed 's/^[0-9a-f]\{8\}  //' | cut -d'|' -f2 | tr -d '|\n' | sudo tee ${TEMP_FILE_2} > /dev/null

    # Check the size of the processed key
    CURRENT_SIZE=$(stat -c%s "${TEMP_FILE_2}")

    echo "Expected size: ${MUNGE_KEY_SIZE} bytes"
    echo "Current size: ${CURRENT_SIZE} bytes"

    if [ "$CURRENT_SIZE" -eq "$MUNGE_KEY_SIZE" ]; then
        echo "Size matches. Installing MUNGE key..."
        sudo cp "${TEMP_FILE_2}" /etc/munge/munge.key
        SUCCESS=true
        echo "Successfully installed MUNGE key"
    else
        echo "Size mismatch. Retrying..."
        ATTEMPT=$((ATTEMPT + 1))
        sleep 5
    fi
done

# Clean up temp files
rm "${TEMP_FILE}" "${TEMP_FILE_2}"

# Set correct permissions for MUNGE key
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key

sudo mkdir -p /run/munge
sudo chown munge:munge /run/munge
sudo chmod 755 /run/munge

echo "export MUNGE_KEY_PATH=/etc/munge/munge.key" >> env_vars

# Restart MUNGE service
# sudo systemctl stop munge
# sudo systemctl enable munge
# sudo systemctl start munge
sudo service munge stop

# Change MUNGE user and group IDs to match controller
sudo usermod -u 991 munge
sudo groupmod -g 991 munge
sudo usermod -u 992 slurm
sudo groupmod -g 992 slurm
# Fix ownership of all MUNGE-related files and directories
sudo chown -R munge:munge /etc/munge /var/run/munge /var/log/munge /var/lib/munge

sudo service munge start
sudo service munge status

echo "Munge IDs: $(id munge)"

# Ensure MUNGE socket is available at the standard path
sleep 2  # Give MUNGE time to create its socket
sudo ln -sf /var/run/munge/munge.socket.2 /var/run/munge/munge.socket

echo "=== Post-MUNGE Setup State ==="
echo "MUNGE process: $(pgrep -l munged || echo 'No munge process')"
echo "MUNGE socket: $(ls -l /var/run/munge/munge.socket 2>/dev/null || echo 'No socket')"
echo "MUNGE key permissions:"
sudo ls -l /etc/munge/munge.key
echo "Testing MUNGE:"
sudo -u munge munge -n || echo "Munge test failed"

# Test MUNGE
echo "Testing MUNGE..."
munge -n | unmunge

# Set SLURM_CONF environment variable
export SLURM_CONF=/usr/local/etc/slurm.conf
echo "export SLURM_CONF=$SLURM_CONF" >> env_vars


if [ "$SUCCESS" = false ]; then

    echo "
Failed to fetch and install correct-sized MUNGE key after ${MAX_ATTEMPTS} attempts.
Here are the manual steps you can try:

################################################################################
#                        Manual MUNGE Key Installation                          #
################################################################################

1. Source environment variables & create a temporary file for the MUNGE key:
   source env_vars
   TEMP_FILE=\$(mktemp)

2. Get MUNGE key hexdump:
   aws ssm start-session \\
       --target \"sagemaker-cluster:\${CLUSTER_ID}_\${HEAD_NODE_NAME}-\${CONTROLLER_ID_0}\" \\
       --document-name AWS-StartInteractiveCommand \\
       --parameters '{\"command\":[\"\n\n sudo hexdump -C /etc/munge/munge.key\"]}' \\
       > \"\${TEMP_FILE}\"

3. Convert hexdump to binary and install:
   cat \"\${TEMP_FILE}\" | grep \"^[0-9a-f].*  |\" | \\
       sed 's/^[0-9a-f]\\{8\\}  //' | \\
       cut -d'|' -f2 | \\
       tr -d '|\\n' | \\
       sudo tee /etc/munge/munge.key > /dev/null

4. Restart MUNGE service:
   sudo service munge restart

5. Verify cluster status:
   sinfo

6. Cleanup:
   rm \${TEMP_FILE}

sinfo should work now!
################################################################################
"

else
    # Test SLURM client
    echo "Testing Slurm configuration..."
    sinfo

    echo "======================================="
    echo "======================================="
    echo "SLURM is now configured! You can now interact with your cluster from your Studio environment!!"
    echo "======================================="
    echo "======================================="
fi


