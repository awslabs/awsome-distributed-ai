#!/bin/bash

# Check if WORKSPACEID argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <WORKSPACEID>"
    exit 1
fi

WORKSPACEID="$1"


# Retrieve IMDSv2 Token to fetch region of current EC2 Instance (Head Node)
echo "Retrieving IMDSv2 Token to fetch region of current EC2 Instance (Head Node)"
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)
REGION=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/placement/region)

# Retrieve AMPRemoteWriteURL from Output Tab of CloudFormation Stack
echo "Retrieving AMPRemoteWriteURL from Output Tab of CloudFormation Stack"
AMPREMOTEWRITEURL=https://aps-workspaces.$REGION.amazonaws.com/workspaces/$WORKSPACEID/api/v1/remote_write


# Retrieve compute nodes from scontrol
echo "Retrieving compute nodes from scontrol"
export COMPUTENODES=$(scontrol show nodes | awk '/NodeAddr/ {print $1}' | cut -d '=' -f 2 | paste -sd "," -)

# Function to generate target lines for a job
generate_targets() {
    local port="$1"
    local nodes="$2"
    IFS=',' read -r -a nodes_array <<< "$nodes"
    for node_ip in "${nodes_array[@]}"; do
        echo "          - '${node_ip}:${port}'"
    done
}

if command -v prometheus &>/dev/null; then
    echo "Prometheus is already installed. Skipping installation."
else
    # Retrieve the latest Prometheus version from GitHub releases
    # echo "Retrieving the latest Prometheus version..."
    # LATEST_VERSION=$(curl -s https://api.github.com/repos/prometheus/prometheus/releases/latest | grep -oP '"tag_name": "\K(.*?)(?=")' | sed 's/^v//')
    
    echo "Using pinned Prometheus version 2.55.1"
    LATEST_VERSION=2.55.1 

    # Check if the latest version retrieval was successful
    if [ -z "$LATEST_VERSION" ]; then
        echo "Error: Failed to retrieve the latest Prometheus version."
        exit 1
    fi

    echo "Latest Prometheus version: $LATEST_VERSION"

    # Construct the download URL with the correct version format
    DOWNLOAD_URL="https://github.com/prometheus/prometheus/releases/download/v$LATEST_VERSION/prometheus-$LATEST_VERSION.linux-amd64.tar.gz"

    # Download the latest Prometheus release tarball
    echo "Downloading Prometheus version $LATEST_VERSION from $DOWNLOAD_URL ..."
    wget --progress=dot:giga "$DOWNLOAD_URL"

    # Extract Prometheus
    echo "Extracting Prometheus"
    tar xvfz prometheus-*.tar.gz

    # Move to Prometheus directory
    cd prometheus-*-amd64

    # Move binaries to /usr/bin/
    echo "Moving Prometheus binaries to /usr/bin/"
    sudo mv prometheus /usr/bin/
    sudo mv promtool /usr/bin/

    # Create Prometheus config directory
    echo "Creating Prometheus config directory"
    sudo mkdir -p /etc/prometheus

    # Move prometheus.yml to config directory
    echo "Moving prometheus.yml to /etc/prometheus/"
    sudo mv prometheus.yml /etc/prometheus/prometheus.yml
fi

# Replace placeholders in the configuration template
echo "Replacing placeholders in the Prometheus configuration template"
sudo tee /etc/prometheus/prometheus.yml > /dev/null <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 15s

scrape_configs:
  - job_name: 'slurm_exporter'
    static_configs:
      - targets:
          - 'localhost:8080'
  - job_name: 'dcgm_exporter'
    static_configs:
      - targets:
$(generate_targets 9400 "$COMPUTENODES")
  - job_name: 'efa_node_exporter'
    static_configs:
      - targets:
$(generate_targets 9100 "$COMPUTENODES")

remote_write:
  - url: ${AMPREMOTEWRITEURL}
    queue_config:
      max_samples_per_send: 1000
      max_shards: 200
      capacity: 2500
    sigv4:
      region: ${REGION}
EOF

# Create Prometheus systemd service file
echo "Creating Prometheus systemd service file"
sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOF
[Unit]
Description=Prometheus Exporter

[Service]
Environment=PATH=/opt/slurm/bin:\$PATH
ExecStart=/usr/bin/prometheus --config.file=/etc/prometheus/prometheus.yml --enable-feature=agent --storage.agent.path="/opt/prometheus/data-agent"
Restart=on-failure
RestartSec=15
Type=simple

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable Prometheus service
echo "Reloading systemd and enabling Prometheus service"
sudo systemctl daemon-reload
sudo systemctl restart prometheus

echo "Prometheus setup completed successfully"
