#!/bin/bash

# Check if a bucket name is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <bucketname>"
    exit 1
fi

# Define variables
bucketname="$1"
mount_dir="/mnt/$bucketname"
mountpoint_install_dir="/opt/mountpoint-s3"
bash_script="/usr/local/bin/mount-s3.sh"
systemd_service="/etc/systemd/system/mount-s3.service"

# Step 1: Create the bash script for mounting the S3 bucket
sudo tee $bash_script > /dev/null <<EOF
#!/bin/bash

bucketname="$bucketname"
mount_dir="$mount_dir"
mountpoint_install_dir="$mountpoint_install_dir"

# Ensure mountpoint directory exists
[ ! -d "\$mountpoint_install_dir" ] && sudo mkdir -p "\$mountpoint_install_dir"

# Install mount-s3 if not installed
if ! command -v mount-s3 &> /dev/null; then
    sudo wget -q -P "\$mountpoint_install_dir" https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
    sudo apt-get install -y "\$mountpoint_install_dir/mount-s3.deb"
fi

# Enable user_allow_other in /etc/fuse.conf if needed
sudo sed -i 's/^#user_allow_other/user_allow_other/' /etc/fuse.conf

# Create mount directory
[ ! -d "\$mount_dir" ] && sudo mkdir -p "\$mount_dir"

# Mount S3 bucket
sudo mount-s3 --allow-other "\$bucketname" "\$mount_dir" 2>&1 | tee /var/log/mount-s3.log

# Verify mount success
if mountpoint -q "\$mount_dir"; then
    echo "Successfully mounted \$bucketname to \$mount_dir"
else
    echo "Failed to mount \$bucketname"
    exit 1
fi
EOF

# Make the bash script executable
sudo chmod +x $bash_script
echo "Created and set executable permissions for $bash_script"

# Step 2: Create the systemd service file
sudo tee $systemd_service > /dev/null <<EOF
[Unit]
Description=Mount S3 Bucket using mount-s3 (fuse)
After=network.target

[Service]
Type=oneshot
ExecStart=$bash_script
RemainAfterExit=true
ExecStop=/bin/fusermount -u $mount_dir

[Install]
WantedBy=multi-user.target
EOF

echo "Created systemd service at $systemd_service"

# # Step 3: Reload systemd to pick up the new service
sudo systemctl daemon-reload
echo "Reloaded systemd"

# Step 4: Start the service
sudo systemctl start mount-s3.service
echo "Started the mount-s3 service"

# Step 5: Enable the service to start at boot
sudo systemctl enable mount-s3.service
echo "Enabled the mount-s3 service to start on boot"
