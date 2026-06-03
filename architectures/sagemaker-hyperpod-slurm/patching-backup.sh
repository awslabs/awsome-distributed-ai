 #!/bin/bash

 set -e

 # Define files/directory to copy.
 LOCAL_ITEMS=(
     "/var/spool/slurmd"
     "/var/spool/slurmctld"
     "/etc/systemd/system/slurmctld.service"
     "/home/ubuntu/backup_slurm_acct_db.sql"
     # Add more items as needed
 )

 SQL_PASSWORD=''

 failed_commands=()

 is_slurmctld_was_running=false

 check_slurmctld_running() {
     if pgrep -x slurmctld > /dev/null; then
         return 0  # Return 0 if running.
     else
         return 1  # Return 1 if not running.
     fi
 }


 # Function to stop services
 stop_services() {
     check_slurmctld_running
     if [ $? -eq 0 ]; then
         is_slurmctld_was_running=true
         sudo systemctl stop slurmctld
         echo "slurmctld service stopped."
     else
         echo "Slurmctld not running ...."
     fi
 }

 # Function to start service
 start_services() {
     sudo systemctl daemon-reload
     echo "Ran systemctl daemon-reload"

     if $is_slurmctld_was_running; then
         sudo systemctl start slurmctld
         echo "slurmctld service started."
     fi
 }

 # Function to save slurm db to local disk
 save_mariadb() {
  sudo mysqldump --single-transaction slurm_acct_db > /home/ubuntu/backup_slurm_acct_db.sql
  echo "Saved slurm_acct_db"
 }

 # Function to restore slurm db from local disk
 restore_mariadb(){
     if [ -e "/home/ubuntu/backup_slurm_acct_db.sql" ]; then
         #sudo mysqldump slurm_acct_db < /home/ubuntu/backup_slurm_acct_db.sql > /dev/null 2>&1
         sudo mysql -u root --password="${SQL_PASSWORD}" slurm_acct_db < /home/ubuntu/backup_slurm_acct_db.sql
         echo "Restored slurm_acct_db"
         sudo rm /home/ubuntu/backup_slurm_acct_db.sql
     else
         echo "/home/ubuntu/backup_slurm_acct_db.sql does not exist"
     fi
 }

 # Function to check if slurm queue is empty
 check_squeue() {
     local squeue_count=$(squeue -h | wc -l)
     if [ "$squeue_count" -gt 0 ]; then
         echo "Error: squeue is not empty. Please wait for the jobs to complete before backup."
         exit 1
     fi
 }


 # Function to get the instance ID using IMDSv2
 get_instance_id() {
     local instance_id
     TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
     instance_id=$(curl -H   "X-aws-ec2-metadata-token: $TOKEN"  http://169.254.169.254/latest/meta-data/instance-id | tail -n 1)

     if [ -z "$instance_id" ]; then
         echo "Error: Unable to retrieve instance ID."
         exit 0
     fi

     echo "$instance_id"
 }

 # Function to check if s3 path exists, if not exit.
 check_s3_path_exits() {
     local s3_path="$1"

     # Remove the "s3://" prefix if present
     s3_path="${s3_path#s3://}"

     # Extract bucket name
     local bucket_name="${s3_path%%/*}"

     # Extract prefix (excluding the bucket name)
     local s3_prefix="${s3_path#*/}"

     local test_s3_path_result
     test_s3_path_result=$(aws s3api list-objects --bucket "${bucket_name}" --prefix "${s3_prefix}" 2>&1)

     if [ -z "$test_s3_path_result" ]; then
         echo "s3://$s3_path path does not exist. Skipping restore..."
         exit 0
     fi
 }


 # Function to save files/directores to S3
 backup() {
     S3_PATH="$(echo "$1" | sed 's:/*$::')"  # Remove trailing '/'

     local instance_id
     instance_id=$(get_instance_id)

     local full_s3_path="${S3_PATH}/${instance_id}/"

     stop_services
     save_mariadb

     # Iterating through the items list and saving to s3 path
     for item in "${LOCAL_ITEMS[@]}"; do
         echo $item
         if [ -f "$item" ]; then
             item_key="${item#/}"
             sudo aws s3 cp "$item" "$full_s3_path/$item_key"
             exit_code=$?
             if [ $exit_code -ne 0 ]; then
                 failed_commands+=("Error code: '$exit_code' command: sudo aws s3 cp '$item' '$full_s3_path/$item_key'")
             else
                 echo "Successfully backed up $item to $full_s3_path/$item_key"
             fi
         elif [ -d "$item" ]; then
         item_prefix="${item#/}"
             sudo aws s3 cp --recursive "$item" "$full_s3_path/$item_prefix"
             exit_code=$?
             if [ $exit_code -ne 0 ]; then
                 failed_commands+=("Error code: '$exit_code' command: sudo aws s3 cp --recursive '$item' '$full_s3_path/$item_prefix'")
             else
                 echo "Successfully backed up $item to $full_s3_path/$item_prefix"
             fi
         else
             failed_commands+=("The item '$item' does not exist.")
             echo "The item $item does not exist."
         fi
     done

     start_services

 }

 # Function to restore from S3
 restore() {
     S3_PATH="$(echo "$1" | sed 's:/*$::')"  # Remove trailing '/'

     local instance_id
     instance_id=$(get_instance_id)

     local full_s3_path="${S3_PATH}/${instance_id}/"

     check_s3_path_exits $full_s3_path

     stop_services

     # Iterating through the local items list and restoring from S3
     for item in "${LOCAL_ITEMS[@]}"; do
         if [ -d "$item" ]; then
             item_prefix="${item#/}"
             sudo aws s3 cp --recursive "$full_s3_path/$item_prefix" "$item"
             exit_code=$?
             if [ $exit_code -ne 0 ]; then
                 failed_commands+=("Error code: '$exit_code' command: sudo aws s3 cp --recursive '$full_s3_path/$item_prefix' '$item'")
             else
                 echo "Successfully restored $full_s3_path/$item_prefix to $item"
             fi
         else
             item_key="${item#/}"
             sudo aws s3 cp "$full_s3_path/$item_key" "$item"
             exit_code=$?
             if [ $exit_code -ne 0 ]; then
                 failed_commands+=("Error code: '$exit_code' command: sudo aws s3 cp '$full_s3_path/$item_key' '$item'")
             else
                 echo "Successfully restored $full_s3_path/$item_key to $item"
             fi
         fi
     done

     # restore saved slurmdb
     restore_mariadb

     start_services

 }

 print_report() {
     echo
     echo "---------------------------------"
     echo "           Summary                "
     echo "---------------------------------"
     echo
     if [ ${#failed_commands[@]} -gt 0 ]; then
         echo "Failed commands/errors:"
         for cmd in "${failed_commands[@]}"; do
             echo "$cmd"
         done
     else
         echo "All operations completed successfully."
     fi
     echo
     echo "---------------------------------"
     echo "       End of Summary             "
     echo "---------------------------------"

 }


 check_root() {
     # Check if the script is run as root (UID 0)
     if [ "$(id -u)" -ne 0 ]; then
         echo "Error: This script must be run as root."
         exit 1
     fi
 }

 ###########################################################################################

 # Actual script starts here.

 ###########################################################################################

 check_root

 # Validate command-line arguments
 if [ "$#" -lt 2 ]; then
     echo "Usage: $0 [--create <s3-path>] | [--restore] <s3-path>"
     exit 1
 fi

 case "$1" in
     --create)
         check_squeue
         backup $2
         ;;
     --restore)
         restore $2
         ;;
     *)
         echo "Invalid option: $1"
         exit 0
         ;;
 esac

 print_report
