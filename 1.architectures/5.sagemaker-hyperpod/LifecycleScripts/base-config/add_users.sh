#!/bin/bash
# Creates users from `shared_users.txt` file
# each line in the `shared_users.txt` file should be of the format:
# ```
# username1,uid1,/fsx/username1
# username2,uid2,/fsx/username2
# ```
# 
# The script should be run as root user
# see `shared_users_sample.txt` for an example

set -e
set -x

SHARED_USER_FILE="shared_users.txt"

# takes in username, uid and home directory as parameters
# if user with username and uid does not exists,
# creates user with uid and creates a home directory for user
create_user() {
  local username=$1
  local uid=$2
  local fsx_home=$3

  # check if username already exists
  if id -u "$username"  >/dev/null 2>&1; then
    echo "User $username already exists. Skipping..."
    return
  fi

  # check if uid already exists
  if getent passwd "$uid" >/dev/null 2>&1; then
    echo "UID $uid is already in use. Skipping adding user: $username..."
    return
  fi

  # Determine home directory based on OpenZFS filesystem availability
  if df -h | grep -q "/home"; then
    echo "OpenZFS is mounted at /home"
    local home="/home/$username"

    # Create user with OpenZFS home
    if useradd -m $username --uid $uid -d $home --shell /bin/bash; then
      echo "Created user $username with uid $uid and home $home."

      # Make sure fsxl directory still exists and is accessible
      sudo mkdir -p $fsx_home
      sudo chown $username:$username $fsx_home
    else
      echo "Failed to create user $username with uid $uid"
    fi
  else
    echo "OpenZFS is not mounted. Using FSxL file system"
    # create user with uid and directory
    if useradd -m $username --uid $uid -d $fsx_home --shell /bin/bash; then
      echo "Created user $username with uid $uid and home $home."
    else
      echo "Failed to create user $username with uid $uid"
    fi
  fi  
}

main() {
  if [[ ! -f $SHARED_USER_FILE ]]; then
    echo "Shared user file $SHARED_USER_FILE does not exist. Skipping adding users."
    exit 0
  fi

  if [[ ! -s $SHARED_USER_FILE ]]; then
    echo "Shared user file $SHARED_USER_FILE is empty. Skipping adding users."
    exit 0
  fi

  while IFS="," read -r username uid home; do
    echo "Requested create user: $username with uid: $uid and home directory: $home"
    create_user "$username" "$uid" "$home"
  done < $SHARED_USER_FILE
}

main "$@"

