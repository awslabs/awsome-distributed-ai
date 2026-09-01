#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


cluster_name=${1:-ml-cluster}

# grab head node instance id
instance_id=$(pcluster describe-cluster -n ${cluster_name} | jq '.headNode.instanceId' | tr -d '"')
os=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$instance_id" "Name=key,Values=parallelcluster:attributes" | jq '.Tags[0].Value' | tr -d '"' | awk -F ',' '{print $1}')

if [ "$os" = 'ubuntu2004' ]; then 
    user='ubuntu'
elif [ "$os" = 'ubuntu2204' ]; then 
    user='ubuntu'
elif [ "$os" = 'alinux2' ]; then
    user='ec2-user'
fi

echo -e "Instance Id: $instance_id"
echo -e "Os: $os"
echo -e "User: $user"

echo -e "Add the following to your ~/.ssh/config to easily connect:"
echo "
cat <<EOF >> ~/.ssh/config
Host ${cluster_name}
  User ${user}
  ProxyCommand sh -c \"aws ssm start-session --target ${instance_id} --document-name AWS-StartSSHSession --parameters 'portNumber=%p'\"
EOF

Add your ssh keypair and then you can do:

$ ssh ${cluster_name}
"

echo "Connecting to $cluster_name..."
aws ssm start-session --target $instance_id