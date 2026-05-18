# Amazon SageMaker Studio Integration with Amazon SageMaker HyperPod SLURM Cluster

This guide provides step-by-step instructions for setting up Amazon SageMaker Studio with Amazon SageMaker Hyperpod SLURM, including FSx Lustre storage configuration.

We will help set up your Studio environment so that:
1. You can use familiar environments such as JupyterLab and CodeEditor to interact with your SLURM SMHP cluster
2. You can access your cluster's FSxL file system from your JupyterLab/CodeEditor instance
3. Your CodeEditor/JupyterLab instance will essentially function as a login node to the SMHP SLURM cluster!

**Why login nodes?**
Login nodes allow users to login to the cluster, submit jobs, and view and manipulate data without running on the critical `slurmctld` scheduler node. This also allows you to run monitoring servers like [aim](https://github.com/aimhubio/aim), [Tensorboard](https://www.tensorflow.org/tensorboard), or [Grafana/Prometheus](https://prometheus.io/docs/visualization/grafana/).

![SageMaker Studio with Hyperpod integration](/1.architectures/5.sagemaker-hyperpod/slurm-studio/media/01-studio-hyperpod-architecture.png)

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Cluster Setup](#cluster-setup)
3. [FSx for Lustre Configuration](#fsx-for-lustre-configuration)
5. [SageMaker Studio Domain Setup](#sagemaker-studio-domain-setup)
6. [SageMaker Studio IDE Configuration](#sagemaker-studio-ide-configuration)
7. [Monitor SLURM Installation](#monitor-slurm-installation)
8. [Pitfalls](#pitfalls)

## Prerequisites

Before starting, ensure you have:

- AWS CLI configured with appropriate permissions
- Access to AWS Management Console
- Familiarity with [SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html), [SLURM](https://slurm.schedmd.com/documentation.html), [SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html), and [FSxL](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html)

***

## Cluster Setup

To create an Amazon SageMaker HyperPod SLURM cluster, you can follow one of these steps:

1. Option 1: [Easy Cluster Setup](https://catalog.workshops.aws/sagemaker-hyperpod/en-US/01-cluster/option-a-easy-cluster-setup)
2. Option 2: [Manual Cluster Setup](https://catalog.workshops.aws/sagemaker-hyperpod/en-US/01-cluster/option-b-manual-cluster-setup)

***

## FSx for Lustre Configuration

A FSx for Lustre (FSxL) file system is created for you as part of the [cluster setup](#cluster-setup)! We will use this file system for both your SMHP cluster nodes and your Studio Domain.

You can move on to the next section.
***

## SageMaker Studio Domain Setup

You can deploy the CloudFormation template, which creates the following resources:

1. SageMaker Studio domain
2. Lifecycle configurations for installing necessary packages for Studio IDE, including SLURM. Lifecycle configurations will be created for both JupyterLab and Code Editor. We will set it up so that your CodeEditor/JupyterLab instance will essentially be configured as a login node for your SageMaker HyperPod cluster!
3. A Lambda function that:
    1. Associates the created `security-group-for-inbound-nfs` security group to the Studio domain
    2. Associates the `security-group-for-inbound-nfs` security group to the FSx for Lustre ENIs
    3. **Optional**: If  **SharedFSx** is set to **True**, creates the partition *shared* in the FSx for Lustre volume, and associates it to the Studio domain

![SageMaker Studio with Hyperpod integration](/1.architectures/5.sagemaker-hyperpod/slurm-studio/media/07-fsx-shared.png)

4. If **SharedFSx** is set to **False**, a Lambda function that:
    1. Creates the partition */{user_profile_name}*, and associates it to the Studio user profile
5. If **SharedFSx** is set to **False**, an Event bridge rule that invokes the previously defined Lambda function each time a new user is created. 

![SageMaker Studio with Hyperpod integration](/1.architectures/5.sagemaker-hyperpod/slurm-studio/media/08-fsx-partitioned.png)

You can deploy the stack via

 [<kbd> <br> 1-Click Deploy 🚀 <br> </kbd>](https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/quickcreate?templateURL=https://awsome-distributed-training.s3.amazonaws.com/templates/studio-slurm.yaml)

The CloudFormation template requires the following parameters:

1. `AdditionalUsers`: Your configured SLURM users (POSIX) that you want to give access to write to your Studio's file system space (comma separated). Make sure you configure users using instructions in [Multi-user](https://catalog.workshops.aws/sagemaker-hyperpod/en-US/05-advanced/01-multi-user) first. `ubuntu` is added by default, so you don't need to add it in.
2. `ExistingFSxLustreId`: Id of the created FSx for Lustre file system
3. `ExistingSubnetIds`: Dropdown menu for selecting the SMHP cluster **Private Subnet IDs**.
4. `ExistingVpcId`: Dropdown menu for selecting the SMHP cluster VPC
5. `HeadNodeName`: The name of your SMHP SLURM cluster's head node (default `controller-machine`)
6. `HyperPodClusterName`: The name of your SMHP SLURM cluster (default: `ml-cluster`)
7. `SecurityGroupId`: Id of the security group that allows communication with the HyperPod Slurm controller node (for MUNGE authentication)
***

## SageMaker Studio IDE Configuration

As an admin user, once your SageMaker Studio Domain is provisioned, you may go in and create users as you see fit.

> [!NOTE]  
> This step *DOES NOT* assume that you already have a Studio Domain. To create one, check out the next section titled **"SageMaker Studio Domain Setup"**.
![SageMaker Studio with Hyperpod integration](/1.architectures/5.sagemaker-hyperpod/slurm-studio/media/09-studio-user.png)


You can now select your preferred IDE from SageMaker Studio. 

![SageMaker Studio with Hyperpod integration](/1.architectures/5.sagemaker-hyperpod/slurm-studio/media/02-studio-home.png)

For the purpose of this workshop, we are going to create a Code Editor environment.

From the top-left menu:

1. Click on **Code Editor**
2. Click on **Create Code Editor Space**
3. Enter a name
4. Click on **Create Space**
5. From the **Attach custom filesystem - optional** dropdown menu, select the FSx for Lustre volume
6. From the **Lifecycle configuration** dropdown menu, select the available lifecycle configuration
 

![SageMaker Studio with Hyperpod integration](/1.architectures/5.sagemaker-hyperpod/slurm-studio/media/03-codeditor-fsx.png)

Click on **Run Space**. Wait until the space is created, then click **Open Code Editor**

To verify that your file system was mounted, you can check if you have a path mounted in the Code Editor space `custom-file-system/fsx_lustre/<FSX_ID>`:

![SageMaker Studio with Hyperpod integration](/1.architectures/5.sagemaker-hyperpod/slurm-studio/media/10-filesystem-check.png)


You can also run:
```bash
df -h
```

If you set `SharedFSx` to `False`, you can verify separate partitions for two users.
Example output from user1:
```
Filesystem                      Size  Used Avail Use% Mounted on
overlay                          37G  494M   37G   2% /
tmpfs                            64M     0   64M   0% /dev
tmpfs                           1.9G     0  1.9G   0% /sys/fs/cgroup
shm                             392M     0  392M   0% /dev/shm
/dev/nvme1n1                    5.0G  529M  4.5G  11% /home/sagemaker-user
/dev/nvme0n1p1                  180G   31G  150G  18% /opt/.sagemakerinternal
10.1.53.46@tcp:/ylacfb4v/aman1  1.2T  7.5M  1.2T   1% /mnt/custom-file-systems/fsx_lustre/fs-0104f3de83efe0f33
127.0.0.1:/                     8.0E     0  8.0E   0% /mnt/custom-file-systems/efs/fs-052756a07c3a5ba97_fsap-0b5e6e7c68f22fee3
tmpfs                           1.9G     0  1.9G   0% /proc/acpi
tmpfs                           1.9G     0  1.9G   0% /sys/firmware
```

Example output from user2:
```
Filesystem                      Size  Used Avail Use% Mounted on
overlay                          37G  478M   37G   2% /
tmpfs                            64M     0   64M   0% /dev
tmpfs                           1.9G     0  1.9G   0% /sys/fs/cgroup
shm                             392M     0  392M   0% /dev/shm
/dev/nvme0n1p1                  180G   31G  150G  18% /opt/.sagemakerinternal
/dev/nvme1n1                    5.0G  529M  4.5G  11% /home/sagemaker-user
127.0.0.1:/                     8.0E     0  8.0E   0% /mnt/custom-file-systems/efs/fs-052756a07c3a5ba97_fsap-0a323a3e5a27e1bdc
10.1.53.46@tcp:/ylacfb4v/aman2  1.2T  7.5M  1.2T   1% /mnt/custom-file-systems/fsx_lustre/fs-0104f3de83efe0f33
tmpfs                           1.9G     0  1.9G   0% /proc/acpi
tmpfs                           1.9G     0  1.9G   0% /sys/firmware
```

The difference here is the mountpoint for FSxl (`ylacfb4v`) has separate partitions set up. You can then `cd /mnt/custom-file-systems/fsx_lustre/fs-0104f3de83efe0f33` and write from each user and verify that the other user isn't able to see those files!

Alternatively, if you set `SharedFSx` to `True`, you can check the the mount using `df -h`, and it will show something like:
```
Filesystem                       Size  Used Avail Use% Mounted on
overlay                           37G  478M   37G   2% /
tmpfs                             64M     0   64M   0% /dev
tmpfs                            1.9G     0  1.9G   0% /sys/fs/cgroup
shm                              392M     0  392M   0% /dev/shm
/dev/nvme0n1p1                   180G   31G  150G  18% /opt/.sagemakerinternal
/dev/nvme1n1                     5.0G  529M  4.5G  11% /home/sagemaker-user
10.1.53.46@tcp:/ylacfb4v/shared  1.2T  7.5M  1.2T   1% /mnt/custom-file-systems/fsx_lustre/fs-0104f3de83efe0f33
127.0.0.1:/                      8.0E     0  8.0E   0% /mnt/custom-file-systems/efs/fs-0e16e272aba907ad3_fsap-08ae9b9f68be028d7
tmpfs                            1.9G     0  1.9G   0% /proc/acpi
tmpfs                            1.9G     0  1.9G   0% /sys/firmware
```
with the `/shared` partition.


***

## Monitor SLURM Installation
Once you create your JupyterLab/CodeEditor instance, it will kick off the LifeCycleConfiguration (LCC). We've configured the LCC so that:
1. It installs necessary packages and dependencies
2. Downloads a script to install SLURM and set up MUNGE authentication
3. Logs progress to a file on your CodeEditor/JupyterLab instance

Before being able to run SLURM commands, please wait until the LCC fully installs SLURM and configures your instance as a login node. You can monitor the progress in the logs. To find the log file, head over to **CloudWatch** --> **Logs** --> **Log Groups**. 

In the search box, search for **/aws/sagemaker/studio** and select it. You will be redirected to all the Log Streams under the `/aws/sagemaker/studio` log group.

Under **Log Streams**, search for **<your-domain-id>/j/CodeEditor/default/LifecycleConfigOnStart** (you can find the domain id from your CloudFormation stack outputs). In the logs, you will see
```
Starting background installation. Check /tmp/slurm_lifecycle_20250326_053740.log for progress...
Installation started in the background. Monitor the progress with:
tail -f /tmp/slurm_lifecycle_20250326_053740.log
```

Grab the `tail` command, and paste it onto your CodeEditor/JupyterLab terminal. You will see that SLURM is getting installed and configured. This process takes ~5-7 minutes, so go grab a cup of coffee!

You'll know that SLURM is installed when you see 
```
Testing Slurm configuration...
PARTITION     AVAIL  TIMELIMIT  NODES  STATE NODELIST
dev*             up   infinite      4   idle ip-10-1-4-244,ip-10-1-31-165,ip-10-1-52-212,ip-10-1-90-199
ml.c5.4xlarge    up   infinite      4   idle ip-10-1-4-244,ip-10-1-31-165,ip-10-1-52-212,ip-10-1-90-199
=======================================
=======================================
SLURM is now configured! You can now interact with your cluster from your Studio environment!!
=======================================
=======================================
```

## Pitfalls and known issues 
1. You can't run `srun`.

You can run all other slurm commands, including `sbatch`, `squeue`, and `sinfo`. However, `srun` requires [specific ports](https://slurm.schedmd.com/network.html#client) to be open for I/O, which isn't possible on Studio IDE containers today. As a workaround, if you MUST run `srun`, try
```bash
# Source environment variables, written by your LCC
source env_vars

# Run your srun command via ssm
aws ssm start-session \
    --target "sagemaker-cluster:${CLUSTER_ID}_${HEAD_NODE_NAME}-${CONTROLLER_ID}" \
    --document-name AWS-StartInteractiveCommand \
    --parameters '{
        "command":["srun -N 4 hostname"]
    }'
```
By using ssm, you are still using the controller machine to submit `srun` jobs to your cluster nodes.

We recommend running `sbatch` commands directly instead.

Example `sbatch` script:
```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/fsx/<partition_name>/slurm_%j.out
#SBATCH --error=/fsx/<partition_name>/slurm_%j.err
#SBATCH --nodes=1

echo "Testing write access to /fsx/<partition)_name>"
date
hostname
whoami
nvidia-smi
```
> [!TIP]
> **Choosing a directory for log/error files**
>
>  When creating `sbatch` files, make sure you specify your `--output` and `--error` paths to an fsx path that both your Studio User and SLURM user (specified in LCC) have permission to write to. The safest bet would be to specify `/fsx/<partition_name>/`, where `<partition_name>` will either be `shared` or your studio user name, depending on what you set for `SharedFSx`. The permissions are handled via an ACL and automatically done by the LCC scripts.

2. SLURM failed to set up

This is a rare occurrence, but it may happen because the MUNGE authentication key was incorrectly copied over from the controller machine. To remediate, you can follow the steps in the logs:
```
Here are the manual steps you can try:

################################################################################
#                        Manual MUNGE Key Installation                          #
################################################################################

1. Source environment variables & create a temporary file for the MUNGE key:
   source env_vars
   TEMP_FILE=\$(mktemp)

2. Get MUNGE key hexdump:
   aws ssm start-session \\
       --target \"sagemaker-cluster:\${CLUSTER_ID}_\${HEAD_NODE_NAME}-\${CONTROLLER_ID}\" \\
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
```

This project provides automated setup and configuration for integrating Amazon SageMaker Studio with SLURM (Simple Linux Utility for Resource Management) on Amazon SageMaker HyperPod clusters. It enables seamless distributed training management through SLURM within the familiar Amazon SageMaker Studio environment (via JupyterLab or CodeEditor).

The solution automates the process of setting up the SLURM client configuration, authentication, and connectivity between SageMaker Studio and HyperPod clusters. It handles installation of required dependencies, SLURM client software compilation, security configuration including MUNGE authentication, and proper directory/permission setup. This allows data scientists to leverage SLURM's powerful distributed training capabilities directly from their familiar SageMaker Studio environment.

## Repository Structure
```
.
└── 1.architectures/
    └── 5.sagemaker-hyperpod/
        └── slurm-studio/
            ├── slurm_lifecycle.sh    # Automated script for SLURM environment setup on SageMaker Studio
            └── studio-slurm.yaml     # CloudFormation template for SageMaker Studio Domain setup
```