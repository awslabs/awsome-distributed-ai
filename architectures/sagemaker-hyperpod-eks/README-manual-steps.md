
# (Optional) Manual Steps for SageMaker HyperPod 

The following is a reference for executing various operational steps manually as needed. 

If you opted to [Deploy HyperPod Infrastructure using CloudFormation](./cfn-templates/README.md) with the default settings, the dependencies Helm chart, HyperPod cluster, and lifecycle script have been automatically deployed for you. 

## Install Dependencies

The HyperPod team provides a Helm chart package, which bundles key dependencies and associated permission configurations. This package contains dependencies such as Health Monitoring Agent, Nvidia device plugins, EFA device plugin, Neuron Device plugin. Below steps shows how to install the helm chart

#### Clone the Repo 

```bash
git clone https://github.com/aws/sagemaker-hyperpod-cli.git
cd sagemaker-hyperpod-cli/helm_chart
```

#### Install the Helm Chart

Locally test the helm chart: 
```bash
helm lint HyperPodHelmChart
```
Update the dependencies: 
```bash 
helm dependencies update HyperPodHelmChart
```
Conduct a dry run: 
```bash 
helm install dependencies HyperPodHelmChart --dry-run
```
Deploy the helm chart: 
```bash 
helm install dependencies HyperPodHelmChart --namespace kube-system
```


## Create SageMaker HyperPod cluster

Now that we have all our infrastructure in place, we can create a cluster. 

We need to setup few environment variables required for creating cluster. You will need to set the below environment parameters accordingly as per your requirement. 

```bash
export ACCEL_INSTANCE_TYPE=ml.g5.12xlarge #change this
export AWS_REGION=us-west-2 #change this
export ACCEL_COUNT=1 #change this
export ACCEL_VOLUME_SIZE=500 #the size in GB of the EBS volume attached to the compute node.
export GEN_INTANCE_TYPE= ml.m5.2xlarge	#The general purpose compute instance type you want to use
export GEN_COUNT=1	#The number of general purpose compute nodes you want to deploy
export GEN_VOLUME_SIZE=500 #The size in GB of the EBS volume attached to the general purpose compute nodes
export NODE_RECOVEY=AUTOMATIC 

```

 If you have used the full deployment option while deploying cloud formation you can use the helper script([create_config.sh](./create_config.sh)) to retrieve all the required. 

 If you used Integrative Deployment Mode set the below parameters

```bash
export EKS_CLUSTER_ARN=<YOUR_EKS_CLUSTER_ARN_HERE>
export EKS_CLUSTER_NAME=<YOUR_EKS_CLUSTER_NAME_HERE>
```

 If you used minimal deployment option you will have to explicitly set the below environment variables 

```bash
export EKS_CLUSTER_ARN=<YOUR_EKS_CLUSTER_ARN_HERE>
export EKS_CLUSTER_NAME=<YOUR_EKS_CLUSTER_NAME_HERE>
export VPC_ID=<YOUR_VPC_ID_HERE>
export SUBNET_ID=<YOUR_SUBNET_ID_HERE>
export SECURITY_GROUP=<YOUR_SECURITY_GROUP_ID_HERE>
```

Once set you can run the create_config.sh to set all the required environment variables.

```bash
export STACK_ID=hyperpod-eks-full-stack # change this accordingly 
bash ./create_config.sh
source env_vars
```


## Lifecycle scripts

Lifecycle scripts tell SageMaker HyperPod how to setup your HyperPod cluster. You can use this to install any node level customizations needed for your cluster. We provide a [base configuration](./LifecycleScripts/base-config) to get started. Below is a brief description of what each script is doing.

| Script                       | Description                                                                                                                                    |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| on_create.sh                 | [required] dummy script that is needed to create cluster                                                                           |


For now, let's just use the base configuration provided. Upload the scripts to the bucket you created earlier.
```
aws s3 cp --recursive LifecycleScripts/base-config s3://${BUCKET_NAME}/LifecycleScripts/base-config
```

## Cluster configuration

Next we can configure our actual cluster. In this case, we are creating a cluster with 2 Instance Groups. One with ml.m5.2xlarge instance and one with ml.g5.12xlarge instance. 

>Note - You can modify the number of instance groups as per your requirement. It is not mandatory to have 2 instance groups for cluster creation.

Lets start by creating cluster-config.json using the below snippet that uses the environment variables. 

```json
cat > cluster-config.json << EOL
{
    "ClusterName": "ml-cluster",
    "Orchestrator": { 
      "Eks": 
      {
        "ClusterArn": "${EKS_CLUSTER_ARN}"
      }
    },
    "InstanceGroups": [
      {
        "InstanceGroupName": "worker-group-1",
        "InstanceType": "${ACCEL_INSTANCE_TYPE}",
        "InstanceCount": ${ACCEL_COUNT},
        "InstanceStorageConfigs": [
          {
            "EbsVolumeConfig": {
              "VolumeSizeInGB": ${ACCEL_VOLUME_SIZE}
            }
          }
        ],
        "LifeCycleConfig": {
          "SourceS3Uri": "s3://${BUCKET_NAME}",
          "OnCreate": "on_create.sh"
        },
        "ExecutionRole": "${EXECUTION_ROLE}",
        "ThreadsPerCore": 1,
        "OnStartDeepHealthChecks": ["InstanceStress", "InstanceConnectivity"]
      },
      {
        "InstanceGroupName": "worker-group-2",
        "InstanceType": "${GEN_INSTANCE_TYPE}",
        "InstanceCount": ${GEN_COUNT},
        "InstanceStorageConfigs": [
          {
            "EbsVolumeConfig": {
              "VolumeSizeInGB": ${GEN_VOLUME_SIZE}
            }
          }
        ],
        "LifeCycleConfig": {
          "SourceS3Uri": "s3://${BUCKET_NAME}",
          "OnCreate": "on_create.sh"
        },
        "ExecutionRole": "${EXECUTION_ROLE}",
        "ThreadsPerCore": 1
      }
    ],
    "VpcConfig": {
      "SecurityGroupIds": ["$SECURITY_GROUP"],
      "Subnets":["$SUBNET_ID"]
    },
    "NodeRecovery": "${NODE_RECOVERY}"
}
EOL
```

- You can configure up to 20 instance groups under the `InstanceGroups` parameter. 
- For `Orchestrator.Eks.ClusterArn`, specify the ARN of the EKS cluster you want to use as the orchestrator. 
- For `OnStartDeepHealthChecks`, add `InstanceStress` and `InstanceConnectivity` to enable deep health checks. 
- For `NodeRecovery`, specify `Automatic` to enable automatic node recovery. HyperPod replaces or reboots instances (nodes) that fail the basic health or deep health checks (when enabled). 
- For the `VpcConfig` parameter, specify the information of the VPC used in the EKS cluster. The subnets must be private


## Launch a new cluster

Now that everything is in place, we can launch our cluster with the below command.


```bash
aws sagemaker create-cluster \
    --cli-input-json file://cluster-config.json \
    --region $AWS_REGION
```

You can see the current state of the cluster with

```bash
aws sagemaker list-clusters \
 --output table \
 --region $AWS_REGION
```

You'll see output similar to the following:

```
-------------------------------------------------------------------------------------------------------------------------------------------------
|                                                                 ListClusters                                                                  |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
||                                                              ClusterSummaries                                                               ||
|+----------------------------------------------------------------+----------------------+----------------+------------------------------------+|
||                           ClusterArn                           |     ClusterName      | ClusterStatus  |           CreationTime             ||
|+----------------------------------------------------------------+----------------------+----------------+------------------------------------+|
||  arn:aws:sagemaker:us-west-2:xxxxxxxxxxxx:cluster/uwme6r18mhic |  ml-cluster          |  Creating     |  2024-07-11T16:30:42.219000-04:00   ||
|+----------------------------------------------------------------+----------------------+----------------+------------------------------------+|
```
## Deleting your HyperPod cluster

When you're done with your HyperPod cluster, you can delete it down with

```bash
aws sagemaker delete-cluster --cluster-name ml-cluster --region $AWS_REGION
```