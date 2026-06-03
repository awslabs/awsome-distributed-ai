
# Deploy HyperPod Infrastructure using CloudFormation
🚨 We recommend following the official [Amazon SageMaker HyperPod EKS Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/2433d39e-ccfe-4c00-9d3d-9917b729258e/en-US) to deploy clusters, which contains detailed instructions and latest best-practices.

As depicted below, the workshop infrastructure can be deployed using a series of nested CloudFormation stacks, each of which is responsible for deploying different aspects of a full HyperPod cluster environment.

<img src="./../cfn-templates/nested-stack-modules.png" width="50%"/>

If you wish to create a new HyperPod cluster **without reusing any pre-existing cloud resources**, you may deploy the main stack by clicking the button below and keeping all of the sub-stacks enabled, just be sure to check the default parameter values, including the `AvailabilityZoneId`, which should correspond to the location of your accelerated compute capacity and must be a valid Availability Zone ID for your target region. The default value `usw2-az2` is valid only for `us-west-2` region. 

Additionally, if you opted to deploy the [SageMaker Code Editor Stack](../cfn-templates/sagemaker-studio-stack.yaml), be sure to use the same `ResourceNamePrefix` and set `UsingSMCodeEditor` to `true` so that an EKS access entry will be created for the IAM Role that Code Editor uses. 

[<kbd> <br> 1-Click Deploy 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home?#/stacks/quickcreate?templateURL=https://ws-assets-prod-iad-r-pdx-f3b3f9f1a7d6a3d0.s3.us-west-2.amazonaws.com/2433d39e-ccfe-4c00-9d3d-9917b729258e/main-stack.yaml&stackName=hyperpod-eks-main-stack)

If, however, you wish to reuse existing cloud resources, like an existing VPC or EKS cluster for example, simply disable the corresponding sub-stack and supply the ID and or name of the resource your wish to reuse. The diagram above depicts the resource IDs / Names that each sub-stack expects to be supplied, either automatically by a sibling stack, or manually by you. 

Click on the sections below for more details on the respective sub-stacks: 

<details>
<summary>VPCStack</summary>
This stacks creates a VPC with a highly available architecture across two Availability Zones with public subnets and NAT Gateways for outbound internet connectivity.

<sp></sp>

Resources Created:
<ul>
    <li>VPC</li>
    <li>Internet Gateway</li>
    <li>2 Public Subnets</li>
    <li>2 NAT Gateways</li>
    <li>Public Route Table</li>
</ul>

Default Parameter Values:
<ul>
    <li>VpcCIDR: 10.192.0.0/16</li>
    <li>PublicSubnet1CIDR: 10.192.10.0/24</li>
    <li>PublicSubnet2CIDR: 10.192.11.0/24</li>
</ul>


Parameter Values Required if Disabled:
<ul>
    <li>VpcId - Used by PrivateSubnetStack, SecurityGroupStack, EKSClusterStack, and S3BucketStack</li>
    <li>NatGatewayId - Used by PrivateSubnetStack</li>
</ul>
</details>

---

<details>
<summary>PrivateSubnetStack</summary>
This stack creates a private subnet designed for SageMaker HyperPod cross-account ENIs, with proper routing through a NAT Gateway for outbound internet access.

<sp></sp>

Resources Created:
<ul>
    <li>Secondary CIDR Block</li>
    <li>Private Subnet</li>
    <li>Private Route Table</li>
</ul>

Default Parameter Values
<ul>
    <li>PrivateSubnet1CIDR: 10.1.0.0/16</li>
    <li>AvailabilityZoneId: usw2-az2</li>
</ul>


Parameter Values Required if Disabled
<ul>
    <li>PrivateSubnetId - Used by HyperPodClusterStack</li>
    <li>PrivateRouteTableId - Used by S3BucketStack</li>
</ul>
</details>

---

<details>
<summary>SecurityGroupStack</summary>
This stack creates a security group configured with rules to allow FSx for Lustre communication along with intra-security group communication for EFA and outbound internet access. 

<sp></sp>

If you are **reusing an existing EKS cluster**, the `SecurityGroupStack` will reference the `SecurityGroupId` parameter to add the required rules to the security group of that cluster. Be sure to provide a valid reference to the target EKS security group using the `SecurityGroupId` parameter. 

You can find the EKS cluster security group by running the following command:
```bash
SECURITY_GROUP_ID=$(aws eks describe-cluster \
--name "$EKS_CLUSTER_NAME" \
--query "cluster.resourcesVpcConfig.clusterSecurityGroupId" \
--output text)

echo $SECURITY_GROUP_ID
```
Resources Created:
<ul>
    <li>Security Group (Conditional, if creating a new EKS clusters)</li>
    <li>Intra-Security Group Rules</li>
    <li>Outbound access to the Internet</li>
    <li>FSx for Lustre Rules (TCP port 988, 1018-1023)</li>
</ul>
Default Parameter Values:
<ul>
    <li>NA</li>
</ul>
Parameter Values Required if Disabled:
<ul>
    <li>SecurityGroupId - Used by EKSClusterStack and HyperPodClusterStack</li>
</ul>
</details>

---

<details>
<summary>EKSClusterStack</summary>
This stack creates an EKS cluster for use as a control plane interface for the HyperPod cluster.  

<sp></sp>

If you are **reusing an existing EKS cluster**, the `SecurityGroupStack` will reference the `SecurityGroupId` parameter to add the required rules to the security group of that cluster. Be sure to provide a valid reference to the target EKS security group using the `SecurityGroupId` parameter.  

You can find the EKS cluster security group by running the following command:
```bash
SECURITY_GROUP_ID=$(aws eks describe-cluster \
--name "$EKS_CLUSTER_NAME" \
--query "cluster.resourcesVpcConfig.clusterSecurityGroupId" \
--output text)

echo $SECURITY_GROUP_ID
```

Resources Created:
<ul>
    <li>2 Private Subnets</li>
    <li>IAM Cluster Role</li>
    <li>EKS Cluster</li>
    <li>EKS Add-ons - VPC CNI, kube-proxy, CoreDNS, Pod Identity Agent</li>
    <li>EKS Access Entry (Conditional) - If you deployed the [SageMaker Code Editor Stack](../cfn-templates/sagemaker-studio-stack.yaml), set the `UsingSMCodeEditor` parameter to `true` to enable the creation of this access entry.</li>
</ul>
Default Parameter Values:
<ul>
    <li>KubernetesVersion: 1.30</li>
    <li>EKSClusterName: sagemaker-hyperpod-eks-cluster</li>
    <li>EKSPrivateSubnet1CIDR: 10.192.7.0/28</li>
    <li>EKSPrivateSubnet2CIDR: 10.192.8.0/28</li>
    <li>UsingSMCodeEditor: false</li>
</ul>
Parameter Values Required if Disabled:
<ul>
    <li>EKSClusterName - Used by HelmChartStack and HyperPodClusterStack</li>
</ul>
</details>

---

<details>
<summary>S3BucketStack</summary>
This stack creates an encrypted S3 bucket. This S3 bucket is used to store the [lifecycle scripts](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-lifecycle-best-practices.html) for your HyperPod cluster.

<sp></sp>

Resources Created:
<ul>
    <li>S3 Bucket</li>
</ul>
Default Parameter Values:
<ul>
    <li>NA </li>
</ul>
Parameter Values Required if Disabled:
<ul>
    <li>S3BucketName - Used by LifeCycleScriptStack and HyperPodClusterStack</li>
</ul>
</details>

---

<details>
<summary>S3EndpointStack</summary>
This stack creates a VPC endpoint for S3 to enable private connectivity for VPC-deployed HyperPod instance groups. 

<sp></sp>

Resources Created:
<ul>
    <li>VPC Endpoint for S3</li>
</ul>
Default Parameter Values:
<ul>
    <li>NA</li>
</ul>
Parameter Values Required if Disabled:
<ul>
    <li>NA</li>
</ul>
Note: If you opt to disable the S3BucketStack, please use the S3BucketName parameter to point to the existing S3 bucket you wish to use to store your lifecycle scripts. 
</details>

---

<details>
<summary>LifeCycleScriptStack</summary>
This stack deploys an AWS Lambda function that creates a [default lifecycle script](https://github.com/awslabs/awsome-distributed-ai/blob/main/architectures/sagemaker-hyperpod-eks/LifecycleScripts/base-config/on_create.sh) and stores it in the referenced S3 bucket.

<sp></sp>

Resources Created:
<ul>
    <li>AWS Lambda Function</li>
    <li>Default Lifecycle Script</li>   
</ul>
Default Parameter Values:
<ul>
    <li>NA</li>
</ul>
Parameter Values Required if Disabled:
<ul>
    <li>NA</li>
</ul>
Note: If you disable this stack, you must manually upload the default lifecycle script into the target S3 bucket prior to deploying your HyperPod cluster.
</details>

---

<details>
<summary>SageMakerIAMRoleStack</summary>
This stack creates an [IAM role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-prerequisites-iam.html#sagemaker-hyperpod-prerequisites-iam-role-for-hyperpod) designed to allow your HyperPod cluster to run and communicate with the necessary AWS resources on your behalf. 

<sp></sp>

Resources Created:
<ul>
    <li>IAM Role</li>
</ul>
Default Parameter Values:
<ul>
    <li>NA</li>
</ul>
Parameter Values Required if Disabled:
<ul>
    <li>SageMakerIAMRoleName - Used by HyperPodClusterStack</li>
</ul>
Note: If you opt to manually create the necessary IAM role for your HyperPod cluster, be sure to follow [the documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-prerequisites-iam.html#sagemaker-hyperpod-prerequisites-iam-role-for-hyperpod). 
</details>

---

<details>
<summary>HelmChartStack</summary>

The HyperPod dependency [Helm charts](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-install-packages-using-helm-chart.html) need to be installed on your EKS cluster prior to kicking off the creation of a new HyperPod cluster. If you choose to disable this stack and you are reusing an existing EKS cluster, be sure to manually install the dependencies prior to deploying the main stack. If you choose to disable this stack but want to create a new EKS cluster using the EKSClusterStack, the HyperPodClusterStack will be automatically disabled as well to avoid any HyperPod cluster creation failures. After the main stack completes, you can then proceed to manually install the dependencies prior to kicking off the manual creation of your HyperPod cluster. 

This stack deploys an AWS Lambda function that automates the instillation of HyperPod dependencies through [Helm charts](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-install-packages-using-helm-chart.html) in the EKS cluster. 

<sp></sp>

Resources Created:
<ul>
    <li></li>
    <li>AWS Lambda Function</li>
    <li>EKS Access Entry (For the Lambda function)</li>
    <li>Kubernetes resources deployed by the HyperPod dependency [Helm charts](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-install-packages-using-helm-chart.html)</li>
</ul>

Default Parameter Values:
Note: These parameters should not need to change unless the location of the Helm charts changes. 
<ul>
    <li>HelmRepoUrl: sagemaker-hyperpod-cli</li>
    <li>HelmRepoPath: helm_chart/HyperPodHelmChart</li>
    <li>Namespace: kube-system</li>
    <li>HelmRelease: hyperpod-dependencies</li>
</ul>

Parameter Values Required if Disabled:
<ul>
    <li>NA</li>
</ul>
</details>

---

<details>
<summary>HyperPodClusterStack</summary>
This stack creates a new SageMaker HyperPod cluster with 3 configurable instance groups, one for general purpose nodes, one for accelerated nodes, and one for restricted nodes for training Amazon Nova models.

<sp></sp>

Resources Created:
<ul>
    <li>HyperPod Cluster</li>
    <li>General Purpose instance group (Optional, disable by setting `CreateGeneralPurposeInstanceGroup` to `false`)</li>
    <li>Accelerated instance group (Optional, disable by setting `CreateAcceleratedInstanceGroup` to `false`)</li>
    <li>Restricted instance group (Optional, disable by setting `CreateRestrictedInstanceGroup` to `false`)</li>
</ul>

Default Parameter Values:
<ul>

<li>HyperPod Cluster Parameters: 
<ul>
    <li>HyperPodClusterName: ml-cluster</li>    
    <li>NodeRecovery: Automatic</li> 
    <li>UseContinuousNodeProvisioningMode: true</li>   
</ul>
</li>

<li>Accelerated Instance Group Parameters:
<ul>
    <li>CreateAcceleratedInstanceGroup: true</li>
    <li>AcceleratedInstanceGroupName: accelerated-instance-group</li>
    <li>AcceleratedInstanceType: ml.g5.xlarge</li>
    <li>AcceleratedInstanceCount: 1</li>
    <li>AcceleratedEBSVolumeSize: 500</li>
    <li>AcceleratedThreadsPerCore: 1</li>
    <li>AcceleratedLifeCycleConfigOnCreate: on_create.sh</li>
    <li>EnableInstanceStressCheck: true</li>
    <li>EnableInstanceConnectivityCheck: true</li>
    <li>AcceleratedLifeCycleConfigOnCreate: on_create.sh</li>
    <li>AcceleratedTrainingPlanArn: None</li>
    <li>AcceleratedImageId: None</li>
</ul>
</li>

<li>General Purpose Instance Group Parameters:
<ul>
    <li>CreateGeneralPurposeInstanceGroup: false</li>
    <li>GeneralPurposeInstanceGroupName: general-purpose-instance-group</li>
    <li>GeneralPurposeInstanceType: ml.m5.2xlarge</li>
    <li>GeneralPurposeInstanceCount 1</li>
    <li>GeneralPurposeEBSVolumeSize: 500</li>
    <li>GeneralPurposeThreadsPerCore: 1</li>
    <li>GeneralPurposeLifeCycleConfigOnCreate: on_create.sh</li>
    <li>GeneralPurposeImageId: None</li>
</ul>
</li>

<li>Restricted Instance Group Parameters: 
<ul>
   <li>CreateRestrictedInstanceGroup: false</li>
   <li>RestrictedInstanceGroupName: restricted-instance-group</li>
   <li>RestrictedInstanceType: ml.g5.8xlarge</li>
   <li>RestrictedInstanceCount: 1</li>
   <li>RestrictedPerUnitStorageThroughput: 250</li>
   <li>RestrictedSizeInGiB: 1200</li>
   <li>RestrictedEBSVolumeSize: 500</li>
   <li>EnableRestrictedInstanceStressCheck: true</li>
   <li>EnableRestrictedInstanceConnectivityCheck: true</li>
   <li>RestrictedThreadsPerCore: 1</li>
   <li>RestrictedTrainingPlanArn: None</li>
</ul>
</li>

</ul>

Parameter Values Required if Disabled:
<ul>
    <li>NA</li>
</ul>
</details>

---

## How Nested CloudFormation Stacks Work:
As you can see in the [main-stack.yaml](./nested-stacks/main-stack.yaml) file, the [`AWS::CloudFormation::Stack`](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-cloudformation-stack.html)resources have a `TemplateURL` property that specifies the [S3 URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html#virtual-hosted-style-access) pointing to the target CloudFormation template. 

The `TemplateURL` property is configured to reference a regional mapping of S3 buckets, which by default points to an AWS owned S3 bucket which is used to host the CloudFormation templates for the [Amazon SageMaker HyperPod EKS Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/2433d39e-ccfe-4c00-9d3d-9917b729258e/en-US). Again, we recommend following the instruction in the workshop, but we've made the templates available here for your to modify and reuse as necessary to meet your specific needs. 

## How to Host the Nested CloudFormation Stacks In Your Own S3 Bucket:  

### [Prerequisite: Build the Helm Chart Injector](./helm-chart-injector/README.md)

---

### Upload the nested CloudFormation stacks to your S3 bucket
```bash 
BUCKET_NAME=<your-bucket-name-here> 

aws s3 cp /nested-stacks/ s3://$BUCKET_NAME --recursive
```
---

### Configure the Main Stack to use the correct parameters

When you deploy the [main-stack.yaml](./nested-stacks/main-stack.yaml) template, be sure to updates the following parameters: 
- `TemplateURL` - Update this to specify the URL of the S3 bucket where you've uploaded the CloudFormation stacks in your own AWS account. 
- `CustomResourceS3Bucket` - Update this to specify the URL of the S3 bucket where you've uploaded the [Helm Chart Injector](./helm-chart-injector/README.md) dependency files. 
    - `LayerS3Key` - Update this to specify the S3 key for the `layer.zip` file you uploaded to your S3 bucket. 
    - `FunctionS3Key` - Update this to specify the S3 key for the `function.zip` file you uploaded to your S3 bucket. 

See the official [Amazon SageMaker HyperPod EKS Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/2433d39e-ccfe-4c00-9d3d-9917b729258e/en-US) for a more detailed explanation of the other parameters used in the nested CloudFormation stacks. 

