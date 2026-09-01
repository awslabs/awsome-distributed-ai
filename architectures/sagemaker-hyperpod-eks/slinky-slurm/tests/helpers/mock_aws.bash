#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# AWS CLI mock for bats-core unit tests.
# Source this file and call mock_aws() to override the `aws` command
# with canned responses. Override per-test as needed.

mock_aws() {
    aws() {
        case "$*" in
            *"sts get-caller-identity"*"--query Account"*)
                echo "123456789012"
                ;;
            *"sts get-caller-identity"*)
                echo '{"UserId":"AIDEXAMPLE","Account":"123456789012","Arn":"arn:aws:iam::123456789012:user/test"}'
                ;;
            *"ec2 describe-availability-zones"*"--zone-names"*"us-west-2a"*)
                # Resolve AZ name to AZ ID
                echo "usw2-az2"
                ;;
            *"ec2 describe-availability-zones"*"--zone-names"*"us-west-2d"*)
                # Resolve AZ name to AZ ID (different AZ for override testing)
                echo "usw2-az4"
                ;;
            *"ec2 describe-availability-zones"*"--zone-names"*)
                # Unknown AZ name
                echo "An error occurred (InvalidParameterValue) when calling the DescribeAvailabilityZones operation" >&2
                return 1
                ;;
            *"ec2 describe-availability-zones"*)
                printf "usw2-az1\tusw2-az2\tusw2-az3\tusw2-az4"
                ;;
            *"ec2 describe-instance-types"*"g5.8xlarge"*)
                # ml.g5.8xlarge: 1 A10G GPU, EFA supported (1 interface)
                if [[ "$*" == *"--query"* ]]; then
                    echo '{"GpuInfo":{"Gpus":[{"Name":"A10G","Manufacturer":"NVIDIA","Count":1,"MemoryInfo":{"SizeInMiB":24576}}]},"NetworkInfo":{"EfaSupported":true,"EfaInfo":{"MaximumEfaInterfaces":1},"MaximumNetworkInterfaces":8},"NeuronInfo":null}'
                else
                    echo '{"InstanceTypes":[{"InstanceType":"g5.8xlarge","GpuInfo":{"Gpus":[{"Name":"A10G","Manufacturer":"NVIDIA","Count":1,"MemoryInfo":{"SizeInMiB":24576}}]},"NetworkInfo":{"EfaSupported":true,"EfaInfo":{"MaximumEfaInterfaces":1},"MaximumNetworkInterfaces":8},"NeuronInfo":null}]}'
                fi
                ;;
            *"ec2 describe-instance-types"*"p5.48xlarge"*)
                # ml.p5.48xlarge: 8 H100 GPUs, EFA supported (32 interfaces)
                if [[ "$*" == *"--query"* ]]; then
                    echo '{"GpuInfo":{"Gpus":[{"Name":"H100","Manufacturer":"NVIDIA","Count":8,"MemoryInfo":{"SizeInMiB":81920}}]},"NetworkInfo":{"EfaSupported":true,"EfaInfo":{"MaximumEfaInterfaces":32},"MaximumNetworkInterfaces":64},"NeuronInfo":null}'
                else
                    echo '{"InstanceTypes":[{"InstanceType":"p5.48xlarge","GpuInfo":{"Gpus":[{"Name":"H100","Manufacturer":"NVIDIA","Count":8,"MemoryInfo":{"SizeInMiB":81920}}]},"NetworkInfo":{"EfaSupported":true,"EfaInfo":{"MaximumEfaInterfaces":32},"MaximumNetworkInterfaces":64},"NeuronInfo":null}]}'
                fi
                ;;
            *"ec2 describe-instance-types"*"g6.12xlarge"*)
                # ml.g6.12xlarge: 4 L4 GPUs, EFA supported (1 interface)
                if [[ "$*" == *"--query"* ]]; then
                    echo '{"GpuInfo":{"Gpus":[{"Name":"L4","Manufacturer":"NVIDIA","Count":4,"MemoryInfo":{"SizeInMiB":24576}}]},"NetworkInfo":{"EfaSupported":true,"EfaInfo":{"MaximumEfaInterfaces":1},"MaximumNetworkInterfaces":8},"NeuronInfo":null}'
                else
                    echo '{"InstanceTypes":[{"InstanceType":"g6.12xlarge","GpuInfo":{"Gpus":[{"Name":"L4","Manufacturer":"NVIDIA","Count":4,"MemoryInfo":{"SizeInMiB":24576}}]},"NetworkInfo":{"EfaSupported":true,"EfaInfo":{"MaximumEfaInterfaces":1},"MaximumNetworkInterfaces":8},"NeuronInfo":null}]}'
                fi
                ;;
            *"ec2 describe-instance-types"*"trn1.32xlarge"*)
                # ml.trn1.32xlarge: No GPUs, 16 Trainium devices
                if [[ "$*" == *"--query"* ]]; then
                    echo '{"GpuInfo":null,"NetworkInfo":{"EfaSupported":true,"EfaInfo":{"MaximumEfaInterfaces":8},"MaximumNetworkInterfaces":32},"NeuronInfo":{"NeuronDevices":[{"Count":16,"Name":"Trainium","CoreInfo":{"Count":2,"Version":2},"MemoryInfo":{"SizeInMiB":32768}}]}}'
                else
                    echo '{"InstanceTypes":[{"InstanceType":"trn1.32xlarge","GpuInfo":null,"NetworkInfo":{"EfaSupported":true,"EfaInfo":{"MaximumEfaInterfaces":8},"MaximumNetworkInterfaces":32},"NeuronInfo":{"NeuronDevices":[{"Count":16,"Name":"Trainium","CoreInfo":{"Count":2,"Version":2},"MemoryInfo":{"SizeInMiB":32768}}]}}]}'
                fi
                ;;
            *"ec2 describe-instance-types"*"m5.xlarge"*)
                # ml.m5.xlarge: No GPUs, no EFA, no Neuron (CPU-only)
                if [[ "$*" == *"--query"* ]]; then
                    echo '{"GpuInfo":null,"NetworkInfo":{"EfaSupported":false,"MaximumNetworkInterfaces":4},"NeuronInfo":null}'
                else
                    echo '{"InstanceTypes":[{"InstanceType":"m5.xlarge","GpuInfo":null,"NetworkInfo":{"EfaSupported":false,"MaximumNetworkInterfaces":4},"NeuronInfo":null}]}'
                fi
                ;;
            *"ec2 describe-instance-types"*)
                # Unknown instance type — return error
                echo "An error occurred (InvalidParameterValue) when calling the DescribeInstanceTypes operation" >&2
                return 1
                ;;
            *"sagemaker describe-training-plan"*"test-plan"*)
                # Active training plan in us-west-2a
                echo '{"TrainingPlanArn":"arn:aws:sagemaker:us-west-2:123456789012:training-plan/test-plan","TrainingPlanName":"test-plan","Status":"Active","ReservedCapacitySummaries":[{"AvailabilityZone":"us-west-2a","InstanceType":"ml.p5.48xlarge","TotalInstanceCount":2,"Status":"Active","ReservedCapacityArn":"arn:aws:sagemaker:us-west-2:123456789012:reserved-capacity/test-rc"}],"TotalInstanceCount":2,"AvailableInstanceCount":2,"InUseInstanceCount":0,"TargetResources":["hyperpod-cluster"]}'
                ;;
            *"sagemaker describe-training-plan"*"az4-plan"*)
                # Active training plan in us-west-2d (different AZ for override testing)
                echo '{"TrainingPlanArn":"arn:aws:sagemaker:us-west-2:123456789012:training-plan/az4-plan","TrainingPlanName":"az4-plan","Status":"Active","ReservedCapacitySummaries":[{"AvailabilityZone":"us-west-2d","InstanceType":"ml.p5.48xlarge","TotalInstanceCount":2,"Status":"Active"}],"TotalInstanceCount":2,"AvailableInstanceCount":2,"InUseInstanceCount":0,"TargetResources":["hyperpod-cluster"]}'
                ;;
            *"sagemaker describe-training-plan"*"expired-plan"*)
                # Expired training plan
                echo '{"TrainingPlanArn":"arn:aws:sagemaker:us-west-2:123456789012:training-plan/expired-plan","TrainingPlanName":"expired-plan","Status":"Expired","ReservedCapacitySummaries":[{"AvailabilityZone":"us-west-2a","InstanceType":"ml.p5.48xlarge","TotalInstanceCount":2,"Status":"Expired"}],"TotalInstanceCount":2,"AvailableInstanceCount":0,"InUseInstanceCount":0,"TargetResources":["hyperpod-cluster"]}'
                ;;
            *"sagemaker describe-training-plan"*"failed-plan"*)
                # Failed training plan
                echo '{"TrainingPlanArn":"arn:aws:sagemaker:us-west-2:123456789012:training-plan/failed-plan","TrainingPlanName":"failed-plan","Status":"Failed","ReservedCapacitySummaries":[],"TotalInstanceCount":0,"AvailableInstanceCount":0,"InUseInstanceCount":0,"TargetResources":["hyperpod-cluster"]}'
                ;;
            *"sagemaker describe-training-plan"*"no-capacity-plan"*)
                # Scheduled plan with no reserved capacity
                echo '{"TrainingPlanArn":"arn:aws:sagemaker:us-west-2:123456789012:training-plan/no-capacity-plan","TrainingPlanName":"no-capacity-plan","Status":"Scheduled","ReservedCapacitySummaries":[],"TotalInstanceCount":0,"AvailableInstanceCount":0,"InUseInstanceCount":0,"TargetResources":["hyperpod-cluster"]}'
                ;;
            *"sagemaker describe-training-plan"*)
                # Unknown training plan
                echo "An error occurred (ResourceNotFound) when calling the DescribeTrainingPlan operation" >&2
                return 1
                ;;
            *"cloudformation validate-template"*"template-url"*"invalid-bucket"*)
                echo "An error occurred (ValidationError) when calling the ValidateTemplate operation: S3 error: Access Denied" >&2
                return 1
                ;;
            *"cloudformation validate-template"*)
                # Return parameter metadata matching the 40 params in fixture/params.json.
                # HyperPodClusterName and ResourceNamePrefix have no default (required).
                # All others have defaults.
                cat <<'VALIDATE_EOF'
{"Parameters":[{"ParameterKey":"HyperPodClusterName","NoEcho":false,"Description":"Name of the HyperPod cluster"},{"ParameterKey":"ResourceNamePrefix","NoEcho":false,"Description":"Prefix for resource names"},{"ParameterKey":"ResourceNameShortPrefix","DefaultValue":"slinky1eks","NoEcho":false,"Description":"Short prefix"},{"ParameterKey":"Stage","DefaultValue":"prod","NoEcho":false,"Description":"Deployment stage"},{"ParameterKey":"NodeRecovery","DefaultValue":"Automatic","NoEcho":false,"Description":"Node recovery mode"},{"ParameterKey":"NodeProvisioningMode","DefaultValue":"Continuous","NoEcho":false,"Description":"Provisioning mode"},{"ParameterKey":"AutoScalerType","DefaultValue":"Karpenter","NoEcho":false,"Description":"Auto scaler type"},{"ParameterKey":"HyperPodClusterRole","DefaultValue":"","NoEcho":false,"Description":"Cluster role ARN"},{"ParameterKey":"InstanceGroupSettings1","DefaultValue":"[]","NoEcho":false,"Description":"Instance group 1 JSON"},{"ParameterKey":"EnableObservabilityFeature","DefaultValue":"false","NoEcho":false,"Description":"Enable observability"},{"ParameterKey":"CreateHyperPodObservabilityRole","DefaultValue":"true","NoEcho":false,"Description":"Create observability role"},{"ParameterKey":"CreatePrometheusWorkspace","DefaultValue":"false","NoEcho":false,"Description":"Create Prometheus workspace"},{"ParameterKey":"CreateGrafanaWorkspace","DefaultValue":"false","NoEcho":false,"Description":"Create Grafana workspace"},{"ParameterKey":"CreateGrafanaRole","DefaultValue":"false","NoEcho":false,"Description":"Create Grafana role"},{"ParameterKey":"TaskGovernanceMetricLevel","DefaultValue":"ClusterLevel","NoEcho":false,"Description":"Task governance metric level"},{"ParameterKey":"NodeMetricLevel","DefaultValue":"ClusterLevel","NoEcho":false,"Description":"Node metric level"},{"ParameterKey":"ClusterMetricLevel","DefaultValue":"ClusterLevel","NoEcho":false,"Description":"Cluster metric level"},{"ParameterKey":"NetworkMetricLevel","DefaultValue":"ClusterLevel","NoEcho":false,"Description":"Network metric level"},{"ParameterKey":"TrainingMetricLevel","DefaultValue":"ClusterLevel","NoEcho":false,"Description":"Training metric level"},{"ParameterKey":"AcceleratedComputeMetricLevel","DefaultValue":"ClusterLevel","NoEcho":false,"Description":"Accelerated compute metric level"},{"ParameterKey":"ScalingMetricLevel","DefaultValue":"ClusterLevel","NoEcho":false,"Description":"Scaling metric level"},{"ParameterKey":"Logging","DefaultValue":"false","NoEcho":false,"Description":"Enable logging"},{"ParameterKey":"TieredStorageConfig","DefaultValue":"","NoEcho":false,"Description":"Tiered storage config"},{"ParameterKey":"CreateTaskGovernanceClusterPolicyStack","DefaultValue":"false","NoEcho":false,"Description":"Create task governance stack"},{"ParameterKey":"PriorityClasses","DefaultValue":"[]","NoEcho":false,"Description":"Priority classes JSON"},{"ParameterKey":"FairShare","DefaultValue":"{}","NoEcho":false,"Description":"Fair share config"},{"ParameterKey":"EnableHPInferenceFeature","DefaultValue":"true","NoEcho":false,"Description":"Enable inference feature"},{"ParameterKey":"EnableHPInferenceAddonFeature","DefaultValue":"false","NoEcho":false,"Description":"Enable inference addon"},{"ParameterKey":"VpcCIDR","DefaultValue":"10.0.0.0/16","NoEcho":false,"Description":"VPC CIDR block"},{"ParameterKey":"AvailabilityZoneIds","DefaultValue":"usw2-az2","NoEcho":false,"Description":"AZ IDs"},{"ParameterKey":"KubernetesVersion","DefaultValue":"1.33","NoEcho":false,"Description":"K8s version"},{"ParameterKey":"HelmOperators","DefaultValue":"","NoEcho":false,"Description":"Helm operators"},{"ParameterKey":"EnableHPTrainingOperatorFeature","DefaultValue":"false","NoEcho":false,"Description":"Enable training operator"},{"ParameterKey":"CreateFsxStack","DefaultValue":"true","NoEcho":false,"Description":"Create FSx stack"},{"ParameterKey":"StorageCapacity","DefaultValue":"1200","NoEcho":false,"Description":"FSx storage capacity"},{"ParameterKey":"PerUnitStorageThroughput","DefaultValue":"250","NoEcho":false,"Description":"FSx throughput"},{"ParameterKey":"DataCompressionType","DefaultValue":"LZ4","NoEcho":false,"Description":"FSx compression"},{"ParameterKey":"FileSystemTypeVersion","DefaultValue":"2.15","NoEcho":false,"Description":"FSx type version"},{"ParameterKey":"DeploymentType","DefaultValue":"SCRATCH_2","NoEcho":false,"Description":"FSx deployment type"},{"ParameterKey":"FsxAvailabilityZoneId","DefaultValue":"usw2-az2","NoEcho":false,"Description":"FSx AZ ID"}],"Description":"HyperPod EKS main stack","Capabilities":["CAPABILITY_IAM","CAPABILITY_NAMED_IAM","CAPABILITY_AUTO_EXPAND"],"CapabilitiesReason":"Required for IAM resources and transforms"}
VALIDATE_EOF
                ;;
            *"cloudformation create-stack"*)
                echo '{"StackId": "arn:aws:cloudformation:us-west-2:123456789012:stack/test-stack/guid-1234"}'
                ;;
            *"cloudformation wait"*)
                return 0
                ;;
            *"cloudformation describe-stacks"*"OutputEKSClusterName"*)
                echo "mock-eks-cluster"
                ;;
            *"cloudformation describe-stacks"*"OutputVpcId"*)
                echo "vpc-mock12345"
                ;;
            *"cloudformation describe-stacks"*"OutputPrivateSubnetIds"*)
                echo "subnet-mock1,subnet-mock2"
                ;;
            *"cloudformation describe-stacks"*"OutputSecurityGroupId"*)
                echo "sg-mock12345"
                ;;
            *"cloudformation describe-stacks"*)
                echo "mock-value"
                ;;
            *)
                echo "UNMOCKED AWS CALL: aws $*" >&2
                return 1
                ;;
        esac
    }
    export -f aws
}
