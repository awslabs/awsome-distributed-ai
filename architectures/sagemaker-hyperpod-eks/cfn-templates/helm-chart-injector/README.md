# Helm Chart Injector

The Helm Chart Injector is an AWS Lambda function that can be used as an [`AWS::CloudFormation::CustomResource`](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-cloudformation-customresource.html) to automatically install the required [Kubernetes packages](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-install-packages-using-helm-chart.html) for orchestrating HyperPod clusters with Amazon EKS.  

## Directory Structure

```bash
.
├── README.md                         <-- this instructions file
├── deploy.sh                         <-- main bash script to kick off a new build
├── run-docker-build.md               <-- sub-script that builds the Lambda layer using Docker
├── Dockerfile                        <-- used to deploy a local AL23 container for layer building
├── build-layer.sh                    <-- script to install kubectl, helm, and aws-iam-authenticator
├── package-function.sh               <-- sub-script to pip install lambda dependencies and zip it up
└── lambda_function                   <-- sub-directory for lambda function code an requirements
    └── lambda_function.py            <-- lambda function code
    └── requirements.txt              <-- lambda function requirements
```
The [Amazon SageMaker HyperPod EKS documentation](https://awslabs.github.io/ai-on-sagemaker-hyperpod/docs/category/eks-orchestration) maintains a copy of the `layer.zip` and `function.zip` files used to instantiate the Helm Chart Injector in an AWS owned S3 bucket, as configured using the `CustomResourceS3Bucket`, `LayerS3Key`, and `FunctionS3Key` parameters in the [main-stack.yaml](./../nested-stacks/main-stack.yaml) template. However, you can follow the steps below to build your own copy of the dependency files and host them in your own S3 bucket.

## How to Build the Helm Chart Injector to Host in Your Own S3 Bucket:
```bash 
## execute the main script
./deploy.sh 

## Set your S3 bucket name
BUCKET_NAME=<your-bucket-name-here> 

## Sync function.zip and lambda-layer.zip with your S3 bucket
aws s3 sync ./outputs s3://$BUCKET_NAME --delete
```

After completing these steps, you can proceed to deploy the [helm-chart-stack.yaml](./../nested-stacks/helm-chart-stack.yaml) template by itself, or as part of the nested stack configuration in the [main-stack.yaml](./../nested-stacks/main-stack.yaml) template. See the main [README.md](./../README.md) for details. 

> **IMPORTANT**: Be sure to update the following parameters to reference your S3 Bucket and the artifacts you uploaded there:    
- `CustomResourceS3Bucket`
- `LayerS3Key`
- `FunctionS3Key`

