IMAGE=registry.gitlab.aws.dev/smml/benchmarking/tools/preflight
DEPLOY_IMAGE=deploy
: "${AWS_REGION:=us-east-1}"
container_build:
	docker build -t ${IMAGE} ./preflight
container_run:
	docker run --privileged --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs3 --gpus=all ${IMAGE}
container: container_build container_run


ami_pcluster_cpu:
	packer build -only 'aws-pcluster-cpu.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee aws-pcluster-cpu_ami.log
ami_pcluster_gpu:
	packer build -only 'aws-pcluster-gpu.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee aws-pcluster-gpu_ami.log
ami_pcluster_neuron:
	packer build -only 'aws-pcluster-neuron.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee aws-pcluster-gpu_ami.log
ami_base:
	packer build -only 'aws-base-gpu.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee base-gpu_ami.log
ami_dlami_gpu:
	packer build -only 'aws-dlami-gpu.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee aws-dlami-gpu_ami.log
ami_dlami_neuron:
	packer build -only 'aws-dlami-neuron.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee aws-dlami-neuron_ami.log
ami_eks_gpu:
	packer build -only 'aws-eks-gpu.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee aws-eks-gpu_ami.log
ami:
	packer build -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee aws-eks-gpu_ami.log

ami_example:
	cd preflight/example_ami && packer build -color=true -var-file variables.json ami.json | tee log
deploy_build:
	docker build -t ${DEPLOY_IMAGE} ./test
deploy: deploy_build
	docker run -v ${HOME}/.aws:/root/.aws:ro -v ${shell pwd}/test:/tmp/test ${DEPLOY_IMAGE} pcluster create-cluster -n test-ami -r ${AWS_REGION} -c /tmp/test/cluster.yaml --rollback-on-failure false
