
algorithm_name=sgl-dev-cu13
dockerfilename=dockerfile

export DOCKER_BUILDKIT=1

region=$(aws configure get region)
account=$(aws sts get-caller-identity --query Account --output text)

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin "763104351884.dkr.ecr.${region}.amazonaws.com"

aws ecr describe-repositories --region $region --repository-names "${algorithm_name}" > /dev/null 2>&1 || {
    echo "create repository:" "${algorithm_name}"
    aws ecr create-repository --region $region  --repository-name "${algorithm_name}" > /dev/null
}

docker build --pull -t ${algorithm_name} -f $dockerfilename .

timestamp=$(date +%Y%m%d%H%M%S)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${timestamp}"
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}

echo $fullname