all: build clean import

IMAGE=aws-megatron-lm

build:
	docker build -t ${IMAGE} -f ${IMAGE}.Dockerfile .

clean:
	-rm ${IMAGE}.sqsh

import:
	enroot import -o ${IMAGE}.sqsh  dockerd://${IMAGE}:latest