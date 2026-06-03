ENROOT_IMAGE=llm-foundry

all: build clean import

build:
	docker build -t ${ENROOT_IMAGE} -f 0.llm-foundry.Dockerfile .

clean:
	-rm ${ENROOT_IMAGE}.sqsh

import:
	enroot import -o ${ENROOT_IMAGE}.sqsh dockerd://llm-foundry:latest
