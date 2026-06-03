IMAGE=node_exporter_efa
push:
	docker build -t ${IMAGE} .
run:
	docker run -d \
		--net="host" \
		--pid="host" \
		-v "/:/host:ro,rslave" \
		${IMAGE}:latest \
		--path.rootfs=/host
