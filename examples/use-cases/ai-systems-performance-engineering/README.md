# AI systems performance engineering on AWS

## Run the AIM347 workshop job

The AIM347 workshop trains SmolLM2-1.7B with FSDP2 on two `g7.48xlarge` nodes of an AWS PCS cluster and changes one setting per stage: NCCL over TCP sockets or EFA, eager or SDPA attention, and no saving, synchronous saves or process-mode saves. [tools/README.md](tools/README.md) describes the tools that run and measure each stage, with their outputs, exit codes and result layout. To prepare a cluster for the job, run [facilitator/prepare-event.sh](facilitator/prepare-event.sh), which builds the training image from the `Dockerfile` and prepares the data, and then [facilitator/install-node-collectors.sh](facilitator/install-node-collectors.sh), which sets the PCS monitoring stack's EFA collector to 5 seconds and adds a Lustre client counter on each GPU node.
