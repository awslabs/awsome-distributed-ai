ARG REGISTRY
ARG IMAGE
ARG TAG

FROM ${REGISTRY}${IMAGE}:${TAG}

# Install MTC dependencies
RUN pip install --no-cache-dir \
    amzn-sagemaker-checkpointing \
    s3torchconnector \
    tenacity \
    boto3

# Create checkpoint log directory
RUN mkdir -p /var/log/sagemaker_checkpointing

# The MTC library writes logs to /var/log/sagemaker_checkpointing
# Make sure this directory is writable
RUN chmod 755 /var/log/sagemaker_checkpointing
