# First, get the head pod name
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)

# Use RAY_DASHBOARD_PORT from env_vars, default to 8265 if not set
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

# Then run the commands
kubectl exec -it $HEAD_POD -- /bin/bash -c "
if [ ! -d \"verl\" ]; then
    git clone https://github.com/volcengine/verl
fi
cd verl && \
export VERL_HOME=/fsx/verl && \
export RAY_ADDRESS=\"http://localhost:${RAY_DASHBOARD_PORT}\" && \
export RAY_DATA_HOME=\"/fsx/verl\" && \
export OVERWRITE=1 && \
bash recipe/dapo/prepare_dapo_data.sh
"