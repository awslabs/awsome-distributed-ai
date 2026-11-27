#!/usr/bin/env bash
# Prepare the lab on one cluster, as root on the login node inside an allocation of both GPU
# nodes: the training image (built from this checkout's Dockerfile), the datasets and the
# model, and the environment file that every tool reads. --base builds on that image
# reference in place of the Dockerfile's FROM tag. --with-serving also imports the vLLM image
# that tools/serve.sh runs. A rerun keeps the image and the data that exist and makes the rest.
#
# Usage: bash facilitator/prepare-event.sh --stage-dir DIR --slurm-version VER --instance-type TYPE
#            --grafana-url URL [--base IMAGE] [--with-serving]
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
stage_dir='' slurm_version='' instance_type='' grafana_url='' base='' serving=0
# The public vLLM 0.20.2 image (linux/amd64) that the serving path was validated with.
vllm_source=vllm/vllm-openai@sha256:68b773151407ca28c05479c02c0c08a573285ba197c8ff35d2103acd97aff78c
usage() {
    echo 'Usage: bash facilitator/prepare-event.sh --stage-dir DIR --slurm-version VER --instance-type TYPE --grafana-url URL [--base IMAGE] [--with-serving]' >&2
    exit 2
}
while (($#)); do
    case $1 in
        --with-serving) serving=1; shift; continue ;;
        --stage-dir) stage_dir=${2:-} ;;
        --slurm-version) slurm_version=${2:-} ;;
        --instance-type) instance_type=${2:-} ;;
        --grafana-url) grafana_url=${2:-} ;;
        --base) base=${2:-} ;;
        *) usage ;;
    esac
    shift 2 || usage
done
[[ -n $stage_dir && -n $slurm_version && -n $instance_type && -n $grafana_url ]] || usage
export PATH="/opt/aws/pcs/scheduler/slurm-$slurm_version/bin:$PATH"
image=$stage_dir/images/aim347-lab.sqsh
serving_image=$stage_dir/images/vllm-v0.20.2.sqsh
data=$stage_dir/data
install -d -m 0755 "$stage_dir/images" "$data" "$stage_dir/results"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
node_field() { scontrol show node -o "$1" | tr ' ' '\n' | sed -n "s/^$2=//p"; }
# Every CPU of the node for the build and the data steps: a step task otherwise gets one.
cpus=$(node_field "${nodes[0]}" CPUEfctv)
# The interface the second node uses to reach the first one carries Socket traffic.
nic=$(srun -N1 -n1 -w "${nodes[1]}" ip -j route get "$(node_field "${nodes[0]}" NodeAddr)" |
      python3 -c 'import json, sys; print(json.load(sys.stdin)[0]["dev"])')

echo 'prepare-event: waiting for the PCS monitoring install on both GPU nodes'
# PCS installs its monitoring stack on a GPU node as a NodeReady action, after the node
# already accepts jobs. The install restarts Docker and only then starts node-exporter, and
# that restart would end a build, so wait until node-exporter answers on both nodes.
srun -N2 --ntasks-per-node=1 bash -c '
    for _ in $(seq 180); do
        curl -sf -o /dev/null http://localhost:9100/metrics && docker info >/dev/null 2>&1 && exit 0
        sleep 10
    done
    echo "prepare-event: the PCS monitoring install has not finished on $(hostname) after 30 minutes;" \
        "read /var/log/amazon/pcs/lifecycle/actions/nodeReady/install-monitoring.log on that node" >&2
    exit 1'

# On the first node: put a Docker image into a squashfs file on the shared file system.
# Enroot works on the node's own disk. Usage: import_image DOCKER_IMAGE FILE
import_image() {
    # shellcheck disable=SC2016
    srun -N1 -n1 -w "${nodes[0]}" --cpus-per-task="$cpus" bash -c '
        set -euo pipefail
        export ENROOT_CACHE_PATH=/tmp/aim347-enroot/cache ENROOT_DATA_PATH=/tmp/aim347-enroot/data
        export ENROOT_RUNTIME_PATH=/tmp/aim347-enroot/runtime
        mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
        rm -f "$2.tmp"
        enroot import -o "$2.tmp" "dockerd://$1"
        mv "$2.tmp" "$2"' bash "$1" "$2"
}

if [[ ! -f $image ]]; then
    echo "prepare-event: building $image"
    build=(docker buildx build --load --build-arg NCCL_BUILD_JOBS=64 -t aim347-lab)
    if [[ -n $base ]]; then
        build+=(--build-context "$(awk '$1 == "FROM" {print $2}' "$lab/Dockerfile")=docker-image://$base")
    fi
    srun -N1 -n1 -w "${nodes[0]}" --cpus-per-task="$cpus" "${build[@]}" "$lab"
    import_image aim347-lab "$image"
fi
if ((serving)) && [[ ! -f $serving_image ]]; then
    echo "prepare-event: importing $serving_image"
    srun -N1 -n1 -w "${nodes[0]}" docker pull "$vllm_source"
    srun -N1 -n1 -w "${nodes[0]}" docker tag "$vllm_source" aim347-vllm:0.20.2
    import_image aim347-vllm:0.20.2 "$serving_image"
fi

# 1.prepare-llm.py writes manifest.json last, and the model's aim347-pin.json after the
# download, so a directory without that file is incomplete and is prepared again.
weights=()
if [[ ! -f $data/model/aim347-pin.json ]]; then
    weights=(--download-weights)
    rm -rf "$data/model" "$data/tokens-2048-1280"
fi
datasets=()
while read -r name config records; do
    [[ -f $data/$name/manifest.json ]] && continue
    rm -rf "${data:?}/$name"
    datasets+=("$name" "$config" "$records")
done <<'DATASETS'
tokens-2048-1280 configs/smol-2048-40.json 1280
tokens-2048-160 configs/smol-2048.json 160
tokens-1024-160 configs/smol-1024.json 160
DATASETS
if ((${#datasets[@]})); then
    echo 'prepare-event: preparing the data'
    # The model comes with the first dataset, tokens-2048-1280.
    # shellcheck disable=SC2016
    srun -N1 -n1 -w "${nodes[0]}" --cpus-per-task="$cpus" --container-image="$image" \
        --container-mounts="$lab:/opt/aim347,$data:/data" --container-workdir=/opt/aim347 --no-container-remap-root \
        bash -c '
        set -euo pipefail
        weights=$1
        shift
        rm -rf /data/preparation-packages
        python3 -m pip install --disable-pip-version-check --target /data/preparation-packages \
            -r requirements-data.txt -c constraints-data.txt
        while (($#)); do
            PYTHONPATH=/data/preparation-packages python3 1.prepare-llm.py --config "$2" --output /data \
                --tokens-name "$1" --records "$3" --minimum-document-tokens 4096 $weights
            weights=
            shift 3
        done' bash "${weights[*]}" "${datasets[@]}"
fi

python3 "$lab/lib/workshop.py" write-env --stage-dir "$stage_dir" --companion "$lab" --slurm-version "$slurm_version" \
    --cpus-per-node "$cpus" --instance-type "$instance_type" --nic "$nic" --grafana-url "$grafana_url" \
    --password-file /home/ubuntu/.aim347/grafana-admin-password
# ubuntu runs the tools.
chown -R ubuntu:ubuntu "$stage_dir/images" "$data" "$stage_dir/results" "$stage_dir/aim347.env"
echo 'prepare-event: ready'
