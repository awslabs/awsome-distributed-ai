#!/usr/bin/env bash
# Provisioning for one participant account, as root on the login node inside a
# two-node exclusive allocation of the GPU partition: the training image (built
# from this checkout's Dockerfile), the datasets and the model, and the
# environment file every tool reads. With --with-serving it also imports the vLLM
# image that tools/serve.sh runs. A rerun keeps what verifies and redoes the rest.
#
# Usage: bash facilitator/prepare-event.sh --stage-dir DIR --slurm-version VER --instance-type TYPE
#            --grafana-url URL [--build-jobs N] [--with-serving]
# Exit: 0 ready; 2 usage or precondition; 3 image; 4 data preparation; 5 verification on a node.
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workshop=(python3 "$lab/lib/workshop.py")
stage_dir='' slurm_version='' instance_type='' grafana_url='' build_jobs=64 serving=0
# The public vLLM 0.20.2 image (linux/amd64) that the serving path was validated with.
vllm_source=vllm/vllm-openai@sha256:68b773151407ca28c05479c02c0c08a573285ba197c8ff35d2103acd97aff78c
usage() {
    echo 'Usage: bash facilitator/prepare-event.sh --stage-dir DIR --slurm-version VER --instance-type TYPE --grafana-url URL [--build-jobs N] [--with-serving]' >&2
    exit 2
}
fail() { echo "prepare-event: $2" >&2; exit "$1"; }
while (($#)); do
    if [[ $1 == --with-serving ]]; then
        serving=1
        shift
        continue
    fi
    [[ $# -ge 2 ]] || usage
    case $1 in
        --stage-dir) stage_dir=$2 ;;
        --slurm-version) slurm_version=$2 ;;
        --instance-type) instance_type=$2 ;;
        --grafana-url) grafana_url=$2 ;;
        --build-jobs) build_jobs=$2 ;;
        *) usage ;;
    esac
    shift 2
done
[[ -n $stage_dir && -n $slurm_version && -n $instance_type && -n $grafana_url ]] || usage
# The tools assume 8 GPUs and two EFA devices per node and request --gres=gpu:8.
# g6e.48xlarge and g7e.48xlarge (8 GPUs, two of their four EFA interfaces attached) are for
# test events only.
[[ $instance_type == g7.48xlarge || $instance_type == g6e.48xlarge || $instance_type == g7e.48xlarge ]] ||
    fail 2 "--instance-type must be g7.48xlarge, g6e.48xlarge or g7e.48xlarge (got $instance_type)"
[[ $slurm_version =~ ^[0-9]+\.[0-9]+$ ]] || fail 2 "--slurm-version must look like 25.11 (got $slurm_version)"
[[ $build_jobs =~ ^[1-9][0-9]*$ ]] || fail 2 "--build-jobs must be a positive number (got $build_jobs)"
[[ $stage_dir == /* && $stage_dir != *[[:space:],:]* && -d $stage_dir ]] ||
    fail 2 "--stage-dir must be an existing absolute path without spaces, commas or colons (got $stage_dir)"
[[ $(id -u) == 0 ]] || fail 2 'run as root'
id ubuntu >/dev/null 2>&1 || fail 2 "no user ubuntu"
[[ -f $lab/Dockerfile && -f $lab/1.prepare-llm.py ]] || fail 2 "$lab is not the companion example directory"
[[ -n ${SLURM_JOB_ID:-} ]] || fail 2 'run inside a two-node exclusive allocation of the GPU partition (salloc)'
export PATH="/opt/aws/pcs/scheduler/slurm-$slurm_version/bin:$PATH"
image=$stage_dir/images/aim347-lab.sqsh
serving_image=$stage_dir/images/vllm-v0.20.2.sqsh
data=$stage_dir/data
failed_dir=$stage_dir/failed-$(date -u +%Y%m%dT%H%M%SZ)

step() {
    local name=$1 started=$SECONDS
    shift
    "$@"
    echo "prepare-event: $name ok ($((SECONDS - started)) s)"
}

move_failed() {
    mkdir -p "$failed_dir"
    mv "$1" "$failed_dir/"
    echo "prepare-event: moved $1 to $failed_dir/"
}

checks() {
    mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    ((${#nodes[@]} == 2)) || fail 2 "the allocation has ${#nodes[@]} nodes, expected 2"
    local node counts=()
    for node in "${nodes[@]}"; do
        counts+=("$(scontrol show node -o "$node" | tr ' ' '\n' | sed -n 's/^CPUEfctv=//p')")
    done
    [[ ${counts[0]} =~ ^[1-9][0-9]*$ && ${counts[0]} == "${counts[1]}" ]] ||
        fail 2 "the nodes need equal nonzero CPUEfctv (got ${counts[*]})"
    cpus=${counts[0]}
    [[ $(findmnt -T "$stage_dir" -n -o FSTYPE) == lustre ]] || fail 2 "$stage_dir is not on Lustre"
    local address
    address=$(scontrol show node -o "${nodes[0]}" | tr ' ' '\n' | sed -n 's/^NodeAddr=//p')
    # The interface the second node uses to reach the first one carries Socket traffic.
    nic=$(srun -N1 -n1 -w "${nodes[1]}" ip -j route get "$address" |
          python3 -c 'import json, sys; print(json.load(sys.stdin)[0]["dev"])') ||
        fail 2 "cannot find the interface from ${nodes[1]} to $address"
}

# PCS installs its monitoring stack on a GPU node as a NodeReady action, after the node
# already accepts jobs. The install reinstalls Docker, configures the NVIDIA runtime,
# restarts Docker and only then starts node-exporter. A docker build or a Pyxis container
# started before that restart can fail, so wait until node-exporter answers on both nodes.
monitoring_install() {
    # shellcheck disable=SC2016
    srun -N2 --ntasks-per-node=1 bash -c '
        for _ in $(seq 180); do
            curl -sf -o /dev/null http://localhost:9100/metrics && docker info >/dev/null 2>&1 && exit 0
            sleep 10
        done
        echo "prepare-event: the PCS monitoring install has not finished on ${SLURMD_NODENAME:-$(hostname)} after 30 minutes;" \
            "read /var/log/amazon/pcs/lifecycle/actions/nodeReady/install-monitoring.log on that node" >&2
        exit 1' ||
        fail 2 'the PCS monitoring install did not finish on both GPU nodes'
}

image_ok() { [[ -f $image && -f $image.sha256 ]] && sha256sum --check --strict --status "$image.sha256"; }

build_image() {
    install -d -m 0755 "$stage_dir/images"
    if image_ok; then
        echo "prepare-event: keeping $image (its SHA-256 checks)"
        return
    fi
    rm -f "$image" "$image.sha256" "$image.verified" "$image.tmp"
    local tag
    tag=aim347-lab:$(git -c safe.directory='*' -C "$lab" rev-parse --short HEAD) || fail 3 "$lab is not a git checkout"
    # Every CPU of the node: a step task otherwise gets one, and mksquashfs uses them all.
    # shellcheck disable=SC2016
    srun -N1 -n1 -w "${nodes[0]}" --cpus-per-task="$cpus" bash -c '
        set -euo pipefail
        export ENROOT_CACHE_PATH=/tmp/aim347-enroot/cache ENROOT_DATA_PATH=/tmp/aim347-enroot/data
        export ENROOT_RUNTIME_PATH=/tmp/aim347-enroot/runtime
        mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
        docker build --build-arg NCCL_BUILD_JOBS="$1" -t "$2" "$3"
        enroot import -o "$4.tmp" "dockerd://$2"' bash "$build_jobs" "$tag" "$lab" "$image" ||
        fail 3 "building or importing $tag failed"
    local hash
    hash=$(sha256sum "$image.tmp" | cut -d' ' -f1) || fail 3 "cannot hash $image.tmp"
    printf '%s  %s\n' "$hash" "$image" > "$image.sha256"
    mv "$image.tmp" "$image"
}

verify_image() {
    # shellcheck disable=SC2016
    srun -N2 --ntasks-per-node=1 bash -c 'unsquashfs -s "$1" >/dev/null && sha256sum --check --strict --status "$1.sha256" &&
        echo "${SLURMD_NODENAME:-$(hostname)} OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"' bash "$image" >> "$image.verified" ||
        fail 5 "$image does not verify on both nodes"
}

serving_image_ok() {
    [[ -f $serving_image && -f $serving_image.sha256 ]] && sha256sum --check --strict --status "$serving_image.sha256"
}

# Optional B of the workshop (tools/serve.sh). The image comes from Docker Hub at a fixed digest.
import_serving_image() {
    if serving_image_ok; then
        echo "prepare-event: keeping $serving_image (its SHA-256 checks)"
        return
    fi
    rm -f "$serving_image" "$serving_image.sha256" "$serving_image.verified" "$serving_image.tmp"
    # shellcheck disable=SC2016
    srun -N1 -n1 -w "${nodes[0]}" --cpus-per-task="$cpus" bash -c '
        set -euo pipefail
        export ENROOT_CACHE_PATH=/tmp/aim347-enroot/cache ENROOT_DATA_PATH=/tmp/aim347-enroot/data
        export ENROOT_RUNTIME_PATH=/tmp/aim347-enroot/runtime
        mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
        docker pull "$1"
        docker tag "$1" aim347-vllm:0.20.2
        enroot import -o "$2.tmp" dockerd://aim347-vllm:0.20.2' bash "$vllm_source" "$serving_image" ||
        fail 3 "pulling or importing $vllm_source failed"
    local hash
    hash=$(sha256sum "$serving_image.tmp" | cut -d' ' -f1) || fail 3 "cannot hash $serving_image.tmp"
    printf '%s  %s\n' "$hash" "$serving_image" > "$serving_image.sha256"
    mv "$serving_image.tmp" "$serving_image"
}

verify_serving_image() {
    # shellcheck disable=SC2016
    srun -N2 --ntasks-per-node=1 bash -c 'unsquashfs -s "$1" >/dev/null && sha256sum --check --strict --status "$1.sha256" &&
        echo "${SLURMD_NODENAME:-$(hostname)} OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"' bash "$serving_image" >> "$serving_image.verified" ||
        fail 5 "$serving_image does not verify on both nodes"
}

prepare_data() {
    install -d -m 0755 "$data"
    local name pending=() model_pending=0
    for name in tokens-2048-1280 tokens-2048-160 tokens-1024-160; do
        "${workshop[@]}" verify-data --data "$data" --dataset "$name" >/dev/null && continue
        if [[ -e $data/$name ]]; then move_failed "$data/$name"; fi
        pending+=("$name")
    done
    if ! "${workshop[@]}" verify-data --data "$data" --model >/dev/null; then
        model_pending=1
        if [[ -e $data/model ]]; then move_failed "$data/model"; fi
        # The first preparation downloads the model together with its tokens.
        if [[ " ${pending[*]} " != *' tokens-2048-1280 '* ]]; then
            move_failed "$data/tokens-2048-1280"
            pending=(tokens-2048-1280 "${pending[@]}")
        fi
    fi
    ((${#pending[@]})) || { echo 'prepare-event: keeping data (every dataset and the model verify)'; return; }
    local jobs=()
    for name in "${pending[@]}"; do
        case $name in
            tokens-2048-1280) jobs+=("$name" configs/smol-2048-40.json 1280 "$model_pending") ;;
            tokens-2048-160) jobs+=("$name" configs/smol-2048.json 160 0) ;;
            tokens-1024-160) jobs+=("$name" configs/smol-1024.json 160 0) ;;
        esac
    done
    # shellcheck disable=SC2016
    srun -N1 -n1 -w "${nodes[0]}" --cpus-per-task="$cpus" --container-image="$image" \
        --container-mounts="$lab:/opt/aim347,$data:/data" --container-workdir=/opt/aim347 --no-container-remap-root \
        bash -c '
        set -euo pipefail
        if [[ ! -f /data/preparation-packages.json ]]; then
            rm -rf /data/preparation-packages
            python3 -m pip install --disable-pip-version-check --target /data/preparation-packages \
                -r requirements-data.txt -c constraints-data.txt
            python3 -m pip list --path /data/preparation-packages --format=json > /data/preparation-packages.json
        fi
        while (($#)); do
            weights=()
            [[ $4 == 1 ]] && weights=(--download-weights)
            PYTHONPATH=/data/preparation-packages python3 1.prepare-llm.py --config "$2" --output /data \
                --tokens-name "$1" --records "$3" --minimum-document-tokens 4096 ${weights[@]+"${weights[@]}"}
            shift 4
        done' bash "${jobs[@]}" || fail 4 "preparing ${pending[*]} failed"
    for name in "${pending[@]}"; do
        "${workshop[@]}" verify-data --data "$data" --dataset "$name" || fail 4 "$name does not verify after preparation"
    done
    "${workshop[@]}" verify-data --data "$data" --model || fail 4 'the model does not verify after preparation'
}

verify_data() {
    srun -N2 --ntasks-per-node=1 "${workshop[@]}" verify-data --data "$data" --dataset tokens-2048-1280 \
        --dataset tokens-2048-160 --dataset tokens-1024-160 --model || fail 5 'the data does not verify on both nodes'
}

write_environment() {
    "${workshop[@]}" write-env --stage-dir "$stage_dir" --companion "$lab" --slurm-version "$slurm_version" \
        --cpus-per-node "$cpus" --instance-type "$instance_type" --nic "$nic" \
        --image-sha256 "$(cut -d' ' -f1 "$image.sha256")" --grafana-url "$grafana_url" \
        --password-file /home/ubuntu/.aim347/grafana-admin-password || fail 2 'cannot write the environment file'
}

ownership() {
    install -d -m 0755 "$stage_dir/results"
    chown -R ubuntu:ubuntu "$stage_dir/images" "$data" "$stage_dir/results" "$stage_dir/aim347.env"
}

step checks checks
step 'monitoring install' monitoring_install
step image build_image
step 'image on both nodes' verify_image
if ((serving)); then
    step 'serving image' import_serving_image
    step 'serving image on both nodes' verify_serving_image
fi
step data prepare_data
step 'data on both nodes' verify_data
step 'environment file' write_environment
step ownership ownership
echo 'prepare-event: ready'
