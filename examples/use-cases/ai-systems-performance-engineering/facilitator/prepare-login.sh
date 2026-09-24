#!/usr/bin/env bash
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$lab"
export PATH="/opt/aws/pcs/scheduler/slurm-${PCS_SLURM_VERSION:-25.05}/bin:$PATH"
if [[ ${1:-} == --llm-prepare ]]; then
    [[ $# == 1 ]] || { printf 'Usage: %s --llm-prepare\n' "$0" >&2; exit 2; }
    : "${SLURM_JOB_ID:?Use a login-node salloc shell for LLM preparation}"
    : "${LLM_RESULTS_DIR:?}" "${LLM_DATA_DIR:?}" "${PARTITION:?}" "${ASSIGNED_NODES:?}"
    : "${GPUS_PER_NODE:?}" "${LOGIN_BIND_IP:?}" "${PROMETHEUS_PORT:?}" "${NCCL_SOCKET_IFNAME:?}"
    [[ $LLM_RESULTS_DIR == /* && $LLM_RESULTS_DIR != *[[:space:],:]* ]] || { printf 'Use an absolute mount-safe results path.\n' >&2; exit 2; }
    [[ ! -e $LLM_RESULTS_DIR && ! -e $LLM_DATA_DIR ]] || { printf 'Use fresh data and result directories.\n' >&2; exit 2; }
    bash "$lab/1.prepare.sh" --llm
    mkdir "$LLM_RESULTS_DIR"
    bash "$lab/facilitator/prepare-login.sh" --llm
    # Monitoring is a separate, explicit invocation on the login node; do not
    # start a second login stack inside a compute-node batch script.
    exit 0
fi
if [[ ${1:-} == --llm ]]; then
    [[ $# == 1 ]] || { printf 'Usage: %s --llm\n' "$0" >&2; exit 2; }
    # The LLM route consumes immutable images and attributed data prepared with
    # 1.prepare-llm.py. Never source or overwrite the legacy workload's .env.
    : "${PARTITION:?}" "${ASSIGNED_NODES:?}" "${LAB_IMAGE:?}" "${VLLM_IMAGE:?}"
    : "${LLM_DATA_DIR:?}" "${LLM_RESULTS_DIR:?}" "${NCCL_SOCKET_IFNAME:?}"
    : "${GPUS_PER_NODE:?}" "${LOGIN_BIND_IP:?}" "${PROMETHEUS_PORT:?}"
    export PARTITION ASSIGNED_NODES LAB_IMAGE VLLM_IMAGE LLM_DATA_DIR LLM_RESULTS_DIR
    export NCCL_SOCKET_IFNAME GPUS_PER_NODE LOGIN_BIND_IP PROMETHEUS_PORT
    python3 - <<'PY'
import hashlib, ipaddress, json, os, shlex, subprocess
from pathlib import Path
cfg = json.loads(Path('configs/llm.json').read_text())
data = Path(os.environ['LLM_DATA_DIR'])
results = Path(os.environ['LLM_RESULTS_DIR'])
for path in (data, results):
    if not path.is_absolute() or not path.is_dir() or any(c in str(path) for c in ' ,:\t\n'):
        raise ValueError('LLM data and results must be existing absolute mount-safe paths')
    if subprocess.check_output(['findmnt', '-T', str(path), '-n', '-o', 'FSTYPE'], text=True).strip() != 'lustre':
        raise ValueError('multinode LLM data and checkpoints require the prepared Lustre mount')
pin = json.loads((data / 'model/aim347-pin.json').read_text())
if pin != {key: cfg[key] for key in ('model_id', 'model_revision')}:
    raise ValueError('model pin mismatch')
manifest = json.loads((data / 'tokens/manifest.json').read_text())
for key in ('model_id', 'tokenizer_revision', 'dataset_id', 'dataset_revision', 'dataset_subset', 'dataset_split', 'sequence_length'):
    if manifest[key] != cfg[key]:
        raise ValueError(f'data provenance mismatch: {key}')
if manifest['records'] < cfg['global_batch_samples'] * cfg['updates']:
    raise ValueError('insufficient prepared records')
digest = hashlib.sha256()
for name in ('tokens.bin', 'lengths.bin'):
    with (data / 'tokens' / name).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
if digest.hexdigest() != manifest['sha256']:
    raise ValueError('prepared token checksum mismatch')
nodes = os.environ['ASSIGNED_NODES'].split(',')
if len(nodes) != 2 or len(set(nodes)) != 2 or any(not node for node in nodes):
    raise ValueError('exactly two distinct assigned compute nodes required')
registered = set(subprocess.check_output(['sinfo', '-p', os.environ['PARTITION'], '-N', '-h', '-o', '%N'], text=True).split())
if not set(nodes) <= registered:
    raise ValueError('assigned nodes are not in the supplied partition')
if os.environ['GPUS_PER_NODE'] != '8':
    raise ValueError('the qualified LLM participant route requires eight GPUs per node')
interface = os.environ['NCCL_SOCKET_IFNAME']
if not interface.startswith('=') or len(interface) < 2 or any(c in interface for c in ', \t\n'):
    raise ValueError('supply the verified exact private NCCL interface')
address = ipaddress.ip_address(os.environ['LOGIN_BIND_IP'])
if address.version != 4 or not address.is_private or address.is_loopback or address.is_unspecified:
    raise ValueError('monitoring requires the private IPv4 login address reachable by compute nodes')
if not 1 <= int(os.environ['PROMETHEUS_PORT']) <= 65535:
    raise ValueError('invalid Prometheus TCP port')
for name in ('LAB_IMAGE', 'VLLM_IMAGE'):
    value = os.environ[name]
    if not Path(value).is_absolute() or any(c in value for c in ' ,:\t\n'):
        raise ValueError('image paths must be absolute and mount-safe')
keys = ('PARTITION', 'ASSIGNED_NODES', 'LAB_IMAGE', 'VLLM_IMAGE', 'LLM_DATA_DIR', 'LLM_RESULTS_DIR',
        'NCCL_SOCKET_IFNAME', 'GPUS_PER_NODE', 'LOGIN_BIND_IP', 'PROMETHEUS_PORT')
values = {key: os.environ[key] for key in keys}
values.update(AIM347_SKIP_LEGACY_ENV='1', COMPUTE_NODES=','.join(nodes), GRAFANA_DASHBOARD_FILE=str(Path('observability/llm-dashboard.json').resolve()),
              VLLM_METRICS_PORTS='8100,8101,8102,8103', PUSHGATEWAY_URL=f"http://{values['LOGIN_BIND_IP']}:9091")
destination = results / 'llm-environment.sh'
with destination.open('x') as stream:
    stream.write(''.join(f'export {key}={shlex.quote(value)}\n' for key, value in values.items()))
print(destination)
PY
    # No implicit GPU allocation, image download, monitoring restart or release pin.
    # The participant sources the generated environment in the supplied companion.
    exit 0
fi
[[ $# == 0 ]] || { printf 'Usage: %s [--llm|--llm-prepare]\n' "$0" >&2; exit 2; }
: "${AIM347_PARTITION:=gpu}" "${AIM347_STAGE_DIR:=/fsx/aim347/participant}"
if [[ ${AIM347_NODE_LOCAL:-0} == 1 ]]; then
    [[ $AIM347_STAGE_DIR == /opt/* && -d $AIM347_STAGE_DIR ]] || exit 2
else
    findmnt -T "$AIM347_STAGE_DIR" -n -o FSTYPE | grep -qx lustre
fi
command -v docker enroot scontrol srun
docker compose version
if [[ ! -f .env ]]; then
    export AIM347_PARTITION AIM347_STAGE_DIR
    python3 - <<'PY'
import json, os, shlex, subprocess
from pathlib import Path
partition = os.environ['AIM347_PARTITION']
nodes = sorted(set(subprocess.check_output(['sinfo', '-p', partition, '-N', '-h', '-o', '%N'], text=True).split()))
if len(nodes) != 2:
    raise SystemExit('The lab needs exactly two registered compute nodes; set .env explicitly for a larger partition')
fields = dict(word.split('=', 1) for word in subprocess.check_output(['scontrol', 'show', 'node', nodes[0], '-o'], text=True).split() if '=' in word)
route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', fields['NodeAddr']], text=True))[0]
nic = os.environ.get('AIM347_SOCKET_INTERFACE')
if not nic:
    peer = fields['NodeAddr']
    discovered = subprocess.check_output(['srun', '-p', partition, '-N1', '-n1', '-w', nodes[1], '--time=00:02:00', 'ip', '-j', 'route', 'get', peer], text=True)
    nic = json.loads(discovered)[0]['dev']
instance_types = subprocess.check_output(['srun', '-p', partition, '-N2', '-n2', '--ntasks-per-node=1', '--time=00:02:00', 'bash', '-c', 'source "$1"; detect_instance_type', 'bash', str(Path('lib/instance-type.sh').resolve())], text=True).split()
if len(instance_types) != len(nodes) or len(set(instance_types)) != 1 or instance_types[0] == 'unknown':
    raise SystemExit('Compute nodes must report the same detected instance type')
stage = os.environ['AIM347_STAGE_DIR']
values = dict(PARTITION=partition, COMPUTE_NODES=','.join(nodes), DATA_DIR=stage,
              LAB_IMAGE=f'{stage}/aim347-lab.sqsh', VLLM_IMAGE=f'{stage}/vllm-v0.20.2.sqsh',
              RUN_ID='participant-trial-a', INSTANCE_TYPE=instance_types[0],
              NCCL_SOCKET_IFNAME=f'={nic}', DENSE_TFLOPS='', STEPS='100', WARMUP='10', MICROBATCH='1', CPU_ROUNDS='2000',
              LOGIN_BIND_IP=route['prefsrc'], PUSHGATEWAY_URL=f"http://{route['prefsrc']}:9091",
              PROMETHEUS_PORT=os.environ.get('AIM347_PROMETHEUS_PORT', '9092'))
if os.environ.get('PREBUILT_LAB_IMAGE'):
    values['PREBUILT_LAB_IMAGE'] = os.environ['PREBUILT_LAB_IMAGE']
Path('.env').write_text(''.join(f'{key}={shlex.quote(value)}\n' for key, value in values.items()))
PY
fi
private_dir=/tmp/aim347-enroot-$(id -u)
install -d -m 0700 "$private_dir" "$private_dir/runtime" "$private_dir/cache" "$private_dir/data"
export ENROOT_RUNTIME_PATH="$private_dir/runtime" ENROOT_CACHE_PATH="$private_dir/cache" ENROOT_DATA_PATH="$private_dir/data"
export ENROOT_MAX_PROCESSORS=${ENROOT_MAX_PROCESSORS:-8}
./1.prepare.sh
./0.deploy-observability.sh login
set -a; source .env; set +a
sha256sum "$LAB_IMAGE" "$VLLM_IMAGE" "$DATA_DIR/tokens/manifest.json"
curl --fail --retry 12 --retry-connrefused --retry-delay 2 "http://127.0.0.1:${PROMETHEUS_PORT:-9090}/-/ready"
curl --fail --retry 12 --retry-connrefused --retry-delay 2 "$PUSHGATEWAY_URL/-/ready"
