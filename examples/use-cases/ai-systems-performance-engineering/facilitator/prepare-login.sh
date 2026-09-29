#!/usr/bin/env bash
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$lab"
export PCS_SLURM_VERSION=${PCS_SLURM_VERSION:-25.05}
export PATH="/opt/aws/pcs/scheduler/slurm-${PCS_SLURM_VERSION}/bin:$PATH"
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
    # 1.prepare-llm.py. Configuration is supplied explicitly.
    : "${PARTITION:?}" "${ASSIGNED_NODES:?}" "${LAB_IMAGE:?}" "${VLLM_IMAGE:?}"
    : "${LLM_DATA_DIR:?}" "${LLM_RESULTS_DIR:?}" "${NCCL_SOCKET_IFNAME:?}"
    : "${GPUS_PER_NODE:?}" "${LOGIN_BIND_IP:?}" "${PROMETHEUS_PORT:?}"
    export PARTITION ASSIGNED_NODES LAB_IMAGE VLLM_IMAGE LLM_DATA_DIR LLM_RESULTS_DIR
    export NCCL_SOCKET_IFNAME GPUS_PER_NODE LOGIN_BIND_IP PROMETHEUS_PORT
    export LLM_CONFIG=${LLM_CONFIG:-configs/llm.json}
    python3 - <<'PY'
import hashlib, ipaddress, json, os, shlex, subprocess
from pathlib import Path
config_path = Path(os.environ['LLM_CONFIG'])
if config_path != Path('configs/llm.json'):
    raise ValueError('participant environment requires configs/llm.json; select recovery explicitly in Lab 5')
cfg = json.loads(config_path.read_text())
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
for relative, config_name, records, fingerprint in (
    ('tokens', 'llm.json', 160, '276a090b9451c92d61cdeb1ca627a92bbff1e7512ad49ad7f322fb537546a166'),
    ('recovery/tokens', 'llm-recovery.json', 1280, 'b9a7d29b22c2b6cac9455323bea87f2dd69b6a39e73d33bbaa1bbcf4faf06488'),
):
    cfg = json.loads((Path('configs') / config_name).read_text())
    manifest = json.loads((data / relative / 'manifest.json').read_text())
    for key in ('model_id', 'tokenizer_revision', 'dataset_id', 'dataset_revision', 'dataset_subset', 'dataset_split', 'sequence_length'):
        if manifest[key] != cfg[key]:
            raise ValueError(f'data provenance mismatch: {key}')
    if manifest['records'] != records or manifest['sha256'] != fingerprint:
        raise ValueError('insufficient prepared records')
    if (data / relative / 'tokens.bin').stat().st_size != manifest['records'] * cfg['sequence_length'] * 4:
        raise ValueError('token file size does not match manifest')
    if (data / relative / 'lengths.bin').stat().st_size != manifest['records'] * 4:
        raise ValueError('length file size does not match manifest')
    digest = hashlib.sha256()
    for name in ('tokens.bin', 'lengths.bin'):
        with (data / relative / name).open('rb') as stream:
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
keys = ('PCS_SLURM_VERSION', 'PARTITION', 'ASSIGNED_NODES', 'LAB_IMAGE', 'VLLM_IMAGE', 'LLM_DATA_DIR', 'LLM_RESULTS_DIR', 'LLM_CONFIG',
        'NCCL_SOCKET_IFNAME', 'GPUS_PER_NODE', 'LOGIN_BIND_IP', 'PROMETHEUS_PORT')
values = {key: os.environ[key] for key in keys}
values.update(COMPUTE_NODES=','.join(nodes), GRAFANA_DASHBOARD_FILE=str(Path('observability/llm-dashboard.json').resolve()),
              VLLM_METRICS_PORTS='8100,8101,8102,8103,8104,8105,8106,8107', PUSHGATEWAY_URL=f"http://{values['LOGIN_BIND_IP']}:9091")
destination = results / 'llm-environment.sh'
with destination.open('x') as stream:
    stream.write(''.join(f'export {key}={shlex.quote(value)}\n' for key, value in values.items()))
    stream.write('export PATH="/opt/amazon/efa/bin:/opt/aws/pcs/scheduler/slurm-${PCS_SLURM_VERSION}/bin:$PATH"\n')
print(destination)
PY
    # No implicit GPU allocation, image download, monitoring restart or release pin.
    # The participant sources the generated environment in the supplied companion.
    exit 0
fi
printf 'Use --llm or --llm-prepare with explicit assigned paths.\n' >&2
exit 2
