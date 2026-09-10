#!/usr/bin/env bash
set -euo pipefail
lab=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$lab"
export PATH="/opt/aws/pcs/scheduler/slurm-${PCS_SLURM_VERSION:-25.05}/bin:$PATH"
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
