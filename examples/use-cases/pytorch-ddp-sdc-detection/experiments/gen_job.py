#!/usr/bin/env python3
"""Write the Kubeflow PyTorchJob manifest that runs run_order.py on every node.

    python3 gen_job.py > job.yaml                    # full run, about 2 hours on 4 x p5.48xlarge
    python3 gen_job.py --smoke > job.yaml            # small model, a few minutes: checks the wiring
    python3 gen_job.py --nodes 3 --healthy-minutes 15 > job.yaml
    python3 gen_job.py --node-selector node.kubernetes.io/instance-type=ml.p5.48xlarge   # HyperPod
"""
import argparse
import time

DEFAULT_IMAGE = ("763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                 "pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-ec2"
                 "@sha256:c4685b08072841d85f46c1e30653320df7b742483384c43ecdcd8507f1dbbd41")
GPUS_PER_NODE = 8   # the fault scenarios in run_order.py inject faults on ranks that assume 8

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--smoke", action="store_true", help="small model and short healthy runs")
p.add_argument("--run-id", default=time.strftime("%Y%m%d%H%M%S", time.gmtime()),
               help="unique ID, used in the job name and labels (default: UTC timestamp)")
p.add_argument("--name", help="job name (default: sdc-order[-smoke]-<run-id>)")
p.add_argument("--namespace", default="sdc-detection")
p.add_argument("--configmap", default="sdc-order-scripts", help="ConfigMap that holds the Python files")
p.add_argument("--image", default=DEFAULT_IMAGE,
               help="PyTorch training image; pin it by digest (repo:tag@sha256:...) for reproducibility")
p.add_argument("--nodes", type=int, default=4, help="at least 3, so the ring audit can confirm")
p.add_argument("--efa-per-node", type=int, default=32,
               help="EFA interfaces per node (32 on p5.48xlarge); 0 leaves out EFA settings (untested)")
p.add_argument("--cpu", default="96", help="CPU request per pod")
p.add_argument("--memory", default="800Gi", help="memory request per pod")
p.add_argument("--node-selector", default="node.kubernetes.io/instance-type=p5.48xlarge",
               help="key=value label that selects the GPU nodes (on SageMaker HyperPod, the "
                    "instance type has an ml. prefix: ml.p5.48xlarge)")
p.add_argument("--toleration", action="append", default=[], metavar="KEY[=VALUE]:EFFECT",
               help="taint the GPU pods tolerate, for example nvidia.com/gpu:NoSchedule; "
                    "repeat for several taints")
p.add_argument("--healthy-minutes", type=float,
               help="length of the deterministic and production-schedule healthy runs; the "
                    "default-kernel run is half as long (default: 45, smoke 0.5)")
p.add_argument("--keep-alive", type=int,
               help="seconds the rank-0 pod waits for `job.sh fetch` before exiting "
                    "(default: 1800, smoke 600)")
a = p.parse_args()

if a.nodes < 3:
    p.error("--nodes must be at least 3")
if a.efa_per_node < 0:
    p.error("--efa-per-node must be 0 or more")
name = a.name or f"sdc-order{'-smoke' if a.smoke else ''}-{a.run_id}"
healthy = a.healthy_minutes if a.healthy_minutes is not None else (0.5 if a.smoke else 45)
keep_alive = a.keep_alive if a.keep_alive is not None else (600 if a.smoke else 1800)
sel_key, sel_val = a.node_selector.split("=", 1)
labels = f"{{app.kubernetes.io/part-of: sdc-detection, sdc-detection/run-id: \"{a.run_id}\"}}"


def toleration_yaml():
    if not a.toleration:
        return ""
    lines = ["          tolerations:"]
    for t in a.toleration:
        kv, sep, effect = t.rpartition(":")
        if not sep or not kv or effect not in ("NoSchedule", "PreferNoSchedule", "NoExecute"):
            p.error(f"--toleration {t!r}: expected KEY[=VALUE]:EFFECT, with EFFECT one of "
                    "NoSchedule, PreferNoSchedule, NoExecute")
        key, _, value = kv.partition("=")
        match = f'operator: Equal, value: "{value}"' if value else "operator: Exists"
        lines.append(f"            - {{key: {key}, {match}, effect: {effect}}}")
    return "\n".join(lines) + "\n"


tolerations = toleration_yaml()
efa_env = ("""                - {name: FI_PROVIDER, value: efa}
                - {name: FI_EFA_USE_DEVICE_RDMA, value: "1"}
""" if a.efa_per_node else "")
efa_res = f"                  vpc.amazonaws.com/efa: {a.efa_per_node}\n" if a.efa_per_node else ""


def template():
    return f"""        metadata:
          labels: {labels}
        spec:
          automountServiceAccountToken: false   # the workload never calls the Kubernetes API
          securityContext:
            seccompProfile: {{type: RuntimeDefault}}
          nodeSelector:
            {sel_key}: {sel_val}
{tolerations}          containers:
            - name: pytorch
              image: {a.image}
              imagePullPolicy: IfNotPresent
              securityContext:
                allowPrivilegeEscalation: false
              command: ["/bin/bash", "-c"]
              args:
                - |
                  echo "NODE_NAME=$NODE_NAME PET_NODE_RANK=$PET_NODE_RANK UTC=$(date -u +%FT%TZ)"
                  nvidia-smi --query-gpu=index,uuid,driver_version --format=csv,noheader | sed "s/^/GPU $NODE_NAME /"
                  mkdir -p /results
                  torchrun --nnodes={a.nodes} --nproc-per-node={GPUS_PER_NODE} --node-rank=$PET_NODE_RANK \\
                    --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT /scripts/run_order.py
                  rc=$?
                  echo "TORCHRUN_EXIT $rc UTC=$(date -u +%FT%TZ)"
                  if [ "$PET_NODE_RANK" = "0" ]; then
                    touch /results/DONE
                    for i in $(seq 1 {keep_alive // 5}); do [ -f /results/FETCHED ] && break; sleep 5; done
                  fi
                  exit $rc
              env:
                - name: NODE_NAME            # maps ranks to nodes when an alarm names a rank
                  valueFrom:
                    fieldRef:
                      fieldPath: spec.nodeName
                - {{name: SDC_HEALTHY_MIN, value: "{healthy}"}}
                - {{name: SDC_SMOKE, value: "{1 if a.smoke else 0}"}}
                - {{name: SDC_TIMEOUT_MIN, value: "{8 if a.smoke else 30}"}}
                - {{name: CUBLAS_WORKSPACE_CONFIG, value: ":4096:8"}}
{efa_env}                - {{name: NCCL_DEBUG, value: WARN}}
                - {{name: PYTHONUNBUFFERED, value: "1"}}
                - {{name: PYTHONDONTWRITEBYTECODE, value: "1"}}
              resources:
                requests:
                  nvidia.com/gpu: {GPUS_PER_NODE}
{efa_res}                  cpu: "{a.cpu}"
                  memory: {a.memory}
                limits:
                  nvidia.com/gpu: {GPUS_PER_NODE}
{efa_res}                  cpu: "{a.cpu}"
                  memory: {a.memory}
              volumeMounts:
                - {{name: scripts, mountPath: /scripts}}
                - {{name: shm, mountPath: /dev/shm}}
                - {{name: results, mountPath: /results}}
          volumes:
            - name: scripts
              configMap: {{name: {a.configmap}}}
            - name: shm
              emptyDir: {{medium: Memory, sizeLimit: 64Gi}}
            - name: results
              emptyDir: {{}}
"""


print(f"""apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {name}
  namespace: {a.namespace}
  labels: {labels}
spec:
  nprocPerNode: "{GPUS_PER_NODE}"
  runPolicy:
    cleanPodPolicy: None
    backoffLimit: 0
    ttlSecondsAfterFinished: 259200     # 3 days, then the operator deletes the job and its pods
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
{template()}    Worker:
      replicas: {a.nodes - 1}
      restartPolicy: Never
      template:
{template()}""")
