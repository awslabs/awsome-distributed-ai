# GPU and EFA tests

The test that decides whether this architecture works. Two GPU nodes must come up, advertise
exactly the GPU and EFA counts the template's mappings claim, expose their local NVMe as a
RAID0 array, hand `/dev/infiniband` to a pod that asks for EFA, and move traffic between pods
on different nodes **over the EFA provider** rather than over TCP.

Run it on `2 x g7e.12xlarge`. For a different type, replace the expected numbers with that
type's `GpuCount` and `NicLayout` values — step 2 reads them out of the template so you do not
have to trust this page.

Placeholders and the pinned test image:

```bash
export REGION=us-east-2
export STACK=eks-gpu-efa
export AZ_A=us-east-2b
export AZ_B=us-east-2a
export BUCKET=my-template-bucket
export EFA_IMAGE=public.ecr.aws/hpc-cloud/nccl-tests:cuda13.1.2-efa1.50.0-ofiv1.21.1-ncclv2.31.2-1-testsv2.20.0
```

`$AZ_A` has to offer the instance type under test. `$BUCKET` is a bucket in `$REGION` you can write to; step 0 publishes this checkout's templates there, because the root creates its children by URL (README section 8).

`$EFA_IMAGE` is the repository's usual EFA and NCCL test container, published by AWS on a
public registry (no credentials, no build). The tag is pinned rather than `latest`; it resolves
to `sha256:655ac5a5de8871570112077612be7209fc1f3982eb6b6df6f34a422a5cf398e5`, and it carries
libfabric with the EFA provider under `/opt/amazon/efa`, `aws-ofi-nccl` and the compiled
`nccl-tests` binaries.

---

## Step 0 — deploy

```bash
cd architectures/amazon-eks
aws s3 sync assets/ "s3://$BUCKET/templates/amazon-eks/" --exclude '*' --include '*.yaml'
aws cloudformation create-stack \
  --stack-name "$STACK" --region "$REGION" \
  --template-url "https://$BUCKET.s3.amazonaws.com/templates/amazon-eks/eks-gpu-cluster-deploy-all.yaml" \
  --capabilities CAPABILITY_IAM \
  --parameters ParameterKey=S3BucketName,ParameterValue=$BUCKET \
               ParameterKey=S3KeyPrefix,ParameterValue=templates/amazon-eks/ \
               ParameterKey=PrimarySubnetAZ,ParameterValue=$AZ_A \
               ParameterKey=SecondarySubnetAZ,ParameterValue=$AZ_B \
               ParameterKey=GpuInstanceType,ParameterValue=g7e.12xlarge \
               ParameterKey=GpuNodeCount,ParameterValue=2
aws cloudformation wait stack-create-complete --stack-name "$STACK" --region "$REGION"

eval "$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`KubeconfigCommand`].OutputValue' --output text)"
```

**Expected:** `CREATE_COMPLETE`. On reserved capacity add
`ParameterKey=CapacityReservationId,ParameterValue=cr-...` and, for a Capacity Block,
`ParameterKey=CapacityReservationType,ParameterValue=capacity-block`.

If the stack fails at the bootstrap custom resource, read the reason before deleting anything:

```bash
aws cloudformation describe-stack-events --stack-name "$STACK" --region "$REGION" \
  --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
  --output text
aws logs tail "$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`BootstrapLogGroup`].OutputValue' --output text)" --since 1h
```

**Expected:** the failure reason is a sentence naming which phase failed — `install` and
`validate` must be distinguishable. "Custom resource did not respond" means the bootstrap
failed to answer CloudFormation at all, which is a defect in the template, not in your account.

---

## Step 1 — (a) both GPU nodes are Ready, labelled and tainted

```bash
kubectl get nodes -L role -L node.kubernetes.io/instance-type
kubectl get nodes -l role=gpu -o custom-columns=\
'NAME:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type'
kubectl get nodes -l role=gpu -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.taints[*].key}{"="}{.spec.taints[*].value}{":"}{.spec.taints[*].effect}{"\n"}{end}'
```

**Expected:** exactly two nodes with `role=gpu`, both `Ready`, both `g7e.12xlarge`, and each
carrying the taint `nvidia.com/gpu=true:NoSchedule`. The system nodes carry `role=system` and
no taint. A GPU node with no taint is a failure: the repository's serving manifests tolerate
that taint and rely on it to keep other pods off the GPUs.

```bash
export NODE_A=$(kubectl get nodes -l role=gpu -o jsonpath='{.items[0].metadata.name}')
export NODE_B=$(kubectl get nodes -l role=gpu -o jsonpath='{.items[1].metadata.name}')
echo "$NODE_A / $NODE_B"
```

**Expected:** two different names. Everything below uses them to pin pods to separate nodes.

---

## Step 2 — (b) and (c) the advertised counts equal the mapped counts

Read what the template promises for this instance type, then read what the nodes advertise. Run it from `architectures/amazon-eks`; it needs PyYAML:

```bash
python3 - <<'PY'
import yaml
class Loader(yaml.SafeLoader): pass
Loader.add_multi_constructor("!", lambda loader, suffix, node: None)   # ignore !Ref, !If, ...
doc = yaml.load(open("assets/eks-add-gpu-nodegroup.yaml"), Loader=Loader)
for table in ("GpuCount", "NicLayout"):
    print(table, "->", doc["Mappings"][table].get("g7e.12xlarge", "MISSING"))
PY

kubectl get nodes -l role=gpu -o jsonpath=\
'{range .items[*]}{.metadata.name}{"\tgpu="}{.status.allocatable.nvidia\.com/gpu}{"\tefa="}{.status.allocatable.vpc\.amazonaws\.com/efa}{"\n"}{end}'
```

**Expected:** for `g7e.12xlarge` the mappings print `GpuCount -> {'Gpus': '2'}` and
`NicLayout -> {'Cards': '1', 'EfaInterfaces': '1', 'PrimaryEfa': 'true', 'SecondaryDeviceIndex': '1'}`, and both nodes
report `gpu=2 efa=1` — the advertised numbers equal the mapped numbers, not merely "more than
zero". A missing `efa` key means the EFA device plugin is not running on that node (check
`kubectl -n kube-system get ds aws-efa-k8s-device-plugin -o wide`); a missing `gpu` key means
the NVIDIA plugin is not (check `kubectl -n nvidia-device-plugin get ds`).

```bash
kubectl -n nvidia-device-plugin get pods -o wide
kubectl -n kube-system get pods -l name=aws-efa-k8s-device-plugin -o wide
```

**Expected:** one `nvidia-device-plugin`, one `gpu-feature-discovery` and one EFA plugin pod per GPU node, `Running`, and **none of those on the system nodes**. The `node-feature-discovery` workers run on every node, the GPU nodes included, because their PCI labels are what the plugin schedules on; its master and gc run on the system nodes.

---

## Step 3 — (d) the local NVMe is a RAID0 array mounted at `/mnt/k8s-disks/0`

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: nvme-check
spec:
  restartPolicy: Never
  nodeName: $NODE_A
  tolerations:
    - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
  containers:
    - name: check
      image: public.ecr.aws/docker/library/busybox:1.36
      command: ["sh","-c","sleep 900"]
      volumeMounts:
        - { name: hostroot, mountPath: /host, readOnly: true }
        - { name: disk0, mountPath: /disk0 }
  volumes:
    - name: hostroot
      hostPath: { path: / }
    - name: disk0
      hostPath: { path: /mnt/k8s-disks/0 }
EOF
kubectl wait --for=condition=Ready pod/nvme-check --timeout=180s

kubectl exec nvme-check -- sh -c 'cat /host/proc/mdstat'
kubectl exec nvme-check -- sh -c 'grep k8s-disks /host/proc/mounts'
kubectl exec nvme-check -- sh -c 'df -h /disk0; dd if=/dev/zero of=/disk0/probe bs=1M count=256 2>&1 | tail -1; rm -f /disk0/probe'
```

**Expected:**

- `/proc/mdstat` lists an **active raid0** array (`md0` or `md127`) whose members are the
  instance-store NVMe devices (`nvme1n1` and friends). A single-disk instance type still gets a
  one-member raid0 array from the AL2023 node bootstrap; seeing a bare `nvme1n1` mounted, or no
  array at all, means the RAID0 strategy was not applied.
- `/proc/mounts` has a line for `/mnt/k8s-disks/0` whose source is that `md` device, with an
  `xfs` or `ext4` filesystem.
- `df -h /disk0` shows the full instance-store capacity (roughly 3.4 TiB for the type's 3.8 TB
  of NVMe on `g7e.12xlarge`, and clearly not the 300 GiB root volume), and the write succeeds with a
  throughput figure. A write that fails, or a size that matches `GpuRootVolumeSize`, means the
  pod is writing to the root EBS volume — which is exactly the mistake this step catches,
  because the disaggregated-inference examples cache model weights there.

```bash
kubectl delete pod nvme-check --wait=false
```

---

## Step 4 — (f) EFA reaches a pod that asks for it, and only such a pod

Two pods, one per node, each asking for one EFA interface:

```bash
for pair in "efa-a:$NODE_A" "efa-b:$NODE_B"; do
  name=${pair%%:*}
  node=${pair#*:}
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $name
spec:
  restartPolicy: Never
  nodeName: $node
  tolerations:
    - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
  containers:
    - name: efa
      image: $EFA_IMAGE
      command: ["sh","-c","sleep 3600"]
      resources:
        limits:   { vpc.amazonaws.com/efa: 1, memory: 8Gi, cpu: "4" }
        requests: { vpc.amazonaws.com/efa: 1, memory: 8Gi, cpu: "4" }
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
EOF
done
kubectl wait --for=condition=Ready pod/efa-a pod/efa-b --timeout=600s
kubectl get pods -o wide efa-a efa-b
```

**Expected:** both `Running`, on **different** nodes (check the `NODE` column). A pod stuck in
`Pending` with `Insufficient vpc.amazonaws.com/efa` means step 2 did not really pass. The first
pull of this image takes a few minutes, and it crosses the NAT gateway: the stack's ECR interface
endpoints serve the Region's private ECR, not `public.ecr.aws`.

```bash
kubectl exec efa-a -- ls -l /dev/infiniband
kubectl exec efa-a -- /opt/amazon/efa/bin/fi_info -p efa
```

**Expected:** `/dev/infiniband/uverbs0` is present, and `fi_info -p efa` prints at least one `provider: efa` entry with
`fabric: efa`, `domain: rdmap...`, `type: FI_EP_RDM`. If `fi_info` prints
`fi_getinfo: -61 (No data available)`, the EFA device is not usable inside the pod — that is a
failure of this architecture, not of libfabric.

Negative control, which is what proves the device plugin is doing the work:

```bash
kubectl run no-efa --image=public.ecr.aws/docker/library/busybox:1.36 --restart=Never \
  --overrides='{"spec":{"nodeName":"'"$NODE_A"'","tolerations":[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}]}}' \
  -- sh -c 'ls /dev/infiniband 2>&1; echo "exit=$?"'
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/no-efa --timeout=180s
kubectl logs no-efa
kubectl delete pod no-efa
```

**Expected:** `ls: /dev/infiniband: No such file or directory` and a non-zero `exit=`. A pod
that never requested EFA must not see the device. If it does, the node is exposing the device
to every container and the plugin's accounting is meaningless.

---

## Step 5 — (e) traffic between the two pods actually runs over EFA

The trap in this step is that almost every transport falls back to TCP and still reports
success. Two independent pieces of evidence are required: libfabric must select the `efa`
provider with no fallback available, **and** the EFA hardware counters on both nodes must move.

### 5.1 — the EFA hardware counters before the run

The counters live on the host. A container's `/sys/class/infiniband/*/ports/1/` has no
`hw_counters` directory, so they are read through a pod that mounts the host's `/sys`, one per
node:

```bash
for pair in "ctr-a:$NODE_A" "ctr-b:$NODE_B"; do
  name=${pair%%:*}
  node=${pair#*:}
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: { name: $name }
spec:
  restartPolicy: Never
  nodeName: $node
  tolerations:
    - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
  containers:
    - name: c
      image: public.ecr.aws/docker/library/busybox:1.36
      command: ["sh","-c","sleep 1800"]
      volumeMounts:
        - { name: sys, mountPath: /hostsys, readOnly: true }
  volumes:
    - name: sys
      hostPath: { path: /sys }
EOF
done
kubectl wait --for=condition=Ready pod/ctr-a pod/ctr-b --timeout=180s

counters() { kubectl exec "$1" -- sh -c \
  'for f in /hostsys/class/infiniband/*/ports/1/hw_counters/*; do \
     printf "%s=%s\n" "$(basename $f)" "$(cat $f 2>/dev/null)"; done' \
  | grep -E '^(tx|rx)_(bytes|pkts)='; }
counters ctr-a > /tmp/efa-a.before
counters ctr-b > /tmp/efa-b.before
cat /tmp/efa-a.before
```

**Expected:** `tx_bytes`, `tx_pkts`, `rx_bytes` and `rx_pkts` for the node's `rdmap*` device.
Note the values, they are the control.

### 5.2 — a libfabric round trip pinned to the EFA provider

`fi_pingpong` serves one client and exits, so a fresh server is started for every client run.
It is started in the foreground of a backgrounded subshell rather than with `nohup ... &` inside
the container, because a process backgrounded inside `kubectl exec` is killed when that exec
session closes and leaves an empty log.

```bash
POD_B_IP=$(kubectl get pod efa-b -o jsonpath='{.status.podIP}')
( kubectl exec efa-b -- sh -c \
    'FI_PROVIDER=efa /opt/amazon/efa/bin/fi_pingpong -p efa -e rdm -I 500 2>&1' \
  > /tmp/pp-server.log 2>&1 ) &
sleep 10
kubectl exec efa-a -- sh -c \
  "FI_PROVIDER=efa /opt/amazon/efa/bin/fi_pingpong -p efa -e rdm -I 500 $POD_B_IP; echo EXIT=\$?" \
  2>&1 | grep -v '^libfabric'
grep -v '^libfabric' /tmp/pp-server.log | tail -7
```

**Expected:** a results table headed
`bytes #sent #ack total time MB/sec usec/xfer Mxfers/sec`, one row per message size from 64 B
up, `EXIT=0`, and the mirror-image table from the server. `-p efa` pins the provider, so there
is no TCP path to fall back to: if EFA cannot carry the traffic the command fails with
`fi_getinfo: -61`, `Unable to resolve address` or `failed to connect: Connection refused`
instead of quietly succeeding. `Connection refused` almost always means the server from a
previous run has already exited — start a new one.

Leave `-S` off. A single pinned size larger than the provider's limit ends with `EXIT=0`, no
table and 4 bytes on the counters, which is the address exchange and nothing else; the sweep the
default performs is both stronger evidence and harder to misread. Add
`FI_LOG_LEVEL=info FI_LOG_PROV=efa` to see the provider select itself, at the cost of burying
the table in log lines.

If this image's libfabric ships the `fabtests` suite instead of `fi_pingpong`,
`/opt/amazon/efa/bin/fi_rdm_pingpong` covers the same ground; check its `--help` for the argument
names, because they are not the same across the two suites, and the expectations are unchanged. `ls /opt/amazon/efa/bin` shows what is there.

### 5.3 — the counters moved

```bash
counters ctr-a > /tmp/efa-a.after
counters ctr-b > /tmp/efa-b.after
paste -d' ' /tmp/efa-a.before /tmp/efa-a.after \
  | awk -F'[= ]' '{printf "  %-10s %12s -> %12s   +%s\n",$1,$2,$4,$4-$2}'
paste -d' ' /tmp/efa-b.before /tmp/efa-b.after \
  | awk -F'[= ]' '{printf "  %-10s %12s -> %12s   +%s\n",$1,$2,$4,$4-$2}'
kubectl delete pod ctr-a ctr-b --wait=false
```

**Expected:** on **both** nodes the transmit and receive byte counters grew by the payload the
table reported, which for the default sweep at `-I 500` is
`500 x (64 + 256 + 1024 + 4096) = 2,720,000` bytes each way, and the packet counters grew with
them, plus a few bytes for the address exchange the run starts with. A delta that accounts for the
payload is what shows the bytes went through the EFA device; a delta far below it means something
else carried them. **Counters that stay flat while `fi_pingpong` reports success must be treated as
a failure**, no matter what the throughput number said: with `-p efa` there is no provider to fall
back to, so flat counters mean the run did not measure what it appears to have measured.

### 5.4 — optional: NCCL across the two nodes

Only if the cluster has the Kubeflow MPI operator installed; the architecture does not install
it, and the two checks above are the ones that gate a merge. Start from the repository's
canonical manifest,
[`micro-benchmarks/nccl-tests/kubernetes/nccl-tests.yaml`](../../../micro-benchmarks/nccl-tests/kubernetes/nccl-tests.yaml),
and change six things for this node shape:

- `slotsPerWorker: 2`, `-np 4`, `-N 2` (two GPUs per node, two nodes)
- `nvidia.com/gpu: 2` and `vpc.amazonaws.com/efa: 1` in both `limits` and `requests`
- `nodeSelector: { node.kubernetes.io/instance-type: g7e.12xlarge }` and **add** a toleration
  for `nvidia.com/gpu:NoSchedule` — the manifest as committed has none, so its worker pods stay
  `Pending` on the tainted GPU nodes of this architecture
- `image: $EFA_IMAGE` instead of the private ECR placeholder
- the same `nodeSelector` and toleration on the launcher: without them it lands on a system node, whose disk cannot hold the image, and is evicted
- drop the `hugepages-2Mi` request unless
  `kubectl get node $NODE_A -o jsonpath='{.status.allocatable.hugepages-2Mi}'` reports a
  non-zero value, otherwise the pods stay `Pending` on a resource the nodes do not have

```bash
kubectl logs -f "$(kubectl get pods -l training.kubeflow.org/job-role=launcher \
  -o jsonpath='{.items[0].metadata.name}')" | grep -Ei 'NET/OFI|Selected Provider|busbw|Using network'
```

**Expected:** `NET/OFI Selected Provider is efa` (once per rank) and an `all_reduce_perf`
bandwidth table. `NET/OFI Selected Provider is tcp`, `NET/Socket`, or no `NET/OFI` line at all
means NCCL used the pod network — the same fallback as in 5.3, and the same verdict. The `nccl-tests` binaries in `$EFA_IMAGE` carry kernels for `sm_80` to `sm_103`, so on the Blackwell RTX PRO GPUs of `g7` and `g7e` (compute capability 12.0) the provider line appears and the test then stops with `no kernel image is available for execution on the device`; the bandwidth table needs an image built for `sm_120`.

---

## Step 6 — clean up the test pods, then the stack

```bash
kubectl delete pod efa-a efa-b --ignore-not-found
```

Delete the stack when the run is finished; `aws cloudformation delete-stack` removes the nested
stacks with it, and the AMI a build produced outlives the stack and has to be deregistered
separately.

---

## Recording the result

A pass is all six observations on the same cluster in one session: (a) two Ready, labelled,
tainted nodes; (b) `gpu=2`; (c) `efa=1`; (d) a raid0 array at `/mnt/k8s-disks/0` with the
instance-store capacity; (e) an EFA-provider round trip **with** moving hardware counters;
(f) `/dev/infiniband` present in the EFA pod and absent in the pod that did not ask for it.
Add the region and date to the instance type's `Launched` entry in
[`../README.md`](../README.md#3-gpu-instance-types) only when all six passed.
