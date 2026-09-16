# GPU and EFA device recovery

The diagnostic owner is [validation/gpu-cluster-healthcheck](https://github.com/awslabs/awsome-distributed-ai/tree/main/validation/gpu-cluster-healthcheck). Run its checks on healthy, unavailable and recovered devices. The helper in this directory performs only the selected device operation; it does not decide whether the node is healthy.

Use a dedicated pair with an external controller and an authorized PCS Slurm reboot route. Keep the instances and reserved capacity throughout the exercise. A GPU PCI removal changes operating-system visibility. It does not reproduce physical damage or establish NVIDIA Xid 79. An EFA driver unbind changes the selected PCI function's binding and RDMA domain; it is a different intervention from hiding an NCCL plugin.

## Prepare the target and evidence

1. Record the live account, Region, instance identity, Slurm node, GPU UUID and PCI BDF, both EFA domains and their PCI BDFs, and the management interface and default route. Fix expected counts before injecting a fault.
2. Keep the controller on the login node or another host. Confirm that its Slurm administrator can request a reboot of the exact target with `scontrol reboot`. PCS can replace an instance after an EC2 or guest reboot outside Slurm; use the [PCS Slurm procedure](https://docs.aws.amazon.com/pcs/latest/userguide/slurm-reboot.html). Keep the target's logs on that controller or in the event's private evidence destination.
3. Install `device-fault.sh` as `/usr/local/sbin/aim344-device-fault`, owned by root with mode `0755`. Create `/etc/aim344-device-fault.json`, owned by root with mode `0600`. Derive every identity from that target. The helper requires these fields:

   ```json
   {
     "instance_id": "<verified instance identifier>",
     "slurm_node": "<registered Slurm node>",
     "gpu_uuid": "<selected GPU UUID>",
     "gpu_bdf": "<selected GPU PCI BDF>",
     "gpu_vendor": "0x10de",
     "gpu_device": "<observed PCI device ID>",
     "efa_bdf": "<selected EFA PCI BDF>",
     "efa_vendor": "0x1d0f",
     "efa_device": "<observed PCI device ID>",
     "efa_rdma_device": "<observed kernel RDMA name>",
     "management_bdf": "<management ENA PCI BDF>",
     "management_interface": "<interface on the default route>",
     "participant_user": "ubuntu",
     "slurm_bin": "/opt/aws/pcs/scheduler/slurm-25.05/bin"
   }
   ```

4. Run `sudo /usr/local/sbin/aim344-device-fault inspect` through the maintenance route. Preserve its output and confirmation tokens outside the target. A token names the instance, operation and GPU UUID or EFA BDF. A token for one operation does not authorize another.
5. Give participants access to the root-owned helper through the facilitator's maintenance session. Do not grant a general root shell for device manipulation. The facilitator keeps reboot and scheduler-resume authority.
6. Record mount sources, filesystem UUIDs, image hashes, Prolog configuration and telemetry state. Store the restore script on the root volume, outside a bind mount that disappears at boot. Verify the existing boot script preserves the staged volume.

Run checks with identifiers `0`, `2`, `3` and `6` on both hosts before the exercise. Run identifiers `1` and `4` exclusively before the session; record and stop the actual native service or container that owns telemetry, then restore its previous state afterward. Keep the raw DCGM tests and skipped coverage. The Level 4 command alone does not establish that EUD ran. Invoke Check 5 once from the allocation coordinator using `10.healthcheck-nccl.sh`, with the native-kernel staged image. Keep both correctness columns and EFA byte deltas.

## Run one fault

GPU removal uses an idle node. Active-removal trials on this stack left the removal helper pending and guest shutdown waiting for that process, including a trial with persistence disabled. Preserve those trials as diagnostic evidence; do not repeat the active GPU operation in the participant session.

For an idle GPU trial, end the allocation, record and pause the target's actual telemetry owner, and record the selected GPU's persistence mode before disabling it. Keep the prior mode on the root volume so recovery can restore it:

```bash
sudo install -d -m 0755 /var/lib/aim344-device-recovery
nvidia-smi -i "$GPU_UUID" --query-gpu=persistence_mode --format=csv,noheader,nounits \
  | sudo tee /var/lib/aim344-device-recovery/selected-persistence-mode.txt
sudo nvidia-smi -i "$GPU_UUID" --persistence-mode=0
```

The root helper requires an empty target allocation, disabled persistence and no compute process on the selected GPU. Restore the saved persistence mode and telemetry state after recovery. Qualify this exact preparation with at least 3 consecutive idle recovery cycles before adopting it.

For active EFA validation, use a dedicated job named `aim344-*` to run `workload.py device` through `torch-node.sh` on the pair. Set `AIM344_EFA_IFACE` to the allowlisted physical RDMA device. The workload logs progress after completed, correctness-checked collectives. Preserve two advancing progress records and that EFA's increasing counters before injection. Immediately before mutation, confirm that the same job is running and that its latest progress record is recent. Abort injection if progress stopped during preparation. Require at least 3 consecutive active EFA recovery cycles before adopting that operation.

For the idle EFA trial, reserve the pair before draining and keep that allocation idle during the unbind. After the mutation, invoke Check 5 once from its coordinator to observe communication with the remaining device. Preserve that result alongside checks 2 and 6, including a completed collective if the remaining path serves it. Cancel the held allocation before rebind. An active fault trial uses the workload allocation and preserves its failure or continued progress separately.

Drain only the verified target. Draining prevents new jobs; it does not stop the current allocation:

```bash
scontrol update NodeName="$TARGET_NODE" State=DRAIN Reason=aim344-device-recovery
```

The target helper rejects other instances, changed device identities, a management-interface target, another user's job, a job outside the `aim344-*` name scope, a supplied job identifier that is no longer present, an incorrect token, and a node without this drain reason. For idle GPU removal, use the saved exact token in the maintenance session:

```bash
sudo /usr/local/sbin/aim344-device-fault gpu-remove --confirm "$GPU_REMOVE_TOKEN"
```

For the independent active EFA round, after complete GPU recovery, use the active job identifier:

```bash
sudo /usr/local/sbin/aim344-device-fault efa-unbind --job "$FAULT_JOB_ID" --confirm "$EFA_UNBIND_TOKEN"
```

Record the helper result, application log, kernel journal, device enumeration and scheduler state. GPU removal can remain pending while existing collective contexts continue to execute. Record continued progress until cancellation separately from a spontaneous workload error, and compare PCI visibility with NVIDIA inventory. Run suite identifiers `0` and `3` for GPU diagnosis and identifiers `2` and `6` for EFA diagnosis through the maintenance route. A drained node cannot accept an ordinary diagnostic batch job. Keep a command that hangs as an incomplete check, with its external deadline and process state. A userspace timeout cannot guarantee termination of an uninterruptible kernel task.

## Recover and qualify reuse

Stop the exact faulted job from the external controller. Preserve logs before reboot. For EFA, after the old allocation and processes have ended, try the allowlisted rebind:

```bash
sudo /usr/local/sbin/aim344-device-fault efa-rebind --confirm "$EFA_REBIND_TOKEN"
```

A rebind restores device availability for new processes; it does not repair an existing communicator. Use the verified reboot route when the GPU removal or an incomplete EFA operation requires it:

```bash
scontrol reboot reason=aim344-device-recovery "$TARGET_NODE"
scontrol show node "$TARGET_NODE"
```

Request the reboot only after the target is drained and its allocation has ended. Use the basic `scontrol reboot` command shown above so the existing drain remains in place. Do not specify `nextstate=DOWN`, which triggers PCS replacement, or `nextstate=RESUME`, which clears the drain before qualification. Verify the observed scheduler state during qualification. Wait for a changed boot identifier on the same EC2 instance and a working maintenance route. Keep the node drained. Restore its existing staging mount without formatting storage, then restore the fixture from the root-volume copy:

```bash
sudo bash /var/lib/aim344-device-recovery/restore-runtime.sh ubuntu /opt/aim344
```

The runtime helper creates the private 32 MiB tmpfs and its marker when absent. It creates `last.json` with step index 0 only when the record is absent, and preserves an existing valid record. This seed records a fixture step; it contains no model or optimizer state.

Restore the selected GPU's recorded persistence mode before returning it to service:

```bash
prior=$(sudo cat /var/lib/aim344-device-recovery/selected-persistence-mode.txt)
case "$prior" in Enabled) mode=1;; Disabled) mode=0;; *) exit 1;; esac
sudo nvidia-smi -i "$GPU_UUID" --persistence-mode="$mode"
```

Restore the prior telemetry owner, verify `slurmd` and the configured Prolog, and compare the image paths and device identities with the saved inventory. Re-run suite identifiers `0`, `2`, `3` and `6` while the node remains drained. Resume only after expected device counts, per-device connectivity and required runtime paths are restored, and the drain reason still belongs to this exercise. Review cumulative-counter warnings against the saved baseline; preserve their WARN status in the record:

```bash
scontrol show node "$TARGET_NODE"
scontrol update NodeName="$TARGET_NODE" State=RESUME
```

Take a fresh allocation. Invoke Check 5 once, verify correctness and per-device EFA traffic, and complete the storage workload using the recreated private fixture. A returned SSH/SSM session is only one recovery checkpoint. Keep elapsed time from mutation through complete reuse, including failed attempts and manual intervention.

Keep the suite's severity unchanged. `ISOLATE` from a known administratively removed device means the node must remain out of service until recovered and qualified; it does not by itself prove that an instance replacement is required. For an unexplained fault, retain the raw evidence and use the platform's established repair or replacement process.
