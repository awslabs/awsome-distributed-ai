# P6e-GB200 EKS node group on a Capacity Block

P6e-GB200 is acquired via **EC2 Capacity Blocks for ML**. The EKS node group must launch into that reservation and use the AL2023-ARM-NVIDIA EKS AMI (driver 580 / CDMM).

## Launch template essentials

```jsonc
// EC2 Launch Template (excerpt) -- attach to the EKS managed/self-managed node group
{
  "InstanceType": "p6e-gb200.36xlarge",
  "ImageId": "<AL2023-ARM-NVIDIA EKS AMI v20251103+ for your K8s minor>",   // arm64 (Grace)
  "InstanceMarketOptions": {},                                              // NOT spot
  "CapacityReservationSpecification": {
    "CapacityReservationTarget": {
      "CapacityReservationId": "<cr-xxxxxxxx from your Capacity Block>"
    }
  },
  "NetworkInterfaces": [
    // p6e-gb200.36xlarge exposes multiple EFA NICs; attach per the instance's NIC layout.
    { "DeviceIndex": 0, "InterfaceType": "efa", "NetworkCardIndex": 0 }
    // ... additional EFA NICs ...
  ]
}
```

## Node group sizing

- **18 nodes** = one full `u-p6e-gb200x72` UltraServer (one 72-GPU NVLink domain).
- Keep the group within a single Capacity Block so all instances land in the same NVLink domain (one `nvidia.com/gpu.clique`). Spanning two UltraServers means two cliques bridged by EFA.

## After the nodes join

1. Confirm GFD applied the clique label: `kubectl get nodes -L nvidia.com/gpu.clique`.
2. Install the GPU Operator (`gpu-operator-values.yaml`) and the NVIDIA DRA driver (`nvidia-dra-driver-values.yaml`).
3. Apply `computedomain-example.yaml` and confirm `Domain State: UP`.

## Notes

- **No MIG**: CDMM (driver 580) disables MIG/vGPU on GB200 — do not enable the MIG manager.
- K8s ≥ 1.33 is required for DRA (the ComputeDomain CRD).
- This file is authored-to-spec; substitute your real AMI id, Capacity Reservation id, and NIC layout.
