# Seoul validation record

Validation date: 2026-09-06 UTC. All GPU results in this file came from **2 p6-b300.48xlarge nodes on EKS in ap-northeast-2, with 8 GPU ranks per node and 16 GPU ranks per run**. They establish mechanisms, not the production p5.48xlarge bandwidth class or PCS integration.

## Runtime and test boundaries

The immutable base image is `public.ecr.aws/hpc-cloud/nccl-tests@sha256:a5390d3f0eb50f3e5854085e2ae0fef90b9eb4f3b7486ed4cab0d47864809d4c`. The measured stack used driver version 595.91.07, kernel version 7.0.0-1011-aws, runtime NCCL version 2.30.4, aws-ofi-nccl version 1.19.0, and libfabric version 2.4.0amzn3.0. The Python fixtures added PyTorch version 2.9.0+cu130 on Python version 3.10. PyTorch's build-version helper reports NCCL version 2.27.7; the startup log confirms that the image's `LD_PRELOAD` loaded runtime NCCL version 2.30.4.

The sweep used the checked-in `sweep-rank.sh` through Open MPI version 4.1.7 with its TCP transport restricted to the primary interface. The Python fixtures used `torch-node.sh` through `torchrun`, without MPI. The checkpoint and DataLoader rounds used a different idle p6-b300.48xlarge pair after the first pair was released.

## Baseline, injection, and recovery

The table contains measurements on the original p6-b300.48xlarge pair. Every sweep checked message sizes from 8 B through 2 GiB and reported 0 mismatches in both columns.

| State | Out-of-place bus bandwidth at 2 GiB | In-place bus bandwidth at 2 GiB | EFA evidence |
| --- | --- | --- | --- |
| Healthy baseline | 827.38 GB/s | 829.38 GB/s | Positive byte deltas on both nodes |
| Only `libnccl-net-ofi.so` moved | 828.49 GB/s | 830.45 GB/s | OFI still selected; alternate network-plugin copy remained |
| Entire OFI library directory moved | 14.94 GB/s | 15.53 GB/s | `NET/Socket`; every exposed EFA byte counter changed by 0 B on both nodes |
| Directory restored | 829.44 GB/s | 827.67 GB/s | OFI selected again; positive byte deltas on both nodes |

The all-reduce command was `/opt/nccl-tests/build/all_reduce_perf -b 8 -e 2G -f 2 -g 1 -c 1`. The following rows are actual output at 2 GiB, in the table's order. NCCL reports size in B, count in elements, time in microseconds, algorithm and bus bandwidth in GB/s, and `#wrong` in mismatches; the root field is a rank index.

```text
# size count type redop root time algbw busbw #wrong time algbw busbw #wrong
  2147483648     536870912     float     sum      -1  4866.61  441.27  827.38       0  4854.84  442.34  829.38       0
  2147483648     536870912     float     sum      -1  4860.09  441.86  828.49       0  4848.59  442.91  830.45       0
  2147483648     536870912     float     sum      -1   269469    7.97   14.94       0   259353    8.28   15.53       0
  2147483648     536870912     float     sum      -1  4854.53  442.37  829.44       0  4864.91  441.42  827.67       0
```

Every one of these completed sweeps reported `# Out of bounds values : 0 OK`. Missing-plugin selection is visible in NCCL debug output; it is not an application error. The full directory removal includes network-plugin and tuner copies, so its bandwidth change is not a controlled measurement of transport alone.

## Storage fault and recovery

On the second p6-b300.48xlarge pair, the healthy checkpoint fixture completed with 0 mismatches. The private checkpoint tmpfs had a configured capacity of 32 MiB. The following are actual writer and peer observations:

```text
checkpoint injection: [Errno 28] No space left on device; filler_bytes=33546240 B
checkpoint write retry: [Errno 28] No space left on device
[rank12]:[E906 03:02:35.091834500 ProcessGroupNCCL.cpp:683] [Rank 12] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=3, OpType=ALLREDUCE, NumelIn=1048576, NumelOut=1048576, Timeout(ms)=30000) ran for 30002 milliseconds before timing out.
```

The checkpoint writer's retry deadline was configured as 60 s and the collective deadline as 30 s. After all failed workers exited, the checkpoint filesystem still had 0 B available, and a clean all-reduce sweep on the same second p6-b300.48xlarge pair reached 829.38 GB/s out-of-place and 829.30 GB/s in-place at 2 GiB with 0 mismatches. Its EFA counters increased. A preceding sweep overlapped worker teardown and is excluded from these results.

Removing only the filler and partial write preserved the last successful checkpoint. The recovery output was:

```text
Resuming from last successful checkpoint: next_step=0 (step index).
Storage workload completed: next_step=1 (step index); 0 mismatches.
```

Both node launchers exited successfully after recovery. This validates the bounded write-exhaustion mechanism and the fixture's retry behavior. FSx quota enforcement, an FSx service outage, and the PCS launch path remain **UNVALIDATED**.

## DataLoader fork probe

Both configurations completed on the second p6-b300.48xlarge pair. The first used `fork` with EFA huge pages enabled; the second used `spawn` with EFA huge pages disabled. Each created 2 DataLoader workers per GPU rank after a completed collective.

```text
After first collective: C environment FI_EFA_FORK_SAFE=b'1'; DataLoader start_method=fork.
DataLoader workload completed: 4 batches; 0 mismatches.
After first collective: C environment FI_EFA_FORK_SAFE=b'1'; DataLoader start_method=spawn.
DataLoader workload completed: 4 batches; 0 mismatches.
```

The historical hang **did not reproduce** on this recorded stack. These runs do not establish the earliest mitigated version or the safety of arbitrary CUDA operations inside forked children. The minimal Python environment emitted an optional NumPy-import warning, which was retained in the raw logs.

## Remaining qualification

The production p5.48xlarge PCS wrappers, pre-job drain/requeue and scheduler-log exercise, Enroot/Pyxis execution, and FSx quota setup are **UNVALIDATED**. The historical fork hang and a hardware corruption case were not observed. The inherited p5.48xlarge reference remains the 400 GB/s class at 2 GiB, not the Seoul numbers above.

The local report prepared with this draft retains the exact Kubernetes commands, actual output, discarded attempts, instance IDs, and cleanup evidence. The initial CPU launcher was evicted for insufficient ephemeral storage; the first MPI attempt also failed before CUDA initialization because of interface selection. Neither failure is a GPU fault signature.
