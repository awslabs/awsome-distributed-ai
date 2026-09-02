# UCCL-EP Benchmark

<https://uccl-project.github.io/posts/uccl-ep/>

```bash
docker build -t uccl-ep -f uccl-ep.Dockerfile .
```

```bash
enroot import -o ./uccl-ep.sqsh dockerd://uccl-ep
```

test_internode.sbatch

```bash
sbatch test_internode.sbatch
```

test_intranode.sbatch

```bash
sbatch test_intranode.sbatch
```

|   Type    | Dispatch #EP  | Bottleneck bandwidth | Combine #EP  | Bottleneck bandwidth |
|:---------:|:-------------:|:--------------------:|:------------:|:--------------------:|
| Intranode | 8             | 318-321 GB/s (NVLink)| 8            | 320-323 GB/s (NVLink)|
| Internode | 16            | 48-54 GB/s (RDMA)    | 16           | 15-19 GB/s (RDMA)    |
| Internode | 24            | 52-55 GB/s (RDMA)    | 24           | 23-29 GB/s (RDMA)    |
| Internode | 32            | 53-55 GB/s (RDMA)    | 32           | 40-42 GB/s (RDMA)    |

test_low_latency.sbatch

```bash
sbatch test_low_latency.sbatch
```
