GEMM peak 167.9 TFLOP/s per GPU, memory read 663 GB/s per GPU, ridge 253 FLOP per byte

| Stage | Transport | Attention | Saving | Work | Seconds | Tokens/s | GPU-s per 1M tokens | MFU | Share of your GEMM peak | Peak memory GiB | vs baseline | vs previous step | Goodput |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | socket | eager | none | 5 x 2,048 | 13.19 | 19,863 | 806 | 7.0 % | 8.5 % | 26.68 | 1.00x |  |  |
| probe-socket-eager | socket | eager | none | 5 x 1,024 | 12.94 | 10,121 | 1581 | 3.4 % | 4.1 % | 9.70 | (1) |  |  |
| sdpa-socket | socket | sdpa | none | 5 x 2,048 | 13.57 | 19,312 | 828 | 6.8 % | 8.3 % | 8.34 | 0.97x | 0.97x |  |
| efa | efa | eager | none | 5 x 2,048 | 5.76 | 45,470 | 352 | 16.1 % | 19.4 % | 26.68 | 2.29x | 2.29x |  |
| probe-efa-eager | efa | eager | none | 5 x 1,024 | 3.46 | 37,870 | 423 | 12.7 % | 15.3 % | 9.70 | (2) |  |  |
| sdpa | efa | sdpa | none | 5 x 2,048 | 3.39 | 77,275 | 207 | 27.3 % | 33.0 % | 8.34 | 3.89x | 1.70x |  |

(1) probe-socket-eager vs baseline: not compared, different workload
(2) probe-efa-eager vs baseline: not compared, different workload

MFU: the model's FLOP per second / (16 x the dense BF16 peak, 202.8 TFLOP/s per GPU for NVIDIA RTX PRO 4500 Blackwell Server Edition, estimated from NVIDIA specifications)
Share of your GEMM peak: the same FLOP per second / (16 x your GEMM peak, 167.9 TFLOP/s per GPU)
