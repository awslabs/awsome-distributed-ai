# PPLX Garden Benchmark

```bash
git clone https://github.com/perplexityai/pplx-garden.git
```

```bash
docker build -t pplx-garden-dev - < ./pplx-garden/docker/dev.Dockerfile && docker build -t pplx-garden -f pplx-garden.Dockerfile .
```

```bash
enroot import -o pplx-garden.sqsh dockerd://pplx-garden
```

```bash
sbatch pplx-garden.sbatch
```

# All-to-All Performance Results

Decode (128 tokens) Dispatch and Combine Median Latency(μs):

|       | My UCCL EP | My pplx-EFA | pplx-EFA | pplx-CX7 | DeepEP-CX7 | x | My UCCL EP | My pplx-EFA | pplx-EFA | pplx-CX7 | DeepEP-CX7 |
|-------|-----------:|------------:|---------:|---------:|-----------:|---|-----------:|------------:|---------:|---------:|-----------:|
| EP128 |            |    424.5    |          |          |            | x |            |    588.5    |          |          |            |
| EP64  |   crash    |    268.6    | 266.7    | 187.5    |   177.9    | x |   crash    |    393.6    | 391.2    | 309.1    |   325.0    |
| EP32  |   358.0    |    230.9    | 229.1    | 153.9    |   159.1    | x |   689.3    |    336.9    | 335.0    | 266.3    |   285.0    |
| EP16  |   301.1    |    218.0    | 214.8    | 110.2    |   123.9    | x |   834.5    |    244.6    | 241.5    | 185.5    |   203.0    |
| EP8   |    66.1    |     50.6    |  49.7    |  50.5    |    42.6    | x |    86.6    |     64.1    |  64.2    |  65.3    |    72.0    |

Prefill (4096 tokens) Dispatch and Combine Median Latency(μs):

| x     | My pplx-EFA |  pplx-EFA |  pplx-CX7 | DeepEP-CX7 | x | My pplx-EFA |  pplx-EFA |  pplx-CX7 | DeepEP-CX7 |
|-------|------------:|----------:|----------:|-----------:|---|------------:|----------:|----------:|-----------:|
| EP128 |   5883.4    |           |           |            | x |  10785.1    |           |           |            |
| EP64  |   5395.6    | 5334.3    | 4665.2    |  5071.6    | x |   9854.9    | 9779.3    | 8771.1    |  5922.7    |
| EP32  |   4605.2    | 4619.0    | 4011.8    |  3680.2    | x |   8286.3    | 8271.5    | 7526.8    |  3565.4    |
| EP16  |   3181.4    | 3196.7    | 2734.8    |  2481.9    | x |   5373.6    | 5379.1    | 1062.2    |  1863.9    |
| EP8   |   1076.0    | 1052.4    | 5071.1    |  1810.3    | x |   1354.0    | 1396.7    | 1405.1    |   962.9    |

# UCCL-EP Benchmark

```bash
git clone https://github.com/uccl-project/uccl.git
```

```bash
docker build -t pplx-garden_uccl-ep -f pplx-garden_uccl-ep.Dockerfile .
```

```bash
enroot import -o pplx-garden_uccl-ep.sqsh dockerd://pplx-garden_uccl-ep
```

```bash
sbatch uccl-pplx-garden.sbatch
```
