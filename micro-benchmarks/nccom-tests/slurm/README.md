# NCCOM-test on Slurm

## Prerequisits
This guide assumes that you have the following:

- A functional Slurm cluster on AWS.
- `aws-neuronx-tools` installed on all the compute instances.

It is recommended that you use the templates in the architectures [directory](../../architectures)


## Running NCCOM Tests


Copy the file `nccom-tests.sbatch` or its content on your cluster then submit a preprocessing jobs with the command below:

```bash
sbatch nccom-tests.sbatch
```

## Results

The command above will create output file ` logs/nccom-all_reduce_perf_125.out`. The output shold look as below:

```txt
    size(B)    count(elems)    type    time:avg(us)    algbw(GB/s)    busbw(GB/s)
          8               2    fp32           63.52           0.00           0.00
         16               4    fp32           312.4           0.00           0.00
         32               8    fp32          294.82           0.00           0.00
         64              16    fp32           67.97           0.00           0.00
        128              32    fp32           71.11           0.00           0.00
        256              64    fp32           64.78           0.00           0.01
        512             128    fp32           73.99           0.01           0.01
       1024             256    fp32          540.21           0.00           0.00
       2048             512    fp32          772.48           0.00           0.00
       4096            1024    fp32          557.84           0.01           0.01
       8192            2048    fp32          778.62           0.01           0.02
      16384            4096    fp32         1019.27           0.01           0.03
      32768            8192    fp32         1015.71           0.03           0.06
      65536           16384    fp32           981.6           0.06           0.12
     131072           32768    fp32          782.15           0.16           0.31
     262144           65536    fp32          791.38           0.31           0.61
     524288          131072    fp32          343.84           1.42           2.80
Avg bus bandwidth:      0.2346GB/s
```