# NCCL Tests on AWS Batch

1. First create a VPC and Subnet in the same AZ as your capacity block. You can use the following template:

[<kbd> <br> 1-Click Deploy 🚀 <br> </kbd>](https://console.aws.amazon.com/cloudformation/home?#/stacks/quickcreate?templateURL=https%3A%2F%2Fawsome-distributed-training.s3.amazonaws.com%2Ftemplates%2F2.vpc-one-az.yaml&stackName=aws-batch-vpc)

2. Next you can deploy the AWS Batch template included in this PR, where `cr-1234567890` is the id of your capacity block and `aws-batch-vpc` is the name of the vpc stack you created above.

```bash
git clone https://github.com/awslabs/awsome-distributed-ai.git
cd awsome-distributed-ai/architectures/aws-batch
aws cloudformation create-stack --stack-name aws-batch-p5 \
                                --template-body file://0.aws-batch-distributed-training-p5.yaml \
                                --parameters ParameterKey=VPCStackParameter,ParameterValue="aws-batch-vpc" \
                                             ParameterKey=CapacityBlockId,ParameterValue="cr-1234567890" \
                                --capabilities CAPABILITY_NAMED_IAM
```

3. Next you can submit a job in the aws batch console, by default the NCCLTest Job Definition uses the pre-built container image:

```
public.ecr.aws/hpc-cloud/nccl-tests:latest
```

4. Check the output by going to `Job` > `Nodes` > `Log Stream`

```
[1,0]<stdout>:#                                                              out-of-place                     in-place                                                                                                                                                                                              
[1,0]<stdout>:#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw  #wrong                                                                                                                                                                                                
[1,0]<stdout>:#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)     
[1,0]<stdout>:           8             2     float     sum      -1    12.90    0.00    0.00      0    12.87    0.00    0.00      0
[1,0]<stdout>:          16             4     float     sum      -1    12.88    0.00    0.00      0    12.83    0.00    0.00      0
[1,0]<stdout>:          32             8     float     sum      -1    22.25    0.00    0.00      0    13.34    0.00    0.00      0
[1,0]<stdout>:          64            16     float     sum      -1    14.48    0.00    0.01      0    14.37    0.00    0.01      0
[1,0]<stdout>:         128            32     float     sum      -1    15.47    0.01    0.01      0    15.27    0.01    0.01      0
[1,0]<stdout>:         256            64     float     sum      -1    15.72    0.02    0.03      0    15.33    0.02    0.03      0
[1,0]<stdout>:         512           128     float     sum      -1    16.19    0.03    0.06      0    15.52    0.03    0.06      0
[1,0]<stdout>:        1024           256     float     sum      -1    16.43    0.06    0.11      0    15.61    0.07    0.11      0
[1,0]<stdout>:        2048           512     float     sum      -1    16.62    0.12    0.22      0    15.90    0.13    0.23      0
[1,0]<stdout>:        4096          1024     float     sum      -1    16.76    0.24    0.43      0    16.06    0.26    0.45      0
[1,0]<stdout>:        8192          2048     float     sum      -1    16.85    0.49    0.85      0    16.20    0.51    0.88      0
[1,0]<stdout>:       16384          4096     float     sum      -1    17.46    0.94    1.64      0    16.78    0.98    1.71      0
[1,0]<stdout>:       32768          8192     float     sum      -1    18.78    1.74    3.05      0    18.09    1.81    3.17      0
[1,0]<stdout>:       65536         16384     float     sum      -1    19.09    3.43    6.01      0    18.26    3.59    6.28      0
[1,0]<stdout>:      131072         32768     float     sum      -1    19.37    6.77   11.84      0    18.57    7.06   12.35      0
[1,0]<stdout>:      262144         65536     float     sum      -1    20.27   12.94   22.64      0    20.13   13.02   22.78      0
[1,0]<stdout>:      524288        131072     float     sum      -1    21.17   24.77   43.35      0    20.96   25.01   43.77      0
[1,0]<stdout>:     1048576        262144     float     sum      -1    28.56   36.71   64.24      0    28.37   36.96   64.68      0
[1,0]<stdout>:     2097152        524288     float     sum      -1    43.45   48.27   84.47      0    43.39   48.34   84.59      0
[1,0]<stdout>:     4194304       1048576     float     sum      -1    55.44   75.66  132.41      0    55.00   76.27  133.46      0
[1,0]<stdout>:     8388608       2097152     float     sum      -1    84.29   99.52  174.16      0    83.22  100.80  176.40      0
[1,0]<stdout>:    16777216       4194304     float     sum      -1    124.0  135.33  236.82      0    123.4  135.92  237.86      0
[1,0]<stdout>:    33554432       8388608     float     sum      -1    198.6  168.93  295.63      0    198.3  169.17  296.06      0
[1,0]<stdout>:    67108864      16777216     float     sum      -1    326.6  205.45  359.54      0    326.5  205.55  359.71      0
[1,0]<stdout>:   134217728      33554432     float     sum      -1    592.8  226.39  396.19      0    592.1  226.69  396.71      0
[1,0]<stdout>:   268435456      67108864     float     sum      -1   1121.4  239.38  418.91      0   1121.2  239.43  419.00      0
[1,0]<stdout>:   536870912     134217728     float     sum      -1   2165.3  247.94  433.90      0   2166.4  247.81  433.67      0
[1,0]<stdout>:  1073741824     268435456     float     sum      -1   4253.0  252.47  441.81      0   4255.4  252.33  441.57      0
[1,0]<stdout>:  2147483648     536870912     float     sum      -1   8463.7  253.73  444.02      0   8466.5  253.65  443.88      0
[1,0]<stdout>:  4294967296    1073741824     float     sum      -1    15791  271.98  475.97      0    15773  272.29  476.51      0
[1,0]<stdout>:  8589934592    2147483648     float     sum      -1    31289  274.53  480.43      0    31294  274.49  480.36      0
[1,0]<stdout>: 17179869184    4294967296     float     sum      -1    62239  276.03  483.05      0    62200  276.20  483.35      0
[1,0]<stdout>:# Out of bounds values : 0 OK
[1,0]<stdout>:# Avg bus bandwidth    : 156.742 
```