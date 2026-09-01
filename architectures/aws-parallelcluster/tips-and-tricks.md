
## Tips and tricks

### Connect to your cluster

To easily login to your cluster via [AWS Systems Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-sessions-start.html) we've included a script `easy-ssh.sh` that you can run like so, assuming `ml-cluster` is the name of your cluster:

```bash
./easy-ssh.sh ml-cluster
```

You'll need a few pre-requisites for this script:
* JQ: `brew install jq`
* aws cli
* `pcluster` cli
* [Session Manager Plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)

Once you've run the script you'll see the following output:

```
Instance Id: i-0096542c11ccb02b5
Os: ubuntu2004
User: ubuntu
Add the following to your ~/.ssh/config to easily connect:

cat <<EOF >> ~/.ssh/config
Host ml-cluster
  User ubuntu
  ProxyCommand sh -c "aws ssm start-session --target i-0095542c11ccb02b5 --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
EOF

Add your ssh keypair and then you can do:

$ ssh ml-cluster

Connecting to ml-cluster...

Starting session with SessionId: ...
root@ip-10-0-24-126:~#
```

1. Add your public key to the file `~/.ssh/authorized_keys`

2. Now paste in the lines from the output of to your terminal, this will add them to your `~/.ssh/config`.

```
cat <<EOF >> ~/.ssh/config
Host ml-cluster
  User ubuntu
  ProxyCommand sh -c "aws ssm start-session --target i-0095542c11ccb02b5 --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
EOF
```
3. Now you ssh in, assuming `ml-cluster` is the name of your cluster with:

```
ssh ml-cluster
```

### Custom Slurm Settings

Parallel Cluster, as of this writing (v3.8.0), does not automatically set the correct number of sockets and cores-per-socket in Slurm partitions. For ML training on multiple GPUs, this should not have adverse impact on performance since NCCL does the topology detection. However, it might have impact on the training performance for single-GPU training or inference, or in the rare case to support Slurm affinity/binding feature.

Using the example from [here](https://github.com/aws/aws-parallelcluster/issues/5797), a `p5.48xlarge` should have just two sockets, but Slurm shows the partition to have more than that.

```bash
$ ssh p5-st-p5-1 /opt/slurm/sbin/slurmd -C
NodeName=p5-st-p5-1 CPUs=192 Boards=1 SocketsPerBoard=2 CoresPerSocket=48 ThreadsPerCore=2 RealMemory=2047961 UpTime=2-00:08:02

$ sinfo -o '%9P %4c %8z %8X %8Y %8Z'
PARTITION CPUS S:C:T    SOCKETS  CORES    THREADS
p5*       192  192:1:1  192      1        1
```

Should you want to correct the partition configuration, edit your cluster configuration file (e.g., the `distributed-training-*.yaml` file) and a new entry `Scheduling` / `SlurmQueues` / `ComputeResources` / `CustomSlurmSettings` similar to below.

```yaml
Scheduling:
  SlurmQueues:
    - Name: compute-gpu
      ...
      ComputeResources:
        - Name: distributed-ml
          InstanceType: p5.48xlarge
          ...
          CustomSlurmSettings:
            Sockets: 2
            CoresPerSocket: 48
```

Each instance type has its own `Sockets` and `CoresPerSocket` values. Below are for instance types commonly used for distributed training.

| Instance type | `Sockets` | `CoresPerSocket` |
| ------------- | --------- | ---------------- |
| p5.48xlarge   | 2         | 48               |
| p4de.24xlarge | 2         | 24               |
| p4d.24xlarge  | 2         | 24               |

For other instance types, you'd need to run an instance to check the values.

