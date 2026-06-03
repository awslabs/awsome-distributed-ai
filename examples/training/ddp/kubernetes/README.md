## 1. Preparation
The guide assumes that you have the following:

* An Amazon EKS cluster on AWS with x86-based CPU nodes, accessible via `kubectl`.
* An Amazon FSx for Lustre persistent volume claim named `fsx-pv`, you can use an example from [here](https://github.com/aws-samples/aws-do-eks/tree/main/Container-Root/eks/deployment/csi/fsx), if you need to create one.
* Docker 

We recommend that you setup a Kubernetes cluster using the templates in the architectures [directory](../../architectures). 


## 3. Submit training job using container

In this example, you'll learn how to use the official PyTorch Docker image 
and execute the container within Kubernetes using kubeflow training operator. 

Make sure kubeflow training operator is deployed to your cluster:
```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
```

Build the container image:

```bash
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
docker build -t ${REGISTRY}ddp:latest ..
```

Push the container image to the Elastic Container Registry in your account:
```bash
# Create registry if needed
REGISTRY_COUNT=$(aws ecr describe-repositories | grep \"ddp\" | wc -l)
if [ "$REGISTRY_COUNT" == "0" ]; then
        aws ecr create-repository --repository-name ddp
fi

# Login to registry
echo "Logging in to $REGISTRY ..."
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY

# Push image to registry
docker image push ${REGISTRY}ddp:latest
```

Create manifest and launch PyTorchJob:
```bash
export IMAGE_URI=${REGISTRY}ddp:latest
export INSTANCE_TYPE=
export NUM_NODES=2
export CPU_PER_NODE=4
cat ddp-custom-container.yaml-template | envsubst > ddp.yaml

kubectl apply -f ./ddp.yaml
```

Check the status of your training job:
```bash
kubectl get pytorchjob 
kubectl get pods 
```

```text
NAME   STATE     AGE
ddp    Running   16s

NAME                    READY   STATUS    RESTARTS   AGE
etcd-7787559c74-w9gwx   1/1     Running   0          18s
ddp-worker-0            1/1     Running   0          18s
ddp-worker-1            1/1     Running   0          18s
```

Each of the pods produces job logs. 
```bash
kubectl logs ddp-worker-0
```

```text
2024-07-19 04:39:07,890] torch.distributed.run: [WARNING] *****************************************
INFO 2024-07-19 04:39:07,958 Etcd machines: ['http://0.0.0.0:2379']
INFO 2024-07-19 04:39:07,964 Attempting to join next rendezvous
INFO 2024-07-19 04:39:07,965 Observed existing rendezvous state: {'status': 'joinable', 'version': '1', 'participants': [0]}
INFO 2024-07-19 04:39:08,062 Joined rendezvous version 1 as rank 1. Full state: {'status': 'frozen', 'version': '1', 'participants': [0, 1], 'keep_alives': []}
INFO 2024-07-19 04:39:08,062 Waiting for remaining peers.
INFO 2024-07-19 04:39:08,063 All peers arrived. Confirming membership.
INFO 2024-07-19 04:39:08,149 Waiting for confirmations from all peers.
INFO 2024-07-19 04:39:08,161 Rendezvous version 1 is complete. Final state: {'status': 'final', 'version': '1', 'participants': [0, 1], 'keep_alives': ['/torchelastic/p2p/run_none/rdzv/v_1/rank_1', '/torchelastic/p2p/run_none/rdzv/v_1/rank_0'], 'num_workers_waiting': 0}
INFO 2024-07-19 04:39:08,161 Creating EtcdStore as the c10d::Store implementation
...
[RANK 1] Epoch 4991 | Batchsize: 32 | Steps: 8
Epoch 4990 | Training snapshot saved at /fsx/snapshot.pt
[RANK 0] Epoch 4991 | Batchsize: 32 | Steps: 8
[RANK 1] Epoch 4992 | Batchsize: 32 | Steps: 8
[RANK 0] Epoch 4992 | Batchsize: 32 | Steps: 8
[RANK 3] Epoch 4992 | Batchsize: 32 | Steps: 8
[RANK 2] Epoch 4992 | Batchsize: 32 | Steps: 8
[RANK 1] Epoch 4993 | Batchsize: 32 | Steps: 8
[RANK 2] Epoch 4993 | Batchsize: 32 | Steps: 8
...
```


Stop the training job:
```bash
kubectl delete -f ./ddp.yaml
```

Note: Prior to running a new job, please stop any currently running or completed ddp job.