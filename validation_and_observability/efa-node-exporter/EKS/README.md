# EFA node Exporter for Prometheus on EKS

Here we will show how to setup the EFA node Exporter for Prometheus on an Amazon EKS cluster with these [helm-charts](https://github.com/prometheus-community/helm-charts/tree/main/charts/prometheus-node-exporter).

# 1. Environment variables

Export these variables to setup your environment first:

```bash
export AWS_REGION=us-west-2
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export IMAGE=efa-node-exporter # Docker Image
export TAG=":1.0.0" # Do not specify tag as "latest"
export LOCAL_PORT=9000 # Local port to curl prometheus metrics
```

# 2. Build Docker Image

To build the Docker image:

```bash
git clone https://github.com/awslabs/awsome-distributed-training.git
cd awsome-distributed-training/validation_and_observability/3.efa-node-exporter/

docker build -t ${REGISTRY}${IMAGE}${TAG} -f Dockerfile .
```

# 3. Push Docker Image to ECR

Next, push the Docker image to ECR:

```bash
echo "Logging in to $REGISTRY ..."
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY

# Create registry if it does not exist
REGISTRY_COUNT=$(aws ecr describe-repositories | grep ${IMAGE} | wc -l)
if [ "$REGISTRY_COUNT" == "0" ]; then
        echo ""
        echo "Creating repository ${IMAGE} ..."
        aws ecr create-repository --repository-name ${IMAGE}
fi

# Push image
docker image push ${REGISTRY}${IMAGE}${TAG}
```

# 4. Add Helm repo 

Before we can install the helm chart, we need to add the repo like below:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

# 5. Install Helm chart

We have customized the [values.yaml](https://github.com/prometheus-community/helm-charts/blob/main/charts/prometheus-node-exporter/values.yaml) in `efa-exporter-values-temp.yaml`. Substitute environment variables to generate `efa-exporter-values.yaml` like below:

```bash
envsubst < ./efa-exporter-values-temp.yaml > efa-exporter-values.yaml
```
Next you can install the chart as below:

```bash
helm install efa-node-exporter -f efa-exporter-values.yaml prometheus-community/prometheus-node-exporter
```
Once done you can see the chart as below:

```bash
root@cb9511473ccc:/eks/deployment/distributed-training/pytorch/pytorchjob/efa-node-exporter/prometheus-node-exporter# helm list
NAME              NAMESPACE REVISION  UPDATED                                 STATUS    CHART                           APP VERSION
efa-node-exporter default   1         2024-05-31 18:19:31.122892691 +0000 UTC deployed  prometheus-node-exporter-4.34.0 1.8.0
```

In addition, you will see efa-node-exporter pods starting up as well like below, one pod per node in the cluster:

```bash
root@cb9511473ccc:/eks/deployment/distributed-training/pytorch/pytorchjob/efa-node-exporter/prometheus-node-exporter# k get pods | grep 'efa'
efa-node-exporter-prometheus-node-exporter-ctwcf   1/1     Running     0          4d10h
efa-node-exporter-prometheus-node-exporter-r6kvl   1/1     Running     0          4d10h
efa-node-exporter-prometheus-node-exporter-vh2zg   1/1     Running     0          4d10h
```
Finally, you will also see a new service like below. 

```bash
root@cb9511473ccc:/eks/deployment/distributed-training/pytorch/pytorchjob/efa-node-exporter/prometheus-node-exporter# kubectl get service
NAME                                         TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)     AGE
efa-node-exporter-prometheus-node-exporter   ClusterIP   10.100.243.108   <none>        9100/TCP    4d10h
```
Note, the default port is 9100. If you wish to change it, you can do so in the following lines in `efa-exporter-values.yaml`:

```bash
service:
  enabled: true
  type: ClusterIP
  port: 9100
  targetPort: 9100
  nodePort:
  portName: metrics
```

# 6. Port-forwarding

Once the helm chart is installed, you can port forward as below. 

```bash
kubectl port-forward svc/efa-node-exporter-prometheus-node-exporter ${LOCAL_PORT}:9100
```

# 7. Verify

To verify, open another shell on the same node and try below to see the metrics

```bash
curl http://127.0.0.1:${LOCAL_PORT}/metrics 
```

You can grep 'efa' to see something like:

```bash
root@cb9511473ccc:/eks# curl http://127.0.0.1:${LOCAL_PORT}/metrics | grep "efa"
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0# HELP node_amazonefa_info Non-numeric data from /sys/class/infiniband/<device>, value is always 1.
# TYPE node_amazonefa_info gauge
node_amazonefa_info{device="rdmap144s27"} 1
node_amazonefa_info{device="rdmap160s27"} 1
node_amazonefa_info{device="rdmap16s27"} 1
node_amazonefa_info{device="rdmap32s27"} 1
# HELP node_amazonefa_lifespan Lifespan of the port
# TYPE node_amazonefa_lifespan counter
node_amazonefa_lifespan{device="rdmap144s27",port="1"} 12
node_amazonefa_lifespan{device="rdmap160s27",port="1"} 12
node_amazonefa_lifespan{device="rdmap16s27",port="1"} 12
node_amazonefa_lifespan{device="rdmap32s27",port="1"} 12
# HELP node_amazonefa_rdma_read_bytes Number of bytes read with RDMA
# TYPE node_amazonefa_rdma_read_bytes counter
node_amazonefa_rdma_read_bytes{device="rdmap144s27",port="1"} 1.047241117296e+12
node_amazonefa_rdma_read_bytes{device="rdmap160s27",port="1"} 1.04201975025e+12
node_amazonefa_rdma_read_bytes{device="rdmap16s27",port="1"} 1.047241667482e+12
node_amazonefa_rdma_read_bytes{device="rdmap32s27",port="1"} 1.047241117316e+12
# HELP node_amazonefa_rdma_read_resp_bytes Number of read reponses bytes with RDMA
# TYPE node_amazonefa_rdma_read_resp_bytes counter
node_amazonefa_rdma_read_resp_bytes{device="rdmap144s27",port="1"} 1.04047386624e+12
node_amazonefa_rdma_read_resp_bytes{device="rdmap160s27",port="1"} 1.035878016126e+12
node_amazonefa_rdma_read_resp_bytes{device="rdmap16s27",port="1"} 1.038950461928e+12
node_amazonefa_rdma_read_resp_bytes{device="rdmap32s27",port="1"} 1.04260614144e+12
# HELP node_amazonefa_rdma_read_wr_err Number of read write errors with RDMA
# TYPE node_amazonefa_rdma_read_wr_err counter
node_amazonefa_rdma_read_wr_err{device="rdmap144s27",port="1"} 0
node_amazonefa_rdma_read_wr_err{device="rdmap160s27",port="1"} 0
node_amazonefa_rdma_read_wr_err{device="rdmap16s27",port="1"} 0
node_amazonefa_rdma_read_wr_err{device="rdmap32s27",port="1"} 0
# HELP node_amazonefa_rdma_read_wrs Number of read rs with RDMA
# TYPE node_amazonefa_rdma_read_wrs counter
node_amazonefa_rdma_read_wrs{device="rdmap144s27",port="1"} 3.416044e+06
node_amazonefa_rdma_read_wrs{device="rdmap160s27",port="1"} 3.238688e+06
node_amazonefa_rdma_read_wrs{device="rdmap16s27",port="1"} 3.338488e+06
node_amazonefa_rdma_read_wrs{device="rdmap32s27",port="1"} 3.464577e+06
# HELP node_amazonefa_rdma_write_bytes Number of bytes wrote with RDMA
# TYPE node_amazonefa_rdma_write_bytes counter
node_amazonefa_rdma_write_bytes{device="rdmap144s27",port="1"} 0
node_amazonefa_rdma_write_bytes{device="rdmap160s27",port="1"} 0
node_amazonefa_rdma_write_bytes{device="rdmap16s27",port="1"} 0
node_amazonefa_rdma_write_bytes{device="rdmap32s27",port="1"} 0
# HELP node_amazonefa_rdma_write_recv_bytes Number of bytes wrote and received with RDMA
# TYPE node_amazonefa_rdma_write_recv_bytes counter
node_amazonefa_rdma_write_recv_bytes{device="rdmap144s27",port="1"} 0
node_amazonefa_rdma_write_recv_bytes{device="rdmap160s27",port="1"} 0
node_amazonefa_rdma_write_recv_bytes{device="rdmap16s27",port="1"} 0
node_amazonefa_rdma_write_recv_bytes{device="rdmap32s27",port="1"} 0
# HELP node_amazonefa_rdma_write_wr_err Number of bytes wrote wr with error RDMA
# TYPE node_amazonefa_rdma_write_wr_err counter
node_amazonefa_rdma_write_wr_err{device="rdmap144s27",port="1"} 0
node_amazonefa_rdma_write_wr_err{device="rdmap160s27",port="1"} 0
node_amazonefa_rdma_write_wr_err{device="rdmap16s27",port="1"} 0
node_amazonefa_rdma_write_wr_err{device="rdmap32s27",port="1"} 0
# HELP node_amazonefa_rdma_write_wrs Number of bytes wrote wrs RDMA
# TYPE node_amazonefa_rdma_write_wrs counter
node_amazonefa_rdma_write_wrs{device="rdmap144s27",port="1"} 0
node_amazonefa_rdma_write_wrs{device="rdmap160s27",port="1"} 0
node_amazonefa_rdma_write_wrs{device="rdmap16s27",port="1"} 0
node_amazonefa_rdma_write_wrs{device="rdmap32s27",port="1"} 0
# HELP node_amazonefa_recv_bytes Number of bytes recv bytes
# TYPE node_amazonefa_recv_bytes counter
node_amazonefa_recv_bytes{device="rdmap144s27",port="1"} 6.858286312e+09
node_amazonefa_recv_bytes{device="rdmap160s27",port="1"} 5.331667316e+09
node_amazonefa_recv_bytes{device="rdmap16s27",port="1"} 6.187744962e+09
node_amazonefa_recv_bytes{device="rdmap32s27",port="1"} 7.275998544e+09
# HELP node_amazonefa_recv_wrs Number of bytes recv wrs
# TYPE node_amazonefa_recv_wrs counter
node_amazonefa_recv_wrs{device="rdmap144s27",port="1"} 3.394439e+06
node_amazonefa_recv_wrs{device="rdmap160s27",port="1"} 3.222012e+06
node_amazonefa_recv_wrs{device="rdmap16s27",port="1"} 3.319097e+06
node_amazonefa_recv_wrs{device="rdmap32s27",port="1"} 3.441609e+06
# HELP node_amazonefa_rx_bytes Number of bytes received
# TYPE node_amazonefa_rx_bytes counter
node_amazonefa_rx_bytes{device="rdmap144s27",port="1"} 1.054099403608e+12
node_amazonefa_rx_bytes{device="rdmap160s27",port="1"} 1.047351417566e+12
node_amazonefa_rx_bytes{device="rdmap16s27",port="1"} 1.053429412444e+12
node_amazonefa_rx_bytes{device="rdmap32s27",port="1"} 1.05451711586e+12
# HELP node_amazonefa_rx_drops Number of packets droped
# TYPE node_amazonefa_rx_drops counter
node_amazonefa_rx_drops{device="rdmap144s27",port="1"} 0
node_amazonefa_rx_drops{device="rdmap160s27",port="1"} 0
node_amazonefa_rx_drops{device="rdmap16s27",port="1"} 0
node_amazonefa_rx_drops{device="rdmap32s27",port="1"} 0
# HELP node_amazonefa_rx_pkts Number of packets received
# TYPE node_amazonefa_rx_pkts counter
node_amazonefa_rx_pkts{device="rdmap144s27",port="1"} 6.810483e+06
node_amazonefa_rx_pkts{device="rdmap160s27",port="1"} 6.4607e+06
node_amazonefa_rx_pkts{device="rdmap16s27",port="1"} 6.657585e+06
node_amazonefa_rx_pkts{device="rdmap32s27",port="1"} 6.906186e+06
# HELP node_amazonefa_send_bytes Number of bytes send
# TYPE node_amazonefa_send_bytes counter
node_amazonefa_send_bytes{device="rdmap144s27",port="1"} 6.92065338e+09
node_amazonefa_send_bytes{device="rdmap160s27",port="1"} 6.290013412e+09
node_amazonefa_send_bytes{device="rdmap16s27",port="1"} 8.447687166e+09
node_amazonefa_send_bytes{device="rdmap32s27",port="1"} 4.77018732e+09
# HELP node_amazonefa_send_wrs Number of wrs send
# TYPE node_amazonefa_send_wrs counter
node_amazonefa_send_wrs{device="rdmap144s27",port="1"} 3.401962e+06
node_amazonefa_send_wrs{device="rdmap160s27",port="1"} 3.331132e+06
node_amazonefa_send_wrs{device="rdmap16s27",port="1"} 3.577494e+06
node_amazonefa_send_wrs{device="rdmap32s27",port="1"} 3.161853e+06
# HELP node_amazonefa_tx_bytes Number of bytes transmitted
# TYPE node_amazonefa_tx_bytes counter
node_amazonefa_tx_bytes{device="rdmap144s27",port="1"} 1.04739451962e+12
node_amazonefa_tx_bytes{device="rdmap160s27",port="1"} 1.042168029538e+12
node_amazonefa_tx_bytes{device="rdmap16s27",port="1"} 1.047398149094e+12
node_amazonefa_tx_bytes{device="rdmap32s27",port="1"} 1.04737632876e+12
# HELP node_amazonefa_tx_pkts Number of packets transmitted
# TYPE node_amazonefa_tx_pkts counter
node_amazonefa_tx_pkts{device="rdmap144s27",port="1"} 1.30916726e+08
node_amazonefa_tx_pkts{device="rdmap160s27",port="1"} 1.30262069e+08
node_amazonefa_tx_pkts{device="rdmap16s27",port="1"} 1.30907467e+08
node_amazonefa_tx_pkts{device="rdmap32s27",port="1"} 1.30933425e+08
node_scrape_collector_duration_seconds{collector="amazonefa"} 0.016049024
node_scrape_collector_success{collector="amazonefa"} 1
100  206k    0  206k    0     0  1266k      0
```

Note, these metrics are counters and when you run an application look for these counters to increase. If for some reason, they are constant, that indicates messages are not sent over EFA.

# 8. Uninstall Exporter

To uninstall the exporter, you can do the following which would also stop the relevant pods and service it created

```bash
helm uninstall efa-node-exporter
```
Finally, to free up the ${LOCAL_PORT}, you can find the process from below and kill the process to free the port:

```bash
ps -Aef | grep 'port'
```
