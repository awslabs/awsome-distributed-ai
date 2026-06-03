# EFA node Exporter for Prometheus

Scripted fork of the [Prometheus Node Exporter](https://github.com/prometheus/node_exporter) and [ProcFS](https://github.com/prometheus/procfs) repositories to export Amazon EFA metrics counters on compatible instances including c5n, hpc6ad, P5, P4.

# 1. How to run the collector

To create the docker image run:

```bash
git clone https://github.com/awslabs/awsome-distributed-training.git
cd awsome-distributed-training/validation_and_observability/3.efa-node-exporter
make
```

Then you can execute it, i.e.

```bash
# alternatively you can run:
# make run
docker run -d \
  --net="host" \
  --pid="host" \
  -v "/:/host:ro,rslave" \
  node_exporter_efa:latest \
  --path.rootfs=/host
```

Now you can make sure the metrics are being correctly exported by running:

```bash
curl -s http://localhost:9100/metrics | grep amazonefa
```

You should see a bunch of metrics like the following:

```
# HELP node_amazonefa_tx_pkts Number of packets transmitted
# TYPE node_amazonefa_tx_pkts counter
node_amazonefa_tx_pkts{device="rdmap113s0",port="1"} 1.664737e+06
node_amazonefa_tx_pkts{device="rdmap114s0",port="1"} 1.664737e+06
node_amazonefa_tx_pkts{device="rdmap115s0",port="1"} 1.664738e+06
node_amazonefa_tx_pkts{device="rdmap116s0",port="1"} 1.664746e+06
node_amazonefa_tx_pkts{device="rdmap130s0",port="1"} 1.6642e+06
node_amazonefa_tx_pkts{device="rdmap131s0",port="1"} 1.664199e+06
node_amazonefa_tx_pkts{device="rdmap132s0",port="1"} 1.6642e+06
node_amazonefa_tx_pkts{device="rdmap133s0",port="1"} 1.664208e+06
node_amazonefa_tx_pkts{device="rdmap147s0",port="1"} 1.663841e+06
node_amazonefa_tx_pkts{device="rdmap148s0",port="1"} 1.66384e+06
node_amazonefa_tx_pkts{device="rdmap149s0",port="1"} 1.663842e+06
node_amazonefa_tx_pkts{device="rdmap150s0",port="1"} 1.66385e+06
node_amazonefa_tx_pkts{device="rdmap164s0",port="1"} 1.65972e+06
node_amazonefa_tx_pkts{device="rdmap165s0",port="1"} 1.659707e+06
node_amazonefa_tx_pkts{device="rdmap166s0",port="1"} 1.65973e+06
node_amazonefa_tx_pkts{device="rdmap167s0",port="1"} 1.659725e+06
node_amazonefa_tx_pkts{device="rdmap181s0",port="1"} 1.658642e+06
node_amazonefa_tx_pkts{device="rdmap182s0",port="1"} 1.658642e+06
node_amazonefa_tx_pkts{device="rdmap183s0",port="1"} 1.658642e+06
node_amazonefa_tx_pkts{device="rdmap184s0",port="1"} 1.658651e+06
node_amazonefa_tx_pkts{device="rdmap198s0",port="1"} 1.655952e+06
node_amazonefa_tx_pkts{device="rdmap199s0",port="1"} 1.655953e+06
node_amazonefa_tx_pkts{device="rdmap200s0",port="1"} 1.655953e+06
node_amazonefa_tx_pkts{device="rdmap201s0",port="1"} 1.655961e+06
node_amazonefa_tx_pkts{device="rdmap79s0",port="1"} 1.667468e+06
node_amazonefa_tx_pkts{device="rdmap80s0",port="1"} 1.667512e+06
node_amazonefa_tx_pkts{device="rdmap81s0",port="1"} 1.667507e+06
node_amazonefa_tx_pkts{device="rdmap82s0",port="1"} 1.667491e+06
node_amazonefa_tx_pkts{device="rdmap96s0",port="1"} 1.664917e+06
node_amazonefa_tx_pkts{device="rdmap97s0",port="1"} 1.664916e+06
node_amazonefa_tx_pkts{device="rdmap98s0",port="1"} 1.664917e+06
node_amazonefa_tx_pkts{device="rdmap99s0",port="1"} 1.664925e+06
node_scrape_collector_duration_seconds{collector="amazonefa"} 0.084817395
node_scrape_collector_success{collector="amazonefa"} 1
```

## 2. Authors / Reviewers

* [A] Pierre-Yves Aquilanti - pierreya@
* [A] Sean Smith - seaam@
* [R] Maxime Hugues - maxhaws@
