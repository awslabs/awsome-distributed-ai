// Copyright 2022 Amazon Web Services
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package collector

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/procfs/sysfs"
)

type AmazonEfaCollector struct {
	fs          sysfs.FS
	metricDescs map[string]*prometheus.Desc
	logger      *slog.Logger
	subsystem   string
}

func init() {
    registerCollector("amazonefa", defaultEnabled, NewAmazonEfaCollector)
}

// NewAmazonEfaCollector returns a new Collector exposing Amazon EFA stats.
func NewAmazonEfaCollector(logger *slog.Logger) (Collector, error) {
	var i AmazonEfaCollector
	var err error

	i.fs, err = sysfs.NewFS(*sysPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open sysfs: %w", err)
	}
	i.logger = logger

	// Detailed description for all metrics.
	descriptions := map[string]string{
		"alloc_pd_err":          		"Number of allocations PD errors",
		"alloc_ucontext_err":   		"Number of allocations UContext errors",
		"cmds_err":              		"Number of commands errors",
		"completed_cmds":        		"Number of completed commands",
		"create_ah_err":         		"Number of create AH errors",
		"create_cq_err":         		"Number of create CQ errors",
		"create_qp_err":         		"Number of create qp errors",
		"impaired_remote_conn_events": 	"Number of EFA SRD connections entered an impaired state, resulting in a reduced throughput rate limit.",
		"keep_alive_rcvd":       		"Number of keep-alive packets received",
		"lifespan":              		"Lifespan of the port",
		"mmap_err":              		"Number of mmap errors",
		"no_completion_cmds":    		"Number of commands with no completion",
		"rdma_read_bytes":       		"Number of bytes read with RDMA",
		"rdma_read_resp_bytes":  		"Number of read reponses bytes with RDMA",
		"rdma_read_wr_err":      		"Number of read write errors with RDMA",
		"rdma_read_wrs":         		"Number of read rs with RDMA",
		"rdma_write_bytes":      		"Number of bytes wrote with RDMA",
		"rdma_write_recv_bytes": 		"Number of bytes wrote and received with RDMA",
		"rdma_write_wr_err":     		"Number of bytes wrote wr with error RDMA",
		"rdma_write_wrs":        		"Number of bytes wrote wrs RDMA",
		"recv_bytes":            		"Number of bytes recv bytes",
		"recv_wrs":              		"Number of bytes recv wrs",
		"reg_mr_err":            		"Number of reg_mr errors",
		"retrans_bytes":		 		"Number of efa_srd bytes retransmitted",
		"retrans_pkts":			 		"Number of efa_srd packets retransmitted",
		"retrans_timeout_events":		"Number of times SRD traffic reached timeout and required network path change",
		"rx_bytes":              		"Number of bytes received",
		"rx_drops":              		"Number of packets droped",
		"rx_pkts":               		"Number of packets received",
		"send_bytes":            		"Number of bytes send",
		"send_wrs":              		"Number of wrs send",
		"submitted_cmds":        		"Number of submitted commands",
		"tx_bytes":              		"Number of bytes transmitted",
		"tx_pkts":               		"Number of packets transmitted",
		"unresponsive_remote_events":	"Number of times SRD connection remote was unresponsive",
	}

	i.metricDescs = make(map[string]*prometheus.Desc)
	i.subsystem = "amazonefa"

	for metricName, description := range descriptions {
		i.metricDescs[metricName] = prometheus.NewDesc(
			prometheus.BuildFQName(namespace, i.subsystem, metricName),
			description,
			[]string{"device", "port"},
			nil,
		)
	}

	return &i, nil
}

func (c *AmazonEfaCollector) pushMetric(ch chan<- prometheus.Metric, name string, value uint64, deviceName string, port string, valueType prometheus.ValueType) {
	ch <- prometheus.MustNewConstMetric(c.metricDescs[name], valueType, float64(value), deviceName, port)
}

func (c *AmazonEfaCollector) pushCounter(ch chan<- prometheus.Metric, name string, value *uint64, deviceName string, port string) {
	if value != nil {
		c.pushMetric(ch, name, *value, deviceName, port, prometheus.CounterValue)
	}
}

func (c *AmazonEfaCollector) Update(ch chan<- prometheus.Metric) error {
	devices, err := c.fs.AmazonEfaClass()
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			c.logger.Debug("msg", "Amazon EFA statistics not found, skipping")
			return ErrNoData
		}
		return fmt.Errorf("error obtaining AmazonEfa class info: %w", err)
	}

	for _, device := range devices {
		infoDesc := prometheus.NewDesc(
			prometheus.BuildFQName(namespace, c.subsystem, "info"),
			"Non-numeric data from /sys/class/infiniband/<device>, value is always 1.",
			[]string{"device"},
			nil,
		)
		infoValue := 1.0
		ch <- prometheus.MustNewConstMetric(infoDesc, prometheus.GaugeValue, infoValue, device.Name)
		
		for _, port := range device.Ports {
			portStr := strconv.FormatUint(uint64(port.Port), 10)

			//c.pushMetric(ch, "state_id", uint64(port.StateID), port.Name, portStr, prometheus.UntypedValue)
			//c.pushMetric(ch, "physical_state_id", uint64(port.PhysStateID), port.Name, portStr, prometheus.UntypedValue)
			//c.pushMetric(ch, "rate_bytes_per_second", port.Rate, port.Name, portStr, prometheus.UntypedValue)
			
			c.pushCounter(ch, "alloc_pd_err", port.Counters.AllocPdErr, port.Name, portStr)
			c.pushCounter(ch, "alloc_ucontext_err", port.Counters.AllocUcontextErr, port.Name, portStr)
			c.pushCounter(ch, "cmds_err", port.Counters.CmdsErr, port.Name, portStr)
			c.pushCounter(ch, "completed_cmds", port.Counters.CompletedCmds, port.Name, portStr)
			c.pushCounter(ch, "create_ah_err", port.Counters.CreateAhErr, port.Name, portStr)
			c.pushCounter(ch, "create_cq_err", port.Counters.CreateCqErr, port.Name, portStr)
			c.pushCounter(ch, "create_qp_err", port.Counters.CreateQpErr, port.Name, portStr)
			c.pushCounter(ch, "impaired_remote_conn_events", port.Counters.ImpairedRemoteConnEvents, port.Name, portStr)
			c.pushCounter(ch, "keep_alive_rcvd", port.Counters.KeepAliveRcvd, port.Name, portStr)
			c.pushCounter(ch, "lifespan", port.Counters.Lifespan, port.Name, portStr)
			c.pushCounter(ch, "mmap_err", port.Counters.MmapErr, port.Name, portStr)
			c.pushCounter(ch, "no_completion_cmds", port.Counters.NoCompletionCmds, port.Name, portStr)
			c.pushCounter(ch, "rdma_read_bytes", port.Counters.RdmaReadBytes, port.Name, portStr)
			c.pushCounter(ch, "rdma_read_resp_bytes", port.Counters.RdmaReadRespBytes, port.Name, portStr)
			c.pushCounter(ch, "rdma_read_wr_err", port.Counters.RdmaReadWrErr, port.Name, portStr)
			c.pushCounter(ch, "rdma_read_wrs", port.Counters.RdmaReadWrs, port.Name, portStr)
			c.pushCounter(ch, "rdma_write_bytes", port.Counters.RdmaWriteBytes, port.Name, portStr)
			c.pushCounter(ch, "rdma_write_recv_bytes", port.Counters.RdmaWriteRecvBytes, port.Name, portStr)
			c.pushCounter(ch, "rdma_write_wr_err", port.Counters.RdmaWriteWrErr, port.Name, portStr)
			c.pushCounter(ch, "rdma_write_wrs", port.Counters.RdmaWriteWrs, port.Name, portStr)
			c.pushCounter(ch, "recv_bytes", port.Counters.RecvBytes, port.Name, portStr)
			c.pushCounter(ch, "recv_wrs", port.Counters.RecvWrs, port.Name, portStr)
			c.pushCounter(ch, "retrans_bytes", port.Counters.RetransBytes, port.Name, portStr)
			c.pushCounter(ch, "retrans_pkts", port.Counters.RetransPkts, port.Name, portStr)
			c.pushCounter(ch, "retrans_timeout_events", port.Counters.RetransTimeoutEvents, port.Name, portStr)
			c.pushCounter(ch, "reg_mr_err", port.Counters.RegMrErr, port.Name, portStr)
			c.pushCounter(ch, "rx_bytes", port.Counters.RxBytes, port.Name, portStr)
			c.pushCounter(ch, "rx_drops", port.Counters.RxDrops, port.Name, portStr)
			c.pushCounter(ch, "rx_pkts", port.Counters.RxPkts, port.Name, portStr)
			c.pushCounter(ch, "send_bytes", port.Counters.SendBytes, port.Name, portStr)
			c.pushCounter(ch, "send_wrs", port.Counters.SendWrs, port.Name, portStr)
			c.pushCounter(ch, "submitted_cmds", port.Counters.SubmittedCmds, port.Name, portStr)
			c.pushCounter(ch, "tx_bytes", port.Counters.TxBytes, port.Name, portStr)
			c.pushCounter(ch, "tx_pkts", port.Counters.TxPkts, port.Name, portStr)
			c.pushCounter(ch, "unresponsive_remote_events", port.Counters.UnresponsiveRemoteEvents, port.Name, portStr)
		}
	}

	return nil
}
