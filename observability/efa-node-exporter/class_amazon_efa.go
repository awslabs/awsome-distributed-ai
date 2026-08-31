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

package sysfs

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"bytes"
	"syscall"
	"github.com/prometheus/procfs/internal/util"
)

const AmazonEfaPath = "class/infiniband"

// AmazonEfaCounters contains counter values from files in
// /sys/class/infiniband/<Name>/ports/<Port>/hw_counters
// for a single port of one Amazon Elastic Fabric Adapter device.
type AmazonEfaCounters struct {
    AllocPdErr               *uint64 // hw_counters/alloc_pd_err
    AllocUcontextErr         *uint64 // hw_counters/alloc_ucontext_err
    CmdsErr                  *uint64 // hw_counters/cmds_err
    CompletedCmds            *uint64 // hw_counters/completed_cmds
    CreateAhErr              *uint64 // hw_counters/create_ah_err
    CreateCqErr              *uint64 // hw_counters/create_cq_err
    CreateQpErr              *uint64 // hw_counters/create_qp_err
    KeepAliveRcvd            *uint64 // hw_counters/keep_alive_rcvd
    ImpairedRemoteConnEvents *uint64 // hw_counters/impaired_remote_conn_events
    Lifespan                 *uint64 // hw_counters/lifespan
    MmapErr                  *uint64 // hw_counters/mmap_err
    NoCompletionCmds         *uint64 // hw_counters/no_completion_cmds
    RdmaReadBytes            *uint64 // hw_counters/rdma_read_bytes
    RdmaReadRespBytes        *uint64 // hw_counters/rdma_read_resp_bytes
    RdmaReadWrErr            *uint64 // hw_counters/rdma_read_wr_err
    RdmaReadWrs              *uint64 // hw_counters/rdma_read_wrs
    RdmaWriteBytes           *uint64 // hw_counters/rdma_write_bytes
    RdmaWriteRecvBytes       *uint64 // hw_counters/rdma_write_recv_bytes
    RdmaWriteWrErr           *uint64 // hw_counters/rdma_write_wr_err
    RdmaWriteWrs             *uint64 // hw_counters/rdma_write_wrs
    RecvBytes                *uint64 // hw_counters/recv_bytes
    RecvWrs                  *uint64 // hw_counters/recv_wrs
    RegMrErr                 *uint64 // hw_counters/reg_mr_err
    RetransBytes             *uint64 // hw_counters/retrans_bytes
    RetransPkts              *uint64 // hw_counters/retrans_pkts
    RetransTimeoutEvents     *uint64 // hw_counters/retrans_timeout_events
    RxBytes                  *uint64 // hw_counters/rx_bytes
    RxDrops                  *uint64 // hw_counters/rx_drops
    RxPkts                   *uint64 // hw_counters/rx_pkts
    SendBytes                *uint64 // hw_counters/send_bytes
    SendWrs                  *uint64 // hw_counters/send_wrs
    SubmittedCmds            *uint64 // hw_counters/submitted_cmds
    TxBytes                  *uint64 // hw_counters/tx_bytes
    TxPkts                   *uint64 // hw_counters/tx_pkts
    UnresponsiveRemoteEvents *uint64 // hw_counters/unresponsive_remote_events
}

// AmazonEfaPort contains info from files in
// /sys/class/infiniband/<Name>/ports/<Port>
// for a single port of one Amazon Elastic Fabric Adapter device.
type AmazonEfaPort struct {
	Name        string
	Port        uint
	State       string // String representation from /sys/class/infiniband/<Name>/ports/<Port>/state
	StateID     uint   // ID from /sys/class/infiniband/<Name>/ports/<Port>/state
	PhysState   string // String representation from /sys/class/infiniband/<Name>/ports/<Port>/phys_state
	PhysStateID uint   // String representation from /sys/class/infiniband/<Name>/ports/<Port>/phys_state
	Rate        uint64 // in bytes/second from /sys/class/infiniband/<Name>/ports/<Port>/rate
	Counters    AmazonEfaCounters
}

// AmazonEfaDevice contains info from files in /sys/class/infiniband for a
// single Amazon Elastic Fabric Adapter (EFA) device.
type AmazonEfaDevice struct {
	Name  string
	Ports map[uint]AmazonEfaPort
}

// AmazonEfaClass is a collection of every Amazon Elastic Fabric Adapter (EFA) device in
// /sys/class/infiniband.
//
// The map keys are the names of the Amazon Elastic Fabric Adapter (EFA) devices.
type AmazonEfaClass map[string]AmazonEfaDevice

// AmazonEfaClass returns info for all Amazon Elastic Fabric Adapter (EFA) devices read from
// /sys/class/infiniband.
func (fs FS) AmazonEfaClass() (AmazonEfaClass, error) {
	path := fs.sys.Path(AmazonEfaPath)

	dirs, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	ibc := make(AmazonEfaClass, len(dirs))
	for _, d := range dirs {
		device, err := fs.parseAmazonEfaDevice(d.Name())
		if err != nil {
			return nil, err
		}

		ibc[device.Name] = *device
	}

	return ibc, nil
}

// Parse one AmazonEfa device.
func (fs FS) parseAmazonEfaDevice(name string) (*AmazonEfaDevice, error) {
	path := fs.sys.Path(AmazonEfaPath, name)
	device := AmazonEfaDevice{Name: name}

	portsPath := filepath.Join(path, "ports")
	ports, err := ioutil.ReadDir(portsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to list AmazonEfa ports at %q: %w", portsPath, err)
	}

	device.Ports = make(map[uint]AmazonEfaPort, len(ports))
	for _, d := range ports {
		port, err := fs.parseAmazonEfaPort(name, d.Name())
		if err != nil {
			return nil, err
		}

		device.Ports[port.Port] = *port
	}

	return &device, nil
}

// Scans predefined files in /sys/class/infiniband/<device>/ports/<port>
// directory and gets their contents.
func (fs FS) parseAmazonEfaPort(name string, port string) (*AmazonEfaPort, error) {
	portNumber, err := strconv.ParseUint(port, 10, 32)
	if err != nil {
		return nil, fmt.Errorf("failed to convert %s into uint", port)
	}
	ibp := AmazonEfaPort{Name: name, Port: uint(portNumber)}

	portPath := fs.sys.Path(AmazonEfaPath, name, "ports", port)
	content, err := ioutil.ReadFile(filepath.Join(portPath, "state"))
	if err != nil {
		return nil, err
	}
	id, name, err := parseState(string(content))
	if err != nil {
		return nil, fmt.Errorf("could not parse state file in %q: %w", portPath, err)
	}
	ibp.State = name
	ibp.StateID = id

	content, err = ioutil.ReadFile(filepath.Join(portPath, "phys_state"))
	if err != nil {
		return nil, err
	}
	id, name, err = parseState(string(content))
	if err != nil {
		return nil, fmt.Errorf("could not parse phys_state file in %q: %w", portPath, err)
	}
	ibp.PhysState = name
	ibp.PhysStateID = id

	content, err = ioutil.ReadFile(filepath.Join(portPath, "rate"))
	if err != nil {
		return nil, err
	}
	ibp.Rate, err = parseRate(string(content))
	if err != nil {
		return nil, fmt.Errorf("could not parse rate file in %q: %w", portPath, err)
	}

	counters, err := parseAmazonEfaCounters(portPath)
	if err != nil {
		return nil, err
	}
	ibp.Counters = *counters

	return &ibp, nil
}

// SysReadFile is a simplified ioutil.ReadFile that invokes syscall.Read directly.
// https://github.com/prometheus/node_exporter/pull/728/files
//
// Note that this function will not read files larger than 128 bytes.
func SysReadFile(file string) (string, error) {
	f, err := os.Open(file)
	if err != nil {
		return "", err
	}
	defer f.Close()

	// On some machines, hwmon drivers are broken and return EAGAIN.  This causes
	// Go's ioutil.ReadFile implementation to poll forever.
	//
	// Since we either want to read data or bail immediately, do the simplest
	// possible read using syscall directly.
	const sysFileBufferSize = 128
	b := make([]byte, sysFileBufferSize)
	n, err := syscall.Read(int(f.Fd()), b)
	if err != nil {
		return "", err
	}

	return string(bytes.TrimSpace(b[:n])), nil
}

// Parse string to UInt64
func parseUInt64(value string) (*uint64, error) {
	// A base value of zero makes ParseInt infer the correct base using the
    // string's prefix, if any.
    const base = 0
    v, err := strconv.ParseUint(value, base, 64)
    if err != nil {
        return nil, err
    }
    return &v, err
}

func parseAmazonEfaCounters(portPath string) (*AmazonEfaCounters, error) {
	var counters AmazonEfaCounters

	path := filepath.Join(portPath, "hw_counters")
	files, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	for _, f := range files {
		if !f.Mode().IsRegular() {
			continue
		}

		name := filepath.Join(path, f.Name())
		value, err := util.SysReadFile(name)
		if err != nil {
			if os.IsNotExist(err) || os.IsPermission(err) || err.Error() == "operation not supported" || err.Error() == "invalid argument" {
				continue
			}
			return nil, fmt.Errorf("failed to read file %q: %w", name, err)
		}

		//const base = 0
		//vp := util.NewValueParser(value)

		switch f.Name() {
			case "impaired_remote_conn_events":
				counters.ImpairedRemoteConnEvents, err = parseUInt64(value)
			case "lifespan":
				counters.Lifespan, err = parseUInt64(value)
			case "rdma_read_bytes":
				counters.RdmaReadBytes, err = parseUInt64(value)
			case "rdma_read_resp_bytes":
				counters.RdmaReadRespBytes, err = parseUInt64(value)
			case "rdma_read_wr_err":
				counters.RdmaReadWrErr, err = parseUInt64(value)
			case "rdma_read_wrs":
				counters.RdmaReadWrs, err = parseUInt64(value)
			case "rdma_write_bytes":
				counters.RdmaWriteBytes, err = parseUInt64(value)
			case "rdma_write_recv_bytes":
				counters.RdmaWriteRecvBytes, err = parseUInt64(value)
			case "rdma_write_wr_err":
				counters.RdmaWriteWrErr, err = parseUInt64(value)
			case "rdma_write_wrs":
				counters.RdmaWriteWrs, err = parseUInt64(value)
			case "recv_bytes":
				counters.RecvBytes, err = parseUInt64(value)
			case "recv_wrs":
				counters.RecvWrs, err = parseUInt64(value)
			case "retrans_bytes":
				counters.RetransBytes, err = parseUInt64(value)
			case "retrans_pkts":
				counters.RetransPkts, err = parseUInt64(value)
			case "retrans_timeout_events":
				counters.RetransTimeoutEvents, err = parseUInt64(value)
			case "rx_bytes":
				counters.RxBytes, err = parseUInt64(value)
			case "rx_drops":
				counters.RxDrops, err = parseUInt64(value)
			case "rx_pkts":
				counters.RxPkts, err = parseUInt64(value)
			case "send_bytes":
				counters.SendBytes, err = parseUInt64(value)
			case "send_wrs":
				counters.SendWrs, err = parseUInt64(value)
			case "tx_bytes":
				counters.TxBytes, err = parseUInt64(value)
			case "tx_pkts":
				counters.TxPkts, err = parseUInt64(value)
			case "unresponsive_remote_events":
				counters.UnresponsiveRemoteEvents, err = parseUInt64(value)

			if err != nil {
				// Ugly workaround for handling https://github.com/prometheus/node_exporter/issues/966
				// when counters are `N/A (not available)`.
				// This was already patched and submitted, see
				// https://www.spinics.net/lists/linux-rdma/msg68596.html
				// Remove this as soon as the fix lands in the enterprise distros.
				if strings.Contains(value, "N/A (no PMA)") {
					continue
				}
				return nil, err
			}
		}
	}

	return &counters, nil
}
