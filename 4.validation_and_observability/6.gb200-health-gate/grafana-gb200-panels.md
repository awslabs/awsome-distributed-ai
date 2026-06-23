# Grafana panels for P6e-GB200

Add these to the dashboard fed by `4.validation_and_observability/4.prometheus-grafana` (dcgm-exporter scraping `gb200-dcgm-metrics.csv`).

## Panel 1 — C2C link status (early warning)

The single most useful GB200 panel. A Grace↔Blackwell C2C link dropping to a degraded state precedes most coherent-memory and offload performance cliffs.

- Query: `c2c_link_status` (per-GPU, per-link)
- Viz: state-timeline / stat; alert when any link != "up".

## Panel 2 — NVLink per-link health

- Queries: `rate(nvlink_crc_flit_errors[5m])`, `rate(nvlink_replay_errors[5m])`, `rate(nvlink_recovery_errors[5m])`
- Viz: time series per `gpu` × `link`; a single link with rising errors is the miscabled/degraded-port signature (correlate with Xid 149).

## Panel 3 — NVSwitch fabric

- Queries: `nvswitch_link_tx`, `nvswitch_link_rx`, `increase(nvswitch_fatal_errors[5m])`
- Viz: time series; any fatal-error increase (SXid) is a domain-level event — page on it.

## Panel 4 — NVLink bandwidth utilization

- Query: `nvlink_bandwidth_total` aggregated per domain (group by `nvidia.com/gpu.clique`)
- Use to confirm a job is actually using the NVLink domain (high) vs falling back to EFA (low intra-domain).

## Panel 5 — Standard GPU vitals

`gpu_utilization`, `fb_used`, `gpu_temp`, `power_usage` — grouped by clique so you see the whole 72-GPU domain at once.

## Alerting summary

| Signal | Severity | Why |
|---|---|---|
| `c2c_link_status != up` | warning | Earliest degradation signal |
| `nvswitch_fatal_errors` increasing | critical | SXid; domain-level fault |
| rising NVLink CRC/replay on one link | warning | Miscabled/degrading port (Xid 149 correlate) |
| NCCL correctness gate `#wrong > 0` | critical | Silent data corruption — quarantine the domain |
