# Tools for the AIM347 workshop job

These tools run one SmolLM2-1.7B training job with PyTorch FSDP2 on two `g7.48xlarge` nodes (16 GPUs) of an AWS PCS Slurm cluster and change it one setting at a time. Each run is measured the same way, so its result can be compared with the earlier runs. The workshop uses them in this order:

1. `bash tools/check.sh` checks the nodes, GPUs, EFA devices, FSx for Lustre, the training image, the data, the monitoring stack and the Grafana sign-in.
2. `bash tools/hold-and-measure.sh` holds both GPU nodes for the session and measures each GPU's GEMM and memory-read ceilings.
3. `bash tools/run.sh <subcommand>` runs one stage of the job on the held nodes. Every training run goes through it.
4. `python3 tools/profile-table.py` reads one profiled update into a fixed table.
5. `python3 tools/progression.py --markdown` prints the table of all your runs.
6. `bash tools/release.sh` releases the nodes.

All six run in a shell on the login node, from this directory. The training runs on both GPU nodes through `srun` steps of the held allocation. The section [Repeat the job on another PCS cluster](#repeat-the-job-on-another-pcs-cluster) describes the provisioning the tools expect.

## The environment and the allocation

Every shell tool first sources the environment file that provisioning writes, `${AIM347_ENV:-/fsx/aim347/participant/aim347.env}`. It sets the partition, the GPU count per node, the image, data and results paths, the NCCL socket interface, the container name and the Grafana URL. A tool exits with status 2 and prints `AIM347: environment file not found: <path>` when the file is missing. The two Python tools read `LLM_RESULTS_DIR` from the shell's environment, so source the file in your shell first (`. /fsx/aim347/participant/aim347.env`); the workshop's login shell already does.

The tools then source `~/.aim347/allocation.env` if it exists. `hold-and-measure.sh` writes that file with the Slurm job ID, node list and CPU count of the held allocation. Because `srun` reads `SLURM_JOB_ID`, every later tool runs its steps inside the same allocation, in this shell or in a new one after a disconnect. If the job named in the file is no longer running, the tools ignore the file, and `run.sh` exits with status 3 and `AIM347: no session allocation. Run: bash tools/hold-and-measure.sh`.

## job.env

`job.env` in this directory holds the three settings that the workshop changes, one line at a time:

| Key | Values | Meaning |
|---|---|---|
| `LLM_TRANSPORT` | `socket`, `efa` | NCCL sends collectives over TCP sockets, or over EFA through the AWS Libfabric plugin |
| `ATTENTION` | `eager`, `sdpa` | The Transformers attention implementation |
| `CHECKPOINT_MODE` | `none`, `sync`, `process` | No saving, synchronous saves, or saves written by a separate process while training continues |

The committed file starts at `socket`, `eager` and `none`. The tools parse the file and never source it. A line with another key, a value outside the table or a repeated key is an error that names its line number. To start over, run `git checkout job.env`. Set `AIM347_JOB_ENV` to use a file elsewhere.

## check.sh

```bash
bash tools/check.sh
```

Prints one line per check, `PASS` or `FAIL`, with the observed value and, on a failure, the expected one. The node checks run on both GPU nodes: inside the held allocation if there is one (with `--overlap`, so they also answer while a stage runs), otherwise in a two-minute job of the partition. They read the GPU name and memory from `nvidia-smi`, and count the entries of `/sys/class/infiniband` whose driver is `efa`. On the login node the tool checks that the stage directory is on Lustre with 300 GiB free, that the image's recorded SHA-256 matches the environment file and was checked on both nodes during provisioning, that each dataset's fingerprint matches its manifest and its pinned value, and that the model snapshot matches the config. It queries Prometheus through `https://localhost/prometheus` for the Lustre collector and the EFA counters on both GPU nodes, and Grafana through `https://localhost/grafana/api/health`. The `grafana` line prints the Grafana URL, the user `admin` and the password from `~/.aim347/grafana-admin-password`.

Exit status: 0 when every check passes; 1 when any fails (the last line says how many, and the first `FAIL` line is the one to fix first); 2 when the environment file is missing.

## hold-and-measure.sh

```bash
bash tools/hold-and-measure.sh
```

Reuses a running job of yours named `aim347-session`, or requests one with `salloc --no-shell` for both nodes of the partition, all their CPUs and GPUs, exclusively, with a time limit of 110 minutes. `--no-shell` lets the allocation outlive the shell that requested it. If Slurm cannot grant it within 120 seconds, the tool prints the partition's queue and exits with status 3.

It then runs one step on both nodes in the training image and measures every GPU in parallel. `aim347-gemm 8192 30` times a dense BF16 GEMM with FP32 accumulation, and `aim347-dram 0` times a 1 GiB streaming read (both built by the `Dockerfile` from `lib/gemm.cu` and `lib/dram.cu`). The per-GPU results go to `$LLM_RESULTS_DIR/ceilings/`. `ceilings.json` there holds the median over all GPUs of the best GEMM rate, the median over all GPUs of the median read bandwidth, and their ratio, the ridge point in FLOP per byte. The tool prints the per-GPU values and these three numbers. The step uses the container name from the environment file, so the image stays unpacked for the session's later runs.

Exit status: 0 when the allocation is held and the ceilings are written; 2 when the environment file is missing; 3 when the allocation is not granted; 4 when the measurement fails. After a failed measurement the allocation is still held, and running the tool again reuses it and measures again.

## run.sh

```bash
bash tools/run.sh <subcommand>
```

Each subcommand first checks `job.env`. If a setting differs from what the subcommand needs, it exits with status 2 and prints the `sed` line that changes it, for example `sed -i 's/^LLM_TRANSPORT=socket$/LLM_TRANSPORT=efa/' job.env`.

| Subcommand | Needs (transport / attention / saving) | Stage directory | Config and data | What it runs |
|---|---|---|---|---|
| `baseline` | socket / eager / none | `baseline` | `smol-2048`, `tokens-2048-160` | One timed run |
| `probe` | any / any / none | `probe-<transport>-<attention>` | `smol-1024`, `tokens-1024-160` | One run at half the sequence length |
| `profile` | any / any / none | `profile-<transport>-<attention>` | `smol-2048`, `tokens-2048-160` | One run with the PyTorch profiler; its time is not compared |
| `sdpa-socket` | socket / eager / none | `sdpa-socket` | `smol-2048`, `tokens-2048-160` | One run with SDPA attention for this run only; `job.env` is unchanged |
| `efa` | efa / eager / none | `efa` | `smol-2048`, `tokens-2048-160` | One timed run |
| `sdpa` | efa / sdpa / none | `sdpa` | `smol-2048`, `tokens-2048-160` | One timed run, with its losses beside those of stage `efa` |
| `checkpoint` | efa / sdpa / sync or process | `ckpt-sync`, then `ckpt-process` | `smol-2048-40`, `tokens-2048-1280` | Two runs that differ only in the saving mode, each saving after updates 5, 20 and 40 |
| `restore` | efa / sdpa / any | `restore` | `smol-2048-40`, `tokens-2048-1280` | Restores the update-20 checkpoint of `ckpt-sync`, runs updates 21 to 25 and stops |
| `interrupt` | efa / sdpa / any | `interrupt` | `smol-2048-40`, `tokens-2048-1280` | A process-mode run stopped after update 30, then resumed from its last completed checkpoint |
| `attach` | | | | Follows the stage in progress, or prints the result of the last one |

`smol-2048` and `smol-1024` run five updates of 32 sequences, with the first update as warmup. `smol-2048-40` runs 40 updates with two warmup updates. All three configs copy `configs/llm-zero-smol.json` and change only the sequence length, the update counts, the saves and the description. `restore` needs a finished `ckpt-sync` stage whose update-20 checkpoint completed. `interrupt` runs `13.measure-recovery.py`, which stops the first run without publishing the save in progress, restores the newest checkpoint that has a `COMPLETED.json`, and runs to update 40.

The trainer arguments are the same for every stage apart from the config, data, attention, saving mode and output, and they appear in the stage's `run.json`. The launcher is `2.run-llm.sh`, which runs one `torchrun` per node in the training image, with the container name from the environment file.

### Following a run

`run.sh` starts each stage in a background worker in its own session, then follows it. It prints a header with the settings, the work and the results path, a line every 10 seconds until the first update finishes, and then rank 0's row for each update: the seconds since the previous update and the loss, plus the save, pending-upload and durable-update columns for runs that save. When the stage ends it prints a result block: the timed interval and its target tokens, tokens per second, GPU-seconds per million tokens, MFU against your measured GEMM peak (or `n/a` before `hold-and-measure.sh` has run), the largest rank's peak memory, the NCCL network each rank used, and the EFA bytes each node sent and received during the run. Runs that save print the inputs of training goodput (the durable tokens, the interval from the end of model loading to the end of the final flush, and the GPU count) instead of a goodput value. `restore` prints its losses beside those of `ckpt-sync`, and `interrupt` prints which checkpoints had completed at the stop and how many updates ran twice.

Ctrl-C, a closed browser tab or a dropped connection stops only the following. `bash tools/run.sh attach` follows the stage again from any shell. While a stage runs, any other subcommand prints `Stage <name> is still running; following it` and follows it.

MFU here is tokens per second times 6N + 12·L·d·T FLOP per token, divided by 16 times your measured GEMM peak. N is 1,711,376,384 parameters, L is 24 layers, d is 2,048 and T is the sequence length. GPU-seconds per million tokens is the timed interval times 16, divided by the timed target tokens, times 10^6.

### Checks after a run

`run.sh` first decides whether the run succeeded. A launcher exit of 0 is success, with two exceptions. In `restore` every rank exits with status 75 after update 25 by design, so the stage succeeds when all 16 rank logs end at update 25 with `controlled_interruption: true`. In `interrupt` the controller must exit 0 and write `campaign/campaign.json`. After a successful run it checks:

- **NCCL network.** The launch log has a `NCCL INFO Using network` line from each of the 16 ranks, all `Socket` for socket stages or all `AWS Libfabric` for EFA stages. For `interrupt`, each segment's log is checked on its own.
- **EFA counters.** On an EFA stage, every node whose counters could be read shows an increase. A node whose counter files are missing or unreadable is reported as `EFA counters unavailable on <node>` and does not fail the stage.
- **Summary.** Every stage except `profile`, `restore` and `interrupt` wrote a completed, unprofiled `summary.json`.
- **Restore start.** Rank 0 of `restore` starts at update 21.

The EFA counters are `tx_bytes`, `rx_bytes`, `rdma_read_bytes`, `rdma_write_bytes` and `rdma_write_recv_bytes` under each EFA device's `ports/1/hw_counters`, read on both nodes by one `srun` step before the run and one after it. The bytes sent and received in the result block are the changes in `tx_bytes` and `rx_bytes`.

Exit status: 0 on success, and also after Ctrl-C (the stage keeps running); 2 for usage, an invalid `job.env`, a setting the subcommand does not accept, or a missing earlier stage; 3 when no session allocation is held; 4 when the run failed (the tool prints the last 40 lines of the launch log, and for `interrupt` also of the segment log written last); 5 when the run succeeded but a check failed (the tool names the check).

### Stage directory

```
$LLM_RESULTS_DIR/<stage>/
  run.json            stage, subcommand, job.env values, settings used, launch command, config, data,
                      allocation ID, start and end times, launcher exit, exit status and checks
  launch.log          output of the launch, including the NCCL INFO lines (interrupt: the controller's output)
  efa-before.json     {node: {device: {counter: value}}} before the run; a node that could not be read maps to {}
  efa-after.json      the same after the run
  train/              trainer output: summary.json, rank-*-updates.jsonl, trace-rank-*.json (profile),
                      checkpoints/, checkpoint-events-rank-*.jsonl (not for interrupt)
  campaign/           interrupt only: interrupted/ and resumed/ (trainer output of each segment),
                      interrupted.log and resumed.log, events.jsonl, campaign.json
$LLM_RESULTS_DIR/attempts/<stage>-<UTC>/   an earlier attempt of the stage, moved here before a rerun
$LLM_RESULTS_DIR/ceilings/                 the GEMM and memory-read results and ceilings.json
~/.aim347/allocation.env                   the held allocation
~/.aim347/running                          the stage in progress and its worker's process ID
~/.aim347/worker.log                       the background worker's own output
```

## profile-table.py

```bash
python3 tools/profile-table.py [STAGE] [--rank N|slowest|all] [--update U]
```

Reads one rank's trace of a `profile-*` stage, by default the most recent one. `--rank slowest` (the default) picks the rank with the longest interval for the chosen update in the rank logs, and only that rank's trace is parsed. `--rank all` prints the mean of each row over all ranks. `--update` defaults to the last update.

The window of update u is the span `ProfilerStep#<u-1>` on the training thread. The profiler records one such span per update, and each update ends with a host synchronization, so its GPU work finishes inside the span. The table gives the update's length and, as seconds and as a share of the update:

- compute kernels busy and NCCL kernels busy (kernels whose name starts with `ncclDevKernel` or `ncclKernel`), each as the union of kernel intervals over all streams
- communication not overlapped: the time NCCL kernels run while no compute kernel runs
- GPU idle: the time no kernel and no copy runs
- input wait: the `input/wait` ranges of the training thread
- attention core kernels: kernels launched inside the trainer's `attention/core` ranges, plus the backward kernels of the autograd functions whose sequence numbers belong to operators inside those ranges
- the five kernels with the most time in the window.

With `--profile`, the trainer marks every attention call with `record_function('attention/core')`. The tool uses only the standard library.

Exit status: 0; 2 when the stage or a trace is missing; 3 when the trace has no `ProfilerStep` span for the update.

## progression.py

```bash
python3 tools/progression.py [--markdown]
```

Prints the ceilings, then one row per stage, in the order `baseline`, `probe-socket-eager`, `sdpa-socket`, `efa`, `probe-efa-eager`, `sdpa`, `ckpt-sync`, `ckpt-process`, `interrupt`, followed by any other stage. It skips `profile-*`, `restore` and `attempts/`. The columns are the settings, the work (updates times positions), the timed seconds, tokens per second, GPU-seconds per million tokens, MFU against your GEMM peak, peak memory, the speed-up against `baseline` and against the previous main-path stage, and the training goodput of the saving and interrupt stages. A speed-up is shown only for a pair of runs that `paired_training_report` in `lib/llm_metrics.py` accepts: the same work on the same allocation, completed and unprofiled. Otherwise the cell refers to a note below the table that names the difference. The main path is `baseline`, `efa`, `sdpa`, and `sdpa-socket` is compared with `baseline`. Because each step's speed-up divides the previous stage's seconds by this stage's seconds, the product of the steps along the main path equals the last stage's speed-up against `baseline`.

`--markdown` prints the same table in Markdown for your notes.

## release.sh

```bash
bash tools/release.sh
```

Cancels the session allocation, waits up to 60 seconds until `squeue` no longer lists it, removes `~/.aim347/allocation.env` and `~/.aim347/running`, and prints `squeue --me`. Exit status: 0; 1 when the job is still listed after 60 seconds.

## Repeat the job on another PCS cluster

The tools expect a cluster built from the [AWS PCS architecture](../../../../architectures/aws-pcs/README.md) with the monitoring stack on the login node (`MonitoringStack=Prometheus-LoginNode`), Pyxis and Enroot on the GPU nodes, Docker on the GPU nodes for the image build, a Slurm partition named `gpu` with two `g7.48xlarge` nodes, and an FSx for Lustre file system mounted at the same path on every node. Preparation downloads the model and the FineWeb-Edu sample from Hugging Face at pinned revisions and needs no token.

1. Clone this repository into a directory on Lustre owned by the user who will run the job, for example `/fsx/aim347/participant/companion`. The workshop keeps everything under `/fsx/aim347/participant`, the default stage directory of the tools.
2. As root on the login node, inside a two-node exclusive allocation of the partition that holds every CPU and GPU of both nodes, run the two provisioning scripts from this directory:

   ```bash
   salloc --partition=gpu --nodes=2 --ntasks-per-node=1 --cpus-per-task=<CPUs per node> --exclusive --gres=gpu:8 --time=04:00:00 \
       bash -c 'bash facilitator/prepare-event.sh --stage-dir /fsx/aim347/participant --slurm-version <Slurm version> \
           --instance-type g7.48xlarge --grafana-url https://<login node address>/grafana/ \
         && srun --nodes=2 --ntasks-per-node=1 bash facilitator/install-node-collectors.sh --verify'
   ```

3. Write the Grafana password to `~/.aim347/grafana-admin-password` of the job's user, with mode 0600. The monitoring installer stores it in the SSM parameter `/pcs/<cluster>/grafana/admin-password`.
4. As that user, run the tools in the order at the top of this page.

### facilitator/prepare-event.sh

```bash
bash facilitator/prepare-event.sh --stage-dir DIR --slurm-version VER --instance-type TYPE --grafana-url URL [--owner USER] [--build-jobs N]
```

It refuses any `--instance-type` other than `g7.48xlarge` before it starts, because the tools request 8 GPUs per node and expect two EFA devices. It then prints `prepare-event: <step> ok (<seconds> s)` after each step and `prepare-event: ready` at the end:

1. **checks**: two nodes in the allocation with equal `CPUEfctv`, `DIR` on Lustre, and the interface the second node uses to reach the first, which becomes `NCCL_SOCKET_IFNAME`.
2. **image**: on the first node, `docker build` of this directory's `Dockerfile` and `enroot import` into `DIR/images/aim347-lab.sqsh`, with its SHA-256 in `aim347-lab.sqsh.sha256`. No registry is used. An existing image whose hash checks is kept.
3. **image on both nodes**: `unsquashfs -s` and the hash check on each node, recorded in `aim347-lab.sqsh.verified`.
4. **data**: in the image on the first node, `1.prepare-llm.py` writes `DIR/data/tokens-2048-1280`, `tokens-2048-160` and `tokens-1024-160` from documents of at least 4,096 tokens, keeping each document's first positions, and downloads the model to `DIR/data/model` with the first dataset. A dataset whose manifest, sizes and fingerprint verify is kept. One that fails is moved to `DIR/failed-<UTC>/` and prepared again.
5. **data on both nodes**: each node recomputes every dataset's fingerprint and checks the model pin.
6. **environment file**: `DIR/aim347.env`, written atomically.
7. **ownership**: `images`, `data`, `results` and `aim347.env` are given to `--owner` (default `ubuntu`).

Steps 2 and 4 request all of the node's CPUs, because a step task otherwise gets one. A rerun after a failure keeps the parts that verify. Exit status: 0 ready; 2 usage or a precondition; 3 the image build, import or hash; 4 the data preparation; 5 a verification on either node.

### facilitator/install-node-collectors.sh

```bash
bash facilitator/install-node-collectors.sh [--verify] [--timeout SECONDS]
```

Runs as root on one GPU node. It writes the drop-in `/etc/systemd/system/efa-metrics.timer.d/aim347-5s.conf`, which changes the monitoring stack's EFA collector from every 30 seconds to every 5 seconds, then reloads systemd and restarts the timer. It starts `lib/counters.py lustre` as the transient unit `aim347-lustre`, which writes the Lustre client counters to `/var/lib/prometheus/node-exporter/lustre.prom` every 5 seconds for node-exporter's textfile collector. A second run changes nothing. With `--verify`, it waits up to `--timeout` seconds (default 300) until the timer runs at 5 seconds, `lustre.prom` holds `aim347_lustre_collection_success 1`, and `http://localhost:9100/metrics` serves that line and `node_amazonefa_tx_bytes`. It does not wait for `lustre_write_bytes_total`, which Lustre reports only after the node has written to the file system. Run it again on a node that replaces one of the pair.

Exit status: 0; 1 when `--verify` times out (the message names the condition it waited for); 2 when not run as root or for usage.

## Tests

The CPU tests of the tools need only the standard library:

```bash
PYTHONPATH=lib python3 -m unittest tests.test_workshop_tools -v
```

The whole suite, `PYTHONPATH=lib python3 -m unittest discover -s tests -v`, also runs the trainer tests when `torch==2.9.1` and `transformers==4.57.6` are installed, and skips them otherwise.
