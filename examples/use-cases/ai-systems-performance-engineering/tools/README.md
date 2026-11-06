# Tools for the AIM347 workshop job

These tools run one SmolLM2-1.7B training job with PyTorch FSDP2 on two `g7.48xlarge` nodes (16 GPUs) of an AWS PCS Slurm cluster and change it one setting at a time. Each run is measured the same way, so its result can be compared with the earlier runs. The workshop uses them in this order:

1. `bash tools/check.sh` checks the nodes, GPUs, EFA devices, FSx for Lustre, the training image, the data, the monitoring stack and the Grafana sign-in.
2. `bash tools/hold-and-measure.sh` holds both GPU nodes for the session and measures each GPU's GEMM and memory-read ceilings.
3. `bash tools/run.sh <subcommand>` runs one stage of the job on the held nodes. Every training run goes through it.
4. `python3 tools/profile-table.py` reads one profiled update into a fixed table and a chart.
5. `python3 tools/progression.py --markdown` prints the table of all your runs and two charts.
6. `bash tools/release.sh` releases the nodes.

`bash tools/nccl-protocols.sh` is optional. It runs nccl-tests on the held nodes to compare NCCL's own protocol choice over EFA with `NCCL_PROTO=Simple`.

`bash tools/serve.sh` is optional too. It serves the model with vLLM on every GPU of the held nodes and measures its decode rate per GPU against the memory-read ceiling. It needs the vLLM image that `facilitator/prepare-event.sh --with-serving` imports.

All the tools run in a shell on the login node, from this directory. The training runs on both GPU nodes through `srun` steps of the held allocation. The section [Repeat the job on another PCS cluster](#repeat-the-job-on-another-pcs-cluster) describes the provisioning the tools expect.

## The environment and the allocation

Every shell tool first sources the environment file that provisioning writes, `${AIM347_ENV:-/fsx/aim347/participant/aim347.env}`. It sets the partition, the GPU count per node, the image, data and results paths, the NCCL socket interface, the container name and the Grafana URL. A tool exits with status 2 and prints `AIM347: environment file not found: <path>` when the file is missing. The two Python tools read `LLM_RESULTS_DIR` from the shell's environment, so source the file in your shell first (`. /fsx/aim347/participant/aim347.env`); the workshop's login shell already does.

The tools then source `~/.aim347/allocation.env` if it exists. `hold-and-measure.sh` writes that file with the Slurm job ID, node list and CPU count of the held allocation. Because `srun` reads `SLURM_JOB_ID`, every later tool runs its steps inside the same allocation, in this shell or in a new one after a disconnect. If the job named in the file is no longer running, the tools ignore the file, and `run.sh` exits with status 3 and `AIM347: no session allocation. Run: bash tools/hold-and-measure.sh`.

## job.env

`job.env` in this directory holds the three settings that the workshop changes, one line at a time:

| Key | Values | Meaning |
|---|---|---|
| `LLM_TRANSPORT` | `socket`, `efa` | NCCL sends collectives over TCP sockets, or over EFA through the AWS Libfabric plugin with `OFI_NCCL_PROTOCOL=RDMA` (`SENDRECV` on `g6e.48xlarge`, where RDMA hangs) and `NCCL_PROTO=Simple` (an `NCCL_PROTO` set in your shell takes its place, see [nccl-protocols.sh](#nccl-protocolssh)) |
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

Reuses a running job of yours named `aim347-session`, or requests one with `salloc --no-shell` for both nodes of the partition, all their CPUs and GPUs, exclusively, with `--time=UNLIMITED`. `--no-shell` lets the allocation outlive the shell that requested it, and it lasts until `release.sh` ends it. The explicit `--time=UNLIMITED` keeps a partition's `DefaultTime` from setting a limit. If Slurm cannot grant it within 120 seconds, the tool prints the partition's queue and exits with status 3.

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
| `try <change>` | efa / sdpa / any | `try-<change>`, then `rerun-<change>` | `smol-2048`, `tokens-2048-160` | Two runs without saving: the configuration of `sdpa` with one change, then the same configuration without it |
| `attach` | | | | Follows the stage in progress, or prints the result of the last one |

`smol-2048` and `smol-1024` run five updates of 32 sequences, with the first update as warmup. `smol-2048-40` runs 40 updates with two warmup updates. All three configs copy `configs/llm-zero-smol.json` and change only the sequence length, the update counts, the saves and the description. `restore` needs a finished `ckpt-sync` stage whose update-20 checkpoint completed. `interrupt` runs `13.measure-recovery.py`, which stops the first run without publishing the save in progress, restores the newest checkpoint that has a `COMPLETED.json`, and runs to update 40.

`try` needs a finished `sdpa` stage. Its `<change>` is one of four trainer changes to the configuration of `sdpa`: `microbatch2` runs `--microbatch 2` in place of `--microbatch 1`, so each GPU trains on its two sequences of an update at once; `fused-optimizer` adds `--fused-optimizer`, AdamW with `fused=True`; `forward-prefetch` adds `--forward-prefetch`, which has FSDP2 issue the next decoder block's all-gather earlier in the forward pass; `no-reshard-after-forward` adds `--no-reshard-after-forward`, which keeps each block's gathered parameters from the forward pass for the backward pass. The second run, `rerun-<change>`, repeats `sdpa` unchanged, so that `sdpa` and the rerun give the spread between two runs of the same configuration. After both runs, `try` prints the spread, `|sdpa - rerun-<change>|`, and the change, `try-<change>` minus the mean of `sdpa` and the rerun, each in seconds of the timed updates and in percent of that mean, and a chart of the two. It gives no verdict. It compares nothing when the three runs did not do the same work on the same allocation, when `sdpa` and the rerun differ in their `summary.json` configuration, when the run with the change differs from the rerun in more than the change, or when its `summary.json` does not record the change, and it prints the difference instead. Running `try` again with the same change replaces both stages.

The trainer arguments are the same for every stage apart from the config, data, attention, saving mode, output and the change of a `try` stage, and they appear in the stage's `run.json`. The launcher is `2.run-llm.sh`, which runs one `torchrun` per node in the training image, with the container name from the environment file.

### Following a run

`run.sh` starts each stage in a background worker in its own session, then follows it. It prints a header with the settings, the work and the results path, a line every 10 seconds until the first update finishes, and then rank 0's row for each update: the seconds since the previous update and the loss, plus the save, pending-upload and durable-update columns for runs that save. When the stage ends it prints a result block: the timed interval and its target tokens, tokens per second, GPU-seconds per million tokens, MFU against the GPU's dense BF16 peak, the share of your measured GEMM peak (or `n/a` before `hold-and-measure.sh` has run), the largest rank's peak memory, the NCCL network each rank used, and the EFA bytes each node sent and received during the run. An EFA stage also prints the libfabric provider, the plugin's transport protocol, the `NCCL_PROTO` value that NCCL logged on each rank (`Simple on 16 of 16 ranks`), and how many network channel lines send from GPU memory (`/GDRDMA`). Runs that save also print their training goodput in the block, with what it is computed from: the retained tokens (the tokens of the updates in the last durable checkpoint), the interval, which runs from the end of setup (model loading, FSDP wrapping, optimizer, checkpoint manager) to the end of the final flush and the closing synchronization across ranks, the retained progress rate (the retained tokens divided by the interval), the fault-free rate (the tokens per second of the timed updates of stage `sdpa`, which does the same work per update without saves) and the training goodput, the first rate divided by the second, with two decimals. Training goodput has no unit. The interval also holds the data loader's start and the warmup updates, which the fault-free rate leaves out, so even a run without saves stays below 1. The block prints `not computed` and the reason in place of the fault-free rate and the goodput when the run itself is not completed, was profiled, ran numerical verification or is marked not performance-eligible; when stage `sdpa` has no completed run, had a failed check or is not a clean measurement in the same ways; when the two differ in their recorded configuration other than the checkpoint mode, in model revision, precision, allocation label or GPU; or when the work per update differs. A value above 1 is printed as measured, with `above 1: check the reference run`. `restore` prints its losses beside those of `ckpt-sync`, and `interrupt` prints which checkpoints had completed at the stop, how many updates ran twice, and the same goodput lines for the whole command, from just before the first launch to the exit of the resumed launch.

Ctrl-C, a closed browser tab or a dropped connection stops only the following. `bash tools/run.sh attach` follows the stage again from any shell. While a stage runs, any other subcommand prints `Stage <name> is still running; following it` and follows it. While `serve.sh` holds the GPUs, `run.sh` starts no stage and exits with status 2.

MFU (model FLOPs utilization, PaLM section 4.1 and appendix B) is tokens per second times 6N + 12·L·d·T FLOP per token, divided by 16 times the dense BF16 Tensor Core peak of one GPU. N is 1,711,376,384 parameters, the tied embedding counted once as PaLM and torchtitan count it, L is 24 layers, d is 2,048 and T is the sequence length. The attention term 12·L·d·T follows PaLM's full, non-causal count; Megatron-LM halves it for the causal mask, which gives a 5.3 % lower FLOP count at 2,048 positions. The tools take the peak from the GPU name in `summary.json` and the table `DENSE_BF16_PEAKS` in `lib/workshop.py`, which gives the source of each value:

| GPU (instance) | Dense BF16 peak per GPU | Source |
|---|---|---|
| NVIDIA RTX PRO 4500 Blackwell Server Edition (`g7.48xlarge`) | 202.8 TFLOP/s, estimated | NVIDIA lists 406 TFLOPS without saying whether it includes 2:4 sparsity. 1,024 dense FLOP per SM per clock × 82 SMs × 2,415 MHz boost clock (product brief) = 202.8, half of 406. 1,024 is the dense FP16 rate with FP16 accumulation in the RTX Blackwell architecture whitepaper (appendix B, table 4: RTX 5080, 225.1 TFLOPS at 84 SMs and 2,617 MHz); that GeForce card runs BF16 with FP32 accumulation at half of it. The estimate assumes that this professional part runs BF16 with FP32 accumulation at the full rate, as NVIDIA lists for the L40S (362.05) against the L40 on the same chip (181, Ada architecture whitepaper, appendix C). The measured GEMM supports it: 167.6 TFLOP/s per GPU, where a half rate would cap at 101.4 |
| NVIDIA L40S (`g6e.48xlarge`) | 362.05 TFLOP/s | NVIDIA's L40S page: "362.05 I 733*", "*With Sparsity" |
| NVIDIA RTX PRO 6000 Blackwell Server Edition (`g7e.48xlarge`) | 467.8 TFLOP/s, estimated | 1,024 × 188 SMs × 2,430 MHz boost clock (product brief), with the same full-rate assumption; a measured 429.1 TFLOP/s GEMM on `g7e.12xlarge` ([README.md](../README.md)) exceeds the 233.9 that a half rate would allow. NVIDIA's page lists 1 PFLOP without a sparsity mark |

For a GPU that the table does not list, the MFU line reads `n/a (no dense peak for <name>)`. Dense BF16 training cannot use a 2:4 sparse rate. The next line, `share of your GEMM peak`, divides the same FLOP rate by 16 times your GEMM peak, the GEMM rate you measured with `hold-and-measure.sh`. It is not MFU: it says how close the run came to what a large GEMM reaches on your GPUs, and the update chart's arithmetic time uses the same measured value. GPU-seconds per million tokens is the timed interval times 16, divided by the timed target tokens, times 10^6.

### Charts under the result block

The Session Manager terminal cannot open image files, so the tools draw their charts with Unicode block characters, at most 100 columns wide, under the result they belong to. Each chart is built from your own result files. A stage that has not run, is still running, failed, or cannot be paired prints the reason in place of its bar; no value is estimated. Each chart ends with the value of one cell. The update chart and the profile chart keep 0.1 s per cell as long as the bars fit, so a bar keeps its length from one module to the next.

- **One update.** Every timed stage without saves, apart from the two of `try`, prints the mean seconds of one timed update as a bar of two parts: the model's arithmetic at your GEMM peak (the targets of one update times 6N + 12·L·d·T FLOP, divided by 16 times the GEMM peak) and the rest. The bar of the stage is drawn below the bar of the stage it is read beside: a `probe` beside the full-length stage with the same transport and attention (`baseline` for `probe-socket-eager`, `efa` for `probe-efa-eager`), `sdpa-socket` and `efa` beside `baseline`, and `sdpa` beside `efa`. Without `ceilings.json` the bar shows the update time alone.
- **Peak memory.** `sdpa` also prints the peak memory of the largest rank of `efa` and `sdpa`.
- **Speed-up over baseline.** `baseline`, `efa` and `sdpa` then print one bar for each of the three stages. The `baseline` bar is 1.00×, and each later bar is the bar before it plus that stage's measured change, so the chart grows as the session proceeds; 1.00× keeps its ten cells as long as the bars fit in 100 columns. A bar uses the same pairs as the `vs baseline` column of `progression.py`: the same work on the same allocation, completed and unprofiled. A change that made the job slower marks the part it lost with `·`.
- **Training goodput.** `ckpt-sync` and `ckpt-process` print the training goodput of both saving stages on an axis from 0 to 1; the key line gives the scale. Only these two have bars, because their intervals start and end at the same events. The goodput of `interrupt`, whose interval also covers both launches and both model loads, is in its result block and in the `Goodput` column of `progression.py`. Each bar is the stage's retained progress rate divided by the fault-free rate of stage `sdpa`, as in the result block, and two lines under the chart state this definition and that even a run without saves stays below 1. A stage whose goodput cannot be computed prints the reason in place of its bar, and a value above 1 prints as measured, with `above 1: check the reference run`, and no bar.
- **A change against the spread.** `rerun-<change>` prints the spread between `sdpa` and the rerun and the size of the change as two bars on one scale, with the direction of the change (`shorter` or `longer`) beside its bar.

### Checks after a run

`run.sh` first decides whether the run succeeded. A launcher exit of 0 is success, with two exceptions. In `restore` every rank exits with status 75 after update 25 by design, so the stage succeeds when all 16 rank logs end at update 25 with `controlled_interruption: true`. In `interrupt` the controller must exit 0 and write `campaign/campaign.json`. After a successful run it checks:

- **NCCL network.** The launch log has a `NCCL INFO Using network` line from each of the 16 ranks, all `Socket` for socket stages or all `AWS Libfabric` for EFA stages. For `interrupt`, each segment's log is checked on its own.
- **EFA counters.** On an EFA stage, every node whose counters could be read shows an increase. A node whose counter files are missing or unreadable is reported as `EFA counters unavailable on <node>` and does not fail the stage.
- **Summary.** Every stage except `profile`, `restore` and `interrupt` wrote a completed, unprofiled `summary.json`.
- **Restore start.** Rank 0 of `restore` starts at update 21.
- **Change.** In `summary.json` of `try-<change>`, the configuration shows the change (for example `microbatch` 2), and in `rerun-<change>` it shows the value without it (`microbatch` 1). The result blocks of both print it on the `setting in summary.json` line.

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
  profile-tables.json profile stages: each table that profile-table.py has read, for its chart
  campaign/           interrupt only: interrupted/ and resumed/ (trainer output of each segment),
                      interrupted.log and resumed.log, events.jsonl, campaign.json
$LLM_RESULTS_DIR/attempts/<stage>-<UTC>/   an earlier attempt of the stage, moved here before a rerun
$LLM_RESULTS_DIR/ceilings/                 the GEMM and memory-read results and ceilings.json
~/.aim347/allocation.env                   the held allocation
~/.aim347/running                          the stage in progress and its worker's process ID
~/.aim347/serving                          serve.sh only: its process ID while its servers run
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

With `--profile`, the trainer marks every attention call with `record_function('attention/core')`.

Below the table, a chart shows where the time of the update goes, as a bar for this profile stage and for each other profile stage whose table you have read with the same `--rank` and `--update`, in the order `profile-socket-eager`, `profile-efa-eager`, `profile-efa-sdpa`. The tool keeps each table it reads in the stage's `profile-tables.json`, so the chart does not parse the other traces again. The parts of a bar do not overlap and add up to the update: attention core kernels, the other compute kernels, communication not overlapped, GPU idle, and, when it reaches 0.0005 s, the time in which only copies run. Input wait is time of the CPU thread and can overlap the GPU parts, so it is printed beside the bar and not in it. A profile stage whose table has not been read prints the command that adds it. The tool uses only the standard library.

Exit status: 0; 2 when the stage or a trace is missing; 3 when the trace has no `ProfilerStep` span for the update.

## progression.py

```bash
python3 tools/progression.py [--markdown]
```

Prints the ceilings, then one row per stage, in the order `baseline`, `probe-socket-eager`, `sdpa-socket`, `efa`, `probe-efa-eager`, `sdpa`, `ckpt-sync`, `ckpt-process`, `interrupt`, then `try-<change>` and `rerun-<change>` for each change you ran with `try`, followed by any other stage. It skips `profile-*`, `restore` and `attempts/`. The columns are the settings, the work (updates times positions), the timed seconds, tokens per second, GPU-seconds per million tokens, MFU against the GPU's dense BF16 peak, the share of your measured GEMM peak, peak memory, the speed-up against `baseline` and against the previous main-path stage, and the training goodput of the saving and interrupt stages, with two decimals. The retained progress of a saving stage comes from its `summary.json`, over the interval from the end of setup (model loading, FSDP wrapping, optimizer, checkpoint manager) to the end of the final flush and the closing synchronization across ranks. The retained progress of `interrupt` comes from `campaign.json`, over the whole command from just before the first launch to the exit of the resumed launch, so it also includes both launches and both model loads. Its goodput cell refers to a note that gives both intervals. Each goodput divides the retained progress rate by the fault-free rate of stage `sdpa`. When the result block would print `not computed`, the cell refers to a note that gives the reason; a value above 1 keeps its value and refers to a note to check the reference run. Below the notes, two lines state the denominators of MFU and of the share of your GEMM peak. A speed-up is shown only for a pair of runs that `paired_training_report` in `lib/llm_metrics.py` accepts: the same work on the same allocation, completed and unprofiled. Otherwise the cell refers to a note below the table that names the difference. The main path is `baseline`, `efa`, `sdpa`. `sdpa-socket` is compared with `baseline`, and the stages of `try` with `sdpa`. Because each step's speed-up divides the previous stage's seconds by this stage's seconds, the product of the steps along the main path equals the last stage's speed-up against `baseline`.

Below the table and its notes, the tool prints two charts: the speed-up over baseline of `baseline`, `efa` and `sdpa`, and the training goodput of `ckpt-sync` and `ckpt-process`. They are the charts that `run.sh` prints under the result blocks ([Charts under the result block](#charts-under-the-result-block)).

`--markdown` prints the same table in Markdown for your notes, with the charts in a fenced code block.

## nccl-protocols.sh

```bash
bash tools/nccl-protocols.sh
```

Runs nccl-tests over EFA on the held nodes, once with NCCL's own protocol choice and once with `NCCL_PROTO=Simple`. The workshop does not need it. `2.run-llm.sh` sets `NCCL_PROTO=Simple` for EFA stages because on two `g7.48xlarge` under `OFI_NCCL_PROTOCOL=RDMA`, NCCL chooses its LL protocol for all-reduce from 64 KiB to 8 MiB, and from 1 MiB that range runs many times slower than with Simple ([VALIDATION.md](../VALIDATION.md#nccl-protocol-on-the-efa-path-2026-11-03)).

`2.run-llm.sh` keeps an `NCCL_PROTO` that is already set in your shell, so you can run a training stage the same way. `NCCL_PROTO= bash tools/run.sh efa` (an empty value) leaves `NCCL_PROTO` unset, so that NCCL chooses its protocol as in the first run of this tool, and `NCCL_PROTO=LL bash tools/run.sh efa` requests LL. The `NCCL_PROTO` line of the result block shows what NCCL logged, and `configuration.nccl_protocol` in `summary.json` records the value (`auto` when unset). Such a run replaces the stage like any rerun: the earlier `efa` moves to `attempts/`, and `progression.py` and the charts use the new run.

`2.run-llm.sh` also keeps `NCCL_DEBUG` and `NCCL_DEBUG_SUBSYS` from your shell, and sets `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV` when they are unset or empty (the result block and the NCCL network check read those lines). `NCCL_DEBUG=info bash tools/run.sh efa` passes the value as typed; NCCL compares the level case-insensitively. `NCCL_DEBUG` must be one of NCCL's levels (`VERSION`, `WARN`, `INFO`, `ABORT`, `TRACE`, any case), otherwise the launcher exits with status 2. The NCCL lines go to the stage's `launch.log` in `$LLM_RESULTS_DIR/<stage>/` (interrupt: `campaign/interrupted.log` and `campaign/resumed.log`). The post-run NCCL network check and the result block read the `Using network` lines, which NCCL logs at level `INFO` under `INIT`; the `NCCL_PROTO` line comes from `ENV`. So a level below `INFO` (`VERSION`, `WARN`) or a `NCCL_DEBUG_SUBSYS` without `INIT` fails the NCCL network check, and one without `ENV` only leaves the result block without the `NCCL_PROTO` line.

The tool needs the session allocation and refuses while a stage runs or while `serve.sh` holds the GPUs. Both runs use `all_reduce_perf` from 1 MiB to 64 MiB on all 16 GPUs, one MPI rank per GPU, in the training image, with the EFA settings of `2.run-llm.sh`. The first leaves `NCCL_PROTO` unset, so that NCCL chooses the protocol, and the second sets `NCCL_PROTO=Simple`. With `NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING`, rank 0 logs the algorithm and protocol of every collective. NCCL writes its log to one file per process, `nccl-<host>-<pid>.log`, so that the table on standard output stays readable. The rank script is `lib/nccl-protocols-rank.sh`.

The tool prints one row per size with the out-of-place time and bus bandwidth of each run and the algorithm and protocol that rank 0 logged, then the `NCCL_PROTO` value each run's log shows and the two log directories, and then a chart of the bus bandwidth of both runs at each size. The results are in `$LLM_RESULTS_DIR/nccl-protocols/<UTC>/nccl-choice/` and `simple/`, each with `all_reduce_perf.txt` and the NCCL logs. `progression.py` does not list them.

Exit status: 0; 2 for usage or while a stage runs; 3 when no session allocation is held; 4 when a run fails (the tool prints the last 40 lines of its output); 5 when nccl-tests reports wrong values.

## serve.sh

```bash
bash tools/serve.sh
```

Serves SmolLM2-1.7B at the pinned revision with vLLM 0.20.2 on the held nodes and measures how fast it decodes. Decode is the part of serving that generates a request's output, one token per step. The workshop does not need it. The tool needs the session allocation, refuses while a stage runs or another `serve.sh` serves, and needs the image `images/vllm-v0.20.2.sqsh` in the stage directory (or the path in `VLLM_IMAGE`), checked on both nodes, which `facilitator/prepare-event.sh --with-serving` imports.

It starts `8.serve-llm.sh --tensor-parallel-size 1 --config configs/smol-2048.json` as one `srun` step: one server per GPU, 16 in all, each with the whole model snapshot that provisioning downloaded to `data/model`. Server n of a node listens on port 8100 + n. Every 15 seconds it prints how many servers answer `/health`, and it gives up after 1,200 seconds. When all answer, `lib/serve_llm.py --collect-manifest` records each server's command and GPU, and the tool sends one warm-up request to each server. It then measures four levels, with 1, 4, 16 and 64 requests per GPU running at the same time. At each level every server gets that many client slots, and each slot sends two requests in turn. Each request is a completion of a fixed prompt that generates exactly 256 tokens (`ignore_eos`).

The step time comes from each server's own counters, read before and after a level: the sum of `vllm:inter_token_latency_seconds` divided by its count is the mean time from one generated token of a request to the next. With every running request in each step, that is the time of one decode step. `lib/serving_metrics.py` checks that the counters moved by exactly the level's requests and tokens, so other traffic is refused. The tool prints one row per level, with the median over the servers:

| Column | Meaning |
|---|---|
| `requests per GPU` | The requests that run on each server at the same time |
| `ms per step` | The median over the servers of the mean step time |
| `decode tokens/s per GPU` | Requests per GPU ÷ seconds per step |
| `MBU` | The time that reading the weights at your memory-read ceiling takes, as a share of the step: 3,422,752,768 bytes (1,711,376,384 BF16 parameters) ÷ the ceiling in `ceilings.json` ÷ seconds per step. It counts the weights only. The ceiling is the rate of the 1 GiB streaming-read kernel, and a decode step can read faster, so MBU can exceed 100 %. `n/a` without `ceilings.json` |

Before the rows it prints the time to read the weights at the ceiling and the decode rate of one request per GPU at a step of that time, the ceiling ÷ the weight bytes. Then it stops the servers by sending SIGTERM to the step with `scancel --signal=TERM`, so that `lib/serve_llm.py` stops its vLLM processes, and prints a chart of the decode rate at each level. Ctrl-C, SIGTERM and a closed terminal (SIGHUP) also stop the servers. While they run, the tool keeps `~/.aim347/serving`, and `run.sh` and `nccl-protocols.sh` refuse to start. Everything goes to `$LLM_RESULTS_DIR/serving-<UTC>/`: `launch.log` (the step's output), `node-<n>/replica-<n>.log` (each server's log), `manifest.json`, `endpoints.txt` and `serving.json` (every level with each server's counter differences). The directory has no `run.json`, so `progression.py` does not list it.

Exit status: 0; 2 for usage, while a stage runs or without the image; 3 when no session allocation is held; 4 when the servers do not all answer or a request fails (the tool prints the last 40 lines of `launch.log`); 5 when the counters do not match the requests; 130 when interrupted.

## release.sh

```bash
bash tools/release.sh
```

Cancels the session allocation, waits up to 60 seconds until `squeue` no longer lists it, removes `~/.aim347/allocation.env` and `~/.aim347/running`, and prints `squeue --me`. Exit status: 0; 1 when the job is still listed after 60 seconds.

## Repeat the job on another PCS cluster

The tools expect a cluster built from the [AWS PCS architecture](../../../../architectures/aws-pcs/README.md) with the monitoring stack on the login node (`MonitoringStack=Prometheus-LoginNode`), Pyxis and Enroot on the GPU nodes, Docker on the GPU nodes for the image build, a Slurm partition named `gpu` with two `g7.48xlarge` nodes, and an FSx for Lustre file system mounted at the same path on every node. Preparation downloads the model and the FineWeb-Edu sample from Hugging Face at pinned revisions and needs no token.

1. Clone this repository, as `ubuntu`, into a directory on Lustre owned by `ubuntu`, for example `/fsx/aim347/participant/companion`. The workshop keeps everything under `/fsx/aim347/participant`, the default stage directory of the tools.
2. As root on the login node, inside a two-node exclusive allocation of the partition that holds every CPU and GPU of both nodes, run the two provisioning scripts from this directory:

   ```bash
   salloc --partition=gpu --nodes=2 --ntasks-per-node=1 --cpus-per-task=<CPUs per node> --exclusive --gres=gpu:8 --time=04:00:00 \
       bash -c 'bash facilitator/prepare-event.sh --stage-dir /fsx/aim347/participant --slurm-version <Slurm version> \
           --instance-type g7.48xlarge --grafana-url https://<login node address>/grafana/ \
         && srun --nodes=2 --ntasks-per-node=1 bash facilitator/install-node-collectors.sh --verify'
   ```

3. Write the Grafana password to `/home/ubuntu/.aim347/grafana-admin-password`. The monitoring installer stores it in the SSM parameter `/pcs/<cluster>/grafana/admin-password`.
4. As `ubuntu`, run the tools in the order at the top of this page.

### facilitator/prepare-event.sh

```bash
bash facilitator/prepare-event.sh --stage-dir DIR --slurm-version VER --instance-type TYPE --grafana-url URL [--build-jobs N] [--with-serving]
```

It accepts `--instance-type g7.48xlarge`, and `g6e.48xlarge` or `g7e.48xlarge` for test events, and refuses any other type before it starts, because the tools request 8 GPUs per node and expect two EFA devices. A `g6e.48xlarge` or `g7e.48xlarge` node has four EFA interfaces, and the test events attach two of them. It then prints `prepare-event: <step> ok (<seconds> s)` after each step and `prepare-event: ready` at the end:

1. **checks**: two nodes in the allocation with equal `CPUEfctv`, `DIR` on Lustre, and the interface the second node uses to reach the first, which becomes `NCCL_SOCKET_IFNAME`.
2. **monitoring install**: on each node, waits up to 30 minutes until node-exporter answers on `http://localhost:9100/metrics` and `docker info` succeeds. PCS installs its monitoring stack on a GPU node as a `NodeReady` action, which runs after the node already accepts jobs. On a GPU node that install restarts Docker before it starts node-exporter, and the restart would end an image build that had already started. On a timeout the script names the node and the install log, `/var/log/amazon/pcs/lifecycle/actions/nodeReady/install-monitoring.log`, and exits 2.
3. **image**: on the first node, `docker build` of this directory's `Dockerfile` and `enroot import` into `DIR/images/aim347-lab.sqsh`, with its SHA-256 in `aim347-lab.sqsh.sha256`. No registry is used. An existing image whose hash checks is kept.
4. **image on both nodes**: `unsquashfs -s` and the hash check on each node, recorded in `aim347-lab.sqsh.verified`.
   With `--with-serving`, two more steps follow for [serve.sh](#servesh). **serving image**: on the first node, `docker pull` of `vllm/vllm-openai` at the digest in the script (vLLM 0.20.2, linux/amd64) and `enroot import` into `DIR/images/vllm-v0.20.2.sqsh`, with its SHA-256 in `vllm-v0.20.2.sqsh.sha256`. **serving image on both nodes**: the same check as step 4, recorded in `vllm-v0.20.2.sqsh.verified`. The image takes about 22 GB on the file system. An existing serving image whose hash checks is kept, and a run without the option leaves it as it is.
5. **data**: in the image on the first node, `1.prepare-llm.py` writes `DIR/data/tokens-2048-1280`, `tokens-2048-160` and `tokens-1024-160` from documents of at least 4,096 tokens, keeping each document's first positions, and downloads the model to `DIR/data/model` with the first dataset. A dataset whose manifest, sizes and fingerprint verify is kept. One that fails is moved to `DIR/failed-<UTC>/` and prepared again.
6. **data on both nodes**: each node recomputes every dataset's fingerprint and checks the model pin.
7. **environment file**: `DIR/aim347.env`, written atomically.
8. **ownership**: `images`, `data`, `results` and `aim347.env` are given to `ubuntu`, who runs the tools.

Steps 3 and 5 request all of the node's CPUs, because a step task otherwise gets one. A rerun after a failure keeps the parts that verify. Exit status: 0 ready; 2 usage or a precondition, including a monitoring install that did not finish; 3 the image build, import or hash; 4 the data preparation; 5 a verification on either node.

### facilitator/install-node-collectors.sh

```bash
bash facilitator/install-node-collectors.sh [--verify] [--timeout SECONDS]
```

Runs as root on one GPU node. It writes the drop-in `/etc/systemd/system/efa-metrics.timer.d/aim347-5s.conf`, which changes the monitoring stack's EFA collector from every 30 seconds to every 5 seconds, then reloads systemd and restarts the timer. It starts `lib/counters.py lustre` as the transient unit `aim347-lustre`, which writes the Lustre client counters to `/var/lib/prometheus/node-exporter/lustre.prom` every 5 seconds for node-exporter's textfile collector. A second run changes nothing. With `--verify`, it waits up to `--timeout` seconds (default 300) until the timer runs at 5 seconds, `lustre.prom` holds `aim347_lustre_collection_success 1`, and `http://localhost:9100/metrics` serves that line and `node_amazonefa_tx_bytes`. It does not wait for `lustre_write_bytes_total`, which Lustre reports only after the node has written to the file system. Run it again on a node that replaces one of the pair.

Exit status: 0; 1 when `--verify` times out (the message names the condition it waited for); 2 when not run as root or for usage.

## Tests

The CPU tests of the tools need only the standard library:

```bash
PYTHONPATH=lib python3 -m unittest tests.test_workshop_tools tests.test_charts tests.test_serving_decode -v
```

The chart tests read the result files of the run on two `g7.48xlarge` of 2026-11-03 in `tests/fixtures/spain-20261103/`.

The whole suite, `PYTHONPATH=lib python3 -m unittest discover -s tests -v`, also runs the trainer tests when `torch==2.9.1` and `transformers==4.57.6` are installed, and skips them otherwise.
