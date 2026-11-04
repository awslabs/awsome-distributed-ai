#!/usr/bin/env python3
"""Optional B of the AIM347 workshop: serve the model on every GPU and measure its decode rate.

`python3 lib/serving_decode.py run` is tools/serve.sh. On the session allocation it starts
one vLLM server per GPU through 8.serve-llm.sh, each with the whole SmolLM2-1.7B snapshot
that provisioning downloaded. When every server answers, it sends requests that each
generate the same number of tokens, with 1, 4, 16 and then 64 requests per GPU at a time,
and reads the time of a decode step from the servers' own counters. It then stops the
servers and prints the decode rate per GPU beside the bound that the memory-read ceiling
of Module 1 sets. Standard library only; it runs on the login node.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

import charts
from serving_metrics import decode_record, snapshot
import workshop

LAB = workshop.LAB
CONFIG = 'configs/smol-2048.json'          # its model pin must match $LLM_DATA_DIR/model/aim347-pin.json
IMAGE_NAME = 'vllm-v0.20.2.sqsh'           # written by facilitator/prepare-event.sh --with-serving
# A decode step reads every weight once: 1,711,376,384 parameters in BF16. SmolLM2 ties
# the output layer to the embedding matrix, so that matrix is read once per step.
WEIGHT_BYTES = 2 * workshop.MODEL_PARAMETERS
LEVELS = (1, 4, 16, 64)                    # requests per GPU that run at the same time
OUTPUT_TOKENS = 256                        # every request generates exactly this many (ignore_eos)
ROUNDS = 2                                 # requests that each client slot sends, one after the other
PROMPT = 'The time a GPU takes to generate one token of text depends on'
STARTUP_SECONDS = 1200
REPORT_SECONDS = 15
REQUEST_TIMEOUT = 300


class Stopped(Exception):
    """SIGTERM or SIGHUP reached the tool; the servers are stopped on the way out."""


class ServingFailed(Exception):
    """A request failed or returned another number of tokens."""


def image_path(env):
    home = env.get('AIM347_HOME') or '/fsx/aim347/participant'
    return Path(env.get('VLLM_IMAGE') or Path(home) / 'images' / IMAGE_NAME)


def image_problem(image, nodes):
    """None when provisioning imported the image and checked it on every node, else what is missing."""
    if not image.is_file() or not Path(f'{image}.sha256').is_file():
        return (f'AIM347: the serving image is not on this cluster: {image}\n'
                'Provisioning imports it only when facilitator/prepare-event.sh runs with --with-serving.')
    verified = Path(f'{image}.verified')
    lines = verified.read_text().splitlines() if verified.is_file() else []
    hosts = {line.split()[0] for line in lines if line.split()[1:2] == ['OK']}
    if len(hosts) < nodes:
        return (f'AIM347: {image} has not been checked on both GPU nodes.\n'
                'Run facilitator/prepare-event.sh --with-serving again.')
    return None


def launch_command():
    """One vLLM server per GPU on every node of the allocation (tensor-parallel size 1)."""
    return ['bash', str(LAB / '8.serve-llm.sh'), '--tensor-parallel-size', '1', '--config', CONFIG]


def collect_command(run_dir, gpus):
    return [sys.executable, str(LAB / 'lib' / 'serve_llm.py'), '--collect-manifest', '--output', str(run_dir),
            '--gpus-per-node', str(gpus), '--tensor-parallel-size', '1', '--config', CONFIG]


def find_step(text, job_id, pid, hostname):
    """The running step whose srun runs on this host with this PID, from `scontrol show step <job> -o`."""
    for line in text.splitlines():
        fields = dict(field.split('=', 1) for field in line.split() if '=' in field)
        host, _, srun_pid = fields.get('SrunHost:Pid', '').rpartition(':')
        if (fields.get('StepId', '').startswith(f'{job_id}.') and fields.get('State') == 'RUNNING'
                and srun_pid == str(pid) and host.split('.')[0] == hostname.split('.')[0]):
            return fields['StepId']
    return None


def answers(endpoint, timeout=5):
    try:
        with urllib.request.urlopen(endpoint.rstrip('/') + '/health', timeout=timeout) as response:
            return response.status == 200
    except OSError:
        return False


def decode_request(endpoint, tokens=OUTPUT_TOKENS, timeout=REQUEST_TIMEOUT):
    """One completion that generates exactly `tokens` tokens. Raises ServingFailed otherwise."""
    payload = dict(model='aim347', prompt=PROMPT, max_tokens=tokens, temperature=0, ignore_eos=True)
    request = urllib.request.Request(endpoint.rstrip('/') + '/v1/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            usage = json.loads(response.read()).get('usage') or {}
    except (OSError, ValueError) as error:
        raise ServingFailed(f'{endpoint}: {error}') from error
    if usage.get('completion_tokens') != tokens:
        raise ServingFailed(f"{endpoint} generated {usage.get('completion_tokens')} tokens, expected {tokens}")
    return dict(endpoint=endpoint, output_tokens=tokens, prompt_tokens=usage.get('prompt_tokens'),
                elapsed_seconds=time.perf_counter() - started)


def measure_level(endpoints, per_gpu, *, request=decode_request, snap=snapshot, sleep=time.sleep, rounds=ROUNDS):
    """per_gpu client slots for each server, each sending `rounds` requests in turn.

    Each server's counters are read before and after, and decode_record turns their
    differences into decode steps and decode seconds. It raises ValueError when the
    counters do not match the requests, for example when other traffic reached a server.
    """
    before = {endpoint: snap(endpoint) for endpoint in endpoints}
    slots = [endpoints[n % len(endpoints)] for n in range(per_gpu * len(endpoints))]
    with ThreadPoolExecutor(len(slots)) as pool:
        rows = [row for rows in pool.map(lambda endpoint: [request(endpoint) for _ in range(rounds)], slots)
                for row in rows]
    servers = {}
    for endpoint in endpoints:
        mine = [row for row in rows if row['endpoint'] == endpoint]
        for attempt in range(50):
            try:
                servers[endpoint] = decode_record(before[endpoint], snap(endpoint), mine)
                break
            except ValueError:
                # The last response can reach the client before the server updates its counters.
                if attempt == 49:
                    raise
                sleep(0.1)
    return dict(requests_per_gpu=per_gpu, requests=len(rows), servers=servers)


def level_summary(level, ceiling=None):
    """The median step time over the servers, and from it the decode rate per GPU and MBU.

    MBU is the time that reading the weights at the memory-read ceiling takes, as a share
    of the step: WEIGHT_BYTES / ceiling / seconds per step. It counts only the weights.
    """
    step = statistics.median(record['server_mean_itl_seconds'] for record in level['servers'].values())
    return dict(requests_per_gpu=level['requests_per_gpu'], seconds_per_step=step,
                tokens_per_second_per_gpu=level['requests_per_gpu'] / step,
                mbu=WEIGHT_BYTES / ceiling / step if ceiling else None)


def bound_lines(ceiling):
    if not ceiling:
        return ['MBU needs your memory-read ceiling: run bash tools/hold-and-measure.sh, then this tool again.']
    read = WEIGHT_BYTES / ceiling
    return [f'Reading the weights ({WEIGHT_BYTES / 1e9:.2f} GB) at your memory-read ceiling '
            f'({ceiling / 1e9:.0f} GB/s per GPU) takes {read * 1e3:.2f} ms.',
            f'At one request per GPU, decode runs at most {1 / read:,.0f} tokens/s per GPU. '
            f'MBU is {read * 1e3:.2f} ms / ms per step.']


TABLE_HEADER = 'requests per GPU   ms per step   decode tokens/s per GPU   MBU'


def mbu_text(mbu):
    return f'{100 * mbu:.0f} %' if mbu is not None else 'n/a'


def row_line(summary):
    return (f"{summary['requests_per_gpu']:>16}   {summary['seconds_per_step'] * 1e3:>11.2f}   "
            f"{summary['tokens_per_second_per_gpu']:>23,.0f}   {mbu_text(summary['mbu'])}")


def chart_lines(summaries):
    first = summaries[0]['tokens_per_second_per_gpu']
    rows = []
    for summary in summaries:
        count, rate = summary['requests_per_gpu'], summary['tokens_per_second_per_gpu']
        detail = f"MBU {mbu_text(summary['mbu'])}"
        if summary is not summaries[0]:
            detail = f'{rate / first:.1f}x one request, ' + detail
        rows.append(dict(label=f"{count} request{'s' if count > 1 else ''} per GPU", value=rate,
                         text=f'{rate:,.0f}  ({detail})'))
    return charts.bar_chart('Decode tokens/s per GPU by requests per GPU (higher is better)', rows,
                            'tokens/s per GPU')


def wait_for_servers(process, run_dir, servers, gpus, env, *, runner, health, clock, sleep, out):
    """(endpoints, seconds) once every server answers /health, or (None, reason)."""
    started, endpoints, report = clock(), None, REPORT_SECONDS
    while True:
        elapsed = clock() - started
        if process.poll() is not None:
            return None, f'the serving step ended with exit status {process.returncode} before every server answered'
        if endpoints is None and len(list(run_dir.glob('node-*/replica-*.json'))) == servers:
            result = runner(collect_command(run_dir, gpus), cwd=str(LAB), env=env, capture_output=True, text=True)
            if result.returncode != 0:
                lines = (result.stderr or result.stdout or '').strip().splitlines()
                return None, 'collecting the server records failed: ' + (lines[-1] if lines else f'exit {result.returncode}')
            endpoints = (run_dir / 'endpoints.txt').read_text().split()
        healthy = sum(1 for endpoint in endpoints or [] if health(endpoint))
        if endpoints and healthy == len(endpoints):
            return endpoints, elapsed
        if elapsed >= STARTUP_SECONDS:
            return None, f'only {healthy} of {servers} servers answered within {STARTUP_SECONDS} s'
        if elapsed >= report:
            out(f'[aim347] {elapsed:.0f} s: {healthy} of {servers} servers answer', flush=True)
            report += REPORT_SECONDS
        sleep(2)


def stop_servers(process, job_id, *, runner, hostname, sleep=time.sleep, wait_seconds=120):
    """SIGTERM to the serving step, so that serve_llm.py stops its servers; SIGKILL after wait_seconds.

    Slurm turns a SIGTERM sent to the local srun into SIGKILL for the step's tasks, so
    the step itself is signalled. Returns the exit status of srun.
    """
    if process.poll() is not None:
        return process.returncode
    step = None
    for _ in range(5):
        text = runner(['scontrol', 'show', 'step', str(job_id), '-o'], capture_output=True, text=True).stdout
        step = find_step(text or '', job_id, process.pid, hostname)
        if step or process.poll() is not None:
            break
        sleep(1)
    if step:
        runner(['scancel', '--signal=TERM', step], capture_output=True, text=True)
    elif process.poll() is None:
        process.terminate()
    try:
        return process.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        if step:
            runner(['scancel', step], capture_output=True, text=True)
        else:
            process.kill()
        return process.wait()


def unique_run_dir(results):
    stamp = workshop.utc_stamp()
    run_dir, suffix = Path(results) / f'serving-{stamp}', 1
    while run_dir.exists():
        suffix += 1
        run_dir = Path(results) / f'serving-{stamp}-{suffix}'
    return run_dir


def measure(process, run_dir, record, summaries, servers, gpus, env, ceiling, *, runner, health, request, snap,
            clock, sleep, out):
    """Wait for the servers, warm them up and measure every level. Returns the exit code."""
    endpoints, waited = wait_for_servers(process, run_dir, servers, gpus, env, runner=runner, health=health,
                                         clock=clock, sleep=sleep, out=out)
    if endpoints is None:
        out(f'[aim347] {waited}. Last 40 lines of {run_dir / "launch.log"}:')
        out('\n'.join(workshop.tail(run_dir / 'launch.log')))
        out(f'[aim347] each server writes its own log, {run_dir}/node-<n>/replica-<n>.log')
        return 4
    out(f'[aim347] all {servers} servers answer after {waited:.0f} s', flush=True)
    record.update(endpoints=endpoints, startup_seconds=waited)
    try:
        with ThreadPoolExecutor(len(endpoints)) as pool:
            warmup = list(pool.map(request, endpoints))
    except ServingFailed as error:
        out(f'[aim347] a request failed: {error}')
        return 4
    record['prompt_tokens'] = warmup[0]['prompt_tokens']
    out(f"[aim347] each request: a prompt of {warmup[0]['prompt_tokens']} tokens, then {OUTPUT_TOKENS} "
        f'generated tokens; each client sends {ROUNDS} requests in turn')
    out('\n'.join(bound_lines(ceiling)))
    out(TABLE_HEADER, flush=True)
    for per_gpu in LEVELS:
        try:
            level = measure_level(endpoints, per_gpu, request=request, snap=snap, sleep=sleep)
        except ServingFailed as error:
            out(f'[aim347] a request failed at {per_gpu} requests per GPU: {error}')
            return 4
        except (OSError, ValueError) as error:
            out(f'[aim347] the server counters at {per_gpu} requests per GPU cannot be used: {error}')
            return 5
        summary = level_summary(level, ceiling)
        summaries.append(summary)
        record['levels'].append(dict(level, **summary))
        workshop.write_atomic(run_dir / 'serving.json', json.dumps(record, indent=2) + '\n')
        out(row_line(summary), flush=True)
    return 0


def run(env=None, state=None, *, popen=subprocess.Popen, runner=subprocess.run, health=answers,
        request=decode_request, snap=snapshot, clock=time.monotonic, sleep=time.sleep, out=print, hostname=None):
    """tools/serve.sh. Returns the exit code."""
    env = dict(os.environ if env is None else env)
    if not env.get('LLM_RESULTS_DIR'):
        raise workshop.UsageError('LLM_RESULTS_DIR is not set; open a new shell so it reads aim347.env')
    results = Path(env['LLM_RESULTS_DIR'])
    nodes, gpus = int(env.get('SLURM_JOB_NUM_NODES', 2)), int(env.get('GPUS_PER_NODE', 8))
    servers = nodes * gpus
    image = image_path(env)
    problem = image_problem(image, nodes)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if workshop.read_running(state):
        print('AIM347: a stage is still running. Run this tool after it finishes; '
              'bash tools/run.sh attach follows it.', file=sys.stderr)
        return 2
    if workshop.read_serving(state):
        print('AIM347: tools/serve.sh is already running in another shell.', file=sys.stderr)
        return 2

    def say(*values, **kwargs):
        # After a hang-up every write to the closed terminal fails; the servers must still stop.
        try:
            out(*values, **kwargs)
        except OSError:
            pass

    ceilings = workshop.read_ceilings(results)
    ceiling = ceilings['memory_read_bytes_per_second_per_gpu'] if ceilings else None
    config = json.loads((LAB / CONFIG).read_text())
    run_dir = unique_run_dir(results)
    run_dir.mkdir(parents=True)
    say(f"[aim347] vLLM 0.20.2 serving {config['model_id']} at {config['model_revision'][:8]}: "
        f'one server per GPU, {servers} servers on {nodes} nodes')
    say(f'[aim347] results {run_dir}', flush=True)
    record = dict(model_id=config['model_id'], model_revision=config['model_revision'], vllm_version='0.20.2',
                  image=str(image), servers=servers, tensor_parallel_size=1, output_tokens=OUTPUT_TOKENS,
                  rounds=ROUNDS, prompt=PROMPT, weight_bytes=WEIGHT_BYTES,
                  memory_read_bytes_per_second_per_gpu=ceiling, allocation_id=env.get('AIM347_ALLOCATION_ID'),
                  start_utc=time.time(), levels=[])
    launch_env = dict(env, VLLM_IMAGE=str(image), SERVING_RUN=run_dir.name)
    # run.sh and nccl-protocols.sh refuse to start while this marker names a live process.
    marker = Path(state or workshop.STATE) / 'serving'
    workshop.write_atomic(marker, json.dumps(dict(pid=os.getpid(), run=run_dir.name)), mode=0o600)
    with (run_dir / 'launch.log').open('w') as log:
        process = popen(launch_command(), cwd=str(LAB), env=launch_env, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
    summaries, code = [], 4
    try:
        code = measure(process, run_dir, record, summaries, servers, gpus, env, ceiling, runner=runner,
                       health=health, request=request, snap=snap, clock=clock, sleep=sleep, out=say)
    except (KeyboardInterrupt, Stopped):
        say('\n[aim347] interrupted')
        code = 130
    finally:
        say('[aim347] stopping the servers', flush=True)
        status = stop_servers(process, env.get('SLURM_JOB_ID', ''), runner=runner,
                              hostname=hostname or socket.gethostname(), sleep=sleep)
        marker.unlink(missing_ok=True)
        record.update(end_utc=time.time(), exit_status=code, serving_step_exit=status)
        workshop.write_atomic(run_dir / 'serving.json', json.dumps(record, indent=2) + '\n')
        say('[aim347] the servers have stopped and the GPUs are free' if status == 0 else
            f'[aim347] the serving step ended with exit status {status}; see {run_dir / "launch.log"}')
    if code == 0:
        say('')
        say('\n'.join(chart_lines(summaries)))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('run', help='tools/serve.sh')
    parser.parse_args(argv)

    def stop(signum, frame):
        # One stop is enough: a second signal must not interrupt stopping the servers.
        for number in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(number, signal.SIG_IGN)
        raise Stopped(signum)

    # A closed browser tab sends SIGHUP; the servers must not outlive the tool.
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, stop)
    try:
        return run()
    except workshop.UsageError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
