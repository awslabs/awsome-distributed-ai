#!/usr/bin/env python3
"""Optional B of the AIM347 workshop: serve the model on every GPU and measure its decode rate.

`python3 lib/serving_decode.py run` is tools/serve.sh. On the session allocation it starts
one vLLM server per GPU through 8.serve-llm.sh, each with the whole SmolLM2-1.7B snapshot
that provisioning downloaded. When every server answers, it sends requests that each
generate the same number of tokens, with 1, 4, 16 and then 64 requests per GPU at a time,
and reads the time of a decode step from the servers' own counters. It prints the decode
rate per GPU beside the bound that the memory-read ceiling of Module 1 sets. The servers
end with the allocation that tools/serve.sh holds. Standard library only; it runs on the
login node.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

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


class ServingFailed(Exception):
    """The servers did not all start, or a request failed or returned another number of tokens."""


def image_path(env):
    home = env.get('AIM347_HOME') or '/fsx/aim347/participant'
    return Path(env.get('VLLM_IMAGE') or Path(home) / 'images' / IMAGE_NAME)


def server_endpoints(nodelist, gpus):
    """The address of every server: 8.serve-llm.sh puts the server of GPU g of a node on port 8100 + g."""
    endpoints = []
    for node in subprocess.check_output(['scontrol', 'show', 'hostnames', nodelist], text=True).split():
        fields = subprocess.check_output(['scontrol', 'show', 'node', node, '-o'], text=True).split()
        address = next(field.split('=', 1)[1] for field in fields if field.startswith('NodeAddr='))
        endpoints += [f'http://{address}:{8100 + gpu}' for gpu in range(gpus)]
    return endpoints


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
    of the step: WEIGHT_BYTES / ceiling / seconds per step. It counts only the weights. The
    ceiling is the rate of the 1 GiB streaming-read kernel, so a step can read faster and
    MBU can exceed 1.
    """
    step = statistics.median(record['server_mean_itl_seconds'] for record in level['servers'].values())
    return dict(requests_per_gpu=level['requests_per_gpu'], seconds_per_step=step,
                tokens_per_second_per_gpu=level['requests_per_gpu'] / step,
                mbu=WEIGHT_BYTES / ceiling / step if ceiling else None)


def bound_lines(ceiling):
    if not ceiling:
        return ['MBU needs your memory-read ceiling: run bash tools/measure-ceilings.sh, then this tool again.']
    read = WEIGHT_BYTES / ceiling
    return [f'Reading the weights ({WEIGHT_BYTES / 1e9:.2f} GB) at your memory-read ceiling '
            f'({ceiling / 1e9:.0f} GB/s per GPU) takes {read * 1e3:.2f} ms.',
            f'At {read * 1e3:.2f} ms per step, a GPU with one request generates {1 / read:,.0f} tokens/s. '
            f'MBU is {read * 1e3:.2f} ms / ms per step.']


TABLE_HEADER = 'requests per GPU   ms per step   decode tokens/s per GPU   MBU'


def mbu_text(mbu):
    return f'{100 * mbu:.0f} %' if mbu is not None else 'n/a'


def row_line(summary):
    return (f"{summary['requests_per_gpu']:>16}   {summary['seconds_per_step'] * 1e3:>11.2f}   "
            f"{summary['tokens_per_second_per_gpu']:>23,.0f}   {mbu_text(summary['mbu'])}")


def wait_for_servers(process, endpoints):
    """The seconds until every server answers /health. Raises ServingFailed when the step ends or time runs out."""
    started, report = time.monotonic(), REPORT_SECONDS
    while True:
        elapsed = time.monotonic() - started
        if process.poll() is not None:
            raise ServingFailed(f'the serving step ended with exit status {process.returncode} '
                                'before every server answered')
        healthy = sum(1 for endpoint in endpoints if answers(endpoint))
        if healthy == len(endpoints):
            return elapsed
        if elapsed >= STARTUP_SECONDS:
            raise ServingFailed(f'only {healthy} of {len(endpoints)} servers answered within {STARTUP_SECONDS} s')
        if elapsed >= report:
            print(f'[aim347] {elapsed:.0f} s: {healthy} of {len(endpoints)} servers answer', flush=True)
            report += REPORT_SECONDS
        time.sleep(2)


def measure(process, run_dir, record, ceiling):
    """Wait for the servers, warm them up and measure every level. Returns the exit code."""
    endpoints = record['endpoints']
    try:
        waited = wait_for_servers(process, endpoints)
    except ServingFailed as error:
        print(f'[aim347] {error}. Last 40 lines of {run_dir / "launch.log"}:')
        print('\n'.join(workshop.tail(run_dir / 'launch.log')))
        print(f'[aim347] each server writes its own log, {run_dir}/<node>-gpu<n>.log')
        return 4
    print(f'[aim347] all {len(endpoints)} servers answer after {waited:.0f} s', flush=True)
    record['startup_seconds'] = waited
    try:
        with ThreadPoolExecutor(len(endpoints)) as pool:
            warmup = list(pool.map(decode_request, endpoints))
    except ServingFailed as error:
        print(f'[aim347] a request failed: {error}')
        return 4
    record['prompt_tokens'] = warmup[0]['prompt_tokens']
    print(f"[aim347] each request: a prompt of {warmup[0]['prompt_tokens']} tokens, then {OUTPUT_TOKENS} "
          f'generated tokens; each client sends {ROUNDS} requests in turn')
    print('\n'.join(bound_lines(ceiling)))
    print(TABLE_HEADER, flush=True)
    for per_gpu in LEVELS:
        try:
            level = measure_level(endpoints, per_gpu)
        except ServingFailed as error:
            print(f'[aim347] a request failed at {per_gpu} requests per GPU: {error}')
            return 4
        except (OSError, ValueError) as error:
            print(f'[aim347] the server counters at {per_gpu} requests per GPU cannot be used: {error}')
            return 5
        summary = level_summary(level, ceiling)
        record['levels'].append(dict(level, **summary))
        workshop.write_atomic(run_dir / 'serving.json', json.dumps(record, indent=2) + '\n')
        print(row_line(summary), flush=True)
    return 0


def run():
    """tools/serve.sh, inside its allocation. Returns the exit code."""
    env = os.environ
    if not env.get('LLM_RESULTS_DIR'):
        raise workshop.UsageError('LLM_RESULTS_DIR is not set; open a new shell so it reads aim347.env')
    results = Path(env['LLM_RESULTS_DIR'])
    image = image_path(env)
    if not image.is_file():
        print(f'AIM347: the serving image is not on this cluster: {image}\n'
              'Provisioning imports it only when facilitator/prepare-event.sh runs with --with-serving.',
              file=sys.stderr)
        return 2
    endpoints = server_endpoints(env['SLURM_JOB_NODELIST'], int(env.get('GPUS_PER_NODE', 8)))
    ceilings = workshop.read_ceilings(results)
    ceiling = ceilings['memory_read_bytes_per_second_per_gpu'] if ceilings else None
    config = json.loads((LAB / CONFIG).read_text())
    run_dir = results / f'serving-{workshop.utc_stamp()}'
    run_dir.mkdir(parents=True)
    print(f"[aim347] vLLM 0.20.2 serving {config['model_id']} at {config['model_revision'][:8]}: "
          f"one server per GPU, {len(endpoints)} servers on {env['SLURM_JOB_NUM_NODES']} nodes")
    print(f'[aim347] results {run_dir}', flush=True)
    record = dict(model_id=config['model_id'], model_revision=config['model_revision'], image=str(image),
                  endpoints=endpoints, output_tokens=OUTPUT_TOKENS, rounds=ROUNDS, prompt=PROMPT,
                  weight_bytes=WEIGHT_BYTES, memory_read_bytes_per_second_per_gpu=ceiling, levels=[])
    with (run_dir / 'launch.log').open('w') as log:
        process = subprocess.Popen(['bash', str(LAB / '8.serve-llm.sh')], stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, env=dict(env, VLLM_IMAGE=str(image), SERVING_RUN=run_dir.name))
    try:
        return measure(process, run_dir, record, ceiling)
    finally:
        # The servers are tasks of the step, so they end with its srun; salloc then releases the nodes.
        print('[aim347] stopping the servers', flush=True)
        process.terminate()
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('run', help='tools/serve.sh')
    parser.parse_args(argv)
    try:
        return run()
    except workshop.UsageError as error:
        print(error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('\n[aim347] interrupted')
        return 130


if __name__ == '__main__':
    sys.exit(main())
