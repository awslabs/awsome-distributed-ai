"""Quality and both latency constraints over every offered serving request."""
import json
import math
import time
import urllib.error
import urllib.request


def quality_valid(text, expected):
    """Declared task: exact structured JSON answer, including all required fields."""
    try:
        # Python equality treats True as 1. Preserve JSON value types while
        # ignoring object-key order and insignificant whitespace.
        return json.dumps(json.loads(text), sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError):
        return False


def stream_request(endpoints, index, specification, *, model, max_tokens, timeout_seconds):
    # Start before routing and encoding, within the closed-loop client slot.
    started = time.perf_counter()
    endpoint = endpoints[index % len(endpoints)]
    row = dict(request_index=index, endpoint=endpoint, status='failed', quality_valid=False,
               output_tokens=0, input_tokens=None, ttft_seconds=None, tpot_seconds=None,
               tpot_definition='client receive-time estimate (last content - first content) / (output tokens - 1); not server ITL',
               content_chunks=0, text='')
    payload = dict(model=model, messages=[{'role': 'user', 'content': specification['prompt']}],
                   max_tokens=max_tokens, temperature=0, seed=347, stream=True,
                   stream_options={'include_usage': True},
                   chat_template_kwargs={'enable_thinking': False})
    request = urllib.request.Request(endpoint.rstrip('/') + '/v1/chat/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    first = last = None
    done = False
    finish_reason = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for raw in response:
                if time.perf_counter() - started > timeout_seconds:
                    raise TimeoutError('request total-duration deadline exceeded')
                if not raw.startswith(b'data:'):
                    continue
                raw = raw[5:].strip()
                if raw == b'[DONE]':
                    done = True
                    break
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise ValueError('stream event must be a JSON object')
                usage = event.get('usage')
                if usage is not None:
                    if not isinstance(usage, dict) or any(
                            type(usage.get(key)) is not int or usage[key] < 0
                            for key in ('completion_tokens', 'prompt_tokens')):
                        raise ValueError('stream usage must contain nonnegative integer token counts')
                    row['output_tokens'] = usage['completion_tokens']
                    row['input_tokens'] = usage['prompt_tokens']
                choices = event.get('choices', [])
                if not isinstance(choices, list):
                    raise ValueError('stream choices must be a list')
                for choice in choices:
                    if not isinstance(choice, dict) or not isinstance(choice.get('delta', {}), dict):
                        raise ValueError('stream choice and delta must be JSON objects')
                    if choice.get('finish_reason') is not None:
                        if not isinstance(choice['finish_reason'], str):
                            raise ValueError('stream finish reason must be a string')
                        finish_reason = choice['finish_reason']
                    content = choice.get('delta', {}).get('content')
                    if content is not None and not isinstance(content, str):
                        raise ValueError('stream content must be a string')
                    if content:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        row['content_chunks'] += 1
                        row['text'] += content
        if not done or first is None or row['output_tokens'] < 1 or finish_reason is None:
            raise ValueError('incomplete stream, missing usage, content or finish reason')
        row['status'] = 'completed'
        row['finish_reason'] = finish_reason
        row['ttft_seconds'] = first - started
        if row['output_tokens'] > 1 and row['content_chunks'] > 1:
            row['tpot_seconds'] = (last - first) / (row['output_tokens'] - 1)
        row['quality_valid'] = quality_valid(row['text'], specification['expected_json'])
    except (TimeoutError, OSError, ValueError, KeyError) as error:
        row['status'] = 'timeout' if isinstance(error, TimeoutError) else 'failed'
        row['error'] = f'{type(error).__name__}: {error}'
    row['elapsed_seconds'] = time.perf_counter() - started
    if row['elapsed_seconds'] > timeout_seconds:
        row['status'] = 'timeout'
        row['quality_valid'] = False
    return row


def serving_goodput(rows, seconds, *, ttft_slo_seconds, tpot_slo_seconds):
    if not rows or any(not math.isfinite(value) or value <= 0 for value in (
            seconds, ttft_slo_seconds, tpot_slo_seconds)):
        raise ValueError('requests, positive measurement duration and declared SLOs required')
    qualified = 0
    for row in rows:
        valid = row['status'] == 'completed' and row['quality_valid']
        for key, limit in [('ttft_seconds', ttft_slo_seconds), ('tpot_seconds', tpot_slo_seconds)]:
            value = row[key]
            valid = valid and value is not None and math.isfinite(value) and 0 <= value <= limit
        qualified += bool(valid)
    offered = len(rows)
    completed = sum(row['status'] == 'completed' for row in rows)
    return dict(metric='Serving goodput', offered_requests=offered, completed_requests=completed,
                failed_requests=offered - completed,
                timed_out_requests=sum(row['status'] == 'timeout' for row in rows),
                quality_valid_requests=sum(row['status'] == 'completed' and row['quality_valid'] for row in rows),
                one_token_tpot_undefined_requests=sum(row['status'] == 'completed' and row['output_tokens'] == 1 for row in rows),
                tpot_unavailable_requests=sum(row['status'] == 'completed' and row['tpot_seconds'] is None for row in rows),
                qualified_requests=qualified, serving_goodput_requests_per_second=qualified / seconds,
                offered_requests_per_second=offered / seconds, qualified_fraction=qualified / offered,
                failure_fraction=(offered - completed) / offered, measurement_seconds=seconds,
                ttft_slo_seconds=ttft_slo_seconds, tpot_slo_seconds=tpot_slo_seconds,
                undefined_tpot_policy='retained in offered totals; cannot satisfy both constraints',
                quality_policy='exact JSON equality against per-prompt expected_json')


def paired_serving_report(before, after, *, experiment):
    if experiment not in ('placement', 'server_batch', 'client_load'):
        raise ValueError('declare placement, server_batch or client_load experiment')
    if before.get('allocation') != after.get('allocation'):
        raise ValueError('serving comparison changed actual allocation')
    for key in ('workload', 'gpu_budget'):
        if before[key] != after[key]:
            raise ValueError(f'serving comparison mismatch: {key}')
    fixed = {'placement': ('client_load', 'server_batch'),
             'server_batch': ('client_load', 'placement'),
             'client_load': ('placement', 'server_batch')}[experiment]
    for key in fixed:
        if before[key] != after[key]:
            raise ValueError(f'{experiment} comparison changed {key}')
    return dict(experiment=experiment, allocation=before.get('allocation'),
        allocation_evidence='recorded' if before.get('allocation') else 'declaration only; hardware qualification requires actual allocation evidence',
        before_configuration={key: before[key] for key in (
        'placement', 'server_batch', 'client_load')}, after_configuration={key: after[key] for key in (
        'placement', 'server_batch', 'client_load')},
        before_goodput_requests_per_second=before['metrics']['serving_goodput_requests_per_second'],
        after_goodput_requests_per_second=after['metrics']['serving_goodput_requests_per_second'],
        interpretation='closed-loop capacity under the declared client slots; not a fixed-arrival-rate SLA')
