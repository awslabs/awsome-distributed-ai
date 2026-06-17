# Cosmos Reason — Client Examples

Reference Python clients exercising three Cosmos Reason use cases against an OpenAI-compatible
vLLM endpoint.

| Script | Use case | Latency target |
|--------|----------|---------------|
| `image_vqa.py` | Single-image visual Q&A | < 1 s for short reply |
| `video_qa.py` | Short video clip Q&A | 5-15 s |
| `auto_label.py` | SDG critic loop — `<think>` reasoning + structured `<answer>` JSON | 10-30 s |

## Setup

```bash
pip install requests urllib3

# Download sample media (image + video)
./download_samples.sh

# If Pod is in-cluster, port-forward first:
kubectl port-forward svc/cosmos-reason 8000:8000 &

# OR set the operator-managed endpoint URL:
export ENDPOINT="https://cosmos-reason-<id>.elb.<region>.amazonaws.com"
```

By default all scripts hit `http://localhost:8000`. Override with `--endpoint` or
`$ENDPOINT`. Use `--insecure` if the endpoint uses a self-signed TLS certificate
(e.g., operator-managed ALB).

## Examples

```bash
# Single image
python3 image_vqa.py --image sample.jpg \
  --prompt "What is the safety risk in this scene?"

# Short video clip
python3 video_qa.py --video sample_meteor.webm \
  --prompt "Describe what is happening in this video."

# Batch SDG auto-labeling (with retry on transient errors)
python3 auto_label.py --image-dir . --output labels.jsonl --limit 1

# With self-signed cert (operator-managed ALB)
python3 image_vqa.py --endpoint https://cosmos-reason.elb.us-west-2.amazonaws.com \
  --image sample.jpg --insecure
```

## Notes

- Cosmos-Reason1 (Qwen2.5-VL) emits `<think>...</think><answer>...</answer>` inline
  in the `content` field. The scripts here parse those tags.
- Cosmos-Reason2 (Qwen3-VL) with `--reasoning-parser qwen3` separates `<think>` into
  the response's `reasoning_content` field. The scripts handle both formats.
- `MODEL_ID` is read from `$MODEL_ID` env var, defaulting to `nvidia/Cosmos-Reason1-7B`.
- `auto_label.py` supports `--max-retries N` (default 3) for transient HTTP errors
  (429, 502, 503, 504) with exponential backoff.
