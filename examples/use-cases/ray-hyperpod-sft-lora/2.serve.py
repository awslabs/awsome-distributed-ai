"""
Step 2: serve the merged model with Ray Serve on the same Ray cluster.

    --config args-s3.yaml   # S3 variant: downloads <s3_base_uri>/output/merged_model
    --config args.yaml      # FSx variant: loads <output_dir>/merged_model from /mnt/fsx

Submit it as a Ray job (see the README), so ray.init() runs on the head node. The application
keeps running after the job ends; remove it with
`curl -X DELETE "$RAY_DASHBOARD/api/serve/applications/"` (the image's `serve` command is not Ray's CLI).

The endpoint has NO authentication. Keep port 8000 inside the cluster (do not put it behind a
LoadBalancer, NodePort or Ingress); for access from outside, use the IAM-authenticated endpoint
from Studio or `kubectl port-forward`. Requests are size-limited below so one caller cannot
hold the GPU with a huge generation.
"""

import argparse
import os
from urllib.parse import urlparse

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import yaml
import ray
from ray import serve

MAX_PROMPT_CHARS = 4000
MAX_NEW_TOKENS = 512
DEFAULT_NEW_TOKENS = 256
LOCAL_MODEL_DIR = "/tmp/qwen3-medical/serve_model"


def parse_request(data) -> tuple:
    """Validate a request body. Returns (prompt, max_new_tokens) or raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("Body must be a JSON object")
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"'prompt' is longer than {MAX_PROMPT_CHARS} characters")
    max_tokens = data.get("max_tokens", DEFAULT_NEW_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"'max_tokens' must be an integer from 1 to {MAX_NEW_TOKENS}")
    return prompt, max_tokens


def _local_path_for_key(local_dir: str, rel: str) -> str:
    """Map the part of an S3 key below the prefix to a path inside local_dir; refuse escapes."""
    root = os.path.realpath(local_dir)
    dest = os.path.realpath(os.path.join(root, rel))
    if dest == root or os.path.commonpath([root, dest]) != root:
        raise ValueError(f"Refusing S3 key that resolves outside {local_dir}: {rel!r}")
    return dest


def download_s3_prefix(s3_uri: str, local_dir: str) -> int:
    import boto3

    parsed = urlparse(s3_uri)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/").rstrip("/") + "/"
    s3 = boto3.client("s3")
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            dest = _local_path_for_key(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, obj["Key"], dest)
            count += 1
    return count


@serve.deployment(ray_actor_options={"num_gpus": 1, "num_cpus": 2}, num_replicas=1)
class MedicalQA:
    def __init__(self, model_uri: str):
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        import torch
        from transformers import pipeline

        if model_uri.startswith("s3://"):
            n = download_s3_prefix(model_uri, LOCAL_MODEL_DIR)
            if n == 0:
                raise RuntimeError(f"No model files under {model_uri}")
            print(f"Downloaded {n} files from {model_uri}")
            model_path = LOCAL_MODEL_DIR
        else:
            model_path = model_uri

        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Model loaded — ready for inference")

    async def __call__(self, http_request):
        from starlette.responses import JSONResponse

        try:
            prompt, max_tokens = parse_request(await http_request.json())
        except ValueError as e:  # includes malformed JSON
            return JSONResponse({"error": str(e)}, status_code=400)
        messages = [{"role": "user", "content": prompt}]
        result = self.pipe(messages, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
        return {"response": result[0]["generated_text"][-1]["content"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="args-s3.yaml (S3 variant) or args.yaml (FSx variant)")
    # Ray Serve binds 127.0.0.1 by default, which other pods cannot reach.
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address inside the Ray pods")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--route-prefix", default="/medical-qa")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if "s3_base_uri" in config:
        model_uri = f"{config['s3_base_uri'].rstrip('/')}/output/merged_model"
    else:
        model_uri = os.path.join(config["script_args"]["output_dir"], "merged_model")

    ray.init(address="auto")
    serve.start(http_options={"host": args.host, "port": args.port})
    serve.run(MedicalQA.bind(model_uri=model_uri), name="medical-qa", route_prefix=args.route_prefix)
    print(f"Deployment 'medical-qa' is live: model {model_uri}, route {args.route_prefix}, port {args.port}")


if __name__ == "__main__":
    main()
