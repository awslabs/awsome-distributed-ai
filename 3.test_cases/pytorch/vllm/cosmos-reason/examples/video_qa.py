#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
video_qa.py — Short video clip Q&A against a Cosmos Reason vLLM endpoint.

Use case: AV scene understanding (Uber pattern), drive-recorder review, video moderation.

Cosmos-Reason2 (Qwen3-VL) is video-native via `--media-io-kwargs '{"video":{"num_frames":-1}}'`.
Cosmos-Reason1 (Qwen2.5-VL) uses `--limit-mm-per-prompt '{"image":10,"video":10}'`.

Example:
    python3 video_qa.py --video clip.mp4 \
        --prompt "Describe the trajectory of the vehicle in this clip."
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
import urllib3

from image_vqa import parse_reasoning_response  # reuse the parser


def encode_video(path: str) -> str:
    suffix = Path(path).suffix.lstrip(".").lower() or "mp4"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:video/{suffix};base64,{b64}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "http://localhost:8000"))
    parser.add_argument("--model", default=os.environ.get("MODEL_ID", "nvidia/Cosmos-Reason1-7B"))
    parser.add_argument("--video", required=True, help="Path to local video file (mp4 / webm)")
    parser.add_argument("--prompt", default="Describe what is happening in this video. Reason about the temporal cues.")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--system-prompt",
                        default="Answer in <think>your reasoning</think><answer>your answer</answer> format.")
    parser.add_argument("--insecure", action="store_true",
                        help="Disable TLS certificate verification (for self-signed certs)")
    args = parser.parse_args()

    verify_tls = not args.insecure
    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if not os.path.exists(args.video):
        print(f"ERROR: video not found at {args.video}", file=sys.stderr)
        return 2

    video_url = encode_video(args.video)
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": args.prompt},
            ]},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    start = time.monotonic()
    r = requests.post(f"{args.endpoint}/v1/chat/completions",
                      headers={"Content-Type": "application/json"},
                      json=payload,
                      verify=verify_tls,
                      timeout=300)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    r.raise_for_status()
    data = r.json()

    msg = data["choices"][0]["message"]
    reasoning, answer = parse_reasoning_response(msg)

    print(f"=== Response ({elapsed_ms} ms, {data['usage']['completion_tokens']} tokens) ===")
    print()
    if reasoning:
        print("--- Reasoning ---")
        print(reasoning)
        print()
    print("--- Answer ---")
    print(answer)
    print()
    print(f"--- Usage ---")
    print(json.dumps(data["usage"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
