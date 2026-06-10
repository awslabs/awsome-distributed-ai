#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
image_vqa.py — Single-image visual question answering against a Cosmos Reason vLLM endpoint.

Use cases: drive-recorder review, content moderation, scene understanding.

Example:
    python3 image_vqa.py --image sample.jpg \
        --prompt "What is happening in this scene? Reason about the visible cues."
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
import urllib3


def encode_image(path: str) -> str:
    suffix = Path(path).suffix.lstrip(".").lower() or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{suffix};base64,{b64}"


def parse_reasoning_response(message: dict) -> Tuple[Optional[str], str]:
    """Return (reasoning_trace, answer) for both Reason1 (inline <think>) and Reason2
    (separate reasoning_content) response shapes."""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    content = message.get("content") or ""

    if reasoning:
        return reasoning.strip(), content.strip()

    # Reason1 path — <think>...</think> inline in content
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", content, re.DOTALL)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
    if think_match:
        trace = think_match.group(1).strip()
        if answer_match:
            return trace, answer_match.group(1).strip()
        # No <answer> tag — return the rest of content after </think>
        rest = content[think_match.end():].strip()
        return trace, rest
    return None, content.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "http://localhost:8000"))
    parser.add_argument("--model", default=os.environ.get("MODEL_ID", "nvidia/Cosmos-Reason1-7B"))
    parser.add_argument("--image", required=True, help="Path to local image file")
    parser.add_argument("--prompt", default="What is in this image, and what is happening? Reason about visible cues.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--system-prompt",
                        default="Answer in <think>your reasoning</think><answer>your answer</answer> format.")
    parser.add_argument("--insecure", action="store_true",
                        help="Disable TLS certificate verification (for self-signed certs)")
    args = parser.parse_args()

    verify_tls = not args.insecure
    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    image_url = encode_image(args.image)
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
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
