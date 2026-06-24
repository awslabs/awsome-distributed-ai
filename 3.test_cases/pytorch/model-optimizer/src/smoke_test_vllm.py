#!/usr/bin/env python
"""Smoke-test a ModelOpt FP8 checkpoint by serving it with vLLM (offline API).

Loads the quantized checkpoint, runs a couple of greedy generations, and prints them so you
can confirm the FP8 weights round-trip to coherent output.

Install vLLM in a SEPARATE venv (it pins specific torch versions):

    python3.12 -m venv ~/model-optimizer-recipe/vllm-venv
    . ~/model-optimizer-recipe/vllm-venv/bin/activate
    pip install 'vllm==0.23.0'
    python smoke_test_vllm.py --model ~/qwen2.5-7b-fp8
"""

import argparse

from vllm import LLM, SamplingParams

DEFAULT_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "In one sentence, what is FP8 quantization and why does it speed up inference?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the exported FP8 checkpoint.")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    # quantization="modelopt" matches the ModelOpt-native hf_quant_config.json.
    # vLLM also auto-detects if you omit it.
    llm = LLM(
        model=args.model,
        quantization="modelopt",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    for out in llm.generate(DEFAULT_PROMPTS, sampling):
        print("=== PROMPT ===")
        print(out.prompt)
        print("=== GENERATION ===")
        print(out.outputs[0].text.strip())
        print()


if __name__ == "__main__":
    main()
