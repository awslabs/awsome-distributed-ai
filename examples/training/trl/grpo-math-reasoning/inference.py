import argparse
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoConfig, AutoTokenizer
from math_verify import parse

def get_tensor_parallel_size(model: str) -> int:
    config = AutoConfig.from_pretrained(model)
    num_key_value_heads = getattr(
        config, "num_key_value_heads", getattr(config, "num_attention_heads", 1)
    )
    vocab_size = getattr(config, "vocab_size", 1)
    gpus_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    tensor_parallel_size = 1
    for tp in reversed(range(1, gpus_count + 1)):
        if num_key_value_heads % tp == 0 and vocab_size % tp == 0:
            tensor_parallel_size = tp
            break
    return tensor_parallel_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="The model to use",
    )
    args = parser.parse_args()

    dataset_id = "AI-MO/NuminaMath-TIR"
    test_dataset = load_dataset(dataset_id, split="test")

    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think>reasoning process here</think><answer>answer here</answer>"
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    test_dataset = test_dataset.map(make_conversation)

    tensor_parallel_size = get_tensor_parallel_size(args.model)
    print(f"{tensor_parallel_size=}")

    llm = LLM(model=args.model, tensor_parallel_size=tensor_parallel_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for i, example in enumerate(test_dataset):
        if i >= 100:
            break
            
        prompt = example["prompt"]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
        response = llm.generate(prompt, sampling_params=SamplingParams(max_tokens=1000, temperature=0.0))
        generated_texts = response[0].outputs[0].text
        actual = parse(generated_texts)
        expected = parse(example['solution'])

        print("-" * 100)
        print(f"{prompt=}")
        print(f"{generated_texts=}")
        print(f"{expected=}")
        print(f"{actual=}")

if __name__ == "__main__":
    main()