import argparse
import os
import re
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from math_verify import parse, verify
from datetime import datetime
import accelerate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm_server_host", type=str, default="", help="The server IP"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="The model to use",
    )
    args = parser.parse_args()

    dataset_id = "AI-MO/NuminaMath-TIR"
    train_dataset = load_dataset(dataset_id, split="train")

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

    train_dataset = train_dataset.map(make_conversation)

    train_dataset = train_dataset.remove_columns(["messages", "problem"])

    def simple_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content in completion_contents:
            reward = 0.0
            if "<think>" in content:
                reward += 1/6
            if content.startswith("<think>"):
                reward += 1/6
            if "</think>" in content:
                reward += 1/6
            if "<answer>" in content:
                reward += 1/6
            if "</answer>" in content:
                reward += 1/6
            if content.endswith("</answer>"):
                reward += 1/6
            rewards.append(reward)
        return rewards

    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    
    def accuracy_reward(completions, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        solutions = kwargs["solution"]
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, solution in zip(completion_contents, solutions):
            gold_parsed = parse(solution)
            if len(gold_parsed) != 0:
                answer_parsed = parse(content)
                if verify(gold_parsed, answer_parsed):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(None)
        return rewards

    parent_dir = os.path.dirname(__file__)
    date_time_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(parent_dir, date_time_dir, args.model + "-GRPO")
    output_dir_list = [output_dir]
    accelerate.utils.broadcast_object_list(output_dir_list)
    output_dir = output_dir_list[0]

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        remove_unused_columns=False,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        report_to="wandb",
        use_vllm=True,
        vllm_server_host=args.vllm_server_host,
        vllm_server_timeout=600,
        save_strategy="steps",
        save_steps=100,
        torch_empty_cache_steps=10,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[simple_format_reward, format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
