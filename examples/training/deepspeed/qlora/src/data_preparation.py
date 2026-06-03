"""
Dataset preparation for medical reasoning fine-tuning.

This module handles loading the medical-o1-reasoning-SFT dataset,
formatting it for instruction tuning, and tokenizing for training.
"""

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Chat template for Qwen3 model with thinking mode
# This format enables the model to show its reasoning process
CHAT_TEMPLATE = """<|im_start|>system
You are a medical AI assistant trained to provide detailed reasoning for medical questions.
Think through the problem step-by-step before providing your final answer.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>
{reasoning}
</think>

{response}<|im_end|>"""


# Alternative simpler format without explicit thinking tags
SIMPLE_CHAT_TEMPLATE = """<|im_start|>system
You are a medical AI assistant. Provide detailed reasoning for medical questions.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
## Reasoning
{reasoning}

## Answer
{response}<|im_end|>"""


def load_medical_dataset(
    dataset_name: str = "FreedomIntelligence/medical-o1-reasoning-SFT",
    config_name: str = "en",
    max_samples: Optional[int] = None,
    train_split: float = 0.95
) -> Tuple[Dataset, Dataset]:
    """
    Load the medical reasoning dataset from Hugging Face.

    The medical-o1-reasoning-SFT dataset contains:
    - Question: Medical question or clinical case
    - Complex_CoT: Chain-of-thought reasoning process
    - Response: Final answer/diagnosis

    Args:
        dataset_name: Hugging Face dataset identifier
        config_name: Dataset configuration ("en" for English, "zh" for Chinese)
        max_samples: Maximum samples to load (None for all)
        train_split: Fraction of data for training

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading dataset: {dataset_name} (config: {config_name})")

    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name, name=config_name, split="train")
    logger.info(f"Loaded {len(dataset)} samples")

    # Optionally limit the number of samples (useful for testing)
    if max_samples is not None and max_samples < len(dataset):
        logger.info(f"Limiting to {max_samples} samples")
        dataset = dataset.select(range(max_samples))

    # Split into train and validation
    split_dataset = dataset.train_test_split(
        test_size=1 - train_split,
        seed=42
    )

    logger.info(f"Train samples: {len(split_dataset['train'])}")
    logger.info(f"Eval samples: {len(split_dataset['test'])}")

    return split_dataset["train"], split_dataset["test"]


def format_example(example: Dict, use_thinking_tags: bool = True) -> Dict:
    """
    Format a single example into the chat template.

    Args:
        example: Dictionary with Question, Complex_CoT, and Response fields
        use_thinking_tags: Whether to use <think> tags (for Qwen3 thinking mode)

    Returns:
        Dictionary with 'text' field containing formatted conversation
    """
    template = CHAT_TEMPLATE if use_thinking_tags else SIMPLE_CHAT_TEMPLATE

    formatted_text = template.format(
        question=example["Question"].strip(),
        reasoning=example["Complex_CoT"].strip(),
        response=example["Response"].strip()
    )

    return {"text": formatted_text}


def format_dataset(
    dataset: Dataset,
    use_thinking_tags: bool = True
) -> Dataset:
    """
    Apply formatting to entire dataset.

    Args:
        dataset: Raw dataset with Question, Complex_CoT, Response columns
        use_thinking_tags: Whether to use <think> tags

    Returns:
        Formatted dataset with 'text' column
    """
    return dataset.map(
        lambda x: format_example(x, use_thinking_tags),
        remove_columns=dataset.column_names,
        desc="Formatting examples"
    )


def tokenize_function(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> Dict:
    """
    Tokenize examples for training.

    This function:
    1. Tokenizes the formatted text
    2. Truncates to max_length
    3. Creates labels for causal language modeling

    Args:
        examples: Batch of examples with 'text' field
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, and labels
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # We'll pad in the data collator
    )

    # For SFT, mask the prompt tokens in labels (set to -100) so the model
    # only trains on generating the assistant response, not the prompt.
    tokenized["labels"] = []
    for i, (input_ids, text) in enumerate(zip(tokenized["input_ids"], examples["text"])):
        labels = list(input_ids)
        # Find where the assistant response starts
        assistant_marker = "<|im_start|>assistant\n"
        marker_pos = text.find(assistant_marker)
        if marker_pos >= 0:
            # Tokenize just the prompt portion to find its length
            prompt_text = text[:marker_pos + len(assistant_marker)]
            prompt_ids = tokenizer(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]
            # Mask prompt tokens with -100
            for j in range(min(len(prompt_ids), len(labels))):
                labels[j] = -100
        tokenized["labels"].append(labels)

    return tokenized


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    use_thinking_tags: bool = True
) -> Dataset:
    """
    Complete preprocessing pipeline: format and tokenize.

    Args:
        dataset: Raw dataset with medical Q&A data
        tokenizer: Tokenizer for the target model
        max_length: Maximum sequence length
        use_thinking_tags: Whether to use <think> tags

    Returns:
        Tokenized dataset ready for training
    """
    logger.info("Formatting dataset...")
    formatted_dataset = format_dataset(dataset, use_thinking_tags)

    logger.info("Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )

    # Log some statistics
    lengths = [len(x) for x in tokenized_dataset["input_ids"]]
    logger.info(f"Sequence lengths - Min: {min(lengths)}, Max: {max(lengths)}, "
                f"Mean: {sum(lengths)/len(lengths):.0f}")

    return tokenized_dataset


def get_data_collator(tokenizer: PreTrainedTokenizer):
    """
    Get the data collator for dynamic padding during training.

    Using DataCollatorForSeq2Seq for proper padding of variable-length
    sequences with labels.

    Args:
        tokenizer: Tokenizer with pad_token set

    Returns:
        Data collator instance
    """
    from transformers import DataCollatorForSeq2Seq

    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,  # Not needed for tokenizer-only padding
        padding=True,
        label_pad_token_id=-100,  # Ignore padding tokens in loss computation
    )


def inspect_sample(dataset: Dataset, tokenizer: PreTrainedTokenizer, idx: int = 0):
    """
    Inspect a sample from the dataset for debugging.

    Args:
        dataset: Tokenized dataset
        tokenizer: Tokenizer for decoding
        idx: Index of sample to inspect
    """
    sample = dataset[idx]

    print(f"\n{'='*60}")
    print(f"Sample {idx}")
    print(f"{'='*60}")
    print(f"Input length: {len(sample['input_ids'])} tokens")
    print(f"\nDecoded text (first 500 tokens):")
    print(tokenizer.decode(sample['input_ids'][:500]))
    print(f"\n{'='*60}\n")
