"""
Inference demo for QLoRA fine-tuned Qwen3-8B model.

This script demonstrates how to load the trained LoRA adapter and run
inference for medical reasoning tasks.

Usage:
    # With LoRA adapter (4-bit quantized, ~8GB VRAM)
    python src/inference_demo.py --adapter_path outputs/final_model

    # Merged full-precision model (~16GB VRAM)
    python src/inference_demo.py --adapter_path outputs/final_model --merge
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model_with_adapter(model_name: str, adapter_path: str):
    """Load the base model in 4-bit and apply the LoRA adapter."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def load_merged_model(model_name: str, adapter_path: str):
    """Load the base model in bf16, merge the adapter, and unload."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512):
    """Generate a medical reasoning response using the Qwen3 chat template."""
    prompt = (
        "<|im_start|>system\n"
        "You are a medical AI assistant trained to provide detailed reasoning "
        "for medical questions.\n"
        "Think through the problem step-by-step before providing your final "
        "answer.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


EXAMPLE_QUESTIONS = [
    "A 45-year-old male presents with sudden onset chest pain radiating to the left arm, "
    "diaphoresis, and shortness of breath. ECG shows ST elevation in leads II, III, and aVF. "
    "What is the most likely diagnosis and immediate management?",
    "A 30-year-old female presents with fatigue, weight gain, cold intolerance, and constipation "
    "for the past 6 months. Lab results show elevated TSH and low free T4. "
    "What is the diagnosis and treatment?",
]


def main():
    parser = argparse.ArgumentParser(description="Inference demo for QLoRA fine-tuned Qwen3-8B")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B", help="Base model name")
    parser.add_argument("--adapter_path", required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into bf16 model")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--question", type=str, default=None, help="Custom question (optional)")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if args.merge:
        print("Loading merged full-precision model (~16GB VRAM)...")
        model = load_merged_model(args.model_name, args.adapter_path)
    else:
        print("Loading 4-bit quantized model with LoRA adapter (~8GB VRAM)...")
        model = load_model_with_adapter(args.model_name, args.adapter_path)

    questions = [args.question] if args.question else EXAMPLE_QUESTIONS
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question[:100]}...")
        print(f"{'='*60}")
        response = generate_response(model, tokenizer, question, args.max_new_tokens)
        print(response)

    print(f"\n{'='*60}")
    print("Inference complete.")


if __name__ == "__main__":
    main()
