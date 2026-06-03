# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Interactive Single-Node Multi-GPU Inference for GPT-OSS 20B.
Uses device_map="auto" to split model across 4 GPUs on g6e.12xlarge (4 x L40S 48GB).

Features:
- Interactive prompt loop (type 'quit' or 'exit' to stop)
- Direct PEFT checkpoint path via --checkpoint
- Reasoning language selection (English, French, German, Spanish, Italian)

Usage:
    # Base model (no LoRA)
    python inference_g6e.py
    
    # With specific PEFT checkpoint
    python inference_g6e.py --checkpoint /fsx/checkpoints/lora-checkpoint-1000-peft
    
    # With reasoning language
    python inference_g6e.py --checkpoint /fsx/checkpoints/lora-checkpoint-1000-peft --reasoning_language French
"""

# Supported reasoning languages (from OpenAI cookbook)
SUPPORTED_LANGUAGES = ["English", "French", "German", "Spanish", "Italian"]

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Mxfp4Config
from peft import PeftModel


def load_model(base_model_path: str, checkpoint: str = None):
    """Load model with device_map='auto' for multi-GPU inference."""
    
    num_gpus = torch.cuda.device_count()
    print(f"Loading base model: {base_model_path}")
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # Quantization config (same as training)
    quantization_config = Mxfp4Config(dequantize=True)
    
    # Set max memory per GPU (leave some headroom)
    max_memory = {i: "44GiB" for i in range(num_gpus)}
    max_memory["cpu"] = "60GiB"
    
    print(f"\nLoading model with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=quantization_config,
        trust_remote_code=True,  # Required: GPT-OSS model uses custom code on HF Hub
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left")
    
    # Print device placement
    if hasattr(model, 'hf_device_map'):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        print(f"Model distributed across devices: {devices_used}")
    
    # Load trained LoRA weights if checkpoint provided
    if checkpoint:
        adapter_safetensors = os.path.join(checkpoint, "adapter_model.safetensors")
        adapter_bin = os.path.join(checkpoint, "adapter_model.bin")
        if os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin):
            print(f"\nLoading trained LoRA adapter from: {checkpoint}")
            model = PeftModel.from_pretrained(model, checkpoint)
            print("Trained LoRA weights loaded successfully!")
        else:
            print(f"\nERROR: No adapter weights found in {checkpoint}")
            print("Expected adapter_model.safetensors or adapter_model.bin")
            raise FileNotFoundError(f"No adapter weights found in {checkpoint}")
    else:
        print("\nUsing base model (no LoRA)")
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 2048, reasoning_language: str = None):
    """Generate response using the model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User prompt
        max_new_tokens: Maximum tokens to generate
        reasoning_language: Language for chain-of-thought reasoning (English, French, German, Spanish, Italian)
    """
    
    # Format prompt as chat with optional reasoning language system prompt
    messages = []
    
    # Add system prompt for reasoning and answer language if specified
    if reasoning_language:
        system_prompt = f"reasoning language: {reasoning_language}\nanswer language: {reasoning_language}"
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        if reasoning_language:
            chat_prompt = f"System: reasoning language: {reasoning_language}\nanswer language: {reasoning_language}\nUser: {prompt}\nAssistant:"
        else:
            chat_prompt = f"User: {prompt}\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    
    # Move inputs to the device of the first model layer
    if hasattr(model, 'hf_device_map'):
        first_device = list(model.hf_device_map.values())[0]
        if isinstance(first_device, int):
            inputs = {k: v.to(f"cuda:{first_device}") for k, v in inputs.items()}
        else:
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print("\nGenerating...")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
    
    # Decode response - show full output to see reasoning
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the response
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    
    # Format the response for better readability
    response = format_response(response)
    
    return response


def format_response(response: str) -> str:
    """Format the response to make analysis and final sections more readable."""
    import re
    
    # Replace "assistantanalysis" with "Analysis:\n"
    response = re.sub(r'assistant\s*analysis', '\n[ANALYSIS]\n', response, flags=re.IGNORECASE)
    
    # Replace "assistantfinal" with "\n\n=== FINAL ANSWER ===\n"
    response = re.sub(r'assistant\s*final', '\n\n' + '='*40 + '\n[FINAL ANSWER]\n' + '='*40 + '\n', response, flags=re.IGNORECASE)
    
    # Also handle cases where it might be "<|assistant|>analysis" or "<|assistant|>final"
    response = re.sub(r'<\|assistant\|>\s*analysis', '\n[ANALYSIS]\n', response, flags=re.IGNORECASE)
    response = re.sub(r'<\|assistant\|>\s*final', '\n\n' + '='*40 + '\n[FINAL ANSWER]\n' + '='*40 + '\n', response, flags=re.IGNORECASE)
    
    return response.strip()


def interactive_loop(model, tokenizer, max_tokens: int = 2048, reasoning_language: str = None):
    """Run interactive prompt loop."""
    
    print("\n" + "="*60)
    print("Interactive Inference Mode")
    if reasoning_language:
        print(f"Reasoning Language: {reasoning_language}")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    import sys
    
    while True:
        try:
            # Check if running interactively
            if sys.stdin.isatty():
                prompt = input("You: ").strip()
            else:
                # Non-interactive mode - read from stdin
                prompt = sys.stdin.readline().strip()
                if not prompt:
                    print("\nEnd of input. Goodbye!")
                    break
                print(f"You: {prompt}")
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            response = generate_response(model, tokenizer, prompt, max_tokens, reasoning_language=reasoning_language)
            
            print("\n" + "-"*60)
            print("Assistant:")
            print("-"*60)
            print(response)
            print("-"*60 + "\n")
            
        except EOFError:
            print("\nEnd of input. Goodbye!")
            break
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS Interactive Inference")
    parser.add_argument("--base_model", type=str, default="openai/gpt-oss-20b", help="Base model path")
    parser.add_argument("--use_trained", action="store_true", help="Deprecated - use --checkpoint instead")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Full path to PEFT checkpoint (e.g., /fsx/checkpoints/lora-checkpoint-1000-peft)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive mode)")
    parser.add_argument("--reasoning_language", type=str, default=None, 
                        choices=SUPPORTED_LANGUAGES,
                        help="Language for chain-of-thought reasoning (English, French, German, Spanish, Italian)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("GPT-OSS Interactive Inference")
    print("="*60)
    print(f"Mode: {'With LoRA checkpoint' if args.checkpoint else 'Base model'}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    if args.reasoning_language:
        print(f"Reasoning Language: {args.reasoning_language}")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model(args.base_model, checkpoint=args.checkpoint)
    
    # Single prompt mode or interactive mode
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        response = generate_response(model, tokenizer, args.prompt, args.max_tokens, reasoning_language=args.reasoning_language)
        print("\n" + "-"*60)
        print("Assistant:")
        print("-"*60)
        print(response)
        print("-"*60 + "\n")
    else:
        # Start interactive loop
        interactive_loop(model, tokenizer, args.max_tokens, reasoning_language=args.reasoning_language)


if __name__ == "__main__":
    main()
