# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Interactive GRPO Inference - Using same prompt format as evaluate_grpo.py

This script uses the SAME prompt format that achieved 100% reasoning accuracy
in evaluation (chat template with system message).

Usage:
    # Interactive mode
    python inference_grpo_new.py --use_grpo --reasoning_language Spanish
    
    # Single prompt
    python inference_grpo_new.py --use_grpo --reasoning_language French --prompt "What is 15% of 80?"
"""

import argparse
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

SUPPORTED_LANGUAGES = ["English", "French", "German", "Spanish", "Italian"]


def load_grpo_model(base_model: str, sft_checkpoint: str, grpo_checkpoint: str):
    """
    Load model with GRPO checkpoint - SAME as evaluate_grpo.py
    
    Chain: Base model → Merge SFT LoRA → Load GRPO LoRA
    """
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,  # Required: GPT-OSS model uses custom code on HF Hub
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and merge SFT checkpoint
    print(f"Loading and merging SFT checkpoint: {sft_checkpoint}")
    model = PeftModel.from_pretrained(model, sft_checkpoint)
    model = model.merge_and_unload()
    print("SFT weights merged into base model!")
    
    # Load GRPO LoRA on top
    print(f"Loading GRPO checkpoint: {grpo_checkpoint}")
    model = PeftModel.from_pretrained(model, grpo_checkpoint)
    print("GRPO LoRA loaded!")
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, reasoning_language: str = "English", max_new_tokens: int = 1024):
    """
    Generate response - SAME format as evaluate_grpo.py
    
    Uses chat template with system message (the format that got 100% accuracy)
    """
    # Build chat messages - SAME as evaluate_grpo.py
    messages = [
        {"role": "system", "content": f"reasoning language: {reasoning_language}\nanswer language: {reasoning_language}"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        chat_prompt = f"System: reasoning language: {reasoning_language}\nanswer language: {reasoning_language}\nUser: {prompt}\nAssistant:"
    
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_length = inputs["input_ids"].shape[1]
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    print("\nGenerating...")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
        response = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
    
    return response.strip()


def extract_reasoning(response: str) -> str:
    """Extract reasoning/analysis section from response - SAME as evaluate_grpo.py"""
    # Pattern 1: assistantanalysis...assistantfinal
    assistant_match = re.search(r'assistantanalysis\s*(.*?)\s*assistantfinal', response, re.DOTALL | re.IGNORECASE)
    if assistant_match:
        return assistant_match.group(1).strip()
    
    # Pattern 2: <analysis>...</analysis>
    analysis_match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL | re.IGNORECASE)
    if analysis_match:
        return analysis_match.group(1).strip()
    
    # Pattern 3: analysis...final
    analysis_final_match = re.search(r'analysis\s*(.*?)\s*final', response, re.DOTALL | re.IGNORECASE)
    if analysis_final_match:
        reasoning = analysis_final_match.group(1).strip()
        if len(reasoning) > 20:
            return reasoning
    
    # Fallback
    final_match = re.search(r'(.*?)(?:final|the answer)', response, re.DOTALL | re.IGNORECASE)
    if final_match:
        reasoning = final_match.group(1).strip()
        if len(reasoning) > 50:
            return reasoning
    
    return response[:500] if len(response) > 500 else response


def extract_final_answer(response: str) -> str:
    """Extract final answer from response - SAME as evaluate_grpo.py"""
    # Pattern 1: assistantfinal...
    assistant_final_match = re.search(r'assistantfinal\s*(.*?)(?:$|assistant)', response, re.DOTALL | re.IGNORECASE)
    if assistant_final_match:
        return assistant_final_match.group(1).strip()
    
    # Pattern 2: <final>...</final>
    final_match = re.search(r'<final>(.*?)</final>', response, re.DOTALL | re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()
    
    # Pattern 3: final...
    final_word_match = re.search(r'final\s*(.*?)(?:\.|$)', response, re.DOTALL | re.IGNORECASE)
    if final_word_match:
        answer = final_word_match.group(1).strip()
        if len(answer) > 0:
            return answer[:500]
    
    return response[-200:] if len(response) > 200 else response


def interactive_loop(model, tokenizer, max_tokens: int = 1024, reasoning_language: str = None):
    """Run interactive prompt loop."""
    import sys
    import textwrap
    
    print("\n" + "="*60)
    print("Interactive GRPO Inference (evaluate_grpo.py format)")
    if reasoning_language:
        print(f"Reasoning Language: {reasoning_language}")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            if sys.stdin.isatty():
                prompt = input("User: ").strip()
            else:
                prompt = sys.stdin.readline().strip()
                if not prompt:
                    print("\nEnd of input. Goodbye!")
                    break
                print(f"User: {prompt}")
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            response = generate_response(model, tokenizer, prompt, reasoning_language, max_tokens)
            
            # Parse response
            reasoning = extract_reasoning(response)
            final_answer = extract_final_answer(response)
            
            print()
            if reasoning and reasoning != response:
                print("Reasoning:")
                wrapped = textwrap.fill(reasoning, width=100)
                for line in wrapped.split('\n'):
                    print(f"    {line}")
                print()
            
            print("Answer:")
            if final_answer and final_answer != response:
                print(f"    {final_answer}")
            else:
                print(f"    {response}")
            print()
            
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
    parser = argparse.ArgumentParser(description="GRPO Interactive Inference (evaluate_grpo.py format)")
    parser.add_argument("--base_model", type=str, default="openai/gpt-oss-20b", help="Base model path")
    parser.add_argument("--use_grpo", action="store_true", help="Use GRPO model (required)")
    parser.add_argument("--sft_checkpoint", type=str, default="/fsx/checkpoints/converted-peft/lora-checkpoint-1000-peft",
                        help="SFT checkpoint path")
    parser.add_argument("--grpo_checkpoint", type=str, default="/fsx/checkpoints/checkpoint-1470-peft",
                        help="GRPO checkpoint path")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive mode)")
    parser.add_argument("--reasoning_language", type=str, default="English",
                        choices=SUPPORTED_LANGUAGES,
                        help="Language for reasoning (default: English)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("GRPO Interactive Inference")
    print("Using evaluate_grpo.py prompt format (100% accuracy)")
    print("="*60)
    print(f"SFT Checkpoint: {args.sft_checkpoint}")
    print(f"GRPO Checkpoint: {args.grpo_checkpoint}")
    print(f"Reasoning Language: {args.reasoning_language}")
    print("="*60)
    
    # Load model
    model, tokenizer = load_grpo_model(
        args.base_model,
        args.sft_checkpoint,
        args.grpo_checkpoint
    )
    
    # Single prompt or interactive
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        response = generate_response(model, tokenizer, args.prompt, args.reasoning_language, args.max_tokens)
        
        reasoning = extract_reasoning(response)
        final_answer = extract_final_answer(response)
        
        print()
        if reasoning and reasoning != response:
            print("Reasoning:")
            import textwrap
            wrapped = textwrap.fill(reasoning, width=100)
            for line in wrapped.split('\n'):
                print(f"    {line}")
            print()
        
        print("Answer:")
        if final_answer and final_answer != response:
            print(f"    {final_answer}")
        else:
            print(f"    {response}")
        print()
    else:
        interactive_loop(model, tokenizer, args.max_tokens, args.reasoning_language)


if __name__ == "__main__":
    main()
