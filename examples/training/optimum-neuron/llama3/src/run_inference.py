import transformers
import argparse
import torch
import os
from transformers import AutoTokenizer, LlamaForCausalLM

huggingface_token = os.environ.get("HF_TOKEN", None)
    
if not huggingface_token:
    raise ValueError("HF_TOKEN environment variable not set")

def load_finetuned_pipeline(args):
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_path,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline

def load_original_pipeline(args):
    model_id = args.model_id
        
    model = LlamaForCausalLM.from_pretrained(model_id, token=huggingface_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    return pipeline

def run_inference(pipeline, name=""):
    print(f"\n=== Running inference with {name} model ===")
    
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    print(f"\n{name} model generated text:")
    print("-" * 50)
    print(outputs[0]["generated_text"][-1])
    print("-" * 50)
    return outputs

def main(args):
    finetuned_pipeline = load_finetuned_pipeline(args)
    run_inference(finetuned_pipeline, "Fine-tuned")
    del finetuned_pipeline

    original_pipeline = load_original_pipeline(args)
    run_inference(original_pipeline, "Original")
    del original_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with both fine-tuned and original models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--model_id", type=str, required=True, help="Model id of original model")
    args = parser.parse_args()

    main(args)
