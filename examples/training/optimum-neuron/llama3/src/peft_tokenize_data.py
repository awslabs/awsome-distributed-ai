import os
import argparse
from transformers import AutoTokenizer, LlamaForCausalLM

def download_model_and_tokenizer(model_id: str, model_output_path: str, tokenizer_output_path: str) -> None:
    model = LlamaForCausalLM.from_pretrained(model_id)
    model.save_pretrained(model_output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(tokenizer_output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face Model id")
    parser.add_argument("--model_output_path", type=str, required=True, help="Path to save model/weights file")
    parser.add_argument("--tokenizer_output_path", type=str, required=True, help="Path to save tokenizer file")
    args, _ = parser.parse_known_args()
    download_model_and_tokenizer(model_id=args.model_id, model_output_path=args.model_output_path, tokenizer_output_path=args.tokenizer_output_path)