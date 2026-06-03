import json
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
import torch
import argparse
from safetensors import safe_open

def merge_lora_weights(args):
    """
    Merge LoRA (Low-Rank Adaptation) weights with a base model to create a new merged model.

    This function takes in the following arguments:
        args (Namespace): A namespace object containing the following attributes:
            base_model_path (str): Path to the base model to be adapted with LoRA weights.
            adapter_config_path (str): Path to the LoRA adapter configuration file (JSON).
            lora_safetensors_path (str): Path to the LoRA adapter weights in SafeTensors format.
            final_model_path (str): Path to save the final merged model.

    Note: This function requires the HuggingFace Transformers library and the PEFT (Parameter-Efficient Fine-Tuning) library.
    """
    # update model with lora config 
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
    with open(args.adapter_config_path, "r") as f:
        config_dict = json.load(f)
    peft_config = LoraConfig(**config_dict)
    model = PeftModel(base_model, peft_config)
    
    # load lora adapter weights and change layer name to be consistent with base model 
    lora_weights_tensors = {}
    with safe_open(args.lora_safetensors_path, framework="pt", device='cpu') as f:
        for k in f.keys():
            lora_weights_tensors[k] = f.get_tensor(k)
            
    for layer_name in list(lora_weights_tensors):
        if 'layer' in layer_name and 'lora' in layer_name:
            new_layer_name = layer_name.replace('weight', 'default.weight')
            lora_weights_tensors[new_layer_name] = lora_weights_tensors[layer_name].clone()
            del lora_weights_tensors[layer_name]
        else: # only keep lora layers. 
            del lora_weights_tensors[layer_name]
            print(f"{layer_name} is deleted!")

    # update the model lora layer weights using the consolidated adapter weight matrix 
    updated_state_dict = model.state_dict().copy()
    for layer, weights in lora_weights_tensors.items():
        updated_state_dict[layer] = weights
    model.load_state_dict(updated_state_dict)    
    merged_model = model.merge_and_unload()    
    merged_model.save_pretrained(args.final_model_path, safe_serialization=True, max_shard_size="5GB")
    print(f"Merged model saved to {args.final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_model_path", type=str)
    parser.add_argument("--adapter_config_path", type=str)
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--lora_safetensors_path", type=str)
    args = parser.parse_args()
    
    merge_lora_weights(args)
    