# Adapted from DistillKit from Arcee! -> https://github.com/arcee-ai/DistillKit

import os
import torch
import torch.nn.functional as F
import argparse
import json
import yaml
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

def parse_arguments():
    parser = argparse.ArgumentParser(description="Distillation training with logits")
    
    # Project settings
    parser.add_argument("--project_name", type=str, default="distil-logits", help="Project name for logging")
    
    # Dataset settings
    parser.add_argument("--dataset_name", type=str, default="mlabonne/FineTome-100k", help="Dataset name")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model settings
    parser.add_argument("--teacher_model", type=str, default="arcee-ai/Arcee-Spark", help="Teacher model name")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen2-1.5B", help="Student model name")
    
    # Tokenizer settings
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--chat_template", type=str, 
                        default="{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
                        help="Chat template for tokenizer")
    
    # Training settings
    parser.add_argument("--save_location", type=str, choices=["hub", "fsx", "both"], default="both", help="Where to save the model: hub, fsx, or both")
    parser.add_argument("--hub_location", type=str, default="Satyach/distilled-model-v1", help="Hub location (required if save_location is hub or both)") 
    parser.add_argument("--output_dir", type=str, default="/fsx", help="FSX output directory (required if save_location is fsx or both)")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 precision")
    
    # Distillation settings
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for distillation loss")
    
    # Model config
    parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention 2")
    
    # Spectrum settings
    parser.add_argument("--layers_to_unfreeze", type=str, default=None, help="Path to spectrum YAML file")
    
    # Config file
    parser.add_argument("--config_file", type=str, default=None, 
                        help="Path to config file (JSON or YAML). Command line args override config file.")
    
    return parser.parse_args()

def load_config_file(config_path):
    """Load configuration from a JSON or YAML file"""
    if not config_path:
        return {}
        
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")

def build_config(args):
    """Build configuration dictionary from args"""
    # First load from config file if provided
    config = {}
    if args.config_file:
        file_config = load_config_file(args.config_file)
        config.update(file_config)
    
    # Then override with command line arguments
    # Project settings
    config["project_name"] = args.project_name
    
    # Dataset settings
    config["dataset"] = {
        "name": args.dataset_name,
        "split": args.dataset_split,
        "seed": args.seed
    }
    if args.num_samples is not None:
        config["dataset"]["num_samples"] = args.num_samples
    
    # Model settings
    config["models"] = {
        "teacher": args.teacher_model,
        "student": args.student_model
    }
    
    # Tokenizer settings
    config["tokenizer"] = {
        "max_length": args.max_length,
        "chat_template": args.chat_template
    }
    
    # Training settings
    config["training"] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "push_to_hub": args.save_location in ["hub", "both"],
        "hub_model_id": args.hub_location if args.save_location in ["hub", "both"] else None
    }
    
    # Save location settings
    config["save_location"] = args.save_location
    
    # Distillation settings
    config["distillation"] = {
        "temperature": args.temperature,
        "alpha": args.alpha
    }
    
    # Model config
    config["model_config"] = {
        "use_flash_attention": args.use_flash_attention
    }
    config["hub_location"] = args.hub_location
    
    # Spectrum settings
    if args.layers_to_unfreeze:
        config["spectrum"] = {
            "layers_to_unfreeze": args.layers_to_unfreeze
        }
    
    return config

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits

class LogitsTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(device)
        
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(model, student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, model, student_logits, teacher_logits, inputs, original_loss):
        device = next(model.parameters()).device
        student_logits, teacher_logits = pad_logits(student_logits.to(device), teacher_logits.to(device))
        
        student_logits_scaled = student_logits / self.config["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / self.config["distillation"]["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (self.config["distillation"]["temperature"] ** 2) / self.config["tokenizer"]["max_length"]

        return self.config["distillation"]["alpha"] * loss_kd + (1 - self.config["distillation"]["alpha"]) * original_loss

def main():
    # Parse arguments and build config
    args = parse_arguments()
    config = build_config(args)
    
    # Set up environment
    os.environ['WANDB_PROJECT'] = config["project_name"]
    
    # Load and preprocess dataset
    print(f"Loading dataset: {config['dataset']['name']}")
    dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    if "num_samples" in config["dataset"]:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))
        print(f"Using {config['dataset']['num_samples']} samples from dataset")
    
    # Load tokenizers
    print(f"Loading tokenizers for teacher ({config['models']['teacher']}) and student ({config['models']['student']})")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
    
    # Apply chat template to student tokenizer
    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]
    
    def sharegpt_format(example):
        conversations = example['conversations']
        message = []
        
        if isinstance(conversations, list):
            for conversation in conversations:
                if isinstance(conversation, dict):
                    if conversation.get('from') == 'human':
                        message.append({"role": "user", "content": conversation.get('value', '')})
                    elif conversation.get('from') == 'gpt':
                        message.append({"role": "assistant", "content": conversation.get('value', '')})
                    elif conversation.get('from') == 'system':
                        message.insert(0, {"role": "system", "content": conversation.get('value', '')})
    
        if not any(msg.get('role') == 'system' for msg in message):
            message.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    
        text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return {"text": text}
    
    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    dataset = dataset.map(sharegpt_format, remove_columns=original_columns)
    
    def tokenize_function(examples):
        return student_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    print("Dataset preparation complete. Loading models...")
    
    # Load models with configurable flash attention
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
    student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)
    
    # Optionally freeze layers of the student model based on spectrum configuration
    if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
        def freeze_student_spectrum(model, unfrozen_layers_file):
            with open(unfrozen_layers_file, 'r') as file:
                unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
            
            for name, param in model.named_parameters():
                if not any(layer in name for layer in unfrozen_layers):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
        # Apply freezing to student model
        print(f"Applying spectrum-based layer freezing from {config['spectrum']['layers_to_unfreeze']}")
        freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"])
    else:
        print("Spectrum configuration not found. All layers of the student model will be trainable.")
    
    # Training arguments
    training_arguments = TrainingArguments(**config["training"])
    
    # Create the custom SFT Trainer
    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_arguments,
    )
    
    # Add the teacher model and config to the trainer
    trainer.teacher_model = teacher_model
    trainer.config = config
    
    # Train the model
    print(f"Starting training for {config['training']['num_train_epochs']} epochs")
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    
    # Save model based on configuration
    print("Training complete. Saving model...")
    
    if config["save_location"] in ["fsx", "both"]:
        print(f"Saving model to FSX: {config['training']['output_dir']}")
        trainer.save_model(config["training"]["output_dir"])
    
    if config["save_location"] in ["hub", "both"]:
        print(f"Pushing model to hub: {config['hub_location']}")
        trainer.push_to_hub(config["hub_location"])
    
    print("Model saved successfully!")

if __name__ == "__main__":
    main()