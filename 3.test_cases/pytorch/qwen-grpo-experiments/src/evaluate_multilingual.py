# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Multilingual evaluation script for Phase 1.

Tests whether the model reasons and answers in the correct language.
Evaluates across 5 languages (en, fr, de, es, it) with 10 prompts each.

Usage:
    python evaluate_multilingual.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --adapter_path /fsx/qwen-grpo/checkpoints/multilingual/sft \
        --output_file /fsx/qwen-grpo/results/multilingual_eval.json
"""

import argparse
import json
import os
import sys
import time

import torch

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except ImportError:
    print("ERROR: langdetect required. pip install langdetect")
    sys.exit(1)

# Test prompts per language
TEST_PROMPTS = {
    "en": [
        "What is photosynthesis and why is it important?",
        "Explain the water cycle in simple terms.",
        "What causes earthquakes?",
        "How does a computer processor work?",
        "What is the difference between weather and climate?",
        "Why do leaves change color in autumn?",
        "How does gravity work?",
        "What is artificial intelligence?",
        "Explain how vaccines work.",
        "What causes thunder and lightning?",
    ],
    "fr": [
        "Qu'est-ce que la photosynthese et pourquoi est-elle importante?",
        "Expliquez le cycle de l'eau en termes simples.",
        "Qu'est-ce qui cause les tremblements de terre?",
        "Comment fonctionne un processeur informatique?",
        "Quelle est la difference entre la meteo et le climat?",
        "Pourquoi les feuilles changent-elles de couleur en automne?",
        "Comment fonctionne la gravite?",
        "Qu'est-ce que l'intelligence artificielle?",
        "Expliquez comment fonctionnent les vaccins.",
        "Qu'est-ce qui cause le tonnerre et les eclairs?",
    ],
    "de": [
        "Was ist Photosynthese und warum ist sie wichtig?",
        "Erklaren Sie den Wasserkreislauf in einfachen Worten.",
        "Was verursacht Erdbeben?",
        "Wie funktioniert ein Computerprozessor?",
        "Was ist der Unterschied zwischen Wetter und Klima?",
        "Warum verfaerben sich Blaetter im Herbst?",
        "Wie funktioniert die Schwerkraft?",
        "Was ist kunstliche Intelligenz?",
        "Erklaren Sie, wie Impfstoffe wirken.",
        "Was verursacht Donner und Blitz?",
    ],
    "es": [
        "Que es la fotosintesis y por que es importante?",
        "Explica el ciclo del agua en terminos simples.",
        "Que causa los terremotos?",
        "Como funciona un procesador de computadora?",
        "Cual es la diferencia entre el tiempo y el clima?",
        "Por que las hojas cambian de color en otono?",
        "Como funciona la gravedad?",
        "Que es la inteligencia artificial?",
        "Explica como funcionan las vacunas.",
        "Que causa los truenos y relampagos?",
    ],
    "it": [
        "Cos'e la fotosintesi e perche e importante?",
        "Spiega il ciclo dell'acqua in termini semplici.",
        "Cosa causa i terremoti?",
        "Come funziona un processore?",
        "Qual e la differenza tra meteo e clima?",
        "Perche le foglie cambiano colore in autunno?",
        "Come funziona la gravita?",
        "Cos'e l'intelligenza artificiale?",
        "Spiega come funzionano i vaccini.",
        "Cosa causa tuoni e fulmini?",
    ],
}

LANG_NAMES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian"}


def detect_language(text: str) -> str:
    """Detect language of text."""
    import re
    try:
        clean = re.sub(r'[0-9\+\-\*\/\=\%\$]', '', text)
        clean = re.sub(r'[^\w\s]', ' ', clean)
        if len(clean.strip()) < 20:
            return "too_short"
        return detect(clean)
    except Exception:
        return "error"


def load_model_and_tokenizer(model_path: str, adapter_path: str = None):
    """Load model with optional LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a single response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def evaluate(model, tokenizer, model_name: str = "unknown"):
    """Run full evaluation across all languages."""
    results = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_language": {},
        "per_example": [],
    }

    total_correct = 0
    total_count = 0

    for lang_code, prompts in TEST_PROMPTS.items():
        lang_name = LANG_NAMES[lang_code]
        correct = 0

        for i, question in enumerate(prompts):
            # Format prompt
            system_msg = (
                f"You are a helpful assistant. "
                f"Reason in {lang_name}. Answer in {lang_name}. "
                f"Keep your final answer concise (1-2 sentences)."
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            response = generate_response(model, tokenizer, prompt)
            detected = detect_language(response)
            is_correct = (detected == lang_code)

            if is_correct:
                correct += 1

            results["per_example"].append({
                "language": lang_code,
                "prompt_idx": i,
                "question": question[:80],
                "response_preview": response[:200],
                "detected_language": detected,
                "expected_language": lang_code,
                "correct": is_correct,
            })

        accuracy = correct / len(prompts) * 100
        results["per_language"][lang_code] = {
            "correct": correct,
            "total": len(prompts),
            "accuracy": accuracy,
        }
        total_correct += correct
        total_count += len(prompts)

        print(f"  {lang_name}: {correct}/{len(prompts)} ({accuracy:.1f}%)")

    results["overall"] = {
        "correct": total_correct,
        "total": total_count,
        "accuracy": total_correct / total_count * 100,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate multilingual language compliance")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output_file", type=str,
                        default="/fsx/qwen-grpo/results/multilingual_eval.json")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for this evaluation run")
    args = parser.parse_args()

    model_name = args.model_name or f"{args.model_path}"
    if args.adapter_path:
        model_name += f" + {os.path.basename(args.adapter_path)}"

    print(f"=== Multilingual Evaluation: {model_name} ===")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.adapter_path)

    results = evaluate(model, tokenizer, model_name)

    print(f"\n=== Overall: {results['overall']['accuracy']:.1f}% "
          f"({results['overall']['correct']}/{results['overall']['total']}) ===")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {args.output_file}")


if __name__ == "__main__":
    main()
