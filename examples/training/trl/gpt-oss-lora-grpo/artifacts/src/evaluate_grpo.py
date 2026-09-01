# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Reasoning Language Evaluation Script v2

Evaluates whether the fine-tuned model reasons in the specified language.
Questions are in the SAME language as the reasoning language argument.

Test Setup:
- 10 prompts translated into 5 languages
- Each prompt tested in its native language with matching reasoning language
- Total: 50 test cases

Output:
- Results saved incrementally after each question
- Summary with accuracy per language

Usage:
    python evaluate_reasoning_language.py --use_trained
"""

import argparse
import re
import os
import torch
from datetime import datetime

# Language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("WARNING: langdetect not installed. Install with: pip install langdetect")

# Translation
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False
    print("WARNING: deep_translator not installed. Install with: pip install deep-translator")


# ============================================================================
# TEST PROMPTS - 10 prompts in ALL 5 languages
# Question language = Reasoning language = Expected answer language
# ============================================================================
TEST_PROMPTS = {
    "English": [
        "What is 15% of 80?",
        "If a train travels at 60 km/h for 2.5 hours, how far does it travel?",
        "A store has a 20% off sale. If a shirt originally costs $45, what is the sale price?",
        "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        "If 5 workers can build a wall in 10 days, how many days would it take 10 workers?",
        "A rectangle has a length of 12 cm and a width of 8 cm. What is its area?",
        "If you have 3 red balls and 5 blue balls in a bag, what is the probability of picking a red ball?",
        "Convert 68 degrees Fahrenheit to Celsius.",
        "A car uses 8 liters of fuel per 100 km. How much fuel is needed for a 350 km trip?",
        "If today is Wednesday, what day will it be in 100 days?",
    ],
    "French": [
        "Combien font 15% de 80 ?",
        "Si un train roule à 60 km/h pendant 2,5 heures, quelle distance parcourt-il ?",
        "Un magasin fait une réduction de 20%. Si une chemise coûte 45€, quel est le prix soldé ?",
        "Quel est le prochain nombre dans la séquence : 2, 6, 12, 20, 30, ?",
        "Si 5 ouvriers peuvent construire un mur en 10 jours, combien de jours faudrait-il à 10 ouvriers ?",
        "Un rectangle a une longueur de 12 cm et une largeur de 8 cm. Quelle est son aire ?",
        "Si vous avez 3 boules rouges et 5 boules bleues dans un sac, quelle est la probabilité de tirer une boule rouge ?",
        "Convertissez 68 degrés Fahrenheit en Celsius.",
        "Une voiture consomme 8 litres de carburant pour 100 km. Combien de carburant faut-il pour un trajet de 350 km ?",
        "Si aujourd'hui est mercredi, quel jour serons-nous dans 100 jours ?",
    ],
    "German": [
        "Was sind 15% von 80?",
        "Wenn ein Zug mit 60 km/h 2,5 Stunden fährt, wie weit kommt er?",
        "Ein Geschäft hat 20% Rabatt. Wenn ein Hemd ursprünglich 45€ kostet, was ist der Verkaufspreis?",
        "Was ist die nächste Zahl in der Folge: 2, 6, 12, 20, 30, ?",
        "Wenn 5 Arbeiter eine Mauer in 10 Tagen bauen können, wie viele Tage würden 10 Arbeiter brauchen?",
        "Ein Rechteck hat eine Länge von 12 cm und eine Breite von 8 cm. Was ist seine Fläche?",
        "Wenn Sie 3 rote Kugeln und 5 blaue Kugeln in einer Tasche haben, wie hoch ist die Wahrscheinlichkeit, eine rote Kugel zu ziehen?",
        "Rechnen Sie 68 Grad Fahrenheit in Celsius um.",
        "Ein Auto verbraucht 8 Liter Kraftstoff pro 100 km. Wie viel Kraftstoff wird für eine 350 km Fahrt benötigt?",
        "Wenn heute Mittwoch ist, welcher Tag wird es in 100 Tagen sein?",
    ],
    "Spanish": [
        "¿Cuánto es el 15% de 80?",
        "Si un tren viaja a 60 km/h durante 2,5 horas, ¿qué distancia recorre?",
        "Una tienda tiene un 20% de descuento. Si una camisa cuesta originalmente 45€, ¿cuál es el precio de venta?",
        "¿Cuál es el siguiente número en la secuencia: 2, 6, 12, 20, 30, ?",
        "Si 5 trabajadores pueden construir un muro en 10 días, ¿cuántos días tardarían 10 trabajadores?",
        "Un rectángulo tiene una longitud de 12 cm y un ancho de 8 cm. ¿Cuál es su área?",
        "Si tienes 3 bolas rojas y 5 bolas azules en una bolsa, ¿cuál es la probabilidad de sacar una bola roja?",
        "Convierte 68 grados Fahrenheit a Celsius.",
        "Un coche consume 8 litros de combustible por cada 100 km. ¿Cuánto combustible se necesita para un viaje de 350 km?",
        "Si hoy es miércoles, ¿qué día será dentro de 100 días?",
    ],
    "Italian": [
        "Quanto è il 15% di 80?",
        "Se un treno viaggia a 60 km/h per 2,5 ore, quanta distanza percorre?",
        "Un negozio ha uno sconto del 20%. Se una camicia costa originariamente 45€, qual è il prezzo scontato?",
        "Qual è il prossimo numero nella sequenza: 2, 6, 12, 20, 30, ?",
        "Se 5 operai possono costruire un muro in 10 giorni, quanti giorni ci vorrebbero per 10 operai?",
        "Un rettangolo ha una lunghezza di 12 cm e una larghezza di 8 cm. Qual è la sua area?",
        "Se hai 3 palline rosse e 5 palline blu in un sacchetto, qual è la probabilità di pescare una pallina rossa?",
        "Converti 68 gradi Fahrenheit in Celsius.",
        "Un'auto consuma 8 litri di carburante per 100 km. Quanto carburante serve per un viaggio di 350 km?",
        "Se oggi è mercoledì, che giorno sarà tra 100 giorni?",
    ],
}

SUPPORTED_LANGUAGES = ["French", "German", "Spanish", "Italian", "English"]

LANG_CODE_MAP = {
    "English": "en",
    "French": "fr", 
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
}


def extract_reasoning(response: str) -> str:
    """Extract reasoning/analysis section from response."""
    # Handle model output format: "assistantanalysis...assistantfinal"
    # Pattern 1: assistantanalysis...assistantfinal (no tags)
    assistant_match = re.search(r'assistantanalysis\s*(.*?)\s*assistantfinal', response, re.DOTALL | re.IGNORECASE)
    if assistant_match:
        return assistant_match.group(1).strip()
    
    # Pattern 2: <analysis>...</analysis>
    analysis_match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL | re.IGNORECASE)
    if analysis_match:
        return analysis_match.group(1).strip()
    
    # Pattern 3: <thinking>...</thinking>
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL | re.IGNORECASE)
    if thinking_match:
        return thinking_match.group(1).strip()
    
    # Pattern 4: analysis...final (with spaces)
    analysis_final_match = re.search(r'analysis\s*(.*?)\s*final', response, re.DOTALL | re.IGNORECASE)
    if analysis_final_match:
        reasoning = analysis_final_match.group(1).strip()
        if len(reasoning) > 20:
            return reasoning
    
    # Fallback: text before "final" or "answer"
    final_match = re.search(r'(.*?)(?:final|the answer)', response, re.DOTALL | re.IGNORECASE)
    if final_match:
        reasoning = final_match.group(1).strip()
        if len(reasoning) > 50:
            return reasoning
    
    return response[:500] if len(response) > 500 else response


def extract_final_answer(response: str) -> str:
    """Extract final answer from response."""
    # Handle model output format: "assistantfinal..."
    # Pattern 1: assistantfinal... (to end or next section)
    assistant_final_match = re.search(r'assistantfinal\s*(.*?)(?:$|assistant)', response, re.DOTALL | re.IGNORECASE)
    if assistant_final_match:
        return assistant_final_match.group(1).strip()
    
    # Pattern 2: <final>...</final>
    final_match = re.search(r'<final>(.*?)</final>', response, re.DOTALL | re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()
    
    # Pattern 3: final... (after the word "final")
    final_word_match = re.search(r'final\s*(.*?)(?:\.|$)', response, re.DOTALL | re.IGNORECASE)
    if final_word_match:
        answer = final_word_match.group(1).strip()
        if len(answer) > 0:
            return answer[:500]
    
    # Pattern 4: "the answer is..."
    answer_match = re.search(r'(?:the answer is|answer:)\s*(.*?)(?:\.|$)', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    return response[-200:] if len(response) > 200 else response


def detect_language(text: str) -> str:
    """Detect the language of text."""
    if not HAS_LANGDETECT:
        return "unknown"
    
    try:
        # First try with original text (keep some punctuation for context)
        clean_text = re.sub(r'[0-9\+\-\*\/\=\%\$\€\£\\{}\[\]]+', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # If still too short, use original text
        if len(clean_text) < 20:
            clean_text = text
        
        # Try detection
        if len(clean_text.strip()) < 10:
            return "too_short"
        
        detected = detect(clean_text)
        return detected
    except Exception as e:
        # Fallback: try with the full response
        try:
            return detect(text)
        except Exception:
            return "error"


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English."""
    if not HAS_TRANSLATOR:
        return "[Translation not available]"
    
    if source_lang == "en":
        return text
    
    try:
        translator = GoogleTranslator(source=source_lang, target='en')
        if len(text) > 4500:
            text = text[:4500]
        return translator.translate(text)
    except Exception as e:
        return f"[Translation error: {str(e)}]"


def check_language_match(expected_lang: str, detected_lang: str) -> bool:
    """Check if detected language matches expected."""
    expected_code = LANG_CODE_MAP.get(expected_lang, expected_lang.lower()[:2])
    return detected_lang == expected_code


def append_result_to_file(result: dict, output_file: str, is_first: bool = False):
    """Append a single result to the output file."""
    mode = 'w' if is_first else 'a'
    
    with open(output_file, mode, encoding='utf-8') as f:
        if is_first:
            f.write("=" * 80 + "\n")
            f.write("GRPO REASONING & ANSWER LANGUAGE EVALUATION\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        f.write(f"Test #{result['test_num']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Language: {result['language']}\n")
        f.write(f"Question: {result['prompt']}\n")
        f.write(f"Expected Language: {result['reasoning_language_arg']}\n")
        f.write(f"Detected Reasoning Language: {result['detected_reasoning_lang']} ({'OK' if result['reasoning_correct'] else 'WRONG'})\n")
        f.write(f"Detected Answer Language: {result['detected_answer_lang']} ({'OK' if result['answer_correct'] else 'WRONG'})\n\n")
        
        f.write("Reasoning:\n")
        f.write(result['reasoning'][:1500] + ("..." if len(result['reasoning']) > 1500 else "") + "\n\n")
        
        f.write("Final Answer:\n")
        f.write(result['final_answer'] + "\n\n")
        
        f.write("=" * 80 + "\n\n")


def write_summary(stats: dict, output_file: str):
    """Append summary to the output file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Tests: {stats['total']}\n")
        f.write(f"Reasoning Language Correct: {stats['reasoning_correct']} ({stats['reasoning_accuracy']:.1f}%)\n")
        f.write(f"Answer Language Correct: {stats['answer_correct']} ({stats['answer_accuracy']:.1f}%)\n")
        f.write(f"Both Correct: {stats['both_correct']} ({stats['both_accuracy']:.1f}%)\n\n")
        
        f.write("By Language:\n")
        for lang, lang_stats in stats["by_language"].items():
            f.write(f"  {lang}:\n")
            f.write(f"    Reasoning: {lang_stats['reasoning_correct']}/{lang_stats['total']} ({lang_stats['reasoning_accuracy']:.1f}%)\n")
            f.write(f"    Answer: {lang_stats['answer_correct']}/{lang_stats['total']} ({lang_stats['answer_accuracy']:.1f}%)\n")


def load_grpo_model(base_model: str, sft_checkpoint: str, grpo_checkpoint: str):
    """
    Load model with GRPO checkpoint.
    
    Chain: Base model → Merge SFT LoRA → Load GRPO LoRA
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    
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


def generate_response_grpo(model, tokenizer, prompt: str, reasoning_language: str = "English", max_new_tokens: int = 1024):
    """Generate response using GRPO model."""
    from transformers import GenerationConfig
    
    # Build chat messages
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
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
    
    return response


def run_evaluation(model, tokenizer, output_file: str, use_grpo_generate: bool = False):
    """Run the full evaluation."""
    
    results = []
    
    print(f"\n{'='*70}")
    print("REASONING LANGUAGE EVALUATION v2")
    print("Question language = Reasoning language = Answer language")
    print(f"{'='*70}")
    print(f"Prompts per language: {len(TEST_PROMPTS['English'])}")
    print(f"Languages: {SUPPORTED_LANGUAGES}")
    print(f"Total tests: {len(TEST_PROMPTS['English']) * len(SUPPORTED_LANGUAGES)}")
    print(f"Output file: {output_file}")
    print(f"{'='*70}\n")
    
    # Use appropriate generate function
    if use_grpo_generate:
        generate_fn = lambda m, t, p, lang: generate_response_grpo(m, t, p, reasoning_language=lang)
    else:
        from inference_g6e import generate_response
        generate_fn = lambda m, t, p, lang: generate_response(m, t, p, max_new_tokens=1024, reasoning_language=lang)
    
    test_num = 0
    is_first = True
    
    for lang in SUPPORTED_LANGUAGES:
        prompts = TEST_PROMPTS[lang]
        
        for i, prompt in enumerate(prompts):
            test_num += 1
            print(f"[{test_num}/50] {lang}: '{prompt[:50]}...'")
            
            # Generate response with reasoning language = question language
            try:
                response = generate_fn(model, tokenizer, prompt, lang)
            except Exception as e:
                response = f"[Generation error: {str(e)}]"
            
            # Extract parts
            reasoning = extract_reasoning(response)
            final_answer = extract_final_answer(response)
            
            # Detect language of reasoning
            detected_lang = detect_language(reasoning)
            
            # Check if correct (reasoning should be in same language as question)
            is_correct = check_language_match(lang, detected_lang)
            
            # Translate reasoning to English
            if detected_lang != "en" and detected_lang not in ["too_short", "error", "unknown"]:
                reasoning_english = translate_to_english(reasoning, detected_lang)
            elif detected_lang == "en":
                reasoning_english = reasoning
            else:
                reasoning_english = "[Could not translate]"
            
            result = {
                "test_num": test_num,
                "language": lang,
                "prompt": prompt,
                "reasoning_language_arg": lang,
                "full_response": response,
                "reasoning": reasoning,
                "final_answer": final_answer,
                "detected_reasoning_lang": detected_lang,
                "detected_answer_lang": detect_language(final_answer),
                "reasoning_english": reasoning_english,
                "reasoning_correct": is_correct,
                "answer_correct": check_language_match(lang, detect_language(final_answer)),
            }
            results.append(result)
            
            # Save immediately after each question
            append_result_to_file(result, output_file, is_first=is_first)
            is_first = False
            
            reasoning_status = "✓" if result['reasoning_correct'] else "✗"
            answer_status = "✓" if result['answer_correct'] else "✗"
            expected_code = LANG_CODE_MAP.get(lang)
            print(f"    Reasoning: {detected_lang} {reasoning_status} | Answer: {result['detected_answer_lang']} {answer_status} | Expected: {expected_code}")
            print(f"    Saved to: {output_file}")
    
    # Calculate and save summary
    stats = calculate_statistics(results)
    write_summary(stats, output_file)
    print_summary(stats)
    
    return results, stats


def calculate_statistics(results):
    """Calculate evaluation statistics."""
    stats = {
        "total": len(results),
        "reasoning_correct": sum(1 for r in results if r["reasoning_correct"]),
        "answer_correct": sum(1 for r in results if r["answer_correct"]),
        "both_correct": sum(1 for r in results if r["reasoning_correct"] and r["answer_correct"]),
        "by_language": {},
    }
    
    for lang in SUPPORTED_LANGUAGES:
        lang_results = [r for r in results if r["language"] == lang]
        reasoning_correct = sum(1 for r in lang_results if r["reasoning_correct"])
        answer_correct = sum(1 for r in lang_results if r["answer_correct"])
        both_correct = sum(1 for r in lang_results if r["reasoning_correct"] and r["answer_correct"])
        stats["by_language"][lang] = {
            "total": len(lang_results),
            "reasoning_correct": reasoning_correct,
            "answer_correct": answer_correct,
            "both_correct": both_correct,
            "reasoning_accuracy": reasoning_correct / len(lang_results) * 100 if lang_results else 0,
            "answer_accuracy": answer_correct / len(lang_results) * 100 if lang_results else 0,
        }
    
    stats["reasoning_accuracy"] = stats["reasoning_correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
    stats["answer_accuracy"] = stats["answer_correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
    stats["both_accuracy"] = stats["both_correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
    
    return stats


def print_summary(stats):
    """Print evaluation summary."""
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {stats['total']}")
    print(f"Reasoning Language Correct: {stats['reasoning_correct']} ({stats['reasoning_accuracy']:.1f}%)")
    print(f"Answer Language Correct: {stats['answer_correct']} ({stats['answer_accuracy']:.1f}%)")
    print(f"Both Correct: {stats['both_correct']} ({stats['both_accuracy']:.1f}%)")
    print(f"\nBy Language:")
    for lang, lang_stats in stats["by_language"].items():
        r_bar = "█" * int(lang_stats['reasoning_accuracy'] / 10) + "░" * (10 - int(lang_stats['reasoning_accuracy'] / 10))
        a_bar = "█" * int(lang_stats['answer_accuracy'] / 10) + "░" * (10 - int(lang_stats['answer_accuracy'] / 10))
        print(f"  {lang:10} Reasoning: {r_bar} {lang_stats['reasoning_accuracy']:.0f}% | Answer: {a_bar} {lang_stats['answer_accuracy']:.0f}%")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning language compliance")
    parser.add_argument("--base_model", type=str, default="openai/gpt-oss-20b", help="Base model path")
    parser.add_argument("--use_trained", action="store_true", help="Use trained model with LoRA (SFT only)")
    parser.add_argument("--checkpoint_dir", type=str, default="/fsx/checkpoints", help="Checkpoint directory")
    parser.add_argument("--grpo_checkpoint", type=str, default=None, 
                        help="GRPO checkpoint path (e.g., /fsx/checkpoints/grpo-singlenode/checkpoint-366)")
    parser.add_argument("--sft_checkpoint", type=str, default="/fsx/checkpoints/converted-peft/lora-checkpoint-1000-peft",
                        help="SFT checkpoint path (used with --grpo_checkpoint)")
    parser.add_argument("--output", type=str, default="/fsx/experiments/grpo_evaluation.txt", help="Output file")
    
    args = parser.parse_args()
    
    # Install dependencies if needed
    if not HAS_LANGDETECT:
        print("ERROR: langdetect not installed. Please rebuild the container image.")
        print("  pip install langdetect")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model based on mode
    if args.grpo_checkpoint:
        # GRPO mode: Base → SFT merged → GRPO LoRA
        print("=" * 60)
        print("GRPO EVALUATION MODE")
        print("=" * 60)
        model, tokenizer = load_grpo_model(
            args.base_model,
            args.sft_checkpoint,
            args.grpo_checkpoint
        )
        use_grpo_generate = True
    else:
        # Standard mode: use inference_g6e
        from inference_g6e import load_model
        print(f"Loading model (use_trained={args.use_trained})...")
        checkpoint_path = args.sft_checkpoint if args.use_trained else None
        model, tokenizer = load_model(args.base_model, checkpoint=checkpoint_path)
        use_grpo_generate = False
    
    # Run evaluation
    results, stats = run_evaluation(model, tokenizer, args.output, use_grpo_generate=use_grpo_generate)
    
    print(f"\nEvaluation complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
