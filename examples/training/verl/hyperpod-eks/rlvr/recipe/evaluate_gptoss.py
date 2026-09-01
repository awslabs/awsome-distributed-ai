# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Evaluate veRL GRPO checkpoint for language compliance.

This script:
  1. Optionally converts FSDP shards → HuggingFace format (via verl.model_merger)
  2. Launches vLLM serving on the merged HF model (20B needs TP across GPUs)
  3. Runs 50 test cases (10 prompts × 5 languages) checking reasoning/answer language
  4. Prints per-language accuracy and saves results to a file

Usage (on a Ray worker pod with 4 GPUs):

    # Step 1: Convert checkpoint (only needed once per checkpoint)
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir /fsx/verl/ckpts/GRPO-GPT-OSS/GRPO-gpt-oss-20b-language/global_step_80/actor \
        --target_dir /fsx/verl/merged_model/gpt-oss-20b-grpo-step80

    # Step 2: Run evaluation
    python evaluate_gptoss.py \
        --model_path /fsx/verl/merged_model/gpt-oss-20b-grpo-step80 \
        --tp 4 \
        --output /fsx/experiments/grpo_eval_step80.txt

    # Or compare against the SFT baseline:
    python evaluate_gptoss.py \
        --model_path /fsx/verl/models/openai/gpt-oss-20b \
        --tp 4 \
        --output /fsx/experiments/sft_baseline_eval.txt

Requires: vllm, langdetect (both in the verl-rlvr Docker image)
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# ---------------------------------------------------------------------------
# Test Prompts — 10 questions × 5 languages = 50 test cases
# ---------------------------------------------------------------------------
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

SUPPORTED_LANGUAGES = ["English", "French", "German", "Spanish", "Italian"]
LANG_CODE_MAP = {"English": "en", "French": "fr", "German": "de", "Spanish": "es", "Italian": "it"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Detect language of text using langdetect."""
    if not HAS_LANGDETECT:
        return "unknown"
    try:
        clean = re.sub(r"[0-9+\-*/=%$€£\\{}[\]]", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) < 20:
            return "too_short"
        return detect(clean)
    except Exception:
        return "error"


def extract_reasoning(response: str) -> str:
    """Extract reasoning/analysis section from response."""
    m = re.search(r"assistantanalysis\s*(.*?)\s*assistantfinal", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"<analysis>(.*?)</analysis>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return response[:500] if len(response) > 500 else response


def extract_final_answer(response: str) -> str:
    """Extract final answer section from response."""
    m = re.search(r"assistantfinal\s*(.*?)(?:$|assistant)", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"<final>(.*?)</final>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return response[-200:] if len(response) > 200 else response


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------

def create_vllm_engine(model_path: str, tp: int, gpu_mem: float, max_model_len: int):
    """Create a vLLM LLM engine for batch inference."""
    from vllm import LLM, SamplingParams

    print(f"Loading model with vLLM: {model_path}")
    print(f"  tensor_parallel_size={tp}, gpu_memory_utilization={gpu_mem}")
    print(f"  max_model_len={max_model_len}, enforce_eager=True")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        enforce_eager=True,            # avoid CUDA graph OOM on A10G
        trust_remote_code=True,        # GPT-OSS uses custom code
        dtype="bfloat16",
    )
    return llm


def generate_batch(llm, prompts: list[dict], max_tokens: int = 512) -> list[str]:
    """Generate responses for a batch of prompts using vLLM.

    Each item in `prompts` is {"text": str, "language": str}.
    Returns list of generated text strings.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.95,
    )

    # Build chat-formatted prompts
    formatted = []
    for p in prompts:
        # Match the system prompt format used during training
        messages = [
            {"role": "system", "content": f"reasoning language: {p['language']}\nanswer language: {p['language']}"},
            {"role": "user", "content": p["text"]},
        ]
        # Use tokenizer's chat template via vLLM
        formatted.append(messages)

    # vLLM chat() method applies the chat template
    outputs = llm.chat(formatted, sampling_params=sampling_params)

    responses = []
    for output in outputs:
        text = output.outputs[0].text
        responses.append(text)
    return responses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(llm, output_file: str, max_tokens: int = 512):
    """Run the 50-question evaluation."""
    results = []
    total = len(SUPPORTED_LANGUAGES) * len(TEST_PROMPTS["English"])

    print(f"\n{'='*70}")
    print("VERL GRPO - LANGUAGE COMPLIANCE EVALUATION")
    print(f"{'='*70}")
    print(f"Languages: {SUPPORTED_LANGUAGES}")
    print(f"Prompts per language: {len(TEST_PROMPTS['English'])}")
    print(f"Total test cases: {total}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")

    # Collect all prompts for batch inference
    all_prompts = []
    prompt_metadata = []
    for lang in SUPPORTED_LANGUAGES:
        for i, text in enumerate(TEST_PROMPTS[lang]):
            all_prompts.append({"text": text, "language": lang})
            prompt_metadata.append({"language": lang, "index": i, "text": text})

    # Batch generate
    print(f"Generating {len(all_prompts)} responses with vLLM...")
    responses = generate_batch(llm, all_prompts, max_tokens=max_tokens)
    print(f"Generation complete.\n")

    # Score each response
    for idx, (meta, response) in enumerate(zip(prompt_metadata, responses)):
        lang = meta["language"]
        expected_code = LANG_CODE_MAP[lang]

        reasoning = extract_reasoning(response)
        final_answer = extract_final_answer(response)
        reasoning_lang = detect_language(reasoning)
        answer_lang = detect_language(final_answer)

        reasoning_correct = reasoning_lang == expected_code
        answer_correct = answer_lang == expected_code

        result = {
            "test_num": idx + 1,
            "language": lang,
            "prompt": meta["text"],
            "expected_code": expected_code,
            "response": response,
            "reasoning": reasoning[:500],
            "final_answer": final_answer[:500],
            "reasoning_lang": reasoning_lang,
            "answer_lang": answer_lang,
            "reasoning_correct": reasoning_correct,
            "answer_correct": answer_correct,
        }
        results.append(result)

        r_mark = "OK" if reasoning_correct else "WRONG"
        a_mark = "OK" if answer_correct else "WRONG"
        print(f"[{idx+1:2d}/{total}] {lang:8s} | reasoning={reasoning_lang} ({r_mark}) | answer={answer_lang} ({a_mark}) | {meta['text'][:40]}...")

    # Calculate statistics
    stats = _calculate_stats(results)
    _print_summary(stats)
    _save_results(results, stats, output_file)

    return results, stats


def _calculate_stats(results):
    total = len(results)
    stats = {
        "total": total,
        "reasoning_correct": sum(1 for r in results if r["reasoning_correct"]),
        "answer_correct": sum(1 for r in results if r["answer_correct"]),
        "both_correct": sum(1 for r in results if r["reasoning_correct"] and r["answer_correct"]),
        "by_language": {},
    }
    for lang in SUPPORTED_LANGUAGES:
        lr = [r for r in results if r["language"] == lang]
        rc = sum(1 for r in lr if r["reasoning_correct"])
        ac = sum(1 for r in lr if r["answer_correct"])
        bc = sum(1 for r in lr if r["reasoning_correct"] and r["answer_correct"])
        n = len(lr)
        stats["by_language"][lang] = {
            "total": n,
            "reasoning_correct": rc, "reasoning_accuracy": rc / n * 100 if n else 0,
            "answer_correct": ac, "answer_accuracy": ac / n * 100 if n else 0,
            "both_correct": bc, "both_accuracy": bc / n * 100 if n else 0,
        }
    stats["reasoning_accuracy"] = stats["reasoning_correct"] / total * 100 if total else 0
    stats["answer_accuracy"] = stats["answer_correct"] / total * 100 if total else 0
    stats["both_accuracy"] = stats["both_correct"] / total * 100 if total else 0
    return stats


def _print_summary(stats):
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {stats['total']}")
    print(f"Reasoning correct: {stats['reasoning_correct']}/{stats['total']} ({stats['reasoning_accuracy']:.1f}%)")
    print(f"Answer correct:    {stats['answer_correct']}/{stats['total']} ({stats['answer_accuracy']:.1f}%)")
    print(f"Both correct:      {stats['both_correct']}/{stats['total']} ({stats['both_accuracy']:.1f}%)")
    print(f"\nPer-language breakdown:")
    for lang, ls in stats["by_language"].items():
        r_bar = "#" * int(ls["reasoning_accuracy"] / 10) + "." * (10 - int(ls["reasoning_accuracy"] / 10))
        a_bar = "#" * int(ls["answer_accuracy"] / 10) + "." * (10 - int(ls["answer_accuracy"] / 10))
        print(f"  {lang:10s} Reasoning: [{r_bar}] {ls['reasoning_accuracy']:5.1f}% | Answer: [{a_bar}] {ls['answer_accuracy']:5.1f}%")
    print(f"{'='*70}\n")


def _save_results(results, stats, output_file):
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("VERL GRPO — LANGUAGE COMPLIANCE EVALUATION\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"Test #{r['test_num']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Language: {r['language']}  Expected: {r['expected_code']}\n")
            f.write(f"Question: {r['prompt']}\n")
            f.write(f"Reasoning lang: {r['reasoning_lang']} ({'OK' if r['reasoning_correct'] else 'WRONG'})\n")
            f.write(f"Answer lang:    {r['answer_lang']} ({'OK' if r['answer_correct'] else 'WRONG'})\n\n")
            f.write(f"Reasoning:\n{r['reasoning']}\n\n")
            f.write(f"Final Answer:\n{r['final_answer']}\n\n")
            f.write("=" * 80 + "\n\n")

        # Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total: {stats['total']}\n")
        f.write(f"Reasoning correct: {stats['reasoning_correct']} ({stats['reasoning_accuracy']:.1f}%)\n")
        f.write(f"Answer correct:    {stats['answer_correct']} ({stats['answer_accuracy']:.1f}%)\n")
        f.write(f"Both correct:      {stats['both_correct']} ({stats['both_accuracy']:.1f}%)\n\n")
        for lang, ls in stats["by_language"].items():
            f.write(f"  {lang}: reasoning={ls['reasoning_correct']}/{ls['total']} ({ls['reasoning_accuracy']:.1f}%), "
                    f"answer={ls['answer_correct']}/{ls['total']} ({ls['answer_accuracy']:.1f}%)\n")

    # Also save machine-readable JSON
    json_file = output_file.replace(".txt", ".json")
    with open(json_file, "w") as f:
        json.dump({"stats": stats, "results": results}, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")
    print(f"JSON saved to:    {json_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate veRL GRPO checkpoint for language compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert FSDP checkpoint first (run once):
  python -m verl.model_merger merge \\
      --backend fsdp \\
      --local_dir /fsx/verl/ckpts/.../global_step_80/actor \\
      --target_dir /fsx/verl/merged_model/grpo-step80

  # Evaluate the GRPO model:
  python evaluate_gptoss.py \\
      --model_path /fsx/verl/merged_model/grpo-step80 \\
      --tp 4

  # Compare against SFT baseline:
  python evaluate_gptoss.py \\
      --model_path /fsx/verl/models/openai/gpt-oss-20b \\
      --tp 4 --output /fsx/experiments/sft_baseline.txt
""",
    )
    parser.add_argument("--model_path", required=True, help="Path to HF-format model (merged checkpoint or base)")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size (default: 4 for g5.12xlarge)")
    parser.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM GPU memory utilization (default: 0.85)")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Max model sequence length (default: 2048)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate (default: 512)")
    parser.add_argument("--output", type=str, default="/fsx/experiments/grpo_eval.txt", help="Output file path")
    args = parser.parse_args()

    if not HAS_LANGDETECT:
        print("ERROR: langdetect not installed. pip install langdetect")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path does not exist: {args.model_path}")
        print("Did you run the checkpoint merge step first?")
        print(f"  python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor --target_dir {args.model_path}")
        sys.exit(1)

    llm = create_vllm_engine(args.model_path, args.tp, args.gpu_mem, args.max_model_len)
    run_evaluation(llm, args.output, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
