#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# =============================================================================
# Unified Data Preparation for GRPO Training
# Prepares 4 coding/reasoning datasets in parallel using Ray tasks.
# Replaces the individual bash scripts (prepare_*_data.sh).
#
# Usage:
#   # Prepare all datasets in parallel via Ray
#   ray job submit --address $RAY_ADDRESS --working-dir . \
#     -- python data/prepare_data.py --datasets eurus apps taco codecontests \
#        --output-dir /fsx/data/verl/data
#
#   # Prepare + mix in one shot
#   ray job submit --address $RAY_ADDRESS --working-dir . \
#     -- python data/prepare_data.py --datasets eurus apps taco codecontests \
#        --output-dir /fsx/data/verl/data \
#        --mix --mix-output /fsx/data/verl/data/mixed \
#        --mix-ratios eurus:1.0 apps:0.5 taco:0.3 codecontests:1.0
#
#   # Local (no Ray) — for testing
#   python data/prepare_data.py --datasets eurus --output-dir /tmp/test-data --local
# =============================================================================

import argparse
import json
import os
import sys
import time

# Some datasets (e.g., APPS) contain very large integers in test cases
# that exceed Python 3.10+'s default string conversion limit
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

import datasets
import numpy as np

# =============================================================================
# Dataset Registry
# =============================================================================

# trust_remote_code EXECUTES Python fetched from the Hub at load time. It is set only
# for the three datasets that shipped a legacy loading script, and it is a no-op on the
# `datasets` versions pinned in data/submit_data_prep.sh (>=3.0), which removed loading
# scripts entirely -- that removal is exactly what the "scripts are no longer supported"
# fallback in load_hf_dataset() handles. The flags are kept only so an older pinned
# `datasets` still loads these repos; if you drop support for that, drop them too.
DATASET_REGISTRY = {
    "eurus": {
        "hf_name": "PRIME-RL/Eurus-2-RL-Data",
        # Ships a loading script; see the note above the registry.
        "trust_remote_code": True,
        "transform_fn": "transform_eurus",
        "description": "Reasoning + code RL data from PRIME-RL",
    },
    "apps": {
        "hf_name": "codeparrot/apps",
        # Ships a loading script; see the note above the registry.
        "trust_remote_code": True,
        "transform_fn": "transform_apps",
        "description": "APPS coding problems",
    },
    "taco": {
        "hf_name": "BAAI/TACO",
        # Ships a loading script; see the note above the registry.
        "trust_remote_code": True,
        "transform_fn": "transform_taco",
        "description": "TACO coding problems from BAAI",
    },
    "codecontests": {
        "hf_name": "ByteDance-Seed/Code-Contests-Plus",
        # Parquet-native, no loading script -- so no remote code execution needed.
        "trust_remote_code": False,
        "transform_fn": "transform_codecontests",
        "description": "Competitive programming with augmented test cases",
        "subset": "1x",
    },
}


# =============================================================================
# Transform Functions
# (Extracted from existing bash scripts — logic preserved exactly)
# =============================================================================


def transform_eurus(dataset_dict, output_dir, hf_name):
    """Transform PRIME-RL/Eurus-2-RL-Data to verl format."""
    output_name_map = {
        "validation": "val",
        "val": "val",
        "dev": "val",
        "test": "test",
        "train": "train",
    }

    results = {}
    for split_name in ["train", "test", "validation"]:
        # Find the actual split name in the dataset
        actual_split = None
        candidates = {
            "train": [],
            "test": ["eval", "evaluation"],
            "validation": ["val", "dev"],
        }
        for candidate in [split_name] + candidates.get(split_name, []):
            if candidate in dataset_dict:
                actual_split = candidate
                break
        if actual_split is None:
            print(f"  Split '{split_name}' not found, skipping...")
            continue

        print(f"  Processing {actual_split} split...")
        split_data = dataset_dict[actual_split]

        if not isinstance(split_data, datasets.Dataset):
            split_data = datasets.Dataset.from_list(list(split_data))

        def make_map_fn(split):
            def process_fn(example, idx):
                prompt = example.get("prompt", None)

                if prompt is None:
                    for key in ["question", "query", "input", "instruction", "problem"]:
                        if key in example and example[key]:
                            prompt = example[key]
                            break
                if prompt is None:
                    prompt = ""

                if isinstance(prompt, str):
                    prompt = [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list):
                    if (
                        len(prompt) > 0
                        and isinstance(prompt[0], dict)
                        and "role" in prompt[0]
                    ):
                        # Eurus data may have nested message structure where the
                        # outer list has a single element whose 'content' is itself
                        # a list of messages (system + user).  Flatten this so verl
                        # sees a standard conversation list.
                        # Nested: [{"role":"user", "content": [{"role":"system",...}, {"role":"user",...}]}]
                        # Flat:   [{"role":"system", "content":"..."}, {"role":"user", "content":"..."}]
                        if (
                            len(prompt) == 1
                            and isinstance(prompt[0].get("content"), (list, np.ndarray))
                            and len(prompt[0]["content"]) > 0
                            and isinstance(prompt[0]["content"][0], dict)
                            and "role" in prompt[0]["content"][0]
                        ):
                            prompt = [
                                {"role": msg["role"], "content": str(msg["content"])}
                                for msg in prompt[0]["content"]
                            ]
                        else:
                            # Ensure each message content is a plain string
                            prompt = [
                                {
                                    "role": msg.get("role", "user"),
                                    "content": str(msg.get("content", "")),
                                }
                                for msg in prompt
                            ]
                    else:
                        prompt = [{"role": "user", "content": str(prompt)}]

                data_source = example.get("data_source", hf_name)

                reward_model = example.get("reward_model", None)
                if reward_model is None:
                    ground_truth = ""
                    for key in [
                        "ground_truth",
                        "expected_answer",
                        "answer",
                        "solution",
                        "label",
                    ]:
                        if key in example and example[key]:
                            ground_truth = example[key]
                            break
                    reward_model = {"style": "rule", "ground_truth": ground_truth}
                elif (
                    isinstance(reward_model, dict)
                    and "ground_truth" not in reward_model
                ):
                    reward_model["ground_truth"] = ""

                extra_info = example.get("extra_info", {})
                if not isinstance(extra_info, dict):
                    extra_info = {}
                extra_info["split"] = split
                extra_info["index"] = idx

                ability = example.get("ability", "reasoning")

                return {
                    "data_source": data_source,
                    "prompt": prompt,
                    "ability": ability,
                    "reward_model": reward_model,
                    "extra_info": extra_info,
                }

            return process_fn

        split_data = split_data.map(
            function=make_map_fn(actual_split),
            with_indices=True,
            remove_columns=split_data.column_names,
        )

        output_name = output_name_map.get(actual_split, actual_split)
        output_file = os.path.join(output_dir, f"{output_name}.parquet")
        split_data.to_parquet(output_file)
        print(f"  Saved {len(split_data)} samples to {output_file}")
        results[output_name] = len(split_data)

        # Print sample
        sample = split_data[0]
        if isinstance(sample["prompt"], list) and len(sample["prompt"]) > 0:
            content = sample["prompt"][-1].get("content", "")
            print(f"  Sample prompt: {content[:200]}...")
        print(f"  data_source: {sample['data_source']}, ability: {sample['ability']}")

    return results


def transform_apps(dataset_dict, output_dir, hf_name):
    """Transform codeparrot/apps to verl format."""
    results = {}

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.get("question", "")

            input_output_raw = example.get("input_output", "")
            try:
                if isinstance(input_output_raw, str) and input_output_raw.strip():
                    input_output = json.loads(input_output_raw)
                elif isinstance(input_output_raw, dict):
                    input_output = input_output_raw
                else:
                    input_output = {}
            except (json.JSONDecodeError, TypeError):
                input_output = {}

            solutions_raw = example.get("solutions", "")
            try:
                if isinstance(solutions_raw, str) and solutions_raw.strip():
                    solutions = json.loads(solutions_raw)
                elif isinstance(solutions_raw, list):
                    solutions = solutions_raw
                else:
                    solutions = []
            except (json.JSONDecodeError, TypeError):
                solutions = []

            starter_code = example.get("starter_code", "")
            prompt_text = question
            if starter_code and starter_code.strip():
                prompt_text += (
                    f"\n\nStarter code:\n```python\n{starter_code.strip()}\n```"
                )

            # Serialize ground_truth as JSON string for consistent Arrow typing
            ground_truth_str = json.dumps(input_output) if input_output else ""

            return {
                "data_source": "apps",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth_str,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "difficulty": example.get("difficulty", ""),
                    "url": example.get("url", ""),
                    "solutions": json.dumps(solutions),
                    "starter_code": starter_code,
                    "problem_id": str(example.get("problem_id", idx)),
                },
            }

        return process_fn

    for split_name in ["train", "test"]:
        if split_name not in dataset_dict:
            print(f"  Split '{split_name}' not found, skipping...")
            continue

        print(f"  Processing {split_name} split...")
        split_data = dataset_dict[split_name]

        split_data = split_data.map(
            function=make_map_fn(split_name),
            with_indices=True,
            remove_columns=split_data.column_names,
        )

        # Filter out samples without test cases
        before_count = len(split_data)
        split_data = split_data.filter(
            lambda x: (
                bool(x["reward_model"].get("ground_truth"))
                and "inputs" in x["reward_model"].get("ground_truth", "")
            )
        )
        after_count = len(split_data)
        if before_count != after_count:
            print(f"  Filtered {before_count - after_count} samples without test cases")

        # APPS has no validation split; use test as val
        output_name = "val" if split_name == "test" else split_name
        output_file = os.path.join(output_dir, f"{output_name}.parquet")
        split_data.to_parquet(output_file)
        print(f"  Saved {len(split_data)} samples to {output_file}")
        results[output_name] = len(split_data)

        sample = split_data[0]
        content = sample["prompt"][0]["content"]
        print(f"  Sample prompt: {content[:200]}...")
        gt_str = sample["reward_model"]["ground_truth"]
        try:
            gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
            n_tests = len(gt.get("inputs", [])) if isinstance(gt, dict) else 0
        except (json.JSONDecodeError, TypeError):
            n_tests = 0
        print(
            f"  difficulty: {sample['extra_info']['difficulty']}, test cases: {n_tests}"
        )

    return results


def _safe_json_parse(raw, fallback=None):
    """Safely parse a JSON string, returning fallback on failure."""
    if fallback is None:
        fallback = {}
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return fallback


def _safe_eval_parse(raw, fallback=None):
    """Safely parse a Python literal string (for tags/skills stored as repr)."""
    if fallback is None:
        fallback = []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            import ast

            return ast.literal_eval(raw)
        except Exception:
            pass
    return fallback


def transform_taco(dataset_dict, output_dir, hf_name):
    """Transform BAAI/TACO to verl format."""
    results = {}

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.get("question", "")

            input_output = _safe_json_parse(example.get("input_output", ""))
            solutions = _safe_json_parse(example.get("solutions", ""), fallback=[])

            tags = _safe_eval_parse(example.get("tags", ""))
            skill_types = _safe_eval_parse(example.get("skill_types", ""))
            raw_tags = _safe_eval_parse(example.get("raw_tags", ""))

            starter_code = example.get("starter_code", "")
            prompt_text = question
            if starter_code and starter_code.strip():
                prompt_text += (
                    f"\n\nStarter code:\n```python\n{starter_code.strip()}\n```"
                )

            # Serialize ground_truth as JSON string for consistent Arrow typing
            ground_truth_str = json.dumps(input_output) if input_output else ""

            return {
                "data_source": "taco",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth_str,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "difficulty": example.get("difficulty", ""),
                    "source": example.get("source", ""),
                    "url": example.get("url", ""),
                    "tags": json.dumps(tags),
                    "skill_types": json.dumps(skill_types),
                    "raw_tags": json.dumps(raw_tags),
                    "solutions": json.dumps(solutions),
                    "time_limit": str(example.get("time_limit", "")),
                    "memory_limit": str(example.get("memory_limit", "")),
                },
            }

        return process_fn

    for split_name in ["train", "test"]:
        if split_name not in dataset_dict:
            print(f"  Split '{split_name}' not found, skipping...")
            continue

        print(f"  Processing {split_name} split...")
        split_data = dataset_dict[split_name]

        split_data = split_data.map(
            function=make_map_fn(split_name),
            with_indices=True,
            remove_columns=split_data.column_names,
        )

        # Filter out samples without test cases
        before_count = len(split_data)
        split_data = split_data.filter(
            lambda x: (
                bool(x["reward_model"].get("ground_truth"))
                and "inputs" in x["reward_model"].get("ground_truth", "")
            )
        )
        after_count = len(split_data)
        if before_count != after_count:
            print(f"  Filtered {before_count - after_count} samples without test cases")

        # TACO has no validation split; use test as val
        output_name = "val" if split_name == "test" else split_name
        output_file = os.path.join(output_dir, f"{output_name}.parquet")
        split_data.to_parquet(output_file)
        print(f"  Saved {len(split_data)} samples to {output_file}")
        results[output_name] = len(split_data)

        sample = split_data[0]
        content = sample["prompt"][0]["content"]
        print(f"  Sample prompt: {content[:200]}...")
        gt_str = sample["reward_model"]["ground_truth"]
        try:
            gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
            n_tests = len(gt.get("inputs", [])) if isinstance(gt, dict) else 0
        except (json.JSONDecodeError, TypeError):
            n_tests = 0
        print(
            f"  difficulty: {sample['extra_info']['difficulty']}, test cases: {n_tests}"
        )

    return results


def transform_codecontests(dataset_dict, output_dir, hf_name, val_ratio=0.1):
    """Transform ByteDance-Seed/Code-Contests-Plus to verl format."""
    results = {}

    def make_map_fn(split):
        def process_fn(example, idx):
            description = example.get("description", "")

            all_inputs = []
            all_outputs = []

            # Code-Contests-Plus uses 'test_cases' as a list of {input, output} dicts
            test_cases = example.get("test_cases", [])
            if isinstance(test_cases, list):
                for tc in test_cases:
                    if isinstance(tc, dict):
                        all_inputs.append(tc.get("input", ""))
                        all_outputs.append(tc.get("output", ""))

            # Fallback for original CodeContests format
            if not all_inputs:
                for test_field in ["public_tests", "private_tests", "generated_tests"]:
                    tests = example.get(test_field, None)
                    if tests and isinstance(tests, dict):
                        inputs = tests.get("input", tests.get("inputs", []))
                        outputs = tests.get("output", tests.get("outputs", []))
                        if isinstance(inputs, list) and isinstance(outputs, list):
                            all_inputs.extend(inputs)
                            all_outputs.extend(outputs)

            # all_inputs / all_outputs are .extend()-ed from SEPARATE upstream fields, so a
            # malformed record can leave them different lengths. Left alone that misaligns every
            # input/output pair in the reward ground truth (and the zip() below would quietly
            # truncate to the shorter one). Normalise it LOUDLY instead of silently.
            if len(all_inputs) != len(all_outputs):
                n_common = min(len(all_inputs), len(all_outputs))
                print(
                    f"  WARNING: codecontests/{split} test-case length mismatch "
                    f"(inputs={len(all_inputs)}, outputs={len(all_outputs)}) -- "
                    f"truncating both to {n_common} to keep ground truth aligned"
                )
                all_inputs = all_inputs[:n_common]
                all_outputs = all_outputs[:n_common]

            ground_truth = {"inputs": all_inputs, "outputs": all_outputs}
            ground_truth_str = json.dumps(ground_truth)
            # Cap ground_truth size to avoid Arrow offset overflow
            # Some problems have extremely large generated test cases (>20MB)
            if len(ground_truth_str) > 1_000_000:
                orig_bytes = len(ground_truth_str)
                # Keep only first N tests that fit within limit
                trimmed_inputs, trimmed_outputs = [], []
                total_size = 0
                # strict=True is safe: the lengths were equalised above.
                for inp, out in zip(all_inputs, all_outputs, strict=True):
                    entry_size = len(str(inp)) + len(str(out))
                    if total_size + entry_size > 500_000:
                        break
                    trimmed_inputs.append(inp)
                    trimmed_outputs.append(out)
                    total_size += entry_size
                # Dropping test cases weakens the reward signal for this problem, so say so.
                print(
                    f"  NOTE: codecontests/{split} ground_truth {orig_bytes} B exceeds the 1 MB "
                    f"Arrow cap -- kept {len(trimmed_inputs)}/{len(all_inputs)} test cases"
                )
                ground_truth_str = json.dumps(
                    {"inputs": trimmed_inputs, "outputs": trimmed_outputs}
                )

            correct_subs = example.get("correct_submissions", [])
            n_correct = len(correct_subs) if isinstance(correct_subs, list) else 0

            return {
                "data_source": "codecontests",
                "prompt": [{"role": "user", "content": description}],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth_str,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "title": example.get("title", ""),
                    "source": str(example.get("source", "")),
                    "problem_id": str(example.get("id", "")),
                    "time_limit": str(example.get("time_limit", 0)),
                    "memory_limit": str(example.get("memory_limit", 0)),
                    "n_correct_submissions": str(n_correct),
                },
            }

        return process_fn

    if "train" not in dataset_dict:
        print("  ERROR: 'train' split not found in dataset")
        return results

    print("  Processing train split...")
    full_data = dataset_dict["train"]

    full_data = full_data.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=full_data.column_names,
        writer_batch_size=500,
    )

    # Filter out samples without test cases
    before_count = len(full_data)
    full_data = full_data.filter(
        lambda x: (
            bool(x["reward_model"].get("ground_truth"))
            and "inputs" in x["reward_model"].get("ground_truth", "")
            and x["reward_model"]["ground_truth"] != '{"inputs": [], "outputs": []}'
        )
    )
    after_count = len(full_data)
    if before_count != after_count:
        print(f"  Filtered {before_count - after_count} samples without test cases")

    # Split into train and val
    full_data = full_data.shuffle(seed=42)
    val_size = max(1, int(len(full_data) * val_ratio))
    train_size = len(full_data) - val_size

    train_data = full_data.select(range(train_size))
    val_data = full_data.select(range(train_size, len(full_data)))

    def update_split(example):
        example["extra_info"]["split"] = "val"
        return example

    val_data = val_data.map(update_split)

    train_file = os.path.join(output_dir, "train.parquet")
    val_file = os.path.join(output_dir, "val.parquet")
    train_data.to_parquet(train_file)
    val_data.to_parquet(val_file)
    print(f"  Saved {len(train_data)} train samples to {train_file}")
    print(f"  Saved {len(val_data)} val samples to {val_file}")
    results["train"] = len(train_data)
    results["val"] = len(val_data)

    sample = train_data[0]
    content = sample["prompt"][0]["content"]
    print(f"  Sample prompt: {content[:200]}...")
    gt_str = sample["reward_model"]["ground_truth"]
    try:
        gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
        n_tests = len(gt.get("inputs", [])) if isinstance(gt, dict) else 0
    except (json.JSONDecodeError, TypeError):
        n_tests = 0
    print(f"  title: {sample['extra_info']['title']}, test cases: {n_tests}")

    return results


# =============================================================================
# Map transform function names to callables
# =============================================================================

TRANSFORM_FNS = {
    "transform_eurus": transform_eurus,
    "transform_apps": transform_apps,
    "transform_taco": transform_taco,
    "transform_codecontests": transform_codecontests,
}


# =============================================================================
# Dataset Preparation (single dataset)
# =============================================================================


def prepare_dataset(name, output_dir):
    """Download, transform, and save a single dataset."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[name]
    hf_name = config["hf_name"]
    trust_remote_code = config.get("trust_remote_code", False)
    transform_fn = TRANSFORM_FNS[config["transform_fn"]]
    subset = config.get("subset", None)

    dataset_output_dir = os.path.join(output_dir, name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Preparing: {name} ({hf_name})")
    print(f"Output:    {dataset_output_dir}")
    print(f"{'=' * 60}")

    # Download from HuggingFace
    t0 = time.time()
    load_kwargs = {}
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    try:
        if subset:
            dataset_dict = datasets.load_dataset(hf_name, subset, **load_kwargs)
        else:
            dataset_dict = datasets.load_dataset(hf_name, **load_kwargs)
    except Exception as e:  # noqa: BLE001 -- inspected below and re-raised if unhandled
        if "scripts are no longer supported" in str(e):
            # Dataset uses a legacy loading script; try loading as json/parquet directly
            print("  Legacy script detected, trying direct file loading...")
            from huggingface_hub import HfApi

            api = HfApi()
            files = list(api.list_repo_files(hf_name, repo_type="dataset"))
            jsonl_files = [f for f in files if f.endswith((".jsonl", ".json"))]
            parquet_files = [f for f in files if f.endswith(".parquet")]

            if jsonl_files:
                data_files = {}
                for f in jsonl_files:
                    split_name = os.path.splitext(os.path.basename(f))[0]
                    # Normalize split names like "test-00000-of-00001" -> "test"
                    import re

                    m = re.match(r"^(\w+?)(?:-\d+-of-\d+)?$", split_name)
                    if m:
                        split_name = m.group(1)
                    data_files.setdefault(split_name, []).append(
                        f"hf://datasets/{hf_name}/{f}"
                    )
                dataset_dict = datasets.load_dataset("json", data_files=data_files)
            elif parquet_files:
                data_files = {}
                for f in parquet_files:
                    fname = os.path.splitext(os.path.basename(f))[0]
                    import re

                    m = re.match(r"^(\w+?)(?:-\d+-of-\d+)?$", fname)
                    split_name = m.group(1) if m else fname
                    data_files.setdefault(split_name, []).append(
                        f"hf://datasets/{hf_name}/{f}"
                    )
                dataset_dict = datasets.load_dataset("parquet", data_files=data_files)
            else:
                raise RuntimeError(f"No loadable files found for {hf_name}") from e
        else:
            print(f"  Error loading dataset: {e}")
            print("  Trying with streaming...")
            if subset:
                dataset_dict = datasets.load_dataset(
                    hf_name, subset, streaming=True, **load_kwargs
                )
            else:
                dataset_dict = datasets.load_dataset(
                    hf_name, streaming=True, **load_kwargs
                )
            dataset_dict = {
                split: datasets.Dataset.from_list(list(ds))
                for split, ds in dataset_dict.items()
            }
    download_time = time.time() - t0
    print(f"  Downloaded in {download_time:.1f}s")

    # Transform
    t0 = time.time()
    results = transform_fn(dataset_dict, dataset_output_dir, hf_name)
    transform_time = time.time() - t0
    print(f"  Transformed in {transform_time:.1f}s")

    return {"name": name, "output_dir": dataset_output_dir, "splits": results}


# =============================================================================
# Dataset Mixing
# =============================================================================


def mix_datasets(
    input_dir, dataset_names, mix_ratios, mix_output, seed=42, val_ratio=0.1
):
    """Mix multiple prepared datasets into a single train/val set."""
    rng = np.random.default_rng(seed)
    os.makedirs(mix_output, exist_ok=True)

    # Parse ratios
    ratios = {}
    for spec in mix_ratios:
        if ":" in spec:
            name, ratio = spec.rsplit(":", 1)
            ratios[name] = float(ratio)
        else:
            ratios[spec] = 1.0

    print(f"\n{'=' * 60}")
    print("Mixing datasets")
    print(f"{'=' * 60}")

    # Load and sample train sets
    train_parts = []
    val_parts = []
    for name in dataset_names:
        ratio = ratios.get(name, 1.0)
        train_path = os.path.join(input_dir, name, "train.parquet")
        val_path = os.path.join(input_dir, name, "val.parquet")

        if not os.path.exists(train_path):
            print(f"  WARNING: {train_path} not found, skipping {name}")
            continue

        print(f"  Loading {name} (ratio={ratio})...")
        ds = datasets.load_dataset("parquet", data_files=train_path, split="train")
        total = len(ds)

        if ratio < 1.0:
            n_samples = max(1, int(total * ratio))
            indices = rng.choice(total, size=n_samples, replace=False)
            indices.sort()
            ds = ds.select(indices.tolist())
            print(f"    Sampled {len(ds)}/{total} train rows")
        else:
            print(f"    Using all {total} train rows")
        train_parts.append(ds)

        # Load val if it exists
        if os.path.exists(val_path):
            val_ds = datasets.load_dataset(
                "parquet", data_files=val_path, split="train"
            )
            if ratio < 1.0:
                n_val = max(1, int(len(val_ds) * ratio))
                val_indices = rng.choice(len(val_ds), size=n_val, replace=False)
                val_indices.sort()
                val_ds = val_ds.select(val_indices.tolist())
            val_parts.append(val_ds)

    if not train_parts:
        print("  ERROR: No datasets loaded for mixing")
        return

    # Concatenate and shuffle
    combined_train = datasets.concatenate_datasets(train_parts)
    combined_train = combined_train.shuffle(seed=seed)

    if val_parts:
        combined_val = datasets.concatenate_datasets(val_parts)
        combined_val = combined_val.shuffle(seed=seed)
    elif val_ratio > 0:
        val_size = max(1, int(len(combined_train) * val_ratio))
        train_size = len(combined_train) - val_size
        combined_val = combined_train.select(range(train_size, len(combined_train)))
        combined_train = combined_train.select(range(train_size))
    else:
        combined_val = None

    # Save
    train_file = os.path.join(mix_output, "train.parquet")
    combined_train.to_parquet(train_file)
    print(f"\n  Saved {len(combined_train)} mixed train samples to {train_file}")

    if combined_val is not None and len(combined_val) > 0:
        val_file = os.path.join(mix_output, "val.parquet")
        combined_val.to_parquet(val_file)
        print(f"  Saved {len(combined_val)} mixed val samples to {val_file}")

    # Summary
    print("\n  Data source distribution (train):")
    sources = {}
    for i in range(len(combined_train)):
        src = combined_train[i].get("data_source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(combined_train)
        print(f"    {src}: {count} ({pct:.1f}%)")


# =============================================================================
# Ray Remote Task
# =============================================================================


def prepare_dataset_remote(name, output_dir):
    """Ray remote task wrapper for prepare_dataset."""
    import ray

    @ray.remote
    def _prepare(name, output_dir):
        # Lift Python 3.12 integer string conversion limit (APPS has huge ints)
        if hasattr(sys, "set_int_max_str_digits"):
            sys.set_int_max_str_digits(0)
        # Set HF cache to shared FSx storage
        os.environ.setdefault("HF_HOME", "/fsx/data/verl/cache/huggingface")
        return prepare_dataset(name, output_dir)

    return _prepare.remote(name, output_dir)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for GRPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Datasets to prepare",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory (each dataset gets a subdirectory)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally without Ray (sequential, for testing)",
    )
    parser.add_argument(
        "--mix",
        action="store_true",
        help="Mix datasets after preparation",
    )
    parser.add_argument(
        "--mix-output",
        default=None,
        help="Output directory for mixed dataset (default: {output-dir}/mixed)",
    )
    parser.add_argument(
        "--mix-ratios",
        nargs="*",
        default=None,
        help="Mixing ratios as name:ratio pairs (e.g., eurus:1.0 apps:0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Set HF cache to FSx if available
    if os.path.isdir("/fsx"):
        os.environ.setdefault("HF_HOME", "/fsx/data/verl/cache/huggingface")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("GRPO Data Preparation")
    print("=" * 60)
    print(f"Datasets:   {', '.join(args.datasets)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode:       {'local' if args.local else 'Ray parallel'}")
    print("=" * 60)

    t_start = time.time()

    if args.local:
        # Sequential local execution
        all_results = []
        for name in args.datasets:
            result = prepare_dataset(name, args.output_dir)
            all_results.append(result)
    else:
        # Parallel execution using Ray
        import ray

        if not ray.is_initialized():
            ray.init()

        futures = []
        for name in args.datasets:
            future = prepare_dataset_remote(name, args.output_dir)
            futures.append(future)

        print(f"\nSubmitted {len(futures)} Ray tasks, waiting for completion...")
        all_results = ray.get(futures)

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'=' * 60}")
    print("Preparation Complete!")
    print(f"{'=' * 60}")
    for result in all_results:
        print(f"  {result['name']}: {result['splits']}")
    print(f"  Total time: {total_time:.1f}s")

    # Mix if requested
    if args.mix:
        mix_output = args.mix_output or os.path.join(args.output_dir, "mixed")
        mix_ratios = args.mix_ratios or [f"{name}:1.0" for name in args.datasets]
        mix_datasets(
            args.output_dir, args.datasets, mix_ratios, mix_output, seed=args.seed
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
