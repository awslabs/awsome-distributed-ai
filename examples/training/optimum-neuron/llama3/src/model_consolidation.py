import shutil
import argparse
from pathlib import Path

from optimum.neuron.models.training import (
    consolidate_model_parallel_checkpoints_to_unified_checkpoint,
)


def copy_additional_files(input_dir: Path, output_dir: Path):
    """
    Copies model configuration and tokenizer files.

    Args:
        input_dir (Path): Source directory containing the files
        output_dir (Path): Destination directory for the files
    """
    files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "adapter_config.json",
    ]

    for file in files_to_copy:
        src = input_dir / file
        dst = output_dir / file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {file} to {dst}")
        else:
            print(f"Note: {file} not found in {input_dir}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate model checkpoints")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory containing model parallel shards",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the consolidated checkpoint",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["safetensors", "pytorch"],
        default="safetensors",
        help="Format to save the consolidated checkpoint",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Consolidating checkpoints from {input_dir} to {output_dir}")

    try:
        consolidate_model_parallel_checkpoints_to_unified_checkpoint(
            checkpoint_dir=input_dir,
            output_dir=output_dir,
            save_format=args.save_format,
        )
        print("Checkpoint consolidation completed successfully.")
    except Exception as e:
        print(f"Error during checkpoint consolidation: {e}")
        return

    # Copy configuration files
    try:
        copy_additional_files(input_dir, output_dir)
    except Exception as e:
        print(f"Error copying additional files: {e}")


if __name__ == "__main__":
    main()
