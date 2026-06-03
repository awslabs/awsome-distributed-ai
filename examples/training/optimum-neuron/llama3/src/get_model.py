import argparse
import os

from transformers import AutoTokenizer, LlamaForCausalLM


def download_model_and_tokenizer(
    model_id: str,
    model_output_path: str,
    tokenizer_output_path: str,
    huggingface_token: str = None,
) -> None:
    """
    Download model and associated tokenizer from the Hugging Face Hub.
    Args:
        model_id (str): The ID of the model to download.
        model_output_path (str): The local path where the model should be saved.
        tokenizer_output_path (str): The local path where the tokenizer should be saved.
        huggingface_token (str, optional): The Hugging Face authentication token. If not provided,
            the function will attempt to use the token from the environment variable `HF_TOKEN`.
    Returns:
        None
    """
    if not huggingface_token:
        huggingface_token = os.environ.get("HF_TOKEN", None)
        if huggingface_token is None:
            print("Huggingface access token is missing!")
        else:
            # 1 Download and save model
            print(f"Downloading model {model_id}...")
            model = LlamaForCausalLM.from_pretrained(model_id, token=huggingface_token)
            print(f"Saving model to {model_output_path}...")
            model.save_pretrained(model_output_path)
            print("Model saved!")

            # 2 Download and save tokenizer
            print(f"Downloading tokenizer for {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
            print(f"Saving tokenizer to {tokenizer_output_path}...")
            tokenizer.save_pretrained(tokenizer_output_path)
            print("Tokenizer saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, required=True, help="Hugging Face Model id"
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        required=True,
        help="Path to save model/weights file",
    )
    parser.add_argument(
        "--tokenizer_output_path",
        type=str,
        required=True,
        help="Path to save tokenizer file",
    )

    args, _ = parser.parse_known_args()

    download_model_and_tokenizer(
        model_id=args.model_id,
        model_output_path=args.model_output_path,
        tokenizer_output_path=args.tokenizer_output_path,
    )
